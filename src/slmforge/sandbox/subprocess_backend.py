"""Subprocess sandbox backend.

NOTE — this is *process isolation*, not a security boundary. It gives each job its
own working directory, a scrubbed environment, a timeout and best-effort CPU
limits (via a preexec hook on POSIX). For untrusted third-party code the Docker
backend must be used. This is stated plainly so nobody mistakes the prototype's
isolation for sandboxing-grade containment.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from ..log import get_logger
from .base import ResourceLimits, Sandbox, SandboxResult

logger = get_logger("sandbox.subprocess")

#: `SIGKILL` does not exist on Windows; SIGTERM maps to TerminateProcess there.
SIGKILL: int = getattr(signal, "SIGKILL", signal.SIGTERM)


def _apply_posix_rlimits(cpu_seconds: int, mem_bytes: int) -> None:
    """Apply CPU and address-space rlimits in the forked child (POSIX only).

    The `resource` members are looked up dynamically because they are POSIX-only:
    this keeps the module importable and type-checkable on every platform.
    """
    import resource

    setrlimit = getattr(resource, "setrlimit", None)
    if setrlimit is None:  # pragma: no cover - non-POSIX
        return

    rlimit_cpu = getattr(resource, "RLIMIT_CPU", None)
    if cpu_seconds and rlimit_cpu is not None:
        setrlimit(rlimit_cpu, (cpu_seconds, cpu_seconds + 30))

    rlimit_as = getattr(resource, "RLIMIT_AS", None)
    if mem_bytes and rlimit_as is not None:
        setrlimit(rlimit_as, (mem_bytes, mem_bytes))


class SubprocessSandbox(Sandbox):
    """Runs a command as a child process with a timeout and a cleaned environment."""

    backend = "subprocess"

    def run(
        self,
        command: list[str],
        *,
        work_dir: Path,
        limits: ResourceLimits | None = None,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        job_id: str | None = None,
    ) -> SandboxResult:
        limits = limits or ResourceLimits.from_settings()
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        clean_env = self._build_env(limits, env)
        log_path = Path(log_path) if log_path else work_dir / "job.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        timed_out = False
        killed = False
        exit_code = -1
        error: str | None = None

        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"$ {' '.join(command)}\n")
                log_file.write(f"# cwd={work_dir} timeout={limits.timeout_s}s\n\n")
                log_file.flush()

                process = subprocess.Popen(
                    command,
                    cwd=str(work_dir),
                    env=clean_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    preexec_fn=self._make_preexec(limits),
                )
                try:
                    exit_code = process.wait(timeout=limits.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    killed = self._terminate(process)
                    exit_code = process.returncode if process.returncode is not None else -1
        except FileNotFoundError as exc:
            error = f"Command not found: {command[0]!r} ({exc})"
            logger.error(error)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Sandbox run failed")

        duration = round(time.perf_counter() - started, 3)
        if error and exit_code == 0:
            exit_code = -1

        return SandboxResult(
            exit_code=exit_code,
            duration_s=duration,
            logs_path=log_path,
            timed_out=timed_out,
            killed=killed,
            output_files=sorted(p for p in work_dir.rglob("*") if p.is_file()),
            error=error,
        )

    # -- helpers --------------------------------------------------------
    def _build_env(self, limits: ResourceLimits, extra: dict[str, str] | None) -> dict[str, str]:
        """Pass through only allow-listed variables, then apply job-specific ones.

        Secrets (HF_TOKEN, OLLAMA keys, ...) are NOT forwarded by default; jobs
        that need them must ask the orchestrator for a scoped token.
        """
        env: dict[str, str] = {}
        for key in limits.env_allowlist:
            if key in os.environ:
                env[key] = os.environ[key]
        env.setdefault("PATH", os.environ.get("PATH", ""))
        # Keep model caches inside the workspace rather than a global home dir.
        from ..config import settings

        env["HF_HOME"] = str(Path(settings.hf_home).resolve())
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        if extra:
            env.update(extra)
        return env

    def _make_preexec(self, limits: ResourceLimits):
        """Build a preexec hook that applies rlimits, or None where unsupported.

        `resource` and the RLIMIT_* constants are POSIX-only, so this returns
        None on Windows (where the wall-clock timeout in `run` is the only bound).
        """
        if os.name != "posix":  # pragma: no cover - Windows path
            return None

        cpu_seconds = limits.timeout_s or (limits.cpu_cores or 1) * 60
        mem_bytes = (limits.memory_mb or 0) * 1024 * 1024

        def _preexec() -> None:  # pragma: no cover - runs in the forked child
            _apply_posix_rlimits(cpu_seconds, mem_bytes)

        return _preexec

    def _terminate(self, process: subprocess.Popen) -> bool:
        """Ask the process group to stop, then force-kill if needed.

        Returns True when the process had to be SIGKILLed.
        """
        logger.warning("Sandbox timeout: terminating pid=%s", process.pid)
        if self._signal_group(process, signal.SIGTERM):
            try:
                process.wait(timeout=10)
                return False
            except subprocess.TimeoutExpired:
                pass
        self._signal_group(process, SIGKILL)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - unkillable process
            logger.error("Sandbox process %s survived SIGKILL", process.pid)
        return True

    @staticmethod
    def _signal_group(process: subprocess.Popen, sig: int) -> bool:
        """Signal the whole process group on POSIX, or just the child elsewhere."""
        if process.poll() is not None:
            return False
        try:
            if os.name == "posix":
                # POSIX-only attributes; typeshed omits them on Windows builds.
                os.killpg(os.getpgid(process.pid), sig)  # type: ignore[attr-defined]
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False


def which(name: str) -> str | None:
    """Resolve an executable, returning None when it is unavailable."""
    return shutil.which(name)
