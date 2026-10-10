"""Docker sandbox backend — the real security boundary for untrusted work.

Unlike `SubprocessSandbox` (process isolation only), a container gives us an
actual boundary: a read-only root filesystem, dropped capabilities, a non-root
user, no network unless explicitly granted, and hard CPU/memory/pid limits.

Every job runs with the same posture:

  --read-only            the image itself cannot be modified
  --tmpfs /tmp           the only writable scratch space
  --cap-drop ALL         no Linux capabilities beyond the baseline
  --security-opt no-new-privileges
  --user 65534:65534     never root, even if the image's default user is root
  --pids-limit           a fork bomb cannot take the host down
  --memory / --cpus      hard resource ceilings    --network none|bridge  network off by default; granted per job

The job's work directory is bind-mounted at ``/work``, which is the only path a
job can write that survives the container. Callers must therefore address files
by their ``/work/...`` path — see :meth:`DockerSandbox.container_path`.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path

from ..config import settings
from ..log import get_logger
from .base import ResourceLimits, Sandbox, SandboxResult, program_output

logger = get_logger("sandbox.docker")

CONTAINER_WORKDIR = "/work"
NOBODY_UID = "65534:65534"


def docker_available() -> bool:
    """True when the Docker CLI exists *and* its daemon answers.

    The Docker Desktop daemon is often stopped even though the CLI is installed,
    so a bare `shutil.which` would report availability falsely.
    """
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


class DockerSandbox(Sandbox):
    """Runs a command inside an isolated container."""

    backend = "docker"

    def __init__(self, image: str | None = None) -> None:
        self.image = image or settings.sandbox_image

    # -- paths ----------------------------------------------------------
    @staticmethod
    def container_path(host_path: Path | str, work_dir: Path) -> str:
        """Map a file inside the work dir to the path the command must use.

        ``/tmp/job/train.py`` on the host is ``/work/train.py`` inside the
        container. Getting this wrong is the classic Docker-sandbox bug — the
        script exists on the host but the container cannot see it.
        """
        relative = Path(host_path).resolve().relative_to(Path(work_dir).resolve())
        return f"{CONTAINER_WORKDIR}/{relative.as_posix()}"

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
        if not docker_available():
            result = SandboxResult(
                exit_code=-1,
                duration_s=0.0,
                error=(
                    "Docker is not available. Start Docker Desktop, or set "
                    "SLMFORGE_SANDBOX_BACKEND=subprocess for a weaker dev-only mode."
                ),
            )
            return result

        limits = limits or ResourceLimits.from_settings()
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cidfile = work_dir / f".slmforge-cid-{uuid.uuid4().hex[:8]}"
        name = f"slmforge-{job_id or uuid.uuid4().hex[:12]}"

        argv = self._build_argv(command, work_dir, limits, env, name, cidfile)
        log_path = Path(log_path) if log_path else work_dir / "job.log"

        started = time.perf_counter()
        timed_out = killed = False
        exit_code = -1
        error: str | None = None

        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"$ {' '.join(command)}\n")
                log_file.write(
                    f"# image={self.image} network={limits.network} "
                    f"cpus={limits.cpu_cores} mem={limits.memory_mb}MB\n\n"
                )
                log_file.flush()
                process = subprocess.Popen(
                    argv,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                try:
                    exit_code = process.wait(timeout=limits.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    killed = self._force_remove(name, cidfile)
                    exit_code = process.returncode if process.returncode is not None else -1
        except FileNotFoundError as exc:
            error = f"Docker CLI not found: {exc}"
        except subprocess.TimeoutExpired as exc:
            # The whole `docker run` call hung (daemon unresponsive).
            timed_out = True
            killed = self._force_remove(name, cidfile)
            error = f"docker run timed out: {exc}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Docker sandbox run failed")
        finally:
            # Always reclaim, whether the job succeeded, failed or timed out.
            self._force_remove(name, cidfile)
            cidfile.unlink(missing_ok=True)

        duration = round(time.perf_counter() - started, 3)
        if error and exit_code == 0:
            exit_code = -1

        captured = ""
        if log_path.exists():
            captured = program_output(log_path.read_text(encoding="utf-8", errors="replace"))

        return SandboxResult(
            exit_code=exit_code,
            duration_s=duration,
            logs_path=log_path,
            stdout=captured,
            timed_out=timed_out,
            killed=killed,
            output_files=sorted(p for p in work_dir.rglob("*") if p.is_file()),
            error=error,
        )

    # -- internals ------------------------------------------------------
    def _build_argv(
        self,
        command: list[str],
        work_dir: Path,
        limits: ResourceLimits,
        env: dict[str, str] | None,
        name: str,
        cidfile: Path,
    ) -> list[str]:
        argv = [
            "docker",
            "run",
            "--name",
            name,
            "--cidfile",
            str(cidfile),
            # Isolation posture ------------------------------------------------
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=512m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            NOBODY_UID,
            "--pids-limit",
            "256",
            "--workdir",
            CONTAINER_WORKDIR,
        ]

        if limits.cpu_cores:
            argv += ["--cpus", str(limits.cpu_cores)]
        if limits.memory_mb:
            argv += ["--memory", f"{limits.memory_mb}m"]
        # Do not let a container's OOM kill silently look like a script error.
        if limits.memory_mb:
            argv += ["--memory-swap", f"{limits.memory_mb}m"]

        argv += ["--network", self._network(limits)]

        # The single writable, host-visible mount.
        argv += ["--volume", f"{self._host_path(work_dir)}:{CONTAINER_WORKDIR}"]

        # Scratch space inside the work dir (nobody-writable), not the read-only root.
        argv += ["--env", f"TMPDIR={CONTAINER_WORKDIR}/tmp"]

        for key, value in (env or {}).items():
            argv += ["--env", f"{key}={value}"]

        if limits.gpu:
            argv += ["--gpus", "all"]

        argv.append(self.image)
        argv += command
        return argv

    @staticmethod
    def _host_path(path: Path) -> str:
        """Docker needs a drive-letter path on Windows, with forward slashes."""
        return str(path).replace("\\", "/")

    @staticmethod
    def _network(limits: ResourceLimits) -> str:
        if limits.network == "deny":
            return "none"
        if limits.network == "allow":
            return "bridge"
        # "allowlist" has no container-native equivalent without extra plumbing;
        # fail rather than silently granting the whole network.
        raise NotImplementedError(
            "Network policy 'allowlist' is not supported by the Docker backend yet; "
            "use 'deny' or 'allow'."
        )

    def _force_remove(self, name: str, cidfile: Path) -> bool:
        """Hard-remove a container so nothing survives a timeout or crash."""
        cid = cidfile.read_text(encoding="utf-8").strip() if cidfile.exists() else ""
        targets = [t for t in (cid, name) if t]
        removed = False
        for target in targets:
            try:
                completed = subprocess.run(
                    ["docker", "rm", "-f", target],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                    check=False,
                )
                if completed.returncode == 0:
                    removed = True
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning("Could not remove container %s: %s", target, exc)
        return removed
