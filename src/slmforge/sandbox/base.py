"""Sandbox interface for running untrusted jobs in isolation.

Everything that touches user-supplied datasets or generated training scripts
runs through a `Sandbox`. The prototype ships a subprocess backend with resource
limits; a Docker backend adds filesystem and network isolation later without
changing the call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

NetworkPolicy = Literal["deny", "allowlist", "allow"]


@dataclass
class ResourceLimits:
    """Hard limits applied to a sandboxed process."""

    cpu_cores: int | None = None
    memory_mb: int | None = None
    timeout_s: int | None = None
    disk_mb: int | None = None
    network: NetworkPolicy = "deny"
    gpu: bool = False
    env_allowlist: tuple[str, ...] = ("PATH", "SYSTEMROOT", "HOME", "TMPDIR", "TEMP")

    @classmethod
    def from_settings(cls, **overrides: Any) -> ResourceLimits:
        from ..config import settings

        limits = cls(
            cpu_cores=settings.sandbox_cpu_limit,
            memory_mb=settings.sandbox_mem_limit_mb,
            timeout_s=settings.sandbox_timeout_s,
            network=settings.sandbox_network,
        )
        for key, value in overrides.items():
            setattr(limits, key, value)
        return limits


@dataclass
class SandboxResult:
    """The outcome of a sandboxed command."""

    exit_code: int
    duration_s: float
    logs_path: Path | None = None
    #: Captured program output ONLY — the command echo and header are excluded.
    #: Asserting on the raw log would also match the echoed source text, so callers
    #: that need "what did the job say" use this rather than reading the file.
    stdout: str = ""
    timed_out: bool = False
    killed: bool = False
    peak_memory_mb: float | None = None
    output_files: list[Path] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.killed


#: Backends write a short header (`$ <command>` + context) followed by a blank
#: line. Everything after it is genuine program output.
LOG_SEPARATOR = "\n\n"


def program_output(log_text: str) -> str:
    """Strip the backend's header, returning only what the job itself printed."""
    if LOG_SEPARATOR not in log_text:
        return ""
    return log_text.split(LOG_SEPARATOR, 1)[1]


class Sandbox(ABC):
    """Runs a command or script in an isolated environment."""

    backend: str = "abstract"

    @abstractmethod
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
        """Execute `command` in `work_dir` under `limits` and return the result."""
        raise NotImplementedError

    def write_script(self, work_dir: Path, name: str, content: str) -> Path:
        """Materialize a generated script inside the work dir and return its path."""
        work_dir.mkdir(parents=True, exist_ok=True)
        path = work_dir / name
        path.write_text(content, encoding="utf-8")
        return path
