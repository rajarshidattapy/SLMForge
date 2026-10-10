"""Sandbox backends for isolated job execution."""

from __future__ import annotations

from .base import NetworkPolicy, ResourceLimits, Sandbox, SandboxResult
from .docker_backend import DockerSandbox, docker_available
from .subprocess_backend import SubprocessSandbox


def get_sandbox(backend: str | None = None) -> Sandbox:
    """Return the configured sandbox backend.

    Backends refuse to silently downgrade: asking for `docker` when the Docker
    daemon is stopped raises with instructions, because quietly running untrusted
    code in the weaker subprocess mode is exactly the failure mode to avoid.
    """
    from ..config import settings

    backend = backend or settings.sandbox_backend
    if backend == "subprocess":
        return SubprocessSandbox()
    if backend == "docker":
        if not docker_available():
            raise RuntimeError(
                "SLMFORGE_SANDBOX_BACKEND=docker, but the Docker daemon is not reachable. "
                "Start Docker Desktop (the CLI may be installed while the daemon is stopped), "
                "or set SLMFORGE_SANDBOX_BACKEND=subprocess for a weaker dev-only mode. "
                "Remember: the subprocess backend is process isolation, not a security boundary."
            )
        return DockerSandbox()
    if backend == "hf_jobs":
        raise NotImplementedError(
            "The hf_jobs sandbox backend is not implemented yet (see docs/ROADMAP.md Stage 6). "
            "It runs work in managed, ephemeral Hugging Face containers and requires "
            "Hugging Face credits; until then use SLMFORGE_SANDBOX_BACKEND=docker."
        )
    raise ValueError(f"Unknown sandbox backend: {backend!r}")


__all__ = [
    "DockerSandbox",
    "NetworkPolicy",
    "ResourceLimits",
    "Sandbox",
    "SandboxResult",
    "SubprocessSandbox",
    "docker_available",
    "get_sandbox",
]
