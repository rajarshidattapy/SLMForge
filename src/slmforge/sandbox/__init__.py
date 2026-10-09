"""Sandbox backends for isolated job execution."""

from __future__ import annotations

from .base import NetworkPolicy, ResourceLimits, Sandbox, SandboxResult
from .subprocess_backend import SubprocessSandbox


def get_sandbox(backend: str | None = None) -> Sandbox:
    """Return the configured sandbox backend.

    `docker` is declared in settings but not implemented yet; fail loudly rather
    than silently downgrading isolation.
    """
    from ..config import settings

    backend = backend or settings.sandbox_backend
    if backend == "subprocess":
        return SubprocessSandbox()
    if backend == "docker":
        raise NotImplementedError(
            "Docker sandbox backend is not implemented yet (see docs/ROADMAP.md Stage 4)."
        )
    raise ValueError(f"Unknown sandbox backend: {backend!r}")


__all__ = [
    "NetworkPolicy",
    "ResourceLimits",
    "Sandbox",
    "SandboxResult",
    "SubprocessSandbox",
    "get_sandbox",
]
