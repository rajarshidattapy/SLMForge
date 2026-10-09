"""Logging setup shared by the API, the worker and the CLI."""

from __future__ import annotations

import contextlib
import logging
import sys

from rich.logging import RichHandler

_CONFIGURED = False


def force_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 so Unicode in logs never crashes a build.

    Windows consoles default to a legacy code page (e.g. cp1252), where writing
    characters such as an em dash or an arrow raises UnicodeEncodeError — in the
    middle of starting a server or printing a real error. Cheap to call, and it
    degrades quietly when the stream cannot be reconfigured (already piped/closed).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # A closed or exotic stream cannot be reconfigured; that is not fatal.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, with a Rich handler on interactive terminals."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    force_utf8_stdio()

    from .config import settings

    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    # Third-party loggers that are noisy at INFO.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced application logger."""
    if not _CONFIGURED and sys.stderr.isatty():
        setup_logging()
    return logging.getLogger(f"slmforge.{name}")
