"""Retry helpers for network calls.

SLMForge talks to the Hugging Face Hub, OpenRouter and Ollama over networks that
drop connections. On the reference machine `huggingface.co` intermittently resets
TLS handshakes (`WinError 10054`, `SSL connect error`) — sometimes succeeding on
the next attempt, sometimes failing several in a row. Treating that as a fatal
error makes the whole pipeline look broken.

So every outbound call goes through `retry_call` / `retry_call_async`:

  * transient transport failures are retried with exponential backoff and jitter;
  * a 4xx that means "you asked wrong" is *not* retried, because retrying cannot help;
  * HTTP 429 / 503 *are* retried, because they mean "later";
  * when all attempts fail we raise `TransientNetworkError` with the attempt count
    so the UI can say "the network is unstable" instead of printing a stack trace.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .log import get_logger

logger = get_logger("net")

T = TypeVar("T")

DEFAULT_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 20.0

#: Status codes worth retrying. 429 = rate limited, 5xx = server side.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class TransientNetworkError(RuntimeError):
    """All retry attempts failed for a reason that may pass later."""

    def __init__(self, message: str, *, attempts: int, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.cause = cause


def is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception is worth retrying.

    Import-agnostic on purpose: callers may raise httpx, duckdb, subprocess or
    plain OSError types, and we do not want this module to depend on all of them.
    """
    # Rate limiting / server errors, when an HTTP client exposes a status code.
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int):
        return status in RETRYABLE_STATUS

    name = type(exc).__name__
    if name in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "TransportError",
        "TimeoutException",
        "ConnectionResetError",
        "ConnectionError",
        "IOException",  # duckdb's remote-read failure type
        "HTTPError",
    }:
        return True

    # DuckDB wraps TLS problems in an IOException with a descriptive message.
    text = str(exc).lower()
    return any(
        hint in text
        for hint in (
            "ssl connect error",
            "connection reset",
            "connection aborted",
            "forcibly closed",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "429",
            "503",
        )
    )


def backoff_delay(attempt: int, base: float, maximum: float) -> float:
    """Exponential backoff with full jitter, capped at `maximum` seconds."""
    return min(maximum, base * (2 ** (attempt - 1))) * random.uniform(0.5, 1.0)


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retry_on: Callable[[BaseException], bool] = is_retryable,
    description: str = "network call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn` with retries, raising `TransientNetworkError` when exhausted."""
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            if not retry_on(exc):
                raise
            last = exc
            if attempt == attempts:
                break
            delay = backoff_delay(attempt, base_delay, max_delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            sleep(delay)
    raise TransientNetworkError(
        f"{description} failed after {attempts} attempts: {last}",
        attempts=attempts,
        cause=last,
    )


async def retry_call_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retry_on: Callable[[BaseException], bool] = is_retryable,
    description: str = "network call",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Async variant of `retry_call`."""
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except BaseException as exc:
            if not retry_on(exc):
                raise
            last = exc
            if attempt == attempts:
                break
            delay = backoff_delay(attempt, base_delay, max_delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            await sleep(delay)
    raise TransientNetworkError(
        f"{description} failed after {attempts} attempts: {last}",
        attempts=attempts,
        cause=last,
    )
