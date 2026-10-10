"""Retry behaviour.

The reference network resets TLS handshakes intermittently, so these tests pin
the distinction that matters: retry things that may pass later, fail fast on
things that cannot, and never retry forever without saying so.
"""

from __future__ import annotations

import pytest

from slmforge.net import (
    TransientNetworkError,
    backoff_delay,
    is_retryable,
    retry_call,
    retry_call_async,
)


class FakeHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_returns_immediately_on_success() -> None:
    calls = []
    result = retry_call(lambda: calls.append(1) or "ok", sleep=lambda _s: None)
    assert result == "ok"
    assert len(calls) == 1


def test_retries_transient_then_succeeds() -> None:
    attempts = {"n": 0}
    slept: list[float] = []

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionResetError("connection reset by peer")
        return "recovered"

    result = retry_call(flaky, attempts=5, sleep=slept.append)
    assert result == "recovered"
    assert attempts["n"] == 3
    assert len(slept) == 2  # slept between attempts only


def test_exhausting_attempts_raises_transient_network_error() -> None:
    def always_fails() -> None:
        raise ConnectionResetError("[WinError 10054] forcibly closed")

    with pytest.raises(TransientNetworkError) as excinfo:
        retry_call(always_fails, attempts=3, sleep=lambda _s: None)
    assert excinfo.value.attempts == 3
    assert excinfo.value.cause is not None


def test_permanent_errors_are_not_retried() -> None:
    """A 404 will never become a 200; retrying it just wastes the user's time."""
    calls = {"n": 0}

    def not_found() -> None:
        calls["n"] += 1
        raise FakeHTTPError(404)

    with pytest.raises(FakeHTTPError):
        retry_call(not_found, attempts=5, sleep=lambda _s: None)
    assert calls["n"] == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_rate_limits_and_server_errors_are_retried(status: int) -> None:
    calls = {"n": 0}

    def limited() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise FakeHTTPError(status)
        return "ok"

    assert retry_call(limited, attempts=3, sleep=lambda _s: None) == "ok"
    assert calls["n"] == 2


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("SSL connect error error for HTTP HEAD", True),
        ("IO Error: SSL connect error", True),
        ("connection reset", True),
        ("An existing connection was forcibly closed by the remote host", True),
        ("request timed out", True),
        ("429 Too Many Requests", True),
        ("field `queue` does not exist", False),
        ("invalid API key", False),
    ],
)
def test_hf_and_duckdb_style_messages_are_classified(message: str, expected: bool) -> None:
    assert is_retryable(RuntimeError(message)) is expected


def test_backoff_grows_and_respects_the_cap() -> None:
    for attempt in range(1, 12):
        delay = backoff_delay(attempt, base=1.0, maximum=20.0)
        assert 0 < delay <= 20.0
    # Nothing above the cap, ever.
    assert backoff_delay(50, base=1.0, maximum=5.0) <= 5.0


def test_no_retry_delay_is_waited_before_the_final_attempt() -> None:
    slept: list[float] = []

    def fail() -> None:
        raise ConnectionResetError("reset")

    with pytest.raises(TransientNetworkError):
        retry_call(fail, attempts=4, sleep=slept.append)
    assert len(slept) == 3


@pytest.mark.asyncio
async def test_async_variant_retries_and_recovers() -> None:
    state = {"n": 0}
    slept: list[float] = []

    async def flaky() -> str:
        state["n"] += 1
        if state["n"] < 2:
            raise ConnectionResetError("reset")
        return "ok"

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    assert await retry_call_async(flaky, attempts=4, sleep=fake_sleep) == "ok"
    assert len(slept) == 1


@pytest.mark.asyncio
async def test_async_variant_gives_up_with_a_clear_error() -> None:
    async def fail() -> None:
        raise ConnectionResetError("reset")

    async def fake_sleep(_seconds: float) -> None:
        return None

    with pytest.raises(TransientNetworkError) as excinfo:
        await retry_call_async(fail, attempts=2, sleep=fake_sleep)
    assert excinfo.value.attempts == 2
