"""Integration helpers: licence policy, metadata parsing and offline behaviour."""

from __future__ import annotations

import httpx
import pytest

from slmforge.config import settings
from slmforge.integrations.huggingface import (
    is_commercial_friendly,
    parse_params_billions,
)
from slmforge.integrations.llm import PlannerLLM, PlannerLLMUnavailable, parse_json_object
from slmforge.integrations.ollama import OllamaClient

#: A port nothing should be listening on.
CLOSED_URL = "http://127.0.0.1:59381"


@pytest.mark.parametrize(
    ("license_id", "expected"),
    [
        ("apache-2.0", True),
        ("MIT", True),
        ("cc-by-4.0", True),
        ("gemma", False),
        ("cc-by-nc-4.0", False),
        ("unknown", False),
        (None, None),
        ("some-custom-license", None),
    ],
)
def test_license_classification(license_id: str | None, expected: bool | None) -> None:
    assert is_commercial_friendly(license_id) is expected


@pytest.mark.parametrize(
    ("repo_id", "expected"),
    [
        ("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
        ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
        ("meta-llama/Llama-3.2-1B", 1.0),
        ("microsoft/Phi-3-mini-4k-instruct", None),
        ("tinyllama", None),
    ],
)
def test_parameter_count_parsing(repo_id: str, expected: float | None) -> None:
    assert parse_params_billions(repo_id) == expected


def test_parse_json_object_handles_fences_and_prose() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_parse_json_object_rejects_non_objects() -> None:
    with pytest.raises(ValueError):
        parse_json_object("[1, 2, 3]")


@pytest.mark.asyncio
async def test_planner_llm_disabled_raises_a_clear_error() -> None:
    llm = PlannerLLM(base_url="", model="x", api_key="")
    llm.provider = "none"
    assert llm.enabled is False
    with pytest.raises(PlannerLLMUnavailable):
        await llm.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_ollama_degrades_gracefully_when_server_is_down() -> None:
    client = OllamaClient(base_url=CLOSED_URL)
    try:
        assert await client.is_available() is False
        assert await client.list_models() == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_planner_llm_unreachable_raises_unavailable() -> None:
    llm = PlannerLLM(base_url=CLOSED_URL, model="x", api_key="")
    try:
        with pytest.raises(PlannerLLMUnavailable):
            await llm.chat([{"role": "user", "content": "hi"}])
    finally:
        await llm._client.aclose()


# --- provider errors in a 200 response -------------------------------------


class _FakeResponse:
    """Enough of httpx.Response for the client's error handling."""

    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://fake/chat/completions"),
                response=self,  # type: ignore[arg-type]
            )

    def json(self) -> object:
        return self._payload


class _FakeClient:
    """Replays a scripted list of responses, counting the calls it received."""

    def __init__(self, script: list[object]) -> None:
        self._script = script
        self.calls = 0

    async def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        if isinstance(item, _FakeResponse):
            return item
        raise item  # type: ignore[misc]

    async def aclose(self) -> None:
        return None


def _scripted_llm(monkeypatch: pytest.MonkeyPatch, script: list[object]) -> PlannerLLM:
    """A PlannerLLM whose transport is scripted, with backoff delays removed."""
    monkeypatch.setattr(settings, "network_attempts", 3)
    monkeypatch.setattr(settings, "network_base_delay_s", 0.0)
    monkeypatch.setattr(settings, "network_max_delay_s", 0.0)
    llm = PlannerLLM(base_url="http://fake", model="m", api_key="k")
    llm.provider = "openrouter"
    llm._client = _FakeClient(script)  # type: ignore[assignment]
    return llm


def _ok(content: str = "{}") -> _FakeResponse:
    return _FakeResponse({"choices": [{"message": {"content": content}}]})


def _provider_error(message: str) -> _FakeResponse:
    """The shape OpenRouter returns with a 200 when its upstream is busy."""
    return _FakeResponse({"error": {"message": message, "code": 503}})


@pytest.mark.asyncio
async def test_a_provider_overload_in_a_200_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A busy free provider must not look like a permanent failure."""
    llm = _scripted_llm(
        monkeypatch,
        [
            _provider_error("Upstream error from Nvidia: Service temporarily overloaded"),
            _ok('{"a": 1}'),
        ],
    )
    assert await llm.chat([{"role": "user", "content": "hi"}]) == '{"a": 1}'
    assert llm._client.calls == 2  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_an_exhausted_provider_error_names_the_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _scripted_llm(
        monkeypatch,
        [_provider_error("provider_overloaded")],
    )
    with pytest.raises(PlannerLLMUnavailable, match="provider error"):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert llm._client.calls == 3  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_permanent_provider_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _scripted_llm(monkeypatch, [_provider_error("Invalid API key provided")])
    with pytest.raises(PlannerLLMUnavailable):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert llm._client.calls == 1  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_retryable_http_status_is_retried_and_a_401_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retried = _scripted_llm(monkeypatch, [_FakeResponse({}, status_code=503), _ok("ok")])
    assert await retried.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert retried._client.calls == 2  # type: ignore[union-attr]

    denied = _scripted_llm(monkeypatch, [_FakeResponse({}, status_code=401)])
    with pytest.raises(PlannerLLMUnavailable):
        await denied.chat([{"role": "user", "content": "hi"}])
    assert denied._client.calls == 1  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_genuinely_malformed_response_still_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _scripted_llm(monkeypatch, [_FakeResponse({"choices": []})])
    with pytest.raises(PlannerLLMUnavailable, match="response shape"):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert llm._client.calls == 1  # type: ignore[union-attr]
