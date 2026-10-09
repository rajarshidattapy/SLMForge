"""Integration helpers: licence policy, metadata parsing and offline behaviour."""

from __future__ import annotations

import pytest

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
