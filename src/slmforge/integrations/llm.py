"""Planner LLM client.

The analyzer and planner need a capable model to reason about the user's
requirement. That model is *separate* from the model SLMForge trains.

Any OpenAI-compatible chat-completions endpoint works (OpenAI, Azure, vLLM,
llama.cpp server, Ollama's `/v1`). When no LLM is configured the caller gets a
`PlannerLLMUnavailable` error and the stage must degrade to a deterministic
heuristic rather than pretending it reasoned.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..log import get_logger

logger = get_logger("integrations.llm")


class PlannerLLMUnavailable(RuntimeError):
    """Raised when no planner LLM is configured or reachable."""


class ChatMessage(BaseModel):
    role: str
    content: str


class PlannerLLM:
    """OpenAI-compatible chat client with JSON-mode helpers."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.provider = settings.llm_provider
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def __aenter__(self) -> PlannerLLM:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._client.aclose()

    @property
    def enabled(self) -> bool:
        return self.provider != "none" and bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion and return the assistant message text."""
        if not self.enabled:
            raise PlannerLLMUnavailable(
                "No planner LLM configured. Set SLMFORGE_LLM_PROVIDER and SLMFORGE_LLM_* or start Ollama."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() if isinstance(m, ChatMessage) else m for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": False},
            }

        try:
            response = await self._client.post(
                "/chat/completions", json=payload, headers=self._headers()
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise PlannerLLMUnavailable(f"Planner LLM request failed: {exc}") from exc

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise PlannerLLMUnavailable(f"Unexpected LLM response shape: {data!r}") from exc

    async def chat_json_dict(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chat and parse the reply as a JSON object, repairing trivial wrapping."""
        raw = await self.chat(messages, json_schema=json_schema)
        return parse_json_object(raw)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating code fences and prose."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


def validate_into(model: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    """Validate a dict into a pydantic model, logging a readable error on failure."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        logger.error("Planner output failed validation for %s: %s", model.__name__, exc)
        raise
