"""Planner LLM client.

The analyzer and planner need a capable model to reason about the user's
requirement. That model is *separate* from the model SLMForge trains.

Any OpenAI-compatible chat-completions endpoint works (OpenAI, Azure, vLLM,
llama.cpp server, Ollama's `/v1`). When no LLM is configured the caller gets a
`PlannerLLMUnavailable` error and the stage must degrade to a deterministic
heuristic rather than pretending it reasoned.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..log import get_logger
from ..net import backoff_delay

logger = get_logger("integrations.llm")


class PlannerLLMUnavailable(RuntimeError):
    """Raised when no planner LLM is configured or reachable."""


#: Phrases in a provider error that mean "try again", not "your request is wrong".
RETRYABLE_MARKERS = (
    "overload",
    "rate limit",
    "temporarily",
    "timeout",
    "timed out",
    "unavailable",
    "429",
    "500",
    "502",
    "503",
    "504",
)
#: HTTP statuses worth another attempt.
RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}


def provider_error(data: Any) -> str | None:
    """The provider's error message when a *successful* HTTP response carries one.

    OpenRouter (and any OpenAI-compatible gateway in front of a busy upstream) can
    answer `200 OK` with `{"error": {...}}`. Treating that as a malformed response
    hides a transient provider problem behind a permanent-looking failure.
    """
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        error = data["error"]
        return str(error.get("message") or error.get("code") or "provider error")
    return None


def is_retryable_message(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in RETRYABLE_MARKERS)


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

        attempts = max(1, settings.network_attempts)
        last_error = "no attempt was made"

        for attempt in range(1, attempts + 1):
            try:
                response = await self._client.post(
                    "/chat/completions", json=payload, headers=self._headers()
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                last_error = f"{exc.response.status_code} {exc}"
                if exc.response.status_code not in RETRYABLE_STATUSES or attempt == attempts:
                    raise PlannerLLMUnavailable(
                        f"Planner LLM request failed: {last_error}"
                    ) from exc
            except httpx.HTTPError as exc:
                last_error = str(exc)
                if attempt == attempts:
                    raise PlannerLLMUnavailable(f"Planner LLM request failed: {exc}") from exc
            else:
                error = provider_error(data)
                if error is None:
                    try:
                        return data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError) as exc:
                        raise PlannerLLMUnavailable(
                            f"Unexpected LLM response shape: {data!r}"
                        ) from exc
                last_error = error
                if not is_retryable_message(error) or attempt == attempts:
                    raise PlannerLLMUnavailable(f"Planner LLM provider error: {error}")

            delay = backoff_delay(
                attempt, settings.network_base_delay_s, settings.network_max_delay_s
            )
            logger.warning(
                "Planner LLM unavailable (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                attempts,
                last_error,
                delay,
            )
            await asyncio.sleep(delay)

        raise PlannerLLMUnavailable(
            f"Planner LLM request failed after {attempts} attempts: {last_error}"
        )

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
