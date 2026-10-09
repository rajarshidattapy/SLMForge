"""Ollama integration.

Used for two distinct things:
1. The local model inventory shown as base-model candidates.
2. A zero-cost, fully local planner LLM for the analyzer (via `llm.py`).
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from ..config import settings
from ..log import get_logger

logger = get_logger("integrations.ollama")


class OllamaModel(BaseModel):
    """One locally installed Ollama model."""

    name: str
    size_bytes: int | None = None
    family: str | None = None
    parameters: str | None = None
    quantization: str | None = None
    modified_at: str | None = None


class OllamaClient:
    """Minimal async client for the Ollama HTTP API."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def is_available(self) -> bool:
        """Return True when an Ollama server answers on the configured base URL."""
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.info("Ollama not reachable at %s: %s", self.base_url, exc)
            return False

    async def list_models(self) -> list[OllamaModel]:
        """List locally installed models, returning [] when the server is down."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Failed to list Ollama models: %s", exc)
            return []

        models: list[OllamaModel] = []
        for item in response.json().get("models", []):
            details = item.get("details") or {}
            models.append(
                OllamaModel(
                    name=item.get("name") or item.get("model") or "unknown",
                    size_bytes=item.get("size"),
                    family=details.get("family"),
                    parameters=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    modified_at=item.get("modified_at"),
                )
            )
        return models

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Run a single non-streaming generation and return the text."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, **(options or {})},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        response = await self._client.post("/api/generate", json=payload, timeout=300.0)
        response.raise_for_status()
        return response.json().get("response", "")

    async def pull(self, model: str) -> None:
        """Ensure a model is present locally (no-op if already installed)."""
        response = await self._client.post("/api/pull", json={"model": model, "stream": False})
        response.raise_for_status()
