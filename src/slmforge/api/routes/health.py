"""Health and capability endpoints."""

from __future__ import annotations

import platform
import sys

from fastapi import APIRouter

from ...config import settings
from ...integrations.ollama import OllamaClient
from ..payloads import CapabilityOut

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "slmforge", "env": settings.env}


@router.get("/capabilities", response_model=CapabilityOut)
async def capabilities(probe: bool = True) -> CapabilityOut:
    """Report what this machine can actually do.

    The dashboard uses this to warn users up front (e.g. CPU-only torch, no Ollama
    server, no GPU) instead of failing mid-build.
    """
    warnings: list[str] = []
    gpu_available = False
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    torch_available = False
    torch_device: str | None = None

    try:
        import torch

        torch_available = True
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_available = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_vram_gb = round(props.total_memory / (1024**3), 1)
        else:
            warnings.append(
                "torch is CPU-only in this environment; fine-tuning will be very slow. "
                "Install a CUDA build of torch to use the GPU."
            )
    except ImportError:
        warnings.append("torch is not installed — install the 'ml' extra to train or evaluate.")

    ollama_available = False
    ollama_models: list[str] = []
    if probe:
        client = OllamaClient()
        try:
            ollama_available = await client.is_available()
            if ollama_available:
                ollama_models = [m.name for m in await client.list_models()]
            else:
                warnings.append(
                    f"Ollama is not reachable at {settings.ollama_base_url}; "
                    "local model candidates will be unavailable."
                )
        finally:
            await client.close()

    planner_configured = settings.llm_provider != "none" and bool(settings.llm_base_url)
    if not planner_configured:
        warnings.append(
            "No planner LLM configured; the analyzer cannot interpret prompts. "
            "Set SLMFORGE_LLM_* or run Ollama."
        )
    elif settings.llm_provider == "ollama" and probe and not ollama_available:
        # Configured on paper but unusable in practice — say so rather than
        # letting the user discover it when the analyze stage fails.
        warnings.append(
            "The planner LLM is set to Ollama, but Ollama is not reachable, so the "
            "analyzer cannot run until the server is started or a hosted provider is set."
        )

    return CapabilityOut(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        sandbox_backend=settings.sandbox_backend,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        torch_available=torch_available,
        torch_device=torch_device,
        ollama_available=ollama_available,
        ollama_models=ollama_models,
        planner_llm_configured=planner_configured,
        planner_llm_model=settings.llm_model if planner_configured else None,
        hf_token_configured=bool(settings.hf_token),
        hf_push_enabled=settings.hf_push_enabled,
        artifact_root=str(settings.artifact_root),
        warnings=warnings,
    )
