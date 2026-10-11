"""HTTP request and response payloads (kept separate from domain schemas)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..models import JobKind
from ..schemas import MLSpec


class CreateProjectRequest(BaseModel):
    prompt: str = Field(min_length=10, description="Natural-language requirement")
    name: str | None = Field(default=None, description="Optional label; derived when omitted")
    constraints: dict[str, Any] = Field(default_factory=dict)


class UpdateSpecRequest(BaseModel):
    """User edits to the generated specification."""

    spec: MLSpec
    note: str | None = None


class StageRunRequest(BaseModel):
    """Explicit stage invocation (used by the 'advance' and 'retry' actions)."""

    params: dict[str, Any] = Field(default_factory=dict)
    approved: bool = False
    force: bool = Field(default=False, description="Re-run even if the stage already succeeded")


class StageRunResponse(BaseModel):
    stage: str
    status: str
    summary: str = ""
    duration_s: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    blocked_on: list[str] = Field(default_factory=list)
    needs_approval: bool = False
    error: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)


class EnqueueJobRequest(BaseModel):
    """Queue a long-running stage for a worker instead of running it in-request."""

    kind: JobKind
    #: Stage parameters, e.g. `{"build": True, "selected": ["hf:..."]}` for a dataset
    #: build. The worker passes them straight to the stage.
    params: dict[str, Any] = Field(default_factory=dict)
    approved: bool = Field(
        default=False, description="Explicit approval for stages that gate on it"
    )
    max_attempts: int = Field(default=1, ge=1, le=5)


class JobOut(BaseModel):
    """State of a queued job, including why it last failed."""

    id: str
    kind: str
    status: str
    attempts: int
    max_attempts: int
    claimed_by: str | None = None
    exit_code: int | None = None
    logs_path: str | None = None
    error: str | None = None
    params: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class EventOut(BaseModel):
    id: str
    at: datetime
    level: str
    stage: str | None = None
    message: str
    data: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class ArtifactOut(BaseModel):
    id: str
    kind: str
    name: str
    path: str
    size_bytes: int | None = None
    sha256: str | None = None
    meta: dict[str, Any] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectOut(BaseModel):
    id: str
    name: str
    prompt: str
    status: str
    spec: dict[str, Any] | None = None
    dataset_plan: dict[str, Any] | None = None
    selected_model: dict[str, Any] | None = None
    training_plan: dict[str, Any] | None = None
    eval_report: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectDetail(ProjectOut):
    events: list[EventOut] = Field(default_factory=list)
    artifacts: list[ArtifactOut] = Field(default_factory=list)


class CapabilityOut(BaseModel):
    """Environment diagnostics shown on the dashboard so users know what will work."""

    python_version: str
    platform: str
    sandbox_backend: str
    #: Non-null when the configured backend cannot actually run (e.g. the Docker
    #: daemon is stopped). Surfacing this up front beats failing mid-build.
    sandbox_error: str | None = None
    gpu_available: bool
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    torch_available: bool
    torch_device: str | None = None
    ollama_available: bool
    ollama_models: list[str] = Field(default_factory=list)
    planner_llm_configured: bool
    planner_llm_model: str | None = None
    hf_token_configured: bool
    hf_push_enabled: bool
    artifact_root: str
    warnings: list[str] = Field(default_factory=list)
