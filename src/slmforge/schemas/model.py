"""Candidate model catalogue and compatibility contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import License, ResourceEstimate


class FineTuneSupport(BaseModel):
    """Whether and how a candidate can actually be adapted."""

    methods: list[str] = Field(default_factory=list, description="sft | lora | qlora")
    supported: bool = False
    min_vram_gb: float | None = None
    requires_gpu: bool = False
    tooling: list[str] = Field(default_factory=list, description="peft/trl/unsloth/...")
    notes: str | None = None


class ModelCandidate(BaseModel):
    """A base model the user could build on."""

    id: str = Field(description="hf:<repo_id> | ollama:<name> | catalog:<slug>")
    name: str
    source: str = Field(description="hf_hub | ollama | catalog")
    family: str | None = None
    parameters_b: float | None = Field(default=None, description="Parameter count, billions")
    context_window: int | None = None
    quantization: str | None = None
    # Capability
    task_fit: list[str] = Field(default_factory=list)
    instruction_tuned: bool = False
    # Practical constraints
    license: License | None = None
    local_available: bool = False
    downloaded: bool = False
    est: ResourceEstimate | None = None
    finetune: FineTuneSupport = Field(default_factory=FineTuneSupport)
    # Ranking
    rank_score: float | None = Field(default=None, ge=0, le=1)
    rank_rationale: str | None = None
    warnings: list[str] = Field(default_factory=list)
    approved: bool = False


class ModelSelection(BaseModel):
    """The user-approved choice plus the alternatives that were considered."""

    selected: ModelCandidate | None = None
    shortlist: list[ModelCandidate] = Field(default_factory=list)
    compared_on: list[str] = Field(default_factory=list)
    decision_notes: str | None = None
    requires_approval: bool = True
