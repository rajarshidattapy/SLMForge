"""The structured ML specification produced from a user's natural-language prompt.

This is the central contract of the platform: every later stage reads it and the
user can review/edit it before anything expensive happens.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .common import (
    ClarifyingQuestion,
    Modality,
    OutputFormat,
    Provenance,
    ResourceEstimate,
    TaskType,
    utcnow,
)


class FieldSpec(BaseModel):
    """One field of the expected input or output payload."""

    name: str
    type: str = Field(description="string | integer | number | boolean | enum | array | object")
    description: str | None = None
    required: bool = True
    enum: list[str] = Field(default_factory=list)
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")

    model_config = {"populate_by_name": True}


class OutputSpec(BaseModel):
    """Shape and constraints of what the model must return."""

    format: OutputFormat = OutputFormat.free_text
    fields: list[FieldSpec] = Field(default_factory=list)
    json_schema: dict[str, Any] | None = None
    grammar: str | None = Field(default=None, description="Optional constrained-decoding grammar")
    max_tokens: int | None = None
    notes: str | None = None


class EvaluationPlan(BaseModel):
    """How success will be measured, decided up front."""

    primary_metric: str = "macro_f1"
    secondary_metrics: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["validation", "test"])
    test_fraction: float = 0.2
    validation_fraction: float = 0.1
    min_acceptance: dict[str, float] = Field(default_factory=dict)
    error_analysis: list[str] = Field(default_factory=list)


class DeploymentPreference(BaseModel):
    target: str = "hf_endpoint"  # hf_endpoint | hf_space | local_openai
    latency_budget_ms: float | None = None
    max_monthly_cost_usd: float | None = None
    privacy: str = "standard"  # standard | no_external_training | local_only
    allowed_licenses: list[str] = Field(default_factory=list)


class MLSpec(BaseModel):
    """A machine-readable description of the model the user wants built."""

    # Identity
    title: str
    summary: str
    task_type: TaskType
    modality: Modality = Modality.text

    # I/O contract
    input_description: str
    input_fields: list[FieldSpec] = Field(default_factory=list)
    output: OutputSpec = Field(default_factory=OutputSpec)
    allowed_labels: list[str] = Field(default_factory=list)
    language: list[str] = Field(default_factory=lambda: ["en"])
    domain: str | None = None

    # Success & constraints
    evaluation: EvaluationPlan = Field(default_factory=EvaluationPlan)
    constraints: dict[str, Any] = Field(default_factory=dict)
    deployment: DeploymentPreference = Field(default_factory=DeploymentPreference)

    # Honesty / process
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[ClarifyingQuestion] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    is_high_stakes: bool = Field(
        default=False,
        description="Medical/legal/financial/employment use — requires extra safeguards.",
    )
    readiness: str = "ready_for_review"  # ready_for_review | needs_clarification | unsupported

    # Method recommendation happens later, but the analyzer may pre-flag it.
    recommended_approach: str | None = Field(
        default=None, description="finetune | prompt_only | rag | needs_data"
    )
    approach_rationale: str | None = None

    # Meta
    generated_by: str | None = None
    generated_at: datetime = Field(default_factory=utcnow)
    estimate: ResourceEstimate | None = None
    provenance: list[Provenance] = Field(default_factory=list)
