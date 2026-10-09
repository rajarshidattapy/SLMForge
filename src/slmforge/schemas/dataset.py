"""Dataset discovery, curation plan and dataset card contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .common import Provenance, utcnow


class DatasetCandidate(BaseModel):
    """A dataset source found for the task, before it is downloaded."""

    id: str = Field(description="hf:<repo_id> | upload:<upload_id> | synthetic:<run_id>")
    name: str
    source: str = Field(description="hf_dataset | user_upload | synthetic")
    uri: str | None = None
    rows: int | None = None
    columns: list[str] = Field(default_factory=list)
    # How well it matches the spec, and why (never just a bare score).
    fit_score: float | None = Field(default=None, ge=0, le=1)
    fit_rationale: str | None = None
    label_coverage: dict[str, int] = Field(default_factory=dict)
    license_note: str | None = None
    commercial_ok: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    approved: bool = False
    provenance: Provenance | None = None


class LabelMapping(BaseModel):
    """How raw dataset labels map onto the spec's `allowed_labels`."""

    raw: str
    mapped: str | None = None
    action: str = "keep"  # keep | map | drop
    reason: str | None = None


class DataQualityIssue(BaseModel):
    kind: str  # missing_label | duplicate | too_short | pii_suspected | schema_mismatch | ...
    severity: str = "warning"  # info | warning | error
    count: int = 0
    detail: str | None = None
    examples: list[str] = Field(default_factory=list)


class SplitSizes(BaseModel):
    train: int = 0
    validation: int = 0
    test: int = 0


class DatasetPlan(BaseModel):
    """The proposed data strategy for the project — reviewed before building."""

    task_type: str
    target_examples: int
    candidates: list[DatasetCandidate] = Field(default_factory=list)
    selected: list[str] = Field(default_factory=list, description="Chosen candidate ids")
    augmentation: list[str] = Field(default_factory=list, description="synthetic | none | ...")
    label_mapping: list[LabelMapping] = Field(default_factory=list)
    dedupe: bool = True
    normalize: bool = True
    splits: SplitSizes = Field(default_factory=SplitSizes)
    leakage_checks: list[str] = Field(default_factory=list)
    sufficiency: str = "unknown"  # sufficient | marginal | insufficient | unknown
    sufficiency_notes: str | None = None
    estimated_build_minutes: float | None = None
    requires_user_input: bool = False


class DatasetCard(BaseModel):
    """What was actually built, with provenance for every contributing source."""

    id: str
    project_id: str
    built_at: datetime = Field(default_factory=utcnow)
    total_examples: int = 0
    splits: SplitSizes = Field(default_factory=SplitSizes)
    label_distribution: dict[str, int] = Field(default_factory=dict)
    sources: list[Provenance] = Field(default_factory=list)
    synthetic_fraction: float = 0.0
    issues: list[DataQualityIssue] = Field(default_factory=list)
    # Relative paths to the materialized split files (JSONL) inside the project artifacts.
    files: dict[str, str] = Field(default_factory=dict)
    sha256: dict[str, str] = Field(default_factory=dict)
    format: str = "chat_jsonl"
    notes: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
