"""Shared enums and small value objects used across stage contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class TaskType(StrEnum):
    """ML task families the platform understands.

    The MVP supports `classification` and `structured_generation` end to end;
    the rest are recognized so the analyzer can report "not yet supported"
    honestly instead of mis-routing a request.
    """

    classification = "classification"
    structured_generation = "structured_generation"
    extraction = "extraction"
    summarization = "summarization"
    rewriting = "rewriting"
    qa_grounded = "qa_grounded"
    embedding = "embedding"
    unsupported = "unsupported"


class Modality(StrEnum):
    text = "text"
    text_pair = "text_pair"


class OutputFormat(StrEnum):
    json_object = "json_object"
    label = "label"
    span = "span"
    free_text = "free_text"
    multi_label = "multi_label"


class License(BaseModel):
    """Normalized license metadata for a dataset or model."""

    id: str = Field(description="SPDX-like identifier, e.g. apache-2.0, mit, cc-by-4.0")
    name: str | None = None
    url: str | None = None
    # Whether the license appears to permit commercial use. None = unknown/needs review.
    commercial_ok: bool | None = None
    share_alike: bool = False
    unknown: bool = False
    notes: str | None = None


class Provenance(BaseModel):
    """Where an artifact came from and how it was transformed."""

    source: str = Field(description="hf_dataset | user_upload | synthetic | catalog")
    uri: str | None = None
    revision: str | None = None
    license: License | None = None
    retrieved_at: datetime = Field(default_factory=utcnow)
    transformations: list[str] = Field(default_factory=list)
    synthetic: bool = False
    synthetic_generator: str | None = None
    notes: str | None = None


class ResourceEstimate(BaseModel):
    """Pre-flight estimate shown to the user before an expensive job runs."""

    vram_gb: float | None = None
    ram_gb: float | None = None
    disk_gb: float | None = None
    cpu_cores: int | None = None
    wall_clock_minutes: float | None = None
    gpu_hours: float | None = None
    estimated_cost_usd: float | None = None
    confidence: str = "low"  # low | medium | high
    notes: str | None = None


class Metric(BaseModel):
    """A single measured value with enough context to be trustworthy."""

    name: str
    value: float | None = None
    unit: str | None = None
    higher_is_better: bool = True
    on_split: str | None = None
    n_samples: int | None = None
    std: float | None = None
    notes: str | None = None


class ClarifyingQuestion(BaseModel):
    """A question the analyzer needs answered before it can proceed safely."""

    id: str
    question: str
    why: str
    options: list[str] = Field(default_factory=list)
    default: str | None = None
    blocking: bool = True
