"""Baseline and baseline-vs-adapted evaluation contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .common import Metric, utcnow


class SamplePrediction(BaseModel):
    """One row of the error-analysis table shown in the dashboard."""

    input: str
    expected: str | None = None
    predicted: str | None = None
    correct: bool | None = None
    latency_ms: float | None = None
    error_kind: str | None = None


class EvalRun(BaseModel):
    """Metrics for a single model on a single split."""

    label: str = Field(description="baseline | finetuned")
    model_id: str
    split: str = "test"
    n_samples: int = 0
    metrics: list[Metric] = Field(default_factory=list)
    # Format/robustness checks separate from task quality.
    valid_output_rate: float | None = None
    schema_violations: int | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    peak_vram_gb: float | None = None
    throughput_tok_s: float | None = None
    cost_per_1k_calls_usd: float | None = None
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    samples: list[SamplePrediction] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Comparison(BaseModel):
    """Baseline vs fine-tuned, with an explicit verdict."""

    metric: str
    baseline: float | None = None
    finetuned: float | None = None
    delta: float | None = None
    delta_pct: float | None = None
    improved: bool | None = None
    statistically_significant: bool | None = None
    n_samples: int | None = None


class EvalReport(BaseModel):
    """The full evaluation artifact attached to a project."""

    id: str
    project_id: str
    created_at: datetime = Field(default_factory=utcnow)
    baseline: EvalRun | None = None
    finetuned: EvalRun | None = None
    comparisons: list[Comparison] = Field(default_factory=list)
    acceptance: dict[str, bool] = Field(default_factory=dict)
    verdict: str = Field(
        default="inconclusive",
        description="accept_finetuned | keep_baseline | inconclusive | needs_more_data",
    )
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    best_alternative: str | None = Field(
        default=None, description="If fine-tuning did not help: prompt_only | rag | different_model"
    )
    seed: int | None = None
    config_fingerprint: str | None = None
