"""Training strategy, hyperparameters and run-result contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .common import ResourceEstimate, utcnow

METHODS = ("sft", "lora", "qlora", "prompt_only", "rag", "no_finetune")


class Hyperparameters(BaseModel):
    """Everything needed to reproduce a run."""

    epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    quantization: str | None = "nf4"  # nf4 | int8 | None
    double_quant: bool = True
    bf16: bool = True
    packing: bool = False
    early_stopping_patience: int | None = 3
    extra: dict[str, Any] = Field(default_factory=dict)


class TrainingPlan(BaseModel):
    """The proposed adaptation strategy — reviewed before an expensive run."""

    recommendation: str = Field(description="finetune | prompt_only | rag | no_finetune")
    method: str = Field(default="qlora", description=" | ".join(METHODS))
    base_model_id: str
    rationale: str
    expected_benefit: str | None = None
    data_requirements: list[str] = Field(default_factory=list)
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    estimate: ResourceEstimate | None = None
    requires_gpu: bool = False
    gpu_available: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    requires_approval: bool = True
    # Reproducibility: the generated script plus a hash of its inputs.
    script_path: str | None = None
    script_sha256: str | None = None
    config_fingerprint: str | None = None


class TrainingRecord(BaseModel):
    """What actually happened during a training job."""

    job_id: str
    method: str
    status: str
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    steps_completed: int | None = None
    final_loss: float | None = None
    loss_curve: list[dict[str, float]] = Field(default_factory=list)
    checkpoint_path: str | None = None
    adapter_path: str | None = None
    merged: bool = False
    peak_vram_gb: float | None = None
    wall_clock_s: float | None = None
    exit_code: int | None = None
    error: str | None = None
    logs_path: str | None = None
