"""Deployment request/record contracts for hosted endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .common import utcnow


class DeployRequest(BaseModel):
    """User-approved deployment intent."""

    target: str = Field(default="hf_endpoint", description="hf_endpoint | hf_space | local_openai")
    model_id: str
    push_to_hub: bool = True
    repo_id: str | None = Field(default=None, description="e.g. org/slmforge-ticket-classifier")
    private: bool = True
    merge_adapter: bool = True
    instance_type: str | None = "nvidia-t4"
    min_replicas: int = 1
    max_replicas: int = 2
    latency_budget_ms: float | None = None
    monthly_budget_usd: float | None = None
    approved: bool = False


class UsageStats(BaseModel):
    """Aggregate usage reported for an endpoint."""

    window: str = "24h"
    requests: int = 0
    errors: int = 0
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    estimated_cost_usd: float | None = None
    collected_at: datetime = Field(default_factory=utcnow)


class DeployRecord(BaseModel):
    """What was published, where, and how to call it."""

    id: str
    target: str
    status: str = "pending"  # pending | provisioning | live | failed | stopped
    model_id: str
    revision: str | None = None
    endpoint_url: str | None = None
    auth_scheme: str | None = Field(default=None, description="bearer | none")
    auth_ref: str | None = Field(
        default=None, description="Secret store key referencing the API key; never the value"
    )
    request_example: dict | None = None
    response_example: dict | None = None
    model_version: str | None = None
    eval_summary: str | None = None
    known_limitations: list[str] = Field(default_factory=list)
    usage: UsageStats | None = None
    created_at: datetime = Field(default_factory=utcnow)
