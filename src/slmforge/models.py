"""ORM models describing the persistent state of a SLMForge build.

Tables
------
Project     one build request, from prompt to deployed endpoint
StageRun    one execution of a pipeline stage (analyze, curate, train, ...)
Job         a unit of sandboxed work claimed by a worker
Artifact    a file produced by a job (dataset, checkpoint, report, logs)
Deployment  a published endpoint for a project's selected model
Event       an append-only audit/telemetry log for the project dashboard
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class ProjectStatus(StrEnum):
    draft = "draft"
    specified = "specified"
    data_ready = "data_ready"
    model_selected = "model_selected"
    baselined = "baselined"
    planned = "planned"
    training = "training"
    trained = "trained"
    evaluated = "evaluated"
    deployed = "deployed"
    failed = "failed"
    cancelled = "cancelled"


class StageName(StrEnum):
    analyze = "analyze"
    curate_data = "curate_data"
    select_model = "select_model"
    baseline = "baseline"
    plan_training = "plan_training"
    train = "train"
    evaluate = "evaluate"
    deploy = "deploy"


class RunStatus(StrEnum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    skipped = "skipped"


class JobKind(StrEnum):
    dataset_build = "dataset_build"
    baseline_eval = "baseline_eval"
    training = "training"
    evaluation = "evaluation"
    deployment = "deployment"


class DeploymentStatus(StrEnum):
    pending = "pending"
    provisioning = "provisioning"
    live = "live"
    failed = "failed"
    stopped = "stopped"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    prompt: Mapped[str] = mapped_column(Text)
    status: Mapped[ProjectStatus] = mapped_column(
        SAEnum(ProjectStatus), default=ProjectStatus.draft
    )
    # Parsed, user-editable ML specification (schemas/spec.py :: MLSpec).
    spec: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    # Chosen dataset plan, selected model, training plan and eval report.
    dataset_plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    selected_model: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    training_plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    eval_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    stage_runs: Mapped[list[StageRun]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="StageRun.started_at"
    )
    jobs: Mapped[list[Job]] = relationship(back_populates="project", cascade="all, delete-orphan")
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    deployments: Mapped[list[Deployment]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    events: Mapped[list[Event]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class StageRun(Base):
    __tablename__ = "stage_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    stage: Mapped[StageName] = mapped_column(SAEnum(StageName))
    status: Mapped[RunStatus] = mapped_column(SAEnum(RunStatus), default=RunStatus.pending)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    duration_s: Mapped[float | None] = mapped_column(Float, default=None)
    # Structured stage output plus a human-readable summary for the dashboard.
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    project: Mapped[Project] = relationship(back_populates="stage_runs")


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    stage_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("stage_runs.id", ondelete="SET NULL"), default=None
    )
    kind: Mapped[JobKind] = mapped_column(SAEnum(JobKind))
    status: Mapped[RunStatus] = mapped_column(SAEnum(RunStatus), default=RunStatus.pending)
    # Worker bookkeeping for safe claiming / lease expiry.
    claimed_by: Mapped[str | None] = mapped_column(String(100), default=None)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=1)
    # Requested and observed resource usage, for cost/latency reporting.
    resources: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    logs_path: Mapped[str | None] = mapped_column(String(500), default=None)
    exit_code: Mapped[int | None] = mapped_column(Integer, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    project: Mapped[Project] = relationship(back_populates="jobs")


class Artifact(TimestampMixin, Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    job_id: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), default=None
    )
    kind: Mapped[str] = mapped_column(String(50))  # dataset | checkpoint | report | logs | model
    name: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(String(1000))
    size_bytes: Mapped[int | None] = mapped_column(Integer, default=None)
    sha256: Mapped[str | None] = mapped_column(String(64), default=None)
    # Provenance: source, license, transformations, synthetic flag.
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    project: Mapped[Project] = relationship(back_populates="artifacts")


class Deployment(TimestampMixin, Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    target: Mapped[str] = mapped_column(String(50))  # hf_endpoint | hf_space | local_openai
    status: Mapped[DeploymentStatus] = mapped_column(
        SAEnum(DeploymentStatus), default=DeploymentStatus.pending
    )
    model_id: Mapped[str] = mapped_column(String(300))
    revision: Mapped[str | None] = mapped_column(String(100), default=None)
    endpoint_url: Mapped[str | None] = mapped_column(String(1000), default=None)
    # Store only a reference/name; the secret value lives in the secret store.
    auth_ref: Mapped[str | None] = mapped_column(String(200), default=None)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    project: Mapped[Project] = relationship(back_populates="deployments")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    level: Mapped[str] = mapped_column(String(20), default="info")
    stage: Mapped[str | None] = mapped_column(String(50), default=None)
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    project: Mapped[Project] = relationship(back_populates="events")
