"""Pipeline stage abstraction.

Every stage in the SLMForge workflow implements `Stage`: it receives a
`StageContext`, does its work, and returns a `StageResult`. The orchestrator
(`pipeline.orchestrator`) is responsible for persistence, event logging, retries
and human-approval gates — stages stay pure with respect to the database.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..config import settings as default_settings
from ..log import get_logger
from ..models import Project, StageName

logger = get_logger("pipeline")


@dataclass
class StageContext:
    """Everything a stage needs to run, without touching the ORM directly."""

    project_id: str
    prompt: str
    project: Project
    artifact_dir: Path
    work_dir: Path
    settings: Settings = field(default_factory=lambda: default_settings)
    # Free-form per-stage parameters (an approved plan, user edits, retry options).
    params: dict[str, Any] = field(default_factory=dict)
    # Records produced by earlier stages in this run.
    upstream: dict[str, Any] = field(default_factory=dict)

    def emit(self, message: str, **data: Any) -> None:
        """Log a progress line that the dashboard will surface."""
        logger.info("[%s] %s", self.project_id[:8], message, extra={"data": data})


@dataclass
class StageResult:
    """The outcome of a stage.

    `output` is persisted on the StageRun row and, for the stages that define the
    project's canonical state, copied onto the Project row by the orchestrator.
    """

    output: dict[str, Any] = field(default_factory=dict)
    status: str = "succeeded"  # succeeded | skipped | failed
    summary: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    logs_path: str | None = None
    error: str | None = None
    duration_s: float | None = None
    # Set when the stage cannot proceed without a human decision.
    needs_approval: bool = False
    blocked_on: list[str] = field(default_factory=list)


class Stage(ABC):
    """Base class for a pipeline stage."""

    name: StageName
    #: Human label used in the UI and logs.
    label: str = ""
    #: If True the orchestrator pauses for explicit user approval before running.
    requires_approval: bool = False
    #: False while the stage is still a documented stub. The API reports this so
    #: the UI can disable steps honestly instead of failing at click time.
    #: Remove the `implemented = False` line in the stage module when you build it.
    implemented: bool = True

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        """Execute the stage and return its result."""
        raise NotImplementedError

    def run_timed(self, ctx: StageContext) -> StageResult:
        started = time.perf_counter()
        try:
            result = self.run(ctx)
        except Exception as exc:
            logger.exception("Stage %s failed", self.name.value)
            result = StageResult(status="failed", error=f"{type(exc).__name__}: {exc}")
        result.duration_s = round(time.perf_counter() - started, 3)
        return result


class NotImplementedStage(Stage):
    """Placeholder used until a stage is implemented in its roadmap milestone.

    Keeping the class explicit (rather than omitting the stage) means the API and
    orchestrator can be developed and tested against the full workflow today.
    """

    implemented = False

    def __init__(self, name: StageName, label: str, plan: str) -> None:
        self.name = name
        self.label = label
        self.plan = plan

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            f"Stage '{self.name.value}' is not implemented yet. Planned: {self.plan}"
        )
