"""Orchestrator: runs stages with persistence, events and approval gates.

This is the only module that is allowed to mutate project state on behalf of the
pipeline. Stages stay free of database concerns so they are easy to test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..log import get_logger
from ..models import Event, Project, ProjectStatus, RunStatus, StageName, StageRun
from .base import Stage, StageContext, StageResult

logger = get_logger("orchestrator")

#: Which project column (if any) a stage's output is copied onto.
_STAGE_TO_COLUMN: dict[StageName, str] = {
    StageName.analyze: "spec",
    StageName.curate_data: "dataset_plan",
    StageName.select_model: "selected_model",
    StageName.plan_training: "training_plan",
    StageName.evaluate: "eval_report",
}

#: Project status to set once a stage succeeds.
_STAGE_TO_STATUS: dict[StageName, ProjectStatus] = {
    StageName.analyze: ProjectStatus.specified,
    StageName.curate_data: ProjectStatus.data_ready,
    StageName.select_model: ProjectStatus.model_selected,
    StageName.baseline: ProjectStatus.baselined,
    StageName.plan_training: ProjectStatus.planned,
    StageName.train: ProjectStatus.trained,
    StageName.evaluate: ProjectStatus.evaluated,
    StageName.deploy: ProjectStatus.deployed,
}


class Orchestrator:
    """Coordinates stage execution against a database session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # -- events ---------------------------------------------------------
    def emit(
        self,
        project_id: str,
        message: str,
        *,
        level: str = "info",
        stage: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> Event:
        event = Event(project_id=project_id, level=level, stage=stage, message=message, data=data)
        self.session.add(event)
        self.session.flush()
        return event

    # -- context --------------------------------------------------------
    def build_context(
        self,
        project: Project,
        stage: Stage,
        *,
        params: dict[str, Any] | None = None,
        upstream: dict[str, Any] | None = None,
    ) -> StageContext:
        from ..config import settings

        artifact_dir = Path(settings.artifact_root) / project.id
        work_dir = artifact_dir / "work" / stage.name.value
        artifact_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        return StageContext(
            project_id=project.id,
            prompt=project.prompt,
            project=project,
            artifact_dir=artifact_dir,
            work_dir=work_dir,
            settings=settings,
            params=params or {},
            upstream=upstream or {},
        )

    # -- execution ------------------------------------------------------
    def run_stage(
        self,
        project: Project,
        stage: Stage,
        *,
        params: dict[str, Any] | None = None,
        upstream: dict[str, Any] | None = None,
    ) -> StageResult:
        """Run one stage and persist its StageRun row, events and status change."""
        stage_run = StageRun(project_id=project.id, stage=stage.name, status=RunStatus.running)
        self.session.add(stage_run)
        self.session.commit()

        self.emit(project.id, f"Stage '{stage.label}' started", stage=stage.name.value)
        ctx = self.build_context(project, stage, params=params, upstream=upstream)
        result = stage.run_timed(ctx)

        stage_run.status = RunStatus(result.status)
        stage_run.output = result.output or None
        stage_run.error = result.error
        stage_run.duration_s = result.duration_s

        level = "info" if result.status == "succeeded" else "error"
        detail = {"metrics": result.metrics, "blocked_on": result.blocked_on}
        self.emit(
            project.id,
            result.summary or f"Stage '{stage.label}' {result.status}",
            level=level,
            stage=stage.name.value,
            data=detail or None,
        )

        if result.status == "succeeded":
            self._apply_result(project, stage.name, result)

        self.session.commit()
        return result

    def _apply_result(self, project: Project, stage_name: StageName, result: StageResult) -> None:
        column = _STAGE_TO_COLUMN.get(stage_name)
        if column and result.output:
            setattr(project, column, result.output)
        status = _STAGE_TO_STATUS.get(stage_name)
        if status:
            project.status = status
        for artifact in result.artifacts:
            from ..models import Artifact

            self.session.add(
                Artifact(
                    project_id=project.id,
                    kind=artifact.get("kind", "file"),
                    name=artifact.get("name", "artifact"),
                    path=str(artifact.get("path", "")),
                    size_bytes=artifact.get("size_bytes"),
                    sha256=artifact.get("sha256"),
                    meta=artifact.get("meta"),
                )
            )
