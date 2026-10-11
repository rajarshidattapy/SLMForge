"""Pipeline registry, ordering and stage-result semantics."""

from __future__ import annotations

from pathlib import Path

from slmforge.models import Project, ProjectStatus, StageName
from slmforge.pipeline import PIPELINE, STAGES, get_stage, next_stage_name
from slmforge.pipeline.base import Stage, StageContext, StageResult


def test_pipeline_order_is_the_documented_workflow() -> None:
    assert [s.name for s in PIPELINE] == [
        StageName.analyze,
        StageName.curate_data,
        StageName.select_model,
        StageName.baseline,
        StageName.plan_training,
        StageName.train,
        StageName.evaluate,
        StageName.deploy,
    ]


def test_stage_registry_lookup() -> None:
    assert get_stage(StageName.train).label == "Fine-tune"
    assert set(STAGES) == {s.name for s in PIPELINE}


def test_next_stage_name_walks_the_pipeline() -> None:
    assert next_stage_name(StageName.analyze) is StageName.curate_data
    assert next_stage_name(StageName.evaluate) is StageName.deploy
    assert next_stage_name(StageName.deploy) is None


def test_costly_stages_require_approval() -> None:
    for name in (
        StageName.select_model,
        StageName.plan_training,
        StageName.train,
        StageName.deploy,
    ):
        assert get_stage(name).requires_approval is True
    for name in (StageName.analyze, StageName.curate_data, StageName.baseline, StageName.evaluate):
        assert get_stage(name).requires_approval is False


def test_unimplemented_stages_are_flagged() -> None:
    # Stages 1 and 2 shipped; every later stage must honestly say so.
    assert {s.name for s in PIPELINE if not s.implemented} == {
        StageName.select_model,
        StageName.baseline,
        StageName.plan_training,
        StageName.train,
        StageName.evaluate,
        StageName.deploy,
    }
    assert get_stage(StageName.analyze).implemented is True
    assert get_stage(StageName.curate_data).implemented is True


def test_run_timed_turns_exceptions_into_failed_results(tmp_path: Path) -> None:
    project = Project(id="p1", name="p", prompt="hi", status=ProjectStatus.draft)
    ctx = StageContext(
        project_id="p1",
        prompt="hi",
        project=project,
        artifact_dir=tmp_path,
        work_dir=tmp_path,
    )
    # select_model is still a stub, so its NotImplementedError becomes a failure.
    result = get_stage(StageName.select_model).run_timed(ctx)
    assert result.status == "failed"
    assert result.error is not None and "NotImplementedError" in result.error
    assert result.duration_s is not None and result.duration_s >= 0


def test_stage_result_ok_defaults() -> None:
    assert StageResult().status == "succeeded"
    assert StageResult().needs_approval is False


# --- approval gates --------------------------------------------------------


class _GatedStage(Stage):
    """A stage that produces a reviewable artifact but is not finished."""

    name = StageName.curate_data
    label = "Gated"

    def run(self, ctx: StageContext) -> StageResult:
        return StageResult(
            output={"plan": "draft"},
            summary="Plan ready for review",
            needs_approval=True,
            blocked_on=["approve_dataset_build"],
        )


class _DoneStage(_GatedStage):
    name = StageName.select_model

    def run(self, ctx: StageContext) -> StageResult:
        return StageResult(output={"chosen": "model"}, summary="Chosen")


def test_a_stage_waiting_for_approval_records_output_but_does_not_advance(
    session,
) -> None:
    """A plan is worth persisting; a completed step is what advances the project."""
    from slmforge.pipeline.orchestrator import Orchestrator

    project = Project(id="gate1", name="p", prompt="hi", status=ProjectStatus.specified)
    session.add(project)
    session.commit()

    result = Orchestrator(session).run_stage(project, _GatedStage())
    assert result.status == "succeeded"
    session.refresh(project)
    assert project.dataset_plan == {"plan": "draft"}
    assert project.status == ProjectStatus.specified, "an unapproved step is not complete"


def test_a_completed_stage_does_advance_the_project(session) -> None:
    from slmforge.pipeline.orchestrator import Orchestrator

    project = Project(id="gate2", name="p", prompt="hi", status=ProjectStatus.data_ready)
    session.add(project)
    session.commit()

    Orchestrator(session).run_stage(project, _DoneStage())
    session.refresh(project)
    assert project.status == ProjectStatus.model_selected
