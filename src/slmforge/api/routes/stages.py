"""Pipeline stage execution and specification review endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ...models import Event, ProjectStatus, StageName, StageRun
from ...pipeline import PIPELINE, get_stage
from ...pipeline.orchestrator import Orchestrator
from ..deps import ProjectDep, SessionDep
from ..payloads import StageRunRequest, StageRunResponse, UpdateSpecRequest

router = APIRouter(prefix="/projects/{project_id}", tags=["pipeline"])


def _is_unimplemented(error: str | None) -> bool:
    return error is not None and error.startswith("NotImplementedError")


@router.get("/pipeline")
def pipeline_overview() -> dict:
    """List the pipeline stages, their order and whether they are implemented yet.

    The web UI uses this to disable/unlock steps honestly while the prototype is
    being built out stage by stage. The `implemented` flag comes from each stage
    class, so it can only be wrong where the code is.
    """
    return {
        "stages": [
            {
                "name": stage.name.value,
                "label": stage.label,
                "requires_approval": stage.requires_approval,
                "implemented": stage.implemented,
            }
            for stage in PIPELINE
        ]
    }


@router.get("/spec")
def get_spec(project: ProjectDep) -> dict:
    """Return the current ML specification (or an empty state)."""
    return {"spec": project.spec, "status": project.status}


@router.put("/spec")
def update_spec(payload: UpdateSpecRequest, project: ProjectDep, session: SessionDep) -> dict:
    """Persist user edits to the specification before later stages consume it."""
    project.spec = payload.spec.model_dump(mode="json", by_alias=True)
    if project.status == ProjectStatus.draft:
        project.status = ProjectStatus.specified
    session.add(
        Event(
            project_id=project.id,
            stage=StageName.analyze.value,
            message="Specification updated by user",
            data={"note": payload.note} if payload.note else None,
        )
    )
    session.commit()
    return {"spec": project.spec, "status": project.status}


@router.post("/stages/{stage_name}/run", response_model=StageRunResponse)
def run_stage(
    stage_name: StageName,
    payload: StageRunRequest,
    project: ProjectDep,
    session: SessionDep,
) -> StageRunResponse:
    """Run a single pipeline stage synchronously.

    Long stages (training, evaluation) must be executed through the job queue
    instead; this endpoint rejects them so the request cannot hang for an hour.
    """
    if stage_name in (StageName.train, StageName.baseline, StageName.evaluate):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Stage '{stage_name.value}' is long-running; enqueue it as a job "
                "rather than calling it synchronously."
            ),
        )

    # Planning a dataset is cheap (one remote profile). Building one downloads and
    # rewrites files, so it goes to the worker: a slow Hub read must not occupy a
    # request thread until the client times out.
    if stage_name is StageName.curate_data and payload.params.get("build"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Building a dataset downloads data; enqueue it with "
                "POST /api/projects/{project_id}/jobs (kind=dataset_build) and let a "
                "worker run it. Run the stage without `build` to review the plan first."
            ),
        )

    stage = get_stage(stage_name)
    # Answer "is this built?" before "did you approve it?": asking a user to approve
    # work that cannot run yet is a worse experience than a plain 501.
    if not stage.implemented:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Stage '{stage_name.value}' is not implemented yet. See docs/ROADMAP.md.",
        )
    if stage.requires_approval and not payload.approved:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail=f"Stage '{stage_name.value}' requires explicit approval (approved=true).",
        )

    orchestrator = Orchestrator(session)
    result = orchestrator.run_stage(project, stage, params=payload.params)

    if _is_unimplemented(result.error):
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=result.error)

    return StageRunResponse(
        stage=stage_name.value,
        status=result.status,
        summary=result.summary,
        duration_s=result.duration_s,
        metrics=result.metrics,
        blocked_on=result.blocked_on,
        needs_approval=result.needs_approval,
        error=result.error,
        output=result.output,
    )


@router.post("/advance", response_model=StageRunResponse)
def advance(
    payload: StageRunRequest,
    project: ProjectDep,
    session: SessionDep,
) -> StageRunResponse:
    """Run the next not-yet-succeeded stage, in canonical order."""
    done = succeeded_stages(session, project.id)
    for stage in PIPELINE:
        if stage.name in done and not payload.force:
            continue
        return run_stage(stage.name, payload, project, session)
    return StageRunResponse(stage="", status="succeeded", summary="All stages already completed")


@router.get("/stages")
def list_stage_runs(project: ProjectDep, session: SessionDep) -> list[dict]:
    """History of stage executions for this project."""
    runs = (
        session.query(StageRun)
        .filter(StageRun.project_id == project.id)
        .order_by(StageRun.started_at.desc())
        .all()
    )
    return [
        {
            "id": run.id,
            "stage": run.stage.value,
            "status": run.status.value,
            "attempt": run.attempt,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "duration_s": run.duration_s,
            "summary": (run.output or {}).get("summary") if run.output else None,
            "error": run.error,
        }
        for run in runs
    ]


@router.get("/next")
def next_step(project: ProjectDep, session: SessionDep) -> dict:
    """Report the next stage the user should run (drives the UI's primary action)."""
    done = succeeded_stages(session, project.id)
    for stage in PIPELINE:
        if stage.name not in done:
            return {
                "next": stage.name.value,
                "label": stage.label,
                "requires_approval": stage.requires_approval,
            }
    return {"next": None, "label": "Complete", "requires_approval": False}


def succeeded_stages(session: SessionDep, project_id: str) -> set[StageName]:
    """Stage names that have at least one successful run for a project."""
    rows = (
        session.query(StageRun.stage)
        .filter(StageRun.project_id == project_id, StageRun.status == "succeeded")
        .all()
    )
    return {row[0] for row in rows}


__all__ = ["advance", "get_spec", "router", "run_stage", "succeeded_stages", "update_spec"]
