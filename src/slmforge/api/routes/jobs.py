"""Job endpoints.

Long work never runs in the API process: a stage that downloads, trains or evaluates
is enqueued here and executed by a worker. The endpoints stay thin — validate,
enqueue, return — because the worker owns the work.

A job's approved stage parameters travel in ``resources["params"]`` (see
`jobs.handlers._run_stage_for_job`), which is also where a resource request lives.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from ...jobs.queue import JobQueue
from ...log import get_logger
from ...models import Event, Job, JobKind, RunStatus, StageName
from ...pipeline import get_stage
from ..deps import ProjectDep, SessionDep
from ..payloads import EnqueueJobRequest, JobOut

router = APIRouter(prefix="/projects/{project_id}", tags=["jobs"])
logger = get_logger("api.jobs")

#: Which pipeline stage a job kind actually runs, so the API can refuse to enqueue
#: work for a stage that is still a stub instead of failing the job minutes later.
_KIND_TO_STAGE: dict[JobKind, StageName] = {
    JobKind.dataset_build: StageName.curate_data,
    JobKind.baseline_eval: StageName.baseline,
    JobKind.training: StageName.train,
    JobKind.evaluation: StageName.evaluate,
    JobKind.deployment: StageName.deploy,
}

#: Job kinds with a handler today; enqueueing the rest would queue work nobody runs.
_IMPLEMENTED_HANDLERS = {JobKind.dataset_build}


def _to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        kind=job.kind.value,
        status=job.status.value,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        claimed_by=job.claimed_by,
        exit_code=job.exit_code,
        logs_path=job.logs_path,
        error=job.error,
        params=(job.resources or {}).get("params") or None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED)
def enqueue_job(payload: EnqueueJobRequest, project: ProjectDep, session: SessionDep) -> JobOut:
    """Queue a stage to run on a worker.

    Approval is explicit here for the same reason it is on the synchronous endpoint:
    a dataset build downloads data and a training job costs money, so neither happens
    because someone clicked "next".
    """
    stage_name = _KIND_TO_STAGE.get(payload.kind)
    if stage_name is None:  # pragma: no cover - every kind is mapped
        raise HTTPException(status_code=422, detail=f"Unmapped job kind: {payload.kind}")

    stage = get_stage(stage_name)
    if not stage.implemented:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Stage '{stage_name.value}' is not implemented yet; there is no work to queue.",
        )
    if payload.kind not in _IMPLEMENTED_HANDLERS:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"No worker handler for job kind '{payload.kind.value}' yet. "
                "Its stage is implemented, but the queued path arrives with its milestone."
            ),
        )
    if stage.requires_approval and not payload.approved:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail=f"Job kind '{payload.kind.value}' requires explicit approval (approved=true).",
        )

    queue = JobQueue(session)
    job = queue.enqueue(
        project.id,
        payload.kind,
        resources={"params": payload.params} if payload.params else None,
        max_attempts=payload.max_attempts,
    )
    session.add(
        Event(
            project_id=project.id,
            stage=stage_name.value,
            message=f"Queued {payload.kind.value} job",
            data={"job_id": job.id, "params": payload.params or None},
        )
    )
    session.commit()
    logger.info("Queued %s job %s for project %s", payload.kind.value, job.id, project.id[:8])
    return _to_out(job)


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(project: ProjectDep, session: SessionDep) -> list[JobOut]:
    """This project's jobs, newest first — the UI's queue view."""
    rows = (
        session.execute(
            select(Job)
            .where(Job.project_id == project.id)
            .order_by(Job.created_at.desc())
            .limit(100)
        )
        .scalars()
        .all()
    )
    return [_to_out(job) for job in rows]


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, project: ProjectDep, session: SessionDep) -> JobOut:
    """Cancel a job.

    A still-pending job is cancelled immediately. A running job is only *asked* to
    stop: the worker owns the process, so it decides when to honour that, and this
    endpoint never claims the work has stopped when it may still be running.
    """
    job = session.get(Job, job_id)
    if job is None or job.project_id != project.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.status not in (RunStatus.pending, RunStatus.running):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is already {job.status.value}.",
        )

    JobQueue(session).request_cancel(job_id)
    session.refresh(job)
    cancel_requested = job.status is RunStatus.running
    session.add(
        Event(
            project_id=project.id,
            message=f"Job {job_id[:8]} cancellation requested",
            data={"job_id": job_id, "still_running": cancel_requested},
        )
    )
    session.commit()
    return _to_out(job)


__all__ = ["cancel_job", "enqueue_job", "list_jobs", "router"]
