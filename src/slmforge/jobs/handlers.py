"""Job handlers.

A handler bridges a queue job to a pipeline stage: it loads the project, builds a
`StageContext`, dispatches work into the sandbox via the stage, and reports the
outcome back to the queue. The heavy stages (training, evaluation) are
implemented in later milestones; the wiring is here now.
"""

from __future__ import annotations

from ..log import get_logger
from ..models import Job, RunStatus
from ..pipeline import get_stage
from ..pipeline.orchestrator import Orchestrator
from ..sandbox import get_sandbox
from .queue import JobQueue

logger = get_logger("jobs.handlers")


def _run_stage_for_job(job: Job, queue: JobQueue, stage_name_value: str) -> RunStatus:
    """Shared driver: run a pipeline stage for the job's project and report status.

    Approved stage parameters travel in ``job.resources["params"]``. The rest of that
    dict is the resource *request* (cpu/memory), which is why the parameters are
    nested rather than mixed in with it.

    There is deliberately **one** session, the queue's own. Opening a second one and
    completing the job through it silently does nothing — the `job` instance belongs
    to the queue's session — which leaves a finished job looking `running` forever.
    """
    from ..models import Project, StageName

    stage_name = StageName(stage_name_value)
    params = (job.resources or {}).get("params") or None
    session = queue.session

    project = session.get(Project, job.project_id)
    if project is None:
        queue.complete(job, status=RunStatus.failed, error=f"Project {job.project_id} not found")
        return RunStatus.failed

    orchestrator = Orchestrator(session)
    stage = get_stage(stage_name)
    queue.heartbeat(job)

    result = orchestrator.run_stage(project, stage, params=params)

    if result.status != "succeeded":
        queue.complete(job, status=RunStatus.failed, error=result.error or result.summary)
        return RunStatus.failed

    if result.needs_approval:
        # The stage ran and stopped at a human gate. The job itself is done, so it is
        # not a failure — but saying nothing would hide why nothing advanced.
        logger.warning(
            "Job %s stopped at an approval gate for project %s: %s",
            job.id,
            job.project_id[:8],
            ", ".join(result.blocked_on) or "unspecified",
        )
        queue.complete(job, status=RunStatus.succeeded, exit_code=0, usage=result.metrics or None)
        return RunStatus.succeeded

    queue.complete(
        job,
        status=RunStatus.succeeded,
        exit_code=0,
        usage=result.metrics or None,
        logs_path=result.logs_path,
    )
    return RunStatus.succeeded


def handle_dataset_build(job: Job, queue: JobQueue) -> RunStatus:
    """Materialize the approved dataset into split files."""
    return _run_stage_for_job(job, queue, "curate_data")


def handle_baseline_eval(job: Job, queue: JobQueue) -> RunStatus:
    """Evaluate candidate base models to establish a baseline."""
    return _run_stage_for_job(job, queue, "baseline")


def handle_training(job: Job, queue: JobQueue) -> RunStatus:
    """Run the approved training script in the sandbox."""
    return _run_stage_for_job(job, queue, "train")


def handle_evaluation(job: Job, queue: JobQueue) -> RunStatus:
    """Evaluate the fine-tuned model and compare it with the baseline."""
    return _run_stage_for_job(job, queue, "evaluate")


def handle_deployment(job: Job, queue: JobQueue) -> RunStatus:
    """Publish the selected model and provision an endpoint."""
    return _run_stage_for_job(job, queue, "deploy")


def sandbox_info() -> dict[str, object]:
    """Report the resolved sandbox backend (used by /health for diagnostics)."""
    try:
        sandbox = get_sandbox()
        return {"backend": sandbox.backend}
    except NotImplementedError as exc:
        return {"backend": "unavailable", "error": str(exc)}
