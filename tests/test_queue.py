"""Job queue semantics: claiming, completion, cancellation and stale leases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from slmforge.jobs.queue import JobQueue
from slmforge.models import Job, JobKind, Project, RunStatus


def _project(session: Session, name: str = "p") -> Project:
    project = Project(name=name, prompt="build a classifier")
    session.add(project)
    session.commit()
    return project


def test_enqueue_then_claim_marks_running(session: Session) -> None:
    project = _project(session)
    queue = JobQueue(session, worker_id="w1")
    job = queue.enqueue(project.id, JobKind.training)
    assert job.status is RunStatus.pending

    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status is RunStatus.running
    assert claimed.claimed_by == "w1"
    assert claimed.attempts == 1

    # The queue is now empty.
    assert queue.claim_next() is None


def test_claim_respects_kind_filter(session: Session) -> None:
    project = _project(session)
    queue = JobQueue(session)
    queue.enqueue(project.id, JobKind.training)
    eval_job = queue.enqueue(project.id, JobKind.evaluation)

    claimed = queue.claim_next([JobKind.evaluation])
    assert claimed is not None and claimed.id == eval_job.id
    # The training job is still claimable.
    assert queue.claim_next([JobKind.training]) is not None


def test_complete_records_outcome(session: Session) -> None:
    project = _project(session)
    queue = JobQueue(session)
    job = queue.enqueue(project.id, JobKind.dataset_build)
    claimed = queue.claim_next()
    assert claimed is not None
    queue.complete(claimed, status=RunStatus.succeeded, exit_code=0, usage={"rows": 10})

    refreshed = session.get(Job, job.id)
    assert refreshed is not None
    assert refreshed.status is RunStatus.succeeded
    assert refreshed.exit_code == 0
    assert refreshed.usage == {"rows": 10}


def test_failure_records_the_error(session: Session) -> None:
    project = _project(session)
    queue = JobQueue(session)
    job = queue.enqueue(project.id, JobKind.training)
    claimed = queue.claim_next()
    assert claimed is not None
    queue.complete(claimed, status=RunStatus.failed, exit_code=1, error="CUDA OOM")

    refreshed = session.get(Job, job.id)
    assert refreshed is not None
    assert refreshed.status is RunStatus.failed
    assert refreshed.error == "CUDA OOM"


def test_cancel_pending_job(session: Session) -> None:
    project = _project(session)
    queue = JobQueue(session)
    job = queue.enqueue(project.id, JobKind.training)

    assert queue.request_cancel(job.id) is True
    refreshed = session.get(Job, job.id)
    assert refreshed is not None and refreshed.status is RunStatus.cancelled
    assert queue.claim_next() is None


def test_requeue_until_attempts_exhausted(session: Session) -> None:
    project = _project(session)
    queue = JobQueue(session)
    job = queue.enqueue(project.id, JobKind.training, max_attempts=2)

    first = queue.claim_next()
    assert first is not None
    assert queue.requeue(first, "transient failure") is True  # attempt 1 of 2 -> requeue

    second = queue.claim_next()
    assert second is not None
    assert queue.requeue(second, "still failing") is False  # attempts exhausted -> failed
    assert session.get(Job, job.id).status is RunStatus.failed


def test_stale_lease_is_reclaimed(session: Session) -> None:
    """A worker that stops heartbeating must not strand its job."""
    project = _project(session)
    queue = JobQueue(session, worker_id="dead-worker")
    job = queue.enqueue(project.id, JobKind.training, max_attempts=2)
    claimed = queue.claim_next()
    assert claimed is not None

    claimed.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
    session.commit()

    # A new worker reclaims the abandoned job.
    other = JobQueue(session, worker_id="live-worker").claim_next()
    assert other is not None
    assert other.id == job.id
    assert other.claimed_by == "live-worker"
