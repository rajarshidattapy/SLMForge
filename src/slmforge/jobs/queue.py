"""Durable job queue.

A single SQLite table doubles as the queue. Workers claim jobs with an
optimistic status transition and keep a lease via heartbeats, so a crashed
worker's job becomes reclaimable instead of being stuck forever. This is enough
for the prototype; a Redis/Postgres queue can replace it behind the same API.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..log import get_logger
from ..models import Job, JobKind, RunStatus

logger = get_logger("jobs.queue")

#: A job whose lease is older than this is considered abandoned.
LEASE_TIMEOUT = timedelta(minutes=15)


def _now() -> datetime:
    return datetime.now(UTC)


class JobQueue:
    """Enqueue and claim jobs against a session."""

    def __init__(self, session: Session, worker_id: str = "worker") -> None:
        self.session = session
        self.worker_id = worker_id

    def enqueue(
        self,
        project_id: str,
        kind: JobKind,
        *,
        stage_run_id: str | None = None,
        resources: dict[str, Any] | None = None,
        max_attempts: int = 1,
    ) -> Job:
        job = Job(
            project_id=project_id,
            stage_run_id=stage_run_id,
            kind=kind,
            status=RunStatus.pending,
            resources=resources,
            max_attempts=max_attempts,
        )
        self.session.add(job)
        self.session.commit()
        logger.info("Enqueued %s for project %s (job %s)", kind.value, project_id[:8], job.id)
        return job

    def claim_next(self, kinds: list[JobKind] | None = None) -> Job | None:
        """Atomically claim the oldest pending (or expired) job, or return None."""
        self._reap_expired()
        stmt = select(Job).where(Job.status == RunStatus.pending)
        if kinds:
            stmt = stmt.where(Job.kind.in_(kinds))
        stmt = stmt.order_by(Job.created_at.asc()).limit(1)

        job = self.session.execute(stmt).scalar_one_or_none()
        if job is None:
            return None

        job.status = RunStatus.running
        job.claimed_by = self.worker_id
        job.claimed_at = _now()
        job.heartbeat_at = _now()
        job.attempts += 1
        self.session.commit()
        return job

    def heartbeat(self, job: Job) -> None:
        """Extend the lease on a running job."""
        job.heartbeat_at = _now()
        self.session.commit()

    def complete(
        self,
        job: Job,
        *,
        status: RunStatus = RunStatus.succeeded,
        exit_code: int | None = None,
        usage: dict[str, Any] | None = None,
        logs_path: str | None = None,
        error: str | None = None,
    ) -> Job:
        job.status = status
        job.exit_code = exit_code
        job.usage = usage
        job.logs_path = logs_path
        job.error = error
        self.session.commit()
        return job

    def request_cancel(self, job_id: str) -> bool:
        """Mark a pending job cancelled; running jobs are stopped by the worker."""
        job = self.session.get(Job, job_id)
        if job is None:
            return False
        if job.status == RunStatus.pending:
            job.status = RunStatus.cancelled
            self.session.commit()
        return True

    def requeue(self, job: Job, reason: str) -> bool:
        """Return a job to the queue if it has attempts left."""
        if job.attempts >= job.max_attempts:
            self.complete(job, status=RunStatus.failed, error=reason)
            return False
        job.status = RunStatus.pending
        job.claimed_by = None
        job.error = reason
        self.session.commit()
        return True

    def _reap_expired(self) -> None:
        """Requeue or fail running jobs whose worker stopped heartbeating."""
        cutoff = _now() - LEASE_TIMEOUT
        stmt = select(Job).where(
            Job.status == RunStatus.running,
            Job.heartbeat_at.is_not(None),
            Job.heartbeat_at < cutoff,
        )
        for job in self.session.execute(stmt).scalars():
            logger.warning("Reclaiming stale job %s from %s", job.id, job.claimed_by)
            if not self.requeue(job, reason="lease expired"):
                self.session.commit()
