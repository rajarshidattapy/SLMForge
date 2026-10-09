"""Background worker.

Claims jobs from the queue, dispatches them to the sandbox, streams progress and
records outcomes. Handler implementations arrive with the stages they serve; the
loop, leasing and cancellation plumbing is in place now so stages can be added
without reworking the execution model.
"""

from __future__ import annotations

import socket
import time
import uuid
from collections.abc import Callable

from ..db import SessionLocal
from ..log import get_logger
from ..models import Job, JobKind, RunStatus
from .queue import JobQueue

logger = get_logger("jobs.worker")

JobHandler = Callable[[Job, JobQueue], RunStatus]


class Worker:
    """Single-threaded job worker.

    Run one worker per GPU/CPU budget; scale out horizontally later.
    """

    def __init__(
        self,
        *,
        worker_id: str | None = None,
        poll_interval_s: float = 2.0,
        kinds: list[JobKind] | None = None,
    ) -> None:
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self.poll_interval_s = poll_interval_s
        self.kinds = kinds
        self.handlers: dict[JobKind, JobHandler] = {}
        self._stop = False
        self._current_job_id: str | None = None

    # -- registration ---------------------------------------------------
    def register(self, kind: JobKind, handler: JobHandler) -> None:
        """Register the handler that executes a job kind."""
        self.handlers[kind] = handler

    def register_defaults(self) -> None:
        """Register the built-in handlers.

        Each handler is a stub until its roadmap milestone lands; registering them
        here keeps `worker.run_forever()` usable from day one.
        """
        from . import handlers

        self.register(JobKind.dataset_build, handlers.handle_dataset_build)
        self.register(JobKind.baseline_eval, handlers.handle_baseline_eval)
        self.register(JobKind.training, handlers.handle_training)
        self.register(JobKind.evaluation, handlers.handle_evaluation)
        self.register(JobKind.deployment, handlers.handle_deployment)

    # -- lifecycle ------------------------------------------------------
    def stop(self) -> None:
        """Ask the worker to exit after the current job finishes."""
        logger.info("Stop requested for worker %s", self.worker_id)
        self._stop = True

    def run_once(self) -> bool:
        """Claim and run a single job. Returns True when a job was processed."""
        session = SessionLocal()
        try:
            queue = JobQueue(session, worker_id=self.worker_id)
            job = queue.claim_next(self.kinds)
            if job is None:
                return False

            self._current_job_id = job.id
            logger.info("Running job %s (%s)", job.id, job.kind.value)
            try:
                handler = self.handlers.get(job.kind)
                if handler is None:
                    status = queue.complete(
                        job,
                        status=RunStatus.failed,
                        error=f"No handler registered for {job.kind.value}",
                    ).status
                else:
                    status = handler(job, queue)
            except Exception as exc:
                logger.exception("Job %s raised", job.id)
                queue.complete(job, status=RunStatus.failed, error=f"{type(exc).__name__}: {exc}")
                status = RunStatus.failed
            finally:
                self._current_job_id = None
            logger.info("Job %s -> %s", job.id, status.value)
            return True
        finally:
            session.close()

    def run_forever(self) -> None:
        """Poll for jobs until `stop()` is called."""
        logger.info("Worker %s started (kinds=%s)", self.worker_id, self.kinds or "all")
        while not self._stop:
            try:
                processed = self.run_once()
            except Exception:
                logger.exception("Worker loop error")
                processed = False
            if not processed:
                time.sleep(self.poll_interval_s)
        logger.info("Worker %s stopped", self.worker_id)
