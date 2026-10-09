"""Stage 7 — Training Execution.

Runs the approved training script inside an isolated sandbox, streaming logs and
recording checkpoints, resource use and failures.

Planned algorithm
-----------------
1. Write the generated script + config YAML into the job work dir.
2. Launch it in the sandbox with CPU/memory/time limits and network per policy.
3. Stream stdout/stderr to the job log artifact; parse structured progress lines
   (step, loss, LR, VRAM) for the dashboard.
4. Persist checkpoints and the final adapter; record peak VRAM and wall clock.
5. Support cancellation (SIGTERM then SIGKILL) and resumable retries.
6. Never mark success on a non-zero exit code, even if a checkpoint exists.

Exit criteria: a `TrainingRecord` with an adapter path, loss curve and honest
resource numbers — or a clearly reported failure.
"""

from __future__ import annotations

from ..models import StageName
from .base import Stage, StageContext, StageResult


class TrainerStage(Stage):
    name = StageName.train
    label = "Fine-tune"
    requires_approval = True
    implemented = False  # Stage 6

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            "TrainerStage: implement sandboxed training, log streaming, checkpointing "
            "and cancellation (see docs/ROADMAP.md Stage 6)."
        )
