"""Stage 6 — Training Planner.

Decides *whether* to fine-tune at all, and if so which method, then emits a
reproducible `TrainingPlan` with a cost/VRAM/time estimate for approval.

Planned algorithm
-----------------
1. Gate on evidence: if the baseline already meets the acceptance criteria, or
   the task is better served by prompting/RAG, recommend `no_finetune` /
   `prompt_only` / `rag` and stop — with the reasoning shown.
2. Choose the method from task type, dataset size, model size and hardware:
   `qlora` when VRAM is tight, `lora` when the base fits in memory, `sft` for
   small models where full fine-tuning is affordable.
3. Derive hyperparameters (LR scaled by method/size, epochs by dataset size,
   max_seq_len from the data's token distribution, LoRA targets from the
   architecture) using documented heuristics, not guesses.
4. Estimate VRAM/disk/time from parameter count and token budget; compare
   against `settings.max_vram_gb` / `max_train_minutes`.
5. Render a reproducible training script to the project artifacts and record its
   hash as the config fingerprint.
6. Require explicit user approval, with the estimate and warnings shown.

Exit criteria: an approved `TrainingPlan` whose script and fingerprint are stored.
"""

from __future__ import annotations

from ..models import StageName
from .base import Stage, StageContext, StageResult


class PlannerStage(Stage):
    name = StageName.plan_training
    label = "Plan fine-tuning"
    requires_approval = True
    implemented = False  # Stage 5

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            "PlannerStage: implement method selection, hyperparameter heuristics, "
            "cost estimation and script generation (see docs/ROADMAP.md Stage 5)."
        )
