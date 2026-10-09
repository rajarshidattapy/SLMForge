"""Stage 5 — Baseline Evaluation.

Measures candidate models *before* training so we can prove whether fine-tuning
actually helps. Runs inside the sandbox, never in the API process.

Planned algorithm
-----------------
1. Materialize a fixed evaluation subset from the held-out splits.
2. For each shortlisted candidate, run inference under a controlled prompt
   (zero-shot and a small few-shot variant) via the sandbox.
3. Parse and validate outputs against the spec's output schema; count
   schema violations and invalid-output rate.
4. Compute task metrics on the test split, plus latency percentiles and VRAM.
5. Persist an `EvalRun` labeled `baseline` and surface it for comparison.

Exit criteria: a reproducible baseline `EvalRun` per shortlisted model with the
exact prompt, decoding params and seed recorded.
"""

from __future__ import annotations

from ..models import StageName
from .base import Stage, StageContext, StageResult


class BaselineStage(Stage):
    name = StageName.baseline
    label = "Establish baseline"
    implemented = False  # Stage 4

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            "BaselineStage: implement sandboxed baseline inference and metrics "
            "(see docs/ROADMAP.md Stage 4)."
        )
