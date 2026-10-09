"""Stage 8 — Evaluation & Comparison.

Evaluates the fine-tuned model on data it never saw and compares it with the
baseline, including regressions and "fine-tuning didn't help" verdicts.

Planned algorithm
-----------------
1. Load the adapter/merged model and run the *same* evaluation harness and
   prompt as the baseline (same subset, decoding params, seed).
2. Compute task metrics + valid-output/schema-violation rates.
3. Measure latency percentiles, peak VRAM and a cost-per-1k-calls estimate.
4. Build a per-class confusion table and an error-analysis sample set.
5. Compare against the baseline: delta, % change, and a significance test
   (e.g. bootstrap CI / McNemar) so small-sample noise isn't sold as a gain.
6. Emit a verdict: `accept_finetuned`, `keep_baseline`, `inconclusive`, or
   `needs_more_data`, with limitations and recommendations.

Exit criteria: an `EvalReport` that states plainly whether the trained model is
better, worse, or indistinguishable — with the evidence attached.
"""

from __future__ import annotations

from ..models import StageName
from .base import Stage, StageContext, StageResult


class EvaluatorStage(Stage):
    name = StageName.evaluate
    label = "Evaluate & compare"
    implemented = False  # Stage 7

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            "EvaluatorStage: implement baseline-vs-finetuned comparison, "
            "significance testing and verdicts (see docs/ROADMAP.md Stage 7)."
        )
