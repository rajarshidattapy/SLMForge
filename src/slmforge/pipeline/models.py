"""Stage 4 — Model Discovery & Compatibility.

Builds a ranked shortlist of candidate base models from the local Ollama
inventory and the Hugging Face Hub, with license and hardware constraints made
explicit.

Planned algorithm
-----------------
1. Enumerate locally available Ollama models (`/api/tags`) and check the HF cache.
2. Query the HF Hub for small instruction-tuned models matching the task family,
   filtering by parameter budget derived from `settings.max_vram_gb`.
3. Normalize metadata: parameter count, context window, license, quantization,
   finetune support (sft/lora/qlora via peft/trl).
4. Rank with a transparent rubric (task fit, size/VRAM fit, license clearance,
   deployability) and attach `rank_rationale` — never a bare score.
5. Flag license-restricted or unverifiable candidates as warnings.
6. Persist a `ModelSelection` requiring explicit user approval of the choice.

Exit criteria: a user-approved `ModelCandidate` plus the considered alternatives.
"""

from __future__ import annotations

from ..models import StageName
from .base import Stage, StageContext, StageResult


class ModelSelectionStage(Stage):
    name = StageName.select_model
    label = "Select base model"
    requires_approval = True
    implemented = False  # Stage 3

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            "ModelSelectionStage: implement Ollama + HF discovery, compatibility "
            "checks and ranking (see docs/ROADMAP.md Stage 3)."
        )
