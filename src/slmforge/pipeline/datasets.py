"""Stage 3 — Dataset Discovery & Curation.

Produces a `DatasetPlan` for review and, once approved, a materialized
`DatasetCard` with train/validation/test JSONL splits and full provenance.

Planned algorithm
-----------------
1. Derive search terms from the `MLSpec` and query the HF datasets server and a
   curated catalog; score candidates for task/label/domain fit with a rationale.
2. Include user uploads (CSV/JSONL) and flag license/commercial constraints.
3. On approval, download and normalize to a canonical record shape
   (`{"input": ..., "output": ...}` or chat messages).
4. Validate: missing labels, schema mismatch, too-short text, suspected PII.
5. Deduplicate (normalized exact match, then MinHash/embedding near-dup).
6. Map raw labels onto `allowed_labels`; drop unmappable examples with reasons.
7. Split with a fixed seed; verify no near-duplicate leakage across splits.
8. If data is insufficient, generate synthetic examples (labeled as synthetic,
   never counted as ground truth) or stop and ask the user.

Exit criteria: split files + a `DatasetCard` whose sources, licenses, synthetic
fraction and quality issues are all recorded.
"""

from __future__ import annotations

from ..models import StageName
from .base import Stage, StageContext, StageResult


class DatasetStage(Stage):
    name = StageName.curate_data
    label = "Curate dataset"
    implemented = False  # Stage 2

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            "DatasetStage: implement discovery, validation, dedupe, label mapping "
            "and splitting (see docs/ROADMAP.md Stage 2)."
        )
