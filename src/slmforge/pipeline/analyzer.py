"""Stage 1 — Prompt & Task Analyzer.

Turns the user's natural-language request into a validated `MLSpec`, asking
clarifying questions when the request is ambiguous and flagging high-stakes or
unsupported requests.

Planned algorithm
-----------------
1. Normalize the prompt and detect language(s).
2. Call the planner LLM with a strict JSON schema derived from `schemas.MLSpec`
   (structured output / tool-calling; repair the JSON if it is invalid).
3. Post-process: validate `task_type` is supported, canonicalize
   `allowed_labels`, infer `output.format`, and refuse to invent metrics.
4. Detect high-stakes domains (medical/legal/financial/employment) and set
   `is_high_stakes`, adding the corresponding safeguards note.
5. If confidence is low or required slots are missing, populate
   `open_questions` and set `readiness = needs_clarification`.
6. Attach a first-pass resource estimate and `recommended_approach`.

Exit criteria: a valid `MLSpec` is persisted on the project and the user can
edit it in the UI before any data or compute is spent.
"""

from __future__ import annotations

from ..models import StageName
from .base import Stage, StageContext, StageResult


class AnalyzerStage(Stage):
    name = StageName.analyze
    label = "Analyze requirement"
    implemented = False  # Stage 1

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            "AnalyzerStage: implement prompt -> MLSpec via the planner LLM "
            "(see docs/ROADMAP.md Stage 1)."
        )
