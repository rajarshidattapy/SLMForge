"""Stage 1 — Prompt & Task Analyzer.

Turns the user's natural-language request into a validated `MLSpec`, asking
clarifying questions when the request is ambiguous and flagging high-stakes or
unsupported requests.

How it works
------------
Two paths, in order:

1. **Planner LLM** (`integrations/llm.py`) — an OpenAI-compatible endpoint (default:
   an OpenRouter free model) is asked for JSON, parsed with
   `parse_json_object`, then validated into an `MLSpec`. Validation failure is not
   fatal: we fall back rather than showing the user a broken spec.
2. **Deterministic heuristic** — pattern matching on the prompt. Weaker, but it
   needs no API key, no network, and it is what makes the demo work offline.

Whatever produced the spec is recorded in `generated_by`, so the UI can say
"honestly derived by a heuristic" rather than implying a model reasoned about
your request. The two paths are never blended into one answer.

Guardrails applied on the way out:

- **High-stakes detection** (medical/legal/financial/employment) sets
  `is_high_stakes` and adds the corresponding risk note — never silently treated
  as an ordinary build.
- **Support gate**: only `classification`, `structured_generation` and
  `extraction` are supported end to end today. Anything else returns
  `readiness="unsupported"` with the reason in `out_of_scope`, instead of
  producing a spec the pipeline cannot honour.
- **Ambiguity**: missing labels, input text or success criteria become
  `open_questions` with `readiness="needs_clarification"` rather than guesses.

Exit criteria: a valid `MLSpec` is persisted on the project and the user can edit
it in the UI before any data or compute is spent.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from ..config import settings
from ..integrations.llm import PlannerLLM, PlannerLLMUnavailable, parse_json_object
from ..log import get_logger
from ..models import StageName
from ..schemas import (
    ClarifyingQuestion,
    EvaluationPlan,
    FieldSpec,
    MLSpec,
    OutputFormat,
    OutputSpec,
    TaskType,
)
from .base import Stage, StageContext, StageResult

logger = get_logger("pipeline.analyzer")

#: Tasks the MVP can actually drive end to end (see docs/ROADMAP.md).
SUPPORTED_TASKS = {TaskType.classification, TaskType.structured_generation, TaskType.extraction}

HIGH_STAKES_PATTERNS: dict[str, list[str]] = {
    "medical": ["medical", "diagnos", "patient", "clinical", "symptom", "prescrib", "dosage"],
    "legal": ["legal", "lawsuit", "court", "contract law", "attorney", "defendant"],
    "financial": [
        "financial advice",
        "investment",
        "loan approval",
        "loan deci",
        "credit deci",
        "tax advice",
    ],
    "employment": [
        # "candidate" is deliberately absent: it appears in ordinary prompts about
        # candidate models and would produce false positives.
        "hiring",
        "hired",
        "resume",
        "résumé",
        "cv screen",
        "applicant",
        "shortlist",
        "termination",
        "firing",
        "promotion decision",
    ],
}

CREATIVE_PATTERN = re.compile(
    r"\b(poem|poetry|novel|screenplay|song lyrics|short story|write a story)\b", re.I
)
MEDIA_PATTERN = re.compile(
    r"\b(text-to-image|image generation|generate images|video generation|"
    r"text-to-speech|speech-to-text|music generation|transcribe audio)\b",
    re.I,
)
SPECIALISED_TASKS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(summariz\w*|summaris\w*|tl;?dr)\b", re.I),
        "Summarization is recognised but not yet supported by the MVP.",
    ),
    (
        re.compile(r"\b(translat\w*)\b", re.I),
        "Translation is recognised but not yet supported by the MVP.",
    ),
    (
        re.compile(r"\b(website|web app|landing page|build an app)\b", re.I),
        "Building an application is not a model task; SLMForge builds models.",
    ),
]
LABEL_HINT_PATTERN = re.compile(r"\b(classif|categor|assign|label|routing|routes|bucket)\w*", re.I)
JSON_HINT_PATTERN = re.compile(r"\b(json|schema|structured output|valid object)\b", re.I)
EXTRACT_HINT_PATTERN = re.compile(r"\b(extract|parse out|pull out|structured record)\b", re.I)

#: Labels usually appear as a comma-separated quoted list near the end of a prompt.
QUOTED_TERM = re.compile(r"[\"'\u201c\u2018]([^\"'\u2019\u201d]{2,48})[\"'\u201d\u2019]")


class AnalyzerStage(Stage):
    name = StageName.analyze
    label = "Analyze requirement"
    implemented = True  # Stage 1

    def run(self, ctx: StageContext) -> StageResult:
        prompt = (ctx.prompt or "").strip()
        if not prompt:
            return StageResult(status="failed", error="The requirement is empty.")

        source, spec, note = self._produce_spec(prompt)

        # Validate / gate what we got, regardless of which path produced it.
        _gate(spec, prompt)
        # Record honestly in the spec itself so the dashboard can show provenance.
        spec.generated_by = source

        payload = spec.model_dump(mode="json", by_alias=True)
        logger.info(
            "Analyzed requirement into task_type=%s readiness=%s via %s",
            spec.task_type.value,
            spec.readiness,
            source,
        )
        return StageResult(
            output=payload,
            summary=f"Specified as {spec.task_type.value} ({spec.readiness}) via {source}",
            metrics={"open_questions": float(len(spec.open_questions))},
            artifacts=[
                {
                    "kind": "spec",
                    "name": "mlspec.json",
                    "path": "spec/mlspec.json",
                    "meta": {"generated_by": source, "note": note},
                }
            ],
        )

    # -- spec production -------------------------------------------------
    def _produce_spec(self, prompt: str) -> tuple[str, MLSpec, str | None]:
        """Return (source label, spec, note). Never raises for network reasons."""
        if settings.llm_provider == "none" or not settings.llm_api_key:
            return "heuristic", _heuristic_spec(prompt), _NO_KEY_NOTE

        try:
            raw = asyncio.run(_ask_planner(prompt))
        except PlannerLLMUnavailable as exc:
            logger.warning("Planner LLM unavailable, using heuristic: %s", exc)
            return "heuristic", _heuristic_spec(prompt), f"Planner LLM unavailable: {exc}"

        try:
            spec = _spec_from_llm(raw, prompt)
        except Exception as exc:
            logger.warning("Planner LLM returned unusable spec (%s); using heuristic", exc)
            return "heuristic", _heuristic_spec(prompt), f"LLM output rejected: {exc}"
        return settings.llm_model, spec, None


_NO_KEY_NOTE = (
    "Derived by the deterministic heuristic because no planner LLM was configured. "
    "Review it before relying on it."
)


async def _ask_planner(prompt: str) -> str:
    """Ask the planner LLM for an MLSpec as JSON."""
    async with PlannerLLM() as llm:
        return await llm.chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(prompt)},
            ],
            temperature=0.0,
            max_tokens=2000,
        )


_SYSTEM_PROMPT = """You turn a plain-language requirement for an AI model into a
machine-learning specification. Respond with ONLY a JSON object (no prose, no
markdown fences) using these keys:

- title: short name
- summary: one sentence
- task_type: one of classification | structured_generation | extraction | summarization |
  rewriting | qa_grounded | embedding | unsupported
- input_description: what the input looks like
- allowed_labels: string array. Use exactly the labels the user names. If they name
  none but the task is classification, propose labels and say so in "assumptions".
- output.format: json_object | label | span | free_text | multi_label
- evaluation.primary_metric: macro_f1 | exact_match | accuracy | rouge | f1
- evaluation.secondary_metrics: string array
- language: string array (default ["en"])
- domain: short domain name or null
- assumptions: string array — anything you inferred rather than were told
- open_questions: array of {id, question, why, blocking}. **Set blocking=true ONLY
  when the answer changes the input/output contract** (unknown labels, unknown output
  format, unknown input field). Questions about training data, model choice, cost or
  hosting must have blocking=false: later stages answer those automatically, so they
  must not stall the user here.
- out_of_scope: string array — things the request implies but does not state
- risk_notes: string array
- is_high_stakes: boolean (medical, legal, financial, employment decisions)

Never invent labels the user did not name. Never claim a metric you were not given
unless the task type implies it."""


def _user_prompt(prompt: str) -> str:
    return (
        "Produce the ML specification for this requirement:\n\n"
        f"{prompt}\n\n"
        "Return only the JSON object."
    )


def _spec_from_llm(raw: str, prompt: str) -> MLSpec:
    """Parse and validate LLM output into an MLSpec, dropping unusable fields."""
    data = parse_json_object(raw)
    # Pydantic rejects unknown/extra keys for the fields we care about; strip
    # anything not in the model so a chatty model does not fail the whole spec.
    allowed = set(MLSpec.model_fields)
    cleaned = {k: v for k, v in data.items() if k in allowed}
    if "task_type" in cleaned:
        cleaned["task_type"] = _normalize_task(cleaned["task_type"])
    if "output" in cleaned and isinstance(cleaned["output"], dict):
        fmt = cleaned["output"].get("format")
        cleaned["output"]["format"] = _normalize_output_format(fmt)
    if "evaluation" in cleaned and isinstance(cleaned["evaluation"], dict):
        cleaned["evaluation"] = {k: v for k, v in cleaned["evaluation"].items() if v is not None}
    if "open_questions" in cleaned:
        cleaned["open_questions"] = [
            q for q in cleaned["open_questions"] if isinstance(q, dict) and q.get("question")
        ]
    spec = MLSpec.model_validate(cleaned)
    if not spec.input_description:
        spec.input_description = "Unspecified input"
    return spec


def _normalize_task(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "text_classification": "classification",
        "classification_task": "classification",
        "structured_output": "structured_generation",
        "struct": "structured_generation",
        "info_extraction": "extraction",
        "information_extraction": "extraction",
        "none": "unsupported",
    }
    text = aliases.get(text, text)
    return text if text in {t.value for t in TaskType} else "unsupported"


def _normalize_output_format(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "json": "json_object",
        "json_object": "json_object",
        "label": "label",
        "text": "free_text",
        "free_text": "free_text",
        "span": "span",
        "multi_label": "multi_label",
    }
    return aliases.get(text, "free_text")


# -- deterministic fallback -------------------------------------------------


def _heuristic_spec(prompt: str) -> MLSpec:
    """Derive a conservative spec from the prompt with pattern matching only."""
    task = _infer_task(prompt)
    labels = _infer_labels(prompt)
    fmt = (
        OutputFormat.json_object
        if JSON_HINT_PATTERN.search(prompt) or task is TaskType.structured_generation
        else OutputFormat.label
        if task is TaskType.classification
        else OutputFormat.free_text
    )
    metric = {
        TaskType.classification: "macro_f1",
        TaskType.extraction: "exact_match",
        TaskType.structured_generation: "exact_match",
    }.get(task, "accuracy")

    return MLSpec(
        title=_first_line(prompt, 70),
        summary=prompt.splitlines()[0][:200],
        task_type=task,
        input_description=_infer_input(prompt),
        allowed_labels=labels,
        output=OutputSpec(format=fmt, fields=[]),
        evaluation=_evaluation_plan(metric),
        language=_infer_languages(prompt),
        assumptions=_infer_assumptions(task, labels, prompt),
        risk_notes=_risk_notes(prompt),
        generated_by=None,
    )


def _infer_task(prompt: str) -> TaskType:
    for pattern, _reason in SPECIALISED_TASKS:
        if pattern.search(prompt):
            return TaskType.unsupported
    if CREATIVE_PATTERN.search(prompt) or MEDIA_PATTERN.search(prompt):
        return TaskType.unsupported
    if EXTRACT_HINT_PATTERN.search(prompt):
        return TaskType.extraction
    if LABEL_HINT_PATTERN.search(prompt):
        return TaskType.classification
    if JSON_HINT_PATTERN.search(prompt):
        return TaskType.structured_generation
    return TaskType.unsupported


def _infer_labels(prompt: str) -> list[str]:
    """Pull the user's own labels out of the prompt, preserving their order.

    Two shapes are recognised, quoted first because it is unambiguous:

    1. Quoted terms: `into "Billing", "Technical"`.
    2. An unquoted enumeration introduced by a cue word: `into Billing, Technical,
       Account Access or General and return JSON`.

    The second shape is how people actually write the requirement the product is
    built for, and missing it would send every such request down the
    "no labels" path. Nothing is inferred beyond what the user named: an ambiguous
    phrase such as "one of four queues" yields no labels, and the analyzer asks
    instead of inventing them.
    """
    seen: list[str] = []
    for match in QUOTED_TERM.finditer(prompt):
        label = match.group(1).strip()
        if not label or label.lower() in {"json", "schema", "valid json object"}:
            continue
        if label not in seen:
            seen.append(label)
    if len(seen) >= 2:
        return seen
    return _enumerated_labels(prompt)


#: Words that introduce a list of categories rather than a further instruction.
LABEL_CUE = re.compile(
    r"\b(?:into|among|between|as|(?:the\s+)?(?:labels?|classes|categories|options|queues|buckets)\s*[:=]?)\b",
    re.IGNORECASE,
)
#: Separators inside such a list.
LABEL_SPLIT = re.compile(r"\s*(?:,|;|/|\bor\b|\band\b)\s*", re.IGNORECASE)
#: The list ends where the next instruction begins — these are not labels.
CLAUSE_STARTS = (
    "return",
    "returns",
    "output",
    "respond",
    "reply",
    "give",
    "provide",
    "include",
    "print",
    "format",
    "use",
    "make",
    "ensure",
    "with",
    "that",
    "which",
    "then",
    "please",
    "and it",
)
MAX_LABEL_WORDS = 4
#: Bare nouns that introduce a list rather than being part of it.
CUE_WORDS = frozenset(
    {
        "labels",
        "label",
        "classes",
        "class",
        "categories",
        "category",
        "options",
        "queues",
        "buckets",
    }
)


def _enumerated_labels(prompt: str) -> list[str]:
    """Labels from an unquoted list, or `[]` when the prompt does not name any."""
    sentence = prompt.splitlines()[0]
    for cue in LABEL_CUE.finditer(sentence):
        tail = sentence[cue.end() :]
        tail = re.split(r"[.!?]", tail, maxsplit=1)[0]
        if not tail.strip():
            continue
        if tail.lstrip()[:1] in {'"', "'", "\u201c", "\u2018"}:
            continue  # quoting is handled with more confidence elsewhere
        # A nested cue ("into the categories: urgent, normal") must not become part of
        # the first label, so cut at a colon that sits before the first separator.
        head, comma, rest = tail.partition(",")
        if ":" in head:
            tail = head.split(":")[-1] + comma + rest
        candidates: list[str] = []
        for chunk in LABEL_SPLIT.split(tail):
            label = chunk.strip().strip("\"'\u201c\u201d\u2018\u2019").strip()
            if not label:
                continue
            lowered = label.lower()
            if lowered in {"json", "schema", "valid json object"}:
                continue
            if lowered.startswith(CLAUSE_STARTS) or len(label.split()) > MAX_LABEL_WORDS:
                break
            if not candidates and lowered.removeprefix("the ") in CUE_WORDS:
                continue  # a bare cue word introduces the list, it is not in it
            candidates.append(label)
        if len(candidates) >= 2:
            return list(dict.fromkeys(candidates))
    return []


def _infer_input(prompt: str) -> str:
    lowered = prompt.lower()
    if "email" in lowered:
        return "Customer support email text"
    if "ticket" in lowered or "message" in lowered:
        return "Customer support ticket text"
    if "document" in lowered or "article" in lowered:
        return "Document text"
    if "code" in lowered:
        return "Source code"
    return "Text input"


def _infer_languages(prompt: str) -> list[str]:
    if re.search(r"\b(german|deutsch)\b", prompt, re.I) and re.search(
        r"\b(english|ingles)\b", prompt, re.I
    ):
        return ["en", "de"]
    if re.search(r"\bgerman\b", prompt, re.I):
        return ["de"]
    return ["en"]


def _evaluation_plan(primary: str) -> EvaluationPlan:
    secondary = ["accuracy", "precision", "recall"] if primary == "macro_f1" else ["f1"]
    return EvaluationPlan(
        primary_metric=primary,
        secondary_metrics=secondary,
        splits=["validation", "test"],
        test_fraction=0.2,
        validation_fraction=0.1,
    )


def _infer_assumptions(task: TaskType, labels: list[str], prompt: str) -> list[str]:
    notes = ["Derived by rule-based heuristic; review before relying on it."]
    if task is TaskType.classification and not labels:
        notes.append(
            "No explicit labels were named; allowed_labels is empty and must be filled in."
        )
    if task is TaskType.unsupported:
        notes.append("Task type could not be determined as a supported ML task.")
    if JSON_HINT_PATTERN.search(prompt):
        notes.append("A JSON output format was requested.")
    return notes


def _risk_notes(prompt: str) -> list[str]:
    return detect_high_stakes(prompt)


def detect_high_stakes(text: str) -> list[str]:
    """Return a note per high-stakes domain the request touches, else []."""
    lowered = text.lower()
    hits = [
        f"Touches {domain}; a wrong prediction has real-world consequences for a person."
        for domain, words in HIGH_STAKES_PATTERNS.items()
        if any(word in lowered for word in words)
    ]
    return hits


def _first_line(text: str, limit: int) -> str:
    first = text.strip().splitlines()[0]
    return first[:limit].rstrip() if len(first) > limit else first


# -- gating -----------------------------------------------------------------


def _gate(spec: MLSpec, prompt: str) -> None:
    """Apply support, ambiguity and high-stakes rules in place.

    Mutating here (rather than branching in `run`) keeps the same rules applied to
    both the LLM and heuristic paths — the fallback must not be able to slip
    through an unsupported task unnoticed.
    """
    spec.is_high_stakes = bool(detect_high_stakes(prompt))
    if spec.is_high_stakes and not spec.risk_notes:
        spec.risk_notes = detect_high_stakes(prompt)

    if spec.task_type is TaskType.unsupported:
        spec.readiness = "unsupported"
        spec.out_of_scope = [*_why_unsupported(prompt), *spec.out_of_scope]
        return

    if spec.task_type not in SUPPORTED_TASKS:
        spec.readiness = "unsupported"
        reason = (
            f"Task type '{spec.task_type.value}' is recognised but not supported by the MVP yet."
        )
        spec.out_of_scope = [reason, *spec.out_of_scope]
        return

    if not spec.allowed_labels and spec.task_type is TaskType.classification:
        spec.open_questions = [
            ClarifyingQuestion(
                id="labels",
                question="Which exact labels may the model choose from?",
                why="Classification needs a closed label set; none was named.",
                options=[],
                blocking=True,
            )
        ]
        spec.readiness = "needs_clarification"
        return

    _resolve_output_schema_questions(spec)

    if not spec.open_questions:
        spec.readiness = "ready_for_review"
    else:
        spec.readiness = (
            "needs_clarification"
            if any(q.blocking for q in spec.open_questions)
            else "ready_for_review"
        )


#: The platform's canonical single-label output shape. Stage 2 writes training
#: records in exactly this form and the later stages read it back, so it is a
#: decided convention rather than an open question.
CANONICAL_LABEL_JSON = '{"label": "<one of the allowed labels>"}'
OUTPUT_SHAPE_QUESTION = re.compile(
    r"\b(json|schema|field|key|structure|shape|format|naming)\b", re.IGNORECASE
)


def _resolve_output_schema_questions(spec: MLSpec) -> None:
    """Answer "what should the JSON look like?" with the canonical shape.

    A single-label classification spec with JSON output already has a decided
    contract, and the platform is the authority on it. Making the user re-type the
    field name would not be more transparent — the assumption is recorded and shown
    next to the spec either way — it would only stop a runnable build for a question
    the product can answer. Questions that go beyond naming (extra fields, confidence
    scores, multi-label) still block, because those really do change the contract.
    """
    if spec.task_type is not TaskType.classification:
        return
    if spec.output.format is not OutputFormat.json_object:
        return
    if len(spec.allowed_labels) < 2:
        return

    answered = False
    for question in spec.open_questions:
        if not question.blocking:
            continue
        if not OUTPUT_SHAPE_QUESTION.search(f"{question.question} {question.why or ''}"):
            continue
        question.blocking = False
        question.default = CANONICAL_LABEL_JSON
        question.why = (
            f"{question.why or ''} Defaulted to the platform's canonical single-label "
            "output, which the dataset, training and evaluation stages all use."
        ).strip()
        answered = True

    if not answered:
        return

    assumption = (
        f'Output is a JSON object with the single key "label" ({CANONICAL_LABEL_JSON}); '
        "no confidence score or extra fields. Edit the spec to change this."
    )
    if assumption not in spec.assumptions:
        spec.assumptions.append(assumption)
    if not spec.output.fields:
        spec.output.fields = [
            FieldSpec(
                name="label",
                type="enum",
                description="The predicted category",
                enum=list(spec.allowed_labels),
            )
        ]


def _why_unsupported(prompt: str) -> list[str]:
    lowered = prompt.lower()
    if CREATIVE_PATTERN.search(prompt) or MEDIA_PATTERN.search(prompt):
        return ["Creative or media generation is outside SLMForge's scope."]
    for pattern, reason in SPECIALISED_TASKS:
        if pattern.search(lowered):
            return [reason]
    return ["The request did not map onto a supported machine-learning task."]
