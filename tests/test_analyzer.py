"""Stage 1 — the analyzer must produce a valid spec, honestly labelled.

Two paths are tested independently, plus the failure modes between them: a
planner LLM that is unavailable, and one that returns garbage. In both failure
cases the stage must still produce a usable spec and admit where it came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slmforge.config import settings
from slmforge.models import Project, ProjectStatus, StageName
from slmforge.pipeline import analyzer as analyzer_module
from slmforge.pipeline.analyzer import AnalyzerStage, detect_high_stakes
from slmforge.pipeline.base import StageContext
from slmforge.schemas import MLSpec

TICKET_PROMPT = (
    "Create a lightweight model that reads customer support tickets and assigns each to "
    '"Billing", "Technical", "Account Access", or "General". It should return valid JSON '
    "and prioritize classification accuracy."
)


def _ctx(prompt: str, tmp_path: Path) -> StageContext:
    project = Project(id="proj1", name="p", prompt=prompt, status=ProjectStatus.draft)
    return StageContext(
        project_id="proj1",
        prompt=prompt,
        project=project,
        artifact_dir=tmp_path,
        work_dir=tmp_path,
    )


def _run(prompt: str, tmp_path: Path):
    return AnalyzerStage().run(_ctx(prompt, tmp_path))


@pytest.fixture()
def planner(monkeypatch: pytest.MonkeyPatch):
    """Enable the planner-LLM path with a fake key and model."""
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "llm_api_key", "sk-test-not-a-real-key")
    monkeypatch.setattr(settings, "llm_model", "provider/test-model:free")
    return monkeypatch


# --- heuristic path (the offline default) ---------------------------------


def test_ticket_prompt_yields_a_classification_spec(tmp_path: Path) -> None:
    result = _run(TICKET_PROMPT, tmp_path)
    assert result.status == "succeeded"

    spec = MLSpec.model_validate(result.output)
    assert spec.task_type == "classification"
    assert spec.allowed_labels == [
        "Billing",
        "Technical",
        "Account Access",
        "General",
    ]
    assert spec.output.format == "json_object"
    assert spec.evaluation.primary_metric == "macro_f1"
    assert spec.generated_by == "heuristic"


def test_output_is_a_valid_mlspec_dict(tmp_path: Path) -> None:
    """The orchestrator persists this dict straight onto the project row."""
    result = _run(TICKET_PROMPT, tmp_path)
    again = MLSpec.model_validate(json.loads(json.dumps(result.output)))
    assert again.task_type == "classification"


def test_blocking_uncertainty_is_surfaced_not_guessed(tmp_path: Path) -> None:
    """A classification request with no labels must ask, not invent."""
    prompt = "Classify support tickets and return JSON."
    spec = MLSpec.model_validate(_run(prompt, tmp_path).output)
    assert spec.task_type == "classification"
    assert spec.allowed_labels == []
    assert spec.readiness == "needs_clarification"
    assert any(q.blocking for q in spec.open_questions)
    assert any(q.id == "labels" for q in spec.open_questions)


@pytest.mark.parametrize(
    "prompt",
    [
        "Write a short poem about the sea.",
        "Generate images from a text description.",
        "Summarize long articles into three bullet points.",
        "Translate product descriptions from English to German.",
        "Build a website that lets me post photos.",
    ],
)
def test_unsupported_requests_are_refused_explained(tmp_path: Path, prompt: str) -> None:
    spec = MLSpec.model_validate(_run(prompt, tmp_path).output)
    assert spec.readiness == "unsupported"
    assert spec.out_of_scope, "an unsupported request must explain why"


def test_extraction_prompt_selects_exact_match(tmp_path: Path) -> None:
    prompt = "Extract the invoice number and due date from each bill into a JSON object."
    spec = MLSpec.model_validate(_run(prompt, tmp_path).output)
    assert spec.task_type == "extraction"
    assert spec.evaluation.primary_metric == "exact_match"


# --- high-stakes safeguard -------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "Build a model that diagnoses patients from symptoms.",
        "Predict whether a candidate should be hired or rejected from a resume.",
        "Decide loan approval for applicants based on their credit history.",
        "Give legal advice about whether a contract is enforceable.",
    ],
)
def test_high_stakes_domains_are_flagged(tmp_path: Path, prompt: str) -> None:
    spec = MLSpec.model_validate(_run(prompt, tmp_path).output)
    assert spec.is_high_stakes is True, prompt
    assert spec.risk_notes, "a high-stakes flag must come with a stated risk"


def test_ordinary_ticket_task_is_not_high_stakes(tmp_path: Path) -> None:
    spec = MLSpec.model_validate(_run(TICKET_PROMPT, tmp_path).output)
    assert spec.is_high_stakes is False
    assert detect_high_stakes("route these messages") == []


def test_candidate_models_is_not_a_hiring_prompt(tmp_path: Path) -> None:
    """'candidate' appears in ordinary prompts about model selection — no false alarm."""
    prompt = "Compare candidate models and pick the smallest one that fits 8 GB VRAM."
    assert detect_high_stakes(prompt) == []
    spec = MLSpec.model_validate(_run(prompt, tmp_path).output)
    assert spec.is_high_stakes is False


# --- LLM path --------------------------------------------------------------


def test_uses_the_planner_llm_when_configured(tmp_path: Path, planner: pytest.MonkeyPatch) -> None:
    llm_payload = {
        "title": "Ticket router",
        "summary": "Routes tickets to a queue.",
        "task_type": "classification",
        "input_description": "Ticket text",
        "allowed_labels": ["Billing", "Technical"],
        "output": {"format": "json_object"},
        "evaluation": {"primary_metric": "macro_f1"},
        "open_questions": [],
        "assumptions": ["Labels taken from the prompt"],
    }

    async def fake_ask(_prompt: str) -> str:
        return json.dumps(llm_payload)

    planner.setattr(analyzer_module, "_ask_planner", fake_ask)
    result = _run(TICKET_PROMPT, tmp_path)

    assert result.status == "succeeded"
    spec = MLSpec.model_validate(result.output)
    assert spec.title == "Ticket router"
    assert spec.allowed_labels == ["Billing", "Technical"]
    assert spec.generated_by == settings.llm_model


def test_unavailable_planner_falls_back_and_says_so(
    tmp_path: Path, planner: pytest.MonkeyPatch
) -> None:
    from slmforge.integrations.llm import PlannerLLMUnavailable

    async def raise_unavailable(_prompt: str) -> str:
        raise PlannerLLMUnavailable("connection refused")

    planner.setattr(analyzer_module, "_ask_planner", raise_unavailable)
    result = _run(TICKET_PROMPT, tmp_path)

    assert result.status == "succeeded", "a dead LLM must not fail the stage"
    spec = MLSpec.model_validate(result.output)
    assert spec.generated_by == "heuristic"
    assert spec.task_type == "classification"


def test_malformed_llm_output_falls_back_instead_of_crashing(
    tmp_path: Path, planner: pytest.MonkeyPatch
) -> None:
    async def garbage(_prompt: str) -> str:
        return "I'm afraid I can't do that. Here's a JSON object: {not really json}"

    planner.setattr(analyzer_module, "_ask_planner", garbage)
    result = _run(TICKET_PROMPT, tmp_path)

    assert result.status == "succeeded"
    spec = MLSpec.model_validate(result.output)
    assert spec.generated_by == "heuristic"


def test_llm_supplied_unsupported_task_is_still_gated(
    tmp_path: Path, planner: pytest.MonkeyPatch
) -> None:
    """The gate must apply to LLM output too, not just the heuristic."""

    async def fake_ask(_prompt: str) -> str:
        return json.dumps(
            {
                "title": "x",
                "summary": "y",
                "task_type": "summarization",
                "input_description": "text",
                "allowed_labels": [],
            }
        )

    planner.setattr(analyzer_module, "_ask_planner", fake_ask)
    spec = MLSpec.model_validate(_run("Do something with text.", tmp_path).output)
    assert spec.readiness == "unsupported"
    assert spec.out_of_scope


def test_non_blocking_questions_do_not_stall_the_build(
    tmp_path: Path, planner: pytest.MonkeyPatch
) -> None:
    """Questions that a downstream stage answers must not block the user.

    "Is there labeled data?" is Stage 2's job to answer; blocking on it here
    would put an avoidable gate in front of every build.
    """
    async def fake_ask(_prompt: str) -> str:
        return json.dumps(
            {
                "title": "Ticket router",
                "summary": "s",
                "task_type": "classification",
                "input_description": "ticket text",
                "allowed_labels": ["Billing", "General"],
                "output": {"format": "json_object"},
                "open_questions": [
                    {
                        "id": "data",
                        "question": "Is there labeled training data?",
                        "why": "dataset stage will resolve",
                        "blocking": False,
                    }
                ],
            }
        )

    planner.setattr(analyzer_module, "_ask_planner", fake_ask)
    spec = MLSpec.model_validate(_run(TICKET_PROMPT, tmp_path).output)
    assert spec.readiness == "ready_for_review"
    assert spec.open_questions, "the question should still be shown to the user"


def test_no_key_means_heuristic_without_touching_the_network(tmp_path: Path) -> None:
    """The default state (no key) must work offline."""
    assert settings.llm_provider == "none"
    spec = MLSpec.model_validate(_run(TICKET_PROMPT, tmp_path).output)
    assert spec.generated_by == "heuristic"


def test_empty_prompt_is_a_clear_failure(tmp_path: Path) -> None:
    result = _run("   ", tmp_path)
    assert result.status == "failed"
    assert result.error


def test_stage_is_marked_implemented_and_registered() -> None:
    from slmforge.pipeline import get_stage

    stage = get_stage(StageName.analyze)
    assert isinstance(stage, AnalyzerStage)
    assert stage.implemented is True


def test_results_carry_a_human_readable_summary(tmp_path: Path) -> None:
    result = _run(TICKET_PROMPT, tmp_path)
    assert "classification" in result.summary
    assert result.metrics.get("open_questions") == 0
