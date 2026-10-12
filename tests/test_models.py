"""Stage 3 — model discovery & ranking contracts.

Every test is hermetic: Ollama/Hub access is monkeypatched away, so the suite
never touches the network. What is asserted is the *contract* — a defensible
shortlist with a stated reason per candidate, an approval gate that advances
nothing until a human decides, and honest handling of oversized, unlicensed or
unreachable options — rather than the internals of discovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from slmforge.models import Project, ProjectStatus
from slmforge.pipeline import models as mo
from slmforge.pipeline.base import StageContext
from slmforge.pipeline.models import ModelSelectionStage
from slmforge.schemas import (
    MLSpec,
    ModelCandidate,
    OutputFormat,
    OutputSpec,
    TaskType,
)


def make_spec() -> MLSpec:
    return MLSpec(
        title="Support ticket router",
        summary="Route customer support tickets to the right queue.",
        task_type=TaskType.classification,
        input_description="ticket text",
        allowed_labels=["Billing", "Technical", "Account Access", "General"],
        output=OutputSpec(format=OutputFormat.json_object),
        domain="customer support",
    )


def _ctx(tmp_path: Path, *, params: dict[str, Any] | None = None) -> StageContext:
    project = Project(
        id="proj-model", name="p", prompt="route tickets", status=ProjectStatus.specified
    )
    project.spec = make_spec().model_dump(mode="json", by_alias=True)
    return StageContext(
        project_id=project.id,
        prompt=project.prompt,
        project=project,
        artifact_dir=tmp_path / "artifacts",
        work_dir=tmp_path / "work",
        params=params or {},
    )


@pytest.fixture()
def hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Ollama server, empty HF cache view, no Hub calls unless asked."""
    monkeypatch.setattr(mo, "ollama_inventory", lambda: [])


def _big_candidate() -> ModelCandidate:
    return ModelCandidate(
        id="hf:some-org/MegaMix-70B-Instruct",
        name="MegaMix-70B-Instruct",
        source="hf_hub",
        parameters_b=70.0,
        instruction_tuned=True,
        task_fit=["classification"],
        license=mo.normalise_license("apache-2.0"),
        finetune=mo.finetune_support(70.0),
    )


# --- estimates -------------------------------------------------------------


def test_qlora_estimate_grows_with_size_and_budget_inverts_it() -> None:
    assert mo.estimate_qlora_gb(0.5) < mo.estimate_qlora_gb(1.5) < mo.estimate_qlora_gb(7.0)
    assert mo.estimate_qlora_gb(0.5) < 8.0 < mo.estimate_qlora_gb(7.0)
    assert mo.max_params_for_budget(8.0) == pytest.approx(4.19, abs=0.01)


def test_fits_budget_is_a_fact_about_size() -> None:
    small = mo.catalog_candidates()[0]
    assert mo.fits_budget(small, 8.0) is True
    assert mo.fits_budget(_big_candidate(), 8.0) is False
    unknown = ModelCandidate(id="hf:x/y", name="y", source="hf_hub")
    assert mo.fits_budget(unknown, 8.0) is False


# --- ranking ----------------------------------------------------------------


def test_known_good_outranks_oversized_with_a_stated_reason(hermetic: None) -> None:
    good = mo.catalog_candidates()[0]
    big = _big_candidate()
    ranked = mo.rank_candidates([big, good], make_spec(), vram_budget_gb=8.0)
    assert ranked[0].id == good.id
    assert (ranked[-1].rank_score or 0.0) == 0.0 or ranked[-1].id == big.id
    assert any("budget" in w for w in big.warnings)
    assert big.rank_rationale is not None and "vram 0.00" in big.rank_rationale
    assert good.rank_rationale is not None and "task" in good.rank_rationale


def test_unknown_licence_is_a_warning_not_a_disappearance(hermetic: None) -> None:
    lead = ModelCandidate(id="hf:some-org/Mystery-1B-Instruct", name="Mystery", source="hf_hub")
    ranked = mo.rank_candidates([lead], make_spec(), vram_budget_gb=8.0)
    assert ranked[0].id == lead.id
    assert any("licence" in w.lower() for w in lead.warnings)
    assert lead.rank_rationale is not None and "licence 0.50" in lead.rank_rationale


def test_ties_break_toward_the_cheaper_fit(hermetic: None) -> None:
    small = next(c for c in mo.catalog_candidates() if c.id.endswith("SmolLM2-360M-Instruct"))
    big = next(c for c in mo.catalog_candidates() if c.id.endswith("SmolLM2-1.7B-Instruct"))
    ranked = mo.rank_candidates([big, small], make_spec(), vram_budget_gb=8.0)
    assert ranked[0].rank_score == ranked[1].rank_score
    assert ranked[0].id == small.id


def test_restricted_licence_scores_low_but_survives(hermetic: None) -> None:
    gemma = next(c for c in mo.catalog_candidates() if c.id.endswith("gemma-3-270m-it"))
    free = next(c for c in mo.catalog_candidates() if c.id.endswith("Qwen2.5-0.5B-Instruct"))
    ranked = mo.rank_candidates([gemma, free], make_spec(), vram_budget_gb=8.0)
    assert ranked[0].id == free.id
    assert any("licence" in w.lower() for w in gemma.warnings)


# --- locality ---------------------------------------------------------------


def test_ollama_match_marks_local_availability() -> None:
    candidates = mo.catalog_candidates()
    mo.apply_locality(candidates, ollama_names=["qwen2.5:0.5b"], hf_cached=[])
    qwen = next(c for c in candidates if c.id.endswith("Qwen2.5-0.5B-Instruct"))
    smol = next(c for c in candidates if c.id.endswith("SmolLM2-1.7B-Instruct"))
    assert qwen.local_available is True
    assert smol.local_available is False


def test_hf_cache_snapshot_marks_downloaded(tmp_path: Path) -> None:
    snap = tmp_path / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    assert mo.local_hf_models(tmp_path) == ["Qwen/Qwen2.5-0.5B-Instruct"]
    candidates = mo.catalog_candidates()
    mo.apply_locality(candidates, ollama_names=[], hf_cached=["Qwen/Qwen2.5-0.5B-Instruct"])
    qwen = next(c for c in candidates if c.id.endswith("Qwen2.5-0.5B-Instruct"))
    assert qwen.downloaded is True and qwen.local_available is True


# --- stage outcomes ---------------------------------------------------------


def test_ranking_run_needs_approval_and_does_not_advance(
    tmp_path: Path, session, hermetic: None
) -> None:
    from slmforge.pipeline.orchestrator import Orchestrator

    project = Project(id="rank1", name="p", prompt="route tickets", status=ProjectStatus.specified)
    project.spec = make_spec().model_dump(mode="json", by_alias=True)
    session.add(project)
    session.commit()

    result = Orchestrator(session).run_stage(
        project, ModelSelectionStage(), params={"offline": True}
    )
    assert result.status == "succeeded"
    assert result.needs_approval is True
    session.refresh(project)
    assert project.status == ProjectStatus.specified
    selection = project.selected_model
    assert selection["selected"] is None
    assert len(selection["shortlist"]) == len(mo.CATALOG)
    top = selection["shortlist"][0]
    assert top["rank_score"] is not None and top["rank_rationale"]


def test_approval_advances_with_alternatives_recorded(
    tmp_path: Path, session, hermetic: None
) -> None:
    from slmforge.pipeline.orchestrator import Orchestrator

    project = Project(id="rank2", name="p", prompt="route tickets", status=ProjectStatus.specified)
    project.spec = make_spec().model_dump(mode="json", by_alias=True)
    session.add(project)
    session.commit()

    top_id = mo.catalog_candidates()[0].id
    result = Orchestrator(session).run_stage(
        project,
        ModelSelectionStage(),
        params={"offline": True, "selected": top_id, "decision_notes": "smallest that fits"},
    )
    assert result.status == "succeeded", result.error
    assert result.needs_approval is False
    session.refresh(project)
    assert project.status == ProjectStatus.model_selected
    selection = project.selected_model
    assert selection["selected"]["id"] == top_id
    assert selection["selected"]["approved"] is True
    assert len(selection["shortlist"]) == len(mo.CATALOG)
    assert selection["decision_notes"] == "smallest that fits"


def test_oversized_selection_is_blocked_not_recorded(
    tmp_path: Path, session, hermetic: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slmforge.pipeline.orchestrator import Orchestrator

    big = _big_candidate()
    monkeypatch.setattr(mo, "catalog_candidates", lambda: [big])

    project = Project(id="rank3", name="p", prompt="route tickets", status=ProjectStatus.specified)
    project.spec = make_spec().model_dump(mode="json", by_alias=True)
    session.add(project)
    session.commit()

    result = Orchestrator(session).run_stage(
        project, ModelSelectionStage(), params={"offline": True, "selected": big.id}
    )
    assert result.status == "failed"
    assert "budget" in (result.error or "")
    session.refresh(project)
    assert project.status == ProjectStatus.specified


def test_unknown_selected_id_fails_with_valid_ids(tmp_path: Path, hermetic: None) -> None:
    result = ModelSelectionStage().run_timed(
        _ctx(tmp_path, params={"offline": True, "selected": "hf:nope/Nope-1B"})
    )
    assert result.status == "failed"
    assert "hf:Qwen/Qwen2.5-0.5B-Instruct" in (result.error or "")


def test_rejection_is_recorded_and_stays_put(tmp_path: Path, session, hermetic: None) -> None:
    from slmforge.pipeline.orchestrator import Orchestrator

    project = Project(id="rank4", name="p", prompt="route tickets", status=ProjectStatus.specified)
    project.spec = make_spec().model_dump(mode="json", by_alias=True)
    session.add(project)
    session.commit()

    result = Orchestrator(session).run_stage(
        project,
        ModelSelectionStage(),
        params={"offline": True, "rejected": True, "decision_notes": "want a smaller one"},
    )
    assert result.status == "succeeded"
    assert result.needs_approval is True
    session.refresh(project)
    assert project.status == ProjectStatus.specified
    assert project.selected_model["decision_notes"] == "want a smaller one"


def test_hub_outage_degrades_to_catalog_with_a_warning(
    tmp_path: Path, hermetic: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network down")

    monkeypatch.setattr(mo.hf_cli, "list_models", _boom)
    result = ModelSelectionStage().run_timed(_ctx(tmp_path))
    assert result.status == "succeeded"
    assert result.needs_approval is True
    assert any("catalog only" in w for c in result.output["shortlist"] for w in c["warnings"])


def test_no_spec_fails_with_a_next_step(tmp_path: Path, hermetic: None) -> None:
    project = Project(id="nospec", name="p", prompt="hi", status=ProjectStatus.draft)
    ctx = StageContext(
        project_id=project.id,
        prompt=project.prompt,
        project=project,
        artifact_dir=tmp_path,
        work_dir=tmp_path,
        params={"offline": True},
    )
    result = ModelSelectionStage().run_timed(ctx)
    assert result.status == "failed"
    assert "analyze" in (result.error or "")
