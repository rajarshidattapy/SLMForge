"""Stage 4 — baseline evaluation contracts.

Inference itself is monkeypatched away: what is asserted is the *contract* —
a fixed seeded subset, strict output parsing, honest metrics, a broken model
reported as a failure (never a fabricated score) — so the suite stays fast and
needs no GPU, no weights and no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slmforge.models import Project, ProjectStatus
from slmforge.pipeline import baseline as bl
from slmforge.pipeline.base import StageContext
from slmforge.pipeline.baseline import BaselineStage
from slmforge.schemas import MLSpec, OutputFormat, OutputSpec, TaskType

LABELS = ["Billing", "Technical", "General"]


def make_spec() -> MLSpec:
    return MLSpec(
        title="Support ticket router",
        summary="Route customer support tickets to the right queue.",
        task_type=TaskType.classification,
        input_description="ticket text",
        allowed_labels=list(LABELS),
        output=OutputSpec(format=OutputFormat.json_object),
        domain="customer support",
    )


def _record(text: str, label: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "Classify."},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps({"label": label})},
        ],
        "label": label,
    }


def _test_file(tmp_path: Path, n: int = 12) -> Path:
    path = tmp_path / "artifacts" / "dataset" / "test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_record(f"ticket {i} about billing", LABELS[i % len(LABELS)]) for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _ctx(tmp_path: Path, *, params: dict[str, Any] | None = None) -> StageContext:
    project = Project(
        id="proj-base", name="p", prompt="route tickets", status=ProjectStatus.model_selected
    )
    project.spec = make_spec().model_dump(mode="json", by_alias=True)
    project.selected_model = {
        "selected": {"id": "hf:Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen2.5-0.5B"},
        "shortlist": [{"id": "hf:Qwen/Qwen2.5-0.5B-Instruct"}],
    }
    _test_file(tmp_path)
    return StageContext(
        project_id=project.id,
        prompt=project.prompt,
        project=project,
        artifact_dir=tmp_path / "artifacts",
        work_dir=tmp_path / "work",
        params=params or {},
    )


# --- parsing ------------------------------------------------------------------


def test_valid_json_label_parses() -> None:
    label, kind = bl.parse_prediction('{"label": "Billing"}', LABELS)
    assert (label, kind) == ("Billing", None)


def test_fenced_json_parses() -> None:
    label, kind = bl.parse_prediction('```json\n{"label": "Technical"}\n```', LABELS)
    assert (label, kind) == ("Technical", None)


def test_bare_label_is_invalid_not_close_enough() -> None:
    label, kind = bl.parse_prediction("Billing", LABELS)
    assert (label, kind) == (None, "not_json")


def test_unknown_label_missing_key_and_empty_are_distinct() -> None:
    assert bl.parse_prediction('{"label": "Refund"}', LABELS) == (None, "unknown_label")
    assert bl.parse_prediction('{"category": "Billing"}', LABELS) == (None, "missing_label_key")
    assert bl.parse_prediction("   ", LABELS) == (None, "empty")
    assert bl.parse_prediction("[1, 2]", LABELS) == (None, "not_json")


# --- metrics ------------------------------------------------------------------


def test_metrics_count_unparseable_as_wrong() -> None:
    stats = bl.compute_metrics(
        ["Billing", "Technical", "General"], ["Billing", None, "Billing"], LABELS
    )
    assert stats["accuracy"] == 1 / 3
    assert 0.0 <= stats["macro_f1"] <= 1.0
    assert stats["per_class"]["Billing"] == {"correct": 1, "total": 1}
    assert stats["confusion"]["Technical"][bl.INVALID] == 1


def test_latency_stats_need_no_samples() -> None:
    assert bl.latency_stats([]) == (None, None)
    p50, p95 = bl.latency_stats([10.0, 20.0, 30.0, 40.0])
    assert p50 is not None and p95 is not None and p50 <= p95


def test_subset_is_deterministic_and_bounded(tmp_path: Path) -> None:
    records = [_record(f"t{i}", "Billing") for i in range(50)]
    idx1, rec1 = bl.sample_subset(records, n=10, seed=42)
    idx2, rec2 = bl.sample_subset(records, n=10, seed=42)
    assert idx1 == idx2 and rec1 == rec2
    idx3, _ = bl.sample_subset(records, n=10, seed=7)
    assert idx3 != idx1
    _, rec_all = bl.sample_subset(records, n=5000, seed=42)
    assert len(rec_all) == 50


# --- stage outcomes (inference stubbed) ----------------------------------------


def _stub_generate(texts: list[str], label: str = "Billing") -> tuple[list[str], list[float], None]:
    return [json.dumps({"label": label}) for _ in texts], [5.0] * len(texts), None


def test_successful_baseline_reports_honest_numbers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bl, "load_model", lambda repo, home: (None, None, "cpu", "abc123"))
    monkeypatch.setattr(
        bl,
        "generate_texts",
        lambda tok, mdl, dev, recs, **kw: (
            [json.dumps({"label": r["label"]}) for r in recs],
            [5.0] * len(recs),
            None,
        ),
    )
    ctx = _ctx(tmp_path, params={"max_samples": 9})
    result = BaselineStage().run_timed(ctx)
    assert result.status == "succeeded", result.error
    assert result.needs_approval is False
    run = result.output["eval_runs"][0]
    assert run["accuracy"] == 1.0
    assert run["valid_output_rate"] == 1.0
    assert run["schema_violations"] == 0
    assert run["n_samples"] == 9
    assert run["device"] == "cpu"
    assert run["revision"] == "abc123"
    assert result.output["config"]["seed"] == 42
    assert len(result.output["config"]["indices"]) == 9
    assert (tmp_path / "artifacts" / "baseline" / "eval_config.json").is_file()


def test_broken_model_is_a_reported_failure_not_a_score(tmp_path: Path, monkeypatch) -> None:
    def _boom(repo: str, home: Any) -> Any:
        raise RuntimeError("no such repo")

    monkeypatch.setattr(bl, "load_model", _boom)
    result = BaselineStage().run_timed(_ctx(tmp_path))
    assert result.status == "failed"
    assert "Qwen2.5-0.5B" in (result.error or "")
    assert not result.output


def test_broken_extra_model_is_reported_while_selected_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    def _loader(repo: str, home: Any) -> Any:
        calls.append(repo)
        if "Broken" in repo:
            raise RuntimeError("unloadable")
        return None, None, "cpu", None

    monkeypatch.setattr(bl, "load_model", _loader)
    monkeypatch.setattr(
        bl, "generate_texts", lambda tok, mdl, dev, recs, **kw: _stub_generate(recs)
    )
    ctx = _ctx(
        tmp_path,
        params={"models": ["hf:Qwen/Qwen2.5-0.5B-Instruct", "hf:x/Broken-1B"], "max_samples": 6},
    )
    ctx.project.selected_model["shortlist"].append({"id": "hf:x/Broken-1B"})
    result = BaselineStage().run_timed(ctx)
    assert result.status == "succeeded", result.error
    assert len(result.output["eval_runs"]) == 2
    broken = result.output["eval_runs"][1]
    assert broken["errors"] and broken["n_samples"] == 0


def test_guards_fail_with_next_steps(tmp_path: Path) -> None:
    no_spec = _ctx(tmp_path)
    no_spec.project.spec = None
    assert BaselineStage().run_timed(no_spec).status == "failed"

    no_model = _ctx(tmp_path)
    no_model.project.selected_model = None
    assert "select_model" in (BaselineStage().run_timed(no_model).error or "")

    no_data = _ctx(tmp_path)
    (tmp_path / "artifacts" / "dataset" / "test.jsonl").unlink()
    assert "dataset" in (BaselineStage().run_timed(no_data).error or "").lower()

    unknown = _ctx(tmp_path, params={"models": ["hf:nope/Nope-1B"]})
    assert "shortlist" in (BaselineStage().run_timed(unknown).error or "").lower()


class _StubInputs(dict):
    """Dict of tensors with `.to(device)`, like a tokenizer batch."""

    def to(self, device: str) -> _StubInputs:
        return _StubInputs({k: v.to(device) for k, v in self.items()})


class _StubTokenizer:
    """Duck-typed tokenizer: char codes mod vocab, no downloads, no files."""

    def __init__(self, vocab: int = 64) -> None:
        self.vocab = vocab
        self.eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs) -> str:
        return "\n".join(str(m.get("content")) for m in messages)

    def __call__(self, text: str, **kwargs) -> _StubInputs:
        import torch

        ids = [ord(c) % self.vocab for c in text[:32]] or [1]
        return _StubInputs({"input_ids": torch.tensor([ids])})

    def decode(self, ids, **kwargs) -> str:
        return " ".join(f"tok{int(i)}" for i in ids)


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=64)
    return GPT2LMHeadModel(config).eval()


def test_consecutive_runs_agree_with_real_generate() -> None:
    """`generate()` consumes RNG state even when greedy, so the harness reseeds
    per sample — two identical runs must agree token-for-token."""
    records = [
        {"messages": [{"role": "user", "content": f"hello {i}"}], "label": "Billing"}
        for i in range(3)
    ]
    kwargs: dict = {"system_prompt": "Classify.", "max_new_tokens": 4, "seed": 7}
    model = _tiny_gpt2()
    first, _, _ = bl.generate_texts(_StubTokenizer(), model, "cpu", records, **kwargs)
    second, _, _ = bl.generate_texts(_StubTokenizer(), model, "cpu", records, **kwargs)
    assert first == second


def test_same_seed_reproduces(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bl, "load_model", lambda repo, home: (None, None, "cpu", None))
    monkeypatch.setattr(
        bl,
        "generate_texts",
        lambda tok, mdl, dev, recs, **kw: (
            [json.dumps({"label": "Technical"}) for _ in recs],
            [5.0] * len(recs),
            None,
        ),
    )
    first = BaselineStage().run_timed(_ctx(tmp_path, params={"max_samples": 9}))
    second = BaselineStage().run_timed(_ctx(tmp_path, params={"max_samples": 9}))
    assert first.output["eval_runs"][0]["accuracy"] == second.output["eval_runs"][0]["accuracy"]
    assert first.output["config"]["indices"] == second.output["config"]["indices"]
