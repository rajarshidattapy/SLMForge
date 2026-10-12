"""Stage 4 — Baseline evaluation (harness, not hardening).

Measures the approved candidate *before* training so Stage 7 can prove whether
fine-tuning helped. Two calls complete it, mirroring the earlier stages:

1. Enqueue — ``POST .../jobs`` with ``{"kind": "baseline_eval"}``. The synchronous
   endpoint refuses with 409: inference takes minutes and must never run in the
   API process.
2. The worker runs this stage, which evaluates the selected model (plus any
   shortlisted extras named in ``params["models"]``) on a fixed, seeded subset
   of the held-out test split and persists one `EvalRun` per model.

Where the work runs: the worker process, on GPU when torch has CUDA. That is a
deliberate exception to "everything through the sandbox": the Docker image is a
slim CPU-only base with no ML stack (by design — small attack surface), and GPU
passthrough there is a separate project. The inference code is ours (not an
untrusted artifact), and the dataset is parsed as data with stdlib `json` —
nothing user-supplied is executed. The API process never touches inference;
"never in the API process" is verifiable by process tree because the only entry
point is the `baseline_eval` job handler.

Reproducibility bundle: every run records model repo + snapshot revision,
device, system prompt, decoding params, seed and the sampled indices, so two
runs with the same seed agree — and anyone can check that they did.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from ..log import get_logger
from ..models import StageName
from ..schemas import EvalRun, MLSpec, SamplePrediction
from ..schemas.common import Metric
from .base import Stage, StageContext, StageResult
from .datasets import build_system_prompt

logger = get_logger("pipeline.baseline")

#: Default eval size. Small enough for minutes-per-model on a laptop GPU, large
#: enough for per-class signal on a 3-4 label task. Capped - a 10k-sample
#: baseline is a Stage 7 problem with a bigger machine, not a silent default.
DEFAULT_MAX_SAMPLES = 200
HARD_MAX_SAMPLES = 1000

#: The answer is a short JSON object (`{"label": "..."}`); anything longer is
#: rambling, not signal, and costs VRAM-seconds on every sample.
DEFAULT_MAX_NEW_TOKENS = 32

#: Error-analysis rows kept per model. All failures first, then correct fills.
SAMPLES_KEPT = 20

#: Marker for predictions that could not be parsed into an allowed label.
INVALID = "__invalid__"


# --- pure helpers (fast, unit-tested, no torch) -------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read chat-format JSONL records; raises a clear error, never partial data."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read eval file {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {lineno} is not valid JSON: {exc}") from exc
    return records


def sample_subset(
    records: list[dict[str, Any]], *, n: int, seed: int
) -> tuple[list[int], list[dict[str, Any]]]:
    """Deterministic subset: same records + seed always yields the same indices."""
    n = max(1, min(n, len(records)))
    rng = random.Random(seed)
    indices = (
        sorted(rng.sample(range(len(records)), n))
        if n < len(records)
        else list(range(len(records)))
    )
    return indices, [records[i] for i in indices]


def user_text(record: dict[str, Any]) -> str:
    """The model input: the user message of a chat record."""
    for message in record.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def parse_prediction(text: str, allowed_labels: list[str]) -> tuple[str | None, str | None]:
    """Parse generated text into an allowed label.

    Returns (label, error_kind). The contract is a JSON object with exactly the
    key "label" — a bare label with no JSON is *invalid*, not "close enough",
    because valid-output rate must measure the contract, not generosity.
    Markdown fences are stripped first; models wrap JSON in them constantly.
    """
    stripped = text.strip()
    if not stripped:
        return None, "empty"
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if len(lines) > 1 else []
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None, "not_json"
    if not isinstance(obj, dict):
        return None, "not_json"
    if "label" not in obj:
        return None, "missing_label_key"
    label = obj["label"]
    if label not in allowed_labels:
        return None, "unknown_label"
    return str(label), None


def compute_metrics(
    expected: list[str], predicted: list[str | None], allowed_labels: list[str]
) -> dict[str, Any]:
    """Task metrics over aligned labels; None predictions count as wrong."""
    from sklearn.metrics import accuracy_score, f1_score

    gold = list(expected)
    pred = [p if p is not None else INVALID for p in predicted]
    labels = list(allowed_labels) + ([INVALID] if INVALID in pred else [])
    accuracy = float(accuracy_score(gold, pred))
    macro_f1 = float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))
    per_class: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for label in allowed_labels:
        total = sum(1 for g in gold if g == label)
        correct = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
        per_class[label] = {"correct": correct, "total": total}
    for g, p in zip(gold, pred, strict=True):
        confusion.setdefault(g, {}).setdefault(p, 0)
        confusion[g][p] += 1
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
    }


def latency_stats(latencies_ms: list[float]) -> tuple[float | None, float | None]:
    """p50 / p95 of per-sample latency; None when nothing ran."""
    if not latencies_ms:
        return None, None
    ordered = sorted(latencies_ms)

    def _pick(quantile: float) -> float:
        return ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]

    return round(_pick(0.50), 1), round(_pick(0.95), 1)


# --- model I/O (worker process, GPU when available) ---------------------------


def bare_repo_id(candidate_id: str) -> str:
    """Strip the source prefix (`hf:org/name` → `org/name`).

    The prefix is routing metadata for the catalog, not part of the repo name —
    the Hub, the cache layout and transformers all reject it, so it is removed
    here, once, at the load boundary.
    """
    repo_id = candidate_id.removeprefix("hf:")
    if repo_id.startswith("ollama:"):
        raise RuntimeError(
            f"Could not load model '{candidate_id}': Ollama-hosted weights are not "
            "supported by the baseline harness yet; approve a Hub candidate."
        )
    return repo_id


def cached_snapshot(repo_id: str, hf_home: Path | str) -> tuple[Path, str] | None:
    """Local snapshot dir + revision for a cached repo, else None."""
    base = Path(hf_home) / "hub" / ("models--" + repo_id.replace("/", "--"))
    snapshots = base / "snapshots"
    try:
        revisions = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
    except OSError:
        return None
    if not revisions:
        return None
    return snapshots / revisions[-1], revisions[-1]


def load_model(repo_id: str, hf_home: Path | str) -> tuple[Any, Any, str, str | None]:
    """Load tokenizer + causal LM; returns (tokenizer, model, device, revision).

    Prefers the local snapshot (no network at run time); otherwise transformers
    downloads it, which is exactly the "verified download path" Stage 3 checked.
    Anything unusable raises with the repo named — a broken model is a reported
    failure, never a fabricated score.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Candidate ids carry a source prefix (`hf:org/name`); the Hub, the cache
    # layout and transformers all want the bare repo — strip it at the boundary
    # so a prefix can never leak into a load path again.
    repo_id = bare_repo_id(repo_id)
    cached = cached_snapshot(repo_id, hf_home)
    source = str(cached[0]) if cached else repo_id
    revision = cached[1] if cached else None
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Any: transformers' dynamic model classes defeat static typing, and the
        # caller only needs `.to/.eval/.generate`, which tests stub anyway.
        model: Any = AutoModelForCausalLM.from_pretrained(
            source, dtype=dtype, trust_remote_code=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model '{repo_id}': {type(exc).__name__}: {exc}"
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device, revision


def generate_texts(
    tokenizer: Any,
    model: Any,
    device: str,
    records: list[dict[str, Any]],
    *,
    system_prompt: str,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[str], list[float], float | None]:
    """Greedy generation per record; returns (texts, latencies_ms, peak_vram_gb).

    Batch size is 1 and sampling is off: determinism beats throughput for a
    baseline, and per-sample timing is the latency signal the UI charts.

    Greedy decoding is only reproducible if the kernels are too, so cuDNN
    autotuning is off and deterministic algorithms are on (warn-only: a kernel
    without a deterministic path still runs, and the seed agreement is checked
    by test, not assumed). This sets process-wide flags — reproducibility is a
    platform principle, not a per-call preference.

    The seed is also re-applied before *every* sample: `generate()` consumes
    RNG state on some stacks even when greedy, so a single seed at the start
    still drifts between runs. Reseeding makes each output a pure function of
    (seed, input) — verified live, where one seed per run demonstrably was not
    enough.
    """
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    pad_id = tokenizer.eos_token_id
    texts: list[str] = []
    latencies: list[float] = []
    for record in records:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text(record)},
        ]
        # Render to text first: across transformers versions
        # `apply_chat_template(..., return_tensors="pt")` returns a Tensor, a
        # BatchEncoding, or a plain list — the two-step form always works.
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[-1]
        started = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                inputs["input_ids"],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
            )
        latencies.append(round((time.perf_counter() - started) * 1000, 1))
        texts.append(tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True))
    peak = None
    if device == "cuda":
        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    return texts, latencies, peak


# --- stage --------------------------------------------------------------------


class BaselineStage(Stage):
    name = StageName.baseline
    label = "Establish baseline"
    implemented = True  # Stage 4

    def run(self, ctx: StageContext) -> StageResult:
        spec_raw = getattr(ctx.project, "spec", None)
        if not spec_raw:
            return StageResult(
                status="failed",
                error="No ML specification on the project. Run the analyze stage first.",
            )
        try:
            from ..schemas import MLSpec

            spec = MLSpec.model_validate(spec_raw)
        except Exception as exc:
            return StageResult(status="failed", error=f"Unreadable spec: {exc}")

        selected = ((getattr(ctx.project, "selected_model", None) or {}).get("selected")) or {}
        if not selected or not selected.get("id"):
            return StageResult(
                status="failed",
                error="No approved base model. Run select_model approval first.",
            )
        shortlist = ((getattr(ctx.project, "selected_model", None) or {}).get("shortlist")) or []
        shortlist_ids: set[str] = {
            str(c["id"]) for c in shortlist if isinstance(c, dict) and c.get("id")
        }

        wanted = ctx.params.get("models") or [selected["id"]]
        if isinstance(wanted, str):
            wanted = [wanted]
        unknown = [m for m in wanted if m not in shortlist_ids and m != selected["id"]]
        if unknown:
            return StageResult(
                status="failed",
                error=f"Not a shortlisted model: {', '.join(unknown)}. Ranked: {sorted(shortlist_ids)}.",
            )

        test_path = ctx.artifact_dir / "dataset" / "test.jsonl"
        if not test_path.is_file():
            return StageResult(
                status="failed",
                error="No test split found. Run the dataset build first.",
            )
        try:
            all_records = read_jsonl(test_path)
        except ValueError as exc:
            return StageResult(status="failed", error=str(exc))
        if not all_records:
            return StageResult(status="failed", error="Test split is empty; nothing to measure.")

        seed = int(ctx.params.get("seed", ctx.settings.seed))
        max_samples = max(
            1, min(int(ctx.params.get("max_samples", DEFAULT_MAX_SAMPLES)), HARD_MAX_SAMPLES)
        )
        indices, records = sample_subset(all_records, n=max_samples, seed=seed)
        system_prompt = build_system_prompt(spec)
        max_new_tokens = int(ctx.params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))

        eval_runs: list[dict[str, Any]] = []
        errors: list[str] = []
        for repo_id in wanted:
            try:
                run = self._evaluate_one(
                    ctx,
                    spec,
                    repo_id,
                    records,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
            except Exception as exc:
                # A model that cannot even load is reported, not hidden — but a
                # broken *selected* model fails the stage: there is no baseline.
                message = f"{repo_id}: {type(exc).__name__}: {exc}"
                logger.warning("Baseline failed for %s", message)
                if repo_id == selected["id"]:
                    return StageResult(status="failed", error=f"Baseline failed: {message}")
                errors.append(message)
                eval_runs.append(
                    {
                        "label": "baseline",
                        "model_id": repo_id,
                        "split": "test",
                        "n_samples": 0,
                        "errors": [message],
                    }
                )
                continue
            eval_runs.append(run)

        config = {
            "models": wanted,
            "split": "test",
            "n_samples": len(records),
            "indices": indices,
            "seed": seed,
            "system_prompt": system_prompt,
            "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens},
            "test_file": "dataset/test.jsonl",
        }
        artifacts = self._write_artifacts(ctx, eval_runs, config)
        first = next((r for r in eval_runs if r.get("metrics")), None)
        summary = (
            f"Baseline for {len(eval_runs)} model(s) on {len(records)} test samples "
            f"(seed {seed})."
            + (f" {selected['id']}: acc {(first or {}).get('accuracy', 0):.3f}." if first else "")
            + (f" {len(errors)} extra model(s) failed." if errors else "")
        )
        return StageResult(
            output={"eval_runs": eval_runs, "config": config},
            summary=summary,
            metrics={
                "models": float(len(eval_runs)),
                "n_samples": float(len(records)),
                "accuracy": float((first or {}).get("accuracy", 0.0)),
                "valid_output_rate": float((first or {}).get("valid_output_rate", 0.0)),
            },
            artifacts=artifacts,
        )

    def _evaluate_one(
        self,
        ctx: StageContext,
        spec: MLSpec,
        repo_id: str,
        records: list[dict[str, Any]],
        *,
        system_prompt: str,
        max_new_tokens: int,
        seed: int,
    ) -> dict[str, Any]:
        tokenizer, model, device, revision = load_model(repo_id, ctx.settings.hf_home)
        texts, latencies, peak_vram = generate_texts(
            tokenizer,
            model,
            device,
            records,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        expected = [str(r.get("label") or "") for r in records]
        parsed: list[str | None] = []
        error_kinds: list[str | None] = []
        violations = 0
        for text in texts:
            label, kind = parse_prediction(text, list(spec.allowed_labels))
            parsed.append(label)
            error_kinds.append(kind)
            if kind is not None:
                violations += 1
        stats = compute_metrics(expected, parsed, list(spec.allowed_labels))
        p50, p95 = latency_stats(latencies)
        valid_rate = round(sum(1 for p in parsed if p is not None) / max(1, len(parsed)), 4)
        order = sorted(
            range(len(records)),
            key=lambda i: (0 if error_kinds[i] else 1, i),
        )[:SAMPLES_KEPT]
        kept = [
            SamplePrediction(
                input=user_text(records[i])[:1000],
                expected=expected[i],
                predicted=parsed[i],
                correct=parsed[i] is not None and parsed[i] == expected[i],
                latency_ms=latencies[i],
                error_kind=error_kinds[i],
            )
            for i in order
        ]
        run = EvalRun(
            label="baseline",
            model_id=repo_id,
            split="test",
            n_samples=len(records),
            metrics=[
                Metric(
                    name="accuracy",
                    value=stats["accuracy"],
                    on_split="test",
                    n_samples=len(records),
                ),
                Metric(
                    name="macro_f1",
                    value=stats["macro_f1"],
                    on_split="test",
                    n_samples=len(records),
                ),
            ],
            valid_output_rate=valid_rate,
            schema_violations=violations,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            peak_vram_gb=peak_vram,
            confusion=stats["confusion"],
            samples=kept,
            errors=[]
            if device == "cuda"
            else [f"Ran on {device}; peak_vram_gb unavailable off-GPU."],
        )
        dumped = run.model_dump(mode="json")
        # Convenience top-levels for the dashboard; the canonical values live in
        # `metrics` (accuracy/macro_f1) and the per-class table below.
        dumped["accuracy"] = stats["accuracy"]
        dumped["macro_f1"] = stats["macro_f1"]
        dumped["per_class"] = stats["per_class"]
        dumped["device"] = device
        dumped["revision"] = revision
        return dumped

    def _write_artifacts(
        self, ctx: StageContext, eval_runs: list[dict[str, Any]], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        out_dir = ctx.artifact_dir / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict[str, Any]] = []
        for run in eval_runs:
            slug = run["model_id"].replace("/", "--")
            path = out_dir / f"eval_{slug}.json"
            path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
            artifacts.append(
                {
                    "kind": "eval_run",
                    "name": path.name,
                    "path": f"baseline/{path.name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "meta": {"model": run["model_id"], "label": "baseline"},
                }
            )
        cfg_path = out_dir / "eval_config.json"
        cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.append(
            {
                "kind": "eval_config",
                "name": cfg_path.name,
                "path": f"baseline/{cfg_path.name}",
                "size_bytes": cfg_path.stat().st_size,
                "sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
            }
        )
        return artifacts


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MAX_SAMPLES",
    "HARD_MAX_SAMPLES",
    "INVALID",
    "SAMPLES_KEPT",
    "BaselineStage",
    "bare_repo_id",
    "cached_snapshot",
    "compute_metrics",
    "generate_texts",
    "latency_stats",
    "load_model",
    "parse_prediction",
    "read_jsonl",
    "sample_subset",
    "user_text",
]
