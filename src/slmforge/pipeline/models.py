"""Stage 3 — Model discovery & ranking.

Builds a ranked shortlist of candidate base models from three sources: a
hand-verified catalog (works offline), the local Ollama inventory and HF cache
(marks what is already here), and Hub search leads (offered as unverified leads,
never as buildable facts).

Two calls complete the stage, mirroring the dataset plan/approve split:

1. Ranking — ``POST .../stages/select_model/run`` with ``{"approved": true}``
   (ranking is read-only and cheap, but the stage-level gate still wants the
   explicit flag). Returns a ``ModelSelection`` with ``selected=None`` and
   ``needs_approval=True``, so the project does **not** advance.
2. Decision — the same endpoint with ``params={"selected": "<candidate id>"}``
   (or ``{"rejected": true, "decision_notes": "..."}``). Approving a shortlisted
   candidate persists it plus the considered alternatives and advances the
   project to ``model_selected``; rejecting records the notes and stays put.

The ranking rubric is deliberately transparent — every term is documented in
``_RUBRIC`` and spelled out per candidate in ``rank_rationale``, because a bare
score is an opinion dressed as a number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..integrations import hf_cli
from ..integrations.huggingface import is_commercial_friendly, parse_params_billions
from ..log import get_logger
from ..models import StageName
from ..schemas import FineTuneSupport, License, MLSpec, ModelCandidate, ModelSelection
from ..schemas.common import ResourceEstimate
from .base import Stage, StageContext, StageResult

logger = get_logger("pipeline.models")

#: VRAM budget headroom for the Hub query: leads are fetched up to the largest
#: size that could conceivably still fit, so the query stays small while ranking
#: stays honest about what actually fits (oversized candidates are flagged, not
#: dropped).
_HUB_QUERY_HEADROOM_B = 1.0

#: Rubric weights. They sum to 1.0 so the score reads as a weighted verdict,
#: and each term is restated in `rank_rationale` — never a bare number.
_RUBRIC: dict[str, float] = {
    "task_fit": 0.35,  # instruction-tuned + suited to the spec's output contract
    "vram_fit": 0.35,  # QLoRA estimate against the machine's VRAM budget
    "licence": 0.20,  # commercial clearance of the declared licence
    "locality": 0.10,  # already here beats a download that may fail later
}

#: The curated catalog. Entries are models a human verified by hand — repo,
#: size, context window and licence as declared on the Hub — so ranking works
#: with no network at all. The demo's reference machine has 8 GB of VRAM, and
#: every entry fits QLoRA inside it; the gated Llama and Gemma entries exist so
#: the licence/auth-warning paths are exercised on every real run, not just in
#: tests.
CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "hf:Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen2.5-0.5B-Instruct",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "qwen2.5",
        "params_b": 0.5,
        "context_window": 32768,
        "license": "apache-2.0",
        "instruction_tuned": True,
        "ollama_aliases": ("qwen2.5:0.5b", "qwen2.5:0.5b-instruct"),
    },
    {
        "id": "hf:HuggingFaceTB/SmolLM2-360M-Instruct",
        "name": "SmolLM2-360M-Instruct",
        "repo_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "family": "smollm2",
        "params_b": 0.36,
        "context_window": 8192,
        "license": "apache-2.0",
        "instruction_tuned": True,
        "ollama_aliases": ("smollm2:360m", "smollm2:360m-instruct"),
    },
    {
        "id": "hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "family": "tinyllama",
        "params_b": 1.1,
        "context_window": 2048,
        "license": "apache-2.0",
        "instruction_tuned": True,
        "ollama_aliases": ("tinyllama", "tinyllama:1.1b-chat"),
    },
    {
        "id": "hf:Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen2.5-1.5B-Instruct",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "qwen2.5",
        "params_b": 1.5,
        "context_window": 32768,
        "license": "apache-2.0",
        "instruction_tuned": True,
        "ollama_aliases": ("qwen2.5:1.5b", "qwen2.5:1.5b-instruct"),
    },
    {
        "id": "hf:HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "name": "SmolLM2-1.7B-Instruct",
        "repo_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "family": "smollm2",
        "params_b": 1.7,
        "context_window": 8192,
        "license": "apache-2.0",
        "instruction_tuned": True,
        "ollama_aliases": ("smollm2:1.7b", "smollm2:1.7b-instruct"),
    },
    {
        "id": "hf:meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama-3.2-1B-Instruct",
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "llama3.2",
        "params_b": 1.24,
        "context_window": 131072,
        "license": "llama3.2",
        "instruction_tuned": True,
        "gated": True,
        "ollama_aliases": ("llama3.2:1b", "llama3.2:1b-instruct"),
    },
    {
        "id": "hf:google/gemma-3-270m-it",
        "name": "gemma-3-270m-it",
        "repo_id": "google/gemma-3-270m-it",
        "family": "gemma3",
        "params_b": 0.27,
        "context_window": 32768,
        "license": "gemma",
        "instruction_tuned": True,
        "gated": True,
        "ollama_aliases": ("gemma3:270m",),
    },
)


def estimate_qlora_gb(params_b: float) -> float:
    """Estimate VRAM (GB) for a QLoRA fine-tune of a `params_b`-billion model.

    Documented heuristic, not a measurement: 4-bit weights cost ~0.55 GB per
    billion parameters; LoRA optimizer states, gradients, activations and CUDA
    overhead run ~1.0 GB per billion; the framework, tokenizer and an eval
    batch need ~1.5 GB whatever the size. Stage 5 compares this estimate
    against the observed peak and says so when it was wrong.
    """
    return round(1.5 + 1.55 * params_b, 1)


def max_params_for_budget(vram_gb: float) -> float:
    """Largest parameter count (billions) whose QLoRA estimate fits `vram_gb`."""
    return max(0.0, round((vram_gb - 1.5) / 1.55, 2))


def normalise_license(license_id: str | None) -> License:
    """Build a `License` verdict that never pretends an unknown is cleared.

    `None` (no licence declared, or an id outside both lists) means *unknown*,
    which is a verdict the caller must surface — not a clearance to assume.
    """
    if not license_id or not license_id.strip():
        return License(
            id="unknown",
            unknown=True,
            notes="No licence declared; review required before training or deployment.",
        )
    normalized = license_id.strip().lower()
    verdict = is_commercial_friendly(normalized)
    if verdict is True:
        return License(id=normalized, commercial_ok=True)
    if verdict is False:
        return License(
            id=normalized,
            commercial_ok=False,
            share_alike="-sa" in normalized,
            notes=f"Licence {normalized} needs review before commercial use or deployment.",
        )
    return License(
        id=normalized,
        unknown=True,
        notes=f"Licence {normalized} is not on the known lists; review required.",
    )


def finetune_support(params_b: float | None, *, method: str = "qlora") -> FineTuneSupport:
    """Describe how a candidate of this size can be adapted, if at all."""
    if params_b is None or params_b <= 0:
        return FineTuneSupport(
            methods=[],
            supported=False,
            notes="Parameter count unknown, so no training method can be vouched for.",
        )
    est = estimate_qlora_gb(params_b)
    return FineTuneSupport(
        methods=["qlora", "lora"],
        supported=True,
        min_vram_gb=est,
        requires_gpu=True,
        tooling=["peft", "trl", "transformers", "bitsandbytes"],
        notes=(
            f"Planned method is {method} (QLoRA estimate {est} GB). Training on CPU "
            "is possible but very slow; a CUDA build of torch is expected."
        ),
    )


def discovery_queries(spec: MLSpec, limit: int = 2) -> list[str]:
    """Hub search terms derived from the spec — small and generic on purpose."""
    queries = ["instruct", "chat"] if "class" in spec.task_type.value else ["instruct"]
    return queries[:limit]


def catalog_candidates() -> list[ModelCandidate]:
    """Verified entries as candidates. Scoring happens once locality is known."""
    out: list[ModelCandidate] = []
    for entry in CATALOG:
        params_b = float(entry["params_b"])
        candidate = ModelCandidate(
            id=str(entry["id"]),
            name=str(entry["name"]),
            source="hf_hub",
            family=str(entry.get("family")),
            parameters_b=params_b,
            context_window=int(entry["context_window"]),
            instruction_tuned=bool(entry.get("instruction_tuned", False)),
            task_fit=["classification", "structured-generation"],
            license=normalise_license(str(entry.get("license"))),
            local_available=False,
            downloaded=False,
            finetune=finetune_support(params_b),
        )
        if entry.get("gated"):
            candidate.warnings.append(
                "Gated repo: downloading needs Hub authentication (`hf auth login`)."
            )
        out.append(candidate)
    return out


def ollama_inventory() -> list[str]:
    """Names of locally installed Ollama models; [] when the server is down.

    The server being down is the normal state on most machines (the reference
    box included), so any failure — unreachable, timeout, a running event loop
    we cannot join — degrades to "nothing local" rather than failing the stage.
    """
    try:
        import asyncio

        from ..integrations import ollama as _ollama

        async def _list() -> list[str]:
            client = _ollama.OllamaClient()
            try:
                return [m.name for m in await client.list_models()]
            finally:
                await client.close()

        return asyncio.run(_list())
    except Exception as exc:
        logger.info("Ollama inventory unavailable: %s", exc)
        return []


def local_hf_models(hf_home: Path | str) -> list[str]:
    """Repo ids with a snapshot in the local HF cache (`models--Org--Name`)."""
    found: list[str] = []
    hub = Path(hf_home) / "hub"
    try:
        entries = sorted(hub.iterdir())
    except OSError:
        return []
    for entry in entries:
        if entry.is_dir() and entry.name.startswith("models--"):
            repo = entry.name.removeprefix("models--").replace("--", "/")
            if (entry / "snapshots").is_dir():
                found.append(repo)
    return found


def hub_leads(
    spec: MLSpec, *, max_params_b: float, limit: int = 8
) -> tuple[list[ModelCandidate], str | None]:
    """Search the Hub for extra candidates. Returns (leads, warning).

    Leads are *unverified*: the search tells us a repo exists in the size band,
    not its licence, context window or fitness. They ship with `approved=False`
    and a warning saying exactly that, and an approving human is the review.
    A dead network degrades to catalog-only rather than failing the stage.
    """
    found: dict[str, ModelCandidate] = {}
    try:
        for query in discovery_queries(spec):
            for row in hf_cli.list_models(search=query, max_params_b=max_params_b, limit=limit):
                repo_id = row.get("id") or row.get("name")
                if not repo_id or f"hf:{repo_id}" in found:
                    continue
                params_b = parse_params_billions(repo_id)
                lowered = repo_id.lower()
                instruct = "instruct" in lowered or "chat" in lowered
                found[f"hf:{repo_id}"] = ModelCandidate(
                    id=f"hf:{repo_id}",
                    name=repo_id,
                    source="hf_hub",
                    parameters_b=params_b,
                    instruction_tuned=instruct,
                    task_fit=["classification", "structured-generation"] if instruct else [],
                    license=normalise_license(None),
                    local_available=False,
                    downloaded=False,
                    finetune=finetune_support(params_b) if params_b else finetune_support(None),
                    warnings=[
                        "Hub lead, not profiled: licence, context window and size "
                        "are unverified until inspected."
                    ],
                    approved=False,
                )
    except Exception as exc:
        logger.info("Hub model search unavailable: %s", exc)
        return list(found.values()), f"Hub search unavailable ({type(exc).__name__}); catalog only."
    return sorted(found.values(), key=lambda c: c.name), None


def apply_locality(
    candidates: list[ModelCandidate], *, ollama_names: list[str], hf_cached: list[str]
) -> None:
    """Mark what is already on this machine, in place. Idempotent."""
    cached = {repo.lower() for repo in hf_cached}
    lowered_ollama = [name.lower() for name in ollama_names]
    for candidate in candidates:
        repo = candidate.id.removeprefix("hf:").lower()
        if repo in cached:
            candidate.downloaded = True
            candidate.local_available = True
            continue
        entry = next((e for e in CATALOG if str(e["id"]).lower() == candidate.id.lower()), None)
        aliases = [str(a).lower() for a in ((entry.get("ollama_aliases") or ()) if entry else ())]
        own = candidate.id.removeprefix("hf:").split("/")[-1].lower()
        if any(name in aliases or own in name for name in lowered_ollama):
            candidate.local_available = True


def score_candidate(
    candidate: ModelCandidate, spec: MLSpec, *, vram_budget_gb: float
) -> tuple[float, str]:
    """Score a candidate against the rubric; returns (score, rationale).

    Oversized candidates are *kept* with a zero VRAM term and a blocking
    warning — ranking must say why a model is unusable, not silently drop it.
    """
    _ = spec  # per-task weights are a future refinement; the rubric is fixed for now.
    terms: dict[str, float] = {}
    notes: list[str] = []

    instruct = candidate.instruction_tuned or bool(candidate.task_fit)
    structured = bool(candidate.task_fit)
    terms["task_fit"] = round(0.20 * instruct + 0.15 * structured, 3)
    notes.append("instruction-tuned" if candidate.instruction_tuned else "not instruction-tuned")

    est = candidate.finetune.min_vram_gb
    if est is None:
        terms["vram_fit"] = 0.4
        notes.append("size unverified, so the QLoRA fit is a guess")
        candidate.warnings.append(
            "Parameter count unknown: the QLoRA fit cannot be vouched for; "
            "inspect the repo before approving."
        )
    elif est > vram_budget_gb:
        terms["vram_fit"] = 0.0
        notes.append(f"QLoRA est {est} GB exceeds the {vram_budget_gb:g} GB budget")
        candidate.warnings.append(
            f"Too large for this machine: QLoRA needs ~{est} GB but only "
            f"{vram_budget_gb:g} GB is budgeted. It cannot be trained here."
        )
    else:
        # Full marks under 75% of budget, sliding to half marks at the limit —
        # tight fits train, but leave no room for longer contexts or batches.
        headroom = (vram_budget_gb - est) / vram_budget_gb
        terms["vram_fit"] = round(0.5 + 0.5 * min(1.0, headroom / 0.25), 3)
        notes.append(f"QLoRA est {est} GB vs {vram_budget_gb:g} GB budget")

    lic = candidate.license
    if lic is not None and lic.commercial_ok is True:
        terms["licence"] = 1.0
        notes.append(f"licence {lic.id} clears commercial use")
    elif lic is not None and lic.commercial_ok is False:
        terms["licence"] = 0.1
        notes.append(f"licence {lic.id} needs review before commercial use")
        candidate.warnings.append(
            f"Licence {lic.id} does not clearly permit commercial use — approving "
            "this candidate is the licence review."
        )
    else:
        terms["licence"] = 0.5
        notes.append("licence unknown, needs review")
        candidate.warnings.append(
            "No usable licence verdict: review the repo's licence before approving."
        )

    if candidate.local_available:
        terms["locality"] = 1.0
        notes.append("already on this machine" if candidate.downloaded else "available via Ollama")
    elif candidate.id.startswith("hf:"):
        terms["locality"] = 0.6
        notes.append("verified Hub repo; downloads at train time")
    else:
        terms["locality"] = 0.3
        notes.append("not present locally")

    score = round(sum(_RUBRIC[k] * terms[k] for k in _RUBRIC), 3)
    rationale = (
        f"task {terms['task_fit']:.2f} ({notes[0]}), "
        f"vram {terms['vram_fit']:.2f} ({notes[1]}), "
        f"licence {terms['licence']:.2f} ({notes[2]}), "
        f"locality {terms['locality']:.2f} ({notes[3]})."
    )
    return score, rationale


def rank_candidates(
    candidates: list[ModelCandidate], spec: MLSpec, *, vram_budget_gb: float
) -> list[ModelCandidate]:
    """Score every candidate in place and return them best-first.

    Ties break toward the cheaper fit (lower QLoRA estimate, then name): with
    no measurements yet, cost is the only honest tie-break, and the baseline
    stage exists precisely to resolve capability ties with real numbers.
    """
    for candidate in candidates:
        candidate.rank_score, candidate.rank_rationale = score_candidate(
            candidate, spec, vram_budget_gb=vram_budget_gb
        )

    def _key(candidate: ModelCandidate) -> tuple[float, float, str]:
        est = candidate.finetune.min_vram_gb
        return (
            -(candidate.rank_score or 0.0),
            est if est is not None else float("inf"),
            candidate.name,
        )

    return sorted(candidates, key=_key)


def fits_budget(candidate: ModelCandidate, budget_gb: float) -> bool:
    """Whether the candidate's QLoRA estimate fits the VRAM budget (a fact)."""
    est = candidate.finetune.min_vram_gb
    return (
        candidate.finetune.supported
        and candidate.parameters_b is not None
        and est is not None
        and est <= budget_gb
    )


def compared_on() -> list[str]:
    """The rubric dimensions, persisted so the UI can label the comparison."""
    return [
        "task fit (instruction tuning, JSON output)",
        "VRAM fit (QLoRA estimate vs budget)",
        "licence clearance",
        "local availability",
    ]


def _spec_from_project(ctx: StageContext) -> MLSpec | None:
    """Parse the project's stored spec. None when analyze has not run yet."""
    raw = getattr(ctx.project, "spec", None)
    if not raw:
        return None
    try:
        return MLSpec.model_validate(raw)
    except Exception as exc:
        logger.warning("Project %s has an unreadable spec: %s", ctx.project_id[:8], exc)
        return None


class ModelSelectionStage(Stage):
    name = StageName.select_model
    label = "Select base model"
    requires_approval = True
    implemented = True  # Stage 3

    def run(self, ctx: StageContext) -> StageResult:
        spec = _spec_from_project(ctx)
        if spec is None:
            return StageResult(
                status="failed",
                error="No ML specification on the project. Run the analyze stage first.",
            )

        budget = float(ctx.settings.max_vram_gb)
        shortlist = self._shortlist(ctx, spec, budget=budget)

        params = ctx.params
        if params.get("rejected"):
            return self._rejection(shortlist, params)
        if params.get("selected"):
            return self._approval(shortlist, str(params["selected"]), params, budget=budget)
        return self._ranking(shortlist, budget=budget)

    # -- discovery ------------------------------------------------------
    def _shortlist(self, ctx: StageContext, spec: MLSpec, *, budget: float) -> list[ModelCandidate]:
        """Catalog + local marks + Hub leads, ranked. Never raises on I/O."""
        candidates = catalog_candidates()
        ollama_names = ollama_inventory()
        hf_cached = local_hf_models(ctx.settings.hf_home)
        apply_locality(candidates, ollama_names=ollama_names, hf_cached=hf_cached)

        notice: str | None = None
        if not ctx.params.get("offline"):
            leads, notice = hub_leads(
                spec, max_params_b=max_params_for_budget(budget) + _HUB_QUERY_HEADROOM_B
            )
            known = {c.id for c in candidates}
            for lead in leads:
                if lead.id not in known:
                    candidates.append(lead)
                    known.add(lead.id)
            apply_locality(candidates, ollama_names=ollama_names, hf_cached=hf_cached)

        ranked = rank_candidates(candidates, spec, vram_budget_gb=budget)
        if notice and ranked:
            ranked[0].warnings.append(notice)
        return ranked

    # -- outcomes -------------------------------------------------------
    def _ranking(self, shortlist: list[ModelCandidate], *, budget: float) -> StageResult:
        """No decision yet: persist the shortlist, advance nothing."""
        pick = next((c for c in shortlist if fits_budget(c, budget)), None)
        summary = (
            f"Ranked {len(shortlist)} candidate(s); recommended "
            f"'{pick.name}' (score {pick.rank_score}). Awaiting approval."
            if pick
            else f"Ranked {len(shortlist)} candidate(s); none fits this machine. "
            "Raise the VRAM budget or approve CPU-only explicitly."
        )
        selection = ModelSelection(
            selected=None,
            shortlist=shortlist,
            compared_on=compared_on(),
            decision_notes=None,
            requires_approval=True,
        )
        return StageResult(
            output=selection.model_dump(mode="json"),
            summary=summary,
            metrics={
                "candidates": float(len(shortlist)),
                "top_score": float(pick.rank_score or 0.0) if pick else 0.0,
            },
            artifacts=[
                {
                    "kind": "model_selection",
                    "name": "model_selection.json",
                    "path": "model_selection.json",
                    "meta": {"selected": None},
                }
            ],
            needs_approval=True,
            blocked_on=["approve_model_choice"],
        )

    def _approval(
        self,
        shortlist: list[ModelCandidate],
        selected_id: str,
        params: dict[str, Any],
        *,
        budget: float,
    ) -> StageResult:
        """Approve one shortlisted candidate. Size is physics, not preference,
        so an oversized pick stays blocked; a flagged licence is allowed
        because the explicit approval *is* the licence review."""
        match = next((c for c in shortlist if c.id == selected_id), None)
        if match is None:
            valid = ", ".join(c.id for c in shortlist) or "none"
            return StageResult(
                status="failed",
                error=f"Unknown model '{selected_id}'. Ranked candidates: {valid}.",
            )
        if not fits_budget(match, budget):
            return StageResult(
                status="failed",
                error=(
                    f"'{match.name}' cannot be trained on this machine "
                    f"(QLoRA ~{match.finetune.min_vram_gb} GB vs {budget:g} GB budget). "
                    "Pick a smaller candidate or raise SLMFORGE_MAX_VRAM_GB."
                ),
            )
        chosen = match.model_copy(deep=True)
        chosen.approved = True
        notes = params.get("decision_notes")
        selection = ModelSelection(
            selected=chosen,
            shortlist=shortlist,
            compared_on=compared_on(),
            decision_notes=str(notes) if notes else None,
            requires_approval=False,
        )
        return StageResult(
            output=selection.model_dump(mode="json"),
            summary=f"Selected '{chosen.name}' (score {chosen.rank_score}).",
            metrics={
                "candidates": float(len(shortlist)),
                "top_score": float(chosen.rank_score or 0.0),
                "params_b": float(chosen.parameters_b or 0.0),
            },
            artifacts=[
                {
                    "kind": "model_selection",
                    "name": "model_selection.json",
                    "path": "model_selection.json",
                    "meta": {"selected": chosen.id},
                }
            ],
        )

    def _rejection(self, shortlist: list[ModelCandidate], params: dict[str, Any]) -> StageResult:
        """Record a refusal without advancing: a rejected recommendation is a
        decision, and the StageRun row keeps it even though the project waits."""
        notes = params.get("decision_notes")
        selection = ModelSelection(
            selected=None,
            shortlist=shortlist,
            compared_on=compared_on(),
            decision_notes=str(notes) if notes else "Recommendation rejected without notes.",
            requires_approval=True,
        )
        return StageResult(
            output=selection.model_dump(mode="json"),
            summary=f"Recommendation rejected: {selection.decision_notes}",
            metrics={"candidates": float(len(shortlist))},
            needs_approval=True,
            blocked_on=["approve_model_choice"],
        )


def build_estimate(candidate: ModelCandidate) -> ResourceEstimate:
    """Pre-flight footprint for the chosen candidate (Stage 5 refines this)."""
    return ResourceEstimate(
        vram_gb=candidate.finetune.min_vram_gb,
        confidence="medium",
        notes="QLoRA training estimate from Stage 3; Stage 5 refines it per dataset.",
    )


__all__ = [
    "CATALOG",
    "ModelSelectionStage",
    "apply_locality",
    "build_estimate",
    "catalog_candidates",
    "compared_on",
    "discovery_queries",
    "estimate_qlora_gb",
    "finetune_support",
    "fits_budget",
    "hub_leads",
    "local_hf_models",
    "max_params_for_budget",
    "normalise_license",
    "ollama_inventory",
    "rank_candidates",
    "score_candidate",
]
