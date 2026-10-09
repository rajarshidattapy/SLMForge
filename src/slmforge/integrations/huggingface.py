"""Hugging Face Hub integration.

Covers the three HF touch-points in the workflow:
1. **Discovery** — search models and datasets for candidates.
2. **Materialization** — download datasets/models for local work.
3. **Publishing** — push fine-tuned adapters/models and model cards.

`huggingface_hub` is imported lazily so the API can boot without the ML extra
installed; only the functions that actually need Hub access import it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import settings
from ..log import get_logger

logger = get_logger("integrations.huggingface")

#: Licenses that are generally safe for commercial use without further review.
COMMERCIAL_FRIENDLY = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "openrail",
    "openrail++",
    "llama3.1",
    "llama3.2",
    "llama3",
}

#: Licenses that need explicit review or are non-commercial / share-alike.
REVIEW_REQUIRED = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-sa-4.0", "gemma", "other", "unknown"}


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=settings.hf_token or None)


def is_commercial_friendly(license_id: str | None) -> bool | None:
    """Classify a license id; None means "needs human review"."""
    if not license_id:
        return None
    normalized = license_id.strip().lower()
    if normalized in COMMERCIAL_FRIENDLY:
        return True
    if normalized in REVIEW_REQUIRED:
        return False
    return None


def search_models(
    query: str,
    *,
    limit: int = 20,
    task: str | None = None,
    max_params_b: float | None = None,
) -> list[dict[str, Any]]:
    """Search the Hub for candidate base models.

    Returns normalized dicts matching `schemas.ModelCandidate` fields where known.
    `max_params_b` filters by the parameter count encoded in the repo id
    (e.g. `-0.5B-`, `-1B-`, `-3B-`) since the Hub API does not expose it directly.
    """
    api = _api()
    results = api.list_models(
        search=query,
        limit=limit,
        sort="downloads",
        direction=-1,
        filter=task,
        full=False,
        cardData=False,
    )
    candidates: list[dict[str, Any]] = []
    for model in results:
        params_b = parse_params_billions(model.id)
        if max_params_b and params_b and params_b > max_params_b:
            continue
        candidates.append(
            {
                "id": f"hf:{model.id}",
                "name": model.id,
                "source": "hf_hub",
                "parameters_b": params_b,
                "downloads": getattr(model, "downloads", None),
                "pipeline_tag": getattr(model, "pipeline_tag", None),
                "tags": list(getattr(model, "tags", []) or []),
            }
        )
    return candidates


def model_metadata(repo_id: str) -> dict[str, Any]:
    """Fetch license and size metadata for a model repo."""
    api = _api()
    info = api.model_info(repo_id, files_metadata=False, cardData=True)
    card = info.cardData or {}
    license_id = card.get("license") or next(
        (t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None
    )
    return {
        "repo_id": info.id,
        "sha": info.sha,
        "license": license_id,
        "commercial_ok": is_commercial_friendly(license_id),
        "library": getattr(info, "library_name", None),
        "tags": list(info.tags or []),
        "downloads": getattr(info, "downloads", None),
    }


def search_datasets(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """Search the Hub for candidate datasets."""
    api = _api()
    results = api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1)
    datasets: list[dict[str, Any]] = []
    for dataset in results:
        datasets.append(
            {
                "id": f"hf:{dataset.id}",
                "name": dataset.id,
                "source": "hf_dataset",
                "downloads": getattr(dataset, "downloads", None),
                "tags": list(getattr(dataset, "tags", []) or []),
            }
        )
    return datasets


def download_dataset(repo_id: str, dest: Path, *, split: str | None = None) -> Path:
    """Download a dataset snapshot into `dest` and return the path."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=None
        if not split
        else [f"*{split}*", "*.jsonl", "*.csv", "*.parquet", "*.md"],
        token=settings.hf_token or None,
    )
    return Path(path)


def push_folder(
    folder: Path,
    repo_id: str,
    *,
    private: bool = True,
    repo_type: str = "model",
    commit_message: str = "SLMForge: publish fine-tuned model",
) -> str:
    """Upload a local folder to the Hub and return the commit URL/oid."""
    api = _api()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    return api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )


def upload_model_card(folder: Path, card_markdown: str) -> Path:
    """Write a model card README into the artifact folder before pushing."""
    path = Path(folder) / "README.md"
    path.write_text(card_markdown, encoding="utf-8")
    return path


def parse_params_billions(repo_id: str) -> float | None:
    """Best-effort parse of a parameter count from a model id (e.g. 'Qwen2.5-1.5B')."""
    import re

    match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*[bB](?![a-z0-9])", repo_id)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:  # pragma: no cover - regex already guarantees a number
        return None
