"""Hugging Face CLI (`hf`) integration — the primary Hub interface.

We drive the official CLI rather than hand-rolling Hub REST calls because it
already knows how to do parameter filtering, parquet URL discovery, resumable
uploads, managed Jobs and Inference Endpoint lifecycle — and it carries its own
authentication (``hf auth login``), so the app never has to hold the token itself.

Every call goes through `net.retry_call`: this network drops TLS handshakes
intermittently, and a transient reset must not read as a product failure.

`hf` 1.8.0 replaced the deprecated `huggingface-cli`; auth now lives under
`hf auth <cmd>`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..log import get_logger
from ..net import TransientNetworkError, retry_call

logger = get_logger("integrations.hf_cli")

DEFAULT_TIMEOUT_S = 120


class HFCLIError(RuntimeError):
    """The `hf` CLI is missing, or returned a non-zero exit status."""


@dataclass
class CLIResult:
    returncode: int
    stdout: str
    stderr: str


def hf_path() -> str | None:
    """Locate the `hf` executable, or None when it is not installed."""
    return shutil.which("hf")


def is_available() -> bool:
    """True when the `hf` CLI is on PATH."""
    return hf_path() is not None


def run(
    args: list[str],
    *,
    timeout: int = DEFAULT_TIMEOUT_S,
    check: bool = True,
    attempts: int = 4,
) -> CLIResult:
    """Run an `hf` subcommand with retries, returning its captured output.

    `hf` writes some failures to stderr with a zero exit status on Windows, so
    both the exit code and known error markers are inspected.
    """
    binary = hf_path()
    if binary is None:
        raise HFCLIError("The `hf` CLI was not found on PATH. Install it from https://hf.co/cli.")

    def _once() -> CLIResult:
        completed = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        result = CLIResult(completed.returncode, completed.stdout, completed.stderr)
        combined = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0 or "ConnectError" in combined or "SSL connect error" in combined:
            raise HFCLIError(_condense(combined))
        return result

    try:
        return retry_call(_once, attempts=attempts, description=f"hf {' '.join(args[:2])}")
    except TransientNetworkError:
        raise
    except HFCLIError:
        if check:
            raise
        return CLIResult(1, "", "")


def _condense(text: str, limit: int = 300) -> str:
    """Reduce CLI output to the line that actually explains the failure.

    Raw tracebacks flood the logs with frames and make a transient reset look like
    a crash, so keep the final exception line instead of the whole dump.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        if "rror:" in line:
            return line[:limit]
    return (lines[-1] if lines else "hf command failed")[:limit]


def run_json(args: list[str], **kwargs: Any) -> Any:
    """Run an `hf` subcommand and parse its output as JSON.

    The CLI is inconsistent, and guessing which commands behave how is how this
    breaks: `datasets parquet` prints a **table** unless told `--format json`,,
    while `datasets info` already emits JSON and **rejects** the flag with "No such
    option". So: prefer the output as given, ask for JSON only when it will not
    parse, and if the flag itself is refused, fall back to the plain command once
    more before reporting a parse failure with the output we actually saw.
    """
    if "--format" in args:
        return _parse_json(run(args, **kwargs), args)

    plain = run(args, **kwargs)
    text = plain.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return _parse_json(run([*args, "--format", "json"], **kwargs), args)
    except HFCLIError as exc:
        # Either the flag is unsupported here, or the command genuinely failed; the
        # plain re-read distinguishes the two and reports the useful output.
        logger.debug("`--format json` unavailable for `hf %s`: %s", " ".join(args), exc)
    return _parse_json(run(args, **kwargs), args)


def _parse_json(result: CLIResult, args: list[str]) -> Any:
    text = result.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HFCLIError(f"Expected JSON from `hf {' '.join(args)}`, got: {text[:200]}") from exc


# --- authentication ----------------------------------------------------


def whoami() -> dict[str, Any]:
    """Return the authenticated account, or {} when not logged in."""
    try:
        data = run_json(["auth", "whoami"], check=False)
    except (HFCLIError, TransientNetworkError) as exc:
        logger.info("hf auth whoami unavailable: %s", exc)
        return {}
    if isinstance(data, list):  # a list of tokens rather than a user object
        return {}
    return data or {}


def is_authenticated() -> bool:
    return bool(whoami().get("name"))


# --- discovery ---------------------------------------------------------


def param_range(min_b: float | None = None, max_b: float | None = None) -> str | None:
    """Build the CLI's parameter-range syntax, e.g. `min:0.4B,max:1B`.

    The CLI does *not* accept a hyphenated range (`0.4B-1B`); it silently returns
    an empty list, which is a nasty way to lose a whole candidate shortlist.
    """
    parts: list[str] = []
    if min_b is not None:
        parts.append(f"min:{min_b:g}B")
    if max_b is not None:
        parts.append(f"max:{max_b:g}B")
    return ",".join(parts) if parts else None


def list_models(
    search: str | None = None,
    *,
    min_params_b: float | None = None,
    max_params_b: float | None = None,
    author: str | None = None,
    pipeline_tag: str | None = None,
    sort: str = "downloads",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search models, optionally bounded by parameter count.

    Delegating the size filter to the CLI beats parsing parameter counts out of
    repo names, which is what we had to do before.
    """
    args = ["models", "list", "--sort", sort, "--limit", str(limit)]
    if search:
        args += ["--search", search]
    if bounds := param_range(min_params_b, max_params_b):
        args += ["--num-parameters", bounds]
    if author:
        args += ["--author", author]
    if pipeline_tag:
        args += ["--filter", pipeline_tag]
    data = run_json(args)
    if isinstance(data, dict):
        return data.get("models", []) or []
    return data or []


def model_info(model_id: str, *, expand: str | None = None) -> dict[str, Any]:
    """Fetch metadata for one model (licence, tags, library, downloads).

    `hf models info` emits JSON natively and does not accept `--revision`/`--format`
    beyond what its own help lists, so keep the surface minimal.
    """
    args = ["models", "info", model_id]
    if expand:
        args += ["--expand", expand]
    data = run_json(args)
    return data if isinstance(data, dict) else {}


def list_datasets(search: str | None = None, *, limit: int = 20) -> list[dict[str, Any]]:
    """Search datasets on the Hub."""
    args = ["datasets", "list", "--sort", "downloads", "--limit", str(limit)]
    if search:
        args += ["--search", search]
    data = run_json(args)
    if isinstance(data, dict):
        return data.get("datasets", []) or []
    return data or []


def dataset_info(dataset_id: str) -> dict[str, Any]:
    """Fetch dataset metadata: gated flag, licence, size, siblings, sha.

    Note the response uses **snake_case** keys (`card_data`, not `cardData`), unlike
    the `models list` payload — read fields defensively rather than assuming shape.
    """
    args = ["datasets", "info", dataset_id, "--expand", _DATASET_INFO_FIELDS]
    data = run_json(args)
    return data if isinstance(data, dict) else {}


#: `--expand` is optional, but asking for everything we use avoids a second call.
_DATASET_INFO_FIELDS = "downloads,likes,tags,gated,cardData,sha,lastModified,siblings"


def dataset_license(dataset_id: str) -> str | None:
    """The dataset's declared licence, from `card_data` or the `license:` tag.

    A Hub dataset commonly declares its licence in one of two places and we must not
    guess when neither is present: `None` means *unknown*, which is a verdict the
    caller has to handle, not a licence we may assume.
    """
    info = dataset_info(dataset_id)
    card = info.get("card_data") or info.get("cardData") or {}
    if isinstance(card, dict):
        licence = card.get("license")
        if isinstance(licence, list):
            licence = licence[0] if licence else None
        if licence:
            return str(licence)
    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def dataset_parquet_urls(
    dataset_id: str, *, subset: str = "default", split: str = "train"
) -> list[str]:
    """Return direct parquet URLs for a dataset config/split.

    Available for any dataset the Hub has auto-converted to parquet — which is
    most of them — and exactly what enables remote querying without a download.
    """
    data = run_json(["datasets", "parquet", dataset_id, "--subset", subset, "--split", split])
    rows = data if isinstance(data, list) else (data or {}).get("parquet_files", [])
    return [row["url"] for row in rows if isinstance(row, dict) and row.get("url")]


# --- Jobs (managed, ephemeral containers) ------------------------------


def jobs_hardware() -> list[dict[str, Any]]:
    """List available Job hardware flavors with their cost per hour."""
    data = run_json(["jobs", "hardware"])
    return data if isinstance(data, list) else (data or {}).get("flavors", []) or []


def jobs_ps(*, all_: bool = False) -> list[dict[str, Any]]:
    """List this account's Jobs."""
    args = ["jobs", "ps"]
    if all_:
        args.append("--all")
    data = run_json(args)
    return data if isinstance(data, list) else (data or {}).get("jobs", []) or []


def jobs_run(
    image: str,
    command: list[str],
    *,
    flavor: str = "cpu-basic",
    timeout: str | None = None,
    env: dict[str, str] | None = None,
    secrets: dict[str, str] | None = None,
    detach: bool = True,
    namespace: str | None = None,
) -> str:
    """Launch a managed Job and return its id.

    Jobs bill per minute, so this is only ever reached from an approved,
    estimated, user-authorised stage — never automatically.
    """
    args = ["jobs", "run", image, *command, "--flavor", flavor]
    if timeout:
        args += ["--timeout", timeout]
    if detach:
        args.append("--detach")
    if namespace:
        args += ["--namespace", namespace]
    for key, value in (env or {}).items():
        args += ["--env", f"{key}={value}"]
    for key, value in (secrets or {}).items():
        args += ["--secrets", f"{key}={value}"]
    result = run(args)
    return result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""


def jobs_logs(job_id: str, *, tail: int | None = None, follow: bool = False) -> str:
    """Fetch (or stream) a Job's logs."""
    args = ["jobs", "logs", job_id]
    if tail is not None:
        args += ["--tail", str(tail)]
    if follow:
        args.append("--follow")
    return run(args).stdout


def jobs_inspect(job_id: str) -> dict[str, Any]:
    """Detailed Job state, including exit status."""
    data = run_json(["jobs", "inspect", job_id])
    if isinstance(data, list):
        return data[0] if data else {}
    return data or {}


def jobs_cancel(job_id: str) -> None:
    """Cancel a running Job."""
    run(["jobs", "cancel", job_id])


# --- Inference Endpoints ----------------------------------------------


def endpoints_list() -> list[dict[str, Any]]:
    """List Inference Endpoints for the authenticated account."""
    data = run_json(["endpoints", "list"])
    return data if isinstance(data, list) else (data or {}).get("endpoints", []) or []


def endpoints_deploy(
    name: str,
    repo_id: str,
    *,
    framework: str = "pytorch",
    task: str = "text-generation",
    accelerator: str = "cpu",
    instance_size: str = "x2",
    instance_type: str = "intel-cpu",
    region: str = "us-east-1",
    vendor: str = "aws",
    min_replica: int = 1,
    max_replica: int = 1,
    scale_to_zero_timeout: int | None = None,
    namespace: str | None = None,
) -> None:
    """Create an Inference Endpoint for a Hub repository."""
    args = [
        "endpoints",
        "deploy",
        name,
        "--repo",
        repo_id,
        "--framework",
        framework,
        "--task",
        task,
        "--accelerator",
        accelerator,
        "--instance-size",
        instance_size,
        "--instance-type",
        instance_type,
        "--region",
        region,
        "--vendor",
        vendor,
        "--min-replica",
        str(min_replica),
        "--max-replica",
        str(max_replica),
    ]
    if scale_to_zero_timeout:
        args += ["--scale-to-zero-timeout", str(scale_to_zero_timeout)]
    if namespace:
        args += ["--namespace", namespace]
    run(args)


def endpoints_describe(name: str, *, namespace: str | None = None) -> dict[str, Any] | None:
    """Describe one Inference Endpoint (URL, status, replica counts)."""
    args = ["endpoints", "describe", name]
    if namespace:
        args += ["--namespace", namespace]
    data = run_json(args, check=False)
    return data if isinstance(data, dict) else None


def endpoints_pause(name: str, *, namespace: str | None = None) -> None:
    """Pause an endpoint (stops compute billing, keeps configuration)."""
    run(["endpoints", "pause", name] + (["--namespace", namespace] if namespace else []))


def endpoints_resume(name: str, *, namespace: str | None = None) -> None:
    """Resume a paused endpoint."""
    run(["endpoints", "resume", name] + (["--namespace", namespace] if namespace else []))


def endpoints_delete(name: str, *, namespace: str | None = None) -> None:
    """Delete an endpoint permanently."""
    run(["endpoints", "delete", name, "--yes"] + (["--namespace", namespace] if namespace else []))


# --- transfer & cache --------------------------------------------------


def download(
    repo_id: str,
    *,
    repo_type: str = "model",
    include: list[str] | None = None,
    local_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> str:
    """Download a repository (optionally a subset) and return the target path."""
    args = ["download", repo_id, "--type", repo_type]
    if local_dir:
        args += ["--local-dir", str(local_dir)]
    if cache_dir:
        args += ["--cache-dir", str(cache_dir)]
    for pattern in include or []:
        args += ["--include", pattern]
    return run(args, timeout=3600).stdout.strip()


def upload(
    repo_id: str,
    local_path: Path,
    *,
    repo_type: str = "model",
    private: bool = True,
    commit_message: str | None = None,
    create_pr: bool = False,
) -> None:
    """Upload a file or folder (single commit). Prefer `upload_large_folder` for size."""
    args = ["upload", repo_id, str(local_path), "--type", repo_type]
    if private:
        args.append("--private")
    if create_pr:
        args.append("--create-pr")
    if commit_message:
        args += ["--commit-message", commit_message]
    run(args, timeout=3600)


def upload_large_folder(
    repo_id: str, local_path: Path, *, repo_type: str = "model", private: bool = True
) -> None:
    """Resumable upload for large folders (checkpoints, merged weights)."""
    args = ["upload-large-folder", repo_id, str(local_path), "--type", repo_type]
    if private:
        args.append("--private")
    run(args, timeout=7200)


def cache_list() -> list[dict[str, Any]]:
    """List cached repositories, for the disk-usage view and cache pruning."""
    data = run_json(["cache", "list"])
    return data if isinstance(data, list) else (data or {}).get("repos", []) or []


def cache_rm(targets: list[str], *, dry_run: bool = True) -> None:
    """Remove cached repositories. Defaults to a dry run."""
    args = ["cache", "rm", *targets]
    if dry_run:
        args.append("--dry-run")
    run(args)
