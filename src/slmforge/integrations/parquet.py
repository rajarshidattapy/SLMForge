"""Dataset profiling and extraction with DuckDB.

Datasets on the Hub are usually published as parquet, so we can inspect and filter
them with SQL *before* deciding to download anything. That matters here: the demo
machine has limited free disk, and dataset sizes range from megabytes to hundreds
of gigabytes.

The strategy is deliberately two-phase:

1. **Profile remotely** — schema, row counts and label distributions straight from
   the parquet URL. No download.
2. **Materialise once** — run one filtering/normalising query that writes the rows
   we actually need to a local file. Never re-query the remote copy in a loop;
   the Hub throttles that (and the reference network resets TLS under load).

DuckDB is imported lazily: the API and worker boot without it, and only dataset
stages need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..log import get_logger
from ..net import TransientNetworkError, retry_call

logger = get_logger("integrations.parquet")


class DuckDBUnavailable(RuntimeError):
    """DuckDB is not installed."""


def _connect(threads: int | None = None):
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DuckDBUnavailable(
            "DuckDB is required for dataset profiling. Install it with `uv pip install duckdb`."
        ) from exc

    # An in-memory connection keeps no state on disk; httpfs enables remote reads.
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    # `INSTALL` needs network; `LOAD` alone works when the extension is cached, so
    # tolerate a failed install and rely on the cached copy.
    try:
        con.execute("INSTALL httpfs")
    except Exception as exc:
        logger.debug("duckdb httpfs install skipped: %s", exc)
    con.execute("LOAD httpfs")
    return con


def _source(urls: list[str] | str) -> str:
    """Build a read_parquet() target, escaping quotes in URLs."""
    if isinstance(urls, str):
        urls = [urls]
    quoted = ", ".join("'" + u.replace("'", "''") + "'" for u in urls)
    return f"read_parquet([{quoted}], union_by_name=true)"


def query(
    urls: list[str] | str,
    sql: str,
    *,
    attempts: int = 8,
    threads: int | None = None,
    description: str = "remote parquet query",
) -> list[tuple]:
    """Run SQL against parquet files (remote URLs or local paths), with retries.

    `sql` may reference the data as `data`, e.g.
    ``SELECT queue, COUNT(*) FROM data GROUP BY queue``.
    """
    source = _source(urls)

    def _once() -> list[tuple]:
        con = _connect(threads=threads)
        try:
            return con.execute(f"WITH data AS (SELECT * FROM {source}) {sql}").fetchall()
        finally:
            con.close()

    return retry_call(_once, attempts=attempts, description=description, base_delay=2.0)


def schema(urls: list[str] | str, *, attempts: int = 8) -> list[dict[str, str]]:
    """Column names and types."""
    source = _source(urls)

    def _once() -> list[dict[str, str]]:
        con = _connect()
        try:
            rows = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        finally:
            con.close()
        return [{"name": r[0], "type": r[1]} for r in rows]

    return retry_call(_once, attempts=attempts, description="parquet schema")


def row_count(urls: list[str] | str, *, attempts: int = 8, threads: int | None = None) -> int:
    """Total row count."""
    rows = query(urls, "SELECT COUNT(*) FROM data", attempts=attempts, threads=threads)
    return int(rows[0][0]) if rows else 0


def value_counts(
    urls: list[str] | str,
    column: str,
    *,
    limit: int = 50,
    attempts: int = 8,
    threads: int | None = None,
) -> dict[str, int]:
    """Distribution of a column, used for label mapping and class balance."""
    safe = column.replace('"', "")
    sql = (
        f'SELECT CAST("{safe}" AS VARCHAR) AS v, COUNT(*) AS n '
        f"FROM data GROUP BY 1 ORDER BY n DESC LIMIT {int(limit)}"
    )
    rows = query(urls, sql, attempts=attempts, threads=threads)
    return {(str(r[0]) if r[0] is not None else ""): int(r[1]) for r in rows}


def profile(
    urls: list[str] | str,
    *,
    label_column: str | None = None,
    attempts: int = 8,
    threads: int | None = None,
) -> dict[str, Any]:
    """A compact, reviewable profile of a dataset: shape, columns and labels."""
    cols = schema(urls, attempts=attempts)
    total = row_count(urls, attempts=attempts, threads=threads)
    result: dict[str, Any] = {
        "rows": total,
        "columns": cols,
        "column_names": [c["name"] for c in cols],
    }
    if label_column and any(c["name"] == label_column for c in cols):
        result["label_distribution"] = value_counts(
            urls, label_column, attempts=attempts, threads=threads
        )
    return result


def materialize(
    urls: list[str] | str,
    sql: str,
    out_path: Path,
    *,
    fmt: str = "parquet",
    attempts: int = 8,
    threads: int | None = None,
    description: str = "dataset materialization",
) -> Path:
    """Run one query and write the result locally.

    This is the single download step for a dataset: select and normalise in SQL,
    then persist only the rows we keep.
    """
    source = _source(urls)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = out_path.as_posix().replace("'", "''")

    options = {
        "parquet": "FORMAT PARQUET",
        "csv": "FORMAT CSV, HEADER",
        "json": "FORMAT JSON",
    }[fmt]

    # The CTE must live *inside* the COPY subquery: DuckDB rejects
    # `WITH ... COPY (...)` but accepts `COPY (WITH ...)`.
    statement = f"COPY (WITH data AS (SELECT * FROM {source}) {sql}) TO '{target}' ({options})"

    def _once() -> Path:
        # A retry must start from a clean file: DuckDB refuses to overwrite by
        # default, and a half-written file from the previous attempt would be
        # silently read back as a successful download. Retries are expected here
        # (this network drops TLS handshakes), so this is the normal path.
        out_path.unlink(missing_ok=True)
        con = _connect(threads=threads)
        try:
            con.execute(statement)
        finally:
            con.close()
        return out_path

    path = retry_call(_once, attempts=attempts, description=description)
    logger.info("Materialized dataset -> %s (%s)", path, fmt)
    return path


__all__ = [
    "DuckDBUnavailable",
    "TransientNetworkError",
    "materialize",
    "profile",
    "query",
    "row_count",
    "schema",
    "value_counts",
]
