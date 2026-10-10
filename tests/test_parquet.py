"""DuckDB dataset profiling.

Runs entirely on a tiny local parquet file, so these tests are fast, offline and
deterministic. DuckDB is optional at runtime, so the whole module skips cleanly
when it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slmforge.integrations import parquet

pytest.importorskip("duckdb")

TICKETS = [
    ("t1", "card declined", "Billing"),
    ("t2", "cannot log in", "Account Access"),
    ("t3", "card declined again", "Billing"),
    ("t4", "server returns 500", "Technical"),
    ("t5", "refund please", "Billing"),
]


@pytest.fixture(scope="module")
def tickets_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the fixture rows to parquet using DuckDB itself."""
    import duckdb

    path = tmp_path_factory.mktemp("data") / "tickets.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE t (id VARCHAR, body VARCHAR, queue VARCHAR);")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", TICKETS)
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()
    return path


def test_schema_reports_column_names_and_types(tickets_parquet: Path) -> None:
    cols = parquet.schema(str(tickets_parquet))
    names = [c["name"] for c in cols]
    assert names == ["id", "body", "queue"]
    assert all(c["type"] == "VARCHAR" for c in cols)


def test_row_count(tickets_parquet: Path) -> None:
    assert parquet.row_count(str(tickets_parquet)) == len(TICKETS)


def test_value_counts_gives_the_label_distribution(tickets_parquet: Path) -> None:
    counts = parquet.value_counts(str(tickets_parquet), "queue")
    assert counts == {"Billing": 3, "Account Access": 1, "Technical": 1}


def test_profile_summarises_a_dataset(tickets_parquet: Path) -> None:
    result = parquet.profile(str(tickets_parquet), label_column="queue")
    assert result["rows"] == 5
    assert "queue" in result["column_names"]
    assert result["label_distribution"]["Billing"] == 3


def test_profile_skips_an_unknown_label_column(tickets_parquet: Path) -> None:
    result = parquet.profile(str(tickets_parquet), label_column="does_not_exist")
    assert "label_distribution" not in result


def test_materialize_filters_rows_and_writes_parquet(tickets_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "billing.parquet"
    written = parquet.materialize(
        str(tickets_parquet),
        "SELECT * FROM data WHERE queue = 'Billing'",
        out,
        fmt="parquet",
    )
    assert written == out and out.exists()

    rows = parquet.query(str(out), "SELECT COUNT(*) FROM data")
    assert rows[0][0] == 3


def test_materialize_supports_jsonl_output(tickets_parquet: Path, tmp_path: Path) -> None:
    """Training splits are written as JSON, so the format matters."""
    out = tmp_path / "split.json"
    parquet.materialize(str(tickets_parquet), "SELECT * FROM data", out, fmt="json")
    assert out.exists()
    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == len(TICKETS)
    assert "Billing" in content[0]


def test_query_can_join_and_aggregate(tickets_parquet: Path) -> None:
    rows = parquet.query(
        str(tickets_parquet),
        "SELECT queue, COUNT(*) AS n FROM data GROUP BY queue ORDER BY n DESC",
    )
    assert rows[0] == ("Billing", 3)


def test_missing_duckdb_raises_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message must tell the user how to fix it, not just that it failed."""
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args, **kwargs):
        if name == "duckdb":
            raise ImportError("No module named 'duckdb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(parquet.DuckDBUnavailable, match="uv pip install duckdb"):
        parquet.schema("/nonexistent.parquet")
