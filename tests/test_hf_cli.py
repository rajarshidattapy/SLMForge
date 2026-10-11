"""The `hf` CLI wrapper.

These tests never touch the Hub: `hf_path` is monkeypatched to the current Python
interpreter, so the wrapper's argv handling, JSON parsing, retry and error
surfacing are exercised against a controllable stub.
"""

from __future__ import annotations

import sys

import pytest

from slmforge.integrations import hf_cli
from slmforge.integrations.hf_cli import HFCLIError
from slmforge.net import TransientNetworkError


@pytest.fixture()
def stub_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `hf` resolve to this interpreter, so args are just Python argv."""
    monkeypatch.setattr(hf_cli, "hf_path", lambda: sys.executable)


def test_missing_cli_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_cli, "hf_path", lambda: None)
    assert hf_cli.is_available() is False
    with pytest.raises(HFCLIError, match="not found on PATH"):
        hf_cli.run(["auth", "whoami"])


def test_available_detects_the_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_cli, "hf_path", lambda: "/usr/bin/hf")
    assert hf_cli.is_available() is True


def test_run_parses_stdout(stub_python: None) -> None:
    result = hf_cli.run(["-c", "print('hello from hf')"])
    assert result.returncode == 0
    assert "hello from hf" in result.stdout


def test_run_json_parses_a_model_list(stub_python: None) -> None:
    output = '[{"id": "Qwen/Qwen2.5-0.5B-Instruct", "downloads": 42}]'
    data = hf_cli.run_json(["-c", f"print({output!r})"])
    assert data[0]["id"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_list_models_unwraps_a_dict_payload(monkeypatch) -> None:
    """The CLI sometimes returns {"models": [...]} instead of a bare list."""
    monkeypatch.setattr(hf_cli, "run_json", lambda args, **kw: {"models": [{"id": "a/b"}]})
    assert hf_cli.list_models("qwen") == [{"id": "a/b"}]


def test_list_models_passes_discovery_filters(monkeypatch) -> None:
    """Parameter filtering must go to the CLI rather than being re-implemented."""
    captured: dict[str, list[str]] = {}

    def fake_run_json(args, **kwargs):
        captured["args"] = list(args)
        return []

    monkeypatch.setattr(hf_cli, "run_json", fake_run_json)
    hf_cli.list_models(
        "qwen", min_params_b=0.4, max_params_b=1.0, pipeline_tag="text-generation", limit=7
    )
    args = captured["args"]
    assert "--num-parameters" in args
    # The CLI needs `min:X,max:Y`; a hyphenated range silently returns nothing.
    assert "min:0.4B,max:1B" in args
    assert "--filter" in args and "text-generation" in args
    assert "--limit" in args and "7" in args
    assert "--search" in args and "qwen" in args


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.4, 1.0, "min:0.4B,max:1B"),
        (None, 3.0, "max:3B"),
        (0.5, None, "min:0.5B"),
        (None, None, None),
    ],
)
def test_param_range_builds_cli_syntax(
    low: float | None, high: float | None, expected: str | None
) -> None:
    assert hf_cli.param_range(low, high) == expected


def test_run_json_falls_back_to_format_flag_for_list_commands(monkeypatch) -> None:
    """`list` needs --format json, but `info` rejects it — handle both."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if "--format" in args:
            return hf_cli.CLIResult(0, '[{"id": "a/b"}]', "")
        return hf_cli.CLIResult(0, "ID    DOWNLOADS\na/b   5", "")

    monkeypatch.setattr(hf_cli, "run", fake_run)
    assert hf_cli.run_json(["models", "list"]) == [{"id": "a/b"}]
    assert len(calls) == 2 and "--format" in calls[1]


def test_run_json_does_not_add_format_when_info_already_returns_json(monkeypatch) -> None:
    """`hf datasets info` rejects --format, so the plain call must be used."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return hf_cli.CLIResult(0, '{"id": "org/ds"}', "")

    monkeypatch.setattr(hf_cli, "run", fake_run)
    assert hf_cli.run_json(["datasets", "info", "org/ds"]) == {"id": "org/ds"}
    assert len(calls) == 1
    assert "--format" not in calls[0]


def test_run_json_asks_for_json_when_the_command_prints_a_table(monkeypatch) -> None:
    """`datasets parquet` defaults to a table: verified against the real CLI.

    Getting this wrong breaks every dataset build, so it is pinned by a test rather
    than left to a command whitelist that silently drifts.
    """
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if "--format" in args:
            return hf_cli.CLIResult(0, '[{"url": "https://x/0000.parquet"}]', "")
        return hf_cli.CLIResult(0, "SUBSET  SPLIT URL  SIZE\ndefault train https://x 1", "")

    monkeypatch.setattr(hf_cli, "run", fake_run)
    urls = hf_cli.dataset_parquet_urls("org/ds")
    assert urls == ["https://x/0000.parquet"]
    assert ["--format" in args for args in calls] == [False, True]


def test_run_json_still_reports_a_real_failure_from_the_plain_command(monkeypatch) -> None:
    """A refused flag must not hide the actual error output."""

    def fake_run(args, **kwargs):
        if "--format" in args:
            raise hf_cli.HFCLIError("Error: No such option: --format")
        return hf_cli.CLIResult(0, "no json here", "")

    monkeypatch.setattr(hf_cli, "run", fake_run)
    with pytest.raises(hf_cli.HFCLIError, match="no json here"):
        hf_cli.run_json(["datasets", "info", "org/ds"])


def test_dataset_info_reads_the_snake_case_licence(monkeypatch) -> None:
    """`datasets info` returns `card_data`, not `cardData` — verified live."""
    monkeypatch.setattr(
        hf_cli,
        "dataset_info",
        lambda dataset_id: {"id": dataset_id, "card_data": {}, "tags": ["license:cc-by-nc-4.0"]},
    )
    assert hf_cli.dataset_license("org/ds") == "cc-by-nc-4.0"

    monkeypatch.setattr(
        hf_cli,
        "dataset_info",
        lambda dataset_id: {"id": dataset_id, "card_data": {"license": "apache-2.0"}},
    )
    assert hf_cli.dataset_license("org/ds") == "apache-2.0"

    monkeypatch.setattr(hf_cli, "dataset_info", lambda dataset_id: {"id": dataset_id})
    assert hf_cli.dataset_license("org/ds") is None, "unknown must not read as licensed"


def test_dataset_parquet_urls_extracts_urls(monkeypatch) -> None:
    monkeypatch.setattr(
        hf_cli,
        "run_json",
        lambda args, **kw: [{"url": "https://example.test/0000.parquet", "size": 1}],
    )
    urls = hf_cli.dataset_parquet_urls("org/ds")
    assert urls == ["https://example.test/0000.parquet"]


def test_run_json_rejects_non_json_output(stub_python: None) -> None:
    with pytest.raises(HFCLIError, match="Expected JSON"):
        hf_cli.run_json(["-c", "print('not json at all')"])


def test_nonzero_exit_is_surfaced_with_output(stub_python: None) -> None:
    with pytest.raises(HFCLIError) as excinfo:
        hf_cli.run(["-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    assert "boom" in str(excinfo.value)


def test_condense_keeps_the_error_line_and_drops_frames() -> None:
    """A transient reset must not dump a 20-line traceback into the logs."""
    raw = '  File "x.py", line 1\n    yield\nhttpx.ConnectError: forcibly closed'
    condensed = hf_cli._condense(raw)
    assert condensed == "httpx.ConnectError: forcibly closed"
    assert "File" not in condensed


def test_transport_failure_is_retried_then_reported(stub_python: None, monkeypatch) -> None:
    """A reset-looking failure must be retried and then reported as transient."""
    monkeypatch.setattr("slmforge.net.time.sleep", lambda _s: None)
    script = (
        "import sys; sys.stderr.write("
        "'httpx.ConnectError: An existing connection was forcibly closed'); sys.exit(1)"
    )
    with pytest.raises(TransientNetworkError) as excinfo:
        hf_cli.run(["-c", script], attempts=3)
    assert excinfo.value.attempts == 3


def test_check_false_swallows_failures(stub_python: None) -> None:
    result = hf_cli.run(["-c", "raise SystemExit(1)"], check=False)
    assert result.returncode == 1


def test_authenticated_is_false_when_whoami_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(hf_cli, "whoami", lambda: {})
    assert hf_cli.is_authenticated() is False


def test_whoami_extracts_the_account(monkeypatch) -> None:
    monkeypatch.setattr(hf_cli, "whoami", lambda: {"name": "someone"})
    assert hf_cli.is_authenticated() is True


def test_jobs_run_builds_the_expected_argv(stub_python: None, monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        return hf_cli.CLIResult(0, "job-123\n", "")

    monkeypatch.setattr(hf_cli, "run", fake_run)
    job_id = hf_cli.jobs_run(
        "python:3.12-slim",
        ["python", "train.py"],
        flavor="cpu-basic",
        timeout="1h",
        env={"HF_HOME": "/tmp/hf"},
    )
    assert job_id == "job-123"
    argv = captured["args"]
    assert argv[:2] == ["jobs", "run"]
    assert "--flavor" in argv and "cpu-basic" in argv
    assert "--timeout" in argv and "1h" in argv
    assert "HF_HOME=/tmp/hf" in argv
    assert "--detach" in argv
