"""Sandbox behaviour: exit codes, logs, timeouts and environment scrubbing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from slmforge.sandbox import ResourceLimits, SubprocessSandbox


@pytest.fixture()
def sandbox() -> SubprocessSandbox:
    return SubprocessSandbox()


def test_ok_is_only_true_for_a_clean_exit(sandbox: SubprocessSandbox, tmp_path: Path) -> None:
    result = sandbox.run([sys.executable, "-c", "print('hello')"], work_dir=tmp_path)
    assert result.ok
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.logs_path is not None
    assert "hello" in result.logs_path.read_text(encoding="utf-8")


def test_nonzero_exit_is_reported(sandbox: SubprocessSandbox, tmp_path: Path) -> None:
    result = sandbox.run(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        work_dir=tmp_path,
    )
    assert not result.ok
    assert result.exit_code == 3


def test_missing_executable_is_an_error_not_a_crash(
    sandbox: SubprocessSandbox, tmp_path: Path
) -> None:
    result = sandbox.run(["definitely-not-a-real-binary-xyz"], work_dir=tmp_path)
    assert not result.ok
    assert result.error is not None
    assert "not found" in result.error.lower()


def test_timeout_kills_the_process(sandbox: SubprocessSandbox, tmp_path: Path) -> None:
    result = sandbox.run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        work_dir=tmp_path,
        limits=ResourceLimits(timeout_s=2),
    )
    assert result.timed_out is True
    assert not result.ok
    assert result.duration_s < 30


def test_environment_is_scrubbed(sandbox: SubprocessSandbox, tmp_path: Path) -> None:
    """Secrets must not leak into jobs by default."""
    os.environ["SLMFORGE_TEST_SECRET"] = "super-secret-value"
    try:
        script = "import os; print('LEAK' if os.environ.get('SLMFORGE_TEST_SECRET') else 'CLEAN')"
        result = sandbox.run([sys.executable, "-c", script], work_dir=tmp_path)
        assert result.logs_path is not None
        assert "CLEAN" in result.logs_path.read_text(encoding="utf-8")
    finally:
        os.environ.pop("SLMFORGE_TEST_SECRET", None)


def test_explicit_env_is_forwarded(sandbox: SubprocessSandbox, tmp_path: Path) -> None:
    script = "import os; print('VALUE=' + os.environ.get('JOB_TOKEN', 'missing'))"
    result = sandbox.run(
        [sys.executable, "-c", script], work_dir=tmp_path, env={"JOB_TOKEN": "scoped-123"}
    )
    assert result.logs_path is not None
    assert "VALUE=scoped-123" in result.logs_path.read_text(encoding="utf-8")


def test_output_files_are_listed(sandbox: SubprocessSandbox, tmp_path: Path) -> None:
    script = "open('produced.txt', 'w').write('x')"
    result = sandbox.run([sys.executable, "-c", script], work_dir=tmp_path)
    assert result.ok
    names = {p.name for p in result.output_files}
    assert "produced.txt" in names


def test_write_script_materialises_generated_code(
    sandbox: SubprocessSandbox, tmp_path: Path
) -> None:
    path = sandbox.write_script(tmp_path, "train.py", "print('trained')")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "print('trained')"
