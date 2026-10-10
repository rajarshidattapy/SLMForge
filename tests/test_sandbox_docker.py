"""Docker sandbox isolation tests.

These are the security assertions for tier-1 isolation: network off, non-root,
read-only filesystem, resource ceilings, and cleanup after a timeout. They need a
running Docker daemon, so they skip when one is absent — consistent with AGENTS.md,
which forbids making the suite depend on an external service.

Run them locally with Docker started: `pytest -m docker`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slmforge.sandbox.base import ResourceLimits
from slmforge.sandbox.docker_backend import DockerSandbox, docker_available

pytestmark = pytest.mark.skipif(not docker_available(), reason="Docker daemon is not running")

IMAGE = "python:3.12-slim"
# The command runs *inside* the container, so it must use the container's own
# interpreter. Passing `sys.executable` would reference the host's venv path,
# which does not exist in the image — a classic container-addressing mistake.
PY = "python"
ALLOW_NET = ResourceLimits(timeout_s=120, network="allow", memory_mb=1024, cpu_cores=2)
NO_NET = ResourceLimits(timeout_s=120, network="deny", memory_mb=1024, cpu_cores=2)


@pytest.fixture()
def sandbox() -> DockerSandbox:
    return DockerSandbox(image=IMAGE)


def test_runs_a_command_and_captures_output(sandbox: DockerSandbox, tmp_path: Path) -> None:
    result = sandbox.run(
        [PY, "-c", "print('hello from inside the container')"],
        work_dir=tmp_path,
        limits=ALLOW_NET,
    )
    assert result.ok, result.error
    assert result.exit_code == 0
    assert result.logs_path is not None
    assert "hello from inside the container" in result.logs_path.read_text(encoding="utf-8")


def test_nonzero_exit_is_reported(sandbox: DockerSandbox, tmp_path: Path) -> None:
    result = sandbox.run(
        [PY, "-c", "import sys; sys.exit(7)"],
        work_dir=tmp_path,
        limits=ALLOW_NET,
    )
    assert not result.ok
    assert result.exit_code == 7


def test_network_is_denied_when_policy_is_deny(sandbox: DockerSandbox, tmp_path: Path) -> None:
    """The whole point of the boundary: no exfiltration by default."""
    script = (
        "import socket\n"
        "try:\n"
        "    socket.gethostbyname('huggingface.co')\n"
        "    print('REACHABLE')\n"
        "except Exception as e:\n"
        "    print('BLOCKED')\n"
    )
    result = sandbox.run([PY, "-c", script], work_dir=tmp_path, limits=NO_NET)
    assert result.ok
    # Assert on program output, not the log: the log header echoes the command.
    assert "BLOCKED" in result.stdout
    assert "REACHABLE" not in result.stdout


def test_network_is_allowed_when_policy_is_allow(sandbox: DockerSandbox, tmp_path: Path) -> None:
    script = (
        "import socket\n"
        "try:\n"
        "    print('RESOLVED', socket.gethostbyname('huggingface.co'))\n"
        "except Exception:\n"
        "    print('BLOCKED')\n"
    )
    result = sandbox.run([PY, "-c", script], work_dir=tmp_path, limits=ALLOW_NET)
    assert result.ok
    assert "RESOLVED" in result.stdout


def test_allowlist_network_policy_is_refused_rather_than_granted(
    sandbox: DockerSandbox, tmp_path: Path
) -> None:
    """Silently granting the full network for an 'allowlist' would be a breach."""
    limits = ResourceLimits(timeout_s=60, network="allowlist")
    with pytest.raises(NotImplementedError, match="allowlist"):
        sandbox.run([PY, "-c", "print(1)"], work_dir=tmp_path, limits=limits)


def test_container_runs_as_non_root(sandbox: DockerSandbox, tmp_path: Path) -> None:
    result = sandbox.run(
        [PY, "-c", "import os; print('uid', os.getuid())"],
        work_dir=tmp_path,
        limits=ALLOW_NET,
    )
    assert result.ok
    assert "uid 65534" in result.stdout
    assert "uid 0" not in result.stdout


def test_root_filesystem_is_read_only(sandbox: DockerSandbox, tmp_path: Path) -> None:
    script = (
        "try:\n"
        "    open('/etc/slmforge-owned','w').write('x')\n"
        "    print('WROTE')\n"
        "except OSError:\n"
        "    print('READONLY')\n"
    )
    result = sandbox.run([PY, "-c", script], work_dir=tmp_path, limits=ALLOW_NET)
    assert result.ok
    assert "READONLY" in result.stdout
    assert "WROTE" not in result.stdout


def test_tmpfs_allows_scratch_writes(sandbox: DockerSandbox, tmp_path: Path) -> None:
    """Read-only root must not break legitimate scratch work."""
    script = (
        "import os, tempfile\n"
        "with tempfile.NamedTemporaryFile('w', dir='/tmp', delete=True) as f:\n"
        "    f.write('scratch'); f.flush()\n"
        "    print('SCRATCH_OK', os.path.basename(f.name))\n"
    )
    result = sandbox.run([PY, "-c", script], work_dir=tmp_path, limits=ALLOW_NET)
    assert result.ok
    assert "SCRATCH_OK" in result.stdout


def test_job_work_dir_is_mounted_and_survives(sandbox: DockerSandbox, tmp_path: Path) -> None:
    """The only host-visible output path must work, and must use container addressing."""
    script_path = tmp_path / "produce.py"
    script_path.write_text("open('artifact.txt','w').write('kept')\n", encoding="utf-8")

    container_script = DockerSandbox.container_path(script_path, tmp_path)
    assert container_script == "/work/produce.py"

    result = sandbox.run([PY, container_script], work_dir=tmp_path, limits=ALLOW_NET)
    assert result.ok, result.error
    assert (tmp_path / "artifact.txt").read_text(encoding="utf-8") == "kept"


def test_timeout_kills_and_removes_the_container(sandbox: DockerSandbox, tmp_path: Path) -> None:
    result = sandbox.run(
        [PY, "-c", "import time; time.sleep(300)"],
        work_dir=tmp_path,
        limits=ResourceLimits(timeout_s=5, network="deny", memory_mb=512),
    )
    assert result.timed_out is True
    assert not result.ok
    # The container must not be left running (it would hold the bind mount).
    assert result.duration_s < 60


def test_memory_ceiling_is_enforced(sandbox: DockerSandbox, tmp_path: Path) -> None:
    """An OOM-killed job must not masquerade as a clean script failure."""
    script = (
        "import sys\n"
        "b = bytearray(400 * 1024 * 1024)  # far above the 128 MB ceiling\n"
        "b[::4096] = b'x' * (len(b)//4096)\n"
        "print('SURVIVED')\n"
    )
    result = sandbox.run(
        [PY, "-c", script],
        work_dir=tmp_path,
        limits=ResourceLimits(timeout_s=90, network="deny", memory_mb=128),
    )
    # An OOM kill (137) must not be reported as a clean exit, and the script
    # must never have reached its final line.
    assert "SURVIVED" not in result.stdout
    assert not result.ok
    assert result.exit_code == 137
