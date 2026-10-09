"""Shared pytest fixtures.

Environment variables are set *before* importing `slmforge`, because the settings
singleton and the SQLAlchemy engine are created at import time. Tests therefore
run against a throwaway SQLite file and a temporary artifact root, and never
touch the developer's `.env` or real data.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="slmforge-tests-"))
os.environ.update(
    {
        "SLMFORGE_ENV": "local",
        "SLMFORGE_DATABASE_URL": f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}",
        "SLMFORGE_ARTIFACT_ROOT": str(_TEST_ROOT / "artifacts"),
        "SLMFORGE_UPLOAD_ROOT": str(_TEST_ROOT / "uploads"),
        "SLMFORGE_HF_HOME": str(_TEST_ROOT / "hf"),
        "SLMFORGE_LLM_PROVIDER": "none",
        "SLMFORGE_SANDBOX_BACKEND": "subprocess",
        "SLMFORGE_LOG_LEVEL": "WARNING",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from slmforge.db import SessionLocal, init_db  # noqa: E402

PROMPT = (
    "Build a small model that classifies customer support tickets into Billing, "
    "Technical, Account Access, or General and returns only a valid JSON object."
)


@pytest.fixture(scope="session", autouse=True)
def _cleanup() -> Iterator[None]:
    yield
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """TestClient with the app lifespan (and therefore `init_db`) executed."""
    from slmforge.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def session() -> Iterator[Session]:
    """A session against the test database with tables created."""
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def project(client: TestClient) -> dict:
    """A freshly created project."""
    response = client.post("/api/projects", json={"prompt": PROMPT})
    assert response.status_code == 201, response.text
    return response.json()
