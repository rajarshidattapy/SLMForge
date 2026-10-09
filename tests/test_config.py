"""Settings parsing and runtime directory creation."""

from __future__ import annotations

from pathlib import Path

from slmforge.config import Settings


def test_settings_read_the_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("SLMFORGE_PORT", "9123")
    monkeypatch.setenv("SLMFORGE_SANDBOX_BACKEND", "docker")
    monkeypatch.setenv("SLMFORGE_CORS_ORIGINS", "http://a.test, http://b.test")
    settings = Settings(_env_file=None)
    assert settings.port == 9123
    assert settings.sandbox_backend == "docker"
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_ensure_dirs_creates_runtime_directories(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        artifact_root=tmp_path / "artifacts",
        upload_root=tmp_path / "uploads",
        hf_home=tmp_path / "hf",
        database_url=f"sqlite:///{(tmp_path / 'db' / 'test.db').as_posix()}",
    )
    settings.ensure_dirs()
    assert (tmp_path / "artifacts").is_dir()
    assert (tmp_path / "uploads").is_dir()
    assert (tmp_path / "hf").is_dir()
    assert (tmp_path / "db").is_dir()


def test_is_sqlite_detection() -> None:
    assert Settings(_env_file=None, database_url="sqlite:///./x.db").is_sqlite is True
    assert Settings(_env_file=None, database_url="postgresql+psycopg://u@h/db").is_sqlite is False
