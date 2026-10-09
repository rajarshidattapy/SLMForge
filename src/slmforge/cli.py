"""SLMForge command-line interface.

slmforge serve      # start the API (and the built web UI, if present)
slmforge worker     # start a background job worker
slmforge init-db    # create database tables
slmforge doctor     # check the environment before a build
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import settings
from .log import force_utf8_stdio, setup_logging

# Typer renders help before any command runs, so fix stdio at import time.
force_utf8_stdio()

app = typer.Typer(
    name="slmforge",
    help="Build a task-specific small language model from a natural-language prompt.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the SLMForge version."""
    console.print(f"slmforge {__version__}")


@app.command("init-db")
def init_db_cmd() -> None:
    """Create the database tables (idempotent)."""
    setup_logging()
    from .db import init_db

    settings.ensure_dirs()
    init_db()
    console.print(f"[green]Database ready[/green] at {settings.database_url}")


@app.command()
def serve(
    host: str = typer.Option(settings.host, help="Bind address"),
    port: int = typer.Option(settings.port, help="Port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)"),
) -> None:
    """Start the HTTP API."""
    import uvicorn

    setup_logging()
    console.print(f"[green]SLMForge API[/green] -> http://{host}:{port}  (docs at /docs)")
    uvicorn.run(
        "slmforge.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command()
def worker(
    kinds: str = typer.Option("", help="Comma-separated job kinds to handle (default: all)"),
    poll_interval: float = typer.Option(2.0, help="Seconds between queue polls"),
) -> None:
    """Start a background job worker."""
    setup_logging()
    from .db import init_db
    from .jobs import Worker
    from .models import JobKind

    init_db()
    parsed_kinds: list[JobKind] | None = None
    if kinds:
        try:
            parsed_kinds = [JobKind(k.strip()) for k in kinds.split(",") if k.strip()]
        except ValueError as exc:
            console.print(f"[red]Unknown job kind:[/red] {exc}")
            raise typer.Exit(code=2) from exc

    job_worker = Worker(kinds=parsed_kinds, poll_interval_s=poll_interval)
    job_worker.register_defaults()
    try:
        job_worker.run_forever()
    except KeyboardInterrupt:
        job_worker.stop()
        console.print("[yellow]Worker interrupted[/yellow]")


@app.command()
def doctor() -> None:
    """Check the local environment and report what will and will not work."""
    setup_logging()
    from .integrations.ollama import OllamaClient

    table = Table(title="SLMForge environment", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Result")
    table.add_column("Notes", style="dim")

    # Python
    table.add_row("Python", sys.version.split()[0], "")

    # torch / GPU
    try:
        import torch

        cuda = torch.cuda.is_available()
        note = ""
        if cuda:
            props = torch.cuda.get_device_properties(0)
            note = f"{props.name}, {props.total_memory / 1024**3:.1f} GB VRAM"
        else:
            note = "CPU-only build — fine-tuning will be slow"
        table.add_row("torch", torch.__version__, note)
    except ImportError:
        table.add_row("torch", "[red]missing[/red]", "install with: uv pip install -e '.[ml]'")

    # ML stack
    for name in ("transformers", "datasets", "peft", "trl", "bitsandbytes"):
        try:
            module = __import__(name)
            table.add_row(name, getattr(module, "__version__", "installed"), "")
        except ImportError:
            table.add_row(name, "[yellow]missing[/yellow]", "optional")

    # Ollama
    client = OllamaClient()

    async def _probe() -> tuple[bool, list[str]]:
        try:
            if await client.is_available():
                return True, [m.name for m in await client.list_models()]
            return False, []
        finally:
            await client.close()

    available, models = asyncio.run(_probe())
    if available:
        table.add_row("Ollama", "[green]reachable[/green]", f"{len(models)} local model(s)")
    else:
        table.add_row("Ollama", "[yellow]unreachable[/yellow]", settings.ollama_base_url)

    # Planner LLM
    table.add_row(
        "Planner LLM",
        settings.llm_provider,
        f"{settings.llm_model} @ {settings.llm_base_url}"
        if settings.llm_provider != "none"
        else "not configured",
    )

    # HF
    table.add_row(
        "HF token",
        "[green]set[/green]" if settings.hf_token else "[yellow]not set[/yellow]",
        "required for gated models and pushing",
    )

    # Paths & disk
    settings.ensure_dirs()
    for label, path in (
        ("artifacts", Path(settings.artifact_root)),
        ("uploads", Path(settings.upload_root)),
        ("hf cache", Path(settings.hf_home)),
    ):
        exists = path.exists()
        table.add_row(
            f"{label} dir",
            "[green]ok[/green]" if exists else "[red]missing[/red]",
            str(path.resolve()),
        )

    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
