"""Enable `python -m slmforge ...` as an alias for the `slmforge` console script."""

from __future__ import annotations

from .cli import app

if __name__ == "__main__":
    app()
