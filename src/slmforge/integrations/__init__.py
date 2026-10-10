"""External service integrations.

Hugging Face is reached primarily through the official `hf` CLI (`hf_cli.py`),
which handles discovery, parquet URLs, managed Jobs, endpoints and transfers, and
carries its own authentication. `huggingface.py` remains for in-process SDK work
(licence classification, card handling). `parquet.py` adds DuckDB profiling over
remote parquet so curation does not need to download a dataset first.
"""

from . import hf_cli, huggingface, parquet
from .hf_cli import HFCLIError
from .llm import PlannerLLM, PlannerLLMUnavailable, parse_json_object, validate_into
from .ollama import OllamaClient, OllamaModel
from .parquet import DuckDBUnavailable

__all__ = [
    "DuckDBUnavailable",
    "HFCLIError",
    "OllamaClient",
    "OllamaModel",
    "PlannerLLM",
    "PlannerLLMUnavailable",
    "hf_cli",
    "huggingface",
    "parquet",
    "parse_json_object",
    "validate_into",
]
