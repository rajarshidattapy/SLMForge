"""The SLMForge pipeline: ordered stages from prompt to deployed endpoint."""

from __future__ import annotations

from ..models import StageName
from .analyzer import AnalyzerStage
from .base import NotImplementedStage, Stage, StageContext, StageResult
from .baseline import BaselineStage
from .datasets import DatasetStage
from .deployer import DeployerStage
from .evaluator import EvaluatorStage
from .models import ModelSelectionStage
from .planner import PlannerStage
from .trainer import TrainerStage

#: Canonical execution order: Describe -> Specify -> Data -> Model -> Baseline
#: -> Plan -> Train -> Evaluate -> Deploy.
PIPELINE: tuple[Stage, ...] = (
    AnalyzerStage(),
    DatasetStage(),
    ModelSelectionStage(),
    BaselineStage(),
    PlannerStage(),
    TrainerStage(),
    EvaluatorStage(),
    DeployerStage(),
)

STAGES: dict[StageName, Stage] = {stage.name: stage for stage in PIPELINE}


def get_stage(name: StageName) -> Stage:
    """Look up a stage by name, raising a clear error for unknown names."""
    try:
        return STAGES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown stage: {name!r}. Known: {sorted(s.value for s in STAGES)}"
        ) from exc


def next_stage_name(current: StageName) -> StageName | None:
    """Return the stage after `current`, or None if it is the last one."""
    order = [stage.name for stage in PIPELINE]
    idx = order.index(current)
    return order[idx + 1] if idx + 1 < len(order) else None


__all__ = [
    "PIPELINE",
    "STAGES",
    "AnalyzerStage",
    "BaselineStage",
    "DatasetStage",
    "DeployerStage",
    "EvaluatorStage",
    "ModelSelectionStage",
    "NotImplementedStage",
    "PlannerStage",
    "Stage",
    "StageContext",
    "StageResult",
    "TrainerStage",
    "get_stage",
    "next_stage_name",
]
