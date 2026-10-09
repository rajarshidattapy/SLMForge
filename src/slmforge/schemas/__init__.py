"""Pydantic contracts exchanged between pipeline stages.

Import from here rather than from individual modules:

    from slmforge.schemas import MLSpec, DatasetPlan, ModelCandidate
"""

from .common import (
    ClarifyingQuestion,
    License,
    Metric,
    Modality,
    OutputFormat,
    Provenance,
    ResourceEstimate,
    TaskType,
)
from .dataset import (
    DataQualityIssue,
    DatasetCandidate,
    DatasetCard,
    DatasetPlan,
    LabelMapping,
    SplitSizes,
)
from .deployment import DeployRecord, DeployRequest, UsageStats
from .evaluation import Comparison, EvalReport, EvalRun, SamplePrediction
from .model import FineTuneSupport, ModelCandidate, ModelSelection
from .spec import DeploymentPreference, EvaluationPlan, FieldSpec, MLSpec, OutputSpec
from .training import METHODS, Hyperparameters, TrainingPlan, TrainingRecord

__all__ = [
    "METHODS",
    "ClarifyingQuestion",
    "Comparison",
    "DataQualityIssue",
    "DatasetCandidate",
    "DatasetCard",
    "DatasetPlan",
    "DeployRecord",
    "DeployRequest",
    "DeploymentPreference",
    "EvalReport",
    "EvalRun",
    "EvaluationPlan",
    "FieldSpec",
    "FineTuneSupport",
    "Hyperparameters",
    "LabelMapping",
    "License",
    "MLSpec",
    "Metric",
    "Modality",
    "ModelCandidate",
    "ModelSelection",
    "OutputFormat",
    "OutputSpec",
    "Provenance",
    "ResourceEstimate",
    "SamplePrediction",
    "SplitSizes",
    "TaskType",
    "TrainingPlan",
    "TrainingRecord",
    "UsageStats",
]
