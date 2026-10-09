"""The stage contracts must round-trip through JSON (they live in JSON columns)."""

from __future__ import annotations

from slmforge.schemas import (
    Hyperparameters,
    MLSpec,
    ModelCandidate,
    OutputFormat,
    OutputSpec,
    TaskType,
    TrainingPlan,
)
from slmforge.schemas.dataset import DatasetCard, DatasetPlan, SplitSizes
from slmforge.schemas.evaluation import Comparison, EvalReport, EvalRun


def test_mlspec_round_trips_through_json() -> None:
    spec = MLSpec(
        title="Ticket classifier",
        summary="Classify support tickets.",
        task_type=TaskType.classification,
        input_description="Ticket text",
        allowed_labels=["Billing", "Technical", "Account Access", "General"],
        output=OutputSpec(format=OutputFormat.json_object),
    )
    dumped = spec.model_dump(mode="json")
    assert dumped["task_type"] == "classification"
    assert dumped["output"]["format"] == "json_object"
    assert MLSpec.model_validate(dumped) == spec


def test_defaults_are_sane() -> None:
    spec = MLSpec(
        title="t",
        summary="s",
        task_type=TaskType.structured_generation,
        input_description="x",
    )
    assert spec.evaluation.primary_metric == "macro_f1"
    assert spec.readiness == "ready_for_review"
    assert spec.is_high_stakes is False
    assert spec.deployment.target == "hf_endpoint"


def test_training_plan_carries_estimates_and_method() -> None:
    plan = TrainingPlan(
        recommendation="finetune",
        method="qlora",
        base_model_id="hf:Qwen/Qwen2.5-0.5B-Instruct",
        rationale="Small data, 8GB VRAM.",
        hyperparameters=Hyperparameters(lora_r=16),
    )
    dumped = plan.model_dump(mode="json")
    assert dumped["method"] == "qlora"
    assert dumped["hyperparameters"]["quantization"] == "nf4"
    assert TrainingPlan.model_validate(dumped) == plan


def test_dataset_and_eval_contracts_round_trip() -> None:
    plan = DatasetPlan(
        task_type="classification",
        target_examples=1000,
        splits=SplitSizes(train=800, validation=100, test=100),
        sufficiency="sufficient",
    )
    assert DatasetPlan.model_validate(plan.model_dump(mode="json")) == plan

    card = DatasetCard(id="d1", project_id="p1", total_examples=1000, synthetic_fraction=0.0)
    assert DatasetCard.model_validate(card.model_dump(mode="json")) == card

    report = EvalReport(
        id="e1",
        project_id="p1",
        baseline=EvalRun(label="baseline", model_id="hf:base", n_samples=100),
        comparisons=[Comparison(metric="macro_f1", baseline=0.5, finetuned=0.8, delta=0.3)],
        verdict="accept_finetuned",
    )
    assert EvalReport.model_validate(report.model_dump(mode="json")) == report


def test_model_candidate_serialises_optional_fields() -> None:
    candidate = ModelCandidate(id="hf:foo/bar", name="bar", source="hf_hub", parameters_b=0.5)
    dumped = candidate.model_dump(mode="json")
    assert dumped["parameters_b"] == 0.5
    assert dumped["finetune"]["methods"] == []
    assert ModelCandidate.model_validate(dumped) == candidate
