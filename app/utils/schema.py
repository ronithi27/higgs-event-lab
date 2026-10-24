"""Schema definitions and validation for experiment output artifacts.

Implements the contract for MID-05 / MID-06:
- Validates model metrics (ROC-AUC, PR-AUC, Brier score, accuracy, runtime).
- Validates curve coordinates (ROC, PR, calibration).
- Provides validation error structures with actionable messages without crashing the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


VALID_FEATURE_GROUPS = ("low_level", "high_level", "all")
VALID_SPLIT_KEYS = ("train", "val", "test")


@dataclass(frozen=True)
class OperatingPoint:
    """Operating point metrics selected on validation and evaluated on test."""

    target_signal_efficiency: float
    selected_threshold: float
    test_background_rejection: float
    test_signal_efficiency: float


@dataclass(frozen=True)
class ExperimentMetrics:
    """Core evaluation metrics for a model run."""

    roc_auc: float
    pr_auc: float
    brier_score: float | None = None
    accuracy: float | None = None
    training_time_sec: float | None = None
    inference_latency_ms: float | None = None


@dataclass(frozen=True)
class ExperimentRecord:
    """Validated representation of an individual experiment run."""

    run_id: str
    model_name: str
    feature_group: str
    seed: int
    sample_size: int
    metrics: ExperimentMetrics
    operating_point: OperatingPoint | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurveData:
    """Validated coordinate arrays for ROC, PR, and Calibration curves."""

    run_id: str
    fpr: list[float]
    tpr: list[float]
    precision: list[float]
    recall: list[float]
    prob_pred: list[float] = field(default_factory=list)
    prob_true: list[float] = field(default_factory=list)


class ValidationError(ValueError):
    """Raised when an artifact payload does not conform to the expected schema."""


def validate_experiment_payload(data: dict[str, Any], source_file: str = "") -> ExperimentRecord:
    """Validate a raw experiment dictionary against the schema.

    Parameters
    ----------
    data : dict[str, Any]
        Raw dictionary loaded from JSON.
    source_file : str, optional
        File path for error diagnostics.

    Returns
    -------
    ExperimentRecord
        Validated immutable experiment record.

    Raises
    ------
    ValidationError
        If required keys are missing or values are out of bounds.
    """
    prefix = f"[{source_file}] " if source_file else ""

    required_top = ["run_id", "model_name", "feature_group", "seed", "sample_size", "metrics"]
    for req in required_top:
        if req not in data:
            raise ValidationError(f"{prefix}Missing required top-level field: '{req}'")

    run_id = str(data["run_id"]).strip()
    if not run_id:
        raise ValidationError(f"{prefix}'run_id' cannot be empty")

    model_name = str(data["model_name"]).strip()
    if not model_name:
        raise ValidationError(f"{prefix}'model_name' cannot be empty")

    feature_group = str(data["feature_group"]).strip()
    if feature_group not in VALID_FEATURE_GROUPS:
        raise ValidationError(
            f"{prefix}Invalid 'feature_group': '{feature_group}'. Must be one of: {VALID_FEATURE_GROUPS}"
        )

    try:
        seed = int(data["seed"])
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"{prefix}'seed' must be an integer, got: {data.get('seed')}") from exc

    try:
        sample_size = int(data["sample_size"])
        if sample_size <= 0:
            raise ValueError()
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"{prefix}'sample_size' must be a positive integer, got: {data.get('sample_size')}") from exc

    # Metrics validation
    metrics_dict = data.get("metrics")
    if not isinstance(metrics_dict, dict):
        raise ValidationError(f"{prefix}'metrics' must be a dictionary")

    for metric_name in ("roc_auc", "pr_auc"):
        if metric_name not in metrics_dict:
            raise ValidationError(f"{prefix}Required metric '{metric_name}' missing from 'metrics'")
        val = metrics_dict[metric_name]
        if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
            raise ValidationError(
                f"{prefix}Metric '{metric_name}' must be a float between 0.0 and 1.0, got: {val}"
            )

    brier = metrics_dict.get("brier_score")
    if brier is not None and (not isinstance(brier, (int, float)) or not (0.0 <= brier <= 1.0)):
        raise ValidationError(f"{prefix}'brier_score' must be between 0.0 and 1.0, got: {brier}")

    accuracy = metrics_dict.get("accuracy")
    if accuracy is not None and (not isinstance(accuracy, (int, float)) or not (0.0 <= accuracy <= 1.0)):
        raise ValidationError(f"{prefix}'accuracy' must be between 0.0 and 1.0, got: {accuracy}")

    train_time = metrics_dict.get("training_time_sec")
    if train_time is not None and (not isinstance(train_time, (int, float)) or train_time < 0):
        raise ValidationError(f"{prefix}'training_time_sec' must be non-negative, got: {train_time}")

    inf_lat = metrics_dict.get("inference_latency_ms")
    if inf_lat is not None and (not isinstance(inf_lat, (int, float)) or inf_lat < 0):
        raise ValidationError(f"{prefix}'inference_latency_ms' must be non-negative, got: {inf_lat}")

    metrics = ExperimentMetrics(
        roc_auc=float(metrics_dict["roc_auc"]),
        pr_auc=float(metrics_dict["pr_auc"]),
        brier_score=float(brier) if brier is not None else None,
        accuracy=float(accuracy) if accuracy is not None else None,
        training_time_sec=float(train_time) if train_time is not None else None,
        inference_latency_ms=float(inf_lat) if inf_lat is not None else None,
    )

    # Optional Operating Point
    op_record = None
    if "operating_point" in data and data["operating_point"] is not None:
        op = data["operating_point"]
        if not isinstance(op, dict):
            raise ValidationError(f"{prefix}'operating_point' must be a dictionary")
        for op_key in ("target_signal_efficiency", "selected_threshold", "test_background_rejection", "test_signal_efficiency"):
            if op_key not in op:
                raise ValidationError(f"{prefix}'operating_point' missing key '{op_key}'")
            if not isinstance(op[op_key], (int, float)) or not (0.0 <= op[op_key] <= 1.0):
                raise ValidationError(f"{prefix}'operating_point.{op_key}' must be between 0.0 and 1.0, got: {op[op_key]}")
        op_record = OperatingPoint(
            target_signal_efficiency=float(op["target_signal_efficiency"]),
            selected_threshold=float(op["selected_threshold"]),
            test_background_rejection=float(op["test_background_rejection"]),
            test_signal_efficiency=float(op["test_signal_efficiency"]),
        )

    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}

    return ExperimentRecord(
        run_id=run_id,
        model_name=model_name,
        feature_group=feature_group,
        seed=seed,
        sample_size=sample_size,
        metrics=metrics,
        operating_point=op_record,
        provenance=provenance,
        raw_payload=data,
    )


def validate_curve_payload(data: dict[str, Any], source_file: str = "") -> CurveData:
    """Validate curve coordinates for plotting ROC, PR, and Calibration.

    Parameters
    ----------
    data : dict[str, Any]
        Raw curve dictionary loaded from JSON.
    source_file : str, optional
        File path for error diagnostics.

    Returns
    -------
    CurveData
        Validated curve coordinate object.

    Raises
    ------
    ValidationError
        If array shapes or value ranges are invalid.
    """
    prefix = f"[{source_file}] " if source_file else ""

    if "run_id" not in data or not str(data["run_id"]).strip():
        raise ValidationError(f"{prefix}Curve payload missing 'run_id'")
    run_id = str(data["run_id"]).strip()

    # ROC validation
    roc = data.get("roc")
    if not isinstance(roc, dict) or "fpr" not in roc or "tpr" not in roc:
        raise ValidationError(f"{prefix}Curve payload must contain 'roc' dict with 'fpr' and 'tpr'")
    fpr = roc["fpr"]
    tpr = roc["tpr"]
    if not isinstance(fpr, list) or not isinstance(tpr, list) or len(fpr) != len(tpr):
        raise ValidationError(f"{prefix}'fpr' and 'tpr' must be equal-length lists")
    if len(fpr) < 2:
        raise ValidationError(f"{prefix}ROC curve requires at least 2 points")

    # PR validation
    pr = data.get("pr")
    if not isinstance(pr, dict) or "precision" not in pr or "recall" not in pr:
        raise ValidationError(f"{prefix}Curve payload must contain 'pr' dict with 'precision' and 'recall'")
    precision = pr["precision"]
    recall = pr["recall"]
    if not isinstance(precision, list) or not isinstance(recall, list) or len(precision) != len(recall):
        raise ValidationError(f"{prefix}'precision' and 'recall' must be equal-length lists")
    if len(precision) < 2:
        raise ValidationError(f"{prefix}PR curve requires at least 2 points")

    # Optional calibration validation
    calib = data.get("calibration", {})
    prob_pred: list[float] = []
    prob_true: list[float] = []
    if isinstance(calib, dict) and "prob_pred" in calib and "prob_true" in calib:
        prob_pred = calib["prob_pred"]
        prob_true = calib["prob_true"]
        if not isinstance(prob_pred, list) or not isinstance(prob_true, list) or len(prob_pred) != len(prob_true):
            raise ValidationError(f"{prefix}'calibration.prob_pred' and 'prob_true' must be equal-length lists")

    # Value range checks
    for name, arr in [("fpr", fpr), ("tpr", tpr), ("precision", precision), ("recall", recall)]:
        for val in arr:
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise ValidationError(f"{prefix}Curve '{name}' values must be in [0.0, 1.0], got: {val}")

    for name, arr in [("prob_pred", prob_pred), ("prob_true", prob_true)]:
        for val in arr:
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise ValidationError(f"{prefix}Calibration '{name}' values must be in [0.0, 1.0], got: {val}")

    return CurveData(
        run_id=run_id,
        fpr=[float(x) for x in fpr],
        tpr=[float(x) for x in tpr],
        precision=[float(x) for x in precision],
        recall=[float(x) for x in recall],
        prob_pred=[float(x) for x in prob_pred],
        prob_true=[float(x) for x in prob_true],
    )
