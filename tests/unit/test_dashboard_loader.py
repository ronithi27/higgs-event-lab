"""Unit tests for the dashboard schema validation, artifact loader, and pipeline checks."""

import tempfile
import json
from pathlib import Path
import pytest

from app.utils.loader import (
    get_project_root,
    inspect_pipeline_status,
    load_experiments,
    load_feature_dictionary,
)
from app.utils.schema import (
    ValidationError,
    validate_curve_payload,
    validate_experiment_payload,
)


def test_load_feature_dictionary_loads_verified_schema():
    """Verify that the feature dictionary loads all 28 features plus the target label."""
    df = load_feature_dictionary()
    assert len(df) == 29
    assert set(df.columns) >= {"file_column_index", "name", "group", "kind", "description"}
    assert (df["group"] == "low_level").sum() == 21
    assert (df["group"] == "high_level").sum() == 7
    assert (df["group"] == "target").sum() == 1
    assert df.loc[df["file_column_index"] == 0, "name"].iloc[0] == "class_label"


def test_inspect_pipeline_status_smoke():
    """Check that pipeline inspection runs without errors and reports existing directories."""
    status = inspect_pipeline_status()
    assert hasattr(status, "has_raw_data")
    assert hasattr(status, "has_processed_data")
    assert hasattr(status, "has_outputs")
    assert hasattr(status, "has_artifacts")


def test_load_experiments_empty_state_does_not_crash(tmp_path):
    """When outputs/ has no run files, load_experiments must return an empty container safely."""
    loaded = load_experiments(root=tmp_path)
    assert loaded.is_empty
    assert len(loaded.experiments) == 0
    assert len(loaded.curves) == 0


def test_validate_experiment_payload_valid():
    """Verify that a compliant experiment record validates cleanly."""
    valid_data = {
        "run_id": "lr_low_level_s42",
        "model_name": "Logistic Regression",
        "feature_group": "low_level",
        "seed": 42,
        "sample_size": 250000,
        "metrics": {
            "roc_auc": 0.68,
            "pr_auc": 0.65,
            "brier_score": 0.22,
            "accuracy": 0.64,
            "training_time_sec": 12.4,
            "inference_latency_ms": 0.8,
        },
        "operating_point": {
            "target_signal_efficiency": 0.50,
            "selected_threshold": 0.52,
            "test_background_rejection": 0.70,
            "test_signal_efficiency": 0.498,
        },
        "provenance": {
            "source_file": "data/processed/sample.csv.gz",
        },
    }
    rec = validate_experiment_payload(valid_data)
    assert rec.run_id == "lr_low_level_s42"
    assert rec.model_name == "Logistic Regression"
    assert rec.feature_group == "low_level"
    assert rec.metrics.roc_auc == 0.68
    assert rec.operating_point is not None
    assert rec.operating_point.target_signal_efficiency == 0.50


def test_validate_experiment_payload_rejects_missing_required():
    """Missing top-level fields must raise ValidationError."""
    bad_data = {
        "run_id": "bad_run",
        "model_name": "MLP",
        # missing feature_group, seed, sample_size, metrics
    }
    with pytest.raises(ValidationError, match="Missing required top-level field"):
        validate_experiment_payload(bad_data)


def test_validate_experiment_payload_rejects_out_of_bounds_metrics():
    """Metric values out of [0, 1] must raise ValidationError."""
    bad_metric = {
        "run_id": "test",
        "model_name": "MLP",
        "feature_group": "all",
        "seed": 42,
        "sample_size": 1000,
        "metrics": {
            "roc_auc": 1.5,  # Invalid
            "pr_auc": 0.7,
        },
    }
    with pytest.raises(ValidationError, match="must be a float between 0.0 and 1.0"):
        validate_experiment_payload(bad_metric)


def test_validate_curve_payload():
    """Valid curve coordinates parse and validate correctly."""
    valid_curve = {
        "run_id": "test_run",
        "roc": {
            "fpr": [0.0, 0.5, 1.0],
            "tpr": [0.0, 0.7, 1.0],
        },
        "pr": {
            "precision": [0.6, 0.8, 1.0],
            "recall": [1.0, 0.5, 0.0],
        },
    }
    curve = validate_curve_payload(valid_curve)
    assert curve.run_id == "test_run"
    assert len(curve.fpr) == 3
    assert len(curve.precision) == 3


def test_validate_curve_payload_mismatched_lengths():
    """Mismatched coordinate array lengths must raise ValidationError."""
    bad_curve = {
        "run_id": "bad_curve",
        "roc": {
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 0.5, 1.0],  # Mismatched length
        },
        "pr": {
            "precision": [1.0, 0.5],
            "recall": [0.5, 1.0],
        },
    }
    with pytest.raises(ValidationError, match="equal-length lists"):
        validate_curve_payload(bad_curve)


def test_validate_curve_payload_rejects_out_of_bounds_coordinates():
    """Coordinate values out of [0.0, 1.0] must raise ValidationError."""
    bad_coords = {
        "run_id": "bad_coords",
        "roc": {
            "fpr": [0.0, 1.2],  # Out of bounds
            "tpr": [0.0, 1.0],
        },
        "pr": {
            "precision": [1.0, 0.5],
            "recall": [0.0, 1.0],
        },
    }
    with pytest.raises(ValidationError, match="must be in \\[0.0, 1.0\\]"):
        validate_curve_payload(bad_coords)


def test_load_experiments_supports_csv(tmp_path: Path):
    """Verify that experiments can be parsed from a CSV summary file."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    csv_file = outputs_dir / "experiments_summary.csv"
    csv_file.write_text(
        "run_id,model_name,feature_group,seed,sample_size,roc_auc,pr_auc,training_time_sec\n"
        "xgb_all_s42,XGBoost,all,42,250000,0.80,0.78,45.2\n",
        encoding="utf-8",
    )
    loaded = load_experiments(root=tmp_path)
    assert len(loaded.experiments) == 1
    assert loaded.experiments[0].run_id == "xgb_all_s42"
    assert loaded.experiments[0].metrics.roc_auc == 0.80
    assert loaded.experiments[0].metrics.training_time_sec == 45.2

