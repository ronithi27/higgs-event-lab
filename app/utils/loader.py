"""Safe artifact loader and environment diagnostics for Higgs Event Lab.

Reads saved output files, feature definitions, and manifest files without
triggering training or falling back to hardcoded scores.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.schema import (
    CurveData,
    ExperimentRecord,
    ValidationError,
    validate_curve_payload,
    validate_experiment_payload,
)


def get_project_root() -> Path:
    """Return the absolute path to the repository root directory."""
    # This file lives in app/utils/loader.py -> parents[2] is the repo root
    return Path(__file__).resolve().parents[2]


@dataclass
class PipelineStatus:
    """System and folder readiness status for display on the dashboard."""

    has_raw_data: bool = False
    raw_files: list[str] = field(default_factory=list)
    has_processed_data: bool = False
    processed_files: list[str] = field(default_factory=list)
    has_manifests: bool = False
    manifest_paths: list[str] = field(default_factory=list)
    has_artifacts: bool = False
    artifact_files: list[str] = field(default_factory=list)
    has_outputs: bool = False
    output_files: list[str] = field(default_factory=list)


@dataclass
class LoadedExperiments:
    """Collection of validated experiments and curves, with any schema warnings."""

    experiments: list[ExperimentRecord] = field(default_factory=list)
    curves: dict[str, CurveData] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.experiments) == 0


def load_feature_dictionary(root: Path | None = None) -> pd.DataFrame:
    """Load the verified feature dictionary from docs/feature_dictionary.csv.

    Parameters
    ----------
    root : Path, optional
        Project root directory. If None, detected automatically.

    Returns
    -------
    pd.DataFrame
        Table of features with column indices, groups, kinds, and physical descriptions.

    Raises
    ------
    FileNotFoundError
        If docs/feature_dictionary.csv does not exist.
    """
    root = root or get_project_root()
    path = root / "docs" / "feature_dictionary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Feature dictionary not found at {path}")

    return pd.read_csv(path)


def inspect_pipeline_status(root: Path | None = None) -> PipelineStatus:
    """Check existence of dataset files, model artifacts, and output metrics."""
    root = root or get_project_root()
    status = PipelineStatus()

    # Raw data check
    raw_dir = root / "data" / "raw"
    if raw_dir.is_dir():
        raw_items = [p.name for p in raw_dir.iterdir() if p.name != "README.md" and not p.name.startswith(".")]
        status.has_raw_data = len(raw_items) > 0
        status.raw_files = raw_items

    # Processed data check
    proc_dir = root / "data" / "processed"
    if proc_dir.is_dir():
        proc_items = [p.name for p in proc_dir.iterdir() if p.name != "README.md" and not p.name.startswith(".")]
        status.has_processed_data = len(proc_items) > 0
        status.processed_files = proc_items

        manifests = [str(p.relative_to(root)) for p in proc_dir.glob("*.manifest.json")]
        status.has_manifests = len(manifests) > 0
        status.manifest_paths = manifests

    # Artifacts check
    art_dir = root / "artifacts"
    if art_dir.is_dir():
        art_items = [p.name for p in art_dir.iterdir() if p.name != "README.md" and not p.name.startswith(".")]
        status.has_artifacts = len(art_items) > 0
        status.artifact_files = art_items

    # Outputs check
    out_dir = root / "outputs"
    if out_dir.is_dir():
        out_items = [p.name for p in out_dir.iterdir() if p.name != "README.md" and not p.name.startswith(".")]
        status.has_outputs = len(out_items) > 0
        status.output_files = out_items

    return status


def load_experiments(root: Path | None = None) -> LoadedExperiments:
    """Safely load and validate all experiment runs and curve artifacts.

    Scans:
    1. outputs/experiments.json (multi-run list)
    2. outputs/runs/*.json (individual run records)
    3. outputs/curves/*.json (curve coordinate records)

    Returns
    -------
    LoadedExperiments
        Container of validated experiment records, curve objects, and diagnostic messages.
    """
    root = root or get_project_root()
    outputs_dir = root / "outputs"
    result = LoadedExperiments()

    if not outputs_dir.is_dir():
        result.warnings.append(f"Directory 'outputs/' does not exist at {outputs_dir}.")
        return result

    # 1. Look for outputs/experiments.json
    agg_file = outputs_dir / "experiments.json"
    if agg_file.is_file():
        try:
            with open(agg_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                for idx, item in enumerate(payload):
                    if isinstance(item, dict):
                        try:
                            rec = validate_experiment_payload(item, source_file=f"experiments.json[{idx}]")
                            result.experiments.append(rec)
                        except ValidationError as err:
                            result.errors.append(str(err))
                    else:
                        result.errors.append(f"[experiments.json[{idx}]] Entry is not a JSON object")
            else:
                result.errors.append(f"[{agg_file.name}] Expected a JSON array of experiment records")
        except json.JSONDecodeError as err:
            result.errors.append(f"[{agg_file.name}] JSON syntax error: {err}")
        except Exception as exc:
            result.errors.append(f"[{agg_file.name}] Failed to read file: {exc}")

    # 2. Look for individual run records in outputs/runs/*.json
    runs_dir = outputs_dir / "runs"
    if runs_dir.is_dir():
        for run_file in sorted(runs_dir.glob("*.json")):
            try:
                with open(run_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    rec = validate_experiment_payload(payload, source_file=f"runs/{run_file.name}")
                    result.experiments.append(rec)
                else:
                    result.errors.append(f"[runs/{run_file.name}] Expected a JSON object")
            except (ValidationError, json.JSONDecodeError) as err:
                result.errors.append(f"[runs/{run_file.name}] {err}")
            except Exception as exc:
                result.errors.append(f"[runs/{run_file.name}] Failed to read: {exc}")

    # 3. Look for CSV summary files (outputs/experiments.csv or outputs/experiments_summary.csv)
    for csv_candidate in ("experiments.csv", "experiments_summary.csv"):
        csv_path = outputs_dir / csv_candidate
        if csv_path.is_file():
            try:
                df_csv = pd.read_csv(csv_path)
                for idx, row in df_csv.iterrows():
                    payload = {
                        "run_id": str(row.get("run_id", "")),
                        "model_name": str(row.get("model_name", "")),
                        "feature_group": str(row.get("feature_group", "")),
                        "seed": row.get("seed"),
                        "sample_size": row.get("sample_size"),
                        "metrics": {
                            "roc_auc": row.get("roc_auc"),
                            "pr_auc": row.get("pr_auc"),
                            "brier_score": row.get("brier_score") if pd.notna(row.get("brier_score")) else None,
                            "accuracy": row.get("accuracy") if pd.notna(row.get("accuracy")) else None,
                            "training_time_sec": row.get("training_time_sec") if pd.notna(row.get("training_time_sec")) else None,
                            "inference_latency_ms": row.get("inference_latency_ms") if pd.notna(row.get("inference_latency_ms")) else None,
                        },
                    }
                    if pd.notna(row.get("target_signal_efficiency")):
                        payload["operating_point"] = {
                            "target_signal_efficiency": row.get("target_signal_efficiency"),
                            "selected_threshold": row.get("selected_threshold"),
                            "test_background_rejection": row.get("test_background_rejection"),
                            "test_signal_efficiency": row.get("test_signal_efficiency"),
                        }
                    rec = validate_experiment_payload(payload, source_file=f"{csv_candidate}[{idx}]")
                    if not any(e.run_id == rec.run_id for e in result.experiments):
                        result.experiments.append(rec)
            except Exception as exc:
                result.errors.append(f"[{csv_candidate}] Failed to parse CSV: {exc}")

    # 4. Look for curves in outputs/curves/*.json
    curves_dir = outputs_dir / "curves"
    if curves_dir.is_dir():
        for curve_file in sorted(curves_dir.glob("*.json")):
            try:
                with open(curve_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    curve = validate_curve_payload(payload, source_file=f"curves/{curve_file.name}")
                    result.curves[curve.run_id] = curve
                else:
                    result.errors.append(f"[curves/{curve_file.name}] Expected a JSON object")
            except (ValidationError, json.JSONDecodeError) as err:
                result.errors.append(f"[curves/{curve_file.name}] {err}")
            except Exception as exc:
                result.errors.append(f"[curves/{curve_file.name}] Failed to read: {exc}")

    return result
