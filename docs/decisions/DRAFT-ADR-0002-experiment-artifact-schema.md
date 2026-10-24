# DRAFT ADR-0002: Experiment Output Artifact Schema Proposal (Task MID-05)

- **Status:** DRAFT / UNDER REVIEW (Proposal for team discussion; NOT formally approved)
- **Author:** Ronit Singh (Product UI / Streamlit Dashboard Lead)
- **Decision Owner:** Krishna (Team Lead / Integration)
- **Reviewers:** Aditya Rana (Dataset Pipeline & Classical Baselines), Rachit Gupta (MLP Pipeline & Reproducibility), Saumya Mishra (Documentation & QA Lead)
- **Related Backlog Tasks:** `MID-05` (Define artifact schema), `MID-06` (Dashboard shell), `END-05` (Export workflow)

---

## 1. Context and Problem Statement

The course brief and project milestones require evaluating multiple machine learning models (Logistic Regression, XGBoost, Compact MLP) across three input feature groups (`low_level`, `high_level`, `all`). Models will be trained independently by Aditya Rana and Rachit Gupta.

Per project rule [`docs/02_REPOSITORY_STRUCTURE.md`](../02_REPOSITORY_STRUCTURE.md) (line 121–122):
> "The dashboard reads saved, versioned experiment outputs; it should not train the whole benchmark on app startup. One agreed artifact schema must be documented before Rachit, Aditya and Ronit integrate independent modules."

This draft specifies the JSON and CSV formats expected by the dashboard's safe loader (`app/utils/loader.py`) and verified by `app/utils/schema.py`. It is submitted for review and sign-off by Krishna, Aditya, and Rachit before connecting training scripts.

---

## 2. Proposed Artifact Specifications

### Artifact A: Run Metrics Summary (JSON or CSV)

The dashboard's loader (`app/utils/loader.py`) supports two interchangeable formats for summary metrics:
1. **JSON List** in `outputs/experiments.json` (or individual run files in `outputs/runs/<run_id>.json`).
2. **Tabular CSV** in `outputs/experiments.csv` or `outputs/experiments_summary.csv`.

#### JSON Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ExperimentRecord",
  "type": "object",
  "required": ["run_id", "model_name", "feature_group", "seed", "sample_size", "metrics"],
  "properties": {
    "run_id": {
      "type": "string",
      "description": "Unique deterministic identifier, e.g. 'lr_low_level_seed42' or 'mlp_all_seed42'"
    },
    "model_name": {
      "type": "string",
      "enum": ["Logistic Regression", "XGBoost", "Compact MLP"]
    },
    "feature_group": {
      "type": "string",
      "enum": ["low_level", "high_level", "all"]
    },
    "seed": {
      "type": "integer",
      "description": "Random seed used for model initialization and data split"
    },
    "sample_size": {
      "type": "integer",
      "description": "Total event count in the sample subset (e.g. 250000 or 1000000)"
    },
    "metrics": {
      "type": "object",
      "required": ["roc_auc", "pr_auc"],
      "properties": {
        "roc_auc": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "pr_auc": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "brier_score": { "type": ["number", "null"], "minimum": 0.0, "maximum": 1.0 },
        "accuracy": { "type": ["number", "null"], "minimum": 0.0, "maximum": 1.0 },
        "training_time_sec": { "type": ["number", "null"], "minimum": 0.0 },
        "inference_latency_ms": { "type": ["number", "null"], "minimum": 0.0 }
      }
    },
    "operating_point": {
      "type": ["object", "null"],
      "description": "Validation-selected threshold evaluated on test split (optional extension)",
      "required": [
        "target_signal_efficiency",
        "selected_threshold",
        "test_background_rejection",
        "test_signal_efficiency"
      ],
      "properties": {
        "target_signal_efficiency": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "selected_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "test_background_rejection": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "test_signal_efficiency": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
      }
    },
    "provenance": {
      "type": "object",
      "description": "Execution context displayed in the dashboard Provenance Inspector",
      "properties": {
        "source_file": { "type": "string" },
        "manifest_path": { "type": "string" },
        "train_val_test_split": { "type": "array", "items": { "type": "number" } },
        "timestamp_utc": { "type": "string" },
        "git_commit_hash": { "type": "string" },
        "hardware_device": { "type": "string" }
      }
    }
  }
}
```

#### Equivalent Tabular CSV Format (`outputs/experiments.csv`)
If exporting pandas tables directly, include these column headers:
- `run_id`, `model_name`, `feature_group`, `seed`, `sample_size`
- `roc_auc`, `pr_auc`, `brier_score`, `accuracy`, `training_time_sec`, `inference_latency_ms`
- Optional operating-point columns: `target_signal_efficiency`, `selected_threshold`, `test_background_rejection`, `test_signal_efficiency`

---

### Artifact B: Curve Coordinates (`outputs/curves/<run_id>.json`)

To plot multi-model overlaid ROC, Precision-Recall, and Calibration curves without loading full test prediction arrays into memory, models should export downsampled coordinate arrays (e.g., 100–500 points).

#### JSON Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CurveData",
  "type": "object",
  "required": ["run_id", "roc", "pr"],
  "properties": {
    "run_id": { "type": "string" },
    "roc": {
      "type": "object",
      "required": ["fpr", "tpr"],
      "properties": {
        "fpr": { "type": "array", "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 } },
        "tpr": { "type": "array", "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 } }
      }
    },
    "pr": {
      "type": "object",
      "required": ["precision", "recall"],
      "properties": {
        "precision": { "type": "array", "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 } },
        "recall": { "type": "array", "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 } }
      }
    },
    "calibration": {
      "type": "object",
      "description": "Optional reliability diagram data; if provided, both arrays must be equal length with values in [0.0, 1.0]",
      "required": ["prob_pred", "prob_true"],
      "properties": {
        "prob_pred": { "type": "array", "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 } },
        "prob_true": { "type": "array", "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 } }
      }
    }
  }
}
```

---

## 3. Review Questions for Krishna, Aditya, and Rachit

Before Krishna formalizes this schema into an accepted decision for `MID-05`:
1. **Aditya (Classical ML)**: Do you prefer exporting summary metrics as `outputs/experiments.json` or as `outputs/experiments.csv`? Both are supported.
2. **Rachit (MLP)**: Does the `provenance` dictionary capture all hyperparameters needed for reproducibility (epochs, batch size, learning rate, patience)?
3. **Krishna (Integration & Protocol)**:
   - Does the required metrics set (`roc_auc`, `pr_auc`) match the minimum evaluation rubric for mid-semester?
   - Is `inference_latency_ms` feasible on your evaluation environment, or should it remain optional?
   - Does the operating-point subfield naming align with your planned extension?
