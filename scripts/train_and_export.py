"""Train all models and export results to outputs/ for the dashboard.

Trains Logistic Regression, XGBoost, and Compact MLP across low_level,
high_level, and all feature groups on the 250k processed sample.
Exports:
  - outputs/experiments.json  (summary metrics)
  - outputs/curves/<run_id>.json  (ROC, PR, calibration curves)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from higgs_event_lab.features import (
    ALL_COLUMNS,
    FEATURE_GROUPS,
    TARGET_COLUMN,
    get_feature_columns,
)

SEED = 42
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "higgs_sample_250k.csv.gz"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CURVES_DIR = OUTPUTS_DIR / "curves"


def load_data() -> pd.DataFrame:
    """Load the 250k processed sample."""
    print(f"Loading data from {SAMPLE_PATH}...")
    df = pd.read_csv(SAMPLE_PATH, header=None, names=list(ALL_COLUMNS))
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Class distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    return df


def split_data(
    df: pd.DataFrame, feature_group: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/val/test (60/20/20) with standard scaling."""
    feature_cols = get_feature_columns(feature_group)
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    # 60/20/20 split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val
    )

    # Fit scaler on train only (leakage control)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_operating_point(
    y_val: np.ndarray, proba_val: np.ndarray,
    y_test: np.ndarray, proba_test: np.ndarray,
    target_eff: float = 0.5,
) -> dict:
    """Select threshold on val for target signal efficiency, evaluate on test."""
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, proba_val)
    # Find threshold closest to target signal efficiency
    idx = np.argmin(np.abs(tpr_val - target_eff))
    threshold = float(thresholds_val[min(idx, len(thresholds_val) - 1)])

    # Evaluate on test
    y_pred_test = (proba_test >= threshold).astype(int)
    test_sig_eff = float(np.mean(y_pred_test[y_test == 1.0]))
    test_bkg_rej = float(1.0 - np.mean(y_pred_test[y_test == 0.0]))

    return {
        "target_signal_efficiency": target_eff,
        "selected_threshold": round(threshold, 6),
        "test_background_rejection": round(test_bkg_rej, 6),
        "test_signal_efficiency": round(test_sig_eff, 6),
    }


def export_curves(run_id: str, y_test: np.ndarray, proba_test: np.ndarray) -> dict:
    """Compute and export ROC, PR, and calibration curve coordinates."""
    # ROC curve (downsample to ~200 points)
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    step = max(1, len(fpr) // 200)
    fpr_ds = fpr[::step].tolist()
    tpr_ds = tpr[::step].tolist()
    # Ensure endpoints
    if fpr_ds[0] != 0.0:
        fpr_ds.insert(0, 0.0)
        tpr_ds.insert(0, 0.0)
    if fpr_ds[-1] != 1.0:
        fpr_ds.append(1.0)
        tpr_ds.append(1.0)

    # PR curve (downsample)
    precision, recall, _ = precision_recall_curve(y_test, proba_test)
    step_pr = max(1, len(precision) // 200)
    prec_ds = precision[::step_pr].tolist()
    rec_ds = recall[::step_pr].tolist()

    # Calibration curve (10 bins)
    try:
        prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=10, strategy="uniform")
        prob_true = prob_true.tolist()
        prob_pred = prob_pred.tolist()
    except Exception:
        prob_true, prob_pred = [], []

    curve_data = {
        "run_id": run_id,
        "roc": {"fpr": [round(v, 6) for v in fpr_ds], "tpr": [round(v, 6) for v in tpr_ds]},
        "pr": {"precision": [round(v, 6) for v in prec_ds], "recall": [round(v, 6) for v in rec_ds]},
        "calibration": {
            "prob_pred": [round(v, 6) for v in prob_pred],
            "prob_true": [round(v, 6) for v in prob_true],
        },
    }

    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = CURVES_DIR / f"{run_id}.json"
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)
    print(f"    Saved curves → {curve_path.relative_to(PROJECT_ROOT)}")

    return curve_data


def train_logistic_regression(
    X_train, X_val, X_test, y_train, y_val, y_test, feature_group: str
) -> dict:
    """Train Logistic Regression and return experiment record."""
    run_id = f"lr_{feature_group}_seed{SEED}"
    print(f"\n  [{run_id}] Training Logistic Regression...")

    t0 = time.time()
    model = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predictions
    t_inf = time.time()
    proba_test = model.predict_proba(X_test)[:, 1]
    inf_time = (time.time() - t_inf) * 1000  # ms

    proba_val = model.predict_proba(X_val)[:, 1]

    record = build_record(
        run_id, "Logistic Regression", feature_group,
        y_val, proba_val, y_test, proba_test,
        train_time, inf_time,
    )
    export_curves(run_id, y_test, proba_test)
    return record


def train_xgboost(
    X_train, X_val, X_test, y_train, y_val, y_test, feature_group: str
) -> dict:
    """Train XGBoost and return experiment record."""
    import xgboost as xgb

    run_id = f"xgb_{feature_group}_seed{SEED}"
    print(f"\n  [{run_id}] Training XGBoost...")

    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, verbose=False)
    train_time = time.time() - t0

    t_inf = time.time()
    proba_test = model.predict_proba(X_test)[:, 1]
    inf_time = (time.time() - t_inf) * 1000

    proba_val = model.predict_proba(X_val)[:, 1]

    record = build_record(
        run_id, "XGBoost", feature_group,
        y_val, proba_val, y_test, proba_test,
        train_time, inf_time,
    )
    export_curves(run_id, y_test, proba_test)
    return record


def train_mlp(
    X_train, X_val, X_test, y_train, y_val, y_test, feature_group: str
) -> dict:
    """Train Compact MLP with PyTorch and return experiment record."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    run_id = f"mlp_{feature_group}_seed{SEED}"
    print(f"\n  [{run_id}] Training Compact MLP...")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"    Using device: {device}")

    n_features = X_train.shape[1]

    # Compact MLP architecture
    class CompactMLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = CompactMLP(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

    # Training with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5
    epochs = 50

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss = criterion(model(xb).squeeze(), yb)
                val_losses.append(val_loss.item())
        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    # Inference
    t_inf = time.time()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits_test = model(X_test_t).squeeze().cpu().numpy()
        proba_test = 1.0 / (1.0 + np.exp(-logits_test))
    inf_time = (time.time() - t_inf) * 1000

    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        logits_val = model(X_val_t).squeeze().cpu().numpy()
        proba_val = 1.0 / (1.0 + np.exp(-logits_val))

    record = build_record(
        run_id, "Compact MLP", feature_group,
        y_val, proba_val, y_test, proba_test,
        train_time, inf_time,
    )
    export_curves(run_id, y_test, proba_test)
    return record


def build_record(
    run_id: str, model_name: str, feature_group: str,
    y_val, proba_val, y_test, proba_test,
    train_time: float, inf_time: float,
) -> dict:
    """Build an experiment record dict from predictions."""
    roc = roc_auc_score(y_test, proba_test)
    pr = average_precision_score(y_test, proba_test)
    brier = brier_score_loss(y_test, proba_test)
    acc = accuracy_score(y_test, (proba_test >= 0.5).astype(int))

    op = compute_operating_point(y_val, proba_val, y_test, proba_test, target_eff=0.5)

    print(f"    ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Brier: {brier:.4f} | Acc: {acc:.4f} | Train: {train_time:.1f}s")

    return {
        "run_id": run_id,
        "model_name": model_name,
        "feature_group": feature_group,
        "seed": SEED,
        "sample_size": 250000,
        "metrics": {
            "roc_auc": round(roc, 6),
            "pr_auc": round(pr, 6),
            "brier_score": round(brier, 6),
            "accuracy": round(acc, 6),
            "training_time_sec": round(train_time, 2),
            "inference_latency_ms": round(inf_time, 2),
        },
        "operating_point": op,
        "provenance": {
            "source_file": "data/processed/higgs_sample_250k.csv.gz",
            "manifest_path": "data/processed/higgs_sample_250k.csv.gz.manifest.json",
            "train_val_test_split": [0.6, 0.2, 0.2],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_device": "local_mac",
            "git_commit_hash": "scaffold",
        },
    }


def main():
    print("=" * 60)
    print("Higgs Event Lab — Full Model Training & Export")
    print("=" * 60)

    df = load_data()
    all_records: list[dict] = []

    for fg in ["low_level", "high_level", "all"]:
        print(f"\n{'='*50}")
        print(f"Feature Group: {fg} ({len(get_feature_columns(fg))} features)")
        print(f"{'='*50}")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, fg)
        print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        # Train all 3 models
        all_records.append(
            train_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test, fg)
        )
        all_records.append(
            train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, fg)
        )
        all_records.append(
            train_mlp(X_train, X_val, X_test, y_train, y_val, y_test, fg)
        )

    # Export experiments.json
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    experiments_path = OUTPUTS_DIR / "experiments.json"
    with open(experiments_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n✅ Saved {len(all_records)} experiment records → {experiments_path.relative_to(PROJECT_ROOT)}")

    # Also export CSV
    csv_records = []
    for r in all_records:
        row = {
            "run_id": r["run_id"],
            "model_name": r["model_name"],
            "feature_group": r["feature_group"],
            "seed": r["seed"],
            "sample_size": r["sample_size"],
        }
        row.update(r["metrics"])
        if r.get("operating_point"):
            row.update(r["operating_point"])
        csv_records.append(row)
    pd.DataFrame(csv_records).to_csv(OUTPUTS_DIR / "experiments.csv", index=False)
    print(f"✅ Saved CSV summary → outputs/experiments.csv")
    print(f"\n{'='*60}")
    print("Done! Refresh the Streamlit dashboard to see real results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
