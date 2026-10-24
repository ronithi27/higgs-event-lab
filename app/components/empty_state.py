"""Honest empty state and pipeline status indicator when model outputs are missing."""

from __future__ import annotations

import streamlit as st
from app.utils.loader import LoadedExperiments, PipelineStatus


def render_empty_state(status: PipelineStatus, loaded: LoadedExperiments) -> None:
    """Render an informative terminal-styled empty state when no model outputs are available."""

    st.markdown(
        """
        <div style="background: rgba(0,229,255,0.04); border: 1px solid rgba(0,229,255,0.2);
                    border-radius: 8px; padding: 20px; margin-bottom: 20px;
                    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 1.8;">
            <div style="color: #00e5ff; font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 2px; margin-bottom: 10px; font-weight: 600;">
                // STATUS
            </div>
            <span style="color: #ffb300;">SCAFFOLD MODE</span>
            <span style="color: #c9d1d9;"> — No ML runs saved to </span>
            <code style="color: #00ff9c; background: none; border: none;">outputs/</code>
            <span style="color: #c9d1d9;"> yet. Dashboard will never fabricate scores or trigger training on startup.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if loaded.errors:
        st.markdown(
            '<div style="color: #ff3d71; font-size: 0.72rem; text-transform: uppercase; '
            'letter-spacing: 2px; margin-bottom: 8px;">⚠ SCHEMA ERRORS</div>',
            unsafe_allow_html=True,
        )
        for err in loaded.errors:
            st.code(err, language="text")

    col1, col2 = st.columns(2)

    with col1:
        # Directory status
        def _status_line(label: str, ok: bool, detail: str) -> str:
            icon = "●" if ok else "○"
            color = "#00ff9c" if ok else "#ff3d71"
            return (
                f'<div style="margin: 8px 0; font-size: 0.8rem;">'
                f'<span style="color: {color};">{icon}</span> '
                f'<code style="color: #c9d1d9; background: none; border: none;">{label}</code>'
                f'<br><span style="color: #6e7681; padding-left: 18px;">{detail}</span></div>'
            )

        rows = (
            _status_line("data/raw/", status.has_raw_data,
                        f"{status.raw_files[0]}" if status.raw_files else "awaiting HIGGS.csv.gz download") +
            _status_line("data/processed/", status.has_processed_data,
                        f"{len(status.processed_files)} files" if status.processed_files else "run data extraction") +
            _status_line("data/processed/*.manifest.json", status.has_manifests,
                        "manifest recorded" if status.has_manifests else "not yet created") +
            _status_line("artifacts/", status.has_artifacts,
                        f"{len(status.artifact_files)} files" if status.artifact_files else "awaiting model training") +
            _status_line("outputs/", status.has_outputs,
                        f"{len(status.output_files)} files" if status.output_files else "awaiting evaluation")
        )

        st.markdown(
            f'<div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 18px;">'
            f'<div style="color: #00e5ff; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; '
            f'margin-bottom: 12px; font-weight: 600;">// REPO READINESS</div>'
            f'{rows}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px;
                        padding: 18px; margin-top: 12px;">
                <div style="color: #00e5ff; font-size: 0.7rem; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 12px; font-weight: 600;">
                    // UPSTREAM NEXT STEPS
                </div>
                <div style="font-size: 0.78rem; color: #c9d1d9; line-height: 1.9;">
                    <span style="color: #6e7681;">01</span>
                    <span style="color: #ffb300;"> aditya</span>
                    <span style="color: #6e7681;"> → Download HIGGS.csv.gz into data/raw/</span><br>
                    <span style="color: #6e7681;">02</span>
                    <span style="color: #ffb300;"> aditya</span>
                    <span style="color: #6e7681;"> → Extract 250k sample to data/processed/</span><br>
                    <span style="color: #6e7681;">03</span>
                    <span style="color: #ffb300;"> aditya+rachit</span>
                    <span style="color: #6e7681;"> → Train LR, XGBoost, MLP</span><br>
                    <span style="color: #6e7681;">04</span>
                    <span style="color: #00ff9c;"> schema</span>
                    <span style="color: #6e7681;"> → Export to outputs/experiments.json</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Expected artifact schema
        schema_snippet = """{
  "run_id": "<model>_<features>_seed<n>",
  "model_name": "Logistic Regression | XGBoost | Compact MLP",
  "feature_group": "low_level | high_level | all",
  "seed": 42,
  "sample_size": 250000,
  "metrics": {
    "roc_auc": float,       // [0.0, 1.0] required
    "pr_auc":  float,       // [0.0, 1.0] required
    "brier_score": float,   // optional
    "accuracy": float,      // optional
    "training_time_sec": float,
    "inference_latency_ms": float
  },
  "operating_point": {
    "target_signal_efficiency": float,
    "selected_threshold": float,
    "test_background_rejection": float,
    "test_signal_efficiency": float
  },
  "provenance": {
    "source_file": "...",
    "timestamp_utc": "ISO-8601"
  }
}"""
        st.markdown(
            '<div style="color: #00e5ff; font-size: 0.7rem; text-transform: uppercase; '
            'letter-spacing: 2px; margin-bottom: 8px; font-weight: 600;">// EXPECTED ARTIFACT SCHEMA</div>',
            unsafe_allow_html=True,
        )
        st.code(schema_snippet, language="json")

        curve_snippet = """{
  "run_id": "<run_id>",
  "roc": { "fpr": [...], "tpr": [...] },
  "pr":  { "precision": [...], "recall": [...] },
  "calibration": { "prob_pred": [...], "prob_true": [...] }
}"""
        st.markdown(
            '<div style="color: #00e5ff; font-size: 0.7rem; text-transform: uppercase; '
            'letter-spacing: 2px; margin: 12px 0 8px 0; font-weight: 600;">// CURVE FORMAT</div>',
            unsafe_allow_html=True,
        )
        st.code(curve_snippet, language="json")
