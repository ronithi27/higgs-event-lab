"""Results comparison view, curve plots, and operating point analysis for completed runs."""

from __future__ import annotations
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
import streamlit as st
from app.utils.loader import LoadedExperiments

# Terminal matplotlib theme
TERM_BG = "#0a0e14"
TERM_BG2 = "#111820"
TERM_GREEN = "#00ff9c"
TERM_CYAN = "#00e5ff"
TERM_AMBER = "#ffb300"
TERM_RED = "#ff3d71"
TERM_MAGENTA = "#c471f5"
TERM_TEXT = "#c9d1d9"
TERM_DIM = "#2a3a4a"

CURVE_COLORS = [TERM_GREEN, TERM_CYAN, TERM_AMBER, TERM_RED, TERM_MAGENTA,
                "#7bed9f", "#74b9ff", "#fd79a8", "#fdcb6e", "#a29bfe"]

MODEL_COLORS = {
    "Logistic Regression": TERM_AMBER,
    "XGBoost": TERM_CYAN,
    "Compact MLP": TERM_GREEN,
}


def _apply_terminal_style(fig: plt.Figure, axes) -> None:
    """Apply dark terminal styling to a matplotlib figure."""
    fig.patch.set_facecolor(TERM_BG2)
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(TERM_BG)
        ax.tick_params(colors=TERM_TEXT, labelsize=8)
        ax.xaxis.label.set_color(TERM_TEXT)
        ax.yaxis.label.set_color(TERM_TEXT)
        ax.title.set_color(TERM_GREEN)
        for spine in ax.spines.values():
            spine.set_edgecolor(TERM_DIM)
        ax.grid(True, linestyle=":", color=TERM_DIM, alpha=0.8)
        ax.title.set_fontfamily("monospace")
        ax.xaxis.label.set_fontfamily("monospace")
        ax.yaxis.label.set_fontfamily("monospace")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily("monospace")
        if ax.get_legend() is not None:
            legend = ax.get_legend()
            legend.get_frame().set_facecolor(TERM_BG2)
            legend.get_frame().set_edgecolor(TERM_DIM)
            for text in legend.get_texts():
                text.set_color(TERM_TEXT)
                text.set_fontfamily("monospace")


def render_results_view(loaded: LoadedExperiments) -> None:
    """Render comparison table, curves, and operating point analysis."""
    st.markdown('<h1 style="font-size:1.6rem;">run.benchmarks</h1>', unsafe_allow_html=True)

    if loaded.is_empty:
        st.warning("No experiment records found.")
        return

    # Build DataFrame
    records: list[dict[str, Any]] = []
    for exp in loaded.experiments:
        row: dict[str, Any] = {
            "run_id": exp.run_id,
            "model": exp.model_name,
            "features": exp.feature_group,
            "seed": exp.seed,
            "n_samples": exp.sample_size,
            "ROC-AUC": exp.metrics.roc_auc,
            "PR-AUC": exp.metrics.pr_auc,
            "Brier": exp.metrics.brier_score if exp.metrics.brier_score is not None else float("nan"),
            "Accuracy": exp.metrics.accuracy if exp.metrics.accuracy is not None else float("nan"),
            "Train(s)": exp.metrics.training_time_sec if exp.metrics.training_time_sec is not None else float("nan"),
            "Inf(ms)": exp.metrics.inference_latency_ms if exp.metrics.inference_latency_ms is not None else float("nan"),
        }
        if exp.operating_point:
            row["OpPt SigEff"] = exp.operating_point.target_signal_efficiency
            row["Threshold"] = exp.operating_point.selected_threshold
            row["BkgRej"] = exp.operating_point.test_background_rejection
            row["TestSigEff"] = exp.operating_point.test_signal_efficiency
        records.append(row)

    df_results = pd.DataFrame(records)

    # Summary metric cards at top
    best_roc = df_results["ROC-AUC"].max()
    best_run = df_results.loc[df_results["ROC-AUC"].idxmax(), "run_id"]
    best_pr = df_results["PR-AUC"].max()
    n_runs = len(df_results)

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;">
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 16px; text-align: center;">
                <div style="color: #6e7681; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Experiments</div>
                <div style="color: #00e5ff; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px rgba(0,229,255,0.3);">{n_runs}</div>
            </div>
            <div style="background: #111820; border: 1px solid rgba(0,255,156,0.3); border-radius: 8px; padding: 16px; text-align: center;
                        box-shadow: 0 0 15px rgba(0,255,156,0.05);">
                <div style="color: #6e7681; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Best ROC-AUC</div>
                <div style="color: #00ff9c; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px rgba(0,255,156,0.4);">{best_roc:.4f}</div>
            </div>
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 16px; text-align: center;">
                <div style="color: #6e7681; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Best PR-AUC</div>
                <div style="color: #c471f5; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px rgba(196,113,245,0.3);">{best_pr:.4f}</div>
            </div>
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 16px; text-align: center;">
                <div style="color: #6e7681; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Top Run</div>
                <div style="color: #ffb300; font-size: 0.85rem; font-weight: 600; margin-top: 8px; word-break: break-all;">{best_run}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Filters ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase; '
        'letter-spacing: 2px; margin-bottom: 10px; font-weight: 600;">// FILTERS</div>',
        unsafe_allow_html=True,
    )
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        all_groups = sorted(df_results["features"].unique().tolist())
        selected_groups = st.multiselect("Feature Groups:", options=all_groups, default=all_groups)
    with col_f2:
        all_models = sorted(df_results["model"].unique().tolist())
        selected_models = st.multiselect("Models:", options=all_models, default=all_models)

    filtered = df_results[
        df_results["features"].isin(selected_groups) & df_results["model"].isin(selected_models)
    ]

    # ── Metric Table ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase; '
        'letter-spacing: 2px; margin: 16px 0 10px 0; font-weight: 600;">// METRIC COMPARISON TABLE</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        filtered.sort_values("ROC-AUC", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ EXPORT TABLE (CSV)",
        data=csv_bytes,
        file_name="higgs_experiment_comparison.csv",
        mime="text/csv",
    )

    # ── ROC-AUC Bar Chart ─────────────────────────────────────────────────────
    st.markdown(
        '<div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase; '
        'letter-spacing: 2px; margin: 24px 0 10px 0; font-weight: 600;">// ROC-AUC BY EXPERIMENT</div>',
        unsafe_allow_html=True,
    )
    fig_bar, ax_bar = plt.subplots(figsize=(12, max(3, 0.45 * len(filtered))))
    run_ids = filtered["run_id"].tolist()
    roc_vals = filtered["ROC-AUC"].tolist()
    bar_colors = [MODEL_COLORS.get(m, TERM_CYAN) for m in filtered["model"].tolist()]
    bars = ax_bar.barh(run_ids, roc_vals, color=bar_colors, alpha=0.85, height=0.6)
    ax_bar.axvline(0.5, color=TERM_RED, linestyle=":", linewidth=1, alpha=0.7, label="Random (0.5)")
    for bar, val in zip(bars, roc_vals):
        ax_bar.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", color=TERM_TEXT, fontsize=8, fontfamily="monospace")
    ax_bar.set_xlabel("ROC-AUC")
    ax_bar.set_title("ROC-AUC by Experiment Run")
    ax_bar.set_xlim(0, 1.05)
    # Legend patches
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
    ax_bar.legend(handles=legend_patches, loc="lower right", fontsize=8)
    _apply_terminal_style(fig_bar, ax_bar)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # ── ROC / PR / Calibration Curves ─────────────────────────────────────────
    st.markdown(
        '<div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase; '
        'letter-spacing: 2px; margin: 24px 0 10px 0; font-weight: 600;">// CURVE OVERLAYS</div>',
        unsafe_allow_html=True,
    )
    runs_with_curves = [exp.run_id for exp in loaded.experiments if exp.run_id in loaded.curves]

    if not runs_with_curves:
        st.markdown(
            '<div style="color: #ffb300; font-size: 0.8rem;">⚠ No curve coordinate files found in outputs/curves/</div>',
            unsafe_allow_html=True,
        )
    else:
        selected_runs = st.multiselect(
            "Select runs to overlay:",
            options=runs_with_curves,
            default=runs_with_curves[:6],
        )

        if selected_runs:
            has_calib = any(
                len(loaded.curves[r].prob_pred) > 0 for r in selected_runs if r in loaded.curves
            )
            n_plots = 3 if has_calib else 2
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
            ax_roc, ax_pr = axes[0], axes[1]

            for i, run_id in enumerate(selected_runs):
                if run_id not in loaded.curves:
                    continue
                curve = loaded.curves[run_id]
                color = CURVE_COLORS[i % len(CURVE_COLORS)]
                label = run_id.replace("_seed42", "")
                ax_roc.plot(curve.fpr, curve.tpr, lw=1.8, label=label, color=color, alpha=0.9)
                ax_pr.plot(curve.recall, curve.precision, lw=1.8, label=label, color=color, alpha=0.9)

            # Diagonal reference
            ax_roc.plot([0, 1], [0, 1], "--", lw=1, color=TERM_DIM, alpha=0.8, label="Random")
            ax_roc.set_title("ROC Curve")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate (Signal Efficiency)")
            ax_roc.legend(loc="lower right", fontsize=7)

            ax_pr.set_title("Precision-Recall Curve")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.legend(loc="lower left", fontsize=7)

            if has_calib:
                ax_cal = axes[2]
                for i, run_id in enumerate(selected_runs):
                    if run_id not in loaded.curves:
                        continue
                    curve = loaded.curves[run_id]
                    if len(curve.prob_pred) > 0 and len(curve.prob_true) > 0:
                        color = CURVE_COLORS[i % len(CURVE_COLORS)]
                        label = run_id.replace("_seed42", "")
                        ax_cal.plot(curve.prob_pred, curve.prob_true, "o-", lw=1.8,
                                    markersize=4, label=label, color=color, alpha=0.9)
                ax_cal.plot([0, 1], [0, 1], "--", lw=1, color=TERM_DIM, alpha=0.8, label="Perfect")
                ax_cal.set_title("Calibration Curve")
                ax_cal.set_xlabel("Mean Predicted Probability")
                ax_cal.set_ylabel("Fraction of Positives")
                ax_cal.legend(loc="lower right", fontsize=7)

            _apply_terminal_style(fig, axes)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Training Time ─────────────────────────────────────────────────────────
    runs_with_timing = filtered.dropna(subset=["Train(s)"])
    if not runs_with_timing.empty:
        st.markdown(
            '<div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase; '
            'letter-spacing: 2px; margin: 24px 0 10px 0; font-weight: 600;">// TRAINING COMPUTE COST</div>',
            unsafe_allow_html=True,
        )
        fig_t, ax_t = plt.subplots(figsize=(12, max(3, 0.45 * len(runs_with_timing))))
        t_colors = [MODEL_COLORS.get(m, TERM_CYAN) for m in runs_with_timing["model"].tolist()]
        t_bars = ax_t.barh(
            runs_with_timing["run_id"], runs_with_timing["Train(s)"],
            color=t_colors, alpha=0.85, height=0.6,
        )
        for bar, val in zip(t_bars, runs_with_timing["Train(s)"]):
            ax_t.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                      f"{val:.1f}s", va="center", color=TERM_TEXT, fontsize=8, fontfamily="monospace")
        ax_t.set_xlabel("Training Time (seconds)")
        ax_t.set_title("Training Wall-Clock Time by Experiment Run")
        _apply_terminal_style(fig_t, ax_t)
        plt.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)

    # ── Operating Points ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase; '
        'letter-spacing: 2px; margin: 24px 0 10px 0; font-weight: 600;">// OPERATING POINT ANALYSIS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color: #6e7681; font-size: 0.78rem;">Thresholds selected on validation data '
        'at target signal efficiency, evaluated on held-out test data.</p>',
        unsafe_allow_html=True,
    )
    op_runs = [exp for exp in loaded.experiments if exp.operating_point is not None]
    if not op_runs:
        st.markdown(
            '<div style="color: #ffb300; font-size: 0.8rem;">⚠ No operating-point data in outputs</div>',
            unsafe_allow_html=True,
        )
    else:
        op_data = []
        for exp in op_runs:
            op = exp.operating_point
            assert op is not None
            op_data.append({
                "Run ID": exp.run_id,
                "Model": exp.model_name,
                "Features": exp.feature_group,
                "Target SigEff": f"{op.target_signal_efficiency:.0%}",
                "Threshold": f"{op.selected_threshold:.4f}",
                "Test SigEff": f"{op.test_signal_efficiency:.2%}",
                "Test BkgRej": f"{op.test_background_rejection:.2%}",
            })
        st.dataframe(pd.DataFrame(op_data), use_container_width=True, hide_index=True)

    # ── Provenance Inspector ──────────────────────────────────────────────────
    st.markdown(
        '<div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase; '
        'letter-spacing: 2px; margin: 24px 0 10px 0; font-weight: 600;">// PROVENANCE INSPECTOR</div>',
        unsafe_allow_html=True,
    )
    all_run_ids = [exp.run_id for exp in loaded.experiments]
    selected_prov = st.selectbox("Select run:", options=all_run_ids, label_visibility="collapsed")
    if selected_prov:
        target_exp = next((e for e in loaded.experiments if e.run_id == selected_prov), None)
        if target_exp:
            prov = target_exp.provenance
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(
                    f'<div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; '
                    f'padding: 16px; font-size: 0.78rem; line-height: 1.9;">'
                    f'<span style="color: #6e7681;">source_file</span> '
                    f'<span style="color: #00ff9c;">{prov.get("source_file", "—")}</span><br>'
                    f'<span style="color: #6e7681;">manifest</span> '
                    f'<span style="color: #c9d1d9;">{prov.get("manifest_path", "—")}</span><br>'
                    f'<span style="color: #6e7681;">split</span> '
                    f'<span style="color: #c9d1d9;">{prov.get("train_val_test_split", "—")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with p2:
                st.markdown(
                    f'<div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; '
                    f'padding: 16px; font-size: 0.78rem; line-height: 1.9;">'
                    f'<span style="color: #6e7681;">timestamp</span> '
                    f'<span style="color: #c9d1d9;">{prov.get("timestamp_utc", "—")}</span><br>'
                    f'<span style="color: #6e7681;">hardware</span> '
                    f'<span style="color: #c9d1d9;">{prov.get("hardware_device", "—")}</span><br>'
                    f'<span style="color: #6e7681;">git_hash</span> '
                    f'<span style="color: #00e5ff;">{prov.get("git_commit_hash", "—")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if target_exp.raw_payload:
                with st.expander("View raw JSON payload"):
                    st.json(target_exp.raw_payload)
