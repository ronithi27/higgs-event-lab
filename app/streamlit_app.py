"""Higgs Event Lab — Streamlit Research Workbench.

Main entrypoint for the project dashboard owned by Ronit Singh.
Visualizes experiment provenance, verified feature schemas, and model benchmarks
with a strict empty state when artifacts are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
from app.components.empty_state import render_empty_state
from app.components.feature_catalog import render_feature_catalog
from app.components.overview import render_project_overview
from app.components.results_view import render_results_view
from app.utils.loader import inspect_pipeline_status, load_experiments

TERMINAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Fira+Code:wght@300;400;500;600;700&display=swap');

/* ===== ROOT THEME ===== */
:root {
    --term-bg: #0a0e14;
    --term-bg-secondary: #0d1117;
    --term-bg-card: #111820;
    --term-border: #1a2332;
    --term-green: #00ff9c;
    --term-green-dim: #00cc7a;
    --term-cyan: #00e5ff;
    --term-cyan-dim: #00b8d4;
    --term-amber: #ffb300;
    --term-red: #ff3d71;
    --term-magenta: #c471f5;
    --term-text: #c9d1d9;
    --term-text-dim: #6e7681;
    --term-text-bright: #e6edf3;
    --term-glow-green: 0 0 10px rgba(0,255,156,0.3), 0 0 20px rgba(0,255,156,0.1);
    --term-glow-cyan: 0 0 10px rgba(0,229,255,0.3), 0 0 20px rgba(0,229,255,0.1);
}

/* ===== GLOBAL OVERRIDES ===== */
.stApp, [data-testid="stAppViewContainer"], .main .block-container {
    background-color: var(--term-bg) !important;
    color: var(--term-text) !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono', monospace !important;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

/* ===== SCANLINE OVERLAY ===== */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,156,0.015) 2px,
        rgba(0,255,156,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ===== HEADER / TITLE ===== */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--term-green) !important;
    text-shadow: var(--term-glow-green);
    letter-spacing: 1px;
}

h1::before, .stMarkdown h1::before { content: "$ "; opacity: 0.5; }
h2::before, .stMarkdown h2::before { content: "> "; opacity: 0.4; }
h3::before, .stMarkdown h3::before { content: "→ "; opacity: 0.3; }

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color: var(--term-bg-secondary) !important;
    border-right: 1px solid var(--term-green) !important;
    box-shadow: 2px 0 15px rgba(0,255,156,0.05);
}

[data-testid="stSidebar"] * {
    color: var(--term-text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stSidebar"] .stRadio label {
    background: var(--term-bg-card) !important;
    border: 1px solid var(--term-border) !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
    margin: 3px 0 !important;
    transition: all 0.3s ease !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    border-color: var(--term-green) !important;
    box-shadow: var(--term-glow-green) !important;
    background: rgba(0,255,156,0.05) !important;
}

[data-testid="stSidebar"] .stRadio [data-checked="true"] + label,
[data-testid="stSidebar"] .stRadio label[data-selected="true"] {
    border-color: var(--term-green) !important;
    background: rgba(0,255,156,0.08) !important;
}

/* ===== METRICS ===== */
[data-testid="stMetric"], [data-testid="stMetricValue"] {
    background: var(--term-bg-card) !important;
    border: 1px solid var(--term-border) !important;
    border-radius: 6px !important;
    padding: 16px !important;
}

[data-testid="stMetricValue"] div {
    color: var(--term-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-shadow: var(--term-glow-cyan);
    font-weight: 600 !important;
}

[data-testid="stMetricLabel"] div {
    color: var(--term-text-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.7rem !important;
}

/* ===== DATA FRAMES & TABLES ===== */
[data-testid="stDataFrame"], .stDataFrame {
    border: 1px solid var(--term-green) !important;
    border-radius: 6px !important;
    overflow: hidden;
    box-shadow: var(--term-glow-green);
}

[data-testid="stDataFrame"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ===== BUTTONS ===== */
.stButton > button, .stDownloadButton > button {
    background: transparent !important;
    color: var(--term-green) !important;
    border: 1px solid var(--term-green) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.78rem !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: rgba(0,255,156,0.1) !important;
    box-shadow: var(--term-glow-green) !important;
    transform: translateY(-1px);
}

/* ===== ALERTS / INFO / WARNING / ERROR / SUCCESS ===== */
[data-testid="stAlert"], .stAlert {
    background: var(--term-bg-card) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

div[data-testid="stAlert"][data-baseweb*="info"],
.element-container .stAlert:has(svg[data-icon="info"]) {
    border-left: 3px solid var(--term-cyan) !important;
}

/* ===== CODE BLOCKS ===== */
code, .stCode, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--term-bg) !important;
    color: var(--term-green) !important;
    border: 1px solid var(--term-border) !important;
    border-radius: 4px !important;
}

/* ===== CONTAINERS / EXPANDERS ===== */
[data-testid="stExpander"] {
    background: var(--term-bg-card) !important;
    border: 1px solid var(--term-border) !important;
    border-radius: 6px !important;
}

[data-testid="stExpander"]:hover {
    border-color: var(--term-cyan) !important;
}

[data-testid="stExpander"] summary span {
    color: var(--term-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] > div[style*="border"] {
    background: var(--term-bg-card) !important;
    border: 1px solid var(--term-border) !important;
    border-radius: 6px !important;
}

/* ===== SELECTBOX / MULTISELECT / INPUTS ===== */
.stSelectbox, .stMultiSelect, .stTextInput {
    font-family: 'JetBrains Mono', monospace !important;
}

.stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div {
    background: var(--term-bg-card) !important;
    border-color: var(--term-border) !important;
    color: var(--term-text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stSelectbox > div > div:focus-within, .stMultiSelect > div > div:focus-within {
    border-color: var(--term-green) !important;
    box-shadow: var(--term-glow-green) !important;
}

/* ===== RADIO BUTTONS ===== */
.stRadio > label {
    color: var(--term-text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: var(--term-bg-secondary) !important;
    border-bottom: 1px solid var(--term-border) !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--term-text-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTabs [aria-selected="true"] {
    color: var(--term-green) !important;
    border-bottom-color: var(--term-green) !important;
}

/* ===== DIVIDER ===== */
hr, [data-testid="stDivider"] {
    border-color: var(--term-border) !important;
    opacity: 0.5;
}

/* ===== CUSTOM TERMINAL CLASSES ===== */
.terminal-box {
    background: var(--term-bg-card);
    border: 1px solid var(--term-border);
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    position: relative;
}

.terminal-box::before {
    content: "●  ●  ●";
    position: absolute;
    top: 8px;
    left: 14px;
    font-size: 8px;
    color: var(--term-text-dim);
    letter-spacing: 3px;
}

.terminal-header {
    color: var(--term-green);
    text-shadow: var(--term-glow-green);
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 12px;
    padding-top: 8px;
}

.terminal-prompt {
    color: var(--term-cyan);
    font-size: 0.85rem;
}

.terminal-prompt::before {
    content: "❯ ";
    color: var(--term-green);
}

.status-online { color: var(--term-green); text-shadow: var(--term-glow-green); }
.status-offline { color: var(--term-red); }
.status-pending { color: var(--term-amber); }

.glow-text {
    color: var(--term-green);
    text-shadow: var(--term-glow-green);
}

.cyan-glow {
    color: var(--term-cyan);
    text-shadow: var(--term-glow-cyan);
}

/* Blinking cursor animation */
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}

.cursor-blink::after {
    content: "█";
    animation: blink 1s infinite;
    color: var(--term-green);
}

/* Pulse animation for status indicators */
@keyframes pulse-glow {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

.pulse {
    animation: pulse-glow 2s ease-in-out infinite;
}

/* Boot sequence animation */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeInUp 0.6s ease-out forwards;
}

/* Typing animation */
@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

.typing-text {
    overflow: hidden;
    white-space: nowrap;
    animation: typing 2s steps(40) forwards;
    border-right: 2px solid var(--term-green);
}

/* Matplotlib figure background */
.stPlotlyChart, [data-testid="stImage"] {
    background: var(--term-bg-card) !important;
    border: 1px solid var(--term-border) !important;
    border-radius: 6px !important;
    padding: 8px !important;
}

/* Caption styling */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--term-text-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--term-bg); }
::-webkit-scrollbar-thumb { background: var(--term-border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--term-green); }

/* Sidebar title glow */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] .stTitle {
    color: var(--term-green) !important;
    text-shadow: var(--term-glow-green) !important;
}

/* Matplotlib dark background (injected via st.pyplot) */
.stPlot > div {
    background: var(--term-bg-card) !important;
}
</style>
"""

BOOT_SEQUENCE_HTML = """
<div style="
    background: #0a0e14;
    border: 1px solid #1a2332;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
    position: relative;
">
    <div style="position: absolute; top: 10px; left: 14px; font-size: 8px; color: #6e7681; letter-spacing: 3px;">●  ●  ●</div>
    <div style="padding-top: 12px; font-size: 0.78rem; line-height: 1.8;">
        <span style="color: #6e7681;">[</span><span style="color: #00ff9c;">SYS</span><span style="color: #6e7681;">]</span>
        <span style="color: #c9d1d9;">Initializing Higgs Event Lab v0.1.0-dev</span><br>
        <span style="color: #6e7681;">[</span><span style="color: #00e5ff;">LOAD</span><span style="color: #6e7681;">]</span>
        <span style="color: #c9d1d9;">Feature schema verified — 28 input features locked</span><br>
        <span style="color: #6e7681;">[</span><span style="color: #00e5ff;">LOAD</span><span style="color: #6e7681;">]</span>
        <span style="color: #c9d1d9;">Artifact loader initialized — scanning outputs/</span><br>
        <span style="color: #6e7681;">[</span><span style="color: #00ff9c;">OK</span><span style="color: #6e7681;">]</span>
        <span style="color: #00ff9c; text-shadow: 0 0 10px rgba(0,255,156,0.3);">{status_line}</span><br>
        <span style="color: #6e7681;">[</span><span style="color: #00ff9c;">OK</span><span style="color: #6e7681;">]</span>
        <span style="color: #00ff9c; text-shadow: 0 0 10px rgba(0,255,156,0.3);">System online — awaiting operator input</span>
        <span style="color: #00ff9c; animation: blink 1s infinite;">█</span>
    </div>
</div>
<style>@keyframes blink {{ 0%,50% {{ opacity:1; }} 51%,100% {{ opacity:0; }} }}</style>
"""


def main() -> None:
    """Run the Higgs Event Lab Streamlit application."""
    st.set_page_config(
        page_title="Higgs Event Lab | Terminal",
        page_icon="⬡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject terminal CSS
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 10px 0 5px 0;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #00ff9c;
                         text-shadow: 0 0 15px rgba(0,255,156,0.4), 0 0 30px rgba(0,255,156,0.15);
                         letter-spacing: 2px; font-weight: 700;">⬡ HIGGS LAB</span><br>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #6e7681;
                         letter-spacing: 3px; text-transform: uppercase;">Event Classification Terminal</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    nav_selection = st.sidebar.radio(
        "NAVIGATION",
        options=[
            "◈ sys.overview",
            "◈ data.features",
            "◈ run.benchmarks",
            "◈ diag.pipeline",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 0 5px;">
            <div style="color: #6e7681; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 2px; font-size: 0.6rem;">
                session info
            </div>
            <div style="color: #c9d1d9; margin-bottom: 4px;">
                <span style="color: #6e7681;">source</span>
                <span style="color: #00e5ff;"> UCI HIGGS #280</span>
            </div>
            <div style="color: #c9d1d9; margin-bottom: 4px;">
                <span style="color: #6e7681;">owner</span>
                <span style="color: #00ff9c;"> ronit.singh</span>
            </div>
            <div style="color: #c9d1d9; margin-bottom: 4px;">
                <span style="color: #6e7681;">phase</span>
                <span style="color: #ffb300;"> MID-06</span>
            </div>
            <div style="color: #c9d1d9; margin-bottom: 8px;">
                <span style="color: #6e7681;">build</span>
                <span style="color: #6e7681;"> v0.1.0-dev</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load system state and experiments
    pipeline_status = inspect_pipeline_status(PROJECT_ROOT)
    loaded_data = load_experiments(PROJECT_ROOT)

    # Boot sequence
    n_exp = len(loaded_data.experiments)
    n_curves = len(loaded_data.curves)
    if n_exp > 0:
        status_line = f"Loaded {n_exp} experiments, {n_curves} curve sets — all systems nominal"
    else:
        status_line = "No experiment artifacts detected — scaffold mode active"

    st.markdown(
        BOOT_SEQUENCE_HTML.format(status_line=status_line),
        unsafe_allow_html=True,
    )

    # Route navigation
    if nav_selection == "◈ sys.overview":
        render_project_overview()

    elif nav_selection == "◈ data.features":
        render_feature_catalog()

    elif nav_selection == "◈ run.benchmarks":
        if loaded_data.is_empty:
            st.markdown(
                '<h1 style="font-size:1.6rem;">run.benchmarks</h1>',
                unsafe_allow_html=True,
            )
            render_empty_state(pipeline_status, loaded_data)
        else:
            render_results_view(loaded_data)

    elif nav_selection == "◈ diag.pipeline":
        st.markdown(
            '<h1 style="font-size:1.6rem;">diag.pipeline</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color: #6e7681; font-size: 0.82rem;">'
            "Verifies folder contracts, dataset extracts, model weights, and evaluation files.</p>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            _render_diag_card("DATA & ARTIFACT DIRECTORIES", [
                _status_row("data/raw/", pipeline_status.has_raw_data,
                           f"{len(pipeline_status.raw_files)} files" if pipeline_status.raw_files else "empty"),
                _status_row("data/processed/", pipeline_status.has_processed_data,
                           f"{len(pipeline_status.processed_files)} files" if pipeline_status.processed_files else "empty"),
                _status_row("artifacts/", pipeline_status.has_artifacts,
                           f"{len(pipeline_status.artifact_files)} files" if pipeline_status.artifact_files else "empty"),
                _status_row("outputs/", pipeline_status.has_outputs,
                           f"{len(pipeline_status.output_files)} files" if pipeline_status.output_files else "empty"),
            ])

        with col2:
            _render_diag_card("SAMPLE MANIFESTS", [])
            if pipeline_status.manifest_paths:
                for mf in pipeline_status.manifest_paths:
                    st.code(mf, language="text")
            else:
                st.markdown(
                    '<p style="color: #ffb300; font-size: 0.8rem;">⚠ No .manifest.json files found</p>',
                    unsafe_allow_html=True,
                )

        if loaded_data.errors:
            st.markdown("### ⚠ Schema Validation Errors")
            for err in loaded_data.errors:
                st.error(err)
        elif loaded_data.warnings:
            st.markdown("### ℹ Warnings")
            for warn in loaded_data.warnings:
                st.warning(warn)
        else:
            st.markdown(
                '<div style="background: rgba(0,255,156,0.05); border: 1px solid rgba(0,255,156,0.2); '
                'border-radius: 6px; padding: 14px; margin-top: 16px; font-size: 0.82rem;">'
                '<span style="color: #00ff9c; text-shadow: 0 0 10px rgba(0,255,156,0.3);">'
                "✓ All folder contracts and schema validations passed</span></div>",
                unsafe_allow_html=True,
            )


def _status_row(label: str, ok: bool, detail: str) -> str:
    color = "#00ff9c" if ok else "#ff3d71"
    icon = "●" if ok else "○"
    return (
        f'<div style="margin: 6px 0; font-size: 0.8rem;">'
        f'<span style="color: {color};">{icon}</span> '
        f'<code style="color: #c9d1d9; background: none; border: none;">{label}</code> '
        f'<span style="color: #6e7681;">— {detail}</span></div>'
    )


def _render_diag_card(title: str, rows: list[str]) -> None:
    inner = "".join(rows)
    st.markdown(
        f'<div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 18px; margin-bottom: 12px;">'
        f'<div style="color: #00e5ff; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; '
        f'margin-bottom: 12px; font-weight: 600;">// {title}</div>'
        f'{inner}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
