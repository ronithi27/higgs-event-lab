"""Feature dictionary and schema browser displaying verified dataset definitions."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from app.utils.loader import load_feature_dictionary


def render_feature_catalog() -> None:
    """Render interactive feature dictionary with terminal styling."""
    st.markdown(
        '<h1 style="font-size:1.6rem;">data.features</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color: #6e7681; font-size: 0.82rem;">'
        'Verified schema from '
        '<span style="color: #00e5ff;">UCI ML Repository #280</span> and '
        '<code style="color: #00ff9c; background: none; border: none;">docs/feature_dictionary.csv</code></p>',
        unsafe_allow_html=True,
    )

    try:
        df = load_feature_dictionary()
    except Exception as exc:
        st.error(f"Failed to load feature dictionary: {exc}")
        return

    # Metric summaries
    n_total_cols = len(df)
    n_target = len(df[df["group"] == "target"])
    n_low = len(df[df["group"] == "low_level"])
    n_high = len(df[df["group"] == "high_level"])

    # Terminal-styled metric cards
    metrics_html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;">
        <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="color: #6e7681; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Raw Columns</div>
            <div style="color: #00e5ff; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px rgba(0,229,255,0.3);">{n_total_cols}</div>
        </div>
        <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="color: #6e7681; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Target Variable</div>
            <div style="color: #ff3d71; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px rgba(255,61,113,0.3);">{n_target}</div>
        </div>
        <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="color: #6e7681; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Low-Level</div>
            <div style="color: #00ff9c; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px rgba(0,255,156,0.3);">{n_low}</div>
        </div>
        <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="color: #6e7681; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">High-Level</div>
            <div style="color: #c471f5; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px rgba(196,113,245,0.3);">{n_high}</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    # Safety invariant
    st.markdown(
        """
        <div style="background: rgba(0,229,255,0.04); border: 1px solid rgba(0,229,255,0.2);
                    border-radius: 6px; padding: 14px; margin-bottom: 20px; font-size: 0.8rem;">
            <span style="color: #00e5ff;">🔒 SAFETY INVARIANT</span>
            <span style="color: #c9d1d9;"> — Column 0 (</span><code style="color: #ff3d71; background: none; border: none;">class_label</code><span style="color: #c9d1d9;">) is strictly excluded from model input matrix X and all fitted scalers.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filtering options
    st.markdown(
        '<div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 2px; '
        'margin-bottom: 10px; font-weight: 600;">// FILTER</div>',
        unsafe_allow_html=True,
    )

    col_filter, col_search = st.columns([1, 1])
    with col_filter:
        filter_choice = st.radio(
            "Feature Group:",
            options=["All (29)", "Low-Level (21)", "High-Level (7)", "Target (1)"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with col_search:
        search_term = st.text_input("Search features:", "", placeholder="e.g. momentum, eta, mass...")

    filtered_df = df.copy()
    if filter_choice == "Low-Level (21)":
        filtered_df = filtered_df[filtered_df["group"] == "low_level"]
    elif filter_choice == "High-Level (7)":
        filtered_df = filtered_df[filtered_df["group"] == "high_level"]
    elif filter_choice == "Target (1)":
        filtered_df = filtered_df[filtered_df["group"] == "target"]

    if search_term.strip():
        term = search_term.strip().lower()
        filtered_df = filtered_df[
            filtered_df["name"].str.lower().str.contains(term)
            | filtered_df["description"].str.lower().str.contains(term)
        ]

    st.markdown(
        f'<div style="color: #6e7681; font-size: 0.72rem; margin-bottom: 8px;">'
        f'Showing {len(filtered_df)} of {len(df)} entries</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "file_column_index": st.column_config.NumberColumn("Col#", format="%d", width="small"),
            "model_input_index": st.column_config.NumberColumn("Input#", format="%d", width="small"),
            "name": st.column_config.TextColumn("Feature", width="medium"),
            "group": st.column_config.TextColumn("Group", width="small"),
            "kind": st.column_config.TextColumn("Kind", width="small"),
            "description": st.column_config.TextColumn("Description", width="large"),
        },
    )

    # Physics glossary
    with st.expander("📖 Physics Terminology Reference"):
        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #c9d1d9; line-height: 1.9;">
                <span style="color: #00e5ff;">pT</span> <span style="color: #6e7681;">—</span>
                Transverse momentum perpendicular to beam line<br>
                <span style="color: #00e5ff;">η</span> <span style="color: #6e7681;">—</span>
                Pseudorapidity (angle relative to beam axis)<br>
                <span style="color: #00e5ff;">φ</span> <span style="color: #6e7681;">—</span>
                Azimuthal angle in the transverse plane<br>
                <span style="color: #00e5ff;">ET_miss</span> <span style="color: #6e7681;">—</span>
                Missing transverse energy from undetected neutrinos<br>
                <span style="color: #00e5ff;">b-tag</span> <span style="color: #6e7681;">—</span>
                Probability a jet originated from a bottom quark<br>
                <span style="color: #00e5ff;">m_jj, m_bb, ...</span> <span style="color: #6e7681;">—</span>
                Reconstructed invariant masses from 4-momentum vectors
            </div>
            """,
            unsafe_allow_html=True,
        )
