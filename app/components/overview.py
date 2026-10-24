"""Project overview, scientific boundaries, team responsibilities, and architecture."""

from __future__ import annotations

import streamlit as st


def render_project_overview() -> None:
    """Render the project overview page with terminal-styled verified context."""

    st.markdown(
        '<h1 style="font-size:1.6rem;">sys.overview</h1>',
        unsafe_allow_html=True,
    )

    # Scientific Boundary Box — terminal style
    st.markdown(
        """
        <div style="background: rgba(255,61,113,0.04); border: 1px solid rgba(255,61,113,0.25);
                    border-radius: 6px; padding: 18px; margin-bottom: 20px; font-family: 'JetBrains Mono', monospace;">
            <div style="color: #ff3d71; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 2px;
                        margin-bottom: 10px; font-weight: 600;">
                ⚠ SCOPE GUARDRAILS
            </div>
            <div style="color: #c9d1d9; font-size: 0.8rem; line-height: 1.7;">
                This project classifies simulated collision events labelled as
                <span style="color: #00ff9c;">signal (1.0)</span> or
                <span style="color: #ff3d71;">background (0.0)</span>.
                It does <b>not</b> simulate the Higgs quantum field, claim particle discovery,
                produce detector-grade analysis, or convert confidence scores into physical discovery significance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        # Problem formulation
        st.markdown(
            """
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px;
                        padding: 20px; margin-bottom: 16px;">
                <div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 14px; font-weight: 600;">
                    // PROBLEM FORMULATION
                </div>
                <div style="color: #c9d1d9; font-size: 0.8rem; line-height: 1.9;">
                    <span style="color: #6e7681;">task</span>
                    <span style="color: #00ff9c;">Binary supervised classification</span> of Monte Carlo LHC events<br>
                    <span style="color: #6e7681;">target</span>
                    <code style="color: #00e5ff; background: none; border: none;">class_label</code> —
                    H→WW*→ℓνjj vs tt̄ background<br>
                    <span style="color: #6e7681;">inputs</span>
                    <span style="color: #c9d1d9;">21 low-level · 7 high-level · 28 combined</span><br>
                    <span style="color: #6e7681;">models</span>
                    <span style="color: #c9d1d9;">Logistic Regression · XGBoost · Compact MLP</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Research questions
        st.markdown(
            """
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px;
                        padding: 20px; margin-bottom: 16px;">
                <div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 14px; font-weight: 600;">
                    // CORE QUESTIONS
                </div>
                <div style="color: #c9d1d9; font-size: 0.8rem; line-height: 1.9;">
                    <span style="color: #00ff9c;">01</span>
                    <span style="color: #6e7681;">│</span>
                    Does deep learning eliminate the need for expert-derived invariant mass features?<br>
                    <span style="color: #00ff9c;">02</span>
                    <span style="color: #6e7681;">│</span>
                    What are the trade-offs in ROC-AUC, PR-AUC, calibration, and training runtime?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Leakage controls
        st.markdown(
            """
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px;
                        padding: 20px; margin-bottom: 16px;">
                <div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 14px; font-weight: 600;">
                    // LEAKAGE CONTROLS
                </div>
                <div style="color: #c9d1d9; font-size: 0.8rem; line-height: 1.9;">
                    <span style="color: #00ff9c;">✓</span> Target column strictly excluded from X and scalers<br>
                    <span style="color: #00ff9c;">✓</span> Split-before-fit on all preprocessors<br>
                    <span style="color: #00ff9c;">✓</span> Frozen held-out test set across all models<br>
                    <span style="color: #00ff9c;">✓</span> Operating thresholds chosen on validation, frozen for test
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Team roster
        st.markdown(
            """
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px;
                        padding: 20px; margin-bottom: 16px;">
                <div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 14px; font-weight: 600;">
                    // TEAM ROSTER
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        team_members = [
            ("Krishna", "LEAD", "ML Experiment Design, Integration & Deployment", "#00ff9c"),
            ("Saumya Mishra", "QA", "Documentation, Bug Triage, Verification", "#00e5ff"),
            ("Aditya Rana", "DATA", "Data Pipeline, EDA, LogReg & XGBoost", "#c471f5"),
            ("Rachit Gupta", "DL", "Compact MLP, Hyperparameter Tuning", "#ffb300"),
            ("Ronit Singh", "UI", "Streamlit Dashboard, Artifact Loader", "#ff3d71"),
        ]

        for name, role_tag, duties, color in team_members:
            st.markdown(
                f"""
                <div style="background: #0d1117; border-left: 3px solid {color};
                            padding: 10px 14px; margin: 6px 0; border-radius: 0 6px 6px 0;">
                    <div style="font-size: 0.82rem;">
                        <span style="color: {color}; font-weight: 600;">{name}</span>
                        <span style="background: {color}20; color: {color}; padding: 2px 8px;
                                     border-radius: 3px; font-size: 0.6rem; margin-left: 8px;
                                     font-weight: 600; letter-spacing: 1px;">{role_tag}</span>
                    </div>
                    <div style="color: #6e7681; font-size: 0.72rem; margin-top: 4px;">{duties}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Citations
        st.markdown(
            """
            <div style="background: #111820; border: 1px solid #1a2332; border-radius: 8px;
                        padding: 20px; margin-top: 16px;">
                <div style="color: #00e5ff; font-size: 0.72rem; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 14px; font-weight: 600;">
                    // REFERENCES
                </div>
                <div style="color: #6e7681; font-size: 0.75rem; line-height: 1.8;">
                    <span style="color: #c9d1d9;">[1]</span> Baldi, Sadowski, & Whiteson (2014).
                    <i>Searching for Exotic Particles in HEP with Deep Learning</i>. Nature Comms 5, 4308.<br>
                    <span style="color: #c9d1d9;">[2]</span> Whiteson, D. (2014).
                    <i>HIGGS Data Set</i>. UCI ML Repository. DOI: 10.24432/C55307.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Limitations section
    st.markdown(
        """
        <div style="background: rgba(255,179,0,0.04); border: 1px solid rgba(255,179,0,0.2);
                    border-radius: 8px; padding: 20px; margin-top: 20px;">
            <div style="color: #ffb300; font-size: 0.72rem; text-transform: uppercase;
                        letter-spacing: 2px; margin-bottom: 12px; font-weight: 600;">
                // KNOWN LIMITATIONS
            </div>
            <div style="color: #c9d1d9; font-size: 0.78rem; line-height: 1.9;">
                <span style="color: #ffb300;">▸</span> <b>Simulation Gap</b> — benchmark uses simulated H→WW* events, not real detector data<br>
                <span style="color: #ffb300;">▸</span> <b>Class Balance</b> — benchmark mixture ≠ physical beam rates; calibration doesn't transfer<br>
                <span style="color: #ffb300;">▸</span> <b>Sample Scale</b> — initial experiments target 250k–1M events, not the full 11M dataset<br>
                <span style="color: #ffb300;">▸</span> <b>No Systematics</b> — real analyses are dominated by detector/theoretical uncertainties
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
