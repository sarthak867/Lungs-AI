from __future__ import annotations

import streamlit as st
from services.inference import AnalysisResult


def render_predictions(result: AnalysisResult) -> None:
    if not result.predictions:
        st.warning("The model returned no supported pathology outputs.")
        return

    # Sort predictions by score descending for scientific clarity
    sorted_predictions = sorted(result.predictions, key=lambda p: p.score, reverse=True)
    top_prediction = sorted_predictions[0] if sorted_predictions else None
    top_label = _escape(top_prediction.label) if top_prediction else "None"
    top_score_str = f"{top_prediction.score:.4f}" if top_prediction else "0.0000"

    # Summary KPI Cards
    st.markdown(
        f"""
        <div class="results-header-card">
            <div class="results-kpi-row">
                <div class="kpi-cell">
                    <span class="kpi-label">Top Indicated Feature</span>
                    <span class="kpi-value highlight-cyan">{top_label}</span>
                    <span class="kpi-sub">Relative Score: {top_score_str}</span>
                </div>
                <div class="kpi-cell">
                    <span class="kpi-label">Evaluated Labels</span>
                    <span class="kpi-value">{len(result.predictions)} Heads</span>
                    <span class="kpi-sub">Multi-label DenseNet-121</span>
                </div>
                <div class="kpi-cell">
                    <span class="kpi-label">Calibration Type</span>
                    <span class="kpi-value text-muted-val">{"Sigmoid" if "sigmoid" in result.score_semantics else "Raw Logit"}</span>
                    <span class="kpi-sub">Relative Activation Scale</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Caution Notice
    st.markdown(
        """
        <div class="scientific-notice">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <span>Model outputs reflect pretrained weights activation. They do not constitute diagnostic confirmation or clinical severity assessment.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive view controls
    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        sort_mode = st.radio(
            "Sort Order",
            ["Highest Activation", "Standard Checkpoint Order"],
            horizontal=True,
            key="score_sort_mode",
            label_visibility="collapsed",
        )
    with col_ctrl2:
        filter_mode = st.radio(
            "Filter Scope",
            ["All 18 Pathologies", "Top 6 Findings"],
            horizontal=True,
            key="score_filter_mode",
            label_visibility="collapsed",
        )

    # Determine display list
    if sort_mode == "Highest Activation":
        display_items = sorted_predictions
    else:
        display_items = list(result.predictions)

    if filter_mode == "Top 6 Findings":
        display_items = sorted(display_items, key=lambda p: p.score, reverse=True)[:6]

    max_magnitude = max((abs(p.score) for p in result.predictions), default=1.0)
    if max_magnitude == 0.0:
        max_magnitude = 1.0

    st.markdown('<div class="scores-container">', unsafe_allow_html=True)
    for rank, prediction in enumerate(display_items, start=1):
        if "sigmoid" in result.score_semantics and "raw model output" not in result.score_semantics:
            # Normalized score range (0.0 to 1.0)
            norm_val = min(1.0, max(0.0, prediction.score))
            pct_width = norm_val * 100.0
            is_prominent = norm_val > 0.45
        else:
            # Relative magnitude within this result
            norm_val = abs(prediction.score) / max_magnitude
            pct_width = norm_val * 100.0
            is_prominent = norm_val > 0.65

        fill_class = "score-fill-prominent" if is_prominent else "score-fill-standard"
        badge_rank = f"{rank:02d}"

        st.markdown(
            f"""
            <div class="modern-score-row">
                <div class="score-meta">
                    <span class="score-rank">{badge_rank}</span>
                    <span class="score-name">{_escape(prediction.label)}</span>
                    <span class="score-num font-mono">{prediction.score:.4f}</span>
                </div>
                <div class="modern-score-track">
                    <div class="{fill_class}" style="width: {pct_width:.1f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Technical interpretation note
    st.markdown(
        f"""
        <div class="score-semantics-caption">
            <span class="semantics-tag">SEMANTICS</span>
            <span>{_escape(result.score_semantics)}. Length of bar represents relative activation magnitude within this inference pass.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
