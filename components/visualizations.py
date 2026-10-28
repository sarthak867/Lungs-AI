from __future__ import annotations

import streamlit as st

from services.inference import AnalysisResult


def render_predictions(result: AnalysisResult) -> None:
    if not result.predictions:
        st.warning("The model returned no supported pathology outputs.")
        return

    sorted_predictions = sorted(result.predictions, key=lambda prediction: prediction.score, reverse=True)
    findings = result.findings
    top_prediction = max(findings or result.predictions, key=lambda prediction: prediction.score)
    top_label = _display_label(top_prediction.label)
    finding_labels = ", ".join(_display_label(prediction.label) for prediction in findings)

    st.markdown(
        f"""
        <div class="results-header-card">
            <div class="results-kpi-row">
                <div class="kpi-cell">
                    <span class="kpi-label">Highest Scoring Output</span>
                    <span class="kpi-value highlight-cyan">{_escape(top_label)}</span>
                    <span class="kpi-sub">Sigmoid score: {top_prediction.score:.4f}</span>
                </div>
                <div class="kpi-cell">
                    <span class="kpi-label">Threshold-Crossing Labels</span>
                    <span class="kpi-value">{len(findings)} / {len(result.predictions)}</span>
                    <span class="kpi-sub">Per-label validation cutoffs</span>
                </div>
                <div class="kpi-cell">
                    <span class="kpi-label">Output Type</span>
                    <span class="kpi-value text-muted-val">14 independent sigmoid scores</span>
                    <span class="kpi-sub">Not mutually exclusive</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Model scores meet the supplied research cutoffs for: " + finding_labels
        if findings
        else "No model output crossed its supplied per-disease cutoff. This does not establish that the image is normal."
    )

    st.markdown(
        """
        <div class="scientific-notice">
            <span>Research outputs only: these scores are not calibrated clinical probabilities or diagnoses. Thresholds were selected on the validation set and may produce false positives and false negatives.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        sort_mode = st.radio(
            "Sort order",
            ["Highest score", "Model label order"],
            horizontal=True,
            key="score_sort_mode",
            label_visibility="collapsed",
        )
    with col_ctrl2:
        filter_mode = st.radio(
            "Filter scope",
            ["All 14 labels", "Threshold-crossing labels"],
            horizontal=True,
            key="score_filter_mode",
            label_visibility="collapsed",
        )

    display_items = sorted_predictions if sort_mode == "Highest score" else list(result.predictions)
    if filter_mode == "Threshold-crossing labels":
        display_items = [prediction for prediction in display_items if prediction.meets_threshold]

    if not display_items:
        st.caption("No labels meet their individual thresholds for this image.")
    st.markdown('<div class="scores-container">', unsafe_allow_html=True)
    for rank, prediction in enumerate(display_items, start=1):
        pct_width = min(100.0, max(0.0, prediction.score * 100.0))
        fill_class = "score-fill-prominent" if prediction.meets_threshold else "score-fill-standard"
        threshold_text = (
            f"cutoff {prediction.threshold:.3f} · "
            f"{'meets cutoff' if prediction.meets_threshold else 'below cutoff'}"
            if prediction.threshold is not None
            else "no cutoff configured"
        )
        st.markdown(
            f"""
            <div class="modern-score-row">
                <div class="score-meta">
                    <span class="score-rank">{rank:02d}</span>
                    <span class="score-name">{_escape(_display_label(prediction.label))}</span>
                    <span class="score-num font-mono">{prediction.score:.4f} · {_escape(threshold_text)}</span>
                </div>
                <div class="modern-score-track">
                    <div class="{fill_class}" style="width: {pct_width:.1f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="score-semantics-caption">
            <span class="semantics-tag">SEMANTICS</span>
            <span>{_escape(result.score_semantics)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_label(label: str) -> str:
    return label.replace("_", " ")


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
