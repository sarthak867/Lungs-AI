from __future__ import annotations

import logging

import streamlit as st
from PIL import Image

from components.sidebar import render_sidebar
from components.visualizations import render_predictions
from config import APP_NAME, PROJECT_ROOT, XRAY_CLASSIFIER_THRESHOLD
from services.explainability import ExplainabilityUnavailableError, generate_gradcam
from services.inference import (
    AnalysisResult,
    InferenceError,
    ModelUnavailableError,
    disease_model_revision,
    run_inference,
)
from services.preprocessing import (
    validate_uploaded_file,
)
from services.report_generator import generate_pdf_report
from services.modality_validation import (
    SuitabilityGateError,
    SuitabilityStatus,
    SuitabilityValidationResult,
    suitability_message,
    validate_chest_xray_suitability,
)
from services.upload_state import (
    ANALYSIS_STATE_KEYS,
    UPLOAD_STATE_KEYS,
    attach_suitability_result,
    clear_analysis_state,
    clear_result_state,
    content_sha256,
    register_invalid_upload,
    register_valid_upload,
)
from services.xray_gate_model import classifier_revision
from Disease_model.explanations import DISCLAIMER as DISEASE_DISCLAIMER
from Disease_model.explanations import EXPLANATIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{APP_NAME} — Medical AI Research",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-canvas: #060D17;
    --bg-surface: #0A1526;
    --bg-card: #0E1C2F;
    --bg-card-elevated: #13263E;
    --bg-interactive: #162E4A;
    --accent-cyan: #38BDF8;
    --accent-cyan-hover: #7DD3FC;
    --accent-blue: #3B82F6;
    --accent-indigo: #6366F1;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --border-subtle: rgba(56, 189, 248, 0.12);
    --border-light: rgba(148, 163, 184, 0.12);
    --border-card: rgba(56, 189, 248, 0.18);
    --status-emerald: #10B981;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
}

/* Global resets & typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    letter-spacing: -0.01em;
}

.font-mono {
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background-color: var(--bg-canvas);
    color: var(--text-primary);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Main Container */
.block-container {
    max-width: 1260px;
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Sidebar components */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 1rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border-light);
}

.sidebar-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);
}

.brand-title-row {
    display: flex;
    align-items: center;
    gap: 8px;
}

.brand-name {
    font-weight: 800;
    font-size: 18px;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.brand-tag {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent-cyan);
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.brand-subtext {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* Telemetry Card */
.telemetry-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    margin-bottom: 1.2rem;
}

.telemetry-header {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 8px;
}

.pulse-indicator {
    width: 7px;
    height: 7px;
    background: var(--status-emerald);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--status-emerald);
}

.telemetry-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #6EE7B7;
    text-transform: uppercase;
}

.telemetry-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 10px;
}

.telemetry-item {
    display: flex;
    flex-direction: column;
}

.telemetry-label {
    font-size: 10px;
    color: var(--text-muted);
}

.telemetry-val {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
}

.nav-section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 8px;
    margin-top: 4px;
}

/* Navigation Radio styling */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 6px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 7px 12px;
    transition: all 0.15s ease;
    cursor: pointer;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(56, 189, 248, 0.06);
    border-color: rgba(56, 189, 248, 0.15);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[aria-checked="true"] {
    background: rgba(56, 189, 248, 0.12) !important;
    border-color: rgba(56, 189, 248, 0.35) !important;
}

.sidebar-spacer {
    height: 1.5rem;
}

.privacy-card {
    background: rgba(14, 28, 47, 0.6);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 12px;
    margin-top: 1rem;
}

.privacy-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--accent-cyan);
    margin-bottom: 6px;
}

.privacy-copy {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.5;
    margin: 0 0 8px 0;
}

.disclaimer-badge {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-align: center;
    padding: 3px 6px;
    background: rgba(245, 158, 11, 0.12);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 4px;
}

/* Hero Section */
.hero-container {
    padding: 0.5rem 0 1.5rem 0;
}

.eyebrow-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: var(--accent-cyan);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    font-size: clamp(34px, 4.5vw, 56px);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.035em;
    color: var(--text-primary);
    margin: 0 0 12px 0;
}

.hero-copy {
    color: var(--text-secondary);
    font-size: 17px;
    line-height: 1.6;
    max-width: 680px;
    margin: 0 0 20px 0;
}

.ribbon-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 24px;
}

.ribbon-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: 6px;
    padding: 4px 9px;
}

.ribbon-chip .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-cyan);
}

/* Surfaces & Cards */
.modern-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: 0 4px 20px -2px rgba(6, 13, 23, 0.5);
    margin-bottom: 20px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 24px 0;
}

.feature-card-modern {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 22px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 160px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.feature-card-modern:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.35);
}

.feature-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    color: var(--accent-cyan);
    margin-bottom: 8px;
}

.feature-card-modern h3 {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 8px 0;
}

.feature-card-modern p {
    font-size: 13px;
    line-height: 1.55;
    color: var(--text-secondary);
    margin: 0;
}

/* Workflow Steps */
.workflow-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 20px 0;
}

.workflow-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 18px;
    position: relative;
}

.workflow-step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: var(--accent-cyan);
    background: rgba(56, 189, 248, 0.1);
    padding: 3px 7px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 8px;
}

.workflow-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.workflow-desc {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.45;
    margin: 0;
}

/* Upload Panel styling */
.dropzone-container {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.dropzone-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.dropzone-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
}

.format-badges {
    display: flex;
    gap: 6px;
}

.format-chip {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    padding: 2px 7px;
    border-radius: 4px;
}

/* Uploaded Image Card */
.image-preview-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 16px;
    margin-top: 14px;
}

.image-meta-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-light);
}

.meta-box {
    display: flex;
    flex-direction: column;
}

.meta-box-label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.meta-box-value {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 2px;
}

.status-badge-valid {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    color: #6EE7B7;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 3px 8px;
    border-radius: 9999px;
}

/* Studio Workspace */
.workspace-status-bar {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.status-bar-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.status-item {
    display: flex;
    flex-direction: column;
}

.status-label {
    font-size: 10px;
    color: var(--text-muted);
}

.status-val {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
}

/* Imaging Studio Frame */
.studio-frame-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px;
    height: 100%;
}

.frame-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.inspect-badge {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    background: rgba(56, 189, 248, 0.12);
    color: var(--accent-cyan);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 4px;
    padding: 2px 7px;
}

/* Magma Colormap Bar */
.colormap-container {
    margin: 14px 0 10px 0;
    padding: 10px 14px;
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
}

.colormap-header {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    font-weight: 700;
    color: var(--text-secondary);
    margin-bottom: 6px;
}

.colormap-bar {
    height: 10px;
    border-radius: 9999px;
    background: linear-gradient(to right, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf);
    margin-bottom: 4px;
}

.colormap-labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
}

.attribution-explainer {
    font-size: 12px;
    line-height: 1.5;
    color: var(--text-muted);
    background: rgba(14, 28, 47, 0.5);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    margin-top: 12px;
}

/* Visualizations KPI & Rows */
.results-header-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    margin-bottom: 12px;
}

.results-kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.kpi-cell {
    display: flex;
    flex-direction: column;
}

.kpi-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
}

.kpi-value {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2px 0;
}

.highlight-cyan {
    color: var(--accent-cyan) !important;
}

.text-muted-val {
    color: #94A3B8 !important;
}

.kpi-sub {
    font-size: 11px;
    color: var(--text-muted);
}

.scientific-notice {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: var(--radius-md);
    padding: 9px 12px;
    font-size: 11px;
    color: #BAE6FD;
    margin-bottom: 14px;
}

.scores-container {
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 14px;
    margin-top: 10px;
}

.modern-score-row {
    margin-bottom: 10px;
}

.modern-score-row:last-child {
    margin-bottom: 0;
}

.score-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
}

.score-rank {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    width: 24px;
}

.score-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    flex-grow: 1;
}

.score-num {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent-cyan);
    font-variant-numeric: tabular-nums;
}

.modern-score-track {
    height: 6px;
    width: 100%;
    background: rgba(30, 48, 71, 0.6);
    border-radius: 9999px;
    overflow: hidden;
}

.score-fill-prominent {
    height: 100%;
    background: linear-gradient(90deg, #0284C7 0%, #38BDF8 100%);
    border-radius: 9999px;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.5);
}

.score-fill-standard {
    height: 100%;
    background: linear-gradient(90deg, #334155 0%, #475569 100%);
    border-radius: 9999px;
}

.score-semantics-caption {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 12px;
    line-height: 1.45;
}

.semantics-tag {
    font-size: 9px;
    font-weight: 700;
    color: var(--accent-cyan);
    background: rgba(56, 189, 248, 0.12);
    padding: 2px 5px;
    border-radius: 3px;
    margin-right: 4px;
}

/* Dossier PDF Panel */
.dossier-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 22px;
    margin-top: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.dossier-info h4 {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 4px 0;
}

.dossier-info p {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0;
}

/* Streamlit button overrides */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #38BDF8 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(56, 189, 248, 0.5) !important;
    border-color: #7DD3FC !important;
}

div.stButton > button[kind="secondary"] {
    background: var(--bg-card-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent-cyan) !important;
    background: var(--bg-interactive) !important;
}

div.stDownloadButton > button {
    background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #2DD4BF !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(13, 148, 136, 0.3) !important;
}

div.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(45, 212, 191, 0.4) !important;
}

/* File uploader skin */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}

[data-testid="stFileUploader"] section {
    background-color: rgba(10, 21, 38, 0.6) !important;
    border: 1.5px dashed rgba(56, 189, 248, 0.25) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent-cyan) !important;
    background-color: rgba(14, 28, 47, 0.8) !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
}

/* Responsive adjustments */
@media (max-width: 900px) {
    .feature-grid {
        grid-template-columns: 1fr;
    }
    .workflow-row {
        grid-template-columns: 1fr 1fr;
    }
    .image-meta-grid {
        grid-template-columns: 1fr 1fr;
    }
    .results-kpi-row {
        grid-template-columns: 1fr;
    }
    .dossier-card {
        flex-direction: column;
        gap: 16px;
        align-items: flex-start;
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def clear_upload() -> None:
    for key in ("xray_upload", *UPLOAD_STATE_KEYS, *ANALYSIS_STATE_KEYS, "gradcam_index"):
        st.session_state.pop(key, None)


def load_sample_radiograph() -> None:
    sample_file = PROJECT_ROOT / "assets" / "Xray sample.jpg"
    if sample_file.exists():
        data = sample_file.read_bytes()
        file_result = validate_uploaded_file(data, sample_file.name)
        if not file_result.accepted or file_result.validated_image is None:
            register_invalid_upload(st.session_state, file_result.reason)
            logger.warning("Bundled sample image failed file validation: %s", file_result.reason)
            return
        register_valid_upload(
            st.session_state,
            data,
            sample_file.name,
            file_result.validated_image,
        )
        st.toast("Loaded bundled sample image", icon="🫁")


def navigate_to(page_name: str) -> None:
    st.session_state["target_nav_page"] = page_name


def show_suitability_status(result: SuitabilityValidationResult) -> None:
    message = suitability_message(result)
    if result.status is SuitabilityStatus.ACCEPTED:
        st.success(message)
    elif result.status is SuitabilityStatus.SERVICE_UNAVAILABLE:
        st.warning(message)
    else:
        st.error(message)


def upload_panel(show_analysis_cta: bool = False) -> None:
    st.markdown(
        """
        <div class="dropzone-container">
            <div class="dropzone-header">
                <div class="dropzone-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <span>Image Upload & File Validation</span>
                </div>
                <div class="format-badges">
                    <span class="format-chip">PNG</span>
                    <span class="format-chip">JPG</span>
                    <span class="format-chip">JPEG</span>
                    <span class="format-chip">MAX 10 MB</span>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if "image_bytes" in st.session_state:
        st.button(
            "Upload another file",
            key="upload_another_file",
            icon=":material/upload_file:",
            on_click=clear_upload,
            width="stretch",
        )
    else:
        uploaded = st.file_uploader(
            "Upload chest X-ray image",
            type=["png", "jpg", "jpeg"],
            key="xray_upload",
            help="Upload a readable PNG or JPEG. File checks do not establish that the image is a chest radiograph.",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            data = uploaded.getvalue()
            file_result = validate_uploaded_file(data, uploaded.name)
            if not file_result.accepted or file_result.validated_image is None:
                register_invalid_upload(st.session_state, file_result.reason)
                st.error(file_result.reason)
                logger.info("Rejected uploaded file (%s): %s", file_result.code.value, file_result.reason)
            else:
                digest = register_valid_upload(
                    st.session_state,
                    data,
                    uploaded.name,
                    file_result.validated_image,
                )
                if (
                    st.session_state.get("suitability_validation_hash") != digest
                    or st.session_state.get("suitability_validation_version") != classifier_revision()
                ):
                    with st.spinner("Checking image..."):
                        suitability = validate_chest_xray_suitability(
                            file_result.validated_image.image,
                            digest,
                        )
                    attach_suitability_result(st.session_state, suitability)
                st.rerun()

    # If image is present in session state, render technical inspection card
    if "image_bytes" in st.session_state:
        file_result = validate_uploaded_file(
            st.session_state.image_bytes,
            st.session_state.image_name,
        )
        if file_result.accepted and file_result.validated_image is not None:
            validated = file_result.validated_image
            suitability = st.session_state.get("suitability_validation")
            if (
                not isinstance(suitability, SuitabilityValidationResult)
                or suitability.content_hash != st.session_state.get("image_hash")
                or st.session_state.get("suitability_validation_hash") != st.session_state.get("image_hash")
                or st.session_state.get("suitability_validation_version") != classifier_revision()
            ):
                clear_analysis_state(st.session_state)
                with st.spinner("Checking image..."):
                    suitability = validate_chest_xray_suitability(
                        validated.image,
                        st.session_state.image_hash,
                    )
                attach_suitability_result(st.session_state, suitability)
            st.markdown(
                f"""
                <div class="image-preview-card">
                    <div class="image-meta-grid">
                        <div class="meta-box">
                            <span class="meta-box-label">Filename</span>
                            <span class="meta-box-value">{_escape(st.session_state.image_name)}</span>
                        </div>
                        <div class="meta-box">
                            <span class="meta-box-label">Source Resolution</span>
                            <span class="meta-box-value font-mono">{validated.width} × {validated.height} px</span>
                        </div>
                        <div class="meta-box">
                            <span class="meta-box-label">Disease Model Input</span>
                            <span class="meta-box-value font-mono">320 × 320 Grayscale RGB</span>
                        </div>
                        <div class="meta-box">
                            <span class="meta-box-label">Integrity Status</span>
                            <span class="status-badge-valid">✓ File Decoded</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_img, col_act = st.columns([1, 2])
            with col_img:
                st.image(
                    validated.image,
                    caption=f"{st.session_state.image_name} · Validated image preview",
                    width=240,
                )
            with col_act:
                st.markdown("#### Chest X-ray suitability")
                st.caption(
                    "File validation checks readability. The local checkpoint classifier classifies the image as "
                    "Normal Photo or X-Ray before pathology inference."
                )
                show_suitability_status(suitability)
                if suitability.xray_score is not None:
                    predicted_class = (
                        "X-Ray (class 1)"
                        if suitability.accepted
                        else "Normal Photo (class 0)"
                        if suitability.status is SuitabilityStatus.REJECTED
                        else "Unable to verify"
                    )
                    st.caption(
                        f"Checkpoint classifier decision: {predicted_class} · X-Ray class score: "
                        f"{suitability.xray_score:.3f} · threshold: {XRAY_CLASSIFIER_THRESHOLD:.2f}"
                    )
                if show_analysis_cta:
                    st.button(
                        "Proceed to Analysis Workspace →",
                        type="primary",
                        use_container_width=True,
                        on_click=navigate_to,
                        args=("Analyze X-Ray",),
                        disabled=not suitability.accepted,
                    )

        else:
            register_invalid_upload(st.session_state, file_result.reason)
            st.error(file_result.reason)
    else:
        # Quick Demo Loader
        col_demo1, col_demo2 = st.columns([2, 1])
        with col_demo1:
            st.caption("A bundled sample is available; it will pass through the same local X-ray suitability gate.")
        with col_demo2:
            st.button("Load Sample Image", on_click=load_sample_radiograph, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_overview() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <div class="eyebrow-pill">
                <span class="pulse-indicator"></span>
                <span>DEEP LEARNING RADIOLOGY RESEARCH · DENSENET-121</span>
            </div>
            <h1 class="hero-title">See beyond the X-ray.</h1>
            <p class="hero-copy">Explore AI-assisted chest X-ray analysis with transparent multi-label predictions and gradient-weighted visual explanations.</p>
            <div class="ribbon-badges">
                <div class="ribbon-chip"><span class="dot"></span>Local DenseNet-121 Checkpoint</div>
                <div class="ribbon-chip"><span class="dot"></span>14 Independent Disease Scores</div>
                <div class="ribbon-chip"><span class="dot"></span>Grad-CAM DenseNet Feature Attribution</div>
                <div class="ribbon-chip"><span class="dot"></span>Zero-Retention In-Memory Privacy</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prominent Upload Panel with primary action
    upload_panel(show_analysis_cta=True)
    # Feature Cards
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card-modern">
                <div>
                    <div class="feature-num">MODULE 01</div>
                    <h3>Multi-Target Pathology Profiling</h3>
                    <p>The local `xray_classifier.pth` `Normal Photo` vs `X-Ray` gate checks the upload before the project disease model runs.</p>
                </div>
            </div>
            <div class="feature-card-modern">
                <div>
                    <div class="feature-num">MODULE 02</div>
                    <h3>Visual Grad-CAM Explainability</h3>
                    <p>Grad-CAM is available only after the X-ray gate passes and pathology analysis succeeds.</p>
                </div>
            </div>
            <div class="feature-card-modern">
                <div>
                    <div class="feature-num">MODULE 03</div>
                    <h3>Auditable Research Dossier</h3>
                    <p>Pathology reports are generated only after input validation and analysis succeed.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Workflow Pipeline
    st.markdown("### Algorithmic Pipeline")
    st.markdown(
        """
        <div class="workflow-row">
            <div class="workflow-card">
                <span class="workflow-step-num">STEP 01</span>
                <div class="workflow-title">Validate Input</div>
                        <p class="workflow-desc">Check file integrity, then require a separate modality check before pathology inference.</p>
            </div>
            <div class="workflow-card">
                <span class="workflow-step-num">STEP 02</span>
                <div class="workflow-title">Image Preparation</div>
                <p class="workflow-desc">The suitability gate and disease model each apply their own checkpoint-specific input preprocessing.</p>
            </div>
            <div class="workflow-card">
                <span class="workflow-step-num">STEP 03</span>
                <div class="workflow-title">Gated Disease Inference</div>
                <p class="workflow-desc">Run the local 14-label DenseNet-121 only after the X-ray suitability checkpoint accepts the image.</p>
            </div>
            <div class="workflow-card">
                <span class="workflow-step-num">STEP 04</span>
                <div class="workflow-title">Explainability & Audit</div>
                <p class="workflow-desc">Run Grad-CAM and enable reports only after a matching gate pass and analysis result.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Research Protocol Surface
    st.markdown(
        """
        <div class="modern-card">
            <div class="eyebrow-pill">RESEARCH PROTOCOL & GOVERNANCE</div>
            <p style="color:var(--text-secondary); font-size:13px; line-height:1.6; margin:6px 0 0 0;">
                LungLens AI uses the project-provided fine-tuned DenseNet-121 checkpoint and a separate X-ray input gate.
                This software is intended strictly for algorithm development, educational demonstrations, and biomedical machine learning inspection.
                It has not been cleared as a medical device by the FDA or international regulatory authorities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_analysis() -> None:
    st.markdown(
        """
        <div class="hero-container" style="padding-bottom: 0.5rem;">
            <div class="eyebrow-pill">ANALYSIS WORKSPACE</div>
            <h1 class="hero-title" style="font-size:38px;">Analyze Radiograph</h1>
            <p class="hero-copy" style="font-size:15px; margin-bottom:12px;">Run the local 14-label DenseNet-121 model and inspect class-specific Grad-CAM attribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_panel(show_analysis_cta=False)

    image_data = st.session_state.get("image_bytes")
    if not image_data:
        st.info("Upload a PNG or JPEG image. The local xray_classifier.pth checks X-ray suitability before pathology analysis.")
        return

    file_result = validate_uploaded_file(image_data, st.session_state.image_name)
    if not file_result.accepted or file_result.validated_image is None:
        register_invalid_upload(st.session_state, file_result.reason)
        st.error(file_result.reason)
        return
    image = file_result.validated_image.image
    image_hash = st.session_state.get("image_hash") or content_sha256(image_data)
    validation = st.session_state.get("suitability_validation")
    if (
        not isinstance(validation, SuitabilityValidationResult)
        or validation.content_hash != image_hash
        or st.session_state.get("suitability_validation_hash") != image_hash
        or st.session_state.get("suitability_validation_version") != classifier_revision()
    ):
        clear_analysis_state(st.session_state)
        with st.spinner("Checking image..."):
            validation = validate_chest_xray_suitability(image, image_hash)
        attach_suitability_result(st.session_state, validation)
    if not validation.accepted:
        clear_result_state(st.session_state)
        return

    if st.session_state.get("analysis_hash") != image_hash:
        clear_result_state(st.session_state)

    # Action Toolbar
    col_run, col_report = st.columns([1, 1])
    with col_run:
        run_requested = st.button(
            "⚡ Run DenseNet-121 Analysis",
            type="primary",
            width="stretch",
            key="run_analysis_btn",
        )
    report_slot = col_report.empty()

    if run_requested:
        clear_result_state(st.session_state)
        with st.spinner("Rechecking chest X-ray suitability..."):
            validation = validate_chest_xray_suitability(image, image_hash)
        attach_suitability_result(st.session_state, validation)
        if not validation.accepted:
            clear_result_state(st.session_state)
            st.error(suitability_message(validation))
        else:
            with st.spinner("Executing local DenseNet-121 analysis and generating attribution…"):
                try:
                    result = run_inference(
                        image,
                        image_hash=image_hash,
                        validation=validation,
                    )
                    st.session_state.analysis = result
                    st.session_state.analysis_hash = image_hash

                    gradcams = {}
                    gradcam_errors = {}
                    for finding in result.findings:
                        target_idx = result.pathologies.index(finding.label)
                        try:
                            gradcams[finding.label] = generate_gradcam(
                                image,
                                result,
                                target_index=target_idx,
                                image_hash=image_hash,
                                validation=validation,
                            )
                        except (ExplainabilityUnavailableError, SuitabilityGateError) as exc:
                            gradcam_errors[finding.label] = (
                                suitability_message(validation)
                                if isinstance(exc, SuitabilityGateError)
                                else str(exc)
                            )
                            logger.warning(
                                "Grad-CAM unavailable for threshold-crossing finding %s: %s",
                                finding.label,
                                exc,
                            )
                    st.session_state.gradcams = gradcams
                    st.session_state.gradcam_errors = gradcam_errors
                    st.session_state.gradcam_label = result.findings[0].label if result.findings else None
                    st.session_state.gradcam = (
                        gradcams.get(st.session_state.gradcam_label)
                        if st.session_state.gradcam_label
                        else None
                    )
                    st.session_state.gradcam_error = None

                except ModelUnavailableError as exc:
                    logger.error("Model unavailable: %s", exc)
                    st.error(
                        "Disease model unavailable. Verify Disease_model/my_densenet121.pt, the threshold JSON, and the PyTorch installation."
                    )
                except InferenceError as exc:
                    logger.error("Inference failed: %s", exc)
                    st.error(str(exc))
                except SuitabilityGateError:
                    logger.warning("Pathology inference was blocked by the chest X-ray suitability gate.")
                    st.error(suitability_message(validation))

    result: AnalysisResult | None = st.session_state.get("analysis")
    if (
        result is not None
        and (
            result.image_hash != image_hash
            or st.session_state.get("analysis_hash") != image_hash
            or result.model_revision != disease_model_revision()
            or not validation.accepted
        )
    ):
        clear_result_state(st.session_state)
        result = None
    if result is None:
        st.markdown(
            """
            <div class="modern-card" style="text-align:center; padding: 36px 20px;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="1.5" style="margin-bottom:12px;">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <div style="font-size:15px; font-weight:600; color:var(--text-primary); margin-bottom:4px;">Workspace Ready</div>
                <div style="font-size:13px; color:var(--text-muted); max-width:420px; margin:0 auto;">
                    Run analysis only after this image passes chest X-ray suitability validation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Telemetry Status Bar
    st.markdown(
        f"""
        <div class="workspace-status-bar">
            <div class="status-bar-left">
                <div class="status-item">
                    <span class="status-label">Specimen</span>
                    <span class="status-val">{_escape(st.session_state.image_name)}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Model Checkpoint</span>
                    <span class="status-val font-mono">{result.checkpoint_id}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Timestamp</span>
                    <span class="status-val font-mono">{result.timestamp[:19].replace('T', ' ')} UTC</span>
                </div>
            </div>
            <div class="status-badge-valid">● Gate passed · model ran</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2-Column Cohesive Layout: Imaging Studio (Left) & Score Matrix (Right)
    col_studio, col_scores = st.columns([1.1, 0.9], gap="medium")

    with col_studio:
        st.markdown(
            """
            <div class="studio-frame-card">
                <div class="frame-title">
                    <span>Dual Imaging Studio</span>
                    <span class="inspect-badge">DenseNet feature attributions</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        threshold_findings = result.findings
        if threshold_findings:
            finding_labels = [finding.label for finding in threshold_findings]
            if st.session_state.get("gradcam_target_selector") not in finding_labels:
                st.session_state.pop("gradcam_target_selector", None)
            previous_label = st.session_state.get("gradcam_label")
            selected_label = st.selectbox(
                "Select Pathology for Grad-CAM Attribution Map:",
                options=finding_labels,
                index=finding_labels.index(previous_label) if previous_label in finding_labels else 0,
                format_func=lambda label: (
                    f"{label.replace('_', ' ')} · "
                    f"{next(finding.score for finding in threshold_findings if finding.label == label):.1%}"
                ),
                key="gradcam_target_selector",
                help="Only outputs meeting their supplied cutoff are available, ordered by score from highest to lowest.",
            )
            st.session_state.gradcam_label = selected_label
            gradcams = st.session_state.get("gradcams", {})
            gradcam = gradcams.get(selected_label)
            st.session_state.gradcam = gradcam
            gradcam_error = st.session_state.get("gradcam_errors", {}).get(selected_label)

            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(image, caption="Original radiograph", width="stretch")
            with img_c2:
                if gradcam is not None:
                    st.image(
                        gradcam,
                        caption=f"Grad-CAM attribution · {selected_label.replace('_', ' ')}",
                        width="stretch",
                    )
                else:
                    st.warning(gradcam_error or "Grad-CAM could not be generated for this finding.")

            st.markdown(
                """
            <div class="colormap-container">
                <div class="colormap-header">
                    <span>ATTRIBUTION MAGNITUDE (MAGMA SCALE)</span>
                    <span class="font-mono">GRAD-CAM ACTIVATION</span>
                </div>
                <div class="colormap-bar"></div>
                <div class="colormap-labels">
                    <span>0.0 (Baseline Focus)</span>
                    <span>0.5 (Intermediate)</span>
                    <span>1.0 (Peak Receptive Field Activation)</span>
                </div>
            </div>
            <div class="attribution-explainer">
                <b>Attribution interpretation:</b> Saliency is backpropagated through the 320 × 320 model input.
                High activation (yellow/orange) indicates features the neural network prioritized for <b>"""
            + _escape(selected_label)
            + """</b>. It denotes algorithm attention, not anatomical lesion borders or confirmed pathology.
            </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            explanation = EXPLANATIONS.get(selected_label)
            if explanation:
                st.info(
                    f"General educational imaging reference for {_escape(selected_label.replace('_', ' '))}: "
                    f"{_escape(explanation)} {DISEASE_DISCLAIMER}"
                )
        else:
            gradcam = None
            st.info(
                "No disease output crossed its supplied per-label cutoff, so there is no threshold-qualified "
                "pathology to select for Grad-CAM."
            )

    with col_scores:
        render_predictions(result)

    with report_slot:
        render_report_section(
            result=result,
            image=image,
            gradcam_images=st.session_state.get("gradcams", {}),
            image_hash=image_hash,
            validation=validation,
            placeholder=report_slot,
        )


def render_report_section(
    result: AnalysisResult,
    image: Image.Image,
    gradcam_images: dict[str, Image.Image],
    image_hash: str,
    validation: SuitabilityValidationResult,
    placeholder,
) -> None:
    try:
        pdf_bytes = generate_pdf_report(
            result=result,
            filename=st.session_state.image_name,
            original_image=image,
            gradcam_images=gradcam_images,
            image_hash=image_hash,
            validation=validation,
        )
    except Exception as exc:
        logger.exception("PDF generation failed: %s", exc)
        placeholder.error(f"Could not compile PDF report dossier: {exc}")
    else:
        placeholder.download_button(
            "Download Research Dossier (PDF)",
            data=pdf_bytes,
            file_name="lunglens-research-dossier.pdf",
            mime="application/pdf",
            type="primary",
            key="research_dossier_download",
            width="stretch",
        )


def page_about() -> None:
    st.markdown(
        """
        <div class="hero-container" style="padding-bottom:0.5rem;">
            <div class="eyebrow-pill">SPECIFICATIONS & METHODOLOGY</div>
            <h1 class="hero-title" style="font-size:38px;">Model Architecture & Methodology</h1>
            <p class="hero-copy" style="font-size:15px; margin-bottom:12px;">Technical documentation for the local X-ray gate, project DenseNet-121 model, preprocessing, thresholds, and explainability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "**Research Prototype Notice:** LungLens AI is an open exploratory research system. Outputs are not clinical diagnoses, "
        "medical guidance, or validated clinical assessments. Performance on diverse target populations has not been clinically established."
    )

    st.markdown("### Chest X-ray input validation")
    st.markdown(
        "Input handling has two separate stages. **File validation** checks PNG/JPEG format, decodability, "
        "size limits, and EXIF orientation. It does not identify modality. **Chest X-ray suitability validation** "
        "uses only the supplied local checkpoint, `xray_classifier.pth`, with outputs Normal Photo (0) and "
        "X-Ray (1). The image is converted to RGB, resized to the gate model's configured input size, converted to "
        "a tensor, and normalized with the gate's ImageNet mean/std. The configurable decision threshold defaults to 0.8; it has not "
        "been tuned or clinically validated. "
        "Scores are not calibrated. A pass is not a guarantee of authenticity, frontal view, or clinical usability."
    )
    st.caption(
        "Suitability-gate pass/reject unit tests inject mock classifier decisions to test application control flow only; "
        "they do not evaluate or establish classifier accuracy."
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(
            """
            <div class="modern-card">
                <h3 style="margin-top:0; color:var(--text-primary);">Project disease model</h3>
                <p style="color:var(--text-secondary); font-size:13px; line-height:1.6;">
                    A locally loaded torchvision DenseNet-121 fine-tuned by the project owner on NIH ChestX-ray14 data. The checkpoint is
                    <code>Disease_model/my_densenet121.pt</code>; its raw 14-output logits are converted with independent
                    sigmoid functions, not softmax.
                </p>
                <div style="background:var(--bg-surface); padding:12px; border-radius:8px; border:1px solid var(--border-light); font-size:12px; color:var(--text-muted); line-height:1.6;">
                    • <b>Input:</b> Grayscale → 320 × 320 bilinear resize → RGB<br>
                    • <b>Normalization:</b> ImageNet mean/std<br>
                    • <b>Outputs:</b> Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax<br>
                    • <b>Thresholds:</b> Per-label validation-set Youden's J values in <code>thresholds_my.json</code><br>
                    • <b>No normal class:</b> No score crossing a cutoff does not establish a normal image
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="modern-card">
                <h3 style="margin-top:0; color:var(--text-primary);">Threshold and evaluation notes</h3>
                <p style="color:var(--text-secondary); font-size:13px; line-height:1.6;">
                    Per-label thresholds were selected by Youden's J on validation data. The owner-reported test summary is mean AUROC 0.799 and mean AUPRC 0.253. Reported specificity is low for some labels (Consolidation ~0.39, Infiltration ~0.50, Effusion ~0.53); the Hernia cutoff is especially uncertain due to few positive cases. These are not clinical performance estimates.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="modern-card">
                <h3 style="margin-top:0; color:var(--text-primary);">Visual Explainability (Grad-CAM)</h3>
                <p style="color:var(--text-secondary); font-size:13px; line-height:1.6;">
                    Gradient-weighted Class Activation Mapping (Grad-CAM) computes gradients for the selected disease logit
                    with respect to the DenseNet-121 feature extractor output.
                </p>
                <div style="background:var(--bg-surface); padding:12px; border-radius:8px; border:1px solid var(--border-light); font-size:12px; color:var(--text-muted); line-height:1.6;">
                    1. <b>Forward Pass:</b> Computes activations from the final convolutional feature extractor.<br>
                    2. <b>Backward Hook:</b> Propagates gradients from the selected output head.<br>
                    3. <b>Spatial Pooling:</b> Global-average-pools gradients to yield neuron importance weights.<br>
                    4. <b>ReLU Rectification:</b> Isolates features with positive influence on the target class.<br>
                    5. <b>Interpolation:</b> Resizes the low-resolution attribution map to the input image size.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="modern-card">
                <h3 style="margin-top:0; color:var(--text-primary);">Privacy & Ethical Stewardship</h3>
                <p style="color:var(--text-secondary); font-size:13px; line-height:1.6;">
                    • <b>Zero Persistent Retention:</b> All image arrays and inference tensors are kept ephemerally in RAM.<br>
                    • <b>De-Identification Required:</b> Operators must ensure uploaded scans have all PHI (Patient Health Information) removed.<br>
                    • <b>Training:</b> Project-owner reports NIH ChestX-ray14 data and an official split.<br>
                    • <b>Use:</b> Educational/research only; not a medical device or diagnosis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Top-level Routing
current_page = render_sidebar()
if current_page == "Overview":
    page_overview()
elif current_page == "Analyze X-Ray":
    page_analysis()
else:
    page_about()
