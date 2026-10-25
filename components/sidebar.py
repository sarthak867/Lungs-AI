from __future__ import annotations

from base64 import b64encode
import streamlit as st

from config import APP_NAME, APP_SUBTITLE, CHECKPOINT_ID, MODEL_NAME, PROJECT_ROOT


def render_sidebar() -> str:
    logo_path = PROJECT_ROOT / "assets" / "logo.svg"
    logo_b64 = b64encode(logo_path.read_bytes()).decode("ascii") if logo_path.exists() else ""

    with st.sidebar:
        # Brand Mark & Wordmark
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <img src="data:image/svg+xml;base64,{logo_b64}" class="sidebar-logo" alt="{APP_NAME} Logo" />
                <div class="sidebar-brand-text">
                    <div class="brand-title-row">
                        <span class="brand-name">{APP_NAME}</span>
                        <span class="brand-tag">RESEARCH</span>
                    </div>
                    <div class="brand-subtext">{APP_SUBTITLE}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model Telemetry & Status Card
        st.markdown(
            f"""
            <div class="telemetry-card">
                <div class="telemetry-header">
                    <span class="pulse-indicator"></span>
                    <span class="telemetry-title">MODEL OPERATIONAL</span>
                </div>
                <div class="telemetry-grid">
                    <div class="telemetry-item">
                        <span class="telemetry-label">Architecture</span>
                        <span class="telemetry-val">DenseNet-121</span>
                    </div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">Heads</span>
                        <span class="telemetry-val">18 Pathologies</span>
                    </div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">Weights</span>
                        <span class="telemetry-val">res224-all</span>
                    </div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">Attribution</span>
                        <span class="telemetry-val">denseblock4</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-section-label">NAVIGATION WORKSPACE</div>', unsafe_allow_html=True)

        nav_options = ["Overview", "Analyze X-Ray", "About / Model Info"]

        # Safely consume any programmatic navigation target before the radio widget is instantiated
        if "target_nav_page" in st.session_state:
            target = st.session_state.pop("target_nav_page")
            if target in nav_options:
                st.session_state["nav_radio_selection"] = target

        if "nav_radio_selection" not in st.session_state:
            st.session_state["nav_radio_selection"] = "Overview"

        selected_page = st.radio(
            "Main Navigation",
            nav_options,
            key="nav_radio_selection",
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

        # Privacy & Session Safety
        st.markdown(
            """
            <div class="privacy-card">
                <div class="privacy-header">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                    </svg>
                    <span>SESSION PRIVACY</span>
                </div>
                <p class="privacy-copy">Images are processed ephemerally in RAM and are never permanently stored. Upload de-identified scans only.</p>
                <div class="disclaimer-badge">NON-CLINICAL EVALUATION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page
