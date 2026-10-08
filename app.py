"""
app.py
======
An AI-Based Deepfake Detection System for Video and Real-Time Webcam Analysis
Using Vision Transformers (ViT-Base/16).

Academic & Presentation-Ready Interface
- Section 1: Video & Webcam Detection Pipeline
- Section 2: Research Results Dashboard
- Section 3: Technical Details & Mathematical Formulation
- Section 4: Paper Figures with PNG & SVG Export
- Section 5: Academic Project Overview
"""

import os
import time
import json
import math
import csv
import io
import tempfile
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Core pipeline imports
from pipeline import (
    RobustFaceDetector,
    get_detector,
    compute_face_quality as pipeline_compute_quality,
    predict_deepfake as pipeline_predict,
    predict_deepfake_with_fallback,
    get_fallback_detector,
    TemporalAggregator,
    aggregate_video_predictions as pipeline_aggregate_video_predictions,
    sample_video_frame_indices,
    ENABLE_QMC
)
from custom_fallback import CustomFallbackDetector, FallbackConfig, DecisionMode
from verdict_logic import interpret_verdict, TIER_CSS, TIER_EMOJI, VERDICT_BANDS

# Research results & figure generator imports
import research_dashboard as rd
from research_results import (
    DATASET, CONFUSION, METRICS, LATENCY, PER_VIDEO,
    MODEL_INFO, LIMITATIONS,
    get_dataset_summary, get_confusion_matrix, get_metrics,
    get_latency, get_per_video_results, get_verdict_bands,
    get_model_info, get_limitations
)

# Configure Streamlit page layout
st.set_page_config(
    page_title="AI Deepfake Detection System | ViT Analysis",
    page_icon="🔍",
    layout="wide"
)

# Backward-compatibility alias for test_detection_logic
def aggregate_video_predictions(predictions: List[Dict[str, Any]]) -> Tuple[str, float, int, int]:
    """Backward-compatible wrapper around pipeline aggregation logic."""
    return pipeline_aggregate_video_predictions(predictions)


# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --accent:          var(--primary-color, #6366f1);
        --accent-muted:    color-mix(in srgb, var(--accent) 14%, transparent);
        --accent-hover:    color-mix(in srgb, var(--accent) 22%, transparent);
        --accent-strong:   color-mix(in srgb, var(--accent) 85%, #000);

        --bg:              var(--background-color, #ffffff);
        --bg-card:         var(--secondary-background-color, #f8fafc);
        --text:            var(--text-color, #0f172a);
        --text-secondary:  color-mix(in srgb, var(--text) 68%, transparent);
        --text-muted:      color-mix(in srgb, var(--text) 45%, transparent);
        --border:          color-mix(in srgb, var(--text) 12%, transparent);
        --border-hover:    color-mix(in srgb, var(--accent) 35%, transparent);

        --shadow-sm:       0 1px 3px color-mix(in srgb, var(--text) 5%, transparent);
        --shadow:          0 4px 12px color-mix(in srgb, var(--text) 7%, transparent);
        --shadow-lg:       0 10px 24px color-mix(in srgb, var(--accent) 12%, transparent);

        --radius-sm: 8px;
        --radius:    12px;
        --radius-lg: 16px;
        --ease:      cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body, [class*="stApp"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .stApp {
        background: var(--bg);
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* Centered responsive container for laptops & desktops */
    .block-container {
        max-width: 1060px;
        margin: 0 auto;
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    /* Eyebrow badge */
    .eyebrow {
        display: block;
        text-align: center;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        background: var(--accent-muted);
        border: 1px solid var(--border-hover);
        border-radius: 9999px;
        padding: 0.35rem 0.95rem;
        width: fit-content;
        margin: 0 auto 0.75rem auto;
    }

    .main-title {
        text-align: center;
        font-size: clamp(1.8rem, 4vw, 2.4rem);
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
        color: var(--text);
    }

    .subtitle {
        text-align: center;
        font-size: 0.98rem;
        color: var(--text-secondary) !important;
        margin-bottom: 1.5rem;
        font-weight: 400;
        max-width: 780px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.5;
    }

    /* Card styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-card);
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border) !important;
        transition: all 0.2s var(--ease);
        margin: 0.6rem 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--border-hover) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.35rem 0.35rem;
    }

    /* Mode toggle pill */
    div.st-key-mode_toggle {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 9999px;
        padding: 4px;
    }
    div.st-key-mode_toggle .stButton>button {
        border-radius: 9999px !important;
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border: none;
        padding: 0.7rem 1.4rem;
        font-size: 0.92rem;
        font-weight: 600;
        border-radius: var(--radius);
        transition: all 0.2s var(--ease);
        cursor: pointer;
    }
    .stButton>button[kind="primary"] {
        background: var(--accent);
        color: #fff;
        box-shadow: var(--shadow-sm);
    }
    .stButton>button[kind="primary"]:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }
    .stButton>button[kind="secondary"] {
        background: var(--bg-card);
        color: var(--text);
        border: 1px solid var(--border);
    }
    .stButton>button[kind="secondary"]:hover {
        background: var(--accent-muted);
        border-color: var(--accent);
        color: var(--accent-strong);
    }

    /* Verdict Banners */
    .result-safe, .result-uncertain, .result-danger, .result-warning {
        color: #fff !important;
        padding: 1.25rem 1.5rem;
        border-radius: var(--radius);
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        margin: 0.8rem 0 0.5rem 0;
        letter-spacing: -0.01em;
    }
    .result-safe {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    }
    .result-uncertain {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.25);
    }
    .result-danger {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.25);
    }
    .result-warning {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.25);
    }
    .result-safe span, .result-uncertain span,
    .result-danger span, .result-warning span { color: #fff !important; }

    /* Verdict Explanation Box */
    .verdict-explanation-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius-sm);
        padding: 0.85rem 1.15rem;
        margin-bottom: 1.1rem;
        font-size: 0.92rem;
        line-height: 1.55;
        color: var(--text-secondary);
    }

    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        padding: 0.85rem 0.4rem;
        border-radius: var(--radius);
        text-align: center;
        border: 1px solid var(--border);
        transition: all 0.2s var(--ease);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100px;
    }
    .metric-card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-sm);
        transform: translateY(-2px);
    }
    .metric-icon { font-size: 1.35rem; margin-bottom: 0.2rem; line-height: 1; }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        line-height: 1.2;
        color: var(--text);
    }
    .metric-label {
        font-size: 0.70rem;
        color: var(--text-muted) !important;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Summary table card */
    .summary-table-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.6rem 1rem;
        margin: 0.8rem 0;
    }
    .summary-table-card table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }
    .summary-table-card tr {
        border-bottom: 1px solid var(--border);
    }
    .summary-table-card tr:last-child {
        border-bottom: none;
    }
    .summary-table-card td {
        padding: 0.65rem 0.5rem;
    }
    .summary-table-card td.label-col {
        font-weight: 600;
        color: var(--text-secondary);
        width: 38%;
    }
    .summary-table-card td.val-col {
        color: var(--text);
        font-weight: 500;
    }

    /* Limitation card */
    .limitation-card {
        background: color-mix(in srgb, #f59e0b 8%, var(--bg-card));
        border: 1px solid color-mix(in srgb, #f59e0b 35%, var(--border));
        border-left: 5px solid #f59e0b;
        border-radius: var(--radius);
        padding: 1.15rem 1.35rem;
        margin: 1rem 0;
    }
    .limitation-card h4 {
        color: #b45309 !important;
        margin-top: 0;
        margin-bottom: 0.6rem;
        font-size: 1.05rem;
    }
    .limitation-card ol {
        margin: 0;
        padding-left: 1.3rem;
        color: var(--text-secondary);
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Academic notice box */
    .academic-notice {
        background: color-mix(in srgb, var(--accent) 8%, var(--bg-card));
        border: 1px solid color-mix(in srgb, var(--accent) 25%, var(--border));
        border-left: 4px solid var(--accent);
        border-radius: var(--radius-sm);
        padding: 0.85rem 1.15rem;
        margin: 0.85rem 0;
        font-size: 0.90rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* Live labels (webcam) */
    .live-label {
        padding: 0.85rem 1.5rem;
        border-radius: var(--radius);
        font-size: 1rem;
        font-weight: 700;
        margin: 0.75rem 0;
        text-align: center;
        display: block;
        letter-spacing: 0.01em;
    }
    .live-label.real {
        background: color-mix(in srgb, #10b981 12%, var(--bg));
        color: #10b981 !important;
        border: 1px solid color-mix(in srgb, #10b981 30%, transparent);
    }
    .live-label.fake {
        background: color-mix(in srgb, #ef4444 12%, var(--bg));
        color: #ef4444 !important;
        border: 1px solid color-mix(in srgb, #ef4444 30%, transparent);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: transparent;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        font-size: 0.92rem;
        color: var(--text-secondary) !important;
        transition: all 0.2s var(--ease);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: #fff !important;
        border-color: var(--accent) !important;
    }
    .stTabs [aria-selected="true"] * { color: #fff !important; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.75rem 1rem 0.5rem 1rem;
        font-size: 0.85rem;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
    }
    .footer p { color: var(--text-muted) !important; margin: 0.15rem 0; }
    .footer strong { color: var(--text-secondary) !important; }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        .metric-value { font-size: 1.2rem; }
        .metric-label { font-size: 0.65rem; }
    }
</style>
""", unsafe_allow_html=True)


# ==================== MODEL & DETECTOR LOADING ====================
@st.cache_resource(show_spinner=False)
def load_deepfake_model() -> Tuple[Optional[AutoImageProcessor], Optional[AutoModelForImageClassification]]:
    """Load the pre-trained deepfake detection model."""
    try:
        model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_resource(show_spinner=False)
def load_face_detector() -> RobustFaceDetector:
    """Load the robust face detector cascade (MTCNN / MediaPipe Tasks / Haar)."""
    return get_detector()


def load_fallback_detector() -> CustomFallbackDetector:
    """Load the fallback detector instance (kept for webcam reset state)."""
    return get_fallback_detector()


# ==================== CORE HELPER FUNCTIONS ====================
def compute_face_quality(face_img: np.ndarray) -> float:
    """Estimate whether a detected face crop is usable for classification."""
    info = pipeline_compute_quality(face_img)
    return info['score']


def get_face_from_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Extract and rank faces from a frame using the robust detector."""
    detector = load_face_detector()
    return detector.detect_faces(frame, max_faces=4)


def predict_deepfake(
    face_image: np.ndarray,
    processor: AutoImageProcessor,
    model: AutoModelForImageClassification
) -> Tuple[Optional[str], float]:
    """Predict if face is real or deepfake using pre-trained ViT."""
    label, conf, _, _ = pipeline_predict(face_image, processor, model)
    return label, conf


def calculate_final_verdict(predictions: List[Dict[str, Any]]) -> Tuple[str, float, int, int]:
    """Backward-compatible wrapper around video aggregation logic."""
    return pipeline_aggregate_video_predictions(predictions)


def render_metric(icon: str, value: str, label: str):
    """Render a polished metric card with icon, value, and label."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_figure_with_downloads(
    fig: plt.Figure,
    filename_base: str,
    title: str,
    caption: str = "",
    key_prefix: str = ""
):
    """Display a matplotlib figure with side-by-side PNG and SVG download buttons."""
    st.markdown(f"#### {title}")
    st.pyplot(fig)

    png_bytes = rd.get_fig_bytes(fig, fmt="png", dpi=300)
    svg_bytes = rd.get_fig_bytes(fig, fmt="svg")

    prefix = f"{key_prefix}_" if key_prefix else ""
    dl_key_base = f"{prefix}{filename_base}"

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label=f"⬇️ Download PNG ({filename_base}.png)",
            data=png_bytes,
            file_name=f"{filename_base}.png",
            mime="image/png",
            key=f"dl_png_{dl_key_base}",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            label=f"⬇️ Download SVG ({filename_base}.svg)",
            data=svg_bytes,
            file_name=f"{filename_base}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{dl_key_base}",
            use_container_width=True
        )
    if caption:
        st.caption(f"ℹ️ {caption}")
    plt.close(fig)


def draw_live_hud_overlay(
    frame: np.ndarray,
    time_remaining: float,
    predictions_count: int,
    real_count: int,
    fake_count: int,
    current_verdict: Optional[str] = None,
    current_confidence: float = 0.0,
    mode: str = "LIVE"
) -> np.ndarray:
    """
    Renders the semi-transparent telemetry HUD overlay matching the research standard
    (Screenshot 2025-10-21 173007.png). Displays title, countdown timer, cumulative
    predictions count, real/fake distribution, and current rolling verdict with confidence.
    """
    height, width = frame.shape[:2]

    # Semi-transparent dark overlay for HUD telemetry
    overlay = frame.copy()
    box_w = min(width - 20, 520)
    box_h = 165
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # 1. Header Title: DEEPFAKE ANALYSIS - LIVE MODE (White)
    title = f"DEEPFAKE ANALYSIS - {mode.upper()} MODE"
    cv2.putText(frame, title, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Time remaining: Time Remaining: 104s (Cyan in BGR: 255, 255, 0)
    time_text = f"Time Remaining: {int(max(0, time_remaining))}s"
    cv2.putText(frame, time_text, (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

    # 3. Predictions collected: Predictions Collected: 8 (White)
    cv2.putText(frame, f"Predictions Collected: {predictions_count}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Stats: Real: 8 | Fake: 0 (White)
    cv2.putText(frame, f"Real: {real_count} | Fake: {fake_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Leading verdict: Leading: REAL (87.7%) or Leading: FAKE (63.8%)
    if current_verdict:
        verdict_color = (0, 255, 0) if current_verdict == "Realism" else (0, 0, 255)
        verdict_text = "REAL" if current_verdict == "Realism" else "FAKE"
        cv2.putText(frame, f"Leading: {verdict_text} ({current_confidence*100:.1f}%)", (20, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, verdict_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Leading: Initializing...", (20, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def render_webcam_session_results(
    predictions: List[Dict[str, Any]],
    agg_result: Dict[str, Any],
    confidence_threshold: float
):
    """Renders comprehensive analysis results at the conclusion of a live webcam session."""
    st.markdown("### 📋 Live Webcam Session Summary")
    render_video_analysis_results(predictions, agg_result, confidence_threshold)


# ==================== PART 1: IMPROVED ANALYSIS RESULTS ====================
def render_video_analysis_results(
    predictions: List[Dict[str, Any]],
    agg_result: Dict[str, Any],
    confidence_threshold: float
):
    """
    Renders the Analysis Results page following the exact academic layout:
    1. Verdict Banner
    2. Short Explanation Box
    3. Key Metrics (5 cards)
    4. Probability Analysis (Large line chart)
    5. Frame Detection Breakdown (Bar chart)
    6. Analysis Summary (Structured table card)
    7. Export Results (JSON & CSV download buttons)
    """
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # 1. VERDICT BANNER
    p_fake_video = agg_result.get('video_fake_probability', 0.0)
    p_real_video = agg_result.get('video_real_probability', 0.0)
    verd = interpret_verdict(p_fake_video, p_real_video)

    css_class = TIER_CSS.get(verd['verdict_tier'], 'result-uncertain')
    emoji = TIER_EMOJI.get(verd['verdict_tier'], '⚠️')

    st.markdown(
        f'<div class="{css_class}">{emoji} {verd["verdict_label"].upper()}<br>'
        f'<span style="font-size:1.05rem; font-weight:500;">'
        f'Manipulation Probability: <b>{verd["manipulation_pct"]}%</b> '
        f'&nbsp;|&nbsp; '
        f'Real Probability: <b>{verd["real_pct"]}%</b>'
        f'</span></div>',
        unsafe_allow_html=True
    )

    # 2. SHORT EXPLANATION
    if verd['verdict_tier'] == "safe":
        explanation_msg = (
            "The model assigns a low manipulation probability to this video. "
            "This result should be interpreted together with face quality and the number of analyzed frames."
        )
    else:
        explanation_msg = verd['interpretation']

    st.markdown(
        f'<div class="verdict-explanation-box">'
        f'<b>Verdict Interpretation:</b> {explanation_msg}'
        f'</div>',
        unsafe_allow_html=True
    )

    # 3. KEY METRICS (5 aligned cards)
    st.markdown("### 🎯 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric("🧮", str(agg_result['total_predictions']), "Analyzed Frames")
    with col2:
        render_metric("✅", str(agg_result['real_count']), "Authentic Frames")
    with col3:
        render_metric("⚠️", str(agg_result['fake_count']), "Fake Frames")
    with col4:
        render_metric("🎯", f"{verd['manipulation_pct']}%", "Manipulation Prob")
    with col5:
        avg_q = agg_result.get('average_quality', 0.0)
        render_metric("✨", f"{avg_q:.1f}", "Avg Face Quality")

    st.markdown("")

    # 4. PROBABILITY ANALYSIS (Large line chart)
    st.markdown("### 📈 Probability Analysis")
    st.caption("Frame-level manipulation probability over time ($P_{\\text{fake}, t}$) with decision threshold reference.")

    fig_prob, ax_prob = plt.subplots(figsize=(10, 3.8))
    times = [p['timestamp'] for p in predictions]
    probs = [p['p_fake'] * 100 for p in predictions]
    d_thresh = float(agg_result.get('decision_threshold', 0.40))
    d_thresh_pct = d_thresh * 100.0

    ax_prob.plot(times, probs, color='#6366f1', marker='o', markersize=4, linewidth=2, label="Manipulation P(fake)")
    ax_prob.axhline(d_thresh_pct, color='#dc2626', linestyle='--', linewidth=1.2, label=f"Decision Threshold ({d_thresh_pct:.0f}%)")
    ax_prob.fill_between(times, probs, d_thresh_pct, where=np.array(probs) >= d_thresh_pct, color='#dc2626', alpha=0.12, interpolate=True)
    ax_prob.fill_between(times, probs, d_thresh_pct, where=np.array(probs) < d_thresh_pct, color='#10b981', alpha=0.10, interpolate=True)
    ax_prob.set_xlabel("Video Timestamp (seconds)", fontsize=9.5)
    ax_prob.set_ylabel("Manipulation Probability (%)", fontsize=9.5)
    ax_prob.set_ylim(0, 100)
    ax_prob.grid(True, linestyle=":", alpha=0.4)
    ax_prob.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig_prob)
    plt.close(fig_prob)

    # 5. FRAME DETECTION BREAKDOWN (Bar chart)
    st.markdown("### 📊 Frame Detection Breakdown")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 3.2))
    cats = ["Authentic (Real)", "Deepfake (Manipulated)"]
    counts = [agg_result['real_count'], agg_result['fake_count']]
    colors = ["#10b981", "#ef4444"]
    bars = ax_bar.bar(cats, counts, color=colors, width=0.42, edgecolor="white", linewidth=1.2)
    ax_bar.set_ylabel("Frame Count", fontsize=9.5)
    ax_bar.set_ylim(0, max(counts) * 1.35 if max(counts) > 0 else 5)
    total_f = agg_result['total_predictions']
    for b, c in zip(bars, counts):
        pct_c = (c / total_f * 100) if total_f > 0 else 0
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                    f"{c} frames ({pct_c:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_bar.grid(axis='y', linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # 6. ANALYSIS SUMMARY
    st.markdown("### 📋 Analysis Summary")
    burst_status = "Detected (Localized glitch sequence observed)" if agg_result.get('anomaly_detected') else "None detected"
    top_k_pct = agg_result.get('top_k_fake_score', 0.0) * 100.0
    st.markdown(f"""
    <div class="summary-table-card">
        <table>
            <tr><td class="label-col">Model Used</td><td class="val-col"><code>prithivMLmods/Deep-Fake-Detector-v2-Model</code> (ViT-Base/16)</td></tr>
            <tr><td class="label-col">Frames Analyzed</td><td class="val-col"><b>{agg_result['total_predictions']}</b> sampled frames</td></tr>
            <tr><td class="label-col">Average Face Quality</td><td class="val-col"><b>{avg_q:.2f}</b> / 100</td></tr>
            <tr><td class="label-col">Manipulation Probability P(fake)</td><td class="val-col"><b>{verd['manipulation_pct']}%</b></td></tr>
            <tr><td class="label-col">Top-20% Suspicious Frames Score</td><td class="val-col"><b>{top_k_pct:.1f}%</b></td></tr>
            <tr><td class="label-col">Authentic Probability P(real)</td><td class="val-col"><b>{verd['real_pct']}%</b></td></tr>
            <tr><td class="label-col">Video-Level Verdict</td><td class="val-col"><b>{verd['verdict_label']}</b></td></tr>
            <tr><td class="label-col">Decision Threshold (τ_d)</td><td class="val-col">{d_thresh:.2f}</td></tr>
            <tr><td class="label-col">Frame Certainty Cutoff (τ_f)</td><td class="val-col">{confidence_threshold:.2f}</td></tr>
            <tr><td class="label-col">Temporal Burst Glitch Check</td><td class="val-col">{burst_status}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 7. EXPORT RESULTS
    st.markdown("### 💾 Export Results")
    export_json = json.dumps({
        'verdict_label': verd['verdict_label'],
        'verdict_tier': verd['verdict_tier'],
        'manipulation_probability': verd['p_fake'],
        'top_k_fake_score': agg_result.get('top_k_fake_score', 0.0),
        'real_probability': verd['p_real'],
        'manipulation_pct': verd['manipulation_pct'],
        'real_pct': verd['real_pct'],
        'interpretation': verd['interpretation'],
        'is_inconclusive': verd['is_inconclusive'],
        'real_count': agg_result['real_count'],
        'fake_count': agg_result['fake_count'],
        'total_predictions': agg_result['total_predictions'],
        'average_quality': agg_result.get('average_quality', 0.0),
        'decision_threshold': d_thresh,
        'confidence_threshold': confidence_threshold,
        'model': 'prithivMLmods/Deep-Fake-Detector-v2-Model',
        'note': 'manipulation_probability is the model P(fake) score, NOT system accuracy.',
        'frame_predictions': predictions
    }, indent=2)

    output_csv = io.StringIO()
    writer = csv.writer(output_csv)
    writer.writerow([
        'Frame Index', 'Timestamp (s)', 'Frame Label',
        'Frame Fake Prob (%)', 'Frame Real Prob (%)',
        'Quality', 'ViT Certainty'
    ])
    for p in predictions:
        writer.writerow([
            p['frame_index'],
            p['timestamp'],
            p['label'],
            f"{p['p_fake']*100:.2f}%",
            f"{p['p_real']*100:.2f}%",
            f"{p['quality']:.1f}",
            f"{p.get('vit_confidence', 0.0)*100:.1f}%"
        ])
    writer.writerow([])
    writer.writerow(['Verdict Label', verd['verdict_label']])
    writer.writerow(['Manipulation Probability (P_fake)', f"{verd['manipulation_pct']}%"])
    writer.writerow(['Real Probability (P_real)', f"{verd['real_pct']}%"])
    writer.writerow(['Model', 'prithivMLmods/Deep-Fake-Detector-v2-Model'])
    writer.writerow(['NOTE', 'Manipulation Probability is P(fake) from the ViT model, NOT system accuracy'])
    export_csv = output_csv.getvalue()

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="⬇️ Download Results (JSON)",
            data=export_json,
            file_name="video_analysis_results.json",
            mime="application/json",
            use_container_width=True
        )
    with col_exp2:
        st.download_button(
            label="⬇️ Download Results (CSV)",
            data=export_csv,
            file_name="video_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_webcam_session_results(
    predictions: List[Dict[str, Any]],
    agg_result: Dict[str, Any],
    confidence_threshold: float
):
    """Render webcam session results in clean academic styling."""
    st.markdown("---")
    st.markdown("### 📊 Webcam Session Summary")

    p_fake_sess = agg_result.get('video_fake_probability', 0.0)
    p_real_sess = agg_result.get('video_real_probability', 0.0)
    verd_sess = interpret_verdict(p_fake_sess, p_real_sess)
    css_sess = TIER_CSS.get(verd_sess['verdict_tier'], 'result-uncertain')
    emoji_sess = TIER_EMOJI.get(verd_sess['verdict_tier'], '⚠️')

    st.markdown(
        f'<div class="{css_sess}">{emoji_sess} {verd_sess["verdict_label"].upper()}<br>'
        f'<span style="font-size:1.05rem; font-weight:500;">'
        f'Manipulation Probability: <b>{verd_sess["manipulation_pct"]}%</b> '
        f'&nbsp;|&nbsp; '
        f'Real Probability: <b>{verd_sess["real_pct"]}%</b>'
        f'</span></div>',
        unsafe_allow_html=True
    )
    st.caption(f"ℹ️ {verd_sess['interpretation']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("🧮", str(len(predictions)), "Total Frames")
    with col2:
        render_metric("✅", str(agg_result['real_count']), "Authentic Frames")
    with col3:
        render_metric("⚠️", str(agg_result['fake_count']), "Fake Frames")
    with col4:
        render_metric("✨", f"{agg_result.get('average_quality', 0.0):.1f}", "Avg Quality")


# ==================== PART 2 & PART 3: RESEARCH RESULTS DASHBOARD ====================
def render_research_dashboard():
    """Renders the comprehensive research results dashboard (Part 3)."""
    st.markdown("## 📊 Research Results Dashboard")

    st.markdown("""
    <div class="academic-notice">
        <b>Preliminary Evaluation — Small Dataset:</b><br>
        This dashboard presents the verified experimental evaluation of our system.
        Total evaluation set consists of <b>6 videos</b> (2 validation, 4 test).
        <b>The results are not representative of general detection performance.</b>
    </div>
    """, unsafe_allow_html=True)

    # ── A. DATASET / EVALUATION SUMMARY ──
    st.markdown("### A. Dataset & Evaluation Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric("📁", str(DATASET["total"]), "Total Videos")
    with c2:
        render_metric("👤", str(DATASET["real"]), "Real Videos")
    with c3:
        render_metric("🎭", str(DATASET["fake"]), "Fake Videos")
    with c4:
        render_metric("🧪", str(DATASET["val"]), "Validation Videos")
    with c5:
        render_metric("🎯", str(DATASET["test"]), "Test Videos")

    render_figure_with_downloads(
        rd.fig_class_distribution(),
        "01_class_distribution",
        "Dataset Class Distribution",
        "Class balance of the preliminary evaluation subset (50% Real / 50% Fake).",
        key_prefix="dash"
    )

    st.markdown("---")

    # ── B. CONFUSION MATRIX ──
    st.markdown("### B. Confusion Matrix (Test Split)")
    st.markdown("""
    The test subset consists of 4 videos evaluated with positive class = Fake (1) and negative class = Real (0):
    - **TP = 0** (Predicted Fake, Actual Fake)
    - **TN = 1** (Predicted Real, Actual Real)
    - **FP = 1** (Predicted Fake, Actual Real — False Alarm)
    - **FN = 2** (Predicted Real, Actual Fake — Missed Detection)
    """)

    render_figure_with_downloads(
        rd.fig_confusion_matrix(),
        "02_confusion_matrix",
        "Test Split Confusion Matrix",
        "Confusion matrix on the 4-video test split using decision threshold τ = 0.50.",
        key_prefix="dash"
    )

    st.markdown("---")

    # ── C. PERFORMANCE METRICS ──
    st.markdown("### C. Performance Metrics Comparison")
    st.markdown("""
    <div class="academic-notice">
        <b>Observation:</b> Both configurations produced identical results on the four-video test subset.
        Temporal aggregation adds video-level decision logic with negligible additional computational overhead,
        but no performance improvement can be established on the current four-video test set.
    </div>
    """, unsafe_allow_html=True)

    render_figure_with_downloads(
        rd.fig_metrics_comparison(),
        "03_metrics_comparison",
        "Performance Metrics: ViT Only vs. ViT + Temporal",
        "Accuracy, Precision, Recall, and F1 score comparison on the test split.",
        key_prefix="dash"
    )

    st.markdown("---")

    # ── D. LATENCY RESULTS ──
    st.markdown("### D. Processing Latency & Throughput (CPU)")
    st.markdown("""
    Evaluation conducted on CPU (mean across test videos).
    <b>Key finding:</b> Temporal aggregation adds negligible computational overhead
    because it operates on already-computed frame probabilities.
    Effective throughput is approximately <b>6.5 FPS</b> (near-real-time CPU processing).
    """, unsafe_allow_html=True)

    render_figure_with_downloads(
        rd.fig_latency_comparison(),
        "04_latency_comparison",
        "Latency Breakdown Comparison",
        "ViT inference latency (ms), End-to-end latency (ms), and Effective FPS on CPU.",
        key_prefix="dash"
    )

    render_figure_with_downloads(
        rd.fig_fps_comparison(),
        "05_fps_comparison",
        "Effective Processing FPS Comparison",
        "Standalone throughput comparison confirming identical near-real-time CPU throughput.",
        key_prefix="dash"
    )

    st.markdown("---")

    # ── E. PER-VIDEO ERROR ANALYSIS ──
    st.markdown("### E. Per-Video Error Analysis & Domain Gap")
    st.markdown("""
    Detailed breakdown of each test video demonstrating the out-of-distribution domain mismatch:
    - <code>real_interview_02</code>: Ground Truth = Real, P(fake) = 47.11% → <b>Correct (TN)</b>
    - <code>real_interview_03</code>: Ground Truth = Real, P(fake) = 64.95% → <b>False Positive (FP)</b>
    - <code>fake_deeperforensics_02</code>: Ground Truth = Fake, P(fake) = 38.88% → <b>False Negative (FN)</b>
    - <code>fake_deeperforensics_03</code>: Ground Truth = Fake, P(fake) = 42.12% → <b>False Negative (FN)</b>
    """)

    render_figure_with_downloads(
        rd.fig_per_video_table(),
        "08_per_video_error_table",
        "Per-Video Error Analysis Summary",
        "Tabular comparison of ground truth labels versus model predictions.",
        key_prefix="dash"
    )

    render_figure_with_downloads(
        rd.fig_pfake_comparison(),
        "06_video_fake_probability",
        "Video-Level Manipulation Probability P(fake)",
        "Demonstrates domain gap: both DeeperForensics fake samples remain below 50% while one real sample crosses it.",
        key_prefix="dash"
    )

    st.markdown("---")

    # ── F. FRAME QUALITY ANALYSIS ──
    st.markdown("### F. Face Quality Score Analysis")
    st.markdown("""
    <div class="academic-notice">
        <b>Quality Assessment Note:</b> The fake videos in this evaluation set had lower average face-quality scores
        (54.70 and 55.14) than the real interview videos (84.36 and 82.42).
        Lower quality does not itself imply manipulation, but the systematic difference may affect model behavior.
    </div>
    """, unsafe_allow_html=True)

    render_figure_with_downloads(
        rd.fig_face_quality(),
        "07_face_quality_comparison",
        "Average Face Quality by Video",
        "Comparison of average multi-cue face quality scores across test videos.",
        key_prefix="dash"
    )

    st.markdown("---")

    # ── H. CONFIGURATION COMPARISON TABLE ──
    st.markdown("### H. Configuration Benchmark Summary")
    bench_data = [
        {"Configuration": "ViT Only (Config A)", "Accuracy": "25.0%", "Precision": "0.0%", "Recall": "0.0%", "F1 Score": "0.0%", "ViT Latency": "88.97 ms", "End-to-End": "153.21 ms", "Throughput": "6.53 FPS"},
        {"Configuration": "ViT + Temporal (Config B)", "Accuracy": "25.0%", "Precision": "0.0%", "Recall": "0.0%", "F1 Score": "0.0%", "ViT Latency": "89.93 ms", "End-to-End": "152.25 ms", "Throughput": "6.57 FPS"},
    ]
    st.table(bench_data)

    st.markdown("---")

    # ── I. LIMITATIONS CARD ──
    st.markdown("### I. System Limitations")
    st.markdown("""
    <div class="limitation-card">
        <h4>⚠️ Documented System Limitations</h4>
        <ol>
            <li>Only six videos were available for system evaluation.</li>
            <li>Only four videos were used for final testing.</li>
            <li>The external ViT was not trained or fine-tuned by us.</li>
            <li>The pre-training dataset of the external model is not documented in our repository.</li>
            <li>All fake evaluation videos came from the DeeperForensics source.</li>
            <li>Threshold tuning cannot be reliably performed with only two validation videos.</li>
            <li>Webcam evaluation has not been performed against a labeled webcam benchmark.</li>
            <li>Only one face per frame is currently analyzed.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ==================== PART 4: TECHNICAL DETAILS ====================
def render_technical_details():
    """Renders the comprehensive Technical Details page with 9 expandable sections."""
    st.markdown("## 📐 Technical Architecture & Implementation")
    st.caption("Complete technical specifications, algorithms, mathematical formulations, and stack verification.")

    # Section 1: Pre-trained ViT Architecture
    with st.expander("1. Pre-trained ViT Architecture", expanded=True):
        st.markdown("""
        #### Pre-trained ViT Model Specification
        - **Model Identifier:** `prithivMLmods/Deep-Fake-Detector-v2-Model`
        - **Architecture:** Vision Transformer (`ViT-Base/16`)
        - **Input Resolution:** $224 \\times 224$ pixels (RGB)
        - **Patch Size:** $16 \\times 16$ pixels (196 total patches + 1 [CLS] token)
        - **Hidden Size ($D$):** 768
        - **Transformer Layers:** 12
        - **Attention Heads:** 12
        - **MLP Dimension:** 3072
        - **Output Classes:** 2 (`Class 0: Realism`, `Class 1: Deepfake`)
        - **Normalization:** $\\mu = (0.5, 0.5, 0.5)$, $\\sigma = (0.5, 0.5, 0.5)$
        - **Execution Device:** CPU during current experiments

        > **Academic Note:** This project uses the publicly available pre-trained model for inference.
        > We did **not** train or fine-tune this model.
        """)

    # Section 2: How the ViT Works
    with st.expander("2. How the Vision Transformer Operates", expanded=False):
        st.markdown("""
        #### Step-by-Step ViT Processing Flow
        ```
        Face Crop
        → Resize to 224 × 224
        → Divide into 16 × 16 Non-Overlapping Patches
        → Linear Patch Projection & Embedding
        → Prepend Learnable [CLS] Token & Add Positional Embeddings
        → 12 Transformer Encoder Layers (Multi-Head Self-Attention + MLP)
        → Layer Normalization
        → Classification Head (Linear Classifier on [CLS])
        → Raw Logits (Realism vs. Deepfake)
        → Softmax Normalization → P(real), P(fake)
        ```
        """)

    # Section 3: Our Algorithm
    with st.expander("3. Our End-to-End Detection Algorithm", expanded=False):
        st.markdown("""
        #### System-Level Detection Pipeline
        Distinguishing **system-level pipeline engineering** from the **external pre-trained backbone**:

        1. **Video/Webcam Input:** Ingest continuous stream or uploaded video file.
        2. **Frame Sampling:** Temporal sampling at uniform intervals (target 3.0 FPS).
        3. **Cascade Face Detection:** Primary MTCNN with MediaPipe Tasks and OpenCV Haar fallback.
        4. **Face Quality Assessment:** Multi-cue scoring across sharpness, exposure, contrast, and size.
        5. **Landmark Alignment & Contextual Crop:** Canonical face alignment and aspect-ratio preserved scaling.
        6. **ViT Classification:** Pre-trained ViT-B/16 forward inference.
        7. **Softmax Output:** Extraction of soft frame probabilities $P(\\text{real}, t)$ and $P(\\text{fake}, t)$.
        8. **Quality & Certainty Weighting:** Dynamic weighting $w_t$ based on sample quality and prediction certainty.
        9. **Temporal Aggregation:** Video-level weighted probability integration.
        10. **Burst Anomaly Detection:** Temporal streak detection for localized deepfake anomalies.
        11. **Final Decision & Interpretation:** Thresholding against $\\tau_d = 0.50$ and assignment of verdict band.
        """)

    # Section 4: Important Formulas
    with st.expander("4. Mathematical Formulations", expanded=False):
        st.markdown("#### Formula 1: Softmax Normalization")
        st.latex(r"p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}")
        st.caption("Converts raw ViT logits $z_i$ into normalized class probabilities for Realism and Deepfake.")

        st.markdown("#### Formula 2: Prediction Certainty")
        st.latex(r"\kappa_t = 2 \cdot |P_{\text{fake}, t} - 0.5|")
        st.caption("Measures model confidence on frame $t$. Approaches 1.0 when prediction is decisive and 0.0 near the decision boundary.")

        st.markdown("#### Formula 3: Frame Weight")
        st.latex(r"w_t = \max\left(0.1, \; \left(\frac{q_t}{100}\right) \cdot (0.5 + 0.5\kappa_t)\right)")
        st.caption(r"Weights frame $t$ jointly by its measured face quality score $q_t$ and its prediction certainty $\kappa_t$.")

        st.markdown("#### Formula 4: Weighted Video-Level Fake Probability")
        st.latex(r"P_{\text{fake}} = \frac{\sum_{t=1}^{N} w_t P_{\text{fake}, t}}{\sum_{t=1}^{N} w_t}")

        st.markdown("#### Formula 5: Weighted Video-Level Real Probability")
        st.latex(r"P_{\text{real}} = \frac{\sum_{t=1}^{N} w_t P_{\text{real}, t}}{\sum_{t=1}^{N} w_t}")

        st.markdown("#### Formula 6: Burst Anomaly Detection")
        st.latex(r"B = (\text{max\_consecutive\_fake} \ge 3) \;\land\; \left(\frac{\text{max\_consecutive\_fake}}{N} \ge 0.15\right)")
        st.caption("Detects localized face-swap glitches across consecutive sampled frames.")

        st.markdown("#### Formula 7: Final Decision Rule")
        st.latex(r"\text{Verdict} = \begin{cases} \text{Deepfake}, & \text{if } P_{\text{fake}} \ge \tau_d \;\lor\; (B \;\land\; n_{\text{fake}} > 2) \\ \text{Realism}, & \text{otherwise} \end{cases}")
        st.caption("Default video decision threshold $\\tau_d = 0.50$; frame certainty threshold $\\tau_f = 0.50$.")

    # Section 5: Face Quality Score
    with st.expander("5. Face Quality Score Formulation", expanded=False):
        st.markdown("#### Multi-Cue Face Quality Formula")
        st.latex(r"q = 100 \cdot (0.4 \bar{b} + 0.2 \bar{e} + 0.2 \bar{c} + 0.2 \bar{s})")
        st.markdown("""
        - $\\bar{b}$: Sharpness contribution computed from Laplacian variance.
        - $\\bar{e}$: Exposure contribution centered around mid-tone illumination (128).
        - $\\bar{c}$: Contrast contribution based on standard deviation of pixel intensities.
        - $\\bar{s}$: Face dimension contribution relative to $224 \\times 224$ ideal crop.

        **Rejection & Skip Thresholds:**
        - Face dimension $< 40$ px
        - Laplacian variance $< 20$
        - Brightness $< 12$ or $> 248$
        - Contrast standard deviation $< 6$

        *Note: In the active implementation, a permissive final skip rule ensures usability across varying webcam environments ($q < 15.0$ and invalid quality tags).*
        """)

    # Section 6: Face Detection Cascade
    with st.expander("6. Face Detection Cascade", expanded=False):
        st.markdown("""
        #### Robust Detector Cascade
        ```
        MTCNN (Multi-task Cascaded Convolutional Networks)
        ↓  (fallback if no face detected or load fails)
        MediaPipe BlazeFace
        ↓  (fallback if MediaPipe fails)
        OpenCV Haar Feature-based Cascade Classifier
        ```
        The cascade guarantees resilient face detection across low-light, extreme angles, and video compression artifacts.
        """)

    # Section 7: Temporal Aggregation
    with st.expander("7. Temporal Aggregation Justification", expanded=False):
        st.markdown("""
        #### Why Temporal Aggregation is Used
        Individual frame predictions can be noisy due to video compression artifacts, motion blur, and momentary pose variation.
        The system combines frame-level probabilities into a video-level decision using certainty and quality weights.

        **Critical design decision:** We do **not** use simple majority voting, as majority voting ignores frame quality disparities and prediction certainty.
        """)

    # Section 8: Verdict Interpretation
    with st.expander("8. Verdict Interpretation Bands", expanded=False):
        st.markdown("""
        #### Engineering Interpretation Bands
        - $P(\\text{fake}) < 40\\%$: **Likely Authentic**
        - $40\\% \\le P(\\text{fake}) < 60\\%$: **Inconclusive — May Be Manipulated**
        - $60\\% \\le P(\\text{fake}) < 80\\%$: **Likely Manipulated**
        - $P(\\text{fake}) \\ge 80\\%$: **Highly Likely Manipulated**

        > **Academic Disclaimer:** These bands represent practical engineering decision boundaries
        > and are **not** statistically calibrated probability intervals.
        """)

    # Section 9: Technology Stack
    with st.expander("9. Verified Technology Stack", expanded=False):
        st.markdown("""
        #### Environment Verification
        - **Programming Language:** Python 3.14
        - **Deep Learning Framework:** PyTorch 2.13.0+cpu
        - **Transformer Library:** Hugging Face Transformers 5.16.1
        - **Vision Backbone:** ViT-Base/16 (`prithivMLmods/Deep-Fake-Detector-v2-Model`)
        - **Face Detection:** facenet-pytorch (MTCNN), MediaPipe 1.0.1, OpenCV 5.0.0
        - **Image Processing:** OpenCV 5.0.0, Pillow, NumPy
        - **Web Application:** Streamlit 1.62.0
        - **Testing Suite:** pytest 9.1.1 (69 automated tests)
        - **Plotting & Publication Export:** Matplotlib 3.11.1
        """)


# ==================== PART 5 & PART 7: PAPER FIGURES ====================
def render_paper_figures():
    """Renders all publication-grade figures with PNG & SVG download options."""
    st.markdown("## 📄 Publication-Ready Paper Figures")
    st.caption("High-resolution, publication-formatted figures with transparent/light backgrounds suitable for IEEE/academic paper submission.")

    # Figure 1: Pipeline
    render_figure_with_downloads(
        rd.fig_pipeline(),
        "00_pipeline_architecture",
        "Figure 1: End-to-End System Architecture & Detection Pipeline",
        "Schematic diagram illustrating the complete detection pipeline from input ingestion to verdict band emission.",
        key_prefix="paper"
    )

    st.markdown("---")

    # Figure 2: Confusion Matrix
    render_figure_with_downloads(
        rd.fig_confusion_matrix(),
        "02_confusion_matrix",
        "Figure 2: Confusion Matrix (Test Split)",
        "2×2 confusion matrix on the preliminary test split (TP=0, TN=1, FP=1, FN=2).",
        key_prefix="paper"
    )

    st.markdown("---")

    # Figure 3: Performance Metrics
    render_figure_with_downloads(
        rd.fig_metrics_comparison(),
        "03_metrics_comparison",
        "Figure 3: Configuration Performance Comparison (Test Split)",
        "Grouped bar chart comparing ViT Only vs. ViT + Temporal across Accuracy, Precision, Recall, and F1.",
        key_prefix="paper"
    )

    st.markdown("---")

    # Figure 4: Video-Level P(fake) Comparison
    render_figure_with_downloads(
        rd.fig_pfake_comparison(),
        "06_video_fake_probability",
        "Figure 4: Video-Level Manipulation Probability Comparison (Test Split)",
        "Evaluation results illustrating the out-of-distribution domain gap relative to the 50% threshold.",
        key_prefix="paper"
    )

    st.markdown("---")

    # Figure 5: Latency Comparison
    render_figure_with_downloads(
        rd.fig_latency_comparison(),
        "04_latency_comparison",
        "Figure 5: Processing Latency Comparison (CPU)",
        "Inference latency (ms), End-to-end processing latency (ms), and effective frames per second.",
        key_prefix="paper"
    )

    st.markdown("---")

    # Figure 5b: FPS Comparison
    render_figure_with_downloads(
        rd.fig_fps_comparison(),
        "05_fps_comparison",
        "Figure 5b: Effective Throughput Comparison (FPS)",
        "Standalone throughput chart illustrating near-real-time CPU processing throughput (6.5 FPS).",
        key_prefix="paper"
    )

    st.markdown("---")

    # Figure 6: Face Quality Comparison
    render_figure_with_downloads(
        rd.fig_face_quality(),
        "07_face_quality_comparison",
        "Figure 6: Average Face Quality by Video (Test Split)",
        "Comparison of face quality scores between authentic interview footage and DeeperForensics fake samples.",
        key_prefix="paper"
    )


# ==================== PROJECT OVERVIEW ====================
def render_project_overview():
    """Renders the academic project overview, viva defence talking points, and summary."""
    st.markdown("## ℹ️ Academic Project Overview")

    st.markdown("""
    ### Project Title
    **"An AI-Based Deepfake Detection System for Video and Real-Time Webcam Analysis Using Vision Transformers"**

    #### Academic Context & Problem Statement
    The widespread availability of deep generative models and face-swapping software has made video verification
    critical in sensitive applications, such as remote job interviews and identity authentication.
    This project explores an end-to-end detection pipeline coupling multi-cue face quality evaluation with
    a pre-trained Vision Transformer (`ViT-B/16`) and uncertainty-aware temporal aggregation.

    #### Key Findings for Viva & Defence
    1. **Pre-trained ViT Capabilities:** The model performs single-frame classification using self-attention across 196 image patches.
    2. **Domain Mismatch:** On our preliminary test subset (DeeperForensics fake videos and real interview footage), the pre-trained ViT exhibited an out-of-distribution domain gap. This highlights that deep learning detectors are sensitive to compression and generative algorithms not seen during pre-training.
    3. **Computational Overhead:** Quality-weighted temporal aggregation adds negligible computational overhead (<1 ms per frame), preserving a near-real-time throughput of approximately 6.5 FPS on standard CPU hardware.
    4. **Uncertainty Awareness:** Because deep learning classifiers output uncalibrated probabilities, the system incorporates engineering interpretation bands (Authentic, Inconclusive, Manipulated) to prevent over-confident misclassifications.
    """)


# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<span class="eyebrow">Academic Research &amp; Demonstration</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">🔍 Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">An AI-Based Deepfake Detection System for Video and Real-Time Webcam Analysis '
        'Using Vision Transformers (ViT-Base/16)</p>',
        unsafe_allow_html=True
    )

    # Load model and face detector
    with st.spinner("🔄 Initializing Vision Transformer & Face Detector..."):
        processor, model = load_deepfake_model()
        detector = load_face_detector()

    if processor is None or model is None:
        st.error("Model failed to load. Please verify internet connectivity or local model cache.")
        return

    fallback_detector = load_fallback_detector()

    # Session state initialization
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.50

    # Top-Level Academic Navigation Tabs
    tab_detect, tab_research, tab_tech, tab_figures, tab_overview = st.tabs([
        "🔍 Detection",
        "📊 Research Results",
        "📐 Technical Details",
        "📄 Paper Figures",
        "ℹ️ Project Overview"
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: DETECTION (VIDEO & WEBCAM)
    # ──────────────────────────────────────────────────────────────────────────
    with tab_detect:
        with st.container(border=True):
            st.markdown("#### Detection Mode & Configuration")
            with st.container(key="mode_toggle"):
                col1, col2 = st.columns(2, gap="small")
                with col1:
                    if st.button(
                        "🎥  Video Detection",
                        key="btn_video",
                        use_container_width=True,
                        type="primary" if st.session_state.mode == "video" else "secondary",
                    ):
                        st.session_state.mode = "video"
                        st.session_state.predictions = []
                        st.session_state.webcam_active = False
                        st.rerun()
                with col2:
                    if st.button(
                        "📡  Live Webcam",
                        key="btn_webcam",
                        use_container_width=True,
                        type="primary" if st.session_state.mode == "webcam" else "secondary",
                    ):
                        st.session_state.mode = "webcam"
                        st.session_state.predictions = []
                        st.rerun()

            st.markdown("")
            col_s1, _ = st.columns(2)
            with col_s1:
                st.slider(
                    "🎯 Frame Certainty Threshold (τ_f)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.confidence_threshold,
                    step=0.05,
                    key="confidence_threshold",
                    help="Minimum frame confidence required for definitive discrete tallying."
                )

            if st.session_state.mode is not None:
                if st.button("🗑️ Reset Detection Session", key="clear_results", use_container_width=True, type="secondary"):
                    st.session_state.predictions = []
                    st.session_state.mode = None
                    st.rerun()

        # VIDEO MODE
        if st.session_state.mode == "video":
            with st.container(border=True):
                st.markdown("### 🎥 Video Detection Mode")
                uploaded_video = st.file_uploader(
                    "Upload a video file (MP4, AVI, MOV)",
                    type=["mp4", "avi", "mov"],
                    help="Upload a video with visible faces for multi-frame deepfake analysis."
                )

                if uploaded_video:
                    st.video(uploaded_video)

                    if st.button("🚀 Run Deepfake Analysis", key="analyze_vid", use_container_width=True, type="primary"):
                        st.markdown("### 🔄 Analyzing Video Frames...")
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        try:
                            tfile.write(uploaded_video.read())
                            tfile.close()

                            cap = cv2.VideoCapture(tfile.name)
                            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 24.0
                            if fps <= 0 or math.isnan(fps):
                                fps = 24.0
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0

                            sampled_frame_indices = sample_video_frame_indices(
                                total_frames=total_frames,
                                fps=fps,
                                max_samples=60,
                                target_sample_fps=3.0
                            )

                            predictions: List[Dict[str, Any]] = []
                            valid_faces_count = 0
                            skipped_low_quality = 0
                            fallback_detector.reset_temporal_state()

                            status_text = st.empty()
                            frame_placeholder = st.empty()
                            total_samples = len(sampled_frame_indices)

                            for idx, frame_no in enumerate(sampled_frame_indices):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    continue

                                timestamp = frame_no / fps if fps > 0 else 0.0
                                faces = detector.detect_faces(frame, max_faces=3)
                                annotated_frame = frame.copy()

                                if faces:
                                    best_face = faces[0]
                                    if not best_face.get('is_valid_quality', True) and best_face.get('quality', 0) < 15.0:
                                        skipped_low_quality += 1
                                        x, y, w, h = best_face['coords']
                                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (148, 163, 184), 2)
                                        cv2.putText(
                                            annotated_frame,
                                            "Low Quality Face",
                                            (x, max(20, y - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            (148, 163, 184),
                                            2
                                        )
                                    else:
                                        valid_faces_count += 1
                                        face_img = best_face['image']
                                        face_landmarks = best_face.get('landmarks')
                                        step_interval = int(sampled_frame_indices[idx] - sampled_frame_indices[idx - 1]) if idx > 0 else 1
                                        eval_res = predict_deepfake_with_fallback(
                                            face_img, processor, model,
                                            quality_score=best_face.get('quality', 50.0),
                                            fallback_detector=fallback_detector,
                                            landmarks=face_landmarks,
                                            frame_interval=step_interval
                                        )

                                        label = eval_res['final_label']
                                        conf = eval_res['final_confidence']

                                        pred_record = {
                                            'frame_index': frame_no,
                                            'timestamp': round(timestamp, 2),
                                            'label': label,
                                            'confidence': conf,
                                            'p_real': eval_res['p_real'],
                                            'p_fake': eval_res['p_fake'],
                                            'quality': best_face.get('quality', 50.0),
                                            'coords': best_face.get('coords'),
                                            'vit_confidence': eval_res.get('vit_confidence', 0.0),
                                        }
                                        predictions.append(pred_record)

                                        x, y, w, h = best_face['coords']
                                        p_fake_frame = eval_res['p_fake']
                                        color = (34, 197, 94) if label == "Realism" else (239, 68, 68)
                                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
                                        cv2.putText(
                                            annotated_frame,
                                            f"P(fake):{p_fake_frame*100:.0f}% Q:{best_face.get('quality',0):.0f}",
                                            (x, max(25, y - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.60,
                                            color,
                                            2
                                        )

                                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                                pct = int((idx + 1) / total_samples * 100) if total_samples > 0 else 100
                                status_text.markdown(
                                    f"⏳ **Analyzing:** sample {idx+1}/{total_samples} "
                                    f"(frame {frame_no}) · {valid_faces_count} faces found · {pct}%"
                                )

                            cap.release()
                        finally:
                            try:
                                os.unlink(tfile.name)
                            except Exception:
                                pass

                        status_text.empty()
                        frame_placeholder.empty()

                        if predictions:
                            agg_result = TemporalAggregator.aggregate(
                                predictions,
                                confidence_threshold=st.session_state.confidence_threshold
                            )
                            # Render improved Part 1 Analysis Results
                            render_video_analysis_results(predictions, agg_result, st.session_state.confidence_threshold)
                        else:
                            st.warning("⚠️ No valid faces detected in the video frames. Please upload footage with clear, visible faces.")

        # WEBCAM MODE
        elif st.session_state.mode == "webcam":
            with st.container(border=True):
                st.markdown("### 📡 Live Webcam Detection Mode")
                st.info("💡 Position your face clearly in front of the camera. Detection runs continuously with near-real-time CPU processing.")

                col_start, col_stop = st.columns(2)
                with col_start:
                    start_btn = st.button("🎥 Start Live Detection", key="start_webcam", use_container_width=True, type="primary")
                with col_stop:
                    stop_btn = st.button("🛑 Stop Detection", key="stop_webcam", use_container_width=True, type="secondary")

                if start_btn:
                    st.session_state.webcam_active = True
                if stop_btn:
                    st.session_state.webcam_active = False

                if st.session_state.webcam_active:
                    st.markdown("---")
                    st.markdown("### 📹 Live Feed")

                    cap = cv2.VideoCapture(0)
                    # Request HD 720p resolution and smooth 30 FPS for crisp camera feed
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if not cap.isOpened():
                        st.error("❌ Cannot access webcam. Please verify camera permissions.")
                        st.session_state.webcam_active = False
                    else:
                        frame_placeholder = st.empty()
                        status_placeholder = st.empty()
                        predictions: List[Dict[str, Any]] = []
                        frame_count = 0
                        FRAME_SKIP = 6
                        ANALYSIS_WINDOW = 120.0  # 2-minute analysis window
                        start_time = time.time()
                        fallback_detector.reset_temporal_state()

                        current_label = None
                        current_confidence = 0.0

                        try:
                            while st.session_state.webcam_active:
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                frame_count += 1
                                current_time = time.time()
                                elapsed_time = current_time - start_time
                                time_remaining = max(0.0, ANALYSIS_WINDOW - elapsed_time)
                                if time_remaining <= 0.0:
                                    start_time = current_time
                                    time_remaining = ANALYSIS_WINDOW

                                faces = detector.detect_faces(frame, max_faces=2)

                                if frame_count % FRAME_SKIP == 0 and len(faces) > 0:
                                    best_face = faces[0]
                                    face_img = best_face['image']
                                    eval_res = predict_deepfake_with_fallback(
                                        face_img, processor, model,
                                        quality_score=best_face.get('quality', 50.0),
                                        fallback_detector=fallback_detector,
                                        landmarks=best_face.get('landmarks'),
                                        frame_interval=FRAME_SKIP
                                    )

                                    label = eval_res['final_label']
                                    confidence = eval_res['final_confidence']

                                    if label and confidence >= st.session_state.confidence_threshold:
                                        predictions.append({
                                            'label': label,
                                            'confidence': confidence,
                                            'p_real': eval_res['p_real'],
                                            'p_fake': eval_res['p_fake'],
                                            'quality': best_face.get('quality', 50.0),
                                            'vit_confidence': eval_res.get('vit_confidence', 0.0)
                                        })
                                        current_label = label
                                        current_confidence = confidence

                                # Draw sleek cyan face bounding boxes matching Screenshot 2025-10-21 173007.png
                                for face_info in faces:
                                    x, y, w, h = face_info['coords']
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                                real_count = sum(1 for p in predictions if p.get('label') == 'Realism')
                                fake_count = sum(1 for p in predictions if p.get('label') == 'Deepfake')

                                # Draw telemetry HUD overlay matching Screenshot 2025-10-21 173007.png
                                draw_live_hud_overlay(
                                    frame,
                                    time_remaining=time_remaining,
                                    predictions_count=len(predictions),
                                    real_count=real_count,
                                    fake_count=fake_count,
                                    current_verdict=current_label,
                                    current_confidence=current_confidence,
                                    mode="Live"
                                )

                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                                if current_label:
                                    if current_label == "Realism":
                                        status_placeholder.markdown(
                                            f'<div class="live-label real">✅ REAL FACE — {current_confidence*100:.1f}% Confidence</div>',
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        status_placeholder.markdown(
                                            f'<div class="live-label fake">⚠️ DEEPFAKE — {current_confidence*100:.1f}% Confidence</div>',
                                            unsafe_allow_html=True
                                        )
                        except Exception as e:
                            st.error(f"Error during live detection: {e}")
                        finally:
                            cap.release()

                        if predictions:
                            agg = TemporalAggregator.aggregate(
                                predictions,
                                confidence_threshold=st.session_state.confidence_threshold
                            )
                            render_webcam_session_results(predictions, agg, st.session_state.confidence_threshold)

        # DEFAULT LANDING VIEW
        else:
            with st.container(border=True):
                st.markdown("### 👆 Select a Detection Mode to Get Started")
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.markdown("""
                    #### 🎥 Video Detection
                    - Multi-frame analysis on MP4, AVI, MOV videos
                    - Face detection with MTCNN / MediaPipe cascade
                    - Multi-cue face quality & blur filtering
                    - Vision Transformer (ViT-B/16) inference
                    - Quality & certainty weighted temporal aggregation
                    - Exportable JSON & CSV results
                    """)
                with col2:
                    st.markdown("""
                    #### 📡 Live Webcam Detection
                    - Near-real-time CPU processing (~6.5 FPS)
                    - Continuous face tracking & alignment
                    - Live overlay of classification probabilities
                    - Burst anomaly & glitch detection
                    - End-of-session aggregate summary
                    """)

            with st.container(border=True):
                st.markdown("### 🎯 System Capabilities")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    render_metric("📊", "ViT-Base/16", "Vision Backbone")
                with col_b:
                    render_metric("🧠", "Cascade", "MTCNN / MP / Haar")
                with col_c:
                    render_metric("⚡", "6.5 FPS", "CPU Latency")
                with col_d:
                    render_metric("🔒", "Weighted", "Aggregation")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: RESEARCH RESULTS DASHBOARD
    # ──────────────────────────────────────────────────────────────────────────
    with tab_research:
        render_research_dashboard()

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: TECHNICAL DETAILS
    # ──────────────────────────────────────────────────────────────────────────
    with tab_tech:
        render_technical_details()

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4: PAPER FIGURES
    # ──────────────────────────────────────────────────────────────────────────
    with tab_figures:
        render_paper_figures()

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5: PROJECT OVERVIEW
    # ──────────────────────────────────────────────────────────────────────────
    with tab_overview:
        render_project_overview()

    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Deepfake Detection System</strong> · Vision Transformer Analysis</p>
        <p>Academic Research &amp; Demonstration · Pre-trained Model: <code>prithivMLmods/Deep-Fake-Detector-v2-Model</code></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()