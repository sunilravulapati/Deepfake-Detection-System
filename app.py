import os
import json
import math
import urllib.request
import streamlit as st
import cv2
import mediapipe as mp
import torch
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
from collections import Counter
import tempfile

# NOTE: st.container(key=...) used for the pill-style mode toggle requires
# Streamlit >= 1.35. If you're on an older version, upgrade with:
#   pip install --upgrade streamlit

st.set_page_config(page_title="Deepfake Detection System", page_icon="🔍", layout="centered")

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ================================================================
       DESIGN TOKENS — derived from Streamlit's built-in theme vars
       so everything auto-switches between light and dark mode.
       ================================================================ */
    :root {
        /* Streamlit exposes these at runtime:
             --primary-color             (accent/brand)
             --background-color          (page bg)
             --secondary-background-color(card/sidebar bg)
             --text-color                (body text)
        */
        --accent:          var(--primary-color, #6366f1);
        --accent-muted:    color-mix(in srgb, var(--accent) 14%, transparent);
        --accent-hover:    color-mix(in srgb, var(--accent) 22%, transparent);
        --accent-strong:   color-mix(in srgb, var(--accent) 85%, #000);

        --bg:              var(--background-color, #ffffff);
        --bg-card:         var(--secondary-background-color, #f8fafc);
        --text:            var(--text-color, #0f172a);
        --text-secondary:  color-mix(in srgb, var(--text) 65%, transparent);
        --text-muted:      color-mix(in srgb, var(--text) 42%, transparent);
        --border:          color-mix(in srgb, var(--text) 10%, transparent);
        --border-hover:    color-mix(in srgb, var(--accent) 35%, transparent);

        --shadow-sm:       0 1px 3px color-mix(in srgb, var(--text) 5%, transparent);
        --shadow:          0 4px 12px color-mix(in srgb, var(--text) 7%, transparent);
        --shadow-lg:       0 12px 28px color-mix(in srgb, var(--accent) 14%, transparent),
                           0 4px 10px color-mix(in srgb, var(--text) 5%, transparent);

        --radius-sm: 10px;
        --radius:    14px;
        --radius-lg: 20px;
        --ease:      cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ---- Base reset ---- */
    html, body, [class*="stApp"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background: var(--bg);
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        max-width: 920px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ---- Eyebrow badge ---- */
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
        padding: 0.35rem 0.9rem;
        width: fit-content;
        margin: 0 auto 1rem auto;
    }

    /* ---- Title ---- */
    .main-title {
        text-align: center;
        font-size: clamp(2rem, 5vw, 2.75rem);
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.03em;
        line-height: 1.15;
        background: linear-gradient(135deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 70%, #a855f7) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        text-align: center;
        font-size: 1.0625rem;
        color: var(--text-secondary) !important;
        margin-bottom: 2rem;
        font-weight: 400;
        max-width: 560px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }

    /* ---- Cards ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-card);
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow);
        border: 1px solid var(--border) !important;
        transition: all 0.25s var(--ease);
        margin: 0.7rem 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: var(--shadow-lg);
        border-color: var(--border-hover) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.35rem 0.35rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown,
    div[data-testid="stVerticalBlockBorderWrapper"] .stButton,
    div[data-testid="stVerticalBlockBorderWrapper"] .stSlider,
    div[data-testid="stVerticalBlockBorderWrapper"] .stFileUploader {
        margin-bottom: 0.4rem;
    }

    /* ---- Mode toggle pill ---- */
    div.st-key-mode_toggle {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 9999px;
        padding: 6px;
    }
    div.st-key-mode_toggle .stButton>button {
        border-radius: 9999px !important;
        font-weight: 600;
    }
    div.st-key-mode_toggle .stButton>button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--text-secondary) !important;
    }
    div.st-key-mode_toggle .stButton>button[kind="secondary"]:hover {
        background: var(--accent-muted) !important;
        color: var(--accent) !important;
        transform: none;
    }
    div.st-key-mode_toggle .stButton>button[kind="primary"] {
        box-shadow: var(--shadow) !important;
    }

    /* ---- Buttons ---- */
    .stButton>button {
        width: 100%;
        border: none;
        padding: 0.8rem 1.5rem;
        font-size: 0.9375rem;
        font-weight: 600;
        border-radius: var(--radius);
        transition: all 0.2s var(--ease);
        cursor: pointer;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 70%, #a855f7) 100%);
        color: #fff;
        box-shadow: var(--shadow-sm);
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
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
        transform: translateY(-1px);
    }
    .stButton>button:active { transform: translateY(0); }

    /* ---- File uploader ---- */
    .stFileUploader {
        background: var(--bg-card) !important;
        border-radius: var(--radius) !important;
        padding: 1.5rem !important;
        border: 2px dashed var(--border-hover) !important;
        transition: all 0.2s var(--ease) !important;
    }
    .stFileUploader:hover {
        border-color: var(--accent) !important;
        background: var(--accent-muted) !important;
    }
    .stFileUploader section { padding: 0 !important; }
    .stFileUploader button {
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        background: var(--bg-card) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ---- Verdict banners ---- */
    .result-success, .result-danger {
        color: #fff !important;
        padding: 1.5rem 2rem;
        border-radius: var(--radius);
        text-align: center;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1.25rem 0;
    }
    .result-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.35);
    }
    .result-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 8px 20px -4px rgba(239, 68, 68, 0.35);
    }
    .result-success span, .result-danger span { color: #fff !important; opacity: 0.92; }

    /* ---- Metric cards ---- */
    .metric-card {
        background: var(--bg);
        padding: 1.1rem;
        border-radius: var(--radius);
        text-align: center;
        border: 1px solid var(--border);
        transition: all 0.25s var(--ease);
        height: 100%;
    }
    .metric-card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-sm);
        transform: translateY(-2px);
    }
    .metric-icon { font-size: 1.35rem; margin-bottom: 0.25rem; }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        line-height: 1.2;
        background: linear-gradient(135deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 70%, #a855f7) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.72rem;
        color: var(--text-muted) !important;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ---- Progress bar ---- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 70%, #a855f7)) !important;
        border-radius: 9999px;
    }
    .stProgress > div > div {
        background: var(--border) !important;
        border-radius: 9999px;
        height: 8px !important;
    }

    /* ---- Live labels (webcam) ---- */
    .live-label {
        padding: 0.85rem 1.5rem;
        border-radius: var(--radius);
        font-size: 1rem;
        font-weight: 700;
        margin: 0.75rem 0;
        text-align: center;
        display: block;
        letter-spacing: 0.01em;
        box-shadow: var(--shadow-sm);
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

    /* ---- Headings ---- */
    h2, h3, h4 { font-weight: 700; }
    h2 { font-size: 1.4rem; margin-bottom: 0.75rem; }
    h3 { font-size: 1.1rem; margin-bottom: 0.6rem; }
    h4 { font-size: 1rem; margin-bottom: 0.4rem; }

    /* ---- Slider ---- */
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:nth-child(2) {
        background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 70%, #a855f7)) !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px var(--accent-muted) !important;
    }

    /* ---- Radio pills ---- */
    .stRadio > div { gap: 0.6rem; }
    .stRadio label {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 9999px;
        padding: 0.55rem 1.25rem;
        font-weight: 600;
        transition: all 0.2s var(--ease);
        cursor: pointer;
    }
    .stRadio label:hover {
        border-color: var(--accent);
        background: var(--accent-muted);
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: var(--text-secondary) !important;
        transition: all 0.2s var(--ease);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: #fff !important;
        border-color: var(--accent) !important;
    }
    .stTabs [aria-selected="true"] * { color: #fff !important; }

    /* ---- Alerts ---- */
    .stAlert {
        border-radius: var(--radius) !important;
        border: none !important;
        padding: 1rem 1.25rem !important;
    }

    hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

    .stSpinner > div { border-top-color: var(--accent) !important; }
    .stSpinner p { color: var(--text-secondary) !important; }

    .stDataFrame { border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); }

    /* ---- Footer ---- */
    .footer {
        text-align: center;
        padding: 1.5rem 1rem 0.5rem 1rem;
        font-size: 0.85rem;
        border-top: 1px solid var(--border);
        margin-top: 1.5rem;
    }
    .footer p { color: var(--text-muted) !important; margin: 0.1rem 0; }
    .footer strong { color: var(--text-secondary) !important; }

    .mode-caption {
        color: var(--text-secondary) !important;
        font-size: 0.85rem;
        text-align: center;
        margin-bottom: 0.6rem;
    }

    /* ---- Status text during analysis ---- */
    .analysis-status {
        font-size: 0.85rem;
        color: var(--text-secondary) !important;
        text-align: center;
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== MODEL & DETECTOR LOADING ====================
from pipeline import (
    RobustFaceDetector,
    get_detector,
    compute_face_quality as pipeline_compute_quality,
    align_and_crop_face,
    predict_deepfake as pipeline_predict,
    TemporalAggregator,
    aggregate_video_predictions,
    sample_video_frame_indices
)

@st.cache_resource(show_spinner=False)
def load_deepfake_model() -> Tuple[Optional[AutoImageProcessor], Optional[AutoModelForImageClassification]]:
    """Load the pre-trained deepfake detection model"""
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
    """Load the robust face detector (MTCNN / MediaPipe Tasks / Haar)"""
    return get_detector()


# ==================== HELPER FUNCTIONS ====================
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
    """Predict if face is real or deepfake"""
    label, conf, _, _ = pipeline_predict(face_image, processor, model)
    return label, conf


def calculate_final_verdict(
    predictions: List[Dict[str, Any]]
) -> Tuple[str, float, int, int]:
    """Backward-compatible wrapper around the improved video aggregation logic."""
    return aggregate_video_predictions(predictions)


def render_metric(icon: str, value: str, label: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<span class="eyebrow">Interview Security</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">🔍 Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered video interview security &amp; authenticity verification</p>', unsafe_allow_html=True)

    # Load model and face detector
    with st.spinner("🔄 Loading AI Model & Face Detector..."):
        processor, model = load_deepfake_model()
        detector = load_face_detector()

    if processor is None or model is None:
        st.error("Model failed to load. Please refresh and try again.")
        return

    # Initialize session state
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.50

    # ---------------- Mode selection card ----------------
    with st.container(border=True):
        st.markdown("#### Choose a detection mode")

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
        st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            key="confidence_threshold",
            help="Minimum confidence percentage for a prediction to count"
        )

        if st.button("🗑️ Clear Results", key="clear_results", use_container_width=True, type="secondary"):
            st.session_state.predictions = []
            st.session_state.mode = None
            st.rerun()

    # ==================== VIDEO MODE ====================
    if st.session_state.mode == "video":
        with st.container(border=True):
            st.markdown("## 🎥 Video Detection Mode")

            uploaded_video = st.file_uploader(
                "Upload a video file (MP4, AVI, MOV)",
                type=["mp4", "avi", "mov"],
                help="Upload a video with visible faces for multi-frame deepfake analysis"
            )

            if uploaded_video:
                st.video(uploaded_video)

                if st.button("🚀 Start Analysis", key="analyze_vid", use_container_width=True, type="primary"):
                    st.markdown("### 🔄 Analyzing Video Frames...")

                    # Save video temporarily
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    try:
                        tfile.write(uploaded_video.read())
                        tfile.close()

                        # Process video
                        cap = cv2.VideoCapture(tfile.name)
                        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 24.0
                        if fps <= 0 or math.isnan(fps):
                            fps = 24.0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0

                        # Sample frames across video duration
                        sampled_frame_indices = sample_video_frame_indices(
                            total_frames=total_frames,
                            fps=fps,
                            max_samples=60,
                            target_sample_fps=3.0
                        )

                        predictions: List[Dict[str, Any]] = []
                        valid_faces_count = 0
                        skipped_low_quality = 0

                        status_text = st.empty()
                        frame_placeholder = st.empty()

                        total_samples = len(sampled_frame_indices)

                        for idx, frame_no in enumerate(sampled_frame_indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                continue

                            timestamp = frame_no / fps if fps > 0 else 0.0

                            # Detect faces with robust detector
                            faces = detector.detect_faces(frame, max_faces=3)

                            annotated_frame = frame.copy()

                            if faces:
                                # Prioritize most prominent face
                                best_face = faces[0]

                                # Filter low quality
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
                                    label, conf, p_real, p_fake = pipeline_predict(face_img, processor, model)

                                    if label:
                                        pred_record = {
                                            'frame_index': frame_no,
                                            'timestamp': round(timestamp, 2),
                                            'label': label,
                                            'confidence': conf,
                                            'p_real': p_real,
                                            'p_fake': p_fake,
                                            'quality': best_face.get('quality', 50.0),
                                            'coords': best_face.get('coords')
                                        }
                                        predictions.append(pred_record)

                                        x, y, w, h = best_face['coords']
                                        color = (34, 197, 94) if label == "Realism" else (239, 68, 68)
                                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
                                        cv2.putText(
                                            annotated_frame,
                                            f"{label}: {conf*100:.1f}% (Q:{best_face.get('quality',0):.0f})",
                                            (x, max(25, y - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7,
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

                    # Clear UI placeholders
                    status_text.empty()
                    frame_placeholder.empty()

                    # Calculate final verdict using TemporalAggregator
                    if predictions:
                        agg_result = TemporalAggregator.aggregate(
                            predictions,
                            confidence_threshold=st.session_state.confidence_threshold
                        )

                        final_label = agg_result['final_label']
                        avg_confidence = agg_result['avg_confidence']
                        real_count = agg_result['real_count']
                        fake_count = agg_result['fake_count']
                        total_preds = agg_result['total_predictions']

                        st.markdown("---")
                        st.markdown("## 📊 Analysis Results")

                        # Display verdict
                        if final_label == "Realism":
                            st.markdown(
                                f'<div class="result-success">✅ AUTHENTIC VIDEO<br>'
                                f'<span style="font-size:1.1rem;">Confidence: {avg_confidence*100:.2f}% ({agg_result["verdict_summary"]})</span></div>',
                                unsafe_allow_html=True
                            )
                            st.balloons()
                        else:
                            st.markdown(
                                f'<div class="result-danger">⚠️ DEEPFAKE DETECTED<br>'
                                f'<span style="font-size:1.1rem;">Confidence: {avg_confidence*100:.2f}% ({agg_result["verdict_summary"]})</span></div>',
                                unsafe_allow_html=True
                            )

                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            render_metric("🧮", str(total_preds), "Analyzed Frames")
                        with col2:
                            render_metric("✅", str(real_count), "Authentic Frames")
                        with col3:
                            render_metric("⚠️", str(fake_count), "Fake Frames")
                        with col4:
                            render_metric("🎯", f"{avg_confidence*100:.1f}%", "Confidence Score")

                        # Chart: Breakdown & Timeline
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            st.markdown("### 📈 Detection Breakdown")
                            chart_data = {
                                'Detection Type': ['Authentic (Real)', 'Deepfake'],
                                'Count': [real_count, fake_count]
                            }
                            st.bar_chart(chart_data, x='Detection Type', y='Count', height=260)

                        with col_c2:
                            st.markdown("### ⏱️ Temporal Authenticity Timeline")
                            timeline_df = {
                                'Time (s)': [p['timestamp'] for p in predictions],
                                'Fake Probability (%)': [round(p['p_fake'] * 100, 1) for p in predictions]
                            }
                            st.line_chart(timeline_df, x='Time (s)', y='Fake Probability (%)', height=260)

                        # Export results
                        st.markdown("### 💾 Export Results")
                        export_format = st.radio("Export format:", ["JSON", "CSV"], horizontal=True)
                        import base64
                        import io

                        def get_download_link(data: str, filename: str, format_type: str) -> str:
                            b64 = base64.b64encode(data.encode()).decode()
                            mime = "application/json" if format_type == "JSON" else "text/csv"
                            return f'<a href="data:{mime};base64,{b64}" download="{filename}.{format_type.lower()}">Download {format_type} File</a>'

                        if export_format == "JSON":
                            export_data = json.dumps({
                                'verdict': final_label,
                                'confidence': avg_confidence,
                                'real_count': real_count,
                                'fake_count': fake_count,
                                'total_predictions': total_preds,
                                'weighted_real_score': agg_result.get('weighted_real_score', 0.0),
                                'weighted_fake_score': agg_result.get('weighted_fake_score', 0.0),
                                'confidence_threshold': st.session_state.confidence_threshold,
                                'frame_predictions': predictions
                            }, indent=2)
                            st.markdown(get_download_link(export_data, "deepfake_detection_results", "JSON"), unsafe_allow_html=True)
                        else:
                            import csv
                            output = io.StringIO()
                            writer = csv.writer(output)
                            writer.writerow(['Frame Index', 'Timestamp (s)', 'Label', 'Confidence', 'Real Prob', 'Fake Prob', 'Quality'])
                            for pred in predictions:
                                writer.writerow([
                                    pred['frame_index'],
                                    pred['timestamp'],
                                    pred['label'],
                                    f"{pred['confidence']*100:.2f}%",
                                    f"{pred['p_real']*100:.2f}%",
                                    f"{pred['p_fake']*100:.2f}%",
                                    f"{pred['quality']:.1f}"
                                ])
                            writer.writerow([])
                            writer.writerow(['Final Verdict', final_label])
                            writer.writerow(['Overall Confidence', f"{avg_confidence*100:.2f}%"])
                            export_csv = output.getvalue()
                            st.markdown(get_download_link(export_csv, "deepfake_detection_results", "CSV"), unsafe_allow_html=True)

                    else:
                        st.warning("⚠️ No faces detected in the video frames. Please upload a clear video with visible faces.")

    # ==================== WEBCAM MODE ====================
    elif st.session_state.mode == "webcam":
        with st.container(border=True):
            st.markdown("## 📡 Live Webcam Detection")

            st.info("💡 Tip: Position your face clearly in front of the camera for best results. Detection runs continuously with live tracking.")

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

                if not cap.isOpened():
                    st.error("❌ Cannot access webcam. Please check your camera permissions.")
                    st.session_state.webcam_active = False
                else:
                    frame_placeholder = st.empty()
                    status_placeholder = st.empty()

                    predictions: List[Dict[str, Any]] = []
                    frame_count = 0
                    FRAME_SKIP = 6

                    current_label = None
                    current_confidence = 0.0

                    try:
                        while st.session_state.webcam_active:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_count += 1

                            # Detect faces with robust detector
                            faces = detector.detect_faces(frame, max_faces=2)

                            # Predict on sampled frames
                            if frame_count % FRAME_SKIP == 0 and len(faces) > 0:
                                best_face = faces[0]
                                face_img = best_face['image']
                                label, confidence, p_real, p_fake = pipeline_predict(face_img, processor, model)

                                if label and confidence >= st.session_state.confidence_threshold:
                                    predictions.append({
                                        'label': label,
                                        'confidence': confidence,
                                        'p_real': p_real,
                                        'p_fake': p_fake,
                                        'quality': best_face.get('quality', 50.0)
                                    })
                                    current_label = label
                                    current_confidence = confidence

                            # Draw boxes for all detected faces
                            for face_info in faces:
                                x, y, w, h = face_info['coords']
                                box_color = (34, 197, 94) if current_label == "Realism" else (239, 68, 68) if current_label == "Deepfake" else (102, 126, 234)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

                            # Display current prediction on frame banner
                            if current_label:
                                verdict_text = "REAL" if current_label == "Realism" else "FAKE"
                                text_color = (34, 197, 94) if current_label == "Realism" else (239, 68, 68)

                                cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
                                cv2.putText(
                                    frame,
                                    f"{verdict_text}: {current_confidence*100:.1f}%",
                                    (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.3,
                                    text_color,
                                    3
                                )

                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                            if current_label:
                                if current_label == "Realism":
                                    status_placeholder.markdown(
                                        f'<div class="live-label real">✅ REAL FACE - {current_confidence*100:.1f}% Confidence</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    status_placeholder.markdown(
                                        f'<div class="live-label fake">⚠️ DEEPFAKE - {current_confidence*100:.1f}% Confidence</div>',
                                        unsafe_allow_html=True
                                    )
                    except Exception as e:
                        st.error(f"Error during live detection: {e}")
                    finally:
                        cap.release()

                    if predictions:
                        st.markdown("---")
                        agg = TemporalAggregator.aggregate(predictions, confidence_threshold=st.session_state.confidence_threshold)
                        final_label = agg['final_label']
                        avg_confidence = agg['avg_confidence']
                        real_count = agg['real_count']
                        fake_count = agg['fake_count']

                        st.markdown("### 📊 Session Summary")

                        if final_label == "Realism":
                            st.markdown(
                                f'<div class="result-success">✅ AUTHENTIC SESSION<br>'
                                f'<span style="font-size:1.1rem;">Avg Confidence: {avg_confidence*100:.2f}%</span></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="result-danger">⚠️ DEEPFAKE DETECTED IN SESSION<br>'
                                f'<span style="font-size:1.1rem;">Avg Confidence: {avg_confidence*100:.2f}%</span></div>',
                                unsafe_allow_html=True
                            )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            render_metric("🧮", str(len(predictions)), "Total Predictions")
                        with col2:
                            render_metric("✅", str(real_count), "Real Detections")
                        with col3:
                            render_metric("⚠️", str(fake_count), "Fake Detections")

    # ==================== DEFAULT VIEW ====================
    else:
        with st.container(border=True):
            st.markdown("### 👆 Select a Detection Mode to Get Started")

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown("""
                #### 🎥 Video Detection
                - Upload MP4, AVI, or MOV files
                - Multi-frame face detection with MTCNN / MediaPipe
                - Face quality and blur filtering
                - Weighted confidence aggregation
                - Detection timeline &amp; breakdown chart
                - Export-ready JSON &amp; CSV results
                """)
            with col2:
                st.markdown("""
                #### 📡 Live Webcam
                - High-precision live face tracking
                - Instant deepfake classification
                - Real-time confidence scoring
                - Session summary &amp; statistics
                """)

        with st.container(border=True):
            st.markdown("### 🎯 System Capabilities")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                render_metric("📊", "94%+", "Accuracy")
            with col_b:
                render_metric("🧠", "ViT + MTCNN", "AI Pipeline")
            with col_c:
                render_metric("⚡", "Multi-frame", "Aggregation")
            with col_d:
                render_metric("🔒", "Secure", "Analysis")

    # ==================== FOOTER ====================
    st.markdown("""
    <div class="footer">
        <p style='font-weight: 700;'>Deepfake Detection System</p>
        <p>AI-powered interview security &amp; authenticity verification</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()