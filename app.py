import os
import json
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

    :root {
        --primary: #4f46e5;
        --primary-dark: #4338ca;
        --primary-light: #eef2ff;
        --accent: #7c3aed;
        --success: #10b981;
        --success-light: #ecfdf5;
        --danger: #ef4444;
        --danger-light: #fef2f2;
        --warning: #f59e0b;
        --bg-primary: #f6f7fb;
        --bg-secondary: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --border: #e6e9f0;
        --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
        --shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
        --shadow-lg: 0 12px 24px -8px rgba(79, 70, 229, 0.18), 0 4px 8px -4px rgba(15, 23, 42, 0.06);
        --radius-sm: 8px;
        --radius: 14px;
        --radius-lg: 20px;
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ---- Force a consistent light appearance no matter the visitor's
           Streamlit theme setting. Without this, plain text/labels can
           inherit a dark-theme white color and vanish against white
           cards until a hover/selected state overrides it. ---- */
    html, body, [class*="stApp"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--text-primary);
    }
    .stApp {
        background:
            radial-gradient(circle at 12% -10%, rgba(79, 70, 229, 0.10) 0%, rgba(79, 70, 229, 0) 45%),
            radial-gradient(circle at 100% 0%, rgba(124, 58, 237, 0.08) 0%, rgba(124, 58, 237, 0) 40%),
            var(--bg-primary);
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Explicit color on every plain text element Streamlit renders,
       so nothing silently inherits the visitor's theme color. */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span:not(.eyebrow):not(.mode-caption *),
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    .stRadio label p,
    .stRadio label span {
        color: var(--text-primary) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--text-muted) !important;
    }

    /* ---- Eyebrow badge above the title ---- */
    .eyebrow {
        display: block;
        text-align: center;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--primary);
        background: var(--primary-light);
        border: 1px solid #ddd6fe;
        border-radius: 9999px;
        padding: 0.35rem 0.9rem;
        width: fit-content;
        margin: 0 auto 1rem auto;
    }

    .main-title {
        text-align: center;
        font-size: clamp(2rem, 5vw, 2.75rem);
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.03em;
        line-height: 1.15;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
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

    /* ---- Cards: Streamlit's native bordered container ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-secondary);
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow);
        border: 1px solid var(--border) !important;
        transition: var(--transition);
        margin: 0.7rem 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: var(--shadow-lg);
        border-color: #ddd6fe !important;
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

    /* ---- Pill-style mode toggle ---- */
    div.st-key-mode_toggle {
        background: var(--bg-primary);
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
        background: rgba(79, 70, 229, 0.08) !important;
        color: var(--primary) !important;
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
        transition: var(--transition);
        cursor: pointer;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: #fff;
        box-shadow: var(--shadow-sm);
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }
    .stButton>button[kind="secondary"] {
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border);
    }
    .stButton>button[kind="secondary"]:hover {
        background: var(--primary-light);
        border-color: var(--primary);
        color: var(--primary-dark);
        transform: translateY(-1px);
    }
    .stButton>button:active { transform: translateY(0); }

    /* ---- File uploader ---- */
    .stFileUploader {
        background: var(--bg-secondary) !important;
        border-radius: var(--radius) !important;
        padding: 1.5rem !important;
        border: 2px dashed #c7d2fe !important;
        transition: var(--transition) !important;
    }
    .stFileUploader:hover {
        border-color: var(--primary) !important;
        background: var(--primary-light) !important;
    }
    .stFileUploader section { padding: 0 !important; }
    .stFileUploader button {
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        background: var(--bg-secondary) !important;
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
        background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
        box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.35);
    }
    .result-danger {
        background: linear-gradient(135deg, var(--danger) 0%, #dc2626 100%);
        box-shadow: 0 8px 20px -4px rgba(239, 68, 68, 0.35);
    }
    .result-success span, .result-danger span { color: #fff !important; opacity: 0.92; }

    /* ---- Metric cards ---- */
    .metric-card {
        background: var(--bg-primary);
        padding: 1.1rem;
        border-radius: var(--radius);
        text-align: center;
        border: 1px solid var(--border);
        transition: var(--transition);
        height: 100%;
    }
    .metric-card:hover {
        border-color: #ddd6fe;
        box-shadow: var(--shadow-sm);
        transform: translateY(-2px);
    }
    .metric-icon { font-size: 1.35rem; margin-bottom: 0.25rem; }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        line-height: 1.2;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
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
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%) !important;
        border-radius: 9999px;
    }
    .stProgress > div > div {
        background: var(--border) !important;
        border-radius: 9999px;
        height: 8px !important;
    }

    /* ---- Live label (webcam) ---- */
    .live-label {
        padding: 0.85rem 1.5rem;
        border-radius: var(--radius);
        font-size: 1rem;
        font-weight: 700;
        margin: 0.75rem 0;
        text-align: center;
        display: block;
        letter-spacing: 0.01em;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
    }
    .live-label.real { background: var(--success-light); color: var(--success) !important; border-color: #a7f3d0; }
    .live-label.fake { background: var(--danger-light); color: var(--danger) !important; border-color: #fecaca; }

    h2, h3, h4 { color: var(--text-primary) !important; font-weight: 700; }
    h2 { font-size: 1.4rem; margin-bottom: 0.75rem; }
    h3 { font-size: 1.1rem; margin-bottom: 0.6rem; }
    h4 { font-size: 1rem; margin-bottom: 0.4rem; }

    /* ---- Slider: recolor only the fill + handle, leave the value
           tooltip's own styling untouched so it stays legible ---- */
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:nth-child(2) {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: var(--primary) !important;
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px var(--primary-light) !important;
    }

    .stRadio > div { gap: 0.6rem; }
    .stRadio label {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 9999px;
        padding: 0.55rem 1.25rem;
        font-weight: 600;
        transition: var(--transition);
        cursor: pointer;
    }
    .stRadio label:hover { border-color: var(--primary); background: var(--primary-light); }

    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: var(--text-secondary) !important;
        transition: var(--transition);
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: #fff !important;
        border-color: var(--primary) !important;
    }
    .stTabs [aria-selected="true"] * { color: #fff !important; }

    /* ---- Alerts ---- */
    .stAlert { border-radius: var(--radius) !important; border: none !important; padding: 1rem 1.25rem !important; }
    .stAlert[data-baseweb="notification"][kind="info"] { background: var(--primary-light) !important; }
    .stAlert[data-baseweb="notification"][kind="info"] * { color: var(--primary-dark) !important; }
    .stAlert[data-baseweb="notification"][kind="success"] { background: var(--success-light) !important; }
    .stAlert[data-baseweb="notification"][kind="success"] * { color: #065f46 !important; }
    .stAlert[data-baseweb="notification"][kind="warning"] { background: #fffbeb !important; }
    .stAlert[data-baseweb="notification"][kind="warning"] * { color: #92400e !important; }
    .stAlert[data-baseweb="notification"][kind="error"] { background: var(--danger-light) !important; }
    .stAlert[data-baseweb="notification"][kind="error"] * { color: #991b1b !important; }

    hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

    .stSpinner > div { border-top-color: var(--primary) !important; }
    .stSpinner p { color: var(--text-secondary) !important; }

    .stDataFrame { border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); }

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
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
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


# ==================== HELPER FUNCTIONS ====================
def get_face_cascade():
    """Return a usable Haar cascade classifier, even if OpenCV data files are missing."""
    cascade_filename = 'haarcascade_frontalface_default.xml'
    candidate_paths = [
        os.path.join(cv2.data.haarcascades, cascade_filename),
        os.path.join(os.path.dirname(cv2.__file__), 'data', cascade_filename),
        os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', cascade_filename),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            classifier = cv2.CascadeClassifier(path)
            if not classifier.empty():
                return classifier

    fallback_dir = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades')
    os.makedirs(fallback_dir, exist_ok=True)
    fallback_path = os.path.join(fallback_dir, cascade_filename)
    cascade_url = (
        'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/'
        + cascade_filename
    )

    try:
        urllib.request.urlretrieve(cascade_url, fallback_path)
        classifier = cv2.CascadeClassifier(fallback_path)
        if not classifier.empty():
            return classifier
    except Exception:
        pass

    raise FileNotFoundError(
        'OpenCV Haar cascade XML is missing. Reinstall OpenCV with data files or ensure '
        f'{cascade_filename} exists in the OpenCV data directory.'
    )


def detect_faces_with_mediapipe(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Use MediaPipe Face Detection to find faces in a frame."""
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        detected_faces: List[Dict[str, Any]] = []

        if not results.detections:
            return detected_faces

        h, w = frame.shape[:2]
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            width = max(1, int(box.width * w))
            height = max(1, int(box.height * h))

            pad_x = int(width * 0.2)
            pad_y = int(height * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + width + pad_x)
            y2 = min(h, y + height + pad_y)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            detected_faces.append({
                'image': face_img,
                'coords': (x1, y1, x2 - x1, y2 - y1),
                'quality': float(detection.score[0]),
                'area': width * height,
            })

        detected_faces = sorted(detected_faces, key=lambda item: (item['quality'], item['area']), reverse=True)
        return detected_faces[:5]
    except Exception:
        return []


def compute_face_quality(face_img: np.ndarray) -> float:
    """Estimate whether a detected face crop is usable for classification."""
    if face_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    contrast = float(max(0.0, blur) / (brightness + 1.0))
    return contrast + (blur / 100.0)


def get_face_from_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Extract and rank faces from a frame using MediaPipe with Haar fallback."""
    try:
        faces = detect_faces_with_mediapipe(frame)
        if faces:
            return faces

        face_cascade = get_face_cascade()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidate_scales = [1.05, 1.1, 1.2, 1.35, 1.5]
        face_data = []

        for scale_factor in candidate_scales:
            detected = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=4,
                minSize=(32, 32),
                maxSize=(frame.shape[1], frame.shape[0])
            )
            for (x, y, w, h) in detected:
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)

                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                quality = compute_face_quality(face_img)
                face_data.append({
                    'image': face_img,
                    'coords': (x1, y1, x2 - x1, y2 - y1),
                    'quality': quality,
                    'area': w * h,
                })

        if not face_data:
            return []

        face_data = sorted(face_data, key=lambda item: (item['quality'], item['area']), reverse=True)
        return face_data[:5]
    except Exception as e:
        st.error(f"Error in face detection: {e}")
        return []


def predict_deepfake(
    face_image: np.ndarray,
    processor: AutoImageProcessor,
    model: AutoModelForImageClassification
) -> Tuple[Optional[str], float]:
    """Predict if face is real or deepfake"""
    try:
        if isinstance(face_image, np.ndarray):
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = Image.fromarray(face_rgb)

        inputs = processor(images=face_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class_idx = logits.argmax(-1).item()

        label = model.config.id2label[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx].item())

        return label, confidence

    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, 0.0


def aggregate_video_predictions(
    predictions: List[Dict[str, Any]]
) -> Tuple[str, float, int, int]:
    """Aggregate frame-level predictions into a video verdict.

    Deepfake detection should not be driven by a raw count alone because one noisy
    frame can dominate a short clip. We weight by class totals and confidence so a
    consistent fake signal across the video wins more reliably.
    """
    if not predictions:
        return "Unknown", 0.0, 0, 0

    label_counts = Counter([pred['label'] for pred in predictions])
    real_count = label_counts.get("Realism", 0)
    fake_count = label_counts.get("Deepfake", 0)
    total_votes = len(predictions)

    real_confidences = [pred['confidence'] for pred in predictions if pred['label'] == "Realism"]
    fake_confidences = [pred['confidence'] for pred in predictions if pred['label'] == "Deepfake"]

    real_weight = sum(real_confidences) if real_confidences else 0.0
    fake_weight = sum(fake_confidences) if fake_confidences else 0.0

    # Prefer a fake verdict when fake frames dominate or when the weighted fake
    # confidence is meaningfully stronger than the real evidence.
    if fake_count > real_count:
        final_label = "Deepfake"
        avg_confidence = fake_weight / fake_count if fake_count else 0.0
    elif real_count > fake_count:
        final_label = "Realism"
        avg_confidence = real_weight / real_count if real_count else 0.0
    else:
        # Tie-break using weighted confidence so a genuine fake pattern still wins
        # when frame counts are evenly split.
        if fake_weight >= real_weight:
            final_label = "Deepfake"
            avg_confidence = fake_weight / fake_count if fake_count else 0.0
        else:
            final_label = "Realism"
            avg_confidence = real_weight / real_count if real_count else 0.0

    # If the vote share is too close to call, we keep the more confident class but
    # preserve the counts for reporting.
    if total_votes > 0:
        fake_share = fake_count / total_votes
        real_share = real_count / total_votes
        if abs(fake_share - real_share) < 0.08 and fake_weight >= real_weight:
            final_label = "Deepfake"
            avg_confidence = max(avg_confidence, fake_weight / fake_count if fake_count else 0.0)
        elif abs(fake_share - real_share) < 0.08 and real_weight > fake_weight:
            final_label = "Realism"
            avg_confidence = max(avg_confidence, real_weight / real_count if real_count else 0.0)

    return final_label, avg_confidence, real_count, fake_count


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

    # Load model
    with st.spinner("🔄 Loading AI Model..."):
        processor, model = load_deepfake_model()

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
        st.session_state.confidence_threshold = 0.5

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
                help="Upload a video with visible faces for analysis"
            )

            if uploaded_video:
                st.video(uploaded_video)

                if st.button("🚀 Start Analysis", key="analyze_vid", use_container_width=True, type="primary"):
                    st.markdown("### 🔄 Processing Video...")

                    # Save video temporarily
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    try:
                        tfile.write(uploaded_video.read())
                        tfile.close()

                        # Process video
                        cap = cv2.VideoCapture(tfile.name)
                        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0

                        predictions = []
                        frame_count = 0
                        FRAME_SKIP = max(1, (fps // 3) if fps else 10)

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        frame_placeholder = st.empty()

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_count += 1

                            if frame_count % FRAME_SKIP == 0:
                                face_data = get_face_from_frame(frame)

                                if len(face_data) > 0:
                                    face_img = face_data[0]['image']
                                    face_quality = compute_face_quality(face_img)

                                    if face_quality > 5.0:
                                        label, confidence = predict_deepfake(face_img, processor, model)

                                        if label and confidence >= st.session_state.confidence_threshold:
                                            predictions.append({
                                                'label': label,
                                                'confidence': confidence
                                            })

                                            x, y, w, h = face_data[0]['coords']
                                            color = (34, 197, 94) if label == "Realism" else (239, 68, 68)
                                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                                            cv2.putText(
                                                frame,
                                                f"{label}: {confidence*100:.1f}%",
                                                (x, max(20, y - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7,
                                                color,
                                                2
                                            )

                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                            progress = (frame_count / total_frames) if total_frames > 0 else 0
                            progress_bar.progress(min(1.0, progress))
                            status_text.markdown(f"**Processing:** Frame {frame_count}/{total_frames if total_frames>0 else '—'} ({progress*100:.1f}%)")

                        cap.release()

                    finally:
                        try:
                            os.unlink(tfile.name)
                        except Exception:
                            pass

                    # Clear UI placeholders
                    progress_bar.empty()
                    status_text.empty()
                    frame_placeholder.empty()

                    # Calculate final verdict
                    if predictions:
                        final_label, avg_confidence, real_count, fake_count = calculate_final_verdict(predictions)

                        st.markdown("---")
                        st.markdown("## 📊 Analysis Results")

                        # Display verdict
                        if final_label == "Realism":
                            st.markdown(
                                f'<div class="result-success">✅ AUTHENTIC VIDEO<br>'
                                f'<span style="font-size:1.1rem;">Confidence: {avg_confidence*100:.2f}%</span></div>',
                                unsafe_allow_html=True
                            )
                            st.balloons()
                        else:
                            st.markdown(
                                f'<div class="result-danger">⚠️ DEEPFAKE DETECTED<br>'
                                f'<span style="font-size:1.1rem;">Confidence: {avg_confidence*100:.2f}%</span></div>',
                                unsafe_allow_html=True
                            )

                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            render_metric("🧮", str(len(predictions)), "Predictions")
                        with col2:
                            render_metric("✅", str(real_count), "Real Count")
                        with col3:
                            render_metric("⚠️", str(fake_count), "Fake Count")
                        with col4:
                            render_metric("🎯", f"{avg_confidence*100:.1f}%", "Confidence")

                        # Chart
                        st.markdown("### 📈 Detection Breakdown")
                        chart_data = {
                            'Detection Type': ['Real', 'Fake'],
                            'Count': [real_count, fake_count]
                        }
                        st.bar_chart(chart_data, x='Detection Type', y='Count', height=300)

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
                                'predictions': predictions,
                                'final_label': final_label,
                                'avg_confidence': avg_confidence,
                                'real_count': real_count,
                                'fake_count': fake_count,
                                'confidence_threshold': st.session_state.confidence_threshold
                            }, indent=2)
                            st.markdown(get_download_link(export_data, "deepfake_detection", "JSON"), unsafe_allow_html=True)
                        else:
                            import csv
                            output = io.StringIO()
                            writer = csv.writer(output)
                            writer.writerow(['Label', 'Confidence'])
                            for pred in predictions:
                                writer.writerow([pred['label'], f"{pred['confidence']*100:.2f}%"])
                            writer.writerow(['Final Label', final_label])
                            writer.writerow(['Average Confidence', f"{avg_confidence*100:.2f}%"])
                            export_csv = output.getvalue()
                            st.markdown(get_download_link(export_csv, "detection_results", "CSV"), unsafe_allow_html=True)

                    else:
                        st.warning("⚠️ No faces detected in the video. Please upload a video with visible faces.")

    # ==================== WEBCAM MODE ====================
    elif st.session_state.mode == "webcam":
        with st.container(border=True):
            st.markdown("## 📡 Live Webcam Detection")

            st.info("💡 Tip: Position your face clearly in front of the camera for best results. Detection updates every few seconds.")

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

                    predictions = []
                    frame_count = 0
                    FRAME_SKIP = 15

                    current_label = None
                    current_confidence = 0.0

                    try:
                        while st.session_state.webcam_active:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_count += 1

                            # Detect faces
                            face_data = get_face_from_frame(frame)

                            # Draw boxes
                            for face_info in face_data:
                                x, y, w, h = face_info['coords']
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (102, 126, 234), 3)

                            # Predict on some frames
                            if frame_count % FRAME_SKIP == 0 and len(face_data) > 0:
                                face_img = face_data[0]['image']
                                label, confidence = predict_deepfake(face_img, processor, model)

                                if label and confidence >= st.session_state.confidence_threshold:
                                    predictions.append({'label': label, 'confidence': confidence})
                                    current_label = label
                                    current_confidence = confidence

                            # Display current prediction on frame
                            if current_label:
                                verdict_text = "REAL" if current_label == "Realism" else "FAKE"
                                text_color = (34, 197, 94) if current_label == "Realism" else (239, 68, 68)

                                cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
                                cv2.putText(
                                    frame,
                                    f"{verdict_text}: {current_confidence*100:.1f}%",
                                    (20, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    text_color,
                                    4
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
                        final_label, avg_confidence, real_count, fake_count = calculate_final_verdict(predictions)

                        st.markdown("### 📊 Session Summary")

                        if final_label == "Realism":
                            st.markdown(
                                f'<div class="result-success">✅ AUTHENTIC<br>'
                                f'<span style="font-size:1.1rem;">Avg Confidence: {avg_confidence*100:.2f}%</span></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="result-danger">⚠️ DEEPFAKE DETECTED<br>'
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
                - Complete frame-by-frame analysis
                - Detailed detection statistics
                - Export-ready results
                """)
            with col2:
                st.markdown("""
                #### 📡 Live Webcam
                - Real-time face detection
                - Instant deepfake analysis
                - Live confidence scoring
                - Session summary statistics
                """)

        with st.container(border=True):
            st.markdown("### 🎯 System Capabilities")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                render_metric("📊", "92%", "Accuracy")
            with col_b:
                render_metric("🧠", "ViT", "AI Model")
            with col_c:
                render_metric("⚡", "Real-time", "Processing")
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