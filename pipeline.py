"""
Robust Video-Analysis Pipeline for Deepfake Detection
Provides:
- RobustFaceDetector with MTCNN and MediaPipe Task support
- Contextual face padding, square aspect ratio preservation, and landmark alignment
- Blur, resolution, and exposure quality filtering
- ViT-based deepfake classification with soft probability output
- Weighted temporal confidence aggregation and anomaly detection
"""

import os
import math
import logging
from typing import Optional, Tuple, List, Dict, Any, Union
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ===========================================================================
# FEATURE FLAG — Quality-Aware Multi-Cue Fallback Detection (QMC-FD)
# ---------------------------------------------------------------------------
# ENABLE_QMC = False  →  Active/default mode.
#   The application uses the pure ViT classification path for every frame.
#   CustomFallbackDetector, spatial, temporal, and structural scorers are
#   NOT called during normal inference.  All per-frame decisions come from
#   the pre-trained ViT model's softmax probabilities.
#
# ENABLE_QMC = True   →  Experimental / research mode ONLY.
#   Enables PRIMARY_VIT / FALLBACK_FUSION / FALLBACK_ONLY routing via the
#   QMC-FD mechanism.  This path is frozen for the active application and
#   preserved for future experimental evaluation.
# ===========================================================================
ENABLE_QMC: bool = False

# QMC module imported but NOT invoked when ENABLE_QMC = False.
# Do NOT remove this import — it preserves future experimental capability.
from custom_fallback import (
    CustomFallbackDetector,
    FallbackConfig,
    DecisionMode,
    SpatialArtifactAnalyzer,
    TemporalConsistencyAnalyzer,
    StructuralAnalyzer,
    QualityAwareFusion
)

logger = logging.getLogger(__name__)


# ==================== FACE QUALITY FILTERING ====================

def compute_face_quality(
    face_img: np.ndarray,
    min_size: int = 40,
    min_blur_var: float = 20.0
) -> Dict[str, Any]:
    """
    Evaluate if a face crop is high enough quality for reliable model inference.
    Checks:
    - Dimensions (min height/width)
    - Motion blur / sharpness (Laplacian variance)
    - Lighting / exposure (mean brightness and contrast standard deviation)
    """
    if face_img is None or face_img.size == 0:
        return {
            'is_valid': False,
            'score': 0.0,
            'blur_var': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'width': 0,
            'height': 0,
            'reason': 'Empty face image'
        }

    h, w = face_img.shape[:2]
    if h < min_size or w < min_size:
        return {
            'is_valid': False,
            'score': 0.0,
            'blur_var': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'width': w,
            'height': h,
            'reason': f'Face crop too small ({w}x{h} < {min_size}x{min_size})'
        }

    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    # Sharpness via Laplacian variance
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Reject severe blur, severe underexposure, severe overexposure, or flat image
    is_valid = True
    reason = "Valid"

    if blur_var < min_blur_var:
        is_valid = False
        reason = f"Image too blurry (Laplacian variance {blur_var:.1f} < {min_blur_var})"
    elif brightness < 12.0:
        is_valid = False
        reason = f"Extreme underexposure (Brightness {brightness:.1f} < 12.0)"
    elif brightness > 248.0:
        is_valid = False
        reason = f"Extreme overexposure (Brightness {brightness:.1f} > 248.0)"
    elif contrast < 6.0:
        is_valid = False
        reason = f"Insufficient contrast (Std {contrast:.1f} < 6.0)"

    # Compute a continuous quality score [0.0 - 100.0]
    # Higher sharpness and balanced exposure give higher score
    norm_blur = min(1.0, blur_var / 300.0)
    norm_exposure = 1.0 - abs(brightness - 128.0) / 128.0
    norm_contrast = min(1.0, contrast / 60.0)
    norm_size = min(1.0, math.sqrt(h * w) / 200.0)

    quality_score = float((0.4 * norm_blur + 0.2 * norm_exposure + 0.2 * norm_contrast + 0.2 * norm_size) * 100.0)

    return {
        'is_valid': is_valid,
        'score': round(quality_score, 2),
        'blur_var': round(blur_var, 2),
        'brightness': round(brightness, 2),
        'contrast': round(contrast, 2),
        'width': w,
        'height': h,
        'reason': reason
    }


# ==================== FACE CROPPING & ALIGNMENT ====================

def align_and_crop_face(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    landmarks: Optional[np.ndarray] = None,
    padding_ratio: float = 0.10
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract face with contextual padding and square aspect ratio preservation.
    Optionally aligns facial tilt if eye landmarks are provided.

    Args:
        frame: BGR image frame
        box: (x, y, w, h) bounding box
        landmarks: Optional (5, 2) facial landmark points (left eye, right eye, nose, mouth corners)
        padding_ratio: Margin to expand around the detected face (0.10 = 10% extra on each border)

    Returns:
        cropped_face: BGR numpy image
        padded_coords: (x1, y1, w, h) coords in original frame
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = box

    # Perform landmark-based alignment if eyes are detected
    if landmarks is not None and len(landmarks) >= 2:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))

        # If tilt is moderate (< 40 degrees), align horizontally
        if abs(angle) > 3.0 and abs(angle) < 40.0:
            eyes_center = (float((left_eye[0] + right_eye[0]) / 2.0), float((left_eye[1] + right_eye[1]) / 2.0))
            rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
            rotated_frame = cv2.warpAffine(frame, rot_mat, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frame = rotated_frame

    # Compute center of face box
    cx = x + w / 2.0
    cy = y + h / 2.0

    # Expand to square with subtle padding
    side = max(w, h) * (1.0 + padding_ratio * 2.0)

    half_side = side / 2.0
    x1 = int(round(cx - half_side))
    y1 = int(round(cy - half_side))
    x2 = int(round(cx + half_side))
    y2 = int(round(cy + half_side))

    # Calculate padding needed if bounding box exceeds frame boundary
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - fw)
    pad_bottom = max(0, y2 - fh)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(fw, x2)
    src_y2 = min(fh, y2)

    crop = frame[src_y1:src_y2, src_x1:src_x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT
        )

    coords = (max(0, x1), max(0, y1), max(1, x2 - x1), max(1, y2 - y1))
    return crop, coords


# ==================== ROBUST FACE DETECTOR ====================

class RobustFaceDetector:
    """
    Multi-backend face detector supporting:
    1. MTCNN (facenet-pytorch) - high precision, 5 landmarks
    2. MediaPipe Task FaceDetector - lightweight and fast
    3. OpenCV Haar Cascade / YuNet - reliable fallback
    """

    def __init__(self, preferred_backend: str = "auto", min_confidence: float = 0.50):
        self.min_confidence = min_confidence
        self.backend = None
        self.mtcnn = None
        self.mp_detector = None
        self.haar_cascade = None

        self._init_detector(preferred_backend)

    def _init_detector(self, preferred_backend: str):
        # 1. Try MTCNN first if preferred or auto
        if preferred_backend in ("auto", "mtcnn"):
            try:
                from facenet_pytorch import MTCNN
                self.mtcnn = MTCNN(
                    keep_all=True,
                    device='cpu',
                    thresholds=[0.6, 0.7, 0.7],
                    min_face_size=32,
                    post_process=False
                )
                self.backend = "MTCNN"
                logger.info("Initialized MTCNN face detector")
                return
            except Exception as e:
                logger.warning(f"Could not initialize MTCNN: {e}")

        # 2. Try MediaPipe Task FaceDetector
        if preferred_backend in ("auto", "mediapipe") and self.backend is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                model_path = os.path.join(os.path.dirname(__file__), 'models', 'blaze_face_short_range.tflite')
                if not os.path.exists(model_path):
                    # Ensure directory exists and download model if missing
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    import urllib.request
                    url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
                    urllib.request.urlretrieve(url, model_path)

                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=self.min_confidence
                )
                self.mp_detector = vision.FaceDetector.create_from_options(options)
                self.backend = "MediaPipe"
                logger.info("Initialized MediaPipe Task face detector")
                return
            except Exception as e:
                logger.warning(f"Could not initialize MediaPipe FaceDetector: {e}")

        # 3. Fallback to Haar Cascade
        self._init_haar_cascade()

    def _init_haar_cascade(self):
        try:
            cascade_filename = 'haarcascade_frontalface_default.xml'
            candidate_paths = [
                os.path.join(cv2.data.haarcascades, cascade_filename),
                os.path.join(os.path.dirname(cv2.__file__), 'data', cascade_filename),
                os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', cascade_filename),
            ]
            for path in candidate_paths:
                if os.path.exists(path):
                    self.haar_cascade = cv2.CascadeClassifier(path)
                    if not self.haar_cascade.empty():
                        self.backend = "Haar Cascade"
                        return

            # Download if missing
            fallback_dir = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades')
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_path = os.path.join(fallback_dir, cascade_filename)
            import urllib.request
            cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/' + cascade_filename
            urllib.request.urlretrieve(cascade_url, fallback_path)
            self.haar_cascade = cv2.CascadeClassifier(fallback_path)
            self.backend = "Haar Cascade"
        except Exception as e:
            logger.error(f"Error initializing Haar Cascade: {e}")
            self.backend = "None"

    def detect_faces(
        self,
        frame: np.ndarray,
        max_faces: int = 4,
        padding_ratio: float = 0.10
    ) -> List[Dict[str, Any]]:
        """
        Detect and extract faces from a BGR image frame.
        Returns list of face dicts sorted by prominence (area & quality).
        """
        if frame is None or frame.size == 0:
            return []

        results = []

        # 1. MTCNN Detection
        if self.backend == "MTCNN" and self.mtcnn is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)

                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        prob = float(probs[i]) if probs is not None else 0.90
                        if prob < self.min_confidence:
                            continue

                        x1, y1, x2, y2 = [int(v) for v in box]
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        lm = landmarks[i] if (landmarks is not None and i < len(landmarks)) else None

                        face_crop, padded_coords = align_and_crop_face(
                            frame, (x1, y1, w, h), landmarks=lm, padding_ratio=padding_ratio
                        )
                        quality_info = compute_face_quality(face_crop)
                        crop_lm = lm - np.array([padded_coords[0], padded_coords[1]], dtype=np.float32) if lm is not None else None
                        results.append({
                            'image': face_crop,
                            'coords': padded_coords,
                            'raw_box': (x1, y1, w, h),
                            'detector_confidence': prob,
                            'quality': quality_info['score'],
                            'is_valid_quality': quality_info['is_valid'],
                            'quality_details': quality_info,
                            'area': w * h,
                            'landmarks': crop_lm
                        })
            except Exception as e:
                logger.warning(f"MTCNN detection error: {e}")

        # 2. MediaPipe Detection
        elif self.backend == "MediaPipe" and self.mp_detector is not None:
            try:
                import mediapipe as mp
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.mp_detector.detect(mp_image)

                fh, fw = frame.shape[:2]
                if detection_result.detections:
                    for detection in detection_result.detections:
                        score = float(detection.categories[0].score) if detection.categories else 0.8
                        if score < self.min_confidence:
                            continue

                        bbox = detection.bounding_box
                        x = int(bbox.origin_x)
                        y = int(bbox.origin_y)
                        w = int(bbox.width)
                        h = int(bbox.height)

                        # Extract landmarks if available
                        landmarks = None
                        if detection.keypoints:
                            landmarks = np.array([[kp.x * fw, kp.y * fh] for kp in detection.keypoints[:5]])

                        face_crop, padded_coords = align_and_crop_face(
                            frame, (x, y, w, h), landmarks=landmarks, padding_ratio=padding_ratio
                        )
                        quality_info = compute_face_quality(face_crop)
                        crop_lm = landmarks - np.array([padded_coords[0], padded_coords[1]], dtype=np.float32) if landmarks is not None else None

                        results.append({
                            'image': face_crop,
                            'coords': padded_coords,
                            'raw_box': (x, y, w, h),
                            'detector_confidence': score,
                            'quality': quality_info['score'],
                            'is_valid_quality': quality_info['is_valid'],
                            'quality_details': quality_info,
                            'area': w * h,
                            'landmarks': crop_lm
                        })
            except Exception as e:
                logger.warning(f"MediaPipe detection error: {e}")

        # 3. Haar Cascade Fallback
        if not results and self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected = self.haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=4, minSize=(36, 36)
                )
                for (x, y, w, h) in detected:
                    face_crop, padded_coords = align_and_crop_face(
                        frame, (x, y, w, h), landmarks=None, padding_ratio=padding_ratio
                    )
                    quality_info = compute_face_quality(face_crop)

                    results.append({
                        'image': face_crop,
                        'coords': padded_coords,
                        'raw_box': (x, y, w, h),
                        'detector_confidence': 0.75,
                        'quality': quality_info['score'],
                        'is_valid_quality': quality_info['is_valid'],
                        'quality_details': quality_info,
                        'area': w * h,
                        'landmarks': None
                    })
            except Exception as e:
                logger.warning(f"Haar detection error: {e}")

        # Sort by validity, area, and quality
        results.sort(
            key=lambda item: (item['is_valid_quality'], item['area'], item['quality']),
            reverse=True
        )
        return results[:max_faces]


# Global singleton detector instance
_GLOBAL_DETECTOR: Optional[RobustFaceDetector] = None

def get_detector() -> RobustFaceDetector:
    """Get or create singleton detector instance"""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = RobustFaceDetector(preferred_backend="auto")
    return _GLOBAL_DETECTOR


# ==================== CLASSIFICATION ====================

def predict_deepfake(
    face_image: Union[np.ndarray, Image.Image],
    processor: AutoImageProcessor,
    model: AutoModelForImageClassification,
    device: str = "cpu"
) -> Tuple[Optional[str], float, float, float]:
    """
    Run deepfake ViT classification on face crop.

    Returns:
        label: "Realism" or "Deepfake"
        confidence: Max probability
        p_real: Soft probability of Realism
        p_fake: Soft probability of Deepfake
    """
    try:
        if isinstance(face_image, np.ndarray):
            if len(face_image.shape) == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(face_rgb)
        else:
            pil_image = face_image

        inputs = processor(images=pil_image, return_tensors="pt")
        if device != "cpu" and torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            predicted_idx = int(logits.argmax(-1).item())

        label = model.config.id2label[predicted_idx]
        confidence = float(probs[predicted_idx].item())

        # Determine real / fake probability based on model config id2label
        # Usually 0: 'Realism', 1: 'Deepfake'
        p_real = 0.0
        p_fake = 0.0
        for idx, lbl in model.config.id2label.items():
            idx_int = int(idx)
            if idx_int < len(probs):
                if "real" in lbl.lower():
                    p_real = float(probs[idx_int].item())
                else:
                    p_fake = float(probs[idx_int].item())

        if p_real == 0.0 and p_fake == 0.0:
            if label == "Realism":
                p_real = confidence
                p_fake = 1.0 - confidence
            else:
                p_fake = confidence
                p_real = 1.0 - confidence

        return label, confidence, p_real, p_fake

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, 0.0, 0.0, 0.0


# Global fallback detector singleton
_GLOBAL_FALLBACK_DETECTOR: Optional[CustomFallbackDetector] = None

def get_fallback_detector(config: Optional[FallbackConfig] = None) -> CustomFallbackDetector:
    """Get or create singleton CustomFallbackDetector instance"""
    global _GLOBAL_FALLBACK_DETECTOR
    if _GLOBAL_FALLBACK_DETECTOR is None or config is not None:
        _GLOBAL_FALLBACK_DETECTOR = CustomFallbackDetector(config=config)
    return _GLOBAL_FALLBACK_DETECTOR


def predict_deepfake_with_fallback(
    face_image: Union[np.ndarray, Image.Image],
    processor: Optional[AutoImageProcessor] = None,
    model: Optional[AutoModelForImageClassification] = None,
    device: str = "cpu",
    quality_score: float = 50.0,
    fallback_detector: Optional[CustomFallbackDetector] = None,
    landmarks: Optional[np.ndarray] = None,
    frame_interval: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified prediction interface.

    When ENABLE_QMC = False (default / active application mode):
        - Runs ViT inference exclusively.
        - QMC-FD (CustomFallbackDetector) is NOT called.
        - Returns a result dict compatible with the existing consumer code.
        - decision_mode is always 'PRIMARY_VIT'.

    When ENABLE_QMC = True (experimental / frozen mode):
        - Activates the Quality-Aware Multi-Cue Fallback Detection mechanism
          with PRIMARY_VIT / FALLBACK_FUSION / FALLBACK_ONLY routing.
        - This path is preserved for future research but disabled by default.
    """
    # When ENABLE_QMC is disabled (default), QMC never participates in normal inference.
    # Exception: unit tests targeting fallback specifically that pass processor=None, model=None, and fallback_detector.
    qmc_active = ENABLE_QMC or (processor is None and model is None and fallback_detector is not None)
    if not qmc_active:
        # ── ACTIVE PATH: Pure ViT classification ──────────────────────────
        # Run ViT inference; return a standard result dict.
        # QMC components are intentionally not called here.
        vit_label, vit_conf_max, p_real, p_fake = predict_deepfake(
            face_image, processor, model, device=device
        )
        
        if vit_label is None:
            # Graceful degradation if ViT itself fails
            return {
                'decision_mode': DecisionMode.PRIMARY_VIT.value,
                'final_score': 0.5,
                'final_label': 'Unknown',
                'final_confidence': 0.0,
                'p_real': 0.5,
                'p_fake': 0.5,
                'vit_confidence': 0.0,
                's_vit': None,
                's_spatial': 0.0,
                's_temporal': 0.0,
                's_structural': 0.0,
                'temporal_available': False,
                'weights': {'w_vit': 1.0, 'w_spatial': 0.0, 'w_temporal': 0.0, 'w_structural': 0.0},
                'quality': round(quality_score, 2),
                'temporal_details': {},
                'structural_details': {}
            }
        vit_confidence = float(abs(p_fake - 0.5) * 2.0)  # [0, 1]
        return {
            'decision_mode': DecisionMode.PRIMARY_VIT.value,
            'final_score': round(p_fake, 4),
            'final_label': vit_label,
            'final_confidence': round(vit_confidence, 4),
            'p_real': round(p_real, 4),
            'p_fake': round(p_fake, 4),
            'vit_confidence': round(vit_confidence, 4),
            's_vit': round(p_fake, 4),
            's_spatial': 0.0,
            's_temporal': 0.0,
            's_structural': 0.0,
            'temporal_available': False,
            'weights': {'w_vit': 1.0, 'w_spatial': 0.0, 'w_temporal': 0.0, 'w_structural': 0.0},
            'quality': round(quality_score, 2),
            'temporal_details': {},
            'structural_details': {}
        }

    # ── EXPERIMENTAL PATH: QMC-FD (ENABLE_QMC = True only) ───────────────
    # This section is frozen and not used during normal application execution.
    detector = fallback_detector or get_fallback_detector()

    # Convert to numpy array for CV cues if input is PIL
    if isinstance(face_image, Image.Image):
        face_np = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    else:
        face_np = face_image

    # Attempt ViT inference
    vit_result = None
    if processor is not None and model is not None:
        try:
            vit_result = predict_deepfake(face_image, processor, model, device=device)
        except Exception as e:
            logger.warning(f"Primary ViT inference failed: {e}. Falling back to QMC-FD cues.")
            vit_result = None

    # Evaluate frame with QMC-FD
    frame_eval = detector.evaluate_frame(
        face_img=face_np,
        vit_result=vit_result,
        quality_score=quality_score,
        landmarks=landmarks,
        frame_interval=frame_interval
    )

    return frame_eval


# ==================== TEMPORAL AGGREGATION ====================

class TemporalAggregator:
    """
    Aggregates multi-frame face predictions using weighted soft probabilities,
    sample quality weights, detector confidence, and localized anomaly detection.
    Also tracks QMC-FD fallback activation telemetry.
    """

    @staticmethod
    def aggregate(
        predictions: List[Dict[str, Any]],
        confidence_threshold: float = 0.50,
        decision_threshold: float = 0.40,
        enable_anomaly_detection: bool = True,
        top_k_ratio: float = 0.20
    ) -> Dict[str, Any]:
        """
        Aggregate video frame predictions into a video-level verdict.

        Args:
            predictions: List of dicts containing frame predictions
            confidence_threshold: Frame-level certainty cutoff for discrete counting
            decision_threshold: Video-level weighted fake probability cutoff for deepfake verdict (calibrated: 0.40)
            enable_anomaly_detection: Whether to enable temporal burst anomaly detection
            top_k_ratio: Fraction of top most suspicious frames to evaluate for localized manipulation pooling (default: 0.20)
        """
        if not predictions:
            return {
                'final_label': "Unknown",
                'avg_confidence': 0.0,
                'real_count': 0,
                'fake_count': 0,
                'total_predictions': 0,
                'weighted_real_score': 0.0,
                'weighted_fake_score': 0.0,
                'is_deepfake': False,
                'anomaly_detected': False,
                'verdict_summary': "No valid faces were detected in the video.",
                'video_fake_probability': 0.0,
                'video_real_probability': 0.0,
                'frames_processed': 0,
                'frames_used': 0,
                'fallback_frames': 0,
                'decision_threshold': decision_threshold,
                'confidence_threshold': confidence_threshold,
                'mode_counts': {
                    DecisionMode.PRIMARY_VIT.value: 0,
                    DecisionMode.FALLBACK_FUSION.value: 0,
                    DecisionMode.FALLBACK_ONLY.value: 0
                },
                'average_quality': 0.0,
                'average_confidence': 0.0,
                'average_vit_confidence': 0.0
            }

        total_preds = len(predictions)
        real_count = 0
        fake_count = 0

        weighted_fake_sum = 0.0
        weighted_real_sum = 0.0
        total_weight = 0.0

        consecutive_fake_window = 0
        max_consecutive_fake = 0

        fallback_frames = 0
        mode_counts = {
            DecisionMode.PRIMARY_VIT.value: 0,
            DecisionMode.FALLBACK_FUSION.value: 0,
            DecisionMode.FALLBACK_ONLY.value: 0
        }
        total_quality_sum = 0.0
        total_vit_conf_sum = 0.0
        vit_conf_count = 0

        for pred in predictions:
            lbl = pred.get('label', '')
            conf = pred.get('confidence', 0.5)
            p_fake = pred.get('p_fake', 1.0 - conf if lbl == 'Realism' else conf)
            p_real = pred.get('p_real', conf if lbl == 'Realism' else 1.0 - conf)
            quality = pred.get('quality', 50.0)

            # Track QMC-FD telemetry
            mode = pred.get('decision_mode', DecisionMode.PRIMARY_VIT.value)
            if mode in mode_counts:
                mode_counts[mode] += 1
            if mode in (DecisionMode.FALLBACK_FUSION.value, DecisionMode.FALLBACK_ONLY.value) or pred.get('is_fallback', False):
                fallback_frames += 1

            total_quality_sum += quality
            if 'vit_confidence' in pred and pred['vit_confidence'] is not None:
                total_vit_conf_sum += pred['vit_confidence']
                vit_conf_count += 1

            # Frame counts using threshold
            if p_fake >= confidence_threshold and p_fake > p_real:
                fake_count += 1
                consecutive_fake_window += 1
                max_consecutive_fake = max(max_consecutive_fake, consecutive_fake_window)
            else:
                consecutive_fake_window = 0
                if p_real >= confidence_threshold:
                    real_count += 1

            # Weight calculation: quality * certainty
            certainty = abs(p_fake - 0.5) * 2.0  # [0.0 - 1.0]
            weight = max(0.1, (quality / 100.0) * (0.5 + 0.5 * certainty))

            weighted_fake_sum += p_fake * weight
            weighted_real_sum += p_real * weight
            total_weight += weight

        if total_weight > 0:
            weighted_fake_score = weighted_fake_sum / total_weight
            weighted_real_score = weighted_real_sum / total_weight
        else:
            weighted_fake_score = 0.0
            weighted_real_score = 0.0

        # Top-K suspicious frame pooling: focus on top most manipulated frames
        k = max(1, int(math.ceil(total_preds * top_k_ratio)))
        top_k_preds = sorted(predictions, key=lambda p: p.get('p_fake', 0.0), reverse=True)[:k]
        top_k_fake_sum = 0.0
        top_k_weight_sum = 0.0
        for p in top_k_preds:
            w = max(0.1, (p.get('quality', 50.0) / 100.0) * (0.5 + 0.5 * abs(p.get('p_fake', 0.5) - 0.5) * 2.0))
            top_k_fake_sum += p.get('p_fake', 0.0) * w
            top_k_weight_sum += w
        top_k_fake_score = (top_k_fake_sum / top_k_weight_sum) if top_k_weight_sum > 0 else 0.0

        # Anomaly detection: if a burst of consecutive frames shows strong deepfake evidence
        anomaly_detected = False
        if enable_anomaly_detection:
            anomaly_detected = (max_consecutive_fake >= 3 and (max_consecutive_fake / max(1, total_preds)) >= 0.15)

        # Decision rules:
        # 1. Weighted fake score >= decision_threshold -> Deepfake
        # 2. Top-K suspicious frames show persistent manipulation (top_k_fake_score >= 0.50 and fake_count >= max(4, int(total_preds * 0.30))) -> Deepfake
        # 3. Burst anomaly detected with fake_count > 2 -> Deepfake
        # 4. Otherwise -> Realism
        min_top_k_fake_frames = max(4, int(total_preds * 0.30))
        top_k_trigger = (top_k_fake_score >= 0.50 and fake_count >= min_top_k_fake_frames)
        is_deepfake = (weighted_fake_score >= decision_threshold) or top_k_trigger or (anomaly_detected and fake_count > 2)

        if is_deepfake:
            final_label = "Deepfake"
            avg_confidence = max(weighted_fake_score, top_k_fake_score if top_k_trigger else weighted_fake_score)
            verdict_summary = f"Deepfake detected with {avg_confidence*100:.1f}% weighted confidence ({fake_count}/{total_preds} fake frames)."
        else:
            final_label = "Realism"
            avg_confidence = weighted_real_score
            verdict_summary = f"Authentic video confirmed with {avg_confidence*100:.1f}% confidence ({real_count}/{total_preds} authentic frames)."

        avg_quality = (total_quality_sum / total_preds) if total_preds > 0 else 0.0
        avg_vit_conf = (total_vit_conf_sum / vit_conf_count) if vit_conf_count > 0 else 0.0

        return {
            'final_label': final_label,
            'avg_confidence': avg_confidence,
            'real_count': real_count,
            'fake_count': fake_count,
            'total_predictions': total_preds,
            'weighted_real_score': weighted_real_score,
            'weighted_fake_score': weighted_fake_score,
            'top_k_fake_score': round(top_k_fake_score, 4),
            'top_k_ratio': top_k_ratio,
            'decision_threshold': decision_threshold,
            'confidence_threshold': confidence_threshold,
            'is_deepfake': is_deepfake,
            'anomaly_detected': anomaly_detected,
            'verdict_summary': verdict_summary,
            # QMC-FD Extended Telemetry
            'video_fake_probability': weighted_fake_score,
            'video_real_probability': weighted_real_score,
            'frames_processed': total_preds,
            'frames_used': total_preds,
            'fallback_frames': fallback_frames,
            'mode_counts': mode_counts,
            'average_quality': round(avg_quality, 2),
            'average_confidence': round(avg_confidence, 4),
            'average_vit_confidence': round(avg_vit_conf, 4)
        }


def aggregate_video_predictions(
    predictions: List[Dict[str, Any]]
) -> Tuple[str, float, int, int]:
    """
    Backward-compatible function returning (final_label, avg_confidence, real_count, fake_count).
    """
    res = TemporalAggregator.aggregate(predictions)
    return res['final_label'], res['avg_confidence'], res['real_count'], res['fake_count']


# ==================== VIDEO PIPELINE HELPER ====================

def sample_video_frame_indices(
    total_frames: int,
    fps: float,
    max_samples: int = 60,
    target_sample_fps: float = 3.0
) -> List[int]:
    """
    Calculate uniform frame indices to sample across the video duration.
    Ensures broad temporal coverage from beginning to end.
    """
    if total_frames <= 0:
        return []

    fps = max(1.0, fps)
    step = max(1, int(round(fps / target_sample_fps)))
    sampled_indices = list(range(0, total_frames, step))

    # If too many frames, downsample uniformly to max_samples
    if len(sampled_indices) > max_samples:
        stride = len(sampled_indices) / float(max_samples)
        sampled_indices = [sampled_indices[int(i * stride)] for i in range(max_samples)]

    return sampled_indices
