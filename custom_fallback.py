"""
Quality-Aware Multi-Cue Fallback Detection (QMC-FD) Module
===========================================================
This module provides a lightweight, deterministic fallback and decision-fusion
mechanism for deepfake detection around a primary Vision Transformer (ViT) model.

Key Components:
- SpatialArtifactAnalyzer: High-frequency residual and boundary texture irregularity analysis.
- TemporalConsistencyAnalyzer: Motion-compensated inter-frame face difference analysis.
- StructuralAnalyzer: Edge field coherence, contour continuity, and gradient symmetry analysis.
- QualityAwareFusion: Reliability-aware dynamic weight fusion based on face quality score.
- CustomFallbackDetector: Controller managing the three decision modes:
    1. PRIMARY_VIT: High-confidence ViT prediction trusted directly.
    2. FALLBACK_FUSION: Uncertain ViT prediction fused with spatial, temporal, and structural cues.
    3. FALLBACK_ONLY: ViT unavailable/failed; decision derived from supporting cues.

NOTE: This is our custom/proposed decision mechanism designed for experimental evaluation,
inspired by multi-domain evidence principles without claiming full reproduction of external
heavy architectures (e.g., SEA-RAFT or multi-ResNet-50 networks).
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np


class DecisionMode(str, Enum):
    PRIMARY_VIT = "PRIMARY_VIT"
    FALLBACK_FUSION = "FALLBACK_FUSION"
    FALLBACK_ONLY = "FALLBACK_ONLY"


@dataclass
class FallbackConfig:
    """
    Configuration parameters for QMC-FD decision fusion and cues.
    All parameters are configurable to prevent hardcoded magic numbers.
    """
    # Confidence threshold for trusting ViT alone: confidence = abs(p_fake - 0.5) * 2
    vit_confidence_threshold: float = 0.70

    # Base weights for fusion when all cues are available
    base_weight_vit: float = 0.40
    base_weight_spatial: float = 0.20
    base_weight_temporal: float = 0.20
    base_weight_structural: float = 0.20

    # Minimum base floor for cue reliability (prevents low quality from completely zeroing weights)
    quality_weight_floor: float = 0.25

    # Spatial cue parameters
    spatial_blur_kernel: int = 5
    spatial_blur_sigma: float = 1.0
    spatial_high_freq_scale: float = 45.0  # Normalization denominator

    # Temporal cue parameters
    temporal_canonical_size: Tuple[int, int] = (128, 128)
    temporal_max_residual_threshold: float = 28.0  # Base expected mean difference after canonical alignment

    # Structural cue parameters
    structural_canonical_size: Tuple[int, int] = (128, 128)
    structural_gradient_scale: float = 65.0  # Normalization denominator for gradient irregularity


# Standard 5-point facial landmarks scaled to canonical 128x128 frontal geometry
CANONICAL_FACE_5PTS = np.array([
    [38.2946, 51.6963],  # Left eye
    [89.7054, 51.6963],  # Right eye (symmetric across x=64)
    [64.0000, 71.7366],  # Nose tip (centered on x=64)
    [43.8340, 92.3655],  # Left mouth corner
    [84.1660, 92.3655]   # Right mouth corner (symmetric across x=64)
], dtype=np.float32)


# ==================== SPATIAL ARTIFACT ANALYZER ====================

class SpatialArtifactAnalyzer:
    """
    Lightweight spatial artifact analyzer.
    Examines high-frequency energy anomalies, local texture irregularities,
    and blending boundary contrast on aligned face crops.

    IMPORTANT: Blur does NOT count as deepfake evidence. Natural blur is handled
    by the face quality gate, which reduces overall cue reliability rather than
    increasing fake likelihood.
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()

    def analyze(self, face_img: np.ndarray, quality_score: float = 50.0) -> float:
        """
        Calculates S_spatial in [0.0, 1.0], where higher values indicate stronger
        evidence of generative or blending artifacts.

        Method:
        1. Extract high-frequency residual: I_high = |I_gray - GaussianBlur(I_gray)|.
        2. Evaluate boundary ring vs inner core energy ratio. Deepfake blending
           boundaries frequently exhibit anomalous high-frequency energy transitions
           at the blending seams.
        3. Evaluate Laplacian residual dispersion (variance of high frequencies).
        4. Normalize into [0.0, 1.0].
        """
        if face_img is None or face_img.size == 0:
            return 0.5

        try:
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img.copy()

            h, w = gray.shape[:2]
            if h < 20 or w < 20:
                return 0.5

            # Standardize analysis scale for deterministic response
            gray_resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

            # High-frequency residual extraction
            ksize = self.config.spatial_blur_kernel
            sigma = self.config.spatial_blur_sigma
            blurred = cv2.GaussianBlur(gray_resized, (ksize, ksize), sigma)
            high_freq = cv2.absdiff(gray_resized, blurred).astype(np.float32)

            # Define inner core vs boundary ring masks
            inner_mask = np.zeros((128, 128), dtype=np.uint8)
            cv2.ellipse(inner_mask, (64, 64), (38, 48), 0, 0, 360, 255, -1)
            outer_mask = cv2.bitwise_not(inner_mask)

            inner_mean = float(np.mean(high_freq[inner_mask > 0])) if np.any(inner_mask > 0) else 1.0
            outer_mean = float(np.mean(high_freq[outer_mask > 0])) if np.any(outer_mask > 0) else 1.0

            # Discrepancy ratio between outer boundary and inner face
            # Manipulated faces often have abnormal edge/blending boundaries
            boundary_discrepancy = abs(outer_mean - inner_mean) / (inner_mean + outer_mean + 1e-5)

            # High-frequency energy dispersion
            hf_std = float(np.std(high_freq))
            norm_dispersion = min(1.0, hf_std / self.config.spatial_high_freq_scale)

            # Combine: 60% boundary transition anomaly + 40% high-frequency dispersion
            raw_score = 0.60 * min(1.0, boundary_discrepancy * 2.5) + 0.40 * norm_dispersion

            # Clamp deterministically to [0.0, 1.0]
            s_spatial = float(np.clip(raw_score, 0.0, 1.0))
            return s_spatial

        except Exception:
            return 0.5


# ==================== TEMPORAL INCONSISTENCY ANALYZER ====================

class TemporalConsistencyAnalyzer:
    """
    Lightweight frame-to-frame temporal inconsistency analyzer.
    Analyzes motion-compensated residual differences between consecutive aligned face crops
    using canonical landmark alignment and interval-scaled thresholds to avoid misclassifying
    natural head motion as manipulation.
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.prev_canonical: Optional[np.ndarray] = None
        self.prev_quality: float = 50.0

    def reset(self) -> None:
        """Reset temporal state across video sequences or webcam cuts."""
        self.prev_canonical = None
        self.prev_quality = 50.0

    def align_to_canonical(
        self,
        gray: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        target_size: Tuple[int, int] = (128, 128)
    ) -> Tuple[np.ndarray, bool]:
        """Align face image to canonical 5-point geometry if landmarks are present."""
        if landmarks is not None and len(landmarks) >= 3:
            try:
                lm_pts = np.array(landmarks, dtype=np.float32)
                h_img, w_img = gray.shape[:2]
                if np.max(lm_pts[:, 0]) > w_img or np.max(lm_pts[:, 1]) > h_img:
                    lm_min = np.min(lm_pts, axis=0)
                    lm_pts = lm_pts - lm_min + np.array([w_img * 0.15, h_img * 0.15], dtype=np.float32)

                if lm_pts.shape == (5, 2):
                    M, _ = cv2.estimateAffinePartial2D(lm_pts, CANONICAL_FACE_5PTS)
                    if M is not None:
                        warped = cv2.warpAffine(gray, M, target_size, borderMode=cv2.BORDER_REFLECT)
                        return warped, True
            except Exception:
                pass
        return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA), False

    def analyze_with_details(
        self,
        current_face: np.ndarray,
        quality_score: float = 50.0,
        landmarks: Optional[np.ndarray] = None,
        frame_interval: int = 1
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculates S_temporal in [0.0, 1.0] and returns detailed intermediate telemetry.
        """
        if current_face is None or current_face.size == 0:
            return 0.5, False, {'status': 'empty_image'}

        try:
            if len(current_face.shape) == 3:
                gray = cv2.cvtColor(current_face, cv2.COLOR_BGR2GRAY)
            else:
                gray = current_face.copy()

            target_size = self.config.temporal_canonical_size
            curr_canon, is_aligned = self.align_to_canonical(gray, landmarks, target_size)

            if self.prev_canonical is None:
                self.prev_canonical = curr_canon
                self.prev_quality = quality_score
                return 0.5, False, {'status': 'first_frame', 'is_aligned': is_aligned}

            # Identical frame check
            if np.array_equal(curr_canon, self.prev_canonical):
                self.prev_canonical = curr_canon
                self.prev_quality = quality_score
                return 0.0, True, {'status': 'identical_frames', 's_temporal': 0.0}

            # Extract stable rigid anchor region (upper face: eyes, bridge of nose, forehead)
            # In authentic humans, this region remains structurally rigid during speech and blinks
            h, w = target_size
            r1 = self.prev_canonical[int(h * 0.20):int(h * 0.65), int(w * 0.20):int(w * 0.80)]
            r2 = curr_canon[int(h * 0.20):int(h * 0.65), int(w * 0.20):int(w * 0.80)]

            # Illumination normalization anchored to the rigid face (eliminates global lighting shifts without min-max contrast distortion)
            m1 = float(np.mean(r1))
            m2 = float(np.mean(r2))
            curr_adjusted = np.clip(curr_canon.astype(np.float32) - (m2 - m1), 0, 255).astype(np.uint8)
            r2_adj = curr_adjusted[int(h * 0.20):int(h * 0.65), int(w * 0.20):int(w * 0.80)]

            # Subpixel phase correlation anchored strictly to the rigid face to eliminate micro-jitter
            # without allowing lower-jaw speech articulation to corrupt whole-head translation
            shift, _ = cv2.phaseCorrelate(r1.astype(np.float32), r2_adj.astype(np.float32))
            dx, dy = shift
            if abs(dx) < 16.0 and abs(dy) < 16.0:
                m_trans = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_prev = cv2.warpAffine(
                    self.prev_canonical,
                    m_trans,
                    target_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
            else:
                aligned_prev = self.prev_canonical

            pixel_diff = cv2.absdiff(curr_adjusted, aligned_prev)

            # Separate rigid facial anchor region from dynamic articulation region (mouth / lower jaw)
            rigid_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(rigid_mask, (int(w * 0.50), int(h * 0.47)), (int(w * 0.25), int(h * 0.18)), 0, 0, 360, 255, -1)

            articul_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(articul_mask, (int(w * 0.50), int(h * 0.74)), (int(w * 0.19), int(h * 0.12)), 0, 0, 360, 255, -1)

            diff_rigid = float(np.mean(pixel_diff[rigid_mask > 0])) if np.any(rigid_mask > 0) else float(np.mean(pixel_diff))
            diff_articul = float(np.mean(pixel_diff[articul_mask > 0])) if np.any(articul_mask > 0) else diff_rigid

            # Natural articulation tolerance: when the rigid facial anchor is stable,
            # natural mouth movement during speech is bounded rather than treated as a deepfake anomaly
            articul_tolerated = min(diff_articul, diff_rigid * 2.5 + 8.0)
            effective_diff = 0.70 * diff_rigid + 0.30 * articul_tolerated

            # Edge discrepancy using dilated boundary edges to prevent single-pixel alignment jitter from penalizing genuine motion
            canny_curr = cv2.Canny(curr_adjusted, 50, 150)
            canny_prev = cv2.Canny(aligned_prev, 50, 150)
            dilated_curr = cv2.dilate(canny_curr, np.ones((3, 3), np.uint8))
            dilated_prev = cv2.dilate(canny_prev, np.ones((3, 3), np.uint8))
            unmatched = np.logical_or(
                np.logical_and(canny_curr > 0, dilated_prev == 0),
                np.logical_and(canny_prev > 0, dilated_curr == 0)
            )
            edge_diff = float(np.mean(unmatched[rigid_mask > 0])) if np.any(rigid_mask > 0) else float(np.mean(unmatched))

            # Scale threshold dynamically with frame interval:
            # 1-frame (42ms) -> 28.0; 4-frame (167ms) -> 35.5; 8-frame (333ms) -> 45.5
            dynamic_threshold = float(self.config.temporal_max_residual_threshold + max(0, frame_interval - 1) * 2.5)
            norm_diff = float(np.clip(effective_diff / dynamic_threshold, 0.0, 1.0))
            raw_score = 0.75 * norm_diff + 0.25 * float(np.clip(edge_diff * 6.0, 0.0, 1.0))

            s_temporal = float(np.clip(raw_score, 0.0, 1.0))

            self.prev_canonical = curr_canon
            self.prev_quality = quality_score

            telemetry = {
                'is_aligned': is_aligned,
                'diff_rigid': round(diff_rigid, 4),
                'diff_articul': round(diff_articul, 4),
                'effective_diff': round(effective_diff, 4),
                'dynamic_threshold': round(dynamic_threshold, 2),
                'norm_diff': round(norm_diff, 4),
                'edge_diff': round(edge_diff, 4),
                'raw_score': round(raw_score, 4),
                's_temporal': round(s_temporal, 4),
                'frame_interval': frame_interval
            }
            return s_temporal, True, telemetry

        except Exception as e:
            return 0.5, False, {'status': 'error', 'error': str(e)}

    def analyze(
        self,
        current_face: np.ndarray,
        quality_score: float = 50.0,
        landmarks: Optional[np.ndarray] = None,
        frame_interval: int = 1
    ) -> Tuple[float, bool]:
        score, avail, _ = self.analyze_with_details(
            current_face, quality_score, landmarks=landmarks, frame_interval=frame_interval
        )
        return score, avail


# ==================== STRUCTURAL ANALYZER ====================

class StructuralAnalyzer:
    """
    Lightweight structural analyzer.
    Evaluates facial structural integrity, edge field coherence, and gradient symmetry
    using landmark-based canonical alignment and circular statistics.
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()

    def align_to_canonical(
        self,
        gray: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        target_size: Tuple[int, int] = (128, 128)
    ) -> Tuple[np.ndarray, bool]:
        """Align face image to canonical 5-point geometry if landmarks are present."""
        if landmarks is not None and len(landmarks) >= 3:
            try:
                lm_pts = np.array(landmarks, dtype=np.float32)
                h_img, w_img = gray.shape[:2]
                if np.max(lm_pts[:, 0]) > w_img or np.max(lm_pts[:, 1]) > h_img:
                    lm_min = np.min(lm_pts, axis=0)
                    lm_pts = lm_pts - lm_min + np.array([w_img * 0.15, h_img * 0.15], dtype=np.float32)

                if lm_pts.shape == (5, 2):
                    M, _ = cv2.estimateAffinePartial2D(lm_pts, CANONICAL_FACE_5PTS)
                    if M is not None:
                        warped = cv2.warpAffine(gray, M, target_size, borderMode=cv2.BORDER_REFLECT)
                        return warped, True
            except Exception:
                pass
        return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA), False

    def analyze_with_details(
        self,
        face_img: np.ndarray,
        quality_score: float = 50.0,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates S_structural in [0.0, 1.0] and returns detailed intermediate telemetry.
        Higher values represent stronger evidence of structural manipulation/irregularity.
        """
        if face_img is None or face_img.size == 0:
            return 0.5, {'status': 'empty_image'}

        try:
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img.copy()

            if np.max(gray) == 0:
                return 0.15, {'status': 'blank_image', 'raw_score': 0.15, 's_structural': 0.15}

            target_size = self.config.structural_canonical_size
            warped, is_aligned = self.align_to_canonical(gray, landmarks, target_size)

            # 1. Bilateral facial symmetry on canonical inner face
            h, w = target_size
            mask = np.zeros(target_size, dtype=np.uint8)
            cv2.ellipse(mask, (64, 68), (32, 44), 0, 0, 360, 255, -1)

            half_w = w // 2
            left_mask = mask[:, :half_w]
            right_mask = cv2.flip(mask[:, half_w:], 1)
            joint_mask = cv2.bitwise_and(left_mask, right_mask)

            left_half = warped[:, :half_w]
            right_mirrored = cv2.flip(warped[:, half_w:], 1)

            diff = np.abs(left_half.astype(float) - right_mirrored.astype(float))
            mean_diff = float(np.mean(diff[joint_mask > 0])) if np.any(joint_mask > 0) else 0.0

            # Normalized asymmetry: natural face difference is 10-22, anomalies > 35
            norm_asymmetry = float(np.clip(mean_diff / 50.0, 0.0, 1.0))

            # 2. Circular-statistics orientation entropy & coherence
            grad_x = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(grad_x**2 + grad_y**2)
            angle_rad = np.arctan2(grad_y, grad_x)

            inner_mag = mag[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
            inner_ang = angle_rad[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]

            total_w = float(np.sum(inner_mag)) + 1e-5
            c2 = float(np.sum(inner_mag * np.cos(2.0 * inner_ang))) / total_w
            s2 = float(np.sum(inner_mag * np.sin(2.0 * inner_ang))) / total_w
            R2 = float(np.sqrt(c2**2 + s2**2))
            circ_var = float(np.clip(1.0 - R2, 0.0, 1.0))

            # Local gradient orientation coherence in 8x8 blocks
            gxx = grad_x**2
            gyy = grad_y**2
            gxy = grad_x * grad_y
            kernel = (8, 8)
            Jxx = cv2.boxFilter(gxx, -1, kernel)
            Jyy = cv2.boxFilter(gyy, -1, kernel)
            Jxy = cv2.boxFilter(gxy, -1, kernel)
            denom = Jxx + Jyy + 1e-5
            coh = np.sqrt((Jxx - Jyy)**2 + 4.0 * (Jxy**2)) / denom
            inner_coh = coh[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
            mean_coh = float(np.sum(inner_coh * inner_mag) / total_w)
            local_incoherence = float(np.clip(1.0 - mean_coh, 0.0, 1.0))

            circular_dispersion = float(0.5 * circ_var + 0.5 * local_incoherence)

            # 3. Contour continuity & edge anomaly
            edges = cv2.Canny(warped, 40, 120)
            edge_density = float(np.count_nonzero(edges)) / float(h * w)

            if edge_density > 0.18:
                edge_anomaly = float(np.clip((edge_density - 0.18) * 8.0, 0.0, 1.0))
            elif edge_density < 0.02:
                edge_anomaly = float(np.clip((0.02 - edge_density) * 30.0, 0.0, 1.0))
            else:
                edge_anomaly = 0.0

            raw_score = 0.40 * norm_asymmetry + 0.35 * circular_dispersion + 0.25 * edge_anomaly
            s_structural = float(np.clip(raw_score, 0.0, 1.0))

            telemetry = {
                'is_aligned': is_aligned,
                'mean_diff': round(mean_diff, 4),
                'norm_asymmetry': round(norm_asymmetry, 4),
                'circ_var': round(circ_var, 4),
                'local_incoherence': round(local_incoherence, 4),
                'circular_dispersion': round(circular_dispersion, 4),
                'edge_density': round(edge_density, 4),
                'edge_anomaly': round(edge_anomaly, 4),
                'raw_score': round(raw_score, 4),
                's_structural': round(s_structural, 4)
            }
            return s_structural, telemetry

        except Exception as e:
            return 0.5, {'status': 'error', 'error': str(e)}

    def analyze(
        self,
        face_img: np.ndarray,
        quality_score: float = 50.0,
        landmarks: Optional[np.ndarray] = None
    ) -> float:
        score, _ = self.analyze_with_details(face_img, quality_score, landmarks=landmarks)
        return score


# ==================== QUALITY-AWARE DECISION FUSION ====================

class QualityAwareFusion:
    """
    Fuses primary ViT soft probability with spatial, temporal, and structural cues
    using reliability-aware dynamic weighting.
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()

    def fuse(
        self,
        s_vit: Optional[float],
        vit_confidence: Optional[float],
        s_spatial: float,
        s_temporal: float,
        s_structural: float,
        quality_score: float,
        temporal_available: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Fuses cues into S_final in [0.0, 1.0].

        Formula:
        1. Normalized quality reliability: Q = clamp(quality_score / 100.0, 0, 1)
        2. Reliability for supporting cues: R_cue = floor + (1 - floor) * Q
        3. Reliability for ViT: R_vit = vit_confidence (distance from uncertain 0.5 midpoint)
        4. Base unnormalized weights:
             w_vit = a_vit * W_vit_base * R_vit
             w_spatial = W_spatial_base * R_cue
             w_temporal = a_temporal * W_temporal_base * R_cue
             w_structural = W_structural_base * R_cue
        5. Normalize weights: w_i' = w_i / sum(w)
        6. Fused Score: S_final = sum(w_i' * S_i)

        Returns:
            (s_final, normalized_weights_dict)
        """
        q = float(np.clip(quality_score / 100.0, 0.0, 1.0))
        r_cue = self.config.quality_weight_floor + (1.0 - self.config.quality_weight_floor) * q

        # Availability flags
        a_vit = 1.0 if s_vit is not None else 0.0
        a_temp = 1.0 if temporal_available else 0.0

        r_vit = float(np.clip(vit_confidence, 0.0, 1.0)) if vit_confidence is not None else 0.5

        # Raw weights
        w_vit = a_vit * self.config.base_weight_vit * (0.3 + 0.7 * r_vit)
        w_spatial = self.config.base_weight_spatial * r_cue
        w_temporal = a_temp * self.config.base_weight_temporal * r_cue
        w_structural = self.config.base_weight_structural * r_cue

        total_weight = w_vit + w_spatial + w_temporal + w_structural

        if total_weight <= 1e-6:
            # Fallback uniform weights
            w_norm_vit = 0.25 if a_vit > 0 else 0.0
            w_norm_temp = 0.25 if a_temp > 0 else 0.0
            remaining = 1.0 - (w_norm_vit + w_norm_temp)
            w_norm_spatial = remaining / 2.0
            w_norm_structural = remaining / 2.0
        else:
            w_norm_vit = w_vit / total_weight
            w_norm_spatial = w_spatial / total_weight
            w_norm_temporal = w_temporal / total_weight
            w_norm_structural = w_structural / total_weight

        # Compute fused score
        val_vit = s_vit if s_vit is not None else 0.5
        val_spatial = float(np.clip(s_spatial, 0.0, 1.0))
        val_temporal = float(np.clip(s_temporal, 0.0, 1.0))
        val_structural = float(np.clip(s_structural, 0.0, 1.0))

        s_final = (
            w_norm_vit * val_vit +
            w_norm_spatial * val_spatial +
            w_norm_temporal * val_temporal +
            w_norm_structural * val_structural
        )
        s_final = float(np.clip(s_final, 0.0, 1.0))

        weights_dict = {
            'w_vit': round(w_norm_vit, 4),
            'w_spatial': round(w_norm_spatial, 4),
            'w_temporal': round(w_norm_temporal, 4),
            'w_structural': round(w_norm_structural, 4)
        }

        return s_final, weights_dict


# ==================== MAIN CUSTOM FALLBACK DETECTOR ====================

class CustomFallbackDetector:
    """
    Main controller for Quality-Aware Multi-Cue Fallback Detection (QMC-FD).
    Orchestrates the 3 operating modes:
    - MODE 1: PRIMARY_VIT (ViT confidence >= threshold)
    - MODE 2: FALLBACK_FUSION (ViT uncertain; fuses ViT + Spatial + Temporal + Structural)
    - MODE 3: FALLBACK_ONLY (ViT failed / unavailable; fuses supporting cues)
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.spatial_analyzer = SpatialArtifactAnalyzer(self.config)
        self.temporal_analyzer = TemporalConsistencyAnalyzer(self.config)
        self.structural_analyzer = StructuralAnalyzer(self.config)
        self.fusion = QualityAwareFusion(self.config)

    def reset_temporal_state(self) -> None:
        """Reset temporal analyzer state (e.g. on new video stream)."""
        self.temporal_analyzer.reset()

    @staticmethod
    def calculate_vit_confidence(p_fake: float) -> float:
        """
        Confidence is the certainty distance from the neutral midpoint (0.50).
        confidence = abs(p_fake - 0.5) * 2.0 in [0.0, 1.0].
        """
        return float(np.clip(abs(p_fake - 0.5) * 2.0, 0.0, 1.0))

    def evaluate_frame(
        self,
        face_img: np.ndarray,
        vit_result: Optional[Tuple[Optional[str], float, float, float]] = None,
        quality_score: float = 50.0,
        landmarks: Optional[np.ndarray] = None,
        frame_interval: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate an aligned face crop with ViT prediction and fallback logic.

        Args:
            face_img: BGR face crop numpy array
            vit_result: Tuple of (label, max_conf, p_real, p_fake) from ViT or None if failed
            quality_score: Face quality score [0.0 - 100.0]
            landmarks: Optional 5 facial landmark coordinates for canonical alignment
            frame_interval: Number of video frames since previous sample (1, 4, 8, etc.)

        Returns:
            Dictionary containing prediction scores, modes, weights, and intermediate details.
        """
        s_vit = None
        vit_conf = 0.0
        p_real_vit = 0.5
        p_fake_vit = 0.5
        vit_available = False

        if vit_result is not None:
            v_label, v_conf, v_preal, v_pfake = vit_result
            if v_label is not None and not math.isnan(v_pfake):
                s_vit = float(v_pfake)
                p_real_vit = float(v_preal)
                p_fake_vit = float(v_pfake)
                vit_conf = self.calculate_vit_confidence(s_vit)
                vit_available = True

        struc_details = {}
        temp_details = {}

        # Determine decision mode
        if vit_available and vit_conf >= self.config.vit_confidence_threshold:
            # MODE 1: PRIMARY_VIT (ViT is confident)
            mode = DecisionMode.PRIMARY_VIT
            final_score = s_vit
            p_fake = s_vit
            p_real = p_real_vit
            final_label = "Deepfake" if s_vit >= 0.50 else "Realism"
            final_confidence = vit_conf

            # Keep temporal state updated even in primary mode for smooth inter-frame continuity
            s_temporal, temp_avail, temp_details = self.temporal_analyzer.analyze_with_details(
                face_img, quality_score, landmarks=landmarks, frame_interval=frame_interval
            )
            s_spatial = 0.0  # Lazy/bypassed in confident ViT mode to save computational resources
            s_structural = 0.0
            weights = {'w_vit': 1.0, 'w_spatial': 0.0, 'w_temporal': 0.0, 'w_structural': 0.0}

        elif vit_available:
            # MODE 2: FALLBACK_FUSION (ViT is uncertain)
            mode = DecisionMode.FALLBACK_FUSION
            s_spatial = self.spatial_analyzer.analyze(face_img, quality_score)
            s_temporal, temp_avail, temp_details = self.temporal_analyzer.analyze_with_details(
                face_img, quality_score, landmarks=landmarks, frame_interval=frame_interval
            )
            s_structural, struc_details = self.structural_analyzer.analyze_with_details(
                face_img, quality_score, landmarks=landmarks
            )

            final_score, weights = self.fusion.fuse(
                s_vit=s_vit,
                vit_confidence=vit_conf,
                s_spatial=s_spatial,
                s_temporal=s_temporal,
                s_structural=s_structural,
                quality_score=quality_score,
                temporal_available=temp_avail
            )

            p_fake = final_score
            p_real = 1.0 - final_score
            final_label = "Deepfake" if final_score >= 0.50 else "Realism"
            final_confidence = abs(final_score - 0.5) * 2.0

        else:
            # MODE 3: FALLBACK_ONLY (ViT is unavailable or failed)
            mode = DecisionMode.FALLBACK_ONLY
            s_spatial = self.spatial_analyzer.analyze(face_img, quality_score)
            s_temporal, temp_avail, temp_details = self.temporal_analyzer.analyze_with_details(
                face_img, quality_score, landmarks=landmarks, frame_interval=frame_interval
            )
            s_structural, struc_details = self.structural_analyzer.analyze_with_details(
                face_img, quality_score, landmarks=landmarks
            )

            final_score, weights = self.fusion.fuse(
                s_vit=None,
                vit_confidence=None,
                s_spatial=s_spatial,
                s_temporal=s_temporal,
                s_structural=s_structural,
                quality_score=quality_score,
                temporal_available=temp_avail
            )

            p_fake = final_score
            p_real = 1.0 - final_score
            final_label = "Deepfake" if final_score >= 0.50 else "Realism"
            final_confidence = abs(final_score - 0.5) * 2.0

        return {
            'decision_mode': mode.value,
            'final_score': round(final_score, 4),
            'final_label': final_label,
            'final_confidence': round(final_confidence, 4),
            'p_real': round(p_real, 4),
            'p_fake': round(p_fake, 4),
            'vit_confidence': round(vit_conf, 4),
            's_vit': round(s_vit, 4) if s_vit is not None else None,
            's_spatial': round(s_spatial, 4),
            's_temporal': round(s_temporal, 4),
            's_structural': round(s_structural, 4),
            'temporal_available': temp_avail,
            'weights': weights,
            'quality': round(quality_score, 2),
            'temporal_details': temp_details,
            'structural_details': struc_details
        }
