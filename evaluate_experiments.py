"""
Evaluation and Experiment Benchmarking CLI for Deepfake Detection
================================================================
Supports research evaluation across:
- Experiment A: ViT only
- Experiment B: ViT + Temporal Aggregation
- Experiment C: Custom Fallback only (Spatial + Temporal + Structural without ViT)
- Experiment D: ViT + Custom Fallback (QMC-FD) + Temporal Aggregation
- Ablation Studies:
    - ViT + Spatial
    - ViT + Spatial + Temporal
    - ViT + Spatial + Temporal + Structural
    - Spatial + Temporal + Structural (no ViT)
- Challenging Condition Perturbations:
    - Clean / Normal
    - Gaussian Blur
    - JPEG Compression Artifacts
    - Low Illumination
    - Low Resolution Downsampling

CRITICAL RESEARCH RULE:
No metrics are ever fabricated or mocked. All metrics, probabilities, FPS,
and latency numbers are computed strictly from real inference and input data.
"""

import os
import sys
import time
import argparse
import json
import math
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from pipeline import (
    RobustFaceDetector,
    get_detector,
    align_and_crop_face,
    compute_face_quality,
    predict_deepfake,
    predict_deepfake_with_fallback,
    TemporalAggregator,
    sample_video_frame_indices
)
from custom_fallback import (
    CustomFallbackDetector,
    FallbackConfig,
    DecisionMode,
    SpatialArtifactAnalyzer,
    TemporalConsistencyAnalyzer,
    StructuralAnalyzer,
    QualityAwareFusion
)


# ==================== METRICS CALCULATION ====================

def calculate_classification_metrics(y_true: List[int], y_pred: List[int], y_scores: List[float]) -> Dict[str, float]:
    """
    Calculate Accuracy, Precision, Recall, F1, and approximate ROC-AUC.
    0 = Realism (Negative), 1 = Deepfake (Positive).
    """
    if not y_true or not y_pred:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0}

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    scores = np.array(y_scores)

    total = len(y_t)
    accuracy = float(np.sum(y_t == y_p) / total)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Approximate ROC-AUC via trapezoidal integration over sorted thresholds
    roc_auc = 0.5
    pos_count = int(np.sum(y_t == 1))
    neg_count = int(np.sum(y_t == 0))

    if pos_count > 0 and neg_count > 0:
        # Rank-sum calculation (Mann-Whitney U statistic)
        order = np.argsort(scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores)) + 1
        pos_ranks_sum = np.sum(ranks[y_t == 1])
        u_stat = pos_ranks_sum - (pos_count * (pos_count + 1)) / 2.0
        roc_auc = float(u_stat / (pos_count * neg_count))
        roc_auc = float(np.clip(roc_auc, 0.0, 1.0))

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


# ==================== CHALLENGING CONDITION PERTURBATIONS ====================

def apply_challenging_condition(frame: np.ndarray, condition: str) -> np.ndarray:
    """
    Applies real synthetic perturbations to benchmark robustness
    under challenging conditions (Step 22).
    """
    if condition == "none" or not condition:
        return frame

    perturbed = frame.copy()
    if condition == "blur":
        # Severe motion/defocus blur
        perturbed = cv2.GaussianBlur(perturbed, (15, 15), 5.0)

    elif condition == "compression":
        # Simulate heavy JPEG / H.264 compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
        _, encimg = cv2.imencode('.jpg', perturbed, encode_param)
        perturbed = cv2.imdecode(encimg, 1)

    elif condition == "low_illumination":
        # Extreme underexposure / dim lighting
        perturbed = cv2.convertScaleAbs(perturbed, alpha=0.35, beta=0)

    elif condition == "low_res":
        # Severe downsampling and upscaling
        h, w = perturbed.shape[:2]
        down = cv2.resize(perturbed, (max(16, w // 4), max(16, h // 4)), interpolation=cv2.INTER_NEAREST)
        perturbed = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

    return perturbed


# ==================== EXPERIMENT RUNNER ====================

class ExperimentEvaluator:
    """
    Evaluates deepfake detection pipelines across standard experimental configurations
    and ablations.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        print("Loading models for evaluation benchmark...")
        model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        if device != "cpu" and torch.cuda.is_available():
            self.model.to(device)
        self.detector = get_detector()
        print("Models loaded successfully.")

    def run_video_evaluation(
        self,
        video_path: str,
        ground_truth_label: Optional[str] = None,
        experiment_type: str = "D",
        ablation_mode: Optional[str] = None,
        condition: str = "none",
        max_samples: int = 40
    ) -> Dict[str, Any]:
        """
        Runs evaluation on a single video file.

        Configurations:
        - "A": ViT only (frame-by-frame ViT thresholding without aggregation)
        - "B": ViT + Temporal Aggregation (baseline)
        - "C": Custom fallback only (no ViT)
        - "D": ViT + QMC-FD Fallback + Temporal Aggregation (proposed)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = sample_video_frame_indices(
            total_frames=total_frames,
            fps=fps,
            max_samples=max_samples,
            target_sample_fps=3.0
        )

        # Configure custom fallback depending on ablation mode
        config = FallbackConfig()
        if ablation_mode == "vit_spatial":
            config.base_weight_vit = 0.50
            config.base_weight_spatial = 0.50
            config.base_weight_temporal = 0.0
            config.base_weight_structural = 0.0
        elif ablation_mode == "vit_spatial_temporal":
            config.base_weight_vit = 0.40
            config.base_weight_spatial = 0.30
            config.base_weight_temporal = 0.30
            config.base_weight_structural = 0.0
        elif ablation_mode == "cues_only":
            config.base_weight_vit = 0.0
            config.base_weight_spatial = 0.34
            config.base_weight_temporal = 0.33
            config.base_weight_structural = 0.33

        fallback_detector = CustomFallbackDetector(config=config)
        fallback_detector.reset_temporal_state()

        frame_records = []
        inference_latencies = []

        for idx, f_no in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = apply_challenging_condition(frame, condition)
            faces = self.detector.detect_faces(frame, max_faces=1)

            if not faces:
                continue

            best_face = faces[0]
            face_img = best_face['image']
            q_score = best_face.get('quality', 50.0)

            t0 = time.perf_counter()

            step_interval = int(frame_indices[idx] - frame_indices[idx - 1]) if idx > 0 else 1
            face_landmarks = best_face.get('landmarks')

            # Execute the specified experiment mode
            if experiment_type == "A":
                # EXPERIMENT A: ViT only
                label, conf, p_real, p_fake = predict_deepfake(face_img, self.processor, self.model, self.device)
                lat = (time.perf_counter() - t0) * 1000.0
                inference_latencies.append(lat)
                frame_records.append({
                    'frame_index': f_no,
                    'label': label,
                    'confidence': conf,
                    'p_real': p_real,
                    'p_fake': p_fake,
                    'quality': q_score,
                    'decision_mode': 'PRIMARY_VIT',
                    'vit_confidence': abs(p_fake - 0.5) * 2.0
                })

            elif experiment_type == "B":
                # EXPERIMENT B: ViT + existing temporal aggregation
                label, conf, p_real, p_fake = predict_deepfake(face_img, self.processor, self.model, self.device)
                lat = (time.perf_counter() - t0) * 1000.0
                inference_latencies.append(lat)
                frame_records.append({
                    'frame_index': f_no,
                    'label': label,
                    'confidence': conf,
                    'p_real': p_real,
                    'p_fake': p_fake,
                    'quality': q_score,
                    'decision_mode': 'PRIMARY_VIT',
                    'vit_confidence': abs(p_fake - 0.5) * 2.0
                })

            elif experiment_type == "C":
                # EXPERIMENT C: Custom fallback only (no ViT)
                eval_res = fallback_detector.evaluate_frame(
                    face_img, vit_result=None, quality_score=q_score,
                    landmarks=face_landmarks, frame_interval=step_interval
                )
                lat = (time.perf_counter() - t0) * 1000.0
                inference_latencies.append(lat)
                frame_records.append({
                    'frame_index': f_no,
                    'label': eval_res['final_label'],
                    'confidence': eval_res['final_confidence'],
                    'p_real': eval_res['p_real'],
                    'p_fake': eval_res['p_fake'],
                    'quality': q_score,
                    'decision_mode': eval_res['decision_mode'],
                    's_spatial': eval_res.get('s_spatial'),
                    's_temporal': eval_res.get('s_temporal'),
                    's_structural': eval_res.get('s_structural')
                })

            elif experiment_type == "D":
                # EXPERIMENT D: ViT + QMC-FD + Temporal Aggregation
                eval_res = predict_deepfake_with_fallback(
                    face_img, self.processor, self.model,
                    device=self.device,
                    quality_score=q_score,
                    fallback_detector=fallback_detector,
                    landmarks=face_landmarks,
                    frame_interval=step_interval
                )
                lat = (time.perf_counter() - t0) * 1000.0
                inference_latencies.append(lat)
                frame_records.append({
                    'frame_index': f_no,
                    'label': eval_res['final_label'],
                    'confidence': eval_res['final_confidence'],
                    'p_real': eval_res['p_real'],
                    'p_fake': eval_res['p_fake'],
                    'quality': q_score,
                    'decision_mode': eval_res['decision_mode'],
                    'vit_confidence': eval_res.get('vit_confidence', 0.0),
                    's_spatial': eval_res.get('s_spatial', 0.0),
                    's_temporal': eval_res.get('s_temporal', 0.0),
                    's_structural': eval_res.get('s_structural', 0.0)
                })

        cap.release()

        # Compute video-level verdict
        if experiment_type == "A":
            # Direct majority frame decision
            fake_votes = sum(1 for p in frame_records if p['p_fake'] >= 0.50)
            is_df = fake_votes > len(frame_records) / 2.0 if frame_records else False
            final_label = "Deepfake" if is_df else "Realism"
            avg_conf = float(np.mean([p['confidence'] for p in frame_records])) if frame_records else 0.0
            video_result = {
                'final_label': final_label,
                'avg_confidence': round(avg_conf, 4),
                'real_count': len(frame_records) - fake_votes,
                'fake_count': fake_votes,
                'total_predictions': len(frame_records),
                'fallback_frames': 0,
                'mode_counts': {'PRIMARY_VIT': len(frame_records), 'FALLBACK_FUSION': 0, 'FALLBACK_ONLY': 0}
            }
        else:
            video_result = TemporalAggregator.aggregate(frame_records)

        # Performance metrics
        avg_lat = float(np.mean(inference_latencies)) if inference_latencies else 0.0
        fps_eff = float(1000.0 / avg_lat) if avg_lat > 0 else 0.0

        # Ground truth evaluation if provided
        metrics = {}
        if ground_truth_label is not None:
            gt_bin = 1 if "fake" in ground_truth_label.lower() else 0
            pred_bin = 1 if video_result['final_label'] == "Deepfake" else 0
            p_score = video_result.get('video_fake_probability', video_result['avg_confidence'] if pred_bin == 1 else 1.0 - video_result['avg_confidence'])
            metrics = calculate_classification_metrics([gt_bin], [pred_bin], [p_score])

        return {
            'video_path': video_path,
            'experiment_type': experiment_type,
            'ablation_mode': ablation_mode or "standard",
            'condition': condition,
            'final_label': video_result['final_label'],
            'avg_confidence': video_result['avg_confidence'],
            'total_frames_analyzed': len(frame_records),
            'fallback_frames': video_result.get('fallback_frames', 0),
            'fallback_activation_rate': round((video_result.get('fallback_frames', 0) / max(1, len(frame_records))) * 100.0, 1),
            'mode_counts': video_result.get('mode_counts', {}),
            'average_quality': round(float(np.mean([p.get('quality', 0) for p in frame_records])), 2) if frame_records else 0.0,
            'average_vit_confidence': round(float(np.mean([p.get('vit_confidence', 0) for p in frame_records])), 4) if frame_records else 0.0,
            'avg_inference_latency_ms': round(avg_lat, 2),
            'effective_fps': round(fps_eff, 2),
            'metrics': metrics
        }


# ==================== CLI INTERFACE ====================

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Research Experiment Evaluation Tool")
    parser.add_argument("--video", type=str, default="01_02__exit_phone_room__YVGY8LOK.mp4",
                        help="Path to video file to evaluate")
    parser.add_argument("--gt-label", type=str, default=None, choices=["Realism", "Deepfake", "real", "fake"],
                        help="Optional ground truth label for accuracy evaluation")
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "A", "B", "C", "D"],
                        help="Experiment to run: A (ViT only), B (ViT+Temporal), C (Fallback only), D (ViT+Fallback+Temporal), all")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["vit_spatial", "vit_spatial_temporal", "cues_only"],
                        help="Optional ablation configuration")
    parser.add_argument("--condition", type=str, default="none",
                        choices=["none", "blur", "compression", "low_illumination", "low_res"],
                        help="Challenging condition perturbation to apply")
    parser.add_argument("--max-samples", type=int, default=30,
                        help="Maximum sampled frames per video")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON experiment report")

    args = parser.parse_args()

    evaluator = ExperimentEvaluator(device=args.device)

    exp_list = ["A", "B", "C", "D"] if args.experiment == "all" else [args.experiment]
    results = []

    print("\n" + "=" * 65)
    print("      RESEARCH EXPERIMENT EVALUATION BENCHMARK")
    print("=" * 65)
    print(f"Target Video: {args.video}")
    print(f"Condition:    {args.condition}")
    print(f"Ground Truth: {args.gt_label or 'Unspecified'}")
    print("=" * 65 + "\n")

    for exp in exp_list:
        print(f"--> Running Experiment {exp}...")
        res = evaluator.run_video_evaluation(
            video_path=args.video,
            ground_truth_label=args.gt_label,
            experiment_type=exp,
            ablation_mode=args.ablation,
            condition=args.condition,
            max_samples=args.max_samples
        )
        results.append(res)
        print(f"    Verdict:           {res['final_label']} ({res['avg_confidence']*100:.1f}%)")
        print(f"    Latency:           {res['avg_inference_latency_ms']} ms/frame ({res['effective_fps']} FPS)")
        print(f"    Fallback Rate:     {res['fallback_activation_rate']}% ({res['fallback_frames']}/{res['total_frames_analyzed']} frames)")
        print(f"    Avg Face Quality:  {res['average_quality']}/100")
        print(f"    Avg ViT Conf:      {res['average_vit_confidence']*100:.1f}%")
        print("-" * 50)

    # Print summary comparative table
    print("\n" + "=" * 78)
    print(f"{'Experiment':<12} | {'Verdict':<10} | {'Confidence':<12} | {'Latency (ms)':<14} | {'FPS':<8} | {'Fallback %':<10}")
    print("=" * 78)
    for r in results:
        print(f"{r['experiment_type']:<12} | {r['final_label']:<10} | {r['avg_confidence']*100:.1f}%{'':<6} | {r['avg_inference_latency_ms']:<14} | {r['effective_fps']:<8} | {r['fallback_activation_rate']:<10}")
    print("=" * 78 + "\n")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
