"""
Dataset-Level Evaluation and Experiment Benchmarking CLI for Deepfake Detection
================================================================================
Comprehensive evaluation engine for IEEE paper experiments:
- Experiment A: ViT only
- Experiment B: ViT + Temporal Aggregation
- Experiment C: QMC-FD Fallback only (Spatial + Temporal + Structural without ViT)
- Experiment D: Full System (ViT + QMC-FD + Temporal Aggregation)
- Ablation Studies: ViT only, ViT+Spatial, ViT+Spatial+Temporal, Full, etc.
- Threshold Calibration: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
- Weight Calibration: W1, W2, W3, W4, grid search
- Robustness Benchmarks: Clean, Compression, Blur, Low Illum, Low Res, Noise

CRITICAL RESEARCH RULE:
Zero metrics are fabricated, simulated, or hardcoded.
All reported scores, FPS, and latencies are computed strictly from real inference
on labeled data.
"""

import os
import sys
import time
import math
import json
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import yaml
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
    DecisionMode
)


# ==================== METRIC CALCULATIONS ====================

def calculate_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: List[float]
) -> Dict[str, Any]:
    """
    Calculate dataset-level classification metrics:
    Accuracy, Precision, Recall, F1, Specificity, FPR, FNR, Confusion Matrix, and exact ROC-AUC.
    0 = Real / Authentic (Negative), 1 = Deepfake (Positive).
    """
    if not y_true or not y_pred:
        return {
            'total_samples': 0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'specificity': 0.0, 'f1': 0.0, 'roc_auc': 0.0, 'fpr': 0.0, 'fnr': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
        }

    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)
    scores = np.array(y_scores, dtype=float)

    total = len(y_t)
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC via Wilcoxon-Mann-Whitney U statistic with fractional mid-ranks for ties
    roc_auc = 0.5
    pos_count = int(np.sum(y_t == 1))
    neg_count = int(np.sum(y_t == 0))

    if pos_count > 0 and neg_count > 0:
        # Compute fractional mid-ranks for ties
        sorter = np.argsort(scores)
        inv = np.empty_like(sorter)
        inv[sorter] = np.arange(len(scores))

        ranks = np.zeros(len(scores), dtype=float)
        unique_scores, inverse_indices, counts = np.unique(scores, return_inverse=True, return_counts=True)
        cum_counts = np.cumsum(counts)
        start_ranks = np.zeros_like(cum_counts)
        start_ranks[1:] = cum_counts[:-1]
        mid_ranks = start_ranks + (counts - 1) / 2.0 + 1.0  # 1-indexed

        for idx, u_idx in enumerate(inverse_indices):
            ranks[idx] = mid_ranks[u_idx]

        pos_ranks_sum = float(np.sum(ranks[y_t == 1]))
        u_stat = pos_ranks_sum - (pos_count * (pos_count + 1.0)) / 2.0
        roc_auc = float(u_stat / (pos_count * neg_count))
        roc_auc = float(np.clip(roc_auc, 0.0, 1.0))
    elif pos_count == 0 or neg_count == 0:
        # If single-class only, report None instead of fabricating
        roc_auc = float('nan')

    return {
        'total_samples': total,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'specificity': round(specificity, 4),
        'f1': round(f1, 4),
        'roc_auc': round(roc_auc, 4) if not math.isnan(roc_auc) else None,
        'fpr': round(fpr, 4),
        'fnr': round(fnr, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


# ==================== PERTURBATIONS ====================

def apply_perturbation(frame: np.ndarray, condition: str) -> np.ndarray:
    """Applies controlled realistic perturbations for robustness benchmarking."""
    if condition == "none" or not condition:
        return frame

    perturbed = frame.copy()
    if condition == "blur":
        perturbed = cv2.GaussianBlur(perturbed, (15, 15), 5.0)
    elif condition == "compression":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
        _, encimg = cv2.imencode('.jpg', perturbed, encode_param)
        perturbed = cv2.imdecode(encimg, 1)
    elif condition == "low_illumination":
        perturbed = cv2.convertScaleAbs(perturbed, alpha=0.35, beta=0)
    elif condition == "low_res":
        h, w = perturbed.shape[:2]
        down = cv2.resize(perturbed, (max(16, w // 4), max(16, h // 4)), interpolation=cv2.INTER_NEAREST)
        perturbed = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    elif condition == "noise":
        np.random.seed(42)
        noise = np.random.normal(0, 15, perturbed.shape).astype(np.int16)
        perturbed = np.clip(perturbed.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return perturbed


# ==================== DATASET EVALUATOR ====================

class DatasetEvaluator:
    """
    Standardized dataset evaluation engine.
    Ensures identical videos are evaluated across all configurations (A, B, C, D)
    with strict isolation, warm-up handling, and full telemetry logging.
    """

    def __init__(self, config_path: str = "experiment_config.yaml", device: str = "cpu"):
        self.device = device
        self.config_path = config_path

        # Load config
        self.cfg = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)

        seed = self.cfg.get('experiment_meta', {}).get('random_seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"Loading ViT Deepfake Detection Model ({self.device})...")
        model_name = self.cfg.get('vit_model', {}).get('model_name', "prithivMLmods/Deep-Fake-Detector-v2-Model")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        if device != "cpu" and torch.cuda.is_available():
            self.model.to(device)

        self.detector = get_detector()
        print("Models loaded. Performing initial warm-up passes...")
        self._warmup_model()
        print("Initialization and warm-up complete.")

    def _warmup_model(self):
        """Warm up model to eliminate PyTorch first-run CUDA/MKL allocation latency."""
        dummy = np.ones((128, 128, 3), dtype=np.uint8) * 128
        for _ in range(3):
            predict_deepfake(dummy, self.processor, self.model, device=self.device)

    def load_dataset_records(
        self,
        dataset_source: Optional[str] = None,
        split_filter: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Loads dataset items from metadata CSV or directory structure (dataset/real, dataset/fake).
        Returns list of {'video_path': ..., 'label': 'real'|'fake', 'split': 'val'|'test'|'all'}.
        """
        records = []
        source = dataset_source or self.cfg.get('dataset', {}).get('metadata_csv', 'dataset/metadata.csv')

        if os.path.exists(source) and source.endswith('.csv'):
            df = pd.read_csv(source)
            for _, row in df.iterrows():
                v_path = str(row['video_path'])
                lbl = str(row['label']).strip().lower()
                split = str(row.get('split', 'all')).strip().lower()
                if split_filter and split_filter != "all" and split != split_filter:
                    continue
                records.append({
                    'video_path': v_path,
                    'label': 'real' if 'real' in lbl else 'fake',
                    'split': split
                })
        else:
            # Fallback to directory scan
            root_dir = dataset_source or self.cfg.get('dataset', {}).get('root_dir', 'dataset')
            for sub, lbl in [('real', 'real'), ('fake', 'fake')]:
                sub_dir = os.path.join(root_dir, sub)
                if os.path.exists(sub_dir):
                    for fname in sorted(os.listdir(sub_dir)):
                        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            records.append({
                                'video_path': os.path.join(sub_dir, fname),
                                'label': lbl,
                                'split': 'all'
                            })

        return records

    def evaluate_video(
        self,
        video_path: str,
        ground_truth: str,
        experiment_type: str = "D",
        ablation_mode: Optional[str] = None,
        condition: str = "none",
        vit_fallback_threshold: Optional[float] = None,
        weights: Optional[Tuple[float, float, float, float]] = None,
        decision_threshold: float = 0.50,
        max_samples: int = 40
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation on a single video with full telemetry.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        t_video_start = time.perf_counter()

        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = sample_video_frame_indices(
            total_frames=total_frames,
            fps=fps,
            max_samples=max_samples,
            target_sample_fps=self.cfg.get('sampling', {}).get('target_sample_fps', 3.0)
        )

        # Configure custom fallback detector
        fb_config = FallbackConfig()
        if vit_fallback_threshold is not None:
            fb_config.vit_confidence_threshold = vit_fallback_threshold
        elif 'vit_model' in self.cfg and 'fallback_uncertainty_threshold' in self.cfg['vit_model']:
            fb_config.vit_confidence_threshold = self.cfg['vit_model']['fallback_uncertainty_threshold']

        if weights is not None:
            fb_config.base_weight_vit, fb_config.base_weight_spatial, fb_config.base_weight_temporal, fb_config.base_weight_structural = weights
        elif ablation_mode == "vit_only":
            fb_config.base_weight_vit = 1.0
            fb_config.base_weight_spatial = 0.0
            fb_config.base_weight_temporal = 0.0
            fb_config.base_weight_structural = 0.0
        elif ablation_mode == "vit_spatial":
            fb_config.base_weight_vit = 0.50
            fb_config.base_weight_spatial = 0.50
            fb_config.base_weight_temporal = 0.0
            fb_config.base_weight_structural = 0.0
        elif ablation_mode == "vit_spatial_temporal":
            fb_config.base_weight_vit = 0.40
            fb_config.base_weight_spatial = 0.30
            fb_config.base_weight_temporal = 0.30
            fb_config.base_weight_structural = 0.0
        elif ablation_mode == "cues_only":
            fb_config.base_weight_vit = 0.0
            fb_config.base_weight_spatial = 0.34
            fb_config.base_weight_temporal = 0.33
            fb_config.base_weight_structural = 0.33

        fallback_detector = CustomFallbackDetector(config=fb_config)
        fallback_detector.reset_temporal_state()

        frame_records = []
        inference_latencies = []
        pipeline_latencies = []
        valid_face_detections = 0

        for idx, f_no in enumerate(frame_indices):
            t_frame_start = time.perf_counter()

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = apply_perturbation(frame, condition)
            faces = self.detector.detect_faces(frame, max_faces=1)

            if not faces:
                continue

            valid_face_detections += 1
            best_face = faces[0]
            face_img = best_face['image']
            q_score = best_face.get('quality', 50.0)
            face_landmarks = best_face.get('landmarks')
            step_interval = int(frame_indices[idx] - frame_indices[idx - 1]) if idx > 0 else 1

            # Timed model inference
            t_inf_start = time.perf_counter()

            if experiment_type == "A" or experiment_type == "B":
                # Primary ViT only
                v_label, v_conf, v_preal, v_pfake = predict_deepfake(face_img, self.processor, self.model, self.device)
                t_inf = (time.perf_counter() - t_inf_start) * 1000.0
                inference_latencies.append(t_inf)

                frame_records.append({
                    'frame_index': f_no,
                    'label': v_label,
                    'confidence': v_conf,
                    'p_real': v_preal,
                    'p_fake': v_pfake,
                    'quality': q_score,
                    'decision_mode': DecisionMode.PRIMARY_VIT.value,
                    'vit_confidence': abs(v_pfake - 0.5) * 2.0,
                    's_spatial': 0.0,
                    's_temporal': 0.0,
                    's_structural': 0.0
                })

            elif experiment_type == "C":
                # QMC-FD cues without ViT
                eval_res = fallback_detector.evaluate_frame(
                    face_img, vit_result=None, quality_score=q_score,
                    landmarks=face_landmarks, frame_interval=step_interval
                )
                t_inf = (time.perf_counter() - t_inf_start) * 1000.0
                inference_latencies.append(t_inf)

                frame_records.append({
                    'frame_index': f_no,
                    'label': eval_res['final_label'],
                    'confidence': eval_res['final_confidence'],
                    'p_real': eval_res['p_real'],
                    'p_fake': eval_res['p_fake'],
                    'quality': q_score,
                    'decision_mode': eval_res['decision_mode'],
                    'vit_confidence': 0.0,
                    's_spatial': eval_res.get('s_spatial', 0.0),
                    's_temporal': eval_res.get('s_temporal', 0.0),
                    's_structural': eval_res.get('s_structural', 0.0)
                })

            elif experiment_type == "D":
                # Full system (ViT + QMC-FD)
                eval_res = predict_deepfake_with_fallback(
                    face_img, self.processor, self.model,
                    device=self.device,
                    quality_score=q_score,
                    fallback_detector=fallback_detector,
                    landmarks=face_landmarks,
                    frame_interval=step_interval
                )
                t_inf = (time.perf_counter() - t_inf_start) * 1000.0
                inference_latencies.append(t_inf)

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

            t_pipe = (time.perf_counter() - t_frame_start) * 1000.0
            pipeline_latencies.append(t_pipe)

        cap.release()
        total_time_s = time.perf_counter() - t_video_start

        # Video-level aggregation
        if not frame_records:
            return {
                'video_path': video_path,
                'ground_truth': ground_truth,
                'predicted_label': 'Unknown',
                'p_real': 0.5,
                'p_fake': 0.5,
                'total_frames': 0,
                'valid_detections': 0,
                'fallback_frames': 0,
                'primary_vit_frames': 0,
                'avg_quality': 0.0,
                'avg_spatial': 0.0,
                'avg_temporal': 0.0,
                'avg_structural': 0.0,
                'processing_time_s': round(total_time_s, 2),
                'fps': 0.0,
                'inference_latency_ms': 0.0,
                'end_to_end_latency_ms': 0.0
            }

        if experiment_type == "A":
            # Majority / average frame voting for ViT only (no temporal aggregator)
            mean_fake = float(np.mean([p['p_fake'] for p in frame_records]))
            pred_lbl = "Deepfake" if mean_fake >= decision_threshold else "Realism"
            fallback_cnt = 0
            primary_vit_cnt = len(frame_records)
            video_result = {
                'final_label': pred_lbl,
                'video_fake_probability': mean_fake,
                'video_real_probability': 1.0 - mean_fake,
                'fallback_frames': 0
            }
        else:
            # Temporal Aggregator
            enable_anomaly = (experiment_type != "B")  # B evaluates standard temporal aggregation without burst boost if desired
            video_result = TemporalAggregator.aggregate(
                frame_records,
                decision_threshold=decision_threshold,
                enable_anomaly_detection=enable_anomaly
            )
            fallback_cnt = video_result.get('fallback_frames', 0)
            primary_vit_cnt = len(frame_records) - fallback_cnt

        mean_inf_lat = float(np.mean(inference_latencies)) if inference_latencies else 0.0
        mean_pipe_lat = float(np.mean(pipeline_latencies)) if pipeline_latencies else 0.0
        eff_fps = float(1000.0 / mean_pipe_lat) if mean_pipe_lat > 0 else 0.0

        p_fake_final = round(float(video_result['video_fake_probability']), 4)
        p_real_final = round(float(1.0 - p_fake_final), 4)

        return {
            'video_path': video_path,
            'ground_truth': ground_truth,
            'predicted_label': video_result['final_label'],
            'final_real_probability': p_real_final,
            'final_fake_probability': p_fake_final,
            'total_frames': len(frame_records),
            'valid_face_detections': valid_face_detections,
            'fallback_frames': fallback_cnt,
            'primary_vit_frames': primary_vit_cnt,
            'average_face_quality': round(float(np.mean([p['quality'] for p in frame_records])), 2),
            'average_spatial_score': round(float(np.mean([p.get('s_spatial', 0.0) for p in frame_records])), 4),
            'average_temporal_score': round(float(np.mean([p.get('s_temporal', 0.0) for p in frame_records])), 4),
            'average_structural_score': round(float(np.mean([p.get('s_structural', 0.0) for p in frame_records])), 4),
            'processing_time_s': round(total_time_s, 2),
            'fps': round(eff_fps, 2),
            'inference_latency_ms': round(mean_inf_lat, 2),
            'end_to_end_latency_ms': round(mean_pipe_lat, 2),
            'frame_predictions': frame_records
        }

    def evaluate_dataset(
        self,
        records: List[Dict[str, str]],
        experiment_type: str = "D",
        ablation_mode: Optional[str] = None,
        condition: str = "none",
        vit_fallback_threshold: Optional[float] = None,
        weights: Optional[Tuple[float, float, float, float]] = None,
        decision_threshold: float = 0.50,
        max_samples: int = 40
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluates a list of video records and computes dataset-level metrics.
        """
        video_results = []
        y_true = []
        y_pred = []
        y_scores = []

        for item in records:
            v_path = item['video_path']
            gt = item['label']
            res = self.evaluate_video(
                video_path=v_path,
                ground_truth=gt,
                experiment_type=experiment_type,
                ablation_mode=ablation_mode,
                condition=condition,
                vit_fallback_threshold=vit_fallback_threshold,
                weights=weights,
                decision_threshold=decision_threshold,
                max_samples=max_samples
            )
            video_results.append(res)

            gt_bin = 1 if "fake" in gt.lower() else 0
            pred_bin = 1 if res['predicted_label'] == "Deepfake" else 0
            score = res['final_fake_probability']

            y_true.append(gt_bin)
            y_pred.append(pred_bin)
            y_scores.append(score)

        metrics = calculate_classification_metrics(y_true, y_pred, y_scores)

        # Aggregate latency & FPS
        all_inf_lat = [r['inference_latency_ms'] for r in video_results if r['inference_latency_ms'] > 0]
        all_pipe_lat = [r['end_to_end_latency_ms'] for r in video_results if r['end_to_end_latency_ms'] > 0]
        metrics['mean_inference_latency_ms'] = round(float(np.mean(all_inf_lat)), 2) if all_inf_lat else 0.0
        metrics['mean_end_to_end_latency_ms'] = round(float(np.mean(all_pipe_lat)), 2) if all_pipe_lat else 0.0
        metrics['mean_effective_fps'] = round(float(1000.0 / metrics['mean_end_to_end_latency_ms']), 2) if metrics['mean_end_to_end_latency_ms'] > 0 else 0.0

        return metrics, video_results


# ==================== CLI INTERFACE ====================

def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection across dataset and experimental configurations")
    parser.add_argument("--config", type=str, default="experiment_config.yaml", help="Path to experiment_config.yaml")
    parser.add_argument("--dataset", type=str, default="dataset/metadata.csv", help="Path to metadata.csv or dataset root dir")
    parser.add_argument("--split", type=str, default="test", choices=["all", "val", "test", "train"], help="Split to evaluate: 'val' or 'test'")
    parser.add_argument("--experiment", type=str, default="D", choices=["A", "B", "C", "D", "all"], help="Experiment mode")
    parser.add_argument("--ablation", type=str, default=None, choices=["vit_only", "vit_spatial", "vit_spatial_temporal", "cues_only", "full_with_temporal", "full_no_temporal"], help="Ablation mode")
    parser.add_argument("--condition", type=str, default="none", choices=["none", "blur", "compression", "low_illumination", "low_res", "noise"], help="Robustness perturbation")
    parser.add_argument("--vit-threshold", type=float, default=None, help="Override ViT fallback uncertainty threshold")
    parser.add_argument("--decision-threshold", type=float, default=0.50, help="Video decision threshold (default: 0.50)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--max-samples", type=int, default=30, help="Max frames sampled per video")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()

    evaluator = DatasetEvaluator(config_path=args.config, device=args.device)
    records = evaluator.load_dataset_records(dataset_source=args.dataset, split_filter=args.split)

    if not records:
        print(f"Error: No video records found for split '{args.split}' from source '{args.dataset}'.")
        sys.exit(1)

    print(f"\nLoaded {len(records)} videos from split '{args.split}':")
    for r in records:
        print(f" - [{r['label'].upper()}] {r['video_path']}")

    exp_list = ["A", "B", "C", "D"] if args.experiment == "all" else [args.experiment]
    os.makedirs(os.path.join(args.output_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "json"), exist_ok=True)

    summary_rows = []

    for exp in exp_list:
        print(f"\n=======================================================")
        print(f"  RUNNING EXPERIMENT {exp} on {args.split.upper()} SET (condition={args.condition})")
        print(f"=======================================================")

        metrics, per_video = evaluator.evaluate_dataset(
            records=records,
            experiment_type=exp,
            ablation_mode=args.ablation,
            condition=args.condition,
            vit_fallback_threshold=args.vit_threshold,
            decision_threshold=args.decision_threshold,
            max_samples=args.max_samples
        )

        metrics['experiment'] = exp
        metrics['split'] = args.split
        metrics['condition'] = args.condition
        metrics['ablation'] = args.ablation or "none"
        metrics['vit_threshold'] = args.vit_threshold or 0.70
        summary_rows.append(metrics)

        print("\n--- EXPERIMENT RESULTS ---")
        for k, v in metrics.items():
            print(f"  {k:<28}: {v}")

        # Save per-video results
        clean_per_video = []
        for pv in per_video:
            entry = {k: v for k, v in pv.items() if k != 'frame_predictions'}
            clean_per_video.append(entry)

        pv_df = pd.DataFrame(clean_per_video)
        pv_csv = os.path.join(args.output_dir, "csv", f"per_video_exp_{exp}_{args.split}.csv")
        pv_df.to_csv(pv_csv, index=False)
        print(f"Saved per-video results to {pv_csv}")

    # Save summary
    sum_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(args.output_dir, "csv", f"results_summary_{args.split}.csv")
    sum_df.to_csv(sum_csv, index=False)

    sum_json = os.path.join(args.output_dir, "json", f"metrics_summary_{args.split}.json")
    with open(sum_json, 'w') as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSummary results saved to {sum_csv} and {sum_json}")


if __name__ == '__main__':
    main()
