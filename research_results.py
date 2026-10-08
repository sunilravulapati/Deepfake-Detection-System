"""
research_results.py
====================
Single source of truth for all verified experimental results.

IMPORTANT:
- These values come from actual evaluation runs on the labeled dataset.
- Do NOT modify numbers to improve reported performance.
- Do NOT add fabricated metrics or confidence intervals.
- Accuracy / Precision / Recall / F1 are dataset-level metrics, NOT individual-video scores.
- P(fake) values for each video are weighted temporal-aggregation scores from the pipeline.
- The model is prithivMLmods/Deep-Fake-Detector-v2-Model (pre-trained, not fine-tuned by us).

Source files:
  results/csv/per_video_results.csv
  results/csv/results_summary.csv
  results/json/metrics_summary.json
"""

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
DATASET = {
    "total": 6,
    "real": 3,
    "fake": 3,
    "val": 2,       # 1 real + 1 fake
    "test": 4,      # 2 real + 2 fake
    "fake_source": "DeeperForensics",
    "real_source": "Interview footage",
    "note": (
        "Preliminary evaluation — small dataset. "
        "Results are not representative of general detection performance."
    ),
}

# ─────────────────────────────────────────────
# CONFUSION MATRIX  (test split, Experiment A)
# Actual label  →  row;  Predicted  →  column
# Positive class = Fake (1), Negative class = Real (0)
# ─────────────────────────────────────────────
CONFUSION = {
    "TP": 0,   # Actual Fake, Predicted Fake
    "TN": 1,   # Actual Real, Predicted Real
    "FP": 1,   # Actual Real, Predicted Fake  (false alarm)
    "FN": 2,   # Actual Fake, Predicted Real  (missed detection)
}

# ─────────────────────────────────────────────
# CLASSIFICATION METRICS  (test split)
# ─────────────────────────────────────────────
METRICS = {
    "vit_only": {
        "label": "ViT Only (Config A)",
        "accuracy":  0.25,
        "precision": 0.00,
        "recall":    0.00,
        "f1":        0.00,
        "roc_auc":   0.00,
        "fpr":       0.50,
        "fnr":       1.00,
    },
    "vit_temporal": {
        "label": "ViT + Temporal (Config B)",
        "accuracy":  0.25,
        "precision": 0.00,
        "recall":    0.00,
        "f1":        0.00,
        "roc_auc":   0.00,
        "fpr":       0.50,
        "fnr":       1.00,
    },
}

# ─────────────────────────────────────────────
# LATENCY  (mean across test videos, CPU)
# ─────────────────────────────────────────────
LATENCY = {
    "vit_only": {
        "label": "ViT Only (Config A)",
        "vit_inference_ms":  88.97,
        "end_to_end_ms":    153.21,
        "fps":                6.53,
    },
    "vit_temporal": {
        "label": "ViT + Temporal (Config B)",
        "vit_inference_ms":  89.93,
        "end_to_end_ms":    152.25,
        "fps":                6.57,
    },
}

# ─────────────────────────────────────────────
# PER-VIDEO RESULTS  (test split, Experiment A)
# ─────────────────────────────────────────────
PER_VIDEO = [
    {
        "name": "real_interview_02",
        "ground_truth": "Real",
        "prediction":   "Real",
        "correct":      True,
        "error_type":   None,
        "p_fake":        0.4711,
        "p_real":        0.5289,
        "avg_quality":   84.36,
        "frames":        15,
    },
    {
        "name": "real_interview_03",
        "ground_truth": "Real",
        "prediction":   "Fake",
        "correct":      False,
        "error_type":   "False Positive",
        "p_fake":        0.6495,
        "p_real":        0.3505,
        "avg_quality":   82.42,
        "frames":        14,
    },
    {
        "name": "fake_deeperforensics_02",
        "ground_truth": "Fake",
        "prediction":   "Real",
        "correct":      False,
        "error_type":   "False Negative",
        "p_fake":        0.3888,
        "p_real":        0.6112,
        "avg_quality":   54.70,
        "frames":        19,
    },
    {
        "name": "fake_deeperforensics_03",
        "ground_truth": "Fake",
        "prediction":   "Real",
        "correct":      False,
        "error_type":   "False Negative",
        "p_fake":        0.4212,
        "p_real":        0.5788,
        "avg_quality":   55.14,
        "frames":        19,
    },
]

# ─────────────────────────────────────────────
# VERDICT BANDS  (from verdict_logic.py)
# ─────────────────────────────────────────────
VERDICT_BANDS = [
    {"label": "Highly Likely Manipulated",         "min": 0.80, "max": 1.00, "tier": "warning"},
    {"label": "Likely Manipulated",                "min": 0.60, "max": 0.80, "tier": "danger"},
    {"label": "Inconclusive — May Be Manipulated", "min": 0.40, "max": 0.60, "tier": "uncertain"},
    {"label": "Likely Authentic",                  "min": 0.00, "max": 0.40, "tier": "safe"},
]

# ─────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────
MODEL_INFO = {
    "name":         "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "architecture": "Vision Transformer (ViT-Base/16)",
    "input_size":   "224 × 224 px",
    "patch_size":   "16 × 16 px",
    "hidden_size":  768,
    "num_layers":   12,
    "num_heads":    12,
    "mlp_size":     3072,
    "output_classes": 2,
    "class_0":      "Realism",
    "class_1":      "Deepfake",
    "norm_mean":    "(0.5, 0.5, 0.5)",
    "norm_std":     "(0.5, 0.5, 0.5)",
    "device":       "CPU",
    "trained_by_us": False,
    "fine_tuned":   False,
    "attribution":  "Publicly available pre-trained model used for inference only.",
}

# ─────────────────────────────────────────────
# SYSTEM LIMITATIONS
# ─────────────────────────────────────────────
LIMITATIONS = [
    "Only six videos were available for system evaluation.",
    "Only four videos were used for final testing.",
    "The external ViT was not trained or fine-tuned by us.",
    "The pre-training dataset of the external model is not documented in our repository.",
    "All fake evaluation videos came from the DeeperForensics source.",
    "Threshold tuning cannot be reliably performed with only two validation videos.",
    "Webcam evaluation has not been performed against a labeled webcam benchmark.",
    "Only one face per frame is currently analyzed.",
]


def get_dataset_summary() -> dict:
    """Return dataset overview metrics."""
    return DATASET


def get_confusion_matrix() -> dict:
    """Return test confusion matrix dict (TP, TN, FP, FN)."""
    return CONFUSION


def get_metrics(config: str = "vit_temporal") -> dict:
    """Return classification performance metrics for a configuration."""
    return METRICS.get(config, METRICS["vit_temporal"])


def get_latency(config: str = "vit_temporal") -> dict:
    """Return latency and throughput metrics for a configuration."""
    return LATENCY.get(config, LATENCY["vit_temporal"])


def get_per_video_results() -> list:
    """Return test per-video results list."""
    return PER_VIDEO


def get_verdict_bands() -> list:
    """Return engineering interpretation verdict bands."""
    return VERDICT_BANDS


def get_model_info() -> dict:
    """Return pre-trained ViT architecture metadata."""
    return MODEL_INFO


def get_limitations() -> list:
    """Return documented system limitations."""
    return LIMITATIONS

