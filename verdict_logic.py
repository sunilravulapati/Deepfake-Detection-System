"""
Uncertainty-Aware Verdict Interpretation Module
================================================
Translates the video-level P(fake) score from the ViT temporal aggregation
into a human-readable interpretation band.

IMPORTANT DESIGN CONSTRAINTS:
- P(fake) is a raw ViT softmax output, NOT a calibrated statistical probability.
- It must NEVER be displayed as "XX% Real Video" or as an "Accuracy" metric.
- These bands are interpretation ranges, not calibrated confidence intervals.
- The pre-trained model is: prithivMLmods/Deep-Fake-Detector-v2-Model
- We did NOT train or fine-tune this ViT.

Verdict bands (initial defaults — tune on validation data if sufficient):
  P(fake) < 0.40  →  Likely Authentic
  0.40 ≤ P(fake) < 0.60  →  Inconclusive — May Be Manipulated
  0.60 ≤ P(fake) < 0.80  →  Likely Manipulated
  P(fake) ≥ 0.80  →  Highly Likely Manipulated
"""

from typing import Dict, Any


# Interpretation band thresholds — adjustable without touching core logic
VERDICT_BANDS = [
    (0.80, "Highly Likely Manipulated",     "warning"),   # red
    (0.60, "Likely Manipulated",            "danger"),    # orange-red
    (0.40, "Inconclusive — May Be Manipulated", "uncertain"),  # amber
    (0.00, "Likely Authentic",              "safe"),      # green
]


def interpret_verdict(
    p_fake: float,
    p_real: float,
    bands: list = None
) -> Dict[str, Any]:
    """
    Convert P(fake) and P(real) into an uncertainty-aware interpretation dict.

    Args:
        p_fake: Weighted video-level fake probability (from TemporalAggregator)
        p_real: Weighted video-level real probability (from TemporalAggregator)
        bands: Optional override band list; defaults to VERDICT_BANDS

    Returns:
        Dict with:
          - verdict_label  : Human-readable interpretation label
          - verdict_tier   : 'safe' | 'uncertain' | 'danger' | 'warning'
          - manipulation_pct : int (0–100) — P(fake) as a percentage
          - real_pct         : int (0–100) — P(real) as a percentage
          - interpretation   : Short explanatory sentence
          - is_inconclusive  : bool — True when P(fake) in [0.40, 0.60)
          - p_fake           : raw float
          - p_real           : raw float
    """
    p_fake = float(p_fake)
    p_real = float(p_real)

    # Clamp to [0, 1]
    p_fake = max(0.0, min(1.0, p_fake))
    p_real = max(0.0, min(1.0, p_real))

    use_bands = bands if bands is not None else VERDICT_BANDS

    verdict_label = "Unknown"
    verdict_tier = "uncertain"
    for threshold, label, tier in use_bands:
        if p_fake >= threshold:
            verdict_label = label
            verdict_tier = tier
            break

    manipulation_pct = int(round(p_fake * 100))
    real_pct = int(round(p_real * 100))

    # Explanatory sentences
    if verdict_tier == "safe":
        interpretation = (
            "The model assigns a low manipulation probability to this video. "
            "The output suggests the content is likely authentic, though the "
            "result depends on face quality and the number of frames analyzed."
        )
    elif verdict_tier == "uncertain":
        interpretation = (
            "The model output is close to the decision boundary. "
            "The result is inconclusive and may require further analysis. "
            "Low face quality, occlusion, or ambiguous content can cause this."
        )
    elif verdict_tier == "danger":
        interpretation = (
            "The model assigns an elevated manipulation probability to this video. "
            "This is consistent with potential deepfake artifacts, though "
            "false positives are possible on compressed or low-quality video."
        )
    else:  # warning — highest
        interpretation = (
            "The model assigns a high manipulation probability to this video. "
            "Strong and consistent deepfake indicators were detected across "
            "multiple frames. Manual review is strongly recommended."
        )

    is_inconclusive = (0.40 <= p_fake < 0.60)

    return {
        "verdict_label": verdict_label,
        "verdict_tier": verdict_tier,
        "manipulation_pct": manipulation_pct,
        "real_pct": real_pct,
        "interpretation": interpretation,
        "is_inconclusive": is_inconclusive,
        "p_fake": round(p_fake, 4),
        "p_real": round(p_real, 4),
    }


# CSS class mapping for Streamlit
TIER_CSS = {
    "safe":      "result-safe",
    "uncertain": "result-uncertain",
    "danger":    "result-danger",
    "warning":   "result-warning",
}

# Emoji mapping for Streamlit
TIER_EMOJI = {
    "safe":      "✅",
    "uncertain": "⚠️",
    "danger":    "🚨",
    "warning":   "🔴",
}
