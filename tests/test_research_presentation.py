"""
tests/test_research_presentation.py
====================================
Comprehensive tests verifying the integrity of the research presentation:
1. Research results loading & schemas
2. Confusion matrix exact values
3. Classification metric calculations & consistency
4. Latency values & throughput
5. Verdict interpretation bands
6. Probability display formatting
7. Matplotlib figure generation and PNG/SVG export integrity
8. Per-video ground-truth & error analysis integrity
"""

import json
import pytest
import numpy as np
import matplotlib.pyplot as plt
from research_results import (
    DATASET, CONFUSION, METRICS, LATENCY, PER_VIDEO,
    VERDICT_BANDS, MODEL_INFO, LIMITATIONS,
    get_dataset_summary, get_confusion_matrix, get_metrics,
    get_latency, get_per_video_results, get_verdict_bands,
    get_model_info, get_limitations
)
from verdict_logic import interpret_verdict, TIER_CSS, TIER_EMOJI
import research_dashboard as rd


# ──────────────────────────────────────────────────────────────────────────────
# 1. RESEARCH RESULTS LOADING & REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
def test_dataset_summary_values():
    d = get_dataset_summary()
    assert d["total"] == 6
    assert d["real"] == 3
    assert d["fake"] == 3
    assert d["val"] == 2
    assert d["test"] == 4
    assert d["real"] + d["fake"] == d["total"]
    assert d["val"] + d["test"] == d["total"]
    assert "Preliminary evaluation" in d["note"]


def test_confusion_matrix_exact_values():
    cm = get_confusion_matrix()
    # Positive class = Fake, Negative class = Real
    assert cm["TP"] == 0, "Predicted Fake, Actual Fake must be 0"
    assert cm["TN"] == 1, "Predicted Real, Actual Real must be 1"
    assert cm["FP"] == 1, "Predicted Fake, Actual Real must be 1"
    assert cm["FN"] == 2, "Predicted Real, Actual Fake must be 2"
    assert cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"] == 4, "Total test samples must equal 4"


def test_metric_calculations_and_consistency():
    cm = get_confusion_matrix()
    tp = cm["TP"]
    tn = cm["TN"]
    fp = cm["FP"]
    fn = cm["FN"]
    total = tp + tn + fp + fn

    calc_acc = (tp + tn) / total
    calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    calc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    calc_f1 = (2 * calc_prec * calc_rec / (calc_prec + calc_rec)) if (calc_prec + calc_rec) > 0 else 0.0

    m_a = get_metrics("vit_only")
    m_b = get_metrics("vit_temporal")

    assert pytest.approx(m_a["accuracy"], 0.001) == calc_acc
    assert pytest.approx(m_a["precision"], 0.001) == calc_prec
    assert pytest.approx(m_a["recall"], 0.001) == calc_rec
    assert pytest.approx(m_a["f1"], 0.001) == calc_f1

    # Both configurations must be identical on the 4-video test set
    assert m_a["accuracy"] == m_b["accuracy"] == 0.25
    assert m_a["precision"] == m_b["precision"] == 0.00
    assert m_a["recall"] == m_b["recall"] == 0.00
    assert m_a["f1"] == m_b["f1"] == 0.00


def test_latency_values():
    lat_a = get_latency("vit_only")
    lat_b = get_latency("vit_temporal")

    assert lat_a["vit_inference_ms"] == 88.97
    assert lat_a["end_to_end_ms"] == 153.21
    assert lat_a["fps"] == 6.53

    assert lat_b["vit_inference_ms"] == 89.93
    assert lat_b["end_to_end_ms"] == 152.25
    assert lat_b["fps"] == 6.57

    # Overhead of temporal aggregation is sub-millisecond on inference
    diff_inf = abs(lat_b["vit_inference_ms"] - lat_a["vit_inference_ms"])
    assert diff_inf < 2.0


def test_per_video_error_analysis():
    pv = get_per_video_results()
    assert len(pv) == 4

    v_map = {v["name"]: v for v in pv}
    assert "real_interview_02" in v_map
    assert "real_interview_03" in v_map
    assert "fake_deeperforensics_02" in v_map
    assert "fake_deeperforensics_03" in v_map

    # Check real_interview_02
    v1 = v_map["real_interview_02"]
    assert v1["ground_truth"] == "Real"
    assert v1["prediction"] == "Real"
    assert v1["correct"] is True
    assert v1["p_fake"] == 0.4711
    assert v1["avg_quality"] == 84.36

    # Check real_interview_03 (False Positive)
    v2 = v_map["real_interview_03"]
    assert v2["ground_truth"] == "Real"
    assert v2["prediction"] == "Fake"
    assert v2["correct"] is False
    assert v2["error_type"] == "False Positive"
    assert v2["p_fake"] == 0.6495
    assert v2["avg_quality"] == 82.42

    # Check fake_deeperforensics_02 (False Negative)
    v3 = v_map["fake_deeperforensics_02"]
    assert v3["ground_truth"] == "Fake"
    assert v3["prediction"] == "Real"
    assert v3["correct"] is False
    assert v3["error_type"] == "False Negative"
    assert v3["p_fake"] == 0.3888
    assert v3["avg_quality"] == 54.70

    # Check fake_deeperforensics_03 (False Negative)
    v4 = v_map["fake_deeperforensics_03"]
    assert v4["ground_truth"] == "Fake"
    assert v4["prediction"] == "Real"
    assert v4["correct"] is False
    assert v4["error_type"] == "False Negative"
    assert v4["p_fake"] == 0.4212
    assert v4["avg_quality"] == 55.14


def test_verdict_bands_and_interpretation():
    # Test all 4 bands via interpret_verdict
    # Band 1: < 0.40 -> Likely Authentic
    res1 = interpret_verdict(0.15, 0.85)
    assert res1["verdict_label"] == "Likely Authentic"
    assert res1["verdict_tier"] == "safe"
    assert res1["manipulation_pct"] == 15
    assert res1["real_pct"] == 85
    assert not res1["is_inconclusive"]

    # Band 2: 0.40 <= p < 0.60 -> Inconclusive
    res2 = interpret_verdict(0.48, 0.52)
    assert res2["verdict_label"] == "Inconclusive — May Be Manipulated"
    assert res2["verdict_tier"] == "uncertain"
    assert res2["manipulation_pct"] == 48
    assert res2["real_pct"] == 52
    assert res2["is_inconclusive"]

    # Band 3: 0.60 <= p < 0.80 -> Likely Manipulated
    res3 = interpret_verdict(0.72, 0.28)
    assert res3["verdict_label"] == "Likely Manipulated"
    assert res3["verdict_tier"] == "danger"
    assert res3["manipulation_pct"] == 72
    assert res3["real_pct"] == 28
    assert not res3["is_inconclusive"]

    # Band 4: >= 0.80 -> Highly Likely Manipulated
    res4 = interpret_verdict(0.92, 0.08)
    assert res4["verdict_label"] == "Highly Likely Manipulated"
    assert res4["verdict_tier"] == "warning"
    assert res4["manipulation_pct"] == 92
    assert res4["real_pct"] == 8
    assert not res4["is_inconclusive"]


def test_model_info_and_limitations():
    info = get_model_info()
    assert info["name"] == "prithivMLmods/Deep-Fake-Detector-v2-Model"
    assert info["trained_by_us"] is False
    assert info["fine_tuned"] is False
    assert "Publicly available" in info["attribution"]

    lims = get_limitations()
    assert len(lims) == 8
    assert any("six videos" in l for l in lims)
    assert any("four videos" in l for l in lims)
    assert any("not trained or fine-tuned" in l for l in lims)
    assert any("DeeperForensics" in l for l in lims)


# ──────────────────────────────────────────────────────────────────────────────
# 2. GRAPH DATA INTEGRITY & EXPORT FORMATS
# ──────────────────────────────────────────────────────────────────────────────
def test_all_figures_generate_and_export():
    figures = [
        ("01_class_distribution", rd.fig_class_distribution()),
        ("02_confusion_matrix", rd.fig_confusion_matrix()),
        ("03_metrics_comparison", rd.fig_metrics_comparison()),
        ("04_latency_comparison", rd.fig_latency_comparison()),
        ("05_fps_comparison", rd.fig_fps_comparison()),
        ("06_video_fake_probability", rd.fig_pfake_comparison()),
        ("07_face_quality_comparison", rd.fig_face_quality()),
        ("00_pipeline_architecture", rd.fig_pipeline()),
        ("08_per_video_table", rd.fig_per_video_table()),
    ]

    for name, fig in figures:
        assert isinstance(fig, plt.Figure), f"{name} must return a matplotlib Figure"

        # Test PNG export
        png_bytes = rd.get_fig_bytes(fig, fmt="png", dpi=150)
        assert len(png_bytes) > 1000, f"{name} PNG must be non-empty"
        # PNG magic header bytes
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", f"{name} PNG must have valid header"

        # Test SVG export
        svg_bytes = rd.get_fig_bytes(fig, fmt="svg")
        assert len(svg_bytes) > 500, f"{name} SVG must be non-empty"
        assert b"<svg" in svg_bytes, f"{name} SVG must contain <svg root element"

        plt.close(fig)


def test_probability_display_academic_language():
    # Verify per-video probabilities sum to 1.0 and are within [0, 1]
    for v in get_per_video_results():
        assert 0.0 <= v["p_fake"] <= 1.0
        assert 0.0 <= v["p_real"] <= 1.0
        assert pytest.approx(v["p_fake"] + v["p_real"], abs=1e-3) == 1.0

    res = interpret_verdict(0.4711, 0.5289)
    # Ensure keys are manipulation_pct, not accuracy
    assert "manipulation_pct" in res
    assert "real_pct" in res
    assert "accuracy" not in res
    assert res["manipulation_pct"] == 47
    assert res["real_pct"] == 53


def test_verdict_tier_css_and_emoji():
    for band in get_verdict_bands():
        tier = band["tier"]
        assert tier in TIER_CSS, f"Tier {tier} must have a CSS mapping"
        assert tier in TIER_EMOJI, f"Tier {tier} must have an emoji mapping"
        assert TIER_CSS[tier].startswith("result-")


def test_json_csv_export_schema():
    # Verify export data payload matches all required research metadata
    res = interpret_verdict(0.35, 0.65)
    sample_payload = {
        'verdict_label': res['verdict_label'],
        'verdict_tier': res['verdict_tier'],
        'manipulation_probability': res['p_fake'],
        'real_probability': res['p_real'],
        'manipulation_pct': res['manipulation_pct'],
        'real_pct': res['real_pct'],
        'interpretation': res['interpretation'],
        'is_inconclusive': res['is_inconclusive'],
        'real_count': 12,
        'fake_count': 2,
        'total_predictions': 14,
        'average_quality': 82.5,
        'decision_threshold': 0.50,
        'model': 'prithivMLmods/Deep-Fake-Detector-v2-Model',
    }
    encoded = json.dumps(sample_payload)
    decoded = json.loads(encoded)
    assert decoded['model'] == 'prithivMLmods/Deep-Fake-Detector-v2-Model'
    assert decoded['verdict_label'] == 'Likely Authentic'
    assert decoded['manipulation_pct'] == 35

