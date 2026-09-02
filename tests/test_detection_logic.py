from app import aggregate_video_predictions
from pipeline import TemporalAggregator


def test_aggregate_video_predictions_prefers_fake_when_fake_votes_dominate():
    predictions = [
        {"label": "Realism", "confidence": 0.72, "p_real": 0.72, "p_fake": 0.28, "quality": 80.0},
        {"label": "Deepfake", "confidence": 0.80, "p_real": 0.20, "p_fake": 0.80, "quality": 85.0},
        {"label": "Deepfake", "confidence": 0.88, "p_real": 0.12, "p_fake": 0.88, "quality": 90.0},
        {"label": "Deepfake", "confidence": 0.91, "p_real": 0.09, "p_fake": 0.91, "quality": 88.0},
    ]

    final_label, avg_confidence, real_count, fake_count = aggregate_video_predictions(predictions)

    assert final_label == "Deepfake"
    assert fake_count == 3
    assert real_count == 1
    assert avg_confidence > 0.70


def test_aggregate_video_predictions_prefers_real_when_real_votes_dominate():
    predictions = [
        {"label": "Realism", "confidence": 0.92, "p_real": 0.92, "p_fake": 0.08, "quality": 90.0},
        {"label": "Realism", "confidence": 0.85, "p_real": 0.85, "p_fake": 0.15, "quality": 85.0},
        {"label": "Realism", "confidence": 0.78, "p_real": 0.78, "p_fake": 0.22, "quality": 75.0},
        {"label": "Deepfake", "confidence": 0.55, "p_real": 0.45, "p_fake": 0.55, "quality": 40.0},
    ]

    final_label, avg_confidence, real_count, fake_count = aggregate_video_predictions(predictions)

    assert final_label == "Realism"
    assert real_count == 3
    assert avg_confidence > 0.75


def test_temporal_aggregator_empty_predictions():
    res = TemporalAggregator.aggregate([])
    assert res['final_label'] == "Unknown"
    assert res['avg_confidence'] == 0.0
    assert res['total_predictions'] == 0
    assert res['real_count'] == 0
    assert res['fake_count'] == 0

