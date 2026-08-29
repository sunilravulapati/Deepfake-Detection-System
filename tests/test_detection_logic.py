from app import aggregate_video_predictions


def test_aggregate_video_predictions_prefers_fake_when_fake_votes_dominate():
    predictions = [
        {"label": "Realism", "confidence": 0.72},
        {"label": "Deepfake", "confidence": 0.80},
        {"label": "Deepfake", "confidence": 0.88},
        {"label": "Deepfake", "confidence": 0.91},
    ]

    final_label, avg_confidence, real_count, fake_count = aggregate_video_predictions(predictions)

    assert final_label == "Deepfake"
    assert fake_count == 3
    assert real_count == 1
    assert avg_confidence > 0.0
