import pytest
import numpy as np
import cv2
from custom_fallback import (
    SpatialArtifactAnalyzer,
    TemporalConsistencyAnalyzer,
    StructuralAnalyzer,
    QualityAwareFusion,
    CustomFallbackDetector,
    FallbackConfig,
    DecisionMode
)
from pipeline import (
    TemporalAggregator,
    predict_deepfake_with_fallback
)


def create_dummy_face(height=100, width=100, pattern="smooth"):
    """Helper to generate dummy face images for deterministic testing."""
    if pattern == "smooth":
        # Smooth gradient simulating real face skin tone
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                img[i, j] = [120 + int(i * 0.3), 140 + int(j * 0.2), 180]
        return img
    elif pattern == "checkerboard":
        # High frequency pattern with edge discontinuities
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[::4, ::4] = 255
        img[1::4, 1::4] = 200
        return img
    elif pattern == "blank":
        return np.ones((height, width, 3), dtype=np.uint8) * 128
    else:
        return np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)


# 1. Spatial cue returns normalized value
def test_spatial_cue_returns_normalized_value():
    analyzer = SpatialArtifactAnalyzer()
    img = create_dummy_face(120, 120, "checkerboard")
    score = analyzer.analyze(img, quality_score=80.0)
    assert 0.0 <= score <= 1.0

    # Also test empty/tiny image edge cases
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert analyzer.analyze(empty) == 0.5
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    assert analyzer.analyze(tiny) == 0.5


# 2. Temporal cue handles first frame
def test_temporal_cue_handles_first_frame():
    analyzer = TemporalConsistencyAnalyzer()
    img = create_dummy_face(120, 120, "smooth")
    score, available = analyzer.analyze(img, quality_score=75.0)
    # First frame must be neutral score and marked not available
    assert score == 0.5
    assert not available


# 3. Temporal cue handles identical frames
def test_temporal_cue_handles_identical_frames():
    analyzer = TemporalConsistencyAnalyzer()
    img = create_dummy_face(120, 120, "smooth")
    # First frame
    analyzer.analyze(img, quality_score=80.0)
    # Second identical frame
    score, available = analyzer.analyze(img, quality_score=80.0)
    assert available
    assert score == 0.0  # Zero inconsistency between identical frames


# 4. Structural cue returns normalized value
def test_structural_cue_returns_normalized_value():
    analyzer = StructuralAnalyzer()
    img = create_dummy_face(120, 120, "smooth")
    score = analyzer.analyze(img, quality_score=85.0)
    assert 0.0 <= score <= 1.0

    # Edge cases
    assert analyzer.analyze(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.5


# 5. Quality score affects reliability
def test_quality_score_affects_reliability():
    fusion = QualityAwareFusion()
    # High quality face (Q=95) vs low quality face (Q=10)
    # When ViT is uncertain (confidence=0.1)
    _, high_q_weights = fusion.fuse(
        s_vit=0.55, vit_confidence=0.1,
        s_spatial=0.6, s_temporal=0.6, s_structural=0.6,
        quality_score=95.0, temporal_available=True
    )

    _, low_q_weights = fusion.fuse(
        s_vit=0.55, vit_confidence=0.1,
        s_spatial=0.6, s_temporal=0.6, s_structural=0.6,
        quality_score=10.0, temporal_available=True
    )

    # In high quality, the supporting cues have significantly higher combined weight
    cues_high_q = high_q_weights['w_spatial'] + high_q_weights['w_temporal'] + high_q_weights['w_structural']
    cues_low_q = low_q_weights['w_spatial'] + low_q_weights['w_temporal'] + low_q_weights['w_structural']

    assert cues_high_q > cues_low_q
    # And correspondingly, uncertain ViT has lower proportion of weight in high Q
    assert high_q_weights['w_vit'] < low_q_weights['w_vit']


# 6. Fusion weights normalize correctly
def test_fusion_weights_normalize_correctly():
    fusion = QualityAwareFusion()
    for q in [0.0, 25.0, 50.0, 80.0, 100.0]:
        for temp_avail in [True, False]:
            for s_vit in [None, 0.4, 0.8]:
                vit_conf = abs(s_vit - 0.5) * 2.0 if s_vit is not None else None
                s_final, weights = fusion.fuse(
                    s_vit=s_vit, vit_confidence=vit_conf,
                    s_spatial=0.3, s_temporal=0.4, s_structural=0.5,
                    quality_score=q, temporal_available=temp_avail
                )
                total_w = sum(weights.values())
                assert pytest.approx(total_w, abs=1e-3) == 1.0
                assert 0.0 <= s_final <= 1.0


# 7. High-confidence ViT activates PRIMARY_VIT
def test_high_confidence_vit_activates_primary_vit():
    detector = CustomFallbackDetector(FallbackConfig(vit_confidence_threshold=0.70))
    face = create_dummy_face(100, 100)

    # p_fake = 0.95 -> confidence = |0.95 - 0.5| * 2 = 0.90 >= 0.70
    vit_res = ("Deepfake", 0.95, 0.05, 0.95)
    result = detector.evaluate_frame(face, vit_result=vit_res, quality_score=80.0)

    assert result['decision_mode'] == DecisionMode.PRIMARY_VIT.value
    assert result['final_label'] == "Deepfake"
    assert result['p_fake'] == 0.95
    assert result['final_confidence'] == 0.90


# 8. Uncertain ViT activates FALLBACK_FUSION
def test_uncertain_vit_activates_fallback_fusion():
    detector = CustomFallbackDetector(FallbackConfig(vit_confidence_threshold=0.70))
    face = create_dummy_face(100, 100)

    # p_fake = 0.55 -> confidence = |0.55 - 0.5| * 2 = 0.10 < 0.70
    vit_res = ("Deepfake", 0.55, 0.45, 0.55)
    result = detector.evaluate_frame(face, vit_result=vit_res, quality_score=80.0)

    assert result['decision_mode'] == DecisionMode.FALLBACK_FUSION.value
    assert 0.0 <= result['final_score'] <= 1.0
    assert result['weights']['w_vit'] < 1.0


# 9. ViT failure activates FALLBACK_ONLY
def test_vit_failure_activates_fallback_only():
    detector = CustomFallbackDetector()
    face = create_dummy_face(100, 100)

    # ViT result is None (simulating inference error, missing model, CUDA OOM, etc.)
    result = detector.evaluate_frame(face, vit_result=None, quality_score=75.0)

    assert result['decision_mode'] == DecisionMode.FALLBACK_ONLY.value
    assert result['s_vit'] is None
    assert result['weights']['w_vit'] == 0.0
    assert 0.0 <= result['final_score'] <= 1.0
    assert result['final_label'] in ("Realism", "Deepfake")


# 10. Missing cue does not crash the system
def test_missing_cue_does_not_crash():
    detector = CustomFallbackDetector()
    # Empty image simulates complete failure of image decoding/cropping
    empty_face = np.zeros((0, 0, 3), dtype=np.uint8)

    result = detector.evaluate_frame(empty_face, vit_result=None, quality_score=0.0)
    assert result['decision_mode'] == DecisionMode.FALLBACK_ONLY.value
    assert 0.0 <= result['final_score'] <= 1.0


# 11. Final score remains in [0, 1]
def test_final_score_remains_in_unit_interval():
    detector = CustomFallbackDetector()
    for mode_type in ["confident_real", "confident_fake", "uncertain", "failed"]:
        face = create_dummy_face(80, 80)
        if mode_type == "confident_real":
            vit_res = ("Realism", 0.98, 0.98, 0.02)
        elif mode_type == "confident_fake":
            vit_res = ("Deepfake", 0.95, 0.05, 0.95)
        elif mode_type == "uncertain":
            vit_res = ("Realism", 0.52, 0.52, 0.48)
        else:
            vit_res = None

        res = detector.evaluate_frame(face, vit_result=vit_res, quality_score=50.0)
        assert 0.0 <= res['final_score'] <= 1.0
        assert 0.0 <= res['p_real'] <= 1.0
        assert 0.0 <= res['p_fake'] <= 1.0
        assert pytest.approx(res['p_real'] + res['p_fake'], abs=1e-3) == 1.0


# 12. Temporal aggregation still works with fallback telemetry
def test_temporal_aggregation_with_fallback_telemetry():
    predictions = [
        {
            'label': 'Realism', 'confidence': 0.90, 'p_real': 0.90, 'p_fake': 0.10,
            'quality': 80.0, 'decision_mode': 'PRIMARY_VIT', 'vit_confidence': 0.80
        },
        {
            'label': 'Deepfake', 'confidence': 0.60, 'p_real': 0.40, 'p_fake': 0.60,
            'quality': 75.0, 'decision_mode': 'FALLBACK_FUSION', 'vit_confidence': 0.20
        },
        {
            'label': 'Deepfake', 'confidence': 0.70, 'p_real': 0.30, 'p_fake': 0.70,
            'quality': 70.0, 'decision_mode': 'FALLBACK_ONLY', 'vit_confidence': 0.0
        }
    ]

    res = TemporalAggregator.aggregate(predictions)
    assert res['total_predictions'] == 3
    assert res['fallback_frames'] == 2
    assert res['mode_counts']['PRIMARY_VIT'] == 1
    assert res['mode_counts']['FALLBACK_FUSION'] == 1
    assert res['mode_counts']['FALLBACK_ONLY'] == 1
    assert res['average_quality'] > 0
    assert 'video_fake_probability' in res
    assert 'video_real_probability' in res


# 13. Webcam pipeline integration does not crash
def test_webcam_pipeline_integration_no_crash():
    # Simulate webcam frame sequence passing through predict_deepfake_with_fallback
    detector = CustomFallbackDetector()
    frame1 = create_dummy_face(100, 100, "smooth")
    frame2 = create_dummy_face(100, 100, "checkerboard")

    # Frame 1: ViT unavailable (e.g. testing fallback in webcam)
    res1 = predict_deepfake_with_fallback(frame1, None, None, quality_score=60.0, fallback_detector=detector)
    assert res1['decision_mode'] == DecisionMode.FALLBACK_ONLY.value

    # Frame 2: temporal analyzer should now have history
    res2 = predict_deepfake_with_fallback(frame2, None, None, quality_score=65.0, fallback_detector=detector)
    assert res2['decision_mode'] == DecisionMode.FALLBACK_ONLY.value
    assert res2['temporal_available']


# 14. Video pipeline integration does not crash
def test_video_pipeline_integration_no_crash():
    # Sequence of multiple frames through temporal aggregation
    detector = CustomFallbackDetector()
    sampled_frames = [create_dummy_face(90, 90, "smooth") for _ in range(5)]

    preds = []
    for idx, f in enumerate(sampled_frames):
        eval_res = detector.evaluate_frame(f, vit_result=("Realism", 0.85, 0.85, 0.15), quality_score=80.0)
        preds.append({
            'frame_index': idx,
            'timestamp': idx * 0.33,
            'label': eval_res['final_label'],
            'confidence': eval_res['final_confidence'],
            'p_real': eval_res['p_real'],
            'p_fake': eval_res['p_fake'],
            'quality': eval_res['quality'],
            'decision_mode': eval_res['decision_mode']
        })

    agg = TemporalAggregator.aggregate(preds)
    assert agg['final_label'] == "Realism"
    assert agg['total_predictions'] == 5
    assert agg['fallback_frames'] == 0


# 15. Structural analyzer circular statistics
def test_structural_circular_statistics():
    analyzer = StructuralAnalyzer()
    img_real = create_dummy_face(128, 128, "smooth")
    score, details = analyzer.analyze_with_details(img_real)

    assert 0.0 <= score <= 1.0
    assert 'circ_var' in details
    assert 'local_incoherence' in details
    assert 'circular_dispersion' in details
    assert 0.0 <= details['circ_var'] <= 1.0
    assert 0.0 <= details['local_incoherence'] <= 1.0
    assert 0.0 <= details['circular_dispersion'] <= 1.0


# 16. Structural analyzer landmark canonical symmetry
def test_structural_landmark_canonical_symmetry():
    analyzer = StructuralAnalyzer()
    dummy_face = create_dummy_face(128, 128, "smooth")
    # Perfectly symmetric dummy landmarks
    landmarks = np.array([
        [38.0, 52.0],
        [90.0, 52.0],
        [64.0, 72.0],
        [44.0, 92.0],
        [84.0, 92.0]
    ], dtype=np.float32)

    score_sym, details_sym = analyzer.analyze_with_details(dummy_face, landmarks=landmarks)

    # Now create an asymmetric manipulated graft
    asym_face = dummy_face.copy()
    asym_face[60:110, 64:120] = np.random.randint(0, 255, (50, 56, 3), dtype=np.uint8)
    score_asym, details_asym = analyzer.analyze_with_details(asym_face, landmarks=landmarks)

    assert details_asym['norm_asymmetry'] > details_sym['norm_asymmetry']
    assert score_asym > score_sym


# 17. Structural analyzer meaningful variation across inputs
def test_structural_meaningful_variation():
    analyzer = StructuralAnalyzer()

    # 1. Real smooth face
    real_face = create_dummy_face(128, 128, "smooth")
    s_real, det_real = analyzer.analyze_with_details(real_face)

    # 2. Manipulated face (asymmetric noise patch)
    manip_face = real_face.copy()
    manip_face[50:100, 60:110] = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    s_manip, det_manip = analyzer.analyze_with_details(manip_face)

    # 3. Random noise image
    noise_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    s_noise, det_noise = analyzer.analyze_with_details(noise_img)

    # 4. Blank image
    blank_img = np.zeros((128, 128, 3), dtype=np.uint8)
    s_blank, det_blank = analyzer.analyze_with_details(blank_img)

    # Crucial check: structural score is NOT frozen at 0.7000!
    assert s_real != 0.7000 or s_manip != 0.7000 or s_noise != 0.7000
    assert s_real < s_noise
    assert s_real < s_manip
    assert s_blank < s_noise
    # Meaningful spread between real face and pure noise
    assert (s_noise - s_real) > 0.30


# 18. Structural analyzer intermediate logging
def test_structural_logs_intermediate_values():
    analyzer = StructuralAnalyzer()
    img = create_dummy_face(128, 128, "smooth")
    score, details = analyzer.analyze_with_details(img)

    expected_keys = [
        'is_aligned', 'mean_diff', 'norm_asymmetry',
        'circ_var', 'local_incoherence', 'circular_dispersion',
        'edge_density', 'edge_anomaly', 'raw_score', 's_structural'
    ]
    for key in expected_keys:
        assert key in details, f"Missing key {key} in structural telemetry"
        assert isinstance(details[key], (float, int, bool))


# 19. Temporal analyzer 1-frame, 4-frame, and 8-frame intervals
def test_temporal_intervals_and_ordinary_motion():
    analyzer = TemporalConsistencyAnalyzer()
    base_frame = create_dummy_face(128, 128, "smooth")

    # Simulate ordinary natural head motion (subtle rotation and translation)
    h, w = base_frame.shape[:2]
    m_shift = np.float32([[1, 0, 3], [0, 1, 2]])
    shifted_frame = cv2.warpAffine(base_frame, m_shift, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Interval 1 (42ms)
    analyzer.reset()
    analyzer.analyze(base_frame, frame_interval=1)
    s_1, avail_1 = analyzer.analyze(shifted_frame, frame_interval=1)
    assert avail_1
    assert s_1 <= 0.35  # Must not blow up to 0.70 on small ordinary motion

    # Interval 4 (167ms)
    analyzer.reset()
    analyzer.analyze(base_frame, frame_interval=4)
    s_4, avail_4 = analyzer.analyze(shifted_frame, frame_interval=4)
    assert avail_4
    assert s_4 <= 0.35

    # Interval 8 (333ms)
    analyzer.reset()
    analyzer.analyze(base_frame, frame_interval=8)
    s_8, avail_8 = analyzer.analyze(shifted_frame, frame_interval=8)
    assert avail_8
    assert s_8 <= 0.35


# 20. Temporal analyzer detects synthesis jitter vs stillness
def test_temporal_detects_synthesis_jitter():
    analyzer = TemporalConsistencyAnalyzer()
    f1 = create_dummy_face(128, 128, "smooth")
    f2_steady = f1.copy()
    f2_jitter = f1.copy()
    # High-frequency synthesis jitter
    f2_jitter[40:80, 40:80] = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)

    analyzer.reset()
    analyzer.analyze(f1)
    s_steady, _ = analyzer.analyze(f2_steady)

    analyzer.reset()
    analyzer.analyze(f1)
    s_jitter, _ = analyzer.analyze(f2_jitter)

    assert s_jitter > s_steady
    assert s_steady == 0.0
    assert s_jitter > 0.20


# 21. predict_deepfake_with_fallback supports landmarks, frame_interval, and kwargs
def test_predict_deepfake_with_fallback_signature_and_kwargs():
    detector = CustomFallbackDetector()
    face = create_dummy_face(100, 100, "smooth")
    landmarks = np.array([[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]], dtype=np.float32)

    res = predict_deepfake_with_fallback(
        face, None, None,
        quality_score=75.0,
        fallback_detector=detector,
        landmarks=landmarks,
        frame_interval=4,
        extra_arbitrary_param="safe"
    )
    assert 'final_label' in res
    assert 'final_confidence' in res
    assert res['decision_mode'] == DecisionMode.FALLBACK_ONLY.value


# 22. Controlled pair: translation and brightness invariance
def test_temporal_translation_and_brightness_invariance():
    analyzer = TemporalConsistencyAnalyzer()
    base = create_dummy_face(128, 128, "smooth")
    lms = np.array([[38, 51], [89, 51], [64, 71], [43, 92], [84, 92]], dtype=np.float32)

    # Translation
    trans_img = cv2.warpAffine(base, np.float32([[1, 0, 3], [0, 1, 2]]), (128, 128))
    analyzer.reset()
    analyzer.analyze(base, landmarks=lms)
    s_trans, _ = analyzer.analyze(trans_img, landmarks=lms + [3, 2])
    assert s_trans < 0.15

    # Brightness shift
    bright_img = cv2.convertScaleAbs(base, alpha=1.0, beta=20)
    analyzer.reset()
    analyzer.analyze(base, landmarks=lms)
    s_bright, _ = analyzer.analyze(bright_img, landmarks=lms)
    assert s_bright < 0.20


# 23. Controlled pair: natural articulation (speech) is not classified as fake
def test_temporal_speech_not_classified_as_fake():
    analyzer = TemporalConsistencyAnalyzer()
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.ellipse(base, (64, 64), (45, 58), 0, 0, 360, (180, 195, 220), -1)
    cv2.circle(base, (45, 52), 6, (70, 50, 40), -1)
    cv2.circle(base, (83, 52), 6, (70, 50, 40), -1)
    cv2.line(base, (64, 62), (64, 75), (140, 150, 180), 2)
    cv2.ellipse(base, (64, 94), (18, 5), 0, 0, 360, (120, 110, 160), -1)
    lms = np.array([[45, 52], [83, 52], [64, 75], [46, 94], [82, 94]], dtype=np.float32)

    speak_img = base.copy()
    cv2.ellipse(speak_img, (64, 94), (18, 12), 0, 0, 360, (50, 40, 60), -1)
    cv2.rectangle(speak_img, (58, 88), (70, 91), (230, 230, 240), -1)

    analyzer.reset()
    analyzer.analyze(base, landmarks=lms)
    s_speak, _ = analyzer.analyze(speak_img, landmarks=lms)
    assert s_speak < 0.45  # Must not classify natural speech as deepfake


# 24. Controlled pair: blinking is tolerated
def test_temporal_blinking_is_tolerated():
    analyzer = TemporalConsistencyAnalyzer()
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.ellipse(base, (64, 64), (45, 58), 0, 0, 360, (180, 195, 220), -1)
    cv2.circle(base, (45, 52), 6, (70, 50, 40), -1)
    cv2.circle(base, (83, 52), 6, (70, 50, 40), -1)
    lms = np.array([[45, 52], [83, 52], [64, 75], [46, 94], [82, 94]], dtype=np.float32)

    blink_img = base.copy()
    cv2.circle(blink_img, (45, 52), 7, (180, 195, 220), -1)
    cv2.line(blink_img, (39, 52), (51, 52), (70, 50, 40), 2)
    cv2.circle(blink_img, (83, 52), 7, (180, 195, 220), -1)
    cv2.line(blink_img, (77, 52), (89, 52), (70, 50, 40), 2)

    analyzer.reset()
    analyzer.analyze(base, landmarks=lms)
    s_blink, _ = analyzer.analyze(blink_img, landmarks=lms)
    assert s_blink < 0.45


# 25. Controlled pair: head rotation compensated by affine alignment
def test_temporal_head_rotation_compensated():
    analyzer = TemporalConsistencyAnalyzer()
    base = create_dummy_face(128, 128, "smooth")
    lms = np.array([[38, 51], [89, 51], [64, 71], [43, 92], [84, 92]], dtype=np.float32)

    rot_m = cv2.getRotationMatrix2D((64, 64), 5.0, 1.0)
    rot_img = cv2.warpAffine(base, rot_m, (128, 128))
    rot_lm = cv2.transform(lms.reshape(1, -1, 2), rot_m).reshape(-1, 2)

    analyzer.reset()
    analyzer.analyze(base, landmarks=lms)
    s_rot, _ = analyzer.analyze(rot_img, landmarks=rot_lm)
    assert s_rot < 0.25


# 26. Controlled pair: heavy compression robustness
def test_temporal_compression_robustness():
    analyzer = TemporalConsistencyAnalyzer()
    base = create_dummy_face(128, 128, "smooth")
    lms = np.array([[38, 51], [89, 51], [64, 71], [43, 92], [84, 92]], dtype=np.float32)

    _, enc = cv2.imencode('.jpg', base, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
    comp_img = cv2.imdecode(enc, 1)

    analyzer.reset()
    analyzer.analyze(base, landmarks=lms)
    s_comp, _ = analyzer.analyze(comp_img, landmarks=lms)
    assert s_comp < 0.35

