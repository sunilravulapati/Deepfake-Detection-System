import pytest
import numpy as np
import cv2
import os

from pipeline import (
    compute_face_quality,
    align_and_crop_face,
    sample_video_frame_indices,
    RobustFaceDetector,
    TemporalAggregator,
    get_detector
)


def test_compute_face_quality_empty():
    res = compute_face_quality(np.zeros((0, 0, 3), dtype=np.uint8))
    assert not res['is_valid']
    assert res['score'] == 0.0


def test_compute_face_quality_tiny():
    tiny = np.ones((20, 20, 3), dtype=np.uint8) * 128
    res = compute_face_quality(tiny, min_size=40)
    assert not res['is_valid']
    assert 'small' in res['reason'].lower()


def test_compute_face_quality_blurry():
    # A completely flat gray image has 0 Laplacian variance
    flat = np.ones((100, 100, 3), dtype=np.uint8) * 120
    res = compute_face_quality(flat, min_blur_var=20.0)
    assert not res['is_valid']
    assert 'blurry' in res['reason'].lower() or 'contrast' in res['reason'].lower()


def test_compute_face_quality_valid_pattern():
    # Create image with textured edges / patterns
    img = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
    res = compute_face_quality(img, min_size=40, min_blur_var=10.0)
    assert res['is_valid']
    assert res['score'] > 20.0


def test_align_and_crop_face_padding():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    box = (100, 100, 100, 100)
    crop, coords = align_and_crop_face(frame, box, padding_ratio=0.25)

    assert crop.shape[0] == crop.shape[1]  # Square
    assert crop.shape[0] >= 100  # Padded larger than raw box
    assert coords[2] >= 100
    assert coords[3] >= 100


def test_sample_video_frame_indices():
    # 24 fps, 240 frames total (10 seconds)
    indices = sample_video_frame_indices(total_frames=240, fps=24.0, max_samples=30, target_sample_fps=3.0)
    assert len(indices) <= 30
    assert indices[0] == 0
    assert indices[-1] < 240
    # Ensure monotonically increasing
    for i in range(len(indices) - 1):
        assert indices[i] < indices[i + 1]


def test_temporal_aggregator_anomaly_detection():
    # Test anomaly detection when a burst of consecutive deepfake frames occurs
    predictions = [
        {"label": "Realism", "confidence": 0.88, "p_real": 0.88, "p_fake": 0.12, "quality": 80.0},
        {"label": "Realism", "confidence": 0.85, "p_real": 0.85, "p_fake": 0.15, "quality": 80.0},
        {"label": "Deepfake", "confidence": 0.95, "p_real": 0.05, "p_fake": 0.95, "quality": 85.0},
        {"label": "Deepfake", "confidence": 0.96, "p_real": 0.04, "p_fake": 0.96, "quality": 85.0},
        {"label": "Deepfake", "confidence": 0.94, "p_real": 0.06, "p_fake": 0.94, "quality": 85.0},
        {"label": "Deepfake", "confidence": 0.93, "p_real": 0.07, "p_fake": 0.93, "quality": 85.0},
        {"label": "Realism", "confidence": 0.84, "p_real": 0.84, "p_fake": 0.16, "quality": 80.0},
        {"label": "Realism", "confidence": 0.82, "p_real": 0.82, "p_fake": 0.18, "quality": 80.0},
    ]
    res = TemporalAggregator.aggregate(predictions)
    assert res['anomaly_detected']
    assert res['fake_count'] == 4
    assert res['is_deepfake']


def test_detector_on_sample_video():
    video_path = '01_02__exit_phone_room__YVGY8LOK.mp4'
    if not os.path.exists(video_path):
        pytest.skip("Sample video not found")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    ret, frame = cap.read()
    assert ret
    cap.release()

    detector = get_detector()
    faces = detector.detect_faces(frame)
    assert len(faces) >= 1
    assert faces[0]['is_valid_quality']
    assert faces[0]['quality'] > 30.0
