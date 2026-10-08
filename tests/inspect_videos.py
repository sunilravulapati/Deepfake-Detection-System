import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from pipeline import get_detector, predict_deepfake
from transformers import AutoImageProcessor, AutoModelForImageClassification

proc = AutoImageProcessor.from_pretrained('prithivMLmods/Deep-Fake-Detector-v2-Model')
model = AutoModelForImageClassification.from_pretrained('prithivMLmods/Deep-Fake-Detector-v2-Model')
detector = get_detector()

for v in ['01_02__exit_phone_room__YVGY8LOK.mp4', '01_02__hugging_happy__YVGY8LOK.mp4']:
    cap = cv2.VideoCapture(v)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=== Video {v} (total {total} frames) ===")
    for idx in range(0, min(total, 120), 15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        faces = detector.detect_faces(frame)
        if faces:
            lbl, conf, p_real, p_fake = predict_deepfake(faces[0]['image'], proc, model)
            q = faces[0]['quality']
            print(f"Frame {idx:3d}: {lbl:<10} | p_fake={p_fake:.4f} | conf={conf:.4f} | Q={q:.1f}")
    cap.release()
