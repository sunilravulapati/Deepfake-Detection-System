import os
import sys
import time
import cv2
import pandas as pd
from pipeline import get_detector

def prepare_dataset():
    os.makedirs('dataset/real', exist_ok=True)
    os.makedirs('dataset/fake', exist_ok=True)
    detector = get_detector()

    # 1. Prepare Fake Videos from available DeeperForensics videos
    fake_sources = [
        '01_02__exit_phone_room__YVGY8LOK.mp4',
        '01_02__hugging_happy__YVGY8LOK.mp4'
    ]

    fake_paths = []
    # Clip 1: from exit_phone_room (full clip)
    p1 = 'dataset/fake/fake_deeperforensics_01.mp4'
    if os.path.exists(fake_sources[0]):
        cap = cv2.VideoCapture(fake_sources[0])
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(p1, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for _ in range(150):
            ret, frame = cap.read()
            if not ret: break
            writer.write(frame)
        cap.release()
        writer.release()
        fake_paths.append(p1)

    # Clip 2 & 3: from hugging_happy (first segment and second segment)
    if os.path.exists(fake_sources[1]):
        cap = cv2.VideoCapture(fake_sources[1])
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        p2 = 'dataset/fake/fake_deeperforensics_02.mp4'
        writer2 = cv2.VideoWriter(p2, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for _ in range(150):
            ret, frame = cap.read()
            if not ret: break
            writer2.write(frame)
        writer2.release()
        fake_paths.append(p2)

        p3 = 'dataset/fake/fake_deeperforensics_03.mp4'
        writer3 = cv2.VideoWriter(p3, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        # Skip 50 frames to get different poses
        for _ in range(50):
            cap.read()
        for _ in range(150):
            ret, frame = cap.read()
            if not ret: break
            writer3.write(frame)
        writer3.release()
        fake_paths.append(p3)
        cap.release()

    # 2. Record authentic real video clips from webcam
    real_paths = []
    cap_cam = cv2.VideoCapture(0)
    if cap_cam.isOpened():
        print("Recording 3 authentic real video clips from webcam (approx 5s each)...")
        w_cam = int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h_cam = int(cap_cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_cam = 20.0

        for clip_idx in range(1, 4):
            real_p = f'dataset/real/real_interview_0{clip_idx}.mp4'
            writer = cv2.VideoWriter(real_p, cv2.VideoWriter_fourcc(*'mp4v'), fps_cam, (w_cam, h_cam))
            frames_recorded = 0
            start_t = time.time()
            while frames_recorded < 100:
                ret, frame = cap_cam.read()
                if not ret: break
                writer.write(frame)
                frames_recorded += 1
                time.sleep(0.04)  # ~25 fps pacing
            writer.release()
            real_paths.append(real_p)
            print(f"Recorded authentic clip: {real_p} ({frames_recorded} frames)")
            time.sleep(0.5)
        cap_cam.release()
    else:
        print("WARNING: Webcam not opened, generating synthetic authentic baseline clips from real screenshot faces...")
        # Fallback if webcam unavailable
        s_img = cv2.imread('Screenshot 2025-10-21 172924.png')
        if s_img is not None:
            faces = detector.detect_faces(s_img)
            if faces:
                face_crop = faces[0]['image']
                for clip_idx in range(1, 4):
                    real_p = f'dataset/real/real_interview_0{clip_idx}.mp4'
                    w, h = 320, 320
                    resized = cv2.resize(face_crop, (w, h))
                    writer = cv2.VideoWriter(real_p, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                    for f_no in range(100):
                        # subtle natural micro-motion
                        dx = int(round(np.sin(f_no / 10.0) * 2))
                        dy = int(round(np.cos(f_no / 12.0) * 2))
                        M = np.float32([[1, 0, dx], [0, 1, dy]])
                        shifted = cv2.warpAffine(resized, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                        writer.write(shifted)
                    writer.release()
                    real_paths.append(real_p)

    # 3. Create metadata CSV with validation and test splits
    rows = []
    # Balance splits: 1 val real, 1 val fake; 2 test real, 2 test fake
    for idx, p in enumerate(real_paths):
        split = "val" if idx == 0 else "test"
        rows.append({'video_path': p, 'label': 'real', 'split': split})
    for idx, p in enumerate(fake_paths):
        split = "val" if idx == 0 else "test"
        rows.append({'video_path': p, 'label': 'fake', 'split': split})

    df = pd.DataFrame(rows)
    df.to_csv('dataset/metadata.csv', index=False)
    print("\nDataset preparation complete:")
    print(df.to_string())

if __name__ == '__main__':
    prepare_dataset()
