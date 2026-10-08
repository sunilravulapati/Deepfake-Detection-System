import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from custom_fallback import StructuralAnalyzer, FallbackConfig
from pipeline import get_detector

def extract_faces_from_video(video_path, max_faces=10):
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = get_detector()
    crops = []
    step = max(1, total // max_faces)
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        faces = detector.detect_faces(frame, max_faces=1)
        if faces:
            crops.append((faces[0]['image'], faces[0].get('landmarks')))
        if len(crops) >= max_faces:
            break
    cap.release()
    return crops

def run_structural_benchmark():
    analyzer = StructuralAnalyzer()
    detector = get_detector()

    # Load real faces from screenshots
    real_faces = []
    for s_path in ['Screenshot 2025-10-21 172924.png', 'Screenshot 2025-10-21 173007.png']:
        if os.path.exists(s_path):
            img = cv2.imread(s_path)
            if img is not None:
                faces = detector.detect_faces(img, max_faces=1)
                if faces:
                    real_faces.append((faces[0]['image'], faces[0].get('landmarks')))

    # Load fake faces from videos
    fake_faces = []
    for v_path in ['01_02__exit_phone_room__YVGY8LOK.mp4', '01_02__hugging_happy__YVGY8LOK.mp4']:
        fake_faces.extend(extract_faces_from_video(v_path, max_faces=8))

    print(f"Loaded {len(real_faces)} real face crops and {len(fake_faces)} manipulated face crops.")

    # Base reference face for controlled variants
    ref_face, ref_lm = real_faces[0] if real_faces else (fake_faces[0] if fake_faces else (np.ones((128,128,3), dtype=np.uint8)*128, None))

    # Generate condition variants
    categories = {}
    
    # 1. Natural real faces
    categories['Natural Real'] = [f for f, lm in real_faces] if real_faces else [ref_face]
    
    # 2. Manipulated / fake faces
    categories['Manipulated Fake'] = [f for f, lm in fake_faces] if fake_faces else [ref_face]

    # 3. Blurred faces
    categories['Blurred'] = [cv2.GaussianBlur(ref_face, (15, 15), 5.0) for _ in range(5)]

    # 4. Low-light faces
    categories['Low Illumination'] = [cv2.convertScaleAbs(ref_face, alpha=0.35, beta=0) for _ in range(5)]

    # 5. Rotated faces (15-20 degrees)
    rot_faces = []
    for angle in [-20.0, -15.0, 10.0, 15.0, 20.0]:
        h, w = ref_face.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rot_faces.append(cv2.warpAffine(ref_face, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    categories['Rotated'] = rot_faces

    # 6. Compressed faces
    comp_faces = []
    for q in [15, 20, 25, 30, 35]:
        _, enc = cv2.imencode('.jpg', ref_face, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        comp_faces.append(cv2.imdecode(enc, 1))
    categories['Compressed'] = comp_faces

    print("\n" + "=" * 85)
    print(f"{'Condition':<20} | {'Count':<6} | {'Mean S_struct':<14} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'Bounded [0,1]':<12}")
    print("=" * 85)

    all_scores = {}
    for cat_name, face_list in categories.items():
        scores = []
        for face in face_list:
            s, tel = analyzer.analyze_with_details(face)
            scores.append(s)
            assert 0.0 <= s <= 1.0, f"Score out of bounds: {s}"
        
        arr = np.array(scores)
        all_scores[cat_name] = scores
        mean_s = float(np.mean(arr))
        std_s = float(np.std(arr))
        min_s = float(np.min(arr))
        max_s = float(np.max(arr))
        bounded = "PASS" if (min_s >= 0.0 and max_s <= 1.0) else "FAIL"

        print(f"{cat_name:<20} | {len(face_list):<6} | {mean_s:<14.4f} | {std_s:<8.4f} | {min_s:<8.4f} | {max_s:<8.4f} | {bounded:<12}")

    print("=" * 85 + "\n")
    return all_scores

if __name__ == '__main__':
    run_structural_benchmark()
