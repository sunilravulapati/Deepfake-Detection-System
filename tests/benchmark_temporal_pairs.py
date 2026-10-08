import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from custom_fallback import TemporalConsistencyAnalyzer, FallbackConfig

def run_controlled_pairs_test():
    analyzer = TemporalConsistencyAnalyzer()

    # Create realistic base face
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.ellipse(base, (64, 64), (45, 58), 0, 0, 360, (180, 195, 220), -1) # face skin
    # Add facial features
    cv2.circle(base, (45, 52), 6, (70, 50, 40), -1) # left eye
    cv2.circle(base, (83, 52), 6, (70, 50, 40), -1) # right eye
    cv2.line(base, (64, 62), (64, 75), (140, 150, 180), 2) # nose
    cv2.ellipse(base, (64, 94), (18, 5), 0, 0, 360, (120, 110, 160), -1) # mouth
    landmarks = np.array([[45, 52], [83, 52], [64, 75], [46, 94], [82, 94]], dtype=np.float32)

    pairs = []

    # A. Identical -> identical
    pairs.append(('A: Identical -> Identical', 'Zero inconsistency expected (0.00)', base.copy(), base.copy(), landmarks, landmarks))

    # B. Duplicate
    pairs.append(('B: Same frame duplicated', 'Zero inconsistency expected (0.00)', base.copy(), base.copy(), landmarks, landmarks))

    # C. Small translation
    trans_m = np.float32([[1, 0, 3], [0, 1, 2]])
    trans_img = cv2.warpAffine(base, trans_m, (128, 128))
    trans_lm = landmarks + np.array([3, 2], dtype=np.float32)
    pairs.append(('C: Small translation (3,2)', 'Low score after alignment (<0.20)', base.copy(), trans_img, landmarks, trans_lm))

    # D. Brightness change
    bright_img = cv2.convertScaleAbs(base, alpha=1.05, beta=15)
    pairs.append(('D: Brightness change (+15)', 'Low score after normalization (<0.30)', base.copy(), bright_img, landmarks, landmarks))

    # E. Natural face movement (smooth affine shear)
    shear_m = np.float32([[1, 0.04, 0], [0.02, 1, 0]])
    shear_img = cv2.warpAffine(base, shear_m, (128, 128))
    shear_lm = cv2.transform(landmarks.reshape(1, -1, 2), shear_m).reshape(-1, 2)
    pairs.append(('E: Natural face movement', 'Tolerate natural movement (<0.40)', base.copy(), shear_img, landmarks, shear_lm))

    # F. Blinking (closed eyelids)
    blink_img = base.copy()
    cv2.circle(blink_img, (45, 52), 7, (180, 195, 220), -1)
    cv2.line(blink_img, (39, 52), (51, 52), (70, 50, 40), 2) # eyelid slit
    cv2.circle(blink_img, (83, 52), 7, (180, 195, 220), -1)
    cv2.line(blink_img, (77, 52), (89, 52), (70, 50, 40), 2)
    pairs.append(('F: Blinking (closed eyes)', 'Natural blink should not be fake (<0.45)', base.copy(), blink_img, landmarks, landmarks))

    # G. Speaking (open mouth)
    speak_img = base.copy()
    cv2.ellipse(speak_img, (64, 94), (18, 12), 0, 0, 360, (50, 40, 60), -1) # dark open oral cavity
    cv2.rectangle(speak_img, (58, 88), (70, 91), (230, 230, 240), -1) # teeth
    pairs.append(('G: Speaking (open mouth)', 'Natural articulation should not be fake (<0.45)', base.copy(), speak_img, landmarks, landmarks))

    # H. Head rotation (5 degrees in-plane)
    rot_m = cv2.getRotationMatrix2D((64, 64), 5.0, 1.0)
    rot_img = cv2.warpAffine(base, rot_m, (128, 128))
    rot_lm = cv2.transform(landmarks.reshape(1, -1, 2), rot_m).reshape(-1, 2)
    pairs.append(('H: Head rotation (5 deg)', 'Compensate rotation (<0.35)', base.copy(), rot_img, landmarks, rot_lm))

    # I. Artificial localized manipulation/noise (face swap patch seam & noise)
    manip_img = base.copy()
    np.random.seed(42)
    noise_patch = np.random.randint(-40, 40, (40, 40, 3), dtype=np.int16)
    p = manip_img[40:80, 44:84].astype(np.int16) + noise_patch
    manip_img[40:80, 44:84] = np.clip(p, 0, 255).astype(np.uint8)
    cv2.ellipse(manip_img, (64, 64), (25, 25), 0, 0, 360, (255, 100, 100), 1) # seam
    pairs.append(('I: Localized manipulation', 'Elevated score expected (>0.40)', base.copy(), manip_img, landmarks, landmarks))

    # J. Compressed frames (JPEG Q=20)
    _, enc = cv2.imencode('.jpg', base, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
    comp_img = cv2.imdecode(enc, 1)
    pairs.append(('J: Compressed frame (Q=20)', 'Moderate compression artifacts (<0.40)', base.copy(), comp_img, landmarks, landmarks))

    print(f"{'Test':<30} | {'Expected Behavior':<42} | {'Score':<7} | {'Diff':<7} | {'Edge':<7}")
    print("-" * 105)
    for name, expected, f1, f2, lm1, lm2 in pairs:
        analyzer.reset()
        analyzer.analyze_with_details(f1, quality_score=80.0, landmarks=lm1)
        score, avail, tel = analyzer.analyze_with_details(f2, quality_score=80.0, landmarks=lm2)
        diff = tel.get('mean_diff', 0.0)
        edge = tel.get('edge_diff', 0.0)
        print(f"{name:<30} | {expected:<42} | {score:<7.4f} | {diff:<7.2f} | {edge:<7.4f}")

if __name__ == '__main__':
    run_controlled_pairs_test()
