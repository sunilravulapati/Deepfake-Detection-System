import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np

def evaluate_pair(f1, f2, lm1, lm2, target_size=(128, 128), frame_interval=1):
    def align(gray, lms):
        if lms is not None and len(lms) >= 3:
            M, _ = cv2.estimateAffinePartial2D(
                lms,
                np.array([[38.2946, 51.6963], [89.7054, 51.6963], [64.0, 71.7366], [43.8340, 92.3655], [84.1660, 92.3655]], dtype=np.float32)
            )
            if M is not None:
                return cv2.warpAffine(gray, M, target_size, borderMode=cv2.BORDER_REFLECT)
        return cv2.resize(gray, target_size)

    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if len(f1.shape) == 3 else f1
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if len(f2.shape) == 3 else f2

    c1 = align(g1, lm1)
    c2 = align(g2, lm2)

    # Identical check
    if np.array_equal(c1, c2):
        return 0.0, 0.0, 0.0

    # Illumination adjustment anchored to the stable rigid face (eyes and nose bridge)
    r1 = c1[25:80, 25:103]
    r2 = c2[25:80, 25:103]
    m1 = float(np.mean(r1))
    m2 = float(np.mean(r2))
    n1 = c1
    n2 = np.clip(c2.astype(np.float32) - (m2 - m1), 0, 255).astype(np.uint8)

    # Subpixel phase correlation anchored strictly to the rigid face to prevent speech/jaw bias
    shift, _ = cv2.phaseCorrelate(r1.astype(np.float32), r2.astype(np.float32))
    dx, dy = shift
    if abs(dx) < 16.0 and abs(dy) < 16.0:
        m_trans = np.float32([[1, 0, dx], [0, 1, dy]])
        n1_aligned = cv2.warpAffine(n1, m_trans, target_size, borderMode=cv2.BORDER_REFLECT)
    else:
        n1_aligned = n1

    h, w = target_size
    diff = cv2.absdiff(n2, n1_aligned)

    # Define masks: rigid (eyes/nose bridge) and articulation (mouth)
    rigid_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(rigid_mask, (64, 60), (32, 22), 0, 0, 360, 255, -1)

    articul_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(articul_mask, (64, 94), (24, 15), 0, 0, 360, 255, -1)

    diff_rigid = float(np.mean(diff[rigid_mask > 0]))
    diff_articul = float(np.mean(diff[articul_mask > 0]))

    # Effective residual with natural articulation gating
    # In authentic human video, rigid anchor remains stable even when articulation is active
    articul_tolerated = min(diff_articul, diff_rigid * 2.5 + 8.0)
    effective_diff = 0.70 * diff_rigid + 0.30 * articul_tolerated

    # Edge discrepancy using dilated edges to avoid micro-misalignment penalty
    canny1 = cv2.Canny(n1_aligned, 50, 150)
    canny2 = cv2.Canny(n2, 50, 150)
    dilated1 = cv2.dilate(canny1, np.ones((3, 3), np.uint8))
    dilated2 = cv2.dilate(canny2, np.ones((3, 3), np.uint8))
    # Unmatched edge points in rigid and boundary zones
    unmatched1 = np.logical_and(canny1 > 0, dilated2 == 0)
    unmatched2 = np.logical_and(canny2 > 0, dilated1 == 0)
    unmatched = np.logical_or(unmatched1, unmatched2)
    edge_diff = float(np.mean(unmatched[rigid_mask > 0]))

    dynamic_thresh = 28.0 + max(0, frame_interval - 1) * 2.5
    norm_diff = float(np.clip(effective_diff / dynamic_thresh, 0.0, 1.0))
    raw_score = 0.75 * norm_diff + 0.25 * float(np.clip(edge_diff * 6.0, 0.0, 1.0))
    score = float(np.clip(raw_score, 0.0, 1.0))

    return score, effective_diff, edge_diff

def main():
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.ellipse(base, (64, 64), (45, 58), 0, 0, 360, (180, 195, 220), -1)
    cv2.circle(base, (45, 52), 6, (70, 50, 40), -1)
    cv2.circle(base, (83, 52), 6, (70, 50, 40), -1)
    cv2.line(base, (64, 62), (64, 75), (140, 150, 180), 2)
    cv2.ellipse(base, (64, 94), (18, 5), 0, 0, 360, (120, 110, 160), -1)
    landmarks = np.array([[45, 52], [83, 52], [64, 75], [46, 94], [82, 94]], dtype=np.float32)

    # Pairs
    shear_m = np.float32([[1, 0.04, 0], [0.02, 1, 0]])
    rot_m = cv2.getRotationMatrix2D((64, 64), 5.0, 1.0)
    
    blink_img = base.copy()
    cv2.circle(blink_img, (45, 52), 7, (180, 195, 220), -1)
    cv2.line(blink_img, (39, 52), (51, 52), (70, 50, 40), 2)
    cv2.circle(blink_img, (83, 52), 7, (180, 195, 220), -1)
    cv2.line(blink_img, (77, 52), (89, 52), (70, 50, 40), 2)

    speak_img = base.copy()
    cv2.ellipse(speak_img, (64, 94), (18, 12), 0, 0, 360, (50, 40, 60), -1)
    cv2.rectangle(speak_img, (58, 88), (70, 91), (230, 230, 240), -1)

    np.random.seed(42)
    manip_img = base.copy()
    p = manip_img[40:80, 44:84].astype(np.int16) + np.random.randint(-40, 40, (40, 40, 3), dtype=np.int16)
    manip_img[40:80, 44:84] = np.clip(p, 0, 255).astype(np.uint8)
    cv2.ellipse(manip_img, (64, 64), (25, 25), 0, 0, 360, (255, 100, 100), 1)

    _, enc = cv2.imencode('.jpg', base, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
    comp_img = cv2.imdecode(enc, 1)

    pairs = [
        ('A. Identical -> identical', 'Expected zero (0.00)', base.copy(), base.copy(), landmarks, landmarks),
        ('B. Same frame duplicated', 'Expected zero (0.00)', base.copy(), base.copy(), landmarks, landmarks),
        ('C. Small translation', 'Low score after alignment (<0.15)', base.copy(), cv2.warpAffine(base, np.float32([[1, 0, 3], [0, 1, 2]]), (128, 128)), landmarks, landmarks + [3, 2]),
        ('D. Brightness change', 'Robust to illumination shift (<0.20)', base.copy(), cv2.convertScaleAbs(base, alpha=1.05, beta=15), landmarks, landmarks),
        ('E. Natural face movement', 'Tolerate natural shear/motion (<0.30)', base.copy(), cv2.warpAffine(base, shear_m, (128, 128)), landmarks, cv2.transform(landmarks.reshape(1, -1, 2), shear_m).reshape(-1, 2)),
        ('F. Blinking', 'Eyelid close is not fake (<0.35)', base.copy(), blink_img, landmarks, landmarks),
        ('G. Speaking', 'Mouth open is not fake (<0.35)', base.copy(), speak_img, landmarks, landmarks),
        ('H. Head rotation', 'Compensated by affine alignment (<0.25)', base.copy(), cv2.warpAffine(base, rot_m, (128, 128)), landmarks, cv2.transform(landmarks.reshape(1, -1, 2), rot_m).reshape(-1, 2)),
        ('I. Localized manipulation', 'Elevated manipulation cue (>0.45)', base.copy(), manip_img, landmarks, landmarks),
        ('J. Compressed frames', 'Moderate compression tolerance (<0.30)', base.copy(), comp_img, landmarks, landmarks),
    ]

    print(f"{'Test':<30} | {'Expected Behavior':<40} | {'Temporal Score':<14} | {'Interpretation'}")
    print("-" * 115)
    for name, exp, f1, f2, lm1, lm2 in pairs:
        score, eff, eg = evaluate_pair(f1, f2, lm1, lm2)
        interp = "Authentic Motion" if score < 0.40 else "Manipulated / Glitch"
        print(f"{name:<30} | {exp:<40} | {score:<14.4f} | {interp}")

if __name__ == '__main__':
    main()
