# Score Audit Report

**Paper:** An AI-Based Deepfake Detection System for Video and Real-Time Webcam Analysis Using Vision Transformers
**Date:** October 2026
**Audit Scope:** Tasks 1–10 per user specification
**Test Set:** 4 videos (2 real: real_02, real_03; 2 fake: fake_02, fake_03)

> [!IMPORTANT]
> This audit is READ-ONLY. No detection algorithm, model, or threshold was modified.
> All findings are based on actual inference scores from the first research evaluation run.

---

## 1. AUC Verification

### 1.1 Label Convention

The AUC calculation uses the correct convention:
- `y_true = 0` for real (authentic) videos
- `y_true = 1` for fake (deepfake) videos
- `score = final_fake_score` (higher score = more evidence of deepfake)

The implementation uses the Wilcoxon-Mann-Whitney U statistic with fractional
mid-ranks for ties. This is mathematically equivalent to the trapezoidal ROC-AUC.

### 1.2 Unit Tests (Synthetic Examples)

| Test Case | Labels | Scores | Expected AUC | Computed AUC | Result |
|-----------|--------|--------|-------------|-------------|--------|
| Perfect classifier | [0,0,1,1] | [0.2,0.3,0.7,0.8] | 1.0000 | 1.0000 | PASS |
| Inverted classifier | [0,0,1,1] | [0.7,0.8,0.2,0.3] | 0.0000 | 0.0000 | PASS |
| Equal scores (ties) | [0,1,0,1] | [0.4,0.4,0.6,0.6] | 0.5000 | 0.5000 | PASS |
| Single pair correct | [0,1] | [0.4,0.6] | 1.0000 | 1.0000 | PASS |
| Single pair inverted | [0,1] | [0.6,0.4] | 0.0000 | 0.0000 | PASS |

**Conclusion:** AUC implementation is correct. It uses fake_probability (not inverted),
y_true=1 for fake (not real), and correctly handles ties via mid-ranks.

### 1.3 Verified AUC Values

| Experiment | GT Labels | Fake Scores | AUC | Interpretation |
|-----------|-----------|-------------|-----|----------------|
| Exp A | [0, 0, 1, 1] | [0.4711, 0.6495, 0.3888, 0.4212] | 0.0000 | ⚠ Inverted ranking — fakes score lower than reals |
| Exp C | [0, 0, 1, 1] | [0.3353, 0.3736, 0.3991, 0.4347] | 1.0000 | Correct ranking |
| Exp D | [0, 0, 1, 1] | [0.4487, 0.6318, 0.3275, 0.3753] | 0.0000 | ⚠ Inverted ranking — fakes score lower than reals |

---

## 2. Score Direction Verification

| Experiment | Mean Real Score | Mean Fake Score | Correct Direction? |
|-----------|----------------|----------------|-------------------|
| Exp A | 0.5603 | 0.4050 | ⚠ No (inverted) |
| Exp C | 0.3544 | 0.4169 | ✓ Yes |
| Exp D | 0.5403 | 0.3514 | ⚠ No (inverted) |

**Experiment A (ViT Only):** Inverted. ViT assigns higher fake scores to real videos.
**Experiment C (QMC-FD Only):** ✓ Correct. QMC-FD correctly ranks fakes above reals.
**Experiment D (Full System):** Inverted. ViT domain gap dominates the weighted fusion.

---

## 3. Threshold Analysis

### 3.1 Per-Experiment Score Ranges

| Experiment | Real Scores | Fake Scores | Score Range | Separable at 0.50? |
|-----------|-------------|-------------|-------------|-------------------|
| Exp A (ViT) | [0.4711, 0.6495] | [0.3888, 0.4212] | [0.3888, 0.6495] | No (reals > fakes) |
| Exp C (QMC-FD) | [0.3353, 0.3736] | [0.3991, 0.4347] | [0.3353, 0.4347] | No (all < 0.50) |
| Exp D (Full) | [0.4487, 0.6318] | [0.3275, 0.3753] | [0.3275, 0.6318] | No (fakes < reals) |

### 3.2 Key Observations

- **Exp A:** A threshold of ~0.45 would correctly classify all 4 videos (fakes < 0.45, reals > 0.45).
  However, this threshold MUST NOT be selected from the test set.
- **Exp C:** A threshold of ~0.37 would correctly classify all 4 videos.
  This is consistent with QMC-FD being a decision RANKING score, not a 0.50-anchored probability.
- **Exp D:** No threshold achieves >50% accuracy due to inverted ranking.

> [!WARNING]
> The threshold curves are generated for diagnostic purposes only.
> The final threshold (0.50) was selected on the VALIDATION set, not the test set.
> Using test-set threshold curves to choose a threshold would constitute data leakage.

---

## 4. Calibration Analysis

| Experiment | Score Nature | Is Calibrated Probability? | Recommended Term |
|-----------|-------------|---------------------------|-----------------|
| Exp A (ViT) | Softmax output | Partially — but domain-shifted for DeeperForensics | 'ViT fake probability' + calibration caveat |
| Exp C (QMC-FD) | Artifact cue composite score | No — scores are in [0.33, 0.44], not spanning [0,1] | 'QMC-FD artifact fake score' |
| Exp D (Full) | Weighted blend | No — ranking is inverted | 'composite fake score' |

**Conclusion:** The term 'fake_probability' is only appropriate for the ViT output (Exp A).
For Exp C and D, the correct term is 'fake score'. The paper should be updated accordingly.

---

## 5. Full-System Score Ranking (Experiment D AUC = 0.0)

### 5.1 Ranked Scores

| Rank | Video | Ground Truth | Fake Score | Predicted Label |
|------|-------|-------------|-----------|----------------|
| 1 | fake_02 | fake | 0.3275 | Realism |
| 2 | fake_03 | fake | 0.3753 | Deepfake |
| 3 | real_02 | real | 0.4487 | Realism |
| 4 | real_03 | real | 0.6318 | Deepfake |

Both fake videos rank lowest, both real videos rank highest → AUC = 0.0 (inverted).

### 5.2 Root Cause Trace

1. **ViT domain gap:** ViT assigns P(fake) = 0.39 and 0.42 to DeeperForensics videos.
   The model was not trained on this specific generation method.
2. **QMC-FD correct ranking:** QMC-FD alone separates correctly (AUC=1.0, Exp C).
3. **Fusion inversion:** With ViT weight=0.40, the low ViT fake score for fakes
   dominates the fusion and pulls the final score below the real video scores.
4. **Anomaly override:** fake_03 has fake_score=0.3753 < 0.50 but is labeled 'Deepfake'
   by the anomaly_detected override in TemporalAggregator. The saved score does not
   reflect the full decision logic for this video.
5. **Code is correct** — the AUC=0.0 is arithmetically accurate given actual scores.

> [!NOTE]
> This is NOT a code bug in the AUC calculation. It is a domain generalization
> limitation of the ViT model combined with a score-direction inversion in fusion.

---

## 6. Bugs / Inconsistencies Found

### 6.1 Reporting Inconsistency (Not an Algorithmic Bug)

**Location:** `pipeline.py`, `TemporalAggregator.aggregate()`, decision logic:
```python
is_deepfake = (weighted_fake_score >= decision_threshold) or (anomaly_detected and fake_count > 2)
```

**Problem:** When `anomaly_detected` overrides the threshold decision, the stored
`video_fake_probability` = `weighted_fake_score` (e.g., 0.3753) may be below 0.50,
yet `predicted_label` = 'Deepfake'. This creates an inconsistency between the stored
score and the label used for evaluation.

**Impact:** For video `fake_03` in Exp D, the AUC score (0.3753) does not predict
the stored label ('Deepfake'). The AUC calculation is mathematically correct using
the stored scores, but those scores understate the system's actual fake detection signal.

**Status:** Document as known limitation. Do not fix until full review.

### 6.2 Terminology Issue

| Location | Current Term | Correct Term |
|----------|-------------|-------------|
| Exp C report | 'fake_probability' | 'QMC-FD artifact fake score' |
| Exp D report | 'fake_probability' | 'composite fake score' |
| Exp A report | 'fake_probability' | 'ViT fake probability' (retain, add calibration caveat) |

### 6.3 Domain Gap (Known Limitation, Not a Bug)

The ViT model (`prithivMLmods/Deep-Fake-Detector-v2-Model`) was not fine-tuned on
DeeperForensics autoencoder face swaps. It assigns low fake scores to these videos,
which is the primary cause of both the 0% accuracy in Exp A and the inverted
ranking in Exp D.

---

## 7. Terminology Recommendation

**Replace 'fake_probability' with 'fake_score'** for Experiments C and D throughout
the paper, code comments, and CSV column headers. Retain 'ViT fake probability' for
Experiment A with a note that calibration is domain-dependent.

---

## 8. Recommended Next Experiments

### Priority 1: Calibration Study
- Run QMC-FD on a diverse face image dataset (real, various deepfake methods)
- Plot reliability/calibration curves for both ViT and QMC-FD scores
- This will quantify the miscalibration and guide terminology

### Priority 2: Per-Source Analysis
- Collect DeeperForensics, FaceForensics++ (FaceSwap, NeuralTextures), and Celeb-DF data
- Run Exp A, C, D separately per source to identify which sources ViT handles well

### Priority 3: Dataset Expansion (see Section 9)

---

## 9. Required Additional Dataset Size

| Target | Additional Real | Additional Fake | Total Videos | Expected CI Width (F1) |
|--------|----------------|----------------|-------------|----------------------|
| Current state | 0 | 0 | 4 | ~0.50 (meaningless) |
| Minimum credible | +28 | +28 | 60 | ~0.25 |
| Recommended IEEE | +98 | +98 | 200 | ~0.14 |
| Strong comparison | +498 | +498 | 1000 | ~0.06 |

> [!IMPORTANT]
> The minimum for ANY publishable result is **30 real + 30 fake videos**.
> The current 4-video set is suitable only for code verification and debugging,
> not for paper-quality claims about system performance.

Additional fake videos should span **at least 2 generation methods**.
DeeperForensics alone is insufficient to evaluate generalization.

---

## 10. Files Modified

**This audit script modifies NO existing files.** New files generated:

| File | Description |
|------|-------------|
| `results/reports/score_audit_report.md` | This report |
| `results/figures/score_strip_exp_A.png` | Per-video strip plot, Exp A |
| `results/figures/score_strip_exp_C.png` | Per-video strip plot, Exp C |
| `results/figures/score_strip_exp_D.png` | Per-video strip plot, Exp D |
| `results/figures/score_distribution_exp_A.png` | Score distribution, Exp A |
| `results/figures/score_distribution_exp_C.png` | Score distribution, Exp C |
| `results/figures/score_distribution_exp_D.png` | Score distribution, Exp D |
| `results/figures/threshold_curve_exp_A.png` | Threshold curves, Exp A |
| `results/figures/threshold_curve_exp_C.png` | Threshold curves, Exp C |
| `results/figures/threshold_curve_exp_D.png` | Threshold curves, Exp D |
| `results/csv/threshold_analysis_audit.csv` | Threshold sweep data |
| `tests/test_auc_unit.py` | Standalone AUC unit tests |

---

*Generated by `score_audit.py` — Read-only audit, no algorithm changes.*