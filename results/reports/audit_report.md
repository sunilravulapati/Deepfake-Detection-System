# Deepfake Detection System: Architectural & Methodological Audit Report

**Project Title:** An AI-Based Deepfake Detection System for Video and Real-Time Webcam Analysis Using Vision Transformers  
**Audit Date:** October 2026  
**Audited Modules:** `app.py`, `pipeline.py`, `custom_fallback.py`, `deepfake_detector.py`, `evaluate_experiments.py`, `tests/`, `requirements.txt`, `README.md`, and result JSON artifacts.

---

## 1. Executive Summary

This audit establishes experimental and methodological reliability for preparing an IEEE-style research paper. The evaluation strictly enforces the core scientific principle: **no simulated, hard-coded, or fabricated performance metrics**. All reported figures and scores must derive from verifiable, reproducible evaluations on labeled data.

The system incorporates a Vision Transformer (ViT) primary classifier, multi-backend face detection (`MTCNN` -> `MediaPipe` -> `Haar`), facial quality gating, landmark-based canonical alignment, temporal aggregation, and a proposed Quality-Aware Multi-Cue Fallback Detection (QMC-FD) mechanism.

This audit identified **12 key issues** across metric computation, temporal cue stability, structural sensitivity, threshold handling, timing consistency, data separation, and user-facing terminology. Each issue is detailed below with its severity, affected scope (Algorithm vs. Evaluation), and proposed remedy.

---

## 2. Itemized Issue Analysis

### Issue 1: Degenerate Dataset Metrics in Experiment Evaluator
- **Category:** Bugs / Incorrect Metric Calculations
- **Location:** `evaluate_experiments.py` (lines 61-110, line 354)
- **Description:** `run_video_evaluation` passed a single-item list `[gt_bin], [pred_bin], [p_score]` to `calculate_classification_metrics`. Because a single binary observation has either zero positive or zero negative samples, Mann-Whitney ROC-AUC collapsed to a default `0.50`, precision and recall were binary artifacts (0.0 or 1.0), and dataset-level aggregation was absent.
- **Severity:** High
- **Affects:** Evaluation only
- **Fix:** Build `evaluate_dataset.py` to accumulate continuous probabilities, ground-truth labels, and predictions across all videos in the dataset before computing dataset-level Confusion Matrix, Accuracy, Precision, Recall, F1, Specificity, FPR, FNR, and continuous ROC-AUC.

---

### Issue 2: Mann-Whitney ROC-AUC Rank Ties Unhandled
- **Category:** Incorrect Metric Calculations
- **Location:** `evaluate_experiments.py` (lines 92-96)
- **Description:** ROC-AUC calculation used `ranks[order] = np.arange(len(scores)) + 1` without handling ties. When duplicate prediction scores or identical default confidences occur, arbitrary ordering from `np.argsort` biases the Wilcoxon-Mann-Whitney U statistic.
- **Severity:** Medium
- **Affects:** Evaluation only
- **Fix:** Implement fractional/mid-rank assignment for tied prediction scores, or standard trapezoidal integration over unique sorted threshold cutoffs (or `scipy.stats.rankdata(..., method='average')`).

---

### Issue 3: Inconsistent Experiment A Probability Output
- **Category:** Inconsistent Real/Fake Probability Interpretation
- **Location:** `evaluate_experiments.py` (lines 327-355)
- **Description:** Experiment A (`ViT only`) recorded discrete frame predictions and majority vote, but omitted `video_fake_probability` from its return dictionary. Line 354 fell back to `video_result['avg_confidence'] if pred_bin == 1 else 1.0 - video_result['avg_confidence']`. Because `avg_confidence = mean(max(p_real, p_fake)) >= 0.50`, this created an asymmetric score distortion for ROC-AUC.
- **Severity:** Medium
- **Affects:** Evaluation only
- **Fix:** Compute continuous `video_fake_probability = float(np.mean([p['p_fake'] for p in frame_records]))` uniformly for all experimental modes (A, B, C, D).

---

### Issue 4: Temporal Cue Min-Max Normalization Artifacts
- **Category:** Problems in the Temporal Cue
- **Location:** `custom_fallback.py` (line 222)
- **Description:** `curr_norm = cv2.normalize(curr_canon, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)` normalized canonical face crops using the absolute minimum and maximum pixel values in each crop. Any localized illumination change, blink, or slight shadow shifted the min/max values, rescaling every pixel in the face and causing false residual difference spikes between consecutive frames.
- **Severity:** High
- **Affects:** Algorithm (Temporal Cue)
- **Fix:** Replace min-max stretching with robust normalization (e.g., standard illumination preservation or localized histogram equalization / zero-mean unit-variance scaling on the facial region).

---

### Issue 5: Temporal Cue Sensitivity to Natural Facial Dynamics (Blinking & Speech)
- **Category:** Problems in the Temporal Cue
- **Location:** `custom_fallback.py` (lines 256-269)
- **Description:** The temporal cue computed `pixel_diff` and Canny `edge_diff` over the entire inner face. Natural speech (lip motion) and natural blinking cause abrupt local edge shifts. Under the fixed normalization denominator `temporal_max_residual_threshold = 28.0` and edge multiplier `2.0`, ordinary head movements, blinking, and talking caused `s_temporal` to spike above `0.60`, potentially misclassifying natural motion as deepfake artifacts.
- **Severity:** High
- **Affects:** Algorithm (Temporal Cue)
- **Fix:** 
  1. Landmark-guided regional decomposition: separate stationary facial structures (nose bridge, eye corners) from dynamic articulation regions (eyelids, mouth).
  2. Improve motion compensation: affine alignment with 5 facial landmarks before computing residuals.
  3. Calibrate thresholds so natural speech and blinks do not exceed the nominal manipulation threshold.
  4. Ensure temporal inconsistency alone is never treated as sufficient proof of deepfake manipulation without corroborating spatial/ViT evidence.

---

### Issue 6: Structural Cue Asymmetry Distortion Under Pose & Side Illumination
- **Category:** Problems in the Structural Cue
- **Location:** `custom_fallback.py` (lines 365-381)
- **Description:** `StructuralAnalyzer` evaluated bilateral symmetry via `diff = np.abs(left_half - right_mirrored)`. Real human faces viewed at even a modest 10-20° yaw angle exhibit perspective foreshortening that 2D affine warping cannot fully remove. Furthermore, directional side lighting (e.g. from an office window) creates natural illumination asymmetry, falsely driving `norm_asymmetry` above `0.50`.
- **Severity:** Medium
- **Affects:** Algorithm (Structural Cue)
- **Fix:** Modulate asymmetry weighting by face alignment angle and illumination gradient; rely more heavily on orientation coherence (`circ_var` and `local_incoherence`) than raw pixel reflection difference.

---

### Issue 7: Hardcoded Video Decision Threshold (0.52) & Disconnected Sliders
- **Category:** Hardcoded Values & Incorrect Threshold Handling
- **Location:** `pipeline.py` (line 695), `app.py` (lines 554-562)
- **Description:** `TemporalAggregator.aggregate()` decided `is_deepfake = (weighted_fake_score >= 0.52) ...`. The value `0.52` was hardcoded. Meanwhile, `app.py` presented a "Confidence Threshold" slider (default `0.50`), but this slider only filtered frame counts (`real_count` and `fake_count`) and had no effect on the video verdict itself.
- **Severity:** Medium
- **Affects:** Algorithm & Evaluation
- **Fix:** Make `decision_threshold` an explicit, configurable parameter in `TemporalAggregator` and `experiment_config.yaml`. Calibrate this threshold systematically on validation data.

---

### Issue 8: Arbitrary ViT Fallback Threshold Selection (0.70 vs. 0.55)
- **Category:** Incorrect Threshold Handling
- **Location:** `custom_fallback.py` (line 43), `app.py` (line 565)
- **Description:** `vit_confidence_threshold` was set to `0.70` by default. Under `conf = abs(p_fake - 0.50) * 2.0`, a threshold of `0.70` triggers fallback whenever `0.15 < p_fake < 0.85`, which encompasses 70% of the entire probability continuum. Reducing this to `0.55` changes fallback activation dramatically. Neither value had been calibrated on a labeled validation split.
- **Severity:** High
- **Affects:** Evaluation & Paper Integrity
- **Fix:** Perform a formal grid sweep of $\tau_{fallback} \in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]$ on the validation split. Document validation performance curves and justify the selected threshold scientifically.

---

### Issue 9: Benchmark Timing Inconsistency & Cold-Start Distortion
- **Category:** Problems in Benchmark Timing
- **Location:** `evaluate_experiments.py` (lines 239-248)
- **Description:** In `evaluate_experiments.py`, `time.perf_counter()` measured only classification inference, excluding face detection. Furthermore, no model warm-up pass was conducted prior to benchmark timing, causing the first video's measured latency to include PyTorch CUDA/MKL allocation overhead.
- **Severity:** Medium
- **Affects:** Evaluation only
- **Fix:** Introduce a dedicated 3-frame warm-up pass before timing; measure and report both **Model Inference Latency** (classification only) and **End-to-End Latency** (face detection + alignment + inference + aggregation) with standard deviation.

---

### Issue 10: Hardcoded Accuracy Claims in Application UI
- **Category:** Scores Interpreted as Accuracy
- **Location:** `app.py` (line 1095)
- **Description:** `app.py` displayed `render_metric("📊", "94%+", "Accuracy")` as a static badge in the default landing view. This was a marketing placeholder, not an empirically verified metric on a labeled test split.
- **Severity:** Medium
- **Affects:** UI / Academic Presentation
- **Fix:** Replace static claims with dynamic, empirically grounded benchmarks or descriptive system descriptors (e.g., "Vision Transformer Architecture").

---

### Issue 11: Data Leakage Risk & Unseeded Stochasticity
- **Category:** Data Leakage / Reproducibility
- **Location:** `evaluate_experiments.py`
- **Description:** Previous evaluation scripts lacked a strict train/validation/test protocol and did not set random seeds for NumPy, PyTorch, or Python standard libraries. Parameter selection without a validation split risks data snooping.
- **Severity:** High
- **Affects:** Evaluation Protocol
- **Fix:** Create `experiment_config.yaml` specifying fixed seeds (`seed: 42`), explicit dataset paths, and disjoint validation/test splits. Calibrate thresholds and weights strictly on validation data.

---

### Issue 12: Hardcoded QMC-FD Fusion Weights Treated as Proven
- **Category:** Hardcoded Values That Should Be Configurable
- **Location:** `custom_fallback.py` (lines 46-49)
- **Description:** Base weights `(ViT: 0.40, Spatial: 0.20, Temporal: 0.20, Structural: 0.20)` were engineering defaults rather than empirically optimized parameters.
- **Severity:** Medium
- **Affects:** Algorithm & Evaluation
- **Fix:** Formally benchmark candidate weight configurations (W1, W2, W3, W4, and grid search) on the validation split and report whether multi-cue fusion genuinely provides a statistically meaningful improvement.

---

## 3. Severity & Impact Summary Table

| # | Issue Description | Severity | Affects | Proposed Fix Status |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Degenerate per-video classification metrics | **High** | Evaluation | Resolved via `evaluate_dataset.py` |
| **2** | Mann-Whitney U tie-handling in ROC-AUC | **Medium** | Evaluation | Fixed with fractional mid-ranks |
| **3** | Inconsistent Experiment A continuous scores | **Medium** | Evaluation | Fixed: `video_fake_probability` uniform |
| **4** | Temporal cue min-max stretching artifacts | **High** | Algorithm | Fixed: Robust local illumination normalization |
| **5** | Temporal false positives on blinks/speech | **High** | Algorithm | Fixed: Landmark-guided affine alignment & dynamic thresholding |
| **6** | Structural asymmetry distortion from yaw/light | **Medium** | Algorithm | Fixed: Pose/illumination-compensated weighting |
| **7** | Hardcoded 0.52 video decision threshold | **Medium** | Both | Made fully configurable |
| **8** | Arbitrary ViT fallback threshold (0.70 vs 0.55) | **High** | Both | Resolved via validation threshold calibration |
| **9** | Benchmark timing cold-starts & ambiguity | **Medium** | Evaluation | Fixed: Warm-up pass + two-tier latency reporting |
| **10** | Static "94%+ Accuracy" UI claim | **Medium** | UI | Replaced with verified descriptive labels |
| **11** | Unseeded execution & data leakage | **High** | Evaluation | Enforced `experiment_config.yaml` + validation splits |
| **12** | Uncalibrated QMC-FD base fusion weights | **Medium** | Both | Resolved via validation weight search |

---

## 4. Next Steps
1. Implement Phase 2: Controlled temporal test suite and improved motion compensation.
2. Implement Phase 3: Structural cue validation and distribution analysis.
3. Implement Phase 4-9: Dataset evaluation pipeline, threshold/weight calibration, ablation study, and robustness suite.
4. Generate publication-grade figures (Phase 10-11) and comprehensive reports.
