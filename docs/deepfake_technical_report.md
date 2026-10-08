# Deepfake Detection System — Full Technical Report
## IEEE Research Paper Technical Audit
**Date:** 2026-10-01 | **Status:** QMC-FD Disabled · ViT-Only Active Pipeline

---

> [!IMPORTANT]
> This report is based exclusively on source code inspection and live execution of the project at `C:\work\DeepfakeDetection`. No values are fabricated. Where verification was not possible from code or execution, the entry is marked **"Not verified from source code."**

---

## PART 1 — QMC-FD DISABLE STATUS

### Changes Made

| File | Change |
|---|---|
| [`pipeline.py`](file:///C:/work/DeepfakeDetection/pipeline.py) | Added `ENABLE_QMC: bool = False` flag (line 22). Rewrote `predict_deepfake_with_fallback()` to branch on flag: when `False`, runs pure ViT path and returns immediately; QMC-FD code preserved in `else`-branch |
| [`app.py`](file:///C:/work/DeepfakeDetection/app.py) | Removed QMC-FD UI slider ("ViT Uncertainty Threshold"), removed "About Detection Method (QMC-FD)" expander content (replaced with ViT description), removed "Detection Method Breakdown (QMC-FD)" section, removed "Multi-Cue Score Breakdown" expander, removed webcam QMC telemetry section, cleaned bounding-box overlay labels |

### What Was NOT Deleted

- `custom_fallback.py` — **preserved intact**
- `CustomFallbackDetector`, `SpatialArtifactAnalyzer`, `TemporalConsistencyAnalyzer`, `StructuralAnalyzer`, `QualityAwareFusion`, `DecisionMode`, `FallbackConfig` — **all preserved, imported but not called during active inference**
- `evaluate_dataset.py` QMC experiment configs — **preserved**
- `tests/test_custom_fallback.py` — **preserved**

### Verification (live execution)

```
ENABLE_QMC = False
decision_mode: PRIMARY_VIT
s_spatial: 0.0
s_temporal: 0.0
s_structural: 0.0
QMC disabled - all cue scores are 0.0: True
```

### Active Inference Path (ENABLE_QMC = False)

```
Video/Webcam
    ↓
Frame Sampling (sample_video_frame_indices)
    ↓
Face Detection (RobustFaceDetector: MTCNN → MediaPipe → Haar)
    ↓
Face Quality Filtering (compute_face_quality)
    ↓
Facial Landmark Alignment + Contextual Crop (align_and_crop_face)
    ↓
Pre-trained Vision Transformer (prithivMLmods/Deep-Fake-Detector-v2-Model)
    ↓
Softmax → p_real, p_fake
    ↓
Temporal Aggregation (TemporalAggregator.aggregate)
    ↓
Final Video/Webcam Verdict
    ↓
Streamlit UI
```

---

## PART 2 — COMPLETE TECHNICAL AUDIT

### Source Files Inspected

| File | Lines | Role |
|---|---|---|
| `app.py` | 1112 | Streamlit application (main entry point) |
| `pipeline.py` | ~830 | Core inference pipeline |
| `custom_fallback.py` | 733 | QMC-FD module (frozen/experimental) |
| `deepfake_detector.py` | 458 | CLI-mode detector (non-Streamlit) |
| `evaluate_dataset.py` | 630 | Dataset evaluation CLI |
| `requirements.txt` | 11 | Dependency specification |
| `dataset/metadata.csv` | 8 | Dataset manifest |
| `tests/test_pipeline.py` | 102 | Pipeline unit tests |
| `results/json/metrics_summary.json` | 90 | Experiment results |

---

## PART 3 — VISION TRANSFORMER DETAILS

### 3.1 Exact Model Identity

| Property | Value | Source |
|---|---|---|
| Hugging Face repository | `prithivMLmods/Deep-Fake-Detector-v2-Model` | `app.py` line 431; `deepfake_detector.py` line 24 |
| Model class | `AutoModelForImageClassification` (resolves to `ViTForImageClassification`) | live execution |
| `model_type` (config) | `vit` | live execution |
| Underlying architecture | ViT-Base/16 | live execution |

### 3.2 Architecture Parameters (verified from live execution)

| Parameter | Value |
|---|---|
| Patch size | 16 × 16 pixels |
| Input image size | 224 × 224 pixels |
| Hidden size | 768 |
| Number of transformer layers | 12 |
| Number of attention heads | 12 |
| Intermediate (MLP) size | 3072 |
| Number of output classes | 2 |
| Class 0 label | `Realism` (authentic) |
| Class 1 label | `Deepfake` (manipulated) |

### 3.3 Pretraining / Fine-tuning Status

> [!WARNING]
> The model is loaded directly from the public Hugging Face Hub (`prithivMLmods/Deep-Fake-Detector-v2-Model`). It is a **pre-trained model** — whether it is additionally fine-tuned by `prithivMLmods` is stated on their Hugging Face model card, but is **not explicitly documented in our repository source code**.

- **Our system uses the model for inference only.** We do **not** train or fine-tune the model weights.
- The model weights are loaded with `AutoModelForImageClassification.from_pretrained(model_name)`.
- **Our contribution is the inference system, the face preprocessing pipeline, and the temporal aggregation mechanism — not the model weights.**
- Correct attribution: "We use the publicly available pre-trained ViT model `prithivMLmods/Deep-Fake-Detector-v2-Model` from the Hugging Face Hub for deepfake classification."

### 3.4 Image Processor Details (verified from live execution)

| Property | Value |
|---|---|
| Processor class | `ViTImageProcessor` |
| Input resolution | 224 × 224 pixels |
| Normalization mean | (0.5, 0.5, 0.5) per channel |
| Normalization std | (0.5, 0.5, 0.5) per channel |
| Pixel range after normalization | [−1.0, +1.0] |
| Channel ordering | RGB (converted from BGR via OpenCV before processing) |
| Resizing/Cropping | Performed by `ViTImageProcessor` internally |

### 3.5 Model Loading Procedure

**File:** [`app.py`](file:///C:/work/DeepfakeDetection/app.py), function `load_deepfake_model()` (lines 427–438)

```python
@st.cache_resource(show_spinner=False)
def load_deepfake_model():
    model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model
```

- Cached with `@st.cache_resource` — loaded once per Streamlit session.
- `model.eval()` called — batch normalization and dropout in inference mode.
- No explicit `torch.dtype` set — defaults to `torch.float32`.
- Device: CPU (`torch.cuda.is_available()` returns `False` on this machine).

### 3.6 Exact Inference Function

**File:** [`pipeline.py`](file:///C:/work/DeepfakeDetection/pipeline.py), function `predict_deepfake()` (lines 438–500)

```python
def predict_deepfake(face_image, processor, model, device="cpu"):
    # 1. Convert BGR numpy → RGB → PIL Image
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)
    # 2. Apply ViTImageProcessor (resize to 224x224, normalize to [-1,1])
    inputs = processor(images=pil_image, return_tensors="pt")
    # 3. Inference with gradients disabled
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(logits.argmax(-1).item())
    # 4. Extract label and confidence
    label = model.config.id2label[predicted_idx]   # "Realism" or "Deepfake"
    confidence = float(probs[predicted_idx].item())
    return label, confidence, p_real, p_fake
```

### 3.7 Softmax and Probability Calculation

**Formula:**
$$p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}, \quad i \in \{0=\text{Realism},\; 1=\text{Deepfake}\}$$

- Logits `z` are the raw output of the ViT classification head.
- `p_real` = `probs[idx]` where `id2label[idx]` contains the substring `"real"` (case-insensitive match).
- `p_fake` = `probs[idx]` for the remaining class (`"Deepfake"`).
- `confidence` = `probs[predicted_idx]` = max probability.

### 3.8 Gradients and Eval Mode

- `torch.no_grad()` is applied around every forward pass (`pipeline.py` line 467).
- `model.eval()` is called at load time (`app.py` line 434).

### 3.9 Final Per-Frame Prediction Rule

```
if p_fake >= 0.50 → label = "Deepfake"
else              → label = "Realism"
```

This threshold is applied at the **ViT output level** in the active path.

---

## PART 4 — COMPLETE VIDEO PIPELINE

### 4.1 Video Upload and Decoding

**File:** `app.py`, lines 592–729

1. User uploads an MP4/AVI/MOV file via `st.file_uploader`.
2. File bytes are written to a `tempfile.NamedTemporaryFile` with `.mp4` suffix.
3. OpenCV `cv2.VideoCapture(tfile.name)` opens the file.
4. `fps = cap.get(cv2.CAP_PROP_FPS)` — sanitized to 24.0 if ≤0 or NaN.
5. `total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)`.

### 4.2 Frame Sampling

**File:** `pipeline.py`, function `sample_video_frame_indices()` (lines ~741+)

| Parameter | Value |
|---|---|
| `target_sample_fps` | 3.0 frames/second |
| `max_samples` | 60 |
| Sampling step | `step = max(1, int(round(fps / target_sample_fps)))` |
| Initial indices | `range(0, total_frames, step)` |
| Downsampling | If `len(indices) > 60`: uniform stride to select exactly 60 |

### 4.3 Frame Reading

- `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)` seeks to each sampled index.
- `ret, frame = cap.read()` reads the BGR frame.

### 4.4 Face Detection

**File:** `pipeline.py`, class `RobustFaceDetector.detect_faces()` (lines 293–422)

- `detector.detect_faces(frame, max_faces=3)` called per sampled frame.
- Returns list sorted by `(is_valid_quality, area, quality)` descending.
- Best face = `faces[0]`.

### 4.5 Face Quality Filtering

**File:** `pipeline.py`, function `compute_face_quality()` (lines 36–118)

Reject condition (in app.py line 653):
```python
if not best_face.get('is_valid_quality', True) and best_face.get('quality', 0) < 15.0:
    skipped_low_quality += 1
    # frame skipped, no ViT inference
```

Note: A face must both fail the quality gate AND score below 15/100 to be skipped.

### 4.6 Face Crop and Alignment

**File:** `pipeline.py`, function `align_and_crop_face()` (lines 123–195)

1. If landmarks available and |angle| in (3°, 40°): rotate frame using `cv2.getRotationMatrix2D` at eyes center.
2. Expand bounding box to square: `side = max(w, h) * (1 + 2 * padding_ratio)`.
3. Padding ratio: `0.03` (3% extra each side).
4. Out-of-frame regions filled with `cv2.BORDER_REFLECT`.

### 4.7 ViT Preprocessing

- Face crop (BGR numpy) → RGB → PIL Image → `ViTImageProcessor`.
- Resize to 224×224; normalize to [−1, 1] with mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5).

### 4.8 ViT Inference

- `predict_deepfake_with_fallback()` called per face.
- With `ENABLE_QMC=False`: routes directly to `predict_deepfake()`.
- Returns `p_real`, `p_fake`, `final_label`, `final_confidence`.

### 4.9 Temporal Aggregation

See Part 5 for full detail. Called once after all frames processed:
```python
agg_result = TemporalAggregator.aggregate(predictions, confidence_threshold=st.session_state.confidence_threshold)
```

### 4.10 Final Verdict

- `final_label` = `"Deepfake"` or `"Realism"` from aggregation.
- `avg_confidence` = weighted fake or real score (see Part 5).
- `confidence_threshold` default = 0.50, adjustable via UI slider.

---

## PART 5 — TEMPORAL AGGREGATION (Active After QMC Disabled)

**File:** `pipeline.py`, class `TemporalAggregator.aggregate()` (lines ~565–726)

> [!NOTE]
> QMC is disabled. The temporal aggregation mechanism described here is the only aggregation performed. It does not involve QMC cues — it uses frame-level ViT soft probabilities and face quality scores only.

### 5.1 Per-Frame Weight Calculation

For each frame prediction `t`:

$$\text{certainty}_t = |\,p_{\text{fake},t} - 0.5\,| \times 2 \quad \in [0, 1]$$

$$w_t = \max\!\left(0.1,\; \frac{q_t}{100} \times \left(0.5 + 0.5 \times \text{certainty}_t\right)\right)$$

where $q_t$ is the face quality score in [0, 100].

### 5.2 Weighted Probability Aggregation

$$P_{\text{fake, video}} = \frac{\sum_t w_t \cdot p_{\text{fake},t}}{\sum_t w_t}$$

$$P_{\text{real, video}} = \frac{\sum_t w_t \cdot p_{\text{real},t}}{\sum_t w_t}$$

### 5.3 Discrete Frame Counts (for anomaly detection)

A frame is counted as `fake` if: `p_fake >= confidence_threshold AND p_fake > p_real`.

A frame is counted as `real` if: `p_real >= confidence_threshold AND p_fake <= p_real`.

### 5.4 Anomaly / Burst Detection

```python
anomaly_detected = (max_consecutive_fake >= 3) and ((max_consecutive_fake / total_preds) >= 0.15)
```

where `max_consecutive_fake` = length of the longest consecutive run of fake-counted frames.

### 5.5 Final Decision Rule

```python
is_deepfake = (P_fake_video >= decision_threshold) OR (anomaly_detected AND fake_count > 2)
```

Default `decision_threshold` = 0.50.

If `is_deepfake`:
- `final_label` = `"Deepfake"`, `avg_confidence` = `P_fake_video`

Else:
- `final_label` = `"Realism"`, `avg_confidence` = `P_real_video`

### 5.6 Parameters Summary

| Parameter | Value |
|---|---|
| Confidence threshold (frame-level) | 0.50 (adjustable via UI) |
| Decision threshold (video-level) | 0.50 (hardcoded default) |
| Weight floor | 0.1 (ensures frames with q≈0 still contribute minimally) |
| Anomaly burst threshold | ≥3 consecutive fake frames AND ≥15% of total |
| Quality effect | Higher quality frames receive proportionally higher weight |
| Multiple faces | Only `faces[0]` (best face by area × quality) is used per frame |
| No face detected | Frame skipped; no prediction added |
| Quality gate failure | If `quality < 15.0`, frame skipped |
| EMA | Not used |
| Majority voting | Not used (soft probability weighting is primary mechanism) |

---

## PART 6 — FACE DETECTION

**File:** `pipeline.py`, class `RobustFaceDetector` (lines 200–422)

### 6.1 Detection Priority and Fallback Chain

| Priority | Backend | Library | Model | Landmarks |
|---|---|---|---|---|
| 1 (preferred) | MTCNN | `facenet-pytorch` 2.6.0 | Internal P-Net/R-Net/O-Net cascade | 5 points (left eye, right eye, nose, left mouth, right mouth) |
| 2 (fallback) | MediaPipe FaceDetector | `mediapipe` 1.0.1 | `blaze_face_short_range.tflite` (downloaded on demand) | Up to 5 keypoints |
| 3 (final fallback) | Haar Cascade | `opencv-contrib-python` 5.0.0 | `haarcascade_frontalface_default.xml` | None |

### 6.2 MTCNN Configuration

```python
MTCNN(
    keep_all=True,
    device='cpu',
    thresholds=[0.6, 0.7, 0.7],   # P-Net, R-Net, O-Net
    min_face_size=32,
    post_process=False
)
```

- Min confidence to accept detection: `min_confidence=0.50` (class default).
- Bounding box format: `(x1, y1, x2, y2)` float → converted to `(x, y, w, h)`.
- Landmarks: `np.ndarray` of shape `(5, 2)`.

### 6.3 MediaPipe Configuration

```python
FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="models/blaze_face_short_range.tflite"),
    min_detection_confidence=0.50
)
```

- Model downloaded from Google storage if not cached locally.
- Bounding box: `(origin_x, origin_y, width, height)` (pixel values).
- Keypoints scaled by `(fw, fh)` to pixel coordinates.

### 6.4 Haar Cascade Configuration

```python
detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(36, 36))
```

- Confidence score hardcoded to `0.75`.
- No landmarks available.

### 6.5 Why Multiple Detectors

No explicit design rationale documented in source code. The code comment reads:
> "Multi-backend face detector supporting: 1. MTCNN — high precision, 5 landmarks; 2. MediaPipe — lightweight and fast; 3. OpenCV Haar Cascade / YuNet — reliable fallback."

The pattern is a robustness mechanism: if a higher-quality detector is unavailable (ImportError or download failure), the pipeline continues with a simpler detector.

---

## PART 7 — FACE ALIGNMENT

**File:** `pipeline.py`, function `align_and_crop_face()` (lines 123–195)

### 7.1 Landmark-Based Rotation

**Source:** Eye landmarks `[0]` (left eye) and `[1]` (right eye) from detector output.

**Rotation angle:**
$$\theta = \arctan2\!\left(\Delta y_{\text{eyes}},\; \Delta x_{\text{eyes}}\right) \text{ (degrees)}$$

**Applied if:** $3° < |\theta| < 40°$ (moderate tilt; extreme tilts skipped).

**Affine rotation matrix:**
$$M_{\text{rot}} = \text{cv2.getRotationMatrix2D}(\text{eyes\_center},\; \theta,\; \text{scale}=1.0)$$

**Applied:**
$$\text{frame\_rotated} = \text{cv2.warpAffine}(\text{frame},\; M_{\text{rot}},\; (f_w, f_h))$$

with `borderMode=cv2.BORDER_REFLECT`.

### 7.2 Square Crop with Padding

$$\text{side} = \max(w, h) \times (1 + 2 \times \text{padding\_ratio})$$

where `padding_ratio = 0.03`.

Crop center = center of original bounding box `(cx, cy)`.

Out-of-frame regions filled with `cv2.BORDER_REFLECT`.

**Output resolution:** Variable (side × side pixels, depends on face size in frame).

### 7.3 Canonical Alignment (QMC internal — not active)

`CANONICAL_FACE_5PTS` used only inside QMC's `TemporalConsistencyAnalyzer.align_to_canonical()` and `StructuralAnalyzer.align_to_canonical()`. These are **not called when ENABLE_QMC=False**.

---

## PART 8 — QUALITY FILTERING

**File:** `pipeline.py`, function `compute_face_quality()` (lines 36–118)

### 8.1 Rejection Criteria (is_valid = False)

| Criterion | Threshold | Variable |
|---|---|---|
| Minimum face size | `height < 40 OR width < 40` pixels | `min_size=40` |
| Blur (Laplacian variance) | `blur_var < 20.0` | `min_blur_var=20.0` |
| Extreme underexposure | `brightness < 12.0` | mean of grayscale |
| Extreme overexposure | `brightness > 248.0` | mean of grayscale |
| Insufficient contrast | `contrast < 6.0` | std dev of grayscale |

**Sharpness measurement:**
$$\text{blur\_var} = \text{Var}\!\left(\nabla^2 I_{\text{gray}}\right) = \text{Var}(\text{cv2.Laplacian}(I_{\text{gray}},\; \text{CV\_64F}))$$

### 8.2 Continuous Quality Score [0, 100]

$$q = 100 \times \left(0.4 \cdot \bar{b} + 0.2 \cdot \bar{e} + 0.2 \cdot \bar{c} + 0.2 \cdot \bar{s}\right)$$

where:
- $\bar{b} = \min(1, \text{blur\_var} / 300)$ (sharpness, weight 40%)
- $\bar{e} = 1 - |\text{brightness} - 128| / 128$ (exposure balance, weight 20%)
- $\bar{c} = \min(1, \text{contrast} / 60)$ (contrast, weight 20%)
- $\bar{s} = \min(1, \sqrt{h \cdot w} / 200)$ (relative size, weight 20%)

> [!IMPORTANT]
> Quality filtering is **input quality control / preprocessing validation**, not deepfake evidence. Low quality causes frames to be skipped, not classified as fake.

### 8.3 Application of Quality Gate in App

```python
# app.py line 653
if not best_face.get('is_valid_quality', True) and best_face.get('quality', 0) < 15.0:
    skipped_low_quality += 1
    # frame skipped — no ViT inference performed
```

The `and quality < 15.0` condition means: skip only if both the boolean gate fails AND score is below 15.

---

## PART 9 — WEBCAM PIPELINE

**File:** `app.py`, lines 894–1064

### 9.1 Camera Capture

```python
cap = cv2.VideoCapture(0)  # Default camera index 0
```

No explicit FPS/resolution properties set in Streamlit mode (differs from `deepfake_detector.py` which sets 640×480@30fps).

### 9.2 Frame Processing Loop

```python
FRAME_SKIP = 6  # Run ViT inference every 6th frame
```

- Every frame: `cap.read()` → face detection → bounding box overlay.
- Every 6th frame AND face detected: face crop → quality check → ViT inference → temporal update.

### 9.3 Inference Trigger

```python
if frame_count % FRAME_SKIP == 0 and len(faces) > 0:
    best_face = faces[0]
    eval_res = predict_deepfake_with_fallback(face_img, processor, model, ...)
```

- `fallback_detector.reset_temporal_state()` called on start (required for temporal state init, even though QMC is disabled).

### 9.4 Display

- Frame displayed via `frame_placeholder.image(frame_rgb, ...)` — Streamlit renders each frame sequentially.
- Status label updated with verdict and confidence.

### 9.5 Webcam Temporal Aggregation

Same `TemporalAggregator.aggregate()` called on session end (when Stop button pressed).

### 9.6 Real-Time Claim

The implementation processes every 6th frame with ViT inference. On CPU (this machine), ViT inference latency measured at ~89 ms per frame. Therefore:
- Face detection runs on every frame (fast).
- ViT inference runs approximately every 6 frames.
- **Streamlit renders frames sequentially** — it is not a true real-time system in the strict sense; it is a near-real-time system.
- No explicit FPS measurement is implemented in the Streamlit webcam code. The `deepfake_detector.py` CLI mode does not measure FPS either.

---

## PART 10 — STREAMLIT APPLICATION

**File:** `app.py` (1112 lines total)

### 10.1 Pages / Tabs

No `st.tabs()` — single-page design with mode-switching via buttons.

| Mode | Key | Description |
|---|---|---|
| Video Detection | `btn_video` | Upload video, analyze frames |
| Live Webcam | `btn_webcam` | Continuous webcam detection |

### 10.2 Mode Toggle Card

- Two `st.button()` elements styled as a pill toggle.
- `st.session_state.mode` ∈ `{None, "video", "webcam"}`.

### 10.3 Upload Interface

```python
st.file_uploader("Upload a video file (MP4, AVI, MOV)", type=["mp4","avi","mov"])
```

### 10.4 Confidence Threshold Slider

- `st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.50, step=0.05)`
- Applies to frame-level counting in `TemporalAggregator`.

### 10.5 Verdict Display

```html
<div class="result-success">✅ AUTHENTIC VIDEO<br>
  <span>Confidence: XX.XX% (summary)</span>
</div>
```
or

```html
<div class="result-danger">⚠️ DEEPFAKE DETECTED<br>
  <span>Confidence: XX.XX% (summary)</span>
</div>
```

### 10.6 Metrics Displayed

- Analyzed Frames, Authentic Frames, Fake Frames, Confidence Score.
- Average Face Quality (scalar).

### 10.7 Charts

- `st.bar_chart`: Authentic vs Fake frame count.
- `st.line_chart`: Fake Probability (%) over time (seconds).

### 10.8 Export

JSON and CSV export via base64 download links.

### 10.9 Session State Keys

| Key | Default | Purpose |
|---|---|---|
| `mode` | `None` | Current detection mode |
| `predictions` | `[]` | Accumulated frame predictions |
| `webcam_active` | `False` | Webcam loop control |
| `confidence_threshold` | `0.50` | Frame confidence cutoff |

### 10.10 Model Caching

`@st.cache_resource` on `load_deepfake_model()` and `load_face_detector()` — loaded once per Streamlit session lifetime.

### 10.11 QMC Elements Removed from Active UI

| Removed element | Location |
|---|---|
| "ViT Uncertainty Threshold (QMC-FD)" slider | Settings card |
| "About Detection Method (QMC-FD)" expander | Settings card |
| "Detection Method Breakdown (QMC-FD)" section | Video results |
| "Multi-Cue Score Breakdown" expander | Video results |
| Spatial / Temporal / Structural cue metrics | Video results |
| "Detection Method Summary (QMC-FD)" | Webcam session summary |
| `[ViT]`/`[Fusion]`/`[Fallback]` mode tags in bounding box | Video + Webcam overlay |
| QMC info box ("QMC-FD was engaged on N frames…") | Video results |

---

## PART 11 — SOFTWARE STACK

| Component | Technology | Verified Version | Purpose |
|---|---|---|---|
| Language | Python | 3.14.3 | Runtime |
| Deep Learning | PyTorch | 2.13.0+cpu | Model inference |
| Vision Transformer | HuggingFace Transformers | 5.16.1 | ViT model + processor |
| Face Detection (primary) | facenet-pytorch (MTCNN) | 2.6.0 | MTCNN face detector |
| Face Detection (secondary) | MediaPipe | 1.0.1 | BlazeFace short-range |
| Face Detection (tertiary) | OpenCV Haar Cascade | via OpenCV | Fallback detector |
| Image Processing | OpenCV-contrib | 5.0.0 | Frame decoding, alignment |
| Image Processing | Pillow | 12.3.0 | PIL conversion for ViT |
| Numerical | NumPy | 2.5.2 | Array operations |
| Web UI | Streamlit | 1.62.0 | Application interface |
| Testing | pytest | 9.1.1 | Unit tests |
| GPU Acceleration | CUDA | Not available | CPU-only on dev machine |

---

## PART 12 — HARDWARE / EXECUTION ENVIRONMENT

| Property | Value | Source |
|---|---|---|
| Operating System | Windows (win32) | pytest output |
| Python version | 3.14.3 (MSC v.1944 64-bit) | live execution |
| PyTorch build | 2.13.0+cpu | live execution |
| CUDA available | False | `torch.cuda.is_available()` |
| CUDA version | N/A (CPU-only build) | — |
| CPU | Not verified from source code | — |
| GPU | None (CPU inference) | live execution |
| RAM | Not verified from source code | — |

> **Note:** This is the development machine. No separate benchmark machine is documented.

---

## PART 13 — DATASET INFORMATION

### 13.1 Dataset in Repository

**File:** `dataset/metadata.csv`

| Category | Files | Labels |
|---|---|---|
| Real | `real_interview_01.mp4`, `real_interview_02.mp4`, `real_interview_03.mp4` | `real` |
| Fake | `fake_deeperforensics_01.mp4`, `fake_deeperforensics_02.mp4`, `fake_deeperforensics_03.mp4` | `fake` |
| **Total** | **6 videos** | — |

| Split | Real | Fake |
|---|---|---|
| val | 1 | 1 |
| test | 2 | 2 |

### 13.2 Dataset Source

- **Real videos**: Labeled `real_interview_*.mp4` — source not explicitly documented in code.
- **Fake videos**: Labeled `fake_deeperforensics_*.mp4` — names suggest DeeperForensics origin, but source, version, and download URL are **not documented in the repository**.

> [!CAUTION]
> The dataset contains only **6 videos**. This is not a sufficiently large formal evaluation dataset. Accuracy metrics derived from this dataset alone cannot be reported as representative results.

### 13.3 Model Training

The ViT model weights are **not trained or fine-tuned** by this project. The project uses the pre-trained model for inference only.

---

## PART 14 — EXPERIMENTS PERFORMED

### 14.A Valid Experiments — Active ViT System

**File:** `results/json/metrics_summary.json`

> [!CAUTION]
> These experiments were run on only **4 videos** (test split from the 6-video dataset). The sample size is too small for statistically meaningful accuracy claims. Results are presented as-is for completeness; they should **not** be reported as final accuracy in the IEEE paper without a properly sized benchmark.

#### Experiment A — ViT Only (no temporal aggregation)

| Metric | Value |
|---|---|
| Videos | 4 (2 real, 2 fake — test split) |
| Accuracy | 0.25 |
| Precision | 0.0 |
| Recall | 0.0 |
| F1 | 0.0 |
| ROC-AUC | 0.0 |
| FPR | 0.5 |
| FNR | 1.0 |
| TP/FP/FN/TN | 0/1/2/1 |
| ViT Inference Latency | 88.97 ms |
| End-to-End Latency | 153.21 ms |
| Effective FPS | 6.53 |

#### Experiment B — ViT + Temporal Aggregation (active system)

| Metric | Value |
|---|---|
| Accuracy | 0.25 |
| Precision | 0.0 |
| Recall | 0.0 |
| F1 | 0.0 |
| ROC-AUC | 0.0 |
| FPR | 0.5 |
| FNR | 1.0 |
| TP/FP/FN/TN | 0/1/2/1 |
| ViT Inference Latency | 89.93 ms |
| End-to-End Latency | 152.25 ms |
| Effective FPS | 6.57 |

> These results are from the current active system configuration and should be interpreted cautiously given the 4-video test set.

### 14.B Deprecated / QMC-Specific Experiments

#### Experiment C — QMC-FD Fallback Only (No ViT) — **FROZEN / DEPRECATED**

| Metric | Value |
|---|---|
| Accuracy | 0.50 |
| Precision | 0.0 |
| Recall | 0.0 |
| F1 | 0.0 |
| ROC-AUC | 1.0 |
| Inference Latency | 3.52 ms |
| Effective FPS | 15.83 |

*Status: QMC-FD is frozen. This experiment is deprecated from the active system description.*

#### Experiment D — Full System (ViT + QMC-FD + Temporal Agg) — **FROZEN / DEPRECATED**

| Metric | Value |
|---|---|
| Accuracy | 0.50 |
| Precision | 0.50 |
| Recall | 0.50 |
| F1 | 0.50 |
| ROC-AUC | 0.0 |
| Inference Latency | 90.21 ms |
| Effective FPS | 6.48 |

*Status: QMC-FD is frozen. This experiment is deprecated from the active system description.*

---

**OVERALL CONCLUSION ON EXPERIMENTS:**
> **"The active system (ViT + Temporal Aggregation) has not yet been evaluated on a sufficiently sized labeled benchmark. Current results are from a 4-video test set and cannot be used to make generalizable accuracy claims."**

---

## PART 15 — LIMITATIONS

The following limitations are supported by the implementation and experiment results.

| # | Limitation | Evidence |
|---|---|---|
| 1 | **Pre-trained model dependency** | Model is loaded from external Hugging Face Hub; generalization depends on pretraining data (not documented in our code) |
| 2 | **CPU-only inference** | CUDA unavailable on dev machine; ~89 ms per ViT inference frame makes dense processing slow |
| 3 | **Small evaluation dataset** | Only 6 videos total, 4 in test split — insufficient for generalizable metrics |
| 4 | **Face detection failure** | If all 3 backends fail, no face is detected and frames are skipped entirely |
| 5 | **Single-face-per-frame assumption** | Only `faces[0]` (largest/best quality) is used; multiple-face scenarios are not jointly analyzed |
| 6 | **Low-quality frame rejection** | Blurry, underexposed, overexposed, or very small faces are skipped — may cause decisions on reduced frame counts |
| 7 | **No GPU acceleration** | Webcam mode is near-real-time, not strictly real-time; inference at ~6–7 effective FPS on CPU |
| 8 | **Webcam frame rate** | `FRAME_SKIP=6` means ViT runs on 1 in every 6 frames, introducing latency between detections |
| 9 | **Domain shift** | ViT pretrained on specific manipulation types may not generalize to novel deepfake methods |
| 10 | **Extreme pose / occlusion** | Face detection may fail or landmark alignment may be incorrect under extreme head poses or partial occlusion |
| 11 | **Compression artifacts** | Video compression (especially at low bitrate) may degrade face quality and trigger quality gate rejection |

---

## PART 16 — RECOMMENDED IEEE PAPER STRUCTURE

```
1. Introduction
   - Rise of deepfakes and societal impact
   - Motivation for practical, accessible detection
   - Contribution: ViT-based system with multi-backend face preprocessing and temporal aggregation

2. Related Work
   - Transformer-based deepfake detection
   - CNN-based methods (XceptionNet, EfficientNet)
   - Self-supervised and contrastive methods
   - Temporal analysis for video deepfakes
   - Face detection and preprocessing in media forensics

3. Proposed System Overview
   - High-level architecture diagram
   - Two operating modes: Video and Webcam
   - Not claiming novelty of the ViT model itself — novelty lies in the inference system design

4. Vision Transformer-Based Classification
   - Pre-trained ViT: prithivMLmods/Deep-Fake-Detector-v2-Model
   - ViT-Base/16 architecture (224×224, patch 16, 12 layers, 768 hidden)
   - Softmax probability output: p_real, p_fake
   - Inference procedure and model loading

5. Face Detection and Preprocessing Pipeline
   - Multi-backend detection (MTCNN → MediaPipe → Haar Cascade)
   - Quality filtering (blur, exposure, size)
   - Alignment and contextual cropping
   - ViT preprocessing (ViTImageProcessor, 224×224, normalization)

6. Temporal Aggregation
   - Quality-weighted probability averaging
   - Burst/anomaly detection
   - Final decision rule
   - Robustness benefits over simple majority voting

7. Implementation
   - Streamlit application
   - Software stack
   - Model caching strategy
   - CPU vs GPU discussion

8. Experimental Methodology
   - Dataset description (size, limitations)
   - Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR)
   - Latency measurement methodology

9. Results and Discussion
   - Available latency results
   - Accuracy results (with clear caveat about dataset size)
   - Temporal aggregation vs frame-only comparison
   - Honest discussion of limitations

10. Limitations and Future Work
    - Dataset scale needed for formal evaluation
    - GPU deployment for real-time performance
    - Multi-face scenarios
    - QMC-FD as future experimental direction (mention briefly)
    - Audio-video multimodal detection
    - Self-supervised representation learning

11. Conclusion

12. References
```

> [!NOTE]
> QMC-FD should appear **only in Section 10 (Future Work)** as an experimental direction, not as part of the main proposed method.

---

## PART 17 — RECOMMENDED FIGURES AND TABLES

### Figures

| # | Title | Contents | Source | Status |
|---|---|---|---|---|
| Fig. 1 | System Architecture Overview | End-to-end block diagram (Video/Webcam → Sampling → Detection → Quality → Alignment → ViT → Aggregation → UI) | This report (Part 1) | **Generate as diagram** |
| Fig. 2 | Video Processing Pipeline | Detailed flow: upload → decode → FPS extraction → frame sampling → face detection → quality filtering → alignment → ViT inference | `app.py` + `pipeline.py` | **Generate as diagram** |
| Fig. 3 | Face Detection and Preprocessing | Visual: MTCNN bounding box → quality check → aligned/cropped face → 224×224 ViT input | `pipeline.py` | **Needs screencaptures** |
| Fig. 4 | Vision Transformer Inference Pipeline | ViT-Base/16 patch embedding diagram → 12 transformer layers → [CLS] head → softmax probabilities | Model config | **Generate as diagram** |
| Fig. 5 | Temporal Aggregation Workflow | Per-frame p_fake scores → quality weights → weighted average → anomaly detection → final verdict | `TemporalAggregator` | **Generate as diagram** |
| Fig. 6 | Streamlit Application Screenshots | UI screenshots showing video upload, results panel, verdict banner, confidence chart | Live app | **Already running — screenshot** |
| Fig. 7 (optional) | Fake Probability Timeline | Line chart: p_fake(%) over time for a sample video | `st.line_chart` output | **Generate from sample video** |

### Tables

| # | Title | Contents | Source | Status |
|---|---|---|---|---|
| Table 1 | Software Stack | Component / Technology / Version / Purpose | Part 11 of this report | **Available now** |
| Table 2 | ViT Model Configuration | Model name, architecture, patch size, image size, normalization, num classes, class labels | Part 3 of this report | **Available now** |
| Table 3 | Dataset Distribution | Video name / Category / Split / File size | `dataset/metadata.csv` | **Available — but note tiny size** |
| Table 4 | Quality Filtering Thresholds | Criterion / Threshold / Variable / Formula | Part 8 of this report | **Available now** |
| Table 5 | Temporal Aggregation Parameters | Decision threshold, weight floor, anomaly parameters | Part 5 of this report | **Available now** |
| Table 6 | Performance Metrics (Active System) | Latency, FPS for Experiments A and B | `metrics_summary.json` | **Available — caveat dataset size** |
| Table 7 | Ablation / Comparison | Frame-only vs. temporal aggregation | Experiments A vs B | **Available — caveat dataset size** |
| Table 8 (future) | Full Benchmark Results | Accuracy, Precision, Recall, F1, AUC on proper dataset | **Not yet available** | **Needs larger dataset** |

---

## PART 18 — FINAL DELIVERABLE SUMMARY

### A. QMC Status

| Item | Status |
|---|---|
| QMC disabled? | ✅ Yes — `ENABLE_QMC = False` in `pipeline.py` |
| Files preserved? | ✅ `custom_fallback.py` fully intact; imports kept in `pipeline.py` |
| Active inference path? | Pure ViT → `predict_deepfake()` → softmax → `TemporalAggregator` |

### B. Final Active Architecture

```
Video/Webcam Input
    ↓
Frame Sampling
    (target_sample_fps=3.0, max_samples=60)
    ↓
RobustFaceDetector.detect_faces()
    [MTCNN → MediaPipe BlazeFace → Haar Cascade]
    ↓
Face Quality Gate: compute_face_quality()
    [min_size=40px, blur_var≥20, 12≤brightness≤248, contrast≥6]
    ↓
align_and_crop_face()
    [eye-landmark rotation if 3°<|θ|<40°, square crop, padding_ratio=0.03]
    ↓
ViTImageProcessor
    [resize 224×224, normalize mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)]
    ↓
ViTForImageClassification (prithivMLmods/Deep-Fake-Detector-v2-Model)
    [ViT-Base/16, 12 layers, 768 hidden, torch.no_grad(), model.eval()]
    ↓
torch.softmax(logits) → (p_real, p_fake)
    ↓
TemporalAggregator.aggregate()
    [quality-weighted avg, burst anomaly detection, threshold=0.50]
    ↓
Final Verdict (Realism / Deepfake) + Confidence Score
    ↓
Streamlit UI
```

### C. Exact Vision Transformer Details

| Property | Value |
|---|---|
| HuggingFace repo | `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| Architecture | ViT-Base/16 |
| Model class | `ViTForImageClassification` |
| Status | Pre-trained model used for inference; not trained/fine-tuned by this project |
| Pretraining dataset | Not documented in our repository |
| Fine-tuning dataset | Not documented in our repository |
| num_labels | 2 |
| id2label | `{0: 'Realism', 1: 'Deepfake'}` |
| Input size | 224 × 224 |
| Normalization | mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5) → pixel range [−1, 1] |
| Device | CPU |
| torch.dtype | float32 (default) |
| Gradients | Disabled (`torch.no_grad()`) |
| Eval mode | Yes (`model.eval()`) |

### D. Exact Preprocessing Details

- BGR frame → `cv2.cvtColor(BGR2RGB)` → PIL Image
- `ViTImageProcessor(images=pil_image, return_tensors="pt")`
- Internally: resize to 224×224, normalize with mean/std=(0.5,0.5,0.5)

### E. Exact Temporal Aggregation Details

$$w_t = \max(0.1,\; (q_t/100) \cdot (0.5 + 0.5 \cdot |\,p_{\text{fake},t} - 0.5\,| \cdot 2))$$

$$P_{\text{fake,video}} = \sum_t w_t p_{\text{fake},t} \;/\; \sum_t w_t$$

Decision: `Deepfake` if $P_{\text{fake,video}} \geq 0.50$ OR (burst ≥ 3 consecutive AND burst/total ≥ 15% AND fake_count > 2)

### F. Exact Video Pipeline

See Part 4 above.

### G. Exact Webcam Pipeline

See Part 9 above.

### H. Exact Software Stack

See Part 11 above.

### I. Experiments / Results Currently Available

- Experiment A (ViT Only) and Experiment B (ViT + Temporal Agg) on 4 test videos: see Part 14.
- **Conclusion: Not yet evaluated on a sufficiently sized labeled benchmark.**

### J. Missing Information Before Final Paper

| Missing Item | Required For |
|---|---|
| Larger labeled dataset (≥50–100 videos per class) | Meaningful accuracy/F1/AUC figures |
| Pretraining/fine-tuning documentation from `prithivMLmods` HuggingFace page | Precise model attribution in Methods section |
| CPU/RAM/hardware specifications of dev machine | Hardware section of paper |
| Performance on diverse manipulation types (FaceSwap, FaceShifter, DeepFaceLab) | Generalization discussion |
| Webcam latency measurements | Real-time performance claims |
| ROC curve data from larger dataset | Figure 7 (ROC curve) |

### K. Recommended Paper Section Structure

See Part 16 above.

### L. Source Files Inspected

- `app.py` (1112 lines)
- `pipeline.py` (~830 lines after modification)
- `custom_fallback.py` (733 lines)
- `deepfake_detector.py` (458 lines)
- `evaluate_dataset.py` (630 lines)
- `requirements.txt`
- `dataset/metadata.csv`
- `tests/test_pipeline.py`
- `results/json/metrics_summary.json`

### M. Files Modified

| File | Changes |
|---|---|
| [`pipeline.py`](file:///C:/work/DeepfakeDetection/pipeline.py) | Added `ENABLE_QMC = False`; rewrote `predict_deepfake_with_fallback()` with ViT-only active path |
| [`app.py`](file:///C:/work/DeepfakeDetection/app.py) | Removed QMC-FD UI elements (slider, expander, breakdown sections, mode tags, telemetry) |

### N. Test Results After Disabling QMC

```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
collected 8 items

tests/test_pipeline.py::test_compute_face_quality_empty          PASSED
tests/test_pipeline.py::test_compute_face_quality_tiny           PASSED
tests/test_pipeline.py::test_compute_face_quality_blurry         PASSED
tests/test_pipeline.py::test_compute_face_quality_valid_pattern  PASSED
tests/test_pipeline.py::test_align_and_crop_face_padding         PASSED
tests/test_pipeline.py::test_sample_video_frame_indices          PASSED
tests/test_pipeline.py::test_temporal_aggregator_anomaly_detection PASSED
tests/test_pipeline.py::test_detector_on_sample_video           PASSED

============================== 8 passed in 5.62s ==============================
```

**All 8 tests pass. QMC verified disabled (all cue scores = 0.0, decision_mode always PRIMARY_VIT).**
