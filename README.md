# An AI-Based Deepfake Detection System for Video and Real-Time Webcam Analysis Using Vision Transformers

## Overview

Deepfakes represent synthetic media generated through deep learning techniques, such as autoencoders and generative adversarial networks (GANs), in which a person's facial likeness or expressions are altered or swapped. As deepfake synthesis becomes increasingly photorealistic, automated, reliable detection systems are critical for digital forensics, media integrity, and information security.

This project implements an AI-based deepfake detection system designed for both offline video file analysis and real-time live webcam streams. The system couples a pre-trained Vision Transformer (`ViT-Base/16`) with an end-to-end computer vision pre-processing and post-processing pipeline:
- **Video & Webcam Support:** Analyzes uploaded video files (`.mp4`, `.avi`, `.mov`) or live streams from a local camera device.
- **Pre-trained ViT-B/16 Backbone:** Employs patch-based self-attention to identify high-level spatial manipulation artifacts.
- **Multi-Backend Face Cascade:** Cascades through MTCNN, Google MediaPipe BlazeFace, and OpenCV Haar Cascade to maximize facial detection recall across challenging poses and lighting conditions.
- **Face Quality Gate:** Filters low-contrast, over/under-exposed, or heavily blurred frames before model inference.
- **Landmark Alignment & Contextual Crop:** Aligns faces via 5-point affine transformation and applies contextual padding ($\rho_{\text{pad}} = 0.10$) to capture boundary manipulation cues.
- **Quality & Certainty Weighted Temporal Aggregation:** Replaces naive averaging with exponential moving averages weighted by face image quality and model confidence.
- **Top-20% Suspicious Frame Pooling:** Focuses video-level classification on the top-20% highest fake-probability frames to detect localized or intermittent manipulations.
- **Burst Anomaly Detection:** Flags clusters of consecutive suspicious frames that indicate transient deepfake artifacts.
- **Streamlit Interface:** Provides an interactive web application with real-time video telemetry, forensic metric breakdowns, and report export.

---

## System Pipeline

```text
Video / Webcam Stream
        ↓
Frame Sampling (sample_video_frame_indices)
        ↓
Face Detection Cascade (MTCNN → MediaPipe BlazeFace → OpenCV Haar)
        ↓
Face Quality Gate (Laplacian variance, brightness, contrast, size)
        ↓
Landmark Alignment + Contextual Crop (5-point affine, ρ_pad = 0.10)
        ↓
Pre-trained ViT-B/16 Feature Extraction & Classification
        ↓
Softmax Probabilities: P(real), P(fake)
        ↓
Quality + Certainty Weighted Temporal Aggregation
        ↓
Top-20% Suspicious Frame Pooling
        ↓
Burst Anomaly Detection (consecutive suspicious frames)
        ↓
Final Calibrated Verdict (Likely Authentic / Suspicious / Highly Suspicious)
```

---

## Features

- **Video File Analysis:** Supports frame sampling, sequential facial tracking, temporal aggregation, and video-level verdict calculation.
- **Real-Time Webcam Analysis:** Processes live video input with periodic inference pacing to maintain smooth display rendering.
- **Multi-Backend Face Detection Cascade:** Robust multi-tier fallback (MTCNN $\rightarrow$ MediaPipe $\rightarrow$ Haar) ensuring high detection recall even under partial occlusion.
- **Face Quality Gate:** Eliminates low-quality, blurry, or extreme-illumination frames that degrade transformer classification accuracy.
- **Facial Landmark Alignment:** Standardizes head tilt and facial orientation using 5-point affine transformation.
- **Contextual Cropping:** Preserves outer facial contours and blending boundaries where synthesis artifacts commonly reside ($\rho_{\text{pad}} = 0.10$).
- **Pre-trained Vision Transformer Inference:** Evaluates global patch correlations across facial regions via multi-head self-attention.
- **Temporal Aggregation:** Exponential smoothing weighted by face sharpness, contrast, and model certainty.
- **Top-20% Suspicious Frame Pooling:** Mitigates dilution from predominantly authentic frames in selectively edited videos.
- **Burst Anomaly Detection:** Detects temporal bursts of consecutive manipulated frames ($\ge 3$ consecutive frames).
- **Forensic Visualization:** Frame-level confidence timelines, metric summaries, and confusion matrix charts.
- **JSON & CSV Export:** Export detailed per-frame and per-video forensic audit logs.
- **Local Processing:** Runs entirely on local hardware (CPU/GPU) without transmitting video data to external cloud services, safeguarding user privacy.

---

## Model

| Parameter | Specification |
| :--- | :--- |
| **Model Name** | `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| **Architecture** | Vision Transformer (`ViT-Base/16`) |
| **Input Resolution** | $224 \times 224$ pixels, 3 channels (RGB) |
| **Patch Size** | $16 \times 16$ |
| **Transformer Layers** | 12 |
| **Attention Heads** | 12 |
| **Hidden Size** | 768 |
| **Output Classes** | 2 (`Realism` / `Deepfake`) |

> **IMPORTANT NOTE:** This model is utilized strictly as a publicly available pre-trained inference backbone accessed via Hugging Face Transformers. The model was **not** trained or fine-tuned by our team.

---

## Face Detection

To guarantee dependable face localization across varied environments, the pipeline implements a three-tier cascade:

1. **MTCNN (Multi-task Cascaded Convolutional Networks) — Primary:** Employs a three-stage deep CNN network (P-Net, R-Net, O-Net) to detect faces and extract 5 facial landmarks (eyes, nose, mouth corners) with high precision.
2. **MediaPipe BlazeFace — Secondary:** If MTCNN fails to detect a face, the system automatically falls back to Google's lightweight BlazeFace model (`blaze_face_short_range.tflite`), optimized for sub-millisecond mobile and webcam inference.
3. **OpenCV Haar Cascade — Tertiary Fallback:** If deep learning detectors are unavailable or produce no candidate bounding box, an OpenCV frontal face Haar Cascade acts as a rapid, lightweight fallback.

---

## Temporal Decision Method

Single-frame classification often produces noisy predictions due to fleeting facial expressions, motion blur, or compression artifacts. The temporal aggregator calculates a robust video-level verdict through the following steps:

1. **Frame-Level Probabilities:** For each analyzed frame $t$, the ViT produces raw softmax probabilities $P(\text{real})_t$ and $P(\text{fake})_t$.
2. **Quality & Certainty Weighting:** Each frame is weighted by its normalized face quality score $Q_t \in [0, 1]$ (computed from Laplacian variance, brightness, and contrast) and classification certainty $C_t = |P(\text{fake})_t - 0.5| \times 2$:
   $$w_t = 0.5 \cdot Q_t + 0.5 \cdot C_t$$
3. **Weighted Temporal Aggregation:** Predictions are accumulated using an exponential moving average:
   $$\bar{P}_{\text{fake}} = \frac{\sum_{t=1}^N w_t \cdot P(\text{fake})_t}{\sum_{t=1}^N w_t}$$
4. **Top-20% Suspicious Frame Pooling:** To catch short-duration or localized face swaps that would be washed out by a video-level mean, the top $K = \lceil 0.20 \times N \rceil$ frames with the highest $P(\text{fake})_t$ are isolated and averaged into a pooled score $S_{\text{top-}k}$.
5. **Burst Anomaly Detection:** A sliding window scans for sequences of $\ge 3$ consecutive frames where $P(\text{fake})_t > 0.60$. If detected, an anomaly flag is raised.
6. **Video-Level Decision:** The final decision compares the composite score against the calibrated decision threshold $\tau_d = 0.40$:
   $$\text{Verdict} = \begin{cases} \text{Deepfake}, & \text{if } \bar{P}_{\text{fake}} \ge 0.40 \text{ or } S_{\text{top-}k} \ge 0.40 \text{ or Anomaly} \\ \text{Real}, & \text{otherwise} \end{cases}$$

---

## Dataset and Evaluation

The system was evaluated on a benchmark sample from the **DeepFake Detection (DFD)** dataset:

- **Total Videos:** 20 labeled video sequences
- **Distribution:** 10 authentic videos (`Real`) and 10 manipulated videos (`Fake`)
- **Total Frames Analyzed:** 770 frames (average 38.5 frames per video)
- **Decision Threshold ($\tau_d$):** 0.40
- **Contextual Padding ($\rho_{\text{pad}}$):** 0.10
- **Pooling Strategy:** Top-20% suspicious-frame pooling

### Evaluation Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **75.0%** (15 / 20 correct) |
| **Precision** | **77.8%** (7 / 9 positive predictions) |
| **Recall (Sensitivity)** | **70.0%** (7 / 10 fakes detected) |
| **F1-Score** | **73.7%** |
| **Specificity** | **80.0%** (8 / 10 authentic videos verified) |

### Confusion Matrix

| | Actual Real | Actual Fake |
| :--- | :---: | :---: |
| **Predicted Real** | **TN = 8** | **FN = 3** |
| **Predicted Fake** | **FP = 2** | **TP = 7** |

> **IMPORTANT EVALUATION NOTE:** These metrics reflect a small preliminary evaluation on 20 videos from the DFD dataset. They are reported transparently for academic inspection and must **not** be presented as large-scale benchmark performance across unconstrained datasets.

---

## Performance

The following benchmarks were recorded during live execution on an Intel/AMD CPU environment (Python 3.14, PyTorch, single-threaded inference):

| Metric | Measured Value | Note |
| :--- | :--- | :--- |
| **ViT Inference Latency** | **87.3 – 90.9 ms** | Raw forward pass per $224 \times 224$ face crop on CPU |
| **End-to-End Latency** | **114.0 – 191.0 ms** | Includes face detection, landmark alignment, cropping, ViT inference, and temporal update |
| **Video Processing Throughput** | **1.31 FPS** | Full pipeline throughput on video file analysis (770 frames processed in 588.72 s total) |
| **Webcam Display FPS** | **20 – 30 FPS** | UI camera frame acquisition and display loop |
| **Webcam ViT Inference Frequency** | **1 – 3 inferences/sec** | Paced inference to preserve fluent camera display responsiveness |

---

## Installation

### Prerequisites
- Python 3.9 or higher (tested on Python 3.10 – 3.14)
- Git
- Web camera (optional, required only for live webcam analysis)

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/sunilravulapati/Deepfake-Detection-System.git
cd Deepfake-Detection-System

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate

# 3. Upgrade pip and install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### 1. Streamlit Web Interface (Recommended)
Launch the interactive web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. From the web dashboard, you can:
- Upload video files (`.mp4`, `.avi`, `.mov`) for frame-by-frame analysis
- Launch the real-time webcam detection stream
- Inspect temporal timelines, quality distributions, and verdict breakdowns
- Export forensic audit logs as JSON and CSV

### 2. Standalone CLI Detection Script
Run the interactive command-line detector:
```bash
python deepfake_detector.py
```

### 3. Running the Test Suite
Execute the comprehensive test suite with `pytest`:
```bash
python -m pytest
```

### 4. Running the Benchmark Evaluation
Execute the standalone 20-video DFD evaluation runner:
```bash
python evaluation/run_evaluation.py
```

---

## Project Structure

```text
Deepfake-Detection-System/
├── app.py                      # Main Streamlit web application
├── pipeline.py                 # Core detection, quality gate & temporal aggregation
├── deepfake_detector.py        # CLI detection script (interactive video/webcam)
├── verdict_logic.py            # 5-tier calibrated verdict classification
├── research_dashboard.py       # Matplotlib figure generators for research UI
├── research_results.py         # Benchmark and dataset telemetry constants
├── custom_fallback.py          # Preserved QMC-FD heuristic fallback (disabled)
├── experiment_config.yaml      # Pipeline & experiment parameters (τd=0.40, ρpad=0.10)
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── .gitignore                  # Git ignore rules
├── paper.tex                   # Academic paper source (LaTeX)
│
├── dataset/                    # Evaluation video dataset (DFD)
│   ├── real/                   # Authentic reference videos (10 videos)
│   ├── fake/                   # Deepfake manipulated videos (10 videos)
│   └── metadata.csv            # Dataset ground-truth annotations
│
├── evaluation/                 # Benchmark evaluation scripts & outputs
│   ├── run_evaluation.py       # Standalone 20-video evaluation runner
│   ├── evaluation_results.csv  # Comprehensive per-video evaluation metrics
│   ├── per_video_results.csv   # Per-video breakdown
│   ├── evaluation_summary.json # Machine-readable metrics summary
│   └── EVALUATION_REPORT.md    # Full evaluation report
│
├── results/                    # Research figures, experiment data & reports
│   ├── confusion_matrix.png    # 20-video DFD evaluation confusion matrix
│   ├── metrics.png             # 20-video DFD evaluation performance metrics
│   ├── p_fake_comparison.png   # 20-video DFD real vs. fake P(fake) distribution
│   ├── fps_comparison.png      # Latency & throughput benchmark comparison
│   ├── csv/                    # Experiment ablation and threshold sweep CSVs
│   ├── figures/                # Research analysis charts and ROC curves
│   ├── json/                   # Experiment metrics JSON summaries
│   └── reports/                # Research experiment audit reports
│
├── docs/                       # Technical reports & documentation
│   └── deepfake_technical_report.md  # Comprehensive technical architecture audit
│
├── models/                     # Lightweight local model assets
│   └── blaze_face_short_range.tflite # MediaPipe BlazeFace detector model
│
└── tests/                      # Unit and integration test suite
    ├── test_pipeline.py        # Pipeline & temporal aggregation tests
    ├── test_verdict_logic.py   # Verdict classification band unit tests
    ├── test_custom_fallback.py # Preserved QMC fallback component tests
    ├── test_research_presentation.py # Metrics integrity tests
    ├── test_auc_unit.py        # AUC Mann-Whitney U metric calculation tests
    ├── test_detection_logic.py # Detection logic smoke tests
    ├── benchmark_structural_cue.py # Structural cue benchmark script
    ├── benchmark_temporal_pairs.py # Controlled temporal pairs benchmark
    ├── test_improved_temporal.py   # Articulation-gated temporal residual test
    └── inspect_videos.py       # Video inspection diagnostic script
```

---

## Limitations

- **Small Preliminary Evaluation Dataset:** Evaluated on a sample of 20 videos from the DFD benchmark; performance figures should not be extrapolated as universal generalization guarantees.
- **No Task-Specific Fine-Tuning:** The Vision Transformer operates using publicly available pre-trained weights without fine-tuning on the evaluation dataset.
- **Domain Shift:** Detector sensitivity may fluctuate when confronted with compression formats, resolutions, or synthesis techniques unrepresented in the pre-training data.
- **Single-Face Primary Tracking:** The current pipeline focuses primarily on the most prominent detected face per frame.
- **CPU Inference Throughput:** Full pipeline inference on CPU operates at ~1.31 FPS, making multi-frame video analysis computationally demanding without GPU acceleration.
- **External Model Dependency:** Relies on the architecture and feature representations learned by `prithivMLmods/Deep-Fake-Detector-v2-Model`.

---

## Future Scope

- **Large-Scale Multi-Dataset Benchmarking:** Comprehensive cross-dataset evaluations on FaceForensics++, Celeb-DF v2, and DFDC.
- **Parameter-Efficient Fine-Tuning (PEFT):** Adapting the Vision Transformer backbone via LoRA (Low-Rank Adaptation) on domain-specific facial forgery datasets.
- **Spatiotemporal Transformer Backbones:** Integrating end-to-end video transformers (e.g., TimeSformer, VideoMAE) to learn joint spatial and temporal cues directly.
- **Simultaneous Multi-Face Tracking:** Extending facial tracking to evaluate multiple subjects concurrently within crowded scenes.
- **Multimodal Audio-Visual Detection:** Incorporating audio-visual synchronization analysis to detect speech-lip desynchronization and synthetic voice generation.

---

## Research / Academic Note

This project is developed solely for research, academic experimentation, and preliminary media screening. It does **not** constitute a legally definitive forensic verification tool. In forensic, legal, or high-stakes contexts, automated model outputs should always be corroborated by trained digital forensics specialists through multi-faceted evidence analysis.
