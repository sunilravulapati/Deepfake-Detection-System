"""
research_dashboard.py
=====================
All research-grade matplotlib figures for the Deepfake Detection project.

Each public function returns:
  fig : matplotlib.figure.Figure

Callers (e.g. app.py) must call plt.close(fig) after using it.

Design goals:
- White background, suitable for IEEE/academic insertion.
- Clear axis labels, titles, captions.
- Publication-quality DPI (150 for screen preview, 300 for download).
- No Streamlit-specific imports here — pure matplotlib.
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from research_results import (
    DATASET, CONFUSION, METRICS, LATENCY, PER_VIDEO, VERDICT_BANDS, MODEL_INFO
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared style
# ──────────────────────────────────────────────────────────────────────────────
_REAL_COLOR  = "#2563eb"   # blue — real videos
_FAKE_COLOR  = "#dc2626"   # red  — fake videos
_NEUTRAL     = "#64748b"   # slate
_GREEN       = "#16a34a"
_AMBER       = "#d97706"
_FONT_TITLE  = {"fontsize": 12, "fontweight": "bold"}
_FONT_LABEL  = {"fontsize": 10}
_FONT_TICK   = 9
_CAPTION_FS  = 8

_COMMON_RCPARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cbd5e1",
    "axes.grid":        True,
    "grid.color":       "#e2e8f0",
    "grid.linewidth":   0.7,
    "font.family":      "DejaVu Sans",
    "xtick.color":      "#374151",
    "ytick.color":      "#374151",
    "axes.labelcolor":  "#1e293b",
    "text.color":       "#1e293b",
}


def _apply_style():
    plt.rcParams.update(_COMMON_RCPARAMS)


def _fig_bytes(fig, fmt: str = "png", dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Class Distribution
# ──────────────────────────────────────────────────────────────────────────────
def fig_class_distribution() -> plt.Figure:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Figure 1 — Dataset Class Distribution", **_FONT_TITLE)

    # Bar chart — left
    ax = axes[0]
    labels   = ["Real", "Fake"]
    counts   = [DATASET["real"], DATASET["fake"]]
    colors   = [_REAL_COLOR, _FAKE_COLOR]
    bars = ax.bar(labels, counts, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, max(counts) + 1.5)
    ax.set_ylabel("Number of Videos", **_FONT_LABEL)
    ax.set_title("Total Videos by Class", fontsize=11)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Donut — right
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, autopct="%1.0f%%",
        colors=colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")
    ax2.set_title("Class Balance (50 / 50)", fontsize=11)

    fig.text(0.5, -0.04,
             "Preliminary evaluation — 6 videos total. "
             "Results are not representative of general detection performance.",
             ha="center", fontsize=_CAPTION_FS, color=_NEUTRAL, style="italic")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────
def fig_confusion_matrix() -> plt.Figure:
    _apply_style()
    plt.rcParams["axes.grid"] = False

    TP = CONFUSION["TP"]
    TN = CONFUSION["TN"]
    FP = CONFUSION["FP"]
    FN = CONFUSION["FN"]
    matrix = np.array([[TP, FN],
                        [FP, TN]])

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Figure 2 — Confusion Matrix (Test Split)", **_FONT_TITLE)

    # Colour grid — darker = more cells
    max_val = matrix.max() if matrix.max() > 0 else 1
    cell_colors = [
        ["#dc2626", "#fca5a5"],   # Fake row: TP (correct-fake), FN (missed)
        ["#fee2e2", "#16a34a"],   # Real row: FP (false alarm), TN (correct-real)
    ]

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                        color=cell_colors[i][j], alpha=0.90))
            tag = {(0, 0): "TP", (0, 1): "FN", (1, 0): "FP", (1, 1): "TN"}[(i, j)]
            ax.text(j + 0.5, 1 - i + 0.58, str(val),
                    ha="center", va="center", fontsize=28, fontweight="bold", color="white")
            ax.text(j + 0.5, 1 - i + 0.28, tag,
                    ha="center", va="center", fontsize=11, color="white", alpha=0.9)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted Fake", "Predicted Real"], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual Real", "Actual Fake"], fontsize=10)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Predicted Label", **_FONT_LABEL)
    ax.set_ylabel("Actual Label", **_FONT_LABEL)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.5, -0.04,
             f"TP={TP}  TN={TN}  FP={FP}  FN={FN} | Test set: 4 videos | "
             "Positive class = Fake",
             ha="center", fontsize=_CAPTION_FS, color=_NEUTRAL, style="italic")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Metrics Comparison
# ──────────────────────────────────────────────────────────────────────────────
def fig_metrics_comparison() -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Figure 3 — Performance Metrics Comparison (Test Split)", **_FONT_TITLE)

    metric_keys  = ["accuracy", "precision", "recall", "f1"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    configs      = list(METRICS.values())
    n_metrics    = len(metric_keys)
    n_configs    = len(configs)
    x            = np.arange(n_metrics)
    width        = 0.30
    colors       = ["#2563eb", "#7c3aed"]

    for ci, (cfg, col) in enumerate(zip(configs, colors)):
        vals = [cfg[k] for k in metric_keys]
        offset = (ci - (n_configs - 1) / 2) * (width + 0.05)
        bars = ax.bar(x + offset, vals, width, label=cfg["label"],
                      color=col, alpha=0.85, edgecolor="white", linewidth=1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{v*100:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Score (0–1)", **_FONT_LABEL)
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.axhline(0, color="#94a3b8", linewidth=0.8)

    fig.text(0.5, -0.04,
             "Both configurations produced identical results on the four-video test subset.",
             ha="center", fontsize=_CAPTION_FS, color=_NEUTRAL, style="italic")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Video-Level P(fake) Comparison
# ──────────────────────────────────────────────────────────────────────────────
def fig_pfake_comparison() -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Figure 4 — Video-Level Manipulation Probability (Test Split)", **_FONT_TITLE)

    names    = [v["name"].replace("_", "\n") for v in PER_VIDEO]
    pfakes   = [v["p_fake"] for v in PER_VIDEO]
    gts      = [v["ground_truth"] for v in PER_VIDEO]
    bar_cols = [_REAL_COLOR if gt == "Real" else _FAKE_COLOR for gt in gts]
    edge_cols= ["#1d4ed8" if gt == "Real" else "#991b1b" for gt in gts]

    bars = ax.bar(names, pfakes, color=bar_cols, edgecolor=edge_cols,
                  linewidth=1.2, alpha=0.88, width=0.5)

    # Decision threshold line
    ax.axhline(0.50, color="#1e293b", linewidth=1.4, linestyle="--",
               label="Decision threshold (τ = 0.50)")

    for bar, val, pv in zip(bars, pfakes, PER_VIDEO):
        lbl = f"{val*100:.1f}%"
        if not pv["correct"]:
            lbl += f"\n({pv['error_type']})"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015, lbl,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("P(fake)", **_FONT_LABEL)
    ax.set_ylim(0, 0.85)
    legend_patches = [
        mpatches.Patch(color=_REAL_COLOR, label="Ground truth: Real"),
        mpatches.Patch(color=_FAKE_COLOR, label="Ground truth: Fake"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="#1e293b", linestyle="--",
                   label="Decision threshold (τ = 0.50)")
    ], fontsize=9, framealpha=0.85)

    fig.text(0.5, -0.06,
             "This preliminary result suggests a domain mismatch between the pre-trained ViT "
             "and the evaluated DeeperForensics samples.\n"
             "Both fake videos score below the 0.50 threshold; one real video crosses it.",
             ha="center", fontsize=_CAPTION_FS, color=_NEUTRAL, style="italic")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Latency Comparison
# ──────────────────────────────────────────────────────────────────────────────
def fig_latency_comparison() -> plt.Figure:
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle("Figure 5 — Processing Latency Comparison (CPU)", **_FONT_TITLE)

    configs = list(LATENCY.values())
    labels  = [c["label"].replace(" (Config ", "\n(") for c in configs]
    colors  = ["#2563eb", "#7c3aed"]

    # Chart 1: ViT inference
    vals1 = [c["vit_inference_ms"] for c in configs]
    b1 = axes[0].bar(labels, vals1, color=colors, alpha=0.88,
                     edgecolor="white", linewidth=1, width=0.45)
    axes[0].set_title("ViT Inference Latency", fontsize=11)
    axes[0].set_ylabel("Latency (ms)", **_FONT_LABEL)
    axes[0].set_ylim(0, max(vals1) * 1.3)
    for bar, v in zip(b1, vals1):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, f"{v:.2f} ms",
                     ha="center", va="bottom", fontsize=9)

    # Chart 2: End-to-end
    vals2 = [c["end_to_end_ms"] for c in configs]
    b2 = axes[1].bar(labels, vals2, color=colors, alpha=0.88,
                     edgecolor="white", linewidth=1, width=0.45)
    axes[1].set_title("End-to-End Latency", fontsize=11)
    axes[1].set_ylabel("Latency (ms)", **_FONT_LABEL)
    axes[1].set_ylim(0, max(vals2) * 1.3)
    for bar, v in zip(b2, vals2):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, f"{v:.2f} ms",
                     ha="center", va="bottom", fontsize=9)

    # Chart 3: FPS
    vals3 = [c["fps"] for c in configs]
    b3 = axes[2].bar(labels, vals3, color=colors, alpha=0.88,
                     edgecolor="white", linewidth=1, width=0.45)
    axes[2].set_title("Effective FPS", fontsize=11)
    axes[2].set_ylabel("Frames per Second", **_FONT_LABEL)
    axes[2].set_ylim(0, max(vals3) * 1.35)
    for bar, v in zip(b3, vals3):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.03, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)

    fig.text(0.5, -0.04,
             "Temporal aggregation adds negligible computational overhead "
             "because it operates on already-computed frame probabilities.",
             ha="center", fontsize=_CAPTION_FS, color=_NEUTRAL, style="italic")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5b — Effective FPS Comparison (Standalone)
# ──────────────────────────────────────────────────────────────────────────────
def fig_fps_comparison() -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.suptitle("Figure 5b — Effective Processing Throughput (FPS, CPU)", **_FONT_TITLE)

    configs = list(LATENCY.values())
    labels  = [c["label"].replace(" (Config ", "\n(") for c in configs]
    colors  = ["#2563eb", "#7c3aed"]
    vals    = [c["fps"] for c in configs]

    bars = ax.bar(labels, vals, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1, width=0.42)
    ax.set_ylabel("Frames per Second (FPS)", **_FONT_LABEL)
    ax.set_ylim(0, max(vals) * 1.35)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08, f"{v:.2f} FPS",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.text(0.5, -0.05,
             "Temporal aggregation adds negligible overhead;\n"
             "throughput is bottlenecked by frame-level ViT inference on CPU.",
             ha="center", fontsize=_CAPTION_FS, color=_NEUTRAL, style="italic")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — Average Face Quality Comparison
# ──────────────────────────────────────────────────────────────────────────────
def fig_face_quality() -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Figure 6 — Average Face Quality by Video (Test Split)", **_FONT_TITLE)

    names    = [v["name"].replace("_", "\n") for v in PER_VIDEO]
    quals    = [v["avg_quality"] for v in PER_VIDEO]
    gts      = [v["ground_truth"] for v in PER_VIDEO]
    bar_cols = [_REAL_COLOR if gt == "Real" else _FAKE_COLOR for gt in gts]

    bars = ax.bar(names, quals, color=bar_cols, alpha=0.88,
                  edgecolor="white", linewidth=1, width=0.5)
    ax.set_ylabel("Average Face Quality Score (0–100)", **_FONT_LABEL)
    ax.set_ylim(0, 105)
    ax.axhline(70, color=_AMBER, linewidth=1.2, linestyle="--", label="Quality = 70 (reference)")

    for bar, val in zip(bars, quals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8, f"{val:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=_REAL_COLOR, label="Ground truth: Real"),
        mpatches.Patch(color=_FAKE_COLOR, label="Ground truth: Fake"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=_AMBER, linestyle="--", label="Quality = 70 (reference)")
    ], fontsize=9, framealpha=0.85)

    fig.text(0.5, -0.06,
             "The fake videos in this evaluation set had lower average face-quality scores "
             "than the real videos.\n"
             "Lower quality does not itself imply manipulation, "
             "but the difference may affect model behavior.",
             ha="center", fontsize=_CAPTION_FS, color=_NEUTRAL, style="italic")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 / Figure 1 — System Architecture & Detection Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def fig_pipeline() -> plt.Figure:
    _apply_style()
    plt.rcParams["axes.grid"] = False

    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.suptitle("Figure 1 — System Architecture & Detection Pipeline", **_FONT_TITLE)
    ax.set_xlim(-0.3, 5.3)
    ax.set_ylim(-0.5, 2.3)
    ax.axis("off")

    # 10 pipeline steps organized in 2 rows of 5
    row1 = [
        ("1. Input Stream", "Video / Webcam\nRGB Frames", "#1e3a5f", "white"),
        ("2. Sampling", "Frame Sampling\n(Target 3 FPS)", "#1d4ed8", "white"),
        ("3. Face Detection", "Cascade Detector\nMTCNN → MP → Haar", "#1d4ed8", "white"),
        ("4. Quality Filter", "Multi-Cue Filter\n(q >= 15 threshold)", "#1d4ed8", "white"),
        ("5. Alignment", "Landmark Align +\nContextual Crop", "#1d4ed8", "white"),
    ]
    row2 = [
        ("6. Backbone", "Pre-trained ViT-B/16\n(prithivMLmods)", "#7c3aed", "white"),
        ("7. Classification", "Softmax Probabilities\nP(real), P(fake)", "#7c3aed", "white"),
        ("8. Aggregation", "Temporal Aggregator\nQuality & Certainty", "#0f766e", "white"),
        ("9. Glitch Check", "Burst Detection\nConsecutive Glitches", "#0f766e", "white"),
        ("10. Output", "Decision & Verdict\nInterpretation Bands", "#dc2626", "white"),
    ]

    box_w = 0.84
    box_h = 0.72

    # Draw Row 1 (top, y=1.5, x=0..4)
    y1 = 1.45
    for i, (title, desc, bg, fg) in enumerate(row1):
        x = i * 1.1 + 0.4
        fancy = mpatches.FancyBboxPatch(
            (x - box_w / 2, y1 - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.03",
            facecolor=bg, edgecolor="white", linewidth=1.5, zorder=2
        )
        ax.add_patch(fancy)
        ax.text(x, y1 + 0.16, title, ha="center", va="center", fontsize=8.5,
                color="#f8fafc", fontweight="bold", zorder=3)
        ax.text(x, y1 - 0.12, desc, ha="center", va="center", fontsize=7.8,
                color=fg, zorder=3)
        if i < 4:
            ax.annotate("", xy=(x + box_w / 2 + 0.22, y1),
                        xytext=(x + box_w / 2 + 0.04, y1),
                        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4), zorder=1)

    # Connecting arrow from Row 1 step 5 down to Row 2 step 6
    ax.annotate("", xy=(4.8, 0.75), xytext=(4.8, 1.05),
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.6), zorder=1)

    # Draw Row 2 (bottom, y=0.35, x=4..0 right to left or 0..4 left to right)
    # Drawing left-to-right matching numerical order 6..10
    # Connecting arrow from 5 down and across to 6
    y2 = 0.35
    for i, (title, desc, bg, fg) in enumerate(row2):
        x = i * 1.1 + 0.4
        fancy = mpatches.FancyBboxPatch(
            (x - box_w / 2, y2 - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.03",
            facecolor=bg, edgecolor="white", linewidth=1.5, zorder=2
        )
        ax.add_patch(fancy)
        ax.text(x, y2 + 0.16, title, ha="center", va="center", fontsize=8.5,
                color="#f8fafc", fontweight="bold", zorder=3)
        ax.text(x, y2 - 0.12, desc, ha="center", va="center", fontsize=7.8,
                color=fg, zorder=3)
        if i < 4:
            ax.annotate("", xy=(x + box_w / 2 + 0.22, y2),
                        xytext=(x + box_w / 2 + 0.04, y2),
                        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4), zorder=1)

    # Curved connector from 5 to 6: from (4.8, 1.09) to (0.4, 0.71)
    ax.annotate("", xy=(0.4, y2 + box_h / 2 + 0.03),
                xytext=(4.8, y1 - box_h / 2 - 0.03),
                arrowprops=dict(arrowstyle="->", color="#6366f1", lw=1.5,
                                connectionstyle="angle,angleA=-90,angleB=180,rad=10"),
                zorder=1)

    # Legend
    legend_items = [
        (mpatches.Patch(color="#1e3a5f"), "Input Stream"),
        (mpatches.Patch(color="#1d4ed8"), "System Preprocessing"),
        (mpatches.Patch(color="#7c3aed"), "Pre-trained ViT Model"),
        (mpatches.Patch(color="#0f766e"), "System Temporal Aggregation"),
        (mpatches.Patch(color="#dc2626"), "Verdict Output"),
    ]
    ax.legend(handles=[h for h, _ in legend_items],
              labels=[l for _, l in legend_items],
              loc="lower center", ncol=5, fontsize=8.5, framealpha=0.95,
              bbox_to_anchor=(0.5, -0.16))

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8 — Per-Video Error Table (visual)
# ──────────────────────────────────────────────────────────────────────────────
def fig_per_video_table() -> plt.Figure:
    _apply_style()
    plt.rcParams["axes.grid"] = False

    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.suptitle("Figure 8 — Per-Video Error Analysis (Test Split, Config A)", **_FONT_TITLE)
    ax.axis("off")

    col_labels = ["Video", "Ground Truth", "Prediction", "P(fake)", "Quality", "Result"]
    rows = []
    cell_colors = []
    for v in PER_VIDEO:
        res = "✓ Correct" if v["correct"] else f"✗ {v['error_type']}"
        rows.append([
            v["name"],
            v["ground_truth"],
            v["prediction"],
            f"{v['p_fake']*100:.2f}%",
            f"{v['avg_quality']:.2f}",
            res,
        ])
        row_color = ["#dcfce7" if v["correct"] else "#fee2e2"] * 6
        cell_colors.append(row_color)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.3, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Helper: get bytes for a figure (used by Streamlit download buttons)
# ──────────────────────────────────────────────────────────────────────────────
def get_fig_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    """Return figure as bytes for download. Caller must close fig afterward."""
    return _fig_bytes(fig, fmt=fmt, dpi=dpi)
