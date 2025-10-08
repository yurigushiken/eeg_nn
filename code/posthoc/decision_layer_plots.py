"""
Decision layer comparison plots.

This module generates side-by-side comparison plots for baseline vs thresholded predictions.

Constitutional compliance: Section V (Audit-Ready Artifacts)

Functions:
    create_confusion_comparison_plot: Side-by-side confusion matrices
    create_per_fold_comparison_bars: Per-fold accuracy comparison bar chart
    generate_all_comparison_plots: Orchestrate all comparison plots
"""

from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def create_confusion_comparison_plot(
    baseline_rows: List[Dict],
    thresholded_rows: List[Dict],
    class_names: List[str],
    out_path: Path,
    title: str = "Overall Confusion: Baseline vs Thresholded"
):
    """
    Create side-by-side confusion matrices comparing baseline and thresholded.
    
    Args:
        baseline_rows: Baseline prediction rows
        thresholded_rows: Thresholded prediction rows
        class_names: List of class names
        out_path: Output PNG path
        title: Plot title
    """
    y_true_base = [row["true_label_idx"] for row in baseline_rows]
    y_pred_base = [row["pred_label_idx"] for row in baseline_rows]
    
    y_true_thresh = [row["true_label_idx"] for row in thresholded_rows]
    y_pred_thresh = [row["pred_label_idx"] for row in thresholded_rows]
    
    # Compute confusion matrices (row-normalized)
    cm_base = confusion_matrix(y_true_base, y_pred_base, labels=list(range(len(class_names))))
    cm_thresh = confusion_matrix(y_true_thresh, y_pred_thresh, labels=list(range(len(class_names))))
    
    # Row-normalize
    row_sums_base = cm_base.sum(axis=1, keepdims=True)
    row_sums_base[row_sums_base == 0] = 1
    cm_base_pct = (cm_base / row_sums_base) * 100
    
    row_sums_thresh = cm_thresh.sum(axis=1, keepdims=True)
    row_sums_thresh[row_sums_thresh == 0] = 1
    cm_thresh_pct = (cm_thresh / row_sums_thresh) * 100
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline
    im1 = ax1.imshow(cm_base_pct, cmap="Blues", vmin=0, vmax=100)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels(class_names)
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Baseline (Argmax)")
    
    # Add cell values
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if cm_base_pct[i, j] > 50 else "black"
            ax1.text(j, i, f"{cm_base_pct[i, j]:.1f}", ha="center", va="center",
                    color=text_color, fontsize=9)
    
    # Thresholded
    im2 = ax2.imshow(cm_thresh_pct, cmap="Blues", vmin=0, vmax=100)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Decision Layer (Ratio Rule)")
    
    # Add cell values
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if cm_thresh_pct[i, j] > 50 else "black"
            ax2.text(j, i, f"{cm_thresh_pct[i, j]:.1f}", ha="center", va="center",
                    color=text_color, fontsize=9)
    
    # Shared colorbar
    fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04, label="Percentage (%)")
    
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_per_fold_comparison_bars(
    fold_comparisons: List[Dict],
    out_path: Path
):
    """
    Create per-fold accuracy comparison bar chart.
    
    Args:
        fold_comparisons: List of fold comparison dicts (fold, baseline, thresholded)
        out_path: Output PNG path
    """
    folds = [fc["fold"] for fc in fold_comparisons]
    baseline_accs = [fc["baseline"]["acc"] for fc in fold_comparisons]
    thresholded_accs = [fc["thresholded"]["acc"] for fc in fold_comparisons]
    
    x = np.arange(len(folds))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, baseline_accs, width, label="Baseline (Argmax)", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, thresholded_accs, width, label="Decision Layer", color="coral", alpha=0.8)
    
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Fold Accuracy: Baseline vs Decision Layer", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_comparison_plots(
    baseline_rows: List[Dict],
    thresholded_rows: List[Dict],
    fold_comparisons: List[Dict],
    class_names: List[str],
    out_dir: Path
):
    """
    Generate all comparison plots.
    
    Args:
        baseline_rows: All baseline prediction rows
        thresholded_rows: All thresholded prediction rows
        fold_comparisons: List of per-fold comparison dicts
        class_names: List of class names
        out_dir: Output directory (plots_outer_threshold_compare)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall confusion comparison
    create_confusion_comparison_plot(
        baseline_rows=baseline_rows,
        thresholded_rows=thresholded_rows,
        class_names=class_names,
        out_path=out_dir / "confusion_comparison.png",
        title="Overall Confusion: Baseline vs Thresholded"
    )
    
    # Per-fold accuracy comparison
    create_per_fold_comparison_bars(
        fold_comparisons=fold_comparisons,
        out_path=out_dir / "per_fold_accuracy_comparison.png"
    )
    
    # Per-fold confusion matrices
    for fc in fold_comparisons:
        fold = fc["fold"]
        baseline_fold_rows = [r for r in baseline_rows if r.get("outer_fold") == fold]
        thresholded_fold_rows = [r for r in thresholded_rows if r.get("outer_fold") == fold]
        
        if baseline_fold_rows and thresholded_fold_rows:
            create_confusion_comparison_plot(
                baseline_rows=baseline_fold_rows,
                thresholded_rows=thresholded_fold_rows,
                class_names=class_names,
                out_path=out_dir / f"fold{fold}_confusion_comparison.png",
                title=f"Fold {fold} Confusion: Baseline vs Thresholded"
            )

