"""Shared plotting utilities for confusion matrices and training curves.

All engines (CNN, Hybrid, ViT, …) should import these helpers so that every
run produces visually identical artefacts.

plot_confusion(...)
plot_curves(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, List

import numpy as np

# --- Headless backend to prevent GUI popups / Windows crashes ---
import matplotlib  # must be set before importing pyplot or seaborn
matplotlib.use("Agg", force=True)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix

# Global style – matches what the ViT / Hybrid scripts used
sns.set(style="white", font="DejaVu Sans", font_scale=1.0)

__all__ = ["plot_confusion", "plot_curves"]


def _ensure_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)


def plot_confusion(y_true, y_pred, class_names, out_fp, title: str | None = None):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_percent = cm * 100
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues', xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=100)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_fp)
    plt.close()


def plot_curves(tr_hist, va_hist, va_acc_hist, out_fp: str, title: str | None = None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Plot loss curves
    ax1.plot(tr_hist, label='Train Loss')
    ax1.plot(va_hist, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot validation accuracy
    ax2.plot(va_acc_hist, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    if title:
        fig.suptitle(title, fontsize=16)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)
    plt.savefig(out_fp)
    plt.close()


