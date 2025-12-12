"""
Writers for temporal generalization artifacts.

We keep these separate from the baseline writers to avoid changing existing
CSV formats (and to minimize disruption to downstream scripts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import csv


def write_generalization_test_predictions(run_dir: Path, rows: List[Dict]) -> Path:
    """
    Write per-trial predictions across test windows.

    Output: test_predictions_outer_generalization.csv
    """
    out_path = run_dir / "test_predictions_outer_generalization.csv"
    if not rows:
        return out_path

    fieldnames = [
        # baseline outer columns
        "outer_fold",
        "trial_index",
        "subject_id",
        "true_label_idx",
        "true_label_name",
        "pred_label_idx",
        "pred_label_name",
        "correct",
        "p_trueclass",
        "logp_trueclass",
        "probs",
        # generalization metadata
        "train_window_start",
        "train_window_end",
        "train_window_center",
        "test_window_start",
        "test_window_end",
        "test_window_center",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return out_path


def write_generalization_outer_eval_metrics(run_dir: Path, rows: List[Dict]) -> Path:
    """
    Write per-fold evaluation metrics for each (train_window, test_window) cell.

    Output: outer_eval_metrics_generalization.csv
    """
    out_path = run_dir / "outer_eval_metrics_generalization.csv"
    if not rows:
        return out_path

    fieldnames = [
        # baseline metrics columns
        "outer_fold",
        "test_subjects",
        "n_test_trials",
        "acc",
        "acc_std",
        "macro_f1",
        "macro_f1_std",
        "min_per_class_f1",
        "min_per_class_f1_std",
        "plur_corr",
        "plur_corr_std",
        "cohen_kappa",
        "cohen_kappa_std",
        "per_class_f1",
        # generalization metadata
        "train_window_start",
        "train_window_end",
        "train_window_center",
        "test_window_start",
        "test_window_end",
        "test_window_center",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return out_path


