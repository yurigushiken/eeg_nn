"""
Temporal generalization utilities.

Goal:
- Train classifier at one time window (train window), then evaluate it across
  many test windows to measure representational stability over time.

This module is intentionally small and testable; the training orchestration
stays in TrainingRunner while evaluation helpers live here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


Window = Tuple[int, int]  # (start_ms, end_ms)


def generate_temporal_windows(epoch_ms: float, window_ms: float, stride_ms: float) -> List[Window]:
    """Generate sliding windows: (start_ms, end_ms)."""
    windows: List[Window] = []
    start = 0.0
    while start + window_ms <= epoch_ms + 1e-9:
        windows.append((int(round(start)), int(round(start + window_ms))))
        start += stride_ms
    return windows


def validate_datasets_compatible(train_dataset, test_dataset) -> None:
    """
    Ensure datasets are compatible for cross-window evaluation:
    - same length
    - same groups ordering (subject ids per trial)
    """
    if len(train_dataset) != len(test_dataset):
        raise ValueError(f"Dataset length mismatch: train={len(train_dataset)} test={len(test_dataset)}")

    train_groups = np.asarray(getattr(train_dataset, "groups", None))
    test_groups = np.asarray(getattr(test_dataset, "groups", None))
    if train_groups.size and test_groups.size:
        if train_groups.shape != test_groups.shape or not np.array_equal(train_groups, test_groups):
            raise ValueError("Dataset groups mismatch across windows (trial ordering differs).")


def _make_test_loader(dataset, te_idx: np.ndarray, batch_size: int) -> DataLoader:
    # Mirror outer test loader defaults: shuffle False, num_workers=0 (Windows-safe).
    ds_eval = copy.copy(dataset)
    if hasattr(ds_eval, "set_transform"):
        ds_eval.set_transform(None)
    return DataLoader(
        Subset(ds_eval, te_idx),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )


def evaluate_generalization_for_fold(
    *,
    cfg: Dict,
    outer_evaluator,
    model_builder: Callable,
    num_classes: int,
    inner_results: List[Dict],
    datasets_by_window: Mapping[Window, object],
    groups: np.ndarray,
    te_idx: np.ndarray,
    class_names: List[str],
    fold: int,
    input_adapter: Callable | None,
    train_window: Window,
) -> List[Dict]:
    """
    Evaluate one trained fold (inner ensemble states) across multiple test windows.

    Returns:
        A flat list of per-trial prediction rows, each augmented with:
          - train_window_{start,end,center}
          - test_window_{start,end,center}
    """
    batch_size = int(cfg.get("batch_size", 16))

    # Ensure deterministic ordering of windows in output (by start time)
    windows_sorted = sorted(datasets_by_window.keys(), key=lambda w: (w[0], w[1]))

    out_rows: List[Dict] = []
    for (tw_start, tw_end) in windows_sorted:
        ds = datasets_by_window[(tw_start, tw_end)]
        # groups array should align to dataset indices; validate in caller if needed
        te_loader = _make_test_loader(ds, te_idx=te_idx, batch_size=batch_size)
        eval_res = outer_evaluator.evaluate(
            model_builder=model_builder,
            num_classes=num_classes,
            inner_results=inner_results,
            test_loader=te_loader,
            groups=groups,
            te_idx=te_idx,
            class_names=class_names,
            fold=int(fold),
            input_adapter=input_adapter,
        )
        rows = list(eval_res.get("test_pred_rows", []))
        for r in rows:
            r["train_window_start"] = int(train_window[0])
            r["train_window_end"] = int(train_window[1])
            r["train_window_center"] = float((train_window[0] + train_window[1]) / 2.0)
            r["test_window_start"] = int(tw_start)
            r["test_window_end"] = int(tw_end)
            r["test_window_center"] = float((tw_start + tw_end) / 2.0)
        out_rows.extend(rows)

    return out_rows


