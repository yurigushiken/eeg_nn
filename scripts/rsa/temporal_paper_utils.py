"""
Small utilities used by paper-style temporal RSA figures.

Kept intentionally lightweight so multiple plotting scripts can share the same
time-span and statistical helpers without duplicating logic.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


def timepoint_edges(times: Sequence[float]) -> np.ndarray:
    """
    Compute bin edges for a sequence of time-window centers.

    Example:
        times = [25, 45, 65] -> edges = [15, 35, 55, 75]
    """
    t = np.asarray(list(times), dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a non-empty 1D sequence")
    t = np.sort(t)
    if t.size == 1:
        # Degenerate: assume a 1 ms bin on each side.
        return np.array([t[0] - 0.5, t[0] + 0.5], dtype=float)

    diffs = np.diff(t)
    if np.any(diffs <= 0):
        raise ValueError("times must be strictly increasing after sorting")

    edges = np.empty(t.size + 1, dtype=float)
    edges[1:-1] = (t[:-1] + t[1:]) / 2.0
    edges[0] = t[0] - diffs[0] / 2.0
    edges[-1] = t[-1] + diffs[-1] / 2.0
    return edges


def mask_to_spans(times: Sequence[float], mask: Sequence[bool]) -> List[Tuple[float, float]]:
    """
    Convert a boolean mask over timepoints into x-spans (start,end) for plotting.

    Spans are computed using `timepoint_edges(times)` so they cover full bins.
    """
    t = np.asarray(list(times), dtype=float)
    m = np.asarray(list(mask), dtype=bool)
    if t.ndim != 1 or m.ndim != 1 or t.size != m.size:
        raise ValueError("times and mask must be 1D and have the same length")

    edges = timepoint_edges(t)

    spans: List[Tuple[float, float]] = []
    start: int | None = None
    for i, is_on in enumerate(m):
        if is_on and start is None:
            start = i
        if (not is_on) and start is not None:
            spans.append((float(edges[start]), float(edges[i])))
            start = None
    if start is not None:
        spans.append((float(edges[start]), float(edges[-1])))
    return spans


def fisher_z(values: Iterable[float], *, eps: float = 1e-6) -> np.ndarray:
    """Fisher z-transform with clipping to avoid ±1 infinities."""
    v = np.asarray(list(values), dtype=float)
    v = np.clip(v, -1.0 + eps, 1.0 - eps)
    return np.arctanh(v)


def sem(values: Iterable[float]) -> float:
    """Standard error of the mean (NaN-safe)."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(np.std(v, ddof=1) / math.sqrt(v.size))


def one_sided_p_from_two_sided(t_stat: float, p_two_sided: float, *, alternative: str = "greater") -> float:
    """
    Convert a two-sided p-value to a one-sided p-value using the sign of t.

    alternative:
      - "greater": H1 mean > null
      - "less":    H1 mean < null
    """
    if not np.isfinite(t_stat) or not np.isfinite(p_two_sided):
        return float("nan")
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")

    if alternative == "greater":
        return float(p_two_sided / 2.0) if t_stat > 0 else float(1.0 - p_two_sided / 2.0)
    return float(p_two_sided / 2.0) if t_stat < 0 else float(1.0 - p_two_sided / 2.0)


def ttest_1samp_greater(values: Iterable[float], *, popmean: float) -> Tuple[float, float, int]:
    """
    One-sample t-test (one-sided, greater-than) with NaN handling.

    Returns:
        (t_stat, p_one_sided, n)
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 2:
        return float("nan"), float("nan"), n

    # Handle degenerate variance explicitly (scipy can return NaNs here).
    std = float(np.std(v, ddof=1))
    mean = float(np.mean(v))
    if std == 0.0:
        if mean > popmean:
            return float("inf"), 0.0, n
        if mean == popmean:
            return 0.0, 1.0, n
        return float("-inf"), 1.0, n

    t_stat, p_two = stats.ttest_1samp(v, popmean=popmean)
    return float(t_stat), one_sided_p_from_two_sided(float(t_stat), float(p_two), alternative="greater"), n


