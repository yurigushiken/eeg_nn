"""
Shared RDM (Representational Dissimilarity Matrix) model builders and helpers.

This module centralizes model-RDM definitions so both static and temporal RSA
analyses use identical assumptions.

Models implemented:
- Pixel (e-only target): absolute differences in white_pixel_area for 1e..6e.
- ANS log-ratio: |log(i) - log(j)|.
- PI-ANS (project-specific): PI set 1-4 uses absolute distance, ANS set 5-6 uses
  the real ANS log-ratio for 5v6, and cross PI↔ANS pairs are separated by a
  large boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


Code = Union[int, np.integer]


def _to_int(code: Code) -> int:
    return int(code)  # np.int64 etc.


def code_to_numerosity(code: Code) -> int:
    """
    Convert a condition code to numerosity.

    Supports:
    - 11,22,...,66  -> 1..6  (cardinality codes)
    - 1..6          -> 1..6  (digit codes)
    """
    c = _to_int(code)
    if 1 <= c <= 9:
        return c
    if c % 11 == 0:
        return c // 11
    raise ValueError(f"Unsupported code for numerosity conversion: {code}")


def build_ans_log_ratio_rdm(codes: Sequence[Code]) -> np.ndarray:
    """ANS model RDM: d(i,j)=|log(i)-log(j)| using numerosity space (1..6)."""
    nums = [code_to_numerosity(c) for c in codes]
    n = len(nums)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mat[i, j] = abs(np.log(nums[i]) - np.log(nums[j]))
    return mat


def build_pixel_rdm_e_only(stimuli_csv: Path, codes: Sequence[Code]) -> np.ndarray:
    """
    Pixel model RDM (e-only target): absolute differences in white_pixel_area.

    The stimuli CSV is expected to contain rows like:
      filename,dot_count,white_pixel_area
      1e.jpg,1,18233
    """
    df = pd.read_csv(stimuli_csv)
    if "filename" not in df.columns or "dot_count" not in df.columns or "white_pixel_area" not in df.columns:
        raise ValueError(f"Stimuli CSV missing required columns: {stimuli_csv}")

    df = df[df["filename"].astype(str).str.contains(r"e\.jpg$", regex=True)].copy()
    if df.empty:
        raise ValueError(f"No e-only stimuli rows found in {stimuli_csv}")

    # Map to cardinality codes (11,22,...) to match how RSA uses codes.
    df["code"] = df["dot_count"].astype(int) * 11
    area_map: Dict[int, float] = df.set_index("code")["white_pixel_area"].astype(float).to_dict()

    # Resolve codes in either 11/22 form or 1..6 digit form.
    resolved_codes: list[int] = []
    for c in codes:
        ci = _to_int(c)
        if 1 <= ci <= 9:
            ci = ci * 11
        resolved_codes.append(ci)

    missing = [c for c in resolved_codes if c not in area_map]
    if missing:
        raise KeyError(f"Missing pixel area for codes {missing} in {stimuli_csv}")

    n = len(resolved_codes)
    mat = np.zeros((n, n), dtype=float)
    for i, ci in enumerate(resolved_codes):
        for j, cj in enumerate(resolved_codes):
            if i == j:
                continue
            mat[i, j] = abs(area_map[ci] - area_map[cj])
    return mat


def build_pi_ans_rdm(codes: Sequence[Code], *, boundary: float = 10.0) -> np.ndarray:
    """
    PI-ANS model RDM (project-specific).

    - PI set: 1-4 uses absolute distance
    - ANS set: 5-6 uses real ANS log-ratio for 5v6
    - Cross PI↔ANS: boundary + scaled abs distance

    boundary is chosen so every cross pair is larger than any within-PI pair.
    """
    nums = [code_to_numerosity(c) for c in codes]
    n = len(nums)
    mat = np.zeros((n, n), dtype=float)

    # Choose weights so within-PI max is 3 (|1-4|) and cross dominates that.
    w_pi = 1.0
    w_cross = 0.25

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = nums[i], nums[j]
            in_pi_a = a <= 4
            in_pi_b = b <= 4
            in_ans_a = a >= 5
            in_ans_b = b >= 5

            if in_pi_a and in_pi_b:
                mat[i, j] = w_pi * abs(a - b)
            elif in_ans_a and in_ans_b:
                mat[i, j] = abs(np.log(a) - np.log(b))
            else:
                mat[i, j] = boundary + w_cross * abs(a - b)
    return mat


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman r for two 1D vectors; returns NaN if undefined."""
    r, _ = spearmanr(x, y)
    return float(r)


def partial_spearman_r(brain_vec: np.ndarray, theory_vec: np.ndarray, pixel_vec: np.ndarray) -> Tuple[float, float]:
    """
    Partial Spearman correlation:
      r(brain,theory | pixel)
    Returns (raw_r, partial_r).
    """
    r_bt, _ = spearmanr(brain_vec, theory_vec)
    r_bp, _ = spearmanr(brain_vec, pixel_vec)
    r_tp, _ = spearmanr(theory_vec, pixel_vec)
    denom = np.sqrt((1 - r_bp**2) * (1 - r_tp**2))
    if denom == 0 or np.isnan(denom):
        partial = np.nan
    else:
        partial = (r_bt - r_bp * r_tp) / denom
    return float(r_bt), float(partial)


def noise_ceiling_loocv_lower(subject_vectors: Mapping[str, np.ndarray]) -> float:
    """
    Lower-bound noise ceiling using leave-one-subject-out mean RDM vector.

    For each subject s:
      r_s = Spearman(vec_s, mean_{k!=s}(vec_k))
    Return mean(r_s).
    """
    items = list(subject_vectors.items())
    if len(items) < 2:
        return float("nan")

    corrs: list[float] = []
    for subj, vec in items:
        others = [v for k, v in items if k != subj]
        mean_vec = np.mean(np.vstack(others), axis=0)
        r = spearman_r(np.asarray(vec, dtype=float), np.asarray(mean_vec, dtype=float))
        if not np.isnan(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else float("nan")


def lower_tri_vector(mat: np.ndarray, *, k: int = -1) -> np.ndarray:
    """Vectorize the lower triangle (k=-1 excludes diagonal by default)."""
    idx = np.tril_indices_from(mat, k=k)
    return mat[idx]


