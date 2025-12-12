"""
Temporal RSA: model fit over time (Figure-12C style).

Given `subject_temporal_means.csv` (seeds averaged), compute per-time-window
Spearman correlations between each subject's brain RDM vector (pairwise decoding
accuracies) and candidate model RDM vectors, then plot mean ± SEM over time.

Models:
- PI-ANS (project-specific): PI set 1-4 with abs distance; ANS set 5-6 uses log-ratio;
  cross PI↔ANS separated by a large boundary.
- Pixel (e-only target): abs differences in white_pixel_area for 1e..6e.
- ANS log-ratio: |log(i) - log(j)|.

Also computes a lower-bound noise ceiling per time window using LOOCV
subject-vs-group Spearman correlation.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is importable when running as a script.
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.rdm_models import (
    build_ans_log_ratio_rdm,
    build_pi_ans_rdm,
    build_pixel_rdm_e_only,
    noise_ceiling_loocv_lower,
    partial_spearman_r,
    spearman_r,
)


DEFAULT_CODES = [11, 22, 33, 44, 55, 66]


def _pairs_from_codes(codes: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i, a in enumerate(codes):
        for b in codes[i + 1 :]:
            out.append((int(a), int(b)))
    return out


def _model_vec_from_rdm(rdm: np.ndarray, codes: Sequence[int], pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    idx = {int(c): i for i, c in enumerate(codes)}
    return np.asarray([float(rdm[idx[a], idx[b]]) for (a, b) in pairs], dtype=float)


def _sem(values: Sequence[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    if vals.size < 2:
        return float("nan")
    return float(np.nanstd(vals, ddof=1) / math.sqrt(vals.size))


def _ensure_required_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def run_temporal_model_fits(
    *,
    subject_data_csv: Path,
    output_dir: Path,
    stimuli_csv: Path = Path("data/stimuli/stimuli_analysis.csv"),
    codes: Optional[Sequence[int]] = None,
    pi_ans_boundary: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute and write temporal model fits (Spearman + partial) and figures.

    Returns:
        (spearman_summary_df, partial_summary_df)
    """
    df = pd.read_csv(subject_data_csv)
    _ensure_required_columns(
        df,
        [
            "ClassA",
            "ClassB",
            "Subject",
            "TimeWindow_Center",
            "Accuracy",
        ],
    )

    if codes is None:
        codes = sorted(set(df["ClassA"]).union(set(df["ClassB"])))
    codes = [int(c) for c in codes]
    pairs = _pairs_from_codes(codes)

    # Build model vectors once (in the same pair order).
    pi_ans_rdm = build_pi_ans_rdm(codes, boundary=pi_ans_boundary)
    pixel_rdm = build_pixel_rdm_e_only(stimuli_csv, codes)
    ans_rdm = build_ans_log_ratio_rdm(codes)

    pi_ans_vec = _model_vec_from_rdm(pi_ans_rdm, codes, pairs)
    pixel_vec = _model_vec_from_rdm(pixel_rdm, codes, pairs)
    ans_vec = _model_vec_from_rdm(ans_rdm, codes, pairs)

    # Ensure output structure
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    spearman_rows: List[Dict[str, float]] = []
    partial_rows: List[Dict[str, float]] = []

    for t_center in sorted(df["TimeWindow_Center"].unique()):
        time_df = df[df["TimeWindow_Center"] == t_center]

        # Build subject→brain vector mapping for this window
        subj_vectors: Dict[str, np.ndarray] = {}
        pi_rs: List[float] = []
        pixel_rs: List[float] = []
        ans_rs: List[float] = []
        pi_partial_rs: List[float] = []

        for subj, group in time_df.groupby("Subject"):
            # Reduce any duplicates defensively
            pair_means = (
                group.groupby(["ClassA", "ClassB"], as_index=False)["Accuracy"]
                .mean()
            )
            acc_map = {(int(r["ClassA"]), int(r["ClassB"])): float(r["Accuracy"]) for _, r in pair_means.iterrows()}

            # Normalize pair key order (ClassA<ClassB in this dataset, but be defensive)
            normalized: Dict[Tuple[int, int], float] = {}
            for (a, b), v in acc_map.items():
                key = (a, b) if a < b else (b, a)
                normalized[key] = v

            if any(p not in normalized for p in pairs):
                # Skip subjects missing any pair for this time window.
                continue

            brain_vec = np.asarray([normalized[p] for p in pairs], dtype=float)
            subj_vectors[str(int(subj))] = brain_vec

            pi_rs.append(spearman_r(brain_vec, pi_ans_vec))
            pixel_rs.append(spearman_r(brain_vec, pixel_vec))
            ans_rs.append(spearman_r(brain_vec, ans_vec))

            _, pi_partial = partial_spearman_r(brain_vec, pi_ans_vec, pixel_vec)
            pi_partial_rs.append(pi_partial)

        if not subj_vectors:
            continue

        nc = noise_ceiling_loocv_lower(subj_vectors)

        spearman_rows.append(
            {
                "TimeWindow_Center": float(t_center),
                "N_Subjects": float(len(subj_vectors)),
                "PI_ANS_Mean": float(np.nanmean(pi_rs)) if pi_rs else float("nan"),
                "PI_ANS_SEM": _sem(pi_rs),
                "Pixel_Mean": float(np.nanmean(pixel_rs)) if pixel_rs else float("nan"),
                "Pixel_SEM": _sem(pixel_rs),
                "ANS_Mean": float(np.nanmean(ans_rs)) if ans_rs else float("nan"),
                "ANS_SEM": _sem(ans_rs),
                "NoiseCeiling_Mean": float(nc),
            }
        )

        partial_rows.append(
            {
                "TimeWindow_Center": float(t_center),
                "N_Subjects": float(len(subj_vectors)),
                "PI_ANS_given_PIXEL_Mean": float(np.nanmean(pi_partial_rs)) if pi_partial_rs else float("nan"),
                "PI_ANS_given_PIXEL_SEM": _sem(pi_partial_rs),
            }
        )

    spearman_df = pd.DataFrame(spearman_rows).sort_values("TimeWindow_Center")
    partial_df = pd.DataFrame(partial_rows).sort_values("TimeWindow_Center")

    spearman_csv = tables_dir / "temporal_model_fits_spearman.csv"
    partial_csv = tables_dir / "temporal_model_fits_partial.csv"
    spearman_df.to_csv(spearman_csv, index=False)
    partial_df.to_csv(partial_csv, index=False)

    # Plot 1: Spearman fits (with noise ceiling)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = spearman_df["TimeWindow_Center"].to_numpy()

    def plot_line(y_col: str, sem_col: str, label: str):
        y = spearman_df[y_col].to_numpy()
        sem = spearman_df[sem_col].to_numpy()
        ax.plot(x, y, linewidth=2, label=label)
        if np.isfinite(sem).any():
            ax.fill_between(x, y - sem, y + sem, alpha=0.15)

    plot_line("PI_ANS_Mean", "PI_ANS_SEM", "PI-ANS (raw)")
    plot_line("Pixel_Mean", "Pixel_SEM", "Pixel (raw)")
    plot_line("ANS_Mean", "ANS_SEM", "ANS log-ratio (raw)")

    ax.plot(x, spearman_df["NoiseCeiling_Mean"].to_numpy(), color="gray", linestyle=":", linewidth=2, label="Noise ceiling (LOOCV)")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Spearman r (model–brain RDM)")
    ax.set_title("Temporal RSA: Model Fit Over Time")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout()
    fig.savefig(figures_dir / "temporal_model_fits_spearman.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Partial (PI-ANS | Pixel)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x2 = partial_df["TimeWindow_Center"].to_numpy()
    y2 = partial_df["PI_ANS_given_PIXEL_Mean"].to_numpy()
    sem2 = partial_df["PI_ANS_given_PIXEL_SEM"].to_numpy()
    ax2.plot(x2, y2, linewidth=2, label="PI-ANS | Pixel")
    if np.isfinite(sem2).any():
        ax2.fill_between(x2, y2 - sem2, y2 + sem2, alpha=0.15)
    ax2.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Partial Spearman r")
    ax2.set_title("Temporal RSA: PI-ANS Fit Controlling Pixel Model")
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
    fig2.tight_layout()
    fig2.savefig(figures_dir / "temporal_model_fits_partial_pi_ans_given_pixel.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    return spearman_df, partial_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal RSA model-fit analysis (model–brain RDM correlation over time).")
    parser.add_argument(
        "--subject-data",
        type=Path,
        required=True,
        help="Path to subject_temporal_means.csv (seeds averaged).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory to write tables/figures into (e.g., results/runs/rsa_temporal_v1).",
    )
    parser.add_argument(
        "--stimuli-csv",
        type=Path,
        default=Path("data") / "stimuli" / "stimuli_analysis.csv",
        help="Stimuli analysis CSV used for pixel model (default: data/stimuli/stimuli_analysis.csv).",
    )
    parser.add_argument(
        "--codes",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of condition codes (default: infer from CSV).",
    )
    parser.add_argument(
        "--pi-ans-boundary",
        type=float,
        default=10.0,
        help="Cross PI↔ANS boundary distance for PI-ANS model (default: 10.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_temporal_model_fits(
        subject_data_csv=args.subject_data,
        output_dir=args.output_dir,
        stimuli_csv=args.stimuli_csv,
        codes=args.codes,
        pi_ans_boundary=float(args.pi_ans_boundary),
    )


if __name__ == "__main__":
    main()


