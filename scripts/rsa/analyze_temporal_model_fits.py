"""
Temporal RSA: model fit over time (Figure-12C style).

Given `subject_temporal_means.csv` (seeds averaged), compute per-time-window
Spearman correlations between each subject's brain RDM vector (pairwise decoding
accuracies) and candidate model RDM vectors, then plot mean ± SEM over time.

Models:
- PI(1-4)-ANS (project-specific): PI set 1-4 with abs distance; ANS set 5-6 uses log-ratio;
  cross PI↔ANS separated by a large boundary.
- PI(1-3)-ANS (variant): PI set 1-3 with abs distance; ANS set 4-6 uses log-ratio; (4 treated as ANS).
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
    build_rt_landing_rdm,
    noise_ceiling_loocv_lower,
    partial_spearman_r,
    spearman_r,
)
from scripts.rsa.naming import prefixed_path, prefixed_title


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
    pi_ans_boundary: float = 4.0,
    rt_summary_csv: Optional[Path] = None,
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
    pi_14_rdm = build_pi_ans_rdm(codes, pi_min=1, pi_max=4, boundary=pi_ans_boundary)
    pi_13_rdm = build_pi_ans_rdm(codes, pi_min=1, pi_max=3, boundary=pi_ans_boundary)
    pi_24_rdm = build_pi_ans_rdm(codes, pi_min=2, pi_max=4, boundary=pi_ans_boundary)
    pixel_rdm = build_pixel_rdm_e_only(stimuli_csv, codes)
    ans_rdm = build_ans_log_ratio_rdm(codes)
    rt_rdm = build_rt_landing_rdm(rt_summary_csv, codes) if rt_summary_csv else None

    pi_14_vec = _model_vec_from_rdm(pi_14_rdm, codes, pairs)
    pi_13_vec = _model_vec_from_rdm(pi_13_rdm, codes, pairs)
    pi_24_vec = _model_vec_from_rdm(pi_24_rdm, codes, pairs)
    pixel_vec = _model_vec_from_rdm(pixel_rdm, codes, pairs)
    ans_vec = _model_vec_from_rdm(ans_rdm, codes, pairs)
    rt_vec = _model_vec_from_rdm(rt_rdm, codes, pairs) if rt_rdm is not None else None

    # output_dir is the run root (e.g., results/runs/rsa_temporal_v1)
    run_root = output_dir

    spearman_rows: List[Dict[str, float]] = []
    partial_rows: List[Dict[str, float]] = []

    for t_center in sorted(df["TimeWindow_Center"].unique()):
        time_df = df[df["TimeWindow_Center"] == t_center]

        # Build subject→brain vector mapping for this window
        subj_vectors: Dict[str, np.ndarray] = {}
        pi14_rs: List[float] = []
        pi13_rs: List[float] = []
        pi24_rs: List[float] = []
        pixel_rs: List[float] = []
        ans_rs: List[float] = []
        rt_rs: List[float] = []
        pi24_partial_rs: List[float] = []

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

            pi14_rs.append(spearman_r(brain_vec, pi_14_vec))
            pi13_rs.append(spearman_r(brain_vec, pi_13_vec))
            pi24_rs.append(spearman_r(brain_vec, pi_24_vec))
            pixel_rs.append(spearman_r(brain_vec, pixel_vec))
            ans_rs.append(spearman_r(brain_vec, ans_vec))
            if rt_vec is not None:
                rt_rs.append(spearman_r(brain_vec, rt_vec))

            _, pi24_partial = partial_spearman_r(brain_vec, pi_24_vec, pixel_vec)
            pi24_partial_rs.append(pi24_partial)

        if not subj_vectors:
            continue

        nc = noise_ceiling_loocv_lower(subj_vectors)

        spearman_rows.append(
            {
                "TimeWindow_Center": float(t_center),
                "N_Subjects": float(len(subj_vectors)),
                "PI_14_ANS_Mean": float(np.nanmean(pi14_rs)) if pi14_rs else float("nan"),
                "PI_14_ANS_SEM": _sem(pi14_rs),
                "PI_13_ANS_Mean": float(np.nanmean(pi13_rs)) if pi13_rs else float("nan"),
                "PI_13_ANS_SEM": _sem(pi13_rs),
                "PI_24_ANS_Mean": float(np.nanmean(pi24_rs)) if pi24_rs else float("nan"),
                "PI_24_ANS_SEM": _sem(pi24_rs),
                "Pixel_Mean": float(np.nanmean(pixel_rs)) if pixel_rs else float("nan"),
                "Pixel_SEM": _sem(pixel_rs),
                "ANS_Mean": float(np.nanmean(ans_rs)) if ans_rs else float("nan"),
                "ANS_SEM": _sem(ans_rs),
                "RT_Landing_Mean": float(np.nanmean(rt_rs)) if rt_rs else float("nan"),
                "RT_Landing_SEM": _sem(rt_rs) if rt_rs else float("nan"),
                "NoiseCeiling_Mean": float(nc),
            }
        )

        partial_rows.append(
            {
                "TimeWindow_Center": float(t_center),
                "N_Subjects": float(len(subj_vectors)),
                "PI_24_ANS_given_PIXEL_Mean": float(np.nanmean(pi24_partial_rs)) if pi24_partial_rs else float("nan"),
                "PI_24_ANS_given_PIXEL_SEM": _sem(pi24_partial_rs),
            }
        )

    spearman_df = pd.DataFrame(spearman_rows).sort_values("TimeWindow_Center")
    partial_df = pd.DataFrame(partial_rows).sort_values("TimeWindow_Center")

    spearman_csv = prefixed_path(run_root=run_root, kind="tables", stem="temporal_model_fits_spearman", ext=".csv")
    partial_csv = prefixed_path(run_root=run_root, kind="tables", stem="temporal_model_fits_partial", ext=".csv")
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

    plot_line("PI_14_ANS_Mean", "PI_14_ANS_SEM", "PI(1-4)-ANS (raw)")
    plot_line("PI_13_ANS_Mean", "PI_13_ANS_SEM", "PI(1-3)-ANS (raw)")
    plot_line("PI_24_ANS_Mean", "PI_24_ANS_SEM", "PI(2-4)-ANS (raw)")
    plot_line("Pixel_Mean", "Pixel_SEM", "Pixel (raw)")
    plot_line("ANS_Mean", "ANS_SEM", "ANS log-ratio (raw)")
    if "RT_Landing_Mean" in spearman_df.columns and spearman_df["RT_Landing_Mean"].notna().any():
        plot_line("RT_Landing_Mean", "RT_Landing_SEM", "RT landing (raw)")

    ax.plot(x, spearman_df["NoiseCeiling_Mean"].to_numpy(), color="gray", linestyle=":", linewidth=2, label="Noise ceiling (LOOCV)")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Spearman r (model-brain RDM)")
    ax.set_title(prefixed_title(run_root=run_root, title="Temporal RSA: Model Fit Over Time"))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout()
    fig.savefig(prefixed_path(run_root=run_root, kind="figures", stem="temporal_model_fits_spearman", ext=".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: PI(2-4)-ANS raw vs partial | Pixel
    x2 = partial_df["TimeWindow_Center"].to_numpy()
    y2_partial = partial_df["PI_24_ANS_given_PIXEL_Mean"].to_numpy()
    sem2_partial = partial_df["PI_24_ANS_given_PIXEL_SEM"].to_numpy()
    # Raw PI(2-4)-ANS (same time grid)
    y2_raw = spearman_df["PI_24_ANS_Mean"].to_numpy()
    sem2_raw = spearman_df["PI_24_ANS_SEM"].to_numpy()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(x2, y2_raw, linewidth=2, label="PI(2-4)-ANS (raw)")
    if np.isfinite(sem2_raw).any():
        ax2.fill_between(x2, y2_raw - sem2_raw, y2_raw + sem2_raw, alpha=0.12)

    ax2.plot(x2, y2_partial, linewidth=2, label="PI(2-4)-ANS | Pixel (partial)")
    if np.isfinite(sem2_partial).any():
        ax2.fill_between(x2, y2_partial - sem2_partial, y2_partial + sem2_partial, alpha=0.12)
    ax2.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Correlation (raw / partial)")
    ax2.set_title(prefixed_title(run_root=run_root, title="Temporal RSA: PI(2-4)-ANS Fit Controlling Pixel Model"))
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
    fig2.tight_layout()
    fig2.savefig(
        prefixed_path(run_root=run_root, kind="figures", stem="temporal_model_fits_partial_pi_24_ans_given_pixel", ext=".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig2)

    return spearman_df, partial_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal RSA model-fit analysis (model-brain RDM correlation over time).")
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
        default=4.0,
        help="Cross PI↔ANS boundary distance for PI-ANS model (default: 4.0).",
    )
    parser.add_argument(
        "--rt-summary-csv",
        type=Path,
        default=None,
        help="Optional RT subject-level summary CSV for building an RT landing model RDM.",
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
        rt_summary_csv=args.rt_summary_csv,
    )


if __name__ == "__main__":
    main()


