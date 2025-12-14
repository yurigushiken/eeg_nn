"""
Paper figure: single "hero" temporal decoding plot.

Creates a clean summary curve showing *overall* numerosity decodability over time:
- per subject: average accuracy across all pairs at each time window
- group: mean ± SEM across subjects
- one-sample test vs chance (50%), BH-FDR across windows
- cluster-based onset (≥2 consecutive significant windows)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.analyze_temporal_stats import apply_fdr_correction, find_significant_clusters
from scripts.rsa.temporal_paper_utils import mask_to_spans, sem, timepoint_edges, ttest_1samp_greater
from scripts.rsa.naming import prefixed_path, prefixed_title


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def run_temporal_decoding_hero(
    *,
    subject_data_csv: Path,
    output_dir: Path,
    baseline: float = 50.0,
    fdr_alpha: float = 0.05,
    min_cluster_size: int = 2,
) -> pd.DataFrame:
    df = pd.read_csv(subject_data_csv)
    _ensure_cols(df, ["ClassA", "ClassB", "Subject", "TimeWindow_Center", "Accuracy"])

    # Define expected pair count from the dataset itself.
    pair_keys = df[["ClassA", "ClassB"]].drop_duplicates()
    expected_pairs = int(len(pair_keys))

    # Compute per-subject mean across all pairs, per window.
    df = df.copy()
    df["PairKey"] = df["ClassA"].astype(str) + "_" + df["ClassB"].astype(str)
    subj_time = (
        df.groupby(["Subject", "TimeWindow_Center"], as_index=False)
        .agg(
            Mean_Accuracy=("Accuracy", "mean"),
            N_Pairs=("PairKey", "nunique"),
        )
    )
    subj_time = subj_time[subj_time["N_Pairs"] == expected_pairs].copy()

    times = np.sort(subj_time["TimeWindow_Center"].unique().astype(float))
    rows: list[dict] = []
    for t in times:
        vals = subj_time[subj_time["TimeWindow_Center"] == t]["Mean_Accuracy"].astype(float).to_numpy()
        t_stat, p_one, n = ttest_1samp_greater(vals, popmean=baseline)
        rows.append(
            {
                "TimeWindow_Center": float(t),
                "N_Subjects": int(n),
                "Mean_Accuracy": float(np.nanmean(vals)) if vals.size else float("nan"),
                "SEM_Accuracy": sem(vals),
                "t_stat": float(t_stat),
                "p_value": float(p_one),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("TimeWindow_Center")
    p_vals = out_df["p_value"].to_numpy(dtype=float)
    p_fdr, rejected = apply_fdr_correction(p_values=p_vals, alpha=fdr_alpha, method="bh")
    out_df["P_FDR"] = p_fdr
    out_df["Significant_FDR"] = rejected

    # Identify first significant (even if isolated) and first cluster-onset.
    first_sig_idx: Optional[int] = None
    sig_mask = out_df["Significant_FDR"].to_numpy(dtype=bool)
    for i, is_sig in enumerate(sig_mask):
        if is_sig:
            first_sig_idx = i
            break

    clusters = find_significant_clusters(p_values=out_df["P_FDR"].to_numpy(dtype=float), alpha=fdr_alpha, min_cluster_size=min_cluster_size)
    cluster_onset_idx: Optional[int] = clusters[0][0] if clusters else None

    peak_idx: Optional[int] = None
    if len(out_df):
        peak_idx = int(out_df["Mean_Accuracy"].to_numpy(dtype=float).argmax())

    out_df["IsFirstSignificantWindow"] = False
    out_df["IsClusterOnset"] = False
    out_df["IsPeak"] = False
    if first_sig_idx is not None:
        out_df.loc[out_df.index[first_sig_idx], "IsFirstSignificantWindow"] = True
    if cluster_onset_idx is not None:
        out_df.loc[out_df.index[cluster_onset_idx], "IsClusterOnset"] = True
    if peak_idx is not None:
        out_df.loc[out_df.index[peak_idx], "IsPeak"] = True

    run_root = output_dir
    out_csv = prefixed_path(run_root=run_root, kind="tables", stem="temporal_decoding_hero", ext=".csv")
    out_df.to_csv(out_csv, index=False)

    # Plot
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    x = out_df["TimeWindow_Center"].to_numpy(dtype=float)
    y = out_df["Mean_Accuracy"].to_numpy(dtype=float)
    y_sem = out_df["SEM_Accuracy"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 5.5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.0, 0.7], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_sig = fig.add_subplot(gs[1], sharex=ax)

    color = "#2E86AB"
    ax.plot(x, y, color=color, linewidth=2.5, label="Mean ± SEM")
    if np.isfinite(y_sem).any():
        ax.fill_between(x, y - y_sem, y + y_sem, color=color, alpha=0.18, linewidth=0)

    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2, label="Chance (50%)")

    # Onset + peak annotations
    if cluster_onset_idx is not None:
        onset_t = float(out_df.iloc[cluster_onset_idx]["TimeWindow_Center"])
        ax.axvline(onset_t, color="black", linestyle=":", linewidth=1.2)
        ax.text(onset_t + 5, float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else baseline + 5, "Onset", fontsize=9, va="bottom")

    if peak_idx is not None:
        peak_t = float(out_df.iloc[peak_idx]["TimeWindow_Center"])
        peak_y = float(out_df.iloc[peak_idx]["Mean_Accuracy"])
        ax.axvline(peak_t, color="gray", linestyle=":", linewidth=1.2)
        ax.text(peak_t + 5, peak_y, "Peak", fontsize=9, va="bottom", color="gray")

    # Formatting
    ax.set_title(prefixed_title(run_root=run_root, title="Temporal Decoding of Numerosity"), fontsize=14, fontweight="bold")
    ax.set_ylabel("Decoding accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(0, 500)
    y_min = float(np.nanmin(np.concatenate([y - y_sem, np.array([baseline - 5])])))
    y_max = float(np.nanmax(np.concatenate([y + y_sem, np.array([baseline + 5])])))
    ax.set_ylim(y_min - 1.0, y_max + 1.0)

    # Significance band (FDR-corrected)
    spans = mask_to_spans(x, sig_mask)
    for start, end in spans:
        ax_sig.axvspan(start, end, color="#E74C3C", alpha=0.85)
    ax_sig.set_yticks([])
    ax_sig.set_ylim(0, 1)
    ax_sig.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
    ax_sig.set_title("")  # no title
    ax_sig.grid(False)
    for spine in ["top", "right", "left"]:
        ax_sig.spines[spine].set_visible(False)

    sig_patch = mpatches.Patch(color="#E74C3C", label="Significant (q<0.05, FDR)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(sig_patch)
    labels.append("Significant (q<0.05, FDR)")
    ax.legend(handles, labels, loc="upper right", frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    fig.savefig(prefixed_path(run_root=run_root, kind="figures", stem="temporal_decoding_hero", ext=".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper figure: hero temporal decoding plot (averaged across pairs).")
    parser.add_argument("--subject-data", type=Path, required=True, help="Path to subject_temporal_means.csv (seeds averaged).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory to write figures/ and tables/ into.")
    parser.add_argument("--baseline", type=float, default=50.0, help="Chance level in percent (default: 50).")
    parser.add_argument("--fdr-alpha", type=float, default=0.05, help="FDR alpha across timepoints (default: 0.05).")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Cluster size for onset annotation (default: 2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_temporal_decoding_hero(
        subject_data_csv=args.subject_data,
        output_dir=args.output_dir,
        baseline=float(args.baseline),
        fdr_alpha=float(args.fdr_alpha),
        min_cluster_size=int(args.min_cluster_size),
    )


if __name__ == "__main__":
    main()


