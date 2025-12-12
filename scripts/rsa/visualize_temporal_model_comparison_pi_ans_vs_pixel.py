"""
Paper figure: model comparison over time (PI-ANS vs Pixel).

Computes, per subject and time window:
  Δr = r(Brain, PI-ANS) - r(Brain, Pixel)

For inference we test Δz > 0 using Fisher-z transformed correlations (one-sided),
then apply BH-FDR across timepoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.analyze_temporal_stats import apply_fdr_correction, find_significant_clusters
from scripts.rsa.rdm_models import build_pi_ans_rdm, build_pixel_rdm_e_only, spearman_r
from scripts.rsa.temporal_paper_utils import fisher_z, mask_to_spans, sem, ttest_1samp_greater


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def _pairs_from_codes(codes: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i, a in enumerate(codes):
        for b in codes[i + 1 :]:
            out.append((int(a), int(b)))
    return out


def _model_vec_from_rdm(rdm: np.ndarray, codes: Sequence[int], pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    idx = {int(c): i for i, c in enumerate(codes)}
    return np.asarray([float(rdm[idx[a], idx[b]]) for (a, b) in pairs], dtype=float)


def run_model_comparison_pi_ans_vs_pixel(
    *,
    subject_data_csv: Path,
    output_dir: Path,
    stimuli_csv: Path,
    codes: Optional[Sequence[int]] = None,
    pi_ans_boundary: float = 4.0,
    fdr_alpha: float = 0.05,
    min_cluster_size: int = 2,
) -> pd.DataFrame:
    df = pd.read_csv(subject_data_csv)
    _ensure_cols(df, ["ClassA", "ClassB", "Subject", "TimeWindow_Center", "Accuracy"])

    if codes is None:
        codes = sorted(set(df["ClassA"]).union(set(df["ClassB"])))
    codes = [int(c) for c in codes]
    pairs = _pairs_from_codes(codes)

    # Model vectors (pair order)
    pi_rdm = build_pi_ans_rdm(codes, boundary=float(pi_ans_boundary))
    px_rdm = build_pixel_rdm_e_only(stimuli_csv, codes)
    pi_vec = _model_vec_from_rdm(pi_rdm, codes, pairs)
    px_vec = _model_vec_from_rdm(px_rdm, codes, pairs)

    # Subject-level correlations per window
    subj_rows: list[dict] = []
    for t_center in sorted(df["TimeWindow_Center"].unique()):
        time_df = df[df["TimeWindow_Center"] == t_center]
        for subj, group in time_df.groupby("Subject"):
            pair_means = group.groupby(["ClassA", "ClassB"], as_index=False)["Accuracy"].mean()
            acc_map = {(int(r["ClassA"]), int(r["ClassB"])): float(r["Accuracy"]) for _, r in pair_means.iterrows()}
            normalized = {(a, b) if a < b else (b, a): v for (a, b), v in acc_map.items()}
            if any(p not in normalized for p in pairs):
                continue
            brain_vec = np.asarray([normalized[p] for p in pairs], dtype=float)
            r_pi = spearman_r(brain_vec, pi_vec)
            r_px = spearman_r(brain_vec, px_vec)
            subj_rows.append(
                {
                    "TimeWindow_Center": float(t_center),
                    "Subject": str(int(subj)) if str(subj).isdigit() else str(subj),
                    "PI_ANS_r": float(r_pi),
                    "Pixel_r": float(r_px),
                }
            )

    subj_df = pd.DataFrame(subj_rows)
    if subj_df.empty:
        raise ValueError("No complete subject vectors found (missing pairs).")

    # Timepoint-wise stats on delta
    out_rows: list[dict] = []
    for t_center in sorted(subj_df["TimeWindow_Center"].unique()):
        g = subj_df[subj_df["TimeWindow_Center"] == t_center]
        r_pi = g["PI_ANS_r"].to_numpy(dtype=float)
        r_px = g["Pixel_r"].to_numpy(dtype=float)
        delta_r = r_pi - r_px
        delta_z = fisher_z(r_pi) - fisher_z(r_px)
        t_stat, p_one, n = ttest_1samp_greater(delta_z, popmean=0.0)
        out_rows.append(
            {
                "TimeWindow_Center": float(t_center),
                "N_Subjects": int(n),
                "Delta_Mean": float(np.nanmean(delta_r)) if delta_r.size else float("nan"),
                "Delta_SEM": sem(delta_r),
                "t_stat": float(t_stat),
                "p_value": float(p_one),
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values("TimeWindow_Center")
    p_fdr, rejected = apply_fdr_correction(out_df["p_value"].to_numpy(dtype=float), alpha=fdr_alpha, method="bh")
    out_df["P_FDR"] = p_fdr
    out_df["Significant_FDR"] = rejected

    clusters = find_significant_clusters(p_values=out_df["P_FDR"].to_numpy(dtype=float), alpha=fdr_alpha, min_cluster_size=min_cluster_size)
    out_df["IsClusterSig"] = False
    for cluster in clusters:
        for idx in cluster:
            out_df.loc[out_df.index[idx], "IsClusterSig"] = True

    # Write table + figure
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "temporal_model_comparison_pi_ans_vs_pixel.csv"
    out_df.to_csv(out_csv, index=False)

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    x = out_df["TimeWindow_Center"].to_numpy(dtype=float)
    y = out_df["Delta_Mean"].to_numpy(dtype=float)
    y_sem = out_df["Delta_SEM"].to_numpy(dtype=float)
    sig_mask = out_df["Significant_FDR"].to_numpy(dtype=bool)

    fig = plt.figure(figsize=(10, 5.5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.0, 0.7], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_sig = fig.add_subplot(gs[1], sharex=ax)

    color = "#2E86AB"
    ax.plot(x, y, color=color, linewidth=2.5, label="Δ Spearman r (mean ± SEM)")
    if np.isfinite(y_sem).any():
        ax.fill_between(x, y - y_sem, y + y_sem, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="No difference (0)")

    ax.set_title("Model Comparison: PI-ANS vs Pixel", fontsize=14, fontweight="bold")
    ax.set_ylabel("Δ Spearman r (PI-ANS − Pixel)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(0, 500)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.3)

    spans = mask_to_spans(x, sig_mask)
    for start, end in spans:
        ax_sig.axvspan(start, end, color="#27AE60", alpha=0.9)
    ax_sig.set_yticks([])
    ax_sig.set_ylim(0, 1)
    ax_sig.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
    ax_sig.grid(False)
    for spine in ["top", "right", "left"]:
        ax_sig.spines[spine].set_visible(False)

    sig_patch = mpatches.Patch(color="#27AE60", label="Δ>0 (q<0.05, FDR)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(sig_patch)
    labels.append("Δ>0 (q<0.05, FDR)")
    ax.legend(handles, labels, loc="upper right", frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    out_png = figures_dir / "temporal_model_comparison_pi_ans_vs_pixel.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper figure: PI-ANS vs Pixel model comparison over time.")
    parser.add_argument("--subject-data", type=Path, required=True, help="Path to subject_temporal_means.csv (seeds averaged).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory to write figures/ and tables/ into.")
    parser.add_argument("--stimuli-csv", type=Path, default=Path("data") / "stimuli" / "stimuli_analysis.csv", help="Stimuli analysis CSV for Pixel model.")
    parser.add_argument("--codes", nargs="*", type=int, default=None, help="Optional code list (default: infer from CSV).")
    parser.add_argument("--pi-ans-boundary", type=float, default=4.0, help="PI-ANS boundary distance (default: 4.0).")
    parser.add_argument("--fdr-alpha", type=float, default=0.05, help="FDR alpha across timepoints (default: 0.05).")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Cluster size for marking clusters (default: 2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_model_comparison_pi_ans_vs_pixel(
        subject_data_csv=args.subject_data,
        output_dir=args.output_dir,
        stimuli_csv=args.stimuli_csv,
        codes=args.codes,
        pi_ans_boundary=float(args.pi_ans_boundary),
        fdr_alpha=float(args.fdr_alpha),
        min_cluster_size=int(args.min_cluster_size),
    )


if __name__ == "__main__":
    main()


