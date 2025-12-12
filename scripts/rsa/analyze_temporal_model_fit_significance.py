"""
Paper analyses: inferential tests for temporal model fits.

Outputs:
- figures/temporal_model_fits_with_significance.png
- figures/temporal_model_fits_partial_pi_ans_given_pixel_with_significance.png
- tables/temporal_model_significance_tests.csv

Statistical approach:
- For each model × time window, compute subject-level Spearman correlations.
- Test mean correlation > 0 using one-sample t-test on Fisher-z values (one-sided).
- BH-FDR correction across all (timepoints × models) for raw fits.
- Partial PI-ANS|Pixel is tested separately across timepoints (BH-FDR).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.analyze_temporal_stats import apply_fdr_correction
from scripts.rsa.rdm_models import (
    build_ans_log_ratio_rdm,
    build_pi_ans_rdm,
    build_pixel_rdm_e_only,
    build_rt_landing_rdm,
    noise_ceiling_loocv_lower,
    partial_spearman_r,
    spearman_r,
)
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


def _compute_subject_vectors_for_time(
    df_time: pd.DataFrame,
    pairs: Sequence[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    subj_vectors: Dict[str, np.ndarray] = {}
    for subj, group in df_time.groupby("Subject"):
        pair_means = group.groupby(["ClassA", "ClassB"], as_index=False)["Accuracy"].mean()
        acc_map = {(int(r["ClassA"]), int(r["ClassB"])): float(r["Accuracy"]) for _, r in pair_means.iterrows()}
        normalized = {(a, b) if a < b else (b, a): v for (a, b), v in acc_map.items()}
        if any(p not in normalized for p in pairs):
            continue
        brain_vec = np.asarray([normalized[p] for p in pairs], dtype=float)
        key = str(int(subj)) if str(subj).isdigit() else str(subj)
        subj_vectors[key] = brain_vec
    return subj_vectors


def run_model_fit_significance(
    *,
    subject_data_csv: Path,
    output_dir: Path,
    stimuli_csv: Path,
    rt_summary_csv: Path,
    codes: Optional[Sequence[int]] = None,
    pi_ans_boundary: float = 4.0,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    df = pd.read_csv(subject_data_csv)
    _ensure_cols(df, ["ClassA", "ClassB", "Subject", "TimeWindow_Center", "Accuracy"])

    if codes is None:
        codes = sorted(set(df["ClassA"]).union(set(df["ClassB"])))
    codes = [int(c) for c in codes]
    pairs = _pairs_from_codes(codes)

    # Model vectors
    pi_rdm = build_pi_ans_rdm(codes, boundary=float(pi_ans_boundary))
    px_rdm = build_pixel_rdm_e_only(stimuli_csv, codes)
    ans_rdm = build_ans_log_ratio_rdm(codes)
    rt_rdm = build_rt_landing_rdm(rt_summary_csv, codes)

    pi_vec = _model_vec_from_rdm(pi_rdm, codes, pairs)
    px_vec = _model_vec_from_rdm(px_rdm, codes, pairs)
    ans_vec = _model_vec_from_rdm(ans_rdm, codes, pairs)
    rt_vec = _model_vec_from_rdm(rt_rdm, codes, pairs)

    # Per-subject correlations per window
    corr_rows: list[dict] = []
    noise_ceiling_by_time: Dict[float, float] = {}

    for t_center in sorted(df["TimeWindow_Center"].unique()):
        df_time = df[df["TimeWindow_Center"] == t_center]
        subj_vectors = _compute_subject_vectors_for_time(df_time, pairs)
        if not subj_vectors:
            continue

        noise_ceiling_by_time[float(t_center)] = float(noise_ceiling_loocv_lower(subj_vectors))

        for subj, brain_vec in subj_vectors.items():
            r_pi = spearman_r(brain_vec, pi_vec)
            r_px = spearman_r(brain_vec, px_vec)
            r_ans = spearman_r(brain_vec, ans_vec)
            r_rt = spearman_r(brain_vec, rt_vec)
            _, r_pi_partial = partial_spearman_r(brain_vec, pi_vec, px_vec)
            corr_rows.append(
                {
                    "TimeWindow_Center": float(t_center),
                    "Subject": subj,
                    "PI_ANS": float(r_pi),
                    "Pixel": float(r_px),
                    "ANS": float(r_ans),
                    "RT_Landing": float(r_rt),
                    "PI_ANS_given_PIXEL": float(r_pi_partial),
                }
            )

    corr_df = pd.DataFrame(corr_rows)
    if corr_df.empty:
        raise ValueError("No correlations computed (missing pairs or empty input).")

    # --- Significance tests: raw models ---
    raw_models = [
        ("PI_ANS", "PI-ANS"),
        ("Pixel", "Pixel"),
        ("ANS", "ANS"),
        ("RT_Landing", "RT"),
    ]

    raw_test_rows: list[dict] = []
    for t_center in sorted(corr_df["TimeWindow_Center"].unique()):
        g = corr_df[corr_df["TimeWindow_Center"] == t_center]
        for col, label in raw_models:
            r_vals = g[col].to_numpy(dtype=float)
            z_vals = fisher_z(r_vals)
            t_stat, p_one, n = ttest_1samp_greater(z_vals, popmean=0.0)
            raw_test_rows.append(
                {
                    "Analysis": "raw",
                    "TimeWindow_Center": float(t_center),
                    "TimeWindow": float(t_center),
                    "Model": label if label != "PI-ANS" else "PI_ANS",
                    "Mean_r": float(np.nanmean(r_vals)) if r_vals.size else float("nan"),
                    "SEM": sem(r_vals),
                    "t_stat": float(t_stat),
                    "p_value": float(p_one),
                    "N_Subjects": int(n),
                }
            )

    raw_tests = pd.DataFrame(raw_test_rows).sort_values(["TimeWindow_Center", "Model"])
    p_fdr, rejected = apply_fdr_correction(raw_tests["p_value"].to_numpy(dtype=float), alpha=fdr_alpha, method="bh")
    raw_tests["p_fdr"] = p_fdr
    raw_tests["significant"] = rejected

    # --- Significance tests: partial PI-ANS|Pixel ---
    partial_rows: list[dict] = []
    for t_center in sorted(corr_df["TimeWindow_Center"].unique()):
        g = corr_df[corr_df["TimeWindow_Center"] == t_center]
        r_vals = g["PI_ANS_given_PIXEL"].to_numpy(dtype=float)
        z_vals = fisher_z(r_vals)
        t_stat, p_one, n = ttest_1samp_greater(z_vals, popmean=0.0)
        partial_rows.append(
            {
                "Analysis": "partial",
                "TimeWindow_Center": float(t_center),
                "TimeWindow": float(t_center),
                "Model": "PI_ANS_given_PIXEL",
                "Mean_r": float(np.nanmean(r_vals)) if r_vals.size else float("nan"),
                "SEM": sem(r_vals),
                "t_stat": float(t_stat),
                "p_value": float(p_one),
                "N_Subjects": int(n),
            }
        )
    partial_tests = pd.DataFrame(partial_rows).sort_values("TimeWindow_Center")
    p_fdr_p, rejected_p = apply_fdr_correction(partial_tests["p_value"].to_numpy(dtype=float), alpha=fdr_alpha, method="bh")
    partial_tests["p_fdr"] = p_fdr_p
    partial_tests["significant"] = rejected_p

    all_tests = pd.concat([raw_tests, partial_tests], ignore_index=True)

    # Write CSV
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_csv = tables_dir / "temporal_model_significance_tests.csv"
    all_tests.to_csv(out_csv, index=False)

    # ---------------- Figure: raw fits + sig bands ----------------
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    # Build summary for plotting (raw)
    x = np.sort(raw_tests["TimeWindow_Center"].unique().astype(float))
    # Map model->(mean, sem, sigmask)
    model_plot: Dict[str, Dict[str, np.ndarray]] = {}
    for model_key in ["PI_ANS", "Pixel", "ANS", "RT"]:
        df_m = raw_tests[raw_tests["Model"] == model_key].sort_values("TimeWindow_Center")
        model_plot[model_key] = {
            "x": df_m["TimeWindow_Center"].to_numpy(dtype=float),
            "mean": df_m["Mean_r"].to_numpy(dtype=float),
            "sem": df_m["SEM"].to_numpy(dtype=float),
            "sig": df_m["significant"].to_numpy(dtype=bool),
        }

    # Noise ceiling series
    nc = np.asarray([noise_ceiling_by_time.get(float(t), float("nan")) for t in x], dtype=float)

    colors = {
        "PI_ANS": "#2E86AB",
        "Pixel": "#C0392B",
        "ANS": "#8E44AD",
        "RT": "#27AE60",
    }

    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.0, 1.4], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_sig = fig.add_subplot(gs[1], sharex=ax)

    for model_key, label in [("PI_ANS", "PI-ANS"), ("Pixel", "Pixel"), ("ANS", "ANS (log-ratio)"), ("RT", "RT landing")]:
        dd = model_plot[model_key]
        ax.plot(dd["x"], dd["mean"], linewidth=2.2, label=label, color=colors[model_key])
        if np.isfinite(dd["sem"]).any():
            ax.fill_between(dd["x"], dd["mean"] - dd["sem"], dd["mean"] + dd["sem"], alpha=0.15, color=colors[model_key], linewidth=0)

    ax.plot(x, nc, color="gray", linestyle=":", linewidth=2.0, label="Noise ceiling (LOOCV)")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_title("Model–Brain RDM Correlation Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Spearman r (model–brain RDM)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(0, 500)

    # Significance bands (one row per model)
    model_rows = [("PI_ANS", "PI-ANS"), ("Pixel", "Pixel"), ("ANS", "ANS"), ("RT", "RT")]
    ax_sig.set_yticks(range(len(model_rows)))
    ax_sig.set_yticklabels([m[1] for m in model_rows], fontsize=9)
    ax_sig.set_ylim(-0.6, len(model_rows) - 0.4)
    ax_sig.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
    ax_sig.grid(False)
    for spine in ["top", "right"]:
        ax_sig.spines[spine].set_visible(False)

    for row_idx, (model_key, _) in enumerate(model_rows):
        dd = model_plot[model_key]
        spans = mask_to_spans(dd["x"], dd["sig"])
        for start, end in spans:
            ax_sig.broken_barh([(start, end - start)], (row_idx - 0.35, 0.7), facecolors=colors[model_key])

    sig_patch = mpatches.Patch(color="black", label="q<0.05 (BH-FDR)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(sig_patch)
    labels.append("q<0.05 (BH-FDR)")
    ax.legend(handles, labels, loc="best", frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    out_png = figures_dir / "temporal_model_fits_with_significance.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure: partial PI-ANS|Pixel + sig ----------------
    df_p = partial_tests.sort_values("TimeWindow_Center")
    x2 = df_p["TimeWindow_Center"].to_numpy(dtype=float)
    y2 = df_p["Mean_r"].to_numpy(dtype=float)
    sem2 = df_p["SEM"].to_numpy(dtype=float)
    sig2 = df_p["significant"].to_numpy(dtype=bool)

    fig2 = plt.figure(figsize=(10, 5.5))
    gs2 = fig2.add_gridspec(nrows=2, ncols=1, height_ratios=[5.0, 0.7], hspace=0.05)
    ax2 = fig2.add_subplot(gs2[0])
    ax2_sig = fig2.add_subplot(gs2[1], sharex=ax2)

    ax2.plot(x2, y2, color=colors["PI_ANS"], linewidth=2.5, label="PI-ANS | Pixel (mean ± SEM)")
    if np.isfinite(sem2).any():
        ax2.fill_between(x2, y2 - sem2, y2 + sem2, color=colors["PI_ANS"], alpha=0.18, linewidth=0)
    ax2.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax2.set_title("Model–Brain RDM Correlation Over Time (Partial)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Partial Spearman r", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.set_xlim(0, 500)

    spans2 = mask_to_spans(x2, sig2)
    for start, end in spans2:
        ax2_sig.axvspan(start, end, color=colors["PI_ANS"], alpha=0.85)
    ax2_sig.set_yticks([])
    ax2_sig.set_ylim(0, 1)
    ax2_sig.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
    for spine in ["top", "right", "left"]:
        ax2_sig.spines[spine].set_visible(False)

    sig_patch2 = mpatches.Patch(color=colors["PI_ANS"], label="q<0.05 (BH-FDR)")
    ax2.legend([sig_patch2], ["q<0.05 (BH-FDR)"], loc="best", frameon=True, fancybox=False, edgecolor="black")
    fig2.tight_layout()
    out_png2 = figures_dir / "temporal_model_fits_partial_pi_ans_given_pixel_with_significance.png"
    fig2.savefig(out_png2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    return all_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper analysis: temporal model-fit significance tests.")
    parser.add_argument("--subject-data", type=Path, required=True, help="Path to subject_temporal_means.csv (seeds averaged).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory to write figures/ and tables/ into.")
    parser.add_argument("--stimuli-csv", type=Path, default=Path("data") / "stimuli" / "stimuli_analysis.csv")
    parser.add_argument("--rt-summary-csv", type=Path, required=True, help="RT subject-level landing summary CSV.")
    parser.add_argument("--codes", nargs="*", type=int, default=None, help="Optional code list (default: infer from CSV).")
    parser.add_argument("--pi-ans-boundary", type=float, default=4.0)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_model_fit_significance(
        subject_data_csv=args.subject_data,
        output_dir=args.output_dir,
        stimuli_csv=args.stimuli_csv,
        rt_summary_csv=args.rt_summary_csv,
        codes=args.codes,
        pi_ans_boundary=float(args.pi_ans_boundary),
        fdr_alpha=float(args.fdr_alpha),
    )


if __name__ == "__main__":
    main()


