"""
RSA pixel control analysis: partial correlation of brain RDM vs theory RDM controlling for pixel RDM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_1samp

# Ensure project root is importable when running as a script.
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.rdm_models import (
    build_ans_log_ratio_rdm,
    build_pi_ans_rdm as build_pi_ans_model_rdm,
    build_pixel_rdm_e_only,
    build_rt_landing_rdm,
    lower_tri_vector,
    partial_spearman_r,
)
from scripts.rsa.naming import prefixed_path, prefixed_title

DEFAULT_CODES = [11, 22, 33, 44, 55, 66]


def build_pi_ans_rdm(
    codes: List[int],
    *,
    pi_min: int = 1,
    pi_max: int = 4,
    boundary: float = 4.0,
) -> np.ndarray:
    """
    PI(k)-ANS model RDM (project-specific).

    - pi_min=1, pi_max=4 -> PI(1-4)-ANS
    - pi_min=1, pi_max=3 -> PI(1-3)-ANS (numerosity 4 is treated as ANS)
    - pi_min=2, pi_max=4 -> PI(2-4)-ANS (numerosity 1 is cross-only)
    """
    return build_pi_ans_model_rdm(codes, pi_min=int(pi_min), pi_max=int(pi_max), boundary=float(boundary))


def build_pixel_rdm(stimuli_csv: Path, codes: List[int]) -> np.ndarray:
    """Pixel model RDM using e-only target stimuli (absolute pixel-area difference)."""
    return build_pixel_rdm_e_only(stimuli_csv, codes)


def _lower_tri(mat: np.ndarray) -> np.ndarray:
    return lower_tri_vector(mat, k=-1)


def partial_spearman(brain_vec: np.ndarray, theory_vec: np.ndarray, pixel_vec: np.ndarray) -> Tuple[float, float]:
    # Backwards-compatible wrapper: partial Spearman of brain↔theory controlling pixels.
    return partial_spearman_r(brain_vec, theory_vec, pixel_vec)


def build_brain_rdms(master_csv: Path, codes: List[int]) -> Dict[str, np.ndarray]:
    df = pd.read_csv(master_csv)
    df = df[df["RecordType"].astype(str).str.lower() == "subject"]
    if df.empty:
        raise ValueError("No subject-level rows found in master CSV.")
    df = df.groupby(["Subject", "ClassA", "ClassB"])["Accuracy"].mean().reset_index()
    subjects = df["Subject"].unique()
    idx_map = {c: i for i, c in enumerate(codes)}
    rdms: Dict[str, np.ndarray] = {}
    for subj in subjects:
        mat = np.full((len(codes), len(codes)), np.nan)
        sub_df = df[df["Subject"] == subj]
        for _, row in sub_df.iterrows():
            a, b, acc = int(row["ClassA"]), int(row["ClassB"]), float(row["Accuracy"])
            i, j = idx_map[a], idx_map[b]
            mat[i, j] = mat[j, i] = acc
        # fill diagonal with 100
        np.fill_diagonal(mat, 100.0)
        if np.isnan(mat).any():
            # skip subjects missing pairs
            continue
        rdms[subj] = _lower_tri(mat)
    if not rdms:
        raise ValueError("All subjects missing pairs for requested codes.")
    return rdms


def infer_codes_from_master_csv(master_csv: Path) -> List[int]:
    """
    Infer condition codes from the master CSV (subject-level rows).
    This enables running confounds on landing-digit runs (ClassA/B in 1..6)
    without needing a separate config file.
    """
    df = pd.read_csv(master_csv)
    if "RecordType" in df.columns:
        df = df[df["RecordType"].astype(str).str.lower() == "subject"]
    if df.empty:
        # Fallback: infer from all rows
        df = pd.read_csv(master_csv)
    codes = sorted(set(df["ClassA"].astype(int)).union(set(df["ClassB"].astype(int))))
    return [int(c) for c in codes]


def plot_heatmap(
    mat: np.ndarray,
    labels: List[int],
    title: str,
    outfile: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    if vmax is None:
        vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else None
    im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title, pad=12)
    for i in range(len(labels)):
        for j in range(len(labels)):
            # Use white text on dark diagonal, black text elsewhere
            color = "white" if i == j else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Avoid clipping multi-line titles: reserve headroom and use tight bbox.
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_heatmap_diverging(
    mat: np.ndarray,
    labels: List[int],
    title: str,
    outfile: Path,
    vcenter: float = 0.0,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title, pad=12)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if i == j else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def run_analysis(
    master_csv: Path,
    stimuli_csv: Path,
    output_dir: Path,
    codes: List[int],
    baseline: float,
    rt_summary_csv: Path | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Model heatmaps are written once, at the run root (next to confounds/), to avoid clutter.
    # If output_dir is ".../<run_root>/confounds", models_dir becomes ".../<run_root>/models".
    run_root = output_dir.parent
    models_dir = run_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Model RDMs (project models + pixel confound)
    pi_14 = build_pi_ans_rdm(codes, pi_min=1, pi_max=4)
    pi_13 = build_pi_ans_rdm(codes, pi_min=1, pi_max=3)
    pi_24 = build_pi_ans_rdm(codes, pi_min=2, pi_max=4)
    ans = build_ans_log_ratio_rdm(codes)
    pixel = build_pixel_rdm(stimuli_csv, codes)
    rt_rdm = build_rt_landing_rdm(rt_summary_csv, codes) if rt_summary_csv else None

    # Model RDMs on their native scale; vmin=0 for consistent comparison
    plot_heatmap(
        pi_14,
        codes,
        prefixed_title(run_root=run_root, title="PI(1-4)-ANS Model RDM"),
        prefixed_path(run_root=run_root, kind="models", stem="pi_14_ans_rdm_heatmap", ext=".png"),
        vmin=0.0,
        vmax=np.nanmax(pi_14),
    )
    plot_heatmap(
        pi_13,
        codes,
        prefixed_title(run_root=run_root, title="PI(1-3)-ANS Model RDM"),
        prefixed_path(run_root=run_root, kind="models", stem="pi_13_ans_rdm_heatmap", ext=".png"),
        vmin=0.0,
        vmax=np.nanmax(pi_13),
    )
    plot_heatmap(
        pi_24,
        codes,
        prefixed_title(run_root=run_root, title="PI(2-4)-ANS Model RDM"),
        prefixed_path(run_root=run_root, kind="models", stem="pi_24_ans_rdm_heatmap", ext=".png"),
        vmin=0.0,
        vmax=np.nanmax(pi_24),
    )
    plot_heatmap(
        ans,
        codes,
        prefixed_title(run_root=run_root, title="ANS Log-Ratio RDM"),
        prefixed_path(run_root=run_root, kind="models", stem="ans_rdm_heatmap", ext=".png"),
        vmin=0.0,
        vmax=np.nanmax(ans),
    )
    plot_heatmap(
        pixel,
        codes,
        prefixed_title(run_root=run_root, title="Pixel Model RDM\n(e-only target)"),
        prefixed_path(run_root=run_root, kind="models", stem="pixel_rdm_heatmap", ext=".png"),
        vmin=0.0,
        vmax=np.nanmax(pixel),
    )
    if rt_rdm is not None:
        plot_heatmap(
            rt_rdm,
            codes,
            prefixed_title(run_root=run_root, title="RT Landing Model RDM\n(|mean RT_i - mean RT_j|)"),
            prefixed_path(run_root=run_root, kind="models", stem="rt_landing_rdm_heatmap", ext=".png"),
            vmin=0.0,
            vmax=np.nanmax(rt_rdm),
        )

    brain_rdms = build_brain_rdms(master_csv, codes)
    records = []
    residuals = []
    raw_means: List[float] = []
    pi14_vec = _lower_tri(pi_14)
    pi13_vec = _lower_tri(pi_13)
    ans_vec = _lower_tri(ans)
    pixel_vec = _lower_tri(pixel)
    rt_vec = _lower_tri(rt_rdm) if rt_rdm is not None else None
    for subj, brain_vec in brain_rdms.items():
        r_pi14_raw, r_pi14_partial = partial_spearman(brain_vec, pi14_vec, pixel_vec)
        r_pi13_raw, r_pi13_partial = partial_spearman(brain_vec, pi13_vec, pixel_vec)
        r_ans_raw, r_ans_partial = partial_spearman(brain_vec, ans_vec, pixel_vec)
        rec = {
            "Subject": subj,
            "Raw_Spearman_R_PI_14": r_pi14_raw,
            "Partial_Spearman_R_PI_14": r_pi14_partial,
            "Raw_Spearman_R_PI_13": r_pi13_raw,
            "Partial_Spearman_R_PI_13": r_pi13_partial,
            "Raw_Spearman_R_ANS": r_ans_raw,
            "Partial_Spearman_R_ANS": r_ans_partial,
        }
        if rt_vec is not None:
            r_rt_raw, r_rt_partial = partial_spearman(brain_vec, rt_vec, pixel_vec)
            rec["Raw_Spearman_R_RT"] = r_rt_raw
            rec["Partial_Spearman_R_RT"] = r_rt_partial
        records.append(rec)
        # Track raw mean accuracy (before ranking) for offsetting residuals
        raw_means.append(float(np.nanmean(brain_vec)))
        # Residualize brain vs pixel in accuracy space (no ranking)
        bx = brain_vec
        px = pixel_vec
        A = np.vstack([px, np.ones_like(px)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
        bx_pred = A @ coeffs
        bx_resid = bx - bx_pred
        residuals.append(bx_resid)

    subj_df = pd.DataFrame(records)
    raw_vals_rt = subj_df["Raw_Spearman_R_RT"].dropna().to_numpy() if "Raw_Spearman_R_RT" in subj_df.columns else np.array([])
    partial_vals_rt = subj_df["Partial_Spearman_R_RT"].dropna().to_numpy() if "Partial_Spearman_R_RT" in subj_df.columns else np.array([])

    # TEST 1: Compare raw vs partial correlations (paired t-test) for PI(1-4)-ANS
    raw_vals = subj_df["Raw_Spearman_R_PI_14"].dropna().to_numpy()
    partial_vals = subj_df["Partial_Spearman_R_PI_14"].dropna().to_numpy()
    diff_vals = partial_vals - raw_vals

    raw_vals_ans = subj_df["Raw_Spearman_R_ANS"].dropna().to_numpy()
    partial_vals_ans = subj_df["Partial_Spearman_R_ANS"].dropna().to_numpy()
    diff_vals_ans = partial_vals_ans - raw_vals_ans

    if diff_vals.size > 0:
        # Fisher z-transform for both
        z_raw = np.arctanh(raw_vals)
        z_partial = np.arctanh(partial_vals)
        z_diff = z_partial - z_raw

        # Paired t-test on z-transformed differences
        t_diff, p_diff = ttest_1samp(z_diff, 0.0)
        mean_diff = np.mean(diff_vals)
    else:
        t_diff, p_diff, mean_diff = np.nan, np.nan, np.nan

    if diff_vals_ans.size > 0:
        z_raw_ans = np.arctanh(raw_vals_ans)
        z_partial_ans = np.arctanh(partial_vals_ans)
        z_diff_ans = z_partial_ans - z_raw_ans
        t_diff_ans, p_diff_ans = ttest_1samp(z_diff_ans, 0.0)
        mean_diff_ans = np.mean(diff_vals_ans)
    else:
        t_diff_ans, p_diff_ans, mean_diff_ans = np.nan, np.nan, np.nan

    # TEST 2: Brain-pixel correlation (averaged across subjects)
    brain_pixel_correlations = []
    for subj, brain_vec in brain_rdms.items():
        r_bp, _ = spearmanr(brain_vec, pixel_vec)
        brain_pixel_correlations.append(r_bp)

    mean_brain_pixel_r = float(np.mean(brain_pixel_correlations)) if brain_pixel_correlations else np.nan

    # Group stats on partial r (original test)
    vals = subj_df["Partial_Spearman_R_PI_14"].dropna().to_numpy()
    vals_ans = subj_df["Partial_Spearman_R_ANS"].dropna().to_numpy()
    t_stat = np.nan
    p_value = np.nan
    n_vals = int(vals.size)
    if vals.size:
        z_vals = np.arctanh(vals)
        t_stat, p_value = ttest_1samp(z_vals, baseline)
        n_vals = int(len(z_vals))

    summary = pd.DataFrame(
        [
            {
                "Subject": "SUMMARY",
                "Raw_Spearman_R_PI_14": float(np.mean(raw_vals)) if raw_vals.size else np.nan,
                "Partial_Spearman_R_PI_14": float(np.mean(partial_vals)) if partial_vals.size else np.nan,
                "Raw_Spearman_R_ANS": float(np.mean(raw_vals_ans)) if raw_vals_ans.size else np.nan,
                "Partial_Spearman_R_ANS": float(np.mean(partial_vals_ans)) if partial_vals_ans.size else np.nan,
                "Raw_Spearman_R_RT": float(np.mean(raw_vals_rt)) if raw_vals_rt.size else np.nan,
                "Partial_Spearman_R_RT": float(np.mean(partial_vals_rt)) if partial_vals_rt.size else np.nan,
                "Group_T": t_stat,
                "Group_P": p_value,
                "N": n_vals,
                "Diff_T": t_diff,
                "Diff_P": p_diff,
                "Mean_Diff": mean_diff,
                "Diff_T_ANS": t_diff_ans,
                "Diff_P_ANS": p_diff_ans,
                "Mean_Diff_ANS": mean_diff_ans,
                "Brain_Pixel_R": mean_brain_pixel_r,
            }
        ]
    )
    subj_df = pd.concat([subj_df, summary], ignore_index=True)

    # Optional: group stats on ANS partial (same baseline)
    if vals_ans.size:
        z_vals_ans = np.arctanh(vals_ans)
        t_stat_ans, p_value_ans = ttest_1samp(z_vals_ans, baseline)
        subj_df.loc[subj_df["Subject"] == "SUMMARY", "Group_T_ANS"] = t_stat_ans
        subj_df.loc[subj_df["Subject"] == "SUMMARY", "Group_P_ANS"] = p_value_ans

    # Residual heatmap averaged across subjects (add back mean to align with accuracy scale)
    if residuals:
        mean_resid = np.nanmean(np.vstack(residuals), axis=0)
        resid_mat = np.zeros((len(codes), len(codes)), dtype=float)
        tri_idx = np.tril_indices(len(codes), k=-1)
        # Add back global mean accuracy (from raw accuracies) to residuals
        overall_mean = float(np.nanmean(raw_means)) if raw_means else 0.0
        # Mean-added residuals (accuracy-like) for paper view
        resid_mat[tri_idx] = mean_resid + overall_mean
        resid_mat = resid_mat + resid_mat.T
        np.fill_diagonal(resid_mat, 100.0)  # match brain diagonal styling
        plot_heatmap(
            resid_mat,
            codes,
            "Residual Brain RDM (Pixels Removed)",
            output_dir / "residual_brain_rdm_heatmap.png",
            vmin=50.0,
            vmax=80.0,
        )

        # Pure residuals (no mean add), diverging colormap centered at 0
        pure_mat = np.zeros((len(codes), len(codes)), dtype=float)
        pure_mat[tri_idx] = mean_resid
        pure_mat = pure_mat + pure_mat.T
        np.fill_diagonal(pure_mat, 0.0)
        plot_heatmap_diverging(
            pure_mat,
            codes,
            "Residual Brain RDM (Pure Deviations)",
            output_dir / "residual_brain_rdm_pure.png",
            vcenter=0.0,
        )

    # Save main results CSV
    out_csv = prefixed_path(run_root=run_root, kind="confounds", stem="confound_stats", ext=".csv")
    subj_df.to_csv(out_csv, index=False)

    # Generate summary statistics table
    summary_rows = [
        {
            "Test": "Raw Brain–PI(1-4)-ANS Correlation",
            "Mean_r": float(np.mean(raw_vals)),
            "SD_r": float(np.std(raw_vals, ddof=1)),
            "t_statistic": np.nan,
            "p_value": np.nan,
            "Interpretation": "Brain matches PI(1-4)-ANS model (before pixel control)",
        },
        {
            "Test": "Partial Brain–PI(1-4)-ANS Correlation (controlling pixels)",
            "Mean_r": float(np.mean(partial_vals)),
            "SD_r": float(np.std(partial_vals, ddof=1)),
            "t_statistic": t_stat,
            "p_value": p_value,
            "Interpretation": "Brain matches PI(1-4)-ANS model (after pixel control)",
        },
        {
            "Test": "Raw Brain–ANS (log-ratio) Correlation",
            "Mean_r": float(np.mean(raw_vals_ans)) if raw_vals_ans.size else np.nan,
            "SD_r": float(np.std(raw_vals_ans, ddof=1)) if raw_vals_ans.size else np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "Interpretation": "Brain matches ANS model (before pixel control)",
        },
        {
            "Test": "Partial Brain–ANS (log-ratio) Correlation (controlling pixels)",
            "Mean_r": float(np.mean(partial_vals_ans)) if partial_vals_ans.size else np.nan,
            "SD_r": float(np.std(partial_vals_ans, ddof=1)) if partial_vals_ans.size else np.nan,
            "t_statistic": subj_df[subj_df["Subject"] == "SUMMARY"].get("Group_T_ANS", pd.Series([np.nan])).iloc[0],
            "p_value": subj_df[subj_df["Subject"] == "SUMMARY"].get("Group_P_ANS", pd.Series([np.nan])).iloc[0],
            "Interpretation": "Brain matches ANS model (after pixel control)",
        },
    ]
    if raw_vals_rt.size or partial_vals_rt.size:
        summary_rows.extend(
            [
                {
                    "Test": "Raw Brain–RT Landing Correlation",
                    "Mean_r": float(np.mean(raw_vals_rt)) if raw_vals_rt.size else np.nan,
                    "SD_r": float(np.std(raw_vals_rt, ddof=1)) if raw_vals_rt.size else np.nan,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "Interpretation": "Brain aligns with behavioral RT distances (raw)",
                },
                {
                    "Test": "Partial Brain–RT Landing Correlation (controlling pixels)",
                    "Mean_r": float(np.mean(partial_vals_rt)) if partial_vals_rt.size else np.nan,
                    "SD_r": float(np.std(partial_vals_rt, ddof=1)) if partial_vals_rt.size else np.nan,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "Interpretation": "Brain aligns with RT distances (after pixel control)",
                },
            ]
        )

    summary_stats = pd.DataFrame(summary_rows + [
        {
            "Test": "Raw vs Partial Difference",
            "Mean_r": mean_diff,
            "SD_r": float(np.std(diff_vals, ddof=1)) if diff_vals.size else np.nan,
            "t_statistic": t_diff,
            "p_value": p_diff,
            "Interpretation": "Pixels did not drive brain–PI(1-4)-ANS match" if abs(mean_diff) < 0.05 else "Pixels affected brain–PI(1-4)-ANS match",
        },
        {
            "Test": "Raw vs Partial Difference (ANS)",
            "Mean_r": mean_diff_ans,
            "SD_r": float(np.std(diff_vals_ans, ddof=1)) if diff_vals_ans.size else np.nan,
            "t_statistic": t_diff_ans,
            "p_value": p_diff_ans,
            "Interpretation": "Pixels did not drive brain–ANS match" if abs(mean_diff_ans) < 0.05 else "Pixels affected brain–ANS match",
        },
        {
            "Test": "Brain-Pixel Correlation",
            "Mean_r": mean_brain_pixel_r,
            "SD_r": float(np.std(brain_pixel_correlations)) if brain_pixel_correlations else np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "Interpretation": "Brain weakly driven by pixels" if abs(mean_brain_pixel_r) < 0.3 else "Brain may use pixel features",
        },
    ])
    summary_csv = prefixed_path(run_root=run_root, kind="confounds", stem="confound_summary_statistics", ext=".csv")
    summary_stats.to_csv(summary_csv, index=False)

    # Generate publication-quality LaTeX-style table
    fig_table, ax_table = plt.subplots(figsize=(12, 4))
    ax_table.axis('tight')
    ax_table.axis('off')

    # Prepare table data with formatted values
    table_data = []
    for _, row in summary_stats.iterrows():
        test_name = row['Test']
        mean_r = f"{row['Mean_r']:.3f}"
        sd_r = f"{row['SD_r']:.3f}"

        # Format statistics with significance stars
        if pd.notna(row['t_statistic']):
            t_val = f"{row['t_statistic']:.2f}"
            p_val = row['p_value']
            if p_val < 0.001:
                p_str = f"{p_val:.2e}***"
            elif p_val < 0.01:
                p_str = f"{p_val:.3f}**"
            elif p_val < 0.05:
                p_str = f"{p_val:.3f}*"
            else:
                p_str = f"{p_val:.3f}"
            stats = f"t={t_val}, p={p_str}"
        else:
            stats = "-"

        interp = row['Interpretation']
        table_data.append([test_name, mean_r, sd_r, stats, interp])

    # Create table
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Test', 'Mean r', 'SD', 'Statistics', 'Interpretation'],
        cellLoc='left',
        loc='center',
        colWidths=[0.25, 0.10, 0.10, 0.20, 0.35]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Header styling
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
            cell.set_edgecolor('white')
            cell.set_linewidth(2)

    plt.title('Confound Control Analysis Summary', fontsize=14, weight='bold', pad=20)

    # Add footnote (dynamic so change-condition runs remain accurate)
    n_subj = len(brain_rdms)
    n_codes = len(codes)
    n_pairs = int(n_codes * (n_codes - 1) / 2)
    footnote = f"*** p<0.001  ** p<0.01  * p<0.05\\nn={n_subj} subjects, {n_pairs} pairs"
    fig_table.text(0.5, 0.02, footnote, ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    table_path = prefixed_path(run_root=run_root, kind="confounds", stem="confound_summary_table", ext=".png")
    fig_table.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close(fig_table)

    print(f"[analyze_rsa_confounds] Summary table saved to {table_path}")

    # Generate comparison plots only when correlations are defined (need >= 2 pairwise entries).
    if raw_vals.size and partial_vals.size:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Scatter plot of raw vs partial
        ax = axes[0]
        ax.scatter(raw_vals, partial_vals, alpha=0.6, s=80, color='C0')
        ax.plot([raw_vals.min(), raw_vals.max()], [raw_vals.min(), raw_vals.max()],
                'k--', alpha=0.5, label='y=x (no change)')
        ax.set_xlabel('Raw Spearman r (Brain–PI(1-4)-ANS)')
        ax.set_ylabel('Partial Spearman r (Brain–PI(1-4)-ANS, controlling pixels)')
        ax.set_title(f'Raw vs Partial Correlations\\n(n={n_subj} subjects)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Plot 2: Distribution of differences
        ax = axes[1]
        ax.hist(diff_vals, bins=15, alpha=0.7, color='C1', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax.axvline(mean_diff, color='green', linestyle='-', linewidth=2,
                   label=f'Mean diff = {mean_diff:.3f}')
        ax.set_xlabel('Difference (Partial - Raw)')
        ax.set_ylabel('Number of Subjects')
        ax.set_title(f'Distribution of Correlation Differences\\nt={t_diff:.2f}, p={p_diff:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Brain-Pixel vs Brain-PI(1-4)-ANS correlations
        ax = axes[2]
        brain_theory_raw = raw_vals
        x_pos = np.arange(len(brain_pixel_correlations))
        width = 0.35
        ax.bar(x_pos - width/2, brain_theory_raw, width, label='Brain–PI(1-4)-ANS (raw)',
               alpha=0.8, color='C0')
        ax.bar(x_pos + width/2, brain_pixel_correlations, width, label='Brain-Pixel',
               alpha=0.8, color='C3')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Subject Index')
        ax.set_ylabel('Spearman r')
        ax.set_title(f'Brain–PI(1-4)-ANS vs Brain-Pixel Correlations\\n(Mean Brain-Pixel r = {mean_brain_pixel_r:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = prefixed_path(run_root=run_root, kind="confounds", stem="confound_comparison_plots", ext=".png")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

        # Save individual plots as well
        # Individual Plot 1: Raw vs Partial scatter
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.scatter(raw_vals, partial_vals, alpha=0.6, s=100, color='C0', edgecolors='black', linewidth=0.5)
        ax1.plot([raw_vals.min(), raw_vals.max()], [raw_vals.min(), raw_vals.max()],
                'k--', alpha=0.5, linewidth=2, label='y=x (no change)')
        ax1.set_xlabel('Raw Spearman r (Brain–PI(1-4)-ANS)', fontsize=12)
        ax1.set_ylabel('Partial Spearman r\n(Brain–PI(1-4)-ANS, controlling pixels)', fontsize=12)
        ax1.set_title(f'Raw vs Partial Correlations\\n(n={n_subj} subjects)', fontsize=14, weight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        plt.tight_layout()
        fig1.savefig(prefixed_path(run_root=run_root, kind="confounds", stem="plot1_raw_vs_partial_scatter", ext=".png"), dpi=300)
        plt.close(fig1)

        # Individual Plot 2: Distribution histogram
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(diff_vals, bins=15, alpha=0.7, color='C1', edgecolor='black', linewidth=1.5)
        ax2.axvline(0, color='red', linestyle='--', linewidth=3, label='No difference', alpha=0.8)
        ax2.axvline(mean_diff, color='green', linestyle='-', linewidth=3,
                   label=f'Mean diff = {mean_diff:.3f}', alpha=0.8)
        ax2.set_xlabel('Difference (Partial - Raw)', fontsize=12)
        ax2.set_ylabel('Number of Subjects', fontsize=12)
        p_sig = "n.s." if p_diff >= 0.05 else ("*" if p_diff < 0.05 else "**")
        ax2.set_title(f'Distribution of Correlation Differences\\nt={t_diff:.2f}, p={p_diff:.4f} {p_sig}',
                      fontsize=14, weight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig2.savefig(prefixed_path(run_root=run_root, kind="confounds", stem="plot2_difference_distribution", ext=".png"), dpi=300)
        plt.close(fig2)

        # Individual Plot 3: Brain–PI(1-4)-ANS vs Brain-Pixel comparison
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(brain_pixel_correlations))
        width = 0.35
        ax3.bar(x_pos - width/2, raw_vals, width, label='Brain–PI(1-4)-ANS (raw)',
               alpha=0.85, color='C0', edgecolor='black', linewidth=0.5)
        ax3.bar(x_pos + width/2, brain_pixel_correlations, width, label='Brain-Pixel',
               alpha=0.85, color='C3', edgecolor='black', linewidth=0.5)
        ax3.axhline(0, color='black', linewidth=1.2)
        ax3.axhline(mean_brain_pixel_r, color='C3', linestyle='--', linewidth=2, alpha=0.5,
                    label=f'Mean Brain-Pixel r={mean_brain_pixel_r:.3f}')
        ax3.set_xlabel('Subject Index', fontsize=12)
        ax3.set_ylabel('Spearman r', fontsize=12)
        ax3.set_title('Brain–PI(1-4)-ANS vs Brain-Pixel Correlations',
                      fontsize=14, weight='bold')
        ax3.legend(fontsize=11, loc='best')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig3.savefig(prefixed_path(run_root=run_root, kind="confounds", stem="plot3_brain_theory_vs_pixel", ext=".png"), dpi=300)
        plt.close(fig3)

        print(f"\n[analyze_rsa_confounds] Summary statistics saved to {summary_csv}")
        print(f"[analyze_rsa_confounds] Comparison plots saved to {plot_path}")
        print(f"[analyze_rsa_confounds] Individual plots saved (plot1, plot2, plot3)")
    else:
        print(
            "\n[analyze_rsa_confounds] Note: Not enough valid correlation values to render comparison plots "
            f"(need >=2 pairs; got n_pairs={n_pairs}). Skipping plot panel outputs."
        )

    return subj_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSA pixel control partial correlation analysis.")
    parser.add_argument("--config", default="configs/tasks/rsa/rsa_pixel_control.yaml")
    parser.add_argument("--master-csv", type=Path, default=None)
    parser.add_argument("--stimuli-csv", type=Path, default=None)
    parser.add_argument(
        "--rt-summary-csv",
        type=Path,
        default=None,
        help="Optional RT subject-level summary CSV for building an RT landing model RDM.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline", type=float, default=None)
    return parser.parse_args()


def load_config(cfg_path: Path) -> Dict:
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    return json.loads(Path(cfg_path).read_text()) if cfg_path.suffix == ".json" else __import__("yaml").safe_load(cfg_path.read_text())


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    master_csv = Path(args.master_csv) if args.master_csv else Path(cfg["master_csv"])
    stimuli_csv = Path(args.stimuli_csv) if args.stimuli_csv else Path(cfg["stimuli_csv"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    baseline = args.baseline if args.baseline is not None else float(cfg.get("baseline_r", 0.4))
    # If the master CSV is explicitly provided, infer codes from it so landing-digit runs work
    # without needing a separate config file.
    if args.master_csv:
        codes = infer_codes_from_master_csv(master_csv)
    else:
        codes = cfg.get("codes", DEFAULT_CODES)
    rt_summary_csv = (
        Path(args.rt_summary_csv)
        if args.rt_summary_csv
        else Path(cfg["rt_summary_csv"])
        if cfg.get("rt_summary_csv")
        else None
    )
    df = run_analysis(
        master_csv,
        stimuli_csv,
        output_dir,
        codes=codes,
        baseline=baseline,
        rt_summary_csv=rt_summary_csv,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

