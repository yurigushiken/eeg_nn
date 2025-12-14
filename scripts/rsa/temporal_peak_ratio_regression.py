from __future__ import annotations

"""
Paper-grade LOSO regression: peak accuracy vs numerical ratio.

Key idea:
  For each pair and each held-out subject, choose the peak time using ONLY the other subjects'
  mean timecourse (LOSO). Then evaluate the held-out subject's accuracy at that selected time.

This avoids reusing the held-out subject to define the peak (reducing peak-selection bias).

Outputs:
  - figures/{analysis_id}__peak_accuracy_vs_log2ratio_loso_regression.png
  - tables/{analysis_id}__peak_accuracy_vs_log2ratio_loso_regression.csv
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.rsa.naming import analysis_id_from_run_root, prefixed_path, prefixed_title


@dataclass(frozen=True)
class LosoRegressionSummary:
    n_subjects: int
    n_pairs: int
    n_ratio_points_per_subject_mean: float
    slope_mean: float
    slope_sem: float
    t_stat: float
    p_value: float
    ci95_low: float
    ci95_high: float


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_loso_peak_accuracy_per_subject_pair(subject_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute LOSO peak time per pair, per held-out subject.

    Returns rows:
      Subject, ClassA, ClassB, Ratio, Log2Ratio, Peak_Time_LOSO, Accuracy_LOSO
    """
    _ensure_cols(subject_df, ["ClassA", "ClassB", "Subject", "TimeWindow_Center", "Accuracy"])

    # Normalize dtypes
    df = subject_df.copy()
    df["ClassA"] = df["ClassA"].astype(int)
    df["ClassB"] = df["ClassB"].astype(int)
    df["Subject"] = df["Subject"].astype(int)
    df["TimeWindow_Center"] = df["TimeWindow_Center"].astype(float)
    df["Accuracy"] = df["Accuracy"].astype(float)

    pairs = df[["ClassA", "ClassB"]].drop_duplicates().sort_values(["ClassA", "ClassB"]).to_numpy()
    subjects_all = np.sort(df["Subject"].unique())

    rows: list[dict] = []
    for class_a, class_b in pairs:
        pair_df = df[(df["ClassA"] == int(class_a)) & (df["ClassB"] == int(class_b))]
        if pair_df.empty:
            continue

        # ratio based on codes' implied numerosity (cardinality coding cancels out in ratio)
        ratio = float(class_b) / float(class_a)
        log2ratio = float(np.log2(ratio))

        for heldout in subjects_all:
            train_df = pair_df[pair_df["Subject"] != int(heldout)]
            if train_df.empty:
                continue

            # select peak time based on training subjects only
            train_means = train_df.groupby("TimeWindow_Center", as_index=True)["Accuracy"].mean()
            if train_means.empty:
                continue

            peak_time = float(train_means.idxmax())

            held_df = pair_df[(pair_df["Subject"] == int(heldout)) & (pair_df["TimeWindow_Center"] == peak_time)]
            if held_df.empty:
                # If this subject is missing that window (should not happen), skip
                continue
            held_acc = float(held_df["Accuracy"].mean())

            rows.append(
                {
                    "Subject": int(heldout),
                    "ClassA": int(class_a),
                    "ClassB": int(class_b),
                    "Ratio": ratio,
                    "Log2Ratio": log2ratio,
                    "Peak_Time_LOSO": peak_time,
                    "Accuracy_LOSO": held_acc,
                }
            )

    return pd.DataFrame(rows)


def _subject_level_slopes(subject_ratio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given subject×ratio mean accuracies, compute within-subject OLS slope vs log2(ratio).
    Returns one row per subject with slope/intercept/r/p/stderr/n_points.
    """
    out_rows: list[dict] = []
    for subj, g in subject_ratio_df.groupby("Subject"):
        gg = g.dropna(subset=["Log2Ratio", "Accuracy_LOSO"]).sort_values("Log2Ratio")
        x = gg["Log2Ratio"].to_numpy(dtype=float)
        y = gg["Accuracy_LOSO"].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        lr = stats.linregress(x, y)
        out_rows.append(
            {
                "Subject": int(subj),
                "n_points": int(len(x)),
                "slope": float(lr.slope),
                "intercept": float(lr.intercept),
                "r": float(lr.rvalue),
                "p": float(lr.pvalue),
                "stderr": float(lr.stderr) if lr.stderr is not None else float("nan"),
            }
        )
    return pd.DataFrame(out_rows)


def summarize_loso_ratio_effect(loso_df: pd.DataFrame) -> tuple[pd.DataFrame, LosoRegressionSummary]:
    """
    Paper-grade inference:
      - Convert to subject×ratio mean accuracies
      - Fit within-subject slopes vs log2(ratio)
      - Test if mean slope differs from 0 across subjects (one-sample t-test)
      - Provide 95% CI of mean slope
    """
    _ensure_cols(loso_df, ["Subject", "Log2Ratio", "Accuracy_LOSO", "ClassA", "ClassB"])

    subj_ratio = (
        loso_df.groupby(["Subject", "Log2Ratio"], as_index=False)["Accuracy_LOSO"]
        .mean()
        .sort_values(["Subject", "Log2Ratio"])
    )
    subj_slopes = _subject_level_slopes(subj_ratio)
    if subj_slopes.empty:
        raise ValueError("Not enough data to compute subject-level slopes (need >=2 ratio points per subject).")

    slopes = subj_slopes["slope"].to_numpy(dtype=float)
    n = int(len(slopes))
    mean_slope = float(np.mean(slopes))
    sem_slope = float(stats.sem(slopes))
    t_stat, p_val = stats.ttest_1samp(slopes, popmean=0.0)
    t_stat = float(t_stat)
    p_val = float(p_val)

    # 95% CI for mean slope
    alpha = 0.05
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1)) if n > 1 else float("nan")
    ci_low = float(mean_slope - tcrit * sem_slope) if np.isfinite(tcrit) else float("nan")
    ci_high = float(mean_slope + tcrit * sem_slope) if np.isfinite(tcrit) else float("nan")

    # meta counts
    n_subjects = int(loso_df["Subject"].nunique())
    n_pairs = int(loso_df[["ClassA", "ClassB"]].drop_duplicates().shape[0])
    ratio_points_per_subject = float(subj_ratio.groupby("Subject")["Log2Ratio"].nunique().mean())

    summary = LosoRegressionSummary(
        n_subjects=n_subjects,
        n_pairs=n_pairs,
        n_ratio_points_per_subject_mean=ratio_points_per_subject,
        slope_mean=mean_slope,
        slope_sem=sem_slope,
        t_stat=t_stat,
        p_value=p_val,
        ci95_low=ci_low,
        ci95_high=ci_high,
    )
    return subj_slopes, summary


def plot_peak_accuracy_vs_log2ratio_loso(
    *,
    loso_df: pd.DataFrame,
    subj_slopes: pd.DataFrame,
    summary: LosoRegressionSummary,
    run_root: Path,
) -> tuple[Path, Path]:
    """
    Save regression plot + regression table.
    """
    analysis_id = analysis_id_from_run_root(run_root)

    fig_path = prefixed_path(
        run_root=run_root,
        kind="figures",
        stem="peak_accuracy_vs_log2ratio_loso_regression",
        ext=".png",
    )
    table_path = prefixed_path(
        run_root=run_root,
        kind="tables",
        stem="peak_accuracy_vs_log2ratio_loso_regression",
        ext=".csv",
    )

    # Subject×ratio means for plotting
    subj_ratio = (
        loso_df.groupby(["Subject", "Log2Ratio"], as_index=False)["Accuracy_LOSO"]
        .mean()
        .sort_values(["Subject", "Log2Ratio"])
    )

    # Group mean ± SEM at each ratio point
    group = subj_ratio.groupby("Log2Ratio")["Accuracy_LOSO"].agg(["mean", "sem"]).reset_index()

    # Regression line using mean slope + mean intercept (across subjects)
    b1 = float(subj_slopes["slope"].mean())
    b0 = float(subj_slopes["intercept"].mean())
    xs = np.linspace(float(group["Log2Ratio"].min()), float(group["Log2Ratio"].max()), 100)
    ys = b0 + b1 * xs

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    # Light scatter: each subject's mean at each ratio
    ax.scatter(
        subj_ratio["Log2Ratio"].to_numpy(),
        subj_ratio["Accuracy_LOSO"].to_numpy(),
        s=25,
        alpha=0.25,
        color="#27AE60",
        edgecolor="none",
        label="Subject mean (LOSO peak)",
    )

    # Group mean ± SEM
    ax.errorbar(
        group["Log2Ratio"].to_numpy(),
        group["mean"].to_numpy(),
        yerr=group["sem"].to_numpy(),
        fmt="o",
        markersize=7,
        capsize=4,
        color="#1E8449",
        ecolor="#1E8449",
        alpha=0.9,
        label="Group mean ± SEM",
    )

    # Regression line
    ax.plot(xs, ys, color="#145A32", linewidth=2.2, label="Regression (mean subject slope)")

    ax.axhline(50.0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Chance")
    ax.set_xlabel("log2(Numerical Ratio)", fontsize=11)
    ax.set_ylabel("Accuracy at LOSO-selected peak (%)", fontsize=11)

    stats_txt = (
        f"Mean slope = {summary.slope_mean:.3f} ± {summary.slope_sem:.3f} (SEM)\n"
        f"t({max(summary.n_subjects - 1, 0)}) = {summary.t_stat:.3f}, p = {summary.p_value:.3g}\n"
        f"95% CI [{summary.ci95_low:.3f}, {summary.ci95_high:.3f}]\n"
        f"N subjects = {summary.n_subjects}, N pairs = {summary.n_pairs}"
    )
    ax.text(
        0.02,
        0.98,
        stats_txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", linewidth=0.8),
    )

    ax.set_title(
        prefixed_title(run_root=run_root, title="LOSO peak accuracy vs log2(ratio) regression"),
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Write a compact table: summary row + per-subject rows
    summary_row = pd.DataFrame(
        [
            {
                "analysis_id": analysis_id,
                "method": "LOSO peak selection; within-subject OLS slopes; one-sample t-test on slopes",
                "x": "log2(ratio)",
                "n_subjects": summary.n_subjects,
                "n_pairs": summary.n_pairs,
                "mean_ratio_points_per_subject": summary.n_ratio_points_per_subject_mean,
                "mean_slope": summary.slope_mean,
                "slope_sem": summary.slope_sem,
                "t_stat": summary.t_stat,
                "p_value": summary.p_value,
                "ci95_low": summary.ci95_low,
                "ci95_high": summary.ci95_high,
            }
        ]
    )
    subj_rows = subj_slopes.copy()
    subj_rows.insert(0, "analysis_id", analysis_id)
    subj_rows.insert(1, "method", "within-subject OLS slope (subject mean accuracy per ratio)")
    out_table = pd.concat([summary_row, subj_rows], ignore_index=True)
    out_table.to_csv(table_path, index=False)

    return fig_path, table_path


def run_peak_accuracy_ratio_loso_regression(*, subject_temporal_means_csv: Path, run_root: Path) -> tuple[Path, Path]:
    """
    End-to-end runner used by the temporal peaks pipeline.
    """
    subject_df = pd.read_csv(subject_temporal_means_csv)
    loso_df = compute_loso_peak_accuracy_per_subject_pair(subject_df)
    subj_slopes, summary = summarize_loso_ratio_effect(loso_df)
    return plot_peak_accuracy_vs_log2ratio_loso(loso_df=loso_df, subj_slopes=subj_slopes, summary=summary, run_root=run_root)


