"""
Analyze reaction time data from EEG experiment.

This script extracts RT data from target trials (correct responses only),
aggregates across subjects, and produces publication-quality tables and figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns


def load_rt_data(data_dir: Path) -> pd.DataFrame:
    """
    Load RT data from all subject .fif files.

    Args:
        data_dir: Path to preprocessed data directory

    Returns:
        DataFrame with columns: SubjectID, Cardinality, RT, Trial_ID
    """
    fif_files = sorted(data_dir.glob("sub-*_preprocessed-epo.fif"))

    all_data = []

    for fif_path in fif_files:
        subject_id = fif_path.stem.split("_")[0].replace("sub-", "")

        epochs = mne.read_epochs(fif_path, preload=False, verbose=False)
        metadata = epochs.metadata

        # Extract cardinality from Prime5 column (e.g., "5a.jpg" -> 5)
        metadata["Cardinality"] = metadata["Prime5"].str.extract(r"(\d+)", expand=False).astype(int)

        # Filter for target trials with correct responses only
        target_correct = metadata[
            (metadata["Target.ACC"] == 1.0) &  # Correct responses
            (metadata["Target.RT"] > 0)  # Valid RT
        ].copy()

        target_correct["SubjectID"] = subject_id
        target_correct["Trial_ID"] = range(len(target_correct))

        all_data.append(target_correct[["SubjectID", "Cardinality", "Target.RT", "Trial_ID"]])

    df = pd.concat(all_data, ignore_index=True)
    df.rename(columns={"Target.RT": "RT"}, inplace=True)

    return df


def compute_rt_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics by cardinality.

    Args:
        df: RT data with SubjectID, Cardinality, RT

    Returns:
        DataFrame with cardinality-level statistics
    """
    summary = df.groupby("Cardinality").agg(
        N_Trials=("RT", "count"),
        N_Subjects=("SubjectID", "nunique"),
        Mean_RT=("RT", "mean"),
        Std_RT=("RT", "std"),
        SEM_RT=("RT", lambda x: x.std() / np.sqrt(len(x))),
        Median_RT=("RT", "median"),
        Min_RT=("RT", "min"),
        Max_RT=("RT", "max"),
        Q1_RT=("RT", lambda x: x.quantile(0.25)),
        Q3_RT=("RT", lambda x: x.quantile(0.75)),
    ).reset_index()

    # Add 95% CI
    summary["CI_Lower"] = summary["Mean_RT"] - 1.96 * summary["SEM_RT"]
    summary["CI_Upper"] = summary["Mean_RT"] + 1.96 * summary["SEM_RT"]

    return summary


def compute_subject_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-subject, per-cardinality statistics.

    Args:
        df: RT data

    Returns:
        DataFrame with subject × cardinality statistics
    """
    subject_stats = df.groupby(["SubjectID", "Cardinality"]).agg(
        N_Trials=("RT", "count"),
        Mean_RT=("RT", "mean"),
        Std_RT=("RT", "std"),
    ).reset_index()

    return subject_stats


def save_summary_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Save summary statistics table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Format for display
    display_summary = summary.copy()
    for col in ["Mean_RT", "Std_RT", "SEM_RT", "Median_RT", "CI_Lower", "CI_Upper"]:
        display_summary[col] = display_summary[col].round(1)
    for col in ["Min_RT", "Max_RT", "Q1_RT", "Q3_RT"]:
        display_summary[col] = display_summary[col].round(0).astype(int)

    display_summary.to_csv(output_path, index=False)
    print(f"[analyze_rt] Summary table saved to {output_path}")


def plot_rt_by_cardinality(
    summary: pd.DataFrame,
    output_path: Path,
    style: str = "bar"
) -> None:
    """
    Plot mean RT by cardinality with error bars.

    Args:
        summary: Summary statistics DataFrame
        output_path: Path to save figure
        style: 'bar' or 'line'
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = summary["Cardinality"].values
    y = summary["Mean_RT"].values
    yerr = summary["SEM_RT"].values

    if style == "bar":
        bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.8,
                      color="#2a6f97", edgecolor="black", linewidth=1.2)

        # Add value labels on bars
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi + yerr[i] + 10, f"{yi:.0f}",
                   ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        ax.plot(x, y, marker="o", markersize=8, linewidth=2.5, color="#2a6f97")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.25, color="#2a6f97")

        # Add value labels
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 15, f"{yi:.0f}",
                   ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Cardinality (Numerosity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Reaction Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Reaction Time by Target Numerosity\n(Correct Trials Only)",
                fontsize=14, fontweight="bold", pad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add sample size annotation
    n_total = summary["N_Trials"].sum()
    n_subjects = summary["N_Subjects"].iloc[0]
    ax.text(0.98, 0.98, f"N = {n_total} trials\n{n_subjects} subjects",
           transform=ax.transAxes, ha="right", va="top",
           fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[analyze_rt] RT plot saved to {output_path}")


def plot_rt_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot RT distributions as violin plots by cardinality."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with overlaid box plot
    parts = ax.violinplot(
        [df[df["Cardinality"] == c]["RT"].values for c in sorted(df["Cardinality"].unique())],
        positions=sorted(df["Cardinality"].unique()),
        widths=0.7,
        showmeans=True,
        showmedians=True,
    )

    # Style violin plots
    for pc in parts["bodies"]:
        pc.set_facecolor("#2a6f97")
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.2)

    ax.set_xlabel("Cardinality (Numerosity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reaction Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title("RT Distribution by Target Numerosity\n(Correct Trials Only)",
                fontsize=14, fontweight="bold", pad=15)

    ax.set_xticks(sorted(df["Cardinality"].unique()))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[analyze_rt] RT distribution plot saved to {output_path}")


def plot_subject_variability(subject_stats: pd.DataFrame, output_path: Path) -> None:
    """Plot per-subject RT means as a heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pivot to subject × cardinality matrix
    pivot = subject_stats.pivot(index="SubjectID", columns="Cardinality", values="Mean_RT")

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Mean RT (ms)"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel("Cardinality (Numerosity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Subject ID", fontsize=12, fontweight="bold")
    ax.set_title("Per-Subject Mean RT by Numerosity\n(Correct Trials Only)",
                fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[analyze_rt] Subject variability heatmap saved to {output_path}")


def main() -> None:
    """Main analysis pipeline."""
    # Paths
    data_dir = Path("data_preprocessed/hpf_1.5_lpf_35_baseline-on")
    output_dir = Path("reaction_time")
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    # Create directories
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[analyze_rt] Loading RT data from all subjects...")
    df = load_rt_data(data_dir)

    print(f"[analyze_rt] Loaded {len(df)} correct target trials from {df['SubjectID'].nunique()} subjects")

    # Summary statistics
    print("[analyze_rt] Computing summary statistics...")
    summary = compute_rt_summary_stats(df)

    # Subject-level statistics
    subject_stats = compute_subject_level_stats(df)

    # Save tables
    save_summary_table(summary, tables_dir / "rt_summary_by_cardinality.csv")
    subject_stats.to_csv(tables_dir / "rt_by_subject_and_cardinality.csv", index=False)

    # Save raw data
    df.to_csv(tables_dir / "rt_all_trials.csv", index=False)

    # Generate plots
    print("[analyze_rt] Generating visualizations...")
    plot_rt_by_cardinality(summary, figures_dir / "rt_by_cardinality_bar.png", style="bar")
    plot_rt_by_cardinality(summary, figures_dir / "rt_by_cardinality_line.png", style="line")
    plot_rt_distribution(df, figures_dir / "rt_distribution_violin.png")
    plot_subject_variability(subject_stats, figures_dir / "rt_subject_heatmap.png")

    # Print summary to console
    print("\n" + "="*60)
    print("REACTION TIME SUMMARY (Correct Trials Only)")
    print("="*60)
    print(summary.to_string(index=False))
    print("="*60)

    print(f"\n[analyze_rt] Analysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
