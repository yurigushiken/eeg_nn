"""
Visualize time-resolved RSA results as temporal emergence curves.

This script reads rsa_temporal_results.csv and generates publication-ready
line plots showing accuracy over time, with error bars representing SEM
across seeds/subjects.

Outputs:
    - temporal_emergence_[pair].png: Individual pair time courses
    - temporal_emergence_all_pairs.png: All pairs overlaid
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_temporal_results(csv_path: Path) -> pd.DataFrame:
    """
    Load temporal results CSV.

    Args:
        csv_path: Path to rsa_temporal_results.csv

    Returns:
        DataFrame with temporal results
    """
    df = pd.read_csv(csv_path)
    validate_temporal_csv(df)
    return df


def validate_temporal_csv(df: pd.DataFrame) -> bool:
    """
    Validate that CSV has required columns.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValueError: If required columns are missing
    """
    required = ["ClassA", "ClassB", "Seed", "TimeWindow_Center", "Accuracy"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return True


def aggregate_temporal_data(
    df: pd.DataFrame,
    pair: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Aggregate temporal data across seeds, computing mean and SEM.

    Args:
        df: Raw temporal results
        pair: Optional (ClassA, ClassB) tuple to filter

    Returns:
        Aggregated DataFrame with Mean_Accuracy, SEM_Accuracy columns
    """
    if pair is not None:
        df = df[(df["ClassA"] == pair[0]) & (df["ClassB"] == pair[1])].copy()

    # Group by time window
    agg = df.groupby("TimeWindow_Center").agg({
        "Accuracy": ["mean", "sem", "count"],
        "MacroF1": ["mean", "sem"],
        "MinClassF1": ["mean", "sem"],
    }).reset_index()

    # Flatten column names
    agg.columns = [
        "TimeWindow_Center",
        "Mean_Accuracy", "SEM_Accuracy", "N_Seeds",
        "Mean_MacroF1", "SEM_MacroF1",
        "Mean_MinClassF1", "SEM_MinClassF1",
    ]

    return agg


def format_pair_label(class_a: int, class_b: int) -> str:
    """
    Format pair label for display (e.g., 11, 22 -> '1 vs 2').

    Args:
        class_a: First class code
        class_b: Second class code

    Returns:
        Formatted label string
    """
    # Convert 11 -> 1, 22 -> 2, etc.
    a = class_a // 11
    b = class_b // 11
    return f"{a} vs {b}"


def find_peak_latency(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Find time of peak accuracy.

    Args:
        df: Aggregated temporal data

    Returns:
        (peak_time_ms, peak_accuracy)
    """
    peak_idx = df["Mean_Accuracy"].idxmax()
    peak_time = df.loc[peak_idx, "TimeWindow_Center"]
    peak_acc = df.loc[peak_idx, "Mean_Accuracy"]
    return float(peak_time), float(peak_acc)


def find_emergence_onset(
    df: pd.DataFrame,
    threshold: float = 51.0
) -> Optional[float]:
    """
    Find first time window where accuracy exceeds threshold.

    Args:
        df: Aggregated temporal data
        threshold: Accuracy threshold (default 51% = 1% above chance)

    Returns:
        Time in ms, or None if never exceeds threshold
    """
    above_thresh = df[df["Mean_Accuracy"] >= threshold]
    if len(above_thresh) == 0:
        return None
    return float(above_thresh.iloc[0]["TimeWindow_Center"])


def create_temporal_plot(
    df: pd.DataFrame,
    pair_label: str,
    chance_level: float = 50.0
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create temporal emergence plot for a single pair.

    Args:
        df: Aggregated temporal data
        pair_label: Display label for the pair
        chance_level: Chance accuracy level

    Returns:
        (fig, ax) tuple
    """
    # Set publication style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot main line with error ribbon
    times = df["TimeWindow_Center"].values
    acc_mean = df["Mean_Accuracy"].values
    acc_sem = df["SEM_Accuracy"].values

    # Main line
    ax.plot(times, acc_mean, linewidth=2.5, color='#2a6f97', label=pair_label, marker='o', markersize=4)

    # SEM ribbon
    ax.fill_between(
        times,
        acc_mean - acc_sem,
        acc_mean + acc_sem,
        alpha=0.25,
        color='#2a6f97'
    )

    # Chance line
    ax.axhline(chance_level, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance (50%)')

    # Find and mark peak
    peak_time, peak_acc = find_peak_latency(df)
    ax.plot(peak_time, peak_acc, 'r*', markersize=15, label=f'Peak: {peak_acc:.1f}% @ {int(peak_time)}ms')

    # Styling
    ax.set_xlabel('Time (ms)', fontweight='bold')
    ax.set_ylabel('Decoding Accuracy (%)', fontweight='bold')
    ax.set_title(f'Temporal Emergence: {pair_label}', fontweight='bold', pad=15)
    ax.set_ylim(40, 70)  # Reasonable range for binary classification
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def create_multi_pair_plot(
    pairs_data: Dict[str, pd.DataFrame],
    chance_level: float = 50.0
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create plot with multiple pairs overlaid.

    Args:
        pairs_data: Dict mapping pair labels to aggregated DataFrames
        chance_level: Chance accuracy level

    Returns:
        (fig, ax) tuple
    """
    # Set publication style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette for multiple pairs
    colors = sns.color_palette("tab10", n_colors=len(pairs_data))

    for idx, (label, df) in enumerate(pairs_data.items()):
        times = df["TimeWindow_Center"].values
        acc_mean = df["Mean_Accuracy"].values
        acc_sem = df["SEM_Accuracy"].values

        color = colors[idx]

        # Main line
        ax.plot(times, acc_mean, linewidth=2.5, color=color, label=label, marker='o', markersize=3)

        # SEM ribbon
        ax.fill_between(times, acc_mean - acc_sem, acc_mean + acc_sem, alpha=0.15, color=color)

    # Chance line
    ax.axhline(chance_level, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance (50%)')

    # Styling
    ax.set_xlabel('Time (ms)', fontweight='bold')
    ax.set_ylabel('Decoding Accuracy (%)', fontweight='bold')
    ax.set_title('Temporal Emergence: All Pairs', fontweight='bold', pad=15)
    ax.set_ylim(40, 75)
    ax.legend(loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def save_temporal_plot(fig: plt.Figure, output_path: Path) -> None:
    """
    Save plot to file.

    Args:
        fig: Matplotlib figure
        output_path: Path to save PNG
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[visualize_temporal] Saved plot to {output_path}")


def get_unique_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Extract unique (ClassA, ClassB) pairs from results.

    Args:
        df: Temporal results DataFrame

    Returns:
        List of (ClassA, ClassB) tuples
    """
    pairs = df[["ClassA", "ClassB"]].drop_duplicates()
    return [tuple(row) for _, row in pairs.iterrows()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize temporal RSA emergence curves.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/runs/rsa_temporal_v1/rsa_temporal_results.csv"),
        help="Path to rsa_temporal_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (defaults to csv_dir/figures)",
    )
    parser.add_argument(
        "--chance",
        type=float,
        default=50.0,
        help="Chance level for reference line (default: 50%)",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        type=int,
        help="Specific pairs to plot (e.g., --pairs 11 22 33 44). If not specified, all pairs are plotted.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution."""
    args = parse_args()
    csv_path = args.csv

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load data
    df = load_temporal_results(csv_path)
    print(f"[visualize_temporal] Loaded {len(df)} rows from {csv_path}")

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = csv_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get pairs to plot
    if args.pairs:
        # Parse pairs from command line (assumes pairs of codes: 11 22 means pair (11, 22))
        if len(args.pairs) % 2 != 0:
            raise ValueError("--pairs requires an even number of codes (pairs of ClassA ClassB)")
        pairs = [(args.pairs[i], args.pairs[i+1]) for i in range(0, len(args.pairs), 2)]
    else:
        pairs = get_unique_pairs(df)

    print(f"[visualize_temporal] Plotting {len(pairs)} pairs: {pairs}")

    # Plot individual pairs
    pairs_data = {}
    for pair in pairs:
        agg = aggregate_temporal_data(df, pair=pair)
        pair_label = format_pair_label(pair[0], pair[1])
        pairs_data[pair_label] = agg

        # Individual plot
        fig, ax = create_temporal_plot(agg, pair_label, chance_level=args.chance)
        output_path = output_dir / f"temporal_emergence_{pair[0]}v{pair[1]}.png"
        save_temporal_plot(fig, output_path)
        plt.close(fig)

        # Print summary stats
        peak_time, peak_acc = find_peak_latency(agg)
        onset = find_emergence_onset(agg, threshold=args.chance + 1.0)
        print(f"[visualize_temporal] {pair_label}: Peak={peak_acc:.1f}% @ {int(peak_time)}ms, Onset={onset}ms")

    # Multi-pair overlay plot
    if len(pairs) > 1:
        fig, ax = create_multi_pair_plot(pairs_data, chance_level=args.chance)
        output_path = output_dir / "temporal_emergence_all_pairs.png"
        save_temporal_plot(fig, output_path)
        plt.close(fig)

    print(f"[visualize_temporal] All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
