"""
Visualize temporal accuracy curves for all pairs.

Creates publication-ready time-resolved accuracy plots with:
- Mean accuracy across subjects
- ±1 SEM error bars
- Significance markers (FDR-corrected)
- Chance level reference line
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Matplotlib settings for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 300


def categorize_pair_difficulty(class_a: int, class_b: int) -> str:
    """Categorize pair by difficulty based on numerical ratio."""
    ratio = class_b / class_a
    if ratio >= 2.0:
        return 'Easy'
    elif ratio >= 1.5:
        return 'Medium'
    else:
        return 'Hard'


def plot_accuracy_over_time(
    subject_data: pd.DataFrame,
    stats_data: Optional[pd.DataFrame],
    class_a: int,
    class_b: int,
    ax: Optional[plt.Axes] = None,
    show_significance: bool = True,
    baseline: float = 50.0
) -> plt.Axes:
    """
    Plot accuracy vs time for a single pair.

    Args:
        subject_data: Subject-level temporal data
        stats_data: Statistical results with FDR corrections
        class_a: First numerosity
        class_b: Second numerosity
        ax: Matplotlib axes (creates new if None)
        show_significance: Mark significant windows
        baseline: Chance level

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Filter to this pair
    pair_data = subject_data[
        (subject_data['ClassA'] == class_a) &
        (subject_data['ClassB'] == class_b)
    ]

    # Compute mean and SEM across subjects
    grouped = pair_data.groupby('TimeWindow_Center')['Accuracy'].agg(['mean', 'sem'])
    times = grouped.index.values
    means = grouped['mean'].values
    sems = grouped['sem'].values

    # Plot line with error band
    ax.plot(times, means, color='#2E86AB', linewidth=2, label=f'{class_a}v{class_b}')
    ax.fill_between(times, means - sems, means + sems, color='#2E86AB', alpha=0.2)

    # Chance level
    ax.axhline(baseline, color='black', linestyle='--', linewidth=1, label='Chance', zorder=1)

    # Mark significant windows
    if show_significance and stats_data is not None:
        pair_stats = stats_data[
            (stats_data['ClassA'] == class_a) &
            (stats_data['ClassB'] == class_b)
        ].sort_values('TimeWindow_Center')

        sig_windows = pair_stats[pair_stats['significant_fdr'] == True]

        for _, row in sig_windows.iterrows():
            t_center = row['TimeWindow_Center']
            acc = row['Mean_Accuracy']
            ax.plot(t_center, acc, 'r*', markersize=8, zorder=10)

    # Labels and formatting
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Pair {class_a}v{class_b} (Ratio: {class_b/class_a:.2f})', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 500)
    ax.set_ylim(40, 75)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')

    return ax


def create_temporal_curve_figure(
    subject_data: pd.DataFrame,
    stats_data: Optional[pd.DataFrame],
    output_path: Path,
    show_all_pairs: bool = True
) -> None:
    """
    Create multi-panel figure with all temporal curves.

    Args:
        subject_data: Subject-level temporal data
        stats_data: Statistical results
        output_path: Output file path
        show_all_pairs: If True, create 15-panel figure; else grouped by difficulty
    """
    # Get unique pairs
    pairs = subject_data[['ClassA', 'ClassB']].drop_duplicates().values
    pairs = sorted(pairs, key=lambda x: (x[0], x[1]))

    if show_all_pairs:
        # 15-panel figure (5 rows x 3 cols)
        fig = plt.figure(figsize=(15, 20))
        gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3)

        for idx, (class_a, class_b) in enumerate(pairs):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])

            plot_accuracy_over_time(
                subject_data=subject_data,
                stats_data=stats_data,
                class_a=int(class_a),
                class_b=int(class_b),
                ax=ax,
                show_significance=True
            )

        fig.suptitle('Temporal Decoding Accuracy: All Pairs', fontsize=16, fontweight='bold', y=0.995)

    else:
        # Grouped by difficulty (3 panels)
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        difficulty_colors = {
            'Easy': '#27AE60',
            'Medium': '#F39C12',
            'Hard': '#E74C3C'
        }

        for diff_idx, difficulty in enumerate(['Easy', 'Medium', 'Hard']):
            ax = axes[diff_idx]

            # Filter pairs by difficulty
            diff_pairs = [(a, b) for a, b in pairs if categorize_pair_difficulty(a, b) == difficulty]

            for (class_a, class_b) in diff_pairs:
                pair_data = subject_data[
                    (subject_data['ClassA'] == class_a) &
                    (subject_data['ClassB'] == class_b)
                ]

                grouped = pair_data.groupby('TimeWindow_Center')['Accuracy'].agg(['mean', 'sem'])
                times = grouped.index.values
                means = grouped['mean'].values
                sems = grouped['sem'].values

                ax.plot(times, means, linewidth=1.5, label=f'{int(class_a)}v{int(class_b)}', alpha=0.7)
                ax.fill_between(times, means - sems, means + sems, alpha=0.1)

            ax.axhline(50, color='black', linestyle='--', linewidth=1, label='Chance')
            ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title(f'{difficulty} Pairs', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 500)
            ax.set_ylim(40, 75)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='upper right', fontsize=8, ncol=2)

        fig.suptitle('Temporal Decoding Accuracy by Difficulty', fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[viz] Saved: {output_path}")
    plt.close()


def plot_significance_heatmap(
    stats_data: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Create heatmap showing significance across pairs × time windows.

    Args:
        stats_data: Statistical results with FDR corrections
        output_path: Output file path
    """
    # Pivot to get pairs x time matrix
    pivot_data = stats_data.pivot_table(
        index=['ClassA', 'ClassB'],
        columns='TimeWindow_Center',
        values='significant_fdr'
    )

    # Handle missing cells (defensive) and convert boolean to int for plotting
    pivot_data = pivot_data.fillna(False).astype(int)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_xticklabels([f'{int(t)}' for t in pivot_data.columns], fontsize=8, rotation=45, ha='right')

    pair_labels = [f'{int(a)}v{int(b)}' for a, b in pivot_data.index]
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_yticklabels(pair_labels, fontsize=9)

    ax.set_xlabel('Time Window Center (ms)', fontsize=11)
    ax.set_ylabel('Numerosity Pair', fontsize=11)
    ax.set_title('Significant Discrimination (FDR q<0.05)', fontsize=13, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not Sig.', 'Significant'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[viz] Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize temporal accuracy curves"
    )
    parser.add_argument(
        "--subject-data",
        type=Path,
        required=True,
        help="Subject-level temporal data (seeds averaged)"
    )
    parser.add_argument(
        "--stats-data",
        type=Path,
        required=False,
        help="Statistical results with FDR corrections"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for figures"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  TEMPORAL RSA: VISUALIZATION")
    print("="*70)

    # Load data
    print(f"[viz] Loading subject data...")
    subject_df = pd.read_csv(args.subject_data)

    stats_df = None
    if args.stats_data and args.stats_data.exists():
        print(f"[viz] Loading statistical results...")
        stats_df = pd.read_csv(args.stats_data)

    # Create all pairs figure
    print(f"[viz] Creating temporal curves (all pairs)...")
    output_all = args.output_dir / "temporal_curves_all_pairs.png"
    create_temporal_curve_figure(
        subject_data=subject_df,
        stats_data=stats_df,
        output_path=output_all,
        show_all_pairs=True
    )

    # Create grouped figure
    print(f"[viz] Creating temporal curves (grouped by difficulty)...")
    output_grouped = args.output_dir / "temporal_curves_grouped.png"
    create_temporal_curve_figure(
        subject_data=subject_df,
        stats_data=stats_df,
        output_path=output_grouped,
        show_all_pairs=False
    )

    # Create significance heatmap
    if stats_df is not None:
        print(f"[viz] Creating significance heatmap...")
        output_heatmap = args.output_dir / "temporal_significance_heatmap.png"
        plot_significance_heatmap(
            stats_data=stats_df,
            output_path=output_heatmap
        )

    print("="*70)
    print("  [OK] COMPLETE: Temporal curves generated")
    print("="*70)


if __name__ == "__main__":
    main()
