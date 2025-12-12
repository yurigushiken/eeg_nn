"""
Analyze temporal peaks and onset latencies.

CRITICAL STATISTICAL PRINCIPLE (RA Recommendation #5):
- Peak statistics are DESCRIPTIVE ONLY (no p-values!)
- Peaks are selected from the same data being analyzed (circular!)
- Report peaks with bootstrap confidence intervals
- Onset latencies = first window in significant cluster (from FDR-corrected tests)

Outputs:
- Peak timing and accuracy for each pair (descriptive + bootstrap CIs)
- Onset latencies (from cluster analysis)
- Comparison to full-epoch RDM results
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Matplotlib settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 300


def find_peak_per_pair(
    subject_data: pd.DataFrame,
    class_a: int,
    class_b: int
) -> Dict[str, float]:
    """
    Find peak accuracy window for a pair (DESCRIPTIVE, no p-value!).

    Args:
        subject_data: Subject-level temporal data
        class_a: First numerosity
        class_b: Second numerosity

    Returns:
        Dictionary with peak_time, peak_accuracy, peak_sd
    """
    # Filter to this pair
    pair_data = subject_data[
        (subject_data['ClassA'] == class_a) &
        (subject_data['ClassB'] == class_b)
    ]

    # Compute mean accuracy per time window
    temporal_means = pair_data.groupby('TimeWindow_Center')['Accuracy'].agg(['mean', 'std'])

    # Find peak
    peak_idx = temporal_means['mean'].idxmax()
    peak_acc = temporal_means.loc[peak_idx, 'mean']
    peak_sd = temporal_means.loc[peak_idx, 'std']

    return {
        'peak_time': float(peak_idx),
        'peak_accuracy': float(peak_acc),
        'peak_sd': float(peak_sd)
    }


def compute_peak_confidence_intervals(
    subject_data: pd.DataFrame,
    class_a: int,
    class_b: int,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for peak statistics.

    Args:
        subject_data: Subject-level temporal data
        class_a: First numerosity
        class_b: Second numerosity
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)

    Returns:
        Dictionary with peak_time_ci_lower, peak_time_ci_upper,
                       peak_accuracy_ci_lower, peak_accuracy_ci_upper
    """
    # Filter to this pair
    pair_data = subject_data[
        (subject_data['ClassA'] == class_a) &
        (subject_data['ClassB'] == class_b)
    ]

    # Get unique subjects and time windows
    subjects = pair_data['Subject'].unique()
    time_windows = sorted(pair_data['TimeWindow_Center'].unique())

    peak_times = []
    peak_accs = []

    # Bootstrap by resampling subjects
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        # Resample subjects with replacement
        boot_subjects = rng.choice(subjects, size=len(subjects), replace=True)

        # Get data for resampled subjects
        boot_data = pair_data[pair_data['Subject'].isin(boot_subjects)]

        # Compute mean per time window
        boot_means = boot_data.groupby('TimeWindow_Center')['Accuracy'].mean()

        # Find peak
        if len(boot_means) > 0:
            peak_time = boot_means.idxmax()
            peak_acc = boot_means.max()
            peak_times.append(peak_time)
            peak_accs.append(peak_acc)

    # Compute percentile confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        'peak_time_ci_lower': float(np.percentile(peak_times, lower_percentile)),
        'peak_time_ci_upper': float(np.percentile(peak_times, upper_percentile)),
        'peak_accuracy_ci_lower': float(np.percentile(peak_accs, lower_percentile)),
        'peak_accuracy_ci_upper': float(np.percentile(peak_accs, upper_percentile))
    }


def find_onset_latencies(
    stats_data: pd.DataFrame,
    class_a: int,
    class_b: int,
    min_cluster_size: int = 2
) -> Optional[Dict[str, float]]:
    """
    Find onset latency = first window in first significant cluster.

    CRITICAL: Uses cluster correction (≥2 consecutive significant windows).
    This prevents reporting onsets from isolated false positives.

    Args:
        stats_data: Statistical results with FDR corrections
        class_a: First numerosity
        class_b: Second numerosity
        min_cluster_size: Minimum consecutive significant windows (default: 2)

    Returns:
        Dictionary with onset_time, onset_accuracy, or None if no significant cluster
    """
    # Filter to this pair and sort by time
    pair_stats = stats_data[
        (stats_data['ClassA'] == class_a) &
        (stats_data['ClassB'] == class_b)
    ].sort_values('TimeWindow_Center')

    # Find clusters of consecutive significant windows
    sig_mask = pair_stats['significant_fdr'].values
    clusters = []
    current_cluster = []

    for idx, is_sig in enumerate(sig_mask):
        if is_sig:
            current_cluster.append(idx)
        else:
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster.copy())
            current_cluster = []

    # Check last cluster
    if len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)

    # If no valid clusters, no onset
    if len(clusters) == 0:
        return None

    # First window of first cluster = onset
    first_cluster = clusters[0]
    onset_idx = first_cluster[0]
    onset_row = pair_stats.iloc[onset_idx]

    return {
        'onset_time': float(onset_row['TimeWindow_Center']),
        'onset_accuracy': float(onset_row['Mean_Accuracy'])
    }


def analyze_all_pairs_peaks(
    subject_data: pd.DataFrame,
    stats_data: Optional[pd.DataFrame],
    n_bootstrap: int = 10000
) -> pd.DataFrame:
    """
    Analyze peaks and onsets for all pairs.

    Args:
        subject_data: Subject-level temporal data
        stats_data: Statistical results (optional, for onset latencies)
        n_bootstrap: Number of bootstrap samples for CIs

    Returns:
        DataFrame with peak and onset statistics per pair
    """
    print(f"[peaks] Analyzing peaks and onsets for all pairs...")

    results = []

    # Get unique pairs
    pairs = subject_data[['ClassA', 'ClassB']].drop_duplicates().values

    for class_a, class_b in pairs:
        # Find peak (descriptive)
        peak_stats = find_peak_per_pair(
            subject_data=subject_data,
            class_a=int(class_a),
            class_b=int(class_b)
        )

        # Bootstrap CIs for peak
        peak_cis = compute_peak_confidence_intervals(
            subject_data=subject_data,
            class_a=int(class_a),
            class_b=int(class_b),
            n_bootstrap=n_bootstrap
        )

        # Find onset (from FDR-corrected tests)
        onset_stats = None
        if stats_data is not None:
            onset_stats = find_onset_latencies(
                stats_data=stats_data,
                class_a=int(class_a),
                class_b=int(class_b)
            )

        # Combine results
        result_row = {
            'ClassA': int(class_a),
            'ClassB': int(class_b),
            'Ratio': float(class_b / class_a),
            'Peak_Time': peak_stats['peak_time'],
            'Peak_Accuracy': peak_stats['peak_accuracy'],
            'Peak_SD': peak_stats['peak_sd'],
            'Peak_Time_CI_Lower': peak_cis['peak_time_ci_lower'],
            'Peak_Time_CI_Upper': peak_cis['peak_time_ci_upper'],
            'Peak_Accuracy_CI_Lower': peak_cis['peak_accuracy_ci_lower'],
            'Peak_Accuracy_CI_Upper': peak_cis['peak_accuracy_ci_upper']
        }

        if onset_stats is not None:
            result_row['Onset_Time'] = onset_stats['onset_time']
            result_row['Onset_Accuracy'] = onset_stats['onset_accuracy']
        else:
            result_row['Onset_Time'] = np.nan
            result_row['Onset_Accuracy'] = np.nan

        results.append(result_row)

    return pd.DataFrame(results)


def compare_peak_to_fullepoch(
    peaks_df: pd.DataFrame,
    fullepoch_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare temporal peaks to full-epoch RDM results.

    Args:
        peaks_df: Temporal peak statistics
        fullepoch_data: Full-epoch RDM statistics (e.g., from rsa_matrix_v1)

    Returns:
        DataFrame with comparison statistics
    """
    print(f"[peaks] Comparing temporal peaks to full-epoch RDM...")

    results = []

    for _, peak_row in peaks_df.iterrows():
        class_a = peak_row['ClassA']
        class_b = peak_row['ClassB']

        # Find matching full-epoch result
        fullepoch_row = fullepoch_data[
            (fullepoch_data['ClassA'] == class_a) &
            (fullepoch_data['ClassB'] == class_b)
        ]

        if len(fullepoch_row) == 0:
            # No matching full-epoch result
            continue

        fullepoch_acc = fullepoch_row.iloc[0]['Mean_Accuracy']
        peak_acc = peak_row['Peak_Accuracy']

        results.append({
            'ClassA': int(class_a),
            'ClassB': int(class_b),
            'Ratio': peak_row['Ratio'],
            'Temporal_Peak_Accuracy': peak_acc,
            'FullEpoch_Accuracy': fullepoch_acc,
            'Difference': peak_acc - fullepoch_acc,
            'Peak_Time': peak_row['Peak_Time']
        })

    return pd.DataFrame(results)


def plot_peak_timing_vs_ratio(
    peaks_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plot peak timing vs numerical ratio.

    Args:
        peaks_df: Peak statistics DataFrame
        output_path: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Peak timing vs ratio
    ax1.errorbar(
        peaks_df['Ratio'],
        peaks_df['Peak_Time'],
        yerr=[
            peaks_df['Peak_Time'] - peaks_df['Peak_Time_CI_Lower'],
            peaks_df['Peak_Time_CI_Upper'] - peaks_df['Peak_Time']
        ],
        fmt='o',
        markersize=8,
        capsize=5,
        color='#2E86AB',
        ecolor='#2E86AB',
        alpha=0.7
    )

    ax1.set_xlabel('Numerical Ratio', fontsize=11)
    ax1.set_ylabel('Peak Timing (ms)', fontsize=11)
    ax1.set_title('Peak Timing vs Numerical Ratio', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(1.0, 6.5)
    ax1.set_ylim(0, 500)

    # Plot 2: Peak accuracy vs ratio
    ax2.errorbar(
        peaks_df['Ratio'],
        peaks_df['Peak_Accuracy'],
        yerr=[
            peaks_df['Peak_Accuracy'] - peaks_df['Peak_Accuracy_CI_Lower'],
            peaks_df['Peak_Accuracy_CI_Upper'] - peaks_df['Peak_Accuracy']
        ],
        fmt='o',
        markersize=8,
        capsize=5,
        color='#27AE60',
        ecolor='#27AE60',
        alpha=0.7
    )

    ax2.axhline(50, color='black', linestyle='--', linewidth=1, label='Chance')
    ax2.set_xlabel('Numerical Ratio', fontsize=11)
    ax2.set_ylabel('Peak Accuracy (%)', fontsize=11)
    ax2.set_title('Peak Accuracy vs Numerical Ratio', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(loc='best')
    ax2.set_xlim(1.0, 6.5)
    ax2.set_ylim(45, 85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[peaks] Saved: {output_path}")
    plt.close()


def plot_onset_latencies_boxplot(
    peaks_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plot onset latencies as boxplot grouped by difficulty.

    Args:
        peaks_df: Peak statistics DataFrame (with onset latencies)
        output_path: Output file path
    """
    # Remove pairs with no onset
    onset_data = peaks_df[~peaks_df['Onset_Time'].isna()].copy()

    if len(onset_data) == 0:
        print(f"[peaks] WARNING: No onset latencies found (no significant clusters)")
        return

    # Categorize by difficulty
    def categorize_difficulty(ratio):
        if ratio >= 2.0:
            return 'Easy'
        elif ratio >= 1.5:
            return 'Medium'
        else:
            return 'Hard'

    onset_data['Difficulty'] = onset_data['Ratio'].apply(categorize_difficulty)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot
    difficulty_order = ['Easy', 'Medium', 'Hard']
    data_to_plot = [
        onset_data[onset_data['Difficulty'] == diff]['Onset_Time'].values
        for diff in difficulty_order
    ]

    bp = ax.boxplot(
        data_to_plot,
        labels=difficulty_order,
        patch_artist=True,
        notch=True,
        showmeans=True
    )

    # Color boxes
    colors = ['#27AE60', '#F39C12', '#E74C3C']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, diff in enumerate(difficulty_order):
        diff_data = onset_data[onset_data['Difficulty'] == diff]['Onset_Time'].values
        x = np.random.normal(i + 1, 0.04, size=len(diff_data))
        ax.scatter(x, diff_data, alpha=0.5, color='black', s=30, zorder=10)

    ax.set_xlabel('Pair Difficulty', fontsize=11)
    ax.set_ylabel('Onset Latency (ms)', fontsize=11)
    ax.set_title('Discrimination Onset Latencies by Difficulty', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax.set_ylim(0, 500)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[peaks] Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze temporal peaks and onset latencies"
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
        help="Statistical results with FDR corrections (for onset latencies)"
    )
    parser.add_argument(
        "--fullepoch-data",
        type=Path,
        required=False,
        help="Full-epoch RDM statistics (for comparison)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for CIs (default: 10000)"
    )

    args = parser.parse_args()

    # Create output directories
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  TEMPORAL RSA: PEAK & ONSET ANALYSIS")
    print("="*70)

    # Load data
    print(f"[peaks] Loading subject data...")
    subject_df = pd.read_csv(args.subject_data)

    stats_df = None
    if args.stats_data and args.stats_data.exists():
        print(f"[peaks] Loading statistical results...")
        stats_df = pd.read_csv(args.stats_data)

    fullepoch_df = None
    if args.fullepoch_data and args.fullepoch_data.exists():
        print(f"[peaks] Loading full-epoch RDM data...")
        fullepoch_df = pd.read_csv(args.fullepoch_data)

    # Analyze peaks and onsets
    peaks_df = analyze_all_pairs_peaks(
        subject_data=subject_df,
        stats_data=stats_df,
        n_bootstrap=args.n_bootstrap
    )

    # Save peaks summary
    output_peaks = tables_dir / "temporal_peaks_summary.csv"
    peaks_df.to_csv(output_peaks, index=False)
    print(f"[peaks] Peaks summary saved: {output_peaks}")

    # Save onset latencies (separate table)
    onset_df = peaks_df[~peaks_df['Onset_Time'].isna()][
        ['ClassA', 'ClassB', 'Ratio', 'Onset_Time', 'Onset_Accuracy']
    ].copy()

    if len(onset_df) > 0:
        output_onsets = tables_dir / "temporal_onset_latencies.csv"
        onset_df.to_csv(output_onsets, index=False)
        print(f"[peaks] Onset latencies saved: {output_onsets}")
    else:
        print(f"[peaks] WARNING: No onset latencies found (no significant clusters)")

    # Compare to full-epoch RDM
    if fullepoch_df is not None:
        comparison_df = compare_peak_to_fullepoch(
            peaks_df=peaks_df,
            fullepoch_data=fullepoch_df
        )

        output_comparison = tables_dir / "temporal_peaks_vs_fullepoch.csv"
        comparison_df.to_csv(output_comparison, index=False)
        print(f"[peaks] Comparison to full-epoch saved: {output_comparison}")

        # Summary statistics
        print(f"\n[peaks] Peak vs Full-Epoch Summary:")
        print(f"  Mean difference: {comparison_df['Difference'].mean():.2f}%")
        print(f"  Median difference: {comparison_df['Difference'].median():.2f}%")
        print(f"  Range: [{comparison_df['Difference'].min():.2f}, {comparison_df['Difference'].max():.2f}]%")

    # Create figures
    print(f"\n[peaks] Creating figures...")

    # Peak timing vs ratio
    output_peak_fig = figures_dir / "peak_timing_vs_ratio.png"
    plot_peak_timing_vs_ratio(
        peaks_df=peaks_df,
        output_path=output_peak_fig
    )

    # Onset latencies boxplot
    output_onset_fig = figures_dir / "onset_latencies_boxplot.png"
    plot_onset_latencies_boxplot(
        peaks_df=peaks_df,
        output_path=output_onset_fig
    )

    print("="*70)
    print("  [OK] COMPLETE: Peak and onset analysis finished")
    print("="*70)


if __name__ == "__main__":
    main()
