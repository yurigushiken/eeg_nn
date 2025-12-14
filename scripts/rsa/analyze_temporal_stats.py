"""
Statistical analysis of temporal RSA with FDR correction.

This script implements rigorous statistical testing following RA recommendations:
1. Subjects as unit of inference (N=24)
2. FDR correction across all 345 pair×window tests
3. Cluster correction (≥2 consecutive significant windows)
4. Fixed null at 50% (theoretical chance)
5. Effect sizes (Cohen's d) and confidence intervals

Output:
- temporal_stats_all_tests.csv (345 rows with all statistics)
- fdr_correction_log.txt (correction details)
- cluster_analysis.csv (significant time clusters)
- effect_sizes.csv (Cohen's d for all tests)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control
from tqdm import tqdm

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.naming import analysis_id_from_run_root, prefixed_path


def run_one_sample_ttest_per_window(
    subject_accuracies: np.ndarray,
    null_value: float = 0.50
) -> Dict[str, float]:
    """
    Run one-sample t-test against fixed null (50% chance).

    Args:
        subject_accuracies: Array of N=24 subject accuracies (as decimals, e.g., 0.60 for 60%)
        null_value: Theoretical chance level (default: 0.50 for binary task)

    Returns:
        Dictionary with t_statistic, p_value, df
    """
    t_stat, p_value = stats.ttest_1samp(subject_accuracies, popmean=null_value)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'df': len(subject_accuracies) - 1
    }


def apply_fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'bh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply FDR correction (Benjamini-Hochberg) across all tests.

    Args:
        p_values: Array of p-values (length 345 for 15 pairs × 23 windows)
        alpha: FDR level (default: 0.05)
        method: 'bh' for Benjamini-Hochberg (scipy method name)

    Returns:
        (p_corrected, rejected)
        - p_corrected: FDR-adjusted p-values
        - rejected: Boolean array (True = reject null at FDR level)
    """
    # Use scipy's false_discovery_control (requires scipy >= 1.11)
    # Returns FDR-adjusted p-values directly - no manual calculation needed
    p_corrected = false_discovery_control(p_values, method=method)
    rejected = p_corrected < alpha

    return p_corrected, rejected


def find_significant_clusters(
    p_values: np.ndarray,
    alpha: float = 0.05,
    min_cluster_size: int = 2
) -> List[List[int]]:
    """
    Find clusters of ≥N consecutive significant windows.

    Args:
        p_values: Array of p-values (one per time window)
        alpha: Significance threshold
        min_cluster_size: Minimum consecutive windows to count as cluster

    Returns:
        List of clusters, each cluster is list of window indices
    """
    sig_windows = p_values < alpha
    clusters = []
    current_cluster = []

    for i, is_sig in enumerate(sig_windows):
        if is_sig:
            current_cluster.append(i)
        else:
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster.copy())
            current_cluster = []

    # Don't forget last cluster
    if len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)

    return clusters


def compute_effect_sizes(
    subject_accuracies: np.ndarray,
    null_value: float = 0.50
) -> Dict[str, float]:
    """
    Compute Cohen's d for one-sample test.

    Args:
        subject_accuracies: Array of subject accuracies (as decimals)
        null_value: Null hypothesis value

    Returns:
        Dictionary with cohens_d
    """
    mean_acc = np.mean(subject_accuracies)
    std_acc = np.std(subject_accuracies, ddof=1)  # Sample SD

    cohens_d = (mean_acc - null_value) / std_acc

    return {'cohens_d': float(cohens_d)}


def compute_confidence_intervals(
    subject_accuracies: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute confidence interval for mean accuracy.

    Args:
        subject_accuracies: Array of subject accuracies (as decimals)
        confidence_level: Confidence level (default: 0.95)

    Returns:
        Dictionary with mean, sem, ci_lower, ci_upper
    """
    mean_acc = np.mean(subject_accuracies)
    sem = stats.sem(subject_accuracies)
    df = len(subject_accuracies) - 1

    ci_lower, ci_upper = stats.t.interval(
        confidence_level,
        df=df,
        loc=mean_acc,
        scale=sem
    )

    return {
        'mean': float(mean_acc),
        'sem': float(sem),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }


def analyze_all_pair_window_combinations(
    subject_data: pd.DataFrame,
    baseline: float = 50.0,
    fdr_alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run statistical tests for all 345 pair×window combinations.

    Args:
        subject_data: DataFrame with subject-level accuracies (seeds averaged)
        baseline: Null hypothesis baseline (50% for binary)
        fdr_alpha: FDR significance level

    Returns:
        DataFrame with 345 rows, one per pair×window combination
    """
    print(f"[stats] Running tests for all pair×window combinations...")

    results = []
    all_p_values = []

    # Get all unique pair×window combinations
    combinations = subject_data[['ClassA', 'ClassB', 'TimeWindow_Start', 'TimeWindow_End', 'TimeWindow_Center']].drop_duplicates()

    for idx, row in tqdm(combinations.iterrows(), total=len(combinations), desc="Statistical tests"):
        class_a = row['ClassA']
        class_b = row['ClassB']
        t_start = row['TimeWindow_Start']
        t_end = row['TimeWindow_End']
        t_center = row['TimeWindow_Center']

        # Filter to this pair×window
        subset = subject_data[
            (subject_data['ClassA'] == class_a) &
            (subject_data['ClassB'] == class_b) &
            (subject_data['TimeWindow_Start'] == t_start)
        ]

        # Get subject accuracies (as decimals for stats)
        subject_accs = subset['Accuracy'].values / 100.0  # Convert percentage to decimal
        n_subjects = len(subject_accs)

        # Skip if insufficient data
        if n_subjects < 3:
            print(f"[stats] WARNING: Only {n_subjects} subjects for {class_a}v{class_b} t={t_start}, skipping")
            continue

        # T-test against 50%
        ttest_result = run_one_sample_ttest_per_window(subject_accs, null_value=baseline/100.0)

        # Effect size
        effect_size = compute_effect_sizes(subject_accs, null_value=baseline/100.0)

        # Confidence intervals
        ci_result = compute_confidence_intervals(subject_accs)

        # Combine results
        result_row = {
            'ClassA': int(class_a),
            'ClassB': int(class_b),
            'TimeWindow_Start': int(t_start),
            'TimeWindow_End': int(t_end),
            'TimeWindow_Center': float(t_center),
            'N_Subjects': n_subjects,
            'Mean_Accuracy': ci_result['mean'] * 100.0,  # Back to percentage
            'SD': float(np.std(subject_accs, ddof=1) * 100.0),
            'SEM': ci_result['sem'] * 100.0,
            'CI_Lower': ci_result['ci_lower'] * 100.0,
            'CI_Upper': ci_result['ci_upper'] * 100.0,
            't_statistic': ttest_result['t_statistic'],
            'df': ttest_result['df'],
            'p_value': ttest_result['p_value'],
            'cohens_d': effect_size['cohens_d'],
            'baseline': baseline
        }

        results.append(result_row)
        all_p_values.append(ttest_result['p_value'])

    # Create DataFrame
    df = pd.DataFrame(results)

    # Apply FDR correction across ALL tests
    print(f"[stats] Applying FDR correction across {len(all_p_values)} tests...")
    p_corrected, rejected = apply_fdr_correction(np.array(all_p_values), alpha=fdr_alpha)

    df['p_value_fdr'] = p_corrected
    df['significant_fdr'] = rejected

    # Count significant results
    n_sig_raw = np.sum(df['p_value'] < fdr_alpha)
    n_sig_fdr = np.sum(rejected)

    print(f"[stats] Significant (raw p<{fdr_alpha}):     {n_sig_raw}/{len(df)}")
    print(f"[stats] Significant (FDR q<{fdr_alpha}):     {n_sig_fdr}/{len(df)}")

    return df


def analyze_clusters_per_pair(
    stats_df: pd.DataFrame,
    min_cluster_size: int = 2,
    use_fdr: bool = True
) -> pd.DataFrame:
    """
    Find significant time clusters for each pair.

    Args:
        stats_df: Statistical results from analyze_all_pair_window_combinations
        min_cluster_size: Minimum consecutive significant windows
        use_fdr: Use FDR-corrected p-values (recommended)

    Returns:
        DataFrame with cluster information per pair
    """
    print(f"[stats] Finding significant clusters (min size={min_cluster_size})...")

    cluster_results = []

    for (class_a, class_b), group in stats_df.groupby(['ClassA', 'ClassB']):
        # Sort by time
        group = group.sort_values('TimeWindow_Center')

        # Get p-values
        if use_fdr:
            p_values = group['p_value_fdr'].values
        else:
            p_values = group['p_value'].values

        # Find clusters
        clusters = find_significant_clusters(
            p_values=p_values,
            alpha=0.05,
            min_cluster_size=min_cluster_size
        )

        if len(clusters) > 0:
            for cluster_idx, cluster_windows in enumerate(clusters):
                # Get time range of cluster
                cluster_times = group.iloc[cluster_windows]['TimeWindow_Center'].values
                cluster_accs = group.iloc[cluster_windows]['Mean_Accuracy'].values

                cluster_results.append({
                    'ClassA': int(class_a),
                    'ClassB': int(class_b),
                    'Cluster_ID': cluster_idx + 1,
                    'N_Windows': len(cluster_windows),
                    'Start_Time': float(cluster_times[0]),
                    'End_Time': float(cluster_times[-1]),
                    'Duration_ms': float(cluster_times[-1] - cluster_times[0] + 50),  # +50 for window width
                    'Mean_Accuracy': float(np.mean(cluster_accs)),
                    'Max_Accuracy': float(np.max(cluster_accs))
                })

    return pd.DataFrame(cluster_results)


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of temporal RSA with FDR correction"
    )
    parser.add_argument(
        "--subject-data",
        type=Path,
        required=True,
        help="Subject-level temporal data (seeds averaged)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=50.0,
        help="Baseline chance level (default: 50%%)"
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="FDR significance level (default: 0.05)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum consecutive significant windows for cluster (default: 2)"
    )

    args = parser.parse_args()

    run_root = args.output_dir
    run_root.mkdir(parents=True, exist_ok=True)
    _ = analysis_id_from_run_root(run_root)  # validate/sanitize early

    print("="*70)
    print("  TEMPORAL RSA: STATISTICAL ANALYSIS WITH FDR CORRECTION")
    print("="*70)
    print(f"Subject data:  {args.subject_data}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Baseline:      {args.baseline}%")
    print(f"FDR alpha:     {args.fdr_alpha}")
    print(f"Min cluster:   {args.min_cluster_size} windows")
    print("="*70)

    # Load subject data
    print(f"\n[stats] Loading subject data...")
    subject_df = pd.read_csv(args.subject_data)
    print(f"[stats] Loaded {len(subject_df)} rows (subject×pair×window)")

    # Run statistical tests
    stats_df = analyze_all_pair_window_combinations(
        subject_data=subject_df,
        baseline=args.baseline,
        fdr_alpha=args.fdr_alpha
    )

    # Save full statistical results
    stats_output = prefixed_path(run_root=run_root, kind="tables", stem="temporal_stats_all_tests", ext=".csv")
    stats_df.to_csv(stats_output, index=False)
    print(f"\n[stats] Statistical results saved: {stats_output}")

    # Analyze clusters
    cluster_df = analyze_clusters_per_pair(
        stats_df=stats_df,
        min_cluster_size=args.min_cluster_size,
        use_fdr=True
    )

    if len(cluster_df) > 0:
        cluster_output = prefixed_path(run_root=run_root, kind="stats", stem="cluster_analysis", ext=".csv")
        cluster_df.to_csv(cluster_output, index=False)
        print(f"[stats] Cluster analysis saved: {cluster_output}")
        print(f"[stats] Found {len(cluster_df)} significant clusters across pairs")
    else:
        print(f"[stats] No significant clusters found (try lowering min_cluster_size)")

    # Save FDR correction log
    fdr_log = prefixed_path(run_root=run_root, kind="stats", stem="fdr_correction_log", ext=".txt")
    with open(fdr_log, 'w') as f:
        f.write("FDR CORRECTION LOG\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total tests:               {len(stats_df)}\n")
        f.write(f"FDR alpha:                 {args.fdr_alpha}\n")
        f.write(f"Baseline (null):           {args.baseline}%\n\n")
        f.write(f"Significant (raw p<0.05):  {np.sum(stats_df['p_value'] < 0.05)}\n")
        f.write(f"Significant (FDR q<0.05):  {np.sum(stats_df['significant_fdr'])}\n\n")

        # Per-pair summary
        f.write("PER-PAIR SUMMARY:\n")
        f.write("-"*70 + "\n")
        for (class_a, class_b), group in stats_df.groupby(['ClassA', 'ClassB']):
            n_sig = np.sum(group['significant_fdr'])
            n_windows = len(group)
            f.write(f"Pair {class_a}v{class_b}: {n_sig}/{n_windows} windows significant\n")

    print(f"[stats] FDR correction log saved: {fdr_log}")

    # Save effect sizes
    effect_sizes_output = prefixed_path(run_root=run_root, kind="stats", stem="effect_sizes", ext=".csv")
    stats_df[['ClassA', 'ClassB', 'TimeWindow_Center', 'cohens_d', 'Mean_Accuracy', 'significant_fdr']].to_csv(
        effect_sizes_output, index=False
    )
    print(f"[stats] Effect sizes saved: {effect_sizes_output}")

    print("\n" + "="*70)
    print("  [OK] COMPLETE: Statistical analysis finished")
    print("="*70)
    print(f"\nNext step: Visualize temporal curves")
    print(f"  python scripts/rsa/visualize_temporal_curves.py \\")
    print(f"    --subject-data {args.subject_data} \\")
    print(f"    --stats-data {stats_output} \\")
    print(f"    --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
