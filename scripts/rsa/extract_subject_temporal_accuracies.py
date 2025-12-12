"""
Extract per-subject temporal accuracies from temporal RSA runs.

This script processes test_predictions_outer.csv files from all temporal runs
to create subject-level accuracy datasets.

Statistical Principle (RA Recommendation #1):
- Subjects are the unit of inference (N=24), NOT seeds
- Seeds are technical replicates that reduce within-subject noise
- We average across seeds BEFORE statistical testing

Outputs:
1. subject_temporal_accuracies.csv (raw: subjects × seeds × pairs × windows)
2. subject_temporal_means.csv (aggregated: subjects × pairs × windows, seeds averaged)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def parse_run_directory_name(run_name: str) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Parse temporal run directory name to extract metadata.

    Format: YYYYMMDD_HHMMSS_rsa_AAvBB_seed_SS_tXXX-YYYms
    Example: 20251207_070258_rsa_11v22_seed_49_t180-230ms

    Returns:
        (class_a, class_b, seed, time_start, time_end) or None if invalid
    """
    try:
        parts = run_name.split('_')

        # Find rsa_AAvBB part (may be "rsa" with "11v22" as next part, or "rsa11v22" as one part)
        rsa_parts = [i for i, p in enumerate(parts) if p.startswith('rsa') or p == 'rsa']
        if not rsa_parts:
            return None

        rsa_idx = rsa_parts[0]
        rsa_part = parts[rsa_idx]

        if rsa_part == 'rsa':
            # Format: ...rsa_11v22_... (rsa and pair are separate)
            pair_str = parts[rsa_idx + 1]
        else:
            # Format: ...rsa11v22_... or ...rsa_11v22_... (combined)
            pair_str = rsa_part.replace('rsa_', '').replace('rsa', '')

        if 'v' not in pair_str:
            return None
        class_a, class_b = map(int, pair_str.split('v'))

        # Find seed
        seed_parts = [i for i, p in enumerate(parts) if p == 'seed']
        if not seed_parts:
            return None
        seed_idx = seed_parts[0]
        seed = int(parts[seed_idx + 1])

        # Find time window (last part: tXXX-YYYms)
        time_str = parts[-1].replace('ms', '').replace('t', '')  # "180-230"
        if '-' not in time_str:
            return None
        time_start, time_end = map(int, time_str.split('-'))

        return (class_a, class_b, seed, time_start, time_end)

    except (IndexError, ValueError) as e:
        return None


def extract_subject_accuracies_from_predictions(
    predictions_csv: Path,
    class_a: int,
    class_b: int,
    seed: int,
    time_window_start: int,
    time_window_end: int
) -> pd.DataFrame:
    """
    Extract per-subject accuracy from a single test_predictions_outer.csv file.

    Args:
        predictions_csv: Path to test_predictions_outer.csv
        class_a: First numerosity code
        class_b: Second numerosity code
        seed: Random seed
        time_window_start: Start time in ms
        time_window_end: End time in ms

    Returns:
        DataFrame with columns: ClassA, ClassB, Subject, Seed, TimeWindow_Start,
                                TimeWindow_End, TimeWindow_Center, Accuracy, N_Trials
    """
    # Load predictions
    df = pd.read_csv(predictions_csv)

    # Group by subject and compute accuracy
    subject_accuracies = []

    for subject_id, group in df.groupby('subject_id'):
        n_trials = len(group)
        n_correct = group['correct'].sum()
        accuracy = (n_correct / n_trials) * 100.0  # Percentage

        subject_accuracies.append({
            'ClassA': class_a,
            'ClassB': class_b,
            'Subject': int(subject_id),
            'Seed': seed,
            'TimeWindow_Start': time_window_start,
            'TimeWindow_End': time_window_end,
            'TimeWindow_Center': (time_window_start + time_window_end) / 2,
            'Accuracy': accuracy,
            'N_Trials': n_trials
        })

    return pd.DataFrame(subject_accuracies)


def create_subject_temporal_dataset(
    runs_dir: Path,
    output_raw: Optional[Path] = None,
    output_aggregated: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create complete subject-level temporal dataset from all runs.

    Args:
        runs_dir: Directory containing temporal run folders
        output_raw: Optional path to save raw dataset (with seeds)
        output_aggregated: Optional path to save aggregated dataset (seeds averaged)

    Returns:
        (raw_df, aggregated_df)
        - raw_df: All subject×seed×pair×window data
        - aggregated_df: Seeds averaged within subjects
    """
    print(f"[extract] Scanning runs directory: {runs_dir}")

    # Find all run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and 'rsa_' in d.name]
    print(f"[extract] Found {len(run_dirs)} run directories")

    all_subject_data = []

    # Process each run
    for run_dir in tqdm(run_dirs, desc="Extracting subject accuracies"):
        # Parse directory name
        metadata = parse_run_directory_name(run_dir.name)
        if metadata is None:
            print(f"[extract] WARNING: Could not parse {run_dir.name}, skipping")
            continue

        class_a, class_b, seed, time_start, time_end = metadata

        # Check if predictions file exists
        pred_csv = run_dir / "test_predictions_outer.csv"
        if not pred_csv.exists():
            print(f"[extract] WARNING: No predictions CSV in {run_dir.name}, skipping")
            continue

        # Extract subject accuracies
        try:
            subject_df = extract_subject_accuracies_from_predictions(
                predictions_csv=pred_csv,
                class_a=class_a,
                class_b=class_b,
                seed=seed,
                time_window_start=time_start,
                time_window_end=time_end
            )
            all_subject_data.append(subject_df)

        except Exception as e:
            print(f"[extract] ERROR processing {run_dir.name}: {e}")
            continue

    # Concatenate all data
    raw_df = pd.concat(all_subject_data, ignore_index=True)
    print(f"[extract] Raw dataset: {len(raw_df)} rows (subject×seed×pair×window)")

    # Aggregate across seeds (CRITICAL: subjects as unit of inference)
    print(f"[extract] Aggregating across seeds (N=24 subjects per pair×window)...")
    aggregated_df = raw_df.groupby(
        ['ClassA', 'ClassB', 'Subject', 'TimeWindow_Start', 'TimeWindow_End', 'TimeWindow_Center']
    ).agg({
        'Accuracy': 'mean',  # Average across seeds
        'N_Trials': 'first'  # Should be same across seeds
    }).reset_index()

    print(f"[extract] Aggregated dataset: {len(aggregated_df)} rows (subject×pair×window)")

    # Save if output paths provided
    if output_raw:
        raw_df.to_csv(output_raw, index=False)
        print(f"[extract] Raw dataset saved: {output_raw}")

    if output_aggregated:
        aggregated_df.to_csv(output_aggregated, index=False)
        print(f"[extract] Aggregated dataset saved: {output_aggregated}")

    return raw_df, aggregated_df


def aggregate_seeds_within_subjects(data: pd.DataFrame) -> pd.DataFrame:
    """
    Average accuracy across seeds for each subject×pair×window.

    This is CRITICAL for proper statistical inference (RA recommendation #1).
    Subjects are biological replicates (N=24), seeds are technical replicates.

    Args:
        data: DataFrame with 'Seed' column

    Returns:
        DataFrame with seeds averaged, 'Seed' column removed
    """
    return data.groupby(
        ['ClassA', 'ClassB', 'Subject', 'TimeWindow_Start', 'TimeWindow_End', 'TimeWindow_Center']
    ).agg({
        'Accuracy': 'mean',
        'N_Trials': 'first'
    }).reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-subject temporal accuracies from RSA runs"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing temporal run folders (e.g., results/runs/rsa_temporal_v1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for raw dataset (default: <runs-dir>/subject_temporal_accuracies.csv)"
    )
    parser.add_argument(
        "--output-aggregated",
        type=Path,
        help="Output path for aggregated dataset (default: <runs-dir>/subject_temporal_means.csv)"
    )

    args = parser.parse_args()

    # Set default output paths if not provided
    if args.output is None:
        args.output = args.runs_dir / "subject_temporal_accuracies.csv"

    if args.output_aggregated is None:
        args.output_aggregated = args.runs_dir / "subject_temporal_means.csv"

    # Create datasets
    print("="*70)
    print("  TEMPORAL RSA: SUBJECT-LEVEL DATA EXTRACTION")
    print("="*70)
    print(f"Runs directory: {args.runs_dir}")
    print(f"Output (raw):   {args.output}")
    print(f"Output (agg):   {args.output_aggregated}")
    print("="*70)

    raw_df, agg_df = create_subject_temporal_dataset(
        runs_dir=args.runs_dir,
        output_raw=args.output,
        output_aggregated=args.output_aggregated
    )

    # Summary statistics
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"Total subjects:      {agg_df['Subject'].nunique()}")
    print(f"Total pairs:         {len(agg_df[['ClassA', 'ClassB']].drop_duplicates())}")
    print(f"Total time windows:  {agg_df['TimeWindow_Start'].nunique()}")
    print(f"Total seeds (raw):   {raw_df['Seed'].nunique()}")
    print(f"\nRaw dataset:         {len(raw_df)} rows")
    print(f"Aggregated dataset:  {len(agg_df)} rows")
    print("="*70)

    # Validation checks
    print("\n" + "="*70)
    print("  VALIDATION")
    print("="*70)

    # Check expected counts
    n_pairs = len(agg_df[['ClassA', 'ClassB']].drop_duplicates())
    n_windows = agg_df['TimeWindow_Start'].nunique()
    n_subjects = agg_df['Subject'].nunique()

    expected_rows = n_pairs * n_windows * n_subjects
    actual_rows = len(agg_df)

    if actual_rows == expected_rows:
        print(f"[OK] Row count correct: {actual_rows} = {n_pairs}x{n_windows}x{n_subjects}")
    else:
        print(f"[WARN] Row count mismatch: {actual_rows} != {expected_rows} (expected)")
        print(f"  Some subjects may be missing in some conditions")

    # Check for missing subjects
    for (class_a, class_b), group in agg_df.groupby(['ClassA', 'ClassB']):
        n_subj_pair = group['Subject'].nunique()
        if n_subj_pair < 24:
            print(f"[WARN] Pair {class_a}v{class_b}: Only {n_subj_pair}/24 subjects")

    print("="*70)
    print("\n[OK] COMPLETE: Subject-level datasets created successfully")
    print(f"\nNext step: Run statistical analysis")
    print(f"  python scripts/rsa/analyze_temporal_stats.py \\")
    print(f"    --subject-data {args.output_aggregated} \\")
    print(f"    --output-dir {args.runs_dir}")


if __name__ == "__main__":
    main()
