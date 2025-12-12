"""
Master orchestration script for complete temporal RSA analysis pipeline.

This script runs all analysis modules in sequence:
1. Extract subject-level accuracies from predictions
2. Run statistical tests with FDR correction
3. Generate temporal curve visualizations
4. Generate RDM evolution visualizations
5. Analyze peaks and onset latencies

Usage:
    python scripts/rsa/run_temporal_analysis_pipeline.py \
        --runs-dir results/runs/rsa_temporal_v1 \
        --output-dir results/runs/rsa_temporal_v1 \
        --fullepoch-comparison results/runs/rsa_matrix_v1/stats_summary.csv
"""

from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def run_command(cmd: list[str], description: str) -> bool:
    """
    Run a subprocess command and handle errors.

    Args:
        cmd: Command as list of strings
        description: Description of the command

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n[OK] {description} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed")
        print(f"Exit code: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete temporal RSA analysis pipeline"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing temporal run folders (e.g., results/runs/rsa_temporal_v1)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for all results (default: same as runs-dir)"
    )
    parser.add_argument(
        "--fullepoch-comparison",
        type=Path,
        required=False,
        help="Full-epoch RDM statistics for comparison (e.g., results/runs/rsa_matrix_v1/stats_summary.csv)"
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
    parser.add_argument(
        "--create-gif",
        action="store_true",
        help="Create animated GIF for RDM evolution (slower, ~1-2 minutes)"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for peak CIs (default: 10000)"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip data extraction (if subject CSVs already exist)"
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip statistical analysis (if stats CSVs already exist)"
    )
    parser.add_argument(
        "--skip-model-fits",
        action="store_true",
        help="Skip temporal model-fit step (model–brain RDM correlation over time)."
    )
    parser.add_argument(
        "--rt-summary-csv",
        type=Path,
        default=None,
        help="Optional RT subject-level summary CSV to include an RT landing model in the model-fit time series.",
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  TEMPORAL RSA ANALYSIS PIPELINE")
    print("="*70)
    print(f"\nRuns directory:       {args.runs_dir}")
    print(f"Output directory:     {args.output_dir}")
    print(f"Baseline:             {args.baseline}%")
    print(f"FDR alpha:            {args.fdr_alpha}")
    print(f"Min cluster size:     {args.min_cluster_size}")
    print(f"Bootstrap samples:    {args.n_bootstrap}")
    print(f"Create GIF:           {args.create_gif}")
    print("="*70)

    # Check runs directory exists
    if not args.runs_dir.exists():
        print(f"\n[ERROR] Runs directory not found: {args.runs_dir}")
        print("Make sure temporal training has completed.")
        return 1

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "stats").mkdir(exist_ok=True)

    # Define intermediate file paths
    subject_raw_csv = args.output_dir / "subject_temporal_accuracies.csv"
    subject_agg_csv = args.output_dir / "subject_temporal_means.csv"
    stats_csv = args.output_dir / "tables" / "temporal_stats_all_tests.csv"

    # Track which steps succeeded
    all_success = True

    # ========================================================================
    # STEP 1: Extract subject-level accuracies
    # ========================================================================
    if not args.skip_extraction:
        cmd_extract = [
            sys.executable,
            str(PROJ_ROOT / "scripts" / "rsa" / "extract_subject_temporal_accuracies.py"),
            "--runs-dir", str(args.runs_dir),
            "--output", str(subject_raw_csv),
            "--output-aggregated", str(subject_agg_csv)
        ]

        success = run_command(
            cmd=cmd_extract,
            description="STEP 1/5: Extract subject-level accuracies"
        )
        all_success = all_success and success

        if not success:
            print("\n[ERROR] PIPELINE FAILED at Step 1")
            return 1
    else:
        print(f"\n[SKIP] STEP 1/5: Skipping data extraction (using existing CSVs)")

    # ========================================================================
    # STEP 2: Statistical analysis with FDR correction
    # ========================================================================
    if not args.skip_stats:
        cmd_stats = [
            sys.executable,
            str(PROJ_ROOT / "scripts" / "rsa" / "analyze_temporal_stats.py"),
            "--subject-data", str(subject_agg_csv),
            "--output-dir", str(args.output_dir),
            "--baseline", str(args.baseline),
            "--fdr-alpha", str(args.fdr_alpha),
            "--min-cluster-size", str(args.min_cluster_size)
        ]

        success = run_command(
            cmd=cmd_stats,
            description="STEP 2/5: Statistical analysis with FDR correction"
        )
        all_success = all_success and success

        if not success:
            print("\n[ERROR] PIPELINE FAILED at Step 2")
            return 1
    else:
        print(f"\n[SKIP] STEP 2/5: Skipping statistical analysis (using existing stats)")

    # ========================================================================
    # STEP 3: Visualize temporal curves
    # ========================================================================
    cmd_curves = [
        sys.executable,
        str(PROJ_ROOT / "scripts" / "rsa" / "visualize_temporal_curves.py"),
        "--subject-data", str(subject_agg_csv),
        "--stats-data", str(stats_csv),
        "--output-dir", str(args.output_dir / "figures")
    ]

    success = run_command(
        cmd=cmd_curves,
        description="STEP 3/5: Generate temporal curve visualizations"
    )
    all_success = all_success and success

    if not success:
        print("\n[WARN] WARNING: Temporal curves failed, continuing...")

    # ========================================================================
    # STEP 4: Visualize RDM evolution
    # ========================================================================
    cmd_rdm = [
        sys.executable,
        str(PROJ_ROOT / "scripts" / "rsa" / "visualize_temporal_rdm_evolution.py"),
        "--subject-data", str(subject_agg_csv),
        "--output-dir", str(args.output_dir / "figures")
    ]

    if args.create_gif:
        cmd_rdm.append("--create-gif")

    success = run_command(
        cmd=cmd_rdm,
        description="STEP 4/5: Generate RDM evolution visualizations"
    )
    all_success = all_success and success

    if not success:
        print("\n[WARN] WARNING: RDM evolution failed, continuing...")

    # ========================================================================
    # STEP 5: Model fits (Figure-12C style)
    # ========================================================================
    if not args.skip_model_fits:
        cmd_model_fits = [
            sys.executable,
            str(PROJ_ROOT / "scripts" / "rsa" / "analyze_temporal_model_fits.py"),
            "--subject-data", str(subject_agg_csv),
            "--output-dir", str(args.output_dir),
        ]
        if args.rt_summary_csv:
            cmd_model_fits.extend(["--rt-summary-csv", str(args.rt_summary_csv)])

        success = run_command(
            cmd=cmd_model_fits,
            description="STEP 5/6: Temporal model fits (Spearman + noise ceiling)"
        )
        all_success = all_success and success

        if not success:
            print("\n[WARN] WARNING: Model fits failed, continuing...")
    else:
        print(f"\n[SKIP] STEP 5/6: Skipping model fits")

    # ========================================================================
    # STEP 6/6: Analyze peaks and onset latencies
    # ========================================================================
    cmd_peaks = [
        sys.executable,
        str(PROJ_ROOT / "scripts" / "rsa" / "analyze_temporal_peaks.py"),
        "--subject-data", str(subject_agg_csv),
        "--stats-data", str(stats_csv),
        "--output-dir", str(args.output_dir),
        "--n-bootstrap", str(args.n_bootstrap)
    ]

    if args.fullepoch_comparison and args.fullepoch_comparison.exists():
        cmd_peaks.extend(["--fullepoch-data", str(args.fullepoch_comparison)])

    success = run_command(
        cmd=cmd_peaks,
        description="STEP 5/5: Analyze peaks and onset latencies"
    )
    all_success = all_success and success

    if not success:
        print("\n[WARN] WARNING: Peak analysis failed, continuing...")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    if all_success:
        print("  [OK] PIPELINE COMPLETE: All steps succeeded")
    else:
        print("  [WARN] PIPELINE COMPLETE: Some steps had warnings/errors")
    print("="*70)

    print("\nGenerated outputs:")
    print(f"  Subject data:          {subject_agg_csv}")
    print(f"  Statistical results:   {stats_csv}")
    print(f"  Figures:               {args.output_dir / 'figures'}")
    print(f"  Tables:                {args.output_dir / 'tables'}")
    print(f"  Stats logs:            {args.output_dir / 'stats'}")

    print("\nKey results to check:")
    print(f"  1. Temporal curves:    {args.output_dir / 'figures' / 'temporal_curves_all_pairs.png'}")
    print(f"  2. RDM snapshots:      {args.output_dir / 'figures' / 'temporal_rdm_snapshots.png'}")
    print(f"  3. Peak summary:       {args.output_dir / 'tables' / 'temporal_peaks_summary.csv'}")
    print(f"  4. FDR correction:     {args.output_dir / 'stats' / 'fdr_correction_log.txt'}")

    if args.create_gif:
        print(f"  5. RDM evolution GIF:  {args.output_dir / 'figures' / 'temporal_rdm_evolution.gif'}")

    print("\n" + "="*70)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
