"""
Static RSA matrix analysis pipeline.

Given a run directory produced by scripts/rsa/run_rsa_matrix.py, this script generates:
- rsa_results_master.csv (subject-level rows, derived from predictions CSVs)
- stats_summary.csv (subject-level t-tests vs baseline, Holm-corrected)
- figures: brain RDM heatmap + MDS scatter (seed-averaged)
- tables: a simple all-pairs CSV (sorted)
- confounds/models (optional): pixel-control RSA model comparison plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
code_root = PROJ_ROOT / "code"
for p in (PROJ_ROOT, code_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.rsa.compile_rsa_results import MASTER_FILENAME, compile_results_dataframe, write_master_csv
from scripts.rsa.analyze_rsa_stats import compute_subject_ttests, filter_subject_rows, load_master_csv
from scripts.rsa.naming import analysis_id_from_run_root, prefixed_path, prefixed_title
from scripts.rsa.visualize_rsa import build_accuracy_matrix, compute_mds_positions, plot_mds_scatter, plot_rdm_heatmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the static RSA matrix analysis pipeline.")
    p.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Run directory produced by run_rsa_matrix.py (contains many *_rsa_* subfolders).",
    )
    p.add_argument(
        "--baseline",
        type=float,
        default=50.0,
        help="Theoretical baseline accuracy for t-tests (default: 50).",
    )
    p.add_argument(
        "--expected-subjects",
        type=int,
        default=None,
        help="Optional QC: expected number of subjects.",
    )
    p.add_argument(
        "--viz-metric",
        default="Accuracy",
        help="Metric column to visualize in the matrix (default: Accuracy). Options include Accuracy, MacroF1, MinClassF1.",
    )
    p.add_argument(
        "--run-confounds",
        action="store_true",
        help="Also run confounds analysis (writes models/ and confounds/). Requires --stimuli-csv.",
    )
    p.add_argument(
        "--stimuli-csv",
        type=Path,
        default=None,
        help="Stimuli CSV for pixel model (e-only target), e.g. data/stimuli/stimuli_analysis.csv.",
    )
    return p.parse_args()


def write_all_pairs_table(stats_summary_csv: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(stats_summary_csv)
    cols = [c for c in ["ClassA", "ClassB", "n_subjects", "mean_accuracy", "std_accuracy", "p_value_holm", "p_value"] if c in df.columns]
    df2 = df[cols].copy() if cols else df.copy()
    df2.sort_values(["p_value_holm", "mean_accuracy"], ascending=[True, False], inplace=True, na_position="last")
    out_path = out_dir / "all_pairs_sorted.csv"
    df2.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    args = parse_args()
    run_root = args.runs_dir
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    # Ensure expected output dirs exist (matches other run roots)
    (run_root / "figures").mkdir(exist_ok=True)
    (run_root / "tables").mkdir(exist_ok=True)

    analysis_id = analysis_id_from_run_root(run_root)

    # 1) Compile master CSV (subject rows)
    master_path = run_root / MASTER_FILENAME
    df_master = compile_results_dataframe(run_root)
    write_master_csv(df_master, master_path)
    print(f"[rsa-matrix-pipeline] Master CSV: {master_path.resolve()}")

    # 2) Subject-level stats
    subject_df = filter_subject_rows(load_master_csv(master_path))
    stats_df = compute_subject_ttests(
        subject_df,
        baseline=float(args.baseline) if args.baseline is not None else None,
        expected_n_subjects=args.expected_subjects,
    )
    stats_path = run_root / "stats_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[rsa-matrix-pipeline] Stats summary: {stats_path.resolve()}")

    # 3) Matrix visualization (seed-averaged)
    results_csv = run_root / "rsa_matrix_results.csv"
    matrix, labels = build_accuracy_matrix(results_csv, metric=str(args.viz_metric), subject_filter="OVERALL")
    positions = compute_mds_positions(matrix, labels)

    # Keep historical filenames when using default Accuracy
    metric_tag = str(args.viz_metric)
    if metric_tag.lower().startswith("acc"):
        heatmap_path = prefixed_path(run_root=run_root, kind="figures", stem="brain_rdm_heatmap", ext=".png")
        scatter_path = prefixed_path(run_root=run_root, kind="figures", stem="mds", ext=".png")
        title = prefixed_title(run_root=run_root, title="RSA Matrix (Higher = Easier to Distinguish)")
    else:
        heatmap_path = prefixed_path(run_root=run_root, kind="figures", stem=f"brain_rdm_heatmap_{metric_tag}", ext=".png")
        scatter_path = prefixed_path(run_root=run_root, kind="figures", stem=f"mds_{metric_tag}", ext=".png")
        title = prefixed_title(run_root=run_root, title=f"RSA Matrix ({metric_tag})")

    plot_rdm_heatmap(matrix, labels, heatmap_path, title=title)
    plot_mds_scatter(
        positions,
        scatter_path,
        flip_xy=True,
        title=prefixed_title(run_root=run_root, title=f"MDS Projection of RSA Matrix ({metric_tag})"),
    )
    print(f"[rsa-matrix-pipeline] Heatmap: {heatmap_path.resolve()}")
    print(f"[rsa-matrix-pipeline] MDS:     {scatter_path.resolve()}")

    # 4) Tables (lightweight, works for change-condition codes too)
    table_path = write_all_pairs_table(stats_path, run_root / "tables")
    print(f"[rsa-matrix-pipeline] All-pairs table: {table_path.resolve()}")

    # 5) Optional confounds/models
    if args.run_confounds:
        if args.stimuli_csv is None:
            raise ValueError("--run-confounds requires --stimuli-csv")
        from scripts.rsa.analyze_rsa_confounds import run_analysis as run_confounds

        confounds_dir = run_root / "confounds"
        confounds_dir.mkdir(exist_ok=True)
        run_confounds(
            master_csv=master_path,
            stimuli_csv=args.stimuli_csv,
            output_dir=confounds_dir,
            codes=sorted(set(subject_df["ClassA"].astype(int)).union(set(subject_df["ClassB"].astype(int)))),
            baseline=0.4,
            rt_summary_csv=None,
        )
        print(f"[rsa-matrix-pipeline] Confounds written under: {confounds_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


