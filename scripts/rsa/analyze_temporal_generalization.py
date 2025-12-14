"""
Temporal generalization (train-time × test-time) post-training analysis (CLI).

This file is intentionally kept small; core logic lives in:
`scripts/rsa/temporal_generalization_analysis_core.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.temporal_generalization_analysis_core import (
    average_pairs_within_subject,
    average_seeds_within_subject,
    build_subject_level_table,
    compute_generalization_matrices,
    pair_label,
    parse_run_dir_name,
    run_temporal_generalization_analysis,
    summarize_run_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-training temporal generalization analysis (train×test).")
    p.add_argument("--run-root", type=Path, required=True, help="Run root containing per-window generalization run dirs.")
    p.add_argument("--baseline", type=float, default=50.0, help="Chance baseline in percent (default: 50.0).")
    p.add_argument("--fdr-alpha", type=float, default=0.05, help="BH-FDR alpha across all cells (default: 0.05).")
    p.add_argument("--no-overall", action="store_true", help="Do not write the overall (pair-averaged) figures/matrices.")
    p.add_argument("--no-per-pair", action="store_true", help="Do not write per-pair figures (2× per pair).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(run_root)
    out = run_temporal_generalization_analysis(
        run_root=run_root,
        baseline=float(args.baseline),
        fdr_alpha=float(args.fdr_alpha),
        write_overall=not bool(args.no_overall),
        write_per_pair=not bool(args.no_per_pair),
    )
    print("[temporal-generalization] Done.")
    if out.get("fig_accuracy"):
        print(f"[temporal-generalization] Accuracy figure:     {out['fig_accuracy']}")
    if out.get("fig_significance"):
        print(f"[temporal-generalization] Significance figure: {out['fig_significance']}")
    if out.get("cell_stats_csv"):
        print(f"[temporal-generalization] Cell stats CSV:      {out['cell_stats_csv']}")
    if out.get("figures_per_pair"):
        print(f"[temporal-generalization] Per-pair figures:    {len(out['figures_per_pair'])} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


