"""
Generate decision-layer comparison plots for a completed run directory.

Outputs:
- <run_dir>/plots_outer_threshold_compare/
  - confusion_comparison.png (baseline vs thresholded)
  - per_fold_accuracy_comparison.png
  - fold<k>_confusion_comparison.png (per-fold)

Usage (PowerShell):
  python -X utf8 -u scripts/generate_decision_layer_plots.py "<run_dir>"

Notes:
- Requires both test_predictions_outer.csv and test_predictions_outer_thresholded.csv
- Class names are loaded from summary_*.json (fallback: numeric strings)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import List, Dict


def _load_csv_rows(csv_path: Path) -> List[Dict]:
    import csv
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["outer_fold"] = int(row["outer_fold"]) if "outer_fold" in row and row["outer_fold"] != "" else 0
            row["true_label_idx"] = int(row["true_label_idx"]) if "true_label_idx" in row else int(row.get("true", 0))
            row["pred_label_idx"] = int(row["pred_label_idx"]) if "pred_label_idx" in row else int(row.get("pred", 0))
            row["correct"] = int(row.get("correct", int(row["true_label_idx"] == row["pred_label_idx"])) )
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate decision-layer comparison plots for a run directory")
    ap.add_argument("run_dir", type=str, help="Path to completed run directory")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    baseline_csv = run_dir / "test_predictions_outer.csv"
    thresholded_csv = run_dir / "test_predictions_outer_thresholded.csv"

    if not baseline_csv.exists():
        raise SystemExit(f"baseline predictions CSV not found: {baseline_csv}")
    if not thresholded_csv.exists():
        raise SystemExit(
            "thresholded predictions CSV not found. Ensure decision layer is enabled for this run "
            f"and file exists: {thresholded_csv}"
        )

    # Load class names from summary_*.json if available
    class_names: List[str] = []
    try:
        summary_path = next(run_dir.glob("summary_*.json"))
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            class_names = summary.get("class_names", [])
    except StopIteration:
        class_names = []
    except Exception:
        class_names = []

    # Fallback: infer from data
    if not class_names:
        all_rows = _load_csv_rows(baseline_csv)
        max_idx = 0
        for r in all_rows:
            max_idx = max(max_idx, int(r["true_label_idx"]), int(r["pred_label_idx"]))
        class_names = [str(i + 1) for i in range(max_idx + 1)]

    # Load rows
    baseline_rows = _load_csv_rows(baseline_csv)
    thresholded_rows = _load_csv_rows(thresholded_csv)

    # Build per-fold comparisons
    from collections import defaultdict
    folds = sorted({r.get("outer_fold", 0) for r in baseline_rows} | {r.get("outer_fold", 0) for r in thresholded_rows})

    from code.posthoc.decision_layer import compute_fold_comparison
    fold_comparisons: List[Dict] = []
    for fold in folds:
        base_fold = [r for r in baseline_rows if r.get("outer_fold", 0) == fold]
        thrs_fold = [r for r in thresholded_rows if r.get("outer_fold", 0) == fold]
        if not base_fold or not thrs_fold:
            continue
        cmp_dict = compute_fold_comparison(
            baseline_rows=base_fold,
            thresholded_rows=thrs_fold,
            fold=fold,
            num_classes=len(class_names)
        )
        fold_comparisons.append(cmp_dict)

    # Generate plots
    from code.posthoc.decision_layer_plots import generate_all_comparison_plots
    out_dir = run_dir / "plots_outer_threshold_compare"
    generate_all_comparison_plots(
        baseline_rows=baseline_rows,
        thresholded_rows=thresholded_rows,
        fold_comparisons=fold_comparisons,
        class_names=class_names,
        out_dir=out_dir
    )

    print(f"[done] decision-layer comparison plots written to: {out_dir}")


if __name__ == "__main__":
    main()


