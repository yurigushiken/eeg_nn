"""
Compile RSA run outputs into a single master CSV that now includes
per-subject metrics alongside OVERALL and per-fold entries.

The script scans run directories produced by run_rsa_matrix.py, extracts
per-fold/overall metrics from outer_eval_metrics.csv, derives true
per-subject accuracies from test_predictions_outer.csv, and writes a
unified rsa_results_master.csv that downstream statistics scripts rely on.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RUN_PATTERN = re.compile(r"rsa_(\d+)v(\d+)_seed_(\d+)")
MASTER_FILENAME = "rsa_results_master.csv"
PREDICTION_FILENAMES = ("test_predictions_outer.csv", "test_predictions.csv")
SUBJECT_COL_CANDIDATES = ("subject_id", "subject", "subj", "subjectIDX", "subject_index")
TRUE_LABEL_COLS = ("true_label_idx", "y_true_idx", "y_true", "true_label_name", "y_true_name")
PRED_LABEL_COLS = ("pred_label_idx", "y_pred_idx", "y_pred", "pred_label_name", "y_pred_name")


def parse_run_metadata(run_dir: Path) -> Tuple[int, int, int]:
    match = RUN_PATTERN.search(run_dir.name)
    if not match:
        raise ValueError(f"Run directory name does not match expected pattern: {run_dir}")
    class_a, class_b, seed = match.groups()
    return int(class_a), int(class_b), int(seed)


def load_outer_eval_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"outer_fold", "acc", "macro_f1", "min_per_class_f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {missing} in {csv_path}")
    return df


def _find_predictions_csv(run_dir: Path) -> Path | None:
    for filename in PREDICTION_FILENAMES:
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    return None


def _detect_subject_column(df: pd.DataFrame) -> str:
    for col in SUBJECT_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        "Predictions CSV is missing a subject column "
        f"(searched {', '.join(SUBJECT_COL_CANDIDATES)})."
    )


def _ensure_correct_column(df: pd.DataFrame) -> pd.Series:
    if "correct" in df.columns:
        try:
            return df["correct"].astype(float)
        except Exception:
            return df["correct"]
    true_col = next((c for c in TRUE_LABEL_COLS if c in df.columns), None)
    pred_col = next((c for c in PRED_LABEL_COLS if c in df.columns), None)
    if not true_col or not pred_col:
        raise KeyError(
            "Predictions CSV is missing both 'correct' and explicit true/pred label columns."
        )
    return (df[pred_col] == df[true_col]).astype(float)


def derive_subject_rows(
    run_dir: Path,
    class_a: int,
    class_b: int,
    seed: int,
) -> List[Dict]:
    pred_csv = _find_predictions_csv(run_dir)
    if pred_csv is None:
        print(
            f"[compile_rsa_results] WARNING: Missing predictions CSV in {run_dir}; "
            "subject-level rows will be skipped.",
            flush=True,
        )
        return []

    df = pd.read_csv(pred_csv)
    if "outer_fold" not in df.columns:
        raise KeyError(f"'outer_fold' column not found in {pred_csv}")

    subject_col = _detect_subject_column(df)
    df["correct"] = _ensure_correct_column(df)

    rows: List[Dict] = []
    for subject_id, group in df.groupby(subject_col):
        fold_values = group["outer_fold"].unique()
        fold_value = str(fold_values[0]) if fold_values.size else "NA"
        accuracy = float(np.nanmean(group["correct"]) * 100.0)
        rows.append(
            {
                "ClassA": class_a,
                "ClassB": class_b,
                "Seed": int(seed),
                "Fold": fold_value,
                "Subject": str(subject_id),
                "RecordType": "subject",
                "Accuracy": accuracy,
                "MacroF1": float("nan"),
                "MinClassF1": float("nan"),
            }
        )
    return rows


def format_subject(row: pd.Series) -> str:
    fold = str(row["outer_fold"]).strip()
    if fold.upper() == "OVERALL":
        return "OVERALL"
    subj = str(row.get("test_subjects", "")).strip()
    return subj if subj else fold


def compile_results_dataframe(runs_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for subdir in runs_root.iterdir():
        if not subdir.is_dir():
            continue
        csv_path = subdir / "outer_eval_metrics.csv"
        if not csv_path.exists():
            continue
        try:
            class_a, class_b, seed = parse_run_metadata(subdir)
        except ValueError:
            continue
        df = load_outer_eval_csv(csv_path)
        for _, row in df.iterrows():
            fold_value = str(row["outer_fold"]).strip()
            record_type = "overall" if fold_value.upper() == "OVERALL" else "fold"
            entry = {
                "ClassA": class_a,
                "ClassB": class_b,
                "Seed": int(seed),
                "Fold": fold_value,
                "Subject": format_subject(row),
                "RecordType": record_type,
                "Accuracy": float(row["acc"]),
                "MacroF1": float(row.get("macro_f1", float("nan"))),
                "MinClassF1": float(row.get("min_per_class_f1", float("nan"))),
            }
            rows.append(entry)

        rows.extend(derive_subject_rows(subdir, class_a, class_b, seed))

    if not rows:
        raise FileNotFoundError(f"No outer_eval_metrics.csv files found under {runs_root}")
    return pd.DataFrame(rows)


def write_master_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sort_cols = ["ClassA", "ClassB", "Seed", "RecordType", "Fold", "Subject"]
    existing = [c for c in sort_cols if c in df.columns]
    df.sort_values(existing, inplace=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile RSA run outputs into a master CSV.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("results") / "runs" / "rsa_matrix_v1",
        help="Directory containing RSA run subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the compiled master CSV (defaults to runs_dir/rsa_results_master.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = compile_results_dataframe(args.runs_dir)
    output_path = args.output
    if output_path is None:
        output_path = args.runs_dir / MASTER_FILENAME
    elif output_path.is_dir():
        output_path = output_path / MASTER_FILENAME
    write_master_csv(df, output_path)
    print(f"[compile_rsa_results] Master CSV written to {output_path.resolve()}")


if __name__ == "__main__":
    main()

