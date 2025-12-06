import csv
from pathlib import Path

import pandas as pd
import pytest


def _write_outer_eval_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "outer_fold",
        "test_subjects",
        "n_test_trials",
        "acc",
        "acc_std",
        "macro_f1",
        "macro_f1_std",
        "min_per_class_f1",
        "min_per_class_f1_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_predictions_csv(path: Path) -> None:
    rows = [
        {"outer_fold": 1, "subject_id": "subj_01", "true_label_idx": 0, "pred_label_idx": 0},
        {"outer_fold": 1, "subject_id": "subj_01", "true_label_idx": 1, "pred_label_idx": 0},
        {"outer_fold": 1, "subject_id": "subj_02", "true_label_idx": 0, "pred_label_idx": 1},
    ]
    fieldnames = ["outer_fold", "subject_id", "true_label_idx", "pred_label_idx"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compile_results_produces_master_csv(tmp_path: Path):
    run_root = tmp_path / "rsa_runs"
    run_root.mkdir()

    run_dir = run_root / "20251205_120000_rsa_11v22_seed_42"
    run_dir.mkdir()
    _write_outer_eval_csv(
        run_dir / "outer_eval_metrics.csv",
        [
            {
                "outer_fold": "1",
                "test_subjects": "9,15",
                "n_test_trials": "72",
                "acc": "60.0",
                "acc_std": "",
                "macro_f1": "58.0",
                "macro_f1_std": "",
                "min_per_class_f1": "55.0",
                "min_per_class_f1_std": "",
            },
            {
                "outer_fold": "OVERALL",
                "test_subjects": "-",
                "n_test_trials": "144",
                "acc": "62.0",
                "acc_std": "2.0",
                "macro_f1": "60.0",
                "macro_f1_std": "1.5",
                "min_per_class_f1": "56.0",
                "min_per_class_f1_std": "1.0",
            },
        ],
    )
    _write_predictions_csv(run_dir / "test_predictions_outer.csv")

    from scripts.compile_rsa_results import compile_results_dataframe

    df = compile_results_dataframe(run_root)
    expected_cols = {
        "ClassA",
        "ClassB",
        "Seed",
        "Subject",
        "Fold",
        "RecordType",
        "Accuracy",
        "MacroF1",
        "MinClassF1",
        "n_trials",
        "n_correct",
        "ChanceRate",
    }
    assert set(df.columns) == expected_cols
    assert len(df) == 4  # 2 outer_eval rows + 2 subject rows

    overall = df[df["RecordType"] == "overall"].iloc[0]
    assert overall["ClassA"] == 11
    assert overall["ClassB"] == 22
    assert overall["Seed"] == 42
    assert overall["Accuracy"] == 62.0

    subj_rows = df[df["RecordType"] == "subject"].sort_values("Subject")
    assert list(subj_rows["n_trials"]) == [2, 1]
    assert list(subj_rows["n_correct"]) == [1, 0]
    # ChanceRate should reflect observed label imbalance (2 of label 0, 1 of label 1) -> 2/3 * 100
    assert pytest.approx(subj_rows["ChanceRate"].iloc[0], rel=1e-3) == 66.6667

