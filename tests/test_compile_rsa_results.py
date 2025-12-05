import csv
from pathlib import Path

import pandas as pd


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

    from scripts.compile_rsa_results import compile_results_dataframe

    df = compile_results_dataframe(run_root)
    expected_cols = {"ClassA", "ClassB", "Seed", "Subject", "Fold", "RecordType", "Accuracy", "MacroF1", "MinClassF1"}
    assert set(df.columns) == expected_cols
    assert len(df) == 2

    overall = df[df["Subject"] == "OVERALL"].iloc[0]
    assert overall["ClassA"] == 11
    assert overall["ClassB"] == 22
    assert overall["Seed"] == 42
    assert overall["Accuracy"] == 62.0

    fold_row = df[df["Subject"] != "OVERALL"].iloc[0]
    assert fold_row["Subject"] == "9,15"
    assert fold_row["Fold"] == "1"
    assert fold_row["Accuracy"] == 60.0

