import pandas as pd
from pathlib import Path


def _make_master_csv(tmp_path: Path) -> Path:
    data = [
        # Pair 11 vs 22
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "Subject": "subject_01", "RecordType": "subject", "Accuracy": 68.0, "MacroF1": 66.0, "MinClassF1": 63.0, "n_trials": 30, "n_correct": 20, "ChanceRate": 60.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "Subject": "subject_02", "RecordType": "subject", "Accuracy": 72.0, "MacroF1": 70.0, "MinClassF1": 67.0, "n_trials": 40, "n_correct": 30, "ChanceRate": 50.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 43, "Subject": "subject_03", "RecordType": "subject", "Accuracy": 71.0, "MacroF1": 69.0, "MinClassF1": 66.0, "n_trials": 20, "n_correct": 14, "ChanceRate": 55.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 43, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 70.0, "MacroF1": 68.0, "MinClassF1": 65.0, "n_trials": None, "n_correct": None, "ChanceRate": 55.0},
        # Pair 33 vs 44
        {"ClassA": 33, "ClassB": 44, "Seed": 42, "Subject": "subject_01", "RecordType": "subject", "Accuracy": 55.0, "MacroF1": 53.0, "MinClassF1": 50.0, "n_trials": 25, "n_correct": 14, "ChanceRate": 60.0},
        {"ClassA": 33, "ClassB": 44, "Seed": 42, "Subject": "subject_02", "RecordType": "subject", "Accuracy": 51.0, "MacroF1": 49.0, "MinClassF1": 46.0, "n_trials": 30, "n_correct": 15, "ChanceRate": 55.0},
        {"ClassA": 33, "ClassB": 44, "Seed": 43, "Subject": "subject_03", "RecordType": "subject", "Accuracy": 54.0, "MacroF1": 52.0, "MinClassF1": 49.0, "n_trials": 18, "n_correct": 10, "ChanceRate": 50.0},
        {"ClassA": 33, "ClassB": 44, "Seed": 43, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 53.0, "MacroF1": 51.0, "MinClassF1": 48.0, "n_trials": None, "n_correct": None, "ChanceRate": 55.0},
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "rsa_results_master.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_compute_subject_stats(tmp_path: Path):
    csv_path = _make_master_csv(tmp_path)

    from scripts.analyze_rsa_stats import (
        load_master_csv,
        filter_subject_rows,
        compute_subject_ttests,
    )

    df = load_master_csv(csv_path)
    subject_df = filter_subject_rows(df)
    stats_df = compute_subject_ttests(subject_df, baseline=None)

    assert {
        "ClassA",
        "ClassB",
        "mean_accuracy",
        "t_stat",
        "p_value",
        "p_value_holm",
        "n_subjects",
        "chance_rate",
    }.issubset(stats_df.columns)

    pair_stats = stats_df.set_index(["ClassA", "ClassB"])
    assert (11, 22) in pair_stats.index
    assert pair_stats.loc[(11, 22), "chance_rate"] >= 55.0
    assert not pd.isna(pair_stats.loc[(11, 22), "t_stat"])


def test_t_test_requires_two_samples():
    from scripts.analyze_rsa_stats import _t_test
    import numpy as np

    t_stat, p_val = _t_test(values=np.array([60.0]), baseline=50.0)
    assert pd.isna(t_stat)
    assert pd.isna(p_val)

