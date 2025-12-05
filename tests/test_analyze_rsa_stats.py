import pandas as pd
from pathlib import Path


def _make_master_csv(tmp_path: Path) -> Path:
    data = [
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "Subject": "subject_01", "RecordType": "subject", "Accuracy": 68.0, "MacroF1": 66.0, "MinClassF1": 63.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "Subject": "subject_02", "RecordType": "subject", "Accuracy": 72.0, "MacroF1": 70.0, "MinClassF1": 67.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 43, "Subject": "subject_03", "RecordType": "subject", "Accuracy": 71.0, "MacroF1": 69.0, "MinClassF1": 66.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 43, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 70.0, "MacroF1": 68.0, "MinClassF1": 65.0},
        {"ClassA": 33, "ClassB": 44, "Seed": 42, "Subject": "subject_01", "RecordType": "subject", "Accuracy": 55.0, "MacroF1": 53.0, "MinClassF1": 50.0},
        {"ClassA": 33, "ClassB": 44, "Seed": 42, "Subject": "subject_02", "RecordType": "subject", "Accuracy": 51.0, "MacroF1": 49.0, "MinClassF1": 46.0},
        {"ClassA": 33, "ClassB": 44, "Seed": 43, "Subject": "subject_03", "RecordType": "subject", "Accuracy": 54.0, "MacroF1": 52.0, "MinClassF1": 49.0},
        {"ClassA": 33, "ClassB": 44, "Seed": 43, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 53.0, "MacroF1": 51.0, "MinClassF1": 48.0},
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
    stats_df = compute_subject_ttests(subject_df, baseline=50.0)

    assert {
        "ClassA",
        "ClassB",
        "mean_accuracy",
        "t_stat",
        "p_value",
        "p_value_holm",
        "n_subjects",
    }.issubset(stats_df.columns)

    pair_stats = stats_df.set_index(["ClassA", "ClassB"])
    assert (11, 22) in pair_stats.index
    assert pair_stats.loc[(11, 22), "mean_accuracy"] > 60.0
    assert pair_stats.loc[(11, 22), "p_value"] < 0.05
    assert pair_stats.loc[(11, 22), "p_value_holm"] >= pair_stats.loc[(11, 22), "p_value"]

