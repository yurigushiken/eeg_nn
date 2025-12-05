import pandas as pd
import tempfile
from pathlib import Path

import pytest


def test_rsa_pair_generation():
    from scripts.run_rsa_matrix import generate_cross_digit_pairs

    pairs = generate_cross_digit_pairs([11, 22, 33, 44, 55, 66])
    assert len(pairs) == 15
    assert pairs[0] == (11, 22)
    assert pairs[-1] == (55, 66)
    # Ensure no same-digit combos
    for a, b in pairs:
        assert a != b


def test_rsa_label_fn_filters_non_targets():
    from tasks import rsa_binary

    rsa_binary.set_active_pair((22, 55))

    df = pd.DataFrame({"Condition": [11, 22, 55, 66]})
    labels = rsa_binary.label_fn(df)

    assert pd.isna(labels.iloc[0])
    assert labels.iloc[1] == "22"
    assert labels.iloc[2] == "55"
    assert pd.isna(labels.iloc[3])


def test_rsa_aggregate_outer_metrics(tmp_path: Path):
    from scripts.run_rsa_matrix import aggregate_outer_metrics

    run_dir = Path(tmp_path)
    csv_path = run_dir / "outer_eval_metrics.csv"
    csv_path.write_text(
        "outer_fold,acc,macro_f1,min_per_class_f1\n"
        "1,60.0,58.0,55.0\n"
        "OVERALL,62.5,60.5,57.0\n"
    )

    rows = list(aggregate_outer_metrics(run_dir, 11, 33, seed=42))
    assert rows == [
        {
            "ClassA": 11,
            "ClassB": 33,
            "Seed": 42,
            "Accuracy": 62.5,
            "MacroF1": 60.5,
            "MinClassF1": 57.0,
        }
    ]

