from __future__ import annotations

from pathlib import Path

import numpy as np

from code.datasets import make_dataset


def label_fn(metadata):
    return metadata["label"].to_numpy()


def test_subjects_below_threshold_are_excluded(materialized_dir, subject_trial_counts, tmp_path) -> None:
    cfg = {
        "materialized_dir": str(materialized_dir),
        "min_trials_per_class": 5,
        "exclusion_log_path": str(tmp_path / "excluded_subjects.csv"),
    }

    dataset = make_dataset(cfg, label_fn)

    assert 3 not in set(np.unique(dataset.groups)), "Subject with <5 trials per class should be excluded"

    excluded = getattr(dataset, "excluded_subjects", None)
    assert excluded is not None, "Dataset should expose excluded_subjects metadata"
    assert 3 in excluded, "Excluded subjects metadata must list the filtered subject"
    info = excluded[3]
    assert info["min_trials_per_class"] == 5
    assert info["reason"] == "insufficient_trials"

    log_path = getattr(dataset, "exclusion_log_path", None)
    assert log_path is not None and Path(log_path).exists(), "Exclusion log file should be persisted"

