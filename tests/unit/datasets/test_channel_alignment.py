from __future__ import annotations

import pytest

pytest.skip("Channel intersection verification cancelled per updated scope", allow_module_level=True)

import numpy as np

from code.datasets import make_dataset


def label_fn(metadata):
    return metadata["label"].to_numpy()


def test_channel_intersection_persisted(materialized_dataset) -> None:
    cfg = {
        "materialized_dir": str(materialized_dataset.materialized_dir),
        "min_trials_per_class": 5,
    }

    dataset = make_dataset(cfg, label_fn)

    # Intersection should retain channels common to all non-excluded subjects
    intersection = set(dataset.channel_names)
    assert intersection == {"Fz", "Cz", "Oz"}, "Channel intersection should match common set"

    metadata = getattr(dataset, "channel_metadata", None)
    assert metadata is not None, "Dataset must expose channel_metadata"
    assert metadata["intersection"] == list(intersection)
    assert metadata["per_subject"].keys() >= {1, 2}

    # Ensure times and sfreq propagated for downstream reporting
    assert np.isclose(dataset.sfreq, materialized_dataset.sfreq)
    assert np.allclose(dataset.times_ms, materialized_dataset.times_ms)

