from __future__ import annotations

import numpy as np

import pytest

from code.datasets import make_dataset
from code import training_runner


def label_fn(metadata):
    return metadata["label"].to_numpy()


class DummyModel:
    def __init__(self, cfg, num_classes):
        pass

    def to(self, device):
        return self

    def __call__(self, xb):
        import torch

        bsz = xb.size(0)
        num_classes = 2
        return torch.zeros((bsz, num_classes))


def identity_aug(cfg, dataset):
    def _apply(x):
        return x

    return _apply


def test_training_runner_enforces_subject_threshold(materialized_dir, synthetic_subject_specs):
    cfg = {
        "materialized_dir": str(materialized_dir),
        "min_trials_per_class": 5,
        "min_subjects": 10,
        "seed": 123,
        "inner_n_folds": 2,
        "epochs": 1,
    }

    dataset = make_dataset(cfg, label_fn)
    runner = training_runner.TrainingRunner(cfg, label_fn)

    with pytest.raises(ValueError, match="fewer than the required minimum of 10 subjects"):
        runner.run(
            dataset,
            groups=np.array(dataset.groups),
            class_names=dataset.class_names,
            model_builder=DummyModel,
            aug_builder=identity_aug,
        )

