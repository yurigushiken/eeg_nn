from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest

from code.datasets import make_dataset
from code import training_runner


def label_fn(metadata):
    return metadata["label"].to_numpy()


class DummyModel:
    def __init__(self, cfg, num_classes):
        import torch
        self.num_classes = num_classes
        self._device = None
        # Dummy parameter so optimizer doesn't complain (must be a leaf tensor)
        self._dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def to(self, device):
        self._device = device
        if hasattr(self, '_dummy_param'):
            self._dummy_param.data = self._dummy_param.data.to(device)
        return self

    def parameters(self):
        # Return list with dummy parameter for optimizer
        return [self._dummy_param]

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, xb):
        import torch

        bsz = xb.size(0)
        # Use the dummy parameter to create outputs with gradients
        logits = self._dummy_param.expand(bsz, self.num_classes) * 0.0
        return logits


def noop_aug(cfg, dataset):
    def _apply(x):
        return x

    return _apply


@pytest.mark.parametrize("outer_eval_mode", ["ensemble", "refit"])
def test_runtime_artifacts_and_logging(materialized_dir, tmp_path, outer_eval_mode):
    cfg = {
        "materialized_dir": str(materialized_dir),
        "min_trials_per_class": 5,
        "min_subjects": 2,
        "seed": 42,
        "inner_n_folds": 2,
        "epochs": 1,
        "batch_size": 2,
        "run_dir": str(tmp_path / f"run_{outer_eval_mode}"),
        "outer_eval_mode": outer_eval_mode,
        "outputs": {
            "write_learning_curves_csv": True,
            "write_outer_eval_csv": True,
            "write_test_predictions_csv": True,
            "write_splits_indices_json": True,
        },
    }

    dataset = make_dataset(cfg, label_fn)
    runner = training_runner.TrainingRunner(cfg, label_fn)

    result = runner.run(
        dataset,
        groups=np.array(dataset.groups),
        class_names=dataset.class_names,
        model_builder=DummyModel,
        aug_builder=noop_aug,
    )

    run_dir = Path(cfg["run_dir"])
    assert run_dir.exists()

    # Check JSONL runtime log per logging contract placeholder
    runtime_log = run_dir / "logs" / "runtime.jsonl"
    assert runtime_log.exists()

    # Check class weights directory and split indices
    class_weights_dir = run_dir / "class_weights"
    assert class_weights_dir.exists()
    assert any(class_weights_dir.iterdir()), "Class weights should be persisted"

    splits_indices = json.loads((run_dir / "splits_indices.json").read_text())
    assert "outer_folds" in splits_indices

    outer_eval_csv = run_dir / "outer_eval_metrics.csv"
    assert outer_eval_csv.exists()

    test_preds_outer = run_dir / "test_predictions_outer.csv"
    assert test_preds_outer.exists()

    # Ensure telemetry and metrics returned
    assert "mean_acc" in result
    assert "inner_mean_macro_f1" in result

