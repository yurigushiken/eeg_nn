from __future__ import annotations
import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Callable, List, Dict

import sys
import types
import numpy as np

# Stub pdb early to avoid stdlib 'code' conflict via torch.distributed
class _NoopPdb:  # minimal stub
    pass

sys.modules.setdefault("pdb", types.SimpleNamespace(Pdb=_NoopPdb))

import torch
import torch.nn as nn


# Import refactor and legacy runners
from code.training_runner import TrainingRunner as RefRunner
from code.training_runner_legacy import TrainingRunner as LegacyRunner


class TinyDataset:
    """Synthetic, deterministic dataset implementing required interface."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        # X: (N, 1, C, T), y: (N,)
        self.X = X.clone()
        self.y = y.clone()
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.transform is not None:
            try:
                x = self.transform(x)
            except Exception:
                pass
        y = self.y[idx]
        return x, y

    def get_all_labels(self) -> np.ndarray:
        return self.y.cpu().numpy().astype(int)


def tiny_label_fn(md) -> List[str]:
    # Not used by our synthetic dataset, present for API compatibility
    return []


def tiny_model_builder(cfg: Dict, num_classes: int) -> nn.Module:
    # Expect input shaped as (B, 1, C, T); flatten to features and linear head
    c = cfg.get("test_C", 4)
    t = cfg.get("test_T", 8)
    in_features = int(c) * int(t)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, num_classes, bias=True),
    )
    # Deterministic init independent of RNG: fill weights/bias with a fixed pattern
    lin = model[1]
    with torch.no_grad():
        # Create a simple ramp pattern
        w = torch.linspace(-0.1, 0.1, steps=lin.weight.numel()).reshape_as(lin.weight)
        b = torch.linspace(0.05, -0.05, steps=lin.bias.numel())
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    return model


def tiny_aug_builder(cfg: Dict, dataset: TinyDataset) -> Callable:
    # Return a no-op transform that preserves determinism
    def _noop(x):
        return x
    return _noop


class EquivalenceRefactorVsLegacyTest(unittest.TestCase):
    def setUp(self):
        # Deterministic environment
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        np.random.seed(42)

        # Tiny synthetic dataset (balanced, subject-aware groups)
        N = 24
        C = 4
        T = 8
        num_classes = 3
        X = torch.arange(N * 1 * C * T, dtype=torch.float32).reshape(N, 1, C, T) / 1000.0
        y = torch.tensor([(i % num_classes) for i in range(N)], dtype=torch.int64)
        # 6 subjects, 4 trials each -> enables GroupKFold(n_splits=2)
        self.groups = np.repeat(np.arange(6), 4)
        self.class_names = [f"cls{i}" for i in range(num_classes)]

        self.dataset = TinyDataset(X, y)

        # Shared minimal config
        self.base_cfg = {
            "seed": 42,
            "n_folds": 2,
            "inner_n_folds": 2,
            "epochs": 2,
            # Legacy requires explicit objective for scientific validity
            "optuna_objective": "inner_mean_macro_f1",
            "batch_size": 4,
            "early_stop": 2,
            "scheduler_patience": 1,
            "lr": 0.01,
            "weight_decay": 0.0,
            "outer_eval_mode": "ensemble",
            "outputs": {
                "write_learning_curves_csv": True,
                "write_outer_eval_csv": True,
                "write_test_predictions_csv": True,
                "write_splits_indices_json": True,
            },
            # Hints for model input dimensions
            "test_C": C,
            "test_T": T,
        }

        # Temporary run dirs
        self.tmpdir = Path(tempfile.mkdtemp(prefix="eeg_refactor_test_"))
        print(f"[equivalence-test] tmpdir={self.tmpdir}")
        self.run_ref = self.tmpdir / "run_refactor"
        self.run_leg = self.tmpdir / "run_legacy"
        self.run_ref.mkdir(parents=True, exist_ok=True)
        self.run_leg.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Preserve tmpdir for post-failure inspection
        pass

    def _run(self, cfg: Dict, run_dir: Path, runner_cls) -> Dict:
        cfg = dict(cfg)
        cfg["run_dir"] = str(run_dir)
        runner = runner_cls(cfg, tiny_label_fn)
        results = runner.run(
            dataset=self.dataset,
            groups=self.groups,
            class_names=self.class_names,
            model_builder=tiny_model_builder,
            aug_builder=tiny_aug_builder,
            input_adapter=None,
            optuna_trial=None,
            labels_override=None,
            predefined_splits=None,
        )
        return results

    def test_equivalence_metrics_and_artifacts(self):
        # Run legacy and refactor with identical config
        res_leg = self._run(self.base_cfg, self.run_leg, LegacyRunner)
        res_ref = self._run(self.base_cfg, self.run_ref, RefRunner)

        # Helper to read file as text safely
        def read_text(p: Path) -> str:
            return p.read_text(encoding="utf-8").strip() if p.exists() else ""

        # Compare outer metrics CSV
        leg_outer = read_text(self.run_leg / "outer_eval_metrics.csv")
        ref_outer = read_text(self.run_ref / "outer_eval_metrics.csv")
        self.assertEqual(leg_outer, ref_outer, "outer_eval_metrics.csv differs")

        # Compare learning curves CSV
        leg_curves = read_text(self.run_leg / "learning_curves_inner.csv")
        ref_curves = read_text(self.run_ref / "learning_curves_inner.csv")
        self.assertEqual(leg_curves, ref_curves, "learning_curves_inner.csv differs")

        # Compare test prediction CSVs (outer)
        leg_pred_outer = read_text(self.run_leg / "test_predictions_outer.csv")
        ref_pred_outer = read_text(self.run_ref / "test_predictions_outer.csv")
        self.assertEqual(leg_pred_outer, ref_pred_outer, "test_predictions_outer.csv differs")

        # Compare test prediction CSVs (inner)
        leg_pred_inner = read_text(self.run_leg / "test_predictions_inner.csv")
        ref_pred_inner = read_text(self.run_ref / "test_predictions_inner.csv")
        self.assertEqual(leg_pred_inner, ref_pred_inner, "test_predictions_inner.csv differs")

        # Compare splits_indices.json structurally and as text
        leg_splits = read_text(self.run_leg / "splits_indices.json")
        ref_splits = read_text(self.run_ref / "splits_indices.json")
        self.assertEqual(leg_splits, ref_splits, "splits_indices.json differs")

        # Compare outer predictions first (root cause of metric diffs)
        leg_outer_lines = (self.run_leg / "test_predictions_outer.csv").read_text(encoding="utf-8").splitlines()
        ref_outer_lines = (self.run_ref / "test_predictions_outer.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(leg_outer_lines, ref_outer_lines, "Outer predictions CSV content differs (line-by-line)")

        # If predictions match, now the summaries must match
        self.assertEqual(res_leg, res_ref, "Returned summary metrics differ between legacy and refactor")


if __name__ == "__main__":
    unittest.main()


