from __future__ import annotations
from typing import Dict, Callable

from code.training_runner import TrainingRunner
from code.datasets import make_dataset
from code.model_builders import RAW_EEG_MODELS, squeeze_input_adapter, build_raw_eeg_aug

"""
EEG engine entry point.

This wraps model/dataset construction and delegates training/evaluation to
TrainingRunner. It is intentionally thin so engines can be swapped/extended.

Contract:
- cfg: consolidated configuration (common → base/resolved → overlays → sampled space)
- label_fn: task-specific function mapping trial metadata → class labels

Input/output shapes:
- Dataset yields X with shape (N, 1, C, T) in microvolts, and y with shape (N,)
- Some models (e.g., EEGNeX) expect (B, C, T); we use an input adapter that
  squeezes the singleton dimension uniformly in the engine.
"""

def run(cfg: Dict, label_fn: Callable):
    """Build dataset and model, then execute TrainingRunner.

    Steps:
    1) Build dataset via make_dataset(cfg, label_fn), which handles materialized_dir,
       cropping, channel policies, and produces groups/class_names.
    2) Select a raw-EEG model by name (cfg.model_name) and prepare a builder that
       injects dataset-dependent shapes (C channels, T time points).
    3) Choose an input adapter if the model expects squeezed inputs.
    4) Create TrainingRunner and call runner.run with augmentation builder,
       allowing Optuna pruning through cfg["optuna_trial"].
    """
    dataset = make_dataset(cfg, label_fn)

    model_name = cfg.get("model_name", "eegnex")
    if model_name not in RAW_EEG_MODELS:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(RAW_EEG_MODELS.keys())}")

    # Model builder is a closure that TrainingRunner calls to instantiate the model
    model_builder = lambda conf, num_cls: RAW_EEG_MODELS[model_name](
        conf, num_cls, C=dataset.num_channels, T=dataset.time_points
    )

    # Some models expect inputs shaped differently; adapter handles that uniformly
    # This keeps TrainingRunner agnostic of per-model input signature differences
    input_adapter = squeeze_input_adapter if model_name in ("cwat", "eegnex") else None

    runner = TrainingRunner(cfg, label_fn)
    summary = runner.run(
        dataset=dataset,
        groups=dataset.groups,
        class_names=dataset.class_names,
        model_builder=model_builder,
        aug_builder=lambda conf, d: build_raw_eeg_aug(conf, d.time_points),
        input_adapter=input_adapter,
        optuna_trial=cfg.get("optuna_trial")
    )
    return summary


