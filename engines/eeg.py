from __future__ import annotations
from typing import Dict, Callable

from code.training_runner import TrainingRunner
from code.datasets import make_dataset
from code.model_builders import RAW_EEG_MODELS, squeeze_input_adapter, build_raw_eeg_aug

def run(cfg: Dict, label_fn: Callable):
    dataset = make_dataset(cfg, label_fn)

    model_name = cfg.get("model_name", "eegnex")
    if model_name not in RAW_EEG_MODELS:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(RAW_EEG_MODELS.keys())}")

    model_builder = lambda conf, num_cls: RAW_EEG_MODELS[model_name](
        conf, num_cls, C=dataset.num_channels, T=dataset.time_points
    )

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


