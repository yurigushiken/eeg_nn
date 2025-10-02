from __future__ import annotations

import pytest

from code import model_builders


@pytest.mark.parametrize(
    "cfg,expected",
    [
        ({}, {"shift_p": 0.0, "noise_p": 0.0, "mixup_alpha": 0.0}),
        (
            {
                "augmentation": {
                    "shift_p": 0.2,
                    "noise_p": 0.3,
                    "time_mask_p": 0.4,
                    "mixup_alpha": 0.1,
                }
            },
            {
                "shift_p": 0.2,
                "noise_p": 0.3,
                "time_mask_p": 0.4,
                "mixup_alpha": 0.1,
            },
        ),
    ],
)
def test_eegnex_builder_aug_parameters(cfg, expected):
    builder = model_builders.get_model_builder("eegnex")
    model = builder(cfg, num_classes=3)

    for key, value in expected.items():
        assert getattr(model.cfg, key) == pytest.approx(value)

