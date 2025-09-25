from __future__ import annotations
from typing import Dict, Any

import torch.nn as nn

"""
Model builders and augmentation utilities for raw EEG.

We wrap external implementations (e.g., Braindecode EEGNeX) behind small
builder functions that translate cfg keys into model constructor arguments.

References:
- EEGNeX (Braindecode models): see Braindecode docs and the associated papers
  for architectural details; here we expose the key hyper-parameters via cfg.

Notes on shapes:
- Models typically expect inputs shaped (B, C, T). Our datasets yield (B, 1, C, T)
  and the engine applies a squeeze adapter where appropriate to keep interfaces simple.
"""

try:
    from braindecode.models import EEGNeX as BD_EEGNeX
except Exception:
    BD_EEGNeX = None


def build_eegnex(cfg: Dict[str, Any], num_classes: int, C: int, T: int) -> nn.Module:
    """Build an EEGNeX model from cfg.

    Key cfg→constructor mappings (defaults in parentheses):
      - activation ('elu'): one of {'elu','relu','leaky_relu','gelu','celu','silu'}
      - depth_multiplier (2): channel expansion factor per block
      - filter_1 (8), filter_2 (32): base conv feature sizes
      - kernel_block_1_2 (32), kernel_block_4 (16), kernel_block_5 (16): kernel widths
      - avg_pool_block4 (4), avg_pool_block5 (8): temporal pooling factors
      - dilation_block_4 (2), dilation_block_5 (4): dilations in later blocks
      - drop_prob (0.5): dropout probability
      - max_norm_conv (1.0), max_norm_linear (0.25): weight max-norm constraints

    Shapes:
      - n_chans=C, n_times=T, n_outputs=num_classes
    """
    if BD_EEGNeX is None:
        raise ImportError("Braindecode EEGNeX not available. Install braindecode>=1.1.0.")

    _act_map = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "celu": nn.CELU,
        "silu": nn.SiLU,
    }
    activation = _act_map.get(str(cfg.get("activation", "elu")).lower(), nn.ELU)

    model = BD_EEGNeX(
        n_chans=C,
        n_outputs=num_classes,
        n_times=T,
        activation=activation,
        depth_multiplier=int(cfg.get("depth_multiplier", 2)),
        filter_1=int(cfg.get("filter_1", 8)),
        filter_2=int(cfg.get("filter_2", 32)),
        drop_prob=float(cfg.get("drop_prob", 0.5)),
        kernel_block_1_2=int(cfg.get("kernel_block_1_2", 32)),
        kernel_block_4=int(cfg.get("kernel_block_4", 16)),
        kernel_block_5=int(cfg.get("kernel_block_5", 16)),
        avg_pool_block4=int(cfg.get("avg_pool_block4", 4)),
        avg_pool_block5=int(cfg.get("avg_pool_block5", 8)),
        dilation_block_4=int(cfg.get("dilation_block_4", 2)),
        dilation_block_5=int(cfg.get("dilation_block_5", 4)),
        max_norm_conv=float(cfg.get("max_norm_conv", 1.0)),
        max_norm_linear=float(cfg.get("max_norm_linear", 0.25)),
    )
    return model


def squeeze_input_adapter(x):
    # (B,1,C,T) -> (B,C,T). Used by engines for models that expect squeezed inputs.
    return x.squeeze(1)


def build_raw_eeg_aug(cfg: Dict[str, Any], T: int):
    """Create a stateless train-time augmentation transform based on cfg.

    The returned callable expects a single-example tensor of shape (1,C,T) and
    applies random time shift, scaling, Gaussian noise, time masking, and channel
    masking according to probabilities in cfg. Returning None means no-op.

    Scientific note:
    - All transforms here are applied at train time only (dataset.set_transform on train split);
      validation/test data remain unaugmented to avoid leakage. Mixup (if enabled) is handled
      inside the training loop.
    """
    import torch
    import random
    from typing import Callable

    mixup_alpha = float(cfg.get("mixup_alpha", 0.0) or 0.0)  # handled in runner
    shift_p = float(cfg.get("shift_p", 0.0) or 0.0)
    shift_max_frac = float(cfg.get("shift_max_frac", 0.0) or 0.0)
    scale_p = float(cfg.get("scale_p", 0.0) or 0.0)
    scale_min = float(cfg.get("scale_min", 1.0) or 1.0)
    scale_max = float(cfg.get("scale_max", 1.0) or 1.0)
    noise_p = float(cfg.get("noise_p", 0.0) or 0.0)
    noise_std = float(cfg.get("noise_std", 0.0) or 0.0)
    time_mask_p = float(cfg.get("time_mask_p", 0.0) or 0.0)
    time_mask_frac = float(cfg.get("time_mask_frac", 0.0) or 0.0)
    chan_mask_p = float(cfg.get("chan_mask_p", 0.0) or 0.0)
    chan_mask_ratio = float(cfg.get("chan_mask_ratio", 0.0) or 0.0)

    def _transform(x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, C, T)
        if shift_p > 0.0 and shift_max_frac > 0.0 and random.random() < shift_p:
            max_shift = max(1, int(round(shift_max_frac * x.shape[-1])))
            s = random.randint(-max_shift, max_shift)
            if s != 0:
                x = torch.roll(x, shifts=s, dims=-1)
        if scale_p > 0.0 and random.random() < scale_p:
            scale = random.uniform(scale_min, scale_max)
            x = x * scale
        if noise_p > 0.0 and noise_std > 0.0 and random.random() < noise_p:
            x = x + torch.randn_like(x) * noise_std
        if time_mask_p > 0.0 and time_mask_frac > 0.0 and random.random() < time_mask_p:
            L = int(round(time_mask_frac * x.shape[-1]))
            if L > 0:
                start = random.randint(0, max(0, x.shape[-1] - L))
                x[..., start:start+L] = 0.0
        if chan_mask_p > 0.0 and chan_mask_ratio > 0.0 and random.random() < chan_mask_p:
            C = x.shape[-2]
            k = max(1, int(round(chan_mask_ratio * C)))
            idx = torch.randperm(C)[:k]
            x[:, idx, :] = 0.0
        return x

    # If all probabilities are zero (no-op), return None
    if all(v == 0.0 for v in [shift_p, scale_p, noise_p, time_mask_p, chan_mask_p]) and mixup_alpha == 0.0:
        return None
    return _transform


RAW_EEG_MODELS = {
    "eegnex": build_eegnex,
}


