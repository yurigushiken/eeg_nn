from __future__ import annotations
from typing import Dict, Any, Callable, Tuple, List

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import mne
from sklearn.preprocessing import LabelEncoder

from .preprocessing.mne_pipeline import (
    spatial_sample,
    apply_crop_ms,
)

"""
Datasets for materialized EEG epochs.

We read per-subject .fif epoch files with attached metadata, standardize channels
across subjects, apply optional time cropping and channel selection, then build
arrays suitable for PyTorch.

Notes:
- On-the-fly preprocessing is intentionally removed to ensure reproducibility;
  set cfg['materialized_dir'] to a prepared dataset directory produced by
  `scripts/prepare_from_happe.py`.
- The label function maps trial metadata → string labels, which are encoded via
  LabelEncoder to integers for model training.
- Final tensor shapes: X is (N, 1, C, T) in microvolts; y is (N,). Channel names
  and time axis (ms) are stored for downstream plotting/XAI.
"""


class BaseEEGDataset(Dataset):
    def __init__(self):
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_all_labels(self) -> np.ndarray:
        return self.y.numpy() if isinstance(self.y, torch.Tensor) else self.y


class MaterializedEpochsDataset(BaseEEGDataset):
    """Reads .fif epochs with metadata, derives labels via label_fn.

    Expects files like sub-XX_preprocessed-epo.fif inside cfg['materialized_dir'].
    Builds tensors X (N, 1, C, T) and y (N,) and exposes groups and class_names.

    Channel alignment:
    - Channels are intersected across subjects and reordered to a canonical order
      to ensure identical C dimension for all examples (CNNs require consistent
      channel order across subjects).
    - Times are validated to be consistent across subjects and converted to ms.
    """
    def __init__(self, cfg: Dict[str, Any], label_fn: Callable):
        super().__init__()
        self.cfg = cfg
        root = Path(cfg["materialized_dir"]) if cfg.get("materialized_dir") else Path(cfg.get("dataset_dir", ""))
        files = sorted(root.glob("sub-*preprocessed-epo.fif"))
        if not files:
            raise FileNotFoundError(f"No .fif files found in {root}")
        sid_re = re.compile(r"sub-(\d+)_preprocessed")
        epochs_list = []
        for fp in files:
            ep = mne.read_epochs(fp, preload=True, verbose=False)
            # Optional time cropping
            ep = apply_crop_ms(ep, cfg.get("crop_ms"))
            # Optional channel selection for materialized data (named lists / include / Cz ring)
            ep = spatial_sample(
                ep,
                cfg.get("use_channel_list"),
                cfg.get("include_channels"),
                cz_step=cfg.get("cz_step"),
                cz_name=cfg.get("cz_name", "Cz"),
                channel_lists=cfg.get("channel_lists"),
            )
            m = sid_re.search(fp.name)
            if m:
                ep.metadata["subject"] = int(m.group(1))
            epochs_list.append(ep)
        # Ensure identical channel set & order across all Epochs (align subjects)
        # Rationale: downstream models require fixed channel dimensions and order
        if len(epochs_list) > 1:
            base = epochs_list[0].ch_names
            common = [ch for ch in base if all(ch in ep.ch_names for ep in epochs_list)]
            if not common:
                raise ValueError("No common channels across subjects after preprocessing.")
            aligned = []
            for ep in epochs_list:
                picks = mne.pick_channels(ep.ch_names, include=common, ordered=True)
                ep2 = ep.copy().pick(picks)
                ep2 = ep2.reorder_channels(common)
                aligned.append(ep2)
            epochs_list = aligned
        # Instead of concatenating Epochs (which is strict), build arrays/metadata manually
        # This avoids MNE concat constraints while preserving trial-level metadata
        times0 = epochs_list[0].times
        for i, ep in enumerate(epochs_list[1:], 1):
            if not np.allclose(ep.times, times0):
                raise ValueError(f"Time axis mismatch at index {i}.")
        X_np_list = [ep.get_data() for ep in epochs_list]
        md_list = [ep.metadata.reset_index(drop=True) for ep in epochs_list]
        X_np = np.concatenate(X_np_list, axis=0)
        md_all = pd.concat(md_list, ignore_index=True)

        # Apply label function to concatenated metadata and filter invalid rows
        md_all["__y"] = label_fn(md_all)
        mask = md_all["__y"].notna().to_numpy()
        if mask.sum() != len(md_all):
            X_np = X_np[mask]
            md_all = md_all.loc[mask].reset_index(drop=True)

        # Build tensors (using the manually concatenated arrays / metadata)
        X = X_np * 1e6  # V->µV for interpretability and numerical stability
        X_t = torch.from_numpy(X).float().unsqueeze(1)
        le = LabelEncoder()
        y_t = torch.from_numpy(le.fit_transform(md_all["__y"]).astype(np.int64))

        self.X = X_t
        self.y = y_t
        self.class_names = list(le.classes_)
        self.groups = md_all["subject"].values
        self._ch_names = epochs_list[0].ch_names
        self._sfreq = float(epochs_list[0].info["sfreq"]) if epochs_list[0].info else None
        self._times_ms = (times0 * 1000.0).astype(np.float32)

    @property
    def num_channels(self) -> int:
        return self.X.shape[2]

    @property
    def time_points(self) -> int:
        return self.X.shape[3]

    @property
    def channel_names(self) -> List[str]:
        return self._ch_names

    @property
    def sfreq(self) -> float:
        return self._sfreq

    @property
    def times_ms(self) -> np.ndarray:
        return self._times_ms


def make_dataset(cfg: Dict[str, Any], label_fn: Callable):
    # On-the-fly preprocessing removed by design; require materialized_dir
    if not cfg.get("materialized_dir"):
        raise ValueError("materialized_dir is required; on-the-fly preprocessing has been removed.")
    return MaterializedEpochsDataset(cfg, label_fn)


