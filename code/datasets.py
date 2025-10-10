from __future__ import annotations
from typing import Dict, Any, Callable, Tuple, List
import hashlib
import json

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import mne
from sklearn.preprocessing import LabelEncoder

from .preprocessing.epoch_utils import (
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
    # In-process dataset cache (module-level)
    _MEM_CACHE: Dict[str, Dict[str, Any]] = {}

    def __init__(self, cfg: Dict[str, Any], label_fn: Callable):
        super().__init__()
        self.cfg = cfg
        self._min_trials = int(cfg.get("min_trials_per_class", 0) or 0)
        self._exclusion_log_path = cfg.get("exclusion_log_path")
        self.excluded_subjects: Dict[int, Dict[str, Any]] = {}
        self.exclusion_log_path: str | None = None
        self.channel_metadata: Dict[str, Any] = {}
        # Optional in-process cache toggle (per Python process)
        use_mem_cache = bool(cfg.get("dataset_cache_memory", False))

        def _canonicalize(obj: Any) -> Any:
            if isinstance(obj, (list, tuple)):
                return [_canonicalize(v) for v in obj]
            if isinstance(obj, dict):
                return {str(k): _canonicalize(obj[k]) for k in sorted(obj.keys(), key=lambda x: str(x))}
            return obj

        def _cache_key() -> str:
            label_fn_name = getattr(label_fn, "__name__", str(label_fn))
            payload = {
                "materialized_dir": str(cfg.get("materialized_dir") or cfg.get("dataset_dir") or ""),
                "crop_ms": _canonicalize(cfg.get("crop_ms")),
                "exclude_channel_list": _canonicalize(cfg.get("exclude_channel_list")),
                "include_channels": _canonicalize(cfg.get("include_channels")),
                "cz_step": int(cfg.get("cz_step") or 0),
                "cz_name": str(cfg.get("cz_name", "Cz")),
                "channel_lists": _canonicalize(cfg.get("channel_lists") or {}),
                "label_fn": label_fn_name,
            }
            return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

        key = _cache_key()
        if use_mem_cache and key in MaterializedEpochsDataset._MEM_CACHE:
            bundle = MaterializedEpochsDataset._MEM_CACHE[key]
            # Restore from RAM cache
            self.X = torch.from_numpy(np.array(bundle["X"]))
            self.y = torch.from_numpy(np.array(bundle["y"]))
            self.class_names = list(bundle["class_names"])
            self.groups = np.array(bundle["groups"])  # type: ignore[assignment]
            self._ch_names = list(bundle["ch_names"])  # type: ignore[assignment]
            self._sfreq = float(bundle["sfreq"])  # type: ignore[assignment]
            self._times_ms = np.array(bundle["times_ms"])  # type: ignore[assignment]
            return
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
                cfg.get("exclude_channel_list"),
                cfg.get("include_channels"),
                cz_step=cfg.get("cz_step"),
                cz_name=cfg.get("cz_name", "Cz"),
                channel_lists=cfg.get("channel_lists"),
            )
            m = sid_re.search(fp.name)
            if m:
                ep.metadata["subject"] = int(m.group(1))
            # Apply label function early to enable per-subject trial counting
            ep.metadata["__y"] = label_fn(ep.metadata)
            epochs_list.append(ep)
        
        # Filter subjects based on min_trials_per_class (T014)
        import csv
        filtered_epochs_list = []
        all_subject_channels: Dict[int, List[str]] = {}
        for ep in epochs_list:
            subject_id = ep.metadata["subject"].iloc[0]
            all_subject_channels[subject_id] = ep.ch_names
            # Count valid trials per class
            valid_trials = ep.metadata[ep.metadata["__y"].notna()]
            if len(valid_trials) == 0:
                # Exclude subjects with no valid trials
                self.excluded_subjects[subject_id] = {
                    "reason": "no_valid_trials",
                    "min_trials_per_class": self._min_trials,
                    "class_counts": {},
                }
                continue
            class_counts = valid_trials["__y"].value_counts().to_dict()
            
            if self._min_trials > 0 and any(count < self._min_trials for count in class_counts.values()):
                self.excluded_subjects[subject_id] = {
                    "reason": "insufficient_trials",
                    "min_trials_per_class": self._min_trials,
                    "class_counts": class_counts,
                }
            else:
                filtered_epochs_list.append(ep)
        
        epochs_list = filtered_epochs_list
        
        # Write exclusion log if configured
        if self._exclusion_log_path and self.excluded_subjects:
            self.exclusion_log_path = str(self._exclusion_log_path)
            with open(self.exclusion_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["subject_id", "reason", "min_trials_per_class", "class_counts"])
                for sid, info in self.excluded_subjects.items():
                    writer.writerow([sid, info["reason"], info["min_trials_per_class"], json.dumps(info["class_counts"])])
        
        # Ensure identical channel set & order across all Epochs (align subjects)
        # Rationale: downstream models require fixed channel dimensions and order
        if len(epochs_list) >= 1:
            # Compute intersection across ALL subjects (including those later excluded)
            base_sid = sorted(all_subject_channels.keys())[0]
            base = list(all_subject_channels[base_sid])
            common = [ch for ch in base if all(ch in chs for chs in all_subject_channels.values())]
            if not common:
                raise ValueError("No common channels across subjects after preprocessing.")
            # Store channel metadata (T014)
            self.channel_metadata = {
                "intersection": common,
                "per_subject": all_subject_channels,
            }
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

        # Optional trial filtering (e.g., keep only accurate trials)
        # Configure via cfg['trial_filter']: 'all' (default) or 'acc1'
        trial_filter = str(cfg.get("trial_filter", "all") or "all").lower()
        if trial_filter not in {"all", "acc1"}:
            raise ValueError(
                f"Unsupported trial_filter='{trial_filter}'. Use 'all' or 'acc1'."
            )
        if trial_filter == "acc1":
            # Default accuracy column produced by prepare_from_happe.py is 'Target.ACC'
            acc_col = str(cfg.get("trial_filter_acc_column", "Target.ACC"))
            if acc_col not in md_all.columns:
                raise ValueError(
                    f"Requested acc1 filtering but column '{acc_col}' not found in metadata. "
                    f"Available columns: {list(md_all.columns)}"
                )
            acc_mask = (md_all[acc_col] == 1).to_numpy()
            if acc_mask.sum() == 0:
                raise ValueError(
                    f"After applying acc1 filter on column '{acc_col}', no trials remain."
                )
            if acc_mask.sum() != len(md_all):
                X_np = X_np[acc_mask]
                md_all = md_all.loc[acc_mask].reset_index(drop=True)

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

        # Save to RAM cache for subsequent trials in the same process
        try:
            if use_mem_cache:
                MaterializedEpochsDataset._MEM_CACHE[key] = {
                    "X": self.X.numpy(),
                    "y": self.y.numpy(),
                    "groups": np.array(self.groups),
                    "ch_names": list(self._ch_names),
                    "sfreq": float(self._sfreq),
                    "times_ms": np.array(self._times_ms),
                    "class_names": list(self.class_names),
                }
        except Exception:
            pass

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


