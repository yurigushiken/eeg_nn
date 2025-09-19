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
    load_raw_mff,
    bandpass_filter,
    remove_noise_and_bads,
    run_ica_if_enabled,
    get_events_and_dict,
    epoch_raw,
    downsample,
    spatial_sample,
    apply_crop_ms,
    apply_event_offset,
    merge_behavior_metadata,
    match_and_reorder_channels,
    unify_and_align_channels,
)
from mne import equalize_channels


def _cfg_fingerprint(cfg: Dict[str, Any]) -> str:
    """Build a cache fingerprint for epoching configuration.

    Important: include behavior-alignment knobs so changes to alignment/QC do
    not silently reuse stale caches created by older logic.
    """
    from pathlib import Path as _Path

    qc_csv = cfg.get("usable_trials_csv")
    qc_tag = _Path(qc_csv).stem if qc_csv else "none"
    strict_tag = int(bool(cfg.get("strict_behavior_align", False)))

    keys = [
        ("f_lo", cfg.get("f_lo")),
        ("f_hi", cfg.get("f_hi")),
        ("t_min", cfg.get("t_min")),
        ("t_max", cfg.get("t_max")),
        ("t1_s", cfg.get("t1_s")),
        ("crop_ms", tuple(cfg.get("crop_ms") or []) if cfg.get("crop_ms") else None),
        ("cz_step", cfg.get("cz_step")),
        ("target_sfreq", cfg.get("target_sfreq")),
        ("offset_ms", cfg.get("offset_ms", 17)),
        ("use_channel_list", cfg.get("use_channel_list")),
        ("include_channels", tuple(cfg.get("include_channels") or [])),
        # Alignment/QC related
        ("strict", strict_tag),
        ("qc", qc_tag),
        # Bump when alignment behavior changes materially
        ("align_v", 2),
        ("prep_v", 2),
        ("ica_v", 1), # <-- Add this
    ]
    return "_".join([f"{k}={v}" for k, v in keys])


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
    """Reads .fif epochs with metadata, derives labels via label_fn."""
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
        # Ensure identical channel set & order across all Epochs
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
        times0 = epochs_list[0].times
        for i, ep in enumerate(epochs_list[1:], 1):
            if not np.allclose(ep.times, times0):
                raise ValueError(f"Time axis mismatch at index {i}.")
        X_np_list = [ep.get_data() for ep in epochs_list]
        md_list = [ep.metadata.reset_index(drop=True) for ep in epochs_list]
        X_np = np.concatenate(X_np_list, axis=0)
        md_all = pd.concat(md_list, ignore_index=True)

        # Apply label function to concatenated metadata and filter
        md_all["__y"] = label_fn(md_all)
        mask = md_all["__y"].notna().to_numpy()
        if mask.sum() != len(md_all):
            X_np = X_np[mask]
            md_all = md_all.loc[mask].reset_index(drop=True)

        # Build tensors (using the manually concatenated arrays / metadata)
        X = X_np * 1e6  # V->µV
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


class OnTheFlyPreprocDataset(BaseEEGDataset):
    def __init__(self, cfg: Dict[str, Any], label_fn: Callable):
        super().__init__()
        self.cfg = cfg
        self.label_fn = label_fn

        raw_root = Path(cfg["data_raw_dir"])  # e.g., eeg-raw/
        beh_root = Path(cfg["behavior_dir"])  # e.g., data_behavior/behavior_data
        montage_path = cfg["montage_path"]
        cache_root = Path("cache") / "epochs" / _cfg_fingerprint(cfg)
        cache_root.mkdir(parents=True, exist_ok=True)

        # Epoch timing: allow legacy t_min/t_max, or derive from t1_s (0 -> t1_s)
        t_min = float(cfg.get("t_min", 0.0))
        t_max = float(cfg.get("t_max", cfg.get("t1_s", 0.6)))
        if t_max is None:
            raise ValueError("Missing epoch duration: provide either t_max or t1_s in config.")

        # Subject discovery
        subjects = sorted([p.name for p in raw_root.glob("subject*.mff") if p.is_dir()])
        subset_n = int(cfg.get("subset_n", 0) or 0)
        if subset_n > 0:
            subjects = subjects[:subset_n]

        sid_re = re.compile(r"subject(\d+)\.mff", re.IGNORECASE)
        epoch_list_per_subject: List[mne.Epochs] = []
        groups_list: List[int] = []

        for subj_dir_name in subjects:
            m = sid_re.match(subj_dir_name)
            if not m:
                continue
            sid = int(m.group(1))
            out_fp = cache_root / f"sub-{sid:02d}_preprocessed-epo.fif"

            # Initialize logging counters defensively
            events_found = 0
            n_epochs_initial = 0
            if out_fp.exists():
                ep = mne.read_epochs(out_fp, preload=True, verbose=False)
                # Populate counters for logging consistency in cached path
                try:
                    events_found = int(len(getattr(ep, 'events', []) or []))
                except Exception:
                    events_found = 0
                n_epochs_initial = len(ep)
                ch_before = len(ep.ch_names)
            else:
                mff_path = str(raw_root / subj_dir_name)
                raw = load_raw_mff(mff_path, montage_path)
                ch_before = len([t for t in raw.get_channel_types() if t == 'eeg'])
                raw = remove_noise_and_bads(raw, cfg=cfg)
                raw = run_ica_if_enabled(raw, cfg)
                bandpass_filter(raw, float(cfg.get("f_lo", 1.0)), float(cfg.get("f_hi", 45.0)))
                try:
                    events, event_id = get_events_and_dict(raw, keep_labels=cfg.get("keep_event_labels"))
                except Exception as _e_ann:
                    print(f"[subject {sid:02d}] Annotation events error: {_e_ann}. Skipping.")
                    continue
                events_found = int(0 if events is None else len(events))
                if events is None or len(events) == 0:
                    print(f"[subject {sid:02d}] No events found. Skipping.")
                    continue
                # Optional event offset (e.g., 17 ms)
                events = apply_event_offset(events, raw, cfg.get("offset_ms", 17))
                ep = epoch_raw(raw, events, event_id, t_min, t_max)
                n_epochs_initial = len(ep)
                if cfg.get("target_sfreq"):
                    downsample(ep, int(cfg.get("target_sfreq")))
                # Optional time cropping
                ep = apply_crop_ms(ep, cfg.get("crop_ms"))
                # Channel selection (supports named lists via channel_lists)
                ep = spatial_sample(
                    ep,
                    cfg.get("use_channel_list"),
                    cfg.get("include_channels"),
                    cz_step=cfg.get("cz_step"),
                    cz_name=cfg.get("cz_name", "Cz"),
                    channel_lists=cfg.get("channel_lists"),
                )
                ep.save(out_fp, overwrite=True)

            # Merge subject behavior and align trials
            beh_fp = beh_root / f"Subject{sid:02d}.csv"
            if beh_fp.exists():
                beh_df = pd.read_csv(beh_fp)
                strict = bool(cfg.get("strict_behavior_align", True))
                ep = merge_behavior_metadata(ep, beh_df, strict=strict)
            else:
                if ep.metadata is None or ep.metadata.empty:
                    meta_df = pd.DataFrame({
                        'SubjectID': [f"{sid:02d}"] * len(ep),
                        'Procedure': ["Mainproc"] * len(ep),
                        'Condition': [np.nan] * len(ep),
                    })
                    ep.metadata = meta_df

            # Optional QC CSV mask: keep only rows listed for this subject
            try:
                qc_csv = cfg.get("usable_trials_csv")
                if qc_csv:
                    qc_df = pd.read_csv(qc_csv)
                    qc_df['Subject'] = qc_df['Subject'].astype(str).str.zfill(2)
                    subj_mask = qc_df['Subject'] == f"{sid:02d}"
                    if 'Trial_Continuous' in qc_df.columns and 'keep' in qc_df.columns:
                        keep_set = set(qc_df.loc[subj_mask & (qc_df['keep'] == 1), 'Trial_Continuous'].astype(int).tolist())
                        if keep_set:
                            md = ep.metadata.reset_index(drop=True)
                            if 'Trial_Continuous' in md.columns:
                                keep_bool = md['Trial_Continuous'].astype(int).isin(keep_set).to_numpy()
                                before_qc = len(ep)
                                ep = ep[keep_bool]
                                print(f"[subject {sid:02d}] QC mask kept {len(ep)}/{before_qc} epochs.")
                    else:
                        if 'Kept_Segs_Indxs' in qc_df.columns:
                            srow = qc_df.loc[subj_mask]
                            if not srow.empty:
                                raw_inds = str(srow.iloc[0]['Kept_Segs_Indxs']).strip()
                                idxs = [int(x) for x in raw_inds.split() if str(x).isdigit()]
                                before_qc = len(ep)
                                keep_bool = pd.Series(False, index=range(before_qc))
                                for i in idxs:
                                    if 0 <= i < before_qc:
                                        keep_bool.iloc[i] = True
                                ep = ep[keep_bool.to_numpy()]
                                print(f"[subject {sid:02d}] QC mask kept {len(ep)}/{before_qc} epochs by indices.")
            except Exception as _e_qc:
                print(f"[subject {sid:02d}] QC mask skipped ({_e_qc}).")

            # Derive labels and filter invalid rows
            ep.metadata["__y"] = label_fn(ep.metadata)
            mask = ep.metadata["__y"].notna().to_numpy()
            if mask.sum() == 0:
                print(f"[subject {sid:02d}] No valid labels after filtering. Skipping.")
                continue
            if mask.sum() != len(ep):
                ep = ep[mask]

            epoch_list_per_subject.append(ep)
            groups_list.extend([sid] * len(ep))

            ch_after = len(ep.ch_names)
            print(
                f"[subject {sid:02d}] events={events_found} epochs0={n_epochs_initial} "
                f"epochs_final={len(ep)} channels_before={ch_before} channels_final={ch_after}"
            )

        if not epoch_list_per_subject:
            raise RuntimeError("No epochs were created. Check events and behavior alignment.")

        # Unify channels across subjects and align order
        epoch_list_per_subject = unify_and_align_channels(epoch_list_per_subject)

        # Build arrays and metadata
        X_np = np.concatenate([ep.get_data() for ep in epoch_list_per_subject], axis=0) * 1e6  # V->µV
        md_all = pd.concat([ep.metadata.reset_index(drop=True) for ep in epoch_list_per_subject], ignore_index=True)

        # Labels
        le = LabelEncoder()
        y_np = le.fit_transform(md_all["__y"]).astype(np.int64)

        # Tensors and attributes
        self.X = torch.from_numpy(X_np).float().unsqueeze(1)
        self.y = torch.from_numpy(y_np)
        self.class_names = list(le.classes_)
        self.groups = np.array(groups_list, dtype=int)
        self._ch_names = epoch_list_per_subject[0].ch_names
        self._sfreq = float(epoch_list_per_subject[0].info["sfreq"]) if epoch_list_per_subject[0].info else None
        self._times_ms = (epoch_list_per_subject[0].times * 1000.0).astype(np.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_all_labels(self) -> np.ndarray:
        return self.y.numpy() if isinstance(self.y, torch.Tensor) else self.y

    @property
    def num_channels(self) -> int:
        return int(self.X.shape[2])

    @property
    def time_points(self) -> int:
        return int(self.X.shape[3])

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
    if cfg.get("materialized_dir") or cfg.get("dataset_dir"):
        return MaterializedEpochsDataset(cfg, label_fn)
    return OnTheFlyPreprocDataset(cfg, label_fn)


