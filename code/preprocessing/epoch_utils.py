from __future__ import annotations
from typing import Any

import numpy as np
import mne


def apply_crop_ms(epochs: mne.Epochs, crop_ms):
    """Crop epochs to a time window in milliseconds, if provided.

    crop_ms: [start_ms, end_ms] or None
    """
    if not crop_ms:
        return epochs
    try:
        start_ms, end_ms = float(crop_ms[0]), float(crop_ms[1])
    except Exception:
        return epochs
    tmin = start_ms / 1000.0
    tmax = end_ms / 1000.0
    print(f"[crop] Applying time cropping from {start_ms:.0f}ms to {end_ms:.0f}ms ({tmin:.3f}s to {tmax:.3f}s)")
    return epochs.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)


def spatial_sample(
    epochs: mne.Epochs,
    use_channel_list: Any,
    include_channels: Any,
    cz_step: int | None = None,
    cz_name: str = "Cz",
    channel_lists: dict | None = None,
):
    """Apply optional channel exclusion/inclusion, then optional Cz-centric sampling.

    Order of operations:
      1) Exclude channels by named list (use_channel_list)
      2) Keep-only explicit include_channels (if provided)
      3) Else, optionally apply Cz ring heuristic via cz_step
    """
    ep = epochs

    # 1) Exclusion by named list
    excl_names = []
    if use_channel_list:
        if isinstance(use_channel_list, str) and isinstance(channel_lists, dict):
            excl_names = list(channel_lists.get(use_channel_list, []))
        elif isinstance(use_channel_list, (list, tuple)):
            excl_names = list(use_channel_list)
        elif use_channel_list is True and isinstance(channel_lists, dict) and "non_scalp" in channel_lists:
            excl_names = list(channel_lists.get("non_scalp", []))
    if excl_names:
        drop = [ch for ch in excl_names if ch in ep.ch_names]
        if drop:
            ep = ep.copy().drop_channels(drop)

    # 2) Keep-only explicit include list
    if include_channels:
        wanted = [ch for ch in include_channels if ch in ep.ch_names]
        if wanted:
            picks = mne.pick_channels(ep.ch_names, include=wanted, ordered=True)
            ep = ep.copy().pick(picks).reorder_channels(wanted)
        return ep

    # 3) Optional Cz-based spatial sampling
    step = int(cz_step or 0)
    if step > 0:
        return spatial_sample_epochs(ep, step, cz_name=cz_name)
    return ep


def spatial_sample_epochs(epochs: mne.Epochs, c_step: int, cz_name: str = "Cz"):
    """Select a Cz-centric ring of channels based on montage 3D positions.

    Heuristic: map c_step to a fraction of channels to keep around Cz.
    Falls back to no-op if positions or Cz are unavailable.
    """
    chs = epochs.ch_names
    if cz_name not in chs or int(c_step or 0) <= 0:
        return epochs

    montage = epochs.get_montage()
    if montage is None:
        return epochs
    try:
        positions = montage.get_positions() or {}
        pos_dict = positions.get("ch_pos", {}) if isinstance(positions, dict) else {}
    except Exception:
        pos_dict = {}
    if not isinstance(pos_dict, dict) or cz_name not in pos_dict:
        return epochs

    cz_pos = np.array(pos_dict.get(cz_name))
    dists = []
    for name in chs:
        p = pos_dict.get(name)
        if p is None:
            d = np.inf
        else:
            d = np.linalg.norm(cz_pos - np.array(p))
        dists.append((name, d))
    dists_sorted = sorted(dists, key=lambda x: x[1])
    frac = min(1.0, max(0.1, int(c_step) * 0.2))
    k = max(1, int(round(len(dists_sorted) * frac)))
    keep = [name for name, _ in dists_sorted[:k]]
    drop = [ch for ch in chs if ch not in keep]
    if drop:
        epochs = epochs.copy().drop_channels(drop)
    return epochs


