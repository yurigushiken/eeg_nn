from __future__ import annotations
from typing import Dict, Any, Tuple, List

import re
import numpy as np
import pandas as pd
import mne
from pyprep.prep_pipeline import PrepPipeline
from mne.preprocessing import ICA


def run_ica_if_enabled(raw: mne.io.Raw, cfg: dict) -> mne.io.Raw:
    """Fit ICA on a 1 Hz high-pass copy, auto-detect EOG/ECG and (optionally) ICLabel,
    then apply to the original raw. Matches current MNE guidance."""
    if not bool(cfg.get("use_ica", False)):
        return raw

    # --- prepare a copy for fitting (HP ≥ 1 Hz recommended by MNE)
    hp = float(cfg.get("ica_hp_for_fit", 1.0))
    raw_for_fit = raw.copy().filter(l_freq=hp, h_freq=None, picks='eeg', verbose=False)

    # --- choose algorithm
    method = str(cfg.get("ica_method", "picard"))
    fit_params = None
    if method == "picard":
        # Picard reproducing Extended Infomax: ortho=False, extended=True (MNE doc)
        fit_params = dict(ortho=False, extended=True)
    elif method == "infomax":
        fit_params = dict(extended=True)  # Extended Infomax
    ica = ICA(
        method=method,
        fit_params=fit_params,
        n_components=cfg.get("ica_n_components", 0.99),
        random_state=int(cfg.get("ica_random_state", 97)),
        max_iter="auto"
    )

    # Speed-up if desired
    decim = int(cfg.get("ica_decim", 1)) or 1

    # --- Use a band-passed copy for fitting ICLabel, as recommended by docs
    raw_for_fit = raw.copy().pick("eeg").filter(1., 100., fir_design="firwin", verbose=False)
    
    # --- NaN/flat channel guard before ICA fit
    raw_for_fit.load_data()
    X = raw_for_fit.get_data()
    bad_ix = np.where((~np.isfinite(X).all(axis=1)) | (np.ptp(X, axis=1) == 0))[0]
    if bad_ix.size:
        bads = [raw_for_fit.ch_names[i] for i in bad_ix]
        print(f"[ica] Found non-finite/flat channels on fitting copy: {bads}. Dropping for ICA fit.")
        raw_for_fit.drop_channels(bads)

    ica.fit(raw_for_fit, picks="eeg", decim=decim)

    # --- auto-detect blink/ECG components
    exclude = set()

    if bool(cfg.get("ica_use_eog", True)) and 'eog' in raw.get_channel_types():
        eog_inds, _ = ica.find_bads_eog(raw, threshold=3.0)  # uses corr with EOG epochs
        exclude.update(eog_inds)

    if bool(cfg.get("ica_use_ecg", True)) and 'ecg' in raw.get_channel_types():
        ecg_inds, _ = ica.find_bads_ecg(raw, method='correlation', threshold='auto')
        exclude.update(ecg_inds)

    # --- optional: ICLabel (mne-icalabel)
    if str(cfg.get("ica_labeler", "")).lower() == "iclabel":
        try:
            from mne_icalabel import label_components
            # ICLabel expects data band-passed 1-100Hz.
            raw_for_iclabel = raw.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)
            labels = label_components(raw_for_iclabel, ica, method="iclabel")
            probs = labels["y_pred_proba"]
            classes = labels["labels"]
            # Shape guard: some versions may return a 1D vector for a single component
            if probs is not None and getattr(probs, "ndim", 2) == 1:
                probs = np.atleast_2d(probs)

            def P(name):
                idx = list(classes).index(name) if name in classes else None
                if idx is None or probs is None:
                    rows = probs.shape[0] if hasattr(probs, "shape") else 0
                    return np.zeros((rows,), dtype=float)
                return probs[:, idx]
            thr = float(cfg.get("ica_label_threshold", 0.90))

            # Exclude high-probability artifact components, but not line noise
            remove_idx = np.where(
                (P("eye blink") >= thr) |
                (P("muscle artifact") >= thr) |
                (P("channel noise") >= thr) |
                (P("heart beat") >= thr)
            )[0]
            exclude.update(list(map(int, remove_idx)))
        except Exception as e:
            print(f"[ica] ICLabel unavailable or failed ({e}); skipping ICLabel auto-label.")

    # --- apply cleaning
    ica.exclude = sorted(list(exclude))
    if ica.exclude:
        print(f"[ica] excluding components: {ica.exclude}")
        ica.apply(raw)  # apply to the original (unfiltered) Raw
    else:
        print("[ica] no components excluded")

    return raw


def _eeg_names(raw: mne.io.Raw) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False, misc=False)
    return [raw.ch_names[p] for p in picks]

def remove_noise_and_bads(raw: mne.io.Raw, cfg: dict | None = None) -> mne.io.Raw:
    cfg = cfg or {}
    # Build explicit EEG ref list, excluding known non-EEG and optional user excludes
    eeg_chs = _eeg_names(raw)
    user_excl = set(map(str, cfg.get("prep_ref_exclude", [])))
    ref_chs = [ch for ch in eeg_chs if ch not in user_excl and ch.lower() != "vertex reference"]

    # Keep line noise minimal; you can make this configurable as prep_line_freqs
    line = cfg.get("prep_line_freqs", [60])  # or [50] in EU grids

    prep_params = {
        "ref_chs": ref_chs,          # explicit list
        "reref_chs": ref_chs,        # same set for re-referencing target
        "line_freqs": line,
        "max_iterations": int(cfg.get("prep_max_iterations", 4)),
    }
    # Keep default per config; set true in common.yaml if desired
    ransac = bool(cfg.get("prep_ransac", False))

    try:
        prep = PrepPipeline(raw, prep_params, raw.get_montage(), ransac=ransac, random_state=42)
        prep.fit()
        print(f"[prep] PREP succeeded. Interpolated channels: {prep.interpolated_channels}")
        return prep.raw
    except Exception as e:
        # Any PREP failure (TooManyBad, RANSAC IndexError, etc.) -> conservative fallback
        print(f"[prep] PREP failed ({type(e).__name__}: {e}). Using conservative fallback.")
        raw_fallback = raw.copy()

        # 1) Find only the most obviously broken channels (flat/NaN)
        try:
            from pyprep.find_noisy_channels import NoisyChannels
            noisy = NoisyChannels(raw_fallback, random_state=42)
            noisy.find_bad_by_nan_flat()
            raw_fallback.info['bads'] = noisy.get_bads()
        except Exception as e2:
            print(f"[prep fallback] NoisyChannels nan/flat check failed: {e2}")
            raw_fallback.info['bads'] = []

        # 2) Set average reference, automatically excluding the bads found above
        if raw_fallback.info['bads']:
            print(f"[prep fallback] Found sure-bads: {raw_fallback.info['bads']}. Excluding from reference.")
        raw_fallback.set_eeg_reference("average", projection=False, verbose=False)

        # 3) Interpolate the channels that were excluded
        if raw_fallback.info['bads']:
            print(f"[prep fallback] Interpolating sure-bads: {raw_fallback.info['bads']}")
            raw_fallback.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)

        return raw_fallback


def load_raw_mff(mff_path: str, montage_path: str):
    raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False)
    montage = mne.channels.read_custom_montage(montage_path)
    raw.set_montage(montage, match_case=False, match_alias=True, on_missing="ignore")

    # --- NEW: ensure correct channel types before PREP
    mapping = {}
    # EGI vertex ref often shows up with this exact name; treat as non-EEG
    for ch in raw.ch_names:
        name_l = ch.lower()
        if ch == "Vertex Reference" or "vertex reference" in name_l:
            mapping[ch] = "misc"  # exclude from EEG reference
        elif name_l.startswith("eog") or "eog" in name_l:
            mapping[ch] = "eog"
        elif name_l.startswith("ecg") or "ekg" in name_l:
            mapping[ch] = "ecg"
        elif "status" in name_l or "stim" in name_l:
            mapping[ch] = "stim"
    if mapping:
        raw.set_channel_types(mapping)

    # Montage geometry sanity log
    pos = raw.get_montage().get_positions() if raw.get_montage() else None
    if pos and "ch_pos" in pos:
        have_pos = [ch for ch in raw.ch_names if raw.get_channel_types(picks=ch)[0]=='eeg' and pos["ch_pos"].get(ch) is not None]
        eeg_total = sum(1 for t in raw.get_channel_types() if t=='eeg')
        print(f"[montage] EEG with 3D positions: {len(have_pos)}/{eeg_total}")

    return raw


def bandpass_filter(raw, f_lo: float, f_hi: float):
    raw.filter(l_freq=f_lo, h_freq=f_hi, method='fir', verbose=False)


def get_events_and_dict(raw, keep_labels=None):
    events, event_dict = mne.events_from_annotations(raw, verbose=False)  # doc-recommended
    if keep_labels is not None:
        event_dict = {k: v for k, v in event_dict.items() if k in keep_labels}
        # Optionally filter events to those IDs:
        keep_ids = set(event_dict.values())
        events = events[[e[2] in keep_ids for e in events]]
    return events, event_dict


def find_events_safely(raw, **kwargs):
    """Deprecated: stim fallback removed for this project. Use annotations only."""
    events, event_id = mne.events_from_annotations(raw)
    return events, event_id


def spatial_sample(epochs, use_channel_list, include_channels, cz_step=None, cz_name="Cz", channel_lists=None):
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


def find_events(raw):
    return mne.find_events(raw, stim_channel='STI 014')


def epoch_raw(raw, events, event_id, t_min, t_max):
    # Ensure event_id is a dictionary, as required by MNE
    # Deduplicate events that share the same onset sample to avoid Epochs errors
    try:
        if events is not None and len(events) > 0:
            order = np.argsort(events[:, 0], kind="stable")
            events_sorted = events[order]
            uniq_mask = np.ones(len(events_sorted), dtype=bool)
            uniq_mask[1:] = events_sorted[1:, 0] != events_sorted[:-1, 0]
            dropped = int(len(events_sorted) - uniq_mask.sum())
            if dropped > 0:
                print(f"[events] Dropping {dropped} duplicate event onsets (same sample).")
            events = events_sorted[uniq_mask]
    except Exception:
        pass

    if event_id is None:
        event_id = {str(event_code): event_code for event_code in np.unique(events[:, 2])}

    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=t_min,
            tmax=t_max,
            preload=True,
            baseline=None,
            event_repeated="drop",
            verbose=False,
        )
    except KeyError as e:
        print(f"[epochs] KeyError: Mismatched event IDs in events array and event_id dict. {e}")
        # Create a generic event_id dict as a fallback
        event_id = {str(event_code): event_code for event_code in np.unique(events[:, 2])}
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=t_min,
            tmax=t_max,
            preload=True,
            baseline=None,
            event_repeated="drop",
            verbose=False,
        )
    return epochs

def average_reference(epochs):
    epochs.set_eeg_reference('average', projection=True)


def apply_event_offset(events: np.ndarray | None, raw: mne.io.Raw, offset_ms: float | int | None):
    """Shift event onsets by offset_ms relative to raw, in milliseconds.

    Positive values move events later; negative values move earlier.
    """
    if events is None or len(events) == 0:
        return events
    try:
        off = float(offset_ms or 0.0)
    except Exception:
        off = 0.0
    if off == 0.0:
        return events
    sfreq = float(raw.info.get('sfreq', 0.0) or 0.0)
    if sfreq <= 0:
        return events
    shift = int(round(off * sfreq / 1000.0))
    if shift == 0:
        return events
    ev = events.copy()
    ev[:, 0] = np.clip(ev[:, 0] + shift, 0, int(raw.n_times) - 1)
    print(f"[events] Applied offset of {off:.1f} ms ({shift} samples) to event onsets.")
    return ev

def downsample(epochs, target_sfreq: float):
    if float(epochs.info['sfreq']) != float(target_sfreq):
        epochs.resample(target_sfreq, npad='auto')

def spatial_sample_epochs(epochs, c_step: int, cz_name: str = "Cz"):
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

def _unify_acc_columns(df: pd.DataFrame) -> pd.Series:
    acc_cols = [c for c in df.columns if ('ACC' in c) and ('Target' in c) and ('OverallAcc' not in c)]
    if not acc_cols:
        return df.get('Target.ACC', pd.Series(index=df.index, dtype=float))
    work = df.copy()
    for c in acc_cols:
        work[c] = pd.to_numeric(work[c], errors='coerce')
    unified = work[acc_cols[0]].copy()
    for c in acc_cols[1:]:
        unified = unified.fillna(work[c])
    return unified

def merge_behavior_metadata(epochs, behavior_df: pd.DataFrame, strict: bool = True):
    df = behavior_df.copy()
    # Drop practice trials and generate Trial_Continuous
    non_practice_mask = df['Procedure[Block]'] != "Practiceproc"
    df.loc[non_practice_mask, 'Trial_Continuous'] = np.arange(1, non_practice_mask.sum() + 1)
    # Remove Condition == 99
    valid_mask = df['CellNumber'].astype(str) != '99'
    df = df[valid_mask].copy()
    # Align lengths; optionally strict
    if len(epochs) != len(df):
        if strict:
            raise ValueError(f"Length mismatch after preprocessing vs behavior filtering: epochs={len(epochs)} vs behavior={len(df)}. Aborting.")
        # Best-effort alignment for smoke tests: crop both to the minimum length
        n = min(len(epochs), len(df))
        epochs = epochs[:n]
        df = df.iloc[:n].reset_index(drop=True)

    df['unified_ACC'] = _unify_acc_columns(df)

    # Derived descriptors (direction/change_group/size) matching legacy behavior
    SMALL_SET = {1,2,3}; LARGE_SET = {4,5,6}
    def direction_label(cond):
        try:
            s = str(int(cond)).zfill(2)
            a, b = int(s[0]), int(s[1])
        except Exception:
            return pd.NA
        if a == b:
            return "NC"
        return "I" if a < b else "D"
    def transition_category(cond):
        try:
            s = str(int(cond)).zfill(2)
            a, b = int(s[0]), int(s[1])
        except Exception:
            return pd.NA
        if a == b:
            return "NC"
        if a in SMALL_SET and b in SMALL_SET:
            return "iSS" if a < b else "dSS"
        if a in LARGE_SET and b in LARGE_SET:
            return "iLL" if a < b else "dLL"
        if a in SMALL_SET and b in LARGE_SET:
            return "iSL"
        if a in LARGE_SET and b in SMALL_SET:
            return "dLS"
        return pd.NA
    def size_category(cond):
        try:
            s = str(int(cond)).zfill(2)
            a, b = int(s[0]), int(s[1])
        except Exception:
            return pd.NA
        if a == b:
            return "NC"
        a_small, b_small = a in SMALL_SET, b in SMALL_SET
        if a_small and b_small:
            return "SS"
        if (not a_small) and (not b_small):
            return "LL"
        return "cross"

    df['direction'] = df['CellNumber'].apply(direction_label)
    df['change_group'] = df['CellNumber'].apply(transition_category)
    df['size'] = df['CellNumber'].apply(size_category)

    # Final metadata columns expected by label_fn
    final_md = pd.DataFrame({
        'SubjectID': df['Subject'].astype(str) if 'Subject' in df else df.get('SubjectID', ''),
        'Block': df.get('Block'),
        'Trial': df.get('Trial'),
        'Procedure': df['Procedure[Block]'],
        'Condition': df['CellNumber'],
        'Target.ACC': df['unified_ACC'],
        'Target.RT': df.get('Target.RT'),
        'Trial_Continuous': df['Trial_Continuous'],
        'direction': df['direction'],
        'change_group': df['change_group'],
        'size': df['size']
    })

    epochs.metadata = final_md.reset_index(drop=True)
    return epochs


def match_and_reorder_channels(epoch_list: List[mne.Epochs]) -> List[mne.Epochs]:
    """Ensure all Epochs share identical channel order before concatenation.

    Strategy: Use the channel order from the first epochs as canonical; for each
    subsequent epochs, reorder channels to match canonical using MNE's
    match_channel_orders.
    """
    if not epoch_list:
        return epoch_list
    canonical = epoch_list[0]
    ref_order = canonical.ch_names
    fixed = [canonical]
    for ep in epoch_list[1:]:
        if ep.ch_names == ref_order:
            fixed.append(ep)
            continue
        # reorder to match canonical (drop extras first if needed)
        drop = [ch for ch in ep.ch_names if ch not in ref_order]
        ep2 = ep.copy()
        if drop:
            ep2 = ep2.drop_channels(drop)
        common = [ch for ch in ref_order if ch in ep2.ch_names]
        ep2 = ep2.reorder_channels(common)
        fixed.append(ep2)
    return fixed

def unify_and_align_channels(epoch_list: List[mne.Epochs]) -> List[mne.Epochs]:
    """Intersect channel sets across epochs and align them to a canonical order.

    This prevents concatenation errors when different subjects end up with slightly
    different channel subsets after spatial sampling.
    """
    if not epoch_list:
        return epoch_list
    # Compute intersection of channel sets
    ch_sets = [set(ep.ch_names) for ep in epoch_list]
    common = set.intersection(*ch_sets)
    if not common:
        raise ValueError("No common channels across subjects after spatial sampling.")
    # Canonical order is the first epochs' order filtered to common
    canonical_order = [ch for ch in epoch_list[0].ch_names if ch in common]
    fixed: List[mne.Epochs] = []
    for ep in epoch_list:
        ep2 = ep.copy()
        drop = [ch for ch in ep2.ch_names if ch not in common]
        if drop:
            ep2 = ep2.drop_channels(drop)
        if ep2.ch_names != canonical_order:
            ep2 = ep2.reorder_channels(canonical_order)
        fixed.append(ep2)
    return fixed


