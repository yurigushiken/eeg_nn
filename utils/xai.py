"""
XAI utilities for EEG models.

Writes:
- Integrated Gradients (IG)                 → xai_analysis/integrated_gradients/
- IG topomaps (3 styles, per fold; robust) → xai_analysis/integrated_gradients_topomaps/
- Grad-CAM heatmaps                        → xai_analysis/gradcam_heatmaps/
- Grad-TopoCAM topomaps (3 styles)        → xai_analysis/gradcam_topomaps/

Note:
If Grad-CAM heatmaps show vertical bands only, choose an earlier conv with
`target_layer_name` so the layer still preserves [channels × time].
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from captum.attr import IntegratedGradients
from captum.attr import LayerGradCam, LayerAttribution

import mne
from mne.viz import plot_topomap


# -------------------------------
# helpers / wrappers
# -------------------------------

class SqueezeAndForward(torch.nn.Module):
    """Allow (B,1,C,T) or (B,C,T) without changing model code."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        return self.model(x)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_last_conv_layer(module: nn.Module) -> Optional[nn.Module]:
    """Fallback: last conv-like layer we can find (Conv1d/2d/3d or class name contains 'conv')."""
    last = None
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            last = m
        else:
            if "conv" in m.__class__.__name__.lower():
                last = m
    return last


def _find_named_layer(module: nn.Module, dotted_path: str) -> Optional[nn.Module]:
    """Resolve 'features.4' or 'encoder.block2' into a submodule; return None if not found."""
    cur = module
    for part in dotted_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur if isinstance(cur, nn.Module) else None


def _robust_vlim(vec: np.ndarray) -> Tuple[float, float]:
    """Robust color scaling for topomap: [2nd, 98th] percentiles of |vec|."""
    v = np.abs(vec[np.isfinite(vec)])
    if v.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(v, 2.0))
    hi = float(np.percentile(v, 98.0))
    if hi <= 0 or hi <= lo:
        hi = float(np.max(v)) if np.max(v) > 0 else 1.0
        lo = 0.0
    return lo, hi


def _add_colorbar(im, ax):
    """Attach a colorbar to a topomap axes."""
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("intensity", rotation=90)
    return cbar


def _info_has_positions(info: mne.Info) -> bool:
    """Return True if info has non-zero 3D locations for multiple channels."""
    try:
        chs = info["chs"]
    except Exception:
        return False
    count = 0
    for ch in chs:
        loc = ch.get("loc", None)
        if loc is None or len(loc) < 3:
            continue
        xyz = np.asarray(loc[:3], dtype=float)
        if np.any(np.isfinite(xyz)) and not np.allclose(xyz, 0.0):
            count += 1
    return count >= max(8, int(0.5 * len(info.get("ch_names", []))))


# -------------------------------
# Integrated Gradients (per-fold heatmap + per-fold topomap)
# -------------------------------

def compute_and_plot_attributions(
    model: nn.Module,
    dataset,
    test_indices: Sequence[int],
    device: torch.device,
    output_dir: Path,
    fold_num: int,
    run_dir_name: str,
    test_subjects: Sequence[str] | Sequence[int],
    ch_names: list,
    times_ms: np.ndarray,
    info_for_plot: Optional[mne.Info] = None,
    plot_ig_topomaps: bool = True,
) -> None:
    """
    IG on correctly predicted test trials; save npy/png/json under integrated_gradients/.
    Optionally also draws per-fold IG topomaps (needs montage in info_for_plot).
    """
    ig_dir = output_dir / "integrated_gradients"
    _ensure_dir(ig_dir)

    model.eval()
    model_wrapper = SqueezeAndForward(model).to(device)
    ig = IntegratedGradients(model_wrapper)

    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=16, shuffle=False)

    all_attr = []
    total_correct = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_wrapper(inputs)
        predicted = torch.argmax(outputs, dim=1)

        # keep only correct trials
        mask = (predicted == labels)
        if not torch.any(mask):
            continue

        x = inputs[mask]
        y = labels[mask]
        total_correct += int(x.size(0))

        x.requires_grad_()
        attr = ig.attribute(x, target=y, internal_batch_size=x.size(0))
        all_attr.append(attr.detach().cpu().numpy())

    if not all_attr:
        print(f" -> no correct predictions for IG on fold {fold_num}. skipping.")
        return

    avg_attr = np.mean(np.concatenate(all_attr, axis=0), axis=0)  # (1,C,T) or (C,T)
    avg_attr = np.squeeze(avg_attr)  # (C,T)

    # save array
    npy_path = ig_dir / f"fold_{fold_num:02d}_xai_attributions.npy"
    np.save(npy_path, avg_attr)

    # heatmap
    png_path = ig_dir / f"fold_{fold_num:02d}_xai_heatmap.png"
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(avg_attr, cmap='inferno', aspect='auto', interpolation='nearest')
    ax.set_title(f'mean feature attributions (IG, fold {fold_num:02d})')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('eeg channels')
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=6)
    xt = np.linspace(0, len(times_ms) - 1, num=10, dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{times_ms[i]:.0f}" for i in xt])
    plt.colorbar(im, ax=ax, label='IG attribution')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    # summary json
    summary_path = ig_dir / f"fold_{fold_num:02d}_xai_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_dir": run_dir_name,
            "fold": fold_num,
            "test_subjects": list(map(str, test_subjects)),
            "xai_method": "Integrated Gradients",
            "num_correct_trials_explained": total_correct,
            "attribution_map_shape": list(avg_attr.shape),
            "attribution_data_file": npy_path.name,
            "heatmap_image_file": png_path.name
        }, f, indent=2)

    print(f" -> IG outputs for fold {fold_num:02d} saved to {ig_dir}")

    # Optional per-fold IG topomaps (needs montage / digitization)
    if plot_ig_topomaps:
        if info_for_plot is None or not _info_has_positions(info_for_plot):
            print(" -> IG topomap skipped (no channel positions available in info).")
            return
        ig_topo_dir = output_dir / "integrated_gradients_topomaps"
        _ensure_dir(ig_topo_dir)
        ch_weights = np.mean(np.abs(avg_attr), axis=1)  # (C,)
        base = ig_topo_dir / f"fold_{fold_num:02d}_ig_topomap"
        top_k_labels = 10
        top_idx = np.argsort(ch_weights)[::-1][:top_k_labels]
        _plot_topomap_variants(
            ch_weights, info_for_plot, base, top_idx,
            f"(IG, fold {fold_num:02d}, abs-mean)"
        )
        print(f" -> IG topomap outputs for fold {fold_num:02d} saved to {ig_topo_dir}")


# -------------------------------
# Grad-CAM (heatmaps)
# -------------------------------

def compute_and_plot_gradcam(
    model: nn.Module,
    dataset,
    test_indices: Sequence[int],
    device: torch.device,
    output_dir: Path,
    fold_num: int,
    run_dir_name: str,
    test_subjects: Sequence[str] | Sequence[int],
    ch_names: list,
    times_ms: np.ndarray,
    relu_attributions: bool = True,
    target_layer_name: Optional[str] = None,
) -> None:
    """Grad-CAM heatmaps; save under gradcam_heatmaps/."""
    gc_dir = output_dir / "gradcam_heatmaps"
    _ensure_dir(gc_dir)

    model.eval()
    model_wrapper = SqueezeAndForward(model).to(device)

    # pick target conv
    target_layer = (_find_named_layer(model_wrapper, target_layer_name)
                    if target_layer_name else _find_last_conv_layer(model_wrapper))
    if target_layer is None and target_layer_name:
        print(f" -> target layer '{target_layer_name}' not found; falling back to last conv.")
        target_layer = _find_last_conv_layer(model_wrapper)
    if target_layer is None:
        print(" -> no convolutional layer found for grad-cam. skipping.")
        return

    gc = LayerGradCam(model_wrapper, target_layer)

    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=16, shuffle=False)
    all_attr = []
    total_correct = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_wrapper(inputs)
        preds = torch.argmax(outputs, dim=1)
        mask = (preds == labels)
        if not torch.any(mask):
            continue
        x = inputs[mask]
        y = labels[mask]
        total_correct += int(x.size(0))

        attr = gc.attribute(x, target=y, relu_attributions=relu_attributions)
        # upsample to (C,T)
        try:
            attr = LayerAttribution.interpolate(attr, x.shape[-2:])
        except Exception:
            pass
        if attr.dim() == 4 and attr.size(1) == 1:
            attr = attr.squeeze(1)  # (B,C,T)
        all_attr.append(attr.detach().cpu().numpy())

    if not all_attr:
        print(f" -> no correct predictions for grad-cam on fold {fold_num}. skipping.")
        return

    avg_attr = np.mean(np.concatenate(all_attr, axis=0), axis=0)  # (C,T)

    # save array
    npy_path = gc_dir / f"fold_{fold_num:02d}_gradcam_attributions.npy"
    np.save(npy_path, avg_attr)

    # heatmap
    png_path = gc_dir / f"fold_{fold_num:02d}_gradcam_heatmap.png"
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(avg_attr, cmap="magma", aspect="auto", interpolation="nearest")
    ax.set_title(f"grad-cam (fold {fold_num:02d})")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("eeg channels")
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=6)
    xt = np.linspace(0, len(times_ms) - 1, num=10, dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{times_ms[i]:.0f}" for i in xt])
    plt.colorbar(im, ax=ax, label="grad-cam intensity")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    # summary json
    summary_path = gc_dir / f"fold_{fold_num:02d}_gradcam_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_dir": run_dir_name,
            "fold": fold_num,
            "test_subjects": list(map(str, test_subjects)),
            "xai_method": "Grad-CAM (heatmap)",
            "num_correct_trials_explained": total_correct,
            "attribution_map_shape": list(avg_attr.shape),
            "attribution_data_file": npy_path.name,
            "heatmap_image_file": png_path.name,
            "target_layer": target_layer_name or "last_conv"
        }, f, indent=2)

    print(f" -> Grad-CAM heatmap outputs for fold {fold_num:02d} saved to {gc_dir}")


# -------------------------------
# Topomap helpers (robust, with contours + colorbar)
# -------------------------------

def _plot_topomap_variants(
    ch_weights: np.ndarray,
    info_for_plot: mne.Info,
    out_base: Path,
    top_idx: np.ndarray,
    title_suffix: str,
):
    """Save three styles: default, contours, sensors, all with robust vlim + colorbar."""
    # label only top-k channels by naming those indices
    names = [""] * len(info_for_plot["ch_names"])
    for i in top_idx:
        names[i] = info_for_plot["ch_names"][i]

    vmin, vmax = _robust_vlim(ch_weights)
    if vmax <= vmin:  # fallback if the vector is (near) constant
        eps = 1e-12
        vmin, vmax = float(np.min(ch_weights) - eps), float(np.max(ch_weights) + eps)

    # 1) default smooth
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    im, cn = plot_topomap(
        ch_weights, info_for_plot, axes=ax, show=False, cmap="inferno",
        names=names, vlim=(vmin, vmax), contours=0, sensors=False, image_interp="linear"
    )
    ax.set_title(f'topomap {title_suffix}', fontsize=14)
    _add_colorbar(im, ax)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches='tight', dpi=180)
    plt.close(fig)

    # 2) contours (publication style)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    im, cn = plot_topomap(
        ch_weights, info_for_plot, axes=ax, show=False, cmap="inferno",
        names=names, vlim=(vmin, vmax), contours=8, sensors=False, image_interp="linear"
    )
    ax.set_title(f'topomap (contours) {title_suffix}', fontsize=14)
    try:
        if hasattr(cn, "collections") and cn.collections is not None:
            for c in cn.collections:
                try:
                    c.set_linewidth(1.0)
                except Exception:
                    pass
    except Exception:
        pass
    _add_colorbar(im, ax)
    fig.savefig(out_base.with_name(out_base.name + "_contours").with_suffix(".png"), bbox_inches='tight', dpi=180)
    plt.close(fig)

    # 3) sensors / nearest (sanity-check)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    im, cn = plot_topomap(
        ch_weights, info_for_plot, axes=ax, show=False, cmap="inferno",
        names=names, vlim=(vmin, vmax), contours=0, sensors=True, image_interp="nearest"
    )
    ax.set_title(f'topomap (sensors) {title_suffix}', fontsize=14)
    _add_colorbar(im, ax)
    fig.savefig(out_base.with_name(out_base.name + "_sensors").with_suffix(".png"), bbox_inches='tight', dpi=180)
    plt.close(fig)


# -------------------------------
# Grad-TopoCAM (Grad-CAM → topomap)
# -------------------------------

def compute_and_plot_gradcam_topomap(
    model: nn.Module,
    dataset,
    test_indices: Sequence[int],
    device: torch.device,
    output_dir: Path,
    fold_num: int,
    run_dir_name: str,
    test_subjects: Sequence[str] | Sequence[int],
    info_for_plot: mne.Info,
    times_ms: np.ndarray,
    time_window_ms: Optional[Tuple[float, float]] = None,
    reduction: str = "positive_mean",  # "positive_mean" | "abs_mean"
    top_k_labels: int = 10,
    target_layer_name: Optional[str] = None,
) -> None:
    """
    Grad-CAM → reduce over time to per-channel weights → three topomap variants.
    Files under gradcam_topomaps/:
      - fold_XX_gradcam_topomap(.png/.npy/.json)  +  _contours.png  +  _sensors.png
    """
    topo_dir = output_dir / "gradcam_topomaps"
    _ensure_dir(topo_dir)

    model.eval()
    model_wrapper = SqueezeAndForward(model).to(device)

    # pick target conv
    target_layer = (_find_named_layer(model_wrapper, target_layer_name)
                    if target_layer_name else _find_last_conv_layer(model_wrapper))
    if target_layer is None and target_layer_name:
        print(f" -> target layer '{target_layer_name}' not found; falling back to last conv.")
        target_layer = _find_last_conv_layer(model_wrapper)
    if target_layer is None:
        print(" -> no convolutional layer found for grad-topocam. skipping.")
        return

    gc = LayerGradCam(model_wrapper, target_layer)

    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=16, shuffle=False)
    all_attr = []
    total_correct = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_wrapper(inputs)
        preds = torch.argmax(outputs, dim=1)
        mask = (preds == labels)
        if not torch.any(mask):
            continue
        x = inputs[mask]
        y = labels[mask]
        total_correct += int(x.size(0))

        attr = gc.attribute(x, target=y, relu_attributions=True)
        try:
            attr = LayerAttribution.interpolate(attr, x.shape[-2:])
        except Exception:
            pass
        if attr.dim() == 4 and attr.size(1) == 1:
            attr = attr.squeeze(1)  # (B,C,T)
        all_attr.append(attr.detach().cpu().numpy())

    if not all_attr:
        print(f" -> no correct predictions for grad-topocam on fold {fold_num}. skipping.")
        return

    avg_attr = np.mean(np.concatenate(all_attr, axis=0), axis=0)  # (C,T)

    # select time window → collapse to channel weights
    if time_window_ms is None:
        s, e = 0, avg_attr.shape[1] - 1
        win_str = "full"
    else:
        t0, t1 = float(time_window_ms[0]), float(time_window_ms[1])
        i0 = int(np.argmin(np.abs(times_ms - t0)))
        i1 = int(np.argmin(np.abs(times_ms - t1)))
        if i1 < i0:
            i0, i1 = i1, i0
        s, e = i0, i1
        win_str = f"{t0:.0f}-{t1:.0f}ms"

    windowed = avg_attr[:, s:e + 1]  # (C, Tw)
    ch_weights = np.mean(np.abs(windowed), axis=1) if reduction == "abs_mean" else np.maximum(windowed, 0).mean(axis=1)

    # save vector
    base = topo_dir / f"fold_{fold_num:02d}_gradcam_topomap"
    np.save(base.with_suffix(".npy"), ch_weights)

    # top-k labels for readability
    top_idx = np.argsort(ch_weights)[::-1][:max(1, top_k_labels)]

    # three visual variants (robust scaled + contours + colorbar)
    _plot_topomap_variants(
        ch_weights, info_for_plot, base, top_idx,
        f"(Grad-TopoCAM, fold {fold_num:02d}, {win_str})"
    )

    # summary json
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_dir": run_dir_name,
            "fold": fold_num,
            "test_subjects": list(map(str, test_subjects)),
            "xai_method": "Grad-TopoCAM",
            "time_window_ms": (None if time_window_ms is None else [float(time_window_ms[0]), float(time_window_ms[1])]),
            "reduction": reduction,
            "vector_file": base.with_suffix(".npy").name,
            "image_files": [
                base.with_suffix(".png").name,
                base.with_name(base.name + "_contours").with_suffix(".png").name,
                base.with_name(base.name + "_sensors").with_suffix(".png").name,
            ],
            "target_layer": target_layer_name or "last_conv"
        }, f, indent=2)

    print(f" -> Grad-TopoCAM outputs for fold {fold_num:02d} saved to {topo_dir}")
