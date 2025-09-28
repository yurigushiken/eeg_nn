"""
Per-run XAI analysis for EEG models.

Outputs:
- xai_analysis/integrated_gradients/...
- xai_analysis/integrated_gradients_topomaps/...   # IG per-fold topos (3 styles)
- xai_analysis/gradcam_heatmaps/...
- xai_analysis/gradcam_topomaps/...
- GRAND-AVERAGES for IG + Grad-TopoCAM
- consolidated_xai_report.html (IG overall + top-2 peak windows, GA heatmap, per-fold IG)
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import importlib.util

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import yaml

# Optional peak finder; if missing we skip peaks gracefully
try:
    from scipy.signal import find_peaks  # type: ignore
except Exception:
    find_peaks = None  # type: ignore

proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))


def _ensure_local_code_package() -> None:
    code_dir = proj_root / "code"
    init_file = code_dir / "__init__.py"
    if not init_file.exists():
        return
    spec = importlib.util.spec_from_file_location(
        "code", init_file, submodule_search_locations=[str(code_dir)]
    )
    if not spec or not spec.loader:
        return
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(code_dir)]  # type: ignore[attr-defined]
    sys.modules["code"] = module
    spec.loader.exec_module(module)


try:
    from code.model_builders import RAW_EEG_MODELS
except Exception:
    _ensure_local_code_package()
    from code.model_builders import RAW_EEG_MODELS

try:
    from code.datasets import MaterializedEpochsDataset as RawEEGDataset
except Exception:
    _ensure_local_code_package()
    from code.datasets import MaterializedEpochsDataset as RawEEGDataset

from utils.xai import (
    compute_and_plot_attributions,
    compute_and_plot_gradcam,
    compute_and_plot_gradcam_topomap,
)
from utils.seeding import seed_everything
import tasks as task_registry


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------- Small helpers ---------------------

def _embed_image(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _robust_vlim(vec: np.ndarray) -> tuple[float, float]:
    v = np.abs(vec[np.isfinite(vec)])
    if v.size == 0:
        return (-1.0, 1.0)
    lo = float(np.percentile(v, 2.0))
    hi = float(np.percentile(v, 98.0))
    if hi <= 0 or hi <= lo:
        hi = float(np.max(v)) if np.max(v) > 0 else 1.0
        lo = 0.0
    return (lo, hi)


# --------- montage alignment that extracts numeric IDs ----------
_NUM_RE = re.compile(r"(\d{1,3})")


def _norm(s: str) -> str:
    return s.strip().replace("\x00", "").replace("\u200b", "")


def _extract_num(name: str) -> Optional[int]:
    m = _NUM_RE.search(_norm(name))
    if not m:
        return None
    try:
        n = int(m.group(1))
        return n if 1 <= n <= 129 else None
    except Exception:
        return None


def _try_build_name_map_numeric(data_chs: List[str], mont_chs: List[str]) -> Dict[str, str]:
    mont_by_num: Dict[int, str] = {}
    for m in mont_chs:
        n = _extract_num(m)
        if n is not None and n not in mont_by_num:
            mont_by_num[n] = m
    name_map: Dict[str, str] = {}
    for d in data_chs:
        n = _extract_num(d)
        if n is not None and n in mont_by_num:
            name_map[d] = mont_by_num[n]
    return name_map


def _attach_montage_with_alignment(info: mne.Info, montage: mne.channels.DigMontage) -> Optional[mne.Info]:
    """Try direct set, then numeric-based auto-rename. Consider success if set_montage does not error."""
    try:
        tmp = info.copy()
        tmp.set_montage(montage, match_case=False, match_alias=True, on_missing="ignore")
        print(f" -> montage attached (direct match): {len(tmp['ch_names'])} channels")
        return tmp
    except Exception as e:
        print(f" -> direct montage set failed: {e}")

    data_chs = list(info["ch_names"])
    mont_chs = list(montage.ch_names)
    name_map = _try_build_name_map_numeric(data_chs, mont_chs)
    hits = len(name_map)
    if hits == 0:
        print(f" -> unable to align names numerically (0/{len(data_chs)}) matched.")
        return None

    aligned = info.copy()
    aligned.rename_channels(name_map)
    try:
        aligned.set_montage(montage, match_case=False, match_alias=True, on_missing="ignore")
        print(f" -> montage attached after numeric alignment: matched {hits}/{len(data_chs)} channels")
        return aligned
    except Exception as e:
        print(f" -> montage set failed after numeric alignment: {e}")
    print(" -> montage still not attached; topomaps will be skipped.")
    return None
# ------------------------------------------------------------------------


# --------------------- Report builder ---------------------

def _two_column_list(items: List[str]) -> str:
    """Return a neat two-column <pre> block string."""
    left = items[0::2]
    right = items[1::2]
    rows = []
    width = max((len(s) for s in left), default=0) + 4
    for i in range(len(left)):
        l = f"{2*i+1:>2}. {left[i]}"
        r = f"{2*i+2:>2}. {right[i]}" if i < len(right) else ""
        rows.append(l.ljust(width) + r)
    return "\n".join(rows)


def create_xai_report(
    run_dir: Path,
    summary_data: dict,
    grand_average_plot_path: Path,
    per_fold_plot_paths: List[Path],
    top_channels_overall: List[str],
    overall_topoplot_path: Optional[Path],
    peak_summaries: List[Tuple[str, List[str], Path]],  # [(window_label, top10 list, fig path)]
) -> None:
    """Generates the consolidated HTML report with IG overall + top-2 peak topoplots."""
    task_name_for_title = re.sub(r"(?<=\d)_(?=\d)", "-", summary_data['hyper']['task'])
    report_title = f"XAI Report: {task_name_for_title} ({summary_data['hyper']['model_name']})"
    html_output_path = run_dir / "consolidated_xai_report.html"

    top_k = int(summary_data.get('hyper', {}).get('xai_top_k_channels', 10) or 10)
    overall_block = _two_column_list(top_channels_overall[:top_k]) if top_channels_overall else "N/A"

    cards_html = "\n".join(
        f'<div class="card"><img src="{_embed_image(p)}"><p>{p.stem}</p></div>'
        for p in per_fold_plot_paths
    )

    peak_html = ""
    if peak_summaries:
        sec = []
        for window_label, top_list, fig_path in peak_summaries:
            tl = _two_column_list(top_list[:top_k]) if top_list else "N/A"
            sec.append(
                "<div class='peak-block'>"
                f"<div class='peak-left'><h3>{window_label}</h3><pre>{tl}</pre></div>"
                f"<div class='peak-right'><img src='{_embed_image(fig_path)}' alt='{fig_path.stem}'></div>"
                "</div>"
            )
        peak_html = "<div><h2>Top 2 Temporal Windows (IG)</h2>" + "".join(sec) + "</div>"

    overall_topo_html = ""
    if overall_topoplot_path and overall_topoplot_path.exists():
        overall_topo_html = (
            "<div><h2>Overall Channel Importance (IG)</h2>"
            f"<img src='{_embed_image(overall_topoplot_path)}' alt='IG overall topoplot'></div>"
        )

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        f"<title>{report_title}</title>"
        "<style>"
        "body{font-family:sans-serif;margin:2em}"
        ".container{max-width:1200px;margin:auto}"
        "h1,h2,h3{text-align:center}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1em}"
        ".card img{width:100%;border:1px solid #ddd}"
        ".card p{text-align:center;font-weight:600}"
        ".summary{display:flex;gap:2em;align-items:flex-start;justify-content:center;margin:1em 0 2em}"
        ".summary .box{flex:1}"
        ".summary pre{background:#f7f7f7;padding:1em;border-radius:6px;font-size:1.05em}"
        ".peak-block{display:flex;gap:2em;align-items:flex-start;justify-content:center;margin:1.5em 0;padding-top:1em;border-top:1px solid #eee}"
        ".peak-left{flex:1} .peak-right{flex:1} .peak-right img{width:100%;border:1px solid #ddd}"
        "</style>"
        "</head><body><div class='container'>"
        f"<h1>{report_title}</h1>"
        "<div class='summary'>"
        f"<div class='box'><h2>Top Channels (IG Overall)</h2><pre>{overall_block}</pre></div>"
        f"{overall_topo_html}"
        "</div>"
        "<div><h2>Grand Average Attribution Heatmap (IG)</h2>"
        f"<img src='{_embed_image(grand_average_plot_path)}' alt='Grand Average Heatmap' style='width:100%;border:1px solid #ddd'></div>"
        f"{peak_html}"
        "<div><h2>Per-Fold Attribution Heatmaps (IG)</h2>"
        f"<div class='grid'>{cards_html}</div></div></div></body></html>"
    )
    html_output_path.write_text(html, encoding="utf-8")
    print(f" -> consolidated XAI HTML report saved to {html_output_path}")


# --------------------- Main ---------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run XAI analysis on a completed training run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gradtopo-window", type=str, default=None,
                        help="Optional time window for Grad-TopoCAM, e.g., '150,250' (ms).")
    parser.add_argument("--target-layer", type=str, default=None,
                        help="Optional dotted path of conv layer to target for Grad-CAM/TopoCAM (e.g., 'features.4').")
    args = parser.parse_args()

    # load run summary/config
    try:
        summary_path = next(args.run_dir.glob("summary_*.json"))
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    except StopIteration:
        sys.exit(f"error: summary_*.json not found in {args.run_dir}")

    cfg = summary_data["hyper"]
    common_yaml = proj_root / "configs" / "common.yaml"
    if common_yaml.exists():
        common_cfg = yaml.safe_load(common_yaml.read_text(encoding="utf-8")) or {}
        common_cfg.update(cfg)
        cfg = common_cfg

    dataset_dir = summary_data["dataset_dir"]
    fold_splits = summary_data["fold_splits"]

    # dataset
    seed_everything(cfg.get("seed", 42))
    label_fn = task_registry.get(cfg["task"])
    dcfg = {"materialized_dir": dataset_dir, **cfg}
    full_dataset = RawEEGDataset(dcfg, label_fn)

    ch_names = full_dataset.channel_names
    sfreq = full_dataset.sfreq
    times_ms = full_dataset.times_ms

    # montage: custom .sfp -> HydroCel-129 -> align
    montage = None
    custom = proj_root / "net" / "AdultAverageNet128_v1.sfp"
    if custom.exists():
        try:
            montage = mne.channels.read_custom_montage(custom)
            print(f" -> using custom montage: {custom.name}")
        except Exception as e:
            print(f" -> failed to read custom montage ({custom.name}): {e}")
    if montage is None:
        try:
            montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
            print(" -> using standard montage: GSN-HydroCel-129")
        except Exception:
            print(" -> no montage available.")
            montage = None

    bare_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    info_for_plot = _attach_montage_with_alignment(bare_info, montage) if montage is not None else None
    if info_for_plot is None:
        print(" -> montage/digitization not attached; continuing without topomaps.")

    # outputs
    xai_root = args.run_dir / "xai_analysis"
    for d in ["integrated_gradients", "integrated_gradients_topomaps", "gradcam_heatmaps", "gradcam_topomaps"]:
        (xai_root / d).mkdir(parents=True, exist_ok=True)

    # knobs
    gt_window = None
    if args.gradtopo_window:
        try:
            t0, t1 = [float(x.strip()) for x in args.gradtopo_window.split(",")]
            gt_window = (t0, t1)
        except Exception:
            print(" -> could not parse --gradtopo-window; expected 'start_ms,end_ms'. using full window.")
            gt_window = None
    target_layer = args.target_layer
    top_k = int(cfg.get('xai_top_k_channels', 10) or 10)

    print("starting XAI analysis...")

    def _resolve_ckpt(run_dir: Path, fold: int) -> Optional[Path]:
        ckpt_dir = run_dir / "ckpt"
        candidates = [
            ckpt_dir / f"fold_{fold:02d}_refit_best.ckpt",
            run_dir / f"fold_{fold:02d}_refit_best.ckpt",
            ckpt_dir / f"fold_{fold:02d}_best.ckpt",
            run_dir / f"fold_{fold:02d}_best.ckpt",
        ]
        for c in candidates:
            if c.exists():
                return c
        inners = sorted((ckpt_dir.glob(f"fold_{fold:02d}_inner_*_best.ckpt") if ckpt_dir.exists() else []))
        if not inners:
            inners = sorted(run_dir.glob(f"fold_{fold:02d}_inner_*_best.ckpt"))
        return inners[0] if inners else None

    # per-fold
    for fold_info in fold_splits:
        fold = fold_info["fold"]
        ckpt = _resolve_ckpt(args.run_dir, fold)
        if not ckpt:
            print(f" --- skipping fold {fold:02d}: no checkpoint found ---")
            continue

        print(f"\n --- processing fold {fold:02d} (subjects: {fold_info['test_subjects']}) ---")

        model = RAW_EEG_MODELS[cfg["model_name"]](
            cfg, len(full_dataset.class_names),
            C=full_dataset.num_channels, T=full_dataset.time_points
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

        test_subjects = fold_info["test_subjects"]
        test_idx = [i for i, g in enumerate(full_dataset.groups) if g in test_subjects]

        # IG
        compute_and_plot_attributions(
            model, full_dataset, test_idx, DEVICE, xai_root, fold, args.run_dir.name,
            test_subjects, ch_names, times_ms,
            info_for_plot=info_for_plot, plot_ig_topomaps=bool(info_for_plot is not None),
        )

        # Grad-CAM heatmaps
        compute_and_plot_gradcam(
            model, full_dataset, test_idx, DEVICE, xai_root, fold, args.run_dir.name,
            test_subjects, ch_names, times_ms,
            relu_attributions=True, target_layer_name=target_layer,
        )

        # Grad-TopoCAM
        if info_for_plot is not None:
            compute_and_plot_gradcam_topomap(
                model, full_dataset, test_idx, DEVICE, xai_root, fold, args.run_dir.name,
                test_subjects, info_for_plot, times_ms,
                time_window_ms=gt_window, reduction="positive_mean",
                top_k_labels=top_k, target_layer_name=target_layer,
            )
        else:
            print(" -> Grad-TopoCAM skipped (no montage/digitization attached).")

    print("\n--- per-fold XAI complete ---")

    # -------------------- GRAND AVERAGES --------------------
    overall_topoplot_path: Optional[Path] = None
    peak_summaries: List[Tuple[str, List[str], Path]] = []
    top_channels_overall: List[str] = []

    # IG GA (heatmap + topoplots if montage attached)
    ig_npys = sorted((xai_root / "integrated_gradients").glob("fold_*_xai_attributions.npy"))
    if ig_npys:
        ig_arrays = [np.load(p) for p in ig_npys]          # each (C,T)
        ig_ga = np.mean(ig_arrays, axis=0)                 # (C,T)
        np.save(xai_root / "grand_average_xai_attributions.npy", ig_ga)

        # Heatmap
        ga_png_path = xai_root / "grand_average_xai_heatmap.png"
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(ig_ga, cmap='inferno', aspect='auto', interpolation='nearest')
        ax.set_title('grand average feature attributions (IG)')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('eeg channels')
        ax.set_yticks(np.arange(len(ch_names)))
        ax.set_yticklabels(ch_names, fontsize=6)
        xt = np.linspace(0, len(times_ms) - 1, num=10, dtype=int)
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{times_ms[i]:.0f}" for i in xt])
        plt.colorbar(im, ax=ax, label='IG attribution')
        plt.tight_layout()
        fig.savefig(ga_png_path)
        plt.close(fig)

        # Overall channel importance & labeled topoplot
        ig_ch_import = np.mean(np.abs(ig_ga), axis=1)
        top_idx = np.argsort(ig_ch_import)[::-1][:top_k]
        top_channels_overall = [ch_names[i] for i in top_idx]
        vmin, vmax = _robust_vlim(ig_ch_import)

        if info_for_plot is not None:
            from mne.viz import plot_topomap
            overall_topoplot_path = xai_root / "grand_average_xai_topoplot.png"
            fig, ax = plt.subplots(figsize=(6.8, 6.8))
            names = [""] * len(ch_names)
            for j in top_idx:
                names[j] = ch_names[j]
            im, cn = plot_topomap(ig_ch_import, info_for_plot, axes=ax, show=False,
                                  cmap='inferno', names=names, vlim=(vmin, vmax), contours=8)
            # robust linewidth
            if cn is not None:
                artists = getattr(cn, "collections", cn if isinstance(cn, (list, tuple)) else [])
                for artist in artists:
                    try:
                        artist.set_linewidth(1.0)
                    except Exception:
                        pass
            ax.set_title('mean channel importance (IG overall)', fontsize=14)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("IG intensity", rotation=90)
            fig.savefig(overall_topoplot_path, bbox_inches='tight', dpi=180)
            plt.close(fig)
        else:
            print(" -> IG overall topoplot skipped (no montage/digitization).")

        # Peak window topoplots (top-2 windows, 50 ms, ≥100 ms apart)
        if info_for_plot is not None and find_peaks is not None:
            mean_time_attr = np.mean(np.abs(ig_ga), axis=0)
            distance_samples = max(1, int(sfreq * 0.1))  # ≥100 ms separation
            peaks, _ = find_peaks(mean_time_attr, distance=distance_samples)
            if peaks.size > 0:
                scores = mean_time_attr[peaks]
                order = np.argsort(scores)[::-1][:2]
                chosen = peaks[order]
                for idx, pidx in enumerate(chosen, start=1):
                    window_ms = 50.0
                    center_ms = float(times_ms[pidx])
                    t0 = center_ms - window_ms / 2.0
                    t1 = center_ms + window_ms / 2.0
                    s0 = int(np.argmin(np.abs(times_ms - t0)))
                    s1 = int(np.argmin(np.abs(times_ms - t1)))
                    window_attr = ig_ga[:, s0:s1+1]
                    ch_imp_win = np.mean(np.abs(window_attr), axis=1)
                    top_idx_win = np.argsort(ch_imp_win)[::-1][:top_k]
                    top_list = [ch_names[i] for i in top_idx_win]
                    vminw, vmaxw = _robust_vlim(ch_imp_win)

                    from mne.viz import plot_topomap
                    out = xai_root / f"grand_average_ig_peak{idx}_topoplot_{int(t0):03d}-{int(t1):03d}ms.png"
                    fig, ax = plt.subplots(figsize=(6.8, 6.8))
                    names = [""] * len(ch_names)
                    for j in top_idx_win:
                        names[j] = ch_names[j]
                    im, cn = plot_topomap(ch_imp_win, info_for_plot, axes=ax, show=False,
                                          cmap='inferno', names=names, vlim=(vminw, vmaxw), contours=8)
                    if cn is not None:
                        artists = getattr(cn, "collections", cn if isinstance(cn, (list, tuple)) else [])
                        for artist in artists:
                            try:
                                artist.set_linewidth(1.0)
                            except Exception:
                                pass
                    ax.set_title(f'IG peak window {int(t0)}–{int(t1)} ms', fontsize=14)
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.set_ylabel("IG intensity", rotation=90)
                    fig.savefig(out, bbox_inches='tight', dpi=180)
                    plt.close(fig)

                    peak_summaries.append((f"Window {int(t0)}–{int(t1)} ms", top_list, out))
            else:
                print(" -> IG peak analysis: no peaks detected in GA timecourse.")
        elif find_peaks is None:
            print(" -> scipy not available; skipping IG peak topoplots.")
        else:
            print(" -> IG peak topoplots skipped (no montage/digitization).")
    else:
        print(" -> no IG attribution .npy files found to summarize (grand average).")
        ga_png_path = xai_root / "grand_average_xai_heatmap.png"  # placeholder path var for report call

    # Grad-TopoCAM GA
    topo_npys = sorted((xai_root / "gradcam_topomaps").glob("fold_*_gradcam_topomap.npy"))
    if topo_npys:
        vecs = [np.load(p) for p in topo_npys]  # (C,)
        topo_ga = np.mean(vecs, axis=0)        # (C,)
        np.save(xai_root / "grand_average_gradcam_topomap.npy", topo_ga)

        vmin, vmax = _robust_vlim(topo_ga)
        gt_top = np.argsort(topo_ga)[::-1][:top_k]

        def _ga_topo(filename_stem: str, variant: str) -> str:
            from mne.viz import plot_topomap
            fig, ax = plt.subplots(figsize=(6.8, 6.8))
            names = [""] * len(ch_names)
            for j in gt_top:
                names[j] = ch_names[j]
            if info_for_plot is None:
                print(" -> Grad-TopoCAM GA topomaps skipped (no montage/digitization).")
                return ""
            if variant == "default":
                im, cn = plot_topomap(topo_ga, info_for_plot, axes=ax, show=False,
                                      cmap="inferno", names=names, vlim=(vmin, vmax), contours=0)
                title = "grad-topocam (grand average)"
            elif variant == "contours":
                im, cn = plot_topomap(topo_ga, info_for_plot, axes=ax, show=False,
                                      cmap="inferno", names=names, vlim=(vmin, vmax), contours=10)
                title = "grad-topocam (grand average, contours)"
                if cn is not None:
                    artists = getattr(cn, "collections", cn if isinstance(cn, (list, tuple)) else [])
                    for artist in artists:
                        try:
                            artist.set_linewidth(1.0)
                        except Exception:
                            pass
            else:
                im, cn = plot_topomap(topo_ga, info_for_plot, axes=ax, show=False,
                                      cmap="inferno", names=names, vlim=(vmin, vmax),
                                      image_interp="nearest", contours=0, sensors=True)
                title = "grad-topocam (grand average, sensors)"
            ax.set_title(title, fontsize=14)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Grad-TopoCAM intensity", rotation=90)
            path = xai_root / f"{filename_stem}.png"
            fig.savefig(path, bbox_inches='tight', dpi=180)
            plt.close(fig)
            return str(path.name)

        if info_for_plot is not None:
            _ga_topo("grand_average_gradcam_topomap", "default")
            _ga_topo("grand_average_gradcam_topomap_contours", "contours")
            _ga_topo("grand_average_gradcam_topomap_sensors", "sensors")
    else:
        print(" -> no Grad-TopoCAM per-fold vectors found to summarize (grand average).")

    # HTML report (with IG overall + peaks)
    per_fold_ig = sorted((xai_root / "integrated_gradients").glob("fold_*_xai_heatmap.png"))
    ga_png_path = xai_root / "grand_average_xai_heatmap.png"
    if ga_png_path.exists():
        create_xai_report(
            run_dir=args.run_dir,
            summary_data=summary_data,
            grand_average_plot_path=ga_png_path,
            per_fold_plot_paths=per_fold_ig,
            top_channels_overall=top_channels_overall,
            overall_topoplot_path=overall_topoplot_path,
            peak_summaries=peak_summaries,
        )

    print("\n--- XAI summary complete ---")


if __name__ == "__main__":
    main()
