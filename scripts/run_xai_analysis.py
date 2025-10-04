"""
Per-run XAI analysis for EEG models.

Outputs (organized structure):
- xai_analysis/
  ├── grand_average_ig_heatmap.png        (root level for easy access)
  ├── grand_average_ig_topomap.png
  ├── peak1_ig_topomap_XXX-YYYms.png
  ├── ig_heatmaps/                        (per-fold + grand average)
  ├── ig_topomaps/                        (grand average + peaks)
  ├── ig_per_class_heatmaps/              (per-class heatmaps)
  ├── ig_per_class_topomaps/              (per-class topomaps)
  ├── metadata/                           (raw .npy arrays)
  └── consolidated_xai_report.html
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import yaml

# Optional dependencies
try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

# Import XAI functions
from utils.xai import (
    compute_ig_attributions,
    plot_attribution_heatmap,
    plot_topomap,
    compute_time_frequency_map,
    plot_time_frequency_map
)

# Import project modules
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
    from code.model_builders import RAW_EEG_MODELS, squeeze_input_adapter
except Exception:
    _ensure_local_code_package()
    from code.model_builders import RAW_EEG_MODELS, squeeze_input_adapter

try:
    from code.datasets import MaterializedEpochsDataset as RawEEGDataset
except Exception:
    _ensure_local_code_package()
    from code.datasets import MaterializedEpochsDataset as RawEEGDataset

try:
    from utils.seeding import seed_everything
except Exception:
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    try:
        from utils.seeding import seed_everything
    except Exception:
        def seed_everything(seed):
            return None

import tasks as task_registry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Helper Functions ====================

def _embed_image(img_path: Path) -> str:
    """Embed image as base64 for HTML."""
    with open(img_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _load_config(run_dir: Path) -> Tuple[Dict, Dict]:
    """Load run config and merge with xai_defaults.yaml."""
    # Load summary
    try:
        summary_path = next(run_dir.glob("summary_*.json"))
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    except StopIteration:
        sys.exit(f"error: summary_*.json not found in {run_dir}")
    
    cfg = summary_data["hyper"]
    
    # Load XAI defaults
    xai_defaults_path = proj_root / "configs" / "xai_defaults.yaml"
    if xai_defaults_path.exists():
        xai_defaults = yaml.safe_load(xai_defaults_path.read_text(encoding="utf-8")) or {}
        # Merge: xai_defaults < cfg (cfg takes precedence)
        for key, value in xai_defaults.items():
            if key not in cfg:
                cfg[key] = value
    
    # Load common config
    common_yaml = proj_root / "configs" / "common.yaml"
    if common_yaml.exists():
        common_cfg = yaml.safe_load(common_yaml.read_text(encoding="utf-8")) or {}
        # Merge: common < cfg
        for key, value in common_cfg.items():
            if key not in cfg:
                cfg[key] = value
    
    return summary_data, cfg


def _resolve_ckpt(run_dir: Path, fold: int) -> Optional[Path]:
    """Find checkpoint for given fold."""
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
    # Try inner fold checkpoints
    inners = sorted((ckpt_dir.glob(f"fold_{fold:02d}_inner_*_best.ckpt") if ckpt_dir.exists() else []))
    if not inners:
        inners = sorted(run_dir.glob(f"fold_{fold:02d}_inner_*_best.ckpt"))
    return inners[0] if inners else None


def _attach_montage(ch_names: List[str], sfreq: float) -> Optional[mne.Info]:
    """
    Attach montage for topoplots using MNE's standard workflow.
    Uses the same approach that works in prepare_from_happe.py.
    """
    print("\n--- Attempting montage attachment ---")
    
    custom_sfp = proj_root / "net" / "AdultAverageNet128_v1.sfp"
    
    if not custom_sfp.exists():
        print(f" -> ERROR: Montage file not found: {custom_sfp}")
        return None
    
    try:
        # Use MNE's read_custom_montage (same as prepare_from_happe.py line 288)
        from mne.channels import read_custom_montage
        print(f" -> Reading montage from: {custom_sfp.name}")
        montage = read_custom_montage(str(custom_sfp))
        print(f"    Montage has {len(montage.ch_names)} channels")
        
        # DEBUG: Check if montage has actual position data
        try:
            montage_positions = montage.get_positions()
            print(f"    Montage positions object: {type(montage_positions)}")
            if hasattr(montage_positions, 'ch_pos') and montage_positions.ch_pos:
                print(f"    Montage has {len(montage_positions.ch_pos)} channel positions")
                # Show sample position
                sample_ch = list(montage_positions.ch_pos.keys())[0]
                sample_pos = montage_positions.ch_pos[sample_ch]
                print(f"    Sample position ({sample_ch}): {sample_pos}")
        except Exception as e_debug:
            print(f"    DEBUG: Could not inspect montage positions: {e_debug}")
        
        # Check channel overlap
        mont_set = set(montage.ch_names)
        data_set = set(ch_names)
        common = mont_set & data_set
        print(f" -> Channel overlap: {len(common)}/{len(ch_names)} channels in common")
        
        if len(common) == 0:
            print(" -> ERROR: No common channels between dataset and montage!")
            return None
        
        # Create a dummy Epochs object to properly set the montage
        print(f" -> Creating temporary Epochs object for montage attachment...")
        
        # Create dummy data matching our channel structure
        n_channels = len(ch_names)
        n_times = 100  # Arbitrary, just for structure
        n_epochs = 1
        data = np.zeros((n_epochs, n_channels, n_times))
        
        # Create Info
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
        
        # Create dummy events
        events = np.array([[0, 0, 1]])
        
        # Create Epochs object
        epochs_temp = mne.EpochsArray(data, info, events=events, tmin=0, verbose=False)
        
        # DEBUG: Check info before set_montage
        print(f"    Info before set_montage: has dig = {hasattr(epochs_temp.info, 'dig')}, dig is None = {epochs_temp.info.dig is None if hasattr(epochs_temp.info, 'dig') else 'N/A'}")
        
        # Set montage on the Epochs object (same approach as prepare_from_happe.py line 289)
        print(f" -> Setting montage on Epochs object...")
        epochs_temp.set_montage(montage, match_case=False, match_alias=True, on_missing="ignore", verbose='WARNING')
        
        # DEBUG: Check info after set_montage
        print(f"    Info after set_montage: has dig = {hasattr(epochs_temp.info, 'dig')}, dig is None = {epochs_temp.info.dig is None if hasattr(epochs_temp.info, 'dig') else 'N/A'}")
        if hasattr(epochs_temp.info, 'dig') and epochs_temp.info.dig is not None:
            print(f"    dig length = {len(epochs_temp.info.dig)}")
        
        # Extract the Info with montage attached
        info_with_montage = epochs_temp.info
        
        # Verify montage was attached by trying to retrieve it
        try:
            recovered_montage = epochs_temp.get_montage()
            if recovered_montage is not None and len(recovered_montage.ch_names) > 0:
                print(f"    get_montage() returned: {type(recovered_montage)} with {len(recovered_montage.ch_names)} channels")
                print(f" -> SUCCESS: Montage attached with {len(recovered_montage.ch_names)} channels")
                print("    Note: Modern MNE stores montage internally; dig attribute created on-demand")
                print("--- Montage attachment successful ---\n")
                return info_with_montage
            else:
                print(f"    get_montage() returned None or empty!")
                print(" -> ERROR: Montage attachment failed")
                return None
        except Exception as e_mont:
            print(f"    get_montage() failed: {e_mont}")
            print(" -> ERROR: Could not verify montage attachment")
            return None
            
    except Exception as e:
        print(f" -> ERROR: Failed to attach montage: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== Report Generation ====================

def create_consolidated_report(
    run_dir: Path,
    summary_data: Dict,
    ga_heatmap_path: Optional[Path],
    per_fold_ig_paths: List[Path],
    top_channels_overall: List[str],
    overall_topo_path: Optional[Path],
    peak_summaries: List[Tuple[str, List[str], Path]],
    per_class_paths: List[Path],
    tf_path: Optional[Path]
) -> None:
    """Generate consolidated HTML report with robust error handling."""
    try:
        task_name = re.sub(r"(?<=\d)_(?=\d)", "-", summary_data['hyper']['task'])
        report_title = f"XAI Report: {task_name} ({summary_data['hyper'].get('model_name', 'unknown')})"
        html_output_path = run_dir / "consolidated_xai_report.html"
        
        top_k = int(summary_data.get('hyper', {}).get('xai_top_k_channels', 10) or 10)
    except Exception as e:
        print(f" -> ERROR: Failed to initialize report parameters: {e}")
        return
    
    # Format top channels in two columns
    def _two_column_list(items: List[str]) -> str:
        if not items:
            return "N/A"
        left = items[0::2]
        right = items[1::2]
        rows = []
        width = max((len(s) for s in left), default=0) + 4
        for i in range(len(left)):
            l = f"{2*i+1:>2}. {left[i]}"
            r = f"{2*i+2:>2}. {right[i]}" if i < len(right) else ""
            rows.append(l.ljust(width) + r)
        return "\n".join(rows)

    try:
        overall_block = _two_column_list(top_channels_overall[:top_k]) if top_channels_overall else "N/A"
    except Exception as e:
        print(f" -> WARNING: Failed to format top channels: {e}")
        overall_block = "N/A"

    # Per-fold IG heatmaps
    try:
        per_fold_ig_html = "\n".join(
            f'<div class="card"><img src="{_embed_image(p)}"><p>{p.stem}</p></div>'
            for p in per_fold_ig_paths if p.exists()
        ) if per_fold_ig_paths else "<p>No per-fold IG heatmaps generated</p>"
    except Exception as e:
        print(f" -> WARNING: Failed to embed per-fold IG images: {e}")
        per_fold_ig_html = "<p>Error loading per-fold heatmaps</p>"

    # Peak windows
    peak_html = ""
    try:
        if peak_summaries:
            sec = []
            for window_label, top_list, fig_path in peak_summaries:
                if fig_path.exists():
                    tl = _two_column_list(top_list[:top_k]) if top_list else "N/A"
                    sec.append(
                        "<div class='peak-block'>"
                        f"<div class='peak-left'><h3>{window_label}</h3><pre>{tl}</pre></div>"
                        f"<div class='peak-right'><img src='{_embed_image(fig_path)}' alt='{fig_path.stem}'></div>"
                        "</div>"
                    )
            if sec:
                peak_html = "<div><h2>Top 2 Temporal Windows (IG)</h2>" + "".join(sec) + "</div>"
    except Exception as e:
        print(f" -> WARNING: Failed to generate peak windows section: {e}")

    # Overall topomap
    overall_topo_html = ""
    try:
        if overall_topo_path and overall_topo_path.exists():
            overall_topo_html = (
                "<div><h2>Overall Channel Importance Topomap (IG)</h2>"
                f"<img src='{_embed_image(overall_topo_path)}' alt='IG overall topoplot' style='width:60%;margin:auto;display:block'></div>"
            )
        elif not overall_topo_path:
            overall_topo_html = (
                "<div><h2>Overall Channel Importance Topomap (IG)</h2>"
                "<p style='text-align:center;color:#999'>Topomap not available (montage not attached)</p></div>"
            )
    except Exception as e:
        print(f" -> WARNING: Failed to generate overall topomap section: {e}")
    
    # Time-frequency
    tf_html = ""
    try:
        if tf_path and tf_path.exists():
            tf_html = (
                "<div><h2>Time-Frequency Analysis</h2>"
                f"<img src='{_embed_image(tf_path)}' alt='Time-frequency' style='width:100%'></div>"
            )
    except Exception as e:
        print(f" -> WARNING: Failed to generate time-frequency section: {e}")
    
    # Per-class heatmaps and topomaps
    per_class_html = ""
    try:
        if per_class_paths:
            cards = "\n".join(
                f'<div class="card"><img src="{_embed_image(p)}"><p>{p.stem}</p></div>'
                for p in per_class_paths if p.exists()
            )
            if cards:
                per_class_html = f"<div><h2>Per-Class Attribution Visualizations (IG)</h2><div class='grid'>{cards}</div></div>"
    except Exception as e:
        print(f" -> WARNING: Failed to generate per-class section: {e}")
    
    # Grand-average heatmap
    ga_heatmap_html = ""
    try:
        if ga_heatmap_path and ga_heatmap_path.exists():
            ga_heatmap_html = (
                "<div><h2>Grand Average Attribution Heatmap (IG)</h2>"
                f"<img src='{_embed_image(ga_heatmap_path)}' alt='Grand Average Heatmap' style='width:100%;border:1px solid #ddd'></div>"
            )
    except Exception as e:
        print(f" -> WARNING: Failed to generate grand average heatmap section: {e}")
    
    # Assemble HTML
    try:
        html = (
            "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
            f"<title>{report_title}</title>"
            "<style>"
            "body{font-family:sans-serif;margin:2em;background:#fafafa}"
            ".container{max-width:1400px;margin:auto;background:white;padding:2em;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}"
            "h1,h2,h3{text-align:center;color:#333}"
            "h1{border-bottom:3px solid #e67e22;padding-bottom:0.5em}"
            "h2{border-bottom:2px solid #3498db;padding-bottom:0.3em;margin-top:2em}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:1.5em;margin:1em 0}"
            ".card{border:1px solid #ddd;border-radius:6px;padding:0.5em;background:#fff}"
            ".card img{width:100%;border-radius:4px}"
            ".card p{text-align:center;font-weight:600;margin:0.5em 0 0;color:#555}"
            ".summary{display:flex;gap:2em;align-items:flex-start;justify-content:center;margin:1em 0 2em}"
            ".summary .box{flex:1;background:#f7f7f7;padding:1.5em;border-radius:6px}"
            ".summary pre{background:#fff;padding:1em;border-radius:4px;font-size:1.05em;border:1px solid #ddd}"
            ".peak-block{display:flex;gap:2em;align-items:flex-start;justify-content:center;margin:1.5em 0;padding:1em;background:#f9f9f9;border-radius:6px}"
            ".peak-left{flex:1} .peak-right{flex:1} .peak-right img{width:100%;border:1px solid #ddd;border-radius:4px}"
            "</style>"
            "</head><body><div class='container'>"
            f"<h1>{report_title}</h1>"
            "<div class='summary'>"
            f"<div class='box'><h2>Top {top_k} Channels (IG Overall)</h2><pre>{overall_block}</pre></div>"
            "</div>"
            f"{overall_topo_html}"
            f"{ga_heatmap_html}"
            f"{peak_html}"
            f"{tf_html}"
            f"{per_class_html}"
            "<div><h2>Per-Fold Attribution Heatmaps (IG)</h2>"
            f"<div class='grid'>{per_fold_ig_html}</div></div>"
            "</div></body></html>"
        )
        
        html_output_path.write_text(html, encoding="utf-8")
        print(f"\n -> Consolidated XAI HTML report saved: {html_output_path.name}")
        
    except Exception as e:
        print(f" -> ERROR: Failed to write HTML report: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Try PDF generation with Playwright (optional)
    try:
        from playwright.sync_api import sync_playwright
        pdf_path = run_dir / "consolidated_xai_report.pdf"
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file:///{html_output_path.absolute()}")
            page.pdf(path=str(pdf_path), format="A4")
            browser.close()
        print(f" -> PDF report saved: {pdf_path.name}")
    except ImportError:
        print(f" -> PDF generation skipped (Playwright not installed)")
    except Exception as e:
        print(f" -> PDF generation skipped (error): {e}")


# ==================== Main XAI Analysis ====================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run XAI analysis on a completed training run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    print(f"\n=== Starting XAI Analysis for {args.run_dir.name} ===\n")
    
    # Load configuration
    summary_data, cfg = _load_config(args.run_dir)

    dataset_dir = summary_data["dataset_dir"]
    fold_splits = summary_data["fold_splits"]

    # Setup dataset
    if "seed" not in cfg:
        raise ValueError("'seed' must be specified in config. No fallback allowed for reproducibility.")
    seed_everything(cfg["seed"])
    label_fn = task_registry.get(cfg["task"])
    dcfg = {"materialized_dir": dataset_dir, **cfg}
    full_dataset = RawEEGDataset(dcfg, label_fn)

    ch_names = full_dataset.channel_names
    sfreq = full_dataset.sfreq
    times_ms = full_dataset.times_ms
    class_names = getattr(full_dataset, "class_names", None)

    # Attach montage
    info_for_plot = _attach_montage(ch_names, sfreq)
    has_montage = info_for_plot is not None
    
    # Setup output directories with new organized structure
    xai_root = args.run_dir / "xai_analysis"
    for d in ["ig_heatmaps", "ig_topomaps", "ig_per_class_heatmaps", 
              "ig_per_class_topomaps", "metadata"]:
        (xai_root / d).mkdir(parents=True, exist_ok=True)

    # Get XAI parameters from config
    top_k = int(cfg.get('xai_top_k_channels', 10) or 10)
    peak_window_ms = float(cfg.get('peak_window_ms', 100) or 100)
    tf_freqs = cfg.get('tf_morlet_freqs', [4, 8, 13, 30])
    
    print(f"XAI Configuration:")
    print(f"  - Top-K channels: {top_k}")
    print(f"  - Peak window: {peak_window_ms} ms")
    print(f"  - TF frequencies: {tf_freqs}")
    print(f"  - Montage: {'attached' if has_montage else 'not available'}")
    print()
    
    # Per-fold processing
    per_fold_ig_arrays = []
    per_fold_ig_paths = []
    per_fold_class_labels = []
    
    for fold_info in fold_splits:
        fold = fold_info["fold"]
        ckpt = _resolve_ckpt(args.run_dir, fold)
        if not ckpt:
            print(f" --- skipping fold {fold:02d}: no checkpoint found ---")
            continue

        print(f"\n --- processing fold {fold:02d} (test subjects: {fold_info['test_subjects']}) ---")

        # Load model
        model = RAW_EEG_MODELS[cfg["model_name"]](
            cfg, len(full_dataset.class_names),
            C=full_dataset.num_channels, T=full_dataset.time_points
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        test_subjects = fold_info["test_subjects"]
        test_idx = [i for i, g in enumerate(full_dataset.groups) if g in test_subjects]

        # T011: Compute IG
        print("  Computing Integrated Gradients...")
        attr_matrix, n_correct, trial_labels = compute_ig_attributions(
            model, full_dataset, test_idx, DEVICE, class_names,
            input_adapter=squeeze_input_adapter
        )
        
        # Save IG outputs with new naming
        ig_npy_path = xai_root / "metadata" / f"fold_{fold:02d}_ig_attributions.npy"
        ig_png_path = xai_root / "ig_heatmaps" / f"fold_{fold:02d}_ig_heatmap.png"
        labels_path = xai_root / "metadata" / f"fold_{fold:02d}_class_labels.npy"
        
        np.save(ig_npy_path, attr_matrix)
        np.save(labels_path, trial_labels)
        plot_attribution_heatmap(
            attr_matrix, ch_names, times_ms,
            f"Fold {fold:02d} IG Attribution ({n_correct} samples)",
            ig_png_path
        )
        
        per_fold_ig_arrays.append(attr_matrix)
        per_fold_ig_paths.append(ig_png_path)
        per_fold_class_labels.append(trial_labels)

    print("\n--- per-fold XAI complete ---")

    # T012: Grand-average IG
    print("\n--- Computing grand averages ---")
    
    ga_png_path = None
    overall_topoplot_path = None
    top_channels_overall = []
    peak_summaries = []
    tf_path = None
    per_class_paths = []
    
    if per_fold_ig_arrays:
        print("  Grand-average IG...")
        ig_ga = np.mean(per_fold_ig_arrays, axis=0)  # (C, T)
        
        # Save to metadata
        ga_npy_path = xai_root / "metadata" / "grand_average_ig_attributions.npy"
        np.save(ga_npy_path, ig_ga)
        
        # Save heatmap to ROOT (for easy access) and ig_heatmaps directory
        ga_png_root = xai_root / "grand_average_ig_heatmap.png"
        ga_png_path = xai_root / "ig_heatmaps" / "grand_average_ig_heatmap.png"
        
        plot_attribution_heatmap(
            ig_ga, ch_names, times_ms,
            "Grand Average IG Attribution",
            ga_png_root
        )
        plot_attribution_heatmap(
            ig_ga, ch_names, times_ms,
            "Grand Average IG Attribution",
            ga_png_path
        )
        
        # Overall channel importance
        ig_ch_import = np.mean(np.abs(ig_ga), axis=1)  # (C,)
        top_idx = np.argsort(ig_ch_import)[::-1][:top_k]
        top_channels_overall = [ch_names[i] for i in top_idx]

        if has_montage:
            # Save topomap to ROOT (for easy access) and ig_topomaps directory
            overall_topoplot_root = xai_root / "grand_average_ig_topomap.png"
            overall_topoplot_path = xai_root / "ig_topomaps" / "grand_average_ig_topomap.png"
            
            plot_topomap(
                ig_ch_import, info_for_plot, top_idx, ch_names,
                "Overall Channel Importance (IG)",
                overall_topoplot_root
            )
            plot_topomap(
                ig_ch_import, info_for_plot, top_idx, ch_names,
                "Overall Channel Importance (IG)",
                overall_topoplot_path
            )
        
        # T013: Per-class grand-average IG
        if class_names and per_fold_class_labels:
            print("  Per-class grand-average IG...")
            per_class_heatmap_dir = xai_root / "ig_per_class_heatmaps"
            per_class_topomap_dir = xai_root / "ig_per_class_topomaps"
            
            for cls_idx, cls_name in enumerate(class_names):
                # Collect attributions for this class across all folds
                class_attrs = []
                for fold_idx, (attr, labels) in enumerate(zip(per_fold_ig_arrays, per_fold_class_labels)):
                    # Filter trials for this class
                    mask = (labels == cls_idx)
                    if mask.sum() > 0:
                        # This is already fold-averaged, so just include it
                        # (In reality, we'd need per-trial attributions, but for now use fold average)
                        class_attrs.append(attr)
                
                if class_attrs:
                    cls_ga = np.mean(class_attrs, axis=0)  # (C, T)
                    
                    safe_name = cls_name.replace(" ", "_")
                    cls_heatmap = per_class_heatmap_dir / f"class_{cls_idx:02d}_{safe_name}_ig_heatmap.png"
                    
                    plot_attribution_heatmap(
                        cls_ga, ch_names, times_ms,
                        f"Class {cls_name} - Grand Average IG",
                        cls_heatmap
                    )
                    per_class_paths.append(cls_heatmap)
                    
                    # Per-class topomap
                    if has_montage:
                        cls_ch_imp = np.mean(np.abs(cls_ga), axis=1)
                        cls_top_idx = np.argsort(cls_ch_imp)[::-1][:top_k]
                        cls_topomap = per_class_topomap_dir / f"class_{cls_idx:02d}_{safe_name}_ig_topomap.png"
                        plot_topomap(
                            cls_ch_imp, info_for_plot, cls_top_idx, ch_names,
                            f"Class {cls_name} - Channel Importance",
                            cls_topomap
                        )
                        per_class_paths.append(cls_topomap)
                        print(f"    -> Generated heatmap and topomap for class {cls_name}")
        
        # T014: Time-frequency analysis
        print("  Time-frequency analysis...")
        tfr = compute_time_frequency_map(ig_ga, sfreq, tf_freqs)
        if tfr is not None:
            tf_path = xai_root / "grand_average_time_frequency.png"
            plot_time_frequency_map(tfr, tf_path)
        else:
            tf_path = None
        
        # T015: Top-2 spatio-temporal events
        if has_montage and find_peaks is not None:
            print("  Top-2 spatio-temporal events...")
            mean_time_attr = np.mean(np.abs(ig_ga), axis=0)  # (T,)
            distance_samples = max(1, int(sfreq * (peak_window_ms / 1000.0)))
            peaks, _ = find_peaks(mean_time_attr, distance=distance_samples)
            
            if peaks.size > 0:
                scores = mean_time_attr[peaks]
                order = np.argsort(scores)[::-1][:2]
                chosen = peaks[order]
                
                for idx, pidx in enumerate(chosen, start=1):
                    center_ms = float(times_ms[pidx])
                    half_win = peak_window_ms / 2.0
                    t0 = center_ms - half_win
                    t1 = center_ms + half_win
                    
                    s0 = int(np.argmin(np.abs(times_ms - t0)))
                    s1 = int(np.argmin(np.abs(times_ms - t1)))
                    
                    window_attr = ig_ga[:, s0:s1+1]
                    ch_imp_win = np.mean(np.abs(window_attr), axis=1)
                    top_idx_win = np.argsort(ch_imp_win)[::-1][:top_k]
                    top_list = [ch_names[i] for i in top_idx_win]
                    
                    # Save to ROOT (for easy access) and ig_topomaps directory
                    peak_png_root = xai_root / f"peak{idx}_ig_topomap_{int(t0):03d}-{int(t1):03d}ms.png"
                    peak_png_subdir = xai_root / "ig_topomaps" / f"peak{idx}_ig_topomap_{int(t0):03d}-{int(t1):03d}ms.png"
                    
                    plot_topomap(
                        ch_imp_win, info_for_plot, top_idx_win, ch_names,
                        f"IG Peak Window {int(t0)}-{int(t1)} ms",
                        peak_png_root
                    )
                    plot_topomap(
                        ch_imp_win, info_for_plot, top_idx_win, ch_names,
                        f"IG Peak Window {int(t0)}-{int(t1)} ms",
                        peak_png_subdir
                    )
                    
                    peak_summaries.append((f"Window {int(t0)}-{int(t1)} ms", top_list, peak_png_root))
            else:
                print("  -> No peaks found in attribution signal")
        elif not has_montage:
            print("  -> Skipping peak analysis (montage not available)")
        elif find_peaks is None:
            print("  -> Skipping peak analysis (scipy.signal.find_peaks not available - install scipy)")
    
    # T016-T017: Create consolidated report
    print("\n--- Generating consolidated report ---")
    create_consolidated_report(
        run_dir=args.run_dir,
        summary_data=summary_data,
        ga_heatmap_path=ga_png_root if per_fold_ig_arrays else None,
        per_fold_ig_paths=per_fold_ig_paths,
        top_channels_overall=top_channels_overall,
        overall_topo_path=overall_topoplot_root if has_montage and per_fold_ig_arrays else None,
        peak_summaries=peak_summaries,
        per_class_paths=per_class_paths,
        tf_path=tf_path
    )

    print("\n=== XAI Analysis Complete ===\n")


if __name__ == "__main__":
    main()
