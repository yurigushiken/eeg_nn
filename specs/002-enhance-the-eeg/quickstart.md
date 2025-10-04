# XAI Analysis Quickstart

This guide walks you through running XAI analysis on a completed EEG training run and interpreting the results.

## Prerequisites

1. **Completed training run** with checkpoints in `results/runs/<run_dir>/ckpt/`
2. **Environment activated**: `conda activate eegnex-env`
3. **Required dependencies**: Captum, MNE-Python, Matplotlib, SciPy (already in `environment.yml`)
4. **Optional**: Playwright for PDF reports (`playwright install`)

## Quick Start

### Step 1: Run XAI Analysis

Navigate to project root and execute:

```powershell
conda activate eegnex-env
python -X utf8 -u scripts/run_xai_analysis.py --run-dir "results\runs\<your_run_dir>"
```

**Example:**
```powershell
python -X utf8 -u scripts/run_xai_analysis.py --run-dir "results\runs\20250103_1430_cardinality_1_3_eeg_seed_42_crop_ms_0_496"
```

### Step 2: Monitor Progress

The script will output progress for each fold:

```
=== Starting XAI Analysis for <run_name> ===

XAI Configuration:
  - Top-K channels: 10
  - Peak window: 100.0 ms
  - TF frequencies: [4, 8, 13, 30]
  - Grad-CAM layer: features.3
  - Montage: attached

 --- processing fold 01 (test subjects: [5, 17]) ---
  Computing Integrated Gradients...
  -> Computed IG for 45/54 correctly classified samples
  -> Saved heatmap: fold_01_xai_heatmap.png
  Computing Grad-CAM...
  -> Computed Grad-CAM using layer 'features.3' (shape: 128)
  -> Saved Grad-CAM plot: fold_01_gradcam_heatmap.png

[... additional folds ...]

--- Computing grand averages ---
  Grand-average IG...
  -> Saved heatmap: grand_average_xai_heatmap.png
  -> Saved topomap: grand_average_xai_topoplot.png
  Per-class grand-average IG...
  -> Saved heatmap: class_00_1_xai_heatmap.png
  -> Saved heatmap: class_01_2_xai_heatmap.png
  -> Saved heatmap: class_02_3_xai_heatmap.png
  Time-frequency analysis...
  -> Computed TFR with 4 frequency bins
  -> Saved time-frequency plot: grand_average_time_frequency.png
  Top-2 spatio-temporal events...
  -> Saved topomap: grand_average_ig_peak1_topoplot_150-250ms.png
  -> Saved topomap: grand_average_ig_peak2_topoplot_350-450ms.png

--- Generating consolidated report ---
 -> consolidated XAI HTML report saved to <run_dir>/consolidated_xai_report.html
 -> PDF report saved to <run_dir>/consolidated_xai_report.pdf

=== XAI Analysis Complete ===
```

Typical runtime: 2-5 minutes for a 3-class, 20-subject LOSO run.

### Step 3: Review Outputs

All XAI artifacts are saved to `<run_dir>/xai_analysis/`:

```
xai_analysis/
├── integrated_gradients/
│   ├── fold_01_xai_attributions.npy
│   ├── fold_01_xai_heatmap.png
│   ├── fold_02_xai_attributions.npy
│   ├── fold_02_xai_heatmap.png
│   └── ...
├── integrated_gradients_per_class/
│   ├── fold_01_class_labels.npy
│   └── ...
├── gradcam_heatmaps/
│   ├── fold_01_gradcam.npy
│   ├── fold_01_gradcam_heatmap.png
│   └── ...
├── grand_average_per_class/
│   ├── class_00_1_xai_heatmap.png
│   ├── class_00_1_xai_topoplot.png
│   └── ...
├── grand_average_gradcam_topomaps/
│   ├── default/grand_average_gradcam_topomap_default.png
│   ├── contours/grand_average_gradcam_topomap_contours.png
│   └── sensors/grand_average_gradcam_topomap_sensors.png
├── grand_average_xai_attributions.npy
├── grand_average_xai_heatmap.png
├── grand_average_xai_topoplot.png
├── grand_average_time_frequency.png
├── grand_average_ig_peak1_topoplot_150-250ms.png
└── grand_average_ig_peak2_topoplot_<t0>-<t1>ms.png
```

Plus consolidated reports at the run root:
```
<run_dir>/
├── consolidated_xai_report.html
└── consolidated_xai_report.pdf
```

### Step 4: Open the HTML Report

**Windows:**
```powershell
start <run_dir>\consolidated_xai_report.html
```

**macOS/Linux:**
```bash
open <run_dir>/consolidated_xai_report.html
```

Or simply double-click the HTML file in your file explorer.

## Interpreting the Report

### 1. Top-K Channel Summary

The report opens with the overall most important channels (default: top 10) averaged across all folds and time points.

**Interpretation:**
- Channels are ranked by absolute attribution magnitude
- High-ranking channels contributed most to the model's decisions
- Compare to known EEG literature (e.g., frontal channels for executive function)

**Example:**
```
Top 10 Channels (IG Overall)
 1. Fz      6. P3
 2. Cz      7. P4
 3. Pz      8. F3
 4. Oz      9. F4
 5. C3     10. C4
```

### 2. Overall Channel Importance Topomap

A scalp topography showing the spatial distribution of channel importance (averaged over time).

**Interpretation:**
- Hot colors (red/yellow) = high importance
- Cool colors (blue/purple) = low importance
- Top-K channels are labeled directly on the scalp
- Look for focal vs. distributed patterns

### 3. Grand Average Attribution Heatmap

Channel × time heatmap showing when and where the model focuses attention.

**Interpretation:**
- **X-axis**: Time (ms) from epoch onset
- **Y-axis**: EEG channels
- **Color intensity**: Attribution magnitude (importance)
- Vertical "hot streaks" = important time windows across multiple channels
- Horizontal "hot streaks" = a single channel important across time
- Compare to event timing (e.g., stimulus onset, response window)

### 4. Top-2 Spatio-Temporal Events

The report identifies the 2 most significant temporal windows and shows:
- Time window (e.g., "150-250 ms")
- Top 10 channels specific to that window
- Topomap of channel importance within that window

**Interpretation:**
- **Early peaks (0-200 ms)**: Sensory/perceptual processing
- **Mid peaks (200-400 ms)**: Cognitive evaluation (e.g., N200, P300 analogs)
- **Late peaks (>400 ms)**: Decision-making, motor preparation
- Window-specific channels may differ from overall top channels (temporal dynamics)

### 5. Time-Frequency Analysis

Shows which oscillatory frequency bands were most important over time.

**Interpretation:**
- **Theta (4-7 Hz)**: Working memory, attention
- **Alpha (8-13 Hz)**: Inhibition, idle states
- **Beta (13-30 Hz)**: Motor preparation, decision-making
- Hot regions = model heavily weighted these frequencies at those times

### 6. Per-Class Attribution Heatmaps

Separate heatmaps for each class (e.g., cardinality 1, 2, 3) showing class-specific patterns.

**Interpretation:**
- Compare across classes to identify discriminative features
- **Unique patterns**: Features specific to one class
- **Shared patterns**: Common processing across classes
- Helps understand what the model learned to distinguish classes

### 7. Per-Fold Gallery

All individual fold heatmaps (IG and Grad-CAM) for detailed inspection.

**Interpretation:**
- Check consistency across folds (stable vs. variable patterns)
- Identify outlier folds with unusual attribution patterns
- Grad-CAM shows spatial localization at a specific layer

### 8. Grand Average Grad-CAM Topomaps

Three visualization styles of Grad-CAM grand-average:
- **Default**: Clean topomap
- **Contours**: With iso-contour lines
- **Sensors**: With channel markers

**Interpretation:**
- Grad-CAM highlights spatial regions at a specific network layer
- Compare to IG topomaps to see if different methods agree
- Flat/uniform maps may indicate poor layer choice (see troubleshooting)

## Configuration

### Customize XAI Parameters

Edit `configs/xai_defaults.yaml`:

```yaml
# Number of top channels to highlight
xai_top_k_channels: 15  # Default: 10

# Peak detection window width (ms)
peak_window_ms: 150  # Default: 100

# Frequencies for time-frequency analysis (Hz)
tf_morlet_freqs: [2, 4, 8, 13, 20, 30]  # Default: [4, 8, 13, 30]

# Grad-CAM target layer (model-specific)
gradcam_target_layer: "block_1.1"  # Default: "features.3"
```

Rerun the analysis to apply changes.

### Change Grad-CAM Layer

If Grad-CAM topomaps are flat/uniform, target an earlier layer:

```powershell
# Find available layers
python -c "from code.model_builders import RAW_EEG_MODELS; import torch; m=RAW_EEG_MODELS['eegnex']({}, 3, 128, 200); print([n for n,_ in m.named_modules()])"

# Common EEGNeX layers:
# - block_1.0, block_1.1, block_2.0 → preserve channel×time (GOOD)
# - block_3.0, block_4.0 → collapsed spatial dim (BAD)
```

Update `gradcam_target_layer` in `xai_defaults.yaml` or pass via CLI (future enhancement).

## Troubleshooting

### Issue: No topomaps generated

**Cause:** Montage not attached (channel names don't match `net/AdultAverageNet128_v1.sfp`)

**Solution:**
1. Check console for montage warnings
2. Verify channel names match standard 10-20 (e.g., Fp1, Fz, Cz, not E1, E2)
3. Ensure `net/AdultAverageNet128_v1.sfp` exists

**Workaround:** Heatmaps are still generated; only topomaps require montage.

### Issue: Time-frequency analysis skipped

**Cause:** Signal too short for lowest frequency wavelet

**Console output:**
```
Signal too short (150 samples) for TFR with lowest freq 4 Hz
(requires >= 250 samples). Skipping time-frequency analysis.
```

**Solution:**
- Reduce `tf_morlet_freqs` minimum (e.g., `[8, 13, 30]` instead of `[4, 8, 13, 30]`)
- Or use longer epochs during training (increase `crop_ms` range)

### Issue: Grad-CAM heatmaps are uniform/flat

**Cause:** Target layer has collapsed spatial dimension

**Solution:**
1. Inspect layer output shapes: `python -c "from code.model_builders import RAW_EEG_MODELS; import torch; m=RAW_EEG_MODELS['eegnex']({}, 3, 128, 200); x=torch.randn(1,128,200); print({n: m.get_submodule(n)(x).shape for n in ['block_1.0','block_2.0','block_3.0']})"`
2. Choose layer with shape `(batch, channels>1, time)` (e.g., `block_1.1`)
3. Update `gradcam_target_layer` in `xai_defaults.yaml`

### Issue: PDF report not generated

**Cause:** Playwright not installed

**Solution:**
```powershell
playwright install
```

**Note:** HTML report is always generated; PDF is optional.

### Issue: Low correctly classified sample count

**Console output:**
```
-> Computed IG for 12/54 correctly classified samples
```

**Interpretation:** Only 12 test trials were correctly classified in this fold (22% accuracy).

**Impact:** IG attributions are based on fewer trials, potentially less stable.

**Action:** Check model performance in `outer_eval_metrics.csv`. Low accuracy may indicate:
- Poor hyperparameters
- Insufficient training
- Task difficulty

## Advanced Usage

### Automatic XAI During Training

Add `--run-xai` flag when running `train.py`:

```powershell
python -X utf8 -u train.py `
  --task cardinality_1_3 `
  --engine eeg `
  --base configs/tasks/cardinality_1_3/base.yaml `
  --run-xai
```

XAI analysis runs immediately after training completes.

### Programmatic Access to Attribution Data

All attribution matrices are saved as NumPy `.npy` files for downstream analysis:

```python
import numpy as np

# Load grand-average IG attributions (C × T)
ga_ig = np.load("xai_analysis/grand_average_xai_attributions.npy")

# Load per-fold IG
fold_01_ig = np.load("xai_analysis/integrated_gradients/fold_01_xai_attributions.npy")

# Load per-fold class labels
fold_01_labels = np.load("xai_analysis/integrated_gradients_per_class/fold_01_class_labels.npy")

# Dimensions
print(ga_ig.shape)  # e.g., (128, 248) for 128 channels, 248 time points
```

Use these for:
- Custom visualizations
- Statistical testing (e.g., cluster-based permutation)
- Integration with other analyses (e.g., source localization)

### Multi-Seed Aggregation

For multi-seed runs, run XAI on each seed's directory:

```powershell
# Seed 1
python scripts/run_xai_analysis.py --run-dir "results\runs\<task>_seed_41_<suffix>"

# Seed 2
python scripts/run_xai_analysis.py --run-dir "results\runs\<task>_seed_42_<suffix>"

# ... etc.
```

Then manually aggregate `.npy` files for cross-seed stability analysis.

## Next Steps

After reviewing the XAI report:

1. **Validate with domain knowledge**: Do important channels/times align with known EEG literature?
2. **Compare across conditions**: Run XAI on different tasks/models to identify task-specific patterns
3. **Statistical testing**: Use attribution matrices for cluster-based permutation tests
4. **Publication**: Include consolidated HTML report as supplementary material

## References

- **Integrated Gradients**: Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks." ICML.
- **Grad-CAM**: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.
- **Captum Library**: https://captum.ai/
- **MNE-Python**: https://mne.tools/

## Support

For issues specific to this XAI implementation, see:
- `specs/002-enhance-the-eeg/spec.md`: Full specification
- `specs/002-enhance-the-eeg/plan.md`: Technical implementation details
- `scripts/run_xai_analysis.py`: Orchestrator source code
- `utils/xai.py`: Core XAI functions

For general EEG analysis questions, consult the main `README.md`.

