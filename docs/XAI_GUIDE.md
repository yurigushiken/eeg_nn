# XAI Guide: Explainable AI Analysis

Comprehensive guide to generating and interpreting Integrated Gradients (IG) attributions, topographic brain maps, and time-frequency analysis.

## Overview

The XAI system reveals which EEG channels and time points drive model predictions using Integrated Gradients (IG), a gradient-based attribution method. This helps answer: **What spatiotemporal patterns does the model use to decode numerosity?**

## Quick Start

### Generate XAI for Completed Run

```powershell
python -X utf8 -u scripts/run_xai_analysis.py \
  --run-dir "results\runs\<your_run_directory>"
```

### Auto-Generate During Training

```powershell
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --run-xai
```

**Expected time:** 5-15 minutes (depends on number of test trials)

## Output Structure

All XAI artifacts are written to `<run_dir>/xai_analysis/`:

### Per-Fold Outputs

- `integrated_gradients/fold_XX_xai_attributions.npy` — IG attribution matrices (Channels × Time)
- `integrated_gradients/fold_XX_xai_heatmap.png` — Channel×time heatmaps
- `integrated_gradients_per_class/fold_XX_class_labels.npy` — Per-trial class labels

### Grand-Average Outputs

**Overall Attributions:**
- `grand_average_xai_attributions.npy` — Mean IG across all folds (C×T)
- `grand_average_xai_heatmap.png` — Grand-average channel×time heatmap
- `grand_average_xai_topoplot.png` — Scalp topography (requires montage)

**Per-Class Attributions:**
- `grand_average_per_class/class_XX_<name>_xai_heatmap.png` — Class-specific heatmaps
- `grand_average_per_class/class_XX_<name>_xai_topoplot.png` — Class-specific topomaps

**Advanced Analyses:**
- `grand_average_time_frequency.png` — Morlet wavelet time-frequency decomposition
- `grand_average_ig_peak1_topoplot_<t0>-<t1>ms.png` — Top temporal window #1
- `grand_average_ig_peak2_topoplot_<t0>-<t1>ms.png` — Top temporal window #2

### Consolidated Reports

- `consolidated_xai_report.html` — Interactive HTML report (all visualizations embedded)
- `consolidated_xai_report.pdf` — Printable PDF version (requires Playwright)

**HTML report includes:**
1. Top-K channel summary (default K=10)
2. Top-2 spatio-temporal events (peak windows + topomaps)
3. Grand-average visualizations
4. Per-class attribution analysis
5. Per-fold gallery

## Configuration

XAI parameters are set in `configs/xai_defaults.yaml`:

```yaml
xai_top_k_channels: 10         # Number of top channels to highlight
peak_window_ms: 100            # Duration for peak window analysis (ms)
tf_morlet_freqs: [4, 8, 13, 30]  # Frequencies for time-frequency (Hz)
```

**Run config takes precedence** over defaults.

### Parameter Explanations

**`xai_top_k_channels`:** Number of most important channels to label in topomaps and list in reports.

**`peak_window_ms`:** Width of time windows for spatio-temporal event analysis. Larger values smooth over temporal fluctuations; smaller values capture transient events.

**`tf_morlet_freqs`:** Frequency bands for Morlet wavelet decomposition. Default covers theta (4 Hz), alpha (8 Hz), beta (13 Hz), and low gamma (30 Hz).

## Technical Details

### Integrated Gradients Method

**What it computes:** For each channel and time point, IG measures how much changing that value affects the model's prediction.

**Formula:**
```
IG(x) = (x - baseline) × ∫₀¹ ∇f(baseline + α(x - baseline)) dα
```

**In practice:**
- Baseline: zero signal (no EEG activity)
- Integration: 50 steps from baseline to actual signal
- Output: Attribution matrix (Channels × Time)

**Properties:**
- **Completeness:** Sum of attributions = prediction score - baseline score
- **Sensitivity:** Non-zero gradient → non-zero attribution
- **Implementation invariance:** Same attributions for functionally equivalent networks

### Averaging Strategy

**Per-fold:** Only correctly classified test trials are included (ensures attributions reflect successful decoding).

**Grand-average:** Mean across all folds, weighted equally regardless of fold size.

**Per-class:** Trials filtered by true label before averaging (reveals class-specific patterns).

### Topomap Generation

**Requirements:**
- Montage file: `net/AdultAverageNet128_v1.sfp` (128-channel sensor positions)
- Channel names must match montage (e.g., `Cz`, `Fz`, `Pz`)

**Method:**
- IG attributions averaged over time → one value per channel
- MNE-Python's `plot_topomap` with spherical spline interpolation
- Top-K channels labeled on scalp

**Graceful degradation:** If montage unavailable, topomaps skipped (other analyses proceed).

### Time-Frequency Analysis

**Method:** Morlet wavelet decomposition of grand-average IG attribution matrix.

**Purpose:** Reveals which frequency bands (theta, alpha, beta, gamma) contribute to decoding.

**Requirements:** Signal length ≥ longest wavelet duration. Gracefully skipped for very short epochs.

**Interpretation:**
- Bright regions: High attribution at that frequency × time
- Dark regions: Low attribution
- Example: Strong theta (4 Hz) at 200-300ms suggests model uses slow oscillations during that window

### Spatio-Temporal Events (Peak Windows)

**Method:**
1. Compute mean attribution across channels at each time point
2. Use SciPy's `find_peaks` to identify top 2 temporal peaks
3. For each peak, extract `±peak_window_ms/2` around peak
4. Compute channel importances within that window
5. Generate topomap showing spatial distribution

**Purpose:** Identify discrete processing events (e.g., early sensory response at 150ms, later cognitive response at 400ms).

**Requires:** Montage (skipped if unavailable).

## Interpretation Guide

### Heatmaps (Channel × Time)

**X-axis:** Time relative to stimulus onset (ms)
**Y-axis:** EEG channels (ordered by montage)
**Color:** Attribution magnitude (bright = high importance)

**What to look for:**
- **Early peaks (0-200ms):** Sensory processing (N1, P1 components)
- **Late peaks (300-500ms):** Cognitive processing (P3b component)
- **Clustered channels:** Coherent spatial patterns (e.g., parietal cluster)

**Example interpretation:**
- Peak at Pz, 400ms → model uses P3b-like response
- Peak at Oz, 150ms → model uses early visual processing

### Topomaps (Scalp Distribution)

**View:** Top-down view of scalp (nose up, back of head down)
**Color:** Attribution strength (red = high, blue = low)
**Labels:** Top-K most important channels

**What to look for:**
- **Central-parietal focus:** Typical for numerical cognition (consistent with ERP literature)
- **Lateral asymmetry:** Left vs right hemisphere differences
- **Frontal vs posterior:** Executive control vs sensory processing

**Example interpretation:**
- Strong parietal activation → numerical magnitude processing
- Strong occipital activation → visual features (dot size/spacing) rather than numerosity

### Per-Class Differences

**Purpose:** Reveal class-specific neural signatures.

**What to look for:**
- **Different topographies:** Class "2" peaks at Pz, class "3" peaks at P4 → distinct spatial codes
- **Different timings:** Class "2" peaks at 350ms, class "3" peaks at 450ms → temporal dissociation
- **Overlap:** Similar patterns suggest shared processing, model uses subtle differences

**Example interpretation (landing_on_2_3):**
- "2" shows earlier parietal peak → faster subitizing
- "3" shows sustained parietal activity → prolonged individuation

### Time-Frequency Patterns

**What to look for:**
- **Theta band (4-8 Hz):** Working memory maintenance
- **Alpha band (8-13 Hz):** Attention modulation (suppression = engagement)
- **Beta band (13-30 Hz):** Sensorimotor processing
- **Gamma band (>30 Hz):** Local cortical processing

**Example interpretation:**
- Strong theta at 300-500ms → model uses sustained working memory signal
- Alpha suppression at 100-300ms → attentional engagement during stimulus processing

## Common Issues

### No Topomaps Generated

**Symptom:** Heatmaps exist, but no topoplot PNGs.

**Cause:** Montage file missing or channel name mismatch.

**Solution:**
1. Check `net/AdultAverageNet128_v1.sfp` exists
2. Verify channel names in EEG data match montage (use MNE's `ch_names` attribute)
3. Check console output for montage attachment warnings

### Time-Frequency Analysis Skipped

**Symptom:** "Signal too short for lowest frequency" warning.

**Cause:** Epoch duration < longest wavelet period.

**Solution:**
1. Use longer epochs (increase `crop_ms` range, e.g., `[0, 800]`)
2. Remove lowest frequency from `tf_morlet_freqs` (e.g., remove 4 Hz)

### PDF Generation Failed

**Symptom:** HTML report exists, but no PDF.

**Cause:** Playwright not installed.

**Solution:**
```powershell
playwright install
```

HTML report always generated; PDF is optional.

### Attributions Look Random/Noisy

**Symptom:** No clear spatial or temporal structure in heatmaps.

**Causes:**
1. Model performance at chance (no learned patterns)
2. Very few test trials (noisy estimates)
3. Model overfitting (learned noise, not signal)

**Solutions:**
1. Check model accuracy in `outer_eval_metrics.csv` (should be >>chance)
2. Use more test trials (increase N subjects or use GroupKFold instead of LOSO)
3. Re-run hyperparameter search with better regularization

## Advanced Usage

### Comparing XAI Across Tasks

**Goal:** Do different tasks (e.g., cardinality_1_3 vs landing_on_2_3) show different neural signatures?

**Method:**
1. Run XAI for both tasks
2. Visually compare grand-average topomaps side-by-side
3. Look for timing differences (heatmap peak latencies) and spatial differences (topomap hotspots)

**Expected:** Tasks within PI range (1-3) may show similar parietal signatures; tasks crossing PI/ANS boundary may show distinct patterns.

### Exporting Attributions for Custom Analysis

Attribution arrays are saved as NumPy `.npy` files:

```python
import numpy as np

# Load grand-average attributions (Channels × Time)
attr = np.load("xai_analysis/grand_average_xai_attributions.npy")

# Load channel names and times from run config or MNE info
# (You'll need to load the original .fif file to get ch_names and times)

# Custom analysis: e.g., compute temporal centroid
time_centroid = np.average(times, weights=attr.mean(axis=0))
print(f"Temporal centroid: {time_centroid:.1f} ms")
```

### Checkpoint Selection for XAI

XAI uses best checkpoint per fold. **Resolution order:**
1. `fold_XX_refit_best.ckpt` (if refit mode)
2. `fold_XX_best.ckpt` (if ensemble mode, single outer-best ckpt)
3. First `fold_XX_inner_YY_best.ckpt` (fallback to any inner ckpt)

**Recommendation:** Use ensemble mode for XAI (more stable attributions averaged across K inner models).

## References

- **Integrated Gradients:** Sundararajan et al. (2017), "Axiomatic Attribution for Deep Networks"
- **Captum library:** https://captum.ai/
- **MNE topoplots:** Gramfort et al. (2013), "MEG and EEG data analysis with MNE-Python"

## Next Steps

- For configuring XAI parameters, see [Configuration Reference](CONFIGURATION.md#xai-configuration)
- For generating XAI as part of workflow, see [Workflows](WORKFLOWS.md#complete-xai-analysis)
- For interpreting results in context of ERP literature, consult Tang-Lonardo (2023) dissertation
