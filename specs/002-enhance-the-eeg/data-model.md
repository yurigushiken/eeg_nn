# XAI Data Model

This document defines the data structures, file formats, and schemas for the XAI analysis system.

## Configuration Schema

### XAI Defaults YAML (`configs/xai_defaults.yaml`)

```yaml
# Top-K channels to highlight in reports and topomaps
xai_top_k_channels: integer (>= 1)
  default: 10
  description: Number of most important channels to label on topomaps and list in reports

# Peak window duration for spatio-temporal event analysis (ms)
peak_window_ms: float (> 0)
  default: 100.0
  description: Time window width (milliseconds) for peak-centered channel importance analysis

# Frequencies for time-frequency decomposition (Hz)
tf_morlet_freqs: list[float] (all > 0)
  default: [4, 8, 13, 30]
  description: Frequency bins for Morlet wavelet time-frequency analysis
  note: Lowest frequency must satisfy: signal_duration_ms >= (n_cycles * 1000 / freq)
        where n_cycles defaults to freq/2

# Grad-CAM target layer (dotted path notation)
gradcam_target_layer: string
  default: "features.3"
  description: Dotted path to target layer for Grad-CAM attribution
  examples:
    - "block_1.0"     # EEGNeX block 1, layer 0
    - "block_1.1"     # EEGNeX block 1, layer 1
    - "features.3"    # Generic feature extractor, layer 3
  requirements:
    - Layer must preserve spatial (channel) dimension
    - Output shape should be (batch, channels > 1, time) or (batch, channels > 1)
    - Layers with collapsed spatial dim (channels=1) produce uniform topomaps
```

**Example:**
```yaml
xai_top_k_channels: 10
peak_window_ms: 100.0
tf_morlet_freqs: [4, 8, 13, 30]
gradcam_target_layer: "features.3"
```

**Merging priority** (highest to lowest):
1. Run config (`summary_*.json` → `hyper` dict)
2. `configs/xai_defaults.yaml`
3. Hard-coded defaults in `scripts/run_xai_analysis.py`

## File Formats

### 1. Attribution Matrices (`.npy`)

**NumPy array files storing raw attribution values.**

#### Per-Fold IG Attributions
- **Path**: `xai_analysis/integrated_gradients/fold_{fold:02d}_xai_attributions.npy`
- **Shape**: `(C, T)` where C = channels, T = time points
- **Dtype**: `float32` or `float64`
- **Content**: Mean Integrated Gradients attribution across correctly classified test trials for this fold
- **Units**: Attribution intensity (arbitrary scale; relative comparisons valid)

#### Per-Fold Class Labels
- **Path**: `xai_analysis/integrated_gradients_per_class/fold_{fold:02d}_class_labels.npy`
- **Shape**: `(N,)` where N = number of correctly classified trials in this fold
- **Dtype**: `int64`
- **Content**: True class label for each correctly classified trial
- **Purpose**: Enables per-class filtering for grand-average per-class analysis

#### Grand-Average IG Attributions
- **Path**: `xai_analysis/grand_average_xai_attributions.npy`
- **Shape**: `(C, T)`
- **Dtype**: `float32` or `float64`
- **Content**: Mean of per-fold IG attribution matrices across all folds

#### Per-Fold Grad-CAM
- **Path**: `xai_analysis/gradcam_heatmaps/fold_{fold:02d}_gradcam.npy`
- **Shape**: `(C, T)` or `(C,)` depending on target layer output
- **Dtype**: `float32` or `float64`
- **Content**: Mean Grad-CAM attribution across all test trials for this fold

**Example usage:**
```python
import numpy as np

# Load attribution matrix
attr = np.load("xai_analysis/grand_average_xai_attributions.npy")
print(attr.shape)  # (128, 248) for 128 channels, 248 time points

# Load class labels
labels = np.load("xai_analysis/integrated_gradients_per_class/fold_01_class_labels.npy")
print(np.unique(labels))  # [0, 1, 2] for 3 classes
```

### 2. Heatmaps (`.png`)

**Channel × time visualizations of attribution intensity.**

#### Per-Fold IG Heatmap
- **Path**: `xai_analysis/integrated_gradients/fold_{fold:02d}_xai_heatmap.png`
- **Format**: PNG image, 150 DPI
- **Dimensions**: ~1200×800 pixels (12×8 inches @ 150 DPI)
- **Colormap**: `inferno` (dark purple → yellow)
- **Axes**:
  - X: Time (ms) from epoch onset
  - Y: EEG channel names (ordered as in dataset)
- **Title**: `"Fold {fold:02d} IG Attribution ({n_correct} samples)"`

#### Grand-Average Heatmap
- **Path**: `xai_analysis/grand_average_xai_heatmap.png`
- **Format**: PNG image, 150 DPI
- **Title**: `"Grand Average IG Attribution"`

#### Per-Class Heatmap
- **Path**: `xai_analysis/grand_average_per_class/class_{cls:02d}_{cls_name}_xai_heatmap.png`
- **Format**: PNG image, 150 DPI
- **Title**: `"Class {cls_name} - Grand Average IG"`

#### Grad-CAM Heatmap
- **Path**: `xai_analysis/gradcam_heatmaps/fold_{fold:02d}_gradcam_heatmap.png`
- **Format**: PNG image, 150 DPI
- **Note**: For 1D Grad-CAM (shape `(C,)`), rendered as bar chart instead of heatmap

### 3. Topomaps (`.png`)

**Scalp topography plots showing spatial distribution of channel importance.**

#### Grand-Average IG Topomap
- **Path**: `xai_analysis/grand_average_xai_topoplot.png`
- **Format**: PNG image, 180 DPI
- **Dimensions**: ~800×800 pixels (square)
- **Colormap**: `inferno`
- **Montage**: Loaded from `net/AdultAverageNet128_v1.sfp`
- **Labels**: Top-K channels labeled at sensor positions
- **Contours**: 8 iso-contour lines
- **Colorbar**: Attribution intensity scale

#### Per-Class Topomap
- **Path**: `xai_analysis/grand_average_per_class/class_{cls:02d}_{cls_name}_xai_topoplot.png`
- **Format**: PNG image, 180 DPI

#### Peak Window Topomaps
- **Path**: `xai_analysis/grand_average_ig_peak{idx}_topoplot_{t0:03d}-{t1:03d}ms.png`
- **Format**: PNG image, 180 DPI
- **Title**: `"IG Peak Window {t0}-{t1} ms"`
- **Content**: Channel importance specific to this temporal window

#### Grad-CAM Topomaps (3 styles)
- **Paths**:
  - `xai_analysis/grand_average_gradcam_topomaps/default/grand_average_gradcam_topomap_default.png`
  - `xai_analysis/grand_average_gradcam_topomaps/contours/grand_average_gradcam_topomap_contours.png`
  - `xai_analysis/grand_average_gradcam_topomaps/sensors/grand_average_gradcam_topomap_sensors.png`
- **Format**: PNG image, 180 DPI
- **Styles**:
  - **default**: Clean topomap with 0 contours
  - **contours**: 10 iso-contour lines
  - **sensors**: Sensor markers displayed (`sensors=True`)

**Requirements:**
- Montage must be successfully attached to `mne.Info` object
- Channel names in dataset must match montage channel names (case-insensitive, with alias matching)
- At least one EEG digitization point (kind=3) must be set in `info.dig`

### 4. Time-Frequency Plot (`.png`)

**Morlet wavelet time-frequency decomposition of grand-average IG.**

- **Path**: `xai_analysis/grand_average_time_frequency.png`
- **Format**: PNG image, 150 DPI
- **Dimensions**: ~1200×600 pixels (12×6 inches)
- **Colormap**: `inferno`
- **Axes**:
  - X: Time (ms)
  - Y: Frequency (Hz)
- **Title**: `"Time-Frequency Attribution (Grand Average)"`
- **Content**: Power spectrum averaged across all channels
- **Algorithm**: MNE's `tfr_morlet` with `n_cycles = freqs / 2.0`

**Computation requirements:**
- Signal length (time points) must be ≥ wavelet length for lowest frequency
- Wavelet length ≈ `n_cycles * sfreq / freq`
- For `freq=4 Hz`, `n_cycles=2`, `sfreq=500 Hz`: requires ≥ 250 samples

**Graceful degradation:**
- If signal too short, TFR is skipped and `tf_path = None`
- Report still generates without time-frequency section

### 5. Consolidated HTML Report

**Interactive HTML document with all XAI visualizations embedded.**

- **Path**: `<run_dir>/consolidated_xai_report.html`
- **Format**: HTML5 with embedded CSS and base64-encoded images
- **Encoding**: UTF-8
- **Dependencies**: None (self-contained single file)
- **Browser support**: Modern browsers (Chrome, Firefox, Edge, Safari)

**Structure:**
```html
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>XAI Report: {task_name} ({model_name})</title>
  <style>
    /* Embedded CSS for styling */
  </style>
</head>
<body>
  <div class='container'>
    <h1>XAI Report Title</h1>
    
    <!-- Top-K Channel Summary -->
    <div class='summary'>
      <div class='box'>
        <h2>Top {K} Channels (IG Overall)</h2>
        <pre>{channel_list}</pre>
      </div>
    </div>
    
    <!-- Overall Topomap -->
    <div>
      <h2>Overall Channel Importance (IG)</h2>
      <img src='data:image/png;base64,{base64_img}' />
    </div>
    
    <!-- Grand Average Heatmap -->
    <div>
      <h2>Grand Average Attribution Heatmap (IG)</h2>
      <img src='data:image/png;base64,{base64_img}' />
    </div>
    
    <!-- Top-2 Spatio-Temporal Events -->
    <div>
      <h2>Top 2 Temporal Windows (IG)</h2>
      <div class='peak-block'>
        <div class='peak-left'>
          <h3>Window {t0}-{t1} ms</h3>
          <pre>{top_channels}</pre>
        </div>
        <div class='peak-right'>
          <img src='data:image/png;base64,{base64_img}' />
        </div>
      </div>
      <!-- Repeat for peak 2 -->
    </div>
    
    <!-- Time-Frequency Analysis -->
    <div>
      <h2>Time-Frequency Analysis</h2>
      <img src='data:image/png;base64,{base64_img}' />
    </div>
    
    <!-- Per-Class Heatmaps -->
    <div>
      <h2>Per-Class Attribution Heatmaps</h2>
      <div class='grid'>
        <div class='card'><img src='...' /><p>class_00_1</p></div>
        <!-- ... -->
      </div>
    </div>
    
    <!-- Per-Fold IG Heatmaps -->
    <div>
      <h2>Per-Fold Attribution Heatmaps (IG)</h2>
      <div class='grid'>
        <div class='card'><img src='...' /><p>fold_01</p></div>
        <!-- ... -->
      </div>
    </div>
    
    <!-- Per-Fold Grad-CAM Heatmaps -->
    <div>
      <h2>Per-Fold Grad-CAM Heatmaps</h2>
      <div class='grid'>
        <!-- ... -->
      </div>
    </div>
    
    <!-- Grand Average Grad-CAM Topomaps -->
    <div>
      <h2>Grand Average Grad-CAM Topomaps</h2>
      <div class='grid'>
        <!-- 3 styles -->
      </div>
    </div>
  </div>
</body>
</html>
```

**Image embedding:**
```python
import base64

def _embed_image(img_path):
    with open(img_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()
```

### 6. PDF Report

**Printable PDF version of the HTML report.**

- **Path**: `<run_dir>/consolidated_xai_report.pdf`
- **Format**: PDF (A4 page size)
- **Generator**: Playwright (headless Chromium)
- **Fallback**: Gracefully skipped if Playwright unavailable

**Generation:**
```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(f"file:///{html_path.absolute()}")
    page.pdf(path=str(pdf_path), format="A4")
    browser.close()
```

## Data Flow

### Input Data
1. **Completed training run** with:
   - `summary_*.json` (hyper-parameters, dataset path, fold splits)
   - `ckpt/fold_{fold:02d}_best.ckpt` (model checkpoints)
   - Dataset in `materialized_dir` (per-subject `.fif` files)

### Processing Pipeline
1. **Load configuration**: Merge `xai_defaults.yaml` with run config
2. **Attach montage**: Try custom montage from `net/AdultAverageNet128_v1.sfp`
3. **Per-fold loop**:
   - Load model checkpoint
   - Get test indices from fold split
   - Compute IG on correctly classified trials → save `.npy`, `.png`
   - Save class labels → `.npy`
   - Compute Grad-CAM → save `.npy`, `.png`
4. **Grand averages**:
   - Average IG across folds → save `.npy`, `.png`
   - Compute overall channel importance → save topomap (if montage)
   - Per-class analysis → save heatmaps, topomaps
   - Time-frequency decomposition → save `.png`
   - Peak detection → save topomaps (if montage)
   - Grad-CAM grand average → save 3 topomap styles (if montage)
5. **Report generation**:
   - Embed all images as base64
   - Generate HTML
   - Generate PDF (if Playwright available)

### Output Artifacts

**Total files generated (typical LOSO run with 20 folds, 3 classes, montage attached, TFR successful):**
- Per-fold: 20 × 4 = 80 files (IG `.npy`, IG `.png`, labels `.npy`, Grad-CAM `.npy`, Grad-CAM `.png`)
- Grand-average: ~15 files (GA IG, GA topomap, per-class 3×2, TF, peak 2×1, Grad-CAM 3×1)
- Reports: 2 files (HTML, PDF)
- **Total**: ~97 files

**Disk usage (estimate):**
- Per-fold `.npy`: ~250 KB each (128 ch × 248 tp × 8 bytes)
- Per-fold `.png`: ~50-200 KB each (heatmaps/topomaps)
- Grand-average files: similar sizes
- HTML report: ~5-20 MB (all images embedded)
- PDF report: ~3-15 MB
- **Total per run**: ~50-100 MB

## Channel Name Requirements

**For topomaps to be generated:**
- Channel names in dataset must match montage channel names
- MNE uses case-insensitive matching with alias support
- Standard 10-20 names work out-of-box: `Fp1`, `Fp2`, `F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`, `Fz`, `Cz`, `Pz`, `Oz`
- Custom montages (like `AdultAverageNet128_v1.sfp`) use EGI naming: `E1`, `E2`, ..., `E129`

**Validation:**
```python
import mne

info = mne.create_info(ch_names=['Fp1', 'Fz', 'Cz'], sfreq=500, ch_types=['eeg']*3)
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, match_case=False, match_alias=True, on_missing="ignore")

# Check if montage attached
if info.dig is None or len(info.dig) == 0:
    print("Montage did not attach (channel name mismatch)")
else:
    eeg_digs = [d for d in info.dig if d['kind'] == 3]
    print(f"{len(eeg_digs)} channels have positions")
```

## Error Handling

**Graceful degradation:**
- **No montage**: Skip topomaps, continue with heatmaps
- **Signal too short**: Skip time-frequency analysis
- **Grad-CAM layer not found**: Log warning, skip Grad-CAM
- **No correctly classified trials**: Save zeros for IG, log warning
- **Playwright unavailable**: Skip PDF, HTML still generated

**Fatal errors** (abort with clear message):
- Run directory not found
- `summary_*.json` not found
- No checkpoints found for any fold
- Dataset directory not accessible
- Captum not installed

## Versioning and Compatibility

**Current version**: 1.0 (as of feature 002 implementation)

**Backward compatibility:**
- New XAI system replaces legacy stub functions in `utils/xai.py`
- Old runs (before feature 002) can still have XAI run on them
- Config merging handles missing `xai_defaults.yaml` gracefully

**Forward compatibility:**
- Additional XAI methods can be added without breaking existing outputs
- New plot types added to separate subdirectories
- Report format is extensible (new sections append after existing ones)

## Testing

**Contract test** (`tests/contract/test_xai_config.py`):
- Validates `xai_defaults.yaml` schema
- Checks all required keys present
- Validates data types and value ranges

**Integration test** (`tests/integration/test_xai_pipeline.py`):
- Runs full XAI pipeline on minimal fixture
- Asserts file existence for all expected outputs
- Validates HTML report generation
- Checks artifact organization

**Fixture** (`tests/fixtures/xai_test_run/`):
- Minimal 2-subject, 3-class, 100-timepoint dataset
- Pre-generated checkpoints
- Mock summary JSON
- Enables fast end-to-end testing

