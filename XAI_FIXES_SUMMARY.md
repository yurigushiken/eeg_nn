# XAI System Fixes - Summary

## Date: October 3, 2025

## Issues Fixed

### 1. **Critical Bug: Montage Return Issue** ✅
**File**: `scripts/run_xai_analysis.py` (Lines 204-209)

**Problem**: Unconditional `return None` statement was preventing successful montage attachment from returning the valid `info` object. This caused ALL topomap generation to be skipped.

**Fix**: Fixed indentation so `return None` only executes when montage attachment actually fails, and `return info` executes when successful.

**Impact**: 
- ✅ IG overall topomap will now be generated
- ✅ Per-class topomaps will now be generated  
- ✅ Top-2 spatio-temporal event topomaps will now be generated
- ✅ Grad-CAM topomaps will now be generated

---

### 2. **Enhanced Grad-CAM Layer Detection** ✅
**File**: `utils/xai.py` (Lines 188-196)

**Problem**: When Grad-CAM target layer wasn't found, error messages were not helpful for debugging.

**Fix**: Added detailed error reporting that shows:
- Available attributes at the current level
- List of all top-level layers in the model
- This helps users identify correct layer names for their model

**Impact**: 
- ✅ Better debugging when layer names are incorrect
- ✅ Users can easily see available layers to choose from

---

### 3. **Improved Logging for Peak Analysis** ✅
**File**: `scripts/run_xai_analysis.py` (Lines 634-639)

**Problem**: When peak analysis was skipped, no informative message was logged.

**Fix**: Added explicit logging for three scenarios:
- When no peaks are found in the signal
- When montage is not available
- When SciPy is not installed

**Impact**: 
- ✅ Clear feedback about why Top-2 events aren't generated
- ✅ Actionable guidance (e.g., "install scipy")

---

### 4. **Per-Class Topomap Generation** ✅
**File**: `scripts/run_xai_analysis.py` (Lines 590-591)

**Problem**: Per-class topomaps were being generated but not added to the paths list for report inclusion.

**Fix**: Added `per_class_paths.append(cls_topo)` and logging statement after per-class topomap generation.

**Impact**: 
- ✅ Per-class topomaps now included in the HTML report
- ✅ Better progress feedback during analysis

---

### 5. **Improved Report Section Labels** ✅
**File**: `scripts/run_xai_analysis.py` (Line 302)

**Problem**: Section title was ambiguous.

**Fix**: Changed "Per-Class Attribution Heatmaps" to "Per-Class Attribution Visualizations (IG)" to clarify it includes both heatmaps and topomaps.

**Impact**: 
- ✅ Clearer report structure
- ✅ Users understand what visualizations are included

---

## Expected Outputs After Fixes

### Generated Files (per spec):
```
xai_analysis/
├── integrated_gradients/
│   └── fold_XX_xai_attributions.npy  (24 files)
├── integrated_gradients_per_class/
│   └── fold_XX_class_labels.npy  (24 files)
├── gradcam_heatmaps/
│   ├── fold_XX_gradcam.npy  (24 files)
│   └── fold_XX_gradcam_heatmap.png  (24 files)
├── grand_average_per_class/
│   ├── class_XX_Y_xai_heatmap.png  (3 files - one per class)
│   └── class_XX_Y_xai_topoplot.png  (3 files - NEW!)
├── grand_average_gradcam_topomaps/
│   ├── default/
│   ├── contours/
│   └── sensors/
│       └── grand_average_gradcam_topomap_*.png  (3 files - NEW!)
├── grand_average_xai_attributions.npy
├── grand_average_xai_heatmap.png
├── grand_average_xai_topoplot.png  (NEW!)
├── grand_average_ig_peak1_topoplot_XXX-YYYms.png  (NEW!)
├── grand_average_ig_peak2_topoplot_XXX-YYYms.png  (NEW!)
└── grand_average_time_frequency.png
```

### HTML Report Structure (per spec):
1. **Title**: XAI Report with task and model name
2. **Top 10 Channels (IG Overall)**: Two-column list
3. **Overall Channel Importance (IG)**: Overall topomap - **NOW INCLUDED** ✅
4. **Grand Average Attribution Heatmap (IG)**: Channel × time heatmap
5. **Top 2 Temporal Windows (IG)**: Peak events with topomaps - **NOW INCLUDED** ✅
6. **Time-Frequency Analysis**: TFR visualization
7. **Per-Class Attribution Visualizations (IG)**: Heatmaps AND topomaps - **TOPOMAPS NOW INCLUDED** ✅
8. **Per-Fold Attribution Heatmaps (IG)**: Gallery of 24 fold heatmaps
9. **Per-Fold Grad-CAM Heatmaps**: Gallery of 24 Grad-CAM heatmaps
10. **Grand Average Grad-CAM Topomaps**: 3 style variations - **NOW INCLUDED** ✅

---

## Technical Notes

### Grad-CAM Layer Configuration
Current config: `gradcam_target_layer: "block_1.1"` (in `configs/xai_defaults.yaml`)

If the layer is not found, the enhanced error logging will show available layers. For EEGNeX models from Braindecode, typical good layers are:
- `block_1.0` - Early convolutional layer
- `block_1.1` - Second layer in first block (current default)
- `block_2.0` - First layer in second block

**Note**: The layer must preserve spatial (channel) dimensions. Later blocks may collapse to single channel and produce uniform topomaps.

### Montage Information
- Custom montage: `net/AdultAverageNet128_v1.sfp` (128-channel EGI system)
- Fallback: Standard montage "GSN-HydroCel-129"
- Channel matching is case-insensitive with alias support

### Dependencies
- **Required**: torch, numpy, matplotlib, mne, captum
- **Optional**: scipy (for peak detection), playwright (for PDF generation)

---

## Files Modified

1. `scripts/run_xai_analysis.py` - Main XAI analysis script
   - Fixed montage return bug (critical)
   - Enhanced logging for peak analysis
   - Added per-class topomap to report paths
   - Improved section labels

2. `utils/xai.py` - XAI utility functions
   - Enhanced Grad-CAM error messages
   - Added layer detection debugging output

---

## Verification Checklist

After running XAI analysis, verify:
- [ ] `grand_average_xai_topoplot.png` exists
- [ ] Two peak topoplot files exist (e.g., `grand_average_ig_peak1_topoplot_*.png`)
- [ ] Per-class topomaps exist (3 files: `class_XX_Y_xai_topoplot.png`)
- [ ] Grad-CAM topomaps exist (3 files in subdirectories)
- [ ] HTML report includes all 10 sections listed above
- [ ] HTML report has "Top 2 Temporal Windows (IG)" section with topomaps

---

## Related Specification

See `specs/002-enhance-the-eeg/spec.md` for full functional requirements:
- FR-003: Grand Average (topomap generation)
- FR-004: Per-Class IG (including topomaps)
- FR-006: Top-2 Events (peak window topomaps)
- FR-007: Consolidated Report (all visualizations)


