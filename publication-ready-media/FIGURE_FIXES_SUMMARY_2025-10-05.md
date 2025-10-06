# Publication Figure Fixes - Complete Summary
**Date:** October 5, 2025  
**Status:** ✅ ALL FIXES COMPLETED & REGENERATED

---

## 🎯 Overview

Successfully fixed and regenerated 6 publication-ready figures (Figures 1, 2, 3, 4, 5, 10) with corrections to text content, layout, colors, and occlusion issues. All figures now meet neuroscience publication standards.

---

## 📊 Figures Modified

### ✅ Figure 1: Pipeline Flowchart (`figure1_pipeline_v4.py`)

**Issues Fixed:**
1. **Raw Data Acquisition**: Updated from "~360 trials/subject" → **"~300 trials/subject"**
2. **Preprocessing Details**: Added comprehensive dataset list
   - Before: Generic "Bandpass 1.5–40 Hz"
   - After: **"18 datasets: 3 HPF (1.0/1.5/2.0 Hz) × 3 LPF (35/40/45 Hz) × 2 baseline (on/off)"**
3. **Data Finalization**: Improved clarity
   - Added "condition alignment" (not just behavioral alignment)
   - Confirmed **~100 channels** after intersection (128 raw → exclude 28 non-scalp → ~100)
4. **Objective Caption**: Updated to reflect new composite metric
   - Before: "inner_mean_min_per_class_f1 (ensures all classes decodable)"
   - After: **"composite (65% min F1 + 35% plurality correctness) — ensures decodability & distinctness"**
5. **Final Evaluation**: Changed prediction method
   - Before: "Ensemble predictions"
   - After: **"Refit predictions"**
6. **Layout Improvement**: Separated Statistical Validation and XAI boxes
   - Before: Side-by-side on same row (confusing equivalence)
   - After: **Separate rows (Statistical Validation above XAI)** — clearer sequential flow
7. **Color Scheme**: Replaced gradient blues with lighter uniform colors
   - Easier to read on screen and in print
   - Specific hex codes: `#D6E9F5`, `#B8D9EE`, `#A8D0EA`, etc.

---

### ✅ Figure 2: Nested CV Schematic (`figure2_nested_cv_v4.py`)

**Issues Fixed:**
1. **Real Subject IDs**: Replaced generic S1-S24 with actual subject numbers
   - Real IDs: **02, 03, 04, 05, 08, 09, 10, 11, 12, 13, 14, 15, 17, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33**
   - Total: 24 subjects (but IDs go up to 33 due to exclusions/gaps)
   - Why: Shows true data provenance for publication transparency

2. **Text Occlusion Fixes**:
   - **Panel B (Outer Loop)**: 
     - Increased y-axis limits: 6.5 → **7.0**
     - Moved fold labels up (y_pos: 5.0/3.0/1.0 → **5.5/3.5/1.5**)
     - Shifted boxes right (x_pos: 1.5 → **1.8**) to give labels more space
   - **Panel C (Inner Loop)**:
     - Increased y-axis limits: 5.5 → **6.2**
     - Moved all inner fold rows up significantly (e.g., Inner Fold 5: 0.5 → **0.8**)
     - Moved title up (y: 5.2 → **5.9**)
     - Bottom caption moved up (y: -0.3 → **0.0**)

3. **Objective Caption**: Updated to composite metric
   - Before: "inner_mean_min_per_class_f1 (averaged across inner folds)"
   - After: **"composite (65% min F1 + 35% plurality correctness) — averaged across inner folds"**

4. **Overall Figure Height**: Increased from 8.5 → **9.2 inches** to accommodate all spacing improvements

---

### ✅ Figure 3: Optuna Optimization (`figure3_optuna_optimization_v4.py`)

**Issues Fixed:**
1. **Y-axis Labels** (all 3 panels): 
   - Before: "Objective: inner_mean_min_per_class_f1 (%)"
   - After: **"Objective: composite (%)"**

2. **Bottom Caption**: Added objective definition
   - Before: "TPE sampler provides Bayesian optimization. MedianPruner enables early stopping. Winner from each stage passed to next stage."
   - After: Same + **"Objective: composite (65% min F1 + 35% plurality correctness)."**

---

### ✅ Figure 4: Confusion Matrices (`figure4_confusion_matrices_v4.py`)

**Issues Fixed:**
1. **Bottom Text Occlusion**:
   - Increased bottom margin: 0.20 → **0.24**
   - Moved footnote up: y=0.02 → **y=0.03**
   - Increased colorbar pad: 0.15 → **0.18**
   - Result: No overlap between footnote and x-axis labels

---

### ✅ Figure 5: Learning Curves (`figure5_learning_curves_v4.py`)

**Issues Fixed:**
1. **Bottom Text Occlusion**:
   - Moved footnote up significantly: y=0.01 → **y=0.03**
   - Result: Clear separation from bottom panel (D) x-axis

---

### ✅ Figure 10: Performance Box Plots (`figure10_performance_boxplots_v4.py`)

**Issues Fixed:**
1. **Bottom Text Occlusion**:
   - Increased bottom margin: 0.12 → **0.15**
   - Moved footnote up: y=0.01 → **y=0.03**
   - Result: No overlap with panel labels or tick marks

---

## 🧠 What We Learned (Mentorship Insights)

### 1. **Understanding the Composite Objective**
Your new optimization objective balances two critical goals:
- **65% weight on min F1**: Ensures **all classes are decodable** (no class left behind)
- **35% weight on plurality correctness**: Ensures correct predictions are **plurality** for each class

**Why plurality correctness matters:**
```
Example confusion matrix:
        Pred:  1   2   3
True 1: [100  30  20]  ← max is 100 (diagonal) ✓
True 2: [ 40  60  50]  ← max is 60 (diagonal) ✓
True 3: [ 10  80  30]  ← max is 80 (OFF-diagonal) ✗

Plurality Correctness = 2/3 = 67% (two classes have correct as plurality)
```
This prevents systematic misclassification (e.g., always predicting "3" when answer is "2").

### 2. **Data Preprocessing Pipeline Structure**
Your preprocessing created **18 datasets** systematically:
- **3 high-pass filters**: 1.0, 1.5, 2.0 Hz (removes slow drift)
- **3 low-pass filters**: 35, 40, 45 Hz (removes high-frequency noise)
- **2 baseline modes**: on/off (baseline correction)
- Total: 3 × 3 × 2 = **18 combinations**

This systematic exploration helps find optimal preprocessing parameters for your EEG data.

### 3. **Channel Intersection Logic**
```
128 raw channels 
  → Exclude 28 non-scalp channels (E1, E8, E14, ..., E128)
  → Keep ~100 channels
  → Compute intersection across ALL subjects (some subjects may have bad channels)
  → Final: ~100 channels common to all subjects
```
The intersection happens **after** exclusion but **across** subjects to ensure all subjects have the same channel set for the model.

### 4. **Real Subject IDs vs. Sequential Numbering**
- You collected data from 24 subjects
- But original IDs: 02, 03, 04, 05, 08, 09, ..., 33 (gaps = excluded/missing subjects)
- **For publication**: Use real IDs to show data provenance
- **For analysis**: Can use sequential 0-23 indexing internally

### 5. **Matplotlib Layout Gotchas**
When using `constrained_layout=True` (modern matplotlib):
- **DON'T** use `fig.subplots_adjust()` — they conflict!
- **DO** use `fig.text()` with higher y-values (0.03+ instead of 0.01)
- **DO** increase figure size if needed (`figsize=(10, 9.2)`)

### 6. **Publication Figure Best Practices**
✅ **Always do:**
- White backgrounds
- Colorblind-safe palettes (Wong colors)
- 600 DPI for PNG, vector for PDF/SVG
- Fonts embedded in PDFs
- Clear spacing (no text occlusion)
- Adequate margins for footnotes

✅ **Systematic approach to fixing occlusion:**
1. Increase figure height/margins
2. Move text up (increase y-values)
3. Increase spacing between elements
4. Test with actual output files (not just preview)

---

## 📁 Output Files

All regenerated figures saved to: `publication-ready-media/outputs/v4/`

### Generated Files:
```
✅ figure1_pipeline_v4.pdf/png/svg
✅ figure2_nested_cv_v4.pdf/png/svg
✅ figure3_optuna_optimization.pdf/png/svg
✅ figure4_confusion_matrices_v4.pdf/png/svg
✅ figure5_learning_curves.pdf/png/svg
✅ figure10_performance_boxplots_V4.pdf/png/svg
```

**Total:** 18 files (6 figures × 3 formats)

---

## 🔍 Quality Checks

### ✅ All Figures Pass:
- [x] No text occlusion
- [x] Correct data labels (subject IDs, objective names)
- [x] Consistent terminology across figures
- [x] Adequate margins and spacing
- [x] 600 DPI PNG output
- [x] Vector PDF with embedded fonts
- [x] Editable SVG for fine-tuning

### ✅ Content Accuracy:
- [x] Real subject IDs (02-33 with gaps, N=24)
- [x] Correct trial count (~300, not ~360)
- [x] Complete preprocessing dataset list (18 datasets)
- [x] Updated objective metric throughout
- [x] Correct evaluation mode (refit, not ensemble)

---

## 🎓 Mentor Tips for Future

### When Updating Figures:
1. **Find related text**: Use grep/search for related terms across all figure files
2. **Update consistently**: Same terminology everywhere (e.g., "composite objective")
3. **Test spacing**: Actually view the output files (PNG/PDF) to verify no occlusion
4. **Document changes**: Keep notes like this summary for reproducibility

### For Your Methods Section:
You can now accurately state:
> "We optimized a composite objective function (65% minimum per-class F1 + 35% plurality correctness) to ensure both class decodability and prediction distinctness. Hyperparameter search was conducted in three stages (Architecture, Learning, Augmentation) using Tree-structured Parzen Estimator (TPE) sampling with median pruning. Final evaluation used Leave-One-Subject-Out cross-validation (N=24 subjects, IDs 02-33 with gaps due to quality control) with refit models on each outer fold's training set."

### For Version Control:
Consider creating a git tag for this stable figure set:
```bash
git add publication-ready-media/code/v4_neuroscience/*.py
git add publication-ready-media/outputs/v4/*
git commit -m "Fix: Updated figures 1-10 with composite objective, real subject IDs, and occlusion fixes"
git tag figures-v4-final
```

---

## 🚀 Next Steps

Your figures are now **publication-ready**! Consider:

1. **Review with PI**: Show updated figures, especially:
   - Figure 1 (pipeline) — verify 18 preprocessing datasets are correct
   - Figure 2 (nested CV) — confirm real subject IDs are appropriate to show
   
2. **Update manuscript text**: Ensure Methods section matches figure details

3. **Generate remaining figures** (6, 7, 8, 9) if they need similar updates

4. **Archive this version**: Keep these outputs safe as "submission version"

---

## 📚 Files Modified

### Python Scripts:
1. `publication-ready-media/code/v4_neuroscience/figure1_pipeline_v4.py`
2. `publication-ready-media/code/v4_neuroscience/figure2_nested_cv_v4.py`
3. `publication-ready-media/code/v4_neuroscience/figure3_optuna_optimization_v4.py`
4. `publication-ready-media/code/v4_neuroscience/figure4_confusion_matrices_v4.py`
5. `publication-ready-media/code/v4_neuroscience/figure5_learning_curves_v4.py`
6. `publication-ready-media/code/v4_neuroscience/figure10_performance_boxplots_v4.py`

### Output Files (overwritten):
- All 18 output files in `publication-ready-media/outputs/v4/`

---

## ✨ Conclusion

All requested figure fixes have been completed successfully! The figures now accurately represent your:
- Data collection (300 trials, 24 subjects with real IDs)
- Preprocessing pipeline (18 systematic parameter combinations)
- Optimization objective (composite metric balancing decodability and distinctness)
- Evaluation strategy (LOSO with refit models)

**Your figures are ready for manuscript submission!** 🎉

---

*Generated: October 5, 2025*  
*Session: Publication Figure Fixes*
