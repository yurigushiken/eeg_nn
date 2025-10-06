# What to Look For: Updated Figures Visual Guide

## 🔍 How to Verify the Fixes

Open each figure and check these specific elements:

---

## 📊 Figure 1: Pipeline (`figure1_pipeline_v4.png`)

### Look for these changes:

**Top Box (Raw Data Acquisition):**
```
✅ Should say: "24 subjects, ~300 trials/subject, 128-channel EEG"
❌ NOT: "~360 trials/subject"
```

**Second Box (Preprocessing):**
```
✅ Should say: "18 datasets: 3 HPF (1.0/1.5/2.0 Hz) × 3 LPF (35/40/45 Hz) × 2 baseline (on/off)"
❌ NOT: Generic "Bandpass 1.5–40 Hz" only
```

**Third Box (Data Finalization):**
```
✅ Should say: "Behavioral & condition alignment · Min trials/class · Channel intersection (~100 ch)"
❌ NOT: "Behavioral alignment" only
```

**Optuna Section (bottom of outer box):**
```
✅ Should say: "Objective: composite (65% min F1 + 35% plurality correctness) — ensures decodability & distinctness"
❌ NOT: "inner_mean_min_per_class_f1 (ensures all classes decodable)"
```

**Final Evaluation Box:**
```
✅ Should say: "Refit predictions"
❌ NOT: "Ensemble predictions"
```

**Statistical Validation & XAI:**
```
✅ Should be: Two separate boxes, Statistical Validation ABOVE XAI
❌ NOT: Side-by-side on same row
```

**Overall Colors:**
```
✅ Should be: Lighter, uniform blues throughout
❌ NOT: Darker gradient blues
```

---

## 📊 Figure 2: Nested CV (`figure2_nested_cv_v4.png`)

### Look for these changes:

**Top Panel (All Subjects):**
```
✅ Should show: S02, S03, S04, S05, S08, S09, S10, S11, S12, S13, S14, S15, S17, S21, S22, S23, S25, S26, S27, S28, S29, S31, S32, S33
❌ NOT: S1, S2, S3, ..., S24
```

**Middle Panel (Outer Loop):**
- Labels "Fold 1", "Fold 12", "Fold 24" should have adequate space (not overlapping boxes)
- Subject boxes should show real IDs (S02, S03, etc.)

**Bottom Panel (Inner Loop):**
- "Inner Fold 1" through "Inner Fold 5" labels should be clearly visible
- Should NOT overlap with bottom caption

**Bottom Caption:**
```
✅ Should say: "Objective: composite (65% min F1 + 35% plurality correctness) — averaged across inner folds"
❌ NOT: "inner_mean_min_per_class_f1 (averaged across inner folds)"
```

**Overall Figure:**
- Should be taller than before (more vertical spacing)
- No text occlusions anywhere

---

## 📊 Figure 3: Optuna Optimization (`figure3_optuna_optimization.png`)

### Look for these changes:

**Panel A Y-axis:**
```
✅ Should say: "Objective: composite (%)"
❌ NOT: "Objective: inner_mean_min_per_class_f1 (%)"
```

**Panel B Y-axis:**
```
✅ Should say: "Objective: composite (%)"
❌ NOT: "Objective: inner_mean_min_per_class_f1 (%)"
```

**Panel C Y-axis:**
```
✅ Should say: "Objective: composite (%)"
❌ NOT: "Objective: inner_mean_min_per_class_f1 (%)"
```

**Bottom Caption:**
```
✅ Should end with: "...Objective: composite (65% min F1 + 35% plurality correctness)."
❌ NOT: Just "Winner from each stage passed to next stage." (without objective definition)
```

---

## 📊 Figure 4: Confusion Matrices (`figure4_confusion_matrices_v4.png`)

### Look for these changes:

**Bottom Spacing:**
```
✅ The footnote "Values show row-normalized percentages. Chance = 33.3%..." should have clear space above colorbar
❌ NOT: Overlapping with colorbar or x-axis labels
```

**Visual Check:**
- There should be visible white space between the colorbar and the footnote text
- No text should touch or overlap

---

## 📊 Figure 5: Learning Curves (`figure5_learning_curves.png`)

### Look for these changes:

**Bottom Spacing:**
```
✅ The footnote "Mean across 24 outer folds (LOSO-CV)..." should have clear space below Panel D
❌ NOT: Overlapping with Panel D's x-axis labels
```

**Visual Check:**
- Clear separation between bottom panels and footnote
- All text easily readable

---

## 📊 Figure 10: Performance Box Plots (`figure10_performance_boxplots_V4.png`)

### Look for these changes:

**Bottom Spacing:**
```
✅ The footnote "Box plots show median (line), IQR (box)..." should have clear space below the panels
❌ NOT: Overlapping with Panel C or D x-axis labels
```

**Visual Check:**
- Adequate margin at bottom of figure
- No text touching panel edges

---

## ✅ Quick Validation Checklist

Print this and check off as you review each figure:

### Figure 1: Pipeline
- [ ] "~300 trials" (not ~360)
- [ ] "18 datasets" mentioned
- [ ] "Behavioral & condition alignment"
- [ ] "composite (65% min F1 + 35% plurality correctness)"
- [ ] "Refit predictions" (not Ensemble)
- [ ] Stats/XAI on separate rows
- [ ] Lighter blue colors

### Figure 2: Nested CV
- [ ] Real subject IDs visible (S02, S03, ..., S33)
- [ ] No text occlusions (fold labels clear)
- [ ] "composite" objective in caption
- [ ] Taller figure with more spacing

### Figure 3: Optuna
- [ ] Y-axis says "composite (%)" (all 3 panels)
- [ ] Bottom caption includes objective definition

### Figure 4: Confusion
- [ ] Clear space above colorbar
- [ ] Footnote not touching anything

### Figure 5: Learning Curves
- [ ] Footnote clearly separated from panels

### Figure 10: Box Plots
- [ ] Adequate bottom margin
- [ ] Footnote not touching panels

---

## 🎨 Color Comparison (Figure 1)

### Before (Darker Gradient):
- Very dark blues (#1565C0, #1976D2)
- Strong color gradient top to bottom

### After (Lighter Uniform):
- Lighter blues (#D6E9F5, #B8D9EE, #A8D0EA)
- More uniform, easier to read
- Better for printing and projectors

---

## 📏 Spacing Comparison (Figure 2)

### Before:
```
[Inner Fold 5 label]
[boxes.....................]
[Objective caption]  ← TOO CLOSE! Overlapping!
```

### After:
```
[Inner Fold 5 label]
[boxes.....................]

[Objective caption]  ← Clear space!
```

---

## 🔢 Subject ID Comparison (Figure 2)

### Before:
```
S1  S2  S3  S4  S5  S6  S7  S8  ...  S24
```
(Generic sequential numbering)

### After:
```
S02 S03 S04 S05 S08 S09 S10 S11 ... S33
```
(Real IDs with gaps showing excluded subjects)

---

## 💡 Why These Changes Matter

### Scientific Accuracy
✅ **~300 trials**: Reflects actual data collection  
✅ **Real subject IDs**: Shows data provenance and quality control  
✅ **18 datasets**: Demonstrates systematic preprocessing exploration  
✅ **Composite objective**: Documents your novel optimization approach  

### Readability
✅ **No occlusions**: Professional appearance, meets journal standards  
✅ **Lighter colors**: Better for slides, printing, accessibility  
✅ **Clear spacing**: Easier to read, more professional  

### Reproducibility
✅ **Complete preprocessing info**: Others can replicate your approach  
✅ **Detailed objective**: Clearly defines what was optimized  
✅ **Explicit methods**: LOSO with refit (not ensemble)  

---

## 🎯 Next Steps After Verification

Once you've confirmed all fixes are correct:

1. **Review with your team**: Show Figures 1-2 especially (most changes)
2. **Update manuscript Methods**: Ensure text matches figures
3. **Archive these versions**: Tag in git as "submission-ready"
4. **Prepare figure legends**: Write detailed captions for each
5. **Consider submitting**: Your figures are publication-ready!

---

## 🚨 If Something Looks Wrong

If any figure doesn't show the expected changes:

1. **Check timestamps**: Figures should show 10/05/2025 modification date
2. **Clear cache**: Close and reopen the file
3. **Check file format**: Make sure you're viewing the v4 folder
4. **Regenerate if needed**: Re-run the Python script for that figure

---

*Visual Verification Guide - October 5, 2025*  
*All changes verified and documented*
