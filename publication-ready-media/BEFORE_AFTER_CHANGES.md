# Before & After: Figure Corrections Quick Reference

## 🔄 Figure 1: Pipeline

### Before → After

| Element | Before | After |
|---------|--------|-------|
| **Raw Data** | ~360 trials/subject | ✅ **~300 trials/subject** |
| **Preprocessing** | Generic "Bandpass 1.5–40 Hz" | ✅ **"18 datasets: 3 HPF × 3 LPF × 2 baseline"** |
| **Data Finalization** | "Behavioral alignment" | ✅ **"Behavioral & condition alignment"** |
| **Channel Count** | "100 ch" | ✅ **"~100 ch"** (clarified as intersection) |
| **Objective** | "inner_mean_min_per_class_f1" | ✅ **"composite (65% min F1 + 35% diagonal dominance)"** |
| **Final Eval** | "Ensemble predictions" | ✅ **"Refit predictions"** |
| **Layout** | Stats & XAI side-by-side | ✅ **Separate rows (Stats above XAI)** |
| **Colors** | Gradient blues | ✅ **Lighter uniform blues** |

---

## 🔄 Figure 2: Nested CV

### Before → After

| Element | Before | After |
|---------|--------|-------|
| **Subject IDs (Panel A)** | S1, S2, S3, ..., S24 | ✅ **S02, S03, S04, S05, S08, ..., S33** (real IDs) |
| **Subject IDs (Panel B)** | S1-S24 | ✅ **S02-S33** (with gaps) |
| **Fold Labels (Panel B)** | Occluding boxes | ✅ **More space, moved up** |
| **Inner Fold Labels (Panel C)** | Too close to bottom | ✅ **Moved up, adequate clearance** |
| **Figure Height** | 8.5 inches | ✅ **9.2 inches** |
| **Objective Caption** | "inner_mean_min_per_class_f1" | ✅ **"composite (65% min F1 + 35% diagonal dominance)"** |

**Key Learning:** Always use real subject IDs in publication figures for data provenance!

---

## 🔄 Figure 3: Optuna Optimization

### Before → After

| Element | Before | After |
|---------|--------|-------|
| **Y-axis Label (Panel A)** | "inner_mean_min_per_class_f1 (%)" | ✅ **"composite (%)"** |
| **Y-axis Label (Panel B)** | "inner_mean_min_per_class_f1 (%)" | ✅ **"composite (%)"** |
| **Y-axis Label (Panel C)** | "inner_mean_min_per_class_f1 (%)" | ✅ **"composite (%)"** |
| **Bottom Caption** | Generic optimization note | ✅ **Added: "Objective: composite (65% min F1 + 35% diagonal dominance)"** |

---

## 🔄 Figure 4: Confusion Matrices

### Before → After

| Element | Before | After |
|---------|--------|-------|
| **Bottom Margin** | 0.20 | ✅ **0.24** |
| **Footnote Y-position** | 0.02 | ✅ **0.03** |
| **Colorbar Pad** | 0.15 | ✅ **0.18** |
| **Result** | Text occlusion | ✅ **Clear spacing, no occlusion** |

---

## 🔄 Figure 5: Learning Curves

### Before → After

| Element | Before | After |
|---------|--------|-------|
| **Footnote Y-position** | 0.01 | ✅ **0.03** |
| **Result** | Occluding bottom panel | ✅ **Clear separation from x-axis** |

---

## 🔄 Figure 10: Performance Box Plots

### Before → After

| Element | Before | After |
|---------|--------|-------|
| **Bottom Margin** | 0.12 | ✅ **0.15** |
| **Footnote Y-position** | 0.01 | ✅ **0.03** |
| **Result** | Text occlusion | ✅ **Clear spacing, no occlusion** |

---

## 📊 Summary Statistics

- **Figures Modified:** 6 (1, 2, 3, 4, 5, 10)
- **Total Changes:** 24 individual fixes
- **Output Files Generated:** 18 (6 figures × 3 formats)
- **Lines of Code Modified:** ~60 lines across 6 files

---

## 🎯 Key Improvements by Category

### Content Accuracy (9 fixes)
✅ Trial count (300 not 360)  
✅ Preprocessing datasets (18 total)  
✅ Real subject IDs (02-33)  
✅ Composite objective (6 instances updated)  
✅ Refit vs ensemble  
✅ Condition alignment added  

### Layout & Spacing (8 fixes)
✅ Separated Stats/XAI boxes  
✅ Increased figure heights  
✅ Moved text up to prevent occlusion  
✅ Adjusted margins (3 figures)  
✅ Increased padding (colorbars, etc.)  

### Visual Design (3 fixes)
✅ Lighter uniform colors (Figure 1)  
✅ Better label positioning (Figure 2)  
✅ Consistent spacing throughout  

### Terminology Consistency (4 fixes)
✅ "composite" objective everywhere  
✅ "~100 ch" with clarification  
✅ "Behavioral & condition alignment"  
✅ Footnote consistency  

---

## 🚀 Impact

### Before Fixes:
- ❌ Incorrect trial count (360 vs 300)
- ❌ Generic subject numbering (S1-S24)
- ❌ Missing preprocessing details
- ❌ Outdated objective metric references
- ❌ Text occlusions in 4 figures
- ❌ Confusing layout (Stats/XAI equivalence)

### After Fixes:
- ✅ **Publication-ready accuracy**
- ✅ **Real data provenance (subject IDs)**
- ✅ **Complete methodology transparency**
- ✅ **Current metric terminology**
- ✅ **No occlusions, clear readability**
- ✅ **Logical sequential flow**

---

## 💡 Best Practices Demonstrated

1. **Data Transparency**: Use real subject IDs, show gaps
2. **Terminology Consistency**: Update metric names everywhere
3. **Layout Testing**: Check actual outputs for occlusions
4. **Systematic Fixes**: Address related issues across all figures
5. **Documentation**: Keep detailed change logs

---

## 📝 Checklist for Next Time

When updating publication figures:

- [ ] Find all instances of old terminology (use grep/search)
- [ ] Update consistently across all figures
- [ ] Check for occlusions in actual output files (not just preview)
- [ ] Verify data accuracy (counts, IDs, parameters)
- [ ] Test all three formats (PDF, PNG, SVG)
- [ ] Document changes for reproducibility
- [ ] Commit to version control with clear message

---

*Quick Reference Guide - October 5, 2025*
