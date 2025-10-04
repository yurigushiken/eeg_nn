# Neuroscience Publication Standards - Implementation Guide

**Date:** 2025-10-04  
**Purpose:** Ensure ALL figures meet neuroscience journal standards  
**Status:** ⚠️ CORRECTION NEEDED - Remove dark themes!

---

## 🔬 **OFFICIAL NEUROSCIENCE JOURNAL STANDARDS**

### **Journals Researched:**
- Journal of Neuroscience (JNeurosci / SfN)
- Nature Neuroscience
- Neuron (Cell Press)
- eNeuro (SfN open-access)
- Journal of Cognitive Neuroscience
- Cerebral Cortex

---

## ✅ **PUBLICATION REQUIREMENTS**

### **1. Background & Theme**
- ✅ **WHITE or light gray background ONLY**
- ❌ **NO dark themes** (dark backgrounds are for presentations ONLY, not publications!)
- ✅ Clean, professional appearance
- ✅ High contrast between elements

### **2. Color Requirements**
- ✅ **RGB mode** (not CMYK) for digital/web
- ✅ **Colorblind-safe palettes** (Wong, Tol, ColorBrewer)
- ✅ **Avoid red-green combinations alone**
- ✅ **Grayscale-legible** (test by converting to B&W)
- ✅ **Conservative colors** (no neon, no rainbow/jet colormaps)
- ✅ **Consistent across all figures** in manuscript

**Recommended Palettes:**
1. **Wong (Nature Methods)** - ✅ CURRENTLY USING, PERFECT!
2. **Viridis/Plasma** - For heatmaps (NOT rainbow!)
3. **Blues** - Single-hue sequential
4. **RdBu** - Diverging (for difference maps)

### **3. Resolution & Format**
- ✅ **300 DPI minimum** for photographs/images
- ✅ **600 DPI** for line art, graphs, text
- ✅ **1200 DPI** for fine detail (optional)
- ✅ **Vector format** (PDF, EPS, SVG) preferred for:
  - Line graphs, bar charts, flowcharts, schematics
- ✅ **TIFF** for:
  - Photographs, microscopy, heatmaps with complex gradients
- ✅ **Embed fonts** in PDFs (fonttype 42)

### **4. Typography**
- ✅ **Arial or Helvetica** (sans-serif) ONLY
- ✅ **8-10 pt** at final size (our 9pt is perfect!)
- ✅ **No decorative fonts**
- ✅ **No bold** within figure (only in axis labels/panel labels)
- ✅ **Consistent sizing** across all panels
- ✅ **Black text** on white background (high contrast)

### **5. Figure Dimensions**
- ✅ **Single column:** 89 mm (3.5 inches) width
- ✅ **Double column:** 183 mm (7.2 inches) width
- ✅ **Maximum height:** 247 mm (9.7 inches)
- ✅ Our figures: 7-7.2" width = double column ✓

### **6. Panel Labels**
- ✅ **Bold uppercase** (A, B, C, D)
- ✅ **Top-left corner** of each panel
- ✅ **10-12 pt** size
- ✅ **Consistent placement** across all figures

### **7. Statistical Indicators**
- ✅ `*` for p < 0.05
- ✅ `**` for p < 0.01
- ✅ `***` for p < 0.001
- ✅ `ns` for not significant
- ✅ Exact p-values in caption or text box
- ✅ Error bars with definition (SEM, SD, CI)

### **8. Accessibility Requirements**
- ✅ **Colorblind-safe** (protanopia, deuteranopia, tritanopia)
- ✅ **Grayscale-legible** (prints well in B&W)
- ✅ **High contrast** (4.5:1 minimum for text)
- ✅ **Large enough text** (readable at 100%)
- ✅ **Clear symbols** (don't rely on color alone)

---

## ⚠️ **CRITICAL CORRECTION NEEDED**

### **PROBLEM:** Dark Theme Figures Are NOT Publication-Standard!

**Figures with DARK backgrounds (need to be regenerated):**
1. ❌ `figure1_pipeline_v2` - Dark theme (Nord palette)
2. ❌ `figure2_nested_cv_v2` - Dark theme (Nord palette)
3. ❌ `figure5_learning_curves` - Dark theme (Nord palette)
4. ❌ `figure10_performance_boxplots_v3` - Dark theme (Nord palette)

**Why this is wrong:**
- Dark backgrounds are for **presentations/posters ONLY**
- Scientific journals require **WHITE backgrounds**
- Print publications cannot handle dark backgrounds well
- Reviewers expect standard formatting

### **SOLUTION:** Regenerate These 4 Figures with Light Theme

**Use:** `pub_style_v3.py` (NOT `pub_style_dark.py`)  
**Result:** White background, Wong palette, publication-ready

---

## ✅ **FIGURES THAT ARE ALREADY CORRECT**

**These figures already meet neuroscience standards:**
1. ✅ `figure3_optuna_optimization` - Light theme, Wong palette
2. ✅ `figure4_confusion_matrices_v3` - Light theme, Blues colormap
3. ✅ `figure6_permutation_v3` - Light theme, Wong palette
4. ✅ `figure7_per_subject_forest_v3` - Light theme, Wong palette
5. ✅ `figure8_xai_spatiotemporal` - Light theme, Magma colormap
6. ✅ `figure9_xai_perclass` - Light theme, Wong + RdBu

**No changes needed for these 6 figures!**

---

## 🔧 **ACTION ITEMS**

### **Immediate (Required):**
1. **Delete dark theme scripts:**
   - `figure1_pipeline_v3_dark.py` → Use V2 light version
   - `figure2_nested_cv_v3_dark.py` → Use V2 light version
   - `figure5_learning_curves_v3_dark.py` → Use V2 light version
   - `figure10_performance_boxplots_v3_dark.py` → Regenerate with V3 light

2. **Regenerate 4 figures with LIGHT theme:**
   - Copy V2 scripts (already light theme!)
   - Ensure they output to `v3_final/`
   - Use `pub_style_v3.py` (white background)

3. **Update master generation script:**
   - Remove "dark theme" section
   - All figures should use light theme (Wong palette)

### **Optional Enhancements:**
1. **Add scale bars** (if applicable to EEG plots)
2. **Standardize error bar notation** in captions
3. **Test grayscale conversion** for all figures
4. **Verify font embedding** in all PDFs

---

## 📊 **RECOMMENDED FINAL FIGURE SET**

**All with WHITE backgrounds (publication standard):**

1. ✅ **Pipeline** - Light theme (fix)
2. ✅ **Nested CV** - Light theme (fix)
3. ✅ **Optuna** - Light theme ✓ (already correct)
4. ✅ **Confusion Matrices** - Light theme ✓ (already correct)
5. ✅ **Learning Curves** - Light theme (fix)
6. ✅ **Permutation** - Light theme ✓ (already correct)
7. ✅ **Per-Subject** - Light theme ✓ (already correct)
8. ✅ **XAI Spatiotemporal** - Light theme ✓ (already correct)
9. ✅ **XAI Per-Class** - Light theme ✓ (already correct)
10. ✅ **Box Plots** - Light theme (fix)

---

## 📚 **CITATIONS & REFERENCES**

**Color Guidelines:**
- Wong, B. (2011). Points of view: Color blindness. *Nature Methods*, 8(6), 441.
- ColorBrewer 2.0: https://colorbrewer2.org/

**Figure Standards:**
- Journal of Neuroscience: https://www.jneurosci.org/content/information-authors
- Nature Neuroscience: https://www.nature.com/neuro/for-authors
- eNeuro: https://www.eneuro.org/content/information-authors

**Accessibility:**
- WCAG 2.1 Guidelines (Web Content Accessibility Guidelines)
- Section 508 Standards (US Federal accessibility)

---

## ✅ **COMPLIANCE CHECKLIST**

**Format & Resolution:**
- [ ] All figures have WHITE backgrounds
- [x] 600 DPI raster output
- [x] Vector PDF with embedded fonts
- [x] RGB color mode
- [x] Appropriate dimensions (7-7.2" width)

**Color & Accessibility:**
- [x] Colorblind-safe palette (Wong) ✓
- [ ] All figures tested in grayscale
- [x] High contrast (black text on white)
- [x] No rainbow/jet colormaps
- [x] Consistent colors across figures

**Typography:**
- [x] Sans-serif font (DejaVu Sans = Arial equivalent)
- [x] 9pt body text ✓
- [x] 8pt annotations ✓
- [x] No decorative fonts
- [x] Consistent sizing

**Elements:**
- [x] Bold panel labels (A, B, C)
- [x] Clear axis labels with units
- [x] Statistical indicators where appropriate
- [x] Error bars defined
- [x] Legends outside plots (no occlusion)

---

## 💡 **WHY WHITE BACKGROUNDS?**

**Scientific Publishing Reasons:**
1. **Print compatibility** - Ink/toner costs, paper stock
2. **Historical convention** - 100+ years of scientific literature
3. **Reviewer expectations** - Dark backgrounds signal "presentation," not "publication"
4. **Accessibility** - Better for photocopying, scanning, archiving
5. **Journal requirements** - Most explicitly require light backgrounds

**When to use dark themes:**
- ✅ Conference presentations
- ✅ Poster sessions
- ✅ PowerPoint slides
- ✅ Web/digital-only content
- ❌ **NOT for journal submissions!**

---

## 🎯 **SUMMARY**

**What's correct:** 6/10 figures (Figs 3, 4, 6, 7, 8, 9)  
**What needs fixing:** 4/10 figures (Figs 1, 2, 5, 10) - regenerate with LIGHT theme  

**Action:** Remove dark theme, use light theme (white background) for ALL figures

**Your Wong palette + white background figures are PERFECT for neuroscience journals!**

---

**Version:** Publication Standards v1.0  
**Last Updated:** 2025-10-04  
**Status:** ⚠️ ACTION REQUIRED - Remove dark themes

