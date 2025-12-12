# Temporal RSA Analysis - Quick Start Guide

**IMPORTANT:** Run these commands AFTER all temporal training completes (1,035 runs)

---

## Status Check

```powershell
# 1. Count completed runs (should be 1,035)
(ls results\runs\rsa_temporal_v1\).Count

# 2. Check all 3 seeds present
ls results\runs\rsa_temporal_v1\ | Select-String "seed_49" | Measure-Object
ls results\runs\rsa_temporal_v1\ | Select-String "seed_50" | Measure-Object
ls results\runs\rsa_temporal_v1\ | Select-String "seed_51" | Measure-Object
# Each should show 345 (15 pairs × 23 windows)
```

---

## Complete Analysis Pipeline (One Command)

```powershell
conda activate eegnex-env
cd D:\eeg_nn

python scripts/rsa/run_temporal_analysis_pipeline.py `
  --runs-dir results/runs/rsa_temporal_v1 `
  --fullepoch-comparison results/runs/rsa_matrix_v1/stats_summary.csv `
  --output-dir results/runs/rsa_temporal_v1
```

**This single command will:**
1. Extract per-subject accuracies from predictions (24,840 data points)
2. Average across seeds (→ 8,280 subject means)
3. Run 345 statistical tests with FDR correction
4. Generate all figures (6 PNGs + 1 GIF)
5. Generate all tables (4 CSVs)
6. Compute peaks, onsets, and comparisons

**Runtime:** ~10-15 minutes

---

## Step-by-Step (If You Want Control)

### Step 1: Extract Subject Data (~3 min)

```powershell
python scripts/rsa/extract_subject_temporal_accuracies.py `
  --runs-dir results/runs/rsa_temporal_v1 `
  --output results/runs/rsa_temporal_v1/subject_temporal_accuracies.csv
```

**Output:**
- `subject_temporal_accuracies.csv` (24,840 rows)
- `subject_temporal_means.csv` (8,280 rows, seeds averaged)

### Step 2: Statistical Analysis (~2 min)

```powershell
python scripts/rsa/analyze_temporal_stats.py `
  --subject-data results/runs/rsa_temporal_v1/subject_temporal_means.csv `
  --output-dir results/runs/rsa_temporal_v1 `
  --baseline 50 `
  --fdr-alpha 0.05 `
  --min-cluster-size 2
```

**Output:**
- `tables/temporal_stats_all_tests.csv` (345 rows with FDR-corrected p-values)
- `stats/fdr_correction_log.txt`
- `stats/cluster_analysis.csv`

### Step 3: Temporal Curves (~1 min)

```powershell
python scripts/rsa/visualize_temporal_curves.py `
  --subject-data results/runs/rsa_temporal_v1/subject_temporal_means.csv `
  --stats-data results/runs/rsa_temporal_v1/tables/temporal_stats_all_tests.csv `
  --output-dir results/runs/rsa_temporal_v1/figures
```

**Output:**
- `temporal_curves_all_pairs.png`
- `temporal_curves_grouped.png`

### Step 4: RDM Evolution (~2 min)

```powershell
python scripts/rsa/visualize_temporal_rdm_evolution.py `
  --subject-data results/runs/rsa_temporal_v1/subject_temporal_means.csv `
  --output-dir results/runs/rsa_temporal_v1/figures `
  --create-gif
```

**Output:**
- `temporal_rdm_evolution.gif`
- `temporal_rdm_snapshots.png`

### Step 5: Peaks & Onsets (~1 min)

```powershell
python scripts/rsa/analyze_temporal_peaks.py `
  --subject-data results/runs/rsa_temporal_v1/subject_temporal_means.csv `
  --stats-data results/runs/rsa_temporal_v1/tables/temporal_stats_all_tests.csv `
  --fullepoch-data results/runs/rsa_matrix_v1/stats_summary.csv `
  --output-dir results/runs/rsa_temporal_v1
```

**Output:**
- `tables/temporal_peaks_summary.csv`
- `tables/temporal_onset_latencies.csv`
- `tables/temporal_peaks_vs_fullepoch.csv`
- `figures/peak_timing_vs_ratio.png`
- `figures/onset_latencies_boxplot.png`

---

## Verify Results

```powershell
# Check all outputs created
Test-Path results\runs\rsa_temporal_v1\subject_temporal_means.csv
Test-Path results\runs\rsa_temporal_v1\tables\temporal_stats_all_tests.csv
Test-Path results\runs\rsa_temporal_v1\figures\temporal_curves_all_pairs.png
Test-Path results\runs\rsa_temporal_v1\figures\temporal_rdm_evolution.gif

# Count significant tests (FDR-corrected)
$stats = Import-Csv results\runs\rsa_temporal_v1\tables\temporal_stats_all_tests.csv
($stats | Where-Object { $_.significant_fdr -eq 'True' }).Count
# Expected: 200-300 / 345
```

---

## Run Tests (Verify Implementation)

```powershell
# Run test suite
python -m pytest tests/test_temporal_rsa_pipeline.py -v

# Expected: All tests PASS
```

---

## Key Results to Check

1. **Peak Timing (1v2 pair):**
   - Open: `tables/temporal_peaks_summary.csv`
   - Look for: Peak ~185ms, Accuracy ~62%

2. **Onset Latencies:**
   - Open: `tables/temporal_onset_latencies.csv`
   - Look for: Onsets between 120-180ms

3. **Temporal Curves:**
   - Open: `figures/temporal_curves_grouped.png`
   - Look for: Clear peaks in 150-250ms range

4. **FDR Correction:**
   - Open: `stats/fdr_correction_log.txt`
   - Verify: Some tests survived correction

---

## Troubleshooting

**"File not found" errors:**
```powershell
# Make sure training completed
ls results\runs\rsa_temporal_v1\ | Measure-Object
# Should show 1,035
```

**"No module named 'scripts.rsa.extract_subject_temporal_accuracies'":**
- Implementation scripts not created yet
- Wait for implementation phase to complete

**"Tests fail":**
```powershell
# Run with verbose output to see which test failed
python -m pytest tests/test_temporal_rsa_pipeline.py -v --tb=long
```

---

## Documentation

Full details in:
- **Implementation Plan:** `docs/TEMPORAL_RSA_IMPLEMENTATION_PLAN.md`
- **Test Suite:** `tests/test_temporal_rsa_pipeline.py`
- **Usage Guide:** This file

---

**YOU ARE HERE:** Tests written ✓, Implementation pending, Training in progress

**NEXT:** Wait for training to complete, then run the single pipeline command above.

**Good luck with your career! This analysis will be rigorous and publication-ready.**
