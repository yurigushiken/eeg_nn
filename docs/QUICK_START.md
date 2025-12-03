# Quick Start Guide

Get up and running with EEG neural decoding in 5 minutes.

## Prerequisites

- Windows 10/11 with PowerShell
- Anaconda or Miniconda installed
- Git installed
- ~10GB free disk space

## Installation

```powershell
# Clone the repository
git clone https://github.com/yourusername/eeg_nn.git
cd eeg_nn

# Create conda environment (takes ~5 minutes)
conda env create -f environment.yml

# Activate environment
conda activate eegnex-env

# Optional: Enable PDF report generation
playwright install
```

## Data Preparation (One-Time Setup)

If you have HAPPE-preprocessed EEG data:

```powershell
# Convert EEGLAB .set files to MNE .fif format
python scripts/prepare_from_happe.py
```

**Expected input structure:**
- `data_input_from_happe/<dataset>/5 - processed .../*.set` (HAPPE output)
- `data_behavior/data_UTF8/SubjectXX.csv` (behavioral data)
- `net/AdultAverageNet128_v1.sfp` (electrode montage)

**Output:**
- `data_preprocessed/<dataset>/sub-XX_preprocessed-epo.fif` (one file per subject)

## Run Your First Experiment

### Option 1: Single LOSO Run (Recommended for Testing)

```powershell
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/base.yaml
```

**What this does:**
- Trains EEGNeX model using Leave-One-Subject-Out cross-validation
- Tests on held-out subjects (prevents data leakage)
- Writes results to `results/runs/<timestamp>_cardinality_1_3_eeg_*`

**Expected runtime:** ~10-30 minutes per fold (depends on GPU)

### Option 2: Hyperparameter Search (For Research)

```powershell
python -X utf8 -u scripts/optuna_search.py \
  --stage step1 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --cfg configs/tasks/cardinality_1_3/step1_search.yaml \
  --space configs/tasks/cardinality_1_3/step1_space_scaffold.yaml \
  --trials 48
```

**What this does:**
- Runs 48 hyperparameter configurations using Optuna TPE
- Searches architecture parameters (kernel sizes, dropout, etc.)
- Saves study database to `results/optuna/`

**Expected runtime:** ~8-24 hours (depends on `--trials` and GPU)

## View Results

After training completes:

```powershell
# Navigate to run directory
cd results/runs/<your_run_directory>

# Key files to check:
# - summary_<task>_<engine>.json          → Overall metrics
# - outer_eval_metrics.csv                → Per-fold performance
# - plots_outer/overall_confusion.png     → Confusion matrix
# - test_predictions_outer.csv            → Trial-level predictions
```

## Generate XAI Analysis

```powershell
python -X utf8 -u scripts/run_xai_analysis.py \
  --run-dir "results\runs\<your_run_directory>"
```

**Output:** `xai_analysis/` folder with:
- Grand-average topomaps (which channels matter most)
- Time-frequency decomposition
- Per-class attribution heatmaps
- Consolidated HTML report

## Common First-Run Issues

**Problem:** `FileNotFoundError: data_preprocessed/...`
**Solution:** Run `python scripts/prepare_from_happe.py` first

**Problem:** `CUDA out of memory`
**Solution:** Reduce `batch_size` in your config (e.g., from 32 to 16)

**Problem:** `AssertionError: Subject overlap detected`
**Solution:** This is a safeguard—check your data has correct `SubjectID` column

**Problem:** Slow training
**Solution:** Enable dataset caching in `configs/common.yaml`: `dataset_cache_memory: true`

## Next Steps

- **Try different tasks:** Change `--task` to `landing_on_2_3` or `cardinality_1_6`
- **Multi-seed runs:** See [Workflows Guide](WORKFLOWS.md#multi-seed-evaluation)
- **Permutation testing:** See [Statistics Guide](STATISTICS.md#permutation-testing)
- **Customize config:** See [Configuration Reference](CONFIGURATION.md)

## Quick Command Reference

```powershell
# Train with custom seed
python train.py --task cardinality_1_3 --engine eeg --base configs/... --set seed=42

# Train with custom hyperparameters
python train.py --task ... --base ... --set batch_size=16 lr=0.001

# Run XAI automatically after training
python train.py --task ... --base ... --run-xai

# Check Optuna study results
python scripts/refresh_optuna_summaries.py
```

For complete command documentation, see [CLI Reference](CLI_REFERENCE.md).
