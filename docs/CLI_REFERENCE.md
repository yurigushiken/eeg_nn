# CLI Reference

Complete command-line interface reference for all scripts in the pipeline.

## Core Scripts

### `train.py` - Main Training Entry Point

Single or multi-seed training runs with optional XAI generation.

**Basic Usage:**
```powershell
python -X utf8 -u train.py \
  --task <task_name> \
  --engine eeg \
  --base <config_yaml>
```

**Arguments:**
- `--task` (str, required): Task name (e.g., `cardinality_1_3`, `landing_on_2_3`)
- `--engine` (str, required): Engine type (always `eeg` for this project)
- `--base` (str, required): Path to base config YAML
- `--set` (str, optional, repeatable): Override config values (e.g., `--set seed=42 batch_size=16`)
- `--run-xai` (flag, optional): Run XAI analysis after training completes

**Examples:**

```powershell
# Single LOSO run
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/base.yaml

# Override hyperparameters
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --set seed=99 batch_size=16 lr=0.0005

# Force LOSO (override n_folds in config)
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml \
  --set n_folds=null

# Run with automatic XAI generation
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --run-xai

# Multi-seed run (define seeds in YAML)
# In YAML: seeds: [41, 42, 43, 44, 45]
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/multiseed.yaml

# Refit mode (no early stop during refit)
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml \
  --set outer_eval_mode=refit refit_val_frac=0.0
```

**Outputs:**
- Run directory: `results/runs/<timestamp>_<task>_<engine>_<...>/`
- Key files: `summary_*.json`, `outer_eval_metrics.csv`, `test_predictions_outer.csv`, `resolved_config.yaml`

---

### `scripts/optuna_search.py` - Hyperparameter Search

Unified Optuna driver for Stage 1/2/3 hyperparameter optimization.

**Basic Usage:**
```powershell
python -X utf8 -u scripts/optuna_search.py \
  --stage <step1|step2|step3> \
  --task <task_name> \
  --base <base_config> \
  --cfg <search_controller> \
  --space <search_space> \
  --trials <n_trials>
```

**Arguments:**
- `--stage` (str, required): Search stage (`step1`, `step2`, or `step3`)
- `--task` (str, required): Task name
- `--base` (str, required): Base config YAML
- `--cfg` (str, required): Search controller YAML (defines study settings)
- `--space` (str, required): Search space YAML (defines parameter ranges)
- `--trials` (int, required): Number of trials to run

**Examples:**

```powershell
# Stage 1: Architecture exploration
python -X utf8 -u scripts/optuna_search.py \
  --stage step1 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --cfg configs/tasks/cardinality_1_3/step1_search.yaml \
  --space configs/tasks/cardinality_1_3/step1_space_scaffold.yaml \
  --trials 48

# Stage 2: Recipe refinement (use step1 winner as base)
python -X utf8 -u scripts/optuna_search.py \
  --stage step2 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/step1_winner.yaml \
  --cfg configs/tasks/cardinality_1_3/step2_search.yaml \
  --space configs/tasks/cardinality_1_3/step2_space_recipe.yaml \
  --trials 48

# Stage 3: Augmentation tuning (optional)
python -X utf8 -u scripts/optuna_search.py \
  --stage step3 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/step2_winner.yaml \
  --cfg configs/tasks/cardinality_1_3/step3_search.yaml \
  --space configs/tasks/cardinality_1_3/step3_space_joint.yaml \
  --trials 48
```

**Outputs:**
- SQLite database: `results/optuna/<study_name>.db`
- Per-trial run directories
- Use `refresh_optuna_summaries.py` to generate reports

---

### `scripts/run_xai_analysis.py` - Explainability Analysis

Generate Integrated Gradients attributions, topomaps, and time-frequency analysis.

**Basic Usage:**
```powershell
python -X utf8 -u scripts/run_xai_analysis.py \
  --run-dir "<path_to_run_directory>"
```

**Arguments:**
- `--run-dir` (str, required): Path to completed training run directory

**Examples:**

```powershell
# Generate XAI for a completed run
python -X utf8 -u scripts/run_xai_analysis.py \
  --run-dir "results\runs\20251010_094949_landing_on_2_3_eeg_step1_t026"

# Windows path with spaces (use quotes)
python -X utf8 -u scripts/run_xai_analysis.py \
  --run-dir "results\runs\My Run Directory"
```

**Outputs (in `<run_dir>/xai_analysis/`):**
- `grand_average_xai_heatmap.png` — Channel×time attribution heatmap
- `grand_average_xai_topoplot.png` — Scalp topography
- `grand_average_time_frequency.png` — Time-frequency decomposition
- `grand_average_per_class/` — Per-class attributions
- `consolidated_xai_report.html` — Interactive report
- `consolidated_xai_report.pdf` — Printable version (requires Playwright)

**Requirements:**
- Run must have completed successfully (checkpoints exist)
- Montage required for topoplots (`net/AdultAverageNet128_v1.sfp`)

---

### `scripts/run_posthoc_stats.py` - Post-Hoc Statistics

Quantify population-level efficacy and subject-level reliability.

**Basic Usage:**
```powershell
python -X utf8 -u scripts/run_posthoc_stats.py \
  --run-dir "<path_to_run_directory>"
```

**Arguments:**
- `--run-dir` (str, required): Path to completed run
- `--alpha` (float, optional): Significance threshold (default: 0.05)
- `--multitest` (str, optional): Multiple testing correction (`fdr` or `none`, default: `fdr`)
- `--glmm` (flag, optional): Run GLMM analysis (requires R with lme4)
- `--forest` (flag, optional): Generate forest plots
- `--chance-rate` (float, optional): Chance rate (auto-detected if omitted)

**Examples:**

```powershell
# Basic run (uses defaults from configs/posthoc_defaults.yaml)
python -X utf8 -u scripts/run_posthoc_stats.py \
  --run-dir "results\runs\<run_dir>"

# Override defaults
python -X utf8 -u scripts/run_posthoc_stats.py \
  --run-dir "results\runs\<run_dir>" \
  --alpha 0.01 \
  --multitest none \
  --glmm \
  --forest

# Custom chance rate (e.g., 6-class task)
python -X utf8 -u scripts/run_posthoc_stats.py \
  --run-dir "results\runs\<run_dir>" \
  --chance-rate 0.1667
```

**Outputs (in `<run_dir>/stats/`):**
- `group_stats.json` — Mean + 95% CI, permutation p-values
- `per_subject_significance.csv` — Per-subject binomial tests
- `per_subject_forest.png` — Forest plot with Wilson CIs
- `glmm_summary.json` — Fixed-effect summary (if `--glmm`)
- `glmm_caterpillar.png` — Subject BLUPs (if `--glmm`)

---

### `scripts/analyze_nloso_subject_performance.py` - Subject Performance Analysis

Detailed per-subject and per-fold performance breakdown.

**Basic Usage:**
```powershell
python -X utf8 -u scripts/analyze_nloso_subject_performance.py \
  "<path_to_run_directory>"
```

**Arguments:**
- Positional argument (str, required): Path to completed run

**Examples:**

```powershell
# Analyze subject performance
python -X utf8 -u scripts/analyze_nloso_subject_performance.py \
  "results\runs\<run_dir>"
```

**Outputs (in `<run_dir>/subject_performance/`):**
- `per_subject_metrics.csv` — Per-subject accuracy + support
- `per_fold_metrics.csv` — Per-fold accuracy + support
- `acc_by_subject_bar.png` — Bar charts with annotations
- `overall_confusion.png` — Overall confusion matrix
- `per_subject_confusion/` — Individual confusion matrices
- `inner_vs_outer/` — Inner vs outer comparison (if both CSVs exist)
- `report.html` — Consolidated HTML report

**If decision layer enabled:**
- `thresholded/` subfolder with side-by-side comparisons

---

### `scripts/prepare_from_happe.py` - Data Preparation

One-time conversion from HAPPE EEGLAB `.set` files to MNE `.fif` epochs.

**Basic Usage:**
```powershell
python scripts/prepare_from_happe.py
```

**No arguments required.** Configuration is hardcoded in the script (you may need to edit paths).

**Expected Input Structure:**
```
data_input_from_happe/<dataset>/5 - processed .../Subject*.set
data_behavior/data_UTF8/Subject*.csv
net/AdultAverageNet128_v1.sfp
```

**Output:**
```
data_preprocessed/<dataset>/sub-XX_preprocessed-epo.fif
```

**What it does:**
1. Loads HAPPE-cleaned `.set` files (128-channel EEG)
2. Aligns with behavioral CSV (trial-level metadata)
3. Removes `Condition==99` trials
4. Encodes labels for each task
5. Attaches montage (`net/AdultAverageNet128_v1.sfp`)
6. Saves per-subject `.fif` epochs

---

### `scripts/final_eval.py` - Multi-Seed Final Evaluation

Multi-seed final evaluation with aggregation (legacy script, prefer `train.py` with `seeds` in YAML).

**Basic Usage:**
```powershell
python scripts/final_eval.py \
  --task <task_name> \
  --cfg <config_yaml> \
  --seeds <n_seeds>
```

**Arguments:**
- `--task` (str, required): Task name
- `--cfg` (str, required): Config YAML
- `--seeds` (int, required): Number of seeds to run
- `--use-best` (flag, optional): Merge best hyperparameters from Optuna stages

**Examples:**

```powershell
# Run 10 seeds with base config
python scripts/final_eval.py \
  --task cardinality_1_3 \
  --cfg configs/tasks/cardinality_1_3/base.yaml \
  --seeds 10

# Use best hyperparameters from Optuna
python scripts/final_eval.py \
  --task cardinality_1_3 \
  --cfg configs/tasks/cardinality_1_3/base.yaml \
  --seeds 10 \
  --use-best
```

**Outputs:**
- One run directory per seed
- Aggregate JSON with cross-seed statistics

---

### `scripts/refresh_optuna_summaries.py` - Optuna Report Generation

Regenerate Optuna study reports, plots, and top-3 summaries.

**Basic Usage:**
```powershell
python scripts/refresh_optuna_summaries.py
```

**Or use Windows batch wrapper:**
```powershell
results\optuna\refresh_all_studies.bat
```

**What it does:**
1. Discovers all Optuna `.db` files in `results/optuna/`
2. For each study:
   - Exports `!all_trials-<study>.csv` (sorted by objective)
   - Generates parallel coordinate plots (HTML + PNG)
   - Creates Top-3 report (HTML + PDF) with enhanced confusion matrices
   - Builds global index: `optuna_runs_index.csv`

**Outputs (per study):**
- `!all_trials-<study>.csv` — All trials sorted by performance
- `<study>_parallel_plot.html` — Interactive parallel coordinates
- `<study>_parallel_plot.png` — Static version (thick lines, high DPI)
- `<study>_top3_report.html` — Top-3 report with metrics/plots
- `<study>_top3_report.pdf` — Printable version

**Global output:**
- `results/optuna/optuna_runs_index.csv` — All trials across all studies

---

## Permutation Testing Workflow

Run permutation test to generate empirical null distribution.

**Step 1: Complete observed run**
```powershell
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/base.yaml
```

**Step 2: Run permutation test (reuse splits)**
```powershell
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml \
  --permute-labels \
  --n-permutations 200 \
  --permute-scope within_subject \
  --permute-stratified \
  --permute-seed 123 \
  --observed-run-dir "results\runs\<observed_run_dir_name>"
```

**Permutation Arguments:**
- `--permute-labels` (flag): Enable permutation mode
- `--n-permutations` (int): Number of permutations (e.g., 200)
- `--permute-scope` (str): `within_subject` or `global`
- `--permute-stratified` (flag): Preserve within-subject class balance
- `--permute-seed` (int): Seed for reproducibility
- `--observed-run-dir` (str): Path to observed run (to reuse splits)

**Outputs (next to observed run):**
- `<observed_run>_perm_test_results.csv` — Per-permutation metrics
- `<observed_run>_perm_summary.json` — Empirical p-values

---

## Common Workflows

### End-to-End Research Workflow

```powershell
# 1. Prepare data (one-time)
python scripts/prepare_from_happe.py

# 2. Stage 1 search: Architecture
python -X utf8 -u scripts/optuna_search.py \
  --stage step1 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --cfg configs/tasks/cardinality_1_3/step1_search.yaml \
  --space configs/tasks/cardinality_1_3/step1_space_scaffold.yaml \
  --trials 48

# 3. Refresh Optuna reports
python scripts/refresh_optuna_summaries.py

# 4. Extract step1 winner, create step1_winner.yaml manually

# 5. Stage 2 search: Recipe refinement
python -X utf8 -u scripts/optuna_search.py \
  --stage step2 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/step1_winner.yaml \
  --cfg configs/tasks/cardinality_1_3/step2_search.yaml \
  --space configs/tasks/cardinality_1_3/step2_space_recipe.yaml \
  --trials 48

# 6. Refresh Optuna reports again
python scripts/refresh_optuna_summaries.py

# 7. Extract step2 winner, create final resolved config

# 8. Multi-seed final evaluation
# (Add seeds: [41, 42, 43, 44, 45] to resolved config)
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml

# 9. Generate XAI for one seed
python -X utf8 -u scripts/run_xai_analysis.py \
  --run-dir "results\runs\<seed_41_run_dir>"

# 10. Run post-hoc stats for one seed
python -X utf8 -u scripts/run_posthoc_stats.py \
  --run-dir "results\runs\<seed_41_run_dir>" \
  --glmm --forest

# 11. Analyze subject performance
python -X utf8 -u scripts/analyze_nloso_subject_performance.py \
  "results\runs\<seed_41_run_dir>"

# 12. Permutation test
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml \
  --permute-labels \
  --n-permutations 200 \
  --permute-scope within_subject \
  --permute-stratified \
  --observed-run-dir "results\runs\<seed_41_run_dir>"
```

---

## PowerShell Tips

### Line Continuation

Use backtick (`) for line continuation:
```powershell
python train.py `
  --task cardinality_1_3 `
  --engine eeg `
  --base configs/tasks/cardinality_1_3/base.yaml
```

### Paths with Spaces

Always use quotes:
```powershell
python train.py --run-dir "results\runs\My Run"
```

### Looping Over Seeds

```powershell
# Multi-seed loop (legacy method)
for ($i=0; $i -lt 10; $i++) {
  $seed = 42 + $i
  python -X utf8 -u train.py `
    --task cardinality_1_3 `
    --engine eeg `
    --base configs/tasks/cardinality_1_3/resolved_config.yaml `
    --set seed=$seed
}
```

**Better:** Use `seeds` list in YAML instead.

---

## Environment Variables

### UTF-8 Encoding (Windows)

Always use `-X utf8` for Python on Windows:
```powershell
python -X utf8 -u train.py ...
```

Or set environment variable:
```powershell
$env:PYTHONUTF8=1
```

### CUDA Settings (Determinism)

Set for reproducibility:
```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

Already handled internally by the pipeline, but can be set explicitly.

---

## Troubleshooting CLI Issues

**Problem:** `ModuleNotFoundError`
**Solution:** Ensure conda environment is activated: `conda activate eegnex-env`

**Problem:** PowerShell doesn't recognize backtick line continuation
**Solution:** Ensure no trailing spaces after backtick

**Problem:** Paths not found (Windows)
**Solution:** Use forward slashes (`/`) or double backslashes (`\\`) in Python args

**Problem:** `--set` not working
**Solution:** Check syntax: `--set key=value` (no spaces around `=`)

**Problem:** Run directory not found
**Solution:** Use absolute paths or ensure working directory is project root

For more help, see [Troubleshooting Guide](TROUBLESHOOTING.md).
