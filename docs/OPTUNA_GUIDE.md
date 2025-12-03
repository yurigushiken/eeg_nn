# Optuna Guide: Hyperparameter Optimization

Complete guide to using Optuna for systematic hyperparameter search in the EEG neural decoding pipeline.

## Overview

The pipeline uses **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler for efficient hyperparameter optimization. The search is divided into three stages:

1. **Stage 1:** Architecture exploration (kernel sizes, filter counts, dropout)
2. **Stage 2:** Recipe refinement (learning rate, batch size, warmup schedules)
3. **Stage 3:** Augmentation tuning (optional, probabilities and magnitudes)

## Quick Start

### Run Stage 1 Search

```powershell
python -X utf8 -u scripts/optuna_search.py \
  --stage step1 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --cfg configs/tasks/cardinality_1_3/step1_search.yaml \
  --space configs/tasks/cardinality_1_3/step1_space_scaffold.yaml \
  --trials 48
```

**Expected time:** 8-24 hours (48 trials, GPU-dependent)

### Generate Reports

```powershell
python scripts/refresh_optuna_summaries.py
```

Or double-click: `results\optuna\refresh_all_studies.bat`

**Outputs:**
- `!all_trials-<study>.csv` — All trials sorted by objective
- `<study>_parallel_plot.html` — Interactive parameter exploration
- `<study>_top3_report.html` — Top-3 configurations with plots
- `optuna_runs_index.csv` — Global index across all studies

## Three-Stage Search Strategy

### Stage 1: Architecture Exploration

**Goal:** Find optimal network structure (large levers that have biggest impact).

**Parameters searched:**
- `F1`: Number of temporal filters (4, 8, 16, 32)
- `D`: Depthwise multiplier (1, 2)
- `kernel_1`: Temporal kernel size (16, 32, 64)
- `kernel_2`: Depthwise kernel size (4, 8, 16)
- `dropout`: Dropout rate (0.25, 0.5, 0.75)
- `pool_mode`: Pooling type ("mean", "max")

**Search space example (`step1_space_scaffold.yaml`):**
```yaml
F1:
  type: choice
  options: [8, 16, 32]

D:
  type: choice
  options: [1, 2]

kernel_1:
  type: choice
  options: [16, 32, 64]

dropout:
  type: choice
  options: [0.25, 0.5, 0.75]
```

**What to do after Stage 1:**
1. Run `python scripts/refresh_optuna_summaries.py`
2. Open `results/optuna/<study>_top3_report.html`
3. Check Top-1 trial hyperparameters
4. Create `step1_winner.yaml` with best config
5. Proceed to Stage 2

### Stage 2: Recipe Refinement

**Goal:** Fine-tune training dynamics (learning rate, batch size, warmup schedules).

**Parameters searched:**
- `lr`: Learning rate (log-uniform, e.g., 1e-4 to 1e-2)
- `batch_size`: Batch size (16, 32, 64)
- `weight_decay`: L2 regularization (0.0, 1e-5, 1e-4, 1e-3)
- `lr_warmup_frac`: LR warmup fraction (0.0, 0.05, 0.1, 0.2)
- `lr_warmup_init`: Initial LR scale (0.1, 0.3, 0.5)
- `aug_warmup_frac`: Augmentation warmup (0.0, 0.1, 0.2)

**Search space example (`step2_space_recipe.yaml`):**
```yaml
lr:
  type: loguniform
  low: 0.0001
  high: 0.01

batch_size:
  type: choice
  options: [16, 32, 64]

weight_decay:
  type: choice
  options: [0.0, 0.00001, 0.0001, 0.001]
```

**Base config:** Use `step1_winner.yaml` as `--base` (architecture fixed, only recipe varies).

**What to do after Stage 2:**
1. Refresh Optuna summaries
2. Extract Top-1 trial from Step 2 report
3. Create `step2_winner.yaml` (merge Step 1 + Step 2 winners)
4. Optionally proceed to Stage 3, or go straight to final evaluation

### Stage 3: Augmentation Tuning (Optional)

**Goal:** Optimize augmentation strategies for generalization.

**Parameters searched:**
- `noise_p`, `noise_std`: Noise injection
- `shift_p`, `shift_max_frac`: Temporal shift
- `time_mask_p`, `time_mask_frac`: Temporal masking
- `mixup_alpha`: Mixup interpolation

**Search space example (`step3_space_joint.yaml`):**
```yaml
noise_p:
  type: uniform
  low: 0.0
  high: 0.8

time_mask_p:
  type: uniform
  low: 0.0
  high: 0.8

mixup_alpha:
  type: choice
  options: [0.0, 0.1, 0.2, 0.4]
```

**Base config:** Use `step2_winner.yaml` as `--base`.

**When to skip Stage 3:** If Step 2 winner already uses augmentations and performs well.

## Optuna Objective (Pruning & Selection)

**Key parameter:** `optuna_objective` in your config determines what metric Optuna optimizes.

**Recommended:** `composite_min_f1_plur_corr` (see [Configuration Reference](CONFIGURATION.md#optimization-objectives-explained))

**How it works:**
1. **Per-epoch pruning:** After each training epoch, Optuna checks if trial is underperforming compared to others. If yes, prunes (stops early).
2. **Final objective:** After training completes, Optuna records the mean inner validation performance for the configured objective.
3. **TPE sampler:** Uses Bayesian optimization to suggest next trial parameters based on past trials.

**Pruner settings (in search controller YAML):**
```yaml
optuna_pruner: MedianPruner
optuna_pruner_kwargs:
  n_warmup_steps: 10  # Don't prune before epoch 10
```

## Search Controller Config

The `--cfg` argument points to a YAML that configures Optuna study settings.

**Example (`step1_search.yaml`):**
```yaml
optuna_study_name: step1_cardinality_1_3_scaffold
optuna_sampler: TPESampler
optuna_sampler_kwargs:
  seed: 42  # Reproducible sampling
optuna_pruner: MedianPruner
optuna_pruner_kwargs:
  n_warmup_steps: 10
optuna_objective: composite_min_f1_plur_corr  # Metric to optimize
composite_min_f1_weight: 0.65  # Weight for composite objective
```

**Key fields:**
- `optuna_study_name`: Unique study identifier (appears in database filename)
- `optuna_sampler`: Always `TPESampler` (Bayesian optimization)
- `optuna_sampler_kwargs.seed`: Seed for reproducible trial suggestions
- `optuna_pruner`: `MedianPruner` (stops underperforming trials early)
- `optuna_objective`: Metric to optimize (must be explicitly set)

## Search Space Syntax

Search spaces are defined in YAML using Optuna's distribution types.

### Choice (Categorical)

```yaml
param_name:
  type: choice
  options: [value1, value2, value3]
```

**Example:** `dropout: {type: choice, options: [0.25, 0.5, 0.75]}`

### Uniform (Continuous)

```yaml
param_name:
  type: uniform
  low: 0.0
  high: 1.0
```

**Example:** `noise_std: {type: uniform, low: 0.01, high: 0.1}`

### Log-Uniform (Continuous, Log Scale)

```yaml
param_name:
  type: loguniform
  low: 0.0001
  high: 0.01
```

**Example:** `lr: {type: loguniform, low: 0.0001, high: 0.01}`

**Use log-uniform for:** Learning rates, weight decay, or any parameter spanning orders of magnitude.

### Integer Choice

```yaml
param_name:
  type: choice
  options: [16, 32, 64, 128]
```

**Example:** `batch_size: {type: choice, options: [16, 32, 64]}`

## Performance Optimization

### Enable Dataset Caching (CRITICAL)

In `configs/common.yaml`:
```yaml
dataset_cache_memory: true
```

**Effect:** First trial builds dataset, subsequent trials reuse (10-100× speedup per trial).

### Use Fewer Folds for Search

```yaml
n_folds: 3           # Instead of LOSO
inner_n_folds: 2     # Minimum
```

**Effect:** Faster per-trial evaluation. Validate winners with LOSO afterwards.

### Reduce Epochs for Search

```yaml
epochs: 50           # Instead of 100
early_stop: 10       # Instead of 20
```

**Effect:** Faster trial completion, may miss subtle convergence differences.

### Parallelize Trials (Multi-GPU)

Run multiple Optuna processes pointing to the same database:

```powershell
# Terminal 1 (GPU 0)
$env:CUDA_VISIBLE_DEVICES=0
python scripts/optuna_search.py ... --trials 24

# Terminal 2 (GPU 1)
$env:CUDA_VISIBLE_DEVICES=1
python scripts/optuna_search.py ... --trials 24
```

**Total:** 48 trials, ~2× faster with 2 GPUs.

**Caveat:** Ensure `optuna_study_name` is identical (shares same DB).

## Interpreting Reports

### All Trials CSV

`!all_trials-<study>.csv` contains one row per trial, sorted by objective (best first).

**Key columns:**
- `trial_number`: Optuna trial ID
- `inner_mean_min_per_class_f1`: Min-F1 (worst class)
- `inner_mean_plur_corr`: Plurality correctness
- `composite`: Composite objective (if using `composite_min_f1_plur_corr`)
- `F1`, `D`, `kernel_1`, ...: Hyperparameter values
- `run_dir`: Path to trial's run directory

**Usage:** Extract Top-1 row, copy hyperparameters to winner YAML.

### Parallel Coordinate Plot

Interactive visualization of hyperparameter → performance relationships.

**How to read:**
- Each vertical axis = one hyperparameter or metric
- Each line = one trial (colored by objective value)
- Bright lines = high-performing trials
- Filter by clicking/brushing to highlight configurations

**What to look for:**
- **Converging lines:** Parameters that consistently appear in top trials
- **Diverging lines:** Parameters with little effect on performance
- **Clusters:** Multiple good configurations in similar parameter regions

**Example insights:**
- All top trials have `dropout=0.5` → important parameter
- Top trials span `batch_size` 16-64 → less important

### Top-3 Report

HTML report with:
1. Hyperparameter table (Top 3 trials side-by-side)
2. Inner vs outer metrics comparison
3. Confusion matrices (using enhanced plots with per-class F1)
4. Learning curves

**Use case:** Quick visual comparison of top configurations before selecting winner.

**Prioritization:** Sorted by `inner_mean_min_per_class_f1` (worst-class focus) to ensure all classes are decodable.

## Common Issues

### Study Already Exists Error

**Symptom:** `DuplicateStudyError: Study 'xxx' already exists`

**Cause:** Running same study name twice.

**Solutions:**
1. **Continue existing study:** Omit `--trials` to resume
2. **Start new study:** Change `optuna_study_name` in controller YAML
3. **Delete old study:** Remove `results/optuna/<study>.db`

### No Trials Pruned

**Symptom:** All trials complete full training (no pruning).

**Causes:**
1. `n_warmup_steps` too high (pruning disabled too long)
2. All trials perform similarly (no clear underperformers)

**Solutions:**
1. Reduce `n_warmup_steps` to 5-10
2. Widen search space to create more performance variance

### Refresh Script Fails

**Symptom:** `FileNotFoundError` when running `refresh_optuna_summaries.py`

**Cause:** No `.db` files in `results/optuna/` or `optuna_tools/` module missing.

**Solution:**
1. Check `results/optuna/` contains `*.db` files
2. Ensure you've run at least one Optuna search

### Top-3 Report Missing

**Symptom:** CSV and parallel plots exist, but no Top-3 report.

**Cause:** Fewer than 3 completed trials, or report generation failed.

**Solution:**
1. Run more trials (need ≥3)
2. Check console output for PDF generation errors (requires Playwright for PDF, but HTML always generated)

## Advanced Usage

### Resuming an Interrupted Search

Optuna automatically resumes from the database:

```powershell
# Original run (interrupted at trial 20/48)
python scripts/optuna_search.py ... --trials 48

# Resume (will run trials 21-48)
python scripts/optuna_search.py ... --trials 48
```

**Note:** Ensure `--cfg` and `--space` are identical.

### Manual Trial Selection

Instead of using Top-1 automatically, manually select a trial based on:
- Inner vs outer generalization gap (prefer small gap)
- Per-class F1 balance (prefer uniform F1 across classes)
- Training stability (check learning curves for smoothness)

**Where to find:** Top-3 report shows all these metrics.

### Multi-Objective Optimization (Not Yet Supported)

Current implementation optimizes single objective. To optimize multiple objectives (e.g., accuracy AND model size):
- Run search on primary objective
- Post-hoc filter trials by secondary constraint
- Or implement multi-objective Optuna sampler (requires code changes)

## File Outputs

### Database (`.db`)

**Location:** `results/optuna/<study_name>.db`

**Format:** SQLite database (Optuna's internal storage)

**Usage:** Automatically loaded by `scripts/optuna_search.py` and `refresh_optuna_summaries.py`

**Warning:** Do not manually edit. Corruption will break study.

### Trial Run Directories

Each trial creates a run directory: `results/runs/<timestamp>_<task>_<engine>_trial<N>_*/`

**Contents:** Same as regular training runs (checkpoints, metrics, plots)

**Usage:** Inspect individual trial results, reuse `resolved_config.yaml`

### Global Index

**Location:** `results/optuna/optuna_runs_index.csv`

**Purpose:** Aggregate all trials across all studies (for meta-analysis).

**Columns:** Study name, trial number, all hyperparameters, all metrics, run directory path.

## Next Steps

- For defining search spaces, see [Configuration Reference](CONFIGURATION.md)
- For complete search workflow, see [Workflows](WORKFLOWS.md#hyperparameter-search-three-stage-optuna)
- For interpreting hyperparameter importance, see plots in `<study>_parallel_plot.html`
