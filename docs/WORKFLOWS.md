# Workflows

Common usage patterns and complete workflows for the EEG neural decoding pipeline.

## Quick Workflows

### 1. Single Evaluation Run (Testing)

**Goal:** Quickly validate a configuration with LOSO cross-validation.

```powershell
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/base.yaml
```

**Expected time:** 10-30 minutes per fold (GPU-dependent)

**Outputs:** `results/runs/<timestamp>_*/` with confusion matrix, metrics, predictions

---

### 2. Multi-Seed Evaluation

**Goal:** Generate stable performance estimates across multiple random seeds.

**Method 1: YAML config (recommended)**

Add to your config YAML:
```yaml
seeds: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
```

Then run once:
```powershell
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml
```

**Outputs:**
- One run directory per seed (e.g., `seed_41_crop_ms_...`)
- Aggregate JSON: `<timestamp>_seeds_aggregate.json`

**Method 2: PowerShell loop (legacy)**
```powershell
for ($i=0; $i -lt 10; $i++) {
  $seed = 42 + $i
  python -X utf8 -u train.py \
    --task cardinality_1_3 \
    --engine eeg \
    --base configs/tasks/cardinality_1_3/resolved_config.yaml \
    --set seed=$seed
}
```

---

### 3. Hyperparameter Search (Three-Stage Optuna)

**Goal:** Systematically optimize architecture, training recipe, and augmentations.

**Stage 1: Architecture Exploration**

Search large levers (kernel sizes, filter counts, dropout).

```powershell
python -X utf8 -u scripts/optuna_search.py \
  --stage step1 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --cfg configs/tasks/cardinality_1_3/step1_search.yaml \
  --space configs/tasks/cardinality_1_3/step1_space_scaffold.yaml \
  --trials 48
```

**Expected time:** 8-24 hours (48 trials)

**What to do next:**
1. Run `python scripts/refresh_optuna_summaries.py`
2. Check `results/optuna/<study>_top3_report.html`
3. Extract top trial hyperparameters, create `step1_winner.yaml`

**Stage 2: Recipe Refinement**

Search learning rate, batch size, warmups, weight decay.

```powershell
python -X utf8 -u scripts/optuna_search.py \
  --stage step2 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/step1_winner.yaml \
  --cfg configs/tasks/cardinality_1_3/step2_search.yaml \
  --space configs/tasks/cardinality_1_3/step2_space_recipe.yaml \
  --trials 48
```

**What to do next:**
1. Refresh Optuna summaries again
2. Extract top trial, create `step2_winner.yaml`

**Stage 3: Augmentation Tuning (Optional)**

Fine-tune augmentation probabilities and magnitudes.

```powershell
python -X utf8 -u scripts/optuna_search.py \
  --stage step3 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/step2_winner.yaml \
  --cfg configs/tasks/cardinality_1_3/step3_search.yaml \
  --space configs/tasks/cardinality_1_3/step3_space_joint.yaml \
  --trials 48
```

**Final step:** Extract best config, create `resolved_config.yaml` for final evaluation.

---

### 4. Permutation Testing Workflow

**Goal:** Generate empirical null distribution to assess statistical significance.

**Step 1: Run observed (real-label) training**

```powershell
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml
```

Note the run directory name (e.g., `20251203_120000_cardinality_1_3_eeg_...`).

**Step 2: Run permutation test (shuffled labels, same splits)**

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
  --observed-run-dir "results\runs\20251203_120000_cardinality_1_3_eeg_..."
```

**Expected time:** ~200× the observed run time (parallelizable if you have multiple GPUs).

**Outputs:**
- `<observed_run>_perm_test_results.csv` — 200 rows (one per permutation)
- `<observed_run>_perm_summary.json` — Empirical p-values

**Interpretation:** Compare observed accuracy/F1 to null distribution. p < 0.05 indicates above-chance performance.

---

### 5. Complete XAI Analysis

**Goal:** Generate explainability analysis (attributions, topomaps, time-frequency).

```powershell
python -X utf8 -u scripts/run_xai_analysis.py \
  --run-dir "results\runs\<your_run_directory>"
```

**Expected time:** 5-15 minutes

**Outputs (in `<run_dir>/xai_analysis/`):**
- Grand-average heatmaps and topomaps
- Per-class attributions
- Time-frequency decomposition
- Top-2 spatio-temporal events
- Consolidated HTML/PDF report

**Requirements:**
- Completed training run with checkpoints
- Montage file: `net/AdultAverageNet128_v1.sfp`

---

### 6. Post-Hoc Statistical Analysis

**Goal:** Quantify group-level significance and subject-level reliability.

```powershell
python -X utf8 -u scripts/run_posthoc_stats.py \
  --run-dir "results\runs\<your_run_directory>" \
  --alpha 0.05 \
  --multitest fdr \
  --glmm \
  --forest
```

**Expected time:** 1-5 minutes (longer if GLMM enabled and R is slow)

**Outputs (in `<run_dir>/stats/`):**
- Group-level 95% CI
- Per-subject binomial tests (FDR-corrected)
- Forest plot with Wilson CIs
- GLMM fixed-effect summary (if `--glmm`)
- Caterpillar plot of subject BLUPs

**Interpretation:** Check `per_subject_significance.csv` for proportion of subjects above chance.


### 7. RSA Binary Matrix (Cardinality 11-66)

**Goal:** Train subject-aware models for every cross-digit cardinality pair and assemble a resumable master dataset.

```powershell
python -X utf8 -u scripts/run_rsa_matrix.py `
  --config configs/tasks/rsa_binary.yaml `
  --output-dir results\runs\rsa_matrix_v1
```

- Uses the locked hyperparameters in `configs/tasks/rsa_binary.yaml`.
- For seeds `[42, 43, 44]`, trains one run per pair (skips same-digit contrasts).
- Stores runs as `results\runs\rsa_matrix_v1\<launch_id>_rsa_<pair>_seed_<N>/`.
- Writes `results\runs\rsa_matrix_v1\rsa_matrix_results.csv` (quick summary).

**Pause / Resume support:**
- The first launch creates `results\runs\rsa_matrix_v1\.rsa_resume_state.json` with the active launch id and progress.
- If the process is interrupted, resume in-place:

```powershell
python -X utf8 -u scripts/run_rsa_matrix.py `
  --config configs/tasks/rsa_binary.yaml `
  --output-dir results\runs\rsa_matrix_v1 `
  --resume
```

- The script reuses the existing launch id, skips combinations that already produced an `outer_eval_metrics.csv` with an `OVERALL` row, and reruns any incomplete directories.
- If multiple launch ids live in the same folder, pass `--resume-launch-id <id>` to choose one explicitly.
- After every completed combination the state file is updated atomically; once all pairs finish the file is marked complete. Remove it manually if you want to start a fresh run in the same folder.

**Compile master dataset:**
```powershell
python -X utf8 -u scripts/compile_rsa_results.py `
  --runs-dir results\runs\rsa_matrix_v1 `
  --output results\runs\rsa_matrix_v1\rsa_results_master.csv
```

This produces `rsa_results_master.csv` with columns `ClassA`, `ClassB`, `Seed`, `Subject`, `Fold`, `RecordType`, `Accuracy`, `MacroF1`, `MinClassF1`.

**Visualization (OVERALL rows by default):**
```powershell
python -X utf8 -u scripts/visualize_rsa.py `
  --csv results\runs\rsa_matrix_v1\rsa_results_master.csv `
  --subject OVERALL `
  --output-dir results\runs\rsa_matrix_v1\figures `
  --prefix rsa_matrix_v1
```

**Subject-level statistics:**
```powershell
python -X utf8 -u scripts/analyze_rsa_stats.py `
  --csv results\runs\rsa_matrix_v1\rsa_results_master.csv `
  --baseline 50 `
  --output results\runs\rsa_matrix_v1\stats_summary.csv
```

**Optional overrides:** Use `--conditions` during training to restrict the cardinality codes (e.g., `--conditions 11 22 33`) or adjust the config to tweak seeds/baselines.

---

### 8. Subject Performance Deep Dive

**Goal:** Per-subject and per-fold performance breakdown, inner vs outer comparison.

```powershell
python -X utf8 -u scripts/analyze_nloso_subject_performance.py \
  "results\runs\<your_run_directory>"
```

**Outputs (in `<run_dir>/subject_performance/`):**
- Per-subject accuracy + support bar charts
- Per-fold accuracy bar charts
- Overall confusion matrix
- Per-subject confusion matrices (one per subject)
- Inner vs outer accuracy scatter plot (if `test_predictions_inner.csv` exists)
- HTML report

**Use case:** Identify which subjects/folds drive performance, detect outliers.

---

## End-to-End Research Pipeline

### Complete Workflow (From Data to Publication)

```powershell
# ============================================
# PHASE 1: DATA PREPARATION
# ============================================

# One-time conversion from HAPPE to .fif
python scripts/prepare_from_happe.py

# ============================================
# PHASE 2: HYPERPARAMETER OPTIMIZATION
# ============================================

# Stage 1: Architecture search (48 trials, ~12 hours)
python -X utf8 -u scripts/optuna_search.py \
  --stage step1 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/base.yaml \
  --cfg configs/tasks/cardinality_1_3/step1_search.yaml \
  --space configs/tasks/cardinality_1_3/step1_space_scaffold.yaml \
  --trials 48

# Refresh Optuna reports
python scripts/refresh_optuna_summaries.py

# MANUAL: Extract top trial params, create step1_winner.yaml

# Stage 2: Recipe search (48 trials, ~12 hours)
python -X utf8 -u scripts/optuna_search.py \
  --stage step2 \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/step1_winner.yaml \
  --cfg configs/tasks/cardinality_1_3/step2_search.yaml \
  --space configs/tasks/cardinality_1_3/step2_space_recipe.yaml \
  --trials 48

# Refresh Optuna reports again
python scripts/refresh_optuna_summaries.py

# MANUAL: Extract top trial params, create resolved_config.yaml

# ============================================
# PHASE 3: FINAL EVALUATION (Multi-Seed)
# ============================================

# Add to resolved_config.yaml:
# seeds: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

# Run 10-seed evaluation (~3-6 hours)
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml

# ============================================
# PHASE 4: PERMUTATION TESTING
# ============================================

# Pick one seed (e.g., seed_41) for permutation test
python -X utf8 -u train.py \
  --task cardinality_1_3 \
  --engine eeg \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml \
  --set seed=41 \
  --permute-labels \
  --n-permutations 200 \
  --permute-scope within_subject \
  --permute-stratified \
  --observed-run-dir "results\runs\<seed_41_run_dir>"

# ============================================
# PHASE 5: ANALYSIS & REPORTING
# ============================================

# XAI analysis (use seed_41 run)
python -X utf8 -u scripts/run_xai_analysis.py \
  --run-dir "results\runs\<seed_41_run_dir>"

# Post-hoc statistics
python -X utf8 -u scripts/run_posthoc_stats.py \
  --run-dir "results\runs\<seed_41_run_dir>" \
  --glmm --forest

# Subject performance analysis
python -X utf8 -u scripts/analyze_nloso_subject_performance.py \
  "results\runs\<seed_41_run_dir>"

# ============================================
# PHASE 6: PUBLICATION FIGURES
# ============================================

# Generate publication-ready figures
cd publication-ready-media/code/v4_neuroscience
python generate_all_v4_figures.py

# Figures saved to: publication-ready-media/outputs/v4/
```

**Total expected time:** ~50-100 hours (mostly Optuna search + permutation testing)

---

## Performance Optimization Tips

### Speed Up Optuna Search

**1. Enable dataset caching**

In `configs/common.yaml`:
```yaml
dataset_cache_memory: true
```

**Effect:** First trial builds dataset, subsequent trials reuse (10-100× faster per trial).

**2. Use fewer folds**

```yaml
n_folds: 3           # Instead of LOSO
inner_n_folds: 2     # Minimum
```

**Effect:** Faster per-trial evaluation (use for initial search, then validate with LOSO).

**3. Reduce epochs for search**

```yaml
epochs: 50           # Instead of 100
early_stop: 10       # Instead of 20
```

**Effect:** Faster convergence detection, may miss subtle differences.

**4. Use smaller batch size (if memory-constrained)**

```yaml
batch_size: 16       # Instead of 32
```

**Effect:** Slower per-epoch, but avoids OOM errors.

### Parallelize Permutations

If you have multiple GPUs, run permutations in parallel:

```powershell
# Terminal 1 (GPU 0): permutations 1-100
$env:CUDA_VISIBLE_DEVICES=0
python train.py ... --n-permutations 100 --permute-seed 123

# Terminal 2 (GPU 1): permutations 101-200
$env:CUDA_VISIBLE_DEVICES=1
python train.py ... --n-permutations 100 --permute-seed 456
```

Then manually merge CSV results.

---

## Common Mistakes to Avoid

### 1. Forgetting to Refresh Optuna Summaries

**Problem:** Run Optuna search, but can't find Top-3 report.

**Solution:** Always run after search:
```powershell
python scripts/refresh_optuna_summaries.py
```

### 2. Using Step N Winner Without Running Step N-1

**Problem:** `step2_winner.yaml` doesn't exist because you skipped Step 1.

**Solution:** Follow stage progression: Step 1 → extract winner → Step 2 → extract winner → final eval.

### 3. Not Enabling Dataset Caching for Optuna

**Problem:** Each trial reloads EEG data from disk (extremely slow).

**Solution:** Set `dataset_cache_memory: true` in `common.yaml`.

### 4. Running Permutation Test Without `--observed-run-dir`

**Problem:** Permutation test uses different splits than observed run (invalidates comparison).

**Solution:** Always pass `--observed-run-dir` to reuse exact splits.

### 5. Forgetting `seed` Parameter

**Problem:** Run fails with `KeyError: 'seed'`.

**Solution:** Always specify `seed` in your config (constitutional requirement).

### 6. Mixing Inner and Outer Predictions

**Problem:** Using `test_predictions_inner.csv` for publication metrics.

**Solution:** Always use `test_predictions_outer.csv` for final reporting (inner is for diagnostics only).

---

## Task-Specific Workflows

### Landing Digit Tasks (Binary/Ternary)

```powershell
# Binary: 2 vs 3
python train.py --task landing_on_2_3 ...

# Ternary: 1 vs 2 vs 3
python train.py --task landing_digit_1_3_within_small ...

# With no-change trials (includes cardinality)
python train.py --task landing_digit_1_3_within_small_and_cardinality ...
```

**Key difference:** Tasks with `_and_cardinality` include no-change condition (uses ALL data, not just ACC=1).

### Cardinality Tasks (Range Classification)

```powershell
# Small range (PI system)
python train.py --task cardinality_1_3 ...

# Large range (ANS system)
python train.py --task cardinality_4_6 ...

# Full range (six-way)
python train.py --task cardinality_1_6 ...
```

**Use case:** Test PI/ANS boundary hypothesis by comparing performance across ranges.

---

## Next Steps

- For detailed parameter explanations, see [Configuration Reference](CONFIGURATION.md)
- For command syntax, see [CLI Reference](CLI_REFERENCE.md)
- For XAI details, see [XAI Guide](XAI_GUIDE.md)
- For Optuna details, see [Optuna Guide](OPTUNA_GUIDE.md)
