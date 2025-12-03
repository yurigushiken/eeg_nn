# Configuration Reference

Complete guide to all configuration parameters in the EEG neural decoding pipeline.

## Configuration File Structure

The pipeline uses layered YAML configuration:

```
configs/
├── common.yaml              # Global defaults (all tasks)
├── xai_defaults.yaml        # XAI analysis parameters
├── posthoc_defaults.yaml    # Post-hoc statistics defaults
└── tasks/<task>/
    ├── base.yaml            # Task-specific base config
    ├── step1_search.yaml    # Optuna Stage 1 controller
    ├── step1_space_*.yaml   # Stage 1 search space
    ├── step2_search.yaml    # Optuna Stage 2 controller
    ├── step2_space_*.yaml   # Stage 2 search space
    ├── step3_search.yaml    # Optuna Stage 3 controller (optional)
    └── step3_space_*.yaml   # Stage 3 search space
```

**Loading order:** `common.yaml` → `task/base.yaml` → command-line overrides (`--set key=value`)

## Required Parameters (Constitutional)

These parameters MUST be explicitly specified. No default fallback is provided to ensure conscious research decisions.

### `seed` (int, REQUIRED)
Random seed for reproducibility.

```yaml
seed: 42
```

**Why required:** Prevents untracked randomness. Every run must be reproducible.

### `optuna_objective` (str, REQUIRED for Optuna)
Metric to optimize during hyperparameter search.

**Options:**
- `composite_min_f1_plur_corr` — **RECOMMENDED** (see Optimization Objectives below)
- `inner_mean_min_per_class_f1` — worst-class focus
- `inner_mean_plur_corr` — plurality correctness (⚠️ use composite instead)
- `inner_mean_macro_f1` — average F1
- `inner_mean_acc` — raw accuracy

```yaml
optuna_objective: composite_min_f1_plur_corr
```

### `composite_min_f1_weight` (float, REQUIRED if using composite objective)
Weight for min-per-class F1 in composite objective (0.0–1.0).

```yaml
composite_min_f1_weight: 0.65  # Prioritize decodability over distinctness
```

**Interpretation:**
- `0.65` → 65% weight on min-F1, 35% on plurality correctness
- `0.50` → Equal balance
- `0.35` → More weight on plurality correctness

### `outer_eval_mode` (str, REQUIRED)
How to generate outer test predictions.

**Options:**
- `ensemble` — **RECOMMENDED**: Average softmax across K inner models (more stable)
- `refit` — Refit single model on full outer-train (use with `refit_val_frac>0` for early stop)

```yaml
outer_eval_mode: ensemble
```

**Why required:** Forces explicit choice between stability (ensemble) vs. simplicity (refit).

## Cross-Validation Parameters

### `n_folds` (int or null)
Number of outer cross-validation folds. If `null`, uses Leave-One-Subject-Out (LOSO).

```yaml
n_folds: 5        # 5-fold GroupKFold
n_folds: null     # LOSO (one fold per subject)
```

**When to use LOSO:** Small N (<30 subjects), want maximum train data per fold.
**When to use K-fold:** Larger N, want faster evaluation.

### `inner_n_folds` (int, default: 3)
Number of inner cross-validation folds for hyperparameter selection.

```yaml
inner_n_folds: 3
```

**Requirement:** Must have ≥`inner_n_folds` unique subjects in each outer train set. Run fails otherwise.

## Training Parameters

### Core Training

```yaml
epochs: 100                    # Maximum epochs per inner fold
early_stop: 20                 # Patience (epochs without improvement)
batch_size: 32                 # Batch size (reduce if OOM)
lr: 0.001                      # Learning rate
weight_decay: 0.0001           # L2 regularization
```

### Learning Rate Warm-Up

```yaml
lr_warmup_frac: 0.1            # Warm-up for first 10% of epochs
lr_warmup_init: 0.1            # Start at 10% of target LR
```

**Effect:** Linear ramp from `lr_warmup_init * lr` to `lr` over first `lr_warmup_frac * epochs`.

**When to use:** Large batch sizes, unstable early training.

### Augmentation Warm-Up

```yaml
aug_warmup_frac: 0.2           # Ramp augmentations for first 20% of epochs
```

**Effect:** Augmentation probabilities/magnitudes ramp from 0 to full over first `aug_warmup_frac * epochs`.

**When to use:** Strong augmentations that might disrupt early learning.

## Data Parameters

### Temporal Cropping

```yaml
crop_ms: [0, 400]              # Use 0-400ms post-stimulus
```

**Effect:** Crops EEG epochs to specified time window (in milliseconds relative to stimulus onset).

### Channel Selection

```yaml
use_channel_list: non_scalp    # Exclude 28 non-scalp channels
# OR
include_channels: [Cz, Pz, Oz, ...]  # Explicit channel list
```

**Options:**
- `non_scalp` — Default: excludes non-scalp channels (leaves ~100 channels)
- `null` — Use all channels
- List of channel names — Explicit subset

### Spatial Sampling (Cz-Ring)

```yaml
cz_step: 0                     # Disabled (default)
cz_step: 3                     # Keep 60% closest to Cz (~60 channels)
```

**Mapping:**
- `0` or `null` → Disabled (keep all channels after `use_channel_list`)
- `1` → 20% (tight central ring, ~20 channels)
- `2` → 40% (~40 channels)
- `3` → 60% (~60 channels)
- `4` → 80% (~80 channels)
- `5` → 100% (all channels, same as disabled)

**Requires:** Montage attachment during data prep.

### Dataset Caching

```yaml
dataset_cache_memory: true     # Cache dataset in RAM (huge speedup for Optuna)
```

**Effect:** First trial builds dataset, subsequent trials reuse instantly. **Highly recommended for hyperparameter search.**

**Caveat:** Cache clears when Python process exits.

## Model Architecture (EEGNeX)

```yaml
# Temporal convolution (Stage 1)
F1: 8                          # Number of temporal filters
D: 2                           # Depthwise multiplier
kernel_1: 32                   # Temporal kernel size (samples)

# Depthwise convolution (Stage 2)
kernel_2: 8                    # Depthwise kernel size

# Pooling
pool_mode: mean                # "mean" or "max"

# Dropout
dropout: 0.5                   # Dropout rate (0-1)
```

**Typical search ranges (Optuna):**
- `F1`: [4, 8, 16, 32]
- `D`: [1, 2]
- `kernel_1`: [16, 32, 64]
- `kernel_2`: [4, 8, 16]
- `dropout`: [0.25, 0.5, 0.75]

## Augmentation Parameters

### Noise Injection

```yaml
noise_p: 0.5                   # Probability of applying noise
noise_std: 0.02                # Noise standard deviation (relative to signal)
```

### Temporal Shift

```yaml
shift_p: 0.5                   # Probability of applying shift
shift_max_frac: 0.1            # Max shift as fraction of epoch length
```

### Temporal Masking

```yaml
time_mask_p: 0.5               # Probability of masking
time_mask_frac: 0.1            # Fraction of epoch to mask
```

**Effect:** Randomly zeroes out a contiguous time segment.

### Mixup

```yaml
mixup_alpha: 0.2               # Mixup interpolation parameter (0=disabled)
```

**Effect:** Interpolates between two random samples: `x_mix = λx₁ + (1-λ)x₂`, where `λ ~ Beta(α, α)`.

## Optimization Objectives Explained

The `optuna_objective` controls what the hyperparameter search optimizes for.

### `composite_min_f1_plur_corr` ⭐ RECOMMENDED

**What it measures:** Weighted sum of min-per-class F1 and plurality correctness.

**Formula:**
```
composite = (composite_min_f1_weight × min_F1) + ((1 - composite_min_f1_weight) × plur_corr)
```

**Why it's best for cognitive neuroscience:**
- **Prevents gaming:** A model can achieve 100% plurality correctness with terrible accuracy (e.g., 10%) by making one class hyper-confident. The composite prevents this by requiring BOTH metrics.
- **Ensures decodability:** All classes must be decodable (min-F1 > 0).
- **Ensures distinctness:** Each class must have the correct prediction as plurality.

**Example (weight=0.65):**
```
Trial A: min_F1=40%, plur_corr=100% → composite = 0.65×40 + 0.35×100 = 61.0
Trial B: min_F1=50%, plur_corr=67%  → composite = 0.65×50 + 0.35×67  = 56.0
Trial C: min_F1=45%, plur_corr=100% → composite = 0.65×45 + 0.35×100 = 64.3 ← BEST
```

**Configuration:**
```yaml
optuna_objective: composite_min_f1_plur_corr
composite_min_f1_weight: 0.65
```

### `inner_mean_min_per_class_f1` (Worst-Class Focus)

**What it measures:** Minimum F1 across all classes (averaged across inner folds).

**When to use:** Ensure model performs on ALL classes, including hardest.

**Example:** If class "2" is hard, this forces model to learn it (not just easy classes).

```yaml
optuna_objective: inner_mean_min_per_class_f1
```

### `inner_mean_plur_corr` (Plurality Correctness)

**What it measures:** Proportion of classes where correct prediction is most frequent.

**⚠️ WARNING:** Using this alone is scientifically unsound. A model can achieve 100% plurality correctness with 10% accuracy. **Use `composite_min_f1_plur_corr` instead.**

**Formula:** For each true class (row in confusion matrix), check if diagonal is row-max. Return proportion where true.

**Example confusion matrix:**
```
        Pred:  1   2   3
True 1: [100  30  20]  ← diagonal wins ✓
True 2: [ 40  60  50]  ← diagonal wins ✓
True 3: [ 10  80  30]  ← off-diagonal wins ✗

Plurality correctness = 2/3 = 0.667 (67%)
```

### `inner_mean_macro_f1` (Average F1)

**What it measures:** Average F1 across all classes (averaged across inner folds).

**When to use:** Want good overall performance, can tolerate some class imbalance.

```yaml
optuna_objective: inner_mean_macro_f1
```

### `inner_mean_acc` (Raw Accuracy)

**What it measures:** Proportion of correct predictions.

**Caveat:** Misleading with class imbalance (model may ignore minority classes).

```yaml
optuna_objective: inner_mean_acc
```

## Decision Layer (Optional Ordinal Refinement)

Post-hoc refinement for adjacent-class confusions (e.g., 2↔3).

```yaml
decision_layer:
  enable: false                # Enable decision layer
  metric: optuna_objective     # Metric to optimize during θ tuning
  theta_grid: [0.30, 0.70, 0.01]  # [start, stop, step]
  min_activation_trials: 50    # Min trials required; else fallback θ=0.50
  save_activation_stats: true  # Persist activation rates
```

**How it works:**
1. For each outer fold, tunes threshold θ on inner validation data
2. Rule: If P(higher) / P(lower) > θ, predict higher; else lower
3. Applies frozen θ to outer test data

**Outputs:**
- `test_predictions_outer_thresholded.csv`
- `decision_layer/thresholds.json` (per-fold θ values, metrics)

**Requirement:** Class names must be numeric (e.g., "1", "2", "3") for ordinal adjacency.

## Permutation Testing (Optional)

Generate empirical null distribution by shuffling labels.

```yaml
permutation:
  enable: false                # Enable permutation testing
  n_permutations: 200          # Number of permutation runs
  permute_scope: within_subject  # "within_subject" or "global"
  permute_stratified: true     # Preserve within-subject class balance
  permute_seed: 123            # Seed for reproducibility
```

**CLI Usage:**
```powershell
python train.py \
  --task cardinality_1_3 \
  --base configs/tasks/cardinality_1_3/resolved_config.yaml \
  --permute-labels \
  --n-permutations 200 \
  --permute-scope within_subject \
  --observed-run-dir "results\runs\<observed_run_dir>"
```

**Outputs:**
- `<observed_run>_perm_test_results.csv` (per-permutation metrics)
- `<observed_run>_perm_summary.json` (empirical p-values)

## Multi-Seed Evaluation

```yaml
seeds: [41, 42, 43, 44, 45]   # Run multiple seeds
```

**Effect:** Runs training once per seed, generates aggregate JSON.

**Output:** `results/runs/<timestamp>_seeds_aggregate.json`

## Refit Mode Parameters

When `outer_eval_mode: refit`:

```yaml
refit_val_frac: 0.2            # Hold out 20% of outer-train for validation
refit_val_frac: 0.0            # No validation (train on full outer-train)
```

**When to use validation:** Want early stopping during refit.
**When to skip:** Maximize training data, rely on `epochs` limit.

## XAI Configuration

See `configs/xai_defaults.yaml`:

```yaml
xai_top_k_channels: 10         # Top channels to highlight
peak_window_ms: 100            # Duration for peak window analysis
tf_morlet_freqs: [4, 8, 13, 30]  # Frequencies for time-frequency
```

## Post-Hoc Statistics Configuration

See `configs/posthoc_defaults.yaml`:

```yaml
posthoc:
  alpha: 0.05                  # Significance threshold
  multitest: fdr               # "fdr" or "none"
  glmm: true                   # Run GLMM analysis (requires R)
  forest: true                 # Generate forest plots
  chance_rate: null            # Auto-detect if null
```

## Common Configuration Patterns

### Minimal LOSO Run

```yaml
seed: 42
n_folds: null                  # LOSO
outer_eval_mode: ensemble
optuna_objective: composite_min_f1_plur_corr
composite_min_f1_weight: 0.65
epochs: 100
early_stop: 20
batch_size: 32
lr: 0.001
```

### Fast Optuna Search

```yaml
seed: 42
n_folds: 3                     # Faster than LOSO
inner_n_folds: 2               # Minimum
outer_eval_mode: ensemble
optuna_objective: composite_min_f1_plur_corr
composite_min_f1_weight: 0.65
epochs: 50                     # Reduce for speed
early_stop: 10
dataset_cache_memory: true     # CRITICAL for speed
```

### Conservative (High Rigor)

```yaml
seed: 42
n_folds: null                  # LOSO (max generalization test)
inner_n_folds: 3
outer_eval_mode: ensemble      # More stable than refit
optuna_objective: composite_min_f1_plur_corr
composite_min_f1_weight: 0.65
epochs: 150
early_stop: 30
lr_warmup_frac: 0.1            # Stability
aug_warmup_frac: 0.2
```

## Command-Line Overrides

Override any config value via `--set`:

```powershell
python train.py --task ... --base ... \
  --set seed=99 \
  --set batch_size=16 \
  --set lr=0.0005 \
  --set n_folds=5
```

**Nested keys:**
```powershell
--set decision_layer.enable=true \
--set permutation.n_permutations=100
```

## Resolved Config Output

Every run writes `resolved_config.yaml` to the run directory. This file contains:
- All parameters used (merged from common + task + CLI)
- Can be reused as `--base` for exact reproduction

**Example:**
```powershell
# Run once
python train.py --task cardinality_1_3 --base configs/tasks/cardinality_1_3/base.yaml

# Reproduce exactly
python train.py \
  --task cardinality_1_3 \
  --base results/runs/<run_dir>/resolved_config.yaml
```

## Troubleshooting Config Issues

**Problem:** `KeyError: 'seed'`
**Solution:** Add `seed: 42` to your config (required parameter).

**Problem:** `KeyError: 'optuna_objective'` during Optuna search
**Solution:** Add `optuna_objective: composite_min_f1_plur_corr` to search config.

**Problem:** `KeyError: 'composite_min_f1_weight'`
**Solution:** Add `composite_min_f1_weight: 0.65` when using composite objective.

**Problem:** `AssertionError: insufficient subjects for inner_n_folds=5`
**Solution:** Reduce `inner_n_folds` to 2 or 3.

**Problem:** Config changes ignored
**Solution:** Check loading order. CLI `--set` overrides YAML. Check `resolved_config.yaml` to see final values.

For more help, see [Troubleshooting Guide](TROUBLESHOOTING.md).
