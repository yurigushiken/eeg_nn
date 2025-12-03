# Technical Details

Implementation specifics, design decisions, and technical architecture of the EEG neural decoding pipeline.

## Nested Cross-Validation

### Outer Loop (Generalization Testing)

**Purpose:** Unbiased estimate of generalization performance.

**Strategy:**
- **LOSO (Leave-One-Subject-Out):** If `n_folds: null`, one fold per subject
- **GroupKFold:** If `n_folds: K`, stratified by subject (no subject in both train and test)

**Process:**
1. For each outer fold:
   - Hold out test subjects (never seen during training/validation)
   - Use remaining subjects for inner loop
   - Evaluate final model on held-out subjects
2. Aggregate metrics across all outer folds

**Safeguards:**
- Assertions verify no subject overlap between train/test
- Split indices exported to `splits_indices.json` for auditability

### Inner Loop (Hyperparameter Selection)

**Purpose:** Select best hyperparameters without touching outer test data.

**Strategy:**
- K-fold cross-validation on outer-train subjects (subject-aware splits)
- Typically `inner_n_folds: 3`

**Process:**
1. For each inner fold:
   - Train model on inner-train subjects
   - Validate on inner-val subjects
   - Record validation metrics (per-epoch for pruning)
2. Select best epoch based on validation performance
3. Use selected configuration for outer test evaluation

**Checkpoint strategy:**
- Save best checkpoint per inner fold: `fold_XX_inner_YY_best.ckpt`
- "Best" determined by configured `optuna_objective`

## Outer Test Evaluation Modes

### Ensemble Mode (Recommended)

**Method:** Average softmax outputs across all K inner models.

**Formula:**
```
P(class|x) = (1/K) Σᵢ softmax(model_i(x))
```

**Advantages:**
- More stable predictions (reduces variance)
- Utilizes all trained models (no wasted computation)
- Better calibrated probabilities

**Disadvantages:**
- Requires storing K models per outer fold
- Slower inference (K forward passes)

**Configuration:**
```yaml
outer_eval_mode: ensemble
```

### Refit Mode

**Method:** Refit single model on full outer-train set, then test.

**Process:**
1. After inner loop completes, take best hyperparameters
2. Train new model on ALL outer-train subjects
3. Optional: Hold out `refit_val_frac` for early stopping
4. Use refitted model for outer test evaluation

**Advantages:**
- Single model per fold (simpler)
- Uses maximum training data (no inner validation hold-out)

**Disadvantages:**
- Less stable (single model vs ensemble)
- May overfit if `refit_val_frac=0` (no early stopping)

**Configuration:**
```yaml
outer_eval_mode: refit
refit_val_frac: 0.2  # Hold out 20% for validation (optional)
```

## Per-Epoch Pruning & Checkpointing

### Optuna Pruning

**Purpose:** Stop underperforming trials early (save compute).

**Method:** MedianPruner compares trial's current performance to median of all trials at same epoch.

**Decision rule:**
```
if trial_metric < median_metric and epoch > n_warmup_steps:
    prune trial
```

**Pruning signal:** Uses configured `optuna_objective` (e.g., `composite_min_f1_plur_corr`).

**Effect:** Poor architectures pruned after ~10-20 epochs, good architectures train to completion.

**Constitutional alignment:** As of Oct 5, 2025, pruning uses same metric as final selection (prevents metric drift).

### Checkpoint Selection

**Strategy:** Save checkpoint with best validation performance (per inner fold).

**Metric:** Uses configured `optuna_objective` (consistent with pruning).

**Tie-breaking:** If multiple epochs have same objective value, choose one with lowest validation loss.

**Persistence:**
- Best checkpoint saved: `fold_XX_inner_YY_best.ckpt`
- Contains model weights, hyperparameters, training metrics

**Reloading:** At outer test time, load best checkpoint for each inner fold (if ensemble mode).

## Objective-Aligned Metrics

**Historical issue:** Before Oct 2025, pruning and checkpoint selection always used `val_macro_f1` regardless of `optuna_objective`.

**Current behavior:** All metric computations use `optuna_objective`:
- Per-epoch pruning signal
- Best checkpoint selection
- Optuna trial objective value

**Rationale:** Ensures scientific consistency. If optimizing for `composite_min_f1_plur_corr`, all decisions should use that metric.

**Implementation:** `code/training/metrics.py` computes objective based on config.

## Determinism Implementation

### Seeding Strategy

**Required parameter:** `seed` must be explicitly set in config (no fallback).

**Seeding points:**
1. **Python:** `random.seed(seed)`
2. **NumPy:** `np.random.seed(seed)`
3. **PyTorch CPU:** `torch.manual_seed(seed)`
4. **PyTorch CUDA:** `torch.cuda.manual_seed_all(seed)`
5. **DataLoader workers:** Per-worker seed = `seed + worker_id`

### CUDA Determinism

**Settings applied:**
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

**Environment variable:**
```
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

**Effect:** Ensures CUDA operations (GEMM, convolutions) use deterministic algorithms (may be slower).

### Determinism Banner

**Printed at runtime:**
```
=== DETERMINISM SETTINGS ===
Python seed: 42
NumPy seed: 42
PyTorch seed: 42
CUDA deterministic: True
CUBLAS workspace: :4096:8
DataLoader workers: seeded per-worker
===========================
```

**Persisted:** In `summary_*.json` and HTML reports.

**Verification:** Re-run with same seed should produce identical results (bit-for-bit).

## Dataset Caching

**Purpose:** Avoid repeated MNE file loads and preprocessing during hyperparameter search.

**Method:** In-process RAM cache (Python dict).

**Key:** Fingerprint of (task, crop_ms, channels, subjects)

**Lifecycle:**
1. First trial in Python process builds dataset (loads .fif, crops, converts to tensors)
2. Cache stores (X, y, groups, channels, times) in RAM
3. Subsequent trials with same fingerprint reuse cache (instant)
4. Cache cleared when Python process exits

**Configuration:**
```yaml
dataset_cache_memory: true  # Enable (highly recommended for Optuna)
```

**Memory usage:** ~1-5 GB per cached dataset (depends on number of trials/channels).

**Trade-off:** Memory for speed. 100× faster trial startup when enabled.

## Modular Training Architecture

**Legacy:** `training_runner.py` was monolithic (~2000 lines).

**Refactor (Oct 2025):** Thin coordinator delegates to modular components.

### Components

**`code/training/setup_orchestrator.py`:**
- JSONL logging setup
- Outer CV split generation (GroupKFold/LOSO)
- Channel topomap plotting

**`code/training/inner_loop.py`:**
- Per-inner-fold training loop
- LR warmup, augmentation warmup
- Mixup integration
- Pruning signal computation
- Per-epoch checkpointing

**`code/training/outer_loop.py`:**
- Orchestrates inner K-fold
- Selects best inner model
- Runs outer test evaluation
- Generates per-fold plots

**`code/training/evaluation.py`:**
- Ensemble mode: Load K models, average softmax
- Refit mode: Refit model on outer-train, then evaluate
- Metric computation (accuracy, F1, confusion matrix)

**`code/training/checkpointing.py`:**
- Objective-aligned early stopping
- Best checkpoint selection (with tie-breaking)
- Checkpoint saving/loading

**`code/training/metrics.py`:**
- Compute configured objective (`composite_min_f1_plur_corr`, etc.)
- Plurality correctness calculation
- Min-per-class F1 computation

**`code/artifacts/`:**
- CSV writers (`learning_curves_inner.csv`, `outer_eval_metrics.csv`, predictions)
- Plot builders (confusion matrices, learning curves)
- Overall plot orchestrator (cross-fold aggregation)

**Benefits:**
- Testability (each component can be unit tested)
- Clarity (each file has single responsibility)
- Maintainability (changes isolated to relevant module)

**Backward compatibility:** Legacy behavior preserved (no breaking changes).

## Provenance Tracking

### Run Metadata

Every run exports `summary_<TASK>_<ENGINE>.json` with:
- Model class (e.g., `braindecode.models.EEGNeX`)
- Library versions (torch, numpy, sklearn, mne, braindecode, captum, optuna, python)
- Hardware (GPU names, CUDA version, CPU, OS)
- Determinism flags (seeds, cudnn settings)
- Hyperparameters (all config values)
- Metrics (outer fold means + std)
- Config hash (for detecting config drift)

**Purpose:** Full reproducibility audit trail.

### Resolved Config

Every run writes `resolved_config.yaml` with:
- Merged config (common + task + CLI overrides)
- All parameters used (no defaults omitted)

**Purpose:** Exact reproduction of run.

**Usage:**
```powershell
# Reproduce run exactly
python train.py --task ... --base results/runs/<run_dir>/resolved_config.yaml
```

### Split Indices

Every run exports `splits_indices.json` with:
- Outer fold indices (trial indices and subject IDs per fold)
- Inner fold indices (per outer fold)

**Purpose:** Audit splits for subject leakage, reproduce exact train/test splits.

**Format:**
```json
{
  "outer_folds": [
    {
      "fold": 0,
      "train_subjects": [1, 2, 3, ...],
      "test_subjects": [24],
      "train_indices": [0, 1, 2, ...],
      "test_indices": [3210, 3211, ...]
    },
    ...
  ],
  "inner_folds": { ... }
}
```

## Class Weighting

**Purpose:** Handle class imbalance (if present).

**Method:** Inverse frequency weighting.

**Formula:**
```
weight_i = n_total / (n_classes × n_i)
```

**Application:** PyTorch `CrossEntropyLoss(weight=class_weights)`

**Persistence:** Weights saved per fold: `fold_XX_inner_YY_class_weights.json`

**Rationale:** Ensures model doesn't ignore minority classes.

**When disabled:** If classes balanced (approximately equal counts), weights ~1.0 (no effect).

## Data Flow

### Training Pipeline

```
.fif files (per subject)
    ↓
Dataset loader (code/datasets.py)
    ↓ (caching layer)
Tensors (X, y, groups) in RAM
    ↓
DataLoader (PyTorch, per-worker seeding)
    ↓
Augmentations (noise, shift, masking, mixup)
    ↓
Model forward pass (EEGNeX)
    ↓
Loss computation (CrossEntropyLoss + class weights)
    ↓
Backprop + optimizer step
    ↓
Checkpoint saving (per epoch)
    ↓ (select best)
Outer test evaluation (ensemble or refit)
    ↓
Metrics + plots + predictions CSV
```

### Hyperparameter Search Pipeline

```
Optuna study (SQLite DB)
    ↓
TPE sampler suggests trial
    ↓
Configure model + training from search space
    ↓
Run training (nested CV)
    ↓ (per epoch)
Compute pruning signal
    ↓ (if underperforming)
Prune trial (stop early) OR continue
    ↓ (after completion)
Record final objective value in Optuna
    ↓ (next trial)
TPE updates posterior, suggests new trial
```

## Performance Bottlenecks

**Without caching:**
- MNE file loading: ~2-5 seconds per subject
- Time cropping + channel selection: ~1 second per subject
- **Total per trial startup:** ~50-100 seconds (24 subjects)

**With caching:**
- First trial: ~50-100 seconds (builds cache)
- Subsequent trials: <1 second (reuses cache)

**Training:**
- Forward pass: ~10-50ms per batch (GPU-dependent)
- Backward pass: ~20-100ms per batch
- **Total per epoch:** ~10-60 seconds (dataset size dependent)

**Outer test evaluation:**
- Ensemble mode: K× forward pass time (K = inner_n_folds)
- Refit mode: 1× forward pass time

## Next Steps

- For configuration options, see [Configuration Reference](CONFIGURATION.md)
- For understanding the training workflow, see [Workflows](WORKFLOWS.md)
- For troubleshooting implementation issues, see [Troubleshooting](TROUBLESHOOTING.md)
