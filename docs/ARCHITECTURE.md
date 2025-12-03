# Architecture Guide

Repository organization and module responsibilities for the EEG neural decoding pipeline.

## Repository Structure

```
eeg_nn/
├── code/                       # Core implementation
│   ├── training/              # Modular training orchestration
│   ├── artifacts/             # CSV writers, plot builders
│   ├── preprocessing/         # MNE pipeline, spatial sampling
│   ├── datasets.py            # EEG data loaders
│   ├── model_builders.py      # EEGNeX architecture
│   ├── training_runner.py     # Thin coordinator
│   └── engines/               # Engine wrappers
├── configs/                    # YAML configuration files
│   ├── common.yaml            # Global defaults
│   ├── xai_defaults.yaml      # XAI parameters
│   ├── posthoc_defaults.yaml  # Statistics defaults
│   └── tasks/                 # Per-task configs
├── scripts/                    # Command-line entry points
│   ├── train.py               # Main training script
│   ├── optuna_search.py       # Hyperparameter search
│   ├── run_xai_analysis.py    # XAI generation
│   ├── run_posthoc_stats.py   # Post-hoc statistics
│   ├── prepare_from_happe.py  # Data preparation
│   └── ...
├── optuna_tools/              # Optuna report generation
│   ├── config.py              # Path/settings configuration
│   ├── discovery.py           # Study discovery
│   ├── csv_io.py              # CSV export
│   ├── plotting.py            # Parallel plots
│   ├── reports.py             # Top-3 report generation
│   └── ...
├── docs/                       # Documentation
├── results/                    # Training outputs
│   ├── runs/                  # Individual training runs
│   └── optuna/                # Hyperparameter search databases
├── publication-ready-media/    # Publication figures
├── data_preprocessed/         # Materialized .fif epochs (gitignored)
├── net/                        # Electrode montage files
└── environment.yml            # Conda environment specification
```

## Core Modules

### `code/training/` - Modular Training Components

#### `setup_orchestrator.py`
**Responsibilities:**
- JSONL runtime logging setup
- Outer CV split generation (GroupKFold or LOSO)
- Subject-aware split validation
- Channel topomap plotting (for documentation)

**Key functions:**
- `setup_logging(run_dir)` → Initialize JSONL logger
- `setup_outer_splits(X, y, groups, config)` → Generate outer folds
- `plot_channel_topomap(channels, montage, output_path)` → Visualize electrode layout

#### `inner_loop.py`
**Responsibilities:**
- Per-inner-fold training loop
- LR warmup scheduling
- Augmentation warmup scheduling
- Mixup application
- Pruning signal computation (for Optuna)
- Per-epoch checkpointing

**Key functions:**
- `train_inner_fold(model, train_loader, val_loader, config, pruner)` → Train one inner fold
- `apply_lr_warmup(optimizer, epoch, config)` → Adjust learning rate
- `apply_aug_warmup(augmentations, epoch, config)` → Ramp augmentation strength

#### `outer_loop.py`
**Responsibilities:**
- Orchestrate inner K-fold cross-validation
- Select best inner model (based on validation metric)
- Run outer test evaluation (ensemble or refit)
- Generate per-fold plots (confusion matrix, learning curves)

**Key functions:**
- `run_outer_fold(fold_idx, train_data, test_data, config)` → Complete outer fold
- `select_best_inner_model(inner_results, config)` → Choose best checkpoint

#### `evaluation.py`
**Responsibilities:**
- Ensemble mode: Load K models, average softmax outputs
- Refit mode: Train single model on full outer-train
- Metric computation (accuracy, F1, confusion matrix, Cohen's kappa)

**Key functions:**
- `evaluate_ensemble(models, test_loader, device)` → Ensemble predictions
- `evaluate_refit(model, test_loader, device)` → Refit mode predictions
- `compute_metrics(y_true, y_pred, y_prob)` → Full metric suite

#### `checkpointing.py`
**Responsibilities:**
- Objective-aligned early stopping
- Best checkpoint selection (with tie-breaking on validation loss)
- Checkpoint saving/loading

**Key functions:**
- `save_checkpoint(model, optimizer, epoch, metrics, path)` → Persist model state
- `load_checkpoint(path, model, optimizer)` → Restore model state
- `select_best_epoch(validation_history, config)` → Find best epoch by objective

#### `metrics.py`
**Responsibilities:**
- Compute configured objective (e.g., `composite_min_f1_plur_corr`)
- Plurality correctness calculation
- Min-per-class F1 computation

**Key functions:**
- `compute_objective(y_true, y_pred, config)` → Objective value
- `compute_plurality_correctness(confusion_matrix)` → Plurality metric
- `compute_min_per_class_f1(y_true, y_pred)` → Worst-class F1

### `code/artifacts/` - Artifact Generation

#### `csv_writers.py`
**Responsibilities:**
- Write `learning_curves_inner.csv`
- Write `outer_eval_metrics.csv` (with OVERALL row)
- Write `test_predictions_outer.csv` and `test_predictions_inner.csv`

**Key functions:**
- `write_learning_curves(history, output_path)`
- `write_outer_metrics(fold_results, output_path)`
- `write_predictions(y_true, y_pred, y_prob, metadata, output_path)`

#### `plot_builders.py`
**Responsibilities:**
- Objective-aware plot titles (e.g., "Composite Objective = 85.3")
- Per-class info strings (F1 scores per class)
- Confusion matrix plotting (standard and enhanced)

**Key functions:**
- `build_confusion_matrix_plot(cm, class_names, title, output_path)`
- `build_learning_curve_plot(history, output_path)`
- `get_objective_title(metrics, config)` → Format title string

#### `overall_plot_orchestrator.py`
**Responsibilities:**
- Aggregate confusion matrices across folds
- Generate overall (cross-fold) plots
- Enhanced plots with inner vs outer comparison

**Key functions:**
- `generate_overall_confusion(fold_cms, output_path)` → Aggregate confusion matrix
- `generate_enhanced_plots(fold_results, output_path)` → Side-by-side comparison

#### `artifact_writer.py`
**Responsibilities:**
- Orchestrate all artifact writing
- Export splits JSON
- Export predictions CSVs
- Coordinate plot generation

**Key functions:**
- `write_all_artifacts(results, config, output_dir)` → Complete artifact suite

### `code/preprocessing/` - Data Processing

#### `mne_pipeline.py`
**Responsibilities:**
- Spatial sampling (Cz-ring channel selection)
- Time cropping
- Channel alignment helpers

**Key functions:**
- `apply_cz_ring(epochs, cz_step, montage)` → Select channels by distance from Cz
- `crop_epochs(epochs, crop_ms)` → Time window selection
- `align_channels(epochs_list)` → Ensure consistent channel order

### `code/datasets.py` - Data Loading

**Responsibilities:**
- Load materialized `.fif` epochs
- Channel intersection and ordering
- Task-specific label extraction
- Dataset caching layer

**Key classes:**
- `MaterializedEEGDataset(Dataset)` → PyTorch dataset for .fif files
- `DatasetCache` → In-memory caching (optional)

### `code/model_builders.py` - Model Architecture

**Responsibilities:**
- EEGNeX model construction
- Train-time augmentation wrappers

**Key functions:**
- `build_eegnex(n_channels, n_times, n_classes, config)` → Instantiate EEGNeX
- `wrap_with_augmentations(model, config)` → Add noise/shift/masking

### `code/training_runner.py` - Thin Coordinator

**Responsibilities:**
- Load config (layered merging)
- Initialize logging
- Delegate to modular components
- Write final summary

**Process flow:**
1. Parse command-line arguments
2. Load and merge configs (common + task + CLI overrides)
3. Setup logging and determinism
4. Call `setup_orchestrator` for splits
5. For each outer fold, call `outer_loop`
6. Aggregate results, write summary
7. Optional: Call XAI analysis

## Scripts

### `scripts/train.py`
**Purpose:** Main entry point for single/multi-seed training runs.

**Flow:**
1. Parse arguments (`--task`, `--engine`, `--base`, `--set`, `--run-xai`)
2. Load config
3. Single seed: Call `training_runner.run_training(config)`
4. Multi-seed: Loop over `seeds` list, call training per seed
5. If `--run-xai`: Call `run_xai_analysis.py` for each run

### `scripts/optuna_search.py`
**Purpose:** Hyperparameter search driver (Stage 1/2/3).

**Flow:**
1. Parse arguments (`--stage`, `--task`, `--base`, `--cfg`, `--space`, `--trials`)
2. Load base config, search controller, search space
3. Create Optuna study (TPE sampler + MedianPruner)
4. For each trial:
   - Sample hyperparameters from search space
   - Merge into base config
   - Call `training_runner.run_training(config, trial=trial_obj)`
   - Return objective value to Optuna
5. Study automatically saved to SQLite database

### `scripts/run_xai_analysis.py`
**Purpose:** Generate Integrated Gradients attributions and XAI artifacts.

**Flow:**
1. Parse `--run-dir`
2. Load run config and checkpoints
3. Load test data
4. For each outer fold:
   - Load best checkpoint
   - Compute IG attributions for correctly classified trials
   - Save per-fold heatmaps
5. Aggregate across folds (grand-average)
6. Generate topomaps (requires montage)
7. Time-frequency analysis (Morlet wavelets)
8. Per-class attributions
9. Generate consolidated HTML/PDF report

### `scripts/run_posthoc_stats.py`
**Purpose:** Post-hoc statistical analysis (group/subject-level).

**Flow:**
1. Parse `--run-dir` and optional flags
2. Load `outer_eval_metrics.csv` and `test_predictions_outer.csv`
3. Compute group-level stats (mean + 95% CI)
4. Per-subject binomial tests (FDR correction)
5. Optional: GLMM analysis via R (if `--glmm`)
6. Generate forest plots, caterpillar plots
7. If permutation summary available, compute empirical p-values

### `scripts/prepare_from_happe.py`
**Purpose:** One-time data preparation (EEGLAB → MNE .fif).

**Flow:**
1. Discover HAPPE `.set` files
2. For each subject:
   - Load EEG epochs
   - Load behavioral CSV
   - Align epochs ↔ behavior (strict matching)
   - Remove `Condition==99` trials
   - Encode task labels
   - Attach montage
   - Save as `.fif`

## Optuna Tools

### `optuna_tools/` - Report Generation Pipeline

#### `config.py`
**Responsibilities:** Build paths and settings (environment-aware).

#### `discovery.py`
**Responsibilities:** Find Optuna `.db` files and completed trials.

#### `csv_io.py`
**Responsibilities:** Export `!all_trials-<study>.csv` (sorted by objective).

#### `plotting.py`
**Responsibilities:** Generate parallel coordinate plots (HTML + PNG).

#### `reports.py`
**Responsibilities:** Generate Top-3 reports (HTML + PDF via Playwright).

#### `meta.py`
**Responsibilities:** Cache metadata (skip up-to-date studies).

#### `index_builder.py`
**Responsibilities:** Build global `optuna_runs_index.csv`.

#### `runner.py`
**Responsibilities:** Orchestrate full refresh (all studies).

### `scripts/refresh_optuna_summaries.py`
**Purpose:** CLI entry point for Optuna refresh.

**Flow:**
1. Call `optuna_tools.runner.refresh_all_studies()`
2. For each discovered study:
   - Export CSV
   - Generate plots
   - Generate Top-3 report
3. Optional: Rebuild global index

## Configuration System

### Layered Config Loading

**Order:**
1. `configs/common.yaml` (global defaults)
2. `configs/tasks/<task>/base.yaml` (task-specific)
3. Command-line `--set` overrides

**Merging:** Later layers override earlier layers (dict merge).

**Validation:** Required parameters checked (e.g., `seed`, `optuna_objective`).

**Output:** `resolved_config.yaml` (merged result, reproducible).

### Config Categories

**Global defaults (`common.yaml`):**
- Dataset caching
- Channel exclusion lists
- Default augmentation settings

**Task-specific (`tasks/<task>/base.yaml`):**
- Task name
- Seed
- Optuna objective
- Training hyperparameters (epochs, LR, batch size)
- Model architecture (F1, D, kernels, dropout)

**Search spaces (`tasks/<task>/stepX_space_*.yaml`):**
- Hyperparameter distributions (choice, uniform, loguniform)

**Search controllers (`tasks/<task>/stepX_search.yaml`):**
- Optuna study name
- Sampler/pruner settings
- Objective metric

## Data Flow Diagram

```
HAPPE .set files
       ↓
[prepare_from_happe.py]
       ↓
    .fif epochs (per subject)
       ↓
[MaterializedEEGDataset]
       ↓ (optional caching)
   PyTorch DataLoader
       ↓
[training_runner.py]
   ├→ [setup_orchestrator] → outer splits
   ├→ [outer_loop] → for each fold
   │    ├→ [inner_loop] → K-fold training
   │    ├→ [checkpointing] → save best
   │    └→ [evaluation] → test on outer fold
   └→ [artifact_writer] → CSVs, plots, summary
       ↓
   results/runs/<run_dir>/
       ├→ summary_*.json
       ├→ outer_eval_metrics.csv
       ├→ test_predictions_outer.csv
       ├→ plots_outer/
       └→ resolved_config.yaml
```

## Key Design Decisions

### Why Modular Training?

**Before:** Monolithic `training_runner.py` (~2000 lines, hard to test/modify)

**After:** Delegated to specialized modules (~200 lines each, single responsibility)

**Benefits:**
- Unit testability (mock dependencies)
- Clear boundaries (inner loop doesn't know about outer loop)
- Easier debugging (isolate failures)

### Why Dataset Caching?

**Problem:** Optuna searches run 50-200 trials. Without caching, each trial reloads .fif files (~50-100 seconds startup).

**Solution:** In-memory cache keyed by (task, crop_ms, channels, subjects). First trial builds, rest reuse.

**Trade-off:** ~2-5 GB RAM per cached dataset, but 100× faster trial startup.

### Why Nested CV?

**Problem:** Using same data for hyperparameter selection and performance evaluation → optimistically biased estimates.

**Solution:** Inner loop selects hyperparameters (never sees outer test). Outer loop evaluates generalization (unbiased).

**Cost:** More compute (K² folds instead of K), but scientifically rigorous.

### Why Ensemble Mode?

**Problem:** Single model has high variance (random initialization, data order).

**Solution:** Average softmax across K models → more stable predictions.

**Alternative:** Refit mode (simpler, but higher variance).

## Next Steps

- For detailed configuration, see [Configuration Reference](CONFIGURATION.md)
- For understanding module interactions, see [Technical Details](TECHNICAL_DETAILS.md)
- For troubleshooting, see [Troubleshooting Guide](TROUBLESHOOTING.md)
