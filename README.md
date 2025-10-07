conda activate eegnex-env

## Overview
This repository implements a streamlined EEG decoding pipeline aligned with the protocol of Borra et al. (2024/2025), adapted to our Numbers‑Cognition EEG study. We run a three‑step Optuna search (Step 3 augmentations optional) and evaluate with subject‑aware CV (GroupKFold or LOSO) and multi‑seed reporting. The project focuses on four tasks:
- cardinality_1_3
- cardinality_4_6
- cardinality_1_6
- landing_digit_1_3_within_small_and_cardinality
- landing_digit_4_6_within_large_and_cardinality

### Key ideas
- Preprocessing is performed externally with HAPPE; we convert EEGLAB `.set` to `.fif` once with `scripts/prepare_from_happe.py`.
- Optuna search runs directly on materialized `.fif` via `scripts/optuna_search.py` with a `--stage` flag; we use TPE with MedianPruner (warmup=10) and per‑epoch pruning on the metric specified by `optuna_objective` (e.g., `composite_min_f1_plur_corr` for balanced decodability+distinctness (recommended), `inner_mean_min_per_class_f1` for worst-class focus, `inner_mean_plur_corr` for plurality correctness, or `inner_mean_macro_f1` for average performance). Per-epoch pruning and checkpoint selection now align with the configured objective to prevent metric drift.
- Stage progression is manual (by design): you choose the next stage and feed winners via YAML overlays; the runner does not auto‑chain stages.
- Behavior alignment is strict by default; any epochs↔behavior mismatch raises and aborts.
- Default channel policy excludes non‑scalp channels (`use_channel_list: non_scalp`). The Cz‑ring knob (`cz_step`) is disabled by default.
- Every run writes `resolved_config.yaml` in its run directory for reproducibility and re‑use.
- Final evaluation can average across ≥10 seeds and generate XAI reports.
- Outer test predictions: set `outer_eval_mode: ensemble` to average softmax across inner K models for stability (recommended), or `outer_eval_mode: refit` to refit one model on the full outer‑train (optional subject‑aware val via `refit_val_frac`) before testing.
 - Determinism: strict seeding for Python/NumPy/Torch, per‑worker DataLoader seeding. A determinism banner is printed and persisted into reports.
 - Provenance: reports include the exact model class (e.g., `braindecode.models.EEGNeX`), library versions (torch, numpy, sklearn, mne, braindecode, captum, optuna, python), and determinism flags.
 - Dataset caching:
   - `dataset_cache_memory: true` enables an in-process RAM cache of the fully built dataset (X, y, groups, channels, times). The first trial in a Python process builds it; subsequent trials reuse instantly (skips repeated MNE loads and cropping). Restarting the Python process clears it.
 - Modular training orchestration (refactor complete): `training_runner.py` is now a thin coordinator that delegates to modular components under `code/training/` (setup, inner loop, outer loop, evaluation, checkpointing, metrics) and `code/artifacts/` (CSV writers, plot builders, overall plot orchestrator, artifact writer). This preserves legacy behavior while improving testability and clarity.

### Per‑run artifacts (besides checkpoints/plots)
- `summary_<TASK>_<ENGINE>.json`: metrics + hyper + determinism + lib versions + model class + hardware (+ config_hash, exclusion summary, telemetry reference). Includes per-fold values for `min_per_class_f1` and `cohen_kappa`.
- `resolved_config.yaml`: the frozen config used for the run
- `splits_indices.json`: exact outer/inner indices and subjects for all folds (auditable splits)
- `learning_curves_inner.csv`: all inner‑fold learning curves across outer folds (epoch‑wise train/val loss/acc/macro‑F1)
- `outer_eval_metrics.csv`: one row per outer fold with held‑out subjects, n_test, acc, macro‑F1, weighted‑F1, per‑class F1, min‑per‑class‑F1, and Cohen's kappa (+ OVERALL row with means and standard deviations)
- `test_predictions_outer.csv`: one row per out‑of‑fold test trial (outer) — subject_id, trial_index, true/pred labels, p_trueclass, logp_trueclass, full probs — ready for mixed‑effects models
- `test_predictions_inner.csv`: one row per inner validation trial — outer_fold, inner_fold, subject_id, trial_index, true/pred labels, p_trueclass, logp_trueclass, full probs — useful for inner vs outer performance comparison
- `logs/runtime.jsonl`: JSONL event log (fold boundaries, class-weights saved, chance-level computed, split export, posthoc start/end)
- `pip_freeze.txt` and (if available) `conda_env.yml`: environment freeze for reproducibility

### Run directory structure
- Run root contains text/JSON/CSV/HTML/PDF summaries and tables
- Subdirectories:
  - `plots_outer/` — standard confusion matrices and learning curves per outer fold; overall confusion matrix
  - `plots_outer_enhanced/` — enhanced plots with inner vs outer metric comparison and per-class F1 scores as side text (used in Optuna Top-3 reports)
  - `ckpt/` — all checkpoints (e.g., `fold_XX_inner_YY_best.ckpt`, `fold_XX_refit_best.ckpt`)
  - `xai_analysis/` — XAI outputs (per‑fold heatmaps, grand average, topoplots, summaries)

### Transparency and safeguards
- Hardware/runtime banner: GPU names, CUDA version, CPU, OS are printed and included in reports.
- Split leakage guards: assertions ensure no subject appears in both train and test for any outer fold; inner folds are subject‑exclusive.
- Class imbalance disclosure: class weights used for `CrossEntropyLoss` are saved per fold (e.g., `fold_XX_inner_YY_class_weights.json`, and `fold_XX_refit_class_weights.json`).

### Permutation testing (empirical null, optional)
- Purpose: build a null distribution by shuffling labels (no-signal hypothesis) with fixed outer/inner splits.
- Config keys (in YAML):
  - `permute_labels: true`
  - `n_permutations: 200`
  - `permute_scope: within_subject` (or `global`)
  - `permute_stratified: true` (preserve within-subject class balance)
  - `permute_seed: 123`
- CLI (reusing observed splits from a completed run):
```powershell
python -X utf8 -u train.py `
  --task cardinality_1_3 `
  --engine eeg `
  --base  configs/tasks/cardinality_1_3/resolved_config.yaml `
  --permute-labels `
  --n-permutations 200 `
  --permute-scope within_subject `
  --permute-stratified `
  --permute-seed 123 `
  --observed-run-dir "results\runs\<observed_run_dir_name>"
```
- Outputs (written next to the observed run):
  - `<observed_run_dir>_perm_test_results.csv`: rows of (perm_id, outer_fold, acc, macro_f1, n_test_trials)
  - `<observed_run_dir>_perm_summary.json`: observed scores, null means/SD, empirical p-values, and bookkeeping.

### Entry points (when to use what)
- `train.py`: Single run or multi‑seed runs for a given task/engine using layered configs; ideal for validating a resolved YAML and optionally running XAI on completion (`--run-xai`). Multi‑seed: add `seeds: [41, 42, ...]` in YAML.
- Outer evaluation mode (must be explicitly specified):
  - `outer_eval_mode: ensemble` (recommended): mean-softmax over K inner models for test predictions.
  - `outer_eval_mode: refit`: refit one model on full outer-train (set `refit_val_frac>0` to enable subject-aware early stop).
- `scripts/optuna_search.py`: Unified Optuna driver for Stage 1/2/3 sweeps (`--stage step1|step2|step3`) over a search space YAML; prunes per‑epoch on inner macro‑F1 and seeds the TPE sampler from config.
- `scripts/final_eval.py`: Multi-seed final evaluation; writes aggregate metrics JSON.
- `scripts/run_posthoc_stats.py`: Post‑hoc stats (GLMM/forest/caterpillar) under `stats/`.
- `scripts/run_xai_analysis.py`: Per‑fold attributions and consolidated XAI HTML.
- `scripts/analyze_nloso_subject_performance.py`: Per‑subject and per‑fold performance analysis; creates `subject_performance/` folder with CSVs, bar charts, confusion matrices, and optional inner vs outer comparison (auto‑detects `test_predictions_outer.csv` and `test_predictions_inner.csv`).
- `scripts/prepare_from_happe.py`: One‑time materialization of per‑subject `.fif` epochs with aligned behavior and montage from HAPPE/EEGLAB `.set`.

## Repository layout
- code/
  - preprocessing/mne_pipeline.py — spatial sampling (Cz ring), time cropping, channel alignment helpers
  - datasets.py — materialized `.fif` loader, channel unification
  - model_builders.py — EEGNeX builder and train‑time augmentations
  - training_runner.py — thin orchestrator that delegates to modular training/artifact components
  - training/ — modular training components
    - setup_orchestrator.py — logging (JSONL), outer CV splits (GroupKFold/LOSO), channel topomap
    - inner_loop.py — per‑inner‑fold training loop (LR/aug warmup, mixup, pruning integration)
    - outer_loop.py — orchestrates one complete outer fold (inner K‑fold, selection, test eval, per‑fold plots)
    - evaluation.py — outer test evaluation (ensemble or refit modes)
    - checkpointing.py — objective‑aligned early stopping and best‑checkpoint selection (tie‑break on val loss)
    - metrics.py — objective computation for pruning/selection (supports composite_min_f1_plur_corr)
  - artifacts/ — artifact and plotting utilities
    - csv_writers.py — learning curves, outer eval metrics (with OVERALL row), predictions CSV writers
    - plot_builders.py — objective‑aware plot titles and per‑class info strings
    - overall_plot_orchestrator.py — overall (cross‑fold) confusion plots (simple/enhanced)
    - artifact_writer.py — orchestrates writing splits JSON, CSVs, and prediction files
- engines/eeg.py — engine wrapper for raw‑EEG models (EEGNeX)
- tasks/ — label functions per task
- configs/
  - common.yaml — global defaults (training, crop_ms, channel lists)
  - tasks/<task>/
    - base.yaml — base hyper-parameters
    - step1_search.yaml — controller for Step 1
    - step1_space_*.yaml — Step 1 search space (architecture/spatial/big levers)
    - step2_search.yaml — controller for Step 2
    - step2_space_*.yaml — Step 2 space (recipe refinement/stability)
    - step3_search.yaml — controller for Step 3
    - step3_space_*.yaml — Step 3 space (augmentations)
- scripts/
  - prepare_from_happe.py — convert HAPPE EEGLAB `.set` to per‑subject `.fif` with metadata + montage
  - optuna_search.py — Unified Optuna TPE driver (Stage 1/2/3/4 via --stage)
  - search_finalist.py — optional finalist tuner (joint sensitive params)
  - final_eval.py — multi‑seed final evaluation and consolidated reporting
  - run_xai_analysis.py — per‑fold attributions and consolidated XAI HTML/PDF
  - run_posthoc_stats.py — post‑hoc statistics (group efficacy and subject reliability)
  - analyze_nloso_subject_performance.py — per‑subject/per‑fold performance + inner vs outer comparison
- optuna_tools/
  - config.py, runner.py, plotting.py, reports.py, discovery.py, db.py, csv_io.py, meta.py, index_builder.py — modular Optuna refresh pipeline (plots/CSV/top‑3 report/index)
- results/optuna/
  - refresh_all_studies.bat — Windows wrapper that calls the Python entry point
- scripts/
  - refresh_optuna_summaries.py — CLI entry for refreshing studies (plots/CSV/top‑3); uses optuna_tools
  - optuna_index_builder.py — rebuild a global `optuna_runs_index.csv` from per‑trial summaries
  - Parallel plots: HTML + PNG only (no SVG/PDF). PNG export uses thicker lines and high scale for readability
- data directories (git‑ignored):
  - eeg-raw/subjectXX.mff
  - data_behavior/behavior_data/SubjectXX.csv
  - cache/epochs/<fingerprint>/
  - data_preprocessed/<dataset>/

## Data model (materialized epochs)
- Per-subject files: `data_preprocessed/<dataset>/sub-XX_preprocessed-epo.fif`
- Each `.fif` contains MNE Epochs with metadata; key columns expected downstream:
  - `SubjectID`, `Block`, `Trial`, `Procedure`, `Condition`
  - `Target.ACC`, `Target.RT`, `Trial_Continuous`
  - Derived: `direction`, `change_group`, `size` (see `scripts/prepare_from_happe.py`)
- Training dataset builds tensors as X: `(N, 1, C, T)` (µV-scaled), y: `(N,)` (label-encoded). Channels are intersected and ordered identically across subjects for CNNs.

## Data expectations
- HAPPE output: `data_input_from_happe/<dataset>/5 - processed .../*.set` and QC CSV in `6 - quality_assessment_outputs ...`
- Behavior: `data_behavior/data_UTF8/SubjectXX.csv`
- Montage: `net/AdultAverageNet128_v1.sfp` (used during conversion to attach 3D positions)

## Typical workflow
1) Convert HAPPE EEGLAB `.set` to `.fif`
```powershell
python scripts/prepare_from_happe.py
```

1a) Single LOSO run (optional)
```powershell
python -X utf8 -u train.py `
  --task cardinality_1_3 `
  --engine eeg `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --run-xai
```

1b) Single LOSO run for 6‑class task
1c) Single LOSO run with outer refit (no early stop during refit)
```powershell
python -X utf8 -u train.py `
  --task cardinality_1_3 `
  --engine eeg `
  --base  configs/tasks/cardinality_1_3/resolved_config.yaml `
  --set n_folds=null outer_eval_mode=refit refit_val_frac=0.0
```
```powershell
python -X utf8 -u train.py `
  --task cardinality_1_6 `
  --engine eeg `
  --base  configs/tasks/cardinality_1_6/base.yaml
```

1d) Multi‑seed LOSO with a fixed resolved config (preferred)
Add to your YAML:
```yaml
seeds: [41, 42, 43, 44, 45]
```
Then run once:
```powershell
python -X utf8 -u train.py `
  --task cardinality_1_3 `
  --engine eeg `
  --base configs/tasks/cardinality_1_3/resolved_config.yaml
```
This produces one run directory per seed (naming includes `seed_<N>` before `crop_ms`) and an aggregate JSON next to the runs: `results/runs/<timestamp>_<task>_<engine>_seeds_aggregate.json`.

Run a single LOSO with an Optuna winner:
```powershell
python -X utf8 -u train.py `
  --task cardinality_1_3 `
  --engine eeg `
  --base configs/tasks/cardinality_1_3/resolved_config.yaml `
  --run-xai
```

2) Optuna search on materialized `.fif` (choose stage)
```powershell
# Step 1 (architecture/spatial/big levers)
python -X utf8 -u scripts/optuna_search.py `
  --stage step1 `
  --task cardinality_1_3 `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --cfg   configs/tasks/cardinality_1_3/step1_search.yaml `
  --space configs/tasks/cardinality_1_3/step1_space_scaffold.yaml `
  --trials 48

# Step 2 (recipe refinement/stability)
python -X utf8 -u scripts/optuna_search.py `
  --stage step2 `
  --task cardinality_1_3 `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --cfg   configs/tasks/cardinality_1_3/step2_search.yaml `
  --space configs/tasks/cardinality_1_3/step2_space_recipe.yaml `
  --trials 48

# Step 3 (augmentations; optional)
python -X utf8 -u scripts/optuna_search.py `
  --stage step3 `
  --task cardinality_1_3 `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --cfg   configs/tasks/cardinality_1_3/step3_search.yaml `
  --space configs/tasks/cardinality_1_3/step3_space_joint.yaml `
  --trials 48
```

#### Performance tips
- Enable in-process dataset cache to avoid repeated MNE loads and time cropping within an Optuna run:
```yaml
# common.yaml (already enabled in this repo)
dataset_cache_memory: true
```
- Cache persists only for the current Python process. Restarting clears it.

3) Final evaluation (multi‑seed, LOSO). With `--use-best`, merges best from step1 → step2 → step3 → finalist (last wins).
```powershell
python scripts/final_eval.py `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/base.yaml `
  --seeds 10
```

Alternative: manual multi‑seed with a fixed resolved config (preferred when working directly with resolved YAML)
```powershell
for ($i=0; $i -lt 10; $i++) {
  $seed = 42 + $i
  python -X utf8 -u train.py `
    --task cardinality_1_3 `
    --engine eeg `
    --base  configs/tasks/cardinality_1_3/resolved_config.yaml `
    --set n_folds=null `
    --set seed=$seed
}
```

### Optuna studies: refresh (modular tools)

Refresh Optuna studies (modular)

Tools live under optuna_tools/ and are driven by scripts/refresh_optuna_summaries.py.

What’s included

config.py – builds paths/settings (env-aware).

discovery.py – finds studies & trials.

csv_io.py – writes !all_trials-<study>.csv (sorted by `inner_mean_min_per_class_f1` if available, else `inner_mean_macro_f1`, then `inner_mean_acc`).

db.py, plotting.py – loads Optuna SQLite and exports plots (HTML + PNG).

reports.py – Top-3 report (HTML + PDF; Playwright w/ matplotlib fallback). Prioritizes trials by `inner_mean_min_per_class_f1` (worst-class performance) to ensure all classes are decodable. Uses enhanced plots from `plots_outer_enhanced/`.

meta.py – cache to skip up-to-date studies.

index_builder.py – writes global optuna_runs_index.csv.

runner.py – orchestrates a full refresh; optional index rebuild.

Run it

Double-click (Windows): results\optuna\refresh_all_studies.bat


## Preprocessing details
- HAPPE produces cleaned `.set` per subject. `prepare_from_happe.py` aligns behavior, removes `Condition==99`, encodes labels, attaches montage (`net/AdultAverageNet128_v1.sfp`), and saves per‑subject `.fif`.
- Default at train time: `use_channel_list: non_scalp` (excludes 28 non-scalp channels, leaving 100 channels). The Cz‑ring (`cz_step`) heuristic is disabled by default (`cz_step: 0`). You can still set `crop_ms` and/or explicit `include_channels`.

### Spatial channel selection (`cz_step`)
The `cz_step` parameter enables progressive spatial sampling centered on Cz:
- **`cz_step: 0`** or **`null`** (default): DISABLED — keeps all channels after non_scalp exclusion (~100 channels)
- **`cz_step: 1`**: Keeps 20% of channels closest to Cz (tight central ring, ~20 channels)
- **`cz_step: 2`**: Keeps 40% of channels closest to Cz (~40 channels)
- **`cz_step: 3`**: Keeps 60% of channels closest to Cz (~60 channels)
- **`cz_step: 4`**: Keeps 80% of channels closest to Cz (~80 channels)
- **`cz_step: 5`**: Keeps 100% of channels (all, same as disabled)

Implementation: `code/preprocessing/epoch_utils.py` uses 3D Euclidean distance from Cz (formula: `frac = min(1.0, max(0.1, cz_step * 0.2))`). Requires montage attachment during data prep. Gracefully degrades to no-op if Cz unavailable.

To include in Optuna searches, add to your `step1_space_*.yaml`, `step2_space_*.yaml`, or `step3_space_*.yaml`:
```yaml
cz_step:
  type: choice
  options: [2, 3, 4]  # Recommended: 40%, 60%, 80% coverage
```

## Reproducibility & Scientific Rigor
- **Outer split:** `GroupKFold` when `n_folds` is set; otherwise `LOSO`.
- **Strict inner subject‑aware K‑fold:** set `inner_n_folds` (≥2). If infeasible (too few unique subjects), the run raises with a clear error.
- **Determinism:** `PYTHONHASHSEED`, Python/NumPy/Torch seeds, per‑worker DataLoader seeding, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`, and `CUBLAS_WORKSPACE_CONFIG` to stabilize CUDA GEMM. Determinism banner is printed and persisted.
- **Provenance:** model class path, library versions, and determinism flags are included in the TXT/HTML report and JSON.
- **Seeds (REQUIRED):** set a single `seed: 1` (or any integer) or a list `seeds: [41, 42, ...]` for multi‑seed loops. **No fallback is provided**—if `seed` is missing, the run fails immediately to prevent untracked randomness. Run directories include `seed_<N>` in the name; a cross‑seed aggregate JSON is written. The parameter `random_state` is no longer used (removed project-wide).
- **Optuna objective (REQUIRED):** TPE sampler is seeded from config. You **must** explicitly specify `optuna_objective` in your config (e.g., `inner_mean_min_per_class_f1` for worst-class focus, `inner_mean_plur_corr` for plurality correctness, `inner_mean_macro_f1` for average performance, or `inner_mean_acc` for accuracy). **No fallback is provided**—this ensures you consciously choose your research objective. Pruning signal is the configured objective metric. Top-3 reports prioritize `inner_mean_min_per_class_f1` to identify trials where all classes are decodable.
- **Outer eval mode (REQUIRED):** You must explicitly set `outer_eval_mode: ensemble` (recommended) or `outer_eval_mode: refit` in your config. **No fallback is provided**—this forces conscious choice of your evaluation strategy.
- **Objective-aligned pruning and checkpointing:** Optuna's per-epoch pruning signal and checkpoint selection ("best inner epoch") now use the same metric as `optuna_objective`, ensuring scientific consistency. Previously (before Oct 5, 2025), these always used `val_macro_f1` regardless of objective, potentially leading to suboptimal model selection. Current behavior: if you optimize for `composite_min_f1_plur_corr`, pruning decisions and checkpoint selection also use the composite score.
- **Audit artifacts:** `splits_indices.json`, `learning_curves_inner.csv`, `outer_eval_metrics.csv`, `test_predictions_outer.csv`, and `test_predictions_inner.csv` capture splits, pruning traces, outer‑fold metrics (including min-per-class-F1 and Cohen's kappa), per‑trial out‑of‑fold predictions (ready for mixed‑effects models), and inner validation predictions (for inner vs outer comparison).
- **XAI checkpoint resolution order per fold:** `fold_XX_refit_best.ckpt` → `fold_XX_best.ckpt` → first `fold_XX_inner_YY_best.ckpt`.

Tip: To force LOSO in any run that has a resolved config with `n_folds`, either set `n_folds: null` inside the YAML, or pass `--set n_folds=null` on the CLI.

### Report PDF export (optional)
- HTML reports are always written. To enable PDF export via Playwright:
  - Ensure `playwright` is installed (listed in `environment.yml` under pip)
  - Then run: `playwright install`

### Configuration knobs
- `configs/tasks/<task>/base.yaml`:
  - **`seed` (REQUIRED):** integer seed for reproducibility. No fallback—run fails if missing.
  - **`optuna_objective` (REQUIRED for Optuna):** metric to optimize. Choose: `composite_min_f1_plur_corr` (balanced decodability+distinctness, recommended), `inner_mean_min_per_class_f1` (worst-class focus), `inner_mean_plur_corr` (plurality correctness/row-wise plurality), `inner_mean_macro_f1` (average F1), or `inner_mean_acc` (accuracy). No fallback—ensures conscious research objective choice.
  - **`composite_min_f1_weight` (REQUIRED if using `composite_min_f1_plur_corr`):** Weight for min-per-class F1 in the composite objective (0.0–1.0). Higher values (e.g., 0.65) prioritize decodability; lower values prioritize distinctness. Constitutional requirement: must be explicitly specified (no default fallback). Example: `composite_min_f1_weight: 0.50` balances both metrics equally.
  - **`outer_eval_mode` (REQUIRED):** `ensemble` (recommended, averages K inner models) or `refit` (single model on full outer-train). No fallback.
  - `n_folds`: number of outer folds (uses GroupKFold). If omitted, uses LOSO.
  - `inner_n_folds`: number of inner folds (must be ≥2). Strictly enforced; insufficient subjects → error.
  - `epochs`, `early_stop`, `batch_size`, `lr`, etc. apply to each inner fold training.

- Stability warm-ups:
  - `lr_warmup_frac`: fraction of epochs (0–1) for linear LR warm-up from `lr_warmup_init * lr` to `lr`. Default 0.
  - `lr_warmup_init`: initial LR scale (0–1) at epoch 1 during warm-up. Default 0.
  - `aug_warmup_frac`: fraction of epochs (0–1) to ramp augmentation probabilities and magnitudes from 0→full. Default 0.

### Optimization Objectives Explained

The `optuna_objective` parameter controls what metric the hyperparameter search optimizes for. Each has distinct implications for model behavior:

#### `composite_min_f1_plur_corr` (Balanced Decodability + Distinctness) **RECOMMENDED** ⭐

- **What it measures:** A weighted sum of min-per-class F1 and plurality correctness, ensuring both decodability and distinctness are continuously optimized
- **Formula:** `composite = (composite_min_f1_weight × min_per_class_f1) + ((1 - composite_min_f1_weight) × plurality_correctness)`
- **When to use:** **DEFAULT CHOICE for cognitive neuroscience research.** Prevents "metric gaming" where a model sacrifices one property to maximize the other
- **Configuration:** Requires `composite_min_f1_weight: 0.50` (or any value in [0.0, 1.0]) in your config YAML
- **Why it's superior to optimizing metrics separately:**
  - **Prevents plurality correctness gaming:** A model can achieve 100% plurality correctness with terrible F1 (e.g., 10% accuracy) by making one class hyper-confident while others collapse. The composite prevents this by requiring BOTH metrics remain high.
  - **Prevents min-F1 stagnation:** Optimizing only min-F1 may produce models where classes barely meet threshold but lack distinct neural signatures. The composite ensures classes remain distinguishable.
  - **Continuous optimization:** Unlike threshold-based objectives (which create discontinuities), the composite always balances both goals throughout the search space.
- **Example with weight=0.65 (prioritizing decodability):**
  ```
  Trial A: min_F1=40%, plur_corr=100% → composite = 0.65×40 + 0.35×100 = 61.0
  Trial B: min_F1=50%, plur_corr=67%  → composite = 0.65×50 + 0.35×67  = 56.0
  Trial C: min_F1=45%, plur_corr=100% → composite = 0.65×45 + 0.35×100 = 64.3 ← BEST
  ```
  Trial C wins because it balances strong plurality correctness (100%) with decent decodability (45%).
- **Scientific rationale:** Your research claim requires BOTH properties: (1) all classes must be decodable (operationalized by min-per-class F1), and (2) each class must have a distinct neural signature (operationalized by plurality correctness). The composite objective is the mathematically sound way to enforce both simultaneously.
- **Per-epoch behavior:** Pruning and checkpoint selection use the same composite score, ensuring consistency between hyperparameter optimization, early stopping, and final model selection.

#### `inner_mean_min_per_class_f1` (Worst-Class Focus)
- **What it measures:** The minimum F1 score across all classes (averaged across inner folds)
- **When to use:** When you want to ensure the model performs adequately on ALL classes, including the hardest one
- **Example:** If class "2" is difficult to distinguish, this metric forces the model to not ignore it
- **Scientific rationale:** Ensures balanced performance; prevents "taking the easy way out" by only learning easy classes

#### `inner_mean_plur_corr` (Plurality Correctness / Plurality Correctness) **NEW**
- **What it measures:** The proportion of classes where the correct prediction is the most frequent prediction (averaged across inner folds)
- **Formula:** For each true class (row in confusion matrix), check if the diagonal element is the row maximum. Return the proportion where this is true (0.0 to 1.0)
- **When to use:** When you want to ensure the model has the "right bias" for each class—even if not perfectly accurate, the correct class should be the plurality prediction
- **Example confusion matrix:**
  ```
          Pred:  1   2   3
  True 1: [100  30  20]  ← max is 100 (diagonal) ✓ counts as 1
  True 2: [ 40  60  50]  ← max is 60 (diagonal) ✓ counts as 1
  True 3: [ 10  80  30]  ← max is 80 (OFF-diagonal) ✗ counts as 0
  
  Plurality Correctness = 2/3 = 0.667 (67%)
  ```
- **Scientific rationale:** Ensures the model's strongest prediction for each true class is the correct one, preventing systematic misclassification patterns (e.g., always predicting "3" when the answer is "2")
- **Interpretation:**
  - 1.0 (100%) = Perfect plurality correctness (for every class, correct prediction is most frequent)
  - 0.67 (67%) = 2 out of 3 classes have correct prediction as plurality
  - 0.33 (33%) = Only 1 class has correct prediction as plurality
  - 0.0 (0%) = No class has correct prediction as plurality (worst case—model systematically wrong)

**⚠️ WARNING: Using `inner_mean_plur_corr` alone is scientifically unsound.**

Optimizing ONLY for plurality correctness can produce models with perfect plurality correctness but useless performance (e.g., 10% accuracy). Example:
- Class 1: 10% correct (diagonal), 5% as 2, 5% as 3 → diagonal wins ✓
- Class 2: 8% correct (diagonal), 4% as 1, 3% as 3 → diagonal wins ✓  
- Class 3: 12% correct (diagonal), 6% as 1, 2% as 2 → diagonal wins ✓

Result: Perfect plurality correctness (1.0), but 10% accuracy. This model is useless for your research but would rank as "best trial" in Optuna.

**Solution:** Use `composite_min_f1_plur_corr` instead, which requires BOTH decodability (min F1) AND distinctness (plurality correctness).

#### `inner_mean_macro_f1` (Average Performance)
- **What it measures:** The average F1 score across all classes (averaged across inner folds)
- **When to use:** When you want good overall performance across classes, but are willing to accept some imbalance
- **Scientific rationale:** Balances precision and recall across classes; standard metric for multi-class problems

#### `inner_mean_acc` (Raw Accuracy)
- **What it measures:** The proportion of correct predictions (averaged across inner folds)
- **When to use:** When classes are balanced and you care about overall correctness
- **Caveat:** Can be misleading with class imbalance; model may ignore minority classes

**Recommendation:** For numerosity/cognitive research where all stimulus classes are equally important, use `composite_min_f1_plur_corr` with `composite_min_f1_weight: 0.50` to 0.65 (recommended) to ensure the model both decodes all classes accurately AND learns distinct neural signatures for each class. This prevents metric gaming and aligns with your scientific claim about distinguishable cognitive representations.

## Commands cheat-sheet
- Convert: `python scripts/prepare_from_happe.py`
- Search (unified): `python scripts/optuna_search.py --stage step1|step2|step3 ...`
- Final eval: `python scripts/final_eval.py ...`
- Refresh Optuna summaries: `results\optuna\refresh_all_studies.bat`
 - Run XAI on a completed run directory:
```powershell
python -X utf8 -u scripts/run_xai_analysis.py --run-dir "results\runs\<run_dir_name>"
```

### XAI Analysis (Integrated Gradients, Topomaps, Time-Frequency)

The XAI system provides comprehensive interpretability for trained EEG models using Integrated Gradients. After completing a training run, generate XAI artifacts using:

```powershell
python -X utf8 -u scripts/run_xai_analysis.py --run-dir "results\runs\<run_dir_name>"
```

Or enable automatic XAI generation during training:
```powershell
python -X utf8 -u train.py --task cardinality_1_3 --engine eeg --base configs/tasks/cardinality_1_3/base.yaml --run-xai
```

#### Output Structure

All XAI artifacts are written to `<run_dir>/xai_analysis/`:

**Per-Fold Outputs:**
- `integrated_gradients/fold_XX_xai_attributions.npy`: IG attribution matrices (C×T) for correctly classified trials
- `integrated_gradients/fold_XX_xai_heatmap.png`: Channel×time heatmaps
- `integrated_gradients_per_class/fold_XX_class_labels.npy`: Per-trial class labels for per-class filtering

**Grand-Average Outputs:**
- `grand_average_xai_attributions.npy`: Mean IG across all folds (C×T)
- `grand_average_xai_heatmap.png`: Grand-average channel×time heatmap
- `grand_average_xai_topoplot.png`: Scalp topography of overall channel importance (requires montage)
- `grand_average_per_class/class_XX_<name>_xai_heatmap.png`: Per-class attribution heatmaps
- `grand_average_per_class/class_XX_<name>_xai_topoplot.png`: Per-class topomaps (requires montage)
- `grand_average_time_frequency.png`: Time-frequency decomposition of grand-average IG (Morlet wavelets)
- `grand_average_ig_peak1_topoplot_<t0>-<t1>ms.png`: Top temporal window #1 with peak channel importance
- `grand_average_ig_peak2_topoplot_<t0>-<t1>ms.png`: Top temporal window #2 with peak channel importance

**Consolidated Reports:**
- `consolidated_xai_report.html`: Interactive HTML report with all visualizations embedded
- `consolidated_xai_report.pdf`: Printable PDF version (requires Playwright)

The HTML report includes:
1. **Top-K Channel Summary**: Overall most important channels (default K=10)
2. **Top-2 Spatio-Temporal Events**: Peak time windows with channel-specific importance
3. **Grand-Average Visualizations**: Heatmaps, topomaps, time-frequency analysis
4. **Per-Class Analysis**: Attribution patterns specific to each class
5. **Per-Fold Gallery**: All fold-wise IG heatmaps

#### Configuration

XAI parameters are controlled via `configs/xai_defaults.yaml`:

```yaml
# Top-K channels to highlight in reports and topomaps
xai_top_k_channels: 10

# Peak window duration for spatio-temporal event analysis (ms)
peak_window_ms: 100

# Frequencies for time-frequency decomposition (Hz)
tf_morlet_freqs: [4, 8, 13, 30]
```

These defaults are merged with the run's config; run config takes precedence.

#### Technical Details

**Integrated Gradients (IG):**
- Computed using Captum's `IntegratedGradients` with 50 integration steps
- Applied only to correctly classified test trials
- Averaged across trials to produce fold-level and grand-average attributions
- Per-class analysis filters trials by true label before averaging

**Topomaps:**
- Generated using MNE-Python's `plot_topomap` with sensor positions from `net/AdultAverageNet128_v1.sfp`
- Requires montage attachment; gracefully skipped if montage unavailable
- Top-K channels are labeled on the scalp plot

**Time-Frequency Analysis:**
- Uses MNE's Morlet wavelet decomposition (`tfr_morlet`)
- Applied to grand-average IG attribution matrix
- Reveals which frequency bands (e.g., theta, alpha, beta) contributed to model decisions
- Requires signal length ≥ wavelet duration; gracefully skipped for short epochs

**Spatio-Temporal Events (Top-2 Peaks):**
- Uses SciPy's `find_peaks` to identify significant temporal windows in the grand-average attribution
- For each peak: computes window-specific channel importance and generates a topomap
- Window duration controlled by `peak_window_ms` (default 100ms)
- Requires montage; skipped if unavailable

#### Troubleshooting

**No topomaps generated:**
- Verify montage file exists: `net/AdultAverageNet128_v1.sfp`
- Check that channel names match montage (e.g., standard 10-20 names like Fp1, Fz, Cz)
- Montage attachment warnings are logged; review console output

**Time-frequency analysis skipped:**
- Signal too short for lowest frequency wavelet
- Reduce `tf_morlet_freqs` minimum or use longer epochs (increase `crop_ms` range)

**PDF generation failed:**
- Install Playwright: `playwright install`
- HTML report is always generated; PDF is optional

#### Post‑hoc stats defaults YAML
Create `configs/posthoc_defaults.yaml` to set default flags for post‑hoc statistics (CLI args override these):
```yaml
posthoc:
  alpha: 0.05
  multitest: fdr  # or "none"
  glmm: true
  forest: true
  chance_rate: null  # auto-detected if null
```


### Post‑hoc statistics (group efficacy and subject reliability)
- Purpose: quantify population‑level efficacy and subject‑level reliability without retraining.
- Inputs: uses `outer_eval_metrics.csv`, `test_predictions_outer.csv`, and permutation summary (if present) from the run directory.
- Defaults: set in `configs/posthoc_defaults.yaml` (alpha, multitest, glmm, forest, chance_rate); CLI args override YAML.
- Run (with defaults from YAML):
```powershell
python -X utf8 -u scripts/run_posthoc_stats.py --run-dir results\runs\<run_dir_name>
```
- Run (with explicit flags, overriding YAML):
```powershell
python -X utf8 -u scripts/run_posthoc_stats.py `
  --run-dir results\runs\<run_dir_name> `
  --alpha 0.05 `
  --multitest fdr `
  --glmm `
  --forest
```
- Outputs (written to `<run_dir>/stats/`):
  - `group_stats.json`: mean and 95% CI for accuracy/macro‑F1; permutation p‑values if available.
  - `per_subject_significance.csv`: per‑subject accuracy, binomial p‑value vs chance, adjusted p‑value, and above‑chance flag.
  - `per_subject_summary.json`: number and proportion of subjects above chance.
  - `glmm_summary.json`: optional population fixed‑effect summary via R lme4, including CI/p‑value.
  - `per_subject_forest.png`: forest plot with Wilson CIs per subject (if `--forest`).
  - `qq_pvalues_fdr.png`: QQ plot of per‑subject p‑values with BH threshold.
  - `glmm_intercept_effect.png`: GLMM intercept effect on log‑odds and probability scales (if `--glmm`).
  - `glmm_caterpillar.png`: caterpillar plot of subject BLUPs on probability scale (if `--glmm`).
  - `perm_density_acc.png`, `perm_density_macro_f1.png`: null densities from permutation test (if available).

### Per‑subject and per‑fold performance analysis
- Purpose: detailed per‑subject and per‑fold accuracy breakdown; optional inner vs outer comparison.
- Inputs: auto‑detects `test_predictions_outer.csv` and `test_predictions_inner.csv` from the run directory.
- Run:
```powershell
python -X utf8 -u scripts/analyze_nloso_subject_performance.py `
  "results\runs\<run_dir_name>"
```
- Outputs (written to `<run_dir>/subject_performance/`):
  - `per_subject_metrics.csv`: per‑subject accuracy and support n
  - `per_fold_metrics.csv`: per‑fold accuracy and support n
  - `acc_by_subject_bar.png`, `acc_by_fold_bar.png`: bar charts with support n annotated
  - `overall_confusion.png`: overall confusion matrix (row‑normalized %)
  - `per_subject_confusion/subject-<id>.png`: per‑subject confusion matrices
  - `report.html`: consolidated HTML report
  - `inner_vs_outer/` (if `test_predictions_inner.csv` is found):
    - `inner_vs_outer_subject_metrics.csv`: per‑subject inner vs outer accuracy with delta
    - `inner_vs_outer_scatter.png`: scatter plot of inner vs outer accuracy by subject

## Publication-ready figures

The project includes a comprehensive system for generating publication-ready figures meeting neuroscience journal standards (Journal of Neuroscience, Nature Neuroscience, Neuron, eNeuro). All figures use white backgrounds, colorblind-safe palettes (Wong 8-color), 600 DPI resolution, and vector formats with embedded fonts.

**Location:** `publication-ready-media/`

**Key features:**
- 10 publication-ready figures (pipeline, nested CV, Optuna optimization, confusion matrices, learning curves, permutation testing, per-subject performance, XAI spatiotemporal, XAI per-class, performance box plots)
- 3 formats per figure (PDF vector, PNG 600 DPI, SVG editable)
- Comprehensive documentation (publication guide, quick reference, complete standards)
- Regeneration scripts with consistent styling (`code/v4_neuroscience/`)

**Quick start:**
```powershell
# View documentation
cd publication-ready-media
# Read README.md for complete guide

# Regenerate all figures (if needed)
cd code/v4_neuroscience
conda activate eegnex-env
python generate_all_v4_figures.py

# Outputs appear in: publication-ready-media/outputs/v4/
```

**For manuscript submission:**
- Use PNG files for initial submission (universal compatibility)
- Use PDF files for final version (vector, scalable)
- See `publication-ready-media/PUBLICATION_GUIDE.md` for complete neuroscience standards
- See `publication-ready-media/QUICK_REFERENCE.md` for 1-page quick start

All figures meet requirements for major neuroscience journals and are ready for immediate submission.

---

## Update github example
PS D:\eeg_nn> conda activate eegnex-env
(eegnex-env) PS D:\eeg_nn> git add .
(eegnex-env) PS D:\eeg_nn> git status
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.md
        modified:   code/training_runner.py

(eegnex-env) PS D:\eeg_nn> git commit -m "Add publication figures and cz_step documentation"
(eegnex-env) PS D:\eeg_nn> git push origin main 
