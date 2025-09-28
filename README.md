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
- Optuna search runs directly on materialized `.fif` via `scripts/optuna_search.py` with a `--stage` flag; we use TPE with MedianPruner (warmup=10) and per‑epoch pruning on inner macro‑F1.
- Behavior alignment is strict by default; any epochs↔behavior mismatch raises and aborts.
- Default channel policy excludes non‑scalp channels (`use_channel_list: non_scalp`). The Cz‑ring knob (`cz_step`) is disabled by default.
- Every run writes `resolved_config.yaml` in its run directory for reproducibility and re‑use.
- Final evaluation can average across ≥10 seeds and generate XAI reports.
- Outer test predictions use an inner‑fold ensemble (mean of softmax across inner K models) for stability; plots use the best inner fold’s curves.
 - Alternatively, set `outer_eval_mode: refit` to refit one model on the full outer‑train (optional subject‑aware val via `refit_val_frac`) before testing; both modes are YAML‑switchable.
 - Determinism: strict seeding for Python/NumPy/Torch, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG` for GEMM determinism, and per‑worker DataLoader seeding. A determinism banner is printed and persisted into reports.
 - Provenance: reports include the exact model class (e.g., `braindecode.models.EEGNeX`), library versions (torch, numpy, sklearn, mne, braindecode, captum, optuna, python), and determinism flags.

### Per‑run artifacts (besides checkpoints/plots)
- `summary_<TASK>_<ENGINE>.json`: metrics + hyper + determinism + lib versions + model class + hardware
- `resolved_config.yaml`: the frozen config used for the run
- `splits_indices.json`: exact outer/inner indices and subjects for all folds (auditable splits)
- `learning_curves_inner.csv`: all inner‑fold learning curves across outer folds (epoch‑wise train/val loss/acc/macro‑F1)
- `outer_eval_metrics.csv`: one row per outer fold with held‑out subjects, n_test, acc, macro‑F1 (+ OVERALL row)
- `test_predictions.csv`: one row per out‑of‑fold test trial (subject_id, trial_index, true/pred labels, p_trueclass, logp_trueclass, full probs) — ready for mixed‑effects models
- `pip_freeze.txt` and (if available) `conda_env.yml`: environment freeze for reproducibility

### Run directory structure
- Run root contains text/JSON/CSV/HTML/PDF summaries and tables
- Subdirectories:
  - `plots/` — all PNG plots generated during the run (per‑fold confusion and curves; overall confusion)
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
- `scripts/optuna_search.py`: Unified Optuna driver for Step 1/2/3 sweeps (`--stage step1|step2|step3`) over a search space YAML; prunes per‑epoch on inner macro‑F1 and seeds the TPE sampler from config.
- `scripts/run_xai_analysis.py`: Post‑hoc per‑fold attributions and consolidated XAI HTML for a completed run directory.
- `scripts/prepare_from_happe.py`: One‑time materialization of per‑subject `.fif` epochs with aligned behavior and montage from HAPPE/EEGLAB `.set`.

## Repository layout
- code/
  - preprocessing/mne_pipeline.py — spatial sampling (Cz ring), time cropping, channel alignment helpers
  - datasets.py — materialized `.fif` loader, channel unification
  - training_runner.py — GroupKFold (outer) or LOSO; strict inner subject‑aware K‑fold; plots and summaries
  - model_builders.py — EEGNeX builder and train‑time augmentations
- engines/eeg.py — engine wrapper for raw‑EEG models (EEGNeX)
- tasks/ — label functions per task
- configs/
  - common.yaml — global defaults (training, crop_ms, channel lists)
  - tasks/<task>/
    - base.yaml — base hyper‑parameters
    - step1_search.yaml — controller for Step 1 (folds, epochs, etc.)
    - step1_space_deep_spatial.yaml — Step 1 space (dataset choice, core model, spatial window, big levers)
    - step2_search.yaml — controller for Step 2
    - step2_space_deep_spatial.yaml — Step 2 space (refinement/stability knobs)
    - step3_search.yaml — controller for Step 3 (augmentations)
    - step3_space_aug.yaml — Step 3 space (train‑time augmentations)
- scripts/
  - prepare_from_happe.py — convert HAPPE EEGLAB `.set` to per‑subject `.fif` with metadata + montage
  - optuna_search.py — Unified Optuna TPE driver (Stage 1/2/3 via --stage)
  - search_finalist.py — optional finalist tuner (joint sensitive params)
  - final_eval.py — multi‑seed final evaluation and consolidated reporting
  - run_xai_analysis.py — per‑fold attributions and consolidated XAI HTML/PDF
  - run_posthoc_stats.py — post‑hoc statistics (group efficacy and subject reliability)
- results/optuna/
  - refresh_optuna_summaries.py — per‑study CSV (sorted by inner_mean_macro_f1 desc) and plots
  - refresh_optuna_summaries.bat — convenience wrapper on Windows
  - Parallel plots: PNG plus SVG/PDF exports; Matplotlib PNG fallback with thicker lines for readability
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
  --space configs/tasks/cardinality_1_3/step1_space_deep_spatial.yaml `
  --trials 48

# Step 2 (refinement/stability)
python -X utf8 -u scripts/optuna_search.py `
  --stage step2 `
  --task cardinality_1_3 `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --cfg   configs/tasks/cardinality_1_3/step2_search.yaml `
  --space configs/tasks/cardinality_1_3/step2_space_deep_spatial.yaml `
  --trials 48

# Step 3 (augmentations; optional)
python -X utf8 -u scripts/optuna_search.py `
  --stage step3 `
  --task cardinality_1_3 `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --cfg   configs/tasks/cardinality_1_3/step3_search.yaml `
  --space configs/tasks/cardinality_1_3/step3_space_aug.yaml `
  --trials 48
```

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

## Preprocessing details
- HAPPE produces cleaned `.set` per subject. `prepare_from_happe.py` aligns behavior, removes `Condition==99`, encodes labels, attaches montage, and saves per‑subject `.fif`.
- Default at train time: `use_channel_list: non_scalp`. The Cz‑ring (`cz_step`) heuristic is disabled by default (can be re‑enabled by adding it to YAML). You can still set `crop_ms` and/or explicit `include_channels`.

## Reproducibility
- Outer split: `GroupKFold` when `n_folds` is set; otherwise `LOSO`.
- Strict inner subject‑aware K‑fold: set `inner_n_folds` (≥2). If infeasible (too few unique subjects), the run raises with a clear error.
- Determinism: `PYTHONHASHSEED`, Python/NumPy/Torch seeds, per‑worker DataLoader seeding, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`, and `CUBLAS_WORKSPACE_CONFIG` to stabilize CUDA GEMM. Determinism banner is printed and persisted.
- Provenance: model class path, library versions, and determinism flags are included in the TXT/HTML report and JSON.
- Seeds: set a single `seed: 42` or a list `seeds: [41, 42, ...]` (multi‑seed loop). Run directories include `seed_<N>` in the name; a cross‑seed aggregate JSON is written.
- Optuna: TPE sampler is seeded from config; pruning signal is inner macro‑F1; objective is inner‑CV mean macro‑F1.
- Audit artifacts: `splits_indices.json`, `learning_curves_inner.csv`, `outer_eval_metrics.csv`, and `test_predictions.csv` capture splits, pruning traces, outer‑fold metrics, and per‑trial out‑of‑fold predictions (ready for mixed‑effects models).
- XAI checkpoint resolution order per fold: `fold_XX_refit_best.ckpt` → `fold_XX_best.ckpt` → first `fold_XX_inner_YY_best.ckpt`.

Tip: To force LOSO in any run that has a resolved config with `n_folds`, either set `n_folds: null` inside the YAML, or pass `--set n_folds=null` on the CLI.

### Report PDF export (optional)
- HTML reports are always written. To enable PDF export via Playwright:
  - Ensure `playwright` is installed (listed in `environment.yml` under pip)
  - Then run: `playwright install`

### Configuration knobs
- `configs/tasks/<task>/base.yaml`:
  - `n_folds`: number of outer folds (uses GroupKFold). If omitted, uses LOSO.
  - `inner_n_folds`: number of inner folds (must be ≥2). Strictly enforced; insufficient subjects → error.
  - `epochs`, `early_stop`, `batch_size`, `lr`, etc. apply to each inner fold training.

## Commands cheat‑sheet
- Convert: `python scripts/prepare_from_happe.py`
- Search (unified): `python scripts/optuna_search.py --stage step1|step2|step3 ...`
- Final eval: `python scripts/final_eval.py ...`
- Refresh Optuna summaries: `results\optuna\refresh_optuna_summaries.bat`
 - Run XAI on a completed run directory:
```powershell
python -X utf8 -u scripts/run_xai_analysis.py --run-dir "results\runs\<run_dir_name>"
```

### XAI (IG, Grad‑CAM heatmaps, Grad‑TopoCAM)
- Outputs are written under `run_dir/xai_analysis/`:
  - `integrated_gradients/`: per‑fold IG arrays (`fold_XX_xai_attributions.npy`), heatmaps (`.png`), and summaries (`.json`).
  - `gradcam_heatmaps/`: per‑fold Grad‑CAM heatmaps aligned as `[channels × time]` when the target conv preserves channels.
  - `gradcam_topomaps/`: per‑fold Grad‑TopoCAM vectors (`fold_XX_gradcam_topomap.npy`) and three topomap variants:
    - default smooth: `fold_XX_gradcam_topomap.png`
    - contours (paper‑friendly): `fold_XX_gradcam_topomap_contours.png`
    - sensors/nearest (sanity‑check): `fold_XX_gradcam_topomap_sensors.png`
  - Grand averages:
    - IG: `grand_average_xai_attributions.npy`, `grand_average_xai_heatmap.png`, `grand_average_xai_topoplot.png`
    - Grad‑TopoCAM: `grand_average_gradcam_topomap.npy` + three PNGs
  - `consolidated_xai_report.html`: a light HTML gallery focusing on IG grand average and per‑fold heatmaps.

- Choosing the conv layer for Grad‑CAM/TopoCAM:
  - If heatmaps show vertical stripes only (vary by time, not by channel), choose an earlier conv that still preserves `[channels × time]`.
  - You can pass `--target-layer` (e.g., `features.3`) or set a default in `configs/xai_defaults.yaml`.

- Time window for Grad‑TopoCAM:
  - Pass `--gradtopo-window start_ms,end_ms` (e.g., `150,250`) to integrate over a latency window before projecting to the scalp; omit for full window.
  - Defaults can be placed in `configs/xai_defaults.yaml`.

#### XAI defaults YAML
Create `configs/xai_defaults.yaml` to document team defaults (used when the corresponding CLI flags are omitted):
```yaml
xai:
  gradtopo_window_ms: [150, 250]
  gradcam_target_layer: features.3
  xai_top_k_channels: 10
```

`xai_top_k_channels` can also be set in your training config; the training config takes precedence over the YAML defaults.

#### Example commands
```powershell
# Use defaults from configs/xai_defaults.yaml
python -X utf8 -u scripts/run_xai_analysis.py `
  --run-dir "D:\eeg_nn\results\runs\20250927_1122_cardinality_1_3_eeg_hpf_1.5_lpf_35_baseline-off_seed_42_crop_ms_0_496"

# Override the layer and window on CLI
python -X utf8 -u scripts/run_xai_analysis.py `
  --run-dir "D:\eeg_nn\results\runs\20250927_1122_cardinality_1_3_eeg_hpf_1.5_lpf_35_baseline-off_seed_42_crop_ms_0_496" `
  --target-layer "features.3" `
  --gradtopo-window 150,250
```

### Post‑hoc statistics (group efficacy and subject reliability)
- Purpose: quantify population‑level efficacy and subject‑level reliability without retraining.
- Inputs: uses `outer_eval_metrics.csv`, `test_predictions.csv`, and permutation summary (if present) from the run directory.
- Run:
```powershell
python -X utf8 -u scripts/run_posthoc_stats.py `
  --run-dir results\runs\<run_dir_name> `
  --alpha 0.05 `
  --multitest fdr `
  --glmm
```
- Outputs (written to the run directory):
  - `group_stats.json`: mean and 95% CI for accuracy/macro‑F1; permutation p‑values if available.
  - `per_subject_significance.csv`: per‑subject accuracy, binomial p‑value vs chance, adjusted p‑value, and above‑chance flag.
  - `per_subject_summary.json`: number and proportion of subjects above chance.
  - `glmm_summary.json`: optional population fixed‑effect summary (cluster‑robust logit fallback), including CI/p‑value.

## update github example: 
PS D:\eeg_nn> conda activate eegnex-env
(eegnex-env) PS D:\eeg_nn> git add .
warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'code/training_runner.py', LF will be replaced by CRLF the next time Git touches it
(eegnex-env) PS D:\eeg_nn> git status
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.md
        modified:   code/training_runner.py
        modified:   configs/common.yaml
        modified:   configs/tasks/cardinality_1_3/base.yaml
        new file:   configs/tasks/cardinality_1_3/resolved_config_20250924_062508_cardinality_1_3_eeg_step1_t142.yaml
        new file:   configs/tasks/cardinality_1_3/step1.5_space_deep_spatial.yaml
        modified:   configs/tasks/cardinality_1_3/step1_space_deep_spatial.yaml

(eegnex-env) PS D:\eeg_nn> git commit -m "Add resolved config and update search space"
[main 9627fa9] Add resolved config and update search space
 7 files changed, 286 insertions(+), 123 deletions(-)
 create mode 100644 configs/tasks/cardinality_1_3/resolved_config_20250924_062508_cardinality_1_3_eeg_step1_t142.yaml
 create mode 100644 configs/tasks/cardinality_1_3/step1.5_space_deep_spatial.yaml
(eegnex-env) PS D:\eeg_nn> git push origin main
Enumerating objects: 22, done.
Counting objects: 100% (22/22), done.
Delta compression using up to 32 threads
Compressing objects: 100% (13/13), done.
Writing objects: 100% (13/13), 4.19 KiB | 715.00 KiB/s, done.
Total 13 (delta 9), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (9/9), completed with 8 local objects.
To https://github.com/yurigushiken/eeg_nn.git
   5899feb..9627fa9  main -> main
(eegnex-env) PS D:\eeg_nn> 
