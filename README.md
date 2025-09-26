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

### Entry points (when to use what)
- `train.py`: Single run for a given task/engine using layered configs; ideal for validating a resolved YAML and optionally running XAI on completion (`--run-xai`).
- `scripts/optuna_search.py`: Unified Optuna driver for Step 1/2/3 sweeps (`--stage step1|step2|step3`) over a search space YAML; prunes per-epoch on inner macro‑F1.
- `scripts/final_eval.py`: Multi-seed LOSO/GroupKFold evaluation using a fixed config (optionally merged with best params); aggregates seed metrics.
- `scripts/run_xai_analysis.py`: Post-hoc per-fold attributions and grand-average XAI summary for a completed run directory.
- `scripts/prepare_from_happe.py`: One-time materialization of per-subject `.fif` epochs with aligned behavior and montage from HAPPE/EEGLAB `.set`.

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
- `seed`, `random_state`, and cache fingerprinting to keep trials consistent.
- Final multi‑seed evaluation recommended (≥10 seeds).
- Optuna uses TPE sampling and a Median Pruner (10 warmup epochs); per‑epoch inner‑val macro‑F1 is reported for pruning. The Optuna objective is the averaged inner macro‑F1 across inner folds and outer folds.
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
