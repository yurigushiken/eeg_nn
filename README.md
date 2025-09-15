conda activate eegnex-env

## Overview
This repository implements a streamlined EEG decoding pipeline aligned with the protocol of Borra et al. (2024/2025), adapted to our Numbers‑Cognition EEG study. We run a two‑step Optuna search and evaluate with LOSO and multi‑seed reporting. The project focuses on four tasks:
- cardinality_1_3
- cardinality_4_6
- landing_digit_1_3_within_small_and_cardinality
- landing_digit_4_6_within_large_and_cardinality

### Key ideas
- On‑the‑fly MNE preprocessing per subject with caching; robust referencing (PREP), optional RANSAC, ICA blink/muscle removal (ICLabel), band‑pass, epoching, downsampling, spatial sampling.
- Step 1 search tunes preprocessing + network + training (augmentation OFF).
- Step 2 search tunes augmentation only on materialized .fif from Step 1 best.
- Final evaluation can average across ≥10 seeds and generate XAI reports.

## Repository layout
- code/
  - preprocessing/mne_pipeline.py — PREP, ICA, epoching, downsampling, spatial sampling
  - datasets.py — builds datasets (on‑the‑fly or materialized .fif), behavior merge, channel unification
  - training_runner.py — LOSO with inner subject‑aware split, plots, summaries
  - model_builders.py — EEGNeX builder and train‑time augmentations
- engines/eeg.py — engine wrapper for raw‑EEG models (EEGNeX)
- tasks/ — label functions per task
- configs/
  - common.yaml — global defaults (ICA, PREP, etc.)
  - tasks/<task>/
    - base.yaml — base hyper‑parameters (includes target_sfreq: 125)
    - step1_search.yaml — controller for Step 1 (folds, epochs, etc.)
    - step1_space.yaml — Step 1 search space (preproc + net + train; no aug)
- scripts/
  - search_step1.py — Step 1 TPE (TPE/ICLabel/LOSO)
  - materialize_from_step1.py — copy cached .fif from Step 1 best into data_preprocessed/
  - search_step2.py — Step 2 TPE (augmentations only, uses materialized .fif)
  - final_eval.py — multi‑seed final evaluation and consolidated reporting
- data directories (git‑ignored):
  - eeg-raw/subjectXX.mff
  - data_behavior/behavior_data/SubjectXX.csv
  - cache/epochs/<fingerprint>/
  - data_preprocessed/<dataset>/

## Data expectations
- Raw EEG: `eeg-raw/subjectXX.mff`
- Behavior: `data_behavior/behavior_data/SubjectXX.csv`
- Montage: `net/AdultAverageNet128_v1.sfp` (kept in repo)

## Typical workflow
1) Step 1 search (preproc + net + train; aug OFF)
```powershell
python scripts/search_step1.py `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/step1_search.yaml `
  --space configs/tasks/cardinality_1_3/step1_space.yaml `
  --trials 48
```

2) Materialize best Step 1 (optional, for Step 2 speed/repro)
```powershell
python scripts/materialize_from_step1.py `
  --task cardinality_1_3 `
  --best results/optuna/<timestamp>_cardinality_1_3/best.json `
  --out  data_preprocessed/cardinality_1_3_step1_best
```

3) Step 2 search (augmentation only)
```powershell
python scripts/search_step2.py `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/step1_search.yaml `
  --space configs/tasks/cardinality_1_3/step2_space.yaml `
  --materialized data_preprocessed/cardinality_1_3_step1_best `
  --trials 48
```

4) Final evaluation (multi‑seed, LOSO)
```powershell
python scripts/final_eval.py `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/base.yaml `
  --seeds 10
```

## Preprocessing details (Step 1)
- PREP robust average reference (`prep_ransac` optional; default off), conservative fallback if PREP fails.
- ICA blink/muscle/ECG removal:
  - `use_ica: true`, `ica_method: picard`, `ica_labeler: iclabel` (threshold 0.9).
- Filters: `f_lo` ~ 0.5–2.0 Hz, `f_hi` ~ 35–50 Hz.
- Epochs: 0 → `t1_s` seconds; downsample to 125 Hz; Cz‑centric spatial sampling via `cz_step`, or explicit channel keep‑list.
- Behavior alignment mirrors legacy scripts (practice removal, Condition handling); optional QC masks.

## Reproducibility
- LOSO outer split; inner subject‑aware validation split
- `seed`, `random_state`, and cache fingerprinting to keep trials consistent.
- Final multi‑seed evaluation recommended (≥10 seeds).

## Commands cheat‑sheet
- Step 1: `python scripts/search_step1.py ...`
- Materialize: `python scripts/materialize_from_step1.py ...`
- Step 2: `python scripts/search_step2.py ...`
- Final eval: `python scripts/final_eval.py ...`