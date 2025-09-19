conda activate eegnex-env

## Overview
This repository implements a streamlined EEG decoding pipeline aligned with the protocol of Borra et al. (2024/2025), adapted to our Numbers‑Cognition EEG study. We run a two‑step Optuna search and evaluate with LOSO and multi‑seed reporting. The project focuses on four tasks:
- cardinality_1_3
- cardinality_4_6
- landing_digit_1_3_within_small_and_cardinality
- landing_digit_4_6_within_large_and_cardinality

### Key ideas
- Preprocessing is performed externally with HAPPE; we convert EEGLAB `.set` to `.fif` once with `scripts/prepare_from_happe.py`.
- Optuna search runs directly on materialized `.fif` (Step 2 driver).
- Final evaluation can average across ≥10 seeds and generate XAI reports.

## Repository layout
- code/
  - preprocessing/mne_pipeline.py — spatial sampling (Cz ring), time cropping, channel alignment helpers
  - datasets.py — materialized `.fif` loader, channel unification
  - training_runner.py — LOSO with inner subject‑aware split, plots, summaries
  - model_builders.py — EEGNeX builder and train‑time augmentations
- engines/eeg.py — engine wrapper for raw‑EEG models (EEGNeX)
- tasks/ — label functions per task
- configs/
  - common.yaml — global defaults (training, crop_ms, channel lists)
  - tasks/<task>/
    - base.yaml — base hyper‑parameters (includes target_sfreq: 125)
    - step1_search.yaml — controller for Step 1 (folds, epochs, etc.)
    - step1_space.yaml — Step 1 search space (preproc + net + train; no aug)
- scripts/
  - prepare_from_happe.py — convert HAPPE EEGLAB `.set` to per‑subject `.fif` with metadata + montage
  - search_step2.py — Optuna TPE on materialized `.fif` (model/training/aug)
  - final_eval.py — multi‑seed final evaluation and consolidated reporting
- data directories (git‑ignored):
  - eeg-raw/subjectXX.mff
  - data_behavior/behavior_data/SubjectXX.csv
  - cache/epochs/<fingerprint>/
  - data_preprocessed/<dataset>/

## Data expectations
- HAPPE output: `data_input_from_happe/<dataset>/5 - processed .../*.set` and QC CSV in `6 - quality_assessment_outputs ...`
- Behavior: `data_behavior/data_UTF8/SubjectXX.csv`
- Montage: `net/AdultAverageNet128_v1.sfp` (used during conversion to attach 3D positions)

## Typical workflow
1) Convert HAPPE EEGLAB `.set` to `.fif`
```powershell
python scripts/prepare_from_happe.py
```

2) Optuna search on materialized `.fif`
```powershell
python -X utf8 -u scripts/search_step2.py `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/base.yaml `
  --space configs/tasks/cardinality_1_3/space_materialized.yaml `
  --materialized data_preprocessed/hpf_1.0_lpf_45_baseline-on-example `
  --trials 48
```

3) Final evaluation (multi‑seed, LOSO)
```powershell
python scripts/final_eval.py `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/base.yaml `
  --seeds 10
```

## Preprocessing details
- HAPPE produces cleaned `.set` per subject. `prepare_from_happe.py` aligns behavior, removes `Condition==99`, encodes labels, attaches montage, and saves per‑subject `.fif`.
- Optional at train time: `crop_ms`, `use_channel_list`, `include_channels`, `cz_step` (Cz‑centric ring on 3D montage).

## Reproducibility
- LOSO outer split; inner subject‑aware validation split
- `seed`, `random_state`, and cache fingerprinting to keep trials consistent.
- Final multi‑seed evaluation recommended (≥10 seeds).

## Commands cheat‑sheet
- Convert: `python scripts/prepare_from_happe.py`
- Search: `python scripts/search_step2.py ...`
- Final eval: `python scripts/final_eval.py ...`