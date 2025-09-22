conda activate eegnex-env

## Overview
This repository implements a streamlined EEG decoding pipeline aligned with the protocol of Borra et al. (2024/2025), adapted to our Numbers‑Cognition EEG study. We run a three‑step Optuna search (Step 3 augmentations optional) and evaluate with LOSO and multi‑seed reporting. The project focuses on four tasks:
- cardinality_1_3
- cardinality_4_6
- landing_digit_1_3_within_small_and_cardinality
- landing_digit_4_6_within_large_and_cardinality

### Key ideas
- Preprocessing is performed externally with HAPPE; we convert EEGLAB `.set` to `.fif` once with `scripts/prepare_from_happe.py`.
- Optuna search runs directly on materialized `.fif` via `scripts/optuna_search.py` with a `--stage` flag.
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

2) Optuna search on materialized `.fif` (choose stage)
```powershell
# Step 1 (architecture/spatial/big levers)
python -X utf8 -u scripts/optuna_search.py `
  --stage step1 `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/step1_search.yaml `
  --space configs/tasks/cardinality_1_3/step1_space_deep_spatial.yaml `
  --trials 48

# Step 2 (refinement/stability)
python -X utf8 -u scripts/optuna_search.py `
  --stage step2 `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/step2_search.yaml `
  --space configs/tasks/cardinality_1_3/step2_space_deep_spatial.yaml `
  --trials 48

# Step 3 (augmentations; optional)
python -X utf8 -u scripts/optuna_search.py `
  --stage step3 `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/step3_search.yaml `
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

## Preprocessing details
- HAPPE produces cleaned `.set` per subject. `prepare_from_happe.py` aligns behavior, removes `Condition==99`, encodes labels, attaches montage, and saves per‑subject `.fif`.
- Optional at train time: `crop_ms`, `use_channel_list`, `include_channels`, `cz_step` (Cz‑centric ring on 3D montage).

## Reproducibility
- LOSO outer split; inner subject‑aware validation split
- `seed`, `random_state`, and cache fingerprinting to keep trials consistent.
- Final multi‑seed evaluation recommended (≥10 seeds).

## Commands cheat‑sheet
- Convert: `python scripts/prepare_from_happe.py`
- Search (unified): `python scripts/optuna_search.py --stage step1|step2|step3 ...`
- Final eval: `python scripts/final_eval.py ...`