# Quickstart: EEG-Decode-Pipeline

## Prerequisites
- Activate the Conda environment: `conda activate eegnex-env`
- Ensure preprocessed subject epochs and behavioral CSVs reside under `D:\eeg_nn\data_input_from_happe`
- Materialize analysis-ready `.fif` files using:
  - `python D:\eeg_nn\scripts\prepare_from_happe.py --input-root D:\eeg_nn\data_input_from_happe --output-root D:\eeg_nn\data_preprocessed --montage standard_1005`
- Verify at least 10 subjects with ≥5 trials per class remain after materialization

## Stage 1: Foundational Hyperparameter Search (manual stage progression)
```powershell
python -X utf8 -u scripts/optuna_search.py `
  --stage step1 `
  --task cardinality_1_3 `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --cfg   configs/tasks/cardinality_1_3/step1_search.yaml `
  --space configs/tasks/cardinality_1_3/step1_space_deep_spatial.yaml `
  --trials 48
```
- Output: `results/optuna/<study>/...` per-trial artifacts, plus per-trial summaries under the study; a best.json persisted under `results/optuna_studies/<task>/step1/`

## Stage 2: Architectural Refinement (manual handoff)
```powershell
python -X utf8 -u scripts/optuna_search.py `
  --stage step2 `
  --task cardinality_1_3 `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --cfg   configs/tasks/cardinality_1_3/step2_search.yaml `
  --space configs/tasks/cardinality_1_3/step2_space_deep_spatial.yaml `
  --trials 48
```
- Use step1 champion params (best.json) as input overlay when authoring the step2 search controller.

## Stage 3: Augmentation + Optimizer Sweep (optional)
```powershell
python -X utf8 -u scripts/optuna_search.py `
  --stage step3 `
  --task cardinality_1_3 `
  --base  configs/tasks/cardinality_1_3/base.yaml `
  --cfg   configs/tasks/cardinality_1_3/step3_search.yaml `
  --space configs/tasks/cardinality_1_3/step3_space_aug.yaml `
  --trials 48
```

## Final Evaluation: LOSO Champion Refit (multi-seed supported)
```powershell
python -X utf8 -u scripts/final_eval.py `
  --task cardinality_1_3 `
  --cfg  configs/tasks/cardinality_1_3/base.yaml `
  --seeds 3
```
- Produces per-seed run directories with LOSO folds, aggregates mean/std across seeds, and writes audit artifacts: `splits_indices.json`, `outer_eval_metrics.csv`, predictions CSVs, and logs/runtime.jsonl.

## Post-hoc Statistics & XAI
```powershell
python -X utf8 -u scripts/run_posthoc_stats.py --run-dir results\runs\<run_dir>
python -X utf8 -u scripts/run_xai_analysis.py   --run-dir results\runs\<run_dir>
```
- Outputs under `stats/` and `xai_analysis/` inside the run directory.

## Refresh Optuna Evidence (CSV/plots/index)
```powershell
python -X utf8 -u scripts/refresh_optuna_summaries.py --results-root results\optuna --rebuild-index
```
- Rebuilds per-study CSV/plots and `results/optuna/optuna_runs_index.csv`.

## Verification Checklist
- ✅ All stages completed with deterministic seeds recorded in `resolved_config.yaml`
- ✅ LOSO results across seeds show consistent accuracy and macro-F1
- ✅ GLMM and permutation outputs archived under `results/runs/<final_timestamp>/stats`
- ✅ XAI reports stored under `results/runs/<final_timestamp>/xai_analysis`
- ✅ JSONL runtime log present at `logs/runtime.jsonl` inside each run directory
