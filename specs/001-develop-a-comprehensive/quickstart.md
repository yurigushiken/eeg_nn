# Quickstart: EEG-Decode-Pipeline

## Prerequisites
- Activate the Conda environment: `conda activate eegnex-env`
- Ensure preprocessed subject epochs and behavioral CSVs reside under `D:\eeg_nn\data_input_from_happe`
- Materialize analysis-ready `.fif` files using:
  - `python D:\eeg_nn\scripts\prepare_from_happe.py --input-root D:\eeg_nn\data_input_from_happe --output-root D:\eeg_nn\data_preprocessed --montage standard_1005`
- Verify at least 10 subjects with ≥5 trials per class remain after materialization

## Stage 1: Foundational Hyperparameter Search
```
python D:\eeg_nn\scripts\optuna_search.py \
  --task cardinality_1_3 \
  --stage stage_1 \
  --config D:\eeg_nn\configs\tasks\cardinality_1_3\step_1_resolved_config-t034_n_loso.yaml \
  --materialized-dir D:\eeg_nn\data_preprocessed\hpf_1.0_lpf_40_baseline-on \
  --study-name eeg_decode_stage1 \
  --n-trials 80
```
- Output: `results/runs/<timestamp>_stage1/` with `resolved_config.yaml`, Optuna study DB, split indices, per-trial predictions, and preliminary metrics

## Stage 2: Architectural Refinement Search
```
python D:\eeg_nn\scripts\optuna_search.py \
  --task cardinality_1_3 \
  --stage stage_2 \
  --base-run results/runs/<stage1_timestamp>_stage1 \
  --study-name eeg_decode_stage2 \
  --n-trials 60
```
- Uses Stage 1 champion configuration as the seed; records refined architectural parameters and updated evidence bundle

## Stage 3: Augmentation + Optimizer Search
```
python D:\eeg_nn\scripts\optuna_search.py \
  --task cardinality_1_3 \
  --stage stage_3 \
  --base-run results/runs/<stage2_timestamp>_stage2 \
  --study-name eeg_decode_stage3 \
  --n-trials 60
```
- Jointly tunes augmentation probabilities and sensitive optimizer settings while logging augmentation ramp parameters

## Final Evaluation: LOSO Champion Refit
```
python D:\eeg_nn\scripts\final_eval.py \
  --task cardinality_1_3 \
  --base-run results/runs/<stage3_timestamp>_stage3 \
  --n-seeds 3 \
  --materialized-dir D:\eeg_nn\data_preprocessed\hpf_1.0_lpf_40_baseline-on \
  --outer-protocol loso
```
- Produces `results/runs/<timestamp>_final_loso_seed_*` folders with per-fold models refit from scratch, aggregated metrics, GLMM outputs, and permutation-ready artifacts

## Post-hoc Statistics & Reporting
```
python D:\eeg_nn\scripts\run_posthoc_stats.py \
  --run-dir results/runs/<final_timestamp>_final_loso_seed_0 \
  --alpha 0.05 --multitest fdr --glmm --forest

python D:\eeg_nn\scripts\run_xai_analysis.py \
  --run-dir results/runs/<final_timestamp>_final_loso_seed_0 \
  --method grad_cam --format html
```
- Generates group-level confidence intervals, GLMM summaries, permutation significance (if requested), subject reliability tables, confusion matrices, forest/caterpillar plots, and XAI interpretability reports ready for publication

## Verification Checklist
- ✅ All stages completed with deterministic seeds recorded in `resolved_config.yaml`
- ✅ LOSO results across seeds show consistent accuracy and macro-F1
- ✅ GLMM and permutation outputs archived under `results/runs/<final_timestamp>/stats`
- ✅ XAI reports stored under `results/runs/<final_timestamp>/xai`
- ✅ Console logs confirm per-epoch metrics, fold timestamps, and runtime budgets
