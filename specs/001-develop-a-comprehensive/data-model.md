# Data Model: EEG-Decode-Pipeline

## Subject
- **Fields**: `subject_id` (int), `prepared_fif_path` (str), `behavior_csv_path` (str), `montage_name` (str), `num_trials` (int), `class_counts` (dict[str,int]), `channel_list` (list[str]), `excluded_reason` (optional[str])
- **Constraints**: `num_trials` ≥ total minimum; `class_counts[label]` ≥ 5 for every class (otherwise `excluded_reason` records the violation)
- **Relationships**: Owns multiple `Trial` entries; contributes to `DatasetVariant`

## Trial
- **Fields**: `subject_id` (int), `trial_index` (int), `epoch_tensor` (torch.Tensor), `label` (str), `condition_code` (int), `timestamp_ms` (float), `quality_flag` (str)
- **Constraints**: One-to-one mapping between EEG epoch and behavioral row; condition code drawn from `ConditionCodebook`
- **Relationships**: Belongs to `Subject`, aggregated by `FoldResult`

## ConditionCodebook
- **Fields**: `label` (str), `code` (int), `description` (str), `global_order` (int)
- **Constraints**: Codes unique across datasets; stored with run artifacts
- **Relationships**: Referenced by `Trial`, `RunConfig`

## DatasetVariant
- **Fields**: `name` (str), `materialized_dir` (str), `channel_intersection` (list[str]), `times_ms` (list[float]), `sfreq` (float)
- **Constraints**: Derived from `prepare_from_happe.py`; channels are the intersection across subjects
- **Relationships**: Contains `Subject` data; referenced by `RunConfig`

## RunConfig
- **Fields**: `task` (str), `engine` (str), `seed` (int), `stage` (str), `cfg` (dict), `resolved_yaml` (Path), `optuna_study` (str), `runtime_budget_minutes` (float), `glmm_enabled` (bool), `permutation_cfg` (dict)
- **Constraints**: Stage ∈ {"stage_1", "stage_2", "stage_3", "final_loso"}; validated before execution
- **Relationships**: Generates `HPOStage` results; produces `FoldResult` and `StatisticalSummary`

## HPOStage
- **Fields**: `stage_name` (str), `search_space` (dict), `n_trials` (int), `best_params` (dict), `best_score` (float), `evidence_path` (str)
- **Constraints**: Sequential dependency on previous stage champion configuration
- **Relationships**: Produced by `RunConfig`, informs next stage or `final_loso`

## FoldResult
- **Fields**: `outer_fold` (int), `inner_fold` (optional[int]), `train_subjects` (list[int]), `test_subjects` (list[int]), `metrics` (dict[str,float]), `confusion_matrix` (list[list[int]]), `class_weights` (dict[str,float]), `split_indices_path` (Path)
- **Constraints**: Subject sets must be disjoint between train and test; metrics recorded per fold; `split_indices_path` references persisted audit file
- **Relationships**: Aggregated into `StatisticalSummary`; stored in run directory

## StatisticalSummary
- **Fields**: `overall_metrics` (dict[str,float]), `confidence_intervals` (dict[str,tuple[float,float]]), `glmm_results` (dict), `permutation_results` (dict)
- **Constraints**: GLMM executed after each experiment; permutation tests only for final champion
- **Relationships**: Derived from `FoldResult`; saved within run artifacts

## Artifact
- **Fields**: `artifact_type` (str), `path` (Path), `hash` (str), `created_at` (datetime), `metadata` (dict)
- **Constraints**: Hash required for reproducibility; path under `results/runs`
- **Relationships**: Linked to `RunConfig`, `HPOStage`, or `StatisticalSummary`

## XAIAttribution
- **Fields**: `fold_id` (int), `subject_id` (int), `attribution_map` (array), `method` (str), `format` (str)
- **Constraints**: Produced per fold/subject when XAI is enabled; stored in HTML/PDF
- **Relationships**: Attached to `Artifact`; summarized in reporting
