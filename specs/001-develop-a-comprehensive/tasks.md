# Tasks: EEG-Decode-Pipeline

**Input**: Design documents from `D:\eeg_nn\specs\001-develop-a-comprehensive\`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/

## Phase 3.1: Setup
- [ ] T001 Create test scaffolding under `D:\eeg_nn\tests\` (contract/, unit/, integration/, utils/) and add `__init__.py` placeholders.
- [ ] T002 Establish deterministic EEG fixtures and shared factories in `D:\eeg_nn\tests\conftest.py` to exercise 5-trial thresholds and subject metadata.
- [ ] T003 [P] Configure `D:\eeg_nn\pytest.ini` with markers for `hpo_stage`, `loso_final`, and seed controls to support staged integration suites.

## Phase 3.2: Tests First (TDD)
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T004 [P] Create contract test `D:\eeg_nn\tests\contract\test_training_runner_config.py` validating `contracts/training_runner_config.yaml` schema (min subjects, 5-trial rule, artifact checks).
- [ ] T005 [P] Add unit test `D:\eeg_nn\tests\unit\datasets\test_trial_threshold.py` ensuring subjects with <5 trials per class are excluded and logged.
- [ ] T006 [P] Add unit test `D:\eeg_nn\tests\unit\datasets\test_channel_alignment.py` verifying channel intersection metadata is persisted and referenced.
- [ ] T007 [P] Add unit test `D:\eeg_nn\tests\unit\training\test_subject_leakage.py` asserting `training_runner.py` blocks subject leakage and enforces ≥10-subject LOSO preflight.
- [ ] T008 [P] Add unit test `D:\eeg_nn\tests\unit\training\test_artifact_outputs.py` covering runtime telemetry, class-weight dumps, split index exports, and JSONL runtime logging per `contracts/logging_contract.yaml` (console output remains as-is; JSONL is required by Option B).
- [ ] T009 [P] Add unit test `D:\eeg_nn\tests\unit\models\test_eegnex_builder.py` confirming opt-in augmentation parameters map to the EEGNeX builder with stable defaults.
- [ ] T010 [P] Add contract/CLI test `D:\eeg_nn\tests\contract\test_optuna_refresh.py` that invokes the master HPO refresh entry point (`scripts/refresh_optuna_summaries.py` / `results/optuna/refresh_all_studies.*`) and asserts regenerated CSVs/plots/logs against `contracts/optuna_refresh_contract.yaml`; verify `results/optuna/optuna_runs_index.csv` rebuild.
- [ ] T011 [P] Add integration test `D:\eeg_nn\tests\integration\test_data_finalization.py` validating data finalization pipeline aligns trials, builds codebook, and writes exclusion report.
- [ ] T012 [P] Add integration test `D:\eeg_nn\tests\integration\test_hpo_stages.py` asserting sequential Optuna stages consume prior champions and emit evidence bundles; also validates outer-loop strategy: with `n_folds=k` use GroupKFold (k splits), without `n_folds` use LOSO (splits equal unique subjects).
- [ ] T013 [P] Add integration test `D:\eeg_nn\tests\integration\test_final_loso_outputs.py` ensuring LOSO refits generate required artifacts (metrics CSVs, GLMM inputs, permutation placeholders).
- [ ] T013a [P] Add unit test `D:\eeg_nn\tests\unit\training\test_naming_conventions.py` verifying zero-padded/snake_case naming for subjects, folds, and artifact files (FR-018).
- [ ] T013b [P] Add integration test `D:\eeg_nn\tests\integration\test_auto_posthoc.py` asserting that train.py automatically triggers post-hoc analysis and generates consolidated HTML reports (FR-021) when `stats.run_posthoc_after_train: true`.
- [ ] T013c [P] Add unit test `D:\eeg_nn\tests\unit\metrics\test_chance_level.py` confirming chance level is computed correctly based on class count and logged (FR-025).
- [ ] T013d [P] Add integration test `D:\eeg_nn\tests\integration\test_visualization_outputs.py` validating required visualizations are generated: overall/per-subject confusion matrices, forest/caterpillar plots, XAI reports (FR-026).

## Phase 3.3: Core Implementation (ONLY after tests fail)
- [ ] T014 Implement 5-trial minimum enforcement, exclusion logging, and metadata exports in `D:\eeg_nn\code\datasets.py`.
- [ ] T015 Extend `D:\eeg_nn\code\training_runner.py` to orchestrate sequential Optuna stages and persist best-configuration evidence.
- [ ] T016 Harden `D:\eeg_nn\code\training_runner.py` leakage guards, subject-count checks (error if unique subjects < 10), and runtime budget enforcement with explicit errors.
- [ ] T017 Enrich `D:\eeg_nn\code\training_runner.py` artifact writes (class_weights/, split_indices.json, predictions CSVs, telemetry logs, GLMM/permutation toggles). Add lightweight JSONL logging per `contracts/logging_contract.yaml` while preserving console output (Option B).
- [ ] T018 [P] Update `D:\eeg_nn\code\model_builders.py` to expose clarified augmentation parameters while keeping legacy defaults.
- [ ] T019 Validate and align Optuna refresh workflow to `D:\eeg_nn\specs\001-develop-a-comprehensive\contracts\optuna_refresh_contract.yaml` (reuse existing `optuna_tools/`); add minimal gaps only if needed (e.g., `refresh_run.log`, optional `COMPLETED.ok`).
- [ ] T019a Implement zero-padded/snake_case naming conventions in `D:\eeg_nn\code\training_runner.py` and `D:\eeg_nn\utils\summary.py` for all artifact identifiers (FR-018).
- [ ] T019b Extend `D:\eeg_nn\train.py` to automatically invoke post-hoc statistical analysis after model execution, generating consolidated HTML reports (FR-021).
- [ ] T019c Implement chance-level computation in `D:\eeg_nn\utils\summary.py` (or metrics utility) based on class count, with logging (FR-025).
- [ ] T019d Enhance `D:\eeg_nn\scripts\run_posthoc_stats.py` and `D:\eeg_nn\scripts\run_xai_analysis.py` to ensure all required visualizations are generated: confusion matrices, forest/caterpillar plots, XAI reports (FR-026).

## Phase 3.4: Integration & Orchestration
- [ ] T020 Align `D:\eeg_nn\scripts\optuna_search.py` with staged champion handoff, study naming, and runtime budget propagation.
- [ ] T021 Validate `D:\eeg_nn\scripts\final_eval.py` multi-seed LOSO/GroupKFold behavior per config and refine aggregate JSON only if needed.
- [ ] T022 [P] Validate `D:\eeg_nn\scripts\run_posthoc_stats.py` to honor new artifact paths and ensure GLMM results and required visuals persist under `stats/`; add minimal gaps only if tests show missing outputs.
- [ ] T023 [P] Enhance `D:\eeg_nn\utils\summary.py` to capture exclusion summaries, runtime telemetry, configuration hashes, chance-level in summary, and refresh outputs in run reports (align with contracts).

## Phase 3.5: Polish & Documentation
- [ ] T024 [P] Refresh `D:\eeg_nn\specs\001-develop-a-comprehensive\quickstart.md` with new CLI flags or artifact descriptions introduced during implementation.
- [ ] T025 [P] Update `D:\eeg_nn\README.md` to document staged HPO workflow, LOSO guarantees, and the Optuna refresh master command with expected outputs.
- [ ] T026 [P] Record changes in `D:\eeg_nn\reference_files\CHANGELOG.md`, noting data-validation, reporting enhancements, and HPO refresh automation.

## Phase 3.6: Validation & Release
- [ ] T027 Run `pytest -m "not slow"` from `D:\eeg_nn` ensuring all new tests pass deterministically.
- [ ] T028 Execute the quickstart workflow (Stages 1–3 + LOSO final) and capture results under `D:\eeg_nn\results\runs` to verify runtime targets.
- [ ] T029 Compile a verification summary in `D:\eeg_nn\results\verification_report.md` covering metrics, artifacts, refresh outputs, and outstanding risks.

## Dependencies
- T001 → T002 → T004–T013d (tests rely on scaffolding and fixtures).
- T004–T013d must complete (and fail initially) before starting T014–T019d.
- T014 precedes T015; T015 precedes T016; T016 precedes T017 (same file sequencing).
- T019 depends on T010; T019a depends on T013a; T019b depends on T013b; T019c depends on T013c; T019d depends on T013d.
- T020 depends on T015–T017; T021 depends on T015–T017; T022 depends on T017 & T019d; T023 depends on T017, T019, T019a, T019c.
- T027 requires completion of all implementation tasks; T028 requires T027; T029 requires T028.

## Parallel Execution Example
```
# After T002 completes, launch these independent test-authoring tasks in parallel:
Task: T004 Create contract test for training_runner_config schema
Task: T005 Add dataset 5-trial exclusion unit test
Task: T006 Add dataset channel metadata persistence test
Task: T007 Add training runner leakage guard unit test
Task: T008 Add training runner artifact logging unit test
Task: T009 Add EEGNeX builder parameter coverage unit test
Task: T010 Add Optuna refresh master script contract test
Task: T011 Add data finalization integration test
Task: T012 Add staged Optuna integration test
Task: T013 Add final LOSO outputs integration test
Task: T013a Add naming conventions validation unit test
Task: T013b Add auto post-hoc integration test
Task: T013c Add chance-level computation unit test
Task: T013d Add visualization outputs integration test
```
