
# Implementation Plan: EEG-Decode-Pipeline

**Branch**: `[001-develop-a-comprehensive]` | **Date**: 2025-10-01 | **Spec**: D:\eeg_nn\specs\001-develop-a-comprehensive\spec.md
**Input**: Feature specification from `D:\eeg_nn\specs\001-develop-a-comprehensive\spec.md`

## Summary
Update the EEG numerosity decoding pipeline so that `D:\eeg_nn\code\datasets.py`, `D:\eeg_nn\code\training_runner.py`, and `D:\eeg_nn\code\model_builders.py` enforce the clarified subject-level trial minimum (≥5 per class) while delivering staged Optuna HPO, LOSO evaluation, rigorous leakage checks, and audit-ready artifacts required by the refreshed specification.

## Technical Context
**Language/Version**: Python 3.11 (Conda `eegnex-env`)
**Primary Dependencies**: PyTorch, Optuna, MNE, Braindecode, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
**Storage**: Materialized `.fif` epochs per subject, behavior CSVs, run directories under `D:\eeg_nn\results\runs`
**Testing**: pytest with deterministic fixtures covering dataset validation, runner orchestration, and reporting outputs
**Target Platform**: Training on GPU-capable Linux nodes; compatible with Windows development
**Project Type**: Single offline scientific pipeline (CLI driven)
**Performance Goals**: Single-seed LOSO run on 24 subjects ≤2 hours; scalable to ≥100 subjects without redesign
**Constraints**: Deterministic seeds/algorithms, subject-aware splits, ≥10 subjects available, ≥5 trials per class per subject, observability via per-epoch logs, environment snapshotting
**Scale/Scope**: Optuna stages 1–3, LOSO champion refits across multiple seeds, reporting for 24–100 subjects
**Implementation Focus**: Iterate within existing architecture—no new frameworks or divergent patterns

## Constitution Check
- **Principle I (Reproducible Design)**: Plan keeps config layering, resolved config persistence, and documented commands (PASS)
- **Principle II (Data Integrity)**: Trial alignment and subject filtering remain deterministic and auditable (PASS)
- **Principle III (Deterministic Training & Search)**: Optuna integration, seeding, and deterministic DataLoader workers enforced (PASS)
- **Principle IV (Validation & Reporting)**: Nested subject-aware splits, LOSO refits, GLMM/permutation tests, and fold telemetry included (PASS)
- **Principle V (Artifact Retention)**: Run directories capture configs, metrics, split indices, environment freezes, statistics, XAI assets (PASS)

## Project Structure
```
D:\eeg_nn\code\
├── datasets.py
├── model_builders.py
├── training_runner.py
├── preprocessing\epoch_utils.py
└── __init__.py

D:\eeg_nn\engines\
├── __init__.py
└── eeg.py

D:\eeg_nn\scripts\
├── prepare_from_happe.py
├── optuna_search.py
├── run_posthoc_stats.py
├── run_xai_analysis.py
└── analyze_nloso_subject_performance.py

D:\eeg_nn\configs\
├── common.yaml
└── tasks\cardinality_1_3\
    ├── step_1_resolved_config-t034_n_loso.yaml
    └── step_1_5_resolved_config-t022_n_loso.yaml

D:\eeg_nn\results\runs\
```

**Structure Decision**: Single CLI-driven research pipeline rooted in `D:\eeg_nn\code\`, supported by `engines`, `scripts`, and versioned `configs`. Stage progression is manual (via CLI), not runner-internal; no additional subprojects required.

## Phase 0: Outline & Research
- Confirmed clarified thresholds: ≥10 subjects available; ≥5 trials per class per subject (adjust dataset filters and run preflight checks accordingly)
<!-- -CANCEL THIS Revalidated channel intersection policy and audit logging requirements to ensure deterministic preprocessing assumptions align with constitution -->
- Assessed Optuna sequential search design to maintain determinism while spanning multistage search spaces
- Catalogued artifact expectations (GLMM, permutation, confusion matrices, XAI) to ensure run directory schema remains audit-ready
- Document findings and decisions in `D:\eeg_nn\specs\001-develop-a-comprehensive\research.md`

**Output**: `research.md` updated with clarifications and rationale

## Phase 1: Design & Contracts
- **Dataset enforcement (`datasets.py`)**
  - Add explicit validation for ≥5 trials per class per subject; log exclusions with subject ID and counts
  <!-- cancel this!- Persist channel intersection metadata and condition codebook per run; surface mismatches with actionable errors -->
  - Ensure dataset metadata exposes subject counts to guard LOSO prerequisites
- **Training orchestration (`training_runner.py`)
  - Maintain nested subject-aware CV with leakage checks and audit logs of split indices
  - Do not auto-orchestrate sequential Optuna stages; provide only `validate_stage_handoff` to check prior-champion availability. Stage execution is manual using CLI with previous `resolved_config.yaml`.
  - Enforce LOSO refits per seed with runtime monitoring and deterministic behavior (DataLoader seeding, class-weight recomputation)
  - Expand artifact production: per-fold metrics, per-trial predictions, GLMM inputs, permutation metadata, class weights, runtime telemetry
  - Implement zero-padded/snake_case naming conventions for all output identifiers (subjects, folds, files) to satisfy FR-018
  - Compute and log chance level automatically based on class count to satisfy FR-025
  - HPO management: keep refresh entry points per FR-020. No internal chaining of stages; users invoke next stage manually after reviewing artifacts.
  - Outer-loop strategy is config-driven: GroupKFold when `n_folds` is set; otherwise LOSO. Tests will cover both modes (FR-011)
  - Logging (Option B): add a lightweight JSONL runtime log at `<run_dir>/logs/runtime.jsonl` with key events while preserving console output (FR-019)

- **Model builders (`model_builders.py`)
  - Keep EEGNeX builder compatible while exposing clarified architectural params; ensure defaults stable
  - Provide augmentation hooks consistent with clarified Stage 3 requirements (mixup, masking, warmup scaling)

- **Contracts**
  - Update `D:\eeg_nn\specs\001-develop-a-comprehensive\contracts\training_runner_config.yaml` with thresholds, augmentation knobs, permutation toggles, and artifact checks
  - Reference existing Optuna refresh stack; align `scripts/refresh_optuna_summaries.py` to tests (CLI alias, index rebuild)
  - Author `contracts/optuna_refresh_contract.yaml` enumerating required CSVs/plots/logs and index behavior used by T010/T019
  - Author `contracts/logging_contract.yaml` specifying runtime JSONL logging (fields/levels/handlers)

- **Data Model**
  - Revise entity definitions to include trial count thresholds, subject exclusion flags, and refined HPO stage artifacts in `data-model.md`

- **Quickstart**
  - Adjust end-to-end runbook to highlight 5-trial filter, sequential Optuna stages, LOSO evaluation, post-hoc stats, and XAI workflows (`quickstart.md`)

- **Testing Strategy**
  - Plan pytest coverage: dataset filters, Optuna stage selection, LOSO leakage guard, GLMM/permutation artifact generation, quickstart validation of runtime goals
  - Add contract/CLI tests asserting that the Optuna refresh master script regenerates summary CSVs, plots, and logs under `results/optuna/`
  - Add validation tests ensuring zero-padded/snake_case naming conventions are enforced across all artifacts (FR-018)
  - Add tests verifying automatic post-hoc execution after train.py runs, producing consolidated HTML reports (FR-021)
  - Add tests confirming chance-level computation and logging based on class count (FR-025)
  - Add tests validating visualization outputs: overall/per-subject confusion matrices, forest plots, caterpillar plots, XAI reports (FR-026)
  - Add tests verifying outer-loop strategy selection: with `n_folds=k` use GroupKFold (k outer splits), without `n_folds` use LOSO (outer splits equal unique subjects) (FR-011)
  - Add tests validating JSONL runtime logging per `contracts/logging_contract.yaml` (FR-019)

- **Reporting & Visualization**
  - Validate `run_posthoc_stats.py` and `run_xai_analysis.py` produce all required visualizations and consolidated reports; add minimal gaps only if tests show missing outputs (FR-026)
  - Ensure `train.py` can automatically trigger post-hoc statistical analysis after each execution (per config)

**Outputs**: updated plan and contracts

## Phase 2: Task Planning Approach
- Base tasks on `.specify\templates\tasks-template.md`
- Derive tasks from Phase 1 artifacts:
  - Each contract rule → contract validation test `[P]`
  - Each entity constraint → dataset/runner implementation task `[P]`
  - User scenarios → integration tests verifying ingestion, staged HPO, LOSO evaluation
  - Reporting requirements → tasks for GLMM, permutation, visualization, XAI artifacts
- Enforce TDD ordering: schema/unit tests → implementation → integration tests → docs
- Respect dependencies: dataset validation before runner logic, runner before reporting
- Flag parallelizable tasks `[P]` (e.g., stats reporting vs augmentation tuning)
- Target 25–30 ordered tasks capturing testing and implementation, including validation of the Optuna study refresh workflow

## Phase 3+: Future Implementation
- **Phase 3**: Generate `tasks.md` via `/tasks`
- **Phase 4**: Execute tasks with deterministic validation and documentation updates
- **Phase 5**: Run pytest, execute quickstart commands, confirm LOSO runtime and artifact completeness

## Complexity Tracking
| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|---------------------------------------|
| None | – | – |

## Progress Tracking
**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented (not required)

---
*Based on Constitution v1.0.0 - See `D:\eeg_nn\.specify\memory\constitution.md`*
