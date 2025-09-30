<!--
Sync Impact Report
- Version change: n/a -> 1.0.0
- Modified principles: initial publication (all principles new)
- Added sections: Data & Artifact Governance; Development Workflow & Review
- Removed sections: none
- Templates:
  [updated] .specify/templates/plan-template.md - version reference aligned
  [reviewed] .specify/templates/spec-template.md - no changes required
  [reviewed] .specify/templates/tasks-template.md - no changes required
- Follow-up TODOs: none
-->
# Numbers Cognition EEG Pipeline Constitution

## Core Principles

### I. Reproducible Experiment Design
Every experiment MUST be reproducible from a version-controlled configuration and documented invocation. Canonical CLI commands in `README.md` and `scripts/` MUST be the only entry point; ad-hoc flags require inline justification committed with the run.
- Every run MUST emit `resolved_config.yaml`, `pip_freeze.txt`, CLI arguments, and Git metadata into its run directory before training starts.
- Configurations under `configs/` MUST remain immutable once a run is recorded; updates require a new file and changelog entry referencing the prior config.
- Researchers MUST record the random seed set, Optuna study name, and hardware banner exactly as printed by the pipeline.

### II. Data Provenance & Integrity
All data transformations MUST be deterministic, auditable, and reversible to the original HAPPE output.
- Source `.set` files, behavior CSVs, and montage definitions MUST remain read-only; conversion to `.fif` MUST occur via `scripts/prepare_from_happe.py` with command history logged.
- Any exclusion or recoding of trials MUST be captured in a machine-readable diff stored alongside the affected dataset.
- Behavior alignment errors MUST abort the run; fixes require documented rationale and reproducible scripts committed to `scripts/` or `utils/`.

### III. Deterministic Training & Search
Search and training procedures MUST preserve determinism so that repeated executions reproduce metrics bit-for-bit when hardware permits it.
- Python, NumPy, Torch, and DataLoader seeds MUST be set from a single canonical seed declared in the run configuration.
- `torch.use_deterministic_algorithms(True)` and required environment variables (for example `CUBLAS_WORKSPACE_CONFIG`) MUST remain enabled unless upstream libraries provide a documented deterministic alternative.
- Optuna trials MUST persist pruned and completed states; reruns MUST reuse the same study storage or export to JSON for archival.

### IV. Rigorous Validation & Reporting
Evaluation MUST reflect subject-aware cross-validation and multi-seed aggregation to prevent optimistic bias.
- Default workflows MUST use GroupKFold or LOSO as defined in the task config; deviations require explicit statistical justification.
- Aggregated reports MUST include per-fold metrics, macro-F1, confusion matrices, and variance across seeds; omission is non-compliant.
- Permutation tests, when executed, MUST document seeds, number of permutations, and reuse the original splits to maintain paired comparisons.

### V. Audit-Ready Artifact Retention
Every scientific claim MUST be backed by accessible artifacts that allow third-party replication without internal knowledge.
- Run directories under `results/` MUST retain checkpoints, plots, split indices, predictions CSVs, and XAI outputs; clean-up scripts MUST never delete these without an archived copy.
- Environment snapshots (`environment.yml` or `pip_freeze.txt`) and resolved configs MUST be referenced in publications or lab notebooks.
- Summary JSON files MUST enumerate library versions, model class, and determinism flags; manual edits are forbidden.

## Data & Artifact Governance
- Raw, intermediate, and derived datasets MUST live in dedicated folders (`data_input_from_happe/`, `data_preprocessed/`, `results/`) with read/write permissions managed via the lab's SOP.
- Any manual notebook exploration MUST write outputs to `reference_files/` with a README describing purpose, input commit hashes, and reproducibility steps.
- External collaborators MUST receive redacted packages containing only the artifacts needed to reproduce published results, plus a copy of this constitution.
- Backups MUST be scheduled for all run artifacts and configuration files; verification logs MUST be retained for at least five years.

## Development Workflow & Review
- Code changes MUST include tests or deterministic run scripts that demonstrate compliance with Core Principles; reviewers MUST reject changes lacking reproducibility evidence.
- Pull requests MUST document affected configs, datasets, and expected impact on metrics; reviewers MUST compare against the baseline artifacts in `results/`.
- Continuous integration MUST execute linting, unit tests, and at least one dry-run configuration validation to ensure determinism flags remain enabled.
- Every merged change MUST append a lab-note entry summarizing rationale, exact commands, and artifact locations.

## Governance
- This constitution supersedes ad-hoc lab practices. Non-compliant work MUST NOT be merged, published, or shared.
- Amendments require consensus from the project leads, documentation of the rationale, and an updated version string with semantic versioning.
- MINOR increments add or materially expand principles or sections; PATCH increments clarify wording without changing obligations; MAJOR increments remove or significantly redefine principles.
- Compliance reviews occur before each publication or dataset release; findings MUST be logged with remediation owners and deadlines.
- Archive copies of every ratified version MUST remain in version control for traceability.

**Version**: 1.0.0 | **Ratified**: 2025-09-30 | **Last Amended**: 2025-09-30




