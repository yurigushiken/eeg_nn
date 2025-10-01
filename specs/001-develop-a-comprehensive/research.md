# Phase 0 Research Notes

## Decision: Standardize data finalization flow before training
- **Rationale**: Aligns with FR-001–FR-005 by ensuring ingestion, alignment, montage attachment, and .fif export happen deterministically before any modeling work. Centralizing this logic lets `D:\eeg_nn\code\datasets.py` assume analysis-ready inputs, minimizing downstream complexity.
- **Alternatives considered**: Embedding HAPPE cleanup or trial alignment inside the training loop was rejected because the specification scopes data cleaning out and the constitution forbids mutating source archives.

## Decision: Extend nested validation with explicit safeguards
- **Rationale**: FR-007, FR-013–FR-020, and Constitution IV demand strict group-aware inner/outer loops plus audit trails. Enhancing `D:\eeg_nn\code\training_runner.py` to persist split indices, enforce subject-count thresholds, and surface leakage errors keeps evaluation auditable and deterministic.
- **Alternatives considered**: Delegating split validation to external scripts risks drift between HPO and final evaluation and weakens reproducibility guarantees.

## Decision: Stage HPO via Optuna studies orchestrated from configurations
- **Rationale**: FR-008–FR-012 require sequential sampling spaces. Reusing the existing Optuna integration in `training_runner.py` and layering configuration-driven study stages (instead of new tooling) keeps changes minimal while satisfying Constitution III on deterministic search.
- **Alternatives considered**: Introducing a new HPO service (e.g., Ray Tune) would violate Simplicity Principle and add redundant infrastructure.

## Decision: Instrument runtime, observability, and artifact retention
- **Rationale**: FR-015–FR-021 and Constitution V expect human-readable artifacts, logging, and environment snapshots. Plan is to expand existing run directory contents (metrics CSVs, GLMM outputs, permutation logs) without altering storage layout, preserving compatibility with `D:\eeg_nn\scripts\run_posthoc_stats.py`.
- **Alternatives considered**: Writing a separate reporting service was rejected as it fragments provenance and duplicates content already handled by the run directory conventions.

## Decision: Support class imbalance handling through configurable losses and metrics
- **Rationale**: FR-018, FR-033, and clarifications mandate class-weighted losses and subject-level reliability checks. Extending loss construction and metrics aggregation inside `training_runner.py` and reusing existing utilities avoids model-specific forks.
- **Alternatives considered**: Generating synthetic examples (SMOTE-style) was dismissed because it conflicts with reproducibility and could mask trial scarcity issues that the specification wants surfaced explicitly.

## Decision: Enforce clarified subject-level trial thresholds
- **Rationale**: Updated FR-007 sets the minimum to 5 trials per class per subject, which aligns with clarified requirements and reduces unnecessary exclusions while preserving statistical reliability. `datasets.py` must validate counts prior to training and provide exclusion logs for transparency.
- **Alternatives considered**: Retaining the previous 10-trial minimum was rejected because it conflicts with the clarified requirement and would drop too many subjects in smaller datasets.
