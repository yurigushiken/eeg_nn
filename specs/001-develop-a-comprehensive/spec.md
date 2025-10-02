# Feature Specification: EEG-Decode-Pipeline

**Feature Branch**: `[001-develop-a-comprehensive]`  
**Created**: 2025-09-30  
**Status**: Draft  
**Input**: User description: "Develop a comprehensive and reproducible (apart from channel order) scientific pipeline for decoding cognitive states from single-trial EEG data. Let's call the project EEG-Decode-Pipeline. The primary user is a computational neuroscience researcher who needs to go from pre-cleaned EEG data to publication-ready statistical results and visualizations, while ensuring the entire process is transparent, auditable, and methodologically rigorous.\n\nThe core scientific task is to classify the cardinal numerosity (e.g., 1 vs. 2 vs. 3) that a participant was viewing based on a short window of their raw EEG signal.\n\nThe pipeline must support the following workflow:\n\n1. Data Finalization and Ingestion:\nThe researcher will provide datasets that have already been preprocessed by an external tool like HAPPE. Each dataset consists of per-subject EEGLAB .set files and corresponding behavioral data in CSV files. The pipeline must have a preliminary step to finalize this data for analysis. This step must:\n\n    Ingest the .set and .csv files.\n\n    Precisely align each EEG trial with its corresponding behavioral and condition labels.\n\n    Enforce a strict one-to-one match, terminating with a clear error if any trial cannot be aligned.\n\n    Create a globally consistent encoding for all experimental conditions across all subjects.\n\n    Attach a standard 3D sensor-location montage to the EEG data.\n\n    Save the fully prepared, analysis-ready data for each subject into a standardized, self-contained format (like MNE-Python's .fif files).\n\n2. Experiment Execution:\nThe researcher must be able to define and execute experiments through configuration files. An experiment consists of a chosen model architecture, task, dataset, and a complete evaluation protocol.\n\n3. Hyperparameter Optimization (HPO):\nThe pipeline must support a systematic, multi-stage hyperparameter optimization search to find the best-performing model configuration. This is necessary because the search space is too large and complex for manual tuning. The HPO process must be structured in sequential stages:\n\n    Stage 1 (Foundational Parameter Search): This initial, broad search must optimize the most impactful components of the pipeline. This includes selecting the optimal preprocessed dataset from a multiverse of candidates, as well as finding the best core architectural dimensions and primary training parameters like learning rate.\n\n    Stage 2 (Architectural Refinement Search): Using the best configuration from Stage 1, this stage must perform a more focused search to fine-tune detailed architectural and regularization hyperparameters.\n\n    Stage 3 (Joint Augmentation and Optimizer Search): Using the best configuration from Stage 2, this final search stage must jointly optimize the parameters of on-the-fly data augmentation techniques alongside the most sensitive and interacting training parameters, such as learning rate and weight decay.\n\n4. Rigorous, Leakage-Free Validation:\nThe entire evaluation framework must be built around a nested, subject-aware cross-validation scheme to produce an unbiased estimate of generalization performance to new individuals and to prevent data leakage.\n\n    The outer loop of the validation must partition subjects, holding out one or more subjects for final testing.\n\n    The inner loop, nested within the outer training set, must be used for all model training and selection. All inner splits must also be subject-aware, ensuring that trials from a single participant are never split between an inner-training and inner-validation set.\n\n5. Final Model Evaluation:\nAfter the HPO process identifies a single champion hyperparameter configuration, the pipeline must support a final, highly rigorous evaluation run. This run must:\n\n    Use a nested Leave-One-Subject-Out (LOSO) protocol.\n\n    For each outer fold, it must refit a single, new model from scratch on the entire outer-training set (all other subjects) using the champion hyperparameters.\n\n    It must support running this entire N-LOSO evaluation multiple times with different random seeds to produce a stable and reliable final performance metric.\n\n6. Comprehensive Reporting and Analysis Artifacts:\nEvery experiment run must produce a set of detailed, auditable artifacts. The system must automatically generate a self-contained run directory that includes:\n\n    A complete record of the final, resolved configuration used for the run.\n\n    Environment snapshots to ensure long-term reproducibility (except for channel order).\n\n    A human-readable text summary of the overall results and fold-by-fold breakdown.\n\n    A comprehensive set of post-hoc statistical analyses, including:\n\n        Group-level performance metrics (mean accuracy, macro-F1) with confidence intervals.\n\n        A formal statistical test of group-level significance, such generalized linear mixed-effects model (GLMM) that accounts for subject variability. A permutation test is the final test of statistical significance once a final set of hyper parameters is identified. \n\n        Subject-level reliability analysis, reporting the proportion of individual subjects whose performance is statistically above chance.\n\n    Visualizations, including:\n\n        An overall confusion matrix aggregated across all test subjects, both inner folds and outer folds. \n\n        Per-subject confusion matrices, both inner folds and outer folds.\n\n        Plots showing the distribution of performance across subjects, such as a forest plot.\n\n        Plots from the statistical models, such as a caterpillar plot showing individual subject effects.\n\n        Model interpretability (XAI) reports, html and pdf, including grand-average and per-fold attribution maps."


## Clarifications

### Session 2025-10-01
- Q: How should the pipeline perform the group-level significance test for final reporting? → A: Run a GLMM with a subject-level random intercept after every experiment and reserve a permutation test for the final champion evaluation once hyperparameters are fixed.
- Q: How should the pipeline enforce minimum data coverage per subject for LOSO runs? → A: Require at least 10 subjects before execution and terminate with a clear error if fewer are available.
- Q: What is the policy for mismatched channel sets across subjects? → A: Use only the intersection of channels common to all subjects in the analysis.
- Q: What are the runtime and scalability targets for the final LOSO evaluation? → A: A single-seed N-LOSO run on 24 subjects must finish within 2 hours on reference hardware, and the design must scale to at least 100 subjects without architectural changes.
- Q: What observability outputs are required during training and evaluation? → A: Provide continuous console output, logging per-epoch metrics and fold completion timestamps.
- Q: What validation strategy should hyperparameter search stages default to? → A: Default to a 6-fold group cross-validation outer loop while allowing LOSO runs without warnings when explicitly requested.
- Q: How should Python dependencies be managed for reproducibility? → A: Use a version-pinned Conda environment defined by environment.yml for all runs.
 - Q: What trial-count threshold applies per subject and class? → A: Require at least 5 trials per class for every subject; exclude noncompliant subjects and log a warning with subject ID and counts.
- Q: What scope boundaries apply regarding real-time decoding and data cleaning? → A: The pipeline is offline/batch only, assumes HAPPE-preprocessed inputs, and does not perform initial data cleaning.
- Q: How should the pipeline handle subjects whose class distributions remain highly imbalanced even after meeting the 5-trial minimum per class? → A: Keep the subject but apply class-weighted loss or metrics adjustments.
 - Q: What is the minimum trials per class threshold that the functional requirements must enforce? → A: Require at least 5 trials per class for every subject; exclude subjects that fall below.


## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors, actions, data, constraints
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```


---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation
When creating this spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - User types and permissions
   - Data retention/deletion policies  
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs


---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a computational neuroscience researcher with pre-cleaned per-subject EEG and behavioral data, I want to run a transparent, auditable pipeline to decode the EEG signals, that ingests my data, prevents leakage through nested subject-aware validation, performs staged hyperparameter optimization to select a champion model, and produces publication-ready results, statistics, and visualizations for single-trial numerosity classification, that can report reliable generalization to new individuals, and that I can use as the basis of my publication.

### Acceptance Scenarios
1. **Given** a set of subjects with pre-cleaned EEG files and corresponding behavioral CSVs, **When** the researcher runs the data finalization step, **Then** the system ingests files, aligns trials to labels with a strict one-to-one mapping, applies a standard sensor montage, creates a global condition codebook across subjects, and saves per-subject analysis-ready files; if any trial cannot be aligned, it fails with a clear error identifying subject and trial indices.
2. **Given** an experiment configuration specifying dataset, model family, and a nested subject-aware evaluation protocol, **When** the researcher executes the experiment, **Then** the system performs leakage-free nested validation with subject-disjoint inner and outer splits and records a resolved, immutable configuration used for the run.
3. **Given** multiple candidate preprocessed datasets and broad core hyperparameter ranges, **When** Stage 1 HPO runs, **Then** the system searches across dataset choice and foundational parameters (e.g., key architecture dimensions, training rate) and returns the best configuration and its evidence.
4. **Given** the best configuration from Stage 1, **When** Stage 2 HPO runs, **Then** the system performs a focused search of architectural and regularization details and returns an improved configuration and evidence.
5. **Given** the best configuration from Stage 2, **When** Stage 3 HPO runs, **Then** the system jointly tunes data augmentation options and sensitive training parameters, returning a final champion configuration.
6. **Given** a champion configuration, **When** the final evaluation is launched with a LOSO protocol and multiple random seeds, **Then** the system refits fresh models per outer fold, aggregates performance across seeds, and reports unbiased metrics with confidence intervals and significance testing.
7. **Given** a completed run, **When** the researcher opens the run directory, **Then** they find environment snapshots, the resolved configuration, fold-by-fold results, overall and per-subject confusion matrices, forest and caterpillar plots, subject-level reliability summaries, and grand-average plus per-fold interpretability reports.

### Edge Cases
- Missing or extra behavioral rows versus EEG trials (misalignment): system halts with a descriptive error listing offending indices and counts.
- Inconsistent condition labels across subjects: system constructs a global codebook and errors if unmappable or ambiguous labels are found.
- Missing channels or montage mismatch across subjects: system reduces to the intersection of channels common to all subjects, logs removed channels, and halts only if the intersection is empty or violates predefined safety thresholds.
- Subjects with fewer than 5 trials in any class: system excludes the subject from analysis, logs a warning with subject ID and per-class counts, and updates condition distributions accordingly.
- Too few subjects to support LOSO or nested validation: system halts with guidance on minimum subject counts.
- Class imbalance or rare classes: system reports class distributions per subject, applies class-weighted loss and metrics adjustments when distributions remain imbalanced despite meeting minimum trial thresholds, and ensures evaluation metrics reflect multi-class setting and chance levels.
- Corrupted or unreadable files: system identifies files by absolute path and subject ID and halts with guidance.

### Out of Scope
- Real-time or online EEG decoding capabilities.
- Upstream EEG cleaning steps such as artifact removal or filtering; inputs must already be HAPPE-processed.


## Requirements *(mandatory)*

    FR-001: System MUST ingest per-subject pre-cleaned EEG files and corresponding behavioral CSVs.

    FR-002: System MUST align each EEG trial with its behavioral and condition labels and enforce a strict one-to-one match, terminating with a clear, actionable error on any mismatch.

    FR-003: System MUST create and persist a globally consistent encoding for experimental conditions across all subjects.

    FR-004: System MUST attach a standard 3D sensor-location montage to each subject's EEG data.

    FR-005: System MUST save prepared data in a standardized .fif format with embedded metadata.

    FR-006: The pipeline MUST verify that a minimum of 10 subjects are available for analysis and halt with an error if this condition is not met.

    FR-007: The pipeline MUST verify that each included subject has at least 5 trials for every class and exclude subjects who do not meet this threshold.

    FR-008: When channel sets differ across subjects, the System MUST restrict analysis to the intersection of channels common to all subjects.

    FR-009: System MUST allow experiments to be defined entirely, but not exclusively, via version-controlled configuration files.

    FR-010: System MUST implement nested, subject-aware cross-validation, ensuring subjects in training and validation/test splits are always disjoint.

    FR-011: The HPO process MAY use either a nested Leave-N-Subjects-Out (e.g., 6-fold group) cross-validation for its outer loop for computational feasibility, or a full nested Leave-One-Subject-Out (LOSO) for maximum rigor.

    FR-012: For final evaluation, System MUST run a nested Leave-One-Subject-Out protocol and, for each outer fold, MUST refit a new model from scratch on all outer-training subjects.

    FR-013: The final evaluation MUST support execution across multiple random seeds to produce stable performance estimates.

    FR-014: System MUST detect and prevent data leakage, including inadvertent subject overlap between inner/outer splits or using test-set statistics during model selection; violations MUST raise errors.

    FR-015: System MUST support a multi-stage HPO pipeline with three sequential stages (Foundational, Architectural Refinement, and Joint Augmentation/Optimizer).

    FR-016: The HPO process MUST select a single champion configuration based on a pre-declared, version-controlled selection criterion (e.g., mean inner-validation macro F1-score).

    FR-017: Every run MUST automatically generate a self-contained output directory containing all configurations, logs, results, and environment snapshots for auditability.

    FR-018: All output filenames and identifiers (e.g., for subjects, folds) MUST be zero-padded and use a consistent snake_case naming convention.

    FR-019: All pipeline scripts MUST provide continuous console logging, including per-epoch metrics, fold completion timestamps, and timing information for observability.

    FR-020: System MUST provide a master script that triggers a comprehensive refresh of all HPO study results, including summary CSVs, overview plots, and a detailed subject performance analysis on the best trial of each study. This refresh MUST execute via the maintained entry points (`scripts/refresh_optuna_summaries.py` and its launcher under `results/optuna/refresh_all_studies.*`) and persist completion logs alongside the regenerated artifacts.

    FR-021: The pipeline MUST automatically run a streamlined post-hoc statistical analysis after every train.py execution, generating a single, consolidated HTML report as the primary output artifact.

    FR-022: The statistical analysis MUST include a group-level Generalized Linear Mixed-Effects Model (GLMM) with a random intercept for subject to test for population-level significance.

    FR-023: The statistical analysis MUST include a subject-level reliability analysis, reporting the proportion of subjects performing significantly above chance with appropriate multiple-comparisons correction.

    FR-024: The pipeline MUST support a final, large-scale permutation test on a champion model to generate a definitive, non-parametric p-value.

    FR-025: System MUST automatically calculate and use the correct chance level based on the number of classes in the current task.

    FR-026: The pipeline MUST produce a rich set of visualizations, including: overall and per-subject confusion matrices; a forest plot showing the distribution of per-subject performance; a caterpillar plot showing individual subject effects from the GLMM; and comprehensive model interpretability (XAI) reports with both grand-average and per-fold attribution maps.


### Key Entities *(include if feature involves data)*
- **Subject**: A participant contributing EEG trials; identified by a unique subject ID; associated with per-subject prepared data and per-fold roles (train/val/test).
- **Trial**: A single-trial EEG segment with a time window and label; attributes include condition code, timestamp indices, and quality flags.
- **Condition (Label)**: Encoded experimental category (e.g., numerosity) with a global codebook mapping human-readable labels to codes.
- **Dataset**: A named collection of subjects and their prepared files; may refer to one of multiple candidate preprocessings.
- **Montage**: Standardized sensor locations applied to subjects; tracked for consistency checks.
- **Experiment**: A configuration-defined run specifying dataset, model family, evaluation protocol, and search stages.
- **Configuration**: Resolved, immutable parameters governing a run; includes seeds, split definitions, and selection criteria.
- **Fold**: A validation unit; includes outer folds (e.g., LOSO per subject) and nested inner folds defined on the outer-training set.
- **HyperparameterSearchStage**: One of three sequential searches (Stage 1/2/3) with its search space, best configuration, and evidence.
- **Run**: A single execution instance producing artifacts, logs, and metrics in a dedicated directory.
- **Metric**: Quantitative outcome (e.g., accuracy, macro-F1) reported per fold, per subject, and aggregated with uncertainty.
- **Artifact**: Any output needed for auditability (resolved config, environment snapshot, logs, figures, tables, codebook, checksums).
- **StatisticalModelResult**: Outputs from group-level tests (e.g., test statistics, p-values, effect sizes) and model diagnostic visuals.
- **XAIAttribution**: Interpretability outputs aggregated across folds and subjects (e.g., attribution maps) with appropriate summaries.


---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous  
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified


---

## Execution Status
*Updated by main() during processing*

- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities marked
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed

---
