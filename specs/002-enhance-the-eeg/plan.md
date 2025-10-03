
# Implementation Plan: XAI Reporting System

**Branch**: `002-enhance-the-eeg` | **Date**: 2025-10-03 | **Spec**: D:\eeg_nn\specs\002-enhance-the-eeg\spec.md
**Input**: Feature specification from `D:\eeg_nn\specs\002-enhance-the-eeg\spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUde.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Implement a multi-modal XAI reporting system by heavily modifying `scripts/run_xai_analysis.py` and `utils/xai.py`. The system will use Captum for attributions (IG, Grad-CAM), MNE-Python for visualizations (topomaps, time-frequency analysis), and Playwright for PDF generation, producing a consolidated HTML/PDF report with per-fold, per-class, and grand-average insights, including a Top-2 Spatio-Temporal Event analysis.

## Technical Context
**Language/Version**: Python 3.11 (Conda `eegnex-env`)  
**Primary Dependencies**: MNE-Python, Captum, Playwright, SciPy, PyTorch, Matplotlib  
**Storage**: N/A (reads from run directories, writes to `xai_analysis/` subdirectory)  
**Testing**: pytest  
**Target Platform**: Windows/Linux (dev/execution)
**Project Type**: Single offline scientific pipeline  
**Performance Goals**: XAI analysis for a 24-subject run completes in < 10 minutes.  
**Constraints**: Must gracefully handle missing optional dependencies (Playwright, SciPy). Must be triggered from `train.py` via `--run-xai` flag (if it exists) or be runnable standalone.  
**Scale/Scope**: Scales to runs with up to 100 subjects.

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Reproducible Design)**: Plan maintains config-driven execution via `xai_defaults.yaml` and run-specific `summary.json`. All outputs are deterministic and traceable to a run ID. (PASS)
- **Principle II (Data Provenance & Integrity)**: The system is read-only regarding input run artifacts. (PASS)
- **Principle III (Deterministic Training & Search)**: XAI analysis respects run seeds for any stochastic processes. (PASS)
- **Principle IV (Rigorous Validation & Reporting)**: The core purpose of this feature is to enhance reporting transparency. (PASS)
- **Principle V (Audit-Ready Artifact Retention)**: All generated plots, data, and reports are stored in a version-controlled structure under the original run directory. (PASS)

## Project Structure

### Documentation (this feature)
```
specs/002-enhance-the-eeg/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   └── xai_config.yaml
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
scripts/
└── run_xai_analysis.py  # HEAVILY MODIFIED

utils/
└── xai.py               # HEAVILY MODIFIED

configs/
└── xai_defaults.yaml    # NEW

tests/
├── contract/
│   └── test_xai_config.py # NEW
└── integration/
    └── test_xai_pipeline.py # NEW
```

**Structure Decision**: The implementation will be contained within the existing `scripts` and `utils` directories, modifying `run_xai_analysis.py` as the orchestrator and `utils/xai.py` for the core XAI logic. A new `xai_defaults.yaml` will be added to `configs`. New tests will be added to `tests/contract` and `tests/integration`.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context**: All technical choices are specified by the user. No research needed.
2. **Generate and dispatch research agents**: N/A.
3. **Consolidate findings** in `research.md`:
   - Decision: Proceed with user-specified stack (Captum, MNE, Playwright).
   - Rationale: Aligns with existing project dependencies and provides all necessary functionality.
   - Alternatives considered: None, as the path is clear.

**Output**: `research.md` confirming the technical approach.

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Add `TimeFrequencyMap` entity.
   - Refine `SpatioTemporalEvent` to include peak window configuration.

2. **Generate contracts**:
   - Create `specs/002-enhance-the-eeg/contracts/xai_config.yaml` defining the schema for `configs/xai_defaults.yaml`. This includes keys for `peak_window_ms`, `gradcam_target_layer`, `tf_morlet_freqs`, etc.

3. **Update `utils/xai.py` interface**:
    - Define clear function signatures for:
        - `compute_ig_attributions`
        - `compute_gradcam_attributions`
        - `plot_attribution_heatmap`
        - `plot_topomap`
        - `compute_time_frequency_map`
        - `plot_time_frequency_map`
    - These functions will take model, data, and config as input and return numpy arrays or figure objects. File I/O will be handled by the orchestrator.

4. **Design Orchestrator (`run_xai_analysis.py`)**:
    - Refactor `main()` to follow the new spec:
        - Load base config from `xai_defaults.yaml` and merge with run config.
        - Loop through folds to compute and save per-fold IG and Grad-CAM artifacts.
        - Compute and save grand-average IG.
        - Compute and save per-class grand-average IG.
        - Compute and save time-frequency analysis from grand-average IG.
        - Perform peak analysis to find Top-2 spatio-temporal events.
        - Generate all plots (heatmaps, topomaps).
        - Call a new `create_consolidated_report` function.

5. **Generate contract tests**:
   - Create `tests/contract/test_xai_config.py` to validate `configs/xai_defaults.yaml` against the contract.

6. **Extract test scenarios** → `quickstart.md`:
   - Document the command to run the XAI analysis: `python scripts/run_xai_analysis.py --run-dir <path_to_run>`.
   - Describe expected outputs in the `xai_analysis` directory and the final HTML/PDF report.

**Output**: `data-model.md`, `contracts/xai_config.yaml`, `quickstart.md`.

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- A TDD approach will be used.
- Create an integration test (`tests/integration/test_xai_pipeline.py`) that runs the full pipeline on a small, deterministic fixture and asserts the existence and basic correctness of all output artifacts (HTML report, key plots, numpy arrays). This test will initially fail.
- Tasks will be generated to implement the functionalities in `utils/xai.py` and `scripts/run_xai_analysis.py` needed to make the integration test pass.

**Ordering Strategy**:
1.  Setup: Create test fixtures and `configs/xai_defaults.yaml`.
2.  Tests: Write the failing contract and integration tests.
3.  Implementation (Bottom-up):
    - Implement core attribution functions in `utils/xai.py` (IG, Grad-CAM). [P]
    - Implement plotting functions in `utils/xai.py` (heatmap, topomap, TF plot). [P]
    - Implement analysis functions (peak-finding, TF compute). [P]
    - Refactor `run_xai_analysis.py` to orchestrate the new functions and generate all artifacts.
    - Implement the final HTML/PDF report generation.
4.  Documentation: Update README and other relevant docs.

**Estimated Output**: 15-20 numbered, ordered tasks in `tasks.md`.

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates `tasks.md`)  
**Phase 4**: Implementation (execute `tasks.md`)  
**Phase 5**: Validation (run tests, execute `quickstart.md`)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None      | N/A        | N/A                                 |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v1.0.0 - See `D:\eeg_nn\.specify\memory\constitution.md`*
