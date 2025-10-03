# Feature Specification: XAI Reporting System (IG/Grad-CAM, Per-Class, Time-Frequency)

**Feature Branch**: `[002-enhance-the-eeg]`  
**Created**: 2025-10-03  
**Status**: Draft  
**Input**: User description: "Enhance the EEG-Decode-Pipeline with a comprehensive and configurable multi-modal eXplainable AI (XAI) reporting system to provide deep, interpretable views into the model's decision-making process. The pipeline must be extended to generate the following XAI artifacts for every completed run: Core Attribution & Visualization: The system must compute and visualize attributions using both Integrated Gradients (IG) and Grad-CAM. For each method, it must produce grand-average and per-fold heatmaps (channel x time) and topomaps (scalp distribution). Per-Condition Analysis: To understand how the model distinguishes between cardinalities, the system must generate a full set of grand-average attribution visualizations (heatmaps and topomaps) for each individual class. Time-Frequency Analysis: The system must compute and visualize a time-frequency representation of the grand-average IG attribution map, revealing which oscillatory bands were most influential to the model's decisions over time. Artifact Organization: All generated plots must be saved to a well-organized xai_analysis directory, with filenames prefixed by the run ID for traceability. Consolidated HTML/PDF Report: The system must automatically generate a single, consolidated consolidated_xai_report.html and a corresponding PDF. The report's primary summary must identify and display the Top 2 Spatio-Temporal Events (The top 10 channels ARE specific to each time window). Each \"event\" will consist of: A significant temporal window, identified by peak analysis of the overall attribution signal. A list of the Top 10 most important channels within that specific window. A topomap visualizing the channel importance for that window. The report will then present the full gallery of all grand-average and per-fold visualizations. Configurability: Key XAI parameters must be controllable via a central configuration file. Always connect to the environment conda activate eegnex-env"

## Clarifications

### Session 2025-10-03
- Q: For IG per-fold attribution averaging, should we include only correctly classified test trials or all test trials? → A: Only correctly classified
- Q: For the two peak windows used in the Top‑2 Spatio‑Temporal Events, should the window length be fixed at 50 ms or configurable? → A: Configurable; default 100 ms
- Q: For Grad‑CAM/TopoCAM, should it be optional (skip if no target layer) or required with a default? → A: Optional; skip if target layer absent
- Q: Do you want per-class outputs limited to grand-average only, or also per-fold per-class artifacts? → A: Grand-average per-class only
- Q: For the time-frequency view (FR-005), which method should be the default? → A: Wavelet transform

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
As a computational neuroscience researcher, after completing a model run, I want a consolidated, interpretable XAI report that explains which channels and time windows most influenced the model's decisions (overall and per class), so I can assess plausibility, communicate findings, and guide further experiments.

### Acceptance Scenarios
1. Given a completed run directory with per-fold checkpoints and a summary file, When the XAI analysis is executed, Then per-fold Integrated Gradients (IG) attribution matrices (channels × time) and heatmaps are saved, grand-average IG artifacts are produced, and a consolidated HTML report is written to the run directory.
2. Given the same run, When a Grad-CAM target layer is configured, Then per-fold Grad-CAM heatmaps/vectors are saved and included in the grand-average summary; When no target is provided, the system logs a clear skip and still completes IG outputs and the report.
3. Given the custom montage file is present, When topoplots are requested, Then scalp topomaps render for overall and peak windows; When montage attachment fails, Then the system continues and logs a skip for topoplots without failing the run.
4. Given multi-class tasks, When XAI analysis completes, Then grand-average, per-class IG visualizations (heatmaps and, when montage attached, topomaps) are produced for each class.
5. Given grand-average IG is available, When time-frequency analysis is enabled, Then a grand-average time-frequency visualization is produced that highlights influential oscillatory bands over time.
6. Given report generation, When peak analysis is performed, Then two most prominent spatio-temporal events (≥100 ms apart) are identified, each with a configurable window (default 100 ms), a top‑10 channel list, and a corresponding topomap in the consolidated report; filenames are prefixed with the run ID and stored under `xai_analysis/`.

### Edge Cases
- Missing checkpoints or summary: fail with a clear error instructing how to regenerate prerequisites.
- No correctly classified test trials in a fold: log warning, skip that fold’s IG, still produce grand-average from remaining folds.
- Missing or unreadable montage: proceed without topomaps and log a clear message.
- Absent SciPy (peak detection): skip peak topoplots and include a note in the report.
- Missing class names: infer class count from data; label plots with generic class indices if names unavailable.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001 (Core IG)**: System MUST compute per-fold Integrated Gradients (IG) attributions on correctly classified test trials and save per‑fold matrices and heatmaps (channels × time) under `xai_analysis/`.
- **FR-002 (Core Grad‑CAM)**: System MUST support per‑fold Grad‑CAM; when a target layer is configured, it MUST save per‑fold heatmaps/vectors and include them in grand‑average summaries; when absent, it MUST skip gracefully with a clear log message and proceed.
- **FR-003 (Grand Average)**: System MUST compute grand‑average IG attributions across folds and save a `.npy` matrix and a heatmap; it SHOULD derive overall channel importance and produce an overall topomap when montage is attached.
- **FR-004 (Per‑Class IG)**: System MUST produce grand‑average IG visualizations per class (heatmaps and, when montage attached, topomaps) to explain class‑specific patterns.
- **FR-005 (Time‑Frequency View)**: System MUST provide a time‑frequency visualization derived from the grand‑average IG map (defaulting to Wavelet transform) to indicate influential oscillatory bands over time.
- **FR-006 (Top‑2 Events)**: System MUST identify two most prominent temporal peaks in the grand‑average attribution signal (separated by ≥100 ms), define configurable windows around each (default 100 ms via central config), list top‑10 channels for each window, and render topomaps when montage is attached.
- **FR-007 (Consolidated Report)**: System MUST generate `consolidated_xai_report.html` (and a PDF counterpart when a renderer is available) embedding per‑fold and grand‑average visuals, overall topomap, and the Top‑2 events.
- **FR-008 (Artifact Organization)**: All XAI artifacts MUST be saved under `<run_dir>/xai_analysis/` with zero‑padded, snake_case filenames prefixed by the run ID for traceability.
- **FR-009 (Configurability)**: Key XAI parameters (e.g., Grad‑Topo window, target layer, top‑K channels, TF options) MUST be centrally configurable (e.g., `configs/xai_defaults.yaml`) and merged with run config.
- **FR-010 (Determinism & Reproducibility)**: XAI analysis MUST respect run seeds and produce deterministic outputs where applicable, logging all parameters used in a lightweight summary JSON.

### Non‑Functional Requirements
- **NFR-001 (Performance)**: XAI analysis SHOULD complete within 10 minutes per run on reference hardware for 24 subjects; if exceeded, it MUST log timing and continue.
- **NFR-002 (Robustness)**: Missing optional dependencies (e.g., SciPy for peaks, PDF renderer) MUST degrade gracefully with explicit messages.
- **NFR-003 (Auditability)**: Filenames, parameters, and derived summaries MUST be self‑describing enough to allow independent replication.

### Key Entities *(include if feature involves data)*
- **Run**: A completed experimental run, identified by a unique `run_dir`, containing model checkpoints, a `summary.json` file, and test predictions.
- **AttributionMap**: A channels × time matrix summarizing IG (or Grad‑CAM) importance for a fold or aggregate.
- **Topomap**: A scalp distribution figure mapping channel importance values to electrode locations under a montage.
- **SpatioTemporalEvent**: A tuple of (window_ms, top_channels[], topomap_path) extracted from peak analysis.
- **XAIConfig**: Consolidated configuration controlling XAI behavior (e.g., target layer, window bounds, top‑K labeling).
- **ConsolidatedReport**: The final, self-contained HTML and PDF artifact that synthesizes all key XAI findings and visualizations.

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
