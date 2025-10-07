# Training Runner Refactoring Strategy & Verification Plan

**Version**: 1.0.0  
**Date**: 2025-10-06  
**Status**: PRE-REFACTORING  
**Constitutional Compliance**: Section III (Deterministic Training) & Section V (Audit-Ready Artifacts)

---

## Executive Summary

This document outlines our strategy to refactor `training_runner.py` from 1,792 lines to a maintainable, modular architecture while **guaranteeing zero behavioral changes**. We will use test-driven development to verify that every single output, checkpoint, CSV, plot, and metric remains bit-for-bit identical after refactoring.

**Critical Success Criteria:**
1. ✅ All existing tests pass without modification
2. ✅ New integration tests verify identical behavior
3. ✅ Deterministic outputs (given same seed) remain identical
4. ✅ All CSV artifacts have identical content
5. ✅ All metrics computed identically
6. ✅ No change to public API

---

## Phase 0: Pre-Refactoring Analysis

### Current Architecture Problems

**File:** `training_runner.py` (1,792 lines)

#### The Monolithic `run()` Method (Lines 393-1786)
- **1,393 lines** in a single method (78% of file!)
- Handles 8+ distinct responsibilities
- Violates Single Responsibility Principle
- Difficult to test in isolation
- Hard to understand, maintain, and extend

#### Specific Complexity Hotspots:

1. **Objective Computation Logic** (Lines 217-268)
   - `_compute_objective_metric()`: 52 lines
   - Dual-mode support (threshold vs weighted)
   - Called from 5+ different locations
   - Mixed with checkpointing logic

2. **Inner Training Loop** (Lines 687-873)
   - ~186 lines of nested logic
   - Handles: warmup, augmentation ramp, mixup, validation
   - Checkpoint selection logic intertwined
   - Early stopping logic intertwined

3. **Outer Evaluation** (Lines 1022-1232)
   - 210 lines split between ensemble and refit modes
   - Duplicated prediction collection logic

4. **Plotting** (Lines 1286-1550)
   - 264 lines of plotting logic
   - Per-fold plots, enhanced plots, overall plots
   - Duplicated title/label generation

5. **CSV Artifact Writing** (Lines 1586-1735)
   - 149 lines of CSV writing
   - 4 different CSV files (learning curves, outer metrics, test predictions inner/outer)

---

## Phase 1: Verification Strategy

### Test-Driven Refactoring Approach

**Philosophy:** Write tests that MUST FAIL if behavior changes, then refactor to make them pass.

#### Test Categories:

##### 1. **Golden Output Tests** (Highest Priority)
- Run training_runner with fixed seed
- Capture all outputs as "golden" reference
- After refactoring, run again and compare bit-for-bit
- **Artifacts to capture:**
  - `outer_eval_metrics.csv` (exact floats)
  - `learning_curves_inner.csv` (exact floats)
  - `test_predictions_outer.csv` (exact floats)
  - `test_predictions_inner.csv` (exact floats)
  - `splits_indices.json` (exact structure)
  - Final metrics dictionary (exact floats)

##### 2. **Metric Computation Tests**
- Test `_compute_objective_metric()` with known inputs
- Test threshold mode vs weighted mode
- Test all 5 objective types
- Test edge cases (zero values, all same predictions)

##### 3. **Checkpoint Selection Tests**
- Test that best checkpoint is selected correctly
- Test tie-breaking behavior (objective metric, then validation loss)
- Test that early stopping uses correct metric

##### 4. **CV Split Tests**
- Test that outer/inner splits are identical
- Test subject leakage detection
- Test predefined splits vs computed splits

##### 5. **Integration Tests**
- Run minimal training (1 epoch, 2 folds)
- Verify all artifacts are created
- Verify no exceptions raised

---

## Phase 2: Refactoring Execution Plan

### Stage 1: Extract Metric Computation (LOW RISK)

**Goal:** Create `ObjectiveComputer` class

**Files to create:**
- `code/training/metrics.py` (~150 lines)

**What moves:**
- `_get_composite_params()` → `ObjectiveComputer.__init__()`
- `_compute_objective_metric()` → `ObjectiveComputer.compute()`
- Add comprehensive docstrings

**API:**
```python
class ObjectiveComputer:
    """Encapsulates objective-aligned metric computation."""
    
    def __init__(self, cfg: Dict):
        """Initialize with config, validate parameters."""
        self.objective = cfg["optuna_objective"]
        self.params = self._parse_composite_params(cfg)
    
    def compute(self, val_acc, val_macro_f1, val_min_f1, val_plur_corr) -> float:
        """Compute objective metric for checkpoint selection."""
        # Handles all 5 objective types
        # Returns float to maximize
    
    def get_mode(self) -> str:
        """Return 'threshold', 'weighted', or 'direct'."""
    
    def get_params(self) -> Dict:
        """Return objective-specific parameters."""
```

**Testing:**
- Create `tests/test_objective_computer.py`
- Test all 5 objectives with known inputs/outputs
- Test threshold edge cases
- Test weighted edge cases
- Test invalid config (must raise ValueError)

**Risk:** LOW - Pure function extraction, easy to test

---

### Stage 2: Extract CSV Writers (LOW RISK)

**Goal:** Move CSV writing logic to dedicated module

**Files to create:**
- `code/artifacts/csv_writers.py` (~200 lines)

**What moves:**
- Learning curves CSV writer
- Outer eval metrics CSV writer
- Test predictions CSV writers (inner/outer)
- All fieldname definitions

**API:**
```python
class LearningCurvesWriter:
    def __init__(self, run_dir: Path):
        self.csv_path = run_dir / "learning_curves_inner.csv"
    
    def write(self, rows: List[Dict]):
        """Write learning curves to CSV."""

class OuterEvalMetricsWriter:
    def __init__(self, run_dir: Path):
        self.csv_path = run_dir / "outer_eval_metrics.csv"
    
    def write(self, rows: List[Dict], aggregate_row: Dict):
        """Write outer eval metrics with aggregate row."""

class TestPredictionsWriter:
    def __init__(self, run_dir: Path, mode: str):
        # mode: 'inner' or 'outer'
        self.csv_path = run_dir / f"test_predictions_{mode}.csv"
    
    def write(self, rows: List[Dict]):
        """Write test predictions to CSV."""
```

**Testing:**
- Create `tests/test_csv_writers.py`
- Test that CSV format is identical to current output
- Test field order preservation
- Test empty rows handling

**Risk:** LOW - Pure I/O extraction, no logic changes

---

### Stage 3: Extract Plotting (LOW RISK)

**Goal:** Move plotting logic to dedicated module (already partially done)

**Files to modify:**
- Ensure `utils/plots.py` is complete
- Create `code/artifacts/plot_builders.py` for title/label generation

**What moves:**
- Title generation logic for fold plots
- Title generation logic for overall plots
- Enhanced plot logic
- Per-class F1 info text generation

**API:**
```python
class PlotTitleBuilder:
    def __init__(self, cfg: Dict, objective_computer: ObjectiveComputer):
        self.trial_dir_name = ...
        self.obj_computer = objective_computer
    
    def build_fold_title(self, fold: int, test_subjects, inner_metrics, outer_acc) -> str:
        """Build title for per-fold confusion matrix."""
    
    def build_overall_title(self, inner_metrics, outer_metrics) -> str:
        """Build title for overall confusion matrix."""
```

**Testing:**
- Create `tests/test_plot_builders.py`
- Test title format for all objective types
- Test threshold vs weighted mode labels

**Risk:** LOW - Mostly string formatting, no computation

---

### Stage 4: Extract Checkpoint Manager (MEDIUM RISK)

**Goal:** Create `CheckpointManager` class

**Files to create:**
- `code/training/checkpointing.py` (~150 lines)

**What moves:**
- Early stopping logic
- Checkpoint selection logic (lines 831-867)
- Best state tracking
- Patience tracking

**API:**
```python
class CheckpointManager:
    """Manages checkpoint selection and early stopping."""
    
    def __init__(self, cfg: Dict, objective_computer: ObjectiveComputer):
        self.early_stop_patience = cfg.get("early_stop", 10)
        self.obj_computer = objective_computer
        self.reset()
    
    def reset(self):
        """Reset for new inner fold."""
        self.best_objective = float("-inf")
        self.best_checkpoint_loss = float("inf")
        self.best_state = None
        self.best_metrics = {}
        self.patience = 0
    
    def update(self, model_state, val_metrics: Dict) -> bool:
        """
        Update checkpoint based on current epoch metrics.
        
        Returns True if checkpoint was updated.
        """
    
    def should_stop(self) -> bool:
        """Return True if early stopping triggered."""
    
    def get_best_state(self) -> Dict:
        """Return best model state."""
    
    def get_best_metrics(self) -> Dict:
        """Return best metrics."""
```

**Testing:**
- Create `tests/test_checkpoint_manager.py`
- Test checkpoint update logic with known sequences
- Test early stopping trigger
- Test tie-breaking (objective metric, then loss)
- Test that best state is preserved

**Risk:** MEDIUM - Core training logic, must test thoroughly

---

### Stage 5: Extract Inner Trainer (HIGH RISK)

**Goal:** Create `InnerTrainer` class for inner fold training loop

**Files to create:**
- `code/training/inner_loop.py` (~350 lines)

**What moves:**
- Epoch loop (lines 687-873)
- LR warmup logic
- Augmentation warmup logic
- Mixup logic
- Training forward/backward pass
- Validation pass
- Learning curve collection

**API:**
```python
class InnerTrainer:
    """Runs one inner fold training loop."""
    
    def __init__(self, cfg: Dict, objective_computer: ObjectiveComputer):
        self.cfg = cfg
        self.obj_computer = objective_computer
    
    def train(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: nn.Module,
        tr_loader: DataLoader,
        va_loader: DataLoader,
        aug_builder: Callable,
        input_adapter: Callable | None,
        checkpoint_manager: CheckpointManager,
        fold_info: Dict,  # outer_fold, inner_fold for logging
        optuna_trial: optuna.Trial | None,
        global_step_offset: int,
    ) -> Dict:
        """
        Train model for one inner fold.
        
        Returns:
            {
                "best_state": model state dict,
                "best_metrics": dict of best metrics,
                "learning_curves": list of dicts,
                "tr_hist": list of train losses,
                "va_hist": list of val losses,
                "va_acc_hist": list of val accs,
            }
        """
```

**Testing:**
- Create `tests/test_inner_trainer.py`
- Test with minimal config (1 epoch)
- Test LR warmup behavior
- Test augmentation warmup behavior
- Test mixup application
- Test Optuna pruning
- Test early stopping interaction

**Risk:** HIGH - Core training logic with many edge cases

---

### Stage 6: Extract Outer Evaluator (HIGH RISK)

**Goal:** Create `OuterEvaluator` class for test evaluation

**Files to create:**
- `code/training/evaluation.py` (~250 lines)

**What moves:**
- Ensemble evaluation logic (lines 1022-1077)
- Refit evaluation logic (lines 1078-1230)
- Prediction collection logic

**API:**
```python
class OuterEvaluator:
    """Handles outer test evaluation (ensemble or refit)."""
    
    def __init__(self, cfg: Dict):
        self.mode = cfg["outer_eval_mode"]
        self.cfg = cfg
    
    def evaluate(
        self,
        model_builder: Callable,
        num_classes: int,
        inner_results: List[Dict],
        dataset,
        y_all: np.ndarray,
        groups: np.ndarray,
        tr_idx: np.ndarray,
        te_idx: np.ndarray,
        aug_transform,
        input_adapter: Callable | None,
        class_names: List[str],
        fold: int,
    ) -> Dict:
        """
        Evaluate on outer test set.
        
        Returns:
            {
                "y_true": list,
                "y_pred": list,
                "test_pred_rows": list of dicts,
            }
        """
    
    def _evaluate_ensemble(self, ...) -> Dict:
        """Ensemble of K inner models."""
    
    def _evaluate_refit(self, ...) -> Dict:
        """Refit single model on full outer train."""
```

**Testing:**
- Create `tests/test_outer_evaluator.py`
- Test ensemble mode with mock inner results
- Test refit mode
- Test prediction collection format

**Risk:** HIGH - Two different evaluation modes, must preserve behavior

---

### Stage 7: Simplify Orchestrator (FINAL)

**Goal:** Reduce `training_runner.py` to orchestration only

**What remains in `run()`:**
- Outer CV loop structure
- Call to `InnerTrainer.train()` for each inner fold
- Call to `OuterEvaluator.evaluate()` for each outer fold
- Call to CSV writers, plot builders
- Summary computation
- Top-level coordination

**Expected line count:** ~400 lines (down from 1,792)

**Testing:**
- Run full integration test
- Compare all outputs to golden reference
- Verify determinism (same seed = same outputs)

**Risk:** MEDIUM - Final integration, but most logic extracted

---

## Phase 3: Verification Checklist

After each stage, verify:

### Unit Test Verification
- [ ] All new unit tests pass
- [ ] All existing unit tests pass
- [ ] Code coverage >= 80% for new modules

### Integration Test Verification
- [ ] Golden output test passes (bit-for-bit match)
- [ ] CSV artifacts identical
- [ ] Metrics dictionary identical
- [ ] Plots generated (visual inspection)
- [ ] No new linter errors

### Behavioral Verification
- [ ] Run with seed=42, compare outputs
- [ ] Run with seed=123, compare outputs
- [ ] Test both threshold and weighted objectives
- [ ] Test both ensemble and refit modes
- [ ] Test with predefined splits
- [ ] Test with Optuna trial (pruning works)

### Constitutional Compliance Verification
- [ ] Determinism preserved (Section III)
- [ ] All artifacts still generated (Section V)
- [ ] No silent fallbacks introduced (Section III)
- [ ] Stage handoff validation unchanged

---

## Phase 4: Risk Mitigation

### Mitigation Strategies:

1. **Incremental Commits**
   - Commit after each successful stage
   - Tag as `refactor-stage-N-complete`
   - Easy rollback if needed

2. **Parallel Verification**
   - Keep old `training_runner.py` as `training_runner_legacy.py`
   - Run both in parallel during transition
   - Compare outputs until confident

3. **Comprehensive Logging**
   - Add debug logging to new modules
   - Compare log outputs between old/new

4. **Gradual Adoption**
   - Make new modules optional (feature flag)
   - Allow fallback to monolithic code
   - Remove fallback only after confidence

---

## Phase 5: Post-Refactoring Benefits

### Immediate Benefits:
1. ✅ **Testability:** Each module has <300 lines, easy to test
2. ✅ **Maintainability:** Clear responsibilities, easy to find bugs
3. ✅ **Collaboration:** Multiple people can work on different modules
4. ✅ **Documentation:** Each class has focused docstring
5. ✅ **Reusability:** Modules can be used in other scripts

### Long-Term Benefits:
1. ✅ **Feature Addition:** New objectives easy to add to `ObjectiveComputer`
2. ✅ **Bug Fixes:** Isolated modules easier to debug
3. ✅ **Refactoring:** Future refactoring easier with clear boundaries
4. ✅ **Career Impact:** Demonstrates software engineering maturity

---

## Phase 6: Rollout Plan

### Week 1: Preparation
- ✅ Write this strategy document
- ✅ Build golden output tests
- ✅ Establish baseline

### Week 2: Low-Risk Stages (1-3)
- Extract `ObjectiveComputer`
- Extract CSV writers
- Extract plot builders
- Verify golden outputs match

### Week 3: Medium-Risk Stage (4)
- Extract `CheckpointManager`
- Comprehensive unit tests
- Integration verification

### Week 4: High-Risk Stages (5-6)
- Extract `InnerTrainer`
- Extract `OuterEvaluator`
- Extensive testing

### Week 5: Final Integration (7)
- Simplify orchestrator
- Full regression testing
- Documentation update

### Week 6: Confidence Building
- Run parallel verification
- Multiple seed tests
- Edge case testing
- Remove legacy code

---

## Success Metrics

### Quantitative:
- [ ] `training_runner.py`: 1,792 lines → ~400 lines
- [ ] Longest method: 1,393 lines → ~150 lines
- [ ] Test coverage: ~60% → 80%+
- [ ] All tests passing
- [ ] Zero behavioral changes (golden outputs match)

### Qualitative:
- [ ] Code review: easier to understand
- [ ] Onboarding time: reduced
- [ ] Bug fix time: reduced
- [ ] Feature addition time: reduced
- [ ] Confidence level: high

---

## Appendix A: Detailed Line Mapping

### Current `training_runner.py` Structure:

```
Lines 1-115: Imports, helpers, device setup
Lines 117-163: Helper functions (_seed_worker, format_*, validate_*)
Lines 165-392: TrainingRunner class setup
  Lines 166-171: __init__
  Lines 173-215: _get_composite_params
  Lines 217-268: _compute_objective_metric
  Lines 270-277: _ensure_run_dir
  Lines 279-322: _resolve_stage_context, validate_stage_handoff
  Lines 325-377: _make_loaders
  Lines 379-391: _validate_subject_requirements

Lines 393-1786: run() method (THE MONSTER)
  Lines 393-449: Setup (labels, validation, logging)
  Lines 450-491: Outer CV split computation
  Lines 493-525: Initialize accumulators
  Lines 527-1412: OUTER FOLD LOOP
    Lines 530-565: Outer fold setup
    Lines 569-584: Inner split computation
    Lines 586-927: INNER FOLD LOOP (training)
      Lines 601-636: DataLoader creation, optimizer setup
      Lines 639-686: Initialize inner training state
      Lines 687-873: EPOCH LOOP (the beast within the beast)
        Lines 689-705: LR & aug warmup
        Lines 707-738: Training pass (with mixup)
        Lines 741-800: Validation pass + learning curves
        Lines 803-826: Optuna pruning
        Lines 831-867: Checkpoint selection & early stopping
      Lines 875-913: Collect inner val predictions
      Lines 915-926: Store inner results
    Lines 930-954: Aggregate inner metrics
    Lines 956-1008: Select best inner model (by objective)
    Lines 1010-1232: Outer test evaluation
      Lines 1022-1077: Ensemble mode
      Lines 1078-1230: Refit mode
    Lines 1234-1277: Compute fold metrics
    Lines 1279-1408: Generate plots (per-fold + enhanced)
    Lines 1410-1412: Log fold completion
  Lines 1414-1441: Compute overall metrics
  Lines 1443-1550: Generate overall plots
  Lines 1552-1601: Write splits_indices.json
  Lines 1603-1735: Write CSV artifacts
  Lines 1737-1786: Return summary metrics

Lines 1789-1791: Module-level helper
```

### After Refactoring:

```
training_runner.py (~400 lines):
  Lines 1-50: Imports, helpers
  Lines 51-150: TrainingRunner class setup (unchanged)
  Lines 151-400: run() method (orchestration only)

training/metrics.py (~150 lines):
  Lines 1-20: Imports
  Lines 21-150: ObjectiveComputer class

training/checkpointing.py (~150 lines):
  Lines 1-20: Imports
  Lines 21-150: CheckpointManager class

training/inner_loop.py (~350 lines):
  Lines 1-30: Imports
  Lines 31-350: InnerTrainer class

training/evaluation.py (~250 lines):
  Lines 1-30: Imports
  Lines 31-250: OuterEvaluator class

artifacts/csv_writers.py (~200 lines):
  Lines 1-20: Imports
  Lines 21-200: Writer classes

artifacts/plot_builders.py (~150 lines):
  Lines 1-20: Imports
  Lines 21-150: PlotTitleBuilder class
```

**Total lines: ~1,650 lines** (vs 1,792 original)
**But distributed across 7 focused files instead of 1 monolith**

---

## Appendix B: Test Data Fixtures

### Golden Reference Test
```python
# tests/test_golden_reference.py
def test_golden_output_match():
    """Run training_runner with fixed seed, compare all outputs."""
    
    # Setup
    cfg = load_test_config()  # Small dataset, 2 epochs, 2 folds
    cfg["seed"] = 42
    cfg["run_dir"] = tmp_path / "test_run"
    
    # Run
    runner = TrainingRunner(cfg, label_fn)
    results = runner.run(dataset, groups, class_names, ...)
    
    # Load golden reference
    golden_results = load_golden_reference()
    golden_csv_outer = load_csv("golden_outer_eval_metrics.csv")
    golden_csv_curves = load_csv("golden_learning_curves.csv")
    
    # Compare
    assert results == golden_results  # Exact float match
    assert csv_outer == golden_csv_outer  # Exact match
    assert csv_curves == golden_csv_curves  # Exact match
```

---

## Appendix C: Constitutional Compliance Matrix

| Constitution Section | Current State | Post-Refactoring | Verification |
|---------------------|---------------|------------------|--------------|
| I. Reproducible Design | ✅ Resolved config saved | ✅ Unchanged | Golden output test |
| II. Data Provenance | ✅ Read-only data | ✅ Unchanged | No file modifications |
| III. Deterministic Training | ✅ Seeds enforced | ✅ Unchanged | Same seed = same output |
| III. Explicit Params | ✅ Fail-fast on missing | ✅ Unchanged | Config validation tests |
| IV. Subject-Aware CV | ✅ GroupKFold enforced | ✅ Unchanged | Split leakage tests |
| V. Artifact Retention | ✅ All artifacts saved | ✅ Unchanged | CSV/plot existence tests |

**Conclusion:** Refactoring is constitutionally compliant—no behavior changes, only code organization.

---

## Appendix D: Decision Log

### Design Decisions Made:

1. **Why extract `ObjectiveComputer` first?**
   - Lowest risk (pure computation)
   - High reuse (called from multiple places)
   - Easy to test (deterministic inputs/outputs)

2. **Why keep `run()` in `TrainingRunner` class?**
   - Maintains public API compatibility
   - Orchestration naturally belongs in runner
   - Allows gradual adoption

3. **Why not extract data loading?**
   - `_make_loaders()` is already small (~50 lines)
   - Tightly coupled to outer/inner index logic
   - Low benefit for refactoring effort

4. **Why separate `InnerTrainer` from `OuterEvaluator`?**
   - Different responsibilities (training vs evaluation)
   - Inner trainer reusable for hyperparameter tuning
   - Outer evaluator has two modes (ensemble/refit)

5. **Why extract CSV writers?**
   - Low risk (pure I/O)
   - Reduces noise in orchestrator
   - Easier to add new CSV formats in future

---

**END OF STRATEGY DOCUMENT**

*This document will be updated as refactoring progresses. All changes will be tracked in git history for constitutional compliance (Section V: Audit-Ready Artifacts).*

