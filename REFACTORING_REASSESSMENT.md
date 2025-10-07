# Refactoring Reassessment: Where We Actually Are

**Date**: 2025-10-07  
**Critical Finding**: We're NOT at Stage 7. We're at an intermediate state.

---

## ❌ **The Problem**

### Plan Says:
- **Stage 7 Target**: ~400 lines in `training_runner.py`
- **`run()` method**: Should be ~150 lines of orchestration only

### Reality:
- **Current State**: 1,155 lines in `training_runner.py`
- **`run()` method**: **825 lines** (lines 409-1233)
- **Gap**: We need to extract ~755 more lines!

---

## 📊 **What We've Actually Completed**

### ✅ Stages 1-6: Supporting Classes Extracted

| Stage | Module | Lines | Status |
|-------|--------|-------|--------|
| 1 | `training/metrics.py` | 200 | ✅ COMPLETE |
| 2 | `artifacts/csv_writers.py` | 260 | ✅ COMPLETE |
| 3 | `artifacts/plot_builders.py` | 300 | ✅ COMPLETE |
| 4 | `training/checkpointing.py` | 120 | ✅ COMPLETE |
| 5 | `training/inner_loop.py` | 450 | ✅ COMPLETE |
| 6 | `training/evaluation.py` | 400 | ✅ COMPLETE |

**Total Extracted**: 1,730 lines into supporting modules ✅

---

## ❌ **What We HAVEN'T Done: The Outer Loop**

### Current `run()` Method Structure (825 lines):

```python
def run(...):  # Line 409
    # Setup (60 lines)
    - Labels processing
    - Validation
    - Logging setup
    - Channel topomap
    
    # Outer CV setup (50 lines)
    - Precompute outer fold pairs
    - Initialize accumulators
    
    # === OUTER FOLD LOOP === (600+ lines!) ===
    for fold, (tr_idx, va_idx) in enumerate(outer_pairs):  # Line ~545
        
        # Per-fold setup (50 lines)
        - Test subjects validation
        - Inner K-fold setup
        - Fold record initialization
        
        # === INNER FOLD LOOP === (200+ lines)
        for inner_fold, (inner_tr_abs, inner_va_abs) in enumerate(inner_iter):
            
            # Inner setup (80 lines)
            - Leakage validation
            - DataLoader creation
            - Model/optimizer/loss setup
            - Class weights computation
            
            # Call InnerTrainer (10 lines) ✅ EXTRACTED
            inner_result = inner_trainer.train(...)
            
            # Post-training (30 lines)
            - Save checkpoint
            - Collect inner validation predictions
            - Store inner result
        
        # Aggregate inner results (30 lines)
        - Compute mean metrics across inner folds
        
        # Select best inner model (50 lines)
        - Based on objective metric
        - Special handling for composite objective
        
        # Call OuterEvaluator (20 lines) ✅ EXTRACTED
        eval_result = outer_evaluator.evaluate(...)
        
        # Compute fold metrics (50 lines)
        - Accuracy, F1, kappa, etc.
        
        # Per-fold plotting (70 lines)
        - Generate confusion matrices
        - Generate learning curves
        - Both simple and enhanced versions
        
        # Record fold results (10 lines)
    
    # === END OUTER FOLD LOOP ===
    
    # Overall metrics computation (80 lines)
    - Aggregate across all folds
    - Compute overall statistics
    
    # Overall plotting (60 lines)
    - Overall confusion matrices
    - Both simple and enhanced versions
    
    # Write artifacts (100 lines)
    - splits_indices.json
    - Learning curves CSV
    - Outer metrics CSV
    - Test predictions CSVs
    
    # Return summary (30 lines)
```

---

## 🎯 **The Missing Extraction: OuterFoldOrchestrator**

### What Needs to be Extracted:

**NEW FILE**: `code/training/outer_loop.py` (~500 lines)

**Class**: `OuterFoldOrchestrator`

**Responsibilities:**
1. Run one complete outer fold (inner training + outer evaluation)
2. Orchestrate inner K-fold loop
3. Setup models/optimizers/losses for inner folds
4. Aggregate inner results
5. Select best inner model
6. Coordinate outer evaluation
7. Compute and return fold metrics

**What moves from `run()`:**
- Lines ~545-993 (the entire outer fold loop body, ~450 lines)
- Inner fold iteration logic
- Model/optimizer setup per inner fold
- Inner result aggregation
- Best model selection
- Fold metrics computation

**What STAYS in `run()`:**
- Outer fold iteration (the `for fold, (tr_idx, va_idx) in enumerate(outer_pairs):` loop)
- Overall setup (labels, validation, splits)
- Overall metrics aggregation
- Overall plotting
- Artifact writing
- Summary return

---

## 📋 **Revised Stage 7 Plan**

### Stage 7a: Extract OuterFoldOrchestrator (HIGH RISK)

**Goal**: Extract the outer fold loop body into `OuterFoldOrchestrator`

**Files to create:**
- `code/training/outer_loop.py` (~500 lines)

**What moves:**
- Inner fold iteration
- Model/optimizer/loss setup per inner fold
- Inner result aggregation
- Best inner model selection
- Per-fold plotting coordination

**API:**
```python
class OuterFoldOrchestrator:
    """Orchestrates one complete outer fold: inner training + outer evaluation."""
    
    def __init__(
        self,
        cfg: Dict,
        objective_computer: ObjectiveComputer,
        inner_trainer: InnerTrainer,
        outer_evaluator: OuterEvaluator,
        plot_title_builder: PlotTitleBuilder,
    ):
        self.cfg = cfg
        self.objective_computer = objective_computer
        self.inner_trainer = inner_trainer
        self.outer_evaluator = outer_evaluator
        self.plot_title_builder = plot_title_builder
    
    def run_fold(
        self,
        fold: int,
        tr_idx: np.ndarray,
        te_idx: np.ndarray,
        dataset,
        y_all: np.ndarray,
        groups: np.ndarray,
        class_names: List[str],
        model_builder: Callable,
        aug_transform,
        input_adapter: Callable | None,
        predefined_inner_splits: List[dict] | None,
        optuna_trial,
        global_step_offset: int,
    ) -> Dict:
        """
        Run one complete outer fold.
        
        Returns:
            {
                "fold": int,
                "test_subjects": list,
                "y_true": list,
                "y_pred": list,
                "metrics": {...},  # acc, macro_f1, etc.
                "inner_results": [...],
                "best_inner_result": {...},
                "fold_record": {...},  # for splits_indices.json
                "learning_curves": [...],
                "test_pred_rows_outer": [...],
                "test_pred_rows_inner": [...],
                "plots_generated": bool,
                "new_global_step": int,
            }
        """
```

**Testing:**
- Create `tests/test_outer_fold_orchestrator.py`
- Test with mock inner trainer and evaluator
- Test with minimal config (1 inner fold, 1 epoch)
- Verify all outputs match expected format

**Risk:** HIGH - Core orchestration logic, affects entire pipeline

---

### Stage 7b: Simplify `run()` to True Orchestrator (FINAL)

**Goal**: Reduce `run()` to high-level orchestration only

**What `run()` becomes (~150 lines):**
```python
def run(...):
    # Setup (30 lines)
    y_all, num_cls = self._setup_labels(...)
    _log_event = self._setup_logging()
    outer_pairs = self._setup_outer_splits(...)
    aug_transform = self._setup_augmentation(...)
    
    # Initialize orchestrators (5 lines)
    fold_orchestrator = OuterFoldOrchestrator(...)
    
    # Outer fold loop (40 lines)
    all_fold_results = []
    global_step = 0
    for fold, (tr_idx, te_idx) in enumerate(outer_pairs):
        fold_result = fold_orchestrator.run_fold(
            fold=fold,
            tr_idx=tr_idx,
            te_idx=te_idx,
            dataset=dataset,
            y_all=y_all,
            groups=groups,
            class_names=class_names,
            model_builder=model_builder,
            aug_transform=aug_transform,
            input_adapter=input_adapter,
            predefined_inner_splits=...,
            optuna_trial=optuna_trial,
            global_step_offset=global_step,
        )
        all_fold_results.append(fold_result)
        global_step = fold_result["new_global_step"]
    
    # Aggregate results (30 lines)
    summary = self._aggregate_fold_results(all_fold_results, class_names)
    
    # Generate overall artifacts (30 lines)
    self._generate_overall_plots(summary, class_names)
    self._write_all_artifacts(summary, all_fold_results)
    
    # Return summary (10 lines)
    return summary
```

**Expected line count:** ~150 lines (orchestration only)

**Testing:**
- Full integration test
- Golden output comparison
- Verify determinism

**Risk:** MEDIUM - Final integration, but logic already extracted

---

## 🔢 **Expected Outcomes**

### After Stage 7a (OuterFoldOrchestrator):
- `training_runner.py`: 1,155 lines → ~650 lines (45% reduction)
- `training/outer_loop.py`: ~500 lines (new)

### After Stage 7b (Simplify Orchestrator):
- `training_runner.py`: ~650 lines → ~400 lines (38% reduction)
- Helper methods extracted to private methods

### Final State:
- `training_runner.py`: **~400 lines** ✅ (matches plan!)
- `run()` method: **~150 lines** ✅ (orchestration only)

---

## ⚠️ **Critical Realization**

We thought we were at Stage 7, but we're actually at:
- **Stage 6 Complete** ✅
- **Stage 7a Not Started** ❌
- **Stage 7b Not Started** ❌

The outer fold orchestration is the BIGGEST remaining extraction, and it's critical for achieving the target of ~400 lines.

---

## 🚀 **Action Plan**

### Next Steps (in order):

1. **Write fail-first tests for OuterFoldOrchestrator**
   - Mock InnerTrainer, OuterEvaluator
   - Test fold result format
   - Test inner iteration
   - Test metrics aggregation

2. **Create `code/training/outer_loop.py`**
   - Implement `OuterFoldOrchestrator` class
   - Extract fold loop body from `run()`
   - Ensure all behaviors preserved

3. **Update `training_runner.py`**
   - Import `OuterFoldOrchestrator`
   - Replace fold loop body with `fold_orchestrator.run_fold()`
   - Keep outer loop structure

4. **Verify integration**
   - Run test with real config
   - Compare outputs to golden reference
   - Check all artifacts generated

5. **Final simplification (Stage 7b)**
   - Extract helper methods
   - Simplify `run()` to true orchestrator
   - Final testing

---

**Status**: Ready to proceed with Stage 7a: OuterFoldOrchestrator extraction

**Estimated Time**: 2-3 hours (this is the big one!)

**Complexity**: HIGH (touches the heart of the training pipeline)

**Reward**: Transforms monolithic orchestrator into clean, modular architecture

