# Decision Layer Reporting Enhancement - Implementation Status

## Overview

This document tracks the implementation of enhanced decision layer reporting as suggested by your consultant. The enhancements add comprehensive baseline vs thresholded comparisons, statistical tests, and side-by-side plots to make the decision layer evaluation transparent and scientifically rigorous.

## ✅ Completed (Phases 1-3)

### Phase 1: TDD Tests ✅
- **File:** `tests/test_decision_layer_reporting.py`
- **Status:** Complete (253 lines)
- **Coverage:**
  - ✅ Metrics computation from prediction rows
  - ✅ Fold-level comparison (baseline vs thresholded)
  - ✅ McNemar test for paired predictions
  - ✅ Paired t-test across folds
  - ✅ TXT report formatting
  - ✅ JSON summary fields
  - ✅ Comparison plot generation

### Phase 2: Core Functions ✅
- **File:** `code/posthoc/decision_layer.py` (extended with 330 new lines)
- **Status:** Complete
- **New Functions:**
  - ✅ `compute_metrics_from_rows()` - Compute acc/F1/kappa/plur_corr from rows
  - ✅ `compute_fold_comparison()` - Compare baseline vs thresholded for one fold
  - ✅ `mcnemar_test()` - Statistical test for paired predictions
  - ✅ `paired_ttest()` - Paired t-test across folds
  - ✅ `format_decision_layer_txt_section()` - Format consultant template for TXT report
  - ✅ `build_decision_layer_json_fields()` - Build consultant schema for JSON

### Phase 3: Plotting Module ✅
- **File:** `code/posthoc/decision_layer_plots.py`
- **Status:** Complete (162 lines)
- **Functions:**
  - ✅ `create_confusion_comparison_plot()` - Side-by-side confusion matrices
  - ✅ `create_per_fold_comparison_bars()` - Per-fold accuracy bar chart
  - ✅ `generate_all_comparison_plots()` - Orchestrate all plots

## 🚧 Remaining Work (Phases 4-6)

### Phase 4: Extend Orchestration Function ⏳
- **File:** `code/posthoc/decision_layer.py`
- **Function:** `tune_and_apply_decision_layer()` (line 386)
- **Required Changes:**
  1. After writing thresholded CSV, load both baseline and thresholded CSVs
  2. Compute per-fold comparisons using `compute_fold_comparison()`
  3. Aggregate overall metrics (baseline & thresholded)
  4. Run statistical tests (McNemar on all trials, paired t-test on fold accuracies)
  5. Build enriched stats dict with all data
  6. Call `generate_all_comparison_plots()` to create plots
  7. **Return the enriched stats dict** (currently returns None)

**Implementation Pseudocode:**
```python
def tune_and_apply_decision_layer(...) -> Dict | None:  # Change return type
    try:
        # ... existing tuning/application logic ...
        
        # NEW: After writing CSVs, compute enriched stats
        from code.posthoc.decision_layer_plots import generate_all_comparison_plots
        
        # Load baseline CSV
        baseline_csv = run_dir / "test_predictions_outer.csv"
        baseline_rows = _load_csv_rows(baseline_csv)
        
        # Compute per-fold comparisons
        fold_comparisons = []
        for fold in sorted(outer_by_fold.keys()):
            baseline_fold = [r for r in baseline_rows if r["outer_fold"] == fold]
            thresholded_fold = outer_by_fold[fold]  # Already have this
            
            comparison = compute_fold_comparison(
                baseline_rows=baseline_fold,
                thresholded_rows=thresholded_fold,
                fold=fold,
                num_classes=len(class_names)
            )
            fold_comparisons.append(comparison)
        
        # Aggregate overall metrics
        baseline_overall = compute_metrics_from_rows(baseline_rows, len(class_names))
        thresholded_overall = compute_metrics_from_rows(all_thresholded_rows, len(class_names))
        
        # Compute deltas
        deltas = {
            key: thresholded_overall[key] - baseline_overall[key]
            for key in ["acc", "macro_f1", "min_per_class_f1", "plur_corr", "cohen_kappa"]
        }
        
        # Statistical tests
        baseline_correct = [int(r["correct"]) for r in baseline_rows]
        thresholded_correct = [int(r["correct"]) for r in all_thresholded_rows]
        mcnemar_result = mcnemar_test(baseline_correct, thresholded_correct)
        
        baseline_fold_accs = [fc["baseline"]["acc"] for fc in fold_comparisons]
        thresholded_fold_accs = [fc["thresholded"]["acc"] for fc in fold_comparisons]
        ttest_result = paired_ttest(baseline_fold_accs, thresholded_fold_accs)
        
        # Build enriched stats
        enriched_stats = {
            "baseline": {"overall": baseline_overall},
            "thresholded": {"overall": thresholded_overall},
            "deltas": deltas,
            "config": thresholds_data["config"],
            "per_fold": [
                {
                    "fold": fold,
                    "theta": thresholds_data["folds"][str(fold)]["theta"],
                    "n_activated": thresholds_data["folds"][str(fold)]["n_activated"],
                    "n_outer_trials": thresholds_data["folds"][str(fold)]["n_outer_trials"],
                    "activation_rate": thresholds_data["folds"][str(fold)]["activation_rate"],
                }
                for fold in sorted(outer_by_fold.keys())
            ],
            "overall_activation_rate": sum(...) / sum(...),
            "total_activated_trials": sum(...),
            "total_test_trials": len(all_thresholded_rows),
            "statistical_tests": {
                "mcnemar_chi2": mcnemar_result["chi2"],
                "mcnemar_p": mcnemar_result["p_value"],
                "paired_t_statistic": ttest_result["t_statistic"],
                "paired_t_p": ttest_result["p_value"],
                "paired_t_df": ttest_result["df"],
            }
        }
        
        # Generate comparison plots
        plots_dir = run_dir / "plots_outer_threshold_compare"
        generate_all_comparison_plots(
            baseline_rows=baseline_rows,
            thresholded_rows=all_thresholded_rows,
            fold_comparisons=fold_comparisons,
            class_names=class_names,
            out_dir=plots_dir
        )
        log_event("decision_layer_plots_written", f"Wrote comparison plots to {plots_dir}")
        
        # Return enriched stats for use by summary writer
        return enriched_stats
        
    except Exception as e:
        log_event("decision_layer_failed", f"Decision layer failed: {e}")
        return None
```

**Helper Function Needed:**
```python
def _load_csv_rows(csv_path: Path) -> List[Dict]:
    """Load prediction rows from CSV."""
    import csv
    import json
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["outer_fold"] = int(row["outer_fold"])
            row["true_label_idx"] = int(row["true_label_idx"])
            row["pred_label_idx"] = int(row["pred_label_idx"])
            row["correct"] = int(row["correct"])
            # probs is already JSON string
            rows.append(row)
    return rows
```

### Phase 5: Update Artifact Writer ⏳
- **File:** `code/artifacts/artifact_writer.py`
- **Function:** `write_test_predictions()` (around line 200)
- **Required Changes:**
  1. Capture return value from `tune_and_apply_decision_layer()`
  2. Store it in the orchestrator for later use by summary writer

**Implementation:**
```python
# In ArtifactWriterOrchestrator class
def __init__(...):
    ...
    self.decision_layer_stats = None  # Add this field

def write_test_predictions(...):
    ...
    if dl_cfg.get("enable"):
        # Existing call, but now capture return value
        self.decision_layer_stats = tune_and_apply_decision_layer(...)
```

### Phase 6: Extend Summary Writer ⏳
- **File:** `utils/summary.py`
- **Function:** `write_summary()` (around line 104)
- **Required Changes:**
  1. Check if `decision_layer/thresholds.json` exists
  2. If yes, load enriched stats (either from run_dir or from summary dict if passed)
  3. Call `format_decision_layer_txt_section()` and append to TXT report
  4. Call `build_decision_layer_json_fields()` and merge into JSON summary

**Implementation:**
```python
def write_summary(run_dir: Path, summary: dict, task: str, engine: str):
    ...
    
    # After writing outer eval metrics section in TXT report
    # Check for decision layer
    decision_layer_dir = run_dir / "decision_layer"
    thresholds_json = decision_layer_dir / "thresholds.json"
    
    if thresholds_json.exists():
        # Load enriched stats (either from summary dict or compute from CSVs)
        decision_layer_stats = summary.get("decision_layer_stats")
        
        if not decision_layer_stats:
            # Fallback: compute from CSVs (for old runs)
            from code.posthoc import decision_layer as dl
            baseline_csv = run_dir / "test_predictions_outer.csv"
            thresholded_csv = run_dir / "test_predictions_outer_thresholded.csv"
            
            if baseline_csv.exists() and thresholded_csv.exists():
                baseline_rows = dl._load_csv_rows(baseline_csv)
                thresholded_rows = dl._load_csv_rows(thresholded_csv)
                num_classes = len(summary.get("class_names", []))
                
                # Compute enriched stats (simplified version)
                baseline_overall = dl.compute_metrics_from_rows(baseline_rows, num_classes)
                thresholded_overall = dl.compute_metrics_from_rows(thresholded_rows, num_classes)
                deltas = {
                    key: thresholded_overall[key] - baseline_overall[key]
                    for key in ["acc", "macro_f1", "min_per_class_f1", "plur_corr", "cohen_kappa"]
                }
                
                # Load thresholds
                thresholds_data = json.loads(thresholds_json.read_text())
                
                decision_layer_stats = {
                    "baseline": {"overall": baseline_overall},
                    "thresholded": {"overall": thresholded_overall},
                    "deltas": deltas,
                    "config": thresholds_data["config"],
                    "per_fold": [
                        {
                            "fold": int(k),
                            "theta": v["theta"],
                            "n_activated": v["n_activated"],
                            "n_outer_trials": v["n_outer_trials"],
                            "activation_rate": v["activation_rate"],
                        }
                        for k, v in thresholds_data["folds"].items()
                    ],
                    "overall_activation_rate": sum(v["n_activated"] for v in thresholds_data["folds"].values()) / 
                                              sum(v["n_outer_trials"] for v in thresholds_data["folds"].values()),
                    "total_activated_trials": sum(v["n_activated"] for v in thresholds_data["folds"].values()),
                    "total_test_trials": len(thresholded_rows),
                }
        
        # Add decision layer section to TXT report
        from code.posthoc import decision_layer as dl
        dl_lines = dl.format_decision_layer_txt_section(decision_layer_stats)
        report_lines.extend(dl_lines)
        
        # Merge decision layer fields into JSON summary
        dl_json_fields = dl.build_decision_layer_json_fields(decision_layer_stats)
        summary.update(dl_json_fields)
    
    # Continue with rest of summary writing...
```

## Testing Strategy

1. **Unit Tests:** Run `pytest tests/test_decision_layer_reporting.py -v`
2. **Integration Test:** Run a small training job with decision layer enabled
3. **Verify Artifacts:**
   - Check `plots_outer_threshold_compare/` exists with 3+ plots
   - Check TXT report has decision layer section with consultant template
   - Check JSON has `decision_layer` nested object and `*_argmax_baseline` fields
   - Check statistical tests appear in both TXT and JSON

## Expected Output Example

### TXT Report Addition:
```
================================================================================
DECISION LAYER ANALYSIS (Ordinal Adjacent-Pair Refinement)
================================================================================

Baseline (Argmax):
  Overall Accuracy:        48.2% (±3.1%)
  Macro F1:                47.8%
  Min Per-Class F1:        38.5%
  Plurality Correctness:   66.7%
  Cohen's Kappa:           0.223

Decision Layer (Ratio Rule):
  Overall Accuracy:        51.7% (±2.9%)    [+3.5 pp]
  Macro F1:                51.2%            [+3.4 pp]
  Min Per-Class F1:        42.1%            [+3.6 pp]
  Plurality Correctness:   100.0%           [+33.3 pp]
  Cohen's Kappa:           0.276            [+0.053]

Decision Layer Configuration:
  Metric Optimized:        composite_min_f1_plur_corr
  Theta Grid:              0.30 → 0.70 (step 0.01)
  Min Activation:          50 trials
  
Per-Fold Thresholds:
  Fold 1: θ=0.48, activated on 127/412 trials (30.8%)
  ...

Statistical Comparison (Argmax vs Decision Layer):
  McNemar Test (2↔3 errors):  χ²=14.7, p=0.0001
  Paired t-test (per-fold):   t(5)=4.2, p=0.008
  
  The decision layer provides statistically significant improvement over baseline.
```

### JSON Fields Addition:
```json
{
  "mean_acc": 51.7,
  "mean_acc_argmax_baseline": 48.2,
  "mean_acc_delta": 3.5,
  ...
  "decision_layer": {
    "enabled": true,
    "metric_optimized": "composite_min_f1_plur_corr",
    "overall_activation_rate": 0.345,
    "per_fold_thetas": {"fold_1": 0.48, "fold_2": 0.52, ...},
    "statistical_tests": {
      "mcnemar_chi2": 14.7,
      "mcnemar_p": 0.0001,
      ...
    }
  }
}
```

## Files Modified Summary

1. ✅ `tests/test_decision_layer_reporting.py` (new, 253 lines)
2. ✅ `code/posthoc/decision_layer.py` (extended, +330 lines)
3. ✅ `code/posthoc/decision_layer_plots.py` (new, 162 lines)
4. ⏳ `code/artifacts/artifact_writer.py` (minor change: capture return value)
5. ⏳ `utils/summary.py` (extend with decision layer section)

## Next Steps

1. Implement Phase 4 (extend orchestration function)
2. Implement Phase 5 (update artifact writer)
3. Implement Phase 6 (extend summary writer)
4. Run integration test
5. Verify all artifacts
6. Update README with new artifacts documentation

## Notes

- All functions follow constitutional compliance (deterministic, leak-free, auditable)
- Statistical tests use scipy.stats for rigor
- Plots use matplotlib with consistent styling
- CSV loading is resilient (handles missing fields gracefully)
- Fallback logic ensures old runs without enriched stats still get basic reporting

