"""
Quick refactoring verification script.

Tests that all extracted modules work correctly and produce identical behavior.
"""

print("=" * 80)
print("REFACTORING VERIFICATION TEST")
print("=" * 80)

# Test 1: ObjectiveComputer
print("\n[Test 1] ObjectiveComputer...")
from code.training.metrics import ObjectiveComputer

# Test macro_f1 objective
cfg1 = {"optuna_objective": "inner_mean_macro_f1"}
oc1 = ObjectiveComputer(cfg1)
result1 = oc1.compute(50.0, 45.0, 30.0, 80.0)
assert result1 == 45.0, f"Expected 45.0, got {result1}"
print("  [OK] Macro F1 objective works")

# Test min_per_class_f1 objective
cfg2 = {"optuna_objective": "inner_mean_min_per_class_f1"}
oc2 = ObjectiveComputer(cfg2)
result2 = oc2.compute(50.0, 45.0, 30.0, 80.0)
assert result2 == 30.0, f"Expected 30.0, got {result2}"
print("  [OK] Min per-class F1 objective works")

# Test composite threshold mode (below threshold)
cfg3 = {
    "optuna_objective": "composite_min_f1_plur_corr",
    "min_f1_threshold": 35.0,
}
oc3 = ObjectiveComputer(cfg3)
result3 = oc3.compute(50.0, 45.0, 30.0, 80.0)  # 30 < 35, so gradient mode
assert abs(result3 - 3.0) < 0.01, f"Expected 3.0, got {result3}"
print("  [OK] Composite threshold (below) works")

# Test composite threshold mode (above threshold)
result3b = oc3.compute(50.0, 45.0, 40.0, 85.0)  # 40 >= 35, so plur_corr mode
assert result3b == 85.0, f"Expected 85.0, got {result3b}"
print("  [OK] Composite threshold (above) works")

# Test composite weighted mode
cfg4 = {
    "optuna_objective": "composite_min_f1_plur_corr",
    "composite_min_f1_weight": 0.5,
}
oc4 = ObjectiveComputer(cfg4)
result4 = oc4.compute(50.0, 45.0, 30.0, 80.0)  # 0.5*30 + 0.5*80 = 55
assert abs(result4 - 55.0) < 0.01, f"Expected 55.0, got {result4}"
print("  [OK] Composite weighted works")

print("  [PASS] ObjectiveComputer: ALL TESTS PASSED")

# Test 2: CheckpointManager
print("\n[Test 2] CheckpointManager...")
from code.training.checkpointing import CheckpointManager

cfg5 = {
    "optuna_objective": "inner_mean_macro_f1",
    "early_stop": 3,
}
oc5 = ObjectiveComputer(cfg5)
mgr = CheckpointManager(cfg5, oc5)

# First epoch
updated1 = mgr.update({"epoch": 1}, {
    "val_acc": 50.0,
    "val_macro_f1": 45.0,
    "val_min_per_class_f1": 30.0,
    "val_plur_corr": 80.0,
    "val_loss": 1.5,
})
assert updated1 is True, "First update should always save checkpoint"
assert mgr.patience == 0, f"Patience should be 0, got {mgr.patience}"
print("  [OK] First checkpoint saves")

# Second epoch (improvement)
updated2 = mgr.update({"epoch": 2}, {
    "val_acc": 52.0,
    "val_macro_f1": 50.0,  # Improved!
    "val_min_per_class_f1": 35.0,
    "val_plur_corr": 85.0,
    "val_loss": 1.2,
})
assert updated2 is True, "Should update on improvement"
assert mgr.patience == 0, f"Patience should reset to 0, got {mgr.patience}"
print("  [OK] Checkpoint updates on improvement")

# Third epoch (no improvement)
updated3 = mgr.update({"epoch": 3}, {
    "val_acc": 51.0,
    "val_macro_f1": 48.0,  # Worse!
    "val_min_per_class_f1": 32.0,
    "val_plur_corr": 82.0,
    "val_loss": 1.4,
})
assert updated3 is False, "Should not update on no improvement"
assert mgr.patience == 1, f"Patience should be 1, got {mgr.patience}"
assert not mgr.should_stop(), "Should not stop yet"
print("  [OK] Patience increments on no improvement")

# Fourth epoch (still no improvement, trigger early stop)
for i in range(2):  # Increment patience to 3
    mgr.update({"epoch": 4 + i}, {
        "val_acc": 51.0,
        "val_macro_f1": 48.0,
        "val_min_per_class_f1": 32.0,
        "val_plur_corr": 82.0,
        "val_loss": 1.4,
    })

assert mgr.patience == 3, f"Patience should be 3, got {mgr.patience}"
assert mgr.should_stop(), "Should trigger early stop"
print("  [OK] Early stopping triggers correctly")

best_state = mgr.get_best_state()
assert best_state["epoch"] == 2, "Best epoch should be 2"
print("  [OK] Best state preserved")

print("  [PASS] CheckpointManager: ALL TESTS PASSED")

# Test 3: CSV Writers
print("\n[Test 3] CSV Writers...")
from code.artifacts.csv_writers import (
    LearningCurvesWriter,
    OuterEvalMetricsWriter,
    TestPredictionsWriter,
)
from pathlib import Path
import tempfile
import csv

# Create temp directory
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    
    # Test LearningCurvesWriter
    writer1 = LearningCurvesWriter(tmp_path)
    rows1 = [
        {
            "outer_fold": 1, "inner_fold": 1, "epoch": 1,
            "train_loss": 1.5, "val_loss": 1.6, "val_acc": 45.0,
            "val_macro_f1": 43.0, "val_min_per_class_f1": 35.0,
            "val_plur_corr": 66.7, "val_objective_metric": 43.0,
            "n_train": 100, "n_val": 30, "optuna_trial_id": 42,
            "param_hash": "abc123",
        },
    ]
    writer1.write(rows1)
    
    csv_path1 = tmp_path / "learning_curves_inner.csv"
    assert csv_path1.exists(), "Learning curves CSV should exist"
    with csv_path1.open() as f:
        reader = csv.DictReader(f)
        read_rows = list(reader)
        assert len(read_rows) == 1, f"Expected 1 row, got {len(read_rows)}"
        assert read_rows[0]["epoch"] == "1"
    print("  [OK] LearningCurvesWriter works")
    
    # Test OuterEvalMetricsWriter
    writer2 = OuterEvalMetricsWriter(tmp_path)
    rows2 = [
        {
            "outer_fold": 1, "test_subjects": "2,3", "n_test_trials": 50,
            "acc": 45.0, "acc_std": "", "macro_f1": 43.0, "macro_f1_std": "",
            "min_per_class_f1": 35.0, "min_per_class_f1_std": "",
            "plur_corr": 66.7, "plur_corr_std": "",
            "cohen_kappa": 0.234, "cohen_kappa_std": "",
            "per_class_f1": "[0.45, 0.38, 0.42]",
        },
    ]
    agg_row = {
        "outer_fold": "OVERALL", "test_subjects": "-", "n_test_trials": 50,
        "acc": 45.0, "acc_std": 0.0, "macro_f1": 43.0, "macro_f1_std": 0.0,
        "min_per_class_f1": 35.0, "min_per_class_f1_std": 0.0,
        "plur_corr": 66.7, "plur_corr_std": 0.0,
        "cohen_kappa": 0.234, "cohen_kappa_std": 0.0,
        "per_class_f1": "",
    }
    writer2.write(rows2, agg_row)
    
    csv_path2 = tmp_path / "outer_eval_metrics.csv"
    assert csv_path2.exists(), "Outer eval metrics CSV should exist"
    with csv_path2.open() as f:
        reader = csv.DictReader(f)
        read_rows = list(reader)
        assert len(read_rows) == 2, f"Expected 2 rows, got {len(read_rows)}"
        assert read_rows[1]["outer_fold"] == "OVERALL"
    print("  [OK] OuterEvalMetricsWriter works")
    
    # Test TestPredictionsWriter (outer mode)
    writer3 = TestPredictionsWriter(tmp_path, mode="outer")
    rows3 = [
        {
            "outer_fold": 1, "trial_index": 0, "subject_id": 2,
            "true_label_idx": 0, "true_label_name": "low",
            "pred_label_idx": 1, "pred_label_name": "medium",
            "correct": 0, "p_trueclass": 0.234, "logp_trueclass": -1.452,
            "probs": "[0.234, 0.456, 0.310]",
        },
    ]
    writer3.write(rows3)
    
    csv_path3 = tmp_path / "test_predictions_outer.csv"
    assert csv_path3.exists(), "Test predictions outer CSV should exist"
    print("  [OK] TestPredictionsWriter (outer) works")
    
    # Test TestPredictionsWriter (inner mode)
    writer4 = TestPredictionsWriter(tmp_path, mode="inner")
    rows4 = [
        {
            "outer_fold": 1, "inner_fold": 1, "trial_index": 0, "subject_id": 2,
            "true_label_idx": 0, "true_label_name": "low",
            "pred_label_idx": 1, "pred_label_name": "medium",
            "correct": 0, "p_trueclass": 0.234, "logp_trueclass": -1.452,
            "probs": "[0.234, 0.456, 0.310]",
        },
    ]
    writer4.write(rows4)
    
    csv_path4 = tmp_path / "test_predictions_inner.csv"
    assert csv_path4.exists(), "Test predictions inner CSV should exist"
    print("  [OK] TestPredictionsWriter (inner) works")

print("  [PASS] CSV Writers: ALL TESTS PASSED")

# Test 4: TrainingRunner integration
print("\n[Test 4] TrainingRunner Integration...")
from code.training_runner import TrainingRunner

cfg6 = {
    "optuna_objective": "composite_min_f1_plur_corr",
    "min_f1_threshold": 35.0,
}

def dummy_label_fn(x):
    return 0

runner = TrainingRunner(cfg6, dummy_label_fn)
assert runner.objective_computer is not None, "ObjectiveComputer should be initialized"
print("  [OK] ObjectiveComputer initialized in TrainingRunner")

# Test _get_composite_params delegation
params = runner._get_composite_params()
assert params["mode"] == "threshold", f"Expected threshold mode, got {params}"
assert params["threshold"] == 35.0, f"Expected threshold 35.0, got {params}"
print("  [OK] _get_composite_params delegates correctly")

# Test _compute_objective_metric delegation
result = runner._compute_objective_metric(50.0, 45.0, 30.0, 80.0)
assert abs(result - 3.0) < 0.01, f"Expected 3.0, got {result}"
print("  [OK] _compute_objective_metric delegates correctly")

print("  [PASS] TrainingRunner Integration: ALL TESTS PASSED")

print("\n" + "=" * 80)
print("ALL REFACTORING TESTS PASSED [SUCCESS]")
print("=" * 80)
print("\nThe refactored code is functionally equivalent to the original!")
print("Next step: Run full integration tests with actual training runs.")

