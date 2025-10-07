"""
Tests for CSV writer classes (artifacts/csv_writers.py).

These tests will FAIL until we extract the CSV writer classes.
They verify that CSV output format is preserved exactly.

Constitutional compliance: Section V (Audit-Ready Artifact Retention)
"""
import pytest
import csv
from pathlib import Path


def test_csv_writers_imports():
    """Test that CSV writer classes can be imported (will fail pre-refactor)."""
    try:
        from code.artifacts.csv_writers import (
            LearningCurvesWriter,
            OuterEvalMetricsWriter,
            TestPredictionsWriter,
        )
        assert LearningCurvesWriter is not None
        assert OuterEvalMetricsWriter is not None
        assert TestPredictionsWriter is not None
    except ImportError as e:
        pytest.fail(f"Cannot import CSV writers: {e}")


def test_learning_curves_writer_creates_csv(tmp_path):
    """Test that LearningCurvesWriter creates CSV with correct format."""
    from code.artifacts.csv_writers import LearningCurvesWriter
    
    writer = LearningCurvesWriter(tmp_path)
    
    rows = [
        {
            "outer_fold": 1,
            "inner_fold": 1,
            "epoch": 1,
            "train_loss": 1.234,
            "val_loss": 1.456,
            "val_acc": 45.67,
            "val_macro_f1": 43.21,
            "val_min_per_class_f1": 35.89,
            "val_plur_corr": 66.67,
            "val_objective_metric": 43.21,
            "n_train": 100,
            "n_val": 30,
            "optuna_trial_id": 42,
            "param_hash": "abc123",
        },
        {
            "outer_fold": 1,
            "inner_fold": 1,
            "epoch": 2,
            "train_loss": 1.123,
            "val_loss": 1.345,
            "val_acc": 47.89,
            "val_macro_f1": 45.67,
            "val_min_per_class_f1": 38.12,
            "val_plur_corr": 66.67,
            "val_objective_metric": 45.67,
            "n_train": 100,
            "n_val": 30,
            "optuna_trial_id": 42,
            "param_hash": "abc123",
        },
    ]
    
    writer.write(rows)
    
    # Verify file exists
    csv_path = tmp_path / "learning_curves_inner.csv"
    assert csv_path.exists()
    
    # Verify content
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Check header
        expected_fields = [
            "outer_fold", "inner_fold", "epoch",
            "train_loss", "val_loss", "val_acc",
            "val_macro_f1", "val_min_per_class_f1",
            "val_plur_corr", "val_objective_metric",
            "n_train", "n_val",
            "optuna_trial_id", "param_hash",
        ]
        assert fieldnames == expected_fields
        
        # Check rows
        read_rows = list(reader)
        assert len(read_rows) == 2
        assert read_rows[0]["outer_fold"] == "1"
        assert read_rows[0]["epoch"] == "1"
        assert read_rows[1]["epoch"] == "2"


def test_learning_curves_writer_empty_rows(tmp_path):
    """Test that empty rows don't create CSV."""
    from code.artifacts.csv_writers import LearningCurvesWriter
    
    writer = LearningCurvesWriter(tmp_path)
    writer.write([])  # Empty list
    
    csv_path = tmp_path / "learning_curves_inner.csv"
    # Should not create file or should be empty
    if csv_path.exists():
        with csv_path.open() as f:
            content = f.read()
            assert content == "" or content.startswith("outer_fold,")


def test_outer_eval_metrics_writer_creates_csv(tmp_path):
    """Test that OuterEvalMetricsWriter creates CSV with correct format."""
    from code.artifacts.csv_writers import OuterEvalMetricsWriter
    
    writer = OuterEvalMetricsWriter(tmp_path)
    
    rows = [
        {
            "outer_fold": 1,
            "test_subjects": "2,3",
            "n_test_trials": 50,
            "acc": 45.67,
            "acc_std": "",
            "macro_f1": 43.21,
            "macro_f1_std": "",
            "min_per_class_f1": 35.89,
            "min_per_class_f1_std": "",
            "plur_corr": 66.67,
            "plur_corr_std": "",
            "cohen_kappa": 0.234,
            "cohen_kappa_std": "",
            "per_class_f1": "[0.45, 0.38, 0.42]",
        },
        {
            "outer_fold": 2,
            "test_subjects": "4,5",
            "n_test_trials": 48,
            "acc": 48.23,
            "acc_std": "",
            "macro_f1": 46.54,
            "macro_f1_std": "",
            "min_per_class_f1": 38.91,
            "min_per_class_f1_std": "",
            "plur_corr": 66.67,
            "plur_corr_std": "",
            "cohen_kappa": 0.267,
            "cohen_kappa_std": "",
            "per_class_f1": "[0.48, 0.41, 0.45]",
        },
    ]
    
    aggregate_row = {
        "outer_fold": "OVERALL",
        "test_subjects": "-",
        "n_test_trials": 98,
        "acc": 46.95,
        "acc_std": 1.28,
        "macro_f1": 44.88,
        "macro_f1_std": 1.67,
        "min_per_class_f1": 37.40,
        "min_per_class_f1_std": 1.51,
        "plur_corr": 66.67,
        "plur_corr_std": 0.0,
        "cohen_kappa": 0.251,
        "cohen_kappa_std": 0.017,
        "per_class_f1": "",
    }
    
    writer.write(rows, aggregate_row)
    
    # Verify file exists
    csv_path = tmp_path / "outer_eval_metrics.csv"
    assert csv_path.exists()
    
    # Verify content
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Check header
        expected_fields = [
            "outer_fold", "test_subjects", "n_test_trials",
            "acc", "acc_std",
            "macro_f1", "macro_f1_std",
            "min_per_class_f1", "min_per_class_f1_std",
            "plur_corr", "plur_corr_std",
            "cohen_kappa", "cohen_kappa_std",
            "per_class_f1",
        ]
        assert fieldnames == expected_fields
        
        # Check rows
        read_rows = list(reader)
        assert len(read_rows) == 3  # 2 folds + 1 aggregate
        assert read_rows[0]["outer_fold"] == "1"
        assert read_rows[1]["outer_fold"] == "2"
        assert read_rows[2]["outer_fold"] == "OVERALL"


def test_test_predictions_writer_outer_mode(tmp_path):
    """Test TestPredictionsWriter in outer mode."""
    from code.artifacts.csv_writers import TestPredictionsWriter
    
    writer = TestPredictionsWriter(tmp_path, mode="outer")
    
    rows = [
        {
            "outer_fold": 1,
            "trial_index": 0,
            "subject_id": 2,
            "true_label_idx": 0,
            "true_label_name": "low",
            "pred_label_idx": 1,
            "pred_label_name": "medium",
            "correct": 0,
            "p_trueclass": 0.234,
            "logp_trueclass": -1.452,
            "probs": "[0.234, 0.456, 0.310]",
        },
    ]
    
    writer.write(rows)
    
    # Verify file exists
    csv_path = tmp_path / "test_predictions_outer.csv"
    assert csv_path.exists()
    
    # Verify fieldnames
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        expected_fields = [
            "outer_fold",
            "trial_index",
            "subject_id",
            "true_label_idx",
            "true_label_name",
            "pred_label_idx",
            "pred_label_name",
            "correct",
            "p_trueclass",
            "logp_trueclass",
            "probs",
        ]
        assert fieldnames == expected_fields


def test_test_predictions_writer_inner_mode(tmp_path):
    """Test TestPredictionsWriter in inner mode."""
    from code.artifacts.csv_writers import TestPredictionsWriter
    
    writer = TestPredictionsWriter(tmp_path, mode="inner")
    
    rows = [
        {
            "outer_fold": 1,
            "inner_fold": 1,
            "trial_index": 0,
            "subject_id": 2,
            "true_label_idx": 0,
            "true_label_name": "low",
            "pred_label_idx": 1,
            "pred_label_name": "medium",
            "correct": 0,
            "p_trueclass": 0.234,
            "logp_trueclass": -1.452,
            "probs": "[0.234, 0.456, 0.310]",
        },
    ]
    
    writer.write(rows)
    
    # Verify file exists
    csv_path = tmp_path / "test_predictions_inner.csv"
    assert csv_path.exists()
    
    # Verify fieldnames (inner has extra "inner_fold" field)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        expected_fields = [
            "outer_fold",
            "inner_fold",  # Extra field for inner mode!
            "trial_index",
            "subject_id",
            "true_label_idx",
            "true_label_name",
            "pred_label_idx",
            "pred_label_name",
            "correct",
            "p_trueclass",
            "logp_trueclass",
            "probs",
        ]
        assert fieldnames == expected_fields


def test_csv_writers_handle_config_toggles(tmp_path):
    """Test that CSV writers respect config toggles (write_*_csv flags)."""
    from code.artifacts.csv_writers import LearningCurvesWriter
    
    # This test verifies the expected behavior when toggles are False
    # The actual toggle logic is in training_runner, but writers should
    # handle empty rows gracefully
    
    writer = LearningCurvesWriter(tmp_path)
    writer.write([])  # Empty when toggle is False
    
    csv_path = tmp_path / "learning_curves_inner.csv"
    # File should not exist or be empty
    if csv_path.exists():
        assert csv_path.stat().st_size == 0 or csv_path.read_text().strip() == ""

