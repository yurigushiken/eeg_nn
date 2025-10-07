"""
Tests for ObjectiveComputer class (training/metrics.py).

These tests will FAIL until we extract the ObjectiveComputer class.
They verify that objective computation behavior is preserved exactly.

Constitutional compliance: Section III (Deterministic Training)
"""
import pytest
import numpy as np


def test_objective_computer_imports():
    """Test that ObjectiveComputer can be imported (will fail pre-refactor)."""
    try:
        from code.training.metrics import ObjectiveComputer
        assert ObjectiveComputer is not None
    except ImportError as e:
        pytest.fail(f"Cannot import ObjectiveComputer: {e}")


def test_objective_computer_init_threshold_mode():
    """Test ObjectiveComputer initialization with threshold mode."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "min_f1_threshold": 35.0,
    }
    
    computer = ObjectiveComputer(cfg)
    
    assert computer.objective == "composite_min_f1_plur_corr"
    params = computer.get_params()
    assert params["mode"] == "threshold"
    assert params["threshold"] == 35.0


def test_objective_computer_init_weighted_mode():
    """Test ObjectiveComputer initialization with weighted mode."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "composite_min_f1_weight": 0.6,
    }
    
    computer = ObjectiveComputer(cfg)
    
    params = computer.get_params()
    assert params["mode"] == "weighted"
    assert params["weight"] == 0.6


def test_objective_computer_ambiguous_config_fails():
    """Test that specifying both threshold and weight raises error."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "min_f1_threshold": 35.0,
        "composite_min_f1_weight": 0.5,
    }
    
    with pytest.raises(ValueError, match="Ambiguous config"):
        ObjectiveComputer(cfg)


def test_objective_computer_missing_params_fails():
    """Test that missing both threshold and weight raises error."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
    }
    
    with pytest.raises(ValueError, match="requires either"):
        ObjectiveComputer(cfg)


def test_objective_computer_compute_macro_f1():
    """Test computation for inner_mean_macro_f1 objective."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_macro_f1"}
    computer = ObjectiveComputer(cfg)
    
    result = computer.compute(
        val_acc=50.0,
        val_macro_f1=45.5,
        val_min_per_class_f1=30.0,
        val_plur_corr=80.0,
    )
    
    assert result == 45.5


def test_objective_computer_compute_min_per_class_f1():
    """Test computation for inner_mean_min_per_class_f1 objective."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_min_per_class_f1"}
    computer = ObjectiveComputer(cfg)
    
    result = computer.compute(
        val_acc=50.0,
        val_macro_f1=45.5,
        val_min_per_class_f1=30.0,
        val_plur_corr=80.0,
    )
    
    assert result == 30.0


def test_objective_computer_compute_acc():
    """Test computation for inner_mean_acc objective."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_acc"}
    computer = ObjectiveComputer(cfg)
    
    result = computer.compute(
        val_acc=50.0,
        val_macro_f1=45.5,
        val_min_per_class_f1=30.0,
        val_plur_corr=80.0,
    )
    
    assert result == 50.0


def test_objective_computer_compute_plur_corr():
    """Test computation for inner_mean_plur_corr objective."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_plur_corr"}
    computer = ObjectiveComputer(cfg)
    
    result = computer.compute(
        val_acc=50.0,
        val_macro_f1=45.5,
        val_min_per_class_f1=30.0,
        val_plur_corr=80.0,
    )
    
    assert result == 80.0


def test_objective_computer_composite_threshold_below():
    """Test composite objective with threshold mode (below threshold)."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "min_f1_threshold": 38.0,
    }
    computer = ObjectiveComputer(cfg)
    
    # min_f1=30.0 < threshold=38.0 → return gradient (30.0 * 0.1 = 3.0)
    result = computer.compute(
        val_acc=50.0,
        val_macro_f1=45.5,
        val_min_per_class_f1=30.0,
        val_plur_corr=80.0,
    )
    
    assert result == pytest.approx(3.0)


def test_objective_computer_composite_threshold_above():
    """Test composite objective with threshold mode (above threshold)."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "min_f1_threshold": 38.0,
    }
    computer = ObjectiveComputer(cfg)
    
    # min_f1=40.0 >= threshold=38.0 → return plur_corr=80.0
    result = computer.compute(
        val_acc=50.0,
        val_macro_f1=45.5,
        val_min_per_class_f1=40.0,
        val_plur_corr=80.0,
    )
    
    assert result == 80.0


def test_objective_computer_composite_threshold_exact():
    """Test composite objective with threshold mode (exactly at threshold)."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "min_f1_threshold": 35.0,
    }
    computer = ObjectiveComputer(cfg)
    
    # min_f1=35.0 == threshold=35.0 → should return plur_corr (not gradient)
    result = computer.compute(
        val_acc=50.0,
        val_macro_f1=45.5,
        val_min_per_class_f1=35.0,
        val_plur_corr=90.0,
    )
    
    assert result == 90.0


def test_objective_computer_composite_weighted():
    """Test composite objective with weighted mode."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "composite_min_f1_weight": 0.5,
    }
    computer = ObjectiveComputer(cfg)
    
    # 0.5 * 30.0 + 0.5 * 80.0 = 15.0 + 40.0 = 55.0
    result = computer.compute(
        val_acc=50.0,
        val_macro_f1=45.5,
        val_min_per_class_f1=30.0,
        val_plur_corr=80.0,
    )
    
    assert result == pytest.approx(55.0)


def test_objective_computer_invalid_objective():
    """Test that invalid objective raises error."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "invalid_objective"}
    computer = ObjectiveComputer(cfg)
    
    with pytest.raises(ValueError, match="Invalid objective"):
        computer.compute(50.0, 45.5, 30.0, 80.0)


def test_objective_computer_threshold_range_validation():
    """Test that threshold must be in valid range [0, 100]."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "min_f1_threshold": 150.0,  # Invalid: > 100
    }
    
    with pytest.raises(ValueError, match="must be in range"):
        ObjectiveComputer(cfg)


def test_objective_computer_weight_range_validation():
    """Test that weight must be in valid range [0, 1]."""
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "composite_min_f1_weight": 1.5,  # Invalid: > 1.0
    }
    
    with pytest.raises(ValueError, match="must be in range"):
        ObjectiveComputer(cfg)

