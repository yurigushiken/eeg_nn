"""
Tests for CheckpointManager class (training/checkpointing.py).

These tests will FAIL until we extract the CheckpointManager class.
They verify that checkpoint selection and early stopping behavior is preserved.

Constitutional compliance: Section III (Deterministic Training)
"""
import pytest
import torch
import copy


def test_checkpoint_manager_imports():
    """Test that CheckpointManager can be imported (will fail pre-refactor)."""
    try:
        from code.training.checkpointing import CheckpointManager
        from code.training.metrics import ObjectiveComputer
        assert CheckpointManager is not None
    except ImportError as e:
        pytest.fail(f"Cannot import CheckpointManager: {e}")


def test_checkpoint_manager_init():
    """Test CheckpointManager initialization."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    assert manager.early_stop_patience == 10
    assert manager.patience == 0
    assert manager.best_objective == float("-inf")


def test_checkpoint_manager_reset():
    """Test that reset() clears state for new inner fold."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    # Simulate some updates
    dummy_state = {"layer.weight": torch.randn(10, 10)}
    manager.update(dummy_state, {
        "val_acc": 50.0,
        "val_macro_f1": 45.0,
        "val_min_per_class_f1": 30.0,
        "val_plur_corr": 80.0,
        "val_loss": 1.5,
    })
    
    assert manager.best_objective > float("-inf")
    
    # Reset
    manager.reset()
    
    assert manager.patience == 0
    assert manager.best_objective == float("-inf")
    assert manager.best_state is None


def test_checkpoint_manager_first_update_always_saves():
    """Test that first epoch always saves checkpoint."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    dummy_state = {"layer.weight": torch.randn(10, 10)}
    
    updated = manager.update(dummy_state, {
        "val_acc": 50.0,
        "val_macro_f1": 45.0,
        "val_min_per_class_f1": 30.0,
        "val_plur_corr": 80.0,
        "val_loss": 1.5,
    })
    
    assert updated is True
    assert manager.best_state is not None
    assert manager.best_metrics["val_macro_f1"] == 45.0


def test_checkpoint_manager_improves_objective():
    """Test that checkpoint updates when objective improves."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    # First epoch
    manager.update({"epoch1": 1}, {
        "val_acc": 50.0,
        "val_macro_f1": 45.0,
        "val_min_per_class_f1": 30.0,
        "val_plur_corr": 80.0,
        "val_loss": 1.5,
    })
    
    # Second epoch (improved)
    updated = manager.update({"epoch2": 2}, {
        "val_acc": 55.0,
        "val_macro_f1": 50.0,  # Improved!
        "val_min_per_class_f1": 35.0,
        "val_plur_corr": 85.0,
        "val_loss": 1.2,
    })
    
    assert updated is True
    assert manager.best_metrics["val_macro_f1"] == 50.0
    assert manager.patience == 0  # Reset patience


def test_checkpoint_manager_no_improvement_increases_patience():
    """Test that patience increases when no improvement."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 3,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    # First epoch
    manager.update({"epoch1": 1}, {
        "val_acc": 50.0,
        "val_macro_f1": 50.0,
        "val_min_per_class_f1": 30.0,
        "val_plur_corr": 80.0,
        "val_loss": 1.5,
    })
    
    assert manager.patience == 0
    
    # Second epoch (no improvement)
    manager.update({"epoch2": 2}, {
        "val_acc": 50.0,
        "val_macro_f1": 48.0,  # Worse!
        "val_min_per_class_f1": 28.0,
        "val_plur_corr": 75.0,
        "val_loss": 1.7,
    })
    
    assert manager.patience == 1
    assert not manager.should_stop()
    
    # Third epoch (still no improvement)
    manager.update({"epoch3": 3}, {
        "val_acc": 50.0,
        "val_macro_f1": 49.0,
        "val_min_per_class_f1": 29.0,
        "val_plur_corr": 76.0,
        "val_loss": 1.6,
    })
    
    assert manager.patience == 2
    assert not manager.should_stop()
    
    # Fourth epoch (trigger early stop)
    manager.update({"epoch4": 4}, {
        "val_acc": 50.0,
        "val_macro_f1": 49.0,
        "val_min_per_class_f1": 29.0,
        "val_plur_corr": 76.0,
        "val_loss": 1.6,
    })
    
    assert manager.patience == 3
    assert manager.should_stop()  # patience >= early_stop


def test_checkpoint_manager_tie_breaking_by_loss():
    """Test that ties in objective are broken by validation loss."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    # First epoch
    manager.update({"epoch1": 1}, {
        "val_acc": 50.0,
        "val_macro_f1": 50.0,
        "val_min_per_class_f1": 30.0,
        "val_plur_corr": 80.0,
        "val_loss": 1.5,
    })
    
    # Second epoch (same objective, but lower loss)
    updated = manager.update({"epoch2": 2}, {
        "val_acc": 50.0,
        "val_macro_f1": 50.0,  # Same objective!
        "val_min_per_class_f1": 31.0,
        "val_plur_corr": 79.0,
        "val_loss": 1.2,  # Better loss!
    })
    
    assert updated is True  # Should update due to lower loss
    assert manager.best_checkpoint_loss == 1.2


def test_checkpoint_manager_tie_breaking_no_update():
    """Test that ties in objective with higher loss don't update."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    # First epoch
    manager.update({"epoch1": 1}, {
        "val_acc": 50.0,
        "val_macro_f1": 50.0,
        "val_min_per_class_f1": 30.0,
        "val_plur_corr": 80.0,
        "val_loss": 1.2,
    })
    
    # Second epoch (same objective, but higher loss)
    updated = manager.update({"epoch2": 2}, {
        "val_acc": 50.0,
        "val_macro_f1": 50.0,  # Same objective!
        "val_min_per_class_f1": 31.0,
        "val_plur_corr": 79.0,
        "val_loss": 1.5,  # Worse loss!
    })
    
    assert updated is False  # Should NOT update
    assert manager.best_checkpoint_loss == 1.2


def test_checkpoint_manager_composite_threshold_objective():
    """Test checkpoint manager with composite threshold objective."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "min_f1_threshold": 35.0,
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    # First epoch: below threshold (gradient mode)
    manager.update({"epoch1": 1}, {
        "val_acc": 50.0,
        "val_macro_f1": 45.0,
        "val_min_per_class_f1": 30.0,  # Below threshold
        "val_plur_corr": 80.0,
        "val_loss": 1.5,
    })
    
    # Objective should be 30.0 * 0.1 = 3.0
    assert manager.best_objective == pytest.approx(3.0)
    
    # Second epoch: above threshold (plur_corr mode)
    updated = manager.update({"epoch2": 2}, {
        "val_acc": 52.0,
        "val_macro_f1": 46.0,
        "val_min_per_class_f1": 36.0,  # Above threshold!
        "val_plur_corr": 85.0,
        "val_loss": 1.3,
    })
    
    # Objective should be 85.0 (plur_corr)
    assert updated is True
    assert manager.best_objective == 85.0


def test_checkpoint_manager_get_best_state():
    """Test that get_best_state returns the correct state."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    state1 = {"epoch": torch.tensor([1])}
    state2 = {"epoch": torch.tensor([2])}
    
    # First epoch
    manager.update(state1, {
        "val_acc": 50.0,
        "val_macro_f1": 45.0,
        "val_min_per_class_f1": 30.0,
        "val_plur_corr": 80.0,
        "val_loss": 1.5,
    })
    
    # Second epoch (better)
    manager.update(state2, {
        "val_acc": 52.0,
        "val_macro_f1": 50.0,
        "val_min_per_class_f1": 35.0,
        "val_plur_corr": 85.0,
        "val_loss": 1.2,
    })
    
    best = manager.get_best_state()
    assert best["epoch"].item() == 2  # Should be state2


def test_checkpoint_manager_get_best_metrics():
    """Test that get_best_metrics returns correct metrics."""
    from code.training.checkpointing import CheckpointManager
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "early_stop": 10,
    }
    obj_computer = ObjectiveComputer(cfg)
    manager = CheckpointManager(cfg, obj_computer)
    
    # Update checkpoint
    manager.update({"state": 1}, {
        "val_acc": 52.0,
        "val_macro_f1": 50.0,
        "val_min_per_class_f1": 35.0,
        "val_plur_corr": 85.0,
        "val_loss": 1.2,
    })
    
    metrics = manager.get_best_metrics()
    assert metrics["val_acc"] == 52.0
    assert metrics["val_macro_f1"] == 50.0
    assert metrics["val_min_per_class_f1"] == 35.0
    assert metrics["val_plur_corr"] == 85.0

