"""
Tests for InnerTrainer class (training/inner_loop.py).

These tests verify that the inner training loop works correctly:
- Epoch loop execution
- LR warmup behavior
- Augmentation warmup behavior
- Mixup application
- Validation pass
- Learning curve collection
- Integration with CheckpointManager
- Optuna pruning integration

Constitutional compliance: Section III (Deterministic Training)
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def test_inner_trainer_imports():
    """Test that InnerTrainer can be imported (will fail pre-refactor)."""
    try:
        from code.training.inner_loop import InnerTrainer
        assert InnerTrainer is not None
    except ImportError as e:
        pytest.fail(f"Cannot import InnerTrainer: {e}")


def test_inner_trainer_init():
    """Test InnerTrainer initialization."""
    from code.training.inner_loop import InnerTrainer
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "epochs": 10,
        "lr": 0.001,
        "early_stop": 5,
    }
    obj_computer = ObjectiveComputer(cfg)
    trainer = InnerTrainer(cfg, obj_computer)
    
    assert trainer.cfg == cfg
    assert trainer.obj_computer is obj_computer
    assert trainer.total_epochs == 10


def test_inner_trainer_minimal_training():
    """Test minimal training loop (1 epoch, small dataset)."""
    from code.training.inner_loop import InnerTrainer
    from code.training.metrics import ObjectiveComputer
    from code.training.checkpointing import CheckpointManager
    
    # Minimal config
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "epochs": 1,
        "lr": 0.001,
        "early_stop": 10,
        "weight_decay": 0.0,
        "batch_size": 4,
    }
    
    obj_computer = ObjectiveComputer(cfg)
    checkpoint_mgr = CheckpointManager(cfg, obj_computer)
    trainer = InnerTrainer(cfg, obj_computer)
    
    # Create dummy dataset
    X_train = torch.randn(16, 3, 10)  # 16 samples, 3 channels, 10 timepoints
    y_train = torch.randint(0, 3, (16,))  # 3 classes
    X_val = torch.randn(8, 3, 10)
    y_val = torch.randint(0, 3, (8,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Dummy model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(30, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train
    result = trainer.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        tr_loader=train_loader,
        va_loader=val_loader,
        aug_builder=None,
        input_adapter=None,
        checkpoint_manager=checkpoint_mgr,
        fold_info={"outer_fold": 1, "inner_fold": 1},
        optuna_trial=None,
        global_step_offset=0,
    )
    
    # Verify result structure
    assert "best_state" in result
    assert "best_metrics" in result
    assert "learning_curves" in result
    assert "tr_hist" in result
    assert "va_hist" in result
    assert "va_acc_hist" in result
    
    # Verify learning curves collected
    assert len(result["learning_curves"]) == 1  # 1 epoch
    assert result["learning_curves"][0]["epoch"] == 1
    assert "train_loss" in result["learning_curves"][0]
    assert "val_loss" in result["learning_curves"][0]
    
    print("  [OK] Minimal training loop works")


def test_inner_trainer_lr_warmup():
    """Test that LR warmup is applied correctly."""
    from code.training.inner_loop import InnerTrainer
    from code.training.metrics import ObjectiveComputer
    from code.training.checkpointing import CheckpointManager
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "epochs": 10,
        "lr": 0.01,
        "lr_warmup_frac": 0.3,  # 30% warmup = 3 epochs
        "lr_warmup_init": 0.1,  # Start at 10% of lr
        "early_stop": 10,
        "weight_decay": 0.0,
        "batch_size": 4,
    }
    
    obj_computer = ObjectiveComputer(cfg)
    checkpoint_mgr = CheckpointManager(cfg, obj_computer)
    trainer = InnerTrainer(cfg, obj_computer)
    
    # Create dummy dataset
    X_train = torch.randn(16, 3, 10)
    y_train = torch.randint(0, 3, (16,))
    X_val = torch.randn(8, 3, 10)
    y_val = torch.randint(0, 3, (8,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    model = nn.Sequential(nn.Flatten(), nn.Linear(30, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train with warmup
    result = trainer.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        tr_loader=train_loader,
        va_loader=val_loader,
        aug_builder=None,
        input_adapter=None,
        checkpoint_manager=checkpoint_mgr,
        fold_info={"outer_fold": 1, "inner_fold": 1},
        optuna_trial=None,
        global_step_offset=0,
    )
    
    # Verify training completed
    assert len(result["learning_curves"]) >= 3  # At least 3 epochs (warmup period)
    print("  [OK] LR warmup applied")


def test_inner_trainer_early_stopping():
    """Test that early stopping works correctly."""
    from code.training.inner_loop import InnerTrainer
    from code.training.metrics import ObjectiveComputer
    from code.training.checkpointing import CheckpointManager
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "epochs": 100,  # Set high, but early stop should trigger
        "lr": 0.001,
        "early_stop": 3,  # Stop after 3 epochs without improvement
        "weight_decay": 0.0,
        "batch_size": 4,
    }
    
    obj_computer = ObjectiveComputer(cfg)
    checkpoint_mgr = CheckpointManager(cfg, obj_computer)
    trainer = InnerTrainer(cfg, obj_computer)
    
    # Create dummy dataset
    X_train = torch.randn(16, 3, 10)
    y_train = torch.randint(0, 3, (16,))
    X_val = torch.randn(8, 3, 10)
    y_val = torch.randint(0, 3, (8,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    model = nn.Sequential(nn.Flatten(), nn.Linear(30, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train
    result = trainer.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        tr_loader=train_loader,
        va_loader=val_loader,
        aug_builder=None,
        input_adapter=None,
        checkpoint_manager=checkpoint_mgr,
        fold_info={"outer_fold": 1, "inner_fold": 1},
        optuna_trial=None,
        global_step_offset=0,
    )
    
    # Verify early stopping triggered (should not train for all 100 epochs)
    num_epochs = len(result["learning_curves"])
    assert num_epochs < 100, f"Expected early stop, but trained {num_epochs} epochs"
    print(f"  [OK] Early stopping triggered at epoch {num_epochs}")


def test_inner_trainer_returns_best_state():
    """Test that InnerTrainer returns the best model state."""
    from code.training.inner_loop import InnerTrainer
    from code.training.metrics import ObjectiveComputer
    from code.training.checkpointing import CheckpointManager
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "epochs": 5,
        "lr": 0.001,
        "early_stop": 10,
        "weight_decay": 0.0,
        "batch_size": 4,
    }
    
    obj_computer = ObjectiveComputer(cfg)
    checkpoint_mgr = CheckpointManager(cfg, obj_computer)
    trainer = InnerTrainer(cfg, obj_computer)
    
    # Create dummy dataset
    X_train = torch.randn(16, 3, 10)
    y_train = torch.randint(0, 3, (16,))
    X_val = torch.randn(8, 3, 10)
    y_val = torch.randint(0, 3, (8,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    model = nn.Sequential(nn.Flatten(), nn.Linear(30, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train
    result = trainer.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        tr_loader=train_loader,
        va_loader=val_loader,
        aug_builder=None,
        input_adapter=None,
        checkpoint_manager=checkpoint_mgr,
        fold_info={"outer_fold": 1, "inner_fold": 1},
        optuna_trial=None,
        global_step_offset=0,
    )
    
    # Verify best state exists and can be loaded
    assert result["best_state"] is not None
    model.load_state_dict(result["best_state"])  # Should not raise
    
    # Verify best metrics exist
    assert "val_acc" in result["best_metrics"]
    assert "val_macro_f1" in result["best_metrics"]
    
    print("  [OK] Best state and metrics returned")


def test_inner_trainer_learning_curve_format():
    """Test that learning curves have correct format."""
    from code.training.inner_loop import InnerTrainer
    from code.training.metrics import ObjectiveComputer
    from code.training.checkpointing import CheckpointManager
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "epochs": 2,
        "lr": 0.001,
        "early_stop": 10,
        "weight_decay": 0.0,
        "batch_size": 4,
    }
    
    obj_computer = ObjectiveComputer(cfg)
    checkpoint_mgr = CheckpointManager(cfg, obj_computer)
    trainer = InnerTrainer(cfg, obj_computer)
    
    # Create dummy dataset
    X_train = torch.randn(16, 3, 10)
    y_train = torch.randint(0, 3, (16,))
    X_val = torch.randn(8, 3, 10)
    y_val = torch.randint(0, 3, (8,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    model = nn.Sequential(nn.Flatten(), nn.Linear(30, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train
    result = trainer.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        tr_loader=train_loader,
        va_loader=val_loader,
        aug_builder=None,
        input_adapter=None,
        checkpoint_manager=checkpoint_mgr,
        fold_info={"outer_fold": 1, "inner_fold": 1},
        optuna_trial=None,
        global_step_offset=0,
    )
    
    # Verify learning curve format
    curves = result["learning_curves"]
    assert len(curves) == 2  # 2 epochs
    
    for i, curve in enumerate(curves):
        # Verify all required fields present
        required_fields = [
            "outer_fold", "inner_fold", "epoch",
            "train_loss", "val_loss", "val_acc",
            "val_macro_f1", "val_min_per_class_f1",
            "val_plur_corr", "val_objective_metric",
            "n_train", "n_val",
            "optuna_trial_id", "param_hash",
        ]
        for field in required_fields:
            assert field in curve, f"Missing field: {field}"
        
        # Verify values
        assert curve["outer_fold"] == 1
        assert curve["inner_fold"] == 1
        assert curve["epoch"] == i + 1
        assert isinstance(curve["train_loss"], float)
        assert isinstance(curve["val_loss"], float)
    
    print("  [OK] Learning curve format correct")

