"""
Checkpoint management and early stopping logic for model training.

This module handles:
- Best checkpoint selection based on objective-aligned metrics
- Early stopping based on patience monitoring
- Tie-breaking using validation loss when objectives are equal

Constitutional compliance: Section III (Deterministic Training)
- Deterministic checkpoint selection
- Explicit objective alignment

Classes:
    CheckpointManager: Manages checkpoint selection and early stopping for one inner fold
"""

from __future__ import annotations
from typing import Dict, Any
import copy

from .metrics import ObjectiveComputer


class CheckpointManager:
    """
    Manages checkpoint selection and early stopping for one inner training fold.
    
    Checkpoint selection logic:
    1. PRIMARY: Maximize objective-aligned metric (same as Optuna objective)
    2. TIE-BREAKER: If objective metrics are equal, prefer lower validation loss
    
    Early stopping logic:
    - Monitor the same objective-aligned metric
    - Increment patience when no improvement
    - Stop when patience >= early_stop threshold
    
    This ensures that checkpoint selection, early stopping, and Optuna optimization
    all align on the same metric, maintaining scientific integrity.
    
    Example usage:
        >>> cfg = {
        ...     "optuna_objective": "inner_mean_macro_f1",
        ...     "early_stop": 10
        ... }
        >>> obj_computer = ObjectiveComputer(cfg)
        >>> manager = CheckpointManager(cfg, obj_computer)
        >>> 
        >>> # Training epoch loop
        >>> for epoch in range(max_epochs):
        ...     # ... train and validate ...
        ...     updated = manager.update(model.state_dict(), {
        ...         "val_acc": val_acc,
        ...         "val_macro_f1": val_macro_f1,
        ...         "val_min_per_class_f1": val_min_f1,
        ...         "val_plur_corr": val_plur_corr,
        ...         "val_loss": val_loss,
        ...     })
        ...     if manager.should_stop():
        ...         break
        >>> 
        >>> best_state = manager.get_best_state()
        >>> best_metrics = manager.get_best_metrics()
    """
    
    def __init__(self, cfg: Dict, objective_computer: ObjectiveComputer):
        """
        Initialize CheckpointManager.
        
        Args:
            cfg: Configuration dictionary containing:
                - early_stop: int, patience threshold (default: 10)
            objective_computer: ObjectiveComputer instance for metric computation
        """
        self.cfg = cfg
        self.obj_computer = objective_computer
        self.early_stop_patience = int(cfg.get("early_stop", 10))
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """
        Reset checkpoint manager state for a new inner fold.
        
        Should be called before training each inner fold to clear previous state.
        """
        self.best_objective = float("-inf")
        self.best_checkpoint_loss = float("inf")
        self.best_state = None
        self.best_metrics = {}
        self.patience = 0
    
    def update(self, model_state: Dict[str, Any], val_metrics: Dict[str, float]) -> bool:
        """
        Update checkpoint based on current epoch's validation metrics.
        
        Checkpoint update rules:
        1. If current objective > best objective: UPDATE (and reset patience)
        2. If current objective == best objective AND current loss < best loss: UPDATE
        3. Otherwise: DON'T UPDATE (and increment patience)
        
        Args:
            model_state: Model state dict to potentially save as checkpoint
            val_metrics: Dictionary containing:
                - val_acc: float (0-100)
                - val_macro_f1: float (0-100)
                - val_min_per_class_f1: float (0-100)
                - val_plur_corr: float (0-100)
                - val_loss: float (for tie-breaking)
        
        Returns:
            bool: True if checkpoint was updated, False otherwise
        """
        # Extract metrics
        val_acc = val_metrics["val_acc"]
        val_macro_f1 = val_metrics["val_macro_f1"]
        val_min_per_class_f1 = val_metrics["val_min_per_class_f1"]
        val_plur_corr = val_metrics["val_plur_corr"]
        val_loss = val_metrics["val_loss"]
        
        # Compute objective-aligned metric
        current_objective = self.obj_computer.compute(
            val_acc, val_macro_f1, val_min_per_class_f1, val_plur_corr
        )
        
        # Update early stopping patience
        if current_objective > self.best_objective:
            self.patience = 0  # Reset patience on improvement
        else:
            self.patience += 1  # Increment patience on no improvement
        
        # Checkpoint selection logic
        update_checkpoint = False
        if current_objective > self.best_objective:
            # PRIMARY: Objective improved
            update_checkpoint = True
        elif (current_objective == self.best_objective) and (val_loss < self.best_checkpoint_loss):
            # TIE-BREAKER: Same objective, but lower validation loss
            update_checkpoint = True
        
        if update_checkpoint:
            self.best_objective = current_objective
            self.best_checkpoint_loss = val_loss
            self.best_state = copy.deepcopy(model_state)
            self.best_metrics = {
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_min_per_class_f1": val_min_per_class_f1,
                "val_plur_corr": val_plur_corr,
                "val_loss": val_loss,
            }
            return True
        else:
            return False
    
    def should_stop(self) -> bool:
        """
        Check if early stopping should be triggered.
        
        Returns:
            bool: True if patience >= early_stop threshold, False otherwise
        """
        return self.patience >= self.early_stop_patience
    
    def get_best_state(self) -> Dict[str, Any] | None:
        """
        Get the best model state dict.
        
        Returns:
            Model state dict of the best checkpoint, or None if no checkpoint saved
        """
        return self.best_state
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get the validation metrics of the best checkpoint.
        
        Returns:
            Dictionary containing val_acc, val_macro_f1, val_min_per_class_f1,
            val_plur_corr, val_loss from the best checkpoint epoch
        """
        return self.best_metrics

