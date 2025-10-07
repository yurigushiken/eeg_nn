"""
Inner fold training loop for nested cross-validation.

This module handles the complete training loop for one inner fold:
- Epoch iteration
- LR warmup (linear)
- Augmentation warmup (gradual strength increase)
- Mixup augmentation
- Training pass (forward/backward)
- Validation pass
- Learning curve collection
- Integration with CheckpointManager for early stopping
- Integration with Optuna for pruning

Constitutional compliance: Section III (Deterministic Training)

Classes:
    InnerTrainer: Runs complete training loop for one inner fold
"""

from __future__ import annotations
from typing import Dict, Callable, List, Any
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from .metrics import ObjectiveComputer
from .checkpointing import CheckpointManager

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def compute_plurality_correctness(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute plurality correctness (row-wise plurality metric).
    
    For each true class (row in confusion matrix), checks if the correct 
    prediction (diagonal element) is the most frequent prediction.
    
    Returns proportion of classes where correct prediction is plurality.
    """
    if not y_true or not y_pred:
        return 0.0
    
    try:
        classes = sorted(set(y_true) | set(y_pred))
        n_classes = len(classes)
        
        if n_classes == 0:
            return 0.0
        
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        diagonal_is_max_count = 0
        for i in range(n_classes):
            row = cm[i, :]
            if len(row) > 0:
                max_val = np.max(row)
                diagonal_val = cm[i, i]
                if diagonal_val == max_val:
                    diagonal_is_max_count += 1
        
        return float(diagonal_is_max_count) / float(n_classes)
    
    except Exception:
        return 0.0


class InnerTrainer:
    """
    Runs complete training loop for one inner fold.
    
    Handles:
    - Epoch iteration with early stopping
    - LR warmup (linear ramp from lr_warmup_init to lr)
    - Augmentation warmup (gradual strength increase)
    - Mixup augmentation (optional)
    - Training pass (forward/backward)
    - Validation pass
    - Learning curve collection
    - Checkpoint selection via CheckpointManager
    - Optuna pruning integration
    
    Example usage:
        >>> cfg = {
        ...     "epochs": 60,
        ...     "lr": 0.001,
        ...     "early_stop": 10,
        ...     "lr_warmup_frac": 0.1,
        ...     "aug_warmup_frac": 0.2,
        ...     "mixup_alpha": 0.2,
        ... }
        >>> obj_computer = ObjectiveComputer(cfg)
        >>> checkpoint_mgr = CheckpointManager(cfg, obj_computer)
        >>> trainer = InnerTrainer(cfg, obj_computer)
        >>> 
        >>> result = trainer.train(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     loss_fn=loss_fn,
        ...     tr_loader=train_loader,
        ...     va_loader=val_loader,
        ...     aug_builder=aug_builder,
        ...     input_adapter=input_adapter,
        ...     checkpoint_manager=checkpoint_mgr,
        ...     fold_info={"outer_fold": 1, "inner_fold": 1},
        ...     optuna_trial=None,
        ...     global_step_offset=0,
        ... )
        >>> 
        >>> best_state = result["best_state"]
        >>> learning_curves = result["learning_curves"]
    """
    
    def __init__(self, cfg: Dict, objective_computer: ObjectiveComputer):
        """
        Initialize InnerTrainer.
        
        Args:
            cfg: Configuration dictionary
            objective_computer: ObjectiveComputer for metric computation
        """
        self.cfg = cfg
        self.obj_computer = objective_computer
        self.total_epochs = int(cfg.get("epochs", 60))
        self.base_lr = float(cfg.get("lr", 7e-4))
        
        # LR warmup parameters
        self.lr_warmup_frac = float(cfg.get("lr_warmup_frac", 0.0) or 0.0)
        self.lr_warmup_init = float(cfg.get("lr_warmup_init", 0.0) or 0.0)
        self.lr_warmup_epochs = int(np.ceil(self.lr_warmup_frac * self.total_epochs)) if self.lr_warmup_frac > 0 else 0
        
        # Augmentation warmup parameters
        self.aug_warmup_frac = float(cfg.get("aug_warmup_frac", 0.0) or 0.0)
        self.aug_warmup_epochs = int(np.ceil(self.aug_warmup_frac * self.total_epochs)) if self.aug_warmup_frac > 0 else 0
        
        # Mixup parameters
        self.mixup_alpha = float(cfg.get("mixup_alpha", 0.0) or 0.0)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: nn.Module,
        tr_loader: DataLoader,
        va_loader: DataLoader,
        aug_builder: Callable | None,
        input_adapter: Callable | None,
        checkpoint_manager: CheckpointManager,
        fold_info: Dict[str, int],
        optuna_trial: Any | None,
        global_step_offset: int,
    ) -> Dict:
        """
        Run complete training loop for one inner fold.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            scheduler: LR scheduler instance
            loss_fn: Loss function
            tr_loader: Training DataLoader
            va_loader: Validation DataLoader
            aug_builder: Function to build augmentation transform (or None)
            input_adapter: Function to adapt inputs (or None)
            checkpoint_manager: CheckpointManager for early stopping
            fold_info: Dict with "outer_fold" and "inner_fold" keys
            optuna_trial: Optuna trial for pruning (or None)
            global_step_offset: Offset for Optuna global step
        
        Returns:
            Dictionary containing:
                - best_state: Best model state dict
                - best_metrics: Dict of best validation metrics
                - learning_curves: List of dicts (one per epoch)
                - tr_hist: List of train losses
                - va_hist: List of val losses
                - va_acc_hist: List of val accuracies
        """
        outer_fold = fold_info["outer_fold"]
        inner_fold = fold_info["inner_fold"]
        
        # Training history
        tr_hist: List[float] = []
        va_hist: List[float] = []
        va_acc_hist: List[float] = []
        learning_curves: List[Dict] = []
        
        global_step = global_step_offset
        
        for epoch in range(1, self.total_epochs + 1):
            # LR warmup
            if self.lr_warmup_epochs > 0 and epoch <= self.lr_warmup_epochs:
                t = epoch / max(1, self.lr_warmup_epochs)
                factor = self.lr_warmup_init + (1.0 - self.lr_warmup_init) * t
                self._set_lr(optimizer, self.base_lr * factor)
            
            # Augmentation warmup
            if aug_builder is not None and self.aug_warmup_epochs > 0 and epoch <= self.aug_warmup_epochs:
                r = epoch / max(1, self.aug_warmup_epochs)
            else:
                r = 1.0
            
            # Update augmentation transform
            if aug_builder is not None:
                try:
                    ds_train = getattr(tr_loader.dataset, "dataset", None)
                    if ds_train is not None:
                        aug_transform_epoch = aug_builder(self._scaled_aug_cfg(r), ds_train)
                        ds_train.set_transform(aug_transform_epoch)
                except Exception:
                    pass
            
            # Training pass
            train_loss = self._train_epoch(
                model, optimizer, loss_fn, tr_loader, input_adapter
            )
            tr_hist.append(train_loss)
            
            # Validation pass
            val_metrics = self._validate_epoch(
                model, loss_fn, va_loader, input_adapter
            )
            val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["val_acc"]
            val_macro_f1 = val_metrics["val_macro_f1"]
            val_min_per_class_f1 = val_metrics["val_min_per_class_f1"]
            val_plur_corr = val_metrics["val_plur_corr"]
            
            va_hist.append(val_loss)
            va_acc_hist.append(val_acc)
            
            # Compute objective metric
            val_objective_metric = self.obj_computer.compute(
                val_acc, val_macro_f1, val_min_per_class_f1, val_plur_corr
            )
            
            # Collect learning curve row
            try:
                learning_curves.append({
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "val_macro_f1": float(val_macro_f1),
                    "val_min_per_class_f1": float(val_min_per_class_f1),
                    "val_plur_corr": float(val_plur_corr),
                    "val_objective_metric": float(val_objective_metric),
                    "n_train": int(len(tr_loader.dataset)),
                    "n_val": int(len(va_loader.dataset)),
                    "optuna_trial_id": int(optuna_trial.number) if optuna_trial else -1,
                    "param_hash": "",
                })
            except Exception:
                pass
            
            # Optuna pruning
            if optuna_trial is not None and OPTUNA_AVAILABLE:
                global_step += 1
                try:
                    optuna_trial.report(val_objective_metric, global_step)
                    if optuna_trial.should_prune():
                        print(
                            f"  [prune] Trial pruned at epoch {epoch} of fold {outer_fold} inner {inner_fold}.",
                            flush=True,
                        )
                        raise optuna.exceptions.TrialPruned()
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception:
                    pass
            
            # Update checkpoint
            val_metrics_with_loss = {
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_min_per_class_f1": val_min_per_class_f1,
                "val_plur_corr": val_plur_corr,
                "val_loss": val_loss,
            }
            checkpoint_manager.update(copy.deepcopy(model.state_dict()), val_metrics_with_loss)
            
            # Scheduler step (after warmup)
            if not (self.lr_warmup_epochs > 0 and epoch <= self.lr_warmup_epochs):
                scheduler.step(val_loss)
            
            # Early stopping check
            if checkpoint_manager.should_stop():
                print(f"  [early_stop] Stopped at epoch {epoch}", flush=True)
                break
            
            # Periodic logging
            if epoch % 5 == 0 or epoch == self.total_epochs:
                best_obj = checkpoint_manager.best_objective
                print(
                    f"  [epoch {epoch}] (outer {outer_fold} inner {inner_fold}) "
                    f"tr_loss={train_loss:.4f} va_loss={val_loss:.4f} "
                    f"va_acc={val_acc:.2f} best_obj={best_obj:.4f}",
                    flush=True,
                )
        
        # Return training results
        return {
            "best_state": checkpoint_manager.get_best_state(),
            "best_metrics": checkpoint_manager.get_best_metrics(),
            "learning_curves": learning_curves,
            "tr_hist": tr_hist,
            "va_hist": va_hist,
            "va_acc_hist": va_acc_hist,
            "best_inner_acc": checkpoint_manager.get_best_metrics().get("val_acc", 0.0),
            "best_inner_macro_f1": checkpoint_manager.get_best_metrics().get("val_macro_f1", 0.0),
            "best_inner_min_per_class_f1": checkpoint_manager.get_best_metrics().get("val_min_per_class_f1", 0.0),
            "best_inner_plur_corr": checkpoint_manager.get_best_metrics().get("val_plur_corr", 0.0),
        }
    
    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        tr_loader: DataLoader,
        input_adapter: Callable | None,
    ) -> float:
        """Run one training epoch."""
        model.train()
        train_loss = 0.0
        
        for xb, yb in tr_loader:
            yb_gpu = yb.to(self.device)
            xb_gpu = (
                xb.to(self.device)
                if not isinstance(xb, (list, tuple))
                else [t.to(self.device) for t in xb]
            )
            xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu
            
            optimizer.zero_grad()
            
            # Optional mixup
            if (
                self.mixup_alpha > 0.0
                and isinstance(xb_gpu, torch.Tensor)
                and xb_gpu.size(0) > 1
            ):
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                perm = torch.randperm(xb_gpu.size(0), device=xb_gpu.device)
                xb_mix = lam * xb_gpu + (1.0 - lam) * xb_gpu[perm]
                out = model(xb_mix)
                yb_perm = yb_gpu[perm]
                loss = lam * loss_fn(out.float(), yb_gpu) + (1.0 - lam) * loss_fn(out.float(), yb_perm)
            else:
                out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                loss = loss_fn(out.float(), yb_gpu)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        return train_loss / max(1, len(tr_loader))
    
    def _validate_epoch(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        va_loader: DataLoader,
        input_adapter: Callable | None,
    ) -> Dict[str, float]:
        """Run one validation epoch."""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true_ep: List[int] = []
        y_pred_ep: List[int] = []
        
        with torch.no_grad():
            for xb, yb in va_loader:
                yb_gpu = yb.to(self.device)
                xb_gpu = (
                    xb.to(self.device)
                    if not isinstance(xb, (list, tuple))
                    else [t.to(self.device) for t in xb]
                )
                xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu
                out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                loss = loss_fn(out.float(), yb_gpu)
                val_loss += loss.item()
                preds = out.argmax(1).cpu()
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                y_true_ep.extend(yb.tolist())
                y_pred_ep.extend(preds.tolist())
        
        val_loss /= max(1, len(va_loader))
        val_acc = 100.0 * correct / max(1, total)
        
        try:
            val_macro_f1 = f1_score(y_true_ep, y_pred_ep, average="macro") * 100
            val_per_class_f1 = f1_score(y_true_ep, y_pred_ep, average=None)
            val_min_per_class_f1 = float(np.min(val_per_class_f1)) * 100 if len(val_per_class_f1) > 0 else 0.0
            val_plur_corr = compute_plurality_correctness(y_true_ep, y_pred_ep) * 100
        except Exception:
            val_macro_f1 = 0.0
            val_min_per_class_f1 = 0.0
            val_plur_corr = 0.0
        
        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_min_per_class_f1": val_min_per_class_f1,
            "val_plur_corr": val_plur_corr,
        }
    
    def _set_lr(self, optimizer: torch.optim.Optimizer, lr_value: float):
        """Set learning rate for all parameter groups."""
        for pg in optimizer.param_groups:
            pg["lr"] = lr_value
    
    def _scaled_aug_cfg(self, r: float) -> Dict:
        """Scale augmentation config by factor r (for warmup)."""
        if r >= 1.0:
            return self.cfg
        
        cfg2 = dict(self.cfg)
        
        # Scale probabilities
        for k in ("shift_p", "scale_p", "noise_p", "time_mask_p", "chan_mask_p"):
            if k in cfg2 and cfg2[k] is not None:
                cfg2[k] = float(cfg2[k]) * r
        
        # Scale strengths/magnitudes
        for k in ("shift_max_frac", "noise_std", "time_mask_frac", "chan_mask_ratio"):
            if k in cfg2 and cfg2[k] is not None:
                cfg2[k] = float(cfg2[k]) * r
        
        # Scale ranges (approach 1.0)
        if "scale_min" in cfg2 and cfg2["scale_min"] is not None:
            smin = float(cfg2["scale_min"])
            cfg2["scale_min"] = 1.0 + r * (smin - 1.0)
        if "scale_max" in cfg2 and cfg2["scale_max"] is not None:
            smax = float(cfg2["scale_max"])
            cfg2["scale_max"] = 1.0 + r * (smax - 1.0)
        
        return cfg2

