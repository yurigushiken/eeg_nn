"""
Outer fold orchestration for nested cross-validation.

This module handles orchestration of one complete outer fold:
- Inner K-fold training loop
- Model/optimizer/loss setup per inner fold  
- Inner result aggregation
- Best inner model selection (objective-aligned)
- Outer test evaluation (ensemble or refit)
- Per-fold plotting and metrics

Constitutional compliance: 
- Section III (Deterministic Training)
- Section IV (Subject-Aware CV)
- Section V (Audit-Ready Artifacts)

Classes:
    OuterFoldOrchestrator: Orchestrates one complete outer fold
"""

from __future__ import annotations
from typing import Dict, List, Callable, Any
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, cohen_kappa_score

# Import refactored modules
from .metrics import ObjectiveComputer
from .checkpointing import CheckpointManager
from .inner_loop import InnerTrainer
from .evaluation import OuterEvaluator
from ..artifacts.plot_builders import PlotTitleBuilder

# Import plotting functions
try:
    from utils import plots as _plots
    plot_confusion = _plots.plot_confusion
    plot_curves = _plots.plot_curves
except Exception:
    def plot_confusion(*args, **kwargs):
        pass
    def plot_curves(*args, **kwargs):
        pass

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_plurality_correctness(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute plurality correctness (imported from training_runner).
    
    For each true class, checks if the correct prediction is the most frequent.
    Returns proportion of classes where correct prediction is plurality (0.0-1.0).
    """
    from sklearn.metrics import confusion_matrix
    
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


class OuterFoldOrchestrator:
    """
    Orchestrates one complete outer fold of nested cross-validation.
    
    Responsibilities:
    - Run inner K-fold loop (training on inner train, validating on inner val)
    - Setup models/optimizers/losses for each inner fold
    - Call InnerTrainer for each inner fold training
    - Aggregate inner results across K folds
    - Select best inner model based on objective metric
    - Coordinate outer test evaluation (ensemble or refit)
    - Generate per-fold plots and metrics
    - Return comprehensive fold results
    
    This class extracts the outer fold loop body from TrainingRunner.run(),
    reducing the orchestrator to high-level coordination.
    
    Example usage:
        >>> orchestrator = OuterFoldOrchestrator(
        ...     cfg=cfg,
        ...     objective_computer=objective_computer,
        ...     inner_trainer=inner_trainer,
        ...     outer_evaluator=outer_evaluator,
        ...     plot_title_builder=plot_title_builder,
        ... )
        >>> 
        >>> fold_result = orchestrator.run_fold(
        ...     fold=0,
        ...     tr_idx=tr_idx,
        ...     te_idx=te_idx,
        ...     dataset=dataset,
        ...     y_all=y_all,
        ...     groups=groups,
        ...     class_names=["low", "high"],
        ...     model_builder=model_builder,
        ...     aug_transform=aug_transform,
        ...     input_adapter=input_adapter,
        ...     predefined_inner_splits=None,
        ...     optuna_trial=None,
        ...     global_step_offset=0,
        ... )
    """
    
    def __init__(
        self,
        cfg: Dict,
        objective_computer: ObjectiveComputer,
        inner_trainer: InnerTrainer,
        outer_evaluator: OuterEvaluator,
        plot_title_builder: PlotTitleBuilder,
        run_dir=None,
        log_event: Callable | None = None,
    ):
        """
        Initialize OuterFoldOrchestrator.
        
        Args:
            cfg: Configuration dictionary
            objective_computer: ObjectiveComputer for metric computation
            inner_trainer: InnerTrainer for inner fold training
            outer_evaluator: OuterEvaluator for outer test evaluation
            plot_title_builder: PlotTitleBuilder for plot titles
            run_dir: Optional run directory for saving artifacts
            log_event: Optional logging function
        """
        self.cfg = cfg
        self.objective_computer = objective_computer
        self.inner_trainer = inner_trainer
        self.outer_evaluator = outer_evaluator
        self.plot_title_builder = plot_title_builder
        self.run_dir = run_dir
        self.log_event = log_event if log_event else lambda *args, **kwargs: None
    
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
        aug_builder,
        aug_transform,
        input_adapter: Callable | None,
        predefined_inner_splits: List[dict] | None,
        optuna_trial,
        global_step_offset: int,
    ) -> Dict:
        """
        Run one complete outer fold.
        
        Args:
            fold: Outer fold index (0-based)
            tr_idx: Outer train indices
            te_idx: Outer test indices
            dataset: Dataset object
            y_all: All labels
            groups: Subject group assignments
            class_names: List of class names
            model_builder: Function to build model
            aug_transform: Augmentation transform
            input_adapter: Optional input adapter function
            predefined_inner_splits: Optional predefined inner splits
            optuna_trial: Optional Optuna trial for pruning
            global_step_offset: Global step offset for Optuna
        
        Returns:
            Dictionary containing:
                - fold: Fold index
                - test_subjects: List of test subject IDs
                - y_true: True labels for outer test set
                - y_pred: Predicted labels for outer test set
                - metrics: Fold metrics (acc, macro_f1, etc.)
                - inner_results: List of inner fold results
                - best_inner_result: Best inner fold result
                - fold_record: Fold record for splits_indices.json
                - learning_curves: Learning curve rows
                - test_pred_rows_outer: Outer test prediction rows
                - test_pred_rows_inner: Inner validation prediction rows
                - new_global_step: Updated global step for Optuna
        """
        num_cls = len(class_names)
        
        # Validate outer fold (no subject leakage)
        test_subjects = np.unique(groups[te_idx]).tolist()
        train_subjects = np.unique(groups[tr_idx]).tolist()
        overlap = set(train_subjects).intersection(set(test_subjects))
        if overlap:
            raise AssertionError(f"Subject leakage detected in outer fold {fold+1}: {sorted(list(overlap))}")
        
        print(
            f"[fold {fold+1}] test_subjects={test_subjects} n_tr={len(tr_idx)} n_te={len(te_idx)}",
            flush=True,
        )
        
        # Setup inner K-fold
        inner_k = int(self.cfg.get("inner_n_folds", 1))
        unique_train_groups = np.unique(groups[tr_idx])
        if inner_k < 2:
            raise ValueError("inner_n_folds must be >= 2 for strict inner K-fold CV")
        if len(unique_train_groups) < inner_k:
            raise ValueError(
                f"Not enough unique groups for inner K-fold: have {len(unique_train_groups)}, need >= {inner_k}"
            )
        
        # Prepare fold record for splits_indices.json
        fold_record = {
            "fold": int(fold + 1),
            "outer_train_idx": [int(i) for i in tr_idx.tolist()],
            "outer_test_idx": [int(i) for i in te_idx.tolist()],
            "outer_train_subjects": [int(s) for s in np.unique(groups[tr_idx]).tolist()],
            "outer_test_subjects": [int(s) for s in np.unique(groups[te_idx]).tolist()],
            "inner_splits": [],
        }
        
        # Get inner splits (predefined or computed)
        inner_iter = self._get_inner_splits(
            predefined_inner_splits=predefined_inner_splits,
            inner_k=inner_k,
            tr_idx=tr_idx,
            y_all=y_all,
            groups=groups,
        )
        
        # Run inner K-fold loop
        inner_results, learning_curves, test_pred_rows_inner, global_step = self._run_inner_loop(
            inner_iter=inner_iter,
            fold=fold,
            tr_idx=tr_idx,
            te_idx=te_idx,
            dataset=dataset,
            y_all=y_all,
            groups=groups,
            class_names=class_names,
            model_builder=model_builder,
            aug_builder=aug_builder,
            aug_transform=aug_transform,
            input_adapter=input_adapter,
            optuna_trial=optuna_trial,
            global_step_offset=global_step_offset,
            num_cls=num_cls,
            fold_record=fold_record,
        )
        
        # Aggregate inner metrics
        inner_metrics = self._aggregate_inner_metrics_all(inner_results)
        
        # Select best inner model
        best_inner_result = self._select_best_inner_model(inner_results)
        
        # Evaluate on outer test set (ensure inner_results are ordered by inner_fold)
        # This protects ensemble averaging order against any iterator differences
        ordered_inner_results = sorted(
            inner_results,
            key=lambda r: int(r.get("fold_info", {}).get("inner_fold", 0))
            if isinstance(r.get("fold_info"), dict) else 0
        )
        eval_result = self._evaluate_outer_test(
            model_builder=model_builder,
            num_cls=num_cls,
            inner_results=ordered_inner_results,
            dataset=dataset,
            y_all=y_all,
            groups=groups,
            tr_idx=tr_idx,
            te_idx=te_idx,
            aug_transform=aug_transform,
            input_adapter=input_adapter,
            class_names=class_names,
            fold=fold + 1,
        )
        
        # Compute fold metrics
        fold_metrics = self._compute_fold_metrics(
            y_true=eval_result["y_true"],
            y_pred=eval_result["y_pred"],
        )
        
        # Generate plots (if run_dir exists)
        if self.run_dir and best_inner_result:
            self._generate_fold_plots(
                fold=fold,
                test_subjects=test_subjects,
                y_true=eval_result["y_true"],
                y_pred=eval_result["y_pred"],
                class_names=class_names,
                best_inner_result=best_inner_result,
                inner_metrics=inner_metrics,
                outer_metrics=fold_metrics,
                per_class_f1=fold_metrics.get("per_class_f1_list"),
            )
        
        # Log fold completion
        self.log_event("fold_end", f"Completed outer fold {fold+1}", fold=fold+1)
        print(
            f"[fold {fold+1}] acc={fold_metrics['acc']:.2f} kappa={fold_metrics['cohen_kappa']:.3f} "
            f"inner_mean_acc={inner_metrics['acc']:.2f} inner_mean_macro_f1={inner_metrics['macro_f1']:.2f}",
            flush=True,
        )
        
        # Return comprehensive fold result
        return {
            "fold": fold,
            "test_subjects": test_subjects,
            "y_true": eval_result["y_true"],
            "y_pred": eval_result["y_pred"],
            "metrics": fold_metrics,
            "inner_results": inner_results,
            "best_inner_result": best_inner_result,
            "fold_record": fold_record,
            "learning_curves": learning_curves,
            "test_pred_rows_outer": eval_result["test_pred_rows"],
            "test_pred_rows_inner": test_pred_rows_inner,
            "new_global_step": global_step,
            "inner_metrics": inner_metrics,  # For plotting and aggregation
        }
    
    def _get_inner_splits(
        self,
        predefined_inner_splits: List[dict] | None,
        inner_k: int,
        tr_idx: np.ndarray,
        y_all: np.ndarray,
        groups: np.ndarray,
    ) -> List[tuple]:
        """
        Get inner fold splits (predefined or computed via GroupKFold).
        
        Returns:
            List of (inner_train_idx, inner_val_idx) tuples
        """
        if predefined_inner_splits:
            return [
                (np.array(sp["inner_train_idx"]), np.array(sp["inner_val_idx"]))
                for sp in predefined_inner_splits
            ]
        else:
            gkf_inner = GroupKFold(n_splits=inner_k)
            return [
                (tr_idx[np.array(inner_tr_rel)], tr_idx[np.array(inner_va_rel)])
                for inner_tr_rel, inner_va_rel in gkf_inner.split(
                    np.zeros(len(tr_idx)), y_all[tr_idx], groups[tr_idx]
                )
            ]
    
    def _run_inner_loop(
        self,
        inner_iter: List[tuple],
        fold: int,
        tr_idx: np.ndarray,
        te_idx: np.ndarray,
        dataset,
        y_all: np.ndarray,
        groups: np.ndarray,
        class_names: List[str],
        model_builder: Callable,
        aug_builder,
        aug_transform,
        input_adapter: Callable | None,
        optuna_trial,
        global_step_offset: int,
        num_cls: int,
        fold_record: Dict,
    ) -> tuple:
        """
        Run inner K-fold loop.
        
        Returns:
            Tuple of (inner_results, learning_curves, test_pred_rows_inner, global_step)
        """
        inner_results = []
        learning_curves = []
        test_pred_rows_inner = []
        global_step = global_step_offset
        
        for inner_fold, (inner_tr_abs, inner_va_abs) in enumerate(inner_iter):
            # Validate inner split (no subject leakage)
            self._validate_inner_split(groups, inner_tr_abs, inner_va_abs, fold, inner_fold)
            
            # Record inner split
            fold_record["inner_splits"].append({
                "inner_fold": int(inner_fold + 1),
                "inner_train_idx": [int(i) for i in inner_tr_abs.tolist()],
                "inner_val_idx": [int(i) for i in inner_va_abs.tolist()],
            })
            
            # Create dataloaders
            tr_ld, va_ld, te_ld = self._make_loaders(
                dataset=dataset,
                y_all=y_all,
                groups=groups,
                tr_idx=tr_idx,
                te_idx=te_idx,
                aug_transform=aug_transform,
                inner_tr_abs=inner_tr_abs,
                inner_va_abs=inner_va_abs,
            )
            
            # Setup model, optimizer, scheduler, loss
            model, opt, sched, loss_fn = self._setup_training_components(
                model_builder=model_builder,
                num_cls=num_cls,
                y_all=y_all,
                inner_tr_abs=inner_tr_abs,
                fold=fold,
                inner_fold=inner_fold,
            )
            
            # Initialize checkpoint manager
            checkpoint_mgr = CheckpointManager(self.cfg, self.objective_computer)
            
            # Train using InnerTrainer
            inner_result = self.inner_trainer.train(
                model=model,
                optimizer=opt,
                scheduler=sched,
                loss_fn=loss_fn,
                tr_loader=tr_ld,
                va_loader=va_ld,
                aug_builder=aug_builder,  # preserve dynamic augmentation warmup behavior
                input_adapter=input_adapter,
                checkpoint_manager=checkpoint_mgr,
                fold_info={"outer_fold": fold + 1, "inner_fold": inner_fold + 1},
                optuna_trial=optuna_trial,
                global_step_offset=global_step,
            )
            
            # Update global step
            global_step += len(inner_result["learning_curves"])
            
            # Collect learning curves
            learning_curves.extend(inner_result["learning_curves"])
            
            # Save checkpoint if requested
            if self.run_dir and self.cfg.get("save_ckpt", True) and inner_result["best_state"] is not None:
                self._save_checkpoint(inner_result["best_state"], fold, inner_fold)
            
            # Collect inner validation predictions
            if inner_result["best_state"] is not None:
                inner_val_preds = self._collect_inner_val_predictions(
                    model=model,
                    best_state=inner_result["best_state"],
                    va_ld=va_ld,
                    inner_va_abs=inner_va_abs,
                    groups=groups,
                    class_names=class_names,
                    fold=fold,
                    inner_fold=inner_fold,
                    input_adapter=input_adapter,
                )
                test_pred_rows_inner.extend(inner_val_preds)
            
            # Store inner result
            inner_results.append(inner_result)
        
        return inner_results, learning_curves, test_pred_rows_inner, global_step
    
    def _validate_inner_split(
        self,
        groups: np.ndarray,
        inner_tr_abs: np.ndarray,
        inner_va_abs: np.ndarray,
        outer_fold: int,
        inner_fold: int,
    ):
        """Validate that inner split has no subject leakage."""
        tr_subj = set(np.unique(groups[inner_tr_abs]).tolist())
        va_subj = set(np.unique(groups[inner_va_abs]).tolist())
        if tr_subj.intersection(va_subj):
            raise AssertionError(
                f"Subject leakage detected in inner fold {inner_fold+1} of outer {outer_fold+1}: "
                f"{sorted(list(tr_subj.intersection(va_subj)))}"
            )
    
    def _make_loaders(
        self,
        dataset,
        y_all: np.ndarray,
        groups: np.ndarray,
        tr_idx: np.ndarray,
        te_idx: np.ndarray,
        aug_transform,
        inner_tr_abs: np.ndarray,
        inner_va_abs: np.ndarray,
    ) -> tuple:
        """
        Create DataLoaders for inner train, inner val, and outer test.
        
        Returns:
            Tuple of (tr_loader, va_loader, te_loader)
        """
        dataset_tr = copy.copy(dataset)
        dataset_eval = copy.copy(dataset)
        # Training transform will be set dynamically by InnerTrainer via aug_builder
        dataset_tr.set_transform(None)
        # Evaluation/validation/test must not apply augmentation
        dataset_eval.set_transform(None)

        num_workers = 0  # safest on Windows
        g = torch.Generator()
        if self.cfg.get("seed") is not None:
            g.manual_seed(int(self.cfg["seed"]))

        batch_size = int(self.cfg.get("batch_size", 16))

        # Import _seed_worker if needed
        from ..training_runner import _seed_worker

        tr_ld = DataLoader(
            Subset(dataset_tr, inner_tr_abs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            generator=g,
        )
        va_ld = DataLoader(
            Subset(dataset_eval, inner_va_abs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        te_ld = DataLoader(
            Subset(dataset_eval, te_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return tr_ld, va_ld, te_ld
    
    def _setup_training_components(
        self,
        model_builder: Callable,
        num_cls: int,
        y_all: np.ndarray,
        inner_tr_abs: np.ndarray,
        fold: int,
        inner_fold: int,
    ) -> tuple:
        """
        Setup model, optimizer, scheduler, and loss function.
        
        Returns:
            Tuple of (model, optimizer, scheduler, loss_fn)
        """
        # Model
        model = model_builder(self.cfg, num_cls).to(DEVICE)
        
        # Optimizer
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.cfg.get("lr", 7e-4)),
            weight_decay=float(self.cfg.get("weight_decay", 0.0)),
        )
        
        # Scheduler
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=int(self.cfg.get("scheduler_patience", 5))
        )
        
        # Class weights (computed from inner-train only)
        cls_w = compute_class_weight(
            "balanced", classes=np.arange(num_cls), y=y_all[inner_tr_abs]
        )
        loss_fn = nn.CrossEntropyLoss(torch.tensor(cls_w, dtype=torch.float32, device=DEVICE))
        
        # Save class weights
        if self.run_dir:
            self._save_class_weights(cls_w, fold, inner_fold)
        
        return model, opt, sched, loss_fn
    
    def _save_class_weights(self, cls_w: np.ndarray, fold: int, inner_fold: int):
        """Save class weights to JSON."""
        try:
            cw_dir = self.run_dir / "class_weights"
            cw_dir.mkdir(parents=True, exist_ok=True)
            cw_path = cw_dir / f"fold_{fold+1:02d}_inner_{inner_fold+1:02d}_class_weights.json"
            cw_payload = {str(i): float(w) for i, w in enumerate(cls_w)}
            cw_path.write_text(json.dumps(cw_payload, indent=2))
            self.log_event(
                "class_weights_saved",
                f"Saved class weights for fold {fold+1} inner {inner_fold+1}",
                fold=fold+1,
                inner_fold=inner_fold+1,
            )
        except Exception:
            pass
    
    def _save_checkpoint(self, best_state: Dict, fold: int, inner_fold: int):
        """Save best checkpoint."""
        try:
            ckpt_dir = self.run_dir / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_dir / f"fold_{fold+1:02d}_inner_{inner_fold+1:02d}_best.ckpt")
        except Exception:
            pass
    
    def _collect_inner_val_predictions(
        self,
        model: nn.Module,
        best_state: Dict,
        va_ld: DataLoader,
        inner_va_abs: np.ndarray,
        groups: np.ndarray,
        class_names: List[str],
        fold: int,
        inner_fold: int,
        input_adapter: Callable | None,
    ) -> List[Dict]:
        """Collect inner validation predictions using best model state."""
        model.load_state_dict(best_state)
        model.eval()

        test_pred_rows = []
        # Iterate through the provided absolute validation indices in the same
        # order the legacy runner emitted them when DataLoader provided the
        # Subset sequentially.
        va_indices_iter = iter(inner_va_abs.tolist())
        with torch.no_grad():
            for xb, yb in va_ld:
                yb_gpu = yb.to(DEVICE)
                xb_gpu = (
                    xb.to(DEVICE)
                    if not isinstance(xb, (list, tuple))
                    else [t.to(DEVICE) for t in xb]
                )
                xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu
                out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                probs = F.softmax(out.float(), dim=1).cpu()
                preds = probs.argmax(1)
                bsz = yb.size(0)

                for j in range(bsz):
                    abs_idx = int(next(va_indices_iter, -1))
                    subj_id = int(groups[abs_idx]) if abs_idx >= 0 else -1
                    true_lbl = int(yb[j].item())
                    pred_lbl = int(preds[j].item())
                    probs_vec = probs[j].tolist()
                    p_true = float(probs_vec[true_lbl]) if 0 <= true_lbl < len(probs_vec) else 0.0
                    logp_true = float(np.log(max(p_true, 1e-12)))

                    test_pred_rows.append({
                        "outer_fold": int(fold + 1),
                        "inner_fold": int(inner_fold + 1),
                        "trial_index": abs_idx,
                        "subject_id": subj_id,
                        "true_label_idx": true_lbl,
                        "true_label_name": str(class_names[true_lbl]) if 0 <= true_lbl < len(class_names) else "",
                        "pred_label_idx": pred_lbl,
                        "pred_label_name": str(class_names[pred_lbl]) if 0 <= pred_lbl < len(class_names) else "",
                        "correct": int(1 if pred_lbl == true_lbl else 0),
                        "p_trueclass": p_true,
                        "logp_trueclass": logp_true,
                        "probs": json.dumps(probs_vec),
                    })

        return test_pred_rows
    
    def _aggregate_inner_metrics_all(self, inner_results: List[Dict]) -> Dict:
        """Aggregate inner metrics across all K folds."""
        if not inner_results:
            return {
                "acc": 0.0,
                "macro_f1": 0.0,
                "min_per_class_f1": 0.0,
                "plur_corr": 0.0,
            }
        
        return {
            "acc": float(np.mean([r["best_inner_acc"] for r in inner_results])),
            "macro_f1": float(np.mean([r["best_inner_macro_f1"] for r in inner_results])),
            "min_per_class_f1": float(np.mean([r["best_inner_min_per_class_f1"] for r in inner_results])),
            "plur_corr": float(np.mean([r["best_inner_plur_corr"] for r in inner_results])),
        }
    
    def _aggregate_inner_metrics(self, inner_results: List[Dict], metric_key: str) -> float:
        """Aggregate a specific inner metric across all K folds."""
        if not inner_results:
            return 0.0
        return float(np.mean([r[metric_key] for r in inner_results]))
    
    def _select_best_inner_model(self, inner_results: List[Dict]) -> Dict | None:
        """Select best inner model based on optuna_objective."""
        if not inner_results:
            return None
        
        objective = self.cfg.get("optuna_objective")
        if not objective:
            raise ValueError(
                "'optuna_objective' must be specified. "
                "Choose from: inner_mean_macro_f1, inner_mean_min_per_class_f1, inner_mean_plur_corr, inner_mean_acc, composite_min_f1_plur_corr"
            )
        
        # Special handling for composite objective
        if objective == "composite_min_f1_plur_corr":
            params = self.objective_computer.get_params()
            
            def compute_composite_score(r):
                min_f1 = r["best_inner_min_per_class_f1"]
                plur_corr = r["best_inner_plur_corr"]
                if params["mode"] == "threshold":
                    if min_f1 < params["threshold"]:
                        return min_f1 * 0.1
                    else:
                        return plur_corr
                else:  # weighted
                    return params["weight"] * min_f1 + (1.0 - params["weight"]) * plur_corr
            
            return max(inner_results, key=compute_composite_score)
        else:
            objective_to_metric = {
                "inner_mean_macro_f1": "best_inner_macro_f1",
                "inner_mean_min_per_class_f1": "best_inner_min_per_class_f1",
                "inner_mean_plur_corr": "best_inner_plur_corr",
                "inner_mean_acc": "best_inner_acc",
            }
            
            if objective not in objective_to_metric:
                raise ValueError(
                    f"Invalid optuna_objective: '{objective}'. "
                    f"Must be one of: {list(objective_to_metric.keys())} or composite_min_f1_plur_corr"
                )
            
            metric_key = objective_to_metric[objective]
            return max(inner_results, key=lambda r: r[metric_key])
    
    def _evaluate_outer_test(
        self,
        model_builder: Callable,
        num_cls: int,
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
        """Evaluate on outer test set (ensemble or refit)."""
        mode = self.outer_evaluator.mode
        
        # Create test loader for ensemble mode (no augmentation, deterministic order)
        dataset_eval = copy.copy(dataset)
        dataset_eval.set_transform(None)
        te_ld = DataLoader(
            Subset(dataset_eval, te_idx),
            batch_size=int(self.cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=0,
        )
        
        if mode == "ensemble":
            return self.outer_evaluator.evaluate(
                model_builder=model_builder,
                num_classes=num_cls,
                inner_results=inner_results,
                test_loader=te_ld,
                groups=groups,
                te_idx=te_idx,
                class_names=class_names,
                fold=fold,
                input_adapter=input_adapter,
            )
        elif mode == "refit":
            return self.outer_evaluator.evaluate_refit(
                model_builder=model_builder,
                num_classes=num_cls,
                dataset=dataset,
                y_all=y_all,
                groups=groups,
                tr_idx=tr_idx,
                te_idx=te_idx,
                test_loader=te_ld,
                aug_transform=aug_transform,
                input_adapter=input_adapter,
                class_names=class_names,
                fold=fold,
            )
        else:
            raise ValueError(f"Unknown outer_eval_mode={mode}")
    
    def _compute_fold_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """Compute metrics for this fold."""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        total = len(y_true)
        acc = 100.0 * correct / max(1, total)
        
        try:
            macro_f1 = f1_score(y_true, y_pred, average="macro") * 100 if y_true else 0.0
            per_class_f1 = f1_score(y_true, y_pred, average=None).tolist() if y_true else None
            plur_corr = compute_plurality_correctness(y_true, y_pred) * 100 if y_true else 0.0
            kappa = cohen_kappa_score(y_true, y_pred) if y_true else 0.0
        except Exception:
            macro_f1 = 0.0
            per_class_f1 = None
            plur_corr = 0.0
            kappa = 0.0
        
        min_per_class_f1 = (
            float(np.min(per_class_f1)) * 100
            if per_class_f1 is not None and len(per_class_f1) > 0
            else 0.0
        )
        
        return {
            "acc": acc,
            "macro_f1": macro_f1,
            "min_per_class_f1": min_per_class_f1,
            "plur_corr": plur_corr,
            "cohen_kappa": kappa,
            "per_class_f1_list": per_class_f1,  # For plotting
        }
    
    def _generate_fold_plots(
        self,
        fold: int,
        test_subjects: List[int],
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        best_inner_result: Dict,
        inner_metrics: Dict,
        outer_metrics: Dict,
        per_class_f1: List[float] | None,
    ):
        """Generate per-fold plots."""
        plots_dir = self.run_dir / "plots_outer"
        plots_dir.mkdir(parents=True, exist_ok=True)
        run_prefix = self.run_dir.name if self.run_dir else "run"
        fold_tag = f"fold{fold+1:02d}"
        
        # Simple fold plots
        fold_title = self.plot_title_builder.build_fold_title_simple(
            fold=fold,
            test_subjects=test_subjects,
            inner_metrics=inner_metrics,
            outer_acc=outer_metrics["acc"],
        )
        
        plot_confusion(
            y_true,
            y_pred,
            class_names,
            plots_dir / f"{run_prefix}_{fold_tag}_confusion.png",
            title=fold_title,
        )
        
        plot_curves(
            best_inner_result["tr_hist"],
            best_inner_result["va_hist"],
            best_inner_result["va_acc_hist"],
            plots_dir / f"{run_prefix}_{fold_tag}_curves.png",
            title=fold_title,
        )
        
        # Enhanced plots
        plots_enhanced_dir = self.run_dir / "plots_outer_enhanced"
        plots_enhanced_dir.mkdir(parents=True, exist_ok=True)
        
        fold_title_enhanced = self.plot_title_builder.build_fold_title_enhanced(
            fold=fold,
            test_subjects=test_subjects,
            inner_metrics=inner_metrics,
            outer_metrics=outer_metrics,
            per_class_f1=per_class_f1,
        )
        
        per_class_info_lines = self.plot_title_builder.build_per_class_info(per_class_f1, class_names)
        
        plot_confusion(
            y_true,
            y_pred,
            class_names,
            plots_enhanced_dir / f"{run_prefix}_{fold_tag}_confusion.png",
            title=fold_title_enhanced,
            hyper_lines=per_class_info_lines if per_class_info_lines else None,
        )
        
        plot_curves(
            best_inner_result["tr_hist"],
            best_inner_result["va_hist"],
            best_inner_result["va_acc_hist"],
            plots_enhanced_dir / f"{run_prefix}_{fold_tag}_curves.png",
            title=fold_title_enhanced,
        )

