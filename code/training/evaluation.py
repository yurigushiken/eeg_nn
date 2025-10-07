"""
Outer test evaluation for nested cross-validation.

This module handles evaluation on the outer test set using either:
1. Ensemble mode: Average predictions from K inner models
2. Refit mode: Train a single model on full outer train set

Constitutional compliance: Section IV (Rigorous Validation & Reporting)

Classes:
    OuterEvaluator: Handles outer test evaluation
"""

from __future__ import annotations
from typing import Dict, Callable, List, Any
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight


class OuterEvaluator:
    """
    Handles outer test evaluation (ensemble or refit modes).
    
    Two evaluation modes:
    
    **Ensemble Mode** (recommended):
    - Load K inner models (from K inner folds)
    - Average their softmax predictions
    - Predict using ensemble average
    - Benefits: More stable, utilizes all inner models
    
    **Refit Mode** (alternative):
    - Train a single model on full outer-train set
    - Optional small validation set for early stopping
    - Predict using refit model
    - Benefits: Simpler, single model per fold
    
    Example usage:
        >>> cfg = {"outer_eval_mode": "ensemble"}
        >>> evaluator = OuterEvaluator(cfg)
        >>> 
        >>> result = evaluator.evaluate(
        ...     model_builder=model_builder,
        ...     num_classes=3,
        ...     inner_results=inner_results,
        ...     test_loader=test_loader,
        ...     groups=groups,
        ...     te_idx=te_idx,
        ...     class_names=class_names,
        ...     fold=1,
        ... )
        >>> 
        >>> y_true = result["y_true"]
        >>> y_pred = result["y_pred"]
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize OuterEvaluator.
        
        Args:
            cfg: Configuration dictionary containing:
                - outer_eval_mode: "ensemble" or "refit"
                - For refit mode:
                    - refit_val_frac: fraction of outer train for validation (default: 0.0)
                    - refit_val_k: number of folds for validation split (default: 5)
                    - refit_early_stop: early stopping patience (default: 10)
                    - epochs: number of training epochs
                    - lr: learning rate
                    - weight_decay: weight decay
        """
        self.cfg = cfg
        self.mode = str(cfg.get("outer_eval_mode", "ensemble")).lower()
        
        if self.mode not in ("ensemble", "refit"):
            raise ValueError(f"Unknown outer_eval_mode: {self.mode}; use 'ensemble' or 'refit'")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate(
        self,
        model_builder: Callable,
        num_classes: int,
        inner_results: List[Dict],
        test_loader: DataLoader,
        groups: np.ndarray,
        te_idx: np.ndarray,
        class_names: List[str],
        fold: int,
        input_adapter: Callable | None = None,
    ) -> Dict:
        """
        Evaluate on outer test set using ensemble of inner models.
        
        Args:
            model_builder: Function that builds model (cfg, num_classes) -> model
            num_classes: Number of classes
            inner_results: List of dicts from inner training, each containing "best_state"
            test_loader: DataLoader for test set
            groups: Subject groups array
            te_idx: Test set indices
            class_names: List of class names
            fold: Outer fold number
        
        Returns:
            Dictionary containing:
                - y_true: List of true labels
                - y_pred: List of predicted labels
                - test_pred_rows: List of dicts (per-trial predictions)
        """
        if self.mode == "ensemble":
            return self._evaluate_ensemble(
                model_builder,
                num_classes,
                inner_results,
                test_loader,
                groups,
                te_idx,
                class_names,
                fold,
                input_adapter,
            )
        else:
            raise ValueError("Use evaluate_ensemble() directly for ensemble mode, or evaluate_refit() for refit mode")
    
    def _evaluate_ensemble(
        self,
        model_builder: Callable,
        num_classes: int,
        inner_results: List[Dict],
        test_loader: DataLoader,
        groups: np.ndarray,
        te_idx: np.ndarray,
        class_names: List[str],
        fold: int,
        input_adapter: Callable | None,
    ) -> Dict:
        """Evaluate using ensemble of K inner models."""
        y_true_fold: List[int] = []
        y_pred_fold: List[int] = []
        test_pred_rows: List[Dict] = []
        
        with torch.no_grad():
            pos = 0
            for xb, yb in test_loader:
                yb_gpu = yb.to(self.device)
                xb_gpu = xb.to(self.device) if not isinstance(xb, (list, tuple)) else [t.to(self.device) for t in xb]
                xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu
                
                # Ensemble: average softmax predictions from all inner models
                accum_probs = None
                for r in inner_results:
                    state = r.get("best_state")
                    if state is None:
                        continue
                    model = model_builder(self.cfg, num_classes).to(self.device)
                    model.load_state_dict(state)
                    model.eval()
                    out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                    probs = F.softmax(out.float(), dim=1).cpu()
                    if accum_probs is None:
                        accum_probs = probs
                    else:
                        accum_probs += probs
                
                if accum_probs is None:
                    continue
                
                preds = accum_probs.argmax(1)
                y_true_fold.extend(yb.tolist())
                y_pred_fold.extend(preds.tolist())
                
                # Collect per-trial predictions
                probs_norm = accum_probs / accum_probs.sum(dim=1, keepdim=True)
                bsz = yb.size(0)
                for j in range(bsz):
                    abs_idx = int(te_idx[pos + j])
                    subj_id = int(groups[abs_idx])
                    true_lbl = int(yb[j].item())
                    pred_lbl = int(preds[j].item())
                    probs_vec = probs_norm[j].tolist()
                    p_true = float(probs_vec[true_lbl]) if 0 <= true_lbl < len(probs_vec) else 0.0
                    logp_true = float(np.log(max(p_true, 1e-12)))
                    test_pred_rows.append({
                        "outer_fold": int(fold),
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
                pos += bsz
        
        return {
            "y_true": y_true_fold,
            "y_pred": y_pred_fold,
            "test_pred_rows": test_pred_rows,
        }
    
    def evaluate_refit(
        self,
        model_builder: Callable,
        num_classes: int,
        dataset: Any,
        y_all: np.ndarray,
        groups: np.ndarray,
        tr_idx: np.ndarray,
        te_idx: np.ndarray,
        test_loader: DataLoader,
        aug_transform: Any,
        input_adapter: Callable | None,
        class_names: List[str],
        fold: int,
    ) -> Dict:
        """
        Evaluate using refit mode (single model on full outer train).
        
        Args:
            model_builder: Function that builds model
            num_classes: Number of classes
            dataset: Training dataset
            y_all: All labels
            groups: Subject groups
            tr_idx: Training set indices (outer train)
            te_idx: Test set indices (outer test)
            test_loader: DataLoader for test set
            aug_transform: Augmentation transform
            input_adapter: Input adapter function
            class_names: List of class names
            fold: Outer fold number
        
        Returns:
            Dictionary containing:
                - y_true: List of true labels
                - y_pred: List of predicted labels
                - test_pred_rows: List of dicts (per-trial predictions)
        """
        # Determine validation split (optional)
        refit_val_frac = float(self.cfg.get("refit_val_frac", 0.0) or 0.0)
        do_val = refit_val_frac > 0.0 and len(np.unique(groups[tr_idx])) >= 2
        
        if do_val:
            # Use deterministic GroupKFold; pick first split
            refit_val_k = int(self.cfg.get("refit_val_k", 5))
            gkf_refit = GroupKFold(n_splits=max(2, refit_val_k))
            inner_tr_rel, inner_va_rel = next(
                gkf_refit.split(np.zeros(len(tr_idx)), y_all[tr_idx], groups[tr_idx])
            )
            refit_tr_abs = tr_idx[inner_tr_rel]
            refit_va_abs = tr_idx[inner_va_rel]
        else:
            refit_tr_abs = tr_idx
            refit_va_abs = tr_idx  # Sentinel (no validation)
        
        # Create dataloaders
        dataset_tr = copy.copy(dataset)
        dataset_eval = copy.copy(dataset)
        dataset_tr.set_transform(aug_transform)
        dataset_eval.set_transform(None)
        
        batch_size = int(self.cfg.get("batch_size", 16))
        num_workers = 0
        
        tr_loader = DataLoader(
            Subset(dataset_tr, refit_tr_abs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        va_loader = DataLoader(
            Subset(dataset_eval, refit_va_abs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        # Build model and optimizer
        model = model_builder(self.cfg, num_classes).to(self.device)
        lr = float(self.cfg.get("lr", 7e-4))
        weight_decay = float(self.cfg.get("weight_decay", 0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=int(self.cfg.get("scheduler_patience", 5))
        )
        
        # Compute class weights
        cls_w = compute_class_weight(
            "balanced", classes=np.arange(num_classes), y=y_all[refit_tr_abs]
        )
        loss_fn = nn.CrossEntropyLoss(torch.tensor(cls_w, dtype=torch.float32, device=self.device))
        
        # Training loop
        best_val = float("inf")
        best_state = None
        patience = 0
        refit_patience = int(self.cfg.get("refit_early_stop", self.cfg.get("early_stop", 10)))
        total_epochs = int(self.cfg.get("epochs", 60))
        
        for epoch in range(1, total_epochs + 1):
            # Train
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
                out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                loss = loss_fn(out.float(), yb_gpu)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(1, len(tr_loader))
            
            if do_val:
                # Validate
                model.eval()
                val_loss = 0.0
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
                val_loss /= max(1, len(va_loader))
                scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience += 1
                    if patience >= refit_patience:
                        break
        
        # Load best state if validation was used
        if do_val and best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        
        # Evaluate on test set
        y_true_fold: List[int] = []
        y_pred_fold: List[int] = []
        test_pred_rows: List[Dict] = []
        
        with torch.no_grad():
            pos = 0
            for xb, yb in test_loader:
                yb_gpu = yb.to(self.device)
                xb_gpu = xb.to(self.device) if not isinstance(xb, (list, tuple)) else [t.to(self.device) for t in xb]
                xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu
                out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                logits = out.float().cpu()
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(1)
                y_true_fold.extend(yb.tolist())
                y_pred_fold.extend(preds.tolist())
                
                # Collect per-trial predictions
                bsz = yb.size(0)
                for j in range(bsz):
                    abs_idx = int(te_idx[pos + j])
                    subj_id = int(groups[abs_idx])
                    true_lbl = int(yb[j].item())
                    pred_lbl = int(preds[j].item())
                    probs_vec = probs[j].tolist()
                    p_true = float(probs_vec[true_lbl]) if 0 <= true_lbl < len(probs_vec) else 0.0
                    logp_true = float(np.log(max(p_true, 1e-12)))
                    test_pred_rows.append({
                        "outer_fold": int(fold),
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
                pos += bsz
        
        return {
            "y_true": y_true_fold,
            "y_pred": y_pred_fold,
            "test_pred_rows": test_pred_rows,
        }

