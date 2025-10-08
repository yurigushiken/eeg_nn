"""
CSV artifact writers for training pipeline outputs.

This module provides dedicated writer classes for each CSV artifact type:
- Learning curves (per-epoch metrics for all inner folds)
- Outer evaluation metrics (per-fold and aggregate metrics)
- Test predictions (inner validation and outer test predictions)

Constitutional compliance: Section V (Audit-Ready Artifact Retention)
- All artifacts have consistent format
- Column order matches original implementation
- Audit-ready with reproducible outputs

Classes:
    LearningCurvesWriter: Writes per-epoch training/validation metrics
    OuterEvalMetricsWriter: Writes per-fold evaluation metrics with aggregate
    TestPredictionsWriter: Writes per-trial predictions with probabilities
"""

from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import csv


class LearningCurvesWriter:
    """
    Writes learning curves CSV for all inner folds across all outer folds.
    
    CSV format:
        outer_fold,inner_fold,epoch,train_loss,val_loss,val_acc,val_macro_f1,
        val_min_per_class_f1,val_plur_corr,val_objective_metric,n_train,n_val,
        optuna_trial_id,param_hash
    
    Usage:
        >>> writer = LearningCurvesWriter(run_dir)
        >>> writer.write(learning_curve_rows)
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize writer with output directory.
        
        Args:
            run_dir: Path to run directory where CSV will be saved
        """
        self.csv_path = run_dir / "learning_curves_inner.csv"
    
    def write(self, rows: List[Dict]):
        """
        Write learning curves to CSV.
        
        Args:
            rows: List of dictionaries, each containing:
                - outer_fold: int
                - inner_fold: int
                - epoch: int
                - train_loss: float
                - val_loss: float
                - val_acc: float
                - val_macro_f1: float
                - val_min_per_class_f1: float
                - val_plur_corr: float
                - val_objective_metric: float
                - n_train: int
                - n_val: int
                - optuna_trial_id: int (-1 if not Optuna)
                - param_hash: str (optional)
        """
        if not rows:
            # Don't create empty CSV or create empty file
            if self.csv_path.exists():
                try:
                    self.csv_path.unlink()
                except Exception:
                    pass
            return
        
        fieldnames = [
            "outer_fold",
            "inner_fold",
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "val_macro_f1",
            "val_min_per_class_f1",
            "val_plur_corr",
            "val_objective_metric",
            "n_train",
            "n_val",
            "optuna_trial_id",
            "param_hash",
        ]
        
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


class OuterEvalMetricsWriter:
    """
    Writes outer evaluation metrics CSV with per-fold and aggregate rows.
    
    CSV format:
        outer_fold,test_subjects,n_test_trials,acc,acc_std,macro_f1,macro_f1_std,
        min_per_class_f1,min_per_class_f1_std,plur_corr,plur_corr_std,
        cohen_kappa,cohen_kappa_std,per_class_f1
    
    Last row has outer_fold="OVERALL" with aggregate statistics.
    
    Usage:
        >>> writer = OuterEvalMetricsWriter(run_dir)
        >>> writer.write(fold_rows, aggregate_row)
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize writer with output directory.
        
        Args:
            run_dir: Path to run directory where CSV will be saved
        """
        self.csv_path = run_dir / "outer_eval_metrics.csv"
    
    def write(self, rows: List[Dict], aggregate_row: Dict):
        """
        Write outer evaluation metrics to CSV.
        
        Args:
            rows: List of per-fold dictionaries containing:
                - outer_fold: int
                - test_subjects: str (comma-separated)
                - n_test_trials: int
                - acc: float
                - acc_std: str (empty for per-fold rows)
                - macro_f1: float
                - macro_f1_std: str (empty for per-fold rows)
                - min_per_class_f1: float
                - min_per_class_f1_std: str (empty for per-fold rows)
                - plur_corr: float
                - plur_corr_std: str (empty for per-fold rows)
                - cohen_kappa: float
                - cohen_kappa_std: str (empty for per-fold rows)
                - per_class_f1: str (JSON array or empty)
            aggregate_row: Dictionary with same fields but outer_fold="OVERALL"
                and std fields populated
        """
        fieldnames = [
            "outer_fold",
            "test_subjects",
            "n_test_trials",
            "acc",
            "acc_std",
            "macro_f1",
            "macro_f1_std",
            "min_per_class_f1",
            "min_per_class_f1_std",
            "plur_corr",
            "plur_corr_std",
            "cohen_kappa",
            "cohen_kappa_std",
            "per_class_f1",
        ]
        
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            # Append aggregate row
            writer.writerow(aggregate_row)


class TestPredictionsWriter:
    """
    Writes per-trial test predictions CSV (inner validation or outer test).
    
    Supports two modes:
    - 'inner': Predictions on inner validation sets (includes inner_fold column)
    - 'outer': Predictions on outer test sets (no inner_fold column)
    
    CSV format (inner):
        outer_fold,inner_fold,trial_index,subject_id,true_label_idx,true_label_name,
        pred_label_idx,pred_label_name,correct,p_trueclass,logp_trueclass,probs
    
    CSV format (outer):
        outer_fold,trial_index,subject_id,true_label_idx,true_label_name,
        pred_label_idx,pred_label_name,correct,p_trueclass,logp_trueclass,probs
    
    Usage:
        >>> writer = TestPredictionsWriter(run_dir, mode='outer')
        >>> writer.write(prediction_rows)
    """
    
    def __init__(self, run_dir: Path, mode: str):
        """
        Initialize writer with output directory and mode.
        
        Args:
            run_dir: Path to run directory where CSV will be saved
            mode: Either 'inner', 'outer', or 'outer_thresholded'
        
        Raises:
            ValueError: If mode is not valid
        """
        if mode not in ("inner", "outer", "outer_thresholded"):
            raise ValueError(f"mode must be 'inner', 'outer', or 'outer_thresholded', got '{mode}'")
        
        self.mode = mode
        # Map mode to filename
        if mode == "outer_thresholded":
            self.csv_path = run_dir / "test_predictions_outer_thresholded.csv"
        else:
            self.csv_path = run_dir / f"test_predictions_{mode}.csv"
    
    def write(self, rows: List[Dict]):
        """
        Write test predictions to CSV.
        
        Args:
            rows: List of prediction dictionaries containing:
                - outer_fold: int
                - inner_fold: int (only for mode='inner')
                - trial_index: int (dataset index)
                - subject_id: int
                - true_label_idx: int
                - true_label_name: str
                - pred_label_idx: int
                - pred_label_name: str
                - correct: int (1 if correct, 0 if incorrect)
                - p_trueclass: float (probability of true class)
                - logp_trueclass: float (log probability)
                - probs: str (JSON array of all class probabilities)
        """
        if not rows:
            # Don't create empty CSV
            return
        
        if self.mode == "inner":
            fieldnames = [
                "outer_fold",
                "inner_fold",
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
        else:  # outer or outer_thresholded (same fieldnames, no inner_fold)
            fieldnames = [
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
        
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

