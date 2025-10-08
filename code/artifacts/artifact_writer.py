"""
Artifact writer orchestrator for training runs.

This module handles writing all output artifacts from a training run:
- splits_indices.json (CV split record)
- Learning curves CSV
- Outer evaluation metrics CSV
- Test predictions CSVs (inner and outer)

Constitutional compliance:
- Section V (Audit-Ready Artifacts): All artifacts retained

Classes:
    ArtifactWriterOrchestrator: Coordinates writing of all artifacts
"""

from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import json
import hashlib
import numpy as np

from .csv_writers import (
    LearningCurvesWriter,
    OuterEvalMetricsWriter,
    TestPredictionsWriter,
)


class ArtifactWriterOrchestrator:
    """
    Orchestrates writing of all training artifacts.
    
    Responsibilities:
    - Write splits_indices.json with CV fold information
    - Write learning curves CSV
    - Write outer evaluation metrics CSV with aggregate row
    - Write test predictions CSVs (inner and outer)
    - Respect output toggles from config
    
    Example usage:
        >>> orchestrator = ArtifactWriterOrchestrator(
        ...     run_dir=Path("results/run_123"),
        ...     cfg=cfg,
        ...     log_event=log_fn,
        ... )
        >>> orchestrator.write_all(
        ...     outer_folds_record=[...],
        ...     learning_curve_rows=[...],
        ...     outer_metrics_rows=[...],
        ...     test_pred_rows_outer=[...],
        ...     test_pred_rows_inner=[...],
        ...     dataset=dataset,
        ...     y_all=y_all,
        ...     groups=groups,
        ...     class_names=["low", "high"],
        ...     mean_acc=82.5,
        ...     std_acc=3.2,
        ...     fold_macro_f1s=[80.1, 82.3],
        ...     fold_min_per_class_f1s=[75.0, 78.0],
        ...     mean_plur_corr=85.0,
        ...     std_plur_corr=2.5,
        ...     mean_kappa=0.65,
        ...     std_kappa=0.05,
        ... )
    """
    
    def __init__(self, run_dir: Path, cfg: Dict, log_event=None):
        """
        Initialize ArtifactWriterOrchestrator.
        
        Args:
            run_dir: Directory to write artifacts to
            cfg: Configuration dictionary
            log_event: Optional logging function
        """
        self.run_dir = run_dir
        self.cfg = cfg
        self.log_event = log_event if log_event else lambda *args, **kwargs: None
        self.decision_layer_stats = None  # Will be populated if decision layer runs
        
        # Parse output toggles from config
        try:
            outputs_cfg = cfg.get("outputs", {}) if isinstance(cfg, dict) else {}
            self.write_curves = bool(outputs_cfg.get("write_learning_curves_csv", True))
            self.write_outer = bool(outputs_cfg.get("write_outer_eval_csv", True))
            self.write_preds = bool(outputs_cfg.get("write_test_predictions_csv", True))
            self.write_splits = bool(outputs_cfg.get("write_splits_indices_json", True))
        except Exception:
            # Default to writing everything
            self.write_curves = True
            self.write_outer = True
            self.write_preds = True
            self.write_splits = True
    
    def write_all(
        self,
        outer_folds_record: List[Dict],
        learning_curve_rows: List[Dict],
        outer_metrics_rows: List[Dict],
        test_pred_rows_outer: List[Dict],
        test_pred_rows_inner: List[Dict],
        dataset,
        y_all: np.ndarray,
        groups: np.ndarray,
        class_names: List[str],
        mean_acc: float,
        std_acc: float,
        fold_macro_f1s: List[float],
        fold_min_per_class_f1s: List[float],
        mean_plur_corr: float,
        std_plur_corr: float,
        mean_kappa: float,
        std_kappa: float,
    ):
        """
        Write all artifacts for this training run.
        
        Args:
            outer_folds_record: List of fold records for splits_indices.json
            learning_curve_rows: Learning curve data rows
            outer_metrics_rows: Per-fold outer metrics rows
            test_pred_rows_outer: Outer test prediction rows
            test_pred_rows_inner: Inner validation prediction rows
            dataset: Dataset object (for metadata)
            y_all: All labels
            groups: Subject group assignments
            class_names: List of class names
            mean_acc: Mean outer accuracy
            std_acc: Std deviation of outer accuracy
            fold_macro_f1s: Per-fold macro F1 scores
            fold_min_per_class_f1s: Per-fold min per-class F1 scores
            mean_plur_corr: Mean plurality correctness
            std_plur_corr: Std deviation of plurality correctness
            mean_kappa: Mean Cohen's kappa
            std_kappa: Std deviation of Cohen's kappa
        """
        # Write splits indices JSON
        self.write_splits_indices(
            outer_folds_record=outer_folds_record,
            dataset=dataset,
            y_all=y_all,
            groups=groups,
            class_names=class_names,
        )
        
        # Write learning curves CSV
        self.write_learning_curves(learning_curve_rows)
        
        # Write outer evaluation metrics CSV
        self.write_outer_eval_metrics(
            outer_metrics_rows=outer_metrics_rows,
            mean_acc=mean_acc,
            std_acc=std_acc,
            fold_macro_f1s=fold_macro_f1s,
            fold_min_per_class_f1s=fold_min_per_class_f1s,
            mean_plur_corr=mean_plur_corr,
            std_plur_corr=std_plur_corr,
            mean_kappa=mean_kappa,
            std_kappa=std_kappa,
        )
        
        # Write test predictions CSVs
        self.write_test_predictions(
            test_pred_rows_outer,
            test_pred_rows_inner,
            class_names=class_names,
        )
    
    def write_splits_indices(
        self,
        outer_folds_record: List[Dict],
        dataset,
        y_all: np.ndarray,
        groups: np.ndarray,
        class_names: List[str],
    ):
        """Write splits_indices.json with CV fold information."""
        if not self.write_splits:
            return
        
        try:
            ds_dir = self.cfg.get("materialized_dir") or self.cfg.get("dataset_dir")
            n_samples = int(len(dataset))
            
            # Class counts map for readability
            try:
                labels_int = np.asarray(y_all).astype(int)
                binc = np.bincount(labels_int, minlength=len(class_names)).tolist()
                class_counts = {str(class_names[i]): int(binc[i]) for i in range(len(class_names))}
            except Exception:
                class_counts = {}
            
            # Simple manifest hash for traceability
            manifest_str = json.dumps({
                "dataset_dir": ds_dir,
                "n_samples": n_samples,
                "groups_unique": [int(s) for s in np.unique(groups).tolist()],
                "labels_unique": [int(s) for s in np.unique(y_all).tolist()],
            }, sort_keys=True)
            manifest_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
            
            splits_payload = {
                "n_samples": n_samples,
                "dataset_dir": ds_dir,
                "class_names": list(class_names),
                "class_counts": class_counts,
                "manifest_hash": manifest_hash,
                "outer_folds": outer_folds_record,
            }
            
            (self.run_dir / "splits_indices.json").write_text(json.dumps(splits_payload, indent=2))
            self.log_event("cv_split_exported", "Exported CV split indices to splits_indices.json")
        except Exception as e:
            self.log_event("cv_split_export_failed", f"Failed to export splits: {e}")
    
    def write_learning_curves(self, learning_curve_rows: List[Dict]):
        """Write learning curves CSV."""
        if not self.write_curves or not learning_curve_rows:
            return
        
        try:
            writer = LearningCurvesWriter(self.run_dir)
            writer.write(learning_curve_rows)
            self.log_event("learning_curves_written", f"Wrote {len(learning_curve_rows)} learning curve rows")
        except Exception as e:
            self.log_event("learning_curves_write_failed", f"Failed to write learning curves: {e}")
    
    def write_outer_eval_metrics(
        self,
        outer_metrics_rows: List[Dict],
        mean_acc: float,
        std_acc: float,
        fold_macro_f1s: List[float],
        fold_min_per_class_f1s: List[float],
        mean_plur_corr: float,
        std_plur_corr: float,
        mean_kappa: float,
        std_kappa: float,
    ):
        """Write outer evaluation metrics CSV with aggregate row."""
        if not self.write_outer or not outer_metrics_rows:
            return
        
        try:
            # Build aggregate row
            agg_row = {
                "outer_fold": "OVERALL",
                "test_subjects": "-",
                "n_test_trials": sum(int(r["n_test_trials"]) for r in outer_metrics_rows),
                "acc": float(mean_acc),
                "acc_std": float(std_acc),
                "macro_f1": float(np.mean(fold_macro_f1s)) if fold_macro_f1s else 0.0,
                "macro_f1_std": float(np.std(fold_macro_f1s)) if fold_macro_f1s else 0.0,
                "min_per_class_f1": float(np.mean(fold_min_per_class_f1s)) if fold_min_per_class_f1s else 0.0,
                "min_per_class_f1_std": float(np.std(fold_min_per_class_f1s)) if fold_min_per_class_f1s else 0.0,
                "plur_corr": float(mean_plur_corr),
                "plur_corr_std": float(std_plur_corr),
                "cohen_kappa": float(mean_kappa),
                "cohen_kappa_std": float(std_kappa),
                "per_class_f1": "",
            }
            
            writer = OuterEvalMetricsWriter(self.run_dir)
            writer.write(outer_metrics_rows, agg_row)
            self.log_event("outer_metrics_written", f"Wrote {len(outer_metrics_rows)} outer metrics rows")
        except Exception as e:
            self.log_event("outer_metrics_write_failed", f"Failed to write outer metrics: {e}")
    
    def write_test_predictions(
        self,
        test_pred_rows_outer: List[Dict],
        test_pred_rows_inner: List[Dict],
        class_names: List[str],
    ):
        """
        Write test predictions CSVs (inner and outer).
        
        If decision layer is enabled in config, also tune θ on inner data,
        apply to outer, and write thresholded predictions + thresholds.json.
        
        Args:
            test_pred_rows_outer: Outer test prediction rows (baseline)
            test_pred_rows_inner: Inner validation prediction rows
            class_names: List of class names (defines ordinal order)
        """
        if not self.write_preds:
            return
        
        # Write baseline outer predictions
        try:
            if test_pred_rows_outer:
                writer = TestPredictionsWriter(self.run_dir, mode="outer")
                writer.write(test_pred_rows_outer)
                self.log_event("outer_predictions_written", f"Wrote {len(test_pred_rows_outer)} outer prediction rows")
        except Exception as e:
            self.log_event("outer_predictions_write_failed", f"Failed to write outer predictions: {e}")
        
        # Write baseline inner predictions
        try:
            if test_pred_rows_inner:
                writer = TestPredictionsWriter(self.run_dir, mode="inner")
                writer.write(test_pred_rows_inner)
                self.log_event("inner_predictions_written", f"Wrote {len(test_pred_rows_inner)} inner prediction rows")
        except Exception as e:
            self.log_event("inner_predictions_write_failed", f"Failed to write inner predictions: {e}")
        
        # Decision layer integration (optional post-hoc refinement)
        # Tunes θ per outer fold on inner data, applies frozen to outer test
        # Constitutional: Section IV (leak-free), Section V (auditable)
        try:
            dl_cfg = self.cfg.get("decision_layer", {})
            if not dl_cfg.get("enable", False):
                return  # Decision layer disabled
            
            if not test_pred_rows_inner or not test_pred_rows_outer:
                self.log_event("decision_layer_skipped", "No inner/outer predictions available")
                return
            
            self.log_event("decision_layer_start", "Starting decision layer tuning and application")
            
            # Import decision layer module
            from code.posthoc.decision_layer import tune_and_apply_decision_layer
            
            # Load ObjectiveComputer to ensure metric alignment
            from code.training.metrics import ObjectiveComputer
            objective_computer = ObjectiveComputer(self.cfg)
            
            # Tune and apply decision layer (capture enriched stats for summary writer)
            # Pass only the decision_layer sub-config (not the full config)
            self.decision_layer_stats = tune_and_apply_decision_layer(
                inner_rows=test_pred_rows_inner,
                outer_rows=test_pred_rows_outer,
                class_names=class_names,
                cfg=dl_cfg,  # Pass decision_layer sub-config only
                objective_computer=objective_computer,
                run_dir=self.run_dir,
                log_event=self.log_event,
            )
            
            if self.decision_layer_stats:
                self.log_event("decision_layer_complete", "Decision layer artifacts and enriched stats computed successfully")
            else:
                self.log_event("decision_layer_warning", "Decision layer completed but enriched stats unavailable")
            
        except Exception as e:
            # Log error but don't block baseline artifacts
            self.log_event("decision_layer_failed", f"Decision layer error (baseline artifacts preserved): {e}")
            import traceback
            self.log_event("decision_layer_traceback", traceback.format_exc())

