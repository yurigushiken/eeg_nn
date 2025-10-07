"""
Plot title and label builders for training artifacts.

This module handles all plot title generation logic:
- Objective-specific metric labels
- Fold titles (simple and enhanced)
- Overall titles (simple and enhanced)
- Per-class F1 info formatting
- Inner vs outer metric comparison labels

Constitutional compliance: Section V (Audit-Ready Artifacts)

Classes:
    PlotTitleBuilder: Builds titles and labels for plots
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np

from ..training.metrics import ObjectiveComputer


class PlotTitleBuilder:
    """
    Builds titles and labels for training plots.
    
    Handles objective-specific labeling, ensuring plots clearly show:
    - Which objective metric was optimized
    - Inner validation performance (model selection)
    - Outer test performance (generalization)
    - Per-class breakdown for diagnostic purposes
    
    Example usage:
        >>> cfg = {"optuna_objective": "inner_mean_macro_f1"}
        >>> obj_computer = ObjectiveComputer(cfg)
        >>> builder = PlotTitleBuilder(cfg, obj_computer, "trial_20241006_123456")
        >>> 
        >>> inner_metrics = {
        ...     "acc": 75.0,
        ...     "macro_f1": 72.0,
        ...     "min_per_class_f1": 65.0,
        ...     "plur_corr": 80.0,
        ... }
        >>> label = builder.build_objective_label(inner_metrics)
        >>> # Returns: "inner-mean macro-F1=72.00"
        >>> 
        >>> title = builder.build_fold_title_simple(
        ...     fold=0,
        ...     test_subjects=[2, 5, 8],
        ...     inner_metrics=inner_metrics,
        ...     outer_acc=68.0,
        ... )
        >>> # Returns: "trial_20241006_123456 · Fold 1 (Subjects: [2, 5, 8]) · inner-mean macro-F1=72.00 · ensemble acc=68.00"
    """
    
    def __init__(self, cfg: Dict, objective_computer: ObjectiveComputer, trial_dir_name: str):
        """
        Initialize PlotTitleBuilder.
        
        Args:
            cfg: Configuration dictionary containing optuna_objective and related params
            objective_computer: ObjectiveComputer instance for metric computation
            trial_dir_name: Trial directory name for plot titles
        """
        self.cfg = cfg
        self.obj_computer = objective_computer
        self.trial_dir_name = trial_dir_name
        self.objective = cfg.get("optuna_objective", "inner_mean_macro_f1")
    
    def build_objective_label(self, inner_metrics: Dict[str, float]) -> str:
        """
        Build objective-specific metric label showing inner validation performance.
        
        Args:
            inner_metrics: Dictionary with keys: acc, macro_f1, min_per_class_f1, plur_corr
        
        Returns:
            String label (e.g., "inner-mean macro-F1=72.30")
        """
        obj = self.objective
        
        if obj == "inner_mean_min_per_class_f1":
            val = inner_metrics["min_per_class_f1"]
            return f"inner-mean min-per-class-F1={val:.2f}"
        
        elif obj == "inner_mean_acc":
            val = inner_metrics["acc"]
            return f"inner-mean acc={val:.2f}"
        
        elif obj == "composite_min_f1_plur_corr":
            min_f1 = inner_metrics["min_per_class_f1"]
            plur_corr = inner_metrics["plur_corr"]
            params = self.obj_computer.get_params()
            
            if params["mode"] == "threshold":
                threshold = params["threshold"]
                if min_f1 < threshold:
                    composite = min_f1 * 0.1
                    return f"composite={composite:.2f} (minF1={min_f1:.2f} < threshold={threshold:.1f}, plurCorr={plur_corr:.2f}, gradient)"
                else:
                    composite = plur_corr
                    return f"composite={composite:.2f} (threshold met, plurCorr={plur_corr:.2f})"
            else:  # weighted
                weight = params["weight"]
                composite = weight * min_f1 + (1.0 - weight) * plur_corr
                return f"composite={composite:.2f} (weight={weight:.2f}, minF1={min_f1:.2f}, plurCorr={plur_corr:.2f})"
        
        elif obj == "inner_mean_plur_corr":
            val = inner_metrics["plur_corr"]
            return f"inner-mean plur-corr={val:.2f}"
        
        else:  # inner_mean_macro_f1 (default)
            val = inner_metrics["macro_f1"]
            return f"inner-mean macro-F1={val:.2f}"
    
    def build_outer_metric_label(self, outer_metrics: Dict[str, float], per_class_f1: List[float] | None = None) -> str:
        """
        Build objective-aligned outer metric label showing test performance.
        
        Args:
            outer_metrics: Dictionary with keys: acc, macro_f1, plur_corr
            per_class_f1: Optional list of per-class F1 scores (for min computation)
        
        Returns:
            String label (e.g., "outer macro-F1=65.30")
        """
        obj = self.objective
        
        if obj == "inner_mean_min_per_class_f1":
            if per_class_f1 is not None and len(per_class_f1) > 0:
                min_f1 = float(np.min(per_class_f1)) * 100
                return f"outer min-per-class-F1={min_f1:.2f}"
            else:
                return "outer min-per-class-F1=N/A"
        
        elif obj == "inner_mean_acc":
            return f"outer acc={outer_metrics['acc']:.2f}"
        
        elif obj == "composite_min_f1_plur_corr":
            if per_class_f1 is not None and len(per_class_f1) > 0:
                min_f1 = float(np.min(per_class_f1)) * 100
            else:
                min_f1 = 0.0
            plur_corr = outer_metrics["plur_corr"]
            params = self.obj_computer.get_params()
            
            if params["mode"] == "threshold":
                threshold = params["threshold"]
                if min_f1 < threshold:
                    composite = min_f1 * 0.1
                    return f"outer composite={composite:.2f} (minF1={min_f1:.2f} < threshold, plurCorr={plur_corr:.2f}, gradient)"
                else:
                    composite = plur_corr
                    return f"outer composite={composite:.2f} (minF1={min_f1:.2f} met, plurCorr={plur_corr:.2f})"
            else:  # weighted
                weight = params["weight"]
                composite = weight * min_f1 + (1.0 - weight) * plur_corr
                return f"outer composite={composite:.2f} (weight={weight:.2f}, minF1={min_f1:.2f}, plurCorr={plur_corr:.2f})"
        
        elif obj == "inner_mean_plur_corr":
            return f"outer plur-corr={outer_metrics['plur_corr']:.2f}"
        
        else:  # inner_mean_macro_f1 (default)
            return f"outer macro-F1={outer_metrics['macro_f1']:.2f}"
    
    def build_fold_title_simple(
        self,
        fold: int,
        test_subjects: List[int],
        inner_metrics: Dict[str, float],
        outer_acc: float,
    ) -> str:
        """
        Build simple fold title for basic plots.
        
        Args:
            fold: Fold index (0-based)
            test_subjects: List of test subject IDs
            inner_metrics: Inner validation metrics
            outer_acc: Outer test accuracy
        
        Returns:
            Title string
        """
        obj_label = self.build_objective_label(inner_metrics)
        return (
            f"{self.trial_dir_name} · Fold {fold+1} (Subjects: {test_subjects}) · "
            f"{obj_label} · ensemble acc={outer_acc:.2f}"
        )
    
    def build_fold_title_enhanced(
        self,
        fold: int,
        test_subjects: List[int],
        inner_metrics: Dict[str, float],
        outer_metrics: Dict[str, float],
        per_class_f1: List[float] | None = None,
    ) -> str:
        """
        Build enhanced fold title with inner/outer comparison.
        
        Args:
            fold: Fold index (0-based)
            test_subjects: List of test subject IDs
            inner_metrics: Inner validation metrics
            outer_metrics: Outer test metrics
            per_class_f1: Optional per-class F1 scores
        
        Returns:
            Multi-line title string
        """
        obj_label = self.build_objective_label(inner_metrics)
        outer_label = self.build_outer_metric_label(outer_metrics, per_class_f1)
        return (
            f"{self.trial_dir_name} · Fold {fold+1} (Subjects: {test_subjects})\n"
            f"inner {obj_label} · {outer_label} · ensemble acc={outer_metrics['acc']:.2f}"
        )
    
    def build_overall_title_simple(
        self,
        inner_metrics: Dict[str, float],
        outer_mean_acc: float,
    ) -> str:
        """
        Build simple overall title.
        
        Args:
            inner_metrics: Aggregated inner metrics across all folds
            outer_mean_acc: Mean outer test accuracy across folds
        
        Returns:
            Title string
        """
        obj_label = self.build_objective_label(inner_metrics)
        return (
            f"{self.trial_dir_name} · Overall · {obj_label} "
            f"· ensemble mean_acc={outer_mean_acc:.2f}"
        )
    
    def build_overall_title_enhanced(
        self,
        inner_metrics: Dict[str, float],
        outer_metrics: Dict[str, float],
        outer_mean_acc: float,
    ) -> str:
        """
        Build enhanced overall title with inner/outer comparison.
        
        Args:
            inner_metrics: Aggregated inner metrics across all folds
            outer_metrics: Aggregated outer metrics across all folds
            outer_mean_acc: Mean outer test accuracy across folds
        
        Returns:
            Multi-line title string
        """
        obj_label = self.build_objective_label(inner_metrics)
        outer_label = self.build_outer_metric_label(outer_metrics)
        return (
            f"{self.trial_dir_name} · Overall\n"
            f"inner {obj_label} · {outer_label} · ensemble mean_acc={outer_mean_acc:.2f}"
        )
    
    def build_per_class_info(
        self,
        per_class_f1: List[float] | None,
        class_names: List[str],
    ) -> List[str]:
        """
        Build per-class F1 info lines for plot annotations.
        
        Args:
            per_class_f1: List of per-class F1 scores (as fractions 0-1)
            class_names: List of class names
        
        Returns:
            List of formatted strings (empty if no data)
        """
        if per_class_f1 is None or len(per_class_f1) == 0:
            return []
        
        lines = ["Per-class F1 (outer):"]
        for i, f1_val in enumerate(per_class_f1):
            class_label = str(class_names[i]) if i < len(class_names) else f"Class {i}"
            lines.append(f"  {class_label}: {f1_val*100:.2f}%")
        
        return lines

