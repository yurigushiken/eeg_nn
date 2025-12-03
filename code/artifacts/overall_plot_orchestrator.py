"""
Overall plot orchestrator for training runs.

This module handles generating overall (across all folds) plots:
- Overall confusion matrix (simple)
- Overall confusion matrix (enhanced with metrics)

Constitutional compliance:
- Section V (Audit-Ready Artifacts): All plots retained

Classes:
    OverallPlotOrchestrator: Coordinates generation of overall plots
"""

from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

from .plot_builders import PlotTitleBuilder

# Import plotting functions
try:
    from utils import plots as _plots
    plot_confusion = _plots.plot_confusion
except Exception:
    def plot_confusion(*args, **kwargs):
        pass  # no-op if not available


class OverallPlotOrchestrator:
    """
    Orchestrates generation of overall (cross-fold) plots.
    
    Responsibilities:
    - Generate overall confusion matrix (simple version)
    - Generate overall confusion matrix (enhanced with inner vs outer metrics)
    - Use PlotTitleBuilder for consistent title formatting
    
    Example usage:
        >>> orchestrator = OverallPlotOrchestrator(
        ...     run_dir=Path("results/run_123"),
        ...     plot_title_builder=plot_title_builder,
        ... )
        >>> orchestrator.generate_overall_plots(
        ...     overall_y_true=[0, 1, 0, 1],
        ...     overall_y_pred=[0, 1, 1, 1],
        ...     class_names=["low", "high"],
        ...     inner_accs=[80.0, 82.0],
        ...     inner_macro_f1s=[78.0, 80.0],
        ...     inner_min_per_class_f1s=[75.0, 77.0],
        ...     inner_plur_corrs=[85.0, 87.0],
        ...     mean_acc=81.0,
        ...     macro_f1=79.0,
        ...     mean_plur_corr=86.0,
        ... )
    """
    
    def __init__(self, run_dir: Path, plot_title_builder: PlotTitleBuilder):
        """
        Initialize OverallPlotOrchestrator.
        
        Args:
            run_dir: Directory to save plots to
            plot_title_builder: PlotTitleBuilder for generating titles
        """
        self.run_dir = run_dir
        self.plot_title_builder = plot_title_builder
    
    def generate_overall_plots(
        self,
        overall_y_true: List[int],
        overall_y_pred: List[int],
        class_names: List[str],
        inner_accs: List[float],
        inner_macro_f1s: List[float],
        inner_min_per_class_f1s: List[float],
        inner_plur_corrs: List[float],
        mean_acc: float,
        macro_f1: float,
        mean_plur_corr: float,
    ):
        """
        Generate overall plots (simple and enhanced).
        
        Args:
            overall_y_true: True labels across all outer test sets
            overall_y_pred: Predicted labels across all outer test sets
            class_names: List of class names
            inner_accs: Per-fold inner accuracies
            inner_macro_f1s: Per-fold inner macro F1 scores
            inner_min_per_class_f1s: Per-fold inner min per-class F1 scores
            inner_plur_corrs: Per-fold inner plurality correctness scores
            mean_acc: Mean outer accuracy
            macro_f1: Overall macro F1 (across all predictions)
            mean_plur_corr: Mean outer plurality correctness
        """
        if not overall_y_true:
            return  # No data to plot
        
        # Aggregate inner metrics
        overall_inner_metrics = {
            "acc": float(np.mean(inner_accs)) if inner_accs else 0.0,
            "macro_f1": float(np.mean(inner_macro_f1s)) if inner_macro_f1s else 0.0,
            "min_per_class_f1": float(np.mean(inner_min_per_class_f1s)) if inner_min_per_class_f1s else 0.0,
            "plur_corr": float(np.mean(inner_plur_corrs)) if inner_plur_corrs else 0.0,
        }
        
        # Generate simple overall plot
        self._generate_simple_plot(
            overall_y_true=overall_y_true,
            overall_y_pred=overall_y_pred,
            class_names=class_names,
            inner_metrics=overall_inner_metrics,
            outer_mean_acc=mean_acc,
        )
        
        # Generate enhanced overall plot
        self._generate_enhanced_plot(
            overall_y_true=overall_y_true,
            overall_y_pred=overall_y_pred,
            class_names=class_names,
            inner_metrics=overall_inner_metrics,
            outer_mean_acc=mean_acc,
            macro_f1=macro_f1,
            mean_plur_corr=mean_plur_corr,
        )
    
    def _generate_simple_plot(
        self,
        overall_y_true: List[int],
        overall_y_pred: List[int],
        class_names: List[str],
        inner_metrics: Dict,
        outer_mean_acc: float,
    ):
        """Generate simple overall confusion matrix."""
        plots_dir = self.run_dir / "plots_outer"
        plots_dir.mkdir(parents=True, exist_ok=True)
        run_prefix = self.run_dir.name if self.run_dir else "run"
        
        # Build title using PlotTitleBuilder
        overall_title = self.plot_title_builder.build_overall_title_simple(
            inner_metrics=inner_metrics,
            outer_mean_acc=outer_mean_acc,
        )
        
        # Generate plot
        plot_confusion(
            overall_y_true,
            overall_y_pred,
            class_names,
            plots_dir / f"{run_prefix}_overall_confusion.png",
            title=overall_title,
        )
    
    def _generate_enhanced_plot(
        self,
        overall_y_true: List[int],
        overall_y_pred: List[int],
        class_names: List[str],
        inner_metrics: Dict,
        outer_mean_acc: float,
        macro_f1: float,
        mean_plur_corr: float,
    ):
        """Generate enhanced overall confusion matrix with inner vs outer metrics."""
        plots_enhanced_dir = self.run_dir / "plots_outer_enhanced"
        plots_enhanced_dir.mkdir(parents=True, exist_ok=True)
        run_prefix = self.run_dir.name if self.run_dir else "run"
        
        # Compute per-class F1 for enhanced info
        try:
            overall_per_class_f1 = f1_score(overall_y_true, overall_y_pred, average=None) if overall_y_true else None
        except Exception:
            overall_per_class_f1 = None
        
        # Build outer metrics dict
        overall_outer_metrics = {
            "acc": outer_mean_acc,
            "macro_f1": macro_f1,
            "plur_corr": mean_plur_corr,
        }
        
        # Build enhanced title (pass per-class F1 so minF1/composite are computed correctly)
        overall_title_enhanced = self.plot_title_builder.build_overall_title_enhanced(
            inner_metrics=inner_metrics,
            outer_metrics=overall_outer_metrics,
            outer_mean_acc=outer_mean_acc,
            per_class_f1=overall_per_class_f1.tolist() if overall_per_class_f1 is not None else None,
        )
        
        # Build per-class info lines
        overall_per_class_info_lines = self.plot_title_builder.build_per_class_info(
            overall_per_class_f1, class_names
        )
        
        # Generate enhanced plot
        plot_confusion(
            overall_y_true,
            overall_y_pred,
            class_names,
            plots_enhanced_dir / f"{run_prefix}_overall_confusion.png",
            title=overall_title_enhanced,
            hyper_lines=overall_per_class_info_lines if overall_per_class_info_lines else None,
        )

