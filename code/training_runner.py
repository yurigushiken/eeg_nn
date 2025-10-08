from __future__ import annotations
from typing import Dict, Callable, List, Any

import copy
import json
import hashlib
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score, confusion_matrix
import optuna

# Import extracted modules for refactored architecture
from .training.metrics import ObjectiveComputer
from .training.checkpointing import CheckpointManager
from .training.inner_loop import InnerTrainer
from .training.evaluation import OuterEvaluator
from .training.outer_loop import OuterFoldOrchestrator
from .training.setup_orchestrator import SetupOrchestrator
from .artifacts.csv_writers import (
    LearningCurvesWriter,
    OuterEvalMetricsWriter,
    TestPredictionsWriter,
)
from .artifacts.plot_builders import PlotTitleBuilder
from .artifacts.artifact_writer import ArtifactWriterOrchestrator
from .artifacts.overall_plot_orchestrator import OverallPlotOrchestrator

try:
    from utils import plots as _plots
    plot_confusion = _plots.plot_confusion
    plot_curves = _plots.plot_curves
except Exception:  # pragma: no cover - fallback to no-op to allow tests to run until implemented
    def plot_confusion(*args, **kwargs):
        pass  # no-op until utils.plots is implemented

    def plot_curves(*args, **kwargs):
        pass  # no-op until utils.plots is implemented

try:
    from utils import channel_viz as _channel_viz
    save_channel_topomap_for_run = _channel_viz.save_channel_topomap_for_run
except Exception:  # pragma: no cover - fallback to no-op if not available
    def save_channel_topomap_for_run(*args, **kwargs):
        pass  # no-op until utils.channel_viz is implemented


# Import compute_plurality_correctness from metrics module to avoid circular imports
from code.training.metrics import compute_plurality_correctness

import random
import re

"""
Training/evaluation orchestration with nested, subject-aware cross-validation.

Outer CV:
  - GroupKFold when cfg['n_folds'] is set (subject-aware K-way)
  - LOSO when 'n_folds' is absent or null

Inner CV:
  - Strict GroupKFold with cfg['inner_n_folds'] (≥2), per outer split (subject-aware)
  - Early stopping, checkpointing, and pruning all use the objective-aligned metric.

Evaluation:
  - For each outer split, predictions on the held-out subjects are obtained via
    an ensemble over inner K models (mean of softmax), improving stability.
  - Alternatively, if cfg['outer_eval_mode'] == 'refit', we refit a single model
    on the full outer-train set (optional small val) and evaluate once.
  - Metrics returned include fold accuracies, overall macro/weighted F1, and
    inner means across folds.

Data handling and leakage guards:
  - Augmentation is applied to training only; validation/test have no transforms.
  - Class weights are computed from the inner-train labels only to balance loss.
  - Inner folds are split strictly by subject groups to avoid any leakage.
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_worker(worker_id):
    # Deterministic DataLoader workers (ensures per-worker RNG reproducibility)
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def format_subject_id(subject: int) -> str:
    """Return zero-padded subject identifier in snake_case (e.g., subject_07)."""
    try:
        subject_int = int(subject)
    except Exception:
        subject_int = subject
    return f"subject_{subject_int:02d}"


def format_fold_id(fold: int) -> str:
    """Return zero-padded fold identifier in snake_case (e.g., fold_03)."""
    try:
        fold_int = int(fold)
    except Exception:
        fold_int = fold
    return f"fold_{fold_int:02d}"


def validate_artifact_name(name: str) -> None:
    """Validate artifact path uses snake_case segments and allowed extensions.

    Accept lowercase letters, numbers, underscores, forward slashes, and dots in extensions.
    Reject camelCase or hyphenated patterns, and disallow unknown extensions.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("artifact name must be a non-empty string")
    # Allowed chars per segment: [a-z0-9_]+, segments separated by '/'
    segments = name.split("/")
    seg_pattern = re.compile(r"^[a-z0-9_]+(\.[a-z0-9]+)?$")
    for seg in segments:
        if not seg_pattern.match(seg):
            raise ValueError(f"invalid artifact segment: {seg}")
    # Validate extension if present (allow csv, json, jsonl, png)
    allowed_ext = {".csv", ".json", ".jsonl", ".png"}
    from pathlib import Path as _P
    ext = _P(name).suffix
    if ext and ext not in allowed_ext:
        raise ValueError(f"invalid artifact extension: {ext}")


class TrainingRunner:
    def __init__(self, cfg: Dict, label_fn: Callable):
        self.cfg = cfg
        self.label_fn = label_fn
        self.run_dir = self._ensure_run_dir(cfg)
        self.stage = str(cfg.get("stage", "") or "")
        self.stage_context = self._resolve_stage_context()
        
        # Initialize objective computer for metric computation (refactored)
        # Only initialize if optuna_objective is specified (for nested CV runs)
        # Some simple scripts may not have this field
        if "optuna_objective" in cfg:
            self.objective_computer = ObjectiveComputer(cfg)
        else:
            self.objective_computer = None
    
    def _get_composite_params(self):
        """
        Get composite objective parameters (DEPRECATED - use objective_computer.get_params()).
        
        Maintained for backward compatibility. Delegates to ObjectiveComputer.
        
        Returns:
            dict: {"mode": "threshold", "threshold": float} or {"mode": "weighted", "weight": float}
        """
        if self.objective_computer is None:
            # Fallback to old implementation if no objective_computer
            # (for backward compatibility with scripts that don't use optuna_objective)
            has_threshold = "min_f1_threshold" in self.cfg
            has_weight = "composite_min_f1_weight" in self.cfg
            
            if has_threshold and has_weight:
                raise ValueError(
                    "Ambiguous config: both min_f1_threshold and composite_min_f1_weight specified. "
                    "Use only ONE: min_f1_threshold (threshold approach) OR composite_min_f1_weight (weighted approach)"
                )
            
            if not has_threshold and not has_weight:
                raise ValueError(
                    "composite_min_f1_plur_corr objective requires either:\n"
                    "  - min_f1_threshold: 35.0  (threshold approach - recommended)\n"
                    "  - composite_min_f1_weight: 0.50  (weighted approach - backward compatible)"
                )
            
            if has_threshold:
                threshold = float(self.cfg["min_f1_threshold"])
                if not (0.0 <= threshold <= 100.0):
                    raise ValueError(f"min_f1_threshold must be in range [0.0, 100.0], got {threshold}")
                return {"mode": "threshold", "threshold": threshold}
            else:
                weight = float(self.cfg["composite_min_f1_weight"])
                if not (0.0 <= weight <= 1.0):
                    raise ValueError(f"composite_min_f1_weight must be in range [0.0, 1.0], got {weight}")
                return {"mode": "weighted", "weight": weight}
        else:
            # Delegate to ObjectiveComputer (refactored path)
            return self.objective_computer.get_params()
    
    def _compute_objective_metric(self, val_acc, val_macro_f1, val_min_per_class_f1, val_plur_corr):
        """
        Compute the per-epoch metric aligned with the configured Optuna objective.
        
        DEPRECATED: Use objective_computer.compute() directly.
        Maintained for backward compatibility. Delegates to ObjectiveComputer.
        
        Args:
            val_acc: Validation accuracy (0-100)
            val_macro_f1: Validation macro-F1 (0-100)
            val_min_per_class_f1: Validation min per-class F1 (0-100)
            val_plur_corr: Validation plurality correctness (0-100)
        
        Returns:
            float: The metric value aligned with the optimization objective (to be maximized)
        """
        if self.objective_computer is not None:
            # Delegate to ObjectiveComputer (refactored path)
            return self.objective_computer.compute(
                val_acc, val_macro_f1, val_min_per_class_f1, val_plur_corr
            )
        else:
            # Fallback to old implementation (for backward compatibility)
            objective = self.cfg.get("optuna_objective", "inner_mean_macro_f1")
            
            if objective == "inner_mean_min_per_class_f1":
                return val_min_per_class_f1
            elif objective == "inner_mean_acc":
                return val_acc
            elif objective == "inner_mean_plur_corr":
                return val_plur_corr
            elif objective == "composite_min_f1_plur_corr":
                params = self._get_composite_params()
                if params["mode"] == "threshold":
                    threshold = params["threshold"]
                    if val_min_per_class_f1 < threshold:
                        return val_min_per_class_f1 * 0.1
                    else:
                        return val_plur_corr
                else:  # weighted mode
                    weight = params["weight"]
                    return weight * val_min_per_class_f1 + (1.0 - weight) * val_plur_corr
            elif objective == "inner_mean_macro_f1":
                return val_macro_f1
            else:
                raise ValueError(
                    f"Invalid objective in _compute_objective_metric: '{objective}'. "
                    f"Must be one of: inner_mean_macro_f1, inner_mean_min_per_class_f1, "
                    f"inner_mean_plur_corr, inner_mean_acc, composite_min_f1_plur_corr"
                )

    def _ensure_run_dir(self, cfg: Dict):
        from pathlib import Path
        run_dir = cfg.get("run_dir")
        if run_dir:
            p = Path(run_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        return None

    def _resolve_stage_context(self) -> Dict[str, Any]:
        stage = self.stage
        if stage in {"stage_2", "stage_3", "final_loso"}:
            base_dir = self.cfg.get("previous_stage_dir") or self.cfg.get("base_run")
            return self.validate_stage_handoff(stage=stage, previous_stage_dir=base_dir)
        return {}

    @staticmethod
    def validate_stage_handoff(stage: str, previous_stage_dir: str | None) -> Dict[str, Any]:
        from pathlib import Path

        stage = str(stage)
        if stage not in {"stage_2", "stage_3", "final_loso"}:
            raise ValueError(f"Unsupported stage '{stage}' for handoff validation")
        if not previous_stage_dir:
            raise ValueError(f"Stage '{stage}' requires previous_stage_dir/base_run to be provided")

        prev_path = Path(previous_stage_dir)
        if not prev_path.exists():
            raise ValueError(f"Previous stage directory does not exist: {previous_stage_dir}")

        resolved_cfg = prev_path / "resolved_config.yaml"
        evidence_dir = prev_path / "evidence"

        missing: List[str] = []
        if not resolved_cfg.exists():
            missing.append("resolved_config.yaml")
        if not evidence_dir.exists():
            missing.append("evidence/")
        else:
            outer_metrics = evidence_dir / "outer_eval_metrics.csv"
            if not outer_metrics.exists():
                missing.append("evidence/outer_eval_metrics.csv")

        if missing:
            raise ValueError(
                "Previous stage handoff is incomplete; missing: " + ", ".join(missing)
            )

        return {
            "stage": stage,
            "champion_config": resolved_cfg,
            "evidence_dir": evidence_dir,
        }


    def _make_loaders(
        self,
        dataset,
        y_all,
        groups,
        tr_idx,
        va_idx,
        aug_transform=None,
        inner_tr_idx_override=None,
        inner_va_idx_override=None,
    ):
        # Require explicit inner indices for scientific clarity and determinism.
        # All callers (inner K-fold and refit) must pass overrides.
        if inner_tr_idx_override is None or inner_va_idx_override is None:
            raise ValueError(
                "Explicit inner index overrides are required for _make_loaders. "
                "Provide inner_tr_idx_override and inner_va_idx_override."
            )
        inner_tr_idx = np.array(inner_tr_idx_override)
        inner_va_idx = np.array(inner_va_idx_override)

        dataset_tr = copy.copy(dataset)
        dataset_eval = copy.copy(dataset)
        # Train: apply augmentation; Eval/Test: no augmentation (prevents leakage)
        dataset_tr.set_transform(aug_transform)
        dataset_eval.set_transform(None)

        num_workers = 0  # safest on Windows; avoids multiprocessing pitfalls
        g = torch.Generator()
        if self.cfg.get("seed") is not None:
            g.manual_seed(int(self.cfg["seed"]))

        tr_ld = DataLoader(
            Subset(dataset_tr, inner_tr_idx),
            batch_size=int(self.cfg.get("batch_size", 16)),
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            generator=g,
        )
        va_ld = DataLoader(
            Subset(dataset_eval, inner_va_idx),
            batch_size=int(self.cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=num_workers,
        )
        te_ld = DataLoader(
            Subset(dataset_eval, va_idx),
            batch_size=int(self.cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=num_workers,
        )
        return tr_ld, va_ld, te_ld, inner_tr_idx

    def _validate_subject_requirements(self, dataset, groups):
        min_subjects = int(self.cfg.get("min_subjects", 0) or 0)
        unique_subjects = int(len(np.unique(groups)))
        if min_subjects > 0 and unique_subjects < min_subjects:
            raise ValueError(
                f"Dataset contains {unique_subjects} unique subjects, fewer than the required minimum of {min_subjects} subjects"
            )

        if hasattr(dataset, "excluded_subjects") and dataset.excluded_subjects:
            print(
                f"[dataset] excluded_subjects={sorted(dataset.excluded_subjects.keys())}",
                flush=True,
            )

    def run(
        self,
        dataset,
        groups,
        class_names,
        model_builder,
        aug_builder,
        input_adapter=None,
        optuna_trial=None,
        labels_override: np.ndarray | None = None,
        predefined_splits: List[dict] | None = None,
    ):
        # Labels (optionally overridden, e.g., for permutation testing)
        if labels_override is not None:
            y_all = np.asarray(labels_override)
            # Ensure dataset uses the overridden labels consistently
            dataset.y = torch.from_numpy(y_all.astype(np.int64))
        else:
            y_all = dataset.get_all_labels()
        num_cls = len(class_names)

        # Sanity checks before any split orchestration
        self._validate_subject_requirements(dataset, groups)

        # Setup and initialization using SetupOrchestrator (refactored - Stage 7d)
        setup_orchestrator = SetupOrchestrator(self.cfg, self.run_dir)
        _log_event = setup_orchestrator.setup_logging()
        setup_orchestrator.log_configuration(_log_event, num_cls)
        setup_orchestrator.generate_channel_topomap()
        outer_pairs = setup_orchestrator.compute_outer_splits(dataset, y_all, groups, predefined_splits)

        fold_accs: List[float] = []
        fold_split_info: List[dict] = []
        overall_y_true: List[int] = []
        overall_y_pred: List[int] = []
        inner_accs: List[float] = []
        inner_macro_f1s: List[float] = []
        inner_min_per_class_f1s: List[float] = []
        inner_plur_corrs: List[float] = []  # Track plurality correctness
        # Per-outer-fold macro-F1, min-per-class-F1, plurality correctness, and kappa for aggregates and CSV
        fold_macro_f1s: List[float] = []
        fold_min_per_class_f1s: List[float] = []
        fold_plur_corrs: List[float] = []  # Track plurality correctness per fold
        fold_kappas: List[float] = []

        # Prepare augmentation transform once (stateless transform expected)
        # Build a base augmentation transform once if provided; the InnerTrainer
        # will update training-time augmentation dynamically via aug_builder.
        aug_transform = aug_builder(self.cfg, dataset) if aug_builder else None
        
        # Extract trial directory name for plot titles
        trial_dir_name = self.run_dir.name if self.run_dir else ""
        
        # Initialize plot title builder (refactored - Stage 3)
        plot_title_builder = PlotTitleBuilder(self.cfg, self.objective_computer, trial_dir_name)
        
        # Initialize orchestrators (refactored - Stage 7a)
        inner_trainer = InnerTrainer(self.cfg, self.objective_computer)
        outer_evaluator = OuterEvaluator(self.cfg)
        fold_orchestrator = OuterFoldOrchestrator(
            cfg=self.cfg,
            objective_computer=self.objective_computer,
            inner_trainer=inner_trainer,
            outer_evaluator=outer_evaluator,
            plot_title_builder=plot_title_builder,
            run_dir=self.run_dir,
            log_event=_log_event,
        )

        # Global step counter for Optuna pruning (increments every epoch across folds)
        global_step = 0

        # Record all split indices for auditability
        outer_folds_record: List[dict] = []
        # Accumulate learning curves (all inner folds across all outer folds)
        learning_curve_rows: List[dict] = []
        # Accumulate per-outer-fold evaluation rows
        outer_metrics_rows: List[dict] = []
        # Accumulate per-trial out-of-fold test predictions (outer)
        test_pred_rows_outer: List[dict] = []
        # Accumulate per-trial inner validation predictions
        test_pred_rows_inner: List[dict] = []

        for fold, (tr_idx, va_idx) in enumerate(outer_pairs):
            if self.cfg.get("max_folds") is not None and fold >= int(self.cfg["max_folds"]):
                break
            
            # Get predefined inner splits for this fold (if any)
            predefined_inner = None
            if predefined_splits and fold < len(predefined_splits):
                predefined_inner = predefined_splits[fold].get("inner_splits")
            
            # Run complete outer fold using OuterFoldOrchestrator (refactored - Stage 7a)
            fold_result = fold_orchestrator.run_fold(
                fold=fold,
                tr_idx=tr_idx,
                te_idx=va_idx,
                dataset=dataset,
                y_all=y_all,
                groups=groups,
                class_names=class_names,
                model_builder=model_builder,
                aug_builder=aug_builder,
                aug_transform=aug_transform,
                input_adapter=input_adapter,
                predefined_inner_splits=predefined_inner,
                optuna_trial=optuna_trial,
                global_step_offset=global_step,
            )
            
            # Update global step
            global_step = fold_result["new_global_step"]
            
            # Collect results from this fold
            test_subjects = fold_result["test_subjects"]
            fold_split_info.append({"fold": fold + 1, "test_subjects": test_subjects})
            print(
                f"[fold {fold+1}] test_subjects={test_subjects} n_tr={len(tr_idx)} n_te={len(va_idx)}",
                flush=True,
            )

            # Accumulate results from this fold (refactored - Stage 7a)
            # Extract metrics from fold_result
            metrics = fold_result["metrics"]
            inner_metrics = fold_result["inner_metrics"]
            
            # Accumulate for overall statistics
            fold_accs.append(metrics["acc"])
            fold_macro_f1s.append(metrics["macro_f1"])
            fold_min_per_class_f1s.append(metrics["min_per_class_f1"])
            fold_plur_corrs.append(metrics["plur_corr"])
            fold_kappas.append(metrics["cohen_kappa"])
            
            # Accumulate inner metrics
            inner_accs.append(inner_metrics["acc"])
            inner_macro_f1s.append(inner_metrics["macro_f1"])
            inner_min_per_class_f1s.append(inner_metrics["min_per_class_f1"])
            inner_plur_corrs.append(inner_metrics["plur_corr"])
            
            # Accumulate predictions
            overall_y_true.extend(fold_result["y_true"])
            overall_y_pred.extend(fold_result["y_pred"])
            
            # Accumulate artifacts
            outer_folds_record.append(fold_result["fold_record"])
            learning_curve_rows.extend(fold_result["learning_curves"])
            test_pred_rows_outer.extend(fold_result["test_pred_rows_outer"])
            test_pred_rows_inner.extend(fold_result["test_pred_rows_inner"])
            
            # Build outer metrics row for CSV
            per_class_f1 = metrics.get("per_class_f1_list")
            outer_metrics_rows.append({
                "outer_fold": int(fold + 1),
                "test_subjects": ",".join(map(str, test_subjects)),
                "n_test_trials": int(len(fold_result["y_true"])),
                "acc": float(metrics["acc"]),
                "macro_f1": float(metrics["macro_f1"]),
                "min_per_class_f1": float(metrics["min_per_class_f1"]),
                "plur_corr": float(metrics["plur_corr"]),
                "cohen_kappa": float(metrics["cohen_kappa"]),
                "acc_std": "",
                "macro_f1_std": "",
                "min_per_class_f1_std": "",
                "plur_corr_std": "",
                "cohen_kappa_std": "",
                "per_class_f1": json.dumps(per_class_f1) if per_class_f1 is not None else "",
            })
            
            # NOTE: Plotting is now handled inside OuterFoldOrchestrator.run_fold()
            # No need to generate plots here
            
            # Fold complete (logging handled by orchestrator)

        # Overall metrics
        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        std_acc = float(np.std(fold_accs)) if fold_accs else 0.0
        mean_min_per_class_f1 = float(np.mean(fold_min_per_class_f1s)) if fold_min_per_class_f1s else 0.0
        std_min_per_class_f1 = float(np.std(fold_min_per_class_f1s)) if fold_min_per_class_f1s else 0.0
        mean_plur_corr = float(np.mean(fold_plur_corrs)) if fold_plur_corrs else 0.0
        std_plur_corr = float(np.std(fold_plur_corrs)) if fold_plur_corrs else 0.0
        mean_kappa = float(np.mean(fold_kappas)) if fold_kappas else 0.0
        std_kappa = float(np.std(fold_kappas)) if fold_kappas else 0.0
        try:
            macro_f1 = (
                f1_score(overall_y_true, overall_y_pred, average="macro") * 100 if overall_y_true else 0.0
            )
            weighted_f1 = (
                f1_score(overall_y_true, overall_y_pred, average="weighted") * 100 if overall_y_true else 0.0
            )
            # Overall Cohen's Kappa from all predictions
            cohen_kappa = (
                cohen_kappa_score(overall_y_true, overall_y_pred) if overall_y_true else 0.0
            )
            class_report_str = (
                classification_report(overall_y_true, overall_y_pred, target_names=class_names)
                if overall_y_true
                else "N/A"
            )
        except Exception:
            macro_f1 = weighted_f1 = cohen_kappa = 0.0
            class_report_str = "Error generating classification report."

        # Generate overall plots using OverallPlotOrchestrator (refactored - Stage 7c)
        if self.run_dir and overall_y_true:
            plot_orchestrator = OverallPlotOrchestrator(self.run_dir, plot_title_builder)
            plot_orchestrator.generate_overall_plots(
                overall_y_true=overall_y_true,
                overall_y_pred=overall_y_pred,
                class_names=class_names,
                inner_accs=inner_accs,
                inner_macro_f1s=inner_macro_f1s,
                inner_min_per_class_f1s=inner_min_per_class_f1s,
                inner_plur_corrs=inner_plur_corrs,
                mean_acc=mean_acc,
                macro_f1=macro_f1,
                mean_plur_corr=mean_plur_corr,
            )

        # Write all artifacts using ArtifactWriterOrchestrator (refactored - Stage 7b)
        decision_layer_stats = None
        if self.run_dir:
            artifact_writer = ArtifactWriterOrchestrator(self.run_dir, self.cfg, _log_event)
            artifact_writer.write_all(
                outer_folds_record=outer_folds_record,
                learning_curve_rows=learning_curve_rows,
                outer_metrics_rows=outer_metrics_rows,
                test_pred_rows_outer=test_pred_rows_outer,
                test_pred_rows_inner=test_pred_rows_inner,
                dataset=dataset,
                y_all=y_all,
                groups=groups,
                class_names=class_names,
                mean_acc=mean_acc,
                std_acc=std_acc,
                fold_macro_f1s=fold_macro_f1s,
                fold_min_per_class_f1s=fold_min_per_class_f1s,
                mean_plur_corr=mean_plur_corr,
                std_plur_corr=std_plur_corr,
                mean_kappa=mean_kappa,
                std_kappa=std_kappa,
            )
            # Capture decision layer stats if available (for summary writer)
            decision_layer_stats = artifact_writer.decision_layer_stats

        # Compute all summary metrics
        summary_inner_mean_acc = float(np.mean(inner_accs)) if inner_accs else 0.0
        summary_inner_mean_macro_f1 = float(np.mean(inner_macro_f1s)) if inner_macro_f1s else 0.0
        summary_inner_mean_min_per_class_f1 = float(np.mean(inner_min_per_class_f1s)) if inner_min_per_class_f1s else 0.0
        summary_inner_mean_plur_corr = float(np.mean(inner_plur_corrs)) if inner_plur_corrs else 0.0
        
        # Compute composite objective if configured (threshold or weighted mode)
        summary_composite_min_f1_plur_corr = None
        if self.cfg.get("optuna_objective") == "composite_min_f1_plur_corr":
            params = self._get_composite_params()
            if params["mode"] == "threshold":
                # Threshold mode with gradient below threshold
                if summary_inner_mean_min_per_class_f1 < params["threshold"]:
                    summary_composite_min_f1_plur_corr = summary_inner_mean_min_per_class_f1 * 0.1
                else:
                    summary_composite_min_f1_plur_corr = summary_inner_mean_plur_corr
            else:  # weighted mode
                min_f1_weight = params["weight"]
                plur_corr_weight = 1.0 - min_f1_weight
                summary_composite_min_f1_plur_corr = (
                    min_f1_weight * summary_inner_mean_min_per_class_f1 + 
                    plur_corr_weight * summary_inner_mean_plur_corr
                )

        summary_dict = {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "cohen_kappa": float(cohen_kappa),
            "mean_kappa": float(mean_kappa),
            "std_kappa": float(std_kappa),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "classification_report": class_report_str,
            "fold_accuracies": fold_accs,
            "fold_macro_f1s": fold_macro_f1s,
            "fold_min_per_class_f1s": fold_min_per_class_f1s,
            "fold_kappas": fold_kappas,
            "fold_splits": fold_split_info,
            "inner_mean_acc": summary_inner_mean_acc,
            "inner_mean_macro_f1": summary_inner_mean_macro_f1,
            "inner_mean_min_per_class_f1": summary_inner_mean_min_per_class_f1,
            "inner_mean_plur_corr": summary_inner_mean_plur_corr,
            "composite_min_f1_plur_corr": summary_composite_min_f1_plur_corr,
            "mean_min_per_class_f1": float(mean_min_per_class_f1),
            "std_min_per_class_f1": float(std_min_per_class_f1),
            "mean_plur_corr": float(mean_plur_corr),
            "std_plur_corr": float(std_plur_corr),
            "fold_plur_corrs": fold_plur_corrs,
            "num_classes": int(num_cls),
        }
        
        # Add decision layer stats if available (for summary writer)
        if decision_layer_stats:
            summary_dict["decision_layer_stats"] = decision_layer_stats
        
        return summary_dict


def validate_stage_handoff(stage: str, previous_stage_dir: str | None) -> Dict[str, Any]:
    """Module-level convenience wrapper for tests/contracts."""
    return TrainingRunner.validate_stage_handoff(stage=stage, previous_stage_dir=previous_stage_dir)
