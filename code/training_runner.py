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
from .artifacts.csv_writers import (
    LearningCurvesWriter,
    OuterEvalMetricsWriter,
    TestPredictionsWriter,
)
from .artifacts.plot_builders import PlotTitleBuilder

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


def compute_plurality_correctness(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute plurality correctness (row-wise plurality metric).
    
    For each true class (row in confusion matrix), checks if the correct 
    prediction (diagonal element) is the most frequent prediction.
    
    Returns proportion of classes where correct prediction is plurality.
    Score ranges from 0.0 (no correct pluralities) to 1.0 (all correct pluralities).
    
    Example:
        Confusion matrix:
            Pred:  1   2   3
        True 1: [100  30  20]  ← max is 100 (correct) ✓
        True 2: [ 40  60  50]  ← max is 60 (correct) ✓
        True 3: [ 10  80  30]  ← max is 80 (incorrect) ✗
        
        Result: 2/3 = 0.667 (two out of three classes have correct prediction as plurality)
    """
    if not y_true or not y_pred:
        return 0.0
    
    try:
        # Get unique classes
        classes = sorted(set(y_true) | set(y_pred))
        n_classes = len(classes)
        
        if n_classes == 0:
            return 0.0
        
        # Compute confusion matrix (rows=true, cols=pred)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # For each row (true class), check if diagonal element is the maximum
        diagonal_is_max_count = 0
        for i in range(n_classes):
            row = cm[i, :]
            if len(row) > 0:
                max_val = np.max(row)
                diagonal_val = cm[i, i]
                # Diagonal is plurality if it equals the max
                if diagonal_val == max_val:
                    diagonal_is_max_count += 1
        
        return float(diagonal_is_max_count) / float(n_classes)
    
    except Exception:
        return 0.0

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

        # Setup JSONL runtime log per logging contract (Option B)
        jsonl_log_path = None
        if self.run_dir:
            logs_dir = self.run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            jsonl_log_path = logs_dir / "runtime.jsonl"
            # Initialize with run start event
            from datetime import datetime, timezone
            def _log_event(event: str, message: str = "", **extra):
                if jsonl_log_path:
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "level": "INFO",
                        "logger": "training_runner",
                        "event": event,
                        "run_id": self.run_dir.name if self.run_dir else "",
                        "message": message,
                        "extra": extra,
                    }
                    with jsonl_log_path.open("a") as f:
                        f.write(json.dumps(record) + "\n")
            _log_event("run_start", f"Starting run with seed={self.cfg.get('seed')}")
        else:
            def _log_event(event: str, message: str = "", **extra):
                pass  # no-op if no run_dir

        # Log chance level for awareness
        try:
            if num_cls > 0:
                chance = 100.0 / float(num_cls)
                _log_event("chance_level_computed", f"chance={chance:.2f}%", num_classes=num_cls)
        except Exception:
            pass

        # Log effective randomness controls for reproducibility assurance
        try:
            outer_mode = "GroupKFold" if self.cfg.get("n_folds") else "LOSO"
            print(
                f"[config] seed={self.cfg.get('seed')} outer_mode={outer_mode}",
                flush=True,
            )
        except Exception:
            pass

        # Generate channel selection topomap for scientific transparency
        if self.run_dir:
            try:
                from pathlib import Path
                proj_root = Path(__file__).resolve().parents[1]
                montage_path = proj_root / "net" / "AdultAverageNet128_v1.sfp"
                if montage_path.exists():
                    save_channel_topomap_for_run(self.cfg, self.run_dir, montage_path)
                else:
                    print(f"[channel_viz] WARNING: Montage not found at {montage_path}", flush=True)
            except Exception as e:
                print(f"[channel_viz] WARNING: Could not generate channel topomap: {e}", flush=True)

        # Precompute outer fold index pairs
        outer_pairs: List[tuple] = []
        if predefined_splits:
            for rec in predefined_splits:
                outer_pairs.append((np.array(rec["outer_train_idx"]), np.array(rec["outer_test_idx"])))
        else:
            if self.cfg.get("n_folds"):
                k = int(self.cfg["n_folds"])
                gkf_outer = GroupKFold(n_splits=k)
                outer_pairs = [
                    (np.array(tr), np.array(te))
                    for tr, te in gkf_outer.split(np.zeros(len(dataset)), y_all, groups)
                ]
            else:
                outer_pairs = [
                    (np.array(tr), np.array(te))
                    for tr, te in LeaveOneGroupOut().split(np.zeros(len(dataset)), y_all, groups)
                ]

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

            # Collect per-outer-fold inner results
            inner_results_this_outer: List[dict] = []
            best_inner_result = None
            te_ld_shared = None

            # Prepare record for this outer fold
            fold_record = {
                "fold": int(fold + 1),
                "outer_train_idx": [int(i) for i in tr_idx.tolist()],
                "outer_test_idx": [int(i) for i in va_idx.tolist()],
                "outer_train_subjects": [int(s) for s in np.unique(groups[tr_idx]).tolist()],
                "outer_test_subjects": [int(s) for s in np.unique(groups[va_idx]).tolist()],
                "inner_splits": [],
            }

            # Determine inner splits (predefined or computed)
            predefined_inner = None
            if predefined_splits and fold < len(predefined_splits):
                predefined_inner = predefined_splits[fold].get("inner_splits")

            if predefined_inner:
                inner_iter = [
                    (np.array(sp["inner_train_idx"]), np.array(sp["inner_val_idx"]))
                    for sp in predefined_inner
                ]
            else:
                gkf_inner = GroupKFold(n_splits=inner_k)
                inner_iter = [
                    (tr_idx[np.array(inner_tr_rel)], tr_idx[np.array(inner_va_rel)])
                    for inner_tr_rel, inner_va_rel in gkf_inner.split(
                        np.zeros(len(tr_idx)), y_all[tr_idx], groups[tr_idx]
                    )
                ]

            # Initialize inner trainer (refactored)
            inner_trainer = InnerTrainer(self.cfg, self.objective_computer)

            for inner_fold, (inner_tr_abs, inner_va_abs) in enumerate(inner_iter):
                # Inner split leakage guard
                tr_subj = set(np.unique(groups[inner_tr_abs]).tolist())
                va_subj = set(np.unique(groups[inner_va_abs]).tolist())
                if tr_subj.intersection(va_subj):
                    raise AssertionError(
                        f"Subject leakage detected in inner fold {inner_fold+1} of outer {fold+1}: {sorted(list(tr_subj.intersection(va_subj)))}"
                    )
                # Record inner split indices for this outer fold
                fold_record["inner_splits"].append({
                    "inner_fold": int(inner_fold + 1),
                    "inner_train_idx": [int(i) for i in inner_tr_abs.tolist()],
                    "inner_val_idx": [int(i) for i in inner_va_abs.tolist()],
                })

                # Create dataloaders
                tr_ld, va_ld, te_ld, _ = self._make_loaders(
                    dataset,
                    y_all,
                    groups,
                    tr_idx,
                    va_idx,
                    aug_transform=aug_transform,
                    inner_tr_idx_override=inner_tr_abs,
                    inner_va_idx_override=inner_va_abs,
                )
                te_ld_shared = te_ld

                # Setup model, optimizer, scheduler, loss
                model = model_builder(self.cfg, num_cls).to(DEVICE)
                opt = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(self.cfg.get("lr", 7e-4)),
                    weight_decay=float(self.cfg.get("weight_decay", 0.0)),
                )
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", patience=int(self.cfg.get("scheduler_patience", 5))
                )
                # Class weights computed from inner-train only (no leakage)
                cls_w = compute_class_weight(
                    "balanced", classes=np.arange(num_cls), y=y_all[inner_tr_abs]
                )
                loss_fn = nn.CrossEntropyLoss(torch.tensor(cls_w, dtype=torch.float32, device=DEVICE))
                
                # Save class weights
                try:
                    if self.run_dir:
                        cw_dir = self.run_dir / "class_weights"
                        cw_dir.mkdir(parents=True, exist_ok=True)
                        cw_path = cw_dir / f"fold_{fold+1:02d}_inner_{inner_fold+1:02d}_class_weights.json"
                        cw_payload = {str(i): float(w) for i, w in enumerate(cls_w)}
                        cw_path.write_text(json.dumps(cw_payload, indent=2))
                        _log_event("class_weights_saved", f"Saved class weights for fold {fold+1} inner {inner_fold+1}", fold=fold+1, inner_fold=inner_fold+1)
                except Exception:
                    pass

                # Initialize checkpoint manager
                checkpoint_mgr = CheckpointManager(self.cfg, self.objective_computer)
                
                # Train using InnerTrainer (refactored)
                inner_result = inner_trainer.train(
                    model=model,
                    optimizer=opt,
                    scheduler=sched,
                    loss_fn=loss_fn,
                    tr_loader=tr_ld,
                    va_loader=va_ld,
                    aug_builder=aug_builder,
                    input_adapter=input_adapter,
                    checkpoint_manager=checkpoint_mgr,
                    fold_info={"outer_fold": fold + 1, "inner_fold": inner_fold + 1},
                    optuna_trial=optuna_trial,
                    global_step_offset=global_step,
                )
                
                # Update global step for Optuna
                global_step += len(inner_result["learning_curves"])
                
                # Collect learning curves
                learning_curve_rows.extend(inner_result["learning_curves"])
                
                # Save best checkpoint if requested
                if self.run_dir and self.cfg.get("save_ckpt", True) and inner_result["best_state"] is not None:
                            ckpt_dir = self.run_dir / "ckpt"
                            ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(inner_result["best_state"], ckpt_dir / f"fold_{fold+1:02d}_inner_{inner_fold+1:02d}_best.ckpt")
                
                # Collect inner validation predictions using best state
                if inner_result["best_state"] is not None:
                    model.load_state_dict(inner_result["best_state"])
                    model.eval()
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
                                abs_idx = int(inner_va_abs[j]) if j < len(inner_va_abs) else -1
                                subj_id = int(groups[abs_idx]) if abs_idx >= 0 else -1
                                true_lbl = int(yb[j].item())
                                pred_lbl = int(preds[j].item())
                                probs_vec = probs[j].tolist()
                                p_true = float(probs_vec[true_lbl]) if 0 <= true_lbl < len(probs_vec) else 0.0
                                logp_true = float(np.log(max(p_true, 1e-12)))
                                test_pred_rows_inner.append({
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

                # Store inner result
                inner_results_this_outer.append(inner_result)

            # Aggregate inner metrics for this outer fold
            inner_mean_acc_this_outer = (
                float(np.mean([r["best_inner_acc"] for r in inner_results_this_outer]))
                if inner_results_this_outer
                else 0.0
            )
            inner_mean_macro_f1_this_outer = (
                float(np.mean([r["best_inner_macro_f1"] for r in inner_results_this_outer]))
                if inner_results_this_outer
                else 0.0
            )
            inner_mean_min_per_class_f1_this_outer = (
                float(np.mean([r["best_inner_min_per_class_f1"] for r in inner_results_this_outer]))
                if inner_results_this_outer
                else 0.0
            )
            inner_mean_plur_corr_this_outer = (
                float(np.mean([r["best_inner_plur_corr"] for r in inner_results_this_outer]))
                if inner_results_this_outer
                else 0.0
            )
            inner_accs.append(inner_mean_acc_this_outer)
            inner_macro_f1s.append(inner_mean_macro_f1_this_outer)
            inner_min_per_class_f1s.append(inner_mean_min_per_class_f1_this_outer)
            inner_plur_corrs.append(inner_mean_plur_corr_this_outer)

            # Select best inner model based on optuna_objective
            if "optuna_objective" not in self.cfg:
                raise ValueError(
                    "'optuna_objective' must be explicitly specified in config for scientific validity. "
                    "No fallback allowed. Choose from: inner_mean_macro_f1, inner_mean_min_per_class_f1, inner_mean_plur_corr, inner_mean_acc, composite_min_f1_plur_corr"
                )
            optuna_objective = self.cfg["optuna_objective"]
            
            # Map objective to the metric key in inner results
            # Special handling for composite objective
            if optuna_objective == "composite_min_f1_plur_corr":
                # For composite objective, compute score per inner fold using same logic as _compute_objective_metric
                # Supports both threshold and weighted modes
                params = self._get_composite_params()
                
                def compute_composite_score(r):
                    min_f1 = r["best_inner_min_per_class_f1"]
                    plur_corr = r["best_inner_plur_corr"]
                    if params["mode"] == "threshold":
                        # Use gradient below threshold (same as _compute_objective_metric)
                        if min_f1 < params["threshold"]:
                            return min_f1 * 0.1
                        else:
                            return plur_corr
                    else:  # weighted
                        return params["weight"] * min_f1 + (1.0 - params["weight"]) * plur_corr
                
                if inner_results_this_outer:
                    best_inner_result = max(inner_results_this_outer, key=compute_composite_score)
                else:
                    best_inner_result = None
            else:
                objective_to_metric = {
                    "inner_mean_macro_f1": "best_inner_macro_f1",
                    "inner_mean_min_per_class_f1": "best_inner_min_per_class_f1",
                    "inner_mean_plur_corr": "best_inner_plur_corr",
                    "inner_mean_acc": "best_inner_acc",
                }
                
                if optuna_objective not in objective_to_metric:
                    raise ValueError(
                        f"Invalid optuna_objective: '{optuna_objective}'. "
                        f"Must be one of: {list(objective_to_metric.keys())} or composite_min_f1_plur_corr"
                    )
                metric_key = objective_to_metric[optuna_objective]
                
                if inner_results_this_outer:
                    best_inner_result = max(
                        inner_results_this_outer,
                        key=lambda r: r[metric_key],
                    )
                else:
                    best_inner_result = None

            # Initialize outer evaluator (refactored)
            if "outer_eval_mode" not in self.cfg:
                raise ValueError(
                    "'outer_eval_mode' must be explicitly specified in config. "
                    "Choose 'ensemble' (default recommendation) or 'refit'. No fallback allowed for scientific validity."
                )
            outer_evaluator = OuterEvaluator(self.cfg)
            mode = outer_evaluator.mode
            
            # Evaluate on outer test set (refactored)
            if mode == "ensemble":
                eval_result = outer_evaluator.evaluate(
                    model_builder=model_builder,
                    num_classes=num_cls,
                    inner_results=inner_results_this_outer,
                    test_loader=te_ld_shared,
                    groups=groups,
                    te_idx=va_idx,
                    class_names=class_names,
                    fold=fold + 1,
                )
            elif mode == "refit":
                eval_result = outer_evaluator.evaluate_refit(
                    model_builder=model_builder,
                    num_classes=num_cls,
                    dataset=dataset,
                    y_all=y_all,
                    groups=groups,
                    tr_idx=tr_idx,
                    te_idx=va_idx,
                    test_loader=te_ld_shared,
                    aug_transform=aug_transform,
                    input_adapter=input_adapter,
                    class_names=class_names,
                    fold=fold + 1,
                )
                # Save refit checkpoint if requested
                if self.run_dir and self.cfg.get("save_ckpt", True):
                    # Note: OuterEvaluator doesn't return model state for refit
                    # Checkpoint saving is handled internally in evaluate_refit if needed
                    pass
                        else:
                raise ValueError(f"Unknown outer_eval_mode={mode}")
            
            # Extract results
            y_true_fold = eval_result["y_true"]
            y_pred_fold = eval_result["y_pred"]
            test_pred_rows_outer.extend(eval_result["test_pred_rows"])
            
            correct = sum(1 for t, p in zip(y_true_fold, y_pred_fold) if t == p)
            total = len(y_true_fold)
            acc = 100.0 * correct / max(1, total)
            # Per-fold macro F1, per-class F1, plurality correctness, and Cohen's Kappa
            try:
                macro_f1_fold = (
                    f1_score(y_true_fold, y_pred_fold, average="macro") * 100 if y_true_fold else 0.0
                )
                per_class_f1 = (
                    f1_score(y_true_fold, y_pred_fold, average=None).tolist() if y_true_fold else None
                )
                plur_corr_fold = (
                    compute_plurality_correctness(y_true_fold, y_pred_fold) * 100 if y_true_fold else 0.0
                )
                kappa_fold = (
                    cohen_kappa_score(y_true_fold, y_pred_fold) if y_true_fold else 0.0
                )
            except Exception:
                macro_f1_fold = 0.0
                per_class_f1 = None
                plur_corr_fold = 0.0
                kappa_fold = 0.0
            fold_macro_f1s.append(macro_f1_fold)
            fold_plur_corrs.append(plur_corr_fold)
            fold_kappas.append(kappa_fold)
            # Compute min-per-class F1 for this fold
            min_f1_fold = float(np.min(per_class_f1)) * 100 if per_class_f1 is not None and len(per_class_f1) > 0 else 0.0
            fold_min_per_class_f1s.append(min_f1_fold)
            
            # Record outer-fold row
            outer_metrics_rows.append({
                "outer_fold": int(fold + 1),
                "test_subjects": ",".join(map(str, test_subjects)),
                "n_test_trials": int(len(y_true_fold)),
                "acc": float(acc),
                "macro_f1": float(macro_f1_fold),
                "min_per_class_f1": float(min_f1_fold),
                "plur_corr": float(plur_corr_fold),
                "cohen_kappa": float(kappa_fold),
                "acc_std": "",
                "macro_f1_std": "",
                "min_per_class_f1_std": "",
                "plur_corr_std": "",
                "cohen_kappa_std": "",
                "per_class_f1": json.dumps(per_class_f1) if per_class_f1 is not None else "",
            })
            fold_accs.append(acc)
            overall_y_true.extend(y_true_fold)
            overall_y_pred.extend(y_pred_fold)
            print(
                f"[fold {fold+1}] acc={acc:.2f} kappa={kappa_fold:.3f} inner_mean_acc={inner_mean_acc_this_outer:.2f} inner_mean_macro_f1={inner_mean_macro_f1_this_outer:.2f}",
                flush=True,
            )

            # Plots per outer fold (using PlotTitleBuilder - Stage 3 refactored)
            if self.run_dir and best_inner_result:
                plots_dir = self.run_dir / "plots_outer"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Build plot data using PlotTitleBuilder (refactored)
                inner_metrics_dict = {
                    "acc": inner_mean_acc_this_outer,
                    "macro_f1": inner_mean_macro_f1_this_outer,
                    "min_per_class_f1": inner_mean_min_per_class_f1_this_outer,
                    "plur_corr": inner_mean_plur_corr_this_outer,
                }
                outer_metrics_dict = {
                    "acc": acc,
                    "macro_f1": macro_f1_fold,
                    "plur_corr": plur_corr_fold,
                }
                
                # Simple fold plots
                fold_title = plot_title_builder.build_fold_title_simple(
                    fold=fold,
                    test_subjects=test_subjects,
                    inner_metrics=inner_metrics_dict,
                    outer_acc=acc,
                )
                plot_confusion(
                    y_true_fold,
                    y_pred_fold,
                    class_names,
                    plots_dir / f"fold{fold+1}_confusion.png",
                    title=fold_title,
                )
                plot_curves(
                    best_inner_result["tr_hist"],
                    best_inner_result["va_hist"],
                    best_inner_result["va_acc_hist"],
                    plots_dir / f"fold{fold+1}_curves.png",
                    title=fold_title,
                )
                
                # Enhanced plots with inner vs. outer metric comparison
                plots_enhanced_dir = self.run_dir / "plots_outer_enhanced"
                plots_enhanced_dir.mkdir(parents=True, exist_ok=True)
                
                fold_title_enhanced = plot_title_builder.build_fold_title_enhanced(
                    fold=fold,
                    test_subjects=test_subjects,
                    inner_metrics=inner_metrics_dict,
                    outer_metrics=outer_metrics_dict,
                    per_class_f1=per_class_f1,
                )
                
                per_class_info_lines = plot_title_builder.build_per_class_info(per_class_f1, class_names)
                
                plot_confusion(
                    y_true_fold,
                    y_pred_fold,
                    class_names,
                    plots_enhanced_dir / f"fold{fold+1}_confusion.png",
                    title=fold_title_enhanced,
                    hyper_lines=per_class_info_lines if per_class_info_lines else None,
                )
                plot_curves(
                    best_inner_result["tr_hist"],
                    best_inner_result["va_hist"],
                    best_inner_result["va_acc_hist"],
                    plots_enhanced_dir / f"fold{fold+1}_curves.png",
                    title=fold_title_enhanced,
                )

            # Append fold record after successful processing
            outer_folds_record.append(fold_record)
            _log_event("fold_end", f"Completed outer fold {fold+1}", fold=fold+1)

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

        # Overall plot (confusion across all outer test predictions) - using PlotTitleBuilder (Stage 3 refactored)
        if self.run_dir and overall_y_true:
            plots_dir = self.run_dir / "plots_outer"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Build overall plot data using PlotTitleBuilder (refactored)
            overall_inner_metrics = {
                "acc": float(np.mean(inner_accs)) if inner_accs else 0.0,
                "macro_f1": float(np.mean(inner_macro_f1s)) if inner_macro_f1s else 0.0,
                "min_per_class_f1": float(np.mean(inner_min_per_class_f1s)) if inner_min_per_class_f1s else 0.0,
                "plur_corr": float(np.mean(inner_plur_corrs)) if inner_plur_corrs else 0.0,
            }
            
            # Simple overall plot
            overall_title = plot_title_builder.build_overall_title_simple(
                inner_metrics=overall_inner_metrics,
                outer_mean_acc=mean_acc,
            )
            plot_confusion(
                overall_y_true,
                overall_y_pred,
                class_names,
                plots_dir / "overall_confusion.png",
                title=overall_title,
            )
            
            # Enhanced overall plot with inner vs. outer metric comparison
            plots_enhanced_dir = self.run_dir / "plots_outer_enhanced"
            plots_enhanced_dir.mkdir(parents=True, exist_ok=True)
            
            # Compute overall outer metrics matching the objective
            try:
                overall_per_class_f1 = f1_score(overall_y_true, overall_y_pred, average=None) if overall_y_true else None
            except Exception:
                overall_per_class_f1 = None
            
            overall_outer_metrics = {
                "acc": mean_acc,
                "macro_f1": macro_f1,
                "plur_corr": mean_plur_corr,
            }
            
            overall_title_enhanced = plot_title_builder.build_overall_title_enhanced(
                inner_metrics=overall_inner_metrics,
                outer_metrics=overall_outer_metrics,
                outer_mean_acc=mean_acc,
            )
            
            overall_per_class_info_lines = plot_title_builder.build_per_class_info(overall_per_class_f1, class_names)
            
            plot_confusion(
                overall_y_true,
                overall_y_pred,
                class_names,
                plots_enhanced_dir / "overall_confusion.png",
                title=overall_title_enhanced,
                hyper_lines=overall_per_class_info_lines if overall_per_class_info_lines else None,
            )

        # Write split indices artifact once per run
        if self.run_dir:
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
                _log_event("cv_split_exported", "Exported CV split indices to splits_indices.json")
            except Exception:
                pass

            # Write learning curves CSV once per run
            try:
                outputs_cfg = self.cfg.get("outputs", {}) if isinstance(self.cfg, dict) else {}
                write_curves = bool(outputs_cfg.get("write_learning_curves_csv", True))
                write_outer = bool(outputs_cfg.get("write_outer_eval_csv", True))
                write_preds = bool(outputs_cfg.get("write_test_predictions_csv", True))
                write_splits = bool(outputs_cfg.get("write_splits_indices_json", True))
            except Exception:
                write_curves = write_outer = write_preds = write_splits = True

            # If toggled off, remove splits payload write above
            if not write_splits:
                try:
                    (self.run_dir / "splits_indices.json").unlink(missing_ok=True)  # type: ignore
                except Exception:
                    pass

            # Write learning curves using extracted writer (refactored)
            try:
                if write_curves:
                    writer = LearningCurvesWriter(self.run_dir)
                    writer.write(learning_curve_rows)
            except Exception:
                pass

            # Write per-outer-fold evaluation metrics using extracted writer (refactored)
            try:
                if write_outer:
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
            except Exception:
                pass

            # Write test predictions using extracted writers (refactored)
            try:
                if write_preds and test_pred_rows_outer:
                    writer = TestPredictionsWriter(self.run_dir, mode="outer")
                    writer.write(test_pred_rows_outer)
            except Exception:
                pass

            try:
                if write_preds and test_pred_rows_inner:
                    writer = TestPredictionsWriter(self.run_dir, mode="inner")
                    writer.write(test_pred_rows_inner)
            except Exception:
                pass

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

        return {
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


def validate_stage_handoff(stage: str, previous_stage_dir: str | None) -> Dict[str, Any]:
    """Module-level convenience wrapper for tests/contracts."""
    return TrainingRunner.validate_stage_handoff(stage=stage, previous_stage_dir=previous_stage_dir)
