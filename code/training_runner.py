from __future__ import annotations
from typing import Dict, Callable, List

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report
import optuna

from utils.plots import plot_confusion, plot_curves

"""
Training/evaluation orchestration with nested, subject-aware cross-validation.

Outer CV:
  - GroupKFold when cfg['n_folds'] is set (subject-aware K-way)
  - LOSO when 'n_folds' is absent or null

Inner CV:
  - Strict GroupKFold with cfg['inner_n_folds'] (≥2), per outer split (subject-aware)
  - Early stopping on inner validation loss; pruning reports inner macro‑F1 (robust to imbalance)

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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class TrainingRunner:
    def __init__(self, cfg: Dict, label_fn: Callable):
        self.cfg = cfg
        self.label_fn = label_fn
        self.run_dir = self._ensure_run_dir(cfg)

    def _ensure_run_dir(self, cfg: Dict):
        from pathlib import Path
        run_dir = cfg.get("run_dir")
        if run_dir:
            p = Path(run_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        return None

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

    def run(
        self,
        dataset,
        groups,
        class_names,
        model_builder,
        aug_builder,
        input_adapter=None,
        optuna_trial=None,
    ):
        y_all = dataset.get_all_labels()
        num_cls = len(class_names)

        # Log effective randomness controls for reproducibility assurance
        try:
            outer_mode = "GroupKFold" if self.cfg.get("n_folds") else "LOSO"
            print(
                f"[config] seed={self.cfg.get('seed')} random_state={self.cfg.get('random_state')} outer_mode={outer_mode}",
                flush=True,
            )
        except Exception:
            pass

        # Split strategy: n_folds (GroupKFold) or LOSO (subject-wise)
        if self.cfg.get("n_folds"):
            k = int(self.cfg["n_folds"])
            gkf_outer = GroupKFold(n_splits=k)
            fold_iter = gkf_outer.split(np.zeros(len(dataset)), y_all, groups)
        else:
            fold_iter = LeaveOneGroupOut().split(np.zeros(len(dataset)), y_all, groups)

        fold_accs: List[float] = []
        fold_split_info: List[dict] = []
        overall_y_true: List[int] = []
        overall_y_pred: List[int] = []
        inner_accs: List[float] = []
        inner_macro_f1s: List[float] = []

        # Prepare augmentation transform once (stateless transform expected)
        aug_transform = aug_builder(self.cfg, dataset) if aug_builder else None

        # Global step counter for Optuna pruning (increments every epoch across folds)
        global_step = 0

        for fold, (tr_idx, va_idx) in enumerate(fold_iter):
            if self.cfg.get("max_folds") is not None and fold >= int(self.cfg["max_folds"]):
                break
            test_subjects = np.unique(groups[va_idx]).tolist()
            fold_split_info.append({"fold": fold + 1, "test_subjects": test_subjects})
            print(
                f"[fold {fold+1}] test_subjects={test_subjects} n_tr={len(tr_idx)} n_te={len(va_idx)}",
                flush=True,
            )

            # Determine inner K-fold configuration (strict)
            inner_k = int(self.cfg.get("inner_n_folds", 1))
            unique_train_groups = np.unique(groups[tr_idx])
            if inner_k < 2:
                raise ValueError("inner_n_folds must be >= 2 for strict inner K-fold CV")
            if len(unique_train_groups) < inner_k:
                raise ValueError(
                    f"Not enough unique groups for inner K-fold: have {len(unique_train_groups)}, need >= {inner_k}"
                )

            # Collect per-outer-fold inner results
            inner_results_this_outer: List[dict] = []
            best_inner_result = None
            te_ld_shared = None

            gkf_inner = GroupKFold(n_splits=inner_k)
            for inner_fold, (inner_tr_rel, inner_va_rel) in enumerate(
                gkf_inner.split(np.zeros(len(tr_idx)), y_all[tr_idx], groups[tr_idx])
            ):
                inner_tr_abs = tr_idx[inner_tr_rel]
                inner_va_abs = tr_idx[inner_va_rel]

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

                best_val = float("inf")
                best_state = None
                best_inner_acc = 0.0
                best_inner_macro_f1 = 0.0
                patience = 0
                tr_hist: List[float] = []
                va_hist: List[float] = []
                va_acc_hist: List[float] = []

                for epoch in range(1, int(self.cfg.get("epochs", 60)) + 1):
                    # Train
                    model.train()
                    train_loss = 0.0
                    for xb, yb in tr_ld:
                        yb_gpu = yb.to(DEVICE)
                        xb_gpu = (
                            xb.to(DEVICE)
                            if not isinstance(xb, (list, tuple))
                            else [t.to(DEVICE) for t in xb]
                        )
                        xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu
                        opt.zero_grad()

                        # Optional mixup: only for tensor inputs with batch_size>1
                        mixup_alpha = float(self.cfg.get("mixup_alpha", 0.0) or 0.0)
                        if (
                            mixup_alpha > 0.0
                            and isinstance(xb_gpu, torch.Tensor)
                            and xb_gpu.size(0) > 1
                        ):
                            lam = np.random.beta(mixup_alpha, mixup_alpha)
                            perm = torch.randperm(xb_gpu.size(0), device=xb_gpu.device)
                            xb_mix = lam * xb_gpu + (1.0 - lam) * xb_gpu[perm]
                            out = model(xb_mix)
                            yb_perm = yb_gpu[perm]
                            loss = lam * loss_fn(out.float(), yb_gpu) + (1.0 - lam) * loss_fn(out.float(), yb_perm)
                        else:
                            out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                            loss = loss_fn(out.float(), yb_gpu)
                        loss.backward()
                        opt.step()
                        train_loss += loss.item()
                    train_loss /= max(1, len(tr_ld))
                    tr_hist.append(train_loss)

                    # Val
                    model.eval()
                    val_loss = 0.0
                    correct = 0
                    total = 0
                    y_true_ep: List[int] = []
                    y_pred_ep: List[int] = []
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
                            loss = loss_fn(out.float(), yb_gpu)
                            val_loss += loss.item()
                            preds = out.argmax(1).cpu()
                            correct += (preds == yb).sum().item()
                            total += yb.size(0)
                            y_true_ep.extend(yb.tolist())
                            y_pred_ep.extend(preds.tolist())
                    val_loss /= max(1, len(va_ld))
                    val_acc = 100.0 * correct / max(1, total)
                    try:
                        val_macro_f1 = f1_score(y_true_ep, y_pred_ep, average="macro") * 100
                    except Exception:
                        val_macro_f1 = 0.0

                    # Optuna pruning: report inner-val macro-F1 per epoch and allow pruning
                    if optuna_trial is not None:
                        global_step += 1
                        try:
                            optuna_trial.report(val_macro_f1, global_step)
                            if optuna_trial.should_prune():
                                print(
                                    f"  [prune] Trial pruned at epoch {epoch} of fold {fold+1} inner {inner_fold+1}.",
                                    flush=True,
                                )
                                raise optuna.exceptions.TrialPruned()
                        except optuna.exceptions.TrialPruned:
                            raise
                        except Exception:
                            # Ignore reporting errors without stopping training
                            pass
                    va_hist.append(val_loss)
                    va_acc_hist.append(val_acc)
                    sched.step(val_loss)

                    # Early stopping remains on val_loss
                    if val_loss < best_val:
                        best_val = val_loss
                        patience = 0
                    else:
                        patience += 1

                    # Checkpoint selection aligns with objective: maximize macro‑F1
                    update_ckpt = False
                    if val_macro_f1 > best_inner_macro_f1:
                        update_ckpt = True
                    elif (val_macro_f1 == best_inner_macro_f1) and (val_loss < best_val):
                        # Tie-break by lower validation loss
                        update_ckpt = True

                    if update_ckpt:
                        best_inner_macro_f1 = val_macro_f1
                        best_inner_acc = val_acc
                        best_state = copy.deepcopy(model.state_dict())
                        if self.run_dir and self.cfg.get("save_ckpt", True):
                            torch.save(
                                model.state_dict(),
                                self.run_dir / f"fold_{fold+1:02d}_inner_{inner_fold+1:02d}_best.ckpt",
                            )
                    if patience >= int(self.cfg.get("early_stop", 10)):
                        break
                    if epoch % 5 == 0:
                        print(
                            f"  [epoch {epoch}] (outer {fold+1} inner {inner_fold+1}) tr_loss={train_loss:.4f} va_loss={val_loss:.4f} va_acc={val_acc:.2f} best_val={best_val:.4f}",
                            flush=True,
                        )

                inner_results_this_outer.append(
                    {
                        "best_state": best_state,
                        "best_inner_acc": best_inner_acc,
                        "best_inner_macro_f1": best_inner_macro_f1,
                        "tr_hist": tr_hist,
                        "va_hist": va_hist,
                        "va_acc_hist": va_acc_hist,
                    }
                )

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
            inner_accs.append(inner_mean_acc_this_outer)
            inner_macro_f1s.append(inner_mean_macro_f1_this_outer)

            # Select best inner model for plotting curves (ensemble used for test predictions)
            if inner_results_this_outer:
                best_inner_result = max(
                    inner_results_this_outer,
                    key=lambda r: r["best_inner_macro_f1"],
                )

            mode = str(self.cfg.get("outer_eval_mode", "ensemble")).lower()

            # Test-time evaluation
            correct = 0
            total = 0
            y_true_fold: List[int] = []
            y_pred_fold: List[int] = []
            if mode == "ensemble":
                # Ensemble of inner models (mean softmax over K inner models)
                with torch.no_grad():
                    for xb, yb in (te_ld_shared if te_ld_shared is not None else []):
                        yb_gpu = yb.to(DEVICE)
                        xb_gpu = xb.to(DEVICE) if not isinstance(xb, (list, tuple)) else [t.to(DEVICE) for t in xb]
                        xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu

                        accum_probs = None
                        for r in inner_results_this_outer:
                            state = r.get("best_state")
                            if state is None:
                                continue
                            model = model_builder(self.cfg, num_cls).to(DEVICE)
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
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                        y_true_fold.extend(yb.tolist())
                        y_pred_fold.extend(preds.tolist())
            elif mode == "refit":
                # Refit a single model on the full outer-train set (optionally with a small deterministic grouped val)
                refit_val_frac = float(self.cfg.get("refit_val_frac", 0.0) or 0.0)
                do_val = refit_val_frac > 0.0 and len(np.unique(groups[tr_idx])) >= 2

                if do_val:
                    # Use deterministic GroupKFold with fixed K; pick first split
                    refit_val_k = int(self.cfg.get("refit_val_k", 5))
                    gkf_refit = GroupKFold(n_splits=max(2, refit_val_k))
                    inner_tr_rel, inner_va_rel = next(
                        gkf_refit.split(np.zeros(len(tr_idx)), y_all[tr_idx], groups[tr_idx])
                    )
                    refit_tr_abs = tr_idx[inner_tr_rel]
                    refit_va_abs = tr_idx[inner_va_rel]
                else:
                    refit_tr_abs = tr_idx
                    refit_va_abs = tr_idx  # sentinel; we will skip early stopping

                tr_ld, va_ld, te_ld_refit, _ = self._make_loaders(
                    dataset,
                    y_all,
                    groups,
                    tr_idx,
                    va_idx,
                    aug_transform=aug_transform,
                    inner_tr_idx_override=refit_tr_abs,
                    inner_va_idx_override=refit_va_abs,
                )

                model = model_builder(self.cfg, num_cls).to(DEVICE)
                opt = torch.optim.AdamW(
                    model.parameters(), lr=float(self.cfg.get("lr", 7e-4)), weight_decay=float(self.cfg.get("weight_decay", 0.0))
                )
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", patience=int(self.cfg.get("scheduler_patience", 5))
                )
                cls_w = compute_class_weight(
                    "balanced", classes=np.arange(num_cls), y=y_all[refit_tr_abs]
                )
                loss_fn = nn.CrossEntropyLoss(torch.tensor(cls_w, dtype=torch.float32, device=DEVICE))

                best_val = float("inf")
                best_state = None
                patience = 0
                refit_patience = int(self.cfg.get("refit_early_stop", self.cfg.get("early_stop", 10)))

                for epoch in range(1, int(self.cfg.get("epochs", 60)) + 1):
                    # Train
                    model.train()
                    train_loss = 0.0
                    for xb, yb in tr_ld:
                        yb_gpu = yb.to(DEVICE)
                        xb_gpu = (
                            xb.to(DEVICE)
                            if not isinstance(xb, (list, tuple))
                            else [t.to(DEVICE) for t in xb]
                        )
                        xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu
                        opt.zero_grad()
                        out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                        loss = loss_fn(out.float(), yb_gpu)
                        loss.backward()
                        opt.step()
                        train_loss += loss.item()
                    train_loss /= max(1, len(tr_ld))

                    if do_val:
                        # Validate
                        model.eval()
                        val_loss = 0.0
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
                                loss = loss_fn(out.float(), yb_gpu)
                                val_loss += loss.item()
                        val_loss /= max(1, len(va_ld))
                        sched.step(val_loss)
                        if val_loss < best_val:
                            best_val = val_loss
                            patience = 0
                            best_state = copy.deepcopy(model.state_dict())
                        else:
                            patience += 1
                            if patience >= refit_patience:
                                break

                if do_val and best_state is not None:
                    model.load_state_dict(best_state)
                model.eval()

                # Save refit model checkpoint if requested
                if self.run_dir and self.cfg.get("save_ckpt", True):
                    torch.save(model.state_dict(), self.run_dir / f"fold_{fold+1:02d}_refit_best.ckpt")

                with torch.no_grad():
                    for xb, yb in te_ld_refit:
                        yb_gpu = yb.to(DEVICE)
                        xb_gpu = xb.to(DEVICE) if not isinstance(xb, (list, tuple)) else [t.to(DEVICE) for t in xb]
                        xb_gpu = input_adapter(xb_gpu) if input_adapter else xb_gpu
                        out = model(xb_gpu) if not isinstance(xb_gpu, (list, tuple)) else model(*xb_gpu)
                        preds = out.argmax(1).cpu()
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                        y_true_fold.extend(yb.tolist())
                        y_pred_fold.extend(preds.tolist())
            else:
                raise ValueError(f"Unknown outer_eval_mode={mode}; use 'ensemble' or 'refit'")

            acc = 100.0 * correct / max(1, total)
            fold_accs.append(acc)
            overall_y_true.extend(y_true_fold)
            overall_y_pred.extend(y_pred_fold)
            print(
                f"[fold {fold+1}] acc={acc:.2f} inner_mean_acc={inner_mean_acc_this_outer:.2f} inner_mean_macro_f1={inner_mean_macro_f1_this_outer:.2f}",
                flush=True,
            )

            # Plots per fold (use best inner fold histories for curves; ensemble/refit guides predictions)
            if self.run_dir and best_inner_result:
                fold_title = (
                    f"Fold {fold+1} (Subjects: {test_subjects}) · "
                    f"inner-mean macro-F1={inner_mean_macro_f1_this_outer:.2f} · acc={acc:.2f}"
                )
                plot_confusion(
                    y_true_fold,
                    y_pred_fold,
                    class_names,
                    self.run_dir / f"fold{fold+1}_confusion.png",
                    title=fold_title,
                )
                plot_curves(
                    best_inner_result["tr_hist"],
                    best_inner_result["va_hist"],
                    best_inner_result["va_acc_hist"],
                    self.run_dir / f"fold{fold+1}_curves.png",
                    title=fold_title,
                )

        # Overall metrics
        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        std_acc = float(np.std(fold_accs)) if fold_accs else 0.0
        try:
            macro_f1 = (
                f1_score(overall_y_true, overall_y_pred, average="macro") * 100 if overall_y_true else 0.0
            )
            weighted_f1 = (
                f1_score(overall_y_true, overall_y_pred, average="weighted") * 100 if overall_y_true else 0.0
            )
            class_report_str = (
                classification_report(overall_y_true, overall_y_pred, target_names=class_names)
                if overall_y_true
                else "N/A"
            )
        except Exception:
            macro_f1 = weighted_f1 = 0.0
            class_report_str = "Error generating classification report."

        # Overall plot (confusion across all outer test predictions)
        if self.run_dir and overall_y_true:
            overall_title = (
                f"Overall · inner_mean_macro_f1={float(np.mean(inner_macro_f1s)) if inner_macro_f1s else 0.0:.2f} "
                f"· mean_acc={mean_acc:.2f}"
            )
            plot_confusion(
                overall_y_true,
                overall_y_pred,
                class_names,
                self.run_dir / "overall_confusion.png",
                title=overall_title,
            )

        return {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "classification_report": class_report_str,
            "fold_accuracies": fold_accs,
            "fold_splits": fold_split_info,
            "inner_mean_acc": float(np.mean(inner_accs)) if inner_accs else 0.0,
            "inner_mean_macro_f1": float(np.mean(inner_macro_f1s)) if inner_macro_f1s else 0.0,
        }


