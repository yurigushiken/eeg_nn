#!/usr/bin/env python
from __future__ import annotations
import argparse
import datetime
import json
import yaml
from pathlib import Path
from typing import Dict, Any

import sys
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root / "code") not in sys.path:
    sys.path.insert(0, str(proj_root / "code"))
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import tasks as task_registry
import engines as engine_registry
from utils.seeding import seed_everything
from utils.summary import write_summary

import optuna

"""
Unified Optuna TPE search driver.

Per-trial config layering (in this order):
  common.yaml → base.yaml (or resolved) → optional --cfg overlay → sampled --space

Objective and pruning:
  - The engine/runner returns a summary with nested-CV metrics. The Optuna objective
    is the averaged inner macro‑F1 across inner folds and outer folds. We report
    per-epoch inner macro‑F1 to the pruner (MedianPruner) to terminate weak trials early.

Artifacts per trial (results/optuna/<study>/<timestamp>_...):
  - summary.json (metrics + hyper)
  - resolved_config.yaml (frozen hyperparameters + bookkeeping)
  - plots for folds + overall
"""


def load_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description="Generic Optuna TPE search driver")
    ap.add_argument("--stage", required=True, choices=["step1", "step1_5", "step2", "step3"], help="Naming and output routing only")
    ap.add_argument("--task", required=True)
    # Support both styles: --base (preferred) and --cfg (overlay). Either/both are allowed.
    ap.add_argument("--base", required=False, help="Base YAML config (defaults to configs/tasks/<task>/base.yaml if omitted)")
    ap.add_argument("--cfg", required=False, help="Optional overlay YAML (e.g., stage controller or winners)")
    ap.add_argument("--space", required=True)
    ap.add_argument("--materialized", required=False, default=None,
                    help="Optional dataset dir. If omitted, uses materialized_dir from space or base.yaml")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--db", type=str, default=None)
    ap.add_argument("--study", type=str, default=None, help="Custom Optuna study name/tag")
    args = ap.parse_args()

    task = args.task
    common = load_yaml(proj_root / "configs" / "common.yaml")
    base_path = Path(args.base) if args.base else (proj_root / "configs" / "tasks" / task / "base.yaml")
    base_cfg = load_yaml(base_path)
    ctrl = load_yaml(Path(args.cfg)) if args.cfg else {}
    space_path = Path(args.space)
    if not space_path.exists():
        sys.exit(f"--space not found: {space_path}")
    space = load_yaml(space_path)
    if not isinstance(space, dict) or not space:
        sys.exit(f"--space YAML is empty: {space_path}. Did you point to the correct file?")

    engine_run = engine_registry.get("eeg")
    label_fn = task_registry.get(task)

    default_study = f"{args.stage}_{task}"
    study_name = args.study if args.study else default_study
    if args.db:
        storage = args.db
    else:
        db_path = proj_root / "optuna_studies" / f"{study_name}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path.as_posix()}"
    # Enable TPE sampling and median pruning (prunes after warmup steps)
    if "seed" not in common:
        raise ValueError("'seed' must be specified in common.yaml or base config. No fallback allowed for reproducibility.")
    seed = int(common["seed"])
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def suggest(trial: optuna.Trial) -> Dict[str, Any]:
        cfg = dict(common)
        # Start from base (explicit --base or task base.yaml)
        cfg.update(base_cfg or {})
        # Optional overlay (controller or winners)
        cfg.update(ctrl or {})
        # Sample hyperparameters defined in space YAML
        for k, spec in space.items():
            m = spec.get("method")
            if m == "uniform":
                cfg[k] = trial.suggest_float(k, spec["low"], spec["high"]) 
            elif m == "log_uniform":
                cfg[k] = trial.suggest_float(k, spec["low"], spec["high"], log=True)
            elif m == "int":
                cfg[k] = trial.suggest_int(k, spec["low"], spec["high"]) 
            elif m == "categorical":
                cfg[k] = trial.suggest_categorical(k, spec["choices"]) 
        # materialized_dir precedence: space > CLI > base.yaml
        # Rationale: search spaces may enumerate candidate datasets; CLI wins if set;
        # base.yaml is a default of last resort for convenience.
        if not cfg.get("materialized_dir"):
            if args.materialized:
                cfg["materialized_dir"] = args.materialized
        if not cfg.get("materialized_dir"):
            raise ValueError("materialized_dir is required; set it via space YAML choices, base.yaml default, or pass --materialized")
        cfg["task"] = task
        cfg["model_name"] = cfg.get("model_name", "eegnex")
        seed_everything(cfg.get("seed"))

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = proj_root / "results" / "optuna" / study_name / f"{ts}_{task}_eeg_{args.stage}_t{trial.number:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["run_dir"] = str(run_dir)

        # Pass the live Optuna trial for pruning inside the training loop
        cfg["optuna_trial"] = trial
        summary_raw = engine_run(cfg, label_fn)
        # Remove non-serializable runtime objects before persisting
        cfg.pop("optuna_trial", None)
        safe_hyper = {k: v for k, v in cfg.items() if k not in {"run_dir", "optuna_trial"}}
        
        # Capture command line for reproducibility
        command = " ".join(sys.argv)
        
        summary = {
            "run_id": ts,
            "dataset_dir": cfg.get("materialized_dir"),
            "command": command,  # Add command line
            **summary_raw,
            "study": study_name,
            "trial_id": trial.number,
            "hyper": safe_hyper,
        }
        write_summary(run_dir, summary, task, "eeg")
        # Optionally run post-hoc statistics per trial if enabled in common.yaml
        try:
            stats_cfg = common.get("stats", {}) if isinstance(common, dict) else {}
            if bool(stats_cfg.get("run_posthoc_after_train", False)):
                import subprocess
                cmd = [
                    sys.executable, "-X", "utf8", "-u", str(proj_root / "scripts" / "run_posthoc_stats.py"),
                    "--run-dir", str(run_dir),
                    "--alpha", str(stats_cfg.get("alpha", 0.05)),
                    "--multitest", str(stats_cfg.get("multitest", "fdr")),
                ]
                if bool(stats_cfg.get("glmm", False)):
                    cmd.append("--glmm")
                if bool(stats_cfg.get("forest", True)):
                    cmd.append("--forest")
                subprocess.run(cmd, check=True)
        except Exception:
            pass
        # Optimize based on configured objective metric
        if "optuna_objective" not in cfg:
            raise ValueError(
                "'optuna_objective' must be explicitly specified in config. "
                "Choose from: inner_mean_macro_f1, inner_mean_min_per_class_f1, inner_mean_diag_dom, inner_mean_acc, composite_min_f1_diag_dom. No fallback allowed."
            )
        objective_metric = cfg["optuna_objective"]
        if objective_metric not in summary:
            raise KeyError(
                f"Objective metric '{objective_metric}' not found in summary.json. "
                f"Available metrics: {list(summary.keys())}"
            )
        obj = summary[objective_metric]
        print(f"[optuna] trial {trial.number:03d} objective={objective_metric}: {obj:.4f}", flush=True)
        return float(obj)

    print(f"[optuna] Sampler=TPE, Pruner=Median(n_warmup_steps=10)", flush=True)
    study.optimize(suggest, n_trials=args.trials, show_progress_bar=False)
    best = study.best_trial
    best_payload = {"value": best.value, "params": best.params}
    out_dir = proj_root / "results" / "optuna_studies" / task / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best.json").write_text(json.dumps(best_payload, indent=2))
    print(f"{args.stage.capitalize()} best:", best_payload)

if __name__ == "__main__":
    main()



