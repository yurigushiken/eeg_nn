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


def load_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description="Step 2 TPE: augmentation-only (fixed preproc)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--materialized", required=True, help="folder with sub-XX_preprocessed-epo.fif (from Step 1 best)")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--db", type=str, default=None)
    ap.add_argument("--study", type=str, default=None, help="Custom Optuna study name/tag")
    args = ap.parse_args()

    task = args.task
    common = load_yaml(proj_root / "configs" / "common.yaml")
    ctrl = load_yaml(Path(args.cfg))
    space = load_yaml(Path(args.space))

    engine_run = engine_registry.get("eeg")
    label_fn = task_registry.get(task)

    study_name = args.study if args.study else f"step2_{task}"
    if args.db:
        storage = args.db
    else:
        db_path = proj_root / "optuna_studies" / f"{study_name}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True)

    def suggest(trial: optuna.Trial) -> Dict[str, Any]:
        cfg = dict(common)
        cfg.update(load_yaml(proj_root / "configs" / "tasks" / task / "base.yaml"))
        # freeze preproc/net/train (already set by base + step1 best merged offline into base if desired)
        # set materialized_dir so dataset loads .fif directly
        cfg["materialized_dir"] = args.materialized
        # be lenient during HPO to avoid aborts
        cfg["strict_behavior_align"] = False
        # sample augmentation-only
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
        cfg["task"] = task
        cfg["model_name"] = cfg.get("model_name", "eegnex")
        seed_everything(cfg.get("seed"))

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = proj_root / "results" / "optuna" / study_name / f"{ts}_{task}_eeg_step2_t{trial.number:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["run_dir"] = str(run_dir)

        summary_raw = engine_run(cfg, label_fn)
        summary = {"run_id": ts, "dataset_dir": cfg.get("materialized_dir"), **summary_raw, "study": study_name, "trial_id": trial.number, "hyper": {k:v for k,v in cfg.items() if k!="run_dir"}}
        write_summary(run_dir, summary, task, "eeg")
        obj = summary.get("inner_mean_macro_f1") or summary.get("inner_mean_acc") or 0.0
        return float(obj)

    study.optimize(suggest, n_trials=args.trials, show_progress_bar=False)
    best = study.best_trial
    best_payload = {"value": best.value, "params": best.params}
    out_dir = proj_root / "results" / "optuna_studies" / task / "step2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best.json").write_text(json.dumps(best_payload, indent=2))
    print("Step2 best:", best_payload)

if __name__ == "__main__":
    main()


