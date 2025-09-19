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
    ap = argparse.ArgumentParser(description="Step 1 TPE: preproc+net+train (no aug)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--cfg", required=True, help="search controller yaml (folds, subset, etc.)")
    ap.add_argument("--space", required=True, help="yaml defining search space")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--db", type=str, default=None)
    ap.add_argument("--study", type=str, default=None, help="Custom Optuna study name/tag")
    ap.add_argument("--usable-trials-csv", type=str, default=None, help="Optional QC CSV with Kept_Segs_Indxs")
    ap.add_argument("--strict", action='store_true', help="Enable strict behavior alignment (recommended)")
    args = ap.parse_args()

    task = args.task
    common = load_yaml(proj_root / "configs" / "common.yaml")
    ctrl = load_yaml(Path(args.cfg))
    space = load_yaml(Path(args.space))

    engine_run = engine_registry.get("eeg")
    label_fn = task_registry.get(task)

    session_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study if args.study else f"step1_{task}_{session_ts}"
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
        # force augmentation off for step1
        cfg.update({
            "mixup_alpha": 0.0,
            "shift_p": 0.0,
            "scale_p": 0.0,
            "noise_p": 0.0,
            "time_mask_p": 0.0,
            "chan_mask_p": 0.0,
        })
        # subset_n for speed
        if "subset_n" in ctrl:
            cfg["subset_n"] = int(ctrl["subset_n"])
        # be lenient during HPO to avoid aborts; ensure ICA is enabled by default
        cfg["strict_behavior_align"] = False
        cfg["use_ica"] = True
        if args.usable_trials_csv:
            cfg["usable_trials_csv"] = args.usable_trials_csv
        # sample from space
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
        # bookkeeping
        cfg["task"] = task
        cfg["model_name"] = cfg.get("model_name", "eegnex")
        seed_everything(cfg.get("seed"))
        # run dir per trial
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_root = proj_root / "results" / "optuna" / f"{session_ts}_{task}"
        run_dir = session_root / f"{ts}_{task}_eeg_step1_t{trial.number:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["run_dir"] = str(run_dir)
        # engine
        summary_raw = engine_run(cfg, label_fn)
        summary = {
            "run_id": ts,
            "dataset_dir": cfg.get("data_raw_dir"),
            **summary_raw,
            "study": study_name,
            "trial_id": trial.number,
            "hyper": {k: v for k, v in cfg.items() if k not in {"run_dir"}},
        }
        write_summary(run_dir, summary, task, "eeg")
        # objective: inner_mean_macro_f1 if present, else inner_mean_acc
        obj = summary.get("inner_mean_macro_f1") or summary.get("inner_mean_acc") or 0.0
        return float(obj)

    study.optimize(suggest, n_trials=args.trials, show_progress_bar=True)
    best = study.best_trial
    best_payload = {"value": best.value, "params": best.params}
    session_root = proj_root / "results" / "optuna" / f"{session_ts}_{task}"
    (session_root / "best.json").write_text(json.dumps(best_payload, indent=2))
    print("Step1 best:", best_payload)

if __name__ == "__main__":
    main()


