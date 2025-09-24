#!/usr/bin/env python
from __future__ import annotations
import argparse
import datetime
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
import subprocess

proj_root = Path(__file__).resolve().parent
code_dir = proj_root / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import tasks as task_registry
import engines as engine_registry
from utils.summary import write_summary
from utils.seeding import seed_everything

def parse_args():
    p = argparse.ArgumentParser(description="Train EEGNeX on a given task/engine pair.")
    p.add_argument("--task", required=True)
    p.add_argument("--engine", required=True, choices=list(engine_registry.ENGINES.keys()))
    # Support both styles: --base (preferred) and optional overlay --cfg
    p.add_argument("--base", help="Base YAML config file (defaults to configs/tasks/<task>/base.yaml)")
    p.add_argument("--cfg", help="Optional overlay YAML with overrides (e.g., winners or controller knobs)")
    p.add_argument("--set", nargs="*", metavar="KEY=VAL", help="Override any hyper-parameter.")
    p.add_argument("--run-xai", action='store_true')
    return p.parse_args()

def build_config(args) -> Dict[str, Any]:
    common_yaml = Path("configs") / "common.yaml"
    cfg = yaml.safe_load(common_yaml.read_text()) if common_yaml.exists() else {}

    base_yaml = Path(args.base) if args.base else (Path("configs") / "tasks" / args.task / "base.yaml")
    if base_yaml.exists():
        base_cfg = yaml.safe_load(base_yaml.read_text()) or {}
        cfg.update(base_cfg)

    # Optional overlay
    if args.cfg:
        overlay_yaml = Path(args.cfg)
        if overlay_yaml.exists():
            overlay_cfg = yaml.safe_load(overlay_yaml.read_text()) or {}
            cfg.update(overlay_cfg)

    if args.set:
        for kv in args.set:
            if "=" not in kv:
                sys.exit(f"--set expects KEY=VAL, not {kv}")
            k, v = kv.split("=", 1)
            try:
                cfg[k] = yaml.safe_load(v)
            except yaml.YAMLError:
                cfg[k] = v
    return cfg

def main():
    args = parse_args()
    cfg = build_config(args)
    cfg["task"] = args.task
    # Record engine in cfg so downstream reports/XAI can reference it
    cfg["engine"] = args.engine

    seed_everything(cfg.get("seed"))

    label_fn = task_registry.get(args.task)
    engine_run = engine_registry.get(args.engine)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ds_tag = Path(cfg.get("dataset_dir", "raw")).name
    run_dir = Path("results") / "runs" / f"{run_id}_{args.task}_{args.engine}_{ds_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = str(run_dir)

    summary_raw = engine_run(cfg, label_fn)
    ds_dir = cfg.get("materialized_dir") or cfg.get("dataset_dir")
    summary = {
        "run_id": run_id,
        "dataset_dir": ds_dir,
        **summary_raw,
        "hyper": {k: v for k, v in cfg.items() if k not in {"dataset_dir", "run_dir"}},
    }
    write_summary(run_dir, summary, args.task, args.engine)

    # Optionally run XAI analysis on the completed run
    if args.run_xai:
        try:
            print("\n--- Running XAI analysis on completed run ---", flush=True)
            script_path = proj_root / "scripts" / "run_xai_analysis.py"
            cmd = [sys.executable, "-X", "utf8", "-u", str(script_path), "--run-dir", str(run_dir)]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"XAI analysis failed: {e}")

if __name__ == "__main__":
    main()


