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

"""
CLI for one-off training/evaluation.

Typical uses:
- Single LOSO/GroupKFold run with a base or resolved YAML
- Multi-seed loops (override seed via --set)

Config layering (no sampling here):
  common.yaml → base.yaml (or resolved) → optional overlay --cfg → --set overrides

Flow overview:
1) Parse CLI, build a consolidated cfg via layered YAMLs and --set overrides
2) Seed all RNGs for reproducibility (utils.seeding.seed_everything)
3) Resolve task label function and engine entry point from registries
4) Create a timestamped run_dir under results/runs and persist there
5) Delegate to the engine (dataset/model build + TrainingRunner orchestration)
6) Persist a summary JSON/TXT/YAML and optional XAI analysis artifacts
"""

def parse_args():
    p = argparse.ArgumentParser(description="Train EEGNeX on a given task/engine pair.")
    p.add_argument("--task", required=True)
    p.add_argument("--engine", required=True, choices=list(engine_registry.ENGINES.keys()))
    # Support both styles: --base (preferred) and optional overlay --cfg
    #   - --base points to the baseline/resolved YAML for this run
    #   - --cfg is a light overlay with minor tweaks (e.g., controller knobs)
    p.add_argument("--base", help="Base YAML config file (defaults to configs/tasks/<task>/base.yaml)")
    p.add_argument("--cfg", help="Optional overlay YAML with overrides (e.g., winners or controller knobs)")
    p.add_argument("--set", nargs="*", metavar="KEY=VAL", help="Override any hyper-parameter.")
    p.add_argument("--run-xai", action='store_true')
    return p.parse_args()

def build_config(args) -> Dict[str, Any]:
    common_yaml = Path("configs") / "common.yaml"
    cfg = yaml.safe_load(common_yaml.read_text()) if common_yaml.exists() else {}
    # Precedence: start from common, then update with base, optional overlay, then CLI --set

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
    # Prefer materialized_dir for dataset tag; omit tag if not set (avoid default 'raw')
    materialized = cfg.get("materialized_dir") or cfg.get("dataset_dir")
    ds_tag = Path(materialized).name if materialized else None
    # Append crop_ms to run directory name for traceability when provided
    crop = cfg.get("crop_ms")
    crop_tag = None
    if isinstance(crop, (list, tuple)) and len(crop) == 2:
        try:
            a, b = int(crop[0]), int(crop[1])
            crop_tag = f"crop_ms_{a}_{b}"
        except Exception:
            pass
    parts = [run_id, args.task, args.engine]
    if ds_tag:
        parts.append(ds_tag)
    if crop_tag:
        parts.append(crop_tag)
    run_dir = Path("results") / "runs" / ("_".join(parts))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = str(run_dir)

    # Delegate to engine (which builds dataset/model and runs TrainingRunner)
    summary_raw = engine_run(cfg, label_fn)
    ds_dir = cfg.get("materialized_dir") or cfg.get("dataset_dir")
    summary = {
        "run_id": run_id,
        "dataset_dir": ds_dir,
        **summary_raw,
        # Persist only run-relevant hyperparameters; omit large/env/path-like keys
        "hyper": {k: v for k, v in cfg.items() if k not in {"dataset_dir", "run_dir"}},
    }
    write_summary(run_dir, summary, args.task, args.engine)

    # Optionally run XAI analysis on the completed run (post-hoc; may require playwright for PDF)
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


