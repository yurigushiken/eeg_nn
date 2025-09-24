#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import yaml
import datetime
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


def load_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description="Final evaluation with multi-seed and optional XAI")
    ap.add_argument("--task", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--use-best", action='store_true', help="merge best step1/step2/finalist params if present")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--run-xai", action='store_true')
    args = ap.parse_args()

    common = load_yaml(proj_root / "configs" / "common.yaml")
    base = load_yaml(Path(args.cfg))
    cfg_base = dict(common); cfg_base.update(base)

    # Merge best params if requested
    if args.use_best:
        # Look in multiple canonical locations for best.json
        candidates = []
        for stage in ("step1", "step2", "step3", "finalist"):
            candidates.append(proj_root / "results" / "optuna_studies" / args.task / stage / "best.json")
        optuna_root = proj_root / "results" / "optuna"
        if optuna_root.exists():
            for p in optuna_root.glob("*_%s/best.json" % args.task):
                candidates.append(p)
        for p in candidates:
            if p.exists():
                try:
                    best = json.loads(p.read_text())
                    cfg_base.update(best.get("params", {}))
                except Exception:
                    pass

    label_fn = task_registry.get(args.task)
    engine_run = engine_registry.get("eeg")

    all_accs = []
    all_summaries = []
    ts_root = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    # strict behavior alignment is always enforced by default now (no flag needed)
    for i in range(args.seeds):
        cfg = dict(cfg_base)
        cfg["seed"] = int(cfg_base.get("seed", 42)) + i
        seed_everything(cfg["seed"])
        run_dir = proj_root / "results" / "runs" / f"{ts_root}_{args.task}_eeg_seed{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["run_dir"] = str(run_dir)
        summary_raw = engine_run(cfg, label_fn)
        summary = {"run_id": ts_root, "dataset_dir": cfg.get("materialized_dir") or cfg.get("data_raw_dir"), **summary_raw, "hyper": {k:v for k,v in cfg.items() if k not in {"run_dir"}}}
        write_summary(run_dir, summary, args.task, "eeg")
        all_summaries.append(summary)
        all_accs.append(summary.get("mean_acc", 0.0))

    # Aggregate
    agg = {
        "seeds": args.seeds,
        "seed_mean_acc": float(sum(all_accs)/len(all_accs)) if all_accs else 0.0,
        "seed_std_acc": float((sum((a - (sum(all_accs)/len(all_accs)))**2 for a in all_accs)/len(all_accs))**0.5) if all_accs else 0.0,
    }
    out_dir = proj_root / "results" / "runs" / f"{ts_root}_{args.task}_eeg_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))

    if args.run_xai:
        # Optionally run XAI per-seed run_dir; here we just note it; can be expanded
        pass

if __name__ == "__main__":
    main()


