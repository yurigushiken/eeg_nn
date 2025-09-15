#!/usr/bin/env python
from __future__ import annotations
import argparse
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

from code.datasets import OnTheFlyPreprocDataset
import tasks as task_registry

def main():
    ap = argparse.ArgumentParser(description="Materialize .fif using best Step1 hyperparameters")
    ap.add_argument("--task", required=True)
    ap.add_argument("--best", required=True, help="results/optuna_studies/<task>/step1/best.json")
    ap.add_argument("--out", required=True, help="output folder under data_preprocessed/")
    ap.add_argument("--usable-trials-csv", type=str, default=None, help="Path to QC CSV with Kept_Segs_Indxs")
    ap.add_argument("--strict", action='store_true', help="enforce strict behavior alignment")
    ap.add_argument("--subset-n", type=int, default=1, help="limit # of subjects for quick debug")
    args = ap.parse_args()

    best = json.loads(Path(args.best).read_text())
    params = best["params"]
    common = yaml.safe_load((proj_root / "configs" / "common.yaml").read_text())
    base = yaml.safe_load((proj_root / "configs" / "tasks" / args.task / "base.yaml").read_text())
    cfg = dict(common); cfg.update(base); cfg.update(params)
    # Ensure no augmentation keys here
    for k in ("mixup_alpha","shift_p","scale_p","noise_p","time_mask_p","chan_mask_p"):
        cfg[k] = 0.0
    # Force full subject set
    cfg["subset_n"] = int(args.subset_n)
    if args.usable_trials_csv:
        cfg["usable_trials_csv"] = args.usable_trials_csv
    if args.strict:
        cfg["strict_behavior_align"] = True

    label_fn = task_registry.get(args.task)
    # Build dataset which will compute and cache epochs using Step1 best
    ds = OnTheFlyPreprocDataset(cfg, label_fn)
    # Now write out the cached .fif files to a stable folder
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate cache root fingerprint
    from code.datasets import _cfg_fingerprint
    cache_root = proj_root / "cache" / "epochs" / _cfg_fingerprint(cfg)
    for fp in cache_root.glob("sub-*preprocessed-epo.fif"):
        target = out_dir / fp.name
        if target.resolve() != fp.resolve():
            target.write_bytes(fp.read_bytes())
    print(f"Materialized to: {out_dir}")

if __name__ == "__main__":
    main()


