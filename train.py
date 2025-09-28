
#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import numpy as np
import csv

# --- project paths / imports -------------------------------------------------
proj_root = Path(__file__).resolve().parent
code_dir = proj_root / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import tasks as task_registry
import engines as engine_registry
from utils.summary import write_summary
from utils.seeding import seed_everything, determinism_banner


"""
CLI for one-off training/evaluation.

Typical uses:
- Single LOSO/GroupKFold run with a base or resolved YAML
- Multi-seed loops (override seed via --set or `seeds: [...]` in YAML)
- Permutation testing (optional) reusing observed splits

Config layering (no sampling here):
  common.yaml → base.yaml (or resolved) → optional overlay --cfg → --set overrides

Flow overview:
1) Parse CLI, build a consolidated cfg via layered YAMLs and --set overrides
2) Seed all RNGs for reproducibility (utils.seeding.seed_everything)
3) Resolve task label function and engine entry point from registries
4) Create a timestamped run_dir under results/runs and persist there
5) Delegate to the engine (dataset/model build + TrainingRunner orchestration)
6) Persist a summary JSON/TXT and environment freeze artifacts; optional post-hoc stats
"""


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train EEGNeX on a given task/engine pair.")
    p.add_argument("--task", required=True)
    p.add_argument("--engine", required=True, choices=list(engine_registry.ENGINES.keys()))
    # Config sources
    p.add_argument("--base", help="Base YAML config file (defaults to configs/tasks/<task>/base.yaml)")
    p.add_argument("--cfg", help="Optional overlay YAML with overrides")
    p.add_argument("--set", nargs="*", metavar="KEY=VAL", help="Override any hyper-parameter")
    p.add_argument("--run-xai", action="store_true")

    # Permutation testing options
    p.add_argument("--permute-labels", action="store_true", help="Run permutation testing instead of observed run")
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--permute-scope", choices=["within_subject", "global"], default=None)
    p.add_argument("--permute-stratified", action="store_true")
    p.add_argument("--permute-seed", type=int, default=None)
    p.add_argument("--observed-run-dir", type=str, default=None, help="Use splits from this observed run for permutations")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Config assembly
# -----------------------------------------------------------------------------
def build_config(args) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    # common.yaml
    common_yaml = Path("configs") / "common.yaml"
    if common_yaml.exists():
        cfg.update(yaml.safe_load(common_yaml.read_text()) or {})

    # base.yaml (task-specific)
    base_yaml = Path(args.base) if args.base else (Path("configs") / "tasks" / args.task / "base.yaml")
    if args.base and not base_yaml.exists():
        sys.exit(f"--base not found: {base_yaml}")
    if base_yaml.exists():
        cfg.update(yaml.safe_load(base_yaml.read_text()) or {})

    # optional overlay
    if args.cfg:
        overlay_yaml = Path(args.cfg)
        if overlay_yaml.exists():
            cfg.update(yaml.safe_load(overlay_yaml.read_text()) or {})

    # --set KEY=VAL overrides
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    base_cfg = build_config(args)
    base_cfg["task"] = args.task
    base_cfg["engine"] = args.engine

    label_fn = task_registry.get(args.task)
    engine_run = engine_registry.get(args.engine)

    # Group all runs under a single timestamp id for this launch
    launch_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Determine base config tag to include in run_dir name
    base_fp = Path(args.base) if args.base else (Path("configs") / "tasks" / args.task / "base.yaml")
    base_tag = base_fp.stem

    # Detect multi-seed mode to avoid directory collisions
    multi_seed_mode = isinstance(base_cfg.get("seeds"), list) and len(base_cfg.get("seeds")) > 1

    # -------------------------------------------------------------------------
    # Per-run executor (single seed)
    # -------------------------------------------------------------------------
    def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
        # Seed per run and log determinism
        seed_everything(cfg.get("seed"))
        det_banner = determinism_banner(cfg.get("seed"))
        print("[determinism]", det_banner, flush=True)

        # Build human-friendly run_dir name: {timestamp}_{task}_{base-stem}
        parts = [launch_run_id, args.task, base_tag]
        # In multi-seed mode, append seed tag to avoid collisions while keeping grouping by launch_run_id
        if multi_seed_mode and cfg.get("seed") is not None:
            try:
                parts.append(f"seed_{int(cfg['seed'])}")
            except Exception:
                parts.append(f"seed_{cfg['seed']}")

        run_dir = Path("results") / "runs" / "_".join(parts)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["run_dir"] = str(run_dir)

        # Delegate to engine (dataset/model build + TrainingRunner orchestration)
        summary_raw = engine_run(cfg, label_fn)

        ds_dir = cfg.get("materialized_dir") or cfg.get("dataset_dir")
        summary = {
            "run_id": launch_run_id,
            "dataset_dir": ds_dir,
            **summary_raw,
            "determinism": det_banner,
            # Persist only run-relevant hyperparameters; omit large/env/path-like keys
            "hyper": {k: v for k, v in cfg.items() if k not in {"dataset_dir", "run_dir"}},
        }
        write_summary(run_dir, summary, args.task, args.engine)

        # Environment freeze artifacts (best-effort)
        try:
            pip_out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT)
            (run_dir / "pip_freeze.txt").write_text(pip_out)
        except Exception as e:
            print(f"[env-freeze] pip freeze failed: {e}")
        try:
            import shutil
            if shutil.which("conda"):
                conda_out = subprocess.check_output(["conda", "env", "export"], text=True, stderr=subprocess.STDOUT)
                (run_dir / "conda_env.yml").write_text(conda_out)
        except Exception as e:
            print(f"[env-freeze] conda env export failed: {e}")

        # Optional: post-hoc stats (driven by common.yaml -> stats.run_posthoc_after_train)
        try:
            stats_cfg = base_cfg.get("stats", {}) if isinstance(base_cfg, dict) else {}
            if bool(stats_cfg.get("run_posthoc_after_train", False)):
                print("\n--- Running post-hoc statistics on completed run ---", flush=True)
                script_path = proj_root / "scripts" / "run_posthoc_stats.py"
                alpha = stats_cfg.get("alpha", 0.05)
                multitest = stats_cfg.get("multitest", "fdr")
                cmd = [
                    sys.executable, "-X", "utf8", "-u", str(script_path),
                    "--run-dir", str(run_dir),
                    "--alpha", str(alpha),
                    "--multitest", str(multitest),
                ]
                if bool(stats_cfg.get("glmm", False)):
                    cmd.append("--glmm")
                # optional forest plot toggle if your script supports it
                if bool(stats_cfg.get("forest", True)):
                    cmd.append("--forest")
                print(f"[posthoc] invoking: {' '.join(map(str, cmd))}")
                subprocess.run(cmd, check=True)
                print("[posthoc] finished successfully.")
        except Exception as e:
            print(f"[posthoc] post-hoc stats failed: {e}")

        return {
            "run_dir": str(run_dir),
            "mean_acc": summary.get("mean_acc", 0.0),
            "macro_f1": summary.get("macro_f1", 0.0),
            "weighted_f1": summary.get("weighted_f1", 0.0),
            "fold_macro_f1s": summary.get("fold_macro_f1s", []),
            "seed": cfg.get("seed"),
        }

    # -------------------------------------------------------------------------
    # Permutation testing path
    # -------------------------------------------------------------------------
    if args.permute_labels:
        if not args.observed_run_dir:
            sys.exit("--observed-run-dir is required when --permute-labels is set")

        obs_dir = Path(args.observed_run_dir)
        splits_fp = obs_dir / "splits_indices.json"
        if not splits_fp.exists():
            sys.exit(f"splits_indices.json not found in observed run dir: {obs_dir}")
        splits = json.loads(splits_fp.read_text())
        predefined_splits: List[dict] = splits.get("outer_folds", [])

        # Determine params (prefer YAML, then CLI, then defaults)
        n_perm = int(base_cfg.get("n_permutations") or args.n_permutations or 0)
        if n_perm <= 0:
            sys.exit("n_permutations must be > 0 for permutation testing")
        scope = (base_cfg.get("permute_scope") or args.permute_scope or "within_subject")
        strat = bool(base_cfg.get("permute_stratified") or args.permute_stratified)
        perm_seed = int(base_cfg.get("permute_seed") or (args.permute_seed if args.permute_seed is not None else 123))
        rng = np.random.RandomState(perm_seed)

        # Load observed summary
        observed_summary_fp = obs_dir / f"summary_{args.task}_{args.engine}.json"
        if not observed_summary_fp.exists():
            sys.exit(f"Observed summary not found: {observed_summary_fp}")
        observed_summary = json.loads(observed_summary_fp.read_text())
        observed_acc = float(observed_summary.get("mean_acc", 0.0))
        observed_macro_f1 = float(observed_summary.get("macro_f1", 0.0))

        # Prepare outputs (next to observed run)
        out_parent = obs_dir.parent
        perm_results_fp = out_parent / f"{obs_dir.name}_perm_test_results.csv"
        perm_summary_fp = out_parent / f"{obs_dir.name}_perm_summary.json"

        with perm_results_fp.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["perm_id", "outer_fold", "acc", "macro_f1", "n_test_trials"]
            )
            writer.writeheader()

        # Imports for dataset/model/runner
        from code.datasets import make_dataset
        from code.training_runner import TrainingRunner
        from code.model_builders import RAW_EEG_MODELS, squeeze_input_adapter, build_raw_eeg_aug

        null_acc: List[float] = []
        null_macro_f1: List[float] = []

        for pid in range(n_perm):
            # Build dataset to obtain groups and base labels
            cfg_perm = dict(base_cfg)
            cfg_perm["run_dir"] = str(out_parent / f"{obs_dir.name}_perm_{pid:04d}")

            dataset = make_dataset(cfg_perm, label_fn)
            groups = dataset.groups
            y = dataset.get_all_labels().copy()

            # Construct permuted labels
            if scope == "within_subject":
                for sid in np.unique(groups):
                    idx = np.where(groups == sid)[0]
                    # within subject permutation already preserves per-subject class counts
                    rng.shuffle(y[idx])
            else:
                if strat:
                    # shuffle indices within each class, then shuffle order of all examples
                    for c in np.unique(y):
                        c_idx = np.where(y == c)[0]
                        rng.shuffle(c_idx)
                    order = np.arange(len(y))
                    rng.shuffle(order)
                    y = y[order]
                else:
                    rng.shuffle(y)

            # Build model + runner
            model_name = cfg_perm.get("model_name", "eegnex")
            model_builder = lambda conf, num_cls: RAW_EEG_MODELS[model_name](
                conf, num_cls, C=dataset.num_channels, T=dataset.time_points
            )
            input_adapter = squeeze_input_adapter if model_name in ("cwat", "eegnex") else None
            aug_builder = lambda conf, d: build_raw_eeg_aug(conf, dataset.time_points)

            runner = TrainingRunner(cfg_perm, label_fn)
            summary_raw = runner.run(
                dataset=dataset,
                groups=dataset.groups,
                class_names=dataset.class_names,
                model_builder=model_builder,
                aug_builder=aug_builder,
                input_adapter=input_adapter,
                optuna_trial=None,
                labels_override=y,
                predefined_splits=predefined_splits,
            )

            # Append per-fold rows from the run's outer_eval_metrics.csv
            try:
                with (Path(cfg_perm["run_dir"]) / "outer_eval_metrics.csv").open("r", newline="") as g:
                    reader = csv.DictReader(g)
                    for row in reader:
                        if str(row["outer_fold"]).upper() == "OVERALL":
                            continue
                        with perm_results_fp.open("a", newline="") as f:
                            writer = csv.DictWriter(
                                f, fieldnames=["perm_id", "outer_fold", "acc", "macro_f1", "n_test_trials"]
                            )
                            writer.writerow({
                                "perm_id": pid,
                                "outer_fold": int(row["outer_fold"]),
                                "acc": float(row["acc"]),
                                "macro_f1": float(row["macro_f1"]),
                                "n_test_trials": int(row["n_test_trials"]),
                            })
                null_acc.append(float(summary_raw.get("mean_acc", 0.0)))
                null_macro_f1.append(float(summary_raw.get("macro_f1", 0.0)))
            except Exception:
                # Fallback to summary only
                null_acc.append(float(summary_raw.get("mean_acc", 0.0)))
                null_macro_f1.append(float(summary_raw.get("macro_f1", 0.0)))

        # Empirical p-values with +1 smoothing
        def p_value(null_vals: List[float], obs: float) -> float:
            m = sum(1 for v in null_vals if v >= obs)
            return float((m + 1) / (len(null_vals) + 1)) if null_vals else 1.0

        perm_summary = {
            "observed_acc": observed_acc,
            "observed_macro_f1": observed_macro_f1,
            "null_mean_acc": float(np.mean(null_acc)) if null_acc else 0.0,
            "null_sd_acc": float(np.std(null_acc)) if null_acc else 0.0,
            "p_value_acc": p_value(null_acc, observed_acc),
            "null_mean_macro_f1": float(np.mean(null_macro_f1)) if null_macro_f1 else 0.0,
            "null_sd_macro_f1": float(np.std(null_macro_f1)) if null_macro_f1 else 0.0,
            "p_value_macro_f1": p_value(null_macro_f1, observed_macro_f1),
            "n_permutations": n_perm,
            "permute_scope": scope,
            "permute_stratified": strat,
            "permute_seed": perm_seed,
            "observed_run_dir": str(obs_dir),
            "splits_indices": str(splits_fp),
        }
        perm_summary_fp.write_text(json.dumps(perm_summary, indent=2))
        print("[permute] done.")
        return

    # -------------------------------------------------------------------------
    # Multi-seed execution (confirmatory runs)
    # -------------------------------------------------------------------------
    seeds = base_cfg.get("seeds") if isinstance(base_cfg.get("seeds"), list) else None
    results: List[Dict[str, Any]] = []

    if not args.permute_labels and seeds and len(seeds) > 1:
        for s in seeds:
            cfg = dict(base_cfg)
            try:
                cfg["seed"] = int(s)
            except Exception:
                cfg["seed"] = s
            results.append(run_one(cfg))

        # Write aggregate JSON in results/runs
        try:
            out_dir = Path("results") / "runs"
            agg = {
                "task": args.task,
                "engine": args.engine,
                "run_id": launch_run_id,
                "seeds": [r.get("seed") for r in results],
                "metrics": results,
                "mean_of_means_acc": float(sum(r["mean_acc"] for r in results) / max(1, len(results))),
                "mean_of_macro_f1": float(sum(r["macro_f1"] for r in results) / max(1, len(results))),
                "mean_of_weighted_f1": float(sum(r["weighted_f1"] for r in results) / max(1, len(results))),
            }
            (out_dir / f"{launch_run_id}_{args.task}_{args.engine}_seeds_aggregate.json").write_text(
                json.dumps(agg, indent=2)
            )
            print("[multi-seed] aggregate written.")
        except Exception as e:
            print(f"[multi-seed] aggregate write failed: {e}")
    elif not args.permute_labels:
        # Single run (existing behavior)
        run_one(base_cfg)
    # else: permutation path already handled above


if __name__ == "__main__":
    main()

