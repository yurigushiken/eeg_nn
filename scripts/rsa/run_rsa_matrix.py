"""
Automate RSA binary training runs across cardinality condition pairs.

Supports resumable execution via --resume. State is recorded in
.output directory allowing interrupted runs to continue without
rerunning completed seeds/pairs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, Optional, Set

import pandas as pd
import yaml

PROJ_ROOT = Path(__file__).resolve().parents[2]
code_root = PROJ_ROOT / "code"
for path in (PROJ_ROOT, code_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from engines import get as get_engine
from tasks import get as get_task
from tasks import rsa_binary
from scripts.rsa.visualize_rsa import (
    build_accuracy_matrix,
    compute_mds_positions,
    plot_rdm_heatmap,
    plot_mds_scatter,
)
from scripts.rsa.naming import analysis_id_from_run_root, prefixed_path, prefixed_title
try:
    from utils.seeding import determinism_banner, seed_everything
except ModuleNotFoundError:
    from importlib.util import module_from_spec, spec_from_file_location

    _PROJ_ROOT = Path(__file__).resolve().parents[2]
    _seed_module_path = _PROJ_ROOT / "utils" / "seeding.py"
    _spec = spec_from_file_location("_rsa_seeding", str(_seed_module_path))
    if _spec and _spec.loader:
        _module = module_from_spec(_spec)
        _spec.loader.exec_module(_module)  # type: ignore[attr-defined]
        determinism_banner = getattr(_module, "determinism_banner")
        seed_everything = getattr(_module, "seed_everything")
    else:
        raise

CARDINALITY_CODES = [11, 22, 33, 44, 55, 66]
LANDING_CODES = [1, 2, 3, 4, 5, 6]
RESULTS_FILENAME = "rsa_matrix_results.csv"
STATE_FILENAME = ".rsa_resume_state.json"


def generate_cross_digit_pairs(codes: Sequence[int]) -> List[Tuple[int, int]]:
    unique_codes = sorted({int(c) for c in codes})
    return [(a, b) for a, b in itertools.combinations(unique_codes, 2)]


def aggregate_outer_metrics(run_dir: Path, class_a: int, class_b: int, seed: int) -> Iterator[Dict[str, float]]:
    metrics_path = Path(run_dir) / "outer_eval_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"outer_eval_metrics.csv not found in {run_dir}")

    with metrics_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("outer_fold", "")).strip().upper() != "OVERALL":
                continue
            yield {
                "ClassA": class_a,
                "ClassB": class_b,
                "Seed": seed,
                "Accuracy": float(row["acc"]),
                "MacroF1": float(row["macro_f1"]),
                "MinClassF1": float(row["min_per_class_f1"]),
            }
            return

    raise ValueError(f"OVERALL row not found in {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSA binary matrix training.")
    parser.add_argument("--config", default="configs/tasks/rsa/rsa_binary.yaml")
    parser.add_argument(
        "--output-dir",
        default="results/runs",
        help="Directory where run subdirectories and the summary CSV will be written.",
    )
    parser.add_argument("--task", default="rsa_binary")
    parser.add_argument("--engine", default="eeg")
    parser.add_argument("--conditions", nargs="*", type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-output-dir", type=Path, default=None)
    parser.add_argument("--viz-prefix", default=None)
    parser.add_argument("--viz-metric", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run in output directory.")
    parser.add_argument("--resume-launch-id", default=None, help="Explicit launch id to resume when multiple runs exist.")
    return parser.parse_args()


def load_base_config(cfg_path: Path) -> Dict:
    cfg: Dict = {}
    common_path = Path("configs") / "common.yaml"
    if common_path.exists():
        cfg.update(yaml.safe_load(common_path.read_text(encoding="utf-8")) or {})

    cfg.update(yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})
    return cfg


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_summary_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    header = ["ClassA", "ClassB", "Seed", "Accuracy", "MacroF1", "MinClassF1"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def state_path(output_root: Path) -> Path:
    return output_root / STATE_FILENAME


def load_state(state_fp: Path) -> Optional[Dict]:
    if state_fp.exists():
        return json.loads(state_fp.read_text())
    return None


def atomic_write_state(state_fp: Path, data: Dict) -> None:
    tmp_fp = state_fp.with_suffix(".tmp")
    tmp_fp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp_fp.replace(state_fp)


def infer_launch_state(
    output_root: Path,
    pairs: List[Tuple[int, int]],
    seeds: Iterable[int],
    requested_launch_id: Optional[str],
) -> Tuple[str, Set[Tuple[int, int, int]], List[Dict[str, int]]]:
    run_pattern = "rsa_{a}v{b}_seed_{seed}"
    candidates: Dict[str, List[Tuple[int, int, int]]] = {}

    for entry in output_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if "_rsa_" not in name:
            continue
        try:
            prefix, rest = name.split("_rsa_", 1)
            pair_part, seed_part = rest.rsplit("_seed_", 1)
            a_str, b_str = pair_part.split("v")
            seed_val = int(seed_part)
            a_val = int(a_str)
            b_val = int(b_str)
        except ValueError:
            continue
        key = (a_val, b_val)
        if key not in pairs:
            continue
        if seed_val not in seeds:
            continue
        candidates.setdefault(prefix, []).append((a_val, b_val, seed_val))

    if not candidates:
        raise RuntimeError("Unable to infer existing run directories for resume.")

    launch_id = requested_launch_id or max(candidates.keys())
    if requested_launch_id and requested_launch_id not in candidates:
        raise RuntimeError(f"Requested launch_id '{requested_launch_id}' not found under {output_root}")

    triples = candidates.get(launch_id, [])
    completed = set()
    for a, b, seed in triples:
        run_dir = output_root / f"{launch_id}_rsa_{a}v{b}_seed_{seed}"
        metrics_fp = run_dir / "outer_eval_metrics.csv"
        if metrics_fp.exists():
            try:
                next(aggregate_outer_metrics(run_dir, a, b, seed))
            except Exception:
                continue
            else:
                completed.add((a, b, seed))
    completed_list = [{"ClassA": a, "ClassB": b, "Seed": seed} for (a, b, seed) in sorted(completed)]
    return launch_id, completed, completed_list


def build_initial_state(
    args,
    output_root: Path,
    pairs: List[Tuple[int, int]],
    seeds: Iterable[int],
) -> Dict:
    state_fp = state_path(output_root)
    existing = load_state(state_fp)
    if args.resume:
        if existing:
            return existing
        launch_id, completed_set, completed_list = infer_launch_state(output_root, pairs, seeds, args.resume_launch_id)
        state = {
            "launch_id": launch_id,
            "pairs": pairs,
            "seeds": list(seeds),
            "completed": completed_list,
            "finished": False,
        }
        atomic_write_state(state_fp, state)
        return state
    else:
        if existing:
            raise RuntimeError("State file already exists. Use --resume to continue or remove the existing state file to start over.")
        launch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        state = {
            "launch_id": launch_id,
            "pairs": pairs,
            "seeds": list(seeds),
            "completed": [],
            "finished": False,
        }
        atomic_write_state(state_fp, state)
        return state


def validate_state(state: Dict, pairs: List[Tuple[int, int]], seeds: Iterable[int]) -> None:
    state_pairs = [tuple(p) for p in state.get("pairs", [])]
    if sorted(state_pairs) != sorted(pairs):
        raise RuntimeError("Resume state pairs do not match current configuration.")
    state_seeds = sorted(state.get("seeds", []))
    if sorted(seeds) != state_seeds:
        raise RuntimeError("Resume state seeds do not match current configuration.")


def tuple_key(entry: Dict[str, int]) -> Tuple[int, int, int]:
    return (int(entry["ClassA"]), int(entry["ClassB"]), int(entry["Seed"]))


def _validate_pair_switch(*, task_name: str, label_fn, expected_pair: Tuple[int, int]) -> None:
    """
    Fail-fast guardrail: ensure the active-pair switch actually affects the label function.

    This prevents wasting compute if a task's global active pair is not being applied
    (e.g., silently decoding the default pair while naming the run directory differently).
    """
    task = str(task_name)
    a, b = int(expected_pair[0]), int(expected_pair[1])

    # For landing-digit decoding, the label_fn expects two-digit codes where source != landing.
    if task.endswith("rsa_landing_binary"):
        meta = pd.DataFrame({"Condition": [(a * 10 + b), (b * 10 + a), (a * 10 + a)]})
        expected_labels = sorted([a, b])
    else:
        # rsa_binary: decode by exact condition code. Works for 11/22/... and 2-digit change codes.
        meta = pd.DataFrame({"Condition": [a, b, a, b]})
        expected_labels = sorted([a, b])

    labels = label_fn(meta).dropna().unique().tolist()
    actual_labels: List[int] = []
    for x in labels:
        try:
            actual_labels.append(int(x))
        except Exception:
            # If labels are non-int strings, treat as mismatch.
            actual_labels.append(-999999)
    actual_sorted = sorted(actual_labels)
    if actual_sorted != expected_labels:
        raise RuntimeError(
            f"Active-pair validation failed for task '{task_name}' with pair {expected_pair}. "
            f"Expected labels {expected_labels} but label_fn produced {actual_sorted}. "
            "This would cause runs to decode the wrong pair and waste compute."
        )


def main() -> None:
    args = parse_args()
    base_cfg = load_base_config(Path(args.config))

    config_visualize = bool(base_cfg.get("rsa_visualize", False))
    config_viz_metric = base_cfg.get("rsa_visualize_metric", "Accuracy")

    seeds: Iterable[int]
    if isinstance(base_cfg.get("seeds"), list) and base_cfg["seeds"]:
        seeds = [int(s) for s in base_cfg["seeds"]]
    elif "seed" in base_cfg:
        seeds = [int(base_cfg["seed"])]
    else:
        seeds = [42]

    base_cfg.pop("seeds", None)

    engine_run = get_engine(args.engine)
    label_fn = get_task(args.task)
    task_module = sys.modules[label_fn.__module__]

    output_root = Path(args.output_dir)
    ensure_directory(output_root)
    analysis_id = analysis_id_from_run_root(output_root)

    # Choose condition codes: CLI overrides; else config override; else task-aware default.
    codes = (
        args.conditions
        if args.conditions
        else base_cfg.get("rsa_conditions")
        if base_cfg.get("rsa_conditions")
        else (LANDING_CODES if args.task == "rsa_landing_binary" else CARDINALITY_CODES)
    )
    pairs = generate_cross_digit_pairs(codes)

    state = build_initial_state(args, output_root, pairs, seeds)
    validate_state(state, pairs, seeds)
    launch_id = state["launch_id"]

    completed_records = state.get("completed", [])
    completed_keys: Set[Tuple[int, int, int]] = {tuple_key(entry) for entry in completed_records}

    rows: List[Dict[str, float]] = []
    to_remove = set()
    for key in completed_keys:
        a, b, seed = key
        run_dir = output_root / f"{launch_id}_rsa_{a}v{b}_seed_{seed}"
        try:
            rows.extend(aggregate_outer_metrics(run_dir, a, b, seed))
        except Exception:
            to_remove.add(key)

    if to_remove:
        completed_keys -= to_remove
        state["completed"] = [
            {"ClassA": a, "ClassB": b, "Seed": seed}
            for (a, b, seed) in sorted(completed_keys)
        ]
        atomic_write_state(state_path(output_root), state)

    for seed in seeds:
        for class_a, class_b in pairs:
            key = (class_a, class_b, int(seed))
            if key in completed_keys:
                continue

            # Set active pair on the task module when available; fall back to rsa_binary
            set_pair = getattr(task_module, "set_active_pair", None)
            if callable(set_pair):
                set_pair((class_a, class_b))
            else:
                rsa_binary.set_active_pair((class_a, class_b))
            _validate_pair_switch(task_name=str(args.task), label_fn=label_fn, expected_pair=(class_a, class_b))
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = int(seed)
            cfg["task"] = args.task
            cfg["engine"] = args.engine

            run_dir = output_root / f"{launch_id}_rsa_{class_a}v{class_b}_seed_{seed}"
            ensure_directory(run_dir)
            cfg["run_dir"] = str(run_dir)

            seed_everything(cfg["seed"])
            print(f"[rsa-matrix] {determinism_banner(cfg['seed'])}")
            print(f"[rsa-matrix] Training pair {class_a} vs {class_b} (seed={seed})", flush=True)

            engine_run(cfg, label_fn)

            metrics = list(aggregate_outer_metrics(run_dir, class_a, class_b, seed))
            rows.extend(metrics)

            completed_keys.add(key)
            state["completed"] = [
                {"ClassA": a, "ClassB": b, "Seed": s}
                for (a, b, s) in sorted(completed_keys)
            ]
            atomic_write_state(state_path(output_root), state)

    # Reset active pair on the task module when available; fall back to rsa_binary
    reset_pair = getattr(task_module, "reset_active_pair", None)
    if callable(reset_pair):
        reset_pair()
    else:
        rsa_binary.reset_active_pair()

    summary_path = output_root / RESULTS_FILENAME
    write_summary_csv(rows, summary_path)
    print(f"[rsa-matrix] Summary written to {summary_path.resolve()}", flush=True)

    state["finished"] = True
    atomic_write_state(state_path(output_root), state)

    visualize = args.visualize or config_visualize
    viz_metric = args.viz_metric if args.viz_metric else config_viz_metric

    if visualize:
        print("[rsa-matrix] Generating visualization assets...", flush=True)
        matrix, labels = build_accuracy_matrix(summary_path, metric=viz_metric, subject_filter="OVERALL")
        positions = compute_mds_positions(matrix, labels)
        heatmap_path = prefixed_path(run_root=output_root, kind="figures", stem="brain_rdm_heatmap", ext=".png")
        scatter_path = prefixed_path(run_root=output_root, kind="figures", stem="mds", ext=".png")
        plot_rdm_heatmap(
            matrix,
            labels,
            heatmap_path,
            title=prefixed_title(run_root=output_root, title="RSA Matrix (Higher = Easier to Distinguish)"),
        )
        plot_mds_scatter(
            positions,
            scatter_path,
            flip_xy=True,
            title=prefixed_title(run_root=output_root, title="MDS Projection of RSA Matrix"),
        )
        print(f"[rsa-matrix] Heatmap saved to {heatmap_path.resolve()}", flush=True)
        print(f"[rsa-matrix] MDS scatter saved to {scatter_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

