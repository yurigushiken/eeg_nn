"""
Time-resolved RSA: Sliding window temporal decoding.

This script extends run_rsa_matrix.py with a temporal dimension, training
models on narrow time windows to track the emergence of numerosity
representations over the course of the EEG epoch.

Methodology:
- Window size: 50ms (≈12-13 samples at 250Hz)
- Stride: 20ms (≈5 samples, 75% overlap)
- Result: 23 time points per 500ms epoch

Outputs:
- rsa_temporal_results.csv with TimeWindow_Start/End/Center columns
- Time-resolved visualizations showing accuracy(t) for each pair
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

import yaml

# This script is in scripts/rsa/, so go up 2 levels to reach project root
PROJ_ROOT = Path(__file__).resolve().parents[2]
code_root = PROJ_ROOT / "code"
for path in (PROJ_ROOT, code_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from engines import get as get_engine
from tasks import get as get_task
from tasks import rsa_binary
from tasks import rsa_landing_binary

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
RESULTS_FILENAME = "rsa_temporal_results.csv"
STATE_FILENAME = ".rsa_temporal_resume_state.json"


def resolve_condition_codes(*, args_conditions: List[int] | None, cfg: Dict) -> List[int]:
    """
    Resolve condition codes explicitly.

    Precedence:
      1) CLI --conditions (most explicit at runtime)
      2) YAML cfg['rsa_conditions'] (explicit, versionable)
      3) Default CARDINALITY_CODES (backwards-compatible)
    """
    if args_conditions:
        return [int(x) for x in args_conditions]
    cfg_codes = cfg.get("rsa_conditions")
    if isinstance(cfg_codes, list) and cfg_codes:
        return [int(x) for x in cfg_codes]
    return [int(x) for x in CARDINALITY_CODES]


def set_active_pair_for_task(*, task_name: str, pair: Tuple[int, int]) -> None:
    """
    Set the active pair on the appropriate task module.

    This is critical for tasks like `rsa_landing_binary` where the label function depends
    on a task-specific global active pair.
    """
    # Accept both "rsa_landing_binary" and "tasks.rsa_landing_binary" style strings.
    norm = str(task_name).split(".")[-1]
    if norm == "rsa_landing_binary":
        rsa_landing_binary.set_active_pair(pair)
        return
    rsa_binary.set_active_pair(pair)


def _validate_pair_switch(*, task_name: str, label_fn, pair: Tuple[int, int]) -> None:
    """
    Ultra-cheap sanity check to prevent wasting compute:
    ensure the provided label_fn actually reflects the requested active pair.

    This catches cases where the active-pair setter is not applied (or applied to the
    wrong module) and all runs silently decode the same default pair.
    """
    norm = str(task_name).split(".")[-1]
    a, b = int(pair[0]), int(pair[1])

    import pandas as pd

    if norm == "rsa_landing_binary":
        # Construct change trials with landing digits a/b (source != landing).
        # Pick sources that differ from landing to avoid being filtered out.
        sa = 2 if a != 2 else 3
        sb = 2 if b != 2 else 3
        meta = pd.DataFrame({"Condition": [sa * 10 + a, sb * 10 + b, sb * 10 + a, sa * 10 + b]})
    else:
        meta = pd.DataFrame({"Condition": [a, b, a, b]})

    labs = set(label_fn(meta).dropna().astype(str).unique().tolist())
    expected = {str(a), str(b)}
    if labs != expected:
        raise RuntimeError(
            f"[temporal-rsa] Active-pair validation failed for task={task_name!r} pair={pair}. "
            f"Label function produced labels {sorted(labs)} but expected {sorted(expected)}. "
            "This would cause many runs to decode the wrong pair and waste compute."
        )


def generate_temporal_windows(
    epoch_ms: float = 500,
    window_ms: float = 50,
    stride_ms: float = 20
) -> List[Tuple[int, int]]:
    """
    Generate sliding time windows for temporal RSA.

    Args:
        epoch_ms: Total epoch duration in milliseconds
        window_ms: Window duration in milliseconds
        stride_ms: Stride between windows in milliseconds

    Returns:
        List of (start_ms, end_ms) tuples
    """
    windows = []
    start = 0
    while start + window_ms <= epoch_ms:
        windows.append((int(start), int(start + window_ms)))
        start += stride_ms
    return windows


def format_window_name(start_ms: int, end_ms: int) -> str:
    """Format window as 't0-50ms' style string."""
    return f"t{start_ms}-{end_ms}ms"


def apply_temporal_window_to_config(
    base_cfg: Dict,
    window_start: int,
    window_end: int
) -> Dict:
    """
    Create a copy of config with crop_ms set to the temporal window.

    Args:
        base_cfg: Base configuration dictionary
        window_start: Window start time in ms
        window_end: Window end time in ms

    Returns:
        New config with crop_ms=[window_start, window_end]
    """
    cfg = copy.deepcopy(base_cfg)
    cfg['crop_ms'] = [window_start, window_end]
    return cfg


def ms_to_samples(ms: float, sfreq: float = 250.0) -> int:
    """Convert milliseconds to samples."""
    return int(ms * sfreq / 1000.0)


def samples_to_ms(samples: int, sfreq: float = 250.0) -> float:
    """Convert samples to milliseconds."""
    return float(samples * 1000.0 / sfreq)


def get_temporal_csv_header() -> List[str]:
    """Return CSV header for temporal results."""
    return [
        "ClassA",
        "ClassB",
        "Seed",
        "TimeWindow_Start",
        "TimeWindow_End",
        "TimeWindow_Center",
        "Accuracy",
        "MacroF1",
        "MinClassF1",
    ]


def aggregate_temporal_metrics(
    run_dir: Path,
    class_a: int,
    class_b: int,
    seed: int,
    window_start: int,
    window_end: int
) -> Iterator[Dict]:
    """
    Aggregate outer metrics with temporal metadata.

    Args:
        run_dir: Directory containing outer_eval_metrics.csv
        class_a: First class code
        class_b: Second class code
        seed: Random seed
        window_start: Window start time in ms
        window_end: Window end time in ms

    Yields:
        Dictionary with metrics and temporal metadata
    """
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
                "TimeWindow_Start": window_start,
                "TimeWindow_End": window_end,
                "TimeWindow_Center": (window_start + window_end) // 2,
                "Accuracy": float(row["acc"]),
                "MacroF1": float(row["macro_f1"]),
                "MinClassF1": float(row["min_per_class_f1"]),
            }
            return

    raise ValueError(f"OVERALL row not found in {metrics_path}")


def format_temporal_run_dir(
    launch_id: str,
    class_a: int,
    class_b: int,
    seed: int,
    window_start: int,
    window_end: int
) -> str:
    """Format run directory name with temporal window."""
    window_name = format_window_name(window_start, window_end)
    return f"{launch_id}_rsa_{class_a}v{class_b}_seed_{seed}_{window_name}"


def generate_cross_digit_pairs(codes: Sequence[int]) -> List[Tuple[int, int]]:
    """Generate all pairwise combinations of condition codes."""
    unique_codes = sorted({int(c) for c in codes})
    return [(a, b) for a, b in itertools.combinations(unique_codes, 2)]


def build_temporal_state(
    launch_id: str,
    pairs: List[Tuple[int, int]],
    seeds: List[int],
    windows: List[Tuple[int, int]],
    completed: List[Dict]
) -> Dict:
    """Build state dictionary for temporal RSA."""
    return {
        "launch_id": launch_id,
        "pairs": pairs,
        "seeds": seeds,
        "windows": windows,
        "total_runs": len(pairs) * len(seeds) * len(windows),
        "completed": completed,
        "finished": False,
    }


def is_run_completed(
    state: Dict,
    class_a: int,
    class_b: int,
    seed: int,
    window_start: int,
    window_end: int
) -> bool:
    """Check if a specific run is already completed."""
    for entry in state.get("completed", []):
        if (
            entry.get("ClassA") == class_a
            and entry.get("ClassB") == class_b
            and entry.get("Seed") == seed
            and entry.get("TimeWindow_Start") == window_start
            and entry.get("TimeWindow_End") == window_end
        ):
            return True
    return False


def mark_run_completed(
    state: Dict,
    class_a: int,
    class_b: int,
    seed: int,
    window_start: int,
    window_end: int
) -> Dict:
    """Mark a run as completed in state."""
    completed = state.get("completed", [])
    completed.append({
        "ClassA": class_a,
        "ClassB": class_b,
        "Seed": seed,
        "TimeWindow_Start": window_start,
        "TimeWindow_End": window_end,
    })
    state["completed"] = completed
    return state


def sort_temporal_results(results: List[Dict]) -> List[Dict]:
    """Sort results by time window, then pair, then seed."""
    return sorted(
        results,
        key=lambda x: (
            x["TimeWindow_Start"],
            x["ClassA"],
            x["ClassB"],
            x["Seed"]
        )
    )


def state_path(output_root: Path) -> Path:
    """Return path to state file."""
    return output_root / STATE_FILENAME


def load_state(state_fp: Path) -> Optional[Dict]:
    """Load state from JSON file."""
    if state_fp.exists():
        return json.loads(state_fp.read_text())
    return None


def atomic_write_state(state_fp: Path, data: Dict) -> None:
    """Atomically write state to file."""
    tmp_fp = state_fp.with_suffix(".tmp")
    tmp_fp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp_fp.replace(state_fp)


def write_temporal_csv(rows: List[Dict], output_path: Path) -> None:
    """Write temporal results to CSV."""
    header = get_temporal_csv_header()
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run time-resolved RSA with sliding windows.")
    parser.add_argument("--config", default="configs/tasks/rsa_temporal.yaml")
    parser.add_argument(
        "--output-dir",
        default="results/runs/rsa_temporal_v1",
        help="Directory for run outputs and summary CSV.",
    )
    parser.add_argument("--task", default="rsa_binary")
    parser.add_argument("--engine", default="eeg")
    parser.add_argument("--conditions", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run.")
    parser.add_argument("--resume-launch-id", default=None, help="Explicit launch ID to resume.")
    return parser.parse_args()


def load_base_config(cfg_path: Path) -> Dict:
    """Load configuration from YAML file."""
    cfg: Dict = {}
    common_path = Path("configs") / "common.yaml"
    if common_path.exists():
        cfg.update(yaml.safe_load(common_path.read_text(encoding="utf-8")) or {})

    cfg.update(yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})
    return cfg


def main() -> None:
    """Main execution function."""
    args = parse_args()
    base_cfg = load_base_config(Path(args.config))

    # Extract temporal parameters
    temporal_cfg = base_cfg.get("temporal", {})
    epoch_ms = float(temporal_cfg.get("epoch_ms", 500))
    window_ms = float(temporal_cfg.get("window_ms", 50))
    stride_ms = float(temporal_cfg.get("stride_ms", 20))

    # Generate temporal windows
    # Optional: allow a restricted set of training windows for pilots (while keeping the
    # canonical epoch/window/stride definition in config for downstream tooling).
    train_windows_cfg = temporal_cfg.get("train_windows")
    if isinstance(train_windows_cfg, list) and train_windows_cfg:
        windows = [(int(w[0]), int(w[1])) for w in train_windows_cfg]
        print(f"[temporal-rsa] Using {len(windows)} explicit train_windows from config.")
    else:
        windows = generate_temporal_windows(epoch_ms, window_ms, stride_ms)
    print(f"[temporal-rsa] Generated {len(windows)} temporal windows:")
    print(f"[temporal-rsa]   Epoch: {epoch_ms}ms, Window: {window_ms}ms, Stride: {stride_ms}ms")
    print(f"[temporal-rsa]   Windows: {windows[0]} ... {windows[-1]}")

    # Extract seeds
    seeds: List[int]
    if isinstance(base_cfg.get("seeds"), list) and base_cfg["seeds"]:
        seeds = [int(s) for s in base_cfg["seeds"]]
    elif "seed" in base_cfg:
        seeds = [int(base_cfg["seed"])]
    else:
        seeds = [42]

    base_cfg.pop("seeds", None)

    # Get engine and task
    engine_run = get_engine(args.engine)
    label_fn = get_task(args.task)

    # Setup output directory
    output_root = Path(args.output_dir)
    ensure_directory(output_root)

    # Generate pairs (explicit precedence: CLI > YAML > default)
    codes = resolve_condition_codes(args_conditions=args.conditions, cfg=base_cfg)
    pairs = generate_cross_digit_pairs(codes)

    print(f"[temporal-rsa] Training configuration:")
    print(f"[temporal-rsa]   Pairs: {len(pairs)} ({pairs[0]} ... {pairs[-1]})")
    print(f"[temporal-rsa]   Seeds: {seeds}")
    print(f"[temporal-rsa]   Total runs: {len(pairs)} × {len(seeds)} × {len(windows)} = {len(pairs) * len(seeds) * len(windows)}")

    # Initialize state
    launch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_fp = state_path(output_root)

    if args.resume:
        existing_state = load_state(state_fp)
        if existing_state:
            state = existing_state
            launch_id = state["launch_id"]
            print(f"[temporal-rsa] Resuming launch: {launch_id}")
            print(f"[temporal-rsa] Completed: {len(state['completed'])}/{state['total_runs']} runs")
        else:
            raise RuntimeError("--resume specified but no state file found")
    else:
        if state_fp.exists():
            raise RuntimeError(
                f"State file exists: {state_fp}\n"
                "Use --resume to continue or remove the state file to start fresh."
            )
        state = build_temporal_state(launch_id, pairs, seeds, windows, completed=[])
        atomic_write_state(state_fp, state)
        print(f"[temporal-rsa] Starting new launch: {launch_id}")

    # Main training loop: seeds × pairs × windows
    rows: List[Dict] = []
    completed_count = len(state.get("completed", []))

    for seed in seeds:
        for class_a, class_b in pairs:
            for window_start, window_end in windows:
                # Check if already completed
                if is_run_completed(state, class_a, class_b, seed, window_start, window_end):
                    # Load existing results
                    run_dir = output_root / format_temporal_run_dir(
                        launch_id, class_a, class_b, seed, window_start, window_end
                    )
                    try:
                        rows.extend(aggregate_temporal_metrics(
                            run_dir, class_a, class_b, seed, window_start, window_end
                        ))
                    except Exception as e:
                        print(f"[temporal-rsa] Warning: Could not load {run_dir}: {e}")
                    continue

                # Set active pair for the selected task
                set_active_pair_for_task(task_name=str(args.task), pair=(class_a, class_b))
                # Fail fast if pair switching didn't take effect (prevents day-long wasted runs).
                _validate_pair_switch(task_name=str(args.task), label_fn=label_fn, pair=(class_a, class_b))

                # Apply temporal window to config
                cfg = apply_temporal_window_to_config(base_cfg, window_start, window_end)
                cfg["seed"] = int(seed)
                cfg["task"] = args.task
                cfg["engine"] = args.engine

                # Create run directory
                run_dir = output_root / format_temporal_run_dir(
                    launch_id, class_a, class_b, seed, window_start, window_end
                )
                ensure_directory(run_dir)
                cfg["run_dir"] = str(run_dir)

                # Train model
                seed_everything(cfg["seed"])
                window_name = format_window_name(window_start, window_end)
                print(
                    f"[temporal-rsa] [{completed_count + 1}/{state['total_runs']}] "
                    f"{determinism_banner(cfg['seed'])}"
                )
                print(
                    f"[temporal-rsa] Training pair {class_a} vs {class_b} "
                    f"(seed={seed}, window={window_name})",
                    flush=True
                )

                engine_run(cfg, label_fn)

                # Aggregate metrics
                metrics = list(aggregate_temporal_metrics(
                    run_dir, class_a, class_b, seed, window_start, window_end
                ))
                rows.extend(metrics)

                # Update state
                state = mark_run_completed(state, class_a, class_b, seed, window_start, window_end)
                atomic_write_state(state_fp, state)
                completed_count += 1

    # Reset active pair
    # Reset pair state for the selected task (best-effort)
    if str(args.task) == "rsa_landing_binary":
        rsa_landing_binary.reset_active_pair()
    else:
        rsa_binary.reset_active_pair()

    # Sort and write results
    rows = sort_temporal_results(rows)
    summary_path = output_root / RESULTS_FILENAME
    write_temporal_csv(rows, summary_path)
    print(f"[temporal-rsa] Summary written to {summary_path.resolve()}", flush=True)

    # Mark as finished
    state["finished"] = True
    atomic_write_state(state_fp, state)

    print(f"[temporal-rsa] Time-resolved RSA complete!")
    print(f"[temporal-rsa] Results: {summary_path}")
    print(f"[temporal-rsa] Total runs: {completed_count}")


if __name__ == "__main__":
    main()
