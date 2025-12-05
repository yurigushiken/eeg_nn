import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pytest
import json


def _write_outer_eval_csv(path: Path, overall: bool = True) -> None:
    rows: List[Dict[str, str]] = [
        {
            "outer_fold": "1",
            "test_subjects": "subj_01",
            "n_test_trials": "72",
            "acc": "60.0",
            "acc_std": "",
            "macro_f1": "58.0",
            "macro_f1_std": "",
            "min_per_class_f1": "55.0",
            "min_per_class_f1_std": "",
        },
    ]
    if overall:
        rows.append(
            {
                "outer_fold": "OVERALL",
                "test_subjects": "-",
                "n_test_trials": "72",
                "acc": "62.0",
                "acc_std": "2.0",
                "macro_f1": "60.0",
                "macro_f1_std": "1.0",
                "min_per_class_f1": "56.0",
                "min_per_class_f1_std": "0.5",
            }
        )

    fieldnames = [
        "outer_fold",
        "test_subjects",
        "n_test_trials",
        "acc",
        "acc_std",
        "macro_f1",
        "macro_f1_std",
        "min_per_class_f1",
        "min_per_class_f1_std",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_run_dir(root: Path, launch_id: str, pair: Tuple[int, int], seed: int, complete: bool) -> Path:
    run_dir = root / f"{launch_id}_rsa_{pair[0]}v{pair[1]}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if complete:
        _write_outer_eval_csv(run_dir / "outer_eval_metrics.csv", overall=True)
    else:
        _write_outer_eval_csv(run_dir / "outer_eval_metrics.csv", overall=False)
    return run_dir


def test_resume_skips_completed_and_runs_pending(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "rsa_matrix_resume"
    runs_dir.mkdir()
    launch_id = "20251205_031857"
    complete_dir = _make_run_dir(runs_dir, launch_id, (11, 22), 42, complete=True)
    pending_dir = _make_run_dir(runs_dir, launch_id, (11, 33), 42, complete=False)

    cfg_path = tmp_path / "resume_config.yaml"
    cfg_path.write_text(
        "seeds: [42]\n"
        "model_name: eegnex\n"
        "optuna_objective: inner_mean_min_per_class_f1\n"
        "materialized_dir: data_preprocessed/hpf_1.5_lpf_40_baseline-on\n"
        "epochs: 1\n"
        "inner_n_folds: 2\n"
        "n_folds: 2\n"
        "outer_eval_mode: ensemble\n"
        "conditions: [11, 22, 33]\n"
    )

    from scripts import run_rsa_matrix

    monkeypatch.setattr(run_rsa_matrix, "seed_everything", lambda seed: None)
    monkeypatch.setattr(run_rsa_matrix, "determinism_banner", lambda seed: f"seed={seed}")

    engine_calls: List[Path] = []

    def fake_engine(cfg, label_fn):
        run_path = Path(cfg["run_dir"]).resolve()
        engine_calls.append(run_path)
        _write_outer_eval_csv(run_path / "outer_eval_metrics.csv", overall=True)

    monkeypatch.setattr(run_rsa_matrix, "get_engine", lambda _: fake_engine)
    monkeypatch.setattr(run_rsa_matrix, "get_task", lambda _: lambda df: df)

    argv = [
        "python",
        "--config",
        str(cfg_path),
        "--output-dir",
        str(runs_dir),
        "--conditions",
        "11",
        "22",
        "33",
        "--resume",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_rsa_matrix.main()

    engine_call_set = {p.resolve() for p in engine_calls}
    assert pending_dir.resolve() in engine_call_set
    assert complete_dir.resolve() not in engine_call_set


def test_resume_seed_mismatch_raises(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "rsa_matrix_resume_mismatch"
    runs_dir.mkdir()
    launch_id = "20251205_111111"
    _make_run_dir(runs_dir, launch_id, (11, 22), 41, complete=True)

    state = {
        "launch_id": launch_id,
        "pairs": [[11, 22]],
        "seeds": [41],
        "completed": [{"ClassA": 11, "ClassB": 22, "Seed": 41}],
        "finished": False,
    }
    (runs_dir / ".rsa_resume_state.json").write_text(json.dumps(state))

    cfg_path = tmp_path / "resume_config.yaml"
    cfg_path.write_text(
        "seeds: [42]\n"
        "model_name: eegnex\n"
        "optuna_objective: inner_mean_min_per_class_f1\n"
        "materialized_dir: data_preprocessed/hpf_1.5_lpf_40_baseline-on\n"
        "epochs: 1\n"
        "inner_n_folds: 2\n"
        "n_folds: 2\n"
        "outer_eval_mode: ensemble\n"
    )

    from scripts import run_rsa_matrix

    argv = [
        "python",
        "--config",
        str(cfg_path),
        "--output-dir",
        str(runs_dir),
        "--resume",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError):
        run_rsa_matrix.main()

