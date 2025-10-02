from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "specs" / "001-develop-a-comprehensive" / "contracts" / "training_runner_config.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def training_runner_contract() -> dict:
    if not CONTRACT_PATH.exists():
        pytest.skip(f"training_runner_config.yaml missing at {CONTRACT_PATH}")
    return _load_yaml(CONTRACT_PATH)


def test_required_min_subjects(training_runner_contract: dict) -> None:
    required = training_runner_contract["config"]["required"]
    assert required["min_subjects"]["min"] >= 10, "min_subjects minimum must be ≥10"


def test_required_min_trials_per_class(training_runner_contract: dict) -> None:
    required = training_runner_contract["config"]["required"]
    assert required["min_trials_per_class"]["min"] >= 5, "min_trials_per_class minimum must be ≥5"


def test_artifact_checks(training_runner_contract: dict) -> None:
    artifacts = training_runner_contract["artifacts"]["run_dir"]
    required_files = set(artifacts["required_files"])

    expected = {
        "resolved_config.yaml",
        "split_indices.json",
        "metrics_outer.csv",
        "metrics_inner.csv",
        "class_weights/",
        "stats/glmm_results.json",
        "stats/permutation_results.json",
        "xai/",
        "predictions/outer_predictions.csv",
        "predictions/inner_predictions.csv",
        "logs/runtime.jsonl",
    }
    missing = expected - required_files
    assert not missing, f"Missing required artifact entries: {sorted(missing)}"

    checks = artifacts["checks"]
    by_name = {check["name"]: check["condition"] for check in checks}
    assert "subject_count_threshold" in by_name
    assert "trials_per_class_threshold" in by_name
    assert "excluded_subject_log" in by_name
    assert "glmm_present" in by_name


