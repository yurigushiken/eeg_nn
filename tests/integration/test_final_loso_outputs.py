from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FINAL_EVAL_PATH = ROOT / "scripts" / "final_eval.py"


def _load_final_eval_module():
    if not FINAL_EVAL_PATH.exists():
        pytest.fail(f"final_eval.py missing at {FINAL_EVAL_PATH}")
    spec = importlib.util.spec_from_file_location("final_eval_cli", FINAL_EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_final_loso_outputs_require_artifacts(tmp_path):
    final_eval = _load_final_eval_module()
    validator = getattr(final_eval, "validate_loso_artifacts", None)
    if validator is None:
        pytest.fail("validate_loso_artifacts must be implemented to enforce LOSO output contract")

    missing = validator(run_dir=tmp_path)
    assert isinstance(missing, (list, set, tuple)), "Validator should return missing artifact identifiers"
    required = {
        "resolved_config.yaml",
        "metrics_outer.csv",
        "metrics_inner.csv",
        "class_weights/",
        "stats/glmm_results.json",
        "stats/permutation_results.json",
        "predictions/outer_predictions.csv",
        "predictions/inner_predictions.csv",
        "logs/runtime.jsonl",
    }
    for item in required:
        assert item in missing, f"Validator must flag missing artifact: {item}"

