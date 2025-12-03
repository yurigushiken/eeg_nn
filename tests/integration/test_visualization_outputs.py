from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
POSTHOC_PATH = ROOT / "scripts" / "run_posthoc_stats.py"
XAI_PATH = ROOT / "scripts" / "run_xai_analysis.py"


def _load_module(path: Path, name: str):
    if not path.exists():
        pytest.fail(f"Expected module at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_visualization_contract_reports_missing_artifacts(tmp_path):
    posthoc_mod = _load_module(POSTHOC_PATH, "run_posthoc_stats")
    validator = getattr(posthoc_mod, "validate_visualization_outputs", None)
    if validator is None:
        pytest.fail("run_posthoc_stats.validate_visualization_outputs must be implemented to enforce FR-026")

    run_dir = tmp_path / "run"
    (run_dir / "stats").mkdir(parents=True, exist_ok=True)
    missing = validator(run_dir)

    required = {
        f"plots_outer/{run_dir.name}_overall_confusion.png",
        f"plots_outer/{run_dir.name}_fold01_confusion.png",
        "stats/per_subject_forest.png",
        "stats/caterpillar_plot.png",
        "stats/xai/summary.html",
    }

    assert isinstance(missing, (set, list, tuple)), "Validator should return iterable of missing artifacts"
    missing_set = set(missing)
    for item in required:
        assert item in missing_set, f"Validator must flag missing visualization: {item}"


@pytest.mark.integration
def test_xai_validator_requires_reports(tmp_path):
    xai_mod = _load_module(XAI_PATH, "run_xai_analysis")
    validator = getattr(xai_mod, "validate_xai_outputs", None)
    if validator is None:
        pytest.fail("run_xai_analysis.validate_xai_outputs must be implemented to enforce FR-026")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    missing = set(validator(run_dir))
    assert "xai/summary_report.html" in missing
    assert "xai/per_subject/subject_01.html" in missing

