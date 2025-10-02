from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "refresh_optuna_summaries.py"
CONTRACT_PATH = ROOT / "specs" / "001-develop-a-comprehensive" / "contracts" / "optuna_refresh_contract.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.skipif(not SCRIPT.exists(), reason="Optuna refresh script not available")
def test_optuna_refresh_contract(tmp_path):
    contract = _load_yaml(CONTRACT_PATH)

    results_dir = tmp_path / "results" / "optuna"
    study_dir = results_dir / "study_A"
    plots_dir = study_dir / "!plots-study_A"
    study_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "!all_trials-study_A.csv").write_text("run_id,trial_id,mean_acc\n", encoding="utf-8")
    (plots_dir / "history-study_A.png").write_bytes(b"")
    (study_dir / "!top3-report-study_A.html").write_text("<html></html>", encoding="utf-8")

    env = {
        "PYTHONPATH": str(ROOT),
        "OPTUNA_REFRESH_RESULTS": str(results_dir),
    }

    proc = subprocess.run(
        ["python", str(SCRIPT), "--results-root", str(results_dir)],
        cwd=str(ROOT),
        env={**env},
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr

    index_csv = results_dir / "optuna_runs_index.csv"
    assert index_csv.exists()

    expected_headers = set(contract["global_outputs"]["index_csv"]["required_headers"])
    headers = (index_csv.read_text(encoding="utf-8").splitlines()[0]).split(",")
    assert expected_headers.issubset(headers)

