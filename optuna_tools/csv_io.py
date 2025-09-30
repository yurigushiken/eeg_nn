# optuna_tools/csv_io.py
from __future__ import annotations

from pathlib import Path


def _all_trials_fp(study_dir: Path) -> Path:
    return study_dir / f"!all_trials-{study_dir.name}.csv"


def write_trials_csv(study_dir: Path, df) -> Path:
    out = _all_trials_fp(study_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
