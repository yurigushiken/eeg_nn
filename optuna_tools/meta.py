# optuna_tools/meta.py
from __future__ import annotations

import json
from pathlib import Path


def _meta_fp(study_dir: Path) -> Path:
    return study_dir / f"!refresh_meta-{study_dir.name}.json"


def needs_refresh(study_dir: Path, n_trials: int, latest_mtime: float) -> bool:
    fp = _meta_fp(study_dir)
    if not fp.exists():
        return True
    try:
        meta = json.loads(fp.read_text(encoding="utf-8"))
        return meta.get("n_trials") != n_trials or meta.get("latest_mtime") != latest_mtime
    except Exception:
        return True


def save_meta(study_dir: Path, n_trials: int, latest_mtime: float) -> None:
    _meta_fp(study_dir).write_text(
        json.dumps({"n_trials": n_trials, "latest_mtime": latest_mtime}, indent=2),
        encoding="utf-8",
    )
