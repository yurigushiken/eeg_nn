# optuna_tools/discovery.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Any


def iter_study_dirs(optuna_root: Path) -> Iterator[Path]:
    """Yield each study directory under results/optuna that isn't a transient '!' dir."""
    if not optuna_root.exists():
        return
    for p in sorted(optuna_root.iterdir()):
        if p.is_dir() and not p.name.startswith("!"):
            yield p


def _pick_summary_file(trial_dir: Path) -> Path | None:
    # Prefer summary.json; fall back to first summary_*.json
    cand = trial_dir / "summary.json"
    if cand.exists():
        return cand
    for x in sorted(trial_dir.glob("summary_*.json")):
        return x
    return None


def collect_trials(study_dir: Path) -> Tuple[List[Dict[str, Any]], float]:
    """
    Scan a single study_dir for per-trial summary*.json files.
    Return (records, latest_mtime) where records are dicts from JSON with 'trial_dir' injected.
    """
    records: List[Dict[str, Any]] = []
    latest_mtime: float = 0.0

    for trial_dir in sorted(study_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        sf = _pick_summary_file(trial_dir)
        if not sf or not sf.exists():
            continue
        latest_mtime = max(latest_mtime, sf.stat().st_mtime)
        try:
            rec = json.loads(sf.read_text(encoding="utf-8"))
            rec["trial_dir"] = trial_dir.name
            records.append(rec)
        except Exception:
            # swallow & continue – one bad file shouldn't kill the run
            pass
    return records, latest_mtime
