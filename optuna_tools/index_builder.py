# optuna_tools/index_builder.py
from __future__ import annotations

import json, csv
from pathlib import Path
from typing import Dict, Any, List

from .config import Config


def _flatten_hyper(h: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in h.items():
        if isinstance(v, (list, tuple)):
            flat[k] = " | ".join(str(x) for x in v)
        elif isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}.{sk}"] = sv
        else:
            flat[k] = v
    return flat


def rebuild_index(cfg: Config) -> Path:
    optuna_root = cfg.optuna_root
    csv_path = optuna_root / "optuna_runs_index.csv"

    rows: List[Dict[str, Any]] = []
    all_keys: set[str] = set()

    print(f"[index] scanning {optuna_root} for per-trial summaries …")
    num_files = 0
    for fp in optuna_root.rglob("summary_*.json"):
        num_files += 1
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = {
            "run_id": data.get("run_id", ""),
            "study": data.get("study", ""),
            "trial_id": data.get("trial_id", ""),
            "mean_acc": data.get("mean_acc", ""),
            "std_acc": data.get("std_acc", ""),
        }
        flat = _flatten_hyper(data.get("hyper", {}))
        row.update(flat)
        all_keys.update(flat.keys())
        rows.append(row)

    header = ["run_id", "study", "trial_id", "mean_acc", "std_acc"] + sorted(all_keys)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            for c in header:
                r.setdefault(c, "")
            w.writerow({k: r[k] for k in header})
    print(f"[index] wrote {csv_path.name} files_scanned={num_files} rows={len(rows)} cols={len(header)}")
    return csv_path
