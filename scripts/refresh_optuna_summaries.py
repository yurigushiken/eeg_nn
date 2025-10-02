#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable so 'optuna_tools' can be found
THIS_DIR = Path(__file__).resolve().parent            # .../eeg_nn/scripts
PROJECT_ROOT = THIS_DIR.parent                        # .../eeg_nn
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optuna_tools import load_config, run_refresh, rebuild_index  # noqa: E402


def _coerce_project_root_from_results_root(results_root: str | None) -> Path | None:
    if not results_root:
        return None
    p = Path(results_root).resolve()
    # If pointed directly at <root>/results/optuna, return <root>
    try:
        if p.name.lower() == "optuna" and p.parent.name.lower() == "results":
            return p.parent.parent
    except Exception:
        pass
    # Otherwise assume it's already project root
    return p


def main():
    ap = argparse.ArgumentParser(description="Refresh Optuna study summaries (CSV/plots) for each study folder.")
    ap.add_argument("--root", default=None, help="Project root (defaults to scripts/..)")
    ap.add_argument("--results-root", dest="results_root", default=None,
                    help="Alias for --root; may point directly to <root>/results/optuna")
    ap.add_argument("--no-lite", dest="no_lite", action="store_true",
                    help="Disable light mode (always plot all variable params)")
    ap.add_argument("--rebuild-index", action="store_true", help="Rebuild global runs index at the end")
    args = ap.parse_args()

    # Prefer results_root when provided; coerce to project root if it points to results/optuna
    effective_project_root = _coerce_project_root_from_results_root(args.results_root)
    if effective_project_root is None:
        effective_project_root = Path(args.root).resolve() if args.root else PROJECT_ROOT

    cfg = load_config(root=str(effective_project_root), force_no_lite=args.no_lite)
    # Always rebuild index to satisfy contract expectations
    run_refresh(cfg, rebuild_index=True)


if __name__ == "__main__":
    main()
