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

from optuna_tools import load_config, run_refresh  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Refresh Optuna study summaries (CSV/plots) for each study folder.")
    ap.add_argument("--root", default=None, help="Project root (defaults to scripts/..)")
    ap.add_argument("--no-lite", dest="no_lite", action="store_true",
                    help="Disable light mode (always plot all variable params)")
    ap.add_argument("--rebuild-index", action="store_true", help="Rebuild global runs index at the end")
    args = ap.parse_args()

    cfg = load_config(root=args.root, force_no_lite=args.no_lite)
    run_refresh(cfg, rebuild_index=args.rebuild_index)


if __name__ == "__main__":
    main()
