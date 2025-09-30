#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optuna_tools import load_config, rebuild_index  # noqa: E402


def main():
    cfg = load_config()
    rebuild_index(cfg)


if __name__ == "__main__":
    main()
