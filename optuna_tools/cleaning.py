# optuna_tools/cleaning.py
from __future__ import annotations

import shutil
from pathlib import Path


def clean_prefix(study_dir: Path) -> None:
    """Remove any outputs starting with '!' in a study dir."""
    for item in study_dir.iterdir():
        if item.name.startswith("!"):
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception:
                pass
