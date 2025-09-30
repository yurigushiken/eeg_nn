# optuna_tools/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_flag(name: str, default: str = "1") -> bool:
    val = os.environ.get(name, default)
    if val is None:
        val = default
    val = str(val).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    project_root: Path
    optuna_root: Path
    optuna_db_dir: Path

    # “light mode” config (env-tunable)
    light_mode: bool
    large_trials_threshold: int
    parallel_dims_large: int
    slice_dims_large: int
    contour_dims_large: int
    skip_png_on_large: bool
    force_png_parallel: bool
    force_png_contour: bool
    png_scale: int


def load_config(root: str | None = None, force_no_lite: bool = False) -> Config:
    """
    Build a Config object using either:
      - explicit 'root' (path to your project root containing results/, optuna_studies/, scripts/)
      - or infer it by assuming this file is in <root>/optuna_tools/config.py
    """
    if root:
        project_root = Path(root).resolve()
    else:
        # infer: <root>/optuna_tools/config.py
        project_root = Path(__file__).resolve().parents[1]

    optuna_root = project_root / "results" / "optuna"
    optuna_db_dir = project_root / "optuna_studies"

    # Env overrides with sensible defaults
    light_mode_env = _env_flag("OPTUNA_LIGHT_MODE", "1")
    light_mode = (not force_no_lite) and light_mode_env

    cfg = Config(
        project_root=project_root,
        optuna_root=optuna_root,
        optuna_db_dir=optuna_db_dir,
        light_mode=light_mode,
        large_trials_threshold=int(os.environ.get("OPTUNA_LIGHT_N_TRIALS", "500")),
        parallel_dims_large=int(os.environ.get("OPTUNA_LIGHT_PARALLEL_DIMS", "10")),
        slice_dims_large=int(os.environ.get("OPTUNA_LIGHT_SLICE_DIMS", "12")),
        contour_dims_large=int(os.environ.get("OPTUNA_LIGHT_CONTOUR_DIMS", "6")),
        skip_png_on_large=_env_flag("OPTUNA_LIGHT_SKIP_PNG", "1"),
        force_png_parallel=_env_flag("OPTUNA_FORCE_PNG_PARALLEL", "0"),
        force_png_contour=_env_flag("OPTUNA_FORCE_PNG_CONTOUR", "0"),
        png_scale=int(os.environ.get("OPTUNA_PNG_SCALE", "2")),
    )
    return cfg
