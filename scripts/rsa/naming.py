from __future__ import annotations

import re
from pathlib import Path
from typing import Literal


Kind = Literal["figures", "tables", "models", "confounds", "stats"]


_SAFE_RE = re.compile(r"[^a-z0-9_-]+")


def sanitize_analysis_id(value: str) -> str:
    """
    Make an analysis identifier safe for filenames.
    Policy: lowercase; allow [a-z0-9_-] only; collapse other runs to '_'.
    """
    v = value.strip().lower()
    v = _SAFE_RE.sub("_", v)
    v = re.sub(r"_+", "_", v).strip("_")
    if not v:
        raise ValueError("analysis_id resolved to empty after sanitization")
    return v


def analysis_id_from_run_root(run_root: Path) -> str:
    return sanitize_analysis_id(Path(run_root).name)


def prefixed_name(*, analysis_id: str, stem: str, ext: str) -> str:
    if not ext.startswith("."):
        raise ValueError(f"ext must start with '.', got {ext!r}")
    s = stem.strip()
    if not s:
        raise ValueError("stem must be non-empty")
    a = sanitize_analysis_id(analysis_id)
    return f"{a}__{s}{ext}"


def prefixed_path(*, run_root: Path, kind: Kind, stem: str, ext: str) -> Path:
    analysis_id = analysis_id_from_run_root(run_root)
    out_dir = Path(run_root) / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / prefixed_name(analysis_id=analysis_id, stem=stem, ext=ext)


def prefixed_title(*, run_root: Path, title: str) -> str:
    analysis_id = analysis_id_from_run_root(run_root)
    t = title.strip()
    # Use ASCII-safe separator to avoid mojibake in some environments/fonts.
    return f"{analysis_id} - {t}" if t else analysis_id


