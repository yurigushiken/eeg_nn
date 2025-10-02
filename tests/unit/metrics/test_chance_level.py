from __future__ import annotations

import pytest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

# Load project summary module directly by path to avoid tests/utils shadowing
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_FP = PROJECT_ROOT / "utils" / "summary.py"
_spec = spec_from_file_location("_proj_summary", str(SUMMARY_FP))
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
summary = _mod  # alias for clarity


def test_compute_chance_level_defined_and_correct():
    fn = getattr(summary, "_compute_chance_level", None)
    assert callable(fn)
    assert pytest.approx(fn(2), rel=1e-6) == 50.0
    assert pytest.approx(fn(3), rel=1e-6) == 33.3333333333
    assert fn(None) == 0.0

