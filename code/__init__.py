from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_stdlib_code() -> ModuleType:
    """Load the standard library's ``code`` module without recursive import."""
    stdlib_path = Path(sysconfig.get_path("stdlib"))
    code_path = stdlib_path / "code.py"
    spec = importlib.util.spec_from_file_location("_stdlib_code", code_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate standard library code module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_code = _load_stdlib_code()

# Mirror the public API of the stdlib ``code`` module.
__all__ = list(getattr(_stdlib_code, "__all__", []))
for _name in __all__:
    globals()[_name] = getattr(_stdlib_code, _name)

# Preserve helpful metadata from the stdlib module when available.
if getattr(_stdlib_code, "__doc__", None):
    __doc__ = _stdlib_code.__doc__  # type: ignore[assignment]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin delegation layer
    if hasattr(_stdlib_code, name):
        return getattr(_stdlib_code, name)
    if name == "TrainingRunner":
        from .training_runner import TrainingRunner  # local import for parity
        return TrainingRunner
    if name == "validate_stage_handoff":
        from .training_runner import validate_stage_handoff
        return validate_stage_handoff
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - cosmetic helper
    exports = set(__all__)
    exports.update(["TrainingRunner", "validate_stage_handoff"])
    exports.update(dir(_stdlib_code))
    return sorted(exports)


# Ensure star-imports see our additional training runner symbols.
_EXTRA_EXPORTS = ["TrainingRunner", "validate_stage_handoff"]
for _extra in _EXTRA_EXPORTS:
    if _extra not in __all__:
        __all__.append(_extra)

