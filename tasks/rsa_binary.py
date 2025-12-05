"""
RSA binary task support with configurable cardinality pairs.

The label function keeps only trials whose condition code matches the
currently active pair (e.g., 22 vs 55) and maps them to string labels.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

DEFAULT_PAIR = (22, 33)
_ACTIVE_PAIR: Tuple[int, int] = DEFAULT_PAIR


def _normalize_pair(pair: Iterable[int]) -> Tuple[int, int]:
    values = tuple(int(x) for x in pair)
    if len(values) != 2:
        raise ValueError(f"Pair must have exactly two elements, got {values}")
    if values[0] == values[1]:
        raise ValueError(f"Pair elements must be distinct, got {values}")
    return values


def set_active_pair(pair: Iterable[int]) -> None:
    """Set the global active condition pair (e.g., (22, 55))."""
    global _ACTIVE_PAIR
    _ACTIVE_PAIR = _normalize_pair(pair)
    # Update label_fn name so dataset cache keys incorporate the active pair.
    label_fn.__name__ = f"rsa_binary_label_{_ACTIVE_PAIR[0]}_{_ACTIVE_PAIR[1]}"


def get_active_pair() -> Tuple[int, int]:
    """Return the currently active condition pair."""
    return _ACTIVE_PAIR


def reset_active_pair() -> None:
    """Reset to the default active pair."""
    set_active_pair(DEFAULT_PAIR)


def label_fn(meta: pd.DataFrame) -> pd.Series:
    """
    Return labels for the active condition pair.

    Trials whose condition is not in the active pair are mapped to NaN.
    """
    if "Condition" not in meta.columns:
        raise KeyError("Expected 'Condition' column in metadata")

    cond = meta["Condition"].astype("Int64")
    a, b = _ACTIVE_PAIR
    mask = cond.isin([a, b])
    filtered = cond.where(mask)
    return filtered.map(lambda x: str(int(x)) if pd.notna(x) else pd.NA)


__all__ = [
    "set_active_pair",
    "get_active_pair",
    "reset_active_pair",
    "label_fn",
]


# Initialize label function name with default pair context.
reset_active_pair()

