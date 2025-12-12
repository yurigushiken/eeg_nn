"""
Backward-compatible wrapper for `scripts/rsa/analyze_rsa_stats.py`.

Note: We explicitly re-export `_t_test` because `from module import *` omits
underscore-prefixed symbols by default.
"""

from scripts.rsa.analyze_rsa_stats import compute_subject_ttests, filter_subject_rows, load_master_csv, main, _t_test

__all__ = [
    "load_master_csv",
    "filter_subject_rows",
    "compute_subject_ttests",
    "_t_test",
]


if __name__ == "__main__":
    main()


