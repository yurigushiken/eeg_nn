"""Task: classify landing on 2 vs 3 within small set.

Valid conditions (prime->target) limited to small digits only:
  - Landing on 2: 12, 32
  - Landing on 3: 13, 23

Labels are strings '2' and '3'; any other condition becomes NaN.
"""
import numpy as np
import pandas as pd

__all__ = ["label_fn", "CONDITIONS_2", "CONDITIONS_3"]

CONDITIONS_2 = {12, 32}
CONDITIONS_3 = {13, 23}
ALL_CONDITIONS = CONDITIONS_2 | CONDITIONS_3


def label_fn(meta: pd.DataFrame):
    cond_int = meta["Condition"].astype(int)
    valid = cond_int.isin(list(ALL_CONDITIONS))
    # landing digit is second digit of the two-digit code
    landing_digit = cond_int % 10
    out = landing_digit.where(valid, other=np.nan)
    return out.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


