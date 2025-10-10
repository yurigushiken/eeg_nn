"""Task: landing digit 1–3 within small set ONLY (no cardinality restriction).

Allowed conditions: any two-digit combination where both digits are in {1,2,3}:
  11, 12, 13, 21, 22, 23, 31, 32, 33

Labels are the landing digit ('1','2','3'). Non-small-set conditions become NaN.
"""
import numpy as np
import pandas as pd

__all__ = ["label_fn"]


def label_fn(meta: pd.DataFrame):
    conditions = {12, 13, 21, 23, 31, 32}
    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10
    out = landing_digit.where(cond_int.isin(conditions), other=np.nan)
    return out.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


