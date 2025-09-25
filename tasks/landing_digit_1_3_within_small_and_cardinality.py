"""Task: landing digit 1–3 within small set with cardinality control.
Allowed conditions: [11,21,31,12,22,32,13,23,33]. Labels are '1','2','3'.
"""
import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    conditions = [11,21,31,12,22,32,13,23,33]
    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10
    out = landing_digit.where(cond_int.isin(conditions), other=np.nan)
    return out.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


