"""Task: landing digit 4–6 within large set with cardinality control.
Allowed conditions: [44,54,64,45,55,65,46,56,66]. Labels are '4','5','6'.
"""
import numpy as np
import pandas as pd

__all__ = ["label_fn"]

def label_fn(meta: pd.DataFrame):
    conditions = [44,54,64,45,55,65,46,56,66]
    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10
    out = landing_digit.where(cond_int.isin(conditions), other=np.nan)
    return out.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


