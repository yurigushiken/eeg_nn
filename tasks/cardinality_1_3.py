"""Task: classify landing digit among {1,2,3} for same-digit pairs (11,22,33).
Label function returns strings '1','2','3'; non-target conditions become NaN.
"""
import numpy as np
import pandas as pd

__all__ = ["label_fn", "CONDITIONS"]

CONDITIONS = [11, 22, 33]

def label_fn(meta: pd.DataFrame):
    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10
    valid = cond_int.isin(CONDITIONS)
    out = landing_digit.where(valid, other=np.nan)
    return out.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


