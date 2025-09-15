import numpy as np
import pandas as pd

__all__ = ["label_fn", "CONDITIONS"]

CONDITIONS = [44, 55, 66]

def label_fn(meta: pd.DataFrame):
    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10
    valid = cond_int.isin(CONDITIONS)
    out = landing_digit.where(valid, other=np.nan)
    return out.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


