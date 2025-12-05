"""
Cardinality 3 vs 4 classification.
Prime 3 landing on 3 (condition 33) vs Prime 4 landing on 4 (condition 44).
"""
import numpy as np
import pandas as pd

CONDITIONS = [33, 44]  # Same-digit pairs: prime=target

def label_fn(meta: pd.DataFrame):
    """
    Extract cardinality labels (3 or 4) for condition codes 33 and 44.

    Returns:
        Series of '3' or '4' (strings), with NaN for excluded trials.
    """
    cond_int = meta["Condition"].astype(int)
    landing_digit = cond_int % 10  # Extract second digit (target)
    valid = cond_int.isin(CONDITIONS)
    out = landing_digit.where(valid, other=np.nan)
    return out.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
