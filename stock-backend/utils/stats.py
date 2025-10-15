# 롤링 상관계수와 95% 신뢰구간, 유효샘플 수 반환 

import numpy as np
import pandas as pd

def rolling_corr_with_ci(s1: pd.Series, s2: pd.Series, window: int = 60, alpha: float = 0.05):
    s1, s2 = s1.align(s2, join="inner")
    r = s1.rolling(window).corr(s2)

    # Fisher Z
    z = np.arctanh(r.clip(-0.999999, 0.999999))
    n = s1.rolling(window).count()
    se = 1.0 / np.sqrt((n - 3).clip(lower=1))
    z_crit = 1.96 if abs(alpha - 0.05) < 1e-9 else float(pd.Series([alpha]).apply(lambda a: 1.96)) # 단순화
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return pd.DataFrame({"corr": r, "low": lo, "high": hi, "n_eff": n})
