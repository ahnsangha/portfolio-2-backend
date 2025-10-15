# HRP/MinVar 등 리스크 기반 배분, 기대수익 추정, 거래비용/턴오버 제약

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def shrink_cov(returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(returns.fillna(0.0).values)
    S = lw.covariance_
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)

def expected_returns(
    returns: pd.DataFrame,
    method: str = "mom6", # 6개월 모멘텀 기본
    shrink: float = 0.5
) -> pd.Series:
    rets = returns.copy()
    if method.startswith("mom"):
        m = int(method.replace("mom", "") or 6)
        mu = (1 + rets).rolling(m*21).apply(lambda x: np.prod(1+x) - 1, raw=False)
        er = mu.iloc[-1].fillna(0.0)
    else:
        er = rets.mean() * 252
    # shrink to 0 (보수적)
    er = (1 - shrink) * er
    return er

def minvar_weights(cov: pd.DataFrame, max_w: float = 0.2) -> pd.Series:
    inv = np.linalg.pinv(cov.values)
    w = inv.sum(axis=1)
    w = np.maximum(w, 1e-12)
    w = w / w.sum()
    w = np.minimum(w, max_w)  # 단순 상한 제약
    w = w / w.sum()
    return pd.Series(w, index=cov.index)

# HRP 간단 구현
def hrp_weights(cov: pd.DataFrame, max_w: float = 0.25) -> pd.Series:
    # 근사: 변동성 역수 가중 + 상한
    vol = np.sqrt(np.diag(cov.values))
    w = 1.0 / (vol + 1e-8)
    w = w / w.sum()
    w = np.minimum(w, max_w)
    w = w / w.sum()
    return pd.Series(w, index=cov.index)

def apply_turnover(prev_w: pd.Series | None, new_w: pd.Series, max_turnover: float = 0.3):
    if prev_w is None:
        return new_w
    diff = (new_w.fillna(0) - prev_w.fillna(0))
    tot = diff.abs().sum()
    if tot <= max_turnover:
        return new_w
    # 스케일 다운
    scaled = prev_w + diff * (max_turnover / max(tot, 1e-8))
    return scaled / scaled.sum()

def portfolio_construct(
    returns: pd.DataFrame,
    method: str = "HRP",
    max_weight: float = 0.25,
    prev_w: pd.Series | None = None,
    max_turnover: float = 0.3
) -> pd.Series:
    cov = shrink_cov(returns)
    if method.upper() == "HRP":
        w = hrp_weights(cov, max_w=max_weight)
    else:
        w = minvar_weights(cov, max_w=max_weight)
    w = apply_turnover(prev_w, w, max_turnover=max_turnover)
    return w
