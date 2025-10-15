# HRP/MinVar 등 리스크 기반 배분, 기대수익 추정, 거래비용/턴오버 제약

from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
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

def get_corr_dist(corr, method='pearson'):
    """
    상관계수 행렬을 거리 행렬로 변환합니다.
    """
    dist = np.sqrt((1 - corr) / 2)
    return dist

def get_quasi_diag(link):
    """
    계층적 클러스터링 결과를 기반으로 자산 순서를 재정렬합니다.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def get_rec_bipart(cov, sort_ix):
    """
    재귀적으로 포트폴리오 가중치를 계산합니다. (HRP의 핵심)
    """
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [
            i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_cov = cov.loc[c_items0, c_items1]
            ivp = 1 / np.diag(c_cov)
            ivp /= ivp.sum()
            w[c_items0] *= ivp.sum()
            w[c_items1] *= (1 - ivp.sum())
    return w

def hrp_alloc(returns):
    """
    HRP(계층적 리스크 패리티) 포트폴리오 배분을 계산합니다.
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        return pd.Series(1.0, index=returns.columns)

    corr = returns.corr()
    cov = returns.cov()
    
    dist = get_corr_dist(corr)
    link = sch.linkage(squareform(dist), 'single')
    sort_ix = get_quasi_diag(link)
    
    sort_ix = corr.index[sort_ix].tolist()
    
    hrp = get_rec_bipart(cov, sort_ix)
    return hrp.sort_index()

