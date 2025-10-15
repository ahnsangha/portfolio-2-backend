# 시장/섹터/요인 효과를 제거하여 '팩터-중립 수익률' 생성 + 부분상관 지원
# 분석 전에 returns를 중립화해 상관 분석의 왜곡을 줄임

from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_concat(cols: list[pd.Series | pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(cols, axis=1).dropna(how="any")

def build_factor_matrix(
    factor_dict: dict[str, pd.Series | pd.DataFrame], # {"MKT": series, "SECTOR_x": series or df(one-hot)}
    index: pd.Index
) -> pd.DataFrame:
    # 인덱스를 맞추고 결측 제거
    mats = []
    for k, v in factor_dict.items():
        df = v.to_frame(name=k) if isinstance(v, pd.Series) else v
        mats.append(df.reindex(index))
    X = _safe_concat(mats)
    return X

# 일자별 횡단면 OLS: r_t = X_t * b_t + e_t
# e_t(잔차)를 '중립화 수익률'로 사용
def neutralize_to_factors(
    returns: pd.DataFrame, # (date x tickers) 단위 수익률
    factors: pd.DataFrame, # (date x factors) 팩터 행렬 
    add_const: bool = True
) -> pd.DataFrame:
    R = returns.copy()
    idx = R.index.intersection(factors.index)
    R = R.loc[idx]
    X = factors.loc[idx]
    if add_const and "CONST" not in X.columns:
        X = X.assign(CONST=1.0)

    resid_list = []
    for date, y in R.iterrows():
        x = X.loc[date]
        # 일부 팩터 결측 시 스킵
        x = x.dropna()
        y = y.dropna()
        common = y.index.intersection(x.index)  # 팩터는 공용, 종목별로 동일
        Xmat = np.tile(x.values, (len(common), 1))
        yvec = y.loc[common].values.reshape(-1, 1)

        # OLS 닫힌해: b = (X'X)^-1 X'y
        XtX = Xmat.T @ Xmat
        try:
            b = np.linalg.pinv(XtX) @ (Xmat.T @ yvec)
        except Exception:
            resid = pd.Series(index=common, data=np.nan)
            resid_list.append(pd.Series({date: resid}))
            continue
        yhat = (Xmat @ b).ravel()
        resid = y.loc[common] - pd.Series(yhat, index=common)
        resid.name = date
        resid_list.append(resid)

    resid_df = pd.DataFrame(resid_list)
    return resid_df

# 부분상관: Precision matrix(공분산의 역행렬)에서 표준화해 근사 계산
def partial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    D = df.dropna(axis=1, how="all").fillna(0.0) # 결측/상수열 제거
    # 표준화
    D = (D - D.mean()) / (D.std(ddof=1) + 1e-8)
    cov = np.cov(D.values, rowvar=False)
    prec = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(prec))
    pcorr = -prec / (d[:, None] * d[None, :])
    np.fill_diagonal(pcorr, 1.0)
    return pd.DataFrame(pcorr, index=D.columns, columns=D.columns)