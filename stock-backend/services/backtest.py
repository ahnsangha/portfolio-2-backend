# 간단 롱숏 전략 백테스트 

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from .portfolio import portfolio_construct
@dataclass
class BTResult:
    equity: pd.Series
    stats: Dict[str, float]
    daily: pd.DataFrame

# 롱/숏 백테스팅
def long_short_backtest(returns: pd.DataFrame,
                        pred_table: pd.DataFrame,
                        top_k: int = 5,
                        bottom_k: int = 5,
                        cost_bps: float = 10.0
                        ) -> BTResult:
    
    # 예측 데이터가 없으면 빈 결과 반환
    if len(pred_table) == 0:
        return BTResult(equity=pd.Series(dtype=float), stats={}, daily=pd.DataFrame())

    # 예측 테이블 복사
    df = pred_table.copy()
    
    # 멀티인덱스 처리 수정
    if not isinstance(df.index, pd.MultiIndex):
        # date가 이미 컬럼에 있는지 확인
        if 'date' in df.columns and 'ticker' in df.columns:
            df = df.set_index(["date", "ticker"])
        elif 'date' not in df.columns:
            # date가 인덱스에 있다면 초기화하지 않고 바로 사용
            df['ticker'] = df.get('ticker', df.index)
            df = df.set_index(["date", "ticker"])
    
    # 날짜와 티커 기준으로 정렬
    df = df.sort_index()

    # 예측이 존재하는 날짜 목록
    dates = df.index.get_level_values(0).unique()
    # 운용할 전체 티커 목록
    tickers = returns.columns

    # 날짜별로 점수 스케일을 맞추기 위해 z 점수로 변환
    def _z(g):
        if 'pred' in g.columns:
            s = g["pred"]
        else:
            # pred 컬럼이 없으면 첫 번째 숫자 컬럼 사용
            numeric_cols = g.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                s = g[numeric_cols[0]]
            else:
                return pd.Series(index=g.index, dtype=float)
        
        mean_val = s.mean()
        std_val = s.std()
        if std_val == 0 or pd.isna(std_val):
            return pd.Series(0, index=s.index)
        return (s - mean_val) / (std_val + 1e-8)
    
    # z-score 계산 수정했음
    try:
        # groupby 후 apply를 더 안전하게 처리
        grouped = df.groupby(level=0)
        zscore_list = []
        
        for date, group in grouped:
            z_values = _z(group.reset_index(level=0, drop=True))
            z_values.name = 'z'
            # date를 다시 추가
            z_df = pd.DataFrame(z_values)
            z_df['date'] = date
            z_df['ticker'] = z_df.index
            zscore_list.append(z_df)
        
        if zscore_list:
            zscores = pd.concat(zscore_list)
            zscores = zscores.set_index(['date', 'ticker'])
        else:
            zscores = pd.DataFrame()
    except Exception as e:
        print(f"Z-score calculation error: {e}")
        # 폴백: 간단한 z-score 계산
        zscores = df[['pred']].copy() if 'pred' in df.columns else df.iloc[:, 0:1].copy()
        zscores.columns = ['z']

    # 날짜별 포트폴리오 가중치 생성
    weights = []
    for d in dates:
        try:
            if d in zscores.index.get_level_values(0):
                # 해당 날짜의 z-score 추출
                daily_z = zscores.loc[d]
                if isinstance(daily_z, pd.DataFrame):
                    daily_z = daily_z['z'] if 'z' in daily_z.columns else daily_z.iloc[:, 0]
                
                # 점수 높은 순으로 정렬
                sorted_z = daily_z.sort_values(ascending=False)
                
                # 상위와 하위 종목 선택
                long = sorted_z.head(top_k).index
                short = sorted_z.tail(bottom_k).index
            else:
                long = []
                short = []
            
            # 기본 가중치는 0
            w = pd.Series(0.0, index=tickers)
            
            # 상위는 동일가중 매수
            if len(long) > 0:
                for ticker in long:
                    if ticker in tickers:
                        w.loc[ticker] = 1.0 / len(long)
            
            # 하위는 동일가중 매도
            if len(short) > 0:
                for ticker in short:
                    if ticker in tickers:
                        w.loc[ticker] = -1.0 / len(short)
            
            weights.append(w)
        except Exception as e:
            print(f"Weight calculation error for date {d}: {e}")
            weights.append(pd.Series(0.0, index=tickers))

    # 가중치가 없으면 빈 결과 반환
    if not weights:
        return BTResult(equity=pd.Series(dtype=float), stats={}, daily=pd.DataFrame())

    # 날짜, 티커 가중치 표로 변환하고 빈 날짜는 0으로 채움
    W = pd.DataFrame(weights, index=dates, columns=tickers).reindex(returns.index).fillna(0.0)

    # 다음 날 수익률로 체결 가정
    ret_next = returns.shift(-1).reindex(W.index)
    
    # 거래비용 차감 전 포트폴리오 수익
    port_ret_gross = (W * ret_next).sum(axis=1)

    # 전일 대비 가중치 변화로 회전율 계산
    Wlag = W.shift(1).fillna(0.0)
    turnover = (W - Wlag).abs().sum(axis=1) * 0.5
    
    # 회전율에 bps 비용을 곱해 거래비용 계산
    cost = turnover * (cost_bps / 10000.0)
    
    # 비용 차감 후 실제 수익
    port_ret_net = port_ret_gross - cost

    # 순수익 누적으로 자산곡선 계산
    equity = (1 + port_ret_net.fillna(0.0)).cumprod()

    # 연율화 계수
    ann = 252
    
    # 성과 지표 계산
    if len(port_ret_net) > 0:
        # 연간 수익률
        mu = port_ret_net.mean() * ann
        # 연간 변동성
        sig = port_ret_net.std() * np.sqrt(ann)
        # 샤프비율
        sharpe = mu / (sig + 1e-8)

        # 최대낙폭 계산
        cum = (1 + port_ret_net.fillna(0.0)).cumprod()
        dd = (cum / cum.cummax() - 1)
        mdd = dd.min()
        
        # CAGR 계산
        if len(equity) > 0 and equity.iloc[-1] > 0:
            cagr = (equity.iloc[-1] ** (252 / len(equity)) - 1)
        else:
            cagr = 0.0
    else:
        mu = sig = sharpe = mdd = cagr = 0.0

    # 요약 지표 정리
    stats = {
        "CAGR": float(cagr),
        "AnnRet": float(mu),
        "AnnVol": float(sig),
        "Sharpe": float(sharpe),
        "MaxDD": float(mdd),
        "Turnover": float(turnover.mean() if len(turnover) > 0 else 0)
    }

    # 일별 결과 표
    daily = pd.DataFrame({
        "ret": port_ret_net,
        "turnover": turnover
    })

    # 최종 결과 반환
    return BTResult(equity=equity, stats=stats, daily=daily)


def walk_forward_backtest(
    price_df: pd.DataFrame,            # date x ticker 종가
    rebal_days: int = 21,
    lookback_days: int = 252,
    cost_bps: float = 10.0,
    method: str = "HRP",
    max_weight: float = 0.25,
    max_turnover: float = 0.3
) -> dict:
    ret_df = price_df.pct_change().fillna(0.0)
    dates = ret_df.index

    weights = {}
    prev_w = None
    equity = [1.0]
    rets_path = []

    for i in range(lookback_days, len(dates)):
        if (i - lookback_days) % rebal_days == 0:
            window = ret_df.iloc[i-lookback_days:i]
            w = portfolio_construct(window, method=method, max_weight=max_weight, prev_w=prev_w, max_turnover=max_turnover)
            # 거래비용
            tc = (prev_w.fillna(0) - w.fillna(0)).abs().sum() * (cost_bps / 10000.0) if prev_w is not None else 0.0
            weights[dates[i]] = w
            prev_w = w
        # 일일 수익
        day_r = (ret_df.iloc[i] * prev_w.reindex(ret_df.columns).fillna(0)).sum()
        day_r -= tc if 'tc' in locals() else 0.0
        equity.append(equity[-1] * (1 + day_r))
        rets_path.append(day_r)

    eq_series = pd.Series(equity[1:], index=dates[lookback_days:])
    ret_series = pd.Series(rets_path, index=dates[lookback_days:])
    stats = {
        "CAGR": (eq_series.iloc[-1]) ** (252/len(eq_series)) - 1 if len(eq_series) > 0 else 0.0,
        "Vol": ret_series.std() * np.sqrt(252),
        "Sharpe": (ret_series.mean() / (ret_series.std() + 1e-8)) * np.sqrt(252),
        "MaxDD": ((eq_series.cummax() - eq_series) / eq_series.cummax()).max()
    }
    return {"weights": weights, "equity": eq_series, "stats": stats}