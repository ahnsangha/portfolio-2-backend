# 시장 국면을 판단해 온도 지표를 만드는 모듈

from __future__ import annotations # 향후 버전의 타입 힌트 문법을 지금 사용하기 위함
from dataclasses import dataclass # 간단한 데이터 담는 클래스를 자동 생성하기 위함
from typing import Dict # 타입 표기를 위한 임포트 
import numpy as np # 수치 계산을 빠르게 처리하기 위함
import pandas as pd # 시계열과 표 형태 데이터를 다루기 위함

@dataclass # 반복 코드 없이 필드만으로 간단한 결과 객체를 만들기 위함
class RegimeResult:
    temperature: float # 현재 시점의 최종 온도 값 
    series: pd.DataFrame # 전체 시계열 결과 표, 컬럼은 vol avg_corr temp를 가짐

def detect_regime(returns: pd.DataFrame) -> RegimeResult: # 수익률 데이터로 시장 국면을 계산하는 함수
    # returns 는 날짜 및 티커 형태의 일간 수익률 데이터라고 가정함
    m = returns.mean(axis=1) # 같은 날 종목들을 평균해 시장의 일간 평균 수익을 만듦
    vol = m.rolling(20).std() # 시장 평균수익의 20일 표준편차를 구해 최근 변동성을 측정함

    # 평균 상관을 근사하기 위한 내부 함수 (한 윈도우에서 종목간 상관의 평균을 구함)
    def _avg_corr(win: pd.DataFrame) -> float: # 한 구간의 수익률 표를 받아 평균 상관 한 값으로 요약함
        if win.shape[0] < 20: # 표본이 너무 적으면 신뢰도 낮으므로 결측 처리함
            return np.nan # 계산 대신 결측 반환
        c = win.corr().values # 종목 간 상관행렬을 계산함
        iu = np.triu_indices_from(c, k=1) # 대각선 위쪽 요소 인덱스만 선택해 자기 자신과의 상관을 제외함
        return float(np.nanmean(c[iu])) # 위쪽 삼각형 값의 평균을 반환함

    avg_corr = returns.rolling(60).apply( # 60일 구간마다 평균 상관을 계산함
        lambda x: _avg_corr(pd.DataFrame(x, columns=returns.columns)), # 윈도우를 데이터프레임으로 바꿔 내부 함수에 전달함
        raw=False # 넘파이 배열이 아닌 데이터프레임 형태로 전달하도록 지정함
    )

    vz = (vol - vol.mean()) / (vol.std() + 1e-8) # 변동성을 표준화해 z 점수로 만듦
    cz = (avg_corr - avg_corr.mean()) / (avg_corr.std() + 1e-8) # 평균 상관도 같은 방식으로 표준화함

    # 변동성과 상관이 높을수록 위험 구간이므로 온도를 낮추는 방향으로 설계함
    temp = 1 / (1 + np.exp(0.8 * (vz.fillna(0) + cz.fillna(0)))) # 시그모이드로 0~1 사이 값으로 압축함
    temp = 0.6 + 0.4 * temp # 운용 편의를 위해 0.6~1.0 사이로 다시 스케일함

    series = pd.DataFrame({"vol": vol, "avg_corr": avg_corr, "temp": temp}).dropna() # 세 지표를 합쳐 결측행을 제거함
    last_temp = float(series["temp"].iloc[-1]) if len(series) else 0.8 # 최신 온도를 뽑고 없으면 안전하게 0.8을 사용함
    return RegimeResult(temperature=last_temp, series=series) # 계산된 온도와 시계열 표를 묶어 반환함