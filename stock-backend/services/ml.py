# ML 분석 모듈 (PCA/KMeans/HRP/IVP/MinVar/GRU/RF) 
# 입력: returns(DataFrame: 날짜, 티커), prices(DataFrame 선택) 
# 목표: 다음 horizon 영업일 평균 수익률을 단면 랭킹으로 예측 및 평가(IC/Hit)
# 안정성: 시드 고정, gradient clipping, LR 스케줄러, 예외 로깅 
# 자동화: 유니버스 크기 기반 하이퍼파라미터 자동 설정·메타 튜닝

from __future__ import annotations # 타입 힌트 호환성
from dataclasses import dataclass # 데이터 클래스 정의용
from typing import Dict, Any, Optional, List, Tuple # 타입 힌트용
import logging # 로깅 모듈
import numpy as np # 수치 계산
import pandas as pd # 데이터프레임

from sklearn.decomposition import PCA # 주성분 분석
from sklearn.cluster import KMeans # K-평균 클러스터링
from sklearn.preprocessing import StandardScaler # 표준화
from sklearn.ensemble import RandomForestRegressor # 랜덤포레스트 회귀
from sklearn.metrics import r2_score # R제곱 점수
from sklearn.covariance import LedoitWolf # Ledoit-Wolf 공분산 추정

from scipy.cluster.hierarchy import linkage, leaves_list # 계층 클러스터링
from scipy.spatial.distance import squareform # 거리행렬 변환

logger = logging.getLogger(__name__) # 로거 인스턴스

# 💡 [최적화 1] TORCH_AVAILABLE 플래그를 사용하여 torch를 조건부로 import 합니다.
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    logger.debug("Torch not available, GRU model disabled.")
    torch = None
    nn = None
    DataLoader = None
    Dataset = None

# 결과 전달용 컨테이너 
@dataclass
class PCAResult:
    components: pd.DataFrame # PCA 주성분 좌표
    explained_var: np.ndarray # 설명된 분산 비율

    @property
    def explained_variance_ratio_(self): # sklearn PCA 호환성 프로퍼티
        return self.explained_var

# 결과 전달용 컨테이너: 클러스터 결과
@dataclass
class ClusterResult:
    labels: pd.Series # 각 종목의 클러스터 라벨
    centers: pd.DataFrame # 클러스터 중심점 좌표

# 결과 전달용 컨테이너: HRP 결과
@dataclass
class HRPResult:
    weights: pd.Series # HRP 가중치
    order: List[str] # 계층 순서

# 결과 전달용 컨테이너: 예측 결과
@dataclass
class PredictionResult:
    horizon: int # 예측 기간
    ic: float # 정보 계수 (예측과 실제의 상관계수)
    hit_rate: float # 적중률 (방향 맞춘 비율)
    r2: Optional[float] # R제곱 점수
    ic_by_date: pd.Series # 날짜별 IC
    preds_vs_real: pd.DataFrame # 예측값 vs 실제값

# 메인 분석기 클래스
class MLAnalyzer:
    def __init__(self, returns: pd.DataFrame, prices: Optional[pd.DataFrame] = None):
        assert isinstance(returns, pd.DataFrame), "returns must be a DataFrame" # DataFrame 체크
        self.returns = returns.sort_index().copy() # 날짜순 정렬 후 복사
        self.returns = self.returns.replace([np.inf, -np.inf], np.nan).fillna(0.0) # 무한값 제거
        self.prices = None if prices is None else prices.reindex(self.returns.index).copy() # 가격 데이터

        # 하단 표시용 메시지 버퍼 (차트 하단에 렌더링할 텍스트)
        self.ui_messages: List[str] = []

    # 하단 메시지 버퍼 유틸
    def _push_msg(self, text: str) -> None:
        # 하단 표시용 메시지 버퍼에 추가(빈 줄/양끝 공백 정리).
        try:
            if not isinstance(text, str): # 문자열이 아니면 변환
                text = str(text)
            text = "\n".join(line.rstrip() for line in text.strip().splitlines()) # 공백 정리
            if text: # 빈 문자열이 아니면 추가
                self.ui_messages.append(text)
        except Exception as e:
            logger.debug("push_msg failed: %s", e) # 실패 시 디버그 로그

    # 자동 튜닝: 유니버스/데이터 길이에 맞춰 합리적 기본값 산출
    def _auto_tune(
        self,
        horizon: int, # 예측 기간
        seq_len: Optional[int], # 시퀀스 길이
        epochs: Optional[int], # 에폭 수
        hidden_size: Optional[int], # 히든 사이즈
        dropout: Optional[float], # 드롭아웃 비율
        hrp_max_weight: Optional[float], # HRP 최대 가중치
        hrp_blend_to_equal: Optional[float], # 균등 가중치 블렌딩 비율
        long_tau: Optional[float], # 롱 포지션 온도 파라미터
        short_tau: Optional[float], # 숏 포지션 온도 파라미터
    ) -> Dict[str, Any]:
        n_days = len(self.returns) # 전체 데이터 길이
        n_stk = self.returns.shape[1] # 종목 수
        max_seq = max(20, int(min(n_days * 0.6, 160))) # 최대 시퀀스 길이

        if n_stk <= 4: # 소규모 유니버스
            seq_len = seq_len or min(50, max_seq) # 시퀀스 길이 50 또는 최대값
            epochs = epochs or 60 # 에폭 60
            hidden_size = hidden_size or 32 # 히든 사이즈 32
            dropout = dropout if dropout is not None else 0.20 # 드롭아웃 20%
            hrp_max_weight = hrp_max_weight # 최대 가중치 제한 없음
            hrp_blend_to_equal = 0.0 if hrp_blend_to_equal is None else hrp_blend_to_equal # 균등 블렌딩 없음
            long_tau = long_tau or 1.0 # 롱 온도 1.0
            short_tau = short_tau or 1.0 # 숏 온도 1.0
        elif n_stk <= 10: # 중간 규모 유니버스
            seq_len = seq_len or min(80, max_seq) # 시퀀스 길이 80
            epochs = epochs or 45 # 에폭 45
            hidden_size = hidden_size or 48 # 히든 사이즈 48
            dropout = dropout if dropout is not None else 0.15 # 드롭아웃 15%
            hrp_max_weight = 0.35 if hrp_max_weight is None else hrp_max_weight # 최대 가중치 35%
            hrp_blend_to_equal = 0.20 if hrp_blend_to_equal is None else hrp_blend_to_equal # 균등 블렌딩 20%
            long_tau = long_tau or 0.85 # 롱 온도 0.85
            short_tau = short_tau or 0.85 # 숏 온도 0.85
        else: # 대규모 유니버스
            seq_len = seq_len or min(120, max_seq) # 시퀀스 길이 120
            epochs = epochs or 30 # 에폭 30
            hidden_size = hidden_size or 64 # 히든 사이즈 64
            dropout = dropout if dropout is not None else 0.10 # 드롭아웃 10%
            hrp_max_weight = 0.25 if hrp_max_weight is None else hrp_max_weight # 최대 가중치 25%
            hrp_blend_to_equal = 0.30 if hrp_blend_to_equal is None else hrp_blend_to_equal # 균등 블렌딩 30%
            long_tau = long_tau or 0.70 # 롱 온도 0.70
            short_tau = short_tau or 0.70 # 숏 온도 0.70

        if seq_len >= n_days - horizon - 5: # 시퀀스가 너무 길면 조정
            seq_len = max(20, int((n_days - horizon - 5) * 0.6))

        return dict( # 튜닝된 파라미터 반환
            seq_len=seq_len, epochs=epochs, hidden_size=hidden_size, dropout=dropout,
            hrp_max_weight=hrp_max_weight, hrp_blend_to_equal=hrp_blend_to_equal,
            long_tau=long_tau, short_tau=short_tau
        )

    # 특징 생성: 모멘텀·변동성, 왜도, 첨도, 베타
    def build_features(self, windows=(5, 20, 60)) -> pd.DataFrame:
        parts = [] # 특징들을 저장할 리스트
        for w in windows: # 각 윈도우에 대해
            m = self.returns.rolling(w).mean() # 모멘텀 (이동평균)
            m.columns = pd.MultiIndex.from_product([m.columns, [f"mom{w}"]]) # 컬럼명 설정

            v = self.returns.rolling(w).std() # 변동성 (이동표준편차)
            v.columns = pd.MultiIndex.from_product([v.columns, [f"vol{w}"]]) # 컬럼명 설정

            sk = self.returns.rolling(w).skew() # 왜도 (비대칭도)
            sk.columns = pd.MultiIndex.from_product([sk.columns, [f"skew{w}"]]) # 컬럼명 설정

            ku = self.returns.rolling(w).kurt() # 첨도 (뾰족함)
            ku.columns = pd.MultiIndex.from_product([ku.columns, [f"kurt{w}"]]) # 컬럼명 설정

            parts += [m, v, sk, ku] # 특징 리스트에 추가

        market_ret = self.returns.mean(axis=1) # 시장 수익률 (평균)
        betas = {} # 베타 저장용 딕셔너리
        for c in self.returns.columns: # 각 종목에 대해
            x = market_ret # 시장 수익률
            y = self.returns[c] # 개별 종목 수익률
            cov = (x * y).rolling(60).mean() - x.rolling(60).mean() * y.rolling(60).mean() # 공분산
            var = x.rolling(60).var() # 시장 분산
            beta = cov / (var.replace(0, np.nan)) # 베타 계산
            betas[c] = beta # 베타 저장
        beta_df = pd.DataFrame(betas) # 베타 DataFrame
        beta_df.columns = pd.MultiIndex.from_product([beta_df.columns, ["beta"]]) # 컬럼명 설정

        X = pd.concat(parts + [beta_df], axis=1) # 모든 특징 결합
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0) # 무한값 제거
        X.columns.names = ["ticker", "feature"] # 컬럼 이름 설정
        return X

    # PCA 2차원 좌표
    def pca_2d(self) -> PCAResult:
        feats = self.build_features() # 특징 생성
        last_row = feats.tail(1).iloc[0] # 마지막 날짜 데이터
        last_row.index = last_row.index.set_names(["ticker", "feature"]) # 인덱스 이름 설정
        stock_features = last_row.unstack("feature").fillna(0.0) # 종목별 특징으로 재구성

        scaler = StandardScaler() # 표준화 객체
        Z = scaler.fit_transform(stock_features.values).copy() # 특징 표준화

        n_comp = int(min(2, Z.shape[0], Z.shape[1])) # 주성분 개수 (최대 2개)
        if n_comp < 1: # 주성분이 없으면
            components = pd.DataFrame(np.zeros((Z.shape[0], 2)), # 영행렬 생성
                                     index=stock_features.index, columns=["PC1", "PC2"])
            explained = np.array([1.0, 0.0]) # 설명된 분산 비율
            return PCAResult(components=components, explained_var=explained)

        pca = PCA(n_components=n_comp, random_state=42) # PCA 객체 생성
        coords = pca.fit_transform(Z) # PCA 변환
        components = pd.DataFrame(coords, index=stock_features.index, # 주성분 좌표
                                 columns=["PC1"] + (["PC2"] if n_comp == 2 else []))
        if n_comp == 1: # 주성분이 1개면
            components["PC2"] = 0.0 # PC2를 0으로 설정
            explained = np.array([pca.explained_variance_ratio_[0], 0.0]) # 설명된 분산
        else:
            evr = np.asarray(pca.explained_variance_ratio_) # 설명된 분산 비율
            explained = np.array([float(evr[0]), float(evr[1]) if len(evr) > 1 else 0.0])
        return PCAResult(components=components[["PC1", "PC2"]], explained_var=explained)

    # KMeans
    def kmeans_clusters(self, pca_result: Optional[PCAResult] = None, k_clusters: int = 4) -> ClusterResult:
        if pca_result is None: # PCA 결과가 없으면 실행
            pca_result = self.pca_2d()
        coords = pca_result.components.values # PCA 좌표
        n_samples = coords.shape[0] # 샘플 수
        k = max(1, min(k_clusters, n_samples)) # 클러스터 수 조정
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto") # KMeans 객체
        except TypeError: # n_init="auto" 지원 안하면
            km = KMeans(n_clusters=k, random_state=42, n_init=10) # 기본값 사용
        labels = km.fit_predict(coords) # 클러스터링 실행
        centers = pd.DataFrame(km.cluster_centers_, columns=["PC1", "PC2"][: km.cluster_centers_.shape[1]]) # 중심점
        return ClusterResult(labels=pd.Series(labels, index=pca_result.components.index), # 라벨
                            centers=centers) # 중심점

    # 캡/플로어 및 정규화
    def _apply_caps(self, w: pd.Series, max_weight: Optional[float], min_weight: float) -> pd.Series:
        w = w.astype(float).copy().clip(lower=min_weight) # 최소 가중치 적용
        if max_weight is not None: # 최대 가중치가 있으면
            w = pd.Series(np.minimum(w.to_numpy(copy=True), float(max_weight)), index=w.index) # 최대 가중치 적용
        s = float(w.sum()) # 가중치 합계
        if not np.isfinite(s) or s <= 0: # 합계가 유효하지 않으면
            return pd.Series(1.0 / len(w), index=w.index) # 균등 가중치 반환
        return w / s # 정규화

    # IVP
    def _weights_ivp(self, cov: pd.DataFrame) -> pd.Series:
        d = np.asarray(cov.values, dtype=float).diagonal().copy() # 대각선 (분산)
        d[d <= 0] = np.nan # 0 이하는 NaN
        ivp = 1.0 / d # 역분산 가중치
        ivp = np.nan_to_num(ivp, nan=0.0) # NaN을 0으로
        if ivp.sum() == 0: # 합계가 0이면
            ivp = np.ones_like(ivp) # 균등 가중치
        ivp = ivp / ivp.sum() # 정규화
        return pd.Series(ivp, index=cov.index)

    # MinVar
    def _weights_minvar(self, cov: pd.DataFrame) -> pd.Series:
        try:
            inv = np.linalg.pinv(cov.values) # 공분산 역행렬 (의사역행렬)
            ones = np.ones(len(cov)) # 1벡터
            w = inv @ ones # 최소분산 가중치
            w = w / (ones @ inv @ ones) # 정규화
            w = pd.Series(w, index=cov.index) # Series로 변환
        except Exception as e:
            logger.exception("MinVar weight failed; falling back to equal-weight. Reason: %s", e) # 실패 로그
            w = pd.Series(1.0 / len(cov), index=cov.index) # 균등 가중치로 폴백
        return w

    # HRP/IVP/MinVar 통합
    def hrp_weights(self,
                    scheme: str = "hrp", # 가중치 방식
                    max_weight: Optional[float] = 0.25, # 최대 가중치
                    min_weight: float = 0.0, # 최소 가중치
                    blend_to_equal: float = 0.30, # 균등 가중치 블렌딩 비율
                    resamples: int = 0, # 리샘플링 횟수
                    seed: int = 42) -> HRPResult: # 시드
        rng = np.random.default_rng(seed) # 랜덤 생성기
        R = self.returns.fillna(0.0) # NaN을 0으로 채움

        def _one(df: pd.DataFrame) -> pd.Series: # 단일 가중치 계산 함수
            cov = pd.DataFrame(LedoitWolf().fit(df).covariance_, index=df.columns, columns=df.columns) # Ledoit-Wolf 공분산
            if scheme.lower() == "ivp": # IVP 방식
                w = self._weights_ivp(cov)
                order = list(w.index)
            elif scheme.lower() == "minvar": # MinVar 방식
                w = self._weights_minvar(cov)
                order = list(w.index)
            else: # HRP 방식
                corr = df.corr().fillna(0.0) # 상관계수 행렬
                dist = np.sqrt(0.5 * (1 - corr)).clip(lower=0) # 거리 행렬
                Z = linkage(squareform(dist, checks=False), method="ward") # 계층 클러스터링
                order_idx = list(leaves_list(Z)) # 리프 순서
                order = [df.columns[i] for i in order_idx] # 컬럼 순서
                w = self._hrp_allocation(cov, order_idx) # HRP 가중치

            if blend_to_equal and blend_to_equal > 0: # 균등 가중치 블렌딩
                w = (1 - blend_to_equal) * w + (blend_to_equal) * (pd.Series(1.0, index=w.index) / len(w))
            w = self._apply_caps(w, max_weight, min_weight) # 캡/플로어 적용
            return w.sort_values(ascending=False) # 내림차순 정렬

        if resamples and resamples > 0: # 리샘플링 사용
            ws = [] # 가중치 리스트
            for _ in range(resamples): # 리샘플링 횟수만큼
                idx = rng.choice(len(R), size=len(R), replace=True) # 부트스트랩 인덱스
                ws.append(_one(R.iloc[idx])) # 가중치 계산
            w = pd.concat(ws, axis=1).mean(axis=1) # 평균 가중치
            w = w / w.sum() # 정규화
            order = list(w.index) # 순서
            return HRPResult(weights=w.sort_values(ascending=False), order=order)

        w = _one(R) # 단일 가중치 계산
        order = list(w.index) # 순서
        return HRPResult(weights=w.sort_values(ascending=False), order=order)

    # HRP 내부: 재귀 분할 
    def _hrp_allocation(self, cov: pd.DataFrame, order_idx: List[int]) -> pd.Series:
        items = [self.returns.columns[i] for i in order_idx] # 순서대로 종목명
        w = pd.Series(1.0, index=items, dtype=float) # 초기 가중치 1
        clusters = [items] # 클러스터 리스트
        while clusters: # 클러스터가 있는 동안
            new_clusters = [] # 새 클러스터 리스트
            for cl in clusters: # 각 클러스터에 대해
                if len(cl) <= 2: # 2개 이하면 스킵
                    continue
                mid = len(cl) // 2 # 중간점
                left, right = cl[:mid], cl[mid:] # 좌우 분할
                w_left = self._cluster_risk(cov, left) # 좌측 리스크
                w_right = self._cluster_risk(cov, right) # 우측 리스크
                alpha = 1 - w_left / (w_left + w_right + 1e-12) # 배분 비율
                w.loc[left]  = w.loc[left].to_numpy(copy=True)  * alpha # 좌측 가중치 조정
                w.loc[right] = w.loc[right].to_numpy(copy=True) * (1 - alpha) # 우측 가중치 조정
                new_clusters.extend([left, right]) # 새 클러스터에 추가
            clusters = new_clusters # 클러스터 업데이트
        return w / w.sum() # 정규화

    # HRP 내부: 클러스터 위험
    def _cluster_risk(self, cov: pd.DataFrame, items: List[str]) -> float:
        sub = cov.loc[items, items] # 서브 공분산 행렬
        ivp = self._weights_ivp(sub) # IVP 가중치
        risk = float(ivp @ sub.values @ ivp.T) # 위험 계산
        return risk

    # 진단/검증 보조
    @staticmethod
    def _winsorize(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
        return df.clip(lower=-float(k), upper=float(k)) # 극값 제거 (k 표준편차)

    def ic_decay(self,
                 horizons: Tuple[int, ...] = (1, 3, 5, 10), # 예측 기간들
                 use_gru: bool = True, # GRU 사용 여부
                 **kwargs) -> pd.DataFrame:
        rows = [] # 결과 행들
        for h in horizons: # 각 기간에 대해
            pr = self.predict_next(horizon=h, use_gru=use_gru, **kwargs) # 예측 실행
            rows.append({"horizon": int(h), "ic": float(pr.ic), "hit": float(pr.hit_rate), "r2": pr.r2}) # 결과 저장
        out = pd.DataFrame(rows).set_index("horizon").sort_index() # DataFrame 생성
        return out

    def permutation_test(self,
                         horizon: int = 5, # 예측 기간
                         use_gru: bool = True, # GRU 사용 여부
                         n_perm: int = 200, # 순열 횟수
                         seed: int = 42, # 시드
                         **kwargs) -> Dict[str, Any]:
        rng = np.random.default_rng(seed) # 랜덤 생성기
        base = self.predict_next(horizon=horizon, use_gru=use_gru, **kwargs) # 기본 예측
        df = base.preds_vs_real.copy() # 예측 vs 실제 데이터
        if df.empty or not isinstance(df.index, pd.MultiIndex): # 데이터가 없거나 멀티인덱스가 아니면
            return {"base_ic": float(base.ic), "null_ic_mean": np.nan, "null_ic_std": np.nan}

        null_ics: List[float] = [] # 널 IC 리스트
        dates = df.index.get_level_values(0).unique() # 유니크한 날짜들
        for _ in range(int(n_perm)): # 순열 횟수만큼
            ics = [] # IC 리스트
            for d in dates: # 각 날짜에 대해
                g = df.xs(d, level=0) # 해당 날짜 데이터
                y = g["real"].to_numpy(copy=True) # 실제값
                rng.shuffle(y) # 실제값 셔플
                ic = pd.Series(g["pred"].values).corr(pd.Series(y), method="spearman") # 순위 상관계수
                ics.append(ic) # IC 추가
            null_ics.append(float(np.nanmean(ics))) # 평균 IC 추가
        return {
            "base_ic": float(base.ic), # 기본 IC
            "null_ic_mean": float(np.nanmean(null_ics)), # 널 IC 평균
            "null_ic_std": float(np.nanstd(null_ics)), # 널 IC 표준편차
            "n_perm": int(n_perm) # 순열 횟수
        }

    # 공통 유틸리티
    @staticmethod
    def _set_seeds(seed: int = 42, deterministic: bool = True) -> None:
        try:
            np.random.seed(seed) # 넘파이 시드 설정
        except Exception as e:
            logger.debug("Numpy seed set failed: %s", e)
        if TORCH_AVAILABLE: # 토치가 사용 가능하면
            try:
                torch.manual_seed(seed) # 토치 시드 설정
                if torch.cuda.is_available(): # CUDA 사용 가능하면
                    torch.cuda.manual_seed_all(seed) # CUDA 시드 설정
                if deterministic: # 결정론적 모드
                    try:
                        torch.backends.cudnn.deterministic = True # cuDNN 결정론적
                        torch.backends.cudnn.benchmark = False # cuDNN 벤치마크 비활성화
                    except Exception as e:
                        logger.debug("cuDNN deterministic flags failed: %s", e)
            except Exception as e:
                logger.debug("Torch seed set failed: %s", e)

    # 예측 메인
    def predict_next(self,
                     horizon: int = 5, # 예측 기간
                     seq_len: int = 120, # 시퀀스 길이
                     use_gru: bool = True, # GRU 사용 여부
                     epochs: int = 30, # 학습 에폭 수
                     hidden_size: int = 64, # 히든 레이어 크기
                     num_layers: int = 1, # GRU 레이어 수
                     dropout: float = 0.1, # 드롭아웃 비율
                     batch_size: int = 512, # 배치 크기
                     lr: float = 1e-3, # 학습률
                     winsor_k: float = 3.0, # 윈소라이징 임계값
                     seed: int = 42, # 랜덤 시드
                     deterministic: bool = True, # 결정론적 모드
                     grad_clip: float = 1.0, # 그라디언트 클리핑
                     scheduler_patience: int = 2, # 스케줄러 참을성
                     device: Optional[str] = None) -> PredictionResult: # 디바이스
        self._set_seeds(seed=seed, deterministic=deterministic) # 시드 설정

        if use_gru and not TORCH_AVAILABLE: # GRU 사용하지만 토치 없으면
            use_gru = False # GRU 비활성화
            self._push_msg("PyTorch not available, falling back to RandomForest") # 메시지 추가
        if len(self.returns) < (seq_len + horizon + 20): # 데이터가 부족하면
            self._push_msg(f"Insufficient data for seq_len={seq_len}, using RandomForest") # 메시지 추가
            return self._predict_with_rf(horizon=horizon) # RF로 폴백
        if not use_gru: # GRU 사용 안하면
            self._push_msg("Using RandomForest for prediction") # 메시지 추가
            return self._predict_with_rf(horizon=horizon) # RF 사용

        df_ret_raw = self.returns.copy() # 원본 수익률 복사
        tickers = df_ret_raw.columns.tolist() # 티커 리스트
        dates = df_ret_raw.index # 날짜 인덱스

        mu = df_ret_raw.mean(axis=1) # 일별 평균 수익률
        sigma = df_ret_raw.std(axis=1).replace(0, np.nan) # 일별 표준편차
        df_ret = (df_ret_raw.sub(mu, axis=0)).div(sigma, axis=0).fillna(0.0) # 표준화

        fut = np.log1p(df_ret_raw).shift(-horizon).rolling(horizon).sum() # 미래 누적 로그 수익률
        t_mu = fut.mean(axis=1) # 타겟 평균
        t_sd = fut.std(axis=1).replace(0, np.nan) # 타겟 표준편차
        target = (fut.sub(t_mu, axis=0)).div(t_sd, axis=0).fillna(0.0) # 타겟 표준화
        if winsor_k is not None and float(winsor_k) > 0: # 윈소라이징 적용
            target = self._winsorize(target, k=float(winsor_k))

        X_list, y_list, d_list, t_list = [], [], [], [] # 데이터 리스트들
        for t in range(seq_len - 1, len(dates) - horizon): # 시퀀스 범위만큼
            window = df_ret.iloc[t - seq_len + 1:t + 1] # 윈도우 데이터
            yrow = target.iloc[t] # 타겟 행
            x_np = window.values.astype(np.float32) # 입력 배열
            y_np = yrow.values.astype(np.float32) # 타겟 배열
            for s, tk in enumerate(tickers): # 각 종목에 대해
                X_list.append(x_np[:, s:s+1]) # 입력 추가
                y_list.append(y_np[s]) # 타겟 추가
                d_list.append(t) # 날짜 인덱스 추가
                t_list.append(tk) # 티커 추가

        if len(X_list) == 0: # 데이터가 없으면
            return self._predict_with_rf(horizon=horizon) # RF로 폴백

        X = np.stack(X_list, axis=0).astype(np.float32, copy=True) # 입력 스택
        y = np.array(y_list, dtype=np.float32).reshape(-1, 1).copy() # 타겟 배열
        d_idx = np.array(d_list, dtype=np.int64) # 날짜 인덱스 배열
        t_arr = np.array(t_list) # 티커 배열

        valid_t = np.arange(seq_len - 1, len(dates) - horizon) # 유효한 시간 인덱스
        tr_end = valid_t[int(len(valid_t) * 0.6)] # 훈련 끝점
        va_end = valid_t[int(len(valid_t) * 0.8)] # 검증 끝점

        tr_mask = d_idx <= tr_end # 훈련 마스크
        va_mask = (d_idx > tr_end) & (d_idx <= va_end) # 검증 마스크
        te_mask = d_idx > va_end # 테스트 마스크

        if tr_mask.sum() < 100 or te_mask.sum() < 100: # 데이터가 부족하면
            return self._predict_with_rf(horizon=horizon) # RF로 폴백

        Xtr, ytr = X[tr_mask], y[tr_mask] # 훈련 데이터
        Xva, yva = X[va_mask], y[va_mask] # 검증 데이터
        Xte, yte = X[te_mask], y[te_mask] # 테스트 데이터
        dte, tte = d_idx[te_mask], t_arr[te_mask] # 테스트 날짜/티커

        class SeqDS(Dataset): # 시퀀스 데이터셋 클래스
            def __init__(self, X, y): # 초기화
                self.X = torch.from_numpy(X) # 입력 텐서
                self.y = torch.from_numpy(y) # 타겟 텐서
            def __len__(self): return len(self.X) # 길이 반환
            def __getitem__(self, i): return self.X[i], self.y[i] # 아이템 반환

        train_loader = DataLoader(SeqDS(Xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=False) # 훈련 로더
        val_loader   = DataLoader(SeqDS(Xva, yva), batch_size=batch_size, shuffle=False, drop_last=False) # 검증 로더
        test_loader  = DataLoader(SeqDS(Xte, yte), batch_size=batch_size, shuffle=False, drop_last=False) # 테스트 로더

        class GRUReg(nn.Module): # GRU 회귀 모델
            def __init__(self, input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout): # 초기화
                super().__init__() # 부모 클래스 초기화
                self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, # GRU 레이어
                                  batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
                self.head = nn.Linear(hidden_size, 1) # 출력 레이어
            def forward(self, x): # 순전파
                out, _ = self.gru(x) # GRU 통과
                h = out[:, -1, :] # 마지막 히든 상태
                return self.head(h) # 출력 레이어 통과

        device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu") # 디바이스 설정
        model = GRUReg().to(device) # 모델을 디바이스로
        opt = torch.optim.Adam(model.parameters(), lr=lr) # Adam 옵티마이저
        loss_fn = nn.MSELoss() # MSE 손실함수

        scheduler = None # 스케줄러 초기화
        try:
            Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau # 스케줄러 클래스
            try:
                scheduler = Scheduler( # 스케줄러 생성 시도
                    opt, mode="min", factor=0.5,
                    patience=int(max(1, scheduler_patience)),
                    min_lr=1e-5
                )
            except TypeError: # 인자 오류 시
                scheduler = Scheduler( # 간단한 스케줄러 생성
                    opt, mode="min", factor=0.5,
                    patience=int(max(1, scheduler_patience))
                )
        except Exception as e:
            logger.debug("LR scheduler init failed: %s", e) # 스케줄러 실패 로그
            scheduler = None

        best_val = np.inf # 최고 검증 점수
        best_state = None # 최고 모델 상태
        
        # 학습 진행 메시지
        self._push_msg(f"Training {epochs} epochs")
        
        for ep in range(1, epochs + 1): # 에폭만큼 반복
            model.train() # 훈련 모드
            for xb, yb in train_loader: # 훈련 배치
                xb = xb.to(device); yb = yb.to(device) # 디바이스로 이동
                opt.zero_grad(set_to_none=True) # 그라디언트 초기화
                yp = model(xb) # 예측
                loss = loss_fn(yp, yb) # 손실 계산
                loss.backward() # 역전파
                if grad_clip and grad_clip > 0: # 그라디언트 클리핑
                    try:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip)) # 클리핑 적용
                    except Exception as e:
                        logger.debug("Grad clip skipped: %s", e)
                opt.step() # 옵티마이저 스텝

            model.eval() # 평가 모드
            va_loss = 0.0 # 검증 손실 초기화
            with torch.no_grad(): # 그라디언트 비활성화
                for xb, yb in val_loader: # 검증 배치
                    xb = xb.to(device); yb = yb.to(device) # 디바이스로 이동
                    yp = model(xb) # 예측
                    va_loss += float(loss_fn(yp, yb).item()) * len(xb) # 손실 누적

            if scheduler is not None: # 스케줄러가 있으면
                try:
                    scheduler.step(va_loss) # 스케줄러 스텝
                except Exception as e:
                    logger.debug("Scheduler step failed: %s", e)

            if va_loss < best_val: # 검증 손실이 개선되면
                best_val = va_loss # 최고 점수 업데이트
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} # 모델 상태 저장
                if ep % 10 == 0:  # 10 에폭마다 진행 상황 출력
                    self._push_msg(f"Epoch {ep}/{epochs}: validation loss improved")

        if best_state is not None: # 최고 상태가 있으면
            model.load_state_dict(best_state) # 모델 상태 로드

        model.eval() # 평가 모드
        preds = [] # 예측값 리스트
        with torch.no_grad(): # 그라디언트 비활성화
            for xb, _ in test_loader: # 테스트 배치
                xb = xb.to(device) # 디바이스로 이동
                yp = model(xb).cpu().numpy().reshape(-1) # 예측 후 numpy로
                preds.append(yp) # 예측값 추가
        preds = np.concatenate(preds, axis=0) # 예측값 연결
        reals = yte.reshape(-1) # 실제값 평탄화

        df_eval = pd.DataFrame({"date_idx": dte, "ticker": tte, "pred": preds, "real": reals}) # 평가 데이터
        ic_by_date = df_eval.groupby("date_idx").apply( # 날짜별 IC 계산
            lambda g: g["pred"].corr(g["real"], method="spearman")
        )
        hit_by_date = df_eval.groupby("date_idx").apply( # 날짜별 적중률 계산
            lambda g: ((g["pred"] > 0) == (g["real"] > 0)).mean()
        )
        ic_avg = float(np.nanmean(ic_by_date.values)) # 평균 IC
        hit_avg = float(np.nanmean(hit_by_date.values)) # 평균 적중률
        try:
            r2 = float(r2_score(reals, preds)) # R2 점수 계산
        except Exception as e:
            logger.debug("R2 score failed: %s", e)
            r2 = None

        pred_table = df_eval.copy() # 예측 테이블 복사
        pred_table["date"] = pred_table["date_idx"].map(lambda i: dates[int(i)]) # 날짜 매핑
        pred_table = pred_table.set_index(["date", "ticker"])[["pred", "real"]].sort_index() # 인덱스 설정

        return PredictionResult( # 예측 결과 반환
            horizon=horizon, ic=ic_avg, hit_rate=hit_avg, r2=r2,
            ic_by_date=pd.Series(ic_by_date.values, index=[dates[int(i)] for i in ic_by_date.index]),
            preds_vs_real=pred_table
        )

    def _predict_with_rf(self, horizon: int = 5) -> PredictionResult:
        X = self.build_features() # 특징 생성
        y = self.returns.mean(axis=1).shift(-horizon).rolling(horizon).mean() # 타겟 생성 (미래 평균 수익률)
        df = pd.concat([X, y.rename("target")], axis=1).dropna() # 특징과 타겟 결합
        if df.empty: # 데이터가 없으면
            return PredictionResult( # 빈 결과 반환
                horizon=horizon, ic=float("nan"), hit_rate=float("nan"), r2=None,
                ic_by_date=pd.Series(dtype=float),
                preds_vs_real=pd.DataFrame(columns=["pred", "real"])
            )

        feature_cols = [c for c in df.columns if c != "target"] # 특징 컬럼
        split_idx = int(len(df) * 0.6) # 분할 인덱스
        Xtr = df.iloc[:split_idx][feature_cols].values # 훈련 특징
        ytr = df.iloc[:split_idx]["target"].values # 훈련 타겟
        Xte = df.iloc[split_idx:][feature_cols].values # 테스트 특징
        yte = df.iloc[split_idx:]["target"].values # 테스트 타겟

        scaler = StandardScaler() # 표준화 객체
        Xtr = scaler.fit_transform(Xtr) # 훈련 데이터 표준화
        Xte = scaler.transform(Xte) # 테스트 데이터 표준화

        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1) # RF 모델
        model.fit(Xtr, ytr) # 모델 훈련
        yp = model.predict(Xte) # 예측

        pred_table = pd.DataFrame({"pred": yp, "real": yte}, index=df.index[split_idx:]) # 예측 테이블
        return PredictionResult( # 예측 결과 반환
            horizon=horizon, ic=float("nan"), hit_rate=float("nan"),
            r2=float(r2_score(yte, yp)),
            ic_by_date=pd.Series(dtype=float),
            preds_vs_real=pd.DataFrame(columns=["pred", "real"])
        )

    # 종목 추천 및 포트폴리오 구성
    def _build_picks(self, pred_res: PredictionResult, base_weights: pd.Series,
                     top_n: int, bottom_n: int, # 상위/하위 개수
                     long_tau: float = 0.7, short_tau: float = 0.7, # 온도 파라미터
                     long_only: bool = False) -> Dict[str, Any]: # 롱온리 여부
        def _softmax(s: pd.Series, tau: float) -> pd.Series: # 소프트맥스 함수
            if len(s) == 0: # 빈 시리즈면
                return s
            x = (s.to_numpy(dtype=float, copy=True) - np.nanmax(s.values)) / max(tau, 1e-6) # 정규화
            w = np.exp(x) # 지수 변환
            denom = w.sum() # 분모
            if not np.isfinite(denom) or denom <= 0: # 유효하지 않으면
                return pd.Series(0.0, index=s.index) # 0 반환
            w = w / denom # 정규화
            return pd.Series(w, index=s.index)

        out = {"asof_date": None, "top": [], "bottom": [], "today_weights": {}} # 출력 초기화
        try:
            df = pred_res.preds_vs_real # 예측 vs 실제 데이터
            if df is None or df.empty or not isinstance(df.index, pd.MultiIndex): # 데이터 확인
                return out

            last_date = df.index.get_level_values(0).max() # 마지막 날짜
            snap = df.xs(last_date, level=0).copy() # 마지막 날짜 데이터
            sig = snap["pred"].astype(float) # 예측 신호

            try:
                from services.data import stock_manager # 주식 매니저 임포트
                name_map = {s["ticker"]: s["name"] for s in stock_manager.get_all_stocks()} # 이름 매핑
            except Exception as e:
                logger.debug("stock_manager not available, using ticker as name: %s", e)
                name_map = {t: t for t in sig.index} # 티커를 이름으로 사용

            base_w = base_weights.reindex(sig.index).fillna(0.0) # 기본 가중치

            long_score  = (sig.clip(lower=0.0) * base_w).replace([np.inf, -np.inf], 0.0) # 롱 스코어
            short_score = ((-sig.clip(upper=0.0)) * base_w).replace([np.inf, -np.inf], 0.0) # 숏 스코어

            long_w  = _softmax(long_score,  long_tau)  if long_score.sum()  > 0 else pd.Series(0.0, index=sig.index) # 롱 가중치
            short_w = _softmax(short_score, short_tau) if short_score.sum() > 0 else pd.Series(0.0, index=sig.index) # 숏 가중치

            top_idx = sig.sort_values(ascending=False).head(top_n).index.tolist() # 상위 종목
            bot_idx = sig.sort_values(ascending=True).head(bottom_n).index.tolist() # 하위 종목

            top = [{ # 상위 종목 정보
                "rank": i + 1, "ticker": t,
                "name": name_map.get(t, t),
                "score": float(sig[t] * 100.0),
                "weight": float(long_w.get(t, 0.0))
            } for i, t in enumerate(top_idx)]

            bottom = [{ # 하위 종목 정보
                "rank": i + 1, "ticker": t,
                "name": name_map.get(t, t),
                "score": float(sig[t] * 100.0),
                "weight": float(short_w.get(t, 0.0))
            } for i, t in enumerate(bot_idx)]

            net_w = long_w if long_only else (long_w - short_w) # 순 가중치
            out = { # 출력 딕셔너리
                "asof_date": pd.Timestamp(last_date).strftime("%Y-%m-%d"),
                "top": top,
                "bottom": [] if long_only else bottom,
                "today_weights": {k: float(v) for k, v in net_w.items()}
            }
            return out
        except Exception as e:
            logger.exception("Failed to build picks: %s", e)
            return out

    # 하이퍼파라미터 자동 탐색
    def _score_config(self, ic: float, hit: float, weights: Optional[pd.Series]) -> float:
        hit_adj = (float(hit) - 0.5) * 2.0 if np.isfinite(hit) else -1.0 # 적중률 조정
        ic_safe = float(ic) if np.isfinite(ic) else -1.0 # IC 안전값
        div_pen = 0.0 # 다양성 페널티
        if isinstance(weights, pd.Series) and len(weights) > 0: # 가중치가 있으면
            w = weights.fillna(0.0).clip(lower=0.0) # 음수 제거
            w = w / (w.sum() if w.sum() > 0 else 1.0) # 정규화
            hhi = float((w ** 2).sum()) # 허핀달 지수
            n = len(w) # 종목 수
            base = 1.0 / n # 기준값
            div_pen = (hhi - base) / max(1e-9, (1.0 - base)) # 다양성 페널티
        return 0.7 * ic_safe + 0.3 * hit_adj - 0.10 * div_pen # 종합 점수

    def run_meta_tune(self,
                      use_gru: bool = True, # GRU 사용 여부
                      horizon_candidates: Optional[List[int]] = None, # 기간 후보들
                      weight_schemes: Optional[List[str]] = None, # 가중치 방식들
                      long_only_options: Optional[List[bool]] = None, # 롱온리 옵션들
                      hrp_max_candidates: Optional[List[Optional[float]]] = None, # HRP 최대 가중치 후보들
                      hrp_blend_candidates: Optional[List[float]] = None, # HRP 블렌딩 후보들
                      tau_candidates: Optional[List[float]] = None, # 온도 후보들
                      max_trials: int = 20, # 최대 시도 횟수
                      random_state: int = 42, # 랜덤 상태
                      quick: bool = True) -> Dict[str, Any]: # 빠른 모드
        rng = np.random.default_rng(random_state) # 랜덤 생성기

        horizon_candidates = horizon_candidates or [3, 5, 10] # 기간 후보들 기본값
        weight_schemes = weight_schemes or ["hrp", "ivp", "minvar"] # 가중치 방식 기본값
        long_only_options = long_only_options or [False, True] # 롱온리 옵션 기본값
        hrp_max_candidates = hrp_max_candidates or [None, 0.35, 0.25] # HRP 최대 가중치 기본값
        hrp_blend_candidates = hrp_blend_candidates or [0.0, 0.2, 0.3, 0.5] # HRP 블렌딩 기본값
        tau_candidates = tau_candidates or [0.7, 0.85, 1.0] # 온도 후보들 기본값

        trials = [] # 시도 결과들
        best = {"score": -1e9} # 최고 결과

        base_auto = self._auto_tune(horizon=5, seq_len=None, epochs=None, hidden_size=None, # 기본 자동 튜닝
                                    dropout=None, hrp_max_weight=None, hrp_blend_to_equal=None,
                                    long_tau=None, short_tau=None)

        for _ in range(max_trials): # 최대 시도 횟수만큼
            h = int(rng.choice(horizon_candidates)) # 랜덤 기간
            scheme = str(rng.choice(weight_schemes)) # 랜덤 가중치 방식
            long_only = bool(rng.choice(long_only_options)) # 랜덤 롱온리
            hrp_max = rng.choice(hrp_max_candidates) # 랜덤 HRP 최대값
            hrp_blend = float(rng.choice(hrp_blend_candidates)) # 랜덤 HRP 블렌딩
            tau = float(rng.choice(tau_candidates)) # 랜덤 온도

            tuned = self._auto_tune( # 파라미터 튜닝
                horizon=h,
                seq_len=None, epochs=None, hidden_size=None, dropout=None,
                hrp_max_weight=(hrp_max if hrp_max is not None else None),
                hrp_blend_to_equal=hrp_blend,
                long_tau=tau, short_tau=tau
            )
            seq_len = tuned["seq_len"] # 시퀀스 길이
            epochs = 8 if quick else tuned["epochs"] # 에폭 수 (빠른 모드)
            hidden = tuned["hidden_size"] # 히든 사이즈
            drop = tuned["dropout"] # 드롭아웃

            try:
                wres = self.hrp_weights( # HRP 가중치 계산
                    scheme=scheme, max_weight=(None if hrp_max is None else float(hrp_max)),
                    min_weight=0.0, blend_to_equal=hrp_blend, resamples=0
                )
                weights = wres.weights # 가중치
            except Exception as e:
                logger.debug("HRP weights failed during meta tune: %s", e)
                weights = None

            try:
                pres = self.predict_next( # 예측 실행
                    horizon=h, use_gru=use_gru,
                    seq_len=seq_len, epochs=epochs, hidden_size=hidden, dropout=drop
                )
                score = self._score_config(pres.ic, pres.hit_rate, weights) # 점수 계산
                trials.append({ # 시도 결과 저장
                    "horizon": h, "weight_scheme": scheme, "long_only": long_only,
                    "hrp_max_weight": (None if hrp_max is None else float(hrp_max)),
                    "hrp_blend_to_equal": float(hrp_blend),
                    "tau": tau, "seq_len": seq_len, "epochs": epochs,
                    "hidden_size": hidden, "dropout": drop,
                    "ic": float(pres.ic), "hit_rate": float(pres.hit_rate), "score": float(score)
                })
                if score > best["score"]: # 최고 점수보다 좋으면
                    best = dict(trials[-1]) # 최고 결과 업데이트
            except Exception as e:
                logger.debug("Meta tune trial failed and was skipped: %s", e)
                continue

        trials_df = pd.DataFrame(trials).sort_values("score", ascending=False).reset_index(drop=True) # 시도 결과 DataFrame
        summary = { # 요약 정보
            "best": best,
            "trials": trials_df.to_dict(orient="records"),
            "n_trials": int(len(trials)),
            "params_default": {
                "base_auto": base_auto,
                "search_space_sizes": {
                    "horizon": len(horizon_candidates),
                    "weight_schemes": len(weight_schemes),
                    "long_only": len(long_only_options),
                    "hrp_max": len(hrp_max_candidates),
                    "hrp_blend": len(hrp_blend_candidates),
                    "tau": len(tau_candidates),
                }
            }
        }
        return summary

# 전체 파이프라인
    def run_all(self,
                k_clusters: int = 4, # 클러스터 개수
                horizon: int = 5, # 예측 기간
                use_gru: bool = True, # GRU 사용 여부
                top_n: int = 5, # 상위 종목 개수
                bottom_n: int = 5, # 하위 종목 개수
                weight_scheme: str = "hrp", # 가중치 방식
                long_only: bool = False, # 롱온리 여부
                seq_len: Optional[int] = None, # 시퀀스 길이
                epochs: Optional[int] = None, # 에폭 수
                hidden_size: Optional[int] = None, # 히든 사이즈
                dropout: Optional[float] = None, # 드롭아웃
                hrp_max_weight: Optional[float] = None, # HRP 최대 가중치
                hrp_min_weight: float = 0.0, # HRP 최소 가중치
                hrp_blend_to_equal: Optional[float] = None, # HRP 균등 블렌딩
                hrp_resamples: int = 0, # HRP 리샘플링
                long_tau: Optional[float] = None, # 롱 온도
                short_tau: Optional[float] = None, # 숏 온도
                meta_tune: bool = False, # 메타 튜닝 여부
                meta_max_trials: int = 20, # 메타 튜닝 최대 시도
                meta_random_state: int = 42, # 메타 튜닝 랜덤 상태
                meta_quick: bool = True, # 메타 튜닝 빠른 모드
                **gru_kwargs) -> Dict[str, Any]: # GRU 추가 인자들
        # 시작 메시지
        self._push_msg(f"ML Analysis started - {self.returns.shape[1]} stocks, {len(self.returns)} days")
        
        meta_result: Optional[Dict[str, Any]] = None # 메타 튜닝 결과 초기화
        if meta_tune: # 메타 튜닝 사용하면
            self._push_msg("Running meta-tuning for optimal parameters...") # 메시지 추가
            meta_result = self.run_meta_tune( # 메타 튜닝 실행
                use_gru=use_gru, max_trials=meta_max_trials,
                random_state=meta_random_state, quick=meta_quick
            )
            if meta_result and meta_result.get("best"): # 최고 결과가 있으면
                b = meta_result["best"] # 최고 결과
                horizon = int(b["horizon"]) # 기간 업데이트
                weight_scheme = str(b["weight_scheme"]) # 가중치 방식 업데이트
                long_only = bool(b["long_only"]) # 롱온리 업데이트
                hrp_max_weight = b["hrp_max_weight"] # HRP 최대 가중치 업데이트
                hrp_blend_to_equal = b["hrp_blend_to_equal"] # HRP 블렌딩 업데이트
                long_tau = b["tau"]; short_tau = b["tau"] # 온도 업데이트
                seq_len = b["seq_len"]; epochs = b["epochs"] # 시퀀스/에폭 업데이트
                hidden_size = b["hidden_size"]; dropout = b["dropout"] # 히든/드롭아웃 업데이트
                self._push_msg(f"Meta-tune best: IC={b.get('ic', 0):.3f}, Hit={b.get('hit_rate', 0):.3f}") # 메시지 추가

        tuned = self._auto_tune( # 자동 튜닝
            horizon=horizon,
            seq_len=seq_len, epochs=epochs, hidden_size=hidden_size, dropout=dropout,
            hrp_max_weight=hrp_max_weight, hrp_blend_to_equal=hrp_blend_to_equal,
            long_tau=long_tau, short_tau=short_tau
        )
        seq_len     = tuned["seq_len"] if seq_len is None else seq_len # 시퀀스 길이 설정
        epochs      = tuned["epochs"]  if epochs  is None else epochs # 에폭 설정
        hidden_size = tuned["hidden_size"] if hidden_size is None else hidden_size # 히든 사이즈 설정
        dropout     = tuned["dropout"] if dropout is None else dropout # 드롭아웃 설정
        if hrp_blend_to_equal is None: hrp_blend_to_equal = tuned["hrp_blend_to_equal"] # HRP 블렌딩 설정
        if long_tau is None: long_tau = tuned["long_tau"] # 롱 온도 설정
        if short_tau is None: short_tau = tuned["short_tau"] # 숏 온도 설정

        # 각 단계별 진행 상황 메시지
        self._push_msg("Running PCA & clustering analysis...") # PCA 클러스터링 메시지
        pca_res = self.pca_2d() # PCA 실행
        clus_res = self.kmeans_clusters(pca_res, k_clusters=min(k_clusters, self.returns.shape[1])) # 클러스터링 실행
        
        self._push_msg(f"Computing {weight_scheme.upper()} portfolio weights...") # 가중치 계산 메시지
        hrp_res  = self.hrp_weights( # HRP 가중치 계산
            scheme=weight_scheme,
            max_weight=hrp_max_weight, min_weight=hrp_min_weight,
            blend_to_equal=hrp_blend_to_equal, resamples=hrp_resamples
        )
        
        model_name = "GRU" if use_gru else "RandomForest" # 모델 이름
        self._push_msg(f"Training {model_name} model (horizon={horizon}, epochs={epochs})...") # 모델 훈련 메시지
        pred_res = self.predict_next( # 예측 실행
            horizon=horizon, use_gru=use_gru,
            seq_len=seq_len, epochs=epochs, hidden_size=hidden_size, dropout=dropout,
            **gru_kwargs
        )
        
        # 결과 메시지 추가
        ic_val = float(pred_res.ic) if np.isfinite(pred_res.ic) else 0.0 # IC 값
        hit_val = float(pred_res.hit_rate) if np.isfinite(pred_res.hit_rate) else 0.0 # 적중률 값
        r2_val  = float(pred_res.r2) if (pred_res.r2 is not None and np.isfinite(pred_res.r2)) else 0.0 # R2 값
        self._push_msg(f"Results: IC={ic_val:.3f}, Hit={hit_val:.3f}, R²={r2_val:.3f}") # 결과 메시지

        # 투자 등급 평가
        if ic_val > 0.10: # IC가 0.10 초과
            self._push_msg("Investment Grade: EXCELLENT - Strong predictive power") # 우수 등급
        elif ic_val > 0.05: # IC가 0.05 초과
            self._push_msg("Investment Grade: GOOD - Acceptable for investment") # 양호 등급
        elif ic_val > 0.02: # IC가 0.02 초과
            self._push_msg("Investment Grade: FAIR - Borderline performance") # 보통 등급
        else: # 그 외
            self._push_msg("Investment Grade: POOR - Not recommended for investment") # 불량 등급

        picks = self._build_picks( # 종목 선택
            pred_res, hrp_res.weights,
            top_n=top_n, bottom_n=bottom_n,
            long_tau=long_tau, short_tau=short_tau,
            long_only=long_only
        )

        explained_array = pca_res.explained_var if pca_res.explained_var is not None else pca_res.explained_variance_ratio_ # 설명된 분산
        explained_list = np.asarray(explained_array).tolist() # 리스트로 변환

        # NumPy 타입을 파이썬 기본 타입으로 변환
        # PCA 좌표 변환
        pca_coords_records = pca_res.components.reset_index().rename(columns={"index": "ticker"}).to_dict(orient="records")
        cleaned_pca_coords = [
            {k: v if isinstance(v, str) else float(v) for k, v in row.items()}
            for row in pca_coords_records
        ]

        # 클러스터 중심점 변환
        centers_records = clus_res.centers.to_dict(orient="records")
        cleaned_centers = [
            {k: float(v) for k, v in row.items()}
            for row in centers_records
        ]
        
        # HRP 가중치 변환
        cleaned_hrp_weights = {
            k: float(v) for k, v in hrp_res.weights.sort_values(ascending=False).to_dict().items()
        }

        out: Dict[str, Any] = {
            "meta": {
                "universe_size": int(self.returns.shape[1]),
                "n_days": int(len(self.returns)),
                "tuned": {
                    "seq_len": seq_len, "epochs": epochs, "hidden_size": hidden_size, "dropout": dropout,
                    "hrp_max_weight": hrp_max_weight, "hrp_blend_to_equal": hrp_blend_to_equal,
                    "long_tau": long_tau, "short_tau": short_tau,
                    "weight_scheme": weight_scheme, "long_only": long_only,
                    "horizon": horizon
                },
                "meta_tune": meta_result or {}
            },
            "pca": {
                "coords": cleaned_pca_coords,
                "explained_var": [float(v) for v in explained_list]
            },
            "clusters": {
                "labels": {k: int(v) for k, v in self._safe_dict(clus_res.labels).items()},
                "centers": cleaned_centers
            },
            "hrp": {
                "weights": cleaned_hrp_weights,
                "order": hrp_res.order
            },
            "prediction": {
                "horizon": int(pred_res.horizon),
                "ic": float(pred_res.ic) if np.isfinite(pred_res.ic) else 0.0,
                "hit_rate": float(pred_res.hit_rate) if np.isfinite(pred_res.hit_rate) else 0.0,
                "r2": float(pred_res.r2) if pred_res.r2 is not None and np.isfinite(pred_res.r2) else None,
                "ic_by_date": {pd.Timestamp(k).strftime("%Y-%m-%d"): float(v) 
                    for k, v in pred_res.ic_by_date.items()},
                # NumPy 타입을 Python float으로 변환하여 JSON 직렬화 해결
                "preds_vs_real": (
                    [
                        {
                            k: (v.isoformat() if isinstance(v, pd.Timestamp) else float(v) if isinstance(v, (np.number, float, int)) else v)
                            for k, v in record.items()
                        }
                        for record in pred_res.preds_vs_real.reset_index().to_dict('records')
                    ]
                    if pred_res.preds_vs_real is not None else None
                )
            },
            "picks": picks,
            "today_weights": picks.get("today_weights", {}),
            "ui_messages": list(getattr(self, "ui_messages", [])),
            "preds_df": pred_res.preds_vs_real.reset_index() if pred_res and hasattr(pred_res, 'preds_vs_real') else None
        }
        return out

    @staticmethod
    def _safe_dict(s: pd.Series) -> Dict[str, Any]: # 안전하게 딕셔너리 변환
        try:
            return s.to_dict() # 딕셔너리로 변환
        except Exception as e:
            logger.debug("_safe_dict fell back to enumerate: %s", e)
            return {str(i): float(v) for i, v in enumerate(s)} # 인덱스로 변환
    
    # ml_charts.py 호환용 alias
    def run_ml_analysis(self, **kwargs) -> Dict[str, Any]:
        # run_all의 alias 
        return self.run_all(**kwargs)
    
# 데이터 품질 자동 분석 클래스 
class AdaptiveMLAnalyzer(MLAnalyzer):
    def auto_detect_data_quality(self) -> Dict[str, Any]:
        # 데이터 품질 자동 평가 및 메시지 생성
        # 상관관계 계산
        corr_matrix = self.returns.corr() # 상관계수 행렬
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean() # 평균 상관계수

        # PCA 분석
        if self.returns.shape[1] >= 3: # 종목이 3개 이상이면
            pca = PCA(n_components=min(3, self.returns.shape[1])) # PCA 객체
            pca.fit(self.returns.T)  # 종목=샘플, 날짜=피처
            explained_ratio = float(pca.explained_variance_ratio_[0]) # 첫 번째 주성분 설명력
        else:
            explained_ratio = 0.9  # 종목 수 부족 가정
        
        # 데이터 충분성 계산
        n_samples = len(self.returns) # 샘플 수
        n_features = self.returns.shape[1] # 피처 수
        sample_per_feature = n_samples / max(1, n_features) # 피처당 샘플 수

        # 변동성 분산도 계산
        vol_dispersion = self.returns.std().std() / (self.returns.std().mean() + 1e-8) # 변동성의 변동성

        # 점수화
        quality_score = { # 품질 점수 딕셔너리
            "correlation": float(avg_corr),
            "diversity": float(1 - explained_ratio),
            "data_sufficiency": float(min(1.0, sample_per_feature / 100)),
            "volatility_dispersion": float(min(1.0, vol_dispersion * 2)),
        }
        total = ( # 총점 계산
            (1 - abs(quality_score["correlation"])) * 0.30
            + quality_score["diversity"] * 0.30
            + quality_score["data_sufficiency"] * 0.25
            + quality_score["volatility_dispersion"] * 0.15
        )
        quality_score["total"] = float(total) # 총점 저장

        # 진단 및 권고사항
        issues, recommendations = [], [] # 문제점과 권고사항 리스트
        if avg_corr > 0.7: # 평균 상관계수가 높으면
            issues.append(f"평균 상관계수 {avg_corr:.2f} (과도하게 높음)")
            recommendations.append("- 섹터 분산 강화 또는 팩터 모델 고려")
        if explained_ratio > 0.8: # 첫 번째 주성분 설명력이 높으면
            issues.append(f"PC1 설명력 {explained_ratio*100:.1f}% (다양성 부족)")
            recommendations.append("- 종목 차별화 부족: 개별 예측 난도 높음")
        if sample_per_feature < 50: # 피처당 샘플이 부족하면
            issues.append(f"종목당 표본 {sample_per_feature:.0f}일 (부족)")
            recommendations.append(f"- 최소 {100 * n_features}일 이상 확보 권장")
        if vol_dispersion < 0.3: # 변동성 분산도가 낮으면
            issues.append("변동성 분산도 낮음 (종목 간 변동성 유사)")
            recommendations.append("- 변동성 기반 전략 효과 제한적")

        # 성능 기대치
        if total >= 0.7: # 총점이 0.7 이상
            performance_msg = "예상 IC: 0.10+ (우수)"
            confidence = "HIGH"
        elif total >= 0.5: # 총점이 0.5 이상
            performance_msg = "예상 IC: 0.05~0.10 (양호)"
            confidence = "MEDIUM"
        elif total >= 0.3: # 총점이 0.3 이상
            performance_msg = "예상 IC: 0.02~0.05 (미흡)"
            confidence = "LOW"
        else: # 그 외
            performance_msg = "예상 IC: <0.02 (무의미)"
            confidence = "VERY LOW"

        # 텍스트 메시지 생성
        msg_lines = [ # 메시지 라인들
            "DATA QUALITY REPORT",
            f"Quality score : {total:.2f} / 1.00",
            f"Confidence    : {confidence}",
            f"{performance_msg}",
            "",
            "Details:",
            f"- Independence (1 - corr): {(1-avg_corr):.2f}",
            f"- Diversity (1 - PC1)   : {quality_score['diversity']:.2f}",
            f"- Sufficiency           : {quality_score['data_sufficiency']:.2f}",
            f"- Volatility dispersion : {quality_score['volatility_dispersion']:.2f}",
        ]
        if issues: # 문제점이 있으면
            msg_lines += ["", "Issues:"] + [f"{it}" for it in issues]
        if recommendations: # 권고사항이 있으면
            msg_lines += ["", "Recommendations:"] + [f"{it}" for it in recommendations]
        quality_score["message"] = "\n".join(msg_lines) # 메시지 결합
        quality_score["issues"] = issues # 문제점 저장
        quality_score["recommendations"] = recommendations # 권고사항 저장
        quality_score["confidence"] = confidence # 신뢰도 저장

        # 하단 렌더용 요약 메시지 추가
        summary_line = ( # 요약 라인
            f"DATA QUALITY total={quality_score['total']:.2f} "
            f"(corr={quality_score['correlation']:.2f}, diversity={quality_score['diversity']:.2f}, "
            f"suff={quality_score['data_sufficiency']:.2f}, volDisp={quality_score['volatility_dispersion']:.2f})"
        )
        self._push_msg(summary_line) # 요약 메시지 추가
        if issues: # 문제점이 있으면
            self._push_msg("Issues: " + " | ".join(issues[:2])) # 문제점 메시지 (최대 2개)
        if recommendations: # 권고사항이 있으면
            self._push_msg("Recommendations: " + " | ".join(recommendations[:2])) # 권고사항 메시지 (최대 2개)

        return quality_score

    def adaptive_model_selection(self) -> Dict[str, Any]:
        # 데이터 특성에 따른 모델 자동 선택
        q = self.auto_detect_data_quality() # 품질 분석
        if q["total"] < 0.3: # 품질이 낮으면
            cfg = { # 설정
                "model": "RandomForest",
                "reason": "품질 낮음: 단순 모델이 유리",
                "params": {"n_estimators": 100, "max_depth": 5, "note": "과적합 방지용 경량 RF"},
            }
        elif q["correlation"] > 0.7: # 상관관계가 높으면
            cfg = { # 설정
                "model": "FactorModel",
                "reason": "상관 높음: 공통 팩터 추출 유리",
                "params": {"n_factors": 3, "use_pca": True, "note": "PCA 기반 팩터"},
            }
        elif q["data_sufficiency"] < 0.5: # 데이터가 부족하면
            cfg = { # 설정
                "model": "SimpleLSTM",
                "reason": "표본 부족: 경량 순환모델",
                "params": {"hidden_size": 32, "num_layers": 1, "note": "연산 가벼움"},
            }
        else: # 그 외
            cfg = { # 설정
                "model": "GRU",
                "reason": "품질 양호: GRU 적합",
                "params": {"hidden_size": 64, "num_layers": 2, "note": "성능/안정 균형"},
            }

        lines = [ # 메시지 라인들
            "MODEL AUTO-SELECTION",
            f"Selected : {cfg['model']}",
            f"Reason   : {cfg['reason']}",
            "Params   : " + ", ".join([f"{k}={v}" for k, v in cfg["params"].items() if k != "note"]),
            f"Note     : {cfg['params'].get('note','')}",
            "",
            "Expected:",
            f"- Accuracy tier : {q['confidence']}",
            f"- Overfit risk  : {'Low' if q['data_sufficiency'] > 0.7 else 'Medium' if q['data_sufficiency'] > 0.4 else 'High'}",
        ]
        cfg["full_message"] = "\n".join(lines) # 전체 메시지

        # 하단 렌더용
        self._push_msg(f"MODEL SELECTED: {cfg['model']} - {cfg['reason']}") # 모델 선택 메시지
        return cfg

    def run_smart_analysis(self, **kwargs) -> Dict[str, Any]:
        # 품질 체크 후 자동 모델 선택 -> 분석 실행
        q = self.auto_detect_data_quality() # 품질 분석
        print(q["message"]) # 품질 메시지 출력
        self._push_msg("Smart analysis started") # 스마트 분석 시작 메시지

        if q["total"] < 0.25: # 품질이 너무 낮으면
            stop_msg = ( # 중단 메시지
                "ANALYSIS STOPPED: data quality too low.\n"
                "- Extend history (>= 2 years)\n"
                "- Increase universe size (>= 10)\n"
                "- Diversify sectors\n"
                f"Current quality: {q['total']:.2f} / 1.00 (min 0.25)"
            )
            self._push_msg("Stopped: quality below threshold") # 중단 메시지
            return { # 실패 결과 반환
                "success": False,
                "message": stop_msg,
                "quality_score": q,
                "ui_messages": list(self.ui_messages),
            }

        cfg = self.adaptive_model_selection() # 모델 선택
        print(cfg["full_message"]) # 모델 선택 메시지 출력

        print("Starting analysis...") # 분석 시작 출력
        self._push_msg("Running analysis...") # 분석 실행 메시지

        if cfg["model"] == "RandomForest": # RF 모델이면
            result = self._predict_with_rf(horizon=kwargs.get("horizon", 5)) # RF로 예측
            result_dict = { # 결과 딕셔너리
                "prediction": {
                    "horizon": result.horizon,
                    "ic": result.ic,
                    "hit_rate": result.hit_rate,
                    "r2": result.r2,
                    "ic_by_date": {pd.Timestamp(k).strftime("%Y-%m-%d"): float(v) for k, v in result.ic_by_date.items()},
                }
            }
        else: # 다른 모델이면
            result_dict = self.run_all(**kwargs) # 전체 분석 실행

        result_dict["quality_analysis"] = { # 품질 분석 결과 추가
            "score": q["total"],
            "confidence": q["confidence"],
            "issues": q["issues"],
            "recommendations": q["recommendations"],
        }
        result_dict["model_used"] = cfg # 사용된 모델 정보

        ic_val = result_dict.get("prediction", {}).get("ic", 0.0) or 0.0 # IC 값
        hit_val = result_dict.get("prediction", {}).get("hit_rate", 0.0) or 0.0 # 적중률 값
        self._push_msg(f"RESULT: IC={ic_val:.3f}, Hit={hit_val:.3f}, Horizon={kwargs.get('horizon', 5)}") # 결과 메시지

        if ic_val > 0.05: # IC가 0.05 초과면
            print( # 결과 출력
                "ANALYSIS RESULT\n"
                f"IC (test): {ic_val:.3f}\n"
                "Decision : investable range" if ic_val > 0.05 else "Decision : borderline"
            )
        else: # 그 외
            print( # 저성능 결과 출력
                "ANALYSIS RESULT (LOW PERFORMANCE)\n"
                f"IC (test): {ic_val:.3f}\n"
                "Potential causes: " + (", ".join(q["issues"][:2]) if q["issues"] else "data characteristics") + "\n"
                "Next steps:\n" + ("\n".join(q["recommendations"][:3]) if q["recommendations"] else "augment data")
            )

        # 하단 메시지 포함
        result_dict["ui_messages"] = list(self.ui_messages) # UI 메시지 추가
        return result_dict

    # 품질 분석과 안정성 분석을 포함한 전체 ML 분석
    def run_all_with_analysis(self, **kwargs) -> Dict[str, Any]:
        # 확장된 분석 - 품질, 안정성 포함
        # 기본 run_all 실행
        result = self.run_all(**kwargs) # 기본 분석 실행
        
        # 품질 분석 추가
        quality = self.auto_detect_data_quality() # 품질 분석
        
        # Walk-forward 분석 추가
        walk_forward = walk_forward_analysis( # 워크포워드 분석
            self,
            horizon=kwargs.get('horizon', 5),
            use_gru=kwargs.get('use_gru', True),
            quick=True
        )
        
        # 결과에 분석 정보 추가
        result['quality_analysis'] = { # 품질 분석 추가
            'score': quality['total'],
            'confidence': quality['confidence'],
            'issues': quality['issues'],
            'recommendations': quality['recommendations'],
            'details': {
                'correlation': quality['correlation'],
                'diversity': quality['diversity'],
                'data_sufficiency': quality['data_sufficiency'],
                'volatility_dispersion': quality['volatility_dispersion']
            }
        }
        
        result['stability_analysis'] = walk_forward # 안정성 분석 추가
        
        # 콘솔 출력용 요약 정보 추가
        r2_val = result.get('prediction', {}).get('r2', None) # R2 값
        r2_txt = f"{r2_val:.3f}" if (r2_val is not None) else "N/A" # R2 텍스트
        result['analysis_summary'] = f"""
데이터 품질 분석 결과                              
품질 점수: {quality['total']:.2f}/1.00
신뢰도: {quality['confidence']}
예상 IC: {quality.get('performance_msg', 'N/A')}

세부 점수:
- 종목 독립성: {(1-quality['correlation']):.2f}/1.00
- 종목 다양성: {quality['diversity']:.2f}/1.00
- 데이터 충분: {quality['data_sufficiency']:.2f}/1.00
- 변동성 분산: {quality['volatility_dispersion']:.2f}/1.00

안정성 분석 결과                                

{walk_forward.get('summary', 'Walk-forward 분석 실패')}

예측 성능 결과                                  

실제 IC: {result.get('prediction', {}).get('ic', 0):.3f}
적중률: {result.get('prediction', {}).get('hit_rate', 0)*100:.1f}%
R²: {r2_txt}

투자 가능 여부: {walk_forward.get('investment_grade', 'UNKNOWN')}
{walk_forward.get('message', '')}
        """ # 분석 요약

        # 콘솔에 출력
        print(result['analysis_summary']) # 요약 출력
        
        # 하단 메시지 업데이트
        summary_lines = [ # 요약 라인들
            f"품질: {quality['total']:.2f} ({quality['confidence']})",
            f"IC샤프: {walk_forward['metrics']['ic_sharpe']:.2f}" if walk_forward.get('success') else "안정성 분석 실패",
            f"투자등급: {walk_forward.get('investment_grade', 'N/A')}"
        ]
        self._push_msg(" | ".join(summary_lines)) # 요약 메시지 추가
        
        return result
    
    # ml_charts.py 호환용 alias
    def run_ml_analysis(self, **kwargs) -> Dict[str, Any]:
        # run_all_with_analysis의 alias - 확장 분석 포함
        return self.run_all_with_analysis(**kwargs)

# 유틸리티 함수들 

def walk_forward_analysis(
    analyzer: MLAnalyzer, # 분석기 객체
    *,
    n_windows: int = 10, # 윈도우 개수
    test_ratio: float = 0.2, # 테스트 비율
    horizon: int = 5, # 예측 기간
    use_gru: bool = True, # GRU 사용 여부
    quick: bool = True, # 빠른 모드
    min_train_days: int = 100, # 최소 훈련 일수
    **kwargs
) -> Dict[str, Any]:
    
    returns = analyzer.returns # 수익률 데이터
    n_days = len(returns) # 전체 일수
    n_stocks = returns.shape[1] # 종목 수
    
    if n_days < min_train_days + 20: # 데이터가 부족하면
        return { # 실패 결과 반환
            'success': False,
            'message': f'데이터 부족: {n_days}일 (최소 {min_train_days + 20}일 필요)',
            'metrics': {
                'ic_mean': 0.0,
                'ic_std': 1.0,
                'ic_sharpe': 0.0
            }
        }
    
    total_test_size = int(n_days * 0.4) # 전체 테스트 크기
    window_size = max(1, total_test_size // max(1, n_windows)) # 윈도우 크기
    
    ic_results = [] # IC 결과 리스트
    hit_results = [] # 적중률 결과 리스트
    window_details = [] # 윈도우 세부사항
    
    for i in range(n_windows): # 각 윈도우에 대해
        try:
            test_end = n_days - (i * window_size) # 테스트 끝점
            test_start = test_end - window_size # 테스트 시작점
            train_end = test_start # 훈련 끝점
            train_start = max(0, train_end - min_train_days) # 훈련 시작점
            
            if train_end - train_start < min_train_days: # 훈련 데이터가 부족하면
                continue # 건너뛰기
                
            train_returns = returns.iloc[train_start:train_end] # 훈련 수익률
            temp_analyzer = MLAnalyzer(train_returns) # 임시 분석기
            
            tuned = temp_analyzer._auto_tune( # 자동 튜닝
                horizon=horizon,
                seq_len=None, epochs=None, hidden_size=None, dropout=None,
                hrp_max_weight=None, hrp_blend_to_equal=None,
                long_tau=None, short_tau=None
            )
            
            if quick: # 빠른 모드면
                tuned['epochs'] = 10 # 에폭 10으로 설정
            
            pred_result = temp_analyzer.predict_next( # 예측 실행
                horizon=horizon,
                use_gru=use_gru,
                seq_len=tuned['seq_len'],
                epochs=tuned['epochs'],
                hidden_size=tuned['hidden_size'],
                dropout=tuned['dropout'],
                **kwargs
            )
            
            test_returns = returns.iloc[test_start:test_end] # 테스트 수익률
            if len(test_returns) > horizon: # 테스트 데이터가 충분하면
                ic = pred_result.ic if not np.isnan(pred_result.ic) else 0.0 # IC 값
                hit = pred_result.hit_rate if not np.isnan(pred_result.hit_rate) else 0.5 # 적중률
                
                ic_results.append(ic) # IC 결과 추가
                hit_results.append(hit) # 적중률 결과 추가
                
                window_details.append({ # 윈도우 세부사항 추가
                    'window': i + 1,
                    'train_period': f'{train_start}-{train_end}',
                    'test_period': f'{test_start}-{test_end}',
                    'ic': ic,
                    'hit_rate': hit
                })
                
        except Exception as e:
            logger.debug(f"Window {i} failed: {e}") # 윈도우 실패 로그
            continue
    
    if not ic_results: # IC 결과가 없으면
        return { # 실패 결과 반환
            'success': False,
            'message': '워크포워드 분석 실패',
            'metrics': {
                'ic_mean': 0.0,
                'ic_std': 1.0,
                'ic_sharpe': 0.0
            }
        }
    
    ic_array = np.array(ic_results) # IC 배열
    hit_array = np.array(hit_results) # 적중률 배열
    
    ic_mean = float(np.mean(ic_array)) # IC 평균
    ic_std = float(np.std(ic_array)) if float(np.std(ic_array)) > 0 else 1e-9 # IC 표준편차
    ic_sharpe = ic_mean / ic_std if ic_std > 0 else 0.0 # IC 샤프 비율
    positive_ic_ratio = float(np.mean(ic_array > 0)) # 양수 IC 비율
    
    cv = ic_std / abs(ic_mean) if abs(ic_mean) > 0.01 else float('inf') # 변동계수
    
    if ic_sharpe >= 2.0 and positive_ic_ratio >= 0.7: # 매우 좋은 성능
        investment_grade = "EXCELLENT"
        message = "실전 투자 가능 - 안정적인 수익 예상"
    elif ic_sharpe >= 1.0 and positive_ic_ratio >= 0.6: # 좋은 성능
        investment_grade = "GOOD"
        message = "투자 가능 - 적절한 리스크 관리 필요"
    elif ic_sharpe >= 0.5 and positive_ic_ratio >= 0.5: # 보통 성능
        investment_grade = "FAIR"
        message = "조건부 투자 가능 - 추가 검증 필요"
    else: # 나쁜 성능
        investment_grade = "POOR"
        message = "실전 사용 불가 - 모델 재검토 필요"
    
    result = { # 결과 딕셔너리
        'success': True,
        'investment_grade': investment_grade,
        'message': message,
        'metrics': { # 지표들
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_sharpe': ic_sharpe,
            'positive_ic_ratio': positive_ic_ratio,
            'hit_rate_mean': float(np.mean(hit_array)) if len(hit_array) else float('nan'),
            'market_regime_stability': 1.0 / (1.0 + cv),
            'n_windows_tested': len(ic_results)
        },
        'window_details': window_details,
        'summary': f"""
안정성 분석 결과:
- 평균 IC: {ic_mean:.3f} {'(양호)' if ic_mean > 0.05 else '(미흡)'}
- IC 표준편차: {ic_std:.3f} {'(안정적)' if ic_std < 0.1 else '(불안정)'}
- IC 샤프: {ic_sharpe:.2f} {'(매우 좋음)' if ic_sharpe > 2 else '(좋음)' if ic_sharpe > 1 else '(보통)' if ic_sharpe > 0.5 else '(나쁨)'}
- 양수 IC 비율: {positive_ic_ratio*100:.0f}% {'(대부분 기간에서 수익)' if positive_ic_ratio > 0.7 else '(변동적)' if positive_ic_ratio > 0.5 else '(불안정)'}
- 시장 국면별 편차: {cv:.2f} {'(국면 무관하게 작동)' if cv < 0.5 else '(국면 의존적)'}
→ {message}
        """ # 요약 정보
    }
    
    return result

# 간편 품질 체크
def quick_quality_check(returns: pd.DataFrame) -> float:
    # 품질 점수만 빠르게 출력
    analyzer = AdaptiveMLAnalyzer(returns) # 적응형 분석기 생성
    quality = analyzer.auto_detect_data_quality() # 품질 분석
    print(quality["message"]) # 품질 메시지 출력
    return quality["total"] # 총점 반환

# 데이터셋 품질 비교
def compare_multiple_datasets(datasets: Dict[str, pd.DataFrame], quiet: bool = False) -> pd.DataFrame:
    # 여러 섹터/바스켓 returns를 받아 품질 점수를 비교
    results = [] # 결과 리스트
    for name, rets in datasets.items(): # 각 데이터셋에 대해
        analyzer = AdaptiveMLAnalyzer(rets) # 적응형 분석기 생성
        q = analyzer.auto_detect_data_quality() # 품질 분석
        results.append({ # 결과 추가
            "섹터": name,
            "종목수": rets.shape[1],
            "기간(일)": len(rets),
            "품질점수": float(q["total"]),
            "신뢰도": q["confidence"],
            "예상IC": ("0.10+" if q["total"] > 0.7 else "0.05~0.10" if q["total"] > 0.5 else "<0.05"),
            "투자가능": ("YES" if q["total"] > 0.5 else "Maybe" if q["total"] > 0.3 else "NO"),
        })
    df = pd.DataFrame(results).sort_values("품질점수", ascending=False).reset_index(drop=True) # DataFrame 생성
    if not quiet: # 조용한 모드가 아니면
        print("DATASET QUALITY COMPARISON") # 비교 제목 출력
        print(df.to_string(index=False)) # DataFrame 출력
    return df