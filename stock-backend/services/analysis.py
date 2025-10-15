# 데이터 수집과 상관 분석과 포트폴리오 계산과 ML 파이프라인을 담당

import warnings
warnings.filterwarnings('ignore') # 경고 메시지 숨김

import sys
import os

import numpy as np # 수치 계산용
import pandas as pd # 데이터프레임 처리용
import yfinance as yf # 야후 파이낸스 데이터
import FinanceDataReader as fdr # 한국 주식 데이터

from typing import List, Optional, Tuple # 타입 힌트용
from sklearn.cluster import KMeans # K-means 클러스터링
from sklearn.preprocessing import StandardScaler # 데이터 정규화

from services.data import stock_manager # 종목 정보 관리
from services.ml import MLAnalyzer # ML 분석 모듈


from utils.stats import rolling_corr_with_ci
from services.factors import neutralize_to_factors, build_factor_matrix

# 백테스팅 모듈 임포트
try:
    from services.backtest import long_short_backtest, BTResult
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from backtest import long_short_backtest, BTResult
    except:
        print("Warning: Backtest module not found. Backtest features will be disabled.")
        long_short_backtest = None
        BTResult = None

# NumPy/Pandas 타입을 JSON 직렬화 가능한 파이썬 기본 타입으로 변환
def sanitize_for_json(data):

    # 1. 묶음 데이터를 가장 먼저 처리
    if isinstance(data, dict):
        return {str(k): sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, pd.DataFrame):
        return sanitize_for_json(data.to_dict(orient='records'))
    if isinstance(data, pd.Series):
        return sanitize_for_json(data.to_dict())
    if isinstance(data, np.ndarray):
        return [sanitize_for_json(item) for item in data.tolist()]
    if isinstance(data, (list, tuple, set)):
        return [sanitize_for_json(item) for item in data]

    # 2. 단일 값을 처리
    # None 또는 기본 타입은 그대로 반환
    if data is None or isinstance(data, (bool, int, str)):
        return data
        
    # Pandas의 결측치 확인
    if pd.isna(data):
        return None
        
    # NumPy 숫자 타입들 처리
    if isinstance(data, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
        return int(data)
    if isinstance(data, (np.floating, np.float64, np.float16, np.float32)):
        value = float(data)
        return value if np.isfinite(value) else None
    if isinstance(data, np.bool_):
        return bool(data)
        
    # Python float 처리
    if isinstance(data, float):
        return data if np.isfinite(data) else None
        
    # Pandas Timestamp 처리
    if isinstance(data, pd.Timestamp):
        return data.isoformat()
    
    # 3. 위에서 처리되지 않은 기타 객체는 문자열로 변환 시도
    try:
        if hasattr(data, '__dict__'):
            return sanitize_for_json(vars(data))
        return str(data)
    except:
        return None

class KoreanStockCorrelationAnalysis:
    def __init__(self):
        self.stock_data: pd.DataFrame = pd.DataFrame() # 종가 데이터
        self.returns: pd.DataFrame = pd.DataFrame() # 일간 수익률
        self.fx_data: pd.Series = pd.Series(dtype=float) # 환율 데이터
        self.market_data: pd.Series = pd.Series(dtype=float) # 코스피 지수
        self.ml_results: dict = {} # ML 분석 결과
        self.backtest_results: dict = {} # 백테스팅 결과
        self.static_corr = pd.DataFrame() # 정적 상관관계
        self.rolling_corr_matrix = None # 롤링 상관관계
        self.avg_corr = pd.Series() # 평균 상관관계
        self.market_volatility = pd.Series() # 시장 변동성

    def collect_data(
        self,
        start_date: str = '2023-01-01',
        end_date: str = '2024-12-31',
        tickers: Optional[List[str]] = None
    ) -> Tuple[bool, List[dict]]:
        # 데이터 수집 메서드
        if not tickers: # 티커가 없으면 기본값 사용
            default_stocks = stock_manager._get_default_stocks()
            tickers = [stock['ticker'] for stock in default_stocks]

        all_stocks = stock_manager.get_all_stocks() # 전체 종목 정보
        stock_info_map = {stock['ticker']: stock['name'] for stock in all_stocks} # 티커-이름 매핑

        stock_data = {} # 수집된 데이터 저장
        collection_status: List[dict] = [] # 수집 상태 기록

        for ticker in tickers:  # 각 종목별로 데이터 수집
            try:
                name = stock_info_map.get(ticker, ticker) # 종목 이름
                print(f"Collecting {name} ({ticker})...") # 진행 상황 출력

                collected = False # 수집 성공 플래그
                # FDR로 먼저 시도
                try:
                    code = ticker.split('.')[0] # 종목 코드 추출
                    df = fdr.DataReader(code, start_date, end_date) # FDR 데이터 읽기

                    if len(df) > 50: # 최소 데이터 길이 확인
                        df.index = pd.to_datetime(df.index) # 인덱스를 날짜 형식으로
                        if hasattr(df.index, 'tz'): # 타임존 있으면
                            df.index = df.index.tz_localize(None) # 타임존 제거
                        stock_data[ticker] = df['Close'] # 종가 데이터 저장
                        collection_status.append({
                            "ticker": ticker,
                            "name": name,
                            "status": "success",
                            "days": len(df),
                            "source": "FinanceDataReader"
                        })
                        collected = True # 수집 성공
                        continue
                except Exception as fdr_error:
                    print(f"FDR failed for {ticker}: {str(fdr_error)[:50]}") # FDR 실패 로그

                # yfinance로 재시도
                if not collected: # FDR 실패시
                    code = ticker.split('.')[0] # 종목 코드 추출
                    for test_ticker in [ticker, code + '.KS', code + '.KQ', code]: # 여러 형식 시도
                        try:
                            df = yf.download( # yfinance로 다운로드
                                test_ticker,
                                start=start_date,
                                end=end_date,
                                progress=False,
                                show_errors=False
                            )
                            if len(df) > 50: # 최소 데이터 길이 확인
                                if hasattr(df.index, 'tz'): # 타임존 있으면
                                    df.index = df.index.tz_localize(None) # 타임존 제거
                                if isinstance(df.columns, pd.MultiIndex): # 멀티인덱스면
                                    df.columns = df.columns.droplevel(1) # 레벨 제거
                                close_col = 'Close' if 'Close' in df.columns else df.columns[0] # 종가 컬럼 찾기
                                stock_data[ticker] = df[close_col] # 종가 데이터 저장
                                collection_status.append({
                                    "ticker": ticker,
                                    "name": name,
                                    "status": "success",
                                    "days": len(df),
                                    "source": f"yfinance ({test_ticker})"
                                })
                                collected = True # 수집 성공
                                break
                        except Exception:
                            continue # 다음 형식 시도

                    if not collected: # 모든 시도 실패
                        collection_status.append({
                            "ticker": ticker,
                            "name": name,
                            "status": "failed",
                            "days": 0,
                            "error": "Could not fetch data from any source"
                        })

            except Exception as e:
                print(f"Error collecting {ticker}: {str(e)[:50]}") # 에러 
                collection_status.append({
                    "ticker": ticker,
                    "name": name,
                    "status": "error",
                    "error": str(e)[:100]
                })

        # 환율 데이터 수집
        try:
            print("Collecting USD/KRW exchange rate...") # 환율 수집 시작
            try:
                fx = fdr.DataReader('USD/KRW', start_date, end_date) # FDR로 환율 데이터
                if len(fx) > 0: # 데이터 있으면
                    self.fx_data = fx['Close'] # 종가 저장
                    self.fx_data.name = 'USD/KRW' # 이름 설정
                    print("FX data collected successfully from FDR")
            except Exception:
                for fx_symbol in ['USDKRW=X', 'KRW=X']: # yfinance 시도
                    try:
                        fx = yf.download(fx_symbol, start=start_date, end=end_date, progress=False, show_errors=False)
                        if len(fx) > 0: # 데이터 있으면
                            if hasattr(fx.index, 'tz'): # 타임존 제거
                                fx.index = fx.index.tz_localize(None)
                            if 'Close' in fx.columns: # 종가 컬럼 있으면
                                self.fx_data = fx['Close'].squeeze() # 시리즈로 변환
                            else:
                                self.fx_data = fx.iloc[:, 0].squeeze() # 첫 컬럼 사용
                            self.fx_data.name = 'USD/KRW' # 이름 설정
                            print(f"FX data collected successfully from yfinance ({fx_symbol})")
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error collecting FX data: {str(e)[:50]}") # 환율 수집 실패
            self.fx_data = pd.Series(dtype=float) # 빈 시리즈

        # 코스피 지수 수집
        try:
            print("Collecting KOSPI index...") # 코스피 수집 시작
            try:
                kospi = fdr.DataReader('KS11', start_date, end_date) # FDR로 코스피 데이터
                if len(kospi) > 0: # 데이터 있으면
                    self.market_data = kospi['Close'] # 종가 저장
                    self.market_data.name = 'KOSPI' # 이름 설정
                    print("KOSPI data collected successfully from FDR")
            except Exception:
                for kospi_symbol in ['^KS11', 'KOSPI', '^KOSPI', 'KS11']: # yfinance 시도
                    try:
                        kospi = yf.download(kospi_symbol, start=start_date, end=end_date, progress=False, show_errors=False)
                        if len(kospi) > 0: # 데이터 있으면
                            if hasattr(kospi.index, 'tz'): # 타임존 제거
                                kospi.index = kospi.index.tz_localize(None)
                            if 'Close' in kospi.columns: # 종가 컬럼 있으면
                                self.market_data = kospi['Close'].squeeze() # 시리즈로 변환
                            else:
                                self.market_data = kospi.iloc[:, 0].squeeze() # 첫 컬럼 사용
                            self.market_data.name = 'KOSPI' # 이름 설정
                            print(f"KOSPI data collected successfully from yfinance ({kospi_symbol})")
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error collecting KOSPI data: {str(e)[:50]}") # 코스피 수집 실패
            self.market_data = pd.Series(dtype=float) # 빈 시리즈

        if stock_data: # 수집된 데이터가 있으면
            self.stock_data = pd.DataFrame(stock_data) # 데이터프레임으로 변환
            thresh_val = int(max(1, np.floor(len(self.stock_data.columns) * 0.5))) # 임계값 계산
            self.stock_data = self.stock_data.dropna(thresh=thresh_val) # 결측치 많은 행 제거
            self.returns = self.stock_data.pct_change().dropna() # 수익률 계산

            print(f"Data collection completed: {len(self.stock_data)} days, {len(self.stock_data.columns)} stocks")
            return True, collection_status # 성공 반환

        print("No data collected")
        return False, collection_status # 실패 반환

    # 상관관계 분석
    def analyze_correlation(self, window: int = 60) -> None:
        if len(self.returns) < window: # 데이터가 윈도우보다 짧으면
            print(f"Warning: Not enough data for window size {window}. Using {len(self.returns)} days.")
            window = max(20, len(self.returns) // 3) # 윈도우 크기 조정

        self.static_corr = self.returns.corr() # 전체 기간 상관관계
        self.rolling_corr_matrix = self.returns.rolling(window).corr() # 롤링 상관관계
        self.avg_corr = self.rolling_corr_matrix.groupby(level=0).apply( # 평균 상관관계
            lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean() # 상삼각 행렬 평균
        )
        self.market_volatility = self.returns.mean(axis=1).rolling(20).std() * np.sqrt(252) # 시장 변동성

    def calculate_portfolio_metrics(self) -> None:
        # 포트폴리오 지표 계산
        if self.returns.empty: # 데이터 없으면
            self.equal_weight_returns = pd.Series(dtype=float) # 빈 시리즈
            self.min_var_returns = pd.Series(dtype=float)
            self.max_sharpe_returns = pd.Series(dtype=float)
            return

        self.equal_weight_returns = self.returns.mean(axis=1) # 동일가중 수익률

        cov_matrix = self.returns.cov() # 공분산 행렬
        inv_cov = np.linalg.pinv(cov_matrix.values) # 의사역행렬
        ones = np.ones(len(cov_matrix)) # 1 벡터
        weights = inv_cov @ ones / (ones @ inv_cov @ ones) # 최소분산 가중치
        self.min_var_returns = (self.returns @ weights) # 최소분산 수익률
        self.min_var_weights = dict(zip(self.returns.columns, weights)) # 가중치 저장

        inv_vol = 1 / self.returns.std() # 변동성의 역수
        sharpe_weights = inv_vol / inv_vol.sum() # 샤프 가중치
        self.max_sharpe_returns = (self.returns * sharpe_weights).sum(axis=1) # 최대샤프 수익률
        self.max_sharpe_weights = dict(zip(self.returns.columns, sharpe_weights))# 가중치 저장

        if hasattr(self, 'avg_corr') and len(self.avg_corr) > 0: # 평균 상관관계 있으면
            dynamic_weights = 1 - self.avg_corr # 역상관 가중치
            dynamic_weights = dynamic_weights.clip(0.3, 1.0) # 범위 제한
            self.dynamic_returns = self.equal_weight_returns[dynamic_weights.index] * dynamic_weights # 동적 수익률

    # 클러스터링
    def perform_clustering(self, n_clusters: int = 3) -> dict:
        try:
            if self.returns.empty: # 데이터 없으면
                return {}

            features = pd.DataFrame({ # 특징 데이터 생성
                'return': self.returns.mean() * 252, # 연간 수익률
                'volatility': self.returns.std() * np.sqrt(252), # 연간 변동성
                'sharpe': (self.returns.mean() * 252) / (self.returns.std() * np.sqrt(252)) # 샤프비율
            }).replace([np.inf, -np.inf], np.nan).fillna(0.0) # 무한대 제거

            scaler = StandardScaler() # 정규화 객체
            features_scaled = scaler.fit_transform(features) # 특징 정규화

            k = int(max(1, min(n_clusters, len(features)))) # 클러스터 수 제한
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') # K-means 모델
            clusters = kmeans.fit_predict(features_scaled) # 클러스터링 수행

            stock_names = {s['ticker']: s['name'] for s in stock_manager.get_all_stocks()} # 종목 이름

            result = {} # 결과 저장
            for i in range(k): # 각 클러스터별로
                cluster_tickers = features.index[clusters == i].tolist() # 해당 클러스터 종목
                result[f'cluster_{i}'] = {
                    'stocks': [stock_names.get(t, t) for t in cluster_tickers], # 종목 이름
                    'avg_return': float(features.loc[cluster_tickers, 'return'].mean() * 100), # 평균 수익률
                    'avg_volatility': float(features.loc[cluster_tickers, 'volatility'].mean() * 100), # 평균 변동성
                    'avg_sharpe': float(features.loc[cluster_tickers, 'sharpe'].mean()), # 평균 샤프
                    'count': int(len(cluster_tickers)) # 종목 수
                }

            self.ml_results['clustering'] = result # 결과 저장
            return result

        except Exception as e:
            print(f"Clustering error: {e}") # 에러 
            return {}
    
    # 리스크 지표 계산
    def calculate_risk_metrics(self) -> dict:
        try:
            if self.returns.empty: # 데이터 없으면
                return {}

            risk_metrics = {} # 결과 저장
            stock_names = {s['ticker']: s['name'] for s in stock_manager.get_all_stocks()} # 종목 이름

            for ticker in self.returns.columns: # 각 종목별로
                rets = self.returns[ticker].dropna() # 결측치 제거

                var_95 = np.percentile(rets, 5) * 100.0 # 95% VaR
                cvar_95 = rets[rets <= np.percentile(rets, 5)].mean() * 100.0 # 95% CVaR

                skewness = float(rets.skew()) # 왜도
                kurtosis = float(rets.kurt()) # 첨도

                downside = rets[rets < 0.0] # 음수 수익률만
                if len(downside) > 1: # 충분한 데이터 있으면
                    downside_vol = float(downside.std() * np.sqrt(252) * 100.0) # 하방 변동성
                else:
                    downside_vol = float('nan') # 데이터 부족

                risk_metrics[ticker] = {
                    'name': stock_names.get(ticker, ticker), # 종목 이름
                    'var_95': float(var_95), # VaR
                    'cvar_95': float(cvar_95), # CVaR
                    'skewness': skewness, # 왜도
                    'kurtosis': kurtosis, # 첨도
                    'downside_volatility': downside_vol # 하방 변동성
                }

            self.ml_results['risk_metrics'] = risk_metrics # 결과 저장
            return risk_metrics

        except Exception as e:
            print(f"Risk metrics error: {e}") # 에러 출력
            return {}

    # ML 분석 실행
    def run_ml_analysis(
        self,
        k_clusters: int = 4,
        horizon: int = 5,
        use_gru: bool = True,
        top_n: int = 5,
        bottom_n: int = 5,
        **kwargs
    ) -> dict:
        try:
            if self.returns is None or len(self.returns) == 0: # 데이터 없으면
                print("No returns data")
                return {}

            ml = MLAnalyzer(returns=self.returns, prices=self.stock_data) # ML 분석기 생성
            ml_out = ml.run_all( # 전체 분석 실행
                k_clusters=k_clusters,
                horizon=horizon,
                use_gru=use_gru,
                top_n=top_n,
                bottom_n=bottom_n,
                **kwargs
            )

            if 'ui_messages' in ml_out: # Ui 메시지 있으면
                self.ui_messages = ml_out['ui_messages'] # 저장
                print(f"ML Analysis messages: {len(self.ui_messages)} messages captured")

            if not isinstance(self.ml_results, dict): # 딕셔너리가 아니면
                self.ml_results = {} # 초기화
            self.ml_results.update(ml_out) # 결과 업데이트
            return ml_out

        except Exception as e:
            print(f"run_ml_analysis error: {e}") # 에러 출력
            return {}
    
    def run_backtest_analysis(self, ml_predictions=None, top_k=5, bottom_k=5, cost_bps=10.0):
        """백테스팅 분석 실행"""
        if long_short_backtest is None:
            return {
                'success': False,
                'message': '백테스팅 모듈을 찾을 수 없습니다',
                'stats': {}
            }
        
        if ml_predictions is None:
            if not hasattr(self, 'ml_results') or not self.ml_results:
                return {
                    'success': False,
                    'message': 'ML 분석을 먼저 실행하세요',
                    'stats': {}
                }
            ml_predictions = self.ml_results
        
        # 예측 데이터 추출
        pred_table = None
        
        # 1. prediction > preds_vs_real 시도
        if 'prediction' in ml_predictions:
            pred_data = ml_predictions['prediction']
            if isinstance(pred_data, dict) and 'preds_vs_real' in pred_data:
                pred_df = pred_data['preds_vs_real']
                if pred_df is not None and hasattr(pred_df, 'reset_index'):
                    pred_table = pred_df.reset_index()
                    print(f"예측 데이터 찾음: prediction > preds_vs_real")
        
        # 2. 직접 preds_df 확인
        if pred_table is None and 'preds_df' in ml_predictions:
            pred_table = ml_predictions['preds_df']
            print(f"예측 데이터 찾음: preds_df")
        
        # 3. DataFrame 형태 확인 및 정리
        if pred_table is not None:
            # MultiIndex 처리
            if hasattr(pred_table, 'index') and hasattr(pred_table.index, 'names'):
                if pred_table.index.names == ['date', 'ticker']:
                    pred_table = pred_table.reset_index()
            
            # 컬럼명 정리
            if 'level_0' in pred_table.columns:
                pred_table = pred_table.rename(columns={'level_0': 'date'})
            if 'level_1' in pred_table.columns:
                pred_table = pred_table.rename(columns={'level_1': 'ticker'})
            
            # 필수 컬럼 확인
            required = ['date', 'ticker', 'pred']
            if all(col in pred_table.columns for col in required):
                print(f"백테스트 시작: {len(pred_table)} 예측값")
                
                # 백테스트 실행
                try:
                    bt_result = long_short_backtest(
                        returns=self.returns,
                        pred_table=pred_table[required],
                        top_k=top_k,
                        bottom_k=bottom_k,
                        cost_bps=cost_bps
                    )
                    
                    # 결과 정리 - 안전한 타입 변환
                    stats_dict = {}
                    if hasattr(bt_result, 'stats') and bt_result.stats:
                        for k, v in bt_result.stats.items():
                            try:
                                if isinstance(v, (np.number, np.float32, np.float64, np.int32, np.int64)):
                                    stats_dict[k] = float(v)
                                else:
                                    stats_dict[k] = v
                            except:
                                stats_dict[k] = str(v)
                    
                    # summary 정리
                    summary = {}
                    for key, stat_key in [
                        ('annual_return', 'AnnRet'),
                        ('annual_volatility', 'AnnVol'),
                        ('sharpe_ratio', 'Sharpe'),
                        ('max_drawdown', 'MaxDD'),
                        ('cagr', 'CAGR'),
                        ('avg_turnover', 'Turnover')
                    ]:
                        val = bt_result.stats.get(stat_key, 0)
                        try:
                            summary[key] = float(val) if val is not None else 0.0
                        except:
                            summary[key] = 0.0
                    
                    backtest_results = {
                        'success': True,
                        'stats': stats_dict,
                        'summary': summary
                    }
                    
                    # 결과 저장 및 반환
                    self.backtest_results = sanitize_for_json(backtest_results)
                    return self.backtest_results
                    
                except Exception as e:
                    print(f"Backtest error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {
                        'success': False,
                        'message': f'백테스팅 실행 중 오류: {str(e)}',
                        'stats': {}
                    }
        
        # 예측 데이터를 찾지 못한 경우
        print("ERROR: 예측 데이터를 찾을 수 없습니다")
        print(f"ML 결과 키: {list(ml_predictions.keys())}")
        
        return {
            'success': False,
            'message': 'ML 예측 데이터를 찾을 수 없습니다',
            'stats': {}
        }

    # ML 분석과 백테스팅을 동시에 실행
    def run_ml_analysis_with_backtest(self, k_clusters=4, horizon=5, use_gru=True,
                                      top_n=5, bottom_n=5, run_backtest=True,
                                      backtest_top_k=5, backtest_bottom_k=5, 
                                      backtest_cost_bps=10.0, **kwargs):
        ml_results = self.run_ml_analysis( # ML 분석 실행
            k_clusters=k_clusters,
            horizon=horizon,
            use_gru=use_gru,
            top_n=top_n,
            bottom_n=bottom_n,
            **kwargs
        )
        
        if run_backtest and ml_results: # 백테스팅 실행 조건
            print("\n백테스팅 분석 시작")
            
            backtest_results = self.run_backtest_analysis( # 백테스팅 실행
                ml_predictions=ml_results,
                top_k=backtest_top_k,
                bottom_k=backtest_bottom_k,
                cost_bps=backtest_cost_bps
            )
            
            ml_results['backtest'] = backtest_results # ML 결과에 백테스팅 추가
            
            if backtest_results.get('success'): # 백테스팅 성공한다면
                print("백테스팅 결과")
                summary = backtest_results.get('summary', {})
                print(f"연간 수익률: {summary.get('annual_return', 0):.2f}%")
                print(f"연간 변동성: {summary.get('annual_volatility', 0):.2f}%")
                print(f"샤프 비율: {summary.get('sharpe_ratio', 0):.2f}")
                print(f"최대 낙폭: {summary.get('max_drawdown', 0):.2f}%")
                print(f"CAGR: {summary.get('cagr', 0):.2f}%")
                print(f"평균 회전율: {summary.get('avg_turnover', 0):.2%}")
                
                # 종합 판정
                ic = ml_results.get('prediction', {}).get('ic', 0) # IC 점수
                sharpe = summary.get('sharpe_ratio', 0) # 샤프 비율
                
                print("종합 투자 판정")
                
                if ic > 0.05 and sharpe > 0.5: # 우수
                    print("투자 권장 - 예측력과 수익성 모두 양호")
                elif ic > 0.02 and sharpe > 0.3: # 보통
                    print("신중한 투자 - 제한적 수익 가능성")
                else: # 미흡
                    print("투자 비권장 - 리스크 대비 수익 미흡")

        # 반환하기 직전 최종 정제
        return sanitize_for_json(ml_results)

    # 투자 조언 생성
    def generate_investment_advice(self) -> dict:
        advice = {
            "strategies": [],
            "position_sizing": "",
            "timing": "",
            "risk_management": [],
            "specific_actions": [],
            "monitoring_points": []
        }
        
        if not hasattr(self, 'ml_results') or not self.ml_results: # ML 결과 없으면
            advice["strategies"].append("ML 분석을 먼저 실행하세요")
            return advice
            
        ic = self.ml_results.get('prediction', {}).get('ic', 0) # IC 점수
        hit_rate = self.ml_results.get('prediction', {}).get('hit_rate', 0.5) # 적중률
        r2 = self.ml_results.get('prediction', {}).get('r2', 0) # R2 점수
        
        picks = self.ml_results.get('picks', {}) # 종목 선택
        top_stocks = picks.get('top', [])[:3] # 상위 3개
        bottom_stocks = picks.get('bottom', [])[:3] # 하위 3개
        
        hrp_weights = self.ml_results.get('hrp', {}).get('weights', {}) # HRP 가중치
        concentrated_stocks = {k: v for k, v in hrp_weights.items() if v > 0.15} # 집중 종목
        
        # 백테스팅 결과 기반 조언
        if hasattr(self, 'backtest_results') and self.backtest_results.get('success'):
            bt_summary = self.backtest_results['summary'] # 백테스팅 요약
            
            if bt_summary['sharpe_ratio'] > 0.5 and bt_summary['annual_return'] > 0: # 성과 양호
                advice["strategies"].append(f"백테스팅 검증 완료: 연 {bt_summary['annual_return']:.1f}% 수익 달성 가능")
            else: # 성과 미흡
                advice["strategies"].append(f"백테스팅 결과 주의: 샤프 {bt_summary['sharpe_ratio']:.2f}로 위험 대비 수익 미흡")
            
            if bt_summary['max_drawdown'] < -30: # 큰 낙폭
                advice["risk_management"].append(f"백테스팅 최대낙폭 {bt_summary['max_drawdown']:.1f}% - 큰 손실 대비 필요")
            
            if bt_summary['avg_turnover'] > 0.5: # 높은 회전율
                advice["monitoring_points"].append(f"높은 회전율({bt_summary['avg_turnover']:.1%}) - 거래비용 관리 중요")
        
        # IC 기반 전략
        if ic > 0.10: # 매우 우수
            advice["strategies"].extend([
                "적극적 매수 전략: 예측 신뢰도가 매우 높아 공격적 포지션 가능",
                "모멘텀 추종: 상위 예측 종목에 집중 투자"
            ])
            advice["position_sizing"] = "전체 자산의 70-80% 투자 가능"
            advice["timing"] = "즉시 진입 가능"
            
        elif ic > 0.05: # 우수
            advice["strategies"].extend([
                "균형 잡힌 매수 전략: 안정적인 수익 추구",
                "분산 투자: 상위 5-7개 종목에 분산"
            ])
            advice["position_sizing"] = "전체 자산의 50-60% 투자"
            advice["timing"] = "2-3회 분할 매수"
            
        elif ic > 0.02: # 보통
            advice["strategies"].extend([
                "보수적 접근: 소규모 테스트 포지션",
                "페어 트레이딩: 상위/하위 종목 동시 활용"
            ])
            advice["position_sizing"] = "전체 자산의 20-30% 이내"
            advice["timing"] = "시장 조정 시 진입"
            
        else: # 미흡
            advice["strategies"].extend([
                "투자 보류: 현재 전략으로는 수익 창출 어려움",
                "데이터 보강 필요"
            ])
            advice["position_sizing"] = "투자 비권장"
            advice["timing"] = "관망"
        
        # 구체적 액션
        if top_stocks: # 상위 종목 있으면
            for i, stock in enumerate(top_stocks[:3], 1): # 상위 3개
                name = stock.get('name', stock.get('ticker', '')) # 종목명
                score = stock.get('score', 0) # 점수
                weight = stock.get('weight', 0) # 가중치
                advice["specific_actions"].append(
                    f"{i}. {name}: 예측 점수 {score:.1f}, 권장 비중 {weight*100:.1f}%"
                )
        
        # 리스크 관리
        volatility = self.returns.std().mean() * np.sqrt(252) # 평균 변동성
        if volatility > 0.3: # 높은 변동성
            advice["risk_management"].append("높은 변동성: 손절선 -5% 엄격 적용")
        advice["risk_management"].extend([
            f"포트폴리오 최대 낙폭 한도: {max(5, min(15, ic * 100)):.0f}%",
            f"개별 종목 손절선: -{max(3, min(10, ic * 50)):.0f}%"
        ])
        
        # 모니터링 포인트
        advice["monitoring_points"].extend([
            f"IC가 {ic*0.7:.3f} 이하로 하락 시 전략 재검토",
            f"적중률이 {max(0.45, hit_rate*0.9):.1%} 이하 시 포지션 축소"
        ])
        
        return advice

    # ML 분석과 투자 조언 통합
    def get_ml_analysis_with_advice(self) -> dict:
        meta = self.get_ml_frontend_meta() # 프론트엔드 메타 정보
        advice = self.generate_investment_advice() # 투자 조언
        
        result = {
            **meta,
            "investment_advice": advice,
            "summary": self._create_advice_summary(advice)
        }
        
        if hasattr(self, 'backtest_results') and self.backtest_results.get('success'): # 백테스팅 있으면
            result["backtest_summary"] = self.backtest_results['summary'] # 백테스팅 요약 추가
        
        return result
    
    # 조언을 한 문장으로 요약
    def _create_advice_summary(self, advice: dict) -> str:
        if not advice["strategies"]: # 전략 없으면
            return "분석 결과를 기다리는 중입니다."
            
        main_strategy = advice["strategies"][0] if advice["strategies"] else "" # 주요 전략
        position = advice["position_sizing"] # 포지션 크기
        timing = advice["timing"] # 타이밍
        
        return f"{main_strategy} {position} 권장하며, {timing}"

    def get_performance_summary(self) -> dict:
        # 성과 요약 생성
        performance = [] # 개별 종목 성과
        stock_info_map = {stock['ticker']: stock['name'] for stock in stock_manager.get_all_stocks()} # 종목 이름

        for col in self.returns.columns: # 각 종목별로
            rets = self.returns[col].dropna() # 결측치 제거
            if rets.empty: # 데이터 없으면
                continue
            cum_ret = (1 + rets).cumprod() # 누적 수익률
            running_max = cum_ret.expanding().max() # 누적 최대값
            drawdown = (cum_ret - running_max) / running_max # 낙폭

            stock_name = stock_info_map.get(col, col) # 종목 이름

            performance.append({
                "name": stock_name,
                "ticker": col,
                "annual_return": float(rets.mean() * 252 * 100), # 연간 수익률
                "annual_volatility": float(rets.std() * np.sqrt(252) * 100), # 연간 변동성
                "sharpe_ratio": float((rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan), # 샤프비율
                "max_drawdown": float(drawdown.min() * 100), # 최대 낙폭
                "total_return": float((cum_ret.iloc[-1] - 1) * 100) # 총 수익률
            })

        strategies = { # 포트폴리오 전략
            '동일가중': getattr(self, 'equal_weight_returns', pd.Series(dtype=float)),
            '최소분산': getattr(self, 'min_var_returns', pd.Series(dtype=float)),
            '최대샤프': getattr(self, 'max_sharpe_returns', pd.Series(dtype=float))
        }
        if hasattr(self, 'dynamic_returns'): # 동적 전략 있으면
            strategies['동적전략'] = self.dynamic_returns

        portfolio_performance = [] # 포트폴리오 성과
        for name, returns in strategies.items(): # 각 전략별로
            if isinstance(returns, pd.Series) and len(returns) > 0: # 데이터 있으면
                cum_ret = (1 + returns).cumprod() # 누적 수익률
                running_max = cum_ret.expanding().max() # 누적 최대값
                drawdown = ((cum_ret - running_max) / running_max).min() # 최대 낙폭

                vol = returns.std() * np.sqrt(252) # 연간 변동성
                sr = (returns.mean() * 252) / vol if vol > 0 else np.nan # 샤프비율

                portfolio_performance.append({
                    "strategy": name,
                    "annual_return": float(returns.mean() * 252 * 100), # 연간 수익률
                    "annual_volatility": float(vol * 100), # 연간 변동성
                    "sharpe_ratio": float(sr), # 샤프비율
                    "max_drawdown": float(drawdown * 100), # 최대 낙폭
                    "final_value": float(cum_ret.iloc[-1]) # 최종 가치
                })

        result = {
            "individual_stocks": performance,
            "portfolio_strategies": portfolio_performance
        }

        if self.ml_results: # ML 결과 있으면
            result["ml_analysis"] = sanitize_for_json(self.ml_results)
            
        if hasattr(self, 'backtest_results') and self.backtest_results.get('success'): # 백테스팅 있으면
            result["backtest_performance"] = sanitize_for_json({
                "strategy": "ML 롱숏전략",
                "annual_return": self.backtest_results['summary']['annual_return'],
                "annual_volatility": self.backtest_results['summary']['annual_volatility'],
                "sharpe_ratio": self.backtest_results['summary']['sharpe_ratio'],
                "max_drawdown": self.backtest_results['summary']['max_drawdown'],
                "cagr": self.backtest_results['summary']['cagr'],
                "avg_turnover": self.backtest_results['summary']['avg_turnover']
            })

        return sanitize_for_json(result)
    
    # 프론트엔드용 ML 메타 정보
    def get_ml_frontend_meta(self) -> dict:
        try:
            root = self.ml_results if isinstance(self.ml_results, dict) else {} # ML 결과

            stability = root.get('stability_analysis') or root.get('prediction', {}).get('stability_analysis') # 안정성 분석
            quality = root.get('quality_analysis') or root.get('prediction', {}).get('quality_analysis') # 품질 분석

            ui = root.get('ui_messages') \
                 or root.get('meta', {}).get('ui_messages') \
                 or root.get('picks', {}).get('ui_messages') \
                 or [] # UI 메시지

            meta = {
                "grade": (stability or {}).get("investment_grade"), # 투자 등급
                "message": (stability or {}).get("message"), # 메시지
                "ic_sharpe": ((stability or {}).get("metrics") or {}).get("ic_sharpe"), # IC 샤프
                "quality_score": (quality or {}).get("score"), # 품질 점수
                "confidence": (quality or {}).get("confidence"), # 신뢰도
                "ui_messages": ui, # UI 메시지
                "backtest_available": hasattr(self, 'backtest_results') and self.backtest_results.get('success', False) # 백테스팅 가능 여부
            }
            return meta
        except Exception:
            return { # 에러시 기본값
                "grade": None,
                "message": "",
                "ic_sharpe": None,
                "quality_score": None,
                "confidence": None,
                "ui_messages": [],
                "backtest_available": False
            }

# 분석 시작 전에 factor-neutral 옵션, 롤링 윈도우 안전장치 등
def prepare_returns(self, use_factor_neutral: bool = True):
    rets = self.prices.pct_change()
    if use_factor_neutral:
        mkt = self.data.get_index_return("KOSPI")
        sectors = self.data.get_sector_factors(self.universe)
        X = build_factor_matrix({"MKT": mkt, "SECTORS": sectors}, rets.index)
        rets = neutralize_to_factors(rets, X)
    # 윈도우 안정장치
    if len(rets) < self.window:
        self.window = max(42, int(len(rets) * 0.6))
    self.returns = rets.dropna(how="all")