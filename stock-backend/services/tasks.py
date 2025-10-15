# 비동기 작업 

import asyncio
import traceback
from datetime import datetime
from core.state import executor, analysis_tasks
from services.analysis import KoreanStockCorrelationAnalysis
from services.data import stock_manager
from models import AnalysisRequest
import numpy as np
import gc # 💡 [최적화 4] 가비지 컬렉션 모듈 import

# 백그라운드 분석 작업
async def run_analysis_task(task_id: str, request: AnalysisRequest):
    try:
        analysis_tasks[task_id]["status"] = "running"
        analysis_tasks[task_id]["message"] = "데이터 수집 중..."
        analysis_tasks[task_id]["progress"] = 0.1
        
        analyzer = KoreanStockCorrelationAnalysis()
        
        loop = asyncio.get_event_loop()
        
        # 1. 데이터 수집
        success, collection_status = await loop.run_in_executor(
            executor,
            analyzer.collect_data,
            request.start_date,
            request.end_date,
            request.tickers
        )
        
        if not success:
            analysis_tasks[task_id]["status"] = "failed"
            analysis_tasks[task_id]["message"] = "데이터 수집 실패 - 모든 종목에서 데이터를 가져올 수 없습니다"
            analysis_tasks[task_id]["collection_status"] = collection_status
            return
        
        successful_stocks = [s for s in collection_status if s.get('status') == 'success']
        if len(successful_stocks) == 0:
            analysis_tasks[task_id]["status"] = "failed"
            analysis_tasks[task_id]["message"] = "데이터 수집 실패 - 유효한 데이터가 없습니다"
            analysis_tasks[task_id]["collection_status"] = collection_status
            return
        
        # 2. 상관관계 분석
        analysis_tasks[task_id]["progress"] = 0.3
        analysis_tasks[task_id]["message"] = f"상관관계 분석 중... ({len(successful_stocks)}개 종목)"
        
        await loop.run_in_executor(executor, analyzer.analyze_correlation, request.window)
        
        # 3. 포트폴리오 지표 계산
        analysis_tasks[task_id]["progress"] = 0.5
        analysis_tasks[task_id]["message"] = "포트폴리오 지표 계산 중..."
        
        await loop.run_in_executor(executor, analyzer.calculate_portfolio_metrics)
        
        # 4. ML 분석 및 백테스팅 (통합 실행)
        analysis_tasks[task_id]["progress"] = 0.7
        analysis_tasks[task_id]["message"] = "ML 분석 및 백테스팅 중..."
        
        # ML 분석과 백테스팅을 함께 실행
        ml_results = await loop.run_in_executor(
            executor,
            analyzer.run_ml_analysis_with_backtest,  # 통합 메서드 사용
            4,      # k_clusters
            5,      # horizon
            True,   # use_gru
            5,      # top_n
            5,      # bottom_n
            True,   # run_backtest
            5,      # backtest_top_k
            5,      # backtest_bottom_k
            10.0    # backtest_cost_bps
        )
        
        # 결과 출력
        if ml_results:
            # ML 결과 출력
            if 'prediction' in ml_results:
                pred = ml_results['prediction']
                ic_val = pred.get('ic', 0)
                hit_val = pred.get('hit_rate', 0)
                print(f"{'ML 분석 완료'}")
                print(f"IC Score: {ic_val:.3f}")
                print(f"Hit Rate: {hit_val:.3f}")
                print(f"R²: {pred.get('r2', 'N/A')}")
                
                # 투자 등급 판단
                if ic_val > 0.10:
                    grade = "EXCELLENT - 강한 예측력"
                elif ic_val > 0.05:
                    grade = "GOOD - 투자 가능"
                elif ic_val > 0.02:
                    grade = "FAIR - 경계선"
                else:
                    grade = "POOR - 투자 비권장"
                print(f"투자 등급: {grade}")
            
            # 백테스팅 결과 출력
            if 'backtest' in ml_results and ml_results['backtest'].get('success'):
                bt = ml_results['backtest']['summary']
                print(f"{'백테스팅 결과'}")
                print(f"연간 수익률: {bt['annual_return']:>10.2f}%")
                print(f"연간 변동성: {bt['annual_volatility']:>10.2f}%")
                print(f"샤프 비율: {bt['sharpe_ratio']:>10.2f}")
                print(f"최대 낙폭: {bt['max_drawdown']:>10.2f}%")
                print(f"CAGR: {bt['cagr']:>10.2f}%")
                print(f"평균 회전율: {bt['avg_turnover']:>10.2%}")
                
                # 종합 판정
                ic_val = ml_results.get('prediction', {}).get('ic', 0)
                sharpe = bt['sharpe_ratio']
                
                print(f"{'종합 투자 판정'}")
                
                if ic_val > 0.05 and sharpe > 0.5:
                    verdict = "투자 권장"
                    detail = "예측력과 수익성 모두 양호"
                elif ic_val > 0.02 and sharpe > 0.3:
                    verdict = "신중한 투자"
                    detail = "제한적 수익 가능성"
                else:
                    verdict = "투자 비권장"
                    detail = "리스크 대비 수익 미흡"
                
                print(f"{verdict}")
                print(f"{detail}")
            
            # UI 메시지 출력
            if 'ui_messages' in ml_results and ml_results['ui_messages']:
                print("\n분석 메시지:")
                for msg in ml_results['ui_messages'][-3:]:
                    print(f"  - {msg}")
                print()
        
        # 5. 결과 생성
        analysis_tasks[task_id]["progress"] = 0.9
        analysis_tasks[task_id]["message"] = "결과 생성 중..."
        
        performance_summary = analyzer.get_performance_summary()
        
        all_stocks = stock_manager.get_all_stocks()
        stock_name_map = {stock['ticker']: stock['name'] for stock in all_stocks}
        
        # 상관행렬을 이름 기준으로 변환
        correlation_matrix_with_names = {}
        for ticker1, correlations in analyzer.static_corr.to_dict().items():
            stock_name1 = stock_name_map.get(ticker1, ticker1)
            correlation_matrix_with_names[stock_name1] = {}
            for ticker2, corr_value in correlations.items():
                stock_name2 = stock_name_map.get(ticker2, ticker2)
                correlation_matrix_with_names[stock_name1][stock_name2] = corr_value
        
        # 상관 통계 계산
        corr_values = analyzer.static_corr.values[np.triu_indices_from(analyzer.static_corr.values, k=1)]
        
        basic_stats = {
            "period": {
                "start": analyzer.stock_data.index[0].strftime('%Y-%m-%d'),
                "end": analyzer.stock_data.index[-1].strftime('%Y-%m-%d'),
                "trading_days": len(analyzer.stock_data)
            },
            "correlation_stats": {
                "average": float(corr_values.mean()),
                "median": float(np.median(corr_values)),
                "std": float(corr_values.std()),
                "min": float(corr_values.min()),
                "max": float(corr_values.max())
            },
            "stocks_analyzed": len(analyzer.stock_data.columns),
            "collection_status": collection_status
        }
        
        portfolio_weights = {
            "min_variance": analyzer.min_var_weights,
            "max_sharpe": analyzer.max_sharpe_weights
        }
        
        # 최종 결과 저장 (백테스팅 포함)
        analysis_tasks[task_id]["result"] = {
            "basic_stats": basic_stats,
            "performance_summary": performance_summary,
            "correlation_matrix": correlation_matrix_with_names,
            "portfolio_weights": portfolio_weights,
            "analyzer": analyzer,
            "ml_analysis": ml_results if ml_results else {},
            "backtest_results": ml_results.get('backtest', {}) if ml_results else {}  # 백테스팅 결과 추가
        }
        
        analysis_tasks[task_id]["status"] = "completed"
        analysis_tasks[task_id]["message"] = f"분석 완료 ({len(successful_stocks)}개 종목)"
        analysis_tasks[task_id]["progress"] = 1.0
        
    except Exception as e:
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["message"] = f"분석 중 오류 발생: {str(e)}"
        analysis_tasks[task_id]["progress"] = 0
        print(f"Analysis error: {str(e)}")
        traceback.print_exc()
    
    finally:
        # 💡 [최적화 4] 작업 성공/실패 여부와 관계없이 메모리 정리를 시도
        if 'analyzer' in analysis_tasks[task_id].get("result", {}):
            del analysis_tasks[task_id]["result"]["analyzer"] # 가장 큰 객체 삭제
        gc.collect() # 가비지 컬렉션 실행