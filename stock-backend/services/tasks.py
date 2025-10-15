# services/tasks.py

import asyncio
import traceback
from datetime import datetime
from core.state import executor, analysis_tasks
from services.analysis import KoreanStockCorrelationAnalysis
from services.data import stock_manager
from models import AnalysisRequest
import numpy as np
import gc

# 백그라운드 분석 작업
async def run_analysis_task(task_id: str, request: AnalysisRequest):
    try:
        # ✅ 진행률 관리 함수 정의
        def update_progress(progress, message, stock_name=None):
            task = analysis_tasks[task_id]
            task["progress"] = round(progress, 3)
            task["message"] = message
            if stock_name:
                task["current_stock"] = stock_name
            elif "current_stock" in task:
                del task["current_stock"]
            print(f"[Progress {int(progress*100)}%] {message} {stock_name or ''}")

        update_progress(0.05, "분석 대기 중...")
        
        analyzer = KoreanStockCorrelationAnalysis()
        loop = asyncio.get_event_loop()
        
        # 1. 데이터 수집 (5% ~ 35%)
        def collection_callback(current, total, stock_name):
            progress = 0.05 + (current / total) * 0.30
            update_progress(progress, f"데이터 수집 중... ({current}/{total})", stock_name)
        
        success, collection_status = await loop.run_in_executor(
            executor, analyzer.collect_data, request.start_date, request.end_date, request.tickers, collection_callback
        )
        
        if not success or not any(s.get('status') == 'success' for s in collection_status):
            analysis_tasks[task_id]["status"] = "failed"
            analysis_tasks[task_id]["message"] = "데이터 수집에 실패했습니다. 유효한 종목이 없습니다."
            analysis_tasks[task_id]["collection_status"] = collection_status
            return
        
        successful_stocks = [s for s in collection_status if s.get('status') == 'success']
        
        # 2. 상관관계 및 포트폴리오 분석 (35% ~ 50%)
        def basic_analysis_callback(progress_base=0.35, progress_range=0.15, message=""):
            # 이 콜백은 각 단계별로 진행률을 조금씩 올립니다.
            current_progress = analysis_tasks[task_id]["progress"]
            new_progress = min(progress_base + progress_range, current_progress + 0.03)
            update_progress(new_progress, message)

        update_progress(0.35, f"상관관계 분석 중... ({len(successful_stocks)}개 종목)")
        await loop.run_in_executor(
            executor, analyzer.analyze_correlation, request.window, 
            lambda message: basic_analysis_callback(message=message)
        )
        
        update_progress(0.45, "포트폴리오 지표 계산 중...")
        await loop.run_in_executor(
            executor, analyzer.calculate_portfolio_metrics,
            lambda message: basic_analysis_callback(message=message)
        )

        # 3. ML 분석 및 백테스팅 (50% ~ 95%)
        def ml_callback(message):
            # ML 분석은 50%에서 95%까지 차지하도록 설정
            current_progress = analysis_tasks[task_id]["progress"]
            # 각 단계마다 약 5~7%씩 진행률을 올립니다.
            new_progress = min(0.95, current_progress + 0.07)
            update_progress(new_progress, message)
        
        update_progress(0.50, "ML 분석 및 백테스팅 초기화...")
        ml_results = await loop.run_in_executor(
            executor,
            analyzer.run_ml_analysis_with_backtest,
            4, 5, True, 5, 5, True, 5, 5, 10.0,
            ml_callback  # ML 콜백 함수 전달
        )
        
        # 4. 결과 생성 (95% ~ 100%)
        update_progress(0.95, "최종 결과 생성 중...")
        performance_summary = analyzer.get_performance_summary()
        
        all_stocks = stock_manager.get_all_stocks()
        stock_name_map = {stock['ticker']: stock['name'] for stock in all_stocks}
        
        correlation_matrix_with_names = {
            stock_name_map.get(t1, t1): {stock_name_map.get(t2, t2): v for t2, v in corrs.items()}
            for t1, corrs in analyzer.static_corr.to_dict().items()
        }
        
        corr_values = analyzer.static_corr.values[np.triu_indices_from(analyzer.static_corr.values, k=1)]
        
        basic_stats = {
            "period": {
                "start": analyzer.stock_data.index[0].strftime('%Y-%m-%d'),
                "end": analyzer.stock_data.index[-1].strftime('%Y-%m-%d'),
                "trading_days": len(analyzer.stock_data)
            },
            "correlation_stats": {
                "average": float(corr_values.mean()), "median": float(np.median(corr_values)),
                "std": float(corr_values.std()), "min": float(corr_values.min()), "max": float(corr_values.max())
            },
            "stocks_analyzed": len(analyzer.stock_data.columns),
            "collection_status": collection_status
        }
        
        portfolio_weights = {"min_variance": analyzer.min_var_weights, "max_sharpe": analyzer.max_sharpe_weights}
        
        analysis_tasks[task_id]["result"] = {
            "basic_stats": basic_stats,
            "performance_summary": performance_summary,
            "correlation_matrix": correlation_matrix_with_names,
            "portfolio_weights": portfolio_weights,
            "analyzer": analyzer,
            "ml_analysis": ml_results or {},
            "backtest_results": ml_results.get('backtest', {}) if ml_results else {}
        }
        
        update_progress(1.0, f"분석 완료 ({len(successful_stocks)}개 종목)")
        analysis_tasks[task_id]["status"] = "completed"
        
    except Exception as e:
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["message"] = f"분석 중 오류 발생: {str(e)}"
        analysis_tasks[task_id]["progress"] = 0
        print(f"Analysis error: {str(e)}")
        traceback.print_exc()
    
    finally:
        if 'analyzer' in locals():
            del analyzer
        gc.collect()