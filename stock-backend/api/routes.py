# FastAPI 관련 모듈 임포트
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Query, Body
# datetime 모듈 임포트
from datetime import datetime
# core 모듈에서 templates, analysis_tasks 임포트
from core.config import templates
from core.state import analysis_tasks
# services 모듈에서 run_analysis_task, stock_manager 등 임포트
from services.tasks import run_analysis_task
from services.data import stock_manager
# utils.charts 모듈에서 차트 및 데이터 생성 함수들 임포트
from utils.charts import (
    create_portfolio_summary_data,
    create_basic_analysis,
    create_advanced_analysis,
    create_correlation_heatmap,
    create_performance_chart,
    create_timeseries_data,
    create_corr_matrix_data,
    create_corr_vs_vol_data,
    create_quarterly_corr_pairs_data,
    create_fx_corr_60d_data,
)

# JSON 직렬화 관련 유틸리티 임포트
from core.json_safe import SafeJSONResponse
# ML 차트 및 헬퍼 함수 임포트
from utils.ml_charts import create_ml_chart
from utils.helpers import fig_to_base64, get_theme_colors, get_ml_theme_colors
# core 모듈에서 executor 임포트
from core.state import executor
# models 모듈에서 데이터 모델 임포트
from models import AnalysisRequest, AnalysisStatus
# uuid 모듈 임포트 
import uuid
# matplotlib.pyplot 임포트 (메모리 누수 방지용)
import matplotlib.pyplot as plt
# pandas 임포트
import pandas as pd
# services 모듈에서 추가 유틸리티 임포트
from services.data import stock_manager
from services.backtest import walk_forward_backtest
from services.portfolio import portfolio_construct
from services.factors import neutralize_to_factors, build_factor_matrix

from utils.helpers import fig_to_base64, setup_korean_font, get_theme_colors, get_ml_theme_colors 

# APIRouter 인스턴스 생성
router = APIRouter()

# create_ml_chart 함수를 다양한 인자 조합으로 호출하기 위한 호환성 레이어 함수
def _call_create_ml_chart_compat(func, analyzer, ml_results, **kwargs):
    # inspect 모듈 내부 임포트
    import inspect as _inspect

    # 키워드 인자 전달 여부를 테스트하는 내부 함수
    def _try(pass_kwargs: bool):
        # 키워드 인자를 설정하거나 비움
        call_kwargs = kwargs if pass_kwargs else {}
        try:
            # 함수의 시그니처(인자 정보)를 가져옴
            sig = _inspect.signature(func)
            params = list(sig.parameters.values()) # 파라미터 리스트
            has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params) # 가변 위치 인자(*args) 존재 여부
            pos_params = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)] # 위치 인자 목록
            if has_varargs or len(pos_params) >= 2: # 인자가 2개 이상이거나 *args가 있으면
                return func(analyzer, ml_results, **call_kwargs) # (analyzer, ml_results) 순서로 호출
            if len(pos_params) == 1: # 인자가 1개인 경우
                name = pos_params[0].name.lower() # 인자 이름을 소문자로 변환
                if name in ("ml_results", "results", "data"): # 인자 이름으로 ml_results 전달 여부 판단
                    return func(ml_results, **call_kwargs)
                else: # 그 외의 경우 analyzer 전달
                    return func(analyzer, **call_kwargs)
        except Exception: # 오류 발생 시 무시
            pass
        # 다양한 인자 조합으로 함수 호출 시도
        for args in ((analyzer, ml_results), (ml_results,), (analyzer,)):
            try:
                return func(*args, **call_kwargs)
            except TypeError: # 타입 에러 발생 시 다음 조합 시도
                continue
        return None # 모든 시도 실패 시 None 반환

# /stocks, /analysis 엔드포인트들
# 주식 목록을 조회하는 엔드포인트
@router.get("/stocks/list")
async def get_stock_list(market: str | None = None):
    try:
        # 모든 주식 정보 가져오기
        all_stocks = stock_manager.get_all_stocks()
        # market 파라미터가 있으면 해당 시장의 주식만 필터링
        if market and market.lower() != 'all':
            stocks = [s for s in all_stocks if s['market'].lower() == market.lower()]
        else: # 없으면 모든 주식 사용
            stocks = all_stocks
        # 조회 결과를 딕셔너리 형태로 반환
        return {
            "total_count": len(all_stocks),
            "returned_count": len(stocks),
            "stocks": stocks[:100] # 최대 100개만 반환
        }
    except Exception as e: # 오류 발생 시
        # 에러 메시지와 기본 주식 목록 반환
        return {
            "error": str(e),
            "stocks": stock_manager._get_default_stocks()
        }

# 주식을 검색하는 엔드포인트
@router.get("/stocks/search")
async def search_stocks(q: str, market: str | None = None, limit: int = 50):
    try:
        # 검색어(q)로 주식 검색
        results = stock_manager.search_stocks(q, market, limit)
        # 검색 결과를 딕셔너리 형태로 반환
        return {"query": q, "count": len(results), "stocks": results}
    except Exception as e: # 오류 발생 시
        # 에러 메시지와 빈 목록 반환
        return {"error": str(e), "stocks": []}

# 분석을 시작하는 엔드포인트
@router.post("/analysis/start")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    # 고유한 작업 ID 생성
    task_id = str(uuid.uuid4())
    # 작업 상태 초기화
    analysis_tasks[task_id] = {
        "status": "pending",
        "message": "분석 대기 중",
        "progress": 0,
        "result": None,
        "created_at": datetime.now().isoformat()
    }
    # 백그라운드에서 분석 작업 실행
    background_tasks.add_task(run_analysis_task, task_id, request)
    # 작업 ID와 상태 URL을 포함한 응답 반환
    return {
        "task_id": task_id,
        "message": "분석이 시작되었습니다",
        "status_url": f"/analysis/status/{task_id}"
    }

# 분석 상태를 확인하는 엔드포인트
@router.get("/analysis/status/{task_id}")
async def get_analysis_status(task_id: str) -> AnalysisStatus:
    # 작업 ID가 존재하지 않으면 404 에러 발생
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    # 해당 작업의 상태 정보 반환
    task = analysis_tasks[task_id]
    return AnalysisStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        progress=task["progress"]
    )

# 분석 결과를 조회하는 엔드포인트
@router.get("/analysis/result/{task_id}", response_class=SafeJSONResponse)
async def get_analysis_result(task_id: str):
    # 작업 ID가 존재하지 않으면 404 에러 발생
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = analysis_tasks[task_id]
    # 작업이 완료되지 않았으면 현재 상태만 반환
    if task["status"] != "completed":
        return {
            "status": task["status"],
            "message": task["message"],
            "progress": task["progress"]
        }

    result = task["result"] # 완료된 작업 결과

    # ---- ML 분석 결과가 있으면 추출 ----
    ml_analysis = result.get("ml_analysis", {})
    if not ml_analysis and "analyzer" in result: # 결과에 직접 없고 analyzer 객체에 있을 경우
        analyzer_obj = result["analyzer"]
        if hasattr(analyzer_obj, "ml_results"):
            ml_analysis = getattr(analyzer_obj, "ml_results")

    # 기본 응답 데이터 구성
    resp = {
        "status": "completed",
        "basic_stats": result["basic_stats"],
        "performance_summary": result["performance_summary"],
        "correlation_matrix": result["correlation_matrix"],
        "portfolio_weights": result["portfolio_weights"],
    }

    # 사용 가능한 차트 목록 정의 (ML 결과 있으면 'ml' 추가)
    available = ["basic", "advanced", "correlation_heatmap", "performance"]
    if ml_analysis:
        available.append("ml")
    resp["available_charts"] = available

    # ML 분석 결과가 있으면 응답에 추가
    if ml_analysis:
        resp["ml_analysis"] = ml_analysis

        # ML 결과 내 다른 유용한 키가 있으면 그대로 전달
        for k in ("quality_analysis", "stability_analysis", "ui_messages"):
            if isinstance(ml_analysis, dict) and k in ml_analysis:
                resp[k] = ml_analysis[k]

        # 프론트엔드용 예측 요약 정보 생성
        if isinstance(ml_analysis, dict) and "prediction" in ml_analysis:
            pred = ml_analysis["prediction"] or {}
            resp["prediction_summary"] = {
                "ic": pred.get("ic", 0),
                "hit_rate": pred.get("hit_rate", 0),
                "r2": pred.get("r2", None),
                "horizon": pred.get("horizon", 5),
            }

        # 투자 조언 생성 기능이 있으면 호출하여 결과에 포함
        analyzer_obj = result.get("analyzer")
        if analyzer_obj is not None and hasattr(analyzer_obj, "generate_investment_advice"):
            try:
                resp["investment_advice"] = analyzer_obj.generate_investment_advice()
            except Exception: # 오류 발생 시 무시
                pass

    return resp # 최종 응답 반환

# 패널 분리 엔드포인트
# 작업이 완료되었는지 확인하고 analyzer 객체를 반환하는 헬퍼 함수
def _require_ready(task_id: str):
    if task_id not in analysis_tasks: # 작업 ID 없음
        raise HTTPException(status_code=404, detail="Task not found")
    task = analysis_tasks[task_id]
    if task["status"] != "completed": # 작업 미완료
        raise HTTPException(status_code=400, detail="Analysis not completed")
    return task["result"]["analyzer"] # analyzer 객체 반환

# 기본 분석 차트 이미지들을 반환하는 엔드포인트
@router.get("/analysis/chart/{task_id}/basic")
async def chart_basic(task_id: str, theme: str = Query("light")):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    images = create_basic_analysis(analyzer, theme=theme) # 차트 이미지 생성
    # base64 문자열을 data URL 형식으로 변환
    images = {k: f"data:image/png;base64,{v}" for k, v in images.items()}
    return {"images": images} # 이미지 딕셔너리 반환

# 고급 분석 차트 이미지들을 반환하는 엔드포인트
@router.get("/analysis/chart/{task_id}/advanced")
async def chart_advanced(task_id: str, theme: str = Query("light")):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    images = create_advanced_analysis(analyzer, theme=theme) # 차트 이미지 생성
    # base64 문자열을 data URL 형식으로 변환
    images = {k: f"data:image/png;base64,{v}" for k, v in images.items()}
    return {"images": images} # 이미지 딕셔너리 반환

# 상관관계 히트맵 차트 이미지를 반환하는 엔드포인트
@router.get("/analysis/chart/{task_id}/correlation_heatmap")
async def chart_heatmap(task_id: str, theme: str = Query("light")):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    out = create_correlation_heatmap(analyzer, theme=theme) # 히트맵 생성

    # create_correlation_heatmap이 딕셔너리를 반환하는 경우
    if isinstance(out, dict):
        images = {k: f"data:image/png;base64,{v}" for k, v in out.items()} # data URL로 변환
        return {"images": images} # 이미지 딕셔너리 반환

    # Figure 객체를 반환하는 호환성을 위한 경우
    fig = out
    try:
        img_b64 = fig_to_base64(fig) # base64로 변환
    finally:
        plt.close(fig) # Figure 객체 메모리 해제
    return {"images": {"heatmap": f"data:image/png;base64,{img_b64}"}}

# 성과 분석 차트 이미지들을 반환하는 엔드포인트
@router.get("/analysis/chart/{task_id}/performance")
async def chart_performance(task_id: str, theme: str = Query("light")):
    print(f"[chart_performance] theme={theme}") # 테마 로그 출력
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    images = create_performance_chart(analyzer, theme=theme) # 차트 이미지 생성
    # base64 문자열을 data URL 형식으로 변환
    images = {k: f"data:image/png;base64,{v}" for k, v in images.items()}
    return {"images": images} # 이미지 딕셔너리 반환

# ML 분석 차트 이미지를 반환하는 엔드포인트
@router.get("/analysis/chart/{task_id}/ml", response_class=SafeJSONResponse)
async def chart_ml(task_id: str, theme: str = Query("light")):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    import logging, traceback # 로깅, 트레이스백 모듈 임포트
    import matplotlib # matplotlib 임포트
    matplotlib.use("Agg") # 백엔드를 'Agg'로 설정 (GUI 없이 이미지 생성)
    import matplotlib.pyplot as plt # pyplot 임포트
    logger = logging.getLogger("ml_chart") # 로거 생성
    try:
        plt.close("all") # 모든 기존 Figure 닫기

        ml_results = None # ML 결과 초기화
        try:
            from services import ml as ml_service # ML 서비스 모듈 임포트
            # 여러 가능한 함수 이름으로 ML 결과 계산 시도
            for fn_name in ("compute_ml_results","build_ml_results","make_results","analyze","build","run"):
                if hasattr(ml_service, fn_name): # 함수가 존재하면
                    fn = getattr(ml_service, fn_name) # 함수 가져오기
                    try:
                        ml_results = fn(analyzer) # 함수 호출
                        break # 성공 시 루프 중단
                    except Exception as sub_e: # 실패 시 경고 로그
                        logger.warning("ML 보조 함수 호출 실패(%s): %s", fn_name, sub_e)
                        continue
        except Exception as sub_e: # 모듈 로딩 실패 시 경고 로그
            logger.warning("ML 보조 함수 모듈 로딩 생략: %s", sub_e)

        # 호환성 레이어를 통해 ML 차트 생성 함수 호출
        fig = _call_create_ml_chart_compat(
            create_ml_chart, analyzer, ml_results or {},
            theme=theme, show_debug_footer=False, show_summary_footer=True, verbose=False,
        )
        if fig is None: # 실패 시 기본 방식으로 재시도
            try:
                fig = create_ml_chart(analyzer)
            except Exception:
                raise RuntimeError("create_ml_chart 호출 실패")

        # 이미지를 메모리 버퍼에 저장하고 base64로 인코딩
        from io import BytesIO
        import base64
        c = get_theme_colors(theme) # 현재 테마 색상 가져오기

        # 다크 테마일 경우, 차트 요소 색상 최종 보정
        if (theme or "light").lower().startswith("dark"):
            m = get_ml_theme_colors(theme) # ML용 테마 색상
            for ax in fig.get_axes(): # 모든 축에 대해
                ax.set_facecolor(m["axes_facecolor"]) # 축 배경색 설정
                for sp in ax.spines.values(): sp.set_color(c["spine_color"]) # 축 선 색상
                ax.tick_params(colors=c["tick_color"]) # 틱 색상
                ax.xaxis.label.set_color(c["text_color"]) # x축 라벨 색상
                ax.yaxis.label.set_color(c["text_color"]) # y축 라벨 색상
                ttl = ax.get_title(); ax.set_title(ttl, color=c["text_color"]) # 제목 색상
                leg = ax.get_legend() # 범례가 있으면
                if leg:
                    leg.get_frame().set_facecolor(m["axes_facecolor"]) # 범례 배경색
                    leg.get_frame().set_edgecolor(c["spine_color"]) # 범례 테두리색
                    for txt in leg.get_texts(): txt.set_color(c["text_color"]) # 범례 텍스트색

        buf = BytesIO() # 메모리 버퍼 생성
        # 버퍼에 Figure를 png 형식으로 저장 (테마 배경색 적용)
        fig.savefig(buf, format="png", dpi=150, facecolor=c["figure_facecolor"], edgecolor=c["figure_facecolor"])
        buf.seek(0) # 버퍼 포인터를 처음으로 이동
        # base64로 인코딩
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # data URL 형식으로 변환하여 반환
        return {"images": {"ml": f"data:image/png;base64,{img_b64}"}}

    except Exception as e: # ML 차트 생성 중 오류 발생 시
        logger.error("ML 차트 생성 오류: %s\n%s", e, traceback.format_exc()) # 에러 로그
        try:
            plt.close("all") # 모든 Figure 닫기
            # 에러 메시지를 담은 대체 이미지 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "ML 차트 생성 오류", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off") # 축 숨기기
            img_b64 = fig_to_base64(fig, title_size=10, title_pad=6, legend_right_margin=0.93)
            # 에러 이미지와 메시지 반환
            return {"images": {"ml": f"data:image/png;base64,{img_b64}"}, "error": str(e)}
        finally:
            plt.close("all") # 최종적으로 모든 Figure 닫기

# 시스템 상태를 확인하는 헬스 체크 엔드포인트
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len([t for t in analysis_tasks.values() if t["status"] == "running"]),
        "total_stocks": len(stock_manager.get_all_stocks())
    }

# 포트폴리오 요약 테이블 데이터를 반환하는 엔드포인트
@router.get("/analysis/table/{task_id}/portfolio_summary")
async def table_portfolio_summary(task_id: str):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    data = create_portfolio_summary_data(analyzer) # 테이블 데이터 생성
    return {"table": data} # JSON 데이터 반환

# 시계열 데이터를 JSON으로 반환하는 엔드포인트
@router.get("/analysis/data/{task_id}/timeseries")
async def data_timeseries(task_id: str, kind: str = Query("normalized"), limit: int = 100):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    # kind에 따라 정규화, 누적수익률, 전략비교 시계열 데이터 생성
    return create_timeseries_data(analyzer, kind=kind, limit=limit)

# 상관관계 행렬 데이터를 JSON으로 반환하는 엔드포인트
@router.get("/analysis/data/{task_id}/corr_matrix")
async def data_corr_matrix(task_id: str, method: str = Query("pearson")):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    return create_corr_matrix_data(analyzer, method=method) # 상관 행렬 데이터 생성

# 롤링 상관계수 vs 변동성 데이터를 JSON으로 반환하는 엔드포인트
@router.get("/analysis/data/{task_id}/corr_vs_vol")
async def data_corr_vs_vol(task_id: str):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    return create_corr_vs_vol_data(analyzer) # 데이터 생성

# 분기별 상관관계 페어 데이터를 JSON으로 반환하는 엔드포인트
@router.get("/analysis/data/{task_id}/quarterly_corr_pairs")
async def data_quarterly_corr_pairs(task_id: str):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    return create_quarterly_corr_pairs_data(analyzer) # 데이터 생성

# 환율-시장 60일 롤링 상관관계 데이터를 JSON으로 반환하는 엔드포인트
@router.get("/analysis/data/{task_id}/fx_corr_60d")
async def data_fx_corr_60d(task_id: str):
    analyzer = _require_ready(task_id) # analyzer 객체 가져오기
    return create_fx_corr_60d_data(analyzer) # 데이터 생성

# 분석 상태를 확인하는 엔드포인트
@router.get("/analysis/status/{task_id}")
async def get_analysis_status(task_id: str) -> AnalysisStatus:
    # 작업 ID가 존재하지 않으면 404 에러 발생
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    # 해당 작업의 상태 정보 반환
    task = analysis_tasks[task_id]
    return AnalysisStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        progress=task["progress"],
        # 현재 처리 중인 종목 정보 추가
        current_stock=task.get("current_stock", "")
    )

# 입력된 조건으로 포트폴리오 비중을 계산하는 엔드포인트
@router.post("/advisor/mix")
def advisor_mix(payload: dict = Body(...)):
    tickers = payload.get("tickers", []) # 종목 티커 리스트
    if not tickers: # 티커가 없으면 400 에러
        raise HTTPException(400, "tickers가 필요합니다")

    prices = stock_manager.get_prices(tickers) # 주가 데이터 가져오기
    rets = prices.pct_change().dropna(how="all") # 수익률 계산

    # 팩터 중립화 옵션이 켜져 있으면
    if payload.get("factor_neutral", True):
        mkt = stock_manager.get_index_return("KOSPI") # 코스피 지수 수익률
        sectors = stock_manager.get_sector_factors(tickers) # 섹터 팩터
        X = build_factor_matrix({"MKT": mkt, **{"SEC": sectors.sum(axis=1)}}, rets.index) # 팩터 행렬 구성
        rets = neutralize_to_factors(rets, X).dropna(how="all") # 수익률을 팩터에 중립화

    lb = int(payload.get("lookback", 252)) # 분석 기간(lookback)
    window = rets.iloc[-lb:] # 해당 기간의 수익률 데이터
    # 포트폴리오 비중 구성
    w = portfolio_construct(window, method=payload.get("method","HRP"),
                            max_weight=float(payload.get("max_weight",0.25)),
                            prev_w=None,
                            max_turnover=float(payload.get("max_turnover",0.3)))

    exp_vol = (window @ w).std() * (252**0.5) # 포트폴리오의 기대 변동성 계산
    # 응답에 포함될 노트 생성
    notes = [
        f"선택 메서드: {payload.get('method','HRP')}",
        f"팩터 중립화: {'ON' if payload.get('factor_neutral', True) else 'OFF'}",
        f"최대 가중치: {float(payload.get('max_weight',0.25))}, 턴오버 상한: {float(payload.get('max_turnover',0.3))}"
    ]
    # 최종 결과 반환
    return {"weights": w.dropna().to_dict(), "expected_vol": float(exp_vol), "notes": notes}

# 백테스트를 실행하는 엔드포인트
@router.post("/backtest/run")
def backtest_run(payload: dict = Body(...)):
    tickers = payload.get("tickers", []) # 종목 티커 리스트
    if not tickers: # 티커가 없으면 400 에러
        raise HTTPException(400, "tickers가 필요합니다")
    prices = stock_manager.get_prices(tickers) # 주가 데이터 가져오기
    # walk-forward 방식의 백테스트 실행
    res = walk_forward_backtest(
        prices,
        rebal_days=int(payload.get("rebal_days",21)), # 리밸런싱 주기
        lookback_days=int(payload.get("lookback_days",252)), # 분석 기간
        cost_bps=float(payload.get("cost_bps",10.0)), # 거래 비용
        method=payload.get("method","HRP"), # 포트폴리오 구성 방법
        max_weight=float(payload.get("max_weight",0.25)), # 최대 비중
        max_turnover=float(payload.get("max_turnover",0.3)), # 최대 턴오버
    )
    # 백테스트 성과 통계만 반환
    return {"stats": res["stats"]}