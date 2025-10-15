# 차트 생성 (패널별 개별 Figure 생성 → base64 반환)
# 분리 모드만 사용: 각 패널을 별도 Figure로 그려서 base64 dict로 반환
# 여백/비율 통일: utils.helpers.apply_uniform_layout 사용

# services.data의 stock_manager 모듈 임포트
from services.data import stock_manager
# numpy 라이브러리 임포트 
import numpy as np
# pandas 라이브러리 임포트 
import pandas as pd
# matplotlib.pyplot 라이브러리 임포트 
import matplotlib.pyplot as plt
# matplotlib.patheffects 라이브러리 임포트 
import matplotlib.patheffects as pe
# mpl_toolkits.axes_grid1에서 make_axes_locatable 임포트 
from mpl_toolkits.axes_grid1 import make_axes_locatable
# matplotlib.colors에서 TwoSlopeNorm 임포트 
from matplotlib.colors import TwoSlopeNorm
# matplotlib.colors에서 LinearSegmentedColormap 임포트 
from matplotlib.colors import LinearSegmentedColormap
# stock_manager 모듈 내부 임포트
from services.data import stock_manager

# utils.helpers에서 유용한 함수들 임포트
from utils.helpers import (
    format_date_axis, # 날짜 축 서식 설정
    fig_to_base64, # Figure를 base64 문자열로 변환
    put_outside_caption, # 차트 외부에 캡션 추가
    style_value_axis, # 값 축 스타일 설정
    format_month_axis_categorical, # 월별 카테고리 축 서식 설정
    set_plot_theme, # 플롯 테마 설정
    get_theme_colors, # 테마 색상 가져오기
)

# apply_uniform_layout이 helpers에 없을 경우를 대비한 안전한 fallback 정의
try:
    # utils.helpers에서 apply_uniform_layout 함수 임포트 시도
    from utils.helpers import apply_uniform_layout
except Exception:
    # 실패 시, 내부적으로 fallback 함수 정의
    def apply_uniform_layout(fig, with_colorbar: bool = False):
        # 공통 여백/비율 조정을 위한 안전한 기본값 시도
        try:
            # Figure의 레이아웃을 자동으로 조절
            fig.tight_layout()
        except Exception:
            # 오류 발생 시 조용히 넘어감
            pass

# 시계열 데이터(JSON) 제공용 함수
def create_timeseries_data(analyzer, kind: str = "normalized", limit: int = 10):

    # 모든 주식 정보에서 티커를 키, 이름을 값으로 하는 딕셔너리 생성
    stock_info = {s["ticker"]: s["name"] for s in stock_manager.get_all_stocks()}

    # 표시할 최대 열 개수 설정 (1 이상)
    max_display = max(1, int(limit or 100))

    # 인덱스를 ISO 형식의 날짜 문자열 리스트로 변환하는 내부 함수
    def _index_to_iso(idx):
        try:
            # datetime 인덱스를 'YYYY-MM-DD' 형식으로 변환
            return [x.strftime("%Y-%m-%d") for x in pd.to_datetime(idx)]
        except Exception:
            # 실패 시 문자열로 변환
            return [str(x) for x in idx]

    # DataFrame 시리즈를 JSON 페이로드 형식으로 변환하는 내부 함수
    def _series_payload(df, name_map=None):
        # 이름 매핑이 없으면 빈 딕셔너리 사용
        name_map = name_map or {}
        # 출력 리스트 초기화
        out = []
        # DataFrame의 각 열에 대해 반복
        for c in df.columns:
            # 열 이름(티커)에 해당하는 주식 이름 가져오기
            nm = name_map.get(c, c)
            # NaN이나 무한대 값은 None으로 처리하여 리스트 생성
            vals = [None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v) for v in df[c].astype(float).tolist()]
            # 결과 리스트에 딕셔너리 추가
            out.append({"name": nm, "ticker": c, "values": vals})
        # 결과 반환
        return out

    # kind가 'strategy'인 경우
    if kind == "strategy":
        # 전략별 수익률 데이터 딕셔너리 생성
        strategies = {
            "동일가중": analyzer.equal_weight_returns,
            "최소분산": analyzer.min_var_returns,
            "최대샤프": analyzer.max_sharpe_returns,
        }
        # 동적 전략이 있으면 추가
        if hasattr(analyzer, "dynamic_returns"):
            strategies["동적전략"] = analyzer.dynamic_returns

        # 각 전략의 누적 수익률 계산
        cum_map = {name: (1 + r).cumprod() for name, r in strategies.items() if getattr(r, "__len__", lambda: 0)() > 0}
        # 누적 수익률 데이터가 없으면 빈 데이터 반환
        if not cum_map:
            return {"kind": kind, "x": [], "series": []}

        # 누적 수익률 맵으로 DataFrame 생성
        df = pd.DataFrame(cum_map)
        # JSON 형식으로 데이터 반환
        return {
            "kind": kind,
            "x": _index_to_iso(df.index),
            "series": _series_payload(df),
        }

    # 종목 시계열 데이터 처리
    cols = analyzer.stock_data.columns[:max_display] # 표시할 종목 선택
    # kind가 'normalized'인 경우
    if kind == "normalized":
        # 첫날 주가를 1로 정규화
        df = analyzer.stock_data[cols] / analyzer.stock_data[cols].iloc[0]
        # JSON 형식으로 데이터 반환
        return {
            "kind": kind,
            "x": _index_to_iso(df.index),
            "series": _series_payload(df, stock_info),
        }
    # kind가 'cumulative'인 경우
    elif kind == "cumulative":
        # 누적 수익률 계산
        df = (1 + analyzer.returns[cols]).cumprod()
        # JSON 형식으로 데이터 반환
        return {
            "kind": kind,
            "x": _index_to_iso(df.index),
            "series": _series_payload(df, stock_info),
        }

    # 기본적으로 빈 데이터 반환
    return {"kind": kind, "x": [], "series": []}

# 상관관계 행렬 데이터 제공용 함수
def create_corr_matrix_data(analyzer, method: str = "pearson"):

    # 수익률 데이터가 있으면 사용, 없으면 주가 데이터로 계산
    if hasattr(analyzer, "returns") and isinstance(analyzer.returns, pd.DataFrame):
        df = analyzer.returns # 기존 수익률 데이터 사용
    else:
        df = analyzer.stock_data.pct_change().dropna() # 일간 수익률 계산

    # 지정된 방법으로 상관계수 계산
    corr = df.corr(method=method)
    # 상관계수 행렬의 열 이름을 라벨로 사용
    labels = list(corr.columns)

    # 티커 코드를 주식 이름으로 매핑하는 로직
    raw = stock_manager.get_all_stocks() # 모든 주식 정보 가져오기
    # 티커를 키, 이름을 값으로 하는 딕셔너리 생성
    _name_by_ticker = {s["ticker"]: s["name"] for s in raw}

    # 티커 코드에 해당하는 주식 이름을 반환하는 내부 함수
    def _name_of(x: str) -> str:
        # 1) 티커와 완전히 일치하는 이름 우선 검색
        if x in _name_by_ticker:
            return _name_by_ticker[x]
        # 2) '.KS' 등 접미사 제거 후 매칭
        base = x.split(".")[0]
        for t, nm in _name_by_ticker.items():
            if t.split(".")[0] == base:
                return nm
        # 3) 매칭 실패 시 원본 티커 반환
        return x

    # 표시용 라벨 리스트 생성 
    display_labels = [_name_of(c) for c in labels]
    # 코드와 이름을 매핑하는 딕셔너리 생성
    code_to_name = {}
    for c in labels:
        nm = _name_of(c) # 이름 가져오기
        code_to_name[c] = nm # 원본 코드 -> 이름
        code_to_name[c.split(".")[0]] = nm # 접미사 없는 코드 -> 이름

    # JSON 형식으로 데이터 반환
    return {
        "labels": labels, # 원본 티커 라벨
        "display_labels": display_labels, # 표시용 한글 라벨
        "code_to_name": code_to_name, # 코드-이름 매핑 딕셔너리
        "matrix": corr.values.round(6).tolist(), # 상관계수 행렬
    }

# 롤링 상관계수 vs 시장 변동성 시계열 JSON 생성 함수
def create_corr_vs_vol_data(analyzer):

    # analyzer 객체에서 평균 상관계수와 시장 변동성 데이터 가져오기
    avg = getattr(analyzer, "avg_corr", None)
    vol = getattr(analyzer, "market_volatility", None)

    # 두 데이터가 모두 없으면 빈 데이터 반환
    if avg is None and vol is None:
        return {"x": [], "series": []}

    # Series의 NaN 값을 제거하는 내부 함수
    def _clean(s):
        if s is None:
            return pd.Series(dtype=float) # None이면 빈 Series 반환
        try:
            s = s.dropna() # NaN 값 제거
        except Exception:
            pass # 오류 발생 시 무시
        return s

    # 데이터 클리닝
    avg = _clean(avg)
    vol = _clean(vol)

    # 두 데이터의 공통 날짜 구간으로 정렬
    if len(avg) and len(vol):
        idx = avg.index.intersection(vol.index) # 교집합 인덱스
        avg = avg.reindex(idx) # 공통 인덱스로 재정렬
        vol = vol.reindex(idx) # 공통 인덱스로 재정렬
    elif len(avg):
        idx = avg.index # 평균 상관계수 인덱스 사용
    else:
        idx = vol.index # 시장 변동성 인덱스 사용

    # 변동성은 % 스케일로 변환
    vol_pct = vol * 100.0 if len(vol) else vol

    # Series를 리스트로 변환하는 내부 함수 (NaN/inf는 None)
    def _to_list(series):
        out = []
        for v in series.astype(float).tolist():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                out.append(None) # 유효하지 않은 값은 None으로
            else:
                out.append(float(v)) # 유효한 값은 float으로
        return out

    # x축 날짜 데이터 생성 
    x = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in idx] # YYYY-MM-DD
    # JSON 형식으로 데이터 반환
    return {
        "x": x, # x축 데이터 (날짜)
        "series": [
            {"name": "평균 상관계수", "values": _to_list(avg) if len(avg) else []},
            {"name": "시장 변동성(%)", "values": _to_list(vol_pct) if len(vol_pct) else []}
        ],
        "yAxis": [{"name": "상관계수"}, {"name": "변동성(%)"}], # 이중 y축 정보
    }

# 분기별 상관관계 변화 시계열 JSON 생성 함수
def create_quarterly_corr_pairs_data(analyzer):

    # 사용할 종목 선택 (최대 5개)
    cols = list(getattr(analyzer, "returns", pd.DataFrame()).columns)[:5]
    # 데이터가 없으면 빈 데이터 반환
    if not cols or not hasattr(analyzer, "rolling_corr_matrix"):
        return {"x": [], "series": [], "yAxis": [{"name": "상관계수(분기)"}]}

    # 티커를 한글 이름으로 매핑하는 딕셔너리 생성
    info = {s["ticker"]: s["name"] for s in stock_manager.get_all_stocks()}

    # 롤링 상관관계 행렬 데이터 가져오기
    rcm = analyzer.rolling_corr_matrix
    # 종목 페어 리스트 초기화
    pairs = []
    # 모든 종목 페어에 대해 반복
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = cols[i], cols[j]
            try:
                # 특정 페어(s1, s2)의 롤링 상관계수 시계열 추출
                ser = rcm.xs(s2, level=1)[s1]
                # 분기별(QE) 평균으로 리샘플링
                q = ser.resample("QE").mean()
                # 페어 이름 생성 (예: '삼성전자-SK하이닉스')
                name = f"{info.get(s1, s1)}-{info.get(s2, s2)}"
                # 결과 리스트에 추가
                pairs.append((name, q))
            except Exception:
                continue # 오류 발생 시 해당 페어는 건너뜀

    # 상위 5개 페어만 사용
    pairs = pairs[:5]

    # 모든 페어의 데이터를 포함하는 공통 x축(분기말 날짜) 생성
    x = None
    for _, q in pairs:
        if x is None or len(q.index) > len(x):
            x = q.index # 가장 긴 인덱스를 기준으로 설정
    # 공통 x축이 없으면 빈 데이터 반환
    if x is None:
        return {"x": [], "series": [], "yAxis": [{"name": "상관계수(분기)"}]}

    # x축 날짜를 'YYYY-MM-DD' 형식의 리스트로 변환
    x = pd.to_datetime(x).strftime("%Y-%m-%d").tolist()

    # Series를 공통 x축에 맞춰 재정렬하고 리스트로 변환하는 함수
    def to_values(q, x_index):
        q2 = q.reindex(pd.to_datetime(x_index)) # 공통 인덱스로 재정렬
        return [None if pd.isna(v) else float(v) for v in q2.astype(float).tolist()]

    # 각 페어의 데이터를 JSON 시리즈 형식으로 변환
    series = [{"name": n, "values": to_values(q, x)} for n, q in pairs]
    # 최종 JSON 데이터 반환
    return {
        "x": x,
        "series": series,
        "yAxis": [{"name": "상관계수(분기)"}],
    }

# 환율과 시장의 60일 롤링 상관관계 시계열 JSON 생성 함수
def create_fx_corr_60d_data(analyzer):

    # 1) 시장 평균 수익률 계산
    returns = getattr(analyzer, "returns", None)
    if returns is not None and not returns.empty:
        mkt_ret = returns.mean(axis=1) # 종목들의 일간 수익률 평균
    else:
        mkt_ret = getattr(analyzer, "market_return", None) # 대체 시장 수익률 데이터
        if mkt_ret is None: # 데이터가 없으면 빈 데이터 반환
            return {"x": [], "series": [], "yAxis": [{"name": "상관계수(60일 롤링)"}]}

    # 2) USD/KRW 환율 시계열 데이터 찾기
    fx_candidates = [ # 여러 후보 속성을 순차적으로 확인
        getattr(analyzer, "fx_data", None),
        getattr(analyzer, "fx", None),
        getattr(analyzer, "fx_rate", None),
        getattr(analyzer, "usdkrw", None),
        getattr(analyzer, "usdkrw_close", None),
    ]
    # 찾은 첫 번째 유효한 데이터를 fx로 사용
    fx = next((s for s in fx_candidates if s is not None), None)

    # 후보 속성에서 찾지 못한 경우, prices DataFrame에서 탐색
    if fx is None:
        prices = getattr(analyzer, "prices", None)
        if prices is not None:
            pick = None
            for c in prices.columns:
                uc = str(c).upper() # 대문자로 변환
                if "USD" in uc and ("KRW" in uc or "KRW=X" in uc or "USDKRW" in uc):
                    pick = c # 컬럼명에 'USD'와 'KRW'가 포함된 경우 선택
                    break
            if pick is not None:
                fx = prices[pick] # 해당 컬럼 데이터를 fx로 사용

    # 환율 데이터를 찾지 못하면 빈 데이터 반환
    if fx is None:
        return {"x": [], "series": [], "yAxis": [{"name": "상관계수(60일 롤링)"}]}

    # 3) 수익률 계산 및 데이터 정렬
    fx_ret = fx.pct_change().dropna() # 환율 일간 수익률
    mkt_ret = mkt_ret.dropna() # 시장 일간 수익률

    idx = fx_ret.index.intersection(mkt_ret.index) # 공통 날짜 인덱스
    fx_ret = fx_ret.reindex(idx) # 공통 인덱스로 재정렬
    mkt_ret = mkt_ret.reindex(idx) # 공통 인덱스로 재정렬

    # 데이터가 비어있으면 빈 데이터 반환
    if fx_ret.empty or mkt_ret.empty:
        return {"x": [], "series": [], "yAxis": [{"name": "상관계수(60일 롤링)"}]}

    # 4) 60일 롤링 상관계수 계산
    corr60 = fx_ret.rolling(60, min_periods=10).corr(mkt_ret)
    corr60 = corr60.replace([np.inf, -np.inf], np.nan) # 무한대 값을 NaN으로 처리

    # x축(날짜)과 y축(값) 데이터 생성
    x = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in corr60.index]
    vals = [None if (pd.isna(v)) else float(v) for v in corr60.astype(float).tolist()]

    # 최종 JSON 데이터 반환
    return {
        "x": x,
        "series": [{"name": "USD/KRW vs 시장 상관 (60D)", "values": vals}],
        "yAxis": [{"name": "상관계수(60일 롤링)"}],
    }

# 내부 유틸
# Figure를 base64로 변환하고 닫는 내부 헬퍼 함수
def _b64_and_close(fig):
    img = fig_to_base64(fig) # Figure를 base64 문자열로 변환
    plt.close(fig) # Figure 객체 메모리 해제
    return img # base64 문자열 반환

# 통일된 diverging 컬러맵 정의
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr_unified", ["#EF9A9A", "#e2eefd", "#7fb8f9"] # 음수(빨강), 중립(회색), 양수(청록)
)

# 기본 분석 차트 생성
def create_basic_analysis(analyzer, theme: str = "light"):
    set_plot_theme(theme) # 지정된 테마로 플롯 스타일 설정
    stock_info = {s["ticker"]: s["name"] for s in stock_manager.get_all_stocks()} # 티커-이름 맵
    max_display = 100 # 최대 표시 종목 수
    cols = analyzer.stock_data.columns[:max_display] # 분석 대상 종목 선택
    palette = plt.cm.tab10(np.linspace(0, 1, len(cols))) # 종목별 색상 팔레트 생성

    out = {} # 결과 이미지를 담을 딕셔너리

    # 1) 정규화 주가 차트
    fig = plt.figure(figsize=(20, 14)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    norm = analyzer.stock_data[cols] / analyzer.stock_data[cols].iloc[0] # 첫날 주가를 1로 정규화
    for i, c in enumerate(cols): # 각 종목에 대해 반복
        ax.plot(norm.index, norm[c], lw=2, color=palette[i], label=stock_info.get(c, c)) # 라인 플롯 그리기
    ax.set_title("정규화 주가", fontsize=15, fontweight="bold", pad=18) # 차트 제목 설정
    put_outside_caption(ax, "정규화 주가") # 외부 캡션 추가
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10) # 범례 설정
    ax.grid(True, alpha=0.3) # 그리드 표시
    format_date_axis(ax, interval_months=3) # x축 날짜 서식 설정
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=2) # y축 스타일 설정
    ax.tick_params(axis="x", colors="#777", labelsize=9, pad=4) # x축 틱 파라미터 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["normalized_price"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 2) 상관관계 행렬 차트
    fig = plt.figure(figsize=(20, 14)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    corr = analyzer.static_corr.loc[cols, cols].copy() # 상관계수 행렬 데이터 복사
    corr.index = [stock_info.get(i, i) for i in corr.index] # 인덱스를 주식 이름으로 변경
    corr.columns = [stock_info.get(i, i) for i in corr.columns] # 컬럼을 주식 이름으로 변경
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) # -1, 0, 1을 기준으로 색상 정규화
    im = ax.imshow(corr.values, cmap=CORR_CMAP, norm=norm, aspect="auto") # 히트맵 이미지 생성
    ax.set_xticks(range(len(corr.columns))) # x축 틱 위치 설정
    ax.set_yticks(range(len(corr.index))) # y축 틱 위치 설정
    ax.set_xticklabels(corr.columns, rotation=45, ha="right") # x축 라벨 설정 
    ax.set_yticklabels(corr.index) # y축 라벨 설정
    # 각 셀에 상관계수 값 텍스트로 표시
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            if i <= j: # 대각선 및 위쪽 삼각형에만 표시
                val = float(corr.iloc[i, j]) # 셀 값
                rgba = CORR_CMAP(norm(val)) # 셀의 색상 값
                r, g, b = rgba[:3] # RGB 값 추출
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b # 인지적 밝기 계산
                txt_color = "black" if lum > 0.6 else "white" # 밝기에 따라 텍스트 색상 결정
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=9) # 텍스트 추가
    divider = make_axes_locatable(ax) # 축 분할기 생성
    cax = divider.append_axes("right", size="2.6%", pad=0.04) # 컬러바를 위한 축 추가
    cbar = fig.colorbar(im, cax=cax) # 컬러바 생성
    _theme = get_theme_colors(theme) # 현재 테마 색상 가져오기
    cbar.set_label("상관계수", rotation=270, labelpad=20, color=_theme["text_color"]) # 컬러바 라벨 설정
    cbar.ax.yaxis.set_tick_params(color=_theme["tick_color"]) # 컬러바 틱 색상 설정
    for _t in cbar.ax.get_yticklabels(): # 컬러바 틱 라벨 색상 설정
        _t.set_color(_theme["tick_color"])
    cbar.outline.set_edgecolor(_theme["spine_color"]) # 컬러바 외곽선 색상 설정
    ax.set_title("상관관계 행렬", fontsize=15, fontweight="bold", pad=18) # 차트 제목 설정
    put_outside_caption(ax, "상관계수(선택 종목)") # 외부 캡션 추가
    apply_uniform_layout(fig, with_colorbar=True) # 레이아웃 통일
    out["corr_matrix"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 3) 일간 수익률 분포 차트 (히스토그램)
    fig = plt.figure(figsize=(20, 14)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    data, labels = [], [] # 데이터와 라벨 리스트 초기화
    for i, c in enumerate(cols): # 각 종목에 대해 반복
        data.append(analyzer.returns[c].dropna().values) # 수익률 데이터 추가
        labels.append(stock_info.get(c, c)) # 종목 이름 라벨 추가
    ax.hist(data, bins=30, alpha=0.6, label=labels, color=palette[:len(labels)], density=True) # 히스토그램 그리기
    ax.set_title("일간 수익률 분포", fontsize=15, fontweight="bold", pad=18) # 차트 제목 설정
    put_outside_caption(ax, "빈도(밀도)") # 외부 캡션 추가
    ax.legend(fontsize=9) # 범례 표시
    ax.grid(True, alpha=0.3) # 그리드 표시
    ax.set_xlim(-0.1, 0.1) # x축 범위 제한
    style_value_axis(ax, axis="x", nbins=6, color="#888", percent=True) # x축 스타일 설정 (백분율)
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=0) # y축 스타일 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["daily_return_hist"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 4) 누적 수익률 차트
    fig = plt.figure(figsize=(20, 14)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    cum = (1 + analyzer.returns[cols]).cumprod() # 누적 수익률 계산
    for i, c in enumerate(cols): # 각 종목에 대해 반복
        ax.plot(cum.index, cum[c], lw=2, color=palette[i], label=stock_info.get(c, c)) # 라인 플롯 그리기
    ax.set_title("누적 수익률", fontsize=15, fontweight="bold", pad=18) # 차트 제목 설정
    put_outside_caption(ax, "누적 수익률") # 외부 캡션 추가
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10) # 범례 설정
    ax.grid(True, alpha=0.3) # 그리드 표시
    format_date_axis(ax, interval_months=3) # x축 날짜 서식 설정
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=2) # y축 스타일 설정
    ax.tick_params(axis="x", colors="#777", labelsize=9, pad=4) # x축 틱 파라미터 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["cumulative_return"] = _b64_and_close(fig) # base64로 변환하여 저장

    return out # 결과 딕셔너리 반환

# 고급 분석 차트 생성
def create_advanced_analysis(analyzer, theme: str = "light"):
    set_plot_theme(theme) # 지정된 테마로 플롯 스타일 설정
    stock_info = {s["ticker"]: s["name"] for s in stock_manager.get_all_stocks()} # 티커-이름 맵

    # 공통으로 사용할 포트폴리오 전략 정의
    strategies = {
        "동일가중": analyzer.equal_weight_returns,
        "최소분산": analyzer.min_var_returns,
        "최대샤프": analyzer.max_sharpe_returns,
    }
    # 동적 전략이 있으면 추가
    if hasattr(analyzer, "dynamic_returns"):
        strategies["동적전략"] = analyzer.dynamic_returns

    out = {} # 결과(base64 이미지)를 담을 딕셔너리

    # 1) 롤링 상관계수 vs 시장 변동성 차트
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1); twin = ax.twinx() # 기본 축과 이중 y축 생성
    ax.plot(analyzer.avg_corr.index, analyzer.avg_corr, "b-", lw=2, label="평균 상관계수") # 평균 상관계수 플롯 
    twin.plot(analyzer.market_volatility.index, analyzer.market_volatility * 100, "r-", lw=2, alpha=0.7, label="시장 변동성") # 시장 변동성 플롯 (빨간색)
    ax.set_title("롤링 상관계수 vs 시장 변동성", fontsize=14, fontweight="bold") # 차트 제목 설정
    put_outside_caption(ax, "평균 상관계수 / 변동성(%)") # 외부 캡션 추가
    ax.grid(True, alpha=0.3) # 그리드 표시
    format_date_axis(ax, interval_months=3) # x축 날짜 서식 설정
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=2) # 왼쪽 y축 스타일 설정
    style_value_axis(twin, axis="y", nbins=5, color="#888", percent=False, decimals=1) # 오른쪽 y축 스타일 설정
    ax.tick_params(axis="x", colors="#777", labelsize=9, pad=4) # x축 틱 파라미터 설정
    lines1, labels1 = ax.get_legend_handles_labels() # 왼쪽 축 범례 정보
    lines2, labels2 = twin.get_legend_handles_labels() # 오른쪽 축 범례 정보
    ax.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9) # 통합 범례 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["rolling_corr_vs_vol"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 2) 위험-수익 프로파일 차트 
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    max_display = 100 # 최대 표시 종목 수
    cols = analyzer.returns.columns[:max_display] if len(analyzer.returns.columns) > max_display else analyzer.returns.columns # 대상 종목 선택
    m = analyzer.returns[cols].mean() * 252 # 연간 기대수익률
    s = analyzer.returns[cols].std() * np.sqrt(252) # 연간 변동성
    palette = plt.cm.tab10(np.linspace(0, 1, len(cols))) # 종목별 색상 팔레트
    for i, t in enumerate(cols): # 각 종목에 대해 반복
        name = stock_info.get(t, t) # 종목 이름
        ax.scatter(s[t]*100, m[t]*100, s=80, alpha=0.75, color=palette[i], label=name, linewidths=0) # 산점도 그리기

        # 각 점에 라벨 추가
        ann = ax.annotate(
            name, (s[t]*100, m[t]*100),
            xytext=(5, 4), textcoords="offset points",
            fontsize=9,
            color=plt.rcParams["text.color"] # 테마에 맞는 텍스트 색상 사용
        )
        # 라벨의 외곽선과 배경 제거
        try:
            ann.set_path_effects([])
            ann.set_bbox(None)
        except Exception:
            pass
    # 모든 텍스트 라벨 스타일 정리
    for _t in ax.texts:
        try:
            _t.set_path_effects([]) # 외곽선 제거
            _t.set_bbox(None) # 배경 제거
        except Exception:
            pass
        _t.set_color(plt.rcParams["text.color"]) # 테마 색상으로 통일
    ax.set_title("위험-수익 프로파일", fontsize=14, fontweight="bold") # 차트 제목 설정
    put_outside_caption(ax, "연간 변동성(%) / 수익률(%)") # 외부 캡션 추가
    ax.grid(True, alpha=0.3) # 그리드 표시
    style_value_axis(ax, axis="x", nbins=6, color="#888", percent=False, decimals=0) # x축 스타일 설정
    style_value_axis(ax, axis="y", nbins=6, color="#888", percent=False, decimals=0) # y축 스타일 설정
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9) # 범례 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["risk_return"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 3) 포트폴리오 전략 비교 차트
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    for name, r in strategies.items(): # 각 전략에 대해 반복
        if len(r) > 0: # 수익률 데이터가 있을 경우
            ax.plot((1 + r).cumprod().index, (1 + r).cumprod(), lw=2, label=name) # 누적 수익률 플롯
    ax.set_title("포트폴리오 전략 비교", fontsize=14, fontweight="bold") # 차트 제목 설정
    put_outside_caption(ax, "누적 수익률") # 외부 캡션 추가
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left") # 범례 설정
    ax.grid(True, alpha=0.3) # 그리드 표시
    format_date_axis(ax, interval_months=3) # x축 날짜 서식 설정
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=2) # y축 스타일 설정
    ax.tick_params(axis="x", colors="#777", labelsize=9, pad=4) # x축 틱 파라미터 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["strategy_compare"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 4) 분기별 상관관계 변화 차트 (상위 5개 페어)
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    pairs = {} # 페어별 데이터를 저장할 딕셔너리
    names = [stock_info.get(c, c) for c in cols] # 종목 이름 리스트
    for i in range(min(len(cols), 5)): # 상위 5개 종목 내에서 페어 생성
        for j in range(i + 1, min(len(cols), 5)):
            s1, s2 = cols[i], cols[j] # 티커 페어
            n1, n2 = names[i], names[j] # 이름 페어
            if s2 in analyzer.rolling_corr_matrix.columns: # 데이터 유효성 확인
                ser = analyzer.rolling_corr_matrix.xs(s2, level=1)[s1] # 롤링 상관계수 시계열 추출
                pairs[f"{n1}-{n2}"] = ser.resample("QE").mean() # 분기별 평균으로 리샘플링하여 저장
    for k, v in list(pairs.items())[:5]: # 상위 5개 페어에 대해 플롯
        ax.plot(v.index, v, label=k, alpha=0.7, lw=1.5)
    ax.set_title("분기별 상관관계 변화 (상위 5개 페어)", fontsize=14, fontweight="bold") # 차트 제목 설정
    put_outside_caption(ax, "상관계수(분기)") # 외부 캡션 추가
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9) # 범례 설정
    ax.grid(True, alpha=0.3) # 그리드 표시
    format_date_axis(ax, interval_months=3) # x축 날짜 서식 설정
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=2) # y축 스타일 설정
    ax.tick_params(axis="x", colors="#777", labelsize=9, pad=4) # x축 틱 파라미터 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["quarterly_corr_pairs"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 5) USD/KRW vs 주식시장 상관관계 차트 
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    if len(analyzer.fx_data) > 0: # 환율 데이터가 있을 경우
        fx = analyzer.fx_data.pct_change().dropna() # 환율 수익률
        avg = analyzer.returns.mean(axis=1) # 시장 평균 수익률
        common = fx.index.intersection(avg.index) # 공통 날짜 인덱스
        if len(common) >= 60: # 60일 이상 데이터가 있을 경우
            fxA, avgA = fx.loc[common], avg.loc[common] # 공통 기간 데이터 추출
            rc = pd.Series(index=common[59:], dtype=float) # 롤링 상관계수를 저장할 Series
            for i in range(59, len(common)): # 롤링 기간에 대해 반복 계산
                rc.iloc[i - 59] = fxA.iloc[i - 59:i + 1].corr(avgA.iloc[i - 59:i + 1])
            ax.plot(rc.index, rc, lw=2, color="purple") # 롤링 상관계수 플롯
            ax.axhline(y=0, color="black", ls="--", alpha=0.5) # 0 기준선 추가
            ax.set_title("USD/KRW vs 주식시장 상관관계 (60일 롤링)", fontsize=14, fontweight="bold") # 차트 제목
            put_outside_caption(ax, "상관계수(환율 vs 시장)") # 외부 캡션
            ax.grid(True, alpha=0.3) # 그리드 표시
            format_date_axis(ax, interval_months=3) # x축 날짜 서식
            style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=2) # y축 스타일
            ax.tick_params(axis="x", colors="#777", labelsize=9, pad=4) # x축 틱 파라미터
        else: # 데이터가 부족할 경우
            ax.text(0.5, 0.5, "롤링 상관관계 계산을 위한 데이터 부족", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("USD/KRW vs 주식시장 상관관계", fontsize=14, fontweight="bold")
    else: # 환율 데이터가 없을 경우
        ax.text(0.5, 0.5, "FX 데이터 없음", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("USD/KRW vs 주식시장 상관관계", fontsize=14, fontweight="bold")
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["fx_corr_60d"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 6) 최대 낙폭 분석 차트
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    for name, r in strategies.items(): # 각 전략에 대해 반복
        if len(r) > 0: # 데이터가 있을 경우
            cum = (1 + r).cumprod() # 누적 수익률
            dd = (cum - cum.expanding().max()) / cum.expanding().max() * 100 # 최대 낙폭 계산
            ax.fill_between(dd.index, dd, 0, alpha=0.3, label=name) # 낙폭 영역 채우기
    ax.set_title("최대 낙폭 분석", fontsize=14, fontweight="bold") # 차트 제목 설정
    put_outside_caption(ax, "낙폭(%)") # 외부 캡션 추가
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left") # 범례 설정
    ax.grid(True, alpha=0.3) # 그리드 표시
    format_date_axis(ax, interval_months=3) # x축 날짜 서식 설정
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=0) # y축 스타일 설정
    ax.tick_params(axis="x", colors="#777", labelsize=9, pad=4) # x축 틱 파라미터 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["max_drawdown"] = _b64_and_close(fig) # base64로 변환하여 저장

    return out # 결과 딕셔너리 반환

# 상관관계 히트맵 생성 함수
def create_correlation_heatmap(analyzer, theme: str = "light"):
    set_plot_theme(theme) # 테마 설정
    max_stocks = 100 # 최대 표시 종목 수
    if len(analyzer.stock_data.columns) > max_stocks: # 종목 수가 최대치를 넘으면
        vol = analyzer.returns.std().sort_values(ascending=False) # 변동성 기준으로 정렬
        selected = vol.index[:max_stocks] # 상위 변동성 종목 선택
        corr_matrix = analyzer.static_corr.loc[selected, selected] # 선택된 종목으로 상관 행렬 생성
    else:
        corr_matrix = analyzer.static_corr # 모든 종목 사용

    fig = plt.figure(figsize=(14, 14)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    try: # 차트 크기 조정 시도
        desired_view_scale = 0.80 # 화면에서 보이는 목표 비율
        scale_inches = 1.0 / desired_view_scale # 크기 조절 계수
        w, h = fig.get_size_inches() # 현재 크기 가져오기
        fig.set_size_inches(w * scale_inches, h * scale_inches, forward=True) # 크기 재설정
    except Exception:
        pass # 오류 발생 시 무시
    stock_info = {s["ticker"]: s["name"] for s in stock_manager.get_all_stocks()} # 티커-이름 맵
    corr = corr_matrix.copy() # 상관 행렬 복사
    cols = list(corr.columns) # 컬럼 리스트 (안전하게 정의)
    def _nm(code: str) -> str: # 티커를 이름으로 변환하는 내부 함수
        if code in stock_info:
            return stock_info[code] # 직접 매칭
        base = code.split(".")[0] # 접미사 제거
        for t, nm in stock_info.items():
            if t.split(".")[0] == base: # 접미사 없는 코드로 매칭
                return nm
        return code # 실패 시 원본 코드 반환

    corr.index   = [_nm(i) for i in corr.index] # 인덱스를 주식 이름으로 변경
    corr.columns = [_nm(i) for i in corr.columns] # 컬럼을 주식 이름으로 변경
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) # 색상 정규화
    im = ax.imshow(corr.values, cmap=CORR_CMAP, norm=norm, aspect="auto") # 히트맵 이미지 생성
    ax.set_xticks(range(len(corr.columns))) # x축 틱 위치 설정
    ax.set_yticks(range(len(corr.index))) # y축 틱 위치 설정
    ax.set_xticklabels(corr.columns, rotation=45, ha="right") # x축 라벨 설정 (45도 회전)
    ax.set_yticklabels(corr.index) # y축 라벨 설정
    # 각 셀에 값 표시
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            if i < j: # 위쪽 삼각형에만 값 표시
                val = float(corr.iloc[i, j]) # 셀 값
                rgba = CORR_CMAP(norm(val)) # 셀 색상
                r, g, b = rgba[:3] # RGB 값
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b # 인지적 밝기
                txt_color = "black" if lum > 0.6 else "white" # 텍스트 색상 결정
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=10) # 텍스트 추가
    divider = make_axes_locatable(ax) # 축 분할기
    cax = divider.append_axes("right", size="2.8%", pad=0.04) # 컬러바 축 추가
    cbar = fig.colorbar(im, cax=cax) # 컬러바 생성
    _theme = get_theme_colors(theme) # 테마 색상 가져오기
    cbar.set_label("상관계수", rotation=270, labelpad=20, color=_theme["text_color"]) # 컬러바 라벨
    cbar.ax.yaxis.set_tick_params(color=_theme["tick_color"]) # 컬러바 틱 색상
    for _t in cbar.ax.get_yticklabels(): # 컬러바 틱 라벨 색상
        _t.set_color(_theme["tick_color"])
    cbar.outline.set_edgecolor(_theme["spine_color"]) # 컬러바 외곽선 색상
    fig.suptitle("주식 상관관계 히트맵", x=0.5, y=0.96, ha="center", fontsize=18, fontweight="bold") # 전체 제목
    put_outside_caption(ax, "상관계수(피어슨)") # 외부 캡션
    ax.tick_params(axis="both", labelsize=9, colors="#555") # 양쪽 축 틱 파라미터
    apply_uniform_layout(fig, with_colorbar=True) # 레이아웃 통일
    img = _b64_and_close(fig) # base64로 변환
    return {"heatmap": img} # 결과 반환

# 성과 분석 차트 생성
def create_performance_chart(analyzer, theme: str = "light"):
    set_plot_theme(theme) # 테마 설정
    stock_info = {s["ticker"]: s["name"] for s in stock_manager.get_all_stocks()} # 티커-이름 맵
    top = analyzer.returns.columns # 분석 대상 종목
    out = {} # 결과 딕셔너리

    # 1) 개별 종목 샤프 비율 차트 
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    sr = (analyzer.returns[top].mean() * 252) / (analyzer.returns[top].std() * np.sqrt(252)) # 샤프 비율 계산
    srs = pd.Series(sr.values, index=[stock_info.get(i, i) for i in sr.index]) # 인덱스를 주식 이름으로 변경
    cols = ["skyblue" if x > 0 else "salmon" for x in srs.values] # 값에 따라 막대 색상 결정 (양수/음수)
    bars = ax.bar(range(len(srs)), srs.values, color=cols) # 막대 차트 그리기
    ax.set_xticks(range(len(srs))) # x축 틱 위치 설정
    ax.set_xticklabels(srs.index, rotation=45, ha="right") # x축 라벨 설정 
    ax.set_title("개별 종목 샤프 비율", fontsize=16, fontweight="bold", pad=15) # 차트 제목
    put_outside_caption(ax, "샤프 비율") # 외부 캡션
    ax.grid(True, alpha=0.3, axis="y") # 가로 그리드 표시
    ax.axhline(y=0, color="black", lw=0.5) # 0 기준선 추가
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=False, decimals=2) # y축 스타일
    for b, v in zip(bars, srs.values): # 각 막대에 값 표시
        y = b.get_height() # 막대 높이
        ax.text(b.get_x() + b.get_width() / 2.0, y + (0.01 if y > 0 else -0.01), f"{v:.2f}",
                ha="center", va="bottom" if y > 0 else "top", fontsize=9)
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["sharpe_by_stock"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 2) 위험-수익 프로파일 차트 
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    m = analyzer.returns[top].mean() * 252 # 연간 기대수익률
    s = analyzer.returns[top].std() * np.sqrt(252) # 연간 변동성
    palette = plt.cm.tab10(np.linspace(0, 1, len(top))) # 종목별 색상 팔레트
    for i, t in enumerate(top): # 개별 종목 산점도
        name = stock_info.get(t, t)
        ax.scatter(s[t]*100, m[t]*100, s=80, alpha=0.7, color=palette[i], label=name)
    strategies = { # 포트폴리오 전략
        "동일가중": analyzer.equal_weight_returns,
        "최소분산": analyzer.min_var_returns,
        "최대샤프": analyzer.max_sharpe_returns,
    }
    markers = ["*", "s", "D"]; pcolors = ["gold", "red", "darkblue"] # 전략별 마커와 색상
    for (name, r), mk, col in zip(strategies.items(), markers, pcolors): # 각 전략에 대해 반복
        if len(r) > 0:
            mr = r.mean() * 252 * 100 # 전략의 연간 수익률
            sr_ = r.std() * np.sqrt(252) * 100 # 전략의 연간 변동성
            ax.scatter(sr_, mr, s=60, marker=mk, c=col, edgecolors="black", lw=1, label=f"{name} (포트폴리오)", zorder=5) # 전략 산점도
    ax.set_title("위험-수익 프로파일", fontsize=16, fontweight="bold", pad=15) # 차트 제목
    put_outside_caption(ax, "연간 변동성(%) / 수익률(%)") # 외부 캡션
    ax.grid(True, alpha=0.3) # 그리드 표시
    style_value_axis(ax, axis="x", nbins=6, color="#888", percent=False, decimals=0) # x축 스타일
    style_value_axis(ax, axis="y", nbins=6, color="#888", percent=False, decimals=0) # y축 스타일
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9) # 범례 설정
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["risk_return_profile"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 3) 월별 전략 수익률 차트 (막대)
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1) # Axes 객체 생성
    df = pd.DataFrame({ # 전략별 수익률로 DataFrame 생성
        "동일가중": analyzer.equal_weight_returns,
        "최소분산": analyzer.min_var_returns,
        "최대샤프": analyzer.max_sharpe_returns,
    })
    monthly = df.resample("ME").apply(lambda x: (1 + x).prod() - 1) # 월별 수익률로 리샘플링
    if len(monthly) > 12: # 데이터가 12개월을 초과하면
        monthly = monthly.tail(12) # 최근 12개월만 사용
    monthly.plot(kind="bar", ax=ax, width=0.8, color=["skyblue", "lightcoral", "lightgreen"]) # 막대 차트 그리기
    ax.set_title("월별 전략 수익률 (최근 12개월)", fontsize=16, fontweight="bold", pad=15) # 차트 제목
    put_outside_caption(ax, "월간 수익률") # 외부 캡션
    ax.legend(title="전략") # 범례 설정
    ax.grid(True, alpha=0.3, axis="y") # 가로 그리드 표시
    style_value_axis(ax, axis="y", nbins=5, color="#888", percent=True) # y축 스타일 (백분율)
    format_month_axis_categorical(ax, monthly.index, max_ticks=8, color="#777") # x축 월별 서식 설정
    ax.set_xlabel("") # x축 라벨 제거
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["monthly_strategy_returns"] = _b64_and_close(fig) # base64로 변환하여 저장

    # 4) 포트폴리오 성과 요약 차트 (테이블)
    fig = plt.figure(figsize=(20, 16)) # Figure 객체 생성
    ax = plt.subplot(1, 1, 1); ax.axis("tight"); ax.axis("off") # 축을 끄고 타이트하게 설정
    rows = [] # 테이블 행 데이터
    for name, r in strategies.items(): # 각 전략에 대해 반복
        if len(r) > 0: # 데이터가 있을 경우
            cum = (1 + r).cumprod() # 누적 수익률
            dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min() # 최대 낙폭
            rows.append([ # 성과 지표 계산 및 추가
                name,
                f"{r.mean() * 252 * 100:.1f}%", # 연간 수익률
                f"{r.std() * np.sqrt(252) * 100:.1f}%", # 변동성
                f"{(r.mean() * 252) / (r.std() * np.sqrt(252)):.2f}", # 샤프 비율
                f"{dd * 100:.1f}%", # 최대 낙폭
                f"{(cum.iloc[-1] - 1) * 100:.1f}%", # 총 수익률
            ])
    if rows: # 행 데이터가 있으면 테이블 생성
        tbl = ax.table(
            cellText=rows,
            colLabels=["전략", "연간수익률", "변동성", "샤프비율", "최대낙폭", "총수익률"],
            cellLoc="center", loc="center",
            colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16],
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.5) # 폰트 및 스케일 설정
        _t = get_theme_colors(theme) # 테마 색상 가져오기
        _header_bg   = _t["table_header_bg"] # 헤더 배경색
        _header_text = _t["table_header_text"] # 헤더 텍스트색
        _row_even    = _t["table_row_even"] # 짝수 행 배경색
        _row_odd     = _t["table_row_odd"] # 홀수 행 배경색
        _txt_color   = _t["text_color"] # 기본 텍스트색

        for i in range(6): # 헤더 스타일 설정
            tbl[(0, i)].set_facecolor(_header_bg)
            tbl[(0, i)].set_text_props(weight="bold", color=_header_text)

        for i in range(1, len(rows) + 1): # 각 행 스타일 설정
            for j in range(6):
                cell = tbl[(i, j)]
                cell.set_facecolor(_row_even if i % 2 == 0 else _row_odd) # 짝/홀수 행 배경색
                cell.get_text().set_color(_txt_color) # 텍스트 색상

    ax.set_title("포트폴리오 성과 요약", fontsize=16, fontweight="bold", pad=20) # 차트 제목
    apply_uniform_layout(fig, with_colorbar=False) # 레이아웃 통일
    out["portfolio_summary_table"] = _b64_and_close(fig) # base64로 변환하여 저장

    return out # 결과 딕셔너리 반환

# 포트폴리오 성과 요약 데이터 생성 함수
def create_portfolio_summary_data(analyzer):
    strategies = { # 포트폴리오 전략
        "동일가중": analyzer.equal_weight_returns,
        "최소분산": analyzer.min_var_returns,
        "최대샤프": analyzer.max_sharpe_returns,
    }
    if hasattr(analyzer, "dynamic_returns"): # 동적 전략이 있으면 추가
        strategies["동적전략"] = analyzer.dynamic_returns

    rows = [] # 결과 행 데이터를 담을 리스트
    for name, r in strategies.items(): # 각 전략에 대해 반복
        if len(r) == 0: # 데이터가 없으면 건너뜀
            continue
        cum = (1 + r).cumprod() # 누적 수익률
        ann_ret = r.mean() * 252 # 연간 수익률
        vol = r.std() * (252 ** 0.5) # 연간 변동성
        sharpe = (ann_ret / vol) if vol > 0 else 0.0 # 샤프 비율 (변동성이 0일때 예외 처리)
        mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min() # 최대 낙폭
        total_ret = (cum.iloc[-1] - 1.0) # 총 수익률

        # 계산된 지표를 딕셔너리로 추가
        rows.append({ 
            "전략": name,
            "연간수익률": round(ann_ret * 100, 1), # % 단위, 소수점 1자리
            "변동성": round(vol * 100, 1), # % 단위, 소수점 1자리
            "샤프비율": round(sharpe, 2), # 소수점 2자리
            "최대낙폭": round(mdd * 100, 1), # % 단위, 소수점 1자리
            "총수익률": round(total_ret * 100, 1), # % 단위, 소수점 1자리
        })

     # 최종 JSON 데이터 반환
    return {
        "columns": ["전략", "연간수익률", "변동성", "샤프비율", "최대낙폭", "총수익률"],
        "rows": rows,
        "unitHints": {"연간수익률":"%","변동성":"%","최대낙폭":"%","총수익률":"%"}, # 단위 정보
    }