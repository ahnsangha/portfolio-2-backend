# ML 시각화 유틸 함수 모음
# Figure만 반환하고 저장과 인코딩은 라우터에서 처리

from .helpers import setup_korean_font, set_plot_theme, get_theme_colors, get_ml_theme_colors
import matplotlib.pyplot as plt # 차트 그리기 도구
import matplotlib.dates as mdates # 날짜 축 포맷 도구
from matplotlib import rcParams # 전역 렌더링 옵션 제어
import pandas as pd # 표 형태 데이터 처리
import numpy as np # 수치 계산
import logging # 로깅을 위해 추가
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# 로거 설정
logger = logging.getLogger(__name__)

# ML 차트 생성
def create_ml_chart(analyzer=None, ml_results=None, theme: str = "light",
                    show_debug_footer: bool = False, show_summary_footer: bool = True,
                    verbose: bool = False):

    # 폰트 설정
    setup_korean_font()
    set_plot_theme(theme)

    # 0) 이미 결과가 넘어왔다면 그대로 사용
    if ml_results is not None and isinstance(ml_results, dict) and ml_results:
        return _render_ml_chart_from_results(ml_results, theme=theme,
                                             show_debug_footer=show_debug_footer,
                                             show_summary_footer=show_summary_footer)

    # 1) analyzer가 api를 갖고 있다면 사용 
    if analyzer is not None:
        for attr in ("run_ml_analysis", "run_all"):
            fn = getattr(analyzer, attr, None)
            if callable(fn):
                try:
                    results = fn(k_clusters=4, horizon=5, use_gru=True)
                    if isinstance(results, dict) and results:
                        return _render_ml_chart_from_results(results, theme=theme,
                                                             show_debug_footer=show_debug_footer,
                                                             show_summary_footer=show_summary_footer)
                except Exception:
                    pass # 없는 경우가 많으므로 조용히 다음 후보 시도

    # 2) services.ml 내 계산 함수 시도 
    try:
        from services import ml as ml_service
        for name in ("compute_ml_results", "build_ml_results", "make_results", "analyze", "build", "run"):
            fn = getattr(ml_service, name, None)
            if callable(fn):
                try:
                    results = fn(analyzer) if analyzer is not None else fn()
                    if isinstance(results, dict) and results:
                        return _render_ml_chart_from_results(results, theme=theme,
                                                             show_debug_footer=show_debug_footer,
                                                             show_summary_footer=show_summary_footer)
                except TypeError:
                    # 시그니처가 다르면 다음 후보
                    continue
                except Exception:
                    # 내부 에러는 폴백으로
                    break
    except Exception:
        pass

    # 3) 최종 폴백: 예외 대신 안내 이미지를 만든다
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.5, 0.6, "AI 예측 준비 필요", ha="center", va="center", fontsize=16, transform=ax.transAxes)
    ax.text(0.5, 0.44,
            "services/ml.py에 ML 계산 함수를 연결하거나\n"
            "분석기의 run_ml_analysis / run_all 구현이 필요합니다.",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    if show_debug_footer:
        ax.text(0.5, 0.08, "debug: no ml_results, analyzer API 미탑재", ha="center", va="center",
                fontsize=9, transform=ax.transAxes, alpha=0.7)
    return fig

def _render_ml_chart_from_results(ml_results: dict, theme: str = "light",
                                  show_debug_footer: bool = False,
                                  show_summary_footer: bool = True):

    # 더 작게
    FS_TITLE  = 10 # suptitle 자체를 없애므로 사실상 영향 없음
    FS_ST     = 9
    FS_LABEL  = 8
    FS_TICK   = 7
    FS_ANN    = 7

    fig = plt.figure(figsize=(15, 8.2))
    gs  = GridSpec(2, 2, height_ratios=[1.05, 1.0], figure=fig)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_weights = fig.add_subplot(gs[0, 1])
    ax_ic      = fig.add_subplot(gs[1, :])
    

    # 테마 색상 강제 적용
    c = get_theme_colors(theme)
    m = get_ml_theme_colors(theme)
    is_dark = (theme or "light").lower().startswith("dark")

    fig.patch.set_facecolor(c["figure_facecolor"])
    for ax in (ax_scatter, ax_weights, ax_ic):
        ax.set_facecolor(m["axes_facecolor"] if is_dark else c["axes_facecolor"])
        for sp in ax.spines.values():
            sp.set_color(c["spine_color"])
        if is_dark:
            ax.tick_params(colors=c["tick_color"])
            ax.xaxis.label.set_color(c["text_color"])
            ax.yaxis.label.set_color(c["text_color"])
        ax.set_axisbelow(True)  # 그리드가 항상 데이터 뒤로 가도록

    # 하단 겹침/잘림 방지: 여백을 넉넉히 고정
    fig.subplots_adjust(
        left=0.07, right=0.985,
        top=0.90, bottom=0.18,   # ← 핵심: 아래 여백 늘림
        hspace=0.42, wspace=0.30
    )

    # 1) PCA 산점도 
    xs, ys, df_coords = None, None, None
    pca_obj = ml_results.get("pca")
    if isinstance(pca_obj, dict) and isinstance(pca_obj.get("coords"), list):
        df_coords = pd.DataFrame(pca_obj["coords"])
        for kx, ky in (("PC1","PC2"), ("x","y"), ("pc1","pc2")):
            if kx in df_coords.columns and ky in df_coords.columns:
                xs, ys = df_coords[kx].astype(float), df_coords[ky].astype(float)
                break
        if "ticker" not in df_coords.columns:
            df_coords["ticker"] = [f"T{i}" for i in range(len(df_coords))]
    if xs is None and "pca2d" in ml_results:
        try:
            pca2d = ml_results["pca2d"]
            xs, ys = pca2d["x"], pca2d["y"]
            df_coords = pd.DataFrame({"ticker": getattr(pca2d, "index", range(len(xs))), "x": xs, "y": ys})
        except Exception:
            pass

    labels_map = None
    clusters = ml_results.get("clusters") or {}
    if isinstance(clusters, dict) and isinstance(clusters.get("labels"), dict):
        labels_map = clusters["labels"]

    if df_coords is not None and xs is not None and ys is not None:
        if labels_map:
            df_coords["cluster"] = df_coords["ticker"].map(lambda t: labels_map.get(str(t)))
            palette = m["cluster_palette"]
            uniques = sorted(df_coords["cluster"].dropna().unique())
            for idx, cid in enumerate(uniques):
                sel = df_coords["cluster"] == cid
                ax_scatter.scatter(
                    df_coords.loc[sel, xs.name], df_coords.loc[sel, ys.name],
                    s=26, alpha=0.9, label=f"Cluster {int(cid)}",
                    color=palette[idx % len(palette)], edgecolors=m["edge"]
                )
            sel = df_coords["cluster"].isna()
            if sel.any():
                ax_scatter.scatter(
                    df_coords.loc[sel, xs.name], df_coords.loc[sel, ys.name],
                    s=24, alpha=0.7, label="Unlabeled",
                    color=m["accent1"], edgecolors=m["edge"]
                )
            leg = ax_scatter.legend(loc="best", fontsize=FS_TICK-1)
            if leg:
                leg.get_frame().set_facecolor(c["axes_facecolor"])
                leg.get_frame().set_edgecolor(c["spine_color"])
                for txt in leg.get_texts():
                    txt.set_color(c["text_color"])
        else:
            ax_scatter.scatter(xs, ys, s=26, alpha=0.95, color=m["accent1"], edgecolors=m["edge"])
        ax_scatter.set_title("PCA (2D) with KMeans clusters", fontsize=FS_ST)
        ax_scatter.set_xlabel(xs.name if xs is not None else "PC1", fontsize=FS_LABEL)
        ax_scatter.set_ylabel(ys.name if ys is not None else "PC2", fontsize=FS_LABEL)
        ax_scatter.tick_params(labelsize=FS_TICK)
        ax_scatter.grid(
            True,
            color=m["grid_color"],
            alpha=(0.10 if is_dark else m["grid_alpha"]), # 0.16 → 0.10
            linewidth=(0.45 if is_dark else 0.8) # 0.6  → 0.45
        )
        # 다크에서만 텍스트, 범례 톤 보정
        if is_dark:
            ax_scatter.set_title("PCA (2D) with KMeans clusters", fontsize=FS_ST, color=c["text_color"])
            ax_scatter.set_xlabel(xs.name if xs is not None else "PC1", fontsize=FS_LABEL, color=c["text_color"])
            ax_scatter.set_ylabel(ys.name if ys is not None else "PC2", fontsize=FS_LABEL, color=c["text_color"])
            ax_scatter.tick_params(labelsize=FS_TICK, colors=c["tick_color"])
            leg = ax_scatter.get_legend()
            if leg:
                leg.get_frame().set_facecolor(m["axes_facecolor"])
                leg.get_frame().set_edgecolor(c["spine_color"])
                for txt in leg.get_texts():
                    txt.set_color(c["text_color"])
    else:
        ax_scatter.text(0.5, 0.5, "PCA 좌표 없음", ha="center", va="center", alpha=0.8,
                        transform=ax_scatter.transAxes, fontsize=FS_ST)
        ax_scatter.axis("off")

    # 2) HRP 상위 가중치 
    weights = None
    hrp = ml_results.get("hrp") or {}
    if isinstance(hrp, dict) and isinstance(hrp.get("weights"), dict):
        weights = hrp["weights"]
    if not weights:
        weights = ml_results.get("today_weights") or (ml_results.get("picks") or {}).get("today_weights")

    if isinstance(weights, dict) and weights:
        import numpy as np
        from services.data import stock_manager  # 종목 메타(이름/코드/티커) 캐시
        s = pd.Series(weights, dtype=float).sort_values(ascending=False).head(20)

        # 티커 -> 종목명 매핑 생성 
        stocks = stock_manager.get_all_stocks()
        name_by_ticker = {row["ticker"]: row["name"] for row in stocks}
        name_by_code   = {row["code"]:   row["name"] for row in stocks}

        def labelize(t):
            t = str(t)
            # 1) 완전한 티커 -> 이름 (예: 064260.KS/KQ)
            if t in name_by_ticker:
                return name_by_ticker[t]
            # 2) 코드만 추출 후 매핑 시도
            code = t.split(".")[0]
            return name_by_code.get(code, t) # 실패 시 원래 티커 표시

        labels = [labelize(lbl) for lbl in s.index]

        # 라벨 적용
        y = np.arange(len(s))
        ax_weights.barh(y, s.values, height=0.6, color=m["accent2"], edgecolor=m["edge"])
        ax_weights.set_yticks(y)
        ax_weights.set_yticklabels(labels)
        ax_weights.invert_yaxis()  # 상위 가중치가 위로

        ax_weights.set_title("HRP Portfolio Weights (Top 20)", fontsize=FS_ST, pad=2)
        ax_weights.set_xlabel("Weight", fontsize=FS_LABEL)

        # 오른쪽/왼쪽 잘림 여유
        xmax = float(s.values.max()) if len(s) else 1.0
        ax_weights.set_xlim(0, xmax * 1.12)
        ax_weights.margins(x=0.06, y=0.08)

        ax_weights.tick_params(labelsize=FS_TICK)
        for lbl in ax_weights.get_yticklabels():
            lbl.set_fontsize(FS_TICK)

        ax_weights.grid(
            True, axis="x",
            color=m["grid_color"],
            alpha=(0.10 if is_dark else m["grid_alpha"]),
            linewidth=(0.45 if is_dark else 0.8)
        )
        # 다크에서만 텍스트 톤 보정
        if is_dark:
            ax_weights.set_title("HRP Portfolio Weights (Top 20)", fontsize=FS_ST, pad=2, color=c["text_color"])
            ax_weights.set_xlabel("Weight", fontsize=FS_LABEL, color=c["text_color"])
            ax_weights.set_yticklabels(labels, color=c["text_color"])
            ax_weights.tick_params(colors=c["tick_color"])
    else:
        ax_weights.text(0.5, 0.5, "가중치 정보 없음", ha="center", va="center", alpha=0.8,
                        transform=ax_weights.transAxes, fontsize=FS_ST)
        ax_weights.axis("off")

    # 3) Rank IC 시계열 
    pred = ml_results.get("prediction") or {}
    ic_series = pred.get("ic_by_date") or {}
    if isinstance(ic_series, dict) and ic_series:
        ser = pd.Series(ic_series, dtype=float)
        try:
            ser.index = pd.to_datetime(ser.index)
            ser = ser.sort_index()
        except Exception:
            pass

        # 라인 그리기
        ax_ic.plot(ser.index, ser.values, marker="o", linewidth=1.2, markersize=2.4, color=m["accent1"])
        ax_ic.axhline(0.0, linewidth=(0.7 if is_dark else 0.8),
                    alpha=(0.55 if is_dark else 0.8), color=m["zero_line"])

        # 제목/라벨 작게 + 패딩 축소
        ax_ic.set_title("Cross-Sectional Rank IC", fontsize=FS_ST, pad=2)
        ax_ic.set_ylabel("Rank IC", fontsize=FS_LABEL)
        ax_ic.tick_params(labelsize=FS_TICK)

        # 다크에서만 텍스트 톤 보정
        if is_dark:
            ax_ic.set_title("Cross-Sectional Rank IC", fontsize=FS_ST, pad=2, color=c["text_color"])
            ax_ic.set_ylabel("Rank IC", fontsize=FS_LABEL, color=c["text_color"])
            ax_ic.tick_params(colors=c["tick_color"])

        # 날짜 라벨 과밀 방지
        locator = AutoDateLocator(minticks=4, maxticks=6)
        ax_ic.xaxis.set_major_locator(locator)
        ax_ic.xaxis.set_major_formatter(AutoDateFormatter(locator))

        # 위/아래 여유를 더 주어 겹침/잘림 제거
        import numpy as _np
        ymin = float(_np.nanmin(ser.values)) if len(ser) else -1.0
        ymax = float(_np.nanmax(ser.values)) if len(ser) else  1.0
        pad  = 0.08
        ax_ic.set_ylim(max(-1.1, ymin - pad), min(1.1, ymax + pad))
        ax_ic.margins(x=0.02, y=0.08)

        # 요약 텍스트도 축 내부에 작게
        try:
            avg_ic = float(pred.get("ic")) if "ic" in pred else float(_np.nanmean(ser.values))
            hit = pred.get("hit_rate"); r2 = pred.get("r2")
            if is_dark:
                ax_ic.text(0.01, 1.01,
                        f"Avg IC: {avg_ic:.3f} | Hit Rate: {hit:.3f} | R²: {r2:.3f}",
                        transform=ax_ic.transAxes, fontsize=FS_ANN, color=c["text_color"])
            else:
                ax_ic.text(0.01, 1.01,
                        f"Avg IC: {avg_ic:.3f} | Hit Rate: {hit:.3f} | R²: {r2:.3f}",
                        transform=ax_ic.transAxes, fontsize=FS_ANN)
        except Exception:
            pass

        ax_ic.grid(
            True,
            color=m["grid_color"],
            alpha=(0.10 if is_dark else m["grid_alpha"]),
            linewidth=(0.45 if is_dark else 0.8)
        )
    else:
        ax_ic.text(0.5, 0.5, "IC 시계열 없음", ha="center", va="center", alpha=0.8,
                transform=ax_ic.transAxes, fontsize=FS_ST)
        ax_ic.axis("off")

    # 하단 메시지가 있으면 너무 아래로 내려가지 않도록 y=0.06
    msgs = ml_results.get("ui_messages") or []
    if show_summary_footer and msgs:
        try:
            text = " · ".join(str(m) for m in msgs[-3:])
            fig.text(
                0.5, 0.075, text, # 0.06 → 0.075 
                ha="center", va="center",
                fontsize=(FS_ANN + 1 if is_dark else FS_ANN),
                color=(c["text_color"] if is_dark else "#111111"),
                alpha=(0.95 if is_dark else 0.7) # 밝게
            )
        except Exception:
            pass

    # 제목/라벨 호출로 색이 초기화되는 경우 방지
    if is_dark:
        for ax in (ax_scatter, ax_weights, ax_ic):
            ax.set_facecolor(m["axes_facecolor"])
            for sp in ax.spines.values():
                sp.set_color(c["spine_color"])
            ax.tick_params(colors=c["tick_color"])
            if ax.xaxis and ax.xaxis.label:
                ax.xaxis.label.set_color(c["text_color"])
            if ax.yaxis and ax.yaxis.label:
                ax.yaxis.label.set_color(c["text_color"])
            ttl = ax.get_title()
            if ttl:
                ax.set_title(ttl, color=c["text_color"])
            ax.set_axisbelow(True) # 안전빵으로 한 번 더

    # 겹침/잘림 제거로 여백 고정
    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.985, hspace=0.46, wspace=0.34)
    return fig

