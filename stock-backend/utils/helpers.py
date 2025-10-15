# 공통 헬퍼(UTILS – PLOT)
# 한글 폰트 자동 설정(setup_korean_font)
# 날짜축 포맷(format_date_axis)
# 캡션(inside/outside) 유틸
# 틱/축 스타일
# fig 저장(fig_to_base64) <- tight-crop 제거
# 카테고리형 월 라벨
# 공통 레이아웃 강제
# 테마(light/dark) 설정

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import platform
import io, base64
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.colors import to_rgba
import matplotlib.font_manager as fm
import os
from pathlib import Path

matplotlib.use("Agg")

# Theme light/dark
THEMES = {
    "light": {
        "figure_facecolor": "#FFFFFF",
        "axes_facecolor":   "#FFFFFF",
        "grid_color":       "#9da4a8",
        "text_color":       "#111827",
        "tick_color":       "#374151",
        "spine_color":      "#D1D5DB",
        "legend_facecolor": "#FFFFFF",
        "legend_edgecolor": "#D1D5DB",
        "table_header_bg":  "#4CAF50",
        "table_header_text":"#FFFFFF",
        "table_row_even":   "#F3F4F6",
        "table_row_odd":    "#FFFFFF",
    },
    "dark": {
        "figure_facecolor": "#191923",
        "axes_facecolor":   "#191923",
        "grid_color":       "#3b3b44",
        "text_color":       "#E7E9EE",
        "tick_color":       "#C9CDD6",
        "spine_color":      "#383E49",
        "legend_facecolor": "#161A21",
        "legend_edgecolor": "#3A404C",
        "table_header_bg":  "#1F2937",
        "table_header_text":"#E5E7EB",
        "table_row_even":   "#141821",
        "table_row_odd":    "#0F1116",
    },
}

def get_theme_colors(theme: str) -> dict:
    return THEMES.get((theme or "light").lower(), THEMES["light"])

# charts.py 쪽에서 호출해 사용
def set_plot_theme(theme: str = "light") -> None:

    c = get_theme_colors(theme)
    rc = plt.rcParams
    rc["figure.facecolor"] = c["figure_facecolor"]
    rc["axes.facecolor"]   = c["axes_facecolor"]
    rc["axes.edgecolor"]   = c["spine_color"]
    rc["axes.labelcolor"]  = c["text_color"]
    rc["xtick.color"]      = c["tick_color"]
    rc["ytick.color"]      = c["tick_color"]
    rc["grid.color"]       = c["grid_color"]
    rc["text.color"]       = c["text_color"]
    rc["savefig.facecolor"]= c["figure_facecolor"]
    rc["savefig.edgecolor"]= c["figure_facecolor"]
    rc["legend.frameon"]   = True
    rc["legend.facecolor"] = c["legend_facecolor"]
    rc["legend.edgecolor"] = c["legend_edgecolor"]

    # 캡션 톤을 테마에 맞춰 동기화
    global CAPTION_TEXT_COLOR, CAPTION_OUTSIDE
    CAPTION_TEXT_COLOR = c["text_color"]
    CAPTION_OUTSIDE["color"] = CAPTION_TEXT_COLOR


# 설정이 이미 완료되었는지 확인하기 위한 플래그
_korean_font_setup_done = False

def setup_korean_font():
    """
    백엔드에 포함된 나눔고딕 폰트를 Matplotlib에 전역으로 설정합니다.
    애플리케이션 시작 시 한 번만 실행되도록 설계되었습니다.
    """
    global _korean_font_setup_done
    if _korean_font_setup_done:
        return

    print("--- initializing korean font setup ---")
    
    try:
        # 1. 폰트 파일이 있는 디렉터리 경로를 확인합니다.
        font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fonts'))
        print(f"searching for fonts in: {font_dir}")

        if not os.path.isdir(font_dir):
            print(f"error: font directory not found at {font_dir}")
            _korean_font_setup_done = True
            return

        # 2. 해당 디렉터리에서 .ttf 폰트 파일을 찾습니다.
        font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.lower().endswith('.ttf')]
        
        if not font_files:
            print(f"error: no .ttf font files found in {font_dir}")
            _korean_font_setup_done = True
            return

        print(f"found {len(font_files)} font file(s): {[os.path.basename(f) for f in font_files]}")

        # 3. 찾은 폰트들을 Matplotlib에 추가합니다.
        for font_path in font_files:
            fm.fontManager.addfont(font_path)
        
        # 4. Matplotlib의 기본 설정을 'NanumGothic'으로 변경합니다.
        matplotlib.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
        
        # 5. 설정이 잘 되었는지 최종 확인합니다.
        font_check = fm.findfont('NanumGothic', fallback_to_default=False)
        print(f"verification: 'NanumGothic' resolved to: {font_check}")
        print("--- korean font setup completed successfully ---")

    except Exception as e:
        # 폰트를 찾지 못하면 여기서 오류가 발생합니다.
        print(f"--- error during font setup: {e} ---")
        print("falling back to default sans-serif font.")
        matplotlib.rc('font', family='sans-serif')
    
    finally:
        _korean_font_setup_done = True


def fig_to_base64(fig, **kwargs):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, **kwargs)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def get_theme_colors(theme: str = "light"):
    if (theme or "light").lower().startswith("dark"):
        return {
            "figure_facecolor": "#1E293B",
            "axes_facecolor": "#334155",
            "text_color": "#F1F5F9",
            "grid_color": "#475569",
            "spine_color": "#94A3B8",
            "tick_color": "#CBD5E1",
        }
    return {
        "figure_facecolor": "white",
        "axes_facecolor": "#F8FAFC",
        "text_color": "black",
        "grid_color": "#E2E8F0",
        "spine_color": "#94A3B8",
        "tick_color": "#64748B",
    }

def get_ml_theme_colors(theme: str = "light"):
    if (theme or "light").lower().startswith("dark"):
        return {
            "figure_facecolor": "#161b22",
            "axes_facecolor": "#0d1117",
            "text_color": "#c9d1d9",
            "grid_color": "#21262d",
            "spine_color": "#30363d",
            "tick_color": "#8b949e",
        }
    return get_theme_colors(theme)


# 날짜축 포맷 
def format_date_axis(ax, interval_months: int | None = None, max_ticks: int = 8):
    import numpy as np
    import matplotlib.dates as mdates
    x0, x1 = ax.get_xlim()
    d0, d1 = mdates.num2date(x0), mdates.num2date(x1)
    span_days = max(1, (d1 - d0).days)
    span_months = max(1, int(round(span_days / 30.44)))

    if span_months >= 24:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=max_ticks)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        if span_months < 60:
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        else:
            ax.xaxis.set_minor_locator(mdates.YearLocator())
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0); lbl.set_ha("center")
        ax.tick_params(axis="x", which="major", pad=6)
        ax.grid(axis="x", which="major", alpha=0.25)
        ax.margins(x=0.01)
        return

    if interval_months is None:
        import numpy as np
        interval = max(1, int(np.ceil(span_months / max_ticks)))
    else:
        interval = max(1, int(interval_months))
        if span_months / interval > (max_ticks + 2):
            import numpy as np
            interval = int(np.ceil(span_months / max_ticks))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    if span_months < 12:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0); lbl.set_ha("center")
    ax.tick_params(axis="x", which="major", pad=6)
    ax.grid(axis="x", which="major", alpha=0.25)
    ax.margins(x=0.01)

# 캡션 유틸 
CAPTION_TEXT_COLOR = '#111827' # 글자색
CAPTION_BG_COLOR   = '#0b1220' # 박스 배경
CAPTION_EDGE_COLOR = '#8da2fb' # 박스 테두리
CAPTION_TEXT_ALPHA = 1.0
CAPTION_BG_ALPHA   = 0.03

def put_inside_caption(ax, text, loc='tr', fontsize=9, alpha=CAPTION_BG_ALPHA, pad=0.28):
    _x, _y = {
        'tl': (0.01, 0.99), 'tr': (0.99, 0.99),
        'bl': (0.01, 0.01), 'br': (0.99, 0.01),
    }.get(loc, (0.99, 0.99))

    ax.text(
        _x, _y, text,
        fontsize=fontsize, color=to_rgba(CAPTION_TEXT_COLOR, CAPTION_TEXT_ALPHA),
        ha='left' if 'l' in loc else 'right',
        va='top'  if 't' in loc else 'bottom',
        transform=ax.transAxes,
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor=to_rgba(CAPTION_BG_COLOR, alpha),
            edgecolor=to_rgba(CAPTION_EDGE_COLOR, alpha * 0.7),
            linewidth=0.6
        )
    )

USE_GLOBAL_CAPTION_STYLE = True
CAPTION_OUTSIDE = {
    "where": "top-left",
    "fontsize": 12,
    "color": CAPTION_TEXT_COLOR, # 글자색 변수 사용
    "alpha": 0.5,
    "dx": 0.004,
    "dy": -0.012,
}

def put_outside_caption(
    ax, text: str, *,
    where: str | None = None, fontsize: float | None = None,
    color: str | None = None, alpha: float | None = None,
    dx: float | None = None, dy: float | None = None,
    force_local: bool = False,
):
    pos_map = {
        "top-left": (0.0, 1.02, "left",  "bottom"),
        "top-right": (1.0, 1.02, "right", "bottom"),
        "bottom-left": (0.0, -0.12, "left",  "top"),
        "bottom-right": (1.0, -0.12, "right", "top"),
    }
    
    if USE_GLOBAL_CAPTION_STYLE and not force_local:
        where = CAPTION_OUTSIDE["where"]
        fontsize = CAPTION_OUTSIDE["fontsize"]
        color = CAPTION_OUTSIDE["color"]
        alpha = CAPTION_OUTSIDE["alpha"]
        dx = CAPTION_OUTSIDE["dx"]
        dy = CAPTION_OUTSIDE["dy"]

    else:
        where = where or CAPTION_OUTSIDE["where"]
        fontsize = fontsize or CAPTION_OUTSIDE["fontsize"]
        color = color or CAPTION_OUTSIDE["color"]
        alpha = alpha if alpha is not None else CAPTION_OUTSIDE["alpha"]
        dx = dx if dx is not None else CAPTION_OUTSIDE["dx"]
        dy = dy if dy is not None else CAPTION_OUTSIDE["dy"]

    x0, y0, ha, va = pos_map.get(where, pos_map["top-left"])
    ax.text(
        x0 + dx, y0 + dy, text,
        transform=ax.transAxes, ha=ha, va=va,
        fontsize=fontsize, color=color, alpha=alpha,
        clip_on=False,
    )

# 틱/축 스타일
def style_value_axis(ax, *, axis="y", nbins=5, color="#777", percent=False, decimals=2):
    tgt = ax.yaxis if axis == "y" else ax.xaxis
    tgt.set_major_locator(MaxNLocator(nbins=nbins))
    if percent:
        tgt.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    else:
        fmt = f"%.{decimals}f"
        tgt.set_major_formatter(FuncFormatter(lambda v, p: fmt % v))
    ax.tick_params(axis=axis, colors=color, labelsize=9, pad=4)

# 통일 레이아웃
# 일반/컬러바 포함일 때 여백을 고정
PLOT_LAYOUT = dict(left=0.08, right=0.94, top=0.92, bottom=0.14, wspace=0.28, hspace=0.34)
PLOT_LAYOUT_WITH_CBAR = dict(left=0.08, right=0.90, top=0.92, bottom=0.14, wspace=0.28, hspace=0.34)

def apply_uniform_layout(fig, *, with_colorbar: bool = False):
    kw = PLOT_LAYOUT_WITH_CBAR if with_colorbar else PLOT_LAYOUT
    fig.subplots_adjust(**kw)

# fig → base64 
def fig_to_base64(
    fig,
    *,
    title_size: int = 26, # 제목 더 크게
    title_pad: int = 20, # 그래프와 제목 간격 더 띄움
    legend_right_margin: float = 0.89
):
    try:
        # figure-level 제목도 키우고 살짝 위로
        st = getattr(fig, "_suptitle", None)
        if st is not None:
            st.set_fontsize(title_size + 2)
            st.set_fontweight("bold")
            try:
                y = getattr(st, "get_position", lambda: (0, 0.98))()[1]
                st.set_y(min(0.99, y + 0.01)) # 조금 더 위로
            except Exception:
                pass

        need_extra_right = False
        for ax in fig.get_axes():
            # 축 제목: 크고 굵게 + 가운데 정렬 + pad 위로
            t = ax.get_title()
            if t:
                ax.set_title(
                    t,
                    fontsize=title_size,
                    fontweight="bold",
                    loc="center",
                    pad=title_pad,
                )

            # 범례: 그래프 바깥 우상단(유지)
            leg = ax.get_legend()
            if leg is not None:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=False,
                )
                need_extra_right = True

        # legend 있을 때만 오른쪽 여백 확보
        if need_extra_right:
            current_right = getattr(fig.subplotpars, "right", 1.0)
            fig.subplots_adjust(right=min(current_right, legend_right_margin))

    except Exception:
        pass

    # 저장 시 facecolor/edgecolor는 현재 rcParams값을 사용
    save_face = plt.rcParams.get("savefig.facecolor", plt.rcParams.get("figure.facecolor", "white"))
    save_edge = plt.rcParams.get("savefig.edgecolor", "none")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=save_face, edgecolor=save_edge)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

# 카테고리형 월 라벨 
def format_month_axis_categorical(ax, dates, max_ticks: int = 8, color: str = "#777"):
    import numpy as np
    import pandas as pd
    if isinstance(dates, pd.PeriodIndex):
        dti = dates.to_timestamp()
    else:
        dti = pd.to_datetime(dates)
    n = len(dti)
    idx = np.arange(n)
    step = max(1, int(np.ceil(n / max_ticks)))
    pick = idx[::step]
    if pick[-1] != idx[-1]:
        pick = np.append(pick, idx[-1])

    labels, prev_year = [], None
    for i in pick:
        dt = dti[i]
        if prev_year is None or dt.year != prev_year:
            labels.append(dt.strftime("%Y-%m")); prev_year = dt.year
        else:
            labels.append(dt.strftime("%m"))

    ax.set_xticks(pick)
    ax.set_xticklabels(labels)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0); lbl.set_ha("center")
    ax.tick_params(axis="x", colors=color, labelsize=9, pad=4)

# ML 전용 팔레트 
ML_THEMES = {
    "light": {
        "figure_facecolor": "#ffffff",
        "axes_facecolor":   "#ffffff",
        "spine_color":      "#9aa0a6",
        "tick_color":       "#374151",
        "text_color":       "#1f2937",
        "grid_color":       "#e5e7eb",
        "grid_alpha":       0.6,
        "accent1":          "#2563eb", # PCA/IC 기본 라인,점
        "accent2":          "#16a34a", # 가중치 막대
        "accent3":          "#f43f5e",
        "edge":             "#0f172a",
        "zero_line":        "#94a3b8",
        "cluster_palette":  ["#2563eb","#16a34a","#f59e0b","#ef4444","#a855f7","#06b6d4","#f43f5e","#84cc16"],
    },
    "dark": {
        "figure_facecolor": "#0f1115",
        "axes_facecolor":   "#13151a",
        "spine_color":      "#3b3f45",
        "tick_color":       "#cbd5e1",
        "text_color":       "#e5e7eb",
        "grid_color":       "#2a2f3a",
        "grid_alpha":       0.35,
        "accent1":          "#8ab4f8", # PCA/IC 기본 라인, 점
        "accent2":          "#34d399", # 가중치 막대
        "accent3":          "#f472b6",
        "edge":             "#1f2937",
        "zero_line":        "#94a3b8",
        "cluster_palette":  ["#8ab4f8","#34d399","#fbbf24","#f87171","#c084fc","#22d3ee","#f472b6","#a3e635"],
    },
}

# ML 예측 차트 전용 팔레트 반환
def get_ml_theme_colors(theme: str) -> dict:
    t = (theme or "light").lower()
    return ML_THEMES["dark"] if t.startswith("dark") else ML_THEMES["light"]
