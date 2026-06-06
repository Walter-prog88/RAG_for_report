"""
主程序：费城半导体指数 (SOX) 与标普500看涨情绪关系研究
运行方式：python sox_research/run.py
"""

import sys
from pathlib import Path
from typing import Optional
import platform

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# 根据操作系统自动选择中文字体
_system = platform.system()
if _system == "Darwin":
    matplotlib.rcParams["font.family"] = ["STHeiti", "Heiti TC", "Arial Unicode MS", "PingFang HK"]
elif _system == "Windows":
    matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "SimSun"]
else:
    matplotlib.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from sox_research.fetch import load, get_current_pc_ratio
from sox_research.analysis import (
    compute_indicators,
    compute_crisis_score,
    historical_stats,
    get_crisis_score_at,
    CRISIS_PERIODS,
    BUILDUP_PERIODS,
)

CHARTS = Path(__file__).parent / "charts"
CHARTS.mkdir(exist_ok=True)

# ── 样式 ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "legend.framealpha": 0.3,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

CRISIS_COLOR   = "#f85149"
BUILDUP_COLOR  = "#d29922"
SOX_COLOR      = "#58a6ff"
SPX_COLOR      = "#3fb950"
VIX_COLOR      = "#bc8cff"
SCORE_COLOR    = "#ff7b72"


def _shade_crises(ax, alpha_crisis=0.18, alpha_buildup=0.10):
    for label, (s, e) in CRISIS_PERIODS.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=CRISIS_COLOR, alpha=alpha_crisis, zorder=0)
    for label, (s, e) in BUILDUP_PERIODS.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=BUILDUP_COLOR, alpha=alpha_buildup, zorder=0)


def _fmt_date(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")


# ── Chart 1：SOX vs SPX 长期归一化走势 ────────────────────────────────────────
def chart_normalized(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})
    fig.suptitle("费城半导体指数 (SOX) vs 标普500 (SPX) — 长期走势", fontsize=14, y=0.98)

    # 归一化为 100（起始点）
    base = df.first_valid_index()
    norm = df[["SOX", "SPX"]] / df[["SOX", "SPX"]].loc[base] * 100

    ax = axes[0]
    ax.semilogy(norm.index, norm["SOX"], color=SOX_COLOR, lw=1.5, label="SOX（半导体）")
    ax.semilogy(norm.index, norm["SPX"], color=SPX_COLOR, lw=1.5, label="SPX（标普500）")
    _shade_crises(ax)
    ax.set_ylabel("对数指数（1993 = 100）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_title("① 价格走势（对数坐标）", fontsize=10, pad=4)

    # SOX / SPX 相对强度比率
    ax2 = axes[1]
    ratio = norm["SOX"] / norm["SPX"]
    ax2.plot(ratio.index, ratio, color=SOX_COLOR, lw=1.2, label="SOX/SPX 比率")
    ax2.axhline(ratio.mean(), color="#8b949e", lw=0.8, ls="--", label=f"历史均值 {ratio.mean():.2f}")
    _shade_crises(ax2)
    ax2.set_ylabel("相对强度比率")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.4)
    ax2.set_title("② SOX 相对 SPX 强度（峰值 = 泡沫顶部）", fontsize=10, pad=4)

    # VIX
    ax3 = axes[2]
    if "VIX" in df.columns:
        ax3.fill_between(df.index, df["VIX"], alpha=0.5, color=VIX_COLOR, label="VIX（恐慌指数）")
        ax3.axhline(20, color="#8b949e", lw=0.8, ls="--", label="VIX=20 基准")
        _shade_crises(ax3)
        ax3.set_ylabel("VIX")
        ax3.legend(loc="upper right", fontsize=9)
        ax3.set_title("③ 市场恐慌指数 VIX（低位 = 过度乐观）", fontsize=10, pad=4)

    legend_patches = [
        mpatches.Patch(color=CRISIS_COLOR, alpha=0.4, label="危机期间"),
        mpatches.Patch(color=BUILDUP_COLOR, alpha=0.4, label="泡沫酝酿期"),
    ]
    fig.legend(handles=legend_patches, loc="lower right", fontsize=8, ncol=2, bbox_to_anchor=(0.98, 0.01))

    _fmt_date(axes[-1])
    plt.tight_layout()
    out = CHARTS / "01_sox_spx_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Chart 2：波动率体制 ─────────────────────────────────────────────────────
def chart_volatility(df: pd.DataFrame, ind: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("SOX 波动率体制分析 — 危机先行指标", fontsize=14, y=0.98)

    ax = axes[0]
    ax.plot(ind.index, ind["sox_vol_1m"] * 100, color=SOX_COLOR, lw=1.0, label="SOX 30日实现波动率 %")
    ax.plot(ind.index, ind["sox_vol_3m"] * 100, color="#f0883e", lw=1.3, ls="--", label="SOX 63日实现波动率 %")
    ax.plot(ind.index, ind["spx_vol_1m"] * 100, color=SPX_COLOR, lw=1.0, alpha=0.6, label="SPX 30日实现波动率 %")
    _shade_crises(ax)
    ax.set_ylabel("年化波动率 (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_title("① SOX vs SPX 实现波动率（骤升 = 危机信号）", fontsize=10, pad=4)

    ax2 = axes[1]
    regime = ind["sox_vol_regime"]
    ax2.plot(regime.index, regime, color=SCORE_COLOR, lw=1.0, label="短期/长期波动率比率")
    ax2.axhline(1.0, color="#8b949e", lw=0.8, ls="--")
    ax2.axhline(1.5, color=CRISIS_COLOR, lw=0.8, ls=":", label="警戒线 1.5×")
    ax2.fill_between(regime.index, 1.5, regime.clip(lower=1.5),
                     color=CRISIS_COLOR, alpha=0.3, label="高波动区间")
    _shade_crises(ax2)
    ax2.set_ylabel("波动率体制比率")
    ax2.set_ylim(0, min(regime.quantile(0.995), 4))
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.4)
    ax2.set_title("② 波动率体制（比率 > 1.5 → 危机扩散阶段）", fontsize=10, pad=4)

    _fmt_date(axes[-1])
    plt.tight_layout()
    out = CHARTS / "02_volatility_regimes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Chart 3：情绪与领先指标 ───────────────────────────────────────────────────
def chart_sentiment(df: pd.DataFrame, ind: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2]})
    fig.suptitle("看涨情绪 & SOX 领先指标研究", fontsize=14, y=0.98)

    ax = axes[0]
    ax.plot(ind.index, ind["sox_rel_ret_6m"] * 100, color=SOX_COLOR, lw=1.0, label="SOX - SPX 6个月超额收益 %")
    ax.axhline(0, color="#8b949e", lw=0.6, ls="--")
    ax.axhline(20, color=BUILDUP_COLOR, lw=0.8, ls=":", label="危险阈值 +20%")
    ax.fill_between(ind.index, 0, ind["sox_rel_ret_6m"] * 100,
                    where=ind["sox_rel_ret_6m"] > 0, color=SOX_COLOR, alpha=0.15)
    _shade_crises(ax)
    ax.set_ylabel("超额收益率 (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_title("① SOX 相对 SPX 的 6 个月超额收益（持续大幅领涨 = 泡沫信号）", fontsize=10, pad=4)

    ax2 = axes[1]
    ax2.fill_between(df.index, df["VIX"], alpha=0.4, color=VIX_COLOR)
    ax2.plot(df.index, df["VIX"], color=VIX_COLOR, lw=0.8, label="VIX（恐慌指数，低=乐观）")
    ax2.axhline(15, color=CRISIS_COLOR, lw=0.8, ls=":", label="VIX=15 过度乐观阈值")
    ax2_r = ax2.twinx()
    ax2_r.plot(ind.index, ind["vix_complacency"] * 100, color=BUILDUP_COLOR,
               lw=1.2, ls="--", alpha=0.8, label="VIX 低位指数 (%, 越高越危险)")
    _shade_crises(ax2)
    ax2.set_ylabel("VIX")
    ax2_r.set_ylabel("VIX 低位指数 (%)", color=BUILDUP_COLOR)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.4)
    ax2.set_title("② VIX 情绪指标（低位 = 市场过度乐观，危机前兆）", fontsize=10, pad=4)

    ax3 = axes[2]
    ax3.plot(ind.index, ind["sox_above_200ma"] * 100, color=SOX_COLOR, lw=1.0, label="SOX 偏离 200MA %")
    ax3.plot(ind.index, ind["spx_above_200ma"] * 100, color=SPX_COLOR, lw=1.0,
             alpha=0.7, label="SPX 偏离 200MA %")
    ax3.axhline(30, color=BUILDUP_COLOR, lw=0.8, ls=":", label="警戒线 +30%")
    ax3.axhline(0, color="#8b949e", lw=0.6, ls="--")
    _shade_crises(ax3)
    ax3.set_ylabel("偏离均线 (%)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.4)
    ax3.set_title("③ SOX / SPX 偏离 200 日均线（过热程度）", fontsize=10, pad=4)

    _fmt_date(axes[-1])
    plt.tight_layout()
    out = CHARTS / "03_sentiment_indicators.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Chart 4：复合危机评分 ─────────────────────────────────────────────────────
def chart_crisis_score(df: pd.DataFrame, score: pd.Series, components: pd.DataFrame):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 主图：危机评分
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(score.index, score, color=SCORE_COLOR, lw=1.5, label="复合危机预警评分")
    ax_main.fill_between(score.index, score, alpha=0.2, color=SCORE_COLOR)
    ax_main.axhline(60, color=BUILDUP_COLOR, lw=1.0, ls="--", label="警戒线 60")
    ax_main.axhline(75, color=CRISIS_COLOR, lw=1.0, ls=":", label="高危线 75")

    # 标注当前值
    current_score = score.iloc[-1]
    ax_main.scatter([score.index[-1]], [current_score], color="white", s=60, zorder=5)
    ax_main.annotate(f"当前: {current_score:.1f}",
                     xy=(score.index[-1], current_score),
                     xytext=(-60, 12), textcoords="offset points",
                     color="white", fontsize=10,
                     arrowprops=dict(arrowstyle="->", color="white", lw=0.8))

    _shade_crises(ax_main)
    ax_main.set_ylabel("危机预警评分 (0-100)")
    ax_main.set_ylim(0, 100)
    ax_main.legend(loc="upper left", fontsize=9)
    ax_main.grid(True, alpha=0.4)
    ax_main.set_title("复合危机预警评分（越高越危险）— 2000 年 & 2008 年峰值对比", fontsize=11, pad=6)
    _fmt_date(ax_main)

    # 五个子信号
    comp_meta = [
        ("sox_bubble",     "SOX 泡沫程度",   SOX_COLOR),
        ("vix_complacency","VIX 低位指数",    VIX_COLOR),
        ("sox_leadership", "SOX 领涨过度",    "#f0883e"),
        ("vol_expansion",  "波动率扩张",      SCORE_COLOR),
        ("ratio_zscore",   "SOX/SPX 比率 Z",  "#79c0ff"),
    ]

    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for i, (col, title, color) in enumerate(comp_meta[:4]):
        row, c = positions[i]
        ax = fig.add_subplot(gs[row, c])
        s = components[col] * 100
        ax.plot(s.index, s, color=color, lw=1.0)
        ax.fill_between(s.index, s, alpha=0.2, color=color)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=9, pad=3)
        ax.grid(True, alpha=0.3)
        _shade_crises(ax, alpha_crisis=0.2, alpha_buildup=0.1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        # 当前值标注
        ax.axhline(float(s.iloc[-1]), color="white", lw=0.7, ls=":", alpha=0.6)
        ax.text(0.98, float(s.iloc[-1]) / 100 + 0.02, f"{s.iloc[-1]:.0f}",
                transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color="white", alpha=0.8)

    fig.suptitle("费城半导体 × 标普500看涨情绪 — 经济危机预警系统", fontsize=13, y=0.99)
    out = CHARTS / "04_crisis_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Chart 5：当前状态 vs 历史危机快照 ─────────────────────────────────────────
def chart_current_snapshot(df: pd.DataFrame, ind: pd.DataFrame,
                           score: pd.Series, components: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("当前市场状态 vs 历史危机顶部快照", fontsize=13)

    # 左图：雷达式条形图
    ax = axes[0]
    comp_labels = {
        "sox_bubble":     "SOX 泡沫",
        "vix_complacency":"VIX 低位",
        "sox_leadership": "SOX 领涨",
        "vol_expansion":  "波动扩张",
        "ratio_zscore":   "比率 Z",
    }

    cols = list(comp_labels.keys())
    current_vals = components[cols].iloc[-1] * 100

    # 历史关键节点
    dates_compare = {
        "2000-03-10 (纳指顶部)": "2000-03-10",
        "2007-10-09 (SPX顶部)":  "2007-10-09",
        "2021-11-22 (SPX高点)":  "2021-11-22",
        "当前":                   score.index[-1],
    }
    colors_compare = [CRISIS_COLOR, "#f0883e", BUILDUP_COLOR, "#58a6ff"]

    x = np.arange(len(cols))
    width = 0.18
    for i, (label, dt) in enumerate(dates_compare.items()):
        try:
            idx = score.index.get_indexer([pd.Timestamp(dt)], method="nearest")[0]
            vals = components[cols].iloc[idx] * 100
        except Exception:
            continue
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors_compare[i], alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([comp_labels[c] for c in cols], fontsize=9, rotation=15)
    ax.set_ylim(0, 100)
    ax.set_ylabel("子信号强度 (0-100)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_title("各子信号当前值 vs 历史危机顶部", fontsize=10)

    # 右图：危机评分近3年走势
    ax2 = axes[1]
    recent = score.loc[score.index[-1] - pd.DateOffset(years=3):]
    ax2.plot(recent.index, recent, color=SCORE_COLOR, lw=2.0, label="危机预警评分")
    ax2.fill_between(recent.index, recent, alpha=0.2, color=SCORE_COLOR)
    ax2.axhline(60, color=BUILDUP_COLOR, lw=1.0, ls="--", label="警戒 60")
    ax2.axhline(75, color=CRISIS_COLOR, lw=1.0, ls=":", label="高危 75")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("危机评分")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.set_title("近 3 年危机评分走势", fontsize=10)

    plt.tight_layout()
    out = CHARTS / "05_current_snapshot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 文字报告 ──────────────────────────────────────────────────────────────────
def print_report(df: pd.DataFrame, ind: pd.DataFrame,
                 score: pd.Series, components: pd.DataFrame,
                 pc_ratio: Optional[float]):

    latest = df.index[-1]
    cur_score = score.iloc[-1]
    cur_vix   = df["VIX"].iloc[-1] if "VIX" in df.columns else None
    sox_200   = ind["sox_above_200ma"].iloc[-1] * 100
    spx_200   = ind["spx_above_200ma"].iloc[-1] * 100
    sox_rel6  = ind["sox_rel_ret_6m"].iloc[-1] * 100
    vol_reg   = ind["sox_vol_regime"].iloc[-1]

    hist = historical_stats(score)

    bar_width = 56
    fill = "█"

    def score_bar(val):
        n = int(round(val / 100 * 30))
        color = "🔴" if val >= 75 else "🟡" if val >= 55 else "🟢"
        return f"{fill * n}{'░' * (30 - n)}  {val:5.1f}/100  {color}"

    sep = "═" * bar_width

    print()
    print(sep)
    print("  费城半导体 × 标普500 — 经济危机预警分析报告")
    print(f"  数据截至：{latest.date()}    模型版本：v1.0")
    print(sep)

    print()
    print("【一】当前市场快照")
    print(f"  SOX 最新价    : {df['SOX'].iloc[-1]:>10,.1f}")
    print(f"  SPX 最新价    : {df['SPX'].iloc[-1]:>10,.1f}")
    if cur_vix:
        print(f"  VIX           : {cur_vix:>10.2f}  {'⚠ 极低(过度乐观)' if cur_vix < 15 else '正常范围' if cur_vix < 25 else '偏高(恐慌)'}")
    if pc_ratio:
        mood = "极度乐观(高危)" if pc_ratio < 0.7 else "乐观" if pc_ratio < 0.9 else "中性" if pc_ratio < 1.1 else "悲观"
        print(f"  SPY Put/Call  : {pc_ratio:>10.3f}  {mood}")
    print(f"  SOX偏离200MA  : {sox_200:>+9.1f}%")
    print(f"  SPX偏离200MA  : {spx_200:>+9.1f}%")
    print(f"  SOX 6M超额收益: {sox_rel6:>+9.1f}%  (vs SPX)")
    print(f"  波动率体制    : {vol_reg:>10.2f}x  {'⚠ 波动扩张' if vol_reg > 1.5 else '正常'}")

    print()
    print("【二】复合危机预警评分")
    print(f"  {score_bar(cur_score)}")

    print()
    comp_labels = {
        "sox_bubble":     "SOX 泡沫程度  ",
        "vix_complacency":"VIX 低位指数  ",
        "sox_leadership": "SOX 领涨过度  ",
        "vol_expansion":  "波动率扩张    ",
        "ratio_zscore":   "SOX/SPX比率 Z ",
    }
    for col, label in comp_labels.items():
        v = components[col].iloc[-1] * 100
        bar = fill * int(v / 100 * 20) + "░" * (20 - int(v / 100 * 20))
        print(f"  {label}: {bar}  {v:4.0f}/100")

    print()
    print("【三】历史危机期间对比")
    print(f"  {'时期':<20} {'均值':>8} {'峰值':>8} {'峰值日期':>12}")
    print(f"  {'-'*52}")
    for label, stats in hist.items():
        print(f"  {label:<20} {stats['mean']:>8.1f} {stats['max']:>8.1f} {stats['peak_date']:>12}")

    print()
    print("【四】结论与风险研判")
    risk = "🔴 高危" if cur_score >= 75 else "🟡 警戒" if cur_score >= 55 else "🟢 低风险"
    print(f"  当前风险等级：{risk}  (评分 {cur_score:.1f}/100)")
    print()

    if cur_score >= 75:
        print("  ⚠  评分进入高危区间，多项信号与2000/2008年顶部特征高度吻合。")
        print("     建议密切关注：SOX从高点回撤幅度、VIX能否快速上升确认趋势逆转。")
    elif cur_score >= 55:
        print("  ⚠  市场存在一定过热迹象，但尚未达到历史危机顶部水平。")
        print("     关键观察点：SOX是否开始跑输SPX（领先指标翻转信号）。")
    else:
        print("  ✓  当前信号偏低，短期系统性危机风险较小。")
        print("     持续跟踪：VIX是否进入极低区间 + SOX是否大幅超越SPX。")

    print()
    print("【五】方法论说明")
    print("  - SOX 领先 SPX：半导体需求是经济前瞻性指标，2000/2008年均提前6-12月见顶")
    print("  - VIX 乐观代理：极低 VIX 反映市场过度自满，历史上危机前均出现低 VIX 窗口")
    print("  - 复合评分校准：基于2000-03和2007-10两个历史顶部点的实际信号强度归一化")
    print("  - 局限性：模型为定量工具，不替代基本面分析；2020年COVID属于外生冲击，")
    print("    模型对其提前预警能力有限")
    print()
    print(sep)


# ── 主入口 ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 56)
    print("  SOX × SPX 经济危机预警研究系统  启动中...")
    print("=" * 56)

    print()
    df = load()

    print()
    print("[指标计算中...]")
    ind = compute_indicators(df)
    score, components = compute_crisis_score(df, ind)
    score = score.dropna()
    components = components.loc[score.index]

    print()
    print("[获取实时情绪数据...]")
    pc_ratio = get_current_pc_ratio()
    if pc_ratio:
        print(f"  ✓ SPY Put/Call 比率: {pc_ratio:.3f}")
    else:
        print("  ✗ Put/Call 数据不可用（市场未开盘或网络问题）")

    print()
    print("[生成图表...]")
    chart_normalized(df)
    chart_volatility(df, ind)
    chart_sentiment(df, ind)
    chart_crisis_score(df, score, components)
    chart_current_snapshot(df, ind, score, components)

    print()
    print_report(df, ind, score, components, pc_ratio)

    print(f"图表已保存至: {CHARTS.resolve()}")


if __name__ == "__main__":
    main()
