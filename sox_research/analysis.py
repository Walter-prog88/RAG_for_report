"""
指标计算与危机评分模块

核心逻辑：
  1. 费城半导体指数 (SOX) 是科技/经济周期的领先指标
  2. 2000 年：SOX 在 2000-03 触顶，SPX 稍滞后；SOX 领跌
  3. 2008 年：SOX 在 2007-10 触顶，早于 SPX 约 2 个月
  4. 复合危机评分 = 乐观情绪过热 × 半导体泡沫 × 波动率扩张
"""

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# ── 历史危机区间（用于图表标注和回溯校准）──────────────────────────────────
CRISIS_PERIODS = {
    "科网泡沫":      ("2000-03-01", "2002-10-09"),
    "次贷危机":      ("2007-10-09", "2009-03-09"),
    "COVID 闪崩":   ("2020-02-19", "2020-03-23"),
}

# 危机「酝酿期」（用于检验信号提前量）
BUILDUP_PERIODS = {
    "科网酝酿期":    ("1999-01-01", "2000-03-01"),
    "次贷酝酿期":    ("2006-01-01", "2007-10-09"),
}

WINDOW_SHORT  = 21   # ~1 个月
WINDOW_MED    = 63   # ~3 个月
WINDOW_LONG   = 252  # ~1 年
WINDOW_200MA  = 200


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    """年化滚动波动率（对数收益率标准差 × √252）"""
    return returns.rolling(window).std() * np.sqrt(252)


def _drawdown(prices: pd.Series, window: int = WINDOW_LONG) -> pd.Series:
    """相对滚动最高价的回撤（负值）"""
    peak = prices.rolling(window, min_periods=1).max()
    return (prices / peak) - 1


def _pct_above_ma(prices: pd.Series, window: int) -> pd.Series:
    """价格相对均线的偏离度"""
    return (prices / prices.rolling(window).mean()) - 1


def _rolling_zscore(series: pd.Series, window: int = WINDOW_LONG * 2) -> pd.Series:
    """滚动 Z-score，用于判断当前值偏离历史的程度"""
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std()
    return (series - m) / s.replace(0, np.nan)


def _vix_complacency(vix: pd.Series, window: int = WINDOW_LONG * 2) -> pd.Series:
    """
    VIX 低位 = 市场过度乐观（看涨情绪代理）
    返回值：VIX 在过去 window 天中的「低分位」排名（VIX 越低，评分越高 → 危险越大）
    取值范围 [0, 1]，越接近 1 越危险（过度乐观）
    """
    def rank_inv(x):
        arr = x.dropna().values
        if len(arr) < 10:
            return np.nan
        pct = percentileofscore(arr, arr[-1], kind="mean") / 100
        return 1 - pct  # 低 VIX = 高评分

    return vix.rolling(window, min_periods=window // 2).apply(rank_inv, raw=False)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有衍生指标，输入 df 需包含列：SOX, SPX, VIX, NDX（可选）

    返回指标 DataFrame（与输入同索引）：
      sox_ret_1m / sox_ret_3m / sox_ret_6m     : SOX 各期收益率
      spx_ret_1m / spx_ret_3m / spx_ret_6m     : SPX 各期收益率
      sox_vol_1m / sox_vol_3m                   : SOX 年化实现波动率
      spx_vol_1m                                : SPX 年化实现波动率
      sox_above_200ma                           : SOX 偏离 200 日均线
      spx_above_200ma                           : SPX 偏离 200 日均线
      sox_spx_ratio                             : SOX/SPX 归一化比率（起始=100）
      sox_spx_ratio_zscore                      : 比率的滚动 Z-score
      sox_rel_ret_6m                            : SOX 相对 SPX 6 个月超额收益
      sox_drawdown_1y                           : SOX 相对 1 年高点回撤
      vix_complacency                           : VIX 低位指数（0-1，越高越危险）
      sox_vol_regime                            : SOX 短期 vs 长期波动率比（>1 = 波动扩张）
    """
    ind = pd.DataFrame(index=df.index)
    log_sox = np.log(df["SOX"]).diff()
    log_spx = np.log(df["SPX"]).diff()

    for period, label in [(WINDOW_SHORT, "1m"), (WINDOW_MED, "3m"), (252 // 2, "6m")]:
        ind[f"sox_ret_{label}"] = df["SOX"].pct_change(period)
        ind[f"spx_ret_{label}"] = df["SPX"].pct_change(period)

    ind["sox_vol_1m"] = _rolling_vol(log_sox, WINDOW_SHORT)
    ind["sox_vol_3m"] = _rolling_vol(log_sox, WINDOW_MED)
    ind["spx_vol_1m"] = _rolling_vol(log_spx, WINDOW_SHORT)

    ind["sox_above_200ma"] = _pct_above_ma(df["SOX"], WINDOW_200MA)
    ind["spx_above_200ma"] = _pct_above_ma(df["SPX"], WINDOW_200MA)

    # SOX/SPX 相对强度
    ratio = df["SOX"] / df["SPX"]
    ind["sox_spx_ratio"] = ratio / ratio.iloc[0] * 100
    ind["sox_spx_ratio_zscore"] = _rolling_zscore(ratio, WINDOW_LONG * 2)

    ind["sox_rel_ret_6m"] = ind["sox_ret_6m"] - ind["spx_ret_6m"]
    ind["sox_drawdown_1y"] = _drawdown(df["SOX"], WINDOW_LONG)

    ind["vix_complacency"] = _vix_complacency(df["VIX"], WINDOW_LONG * 2)
    ind["sox_vol_regime"] = ind["sox_vol_1m"] / ind["sox_vol_3m"]

    return ind


def compute_crisis_score(df: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
    """
    复合危机预警评分（0-100，越高越危险）

    五个子信号，各映射到 [0, 1]：
      A. SOX 泡沫程度   — SOX 偏离 200 日均线（正向 z-score）        权重 25%
      B. 乐观情绪过热   — VIX 低位指数                               权重 25%
      C. 半导体领涨过度  — SOX 相对 SPX 6 个月超额收益（正向极端）   权重 20%
      D. 波动率扩张信号  — 短期 vs 长期波动率比（>1 开始警示）        权重 15%
      E. SOX 均线偏离 z — 滚动 2 年 z-score                          权重 15%
    """

    def clip01(s: pd.Series) -> pd.Series:
        return s.clip(0, 1)

    def sigmoid_scale(s: pd.Series, center: float, scale: float) -> pd.Series:
        return 1 / (1 + np.exp(-(s - center) / scale))

    # A: SOX 偏离 200MA（>0.3 时接近满分）
    A = clip01(sigmoid_scale(ind["sox_above_200ma"], center=0.15, scale=0.10))

    # B: VIX 低位乐观（已在 0-1 之间）
    B = clip01(ind["vix_complacency"])

    # C: SOX 相对 SPX 6M 超额收益（>0.2 时高危）
    C = clip01(sigmoid_scale(ind["sox_rel_ret_6m"], center=0.10, scale=0.08))

    # D: 波动率体制切换（短期/长期 vol > 1.5 = 波动扩张，危机开始阶段）
    D = clip01(sigmoid_scale(ind["sox_vol_regime"], center=1.3, scale=0.3))

    # E: SOX/SPX 比率 z-score（> 1.5 标准差时高危）
    E = clip01(sigmoid_scale(ind["sox_spx_ratio_zscore"], center=1.2, scale=0.5))

    weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    components = pd.concat([A, B, C, D, E], axis=1)
    components.columns = ["sox_bubble", "vix_complacency", "sox_leadership", "vol_expansion", "ratio_zscore"]

    score = (components * weights).sum(axis=1) * 100
    return score.rename("crisis_score"), components


def get_crisis_score_at(score: pd.Series, dt: str) -> float:
    """获取指定日期的危机评分"""
    try:
        return float(score.loc[dt])
    except KeyError:
        idx = score.index.get_indexer([pd.Timestamp(dt)], method="nearest")[0]
        return float(score.iloc[idx])


def historical_stats(score: pd.Series) -> dict:
    """返回危机期间和酝酿期间的平均评分，用于横向比较"""
    result = {}
    for label, (start, end) in {**CRISIS_PERIODS, **BUILDUP_PERIODS}.items():
        subset = score.loc[start:end]
        if not subset.empty:
            result[label] = {
                "mean": round(subset.mean(), 1),
                "max":  round(subset.max(),  1),
                "peak_date": str(subset.idxmax().date()),
            }
    return result
