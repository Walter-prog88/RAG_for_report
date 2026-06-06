"""
数据获取模块：下载并缓存 SOX / SPX / VIX / NDX 历史数据
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import date
from typing import Optional

from data_providers import yahoo

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

TICKERS = {
    "SOX": "^SOX",
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "NDX": "^NDX",
}
START = "1993-01-01"
STALE_DAYS = 2  # 缓存超过此天数则重新下载


def _legacy_close(ticker: str) -> pd.Series | None:
    """读取旧的 sox_research/cache CSV，用作限流或离线时的兜底。"""
    safe = ticker.replace("^", "").replace("/", "_")
    fp = CACHE / f"{safe}.csv"

    if not fp.exists():
        return None

    cached = pd.read_csv(fp, index_col=0, parse_dates=True)
    if "Close" not in cached.columns or cached.empty:
        return None
    return cached["Close"].squeeze()


def _legacy_is_fresh(ticker: str) -> bool:
    safe = ticker.replace("^", "").replace("/", "_")
    fp = CACHE / f"{safe}.csv"
    if not fp.exists():
        return False
    cached = pd.read_csv(fp, index_col=0, parse_dates=True)
    if cached.empty:
        return False
    last = pd.Timestamp(cached.index[-1]).date()
    return (date.today() - last).days <= STALE_DAYS


def _write_legacy_cache(ticker: str, frame: pd.DataFrame) -> None:
    safe = ticker.replace("^", "").replace("/", "_")
    fp = CACHE / f"{safe}.csv"
    frame.to_csv(fp)


def load(refresh: bool = False) -> pd.DataFrame:
    """
    返回对齐后的日频收盘价 DataFrame，列名：SOX, SPX, VIX, NDX。
    refresh=True 则忽略缓存强制重下。
    """
    if refresh:
        for f in CACHE.glob("*.csv"):
            f.unlink()

    series: dict[str, pd.Series] = {}
    print("[数据加载]")

    use_legacy = not refresh and all(_legacy_is_fresh(tk) for tk in TICKERS.values())
    histories: dict[str, pd.DataFrame] = {}
    if not use_legacy:
        try:
            histories = yahoo.batch_history(
                TICKERS.values(),
                start=START,
                interval="1d",
                auto_adjust=True,
                refresh=refresh,
            )
        except Exception as e:
            print(f"  ! Yahoo 批量下载失败，尝试旧缓存兜底: {e}")

    for name, tk in TICKERS.items():
        try:
            frame = histories.get(tk)
            if frame is not None and not frame.empty and "Close" in frame.columns:
                _write_legacy_cache(tk, frame)
                s = frame["Close"].squeeze()
            else:
                s = _legacy_close(tk)
                if s is None:
                    raise ValueError(f"无可用数据: {tk}")
            s.index = pd.to_datetime(s.index).tz_localize(None)
            series[name] = s
            print(f"  ✓ {name:4s} {s.index[0].date()} → {s.index[-1].date()}  ({len(s):,} bars)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    df = pd.DataFrame(series)
    df = df.dropna(subset=["SOX", "SPX"])
    df.index.name = "date"
    return df


def get_current_pc_ratio() -> Optional[float]:
    """
    用 SPY 近三个月期权计算当前 Put/Call 成交量比率（看涨情绪代理）。
    比率 < 0.7 = 极度乐观；> 1.2 = 恐慌。
    """
    try:
        expiries = yahoo.option_expirations("SPY")[:4]
        total_calls, total_puts = 0.0, 0.0
        for exp in expiries:
            chain = yahoo.option_chain("SPY", exp)
            total_calls += chain["calls"]["volume"].fillna(0).sum()
            total_puts += chain["puts"]["volume"].fillna(0).sum()
        return round(total_puts / total_calls, 3) if total_calls > 0 else None
    except Exception:
        return None
