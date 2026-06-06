"""
AkShare provider for A-share, HK, US, and index historical data.

AkShare endpoints are often source-site limited. The main speed gains here are:
local cache, small retry logic, and optional concurrent batch downloads.
"""

from __future__ import annotations

import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

import pandas as pd

from .cache import make_key, read_frame, write_frame


DAILY_TTL_SECONDS = 12 * 60 * 60
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")


def disable_env_proxies() -> None:
    """
    AkShare's Chinese upstream sources often fail through generic local proxies.

    By default this provider removes requests-compatible proxy environment
    variables for the current process. Set FIN_AKSHARE_USE_ENV_PROXY=1 to keep
    the user's proxy environment untouched.
    """
    if os.environ.get("FIN_AKSHARE_USE_ENV_PROXY") == "1":
        return
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    for key in NO_PROXY_ENV_KEYS:
        os.environ[key] = "*"


def _ak():
    disable_env_proxies()
    import akshare as ak

    return ak


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in frame.columns:
        if "日期" in str(col) or str(col).lower() in {"date", "datetime", "time"}:
            try:
                frame[col] = pd.to_datetime(frame[col])
            except Exception:
                pass
    return frame.replace([float("inf"), float("-inf")], pd.NA)


def _cached_call(
    namespace: str,
    key_parts: tuple,
    fetcher: Callable[[], pd.DataFrame],
    *,
    refresh: bool,
    ttl_seconds: int | None,
    retries: int = 2,
    retry_sleep: float = 0.8,
) -> pd.DataFrame:
    key = make_key(namespace, *key_parts)
    if not refresh:
        cached = read_frame(namespace, key, ttl_seconds)
        if cached is not None:
            return cached

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            frame = fetcher()
            if frame is None:
                return pd.DataFrame()
            frame = _clean_frame(frame)
            if not frame.empty:
                write_frame(namespace, key, frame)
            return frame
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(retry_sleep)
    raise RuntimeError(f"AkShare request failed: {last_error}") from last_error


def a_share_history(
    symbol: str,
    *,
    period: str = "daily",
    start_date: str = "20200101",
    end_date: str = "20261231",
    adjust: str = "qfq",
    refresh: bool = False,
    ttl_seconds: int | None = DAILY_TTL_SECONDS,
) -> pd.DataFrame:
    """A-share historical OHLCV via ak.stock_zh_a_hist."""
    ak = _ak()
    return _cached_call(
        "akshare",
        ("a_share_history", symbol, period, start_date, end_date, adjust),
        lambda: ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        ),
        refresh=refresh,
        ttl_seconds=ttl_seconds,
    )


def hk_history(
    symbol: str,
    *,
    period: str = "daily",
    start_date: str = "20200101",
    end_date: str = "20261231",
    adjust: str = "qfq",
    refresh: bool = False,
    ttl_seconds: int | None = DAILY_TTL_SECONDS,
) -> pd.DataFrame:
    """Hong Kong stock historical OHLCV via ak.stock_hk_hist."""
    ak = _ak()
    return _cached_call(
        "akshare",
        ("hk_history", symbol, period, start_date, end_date, adjust),
        lambda: ak.stock_hk_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        ),
        refresh=refresh,
        ttl_seconds=ttl_seconds,
    )


def us_history(
    symbol: str,
    *,
    period: str = "daily",
    start_date: str = "20200101",
    end_date: str = "20261231",
    adjust: str = "",
    refresh: bool = False,
    ttl_seconds: int | None = DAILY_TTL_SECONDS,
) -> pd.DataFrame:
    """US stock historical OHLCV via AkShare EastMoney endpoint."""
    ak = _ak()
    return _cached_call(
        "akshare",
        ("us_history", symbol, period, start_date, end_date, adjust),
        lambda: ak.stock_us_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        ),
        refresh=refresh,
        ttl_seconds=ttl_seconds,
    )


def china_index_history(
    symbol: str,
    *,
    start_date: str = "20200101",
    end_date: str = "20261231",
    refresh: bool = False,
    ttl_seconds: int | None = DAILY_TTL_SECONDS,
) -> pd.DataFrame:
    """Chinese index history via EastMoney. Example: symbol='000852'."""
    ak = _ak()
    return _cached_call(
        "akshare",
        ("china_index_history", symbol, start_date, end_date),
        lambda: ak.stock_zh_index_daily_em(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        ),
        refresh=refresh,
        ttl_seconds=ttl_seconds,
    )


def csi_constituents(symbol: str = "000852", *, refresh: bool = False) -> pd.DataFrame:
    """CSI index constituent list. Default 000852 = CSI 1000."""
    ak = _ak()
    return _cached_call(
        "akshare",
        ("csi_constituents", symbol),
        lambda: ak.index_stock_cons(symbol=symbol),
        refresh=refresh,
        ttl_seconds=24 * 60 * 60,
    )


def concept_constituents(concept_name: str, *, refresh: bool = False) -> pd.DataFrame:
    """EastMoney concept-board constituents, e.g. concept_name='商业航天'."""
    ak = _ak()
    return _cached_call(
        "akshare",
        ("concept_constituents", concept_name),
        lambda: ak.stock_board_concept_cons_em(symbol=concept_name),
        refresh=refresh,
        ttl_seconds=60 * 60,
    )


def batch_a_share_history(
    symbols: Iterable[str],
    *,
    start_date: str = "20200101",
    end_date: str = "20261231",
    adjust: str = "qfq",
    max_workers: int = 4,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch many A-share histories concurrently.

    Keep max_workers modest. AkShare sources can throttle aggressive scraping.
    """
    symbols_list = [str(s).strip() for s in symbols if str(s).strip()]
    result: dict[str, pd.DataFrame] = {}
    if not symbols_list:
        return result

    def fetch_one(sym: str) -> tuple[str, pd.DataFrame]:
        return sym, a_share_history(
            sym,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            refresh=refresh,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols_list}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                _, frame = fut.result()
                result[sym] = frame
            except Exception:
                result[sym] = pd.DataFrame()
    return result
