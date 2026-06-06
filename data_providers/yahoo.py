"""
Yahoo Finance provider.

Use this for US/global equities, ETFs, indices, FX, crypto, basic company
metadata, news, and option-chain summaries.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable

import pandas as pd
import yfinance as yf

from .cache import make_key, read_frame, read_json, write_frame, write_json


QUOTE_TTL_SECONDS = 60
HISTORY_TTL_SECONDS = 6 * 60 * 60
INFO_TTL_SECONDS = 24 * 60 * 60
NEWS_TTL_SECONDS = 30 * 60
OPTIONS_TTL_SECONDS = 15 * 60
QUOTE_COLUMNS = [
    "price",
    "change",
    "change_percent",
    "open",
    "high",
    "low",
    "previous_close",
    "volume",
    "timestamp",
]


def set_proxy(proxy_url: str | None) -> None:
    """Set or clear the yfinance proxy."""
    yf.set_config(proxy=proxy_url)


def _normalize_symbols(symbols: str | Iterable[str]) -> list[str]:
    if isinstance(symbols, str):
        return [symbols.upper()]
    return [str(s).strip().upper() for s in symbols if str(s).strip()]


def _clean_download_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame[~frame.index.duplicated()]


def history(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "1y",
    interval: str = "1d",
    auto_adjust: bool = True,
    refresh: bool = False,
    ttl_seconds: int | None = HISTORY_TTL_SECONDS,
) -> pd.DataFrame:
    """Fetch historical OHLCV for one symbol with CSV cache."""
    symbol = symbol.upper()
    key = make_key("history", symbol, start, end, period, interval, auto_adjust)
    if not refresh:
        cached = read_frame("yahoo", key, ttl_seconds)
        if cached is not None:
            return cached

    kwargs: dict[str, Any] = {
        "tickers": symbol,
        "interval": interval,
        "auto_adjust": auto_adjust,
        "progress": False,
        "threads": False,
    }
    if start or end:
        kwargs.update({"start": start, "end": end})
    else:
        kwargs["period"] = period or "1y"

    data = yf.download(**kwargs)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)
    data = _clean_download_frame(data)
    if not data.empty:
        write_frame("yahoo", key, data)
    return data


def batch_history(
    symbols: Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = True,
    refresh: bool = False,
    ttl_seconds: int | None = HISTORY_TTL_SECONDS,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical data for many symbols in one yfinance request.

    This is usually much faster than looping over Ticker(...).history().
    """
    symbols_list = _normalize_symbols(symbols)
    if not symbols_list:
        return {}

    key = make_key("batch_history", symbols_list, start, end, period, interval, auto_adjust)
    if not refresh:
        cached = read_json("yahoo", key, ttl_seconds)
        if isinstance(cached, dict) and cached.get("symbols") == symbols_list:
            out = {}
            for sym in symbols_list:
                frame_key = cached.get("frames", {}).get(sym)
                if frame_key:
                    frame = read_frame("yahoo", frame_key, ttl_seconds)
                    if frame is not None:
                        out[sym] = frame
            if out:
                return out

    kwargs: dict[str, Any] = {
        "tickers": symbols_list,
        "interval": interval,
        "group_by": "ticker",
        "auto_adjust": auto_adjust,
        "progress": False,
        "threads": True,
    }
    if start or end:
        kwargs.update({"start": start, "end": end})
    else:
        kwargs["period"] = period

    raw = yf.download(**kwargs)

    result: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return result

    for sym in symbols_list:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if sym in raw.columns.get_level_values(0):
                    frame = raw[sym]
                else:
                    frame = raw.xs(sym, axis=1, level=1)
            else:
                frame = raw
            frame = _clean_download_frame(frame.dropna(how="all"))
            if not frame.empty:
                result[sym] = frame
        except Exception:
            continue

    frame_keys = {}
    for sym, frame in result.items():
        frame_key = make_key("batch_history_frame", sym, start, end, period, interval, auto_adjust)
        write_frame("yahoo", frame_key, frame)
        frame_keys[sym] = frame_key
    write_json("yahoo", key, {"symbols": symbols_list, "frames": frame_keys})
    return result


def batch_quotes(
    symbols: Iterable[str],
    *,
    refresh: bool = False,
    ttl_seconds: int | None = QUOTE_TTL_SECONDS,
) -> pd.DataFrame:
    """Fast quote snapshot for many symbols based on recent daily bars."""
    symbols_list = _normalize_symbols(symbols)
    key = make_key("batch_quotes", symbols_list)
    if not refresh:
        cached = read_frame("yahoo", key, ttl_seconds)
        if cached is not None:
            return cached

    histories = batch_history(symbols_list, period="5d", interval="1d", auto_adjust=False, refresh=True)
    rows = []
    now = int(datetime.now().timestamp())
    for sym, frame in histories.items():
        if frame.empty or "Close" not in frame:
            continue
        last = frame.dropna(subset=["Close"]).iloc[-1]
        prev_close = frame["Close"].dropna().iloc[-2] if len(frame["Close"].dropna()) >= 2 else last["Close"]
        price = float(last["Close"])
        change = price - float(prev_close)
        change_percent = change / float(prev_close) * 100 if prev_close else 0.0
        rows.append({
            "symbol": sym,
            "price": price,
            "change": change,
            "change_percent": change_percent,
            "open": float(last.get("Open")) if pd.notna(last.get("Open")) else None,
            "high": float(last.get("High")) if pd.notna(last.get("High")) else None,
            "low": float(last.get("Low")) if pd.notna(last.get("Low")) else None,
            "previous_close": float(prev_close),
            "volume": int(last.get("Volume")) if pd.notna(last.get("Volume")) else None,
            "timestamp": now,
        })

    quotes = pd.DataFrame(rows)
    if not quotes.empty:
        write_frame("yahoo", key, quotes.set_index("symbol"))
        quotes = quotes.set_index("symbol")
    else:
        quotes = pd.DataFrame(columns=QUOTE_COLUMNS)
        quotes.index.name = "symbol"
    return quotes


def company_info(symbol: str, *, refresh: bool = False) -> dict[str, Any]:
    """Fetch selected company metadata. This can be slower than price history."""
    symbol = symbol.upper()
    key = make_key("company_info", symbol)
    if not refresh:
        cached = read_json("yahoo", key, INFO_TTL_SECONDS)
        if isinstance(cached, dict):
            return cached

    info = yf.Ticker(symbol).info or {}
    data = {
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "exchange": info.get("exchange"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "beta": info.get("beta"),
        "dividend_yield": info.get("dividendYield"),
        "website": info.get("website"),
        "description": info.get("longBusinessSummary"),
    }
    write_json("yahoo", key, data)
    return data


def news(symbol: str, *, count: int = 20, refresh: bool = False) -> list[dict[str, Any]]:
    """Fetch recent Yahoo Finance news for a symbol."""
    symbol = symbol.upper()
    key = make_key("news", symbol, count)
    if not refresh:
        cached = read_json("yahoo", key, NEWS_TTL_SECONDS)
        if isinstance(cached, list):
            return cached

    items = yf.Ticker(symbol).news or []
    articles = []
    for item in items[:count]:
        content = item.get("content", item)
        provider = content.get("provider") or {}
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        articles.append({
            "symbol": symbol,
            "title": content.get("title") or item.get("title"),
            "publisher": provider.get("displayName") or item.get("publisher"),
            "published": content.get("pubDate") or item.get("providerPublishTime"),
            "summary": content.get("summary"),
            "url": url_obj.get("url") if isinstance(url_obj, dict) else item.get("link"),
        })
    write_json("yahoo", key, articles)
    return articles


def option_expirations(symbol: str) -> list[str]:
    """Return available option expiration dates."""
    return list(yf.Ticker(symbol.upper()).options or [])


def option_chain(
    symbol: str,
    expiration: str,
    *,
    refresh: bool = False,
    ttl_seconds: int | None = OPTIONS_TTL_SECONDS,
) -> dict[str, pd.DataFrame]:
    """Fetch calls and puts for one expiration."""
    symbol = symbol.upper()
    key_calls = make_key("option_chain_calls", symbol, expiration)
    key_puts = make_key("option_chain_puts", symbol, expiration)
    if not refresh:
        calls = read_frame("yahoo", key_calls, ttl_seconds)
        puts = read_frame("yahoo", key_puts, ttl_seconds)
        if calls is not None and puts is not None:
            return {"calls": calls, "puts": puts}

    chain = yf.Ticker(symbol).option_chain(expiration)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    if not calls.empty:
        write_frame("yahoo", key_calls, calls)
    if not puts.empty:
        write_frame("yahoo", key_puts, puts)
    return {"calls": calls, "puts": puts}


def option_volume_summary(
    symbol: str,
    *,
    max_expirations: int | None = 6,
    min_days_to_expiry_for_leap: int = 90,
    refresh: bool = False,
) -> dict[str, Any]:
    """Aggregate call/put volume and open interest across option expirations."""
    symbol = symbol.upper()
    expirations = option_expirations(symbol)
    if max_expirations is not None:
        expirations = expirations[:max_expirations]

    today = datetime.now().date()
    leap_threshold = today + timedelta(days=min_days_to_expiry_for_leap)
    total_call_volume = total_put_volume = 0.0
    total_call_oi = total_put_oi = 0.0
    leap_call_volume = leap_put_volume = 0.0

    hottest: dict[str, Any] = {"contract": None, "volume": -1, "expiration": None, "type": None}

    for exp in expirations:
        chain = option_chain(symbol, exp, refresh=refresh)
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        for opt_type, frame in chain.items():
            if frame.empty:
                continue
            volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0)
            open_interest = pd.to_numeric(frame.get("openInterest"), errors="coerce").fillna(0)
            vol_sum = float(volume.sum())
            oi_sum = float(open_interest.sum())

            if opt_type == "calls":
                total_call_volume += vol_sum
                total_call_oi += oi_sum
                if exp_date > leap_threshold:
                    leap_call_volume += vol_sum
            else:
                total_put_volume += vol_sum
                total_put_oi += oi_sum
                if exp_date > leap_threshold:
                    leap_put_volume += vol_sum

            if not volume.empty:
                idx = volume.idxmax()
                max_vol = float(volume.loc[idx])
                if max_vol > hottest["volume"]:
                    hottest = {
                        "contract": frame.loc[idx].get("contractSymbol"),
                        "volume": max_vol,
                        "expiration": exp,
                        "type": "call" if opt_type == "calls" else "put",
                    }

    put_call_volume = total_put_volume / total_call_volume if total_call_volume else None
    call_put_volume = total_call_volume / total_put_volume if total_put_volume else None
    leap_call_put = leap_call_volume / leap_put_volume if leap_put_volume else None

    return {
        "symbol": symbol,
        "expirations_scanned": expirations,
        "call_volume": total_call_volume,
        "put_volume": total_put_volume,
        "total_volume": total_call_volume + total_put_volume,
        "call_open_interest": total_call_oi,
        "put_open_interest": total_put_oi,
        "put_call_volume_ratio": put_call_volume,
        "call_put_volume_ratio": call_put_volume,
        "leap_call_volume": leap_call_volume,
        "leap_put_volume": leap_put_volume,
        "leap_call_put_ratio": leap_call_put,
        "most_active": hottest,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
