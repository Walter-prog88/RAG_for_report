#!/usr/bin/env python3
"""
Historical SK hynix cycle gain analysis.

Data source: Yahoo Finance chart API for 000660.KS.
The script uses unadjusted close/high/low because Yahoo's early adjusted-close
series for this ticker contains invalid negative values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DAILY_OUTPUT = ROOT / "sk_hynix_daily_history.csv"
CYCLE_OUTPUT = ROOT / "cycle_gain_summary.csv"
SUMMARY_OUTPUT = ROOT / "cycle_gain_summary.json"

YAHOO_CHART_URL = (
    "https://query2.finance.yahoo.com/v8/finance/chart/000660.KS"
    "?period1=946944000&period2=1778563200&interval=1d"
    "&events=history&includeAdjustedClose=true"
)

MANUAL_CYCLES = [
    ("dotcom_dram_recovery", "Dot-com/DRAM recovery", "2003-03-26", "2006-09-18"),
    ("gfc_recovery", "GFC recovery", "2008-11-24", "2011-04-21"),
    ("mobile_cloud_upcycle", "Mobile/cloud DRAM upcycle", "2011-08-19", "2014-07-08"),
    ("server_dram_supercycle", "2016-18 server DRAM supercycle", "2016-05-09", "2018-05-23"),
    ("covid_ai_early_recovery", "COVID/AI early recovery", "2019-01-03", "2021-02-25"),
    ("current_from_2022_trough", "Current AI/HBM cycle from 2022 trough", "2022-12-29", "2026-05-11"),
    ("current_from_2024_selloff", "Current HBM leg from 2024 selloff", "2024-08-05", "2026-05-11"),
    ("current_from_2025_low", "Current from 2025 April low", "2025-04-07", "2026-05-11"),
]


def fetch_daily_history() -> tuple[pd.DataFrame, dict]:
    response = requests.get(YAHOO_CHART_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(quote)
    timestamps = pd.Series(pd.to_datetime(result["timestamp"], unit="s"))
    frame["timestamp"] = timestamps.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["close"]).reset_index(drop=True)
    return frame, result["meta"]


def cycle_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    by_date = frame.set_index("date")
    for cycle_id, label, low_date, high_date in MANUAL_CYCLES:
        low_row = by_date.loc[low_date]
        high_row = by_date.loc[high_date]
        close_multiple = float(high_row["close"]) / float(low_row["close"])
        intraday_multiple = float(high_row["high"]) / float(low_row["low"])
        rows.append({
            "cycle_id": cycle_id,
            "label": label,
            "low_date": low_date,
            "low_close": float(low_row["close"]),
            "low_intraday": float(low_row["low"]),
            "high_date": high_date,
            "high_close": float(high_row["close"]),
            "high_intraday": float(high_row["high"]),
            "close_multiple": close_multiple,
            "close_gain": close_multiple - 1,
            "intraday_low_to_high_multiple": intraday_multiple,
            "intraday_low_to_high_gain": intraday_multiple - 1,
        })
    return rows


def max_drawup(frame: pd.DataFrame) -> dict:
    min_close = float("inf")
    min_idx = 0
    best = {}
    for idx, row in frame.iterrows():
        close = float(row["close"])
        if close < min_close:
            min_close = close
            min_idx = idx
        multiple = close / min_close
        if not best or multiple > best["close_multiple"]:
            best = {
                "low_date": frame.loc[min_idx, "date"],
                "low_close": min_close,
                "high_date": row["date"],
                "high_close": close,
                "close_multiple": multiple,
                "close_gain": multiple - 1,
            }
    return best


def main() -> None:
    frame, meta = fetch_daily_history()
    frame.to_csv(DAILY_OUTPUT, index=False)

    rows = cycle_rows(frame)
    cycle_frame = pd.DataFrame(rows)
    cycle_frame.to_csv(CYCLE_OUTPUT, index=False)

    current = cycle_frame[cycle_frame["cycle_id"].eq("current_from_2022_trough")].iloc[0].to_dict()
    current_hbm_leg = cycle_frame[cycle_frame["cycle_id"].eq("current_from_2024_selloff")].iloc[0].to_dict()
    prior = cycle_frame[~cycle_frame["cycle_id"].str.startswith("current_")]
    prior_max = prior.sort_values("close_multiple", ascending=False).iloc[0].to_dict()
    all_time = max_drawup(frame)

    summary = {
        "source_symbol": meta.get("symbol"),
        "currency": meta.get("currency"),
        "regular_market_price": meta.get("regularMarketPrice"),
        "fifty_two_week_high": meta.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": meta.get("fiftyTwoWeekLow"),
        "history_start": str(frame["date"].iloc[0]),
        "history_end": str(frame["date"].iloc[-1]),
        "history_rows": len(frame),
        "max_drawup_all_history": all_time,
        "prior_cycle_max": prior_max,
        "current_from_2022_trough": current,
        "current_from_2024_selloff": current_hbm_leg,
        "current_exceeds_prior_cycle_max": current["close_multiple"] > prior_max["close_multiple"],
        "current_hbm_leg_exceeds_prior_cycle_max": current_hbm_leg["close_multiple"] > prior_max["close_multiple"],
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("SK hynix historical cycle gain analysis")
    print("=" * 72)
    print(f"History range             : {summary['history_start']} -> {summary['history_end']} ({summary['history_rows']} rows)")
    print(f"Current close             : KRW {summary['regular_market_price']:,.0f}")
    print(f"All-history max drawup    : {all_time['close_multiple']:.2f}x ({all_time['close_gain']:.2%})")
    print(
        "Prior memory-cycle max    : "
        f"{prior_max['close_multiple']:.2f}x ({prior_max['close_gain']:.2%}) "
        f"[{prior_max['label']}]"
    )
    print(
        "Current from 2022 trough  : "
        f"{current['close_multiple']:.2f}x ({current['close_gain']:.2%})"
    )
    print(
        "Current from 2024 selloff : "
        f"{current_hbm_leg['close_multiple']:.2f}x ({current_hbm_leg['close_gain']:.2%})"
    )
    print(f"Cycle table saved         : {CYCLE_OUTPUT}")
    print(f"Summary saved             : {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
