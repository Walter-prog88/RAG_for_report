from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from .baostock_daily import fetch_a_share_daily_baostock
except ImportError:
    from baostock_daily import fetch_a_share_daily_baostock


REQUIRED_INPUT_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover_rate",
    "pctChg",
]

FACTOR_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "mom_20d",
    "vol_ratio_5",
    "turn_ma_20",
    "price_ma_ratio",
    "y_future_5d",
]


def build_factor_dataframe(df: pd.DataFrame, *, drop_missing: bool = True) -> pd.DataFrame:
    """
    Add factor columns to a daily stock DataFrame.

    X factors use only date t and earlier data. y_future_5d intentionally uses t+5.
    """
    missing_columns = sorted(set(REQUIRED_INPUT_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Input DataFrame missing columns: {missing_columns}")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    numeric_columns = [column for column in REQUIRED_INPUT_COLUMNS if column != "date"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    close = result["close"]
    volume = result["volume"]
    turnover_rate = result["turnover_rate"]

    result["ret_1d"] = np.log(close / close.shift(1))
    result["ret_5d"] = np.log(close / close.shift(5))
    result["ret_20d"] = np.log(close / close.shift(20))
    result["vol_20d"] = result["ret_1d"].rolling(window=20, min_periods=20).std()
    result["mom_20d"] = close / close.shift(20) - 1
    result["vol_ratio_5"] = volume / volume.rolling(window=5, min_periods=5).mean()
    result["turn_ma_20"] = turnover_rate.rolling(window=20, min_periods=20).mean()
    result["price_ma_ratio"] = close / close.rolling(window=20, min_periods=20).mean()
    result["y_future_5d"] = close.shift(-5) / close - 1

    result = result.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    if drop_missing:
        result = result.dropna(subset=FACTOR_COLUMNS).reset_index(drop=True)

    return result


def print_validation(before_shape: tuple[int, int], df: pd.DataFrame) -> None:
    print(f"shape_before={before_shape}")
    print(f"shape_after={df.shape}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\nhead(5):")
        print(df.head(5))
        print("\ntail(5):")
        print(df.tail(5))
        print("\ndescribe():")
        print(df.describe())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build factor columns from BaoStock daily data.")
    parser.add_argument("--symbol", default="300308", help="A-share code, e.g. 300308 or sz.300308")
    parser.add_argument("--start-date", default="20220101", help="Start date, e.g. 20220101")
    parser.add_argument("--end-date", default="20250101", help="End date, e.g. 20250101")
    parser.add_argument(
        "--adjustflag",
        default="2",
        choices=["1", "2", "3"],
        help="BaoStock adjustment: 1=backward, 2=forward, 3=unadjusted",
    )
    parser.add_argument("--csv", help="Optional path to save the factor DataFrame as CSV")
    args = parser.parse_args()

    raw_df = fetch_a_share_daily_baostock(
        args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        adjustflag=args.adjustflag,
    )
    factor_df = build_factor_dataframe(raw_df)

    if args.csv:
        factor_df.to_csv(args.csv, index=False, encoding="utf-8-sig")

    print_validation(raw_df.shape, factor_df)


if __name__ == "__main__":
    main()
