from __future__ import annotations

import argparse
from datetime import date

import baostock as bs
import pandas as pd


BAOSTOCK_FIELDS = "date,open,high,low,close,volume,turn,pctChg"
RENAME_COLUMNS = {
    "turn": "turnover_rate",
}


def parse_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def baostock_symbol(symbol: str) -> str:
    """Convert 300308 to sz.300308 and 600519 to sh.600519."""
    if "." in symbol:
        return symbol
    if symbol.startswith(("6", "9")):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


def baostock_date(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def fetch_a_share_daily_baostock(
    symbol: str = "300308",
    *,
    years: int = 3,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    adjustflag: str = "2",
) -> pd.DataFrame:
    """
    Fetch clean A-share daily OHLCV and turnover-rate data from BaoStock.

    adjustflag: "1" backward adjusted, "2" forward adjusted, "3" unadjusted.
    """
    end_day = parse_date(end_date) or date.today()
    start_day = parse_date(start_date)
    if start_day is None:
        start_day = (pd.Timestamp(end_day) - pd.DateOffset(years=years)).date()

    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")

    try:
        result = bs.query_history_k_data_plus(
            baostock_symbol(symbol),
            BAOSTOCK_FIELDS,
            start_date=baostock_date(start_day),
            end_date=baostock_date(end_day),
            frequency="d",
            adjustflag=adjustflag,
        )
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock query failed: {result.error_msg}")
        raw = result.get_data()
    finally:
        bs.logout()

    df = raw.rename(columns=RENAME_COLUMNS).copy()
    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = ["open", "close", "high", "low", "volume", "turnover_rate", "pctChg"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.loc[:, ["date", "open", "high", "low", "close", "volume", "turnover_rate", "pctChg"]]
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna(subset=["date", *numeric_columns])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch clean daily A-share OHLCV data with turnover rate via BaoStock."
    )
    parser.add_argument("--symbol", default="300308", help="A-share code, e.g. 300308 or sz.300308")
    parser.add_argument("--years", type=int, default=3, help="Number of past years to fetch")
    parser.add_argument(
        "--start-date",
        help="Start date, e.g. 20220101 or 2022-01-01. Overrides --years if provided.",
    )
    parser.add_argument("--end-date", help="End date, e.g. 20250101 or 2025-01-01")
    parser.add_argument(
        "--adjustflag",
        default="2",
        choices=["1", "2", "3"],
        help="BaoStock adjustment: 1=backward, 2=forward, 3=unadjusted",
    )
    parser.add_argument("--csv", help="Optional path to save the cleaned DataFrame as CSV")
    args = parser.parse_args()

    df = fetch_a_share_daily_baostock(
        symbol=args.symbol,
        years=args.years,
        start_date=args.start_date,
        end_date=args.end_date,
        adjustflag=args.adjustflag,
    )

    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")

    print(df)
    if df.empty:
        print("\nrows=0")
    else:
        print(f"\nrows={len(df)}, start={df['date'].min().date()}, end={df['date'].max().date()}")
        requested_end = parse_date(args.end_date)
        actual_end = df["date"].max().date()
        if requested_end is not None and actual_end < requested_end:
            print(
                "warning="
                f"BaoStock only returned data through {actual_end}; requested end_date={requested_end}. "
                "Daily bars are usually updated after market close or later, so same-day data may be unavailable."
            )


if __name__ == "__main__":
    main()
