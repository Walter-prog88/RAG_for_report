"""
Download CSI 300 data for the past 6 years and save as Parquet.

Datasets:
  data/raw/hs300_weights.parquet   — monthly constituent snapshots
  data/raw/daily/<code>.parquet    — daily OHLCV per stock
  data/raw/fina/<code>.parquet     — financial indicators per stock
  data/raw/forecast/<code>.parquet — analyst forecasts per stock

Resumable: skips files that already exist.
Run:
  export TUSHARE_TOKEN='...'
  python download_data.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from tushare_quant.client import get_pro_api

# ── config ────────────────────────────────────────────────────────────────────
END_DATE   = datetime.today().strftime("%Y%m%d")
START_DATE = (datetime.today() - timedelta(days=6 * 365)).strftime("%Y%m%d")
DATA_DIR   = Path(__file__).parent / "data" / "raw"
INDEX_CODE = "399300.SZ"
SLEEP      = 0.5   # seconds between API calls — stay well under rate limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────
def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _per_stock(stocks: list[str], out_dir: Path, fetch_fn, desc: str) -> None:
    """Call fetch_fn(code) for each stock; skip if file already exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for code in tqdm(stocks, desc=desc, ncols=80):
        out = out_dir / f"{code}.parquet"
        if out.exists():
            continue
        try:
            df = fetch_fn(code)
            if df is not None and not df.empty:
                _save(df, out)
        except Exception as exc:
            log.warning("%s  %s: %s", desc, code, exc)
            errors.append(code)
        time.sleep(SLEEP)

    if errors:
        log.warning("%s failed for %d stocks: %s", desc, len(errors), errors[:10])


# ── step 1: constituent weights ───────────────────────────────────────────────
def step1_weights(pro) -> list[str]:
    out = DATA_DIR / "hs300_weights.parquet"
    if out.exists():
        log.info("Skip (exists): %s", out)
        return pd.read_parquet(out)["con_code"].unique().tolist()

    log.info("Fetching CSI 300 constituent weights by month …")
    frames: list[pd.DataFrame] = []

    # one snapshot per month-end across the 6-year window
    month_ends = pd.date_range(START_DATE, END_DATE, freq="ME")
    for d in tqdm(month_ends, desc="index_weight", ncols=80):
        date_str = d.strftime("%Y%m%d")
        try:
            df = pro.index_weight(index_code=INDEX_CODE, trade_date=date_str)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as exc:
            log.warning("index_weight %s: %s", date_str, exc)
        time.sleep(SLEEP)

    all_weights = pd.concat(frames, ignore_index=True).drop_duplicates()
    _save(all_weights, out)
    log.info("Saved %d rows -> %s", len(all_weights), out)

    stocks = all_weights["con_code"].unique().tolist()
    log.info("Unique stocks in CSI 300 over 6 years: %d", len(stocks))
    return stocks


# ── step 2: daily OHLCV ───────────────────────────────────────────────────────
def step2_daily(pro, stocks: list[str]) -> None:
    log.info("Downloading daily OHLCV for %d stocks …", len(stocks))

    def fetch(code: str) -> pd.DataFrame:
        df = pro.daily(ts_code=code, start_date=START_DATE, end_date=END_DATE)
        if df is not None and not df.empty:
            df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    _per_stock(stocks, DATA_DIR / "daily", fetch, "daily")


# ── step 3: financial indicators ──────────────────────────────────────────────
def step3_fina(pro, stocks: list[str]) -> None:
    log.info("Downloading financial indicators for %d stocks …", len(stocks))

    def fetch(code: str) -> pd.DataFrame:
        df = pro.fina_indicator(ts_code=code, start_date=START_DATE, end_date=END_DATE)
        if df is not None and not df.empty:
            df = df.sort_values("end_date").reset_index(drop=True)
        return df

    _per_stock(stocks, DATA_DIR / "fina", fetch, "fina_indicator")


# ── step 4: analyst forecasts ─────────────────────────────────────────────────
def step4_forecast(pro, stocks: list[str]) -> None:
    log.info("Downloading analyst forecasts for %d stocks …", len(stocks))

    def fetch(code: str) -> pd.DataFrame:
        return pro.report_rc(ts_code=code, start_date=START_DATE, end_date=END_DATE)

    _per_stock(stocks, DATA_DIR / "forecast", fetch, "report_rc")


# ── summary ───────────────────────────────────────────────────────────────────
def print_summary() -> None:
    for name, subdir in [("daily", "daily"), ("fina", "fina"), ("forecast", "forecast")]:
        d = DATA_DIR / subdir
        if d.exists():
            files = list(d.glob("*.parquet"))
            total_rows = sum(len(pd.read_parquet(f)) for f in files)
            log.info("%-12s  %3d files  %d rows", name, len(files), total_rows)
    weights = DATA_DIR / "hs300_weights.parquet"
    if weights.exists():
        log.info("%-12s  %d rows", "hs300_weights", len(pd.read_parquet(weights)))


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("Date range: %s → %s", START_DATE, END_DATE)
    pro = get_pro_api()

    stocks = step1_weights(pro)
    step2_daily(pro, stocks)
    step3_fina(pro, stocks)
    step4_forecast(pro, stocks)

    log.info("=" * 50)
    log.info("Download complete. Summary:")
    print_summary()


if __name__ == "__main__":
    main()