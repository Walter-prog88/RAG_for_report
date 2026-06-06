from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import tushare as ts
from tqdm import tqdm

try:
    from .client import get_pro_api
except ImportError:
    from client import get_pro_api


DEFAULT_START_DATE = "20200101"
DEFAULT_END_DATE = "20260515"
DEFAULT_INDEX_CODE = "399300.SZ"
DEFAULT_DATA_DIR = Path("tushare_quant") / "quant_data"

DATASET_ORDER = [
    "index_weight",
    "stock_basic",
    "index_daily",
    "moneyflow_hsgt",
    "daily",
    "daily_basic",
    "fina_indicator",
    "moneyflow",
    "report_rc",
]


def ymd(value: str) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def ymd_dash(value: str) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


class TushareDownloader:
    def __init__(
        self,
        *,
        start_date: str,
        end_date: str,
        index_code: str,
        data_dir: Path,
        retries: int,
        retry_sleep: float,
        request_sleep: float,
        skip_existing: bool,
        max_stocks: int | None,
    ) -> None:
        self.start_date = ymd(start_date)
        self.end_date = ymd(end_date)
        self.index_code = index_code
        self.data_dir = data_dir
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.request_sleep = request_sleep
        self.skip_existing = skip_existing
        self.max_stocks = max_stocks
        self.pro = get_pro_api()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.failures: list[dict] = []

    def path(self, filename: str) -> Path:
        return self.data_dir / filename

    def should_skip(self, filename: str) -> bool:
        path = self.path(filename)
        return self.skip_existing and path.exists() and path.stat().st_size > 0

    def save_parquet(self, df: pd.DataFrame | None, filename: str) -> None:
        if df is None or df.empty:
            print(f"WARNING: {filename} data empty; skip saving")
            return
        path = self.path(filename)
        df.to_parquet(path, index=False)
        print(f"OK: saved {path} shape={df.shape}")

    def read_existing(self, filename: str) -> pd.DataFrame | None:
        path = self.path(filename)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def safe_call(self, func: Callable, dataset: str, **kwargs) -> pd.DataFrame | None:
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                return func(**kwargs)
            except Exception as exc:
                last_error = exc
                print(f"call failed dataset={dataset} attempt={attempt}/{self.retries}: {exc}")
                time.sleep(self.retry_sleep)
        self.failures.append({"dataset": dataset, "kwargs": kwargs, "error": str(last_error)})
        return None

    def save_failures(self) -> None:
        if not self.failures:
            return
        path = self.path("download_failures.json")
        path.write_text(json.dumps(self.failures, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"WARNING: failures saved to {path}")

    def save_metadata(self, stock_codes: list[str]) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "index_code": self.index_code,
            "data_dir": str(self.data_dir),
            "stock_count": len(stock_codes),
            "stock_codes": stock_codes,
            "datasets": DATASET_ORDER,
        }
        path = self.path("metadata.json")
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OK: metadata saved {path}")

    def download_index_weight(self) -> list[str]:
        filename = "hs300_weight.parquet"
        print("\n========== download hs300 historical weights ==========")
        if self.should_skip(filename):
            existing = self.read_existing(filename)
            if existing is not None and "con_code" in existing.columns:
                codes = sorted(existing["con_code"].dropna().unique().tolist())
                if self.max_stocks:
                    codes = codes[: self.max_stocks]
                print(f"SKIP: {filename}; stock_count={len(codes)}")
                return codes

        frames = []
        current = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        while current <= end:
            month_end = current + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            if month_end > end:
                month_end = end
            df = self.safe_call(
                self.pro.index_weight,
                "index_weight",
                index_code=self.index_code,
                start_date=current.strftime("%Y%m%d"),
                end_date=month_end.strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                frames.append(df)
            current = month_end + pd.Timedelta(days=1)
            time.sleep(self.request_sleep)

        if not frames:
            return []
        result = pd.concat(frames, ignore_index=True).drop_duplicates()
        self.save_parquet(result, filename)
        codes = sorted(result["con_code"].dropna().unique().tolist())
        print(f"historical constituent count={len(codes)}")
        if self.max_stocks:
            codes = codes[: self.max_stocks]
            print(f"max_stocks enabled; using first {len(codes)} codes")
        return codes

    def download_stock_basic(self) -> None:
        filename = "stock_basic.parquet"
        print("\n========== download stock_basic ==========")
        if self.should_skip(filename):
            print(f"SKIP: {filename}")
            return
        df = self.safe_call(
            self.pro.stock_basic,
            "stock_basic",
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        self.save_parquet(df, filename)

    def download_index_daily(self) -> None:
        filename = "hs300_index.parquet"
        print("\n========== download hs300 index_daily ==========")
        if self.should_skip(filename):
            print(f"SKIP: {filename}")
            return
        df = self.safe_call(
            self.pro.index_daily,
            "index_daily",
            ts_code=self.index_code,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.save_parquet(df, filename)

    def download_hsgt(self) -> None:
        filename = "moneyflow_hsgt.parquet"
        print("\n========== download moneyflow_hsgt ==========")
        if self.should_skip(filename):
            print(f"SKIP: {filename}")
            return
        frames = []
        current = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        while current <= end:
            month_end = current + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            if month_end > end:
                month_end = end
            df = self.safe_call(
                self.pro.moneyflow_hsgt,
                "moneyflow_hsgt",
                start_date=current.strftime("%Y%m%d"),
                end_date=month_end.strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                frames.append(df)
            current = month_end + pd.Timedelta(days=1)
            time.sleep(self.request_sleep)
        if not frames:
            print("WARNING: no data for moneyflow_hsgt")
            return
        result = pd.concat(frames, ignore_index=True).drop_duplicates()
        self.save_parquet(result, filename)

    def download_by_stock(
        self,
        *,
        stock_codes: list[str],
        dataset: str,
        filename: str,
        fetcher: Callable[[str], pd.DataFrame | None],
        sleep_seconds: float | None = None,
    ) -> None:
        print(f"\n========== download {dataset} stocks={len(stock_codes)} ==========")
        if self.should_skip(filename):
            print(f"SKIP: {filename}")
            return

        frames = []
        for code in tqdm(stock_codes, desc=dataset):
            df = fetcher(code)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(self.request_sleep if sleep_seconds is None else sleep_seconds)

        if not frames:
            print(f"WARNING: no data for {dataset}")
            return
        result = pd.concat(frames, ignore_index=True)
        self.save_parquet(result, filename)

    def download_daily(self, stock_codes: list[str]) -> None:
        self.download_by_stock(
            stock_codes=stock_codes,
            dataset="daily",
            filename="daily.parquet",
            fetcher=lambda code: self.safe_call(
                ts.pro_bar,
                "daily",
                api=self.pro,
                ts_code=code,
                adj="qfq",
                start_date=self.start_date,
                end_date=self.end_date,
            ),
            sleep_seconds=0.15,
        )

    def download_daily_basic(self, stock_codes: list[str]) -> None:
        fields = (
            "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
            "pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,total_mv,circ_mv"
        )
        self.download_by_stock(
            stock_codes=stock_codes,
            dataset="daily_basic",
            filename="daily_basic.parquet",
            fetcher=lambda code: self.safe_call(
                self.pro.daily_basic,
                "daily_basic",
                ts_code=code,
                start_date=self.start_date,
                end_date=self.end_date,
                fields=fields,
            ),
            sleep_seconds=0.15,
        )

    def download_fina_indicator(self, stock_codes: list[str]) -> None:
        self.download_by_stock(
            stock_codes=stock_codes,
            dataset="fina_indicator",
            filename="fina_indicator.parquet",
            fetcher=lambda code: self.safe_call(
                self.pro.fina_indicator,
                "fina_indicator",
                ts_code=code,
                start_date=self.start_date,
                end_date=self.end_date,
            ),
            sleep_seconds=0.2,
        )

    def download_moneyflow(self, stock_codes: list[str]) -> None:
        self.download_by_stock(
            stock_codes=stock_codes,
            dataset="moneyflow",
            filename="moneyflow.parquet",
            fetcher=lambda code: self.safe_call(
                self.pro.moneyflow,
                "moneyflow",
                ts_code=code,
                start_date=self.start_date,
                end_date=self.end_date,
            ),
            sleep_seconds=0.15,
        )

    def download_report_rc(self, stock_codes: list[str]) -> None:
        self.download_by_stock(
            stock_codes=stock_codes,
            dataset="report_rc",
            filename="report_rc.parquet",
            fetcher=lambda code: self.safe_call(
                self.pro.report_rc,
                "report_rc",
                ts_code=code,
                start_date=self.start_date,
                end_date=self.end_date,
            ),
            sleep_seconds=0.2,
        )

    def run(self, datasets: set[str]) -> None:
        stock_codes = self.download_index_weight()
        if not stock_codes:
            raise SystemExit("No HS300 historical constituents downloaded; abort.")

        self.save_metadata(stock_codes)

        if "stock_basic" in datasets:
            self.download_stock_basic()
        if "index_daily" in datasets:
            self.download_index_daily()
        if "moneyflow_hsgt" in datasets:
            self.download_hsgt()
        if "daily" in datasets:
            self.download_daily(stock_codes)
        if "daily_basic" in datasets:
            self.download_daily_basic(stock_codes)
        if "fina_indicator" in datasets:
            self.download_fina_indicator(stock_codes)
        if "moneyflow" in datasets:
            self.download_moneyflow(stock_codes)
        if "report_rc" in datasets:
            self.download_report_rc(stock_codes)

        self.save_failures()


def parse_datasets(value: str) -> set[str]:
    if value == "all":
        return set(DATASET_ORDER)
    datasets = {item.strip() for item in value.split(",") if item.strip()}
    unknown = sorted(datasets - set(DATASET_ORDER))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown datasets: {unknown}. Available: {DATASET_ORDER}")
    datasets.add("index_weight")
    return datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download HS300 quant datasets from Tushare and save as Parquet.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date, e.g. 20200101")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="End date, e.g. 20260515")
    parser.add_argument("--index-code", default=DEFAULT_INDEX_CODE, help="Index code, default HS300 399300.SZ")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Output directory for parquet files.")
    parser.add_argument("--datasets", type=parse_datasets, default=set(DATASET_ORDER), help="all or comma list.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.3)
    parser.add_argument("--skip-existing", action="store_true", help="Skip parquet files already present.")
    parser.add_argument("--max-stocks", type=int, default=None, help="Debug mode: only download first N constituents.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    downloader = TushareDownloader(
        start_date=args.start_date,
        end_date=args.end_date,
        index_code=args.index_code,
        data_dir=Path(args.data_dir),
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        request_sleep=args.request_sleep,
        skip_existing=args.skip_existing,
        max_stocks=args.max_stocks,
    )
    print("Start Tushare HS300 quant data download")
    print(f"range={downloader.start_date}~{downloader.end_date}")
    print(f"data_dir={downloader.data_dir}")
    print(f"datasets={sorted(args.datasets)}")
    downloader.run(args.datasets)
    print("\nAll requested datasets finished.")


if __name__ == "__main__":
    main()
