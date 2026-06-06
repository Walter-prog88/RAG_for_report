from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIRS = [
    Path("tushare_quant") / "quant_data_2020_20250514",
    Path("tushare_quant") / "quant_data_1y",
]
DEFAULT_OUTPUT_DIR = Path("tushare_quant") / "quant_data_6y"

DATASET_KEYS = {
    "hs300_weight.parquet": ["index_code", "con_code", "trade_date"],
    "daily.parquet": ["ts_code", "trade_date"],
    "daily_basic.parquet": ["ts_code", "trade_date"],
    "fina_indicator.parquet": ["ts_code", "ann_date", "end_date"],
    "moneyflow.parquet": ["ts_code", "trade_date"],
    "moneyflow_hsgt.parquet": ["trade_date"],
    "report_rc.parquet": ["ts_code", "report_date", "org_name", "author_name", "report_title", "quarter"],
    "stock_basic.parquet": ["ts_code"],
    "hs300_index.parquet": ["ts_code", "trade_date"],
}

SORT_COLUMNS = {
    "hs300_weight.parquet": ["trade_date", "index_code", "con_code"],
    "daily.parquet": ["ts_code", "trade_date"],
    "daily_basic.parquet": ["ts_code", "trade_date"],
    "fina_indicator.parquet": ["ts_code", "end_date", "ann_date"],
    "moneyflow.parquet": ["ts_code", "trade_date"],
    "moneyflow_hsgt.parquet": ["trade_date"],
    "report_rc.parquet": ["ts_code", "report_date", "org_name"],
    "stock_basic.parquet": ["ts_code"],
    "hs300_index.parquet": ["ts_code", "trade_date"],
}


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def merge_one(filename: str, input_dirs: list[Path], output_dir: Path) -> dict:
    frames = []
    sources = []
    for input_dir in input_dirs:
        path = input_dir / filename
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        frames.append(df)
        sources.append({"path": str(path), "rows": len(df)})

    if not frames:
        return {"filename": filename, "rows": 0, "sources": sources, "status": "missing"}

    merged = pd.concat(frames, ignore_index=True)
    rows_before = len(merged)

    key_columns = existing_columns(merged, DATASET_KEYS.get(filename, []))
    if key_columns:
        merged = merged.drop_duplicates(subset=key_columns, keep="last")
    else:
        merged = merged.drop_duplicates(keep="last")

    sort_columns = existing_columns(merged, SORT_COLUMNS.get(filename, []))
    if sort_columns:
        merged = merged.sort_values(sort_columns).reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    output_path = output_dir / filename
    merged.to_parquet(output_path, index=False)
    print(f"OK: merged {filename} rows={len(merged)} dropped_duplicates={rows_before - len(merged)}")
    return {
        "filename": filename,
        "rows": len(merged),
        "columns": list(merged.columns),
        "sources": sources,
        "duplicate_rows_dropped": rows_before - len(merged),
        "status": "ok",
    }


def merge_metadata(input_dirs: list[Path], output_dir: Path, file_results: list[dict]) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "files": file_results,
    }
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: metadata saved {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge local Tushare parquet datasets without CSV export.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUT_DIRS],
        help="Input directories in priority order. Later directories win on duplicate keys.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Merged parquet output directory.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=sorted(DATASET_KEYS),
        help="Parquet filenames to merge. Defaults to all project datasets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dirs = [Path(value) for value in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_results = []
    for filename in args.files:
        file_results.append(merge_one(filename, input_dirs, output_dir))
    merge_metadata(input_dirs, output_dir, file_results)


if __name__ == "__main__":
    main()
