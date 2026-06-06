from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("tushare_quant") / "quant_data"


def list_parquet_files(data_dir: Path) -> None:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {data_dir}")
        return

    print(f"Parquet files in {data_dir}:")
    for path in files:
        try:
            df = pd.read_parquet(path)
            print(f"- {path.name}: shape={df.shape}, columns={list(df.columns)}")
        except Exception as exc:
            print(f"- {path.name}: failed to read: {exc}")


def preview_file(path: Path, rows: int) -> None:
    df = pd.read_parquet(path)
    print(f"path={path}")
    print(f"shape={df.shape}")
    print(f"columns={list(df.columns)}")
    print(df.head(rows).to_string(index=False))


def export_file(path: Path, out_dir: Path, rows: int | None, fmt: str) -> Path:
    df = pd.read_parquet(path)
    if rows is not None:
        df = df.head(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "csv" if fmt == "csv" else "xlsx"
    output_path = out_dir / f"{path.stem}.{suffix}"

    if fmt == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)

    print(f"Exported {output_path} shape={df.shape}")
    return output_path


def export_all(data_dir: Path, out_dir: Path, rows: int | None, fmt: str) -> None:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {data_dir}")
        return

    for path in files:
        export_file(path, out_dir, rows, fmt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or export Tushare parquet files.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing parquet files.")
    parser.add_argument("--file", help="Specific parquet file path or filename inside data-dir.")
    parser.add_argument("--rows", type=int, default=1000, help="Rows to preview/export. Use 0 for all rows.")
    parser.add_argument("--out-dir", default=None, help="Output directory for exported files.")
    parser.add_argument("--format", choices=["csv", "xlsx"], default="csv")
    parser.add_argument("--list", action="store_true", help="List parquet files with shapes and columns.")
    parser.add_argument("--preview", action="store_true", help="Print a text preview of one parquet file.")
    parser.add_argument("--export", action="store_true", help="Export one parquet file.")
    parser.add_argument("--export-all", action="store_true", help="Export every parquet file in data-dir.")
    return parser


def resolve_file(data_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = data_dir / value
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot find parquet file: {value}")


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    row_limit = None if args.rows == 0 else args.rows
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "exports"

    if args.list:
        list_parquet_files(data_dir)
        return

    if args.preview:
        if not args.file:
            raise SystemExit("--preview requires --file")
        preview_file(resolve_file(data_dir, args.file), args.rows)
        return

    if args.export:
        if not args.file:
            raise SystemExit("--export requires --file")
        export_file(resolve_file(data_dir, args.file), out_dir, row_limit, args.format)
        return

    if args.export_all:
        export_all(data_dir, out_dir, row_limit, args.format)
        return

    list_parquet_files(data_dir)


if __name__ == "__main__":
    main()
