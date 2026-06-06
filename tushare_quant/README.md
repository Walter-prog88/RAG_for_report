# tushare_quant

Independent Tushare-based quant data project.

This folder is separate from `akshare_daily_data` and `data_providers` because
Tushare has token-based permissions, point thresholds, and its own Pro API
semantics.

## Configuration

Do not hardcode tokens in source files.

```bash
export TUSHARE_TOKEN="your_token"
export TUSHARE_HTTP_URL="http://8.136.22.187:8011/"
```

`TUSHARE_HTTP_URL` defaults to `http://8.136.22.187:8011/`.

## Smoke Test

```bash
python3 tushare_quant/smoke_test.py
```

The smoke test checks:

1. `index_basic`
2. `ts.pro_bar`
3. `fina_indicator`
4. `report_rc`
5. `index_weight` for HS300 constituents

## Phase 1: Local Data Download

Download all HS300 project datasets into local Parquet files:

```bash
python3 tushare_quant/download_data.py
```

Default range:

- `2020-01-01` to `2026-05-15`
- Index: `399300.SZ`
- Output: `tushare_quant/quant_data/`

Outputs:

- `hs300_weight.parquet`
- `daily.parquet`
- `daily_basic.parquet`
- `fina_indicator.parquet`
- `moneyflow.parquet`
- `moneyflow_hsgt.parquet`
- `report_rc.parquet`
- `stock_basic.parquet`
- `hs300_index.parquet`
- `metadata.json`
- `download_failures.json` if any per-symbol calls fail

Resume/skip existing files:

```bash
python3 tushare_quant/download_data.py --skip-existing
```

Small test run:

```bash
python3 tushare_quant/download_data.py --max-stocks 3 --data-dir tushare_quant/quant_data_sample
```

Only selected datasets:

```bash
python3 tushare_quant/download_data.py --datasets daily,daily_basic --skip-existing
```

## Inspect Parquet Files

Parquet is a binary columnar format. If your IDE cannot open it directly,
export a small CSV preview first:

```bash
python3 tushare_quant/inspect_parquet.py --data-dir tushare_quant/quant_data --list
python3 tushare_quant/inspect_parquet.py --data-dir tushare_quant/quant_data --file hs300_weight.parquet --export --rows 1000
```

Export every parquet file to CSV previews:

```bash
python3 tushare_quant/inspect_parquet.py --data-dir tushare_quant/quant_data --export-all --rows 1000
```

Export all rows with `--rows 0`. Excel format is also supported:

```bash
python3 tushare_quant/inspect_parquet.py --data-dir tushare_quant/quant_data --file hs300_weight.parquet --export --format xlsx
```

## Merge Local Parquet Ranges

Merge non-overlapping local download ranges into one Parquet directory:

```bash
python3 tushare_quant/merge_data.py \
  --input-dirs tushare_quant/quant_data_2020_20250514 tushare_quant/quant_data_1y \
  --output-dir tushare_quant/quant_data_6y
```

## Suggested Structure

Future modules should stay here:

- `client.py`: Tushare client/config
- `smoke_test.py`: permission and connectivity checks
- `download_data.py`: phase-1 local Parquet data materialization
- `providers/`: reusable data fetchers
- `datasets/`: scripts that materialize local CSV/Parquet datasets
- `models/`: model training code using Tushare data
- `outputs/`: generated local artifacts
