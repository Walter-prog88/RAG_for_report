# AkShare Daily Stock Data

Fetch a clean daily A-share DataFrame for the past 3 years.

Default example: `300308` 中际旭创.

```bash
python3 akshare_daily_data/zhong_ji_daily.py
```

Save to CSV:

```bash
python3 akshare_daily_data/zhong_ji_daily.py --csv akshare_daily_data/300308_daily.csv
```

Fetch a longer date range:

```bash
python3 akshare_daily_data/zhong_ji_daily.py --start-date 20220101 --end-date 20250101 --adjust qfq
```

Increase retries and delay:

```bash
python3 akshare_daily_data/zhong_ji_daily.py --retries 8 --retry-sleep 3
```

BaoStock source:

```bash
python3 akshare_daily_data/baostock_daily.py --symbol 300308 --start-date 20220101 --end-date 20250101 --adjustflag 2
```

Use in later processing:

```python
from akshare_daily_data.zhong_ji_daily import fetch_a_share_daily

df = fetch_a_share_daily("300308", start_date="20220101", end_date="20250101", adjust="qfq")
```

Use BaoStock in later processing:

```python
from akshare_daily_data import fetch_a_share_daily_baostock

df = fetch_a_share_daily_baostock("300308", start_date="20220101", end_date="20250101")
```

Returned columns:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `turnover_rate`
- `pctChg`

Build factor columns:

```bash
python3 akshare_daily_data/factor_stage.py --symbol 300308 --start-date 20220101 --end-date 20250101
```

Save factor DataFrame:

```bash
python3 akshare_daily_data/factor_stage.py --symbol 300308 --start-date 20220101 --end-date 20250101 --csv akshare_daily_data/300308_factors.csv
```
