# data_providers

Lightweight data access layer for this project.

It focuses on the fast parts that are useful here:

- Batch Yahoo Finance history and quote snapshots through `yf.download(..., threads=True)`.
- Cache historical data, quotes, company info, news, and option chains under `.cache/data_providers/`.
- Wrap AkShare A-share/HK/US/index historical data with retry, cache, and modest concurrent batch fetches.
- Return `pandas.DataFrame` or normal Python dictionaries instead of subprocess JSON.

Examples:

```python
from data_providers import yahoo, akshare_cn

quotes = yahoo.batch_quotes(["TSLA", "NVDA", "AAPL"])
histories = yahoo.batch_history(["^SOX", "^GSPC", "^VIX"], period="5y")
summary = yahoo.option_volume_summary("TSLA", max_expirations=4)

csi1000 = akshare_cn.csi_constituents("000852")
bars = akshare_cn.a_share_history("000001", start_date="20240101", adjust="qfq")
many = akshare_cn.batch_a_share_history(["000001", "000002"], max_workers=2)
```

Cache location can be changed:

```bash
export FIN_DATA_CACHE=/absolute/path/to/cache
```

Notes:

- Yahoo option-chain requests are still per symbol and per expiration. Cache is the main speed-up there.
- AkShare speed depends on upstream Chinese data sites. Use `max_workers` carefully to avoid throttling.
- This package does not copy code from Fincept Terminal; it reimplements the useful patterns for this project.
