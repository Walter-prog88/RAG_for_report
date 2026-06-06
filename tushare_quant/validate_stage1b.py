"""
Stage 1B 结果验证
"""
import pandas as pd

panel = pd.read_parquet('./quant_data_6y/aligned_panel.parquet')

print("=" * 60)
print("【1】总体规模")
print("=" * 60)
print(f"shape: {panel.shape}")
print(f"日期范围: {panel['trade_date'].min()} ~ {panel['trade_date'].max()}")
print(f"股票数: {panel['ts_code'].nunique()}")
print(f"交易日数: {panel['trade_date'].nunique()}")

print("\n" + "=" * 60)
print("【2】字段缺失情况")
print("=" * 60)
missing = panel.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(panel) * 100).round(2)
result = pd.DataFrame({'缺失数': missing, '缺失率(%)': missing_pct})
print(result.head(20))

print("\n" + "=" * 60)
print("【3】成分股数量按年统计")
print("=" * 60)
panel['year'] = pd.to_datetime(panel['trade_date']).dt.year
yearly = panel.groupby('year').agg({
    'is_hs300': 'sum',
    'ts_code': 'nunique',
    'trade_date': 'nunique'
}).rename(columns={
    'is_hs300': '成分股样本数',
    'ts_code': '股票数',
    'trade_date': '交易日数'
})
print(yearly)

print("\n" + "=" * 60)
print("【4】关键因子可用性（仅看沪深300成分股）")
print("=" * 60)
hs300_panel = panel[panel['is_hs300'] == True]
key_factors = ['pe_ttm', 'pb', 'turnover_rate', 'roe', 'eps',
               'gross_margin', 'eps_forecast', 'rating_score', 'net_mf_amount']
for f in key_factors:
    if f in hs300_panel.columns:
        non_null = hs300_panel[f].notna().sum()
        total = len(hs300_panel)
        pct = non_null / total * 100
        print(f"  {f:20s}: {non_null:>7d} / {total:>7d} ({pct:5.1f}%)")
