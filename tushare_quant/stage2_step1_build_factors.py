"""
Stage 2 Step 1: 构造候选因子库
输入：aligned_panel.parquet
输出：panel_with_factors.parquet
"""

import numpy as np
import pandas as pd


DATA_DIR = "./quant_data_6y"

print("📥 加载对齐面板数据...")
panel = pd.read_parquet(f"{DATA_DIR}/aligned_panel.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"原始 shape: {panel.shape}")


# ========== 价值类因子 ==========
print("\n🔧 构造价值类因子...")
panel["fac_ep"] = 1 / panel["pe_ttm"]  # 盈利收益率 = 1/PE，越大越便宜
panel["fac_bp"] = 1 / panel["pb"]  # 账面市值比 = 1/PB
panel["fac_sp"] = 1 / panel["ps_ttm"]  # 销售收益率 = 1/PS
panel["fac_log_mv"] = np.log(panel["total_mv"])  # 对数市值（小市值因子）


# ========== 质量类因子 ==========
print("🔧 构造质量类因子...")
panel["fac_roe"] = panel["roe"]
panel["fac_gross_margin"] = panel["gross_margin"]
panel["fac_debt"] = -panel["debt_to_assets"]  # 负债越低越好，取负号
panel["fac_eps"] = panel["eps"]


# ========== 动量/反转类因子 ==========
print("🔧 构造动量/反转类因子...")
panel["ret_1d"] = panel.groupby("ts_code")["close"].pct_change(1, fill_method=None)

# 短期反转（5日、20日反转）
panel["fac_rev_5d"] = -panel.groupby("ts_code")["close"].pct_change(5, fill_method=None)
panel["fac_rev_20d"] = -panel.groupby("ts_code")["close"].pct_change(20, fill_method=None)

# 中期动量（60日、120日动量）
panel["fac_mom_60d"] = panel.groupby("ts_code")["close"].pct_change(60, fill_method=None)
panel["fac_mom_120d"] = panel.groupby("ts_code")["close"].pct_change(120, fill_method=None)

# 已实现波动率（过去20日收益率标准差，越低越好）
panel["fac_vol_20d"] = (
    -panel.groupby("ts_code")["ret_1d"]
    .rolling(20)
    .std()
    .reset_index(0, drop=True)
)


# ========== 量价/流动性类因子 ==========
print("🔧 构造量价/流动性类因子...")
panel["fac_turn_20d"] = (
    -panel.groupby("ts_code")["turnover_rate"]
    .rolling(20)
    .mean()
    .reset_index(0, drop=True)
)

panel["fac_volume_ratio"] = panel.groupby("ts_code")["volume"].transform(
    lambda x: x / x.rolling(20).mean()
)

panel["fac_mf_20d"] = (
    panel.groupby("ts_code")["net_mf_amount"]
    .rolling(20)
    .mean()
    .reset_index(0, drop=True)
)

panel["fac_hsgt_5d"] = (
    panel.groupby("ts_code")["hsgt_net_buy"]
    .rolling(5)
    .mean()
    .reset_index(0, drop=True)
)


# ========== 预期类因子 ==========
print("🔧 构造预期类因子...")
panel["fac_rating"] = panel["rating_score"]
panel["fac_eps_yield"] = panel["eps_forecast"] / panel["close"]
panel["fac_eps_revision_60d"] = panel.groupby("ts_code")["eps_forecast"].pct_change(60, fill_method=None)
panel["fac_eps_revision_20d"] = panel.groupby("ts_code")["eps_forecast"].pct_change(20, fill_method=None)
panel["fac_eps_growth_expect"] = panel["eps_forecast"] / panel["eps"]


# ========== 构造目标变量 y ==========
print("\n🎯 构造目标变量...")
panel["y_future_5d"] = panel.groupby("ts_code")["close"].transform(
    lambda x: x.shift(-5) / x - 1
)


# ========== 保存 ==========
factor_cols = [c for c in panel.columns if c.startswith("fac_")]
panel[factor_cols + ["y_future_5d"]] = panel[factor_cols + ["y_future_5d"]].replace(
    [np.inf, -np.inf], np.nan
)

print(f"\n✅ 总共构造了 {len(factor_cols)} 个因子：")
for f in factor_cols:
    print(f"   - {f}")

panel.to_parquet(f"{DATA_DIR}/panel_with_factors.parquet", index=False)
print(f"\n💾 已保存：{DATA_DIR}/panel_with_factors.parquet")
print(f"   shape: {panel.shape}")
