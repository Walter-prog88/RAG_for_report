"""
Stage 2 Step 2: 单因子IC检验
对每个因子计算每日IC和RankIC，汇总成因子评估表
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, NearConstantInputWarning, pearsonr, spearmanr
from tqdm import tqdm


DATA_DIR = "./quant_data_6y"1
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=NearConstantInputWarning)

print("📥 加载因子面板...")
panel = pd.read_parquet(f"{DATA_DIR}/panel_with_factors.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

# 关键过滤：只用沪深300成分股
panel = panel[panel["is_hs300"] == True].copy()
print(f"沪深300样本 shape: {panel.shape}")

factor_cols = [c for c in panel.columns if c.startswith("fac_")]
print(f"待检验因子数: {len(factor_cols)}")


def compute_daily_ic(panel, factor_col, y_col="y_future_5d"):
    """
    对每个交易日，计算该因子的IC和RankIC。
    返回：DataFrame，每行一个日期+IC+RankIC。
    """
    results = []

    for date, group in panel.groupby("trade_date"):
        valid = group[[factor_col, y_col]].dropna()

        # 样本量太少跳过（需要至少30只股票才有统计意义）
        if len(valid) < 30:
            continue

        x = valid[factor_col].to_numpy()
        y = valid[y_col].to_numpy()

        # 如果因子或未来收益没有横截面差异，相关性没有意义
        if x.std() == 0 or y.std() == 0:
            continue

        try:
            ic, _ = pearsonr(x, y)
        except Exception:
            ic = np.nan

        try:
            rank_ic, _ = spearmanr(x, y)
        except Exception:
            rank_ic = np.nan

        results.append(
            {
                "trade_date": date,
                "ic": ic,
                "rank_ic": rank_ic,
                "n_stocks": len(valid),
            }
        )

    return pd.DataFrame(results)


print("\n🔬 开始计算每个因子的IC和RankIC...")
summary_records = []
all_ic_series = {}

for factor in tqdm(factor_cols, desc="计算因子IC"):
    ic_df = compute_daily_ic(panel, factor)
    all_ic_series[factor] = ic_df

    if ic_df.empty:
        summary_records.append(
            {
                "factor": factor,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "icir": np.nan,
                "rank_ic_mean": np.nan,
                "rank_ic_std": np.nan,
                "rank_icir": np.nan,
                "ic_positive_rate": np.nan,
                "n_days": 0,
            }
        )
        continue

    ic_std = ic_df["ic"].std()
    rank_ic_std = ic_df["rank_ic"].std()

    summary_records.append(
        {
            "factor": factor,
            "ic_mean": ic_df["ic"].mean(),
            "ic_std": ic_std,
            "icir": ic_df["ic"].mean() / ic_std if ic_std > 0 else np.nan,
            "rank_ic_mean": ic_df["rank_ic"].mean(),
            "rank_ic_std": rank_ic_std,
            "rank_icir": ic_df["rank_ic"].mean() / rank_ic_std if rank_ic_std > 0 else np.nan,
            "ic_positive_rate": (ic_df["ic"] > 0).sum() / len(ic_df),
            "n_days": len(ic_df),
        }
    )

summary = pd.DataFrame(summary_records)

# 排序：按 |rank_icir| 降序（看绝对值，因为负向因子也有价值）
summary["abs_rank_icir"] = summary["rank_icir"].abs()
summary = summary.sort_values("abs_rank_icir", ascending=False).reset_index(drop=True)


print("\n" + "=" * 100)
print("📊 因子IC检验汇总（按 |RankICIR| 排序）")
print("=" * 100)

display_cols = [
    "factor",
    "ic_mean",
    "icir",
    "rank_ic_mean",
    "rank_icir",
    "ic_positive_rate",
    "n_days",
]
print(summary[display_cols].to_string(index=False, float_format="%.4f"))


print("\n" + "=" * 100)
print("🏆 因子评级（基于 RankIC + RankICIR）")
print("=" * 100)


def grade_factor(row):
    abs_ic = abs(row["rank_ic_mean"])
    abs_icir = abs(row["rank_icir"])

    if abs_ic >= 0.05 and abs_icir >= 0.5:
        return "⭐⭐⭐ 优秀"
    if abs_ic >= 0.03 and abs_icir >= 0.3:
        return "⭐⭐ 良好"
    if abs_ic >= 0.02 and abs_icir >= 0.2:
        return "⭐ 可用"
    return "❌ 弱"


summary["grade"] = summary.apply(grade_factor, axis=1)

for grade in ["⭐⭐⭐ 优秀", "⭐⭐ 良好", "⭐ 可用", "❌ 弱"]:
    factors_in_grade = summary[summary["grade"] == grade]
    if len(factors_in_grade) > 0:
        print(f"\n{grade}:")
        for _, row in factors_in_grade.iterrows():
            direction = "正向" if row["rank_ic_mean"] > 0 else "负向"
            print(
                f"  {row['factor']:30s} "
                f"RankIC={row['rank_ic_mean']:+.4f}  "
                f"RankICIR={row['rank_icir']:+.3f}  "
                f"方向={direction}"
            )


summary.to_csv(f"{DATA_DIR}/factor_ic_summary.csv", index=False)
print(f"\n💾 因子IC汇总已保存：{DATA_DIR}/factor_ic_summary.csv")

with open(f"{DATA_DIR}/all_ic_series.pkl", "wb") as f:
    pickle.dump(all_ic_series, f)
print(f"💾 每日IC时间序列已保存：{DATA_DIR}/all_ic_series.pkl")
