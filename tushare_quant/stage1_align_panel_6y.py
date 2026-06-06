from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "quant_data_6y"

DATASETS = {
    "daily": "daily.parquet",
    "daily_basic": "daily_basic.parquet",
    "fina": "fina_indicator.parquet",
    "moneyflow": "moneyflow.parquet",
    "report_rc": "report_rc.parquet",
    "hs300_weight": "hs300_weight.parquet",
    "stock_basic": "stock_basic.parquet",
    "hs300_index": "hs300_index.parquet",
    "hsgt": "moneyflow_hsgt.parquet",
}

RATING_SCORE = {
    "强烈推荐": 5,
    "强推": 5,
    "推荐": 4,
    "买入": 4,
    "增持": 3,
    "优于大市": 3,
    "跑赢行业": 3,
    "outperform": 3,
    "overweight": 3,
    "持有": 2,
    "中性": 2,
    "neutral": 2,
    "equal-weight": 2,
    "无": np.nan,
    "回避": 1,
    "减持": 1,
    "卖出": 1,
    "underperform": 1,
    "underweight": 1,
}

TARGET_COLUMNS = [
    "trade_date",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "total_mv",
    "turnover_rate",
    "roe",
    "gross_margin",
    "debt_to_assets",
    "eps",
    "net_mf_amount",
    "hsgt_net_buy",
    "eps_forecast",
    "target_price",
    "rating_score",
    "industry",
    "is_hs300",
]


def resolve_data_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path

    script_relative = SCRIPT_DIR / value
    if script_relative.exists():
        return script_relative

    raise FileNotFoundError(f"Cannot find data directory: {value}")


def parse_yyyymmdd(series: pd.Series, column_name: str) -> pd.Series:
    parsed = pd.to_datetime(series.astype("string"), format="%Y%m%d", errors="coerce")
    missing = parsed.isna().sum()
    if missing:
        raise ValueError(f"{column_name} has {missing} invalid dates")
    return parsed


def to_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in df.columns]
    if existing:
        df[existing] = df[existing].apply(pd.to_numeric, errors="coerce")
    return df


def load_all_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    print("📥 加载数据...")
    data = {}
    for key, filename in DATASETS.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_parquet(path)
        data[key] = df
        print(f"  {key}: {df.shape}")
    return data


def build_main_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n🔧 构建主面板...")

    daily = data["daily"].copy()
    daily_basic = data["daily_basic"].copy()
    moneyflow = data["moneyflow"].copy()

    for name, df in [
        ("daily.trade_date", daily),
        ("daily_basic.trade_date", daily_basic),
        ("moneyflow.trade_date", moneyflow),
    ]:
        df["trade_date"] = parse_yyyymmdd(df["trade_date"], name)

    daily = daily.rename(columns={"vol": "volume"})

    daily_cols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
    ]
    daily_basic_cols = [
        "ts_code",
        "trade_date",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "turnover_rate",
    ]
    moneyflow_cols = ["ts_code", "trade_date", "net_mf_amount"]

    panel = daily[[column for column in daily_cols if column in daily.columns]].merge(
        daily_basic[[column for column in daily_basic_cols if column in daily_basic.columns]],
        on=["ts_code", "trade_date"],
        how="left",
    )
    panel = panel.merge(
        moneyflow[[column for column in moneyflow_cols if column in moneyflow.columns]],
        on=["ts_code", "trade_date"],
        how="left",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "turnover_rate",
        "net_mf_amount",
    ]
    panel = to_numeric_columns(panel, numeric_columns)
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    print(f"  主面板 shape: {panel.shape}")
    return panel


def align_fina_data(panel: pd.DataFrame, fina: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 对齐财务数据...")

    fina = fina.copy()
    fina["ann_date"] = parse_yyyymmdd(fina["ann_date"], "fina.ann_date")

    keep_cols = [
        "ts_code",
        "ann_date",
        "roe",
        "gross_margin",
        "grossprofit_margin",
        "debt_to_assets",
        "eps",
    ]
    fina = fina[[column for column in keep_cols if column in fina.columns]]

    if "gross_margin" not in fina.columns and "grossprofit_margin" in fina.columns:
        fina = fina.rename(columns={"grossprofit_margin": "gross_margin"})
    elif "gross_margin" in fina.columns and "grossprofit_margin" in fina.columns:
        fina["gross_margin"] = fina["gross_margin"].fillna(fina["grossprofit_margin"])
        fina = fina.drop(columns=["grossprofit_margin"])

    fina = to_numeric_columns(fina, ["roe", "gross_margin", "debt_to_assets", "eps"])
    fina = fina.rename(columns={"ann_date": "trade_date"})
    fina = fina.dropna(subset=["trade_date"])
    fina = fina.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    result = pd.merge_asof(
        panel.sort_values(["trade_date", "ts_code"]),
        fina.sort_values(["trade_date", "ts_code"]),
        on="trade_date",
        by="ts_code",
        direction="backward",
    )

    print(f"  财务记录数: {len(fina)}")
    print(f"  对齐后 shape: {result.shape}")
    return result


def normalize_rating(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan

    lowered = text.lower()
    if lowered in RATING_SCORE:
        return RATING_SCORE[lowered]
    if text in RATING_SCORE:
        return RATING_SCORE[text]

    for key, score in RATING_SCORE.items():
        if key and key in lowered:
            return score
    return np.nan


def align_report_rc(panel: pd.DataFrame, report_rc: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 对齐盈利预测...")

    rc = report_rc.copy()
    rc["report_date"] = parse_yyyymmdd(rc["report_date"], "report_rc.report_date")

    print(f"  report_rc columns: {rc.columns.tolist()}")

    keep_cols = [
        "ts_code",
        "report_date",
        "quarter",
        "eps",
        "pe",
        "roe",
        "target_price",
        "max_price",
        "min_price",
        "rating",
    ]
    rc = rc[[column for column in keep_cols if column in rc.columns]]

    numeric_columns = ["eps", "pe", "roe", "target_price", "max_price", "min_price"]
    rc = to_numeric_columns(rc, numeric_columns)

    if "target_price" not in rc.columns:
        price_columns = [column for column in ["max_price", "min_price"] if column in rc.columns]
        if price_columns:
            rc["target_price"] = rc[price_columns].mean(axis=1, skipna=True)
        else:
            rc["target_price"] = np.nan

    if "rating" in rc.columns:
        rc["rating_score"] = rc["rating"].map(normalize_rating)
    else:
        rc["rating_score"] = np.nan

    rename_map = {
        "report_date": "trade_date",
        "eps": "eps_forecast",
        "pe": "pe_forecast",
        "roe": "roe_forecast",
    }
    rc = rc.rename(columns=rename_map)

    # 每只股票同一天可能有多个机构、多期预测。取同一发布日的均值，避免同日记录顺序影响结果。
    agg_spec = {}
    for column in ["eps_forecast", "pe_forecast", "roe_forecast", "target_price", "rating_score"]:
        if column in rc.columns:
            agg_spec[column] = "mean"

    rc = (
        rc.groupby(["ts_code", "trade_date"], as_index=False)
        .agg(agg_spec)
        .sort_values(["trade_date", "ts_code"])
    )

    result = pd.merge_asof(
        panel.sort_values(["trade_date", "ts_code"]),
        rc,
        on="trade_date",
        by="ts_code",
        direction="backward",
    )

    print(f"  研报发布日聚合记录数: {len(rc)}")
    print(f"  对齐后 shape: {result.shape}")
    return result


def add_hsgt(panel: pd.DataFrame, hsgt: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 加入沪深港通资金...")

    hsgt = hsgt.copy()
    hsgt["trade_date"] = parse_yyyymmdd(hsgt["trade_date"], "moneyflow_hsgt.trade_date")
    hsgt = to_numeric_columns(hsgt, ["north_money", "hgt", "sgt"])

    if "north_money" in hsgt.columns:
        hsgt["hsgt_net_buy"] = hsgt["north_money"]
    elif {"hgt", "sgt"}.issubset(hsgt.columns):
        hsgt["hsgt_net_buy"] = hsgt["hgt"] + hsgt["sgt"]
    else:
        hsgt["hsgt_net_buy"] = np.nan

    hsgt = hsgt[["trade_date", "hsgt_net_buy"]].drop_duplicates("trade_date", keep="last")
    result = panel.merge(hsgt, on="trade_date", how="left")

    print(f"  HSGT交易日数: {hsgt['trade_date'].nunique()}")
    print(f"  hsgt_net_buy 缺失行数: {result['hsgt_net_buy'].isna().sum()}")
    return result


def add_industry(panel: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 加入行业信息...")

    stock_basic = stock_basic.copy()
    industry_map = stock_basic[["ts_code", "industry"]].drop_duplicates("ts_code", keep="last")
    result = panel.merge(industry_map, on="ts_code", how="left")

    print(f"  行业缺失: {result['industry'].isna().sum()}")
    return result


def mark_hs300_member(panel: pd.DataFrame, hs300_weight: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 标记成分股...")

    weight = hs300_weight.copy()
    weight["trade_date"] = parse_yyyymmdd(weight["trade_date"], "hs300_weight.trade_date")
    if "ts_code" not in weight.columns and "con_code" in weight.columns:
        weight = weight.rename(columns={"con_code": "ts_code"})
    if "ts_code" not in weight.columns:
        raise ValueError("hs300_weight must contain ts_code or con_code")

    members = (
        weight[["trade_date", "ts_code"]]
        .drop_duplicates()
        .rename(columns={"trade_date": "snapshot_date"})
    )

    snapshot_dates = pd.DataFrame({"snapshot_date": sorted(members["snapshot_date"].unique())})
    trade_dates = pd.DataFrame({"trade_date": sorted(panel["trade_date"].unique())})
    trade_date_map = pd.merge_asof(
        trade_dates,
        snapshot_dates,
        left_on="trade_date",
        right_on="snapshot_date",
        direction="backward",
        tolerance=pd.Timedelta(days=45),
    )
    missing_snapshot = trade_date_map["snapshot_date"].isna()
    if missing_snapshot.any():
        # The one-year sample starts before its first HS300 weight snapshot.
        # Use the available snapshot from the same calendar month for that leading gap.
        same_month_snapshots = snapshot_dates.copy()
        same_month_snapshots["month"] = same_month_snapshots["snapshot_date"].dt.to_period("M")
        same_month_snapshots = same_month_snapshots.drop_duplicates("month", keep="first")

        same_month_map = trade_dates.loc[missing_snapshot].copy()
        same_month_map["month"] = same_month_map["trade_date"].dt.to_period("M")
        same_month_map = same_month_map.merge(
            same_month_snapshots[["month", "snapshot_date"]],
            on="month",
            how="left",
        )
        trade_date_map.loc[missing_snapshot, "snapshot_date"] = same_month_map["snapshot_date"].to_numpy()

    result = panel.merge(trade_date_map, on="trade_date", how="left")
    result = result.merge(
        members.assign(is_hs300=True),
        on=["snapshot_date", "ts_code"],
        how="left",
    )
    result["is_hs300"] = result["is_hs300"].eq(True)
    result = result.drop(columns=["snapshot_date"])

    print(f"  成分股快照数: {members['snapshot_date'].nunique()}")
    print(f"  无可用成分股快照交易日数: {trade_date_map['snapshot_date'].isna().sum()}")
    print(f"  成分股样本数: {int(result['is_hs300'].sum())}")
    return result


def finalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    for column in TARGET_COLUMNS:
        if column not in panel.columns:
            panel[column] = np.nan

    panel = panel[TARGET_COLUMNS]
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return panel


def print_quality_report(panel: pd.DataFrame) -> None:
    print("\n📊 数据质量检查:")
    print(f"   日期范围: {panel['trade_date'].min()} ~ {panel['trade_date'].max()}")
    print(f"   股票数: {panel['ts_code'].nunique()}")
    print(f"   交易日数: {panel['trade_date'].nunique()}")
    print("\n   字段缺失情况:")
    print(panel.isna().sum().sort_values(ascending=False).head(20))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1: align Tushare parquet datasets into a clean panel.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing Stage 0 parquet files.")
    parser.add_argument("--output", default=None, help="Output parquet path. Defaults to DATA_DIR/aligned_panel.parquet.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    output_path = Path(args.output).expanduser() if args.output else data_dir / "aligned_panel.parquet"

    print("=" * 60)
    print("Stage 1: 数据加载与对齐")
    print("=" * 60)
    print(f"DATA_DIR: {data_dir}")

    data = load_all_data(data_dir)
    panel = build_main_panel(data)
    panel = align_fina_data(panel, data["fina"])
    panel = align_report_rc(panel, data["report_rc"])
    panel = add_hsgt(panel, data["hsgt"])
    panel = add_industry(panel, data["stock_basic"])
    panel = mark_hs300_member(panel, data["hs300_weight"])
    panel = finalize_panel(panel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"✅ 完成！面板数据保存到: {output_path}")
    print(f"   总行数: {len(panel)}")
    print(f"   字段数: {len(panel.columns)}")
    print(f"   字段列表: {panel.columns.tolist()}")
    print("=" * 60)

    print_quality_report(panel)


if __name__ == "__main__":
    main()
