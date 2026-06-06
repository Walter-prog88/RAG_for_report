from __future__ import annotations

import argparse
import traceback

import tushare as ts

try:
    from .client import get_pro_api
except ImportError:
    from client import get_pro_api


def print_section(title: str) -> None:
    print("=" * 50)
    print(title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Tushare Pro permissions and project data access.")
    parser.add_argument("--ts-code", default="600519.SH", help="Stock ts_code for daily/fundamental tests.")
    parser.add_argument("--start-date", default="20240101", help="Daily start date.")
    parser.add_argument("--end-date", default="20240131", help="Daily end date.")
    parser.add_argument("--fina-start", default="20230101", help="Financial indicator start date.")
    parser.add_argument("--fina-end", default="20231231", help="Financial indicator end date.")
    parser.add_argument("--index-code", default="399300.SZ", help="Index code for constituent weight test.")
    args = parser.parse_args()

    pro = get_pro_api()

    print_section("测试1：指数基础信息")
    df = pro.index_basic(limit=5)
    print(df)

    print_section("测试2：股票日线")
    df = ts.pro_bar(api=pro, ts_code=args.ts_code, start_date=args.start_date, end_date=args.end_date)
    print(df.head())

    print_section("测试3：财务指标（验证是否真的高积分）")
    df = pro.fina_indicator(ts_code=args.ts_code, start_date=args.fina_start, end_date=args.fina_end)
    print(df.head())

    print_section("测试4：盈利预测数据（验证10000积分权限）")
    try:
        df = pro.report_rc(ts_code=args.ts_code, start_date=args.fina_start, end_date=args.fina_end)
        print(df.head())
        print("OK: 10000积分权限可用")
    except Exception as e:
        print(f"NO: 没有10000积分权限：{e}")
        traceback.print_exc(limit=1)

    print_section("测试5：沪深300成分股")
    df = pro.index_weight(index_code=args.index_code, start_date=args.start_date, end_date=args.end_date)
    print(df.head())


if __name__ == "__main__":
    main()
