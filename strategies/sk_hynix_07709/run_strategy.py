#!/usr/bin/env python3
"""
Run the SK hynix / 07709 target-price mapping strategy.

Examples:
    python3 strategies/sk_hynix_07709/run_strategy.py
    python3 strategies/sk_hynix_07709/run_strategy.py --anchor bull_case_3m
    python3 strategies/sk_hynix_07709/run_strategy.py --sk-price 1880000 --etf-price 100
    python3 strategies/sk_hynix_07709/run_strategy.py --target 1800000 --holding-days 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model import ProductAssumptions, estimate_range, signal_from_target_gap, summarize


ROOT = Path(__file__).resolve().parent
ASSUMPTIONS_PATH = ROOT / "assumptions.json"
OUTPUT_PATH = ROOT / "latest_projection.csv"
SUMMARY_OUTPUT_PATH = ROOT / "latest_anchor_summary.csv"


def load_assumptions() -> dict:
    return json.loads(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))


def try_latest_yahoo_price(symbol: str) -> float | None:
    try:
        from data_providers import yahoo

        quotes = yahoo.batch_quotes([symbol])
        if quotes.empty or symbol not in quotes.index:
            return None
        value = quotes.loc[symbol, "price"]
        return float(value) if pd.notna(value) else None
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Map SK hynix target price to 07709.HK scenarios.")
    parser.add_argument("--sk-price", type=float, default=None, help="Current SK hynix price in KRW.")
    parser.add_argument("--etf-price", type=float, default=None, help="Current 07709 price/NAV in HKD.")
    parser.add_argument("--target", type=float, default=None, help="SK hynix target price in KRW.")
    parser.add_argument("--anchor", default=None, help="Run only one configured anchor, e.g. bull_case_3m.")
    parser.add_argument("--holding-days", type=int, default=None, help="Expected holding period in calendar days.")
    parser.add_argument("--refresh-market", action="store_true", help="Try live Yahoo prices before fallback values.")
    return parser


def _format_krw(value: float) -> str:
    return f"KRW {value:,.0f}"


def _format_hkd(value: float) -> str:
    return f"HKD {value:,.2f}"


def select_anchors(cfg: dict, *, target: float | None, anchor_name: str | None) -> list[dict]:
    if target is not None:
        return [{
            "name": "manual_target",
            "label": "Manual target",
            "target_krw": target,
            "date": cfg["as_of"],
            "stance": "manual",
            "source_note": "Command-line override.",
        }]

    anchors = list(cfg.get("target_anchors", []))
    if anchor_name:
        anchors = [a for a in anchors if a["name"] == anchor_name]
        if not anchors:
            available = ", ".join(a["name"] for a in cfg.get("target_anchors", []))
            raise SystemExit(f"Unknown anchor '{anchor_name}'. Available: {available}")

    def sort_key(anchor: dict) -> tuple[int, float]:
        if anchor.get("stance") == "bull_case":
            return (0, -float(anchor["target_krw"]))
        return (1, -float(anchor["target_krw"]))

    return sorted(anchors, key=sort_key)


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_assumptions()
    underlying_cfg = cfg["underlying"]
    product_cfg = cfg["product"]
    scenario_cfg = cfg["scenario"]

    sk_price = args.sk_price
    etf_price = args.etf_price

    if args.refresh_market:
        sk_price = sk_price or try_latest_yahoo_price(underlying_cfg["ticker"])
        etf_price = etf_price or try_latest_yahoo_price(product_cfg["ticker"])

    sk_price = sk_price or float(underlying_cfg["reference_price_krw"])
    etf_price = etf_price or float(product_cfg.get("current_price_hkd") or product_cfg["reference_nav_hkd"])
    holding_days = args.holding_days or int(scenario_cfg["holding_days"])
    anchors = select_anchors(cfg, target=args.target, anchor_name=args.anchor)

    assumptions = ProductAssumptions(
        leverage=float(product_cfg["leverage"]),
        annual_ongoing_charge=float(product_cfg["annual_ongoing_charge"]),
        estimated_annual_tracking_difference=float(product_cfg["estimated_annual_tracking_difference"]),
    )

    scenario_rows = []
    summary_rows = []
    for anchor in anchors:
        target = float(anchor["target_krw"])
        results = estimate_range(
            underlying_current=sk_price,
            underlying_target=target,
            product_current=etf_price,
            holding_days=holding_days,
            assumptions=assumptions,
            decay_rates=scenario_cfg["decay_rates"],
            premium_discount_rates=scenario_cfg["premium_discount_rates"],
        )
        summary = summarize(results)
        signal = signal_from_target_gap(summary["underlying_return"])
        for row in results:
            scenario_rows.append({
                "anchor": anchor["name"],
                "label": anchor["label"],
                "target_krw": target,
                **row.__dict__,
            })
        summary_rows.append({
            "anchor": anchor["name"],
            "label": anchor["label"],
            "stance": anchor["stance"],
            "target_krw": target,
            "target_date": anchor["date"],
            "underlying_return": summary["underlying_return"],
            "gross_2x_return": summary["gross_2x_return"],
            "min_projected_price": summary["min_projected_price"],
            "median_projected_price": summary["median_projected_price"],
            "max_projected_price": summary["max_projected_price"],
            "min_product_return": summary["min_product_return"],
            "max_product_return": summary["max_product_return"],
            "signal": signal,
            "source_note": anchor["source_note"],
        })

    detail_frame = pd.DataFrame(scenario_rows).sort_values(["target_krw", "projected_price"], ascending=[False, True])
    summary_frame = pd.DataFrame(summary_rows).sort_values("target_krw", ascending=False)
    detail_frame.to_csv(OUTPUT_PATH, index=False)
    summary_frame.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    bull_row = None
    if not summary_frame.empty:
        bull_candidates = summary_frame[summary_frame["stance"].eq("bull_case")]
        bull_row = bull_candidates.iloc[0] if not bull_candidates.empty else summary_frame.iloc[0]

    etf_reference_label = (
        f"current approx ({product_cfg.get('current_price_date')})"
        if product_cfg.get("current_price_hkd") and args.etf_price is None
        else "manual/live input"
        if args.etf_price is not None or args.refresh_market
        else f"NAV/reference ({product_cfg.get('reference_nav_date')})"
    )

    print("=" * 72)
    print("SK hynix -> 07709.HK target mapping")
    print("=" * 72)
    print(f"As of config             : {cfg['as_of']}")
    print(f"SK hynix current/reference: KRW {sk_price:,.0f}")
    print(f"07709 price basis         : HKD {etf_price:,.3f} [{etf_reference_label}]")
    print(f"Holding days              : {holding_days}")
    print("-" * 72)
    if bull_row is not None:
        print("Bull case focus")
        print(f"  Anchor                 : {bull_row['label']} ({bull_row['anchor']})")
        print(f"  Target                 : {_format_krw(float(bull_row['target_krw']))}")
        print(f"  Underlying upside      : {bull_row['underlying_return']:.2%}")
        print(f"  Gross 2x mapped return : {bull_row['gross_2x_return']:.2%}")
        print(
            "  07709 scenario range   : "
            f"{_format_hkd(float(bull_row['min_projected_price']))} - "
            f"{_format_hkd(float(bull_row['max_projected_price']))}"
        )
        print(f"  Signal                 : {bull_row['signal']}")
        print("-" * 72)

    print("Anchor summary")
    for _, row in summary_frame.iterrows():
        print(
            f"  {row['anchor']:<16} "
            f"target={_format_krw(float(row['target_krw'])):<16} "
            f"underlying={row['underlying_return']:>7.2%} "
            f"07709={_format_hkd(float(row['min_projected_price']))}~{_format_hkd(float(row['max_projected_price']))}"
        )

    valuation = cfg.get("valuation_context", {})
    if valuation:
        print("-" * 72)
        print("Valuation context")
        print(f"  Market cap             : KRW {valuation['market_cap_krw'] / 1e12:,.1f}T")
        print(f"  TTM revenue            : KRW {valuation['ttm_revenue_krw'] / 1e12:,.1f}T")
        print(f"  P/S                    : {valuation['ps_ratio']:.2f}x")
        print(f"  Bull 2026 OP estimate  : KRW {valuation['sk_securities_2026_operating_profit_krw'] / 1e12:,.1f}T")
        print(f"  Bull 2027 OP estimate  : KRW {valuation['sk_securities_2027_operating_profit_krw'] / 1e12:,.1f}T")

    print("-" * 72)
    print(f"Scenario table saved     : {OUTPUT_PATH}")
    print(f"Anchor summary saved     : {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
