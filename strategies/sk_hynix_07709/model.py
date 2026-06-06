"""
SK hynix target-price to 07709.HK range mapping.

This is a scenario model, not a price target engine. 07709 is a daily reset
2x leveraged product, so multi-day returns depend on path, tracking, fees,
premium/discount, and FX effects. The model intentionally reports ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable


@dataclass(frozen=True)
class ProductAssumptions:
    leverage: float = 2.0
    annual_ongoing_charge: float = 0.02
    estimated_annual_tracking_difference: float = -0.0015


@dataclass(frozen=True)
class ScenarioResult:
    underlying_return: float
    leveraged_return_before_costs: float
    holding_days: int
    decay_rate: float
    premium_discount_rate: float
    fee_drag: float
    tracking_drag: float
    product_return: float
    projected_price: float


def underlying_return(current_price: float, target_price: float) -> float:
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    return target_price / current_price - 1


def estimate_range(
    *,
    underlying_current: float,
    underlying_target: float,
    product_current: float,
    holding_days: int,
    assumptions: ProductAssumptions,
    decay_rates: Iterable[float],
    premium_discount_rates: Iterable[float],
) -> list[ScenarioResult]:
    """
    Map underlying target return to 07709 scenario prices.

    Formula:
        underlying_return = target / current - 1
        gross_2x_return = leverage * underlying_return
        product_return = gross_2x_return - decay - fee_drag + tracking_drag + premium_discount

    `tracking_drag` keeps the sign from the product factsheet. A negative annual
    tracking difference lowers the projected return.
    """
    u_ret = underlying_return(underlying_current, underlying_target)
    gross = assumptions.leverage * u_ret
    fee_drag = assumptions.annual_ongoing_charge * holding_days / 365
    tracking_drag = assumptions.estimated_annual_tracking_difference * holding_days / 365

    rows: list[ScenarioResult] = []
    for decay_rate, premium_discount_rate in product(decay_rates, premium_discount_rates):
        product_return = gross - decay_rate - fee_drag + tracking_drag + premium_discount_rate
        projected_price = product_current * (1 + product_return)
        rows.append(
            ScenarioResult(
                underlying_return=u_ret,
                leveraged_return_before_costs=gross,
                holding_days=holding_days,
                decay_rate=decay_rate,
                premium_discount_rate=premium_discount_rate,
                fee_drag=fee_drag,
                tracking_drag=tracking_drag,
                product_return=product_return,
                projected_price=max(projected_price, 0.0),
            )
        )
    return rows


def summarize(results: list[ScenarioResult]) -> dict:
    if not results:
        return {}
    prices = [r.projected_price for r in results]
    returns = [r.product_return for r in results]
    return {
        "scenario_count": len(results),
        "min_projected_price": min(prices),
        "median_projected_price": sorted(prices)[len(prices) // 2],
        "max_projected_price": max(prices),
        "min_product_return": min(returns),
        "max_product_return": max(returns),
        "underlying_return": results[0].underlying_return,
        "gross_2x_return": results[0].leveraged_return_before_costs,
    }


def signal_from_target_gap(gap: float) -> str:
    if gap >= 0.15:
        return "target_implies_positive_2x_upside"
    if gap >= 0.05:
        return "modest_positive_upside"
    if gap >= -0.05:
        return "target_near_spot"
    return "target_below_spot"
