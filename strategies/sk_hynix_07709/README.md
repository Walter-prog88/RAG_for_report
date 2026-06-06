# SK hynix / 07709 Target Mapping Strategy

This folder models one specific trade idea, with bull case as the main focus:

1. Maintain several SK hynix target anchors: stale Goldman base, consensus average, street high, and a 3M KRW bull case.
2. Compare each target with the current/reference SK hynix share price.
3. Map the implied underlying return to `07709.HK`, a Hong Kong listed daily 2x leveraged product on SK hynix.
4. Report a price range instead of a single target because 07709 has daily reset, fees, tracking error, FX, and premium/discount effects.

## Current Base Case

Config file: `assumptions.json`

- Underlying: SK hynix Inc. (`000660.KS`)
- Product: CSOP SK Hynix Daily (2x) Leveraged Product (`07709.HK`)
- Goldman target used here: KRW 1,350,000, reported on 2026-03-12
- Consensus average target: KRW 1,771,866.34
- Street high target: KRW 2,500,000
- Bull case target: KRW 3,000,000
- Reference SK hynix price: KRW 1,880,000, dated 2026-05-11
- Current 07709 price basis: HKD 100.0, an approximate user-observed market price on 2026-05-11
- Reference 07709 NAV kept for context: HKD 78.339, dated 2026-05-08

With those references, Goldman target is below SK hynix spot/reference price. The default strategy state should therefore be conservative for the Goldman anchor. The bull case is different: it assumes the market accepts a higher valuation framework for durable HBM earnings.

## Run

```bash
python3 strategies/sk_hynix_07709/run_strategy.py
```

Run only the bull case:

```bash
python3 strategies/sk_hynix_07709/run_strategy.py --anchor bull_case_3m
```

Override assumptions:

```bash
python3 strategies/sk_hynix_07709/run_strategy.py \
  --sk-price 1601000 \
  --etf-price 78.339 \
  --target 1800000 \
  --holding-days 20
```

Try live Yahoo prices first:

```bash
python3 strategies/sk_hynix_07709/run_strategy.py --refresh-market
```

The scenario table is written to `latest_projection.csv`.
The anchor-level summary is written to `latest_anchor_summary.csv`.

## Model

```text
underlying_return = target_price / current_price - 1
gross_2x_return = 2 * underlying_return
product_return = gross_2x_return - decay - fee_drag + tracking_drag + premium_discount
projected_07709 = current_07709 * (1 + product_return)
```

`decay` and `premium_discount` are scenario inputs, not predictions.

## Bull Case Interpretation

The bull case target is KRW 3,000,000. At a KRW 1,880,000 SK hynix reference price, that implies roughly 59.6% underlying upside. A simple 2x mapping gives roughly 119.1% before path, fee, tracking, FX, and premium/discount effects.

At a HKD 100 07709 price basis, the scenario range is generated across decay and premium/discount assumptions, rather than assuming a clean 2x terminal return.

## Important Limits

- 07709 seeks 2x of SK hynix's **daily** performance, not 2x of a multi-week or multi-month target move.
- Multi-day results depend on path. A volatile sideways path can lose money even if the underlying ends near flat.
- 07709 trades in HKD, has USD base currency, and references a KRW stock, so FX and time-zone gaps matter.
- The product may trade at a premium/discount to NAV.
- This is a research scaffold, not investment advice.
