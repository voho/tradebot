# Does any of this generalize? The ETH falsification test

Every conclusion in this repo rested on BTC 2017–2026. That sounds like
1.01M observations and is really about **three** independent regime
events, so a filter fitted to those would look identical to one that
works. This is the cheapest experiment that can tell the difference.

## Design

Both series come from the **same venue** (Bitfinex, via
[Zombie-3000/Bitfinex-historical-data](https://github.com/Zombie-3000/Bitfinex-historical-data))
over the **same window**, so period and venue are held constant and only
the asset varies. BTC is the control: the strategy is known to work on
BTC elsewhere, so if the pipeline is sound it should behave sensibly here
too.

- Window: **2016-03-09 → 2019-12-31**, 376,878 5m bars each
- Rebuild with `python scripts/build_bitfinex_dataset.py --source <dir>`
- Covers the 2017 bull and the 2018 bear (BTC −84%, ETH −94%)
- It does **not** cover 2020–2026; that data is not reachable from here

## Result

$1,000 start, 0.10% spot / 0.05% futures fees, no funding.

### Spot (1x)

| asset | buy & hold | `kelly_regime_v4` | ratio | DD (v4) | DD (hold) |
|---|---|---|---|---|---|
| BTC *(control)* | $17,477 | $10,174 | **0.58x** | **40.1%** | 83.8% |
| ETH *(test)* | $11,550 | $5,482 | **0.47x** | **36.5%** | 94.2% |

### Futures (5x)

| asset | buy & hold | `kelly_regime_v4` | ratio | DD (v4) | DD (hold) |
|---|---|---|---|---|---|
| BTC *(control)* | $83,264 | $21,536 | 0.26x | **32.1%** | 85.2% |
| ETH *(test)* | **$18** (liquidated) | $4,263 | **236x** | **35.1%** | 99.3% |

## What it says

**The risk property transfers; the return property does not exist.** In
all four cells the strategy roughly halves-to-thirds the drawdown — BTC
83.8%→40.1%, ETH 94.2%→36.5%, and on leverage 85.2%→32.1% and
99.3%→35.1%. That is the same finding the BTC-only work reached from a
completely different direction, now replicated on a second asset. It is
the strongest evidence in this project that the mechanism is real rather
than fitted.

**On return it loses to holding on both assets on spot**, 0.58x and
0.47x. Consistent with everything else here: there is no return alpha,
on either asset.

**The one cell where it wins enormously is the one where holding died.**
Leveraged ETH buy-and-hold was liquidated to $18 in the 2018 bear; the
strategy finished at $4,263. That is not a 236x edge, it is the
difference between surviving and not — the same claim as the BTC stress
test (holding liquidated in 26 of 40 windows), reproduced on a second
asset.

**And the control behaves as it should.** Leveraged BTC holding *survived*
this particular window and beat the strategy 0.26x, because a position
opened in early 2016 had multiplied enough before the 2018 bear that a
84% fall no longer reached its liquidation price. Same strategy, same
period, different asset, opposite outcome — which is exactly how much a
single path is worth, and why the ETH cell should not be quoted as a 236x
edge either.

## Verdict

The sample-size objection is **partly answered**. The drawdown reduction
is not BTC-specific, which was the thing most at risk of being an
artifact. The absence of return alpha is also not BTC-specific.

What remains unanswered: this window shares the 2018 bear with the main
dataset, so the two tests are not fully independent, and 2020–2026 ETH
was not reachable. A second bear on a second asset in a *different*
period is still the missing experiment.
