# Alternative directions, ranked by evidence

Where this project has actually got to, stated plainly: **no prediction
edge was found on 5-minute BTC bars**, and the one thing that worked —
regime-gated fractional-Kelly sizing — turns out to be worth about 1.33x
holding *gross* on spot, less than fees consume at any realistic tier,
and its futures headline shrinks 2-3x once funding is charged. The
strategies are a risk overlay on a levered long, not alpha.

Three structural facts constrain anything built next:

1. **The effective sample size is about 3, not a million.** The dataset
   has 1.01M bars, but the regime filter acts on multi-year cycles and
   has been tested against roughly three independent bear events (2018,
   2020, 2022). The bars are autocorrelated detail inside those three
   trials; the Monte Carlo windows resample the same events, so they
   measure path sensitivity, not out-of-sample validity.
2. **Turnover is fatal.** Four independent turnover-reduction mechanisms
   all produced in-sample winners and out-of-sample losers
   ([LIVE.md](LIVE.md#can-it-be-tuned-to-beat-the-fee-no-and-the-attempt-is-instructive)).
3. **The signal and its cost share a cause.** Funding runs +20%/yr while
   the strategy holds against +2.8% while flat, because the crowding the
   strategy detects is what sets the rate
   ([VALIDATION.md](VALIDATION.md#funding-the-cost-that-was-missing-and-what-it-does)).

Directions below are ordered by how much evidence already exists, not by
how exciting they are.

---

## 1. Harvest the funding premium instead of paying it

**The idea.** Fact 3 says this family is structurally short a large,
persistent premium. The obvious response is to take the other side: hold
spot and short the perpetual against it, delta-neutral, and collect
funding. Market direction cancels; the return is the funding stream.

**What the committed data says.** Compounding the real Binance BTCUSDT
funding series, 2020–2023:

| | |
|---|---|
| gross funding stream | **+82.0%** over 4.0 years = **+16.2%/yr** |
| after 0.10% taker on both legs, quarterly rebalance | +14.6%/yr |
| after 0.40% taker on both legs, quarterly rebalance | +9.8%/yr |
| settlements where the payer flips (negative rate) | 13.5% |
| worst 30-day run of the funding stream | **−1.31%** |

A −1.31% worst month against a +16%/yr carry is a risk profile nothing
else in this repo comes close to — `kelly_regime_v4` at its best has a
35% drawdown.

**Grounding.** This is the crypto cash-and-carry / funding-harvest trade,
with a real literature: He et al. (2024) derive the no-arbitrage relation
between perp price, spot price and funding rate, and empirical work
covering 2020–2025 reports a carry Sharpe around 6.45 driven mostly by
the funding rate itself.

**The reason to be careful, and it is a serious one.** That same work
reports the Sharpe falling to **4.06 from 2024 and turning negative in
2025** as the trade crowded. The committed data stops at **2023** —
precisely where the literature says the premium began to break. So this
project's measurement covers the good years and none of the bad ones,
which is exactly the mistake the rest of this repo exists to avoid.

**What the numbers above do *not* model:** basis risk at entry and exit
(the legs are assumed to converge), margin and liquidation risk on the
short perp leg even though the book is delta-neutral, exchange and
custody risk (the failure mode that actually destroyed carry desks in
2022), and borrow costs if the spot leg is financed.

**Next step if pursued:** extend the funding series through 2026 and
re-measure before anything else. If the premium really has inverted, this
direction closes; if it has merely compressed, it is still the best
risk-adjusted candidate here. That single data fetch decides it.

---

## 2. Funding as a positioning signal, not just a cost

**The idea.** Rich funding means crowded longs. Crowded positioning is a
classic reversal condition, and unlike anything else tried here the
signal is *not* derived from price — it is a direct observation of what
other participants are paying to hold their positions.

**What the data says.** Forward spot return by funding quintile,
2020–2023 (rank-binned; Binance clamps the rate, so the middle quintiles
share a value and should be read as one bucket):

| horizon | Q1 (cheapest) | Q5 (richest) | spread |
|---|---|---|---|
| 1 day | +0.60% | −0.10% | +0.70pp |
| 7 days | +3.02% | +0.30% | +2.72pp |
| 14 days | +4.13% | +0.56% | +3.57pp |

Controlling for momentum — mean 7-day forward return, funding tercile
against trailing-7-day-return tercile:

| | past low | past mid | past high |
|---|---|---|---|
| **funding low** | +2.83% | +1.74% | +2.16% |
| **funding mid** | +0.55% | +1.36% | +3.24% |
| **funding high** | **−1.68%** | **−1.54%** | +1.22% |

Correlation between funding and trailing return is only **0.39**, so this
is not simply a momentum proxy: high funding predicts *negative* forward
returns unless price is also rising strongly.

**Honest assessment.** Weaker than the raw quintile table suggests. The
middle quintiles are non-monotone (Q3 +3.06%, Q4 −1.02% at identical mean
rates) — that is noise from splitting a tied cluster, and it is a warning
about how much of the rest is noise too. Four years, one asset, and this
repo's own track record is that every apparent predictor died
out-of-sample. Treat it as a hypothesis to test properly, not a finding.

**Two ways to use it, cheapest first:** as a *gate* on the existing
strategy (stand flat when funding is in its top decile — low turnover,
directly targets the adverse timing in fact 3), or as a standalone
reversal signal (higher turnover, and fact 2 says that is where
strategies go to die).

---

## 3. Cross-sectional, to attack the sample-size problem

**The idea.** Every result here rests on ~3 regime events in one asset.
Trading BTC against ETH — or a small basket — turns one time series into
a cross-section, which multiplies independent observations without
needing more history, and is market-neutral so it does not depend on
crypto drifting up forever.

**Why it is worth doing even if it fails.** Running the *existing*
`kelly_regime` on ETH is a near-free falsification test. If the regime
filter reflects something real about crowd flow it should appear there
too. If it only works on BTC 2017–2026, fact 1 becomes fatal and a large
part of this repo's conclusions should be downgraded. The framework
already supports it — `tradebot fetch` plus a second data file — so this
is hours of work, not weeks.

**Do this before pursuing direction 1 or 2 seriously.** It is the
cheapest experiment with the highest information content about whether
any of this generalizes.

---

## Ruled out, with reasons

- **More indicators / more ML on 5m bars.** Twenty strategies, a second
  ML/DL research round, and every pure predictor lost to fees. The bar is
  no longer "does it show an edge" but "does it show an edge that
  survives 0.10% taker, walk-forward selection, and a ±0.2 Sharpe noise
  floor". Nothing tried has.
- **Tuning turnover to fit a fee tier.** Settled empirically: 28 of 32
  configurations beat holding in-sample, 0 of 28 out-of-sample.
- **Higher leverage as a fix for fees.** Fees are charged on notional, so
  leverage multiplies cost and return together. It changes the risk
  profile, not the sign.
- **Options / volatility risk premium.** Plausibly real, but there is no
  options data here and no way to validate it with this framework. Not
  ruled out on merit — ruled out on what can be checked.
