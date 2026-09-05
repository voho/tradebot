# R-190: ten execution variations of the accepted Kelly parents

Research checked on 2026-09-05, before this round's financial evaluations.
The decision rule, evaluation counts and outcomes belong to the R-190
pre-registration and ledger entry; this note records the mechanism and its
limits under [ROUTINE.md](ROUTINE.md).

The accepted parents are `kelly_regime_v4`, `kelly_regime_v3` and
`kelly_regime`: L-01, L-02 and L-04 are marked PROMOTED in the ledger.
Although `kelly_regime_v2` ranks third by the old comparison's final
balance, L-03 explicitly says NOT PROMOTED. R-189's ten games are all
RESEARCH ONLY. The old futures ranking excludes funding; v4's supported
finding is a BTC/ETH drawdown property, with its return increment inside
the noise floor and its matched-risk scope limited by R-57/R-62.

## Mechanism and ten candidates

The question is whether scheduling and measuring rebalances against actual
holdings improves the accepted parents' net performance and trading
cadence. This attacks COST. It adds no price information, independent
instruments or new regime observations.

Preserve each parent's existing causal target generation, parameters,
volatility logic and warmup. At fixed UTC four-hour decision times,
compare the executable target with `position * close / equity`; allow an
adjustment when the gap exceeds the specified equity-notional band.
The target is evaluated at the closed candle and an allowed order fills
at the next open. Ordinary price drift can therefore trigger a scheduled
rebalance even when the parent's target level has not changed. This is
the intended difference from its native change-triggered callback.

| Candidate | Target source | Execution band, equity notional |
|---|---|---|
| 1 | `kelly_regime_v4` | 0.05 |
| 2 | `kelly_regime_v4` | 0.10 |
| 3 | `kelly_regime_v4` | 0.20 |
| 4 | `kelly_regime_v3` | 0.05 |
| 5 | `kelly_regime_v3` | 0.10 |
| 6 | `kelly_regime_v3` | 0.20 |
| 7 | `kelly_regime` | 0.05 |
| 8 | `kelly_regime` | 0.10 |
| 9 | `kelly_regime` | 0.20 |
| 10 | Equal-weight mean of all three parent targets | 0.10 |

These are ten total variations, with a declared 0.05/0.10/0.20
neighbourhood for each parent. The common clock offers six decisions per
day; it does not guarantee six fills or a few completed round trips.
There is no forced exit solely to manufacture activity. The blend shares
the same instrument and closely related signals, so three parents do not
constitute three independent bets.

The broker's ordinary target-order band remains part of the experiment:
5% of maximum notional equals 0.25 equity notional at 5x leverage. It can
supersede a candidate's smaller band. Report actual fills, completed
episodes, traded notional and active calendar days separately, along with
target changes and suppressed requests when available. Neither the old
parent defaults nor broker mechanics should be silently changed to raise
the counts.

## What the primary sources establish

**Bell and Cover (1980), “Competitive Optimality of Logarithmic
Investment,” Mathematics of Operations Research 5(2), 161–166.** This is
the game-theory foundation inherited from the Kelly parents. In a
theoretical two-investor game with a known return distribution and any
number of stocks, the competitive policy combines the log-optimal
portfolio with a particular random wealth multiplier. There is no BTC
sample or empirical fee-tier result. These deterministic, estimated
trend/volatility rules do not implement that equilibrium exactly, and
the theorem does not guarantee their profits after fees.
[Publisher paper](https://pubsonline.informs.org/doi/10.1287/moor.5.2.161).

**Davis and Norman (1990), “Portfolio Selection with Transaction Costs,”
Mathematics of Operations Research 15(4), 676–713.** They analyze a bank
account and one risky stock following a lognormal diffusion, with
proportional transaction charges and an infinite-horizon consumption
objective. Their optimal policy has a no-trade region, characterized by a
free-boundary problem. This is a mathematical model, not an empirical
multimarket backtest. It motivates accepting some position drift to save
costs; it does not derive our fixed widths, four-hour clock or
trade-to-target destination.
[Publisher paper](https://pubsonline.informs.org/doi/10.1287/moor.15.4.676).

**de Lataillade and Chaouki (2020), “Equations and Shape of the Optimal
Band Strategy,” arXiv:2003.04646.** With a predictive signal, linear costs
and quadratic risk, they derive position-space no-trade boundaries and
analyze an Ornstein–Uhlenbeck predictor. Their numerical checks concern
the modeled predictor and theoretical bands; they do not establish
profitability on a historical BTC/ETH panel. The optimal boundaries
depend on signal dynamics and costs, and need not be symmetric. R-190
uses simple fixed symmetric thresholds as an empirical execution
neighbourhood, not as their analytical optimum.
[Author paper](https://arxiv.org/html/2003.04646v2).

These established sources justify testing a cost mechanism. The variation
round does not claim a new state-of-the-art algorithm or reuse a paper's
headline Sharpe as evidence for this dataset.

## Verified venue costs and simulation scope

The official Bitstamp spot schedule, valid from **1 September 2026**, was
checked in the rendered exchange page on 5 September. For standard
pairs, below $10,000 of trailing 30-day volume the taker fee is **0.40%**
and maker fee 0.30%. Standard taker tiers shown at higher volume are
0.30% above $10,000; 0.20% above $100,000; 0.18% above $500,000; 0.16%
above $1.5 million; 0.12% above $5 million; and 0.10% above $20 million.
Thus a constant 40bp test represents the entry tier; 10bp is the legacy
comparison convention, not the small-account entry tier. The schedule
also specifies a $10 minimum for USD-denominated orders.
[Official Bitstamp fee schedule](https://www.bitstamp.net/fee-schedule/).

The official Deribit English fee page, updated 17 August 2026, lists a
Standard perpetual/futures taker rate of **3.5bp** in a table explicitly
described as upcoming. The older official Spanish table still lists
**5bp** for BTC futures/perpetuals. The page does not settle that
effective-date discrepancy, so 5bp can be retained as a declared
conservative legacy cost scenario; it should not be represented as a
verified current account-specific rate. Funding must still be charged
separately.
[Current English schedule](https://support.deribit.com/hc/en-us/articles/25944746248989-Fees),
[older official Spanish schedule](https://support.deribit.com/hc/es/articles/25944746248989-Honorarios).

Fixed fee tiers across historical data are scenarios, not a reconstruction
of historical discounts or the account's evolving volume tier. A stated
1bp slippage assumption is not an observed spread. OHLCV and next-open
fills do not model maker queues, order-book liquidity, market impact,
venue order-size rounding or every exchange minimum. A surviving claim
would need those execution limits checked before promotion to deployment.

## Existing experiments this overlaps

This is explicitly an operator-requested composition and replication of
known mechanisms. The difference worth measuring is a shared execution
policy across all three accepted parents with the same width
neighbourhood and actual-account trigger.

| Previous work | Overlap and constraint on the new claim |
|---|---|
| R-37 | Target-volatility/leverage retuning and state-conditional Kelly failed. This round preserves those parent settings. |
| R-40, R-147 | Ladder bagging and reliability-weighted ensembles failed. The fixed three-parent blend has no learned weights or independent-breadth claim. |
| R-59, R-60, R-62 | Scale retunes and factor isolation locate the surviving signature in the vote. Do not call a lower exposure new volatility-timing alpha. |
| L-05/L-06, R-64, R-66 | EV bands, smoothing and trade-to-boundary already tested COST. Fixed actual-account bands are not a newly derived optimum. |
| R-89, R-90 | Asymmetric vote thresholds, response transforms and trailing stops failed. None is added here. |
| R-131, R-133 | Turnover corridors and dual throttles failed; splitting orders can raise fill counts. Count both fill and episode units. |
| R-186 | Phase averaging found only +0.029 Sharpe. Fixed UTC slots do not establish removal of timing luck. |
| R-189 | Four-hour scheduling and actual-account checks existed on council games. This round tests their transfer to unchanged accepted parents. |

The full prior outcomes are retained in [LEDGER.md](LEDGER.md).

## Reuse for risk matching and inference

The existing `experiments.matched_hold.ConstantExposureHold` is the
appropriate passive control: it measures actual holdings and uses
explicit quantities so the futures broker's broad target-order band does
not silently convert it to a static position. The proportional solver in
`experiments/run_matched_hold.py::solve_c` and its `windows` procedure
provide the established pattern for matching realized risk inside each
evaluation window. Solver runs must be counted. The driver loads the
default dataset at import, so an inner-stage harness should reuse the
narrow strategy class and solver pattern without importing that driver.

Use the shared daily-return functions in `tradebot.inference`:
`stationary_bootstrap_indices` for common 30-day-block resamples,
`paired_bootstrap` for paired growth and drawdown intervals, and the
existing studentized/HAC Sharpe-difference helpers for Sharpe inference.
Keep the first measured day's PnL relative to the fresh initial balance
explicit. Report achieved volatility and time in market for every arm;
declare a cell unmatched rather than scoring a tail advantage when it
misses the frozen tolerance. Simply scaling an already-costed daily
return series is an ex-post diagnostic, not a new executable control.

The anticipated failure is that drift rebalancing buys activity at a
larger fee bill, while scheduled decisions delay the parent's useful
regime changes. Another failure is an apparent drawdown improvement
explained entirely by reduced exposure. The frozen evaluation must be
able to reject both, including when no variation improves the parent.
