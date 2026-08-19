# B-05 (conservative): funding-decile gate on `kelly_regime_v4`

Backlog item **B-05**, conservative variant. Files: `funding_gate_conservative.py`
(strategy, unregistered), `run_funding_gate_conservative.py` (harness). A
second, independent branch explores a continuous/analytically-derived
treatment of the same idea in separate files (`funding_crowding_novel*.py`,
not read here). Per ROUTINE.md's parallel-round rules, this branch's
configuration count is 6 and must be added to the other branch's count at
the program level, not reported alone.

**Scope boundary respected throughout**: every evaluation in this file is
restricted to `2020-01-01..2022-12-31` (funding-covered inner-train +
inner-validation). Nothing on or after `2023-01-01` was run, printed, or
inspected — `run_funding_gate_conservative.py` hard-codes and enforces
this in `_assert_in_bounds()`.

## Mechanism (one sentence)

Whenever the current bar's causal funding rate sits at or above the 90th
(or 95th) percentile of its own trailing distribution — a rolling,
strictly-backward-looking rank, never a whole-series quantile — force
`kelly_regime_v4`'s target exposure to exactly 0.0 for that bar, on the
theory (R-16, and Cardaliaguet & Lehalle 2018's mean-field crowding view)
that a richly-positive funding rate is a *directly observed* measurement
of crowded longs and therefore a bad moment to be adding to one, layered
on top of — not replacing — the base strategy's price-inferred regime
vote.

## Pre-registration (written before the sweep below was run)

**Selection rule**: of the 6 `(lookback_days, threshold)` configs, freeze
the one with the best **inner-train Sharpe on futures (5x)**. Applied
mechanically to whatever the sweep printed — see `select_frozen()` in the
harness, called immediately after `sweep()` with no intervening
eyeballing step.

**Falsification test**: at the frozen config, re-run inner-train on spot
at the Bitstamp entry taker tier (0.40%, vs 0.10% everywhere else in this
file) alongside the unmodified `kelly_regime_v4` baseline at the same fee.
The gate adds trades relative to the baseline (extra flatten/re-enter
events whenever funding spikes) purely to avoid a funding cost that is
free on spot to begin with — so if that turnover cost is not smaller than
the crowding signal is worth, the gate's advantage should shrink or
reverse at the higher tier. This is chosen specifically because it can
fail the idea for a boring reason (turnover) rather than a deep one
(the signal not existing), which is exactly the failure mode R-12/R-13
established as this project's most common trap.

Both paragraphs above are verbatim from the harness's own module
docstring/comments, committed before `sweep()` was ever invoked.

## Step 3 — inner-train sweep (2020-01-01 .. 2021-12-31), $1,000 start

**Configurations evaluated: 6** (`lookback_days ∈ {30, 90, 180}` ×
`threshold ∈ {0.90, 0.95}`), each run on both markets = 12 backtests, plus
2 baseline backtests = **14 distinct backtests** for this step.

| lookback | threshold | market | final | Δ vs baseline | Sharpe | Δ Sharpe | DD | trades |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| — | — | spot baseline | $4,336 | — | 1.99 | — | 24.3% | 40 |
| — | — | futures baseline | $5,502 | — | 2.22 | — | 21.5% | 40 |
| 30 | 0.90 | spot | $4,243 | −2.1% | 2.27 | +0.28 | 21.5% | 143 |
| 30 | 0.90 | futures | $5,289 | −3.9% | 2.49 | +0.27 | 22.7% | 143 |
| 30 | 0.95 | spot | $4,162 | −4.0% | 2.14 | +0.15 | 24.2% | 122 |
| 30 | 0.95 | futures | $5,611 | +2.0% | 2.47 | +0.25 | 22.9% | 122 |
| 90 | 0.90 | spot | $3,390 | −21.8% | 2.06 | +0.07 | 23.2% | 137 |
| 90 | 0.90 | futures | $4,513 | −18.0% | 2.40 | +0.18 | 21.3% | 137 |
| 90 | 0.95 | spot | $4,042 | −6.8% | 2.16 | +0.17 | 26.4% | 119 |
| 90 | 0.95 | futures | $5,152 | −6.4% | 2.42 | +0.20 | 26.0% | 119 |
| 180 | 0.90 | spot | $3,644 | −16.0% | 2.16 | +0.17 | 18.7% | 130 |
| 180 | 0.90 | futures | $4,885 | −11.2% | 2.49 | +0.27 | 19.3% | 130 |
| **180** | **0.95** | **spot** | **$4,442** | **+2.4%** | **2.29** | **+0.30** | **22.8%** | **104** |
| **180** | **0.95** | **futures** | **$5,926** | **+7.7%** | **2.60** | **+0.38** | **21.3%** | **104** |

Every one of the 6 configs improves inner-train Sharpe on futures (+0.07 to
+0.38, all inside R-20's ±0.2 noise floor for 4 of 6 cells — only the
30d/0.90 and 180d/0.95 cells clear it), so this reads as a genuine, if
mild, plateau rather than a single spike — but on **final balance** the
picture is mixed: 4 of 6 configs finish *below* the baseline on spot, and
`lookback=90` is uniformly the worst cell on both markets. The
higher-Sharpe, lower-drawdown effect and the lower-final-balance effect
are the same mechanism seen from two sides: the gate trades a slice of
gross return for a smoother path by standing aside during funding spikes
that (in this bull-heavy inner-train window) often preceded further
upside.

**Selection rule applied**: `lookback_days=180, threshold=0.95` (futures
Sharpe 2.60, the sweep's maximum) — **frozen**.

## Step 4 — inner-validation (2022-01-01 .. 2022-12-31), frozen config

4 backtests (baseline + gate, both markets).

| | market | final | Δ vs baseline | Sharpe | DD | trades |
|---|---|---:|---:|---:|---:|---:|
| baseline `kelly_regime_v4` | spot | $766 | — | −1.21 | 28.2% | 26 |
| gate (frozen) | spot | $767 | +0.1% | −1.21 | 28.0% | 26 |
| baseline `kelly_regime_v4` | futures | $741 | — | −1.36 | 30.6% | 26 |
| gate (frozen) | futures | $746 | +0.7% | −1.36 | 30.2% | 26 |

**The inner-train edge does not replicate.** 2022 is a bear year in which
`kelly_regime_v4`'s own regime vote is already flat for most of the
period, so there is little exposure left for the funding gate to remove —
the two strategies trade the same 26 episodes and finish within 1% of
each other on both markets, with identical Sharpe to two decimals. This
is not "the gate hurts" and not "the gate helps"; it is "the gate has
almost nothing to do" in the one regime where the base strategy is
already doing what the gate would ask of it anyway.

## Step 5 — falsification: Bitstamp 0.40% taker, inner-train, spot

| | fee | final | Sharpe | DD | trades |
|---|---:|---:|---:|---:|---:|
| baseline | 0.10% | $4,336 | 1.99 | 24.3% | 40 |
| gate (frozen) | 0.10% | $4,442 | 2.29 | 22.8% | 104 |
| baseline | 0.40% | $3,631 | 1.78 | 27.9% | 40 |
| gate (frozen) | 0.40% | $2,842 | 1.65 | 26.7% | 104 |

Gate advantage over baseline: **+2.4% at 0.10%, −21.7% at 0.40%.**

**FALSIFIED, as pre-registered.** The gate trades 104 times against the
baseline's 40 (2.6x the turnover — most of it is the extra flatten/
re-enter cycle around funding spikes) purely to avoid a cost that costs
nothing on spot. At Bitstamp's real entry tier that turnover is expensive
enough to flip the already-small edge negative. This is exactly R-13's
general finding ("the spot edge lives entirely inside the 0.10%-0.40%
fee margin") reproducing itself on a new strategy, and exactly the
failure mode this test was chosen to catch.

## Step 6 — real funding cost, combined inner period (2020-01-01..2022-12-31), futures 5x

4 backtests (2 strategies × {funding-free, funding-charged}).

| strategy | funding-free | with funding | cost (%) | funding paid |
|---|---:|---:|---:|---:|
| baseline `kelly_regime_v4` | $4,218 | $3,060 | −27% | $948 |
| gate (frozen) | $4,543 | $3,615 | −20% | $763 |

**The mechanism's actual point holds**: the gate pays $763 in real funding
against the baseline's $948 — a **19.5% reduction in the dollar cost**,
and a milder drag on the balance (−20% vs −27%). It is a genuine,
non-trivial effect, consistent with R-14 (funding runs richest exactly
when a long-biased strategy wants to be long) and R-16 (funding predicts
its own reversal). But it is a cost reduction on an already-small
baseline number, not a return edge — see steps 4 and 5.

## Step 7 — causality probe (R-28-style, by hand)

Two-opposite-tampers procedure on the frozen config: bars from position
195,000 of a 200,000-bar tail onward were multiplied by 3 (prices,
volume, **and** the funding column) in one copy and divided by 3 in
another. Checked at offsets {1, 2, 3, 5, 10, 20, 100, 1,000} bars before
the cut:

```
PASS - every order decision at or before the cut is unchanged
  column target         max |difference| before the cut = 0.000e+00  PASS
  column funding_pct    max |difference| before the cut = 0.000e+00  PASS
  column funding_gate   max |difference| before the cut = 0.000e+00  PASS
```

**PASS.** No lookahead in either the inherited `kelly_regime_v4` sizing
or the funding-percentile gate; the rolling `.rank(pct=True)` was also
verified in isolation (a synthetic series, confirming a later value never
moves an earlier row's rank — see harness derivation, not committed as a
separate test since this is an unregistered experiment).

## Bookkeeping

- **Configurations evaluated (step 3, for deflated Sharpe): 6.**
- **Total distinct `(strategy, market, period, fee-tier)` backtests
  actually run: 26** — 14 (sweep: 2 baseline + 12 gate) + 4
  (inner-validation: 2 baseline + 2 gate) + 4 (fee-tier falsification: 2
  strategies × 2 fee tiers) + 4 (funding-cost: 2 strategies × 2 funding
  settings) = 26. The causality probe is not a backtest (no broker P&L
  compared) and is not counted in this total.
- This branch's 6 configurations must be summed with the parallel
  branch's count at the program level, per ROUTINE.md's parallelism
  rules — this file does not know that count and does not attempt to
  state a combined total.
- No bar at or after 2023-01-01 was read by any command in this file.
  The holdout counter is unaffected by this branch.

## Verdict: NEGATIVE

The gate does what it says on the label — it measurably reduces the real
dollar funding cost paid (step 6, a genuine and moderately-sized effect)
— but that reduction never converts into a return or risk-adjusted edge
that survives contact with either a second period or real costs:

- The inner-train Sharpe improvement, real as it is, is a *plateau at the
  edge of the noise floor* (2 of 6 cells clear ±0.2, 4 do not), and 4 of 6
  configs are worse than baseline on raw final balance — this was already
  a soft signal before validation.
- It **does not replicate on inner-validation**: baseline and gate finish
  within 1% of each other on both markets in 2022, because the base
  strategy's own regime vote is already flat through most of the bear
  year the gate would otherwise act in. There is no regime left for a
  funding overlay to improve in the one inner-validation year available.
- It **fails its own pre-registered falsification test**: the extra
  turnover the gate requires (2.6x the baseline's trade count) is not
  free, and at Bitstamp's real 0.40% entry tier the already-marginal
  edge reverses to −21.7%.

None of the three legs of the promotion bar (beats baseline
out-of-sample beyond the noise floor; survives its falsification test;
plateau, not peak) clears on this evidence. This is not worth spending
the project's 2023+ holdout on as it stands. The one thing worth keeping
from this branch: the funding-cost-reduction number in step 6 is real and
could motivate a *lower-turnover* version of the same idea (e.g. gating
only when funding is both high **and** the base vote is already close to
its deadband edge, so the override rarely fires a trade it wasn't already
close to making) — but that is a new, untested variant, not a promotion
of this one.
