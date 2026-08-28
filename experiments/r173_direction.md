# R-173 direction — causal microstructure spread estimation as a COST-axis instrument (08-28)

**Written by the operator, based on a research sub-agent's proposal
(independently checked below), BEFORE either branch was dispatched. This
is the frozen Step 1/Step 2 pre-registration; neither branch may loosen
anything here, only add ablations (ROUTINE.md's "additions after the
freeze" rule).**

## Direction, one sentence

Two decades-old, purely price/range-based microstructure spread
estimators — Roll (1984, *Journal of Finance* 39(4), 1127-1139, the
implicit effective-spread estimator from the negative serial covariance
of price changes) and Corwin & Schultz (2012, *Journal of Finance* 67(2),
719-760, the high-low range estimator) — are computed causally from this
project's own OHLCV bars and used to test whether `kelly_regime_v4`'s
cost model, currently a **flat** fee tier plus optional flat slippage
(`src/tradebot/broker.py`), materially understates the true, *time-varying*
transaction friction the strategy actually pays, and whether trading
around that friction (rather than merely re-pricing it) improves net
economics.

## Step 1 — the four questions

**1. Which constraint does it attack?** **COST** — "costs scale *with*
the signal." The current cost model is a constant rate; this round tests
a friction measure that varies with market state, which is the COST
axis's own standing description applied literally to the transaction-cost
side (funding was the first cost this project found to move this way;
this is the second).

**2. Which ledger entries is it not a duplicate of?**
- Not L-14/L-15/L-16 (`camouflage_flow`/`stealth_trend`/`flow_regime`,
  BVC/VPIN order-flow-direction proxies): those infer trade **direction**
  as an INFO-axis trading signal. Roll/Corwin-Schultz estimate the
  **spread itself**, produce no directional signal, and never enter the
  vote/scale path — a COST-axis re-pricing/execution-timing object, not a
  ninth or tenth INFO signal.
- Not R-56/R-77 (patient-limit execution, B-24): those model fill
  *probability* against a resting limit from future high/low touches and
  scale urgency by a raw volatility feature. No spread is estimated;
  execution risk is modeled, not cost.
- Not R-64/65/67/68/165 (Gârleanu-Pedersen partial adjustment / rate
  smoothing): those assume a quadratic-cost functional form and *derive*
  an optimal trading rate; they never measure a cost from data.
- Not R-131/133/134/151 (turnover corridor/throttle, `HybridBroker`
  fixes): those cap trading rate against a fixed resource budget with no
  cost estimator.
- Not R-145 (funding-aware venue routing): funding cost by venue, not
  spread/slippage.
- Not the 08-26 verification pass (LEDGER.md, section E archive) that
  named Roll/Amihud/Kyle/VPIN/Almgren-Chriss together and dismissed all
  five in one sentence as needing order-book data: that pass never
  implemented anything (0 configurations, not an R-numbered round) and is
  simply wrong about two of the five names — Roll uses only `close`,
  Corwin-Schultz uses only `high`/`low`. `grep -c "Corwin\|Roll (1984)"
  experiments/*.py` returns 0 implementation hits before this round.
- `grep -n "Roll (1984)\|Corwin.*Schultz\|Corwin & Schultz"
  docs/LEDGER.md` confirms no prior round cites either paper.

**3. Is it simulable here?** Yes. Roll needs only `close`; Corwin-Schultz
needs only `high`/`low`. Both are already present in every committed
OHLCV file (BTC, ETH, the 6-instrument Coinbase panel), computed on
trailing causal windows, fully compatible with `prepare()`'s causal-column
convention and `tests/test_causality_strict.py`.

**4. What would make it fail, named now:**
- **Step-0 degeneracy gate** (measured below, real data, BEFORE either
  branch was dispatched): if the estimator is degenerate (near-always
  undefined/zero) at 5-minute resolution, or does not elevate during this
  project's own six canonical stress episodes, both branches close
  NEGATIVE at Step 0 with no backtest.
- **Falsification test (frozen now):** median of the six episode-window
  elevation ratios (mean estimated spread in a ±3-day window around each
  episode's onset, divided by the whole-pre-holdout-period unconditional
  daily median) must exceed **1.0**. This is a weak, symmetric bar chosen
  deliberately — a real liquidity-stress proxy should show *typical*
  elevation across six independent shocks, but need not spike on every
  single one (five of eleven prior N≈3-axis mechanisms in this ledger
  have partial hit rates on the same six-event calendar; a strict 6/6 bar
  would reject estimators that are still informative on average). The
  median (not the mean) is used specifically because the mean is fragile
  to one outlier episode inflating an otherwise-unremarkable estimator.
- **Conservative-branch failure, named in advance:** the estimated
  friction, applied as a re-pricing on top of the current fee tier, does
  not materially change `kelly_regime_v4`'s already-established OOS
  verdict (a coin flip against `buy_and_hold`, R-172's own restated
  finding) — i.e., COST was already priced correctly enough by the flat
  tier and this measurement adds a number without changing a conclusion.
- **Novel-branch failure, named in advance, because it is this ledger's
  own base rate on every COST-axis throttle tried to date** (R-56, R-77,
  R-131, R-133): the deadband-widening throttle suppresses exactly the
  trades that carry the edge, reproducing the same failure by a new
  mechanism rather than escaping it.

## Step-0 measurement (real BTC data, pre-holdout only, before either branch dispatched)

```
bars: 631,008 (2017-01-01 -> 2022-12-31, the full pre-holdout period)

Roll (1984), rolling 1-day (288-bar) covariance window:
  fraction of windows with negative Cov(dP_t, dP_{t-1}) (spread DEFINED): 76.3%

Corwin & Schultz (2012), raw 2-bar (10-minute) estimator:
  fraction of bars with a positive (informative, non-degenerate) estimate: 63.2%
  (36.8% negative -> clipped to 0 per the paper's own published convention)

Stress-episode elevation ratios (CS estimator, 1-day rolling mean,
+/-3-day window vs whole-period unconditional daily median = 0.000687):
  2018 bear onset (2018-01-17):        3.40x
  2018 bear bottom (2018-12-15):       0.90x
  2020-03 COVID crash (2020-03-12):    3.15x
  2021-11 top (2021-11-10):            0.72x
  2022-05 Terra/Luna (2022-05-09):     1.67x
  2022-11 FTX collapse (2022-11-08):   1.20x
  MEDIAN: 1.435x   ->  PASSES the frozen >1.0 gate (4 of 6 episodes individually
  elevated; 2018 bottom and the 2021-11 top are the two non-elevated cells,
  disclosed rather than dropped)
```

**Verdict: both estimators clear the Step-0 degeneracy gate and the
falsification test's median-elevation bar. Both branches are dispatched.**

## Step 2 — mechanism and the conservative/novel split

**Mechanism, one sentence:** replace this project's flat transaction-cost
assumption with a causally-estimated, time-varying friction measure, then
test both what it says about the cost `kelly_regime_v4` already pays
(conservative) and whether trading around it changes the strategy's own
behavior for the better (novel).

- **CONSERVATIVE — illiquidity-adjusted re-pricing (pure measurement, no
  behavior change).** Reuses this project's own existing fee-sensitivity
  methodology (`fee_at()`, the same primitive `scripts/fee_study.py` uses
  for the 0.40% tier) rather than inventing new engine machinery: sample
  the causal Corwin-Schultz spread at every bar where `kelly_regime_v4`
  actually re-targets (its own real deadband-triggered rebalances),
  volume-weight by `|Δtarget|`, and add the resulting average implied
  half-spread on top of the current flat taker fee to build an
  "illiquidity-adjusted" `MarketSpec`. Re-run the UNCHANGED, already-
  registered `kelly_regime_v4` under this adjusted tier and compare
  against its own current 0.10%/0.05% and 0.40% (already-registered)
  readings. Zero strategy-logic changes; this is a cost audit, not a new
  strategy — its "promotion" question is whether it changes the existing
  README/LEDGER cost caveats, not whether a new strategy is registered.
- **NOVEL — spread-conditioned deadband widening (behavior change).**
  `kelly_regime_v4`'s existing 10% re-balance deadband
  (`V4_DEADBAND`/`apply_deadband` in `experiments/r102_shared.py`)
  becomes dynamic: `deadband_t = V4_DEADBAND * (1 + k * pctile_t)`, where
  `pctile_t` is the causal Corwin-Schultz spread's own trailing percentile
  rank (0 to 1) and `k` is swept on the training period only. When
  estimated friction is elevated, a bigger vote/scale move is required
  before the strategy re-targets — deferring exactly the trades a real
  elevated spread would make more expensive, mirroring R-56/R-77's
  patient-execution shape but keyed on a genuine liquidity-cost estimate
  rather than a raw volatility feature or a fixed patience window. Fully
  expressible as a `build_target(df) -> np.ndarray` function inside the
  existing `TargetStrategy`/`compare()` harness (`experiments/r102_shared.py`)
  — no engine changes, evaluated against `v4_target` as the control on
  inner-train/inner-val/ETH exactly as R-92 through R-172 already do.

## Step 4 — pre-registered decision rules (frozen before any holdout read)

**CONSERVATIVE.** Outcome space partition:
- If the illiquidity-adjusted tier's OOS-analog reading (inner-val, then
  holdout once frozen) keeps `kelly_regime_v4` distinguishable from
  `buy_and_hold` in the same direction as the currently-registered
  0.10%/0.40% readings (i.e., still `≈`/coin-flip, or still beats, or
  still loses, matching the existing sign) → **NEGATIVE / no new
  caveat** — the flat-fee approximation was already adequate at the
  scale this strategy trades.
- If the adjusted tier flips the sign of the OOS comparison (v4 stops
  beating `buy_and_hold`, or an already-losing comparison becomes a
  clean loss with a materially wider margin) → **POSITIVE finding,
  filed as a new README/LEDGER cost caveat** (same status class as the
  funding warning), not a strategy promotion.

**NOVEL.** Standard promotion bar (ROUTINE.md, default REJECT): PROMOTE
only if, on the true 2023+ holdout, the deadband-widened variant (a) beats
`buy_and_hold` after real costs, (b) improves on `kelly_regime_v4` itself
by more than the ±0.2 Sharpe noise floor OR by a matched-risk drawdown/tail
improvement (R-33's matching rule applied — report exposure and realized
vol for every cell), (c) survives ETH replication, (d) the `k` neighbourhood
is a plateau, not an isolated peak. Anything short of all four is NEGATIVE.

**Holdout counter:** both branches read the true 2023+ holdout at most
once each, after this document is committed and the branches' own module
files are frozen. Increment `docs/LEDGER.md`'s holdout-consultation list
by however many distinct `ev(..., start=OOS_START)`-equivalent reads each
branch performs, summed, exactly as R-163 through R-172 already do.

## Configs evaluated by this file: 0 (pre-registration and the Step-0
measurement above only; each branch logs its own count, summed in the
ledger entry per R-163's convention).
