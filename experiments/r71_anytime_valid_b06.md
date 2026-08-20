# R-71 pre-registration — anytime-valid inference for B-06's growing forward record

**Status: methodology contribution, not a mechanism round.** This is not a
new strategy variant and spends **0** holdout consultations (see "Holdout
accounting" below — this is the point that matters most and is stated
precisely, not glossed). It adds one tool to `tradebot.inference` so that a
**future** round can read B-06's live paper-trading record (`reports/
paper_trading/*.csv`, `scripts/paper_trade.py`) at an arbitrary, unplanned
time — and again the next day, and the day after that — without the
repeated-testing inflation that checking a fixed-sample-size test over and
over would cause. This file is that future round's pre-registration: what
question the tool answers, what it assumes, what it costs, and how to run
it. It does not itself run a verdict on `kelly_regime_v4` vs
`buy_and_hold` — B-06 has one calendar day of data as of 2026-08-20, and any
verdict today would be exactly the "underpowered peek" this construction
is built to make statistically honest rather than statistically dangerous.

## Direction

Filed off-backlog, per R-67/R-68/R-69/R-70's own converging conclusion
(`docs/LEDGER.md`) that the SIZE/COST axis on the committed 2017-2026
dataset is exhausted at ~627 program-level holdout consultations — "no
mechanism can narrow an interval — only more data, more breadth, or
forward evidence can" (R-67). The ledger's standing recommendation is
**B-06**: let the paper-trading recorder accumulate real forward evidence,
since it is the one source this dataset has not already spent. This round
is the missing piece that recommendation needed: B-06's record will be
read *repeatedly*, at *no fixed sample size decided in advance* — daily,
by whoever next asks "is there a difference yet" — and `docs/ROUTINE.md`'s
own discipline is unambiguous that repeated, undisciplined peeking at
growing data is the classical multiple-testing trap. Attacks **ERR** (no
error control in the signal path) as directly as R-70's Ledoit-Wolf work
did, applied to the one dataset axis (forward, not backtest) this project
has left. Not a duplicate of R-70 (`ledoit_wolf_sharpe_diff`,
`bootstrap_studentized_sharpe_diff`): those are fixed-`n` tests, correct
for a frozen backtest window, wrong tool for a series that grows one row
per candle with no pre-committed stopping rule.

## What this tool is

`tradebot.inference.empirical_bernstein_confidence_sequence(diffs, bound,
alpha=0.05, c=0.75) -> pd.DataFrame` — Waudby-Smith & Ramdas (2024, JRSSB
86(1), pp. 1-27; arXiv:2010.09686), Theorem 2, the closed-form
"predictable plug-in empirical Bernstein" (PrPl-EB) confidence sequence.
Companion helper: `anytime_valid_first_exclusion(cs) -> int | None`, the
first `n` at which the sequence excludes zero (or `None`).

**The literature it rests on, cited the way this project's own docs
require (author, year, venue):**

- Waudby-Smith, I. & Ramdas, A. (2024). "Estimating means of bounded
  random variables by betting." *Journal of the Royal Statistical Society
  Series B*, 86(1), 1-27. arXiv:2010.09686. Theorem 2 is the exact
  construction implemented — a closed-form, variance-adaptive
  empirical-Bernstein confidence sequence, chosen over the same paper's
  **betting** confidence sequence (their headline result, and tighter in
  their own experiments) because the betting CS requires an online
  mixture/ONS-style adaptive bet that is easy to get subtly wrong without
  a calibration test to catch it, and the paper's own Figure 2 shows the
  closed-form PrPl-EB tracks their conjugate-mixture empirical-Bernstein
  CS closely in practice. Simpler and independently checkable beat
  marginally tighter for a project whose whole discipline is "don't trust
  a number you can't re-derive."
- Howard, S., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021). "Time-uniform,
  nonparametric, nonasymptotic confidence sequences." *Annals of
  Statistics*, 49(2), 1055-1080. The general uniform-boundary /
  test-supermartingale framework Waudby-Smith & Ramdas build on, and the
  source (their own Section A.8) of the "swap the variance term's target
  mean for the *predictable, previous-step* mean estimate" trick that
  makes Theorem 2 closed-form instead of needing root-finding at every
  step.
- Wang, Q., Wang, R. & Ziegel, J. (2022). "E-backtesting." arXiv:2209.00991.
  Background, not a formula source: this is the literature on using
  e-values/e-processes to sequentially backtest risk-measure and trading
  claims without fixing a sample size in advance, and it is the reason
  "anytime-valid" rather than "one more fixed-`n` test" is the right frame
  for a live, growing paper-trading record at all.

## The null hypothesis this tests, precisely

For a chosen pair of paper-traded arms (e.g. `kelly_regime_v4` vs
`buy_and_hold`, both on Bitstamp spot, from `reports/paper_trading/`),
build the paired **daily log-return difference** series
`d_t = r_kelly,t - r_bh,t` using this module's own `daily_returns()`
convention (last equity print of each UTC day; the module's established
daily-not-5m-bar unit). The null is

    H0: E[d_t] = 0 for all t   (no persistent difference between the arms)

against the two-sided alternative `E[d_t] != 0`. The output is a running
`(lower, upper)` interval, one row per day, such that
`P(0 not in [lower_t, upper_t] for some t) <= alpha` — **simultaneously
over every `t`**, not just the `t` a reader happens to stop at. Reading
the sequence today, and again next week, and stopping the first day it
excludes zero, is therefore a valid level-`alpha` test of H0. Doing the
exact same "check today, check again next week, stop at the first win"
with any of this module's *other* tools (`paired_bootstrap`,
`ledoit_wolf_sharpe_diff`, `bootstrap_studentized_sharpe_diff`) is **not**
valid — those guarantee coverage only at one sample size fixed in advance,
and repeated peeking inflates their true false-positive rate well above
the nominal `alpha`, which is precisely the failure mode this tool exists
to avoid.

## The bound: what it is, why this value, and its cost

WSR's construction needs the observations almost-surely bounded (it
rescales to `[0, 1]`). A paired daily log-return difference between two
Bitstamp paper-trading accounts is not literally bounded the way a coin
flip is, so a bound has to be chosen and its cost stated — not silently
assumed:

- **Both arms are spot, unleveraged, no-short.** `scripts/paper_trade.py`
  constructs both accounts via `MarketSpec.spot(fee_rate=...)`
  (`leverage=1.0`, `allow_short=False`, confirmed by reading
  `src/tradebot/broker.py`), so each account's target exposure is clamped
  to `[0, 1]` of equity notional in BTC. A 0%-exposure day earns
  approximately the (small, fee-only) cash drift; a 100%-exposure day
  earns approximately that day's own BTCUSD return, minus fees. The paired
  difference between two such accounts on the same day is therefore
  bounded, to a good approximation, by the magnitude of **that day's own
  BTCUSD log return** (an exposure-0 vs exposure-1 day is the worst case;
  any exposure mix in between is smaller in magnitude).
- **Chosen bound: `bound=0.5`** (a 50-percentage-point daily log-return
  difference). This is a deliberately round, generous number: modern-era
  (post-2017) BTCUSD daily closes essentially never move ±50% in a single
  UTC day even on the worst recorded crash days (e.g. mid-March 2020's
  multi-day crash was itself spread over several sessions, each smaller
  than this), so the bound is not expected to bind in practice, and the
  fee/leverage structure above means the *typical* daily paired difference
  is a few percent at most, not tens of percent.
- **The cost of clipping, stated precisely (per this tool's own
  docstring):** the confidence sequence returned is a valid `(1-alpha)`
  sequence for the mean of the **clipped** variable
  `clip(d_t, -0.5, 0.5)`, not automatically for the raw, unclipped mean.
  If no observed day ever reaches `±0.5`, the two means are identical and
  the distinction is moot — check the `clipped` column the function
  returns; it flags exactly which days bit. If a genuine black-swan day
  ever *did* produce a paired difference beyond `±0.5`, the reported
  interval would still be valid, but valid for a mean that has been pulled
  toward zero relative to the true one — a conservative distortion of the
  claim, not a broken guarantee, and the honest thing to do on such a day
  is to re-run with a larger bound and report both.
- **This is an engineering bound (option (a) in the task brief), not a
  literature-derived sub-Gaussian/bounded-difference analogue (option
  (b)).** A rigorous derivation of the true tail behaviour of a
  fee-charged, target-clamped spot-account return difference (accounting
  for the exact fee schedule and rebalancing rule) was judged not worth
  the additional complexity given how conservative `bound=0.5` already is
  relative to any plausible daily move; a future round that wants a
  tighter, principled bound could pursue Howard et al. (2021)'s
  sub-exponential/sub-Gamma boundary families instead, at the cost of one
  more parameter (the sub-exponential scale) to justify.

## Holdout accounting — the distinction that must not be glossed

**This tool touches zero rows of the committed 2017-2026 backtest
dataset, ever, in any use this pre-registration recommends, and increments
the "Holdout consultations to date" counter in `docs/LEDGER.md` by exactly
0.** The only inputs it is meant to be pointed at are the live paper-trading
CSVs under `reports/paper_trading/` — a dataset that:

- did not exist before `scripts/paper_trade.py` started running (B-06,
  first committed by an earlier round),
- grows only by one row per closed Bitstamp candle going forward, never by
  re-reading a historical file,
- is causally incapable of leaking information about the committed
  backtest dataset's 2023+ holdout, because every row postdates the
  session that recorded it and none of them derive from
  `simulate_portfolio` or any of the OHLCV parquet files the backtest
  engine reads.

Reading this confidence sequence — today, tomorrow, every day, as many
times as anyone wants — is therefore **not** a "holdout consultation" in
this project's sense and must **not** be added to the running total at the
top of `docs/LEDGER.md`'s "Holdout consultations to date" list. That
counter tracks consultations of the *committed, fixed, 2017-2026 dataset*
specifically because re-reading a fixed, finite dataset is what deflates a
Sharpe claim (Bailey & López de Prado 2014) — the mechanism is "the same
data was checked N times, so correct for N." A live feed that adds a
genuinely new, previously-unobserved row between any two reads is a
different object entirely: this is exactly the "more data" R-67 named as
the one thing that *can* still narrow an interval, and the whole reason
B-06 was recommended in the first place. A future round applying this
tool to B-06 should say so explicitly in its ledger entry — "N reads of
`empirical_bernstein_confidence_sequence` against the B-06 CSVs, 0 holdout
consultations" — rather than leaving the distinction implicit.

## Illustrative application to the real B-06 CSVs today (2026-08-20)

**This is a demonstration of the tool working end-to-end, not a verdict.**
As of this writing, `reports/paper_trading/kelly_regime_v4_bitstamp.csv`
and `buy_and_hold_bitstamp.csv` hold 8 decision rows each, spanning
2026-08-19 23:05 UTC to 2026-08-20 21:50 UTC — under two calendar days,
and both accounts' `new_target` converged to the same ~1.0x exposure
within the first two decisions (the `MarketSpec.spot()` leverage=1.0 cap
binds on `kelly_regime_v4`'s own inception-catchup target of 1.545), so
their recorded `equity_after` paths are, so far, identical. Resampling to
`daily_returns()` (the module's own convention) yields exactly **one**
completed calendar day (2026-08-20 00:00 UTC) for each arm, both equal to
+0.049555, giving a single paired difference of **0.0**:

```python
import pandas as pd
from tradebot.inference import (
    daily_returns, empirical_bernstein_confidence_sequence,
    anytime_valid_first_exclusion,
)

kelly = pd.read_csv("reports/paper_trading/kelly_regime_v4_bitstamp.csv",
                     parse_dates=["timestamp"]).set_index("timestamp")["equity_after"]
bh = pd.read_csv("reports/paper_trading/buy_and_hold_bitstamp.csv",
                  parse_dates=["timestamp"]).set_index("timestamp")["equity_after"]

diffs = (daily_returns(kelly) - daily_returns(bh)).dropna()
cs = empirical_bernstein_confidence_sequence(diffs, bound=0.5, alpha=0.05)
print(cs)
print("first exclusion:", anytime_valid_first_exclusion(cs))
```

Output, run 2026-08-20:

```
                           n  mean  lower  upper  excludes_zero  clipped
timestamp
2026-08-20 00:00:00+00:00  1   0.0   -0.5    0.5          False    False
first exclusion: None
```

**Read this exactly as it looks: completely uninformative, as it should
be at n=1.** The interval spans the entire rescaled range because a single
observation carries almost no information under an empirical-Bernstein
construction (the running variance estimate starts at its maximum,
`sigma_hat_0^2 = 0.25`) — this is the correct, honest behaviour, not a
bug. Nothing here says anything about whether `kelly_regime_v4` differs
from `buy_and_hold` going forward; it only confirms the tool runs
end-to-end against the real CSV schema and produces a sane, appropriately
wide answer at the current, tiny sample size. **Re-run this exact snippet
on a future date, against the then-current CSVs, to get a real reading —
that reading is valid at whatever calendar day it happens to be read,
precisely because this is what "anytime-valid" means.**

## Calibration (synthetic, before touching any real B-06 data)

Per `docs/ROUTINE.md`'s own two-gate convention (R-46/R-59's
synthetic-calibration gates, R-70's dual-estimator check): calibrated on
synthetic data with a **known, exactly-zero true mean difference**, across
many independent repeated runs, checking the **time-uniform** false
rejection rate — `P(the sequence ever excludes zero, at any n across the
whole run)` — not fixed-`n` coverage, which would not distinguish an
anytime-valid method from an invalid one. Three settings, 500 independent
runs of `n=1,000` observations each, `alpha=0.05`, `bound=0.5`:

| setting | description | false-rejection rate (500 runs) |
|---|---|---|
| i.i.d. normal | `N(0, 0.02)` | **0.000** |
| AR(1), phi=0.6 | mean-zero, same marginal variance | **0.004** |
| fat-tailed | Student-t, df=3, rescaled to sd=0.02 | **0.000** |

All three sit comfortably at or below the nominal `alpha=0.05` — the
construction is conservative in every setting tested, never anywhere near
exceeding it. (Reproduced in
`tests/test_inference.py::test_empirical_bernstein_cs_time_uniform_false_rejection_rate`,
which asserts a looser `<= 2*alpha` bound to absorb Monte Carlo noise
without weakening the property checked.)

**Honest limitation on the AR(1) setting:** WSR's supermartingale argument
formally needs each observation's *conditional* mean (given the past) to
equal the true mean `mu` — an i.i.d. or martingale-difference assumption.
An AR(1) series with `phi != 0` has a conditional mean that depends on the
previous observation and only equals the unconditional mean `mu=0` on
average, not pointwise; this is not the exact hypothesis class the theorem
covers, even though the unconditional mean is correctly zero. The
calibration run above found the construction still conservative under a
plausible `phi=0.6` AR(1) case — this is evidence of practical robustness
in one specific dependence structure, **not a proof that arbitrary serial
dependence is safe**. A future round applying this to B-06's real record,
where daily strategy-difference autocorrelation is plausible (this
module's own bootstrap tools use a 30-day mean block for exactly this
reason), should treat this as a noted-but-not-fully-closed assumption, the
same way R-70 treated its own two disagreeing HAC/bootstrap estimators as
an honest open question rather than a discrepancy to paper over.

## What would make this tool wrong, named in advance

- If a future round's synthetic calibration check (re-run periodically as
  B-06's own data characteristics become clearer — e.g. actual measured
  autocorrelation of the real daily difference series once there is
  enough of it to measure) finds the time-uniform false-rejection rate
  clearly exceeding `alpha` under a dependence structure resembling the
  real data, the construction should be treated as miscalibrated for this
  use and not trusted until re-derived for that dependence structure.
- If `bound=0.5` is ever observed to bind on real B-06 data (the `clipped`
  column reports `True`), the reported interval is for the clipped mean,
  not the raw one, and should be re-run at a larger bound before being
  read as a verdict.

## Next step for whoever reads B-06 next

Nothing about this pre-registration commits a future round to a decision
rule or a promotion bar — that is out of scope for a piece of
infrastructure, and would be premature at B-06's current size regardless.
What it fixes in advance is *how* to ask the question honestly: build the
paired daily-return-difference series between the two arms of interest
from `reports/paper_trading/`, call
`empirical_bernstein_confidence_sequence(diffs, bound=0.5)`, and read
`excludes_zero` / `anytime_valid_first_exclusion` — on any day, as many
times as wanted, without spending anything from the backtest holdout
counter and without needing to pre-commit to a sample size first.
