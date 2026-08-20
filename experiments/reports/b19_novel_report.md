# b19_risk_parity_rebalance — B-19 NOVEL branch (08-20)

Unregistered experiment. Code: `experiments/b19_risk_parity_rebalance.py`.
Not `@register`ed, not auto-discovered, nothing committed — a human
operator merges and commits after both B-19 branches report. This branch
does not touch `kelly_regime_covkelly*.py`, `kelly_regime_dual_fixed.py`,
`kelly_regime_v4.py`, `src/tradebot/multiasset.py`, or any other existing
`experiments/*.py` file; everything from those files used here is
imported unchanged. All evaluation below is restricted to inner-train
(2019-03-14 → 2020-12-31) / inner-validation (2021-01-01 → 2022-12-31)
plus one secondary 5x-futures pass on the same splits. **The 2023+
holdout was never read** (see verdict).

## Idea, mechanism, and why it is genuinely new (pre-registered before running)

**Idea, one sentence.** Weight the BTC and ETH legs of a periodically-
rebalanced `kelly_regime_v4` portfolio inversely to each leg's own
trailing realized volatility — no covariance matrix, no expected-return
estimate — and rebalance at cadences (quarterly, semiannual) this
backlog item has never tried, through an extension of R-50's continuous
(non-restarting) engine that also, for the first time on this backlog
item, charges an explicit transaction cost for the portfolio-level
rebalance itself.

**Constraint attacked.** SIZE (how much to hold in each leg — the one
axis this project's standing diagnosis says has ever worked), N≈3 (a
second asset raises the number of quasi-independent regime observations,
the axis R-42/R-43/R-50 opened), and COST (this file is the first round
on this item to price the rebalance's own turnover, not just each leg's
internal v4-driven trades).

**Not a duplicate of, cited precisely:**
- `kelly_regime_covkelly.py`/`_v2.py` (**R-42, R-43** — REJECTED). That
  allocator solves the closed-form Σ⁻¹μ Kelly weight, which needs a
  trailing EWM *mean-return* estimate per leg — the exact kind of
  estimation risk R-37/R-38/R-45 found too noisy at 5-minute-bar cadence
  to exploit, and the estimator R-43's own novel branch spent an entire
  round trying to de-noise without success. This file's weights use
  **only** each leg's own trailing volatility: no mean, no covariance
  matrix, no cross-asset term at all — per Maillard, Roncalli & Teiletche
  (2010), *The Properties of Equally Weighted Risk Contribution
  Portfolios*, Journal of Portfolio Management 36(4), 60-70 (SSRN
  1271972). That paper's own derivation shows the Equal Risk Contribution
  portfolio reduces exactly to inverse-volatility weighting when
  correlations are equal — automatic in the two-asset case, since there
  is only one pairwise correlation to be "equal" to. Practitioner
  precedent for the same idea: Qian, E. (2005), *Risk Parity Portfolios:
  Efficient Portfolios Through True Diversification*, PanAgora Asset
  Management working paper. This sits, by construction, strictly between
  R-50's static 50/50 (zero information used) and the rejected Σ⁻¹μ
  allocator (mean *and* covariance) — the specific "well-established
  middle ground" this branch was asked to test.
- R-50's own static-50/50-via-continuous-engine finding (the B-19 lead
  itself: ΔSharpe +0.79/+0.80 vs v4-solo, monthly/weekly, inner-
  validation only, max DD 33.2%→27.1%). That portfolio's weights are
  fixed at (0.5, 0.5) forever; this file's weights move every rebalance.
  **The static-50/50 number is re-derived independently in this file**
  (`run_portfolio_continuous_costed(..., "fixed5050", ...)`), not copied
  from R-50's printed number — see §2.
- R-50 (**B-18, R-50**) only ever tested monthly and weekly cadence. This
  file's cadence axis — monthly, quarterly, semiannual — is chosen from
  Dichtl, H., Drobetz, W., Wambach, M. (2016 journal publication;
  originally circulated 2012-2014), *Testing Rebalancing Strategies for
  Stock-Bond Portfolios Across Different Asset Allocations*, Applied
  Economics 48(9), 772-788 (SSRN 1927764 / 2479384): a double block
  bootstrap on US/UK/German stock-bond data found realistic-cost-optimal
  rebalancing frequency sits between quarterly and yearly — a genuinely
  untried region for this backlog item.

## A gap in R-50's engine, found and fixed here (not a criticism of R-50)

R-50's `run_continuous_full` reallocates pooled capital between legs by
an **algebraic rescale** of each leg's own independently-computed
continuous curve — by design, so the strategy is never re-invoked and
its latch state is never disturbed. That is exactly right for removing
the segment-restart artifact R-50 was built to fix. Its side effect:
moving dollars from the BTC sub-account into the ETH sub-account at a
rebalance boundary costs **nothing** in that engine — no
`fee_rate * traded_notional` is ever charged for the portfolio-level
reallocation itself, only for each leg's own v4-driven trades (correctly
preserved inside the rescale). Nobody in R-50 needed to notice this
because both of R-50's arms (fixed 50/50, dynamic Σ⁻¹μ) were compared to
each other on the same zero-rebalance-cost basis. For a round whose
second axis is explicitly "does rebalancing less often help once
realistic costs bite," reusing R-50's engine unmodified would make
cadence a free variable with no cost to trade off against — silently
erasing the COST axis this branch exists to test.

**Fix**, kept as close as possible to R-50's mechanism:
`run_portfolio_continuous_costed` in `b19_risk_parity_rebalance.py` is
R-50's engine, generalized to accept either weight mode (`fixed5050` or
`invvol`) and extended with one new step per rebalance boundary: the
dollar amount that must be sold from the overweight leg and bought into
the underweight leg to reach the new target weight (`shift`) is computed,
and `2 × rebalance_fee_rate × shift` (a full round trip — one taker fee
on each side) is deducted from pooled capital before the new segment
starts. `rebalance_fee_rate` defaults to `market.fee_rate` — the
portfolio rebalance is assumed to pay the same taker tier as each leg's
own trades, an explicit assumption, stated because this project has no
combined-order venue data to check it against. The underlying per-leg
continuous-curve mechanism, `_segment_returns`, `_segment_bounds`, and
`weight_at` are all imported **unchanged** from R-50's and R-42's files;
only the capital-combination step gains a cost. This cost is applied
**identically** to the fixed-50/50 reference, re-derived on the same
costed engine — a fixed split still needs occasional rebalancing back to
50/50 as prices diverge, and it should pay for that exactly as the
inverse-vol candidate pays for its own reallocations.

## A second bug found and fixed while building this: a fee-tier cache collision

R-50's `continuous_leg_equity` memoizes on
`(id(df), market.name, v4_kwargs, start, end, start_balance)`. It never
needed `market.fee_rate` in that key because R-50 always ran at one fixed
fee tier per process. This branch's F2 fee-tier stress test and the
(unused) `holdout()` path both call the **same** leg dataframe at two fee
tiers (0.10% and 0.40%) inside one process — `MarketSpec.spot(fee_rate=
0.001)` and `MarketSpec.spot(fee_rate=0.004)` both have `market.name ==
"spot"`, so the imported cache silently returned the **wrong** (already-
cached 0.10%-tier) curve on the second call. Caught by hand: an early
`feetier` run and the `all`-command's internal `feetier_check` call
printed two different 0.40%-tier balances for the identical
configuration — one correct cache-miss ($1,206), one silently wrong
cache-hit reusing the 0.10% curve ($1,431, indistinguishable from the
baseline number). Fixed with a locally, correctly-keyed cache
(`leg_equity`, includes `market.fee_rate`/`market.leverage` in the key)
built on the identical underlying call R-50's own function makes
(`run_period(KellyRegimeV4(**kwargs), df, ...)`) — the continuous-replay
mechanism is unchanged; only the memoization key, a pure performance
detail, is corrected. Flagging this for any future session that reuses
`continuous_leg_equity` across more than one fee tier in one process.

## Causality probe (unregistered strategy, no CI coverage)

Multiply/divide truncation-tamper procedure (the same convention R-42/
R-50 used), on the winning configuration (lookback=60d, quarterly), bars
after 2021-06-30 multiplied by K=137 in one copy / divided by K in the
other:

| check | max\|diff\| before cut | result |
|---|---|---|
| pooled equity (up-tampered) | 0.000e+00 | PASS |
| pooled equity (down-tampered) | 0.000e+00 | PASS |

No lookahead detected.

## Pre-registered falsification test and decision rule (written before any result existed)

**Falsification test — the candidate is rejected, holdout never read, if EITHER fires on the inner splits:**
- **F1, exposure-artifact.** Regress the candidate's return series against
  a flat-rescaled BTC-solo `kelly_regime_v4` benchmark (`r_squared`,
  imported unchanged from `kelly_regime_covkelly.py`, same >0.95
  threshold used throughout R-34/R-42/R-43/R-50). R² > 0.95 in either
  inner split = FAIL.
- **F2, fee-tier survival.** Re-run the selected candidate and the
  fixed-50/50 reference at the 0.40% Bitstamp taker tier
  (`scripts/fee_study.py`'s own convention). FAIL if the candidate's
  inner-validation Sharpe advantage over fixed-50/50 at 0.40% is negative,
  or if the candidate no longer beats `buy_and_hold` at 0.40%.

**Promotion decision rule — read the holdout once, on the frozen winner, only if F1 and F2 both PASS.** Promote iff ALL of:
- **P1** beats `buy_and_hold` OOS at 0.10% and at 0.40%;
- **P2** beats the re-derived static-50/50-continuous-engine reference by
  more than the ±0.2 Sharpe noise floor, OR shows a drawdown/tail
  improvement over it;
- **P3** survives F1 and F2 again on the holdout itself;
- **P4** the (lookback, cadence) neighbourhood is a plateau, not a peak.

Anything else is NEGATIVE. This rule was written into the module
docstring before the sweep below was run — `git status` on this branch
shows the file was never edited to move it after seeing a result (the
only post-hoc edit was the cache-bug fix described above, which changed
*correctness* of two already-printed numbers, not the rule itself, and
is disclosed in full rather than silently folded in).

## 1. Sweep — 12 configurations (spot, inner splits)

`lookback_days ∈ {30, 60, 90, 180} × rebalance_freq ∈ {monthly (MS),
quarterly (QS), semiannual (2QS)}`, selection rule fixed in advance
(`min(train_sharpe, valid_sharpe)`, tie-break `-valid_max_dd_pct` —
guards against the train-loses/validation-wins overfit signature that
sank R-37/R-38/R-40):

| lookback | freq | train Sharpe | train DD | valid Sharpe | valid DD | rebalance fees |
|---|---|---|---|---|---|---|
| 30d | monthly | 2.96 | 29.6% | 0.86 | 28.1% | $25.79 |
| 30d | quarterly | 3.00 | 29.9% | 0.88 | 27.4% | $13.54 |
| 30d | semiannual | 2.94 | 31.1% | 0.74 | 27.2% | $11.21 |
| 60d | monthly | 2.94 | 29.5% | 0.87 | 28.4% | $17.51 |
| **60d** | **quarterly** | **2.95** | **29.8%** | **0.88** | **27.8%** | **$10.83** |
| 60d | semiannual | 2.89 | 30.5% | 0.76 | 27.7% | $9.51 |
| 90d | monthly | 2.96 | 29.5% | 0.87 | 28.2% | $13.11 |
| 90d | quarterly | 2.95 | 29.8% | 0.86 | 28.0% | $10.46 |
| 90d | semiannual | 2.89 | 30.5% | 0.75 | 27.9% | $7.58 |
| 180d | monthly | 2.89 | 29.9% | 0.84 | 27.9% | $9.36 |
| 180d | quarterly | 2.92 | 30.2% | 0.85 | 27.6% | $6.80 |
| 180d | semiannual | 2.89 | 30.7% | 0.78 | 27.6% | $7.36 |

**Winner: lookback=60d, quarterly** (bold row) — tied for the top valid
Sharpe (0.88) with 30d/quarterly, wins the drawdown tie-break. The
neighbourhood is a genuine plateau on the monthly/quarterly side (valid
Sharpe 0.84–0.88 across 8 of 12 cells) and drops off cleanly on
semiannual (0.74–0.78) — **P4 (plateau not peak) holds for the winner's
immediate monthly/quarterly neighbours, but semiannual is a visibly
worse region, not noise**, which matters for the Axis-2 reading below.

## 2. Headline vs both references (spot, quarterly, lookback=60d)

| candidate | period | final | Sharpe | max DD |
|---|---|---|---|---|
| inverse-vol (candidate) | train | $4,762 | 2.95 | 29.8% |
| inverse-vol (candidate) | valid | $1,434 | 0.88 | 27.8% |
| fixed 50/50 (**re-derived**, same costed engine) | train | $4,540 | 2.92 | 30.1% |
| fixed 50/50 (**re-derived**, same costed engine) | valid | $1,461 | 0.92 | 27.1% |
| v4 BTC-solo | train | $6,207 | 2.62 | 30.4% |
| v4 BTC-solo | valid | $1,051 | 0.23 | 33.2% |
| `buy_and_hold` BTC | train | $7,459 | 1.81 | 71.8% |
| `buy_and_hold` BTC | valid | $574 | 0.08 | 77.3% |

Inner-validation ΔSharpe vs v4-solo: **+0.65**; ΔmaxDD vs v4-solo:
**−5.5pp**. The underlying diversification effect R-50 found (BTC+ETH
beats BTC-solo) replicates cleanly under inverse-vol weighting too — this
candidate is not a dud relative to the single-asset incumbent.

Inner-validation ΔSharpe vs the **re-derived** fixed-50/50 reference:
**−0.04**; ΔmaxDD vs fixed-50/50: **+0.6pp (worse)**. The re-derived
fixed-50/50 number (Sharpe 0.92, DD 27.1% at quarterly cadence) lands
close to R-50's own printed monthly/weekly numbers (Sharpe ≈0.93/0.94,
DD 27.1%), which is itself a useful cross-check that this file's engine
reproduces R-50's mechanism correctly.

**Cadence-by-cadence, all 12 configurations, candidate vs the same-cadence fixed-50/50 reference:**

| cadence | invvol valid Sharpe range | fixed-50/50 valid Sharpe | ΔSharpe range |
|---|---|---|---|
| monthly | 0.84 – 0.87 | 0.925 | **−0.081 to −0.054** |
| quarterly | 0.86 – 0.88 | 0.918 | **−0.071 to −0.036** |
| semiannual | 0.74 – 0.78 | 0.859 | **−0.114 to −0.081** |

**Every one of the 12 configurations underperforms the fixed-50/50
reference at its own cadence.** The gap is small — always inside the
±0.2 Sharpe noise floor — but it is one-directional across the entire
grid, not a coin flip: inverse-vol weighting never beats the reference
it was designed to improve on. Consistent with this, the R² between the
candidate and the fixed-50/50 reference is 0.996 on both inner splits
(§3) — the two portfolios are nearly indistinguishable return series;
inverse-vol weighting is mostly relabeling the static split with a small
amount of extra turnover.

**Axis-2 finding, stated plainly because it is the opposite of what the
stock-bond rebalancing literature predicts.** Both arms get *worse*, not
better, as cadence lengthens (fixed-50/50 Sharpe: 0.925 monthly → 0.918
quarterly → 0.859 semiannual). Rebalancing fees here are trivially small
relative to account size ($6.80–$25.79 against $1,350–$1,460
inner-validation balances, i.e. well under 2% drag even at the most
expensive monthly/30-day-lookback cell) — nowhere near large enough for
Dichtl/Drobetz/Wambach's cost-driven "quarterly-to-yearly is optimal"
mechanism to bite. Instead, less frequent rebalancing appears to cost
more in **diversification-maintenance value** (staying close to the
target split as BTC/ETH prices and volatilities diverge between
rebalances) than it saves in fees, at this pair's volatility level and at
this project's own realistic cost tier. This is a genuine, well-measured
negative for the "rebalance less" hypothesis on this specific asset pair
— not a null result, and not what was predicted going in.

## 3. Exposure-artifact check (F1)

| split | candidate vs flat-rescaled v4-BTC-solo | candidate vs fixed-50/50 |
|---|---|---|
| train | R² = 0.9386 (ok) | R² = 0.9963 |
| valid | R² = 0.5803 (ok) | R² = 0.9956 |

**F1: PASS.** Neither inner split's R² against solo BTC exceeds 0.95 —
the diversification effect against BTC-solo is a real return-series
difference, not a flat leverage rescale of one asset. (The very high R²
against the fixed-50/50 reference is not itself the F1 test — it is
diagnostic confirmation of §2's finding that the candidate is close to a
relabeled static split.)

## 4. Fee-tier stress test (F2) — 0.40% Bitstamp taker tier, spot, inner-validation

| candidate | final | Sharpe |
|---|---|---|
| inverse-vol (candidate) | $1,206 | 0.51 |
| fixed 50/50 | $1,230 | 0.56 |
| `buy_and_hold` | $572 | 0.07 |

ΔSharpe (candidate − fixed50/50) at 0.40%: **−0.04**. Candidate still
beats `buy_and_hold` at 0.40% (True). **F2: FAIL** — the pre-registered
rule required the candidate's advantage over fixed-50/50 to stay
non-negative at the real fee tier; it does not (consistent with, and no
worse than, the −0.04 already measured at the 0.10% baseline — turnover
cost is not what kills this candidate, it was never ahead to begin
with).

## 5. Secondary check: 5x futures (same winning config)

| candidate | period | final | Sharpe | max DD |
|---|---|---|---|---|
| inverse-vol (candidate) | train | $5,859 | 3.17 | 30.1% |
| inverse-vol (candidate) | valid | $1,461 | 0.91 | 28.2% |
| fixed 50/50 | train | $5,517 | 3.14 | 30.1% |
| fixed 50/50 | valid | $1,477 | 0.93 | 27.7% |
| v4 BTC-solo | train | $8,647 | 2.91 | 29.3% |
| v4 BTC-solo | valid | $1,082 | 0.28 | 32.3% |
| `buy_and_hold` | train | $33,246 | 2.05 | 92.3% |
| `buy_and_hold` | valid | $18 | 0.43 | 99.8% |

Same pattern replicates on leveraged futures: ΔSharpe (candidate vs
fixed-50/50) = **−0.02**, ΔmaxDD = **+0.5pp (worse)**. The negative
finding is not spot-specific.

## Configurations evaluated

- **12** distinct inverse-vol candidate configurations (`SWEEP_GRID`:
  4 lookbacks × 3 cadences) — this is the number that matters for this
  project's deflated-Sharpe bookkeeping, matching the established
  convention (R-42, R-43, R-50) of counting distinct candidate
  configurations, not baseline/reference/diagnostic re-runs. The grid
  itself never changed across development; it was re-executed several
  times while debugging the two bugs described above, but that is
  re-computation of the same 12 configurations, not a wider search — the
  same distinction R-30 draws between "recomputing an already-drawn
  statistic" and "asking a new question."
- **28** distinct backtests total in this file's own broader accounting
  (`N_BACKTESTS_TOTAL`, printed by `python
  experiments/b19_risk_parity_rebalance.py all`): 12 sweep + 2 headline
  (candidate + re-derived fixed-50/50) + 2 headline `buy_and_hold` +
  3 causality tamper probe (base/up/down) + 2 exposure-artifact (F1) +
  3 fee-tier stress test (F2: candidate + fixed-50/50 + `buy_and_hold`)
  + 4 futures secondary check (candidate + fixed-50/50 + 2×`buy_and_hold`)
  = 28. Underlying per-leg continuous-curve computations
  (`leg_equity`/`continuous_leg_equity`) are cached and reused across
  many of the above, and are not separately counted, matching R-42/R-43/
  R-50's own convention of counting configurations rather than leg-level
  calls.
- **0** holdout consultations by this branch (see verdict).

## Verdict: NEGATIVE

Inverse-volatility weighting **survives F1** (not an exposure artifact)
but **fails F2** (does not clear the 0.40% real fee tier's requirement
of a non-negative Sharpe advantage over the static reference) and, more
fundamentally, **fails the promotion rule's P2 clause before the holdout
was ever considered**: across all 12 configurations, at all three
cadences, on both spot and 5x futures, inverse-vol weighting never beats
the re-derived static-50/50-continuous-engine reference by more than the
noise floor — it loses by a small, consistent margin (−0.02 to −0.11
Sharpe) and shows a slightly *worse* drawdown, not better. Per the
pre-registered rule and ROUTINE.md's own instruction ("do not spend the
holdout on a candidate that already failed"), **the 2023+ holdout was
never read.**

The underlying diversification effect R-50 surfaced (BTC+ETH beats
BTC-solo) does replicate here (ΔSharpe +0.65 vs solo on inner-validation,
matching R-50's ballpark), which is itself informative: it means B-19's
core finding is *not* an artifact of the specific static 50/50 split —
it survives a different, information-using weighting scheme too. What
does **not** survive is the more ambitious claim that *using* volatility
information to move away from 50/50 helps: at the 2-asset BTC/ETH scale
this project can test, inverse-vol weighting is statistically
indistinguishable from (and numerically always slightly behind) doing
nothing more sophisticated than a fixed 50/50 split, and lengthening the
rebalance cadence to reduce turnover cost makes both arms *worse*, not
better, because the diversification-maintenance value of frequent
rebalancing outweighs its (here, trivially small) fee cost.

**One-line lesson:** at 2-asset BTC/ETH scale, using each leg's own
trailing volatility to move away from a static 50/50 split adds
turnover and a small, consistent Sharpe cost without a measurable
diversification benefit over doing nothing — the "well-established
middle ground" between no-information and mean-variance weighting is not
where R-50's edge lives, and it lives at monthly-or-shorter cadence, not
the quarterly/semiannual region the cost literature predicted.

**Next step, for a future session (not pursued here):** the conservative
static-50/50 B-19 branch (the disjoint parallel branch on this same
backlog item) is the more promising direction to pursue toward
registration infrastructure — this file's own numbers corroborate its
core comparison (v4-solo vs a 2-asset portfolio) rather than displacing
it. If a future round wants to keep testing the risk-parity axis, N>2
assets (where correlation structure genuinely varies leg-to-leg, unlike
the single-correlation 2-asset case) is the natural place inverse-vol
weighting might start to differ meaningfully from a fixed split; that is
blocked today on this project having only two real crypto price series
committed (BTC, ETH).
