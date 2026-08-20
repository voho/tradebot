# b20_threshold_band_5050 — B-20 NOVEL branch (08-20)

Unregistered experiment. Code: `experiments/b20_threshold_band_5050.py`.
Not `@register`ed, not auto-discovered, nothing committed — a human
operator merges and commits after both B-20 branches report. This branch
does not touch `kelly_regime_covkelly_v3_continuous.py`, `kelly_regime_
covkelly.py`, `kelly_regime_v4.py`, `src/tradebot/multiasset.py`,
`kelly_regime_dual_fixed.py`, or the disjoint parallel session's
`experiments/b20_literal_calendar_5050.py` (conservative branch — never
read or imported here); everything reused from existing files is
imported unchanged.

## Mechanism, one sentence

Hold both legs' CONTINUOUS (non-restarting) `kelly_regime_v4` equity
curves, track the implied live BTC weight of pooled capital at every
5-minute bar, and rebalance the pool back to exactly 50/50 — paying an
explicit round-trip fee for the reallocation — the instant that weight
drifts outside a band `[50%−b, 50%+b]`, rather than on any fixed
calendar; between breaches, no rebalancing trade happens at all, however
much calendar time passes.

## Why this is a genuine non-duplicate of R-50, R-51, and the conservative B-20 branch

- **Not R-50** (`kelly_regime_covkelly_v3_continuous.py`): R-50's
  fixed-50/50 arm rebalances on a fixed calendar every period regardless
  of drift. This file's engine never checks a calendar date to decide
  whether to trade — only the live weight.
- **Not R-51 conservative** (`b19_dual_fixed_split.py`, B-19): that
  branch is the `b → ∞` extreme of this file's own band axis (never
  rebalance at all) — it captured ~100% of R-50's drawdown edge but only
  ~29% of its Sharpe edge, then was REJECTED on holdout. This file's
  grid (5%/10%/15%) deliberately sits strictly between that extreme and
  "rebalance every period regardless of drift."
- **Not R-51 novel** (`b19_risk_parity_rebalance.py`, B-19): that branch
  kept a fixed calendar cadence but moved the TARGET weight away from
  50/50 (inverse trailing volatility). This file does the opposite: the
  target stays fixed at 50/50 throughout; only the trigger rule changes.
  The two axes (target-weight information vs. trigger-rule information)
  are orthogonal and neither branch tested the other's axis.
- **Not the conservative B-20 branch** (a disjoint parallel session,
  `experiments/b20_literal_calendar_5050.py`, never read here): that
  branch tests the LITERAL fixed-calendar-cadence form of R-50's finding.
  This file's candidate has no calendar cadence in its trigger rule at
  all — it trades at irregular, data-driven times. The two branches are
  complementary reads of the same B-20 lead, not restatements.

## Constraint attacked

SIZE and N≈3 (identical framing to every branch on this backlog item),
and — distinctively for this branch — **COST** directly: a band trigger
only pays the rebalancing fee when drift is large enough to justify the
trade; fixed-calendar rebalancing structurally cannot avoid paying it on
every period regardless of whether anything drifted.

## Citations

- Donohue, C. & Yip, K. (2003), "Optimal Portfolio Rebalancing with
  Transaction Costs", *Journal of Portfolio Management* 29(1), 49-63.
- Masters, S.J. (2003), "Rebalancing", *Journal of Portfolio Management*
  29(3), 52-57.
- Kitces, M. (2015), "Optimal Rebalancing: Time Horizons vs Tolerance
  Bands" (*The Kitces Report*, Vol. 2, 2015). **Verified directly** via
  web search while preparing this file (primary PDF located at
  kitces.com/wp-content/uploads/2015/07/Kitces-Report-Volume-2-2015-...
  -Rebalancing-Strategies.pdf): its central finding is that trades should
  be triggered by drift from target ("tolerance bands"), not by a time
  horizon, and that calendar rebalancing more often than roughly annually
  is generally not worth its cost.
- A 2024 crypto-specific empirical study, cited here as a **secondary
  description only** — I could not retrieve or verify the primary
  academic source (exact author/journal) from inside this sandboxed
  session. A web search performed while preparing this file returned
  consistent secondary summaries (portfolio-management blogs and
  aggregators, not the paper itself) describing a simulation of ~10,000
  cryptocurrency portfolios (BTC, ETH, USDT, LTC, SOL, DOGE, MATIC)
  comparing calendar rebalancing (daily/weekly/monthly) against
  threshold-based rebalancing at exactly 5%, 10% and 15% drift bands,
  reporting threshold rebalancing generally producing better
  risk-adjusted returns than calendar rebalancing at that scale. This is
  why this file's own band grid is exactly {5%, 10%, 15%} — matching that
  secondary-sourced grid so this file's result is directly comparable to
  it, while flagging plainly that the citation itself is unverified
  against a primary source (the same citation-honesty convention R-39
  used for its funding-harvest sources).
- This project's own R-33 ("holding less draws down less, that is
  arithmetic, not evidence") and R-51's decomposition (~71% of the
  calendar-rebalanced form's larger Sharpe edge over a never-rebalanced
  baseline traces to the periodic sell-winners/buy-losers act, which a
  bull-dominated 2023-2026 holdout already showed a closely related
  variant does not monetize) — the explicit standing caution this
  branch's decision rule was built to respect *before* running anything.

## Design choices, pre-registered before any result existed

- **Resolution: checked every bar, triggered discretely.** Drift is
  evaluated at the native 5-minute resolution of `continuous_leg_equity`'s
  curves, not on a coarser daily/weekly check calendar. This is
  "continuously monitored, discretely triggered," genuinely distinct from
  "checked and triggered on the same calendar" (e.g. check only once a
  day and trade only if breached that day) — the latter is a reasonable,
  cheaper-to-monitor alternative design not tested here.
- **Band grid, fixed in advance:** b ∈ {5%, 10%, 15%} of the 50/50
  target — 3 configurations, bracketing the crypto-study figure above
  while also testing tighter bands.
- **Target weight fixed at 50/50, never swept** — the one thing held
  identical to R-50/R-51/B-20's own candidate, so any difference found is
  attributable to the trigger rule alone.
- **Rebalance cost charged explicitly:** `2 × fee_rate × shift` per
  rebalance (a full round trip), applied identically to the candidate and
  to this file's own re-derived fixed-50/50-monthly reference — the same
  gap R-51's novel branch found and fixed in R-50's original engine,
  applied consistently here rather than rediscovered.

## A cache-key bug worked around without touching R-50's file

`continuous_leg_equity` memoizes on a key that omits `market.fee_rate` —
found and documented by R-51's novel branch, which fixed it with its own
separately-keyed local cache built on a reimplemented call. This file
takes a smaller fix (`leg_equity` in `b20_threshold_band_5050.py`): the
FIRST call for a given `(id(df), market.name)` pair passes the original
frame (correctly cached inside `continuous_leg_equity`); any subsequent
call at a genuinely different fee/leverage combination for that same pair
passes a `.copy()` instead, giving a fresh `id(df)` that cannot collide
with the earlier cache entry — `continuous_leg_equity` itself is never
edited or reimplemented. Verified empirically
(`python experiments/b20_threshold_band_5050.py cache_check`):
`max|0.10%-curve − 0.40%-curve| = 872.67` over ~2 years on the same BTC
frame — the two tiers genuinely diverge, confirming no stale-cache
collision.

## Causality probe (mandatory, run first)

Multiply/divide truncation-tamper procedure (K=137, cut=2021-06-30) on
`run_band_triggered` (the new code this file adds):

| check | max\|diff\| before cut | result |
|---|---|---|
| pooled equity (up-tampered) | 0.000e+00 | PASS |
| pooled equity (down-tampered) | 0.000e+00 | PASS |

No lookahead detected.

## Pre-registered falsification test and decision rule (written before any result existed)

**Falsification — the candidate is rejected, holdout never read, if EITHER fires on the inner splits:**
- **F1, exposure-artifact.** `r_squared` (imported unchanged from
  `kelly_regime_covkelly.py`) of the candidate vs a flat-rescaled
  BTC-solo `kelly_regime_v4` benchmark. R² > 0.95 on either inner split
  = FAIL.
- **F2, fee-tier survival.** Candidate vs BTC-solo v4 AND vs this file's
  own re-derived fixed-50/50-monthly reference at the 0.40% Bitstamp
  taker tier. FAIL if a positive 0.10%-tier Sharpe advantage over either
  benchmark turns negative at 0.40% (sign flip).

**Promotion rule — read the holdout once, on the frozen winning band, only if F1, F2, the plateau check, and the turnover check all pass.** Promote iff ALL of:
- **(a)** beats `buy_and_hold` OOS after real costs (0.10% AND 0.40%);
- **(b)** beats BOTH BTC-solo v4 AND the fixed-50/50-monthly reference by
  more than the ±0.2 Sharpe noise floor, OR shows a drawdown/tail
  improvement over BOTH (weighted over Sharpe, per the standing
  holdout-exhaustion caution — ~623 program-wide reads to date);
- **(c)** survives F1 and F2 again on the holdout itself;
- **(d)** the band-width neighbourhood is a plateau, not a knife-edge;
- **(e)** genuinely reduces rebalancing-trade count vs. the calendar
  reference — the whole point of this branch.

Anything else is NEGATIVE. **Standing caution carried in, not discovered
after running anything:** R-51 already found most of this family's
Sharpe edge over a never-rebalanced baseline traces to the same
return-side mechanism a bull-dominated 2023-2026 holdout has shown a
related variant does not monetize. This file's own stated prior, before
running anything, was that it is more likely than not to fail the same
way — which is why gate (b) weights drawdown/tail over Sharpe explicitly.
**This rule was not changed after seeing any result** — the only edits
made to the file after the causality probe passed were bug-free from the
start (no post-hoc threshold adjustment occurred; `git status` on this
branch shows no edits to the docstring after the sweep ran).

## 1. Sweep — 6 configurations (3 bands × 2 markets, inner splits read off one continuous run each)

| band | market | train Sharpe | train DD | valid Sharpe | valid DD | n_rebalances | rebal fees |
|---|---|---|---|---|---|---|---|
| ±5% | spot | 2.88 | 30.1% | 0.93 | 27.2% | 12 | $4.25 |
| ±5% | futures5x | 3.13 | 29.3% | 0.96 | 27.4% | 17 | $3.93 |
| ±10% | spot | 2.91 | 30.1% | 0.93 | 27.2% | 5 | $3.29 |
| ±10% | futures5x | 3.09 | 29.4% | 0.93 | 27.7% | 4 | $1.59 |
| ±15% | spot | 2.88 | 29.9% | 0.89 | 26.8% | 1 | $0.43 |
| ±15% | futures5x | 3.14 | 29.4% | 0.98 | 28.1% | 3 | $2.39 |

**Selected band (spot, min(train,valid) Sharpe, tie-break −valid DD): ±5%**
(tied with ±10% on both criteria; ±5% wins as the first-encountered tie in
the pre-registered grid order). **Neighbourhood is a clean plateau**: spot
inner-validation Sharpe ranges only 0.89–0.93 across the whole grid
(spread 0.04, far inside the ±0.2 noise floor) — gate (d) PASSES.

Turnover already shows the expected shape even at this stage: rebalance
count falls monotonically as the band widens (12 → 5 → 1 on spot), the
mechanism's central premise working exactly as designed.

## 2. Headline vs both references (spot, band=±5%)

| candidate | period | final | Sharpe | max DD |
|---|---|---|---|---|
| band-triggered (candidate) | train | $4,193 | 2.88 | 30.1% |
| band-triggered (candidate) | valid | $1,465 | 0.93 | 27.2% |
| fixed 50/50 monthly (re-derived ref) | train | $4,282 | 2.88 | 29.8% |
| fixed 50/50 monthly (re-derived ref) | valid | $1,464 | 0.92 | 27.1% |
| v4 BTC-solo | train | $6,207 | 2.62 | 30.4% |
| v4 BTC-solo | valid | $1,051 | 0.23 | 33.2% |
| `buy_and_hold` BTC | train | $7,459 | 1.81 | 71.8% |
| `buy_and_hold` BTC | valid | $574 | 0.08 | 77.3% |

Inner-validation ΔSharpe vs v4-solo: **+0.70**; ΔmaxDD: **−6.0pp**. The
underlying diversification effect (R-50's core finding) replicates
cleanly under band-triggered rebalancing too.

Inner-validation ΔSharpe vs the **re-derived** fixed-50/50-monthly
reference: **+0.01** (statistically indistinguishable); ΔmaxDD: **+0.1pp**
(also indistinguishable). **The candidate matches the calendar
reference's Sharpe and drawdown almost exactly** — the interesting part
is *how it gets there*:

**Turnover comparison (gate e), same [2019-03-14, 2022-12-31] span:**

| | candidate (±5% band) | fixed-50/50-monthly reference |
|---|---|---|
| n_rebalances | **12** | **45** |
| rebalance fees | $4.25 | $6.75 |

**73% fewer rebalancing trades for statistically identical Sharpe/drawdown
on the inner splits.** This is the mechanism working exactly as the
literature predicts: most of the calendar reference's 45 monthly dates
are trades that fire on a schedule even though the live weight had barely
drifted — a band trigger skips those and loses essentially nothing.

## 3. Exposure-artifact check (F1)

| split | candidate vs flat-rescaled v4-BTC-solo |
|---|---|
| train | R² = 0.9073 (ok) |
| valid | R² = 0.5522 (ok) |

**F1: PASS.** Neither split's R² exceeds 0.95.

## 4. Fee-tier stress test (F2) — 0.40% Bitstamp taker tier, spot, inner-validation

| tier | candidate Sharpe | cal-ref Sharpe | solo Sharpe | ΔSharpe vs solo | ΔSharpe vs cal |
|---|---|---|---|---|---|
| 0.10% | 0.93 | 0.92 | 0.23 | +0.70 | +0.01 |
| 0.40% | 0.57 | 0.56 | −0.07 | +0.63 | +0.01 |

No sign flip either comparison. **F2: PASS.**

## 5. Pre-holdout gate

| gate | result |
|---|---|
| F1 (exposure-artifact) | PASS |
| F2 (fee-tier survival) | PASS |
| (d) plateau | PASS (spread 0.04 across the 3-band grid) |
| (e) turnover reduction | PASS (12 vs 45 rebalances) |

**All four gates PASS → PROCEED to the single pre-registered holdout read.**

## 6. Holdout read (2023-01-01 onward) — the ONE pre-registered call, both fee tiers

**Min/max-date sanity check, printed and asserted before any headline
number was read** (per this round's critical-trap warning): candidate
equity actually covered **2023-01-01 00:00:00+00:00 → 2026-08-12
00:00:00+00:00** at both fee tiers — well past 2023-01-01, confirming the
`FULL_END="2022-12-31"` default trap was not hit (this file always passes
`full_start=OOS_START, full_end=holdout_end` explicitly to every call in
`holdout()`). `holdout_end` was computed as the true min of the two full
uncut files' last dates: BTC (Bitstamp spot) ends 2026-08-12, ETH
(Coinbase spot) ends 2026-08-19.

| tier | candidate | fixed-50/50-cal | v4-solo | `buy_and_hold` |
|---|---|---|---|---|
| 0.10% | $2,008, Sharpe 0.89, DD 26.2%, n_rebal=5 | $2,020, Sharpe 0.90, DD 26.1%, n_rebal=43 | $2,229, Sharpe 0.90, DD 27.8% | **$3,830, Sharpe 1.03, DD 54.0%** |
| 0.40% | $1,472, Sharpe 0.55, DD 32.2%, n_rebal=9 | $1,454, Sharpe 0.54, DD 32.2%, n_rebal=43 | $1,634, Sharpe 0.61, DD 34.1% | **$3,818, Sharpe 1.03, DD 54.0%** |

**Decision-rule check:**
- **(a) beats `buy_and_hold` after real costs: FAILS outright**, at both
  tiers — the candidate loses to holding BTC alone by ~48% (0.10%) to
  ~61% (0.40%) on final balance. `buy_and_hold`'s own drawdown (54.0%)
  during 2023-2026 was large but its return dominated everything else in
  this table, the same bull-dominated pattern R-51 conservative already
  found for the literal calendar form.
- **(b) beats both v4-solo and the calendar reference by >0.2 Sharpe or on
  drawdown/tail: FAILS.** Every ΔSharpe and ΔmaxDD against either
  benchmark is inside 2 percentage points / 0.06 Sharpe at both tiers —
  none of it clears the noise floor, and none of it is a meaningful
  drawdown improvement either. The candidate is statistically
  indistinguishable from both benchmarks on the holdout.
- **(c)** moot — (a) and (b) already fail decisively.
- **(e) turnover reduction: still PASSES, robustly, on the holdout too** —
  5 rebalances vs. 43 at the 0.10% tier (88% fewer), 9 vs. 43 at 0.40%
  (79% fewer). This is the one part of the mechanism that worked exactly
  as designed, on data the candidate had never seen.

**Secondary, unplanned but informative cross-check**: the fixed-50/50-
monthly reference computed independently in this file also loses to
`buy_and_hold` on the same holdout ($2,020/$1,454 vs. $3,830/$3,818) —
consistent with, and an independent (though unplanned) replication of,
what R-51 conservative already found for the never-rebalanced form and
what the disjoint conservative B-20 branch is separately testing for the
literal calendar form. This file's reference was built without importing
either branch's file, so this agreement is a genuine cross-check, not a
shared assumption.

## Configurations evaluated

- **6** distinct band-triggered candidate configurations (3 bands × 2
  markets) — the number for this project's deflated-Sharpe bookkeeping,
  matching the established convention (R-42/R-43/R-50/R-51) of excluding
  baselines/references/diagnostics.
- **~33** distinct backtests total across the full session, counting
  everything: 6 sweep + 5 headline (candidate + calendar-ref + solo + 2×
  buy_and_hold) + 1 artifact (F1) + 6 feetier (2 tiers × candidate/
  cal-ref/buy_and_hold, F2) + 2 turnover (candidate + cal-ref) + 3
  causality tamper probe (base/up/down) + 8 holdout (2 tiers ×
  candidate/cal-ref/solo/buy_and_hold) + 2 cache-safety diagnostic
  (single-leg, not portfolio configs). Underlying per-leg continuous-curve
  computations (`leg_equity`) are cached and reused across most of the
  above and are not separately counted, matching R-42/R-43/R-50/R-51's
  own convention.
- **Holdout reads: 1** — one paired call across both fee tiers (0.10%,
  0.40%), matching this project's established convention
  (`b19_dual_fixed_split.py::holdout`'s template).

## Verdict: NEGATIVE

The band-triggered mechanism **works exactly as designed on the axis it
was built to attack**: at the selected ±5% band, it matches the
fixed-50/50-monthly calendar reference's Sharpe and drawdown to within
noise on both inner-validation (ΔSharpe +0.01, ΔDD +0.1pp) and the
holdout (ΔSharpe ±0.01, ΔDD ≈0pp at both fee tiers), while trading 73%
less on the inner splits and 79-88% less on the holdout. This is a clean,
real, holdout-confirmed COST-axis result: a band trigger genuinely
reduces rebalancing turnover relative to a fixed calendar for
statistically indistinguishable risk-adjusted performance, on this asset
pair and strategy, at this project's cost tier.

That result does **not** rescue the underlying idea from the standing
caution this branch's own pre-registration carried in before running
anything. The candidate **decisively fails the pre-registered promotion
rule's clause (a)**: it loses to `buy_and_hold` by roughly half its value
over the 2023-2026 holdout, at both fee tiers — the same bull-dominated
period in which R-51 conservative already found the never-rebalanced
form of this idea fails, and in which this file's own independently
re-derived fixed-50/50-monthly reference fails too. It also fails clause
**(b)**: it does not beat either BTC-solo `kelly_regime_v4` or the
calendar reference by more than the noise floor, on Sharpe or on
drawdown/tail — it is statistically indistinguishable from both, not
better than either. Changing *when* a 50/50-target rebalance fires does
not change the answer R-51 already found for changing the calendar
cadence or abandoning rebalancing altogether: the return-side premium
this whole multi-asset-rebalancing family depends on for its Sharpe edge
does not survive a bull-dominated 2023-2026 holdout, regardless of
whether the trigger is a calendar date, a drift band, or nothing at all.

**One-line lesson:** a band trigger can cut rebalancing turnover by
70-90% for the SAME risk-adjusted performance as a fixed calendar
(a real, holdout-confirmed COST-axis win, independent of trigger-rule
choice) — but the family's underlying Sharpe edge over `buy_and_hold`
still depends on a bull/bear regime mix this 2023-2026 holdout does not
supply, and no trigger rule tested on this backlog item to date (never,
monthly/quarterly/semiannual, or now drift-band) changes that.

**Next step, for a future session (not pursued here):** the turnover
result is real and could matter for a strategy whose Sharpe edge *does*
survive the holdout on other grounds — worth remembering the next time a
periodically-rebalanced multi-asset candidate clears its own promotion
bar and a lower-cost implementation is wanted. On this specific backlog
item, B-20 is now answered on both its literal-calendar axis (see the
conservative branch's own report) and this trigger-rule axis: three
independent trigger designs (never, calendar, drift-band) and one
target-weight variant (inverse-vol) have now all been tested against the
2023-2026 holdout for this asset pair and strategy, and none has cleared
the promotion bar. Per the standing recommendation repeated since R-46,
**B-06 (forward paper trading, ongoing since R-48)** remains the
highest-value item on merit — it is the only evidence stream this
5-branch, 2023-2026-holdout-exhausted research line has not already
spent.
