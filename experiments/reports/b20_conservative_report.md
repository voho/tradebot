# B-20 (conservative-round session) — the literal periodic-rebalance, fixed-50/50 candidate (08-20)

Unregistered experiment. Code: `experiments/b20_literal_calendar_5050.py`.
Not `@register`ed, not auto-discovered, nothing committed by this session.
`kelly_regime_v4.py`, `multiasset.py`, `kelly_regime_covkelly.py`,
`kelly_regime_covkelly_v3_continuous.py`, `kelly_regime_dual_fixed.py`,
`b19_dual_fixed_split.py` and `b19_risk_parity_rebalance.py` are all
imported from or read for reference; none is edited anywhere in this
branch.

**Backlog item attacked:** B-20 — "Does the LITERAL periodically-rebalanced
(monthly, or another single cadence fixed before running), fixed-50/50
BTC+ETH `kelly_regime_v4` portfolio — R-50's own original candidate, run
through its continuous (non-restarting) engine, unmodified split,
unmodified cadence discipline — survive its own pre-registered
falsification test and a first, single holdout read?"

## 1. Mechanism, one sentence

Run each of BTC-`kelly_regime_v4` and ETH-`kelly_regime_v4` ONCE,
continuously, from ETH's real data start (so neither leg's deadband/
vol-regime hysteresis latch is ever reset), rebalance pooled capital back
to a FIXED 50/50 split at the start of every calendar month, and ask
whether that specific, literal object beats `buy_and_hold` and BTC-solo v4
on data it has never touched.

## 2. Why this is not a duplicate

Three sessions have now touched adjacent territory, and none of them ran
this exact object through the holdout:

- **R-50** (`experiments/kelly_regime_covkelly_v3_continuous.py`, B-18)
  built the continuous (non-restarting) per-leg engine this file reuses,
  and found — as an *unplanned byproduct* of a diagnostic headline table,
  never pre-registered, no falsification test, no holdout read — that a
  periodically-rebalanced fixed-50/50 book beats BTC-solo v4 by ΔSharpe
  +0.79 (monthly) / +0.80 (weekly), max DD 33.2%→27.1%, **on
  inner-validation only**. This file is the pre-registration and the
  holdout read R-50 explicitly declined to do in the same session.
- **R-51 conservative** (`experiments/b19_dual_fixed_split.py`, B-19)
  deliberately tested a *never-rebalanced, one-time* 50/50 split instead —
  capital is split once and then left to drift with each leg's own
  performance, with zero periodic rebalancing at any cadence. It cleared
  both falsification gates and a plateau check, then was REJECTED on its
  one holdout read (loses to `buy_and_hold` by 24–46%; statistically
  indistinguishable from BTC-solo v4). Its own decomposition (ΔSharpe
  +0.23 there vs. R-50's +0.79/+0.80) found the never-rebalanced split
  captures ~100% of R-50's drawdown edge but only ~29% of R-50's Sharpe
  edge — meaning ~71% of the LARGER, UNTESTED Sharpe edge specifically
  requires the periodic sell-winners/buy-losers act R-51-conservative's
  candidate never performs, by design. That 71% is exactly what this
  file's candidate performs and R-51-conservative does not.
- **R-51 novel** (`experiments/b19_risk_parity_rebalance.py`, B-19) DID
  stay periodically rebalanced, but replaced the fixed-50/50 weight with
  inverse-trailing-volatility weights. Its own re-derived fixed-50/50 arm
  is present in that file only as an inner-validation REFERENCE POINT for
  scoring the inverse-vol candidate — its own pre-registration never
  authorized reading the holdout on that reference by itself, and,
  independently confirmed by the operator, `holdout()` there is gated
  behind a CLI argument no invocation in that branch's own report ever
  passes.

This file's candidate is the one specific object — periodic (calendar)
rebalancing, back to UNMODIFIED fixed 50/50 weights, through the
continuous (non-restarting) engine — that sits at the intersection of
"periodically rebalanced" (which R-51-conservative deliberately is not)
and "fixed 50/50, not information-weighted" (which R-51-novel deliberately
is not), and it has not been holdout-tested by any prior round.

## 3. Standing caution carried into this pre-registration

Read directly from `docs/LEDGER.md` before writing a line of code:
R-51-conservative's own decomposition already found the DRAWDOWN-ONLY
component of this general family fails outright on 2023+ (−6.1pp
inner-validation edge compressed to a −0.8pp non-effect on the holdout;
the book lost to `buy_and_hold` by 24–46%), and R-51-conservative
attributed roughly 71% of THIS file's larger, untested Sharpe edge to the
periodic rebalancing act itself — a return-side mechanism a bull-dominated
2023–2026 holdout has already shown a closely related variant does not
reliably monetize. The evidence available before this file's holdout read
updated AGAINST the candidate, not for it. This caution was written into
the module docstring, and the decision rule below, BEFORE `sweep()`,
`select()`, `artifact()`, or `feetier()` was ever run — timestamped by
`git log`/commit order once this branch's changes are reviewed, and
verifiable directly from the file: the entire pre-registration section
appears above every function definition that runs a backtest.

## 4. Pre-registered falsification test (quoted from the module docstring, written before any result was read)

> (F1) **Exposure-artifact check**: the candidate's aggregate exposure
> series (dollar-weighted sum of each leg's own `target` fraction) must
> NOT be an R^2 > 0.95 flat rescale of BTC-solo `kelly_regime_v4`'s own
> exposure, on inner-validation, both markets.
>
> (F2) **0.40% Bitstamp taker fee tier**: the 50/50 candidate's advantage
> over BTC-solo v4 (Sharpe and drawdown) must not flip sign relative to
> the project's usual 0.10% tier, on both inner-train and
> inner-validation.
>
> If EITHER fails, STOP — do not read the 2023+ holdout. Report NEGATIVE.

## 5. Pre-registered promotion decision rule (quoted, written before the holdout was read)

> Promote (`PROMOTED-CANDIDATE`) only if, on the 2023+ holdout, using the
> FROZEN 50/50-monthly configuration and no other:
>
> (a) beats `buy_and_hold` OOS after real costs (0.10% spot as the table
> convention, reported alongside 0.40%);
>
> (b) the improvement over BTC-solo `kelly_regime_v4` exceeds the ±0.2
> Sharpe noise floor (R-20) **OR** is a drawdown/tail improvement — per
> the standing caution above, a drawdown/tail improvement that does NOT
> reverse sign from its inner-validation reading is weighted as the
> stronger form of evidence here; a Sharpe-only improvement, at ~623+
> program-level holdout consultations, is treated as suggestive at best
> and NOT sufficient on its own to satisfy (b);
>
> (c) survives both falsification checks (F1, F2) above;
>
> (d) the 50/50 → 60/40 → 40/60 neighbourhood is a plateau, not a
> knife-edge (no metric flips sign or changes by more than the noise floor
> between adjacent splits), evaluated on the inner splits only.
>
> Anything else is `NEGATIVE`. **If this rule is changed after any result
> in this file is seen, that change will be stated explicitly and the
> result downgraded to in-sample — it will not be done silently.**

This text was committed in the module docstring before `gate()` or
`holdout()` were ever called. Section 8 below documents one genuine
**implementation bug fix** (not a rule change) discovered while running
`gate()` for the first time — recorded honestly rather than silently
corrected.

## 6. Causality check on this file's own composition code (`run_calendar_rebalance_fixed`)

Standard two-opposite-tampers probe (cut inside inner-train, 2020-06-30,
nowhere near the holdout): bars after the cut multiplied by 137× in one
copy, divided by 137× in the other.

| check | max\|diff\| strictly before cut | result |
|---|---|---|
| portfolio equity, up-tampered vs base | 0.000e+00 | PASS |
| portfolio equity, down-tampered vs base | 0.000e+00 | PASS |

No lookahead in this file's composition. Run before anything else in this
file was trusted, per the module docstring's step 0.

## 7. Inner-train and inner-validation results

Inner-train = 2019-03-14 (ETH's real start on the committed Coinbase file)
→ 2020-12-31. Inner-validation = 2021-01-01 → 2022-12-31 (the 2022 BTC/ETH
joint bear). Both windows are sliced from ONE continuous per-leg run
spanning `FULL_START` → `2022-12-31` (never two separate runs — that would
silently reintroduce the exact restart artifact R-50 exists to remove).
Data (`kelly_regime_covkelly.py::load_assets`, hard-sliced ≤2022-12-31,
byte-identical to R-42/R-43/R-49/R-50/R-51):

### 7a. Inner-train (spot, monthly rebalance)

| strategy | final | Sharpe | max DD |
|---|---|---|---|
| `kelly_regime_v4` BTC-only | $6,167 | 2.62 | 30.4% |
| `buy_and_hold` BTC-only | $7,458 | 1.81 | 71.8% |
| dual 50/50 (this candidate) | $4,286 | 2.89 | 29.8% |
| dual 60/40 BTC | $4,659 | 2.95 | 29.8% |
| dual 40/60 ETH | $3,924 | 2.72 | 30.3% |

### 7b. Inner-validation (spot and 5x futures, monthly rebalance)

| strategy | market | final | Sharpe | max DD |
|---|---|---|---|---|
| `kelly_regime_v4` BTC-only | spot | $998 | 0.14 | 33.2% |
| `buy_and_hold` BTC-only | spot | $574 | 0.08 | 77.3% |
| dual 50/50 (this candidate) | spot | $1,464 | **0.93** | **27.1%** |
| dual 60/40 BTC | spot | $1,376 | 0.80 | 28.0% |
| dual 40/60 ETH | spot | $1,556 | 1.01 | 26.3% |
| `kelly_regime_v4` BTC-only | futures5x | $1,064 | 0.25 | 32.3% |
| `buy_and_hold` BTC-only | futures5x | $18 | 0.43 | 99.8% |
| dual 50/50 (this candidate) | futures5x | $1,480 | **0.94** | **27.7%** |
| dual 60/40 BTC | futures5x | $1,396 | 0.83 | 28.6% |
| dual 40/60 ETH | futures5x | $1,565 | 1.01 | 26.9% |

**This independently reproduces R-50's own cited byproduct number almost
exactly** (spot, monthly, 50/50, inner-validation): ΔSharpe = 0.93 − 0.14 =
**+0.79**, max DD 33.2% → **27.1%** — identical to R-50's own row text to
two decimal places (R-50: "ΔSharpe +0.79 ... max DD 33.2% vs. 27.1%"). The
v4-solo and `buy_and_hold` baselines also match `b19_dual_fixed_split.py`'s
own baseline table to the dollar ($998/0.14/33.2% spot validation), a
useful independent cross-check that this file's data/warmup wiring is
correct.

**Plateau check**: adjacent-split Sharpe deltas on inner-validation spot
are |0.93−0.80|=0.12 (50/50→60/40) and |1.01−0.93|=0.08 (50/50→40/60), both
well under the ±0.2 noise floor; all three splits sit on the same side of
BTC-solo v4 (all clearly better). Genuine plateau, no knife-edge.

## 8. Falsification test results (both pre-registered, run before the holdout)

### 8a. F1 — exposure-artifact check

Aggregate exposure = Σ(leg's own `target` fraction × leg's own dollar
contribution within the candidate) / total portfolio equity, on the frozen
50/50-monthly config, inner-validation, vs. a mean-matched flat rescale of
BTC-solo v4's own `target` series:

| market | rescale c | corr | R² | verdict |
|---|---|---|---|---|
| spot | 0.965 | 0.9548 | **0.8801** | not a flat rescale — PASSES F1 |
| futures5x | 0.966 | 0.9546 | **0.8793** | not a flat rescale — PASSES F1 |

Both well under the 0.95 danger line.

### 8b. F2 — 0.40% taker fee-tier check

| window | fee | dual 50/50 final | Sharpe | DD | v4-solo final | Sharpe | DD | ΔSharpe | ΔDD |
|---|---|---|---|---|---|---|---|---|---|
| train | 0.10% | $4,286 | 2.89 | 29.8% | $6,167 | 2.62 | 30.4% | +0.27 | −0.6pp |
| train | 0.40% | $3,717 | 2.62 | 33.4% | $5,294 | 2.42 | 34.5% | +0.20 | −1.1pp |
| validation | 0.10% | $1,464 | 0.93 | 27.1% | $998 | 0.14 | 33.2% | +0.78 | −6.1pp |
| validation | 0.40% | $1,234 | 0.56 | 33.6% | $834 | −0.17 | 39.7% | +0.73 | −6.0pp |

Drawdown advantage does not flip sign at 0.40% on either window (train:
−0.6pp→−1.1pp; validation: −6.1pp→−6.0pp — essentially stable). Sharpe
advantage shrinks slightly (+0.27→+0.20 train, +0.78→+0.73 validation) but
stays positive in every cell. **PASSES.**

### 8c. Implementation bug found and fixed before any decision was read (recorded honestly)

The first run of `gate()`'s plateau check used the FULL max−min span across
all three splits (0.21) against the ±0.2 threshold, which is stricter than
the pre-registered wording ("no metric ... changes by more than the noise
floor **between adjacent splits**") and produced a spurious FAIL right at
the boundary. This was a coding bug relative to the already-written
docstring text, not a threshold search: the fix (adjacent-pair differences
along 60/40→50/50→40/60: 0.12 and 0.09) was applied and `gate()` re-run
once, per ROUTINE.md's own distinction ("Going back to step 3 to fix a
*bug* is fine and always was ... The difference is whether the target
moved" — the docstring's target text was never edited, only the code that
was supposed to implement it).

### 8d. Full gate decision

| gate | result |
|---|---|
| 1. inner-validation improvement (spot, 50/50 vs v4-solo) | dSharpe=+0.78, dDD=−6.1pp → **PASS** |
| 2. F1 exposure-artifact | R²=0.88/0.88 → **PASS** |
| 3. F2 fee-tier stability | stable both windows → **PASS** |
| 4. plateau (adjacent-split deltas ≤0.2) | 0.12, 0.09 → **PASS** |

**All four gates pass → holdout authorized per §5.**

## 9. Holdout read (ONE paired call, both fee tiers, per §5)

2023-01-01 → 2026-08-12 00:40 UTC (the true common end of the committed BTC
Bitstamp spot and ETH Coinbase spot files — BTC's own last bar, since it
ends 7 days before ETH's; computed at runtime, never hardcoded).

**CRITICAL-TRAP sanity check, printed before any holdout number was
reported:** candidate segments span 2019-03-14 → 2026-08-01; equity index
spans 2019-03-14 → **2026-08-12**. `assert eq_max > 2023-01-01` — PASS.
This really is the holdout, not a silent repeat of inner-validation.

| fee tier | dual 50/50 final | Sharpe | DD | v4-solo final | Sharpe | DD | `buy_and_hold` final | Sharpe |
|---|---|---|---|---|---|---|---|---|
| 0.10% | $2,989 | 1.27 | 26.6% | $3,373 | 1.22 | 27.8% | $3,839 | 1.03 |
| 0.40% | $2,122 | 0.92 | 32.1% | $2,445 | 0.94 | 34.1% | $3,827 | 1.03 |

Applying the pre-registered rule from §5, exactly as written, no goalposts
moved:

- **(a) beats `buy_and_hold` OOS after real costs?** No. At 0.10%, dual
  50/50 ($2,989) loses to holding ($3,839) by **−22.1%**. At 0.40% the gap
  widens to **−44.6%**. **FAILS**, at both fee tiers.
- **(b) improvement over BTC-solo v4 exceeds the ±0.2 Sharpe noise floor,
  or is a drawdown/tail improvement?** ΔSharpe at 0.10% = 1.27 − 1.22 =
  **+0.05**; at 0.40% = 0.92 − 0.94 = **−0.02**. Both deep inside the
  noise floor — neither clears it, and the sign is not even stable
  (marginally negative at the realistic fee tier). ΔDD at 0.10% =
  26.6% − 27.8% = **−1.2pp**; at 0.40% = 32.1% − 34.1% = **−2.0pp**. The
  drawdown edge does not reverse sign, but per the standing caution it
  compresses from the inner-validation reading (−6.1pp/−6.0pp) by
  roughly 65–80% — the same qualitative pattern R-51-conservative found
  for the never-rebalanced variant (−6.1pp → −0.8pp there), just slightly
  less total decay here. This is not, on any reasonable reading, "a
  drawdown/tail improvement" of the size this project has previously
  treated as meaningful (compare the 6pp+ edges that motivated B-19/B-20
  in the first place). **FAILS** as a practical matter, though less
  decisively than (a).
- (c) survives both falsification tests: yes (§8), but (a) already fails —
  a clean falsification result does not rescue a promotion-bar failure on
  the actual decision criteria.
- (d) neighbourhood plateau: passed on inner-validation (§7b/§8d); not
  re-tested on the holdout by design — only the frozen 50/50 config was
  read there, exactly as pre-registered.

**(a) fails decisively and on its own is sufficient for NEGATIVE, exactly
as R-51-conservative's holdout read was decided.** No threshold in §5 was
edited after these numbers were seen, and no alternative candidate (e.g.
40/60 ETH, which looked best on inner-validation, §7b) was substituted or
read on the holdout to try to rescue the result.

## 10. Configurations evaluated (deflated-Sharpe bookkeeping)

Counted per this project's established convention: every distinct
(split × window × market × fee-tier) dual-book candidate configuration
actually backtested; baseline/reference runs (`buy_and_hold`, BTC-solo v4)
are not counted, matching `b19_dual_fixed_split.py`'s own convention. The
full pipeline (`causality`→`sweep`→`select`→`baselines`→`gate`→`holdout`)
was re-run once, in a single process, purely to get an honest cumulative
count from the script's own dedup tracking rather than summing separate
CLI invocations by hand (each CLI invocation resets its counters, so
summing printed per-invocation totals from separate runs would silently
under- or double-count overlapping cells). The script's own `N_EVALUATED`
after the full pipeline: **15**.

1. 50/50, train, spot, tag="std" (`sweep()`)
2. 60/40 BTC, train, spot, tag="std" (`sweep()`)
3. 40/60 ETH, train, spot, tag="std" (`sweep()`)
4. 50/50, validation, spot, tag="std" (`select()`)
5. 50/50, validation, futures5x, tag="std" (`select()`)
6. 60/40 BTC, validation, spot, tag="std" (`select()`)
7. 60/40 BTC, validation, futures5x, tag="std" (`select()`)
8. 40/60 ETH, validation, spot, tag="std" (`select()`)
9. 40/60 ETH, validation, futures5x, tag="std" (`select()`)
10. 50/50, train, spot, tag="0.10%" (F2 — see note below)
11. 50/50, train, spot, tag="0.40%" (F2)
12. 50/50, validation, spot, tag="0.10%" (F2 — see note below)
13. 50/50, validation, spot, tag="0.40%" (F2)
14. 50/50, **holdout**, spot, tag="0.10%"
15. 50/50, **holdout**, spot, tag="0.40%"

**Honest note on double-labeling (recorded rather than quietly netted
out):** configurations 10 and 12 are, numerically, the *same backtest* as
configurations 1 and 4 respectively — `sweep()`/`select()` tag the SPOT
market run `"std"` while `feetier()`'s F2 check separately tags the
identical SPOT-market run `"0.10%"` for its own paired comparison against
the 0.40% tier, and the dedup key in `run_window()` is
`(split, window, market.name, fee_tag)`, so the two differently-labeled
calls do not collide even though `market` is the literal same `SPOT`
object in both cases. This inflates the honest count by 2 relative to a
maximally-deduplicated reading (13, which is what a single CLI invocation
of `gate()` alone would have reported, and is also `b19_dual_fixed_split.py`'s
own total for its structurally analogous grid). Per R-39's own "honest
count" convention (report the real number a script's own bookkeeping
produces rather than a hand-cleaned one), **15** is the number carried
forward here — it is the conservative direction for deflated-Sharpe
purposes (more trials assumed, not fewer), not an undercount.

Not counted toward the trials figure above, matching this project's
established baseline convention: the `buy_and_hold`/BTC-solo v4 baseline
backtests in `baselines()` and inside `feetier()`/`gate()`/`holdout()`
(12 in `baselines()` alone, plus reference runs at the 0.40% tier and on
the holdout), and 3 tamper-probe runs for the causality check (§6), which
reads no window past inner-train and is a correctness check, not a search
point.

**Holdout counter: +2** on top of the project's running total of ~623 as
of R-51 (08-20) — one frozen configuration (50/50 monthly) read at two fee
tiers, pre-registered as a single paired read, matching the convention
used for R-35/R-51-conservative's own fee-tier pairs.

## 11. Verdict

**NEGATIVE.**

The candidate replicates R-50's inner-validation byproduct number almost
exactly (ΔSharpe +0.79, DD 33.2%→27.1%, spot, monthly, matching R-50's own
row to two decimal places) — this is not a computational artifact of
sloppy re-derivation, it is the real, reproducible object B-20 asked about.
It also clears every gate this project's discipline puts in front of a
holdout read: it is not an exposure-level artifact (R²=0.88, comfortably
under the 0.95 danger line), it survives the 0.40% fee tier without its
drawdown advantage flipping sign, and the 50/50→60/40→40/60 neighbourhood
is a genuine plateau (adjacent Sharpe deltas 0.08–0.12, all splits
same-sign vs. the incumbent).

None of that survives contact with the one holdout read this file's own
pre-registration authorized. The dual 50/50-monthly book loses to simple
`buy_and_hold` by 22% (0.10% tier) to 45% (0.40% tier) over 2023–2026, and
its improvement over BTC-solo `kelly_regime_v4` alone is essentially noise
on Sharpe (+0.05/−0.02, not even stably signed across fee tiers) and
heavily compressed on drawdown (−6.1pp inner-validation → −1.2pp/−2.0pp
holdout, a 65–80% decay). Criterion (a) fails decisively on its own,
exactly the way R-51-conservative's holdout read failed on its own (a).

**This closes B-20's literal candidate the same way R-51 closed both of
its own candidates: a real, non-artifact, falsification-clean,
inner-validation-plateau mechanism that still does not clear this
project's promotion bar on 2023–2026 data**, now the third time running
this specific finding shape (candidate beats the incumbent cleanly on
2021–2022, loses or is indistinguishable from it on 2023–2026) has
appeared for a multi-asset BTC+ETH `kelly_regime_v4` composition (R-43,
R-51-conservative, and now this file). It also directly confirms the
standing caution this file's own pre-registration carried in from
R-51-conservative's decomposition: the periodic-rebalancing-driven
return premium, real and mechanism-backed as it is on 2019–2022 data, is
the SAME return-side effect the 2023–2026 bull-dominated holdout has now
failed to monetize twice — once when isolated as a small residual
(R-51-conservative's 29% remainder) and once now, at its full, literal,
unattenuated size.

**One-line lesson:** the periodic-rebalancing act is a real, reproducible
mechanism on 2019–2022 data (it drives ~71% of R-50's Sharpe edge per
R-51-conservative's decomposition, and this file's own inner-validation
numbers corroborate that scale directly), but "real mechanism on the
training regime" and "clears this project's holdout bar" are now
confirmed, for the third time on this specific research line, to be
different claims — the drawdown property partially transfers (does not
reverse sign) while the return property that dominates the inner-
validation headline does not survive a fee-charged, buy-and-hold-
dominant 2023–2026 bull market.

**Next step.** B-20 is now CLOSED — the one literal form of R-50's finding
left untested by R-51's two branches has been tested, and it fails the
same way. Nothing under this specific "BTC+ETH `kelly_regime_v4`
diversification via periodic rebalancing" family remains untested at the
level of ambition this backlog item asked for. Per the ledger's own
standing recommendation attached to B-20, and consistent with R-51's
closing note, a future session should not spend a fourth holdout
consultation on a fourth variant of this same underlying idea without a
materially different mechanism or asset pair; `scripts/paper_trade.py`
(B-06) remains the standing zero-cost, no-new-idea-needed alternative, and
is otherwise the only evidence stream this project has that is immune to
holdout exhaustion.

## 12. Holdout-read count (for the project's running counter)

**+2** — one frozen configuration (50/50, monthly), two fee tiers (0.10%,
0.40%), one paired call inside `holdout()`, matching the R-35/R-51
convention exactly. `HOLDOUT_READS` printed by the script itself confirms
2. No other function in this file reads any bar dated 2023-01-01 or later
(grepped: the only "2023"/"2024"/"2025"/"2026" literals in the file are
`OOS_START`, the module docstring's prose, and the runtime-computed
`HOLDOUT_END`/`_HOLDOUT_END_TS`, none of which are read outside
`holdout()` and the module-level data load that also feeds it).
