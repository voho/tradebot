# kelly_regime_v14_macro_lead — R-53 NOVEL branch (08-20)

Unregistered experiment. Code: `experiments/kelly_regime_v14_macro_lead.py`.
Not `@register`ed, not auto-discovered, nothing committed by this branch's
own choice — a human operator merges and commits after both R-53 branches
report. This branch imports `experiments/_macro_signal.py` unchanged (per
its own docstring, shared and not editable by either branch) and does not
touch `kelly_regime_v4.py`, `kelly_regime_v3.py`, `kelly_regime.py`,
`src/tradebot/data.py`, `docs/LEDGER.md`, or the disjoint parallel
conservative branch's file (`kelly_regime_v14_macro_brake.py`, not read,
not coordinated with). All evaluation below is restricted to inner-train
(2017-01-01 → 2020-12-31), inner-validation (2021-01-01 → 2022-12-31), and
the standard pre-2020 ETH/BTC falsification pair. **The 2023+ holdout was
never read** — grepped for date literals ≥ 2023-01-01: only the `OOS_START`
sentinel itself appears, used exclusively as an exclusive upper bound
(`DF.index < OOS_START`) for the causality probe's own pre-2023 restriction,
never as data consumed by any backtest.

## Idea, mechanism, and why it is genuinely new (pre-registered before running)

**Idea, one sentence.** Add VIX/DXY macro stress (`stress_z`, imported
unchanged from the shared `experiments/_macro_signal.py`) as a FOURTH,
latched vote inside `kelly_regime_v4`'s own three-anchor REGIME GATE —
never as a multiplicative haircut on the vol-targeted SIZE formula, which
is the disjoint conservative branch's job on this same round — testing the
literature's specific claim that VIX/DXY risk-off moves *lead* crypto price
moves, i.e. that a macro vote could flip the gate bearish faster than the
three slow 20/40/80-day price anchors can on their own.

**Constraint attacked.** INFO (one price series). VIX and DXY are two
genuinely new, price-independent information channels — the second
attempt at INFO this project has made, after on-chain metrics (B-07, R-44,
both branches NEGATIVE for reasons unrelated to data quality).

**Not a duplicate of, cited precisely:**
- R-44's `kelly_regime_v10_hashribbon_vote.py` (this file's own structural
  template): same architecture family — a latched 4th vote,
  precision-weighted into the vote-*generation* mechanism, v3's SIZE
  formula never touched — but a different signal (hash-ribbon miner
  capitulation, price-independent via mining economics) and, crucially, the
  **opposite sign discipline**: R-44's vote only ever pushes exposure UP
  (recovery = bullish); this file's vote only ever pushes exposure DOWN
  (stress = bearish veto-leaning), matching the spillover literature's
  risk-off-specific claim rather than a symmetric confirmation signal.
- L-04/L-01 (`kelly_regime_v4`, the incumbent): `macro_weight → 0` recovers
  v4 exactly — verified as an explicit identity check in `causality()`
  (max|diff| = 0.0 across the full `target` series), not merely asserted.
- R-08 (cited in B-07's own standing warning): a *better* volatility
  forecast made this strategy family WORSE by de-levering more promptly
  into BTC's highest-forward-Sharpe high-vol states (Baur & Dimpfl 2018's
  inverse leverage effect). This file's mechanism respects that sign
  discipline throughout: the macro vote never touches the SIZE/vol-target
  formula at all, only the vote fraction that feeds it, and only in the
  literature's risk-off direction.
- The disjoint parallel conservative branch this round
  (`kelly_regime_v14_macro_brake.py`): architecturally different by
  design — a continuous multiplicative haircut on SIZE vs. this file's
  additive vote input into the GATE. Not read, not coordinated with.

## Mechanism, precisely

`stress_z = 0.5·vix_z + 0.5·dxy_mom_z` (unchanged import; positive =
elevated fear and/or dollar strengthening = risk-off). A latched macro
vote, `macro_vote ∈ {0,1}`: fires to 0 ("stress") only when `stress_z`
crosses **above** `thresh_hi = 1.0` (fixed a-priori, one trailing std,
never swept — the same discipline v4's own 1% anchor band receives);
re-arms to 1 ("calm", the default) only when `stress_z` falls back
**below** `thresh_lo = thresh_hi − gap`. `gap = 0.0` collapses to a single
memoryless threshold with no hysteresis dead-zone — the explicit negative
control this round's brief asked for. Absent macro data, `macro_vote`
defaults to 1 (no veto) — a candidate with no macro coverage recovers v4's
own anchor-only vote exactly.

Combined vote: `frac = (anchor_sum + macro_weight·macro_vote) / (3 +
macro_weight)`. Because `macro_vote` defaults to 1 and only ever moves to
0 on a confirmed above-threshold reading, it can **only ever pull `frac`
down** relative to the three anchors alone — never manufacture bullish
exposure the anchors would not already grant. This is a **regime-detection
mechanism**, not a scale multiplier: the vote is combined with the anchor
votes *before* the vol-targeting SIZE formula runs, so a macro flip changes
`frac` on the exact bar it latches, not smoothed or gated behind the SIZE
formula's own separate volatility hysteresis — the "faster than the
slowest anchor" property the mechanism claims to test.

## Configurations evaluated

**15 distinct configurations**, matching this project's established
counting convention (R-44, B-19): distinct candidate parameterizations,
not baseline/reference/diagnostic re-reads.
- **12** primary candidates (`KellyRegimeV14MacroLead`): `gap ∈ {0.0, 0.75,
  1.25}` × `macro_weight ∈ {0.15, 0.33, 0.5, 1.0}` (`gap=0.0` and
  `weight=1.0` are explicit negative controls — no hysteresis, and a naive
  unweighted 4-way average, respectively).
- **3** ablation candidates (`KellyRegimeV14MacroOverride`, the hard-veto
  simplification): `gap ∈ {0.0, 0.75, 1.25}`, no weight parameter.
- `thresh_hi = 1.0` is fixed throughout, never swept, so it is not counted
  as a searched axis.
- Diagnostic re-reads (v4/`buy_and_hold` benchmarks, train-window re-checks
  inside `select()`, the plateau table, causality tamper probes, the
  exposure-artifact check, ETH control runs) are not separately counted,
  per the R-42/R-43/R-44/R-50/B-19 convention.

## Failure mode (1): does the macro vote actually LEAD the price gate?

Checked directly by comparing flip **timestamps** over inner-train +
inner-validation (2017–2022), not aggregate Sharpe, per the round's
explicit instruction — and this caught a real bug along the way: the first
version of the daily-transition dedup logic used
`~is_target.shift().fillna(False)`, which upcasts to object dtype and makes
`~` do **bitwise**, not logical, negation on Python `bool` objects
(`~True == -2`, truthy) — every day after a genuine stress onset was
miscounted as a fresh onset, inflating "12 real macro episodes" into "41
onsets" including runs of consecutive calendar days. Fixed with
`shift(fill_value=False)`, which stays boolean-dtype; the corrected count
(12 episodes at `gap=0.75`) independently matches `descriptive()`'s
bar-level flip count computed a completely different way. Flagging this for
any future session that reuses `.shift().fillna(...)` on a boolean Series.

With the fix, using the pre-registered primary config (`thresh_hi=1.0,
gap=0.75`), 12 macro bear-onset episodes were matched to the nearest onset
of two reference series within a ±180-day window:

| comparison | matched episodes | macro leads | median lead (days) |
|---|---|---|---|
| vs. fastest single (20d) anchor | 12 | 7/12 (58%) | +4.5 |
| vs. 3-anchor MAJORITY (the actual gate-flip proxy) | 12 | 4/12 (33%) | **−5.5** |

**The mechanism's core claim does not hold against the metric that
actually matters.** Against the single fastest anchor, macro leads a bare
majority of the time by a few days — mildly encouraging in isolation. But
against the 3-anchor *majority* vote — the thing that actually determines
whether v4's gate itself flips — macro leads only 4 of 12 times, and the
**median offset is negative**: on net, across this project's available
stress episodes, the macro vote tends to flip at the same time as or
slightly *after* the price gate, not before it. This is failure mode (1)
as pre-registered, and it already explains why the mechanism does not pay
off below.

## Inner-train (sweep, spot, 12 primary configs)

| candidate | final | Sharpe | max DD |
|---|---|---|---|
| `buy_and_hold` | $29,803 | 1.38 | 84.1% |
| `kelly_regime_v4` | $18,477 | 2.03 | 43.3% |
| primary (`gap=0.75, w=0.33`) | $21,291 | 2.11 | 39.2% |
| best in-train (`gap=0.75, w=1.00` unweighted) | $22,591 | 2.12 | 39.0% |
| worst in-train (`gap=1.25, w=0.15`) | $18,299 | 2.03 | 44.8% |

Every one of the 12 configurations beats v4 on inner-train, by a modest
margin (Sharpe 2.03–2.13) — the standard shape of an in-sample-favorable
elaboration this project has seen fail to replicate before (R-40, R-46).

## Inner-validation vs v4 (both markets, all 12 primary configs)

| candidate | market | final | Sharpe | max DD |
|---|---|---|---|---|
| `kelly_regime_v4` (control) | spot | $998 | 0.14 | 33.2% |
| `kelly_regime_v4` (control) | futures 5x | $1,064 | 0.25 | 32.3% |
| primary (`gap=0.75, w=0.33`) | spot | $948 | 0.06 | 37.5% |
| primary (`gap=0.75, w=0.33`) | futures 5x | $879 | −0.07 | 45.1% |
| best-by-selection-rule (`gap=0.0, w=0.15`) | spot | $982 | 0.12 | 34.7% |
| best-by-selection-rule (`gap=0.0, w=0.15`) | futures 5x | $948 | 0.05 | 41.0% |

**No configuration in the 12-cell grid beats v4 on inner-validation spot
Sharpe.** The parameter-neighbourhood table (all 12 cells, spot):

| gap | w=0.15 | w=0.33 | w=0.50 | w=1.00 |
|---|---|---|---|---|
| 0.00 | 0.12 | 0.09 | 0.11 | 0.08 |
| 0.75 | 0.10 | 0.06 | 0.11 | 0.08 |
| 1.25 | 0.10 | 0.04 | 0.04 | −0.04 |

v4's own control Sharpe (0.14) sits **above every single cell** — this is
not a peak-vs-plateau question; the entire searched region underperforms
the incumbent. Drawdown is also worse in 11 of 12 spot cells and in most
futures cells (up to 45–46% vs v4's 32–33%) — the opposite of the tail
protection the mechanism was built to add. This alone fails the promotion
bar's first clause (must beat v4 by more than the ±0.2 noise floor, or show
a drawdown/tail improvement) well before any holdout consideration, and is
consistent with the lead-time finding above: a vote that does not reliably
arrive early cannot buy the tail protection it was designed for.

## Failure mode (4): ablation — precision-weighted average vs. the simplest baseline

Per the round's pre-registered instruction to check honestly whether the
extra averaging machinery earns its keep (the R-40/R-46 pattern), the
primary weighted-average candidate was compared against
`KellyRegimeV14MacroOverride` — no weight parameter, no averaging: a hard
veto (`frac=0` while `macro_vote==0`, v4's own anchor average otherwise) —
at matching `gap` values, both inner splits, both markets:

| gap | split | market | weighted-avg Sharpe | override Sharpe | Δ(avg−override) |
|---|---|---|---|---|---|
| 0.00 | VALID | spot | 0.09 | **0.34** | −0.250 |
| 0.00 | VALID | futures 5x | 0.03 | **0.39** | −0.360 |
| 0.75 | VALID | spot | 0.06 | **0.32** | −0.263 |
| 0.75 | VALID | futures 5x | −0.07 | **0.41** | −0.478 |
| 1.25 | VALID | spot | 0.04 | 0.02 | +0.018 |
| 1.25 | VALID | futures 5x | −0.11 | **0.14** | −0.244 |

(TRAIN-split deltas are smaller but point the same direction: −0.02 to
−0.15 favoring the override in 5 of 6 train cells.)

**The elaboration does not earn its keep — it actively hurts.** The hard
override beats the precision-weighted average in 10 of 12 matched cells,
often by 0.25–0.48 Sharpe, comfortably outside the ±0.2 noise floor. This
confirms the pre-registered failure pattern this round was told to watch
for explicitly.

**A genuinely new, unrushed observation, reported honestly rather than
chased:** the override ablation's own inner-validation numbers
(`gap=0.0`: spot Sharpe 0.34/DD 26.4%, futures Sharpe 0.39/DD 27.4%; both
comfortably better than v4's 0.14/33.2% and 0.25/32.3%) are more promising
than anything the primary candidate produced. This was **not** the
pre-registered candidate for this round — it surfaced only as the
ablation's comparison arm — and has not been carried through this file's
own ETH falsification or a dedicated causality/plateau check as its own
subject. It is flagged below as a possible lead for a future session, not
promoted or further pursued here: promoting on a side-observation inside a
round pre-registered for a different mechanism is exactly the
goalpost-moving ROUTINE.md warns against.

## Exposure-artifact check

R² of the primary candidate's `target` series against a mean-notional-
matched flat rescale of v4's own `target`, inner-validation, both markets:

| market | mean\|v4\| | mean\|cand\| | alpha | R² | raw corr | verdict |
|---|---|---|---|---|---|---|
| spot | 0.289 | 0.321 | 1.108 | 0.9396 | 0.9915 | genuinely different exposure shape |
| futures 5x | 0.289 | 0.321 | 1.108 | 0.9396 | 0.9915 | genuinely different exposure shape |

**PASS** (R² < 0.95 threshold) — the candidate is not a relabeled flat
rescale of v4's own exposure, even though it fails to beat v4. It fails on
its own (unhelpful) merits, not as an exposure-level artifact.

## Causality probe (unregistered strategy, no CI coverage)

Two independent pathways tampered separately and together, on strictly
pre-2023 bars: price OHLCV (×3 / ÷3 multiplicative tamper, the project's
standard) and, new to this branch, the macro `stress_z` pathway itself —
the raw `spx/vix/dxy` CSVs copied into a temp directory and multiplied by
50× / divided by 50× from the tamper's calendar day forward (never
touching the real `data/` directory), verifying the strategy's own
`compute_macro_stress(df, data_dir)` fallback path is exercised correctly.

| probe | decisions at/before cut | `target`/`v14_frac`/`v14_macro_vote`/`v14_anchor_sum` max\|diff\| before cut |
|---|---|---|
| PRICE tamper | PASS | 0.000e+00 (all 4 columns) |
| MACRO tamper (new pathway) | PASS | 0.000e+00 (all 4 columns) |
| both at once | PASS | 0.000e+00 (all 4 columns) |
| identity check (`macro_weight=0` ≡ v4) | — | max\|diff\| = 0.000e+00, PASS |

No lookahead detected on either information pathway, and the additive-vote
mechanism recovers v4 exactly at zero weight, as designed.

## ETH falsification test (pre-registered rule, fixed before running)

ETH-USD Bitfinex spot (2016-03 → 2019-12-31) against macro coverage
(2016-06 →, 0 NaN in the overlap) — the standard pre-2020 falsification
window this project always uses. Since VIX/DXY are market-wide, not
BTC-specific, the pre-registered rule was: if the candidate underperforms
v4 on ETH, or is visibly worse on ETH than on the identical-pipeline BTC
control, this direction fails — and an ETH-only failure must be reported,
not hidden.

| config | market | BTC ratio (cand/v4) | ETH ratio (cand/v4) | flag |
|---|---|---|---|---|
| all 12 configs | spot | 0.96×–1.19× | **0.85×–0.99×** | **FAIL** (every config) |
| all 12 configs | futures 5x | 0.71×–1.14× | 0.74×–1.24× | mostly ok |

**FAIL on spot, the project's primary market.** Every single one of the 12
configurations underperforms v4 on ETH spot (ratio strictly below 1.0 in
all 12 cells), while several of the identical configurations equal or beat
v4 on the BTC control (up to 1.19×) — an asset-specific pattern a
genuinely asset-agnostic macro signal should not produce if the mechanism
were real. The futures-market picture is more mixed and often favorable,
but per this project's spot-primary convention and the literal
pre-registered rule (which the script itself evaluates and prints), the
overall verdict is **FAIL**.

## Verdict: NEGATIVE

The primary candidate — VIX/DXY macro stress as a precision-weighted 4th
vote in `kelly_regime_v4`'s regime gate — is rejected on **four
independent grounds**, any one of which was pre-registered as sufficient:

1. **Fails the lead-time check.** Against the 3-anchor majority vote that
   actually determines the gate's own flip, the macro vote leads only 4 of
   12 matched episodes (33%), with a negative median offset (−5.5 days) —
   it does not reliably arrive early enough to buy the tail protection the
   mechanism was designed for.
2. **Fails to beat v4 on inner-validation anywhere in the 12-cell grid** —
   every cell's spot Sharpe sits below v4's control, and drawdown is worse
   in 11 of 12 spot cells. This is not a peak-vs-plateau problem; the
   entire region underperforms.
3. **Loses to its own simplest ablation** in 10 of 12 matched cells,
   often by 0.25–0.48 Sharpe — the precision-weighted averaging machinery
   this file built is not earning its keep, the exact R-40/R-46 pattern
   this round was told to watch for.
4. **Fails the pre-registered ETH falsification rule on spot** — every
   configuration underperforms v4 on ETH while several beat v4 on BTC, an
   asset-specific signature that should not appear from a market-wide,
   asset-agnostic signal if the mechanism were genuine.

It **passes** both integrity checks that were run: the exposure-artifact
check (R²=0.94, a genuinely different exposure shape, not a relabeled
flat rescale) and the causality probe (0.0 lookahead on both the price and
the new macro information pathway, plus an exact `macro_weight=0` identity
recovery of v4). The mechanism is implemented correctly; it is simply not
useful, and the lead-time check gives a specific, mechanistic reason why
not: on this project's available stress episodes, VIX/DXY stress does not
reliably arrive before BTC's own price-anchor gate does.

Per ROUTINE.md's own instruction not to spend the holdout on a candidate
that already failed pre-registered gates, **the 2023+ holdout was never
read.**

**One-line lesson:** market-wide macro stress (VIX/DXY) does not lead
`kelly_regime_v4`'s own price-anchor gate in this project's available
stress episodes (2018, 2020-03, 2022) — the median timing is a wash-to-slight-lag
against the majority-anchor flip that actually matters, so adding it as a
faster gate input costs Sharpe and drawdown rather than buying tail
protection, and a simple hard-override ablation of the same underlying
vote consistently outperforms the more elaborate precision-weighted
average this file was pre-registered to test.

**Next step, for a future session (not pursued here):** the override
ablation's own inner-validation numbers (spot Sharpe up to 0.34–0.39 vs
v4's 0.14, DD down to 26–27% vs 33%) are a genuinely new, unvetted lead —
but it surfaced only as this round's comparison arm, was never the
pre-registered candidate, and has not been run through its own lead-time
check, ETH falsification, or plateau neighbourhood. A future round should
treat "hard macro veto, no averaging" as its own idea, pre-register its own
falsification test before running anything further, and check specifically
whether its apparent edge also survives the same lead-time and ETH tests
that killed the averaged-vote version here — a mechanism whose own vote
timing does not lead the gate it overrides is not an obvious candidate for
a stronger result under a blunter combination rule, and that tension
should be resolved before any promotion is considered.
