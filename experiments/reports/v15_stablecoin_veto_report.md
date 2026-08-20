# kelly_regime_v15_stablecoin_veto — R-54 NOVEL branch (08-20)

Unregistered experiment. Code: `experiments/kelly_regime_v15_stablecoin_veto.py`,
signal module `experiments/_stablecoin_signal.py`, fetch script
`scripts/fetch_stablecoin_supply.py`, data `data/stablecoin_supply_daily.csv.gz`,
additive loader functions `load_stablecoin_supply`/`align_stablecoin_causal`
appended to `src/tradebot/data.py` (nothing existing in that file edited or
removed). Not `@register`ed, not auto-discovered, nothing committed by this
branch's own choice — a human operator merges and commits after both R-54
branches report. This branch does not touch `kelly_regime_v4.py`,
`kelly_regime_v3.py`, `kelly_regime.py`, `docs/LEDGER.md`, or the disjoint
parallel CONSERVATIVE branch's files (`experiments/kelly_regime_v15_macro_veto.py`,
`experiments/reports/v15_macro_veto_report.md` — neither read, neither touched).
All evaluation below is restricted to inner-train (2017-01-01 → 2020-12-31),
inner-validation (2021-01-01 → 2022-12-31), and the standard pre-2020 ETH
falsification pair (compared against a full 2017–2022 BTC "control", matching
R-53's own `eth()` convention exactly). **The 2023+ holdout was never read** —
grep proof at the bottom of this report.

## Idea, mechanism, and why it is genuinely new (pre-registered before running)

**Idea, one sentence.** Aggregate USDT circulating-supply deceleration or
contraction — a proxy for dollar liquidity actually leaving the crypto
trading system — as a hard, unweighted veto (`frac = 0` while latched
"stress") on `kelly_regime_v4`'s own 3-anchor price-gate vote, testing
whether a *crypto-native* liquidity signal can do what R-53's *external*
VIX/DXY signal could not: actually lead the price-anchor gate rather than
lag it.

**Constraint attacked.** INFO. This is the THIRD distinct information
channel this project has tried: on-chain chain-activity metrics (B-07,
R-44 — active addresses/hash rate, a supply-side/usage signal about BTC's
own network) and external macro (R-53 — VIX/DXY, indices describing the
rest of the financial system) both failed. Aggregate stablecoin supply is
neither: it is crypto-native (unlike R-53) but describes capital FLOW
through the trading system's own on-ramp/off-ramp, not chain ACTIVITY
(unlike B-07).

**Not a duplicate of, cited precisely:**
- B-07/R-44's on-chain branches: BTC's own active-address/hash-rate
  activity, a supply-side/usage signal about the chain BTC trades on —
  says nothing about how much dollar liquidity is currently inside the
  trading system.
- R-53's macro branches: VIX/DXY are external, indirectly-correlated
  indices — R-53's own lead-time check found this indirection costs real
  time (median lag −5.5 days). Stablecoin supply is the dollar balance
  literally sitting inside the trading system, not an outside proxy.
- L-12/`harsanyi_crowd`: a price-*derived* crowding signal. Stablecoin
  supply never touches BTC/ETH OHLCV at all.
- **B-21** (this project's own backlog item, filed by R-53): the hard,
  unweighted macro veto that beat v4 on inner-validation but was never
  lead-time-tested or pre-registered. This file reuses B-21's
  *architecture* (hard override, no precision-weighted averaging)
  deliberately, fed by a different signal — so that any difference from
  R-53's result is attributable to the signal, not the combination rule.
  B-21 itself (VIX/DXY-fed) is this round's CONSERVATIVE branch's job,
  not this one's.

## Exact feature formula and sign hypothesis (pre-registered, full derivation in `_stablecoin_signal.py`)

`growth_14d = log(supply_t) - log(supply_{t-14})` — 14-calendar-day log
growth of aggregate USDT circulating supply. `stablecoin_stress_z = -1 *
zscore(growth_14d, trailing 365d, min_periods=60)` — positive means growth
is unusually slow or supply is contracting (risk-off), matching R-53's
sign convention. **Both windows (14-day growth, 365-day z-score) are fixed
a-priori and never swept anywhere in this file** — the same discipline
`_macro_signal.py` used for VIX/DXY.

**Sign hypothesis, stated before any code ran:** elevated
`stablecoin_stress_z` should LEAD the 3-anchor majority price-gate's own
bearish flip, because capital leaving the system is a cause of the
subsequent price weakness the anchors react to, not merely a correlate.

**Data-scope decision, stated plainly.** USDT alone, for the entire
2017–2026 period. USDC's `SplyCur` is reachable and clean from this
environment but its real (non-placeholder) minted supply only starts
2018-09-25 — USDT is overwhelmingly dominant through most of the
2017–2022 span this round's episodes fall in. Combining the two would
either treat USDC's pre-launch period as an implicit zero (a literal
fact, but easy to mistake for backfilling) or introduce a structural
discontinuity the day USDC's real history starts. USDT alone is the
plainer, harder-to-misread of the two options this round's brief
explicitly sanctioned. USDC data was fetched and inspected only to
confirm reachability; not used anywhere in the signal.

**Named risk, stated before any code ran:** it is a fully legitimate
possibility that this signal ALSO lags rather than leads — daily on-chain
data feeding a 5-minute-bar strategy has a coarser native cadence than
what it is being asked to lead. If the lead-time check found a lag, that
was to be reported plainly, not explained away.

**Pre-registered falsification test, centerpiece of this round.** Compare
flip *timestamps* — not aggregate Sharpe — of the latched stablecoin-veto
vote against the 3-anchor majority vote, over the stress episodes
available in inner-train + inner-validation (2018, 2020-03, and the 2022
bear), exactly as R-53's `leadtime()` did, using `shift(fill_value=False)`
(not `.shift().fillna(False)`, which R-53 found silently does bitwise
negation on object-dtype booleans — this file starts from the corrected
version, not the bug).

**UST/Terra collapse (May 2022) scope decision, stated before running
anything:** IN SCOPE as a labeled *secondary* check, not pooled with the
primary matched-episode set. UST/LUNA was an algorithmic stablecoin — a
mechanically different instrument from USDT — but the panic produced real
stress on USDT itself (brief secondary de-peg, elevated redemption
pressure), so it is a genuine test of whether USDT supply reacted with
useful lead time to a stablecoin-specific liquidity event.

## Code reuse decision

The anchor-vote and latched-hysteresis-vote helper functions are
**duplicated** (not imported) from `kelly_regime_v14_macro_lead.py` — a
private, unregistered experiment from a prior round, not shared
infrastructure. `kelly_regime_v14_macro_lead.py` itself is not edited
anywhere in this session.

## Configurations evaluated

**9 distinct configurations** (`thresh_hi ∈ {0.75, 1.0, 1.25} × gap ∈
{0.0, 0.75, 1.25}`), the veto-sensitivity axis of the fixed hard-override
architecture — the signal formula itself (14d growth, 365d z-score) is
what stays fixed a-priori. No averaged-vote vs. hard-override ablation is
run in this file: the hard override *is* the primary candidate here, per
this round's instruction, as the direct continuation of R-53's own
ablation finding rather than a re-derivation of the weighted-average
architecture R-53 already rejected. Diagnostic re-reads (v4/`buy_and_hold`
benchmarks, train-window re-checks inside `select()`, the plateau table,
causality tamper probes, the exposure-artifact check, ETH control runs)
are not separately counted, per the R-42/R-44/R-53/B-19 convention.

## Centerpiece check: does the stablecoin vote actually LEAD the price gate?

Using the primary config (`thresh_hi=1.0, gap=0.75`), 12 stablecoin
bear-onset episodes over 2017–2022 were matched to the nearest onset of
two reference series within a ±180-day window:

| comparison | matched episodes | stablecoin leads | median lead (days) |
|---|---|---|---|
| vs. fastest single (20d) anchor | 12 | 9/12 (75%) | +15.0 |
| vs. 3-anchor MAJORITY (the actual gate-flip proxy) | 12 | 9/12 (75%) | **+16.5** |

**This is the opposite result from R-53's VIX/DXY signal.** Against the
metric that actually matters — the 3-anchor majority that determines
whether v4's own gate flips — the stablecoin-supply vote leads 9 of 12
matched episodes, with a **positive** median offset of +16.5 days. Two
episodes lead by large margins that overlap genuine crashes (2017-05-03
leading by 53 days ahead of the early-2017 correction's majority-anchor
onset; 2021-06-12 leading by 39 days ahead of the mid-2021 correction).
Three episodes lag (2018-02-07 by 15 days, 2018-10-25 by 15 days,
2022-01-15 by 34 days) — not every episode leads, but the aggregate
direction and magnitude are genuinely different from R-53's median −5.5
days.

**UST/Terra (May 2022), secondary check.** The stablecoin vote latched to
"stress" on 2022-05-13 — 4 days after UST's initial de-peg (May 9) and a
full month before BTC's price low inside the window (2022-06-15,
$20,111). `stablecoin_stress_z` itself peaked later still (2022-05-24,
z=3.68), after the vote had already latched. Genuine lead time over the
price bottom on this specific event, consistent with the primary finding.

**This is a real, novel, positive finding on the axis R-53 was built to
test — but it does not, by itself, rescue the trading result below.**

## Inner-train (sweep, spot, 9 configs)

| candidate | final | Sharpe | max DD |
|---|---|---|---|
| `buy_and_hold` | $29,803 | 1.38 | 84.1% |
| `kelly_regime_v4` | $18,477 | 2.03 | 43.3% |
| thresh=1.25, gap=0.75 (best) | $19,075 | 2.05 | 41.8% |
| thresh=0.75, gap=0.75 (worst) | $4,924 | 1.44 | 28.1% |

Only 1 of 9 configs (the loosest threshold/gap pair) edges out v4 on
inner-train; every tighter configuration underperforms, some sharply —
the veto is firing often enough at low thresholds to cost real return
even in-sample.

## Inner-validation vs v4 (both markets, all 9 configs)

| candidate | market | final | Sharpe | max DD |
|---|---|---|---|---|
| `kelly_regime_v4` (control) | spot | $998 | 0.14 | 33.2% |
| `kelly_regime_v4` (control) | futures 5x | $1,064 | 0.25 | 32.3% |
| best (thresh=0.75, gap=1.25) | spot | $1,011 | 0.13 | 27.5% |
| best (thresh=0.75, gap=1.25) | futures 5x | $1,034 | 0.18 | 25.5% |
| worst (thresh=0.75, gap=0.75) | spot | $722 | −0.61 | 48.2% |
| worst (thresh=0.75, gap=0.75) | futures 5x | $751 | −0.51 | 45.8% |

**No configuration beats v4 on inner-validation Sharpe.** The single best
cell is a statistical wash (Δ Sharpe −0.01 spot, and it is not even the
same config that was near-best on inner-train) with a real drawdown
improvement (33.2%→27.5%, −5.7pp) — but 8 of 9 cells are decisively worse
on both Sharpe (as low as −0.61) and drawdown (as high as 48.8% vs v4's
33.2%). Full parameter-neighbourhood table (spot Sharpe):

| thresh | gap=0.00 | gap=0.75 | gap=1.25 |
|---|---|---|---|
| 0.75 | −0.53 | −0.61 | **0.13** |
| 1.00 | −0.45 | −0.45 | −0.11 |
| 1.25 | −0.17 | −0.20 | −0.22 |

**Not a plateau.** The neighbourhood around the single near-tied cell is
steep — moving one gap-step away (thresh=0.75, gap=0.75) costs 0.74
Sharpe, and every other direction from any cell in the grid is negative.
This fails the promotion bar's plateau requirement on its own, independent
of the Sharpe result.

## Falsification test: BTC full-history control vs ETH (pre-registered rule)

ETH-USD Bitfinex spot (2016-03 → 2019-12-31) against USDT-supply coverage
(2017-01-01 →, overlap 2017-03-16 → 2019-12-31, 0 NaN). Same rule as
R-53's `eth()`: candidate must not be visibly worse on ETH than on the
identical-pipeline BTC control (here, BTC's full 2017–2022 span, matching
R-53's own `eth()` convention exactly — not merely the pre-2020 slice).

| config | market | BTC ratio (cand/v4) | ETH ratio (cand/v4) | flag |
|---|---|---|---|---|
| all 9 configs | spot | 0.198×–0.886× | 0.653×–1.023× | **ok** (no ETH-specific weakness) |
| all 9 configs | futures 5x | 0.131×–0.902× | 0.703×–1.025× | **ok** |

**No outright FAIL by the pre-registered ETH-vs-BTC differential rule**
— every configuration's ETH ratio is at or above its BTC ratio, so there
is no asset-specific degradation the way R-53's averaged macro vote showed.
**But this is not a clean pass in substance.** The reason no config fails
the *differential* test is that every single configuration already
underperforms v4 by a wide margin on the BTC side of the comparison too
(ratios 0.13×–0.90×, always below 1.0×, across all 9 configs and both
markets, over the full 2017–2022 span) — the same "loses to v4 on its own
control before ETH is even informative" signature this project has now
seen repeatedly (R-37, R-38, R-40, R-46). The best single cell
(thresh=1.25, gap=0.75, futures) still only reaches 0.902× of v4's
BTC balance.

## Exposure-artifact check

R² of the primary candidate's `target` series against a mean-notional-
matched flat rescale of v4's own `target`, inner-validation, both markets:

| market | mean\|v4\| | mean\|cand\| | alpha | R² | raw corr | verdict |
|---|---|---|---|---|---|---|
| spot | 0.289 | 0.202 | 0.698 | 0.6091 | 0.7831 | genuinely different exposure shape |
| futures 5x | 0.289 | 0.202 | 0.698 | 0.6091 | 0.7831 | genuinely different exposure shape |

**PASS**, well clear of the 0.95 threshold — the candidate is not a
relabeled flat rescale of v4's own exposure; it fails on its own
(unhelpful) merits.

## Causality probe (unregistered strategy, no CI coverage)

Two independent pathways tampered separately and together, on strictly
pre-2023 bars: price OHLCV (×3 / ÷3, the project's standard) and the new
stablecoin-supply pathway (the raw CSV copied into a temp directory and
multiplied by 50× / divided by 50× from the tamper's calendar day
forward, never touching the real `data/` directory).

| probe | decisions at/before cut | `target`/`v15_frac`/`v15_stable_vote`/`v15_anchor_sum` max\|diff\| before cut |
|---|---|---|
| PRICE tamper | PASS | 0.000e+00 (all 4 columns) |
| STABLECOIN tamper (new pathway) | PASS | 0.000e+00 (all 4 columns) |
| both at once | PASS | 0.000e+00 (all 4 columns) |
| identity check (`enabled=False` ≡ v4) | — | max\|diff\| = 0.000e+00, PASS |

No lookahead on either information pathway; the override mechanism
recovers v4 exactly when disabled, as designed.

## Verdict: NEGATIVE

The stablecoin-supply hard veto is rejected on the promotion bar, on
multiple independent grounds — but the mechanism's *central premise* is
partially confirmed for the first time in this project's INFO research
line, and that distinction matters for how this negative should be read:

1. **Lead-time check PASSES, unlike R-53.** Against the 3-anchor majority
   that actually flips v4's gate, the stablecoin-stress vote leads 9 of 12
   matched episodes (75%), median +16.5 days — the opposite of R-53's
   VIX/DXY finding (33%, median −5.5 days). The crypto-native-liquidity
   hypothesis this round set out to test is genuinely supported on this
   specific axis.
2. **Does not translate into a working strategy.** No configuration beats
   v4 on inner-validation Sharpe; the one near-tied cell sits inside the
   noise floor and is not part of a plateau (its immediate gap-neighbour
   loses 0.74 Sharpe). 8 of 9 cells are decisively worse on both Sharpe
   and drawdown.
3. **Loses to v4 in an absolute sense across nearly the whole 2017–2022
   span, on every configuration, both markets** (BTC "control" ratios
   0.13×–0.90×, best cell 0.90×) — the ETH differential test technically
   passes only because there is no *additional* ETH-specific degradation
   on top of that already-large absolute underperformance.
4. **Passes both integrity checks**: exposure-artifact R²=0.61 (a
   genuinely different exposure shape, not a relabeled flat rescale) and
   causality (0.0 lookahead on price, the new stablecoin pathway, and both
   combined, plus an exact `enabled=False` identity recovery of v4).

**The mechanistic reason the confirmed lead-time result still fails to
pay off**, read off the data directly: the threshold sensitive enough to
catch real stress episodes early (thresh=0.75, the tightest) also fires
on far more transient supply fluctuations that are not followed by
genuine price weakness (24 stress-onset events at thresh=0.75/gap=0.0 vs.
only 12 at the primary thresh=1.0/gap=0.75 — see `descriptive()`), so standing flat through
every one of those false alarms costs more Sharpe than the genuine early
exits recover. Loosening the threshold to reduce false positives
(thresh=1.25) recovers most of the lost Sharpe but also gives back most
of the timing advantage, converging toward v4's own behavior rather than
improving on it. **Timing was the axis R-53 found broken; this round
fixes it, and finds precision/specificity is a second, independent axis
that also has to hold — fixing one does not automatically fix the other.**

Per ROUTINE.md's own instruction not to spend the holdout on a candidate
that already failed pre-registered gates, **the 2023+ holdout was never
read.**

**One-line lesson:** a genuinely crypto-native liquidity signal (aggregate
USDT supply deceleration) *can* lead `kelly_regime_v4`'s own price-anchor
gate — confirming this round's central hypothesis and reversing R-53's
timing failure — but a hard veto built on it still loses to the incumbent
because the same threshold that buys useful lead time also fires on too
many transient supply wobbles that are not followed by real price
weakness; INFO's binding constraint in this project is turning out to be
precision as much as timing, not timing alone.

## Configurations evaluated

**9** (this branch's total; the parallel CONSERVATIVE branch reports its
own count separately — this round's project-level trials count is the sum
of both, per ROUTINE.md's parallelism rules, to be totaled by the operator
when both reports are in).

## Holdout

**+0.** Never read. Grep proof, every date literal ≥2023 in this branch's
three new files, with call-site context:

```
=== kelly_regime_v15_stablecoin_veto.py ===
198:OOS_START = "2023-01-01"                 # never read in this file
678:    strictly pre-2023 bars."""
681:    pre2023 = DF[DF.index < OOS_START]
682:    df = pre2023.iloc[-300_000:].copy()
797:    frames = {"BTC (control)": DF[DF.index < OOS_START], "ETH (test)": eth_df}

=== _stablecoin_signal.py ===
(no ≥2023 date literals — only literature-citation years 2025/2026 in prose)

=== fetch_stablecoin_supply.py ===
(no ≥2023 date literals in code; --end defaults default to today via
 datetime.now(), and --start/--end examples in the module docstring use
 2017-01-01/2018-01-01/2026-08-20, all either <2023 or exploratory-only
 usage examples, never executed by this branch's own report)

=== src/tradebot/data.py (stablecoin additions only) ===
(no ≥2023 date literals in `load_stablecoin_supply`/`align_stablecoin_causal`
 or their docstrings; grep hits elsewhere in the file are pre-existing code
 this branch did not write or touch)
```

Both `OOS_START` call sites are exclusive upper bounds restricting the
causality probe and the `eth()` control comparison to strictly pre-2023
bars — never a data read past the boundary. Independently verifiable by
re-running `grep -n "202[3-9]" experiments/kelly_regime_v15_stablecoin_veto.py
experiments/_stablecoin_signal.py scripts/fetch_stablecoin_supply.py`.

## Test suite

`pytest` (from `.venv`): **457 passed**, unchanged from the session's
starting count — the additive `load_stablecoin_supply`/
`align_stablecoin_causal` functions in `src/tradebot/data.py` did not
require or break any existing test, and no new tests were added (not
required for a NEGATIVE, unregistered experiment per ROUTINE.md).

## Next step

The confirmed lead-time result is worth keeping as a standing fact for
any future crypto-native-liquidity attempt on this codebase, but this
specific mechanism (hard veto, single-signal threshold) is not worth
re-trying without a materially different way of separating genuine stress
onsets from transient supply noise — e.g. requiring the vote to persist
for a minimum number of days before it can veto (a magnitude-*and*-
duration filter, rather than magnitude alone), or combining the
stablecoin signal with the existing price anchors as a *confirming*
speed-up rather than a unilateral override (closer to R-53's originally
pre-registered precision-weighted-average architecture, which this round
deliberately did not re-test, but which might behave differently fed by
a signal that actually leads rather than lags). Filed as a candidate
backlog item for a future session; not pursued further here per
ROUTINE.md's goalpost discipline.
