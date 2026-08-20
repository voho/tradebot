# kelly_regime_v16_stablecoin_persist — R-55 CONSERVATIVE branch (08-20)

Closes backlog item **B-22**'s first named next step. Unregistered
experiment. Code: `experiments/kelly_regime_v16_stablecoin_persist.py`.
Reuses `experiments/_stablecoin_signal.py`'s `compute_stablecoin_stress`
unchanged (imported, not duplicated); duplicates the anchor-vote and
latched-hysteresis helper functions from
`kelly_regime_v15_stablecoin_veto.py`, per that file's own precedent.
Not `@register`ed, not auto-discovered, nothing committed by this
branch's own choice — a human operator merges and commits after both
R-55 branches report. This branch does not touch
`kelly_regime_v15_stablecoin_veto.py`, `kelly_regime_v15_macro_veto.py`,
`kelly_regime_v14_macro_lead.py`, `kelly_regime_v4.py`,
`kelly_regime_v3.py`, `kelly_regime.py`, `docs/LEDGER.md`, or the
disjoint parallel NOVEL branch's files
(`experiments/kelly_regime_v16_stablecoin_confirm.py`,
`experiments/reports/v16_stablecoin_confirm_report.md` — neither read,
neither touched). All evaluation below is restricted to inner-train
(2017-01-01 → 2020-12-31), inner-validation (2021-01-01 → 2022-12-31),
and the standard pre-2020 BTC-control/ETH falsification pair. **The
2023+ holdout was never read** — grep proof at the bottom of this
report.

**Note on the environment's own checkpoint commit:** partway through
this session the working tree was auto-committed by the environment
(`cdd7687`, "R-55 checkpoint: B-22 branches in progress") — not a commit
this branch issued. It snapshots this branch's file and the parallel
branch's file as they stood at that moment; this branch has still not
run `git commit` itself and has not read the parallel branch's file.
Flagged here for the operator's awareness, not acted on further.

## Idea, mechanism, and pre-registration (written before any code ran)

**Mechanism, one sentence.** Transient day-to-day wobbles in aggregate
USDT supply growth reverse within a few days and carry no forward
information about price weakness, so requiring `stablecoin_stress_z` to
stay continuously above `thresh_hi` for a minimum number of consecutive
days (`persist_days`) before the latched vote is allowed to enter
"stress" — the entry side only; recovery/exit stays governed by the
existing `thresh_lo` hysteresis crossing, unfiltered, per R-54's own
diagnosis that the false positives are fast-reversing onsets, not slow
recoveries — should filter out most of the ~12–24 transient stress-onset
false positives R-54 found while preserving most of the genuine
multi-week early warning R-54 confirmed (9/12 matched episodes lead the
3-anchor majority, median +16.5 days).

**Constraint attacked.** INFO, same axis as R-54. This round introduces
no new information channel — it is a minimal architectural variant of
R-54's own hard veto, adding exactly one new free parameter
(`persist_days`) and changing nothing else about the combination rule,
the signal formula, or the sizing math underneath.

**Not a duplicate of, cited precisely:**
- `kelly_regime_v15_stablecoin_veto.py` (R-54 NOVEL, this file's direct
  ancestor): identical hard-override architecture and identical signal;
  the only difference is that entry into "stress" now requires
  `persist_days` of continuous confirmation. `persist_days=0` is this
  file's exact negative control, verified below to reproduce R-54's
  `_stable_vote` bit-for-bit.
- `kelly_regime_v15_macro_veto.py` (R-54 CONSERVATIVE, VIX/DXY-fed): not
  read, disjoint signal, disjoint files.
- This round's own disjoint NOVEL branch
  (`kelly_regime_v16_stablecoin_confirm.py`, presumably B-22's second
  named next step — a confirming, non-overriding combination rule): not
  read, not coordinated with, per ROUTINE.md's parallelism isolation
  rule.
- **B-22** itself: "a magnitude-*and*-duration filter... require the
  vote to persist for a minimum number of days before it can veto,
  rather than firing on any single-bar threshold crossing." This file is
  exactly that, the first of B-22's two named next steps.

**Sources.**
- Shu, Yu & Mulvey (2024), "Downside Risk Reduction Using Regime-
  Switching Signals: A Statistical Jump Model Approach," arXiv:2402.05272
  — already cited in this project's R-02 ledger row for its jump-penalty
  mechanism; cited here specifically for its independent use of a
  minimum-duration/persistence requirement before confirming a regime
  change, trading detection speed against false-positive rate — the
  general form of the idea this file tests concretely.
- Industry practice on stablecoin-flow persistence windows for
  confirmation commonly falls in a 3–7 day range (web research for this
  round) — motivation for this file's parameter-grid *center*
  (`PRIMARY_KW` uses `persist_days=3`), not a number copied blindly; the
  actual grid spans 0–14 days so the literature-motivated region is
  interior to the grid, not an edge choice.

**Pre-registered falsification test (stated before any code ran).**
(a) Does adding the persistence filter PRESERVE OR IMPROVE R-54's
lead-time-vs-3-anchor-majority result — checked explicitly across the
full `persist_days` grid at the primary thresh/gap, to see explicitly
whether/where the lead-time advantage collapses as confirmation delay
grows. (b) Does the best surviving configuration pass the same
pre-2020 BTC-control vs ETH falsification test R-54 used. Both must
hold, AND the candidate must beat v4 on inner-validation Sharpe outside
the ±0.2 noise floor with a plateau neighbourhood, before any holdout
read is considered.

**Pre-registered holdout decision rule, stated before any result was
read:** read the 2023+ holdout **if and only if** the primary candidate
(1) beats v4 on inner-validation Sharpe outside the ±0.2 noise floor,
(2) sits in a plateau neighbourhood (not a spike), (3) preserves a
genuine lead-time advantage per test (a), and (4) passes the ETH
falsification test per (b). If any one of these fails, report NEGATIVE
and do not read the holdout.

## Configurations evaluated

**24 configurations**: `THRESH_GAP_COMBOS` (3: primary `thresh_hi=1.0,
gap=0.75`; R-54's tightest/worst `thresh_hi=0.75, gap=0.0` — the config
with the most false-positive stress-onsets, 24, per R-54's own
`descriptive()`; and R-54's worst-scoring cell `thresh_hi=0.75, gap=0.75`,
inner-validation Sharpe −0.61) × `PERSIST_DAYS` (8: 0, 1, 2, 3, 5, 7, 10,
14). `thresh_hi`/`gap` are fixed exactly at R-54's values; `persist_days`
is the only new axis swept. Diagnostic re-reads (v4/`buy_and_hold`
benchmarks, train-window re-checks inside `select()`, the plateau table,
identity checks, causality tamper probes, the exposure-artifact check,
ETH control runs) are not separately counted, per the R-42/R-44/R-53/
R-54 convention.

## Step 0: mandatory identity checks (run first, before anything else)

| check | result |
|---|---|
| `persist_days=0` vs v15's original `_stable_vote`, all 3 thresh/gap combos | max\|diff\|=0.000e+00, 0/631,008 bars differing, all 3 — **PASS, bit-for-bit** |
| `thresh_hi=1e9` (veto never fires) vs v4 | max\|diff\|=0.000e+00 — **PASS** |
| `enabled=False` vs v4 | max\|diff\|=0.000e+00 — **PASS** |

The persist_days=0 negative control reproduces R-54's v15 output exactly
bit-for-bit, on all three thresh/gap combos — the strongest form of the
sanity check requested. Both v4-identity checks also pass exactly.

## Descriptive: does persistence actually shrink false-positive onset counts?

Stress-onset event counts, inner-train + inner-validation
(2017-01-01 → 2022-12-31):

| persist_days | primary (thresh=1.00, gap=0.75) | worst-tightest (thresh=0.75, gap=0.00) | worst-primary-gap (thresh=0.75, gap=0.75) |
|---|---|---|---|
| 0 | 12 | 24 | 13 |
| 1 | 12 | 23 | 13 |
| 2 | 12 | 22 | 13 |
| 3 | 12 | 22 | 13 |
| 5 | 9 | 19 | 11 |
| 7 | 7 | 16 | 10 |
| 10 | 7 | 15 | 9 |
| 14 | 5 | 11 | 8 |

The mechanism does reduce onset counts, monotonically, as persistence
grows — but **not at the literature-motivated 3–7 day center at the
primary threshold**: persist_days=0,1,2,3 all give the *identical* 12
onset events at the primary combo, meaning small persistence requirements
buy nothing there. Meaningful reduction only shows up at persist_days≥5,
and even at persist_days=14 the worst-tightest combo still shows 11
onsets — nowhere near the handful of genuine stress episodes (~3–4) this
project's stress-episode inventory identifies. Read plainly: much of the
transient noise in this signal persists for many days, not one bar, so a
short-to-moderate duration filter does not cleanly separate it from
genuine stress.

## Falsification test (a): lead-time vs persist_days, primary combo (thresh=1.00, gap=0.75)

3-anchor-majority bear-onset reference: 40 episodes (fixed, independent
of persist_days).

| persist_days | onsets | matched | leads/matched | median lead (days) |
|---|---|---|---|---|
| 0 (R-54 negative control) | 12 | 12 | 9/12 (75%) | **+16.5** |
| 1 | 12 | 12 | 8/12 (67%) | +10.5 |
| 2 | 12 | 12 | 8/12 (67%) | +11.5 |
| 3 | 12 | 12 | 7/12 (58%) | +7.0 |
| 5 | 9 | 9 | 4/9 (44%) | **−10.0** |
| 7 | 7 | 7 | 3/7 (43%) | −8.0 |
| 10 | 7 | 7 | 3/7 (43%) | −5.0 |
| 14 | 5 | 5 | 3/5 (60%) | +26.0 (n=5, noisy) |

**Falsification test (a) FAILS beyond a narrow, weak margin.** The
lead-time advantage erodes steadily from persist_days=0 to persist_days=3
(75%→58% leading, median +16.5d→+7.0d) and then **flips to net LAG** at
persist_days=5–10 (fewer than half the matched episodes lead; median lag
−5 to −10 days) — the reverse of R-54's own central finding. It recovers
to a positive median at persist_days=14, but on only 5 matched episodes
(too thin to trust, and the individual leads at that setting are highly
mixed: −28, −1, +53, +26, +46). The literature-motivated 3–7 day center
this round's own pre-registration named already sits inside the
degrading-to-negative region. This directly confirms the concern named in
this round's own assignment brief: a filter that delays confirmation too
long erases the lead-time advantage — here it erases it by persist_days≈5,
well inside the tested and literature-motivated range, not at some
extreme edge of the grid.

## Inner-train (sweep, spot, 24 configs)

Selected rows (full table in the script's stdout):

| candidate | final | Sharpe | max DD |
|---|---|---|---|
| `buy_and_hold` | $29,803 | 1.38 | 84.1% |
| `kelly_regime_v4` | $18,477 | 2.03 | 43.3% |
| primary, persist=0 (R-54 negative control) | $11,550 | 1.84 | 35.6% |
| primary, persist=3 (literature center) | $11,692 | 1.84 | 35.6% |
| primary, persist=14 (best in-sample) | $19,954 | 2.10 | 38.1% |
| worst-tightest, persist=0 | $7,579 | 1.62 | 47.0% |
| worst-primary-gap, persist=0 | $4,924 | 1.44 | 28.1% |

As persistence grows, in-sample results drift back *toward* (and
slightly past) v4 — exactly the "converges back toward v4 rather than
improving on it" pattern R-54 diagnosed for the *threshold* axis; here it
recurs on the *persistence* axis instead.

## Inner-validation vs v4 (both markets, all 24 configs)

| candidate | market | final | Sharpe | max DD |
|---|---|---|---|---|
| `kelly_regime_v4` (control) | spot | $998 | 0.14 | 33.2% |
| `kelly_regime_v4` (control) | futures 5x | $1,064 | 0.25 | 32.3% |
| best overall: primary, persist=14 | spot | $774 | **−0.38** | 44.5% |
| best overall: primary, persist=14 | futures 5x | $798 | −0.34 | 42.5% |
| primary, persist=3 (literature center) | spot | $764 | −0.43 | 45.2% |
| worst-tightest, persist=5 | spot | $688 | −0.60 | 50.7% |
| worst-primary-gap, persist=5 | spot | $692 | −0.70 | 50.4% |

**No configuration — none of the 24 — beats v4 on inner-validation
Sharpe.** The best cell across the entire grid (primary combo,
persist_days=14, i.e. nearly the loosest persistence setting tested)
still sits at Sharpe −0.38 vs v4's +0.14, a gap of 0.52 — more than twice
the ±0.2 noise floor, and *worse than R-54's own best cell* (0.13, at
thresh=0.75/gap=1.25, a combo not in this round's grid because it was not
one of R-54's flagged tight/false-positive-heavy configs). This is a
decisive rejection at the very first promotion-bar gate; the candidate
does not merely fail to clear the noise floor, it fails to reach v4 at
all, anywhere in the grid.

Overfitting-signature check, best cell (primary, persist=14): train
Sharpe 2.10 vs v4's 2.03 (candidate ahead in-sample); validation Sharpe
−0.38 vs v4's 0.14 (candidate 0.52 behind out-of-sample) — the same
train-wins/validation-loses signature this project's promotion-bar rule
exists to catch.

## Parameter-neighbourhood plateau check (spot, inner-validation Sharpe)

| thresh/gap combo | p=0 | p=1 | p=2 | p=3 | p=5 | p=7 | p=10 | p=14 |
|---|---|---|---|---|---|---|---|---|
| primary (1.00/0.75) | −0.45 | −0.45 | −0.43 | −0.43 | −0.51 | −0.54 | −0.48 | **−0.38** |
| worst-tightest (0.75/0.00) | −0.53 | −0.53 | −0.50 | −0.51 | −0.60 | −0.56 | −0.53 | −0.47 |
| worst-primary-gap (0.75/0.75) | −0.61 | −0.60 | −0.59 | −0.61 | −0.70 | −0.66 | −0.63 | −0.57 |

Every cell in the entire 24-cell grid is negative. There is no candidate
anywhere near v4's 0.14, so the plateau question is moot — there is no
peak worth checking for flatness because there is no cell that clears
the first bar (beating v4) at all. This is a clean, decisive negative,
not a marginal one.

## Falsification test (b): pre-2020 BTC control vs ETH

Not required for the verdict (gate (1) — beating v4 on inner-validation
Sharpe — already failed decisively for all 24 configs), but run in full
for completeness and report-template consistency, exactly as R-54 did.

| config set | market | BTC ratio (cand/v4) range | ETH ratio (cand/v4) range | flag |
|---|---|---|---|---|
| all 24 configs | spot | 0.151×–0.888× | 0.605×–1.035× | **ok** (no ETH-specific weakness by the differential rule) |
| all 24 configs | futures 5x | 0.151×–0.888× | 0.617×–1.055× | **ok** |

No outright FAIL by the pre-registered differential rule — same
signature R-54 found. But as R-54's report noted for its own result,
this is not a substantive pass: every configuration already underperforms
v4 on its own BTC control by a wide margin (ratios 0.15×–0.89×, always
below 1.0×, over the full 2017–2022 span), so the "no *additional*
ETH-specific degradation" reading is the only sense in which this
"passes." The best single cell (primary, persist=10, futures) still only
reaches 0.888× of v4's BTC balance.

## Exposure-artifact check

R² of the primary candidate's (`persist_days=3`) `target` series against
a mean-notional-matched flat rescale of v4's own `target`,
inner-validation, both markets:

| market | mean\|v4\| | mean\|cand\| | alpha | R² | raw corr | verdict |
|---|---|---|---|---|---|---|
| spot | 0.289 | 0.204 | 0.705 | 0.6103 | 0.7833 | genuinely different exposure shape |
| futures 5x | 0.289 | 0.204 | 0.705 | 0.6103 | 0.7833 | genuinely different exposure shape |

**PASS**, well clear of the 0.95 threshold. Nearly identical to R-54's
own R²=0.6091 at persist_days=0, as expected — persistence changes *when*
the veto fires, not its shape once fired.

## Causality probe (unregistered strategy, no CI coverage)

Two independent pathways tampered separately and together, on strictly
pre-2023 bars: price OHLCV (×3/÷3) and the stablecoin-supply pathway
(raw CSV, ×50/÷50 from the tamper day forward, in a temp directory —
never touching the real `data/` directory).

| probe | decisions at/before cut | `target`/`v16_frac`/`v16_stable_vote`/`v16_anchor_sum` max\|diff\| before cut |
|---|---|---|
| PRICE tamper | PASS | 0.000e+00 (all 4 columns) |
| STABLECOIN tamper (new pathway) | PASS | 0.000e+00 (all 4 columns) |
| both at once | PASS | 0.000e+00 (all 4 columns) |
| identity: `enabled=False` ≡ v4 | — | max\|diff\| = 0.000e+00, PASS |
| identity: `thresh_hi=1e9` (never fires) ≡ v4 | — | max\|diff\| = 0.000e+00, PASS |

No lookahead on either information pathway. The persistence filter's
`.resample("1D").last().ffill()` + backward-only `.rolling(window,
min_periods=window)` construction is causal by inspection, and the
tamper probe confirms it empirically.

## Verdict: NEGATIVE

The magnitude-and-duration filter fails decisively, on grounds more
severe than R-54's own hard-veto result, not milder:

1. **Falsification test (a) FAILS.** The pre-registered concern — that a
   filter delaying confirmation too long could erase the lead-time
   advantage entirely — is exactly what happens, and it happens inside
   the literature-motivated 3–7 day range this round's own pre-
   registration named as the grid's center, not at some extreme edge:
   the fraction of matched episodes leading falls from 75% (persist=0)
   to 58% (persist=3) to 44% (persist=5), and the median offset flips
   from +16.5 days of LEAD to −10.0 days of LAG by persist_days=5.
2. **Promotion-bar gate (1) fails decisively, not marginally.** No
   configuration in the entire 24-cell grid beats v4 on inner-validation
   Sharpe. The single best cell (primary combo, persist_days=14 — nearly
   the loosest setting tested, and one where the vote fires infrequently
   enough that the strategy is drifting back toward v4's own behavior)
   still trails v4 by 0.52 Sharpe, more than double the ±0.2 noise floor.
3. **No plateau exists to report** — there is no cell near or above v4,
   so gate (2) is moot.
4. **Falsification test (b) (ETH) is not reached as a binding
   constraint** — gate (1) already fails outright — but was run in full
   anyway for completeness: no outright FAIL by the differential rule,
   for the same reason R-54 found (every config already underperforms
   v4's own BTC control by a wide margin, so there is no *additional*
   ETH-specific degradation to detect).
5. **Passes both integrity checks cleanly**: exposure-artifact
   R²=0.6103 (genuinely different exposure shape) and causality (0.0
   lookahead on price, the stablecoin pathway, and both combined, plus
   two exact identity recoveries of v4).
6. **Step-0 sanity check passes bit-for-bit**: `persist_days=0`
   reproduces R-54's original `_stable_vote` output exactly (max\|diff\|
   = 0.0 on all three thresh/gap combos tested), confirming this file's
   implementation is a faithful, minimal extension of R-54's mechanism
   and not a different architecture in disguise.

**Diagnosed mechanism, read directly off the data**: the descriptive
step's own onset-count table shows why the persistence filter does not
help at the parameter values that matter — at the primary threshold,
persist_days 0 through 3 produce the *identical* 12 onset events, meaning
the transient noise R-54 diagnosed does not reverse within 1–3 days; it
persists for roughly as long as some of the genuine episodes do. A
duration filter can only separate signal from noise if the two classes
occupy different duration regimes, and at this signal's native cadence
(daily on-chain data, 14-day growth window, 365-day z-score window) they
apparently do not, at least not within the 0–14 day range this round
tested (0–14 days is itself sub-one-cycle relative to the signal's own
14-day growth window, which may be the deeper reason: a persistence
requirement shorter than the feature's own smoothing window mostly
re-measures the same underlying noise the feature already contains).
Loosening `persist_days` far enough to actually reduce onset counts
(≥5–7 days) costs most of the lead-time advantage that was this
signal's one confirmed asset, so the round's central hope — that
precision and timing could be decoupled by adding a duration axis
orthogonal to the threshold axis — does not hold: at this signal's native
timescale, duration and timing draw from the same underlying noise
structure, and tightening one loosens the other, the same substantive
trade-off R-54 found on the threshold axis alone.

Per ROUTINE.md's own instruction not to spend the holdout on a candidate
that already failed pre-registered gates, **the 2023+ holdout was never
read.**

**One-line lesson:** a magnitude-and-duration filter is not automatically
the right way to convert a confirmed lead-time signal into a working
strategy — at this signal's native cadence (14-day growth window), the
duration axis and the lead-time axis draw on the same underlying noise
rather than being separable, so the same precision-vs-timing trade-off
R-54 found on the threshold axis alone reappears, worse, on the duration
axis; a duration requirement shorter than the feature's own smoothing
window mostly re-measures noise the feature already contains, rather
than adding new discriminating information.

## Holdout

**+0.** Never read. Grep proof, every date literal ≥2023 in this
branch's file:

```
$ grep -n "202[3-9]" experiments/kelly_regime_v16_stablecoin_persist.py
56:- Shu, Yu & Mulvey (2024), "Downside Risk Reduction Using Regime-
164:OOS_START = "2023-01-01"                 # never read in this file
693:    strictly pre-2023 bars. Structure duplicated from
697:    pre2023 = DF[DF.index < OOS_START]
698:    df = pre2023.iloc[-300_000:].copy()
```

Line 56 is the literature-citation year (2024) in prose. `OOS_START` is
used exclusively as an exclusive upper bound restricting the causality
probe (`pre2023 = DF[DF.index < OOS_START]`) and the `eth()` control
comparison (`DF[DF.index < OOS_START]`) to strictly pre-2023 bars — never
a data read past the boundary. Independently verifiable by re-running the
grep above. The pre-registered holdout decision rule (stated in full
above, before any result was read) required all of: beats v4 on
inner-validation Sharpe outside the noise floor, a plateau neighbourhood,
a preserved lead-time advantage, and an ETH-falsification pass. Gate (1)
failed decisively for all 24 configurations, so per that pre-registered
rule the holdout was correctly never touched.

## Test suite

`pytest` (from `.venv`): **457 passed** both before and after this
session's work — this branch added one new file
(`experiments/kelly_regime_v16_stablecoin_persist.py`) and edited nothing
else; no existing test was affected and no new tests were added (not
required for a NEGATIVE, unregistered experiment per ROUTINE.md).

## Next step

This closes the first of B-22's two named next steps (magnitude-and-
duration filter) as NEGATIVE. The second — feeding the signal into a
confirming, non-overriding combination rule closer to R-53's originally
pre-registered precision-weighted-average architecture — is this round's
disjoint parallel NOVEL branch's job
(`kelly_regime_v16_stablecoin_confirm.py`), not re-attempted here. If a
future session wants to revisit the duration-filter idea specifically,
the diagnosed mechanism above suggests the natural next move is not a
longer `persist_days` value (already tested to 14 days and shown to trade
away the lead-time advantage well before that point) but a duration
filter built on a *different, faster-native-cadence* re-derivation of the
stablecoin signal — e.g. a shorter growth window than the current 14-day
one, so that a multi-day persistence requirement is a genuinely
orthogonal confirmation rather than mostly re-measuring the same
smoothing the feature already performs. That is a signal-formula change,
not an architecture change, and was explicitly out of scope for this
round's minimal, one-new-parameter brief.
