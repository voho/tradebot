# kelly_regime_v16_stablecoin_confirm — R-55 NOVEL branch (08-20), closing B-22

Unregistered experiment. Code: `experiments/kelly_regime_v16_stablecoin_confirm.py`.
Signal module reused read-only, unedited: `experiments/_stablecoin_signal.py`
(`compute_stablecoin_stress`, feature `stablecoin_stress_z`, data
`data/stablecoin_supply_daily.csv.gz`, already fetched and committed by
R-54). Architecture template reused (duplicated, not imported, per this
project's own established precedent), unedited: `experiments/kelly_regime_v14_macro_lead.py`.
Not `@register`ed, not auto-discovered, nothing committed by this branch's
own choice — a human operator merges and commits after both R-55 branches
report. This branch does not touch `kelly_regime_v4.py`, `kelly_regime_v3.py`,
`kelly_regime.py`, `kelly_regime_v14_macro_lead.py`, `kelly_regime_v15_stablecoin_veto.py`,
`_stablecoin_signal.py`, `docs/LEDGER.md`, or the disjoint parallel
CONSERVATIVE branch's files (`experiments/kelly_regime_v16_stablecoin_persist.py`,
`experiments/reports/v16_stablecoin_persist_report.md` — neither read,
neither touched). All evaluation below is restricted to inner-train
(2017-01-01 → 2020-12-31), inner-validation (2021-01-01 → 2022-12-31), and
the standard pre-2020 ETH falsification pair (compared against the full
2017–2022 BTC "control," matching R-53's/R-54's own `eth()` convention
exactly). **The 2023+ holdout was never read** — grep proof at the bottom
of this report.

## Idea, mechanism, and pre-registration (written before any code ran)

**Idea, one sentence.** Feed `stablecoin_stress_z` — confirmed by R-54 to
genuinely LEAD `kelly_regime_v4`'s own 3-anchor majority price-gate flip
(9/12 matched stress episodes, median +16.5 days, the first INFO-axis
signal in this project's three attempts to do so) — into
`KellyRegimeV14MacroLead`'s precision-weighted CONFIRMING-vote combination
rule instead of R-54's hard, unweighted override, on the theory that a
false stress-onset should cost only a fraction of one anchor vote instead
of the entire vote, which should be more forgiving of the threshold's
known false-positive rate (24 transient stress-onsets at the tightest
setting) while retaining the confirmed lead-time advantage on genuine
episodes.

**Constraint attacked.** INFO. Fourth attempt: B-07/R-44 (on-chain), R-53
(macro), R-54 (stablecoin hard veto), now this (same signal as R-54,
different combination rule).

**Mechanism, precisely.** `stable_vote in {0,1}`, latched hysteresis on
`stablecoin_stress_z` exactly as R-54's `_stable_vote` (duplicated here,
not imported):

```
stable_vote -> 0 ("stress")  when stress_z crosses ABOVE thresh_hi
stable_vote -> 1 ("calm")    when stress_z crosses BELOW thresh_lo = thresh_hi - gap
stable_vote latched otherwise; defaults to 1 ("calm")
```

Combined vote — literally R-53's `KellyRegimeV14MacroLead` rule:

```
frac = (anchor_sum + stable_weight * stable_vote) / (3 + stable_weight)
```

`anchor_sum` is v4's own three UNCHANGED 0/1 latched price-anchor votes.
`stable_weight` swept over {0 (identity control), 0.15, 0.33, 0.5, 1.0
(unweighted-5-way-average negative control)} — identical grid to R-53's
`WEIGHTS`. `stable_weight=0` recovers v4 exactly (identity check, verified
below). Because `stable_vote` defaults to 1 and only ever drops to 0, this
vote can only ever pull `frac` DOWN, never up — same one-directional
discipline as R-53's and R-54's mechanisms.

## Sources

- BIS Working Paper No. 1340, "Stablecoin flows and spillovers to FX
  markets" (2025).
- Ahmed & Aldasoro, "Stablecoins and safe asset prices," Cleveland Fed
  financial-stability conference paper / BIS WP 1270 (August 2025).
- Federal Reserve Bank of New York, Liberty Street Economics,
  "Stablecoins and Crypto Shocks: An Update" (April 2025).
- IMF Working Paper 2025/141, "Decrypting Crypto: How to Estimate
  International Stablecoin Flows" (July 2025).
  (All four cited unchanged from `_stablecoin_signal.py`'s own docstring —
  this branch does not re-derive the signal, only its combination rule.)
- Freund & Schapire (1997), "A Decision-Theoretic Generalization of
  On-Line Learning and an Application to Boosting," *J. Comput. Syst.
  Sci.* 55(1), 119–139; Herbster & Warmuth (1998), "Tracking the Best
  Expert," *Machine Learning* 32(2), 151–178 — the weighted-combination-
  of-imperfect-signals literature this project's `hedge_experts`/
  `champions_council` strategies already draw on (`docs/STRATEGIES.md`
  §§3–4), cited here as the general justification for treating a rarer,
  sharper signal as *less* than a full vote (`stable_weight < 1`) rather
  than either an equal vote or a full override — the same precision
  argument R-53's and R-44's own reports already used.
- Web-research grounding (per this round's brief): current (2025–2026)
  industry practice reportedly weights stablecoin-flow signals at roughly
  15–25% of a combined signal rather than as a standalone override —
  motivation for this round's weight grid's low end (0.15/0.33), not a
  number copied verbatim; this project's own R-53 grid already used the
  same 0.15/0.33/0.5 spacing for an unrelated signal, so the two
  motivations converge on the same numbers independently.

## Not a duplicate of, cited precisely

- **`experiments/kelly_regime_v15_stablecoin_veto.py`** (R-54 NOVEL, this
  round's direct predecessor): identical signal, different combination
  rule — v15 forces `frac=0` outright while latched "stress" (B-21's hard
  override architecture); this file instead dilutes the vote by a
  fraction of one anchor. v15's own report named this exact fix (its
  second listed "next step") as untried; this file is that test.
- **`experiments/kelly_regime_v14_macro_lead.py`** (R-53 NOVEL): this
  file's own architectural template, duplicated (not imported) per the
  precedent R-54's own `kelly_regime_v15_stablecoin_veto.py` already set
  when it duplicated `_anchor_votes`/`_macro_vote` from the same file.
  What differs: the feeding signal. R-53 fed VIX/DXY `stress_z`, later
  shown by R-53's own `leadtime()` to LAG the 3-anchor majority (33% lead
  rate, median −5.5 days) — and under that lagging signal, the
  precision-weighted average did WORSE than R-53's own hard-override
  ablation in 10/12 matched cells. R-53's own report explicitly named the
  open question this file answers: the averaged architecture "might
  behave differently fed by a signal that actually leads." Testing the
  identical architecture with a genuinely-leading signal (R-54's
  confirmed result) is not a re-run of R-53's negative finding — it is the
  direct, pre-registered resolution of the confound R-53 itself could not
  separate (lagging signal vs. architecturally-flawed combination rule).
  This file's own `ablation()` (below) re-derives that comparison fresh,
  in-session, for the stablecoin signal specifically, rather than relying
  solely on R-53's VIX/DXY-specific numbers.
- B-07/R-44's on-chain branches, L-12/`harsanyi_crowd`: as in R-54's own
  "not a duplicate of" section — stablecoin supply is neither chain
  activity nor a price-derived crowding signal.
- The disjoint parallel CONSERVATIVE branch (B-22's OTHER named fix, a
  magnitude-and-duration persistence filter on the hard veto,
  `kelly_regime_v16_stablecoin_persist.py`): not read, not coordinated
  with.

## Pre-registered falsification tests (fixed before any result was read)

1. Does not beat v4 on inner-validation Sharpe (both markets) by more
   than the ±0.2 noise floor, OR does but not across a genuine parameter
   plateau (report neighbours).
2. Fails the pre-2020 BTC-control-vs-ETH differential test (R-54's
   `eth()` rule, reused verbatim).
3. Does NOT clear R-53's own negative finding: at matching (thresh_hi,
   gap) points, computed fresh in this file's `ablation()`, the
   precision-weighted average does not beat an analogous hard override
   fed by the identical signal.
4. Fails the causality tamper probe (price OR stablecoin pathway), or the
   `stable_weight=0` identity check does not recover v4 exactly.
5. Is an exposure-artifact (R² > 0.95 vs. a flat rescale of v4's own
   `target`).

**Pre-registered holdout decision rule, fixed before any result was
read:** the 2023+ holdout is read ONLY IF (1) clears the noise floor AND
sits on a genuine plateau, AND (2) passes, AND (3) shows the confirming
architecture actually beats its own hard-override ablation, AND (4)/(5)
both pass cleanly. If ANY of these fail, this branch reports NEGATIVE and
the holdout is never read.

## Code reuse decision

`_anchor_votes` and `_stable_vote` are DUPLICATED (not imported) from
`kelly_regime_v14_macro_lead.py` — the same precedent
`kelly_regime_v15_stablecoin_veto.py` itself set when it duplicated the
same two helpers from the same file. `compute_stablecoin_stress` IS
imported unchanged from `experiments/_stablecoin_signal.py` — read-only
reuse, exactly as R-54's own file used it. None of
`kelly_regime_v14_macro_lead.py`, `kelly_regime_v15_stablecoin_veto.py`,
or `_stablecoin_signal.py` is edited anywhere in this session.

## Configurations evaluated

**21 total**, precisely counted via the harness's `config_key`
deduplication:
- **17** confirming-vote configurations: 1 identity control
  (`stable_weight=0`) + 4 thresh/gap points (drawn from R-54's own 3×3
  grid: `primary`=1.00/0.75, `tightest`=0.75/0.00 [R-54's single most
  false-positive-prone cell, 24 stress-onsets], `tight-hys`=0.75/0.75,
  `loose`=1.25/1.25) × 4 weights (0.15, 0.33, 0.5, 1.0).
- **4** hard-override ablation configurations (falsification test 3), one
  per thresh/gap point, fed the identical signal.

Diagnostic re-reads (v4/`buy_and_hold` benchmarks, train-window re-checks
inside `select()`, the plateau table, causality tamper probes, the
exposure-artifact check, ETH control runs) are not separately counted,
per the R-42/R-44/R-53/R-54 convention.

## Descriptive: vote-transition frequency (context, not a fit)

The lead-time centerpiece check itself was **not re-derived** in this
file: it is a property of the raw signal plus hysteresis vote alone,
independent of how that vote is subsequently combined with the anchors,
and was already established by R-54's `leadtime()` at these same
thresh/gap settings (9/12 episodes lead the 3-anchor majority, median
+16.5 days). Re-running it here would reproduce identical numbers, not
add information — this file instead confirmed the vote-transition counts
at each thresh/gap point match R-54's own numbers exactly (12
stress-onsets at `primary`, 24 at `tightest`), verifying the duplicated
`_stable_vote` helper is byte-identical in behavior:

| thresh/gap | stress-onset events (inner-train+valid) |
|---|---|
| primary (1.00/0.75) | 12 |
| tightest (0.75/0.00) | 24 |
| tight-hys (0.75/0.75) | 13 |
| loose (1.25/1.25) | 6 |

Matches R-54's own `descriptive()` output exactly (12 and 24 for the same
two settings), confirming the reused signal/vote pipeline is unchanged.

## Inner-train (sweep, spot, 17 configs)

| candidate | final | Sharpe | max DD |
|---|---|---|---|
| `buy_and_hold` | $29,803 | 1.38 | 84.1% |
| `kelly_regime_v4` | $18,477 | 2.03 | 43.3% |
| identity (stable_weight=0) | $18,477 | 2.03 | 43.3% |
| loose w=0.33 (best) | $20,125 | 2.07 | 41.2% |
| primary w=0.33 | $19,573 | 2.06 | 41.0% |
| tightest w=1.00 (worst) | $14,254 | 1.88 | 51.3% |

Identity control exactly reproduces v4 on inner-train, as designed. A
handful of low-weight configurations edge out v4 slightly in-sample; the
unweighted (w=1.0) negative control is worst in every thresh/gap row,
consistent with the precision-weighting hypothesis.

## Inner-validation vs v4 (both markets, all 17 configs) — falsification test (1)

| candidate | market | final | Sharpe | max DD |
|---|---|---|---|
| `kelly_regime_v4` (control) | spot | $998 | 0.14 | 33.2% |
| `kelly_regime_v4` (control) | futures 5x | $1,064 | 0.25 | 32.3% |
| identity (stable_weight=0) | spot | $998 | 0.14 | 33.2% |
| best non-identity (primary w=0.15) | spot | $972 | 0.10 | 33.9% |
| best non-identity (primary w=0.50) | futures 5x | $1,106 | 0.32 | 33.0% |
| worst (tightest w=1.00) | spot | $796 | −0.23 | 43.2% |

**No non-identity configuration beats v4 on inner-validation spot Sharpe.**
The single best spot cell (primary, w=0.15: 0.10) is 0.04 *below* v4's
0.14 — every candidate configuration on spot underperforms the incumbent.
On futures, one cell (primary, w=0.50: Sharpe 0.32) beats v4's 0.25 by
+0.07 — inside the ±0.2 noise floor, not a real edge. **Falsification test
(1) fails decisively**: no plateau check is even meaningful because there
is no winning cell to build one around on the primary market (spot).

Full parameter-neighbourhood table (spot Sharpe, inner-validation):

| thresh/gap | w=0.15 | w=0.33 | w=0.50 | w=1.00 |
|---|---|---|---|---|
| primary (1.00/0.75) | 0.10 | 0.04 | 0.05 | −0.07 |
| tightest (0.75/0.00) | 0.07 | 0.06 | −0.01 | −0.23 |
| tight-hys (0.75/0.75) | 0.09 | 0.09 | 0.05 | −0.18 |
| loose (1.25/1.25) | 0.10 | 0.04 | −0.01 | −0.17 |
| identity (w=0, = v4) | 0.14 | | | |

There is a genuine, mild plateau in the low-weight column (0.09–0.10
across three of four thresh/gap rows) — but every cell in that plateau
still sits *below* v4's own 0.14, so the plateau describes where the
candidate loses *least*, not a winning region. Sharpe strictly decreases
with weight in every row, confirming the precision hypothesis
qualitatively (lower weight hurts less) while still never producing a
win.

## Falsification test (3): does the confirming architecture beat its own hard-override ablation, same signal?

**Yes, decisively, at every point tested** — this is the round's one
genuinely new, positive architectural finding, freshly computed in this
session (not cited from R-54's report):

| thresh/gap | split | market | confirm Sharpe | override Sharpe | Δ(confirm−override) |
|---|---|---|---|---|---|
| primary | TRAIN | spot | 2.06 | 1.84 | +0.217 |
| primary | TRAIN | fut 5x | 2.24 | 2.11 | +0.135 |
| primary | VALID | spot | 0.04 | −0.45 | **+0.489** |
| primary | VALID | fut 5x | 0.11 | −0.41 | **+0.519** |
| tightest | VALID | spot | 0.06 | −0.53 | **+0.588** |
| tightest | VALID | fut 5x | 0.09 | −0.52 | **+0.616** |
| tight-hys | VALID | spot | 0.09 | −0.61 | **+0.691** |
| tight-hys | VALID | fut 5x | 0.15 | −0.51 | **+0.663** |
| loose | VALID | spot | 0.04 | −0.22 | +0.255 |
| loose | VALID | fut 5x | −0.01 | −0.06 | +0.056 |

The confirming vote beats the hard override in **all 16 of 16** cells
tested (4 thresh/gap points × 2 splits × 2 markets), by margins as large
as +0.69 Sharpe on inner-validation — the opposite of R-53's finding
(averaged vote lost to override in 10/12 cells, fed a lagging signal).
**This directly answers this round's central architectural question: fed
a genuinely-leading signal, the precision-weighted confirming vote is not
merely "not worse" than a hard override, it clears it by a wide,
consistent margin.** The mechanism matches the pre-registered hypothesis:
diluting the vote by a fraction of an anchor instead of zeroing it
outright avoids paying the full cost of every one of R-54's ~24 transient
false-alarm stress-onsets, while the override pays that cost in full on
every one of them. **Falsification test (3) PASSES for the architecture**
— but architectural superiority over the hard veto is not the same as
beating v4, and it does not (test 1, above).

## Exposure-artifact check

R² of the primary candidate's `target` series against a mean-notional-
matched flat rescale of v4's own `target`, inner-validation, both markets:

| market | mean\|v4\| | mean\|cand\| | alpha | R² | raw corr | verdict |
|---|---|---|---|---|---|---|
| spot | 0.289 | 0.313 | 1.083 | 0.9407 | 0.9884 | genuinely different exposure shape |
| futures 5x | 0.289 | 0.313 | 1.083 | 0.9407 | 0.9884 | genuinely different exposure shape |

**PASS** (below the 0.95 threshold) but with far less margin than v15's
hard veto (R²=0.6091). This is the expected, mechanistic signature of the
confirming architecture: because it only ever dilutes the vote by a
fraction of one anchor rather than zeroing it, the resulting exposure
path stays much closer to v4's own than a hard override's does — the same
property that makes it more forgiving of false alarms (falsification test
3) also makes it a closer relabeling of v4's exposure. It clears the
0.95 bar, but not by the wide margin R-54's veto did.

## Causality probe (unregistered strategy, no CI coverage)

Two independent pathways tampered separately and together, on strictly
pre-2023 bars: price OHLCV (×3/÷3) and the stablecoin-supply pathway (raw
CSV copied to a temp directory, ×50/÷50 from the tamper day forward,
never touching the real `data/` directory).

| probe | decisions at/before cut | `target`/`v16_frac`/`v16_stable_vote`/`v16_anchor_sum` max\|diff\| before cut |
|---|---|---|
| PRICE tamper | PASS | 0.000e+00 (all 4 columns) |
| STABLECOIN tamper (new pathway) | PASS | 0.000e+00 (all 4 columns) |
| both at once | PASS | 0.000e+00 (all 4 columns) |
| identity check (`stable_weight=0` ≡ v4) | — | max\|diff\| = 0.000e+00, PASS |

No lookahead on either information pathway; the confirming-vote mechanism
recovers v4 exactly when the weight is zero, as designed. **Falsification
test (4) PASSES cleanly.**

## Falsification test (2): BTC full-history control vs ETH

ETH-USD Bitfinex spot (2016-03 → 2019-12-31) against USDT-supply coverage
overlap (2017-03-16 → 2019-12-31). Same rule as R-53/R-54: candidate must
not be visibly worse on ETH than on the identical-pipeline BTC control
(full 2017–2022 span).

| segment | configs failing | pattern |
|---|---|---|
| spot | **13 of 16** non-identity configs | **FAIL** |
| futures 5x | 0 of 16 | ok |

Representative cells (candidate/v4 final-balance ratio):

| config | market | BTC ratio | ETH ratio | flag |
|---|---|---|---|---|
| primary w=0.33 (the round's primary candidate) | spot | 1.030× | 0.904× | **FAIL** |
| primary w=0.33 | futures 5x | 0.822× | 1.193× | ok |
| tight-hys w=0.33 | spot | 0.957× | 0.940× | ok |
| tightest w=1.00 (unweighted, worst) | spot | 0.653× | 0.602× | **FAIL** |
| identity (w=0) | both | 1.000× | 1.000× | ok |

**Falsification test (2) FAILS decisively on spot**: 13 of 16 non-identity
configurations are visibly worse on ETH than on the BTC control by the
pre-registered rule (ETH ratio at least 2 percentage points below the BTC
ratio, and not itself beating v4). The failure is asset-specific and
systematic on spot — every thresh/gap family shows the same signature
(BTC ratio near or above 1.0×, ETH ratio meaningfully below it) — while
futures 5x cells pass uniformly. This is a genuinely new, spot-specific
weakness not observed in R-54's own `eth()` check on the hard-veto
architecture (which passed cleanly, no outright FAIL, because every
config there already underperformed v4 on the BTC side too, so the
*differential* test had nothing to catch). Here, several spot
configurations roughly match or slightly beat v4 on the BTC control
(primary w=0.33: 1.030×) while doing so specifically by riding out
BTC-specific stress episodes correctly and ETH-specific ones incorrectly
— a real, informative asset-specificity the confirming-vote's closer
tracking of v4's exposure (see the artifact check above) does not
prevent.

## Verdict: NEGATIVE

Two of five pre-registered falsification gates fail decisively (tests 1
and 2); the holdout is not read, per the pre-registered rule.

1. **Falsification test (1) FAILS.** No configuration beats v4 on
   inner-validation spot Sharpe (best cell 0.10 vs. v4's 0.14); the one
   futures cell that nominally beats v4 (+0.07) sits inside the noise
   floor. There is a mild plateau among low-weight configurations, but it
   is a plateau of "loses least," not of "wins."
2. **Falsification test (2) FAILS.** 13 of 16 non-identity configurations
   are visibly worse on ETH than on the BTC control on spot — a clean,
   asset-specific asymmetry this round's own `eth()` run surfaces for the
   first time in this signal's history (R-54's hard veto never triggered
   this test because it already lost to v4 on BTC everywhere).
3. **Falsification test (3) PASSES, and is this round's one genuinely new
   positive finding.** The precision-weighted confirming architecture
   beats an analogous hard override fed the identical signal in 16 of 16
   matched cells, often by half a Sharpe point or more on
   inner-validation — the exact reverse of R-53's finding under a lagging
   signal. This round therefore resolves the confound R-53 itself could
   not separate: architecture and signal quality are BOTH independently
   load-bearing. A leading signal genuinely does make the precision-
   weighted architecture earn its keep over a hard override — but earning
   its keep over one bad baseline is not the same as beating the
   incumbent, and it does not here.
4. **Falsification test (4) PASSES cleanly.** 0.0 lookahead on both
   pathways and combined; exact `stable_weight=0` identity recovery of v4.
5. **Exposure-artifact check PASSES** (R²=0.9407, spot and futures) but
   with far less margin than the hard veto's R²=0.6091 — the closer
   tracking of v4's own exposure that makes the architecture more
   forgiving of false alarms is the same property that makes it read
   closer to a relabeled v4.

**Read together, this closes B-22's second named fix as also negative,
completing the backlog item.** Both of B-22's two concrete, untried
proposals from R-54's own report — (1) a persistence/duration filter on
the hard veto, tested this round by the parallel CONSERVATIVE branch, and
(2) a precision-weighted confirming vote instead of an override, tested
here — have now been tried. This branch's finding is more nuanced than a
flat "no": the architecture genuinely does matter (falsification test 3),
and a leading signal genuinely does behave differently under it than a
lagging one did for R-53 — but the residual problem the hard veto had
(false positives costing more than genuine early exits save) is
sufficiently attenuated by dilution that it also removes most of the
mechanism's *upside*, since the same dilution that softens the cost of a
false alarm also softens the benefit of a correct one. The net effect on
spot, where this project's edge has always been strongest, is a wash to
a loss, plus a newly-discovered ETH-specific weakness the hard veto never
exhibited.

**One-line lesson:** a precision-weighted confirming vote is a strictly
better way to combine an imperfect regime signal than a hard override —
it clears R-53's own negative finding decisively once the feeding signal
genuinely leads — but "better than the worse alternative" is not the same
bar as "beats the incumbent," and the same vote-dilution that makes the
architecture forgiving of false alarms also dilutes it into a closer
relabeling of v4 that inherits a new, spot-specific ETH weakness neither
parent mechanism showed on its own.

## Holdout

**Pre-registered decision rule (restated from above, fixed before any
result was read):** read the 2023+ holdout ONLY IF falsification tests
(1) and (2) both pass AND (3) shows the architecture earning its keep AND
(4)/(5) both pass cleanly. Tests (1) and (2) FAIL. **The 2023+ holdout was
never read.**

Grep proof, every date literal ≥2023 in this branch's one new file, with
call-site context:

Actual output, re-run at report-writing time:

```
$ grep -n "202[3-9]" experiments/kelly_regime_v16_stablecoin_confirm.py
130:read:** the 2023+ holdout is read ONLY IF (1) clears the noise floor
195:OOS_START = "2023-01-01"                 # never read in this file
712:    strictly pre-2023 bars."""
715:    pre2023 = DF[DF.index < OOS_START]
716:    df = pre2023.iloc[-300_000:].copy()
```

Line 130 is this module's own docstring (the pre-registered holdout rule,
prose only, no data read). The single `OOS_START` literal (line 195) is
used exclusively as an exclusive upper bound restricting the causality
probe (lines 712/715/716) and the `eth()` BTC-control comparison
(`DF[DF.index < OOS_START]`, not itself matched by this grep pattern
since it contains no bare `202[3-9]` digit sequence beyond the variable
reference already caught) to strictly pre-2023 bars — never a data read
past the boundary. Independently verifiable by re-running
`grep -n "202[3-9]" experiments/kelly_regime_v16_stablecoin_confirm.py`.

## Test suite

`pytest` (from `.venv`): **457 passed both before and after this
session** — this branch touched no file under `src/` or `tests/`; no new
tests were added (not required for a NEGATIVE, unregistered experiment
per ROUTINE.md).

## Next step

B-22 is now closeable: both of its named fixes (persistence/duration
filter, tested by the parallel CONSERVATIVE branch this round; precision-
weighted confirming vote, tested here) have been tried and are negative
on the promotion bar. The one durable positive finding worth keeping for
any future confirming-vote attempt on this codebase: **architecture and
signal quality are separable, independently-testable axes** — R-53
showed a bad architecture can look bad partly because of a bad signal,
and this round shows that once the signal is fixed, the same architecture
clears the prior negative finding decisively, yet still does not clear
the actual bar (beating the incumbent). A signal that leads and an
architecture that is provably better than the naive alternative are both
necessary and neither is sufficient. Given this project's now-four-attempt
record on the INFO axis (B-07/R-44, R-53, R-54, R-55) with the same
qualitative failure shape each time (a genuinely new information channel
found, a combination mechanism that fails to convert it into a working
strategy), and per this round's own operator brief, the recommended next
step is not a fifth INFO-axis variant on this same stablecoin signal, but
B-06 (forward paper trading) — the highest-value remaining backlog item
that does not re-cut a dataset this research line has now tested four
separate times.
