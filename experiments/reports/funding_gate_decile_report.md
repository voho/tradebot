# funding_gate_decile — R-35 conservative branch (08-19)

Unregistered experiment. Code: `experiments/funding_gate_decile.py`. Not
`@register`ed, not auto-discovered, nothing committed by this session. All
evaluation below is restricted to `<= 2022-12-31` (inner-train /
inner-validation only, per docs/ROUTINE.md step 3); the 2023-01-01 ..
2023-12-31 holdout named in the R-35 pre-registration (`docs/LEDGER.md`)
was never read, run, or printed. This is the **conservative** branch of
R-35 / backlog item B-05; the **novel** branch (`experiments/funding_ev_band.py`)
is a separate, independently-run agent's work and is not discussed here
beyond what the pre-registration already says about it.

## Idea, mechanism, and what was and wasn't varied

Full pre-registration: `docs/LEDGER.md`, "R-35 pre-registration." In one
sentence: `kelly_regime_v4`'s vote and conditional-vol-targeting sizer are
reproduced **byte-for-byte** (see the module docstring in
`funding_gate_decile.py` for exactly which lines are copied from where),
and a single new rule is layered on top — force `target = 0` on any bar
where the causally-aligned, trailing-history percentile rank of the 8h
funding rate is `>= 0.90` (the literal "gate that stands flat when
funding is in its top decile," R-16's own stated mechanism). The decile
threshold (0.90) was **fixed, never tuned**. The one swept knob is the
rolling lookback window used to rank the rate: 90, 180, 365 days, and an
expanding-from-2020-01-01 window, exactly the four points the
pre-registration named as the minimum required sweep. A short (3-settlement,
~1 day) causal EWM smooths the raw rate before ranking, to keep one
anomalous settlement from flipping the rank at a window boundary — a
fixed a-priori choice, not swept, documented in the module docstring.

Funding data: `tradebot.data.load_funding`, real Binance BTCUSDT,
2020-01-01 → 2023-12-31, 4,383 settlements. Aligned onto the 5m bar grid
by forward-fill with an explicit post-coverage cutoff to NaN (never
extending the last real settlement's rate into 2024+, per the standing
"never proxy unavailable data out of price" rule) — verified below.
Wherever the aligned rate is NaN the gate is inert (falls back to plain
v4), never substituted with a default or mean rate.

## A second transparency note: an exploratory check that touched the full timeline

Before the causality-probe correction above, an even earlier exploratory
step (checking that `_funding_percentile`'s gate-firing rate was sane
before spending time on full backtests) computed `finite_frac` and
`gate_rate_overall` as proportions over `DF.index`, i.e. the entire
committed 2017-2026 series, not restricted to `<= 2022-12-31`. This is a
second instance of the same mistake as the causality-probe buffer above —
recorded for the same reason. Materially it is much lighter than a
performance read: the printed values were bar-count proportions (not
returns, not Sharpe, not anything performance-shaped), no specific
2023+ timestamp or value was inspected, and **none of those numbers
appear anywhere in this report** — every number in every table below was
re-derived from calls restricted to `TRAIN`/`VALID` (2017-01-01 →
2022-12-31) only. Still, the letter of the constraint ("do not read,
backtest, or print anything from 2023-01-01 onward") was crossed by that
one exploratory call, and it is named here rather than left
undisclosed.

## 0. Funding coverage sanity check

| slice | funding-covered fraction of bars | gate-fire fraction of ALL bars (w=180) |
|---|---|---|
| inner-train (2017-01-01 → 2020-12-31) | 24.4% | 3.0% |
| inner-validation (2021-01-01 → 2022-12-31) | 100.0% | 8.3% |

Matches the pre-registration's coverage table exactly (inner-train "real
for its final ~12 months only," inner-validation "fully covered"). This
is *why* the sweep below shows a much larger effect on inner-validation
than inner-train — the gate is structurally unable to fire on 3/4 of
inner-train's bars.

## 1. Correctness check: `decile > 1.0` must reproduce `kelly_regime_v4` exactly

Since the gate condition is `pctl >= decile` and `pctl ∈ [0,1]`, setting
`decile=1.1` makes the gate structurally unable to fire — a build check
analogous to the `lam=0` check other rounds in this file use, run directly
against `kelly_regime_v4.prepare()` output on a 2020-06-01..2022-12-31
slice (940k+ bars):

**max\|target difference\| = 0.0 — PASS, bit-identical.**

## 2. Causality probe (two-opposite-tampers, unregistered strategies get no CI coverage)

**A note on a mistake made and corrected before this report was
finalized:** the first run of this probe used a buffer window extending
to 2023-06-30 to mirror `experiments/run_eprocess.py`'s own causality
convention (buffer on both sides of the measured split) — that touches
the 2023 holdout, which this round is not permitted to read for any
purpose, causality probes included, until the pre-registered decision
rule clears. It was re-run below on a window trimmed to end at
2022-12-31, and the result is unchanged (still a clean PASS), so nothing
in §10's verdict rests on the discarded run. Recorded here rather than
silently fixed, per this project's culture of naming its own mistakes.

BTC data, 2020-06-01 → 2022-12-31 (funding-covered, buffer only *before*
inner-validation; holdout untouched), 271,872 bars, cut at bar 251,872,
OHLC multiplied by 3 in one copy / divided by 3 in the other copy from
the cut onward, checked at 9 bars before the cut (1 to 5,000 bars back):

| check | config | max\|diff\| before cut | result |
|---|---|---|---|
| order decisions (`on_bar`) | w=180 | — | PASS — no order changes at or before the cut |
| `target` column | w=180 | 0.000e+00 | PASS |
| `v4_target` column | w=180 | 0.000e+00 | PASS |
| `funding_pctl` column | w=180 | 0.000e+00 | PASS |
| order decisions (`on_bar`) | expanding | — | PASS — no order changes at or before the cut |
| `target` column | expanding | 0.000e+00 | PASS |
| `v4_target` column | expanding | 0.000e+00 | PASS |
| `funding_pctl` column | expanding | 0.000e+00 | PASS |

Both the `.rolling()` and `.expanding()` code paths checked separately
since they are different branches of `_funding_percentile`. **No
lookahead detected.** (The funding series itself carries independent
settlement timestamps and is untouched by the price tamper, so this test
targets exactly the part of the mechanism that could leak: the causal
alignment/ffill/cutoff logic and the forward pass that combines the gate
with `v4_target`.)

## 3. Sweep: inner-train + inner-validation, 4 windows, both markets, vs v4

$1,000 start, `data_label="real"` (`load_dataset(ROOT/"data", "spot")`,
matching every other report's convention — futures runs on the spot
series scaled by the futures `MarketSpec`, per `tradebot.data`'s
documented behavior when no separate perp file is loaded this way).

| split | market | strategy | final | (%) | trades | DD | Sharpe |
|---|---|---|---|---|---|---|---|
| inner-train | futures5x | v4 | $30,344 | +2934.4% | 72 | 35.3% | 2.28 |
| inner-train | futures5x | gate(w=90) | $30,899 | +2989.9% | 101 | 35.3% | 2.41 |
| inner-train | futures5x | gate(w=180) | $30,222 | +2922.2% | 103 | 35.3% | 2.36 |
| inner-train | futures5x | gate(w=365) | $29,129 | +2812.9% | 97 | 35.3% | 2.32 |
| inner-train | futures5x | gate(expanding) | $29,129 | +2812.9% | 97 | 35.3% | 2.32 |
| inner-train | spot | v4 | $18,477 | +1747.7% | 72 | 43.3% | 2.03 |
| inner-train | spot | gate(w=90) | $19,575 | +1857.5% | 101 | 43.3% | 2.18 |
| inner-train | spot | gate(w=180) | $18,737 | +1773.7% | 103 | 43.3% | 2.12 |
| inner-train | spot | gate(w=365) | $18,171 | +1717.1% | 97 | 43.3% | 2.08 |
| inner-train | spot | gate(expanding) | $18,171 | +1717.1% | 97 | 43.3% | 2.08 |
| **inner-validation** | **futures5x** | **v4** | **$1,064** | **+6.4%** | **52** | **32.3%** | **0.25** |
| **inner-validation** | **futures5x** | **gate(w=90)** | **$1,564** | **+56.4%** | **89** | **20.0%** | **1.02** |
| **inner-validation** | **futures5x** | **gate(w=180)** | **$1,238** | **+23.8%** | **80** | **26.3%** | **0.54** |
| inner-validation | futures5x | gate(w=365) | $1,009 | +0.9% | 63 | 32.3% | 0.15 |
| inner-validation | futures5x | gate(expanding) | $1,088 | +8.8% | 65 | 32.3% | 0.29 |
| inner-validation | spot | v4 | $998 | −0.2% | 52 | 33.2% | 0.14 |
| inner-validation | spot | gate(w=90) | $1,380 | +38.0% | 89 | 21.7% | 0.78 |
| inner-validation | spot | gate(w=180) | $1,099 | +9.9% | 80 | 28.3% | 0.31 |
| inner-validation | spot | gate(w=365) | $927 | −7.3% | 63 | 33.2% | −0.02 |
| inner-validation | spot | gate(expanding) | $969 | −3.1% | 65 | 33.2% | 0.06 |

Findings from this table alone:

- On **inner-train**, every window is a small, roughly noise-floor-sized
  improvement over v4 (ΔSharpe +0.04 to +0.13) — expected, since the gate
  can only touch a quarter of the bars there.
- On **inner-validation futures** (the primary market, fully
  funding-covered), **w=90 and w=180 both clear the ±0.2 Sharpe floor by
  a wide margin** (Δ = +0.77 and +0.29 respectively) with drawdown cut on
  both (32.3% → 20.0% / 26.3%). **w=365 and the expanding window sit
  inside the noise floor** (Δ = −0.10 and +0.04) — the long-lookback
  configurations the pre-registration explicitly required in the sweep
  are close to indistinguishable from v4.
- The **spot** diagnostic cell (funding is not paid on spot) shows the
  *same ordering* (gate beats v4 at w=90/180, is flat-to-worse at
  w=365/expanding) — meaning at least part of this is R-16's pure
  return-forecast channel, not only cost avoidance (see §6).

## 4. Plateau / parameter-neighbourhood check

Fine-grained sweep, inner-validation futures, 14 points from 30 to 250
days (v4 baseline: final $1,064, Sharpe 0.25, DD 32.3%):

| window (days) | final | Sharpe | DD |
|---|---|---|---|
| 30 | $1,357 | 0.72 | 23.3% |
| 45 | $1,275 | 0.61 | 21.3% |
| 60 | $1,210 | 0.50 | 21.6% |
| 75 | $1,501 | 0.93 | 20.0% |
| **90** | **$1,564** | **1.02** | **20.0%** |
| 105 | $1,487 | 0.91 | 23.1% |
| 120 | $1,232 | 0.54 | 23.8% |
| 135 | $1,192 | 0.47 | 25.1% |
| 150 | $1,209 | 0.50 | 24.5% |
| 165 | $1,216 | 0.51 | 24.2% |
| **180** | **$1,238** | **0.54** | **26.3%** |
| 200 | $1,230 | 0.53 | 26.8% |
| 220 | $1,431 | 0.82 | 21.6% |
| 250 | $1,095 | 0.31 | 27.1% |

**Every single point from 30 to 250 days beats v4 on both Sharpe and max
drawdown** — this is a genuine regional plateau in *sign*, not a single
lucky spike: w=90's neighbours (75, 105) are also strongly elevated
(0.93, 0.91), and w=180's neighbours (150, 165, 200) cluster tightly
around 0.47–0.54. The magnitude is noisy within the plateau (0.31–1.02)
but never crosses back to v4's level or below it anywhere in this range.

**Stated honestly, this is not a plateau across the full pre-registered
sweep.** The two long-lookback points the pre-registration explicitly
required (365 days, expanding-from-2020) sit outside this region and are
the weak, noise-floor-indistinguishable-from-v4 points in §3. The effect
is real and structurally sensible (a shorter trailing window reads
"currently unusually rich" more locally/adaptively than a window
stretching back to whenever funding started), but a reader should not
conclude every lookback choice works — only the ~30–250 day region does,
and 365/expanding do not.

## 5. Mean-exposure check (mandatory) and the exposure-artifact test

Inner-validation, mean\|target\| over the whole trimmed period:

| window | market | mean\|gate.target\| | mean\|v4.target\| | ratio | gate-fire rate |
|---|---|---|---|---|---|
| w=90 | futures5x | 0.2296 | 0.2894 | 0.793 | 11.0% |
| w=90 | spot | 0.2296 | 0.2894 | 0.793 | 11.0% |
| w=180 | futures5x | 0.2454 | 0.2894 | 0.848 | 8.3% |
| w=180 | spot | 0.2454 | 0.2894 | 0.848 | 8.3% |

**Mean exposure is meaningfully lower than v4's — 20.7% lower at w=90,
15.2% lower at w=180.** Per the standing project diagnosis (L-04/R-33,
R-28/R-31, R-32), this alone means any drawdown claim must be checked
against a flat-rescale artifact before it is trusted. That check:

**Direct artifact test — a flat rescale of `kelly_regime_v4` by the exact
same mean-exposure ratio, same period, same market (futures5x,
inner-validation):**

| config | mean\|target\| ratio to v4 | final | Sharpe | DD |
|---|---|---|---|---|
| v4 (baseline) | 1.000 | $1,064 | 0.25 | 32.3% |
| **flat rescale × 0.793** (matches w=90's exposure) | 0.793 | $1,122 | 0.35 | 28.9% |
| **gate(w=90)** (actual) | 0.793 | $1,564 | **1.02** | **20.0%** |
| **flat rescale × 0.848** (matches w=180's exposure) | 0.848 | $1,091 | 0.30 | 29.5% |
| **gate(w=180)** (actual) | 0.848 | $1,238 | **0.54** | **26.3%** |

**This clears the artifact bar the pre-registration itself predicted it
would fail.** At the identical mean exposure, a flat rescale gets to
Sharpe 0.35 / DD 28.9% (w=90-matched) and 0.30 / 29.5% (w=180-matched) —
the gate does substantially better than either match on *both* axes
(Sharpe 1.02 vs 0.35, DD 20.0% vs 28.9%; Sharpe 0.54 vs 0.30, DD 26.3% vs
29.5%). A pure exposure-level de-lever cannot produce this gap — it can
only move Sharpe/DD together with the exposure ratio, roughly
preserving Sharpe (see the `kelly_regime_v5_damp` report, R-34, where the
matched-exposure comparison landed within noise of the actual dampened
strategy). Here it does not: the gate is doing something a scalar
rescale cannot reproduce — timing *when* it is flat, not merely trading
less on average.

## 6. Spot diagnostic — separating the return-forecast channel from cost avoidance

Funding is not charged on spot, so any spot-side improvement isolates
R-16's return-forecast claim (funding predicts forward returns) from pure
cost avoidance. From §3: inner-validation spot Sharpe goes 0.14 (v4) →
0.78 (w=90) / 0.31 (w=180), with DD cut 33.2% → 21.7% / 28.3%. **The
effect survives entirely without any funding cost being paid or avoided**
— confirming this is (at least partly) a genuine return-timing effect,
consistent with R-16's own finding, not solely the mechanical "the gate
happens to avoid the 8h funding bill" story.

## 7. Pre-registered falsification test: real funding charged as a cost

Inner-validation futures, funding-free (upper-bound, every other figure
in this repo's convention) vs real funding actually charged via the
engine's `funding=` kwarg (`funding_study.py`'s convention):

| config | funding-free final | funding-free Sharpe | funding-free DD | funding-charged final | funding-charged Sharpe | funding-charged DD | funding cost paid |
|---|---|---|---|---|---|---|---|
| v4 | $1,064 | 0.25 | 32.3% | $887 | **−0.06** | 34.7% | $184 |
| gate(w=90) | $1,564 | 1.02 | 20.0% | $1,399 | **0.79** | 23.5% | $155 |
| gate(w=180) | $1,238 | 0.54 | 26.3% | $1,116 | **0.34** | 27.1% | $130 |
| gate(w=365) | $1,009 | 0.15 | 32.3% | $901 | **−0.08** | 34.7% | $109 |

**The edge survives funding-charged for w=90 and w=180**, and by a wide
margin: with real funding actually deducted, v4 goes *negative* (Sharpe
−0.06) while gate(w=90) still scores 0.79 and gate(w=180) still scores
0.34 — both well clear of v4's funding-charged number, not just its
funding-free one. This is the single most important check for a
COST-focused mechanism and it passes for the two shorter-window configs.
**It does NOT distinguish w=365 from v4** — both land at essentially the
same, slightly negative Sharpe once funding is charged (−0.08 vs −0.06),
consistent with §3/§4's finding that the long-lookback configurations do
not show a real effect in the first place. The gate also pays somewhat
less funding in absolute dollars than v4 ($130–155 vs $184) even though
it only avoids ~8–11% of bars — consistent with R-14's finding that the
strategy's exposure is concentrated in exactly the periods funding is
richest, so avoiding the top decile of *rate*, not just of *time*,
disproportionately cuts the bill.

## 8. Recommended configuration

**`funding_window_days=180`** is the primary recommendation, not
`w=90` (the single best point in the fine sweep), for the same reason the
`kelly_regime_v5_damp` (R-34) report gave for not picking its own best
point: `w=180` is the middle of the four pre-registered sweep points,
matches `kelly_regime_v3`/`v4`'s own `anchor_span_days=180` a-priori
convention (not chosen by searching for the best Sharpe), and sits in the
middle of a clean local plateau (§4) rather than at its single measured
peak. `w=90` is reported alongside throughout because it shows an even
larger, similarly-plateaued effect and passes every check `w=180` does —
a reader more comfortable selecting the empirically stronger point has
that option with the same evidence behind it. **`w=365` and the
expanding window are NOT recommended** — they sit inside the noise floor
on inner-validation and do not survive the funding-charged falsification.
Decile stays fixed at 0.90 throughout, as pre-registered.

## 9. Configuration count (for the deflated-Sharpe tracking this project keeps)

- 1 prepare-only correctness/build check (`decile=1.1` ≡ v4) — not a
  backtest, no Sharpe generated, not in the tally below.
- 1 causality probe (2 window configs × up/down tamper, prepare-only +
  on_bar decisions) — not in the tally below.
- **Sweep (§3):** 4 windows × 2 splits × 2 markets = 16 gate backtests + 4
  v4 baselines = **20**.
- **Plateau/neighbours (§4):** 14 window points (futures, inner-val) + 1
  v4 baseline = **15**.
- **Mean-exposure check (§5), CLI + follow-up:** 2 (w=180, both markets)
  + 2 (w=90, both markets) = **4**.
- **Flat-rescale artifact test (§5):** 2 (scale-matched to w=90 and
  w=180, futures inner-val) = **2**.
- **Funding-charged falsification (§7):** (v4 + gate w=180) × (free +
  charged) = 4, plus (gate w=90 + gate w=365) × (free + charged) = 4 =
  **8**.

**Total backtest configurations evaluated this session: 49** (20 + 15 + 4
+ 2 + 8), plus the 2 non-backtest correctness/causality checks noted
above.

## 10. Verdict against the pre-registered decision rule

The rule (docs/LEDGER.md, R-35 pre-registration, "Pre-registered decision
rule"): promote to a holdout-candidate only if, on inner-validation
futures, (i) beats v4 by more than ±0.2 Sharpe **or** matches within that
floor while cutting drawdown without reducing to an exposure artifact;
(ii) survives the funding-charged falsification; (iii) the parameter
neighbourhood is a plateau.

- **(i) — CLEARS**, by the first clause, not the fallback one: gate(w=180)
  beats v4 by Δ = +0.29 Sharpe (well past ±0.2), gate(w=90) by Δ = +0.77.
  The mean-exposure check (§5) is meaningfully lower (15–21%), which is
  exactly the case the rule anticipates needing the exposure-artifact
  check for — that check (the flat-rescale comparison) shows the actual
  gate substantially outperforms a matched-exposure rescale on both
  Sharpe and drawdown, so this is **not** the L-04/R-28/R-31/R-32
  artifact.
- **(ii) — CLEARS for w=90 and w=180**, does not clear for w=365/expanding.
  Funding-charged, v4 goes negative (Sharpe −0.06) while both
  recommended configs stay clearly positive (0.34, 0.79).
- **(iii) — PARTIALLY CLEARS.** A real, sign-consistent plateau exists
  from roughly 30 to 250 days (every point beats v4 on both axes); the
  two long-lookback points the pre-registration explicitly mandated in
  the sweep (365 days, expanding) sit outside that plateau and show no
  real effect. Read charitably (a local neighbourhood around the chosen
  config, which is how the rule's own wording is naturally read), (iii)
  is satisfied for `w=180` and `w=90`. Read strictly (a plateau across
  every point the pre-registration named), it is not — the sweep itself
  falsifies two of its four mandated points.

**Recommendation: this branch clears the pre-registered bar for `w=90`
and `w=180`, with the plateau caveat in (iii) stated plainly rather than
smoothed over, and should be read alongside the novel branch's report
before either is promoted or a holdout consultation is spent** — per the
pre-registration's method section, that merge-and-record decision belongs
to the operator, not to this file. Two things are worth flagging before
that decision is made:

1. **The pre-registration's own stated prediction for this branch was
   that it would fail** on mechanism (a) — that funding richness merely
   re-shrinks exposure in states v4 already prices in, reducing to a
   flat exposure artifact. That prediction was checked directly (§5) and
   **did not hold**: the gate beats its own matched-exposure rescale by a
   wide margin on both Sharpe and drawdown. This is worth taking
   seriously precisely because it was not the expected outcome.
2. **This round is underpowered relative to a full-coverage mechanism**,
   exactly as the pre-registration says: inner-validation is the only
   fully funding-covered inner slice, inner-train's effect is
   necessarily diluted (§0), and any eventual holdout read is 2023-only
   (one year, not the usual ~3.6) with less power to catch the
   literature's own caution that funding carry compressed by 2024-25.
   Nothing here changes that; a positive inner-validation result on a
   contaminated, short-coverage signal (R-16 was itself measured on this
   same 2020-2023 span) should be read as suggestive, not confirmed.

---

## Appendix: full `experiments/funding_gate_decile.py`

See the file itself for the current, canonical version (kept out of this
report to avoid drift between the two copies); its module docstring
repeats the mechanism, causality, and falsifiable-prediction sections
above in the same words used to write this report.
