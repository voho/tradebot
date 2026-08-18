# Bayes-Stein shrinkage Kelly sizing — matched-risk report

**Session date:** 2026-08-18. **Branch:** parallel research round, ERR-constraint
lane. Not registered — this document plus `experiments/bayes_stein_kelly.py`
and `experiments/run_bayes_stein_kelly.py` are the record, per ROUTINE.md
step 5 ("NEGATIVE → ledger row plus code under `experiments/`").

This file is written as a `docs/LEDGER.md`-style row (R-28/R-31/R-32's
format) so the operator can fold it in directly. **Everything under
"Pre-registration" below was written and committed to this file before the
2023+ holdout was read** — the holdout commands (`holdout`, `interval`,
`costs`, `windows`) were run only after the pre-registration section
existed in this form. The "Results" sections that follow report what
happened, including where it disagreed with the stated predictions.

---

## Idea, in one sentence

Estimate the recent drift (mean log-return) with a causal EWM estimator,
shrink it toward a **zero-drift prior** by the classical single-parameter
empirical-Bayes / James-Stein weight (the fraction of the estimate *not*
explained by its own estimation noise), and use the surviving fraction
directly as a continuous, confidence-weighted gate multiplying the
incumbent's inverse-volatility sizer — no latch, no sequential
wealth-accumulation state.

**Citation.** Jorion, P. (1986), "Bayes-Stein Estimation for Portfolio
Analysis," *Journal of Financial and Quantitative Analysis* 21(3),
279–292 — the sample mean is an inadmissible estimator of expected return
under squared loss (Stein 1956; James & Stein 1961) and shrinking it
toward a common target reduces realized estimation error out of sample.
Jorion's estimator shrinks a *cross-section* of asset means toward the
grand mean of the minimum-variance portfolio; there is one asset and one
time series here, so this row shrinks a single time-varying scalar (this
bar's drift estimate) toward the more common single-parameter
empirical-Bayes choice for a "no edge" prior — zero — using the
Efron–Morris / Morris (1983, *JASA* 78(381)) shrink-to-zero form
`shrinkage_weight = se² / (se² + μ̂²)`. A web search for 2023–2026
extensions of shrinkage estimators to crypto or intraday sizing
specifically returned nothing published in that window; this row applies
the 1975–1986 estimator directly rather than citing a newer variant that
does not appear to exist yet in the literature this project can find.

**Exact formula implemented** (`experiments/bayes_stein_kelly.py`):

```
mu_hat_t   = EWM_mean(r, span=S).shift(1)              # r = log-return
sigma_bar_t= EWM_std(r,  span=S).shift(1)               # same span
n_eff      = S                                          # EXACT for pandas EWM:
                                                          # 1/sum(w_k^2) = (2-a)/a = S
se_t       = sigma_bar_t / sqrt(n_eff)
z_t        = clip(mu_hat_t / se_t, -z_clip, z_clip)
shrink_wt_t= 1 / (1 + z_t^2)                             # "B" in Morris's notation
conf_t     = (1 - shrink_wt_t)  if mu_hat_t > 0 else 0   # = z_t^2/(1+z_t^2), in [0,1]
target_t   = conf_t * scale_t                            # scale = incumbent's
                                                          # min(k*target_vol/vol, k*max_leverage)
```

The `n_eff = S` identity was verified numerically before this file was
written (`sum(w_k^2)` for `w_k = alpha(1-alpha)^k`, `alpha = 2/(S+1)`,
gives `1/sum(w_k^2) = 20.000000000000004` for `S = 20` — exact, not a rule
of thumb). `conf_t` is bounded in `[0, 1]` by construction, so it drops
into `matched_risk.GatedKelly`'s `conf * scale` architecture with no extra
normalisation constant, and the `vol` used by `scale` is the **unchanged**
`vol_span = 8 day` incumbent estimator — only the drift side is new.

## Constraint attacked

**ERR** — explicit treatment of parameter (drift) estimation uncertainty
in the signal path — via a mechanism not yet in the ledger: empirical-Bayes
shrinkage of a point estimate, as distinct from anytime-valid sequential
hypothesis testing (R-28/R-31/R-32) or latched voting (L-01–L-04).

## Not a duplicate of R-28 / R-31 / R-32 — or of R-01/R-02/R-03/R-08/R-09

Both this gate and R-28's e-process produce a confidence value in `[0,1]`
multiplying an inverse-volatility sizer, and both attack ERR. That is
where the resemblance ends, and the difference is mechanical, not
cosmetic:

- **The e-process is a sequential hypothesis test with a stopping-time
  guarantee and a state variable.** Its `conf` is `wealth_t / log(1/alpha)`,
  where `wealth_t = min(cap, max(0, wealth_{t-1} + log1p(lam_t*z_t)))` — an
  accumulator that only grows on favourable evidence and decays only
  through an explicit cap (fixed at 1.0 by this project's convention).
  R-28 measured its half-life-20d incarnation needing **3.8 years** of
  the measured drift/noise ratio to cross the α=0.05 threshold, and its
  mean gate over a decade was **0.145** — evidence built in the 2017 bull
  persisted for years.
- **Bayes-Stein shrinkage has no accumulator and no state.** `conf_t`
  is a pure function of `mu_hat_t` and `se_t`, both re-estimated from
  scratch every bar over a window of `S` days (10–60 in this row, not
  years). Nothing plays the role of "wealth"; nothing carries forward
  from bar to bar except what is already inside the rolling window. It
  buys a different property (lower mean-squared error of the point
  estimate under squared loss, bar-by-bar) rather than the e-process's
  anytime-valid Type-I control — this project has never measured that
  trade explicitly before.
- **This implies a specific, checkable behavioral difference**, not just
  an algebraic one: the Bayes-Stein gate should re-open and re-close on
  the timescale of one `drift_span` window around each local drift
  reversal and should *not* show the e-process's multi-year lag. This is
  measured directly below (`memory`), not assumed from the formula — see
  Falsification/failure-mode (d).
- **Not R-01 (HMM) / R-02 (jump models) / R-03 (BOCPD).** Those detect a
  discrete regime *state*. Bayes-Stein shrinkage asserts no such state —
  it is a continuous correction to one point estimate with no notion of
  "which regime."
- **Not R-08/R-09 (volatility forecasting, which hurt).** The volatility
  input to the sizer (`vol_span = 8d`) is byte-for-byte the incumbent's;
  only the drift-side estimator is new. R-08's finding (a genuinely
  better vol forecast de-levers into BTC's high-vol/high-forward-Sharpe
  states and loses) does not apply here because nothing about the vol
  estimator changed.
- **Not L-01–L-04** (latched multi-anchor vote — binary/tri-state, and
  carries no notion of *how much* estimation error there is in the signal,
  only whether price is above or below an anchor).

## Simulable here?

Yes. One price series, causal EWM statistics only, no new data, no fetch.

---

## Pre-registration — written before the 2023+ holdout was read

### Pre-registered failure modes

**(a)** Shrinkage collapses to a smoothed momentum indicator in disguise —
`z²/(1+z²)` applied to an EWM mean is mechanically close to a
soft-thresholded momentum signal, and if the resulting confidence
correlates near 1.0 with the incumbent vote's binary state, the
"uncertainty treatment" framing is decorative rather than substantive.

**(b)** It only "wins" by holding less exposure — must be checked at
**matched realized risk** against `vote` (and `evidence`), never at
natural exposure, per R-31/R-32's finding that this is the dominant
artifact in this entire line of work.

**(c)** No plateau in the shrinkage window (`drift_span_days`) or
`z_clip` — if the frontier is a fitted spike rather than a region, the
selected configuration is a peak, not a principled choice.

**(d)** *(specific to this mechanism, not copied from R-28)* The
shrinkage intensity has no decay/memory term, so — unlike the e-process —
it should re-open and re-close within roughly one `drift_span` window
around every local drift reversal. If it turns out to move as slowly as
the e-process in practice (multi-year persistence, low turnover), the
"no accumulation" distinction claimed above is cosmetic rather than
behavioral, and that must be measured directly (gate autocorrelation,
run length, turnover/fill count), not asserted from the algebra.

### Falsification test (chosen now, before any holdout data was read)

Same convention as R-17/R-28/D3(R-31)/P3(R-32): Bitfinex BTC (control)
and ETH (test), same window, exposures **re-matched on each asset's own
realized volatility** (the matching is a property of the risk axis, not a
parameter fitted on BTC). The **ordering** between `bayes_stein` and
`vote`, and between `bayes_stein` and `evidence`, at matched risk on the
BTC holdout window must replicate on ETH. If it flips, any BTC matched-risk
claim from this row is dead, exactly as R-31 ruled for R-28's P3.

### Decision rules, fixed in advance

- **Q1 (primary).** Paired Δ log growth (`bayes_stein − reference`) on the
  2023+ holdout, 95% stationary block bootstrap (30-day mean block, 2,000
  resamples, identical indices for both arms — the R-29/R-30/R-31 method,
  reused via `tradebot.inference` directly). Established only if the
  interval **excludes zero, in every VALID cell**, for both reference
  gates (`vote` primary, `evidence` secondary/"ideally", per the brief).
- **Q2.** Same test on Δ max drawdown.
- **V — validity gate**, applied before any decision rule is read, R-31's
  rule verbatim: a cell is scored only if the two arms' realized
  volatilities on the holdout are within 20% of each other **and** the
  notional-cap clamp fraction is below 1% for both arms. Otherwise VOID.
- **P1.** Holdout spot final balance beats `buy_and_hold`.
- **P2.** > +0.2 Sharpe (R-20 noise floor) **or** ≥ 10pp drawdown
  improvement vs `buy_and_hold`.
- **P3 (falsification).** The `bayes_stein`-vs-`vote` and
  `bayes_stein`-vs-`evidence` orderings replicate on ETH.
- **P4.** Plateau in `drift_span_days` and `z_clip` on inner-validation.
- **Promotion bar (default reject).** Only a candidate if Q1 (or Q2 on
  drawdown) is established in every valid cell **and** P1–P4 all pass.

### Stated prediction before looking

Given R-31/R-32's finding that gate *mechanism* is largely irrelevant once
risk is matched ("the gate is worth more than the choice of gate"), the
prediction is that **Q1 will be NOT ESTABLISHED** (every interval contains
zero) in every valid cell, mirroring R-31/R-32's D1/Q1 results, and that
several holdout cells will be **VOID** under the R-31 validity gate (the
spot notional cap), also mirroring R-31/R-32. The one place this
mechanism is expected to actually *differ* from the e-process is failure
mode (d): `bayes_stein`'s gate should show measurably higher turnover and
shorter persistence than `evidence`'s, even if the holdout return/drawdown
numbers end up statistically indistinguishable from both references — an
honest positive finding buried inside an expected-negative result, the
same shape as R-31/R-32's own headline.

---

## Step 3 — implementation and inner-split results

**Frozen a priori before the sweep ran:** `drift_span_days = 20`, chosen
for the same reason R-28 chose a 20-day evidence half-life — it coincides
with R-07's independently-found robust 18–28 day anchor region and shares
a timescale with both `vote`'s fastest anchor and `evidence`'s default
half-life, so a frontier difference cannot be blamed on "different
lookback." `z_clip = 10` (a priori default, loose enough that the
mainline sweep essentially never clips — confirmed in the plateau check
below). Sizer fixed to `plain` throughout (the `conditional` sizer is
excluded from this round for the same reason R-31 excluded it from its
own holdout — at ~124+ prior consultations the cheapest thing this
project can do with the holdout is ask it fewer questions).

**Step-3 sweep (`frontier`):** 3 candidate spans (10d / 20d / 60d) × 9
exposure multipliers (`k` ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6} — the
identical grid R-31/R-32 traced) on inner-train and inner-validation, both
markets. **27 distinct configurations**, counted once each (the R-28/R-31
convention: scoring a configuration on a second market or split is another
backtest, not another trial).

Inner-validation Sharpe by span (spot + futures pooled, `k` swept):

| span | mean Sharpe | min | max |
|---|---|---|---|
| 10d | **+0.39** | +0.04 | +0.62 |
| 20d | **−0.14** | −0.43 | +0.28 |
| 60d | **+0.10** | −0.32 | +0.30 |

**Honesty check on the pre-registered choice.** The sweep's own numbers do
*not* favour the a-priori pick — 10d nominally beats both 20d and 60d on
inner-validation Sharpe. `drift_span_days = 20` is kept frozen anyway,
because it was committed to for a stated, non-performance reason (matching
R-07's robust window and the other two gates' shared timescale) before the
sweep ran; switching to whichever span the sweep happened to favour would
just be a second, hidden selection step on inner-validation, the exact
thing pre-registration exists to prevent. The 10d-vs-20d gap is recorded
here as a legitimate open alternative for a future session, not smoothed
over.

**Configurations evaluated in step 3: 27** (frontier) **+ 18** (P4
neighbourhood, below, counted separately per R-28's own convention) **=
45 total this session.**

---

## Step 3 — P4 plateau check

One-knob-at-a-time neighbourhood around the frozen point (span=20d,
`k = 2.373`, the spot vs-vote match-up exposure — see Step 4), spot,
inner-validation:

| span (d) | z_clip | vol | final | DD | Sharpe |
|---|---|---|---|---|---|
| 15 | 5/10/20 | 0.327 | $861 | 42.7% | −0.07 |
| 18 | 5/10/20 | 0.315 | $928 | 38.1% | +0.04 |
| 20 | 5/10/20 | 0.320 | $833 | 43.0% | −0.13 |
| 25 | 5/10/20 | 0.330 | $1,005 | 38.8% | +0.17 |
| 30 | 5/10/20 | 0.326 | $935 | 43.1% | +0.06 |
| 60 | 5/10/20 | 0.363 | $1,043 | 30.6% | +0.24 |

**z_clip is inert across the entire neighbourhood** — identical output at
5, 10 and 20 for every span tested, meaning `z_t` essentially never
approaches even the tightest clip in practice; this parameter turned out
not to bind, which is itself worth recording rather than treating as an
unexamined knob. Across span 15–60d the matched volatility stays in a
narrow 0.315–0.363 band and Sharpe scatters in [−0.13, +0.24] with no
sharp peak at the frozen point — **P4 (plateau, not a spike): PASS.**

**Configurations evaluated in the plateau check: 18** (6 spans × 3
z_clip values).

---

## Step 4 — frozen configuration

Exposures solved on **inner-validation only** (`match`), matching
`bayes_stein` (span=20d) to each reference gate's realized volatility in
both directions, then frozen before the holdout was read:

| market | direction | matched to | vote k | evidence k | bayes_stein k |
|---|---|---|---|---|---|
| spot | vs vote, up | vote's vol 0.325 | 1.000 | — | **2.373** |
| spot | vs vote, down | bayes_stein's vol 0.156 | **0.467** | — | 1.000 |
| spot | vs evidence, up | evidence's vol 0.087 | — | 1.000 | **0.572** |
| spot | vs evidence, down | bayes_stein's vol 0.156 | — | **1.784** | 1.000 |
| futures | vs vote, up | vote's vol 0.322 | 1.000 | — | **1.820** |
| futures | vs vote, down | bayes_stein's vol 0.177 | **0.502** | — | 1.000 |
| futures | vs evidence, up | evidence's vol 0.136 | — | 1.000 | **0.887** |
| futures | vs evidence, down | bayes_stein's vol 0.177 | — | **1.126** | 1.000 |

**Reproduction check.** The `vote`/`evidence` reference vols at k=1
(spot: vote 0.325, evidence 0.087; futures: vote 0.322, evidence 0.136)
match R-31's published `frozen.json` figures exactly (R-31: spot
match-down vote-target 0.087, futures match-down vote-target 0.136), an
independent confirmation that this row's copy of `GatedKelly`'s
`vote`/`evidence` gates reproduces R-31/R-32's numbers before any
`bayes_stein`-specific result is trusted.

At k=1 on inner-validation the un-matched exposure ranking is
`vote (0.325) > bayes_stein (0.156) > evidence (0.087)` on spot and
`vote (0.322) > bayes_stein (0.177) > evidence (0.136)` on futures — the
shrinkage gate's natural exposure sits *between* the latched vote and the
e-process, consistent with having neither a hard binary latch (which
holds more) nor a multi-year evidence threshold (which holds less).

---

## Step 4 — causality probe

By-hand two-opposite-tampers procedure (`causality`), identical method to
R-28/R-31: bars after a cut (bar 195,000 of 200,000) multiplied by 3 in
one copy and divided by 3 in the other; every order at or before the cut
must be identical, and `target`/`conf`/`scale` must match exactly before
the cut.

```
span=   10d  orders match   max |column difference| before the cut = 0.000e+00   PASS
span=   20d  orders match   max |column difference| before the cut = 0.000e+00   PASS
span=   60d  orders match   max |column difference| before the cut = 0.000e+00   PASS

tampered from bar 195,000 of 200,000; PASS - no decision at or before the cut moves
```

**`pytest`: 436 passed**, run from repo root before and after this row's
work — no existing file was touched, and the suite is unaffected (as
expected, since this session created only new files under `experiments/`).

---

## Step 4 — holdout results (2023-01-01 →), validity gate applied to every cell

**Reference strategies:**

| | spot final | spot DD | spot Sharpe | fut 5x final | fut DD | fut Sharpe |
|---|---|---|---|---|---|---|
| `buy_and_hold` | $3,839 | 54.0% | 1.03 | $15,176 | 60.3% | 1.44 |
| `kelly_regime_v4` | $3,373 | 27.8% | 1.22 | $4,901 | 33.0% | 1.36 |

**Matched-risk cells** (V = R-31's validity gate: vol gap < 20% AND both
arms' clamp fraction < 1%):

| market | pair | reference (k) | final | DD | bayes_stein (k) | final | DD | V |
|---|---|---|---|---|---|---|---|---|
| spot | vs vote, up | vote (1.000) | $3,277 | 26.7% | 2.373 | $2,086 | 29.6% | **VOID** (clamp 41.0%/29.1%) |
| spot | vs vote, down | vote (0.467) | $2,165 | 16.6% | 1.000 | $1,702 | 28.0% | **VOID** (clamp 0.0%/4.1% — bs clamp ≥1%) |
| spot | vs evidence, up | evidence (1.000) | $1,607 | 11.6% | 0.572 | $1,441 | 17.7% | **VOID** (clamp 1.3%/0.0%) |
| spot | vs evidence, down | evidence (1.784) | $2,045 | 17.0% | 1.000 | $1,702 | 28.0% | **VOID** (clamp 6.9%/4.1%) |
| futures | vs vote, up | vote (1.000) | $4,980 | 32.5% | 1.820 | $4,660 | 38.5% | **VALID** |
| futures | vs vote, down | vote (0.502) | $2,400 | 17.9% | 1.000 | $2,174 | 26.0% | **VALID** |
| futures | vs evidence, up | evidence (1.000) | $1,776 | 14.3% | 0.887 | $2,255 | 22.0% | **VALID** |
| futures | vs evidence, down | evidence (1.126) | $2,025 | 16.0% | 1.000 | $2,174 | 26.0% | **VALID** |

**All four spot cells are VOID** — every one trips the 1% notional-clamp
threshold on at least one arm, the identical failure mode R-31/R-32 both
hit on spot ("the 1.0-notional cap sets the position on a third of bars,
not the gate"). **All four futures cells are VALID.**

**Point estimates on the four valid (futures) cells: `bayes_stein` is
worse than both references on drawdown in all four, and worse on return
in three of four** (it edges out `evidence` on return in the
match-up/match-down cells while still drawing down more — $2,255 vs
$1,776 and $2,174 vs $2,025, both at 6–10pp deeper DD).

**Turnover, an unplanned but striking difference.** `bayes_stein`'s fill
counts dwarf both references at every matched exposure — spot vs-vote-up:
1,761 fills vs vote's 349; spot vs-evidence-down: 3,270 vs evidence's 249.
This is the mechanical fingerprint of having no latch and no accumulator:
without hysteresis or a threshold to cross, a continuously-varying
confidence signal re-crosses the deadband far more often than a state
machine with 2–3 discrete levels. It also explains why spot trips the
clamp differently on each arm — churn interacts with the deadband
differently than a level-hold does — and it costs real money: fees paid
are 3–14x the reference gate's at the same matched exposure (e.g. spot
vs-vote-up: $943 vs $295; spot vs-evidence-up: $335 vs $50).

## Step 4 — Q1/Q2, paired bootstrap on the holdout

95% stationary block bootstrap, 30-day mean block, 2,000 resamples,
identical indices for both arms, `tradebot.inference.paired_bootstrap`
directly on daily returns (`bayes_stein − reference`), 1,319 daily
observations:

| market | pair | Δ log growth | 95% CI | Δ max DD (pp) | 95% CI |
|---|---|---|---|---|---|
| spot *(void)* | vs vote, up | −0.452 | [−1.020, +0.101] | +4.09 | [−13.2, +16.7] |
| spot *(void)* | vs vote, down | −0.240 | [−0.629, +0.163] | +11.4 | [−5.2, +21.5] |
| spot *(void)* | vs evidence, up | −0.109 | [−0.468, +0.205] | +5.22 | [−5.8, +12.9] |
| spot *(void)* | vs evidence, down | −0.184 | [−0.586, +0.236] | +11.3 | [−6.2, +22.4] |
| **futures (valid)** | vs vote, up | −0.066 | [−0.795, +0.693] | +4.03 | [−19.7, +13.9] |
| **futures (valid)** | vs vote, down | −0.099 | [−0.479, +0.267] | +6.78 | [−7.4, +16.0] |
| **futures (valid)** | vs evidence, up | +0.239 | [−0.288, +0.781] | +7.61 | [−4.3, +19.2] |
| **futures (valid)** | vs evidence, down | +0.071 | [−0.447, +0.625] | +8.83 | [−3.4, +22.8] |

**Every interval contains zero, in every cell, valid or void, on both
axes.** The sign is not even stable (Δ log growth is negative in 6 of 8
cells and positive in the two valid `vs evidence` cells; Δ max drawdown is
positive — `bayes_stein` deeper — in all 8 of 8 point estimates, but no
interval clears its own error bar).

**Q1: NOT ESTABLISHED**, in every valid cell (both futures pairs, both
references). **Q2: NOT ESTABLISHED** in every cell as well, despite the
drawdown point estimate leaning the same direction (deeper) in all eight
cells — the same shape R-31 found for the e-process's drawdown "edge":
consistent in sign, inside the noise floor.

**P1: FAIL.** Holdout spot final balance: `bayes_stein` loses to
`buy_and_hold`'s $3,839 in every matched cell (best: $2,086). Note P1 is
evaluated on the *spot* holdout per the pre-registered rule, and every
spot cell is additionally void — so P1 fails on grounds that do not even
need the validity question resolved.

**P2:** moot — P1 already fails and gates it.

---

## Step 4 — falsification test (ETH vs BTC control)

Bitfinex BTC (control, 396,449 bars, 2016-01-01→2019-12-31) and ETH (test,
342,929 bars, 2016-03-09→2019-12-31), the identical R-17/R-28/R-31 window,
exposures **re-matched on each asset's own volatility** (`eth`):

| market | pair | BTC winner (return) | BTC winner (DD) | ETH winner (return) | ETH winner (DD) | replicates? |
|---|---|---|---|---|---|---|
| spot | vs vote, up | **bs** ($11,281 vs $8,778) | **bs** (23.1 vs 37.9%) | **bs** ($5,463 vs $5,186) | **bs** (32.5 vs 36.3%) | yes |
| spot | vs vote, down | **bs** ($4,814 vs $4,495) | **bs** (16.5 vs 25.5%) | vote ($2,790 vs $2,678, **flip**) | **bs** (19.8 vs 21.9%) | return flips (near-tie) |
| spot | vs evidence, up | **bs** ($4,096 vs $3,898) | evidence (15.0 vs 14.4%, **bs deeper**) | **bs** ($2,212 vs $1,944) | **bs** (15.7 vs 19.5%) | DD flips (near-tie on BTC) |
| spot | vs evidence, down | **bs** ($4,814 vs $4,424) | bs (16.5 vs 16.0%, **bs deeper**, near-tie) | **bs** ($2,678 vs $2,284) | **bs** (19.8 vs 24.1%) | yes |
| futures | vs vote, up | vote ($18,367 vs $16,104) | **bs** (24.4 vs 36.0%) | vote ($7,330 vs $4,087, **wider**) | vote (43.9 vs 36.1%, **DD flips**) | **DD ordering reverses** |
| futures | vs vote, down | **bs** ($5,821 vs $4,914) | **bs** (17.6 vs 22.3%) | vote ($2,358 vs $2,291, **flip**, near-tie) | **bs** (26.6 vs 27.8%) | return flips (near-tie) |
| futures | vs evidence, up | **bs** ($5,378 vs $3,897) | **bs** (19.2 vs 20.1%) | **bs** ($2,291 vs $2,079) | **bs** (26.6 vs 36.9%) | yes |
| futures | vs evidence, down | **bs** ($5,821 vs $3,771) | **bs** (17.6 vs 21.3%) | **bs** ($2,291 vs $2,079) | **bs** (26.6 vs 36.9%) | yes |

**P3: the ordering does not cleanly replicate.** Five of eight cells hold
their BTC ordering exactly; three do not, and one of the three is not a
near-tie: **the futures `vs vote, match-up` drawdown ordering reverses
outright** — `bayes_stein` is 12pp *shallower* than `vote` on the BTC
control (24.4% vs 36.0%) and 8pp *deeper* on ETH (43.9% vs 36.1%). This is
the same shape and the same cell type (futures match-up) where R-31 found
its own headline reversal for the e-process gate. The other two
disagreements (`vs vote, spot/futures match-down` return ordering) are
narrow enough ($2,790 vs $2,678; $2,358 vs $2,291 — both under 5%) to be
plausibly inside noise rather than a real flip, but the pre-registered
rule does not have a tolerance band for "close enough," and a rule that
would need one after the fact is not the rule that was written down.

**P3: FAIL** on the strict pre-registered reading (ordering must
replicate in every direction). Read charitably (only the wide,
unambiguous reversal counts), 7 of 8 orderings hold — but the one that
breaks is exactly the shape (a matched-risk drawdown "edge" that a second
asset erases) this project has now seen twice (R-31's ETH futures
reversal for the e-process gate). Under either reading, the one clean
lesson is: **no BTC-holdout matched-risk drawdown edge for `bayes_stein`
should be trusted without checking it survives a second asset**, and at
least one such edge does not.

---

## Step 4 — costs

**Real fee tier (`costs`), spot holdout, 0.40% Bitstamp entry tier vs the
table's 0.10% assumption:**

| pair | arm | 0.10% final | 0.10% DD | 0.40% final | 0.40% DD | 0.40% Sharpe |
|---|---|---|---|---|---|---|
| vs vote, up | vote | $3,277 | 26.7% | $2,373 | 33.0% | 0.92 |
| vs vote, up | **bayes_stein** | $2,086 | 29.6% | **$540** | **66.6%** | **−0.48** |
| vs vote, down | vote | $2,165 | 16.6% | $1,698 | 21.8% | 0.89 |
| vs vote, down | **bayes_stein** | $1,702 | 28.0% | **$530** | **63.5%** | **−0.80** |
| vs evidence, up | evidence | $1,607 | 11.6% | $1,435 | 12.4% | 0.79 |
| vs evidence, up | **bayes_stein** | $1,441 | 17.7% | **$719** | **43.8%** | **−0.73** |
| vs evidence, down | evidence | $2,045 | 17.0% | $1,790 | 18.2% | 0.87 |
| vs evidence, down | **bayes_stein** | $1,702 | 28.0% | **$530** | **63.5%** | **−0.80** |

**This is the clearest, least ambiguous finding in the whole row.** At
0.10% both reference gates and `bayes_stein` degrade gracefully (final
balance down 10-25%, drawdown up a few points). At 0.40% the reference
gates *still degrade gracefully* — `vote` and `evidence` stay profitable
and keep positive Sharpe in every cell — while every `bayes_stein` cell
**collapses**: final balance falls 61-75% further, drawdown roughly
*doubles or triples* (29.6%→66.6%, 17.7%→43.8%), and Sharpe turns
negative in all four cells. The mechanism is exactly the turnover
difference flagged in Step 4 above: `bayes_stein` trades 5-13x as often as
the gate it is matched to at the same realized volatility, because it has
no latch to hold a decision once made, so it pays the round-trip cost
proportionally more often. This is failure mode (b) generalized — not
just "it wins by holding less notional" but "it loses badly once realistic
turnover costs are charged," which is the L-06/R-12 turnover lesson
landing on a *new* mechanism.

**Funding charged on 5x futures** (real Binance series through 2023-12,
blended forward at its own mean, `costs`): `buy_and_hold` is liquidated
outright (as in every prior row). Among the matched pairs, `bayes_stein`
is close to its reference on return in three of four cells (slightly
ahead in `vs vote up`: $3,323 vs $3,120; `vs evidence up`: $1,880 vs
$1,614) and paid *less* funding in two cells (`vote up`: $1,122 vs
$1,283; `vote down`: $369 vs $407) because holding less continuously means
paying funding on fewer 8-hour marks — but drawdown is deeper than the
reference in **all four** cells (42.6% vs 38.1%, 29.1% vs 21.2%, 24.6% vs
15.6%, 29.1% vs 17.5%), the same pattern as the un-funded holdout above.

---

## Failure-mode (d) — the gate-memory probe

Daily-mean-`conf` autocorrelation and "open" (conf > 0.5) run-length
statistics over the full 2017-2026 spot series (`memory`), not a
performance measurement and not counted toward the trials tally:

| gate | 1-day autocorr | median open-run (days) | n runs | mean conf |
|---|---|---|---|---|
| `vote` | 0.951 | 6 | 96 | 0.531 |
| `evidence` | **0.983** (highest) | 6 | 35 (fewest) | 0.145 |
| `bayes_stein` (20d, **frozen**) | **0.929** (lowest) | **3** (shortest) | **108** (most) | 0.202 |
| `bayes_stein` (60d) | 0.978 | 4 | 68 | 0.231 |

**This is the one place the pre-registered mechanism difference shows up
cleanly, and in the predicted direction.** At the frozen 20-day span,
`bayes_stein` has the *lowest* day-to-day persistence of all four gates
(autocorrelation 0.929, below `vote`'s 0.951 and well below `evidence`'s
0.983), the *shortest* median "open" episode (3 days against 6 for both
references), and by far the *most* distinct open episodes (108 over the
period, against 96 for `vote` and only 35 for `evidence`). This is the
direct, measured confirmation of the mechanism claim in the "not a
duplicate" section: with no accumulator, the confidence signal opens and
closes on the timescale of its own estimation window rather than
persisting across years.

**But the distinction is graded, not categorical, and honesty about that
matters.** The 60-day variant's autocorrelation (0.978) sits much closer
to `evidence`'s (0.983) than to `vote`'s (0.951) or to the 20-day
variant's own (0.929) — lengthening the shrinkage window measurably moves
the gate's *behavior* toward the e-process's slow-accumulation regime,
even though the *mechanism* (no state variable, no wealth, re-estimated
from scratch every bar) never changes. Pre-registered failure mode (a)
("shrinkage collapses to a smoothed momentum indicator... if it moves as
slowly as the e-process, the distinction is cosmetic") is **not** true at
the frozen 20-day configuration — it is a real, measured, and reasonably
large behavioral difference — but it would very plausibly become true at
a long enough span, which is worth recording as a boundary condition
rather than leaving implicit.

---

## Deflated Sharpe

Inner-validation Sharpe dispersion across this session's 27 frontier
configurations (54 rows: 27 configs × 2 markets) is **sd = 0.270** — the
same order of magnitude as R-28's 0.223 and R-32's independently-measured
0.222, another small confirmation that these dispersion estimates are
stable across sessions and search designs. Using the one VALID holdout
cell with the cleanest signal (futures, vs vote match-up, `bayes_stein`
k=1.820, daily Sharpe 1.275 on 1,319 observations, skew 3.03, kurtosis
23.9 — this equity curve is heavily fat-tailed, which the deflation
formula accounts for):

- against this **session's own 45 trials** (27 frontier + 18 plateau):
  **DSR = 0.920** — under the conventional 0.95 bar on this session's
  search alone.
- against the **project-level trials count**, 103 (pre-R-28 floor) + 36
  (R-31) + 33 (R-32) + 45 (this row) = **217**: **DSR = 0.862**.

Neither clears 0.95. Consistent with R-28/R-29/R-32: no Sharpe-based claim
from this dataset is supportable at this point in the project's history,
independent of anything else in this row — which is one more reason Q1/Q2
above are judged on the bootstrap interval (which already accounts for
none of this) rather than on the raw Sharpe.

---

## Step 4 — path sensitivity (40 random windows)

R-19/R-31 design: 40 random windows (90–730 days), identical windows
across every strategy, carrying the frozen spot exposures (`windows`):

| market / pair | Δ median return (bs − ref) | bs higher in | Δ median DD (bs − ref) | bs deeper in |
|---|---|---|---|---|
| spot / vs vote, up | −11.8pp | 25% | −1.9pp | 32% |
| spot / vs vote, down | −2.9pp | 32% | +1.8pp | 62% |
| spot / vs evidence, up | −6.0pp | 30% | −0.1pp | 45% |
| spot / vs evidence, down | −8.1pp | 22% | +1.0pp | 55% |
| futures / vs vote, up | **+26.8pp** | **75%** | +3.9pp | 68% |
| futures / vs vote, down | −5.9pp | 28% | +2.7pp | 72% |
| futures / vs evidence, up | −3.1pp | 38% | **−4.1pp** | **25%** |
| futures / vs evidence, down | −13.0pp | 30% | **−4.0pp** | **35%** |

**No consistent winner.** On spot, `bayes_stein` loses on median return in
all four pairs (higher in only 22–32% of windows) and is deeper on
drawdown in two of four. On futures the picture reverses direction
depending on *which* reference and *which* direction: `bayes_stein` wins
median return 75% of the time against `vote`/match-up (the one clearly
favourable cell in the whole table) while also drawing down more often
there (68%); against `evidence` it loses return most of the time but is
*shallower* on drawdown in 65–75% of windows. This is the same shape as
R-31/R-32's own window tables — directionally noisy, no cell clean enough
to read as a path-level finding — and it corroborates rather than
contradicts the interval result above: eight paired comparisons, eight
different signs and magnitudes, no stable ordering. Given the interval
result already governs the decision (Q1/Q2), this table is read as
supporting evidence, not a separate claim (the same convention R-31/R-32
used for their own window tables).

---

## Verdict

**Configurations evaluated this session: 45** (27 frontier + 18 plateau,
per the R-28 convention of counting a diagnostic neighbourhood separately
from the primary sweep but adding both to the trials tally). Combined
with the project total after R-32 (103 + 36 + 33 = 172), the running total
this row leaves the project at **217**.

**Holdout counter.** This row reads the 2023+ holdout 8 times for the
matched pairs (`holdout`) across two markets, plus `interval`'s bootstrap
(reused resamples of runs already computed, not new consultations, same
accounting convention R-30 established), plus 6 re-runs at the 0.40%
spot tier and 8 at the funding-charged futures tier (`costs`) — **~22
consultations this row**. The falsification test (`eth`) and the
window resample (`windows`) do not touch the 2023+ BTC holdout, per the
R-19/R-28/R-31 convention. Taking R-32's last recorded figure of ~124 as
the base, this row leaves the project's holdout counter at **~146**.

**Scoreboard against the pre-registered decision rules:**

| rule | result |
|---|---|
| **Q1** (Δ log growth, matched risk) established? | **NO** — 8/8 intervals contain zero |
| **Q2** (Δ max drawdown, matched risk) established? | **NO** — 8/8 intervals contain zero (point estimate leans "bs deeper" in all 8, but none clears its bar) |
| **V** (validity gate) | 4/8 cells VOID (all of spot); 4/8 VALID (all of futures) |
| **P1** (beats `buy_and_hold` on spot holdout) | **FAIL** — best matched spot cell is $2,086 against $3,839, and every spot cell is void besides |
| **P2** (+0.2 Sharpe or ≥10pp DD vs hold) | moot, gated by P1 |
| **P3** (ETH falsification, ordering replicates) | **FAIL** on the strict pre-registered reading — 5/8 orderings hold, one reverses outright (futures vs-vote-up drawdown), two flip on narrow, plausibly-noise margins |
| **P4** (plateau in span / z_clip) | **PASS** — flat neighbourhood in span 15–60d; z_clip inert throughout |
| Causality probe | **PASS** — 0.0 max column difference before the cut, all three spans |
| Deflated Sharpe (this session's 45 trials / project's 217) | **0.920 / 0.862** — neither clears 0.95 |

**Promotion bar:** requires Q1 or Q2 established in every valid cell
**and** P1–P4 all pass. **Not reached — P1 fails outright and Q1/Q2 are
not established even where V permits scoring them.**

### Verdict: **NEGATIVE**

The pre-registered prediction was correct in its main claim: at matched
realized risk, `bayes_stein` is statistically indistinguishable from both
`vote` and `evidence` on the 2023+ holdout (Q1/Q2, 8/8 intervals contain
zero), matching R-31/R-32's finding that gate *mechanism* does not appear
to be the thing that matters once exposure is controlled for. This row
adds three things to that finding rather than only reproducing it:

1. **The mechanism is genuinely, measurably different** — not just on
   paper. The gate-memory probe (failure mode (d), the one prediction
   this row made that was specific to Bayes-Stein rather than borrowed
   from R-28/R-31) confirms it directly: at the frozen 20-day span,
   `bayes_stein`'s confidence signal has the lowest day-to-day
   persistence and the shortest "open" episodes of all three gates
   (autocorrelation 0.929 vs `vote`'s 0.951 and `evidence`'s 0.983;
   median open-run 3 days vs 6 for both references), exactly the
   behavioral signature a no-accumulator estimator should have and the
   e-process should not. This is a real, positive, measured finding
   inside an otherwise negative row.
2. **That same property is actively harmful, not neutral, once realistic
   costs are charged.** No latch means no discount on turnover: at the
   project's own 0.10% fee assumption `bayes_stein` is merely worse
   (inside noise); at Bitstamp's real 0.40% entry tier it **collapses** —
   final balance falls a further 61–75%, drawdown roughly doubles or
   triples, Sharpe turns negative in every one of the four spot cells —
   while `vote` and `evidence`, matched to the identical realized
   volatility, both stay profitable with positive Sharpe. This is a
   *new* instance of the L-06/R-12 turnover lesson landing on a mechanism
   that had never been tested against it: continuity in the confidence
   signal is not free, and the incumbent's latch was quietly buying
   something (turnover discipline) that this design gave up in exchange
   for a property (no accumulation lag) the holdout ended up not
   rewarding.
3. **The one holdout-favourable drawdown result does not survive a
   second asset**, the same shape R-31 found for R-28's headline: the
   futures `vs vote, match-up` cell is the row's best-looking valid
   number for `bayes_stein` on the BTC control (24.4% DD vs `vote`'s
   36.0% in the ETH/BTC-window replication), and it **reverses** on ETH
   (43.9% vs 36.1%) — P3 fails on exactly this cell.

Taken together this is a clean, informative negative: a genuinely
different statistical mechanism for treating parameter uncertainty in the
signal path (empirical-Bayes shrinkage vs. R-28's anytime-valid testing),
implemented and verified to behave differently from its predecessor, that
still lands in the same place R-31/R-32 already mapped — indistinguishable
at matched risk — and that additionally demonstrates a distinct new way
to lose (turnover) that neither of the ledger's two prior gates
exhibited. Per ROUTINE.md, this is exactly the kind of session the
routine is built to produce: nothing is promoted, and the record is
better for having tried.

### Next step, if this line is picked up again

The natural follow-up this row's own data points to is a **latched or
hysteresis-banded version of the shrinkage confidence** — e.g. only
update `conf` when the shrinkage-implied position moves by more than a
band, or apply the deadband in *confidence* space in addition to
*position* space — which would test whether the turnover collapse in the
costs section is a fixable implementation detail or an intrinsic property
of continuous confidence-weighting. That is a new, disjoint experiment
(a different sizer/deadband interaction, not a different gate), not a
rerun of this one, and it should be pre-registered and costed against the
0.40% tier from the start rather than discovered there as this row
discovered it.
