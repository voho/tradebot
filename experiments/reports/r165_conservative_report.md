# R-165 · conservative branch — trade-to-the-boundary on `scale` alone

**Verdict: NEGATIVE (REJECT).** Code: `experiments/r165_conservative_boundary.py`.
Pre-registration (frozen, unedited): `experiments/r165_shared.py`.
CSVs: `experiments/reports/r165_conservative_{diagnostic,sweep,holdout}.csv`.

---

## Mechanism, in one sentence

Insert one internal state variable `eff_scale` between `kelly_regime_v4`'s
volatility-target ratio `scale` and the product `desired = frac * scale`, and
let it track `scale` by the proportional-cost optimal policy — a no-trade
region of half-width `sub = (1-k)*w` with a **trade-to-the-nearest-boundary**
destination (Constantinides 1986, *JPE* 94(4); Davis & Norman 1990, *Math. OR*
15(4)) — leaving `frac`'s three-anchor vote, the high/low volatility-breakout
hysteresis state machine and v4's own `deadband` position update byte-for-byte
unchanged.

```python
gap = scale - eff
if   gap >  sub: eff = scale - sub          # move only to the near boundary
elif gap < -sub: eff = scale + sub
desired = frac[i] * eff                     # v4: frac[i] * scale
if abs(desired - pos) > self.deadband: pos = desired   # v4's, untouched
```

`sub = (1-k)*w`, so **k = 1 ⇒ sub = 0 ⇒ `eff == scale` at every bar ⇒
`kelly_regime_v4` bit-for-bit** (the regression check, asserted in
`cmd_causality` and confirmed again in the backtest: `dlog = +0.0000
[+0.000, +0.000]`, identical final balances to the cent on all four
inner cells). k = 0 is the widest, most-lagged member of the family.

**Declared deviation from the brief's wording.** The brief describes `k=0` as
"never move / frozen". A literally frozen `eff_scale` is a constant, which is
not a member of the trade-to-boundary family at all and would make `k` index
two different objects at its two ends. `k` therefore indexes the sub-band
width as a fraction of `w`, monotone from "no band" (k=1 ≡ v4) to "the full
band" (k=0 ≡ maximally lagged, fewest trades). Both endpoints that matter are
honoured: k=1 reproduces v4 exactly, k=0 trades least. This is stated in the
implementation file's docstring, written before the first sweep ran.

**Why this is not R-64 re-run.** R-64 applied the same destination policy to
the *whole product*, and its conservative arm died of a side effect: a
no-trade region on `desired` means the position never returns to exactly flat,
so it carried a residual long through every bear regime. Here the region is on
`scale` only and `frac` still multiplies it, so `desired` reaches exactly 0
whenever the vote does — R-64's specific killer is structurally absent, and
the D0 risk-match numbers below confirm it (mean notional within 0.24% /
1.20% of v4's on the holdout, time-in-market identical to 0.00%).

---

## Configurations evaluated

| what | count |
|---|---|
| distinct arm parameterizations in the grid (k × w) | **10** |
| …of which distinct *mechanisms* after aliasing (see below) | **7** |
| arm cells backtested in the inner sweep (10 × 2 splits × 2 markets) | 40 |
| paired `kelly_regime_v4` baselines for those cells | 40 |
| arm cells backtested at freeze time (holdout ×2 markets, 0.40% stress, 2 D5 neighbours, vs-`buy_and_hold`, ETH-A ×2) | 8 |
| paired baselines for those cells | 8 |
| **total backtests run by this branch** | **96** |
| free (zero-backtest) diagnostics: half-life, band-hit/turnover accounting, causality probe | 0 |

**Holdout consultations added by this branch: 12** backtests reading bars
dated ≥ 2023-01-01 (6 arm + 6 benchmark: 2 markets × {arm, v4} at 0.10%, spot
× {arm, v4} at 0.40%, 2 neighbours × {arm, v4}, and one `buy_and_hold`
baseline paired with an arm re-run). ETH-A (Bitfinex 2016-03 → 2019-12) is
entirely pre-2020 and costs zero. The sweep loaded BTC **truncated at
2022-12-31**, so no holdout bar was reachable during iteration.

**Aliasing, used as a consistency check rather than as extra evidence.**
`(k=0.5, w=0.10)` ≡ `(k=0.0, w=0.05)` (sub 0.050), `(k=0.75, w=0.10)` ≡
`(k=0.5, w=0.05)` (sub 0.025), and `k=1.0` at either `w` is v4. All alias
pairs reproduced each other to every printed digit — an internal check that
the mechanism depends only on `sub`, as it must.

---

## Pre-work: is the mechanism live at all?

Free diagnostics on the inner splits (`r165_conservative_diagnostic.csv`):

| split | sub | boundary hits | mean carried \|gap\| | target-path steps | turnover Σ\|Δpos\| |
|---|---|---|---|---|---|
| inner-train | 0.000 (=v4) | 385,197 | 0.0002 | 518 | 119.58 |
| inner-train | 0.025 | 130,472 | 0.0253 | 507 | 116.47 |
| inner-train | 0.050 | 125,964 | 0.0503 | 491 | 112.41 |
| inner-train | 0.100 | 110,027 | 0.1003 | 491 | 107.96 |
| inner-val | 0.000 (=v4) | 206,865 | 0.0001 | 266 | 64.35 |
| inner-val | 0.050 | 82,110 | 0.0501 | 253 | 61.74 |
| inner-val | 0.100 | 58,795 | 0.1001 | 259 | 59.32 |

The mechanism operates, and its ceiling is small by construction: at the
widest band it removes ~5% of target-path steps and ~10% of turnover. It
cannot do more, because `desired = frac·eff` with `frac ≤ 1`, so a sub-band of
`w` perturbs `desired` by at most `w` — exactly the size of v4's own position
deadband, which then absorbs most of the perturbation. This was named as
failure mode 1 in the implementation file before any backtest ran.

**A measurement-provenance note, reported and not acted on.**
`r165_shared`'s docstring states `scale`'s feeding series (v4's 8-day EWM vol)
has a causal inner-train half-life of **47.2 days**. Re-running the shared
file's own helper (`realized_vol_series` + `causal_autocorr_halflife_days`,
defaults, `fit_end=2020-12-31`) on the canonical BTC frame gives **38.8 days**
(ACF(1d) = 0.968, n = 1,460 daily obs) — identical under `load_dataset` and
`load_ohlcv_csv`, 21.2 d at `max_lag_days=30`, 39.9 d at 90, 28.0 d with the
pre-2017 file appended. I could not reproduce 47.2 with any variant. Nothing
was edited in `r165_shared.py`. It does not change any gate: every variant
(21–40 days) is far outside the pre-registered 1–15-day falsification band, so
the shared file's "half-life inside the 1–15 day band ⇒ NEGATIVE by
construction" clause does **not** fire either way. It is worth recording that
the helper's own `order_of_magnitude_gap` boolean *does* flip between the two
numbers (`in_same_band=True` at 38.8 d, `False` at 47.2 d, because the ratio
crosses 10.0×) — a pre-registered switch sitting on a knife-edge. The frozen
prose's explicit 1–15-day band governs, and it does not fire.

---

## Inner-train / inner-validation sweep

Arm minus `kelly_regime_v4`, total log growth, paired stationary block
bootstrap (30-day blocks, 2,000 resamples, seed 7). Spot @ 0.10%, futures 5x @
0.05%. **The holdout was not read at this stage.**

| k | w | sub | spot inner-train | spot inner-val | fut inner-train | fut inner-val |
|---|---|---|---|---|---|---|
| 0.00 | 0.05 | 0.0500 | −0.0424 | +0.0201 | −0.0384 | −0.0189 |
| 0.00 | 0.10 | 0.1000 | −0.0863 | +0.0631 | −0.4527 | −0.1304 |
| 0.25 | 0.05 | 0.0375 | −0.0313 | +0.0174 | −0.0584 | −0.0002 |
| 0.25 | 0.10 | 0.0750 | −0.0409 | +0.0544 | −0.3161 | −0.0762 |
| 0.50 | 0.05 | 0.0250 | +0.0008 | +0.0162 | +0.0188 | −0.0286 |
| 0.50 | 0.10 | 0.0500 | −0.0424 | +0.0201 | −0.0384 | −0.0189 |
| 0.75 | 0.05 | 0.0125 | +0.0088 | +0.0088 | +0.0261 | −0.0154 |
| 0.75 | 0.10 | 0.0250 | +0.0008 | +0.0162 | +0.0188 | −0.0286 |
| 1.00 | any | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

**One** of the 32 non-v4 inner cells has an interval excluding zero, and it is
the one that undercuts the cost story rather than supporting it: `k=0.75,
w=0.05` (sub 0.0125), inner-val spot, Δlog +0.0088 [+0.0009, +0.0173] and
ΔSharpe +0.0158 [+0.0016, +0.0302] — a cell whose fill count did **not** fall
(468 vs 467 on inner-train, 256 vs 256 on inner-val, 145 vs 143 on inner-val
futures). It is the smallest perturbation in the grid, it trades marginally
*more* than v4, and it is exactly the cell the frozen S1 screen disqualifies.
Every other interval, including the widest point estimate (−0.4527, carrying
[−1.231, +0.128]), contains zero.

**The selection surface is noise, and it says so before the holdout was
touched.** Across the 8 non-v4 cells, the rank correlation between the
inner-train and inner-validation log-growth difference is **−0.93 on spot**
(3/8 sign agreements) and +0.44 on futures (5/8). The mechanism helps on spot
inner-validation exactly where it hurts on spot inner-train. That is R-64's
own diagnosis of its novel arm reproduced on a different mechanism, and it is
the reason the holdout result below should surprise nobody.

---

## The frozen configuration

**`BoundaryScale(k=0.25, band_w=0.05)` — sub-band 0.0375**, every other
constant `kelly_regime_v4`'s shipped default.

Selected by the rule written into the implementation file's docstring *before
the first sweep ran* and applied mechanically by `select_frozen()`:

- **S1 (mechanism operates):** total fill count strictly below v4's on both
  inner splits, spot. Passed by 7 of 8 candidates (k=0.75, w=0.05 failed).
- **S2 (risk match, ROUTINE's first standing rule):** mean notional and
  time-in-market within 10% of v4's on both splits and both markets. Passed by
  all 8.
- **SELECT:** highest inner-**validation** mean log-growth difference across
  {spot, futures} among the S1∧S2 survivors; ties → larger sub-band.
  Winner: k=0.25, w=0.05 at +0.0086 (spot +0.0174, futures −0.0002).
- k=1 was excluded from candidacy: it *is* the incumbent.

The rule did not move at any point, and the fallback branch (freeze the
a-priori theory-literal k=0, w=0.10 cell) was not needed.

---

## Holdout, run once (2023-01-01 → 2026-08-12)

Frozen arm vs `kelly_regime_v4`, paired block bootstrap, same settings.

| cell | arm final | v4 final | arm Sharpe | v4 Sharpe | arm DD | v4 DD | arm fills | v4 fills | arm fees | v4 fees | Δlog growth [95%] | ΔSharpe [95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spot @0.10% | $3,365 | $3,373 | 1.22 | 1.22 | 27.9% | 27.8% | 318 | 332 | $309 | $310 | **−0.0023 [−0.0257, +0.0228]** | −0.0001 [−0.0203, +0.0214] |
| futures_5x @0.05% | $4,832 | $4,901 | 1.36 | 1.36 | 32.8% | 33.0% | 326 | 328 | $257 | $265 | **−0.0142 [−0.0616, +0.0302]** | +0.0006 [−0.0231, +0.0273] |
| spot @0.40% (D2) | $2,443 | $2,445 | 0.94 | 0.94 | 34.1% | 34.1% | 318 | 332 | $1,026 | $1,027 | −0.0006 [−0.0267, +0.0281] | — |
| ETH-A @0.10% (D3) | $6,764 | $5,482 | 1.52 | 1.48 | 39.7% | 36.5% | 415 | 448 | $412 | $349 | +0.2101 [−0.2061, +0.7809] | — |
| ETH-A @0.40% (D3) | $5,320 | $4,269 | 1.36 | 1.30 | 43.5% | 41.7% | 415 | 448 | $1,434 | $1,207 | +0.2201 [−0.198, +0.800] | — |
| neighbour k=0.00 | $3,365 | $3,373 | 1.219 | 1.22 | 27.8% | 27.8% | 327 | 332 | $309 | $310 | −0.0025 [−0.0266, +0.0225] | — |
| neighbour k=0.50 | $3,347 | $3,373 | 1.215 | 1.22 | 27.9% | 27.8% | 318 | 332 | $307 | $310 | −0.0079 [−0.0275, +0.0115] | — |

Round trips (`num_trades`, flat-to-flat episodes) are **51 vs 51** on both
markets — carried separately from the fill count per ROUTINE's standing rule,
because they are the two different turnover units and only the fill count
moves here.

Standing bar vs `buy_and_hold`, holdout spot @0.10%: arm $3,365, v4 $3,373,
`buy_and_hold` **$3,839** (Sharpe 1.03, max DD 54.0% against the arm's 1.22
and 27.9%). Neither the arm nor v4 beats hold on raw growth over this holdout;
that is a property of the incumbent, unchanged by this mechanism, and the
paired comparison is D0-void anyway (hold carries ~1.7× the arm's exposure).

---

## Decision rule D0–D6 (`r165_shared`, frozen before any number was read)

| gate | outcome | evidence |
|---|---|---|
| **D0** risk-match | **PASS** | spot: mean notional 0.5857 vs 0.5871 (+0.24%), time-in-market 70.8% vs 70.8% (+0.00%), realized vol 0.320 vs 0.321 (+0.22%). futures: 0.6776 vs 0.6858 (+1.20%), TIM +0.00%, vol +1.13%. All well inside the 10% tolerance — R-64's never-returns-to-flat artifact is genuinely absent when the band sits on `scale` alone. |
| **D1** holdout vs v4 | **FAIL** | Requires an interval excluding zero on the favourable side for ≥1 of {log growth, Sharpe} on **both** markets. Neither metric excludes zero on either market; both log-growth point estimates are mildly **negative** (−0.0023 spot, −0.0142 futures). |
| **D2** cost mechanism | **PASS as written, and vacuously so** | Requires Δlog(0.40%) > Δlog(0.10%): −0.0006 > −0.0023, so it passes. But the arm is *behind* v4 at both tiers — this is a deficit shrinking toward zero as the fee rises, not an advantage growing. The saved fee stream is real (fills 318 vs 332) and it is worth $1 of $1,027 at the 0.40% tier. Reported as the frozen rule computes it, with the arithmetic stated so nobody reads it as support. |
| **D3** ETH-A falsification | **PASS** | +0.2101 [−0.2061, +0.7809] at 0.10% and +0.2201 at 0.40% — same sign as BTC's… except BTC's sign is negative, so "not reversed" is satisfied only in the weak sense that ETH's interval also contains zero. The ETH point estimate is 90× the BTC one on a shorter, noisier series; it is not evidence of anything. |
| **D4** turnover falls | **PASS** | Fills 318 vs 332 (−4.2%) spot, 326 vs 328 (−0.6%) futures; fees $309 vs $310 and $257 vs $265. The mechanism operated; it is simply too small to matter. |
| **D5** plateau not peak | **PASS** | Neighbour arm Sharpe 1.219 (k=0.00) and 1.215 (k=0.50) against the frozen cell's 1.219 — gaps of 0.000 and 0.004 against the ±0.2 noise floor. A plateau, and an indistinguishable one: every member of the family, including v4 itself, is the same strategy to within measurement error on this holdout. |
| **D6** funding | **not run** | `r165_shared` runs it only if D1–D5 all pass. D1 failed. |

**Verdict: REJECT / NEGATIVE.** `r165_shared`'s default is REJECT and its bar
requires D1 (both markets, either metric), D2, D3 and D4 to hold
simultaneously; D1 fails. The rule was not modified after any number was read,
no threshold was revisited, and nothing was re-scored after the fact.

Supporting causality evidence (the class is unregistered, so
`tests/test_causality_strict.py` does not reach it; the probe is carried
in-file and all parts PASS): k=1 target path bit-identical to
`KellyRegimeV4`'s at both `w`; truncation at two cut points reproduces the
prefix exactly; ×3 and ÷3 tampering of all later bars leaves every earlier
target unchanged and the two tampers agree; and — the R-21 check — the
**orders** queued at 8 bars where the arm actually trades are identical under
opposite tampers of every subsequent bar (8 orders, non-vacuous).
`pytest tests/test_causality_strict.py -q` also passes (55 passed) against the
unmodified repo.

---

## Lesson

**Moving the no-trade region from the product onto the `scale` factor alone
removes R-64's exposure artifact completely (risk matched to within 1.2%) and
in exchange removes the mechanism itself: `frac ≤ 1` means a band of width `w`
on `scale` perturbs the target by at most `w`, which is precisely what v4's own
0.10 position deadband already absorbs — so the policy cuts 4% of fills, $1 of
fees at the 0.40% tier, and nothing else.**

Secondary, and the more transferable half: the spot inner-train/inner-val rank
correlation of **−0.93** across the 8 candidate cells announced that this
selection surface was noise *before* the holdout was opened, and the holdout
agreed (the inner-val-selected cell landed at −0.0023). A cheap pre-holdout
diagnostic — rank-correlate the two inner splits over the candidate grid —
would have predicted this round's outcome for free.
