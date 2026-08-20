# R-60 (conservative branch) — per-asset OU half-life anchor rescaling (08-20)

Backlog **B-26**. Pre-registration (shared, both branches): `experiments/r60_shared.py`.
Implementation: `experiments/r60_conservative_ou_halflife.py`. Raw cells:
`reports/r60_conservative/*.csv`. Nothing under `src/tradebot/strategies/` is
touched — `kelly_regime_v4` is used exactly as shipped, with `horizons` passed
as the constructor argument it already accepts; the vote mechanism (moving-
average crossing, 1% band, latching hysteresis), the conditional fractional-
Kelly vol targeting, the 10% deadband, `target_vol=0.55` and `max_leverage=2.0`
are unchanged for every asset.

## 1. The question, one sentence

R-57 found `kelly_regime_v4`'s one surviving property — a matched-exposure
drawdown advantage — inverts on 6 of 6 further Coinbase instruments; R-59
tested the hypothesis that the vote/gate's exposure **scale** (`target_vol`,
or a dimensionless relative-vol scale) was the binding constraint and both
branches failed identically (0-for-19 cumulative on the SIZE axis). This
round (B-26) asks whether the vote's **timing** — how fast the (20,40,80)-day
anchor ladder reacts, rather than how big the resulting exposure is — is the
binding constraint instead: does rescaling each asset's anchor ladder to its
own structurally-estimated mean-reversion half-life restore the property?

## 2. Mechanism

For each of 8 assets (BTC, ETH, and R-57's six-asset panel), an
Ornstein-Uhlenbeck mean-reversion half-life is estimated by OLS-through-the-
origin on daily-resampled log-close over PANEL_TRAIN/CONTROL
(2020-04-01 → 2022-12-31, the same calendar range under two names):

```
p_t   = log(close_t), one obs/day
mu    = mean(p_t) over the window
x_t   = p_{t-1} - mu
y_t   = p_t - p_{t-1}
beta  = sum(x*y) / sum(x*x)          (no intercept — verified against an
                                       OLS-with-intercept variant before
                                       freezing this choice; intercepts
                                       differ from zero by <=0.002 log-price
                                       units on all 8 assets, i.e. the two
                                       variants agree to 4 significant
                                       figures, so nothing was chosen because
                                       one scored better on anything)
theta = -beta                        (mean-reversion speed, per day)
halflife = ln(2) / theta             (calendar days)
```

BTC is the reference asset (`kelly_regime_v4`'s own fitting asset). For every
asset i: `ratio_i = halflife_i / halflife_BTC`, clamped to `[18/20, 28/20] =
[0.90, 1.40]` — the R-07-validated 18–28-day plateau bound for the *shortest*
anchor, the identical derivation logic R-57's own panel-selection amendment
used (a validated empirical bound converted into a clamp via the same
algebra, not a threshold invented for this round). Horizons become
`(20, 40, 80) * ratio_clamped_i`, preserving the ladder's own doubling
structure. By construction `ratio_BTC = 1.0` exactly (a number divided by
itself), so BTC's own candidate is byte-identical to shipped `kelly_regime_v4`
— named in the pre-registration before any BTC number was computed, as an
algebraic consequence, not a data finding.

**Correctness fix required and applied**: each candidate's `.warmup` instance
attribute is overridden to `int(round(max(horizons_i) * BARS_PER_DAY)) + 10`
(generalizing v4's own `80 * BARS_PER_DAY + 10`), so an asset whose ladder is
scaled *up* still enters the measured window with a fully-formed longest
anchor — otherwise this would reproduce an R-22-class warmup-prefix bias.

## 3. Per-asset calibration table (structural, price-only, computed before any
backtest was run)

| asset | theta (per day) | half-life (days) | ratio (raw) | ratio (clamped) | clamp | horizons (days) |
|---|---|---|---|---|---|---|
| BTC (ref) | 0.003951 | 175.44 | 1.000 | 1.000 | — | (20.0, 40.0, 80.0) |
| ETH | 0.004117 | 168.35 | 0.960 | 0.960 | — | (19.2, 38.4, 76.8) |
| BCH | 0.002279 | 304.10 | 1.733 | **1.400** | **CEILING** | (28.0, 56.0, 112.0) |
| LTC | 0.005027 | 137.87 | 0.786 | **0.900** | **FLOOR** | (18.0, 36.0, 72.0) |
| ETC | 0.003939 | 175.99 | 1.003 | 1.003 | — | (20.1, 40.1, 80.3) |
| DASH | 0.004167 | 166.33 | 0.948 | 0.948 | — | (19.0, 37.9, 75.8) |
| LINK | 0.006901 | 100.44 | 0.573 | **0.900** | **FLOOR** | (18.0, 36.0, 72.0) |
| XTZ | 0.004439 | 156.16 | 0.890 | **0.900** | **FLOOR** | (18.0, 36.0, 72.0) |

**Anomaly worth flagging**: the clamp binds for **4 of the 6 panel assets**
(LTC, LINK, XTZ at the floor, BCH at the ceiling) — only ETC and DASH land in
the unclamped interior. The structural half-life estimate's own dynamic
range (100–304 days across the panel, a ~3x spread) is wide enough that the
R-07-validated plateau bound saturates it on two-thirds of the panel. This
means the "structural, data-derived" calibration mostly collapses to one of
two fixed points (18 or 28 days shortest-anchor) rather than expressing eight
genuinely distinct ladders — worth naming plainly rather than presenting the
per-asset table as more differentiated than it actually is.

## 4. Causality tamper probe

R-57/R-59's tamper methodology, adapted to construct `KellyRegimeV4(horizons=
horizons_i)` directly (this branch's candidate is never registry-default
`kelly_regime_v4` except for BTC). Run on BTC and 3 panel assets (BCH, LTC,
ETC), each with its own rescaled horizons and correctly-extended warmup:

**PASS on all 4** — decisions at and before the tamper cut bar are
bit-identical under opposite post-cut price/volume tampers (×3/÷3 price,
×7/÷7 volume), for every horizon ladder tested, including the CEILING-clamped
BCH (112-day longest anchor) and FLOOR-clamped LTC (72-day longest anchor).
No lookahead bug.

## 5. D1 (primary) — PANEL_TRAIN, spot @0.10%

| asset | horizons (d) | cand max DD | matched hold max DD | Δ DD (pp, + = cand worse) | 95% paired interval |
|---|---|---|---|---|---|
| BCH | (28.0,56.0,112.0) | 49.2% | 38.5% | **+10.2** | [−6.3, +37.7] |
| LTC | (18.0,36.0,72.0) | 38.3% | 38.9% | **−1.4** | [−11.4, +24.9] |
| ETC | (20.1,40.1,80.3) | 40.0% | 25.0% | **+15.4** | [+1.2, +37.0] |
| DASH | (19.0,37.9,75.8) | 44.5% | 29.3% | **+15.0** | [−2.2, +36.3] |
| LINK | (18.0,36.0,72.0) | 45.1% | 31.1% | **+16.5** | [−4.1, +37.3] |
| XTZ | (18.0,36.0,72.0) | 48.4% | 30.2% | **+18.3** | [+4.0, +43.2] |

**1 of 6.** Only LTC beats the matched hold, and only marginally (−1.4pp,
interval spans zero — not a real win). One interval (ETC) excludes zero,
against the candidate. **D1 verdict: FAILS** (exact binomial p = 0.9844).
Compared with R-57's frozen, unrescaled cells on the identical window/panel
(0/6, +5.2 to +33.8pp), rescaling the anchor ladder to each asset's own
structural half-life **moved almost nothing** — every asset's Δ DD is within
a few points of R-57's own uncalibrated number, and the one asset that
crossed zero (LTC, −1.4pp) is the FLOOR-clamped case, i.e. the shortest
possible ladder this round's own clamp allowed, not a genuinely different
calibration.

## 6. D2 (falsification control) — CONTROL window, BTC and ETH

| asset | horizons (d) | cand max DD | matched hold max DD | Δ DD (pp) | R-57 control | within 5pp tolerance? |
|---|---|---|---|---|---|---|
| BTC | (20.0,40.0,80.0) | 33.2% | 39.3% | **−5.6** | −5.6 | yes (identical — ratio_BTC=1.0 is a no-op by construction) |
| ETH | (19.2,38.4,76.8) | 27.6% | 39.9% | **−10.7** | −11.5 | yes (+0.8pp, well inside tolerance) |

**D2 PASSES.** BTC is unchanged by construction. ETH's advantage is
essentially unchanged (−10.7 vs R-57's −11.5pp), a negligible slackening well
inside the 5pp tolerance.

## 7. D3 (crash-transition-lag, gating) — BTC, three CRASH_WINDOWS, spot

| window | baseline lag (bars) | candidate lag (bars) |
|---|---|---|
| 2018-11 | 2618.0 | 2618.0 |
| 2020-03-covid | 4607.0 | 4607.0 |
| 2022-11-ftx | 1138.0 | 1138.0 |

Mean lag: baseline 2787.67 bars, candidate 2787.67 bars → **PASSES** (tolerance
+2 bars). **This cell is trivial by construction, named as such in the
pre-registration before any BTC number was read**: BTC's candidate horizons
equal `(20,40,80)` exactly (`ratio_BTC = halflife_BTC / halflife_BTC = 1.0`
for any pair of numbers), so the candidate and baseline signals are
byte-identical on BTC and the "pass" carries no information about whether
rescaling helps or hurts crash-transition timing — it only confirms the
harness reproduces v4 exactly when the scale factor is 1.0.

**D3 supplementary (non-gating, robustness only) — ETH**, 2 of 3 windows
(no ETH data before 2019-03-14, so 2018-11 is skipped):

| window | baseline lag (bars) | candidate lag (bars) | Δ (bars) |
|---|---|---|---|
| 2020-03-covid | 7406.0 | 7388.0 | −18 (candidate faster) |
| 2022-11-ftx | 1052.0 | 1088.0 | +36 (candidate slower) |

ETH's ratio (0.960) is genuinely non-trivial, and the two windows disagree in
sign — 18 bars (1.5h) faster on COVID, 36 bars (3h) slower on FTX — both well
inside a single trading session and both far smaller than the multi-thousand-
bar total lag either window takes to fully flatten. Not decisive either way,
and not part of the gate.

## 8. D4 (generalization, reported not gating) — PANEL_TEST, spot @0.10%, frozen horizons

| asset | cand max DD | matched hold max DD | Δ DD (pp) | 95% interval |
|---|---|---|---|---|
| BCH | 42.1% | 34.1% | +8.4 | [−6.5, +43.9] |
| LTC | 68.8% | 31.4% | **+38.7** | [+9.3, +55.9] |
| ETC | 50.7% | 32.1% | +18.2 | [+8.7, +47.6] |
| DASH | 45.7% | 21.5% | +25.6 | [−0.5, +35.1] |
| LINK | 41.9% | 33.7% | +7.8 | [−6.3, +32.2] |
| XTZ | 38.2% | 32.9% | +4.5 | [−4.8, +31.7] |

**0 of 6.** LTC — D1's one (weak) win — is the *worst* cell here (+38.7pp,
interval excludes zero against the candidate): the D1 near-miss did not
generalize, the identical failure mode R-59's conservative branch found for
its own single near-tie asset. Descriptive only, consistent with D1's
failure rather than a separate gate.

## 9. D5 (0.40% fee falsification) — PANEL_TRAIN, spot @0.40%, frozen horizons

Candidate beats `buy_and_hold`'s final balance in **3 of 6** (LTC, ETC, DASH).
Threshold was ≥5/6. **FAILS, as predicted before the run.**

## 10. Verdict

`experiments.r60_shared.promoted(k1=1, dd_advantage={"BTC": -5.6, "ETH":
-10.7}, candidate_lag=2787.67, baseline_lag=2787.67)` applied mechanically:
`k1 >= 5` is False, so **`promoted()` returns False** regardless of D2/D3
(both of which pass on their own). Per the pre-registration's promotion bar
(D1 ≥ 5/6 AND D2 passes on both BTC and ETH AND D3 passes):

**NEGATIVE.**

Stated precisely, because the failure mode is worth naming exactly: this is
**not** the "fix breaks BTC/ETH" mirror-risk the pre-registration flagged —
D2 passes cleanly on both, and D3's BTC cell is a trivial identity pass by
construction (the reference asset's ladder never moves). It is the plainer
failure R-59's own two branches already found on the SCALE axis, now
reproduced on the TIMING axis: rescaling each asset's anchor ladder to its
own structurally-estimated OU half-life barely moves any panel asset's Δ DD
relative to R-57's frozen, unrescaled numbers, and the clamp binds on 4 of 6
assets, meaning most of the panel's "structural" calibration collapses to one
of the two boundary values this project has already swept and found flat
(R-07's own 18–28 day plateau — "EVERY variant cut max drawdown... below ~18
days the plateau breaks sharply" — was itself evidence that the timing axis,
inside the range this round's clamp permits, does not materially change the
strategy's risk behaviour). The one asset whose ladder is genuinely
unclamped and materially different from BTC's (BCH, ratio 1.733 clamped down
to 1.400) is also the asset with the *worst* D1 cell (+10.2pp) among the
"win" candidates, going the wrong direction for the hypothesis. Combined with
R-59's own two branches, R-57's own diagnosis is now tested and not falsified
on a third independent axis: neither the sizing constant's magnitude
(R-59-conservative), its dimensional form (R-59-novel), nor — this round —
the vote's timing/cadence moves the matched hold's advantage. R-57's own
named alternative explanation (the matched hold rebalances back to a constant
fraction and is quietly a buy-the-dip rule on these higher-volatility,
more mean-reverting instruments, a benefit orthogonal to how fast or slow
`kelly_regime_v4`'s trend-following vote arrives) remains the best-supported
account and is not challenged by any of the three SIZE/TIMING-axis rounds run
against it so far.

## 11. Accounting

- **Backtest configurations evaluated (bucket a, `run_period`/`measure()`):
  60** — 6 panel assets × 3 arms (candidate, `buy_and_hold`, matched hold) ×
  1 window [PANEL_TRAIN@0.10%, D1] = 18; 2 control assets × 3 arms × 1 window
  [CONTROL@0.10%, D2] = 6; 6 panel assets × 3 arms × 1 window
  [PANEL_TEST@0.10%, D4] = 18; 6 panel assets × 3 arms × 1 window
  [PANEL_TRAIN@0.40%, D5] = 18. Total 18+6+18+18 = 60.
- **OU half-life regressions (bucket b, closed-form OLS-through-origin, no
  solver, no iteration): 8** — one per asset (BTC, ETH, and the 6-asset
  panel).
- **prepare()-only D3 signal evaluations (bucket c, not backtests — same
  convention as R-57's own causality probe, which is likewise excluded from
  the backtest count because it calls `prepare()`/`on_bar()` directly rather
  than `run_period`): 10** — BTC candidate+baseline × 3 crash windows = 6,
  ETH candidate+baseline × 2 crash windows (2018-11 skipped, no data) = 4.
- **Total evaluations, all buckets: 78.**
- **Holdout consultations: +0.** BTC and ETH frames are truncated at
  2022-12-31 immediately after loading, before any other line of code
  touches them (`load_control_assets()`); no 2023+ bar of either is ever
  read anywhere in this module. Panel-asset reads (PANEL_TRAIN or
  PANEL_TEST) cost +0 per the pre-registration (new-instrument evidence, not
  the reserved BTC/ETH holdout). The three D3 crash windows are all pre-2023
  and are already read by every registered strategy's own backtest.
- **Decision rules moved: no.** D1–D5 and the promotion bar are exactly as
  committed in `experiments/r60_shared.py`, frozen before either branch ran;
  nothing was chosen after seeing a drawdown, Sharpe, or final-balance
  number. The half-life formula, reference asset and clamp bound were fixed
  by algebra and by R-07's already-published finding, not by anything
  measured in this round (section 2's pre-registration).
- `pytest -q`: unchanged (this branch added no code under `src/`, only two
  new files under `experiments/` — this file's implementation and this
  report).

## 12. What this changes

The conservative branch's direct reading of B-26's own question — "rescale
the vote's cadence to each instrument's own structurally-measured
mean-reversion timescale" — does not restore the matched-exposure drawdown
property, and the BTC/ETH control stays intact throughout (D2 and D3 both
pass cleanly, though D3's BTC cell is trivial by construction), which
localizes the failure precisely: it is not that per-asset timing recalibration
breaks something that worked on BTC/ETH, it is that a wide (100–304 day)
range of structurally-estimated half-lives, clamped into the one region this
project has already validated (R-07's 18–28 day plateau), produces anchor
ladders that are — by the plateau's own prior finding — expected to behave
almost identically to the shipped default, and the D1 numbers confirm exactly
that expectation rather than overturning it. This closes B-26's conservative
branch: pending the novel (CUSUM) branch's own result for the round total,
this extends the strategy family's SIZE/TIMING-axis record on the panel to
**0-for-20** (R-34→R-46, R-53→R-56, R-59's two branches, this branch).
