# Downside-only / drawdown-sensitive risk denominator for the Kelly sizer

**Ledger ID:** `R-XX` (placeholder — assign on merge). **Session date:** 2026-08-18.
**Branch:** novel branch of a parallel round; the other branch worked backlog
item B-05 (funding as a gate) in disjoint files (`experiments/funding_gate.py`
and friends). Files touched by this session: **only**
`experiments/downside_sizing.py` (new) and this report (new). Nothing else
in the repo was read for the purpose of editing, and nothing else was
edited.

**Verdict up front: NEGATIVE.** Read on for why, but the short version: the
plateau check fails on the mechanism's two defining knobs before the
holdout is even touched, the holdout confirms P1 fails on both markets, and
the pre-registered falsification test does not just fail to replicate — the
ordering against the incumbent *reverses* on the second asset.

---

## 1. Step-1 justification (written before any code)

1. **Which constraint does it attack?** SIZE. The change is entirely inside
   the sizing arithmetic: `target_t = vote_t * min(target_level /
   risk_measure_t, max_leverage)`, byte-for-byte the incumbent's
   `target_vol / realized_vol` shape with only `risk_measure_t` swapped.
   The vote gate — the part of the mechanism that answers "is the regime
   bullish" — is untouched, copied from `kelly_regime_v4` verbatim
   (20/40/80-day latched anchors, 1% band). This is a "how much" question,
   not a "what happens next" predictor: it consumes the same OHLCV bars,
   produces no forecast of direction, and changes nothing about when the
   gate opens or closes.

2. **Which ledger rows is this not a duplicate of?**
   - **L-01/L-02/L-03/L-04**: all vary the *vote* (anchor count, timescale,
     convex response). None touch the sizing denominator.
   - **`kelly_regime_v3`/R-07**: switches between continuous and
     steady-state targeting of the *same symmetric* volatility at breakout
     extremes. It still cannot tell an up-burst from a down-burst; it just
     delays reacting to either. This file's semidev/CDaR measures are
     mechanically *insensitive* to up-bursts by construction, which v3's
     hysteresis is not.
   - **R-09** (range estimators: Parkinson/Garman-Klass/Rogers-Satchell/
     Yang-Zhang): different *estimators* of the same symmetric
     total-variance quantity, reading 7-18% low from discretisation bias —
     a calibration question. This file measures a structurally different,
     asymmetric quantity.
   - **R-08**: made the *same symmetric* estimate more accurate and found
     that this makes the strategy worse, because better calibration
     de-levers more promptly into BTC's good high-vol states. The
     *theoretical* bet here is different: a risk measure that zeroes out
     up-bursts should not de-lever into them at all, so it predicts the
     *opposite* direction of effect from R-08. Section 4 reports where this
     theoretical distinction survives contact with the data and where it
     does not (short version: it collapses for the semi-deviation variant
     specifically, because semi-deviation turns out to be 99.4% correlated
     with symmetric vol in this data — see the discussion under Failure
     Mode (a) below. It does not collapse for CDaR, which behaves
     genuinely differently, but that different behaviour turns out to be
     an unstable peak rather than an edge).
   - **R-11** (Grossman-Zhou drawdown cushion): a discrete multiplicative
     *brake*, `exposure *= (wealth - floor) / wealth`, bolted on top of the
     existing sizer. This file does not add a brake on top of anything — it
     *replaces* the sizer's own denominator with a continuous risk
     statistic, so downside/drawdown risk changes the position through the
     same `target / risk` channel symmetric vol used, never through a
     separate wealth-floor gate. R-11's whole-book version destroyed
     return; this file's mechanism is structurally incapable of reproducing
     that specific failure mode (there is no floor-of-wealth term anywhere)
     but is capable of its own, reported below.
   - **R-31/R-32** (matched-risk gate comparison): those rounds vary the
     *gate* at fixed sizer/exposure and conclude "the gate is worth more
     than the choice of gate" — which is exactly why this file leaves the
     gate alone and varies the *denominator* instead, the one axis their
     own conclusion says is untested by that work.

3. **Is it simulable here?** Yes. Every risk measure is derived purely from
   `close`, using `.ewm(...).shift(1)` (semi-deviation, identical
   construction to the incumbent's own vol) or a resample-to-daily /
   rolling-tail-mean / `.shift(1)` / `reindex(..., ffill)` construction
   (CDaR). No new data, no fetch. The CDaR construction is the one genuine
   novelty relative to the incumbent's plain `.shift(1)`, and it is exactly
   the kind of resample-then-reindex pattern that can silently leak a
   day's own close into that day's bars if done carelessly (see the module
   docstring's "Design notes: CDaR causality" for the argument, and Section
   5 below for the check that verified it rather than assumed it).

4. **What would make it fail — named before any code ran:**
   - **(a)** Downside semi-deviation/CDaR turns out highly correlated with
     symmetric realized volatility in this BTC data specifically (crashes
     come with elevated vol on both sides), so the sizing path barely
     differs from v3/v4 and this is a null result in disguise.
   - **(b)** A naive rolling max-drawdown-from-peak statistic resets toward
     zero right after a new all-time high, spiking exposure exactly at tops
     just before reversals — checked for explicitly if a drawdown-based
     statistic is used.
   - **(c)** Even a real effect sits inside the project's measured ±0.2
     Sharpe noise floor (R-20), like every SIZE variant tried so far except
     the drawdown property.
   - **(d)** It doesn't survive ETH replication or degrades under the 0.40%
     fee tier because a noisier denominator increases turnover.

   All four were checked. The results are in Section 7-9; the short version
   is (a) is confirmed for semi-deviation, (b) is confirmed as a real
   pathology at short CDaR windows (mitigated by the floor) but is not what
   sinks the frozen 60-day configuration, (c) doesn't quite describe what
   happened to CDaR (its problem is a peak, not noise-floor smallness), and
   (d) is confirmed on the ETH leg (the ordering reverses) though not on
   the fee-turnover mechanism named.

---

## 2. Literature

**Foundational (from the assignment brief):**

- Sortino, F. A. & van der Meer, R. (1991). "Pricing Downside Risk and
  Sizing Positions" [Downside Risk]. *Journal of Portfolio Management*,
  17(4), 27-31. Downside semi-deviation: the RMS of returns below a target
  (here, 0), ignoring upside dispersion entirely. Single-asset, no cost
  model — a risk-measure definition paper, not a backtest.
- Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005). "Drawdown Measure in
  Portfolio Optimization." *Journal of Risk*, 8(2), 13-58. Conditional
  Drawdown-at-Risk (CDaR): the mean of the worst α% of drawdowns over a
  window, in a portfolio-optimization framework. Multi-asset equity
  portfolios in the original paper, no crypto, no realistic per-trade cost
  model (it's an optimization objective, not a trading rule).

**Additional 2023-2026 sources found for this session, applied with the
R-05 discipline — what data, what cost assumption, how many instruments:**

- **Yang, A. (2025). "Cryptocurrency market risk-managed momentum
  strategies." *Finance Research Letters*.** Applies Barroso &
  Santa-Clara's (2015) *symmetric* volatility-scaling to cross-sectional
  cryptocurrency momentum. Headline: Sharpe rises from 1.12 to 1.42, and —
  the useful part for this file's hypothesis — the paper explicitly finds
  the improvement comes from **augmented returns, not downside
  mitigation**, unlike equities, "reflecting distinct market dynamics
  characterized by the absence of extended momentum crashes." This is an
  independent confirmation, from a *different* strategy family (cross-
  sectional momentum, not single-asset trend-following) and a *symmetric*
  risk measure, of the same asymmetry R-10 measured here (Baur & Dimpfl
  2018): crypto vol-scaling does not behave like the equity case it was
  designed for. **Discount:** cross-sectional strategy over (implicitly)
  many coins — instrument count and per-trade cost assumption not
  available from the abstract — so it is evidence about the *sign* of the
  asymmetry, not a transferable Sharpe number.
- **Bui, D. & Nguyen, T. (2026). "Systematic Trend-Following with Adaptive
  Portfolio Construction: Enhancing Risk-Adjusted Alpha in Cryptocurrency
  Markets." arXiv:2602.11708.** A dynamic trailing-stop mechanism
  "calibrated to intra-day volatility regimes" — the closest published
  analogue to a downside/drawdown-triggered sizing rule found in this
  search. Data: **150+ cryptocurrency pairs, 36-month window (2022-2024)**.
  Costs: **4bps taker fee**, plus a slippage and funding model. Headline:
  Sharpe 2.41, max drawdown -12.7%. **Heavily discounted, on exactly R-05's
  grounds**: 150+ instruments (diversification, not this project's single
  BTC series) at 4bps against this project's 10bps spot / 0.40bps futures
  tier — the same shape of gap (many cheap instruments vs. one expensive
  one) that got the deep-learning round rejected on grounds in R-05. Cited
  here as evidence that the *general idea* (downside/drawdown-triggered
  sizing in crypto trend-following) is an active area, not as a
  transferable performance number. Also outside the requested 2022-2025
  window by six weeks (submitted Feb 2026); flagged rather than silently
  included.
- **Sadaqat, M. & Butt, H. A. (2023). "Stop-loss rules and momentum payoffs
  in cryptocurrencies." *Journal of Behavioral and Experimental Finance*,
  39, 100833.** Cross-sectional momentum across **147 cryptocurrencies,
  January 2015 - June 2022**, overlaid with a stop-loss (a discrete,
  drawdown-triggered exit, closer in spirit to R-11's brake than to this
  file's continuous denominator) rule. Finds the stop-loss variant raises
  raw and risk-adjusted returns and outperforms across market states.
  **Discount:** 147-instrument cross-sectional design (the diversification
  trap again), a *discrete* exit rather than continuous sizing, and no
  disclosed per-trade cost assumption recovered from the abstract.

**What the literature search adds, net of discounting:** the general
direction — that crypto's volatility asymmetry (R-10, confirmed
independently by Yang 2025) makes a *symmetric* risk denominator the wrong
tool, and that downside/drawdown-triggered mechanisms are an active
published direction in crypto trend-following (Bui & Nguyen 2026, Sadaqat &
Butt 2023) — is real. None of the three 2023-2026 sources offers a
performance number this project can treat as transferable: two are
diversified across 147-150+ instruments at 2-4bps costs (exactly R-05's
red flag), and none uses this project's cost tier or a single-asset design.
The literature justifies *trying* the idea; it does not pre-certify a
result, and none is claimed on its authority.

---

## 3. Design and falsification test (chosen before any code)

Three risk-measure families were implemented behind an identical vote gate
and sizer shape (`experiments/downside_sizing.py`, class `DownsideKelly`):

- **`risk="symmetric"`** — control arm. The incumbent's own EWM standard
  deviation of *all* returns, on the v4 vote gate and a plain
  `min(target_vol/vol, max_leverage)` sizer (not v4's own conditional/
  extreme-only sizer, so that any difference from this file's other arms
  can be attributed to the risk axis and not silently to the sizer-shape
  difference between `kelly_regime` and `kelly_regime_v3`/`v4`).
- **`risk="semidev"`** — Sortino & van der Meer (1991) downside
  semi-deviation: EWM RMS of `min(r_t, 0)`, annualized, floored at
  `floor_frac * symmetric_vol_t` (same bar, same span) so the denominator
  cannot collapse to zero in a long uninterrupted uptrend (BTC has had
  several: 2017 H2, 2020-21, most of 2023-24).
- **`risk="cdar"`** — Chekhlov, Uryasev & Zabarankin (2005) CDaR, computed
  causally at daily granularity (a bar-level rolling quantile over a
  60-90 day / 17,000+ bar window is computationally intractable on a
  million-row frame) and broadcast to bars via a one-calendar-day shift,
  floored at a fixed `floor_dd` specifically to guard against pathology
  (b) — checked explicitly in Section 6.

**Falsification test, chosen now:** ETH on Bitfinex, the R-17/R-28/R-31
window (2016 - 2019-12-31), BTC on the same venue/window as the control.
This is the mechanism-generalization question this project has asked every
time a SIZE-axis change was proposed (R-17, R-28's P3, R-31's D3): does the
downside/drawdown-sensitive sizing property transfer to a second asset with
its own volatility/drawdown dynamics, or is it a fit to BTC's particular
2021-22 top-and-bear? Chosen over the fee-tier and funding tests because
the open question here is about the *mechanism*, and ETH is this project's
only committed second-asset series long enough to ask it.

---

## 4. Step 3 — configs evaluated: **44** (inner-train / inner-validation only)

`python experiments/downside_sizing.py sweep` (21 distinct configurations:
1 symmetric control + 12 semidev [3 spans × 2 floors × 2 target_vol] + 8
CDaR [2 windows × 2 alphas × 2 target_dd]), each scored on both inner
splits and both markets (84 backtests), counted once per distinct config.
`python experiments/downside_sizing.py neighbours` (23 more: a one-knob-at-
a-time scan around the inner-validation winner), also on both splits/
markets. **21 + 23 = 44**, the number that goes into the project's
deflated-Sharpe accounting (in addition to the existing 172 established
through R-32).

### Inner-validation, spot (selection split) — selected rows

| config | final | DD | Sharpe |
|---|---|---|---|
| `buy_and_hold` | $574 | 77.3% | 0.08 |
| `kelly_regime_v4` | $998 | 33.2% | 0.14 |
| symmetric control | $909 | 38.1% | 0.02 |
| semidev best (span=4d, floor=0.2/0.4, tv=0.55) | $1,117 | 43.1% | 0.34 |
| **CDaR w=60d, α=0.10, target_dd=0.12 (selected)** | **$1,123** | **24.1%** | **0.35** |
| CDaR w=60d, α=0.20, target_dd=0.18 | $1,247 | 33.5% | 0.47 |

### Inner-validation, futures 5x (same split)

| config | final | DD | Sharpe |
|---|---|---|---|
| `buy_and_hold` | $18 (liq.) | 99.8% | 0.43 |
| `kelly_regime_v4` | $1,064 | 32.3% | 0.25 |
| symmetric control | $948 | 39.3% | 0.08 |
| **CDaR w=60d, α=0.10, target_dd=0.12 (selected)** | **$1,238** | **21.3%** | **0.53** |

**Selection.** CDaR at `window=60d, alpha=0.10, target_dd=0.12` is the only
one of all 21 step-3 configurations that beats `kelly_regime_v4` on
**return, Sharpe and drawdown simultaneously on both markets** on
inner-validation. Selected on that basis, per the routine.

**This selection was treated with suspicion, not celebrated, before any
further evidence was gathered** — because the same config badly
*underperforms* v4 on **inner-train** (spot: $10,357 vs v4's $18,477,
Sharpe 1.58 vs 2.03; futures: $13,634 vs v4's $30,344, Sharpe 1.69 vs
2.28). A config that wins the bear/chop split and loses the bull split is
exactly the split-disagreement pattern R-28/R-31 traced to *lower average
exposure suiting the validation window's regime*, not to a genuine
risk-discrimination advantage — the CDaR config trades far less often on
inner-train than v4 (166 fills vs 261, futures) which is consistent with
that story. The neighbourhood check (Section 6) is what actually decides
which story is right, and it decides against the mechanism.

**Semidev, checked against its own predicted failure mode.** `inspect()`
found `corr(symmetric_vol, downside_semidev) = 0.994` in this data, and
semidev averages **0.71x** symmetric vol at every point in the cycle (2017
mania: ratio 0.73; 2022 bear: ratio 0.71; 2023-24 bull: ratio 0.71) — i.e.
close to `sqrt(0.5)`, the value expected if positive and negative return
magnitudes are simply symmetric around zero in this data. This means the
target_vol=0.55 semidev configs' apparent gains in the sweep table above
are largely **overleveraging from a smaller average denominator**, not
asymmetric risk discrimination: at `target_vol=0.40` (chosen to
approximately cancel the 0.71x scale gap and match the symmetric control's
average leverage), semidev and the symmetric control become nearly
indistinguishable — confirmed on the holdout in Section 7. This is failure
mode (a), named in advance, confirmed exactly as predicted.

---

## 5. Causality self-check

Two-opposite-tampers procedure (the R-28/R-31 convention — an unregistered
experiment gets none of `test_causality_strict.py`'s automatic
protection): every bar strictly after a cut (bar 295,000 of a 300,000-bar
window) is multiplied by 3 in one copy and divided by 3 in another; every
order and every `target`/`risk_measure`/`vote` column value at or before the
cut must be bit-identical across all three runs.

```
risk=symmetric  orders match   max |column difference| before the cut = 0.000e+00   PASS
risk=semidev    orders match   max |column difference| before the cut = 0.000e+00   PASS
risk=cdar       orders match   max |column difference| before the cut = 0.000e+00   PASS

tampered from bar 295,000 of 300,000; PASS - no decision at or before the cut moves
```

**All PASS.** This is the check the CDaR resample-then-shift construction
most needed — the trap named in the module docstring ("Design notes: CDaR
causality") is real in principle for a careless implementation, and the
check confirms this one avoids it: shifting the finished daily CDaR series
by one full calendar day before broadcasting to bars, rather than shifting
raw `close` before resampling, keeps every intraday bar reading only fully
completed prior-day information.

---

## 6. Frozen configuration and pre-registered decision rule

**Written, and the config frozen, before any 2023-01-01+ data was read.**
(`holdout()` was run after this text was fixed; see the module for the
`FROZEN` dict, unedited since before that run.)

**Frozen configuration:**

```python
DownsideKelly(risk="cdar", cdar_window_days=60, cdar_alpha=0.10,
              target_dd=0.12, floor_dd=0.03, deadband=0.10, max_leverage=2.0)
```
(vote gate: horizons=(20,40,80), band=0.01 — identical to `kelly_regime_v4`, unchanged)

**Decision rule, copied from `docs/ROUTINE.md`'s promotion bar, verbatim in
structure:**

> Promote only if **all** of:
> - **P1** it beats `buy_and_hold` out-of-sample after real costs (spot:
>   0.10% taker);
> - **P2** the improvement over `buy_and_hold` exceeds the ±0.2 Sharpe
>   noise floor (R-20), **or** is a drawdown/tail improvement;
> - **P3** *(falsification)* it survives the pre-registered falsification
>   test: on ETH (Bitfinex, the R-17/R-28/R-31 window), with the BTC
>   control run identically, the **ordering** established against
>   `kelly_regime_v4` on the main holdout must replicate — not reverse;
> - **P4** the parameter neighbourhood is a plateau, not a peak (report
>   neighbours, not just the winner).
>
> **Comparison rule against `kelly_regime_v4`** (this file's specific
> purpose, in addition to the `buy_and_hold` bar): `DownsideKelly` must
> also beat `kelly_regime_v4` out-of-sample on spot after real costs,
> either on final balance (return) or on a drawdown/tail improvement that
> does not give up more than the ±0.2 Sharpe noise floor relative to v4.
>
> Default is **REJECT**. Any failing leg is decisive; nothing here is
> weighted or averaged against the others.

**One leg of this rule was already decided before the holdout was read.**
The neighbourhood check in step 3 (Section 4/Section 8) was run entirely on
inner-train/inner-validation, and it already shows `cdar_window_days` and
`cdar_alpha` are **not** a plateau around the selected point — Sharpe on
inner-validation swings from strongly negative (window 30-45d) to strongly
positive (window 60-75d) and back down (window 90d) on both markets. That
is evidence gathered honestly under the pre-registered rule (P4 uses only
inner-split data), and it already fails P4. This is stated here, before the
holdout numbers below, precisely so that a favorable holdout number cannot
be read as overriding it: **the decision rule was fixed first, and P4 was
already lost before Section 7 was run.** The holdout was still run in full,
per `docs/ROUTINE.md` step 4 and this project's convention of reporting a
predicted failure rather than skipping the measurement (R-28 did the same
for its own P1).

---

## 7. Holdout results (2023-01-01 →, run once, as pre-registered)

| market | strategy | final | return | DD | Sharpe |
|---|---|---|---|---|---|
| spot | `buy_and_hold` | $3,839 | +283.9% | 54.0% | 1.03 |
| spot | `kelly_regime_v4` | $3,373 | +237.3% | 27.8% | 1.22 |
| spot | **`downside_kelly` (frozen CDaR)** | **$2,499** | **+149.9%** | **21.1%** | **1.08** |
| futures 5x | `buy_and_hold` | $15,176 | +1417.6% | 60.3% | 1.44 |
| futures 5x | `kelly_regime_v4` | $4,901 | +390.1% | 33.0% | 1.36 |
| futures 5x | **`downside_kelly` (frozen CDaR)** | **$2,519** | **+151.9%** | **27.2%** | **0.96** |

- **P1: FAIL on both markets.** $2,499 < $3,839 (spot); $2,519 < $15,176
  (futures).
- **Comparison vs. `kelly_regime_v4`: FAIL on return, mixed on drawdown.**
  Loses on final balance on both markets ($2,499 vs $3,373 spot; $2,519 vs
  $4,901 futures). Drawdown *is* shallower than v4's — 21.1% vs 27.8% spot
  (6.7pp), 27.2% vs 33.0% futures (5.8pp) — a real, if modest, improvement.
  Futures Sharpe drops by 0.40 relative to v4 (0.96 vs 1.36), outside the
  ±0.2 noise floor in the wrong direction; spot Sharpe is close (1.08 vs
  1.22, a 0.14 gap, inside the floor).
- **Fee tier (0.40% Bitstamp entry), spot only:** `downside_kelly` $1,994
  vs `kelly_regime_v4`'s $2,445 vs `buy_and_hold`'s $3,827. Lower turnover
  than v4 (315 vs 332 fills → $544 fees vs $1,027) softens the degradation
  in relative terms but `downside_kelly` still loses to both on final
  balance. Does not change the verdict.
- **Secondary arm (semidev, ALSO, `target_vol=0.40` — the leverage-matched
  configuration used to isolate the risk-axis effect from failure mode (a)
  above):** spot $3,365 / DD 27.3% / Sharpe 1.21 vs symmetric control's
  $3,277 / DD 26.7% / Sharpe 1.20 — **nearly identical**, exactly the
  predicted "null result in disguise." Futures $5,764 / DD 33.7% / Sharpe
  1.38 vs symmetric control's $4,980 / DD 32.5% / Sharpe 1.33 — a modest
  edge, not decisive, and semidev was not the frozen candidate so this is
  reported for completeness, not scored against the decision rule.

### P3 — falsification test (ETH, Bitfinex, R-17/R-28/R-31 window)

| asset | market | strategy | final | DD | Sharpe |
|---|---|---|---|---|---|
| BTC (control) | spot | `kelly_regime_v4` | $12,278 | 40.1% | 1.86 |
| BTC (control) | spot | `downside_kelly` | $6,043 | 43.0% | 1.25 |
| BTC (control) | futures | `kelly_regime_v4` | $25,681 | 32.1% | 2.19 |
| BTC (control) | futures | `downside_kelly` | $7,131 | 49.3% | 1.27 |
| ETH (test) | spot | `kelly_regime_v4` | $5,482 | 36.5% | 1.48 |
| ETH (test) | spot | `downside_kelly` | $15,093 | 41.3% | 1.65 |
| ETH (test) | futures | `kelly_regime_v4` | $4,263 | 35.1% | 1.25 |
| ETH (test) | futures | `downside_kelly` | $14,226 | 47.1% | 1.60 |

On the BTC control window (2016-2019), `downside_kelly` **loses to v4 on
every axis** — return, drawdown, and Sharpe, on both markets — consistent
with the "peak, not plateau" story: this window is nothing like the
2021-22 bear the config was implicitly selected against. On ETH, the
ordering **flips**: `downside_kelly` beats v4 on return and Sharpe (though
still loses on drawdown) on both markets.

**P3: FAIL.** The pre-registered test asked whether the ordering
established against `kelly_regime_v4` replicates on a second asset. It does
not replicate — it **reverses**, and it reverses in the *same time window*
where the BTC control itself already disagrees with the 2023+ holdout's
ordering. That is about as clean a falsification-test failure as this
project's holdout-avoiding tests produce: not "inside the noise floor,"
but "the sign is not even stable within one fixed historical window across
two different assets."

---

## 8. Plateau / neighbourhood check (23 configurations, inner splits only)

| knob varied | inner-validation spot Sharpe range | shape |
|---|---|---|
| `cdar_window_days` (20/30/45/**60**/75/90) | −0.29 .. +0.38 | **peak, not plateau** — deeply negative at 30-45d, positive only at 60-75d, negative again at 90d |
| `cdar_alpha` (0.05/**0.10**/0.15/0.20/0.30) | −0.08 .. +0.35 | **peak, not plateau** — a dip to strongly negative at 0.15-0.20, either side of the selected 0.10 |
| `target_dd` (0.08/0.10/**0.12**/0.15/0.18/0.22) | +0.24 .. +0.35 | plateau (smooth, monotonic-ish decline) |
| `floor_dd` (0/0.01/**0.03**/0.05) | identical to 4 decimal places at every value | inert at this window length (floor never binds) |
| `deadband` (0.05/**0.10**/0.15/0.20) | +0.31 .. +0.35 | plateau |
| `max_leverage` (1.5/**2.0**/3.0) | identical (leverage cap never binds here) | plateau (trivially) |

Futures inner-validation shows the same shape: `cdar_window_days` swings
−0.50 (30d) to +0.58 (75d) to −0.04 (90d); `cdar_alpha` swings +0.51 (0.05)
down to −0.08 (0.15-0.20) back up to +0.44 (0.30).

**P4: FAIL.** Two of the six knobs — and specifically the two that define
the CDaR statistic's basic shape (how far back it looks, how deep into the
tail it reaches) — are knife-edges, not plateaus. The other four
(`target_dd`, `deadband`, `floor_dd`, `max_leverage`) are genuine plateaus,
which rules out a trivial bug (the strategy is not simply noisy
everywhere) but does not rescue the selection: a promotion candidate needs
*all* its defining knobs to be a plateau, and the two structural ones for
this mechanism are not.

**ATH-reset pathology (b), checked explicitly.** At the frozen 60-day
window, the floor never binds (confirmed twice: `athcheck()` and the
`floor_dd` row above), because a 60-day trailing window on BTC almost
always contains a real double-digit pullback even right after a fresh
all-time high (mean CDaR at new-ATH bars: 0.1845 vs 0.2518 off-ATH — lower,
but not near zero). At shorter windows the pathology is real: at 14 days,
19.8% of new-ATH bars have an unfloored CDaR below 0.03 (minimum observed
0.0055), which is exactly the exposure-spike-at-tops failure named in
advance. The floor does its job at short windows; it is simply irrelevant
to why the 60-day frozen config fails, which is the window-instability
shown above, not this pathology.

**40-window Monte Carlo (R-19 design, identical windows across
strategies), for completeness:**

| market | median return | median DD | beat hold | paired DD vs v4 | paired return vs v4 |
|---|---|---|---|---|---|
| spot | +76.2% (v4: +82.1%) | 20.9% (v4: 23.7%) | 52% (v4: 48%) | deeper in 32% (median −2.4pp) | higher in 32% (median −10.0pp) |
| futures | +77.9% (v4: +116.3%) | 27.0% (v4: 23.6%) | 65% (v4: 65%) | deeper in 65% (median +2.5pp) | higher in 32% (median −15.8pp) |

On spot, `downside_kelly` gives a small, fairly consistent drawdown
improvement (shallower in 68% of paired windows) at a return cost (lower in
68%) — a real if modest risk/return trade rather than a clean win. On
futures it is dominated by v4 outright: deeper drawdown in 65% of windows
*and* lower return in 68%. Neither pattern rescues P1, P3 or P4.

---

## 9. Verdict: **NEGATIVE**

Against the decision rule exactly as pre-registered in Section 6:

- P1 (beat `buy_and_hold` OOS): **FAIL**, both markets.
- P2 (±0.2 Sharpe or drawdown/tail improvement over `buy_and_hold`): moot —
  P1 already fails.
- P3 (falsification, ETH): **FAIL** — the ordering against v4 reverses
  rather than replicates.
- P4 (plateau): **FAIL** — `cdar_window_days` and `cdar_alpha` are peaks.
- Comparison vs. `kelly_regime_v4`: **FAIL** on return on both markets;
  drawdown is modestly better (6.7pp spot, 5.8pp futures) but that alone
  does not clear a rule requiring the *other* legs too.

Every leg with an objective test fails. Default REJECT stands; nothing was
re-argued after the fact, and P4 was already lost before the holdout was
read (Section 6 states this explicitly, in writing, before Section 7's
numbers appear).

**What is true and worth keeping, despite the rejection:** the semi-
deviation arm confirms, cleanly and exactly as predicted, that downside and
symmetric realized volatility are too highly correlated in this specific
BTC series (r = 0.994) for a downside-only cut of *variance* to change the
sizing path in any way that survives controlling for the leverage-scale
artifact it otherwise introduces. That is a real, useful negative — it
narrows what "asymmetric risk measure" can mean here to genuinely different
statistics (like CDaR, which is peak-relative rather than variance-based),
and CDaR is the one that actually behaved differently. It just behaved
differently in a way that turned out to be a fit to one particular
historical episode (the 2021-22 top-and-bear) rather than a stable
property — which the plateau check and the ETH reversal both catch, before
and after the holdout respectively.

---

## 10. One-line lesson

**A risk measure can be genuinely, mechanically asymmetric (CDaR is) and
still be worthless if the one window length that makes it work is a peak
rather than a plateau — asymmetry is necessary for this idea to be
different from R-08, but it is not sufficient for it to be a finding.**

---

## 11. Holdout counter contribution (this session)

- `holdout()`: 6 backtests (3 strategies × 2 markets, all `start=2023-01-01`).
- `costs()` (0.40% fee tier): 6 backtests (3 strategies × 2 tiers × spot
  only, all `start=2023-01-01`).
- Secondary semidev/symmetric-control holdout check (Section 7's "ALSO"
  arm): 4 backtests (2 configs × 2 markets, `start=2023-01-01`).
- `eth()` (falsification): **0** — this reads `btcusd_bitfinex_5m.csv.gz`
  and `ethusd_bitfinex_5m.csv.gz`, both entirely 2016-01 through
  2019-12-31, a different file from the main dataset and entirely before
  the 2023+ split. Correctly does not touch the holdout.
- `windows()` (40-window Monte Carlo): random windows are drawn uniformly
  over the *entire* dataset's index range, not restricted to pre-2023 data.
  **This session checked, rather than assumed, how many of the 40 windows
  (fixed seed=42) actually overlap 2023-01-01 or later: 19 of 40.** Each
  overlapping window is scored for 3 strategies × 2 markets = 6 backtests,
  so **19 × 6 = 114** backtests in this run have at least one bar on or
  after 2023-01-01.

**Total for this session: 6 + 6 + 4 + 0 + 114 = 130.**

**A methodological note for the operator, not a verdict-changing one:** R-19
/ R-28 / R-31 / R-32 each state in the ledger that their 40-window resample
"does not touch the 2023+ BTC holdout." This session's direct check (above)
suggests that claim is not literally true for a window sampler drawn
uniformly over the full 2017-2026 index — roughly half of 40 random
90-730-day windows will contain some bars past 2023-01-01 by simple
arithmetic (2023-2026 is about a third of the dataset's span, and windows
average roughly a year). This session's own `windows()` run follows the
same code shape as those prior rows, so the same caveat likely applies
retroactively to some of their reported "0" contributions — worth a look
if the project ever needs to audit the holdout counter precisely, but not
re-litigated further here since it does not change this row's verdict.

**Project-level total, if folded in:** ledger stood at ~124 after R-32
(08-18). Adding this session's 130 (and whatever the parallel B-05 branch
contributes) is the operator's arithmetic to do per
`docs/ROUTINE.md`'s parallel-round rule (total across branches, not
per-branch) when merging both branches' reports.

**Trials contributed to the project-level deflated-Sharpe count: 44**
(21 sweep + 23 neighbours, Section 4), on top of the 172 established
through R-32.
