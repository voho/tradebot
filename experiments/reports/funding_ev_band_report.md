# funding_ev_band — R-35 novel branch (08-19)

Unregistered experiment. Code: `experiments/funding_ev_band.py`. Not
`@register`ed, not auto-discovered, nothing committed to git. All
evaluation below is restricted to `<= 2022-12-31` (inner-train /
inner-validation only, hard-asserted in the driver scripts); the
2023-01-01+ holdout was never read, printed, or backtested by this
branch. This is the **novel** branch of R-35 (docs/LEDGER.md); a
separate, independent agent ran the **conservative** branch
(`experiments/funding_gate_decile.py`) on a different file at the same
time — this report does not reference or depend on its result.

## 1. Idea, mechanism, pre-registered prediction

`kelly_regime_ev`'s no-trade band is derived purely from a one-time
**taker fee**: `|Δf| > 2·fee/(H·σ²)`. It never touches perpetual funding
— a *running* cost proportional to notional held over time. R-14 measured
`kelly_regime_v4` paying +20.05%/yr in funding while it holds (vs
+2.78%/yr while flat); R-16 found funding forecasts forward returns. This
file prices the running cost into the Kelly sizer, per the R-35
pre-registration.

**Mechanism, both pieces implemented** (see the module docstring in
`experiments/funding_ev_band.py` for the full derivation):

1. **Haircut the target itself.** Treating v4's `frac*scale` as the `f*`
   of a Merton/Kelly problem, `f* = (mu-c)/sigma^2` becomes
   `target = max(frac*scale - haircut, 0)` where
   `haircut = funding_haircut_scale * max(funding_forecast_annual, 0) / sigma^2`.
   The floor at 0 is safe only because this strategy family never goes
   short (vote fraction in [0,1], scale ≥ 0), so it can only remove
   exposure, never flip its sign.
2. **Widen the no-trade band, asymmetrically, on the side that ADDS
   exposure**, reusing `KellyRegimeEV._band()`'s literal formula with an
   inflated fee term: `fee_up = fee + funding_band_scale *
   max(funding_forecast_annual, 0) * horizon_years`. Cutting exposure
   keeps the unmodified (tighter) band, so a richness signal can slow the
   strategy down entering a position but never trap it in one.

Both terms are gated on `max(funding_forecast_annual, 0)` — a negative
forecast (longs being paid) is clamped to exactly zero, never used to
enlarge the target or shrink the band, matching the hard "can only
reduce, never raise exposure" requirement.

**Prediction, stated before evaluation:** v4 is already flat or
de-levered in exactly the bear/high-funding-mismatch regimes where carry
matters most (the pre-registration's own stated expectation), so this
mechanism should show a real but *small* effect — a modest Sharpe/DD
improvement, not a large one, and it is more likely to survive the
mean-exposure check than the conservative branch's binary gate because it
acts on the sizing target directly rather than adding an override.

## 2. Funding data handling — the causality-critical fix

Per the standing "never proxy unavailable data out of price" rule, bars
outside 2020-01-01..2023-12-31 get no funding value substituted. The
alignment follows the pre-registration's recipe exactly (reindex the 8h
series onto the 5m index, ffill within coverage, then explicitly re-NaN
any bar after `funding.index.max()`), and the EWMA forecast is computed
on that already-NaN-bounded aligned series.

**A bug found and fixed before any evaluation ran, recorded here per this
project's culture of naming the near-miss, not just the final code:**
`pandas.Series.ewm(...).mean()` does **not** emit NaN at rows whose input
is NaN once any real value has been seen — it silently carries the last
computed smoothed value forward through the gap
(`pd.Series([nan,nan,1,2,nan,nan]).ewm(span=3,min_periods=1).mean()`
keeps outputting `1.6667` through the trailing NaNs). Left unguarded,
this would have made the forecast — and therefore the haircut and the
widened band — nonzero for every bar from 2024 onward, exactly the
proxy-out-of-price bug the alignment step exists to prevent. Fixed by
explicitly re-masking the EWMA output to `0.0` wherever the pre-EWMA
aligned value was NaN, rather than trusting the EWMA's own NaN handling.
Verified directly (section 4 below): a live backtest on the pre-2020
prefix of inner-train, with real funding loaded and default nonzero
scales, is bit-identical to `kelly_regime_ev` (`equity max|diff| = 0.0`).

## 3. Class shape

`FundingEVBand(KellyRegimeEV)`. `prepare()` reproduces
`kelly_regime_v3`/`v4`'s vote + conditional vol-targeting math
byte-for-byte in one causal forward pass (the same pattern
`experiments/kelly_regime_v5_damp.py` used), inserting the haircut before
v3/v4's own 0.10 position-deadband latch. `on_bar()` reuses
`KellyRegimeEV._band()` literally, applied asymmetrically. `warmup =
80*BARS_PER_DAY+10`, identical to v4/EV.

New knobs: `funding_forecast_span_days` (EWMA smoothing, in days),
`funding_haircut_scale` (mechanism 1 strength), `funding_band_scale`
(mechanism 2 strength), `widen_band` (bool, on/off for mechanism 2),
`funding` (inject a series for testing; defaults to
`tradebot.data.load_funding`).

## 4. Correctness check: forecast=0 reduces exactly to `kelly_regime_ev`

Three independent checks, all `equity max|diff| = 0.0`, bit-identical,
fills identical:

| scenario | equity max\|diff\| | fills (band, ev) |
|---|---|---|
| no funding data at all (`funding=pd.Series(dtype=float)`), default nonzero scales | 0.0 | 100, 100 |
| real funding present, `funding_haircut_scale=0` **and** `funding_band_scale=0` | 0.0 | 100, 100 |
| real funding present, default nonzero scales, backtest restricted to the pre-2020 (funding-uncovered) slice of inner-train | 0.0 | matched |

(An earlier, sloppier version of this check set only
`funding_haircut_scale=0` while leaving `funding_band_scale` at its
default 0.5 — that leaves mechanism 2 active whenever real funding data
exists, since it is gated on the *forecast*, not on the haircut knob, so
the equity curves correctly differed there. Recorded so a reader doesn't
mistake it for a bug in the strategy: both new knobs must be zero, or the
forecast itself must be zero, and the table above tests the actual
zero-forecast conditions the pre-registration specifies.)

`target` and `_ev_vol` columns: `max|diff| = 0.000e+00` against
`kelly_regime_ev` over inner-validation-plus-buffer, matching the design.

## 5. Causality probe

Two-opposite-tampers procedure plus a real (untampered) control, 200,000
BTC-spot bars ending at 2022-12-31 (holdout never touched), cut 5,000 bars
before the end, bars after the cut multiplied by 3 / divided by 3:

| check | max\|diff\| before cut (real/up/down, all pairs) | result |
|---|---|---|
| `on_bar` order decisions, 8 probed bars | — | PASS |
| `target` column | 0.000e+00 | PASS |
| `_ev_vol` column | 0.000e+00 | PASS |
| `_funding_annual` column | 0.000e+00 | PASS |
| `_haircut` column | 0.000e+00 | PASS |
| `_pre_haircut_target` column | 0.000e+00 | PASS |

No lookahead detected. (`_funding_annual`/`_haircut` are trivially
price-independent — they come from the external funding file, not
OHLCV — but were checked anyway since a coding error could still make
them price-dependent by accident; it did not.)

## 6. Sweep — 9 primary configs, futures, inner-train + inner-validation

`funding_haircut_scale ∈ {0.25, 0.5, 1.0}` × `funding_forecast_span_days
∈ {1, 3, 10}` days, `widen_band=True`.

Baselines (futures):

| strategy | period | final | Sharpe | maxDD% | fills |
|---|---|---|---|---|---|
| kelly_regime_v4 | TRAIN | $30,344 | 2.28 | 35.3% | 261 |
| kelly_regime_ev | TRAIN | $23,693 | 2.17 | 35.0% | 182 |
| kelly_regime_v4 | VALID | $1,064 | 0.25 | 32.3% | 143 |
| kelly_regime_ev | VALID | $1,072 | 0.27 | 36.3% | 100 |

Sweep, inner-validation (futures), sorted by haircut:

| hc | span(d) | Sharpe | maxDD% | final | ΔSharpe vs v4 | ΔSharpe vs EV |
|---|---|---|---|---|---|---|
| 0.25 | 1  | 0.43 | 25.4% | $1,146 | +0.18 | +0.17 |
| 0.25 | 3  | 0.38 | 26.8% | $1,121 | +0.13 | +0.12 |
| 0.25 | 10 | 0.37 | 24.4% | $1,121 | +0.12 | +0.11 |
| 0.5  | 1  | 0.27 | 29.8% | $1,070 | +0.02 | +0.01 |
| 0.5  | 3  | 0.34 | 26.2% | $1,093 | +0.09 | +0.08 |
| **0.5** | **10** | **0.48** | **26.0%** | **$1,148** | **+0.23** | **+0.22** |
| 1.0  | 1  | -0.22 | 29.2% | $911 | -0.47 | -0.48 |
| 1.0  | 3  | 0.27 | 25.3% | $1,058 | +0.01 | +0.00 |
| 1.0  | 10 | 0.28 | 23.1% | $1,060 | +0.03 | +0.01 |

`hc=0.5, span=10d` is the only cell that individually clears the ±0.2
Sharpe noise floor on inner-validation futures against **both** baselines
(+0.23 / +0.22).

**Plateau assessment — the honest, load-bearing part.** The winning cell
is *not* surrounded by a flat neighborhood: its immediate span-neighbors
at the same haircut (`hc=0.5, span=3` → 0.34; `hc=0.5, span=1` → 0.27)
sit 0.11-0.21 Sharpe below it, and its haircut-neighbors at the same span
(`hc=0.25, span=10` → 0.37; `hc=1.0, span=10` → 0.28) sit 0.11-0.20 below
it too — gaps that are themselves comparable in size to the whole ±0.2
noise floor this project uses to call a difference real. `hc=1.0,
span=1` is an outright failure (-0.22, -0.47 vs v4), confirming the
pre-registration's own named failure mode (b): a short forecast span
combined with an aggressive haircut re-trades on 8h funding noise faster
than the deadband can absorb. There IS a genuine, interpretable structure
— span≥3 is uniformly safer than span=1 at every haircut scale, matching
the "funding is autocorrelated, needs smoothing" prior — but the single
best cell is better characterized as a **soft peak inside a safe region**
than as a flat plateau. `hc=0.25` (span 1/3/10 → 0.43/0.38/0.37) is the
flatter, more plateau-like row, but its best member (+0.18 vs v4) falls
just short of the ±0.2 floor.

**28 configurations evaluated in the sweep + diagnostics below** (see
section 10 for the full count and methodology); every one is reported,
including the two that lost to both baselines (`hc=1.0,span=1`) or
matched inside the noise floor.

## 7. Mean-exposure check (mandatory) — inner-validation, futures

| | mean \|target\| |
|---|---|
| kelly_regime_v4 | 0.2894 |
| funding_ev_band (hc=0.5, span=10d) | 0.1861 |
| funding_ev_band, pre-haircut shadow (= v4's own latch, verified identical) | 0.2894 |

Ratio: **0.643** — the best config holds only 64% of v4's mean exposure
on inner-validation futures. This is meaningfully below v4's, stated
plainly per the standing rule (three prior rounds — L-04/R-33, R-28/R-31,
R-32 — have hit "drawdown improvement is really just holding less").

**Rescale diagnostic (R-34's method, run as required):** rescaled the
best config's `target` by 1.555× to match v4's mean exposure exactly:

| period | final | Sharpe | maxDD% | fills |
|---|---|---|---|---|
| TRAIN | $63,514 | 2.16 | 57.5% | 304 |
| VALID | $1,365 | **0.67** | 33.8% | 132 |

Compared to the un-rescaled best (VALID: Sharpe 0.48, DD 26.0%) and to
v4 (VALID: Sharpe 0.25, DD 32.3%):

- **Sharpe survives rescaling and grows** (0.67 vs v4's 0.25 — a *larger*
  gap than the un-rescaled config's, +0.42 vs +0.23). This is the
  opposite of the exposure-artifact failure mode: at matched mean
  exposure, the strategy still clearly outperforms v4 on a risk-adjusted
  basis, meaning the timing information in the funding forecast is doing
  real work, not just de-levering.
- **Drawdown does not survive rescaling equally well** (33.8%, essentially
  on par with — slightly worse than — v4's 32.3%, versus the un-rescaled
  config's 26.0%). So the drawdown-cut component of the story specifically
  *is* substantially an exposure-level effect, even though the
  Sharpe-improvement component is not.

**Verdict on this check:** mixed, and reported as such rather than
rounded to a clean yes/no. The Sharpe edge is not an exposure artifact;
the drawdown edge largely is.

## 8. Ablation — band-widening's incremental contribution

`widen_band=False` (haircut only) vs `widen_band=True` (full mechanism),
same `hc=0.5, span=10d`, inner-validation futures:

| | Sharpe | maxDD% | final | funding-charged Sharpe | funding paid |
|---|---|---|---|---|---|
| widen_band=False (haircut only) | 0.51 | 25.8% | $1,169 | 0.30 | $99 |
| widen_band=True (full mechanism) | 0.48 | 25.8→26.0% | $1,148 | **0.38** | **$58** |

Funding-free, the band-widening term is a wash or slightly negative
(0.51 vs 0.48) — the haircut on the target already does most of the work
of avoiding richness. Under the funding-charged falsification (section 9)
the band-widening term earns its keep exactly as designed: it more than
halves the actual funding bill ($58 vs $99) and preserves more of the
Sharpe once that cost is real (0.38 vs 0.30). This is a small but genuine
and interpretable effect in the direction the mechanism was built for.

## 9. Pre-registered falsification test (mandatory) — real funding charged, inner-validation futures

| strategy | funding regime | final | maxDD% | Sharpe | fills | funding paid |
|---|---|---|---|---|---|---|
| kelly_regime_v4 | funding-FREE | $1,064 | 32.3% | 0.25 | 143 | $0 |
| kelly_regime_v4 | funding-CHARGED | $887 | 34.7% | **-0.06** | 143 | $184 |
| kelly_regime_ev | funding-FREE | $1,072 | 36.3% | 0.27 | 100 | $0 |
| kelly_regime_ev | funding-CHARGED | $892 | 38.4% | **-0.06** | 100 | $190 |
| **funding_ev_band BEST (hc=0.5,sp=10d)** | funding-FREE | $1,148 | 26.0% | 0.48 | 48 | $0 |
| **funding_ev_band BEST (hc=0.5,sp=10d)** | funding-CHARGED | $1,105 | 26.2% | **0.38** | 48 | **$58** |

Also checked for the plateau candidate and the ablation, for completeness:

| strategy | funding-CHARGED Sharpe | funding paid |
|---|---|---|
| PLATEAU (hc=0.25, sp=3d) | 0.34 | $114 |
| no-widen-band (hc=0.5, sp=10d, mechanism 1 only) | 0.30 | $99 |

**This is a clean, unambiguous PASS.** Both baselines flip from
marginally-positive to negative Sharpe once real funding is charged
(0.25→-0.06, 0.27→-0.06) and both end the two-year window below their
$1,000 start balance. `funding_ev_band`'s advantage does not merely
survive — it widens: it pays roughly a third of the funding baselines
pay ($58 vs $184-190) while staying solidly positive (0.38). This is the
single most important check for a COST-focused strategy and it is the
strongest single result in this report.

## 10. Configurations evaluated (counting methodology, stated per this project's convention)

Following the `ev()`/`count=` convention used in
`experiments/run_eprocess.py` and `run_matched_hold.py`: every distinct
backtest of a `funding_ev_band` configuration (any combination of
haircut/span/widen_band, period, market, or funding-cost regime) is
counted as one trial; baseline (`kelly_regime_v4`, `kelly_regime_ev`)
runs are reported for comparison but not counted, matching the repo's
existing driver scripts.

- Sweep (section 6): 9 configs × {TRAIN, VALID} × futures = **18**
- Rescale diagnostic (section 7): 1 config × {TRAIN, VALID} = **2**
- Band-widening ablation (section 8): 1 config × {TRAIN, VALID} = **2**
- Plateau candidate on both markets (section 6/9): 1 config × {TRAIN,
  VALID} × {futures, spot} = **4**
- Best config on spot, secondary/diagnostic cell: 1 config × {TRAIN,
  VALID} = **2**
- Funding-charged falsification, new cost-regime evaluations not already
  counted above (section 9): best config charged (1), plateau charged
  (1), no-widen-band charged (1) = **3**

**Total: 31 configurations of `funding_ev_band` evaluated in this
branch.** (Baseline `kelly_regime_v4`/`kelly_regime_ev` were additionally
run 12 times across periods/markets/cost-regimes for comparison, not
counted as trials, per the repo's standing convention.)

## 11. Secondary/diagnostic cell — spot

Funding is not paid on spot, so this isolates R-16's pure
return-forecast channel from the cost-avoidance channel:

| strategy | period | final | Sharpe | maxDD% |
|---|---|---|---|---|
| kelly_regime_v4 | TRAIN | $18,477 | 2.03 | 43.3% |
| kelly_regime_ev | TRAIN | $14,472 | 1.92 | 44.4% |
| funding_ev_band BEST | TRAIN | $15,953 | 2.09 | 44.4% |
| kelly_regime_v4 | VALID | $998 | 0.14 | 33.2% |
| kelly_regime_ev | VALID | $930 | 0.01 | 36.5% |
| funding_ev_band BEST | VALID | $1,059 | 0.25 | 29.3% |

Spot inner-validation Sharpe (+0.25 vs v4's +0.14, ΔSharpe +0.11, inside
the noise floor) is directionally positive but modest — consistent with
the pre-registration's own framing: since funding is not paid on spot,
any spot effect isolates the return-forecast channel, which R-16 found
real but smaller than the cost-avoidance channel this branch is primarily
built to attack.

## 12. Recommendation against the pre-registered decision rule

Ledger criteria, in order:

**(i) Beats v4 by >±0.2 Sharpe on futures inner-validation, or matches
within the floor while cutting drawdown, AND the mean-exposure check does
not reduce the finding to a flat rescale.** The best config clears the
raw number (+0.23 Sharpe). The mean-exposure check is mixed rather than
clean: the Sharpe edge survives being rescaled to v4's mean exposure (in
fact widens), so it is not a flat rescale — but the drawdown edge is
substantially explained by lower average exposure and mostly disappears
under the same rescale. Read generously: **passes**, on the Sharpe route
specifically, with the drawdown claim explicitly weakened.

**(ii) Survives the funding-charged falsification test.** **Clean pass**
— the strongest result in this report (section 9).

**(iii) The parameter neighbourhood is a plateau, not a peak.** **Does
not cleanly pass.** The winning cell is inside a broadly safe region
(span≥3 is uniformly better than span=1, an interpretable and
pre-registered-consistent pattern) but is itself elevated 0.11-0.21
Sharpe above its immediate neighbors — a gap on the same order as the
noise floor being used to judge it — in a validation window where both
baselines' own Sharpe sits near zero (0.25, 0.27), i.e. a low-power
setting where a single grid cell landing 0.2+ above its neighbors is a
plausible noise outcome, not only a plausible signal.

**Overall: borderline, leaning short of the bar as pre-registered.**
Criteria (i) and (ii) are satisfied or generously satisfiable; (iii) is
the genuine sticking point, and this project's culture treats "peak, not
plateau" as a promotion-blocker rather than a detail to average away. My
recommendation is that this branch alone should **not** trigger the
2023-01-01..2023-12-31 holdout read under the pre-registration's letter —
`hc=0.25` (the flatter row) falls just short of the Sharpe floor, and
`hc=0.5,span=10d` (the row that clears it) does not sit in a flat enough
neighborhood to trust as a plateau finding on two years of data that
substantially overlaps R-16's own discovery sample (the pre-registration's
own named "no funding-covered period this idea has not already been
looked at once" caveat). The funding-charged falsification result
(section 9) is genuinely strong and worth recording regardless of the
promotion verdict — it is direct, causally clean evidence that pricing
funding into the sizer measurably reduces the real funding bill without
giving back the return, which is exactly the COST-constraint claim R-35
set out to test — but it is evidence for the *mechanism*, not by itself
sufficient to clear the plateau bar the pre-registration set.

This is a report for the operator to weigh jointly against the
conservative branch's independent result, per ROUTINE.md's parallelism
rules; this session did not read that branch's file or report.

## Falsifiable prediction, checked

Stated in section 1 before evaluation: a real but small effect. Measured:
real (survives funding-charged falsification, section 9; genuine,
non-exposure-artifact Sharpe edge, section 7) but not unambiguously large
enough to clear the plateau criterion on two years of validation data —
consistent with the prediction that this mechanism has "a real chance"
but limited room to do additional work, rather than the large, clean win
a from-scratch idea might have shown.
