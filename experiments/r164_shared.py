"""Shared, read-only utilities and pre-registration for the R-164 round (08-27).

DIRECTION, in one sentence: give `kelly_regime_v4`'s SCALE slot a
risk-managed-momentum overlay -- targeting the realized variance of the
STRATEGY'S OWN payoff (Barroso & Santa-Clara 2015, conservative) or a
compound trend+volatility "panic/calm-bull" state (Daniel & Moskowitz
2016, novel) -- instead of v4's own realized PRICE variance, which is
what its shipped `conditional_target_scale` already targets.

**Which constraint this attacks: SIZE (primary).** No new external data
source is introduced (both branches are pure functions of BTC/ETH's own
OHLCV, reusing v4's own vote and vol estimator where noted), so this does
not touch INFO; it adds no calibrated statistical guarantee or flip-timing
gate (distinct from every ERR-axis construction, most recently R-160's
online-FDR flip gate and R-161's Conformal Risk Control SCALE cap); it
changes no order/throttle applied to an already-decided target (distinct
from every COST-axis construction, R-131/R-133/R-145 -- see the standing
finding those rounds closed: "a COST-axis attack has to change the
DECISION, not the order that follows it"). It changes how much to hold
using only information already in hand -- the one family the standing
diagnosis credits as having worked.

**Literature:**
- Barroso, P. & Santa-Clara, P. (2015), "Momentum Has Its Moments,"
  Journal of Financial Economics 116(1), 111-120. Scales a momentum
  portfolio's exposure by `sqrt(target_variance / realized_variance)`,
  where BOTH variances are of the MOMENTUM STRATEGY'S OWN realized daily
  returns (a 6-month trailing window for the realized leg), not of the
  underlying asset's price. Their central empirical claim: the momentum
  factor's own realized variance has far higher out-of-sample forecast
  power for the factor's OWN future risk (R^2 = 57.8%, their Table 2)
  than plain price-volatility measures do, because it captures the
  option-like payoff asymmetry embedded in a trend/momentum position that
  price variance alone does not. Managed this way, momentum's Sharpe
  ratio in their US equity sample roughly doubles and its worst crashes
  are substantially smaller. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429
- Daniel, K. & Moskowitz, T. J. (2016), "Momentum Crashes," Journal of
  Financial Economics 122(2), 221-247. Documents that momentum's rare,
  severe crashes cluster in "panic" states -- specifically, when the
  market's own TRAILING LONG-HORIZON return has been negative (a bear
  market) AND current volatility is elevated -- and are driven by the
  option-like payoff of past losers snapping back in a rebound. Proposes
  a DYNAMIC strategy that de-risks momentum specifically in that compound
  state (not a blanket, always-on variance target) and shows it improves
  the strategy's ex-ante Sharpe materially over the unconditional
  benchmark. https://www.kentdaniel.net/papers/published/jfe_16.pdf (also
  SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2371227)
- Grobys, K., Kolari, J. W., Sandretto, D., Shahzad, S. J. H. & Äijö, J.
  (2025), "Cryptocurrency momentum has (not) its moments," Financial
  Markets and Portfolio Management. https://link.springer.com/article/10.1007/s11408-025-00474-9
  -- a 2025 application of the Barroso-Santa Clara realized-variance-scaling
  mechanism specifically to crypto momentum, reporting (per the publicly
  indexed abstract; the full text sits behind a paywall this session could
  not reach, so its exact numbers are NOT relied on here, only its
  existence and headline direction) that crypto momentum is subject to
  severe, fat-tailed crashes and that volatility management of this same
  Barroso-Santa Clara form measurably mitigates them. Read as recent,
  independent confirmation that this exact mechanism is a live, contested
  question specifically on crypto data -- CONTEXT ONLY, not adopted as a
  numeric benchmark, matching R-161's own disclosed treatment of its
  Schmitt (2026) citation.

**Not a duplicate of, by ledger ID:**
- A ledger-wide grep (`Barroso|Santa-Clara|Daniel.*Moskowitz|momentum
  crash|risk-managed momentum|momentum has its moments`, case-insensitive)
  returns zero hits anywhere in docs/LEDGER.md outside this entry.
- R-93 / R-152 (Grossman-Zhou / CDaR drawdown-budgeted exposure) -- both
  driven by the STRATEGY'S OWN TRAILING REALIZED DRAWDOWN of its equity
  curve, a path statistic over LOSSES specifically. The conservative
  branch here is driven by the STRATEGY'S OWN REALIZED VARIANCE (a
  symmetric second-moment statistic over ALL returns, gains included);
  the novel branch is driven by a compound PRICE trend + PRICE volatility
  state, referencing the equity curve nowhere.
- R-136 (HAR-RV / DVOL substitution for v4's PRICE-return volatility
  ESTIMATOR feeding the existing `target_vol/realized_vol` slot; found
  the leveraged-futures failure is structural regardless of forecast
  quality). That round improves the FORECAST of the same input v4 already
  uses (price variance). This round's conservative branch replaces WHAT
  is being forecast: the STRATEGY's own realized payoff variance, a
  materially different quantity from price variance whenever `frac` is
  not constant (Barroso & Santa-Clara's own point of departure from plain
  vol-targeting). R-136's own inverse-leverage-effect finding is treated
  below as a NAMED FAILURE RISK for this round too (item 2), not waved
  away by the different input.
- R-59 / R-60 (self-normalizing relative-vol scale, `target_vol/(vol/
  long_run_vol)`, REPLACING v4's absolute-vol scale with a differently
  normalized PRICE-vol ratio; closes B-25). Still a price-vol statistic,
  and a single scalar normalization, not a strategy-payoff variance nor a
  compound trend+vol state.
- R-91 (Goulding-Harvey-Mazzoleni turning-point discount, states built
  from v4's OWN fast/slow ANCHOR VOTE agreement -- Bull/Bear/Correction/
  Rebound). This round's novel branch builds its bear/calm state directly
  from TRAILING PRICE RETURNS over a 24-month window and a volatility
  ratio, never from anchor-vote agreement, and is continuous rather than
  a categorical state lookup.
- R-73 conservative / R-41 conservative / R-46 conservative / R-53
  conservative (the closed "never-increase-only bounded-brake"
  architecture -- ruled out "regardless of the feeding signal," see
  section C). Neither branch here is a one-sided brake layered on top of
  v4's exposure: the conservative branch REPLACES the functional form of
  the scale ratio itself with `sqrt(target/realized)`, which is exactly
  as two-sided as v4's own shipped scale (it can raise OR lower exposure
  depending on whether realized strategy variance sits below or above its
  own long-run target); the novel branch's multiplier is an explicit
  two-sided `1 + kappa*tanh(...)` construction (positive in calm-bull
  states, negative in panic states), the same escape R-163's novel branch
  used against this exact exclusion.
- R-141 novel (LPPLS one-sided dampener, provably degenerate under
  EQUALITY exposure-matching because `damp<=1` always). Both branches
  here are two-sided by construction (see above), so this degeneracy
  argument does not apply, and neither branch is calibrated to match v4's
  mean exposure by construction -- any A3 read is reported, not enforced.
- R-85 (CSD joint AND-gate collapse: two NEAR-UNCORRELATED, FAST indicators
  ANDed together fire almost nowhere). The novel branch's compound score
  multiplies two SLOW, PERSISTENT statistics (a 24-month trailing return
  and a volatility-vs-its-own-1-year-median ratio) as a CONTINUOUS product,
  not a binary AND of two independently-thresholded fast alarms -- the
  failure mode R-85 diagnosed (marginal rates pre-matched, joint rate
  collapsing to near-zero) does not apply to a continuous product of two
  slow-moving series that are, by construction, positively associated in
  genuine bear regimes (Daniel & Moskowitz's own empirical claim).
- R-53/R-54 (VIX/DXY external macro stress brake) and R-73 novel (DVOL
  variance-risk-premium brake) -- both fed by an EXTERNAL series (macro
  index or implied vol). Both branches here are fed exclusively by BTC's
  own realized price/return history, nothing external.
- R-146 (median / jump-exclusion anchor STATISTIC replacing the vote's own
  rolling-mean anchors). Leaves the vote completely alone; this round
  never touches the three anchors or the vote, only the SCALE multiplier
  applied to `frac*scale`.
- R-163 (Turtle-style episode-relative excursion, state reset at every
  vote-flip, keyed on ATR units since the CURRENT bullish episode began).
  Both statistics here are slow, calendar-time rolling/expanding windows
  (63-252 trading days; 365-730 calendar days) that never reset on a vote
  flip and carry memory across episode boundaries -- the opposite state
  -construction choice.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- the r89-r163 convention. Nothing here reads a
bar at or after OOS_START (2023-01-01); `compare()` (imported from
`r161_shared`, chained from r147/r105/r102_shared) never touches the
holdout.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) UNMATCHED-EXPOSURE ARTIFACT -- either branch could mechanically raise
    or lower mean exposure/realized vol relative to v4 (R-33/R-141's
    standing pattern); `risk_matched` is reported for every cell, not
    treated as a pass/fail gate (A3 below), and any headline improvement
    must be read through it.
(2) INVERSE-LEVERAGE-EFFECT ON FUTURES -- R-136's own closing line:
    "the inverse-leverage-effect mechanism binds regardless of forecast
    quality or construction" for ANY volatility-scaling substitution on
    v4's leveraged-futures market. Both branches here are, at bottom,
    volatility-scaling mechanisms; this is this round's single most
    likely failure mode and is named as such before any cell is read.
(3) NON-COLLINEARITY FAILURE (A2 kill switch, both branches) -- v4's own
    scale already targets price variance, and when `frac` sits near a
    constant for extended stretches (v4's dominant BTC/ETH-trending
    regime), the strategy's own realized variance is arithmetically close
    to `frac^2 * price_variance`, which could still correlate highly with
    v4's existing scale path even though the INPUT quantity differs. This
    is a real, disclosed risk, not assumed away by the different
    citation.
(4) BTC-ONLY ARTIFACT -- clears BTC but the sign inverts on ETH. This
    project's single most common SIZE-axis failure mode (R-33/57/64/127/
    137/145/149/150/162/163 among others). Chosen as THE pre-registered
    falsification test for both branches below.
(5) FAT-TAILED / UNSTABLE VARIANCE-RATIO DEGENERACY -- Barroso & Santa-
    Clara's own denominator can be driven arbitrarily close to zero
    whenever the vote-only isolate holds flat (`frac=0`) for an extended
    stretch, since its realized-return series is then identically zero;
    `sqrt(target/realized)` is unbounded as `realized -> 0`. A disclosed,
    non-fitted multiplier clip (BSC_MULT_CLIP) bounds this by construction
    rather than by tuning it away after seeing the effect.
(6) COLD-START / SPARSE-ACTIVATION DEGENERACY (novel branch specifically)
    -- the compound panic/calm-bull score needs a full 24-month trailing
    window before it is even defined (`trailing_cum_return`'s own
    `window_days=730`), and this project's own N~3 diagnosis means only a
    handful of genuine 24-month-bear windows exist pre-holdout at all
    (2018-19, and arguably 2022 alone); reported as an explicit,
    disclosed activity diagnostic, not folded into the headline number.

=====================================================================
PRE-REGISTRATION -- frozen before either branch is dispatched
=====================================================================

Both branches leave `kelly_regime_v4`'s vote (`v4_vote_frac`) and 10%
deadband completely untouched; the only difference from v4 is a
multiplicative factor applied to `v4_raw_desired` (`frac*scale`) BEFORE
the deadband, then clipped to v4's own `+/-V4_MAX_LEVERAGE` envelope.
Configurations swept:

  conservative (Barroso & Santa-Clara 2015): BSC_WINDOW_GRID =
    (0, 63, 126, 252) trading days (0 = identity/A1 check; 126 = the
    literal 6-month PRIMARY the paper itself uses; 63/252 = 3-month/
    12-month bracketing robustness cells, both also used in follow-on
    risk-managed-momentum literature, non-fitted) x 2 markets x 3 slices
    (inner-train, inner-validation, eth_replication) = 24 cells, plus a
    0.40%-fee-tier re-run of PRIMARY (window_days=126) on both markets
    (+2) = 26 configurations.
  novel (Daniel & Moskowitz 2016): NOVEL_KAPPA_GRID = (0.0, 0.5, 1.0, 1.5)
    (0.0 = identity/A1 check; 1.0 = PRIMARY, the same kappa scale R-163's
    own novel branch used) x 2 markets x 3 slices = 24 cells, plus a
    0.40%-fee-tier re-run of PRIMARY (kappa=1.0) on both markets (+2) =
    26 configurations.

**52 total** across both branches -- the number that goes into the
deflated-Sharpe calculation at Step 4, per ROUTINE.md's "trials count is
the total across all parallel branches" rule.

Falsification test (pre-registered, IDENTICAL for both branches, chosen
per the "what would make this fail" list above, item 4 -- the project's
own most common SIZE-axis failure mode and the one every recent round on
this slot has used): ETH sign-replication. On whichever market (spot or
futures_5x) a branch's PRIMARY config clears `clears_bar()` on
inner-validation, the identical construction run on ETH's replication
slice must show the SAME SIGN of `d_log_growth` vs. the v4 control. Sign
inversion on the market that passed -> REJECT for that branch,
independent of every other cell.

Step-0 non-degeneracy kill switches (checked before any Sharpe number is
trusted):
  A1 (identity sanity): window_days=0 (conservative) / kappa=0.0 (novel)
      must reproduce v4_target bit-for-bit (R^2 == 1.0, `d_log_growth` ==
      0.0) -- proves the construction collapses to v4 exactly at its
      neutral setting.
  A2 (non-inertness / non-collinearity): R^2 of the candidate's raw
      pre-deadband exposure path vs v4's own `v4_raw_desired`, at the
      PRIMARY setting, must be < CONST_CAP_R2_THRESH (0.98) -- else this
      mechanism, whatever its cited motivation, never meaningfully moves
      exposure away from v4's own existing scale and is a relabeling, not
      a tested new input (see failure mode 3 above, named as a real risk
      rather than assumed away).
  A3 (exposure match, DISCLOSED not calibrated): `compare()`'s own
      `risk_matched` field (exposure AND realized-vol ratio both in
      [0.9, 1.1]) reported for every cell. Unlike R-163's turtle pyramid
      (which had a strongly predicted one-directional exposure bias),
      neither branch here has a clear a priori direction -- Barroso and
      Santa-Clara's target IS the strategy's own long-run variance, so
      the multiplier should average near 1.0 in-sample by construction,
      though fat-tailed realized variance (Jensen's-inequality bias
      through the sqrt/reciprocal) could still push it off; this is
      reported, not assumed.

Decision rule (exhaustive partition of the outcome space, evaluated
independently for each branch, identical shape to R-160/R-161/R-162/
R-163):
  CLEAR(m) = clears_bar() on inner-validation for at least one non-zero
             window_days/kappa in the grid, on market m.
  GATE_OK  = ETH-falsification passes (no sign inversion on the market
             that cleared, at the PRIMARY config) AND the non-zero grid
             values that clear share the same sign of d_log_growth on
             inner-validation (a plateau, not an isolated spike).

  | GATE_OK | CLEAR(spot) | CLEAR(futures) | Verdict  |
  |---------|-------------|-----------------|----------|
  | false   | --          | --              | REJECT   |
  | true    | false       | false           | REJECT   |
  | true    | true        | false (or v.v.) | PARTIAL  |
  | true    | true        | true            | PROMOTE  |

A branch that reaches PARTIAL or PROMOTE moves to the holdout
(`ev(..., start=OOS_START)`) ONLY after the operator confirms the frozen
window_days/kappa value here -- no further tuning.

Configs evaluated so far by this file: 0 (shared infrastructure only;
each branch's own count is logged in its own module and summed in the
R-164 ledger entry).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import v4_symmetric_vol  # noqa: E402
from experiments.r161_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    V4_BAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    broadcast_daily_lambda,
    causal_truncation_probe_series,
    compare,
    daily_close,
    daily_last_of,
    daily_log_return,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND

V4_DEADBAND = 0.10
V4_MAX_LEVERAGE = 2.0

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
BSC_WINDOW_GRID = (0, 63, 126, 252)   # trading days; 0 = identity (A1), 126 = PRIMARY
BSC_PRIMARY = 126                      # Barroso & Santa-Clara (2015)'s own 6-month window
BSC_MIN_TARGET_DAYS = 365              # min history before the expanding target is defined
BSC_MULT_CLIP = (0.0, 4.0)             # disclosed, non-fitted safety bound (failure mode 5)

NOVEL_KAPPA_GRID = (0.0, 0.5, 1.0, 1.5)   # 0.0 = identity (A1), 1.0 = PRIMARY
NOVEL_PRIMARY = 1.0
DM_TREND_WINDOW_DAYS = 730              # Daniel & Moskowitz (2016): 24-month bear indicator
DM_VOL_MEDIAN_WINDOW_DAYS = 365         # disclosed simplification: 1y trailing median reference
DM_TREND_CLIP = 1.0                     # bound on |24m cum log return| contribution, non-fitted
DM_VOLR_CLIP = 3.0                      # bound on vol-ratio-excess contribution, non-fitted
DM_SCORE_REF = 0.25                     # tanh reference scale, tied to the score's own
                                         # typical (median-nonzero) magnitude on BTC
                                         # pre-holdout history -- a structural calibration
                                         # of the mechanism's SHAPE (checked against the
                                         # exposure path only, never against a Sharpe/
                                         # log-growth number) so PRIMARY is not degenerately
                                         # near-collinear with v4's own scale by construction
NOVEL_MULT_CLIP = (0.0, 2.0)            # same bound R-163's novel branch used

FEE_TIER = 0.0040                       # cost-robustness sensitivity re-run
SHARPE_NOISE_FLOOR = 0.2                # ROUTINE.md's own promotion bar (R-20)
CONST_CAP_R2_THRESH = 0.98              # A2 kill switch


# ================================================================== (1)
# Shared primitive: the daily return earned by v4's VOTE ALONE (frac in
# {0, 1/3, 2/3, 1}, no scale, no deadband) -- R-62's own isolate, reused
# here as the "strategy payoff" whose OWN variance both branches read.
# ==================================================================

def vote_only_daily_return(df: pd.DataFrame) -> pd.Series:
    """Daily log return of holding v4's directional vote alone at full
    notional. Causal: day d's return is weighted by the vote fraction
    already DECIDED at the close of day d-1 (`daily_last_of(...).shift(1)`),
    never by same-day information."""
    frac_by_day = daily_last_of(v4_vote_frac(df).to_numpy(), df.index)
    ret = daily_log_return(df)
    idx = frac_by_day.index.intersection(ret.index)
    r = frac_by_day.reindex(idx).shift(1) * ret.reindex(idx)
    return r.dropna()


# ================================================================== (2)
# CONSERVATIVE construction: Barroso & Santa-Clara (2015), literal --
# scale v4's raw exposure by sqrt(target_variance / realized_variance) of
# the STRATEGY'S OWN payoff, both variances of `vote_only_daily_return`.
# ==================================================================

def bsc_realized_variance(df: pd.DataFrame, window_days: int) -> pd.Series:
    """Trailing rolling variance of the strategy's own realized daily
    returns over `window_days` (Barroso & Santa-Clara's own 6-month
    default at window_days=126). Causal: `.shift(1)` so day d's value uses
    only returns realized through day d-1."""
    r = vote_only_daily_return(df)
    var = r.rolling(window_days, min_periods=max(20, window_days // 3)).var()
    return var.shift(1)


def bsc_target_variance(df: pd.DataFrame) -> pd.Series:
    """Barroso & Santa-Clara's own 'target' variance: the strategy's
    long-run unconditional variance. Computed here as a CAUSAL EXPANDING
    (never full-series-lookahead) statistic, so day d's target uses only
    returns realized through day d-1, exactly like v4's own anchors."""
    r = vote_only_daily_return(df)
    return r.expanding(min_periods=BSC_MIN_TARGET_DAYS).var().shift(1)


def bsc_scale_multiplier(df: pd.DataFrame, window_days: int) -> pd.Series:
    """sqrt(target_var / realized_var), clipped -- the SAME target/realized
    ratio functional FORM v4's own `conditional_target_scale` already
    uses, applied to the strategy's own payoff variance instead of price
    variance (see module docstring: not a duplicate of R-136/R-59/R-60)."""
    realized = bsc_realized_variance(df, window_days)
    target = bsc_target_variance(df)
    idx = realized.index.intersection(target.index)
    realized = realized.reindex(idx).to_numpy()
    target = target.reindex(idx).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mult = np.sqrt(np.where(realized > 0, target / realized, np.nan))
    mult = np.where(np.isfinite(mult), mult, 1.0)
    mult = np.clip(mult, BSC_MULT_CLIP[0], BSC_MULT_CLIP[1])
    return pd.Series(mult, index=idx)


def build_conservative_target(df: pd.DataFrame, window_days: int) -> np.ndarray:
    """v4's own unmodified frac*scale, multiplied by the Barroso-Santa
    Clara strategy-variance-targeting factor, clipped to v4's own
    max_leverage envelope, then v4's own unmodified 10% deadband.
    window_days<=0 is the A1 identity check (bit-for-bit v4)."""
    raw = v4_raw_desired(df)
    if window_days <= 0:
        return apply_deadband(raw, deadband=V4_DEADBAND)
    mult_daily = bsc_scale_multiplier(df, window_days)
    mult_bars = broadcast_daily_lambda(mult_daily, df.index)
    combined = np.clip(raw * mult_bars, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)
    return apply_deadband(combined, deadband=V4_DEADBAND)


# ================================================================== (3)
# NOVEL construction: Daniel & Moskowitz (2016), differently operationalized
# -- a continuous, TWO-SIDED compound trend+volatility "panic/calm-bull"
# multiplier, built purely from BTC/ETH's own price history and v4's own
# (unmodified) realized-vol estimator.
# ==================================================================

def trailing_cum_return(df: pd.DataFrame, window_days: int = DM_TREND_WINDOW_DAYS) -> pd.Series:
    """Daniel & Moskowitz's own bear-market state variable: trailing
    cumulative log return over `window_days` calendar days (24 months by
    default). Causal: `.shift(1)` so day d's value is known before day d
    starts."""
    log_close = np.log(daily_close(df))
    cum = (log_close - log_close.shift(window_days)).shift(1)
    return cum


def trailing_vol_ratio(df: pd.DataFrame,
                        median_window_days: int = DM_VOL_MEDIAN_WINDOW_DAYS) -> pd.Series:
    """Ratio of v4's OWN realized volatility estimator (`v4_symmetric_vol`,
    reused unmodified -- this round does not re-derive a vol estimator,
    unlike R-136) to its own trailing rolling MEDIAN: how elevated current
    vol is relative to its own recent normal range. Causal throughout:
    `vol_daily` itself is shifted a full day BEFORE either the numerator or
    the rolling-median denominator is built, so day d's ratio depends only
    on data realized through day d-1 (matching `broadcast_daily_lambda`'s
    own required convention -- an earlier version of this function shifted
    only the median leg and was caught by `causal_truncation_probe_series`
    peeking at same-day data through the un-shifted numerator)."""
    vol_daily = daily_last_of(v4_symmetric_vol(df), df.index).shift(1)
    median_vol = vol_daily.rolling(
        median_window_days, min_periods=max(30, median_window_days // 3)
    ).median()
    idx = vol_daily.index.intersection(median_vol.index)
    v = vol_daily.reindex(idx).to_numpy()
    m = median_vol.reindex(idx).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(m > 0, v / m, np.nan)
    return pd.Series(ratio, index=idx)


def dm_panic_calm_score(df: pd.DataFrame) -> pd.Series:
    """Continuous, two-sided compound score: positive in Daniel &
    Moskowitz's "calm bull" state (positive trailing 24m return AND
    below-median volatility, their own best-performing momentum regime),
    negative in their "panic" state (negative trailing 24m return AND
    above-median volatility, their own named crash-risk regime), zero
    otherwise. A continuous PRODUCT of two slow, persistent series, not a
    binary AND of two fast/noisy alarms (not the R-85 AND-gate failure
    mode -- see module docstring)."""
    trend = trailing_cum_return(df)
    volr = trailing_vol_ratio(df)
    idx = trend.index.intersection(volr.index)
    trend_v = trend.reindex(idx).to_numpy()
    volr_v = volr.reindex(idx).to_numpy()
    ok = np.isfinite(trend_v) & np.isfinite(volr_v)
    trend_bear = np.clip(-trend_v, 0.0, DM_TREND_CLIP)
    trend_bull = np.clip(trend_v, 0.0, DM_TREND_CLIP)
    vol_high = np.clip(volr_v - 1.0, 0.0, DM_VOLR_CLIP)
    vol_low = np.clip(1.0 - volr_v, 0.0, DM_VOLR_CLIP)
    panic = trend_bear * vol_high
    calm_bull = trend_bull * vol_low
    score = np.where(ok, calm_bull - panic, 0.0)
    return pd.Series(score, index=idx)


def dm_multiplier(df: pd.DataFrame, kappa: float) -> pd.Series:
    """1 + kappa*tanh(score / DM_SCORE_REF), clipped to NOVEL_MULT_CLIP.
    kappa=0.0 is the identity check (multiplier==1.0 everywhere -> v4)."""
    if kappa == 0.0:
        idx = daily_close(df).index
        return pd.Series(1.0, index=idx)
    score = dm_panic_calm_score(df)
    mult = 1.0 + kappa * np.tanh(score.to_numpy() / DM_SCORE_REF)
    mult = np.clip(mult, NOVEL_MULT_CLIP[0], NOVEL_MULT_CLIP[1])
    return pd.Series(mult, index=score.index)


def build_novel_target(df: pd.DataFrame, kappa: float) -> np.ndarray:
    """v4's own unmodified frac*scale, multiplied by the two-sided
    panic/calm-bull factor, clipped to v4's own max_leverage envelope,
    then v4's own unmodified 10% deadband."""
    raw = v4_raw_desired(df)
    if kappa == 0.0:
        return apply_deadband(raw, deadband=V4_DEADBAND)
    mult_daily = dm_multiplier(df, kappa)
    mult_bars = broadcast_daily_lambda(mult_daily, df.index)
    combined = np.clip(raw * mult_bars, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)
    return apply_deadband(combined, deadband=V4_DEADBAND)


# ================================================================== (4)
# Disclosed activity diagnostic (failure mode 6) -- not a kill switch.
# ==================================================================

def dm_activity_diagnostic(df: pd.DataFrame) -> dict:
    """How often the novel branch's compound state is even DEFINED and
    non-trivially non-zero, across the whole pre-holdout series."""
    score = dm_panic_calm_score(df)
    n = len(score)
    nonzero = int(np.sum(np.abs(score.to_numpy()) > 1e-9))
    panic_bars = int(np.sum(score.to_numpy() < -1e-9))
    calm_bull_bars = int(np.sum(score.to_numpy() > 1e-9))
    return {
        "days_scored": n,
        "days_nonzero": nonzero,
        "fraction_nonzero": (nonzero / n) if n else 0.0,
        "days_panic": panic_bars,
        "days_calm_bull": calm_bull_bars,
    }


# ================================================================== (5)
# Decision-rule predicate, applied identically by both branches.
# ==================================================================

def clears_bar(row: dict, sharpe_floor: float = SHARPE_NOISE_FLOOR) -> bool:
    """ROUTINE.md's own promotion-bar predicate, applied to one compare()
    row (candidate vs v4 control on one slice/market): a bootstrap CI on
    log-growth excluding zero on the positive side, OR a Sharpe gain
    beyond the noise floor, OR a real (risk-matched) drawdown improvement."""
    if row["d_log_growth"] > 0 and row["excludes_zero"]:
        return True
    if row["d_sharpe"] >= sharpe_floor:
        return True
    if row["risk_matched"] and row["d_dd"] < 0:
        return True
    return False
