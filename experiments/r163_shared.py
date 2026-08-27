"""Shared, read-only utilities and pre-registration for the R-163 round (08-27).

DIRECTION, in one sentence: give kelly_regime_v4 a path-dependent, trade-
level SIZE mechanism -- Turtle-style pyramiding, adding to (or, in the
novel branch, also trimming) exposure as a function of how far price has
moved in favor of (or against) the CURRENT bullish episode since it
started, rather than any global rolling market statistic -- at two
different constructions: a literal discrete unit-stack (conservative) and
a continuous, two-sided excursion multiplier (novel).

**Which constraint this attacks: SIZE.** No new external data source is
introduced (ATR/True Range and the episode boundary are both pure
functions of the same OHLC columns v4 already reads), so this does not
touch INFO; it adds no formal statistical guarantee or flip-timing gate
(distinct from every ERR-axis construction in this ledger, most recently
R-160's online-FDR flip gate and R-161's Conformal Risk Control SCALE
cap); it adds no rebalance-rate constraint, deferral or shadow price
(distinct from every COST-axis construction, R-131/R-133/R-145). It
changes how much to hold using only information v4 already has -- the one
family the standing diagnosis credits as having worked -- but the STATE
VARIABLE driving it (favorable/adverse excursion in ATR units SINCE THE
CURRENT EPISODE STARTED, i.e. trade-level path dependence) is new: a
ledger-wide grep (`pyramid|tranche|unit.?add|scale.?in|MFE|MAE\\b|ATR`)
returns zero hits for any trade-level, episode-relative construction
anywhere in 24,036 lines. Every SIZE-axis attempt to date (28+, R-34
through R-162) is driven by a GLOBAL rolling statistic (a market-wide
trend/vol/efficiency measure) applied uniformly regardless of how the
CURRENT position itself has performed since it opened.

**Literature:**
- Faith, Curtis M., *Way of the Turtle* (McGraw-Hill, 2007) -- the
  published Turtle Trading rules: add one unit every 0.5N (N = 20-day
  ATR) the market moves in favor of an open position, up to 4 units;
  each new unit ratchets the whole stack's stop to 2N below (long) /
  above (short) that unit's own entry price. A practitioner rulebook, not
  a peer-reviewed backtest -- disclosed, not proxied.
  https://www.theturtletrader.com/turtle-trading-rules/
- Zarattini, Carlo (Concretum Group), "Position Sizing in Trend-Following:
  Comparing Volatility Targeting, Volatility Parity, and Pyramiding"
  (2026) -- a 2026 practitioner replication on 40 liquid futures markets
  since inception, weekly rebalancing, no transaction costs disclosed:
  Volatility-Parity-with-Pyramiding roughly triples realized volatility
  (10.8% -> 25.4%) vs. plain volatility targeting and "introduces
  convexity ... exposure increases when trends persist ... risk is given
  back if the trade reverses." The source's own headline explicitly
  trades higher realized vol for a fatter right tail -- named below as
  this round's own predicted failure mode against an UNMATCHED-exposure
  comparison, not proxied over.
  https://concretumgroup.com/position-sizing-in-trend-following-comparing-volatility-targeting-volatility-parity-and-pyramiding/

**Not a duplicate of, by ledger ID:**
- R-90 / R-46 (trailing stop / CPPI drawdown-floor scaling) -- both
  REDUCE exposure on adverse excursion from a peak. This construction's
  conservative branch only ADDS on favorable excursion (the opposite
  trigger direction); its novel branch is two-sided but keyed on
  excursion since EPISODE START, not a floor/peak-drawdown distance.
- R-93 / R-152 (Grossman-Zhou / CDaR drawdown-budgeted exposure) --
  driven by trailing REALIZED DRAWDOWN of the strategy's own equity
  curve, a portfolio-level statistic. This is driven by PRICE excursion
  since the current episode's own entry, a trade-level statistic that
  does not reference the equity curve at all.
- R-162 (Kaufman Efficiency Ratio, two-sided SCALE/VOTE conviction
  modifier) -- driven by a GLOBAL rolling trend-straightness ratio
  computed the same way at every bar regardless of when the current
  episode started. This is driven by a PATH-DEPENDENT quantity (bars
  since episode start, price since episode entry) that resets to zero at
  every episode boundary and is undefined by a rolling window of fixed
  length.
- R-89 (response-curve shape on a 365-day rolling-RMS trend-magnitude
  z-score, reshaping the VOTE's output curve) -- same objection: R-89's
  statistic is a fixed-window rolling transform, not an episode-relative
  one, and never resets.
- R-146 (anchor statistic: median / jump-exclusion) -- changes how each
  anchor's OWN trend value is computed; leaves the vote-to-exposure
  mapping alone. This leaves all three anchors and v4's own vote/scale
  entirely untouched and only adds a stack on top.
- R-59/R-60 (target_vol/max_leverage magnitude retunes) -- a single
  constant, uniform across all bars. This is a discrete/continuous
  function of within-episode path history, never a flat retune.
- Ledger-wide grep (`pyramid|Turtle|Faith 2007|Zarattini|scale.?in|unit
  add|ATR\\b`, case-insensitive) returns zero hits anywhere in
  docs/LEDGER.md outside this entry, and the construction names nothing
  on the 08-26/27 verification-pass archive's own closed-candidate lists
  (wavelets, GP regression, particle filters, DRO/Wasserstein-Kelly,
  EVT/Hill-POT-GPD, meta-labeling, path-signatures, execution bandits,
  BOCPD/HSMM, RL sizing, TDA, stablecoin-copula, liquidation-cascade CSD,
  Fear&Greed, ZigZag/Elliott, HRP, quantile regression, CPPI variants).

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- the r89-r162 convention. Nothing here reads a
bar at or after OOS_START (2023-01-01); `compare()` (imported from
`r161_shared`, chained from r147/r105/r102_shared) never touches the
holdout.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) UNMATCHED-EXPOSURE ARTIFACT -- this is this round's PREDICTED failure
    mode, named directly from the cited source's own headline: adding
    size on favorable excursion mechanically raises realized volatility
    and exposure (Zarattini 2026's own 10.8%->25.4%), which is exactly
    the R-33/R-141 exposure-artifact pattern this project's `clears_bar()`
    predicate is built to catch (a `risk_matched` pass is NOT expected;
    promotion, if it happens at all, must clear on Sharpe or CI-excluding-
    zero log-growth despite the extra volatility, not on drawdown).
(2) WRONG-TIMING ARTIFACT -- pyramiding, by construction, adds size at the
    MATURE stage of an already-confirmed trend, which is exactly where
    this project's own six named regime-transition episodes (2018 bear
    onset, COVID crash, 2021-11 top, Terra/Luna, FTX) sit. Piling on
    right before a reversal is this mechanism's own textbook failure mode
    and this project's own repeated empirical one.
(3) BTC-ONLY ARTIFACT -- clears BTC but the sign inverts on ETH. This
    project's single most common failure mode for SIZE-axis constructions
    on this slot (R-33/57/64/127/137/145/149/150 among others). Chosen as
    THE pre-registered falsification test for both branches below,
    exactly as R-141/R-147/R-162 used it.
(4) COLD-START / SPARSE-EPISODE DEGENERACY -- a bullish episode must
    persist long enough, and move far enough, to reach even one 0.5N
    threshold before either branch differs from v4 at all; if v4's own
    10% deadband and latching hysteresis already suppress most short
    episodes, num_units may sit at 0 (conservative) or the multiplier at
    ~1 (novel) for most of the series -- reported as an explicit
    diagnostic (`pyramid_activity_diagnostic` below), not folded into the
    headline number.
(5) NON-COLLINEARITY FAILURE (both branches) -- if the candidate's raw
    pre-deadband exposure path is a near-exact rescale of v4's own (A2
    kill switch below, threshold 0.98), this is a relabeling of v4's
    existing SCALE, not a tested new mechanism, and both branches should
    be expected to differ substantially given the state variable involved
    resets at episode boundaries v4's own scale does not track.

=====================================================================
PRE-REGISTRATION -- frozen before either branch is dispatched
=====================================================================

Both branches use the SAME episode definition (a maximal run of bars with
v4's own vote_frac >= 0.5, i.e. "the crowd-regime vote leans bullish"; the
strategy is long-only/frac in [0,1] by construction -- see
`tradebot/strategies/kelly_regime.py` -- so there is no symmetric short
episode to pyramid) and the SAME 20-day rolling-mean True Range as N (a
simple rolling mean, not Wilder's exact recursive RMA smoothing -- a
disclosed simplification, consistent with this project's general
preference for plain causal rolling constructs over parametrically-tuned
smoothers). The only difference between the branches is HOW the
episode-relative excursion is turned into extra exposure: a literal
discrete 4-unit stack with its own stop-ratchet (conservative) vs. a
continuous, symmetric tanh multiplier applied to v4's own raw_desired
exposure (novel, same locus as R-162's conservative SCALE-multiplier
construction, but keyed on a path-dependent quantity instead of a global
rolling statistic). Configurations swept:

  conservative: num_units_cap in {0, 1, 2, 4} (0 = identity/A1 check, 4 =
    the literal Faith (2007) standard) x 2 markets x 3 slices
    (inner-train, inner-validation, eth_replication) = 24 cells, plus a
    0.40%-fee-tier re-run of the PRIMARY config (num_units_cap=4) on both
    markets (+2) = 26 configurations.
  novel: kappa in {0.0, 0.5, 1.0, 1.5} (0.0 = identity/A1 check, 1.0 =
    PRIMARY) x 2 markets x 3 slices = 24 cells, plus a 0.40%-fee-tier
    re-run of PRIMARY (kappa=1.0) on both markets (+2) = 26
    configurations.

**52 total** across both branches -- the number that goes into the
deflated-Sharpe calculation at Step 4, per ROUTINE.md's "trials count is
the total across all parallel branches" rule.

Falsification test (pre-registered, IDENTICAL for both branches, chosen
per the "what would make this fail" list above, item 3 -- the project's
own most common SIZE-axis failure mode and the one every recent round on
this slot has used): ETH sign-replication. On whichever market (spot or
futures_5x) a branch's PRIMARY config clears `clears_bar()` on
inner-validation, the identical construction run on ETH's replication
slice must show the SAME SIGN of `d_log_growth` vs. the v4 control. Sign
inversion on the market that passed -> REJECT for that branch,
independent of every other cell.

Step-0 non-degeneracy kill switches (checked before any Sharpe number is
trusted):
  A1 (identity sanity): num_units_cap=0 (conservative) / kappa=0.0
      (novel) must reproduce v4_target bit-for-bit (R^2 == 1.0,
      `d_log_growth` == 0.0) -- proves the construction collapses to v4
      exactly at its neutral setting.
  A2 (non-inertness / non-collinearity): R^2 of the candidate's raw
      pre-deadband exposure path vs v4's own `v4_raw_desired`, at the
      PRIMARY setting, must be < CONST_CAP_R2_THRESH (0.98) -- else the
      pyramiding factor never meaningfully moves exposure and this is a
      relabeling of v4, not a tested mechanism.
  A3 (exposure match, DISCLOSED not calibrated): `compare()`'s own
      `risk_matched` field (exposure AND realized-vol ratio both in
      [0.9, 1.1]) reported for every cell. This mechanism is explicitly
      NOT expected to pass A3 -- see failure mode (1) above -- so a
      `risk_matched=False` reading is not grounds to refit; it is the
      predicted, disclosed shape of the result, and any headline
      improvement must be interpreted through it per the standing R-33
      rule ("holding [more] draws down [more]; that is arithmetic, not
      evidence").

Decision rule (exhaustive partition of the outcome space, evaluated
independently for each branch, identical shape to R-160/R-161/R-162):
  CLEAR(m) = clears_bar() on inner-validation for at least one non-zero
             num_units_cap/kappa in the grid, on market m.
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
num_units_cap/kappa value here -- no further tuning.

Configs evaluated so far by this file: 0 (shared infrastructure only;
each branch's own count is logged in its own module and summed in the
R-163 ledger entry).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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
    causal_truncation_probe_series,
    compare,
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
ATR_WINDOW_DAYS = 20                # Faith (2007)'s own "N" window, non-fitted
ADD_THRESHOLD_N = 0.5                # Faith (2007): add a unit every 0.5N
STOP_THRESHOLD_N = 2.0               # Faith (2007): stop 2N from the newest unit's entry
N_MAX_UNITS = 4                      # Faith (2007): standard 4-unit cap
UNIT_SIZE = V4_MAX_LEVERAGE / N_MAX_UNITS   # 0.5 -- 4 units span v4's own max_leverage
CONSERVATIVE_GRID = (0, 1, 2, 4)     # num_units_cap sweep; 0 = identity (A1)
CONSERVATIVE_PRIMARY = 4              # literal Faith (2007) standard
NOVEL_GRID = (0.0, 0.5, 1.0, 1.5)    # kappa sweep; 0.0 = identity (A1)
NOVEL_PRIMARY = 1.0
NOVEL_REF_ATR_UNITS = 2.0            # ties tanh saturation to the SAME 2N stop distance
NOVEL_MULT_CLIP = (0.0, 2.0)         # symmetric, disclosed, non-fitted safety bound
FEE_TIER = 0.0040                    # cost-robustness sensitivity re-run
SHARPE_NOISE_FLOOR = 0.2             # ROUTINE.md's own promotion bar (R-20)
CONST_CAP_R2_THRESH = 0.98           # A2 kill switch


# ================================================================== (1)
# Shared episode/ATR primitives -- both branches read only these.
# ==================================================================

def true_range(df: pd.DataFrame) -> np.ndarray:
    """Causal True Range at native 5-minute bar granularity."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def atr_n(df: pd.DataFrame, window_days: int = ATR_WINDOW_DAYS) -> np.ndarray:
    """Simple rolling-mean True Range over `window_days` -- Faith (2007)'s
    'N', a disclosed simplification of Wilder's exact RMA smoothing."""
    tr = true_range(df)
    window_bars = window_days * BARS_PER_DAY
    return pd.Series(tr).rolling(window_bars, min_periods=BARS_PER_DAY).mean().to_numpy()


def bullish_episode_state(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """v4's own vote_frac >= 0.5 defines a "bullish episode"; returns
    (bullish bool array, entry_price array) where entry_price holds the
    close at the bar the CURRENT episode began (constant within an
    episode, causal: only ever set from bars <= i)."""
    frac = v4_vote_frac(df).to_numpy()
    bullish = frac >= 0.5
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    entry_price = np.empty(n)
    cur_entry = close[0]
    was_bullish = False
    for i in range(n):
        if bullish[i] and not was_bullish:
            cur_entry = close[i]
        was_bullish = bullish[i]
        entry_price[i] = cur_entry
    return bullish, entry_price


def pyramid_activity_diagnostic(num_units_path: np.ndarray, bullish: np.ndarray) -> dict:
    """Disclosed diagnostic (not a kill switch): how often bullish episodes
    actually reach any pyramid activity at all -- failure mode (4)."""
    bullish_bars = int(bullish.sum())
    active_bars = int(np.sum((num_units_path > 0) & bullish))
    full_stack_bars = int(np.sum((num_units_path >= N_MAX_UNITS) & bullish))
    return {
        "bullish_bars": bullish_bars,
        "bars_with_any_units": active_bars,
        "fraction_bullish_with_units": (active_bars / bullish_bars) if bullish_bars else 0.0,
        "bars_at_full_stack": full_stack_bars,
        "fraction_bullish_at_full_stack": (full_stack_bars / bullish_bars) if bullish_bars else 0.0,
    }


# ================================================================== (2)
# CONSERVATIVE construction: literal Faith (2007) discrete unit stack,
# added on top of v4's own raw_desired exposure, capped at max_leverage.
# ==================================================================

def conservative_unit_path(df: pd.DataFrame, num_units_cap: int) -> np.ndarray:
    """Per-bar unit count (0..num_units_cap), the literal 0.5N-add /
    2N-from-last-add-stop state machine, reset at every new bullish
    episode. num_units_cap=0 -> permanently 0 (A1 identity)."""
    bullish, _entry = bullish_episode_state(df)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    units = np.zeros(n, dtype=int)
    if num_units_cap <= 0:
        return units
    atr = atr_n(df)
    num_units = 0
    last_add_price = close[0]
    was_bullish = False
    for i in range(n):
        if bullish[i] and not was_bullish:
            num_units = 0
            last_add_price = close[i]
        was_bullish = bullish[i]
        if not bullish[i]:
            num_units = 0
            units[i] = 0
            continue
        N = atr[i]
        if np.isfinite(N) and N > 0:
            if num_units < num_units_cap and (close[i] - last_add_price) >= ADD_THRESHOLD_N * N:
                num_units += 1
                last_add_price = close[i]
            if num_units > 0 and (last_add_price - close[i]) >= STOP_THRESHOLD_N * N:
                num_units = 0
                last_add_price = close[i]
        units[i] = num_units
    return units


def build_conservative_target(df: pd.DataFrame, num_units_cap: int) -> np.ndarray:
    """v4's own unmodified frac*scale, plus a discrete unit stack sized in
    UNIT_SIZE increments, clipped to v4's own max_leverage envelope, then
    v4's own unmodified 10% deadband."""
    raw = v4_raw_desired(df)
    if num_units_cap <= 0:
        return apply_deadband(raw, deadband=V4_DEADBAND)
    units = conservative_unit_path(df, num_units_cap)
    combined = raw + units * UNIT_SIZE
    combined = np.clip(combined, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)
    return apply_deadband(combined, deadband=V4_DEADBAND)


# ================================================================== (3)
# NOVEL construction: continuous, two-sided, episode-relative excursion
# multiplier on v4's raw_desired exposure (never touches the discrete
# unit/stop machinery).
# ==================================================================

def excursion_atr_units(df: pd.DataFrame) -> np.ndarray:
    """(close - episode_entry_price) / N_t, in ATR units, since the
    CURRENT bullish episode began; 0.0 while not in a bullish episode."""
    bullish, entry_price = bullish_episode_state(df)
    close = df["close"].to_numpy(dtype=float)
    atr = atr_n(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        units = np.where(atr > 0, (close - entry_price) / atr, 0.0)
    units = np.where(np.isfinite(units), units, 0.0)
    return np.where(bullish, units, 0.0)


def novel_multiplier(df: pd.DataFrame, kappa: float) -> np.ndarray:
    """1 + kappa*tanh(excursion_atr_units / NOVEL_REF_ATR_UNITS), clipped
    to NOVEL_MULT_CLIP. kappa=0.0 is the identity check (multiplier==1.0
    everywhere -> bit-for-bit v4)."""
    if kappa == 0.0:
        return np.ones(len(df))
    units = excursion_atr_units(df)
    mult = 1.0 + kappa * np.tanh(units / NOVEL_REF_ATR_UNITS)
    return np.clip(mult, NOVEL_MULT_CLIP[0], NOVEL_MULT_CLIP[1])


def build_novel_target(df: pd.DataFrame, kappa: float) -> np.ndarray:
    """v4's own unmodified frac*scale, multiplied by the two-sided
    excursion factor, clipped to v4's own max_leverage envelope, then
    v4's own unmodified 10% deadband."""
    raw = v4_raw_desired(df)
    if kappa == 0.0:
        return apply_deadband(raw, deadband=V4_DEADBAND)
    mult = novel_multiplier(df, kappa)
    combined = np.clip(raw * mult, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)
    return apply_deadband(combined, deadband=V4_DEADBAND)


# ================================================================== (4)
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
