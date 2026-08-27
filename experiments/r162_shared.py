"""Shared, read-only utilities and pre-registration for the R-162 round (08-27).

DIRECTION, in one sentence: give kelly_regime_v4's SIZE mechanism a
genuinely TWO-SIDED conviction modifier -- Kaufman's (1995) Efficiency
Ratio (ER), a causal, price-only, symmetric [0,1] trend-straightness
statistic -- applied at two different points in v4's architecture: the
post-vote SCALE output (conservative) and the pre-latch VOTE combination
(novel).

**Which constraint this attacks: SIZE.** No new external data source is
introduced (ER is a pure function of the same `close` column v4 already
reads), so this does not touch INFO; it does not alter vote-flip timing or
add a formal statistical guarantee (distinct from R-160's online-FDR flip
gate and R-161's Conformal Risk Control SCALE cap, both ERR-axis); it adds
no no-trade-band or execution-cost logic (distinct from every COST-axis
construction in this ledger). It changes how much to hold, using only
information v4 already has -- exactly the family the standing diagnosis
credits as the one that has worked.

**Literature:**
- Perry J. Kaufman, *Smarter Trading* (1995) -- defines the Efficiency
  Ratio (`|net move| / sum(|bar-to-bar move|)`), the signal-to-noise
  engine behind Kaufman's Adaptive Moving Average (KAMA). A definitional/
  heuristic construction, not an empirically-backtested cross-instrument
  claim -- that limitation is disclosed, not proxied over.
  https://www.tradingview.com/support/solutions/43000773012-kaufman-s-adaptive-moving-average-kama/
- "Trends, Volatility, Correlations, and Critical Phenomena in Financial
  Markets" (2026), arXiv:2606.20145 -- finds next-day variance is a
  function of current trend strength (a leverage-effect analogue) on
  broad equity/futures price data (not crypto-specific, no per-instrument
  cost breakdown -- disclosed, not proxied). Grounds *why* trend
  straightness should carry forward information distinct from realized
  volatility, which is what v4's SCALE term already measures.
  https://arxiv.org/html/2606.20145

**Not a duplicate of, by ledger ID:**
- R-34 / R-41(conservative) / R-46(conservative) / R-53(conservative) --
  four confirmed instances of a ONE-SIDED, never-increase-only
  multiplicative brake fed by an EXTERNAL signal (Harsanyi posterior,
  basis magnitude, CPPI/Hurst, VIX/DXY), all failing the identical
  exposure-artifact test under equality-matching. This construction is
  explicitly TWO-SIDED (can push the multiplier/weight above 1 as well as
  below), self-normalizes to a neutral value by construction, and is
  checked with the inequality/tolerance exposure-match `compare()` already
  computes (`risk_matched`, exposure_ratio in [0.9, 1.1]) rather than
  calibrated to it -- the exact fix R-141 named and left untested: "use
  inequality matching or a two-sided (both-directions) multiplier
  instead."
- R-89 (response-curve shape -- sign/linear/cubic on a continuous 365-day
  rolling-RMS trend-MAGNITUDE statistic, reshaping the VOTE's own output
  curve). Different statistic (Kaufman displacement/path-length
  straightness, not an RMS z-score) at a different locus (SCALE for the
  conservative branch; per-anchor vote WEIGHTS for the novel branch --
  never the vote's response curve).
- R-40 (Bayesian-Kelly shrink by cross-ladder disagreement across R-07's
  separately-bagged ensemble, one-sided-only, applied to the whole blended
  vote). Different statistic (ER, not disagreement-STD), applied per-anchor
  as a two-sided weight, on v4's own unmodified 20/40/80 architecture.
- R-146 (median / jump-exclusion anchor statistic -- changes how each
  anchor's OWN trend value is computed). This leaves each anchor's raw
  trend calculation untouched and only reweights the three anchors'
  contribution to the vote, or multiplies the post-vote scale -- a
  different locus entirely.
- R-160 (online-FDR gating flip TIMING) / R-161 (Conformal Risk Control /
  RCPS capping SCALE via a calibrated tail-loss bound). Both are formal
  statistical-guarantee machinery on different objects (flip acceptance;
  a lambda cap calibrated against a tail-exceedance functional). This is a
  structural, non-fitted, self-normalizing overlay with no coverage
  guarantee -- a different mechanism class, not a re-try of either with
  different constants.
- R-59/R-60 (per-asset or self-normalizing `target_vol`/`max_leverage`
  retunes -- the MAGNITUDE of the existing scale constant). This
  multiplies the existing scale by an independently-derived conviction
  ratio, leaving `target_vol`/`max_leverage` untouched.
- Ledger-wide grep (`efficiency ratio|Kaufman|KAMA|fractal efficiency`)
  returns zero hits anywhere in docs/LEDGER.md outside this entry, and the
  construction names nothing on the 08-26/27 twenty-pass verification
  archive's own closed-candidate lists (wavelets, GP regression, particle
  filters, DRO/Wasserstein-Kelly, EVT/Hill-POT-GPD, meta-labeling,
  path-signature detection, execution bandits, BOCPD/HSMM, RL sizing, TDA,
  stablecoin-copula, liquidation-cascade CSD, Fear&Greed, ZigZag/Elliott).

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- the r89-r161 convention. Nothing here reads a
bar at or after OOS_START (2023-01-01); `compare()` (imported from
`r161_shared`, chained from r147/r105/r102_shared) never touches the
holdout.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) BTC-ONLY ARTIFACT -- clears BTC but ER's sign inverts on ETH. This
    project's single most common failure mode for SIZE-axis constructions
    on this slot (R-33/57/64/127/137/145/149/150 among others). Chosen as
    THE pre-registered falsification test for both branches below.
(2) COLD-START DEGENERACY -- ER's 365-day trailing-median reference has no
    real data for its first year (defaults to the neutral midpoint 0.5),
    so the multiplier/weight is inert during that stretch by construction,
    not by finding -- disclosed below, not hidden, and excluded from
    neither slice (inner-train is long enough to absorb it).
(3) MICROSTRUCTURE-DOMINATED ER -- at 5-minute-bar granularity, ER over an
    80-day (23,040-bar) window may be dominated by pure back-and-forth
    noise rather than genuine trend quality, making it a disguised
    inverse-volatility proxy collinear with v4's own SCALE term (the A2
    kill switch below: R^2 of the candidate's raw exposure path vs v4's
    own must be < 0.98, else this is a relabeling, not a new mechanism).
(4) DELAYED-FLIP COST (novel branch specifically) -- down-weighting a
    noisy anchor exactly as it approaches a flip delays that flip, the
    identical failure mode that sank R-160's online-FDR flip-gate
    (delaying costs more Sharpe/drawdown than the added precision buys).
    Reported as an explicit diagnostic (`delayed_flip_diagnostic` below),
    not folded into the headline number.

=====================================================================
PRE-REGISTRATION -- frozen before either branch is dispatched
=====================================================================

Both branches gate the SAME shipped v4 anchors/scale (20/40/80-day vote,
1% band, conditional target-vol scale) and use the SAME ER construction
(`efficiency_ratio` / `er_reference` below) -- the only difference between
them is WHERE the ER-derived factor is applied: post-vote SCALE
(conservative) vs pre-latch VOTE combination (novel). Configurations swept:
gamma/beta in {0.0 (identity check), 0.5, 1.0, 1.5} x 2 markets x 3 slices
(inner-train, inner-validation, ETH replication) = 24 cells per branch,
plus a 0.40%-fee-tier re-run of each branch's finalist config on both
markets (+2 cells) = 26 configurations per branch, **52 total** across both
branches -- the number that goes into the deflated-Sharpe calculation at
Step 4, per ROUTINE.md's "trials count is the total across all parallel
branches" rule.

Falsification test (pre-registered, IDENTICAL for both branches, chosen
per the "what would make this fail" list above): ETH sign-replication.
On whichever market (spot or futures_5x) a branch's primary config
(gamma=1.0 / beta=1.0) clears `clears_bar()` on inner-validation, the
identical construction run on ETH's replication slice must show the SAME
SIGN of `d_log_growth` vs the v4 control. Sign inversion on the market
that passed -> REJECT for that branch, independent of every other cell.

Step-0 non-degeneracy kill switches (checked before any Sharpe number is
trusted):
  A1 (identity sanity): gamma=0.0 / beta=0.0 must reproduce v4_target
      bit-for-bit (R^2 == 1.0, `d_log_growth` == 0.0) -- proves the
      construction collapses to v4 exactly at its neutral setting.
  A2 (non-inertness / non-collinearity): R^2 of the candidate's raw
      pre-deadband exposure path vs v4's own `v4_raw_desired`, at the
      PRIMARY (gamma=1.0 / beta=1.0) setting, must be < CONST_CAP_R2_THRESH
      (0.98) -- else the ER factor never meaningfully moves exposure and
      this is a relabeling of v4, not a tested mechanism.
  A3 (exposure match): `compare()`'s own `risk_matched` field (exposure
      AND realized-vol ratio both in [0.9, 1.1], the ledger's standard
      inequality-tolerance convention) reported for every cell -- not
      calibrated to, since this construction is self-normalizing by
      design; failing it is itself a disclosed finding, not grounds to
      refit.

Decision rule (exhaustive partition of the outcome space, evaluated
independently for each branch):
  CLEAR(m) = clears_bar() on inner-validation for at least one non-zero
             gamma/beta in the grid, on market m.
  GATE_OK  = ETH-falsification passes (no sign inversion on the market
             that cleared) AND the non-zero grid values that clear share
             the same sign of d_log_growth on inner-validation (a
             plateau, not an isolated spike).

  | GATE_OK | CLEAR(spot) | CLEAR(futures) | Verdict  |
  |---------|-------------|-----------------|----------|
  | false   | --          | --              | REJECT   |
  | true    | false       | false           | REJECT   |
  | true    | true        | false (or v.v.) | PARTIAL  |
  | true    | true        | true            | PROMOTE  |

A branch that reaches PARTIAL or PROMOTE moves to the holdout
(`ev(..., start=OOS_START)`) ONLY after the operator confirms the frozen
gamma/beta value here -- no further tuning -- exactly as R-161 required.

Configs evaluated so far by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the R-162
ledger entry).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import _latched_anchor_vote  # noqa: E402
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

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
ER_REF_DAYS = 365                   # structural, non-fitted baseline (R-89 convention)
GRID = (0.0, 0.5, 1.0, 1.5)         # shared gamma (conservative) / beta (novel) grid
PRIMARY = 1.0                       # primary config for the A2 kill switch + falsification
SCALE_MULT_CLIP = (0.5, 1.5)        # conservative: fixed, disclosed, non-fitted symmetry cap
VOTE_WEIGHT_CLIP = (0.0, 2.0)       # novel: per-anchor weight floor/ceiling
FEE_TIER = 0.0040                   # cost-robustness sensitivity re-run
SHARPE_NOISE_FLOOR = 0.2            # ROUTINE.md's own promotion bar (R-20)
CONST_CAP_R2_THRESH = 0.98          # A2 kill switch: candidate raw exposure path vs v4's own
ER_SCALE_WINDOW_DAYS = V4_HORIZONS[-1]  # 80 days -- v4's own slowest anchor, zero new parameter


# ================================================================== (1)
# Kaufman (1995) Efficiency Ratio -- causal, price-only, symmetric [0,1]
# trend-straightness statistic. Computed once here so both branches use
# the identical primitive; only WHERE it is applied differs between them.
# ==================================================================

def efficiency_ratio(close: pd.Series, window_bars: int) -> pd.Series:
    """|net move over window_bars| / sum(|bar-to-bar move|) over the same
    window -- 1.0 for a straight-line move, ->0 for pure back-and-forth
    chop. Causal: row i uses only close[i-window_bars .. i]."""
    net_move = (close - close.shift(window_bars)).abs()
    path_length = close.diff().abs().rolling(window_bars).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(path_length.to_numpy() > 0,
                      net_move.to_numpy() / path_length.to_numpy(), np.nan)
    return pd.Series(er, index=close.index).fillna(0.0)


def er_reference(er: pd.Series, ref_days: int = ER_REF_DAYS) -> pd.Series:
    """Trailing rolling MEDIAN of ER over `ref_days`, shifted one bar so
    bar i's reference uses only ER values strictly before bar i -- a
    structural, non-fitted baseline (the 365-day rolling-window convention
    R-89 used for its own trend-magnitude statistic). Defaults to 0.5 (the
    midpoint of ER's own [0,1] range, i.e. "no information yet" maps to a
    neutral multiplier/weight) before a full reference window exists --
    the disclosed cold-start degeneracy named in the module docstring."""
    ref_bars = ref_days * BARS_PER_DAY
    return er.rolling(ref_bars, min_periods=BARS_PER_DAY).median().shift(1).fillna(0.5)


# ================================================================== (2)
# CONSERVATIVE construction: post-vote, pre-deadband SCALE multiplier.
# ==================================================================

def er_scale_multiplier(df: pd.DataFrame, gamma: float,
                         clip: tuple[float, float] = SCALE_MULT_CLIP) -> np.ndarray:
    """1 + gamma*(ER_t - ER_ref_t), clipped to `clip`. gamma=0.0 is the
    identity check (multiplier == 1.0 everywhere -> bit-for-bit v4)."""
    close = df["close"]
    er = efficiency_ratio(close, ER_SCALE_WINDOW_DAYS * BARS_PER_DAY)
    ref = er_reference(er)
    mult = 1.0 + gamma * (er.to_numpy() - ref.to_numpy())
    return np.clip(mult, clip[0], clip[1])


def build_conservative_target(df: pd.DataFrame, gamma: float) -> np.ndarray:
    """v4's own unmodified frac*scale, multiplied by the ER conviction
    factor, then v4's own unmodified 10% deadband -- the ONLY change from
    v4 is this one multiplicative factor inserted before the deadband."""
    raw = v4_raw_desired(df)
    mult = er_scale_multiplier(df, gamma)
    return apply_deadband(raw * mult, deadband=V4_DEADBAND)


# ================================================================== (3)
# NOVEL construction: per-anchor ER conviction weighting of the VOTE
# combination step (never touches SCALE).
# ==================================================================

def er_vote_weighted_frac(df: pd.DataFrame, beta: float,
                           clip: tuple[float, float] = VOTE_WEIGHT_CLIP) -> pd.Series:
    """Replace v4's plain mean of three latched anchor votes with a
    per-anchor ER-conviction-weighted mean: an anchor whose OWN horizon is
    currently trending cleanly counts more than one that is currently
    chopping at its own timescale. beta=0.0 is the identity check (equal
    weights -> bit-for-bit v4_vote_frac)."""
    close = df["close"]
    votes, weights = [], []
    for days in V4_HORIZONS:
        v = _latched_anchor_vote(close, days, V4_BAND)
        er = efficiency_ratio(close, days * BARS_PER_DAY)
        ref = er_reference(er)
        w = np.clip(1.0 + beta * (er.to_numpy() - ref.to_numpy()), clip[0], clip[1])
        votes.append(v.to_numpy())
        weights.append(w)
    votes_arr = np.asarray(votes)      # (3, n)
    weights_arr = np.asarray(weights)  # (3, n)
    wsum = weights_arr.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(wsum > 0, (weights_arr * votes_arr).sum(axis=0) / wsum, 0.0)
    return pd.Series(frac, index=close.index)


def build_novel_target(df: pd.DataFrame, beta: float) -> np.ndarray:
    """v4's own unmodified SCALE, combined with the ER-weighted vote in
    place of v4's plain mean vote, then v4's own unmodified 10% deadband."""
    frac = er_vote_weighted_frac(df, beta).to_numpy()
    scale = v4_scale(df)
    return apply_deadband(frac * scale, deadband=V4_DEADBAND)


def delayed_flip_diagnostic(df: pd.DataFrame, beta: float) -> dict:
    """Disclosed diagnostic (not a kill switch): counts bars where v4's
    own unweighted combined vote crosses its 0.5 latch threshold and the
    ER-weighted vote does not cross in the same direction within the next
    30 bars -- the R-160 delayed-flip failure signature, reported
    explicitly rather than folded into the headline number."""
    v4_frac = v4_vote_frac(df).to_numpy()
    novel_frac = er_vote_weighted_frac(df, beta).to_numpy()

    def crossings(frac: np.ndarray) -> np.ndarray:
        above = frac > 0.5
        return np.flatnonzero(above[1:] != above[:-1]) + 1

    v4_cross = crossings(v4_frac)
    novel_cross = crossings(novel_frac)
    delayed = 0
    for i in v4_cross:
        window = novel_cross[(novel_cross >= i) & (novel_cross <= i + 30)]
        if len(window) == 0:
            delayed += 1
    return {
        "v4_flip_count": int(len(v4_cross)),
        "novel_flip_count": int(len(novel_cross)),
        "flips_with_no_match_within_30bars": int(delayed),
        "delayed_fraction": float(delayed / len(v4_cross)) if len(v4_cross) else 0.0,
    }


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
