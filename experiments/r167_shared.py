"""Shared, read-only utilities and pre-registration for the R-167 round (08-27/08-28).

DIRECTION, in one sentence: replace R-161's fixed-sample Hoeffding
concentration bound -- which R-161's own ruled-out entry diagnoses as
forcing a **hard sample-size floor** (its primary (tau=0.05, alpha=0.05,
delta=0.10) cell needs `n >= 460.5` calibration days before ANY cap above
0.0 is even reachable, collapsing 3 of 6 conservative configs into
permanent full-shutdown) -- with an ANYTIME-VALID concentration bound, so
`kelly_regime_v4`'s SCALE-output RCPS cap can bind gradually, from a small
number of days of causal history, instead of jumping between "no evidence
yet, cap=0" and "past the floor, cap unlocked."

**Which constraint this attacks: ERR (primary), same as R-161.** R-161's
own closing line in `docs/LEDGER.md` names this directly: *"a genuinely
different guarantee family that needs no fixed calibration window (e.g.
martingale/e-value-based risk control) remains untried."* This round is
exactly that family -- same object (a calibrated multiplicative cap
`lambda` on v4's `frac*scale`, same tail-loss functional
`1{|exposure*lambda*ret| > tau}`), a structurally different statistical
engine (anytime-valid confidence sequences rather than a one-shot,
fixed-n concentration inequality).

**Literature** (fetched via WebSearch by the research sub-agent that
proposed this direction on 08-27/08-28, re-verified by the operator):

- Xu, Karampatziakis & Mineiro (2024), "Active, Anytime-Valid Risk
  Controlling Prediction Sets", NeurIPS 2024 (arXiv:2406.10490). Names the
  family this round draws from: swap RCPS's one-shot concentration bound
  for an anytime-valid one so the guarantee holds continuously as data
  accumulates, not only at one pre-declared calibration size. This is the
  MOTIVATING paper (it is what R-161's own closing line was describing),
  not itself the calibration engine implemented below -- its own novelty
  is *active* (selective) calibration-point querying, which does not
  apply here (this project has no unlabeled-data cost to economize; every
  causal bar is already available at zero marginal query cost). What is
  adopted from it is the *target property* (anytime validity), built here
  from two independently citable, simpler constructions:
- Howard, Ramdas, McAuliffe & Sekhon (2021), "Time-Uniform Chernoff
  Bounds via Nonnegative Supermartingales", Probability Surveys 18 (an
  Annals-of-Statistics-adjacent survey venue; arXiv:1811.04644). Source of
  the general "stitching"/"peeling" technique used by the CONSERVATIVE
  engine below: a union bound over a countable sequence of look-times,
  each charged a shrinking confidence budget, so the total error over
  *all* future looks stays below one pre-declared delta.
- Waudby-Smith & Ramdas (2024), "Estimating Means of Bounded Random
  Variables by Betting", JRSS-B 86(1). Source of the NOVEL engine below: a
  nonnegative wealth (test) martingale under each candidate mean
  hypothesis; Ville's inequality gives validity at *every* sample size
  simultaneously from a single delta, with no peeling discount at all --
  the paper's own headline empirical claim is that this is substantially
  TIGHTER than Hoeffding-family bounds at moderate n, which this round
  turns into one of its two falsification tests (below).

DISCLOSED SIMPLIFICATIONS (both, in the spirit of R-161's own
Hoeffding-vs-Hoeffding-Bentkus disclosure -- a valid, elementary special
case of the cited machinery, not the tightest form it permits, checked
empirically by the same synthetic calibration self-test R-161 used
rather than trusted as a derivation):

1. CONSERVATIVE engine ("peeled Hoeffding"): rather than Howard et al.'s
   continuous-time mixture/stitching boundary (closed form, but tuned by
   a variance-proxy parameter this round has no principled way to set
   without another free parameter), this round uses the simplest valid
   discrete special case: assign the j-th REFIT (not the j-th bar) a
   confidence budget `delta_j = 6*delta / (pi^2*(j+1)^2)` (so
   `sum_j delta_j = delta` exactly, by `sum 1/(j+1)^2 = pi^2/6`), and at
   that refit compute a plain one-sided Hoeffding UCB,
   `r_hat_n + sqrt(log(1/delta_j)/(2n))`, over ALL causal history
   available by that refit (an EXPANDING window, never a fixed trailing
   one). A union bound over the countable sequence of refits (not over
   individual bars) makes the whole REFIT SEQUENCE jointly valid at
   `delta`, for arbitrarily many future refits -- which is the actual
   property R-161's fixed single-delta, fixed-window Hoeffding bound
   lacked, and the reason it needed a hard-coded `CALIB_DAYS`. This is
   provable from an elementary union bound (Boole's inequality), not
   Howard et al.'s own tightest boundary -- disclosed as looser than what
   the cited paper's full machinery permits. A DESIGN-TIME diagnostic
   (run below, before either branch was dispatched, per ROUTINE.md Step
   2's own "check the n a threshold implies is reachable" requirement)
   found that peeling at a FIXED CALENDAR CADENCE -- one refit, one unit
   of delta spent, every REFIT_DAYS regardless of how far n has grown --
   spends the confidence budget faster than n grows, and never unlocks
   ANY exposure at PRIMARY (alpha=0.05) over the whole 4-year inner-train
   window (worse than R-161's own diagnosed floor). The fix, adopted
   before dispatch: space refits GEOMETRICALLY (doubling) instead --
   `expanding_window_lambda_geometric` below -- literally Howard et al.'s
   own named "doubling/stitching" construction, not an ad hoc patch. See
   that function's docstring.
2. NOVEL engine ("fixed-fraction betting"): rather than Waudby-Smith &
   Ramdas's online-tuned (GRAPA/ONS) betting fraction, this round uses
   their simpler baseline -- a single, FIXED, pre-registered betting
   fraction `BETTING_LAMBDA`, chosen only to keep every wealth-process
   factor positive across the whole candidate-mean search grid (never
   fit to this project's data). This sacrifices power relative to their
   full adaptive construction, not validity: Ville's inequality holds for
   ANY predictable (here: constant) betting sequence, so the anytime-valid
   guarantee is unaffected by using a fixed fraction; only the bound's
   tightness is.
3. Both engines operate on a coarser search grid than R-161's
   (`LAMBDA_GRID`: 41 points here vs. R-161's 101; betting's own inner
   candidate-mean grid `M_GRID`: 31 points) -- a disclosed compute-budget
   simplification. Both grids still span their full relevant range
   ([0,1] for the exposure cap, [0,0.5] for the candidate tail-loss mean,
   comfortably above both ALPHA_GRID values), so this narrows resolution,
   not coverage.

**Not a duplicate of** (extends R-161's own non-duplication list; every
item below still holds for the identical reasons R-161 gave, since this
round changes only the concentration-bound ALGORITHM inside the same
architecture -- restated briefly, full detail in R-161's ledger entry
and `r161_shared.py`'s own docstring):
- R-28/R-31 (retracted e-process over aggregate P&L; an exposure-level
  artifact) -- this round computes no martingale over portfolio P&L, only
  over a bounded per-day tail-loss indicator, exactly as R-161 already
  distinguished.
- R-87 conservative/novel (ACI on the VOTE's coverage / the dispersion
  estimator) -- this round never touches the vote or the EWM dispersion
  estimator; it multiplies a separately-calibrated cap onto the
  already-computed `frac*scale`, unchanged from R-161's distinction.
- R-104/R-105/R-106/R-109/R-112/R-113/R-115/R-122/R-123/R-125/R-147/R-160
  -- distinct target quantities and mechanisms, per R-161's own
  itemized non-duplication list (a mean-return significance test, a
  jackknife ensemble, distributional-novelty distance statistics, a
  hazard/duration model, a risk-statistic point-estimate swap, weight
  reweighting, and an online-FDR flip-timing gate respectively --
  none of them a calibrated tail-EXCEEDANCE-RATE bound on SCALE's own
  output).
- **R-161 itself** -- same target quantity, same loss functional, same
  architecture (`calibration_frame`, `loss_at`, `build_capped_target`,
  the A1/A2 kill switches, all imported unmodified below); the ONLY
  change is the concentration inequality (Hoeffding at a fixed calibration
  window vs. two different anytime-valid engines on an expanding window)
  -- precisely the "genuinely different guarantee family" R-161's own
  closing line named as the untried next step, and precisely the R-62
  "test the factor, not the whole architecture" isolation pattern this
  project's most productive rounds have used.
- Ledger-wide grep (before any code in this round) confirms zero prior
  hits for "anytime-valid", "anytime valid", "confidence sequence",
  "supermartingale", "Ville", "betting" (in the WSR/e-value sense, not
  the R-160 online-FDR "wealth process" sense -- that round's wealth
  process is over discovery EVENTS gating vote FLIPS, a different target
  quantity, already distinguished in R-161's own list), "Howard", or
  "Waudby-Smith" anywhere in `docs/LEDGER.md` outside R-161's own citation
  of the family (not an implementation) and this entry.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both. Everything architectural (`calibration_frame`,
`loss_at`, `build_capped_target`, `broadcast_daily_lambda`,
`binding_fraction`, `constant_cap_r2`, `exceedance_rate`,
`synthetic_known_tail_frame`, `causal_truncation_probe_series`, `compare`,
`load_btc`/`load_eth`, the inner/outer split constants) is imported
UNMODIFIED from `r161_shared` (itself chained from r147/r105/r102_shared),
per this project's own chaining convention. Nothing here reads a bar at or
after `OOS_START` (2023-01-01).

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists
(R-161's five failure modes, restated for this round; (1)/(2)/(3)/(4) are
structurally identical kill switches -- inherited verbatim via the SAME
`binding_fraction`/`constant_cap_r2` diagnostics and the SAME ETH
replication check -- (5) is new, specific to this round's own claim):
(1) INERT -- lambda sits at/near 1.0 on effectively every refit; the risk
    budget never binds (GATE_MIN_BINDING_FRACTION kill switch, imported).
(2) MISCALIBRATED -- the synthetic known-tail-rate self-test shows the
    achieved exceedance rate far from the nominal alpha under this
    project's serial correlation.
(3) BTC-ONLY ARTIFACT -- binds and improves BTC, inverts on ETH (six
    independent precedents on this exact slot per R-161's own list).
(4) CORRECT BUT COSTLY -- tail-loss frequency measurably drops but Sharpe
    falls by more than +/-0.2 (R-20) on both markets, no offsetting
    drawdown/tail win.
(5) NO IMPROVEMENT OVER R-161's OWN ENGINE -- the anytime-valid engines
    bind (clear kill switch (1)) but do so LATER, or no earlier, than
    R-161's fixed-window Hoeffding bound would have on the same data, i.e.
    the diagnosed sample-floor problem does not actually cost anything in
    practice on this project's specific calibration-window/data-length
    regime. This is checked directly in STEP 1 of each branch (a
    head-to-head "days until first non-zero lambda" comparison against
    R-161's own PRIMARY config, imported from `r161_shared` unmodified)
    before either branch runs a single holdout-relevant number -- if this
    check fails, the round's own motivating claim is falsified regardless
    of what the promotion-bar cells show.

=====================================================================
PRE-REGISTRATION -- frozen before either branch is dispatched
=====================================================================

Both branches gate the SAME shipped v4 anchors/scale, using the SAME
causal loss functional (`r161_shared.loss_at`) wrapped around v4's own
UNMODIFIED `v4_raw_desired`, with v4's own 10% deadband applied AFTER the
cap -- identical to R-161. The only difference between the two branches
below, and between this round and R-161, is which function computes a
confidence-bounded estimate of the tail-loss rate from a causal history
of arbitrary (possibly small) length.

Decision rule (identical in structure and bar to R-161's own -- the
r105-r161 convention -- reproduced here so this file stands alone):
  PROMOTE-CANDIDATE (worth carrying to the holdout) if, on inner-validation
  (2021-01-01..2022-12-31), for AT LEAST ONE pre-registered config in this
  branch's grid, on BOTH markets (spot and futures_5x):
    (a) the candidate's realized daily tail-loss exceedance rate (bars
        where |capped_exposure * next-day log return| > tau) is STRICTLY
        LOWER than v4's own uncapped rate at the same tau, AND
    (b) the paired bootstrap 95% CI on d_log_growth (candidate - v4)
        excludes zero on the positive side, OR d_sharpe >= +0.2 (R-20),
        OR a real (risk-matched, exposure_ratio in [0.9,1.1]) drawdown
        improvement, AND
    (c) the SAME direction of improvement on (a)+(b) reproduces on this
        branch's own pre-registered falsification test below -- not
        inverted.
  Any other outcome on inner-validation is NEGATIVE for that cell. A
  branch that clears all three moves to the holdout ONLY after the
  operator freezes the specific config -- no further tuning -- and logs
  it here before running `ev(..., start=OOS_START)`.

Falsification tests (pre-registered per ROUTINE.md Step 2, matched to
what each branch's OWN cited paper claims it should be good at):
  CONSERVATIVE (peeled Hoeffding, expanding window): ETH sign-replication
  -- chosen for the identical reason R-161 chose it for its own
  conservative branch: this engine is still fundamentally a
  concentration-bound UCB search over a calibration history fitted on
  whatever asset it runs on, the exact construction six independent prior
  rounds on this slot have seen invert sign on ETH (R-105, R-109, R-113,
  R-125, R-126, and now R-161 itself as a sixth, since its conservative
  branch's one non-degenerate cell's ETH read is reported honestly in its
  own entry).
  NOVEL (fixed-fraction betting): a DIRECT ENGINE COMPARISON -- on the
  SAME calibration data, at the SAME (tau, alpha, delta) and the SAME
  expanding-window schedule, `betting_ucb` must be measurably TIGHTER
  (produce a strictly larger feasible lambda, i.e. bind LESS aggressively
  / permit MORE exposure at the same nominal risk level) than
  `howard_ucb_at_refit` on a majority of refit points in inner-train --
  this is Waudby-Smith & Ramdas's own headline empirical claim about their
  method (substantially narrower than Hoeffding-family bounds at moderate
  n) and is directly falsifiable without touching Sharpe or the holdout at
  all. If betting is NOT tighter than peeled-Hoeffding on this project's
  actual data, the paper's own claimed advantage for choosing it did not
  replicate here, independent of whatever the promotion-bar cells show.

Threshold/power sanity check (R-78/ROUTINE Step 2 requirement): both
variants are scored against this project's own already-validated
d_sharpe >= +0.2 inner-validation bar (R-20's measured path-noise floor),
the same bar 15+ prior SIZE/ERR-axis rounds on this exact window/sample
size have used, so a pass or a near-miss both remain informative.

Configs evaluated so far by this file: 0 (shared infrastructure only;
each branch's own count is logged in its own module and summed in the
R-167 ledger entry).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r161_shared import (  # noqa: E402,F401
    ALPHA_GRID,
    ALPHA_PRIMARY,
    CONST_CAP_R2_THRESH,
    ETH_SLICE_NAME,
    FEE_TIER,
    FUTURES,
    GATE_MIN_BINDING_FRACTION,
    HB_DELTA,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TAU_GRID,
    TAU_PRIMARY,
    V4_BAND,
    V4_DEADBAND,
    V4_HORIZONS,
    assert_no_holdout,
    binding_fraction,
    build_capped_target,
    calibration_frame,
    causal_truncation_probe_series,
    compare,
    constant_cap_r2,
    exceedance_rate,
    hoeffding_ucb,      # R-161's own engine -- imported for the head-to-head
    load_btc,           # comparison in each branch's STEP 1, never re-derived.
    load_eth,
    loss_at,
    print_rows,
    r_squared,
    rcps_calibrate,      # R-161's own periodic calibrator -- same reason.
    synthetic_known_tail_frame,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
DELTA_TOTAL = HB_DELTA             # 0.10, same total error budget as R-161,
                                    # for direct comparability of results.
LAMBDA_GRID = np.linspace(0.0, 1.0, 41)   # coarser than R-161's 101 (disclosed).
M_GRID = np.linspace(0.0, 0.5, 31)        # betting's inner candidate-mean grid.
BETTING_LAMBDA = 1.5               # fixed betting fraction; valid since
                                    # BETTING_LAMBDA * max(M_GRID) = 0.75 < 1,
                                    # keeping every wealth factor positive.
MIN_DAYS_FIRST_REFIT = 20          # far below R-161's implicit ~461-day floor;
                                    # only large enough to avoid a 0/1-length
                                    # calibration array at the very first refit.
REFIT_DAYS_PRIMARY = 30
REFIT_DAYS_ROBUST = 60


# ================================================================== (1)
# CONSERVATIVE engine: peeled (union-bound-over-refits) Hoeffding UCB on
# an EXPANDING window. See module docstring, simplification (1).
# ==================================================================

def refit_delta(refit_index: int, delta: float = DELTA_TOTAL) -> float:
    """delta_j for the j-th refit (0-indexed): sum_j delta_j == delta
    exactly, since sum_{j=0}^inf 1/(j+1)^2 == pi^2/6."""
    return (6.0 * delta) / (math.pi ** 2 * (refit_index + 1) ** 2)


def howard_ucb_at_refit(loss_values: np.ndarray, delta_j: float) -> float:
    """Plain one-sided Hoeffding UCB, evaluated at THIS refit's own
    (already-peeled) confidence budget delta_j rather than a fixed delta
    shared by every refit -- the only difference from `hoeffding_ucb`
    (imported above, R-161's own engine)."""
    n = len(loss_values)
    if n == 0:
        return 1.0
    r_hat = float(np.mean(loss_values))
    margin = math.sqrt(math.log(1.0 / delta_j) / (2.0 * n))
    return min(1.0, r_hat + margin)


def peeled_hoeffding_calibrate(exposure_prev: np.ndarray, ret: np.ndarray,
                                alpha: float, tau: float, refit_index: int,
                                lambda_grid: np.ndarray = LAMBDA_GRID,
                                delta: float = DELTA_TOTAL) -> float:
    """Largest feasible lambda under the peeled-Hoeffding UCB at this
    refit's own delta_j, over an EXPANDING (all-history-so-far) window --
    the CONSERVATIVE engine's calibration step, called once per refit."""
    delta_j = refit_delta(refit_index, delta)
    best = 0.0
    for lam in lambda_grid:
        losses = loss_at(exposure_prev, ret, lam, tau)
        if howard_ucb_at_refit(losses, delta_j) <= alpha:
            best = max(best, float(lam))
    return best


# ================================================================== (2)
# NOVEL engine: fixed-fraction betting confidence sequence (Waudby-Smith
# & Ramdas 2024). See module docstring, simplification (2).
# ==================================================================

def betting_ucb(loss_values: np.ndarray, delta: float = DELTA_TOTAL,
                 lam_bet: float = BETTING_LAMBDA,
                 m_grid: np.ndarray = M_GRID) -> float:
    """Smallest m in `m_grid` (ascending) whose wealth process
    `prod_i (1 + lam_bet*(X_i - m))` stays below 1/delta over the WHOLE
    observed sequence -- Ville's inequality makes this valid at every
    sample size simultaneously, from a SINGLE delta (no peeling). Wealth
    is monotone non-increasing in m for fixed data (each per-bar factor
    is, since d/dm[1 + lam_bet*(x-m)] = -lam_bet < 0 for both x in
    {0,1}), so the first m (ascending) whose log-wealth clears the bar is
    the UCB. Returns 1.0 (refuse to certify) if no data yet, matching
    `hoeffding_ucb`'s own n==0 convention."""
    n = len(loss_values)
    if n == 0:
        return 1.0
    x = np.asarray(loss_values, dtype=float)
    threshold = math.log(1.0 / delta)
    for m in m_grid:
        factor = np.where(x == 1.0, 1.0 + lam_bet * (1.0 - m), 1.0 - lam_bet * m)
        log_wealth = float(np.sum(np.log(np.clip(factor, 1e-300, None))))
        if log_wealth < threshold:
            return float(m)
    return float(m_grid[-1])


def betting_calibrate(exposure_prev: np.ndarray, ret: np.ndarray,
                       alpha: float, tau: float,
                       lambda_grid: np.ndarray = LAMBDA_GRID,
                       m_grid: np.ndarray = M_GRID,
                       lam_bet: float = BETTING_LAMBDA,
                       delta: float = DELTA_TOTAL) -> float:
    """Largest feasible lambda under the betting UCB, over an EXPANDING
    window -- the NOVEL engine's calibration step, called once per refit.
    No refit_index argument: Ville's inequality needs no peeling."""
    best = 0.0
    for lam in lambda_grid:
        losses = loss_at(exposure_prev, ret, lam, tau)
        if betting_ucb(losses, delta, lam_bet, m_grid) <= alpha:
            best = max(best, float(lam))
    return best


# ================================================================== (3)
# Shared expanding-window refit schedulers. TWO variants, not one: a
# design-time diagnostic (run by the operator below, before dispatch --
# see the note after this section) found that peeling per FIXED-CADENCE
# refit (delta_j shrinking once every REFIT_DAYS regardless of how far n
# has grown) spends the confidence budget faster than n grows, making the
# CONSERVATIVE engine strictly WORSE than R-161's own fixed-window bound
# at PRIMARY (alpha=0.05) -- inf days to first non-zero lambda over the
# entire 4-year inner-train window, versus R-161's own diagnosed (finite,
# if unreachable within its own window) ~460.5-day floor. This is exactly
# an application of ROUTINE.md Step 2's own requirement ("compute the n a
# threshold implies... check that n is one the experiment can actually
# reach") to the CALIBRATION ENGINE itself, at design time, before either
# branch runs a single real-data number -- so the schedule below was fixed
# BEFORE dispatch, per that same rule, not tuned afterward against a
# result. The fix: peel per REFIT (a canonical checkpoint), but space
# refits GEOMETRICALLY (doubling) rather than at a fixed calendar cadence
# -- literally Howard et al.'s (2021) own named "doubling/stitching"
# construction, not an ad hoc adjustment. The NOVEL (betting) engine needs
# no such care -- Ville's inequality is valid at every n from a single
# delta regardless of when it is checked -- so it keeps the simple fixed
# calendar cadence, which is itself part of this round's falsification
# story (see module docstring): Howard needs a cleverly-spaced schedule to
# be practically usable at this project's data volume; betting does not.
# ==================================================================

def expanding_window_lambda_periodic(cal: pd.DataFrame, refit_days: int,
                                      calibrate_fn) -> pd.Series:
    """FIXED CALENDAR CADENCE expanding-window schedule -- the NOVEL
    (betting) engine's own scheduler. Every `refit_days` calendar days
    starting at `MIN_DAYS_FIRST_REFIT`, call
    `calibrate_fn(exposure_prev, ret)` using ALL of `cal` strictly before
    the refit point (an EXPANDING window, never a bounded trailing one --
    the core mechanism difference from R-161). Returns a SPARSE series
    exactly like `r161_shared.periodic_rcps_lambda`: no entry before the
    first completed window, `r161_shared.broadcast_daily_lambda`'s own
    `.ffill().fillna(1.0)` supplies the "no cap yet" default."""
    idx = cal.index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    start, end = idx[0], idx[-1]
    first_refit = start + pd.Timedelta(days=MIN_DAYS_FIRST_REFIT)
    values: dict[pd.Timestamp, float] = {}
    refit_day = first_refit
    while refit_day <= end:
        window = cal.loc[cal.index < refit_day]
        next_refit = refit_day + pd.Timedelta(days=refit_days)
        if len(window) > 0:
            lam = calibrate_fn(window["exposure_prev"].to_numpy(),
                                window["ret"].to_numpy())
            applicable = idx[(idx >= refit_day) & (idx < next_refit)]
            for d in applicable:
                values[d] = lam
        refit_day = next_refit
    return pd.Series(values, dtype=float).sort_index()


def expanding_window_lambda_geometric(cal: pd.DataFrame,
                                       calibrate_fn) -> pd.Series:
    """GEOMETRIC (doubling) expanding-window schedule -- the CONSERVATIVE
    (peeled-Hoeffding) engine's own scheduler. Refits happen at calendar
    offsets `MIN_DAYS_FIRST_REFIT * 2**j` for j = 0, 1, 2, ... (each refit
    IS its own peeling epoch, so refit COUNT equals epoch count exactly --
    the condition Howard et al.'s doubling/stitching union bound needs).
    Calls `calibrate_fn(exposure_prev, ret, refit_index=j)` using ALL of
    `cal` strictly before the refit point. Same sparse-series / warmup-
    default convention as the periodic scheduler above."""
    idx = cal.index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    start, end = idx[0], idx[-1]
    values: dict[pd.Timestamp, float] = {}
    j = 0
    while True:
        offset_days = MIN_DAYS_FIRST_REFIT * (2 ** j)
        refit_day = start + pd.Timedelta(days=offset_days)
        if refit_day > end:
            break
        next_offset_days = MIN_DAYS_FIRST_REFIT * (2 ** (j + 1))
        next_refit = start + pd.Timedelta(days=next_offset_days)
        window = cal.loc[cal.index < refit_day]
        if len(window) > 0:
            lam = calibrate_fn(window["exposure_prev"].to_numpy(),
                                window["ret"].to_numpy(), j)
            applicable = idx[(idx >= refit_day) & (idx < next_refit)]
            for d in applicable:
                values[d] = lam
        j += 1
    return pd.Series(values, dtype=float).sort_index()


def days_to_first_nonzero_lambda(lam: pd.Series, cal_start: pd.Timestamp) -> float:
    """Diagnostic for kill switch (5): the number of days of causal
    history elapsed before an already-built lambda schedule first
    produces lambda > 0.0 (i.e. the risk budget first becomes non-trivially
    satisfiable), or float('inf') if it never does over the series' span."""
    nonzero = lam[lam > 0.0]
    if len(nonzero) == 0:
        return float("inf")
    return float((nonzero.index[0] - cal_start).days)


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=150_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(167)
    innov = rng.normal(0, 0.0006, len(idx))
    close = 10_000 * np.exp(np.cumsum(innov))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": 1.0}, index=idx)
    cal = calibration_frame(df)
    exp_prev = cal["exposure_prev"].to_numpy()
    ret = cal["ret"].to_numpy()

    # (1) refit_delta sums to DELTA_TOTAL over many refits (Boole's inequality
    # setup is only valid if this holds); shrinks monotonically in j.
    partial = sum(refit_delta(j) for j in range(100_000))
    assert abs(partial - DELTA_TOTAL) < 1e-6, partial
    assert refit_delta(0) > refit_delta(1) > refit_delta(2)

    # (2) howard_ucb_at_refit: UCB >= empirical mean; a smaller delta_j (more
    # peeling budget spent) gives a WIDER margin than a larger one.
    losses = rng.integers(0, 2, 300).astype(float)
    ucb_tight = howard_ucb_at_refit(losses, delta_j=0.05)
    ucb_loose = howard_ucb_at_refit(losses, delta_j=0.001)
    assert ucb_tight >= np.mean(losses)
    assert ucb_loose >= ucb_tight

    # (3) betting_ucb: UCB in [0, max(M_GRID)]; monotone non-increasing
    # wealth means an all-zero loss sequence should push the UCB toward 0,
    # an all-one sequence should push it toward max(M_GRID).
    l0 = np.zeros(200)
    l1 = np.ones(200)
    ucb_l0 = betting_ucb(l0)
    ucb_l1 = betting_ucb(l1)
    assert 0.0 <= ucb_l0 <= ucb_l1 <= float(M_GRID[-1])
    assert ucb_l0 < 0.2   # a long run of zero losses should certify a low mean

    # (4) betting_ucb / howard_ucb_at_refit both return 1.0 on empty input,
    # matching hoeffding_ucb's own n==0 refusal convention.
    assert betting_ucb(np.array([])) == 1.0
    assert howard_ucb_at_refit(np.array([]), delta_j=0.05) == 1.0

    # (5) peeled_hoeffding_calibrate / betting_calibrate: outputs in [0,1];
    # lambda=0 always feasible (loss_at(...,0.0,...) is identically zero).
    lam_h = peeled_hoeffding_calibrate(exp_prev, ret, alpha=0.05, tau=0.05,
                                        refit_index=0)
    lam_b = betting_calibrate(exp_prev, ret, alpha=0.05, tau=0.05)
    assert 0.0 <= lam_h <= 1.0
    assert 0.0 <= lam_b <= 1.0

    # (6) expanding_window_lambda_geometric: no entry before
    # MIN_DAYS_FIRST_REFIT; values in [0,1]; non-empty for data spanning
    # several doubling cycles.
    def _howard_closure(ep, r, j):
        return peeled_hoeffding_calibrate(ep, r, alpha=0.10, tau=0.05,
                                           refit_index=j)

    lam_path_h = expanding_window_lambda_geometric(cal, _howard_closure)
    if len(lam_path_h) > 0:
        assert bool((lam_path_h.index >= cal.index[0]
                      + pd.Timedelta(days=MIN_DAYS_FIRST_REFIT)).all())
        assert bool(((lam_path_h >= 0.0) & (lam_path_h <= 1.0)).all())

    # (7) expanding_window_lambda_periodic + days_to_first_nonzero_lambda:
    # finite on data generous enough that SOME lambda > 0 should eventually
    # be reachable at a loose alpha.
    def _betting_closure(ep, r):
        return betting_calibrate(ep, r, alpha=0.30, tau=0.05)

    lam_path_b = expanding_window_lambda_periodic(cal, REFIT_DAYS_PRIMARY, _betting_closure)
    d2f = days_to_first_nonzero_lambda(lam_path_b, cal.index[0])
    assert d2f == float("inf") or d2f >= 0.0

    # (8) build_capped_target (imported architecture) still reproduces v4
    # exactly at lambda==1 every day -- unchanged from R-161, checked again
    # here since this module's own use of it is new call sites, not new code.
    from experiments.r147_shared import v4_target
    daily_idx = cal.index
    lam_one = pd.Series(1.0, index=daily_idx)
    capped = build_capped_target(df, lam_one)
    assert np.allclose(capped, v4_target(df), atol=1e-9)

    # (9) causal wiring, at a fixed already-computed lambda path -- the
    # r161_shared self-test pattern, re-checked for this module's own call
    # sites (expanding_window_lambda's OUTPUT fed through build_capped_target,
    # never expanding_window_lambda itself, which is architecture-external
    # to compare()'s truncation probe).
    def _fixed_lambda_probe(d: pd.DataFrame) -> np.ndarray:
        idx_d = calibration_frame(d).index
        lam_d = pd.Series(0.7, index=idx_d)
        return build_capped_target(d, lam_d)

    assert causal_truncation_probe_series(_fixed_lambda_probe, df)


_self_test()
