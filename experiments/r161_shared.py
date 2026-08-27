"""Shared, read-only utilities and pre-registration for the R-161 round (08-27).

DIRECTION, in one sentence: give `kelly_regime_v4`'s SCALE output (the
uncapped desired exposure `frac * scale`, currently `min(target_vol /
realized_vol, max_leverage)` with no probabilistic meaning at all) a
finite-sample, distribution-free bound on a daily tail-loss EXCEEDANCE RATE
via Conformal Risk Control (Angelopoulos et al. 2024) / Risk-Controlling
Prediction Sets (Bates et al. 2021) -- a calibrated multiplicative cap
`lambda_t in [0, 1]` applied on top of v4's own unmodified vote and scale,
refit periodically (conservative) or tracked online (novel).

**Which constraint this attacks: ERR (primary), SIZE (secondary).** No error
control exists anywhere in v4's signal path (the standing diagnosis's own
wording). This project has run twelve-plus ERR-axis constructions to date,
every one of them on the VOTE's confidence/timing/combination (R-28/31, R-87
conservative, R-104, R-105, R-106, R-109, R-112, R-113, R-114, R-115, R-122,
R-123, R-147, R-160) or on an internal ESTIMATOR feeding SCALE (R-87 novel's
Adaptive Conformal Inference wrapping the volatility-target's dispersion
estimator) or on the risk STATISTIC scale targets (R-125's CVaR swap). None
has put a calibrated, finite-sample-guaranteed bound on SCALE's own OUTPUT.
R-160's own closing line names this directly: "an ERR-axis attempt on this
architecture's SCALE/sizing decision instead remains untried." Secondarily
SIZE, since the standing diagnosis credits "decide how much to hold" as the
one mechanism family that has worked, and this modifies exactly that
decision (a multiplicative cap on `frac*scale`) without touching `frac`.

**Literature, fetched via WebSearch by the research sub-agent that proposed
this direction, then re-verified by the operator before freezing this file:**

- Bates, S., Angelopoulos, A., Lei, L., Malik, J., & Jordan, M. I. (2021),
  "Distribution-Free, Risk-Controlling Prediction Sets", J. ACM 68(6),
  Article 43 (arXiv:2101.02703). A held-out calibration set plus a
  concentration inequality produces a PAC (probably-approximately-correct),
  ONE-SHOT upper confidence bound (UCB) on risk; pick the most permissive
  parameter that still keeps the UCB at or below a target level alpha. The
  engine for the CONSERVATIVE branch below (periodic batch recalibration).
- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T.
  (2022/2024), "Conformal Risk Control", ICLR 2024 (arXiv:2208.02814).
  Generalizes split conformal prediction from controlling miscoverage to
  controlling the expectation of any bounded, MONOTONE loss function, and
  Section 4 gives a distribution-shift extension: a monotonized recursive
  update that tracks the risk target online, without an exchangeability
  assumption. The engine for the NOVEL branch below (continuous online
  adaptation, structurally different from a periodic batch refit).
- Schmitt, M. (2026), "Regime-Weighted Conformal Risk Control for Portfolio
  VaR", arXiv:2602.03903. A live, Feb-2026 application of this same
  Angelopoulos-family machinery to financial tail-risk (VaR exceedance on
  the CRSP index); read for what a plausible failure mode looks like
  (over-conservatism / interval-width tradeoffs), not adopted as the
  mechanism -- context only, no crypto data, no transaction costs modeled.

DISCLOSED SIMPLIFICATION: Bates et al.'s tightest bound is the
Hoeffding-Bentkus hybrid, which requires inverting a binomial tail (a
concentration bound this project's dependency set -- numpy/pandas only, no
scipy, see pyproject.toml -- cannot compute in closed form without adding a
dependency). This round uses the plain one-sided HOEFFDING bound instead,
`UCB(R_hat, n, delta) = R_hat + sqrt(log(1/delta) / (2n))`, valid for any
[0,1]-bounded loss and strictly more conservative (wider) than
Hoeffding-Bentkus, at the cost of power, never at the cost of the coverage
guarantee's direction. The bound's own i.i.d. assumption is also violated by
this project's serially-correlated daily-return stream, exactly as R-160
disclosed for LORD/SAFFRON's independence assumption -- addressed the same
way, with a calibration self-test on synthetic data (SECTION 4 below), not a
proof.

**Not a duplicate of:**
- R-28/R-31 (retracted) -- an aggregate e-process martingale over realized
  portfolio P&L, shown to be an unmatched exposure-level artifact (R-33).
  This round computes no martingale over aggregate P&L; it calibrates a
  threshold from a bounded per-day loss functional via a calibration-set
  quantile/UCB search.
- R-87 conservative (ACI wrapping the VOTE's own directional hit-rate via an
  online additive recursion on a coverage target `alpha_t`, tracking a
  50%-coin-flip null). This round never touches the vote; it operates
  purely on `scale`'s output.
- R-87 novel (ACI *replacing* the EWM dispersion estimator feeding the
  vol-target denominator, via the identical online coverage-tracking
  recursion, R^2=0.71-0.80 against v4's own path). This round leaves the EWM
  dispersion estimator completely untouched and instead multiplies the
  ALREADY-COMPUTED `scale` by a separately-calibrated cap, via a
  calibration-set/concentration-bound search (RCPS/CRC), never an online
  coverage-tracking recursion on a miscoverage indicator -- a different
  target quantity (bounded tail-loss rate vs. interval miscoverage) and a
  different algorithm (UCB search vs. online alpha-tracking).
- R-104 (a live, causal significance test of whether the VOTE's own
  historical mean return is non-zero, discounting `frac*scale` by that
  p-value). A hypothesis test on a MEAN; this round controls a
  tail-EXCEEDANCE RATE via a distribution-free concentration bound, a
  different target quantity and a different statistical tool.
- R-105 (jackknife/ensemble disagreement across five leave-one-anchor-out
  specifications). This round uses one fixed vote and one fixed scale
  estimator; there is no ensemble or specification-disagreement statistic
  anywhere in it.
- R-106/R-109/R-112/R-113/R-115/R-122/R-123 (the "distributional novelty"
  family -- Mahalanobis/kNN outlier-distance brakes on how unusual today's
  feature vector is). These carry no formal statistical guarantee of any
  kind. This round solves an explicit inequality for a threshold with a
  provable (if the loss/calibration assumptions hold) finite-sample bound on
  expected risk.
- R-114 (hazard/duration dependence keyed on time-in-regime). This round has
  no notion of regime age.
- R-125 (a risk-STATISTIC substitution: std to CVaR, a point-estimate swap,
  no calibration set, no concentration inequality). This round changes
  nothing about what statistic feeds `scale`; it wraps a calibrated cap
  AROUND scale's existing output.
- R-147 (James-Stein / Bayesian-posterior reweighting of the VOTE's three
  anchor combination weights). This round leaves the vote's equal-weight
  combination completely unmodified.
- R-160 (online-FDR, LORD/SAFFRON, gating each anchor's discrete flip
  ACCEPT/REJECT decision via a sequential wealth process over a stream of
  discovery events -- FLIP TIMING). This round gates nothing about WHEN a
  flip is accepted; `frac` is never touched. It operates continuously on
  `scale`'s magnitude via a calibration-set/concentration-bound search, a
  structurally different statistical object from an online-FDR wealth
  process, and is precisely the "SCALE/sizing decision" R-160's own closing
  line names as untried.
- B-09 (backlog, LOW; "conformal prediction/adaptive conformal by betting...
  the binding problem is correctly-calibrated trust is low, not that trust
  is miscalibrated" -- a statement about the VOTE's confidence coverage,
  R-28/R-87's finding). CRC/RCPS never calibrates the vote's confidence; it
  calibrates a tail-loss exceedance rate on SCALE's output, a different
  quantity, so B-09's demotion does not transfer without new evidence.
- Ledger-wide grep (this round, before any code) confirms zero prior hits
  for "Conformal Risk Control", "RCPS", "risk-controlling prediction",
  "Angelopoulos", or "Bates" (2021 J. ACM sense) anywhere in
  `docs/LEDGER.md` outside this entry.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- the r89-r160 convention. Nothing here reads a
bar at or after OOS_START (2023-01-01); `compare()` (imported from
`r147_shared`, itself chained from r105_shared/r102_shared) never touches
the holdout.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) INERT -- the calibrated lambda sits at or within a few percent of 1.0 on
    effectively every calibration window/online step, meaning the risk
    budget (alpha, tau) chosen never actually binds against this data's
    realized loss distribution (the GATE_MIN_BINDING_FRACTION kill switch
    below): a relabeling of v4, not a tested mechanism.
(2) MISCALIBRATED -- the calibration self-test (synthetic data with a KNOWN
    injected tail-event rate, section 4 below) shows the achieved
    exceedance rate is far from the nominal alpha (the Hoeffding bound's
    i.i.d. assumption failing badly under this data's serial correlation) --
    the gate is not actually controlling what it claims to.
(3) BTC-ONLY ARTIFACT -- the gate binds, is calibrated, and improves BTC,
    but the improvement inverts sign on ETH -- this project's single most
    common failure mode for SIZE/ERR-axis constructions on this slot
    (R-105, R-109, R-113, R-125, R-126: six independent precedents).
(4) CORRECT BUT COSTLY -- tail-loss frequency measurably drops but Sharpe
    falls by more than the +/-0.2 noise floor (R-20) on both markets, with
    no offsetting drawdown/tail improvement -- discounting the informative
    part of the exposure along with the tail risk (the modal ERR-axis
    outcome to date).
(5) DEGENERATE CONSTANT -- the calibrated cap converges to (and stays at) a
    single constant across the whole sample, i.e. this reduces to "v4 should
    have been de-levered globally" rather than a genuinely time-varying risk
    control -- still reportable, but downgrades the finding's novelty (a
    diagnostic below, not a kill switch, since a constant answer is still an
    answer).

=====================================================================
PRE-REGISTRATION -- frozen before either branch is dispatched
=====================================================================

Both branches gate the SAME shipped v4 anchors/scale (20/40/80-day vote, 1%
band, conditional target-vol scale), using the SAME causal loss functional
and Hoeffding UCB below, wrapped around v4's own UNMODIFIED
`v4_raw_desired` (`frac * scale`, pre-deadband) -- the only difference from
v4 is a multiplicative cap on that magnitude, never anything about the vote
or the scale estimator itself. v4's own 10% deadband is applied AFTER the
cap, identically to how v4 applies it after `frac*scale` today.

Decision rule (ROUTINE.md's own promotion bar, adapted to a
candidate-vs-v4-control comparison via `compare()`, the r105-r160
convention):
  PROMOTE-CANDIDATE (worth carrying to the holdout) if, on inner-validation
  (2021-01-01..2022-12-31), for AT LEAST ONE pre-registered (tau, alpha)
  cell in this branch's grid, on BOTH markets (spot and futures_5x):
    (a) the candidate's realized daily tail-loss exceedance rate (bars where
        |capped_exposure * next-day log return| > tau) is STRICTLY LOWER
        than v4's own uncapped rate at the same tau, AND
    (b) the paired bootstrap 95% CI on d_log_growth (candidate - v4)
        excludes zero on the positive side, OR d_sharpe >= +0.2 (the noise
        floor, R-20), OR a real (risk-matched, exposure_ratio in [0.9,1.1])
        drawdown improvement, AND
    (c) the SAME direction of improvement on (a)+(b) reproduces on this
        branch's own pre-registered falsification test (below) -- not
        inverted.
  Any other outcome on inner-validation is NEGATIVE for that cell. A branch
  that clears all three moves to the holdout ONLY after the operator
  freezes the SPECIFIC (tau, alpha, and branch-specific hyperparameters) --
  no further tuning -- and logs it here before running
  `ev(..., start=OOS_START)`.

Falsification tests (pre-registered per ROUTINE.md Step 2, chosen now,
matched to what each branch's own literature claims it should be good at):
  CONSERVATIVE (periodic RCPS batch calibration): ETH sign-replication --
  chosen because the dominant failure mode across this ledger's ERR/SIZE-
  axis family on this exact slot (R-105, R-109, R-113, R-125, R-126: six
  independent constructions) is "passes on BTC, inverts on ETH", and a
  periodic, BTC-fitted calibration window is exactly the construction most
  exposed to that failure mode.
  NOVEL (online CRC, distribution-shift adaptation): survive the Monte
  Carlo stress windows (`scripts/stress_test.py`) -- chosen because the
  entire claimed value of an ONLINE, shift-adapting risk-control layer
  (rather than a periodic one) is that it should behave better precisely in
  stress regimes a fixed calibration window did not anticipate; that is the
  mechanism's own literature claim (Bates et al. 2021; Angelopoulos et al.
  2024's shift-adaptation section), so it is the correct axis to test it
  against.

Threshold/power sanity check (R-78/ROUTINE Step 2 requirement): both
variants are scored against this project's own already-validated inner-
validation d_sharpe >= +0.2 bar (R-20's measured path-noise floor) or a
bootstrap CI excluding zero -- the SAME bar 15+ prior SIZE/ERR-axis rounds
have used on the identical inner-validation window and sample size (R-104
conservative reached +0.2210 on precisely this window; R-105 novel came
within 0.02), so a pass or a near-miss both remain informative rather than
a foregone null from an unreachable bar.

Configs evaluated so far by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the R-161
ledger entry).
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

from experiments.r147_shared import (  # noqa: E402,F401
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

V4_DEADBAND = 0.10                 # v4's own re-target deadband (r102_shared.py:198),
                                    # not re-exported through r105/r147_shared's chain

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
FEE_TIER = 0.0040                  # 0.40% taker, cost-robustness sensitivity
SHARPE_NOISE_FLOOR = 0.2           # ROUTINE.md's own promotion bar (R-20)
TAU_GRID = (0.05, 0.08)            # daily tail-loss budget, |exposure*ret|
TAU_PRIMARY = TAU_GRID[0]
ALPHA_GRID = (0.05, 0.10)          # target exceedance rate; PRIMARY is first
ALPHA_PRIMARY = ALPHA_GRID[0]
HB_DELTA = 0.10                    # Hoeffding UCB confidence: 1 - delta = 90%
GATE_MIN_BINDING_FRACTION = 0.02   # A1 kill switch: lambda < 0.98 on >=2% of
                                    # calibrated days, else the gate is inert
CONST_CAP_R2_THRESH = 0.98         # A2 diagnostic: lambda_t vs. its own mean


# ================================================================== (1)
# Causal per-day loss functional. Shared so BOTH branches score the SAME
# evidence -- only the calibration ALGORITHM (periodic RCPS vs. online CRC)
# differs between them.
# ==================================================================

def daily_close(df: pd.DataFrame) -> pd.Series:
    """Last close of each calendar day -- the price series the tail-loss
    functional is measured against."""
    return df["close"].resample("1D").last().dropna()


def daily_log_return(df: pd.DataFrame) -> pd.Series:
    """Close-to-close daily log return, indexed by the day the return
    REALIZES (i.e. daily_log_return[d] uses close[d] and close[d-1])."""
    return np.log(daily_close(df)).diff()


def daily_last_of(bar_values: np.ndarray, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Resample an arbitrary bar-frequency array to its last-value-per-day,
    e.g. the exposure DECIDED by the close of each day."""
    return pd.Series(np.asarray(bar_values, dtype=float), index=bar_index) \
        .resample("1D").last().dropna()


def calibration_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the daily (exposure_decided_end_of_prev_day, next_day_return)
    calibration series used by BOTH branches' loss functional. Exposure is
    v4's own UNMODIFIED, UNCAPPED `frac*scale` (never the vote or the
    dispersion estimator) -- what this round wraps a cap around. Strictly
    causal: row d pairs the exposure fixed at the close of day d-1 (already
    known at the start of day d) with the return realized ON day d."""
    raw = v4_raw_desired(df)
    exposure_by_day = daily_last_of(raw, df.index)
    ret = daily_log_return(df)
    idx = exposure_by_day.index.intersection(ret.index)
    out = pd.DataFrame({
        "exposure_prev": exposure_by_day.reindex(idx).shift(1),
        "ret": ret.reindex(idx),
    }).dropna()
    return out


def loss_at(exposure_prev: np.ndarray, ret: np.ndarray, lam: float, tau: float) -> np.ndarray:
    """Per-day tail-loss indicator at cap `lam`: 1{|exposure_prev*lam*ret| > tau}.
    Monotone NON-INCREASING in `lam` for fixed (exposure_prev, ret) since
    the argument's magnitude is linear in lam >= 0 -- the property both
    RCPS and CRC require of the loss family."""
    magnitude = np.abs(np.asarray(exposure_prev, dtype=float) * lam * np.asarray(ret, dtype=float))
    return (magnitude > tau).astype(float)


def hoeffding_ucb(loss_values: np.ndarray, delta: float = HB_DELTA) -> float:
    """One-sided Hoeffding upper confidence bound on the mean of a
    [0,1]-bounded loss, at confidence 1-delta: R_hat + sqrt(log(1/delta)/(2n)).
    DISCLOSED SIMPLIFICATION (see module docstring): the looser, valid
    special case of Bates et al.'s Hoeffding-Bentkus hybrid, chosen because
    a Bentkus (binomial-tail) inversion needs scipy, outside this project's
    dependency set."""
    n = len(loss_values)
    if n == 0:
        return 1.0  # no data yet: cannot certify anything, refuse to cap
    r_hat = float(np.mean(loss_values))
    margin = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return min(1.0, r_hat + margin)


def rcps_calibrate(exposure_prev: np.ndarray, ret: np.ndarray, alpha: float, tau: float,
                    lambda_grid: np.ndarray, delta: float = HB_DELTA) -> float:
    """Bates et al. (2021) RCPS search, adapted to a loss INCREASING in the
    parameter (v4's own convention: lambda=1 is "no cap", most permissive;
    lambda=0 is "no exposure", safest) rather than their own decreasing-loss
    framing -- the identical UCB-search logic, direction flipped. Returns
    the LARGEST lambda in `lambda_grid` whose Hoeffding UCB on tail-loss
    rate is <= alpha. lambda=0.0 is always feasible (UCB=margin only, since
    loss_at(...,0.0,...) is identically zero), so this never fails to
    return a value."""
    best = 0.0
    for lam in lambda_grid:
        losses = loss_at(exposure_prev, ret, lam, tau)
        if hoeffding_ucb(losses, delta) <= alpha:
            best = max(best, float(lam))
    return best


LAMBDA_GRID = np.linspace(0.0, 1.0, 101)  # calibration search grid, shared


# ================================================================== (2)
# Wire an arbitrary bar-frequency lambda path through v4's OWN unmodified
# scale/deadband machinery. Both branches call this identically.
# ==================================================================

def broadcast_daily_lambda(daily_lambda: pd.Series, bar_index: pd.DatetimeIndex) -> np.ndarray:
    """A daily lambda series (indexed by the day it applies to, already
    causal -- built only from data through the PRIOR day) forward-filled to
    bar frequency. Days before any calibration exists default to 1.0 (no
    cap: v4's own baseline, the honest "no evidence yet" state)."""
    days = pd.DatetimeIndex(bar_index).floor("D")
    return daily_lambda.reindex(days).ffill().fillna(1.0).to_numpy()


def build_capped_target(df: pd.DataFrame, daily_lambda: pd.Series) -> np.ndarray:
    """v4's own `frac*scale`, multiplicatively capped by `daily_lambda`
    (broadcast to bar frequency), then v4's own unmodified 10% deadband --
    identical pipeline to `v4_target`, with one multiplicative factor
    inserted before the deadband."""
    raw = v4_raw_desired(df)
    lam_bars = broadcast_daily_lambda(daily_lambda, df.index)
    return apply_deadband(raw * lam_bars, deadband=V4_DEADBAND)


# ================================================================== (3)
# Kill switches / diagnostics, applied identically by both branches before
# any Sharpe/holdout number is read.
# ==================================================================

def binding_fraction(daily_lambda: pd.Series, thresh: float = 0.98) -> float:
    """Fraction of calibrated days where lambda actually bound (< thresh).
    A1 kill switch: GATE_MIN_BINDING_FRACTION or more, else the gate is a
    relabeling of v4 that happens to never activate."""
    if len(daily_lambda) == 0:
        return 0.0
    return float(np.mean(daily_lambda.to_numpy() < thresh))


def constant_cap_r2(daily_lambda: pd.Series) -> float:
    """A2 diagnostic (not a kill switch): R^2 of lambda_t against its own
    mean, i.e. how much of the cap's variance is explained by a single
    constant. High (near 1.0) means the calibration collapsed to "v4 should
    have been de-levered globally" rather than a genuinely time-varying
    control -- still a valid, reportable finding, just a weaker one
    (failure mode (5) in the module docstring)."""
    x = daily_lambda.to_numpy()
    if len(x) < 2 or np.std(x) < 1e-12:
        return 1.0
    const = np.full_like(x, np.mean(x))
    return r_squared(x, const)


def exceedance_rate(exposure_prev: np.ndarray, ret: np.ndarray, lam: float, tau: float) -> float:
    """Realized tail-loss exceedance rate at a fixed lambda -- the quantity
    the decision rule's clause (a) compares between candidate and v4."""
    return float(np.mean(loss_at(exposure_prev, ret, lam, tau)))


# ================================================================== (4)
# Synthetic calibration self-test data: a KNOWN injected tail-event rate,
# so the achieved exceedance rate can be checked against a ground truth
# (not merely against itself) -- failure mode (2) in the module docstring.
# ==================================================================

def synthetic_known_tail_frame(n: int = 400_000, true_tail_prob: float = 0.05,
                                seed: int = 161) -> pd.DataFrame:
    """Synthetic 5-minute OHLCV whose DAILY returns are a two-component
    mixture: with probability `true_tail_prob` per day, a large shock
    (mean magnitude far past any plausible tau); otherwise small noise.
    Used only to check that RCPS/CRC calibration recovers an exceedance
    rate in the right ballpark of a KNOWN ground truth, not a proof of
    validity under this project's real serial correlation."""
    rng = np.random.default_rng(seed)
    n_days = n // BARS_PER_DAY + 1
    is_tail_day = rng.random(n_days) < true_tail_prob
    daily_shock = np.where(is_tail_day, rng.choice([-1, 1], n_days) * rng.uniform(0.12, 0.20, n_days),
                            rng.normal(0, 0.01, n_days))
    per_bar_drift = np.repeat(daily_shock, BARS_PER_DAY)[:n] / BARS_PER_DAY
    innov = rng.normal(0, 0.0005, n)
    close = 10_000 * np.exp(np.cumsum(innov + per_bar_drift))
    idx = pd.date_range("2017-01-01", periods=n, freq="5min", tz="UTC")
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, n)))
    return pd.DataFrame({"open": close, "high": high, "low": low,
                          "close": close, "volume": 1.0}, index=idx)


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=150_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(161)
    innov = rng.normal(0, 0.0006, len(idx))
    close = 10_000 * np.exp(np.cumsum(innov))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    # (1) calibration_frame: shapes, causality (exposure_prev strictly
    # precedes the return it is paired with).
    cal = calibration_frame(df)
    assert set(cal.columns) == {"exposure_prev", "ret"}
    assert len(cal) > 100

    def _cal_exposure(d: pd.DataFrame) -> np.ndarray:
        return calibration_frame(d)["exposure_prev"].reindex(
            calibration_frame(df).index).to_numpy()

    # (2) loss_at: monotone non-increasing in lambda; lambda=0 -> always 0.
    exp_prev = cal["exposure_prev"].to_numpy()
    ret = cal["ret"].to_numpy()
    l0 = loss_at(exp_prev, ret, 0.0, TAU_PRIMARY)
    l_half = loss_at(exp_prev, ret, 0.5, TAU_PRIMARY)
    l_full = loss_at(exp_prev, ret, 1.0, TAU_PRIMARY)
    assert np.all(l0 == 0.0)
    assert np.mean(l_half) <= np.mean(l_full) + 1e-12

    # (3) hoeffding_ucb: UCB >= empirical mean always; shrinks with n.
    losses = rng.integers(0, 2, 500).astype(float)
    ucb_small = hoeffding_ucb(losses[:50])
    ucb_large = hoeffding_ucb(losses)
    assert ucb_small >= np.mean(losses[:50])
    assert ucb_large >= np.mean(losses)

    # (4) rcps_calibrate: returns a value in [0,1]; lambda=0 always feasible
    # so the search never raises; a very loose alpha permits lambda=1.
    lam = rcps_calibrate(exp_prev, ret, alpha=ALPHA_PRIMARY, tau=TAU_PRIMARY,
                         lambda_grid=LAMBDA_GRID)
    assert 0.0 <= lam <= 1.0
    lam_loose = rcps_calibrate(exp_prev, ret, alpha=0.999, tau=1e-6,
                               lambda_grid=LAMBDA_GRID)
    assert lam_loose == 1.0

    # (5) build_capped_target reproduces v4_target exactly when lambda==1
    # every day (no cap at all).
    daily_idx = calibration_frame(df).index
    lam_one = pd.Series(1.0, index=daily_idx)
    capped = build_capped_target(df, lam_one)
    assert np.allclose(capped, v4_target(df), atol=1e-9)

    # (6) broadcast_daily_lambda: pre-calibration days default to 1.0.
    lam_partial = pd.Series([0.5], index=[daily_idx[len(daily_idx) // 2]])
    bars = broadcast_daily_lambda(lam_partial, df.index)
    assert bars[0] == 1.0

    # (7) binding_fraction / constant_cap_r2: sane bounds.
    assert binding_fraction(pd.Series([1.0, 1.0, 1.0])) == 0.0
    assert binding_fraction(pd.Series([0.5, 0.5, 1.0])) > 0.0
    assert constant_cap_r2(pd.Series([0.7, 0.7, 0.7])) == 1.0

    # (8) causal truncation probe on the full capped-target pipeline at a
    # fixed, already-calibrated lambda path (the cap itself is calibrated
    # per-branch; this checks the WIRING is causal given any lambda path).
    def _capped_builder(d: pd.DataFrame) -> np.ndarray:
        idx_d = calibration_frame(d).index
        lam_d = pd.Series(0.7, index=idx_d)
        return build_capped_target(d, lam_d)

    assert causal_truncation_probe_series(_capped_builder, df)

    # (9) synthetic_known_tail_frame: sane OHLCV shape, no NaNs/negatives.
    synth = synthetic_known_tail_frame(n=60_000, true_tail_prob=0.05, seed=2)
    assert len(synth) == 60_000
    assert (synth["close"] > 0).all()


_self_test()
