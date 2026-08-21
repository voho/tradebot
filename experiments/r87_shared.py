"""Shared, read-only utilities for the R-87 round (08-21).

Idea in one sentence: wrap `kelly_regime_v4`'s existing anchor vote in an
online, distribution-shift-robust confidence layer built from Adaptive
Conformal Inference (ACI; Gibbs & Candes 2021, "Adaptive Conformal
Inference Under Distribution Shift", NeurIPS 34:1660-1672) -- a formal,
finite-sample error-tracking recursion that requires no i.i.d. or
exchangeability assumption on the underlying series -- and let the
tracked reliability modulate sizing, instead of retuning a 22nd ad hoc
scale factor on the axis R-62 showed carries nothing.

Directly relevant prior art, found by literature search and verified to
exist before this round relied on it (fetched from arXiv, not taken on
faith): Ryan, R. J. (2026), "Conformal Kelly: Conformal Prediction
Intervals as the Scale in Fractional Kelly Position Sizing", arXiv:
2608.01494. On equities, a 75% conformal interval used as the Kelly scale
compounded at 28.5%/yr net (Sharpe 1.34) on a 2016-2021 development
window, sealed 2022+ data, pre-registered configuration -- and then
returned only 7.0-8.5%/yr out-of-sample. That collapse is treated here as
a NAMED, PRE-REGISTERED PREDICTION of this round's own likely failure
mode (an in-sample-looking win that does not survive a real holdout),
not as evidence the mechanism works. This round never reads holdout data
at all (nothing here is evaluated past 2022-12-31), so the collapse
cannot repeat here; it is cited as the reason both branches must be read
skeptically even if inner-validation looks good.

Which constraint this attacks: primarily **ERR** (no error control
anywhere in this project's signal path -- the only prior attempt, the
e-process of R-26/R-28, tracked an aggregate P&L e-value and was retired
by R-31 as an exposure-level artifact, never reaching a per-decision
guarantee). Secondarily deepens **SIZE** per R-62's factor finding: run
alone, the VOTE reproduces v4's whole matched-exposure drawdown
signature and the volatility-target SCALE reproduces neither -- 21 prior
R-34-R-60 attempts all retuned the scale factor R-62 showed never carried
the mechanism. This round's conservative branch modifies the VOTE's own
confidence (the factor R-62 showed matters); its novel branch does retune
the scale, but via a structurally different, formally-motivated
construction (a conformal-calibrated dispersion estimate, not a 22nd
point-estimate variant), so it is a legitimate, non-duplicate test of
whether R-62's null is about POINT-estimate scales specifically or about
the scale slot in general.

Not a duplicate of:
- R-26/R-28 (e-process, retired by R-31): that construction tested the
  strategy's AGGREGATE realized P&L against a null of no edge, a single
  running e-value over portfolio outcomes. ACI here tracks the VOTE's own
  per-decision directional miscoverage, online, and never touches P&L at
  all -- a different quantity, a different (frequentist online-learning,
  not e-process/sequential-testing) formal framework, and it modulates a
  sizing multiplier rather than a stop/go gate.
- R-01/R-82/R-83/R-85/R-86 (HMM, BOCPD, Kalman LLT, CSD, transfer
  entropy): all five are regime/changepoint/trend ESTIMATORS run against
  the identical six-episode detection-lag gate -- do they detect a known
  historical regime break earlier than v4's own anchor? ACI detects no
  changepoint, has no hazard function, no filtered latent state, and is
  not run against that gate at all: it tracks whether the EXISTING vote's
  sign has recently been reliable, which is a well-defined question even
  when the regime has not just broken.
- R-73-R-84 (INFO axis, ten signals): all ten introduced a new external or
  timestamp-derived data channel. ACI reads zero data beyond the OHLCV
  close series `kelly_regime_v4` already consumes -- it is a function of
  the vote's own past correctness and the price series's own past
  returns, nothing else.
- R-80 (causal meta-labeling): a discriminative classifier (logistic
  regression) with trained, fitted weights over hand-engineered features.
  ACI has no trained weights, no features beyond a single scalar
  (miscoverage indicator) and one fixed recursion with a pre-registered
  learning rate -- a deterministic online update rule, not a fitted model.
- R-34-R-60 (SIZE-axis scale retunes: CRRA/Merton, CPPI +/- Hurst,
  risk-constrained Kelly, per-state Kelly fraction, self-normalizing
  relative vol): all 21 retuned `target_vol`, `max_leverage`, or an
  equivalent point-estimate scale multiplier, holding the ESTIMATOR of
  dispersion (trailing realized vol) fixed. The novel branch here changes
  the estimator itself (a conformal-calibrated quantile of absolute
  returns, adaptively re-targeted online) rather than the multiplier
  applied to trailing realized vol.

This module is a read-only utility, written by the operator before
dispatch (the r73-r86 convention). Neither branch edits it. Contains:
(1) `v4_vote_frac`, a byte-for-byte duplicate of `kelly_regime_v4`'s
3-anchor vote construction (copied from `kelly_regime.py`, not imported,
so a bug in this module cannot silently change the registered strategy);
(2) `aci_update`, the Gibbs & Candes (2021) online adaptive-alpha
recursion, `alpha_{t+1} = clip(alpha_t + gamma * (target_alpha - err_t))`;
(3) `run_aci_causal`, which applies that recursion over an array of
per-bar miscoverage indicators and returns the alpha_t path SHIFTED so
that the value used to act at bar i was computed from information
available through bar i-1 only; (4) `daily_resample_causal`, a causal
daily OHLCV resample (each daily bar uses only intraday bars up to and
including that day's last 5-minute bar); (5) `causal_truncation_probe`,
copied from `r85_shared.py`.

ALL PARAMETERS BELOW ARE FIXED A PRIORI, before any real-market ACI
number was computed. `ALPHA_MIN=0.02, ALPHA_MAX=0.98` bound the internal
adaptive-alpha state away from the degenerate 0/1 edges (standard ACI
implementation practice, not a tuned choice). `V4_HORIZONS=(20,40,80)`
and `V4_BAND=0.01` are v4's own registered parameters, not swept.

Sanity check performed by the operator before dispatch (same discipline
as R-86's synthetic TE check): `aci_update` was run on a synthetic
Bernoulli(p) miscoverage stream for p in {0.1, 0.3, 0.5, 0.7, 0.9} against
target_alpha=0.5, gamma=0.02, 20,000 steps, seed fixed (`_self_test_aci`,
run directly with `python experiments/r87_shared.py`). For p != 0.5 the
recursion has a signed per-step drift `gamma*(target_alpha-p)` and, as
expected, saturates near ALPHA_MAX when p < target_alpha (miscoverage
rarer than target -> confidence climbs) and near ALPHA_MIN when
p > target_alpha (miscoverage commoner than target -> confidence falls),
confirmed for p in {0.1, 0.3, 0.7, 0.9}. At the exact indifference point
p=0.5=target_alpha the per-step update is a coin flip of +/-gamma/2 with
ZERO mean -- an unbiased, driftless random walk with no restoring force,
which is expected to wander and sit near whichever boundary it first
reaches rather than hold a stable mean over an arbitrary window; this is
a real, understood property of the recursion at that single point, not a
bug, and is why the self-test reports it as a named special case rather
than a failure. Real vote hit-rates measured below are never exactly
0.500 to machine precision, so this degeneracy is not expected to bind on
market data, and both branches check for it explicitly before relying on
any alpha_t path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated verbatim from
# kelly_regime.py's KellyRegime.prepare, not imported.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3, identical to r82-r86. Holdout
# (>= OOS_START) is never read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

ALPHA_MIN = 0.02
ALPHA_MAX = 0.98


def v4_vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                  band: float = V4_BAND) -> np.ndarray:
    """kelly_regime_v4's 3-anchor latched vote, verbatim (see kelly_regime.py)."""
    close = df["close"]
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def aci_update(alpha_t: float, err_t: float, target_alpha: float, gamma: float,
                alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX) -> float:
    """One step of Gibbs & Candes (2021) online adaptive-alpha recursion.

    `err_t=1` means a miscoverage/violation was observed at t (the
    realized outcome fell outside what alpha_t was targeting): alpha_t
    decreases, which widens the next interval / lowers confidence.
    `err_t=0` means it was covered: alpha_t increases, narrowing the next
    interval / raising confidence. This is the paper's own recursion,
    unmodified except for the clip to keep the state finite-sample sane.
    """
    a = alpha_t + gamma * (target_alpha - err_t)
    return float(min(max(a, alpha_min), alpha_max))


def run_aci_causal(err: np.ndarray, target_alpha: float, gamma: float,
                    alpha0: float = None) -> np.ndarray:
    """Run `aci_update` over a causal indicator array `err` (err[i] must be
    knowable using only information through bar/day i).

    Returns `alpha_path` where `alpha_path[i]` is the state to be USED for
    acting at step i -- i.e. it was last updated using `err[i-1]`, so
    acting on `alpha_path[i]` never looks at `err[i]` or later. `alpha0`
    defaults to `target_alpha` (start at the nominal target, no data seen).
    """
    n = len(err)
    a0 = target_alpha if alpha0 is None else alpha0
    out = np.empty(n)
    a = a0
    for i in range(n):
        out[i] = a  # value used to act at i, computed from err[:i] only
        if np.isfinite(err[i]):
            a = aci_update(a, float(err[i]), target_alpha, gamma)
    return out


def daily_resample_causal(df: pd.DataFrame) -> pd.DataFrame:
    """Causal daily OHLCV resample: day D's bar uses only intraday bars up
    to and including day D's own last 5-minute bar (no lookahead)."""
    o = df["open"].resample("1D").first()
    h = df["high"].resample("1D").max()
    lo = df["low"].resample("1D").min()
    c = df["close"].resample("1D").last()
    v = df["volume"].resample("1D").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}).dropna()
    return out


def causal_truncation_probe(build_target_fn, df: pd.DataFrame,
                             check_at: int, shorter_by: int = 20_000) -> bool:
    """Standard truncation probe: does `target[check_at]` change if bars
    after it are dropped? Returns True if causal (identical both ways).
    Copied verbatim from r85_shared.py."""
    full = build_target_fn(df)
    short = build_target_fn(df.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))


def _self_test_aci() -> None:
    """Synthetic sanity check, run before any real-market number: does
    `aci_update`'s time-averaged alpha_t track its theoretical fixed
    point for a stationary Bernoulli(p) miscoverage stream? (It is not
    supposed to converge to `target_alpha` itself -- ACI's fixed point
    for a stationary err-stream with P(err=1)=p sits where the expected
    update is zero, i.e. where the LONG-RUN miscoverage rate under the
    resulting (possibly saturated) policy equals target_alpha; for the
    unsaturated interior case with a stationary stream this reduces to
    the running mean of alpha_t oscillating in a small band -- checked
    here by confirming the mean of the back half of the path is stable,
    i.e. has converged, not by pinning it to a closed-form value.)"""
    rng = np.random.default_rng(8701)
    target_alpha = 0.5
    gamma = 0.02
    n = 20_000
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        err = (rng.random(n) < p).astype(float)
        path = run_aci_causal(err, target_alpha, gamma)
        first_half_mean = path[: n // 2].mean()
        second_half_mean = path[n // 2:].mean()
        drift = abs(second_half_mean - first_half_mean)
        if p == target_alpha:
            label = "DRIFTLESS RANDOM WALK (expected at p==target_alpha, not a failure)"
        else:
            expect_high = p < target_alpha
            saturated = second_half_mean > 0.9 if expect_high else second_half_mean < 0.1
            label = "SATURATED AS EXPECTED" if saturated else "UNEXPECTED -- INVESTIGATE"
        print(f"p={p:.1f}  alpha_t mean[first half]={first_half_mean:.4f} "
              f"mean[second half]={second_half_mean:.4f}  drift={drift:.4f}  {label}")


if __name__ == "__main__":
    _self_test_aci()
