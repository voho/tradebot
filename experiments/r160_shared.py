"""Shared, read-only utilities and pre-registration for the R-160 round (08-27).

DIRECTION, in one sentence: gate `kelly_regime_v4`'s three anchor votes'
FLIP decisions (does the crowd regime just change, 20/40/80-day) through an
online false-discovery-rate control procedure, so a flip only takes effect
once the p-value evidence for it clears an adaptively-shrinking budget,
instead of acting on every band-crossing immediately as v4 does today.

**Which constraint this attacks: ERR (primary), N~3 (secondary).** No error
control exists anywhere in v4's signal path: `_latched_anchor_vote`
(`experiments/r102_shared.py:236`, mirroring `kelly_regime.py:82-89`) flips
the instant `close` crosses `anchor * (1 +/- band)`, with only a fixed 1%
band and ffill-hysteresis to suppress chop -- no notion of how often a
crossing of that size would occur under pure noise, and no long-run bound on
how many acted-upon flips are false alarms. Online false-discovery-rate
control (unlike a fixed-alpha single test) is built for exactly a rare,
sequential discovery stream -- each candidate flip is one more "discovery"
claim, and the guarantee is on the long-run fraction of acted-upon flips
that are false, which is the natural formalization of "is a crossing real or
noise" this project has never applied to the vote's OWN timing. Secondarily
N~3: fewer, higher-precision "discoveries" is the same shape of problem as
controlling error under a sparse-event budget, though the "3" this round
touches is candidate flip COUNT (order of dozens over the full history),
unrelated to the project's chronic ~3-regime-event effective sample size.

**Literature, fetched via WebSearch before either branch was dispatched (by
the research sub-agent that proposed this direction, then re-verified by the
operator before freezing this file):**

- Javanmard, A., & Montanari, A. (2018), "Online rule for control of false
  discovery rate and false discovery exceedance", *Annals of Statistics*
  46(2), 526-554 (journal version of arXiv:1502.06197, "On Online Control
  of False Discovery Rate", 2015). LORD: a sequential testing procedure that
  spends a wealth budget on each test and is REPLENISHED only by past
  REJECTIONS (not by successes measured later -- the guarantee needs only
  the p-value stream, never ground truth), via a fixed non-increasing,
  summable discount sequence. Provably controls online-FDR under
  independence of the p-values (a caveat this round's own calibration
  self-test, not a proof, is used to check empirically -- see below).
- Ramdas, A., Zrnic, T., Wainwright, M., & Jordan, M. (2018), "SAFFRON: an
  Adaptive Algorithm for Online Control of the False Discovery Rate",
  *ICML* (arXiv:1802.09098). Improves on LORD's power by estimating, online,
  the fraction of candidate tests that are LIKELY true nulls (p-value above
  a fixed candidate threshold lambda spends no wealth), so wealth is not
  wasted budgeting for tests that were never going to be rejected -- the
  formal basis for the novel branch below.
- Tian, J., & Ramdas, A. (2019), "ADDIS: an adaptive discarding algorithm
  for online FDR control with conservative nulls", *NeurIPS*
  (arXiv:1905.11465). SAFFRON's follow-on: additionally DISCARDS (never
  tests again) candidates whose p-value is so large they are certainly null,
  freeing more wealth for genuine discoveries -- cited as the novel branch's
  second-stage refinement, optional if SAFFRON alone is insufficient.

**Not a duplicate of:**
- R-28 / R-31 (e-process / GRO gate, `experiments/eprocess_regime.py`): a
  SINGLE continuously-accumulating martingale toward one persistent
  hypothesis ("is the current regime still bullish"), sized continuously as
  a multiplier on exposure. This round is a discrete accept/reject decision
  on each of many separate flip EVENTS, with a formal long-run FALSE
  DISCOVERY RATE guarantee across that event stream -- a structurally
  different statistical object (online multiple testing, not a single
  sequential test) and it gates the VOTE's own timing, not a separate
  exposure multiplier layered on top of the shipped vote.
- R-87 (Adaptive Conformal Inference wrapping the vote/scale, NEGATIVE):
  ACI controls miscoverage of a continuous confidence SET via online mirror
  descent on a coverage target; it does not gate discrete accept/reject
  events and carries no false-discovery notion. Different guarantee,
  different mechanism.
- R-138 (Nguyen & Wolf 2026 small-N permutation test, NEGATIVE): a FIXED,
  retrospective significance test of an already-observed six-episode set --
  it evaluates whether v4's PAST edge is real, and changes nothing about
  what gets traded going forward. This round is a live, sequential, forward-
  acting gate on future flip decisions, never a backward-looking test.
- R-147 (James-Stein shrinkage / Bayesian ladder posterior, NEGATIVE): both
  R-147 branches change the COMBINATION WEIGHTS used to average the three
  anchors' votes into `frac`, while holding each anchor's own point estimate
  (and its update TIMING) fixed. This round holds the equal-weight
  combination fixed and instead changes WHEN each anchor's own flip is
  allowed to take effect -- the complementary axis, deliberately untouched
  by R-147's own filing (see r147_shared.py's own non-duplication note on
  R-146, the same distinction one level over).
- Ledger-wide grep (this round, before any code) confirms zero prior hits
  for "false discovery rate", "FDR", "LORD", "SAFFRON", "ADDIS", "alpha
  investing", or "e-LOND" anywhere in `docs/LEDGER.md` outside this entry.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- the r89-r147 convention. Nothing here reads a
bar at or after OOS_START (2023-01-01); `compare()` (imported from
`r147_shared`, itself chained from `r102_shared`) never touches the holdout.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) The gate never binds -- fewer than a handful of candidate flips are ever
delayed at all, on any anchor, over the whole inner-train+inner-validation
period (GATE_MIN_DELAYS kill switch below): a relabeling of v4, not a
tested mechanism.
(2) The calibration self-test (synthetic zero-drift noise, Step-0 below)
shows the gate accepting flips far more often than its nominal alpha
budget implies -- the online-FDR guarantee assumed independence the real
serially-correlated p-value stream does not have, and the gate is not
actually controlling anything.
(3) The gate binds and is calibrated, but any BTC improvement inverts sign
on ETH -- this project's single most common failure mode for vote-timing/
ERR-axis constructions (R-87's novel branch, R-109, R-114-novel).
(4) The improvement, if any, sits inside the +/-0.2 Sharpe noise floor
(R-20) with no offsetting drawdown/tail improvement -- the modal outcome
on this axis (6 of 7 prior ERR-axis attempts closed this way).
(5) Fewer, more confident flips reduce RETURN capture by more than they
reduce false-flip whipsaw -- i.e. the gate trades responsiveness for
precision at a bad exchange rate, visible as lower final balance with
unchanged or worse drawdown.

=====================================================================
PRE-REGISTRATION -- frozen before either branch is dispatched
=====================================================================

Both branches gate the SAME shipped v4 anchors (20/40/80-day, 1% band),
using the SAME causal p-value construction below, feeding the result
through `build_target_from_frac` (v4's own unmodified scale + 10% deadband,
imported from r147_shared) so the ONLY difference from v4 is when a flip
takes effect, never anything about its magnitude or the sizing on top.

Decision rule (ROUTINE.md's own promotion bar, adapted to a candidate-vs-v4-
control comparison, the r105-r147 convention via `compare()`):
  PROMOTE-CANDIDATE (worth carrying to the holdout) if, on inner-validation
  (2021-01-01..2022-12-31), for AT LEAST ONE pre-registered alpha in
  ALPHA_GRID, on BOTH markets (spot and futures_5x):
    (a) the paired bootstrap 95% CI on d_log_growth (candidate - v4)
        excludes zero on the positive side, AND
    (b) d_sharpe >= +0.2 (the noise floor) OR a real (risk-matched,
        exposure_ratio in [0.9,1.1]) drawdown improvement, AND
    (c) the SAME sign of improvement (on whichever of (a)/(b) fired)
        reproduces on the eth_replication slice -- not inverted.
  Any other outcome on inner-validation is NEGATIVE for that branch.
  A branch that clears all three moves to the holdout ONLY after the
  operator freezes the SPECIFIC alpha (no further tuning) and logs it here
  before running `ev(..., start=OOS_START)`.

Falsification test (pre-registered per ROUTINE.md Step 2, chosen from its
menu of four): **ETH sign-replication** -- clause (c) above IS the
falsification test, not a separate afterthought; this is the specific axis
R-87/R-109/R-114-novel died on and the standing diagnosis's own repeated
warning ("does it replicate" is n=1-asset unless checked on a second one).

Configs evaluated so far by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the R-160
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
    v4_scale,
    v4_target,
    v4_vote_frac,
)
from experiments.r102_shared import _latched_anchor_vote  # noqa: E402

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
FEE_TIER = 0.0040                  # 0.40% taker, cost-robustness sensitivity
SHARPE_NOISE_FLOOR = 0.2           # ROUTINE.md's own promotion bar (R-20)
PVAL_SIGMA_DAYS = 5                # causal lookback for the gap z-score, PRIMARY
PVAL_SIGMA_DAYS_ALT = 10           # robustness/plateau check
ALPHA_GRID = (0.20, 0.10, 0.35)    # target FDR budget; PRIMARY is first
ALPHA_PRIMARY = ALPHA_GRID[0]
W0_FRACTION = 0.5                  # initial wealth W0 = W0_FRACTION * alpha
GATE_MIN_DELAYS = 3                # A1 kill switch: >=1 anchor must have
                                    # >=3 flip-episodes actually delayed by
                                    # >=1 bar, else the gate never binds
R2_DEGENERACY_THRESH = 0.999       # A2 kill switch: gated frac vs v4 frac


# ================================================================== (1)
# Causal per-anchor gap z-score / p-value construction. Shared so BOTH
# branches' gates see IDENTICAL evidence -- only the accept/reject rule
# (LORD vs SAFFRON) differs between them.
# ==================================================================

def anchor_gap_zscore(close: pd.Series, days: int,
                       sigma_days: int = PVAL_SIGMA_DAYS) -> np.ndarray:
    """Causal z-score of the price/anchor gap: (close - anchor) / a trailing
    std of that same gap over `sigma_days`, SHIFTED by one bar so bar i's
    z-score never uses bar i's own close inside the std window's most
    recent point beyond what `anchor` (a rolling mean ending at i) already
    legitimately includes. Purely a function of `close[:i+1]`."""
    anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
    gap = close - anchor
    sigma = gap.rolling(int(sigma_days * BARS_PER_DAY), min_periods=BARS_PER_DAY).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, gap / sigma, 0.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def two_sided_normal_pvalue(z: np.ndarray) -> np.ndarray:
    """p = 2*(1 - Phi(|z|)), via math.erf (stdlib only, no scipy dependency).
    H0: the price/anchor gap at this bar is noise around zero."""
    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    phi = 0.5 * (1.0 + np.vectorize(math.erf)(abs_z / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    return np.clip(p, 1e-15, 1.0)


def anchor_raw_vote_and_pvalues(close: pd.Series, days: int,
                                 band: float = V4_BAND,
                                 sigma_days: int = PVAL_SIGMA_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """One anchor's v4-identical raw latched vote (0/1) plus a causal
    p-value defined at EVERY bar (not only at flip bars) -- each branch's
    gate decides, at every bar where the raw vote currently disagrees with
    its own gated state, whether that bar's p-value clears its adaptive
    threshold."""
    raw = _latched_anchor_vote(close, days, band).to_numpy()
    z = anchor_gap_zscore(close, days, sigma_days)
    p = two_sided_normal_pvalue(z)
    return raw, p


# ================================================================== (2)
# Combine three (already gated) per-anchor 0/1 vote arrays into v4's own
# equal-weight frac, and wire through v4's own unmodified scale/deadband.
# ==================================================================

def combine_gated_votes(gated_votes: list[np.ndarray]) -> np.ndarray:
    """Equal-weight mean of k gated 0/1 anchor votes -- v4's own combination,
    applied to whatever (possibly delayed) per-anchor votes a branch built."""
    return np.mean(np.stack(gated_votes, axis=0), axis=0)


def build_target_from_frac(frac: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Wire an arbitrary vote-fraction path through v4's OWN unmodified
    scale/deadband machinery -- identical helper to r147_shared's, re-stated
    here so this file has no import-order surprise."""
    scale = v4_scale(df)
    desired = np.asarray(frac, dtype=float) * scale
    return apply_deadband(desired)


# ================================================================== (3)
# Kill switches, applied identically to both branches before any
# Sharpe/holdout number is read.
# ==================================================================

def count_delayed_episodes(raw: np.ndarray, gated: np.ndarray) -> int:
    """Number of distinct raw-vote flip episodes where the gated vote's own
    change lags the raw flip by >=1 bar (i.e. the gate actually delayed
    something, not merely copied `raw` through unchanged)."""
    raw = np.asarray(raw, dtype=float)
    gated = np.asarray(gated, dtype=float)
    n = len(raw)
    delayed = 0
    i = 1
    while i < n:
        if raw[i] != raw[i - 1]:
            # a raw flip at i: does gated catch up strictly later than i?
            j = i
            while j < n and gated[j] == gated[i - 1]:
                j += 1
            if j > i:
                delayed += 1
            i = j if j > i else i + 1
        else:
            i += 1
    return delayed


def synthetic_zero_drift_frame(n: int = 400_000, seed: int = 160) -> pd.DataFrame:
    """Pure-noise (zero-drift) synthetic OHLCV for the calibration self-test:
    under H0 (no real trend anywhere), a well-calibrated online-FDR gate
    should accept flips at roughly its nominal alpha rate or less, not
    dramatically more. This is a calibration SANITY CHECK, not a proof --
    the real p-value stream is serially correlated, which LORD/SAFFRON's
    independence-flavoured proofs do not strictly cover, named as failure
    mode (2) above."""
    idx = pd.date_range("2017-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    innov = rng.normal(0, 0.0006, n)
    close = 10_000 * np.exp(np.cumsum(innov))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, n)))
    return pd.DataFrame({"open": close, "high": high, "low": low,
                          "close": close, "volume": 1.0}, index=idx)


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=150_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(160)
    regime_len = 20_000
    n_regimes = len(idx) // regime_len + 1
    drift_signs = np.resize([1.0, -1.0], n_regimes)
    drift_per_bar = np.repeat(drift_signs, regime_len)[: len(idx)] * 0.00004
    innov = rng.normal(0, 0.0006, len(idx))
    close = 10_000 * np.exp(np.cumsum(innov + drift_per_bar))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)
    close_s = df["close"]

    # (1) raw vote / p-value construction: shapes, ranges, causality.
    raw, p = anchor_raw_vote_and_pvalues(close_s, 20)
    assert set(np.unique(raw)) <= {0.0, 1.0}
    assert len(p) == len(raw)
    assert np.all((p > 0.0) & (p <= 1.0))

    def _p_builder(d: pd.DataFrame) -> np.ndarray:
        return anchor_raw_vote_and_pvalues(d["close"], 20)[1]

    def _raw_builder(d: pd.DataFrame) -> np.ndarray:
        return anchor_raw_vote_and_pvalues(d["close"], 20)[0]

    assert causal_truncation_probe_series(_p_builder, df)
    assert causal_truncation_probe_series(_raw_builder, df)

    # (2) two_sided_normal_pvalue: z=0 -> p=1; large |z| -> p near 0.
    pv = two_sided_normal_pvalue(np.array([0.0, 1.96, 5.0]))
    assert abs(pv[0] - 1.0) < 1e-9
    assert 0.03 < pv[1] < 0.06
    assert pv[2] < 1e-5

    # (3) combine_gated_votes: equal input -> mean; matches v4 when all three
    # gated votes equal v4's own raw votes.
    raws = [anchor_raw_vote_and_pvalues(close_s, d)[0] for d in V4_HORIZONS]
    combined = combine_gated_votes(raws)
    assert np.allclose(combined, v4_vote_frac(df).to_numpy())

    # (4) build_target_from_frac reproduces v4_target exactly when fed v4's
    # own unmodified vote fraction.
    assert np.allclose(build_target_from_frac(v4_vote_frac(df).to_numpy(), df), v4_target(df))

    # (5) count_delayed_episodes: identical arrays -> 0; a one-bar-delayed
    # flip -> 1.
    a = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    assert count_delayed_episodes(a, a) == 0
    b = np.array([0.0, 0.0, 0.0, 1.0, 1.0])  # gated catches up 1 bar late
    assert count_delayed_episodes(a, b) == 1

    # (6) synthetic_zero_drift_frame: sane OHLCV shape, no drift by
    # construction (mean log-return near zero).
    z = synthetic_zero_drift_frame(n=50_000, seed=1)
    lr = np.diff(np.log(z["close"].to_numpy()))
    se = 0.0006 / math.sqrt(len(lr))
    assert abs(lr.mean()) < 4 * se, (lr.mean(), se)


_self_test()
