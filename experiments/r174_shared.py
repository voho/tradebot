"""Shared, read-only pre-registration for the R-174 round (08-28).

DIRECTION, one sentence: gate `kelly_regime_v4`'s `frac*scale` re-sizing
decision behind an anytime-valid sequential test of "has the recent return
process shown enough evidence of positive drift to justify it", but ONLY on
bars where the move would INCREASE exposure -- decreases stay immediate and
ungated, exactly as v4 ships today.

Full Step 1/Step 2 design (constraint attacked [ERR primary, SIZE
secondary], non-duplication against R-28/R-31/R-87/R-160/R-161/R-167/R-171/
R-172, simulability, named failure modes, the pre-registered decision rule)
is in `experiments/r174_direction.md`, written by the operator BEFORE
either branch was dispatched. This module implements NEITHER branch's own
statistical test -- that is each branch's own job, on top of the
`run_asymmetric_gate` state machine below, which is deliberately neutral
between them (it does not know or care whether `step_fn` is a Wald SPRT or
a GROW e-value). Neither branch may edit this file or each other's file
(R-89-through-R-173's own convention).

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import (  # noqa: E402,F401
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
    V4_DEADBAND,
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
    v4_symmetric_vol,
    v4_target,
    v4_vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND
assert abs(V4_DEADBAND - 0.10) < 1e-12, V4_DEADBAND

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch touches real data.
# MU1/TAU are DERIVED below from inner-train only (2017-2020), not
# hand-copied, so a transcription error cannot silently produce a wrong
# value (this project's own convention, e.g. r172_shared's norm_ppf
# derivation).
# ------------------------------------------------------------------------
_btc_full = load_btc()
_train_close = _btc_full.loc[INNER_TRAIN_START:INNER_TRAIN_END, "close"]
_train_r = np.diff(np.log(_train_close.to_numpy()))
MU1 = float(_train_r.mean())          # inner-train mean per-bar log-return
TAU = MU1                             # novel branch's mixture-prior scale
assert 1e-7 < MU1 < 1e-4, MU1         # sanity: a plausible per-bar BTC drift
del _btc_full, _train_close, _train_r  # do not leak inner-train frames

ALPHA_GRID = (0.10, 0.05, 0.20)       # target type-I budget; PRIMARY first
ALPHA_PRIMARY = ALPHA_GRID[0]
SHARPE_NOISE_FLOOR = 0.2              # ROUTINE.md's own promotion bar (R-20)
GATE_MIN_DELAYS = 3                   # A1 kill switch, ported from R-160
R2_DEGENERACY_THRESH = 0.999          # A2 kill switch, ported from R-160


# ================================================================== (1)
# Causal per-bar return / sigma inputs, identical for both branches.
# ==================================================================

def causal_bar_returns(df: pd.DataFrame) -> np.ndarray:
    """Bar i's own realized log return, log(close_i) - log(close_{i-1}) --
    known at bar i's close, the same epistemic cutoff v4's own anchor vote
    uses when it compares `close` at bar i against a rolling anchor. First
    element is NaN (no prior bar)."""
    r = np.log(df["close"]).diff()
    return r.to_numpy()


def causal_bar_sigma(df: pd.DataFrame) -> np.ndarray:
    """A causal per-bar standard-deviation estimate for the sequential
    tests' Gaussian model, DERIVED from v4's own shipped volatility input
    (`v4_symmetric_vol`, already annualized and already `.shift(1)`'d so it
    uses no information past bar i-1) rather than inventing a new
    estimator. `sigma[i]` is therefore a plug-in nuisance parameter known
    strictly BEFORE `causal_bar_returns(df)[i]` is observed -- the standard
    "known variance" setup both SPRT and the Gaussian-mixture e-value
    assume, with no leakage between the two."""
    annualized = v4_symmetric_vol(df)
    return annualized / np.sqrt(BARS_PER_YEAR)


# ================================================================== (2)
# The gate itself: statistically neutral. Everything about WHICH bars are
# "increases", how a decrease is handled, when an episode starts/ends/
# resets, and the two kill-switch counters is identical for both branches
# -- only `new_episode_state`/`step_fn` (each branch's own statistical
# test) differs.
# ==================================================================

def run_asymmetric_gate(desired: np.ndarray, r: np.ndarray, sigma: np.ndarray,
                        new_episode_state, step_fn,
                        deadband: float = V4_DEADBAND) -> tuple[np.ndarray, int, int]:
    """Gate INCREASES in `desired` (v4's own frac*scale, always >= 0 since
    v4 never shorts -- see `v4_raw_desired`) behind a sequential test;
    decreases apply immediately and unconditionally, exactly like v4's own
    `apply_deadband`.

    - `desired[i] <= pos + deadband`: no rebalance-worthy increase is
      requested. Handled EXACTLY as `apply_deadband` -- if
      `abs(desired[i] - pos) > deadband` (a genuine decrease), apply it
      immediately. Cancels any pending increase episode.
    - `desired[i] > pos + deadband`: a genuine increase request. Starts a
      fresh episode (`new_episode_state()`) if none is pending, then calls
      `step_fn(state, r[i], sigma[i])` once per bar while pending.
      "accept" grants the CURRENT bar's `desired[i]` (not a stale value)
      and resets; "reject" resets with no grant (a fresh episode starts
      from zero evidence if the request still holds next bar); "pending"
      holds position unchanged, one more bar of evidence accumulates.
      A bar where `r[i]`/`sigma[i]` is not finite (warmup) contributes no
      evidence and is treated as "pending".

    Returns `(target_path, delayed_episodes, resolved_after_delay)`:
    `delayed_episodes` counts episodes resolved (accept or reject) on a
    LATER bar than the one that started them -- R-160's own "did the gate
    actually delay something" kill-switch definition (A1). Both kill-switch
    counters are computed identically for either branch, since they are a
    property of the GATE's behaviour, not of which test drives it.
    """
    desired = np.asarray(desired, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(desired)
    assert len(r) == n and len(sigma) == n

    out = np.zeros(n)
    pos = 0.0
    state = None
    pending = False
    episode_start = -1
    total_episodes = 0
    delayed_episodes = 0

    for i in range(n):
        d = desired[i]
        if d <= pos + deadband:
            if abs(d - pos) > deadband:
                pos = d
            pending = False
            state = None
        else:
            if not pending:
                pending = True
                state = new_episode_state()
                episode_start = i
                total_episodes += 1
            ri, si = r[i], sigma[i]
            if np.isfinite(ri) and np.isfinite(si) and si > 0.0:
                state, decision = step_fn(state, ri, si)
            else:
                decision = "pending"
            if decision == "accept":
                pos = d
                if i > episode_start:
                    delayed_episodes += 1
                pending = False
                state = None
            elif decision == "reject":
                if i > episode_start:
                    delayed_episodes += 1
                pending = False
                state = None
            # "pending": pos unchanged, wait for the next bar.
        out[i] = pos
    return out, delayed_episodes, total_episodes


def synthetic_zero_drift_frame(n: int = 400_000, seed: int = 174) -> pd.DataFrame:
    """Pure-noise (zero-drift) synthetic OHLCV, identical shape to R-160's
    own calibration-null generator, so both branches' calibration
    self-tests use the SAME null. Under H0 (no real drift anywhere), a
    well-calibrated gate should accept an increase at roughly its nominal
    alpha rate or less -- a sanity check, not a proof (the real return
    stream is not iid Gaussian, which both branches' Gaussian model
    disclaims up front)."""
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
    rng = np.random.default_rng(174)
    innov = rng.normal(0.00003, 0.0006, len(idx))
    close = 10_000 * np.exp(np.cumsum(innov))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    # (1) causal_bar_returns / causal_bar_sigma: shapes, causality.
    r = causal_bar_returns(df)
    sigma = causal_bar_sigma(df)
    assert len(r) == len(df) and len(sigma) == len(df)
    assert np.isnan(r[0])
    assert causal_truncation_probe_series(causal_bar_returns, df)
    assert causal_truncation_probe_series(causal_bar_sigma, df)

    # (2) run_asymmetric_gate: a trivial "always accept immediately" test
    # must reproduce v4's own apply_deadband exactly (the gate degenerates
    # to the unconditional rule when step_fn never blocks anything).
    desired = v4_raw_desired(df)

    def _always_accept(state, ri, si):
        return state, "accept"

    gated, delayed, total = run_asymmetric_gate(desired, r, sigma, lambda: {}, _always_accept)
    assert np.allclose(gated, apply_deadband(desired)), "always-accept gate must equal v4's own deadband"
    assert delayed == 0, "an always-accept step_fn should never delay anything"

    # (3) a trivial "always reject" test must never increase past the
    # starting position (0.0) -- decreases (never triggered here, since
    # desired stays >= 0 and pos starts at 0) would still pass through.
    def _always_reject(state, ri, si):
        return state, "reject"

    gated2, delayed2, total2 = run_asymmetric_gate(desired, r, sigma, lambda: {}, _always_reject)
    assert np.all(gated2 <= 1e-9), "always-reject gate must never grant an increase from 0"

    # (4) a "pending forever" test used with a 2-step schedule: accept
    # exactly one bar after the episode starts, to check the "delayed"
    # counter and that the GRANTED value is the CURRENT bar's desired, not
    # the stale value from when the episode opened.
    toy_desired = np.array([0.0, 0.5, 0.6, 0.6, 0.0, 0.3])
    toy_r = np.zeros(len(toy_desired))
    toy_sigma = np.ones(len(toy_desired))

    def _accept_second_call():
        return {"n": 0}

    def _accept_on_second(state, ri, si):
        state["n"] += 1
        return state, ("accept" if state["n"] >= 2 else "pending")

    toy_gated, toy_delayed, toy_total = run_asymmetric_gate(
        toy_desired, toy_r, toy_sigma, _accept_second_call, _accept_on_second, deadband=0.10)
    # bar1 requests 0.5 (episode starts), bar2 still pending (n=1->accept
    # check uses n after increment, so bar1: n=1 pending; bar2: n=2 accept
    # at desired[2]=0.6, not the stale 0.5 from bar1).
    assert toy_gated[1] == 0.0, toy_gated
    assert toy_gated[2] == 0.6, toy_gated
    assert toy_delayed == 1, (toy_delayed, toy_total)
    # bar4 (index 4) is a decrease to 0.0: immediate, no gate.
    assert toy_gated[4] == 0.0, toy_gated
    # bar5 (index 5) requests 0.3 from 0.0: a fresh episode starts, still
    # pending at series end (n=1) so position holds at 0.0.
    assert toy_gated[5] == 0.0, toy_gated
    assert toy_total == 2, toy_total  # one episode at bar1, one at bar5

    # (5) MU1/TAU sanity and reproducibility (re-derive independently here).
    btc = load_btc()
    train_close = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END, "close"]
    train_r = np.diff(np.log(train_close.to_numpy()))
    assert abs(float(train_r.mean()) - MU1) < 1e-15
    assert TAU == MU1

    # (6) synthetic_zero_drift_frame: sane OHLCV shape, no drift by
    # construction.
    import math
    z = synthetic_zero_drift_frame(n=50_000, seed=1)
    lr = np.diff(np.log(z["close"].to_numpy()))
    se = 0.0006 / math.sqrt(len(lr))
    assert abs(lr.mean()) < 4 * se, (lr.mean(), se)


_self_test()
