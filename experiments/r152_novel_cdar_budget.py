"""R-152 NOVEL branch: CDaR-budgeted exposure for ``kelly_regime_v4``.

Frozen pre-registration: ``experiments/r152_shared.py`` (not edited here).
This file implements exactly the "Novel" branch of that pre-registration
and is NOT registered (``experiments/`` only, per ROUTINE.md step 5) and
NOT auto-discovered.

**Mechanism.** ``kelly_regime_v4`` (== ``kelly_regime_v3`` on a 20/40/80-day
anchor ladder) sizes with two magnitude arrays fed into a hysteresis state
machine that switches between them:

    full   = min(target_vol / vol,  max_leverage)   # reactive
    steady = min(target_vol / slow, max_leverage)   # EMA-smoothed(vol)
    scale  = full  if state != 0 (vol-breakout latched)
             steady otherwise

This branch replaces ONLY the risk statistic each of those two ratios is
built from -- ``vol`` and its EMA-smoothed sibling ``slow`` -- with a
rolling CDaR_0.95 of the strategy's own unit vote-scaled return stream,
via the closed-form solve named in the pre-registration (CDaR is
homogeneous of degree 1 in exposure, Chekhlov/Uryasev/Zabarankin 2005
Prop. 3, so no iterative optimizer is needed):

    unit_returns = frac * r                          # r = log-return, v4's own
    cdar         = rolling_cdar(unit_returns, window_bars(cdar_window_days))
    cdar_slow    = ewm(cdar, span=anchor_span_days)   # the CDaR-axis analogue of `slow`
    full         = clip(budget / cdar,      0, max_leverage)
    steady       = clip(budget / cdar_slow, 0, max_leverage)
    scale        = full if state != 0 else steady     # SAME state machine as v4

**Design decision, disclosed rather than hidden.** The pre-registration's
mechanism paragraph gives exactly one closed form, `f* = budget /
CDaR_0.95(unit_returns)` -- a single quantity, not a fast/slow pair. Two
readings were possible: (a) use that one f* for both `full` and `steady`,
which leaves the retained state machine computed but inert (it would
never change the emitted scale), or (b) extend the SAME substitution v4
already applies to `vol` (a fast EWM stat -> `slow`, an EMA-smoothed
version of it) to `cdar`, since `target_vol / X` is quite literally
`budget / cdar` with `X` relabeled. (b) is what is implemented: it is the
minimal, mechanical extension of the one given formula that keeps the
hysteresis state machine doing the same JOB it did in v4 (reactive vs.
held-steady exposure) rather than retaining dead code, and it changes
nothing about the vol-ratio trigger axis itself (`ratio = vol/slow` is
untouched, exactly as instructed). `full` alone -- the direct,
un-smoothed `budget / cdar` -- is what diagnostic B2 and the smoke test
below treat as "the derived f*", since that is the one formula the
pre-registration actually states.

**Why `r` (log returns), not simple returns, feeds `rolling_cdar`.**
`rolling_cdar`'s docstring calls its input "per-bar simple returns", but
the pre-registration's own mechanism text is explicit: "the unit
vote-scaled returns are `frac[i] * r[i]`" where `r` is v4's own return
variable -- `np.log(close).diff()`, log returns, used unchanged everywhere
else in `kelly_regime`/`kelly_regime_v3` (including to build `vol`). At
5-minute bars, per-bar returns are ~1e-4 in magnitude, where log and
simple returns agree to within their own square -- an O(1e-8) relative
error that is irrelevant next to CDaR's own estimation noise. Followed
literally rather than silently "corrected" to a different input.

**Cold-start note.** The committed dataset starts exactly at
2017-01-01 -- inner-train's own start date -- so there is NO warmup
runway before it, for v4 or this branch (`scripts/experiment.py`'s
`run_period` already documents this as unavoidable for the first
in-sample split). A 545-day CDaR window is therefore genuinely NaN (and
this branch sits flat, `fstar = 0`, matching how v4 itself goes flat
during its own vol/slow warmup) for the first ~545 days of the ENTIRE
9-year series, i.e. ~37% of the 4-year inner-train period, only for the
longest sweep member. This is a real, disclosed property of the
mechanism on this dataset, not a bug -- and it is exactly why the
budget calibration below is run separately per window length rather than
reused across the sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r152_shared import (  # noqa: E402
    B2_CORRELATION_FLAG,
    CDAR_BETA,
    CDAR_WINDOW_DAYS_DEFAULT,
    CDAR_WINDOW_DAYS_SWEEP,
    DRAWDOWN_IMPROVEMENT_PP,
    EXPOSURE_MATCH_TOL_PP,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    rolling_cdar,
    window_bars,
)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

INNER_TRAIN_START = "2017-01-01"  # not exported by r152_shared; matches its own docstring verbatim
FUTURES = MarketSpec.futures(leverage=5.0)


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def load_train_val(kind: str = "spot"):
    """Load the committed dataset, hard-truncated at inner-validation's end.

    Every computation in this file is downstream of this call, so nothing
    below can read a holdout bar even by accident.
    """
    df, label = load_dataset(ROOT / "data", kind)
    train = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train, label


# --------------------------------------------------------------------- #
# Duplicated, read-only pieces of kelly_regime_v3 / kelly_regime_v4's own
# math. v3/v4 are frozen strategies (not edited) -- these exist only so
# the branch (a) can compute v4's own `scale` array to calibrate against,
# without hacking into the engine, and (b) reuses IDENTICAL formulas to
# v4's own `vote`/`vol`/hysteresis so the branch's `frac`, `ratio` and
# `state` are byte-identical to v4's, not independently re-derived
# (independently re-deriving them is exactly how two "identical" columns
# drift apart silently). Same convention as experiments/matched_risk.py
# and experiments/r125_shared.py's v4_reference_target.
# --------------------------------------------------------------------- #

def vote_frac(close: pd.Series, horizons: tuple[int, ...], band: float,
              vote_gamma: float) -> np.ndarray:
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    frac = (sum(votes) / len(votes)).to_numpy()
    if vote_gamma != 1.0:
        frac = frac ** vote_gamma
    return frac


def realized_vol(r: pd.Series, vol_span: int) -> np.ndarray:
    return (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def ewm_smooth(x: np.ndarray, span_days: int) -> np.ndarray:
    return (pd.Series(x).ewm(span=span_days * BARS_PER_DAY, min_periods=BARS_PER_DAY)
            .mean().to_numpy())


def hysteresis_state(ratio: np.ndarray, high_in: float, high_out: float,
                      low_in: float, low_out: float) -> np.ndarray:
    """v3/v4's own latching state machine on `ratio` (== vol/slow), unchanged.

    state: 0 normal band, +1 high-vol breakout latched, -1 low-vol breakout
    latched. Shared, byte-identical logic between v4's own scale and this
    branch's -- both call this same function on the same `ratio` array.
    """
    n = len(ratio)
    out = np.empty(n, dtype=np.int8)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > high_in else (-1 if x < low_in else 0)
            elif state == 1 and x < high_out:
                state = 0
            elif state == -1 and x > low_out:
                state = 0
        out[i] = state
    return out


def v4_reference_scale(df: pd.DataFrame, v4: KellyRegimeV4) -> dict:
    """v4/v3's own frac/vol/slow/ratio/state/scale, computed directly (not
    via the engine) so this branch can calibrate against v4's own mean
    `scale` and diagnose B2 against v4's own `vol`, cheaply and repeatedly.
    """
    close = df["close"]
    r = np.log(close).diff()
    frac = vote_frac(close, v4.horizons, v4.band, v4.vote_gamma)
    vol = realized_vol(r, v4.vol_span)
    slow = ewm_smooth(vol, v4.anchor_span_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(v4.target_vol / vol, v4.max_leverage)
        steady = np.minimum(v4.target_vol / slow, v4.max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)
    state = hysteresis_state(ratio, v4.high_in, v4.high_out, v4.low_in, v4.low_out)
    scale = np.where(state != 0, full, steady)
    return {"r": r.to_numpy(), "frac": frac, "vol": vol, "slow": slow,
            "ratio": ratio, "state": state, "scale": scale}


# --------------------------------------------------------------------- #
# The branch strategy.
# --------------------------------------------------------------------- #

class KellyRegimeV4CDaRBudget(KellyRegimeV4):
    """``kelly_regime_v4`` with its vol-target ratio replaced by a CDaR_0.95
    exposure budget (R-152 NOVEL branch; see module docstring).

    The vote (`frac`), the vol-ratio hysteresis trigger (`ratio = vol/slow`,
    `high_in/high_out/low_in/low_out`), and the deadband are v4's own,
    unchanged. Only what feeds `full`/`steady` changes: `budget / cdar`
    and `budget / ewm(cdar)` in place of `target_vol / vol` and
    `target_vol / slow`.

    ``cdar_budget`` has no safe default -- it must be calibrated on
    inner-train against v4's own mean exposure (``calibrate_budget``
    below) before this class is instantiated for a real run. Passing a
    blind guess would silently violate R-33's matched-risk discipline
    this whole branch exists to pay up front.
    """

    name = "r152_novel_cdar_budget"  # experiments/ only -- no @register

    def __init__(self, cdar_window_days: int = CDAR_WINDOW_DAYS_DEFAULT,
                 cdar_budget: float | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if cdar_budget is None or cdar_budget <= 0:
            raise ValueError(
                "cdar_budget must be a calibrated positive scalar "
                "(see calibrate_budget()) -- refusing an uncalibrated default.")
        self.cdar_window_days = cdar_window_days
        self.cdar_budget = cdar_budget

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        frac = vote_frac(close, self.horizons, self.band, self.vote_gamma)

        # Vol-ratio trigger axis: UNCHANGED from v3/v4. This is what the
        # hysteresis state machine watches; the pre-registration keeps it
        # "on the vol-ratio axis" explicitly -- only the magnitude fed into
        # `full`/`steady` below is replaced.
        vol = realized_vol(r, self.vol_span)
        slow = ewm_smooth(vol, self.anchor_span_days)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)

        # --- NOVEL SCALE: closed-form CDaR-budgeted exposure ---
        unit_returns = frac * np.nan_to_num(r.to_numpy(), nan=0.0)
        cdar = rolling_cdar(unit_returns, window_bars(self.cdar_window_days), beta=CDAR_BETA)
        cdar_slow = ewm_smooth(cdar, self.anchor_span_days)

        with np.errstate(divide="ignore", invalid="ignore"):
            full = np.where(cdar > 0, self.cdar_budget / cdar, np.inf)
            steady = np.where(cdar_slow > 0, self.cdar_budget / cdar_slow, np.inf)
        full = np.clip(full, 0.0, self.max_leverage)
        steady = np.clip(steady, 0.0, self.max_leverage)
        full = np.where(np.isfinite(cdar), full, 0.0)
        steady = np.where(np.isfinite(cdar_slow), steady, 0.0)

        state = hysteresis_state(ratio, self.high_in, self.high_out, self.low_in, self.low_out)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            scale = full[i] if state[i] != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["cdar_full"] = full          # the direct f* = budget/CDaR(unit_returns) -- used for B2
        df["cdar_steady"] = steady
        df["cdar"] = cdar
        return df


# --------------------------------------------------------------------- #
# Calibration: pick `cdar_budget` so this branch's inner-train mean scale
# matches v4's own inner-train mean scale (R-33's matched-risk discipline,
# paid up front). Bisection in log-space -- `scale(budget)` is monotone
# nondecreasing in `budget` before saturating at `max_leverage`, so this
# always converges; state/cdar/cdar_slow are budget-independent and
# computed once, so each bisection step is a handful of vectorized numpy
# ops, not a re-run of `rolling_cdar` or the engine.
# --------------------------------------------------------------------- #

def calibrate_budget(df: pd.DataFrame, v4: KellyRegimeV4, cdar_window_days: int,
                      target_mean_scale: float, max_leverage: float,
                      anchor_span_days: int, lo: float = 1e-4, hi: float = 4.0,
                      iters: int = 30) -> dict:
    close = df["close"]
    r = np.log(close).diff()
    frac = vote_frac(close, v4.horizons, v4.band, v4.vote_gamma)
    unit_returns = frac * np.nan_to_num(r.to_numpy(), nan=0.0)
    cdar = rolling_cdar(unit_returns, window_bars(cdar_window_days), beta=CDAR_BETA)
    cdar_slow = ewm_smooth(cdar, anchor_span_days)

    vol = realized_vol(r, v4.vol_span)
    slow = ewm_smooth(vol, v4.anchor_span_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
    state = hysteresis_state(ratio, v4.high_in, v4.high_out, v4.low_in, v4.low_out)

    train_mask = (df.index >= pd.Timestamp(INNER_TRAIN_START, tz=df.index.tz)) & \
                 (df.index <= pd.Timestamp(INNER_TRAIN_END, tz=df.index.tz))

    def combined_scale(budget: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            full = np.where(cdar > 0, budget / cdar, np.inf)
            steady = np.where(cdar_slow > 0, budget / cdar_slow, np.inf)
        full = np.clip(full, 0.0, max_leverage)
        steady = np.clip(steady, 0.0, max_leverage)
        full = np.where(np.isfinite(cdar), full, 0.0)
        steady = np.where(np.isfinite(cdar_slow), steady, 0.0)
        return np.where(state != 0, full, steady)

    evals: list[tuple[float, float]] = []

    def mean_at(budget: float) -> float:
        m = float(np.nanmean(combined_scale(budget)[train_mask]))
        evals.append((budget, m))
        return m

    m_lo, m_hi = mean_at(lo), mean_at(hi)
    while m_hi < target_mean_scale and hi < 1e5:
        hi *= 2.0
        m_hi = mean_at(hi)

    mid = lo
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        m_mid = mean_at(mid)
        if m_mid < target_mean_scale:
            lo = mid
        else:
            hi = mid

    return {
        "budget": mid,
        "achieved_mean_scale": mean_at(mid),
        "target_mean_scale": target_mean_scale,
        "n_calibration_evals": len(evals) + 1,
        "evals": evals,
    }


# --------------------------------------------------------------------- #
# Smoke test (ROUTINE.md step 2 / the R-89 lesson): before any real-data
# run, confirm f* moves the RIGHT WAY on a short synthetic series where
# the answer is known by construction.
# --------------------------------------------------------------------- #

def _synthetic_returns(seed: int = 0) -> np.ndarray:
    """quiet -> sharp drawdown -> quiet, small bars so the test is fast."""
    rng = np.random.default_rng(seed)
    quiet1 = rng.normal(0.0, 0.0004, 3000)
    drawdown = -np.abs(rng.normal(0.006, 0.002, 150))  # a hard, sustained selloff
    quiet2 = rng.normal(0.0, 0.0004, 3000)
    return np.concatenate([quiet1, drawdown, quiet2])


def smoke_test() -> bool:
    x = _synthetic_returns()
    window_bars_test = 800
    recompute_every_test = 20
    cdar = rolling_cdar(x, window_bars_test, beta=CDAR_BETA, recompute_every=recompute_every_test)

    budget = 0.05
    max_leverage = 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fstar = np.where(cdar > 0, budget / cdar, np.inf)
    fstar = np.clip(fstar, 0.0, max_leverage)
    fstar = np.where(np.isfinite(cdar), fstar, 0.0)

    ok = True

    # 1. Clipping: f* never exceeds max_leverage, never negative.
    valid = np.isfinite(cdar)
    if not (np.nanmax(fstar) <= max_leverage + 1e-9 and np.nanmin(fstar[valid]) >= 0.0):
        print("SMOKE FAIL: clipping violated"); ok = False

    # 2. Pre-drawdown (quiet1 fully inside the window, before the selloff
    #    enters it): CDaR should be low and f* should sit at or near the cap.
    pre_idx = 2900  # near end of quiet1, drawdown not yet in the trailing window
    pre_fstar = fstar[pre_idx]
    if not (np.isfinite(cdar[pre_idx]) and pre_fstar > 1.5):
        print(f"SMOKE FAIL: pre-drawdown f*={pre_fstar:.3f} (expected > 1.5, near cap "
              f"{max_leverage}); cdar={cdar[pre_idx]:.5f}")
        ok = False

    # 3. Well after the drawdown has fully entered the trailing window (but
    #    still within it, window_bars_test=800 bars past its end at index
    #    3150): CDaR should be materially higher, f* should have shrunk
    #    materially relative to the pre-drawdown value.
    post_idx = 3150 + 500
    post_fstar = fstar[post_idx]
    if not (np.isfinite(cdar[post_idx]) and cdar[post_idx] > cdar[pre_idx] * 2.0):
        print(f"SMOKE FAIL: CDaR did not rise after the synthetic drawdown "
              f"(pre={cdar[pre_idx]:.5f}, post={cdar[post_idx]:.5f})")
        ok = False
    if not (post_fstar < pre_fstar * 0.6):
        print(f"SMOKE FAIL: f* did not shrink after the synthetic drawdown "
              f"(pre={pre_fstar:.3f}, post={post_fstar:.3f})")
        ok = False

    # 4. Once the drawdown has fully rolled OUT of the trailing window again
    #    (window_bars_test bars past the end of the drawdown, deep into
    #    quiet2), f* should recover back toward the cap.
    recovered_idx = 3150 + window_bars_test + 400
    if recovered_idx < len(fstar):
        rec_fstar = fstar[recovered_idx]
        if not (rec_fstar > 1.5):
            print(f"SMOKE FAIL: f* did not recover once the drawdown rolled out of the "
                  f"window (f*={rec_fstar:.3f}, expected > 1.5)")
            ok = False

    # 5. Causality: recomputing on a truncated prefix must reproduce every
    #    already-settled value exactly (this is the exact class of bug
    #    named in r152_shared's own rolling_cdar docstring, and in R-89).
    cut = 3400
    trunc = rolling_cdar(x[:cut], window_bars_test, beta=CDAR_BETA, recompute_every=recompute_every_test)
    safe = cut - recompute_every_test - 1
    if not np.allclose(cdar[:safe], trunc[:safe], equal_nan=True, rtol=1e-9):
        print("SMOKE FAIL: truncation probe -- cdar reads ahead of its own cutoff")
        ok = False

    print(f"smoke_test: {'PASS' if ok else 'FAIL'} "
          f"(pre f*={pre_fstar:.3f}, post f*={post_fstar:.3f}, "
          f"cdar pre={cdar[pre_idx]:.5f} post={cdar[post_idx]:.5f})")
    return ok


def smoke_test_causal_prepare(n_bars: int = 250_000) -> bool:
    """Truncation probe on the ACTUAL `prepare()` (not just `rolling_cdar`
    in isolation) -- catches lookahead introduced by code written for this
    branch (the ewm smoothing, the hysteresis loop, frac), not just bugs
    already covered by r152_shared's own test of `rolling_cdar` itself.
    """
    df, _ = load_train_val()
    frame = df.iloc[:n_bars].copy()
    strat = KellyRegimeV4CDaRBudget(cdar_window_days=180, cdar_budget=0.15)
    full = strat.prepare(frame.copy())["target"].to_numpy()

    cut = n_bars - 60_000
    trunc_frame = df.iloc[:cut].copy()
    trunc = strat.prepare(trunc_frame)["target"].to_numpy()

    safe = cut - BARS_PER_DAY * (180 + 2)  # clear of the truncated tail's warmup/recompute lag
    if safe <= 0:
        print("smoke_test_causal_prepare: SKIPPED (not enough bars for a safe margin)")
        return True
    ok = np.allclose(full[:safe], trunc[:safe], equal_nan=True, rtol=1e-9)
    print(f"smoke_test_causal_prepare: {'PASS' if ok else 'FAIL'} (safe={safe:,} bars checked)")
    return ok


# --------------------------------------------------------------------- #
# Diagnostic B2 and the inner-validation comparison.
# --------------------------------------------------------------------- #

def b2_correlation(fstar_full: np.ndarray, vol: np.ndarray, mask: np.ndarray) -> float:
    a, b = fstar_full[mask], vol[mask]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 100:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def run_metrics(strategy, df: pd.DataFrame, market: MarketSpec, start: str, end: str) -> dict:
    res = run_period(strategy, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label="")
    m = compute_metrics(res)
    target = res.df["target"].to_numpy() if "target" in res.df else np.full(len(res.df), np.nan)
    mean_exposure = float(np.nanmean(np.abs(target)))
    return {
        "sharpe": m.sharpe, "max_drawdown_pct": m.max_drawdown_pct,
        "time_in_market_pct": m.time_in_market_pct, "mean_exposure": mean_exposure,
        "final_balance": m.final_balance, "profit_pct": m.profit_pct,
        "num_trades": m.num_trades,
    }


def main() -> None:
    print("=" * 70)
    print("R-152 NOVEL: smoke tests")
    print("=" * 70)
    ok1 = smoke_test()
    ok2 = smoke_test_causal_prepare()
    if not (ok1 and ok2):
        print("SMOKE TEST FAILED -- refusing to run on real data.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Loading data (train + inner-validation only, holdout guarded)")
    print("=" * 70)
    df, label = load_train_val("spot")
    print(f"{len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (data: {label})")

    v4 = KellyRegimeV4()
    ref = v4_reference_scale(df, v4)
    train_mask = (df.index >= pd.Timestamp(INNER_TRAIN_START, tz=df.index.tz)) & \
                 (df.index <= pd.Timestamp(INNER_TRAIN_END, tz=df.index.tz))
    v4_train_mean_scale = float(np.nanmean(ref["scale"][train_mask]))
    print(f"\nv4 own inner-train mean `scale`: {v4_train_mean_scale:.4f}")

    print("\n" + "=" * 70)
    print("Control: kelly_regime_v4 (registered defaults), futures 5x")
    print("=" * 70)
    control_val = run_metrics(KellyRegimeV4(), df, FUTURES, INNER_VAL_START, INNER_VAL_END)
    control_train = run_metrics(KellyRegimeV4(), df, FUTURES, INNER_TRAIN_START, INNER_TRAIN_END)
    print("inner-validation:", control_val)
    print("train (context): ", control_train)

    n_calibration_evals_total = 0
    rows = []
    for cdar_window_days in CDAR_WINDOW_DAYS_SWEEP:
        print("\n" + "=" * 70)
        print(f"Window = {cdar_window_days} days")
        print("=" * 70)
        cal = calibrate_budget(df, v4, cdar_window_days, v4_train_mean_scale,
                                v4.max_leverage, v4.anchor_span_days)
        n_calibration_evals_total += cal["n_calibration_evals"]
        print(f"calibrated cdar_budget = {cal['budget']:.6f}  "
              f"(target mean scale {cal['target_mean_scale']:.4f}, "
              f"achieved {cal['achieved_mean_scale']:.4f}, "
              f"{cal['n_calibration_evals']} calibration evals)")

        branch = KellyRegimeV4CDaRBudget(cdar_window_days=cdar_window_days,
                                          cdar_budget=cal["budget"])
        # B2: correlate this branch's `full` (== budget/CDaR(unit_returns), the direct
        # f* the pre-registration names) against v4's own `vol`, on inner-train.
        # `prepare()` is causal, so computing it once on the whole train+val
        # frame and slicing by date afterwards is equivalent to computing it
        # fresh on each sub-period (no leakage either way).
        prepared_full = branch.prepare(df.copy())
        b2 = b2_correlation(prepared_full["cdar_full"].to_numpy(), ref["vol"], train_mask)
        flag = "FLAGGED (|r|>=%.2f)" % B2_CORRELATION_FLAG if abs(b2) >= B2_CORRELATION_FLAG else "ok"
        print(f"B2 (branch f* vs v4 vol, inner-train): r={b2:+.4f} [{flag}]")

        branch_val = run_metrics(branch, df, FUTURES, INNER_VAL_START, INNER_VAL_END)
        branch_train = run_metrics(branch, df, FUTURES, INNER_TRAIN_START, INNER_TRAIN_END)
        print("inner-validation:", branch_val)
        print("train (context): ", branch_train)

        exposure_gap_pp = abs(branch_val["mean_exposure"] - control_val["mean_exposure"]) * 100.0
        tim_gap_pp = abs(branch_val["time_in_market_pct"] - control_val["time_in_market_pct"])
        crit1 = (exposure_gap_pp <= EXPOSURE_MATCH_TOL_PP) and (tim_gap_pp <= EXPOSURE_MATCH_TOL_PP)
        d_sharpe = branch_val["sharpe"] - control_val["sharpe"]
        dd_improve_pp = control_val["max_drawdown_pct"] - branch_val["max_drawdown_pct"]
        crit2 = (d_sharpe >= -SHARPE_NOISE_FLOOR) or \
                (dd_improve_pp >= DRAWDOWN_IMPROVEMENT_PP and crit1)

        rows.append({
            "cdar_window_days": cdar_window_days, "budget": cal["budget"], "b2": b2,
            "branch_val": branch_val, "branch_train": branch_train,
            "exposure_gap_pp": exposure_gap_pp, "tim_gap_pp": tim_gap_pp,
            "crit1_exposure_match": crit1, "d_sharpe": d_sharpe,
            "dd_improve_pp": dd_improve_pp, "crit2_sharpe_or_dd": crit2,
        })

    print("\n" + "=" * 70)
    print("Selection rule")
    print("=" * 70)
    signs = [1 if r["d_sharpe"] > 0 else (-1 if r["d_sharpe"] < 0 else 0) for r in rows]
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    plateau = (pos >= 2) or (neg >= 2)
    for r in rows:
        eligible = r["crit1_exposure_match"] and r["crit2_sharpe_or_dd"] and plateau
        print(f"window={r['cdar_window_days']:>3}d  budget={r['budget']:.5f}  B2={r['b2']:+.3f}  "
              f"d_sharpe={r['d_sharpe']:+.3f}  dd_improve_pp={r['dd_improve_pp']:+.2f}  "
              f"exposure_gap_pp={r['exposure_gap_pp']:.2f}  tim_gap_pp={r['tim_gap_pp']:.2f}  "
              f"crit1={r['crit1_exposure_match']}  crit2={r['crit2_sharpe_or_dd']}  "
              f"plateau={plateau}  ELIGIBLE_FOR_HOLDOUT={eligible}")

    print(f"\nplateau: {pos} of {len(rows)} window lengths have d_sharpe>0, "
          f"{neg} have d_sharpe<0 -> plateau={plateau}")
    print(f"\nconfigs evaluated: {len(rows)} window-length configs + 1 control "
          f"= {len(rows) + 1} declared configs; {n_calibration_evals_total} additional "
          f"calibration evaluations (cheap array ops, not backtests, disclosed separately)")


if __name__ == "__main__":
    main()
