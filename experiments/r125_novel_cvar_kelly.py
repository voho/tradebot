#!/usr/bin/env python
"""R-125 NOVEL branch: ``NovelCVaRKelly`` -- a vote-CONDITIONAL, daily-cadence
fractional-Kelly growth optimizer, capped by Rockafellar & Uryasev (2000)'s
CVaR as a budget CONSTRAINT rather than a target to hit.

The complete pre-registration for this round -- direction, literature
citations, non-duplication argument against the 27+ prior SIZE-axis rounds,
and the named failure mode -- lives in ``experiments/r125_shared.py``'s own
module docstring, written by the operator before either branch was
dispatched, and is NOT re-derived here: read that file in full first. This
file imports ONLY from ``experiments.r125_shared``, never edits it, never
coordinates with the conservative branch's file, and never reads a bar at or
after ``r125_shared.OOS_START`` (2023-01-01).

MECHANISM, in one sentence: at a DAILY cadence, solve for the exposure
fraction ``f in [0, max_leverage]`` that maximizes ``mean(log(1 + f*R))``
(``R`` = simple daily return, the quantity the Kelly criterion is actually
defined over -- NOT the log return ``r125_shared`` uses for its own CVaR
estimator) over the CAUSAL trailing empirical distribution of days that
shared the CURRENT day's 3-anchor vote state (falling back to the pooled,
unconditional trailing distribution when that conditional sample is smaller
than ``min_samples``), then cap the result by the CVaR budget:
``f* = min(f_kelly, budget / realized_cvar)``, where ``realized_cvar`` is
``r125_shared.annualized_cvar`` UNCHANGED (pooled, unconditional -- only
``f_kelly`` is vote-conditional, per the pre-registration).

Because ``log`` is strictly concave, ``f_kelly`` frequently sits BELOW the
CVaR cap -- the two-part ``min()`` is the point of the construction, not an
incidental detail. This is architecturally distinct from both v4's own
`frac * (target_vol/realized_vol)` linear rule and the conservative branch's
like-for-like `frac * (target_cvar/realized_cvar)` substitution: `frac`
itself never multiplies anything here. It instead SELECTS which trailing
sample of history the growth-optimizer conditions on -- "what has price
done historically when the vote looked like this" -- so a historically
bearish vote state naturally pulls f_kelly toward (or to) zero on its own,
without a separate vote-scale product.

WHY DAILY, NOT PER-5M-BAR: identical efficiency argument to
``r125_shared.annualized_cvar`` -- a rolling empirical-distribution solve
evaluated once per 5m bar is computationally intractable at this dataset's
size (~1e6 bars), 288x more expensive than necessary, and does not match
the natural resolution of the underlying signal (a discrete 3-anchor daily-
scale vote). The daily f* is forward-filled onto the 5m bar grid via
``_daily_to_bars`` below, written independently in this file (the shared
module's own ``_calendar_daily_close``/ffill PATTERN is reused, not its
private function, per the pre-registration's own instruction not to touch
``r125_shared.py``).

NO SCIPY IN THIS ENVIRONMENT: the 1-D bounded concave maximization is
solved with a dependency-free golden-section search (``_golden_max`` below)
rather than ``scipy.optimize.minimize_scalar`` -- mathematically equivalent
for a strictly unimodal objective, which ``mean(log(1+f*R))`` is on `R`
bounded away from ``-1/f``.

NAMED FAILURE MODE, measured not dodged: the vote-state-conditional sample
may be too small in the transitional 1/3 and 2/3 states to estimate a
stable f_kelly. ``NOVEL_SAMPLE_COUNTS`` (this file's diagnostic, computed
by ``sample_diagnostics()`` below) reports, per discrete vote state, how
many trailing-window days used the CONDITIONAL sample vs. fell back to the
POOLED one, and the median/min sample size in each case -- disclosed
explicitly, not papered over.

CAUSAL SAFETY FIRST: ``causal_truncation_probe`` (bottom of the mechanism
section) is run on real BTC data before ANY inner-validation/ETH number is
trusted, per this file's own ``main()`` and the module-level ``_self_test``
on synthetic data.

PRE-REGISTERED DECISION RULE, stated verbatim and NOT altered after seeing
any number: PROMOTE-candidate only if the causal-truncation probe AND B1
(both markets) AND B3 (plateau majority) AND B4 (full) AND B5 all pass.
Step-0 KILLs first if R^2 vs. v4's own target exceeds 0.98. Anything else
is NEGATIVE. Default: NEGATIVE.

USAGE
-----
    python experiments/r125_novel_cvar_kelly.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r125_shared  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

BARS_PER_DAY = r125_shared.BARS_PER_DAY if hasattr(r125_shared, "BARS_PER_DAY") else 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# ------------------------------------------------------------------------
# Pre-registered constants -- fixed before any inner-validation number is
# read. KELLY_WINDOW_GRID / MIN_SAMPLES_GRID are the B3 plateau sweep;
# CVAR_WINDOW_DAYS is held fixed (not part of this branch's new mechanism --
# it is r125_shared's own CVaR estimator, reused unchanged) at the same
# 90-day window r125_shared's own module self-test exercises.
# ------------------------------------------------------------------------
KELLY_WINDOW_PRIMARY = 365   # trailing days of history the vote-conditional
                             # Kelly sample is drawn from -- longer than a
                             # typical CVaR window on purpose: conditioning
                             # on 4 discrete vote states needs more raw
                             # history to leave enough days per state.
KELLY_WINDOW_GRID = (180, 365, 730)
MIN_SAMPLES_PRIMARY = 30    # pre-registered minimum conditional sample size
MIN_SAMPLES_GRID = (20, 30)
CVAR_WINDOW_DAYS = 90
CVAR_ALPHA = r125_shared.CVAR_ALPHA
HORIZONS = (20, 40, 80)     # v4's own anchor ladder, reproduced verbatim
BAND = 0.01                 # v4's own vote band, reproduced verbatim


# ================================================================== (1)
# The mechanism itself: vote (frac) -> daily vote-conditional Kelly solve
# (with pooled fallback) -> CVaR-budget cap -> forward-fill onto 5m bars.
# ==================================================================

def vote_frac(close: pd.Series, horizons: tuple[int, ...] = HORIZONS,
              band: float = BAND) -> pd.Series:
    """v4's own 3-anchor latched vote, reproduced byte-for-byte (see
    ``KellyRegime.prepare`` / ``KellyRegimeV3.prepare``): frac in
    {0, 1/3, 2/3, 1}, causal (rolling mean + ffill hysteresis)."""
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return sum(votes) / len(votes)


def _calendar_daily_close(close: pd.Series) -> pd.Series:
    """Last 5m close of each UTC calendar day -- causal by construction.
    Written independently in this file (not imported from r125_shared,
    which is frozen and not to be edited); identical pattern."""
    return close.resample("1D").last().dropna()


def _golden_max(obj, lo: float, hi: float, tol: float = 1e-3,
                 max_iter: int = 60) -> float:
    """Dependency-free bounded scalar maximizer (golden-section search) for
    a strictly unimodal (here: strictly concave) objective on [lo, hi].
    Used in place of ``scipy.optimize.minimize_scalar`` -- unavailable in
    this environment -- which the pre-registration named only as an
    example, not a requirement."""
    invphi = (np.sqrt(5.0) - 1.0) / 2.0  # 1/phi ~= 0.618
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = obj(c), obj(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = obj(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = obj(d)
    return float((a + b) / 2.0)


def _solve_kelly_f(simple_ret: np.ndarray, max_leverage: float) -> float:
    """argmax_{f in [0, max_leverage]} mean(log(1 + f*R)), R = SIMPLE daily
    return (not the log return r125_shared's CVaR estimator uses -- the
    Kelly criterion's growth identity `E[log(1+f*R)]` is defined over
    simple returns; using log returns here would be a silent bug). f is
    long-only, matching v4's own no-shorting philosophy (a historically
    bearish conditional sample naturally pulls the optimizer to f=0 without
    needing a separate vote multiplier)."""
    r = simple_ret

    def obj(f: float) -> float:
        g = 1.0 + f * r
        g = np.clip(g, 1e-9, None)
        return float(np.mean(np.log(g)))

    f_star = _golden_max(obj, 0.0, float(max_leverage))
    return float(np.clip(f_star, 0.0, max_leverage))


def solve_daily_kelly(close: pd.Series, frac: pd.Series, window_days: int,
                       min_samples: int, max_leverage: float) -> dict:
    """Once-per-calendar-day vote-conditional Kelly solve, pooled fallback
    on small conditional samples. Returns a dict of daily-indexed arrays
    (all strictly causal: day D's f_kelly uses only days < D, both for the
    trailing window AND for the vote state used to condition/query it --
    the vote state used is the one in effect at the START of day D, i.e.
    the closing vote state of day D-1, exactly mirroring
    ``r125_shared.rolling_cvar_daily_index``'s own ``.shift(1)`` convention).

    Kelly uses SIMPLE daily returns (see ``_solve_kelly_f``); this is
    independent of r125_shared's own CVaR estimator, which correctly uses
    log returns for a different purpose (tail-risk sizing, not growth).
    """
    daily_close = _calendar_daily_close(close)
    daily_ret = daily_close.pct_change()  # simple returns, for Kelly
    daily_state_close = frac.resample("1D").last().dropna()  # vote as of
    # the LAST bar of each day (causal: only that day's own already-closed
    # bars contribute).

    idx = daily_ret.index.intersection(daily_state_close.index)
    ret = daily_ret.reindex(idx).to_numpy()
    state_close = daily_state_close.reindex(idx).to_numpy()
    n = len(idx)

    # state in effect at the START of day D = the CLOSING vote state of
    # day D-1 -- never day D's own (not-yet-fully-observed) state.
    state_start = np.full(n, np.nan)
    state_start[1:] = state_close[:-1]

    f_kelly = np.zeros(n)
    sample_count = np.zeros(n, dtype=int)
    fallback = np.zeros(n, dtype=bool)
    state_used = np.full(n, np.nan)

    for i in range(n):
        cur_state = state_start[i]
        if not np.isfinite(cur_state):
            continue
        lo = max(0, i - window_days)
        hi = i  # exclusive of day i itself -- strictly causal
        if hi <= lo:
            continue
        w_ret = ret[lo:hi]
        w_state = state_start[lo:hi]
        finite = np.isfinite(w_ret) & np.isfinite(w_state)
        w_ret, w_state = w_ret[finite], w_state[finite]
        if w_ret.size < 5:
            continue

        cond_mask = np.isclose(w_state, cur_state, atol=1e-9)
        cond_sample = w_ret[cond_mask]
        if cond_sample.size >= min_samples:
            sample, fb = cond_sample, False
        else:
            sample, fb = w_ret, True  # pooled (unconditional) fallback
        if sample.size < 5:
            continue

        f_kelly[i] = _solve_kelly_f(sample, max_leverage)
        sample_count[i] = int(sample.size)
        fallback[i] = fb
        state_used[i] = cur_state

    return dict(index=idx, f_kelly=pd.Series(f_kelly, index=idx),
                sample_count=sample_count, fallback=fallback,
                state_used=state_used)


def _daily_to_bars(daily_series: pd.Series, bar_index: pd.DatetimeIndex) -> np.ndarray:
    """Forward-fill a daily-indexed series onto ``bar_index``'s 5m grid,
    analogous to ``r125_shared.annualized_cvar``'s own reindex/ffill
    pattern (written independently here, not imported, per the
    pre-registration's instruction not to edit the shared module). Safe
    with no extra shift: every value in ``daily_series`` at day D was
    already computed using only information available before day D began
    (see ``solve_daily_kelly``'s own ``state_start``/window construction),
    so applying it to every bar WITHIN day D introduces no lookahead."""
    if len(daily_series) == 0:
        return np.zeros(len(bar_index))
    day_of_bar = bar_index.floor("D")
    full_range = pd.date_range(daily_series.index.min(), day_of_bar.max(),
                                freq="1D", tz=daily_series.index.tz)
    by_day = daily_series.reindex(full_range).ffill()
    return by_day.reindex(day_of_bar).to_numpy()


def v4_scale_component(df: pd.DataFrame) -> np.ndarray:
    """Recompute kelly_regime_v4's own SCALE component (target_vol /
    realized_vol, hysteresis-latched between continuous and slow-anchored
    targeting) independently -- the pre-`frac` half of v4's sizing rule.
    r125_shared exposes only v4's combined, post-deadband `target` column
    (via ``v4_reference_target``), not this component alone, and this is
    needed ONLY as ``r125_shared.calibrate_target_cvar``'s reference `scale`
    array (its own docstring: "match v4's own mean scale"). Parameters are
    read off a live ``kelly_regime_v4`` instance (via
    ``r125_shared.get_strategy``) rather than hardcoded, so this can never
    silently drift from the registered strategy's own defaults."""
    v4 = r125_shared.get_strategy("kelly_regime_v4")
    close = df["close"]
    r = np.log(close).diff()
    vol = (r.ewm(span=v4.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=v4.anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(v4.target_vol / vol, v4.max_leverage)
        steady = np.minimum(v4.target_vol / slow, v4.max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(df)
    scale = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > v4.high_in else (-1 if x < v4.low_in else 0)
            elif state == 1 and x < v4.high_out:
                state = 0
            elif state == -1 and x > v4.low_out:
                state = 0
        scale[i] = full[i] if state != 0 else steady[i]
    return scale


class NovelCVaRKelly(Strategy):
    """Vote-conditional daily fractional-Kelly growth optimizer, capped by
    a pooled CVaR budget: f* = min(f_kelly(vote state), budget/realized_cvar).
    """

    name = "r125_novel_cvar_kelly"  # experiments/-only; not registered

    def __init__(self, kelly_window_days: int = KELLY_WINDOW_PRIMARY,
                 min_samples: int = MIN_SAMPLES_PRIMARY,
                 cvar_window_days: int = CVAR_WINDOW_DAYS,
                 max_leverage: float = 2.0, budget: float | None = None,
                 alpha: float = CVAR_ALPHA, horizons: tuple[int, ...] = HORIZONS,
                 band: float = BAND) -> None:
        self.kelly_window_days = kelly_window_days
        self.min_samples = min_samples
        self.cvar_window_days = cvar_window_days
        self.max_leverage = max_leverage
        self.budget = budget
        self.alpha = alpha
        self.horizons = horizons
        self.band = band
        self.warmup = int((max(kelly_window_days, cvar_window_days, max(horizons)) + 30)
                           * BARS_PER_DAY)
        self.sample_diag: dict | None = None  # populated by prepare()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        frac = vote_frac(close, self.horizons, self.band)
        diag = solve_daily_kelly(close, frac, self.kelly_window_days,
                                  self.min_samples, self.max_leverage)
        self.sample_diag = diag

        f_kelly_bar = _daily_to_bars(diag["f_kelly"], df.index)
        f_kelly_bar = np.where(np.isfinite(f_kelly_bar), f_kelly_bar, 0.0)

        cvar_bar = r125_shared.annualized_cvar(close, self.cvar_window_days,
                                                self.alpha).to_numpy()
        budget = self.budget if self.budget is not None else DEFAULT_BUDGET
        with np.errstate(divide="ignore", invalid="ignore"):
            cap_ratio = budget / cvar_bar
        cap_ratio = np.where(np.isfinite(cap_ratio) & (cap_ratio >= 0), cap_ratio, 0.0)

        target = np.minimum(f_kelly_bar, cap_ratio)
        target = np.clip(target, 0.0, self.max_leverage)
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


DEFAULT_BUDGET = 0.5  # overwritten by calibrate_budget() before any real run


def calibrate_budget(btc_inner_train: pd.DataFrame, cvar_window_days: int = CVAR_WINDOW_DAYS,
                      alpha: float = CVAR_ALPHA) -> float:
    """Calibrate the CVaR budget so mean(budget/realized_cvar) on BTC
    inner-train matches v4's own mean scale -- the same exposure-matching
    discipline R-33/R-59 established, and the conservative branch's own
    convention, via ``r125_shared.calibrate_target_cvar`` reused directly
    (never re-derived)."""
    close = btc_inner_train["close"]
    v4_scale = v4_scale_component(btc_inner_train)
    return r125_shared.calibrate_target_cvar(close, v4_scale, cvar_window_days, alpha)


# ================================================================== (2)
# NOVEL_SAMPLE_COUNTS: the pre-registered, named-failure-mode diagnostic --
# how often the transitional 1/3 and 2/3 vote states fall back to the
# pooled sample, by discrete vote state.
# ==================================================================

STATE_LABELS = {0.0: "0 (bear)", 1 / 3: "1/3", 2 / 3: "2/3", 1.0: "1 (bull)"}


def sample_diagnostics(diag: dict, restrict_index: pd.DatetimeIndex | None = None) -> dict:
    idx = diag["index"]
    mask = np.ones(len(idx), dtype=bool)
    if restrict_index is not None and len(restrict_index) > 0:
        mask = (idx >= restrict_index.min()) & (idx <= restrict_index.max())

    state_used = diag["state_used"][mask]
    sample_count = diag["sample_count"][mask]
    fallback = diag["fallback"][mask]
    valid = np.isfinite(state_used)

    out = {}
    for key, label in STATE_LABELS.items():
        m = valid & np.isclose(state_used, key, atol=1e-9)
        n_total = int(m.sum())
        n_fallback = int((m & fallback).sum())
        n_conditional = n_total - n_fallback
        counts = sample_count[m]
        out[label] = dict(
            n_days=n_total, n_conditional=n_conditional, n_fallback=n_fallback,
            fallback_rate=(n_fallback / n_total if n_total else float("nan")),
            median_sample_size=(float(np.median(counts)) if n_total else float("nan")),
            min_sample_size=(int(counts.min()) if n_total else 0),
        )
    n_no_state = int((~valid).sum())
    out["_no_state_yet"] = n_no_state
    return out


def print_sample_counts(label: str, counts: dict) -> None:
    print(f"\nNOVEL_SAMPLE_COUNTS ({label}):")
    for state, row in counts.items():
        if state == "_no_state_yet":
            continue
        print(f"  state={state:>8s}  n_days={row['n_days']:>5d}  "
              f"conditional={row['n_conditional']:>5d}  fallback={row['n_fallback']:>5d}  "
              f"fallback_rate={row['fallback_rate']:.2%}  "
              f"median_sample={row['median_sample_size']:.0f}  min_sample={row['min_sample_size']}")
    print(f"  days with no vote state yet (warmup): {counts['_no_state_yet']}")


# ================================================================== (3)
# Causal-truncation self-test on this file's OWN code.
# ==================================================================

def causal_truncation_probe(strategy_factory, df: pd.DataFrame, cut: int,
                             skip_days: int) -> bool:
    """Truncate ``df`` at bar ``cut``, recompute the full ``target`` array
    on both the full and truncated frames, and require the truncated run's
    array to be a bit-identical PREFIX of the full run's -- the R-21-class
    lookahead check this file's own conditional-sampling construction is
    exactly the kind of code that has produced real bugs in this project
    before (grepped below for any non-rolling .mean()/.std()/.quantile()/
    np.percentile() call -- there is none; every reduction in this file is
    either a positional array[lo:hi] slice with hi <= the current day, or a
    pandas .rolling()/.ewm() call)."""
    full = strategy_factory().prepare(df.copy())["target"].to_numpy()
    trunc = strategy_factory().prepare(df.iloc[:cut].copy())["target"].to_numpy()
    n_check = min(len(trunc), cut) - BARS_PER_DAY * skip_days
    if n_check <= 0:
        raise ValueError("cut too small for skip_days buffer")
    return bool(np.allclose(full[:n_check], trunc[:n_check], equal_nan=True,
                             rtol=1e-9, atol=1e-12))


# ================================================================== (4)
# B1/B3/B4/B5 harness, built from r125_shared's own b1_signal (the
# identical primary decisive cell every prior SIZE/ERR round uses).
# ==================================================================

def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run_b1(strategy_factory, df: pd.DataFrame, market) -> dict:
    return r125_shared.b1_signal(strategy_factory, df, market)


def run_step0(strategy_factory, btc: pd.DataFrame) -> dict:
    candidate_target = strategy_factory().prepare(btc.copy())["target"].to_numpy()
    v4_target = r125_shared.v4_reference_target(btc)
    return r125_shared.step0_gate(candidate_target, v4_target)


def run_b3_grid(make_factory, btc: pd.DataFrame,
                 window_grid=KELLY_WINDOW_GRID, samples_grid=MIN_SAMPLES_GRID) -> tuple[list[dict], bool]:
    rows = []
    for w in window_grid:
        for ms in samples_grid:
            for market, mname in ((r125_shared.SPOT, "spot"), (r125_shared.FUTURES, "futures_5x")):
                cell = run_b1(make_factory(w, ms), btc, market)
                rows.append(dict(kelly_window_days=w, min_samples=ms, market=mname, **cell))
    positive = sum(1 for r in rows if r["d_sharpe"] > 0)
    b3_pass = positive >= len(rows) / 2.0
    return rows, b3_pass


def run_b4(strategy_factory, eth: pd.DataFrame, btc_inner_val_spot_d_sharpe: float) -> dict:
    cell = run_b1(strategy_factory, eth, r125_shared.SPOT)
    same_sign = (cell["d_sharpe"] > 0) == (btc_inner_val_spot_d_sharpe > 0)
    cell["same_sign_as_btc"] = same_sign
    return cell


def run_b5(strategy_factory, btc: pd.DataFrame) -> list[dict]:
    rows = []
    for market, mname in ((r125_shared.SPOT_HIGH_FEE, "spot_040bps"),
                           (r125_shared.FUTURES_HIGH_FEE, "futures_5x_040bps")):
        cell = run_b1(strategy_factory, btc, market)
        rows.append(dict(market=mname, **cell))
    return rows


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    n_configs = 0
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-125 NOVEL: NovelCVaRKelly -- vote-CONDITIONAL daily fractional-Kelly "
       "growth optimizer, CVaR-budget capped")
    print("mechanism: once per calendar day, argmax_f mean(log(1+f*R)) over the trailing")
    print("window_days of daily SIMPLE returns whose OWN vote state matches the current one")
    print("(pooled fallback below min_samples), then f* = min(f_kelly, budget/realized_cvar)")
    print("with realized_cvar = r125_shared.annualized_cvar() UNCHANGED (pooled, unconditional).")
    print(f"\nKELLY_WINDOW_PRIMARY={KELLY_WINDOW_PRIMARY}d  MIN_SAMPLES_PRIMARY={MIN_SAMPLES_PRIMARY}  "
          f"CVAR_WINDOW_DAYS={CVAR_WINDOW_DAYS}d  CVAR_ALPHA={CVAR_ALPHA}")
    print(f"B3 sweep: KELLY_WINDOW_GRID={KELLY_WINDOW_GRID}  MIN_SAMPLES_GRID={MIN_SAMPLES_GRID}")

    btc, btc_label = r125_shared.load_btc_train("spot")
    max_ts_seen.append(btc.index.max())
    assert btc.index.max() < pd.Timestamp(r125_shared.OOS_START, tz="UTC")  # belt-and-suspenders; load_btc_train already asserts this internally
    print(f"\nBTC ({btc_label}, truncated < {r125_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ---------------------------------------------------- budget calibration
    hr("CALIBRATION: budget so mean(f*) on BTC inner-train matches v4's own mean scale")
    btc_inner_train = btc.loc[:r125_shared.INNER_TRAIN_END]
    budget = calibrate_budget(btc_inner_train)
    print(f"calibrated budget = {budget:.4f}  (grid search over r125_shared.calibrate_target_cvar's "
          f"own [0.05, 2.0] x 80 grid, matching v4_scale_component's mean on inner-train)")

    def make_primary():
        return NovelCVaRKelly(kelly_window_days=KELLY_WINDOW_PRIMARY, min_samples=MIN_SAMPLES_PRIMARY,
                               cvar_window_days=CVAR_WINDOW_DAYS, budget=budget)

    def make_grid(w, ms):
        def _f():
            return NovelCVaRKelly(kelly_window_days=w, min_samples=ms,
                                   cvar_window_days=CVAR_WINDOW_DAYS, budget=budget)
        return _f

    # ============================================== CAUSAL SAFETY FIRST
    hr("CAUSAL TRUNCATION PROBE (real BTC data, primary config, before ANY inner-val/ETH number)")
    cut = 400_000
    skip_days = max(KELLY_WINDOW_PRIMARY, CVAR_WINDOW_DAYS) + 3
    try:
        probe_ok = causal_truncation_probe(make_primary, btc, cut, skip_days)
        print(f"causal_truncation_probe(primary config, cut={cut}, skip_days={skip_days}): "
              f"{'PASS' if probe_ok else 'FAIL'}")
    except Exception as e:  # noqa: BLE001
        probe_ok = False
        print(f"causal_truncation_probe FAILED WITH EXCEPTION: {e}")
    print(f"\nCAUSAL SAFETY (truncation probe) PASS: {probe_ok}")
    if not probe_ok:
        hr("STOPPING: causal-safety probe failed -- no inner-validation/ETH number is trustworthy")
        elapsed = time.time() - t0
        print(f"VERDICT: NEGATIVE (causal-truncation probe failed)")
        print(f"\n[{elapsed:.0f}s]")
        return dict(verdict="NEGATIVE (causal-truncation probe failed)", probe_ok=False,
                    n_configs=0, max_ts=max(max_ts_seen))

    # ============================================================= STEP 0
    hr("STEP 0 -- non-degeneracy kill switch (R^2 vs v4's own target, BTC inner-train+inner-val)")
    step0 = run_step0(make_primary, btc)
    n_configs += 1
    print(f"R^2 vs v4_target = {step0['r2_vs_v4']:.4f}  (KILL if > 0.98)")
    print(f"Step-0 KILL: {step0['kill']}")
    if step0["kill"]:
        hr("STOPPING: Step-0 kill switch fired -- candidate is a rescaled copy of v4")
        elapsed = time.time() - t0
        print("VERDICT: NEGATIVE (Step-0 kill switch)")
        print(f"\nconfigurations evaluated (total): {n_configs}")
        print(f"\n[{elapsed:.0f}s]")
        return dict(verdict="NEGATIVE (Step-0 kill switch)", probe_ok=True, step0=step0,
                    n_configs=n_configs, max_ts=max(max_ts_seen))

    # ------------------------------------------------ NOVEL_SAMPLE_COUNTS
    hr("NOVEL_SAMPLE_COUNTS -- primary config, computed over BTC inner-train+inner-val")
    close = btc["close"]
    frac = vote_frac(close)
    diag_full = solve_daily_kelly(close, frac, KELLY_WINDOW_PRIMARY, MIN_SAMPLES_PRIMARY, 2.0)
    inner_val_days = pd.date_range(r125_shared.INNER_VAL_START, r125_shared.INNER_VAL_END,
                                    freq="1D", tz=diag_full["index"].tz)
    counts_full = sample_diagnostics(diag_full)
    counts_inner_val = sample_diagnostics(diag_full, restrict_index=inner_val_days)
    print_sample_counts("full BTC inner-train+inner-val history", counts_full)
    print_sample_counts("inner-validation period only (governs the B1 headline numbers)", counts_inner_val)

    # ============================================================= B1
    hr("B1 -- BTC signal, primary config, inner-validation, both markets")
    b1_spot = run_b1(make_primary, btc, r125_shared.SPOT)
    b1_fut = run_b1(make_primary, btc, r125_shared.FUTURES)
    n_configs += 2
    for mname, cell in (("spot", b1_spot), ("futures_5x", b1_fut)):
        print(f"  {mname:>10s}  sharpe_cand={cell['sharpe_cand']:.3f}  sharpe_v4={cell['sharpe_v4']:.3f}  "
              f"d_sharpe={cell['d_sharpe']:+.4f}  paired_diff={cell['paired_diff']:+.4f} "
              f"[{cell['paired_lo']:+.4f}, {cell['paired_hi']:+.4f}]  significant={cell['significant']}  "
              f"dd_cand={cell['dd_cand']:.1f}%  dd_v4={cell['dd_v4']:.1f}%")
    b1_pass = (b1_spot["d_sharpe"] > 0) and (b1_fut["d_sharpe"] > 0)
    print(f"B1 PASS (both markets d_sharpe > 0): {b1_pass}")

    # ============================================================= B3
    hr("B3 -- plateau: KELLY_WINDOW_GRID x MIN_SAMPLES_GRID, inner-validation, both markets")
    b3_rows, b3_pass = run_b3_grid(make_grid, btc)
    n_configs += len(b3_rows)
    for r in b3_rows:
        print(f"  window={r['kelly_window_days']:>3d}d  min_samples={r['min_samples']:>2d}  "
              f"{r['market']:>11s}  d_sharpe={r['d_sharpe']:+.4f}  significant={r['significant']}")
    print(f"B3 PASS (majority of {len(b3_rows)} cells d_sharpe > 0): {b3_pass}")

    # ============================================================= B4
    hr("B4 -- ETH falsification (pre-registered), spot only "
       "(r125_shared.load_eth_train() exposes no ETH futures loader; "
       "raw ETH perp data exists in data/ethusdt_deribit_perp_5m.csv.gz but is "
       "not wired into the frozen helpers, so futures is NOT tested here -- disclosed, not dodged)")
    eth = r125_shared.load_eth_train()
    max_ts_seen.append(eth.index.max())
    assert eth.index.max() < pd.Timestamp(r125_shared.OOS_START, tz="UTC")  # belt-and-suspenders
    print(f"ETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {r125_shared.OOS_START})")
    b4 = run_b4(make_primary, eth, b1_spot["d_sharpe"])
    n_configs += 1
    print(f"  spot  sharpe_cand={b4['sharpe_cand']:.3f}  sharpe_v4={b4['sharpe_v4']:.3f}  "
          f"d_sharpe={b4['d_sharpe']:+.4f}  paired_diff={b4['paired_diff']:+.4f} "
          f"[{b4['paired_lo']:+.4f}, {b4['paired_hi']:+.4f}]  significant={b4['significant']}  "
          f"same_sign_as_btc_spot={b4['same_sign_as_btc']}")
    b4_full = bool(b4["same_sign_as_btc"])
    print(f"B4 FULL PASS (ETH spot sign matches BTC spot; ETH futures untested, see note above): {b4_full}")

    # ============================================================= B5
    hr("B5 -- fee tier (0.40% taker), primary config, BTC inner-validation, both markets")
    b5_rows = run_b5(make_primary, btc)
    n_configs += len(b5_rows)
    base_signs = {"spot_040bps": b1_spot["d_sharpe"] > 0, "futures_5x_040bps": b1_fut["d_sharpe"] > 0}
    b5_pass = True
    for r in b5_rows:
        no_flip = (r["d_sharpe"] > 0) == base_signs[r["market"]]
        r["no_reversal"] = no_flip
        b5_pass = b5_pass and no_flip
        print(f"  {r['market']:>19s}  d_sharpe={r['d_sharpe']:+.4f}  no_reversal_vs_010bps={no_flip}")
    print(f"B5 PASS (no sign flip vs 0.10% tier, both markets): {b5_pass}")

    # ================================================================ VERDICT
    hr("VERDICT")
    print(f"causal-truncation probe: {probe_ok}")
    print(f"Step-0: PASS (not a v4 rescale, R^2={step0['r2_vs_v4']:.4f})")
    print(f"B1={b1_pass}  B3={b3_pass}  B4_full={b4_full}  B5={b5_pass}")
    all_pass = probe_ok and b1_pass and b3_pass and b4_full and b5_pass
    verdict = "PROMOTE-candidate" if all_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_pass}")
    print(f"VERDICT: {verdict}")

    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total, this branch): {n_configs} "
          f"(1 Step-0 + 2 B1 + {len(b3_rows)} B3 + 1 B4 + {len(b5_rows)} B5)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {r125_shared.OOS_START}: {max_ts < pd.Timestamp(r125_shared.OOS_START, tz='UTC')})")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(
        verdict=verdict, probe_ok=probe_ok, step0=step0, budget=budget,
        counts_full=counts_full, counts_inner_val=counts_inner_val,
        b1_spot=b1_spot, b1_fut=b1_fut, b1_pass=b1_pass,
        b3_rows=b3_rows, b3_pass=b3_pass,
        b4=b4, b4_full=b4_full,
        b5_rows=b5_rows, b5_pass=b5_pass,
        n_configs=n_configs, max_ts=max_ts,
    )


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    """Causal-truncation probe on synthetic data, run at import time --
    before any real-data number from main() is trusted. Mirrors
    r125_shared.py's own module-level self-test convention."""
    idx = pd.date_range("2017-01-01", periods=250_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(125)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                        "close": close, "volume": rng.lognormal(0, 0.5, len(idx))},
                       index=idx)

    def factory():
        return NovelCVaRKelly(kelly_window_days=90, min_samples=15, cvar_window_days=30, budget=0.5)

    cut = 150_000
    skip_days = 90 + 3
    ok = causal_truncation_probe(factory, df, cut, skip_days)
    assert ok, "causal truncation probe failed on synthetic data -- lookahead bug"

    path = factory().prepare(df.copy())["target"].to_numpy()
    assert np.isfinite(path).sum() > 1000, "produced almost no finite output on synthetic data"
    assert (path >= -1e-9).all(), "target went negative -- f should be long-only"
    assert (path <= 2.0 + 1e-9).all(), "target exceeded max_leverage"

    # golden-section sanity: a pure-upward-drift sample should push f_kelly
    # to a strictly positive interior optimum, not to either boundary.
    r_up = rng.normal(0.001, 0.01, 400)
    f = _solve_kelly_f(r_up, max_leverage=2.0)
    assert 0.0 < f < 2.0, f"expected an interior Kelly optimum for a positive-drift sample, got {f}"
    # a pure-downward-drift sample should push f_kelly to the f=0 boundary.
    r_down = rng.normal(-0.01, 0.01, 400)
    f_down = _solve_kelly_f(r_down, max_leverage=2.0)
    assert f_down < 1e-3, f"expected f_kelly~0 for a negative-drift sample, got {f_down}"


_self_test()


if __name__ == "__main__":
    main()
