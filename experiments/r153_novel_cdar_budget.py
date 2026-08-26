#!/usr/bin/env python
"""R-152 NOVEL branch: ``CdarBudgetKelly`` -- CDaR-budgeted exposure replacing
``kelly_regime_v4``'s vol-target ratio with a closed-form CDaR_0.95 budget solve.

The complete pre-registration for this round -- direction, literature
citations, non-duplication argument, falsification test and the frozen
decision rule -- lives in ``experiments/r152_shared.py``'s own module
docstring, read in full before this file was written. This file imports
ONLY from ``experiments.r152_shared`` (never edits it, never redefines its
frozen constants), never touches any "conservative"-named file, and never
reads a bar at or after ``r152_shared.OOS_START`` (2023-01-01) unless the
selection rule below (frozen, evaluated in ``main()``) is actually passed.

MECHANISM. v4's own state machine computes, at every bar,
``full = min(target_vol/vol, max_leverage)`` and
``steady = min(target_vol/slow, max_leverage)`` (``vol`` = fast EWM-std
realized vol, ``slow`` = its own EWM smoothing of ``vol`` over
``anchor_span_days``=180), latches between them via a hysteresis state
machine on ``ratio = vol/slow``, then ``desired = frac[i]*scale`` and a 10%
deadband. This branch replaces ONLY the ratio construction feeding
``full``/``steady`` -- ``frac`` (the vote), the anchors, the hysteresis
latch thresholds (``high_in=1.70, high_out=1.20, low_in=0.55, low_out=0.85``),
the deadband and ``max_leverage=2.0`` are v4's own, read off a live
``kelly_regime_v4`` instance rather than re-hardcoded (so this can never
silently drift from the registered strategy's defaults).

Chekhlov, Uryasev & Zabarankin (2005, *Int. J. Theor. Appl. Finance* 8(1),
Prop. 3) show CDaR_beta is positively homogeneous of degree 1 in the
exposure fraction for a fixed return path, so the largest ``f`` keeping
``CDaR_0.95(f * unit_returns) <= budget`` is closed-form:
``f* = budget / CDaR_0.95(unit_returns)`` -- no iterative optimizer. Here
``unit_returns[i] = frac[i] * r[i]`` (``r = log(close).diff()``), the SAME
"unit vote-scaled returns" object the conservative branch's own diagnostic
uses (r152_shared's module docstring) -- the raw signal before any sizing,
not the executed position, so there is no circularity between the derived
f* and the path used to derive it.

FULL/STEADY COMPOSITION (explicit design choice, required by
r152_shared's own instruction that "both branches keep ... the hysteresis
state machine ... byte-identical" -- the shared file names the closed-form
for a single f*, not the full/steady split, so this composition is this
file's own required elaboration, not optional invention). Two rolling CDaR
series are built from the identical ``unit_returns`` stream:
(1) ``cdar_full`` = ``r152_shared.rolling_cdar(unit_returns, window_bars)``
    at the branch's own swept window length (180/365/545 days) -- the
    fast/full side.
(2) ``cdar_slow`` = an EWM smoothing of ``cdar_full`` over v4's own
    ``anchor_span_days`` (180 days, ``min_periods=1 day``) -- mirroring,
    bar for bar, exactly how v4 derives ``slow`` from ``vol``
    (``pd.Series(vol).ewm(span=anchor_span_days*BARS_PER_DAY,
    min_periods=BARS_PER_DAY).mean()``), just applied to the CDaR series
    instead of the vol series.
``ratio = cdar_full / cdar_slow`` (guarded, NaN where ``cdar_slow<=0``)
replaces v4's own ``vol/slow`` in the IDENTICAL hysteresis latch (same
``high_in/high_out/low_in/low_out`` constants, same 3-state machine: 0
normal, +1 high breakout, -1 low breakout). ``full_f* = min(budget /
cdar_full, max_leverage)``, ``steady_f* = min(budget / cdar_slow,
max_leverage)`` (each guarded against NaN/div-by-zero -> 0 exposure, the
same guard v3/v4 apply to their own ``full``/``steady``). The state
machine picks ``scale = full_f*`` in either breakout state, ``steady_f*``
in the normal band -- byte-identical selection logic to
``KellyRegimeV3.prepare``. ``desired = frac[i] * scale``, then v4's own
10% deadband, unchanged.

``budget`` is calibrated ONCE per window length on INNER-TRAIN
(-> 2020-12-31 only) by grid search so this branch's mean exposure on
inner-train matches ``kelly_regime_v4``'s own inner-train mean exposure --
R-33's matched-risk discipline, the same idea as
``r125_shared.calibrate_target_cvar``/``r125_novel_cvar_kelly.calibrate_budget``,
adapted to this closed-form (grid search over REALIZED EXPOSURE, never a
fit to performance). The calibrated ``budget`` is then FROZEN for
inner-validation scoring and, only if eligible, for holdout.

DIAGNOSTIC B2 (run first, reported regardless): Pearson correlation, on
inner-train, between this branch's own ``cdar_full`` (at
``CDAR_WINDOW_DAYS_DEFAULT=365``, before it is turned into f*) and v4's own
realized-volatility ``vol`` column (EWM-std, ``shift(1)``'d, exactly as
``kelly_regime.KellyRegime.prepare`` computes it). ``|r| >= 0.85`` is
flagged per r152_shared's own B2 threshold.

FALSIFICATION TEST (named in r152_shared before any code ran): survives
the Monte Carlo stress windows (``scripts/stress_test.py``'s own random-
window generator and per-window evaluation, reused here -- this branch is
not ``@register``-ed, so ``scripts/stress_test.py``'s registry-name-based
``evaluate()``/``run()`` cannot call it directly; ``evaluate_instance``/
``run_stress_battery`` below are the identical logic, parameterized on a
strategy INSTANCE instead of a registry name, so the window-generation
formula, the warmup-prefix/trade_start discipline and the per-window
return/drawdown computation are reused unchanged, not reinvented).

DECISION RULE, read for clause evaluation as follows (my own explicit
reading where r152_shared states a general clause without naming which of
the 3 window lengths it applies to -- disclosed, not silently resolved):
clauses (1) matched-exposure and (2) Sharpe/drawdown are evaluated at
``CDAR_WINDOW_DAYS_DEFAULT=365`` (r152_shared's own naming of 365 as
"default", distinct from the 3-length "sweep", is the basis for treating
it as the headline config -- the same primary-vs-sweep split
``r125_novel_cvar_kelly.py`` uses, ``KELLY_WINDOW_PRIMARY`` vs
``KELLY_WINDOW_GRID``); clause (3) plateau is evaluated across all 3 swept
window lengths' Sharpe-delta sign vs control, per r152_shared verbatim.

USAGE
-----
    python experiments/r152_novel_cdar_budget.py
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

from experiments import r152_shared  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# Standard futures/spot convention used throughout this ledger (docs/VALIDATION.md
# line 9: "spot (1x, 0.10% taker) and 5x futures (0.05% taker)"; r125_shared's own
# FUTURES/FUTURES_HIGH_FEE construction, reused verbatim by every SIZE-axis round).
FUTURES = MarketSpec.futures(leverage=5.0)
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)

CDAR_BETA = r152_shared.CDAR_BETA
CDAR_WINDOW_DAYS_SWEEP = r152_shared.CDAR_WINDOW_DAYS_SWEEP
CDAR_WINDOW_DAYS_DEFAULT = r152_shared.CDAR_WINDOW_DAYS_DEFAULT
EXPOSURE_MATCH_TOL_PP = r152_shared.EXPOSURE_MATCH_TOL_PP
SHARPE_NOISE_FLOOR = r152_shared.SHARPE_NOISE_FLOOR
DRAWDOWN_IMPROVEMENT_PP = r152_shared.DRAWDOWN_IMPROVEMENT_PP
B2_CORRELATION_FLAG = r152_shared.B2_CORRELATION_FLAG
INNER_TRAIN_END = r152_shared.INNER_TRAIN_END
INNER_VAL_START = r152_shared.INNER_VAL_START
INNER_VAL_END = r152_shared.INNER_VAL_END
OOS_START = r152_shared.OOS_START


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


# ================================================================== (1)
# The mechanism itself: vote (frac, v4's own) -> two rolling CDaR series ->
# v4's own hysteresis latch (recomputed on the CDaR ratio) -> budget/CDaR
# closed-form f* -> v4's own deadband.
# ==================================================================

def vote_frac(close: pd.Series, horizons: tuple[int, ...], band: float) -> pd.Series:
    """v4's own 3-anchor latched vote, reproduced byte-for-byte (see
    ``KellyRegime.prepare``): frac in {0, 1/3, 2/3, 1}, causal."""
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


def cdar_components(close: pd.Series, window_days: int, beta: float,
                     anchor_span_days: int, horizons: tuple[int, ...], band: float) -> dict:
    """frac, the unit vote-scaled return stream, and the full/slow CDaR pair.

    ``cdar_full`` = rolling_cdar(frac*r, window_days) -- the fast/full side.
    ``cdar_slow`` = EWM(``cdar_full``, span=anchor_span_days) -- mirrors v4's
    own vol -> slow smoothing exactly, applied to CDaR instead of vol.
    """
    frac = vote_frac(close, horizons, band)
    r = np.log(close).diff()
    x_unit = (frac * r).to_numpy()
    wbars = r152_shared.window_bars(window_days)
    cdar_full = r152_shared.rolling_cdar(x_unit, wbars, beta=beta)
    cdar_slow = (pd.Series(cdar_full)
                 .ewm(span=int(anchor_span_days * BARS_PER_DAY), min_periods=BARS_PER_DAY)
                 .mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cdar_slow > 0, cdar_full / cdar_slow, np.nan)
    return dict(frac=frac.to_numpy(), x_unit=x_unit, cdar_full=cdar_full,
                cdar_slow=cdar_slow, ratio=ratio)


def hysteresis_state(ratio: np.ndarray, high_in: float, high_out: float,
                      low_in: float, low_out: float) -> np.ndarray:
    """v3/v4's own 3-state latch, computed on the CDaR ratio instead of
    vol/slow. Returns a bool array: True where the state machine selects
    the "full" (breakout) branch, False where it selects "steady"
    (normal band) -- budget-independent, so it is computed once and reused
    across every budget grid-search candidate for a given window length."""
    n = len(ratio)
    use_full = np.zeros(n, dtype=bool)
    state = 0  # 0 normal, +1 high breakout, -1 low breakout
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > high_in else (-1 if x < low_in else 0)
            elif state == 1 and x < high_out:
                state = 0
            elif state == -1 and x > low_out:
                state = 0
        use_full[i] = state != 0
    return use_full


def apply_budget(frac: np.ndarray, cdar_full: np.ndarray, cdar_slow: np.ndarray,
                  use_full: np.ndarray, budget: float, max_leverage: float,
                  deadband: float) -> np.ndarray:
    """budget/CDaR closed-form f*, v4's own hysteresis selection and
    deadband. Guarded exactly as v3/v4 guard their own full/steady arrays
    (non-finite or negative -> 0 exposure, never NaN into the position)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        full_f = budget / cdar_full
        steady_f = budget / cdar_slow
    full_f = np.where(np.isfinite(full_f), np.minimum(full_f, max_leverage), 0.0)
    steady_f = np.where(np.isfinite(steady_f), np.minimum(steady_f, max_leverage), 0.0)
    full_f = np.maximum(full_f, 0.0)
    steady_f = np.maximum(steady_f, 0.0)
    scale = np.where(use_full, full_f, steady_f)

    n = len(frac)
    target = np.zeros(n)
    pos = 0.0
    for i in range(n):
        desired = frac[i] * scale[i]
        if abs(desired - pos) > deadband:
            pos = desired
        target[i] = pos
    return target


class CdarBudgetKelly(Strategy):
    """v4's vote and hysteresis latch, CDaR-budget sized: f* = min(budget /
    CDaR_0.95(unit_returns), max_leverage), full/steady picked by the same
    latch v3/v4 use, now computed on the CDaR ratio (see module docstring).
    """

    name = "r152_novel_cdar_budget"  # experiments/-only; not registered

    def __init__(self, window_days: int, budget: float, beta: float = CDAR_BETA,
                 anchor_span_days: int | None = None, max_leverage: float | None = None,
                 deadband: float | None = None, horizons: tuple[int, ...] | None = None,
                 band: float | None = None, high_in: float | None = None,
                 high_out: float | None = None, low_in: float | None = None,
                 low_out: float | None = None) -> None:
        v4 = get_strategy("kelly_regime_v4")  # read defaults off the live registration
        self.window_days = window_days
        self.budget = budget
        self.beta = beta
        self.anchor_span_days = v4.anchor_span_days if anchor_span_days is None else anchor_span_days
        self.max_leverage = v4.max_leverage if max_leverage is None else max_leverage
        self.deadband = v4.deadband if deadband is None else deadband
        self.horizons = v4.horizons if horizons is None else horizons
        self.band = v4.band if band is None else band
        self.high_in = v4.high_in if high_in is None else high_in
        self.high_out = v4.high_out if high_out is None else high_out
        self.low_in = v4.low_in if low_in is None else low_in
        self.low_out = v4.low_out if low_out is None else low_out
        self.warmup = int((max(self.window_days, self.anchor_span_days, max(self.horizons)) + 90)
                           * BARS_PER_DAY)
        self.diag: dict | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        comp = cdar_components(close, self.window_days, self.beta, self.anchor_span_days,
                                self.horizons, self.band)
        use_full = hysteresis_state(comp["ratio"], self.high_in, self.high_out,
                                     self.low_in, self.low_out)
        target = apply_budget(comp["frac"], comp["cdar_full"], comp["cdar_slow"], use_full,
                               self.budget, self.max_leverage, self.deadband)
        df["target"] = target
        self.diag = dict(**comp, use_full=use_full)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ================================================================== (2)
# Budget calibration (inner-train only) and the B2 diagnostic.
# ==================================================================

def calibrate_budget(df_inner_train: pd.DataFrame, window_days: int, target_mean_exposure: float,
                      beta: float = CDAR_BETA, grid: np.ndarray | None = None) -> float:
    """Grid search: pick ``budget`` so mean(target) on inner-train matches
    v4's own mean(target) on the identical inner-train slice -- R-33's
    matched-risk discipline, objective = realized exposure, never Sharpe.
    ``hysteresis_state`` is budget-independent (the ratio never involves
    ``budget``), so it is computed once per window length and reused across
    every grid point -- only ``apply_budget``'s cheap O(n) pass reruns."""
    v4 = get_strategy("kelly_regime_v4")
    close = df_inner_train["close"]
    comp = cdar_components(close, window_days, beta, v4.anchor_span_days, v4.horizons, v4.band)
    use_full = hysteresis_state(comp["ratio"], v4.high_in, v4.high_out, v4.low_in, v4.low_out)

    if grid is None:
        grid = np.linspace(0.005, 1.0, 100)
    best, best_gap = float(grid[0]), np.inf
    for b in grid:
        t = apply_budget(comp["frac"], comp["cdar_full"], comp["cdar_slow"], use_full,
                          float(b), v4.max_leverage, v4.deadband)
        gap = abs(float(np.nanmean(t)) - target_mean_exposure)
        if gap < best_gap:
            best, best_gap = float(b), gap
    return best


def v4_prepared_target(df: pd.DataFrame) -> np.ndarray:
    v4 = get_strategy("kelly_regime_v4")
    out = v4.prepare(df.copy())
    return out["target"].to_numpy()


def b2_diagnostic(df_inner_train: pd.DataFrame, window_days: int = CDAR_WINDOW_DAYS_DEFAULT) -> dict:
    """Pearson correlation, inner-train, between this branch's own
    ``cdar_full`` (before it is turned into f*) and v4's own realized
    volatility ``vol`` column (EWM-std, shift(1)'d, kelly_regime.py's own
    construction)."""
    v4 = get_strategy("kelly_regime_v4")
    close = df_inner_train["close"]
    comp = cdar_components(close, window_days, CDAR_BETA, v4.anchor_span_days, v4.horizons, v4.band)
    r = np.log(close).diff()
    vol = (r.ewm(span=v4.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    cdar_full = comp["cdar_full"]
    mask = np.isfinite(cdar_full) & np.isfinite(vol)
    corr = float(np.corrcoef(cdar_full[mask], vol[mask])[0, 1]) if mask.sum() > 2 else float("nan")
    return dict(corr=corr, n=int(mask.sum()), window_days=window_days,
                flagged=bool(np.isfinite(corr) and abs(corr) >= B2_CORRELATION_FLAG))


# ================================================================== (3)
# Causal-truncation self-test on this file's OWN code.
# ==================================================================

def causal_truncation_probe(factory, df: pd.DataFrame, cut: int, extra_bars: int = 2000,
                             skip_days: int = 60) -> bool:
    """Compute ``target`` on data through bar ``cut`` and, separately,
    through bar ``cut+extra_bars`` (more history appended at the end);
    require the first run to be a prefix of the second, modulo a
    ``skip_days`` buffer at the boundary (rolling_cdar recomputes only
    once/day and forward-fills, so a boundary buffer this generous is far
    more than the one-day recompute cadence needs)."""
    full = factory().prepare(df.iloc[:cut + extra_bars].copy())["target"].to_numpy()
    trunc = factory().prepare(df.iloc[:cut].copy())["target"].to_numpy()
    n_check = cut - int(skip_days * BARS_PER_DAY)
    if n_check <= 0:
        raise ValueError("cut too small for skip_days buffer")
    return bool(np.allclose(full[:n_check], trunc[:n_check], equal_nan=True,
                             rtol=1e-9, atol=1e-12))


# ================================================================== (4)
# Selection-stage harness (inner-validation, futures 5x, BTC).
# ==================================================================

def score_arm(strategy, df: pd.DataFrame, market: MarketSpec, label: str) -> dict:
    res = run_period(strategy, df, start=INNER_VAL_START, end=INNER_VAL_END,
                      market=market, start_balance=1000.0, data_label=label)
    m = compute_metrics(res)
    mean_exposure_pct = 100.0 * float(np.nanmean(res.df["target"].to_numpy()))
    return dict(
        label=label, final_balance=m.final_balance, profit_pct=m.profit_pct,
        sharpe=m.sharpe, max_drawdown_pct=m.max_drawdown_pct,
        time_in_market_pct=m.time_in_market_pct, mean_exposure_pct=mean_exposure_pct,
        num_trades=m.num_trades, liquidated=m.liquidated,
    )


def score_holdout(strategy, df: pd.DataFrame, market: MarketSpec, label: str) -> dict:
    res = run_period(strategy, df, start=OOS_START, end=None,
                      market=market, start_balance=1000.0, data_label=label)
    m = compute_metrics(res)
    mean_exposure_pct = 100.0 * float(np.nanmean(res.df["target"].to_numpy()))
    return dict(
        label=label, final_balance=m.final_balance, profit_pct=m.profit_pct,
        sharpe=m.sharpe, max_drawdown_pct=m.max_drawdown_pct,
        time_in_market_pct=m.time_in_market_pct, mean_exposure_pct=mean_exposure_pct,
        num_trades=m.num_trades, liquidated=m.liquidated,
    )


# ================================================================== (5)
# Falsification battery: Monte Carlo stress windows (holdout only).
# Reuses scripts/stress_test.py's own window-generation formula and
# per-window evaluation logic verbatim, parameterized on a strategy
# INSTANCE (scripts/stress_test.py itself only accepts registry names,
# which this unregistered branch does not have).
# ==================================================================

def evaluate_instance(strategy, window: pd.DataFrame, eval_start: int,
                       market: MarketSpec, balance: float = 1000.0) -> dict:
    """Identical body to scripts/stress_test.py's own ``evaluate()``,
    taking a strategy instance instead of a registry name."""
    result = run_backtest(strategy, window, market, balance, trade_start=eval_start)
    equity = result.equity.to_numpy(dtype=float)
    base = equity[eval_start]
    if not np.isfinite(base) or base <= 0:
        return {"return_pct": -100.0, "max_dd_pct": 100.0, "trades": 0, "liquidated": True}
    seg = equity[eval_start:]
    start_ts = window.index[eval_start]
    return {
        "return_pct": 100.0 * (seg[-1] / base - 1.0),
        "max_dd_pct": max_drawdown_pct(seg),
        "trades": sum(1 for t in result.trades if t.entry_ts >= start_ts),
        "liquidated": result.liquidated,
    }


def run_stress_battery(candidate_factory, trials: int = 40, min_days: int = 90,
                        max_days: int = 730, seed: int = 42,
                        market: MarketSpec = FUTURES) -> pd.DataFrame:
    """Identical window-generation formula to scripts/stress_test.py's own
    ``run()`` (same rng calls, same warmup/trade_start discipline); reads
    the full dataset (including bars at/after OOS_START), which is safe
    ONLY because this is called after the branch has already passed the
    selection rule and holdout has been authorized."""
    df, label = load_dataset(ROOT / "data", "spot")
    factories = {
        "r152_novel_cdar_budget": candidate_factory,
        "kelly_regime_v4": lambda: get_strategy("kelly_regime_v4"),
        "buy_and_hold": lambda: get_strategy("buy_and_hold"),
    }
    warmup = max(f().warmup for f in factories.values()) + 10
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(trials):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        start = int(rng.integers(warmup, len(df) - length))
        window = df.iloc[start - warmup: start + length]
        eval_start = warmup
        for name, factory in factories.items():
            stats = evaluate_instance(factory(), window, eval_start, market)
            rows.append({"trial": k, "days": length // BARS_PER_DAY, "strategy": name, **stats})
    return pd.DataFrame(rows)


def summarize_stress(res: pd.DataFrame) -> dict:
    cand = res[res.strategy == "r152_novel_cdar_budget"].set_index("trial")["return_pct"]
    ctrl = res[res.strategy == "kelly_regime_v4"].set_index("trial")["return_pct"]
    bench = res[res.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
    idx = cand.index.intersection(ctrl.index)
    delta = (cand.loc[idx] - ctrl.loc[idx])
    corr = float(np.corrcoef(cand.loc[idx], ctrl.loc[idx])[0, 1]) if len(idx) > 2 else float("nan")
    return dict(
        n_trials=len(idx),
        cand_median=float(cand.median()), ctrl_median=float(ctrl.median()),
        cand_beat_hold_pct=float((cand.loc[idx] > bench.loc[idx]).mean() * 100.0),
        ctrl_beat_hold_pct=float((ctrl.loc[idx] > bench.loc[idx]).mean() * 100.0),
        cand_median_dd=float(res[res.strategy == "r152_novel_cdar_budget"]["max_dd_pct"].median()),
        ctrl_median_dd=float(res[res.strategy == "kelly_regime_v4"]["max_dd_pct"].median()),
        corr_vs_control=corr,
        mean_abs_delta_pp=float(delta.abs().mean()),
        median_delta_pp=float(delta.median()),
        # Differentiation threshold (this file's own, not pre-specified numerically
        # by r152_shared beyond "not indistinguishable"): a null result is
        # correlation >= 0.98 AND mean|delta| < 1pp -- i.e. the candidate's
        # per-window return is, to within noise, a linear copy of the control's.
        differentiated=bool(not (corr >= 0.98 and float(delta.abs().mean()) < 1.0)),
    )


# --------------------------------------------------------------------- main

def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> dict:
    t0 = time.time()
    n_configs = 0

    hr("R-152 NOVEL: CdarBudgetKelly -- CDaR-budgeted exposure "
       "(closed-form f* = budget / CDaR_0.95(unit_returns), v4's own hysteresis latch)")

    df, label = load_dataset(ROOT / "data", "spot")
    df_sel = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(df_sel)
    print(f"BTC ({label}, truncated < {OOS_START}): {len(df_sel):,} bars, "
          f"{df_sel.index[0]} -> {df_sel.index[-1]}")

    # ============================================== CAUSAL SAFETY FIRST
    hr("CAUSAL TRUNCATION PROBE (real BTC data, default window=365d, before ANY "
       "inner-validation number)")
    cut = 300_000
    probe_ok = causal_truncation_probe(
        lambda: CdarBudgetKelly(window_days=CDAR_WINDOW_DAYS_DEFAULT, budget=0.1),
        df_sel, cut=cut, extra_bars=2000, skip_days=60)
    print(f"causal_truncation_probe(cut={cut}, +2000 bars, skip_days=60): "
          f"{'PASS' if probe_ok else 'FAIL'}")
    if not probe_ok:
        hr("STOPPING: causal-safety probe failed")
        return dict(verdict="BLOCKED (causal-truncation probe failed)", n_configs=0)

    # =============================================================== B2
    hr("DIAGNOSTIC B2 (inner-train, window=365d default) -- cdar_full vs v4's own vol")
    df_train = df_sel.loc[:INNER_TRAIN_END]
    b2 = b2_diagnostic(df_train, CDAR_WINDOW_DAYS_DEFAULT)
    print(f"Pearson r(cdar_full, v4_vol) = {b2['corr']:+.4f}  (n={b2['n']:,})")
    print(f"B2 FLAG (|r| >= {B2_CORRELATION_FLAG}): {b2['flagged']}")

    # ==================================================== budget calibration
    hr("BUDGET CALIBRATION (inner-train only, per window length, matched to v4's "
       "own inner-train mean exposure)")
    v4_train_target = v4_prepared_target(df_train)
    v4_train_mean_exposure = float(np.nanmean(v4_train_target))
    print(f"v4 inner-train mean(target) = {v4_train_mean_exposure:.4f}")

    budgets = {}
    for w in CDAR_WINDOW_DAYS_SWEEP:
        b = calibrate_budget(df_train, w, v4_train_mean_exposure)
        budgets[w] = b
        print(f"  window={w:>3d}d  calibrated budget={b:.4f}")

    # ============================================= SELECTION STAGE (4 configs)
    hr("SELECTION STAGE -- inner-validation (2021-01-01 -> 2022-12-31), futures 5x, BTC")
    control = score_arm(get_strategy("kelly_regime_v4"), df_sel, FUTURES, "kelly_regime_v4 (control)")
    n_configs += 1
    print(f"  {control['label']:<28s} final=${control['final_balance']:>12,.0f} "
          f"sharpe={control['sharpe']:>6.3f} DD={control['max_drawdown_pct']:>5.1f}% "
          f"time_in_mkt={control['time_in_market_pct']:>5.1f}% "
          f"mean_exposure={control['mean_exposure_pct']:>6.1f}%")

    arms = {}
    for w in CDAR_WINDOW_DAYS_SWEEP:
        cand = CdarBudgetKelly(window_days=w, budget=budgets[w])
        row = score_arm(cand, df_sel, FUTURES, f"r152_novel_cdar_budget(window={w}d)")
        arms[w] = row
        n_configs += 1
        print(f"  {row['label']:<28s} final=${row['final_balance']:>12,.0f} "
              f"sharpe={row['sharpe']:>6.3f} DD={row['max_drawdown_pct']:>5.1f}% "
              f"time_in_mkt={row['time_in_market_pct']:>5.1f}% "
              f"mean_exposure={row['mean_exposure_pct']:>6.1f}%")

    # ============================================= FROZEN SELECTION RULE
    hr("SELECTION RULE VERDICT (r152_shared, verbatim)")
    primary = arms[CDAR_WINDOW_DAYS_DEFAULT]
    clause1 = (abs(primary["time_in_market_pct"] - control["time_in_market_pct"]) <= EXPOSURE_MATCH_TOL_PP
               and abs(primary["mean_exposure_pct"] - control["mean_exposure_pct"]) <= EXPOSURE_MATCH_TOL_PP)
    d_sharpe_primary = primary["sharpe"] - control["sharpe"]
    dd_improve_primary = control["max_drawdown_pct"] - primary["max_drawdown_pct"]
    clause2 = (d_sharpe_primary >= -SHARPE_NOISE_FLOOR) or (dd_improve_primary > DRAWDOWN_IMPROVEMENT_PP)

    deltas = {w: arms[w]["sharpe"] - control["sharpe"] for w in CDAR_WINDOW_DAYS_SWEEP}
    signs = [np.sign(d) if d != 0 else 0.0 for d in deltas.values()]
    pos_n = sum(1 for s in signs if s > 0)
    neg_n = sum(1 for s in signs if s < 0)
    clause3 = max(pos_n, neg_n) >= 2

    print(f"clause (1) matched exposure (primary=365d vs control, tol={EXPOSURE_MATCH_TOL_PP}pp): "
          f"time_in_mkt diff={abs(primary['time_in_market_pct'] - control['time_in_market_pct']):.1f}pp, "
          f"mean_exposure diff={abs(primary['mean_exposure_pct'] - control['mean_exposure_pct']):.1f}pp "
          f"-> {'PASS' if clause1 else 'FAIL'}")
    print(f"clause (2) sharpe/DD (primary=365d): d_sharpe={d_sharpe_primary:+.4f} "
          f"(floor=-{SHARPE_NOISE_FLOOR}), dd_improve={dd_improve_primary:+.2f}pp "
          f"(needs >{DRAWDOWN_IMPROVEMENT_PP}pp) -> {'PASS' if clause2 else 'FAIL'}")
    print(f"clause (3) plateau: sign(d_sharpe) per window = "
          f"{ {w: f'{deltas[w]:+.4f}' for w in CDAR_WINDOW_DAYS_SWEEP} }, "
          f"agreement={max(pos_n, neg_n)}/3 -> {'PASS' if clause3 else 'FAIL'}")

    eligible = clause1 and clause2 and clause3
    print(f"\nELIGIBLE FOR HOLDOUT: {eligible}")
    if not eligible:
        failed = [name for name, ok in (("(1) matched exposure", clause1),
                                         ("(2) sharpe/DD", clause2),
                                         ("(3) plateau", clause3)) if not ok]
        verdict = f"NEGATIVE at inner-validation (failed: {', '.join(failed)})"
        print(f"VERDICT: {verdict}")
        print(f"\nconfigurations evaluated (total): {n_configs} "
              f"(1 control + {len(arms)} window-length arms)")
        print(f"[{time.time() - t0:.0f}s]")
        return dict(verdict=verdict, b2=b2, control=control, arms=arms, budgets=budgets,
                    clause1=clause1, clause2=clause2, clause3=clause3, n_configs=n_configs)

    # ======================================= HOLDOUT (only if eligible)
    hr("HOLDOUT -- eligible; selecting window length to carry forward")
    correct_sign_windows = [w for w in CDAR_WINDOW_DAYS_SWEEP
                             if np.sign(deltas[w]) == (1.0 if pos_n >= neg_n else -1.0) and deltas[w] != 0]
    candidates_for_selection = correct_sign_windows or list(CDAR_WINDOW_DAYS_SWEEP)
    selected_w = max(candidates_for_selection, key=lambda w: arms[w]["sharpe"])
    print(f"selection rule used (this file's own inference, not specified by r152_shared.py): "
          f"best inner-validation Sharpe among the correct-sign-plateau windows "
          f"({candidates_for_selection}) -> selected window_days={selected_w}")

    df_full = df  # holdout requires reading OOS bars now that we are eligible
    _ = df_full  # (kept for clarity; used below)

    holdout_control = score_holdout(get_strategy("kelly_regime_v4"), df_full, FUTURES,
                                     "kelly_regime_v4 (control, holdout)")
    n_configs += 1
    holdout_bh = score_holdout(get_strategy("buy_and_hold"), df_full, FUTURES,
                               "buy_and_hold (holdout)")
    n_configs += 1
    holdout_candidate = score_holdout(CdarBudgetKelly(window_days=selected_w, budget=budgets[selected_w]),
                                       df_full, FUTURES,
                                       f"r152_novel_cdar_budget(window={selected_w}d, holdout)")
    n_configs += 1

    for row in (holdout_control, holdout_bh, holdout_candidate):
        print(f"  {row['label']:<40s} final=${row['final_balance']:>12,.0f} "
              f"sharpe={row['sharpe']:>6.3f} DD={row['max_drawdown_pct']:>5.1f}%")

    holdout_other_windows = {}
    for w in CDAR_WINDOW_DAYS_SWEEP:
        if w == selected_w:
            continue
        row = score_holdout(CdarBudgetKelly(window_days=w, budget=budgets[w]), df_full, FUTURES,
                             f"r152_novel_cdar_budget(window={w}d, holdout, plateau-check)")
        holdout_other_windows[w] = row
        n_configs += 1
        print(f"  {row['label']:<40s} sharpe={row['sharpe']:>6.3f}")

    hr("HOLDOUT PROMOTION-RULE CLAUSES")
    beats_bh = holdout_candidate["final_balance"] > holdout_bh["final_balance"]
    d_sharpe_holdout = holdout_candidate["sharpe"] - holdout_control["sharpe"]
    dd_improve_holdout = holdout_control["max_drawdown_pct"] - holdout_candidate["max_drawdown_pct"]
    promo_clause_a = beats_bh
    promo_clause_b = (d_sharpe_holdout > SHARPE_NOISE_FLOOR) or (dd_improve_holdout > DRAWDOWN_IMPROVEMENT_PP)

    hr("STRESS TEST (Monte Carlo falsification, this branch's named test)")
    stress_res = run_stress_battery(lambda: CdarBudgetKelly(window_days=selected_w, budget=budgets[selected_w]))
    n_configs += 1
    stress_summary = summarize_stress(stress_res)
    print(f"stress summary: {stress_summary}")
    promo_clause_c = stress_summary["differentiated"]

    holdout_deltas = {selected_w: d_sharpe_holdout}
    for w, row in holdout_other_windows.items():
        holdout_deltas[w] = row["sharpe"] - holdout_control["sharpe"]
    h_signs = [np.sign(v) if v != 0 else 0.0 for v in holdout_deltas.values()]
    h_pos = sum(1 for s in h_signs if s > 0)
    h_neg = sum(1 for s in h_signs if s < 0)
    promo_clause_d = max(h_pos, h_neg) >= 2

    print(f"(a) beats buy_and_hold, holdout, 0.05% futures fee: {promo_clause_a}")
    print(f"(b) sharpe/DD improvement over control at holdout: d_sharpe={d_sharpe_holdout:+.4f}, "
          f"dd_improve={dd_improve_holdout:+.2f}pp -> {promo_clause_b}")
    print(f"(c) survives MC stress-window falsification (differentiated from control): {promo_clause_c}")
    print(f"(d) window-length plateau holds on holdout ({max(h_pos, h_neg)}/3 agree in sign): {promo_clause_d}")

    promote = promo_clause_a and promo_clause_b and promo_clause_c and promo_clause_d
    verdict = "PROMOTE-eligible" if promote else "NEGATIVE at holdout"
    print(f"\nVERDICT: {verdict}")
    print(f"\nconfigurations evaluated (total): {n_configs}")
    print(f"[{time.time() - t0:.0f}s]")

    return dict(
        verdict=verdict, b2=b2, control=control, arms=arms, budgets=budgets,
        clause1=clause1, clause2=clause2, clause3=clause3,
        selected_w=selected_w, holdout_control=holdout_control, holdout_bh=holdout_bh,
        holdout_candidate=holdout_candidate, holdout_other_windows=holdout_other_windows,
        promo_clause_a=promo_clause_a, promo_clause_b=promo_clause_b,
        promo_clause_c=promo_clause_c, promo_clause_d=promo_clause_d,
        stress_summary=stress_summary, n_configs=n_configs,
    )


if __name__ == "__main__":
    main()
