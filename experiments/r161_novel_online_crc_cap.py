#!/usr/bin/env python
"""R-161 NOVEL branch: ONLINE Conformal Risk Control (Angelopoulos, Bates,
Fisch, Lei & Schuster 2024, "Conformal Risk Control", ICLR, arXiv:2208.02814,
Sec. 4's distribution-shift extension) tracking a multiplicative cap
``lambda_t in [0,1]`` on ``kelly_regime_v4``'s own unmodified
``frac*scale`` output, CONTINUOUSLY, day by day -- as opposed to the sibling
CONSERVATIVE branch's periodic batch RCPS recalibration
(``r161_conservative_rcps_cap.py``, not touched, not read, by this file).

Full citation trail, the "not a duplicate of" argument against every prior
round, the pre-registered decision rule, the two branches' respective
falsification tests, and the five named failure modes are all in
``r161_shared.py``'s own module docstring (read in full before this file was
written); not re-derived here. This file never edits, and never reads a bar
at or after ``r161_shared.OOS_START`` from, ``r161_shared.py`` or any other
file.

EXACT CONSTRUCTION -- the online recursion (Angelopoulos et al. 2024 Sec 4's
monotonized generalization of Gibbs & Candes' (2021) ACI recursion from
tracking MISCOVERAGE to tracking an arbitrary bounded MONOTONE loss;
``r161_shared.loss_at`` is exactly such a loss, monotone non-increasing in
lambda for fixed (exposure_prev, ret)):

    lambda_{d+1} = clip(lambda_d + eta * (alpha - loss_d(lambda_d)), 0, 1)

where ``loss_d(lambda_d) = r161_shared.loss_at(exposure_prev_d, ret_d,
lambda_d, tau)`` is evaluated on day d's own REALIZED row of
``r161_shared.calibration_frame(df)`` (exposure DECIDED at the close of day
d-1, paired with day d's own realized return -- already strictly causal by
construction). ``lambda_0 = 1.0`` (v4's own baseline: no cap until evidence
accumulates). Intuition: when day d's realized loss exceeds the alpha
target, the recursion pushes lambda DOWN (de-risk); when it is under
target, lambda drifts back UP toward 1.0 (no unnecessary de-risking) --
self-correcting toward the target exceedance rate as data arrives, with no
scheduled batch refit and no exchangeability assumption, unlike the
conservative branch's periodic RCPS/Hoeffding-UCB search.

GRID (9 configs total, pre-registered in the dispatch message, not tuned
after seeing any real-data number):
  1. ``itertools.product(TAU_GRID, ALPHA_GRID, (0.01, 0.02))`` for eta --
     2x2x2 = 8 configs, lambda_0 = 1.0 throughout. eta=0.01 is PRIMARY
     (paired with TAU_PRIMARY, ALPHA_PRIMARY); eta=0.02 is a faster-
     adaptation robustness check.
  2. One extra WARM-START variant at (TAU_PRIMARY, ALPHA_PRIMARY,
     eta=0.01) with lambda_0 = 0.8 instead of 1.0 -- does a more
     conservative initial condition change the converged behavior
     materially, or does the recursion wash it out quickly (the
     literature's own convergence claim)?
  Total: 9 configs, each scored via ``compare()`` on BOTH markets (spot,
  futures_5x) and all three slices (inner_train, inner_val,
  eth_replication) -- 9 x 3 x 2 = 54 compare() cells.

Every config's daily lambda path is built as a ``pd.Series`` indexed by day
(the value used AT day d is fixed by data through day d-1 only -- causal by
construction of the recursion itself) and wired through v4's own unmodified
scale/deadband machinery via ``r161_shared.build_capped_target``.

FALSIFICATION TEST (pre-registered in r161_shared.py's own docstring,
NOVEL branch): survive the Monte Carlo stress windows. ``scripts/
stress_test.py`` itself is strategy-registry-based (``get_strategy(name)``
by string, run against a fixed dataset), not directly callable against an
arbitrary ``df -> np.ndarray`` candidate, so this file adapts the same
underlying primitives (``TargetStrategy`` / ``run_slice`` / ``run_period``)
the SAME way this project's own prior novel branch already solved exactly
this problem -- see ``experiments/r147_novel_bma_ladder.py``'s
``monte_carlo_windows()``: compute the candidate's full causal path ONCE,
continuously, over the whole inner-train + inner-validation span (so the
recursion accumulates real history from the span's own true start and is
NEVER reset cold at a window boundary -- the online method's own claimed
advantage over a periodic refit), then re-run each of ~24 random
(start, length) windows through the ordinary fee-charging backtest engine,
warmup forced to 0 since the path is already fully computed. As a second,
independent confirmation this file ALSO runs a paired stationary-block-
bootstrap resample (``tradebot.inference.stationary_bootstrap_indices``,
the same primitive ``r161_shared``'s own dependency chain already uses via
``paired_diff``) directly over the PRIMARY config's inner-validation daily
return series vs v4's, and reports what fraction of resampled blocks the
observed direction of advantage survives in.

USAGE
-----
    python experiments/r161_novel_online_crc_cap.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.inference import (  # noqa: E402
    annualized_sharpe,
    stationary_bootstrap_indices,
    total_log_return,
)

from experiments.r161_shared import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_PRIMARY,
    BARS_PER_DAY,
    CONST_CAP_R2_THRESH,
    ETH_SLICE_NAME,
    FUTURES,
    GATE_MIN_BINDING_FRACTION,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    SLICES,
    SPOT,
    TAU_GRID,
    TAU_PRIMARY,
    TargetStrategy,
    assert_no_holdout,
    binding_fraction,
    build_capped_target,
    calibration_frame,
    causal_truncation_probe_series,
    compare,
    constant_cap_r2,
    load_btc,
    load_eth,
    loss_at,
    paired_diff,
    print_rows,
    run_slice,
    synthetic_known_tail_frame,
    v4_target,
)

# Novel-branch-owned hyperparameter grid: the online learning-rate ``eta``,
# not part of r161_shared's frozen (shared) TAU_GRID/ALPHA_GRID since it has
# no analogue in the conservative (periodic RCPS) branch.
ETA_GRID: tuple[float, ...] = (0.01, 0.02)
ETA_PRIMARY = ETA_GRID[0]
LAMBDA0_DEFAULT = 1.0
LAMBDA0_WARM_START = 0.8

Config = tuple[float, float, float, float]  # (tau, alpha, eta, lambda0)

CONFIGS: list[Config] = [
    (tau, alpha, eta, LAMBDA0_DEFAULT)
    for tau, alpha, eta in itertools.product(TAU_GRID, ALPHA_GRID, ETA_GRID)
]
CONFIGS.append((TAU_PRIMARY, ALPHA_PRIMARY, ETA_PRIMARY, LAMBDA0_WARM_START))
assert len(CONFIGS) == 9, len(CONFIGS)
PRIMARY_CONFIG: Config = (TAU_PRIMARY, ALPHA_PRIMARY, ETA_PRIMARY, LAMBDA0_DEFAULT)
assert PRIMARY_CONFIG in CONFIGS


def hr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def cfg_label(cfg: Config) -> str:
    tau, alpha, eta, lambda0 = cfg
    tag = f"t{tau:g}_a{alpha:g}_e{eta:g}"
    if abs(lambda0 - LAMBDA0_DEFAULT) > 1e-12:
        tag += f"_l0{lambda0:g}"
    return tag


# ================================================================== (1)
# The online recursion itself.
# ==================================================================

def online_crc_lambda_path(cal: pd.DataFrame, tau: float, alpha: float, eta: float,
                           lambda0: float = LAMBDA0_DEFAULT) -> pd.Series:
    """Angelopoulos et al. (2024) Sec 4's online CRC recursion, applied to
    ``r161_shared.loss_at`` day by day over ``cal`` (already causal:
    ``cal.index[d]``'s row pairs exposure DECIDED at the close of day d-1
    with day d's own realized return).

    ``lam_path[d]`` (the value STORED/returned at day d) is the lambda that
    was fixed by data through day d-1 only -- it is what is USED to size
    day d's exposure. The update ``lam_{d+1} = clip(lam_d + eta*(alpha -
    loss_d(lam_d)), 0, 1)`` consumes day d's own just-realized loss (using
    the SAME lam_d that sized day d) to produce lam_{d+1}, which then sizes
    day d+1 -- strictly causal, one day at a time, no lookahead, no
    scheduled batch refit."""
    idx = cal.index
    n = len(idx)
    if n == 0:
        return pd.Series(dtype=float)
    exposure_prev = cal["exposure_prev"].to_numpy(dtype=float)
    ret = cal["ret"].to_numpy(dtype=float)
    lam_path = np.empty(n, dtype=float)
    lam_cur = float(lambda0)
    for i in range(n):
        lam_path[i] = lam_cur
        loss_i = float(loss_at(exposure_prev[i:i + 1], ret[i:i + 1], lam_cur, tau)[0])
        lam_cur = float(np.clip(lam_cur + eta * (alpha - loss_i), 0.0, 1.0))
    return pd.Series(lam_path, index=idx)


# ================================================================== (2)
# Wire the recursion through v4's own unmodified scale/deadband pipeline,
# with content-keyed caching -- r160/r147's own convention -- so the 2x
# (market) redundancy inside one compare() call, and repeated kill-switch /
# falsification-test reads of the SAME (df, hyperparameters) pair, do not
# re-run the pure-Python per-day recursion needlessly. Keyed on CONTENT
# (index bounds + a close-price fingerprint), never on id(df), because
# run_period slices a fresh DataFrame object per call even when the
# content is identical -- and the causal truncation probe feeds two
# SAME-SHAPED, SAME-INDEX frames (full vs tail-perturbed) that must NOT
# collide in the cache, or the probe would silently pass vacuously.
# ==================================================================

_COMPONENT_CACHE: dict[tuple, tuple] = {}


def _frame_key(df: pd.DataFrame) -> tuple:
    idx = df.index
    c = df["close"].to_numpy()
    return (int(idx[0].value), int(idx[-1].value), len(df), float(np.sum(c)), float(c[-1]))


def online_components(df: pd.DataFrame, tau: float, alpha: float, eta: float,
                      lambda0: float = LAMBDA0_DEFAULT) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    key = (_frame_key(df), round(tau, 6), round(alpha, 6), round(eta, 6), round(lambda0, 6))
    if key not in _COMPONENT_CACHE:
        cal = calibration_frame(df)
        lam_path = online_crc_lambda_path(cal, tau, alpha, eta, lambda0)
        target = build_capped_target(df, lam_path)
        _COMPONENT_CACHE[key] = (cal, lam_path, target)
    return _COMPONENT_CACHE[key]


def make_build_target(tau: float, alpha: float, eta: float, lambda0: float = LAMBDA0_DEFAULT):
    def _build(df: pd.DataFrame) -> np.ndarray:
        _cal, _lam, target = online_components(df, tau, alpha, eta, lambda0)
        return target
    _build.__name__ = f"online_crc_{cfg_label((tau, alpha, eta, lambda0))}"
    return _build


def fixed_lambda_builder(df: pd.DataFrame, lam_value: float = 0.7) -> np.ndarray:
    """Wiring-only probe builder, mirroring ``r161_shared._self_test``'s own
    pattern exactly: a FIXED, already-"calibrated" constant lambda, so the
    causal truncation probe isolates ``build_capped_target``'s plumbing
    from whether the calibration ALGORITHM (the online recursion) is
    itself causal (checked separately, below, via the full pipeline)."""
    idx_d = calibration_frame(df).index
    lam_d = pd.Series(lam_value, index=idx_d)
    return build_capped_target(df, lam_d)


def _ts(x, tz: str = "UTC"):
    if x is None:
        return None
    ts = pd.Timestamp(x)
    return ts.tz_localize(tz) if ts.tzinfo is None else ts


def windowed_cal_and_lambda(df: pd.DataFrame, start: str | None, end: str | None,
                            tau: float, alpha: float, eta: float, lambda0: float,
                            warmup: int = TargetStrategy.warmup) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce EXACTLY what ``compare()``/``run_slice()``/``run_period()``
    hand a candidate build function for one slice -- an 80-day (v4's own
    slowest anchor) warmup PREFIX before ``start``, then the window itself
    -- so the online recursion here is initialized/reset the SAME way the
    actual backtest that produces the Sharpe/growth numbers resets it
    (this project's r147-r160 shared convention: every slice is its own
    independent evaluation window, not a continuation of the previous
    slice's accumulated state). Returns ``(cal, lam_path)`` restricted to
    ``[start, end]`` itself (the warmup-only prefix rows are dropped)."""
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    prefix = int(min(lo, max(warmup, 0)))
    frame = df.iloc[lo - prefix: hi]
    cal = calibration_frame(frame)
    lam_path = online_crc_lambda_path(cal, tau, alpha, eta, lambda0)
    start_ts, end_ts = _ts(start), _ts(end)
    if start_ts is not None:
        cal = cal.loc[cal.index >= start_ts]
        lam_path = lam_path.loc[lam_path.index >= start_ts]
    if end_ts is not None:
        cal = cal.loc[cal.index <= end_ts]
        lam_path = lam_path.loc[lam_path.index <= end_ts]
    return cal, lam_path


def slice_exceedance_rates(df: pd.DataFrame, start: str | None, end: str | None,
                           tau: float, alpha: float, eta: float, lambda0: float) -> tuple[float, float]:
    """Candidate (varying lambda) vs v4-uncapped (lambda==1.0 always)
    realized tail-loss exceedance rate on the SAME (per-slice-reset)
    exposure/return series compare() itself would trade for this slice."""
    cal, lam_path = windowed_cal_and_lambda(df, start, end, tau, alpha, eta, lambda0)
    if len(cal) == 0:
        return float("nan"), float("nan")
    exposure_prev = cal["exposure_prev"].to_numpy(dtype=float)
    ret = cal["ret"].to_numpy(dtype=float)
    lam = lam_path.reindex(cal.index).to_numpy(dtype=float)
    cand_rate = float(np.mean(loss_at(exposure_prev, ret, lam, tau)))
    ctrl_rate = float(np.mean(loss_at(exposure_prev, ret, 1.0, tau)))
    return cand_rate, ctrl_rate


# ================================================================== (3)
# Kill switches (A1 binding, A2 non-degeneracy diagnostic), computed on the
# FULL pre-holdout frame per instrument (== inner-train + inner-validation
# combined for BTC, the whole eth_replication span for ETH) -- a SINGLE
# CONTINUOUS run of the recursion, r160's own convention for this exact
# diagnostic (``kill_switch_row`` there runs on the full pre-holdout BTC
# frame directly, not through compare()'s per-slice reset).
# ==================================================================

def kill_switch_row(btc: pd.DataFrame, eth: pd.DataFrame, cfg: Config) -> dict:
    tau, alpha, eta, lambda0 = cfg
    _cal_b, lam_btc, _t_b = online_components(btc, tau, alpha, eta, lambda0)
    _cal_e, lam_eth, _t_e = online_components(eth, tau, alpha, eta, lambda0)
    bf_btc = binding_fraction(lam_btc)
    bf_eth = binding_fraction(lam_eth)
    r2_btc = constant_cap_r2(lam_btc)
    r2_eth = constant_cap_r2(lam_eth)
    a1_pass = bool(bf_btc >= GATE_MIN_BINDING_FRACTION or bf_eth >= GATE_MIN_BINDING_FRACTION)
    # NOTE (disclosed, not fixed -- r161_shared.py is frozen/read-only):
    # constant_cap_r2's ss_tot uses np.std(const) where `const` is built via
    # np.full_like(x, np.mean(x)); floating-point round-off makes that std a
    # tiny nonzero epsilon (~1e-16) rather than exactly 0 for any real
    # (non-bitwise-constant) lambda path, so the function's own
    # `np.std(b) == 0` guard never triggers and it divides by a near-zero
    # ss_tot, producing numerically unstable, uninterpretable values (huge
    # magnitude, either sign, or NaN depending on rounding direction) rather
    # than a meaningful R^2 -- observed directly below and in the synthetic
    # self-test. Reported honestly per config; NOT treated as evidence
    # either way. The raw lambda_std/min/max below are the credible
    # non-degeneracy evidence this branch actually relies on.
    degenerate_btc = np.isfinite(r2_btc) and r2_btc > CONST_CAP_R2_THRESH
    degenerate_eth = np.isfinite(r2_eth) and r2_eth > CONST_CAP_R2_THRESH
    lam_btc_arr = lam_btc.to_numpy() if len(lam_btc) else np.array([np.nan])
    lam_eth_arr = lam_eth.to_numpy() if len(lam_eth) else np.array([np.nan])
    return dict(cfg=cfg, label=cfg_label(cfg),
               binding_fraction_btc=bf_btc, binding_fraction_eth=bf_eth,
               constant_cap_r2_btc=r2_btc, constant_cap_r2_eth=r2_eth,
               a1_pass=a1_pass, degenerate_btc=bool(degenerate_btc),
               degenerate_eth=bool(degenerate_eth),
               lam_btc_mean=float(np.mean(lam_btc_arr)), lam_eth_mean=float(np.mean(lam_eth_arr)),
               lam_btc_std=float(np.std(lam_btc_arr)), lam_eth_std=float(np.std(lam_eth_arr)),
               lam_btc_min=float(np.min(lam_btc_arr)), lam_btc_max=float(np.max(lam_btc_arr)),
               lam_eth_min=float(np.min(lam_eth_arr)), lam_eth_max=float(np.max(lam_eth_arr)))


# ================================================================== (4)
# Calibration self-test: synthetic data with a KNOWN injected tail-event
# rate. Checks (i) the achieved exceedance rate (second half of the
# series, after the recursion has had time to converge) against the KNOWN
# ground truth, and (ii) how fast the running lambda stabilizes -- the
# online method's own claimed advantage over a periodic batch refit.
# ==================================================================

def calibration_self_test(tau: float = TAU_PRIMARY, alpha: float = ALPHA_PRIMARY,
                          eta: float = ETA_PRIMARY, lambda0: float = LAMBDA0_DEFAULT,
                          n: int = 400_000, true_tail_prob: float = 0.05,
                          seed: int = 161) -> dict:
    synth = synthetic_known_tail_frame(n=n, true_tail_prob=true_tail_prob, seed=seed)
    assert_no_holdout(synth, "calibration_self_test(): synthetic")
    cal = calibration_frame(synth)
    lam_path = online_crc_lambda_path(cal, tau, alpha, eta, lambda0)

    n_days = len(lam_path)
    half = n_days // 2
    exposure_prev = cal["exposure_prev"].to_numpy(dtype=float)
    ret = cal["ret"].to_numpy(dtype=float)
    lam = lam_path.to_numpy(dtype=float)

    second_half_rate = float(np.mean(loss_at(exposure_prev[half:], ret[half:], lam[half:], tau)))
    long_run_avg = float(np.mean(lam[half:]))

    # Convergence speed: first day index after which lambda NEVER again
    # strays outside +/-20% of its own eventual (second-half) long-run
    # average.
    band = 0.20 * abs(long_run_avg) if abs(long_run_avg) > 1e-9 else 0.05
    within = np.abs(lam - long_run_avg) <= band
    bad_idx = np.where(~within)[0]
    conv_day = int(bad_idx[-1] + 1) if len(bad_idx) > 0 else 0

    return dict(n_bars=n, n_days=n_days, tau=tau, alpha=alpha, eta=eta, lambda0=lambda0,
               true_tail_prob=true_tail_prob, second_half_exceedance_rate=second_half_rate,
               long_run_avg_lambda=long_run_avg, convergence_day=conv_day,
               binding_fraction=binding_fraction(lam_path),
               constant_cap_r2=constant_cap_r2(lam_path),
               lam_std=float(np.std(lam)), lam_min=float(np.min(lam)), lam_max=float(np.max(lam)))


# ================================================================== (5)
# Falsification test: Monte Carlo stress windows, adapted from this
# project's own prior novel branch (r147_novel_bma_ladder.py's
# monte_carlo_windows()) since scripts/stress_test.py is registry-based
# and not directly callable on an arbitrary df -> np.ndarray candidate.
# The full causal recursion is computed ONCE, continuously, over the whole
# inner-train + inner-validation span -- never reset at a window boundary
# -- so the online method is stress-tested the way its own literature
# claims it should behave well: adapting through a regime shift a fixed
# calibration window did not anticipate, not restarting cold at one.
# ==================================================================

def _build_from_series(series: pd.Series, name: str):
    def _build(frame: pd.DataFrame) -> np.ndarray:
        return series.reindex(frame.index).to_numpy()
    _build.__name__ = name
    return _build


def monte_carlo_windows(btc: pd.DataFrame, cfg: Config, market, n_windows: int = 24,
                        seed: int = 161, min_start_offset_days: int = 80) -> dict:
    tau, alpha, eta, lambda0 = cfg
    span_start = pd.Timestamp(INNER_TRAIN_START, tz="UTC")
    span_end = pd.Timestamp(INNER_VAL_END, tz="UTC")
    span = btc.loc[(btc.index >= span_start) & (btc.index <= span_end)]
    assert_no_holdout(span, "monte_carlo_windows(): span")
    n = len(span)

    _cal, lam_path, target_arr = online_components(span, tau, alpha, eta, lambda0)
    novel_target = pd.Series(target_arr, index=span.index)
    v4_full_target = pd.Series(v4_target(span), index=span.index)

    cand_strategy = TargetStrategy(_build_from_series(novel_target, "online_crc_mc"),
                                   name=f"online_crc_mc_{cfg_label(cfg)}", warmup=0)
    ctrl_strategy = TargetStrategy(_build_from_series(v4_full_target, "v4_precomp"),
                                   name="kelly_regime_v4_mc", warmup=0)

    min_start_bar = min_start_offset_days * BARS_PER_DAY
    length_min_bars = 30 * BARS_PER_DAY
    length_max_bars = min(400 * BARS_PER_DAY, n - min_start_bar - 1)
    assert length_max_bars > length_min_bars, "span too short for the requested MC window grid"

    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_windows):
        length_bars = int(rng.integers(length_min_bars, length_max_bars + 1))
        max_start_bar = n - length_bars
        start_bar = int(rng.integers(min_start_bar, max_start_bar + 1))
        end_bar = start_bar + length_bars - 1

        w_start = span.index[start_bar]
        w_end = span.index[end_bar]
        w_start_s = w_start.tz_localize(None).isoformat()
        w_end_s = w_end.tz_localize(None).isoformat()

        a = run_slice(cand_strategy, span, w_start_s, w_end_s, f"mc_window_{i}", market)
        b = run_slice(ctrl_strategy, span, w_start_s, w_end_s, f"mc_window_{i}", market)
        d_sharpe = a.sharpe - b.sharpe
        d_loggrowth = a.log_growth - b.log_growth
        d_dd = a.max_drawdown_pct - b.max_drawdown_pct
        results.append(dict(
            i=i, start=str(w_start.date()), end=str(w_end.date()),
            length_days=length_bars // BARS_PER_DAY,
            cand_sharpe=a.sharpe, ctrl_sharpe=b.sharpe, d_sharpe=d_sharpe,
            cand_loggrowth=a.log_growth, ctrl_loggrowth=b.log_growth, d_loggrowth=d_loggrowth,
            cand_dd=a.max_drawdown_pct, ctrl_dd=b.max_drawdown_pct, d_dd=d_dd,
            cand_wins_sharpe=bool(d_sharpe > 0), cand_wins_loggrowth=bool(d_loggrowth > 0),
            cand_wins_dd=bool(d_dd < 0),
        ))

    win_frac_sharpe = float(np.mean([r["cand_wins_sharpe"] for r in results]))
    win_frac_loggrowth = float(np.mean([r["cand_wins_loggrowth"] for r in results]))
    win_frac_dd = float(np.mean([r["cand_wins_dd"] for r in results]))
    return dict(cfg=cfg, market=market.name, n_windows=len(results), results=results,
               win_frac_sharpe=win_frac_sharpe, win_frac_loggrowth=win_frac_loggrowth,
               win_frac_dd=win_frac_dd, majority_pass=bool(win_frac_sharpe > 0.5))


def print_mc_summary(mc: dict) -> None:
    print(f"config={cfg_label(mc['cfg'])}  market={mc['market']}  n_windows={mc['n_windows']}  "
         f"candidate wins on Sharpe: {mc['win_frac_sharpe']:.1%}   "
         f"candidate wins on log-growth: {mc['win_frac_loggrowth']:.1%}   "
         f"candidate wins on drawdown: {mc['win_frac_dd']:.1%}")


def stationary_bootstrap_falsification(a_daily: np.ndarray, b_daily: np.ndarray,
                                       n_boot: int = 24, mean_block: float = 30.0,
                                       seed: int = 161) -> dict:
    """Second, independent falsification check: a paired stationary-block-
    bootstrap resample of the candidate/control daily-return SERIES (not a
    resample of historical windows of the raw price path, as
    ``monte_carlo_windows`` above does) -- the same
    ``tradebot.inference.stationary_bootstrap_indices`` primitive
    ``r161_shared``'s own ``paired_diff`` already builds on. Reports what
    fraction of the resampled blocks the OBSERVED (whole-period) direction
    of advantage on d_sharpe / d_log_growth survives in."""
    n = min(len(a_daily), len(b_daily))
    a, b = np.asarray(a_daily[-n:], dtype=float), np.asarray(b_daily[-n:], dtype=float)
    obs_d_sharpe = float(annualized_sharpe(a) - annualized_sharpe(b))
    obs_d_loggrowth = float(total_log_return(a) - total_log_return(b))

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    a_res, b_res = a[idx], b[idx]
    d_sharpe_res = annualized_sharpe(a_res) - annualized_sharpe(b_res)
    d_loggrowth_res = total_log_return(a_res) - total_log_return(b_res)

    survive_sharpe = (float(np.mean(np.sign(d_sharpe_res) == np.sign(obs_d_sharpe)))
                      if obs_d_sharpe != 0 else float("nan"))
    survive_loggrowth = (float(np.mean(np.sign(d_loggrowth_res) == np.sign(obs_d_loggrowth)))
                         if obs_d_loggrowth != 0 else float("nan"))
    return dict(n_boot=n_boot, n=n, obs_d_sharpe=obs_d_sharpe, obs_d_loggrowth=obs_d_loggrowth,
               d_sharpe_resampled=d_sharpe_res, d_loggrowth_resampled=d_loggrowth_res,
               survive_sharpe_frac=survive_sharpe, survive_loggrowth_frac=survive_loggrowth)


# ================================================================== (6)
# Decision rule -- r161_shared.py's own PROMOTE-CANDIDATE clauses (a)/(b)/
# (c), restated verbatim, evaluated per config per market on inner-val:
#   (a) candidate's realized tail-loss exceedance rate STRICTLY LOWER than
#       v4's own uncapped rate, at the SAME tau;
#   (b) paired bootstrap 95% CI on d_log_growth excludes zero on the
#       positive side, OR d_sharpe >= +0.2, OR a risk-matched drawdown
#       improvement;
#   (c) the SAME direction of whichever of (a)/(b) fired reproduces on the
#       Monte Carlo stress-window falsification test -- not inverted.
# ==================================================================

def evaluate_ab(rows: list[dict], btc: pd.DataFrame, eth: pd.DataFrame, cfg: Config) -> dict:
    tau, alpha, eta, lambda0 = cfg
    inner_val = {r["market"]: r for r in rows if r["slice"] == "inner_val"}
    per_market = {}
    for market_name, df, start, end in (
        ("spot", btc, INNER_VAL_START, INNER_VAL_END),
        ("futures_5x", btc, INNER_VAL_START, INNER_VAL_END),
    ):
        iv = inner_val.get(market_name)
        if iv is None:
            per_market[market_name] = dict(clause_a=False, clause_b=False, note="missing row")
            continue
        cand_rate, ctrl_rate = slice_exceedance_rates(df, start, end, tau, alpha, eta, lambda0)
        clause_a = bool(np.isfinite(cand_rate) and np.isfinite(ctrl_rate) and cand_rate < ctrl_rate)
        sharpe_edge = iv["d_sharpe"] >= SHARPE_NOISE_FLOOR
        dd_edge = bool(iv["risk_matched"] and iv["d_dd"] < 0)
        boot_edge = bool(iv["boot_lo"] > 0)
        clause_b = bool(boot_edge or sharpe_edge or dd_edge)
        fired_via = "boot" if boot_edge else ("sharpe" if sharpe_edge else ("dd" if dd_edge else None))
        per_market[market_name] = dict(clause_a=clause_a, clause_b=clause_b, fired_via=fired_via,
                                       cand_exceedance=cand_rate, ctrl_exceedance=ctrl_rate,
                                       ab_pass=bool(clause_a and clause_b))
    return per_market


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    hr("R-161 NOVEL: Online Conformal Risk Control cap on kelly_regime_v4's "
      "SCALE output -- continuous, day-by-day tracking")
    print("mechanism: lambda_{d+1} = clip(lambda_d + eta*(alpha - loss_d(lambda_d)), 0, 1),")
    print("Angelopoulos et al. (2024) Sec 4's online CRC recursion, applied to r161_shared's own")
    print("causal per-day tail-loss functional on v4's UNMODIFIED frac*scale. lambda_0 = 1.0 (no")
    print("cap until evidence accumulates); a fresh recursion per compare() slice (the r147-r160")
    print("shared per-slice-reset convention) for the Sharpe/growth numbers, and one CONTINUOUS,")
    print("never-reset run over the whole inner-train+inner-val span for kill switches and the")
    print("Monte Carlo falsification test -- the online method's own claimed advantage.")
    print(f"\nTAU_GRID (frozen, r161_shared)  = {TAU_GRID}   TAU_PRIMARY   = {TAU_PRIMARY}")
    print(f"ALPHA_GRID (frozen, r161_shared) = {ALPHA_GRID}   ALPHA_PRIMARY = {ALPHA_PRIMARY}")
    print(f"ETA_GRID (this branch's own)     = {ETA_GRID}       ETA_PRIMARY   = {ETA_PRIMARY}")
    print(f"WARM-START variant: lambda_0={LAMBDA0_WARM_START} at (tau={TAU_PRIMARY}, "
         f"alpha={ALPHA_PRIMARY}, eta={ETA_PRIMARY})")
    print(f"\nCONFIGS ({len(CONFIGS)} total):")
    for cfg in CONFIGS:
        print(f"    {cfg_label(cfg)}  (lambda0={cfg[3]})")

    btc = load_btc()
    eth = load_eth()
    max_ts_seen += [btc.index.max(), eth.index.max()]
    assert_no_holdout(btc, "main(): btc")
    assert_no_holdout(eth, "main(): eth")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
         f"{btc.index[0]} -> {btc.index[-1]}")
    print(f"ETH (Bitfinex replication, truncated < {OOS_START}): {len(eth):,} bars, "
         f"{eth.index[0]} -> {eth.index[-1]}")

    # ============================================================= STEP 1
    hr("STEP 1 -- CAUSALITY: truncation probes")
    print("(1a) wiring-only probe (FIXED lambda=0.7), mirroring r161_shared's own self-test:")
    try:
        probe_wiring_ok = causal_truncation_probe_series(fixed_lambda_builder, btc)
        print(f"     causal_truncation_probe_series(fixed_lambda_builder, btc): PASS")
    except AssertionError as e:
        probe_wiring_ok = False
        print(f"     causal_truncation_probe_series(fixed_lambda_builder, btc): FAIL: {e}")

    print("(1b) FULL pipeline probe (online recursion computed fresh per truncation), PRIMARY config:")
    build_primary = make_build_target(*PRIMARY_CONFIG)
    try:
        probe_full_ok = causal_truncation_probe_series(build_primary, btc)
        print(f"     causal_truncation_probe_series({build_primary.__name__}, btc): PASS")
    except AssertionError as e:
        probe_full_ok = False
        print(f"     causal_truncation_probe_series({build_primary.__name__}, btc): FAIL: {e}")

    probe_ok = bool(probe_wiring_ok and probe_full_ok)

    # ============================================================= STEP 2
    hr("STEP 2 -- CALIBRATION SELF-TEST: synthetic KNOWN-tail-rate data "
      "(n=400,000 bars, true_tail_prob=0.05, seed=161), PRIMARY (tau,alpha,eta)")
    calib = calibration_self_test(*PRIMARY_CONFIG, n=400_000, true_tail_prob=0.05, seed=161)
    print(f"  n_bars={calib['n_bars']:,}  n_days={calib['n_days']:,}  "
         f"tau={calib['tau']}  alpha={calib['alpha']}  eta={calib['eta']}  lambda0={calib['lambda0']}")
    print(f"  KNOWN true_tail_prob (ground truth)           = {calib['true_tail_prob']:.4f}")
    print(f"  ACHIEVED exceedance rate, 2nd half of series  = {calib['second_half_exceedance_rate']:.4f}")
    print(f"  |achieved - true_tail_prob|                   = "
         f"{abs(calib['second_half_exceedance_rate'] - calib['true_tail_prob']):.4f}")
    print(f"  long-run average lambda (2nd half)            = {calib['long_run_avg_lambda']:.4f}")
    print(f"  binding_fraction (whole path)                 = {calib['binding_fraction']:.4f}")
    print(f"  lambda std / min / max (whole path)           = {calib['lam_std']:.4f} / "
         f"{calib['lam_min']:.4f} / {calib['lam_max']:.4f}")
    print(f"  constant_cap_r2 (whole path)                  = {calib['constant_cap_r2']:.4g}  "
         f"(numerically unstable for non-constant input -- see STEP 3's caveat; not treated as "
         f"evidence, the std/range above is)")
    print(f"  CONVERGENCE SPEED: first day after which lambda never again strays outside +/-20% "
         f"of its own eventual long-run average: day {calib['convergence_day']} "
         f"(of {calib['n_days']:,} total calibration days)")

    # ============================================================= STEP 3
    hr(f"STEP 3 -- FULL SWEEP: {len(CONFIGS)} configs, compare() over inner_train/inner_val/"
      "eth_replication, SPOT+FUTURES, plus A1/A2 kill switches and (a)/(b) decision-rule clauses")
    print("CAVEAT (disclosed, not fixed -- r161_shared.py's constant_cap_r2 is frozen/read-only):")
    print("  its ss_tot uses np.std(const) on a np.full_like(x, mean(x)) array, which floating-")
    print("  point round-off makes a tiny nonzero epsilon rather than exactly 0 for any real")
    print("  (non-bitwise-constant) lambda path -- the function's own zero-guard never triggers,")
    print("  so it divides by a near-zero denominator and returns numerically unstable, huge-")
    print("  magnitude or NaN values below, NOT a meaningful R^2. Reported per config for the")
    print("  record; the credible non-degeneracy evidence is the raw lambda mean/std/range and")
    print("  binding_fraction printed alongside it.")
    all_rows: dict[Config, list[dict]] = {}
    kill_rows: dict[Config, dict] = {}
    ab_by_config: dict[Config, dict] = {}
    for cfg in CONFIGS:
        tau, alpha, eta, lambda0 = cfg
        label = cfg_label(cfg)
        print(f"\n--- config {label}  (tau={tau}, alpha={alpha}, eta={eta}, lambda0={lambda0}) ---")

        kr = kill_switch_row(btc, eth, cfg)
        kill_rows[cfg] = kr
        print(f"  A1/A2 (full pre-holdout, continuous, per-instrument):")
        print(f"    BTC: binding_fraction={kr['binding_fraction_btc']:.4f} "
             f"(need >={GATE_MIN_BINDING_FRACTION})  constant_cap_r2={kr['constant_cap_r2_btc']:.4g} "
             f"(degenerate if >{CONST_CAP_R2_THRESH}; SEE NUMERICAL-INSTABILITY CAVEAT ABOVE)  "
             f"lambda: mean={kr['lam_btc_mean']:.4f} std={kr['lam_btc_std']:.4f} "
             f"range=[{kr['lam_btc_min']:.4f},{kr['lam_btc_max']:.4f}]  "
             f"{'DEGENERATE' if kr['degenerate_btc'] else 'time-varying (by raw std/range)'}")
        print(f"    ETH: binding_fraction={kr['binding_fraction_eth']:.4f} "
             f"(need >={GATE_MIN_BINDING_FRACTION})  constant_cap_r2={kr['constant_cap_r2_eth']:.4g} "
             f"(degenerate if >{CONST_CAP_R2_THRESH}; SEE NUMERICAL-INSTABILITY CAVEAT ABOVE)  "
             f"lambda: mean={kr['lam_eth_mean']:.4f} std={kr['lam_eth_std']:.4f} "
             f"range=[{kr['lam_eth_min']:.4f},{kr['lam_eth_max']:.4f}]  "
             f"{'DEGENERATE' if kr['degenerate_eth'] else 'time-varying (by raw std/range)'}")
        print(f"    A1 (binding on >=1 instrument, i.e. >=1 'market' by this mechanism's own "
             f"construction -- lambda calibration is price-data-only, market-spec-independent): "
             f"{'PASS' if kr['a1_pass'] else 'FAIL'}")

        build_fn = make_build_target(tau, alpha, eta, lambda0)
        rows = compare(build_fn, label=label, btc=btc, eth=eth, markets=(SPOT, FUTURES), include_eth=True)
        print_rows(rows)
        n_configs += len(rows)
        all_rows[cfg] = rows

        ab = evaluate_ab(rows, btc, eth, cfg)
        ab_by_config[cfg] = ab
        for market, d in ab.items():
            if "note" in d:
                print(f"    decision-rule[{market}]: {d['note']}")
                continue
            print(f"    decision-rule[{market}]: (a) cand_exceedance={d['cand_exceedance']:.4f} "
                 f"vs ctrl_exceedance={d['ctrl_exceedance']:.4f} -> a={d['clause_a']}   "
                 f"(b) fired_via={d['fired_via']} -> b={d['clause_b']}   ab_pass={d['ab_pass']}")

    # ============================================================= STEP 4
    hr("STEP 4 -- FALSIFICATION TEST: MONTE CARLO STRESS WINDOWS (pre-registered NOVEL-branch "
      "falsification test), PRIMARY config, both markets, ~24 windows")
    mc_by_market: dict[str, dict] = {}
    for market in (SPOT, FUTURES):
        mc = monte_carlo_windows(btc, PRIMARY_CONFIG, market, n_windows=24, seed=161,
                                 min_start_offset_days=80)
        mc_by_market[market.name] = mc
        print_mc_summary(mc)
        n_configs += mc["n_windows"]

    hr("STEP 4b -- SECOND, INDEPENDENT FALSIFICATION CHECK: paired stationary-block-bootstrap "
      "resample (tradebot.inference.stationary_bootstrap_indices) of PRIMARY config's own "
      "inner-validation DAILY RETURNS vs v4's, both markets")
    boot_by_market: dict[str, dict] = {}
    cand_primary = TargetStrategy(make_build_target(*PRIMARY_CONFIG), name="online_crc_primary_boot")
    ctrl_primary = TargetStrategy(v4_target, name="kelly_regime_v4_boot")
    for market in (SPOT, FUTURES):
        a = run_slice(cand_primary, btc, INNER_VAL_START, INNER_VAL_END, "inner_val_boot", market)
        b = run_slice(ctrl_primary, btc, INNER_VAL_START, INNER_VAL_END, "inner_val_boot", market)
        bf = stationary_bootstrap_falsification(a.daily, b.daily, n_boot=24, mean_block=30.0, seed=161)
        boot_by_market[market.name] = bf
        print(f"  {market.name:>10s}  n_days={bf['n']}  observed d_sharpe={bf['obs_d_sharpe']:+.4f}  "
             f"observed d_log_growth={bf['obs_d_loggrowth']:+.4f}")
        print(f"             fraction of {bf['n_boot']} resampled blocks matching observed sign: "
             f"d_sharpe={bf['survive_sharpe_frac']:.1%}  d_log_growth={bf['survive_loggrowth_frac']:.1%}")

    # Extend the falsification test to any OTHER config that clears (a)+(b) on BOTH
    # markets, so clause (c) can be evaluated for it too (PRIMARY is always run above).
    ab_pass_configs = [cfg for cfg, ab in ab_by_config.items()
                      if all(ab.get(m, {}).get("ab_pass", False) for m in ("spot", "futures_5x"))]
    extra_configs = [c for c in ab_pass_configs if c != PRIMARY_CONFIG]
    if extra_configs:
        hr("STEP 4c -- EXTRA CONFIGS clearing (a)+(b) on both markets: running their own "
          "Monte Carlo falsification test too")
        for cfg in extra_configs:
            for market in (SPOT, FUTURES):
                mc = monte_carlo_windows(btc, cfg, market, n_windows=24, seed=161,
                                         min_start_offset_days=80)
                mc_by_market[f"{cfg_label(cfg)}::{market.name}"] = mc
                print_mc_summary(mc)
                n_configs += mc["n_windows"]
    else:
        print("\nNo non-primary config cleared (a)+(b) on both markets -- only the PRIMARY "
             "config's Monte Carlo falsification test is needed to fully evaluate the decision rule.")

    # ============================================================= STEP 5
    hr("STEP 5 -- FULL DECISION-RULE EVALUATION (clauses a/b/c), every config, both markets")
    decision_by_config: dict[Config, dict] = {}
    for cfg in CONFIGS:
        ab = ab_by_config[cfg]
        per_market = {}
        for market_name in ("spot", "futures_5x"):
            d = ab.get(market_name, {})
            if not d.get("ab_pass", False):
                per_market[market_name] = dict(passes=False, reason="a+b not both true")
                continue
            mc_key = market_name if cfg == PRIMARY_CONFIG else f"{cfg_label(cfg)}::{market_name}"
            mc = mc_by_market.get(mc_key)
            if mc is None:
                per_market[market_name] = dict(passes=False, reason="no MC falsification run for this cell")
                continue
            fired_via = d.get("fired_via")
            if fired_via == "dd":
                clause_c = mc["win_frac_dd"] > 0.5
            else:
                clause_c = mc["majority_pass"]  # win_frac_sharpe > 0.5
            per_market[market_name] = dict(passes=bool(clause_c), clause_c=bool(clause_c),
                                           win_frac_sharpe=mc["win_frac_sharpe"],
                                           fired_via=fired_via)
        both_pass = all(per_market[m].get("passes", False) for m in ("spot", "futures_5x"))
        decision_by_config[cfg] = dict(per_market=per_market, both_markets_pass=both_pass)
        print(f"  {cfg_label(cfg):>28s}  spot={per_market['spot']}  futures_5x={per_market['futures_5x']}  "
             f"BOTH_MARKETS_PASS={both_pass}")

    # ============================================================= STEP 6
    hr("STEP 6 -- CONFIGURATION COUNT")
    n_sweep = sum(len(v) for v in all_rows.values())
    n_calib = 1  # one synthetic self-test cell (PRIMARY hyperparameters)
    n_kill = len(CONFIGS) * 2  # BTC + ETH continuous diagnostic run, per config
    n_mc = sum(mc["n_windows"] for mc in mc_by_market.values())
    n_boot = sum(bf["n_boot"] for bf in boot_by_market.values())
    total = n_sweep + n_calib + n_kill + n_mc + n_boot
    print(f"main sweep cells ({len(CONFIGS)} configs x 3 slices x 2 markets):  {n_sweep}")
    print(f"calibration self-test cells (synthetic, PRIMARY):                 {n_calib}")
    print(f"kill-switch diagnostic runs (config x {{BTC,ETH}}):                 {n_kill}")
    print(f"Monte Carlo stress-window cells:                                  {n_mc}")
    print(f"stationary-bootstrap falsification resamples:                     {n_boot}")
    print(f"TOTAL CONFIGURATIONS/CELLS EVALUATED (this file):                 {total}")
    print(f"(n_configs accumulator, compare()+MC cells only, matching r160's own counting "
         f"convention): {n_configs}")

    # ============================================================= VERDICT
    hr("VERDICT")
    promote_configs = [cfg for cfg, d in decision_by_config.items() if d["both_markets_pass"]]
    any_bind = any(kr["a1_pass"] for kr in kill_rows.values())
    any_degenerate = any(kr["degenerate_btc"] or kr["degenerate_eth"] for kr in kill_rows.values())
    calib_err = abs(calib["second_half_exceedance_rate"] - calib["true_tail_prob"])
    calib_ok = calib_err < 0.03  # reported honestly either way; not a formal kill switch

    print(f"causal truncation probes PASS (wiring + full pipeline):  {probe_ok}")
    print(f"A1 kill switch (binds on >=1 config/instrument):         {any_bind}")
    print(f"A2 diagnostic (any config R^2-degenerate, i.e. constant):{any_degenerate}  "
         f"(diagnostic only, per module docstring failure mode (5) -- not a kill switch)")
    print(f"calibration self-test: |achieved - true_tail_prob| = {calib_err:.4f}  "
         f"({'within' if calib_ok else 'OUTSIDE'} an informal 0.03 sanity band, reported either way)")
    print(f"configs clearing full PROMOTE-CANDIDATE decision rule (a+b+c, both markets): "
         f"{[cfg_label(c) for c in promote_configs] if promote_configs else 'NONE'}")

    verdict = ("PROMOTE-candidate (gate clears; holdout may be consulted)"
              if (probe_ok and any_bind and promote_configs) else "NEGATIVE")
    print(f"\nVERDICT: {verdict}")
    if verdict.startswith("PROMOTE"):
        print("\nGate clears causality + A1 kill switch + full decision rule (a+b+c) on >=1 config --")
        print("holdout MAY be consulted per docs/ROUTINE.md step 4. This file does NOT itself")
        print("read OOS_START; that is a separate, explicit step left to the operator.")
    else:
        print("\nGate does NOT clear the full decision rule on any swept config -- per")
        print("docs/ROUTINE.md's own discipline, the holdout is precious and is NOT touched. No")
        print("bar at or after OOS_START is read anywhere in this file.")

    max_ts = max(max_ts_seen)
    print(f"\nmax timestamp read anywhere in this branch: {max_ts}  "
         f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, probe_ok=probe_ok, calib=calib, all_rows=all_rows,
               kill_rows=kill_rows, ab_by_config=ab_by_config, mc_by_market=mc_by_market,
               boot_by_market=boot_by_market, decision_by_config=decision_by_config,
               n_configs=total, any_bind=any_bind, any_degenerate=any_degenerate,
               max_ts=max_ts, verdict=verdict, promote_configs=promote_configs)


if __name__ == "__main__":
    main()
