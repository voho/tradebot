"""R-171 NOVEL branch: vol-state-conditioned ("sleeping-expert") ONS.

MECHANISM (one sentence, design doc S3): run THREE independent Online
Newton Step (ONS) leverage learners, one per state of `kelly_regime_v3`'s
own existing hysteresis vol-state machine (`state in {-1 low-vol
breakout, 0 normal, +1 high-vol breakout}`), each accumulating its own
`(a, b)` ONS state and updating ONLY on bars where its own state is
active -- the "sleeping experts" framework (Freund, Schapire, Singer &
Warmuth 1997, "Using and Combining Predictors that Specialize", STOC) --
so `desired[i] = frac[i] * b_state[i][i]`, then v4's own unmodified 10%
deadband. `frac` and the deadband are byte-identical to v4 (R-62).

This file is built ENTIRELY on top of `experiments/r171_shared.py` (frozen,
not editable) and, for the pieces the shared module intentionally does not
carry (v4's own vol-state hysteresis constants and `v4_symmetric_vol`, since
`r171_shared.py` only ever runs ONE accumulator), directly on
`experiments/r102_shared.py` -- the same ultimate source-of-truth module
`r171_shared.py` itself imports `V4_MAX_LEVERAGE`/`V4_TARGET_VOL` from.
Neither file is edited.

--------------------------------------------------------------------------
DISCLOSED METHODS NOTES (made before any real-data number was read; see
each section below for where the corresponding code lives)
--------------------------------------------------------------------------

(1) BIT-IDENTICAL STATE MACHINE. `hysteresis_state()` below is the loop
    body of `KellyRegimeV3.prepare()` / `r102_shared.conditional_target_scale()`
    copied verbatim, with the one difference that it returns the `state`
    array itself instead of the `full`/`steady`-selected scale. `_self_test()`
    (bottom of this file) proves the two are the same machine by
    reconstructing `full`/`steady`-style scale FROM this file's own `state`
    output and checking it against `r102_shared.v4_scale()` bit-for-bit on
    synthetic data -- not merely asserted by inspection.

(2) ONS CONSTANTS REUSED, NOT RE-DERIVED, ACROSS STATES AND ASSETS.
    `r171_shared.py`'s own derivation of `ONS_EPS_BTC`/`ONS_BETA_BTC` is for
    a SINGLE accumulator fit to BTC inner-train's own gradient bound `G`. No
    separate per-state or per-asset `G` is derived here: the frozen BTC
    constants are reused identically for all three (or two) accumulators,
    on both BTC and ETH -- the direct three/two-way generalization of the
    conservative arm's own disclosed policy ("used identically -- not
    re-derived -- for ETH by the conservative branch"). A per-state
    re-derivation would need per-state gradient-bound statistics computed
    from inner-train BTC only, which is feasible in principle but was not
    done: re-deriving `G` per state after seeing which state's bars produce
    a larger or smaller bound would risk exactly the kind of after-the-fact
    tuning ROUTINE.md prohibits, and the design doc's own parameter-light
    framing for this whole round argues for reusing one set of constants,
    not multiplying the number of fitted knobs by 3.

(3) ETH COVERAGE. The Bitfinex ETH replication file
    (`ethusd_bitfinex_5m.csv.gz`) covers 2016-03-09 .. 2019-12-31 -- entirely
    BEFORE `INNER_VAL_START` (2021-01-01) and therefore before both
    inner-validation and the holdout (`OOS_START`=2023-01-01). This is a
    pre-existing, disclosed data-coverage fact (already the basis of every
    prior round's "ETH-A falsification" convention, e.g. r165_novel_ewma.py's
    own D3 stage), not a choice made by this file. Consequences, stated
    explicitly rather than silently worked around:
      - `compare()`'s own `eth_replication` slice (imported unmodified from
        `r171_shared`) scores the ONE slice ETH data actually supports (its
        own full 2016-2019 history), not a `inner_train`/`inner_val` split --
        exactly how r102/r147/r165's own ETH slices already work.
      - The Monte Carlo stress-window falsification test (S3's own novel
        branch falsification test) is run on ETH's own full available span,
        not on the inner-train+inner-val BTC span used for BTC.
      - The pre-registered HOLDOUT read (S4, `start=OOS_START`) is BTC-ONLY:
        there is no ETH bar at or after 2023-01-01 anywhere in this
        project's committed data. Reported as a fall-through on the ETH leg
        of the holdout requirement, not silently omitted.

(4) KILL-SWITCH ORDER. Per R-170's own precedent ("a branch that fails
    [a gate] STOPS before Step B -- no strategy code is run past that
    point"): KS-a (per-state corner lock-in) and KS-c (composite
    exposure-artifact R^2) need ONLY the pipeline's own array outputs
    (`state`, `b_out`) and `v4_scale()` -- no backtest, no `compare()` call.
    Both are computed and checked FIRST, before any `compare()`/backtest
    call. If either trips, `compare()`, the falsification test, and the
    holdout are never run.

(5) "NOT OBVIOUSLY HOPELESS" GATE BEFORE THE HOLDOUT (S6). The one
    threshold this file adds beyond the pre-registered decision rule
    itself, used ONLY to decide whether the single permitted holdout read
    is worth spending: proceed to holdout only if kill switches (both
    configs) and the falsification test (both configs) do not trip, AND
    inner-validation `d_sharpe` on BOTH BTC and ETH (spot market) is not
    below -SHARPE_NOISE_FLOOR (-0.2) -- i.e. not already a clean, two-market
    negative before the holdout is even touched. This gate cannot promote
    anything by itself; it can only skip a doomed holdout read.

Usage::

    python experiments/r171_novel_ons_sleeping.py probe    # causal probes only
    python experiments/r171_novel_ons_sleeping.py ks       # KS-a / KS-c only
    python experiments/r171_novel_ons_sleeping.py inner    # + compare() + MC falsification, both configs
    python experiments/r171_novel_ons_sleeping.py holdout  # + the single holdout read (once)
    python experiments/r171_novel_ons_sleeping.py all      # everything, in the order above
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.inference import annualized_sharpe, paired_bootstrap  # noqa: E402

from experiments.r102_shared import (  # noqa: E402
    V4_ANCHOR_SPAN_DAYS,
    V4_HIGH_IN,
    V4_HIGH_OUT,
    V4_LOW_IN,
    V4_LOW_OUT,
    v4_symmetric_vol,
)
from experiments.r171_shared import (  # noqa: E402
    BARS_PER_DAY,
    CORNER_LOCKIN_THRESH,
    DD_REDUCTION_PROMOTE_PP,
    EXPOSURE_MATCH_BAND,
    FUTURES,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    ONS_B0,
    ONS_BETA_BTC,
    ONS_EPS_BTC,
    OOS_START,
    R2_KILL_THRESH,
    SHARPE_DELTA_PROMOTE,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TargetStrategy,
    V4_MAX_LEVERAGE,
    apply_deadband,
    asset_simple_return,
    assert_no_holdout,
    causal_truncation_probe_series,
    compare,
    corner_lockin_fraction,
    exposure_artifact_r2,
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

N_MC_WINDOWS = 24                    # design doc S3's own novel-branch falsification test
MC_SEED = 171
MC_MIN_START_OFFSET_DAYS = 80        # >= V4_WARMUP_BARS in days, matches r161_novel's own convention
FEE_TIER_STRESS = 0.0040             # design doc S4 item 4, 0.40% taker tier
BTC_INNER_VAL_SPOT_D_SHARPE_HOPELESS = -SHARPE_NOISE_FLOOR   # see methods note (5)

# Running configuration counter (ROUTINE.md requirement). Reset per `main()`
# call; module-level so every stage function can add to it.
N_CONFIGS_EVALUATED = 0


# ==========================================================================
# (1) Bit-identical hysteresis state machine (methods note (1) above).
# ==========================================================================

def hysteresis_state(ratio: np.ndarray, high_in: float = V4_HIGH_IN,
                      high_out: float = V4_HIGH_OUT, low_in: float = V4_LOW_IN,
                      low_out: float = V4_LOW_OUT) -> np.ndarray:
    """`KellyRegimeV3.prepare()`'s own state machine, factored to return
    `state` itself (verbatim loop body -- see methods note (1))."""
    ratio = np.asarray(ratio, dtype=float)
    n = len(ratio)
    state = np.empty(n, dtype=np.int8)
    st = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if st == 0:
                st = 1 if x > high_in else (-1 if x < low_in else 0)
            elif st == 1 and x < high_out:
                st = 0
            elif st == -1 and x > low_out:
                st = 0
        state[i] = st
    return state


def vol_ratio(df: pd.DataFrame) -> np.ndarray:
    """v3/v4's own `vol / slow` ratio, the hysteresis machine's input."""
    vol = v4_symmetric_vol(df)
    slow = (pd.Series(vol).ewm(span=V4_ANCHOR_SPAN_DAYS * BARS_PER_DAY,
                               min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(slow > 0, vol / slow, np.nan)


# ==========================================================================
# (2) Sleeping-experts ONS: a copy-adapted, multi-accumulator generalization
#     of r171_shared.ons_scale()'s own per-bar recursion (that function runs
#     exactly ONE accumulator over every bar; this runs one accumulator PER
#     STATE, advancing only the currently-active state's own (a, b) each
#     bar and leaving the others frozen/"sleeping" -- Freund, Schapire,
#     Singer & Warmuth 1997, applied to ONS's per-context learning). No line
#     of `r171_shared.py` is modified; this is a new loop in this file.
# ==========================================================================

def sleeping_ons(states: np.ndarray, frac: np.ndarray, ret: np.ndarray,
                  state_ids: tuple[int, ...], max_leverage: float = V4_MAX_LEVERAGE,
                  eps: float = ONS_EPS_BTC, beta: float = ONS_BETA_BTC,
                  b0: float = ONS_B0) -> np.ndarray:
    """Per-bar leverage `b_t`, one independent ONS accumulator per state id.

    Causal by the identical construction as `ons_scale`: `out[t]` is
    recorded from the ACTIVE state's `(a, b)` as carried in from bars
    `s < t` (of that state only), then `r_t = frac[t] * ret[t]` -- observed
    at the close of bar t -- advances ONLY that state's own accumulator,
    used from `t+1` onward. A state with no active bars before `t` simply
    never updates: its `b` stays at `b0` (the "sleeping" no-evidence-yet
    default), exactly as `ons_scale` starts at `b0` before its first bar.
    """
    states = np.asarray(states)
    frac = np.asarray(frac, dtype=float)
    ret = np.asarray(ret, dtype=float)
    n = len(states)
    assert len(frac) == n and len(ret) == n, (len(states), len(frac), len(ret))
    b_of = {int(s): float(b0) for s in state_ids}
    a_of = {int(s): float(eps) for s in state_ids}
    out = np.empty(n, dtype=float)
    for t in range(n):
        s = int(states[t])
        b = b_of[s]
        out[t] = b
        r_t = frac[t] * ret[t]
        if not np.isfinite(r_t):
            continue
        denom = 1.0 + b * r_t
        denom = denom if denom > 1e-6 else 1e-6
        grad = -r_t / denom
        a = a_of[s] + grad * grad
        b = b - (1.0 / beta) * grad / a
        b = min(max(b, 0.0), max_leverage)
        b_of[s] = b
        a_of[s] = a
    return out


STATE_IDS_3 = (-1, 0, 1)
STATE_IDS_2 = (0, 1)


def build_pipeline(df: pd.DataFrame, n_states: int = 3) -> dict:
    """Full causal pipeline: vote, hysteresis state, sleeping-ONS scale,
    deadbanded target. `n_states=3` is the primary (design doc S3);
    `n_states=2` collapses {-1,+1} (any breakout) into one accumulator vs
    `0` (normal) -- the plateau check (design doc S4 item 3)."""
    assert n_states in (2, 3), n_states
    frac = v4_vote_frac(df).to_numpy()
    ratio = vol_ratio(df)
    state3 = hysteresis_state(ratio)
    if n_states == 3:
        states, state_ids = state3, STATE_IDS_3
    else:
        states, state_ids = np.where(state3 == 0, 0, 1).astype(np.int8), STATE_IDS_2
    ret = asset_simple_return(df)
    b_out = sleeping_ons(states, frac, ret, state_ids)
    desired = frac * b_out
    target = apply_deadband(desired)
    return {"frac": frac, "ratio": ratio, "state3": state3, "states": states,
            "state_ids": state_ids, "b_out": b_out, "desired": desired, "target": target}


def build_target_3state(df: pd.DataFrame) -> np.ndarray:
    return build_pipeline(df, 3)["target"]


def build_target_2state(df: pd.DataFrame) -> np.ndarray:
    return build_pipeline(df, 2)["target"]


BUILDERS = {3: build_target_3state, 2: build_target_2state}


# ==========================================================================
# (3) Causal-truncation probes (design doc's item 7 / r171_shared's own
#     `_ons_build`/`_self_test` convention, generalized to the multi-
#     accumulator pipeline). Synthetic data only, same style as
#     r171_shared._self_test's own generator, different seed.
# ==========================================================================

def _synthetic_frame(seed: int, periods: int = 60_000) -> pd.DataFrame:
    idx = pd.date_range("2017-01-01", periods=periods, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


def _probe_build_3state(df: pd.DataFrame) -> np.ndarray:
    return build_pipeline(df, 3)["b_out"]


def _probe_build_2state(df: pd.DataFrame) -> np.ndarray:
    return build_pipeline(df, 2)["b_out"]


def run_causal_probes() -> dict:
    """Truncation probes on the raw `b_out` path (the quantity most exposed
    to a "sleeping accumulator peeked at a future bar" bug) for both
    configs, on two independently seeded synthetic frames."""
    out = {}
    for n_states, build in ((3, _probe_build_3state), (2, _probe_build_2state)):
        ok = True
        for seed in (171001, 171002):
            df = _synthetic_frame(seed)
            ok = ok and causal_truncation_probe_series(build, df)
        out[n_states] = ok
    return out


# ==========================================================================
# (4) KS-a / KS-c: computed from pure array outputs, no backtest -- run
#     BEFORE any compare()/falsification/holdout call (methods note (4)).
# ==========================================================================

def _inner_train_btc(btc_full: pd.DataFrame) -> pd.DataFrame:
    return btc_full[btc_full.index < pd.Timestamp(INNER_VAL_START, tz="UTC")]


def ks_a_corner_lockin(train_btc: pd.DataFrame, n_states: int) -> dict:
    """Per-state corner lock-in on inner-train BTC (design doc S2(4)(a),
    generalized per-state as the design doc's own novel-branch section
    instructs): each accumulator's own `b` path, RESTRICTED to the bars
    where that state was active, must not sit >CORNER_LOCKIN_THRESH of the
    time at 0 or max_leverage."""
    pipe = build_pipeline(train_btc, n_states)
    states, b_out = pipe["states"], pipe["b_out"]
    per_state = {}
    for s in pipe["state_ids"]:
        mask = states == s
        n_active = int(mask.sum())
        frac_lock = (corner_lockin_fraction(b_out[mask], V4_MAX_LEVERAGE)
                     if n_active else float("nan"))
        per_state[int(s)] = {
            "n_active_bars": n_active,
            "corner_lockin_fraction": frac_lock,
            "trips": bool(np.isfinite(frac_lock) and frac_lock > CORNER_LOCKIN_THRESH),
        }
    any_trips = any(v["trips"] for v in per_state.values())
    return {"n_states": n_states, "per_state": per_state, "trips": any_trips}


def ks_c_exposure_r2(train_btc: pd.DataFrame, n_states: int) -> dict:
    """Composite exposure-artifact R^2 (design doc S2(4)(c)/KS-B): the
    STATE-SELECTED `b_t` at every bar (not one accumulator alone) vs v4's
    own unmodified `scale` path, on inner-train BTC."""
    pipe = build_pipeline(train_btc, n_states)
    r2 = exposure_artifact_r2(pipe["b_out"], v4_scale(train_btc))
    return {"n_states": n_states, "r2": r2,
            "trips": bool(np.isfinite(r2) and r2 > R2_KILL_THRESH)}


def run_kill_switches(train_btc: pd.DataFrame, n_states: int) -> dict:
    a = ks_a_corner_lockin(train_btc, n_states)
    c = ks_c_exposure_r2(train_btc, n_states)
    return {"ks_a": a, "ks_c": c, "trips": bool(a["trips"] or c["trips"])}


# ==========================================================================
# (5) Monte Carlo stress-window falsification test (design doc S3's own
#     novel-branch falsification test). Pattern adapted from
#     r161_novel_online_crc_cap.py's own monte_carlo_windows(): the full
#     causal path is computed ONCE, continuously, over the asset's own
#     available pre-holdout span (never reset at a window boundary), then
#     re-run through the ordinary backtest engine over ~24 random windows.
# ==========================================================================

def _build_from_series(series: pd.Series, name: str):
    def _build(frame: pd.DataFrame) -> np.ndarray:
        return series.reindex(frame.index).to_numpy()
    _build.__name__ = name
    return _build


def _mc_span(df: pd.DataFrame, span_start: str | None, span_end: str | None) -> pd.DataFrame:
    if span_start is not None or span_end is not None:
        lo = 0 if span_start is None else int(df.index.searchsorted(pd.Timestamp(span_start, tz="UTC")))
        hi = len(df) if span_end is None else int(df.index.searchsorted(pd.Timestamp(span_end, tz="UTC"), side="right"))
        span = df.iloc[lo:hi]
    else:
        span = df
    assert_no_holdout(span, "monte_carlo_windows(): span")
    return span


def monte_carlo_windows_from_series(span: pd.DataFrame, cand_series: pd.Series, ctrl_series: pd.Series,
                                    n_states: int, market, *, n_windows: int = N_MC_WINDOWS,
                                    seed: int = MC_SEED,
                                    min_start_offset_days: int = MC_MIN_START_OFFSET_DAYS) -> dict:
    """Same window logic as `monte_carlo_windows` but takes an ALREADY-BUILT
    candidate/control target series (computed once per asset, shared across
    both SPOT and FUTURES) -- an efficiency refactor only, no change to the
    windowing/kill-outcome logic itself."""
    n = len(span)
    cand_strategy = TargetStrategy(_build_from_series(cand_series, "cand"),
                                   name=f"ons_sleeping_{n_states}s_mc", warmup=0)
    ctrl_strategy = TargetStrategy(_build_from_series(ctrl_series, "ctrl"),
                                   name="kelly_regime_v4_mc", warmup=0)

    min_start_bar = min_start_offset_days * BARS_PER_DAY
    length_min_bars = 30 * BARS_PER_DAY
    length_max_bars = min(400 * BARS_PER_DAY, n - min_start_bar - 1)
    assert length_max_bars > length_min_bars, "span too short for the MC window grid"

    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_windows):
        length_bars = int(rng.integers(length_min_bars, length_max_bars + 1))
        max_start_bar = n - length_bars
        start_bar = int(rng.integers(min_start_bar, max_start_bar + 1))
        end_bar = start_bar + length_bars - 1
        w_start, w_end = span.index[start_bar], span.index[end_bar]
        w_start_s = w_start.tz_localize(None).isoformat()
        w_end_s = w_end.tz_localize(None).isoformat()

        a = run_slice(cand_strategy, span, w_start_s, w_end_s, f"mc_{i}", market)
        b = run_slice(ctrl_strategy, span, w_start_s, w_end_s, f"mc_{i}", market)
        results.append({
            "i": i, "start": str(w_start.date()), "end": str(w_end.date()),
            "days": length_bars // BARS_PER_DAY,
            "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
            "cand_deeper": bool(a.max_drawdown_pct > b.max_drawdown_pct),
            "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
        })

    n_deeper = sum(1 for r in results if r["cand_deeper"])
    return {"n_states": n_states, "market": market.name, "n_windows": len(results),
            "results": results, "n_deeper": n_deeper,
            "trips": bool(n_deeper > n_windows // 2)}   # >12 of 24


def run_falsification(btc_full: pd.DataFrame, eth_full: pd.DataFrame, n_states: int) -> dict:
    """Both BTC (inner-train+inner-val span) and ETH (its own whole
    pre-holdout span), both market specs -- design doc S3's "on either
    market" read as BTC-vs-ETH per the task's own re-statement, with SPOT
    and FUTURES both run for robustness (see module docstring). The causal
    pipeline (the expensive part: the per-bar sleeping-ONS loop) is built
    ONCE per asset and its resulting target series is reused across both
    market specs -- target values do not depend on the market spec, only
    the backtest engine's fee/leverage handling does."""
    out = {}
    for asset_name, full_df, span_start, span_end in (
        ("BTC", btc_full, INNER_TRAIN_START, INNER_VAL_END),
        ("ETH", eth_full, None, None),
    ):
        span = _mc_span(full_df, span_start, span_end)
        pipe = build_pipeline(span, n_states)
        cand_series = pd.Series(pipe["target"], index=span.index)
        ctrl_series = pd.Series(v4_target(span), index=span.index)
        for market in (SPOT, FUTURES):
            out[(asset_name, market.name)] = monte_carlo_windows_from_series(
                span, cand_series, ctrl_series, n_states, market)
    trips = any(v["trips"] for v in out.values())
    return {"n_states": n_states, "cells": out, "trips": trips}


# ==========================================================================
# (6) Holdout (design doc S4, read ONCE, only if the gates above clear).
# ==========================================================================

def _load_btc_full() -> pd.DataFrame:
    df, _label = load_dataset(ROOT / "data", "spot")
    return df


def _load_eth_full() -> pd.DataFrame:
    return load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")


def _sharpe_paired(a_daily: np.ndarray, b_daily: np.ndarray, seed: int = 171) -> dict:
    n = min(len(a_daily), len(b_daily))
    a, b = a_daily[-n:], b_daily[-n:]
    pr = paired_bootstrap(a, b, annualized_sharpe, mean_block=30.0, n_boot=2_000, seed=seed)
    return {"d_sharpe_boot": pr.diff.point, "lo": pr.diff.lo, "hi": pr.diff.hi,
            "excl0": bool(pr.diff.lo > 0 or pr.diff.hi < 0)}


def run_holdout(n_states: int) -> dict:
    """BTC-only (methods note (3)): start=OOS_START, both markets, plus the
    0.40% fee-tier sign check. Read ONCE."""
    build = BUILDERS[n_states]
    btc = _load_btc_full()
    out = {"n_states": n_states, "markets": {}, "fee40": {}}
    cand = TargetStrategy(build, name=f"ons_sleeping_{n_states}s_holdout")
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4_holdout")
    for market in (SPOT, FUTURES):
        a = run_slice(cand, btc, OOS_START, None, "holdout", market)
        b = run_slice(ctrl, btc, OOS_START, None, "holdout", market)
        pr = paired_diff(a.daily, b.daily, seed=171)
        sp = _sharpe_paired(a.daily, b.daily)
        exp_ratio = a.mean_abs_exposure / b.mean_abs_exposure if b.mean_abs_exposure else float("nan")
        vol_ratio = a.realized_vol / b.realized_vol if b.realized_vol else float("nan")
        risk_matched = bool(EXPOSURE_MATCH_BAND[0] <= exp_ratio <= EXPOSURE_MATCH_BAND[1]
                            and EXPOSURE_MATCH_BAND[0] <= vol_ratio <= EXPOSURE_MATCH_BAND[1])
        out["markets"][market.name] = {
            "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe, "d_sharpe": a.sharpe - b.sharpe,
            "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
            "d_dd_pp": a.max_drawdown_pct - b.max_drawdown_pct,
            "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio, "risk_matched": risk_matched,
            "d_log_growth": pr.diff.point, "loggrowth_lo": pr.diff.lo, "loggrowth_hi": pr.diff.hi,
            "loggrowth_excl0": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            "d_sharpe_boot": sp["d_sharpe_boot"], "sharpe_lo": sp["lo"], "sharpe_hi": sp["hi"],
            "sharpe_excl0": sp["excl0"],
        }
        m40 = fee_at(market, FEE_TIER_STRESS)
        a40 = run_slice(cand, btc, OOS_START, None, "holdout_fee40", m40)
        b40 = run_slice(ctrl, btc, OOS_START, None, "holdout_fee40", m40)
        out["fee40"][market.name] = {
            "cand_sharpe": a40.sharpe, "ctrl_sharpe": b40.sharpe,
            "d_sharpe": a40.sharpe - b40.sharpe,
            "d_log_growth": a40.log_growth - b40.log_growth,
            "sign_matches_holdout": bool(np.sign(a40.log_growth - b40.log_growth)
                                        == np.sign(a.log_growth - b.log_growth)),
        }
    return out


def promotion_clauses(inner_rows_3: list[dict], holdout: dict | None) -> dict:
    """Design doc S4's PROMOTE clauses 1/2/4, evaluated on the HOLDOUT
    numbers (clause 1 is written for inner-validation in the design doc's
    own text but the pre-registered EVALUATION happens at holdout, per
    ROUTINE.md Step 4 -- this function reports the holdout read; the
    inner-validation ΔSharpe is reported separately, unconditionally)."""
    if holdout is None:
        return {"note": "holdout not read; see gate before S6", "clause1": False,
                "clause2": None, "clause3": None, "clause4": None}
    m = holdout["markets"]
    spot, fut = m["spot"], m["futures_5x"]
    clause1_sharpe = (spot["d_sharpe"] >= SHARPE_DELTA_PROMOTE
                      and fut["d_sharpe"] >= SHARPE_DELTA_PROMOTE)
    clause1_dd = (spot["risk_matched"] and fut["risk_matched"]
                 and (-spot["d_dd_pp"]) >= DD_REDUCTION_PROMOTE_PP
                 and (-fut["d_dd_pp"]) >= DD_REDUCTION_PROMOTE_PP)
    clause1 = bool(clause1_sharpe or clause1_dd)
    clause4_spot = holdout["fee40"]["spot"]["sign_matches_holdout"]
    clause4_fut = holdout["fee40"]["futures_5x"]["sign_matches_holdout"]
    clause4 = bool(clause4_spot and clause4_fut)
    return {"clause1": clause1, "clause1_via": "sharpe" if clause1_sharpe else
            ("dd" if clause1_dd else "neither"), "clause4": clause4}


# ==========================================================================
# main
# ==========================================================================

def hr(msg: str) -> None:
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


def main(argv: list[str]) -> None:
    global N_CONFIGS_EVALUATED
    N_CONFIGS_EVALUATED = 0
    stage = argv[1] if len(argv) > 1 else "all"

    hr("R-171 NOVEL: vol-state-conditioned (\"sleeping-expert\") ONS on kelly_regime_v4's SCALE")
    print("mechanism: 3 independent ONS accumulators, one per KellyRegimeV3's own hysteresis "
          "vol-state, each updating only on its own active bars (sleeping experts).")

    # ---- (0) bit-identity self-check of the reused state machine ----
    synth = _synthetic_frame(seed=999999, periods=40_000)
    ratio = vol_ratio(synth)
    st3 = hysteresis_state(ratio)
    full = np.minimum(0.55 / v4_symmetric_vol(synth), V4_MAX_LEVERAGE)
    slow = pd.Series(v4_symmetric_vol(synth)).ewm(
        span=V4_ANCHOR_SPAN_DAYS * BARS_PER_DAY, min_periods=BARS_PER_DAY).mean().to_numpy()
    steady = np.minimum(0.55 / slow, V4_MAX_LEVERAGE)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)
    reconstructed_scale = np.where(st3 != 0, full, steady)
    v4_scale_ref = v4_scale(synth)
    identical = np.array_equal(reconstructed_scale, v4_scale_ref)
    print(f"\nstate-machine bit-identity self-check (synthetic data): "
          f"reconstructed-scale == v4_scale: {identical}")
    assert identical, "hysteresis_state() is NOT bit-identical to v3/v4's own state machine"

    # ---- causal probes (design doc item 7) ----
    hr("CAUSAL-TRUNCATION PROBES (both configs, on the raw b_out path)")
    probes = run_causal_probes()
    for n_states, ok in probes.items():
        print(f"  n_states={n_states}: causal_truncation_probe_series == {ok}")
        assert ok, f"CAUSALITY PROBE FAILED for n_states={n_states}"
    if stage == "probe":
        return

    # ---- kill switches, BEFORE any backtest (methods note (4)) ----
    hr("KILL SWITCHES (KS-a per-state corner lock-in, KS-c composite exposure-artifact R^2), "
       "inner-train BTC, checked BEFORE any compare()/backtest call")
    btc_full = load_btc()
    eth_full = load_eth()
    train_btc = _inner_train_btc(btc_full)
    print(f"inner-train BTC: {len(train_btc):,} bars ({train_btc.index[0]} .. {train_btc.index[-1]})")

    ks_by_config: dict[int, dict] = {}
    for n_states in (3, 2):
        ks = run_kill_switches(train_btc, n_states)
        ks_by_config[n_states] = ks
        print(f"\n  n_states={n_states}:")
        for s, v in ks["ks_a"]["per_state"].items():
            print(f"    KS-a state={s:+d}: n_active_bars={v['n_active_bars']:,}  "
                 f"corner_lockin_fraction={v['corner_lockin_fraction']:.4f}  "
                 f"trips(>{CORNER_LOCKIN_THRESH})={v['trips']}")
        print(f"    KS-c composite R^2 vs v4_scale: {ks['ks_c']['r2']:.4f}  "
             f"trips(>{R2_KILL_THRESH})={ks['ks_c']['trips']}")
        print(f"    => n_states={n_states} kill-switch verdict: "
             f"{'TRIPPED' if ks['trips'] else 'clear'}")

    any_ks_trips = any(v["trips"] for v in ks_by_config.values())
    if stage == "ks":
        return
    if any_ks_trips:
        print("\n*** KILL SWITCH TRIPPED for at least one config -- STOPPING before compare()/"
             "falsification/holdout, per R-170's own precedent. No further strategy backtest "
             "is run. ***")
        hr("VERDICT")
        print("NEGATIVE (kill switch tripped). See KS output above.")
        print(f"\nconfigurations evaluated: {N_CONFIGS_EVALUATED}")
        return

    # ---- inner-train / inner-validation compare(), both configs (design doc item 2 + 5) ----
    hr("INNER-TRAIN / INNER-VALIDATION vs kelly_regime_v4 -- compare(), both configs, "
       "BTC (inner_train, inner_val) + ETH (eth_replication), both markets")
    rows_by_config: dict[int, list[dict]] = {}
    for n_states in (3, 2):
        rows = compare(BUILDERS[n_states], label=f"ons_sleeping_{n_states}state",
                       btc=btc_full, eth=eth_full)
        rows_by_config[n_states] = rows
        N_CONFIGS_EVALUATED += len(rows)
        print(f"\n  n_states={n_states}:")
        print_rows(rows)

    if stage == "inner_only":
        print(f"\nconfigurations evaluated so far: {N_CONFIGS_EVALUATED}")
        return

    # ---- falsification test: Monte Carlo stress windows, both configs ----
    hr("FALSIFICATION TEST (design doc S3): Monte Carlo stress-window drawdown survival, "
      f"{N_MC_WINDOWS} windows, BTC + ETH, SPOT + FUTURES, both configs")
    mc_by_config: dict[int, dict] = {}
    for n_states in (3, 2):
        mc = run_falsification(btc_full, eth_full, n_states)
        mc_by_config[n_states] = mc
        print(f"\n  n_states={n_states}:")
        for (asset, market_name), cell in mc["cells"].items():
            print(f"    {asset:>3s}/{market_name:<12s} n_deeper={cell['n_deeper']}/{cell['n_windows']}  "
                 f"trips(>12)={cell['trips']}")
            N_CONFIGS_EVALUATED += cell["n_windows"]
        print(f"    => n_states={n_states} falsification verdict: "
             f"{'TRIPPED' if mc['trips'] else 'survives'}")

    any_mc_trips = any(v["trips"] for v in mc_by_config.values())

    hr("PLATEAU CHECK (design doc S4 item 3): does the 3-state result's direction hold "
      "for the collapsed 2-state version?")
    def _mean_d_sharpe(rows: list[dict], slice_name: str) -> float:
        cells = [r["d_sharpe"] for r in rows if r["slice"] == slice_name]
        return float(np.mean(cells)) if cells else float("nan")
    for slice_name in ("inner_train", "inner_val", "eth_replication"):
        d3 = _mean_d_sharpe(rows_by_config[3], slice_name)
        d2 = _mean_d_sharpe(rows_by_config[2], slice_name)
        same_sign = bool(np.sign(d3) == np.sign(d2)) if np.isfinite(d3) and np.isfinite(d2) else False
        print(f"  {slice_name:16s}: 3-state mean d_sharpe={d3:+.3f}  2-state mean d_sharpe={d2:+.3f}  "
             f"same sign={same_sign}")

    if stage == "inner":
        hr("VERDICT (through falsification/plateau; holdout not requested for this stage)")
        print(f"kill switches: {'clear' if not any_ks_trips else 'TRIPPED'}   "
             f"falsification: {'survives' if not any_mc_trips else 'TRIPPED'}")
        print(f"\nconfigurations evaluated: {N_CONFIGS_EVALUATED}")
        return

    if any_mc_trips:
        print("\n*** FALSIFICATION TEST TRIPPED for at least one config -- STOPPING before "
             "holdout, per the design doc's pre-registered exact kill outcome. ***")
        hr("VERDICT")
        print("NEGATIVE (falsification test tripped). See MC output above.")
        print(f"\nconfigurations evaluated: {N_CONFIGS_EVALUATED}")
        return

    # ---- "not obviously hopeless" gate before spending the one holdout read ----
    spot_btc_val = next(r for r in rows_by_config[3] if r["slice"] == "inner_val" and r["market"] == "spot")
    eth_spot = next(r for r in rows_by_config[3] if r["slice"] == "eth_replication" and r["market"] == "spot")
    hopeless = (spot_btc_val["d_sharpe"] < BTC_INNER_VAL_SPOT_D_SHARPE_HOPELESS
               and eth_spot["d_sharpe"] < BTC_INNER_VAL_SPOT_D_SHARPE_HOPELESS)
    hr("HOLDOUT GATE (methods note (5)): kill switches clear, falsification survives, "
      "inner-validation not a clean two-market negative")
    print(f"  BTC inner_val spot d_sharpe={spot_btc_val['d_sharpe']:+.3f}   "
         f"ETH eth_replication spot d_sharpe={eth_spot['d_sharpe']:+.3f}   "
         f"'obviously hopeless'={hopeless}")

    if stage != "all" and stage != "holdout":
        print(f"\nconfigurations evaluated: {N_CONFIGS_EVALUATED}")
        return

    if hopeless:
        print("\n*** Inner-validation is a clean two-market negative -- the holdout read is "
             "skipped rather than spent on a doomed config. This is a diagnostic skip, NOT one "
             "of the pre-registered kill switches. ***")
        hr("VERDICT")
        print("NEGATIVE (inner-validation hopeless on both markets; holdout not consulted).")
        print(f"\nconfigurations evaluated: {N_CONFIGS_EVALUATED}")
        return

    # ---- the ONE holdout read (design doc S4), primary (3-state) config only ----
    hr("HOLDOUT (read ONCE, PRIMARY = 3-state config, BTC only -- ETH has no holdout-period "
      "data, methods note (3))")
    holdout3 = run_holdout(3)
    N_CONFIGS_EVALUATED += 2 * 2   # 2 markets x (holdout + fee40)
    for market_name, v in holdout3["markets"].items():
        print(f"\n  {market_name}: d_sharpe={v['d_sharpe']:+.3f} "
             f"[boot {v['sharpe_lo']:+.3f},{v['sharpe_hi']:+.3f}] excl0={v['sharpe_excl0']}   "
             f"d_dd_pp={v['d_dd_pp']:+.2f}   exp_ratio={v['exposure_ratio']:.2f} "
             f"vol_ratio={v['vol_ratio']:.2f} risk_matched={v['risk_matched']}   "
             f"d_log_growth={v['d_log_growth']:+.4f} "
             f"[{v['loggrowth_lo']:+.4f},{v['loggrowth_hi']:+.4f}] excl0={v['loggrowth_excl0']}")
        f40 = holdout3["fee40"][market_name]
        print(f"    fee40: d_sharpe={f40['d_sharpe']:+.3f}  d_log_growth={f40['d_log_growth']:+.4f}  "
             f"sign_matches_holdout={f40['sign_matches_holdout']}")

    clauses = promotion_clauses(rows_by_config[3], holdout3)
    hr("PROMOTION-RULE CLAUSES (design doc S4), evaluated on the holdout read")
    print(f"  clause 1 (dSharpe>=+0.2 both mkts, or matched-exposure DD reduction>=5pp both mkts): "
         f"{clauses['clause1']}  (via: {clauses['clause1_via']})")
    print(f"  clause 2 (survives falsification test)                                        : "
         f"{not any_mc_trips}")
    print(f"  clause 3 (plateau: 3-state and 2-state agree in direction)                     : "
         f"see PLATEAU CHECK above")
    print(f"  clause 4 (sign does not reverse at 0.40% fee tier)                             : "
         f"{clauses['clause4']}")

    verdict = "PROMOTE-CANDIDATE" if (clauses["clause1"] and not any_mc_trips
                                      and clauses["clause4"]) else "NEGATIVE"
    hr("FINAL VERDICT (assessment against the frozen decision rule; not a promotion decision)")
    print(f"  {verdict}")
    print(f"\nconfigurations evaluated: {N_CONFIGS_EVALUATED}")


if __name__ == "__main__":
    main(sys.argv)
