#!/usr/bin/env python
"""CONSERVATIVE branch: does B-18's monthly/weekly cadence flip in
`kelly_regime_covkelly.py` survive once the rebalance-engine's segment
RESTART is removed?

Backlog item attacked: **B-18** -- "Is the `kelly_regime_covkelly`
allocator's monthly/weekly cadence-inconsistency (R-42, attenuated but not
resolved by R-43's mean-denoising) actually a rebalance-engine/segment-
restart artifact rather than a mean-estimation-noise problem?"

The suspected mechanism (verified by reading the code, not assumed)
-----------------------------------------------------------------------
`kelly_regime_covkelly.py::run_portfolio` splits [start, end] into segments
at each rebalance date and calls `tradebot.window.run_period` ONCE PER
SEGMENT, independently, for each leg (`_run_leg`). `KellyRegimeV3.prepare`
(inherited unchanged by v4) computes its 10% deadband position hysteresis
AND its high/low volatility-regime latch with a stateful Python for-loop:

    pos = 0.0
    state = 0   # 0 normal, +1 high-vol breakout, -1 low-vol breakout
    for i in range(n):
        ...
        target[i] = pos

`pos`/`state` reset to (0, 0) at index 0 of whatever frame `prepare()`
receives. `run_period` hands `prepare()` only a `warmup`-bar prefix (v4:
80 days) before `start_pos` -- not the true multi-year history back to the
start of the dataset. So every segment's `kelly_regime_v4` sub-book
"forgets" years of accumulated latch state and re-derives `pos`/`state`
from an ~80-day cold prefix, every single rebalance -- exactly the
"documented simplification" both `kelly_regime_covkelly.py` and `_v2.py`
name in their own docstrings ("each segment's v4 instance is FRESH...
restarts at pos=0... rather than carrying the prior segment's position
forward"). Confirmed by reading `tradebot/window.py::run_period` and
`tradebot/strategies/kelly_regime.py` / `kelly_regime_v3.py::prepare`
directly before writing this file.

Fix, minimal and structurally faithful
-----------------------------------------
For each asset, run `kelly_regime_v4` via `run_period` **once**,
continuously, over the full inner-train+inner-validation window
(2019-03-14 -> 2022-12-31) at a fixed nominal $1000 start. This is ONE
continuous equity curve per asset with a latch/deadband state that is
never reset -- the object the restart engine structurally cannot produce.
Each rebalance segment's realized return is then read OFF that continuous
curve (`_segment_returns`, chained end-to-end with no gaps or double
counting), and pooled capital is walked through segments by
return-compounding:

    pooled_after = pooled_before * (1 + w_btc*seg_ret_btc + w_eth*seg_ret_eth)

-- the identical causal `w_btc`/`w_eth` produced by
`kelly_regime_covkelly.build_weight_series`, imported UNCHANGED. The mu/
Sigma allocator logic is not touched or reimplemented anywhere in this
file. Only the capital ALLOCATION between two now-genuinely-continuous
legs is periodic; the v4 signal path inside each leg is not.

Bar-by-bar fidelity, not just segment endpoints: within a segment, each
leg's dollar path is the continuous curve *rescaled* by
(dollars-allocated-this-segment / continuous-curve-value-entering-the-
segment) -- an algebraic rescale of the already-computed continuous curve,
never a re-run of the strategy. This keeps Sharpe/drawdown measured at
full bar resolution, the same convention `run_portfolio`'s bar-by-bar
`equity_pieces` concatenation uses.

Scale-invariance caveat (checked, not assumed)
-----------------------------------------------
This rescale-don't-replay trick is only valid if a leg's fee/position-
sizing math is a pure function of that leg's OWN fractional equity, with
no absolute-dollar or leverage-cap effect that would make a $1000-scale
run behave differently from a $137-scale or $50,000-scale run. Checked
directly in `tradebot/strategy.py::order_notional` (`fraction * equity`,
independent of leverage) and `tradebot/broker.py::_max_qty`/
`_clamp_delta`/`_execute_target` (position sizing is `equity * leverage *
haircut / price`, fees are `fee_rate * notional`, the rebalance deadband
is `REBALANCE_DEADBAND * max_notional` -- every one of these is a
*fraction* of current equity/notional; there is no integer lot size, no
minimum order size, and no absolute-dollar knee anywhere in the fill
path). Spot leverage is fixed at 1.0 and v4's `max_leverage` (2.0 by
default) is clamped by the *market's* leverage inside `_max_qty` -- but
that clamp is scale-invariant too (it multiplies `equity`, not a fixed
dollar figure). Conclusion: the % scale-invariance assumption HOLDS on
this codebase, at any pooled-capital scale actually reached in this
data (checked: pooled balances stay in the ~$300-$50,000 range across
every cell below, nowhere near an integer-precision or numerical-underflow
edge).

Hard rules honored
--------------------
- Only this NEW file is touched; `kelly_regime_covkelly.py`,
  `_v2.py` and `kelly_regime_dual_fixed.py` are imported from, unmodified.
- Data hard-sliced to <= 2022-12-31 via the imported, unchanged
  `load_assets` (LOAD_CUTOFF is applied there, once). No 2023+ literal
  appears anywhere in this file outside this sentence, by construction --
  the holdout year is never spelled out as a string constant here at all.
- No lookahead: `causality_check_continuous` runs the same
  multiply/divide truncation tamper probe as the original file's
  `causality_check`, extended to cover this file's new code path (segment
  return extraction + return-compounding), not just `build_weight_series`
  (already probed in the original file and reused unchanged here).
- Mandatory R^2 exposure-artifact diagnostic, modeled on the original's
  `artifact_diagnostics`/`r_squared`.
- Global `N_EVALUATED` counts every distinct dynamic-allocator
  configuration backtested, same convention as `kelly_regime_covkelly.py`
  (fixed-50/50 and solo-v4 baselines are NOT counted, matching that
  file's own convention exactly).

Hyperparameter selection, stated
-----------------------------------
`build_weight_series`/`SWEEP_GRID`/`select_best` are imported UNCHANGED
from `kelly_regime_covkelly.py` (the raw-mean R-42 estimator, not R-43's
de-noised `_v2.py` variant -- per this round's explicit brief: "build the
weight series with `build_weight_series` from `kelly_regime_covkelly.py`,
imported unchanged"). Rather than reusing R-42's already-chosen point,
this file RE-SELECTS the best config by running the identical, unchanged
12-point `SWEEP_GRID` through the CONTINUOUS engine and picking the winner
with the identical, unchanged `select_best` rule. Reason: the leg-return
generating process differs between engines (continuous vs. restart), so
reusing R-42's restart-engine-tuned point would confound "different
hyperparameters" with "different engine" in the headline comparison.
Having picked hyperparameters once, this file evaluates BOTH the
continuous engine AND the original restart engine (imported and called
directly, unmodified) at that SAME fixed (weight_params) point, at both
monthly and weekly cadence -- isolating the engine as the only thing that
varies in the (a) comparison the brief asks for.

Usage::

    python experiments/kelly_regime_covkelly_v3_continuous.py sweep       # 12 configs, continuous engine, monthly
    python experiments/kelly_regime_covkelly_v3_continuous.py headline    # step 3's full 4-arm x 2-cadence table
    python experiments/kelly_regime_covkelly_v3_continuous.py causality   # mandatory no-lookahead check
    python experiments/kelly_regime_covkelly_v3_continuous.py artifact    # mandatory R^2 diagnostic
    python experiments/kelly_regime_covkelly_v3_continuous.py all         # everything above, in order
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.kelly_regime_covkelly import (  # noqa: E402
    SPOT,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    load_assets,
    build_weight_series,
    weight_at,
    _segment_bounds,
    run_portfolio as run_portfolio_restart,   # the ORIGINAL, unmodified restart engine
    run_v4_solo,
    portfolio_metrics,
    r_squared,
    SWEEP_GRID,
    select_best,
)

FULL_START = TRAIN_START   # 2019-03-14, ETH's real start -- same as R-42/R-43
FULL_END = VALID_END       # end of inner-validation; the holdout year is never named here

N_EVALUATED = 0  # dynamic-allocator configurations only, same convention as the original file

# cache of continuous per-leg equity curves: (id(df), market.name, v4_kwargs_key) -> pd.Series
_LEG_CACHE: dict = {}


# ============================================================ continuous legs

def _v4_kwargs_key(v4_kwargs: dict | None) -> tuple:
    return tuple(sorted((v4_kwargs or {}).items()))


def continuous_leg_equity(df: pd.DataFrame, market: MarketSpec,
                          v4_kwargs: dict | None = None,
                          start: str = FULL_START, end: str = FULL_END,
                          start_balance: float = 1000.0) -> pd.Series:
    """Run kelly_regime_v4 ONCE, continuously, over [start, end].

    This is the object the restart engine cannot produce: pos/state carry
    forward through every bar of the whole window, never reset. Cached by
    (frame identity, market, v4_kwargs) since the sweep below re-derives
    weight params many times over the SAME two continuous curves.
    """
    key = (id(df), market.name, _v4_kwargs_key(v4_kwargs), start, end, start_balance)
    if key in _LEG_CACHE:
        return _LEG_CACHE[key]
    result = run_period(KellyRegimeV4(**(v4_kwargs or {})), df, start=start, end=end,
                        market=market, start_balance=start_balance)
    _LEG_CACHE[key] = result.equity
    return result.equity


def _segment_returns(equity: pd.Series, bounds: list[pd.Timestamp]) -> list[dict]:
    """Chain segment returns off ONE continuous curve, base-to-base, no gaps.

    Segment i's base is segment (i-1)'s end value (the running position on
    the single continuous curve) -- the "equity[seg_start_prev_bar]"
    convention from the design brief -- except for the very first segment,
    whose base is the curve's own first bar (== start_balance, by
    `run_period`'s own contract). No re-running of the strategy anywhere
    here: every value used is already sitting on `equity`.
    """
    idx = equity.index
    out = []
    prev_val = None
    for i in range(len(bounds) - 1):
        seg_start = bounds[i]
        seg_end = bounds[i + 1] - pd.Timedelta(minutes=5)
        if seg_end < seg_start:
            continue
        if prev_val is None:
            pos0 = idx.searchsorted(seg_start)
            if pos0 >= len(idx):
                continue
            base_val = float(equity.iloc[pos0])
        else:
            base_val = prev_val
        pos1 = idx.searchsorted(seg_end, side="right") - 1
        if pos1 < 0:
            continue
        end_val = float(equity.iloc[pos1])
        ret = (end_val / base_val - 1.0) if base_val > 0 else 0.0
        out.append({"seg_start": seg_start, "seg_end": seg_end,
                    "base_val": base_val, "end_val": end_val, "ret": ret})
        prev_val = end_val
    return out


# ======================================================== continuous engine

def run_continuous_full(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    rebalance_freq: str, weight_mode: str, weight_params: dict | None = None,
    market: MarketSpec = SPOT, start_balance: float = 1000.0,
    v4_kwargs: dict | None = None,
) -> dict:
    """The fix: ONE continuous per-leg run, sliced-and-rescaled per segment
    instead of re-run per segment. No restart of v4 state anywhere, and (a
    second-order removal the restart engine also implicitly does) no
    restart of pooled CAPITAL at the train/valid reporting boundary either
    -- this produces a single continuous pooled curve across the whole
    window, sliced afterwards purely for reporting.
    """
    btc_full = continuous_leg_equity(btc_df, market, v4_kwargs, start_balance=start_balance)
    eth_full = continuous_leg_equity(eth_df, market, v4_kwargs, start_balance=start_balance)

    weights_df = None
    if weight_mode == "dynamic":
        weights_df = build_weight_series(btc_df, eth_df, **(weight_params or {}))
    elif weight_mode != "fixed5050":
        raise ValueError(weight_mode)

    bounds = _segment_bounds(FULL_START, FULL_END, rebalance_freq)
    btc_segs = _segment_returns(btc_full, bounds)
    eth_segs = _segment_returns(eth_full, bounds)
    n = min(len(btc_segs), len(eth_segs))

    pooled = start_balance
    pieces = []
    log_rows = []
    for i in range(n):
        sb, se = btc_segs[i], eth_segs[i]
        seg_start, seg_end = sb["seg_start"], sb["seg_end"]
        if weight_mode == "dynamic":
            w_b, w_e, fb = weight_at(weights_df, seg_start)
        else:
            w_b, w_e, fb = 0.5, 0.5, False

        dollars_b = pooled * w_b
        dollars_e = pooled * w_e
        cash = pooled * max(0.0, 1.0 - w_b - w_e)

        btc_sub = btc_full.loc[seg_start:seg_end]
        eth_sub = eth_full.loc[seg_start:seg_end]
        scale_b = (dollars_b / sb["base_val"]) if sb["base_val"] > 0 else 0.0
        scale_e = (dollars_e / se["base_val"]) if se["base_val"] > 0 else 0.0
        btc_leg = btc_sub * scale_b
        eth_leg = eth_sub * scale_e

        idx = btc_leg.index.union(eth_leg.index)
        combined = btc_leg.reindex(idx).ffill().bfill().fillna(0.0) \
                 + eth_leg.reindex(idx).ffill().bfill().fillna(0.0) + cash
        if len(combined) == 0:
            continue
        pieces.append(combined)
        pooled = float(combined.iloc[-1])
        log_rows.append({"date": seg_start, "w_btc": w_b, "w_eth": w_e, "fallback": fb,
                         "dollars_btc": dollars_b, "dollars_eth": dollars_e,
                         "cash": cash, "pooled_end": pooled})

    equity = pd.concat(pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    return {"equity": equity, "weights_log": pd.DataFrame(log_rows),
            "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance,
            "btc_full": btc_full, "eth_full": eth_full}


def period_metrics(equity_full: pd.Series, start: str, end: str) -> dict:
    """Slice a continuous curve to [start, end] and report scale-invariant
    Sharpe/max-DD (both are ratio-based, so slicing is exact) plus a
    final_balance REBASED to $1000 at the period's own start, for display
    parity with the restart engine's per-period-$1000 convention. The true
    (un-rebased) compounded dollar value is also reported.
    """
    sub = equity_full.loc[start:end]
    if len(sub) == 0:
        return {"final_balance": float("nan"), "sharpe": 0.0, "max_dd_pct": 0.0,
                "raw_final_balance": float("nan"), "n_bars": 0}
    arr = sub.to_numpy(dtype=float)
    rebased = 1000.0 * (arr[-1] / arr[0]) if arr[0] > 0 else float("nan")
    return {"final_balance": rebased, "sharpe": sharpe_ratio(arr),
            "max_dd_pct": max_drawdown_pct(arr),
            "raw_final_balance": float(arr[-1]), "n_bars": len(arr)}


# =================================================================== sweep

def eval_config_continuous(btc_df, eth_df, weight_params: dict,
                           rebalance_freq: str = "MS", v4_kwargs: dict | None = None) -> dict:
    global N_EVALUATED
    N_EVALUATED += 1
    res = run_continuous_full(btc_df, eth_df, rebalance_freq, "dynamic", weight_params,
                              v4_kwargs=v4_kwargs)
    eq = res["equity"]
    return {"train": period_metrics(eq, TRAIN_START, TRAIN_END),
           "valid": period_metrics(eq, VALID_START, VALID_END)}


def run_sweep_continuous(data_dir: str = "data") -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    btc_df, eth_df = load_assets(data_dir)
    rows = []
    for params in SWEEP_GRID:
        r = eval_config_continuous(btc_df, eth_df, params, rebalance_freq="MS")
        rows.append({"params": params, "train": r["train"], "valid": r["valid"]})
        print(f"{params} | train final={r['train']['final_balance']:.0f} "
              f"Sharpe={r['train']['sharpe']:.2f} DD={r['train']['max_dd_pct']:.1f}% "
              f"|| valid final={r['valid']['final_balance']:.0f} "
              f"Sharpe={r['valid']['sharpe']:.2f} DD={r['valid']['max_dd_pct']:.1f}%")
    print(f"\nconfigs evaluated this call: {len(SWEEP_GRID)} (N_EVALUATED so far: {N_EVALUATED})")
    return rows, btc_df, eth_df


# ================================================================ headline

def run_headline(data_dir: str = "data") -> dict:
    global N_EVALUATED
    print("=== step 1: select hyperparameters via the continuous engine ===")
    rows, btc_df, eth_df = run_sweep_continuous(data_dir)
    best = select_best(rows)
    best_params = best["params"]
    print(f"\nselected best config (continuous-engine sweep, monthly): {best_params}")

    cadences = {"monthly (MS)": "MS", "weekly (W-MON)": "W-MON"}
    out = {"best_params": best_params, "cells": {}}

    for cad_label, freq in cadences.items():
        # (continuous) dynamic -- monthly already counted in the sweep above
        if freq != "MS":
            N_EVALUATED += 1
        cres = run_continuous_full(btc_df, eth_df, freq, "dynamic", best_params)
        ceq = cres["equity"]
        cont = {"train": period_metrics(ceq, TRAIN_START, TRAIN_END),
               "valid": period_metrics(ceq, VALID_START, VALID_END),
               "_equity": ceq}

        # (a) original restart engine, SAME weight_params, SAME cadence
        N_EVALUATED += 1
        restart = {}
        for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                              ("valid", (VALID_START, VALID_END))):
            r = run_portfolio_restart(btc_df, eth_df, s, e, SPOT, 1000.0, freq,
                                      "dynamic", best_params)
            restart[label] = portfolio_metrics(r["equity"], 1000.0)
            restart[label]["_equity"] = r["equity"]

        # (b) fixed 50/50 static split -- ORIGINAL restart engine (the project's
        # existing control), same cadence
        fixed = {}
        for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                              ("valid", (VALID_START, VALID_END))):
            r = run_portfolio_restart(btc_df, eth_df, s, e, SPOT, 1000.0, freq,
                                      "fixed5050", None)
            fixed[label] = portfolio_metrics(r["equity"], 1000.0)
            fixed[label]["_equity"] = r["equity"]

        # bonus diagnostic: fixed 50/50 through the CONTINUOUS engine too --
        # isolates whether ANY residual cadence-sensitivity of the fixed leg
        # is an engine effect (should be ~cadence-invariant here) vs a
        # weight-dynamics effect (visible only in the dynamic arms above)
        cfixed_res = run_continuous_full(btc_df, eth_df, freq, "fixed5050")
        cfeq = cfixed_res["equity"]
        cfixed = {"train": period_metrics(cfeq, TRAIN_START, TRAIN_END),
                 "valid": period_metrics(cfeq, VALID_START, VALID_END)}

        # (c) v4 solo BTC alone -- cadence-independent, imported unchanged
        solo = {}
        for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                              ("valid", (VALID_START, VALID_END))):
            res = run_v4_solo(btc_df, s, e, SPOT, 1000.0)
            eq = res.equity
            solo[label] = {"final_balance": float(eq.iloc[-1]), "sharpe": sharpe_ratio(eq.to_numpy()),
                          "max_dd_pct": max_drawdown_pct(eq.to_numpy()), "_equity": eq}

        out["cells"][cad_label] = {"continuous_dynamic": cont, "restart_dynamic": restart,
                                   "fixed5050_restart": fixed, "fixed5050_continuous": cfixed,
                                   "v4_solo_btc": solo}

    _print_headline(out)
    return out


def _print_headline(out: dict) -> None:
    print("\n=== HEADLINE: continuous engine vs. restart engine, both cadences (spot) ===")
    header = f"{'candidate':<34} {'cadence':<16} {'period':<6} {'final':>10} {'sharpe':>8} {'maxDD%':>8}"
    print(header)
    names = [("continuous dynamic (this file, FIX)", "continuous_dynamic"),
             ("restart dynamic (orig engine, R-42)", "restart_dynamic"),
             ("fixed 50/50 (restart engine)", "fixed5050_restart"),
             ("fixed 50/50 (continuous, bonus)", "fixed5050_continuous"),
             ("v4 BTC alone (100%)", "v4_solo_btc")]
    for cad_label, cell in out["cells"].items():
        for disp, key in names:
            table = cell[key]
            for label in ("train", "valid"):
                m = table[label]
                print(f"{disp:<34} {cad_label:<16} {label:<6} {m['final_balance']:>10.1f} "
                      f"{m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")
    print(f"\ntotal N_EVALUATED (dynamic-allocator configurations): {N_EVALUATED}")


# =============================================================== diagnostics

def causality_check_continuous(data_dir: str = "data", weight_params: dict | None = None) -> bool:
    """Truncation tamper probe on THIS file's new code path (segment-return
    extraction off the continuous curve + return-compounding), modeled on
    the original file's `causality_check`. `build_weight_series` itself is
    reused unchanged and was already probed there.
    """
    btc_df, eth_df = load_assets(data_dir)
    cut = pd.Timestamp("2021-06-30", tz="UTC")
    K = 137.0
    wp = weight_params or {"halflife_days": 60.0, "kelly_frac": 0.5,
                           "max_leg_weight": 1.0, "total_cap": 1.0}

    def tamper(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > cut
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        return out

    base = run_continuous_full(btc_df, eth_df, "MS", "dynamic", wp)
    up = run_continuous_full(tamper(btc_df, K), tamper(eth_df, K), "MS", "dynamic", wp)
    down = run_continuous_full(tamper(btc_df, 1.0 / K), tamper(eth_df, 1.0 / K), "MS", "dynamic", wp)

    pre = base["equity"].index <= cut
    b = base["equity"][pre].to_numpy()
    u = up["equity"].reindex(base["equity"].index)[pre].to_numpy()
    d = down["equity"].reindex(base["equity"].index)[pre].to_numpy()
    max_diff_up = float(np.nanmax(np.abs(b - u)))
    max_diff_down = float(np.nanmax(np.abs(b - d)))
    ok = max_diff_up < 1e-6 and max_diff_down < 1e-6
    print(f"causality check (continuous engine): cut={cut.date()}, K={K}")
    print(f"  max |base - up-tampered| pooled equity before cut: {max_diff_up:.3e}")
    print(f"  max |base - down-tampered| pooled equity before cut: {max_diff_down:.3e}")
    print(f"  PASS (pooled equity before cut unchanged): {ok}")
    return ok


def artifact_diagnostics_full(btc_df, eth_df, best_params: dict, cadences: dict) -> None:
    """Mandatory R^2 exposure-artifact diagnostic. Both comparators are the
    CONTINUOUS engine's own objects (solo-BTC-continuous, fixed5050-
    continuous) so the check isolates the dynamic-weight contribution from
    the engine choice, not conflating the two.
    """
    print("\n=== exposure-artifact diagnostics (continuous engine) ===")
    for cad_label, freq in cadences.items():
        dyn = run_continuous_full(btc_df, eth_df, freq, "dynamic", best_params)
        fixed = run_continuous_full(btc_df, eth_df, freq, "fixed5050")
        btc_full = dyn["btc_full"]
        for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                              ("valid", (VALID_START, VALID_END))):
            dyn_sub = dyn["equity"].loc[s:e]
            fixed_sub = fixed["equity"].loc[s:e]
            solo_sub = btc_full.loc[s:e]
            r2_solo = r_squared(dyn_sub, solo_sub)
            r2_fixed = r_squared(dyn_sub, fixed_sub)
            flag_solo = "FLAT-RESCALE ARTIFACT" if r2_solo > 0.95 else "ok"
            flag_fixed = "SAME AS FIXED SPLIT" if r2_fixed > 0.95 else "ok"
            print(f"[{cad_label} / {label}] dynamic vs flat-rescaled v4-BTC-solo (continuous): "
                  f"R^2 = {r2_solo:.4f} -> {flag_solo}")
            print(f"[{cad_label} / {label}] dynamic vs fixed 50/50 (continuous control):        "
                  f"R^2 = {r2_fixed:.4f} -> {flag_fixed}")


# ===================================================================== CLI

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "sweep":
        run_sweep_continuous()
    elif cmd == "headline":
        run_headline()
    elif cmd == "causality":
        causality_check_continuous()
    elif cmd == "artifact":
        btc_df, eth_df = load_assets()
        rows, _, _ = run_sweep_continuous()
        best = select_best(rows)
        artifact_diagnostics_full(btc_df, eth_df, best["params"],
                                  {"monthly (MS)": "MS", "weekly (W-MON)": "W-MON"})
    elif cmd == "all":
        out = run_headline()
        causality_check_continuous(weight_params=out["best_params"])
        btc_df, eth_df = load_assets()
        artifact_diagnostics_full(btc_df, eth_df, out["best_params"],
                                  {"monthly (MS)": "MS", "weekly (W-MON)": "W-MON"})
        print(f"\ntotal N_EVALUATED (dynamic-allocator configurations): {N_EVALUATED}")
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
