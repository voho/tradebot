#!/usr/bin/env python
"""CONSERVATIVE branch, R-96: performance-adaptive Hedge (multiplicative
weights) reallocation of pooled capital between two continuously-run,
independent ``kelly_regime_v4`` books -- BTC (Bitstamp spot) and ETH
(Coinbase spot) -- as a sixth attempt at the BTC+ETH combination line that
B-16/B-17/B-19/B-20 all closed NEGATIVE.

Backlog / duplication check
---------------------------
Five prior BTC+ETH `kelly_regime_v4` combination attempts are on record,
ALL REJECTED (docs/LEDGER.md):

  - R-43 (B-16): covariance/mean estimator (Sigma^-1 mu), robustified twice.
  - R-51-conservative (B-19): one-time, never-rebalanced fixed 50/50 split.
  - R-51-novel (B-19): periodically-rebalanced inverse-trailing-volatility
    weighting.
  - R-52-conservative (B-20): literal calendar-rebalanced fixed 50/50.
  - R-52-novel (B-20): drift-band-triggered fixed 50/50.

Every one of these five is a FIXED-FORMULA weight: either a point estimate
of a covariance/mean model, a single frozen ratio, an inverse-volatility
ratio, or a band/calendar trigger around a frozen 50/50 target. None of
them change the split based on which leg's OWN capital has actually been
compounding faster recently -- they all decide the split from either a
model of the assets or a calendar, never from the two books' own realized
scoreboard. This round is the first to try a genuinely different
mechanism class: **Hedge / multiplicative weights** (Freund & Schapire
1997, JCSS 55(1):119-139; Herbster & Warmuth 1998, Machine Learning
32(2):151-178 for the fixed-share variant that lets leadership drift
rather than converge permanently to one leg), applied ACROSS the two
per-asset books using each book's own trailing realized performance as
the Hedge "gain" signal -- the identical combination rule this project
already uses successfully across DIFFERENT strategies on ONE asset in
`src/tradebot/strategies/champions_council.py`, here applied across TWO
ASSETS running the SAME strategy instead.

Constraint attacked: **SIZE** (how much to hold, generalized across
instruments) -- no new data channel is used (INFO is untouched).

Continuous-engine bug this file avoids (read in full before writing this
file, not modified): ``experiments/kelly_regime_covkelly.py``'s
``run_portfolio`` and the original ``kelly_regime_covkelly_v2.py`` both
call ``run_period`` ONCE PER REBALANCE SEGMENT, which resets
``kelly_regime_v4``'s internal latch/deadband state (``pos``/``state`` in
``KellyRegimeV3.prepare``) to ``(0, 0)`` at the start of every segment --
a documented, confirmed bug (B-18, R-50). ``kelly_regime_covkelly_v3_
continuous.py`` fixed it: run each leg's ``kelly_regime_v4`` ONCE,
continuously, over the whole window, and do the cross-asset capital
reallocation as a separate, causal, post-hoc step that only reads each
leg's own already-computed equity curve. This file uses the SAME
pattern -- ``continuous_leg_result`` below is call-for-call identical to
that file's ``continuous_leg_equity`` (verified by inspection), just
additionally caching the full ``BacktestResult`` (not only ``.equity``)
because this round's reporting needs each leg's own trade count too.
``_segment_returns`` and ``_segment_bounds`` are imported UNCHANGED from
that file / ``kelly_regime_covkelly.py`` respectively, not reimplemented.

Mechanism, one paragraph
------------------------
Run `kelly_regime_v4` independently and continuously on BTC and ETH from
ETH's real data start (2019-03-14) through the end of inner-validation
(2022-12-31), so neither leg's latch state is ever reset. Each CALENDAR
DAY (a single fixed cadence, chosen in advance -- daily was picked
specifically because B-20 already closed the cadence-sensitivity question
for the fixed-formula line, and this round does not want to reopen it as
a second free parameter), compute each leg's own trailing, fee-adjusted
(fees are already inside each leg's own equity curve, so no separate fee
adjustment is needed), volatility-normalized realized daily return, and
feed it as a Hedge "gain" to a 2-expert multiplicative-weights update
(`logw_i += eta * gain_i`, softmax, fixed-share mixed toward 50/50 to let
leadership drift per Herbster & Warmuth rather than lock in permanently)
-- computed causally so that the weight applied to day D uses only
information through day D-1's close. Pooled capital then compounds
through each day using that day's chosen weight and each leg's OWN
already-computed continuous equity curve, algebraically rescaled per
segment (never re-run), the identical rescale-don't-replay trick
``kelly_regime_covkelly_v3_continuous.py`` uses and whose scale-invariance
precondition it already checked directly against ``strategy.py``/
``broker.py`` (reused here unchanged, not re-derived).

Falsification test (fixed before any result was read)
------------------------------------------------------
On inner-validation (2021-01-01 -> 2022-12-31), this branch is FALSIFIED
(report NEGATIVE, no further tuning) unless the Hedge construction beats
BOTH BTC-solo-or-ETH-solo (whichever is stronger) AND a same-cadence fixed
50/50 reference, on BOTH Sharpe AND a max-drawdown-adjusted growth metric
(defined here, before any sweep ran: `profit_pct / max_dd_pct`, the
Calmar-style ratio; when max_dd_pct rounds to 0 the metric falls back to
plain `profit_pct`) -- and this must hold as a PLATEAU across the eta
sweep below, not a single lucky value, per this round's brief and per
ROUTINE.md's own promotion bar ("the parameter neighbourhood is a
plateau, not a peak").

Eta sweep, fixed before any result was read: {0.01, 0.03, 0.06, 0.10,
0.20, 0.40} -- 6 values spanning slow to fast adaptation (0.06 is
`champions_council`'s own shipped default, included as a natural
mid-point). `fixed_share = 1e-3` and the vol-normalizer
(EWM halflife 20 trading days, min_periods 20, gain clipped to +/-3) are
both fixed once, a priori, and held constant across the whole eta sweep
-- they are not swept, so they cannot be what turns a rejection into a
promotion.

Absolutely forbidden and honored: no bar timestamped 2023-01-01 or later
is ever read. ``load_assets`` (imported unchanged from
``kelly_regime_covkelly.py``) hard-slices to <= 2022-12-31 immediately
after loading; ``FULL_END`` is hardcoded to ``VALID_END`` ("2022-12-31")
with no CLI path that can override it. Grep this file for "2023": it
appears only in this docstring's own sentences about what is forbidden.

Hard rules honored
------------------
- Only this NEW file is written. ``kelly_regime_covkelly*.py``,
  ``multiasset.py``, ``kelly_regime_v4.py``/``v3.py`` and
  ``champions_council.py`` are all imported from or read for reference,
  never modified.
- Mandatory truncation-tamper causality probe on this file's OWN new code
  (the Hedge weight construction + the daily pooling loop), modeled on
  the pattern in the imported files.
- Mandatory R^2 exposure-artifact diagnostic vs. BTC-solo, ETH-solo and
  the fixed-50/50 (same cadence) reference.
- `N_EVALUATED` (this project's internal convention: dynamic-allocator
  configs only) and a separate, more inclusive `TOTAL_CONFIGS_EVALUATED`
  counter (every reference/baseline run too, per this round's explicit
  brief for the ledger's trials count) are both tracked and printed.

Usage::

    python experiments/r96_conservative_hedge_allocator.py eta_sweep   # the falsification test
    python experiments/r96_conservative_hedge_allocator.py headline    # full train+valid table, a-d
    python experiments/r96_conservative_hedge_allocator.py causality   # mandatory no-lookahead check
    python experiments/r96_conservative_hedge_allocator.py artifact    # mandatory R^2 diagnostic
    python experiments/r96_conservative_hedge_allocator.py all         # everything above, in order
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
    _segment_bounds,
    weight_at,
    r_squared,
)
from experiments.kelly_regime_covkelly_v3_continuous import (  # noqa: E402
    FULL_START, FULL_END,
    _segment_returns,
    run_continuous_full,
    period_metrics,
)

# ----------------------------------------------------------------- fixed a priori
GAIN_HALFLIFE_DAYS = 20.0
GAIN_MIN_PERIODS_DAYS = 20
GAIN_CLIP = 3.0
FIXED_SHARE = 1e-3   # Herbster & Warmuth 1998 fixed-share mixing constant
ETA_GRID = [0.01, 0.03, 0.06, 0.10, 0.20, 0.40]  # pre-registered sweep, 6 values

N_EVALUATED = 0             # dynamic (Hedge) allocator configs only, project convention
TOTAL_CONFIGS_EVALUATED = 0  # every reference/baseline run too, per this round's brief

_LEG_RESULT_CACHE: dict = {}


# ======================================================= continuous legs (full result)

def continuous_leg_result(df: pd.DataFrame, market: MarketSpec,
                          v4_kwargs: dict | None = None,
                          start: str = FULL_START, end: str = FULL_END,
                          start_balance: float = 1000.0):
    """Run kelly_regime_v4 ONCE, continuously, over [start, end].

    Call-for-call identical to
    ``kelly_regime_covkelly_v3_continuous.py::continuous_leg_equity`` --
    reproduced (not imported) only because this round's reporting needs
    each leg's own trade count/fees too, which that function discards by
    returning ``.equity`` alone. The engine call itself is byte-identical:
    ``run_period(KellyRegimeV4(**kwargs), df, start=start, end=end,
    market=market, start_balance=start_balance)``.
    """
    key = (id(df), market.name, tuple(sorted((v4_kwargs or {}).items())), start, end, start_balance)
    if key in _LEG_RESULT_CACHE:
        return _LEG_RESULT_CACHE[key]
    result = run_period(KellyRegimeV4(**(v4_kwargs or {})), df, start=start, end=end,
                        market=market, start_balance=start_balance)
    _LEG_RESULT_CACHE[key] = result
    return result


def period_trade_count(result, start: str, end: str) -> int:
    """Trades whose entry lands inside [start, end] -- descriptive only."""
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return sum(1 for t in result.trades if s <= t.entry_ts < e)


# ============================================================ Hedge weight series

def _daily_log_returns_from_equity(equity: pd.Series) -> pd.Series:
    """Causal: day D's value uses only the equity curve's own bars up to
    and including day D's close (the curve itself is already causal)."""
    daily = equity.resample("1D").last().ffill()
    return np.log(daily).diff()


def build_hedge_weight_series(
    btc_full: pd.Series, eth_full: pd.Series, eta: float,
    fixed_share: float = FIXED_SHARE,
    halflife_days: float = GAIN_HALFLIFE_DAYS,
    min_periods_days: int = GAIN_MIN_PERIODS_DAYS,
    clip: float = GAIN_CLIP,
) -> pd.DataFrame:
    """Causal 2-expert Hedge/multiplicative-weights daily weight series.

    Sequential predict-observe-update, the same shape as
    ``champions_council.py``'s bar-by-bar loop, run once per CALENDAR DAY
    here instead of once per bar, and over two per-asset BOOKS (each
    already a full ``kelly_regime_v4`` equity curve, fees included) rather
    than two per-bar signal columns on one asset.

    Causality, made explicit: the weight recorded for day D
    (``w_btc``/``w_eth`` at row D) is read off ``logw`` as it stood
    BEFORE day D's own return is folded in -- ``logw`` at that point
    reflects only gains observed on days < D. Day D's own (vol-normalized)
    gain is folded into ``logw`` only AFTER day D's weight has already
    been recorded, for use starting day D+1. The volatility normalizer
    itself is `.ewm(...).shift(1)`, so it too reflects only days < D.
    """
    r_btc = _daily_log_returns_from_equity(btc_full).rename("btc")
    r_eth = _daily_log_returns_from_equity(eth_full).rename("eth")
    rets = pd.concat([r_btc, r_eth], axis=1).dropna(how="any")

    vol_btc = rets["btc"].ewm(halflife=halflife_days, min_periods=min_periods_days).std().shift(1)
    vol_eth = rets["eth"].ewm(halflife=halflife_days, min_periods=min_periods_days).std().shift(1)

    dates = rets.index
    vb_arr = vol_btc.to_numpy()
    ve_arr = vol_eth.to_numpy()
    rb_arr = rets["btc"].to_numpy()
    re_arr = rets["eth"].to_numpy()

    logw = np.zeros(2)
    rows = []
    for i, d in enumerate(dates):
        vb, ve = vb_arr[i], ve_arr[i]
        ready = np.isfinite(vb) and np.isfinite(ve) and vb > 0 and ve > 0
        if ready:
            p = np.exp(logw - logw.max())
            p = p / p.sum()
            p = (1.0 - fixed_share) * p + fixed_share / 2.0
        else:
            p = np.array([0.5, 0.5])
        rows.append({"date": d, "w_btc": float(p[0]), "w_eth": float(p[1]), "fallback": not ready})
        if ready:
            g_btc = float(np.clip(rb_arr[i] / vb, -clip, clip))
            g_eth = float(np.clip(re_arr[i] / ve, -clip, clip))
            logw = logw + eta * np.array([g_btc, g_eth])
            logw = logw - logw.max()

    return pd.DataFrame(rows).set_index("date")


# ================================================================= Hedge portfolio

def run_hedge_full(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame, eta: float,
    market: MarketSpec = SPOT, start_balance: float = 1000.0,
    v4_kwargs: dict | None = None, fixed_share: float = FIXED_SHARE,
) -> dict:
    """The Hedge-weighted portfolio: two continuous legs, reallocated daily
    by ``build_hedge_weight_series``, capital compounded segment-by-segment
    off each leg's OWN continuous curve (never re-run) -- structurally the
    same pooling loop as
    ``kelly_regime_covkelly_v3_continuous.py::run_continuous_full``'s
    dynamic branch, adapted here to a daily cadence and this file's own
    Hedge weight source instead of the Sigma^-1 mu weight source.
    """
    btc_res = continuous_leg_result(btc_df, market, v4_kwargs, start_balance=start_balance)
    eth_res = continuous_leg_result(eth_df, market, v4_kwargs, start_balance=start_balance)
    btc_full, eth_full = btc_res.equity, eth_res.equity

    weights_df = build_hedge_weight_series(btc_full, eth_full, eta=eta, fixed_share=fixed_share)

    bounds = _segment_bounds(FULL_START, FULL_END, "D")
    btc_segs = _segment_returns(btc_full, bounds)
    eth_segs = _segment_returns(eth_full, bounds)
    n = min(len(btc_segs), len(eth_segs))

    pooled = start_balance
    pieces = []
    log_rows = []
    for i in range(n):
        sb, se = btc_segs[i], eth_segs[i]
        seg_start, seg_end = sb["seg_start"], sb["seg_end"]
        w_b, w_e, fb = weight_at(weights_df, seg_start)

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
            "btc_full": btc_full, "eth_full": eth_full,
            "btc_res": btc_res, "eth_res": eth_res}


def target_weight_turnover(weights_log: pd.DataFrame, start: str, end: str) -> dict:
    """Mean/cumulative |change in TARGET weight| inside [start, end] --
    distinguishes an adaptively-reallocating candidate (nonzero) from a
    fixed-50/50 reference (exactly zero by construction, since its target
    never moves) without needing to re-derive each leg's own v4 trades
    (those are identical across every portfolio variant that shares the
    same two continuous curves -- only the ALLOCATION overlay differs)."""
    if weights_log is None or len(weights_log) == 0:
        return {"mean_abs_daily_dw": 0.0, "cum_abs_dw": 0.0, "n_days": 0}
    wl = weights_log.set_index("date").sort_index()
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    sub = wl.loc[(wl.index >= s) & (wl.index < e)]
    if len(sub) < 2:
        return {"mean_abs_daily_dw": 0.0, "cum_abs_dw": 0.0, "n_days": len(sub)}
    dw = sub["w_btc"].diff().abs().dropna()
    return {"mean_abs_daily_dw": float(dw.mean()), "cum_abs_dw": float(dw.sum()), "n_days": len(sub)}


# ================================================================== metrics

def dd_adjusted_growth(profit_pct: float, max_dd_pct: float) -> float:
    """Calmar-style drawdown-adjusted growth, defined before any sweep ran.
    Falls back to plain profit_pct when max_dd_pct rounds to ~0 (nothing to
    adjust by)."""
    if max_dd_pct < 1e-6:
        return profit_pct
    return profit_pct / max_dd_pct


def _metrics_from_equity(eq: pd.Series, start_balance: float = 1000.0) -> dict:
    arr = eq.to_numpy(dtype=float)
    final = float(arr[-1]) if len(arr) else start_balance
    profit_pct = 100.0 * (final / start_balance - 1.0)
    mdd = max_drawdown_pct(arr)
    return {"final_balance": final, "profit_pct": profit_pct, "sharpe": sharpe_ratio(arr),
            "max_dd_pct": mdd, "dd_adj_growth": dd_adjusted_growth(profit_pct, mdd)}


# =============================================================== reference configs

def config_solo(leg_res, label: str, start: str, end: str) -> dict:
    m = period_metrics(leg_res.equity, start, end)
    metrics = {"final_balance": m["final_balance"], "sharpe": m["sharpe"], "max_dd_pct": m["max_dd_pct"]}
    profit_pct = 100.0 * (m["final_balance"] / 1000.0 - 1.0)
    metrics["profit_pct"] = profit_pct
    metrics["dd_adj_growth"] = dd_adjusted_growth(profit_pct, m["max_dd_pct"])
    metrics["num_trades"] = period_trade_count(leg_res, start, end)
    metrics["label"] = label
    return metrics


_FIXED5050_CACHE: dict = {}


def _metrics_dict_from_period(res_equity: pd.Series, start: str, end: str) -> dict:
    m = period_metrics(res_equity, start, end)
    profit_pct = 100.0 * (m["final_balance"] / 1000.0 - 1.0)
    return {"final_balance": m["final_balance"], "sharpe": m["sharpe"], "max_dd_pct": m["max_dd_pct"],
            "profit_pct": profit_pct, "dd_adj_growth": dd_adjusted_growth(profit_pct, m["max_dd_pct"])}


def config_fixed5050(btc_df, eth_df, freq: str, start: str, end: str, count: bool = True) -> dict:
    """One frozen fixed-50/50 rule, run ONCE over the whole window and
    sliced for whichever period is asked for here -- counted once per
    (freq) the first time it is evaluated, not once per period read off
    it, matching this project's established "distinct configuration"
    counting convention (``kelly_regime_covkelly_v3_continuous.py``'s own
    ``eval_config_continuous`` reports train+valid from one run and
    increments its counter once)."""
    global TOTAL_CONFIGS_EVALUATED
    if freq not in _FIXED5050_CACHE:
        if count:
            TOTAL_CONFIGS_EVALUATED += 1
        _FIXED5050_CACHE[freq] = run_continuous_full(btc_df, eth_df, freq, "fixed5050")
    res = _FIXED5050_CACHE[freq]
    out = _metrics_dict_from_period(res["equity"], start, end)
    out["label"] = f"fixed 50/50 ({freq})"
    return out


_HEDGE_CACHE: dict = {}


def config_hedge(btc_df, eth_df, eta: float, start: str, end: str, count: bool = True) -> dict:
    """One frozen Hedge rule (fixed eta), run ONCE over the whole window
    and sliced for whichever period is asked for here -- counted once per
    eta the first time it is evaluated, not once per period read off it
    (see ``config_fixed5050``'s docstring for the convention this
    matches)."""
    global N_EVALUATED, TOTAL_CONFIGS_EVALUATED
    if eta not in _HEDGE_CACHE:
        if count:
            N_EVALUATED += 1
            TOTAL_CONFIGS_EVALUATED += 1
        _HEDGE_CACHE[eta] = run_hedge_full(btc_df, eth_df, eta)
    res = _HEDGE_CACHE[eta]
    out = _metrics_dict_from_period(res["equity"], start, end)
    turnover = target_weight_turnover(res["weights_log"], start, end)
    out["turnover_mean_abs_daily_dw"] = turnover["mean_abs_daily_dw"]
    out["turnover_cum_abs_dw"] = turnover["cum_abs_dw"]
    out["label"] = f"hedge (eta={eta})"
    out["_equity"] = res["equity"]
    out["_weights_log"] = res["weights_log"]
    return out


# ================================================================ falsification test

def eta_sweep(data_dir: str = "data") -> dict:
    """The pre-registered falsification test, run on inner-validation, and
    also reported (not gated) on inner-train."""
    global TOTAL_CONFIGS_EVALUATED
    btc_df, eth_df = load_assets(data_dir)

    btc_res = continuous_leg_result(btc_df, SPOT)
    eth_res = continuous_leg_result(eth_df, SPOT)
    TOTAL_CONFIGS_EVALUATED += 2  # BTC-solo, ETH-solo baselines

    print("=== R-96 conservative Hedge allocator: pre-registered falsification test ===")
    print(f"inner-train: {TRAIN_START} -> {TRAIN_END}   inner-validation: {VALID_START} -> {VALID_END}")

    results = {"train": {}, "valid": {}}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
        a = config_solo(btc_res, "BTC-solo v4", s, e)
        b = config_solo(eth_res, "ETH-solo v4", s, e)
        c = config_fixed5050(btc_df, eth_df, "D", s, e)
        results[label]["a_btc_solo"] = a
        results[label]["b_eth_solo"] = b
        results[label]["c_fixed5050_daily"] = c
        d_by_eta = {}
        for eta in ETA_GRID:
            d_by_eta[eta] = config_hedge(btc_df, eth_df, eta, s, e)
        results[label]["d_hedge_by_eta"] = d_by_eta

    header = f"{'config':<26} {'period':<6} {'final':>10} {'profit%':>9} {'sharpe':>8} {'maxDD%':>8} {'ddAdjGrowth':>12}"
    print(header)
    for label in ("train", "valid"):
        for key, disp in (("a_btc_solo", "(a) BTC-solo v4"), ("b_eth_solo", "(b) ETH-solo v4"),
                          ("c_fixed5050_daily", "(c) fixed 50/50 (D)")):
            m = results[label][key]
            print(f"{disp:<26} {label:<6} {m['final_balance']:>10.1f} {m['profit_pct']:>+8.1f}% "
                  f"{m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f} {m['dd_adj_growth']:>12.3f}")
        for eta, m in results[label]["d_hedge_by_eta"].items():
            print(f"{'(d) hedge eta=' + str(eta):<26} {label:<6} {m['final_balance']:>10.1f} "
                  f"{m['profit_pct']:>+8.1f}% {m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f} "
                  f"{m['dd_adj_growth']:>12.3f}  turnover(mean|dw|/day)={m['turnover_mean_abs_daily_dw']:.4f}")

    # --- the falsification gate, on inner-validation only, as pre-registered ---
    v = results["valid"]
    best_solo_sharpe = max(v["a_btc_solo"]["sharpe"], v["b_eth_solo"]["sharpe"])
    best_solo_ddadj = max(v["a_btc_solo"]["dd_adj_growth"], v["b_eth_solo"]["dd_adj_growth"])
    fixed_sharpe = v["c_fixed5050_daily"]["sharpe"]
    fixed_ddadj = v["c_fixed5050_daily"]["dd_adj_growth"]

    print("\n=== falsification gate (inner-validation) ===")
    print(f"best solo (BTC or ETH)   Sharpe={best_solo_sharpe:.3f}  ddAdjGrowth={best_solo_ddadj:.3f}")
    print(f"fixed 50/50 (daily)      Sharpe={fixed_sharpe:.3f}  ddAdjGrowth={fixed_ddadj:.3f}")

    gate_pass = {}
    for eta, m in v["d_hedge_by_eta"].items():
        passes = (m["sharpe"] > best_solo_sharpe and m["sharpe"] > fixed_sharpe and
                 m["dd_adj_growth"] > best_solo_ddadj and m["dd_adj_growth"] > fixed_ddadj)
        gate_pass[eta] = passes
        print(f"eta={eta:<5} Sharpe={m['sharpe']:>7.3f}  ddAdjGrowth={m['dd_adj_growth']:>8.3f}  "
              f"-> {'PASS' if passes else 'fail'}")

    n_pass = sum(gate_pass.values())
    # "plateau, not a peak": require at least 2 ADJACENT eta values (in the
    # pre-registered grid order) to both pass, not merely any single one.
    etas_sorted = ETA_GRID
    adjacent_pass = any(gate_pass[etas_sorted[i]] and gate_pass[etas_sorted[i + 1]]
                        for i in range(len(etas_sorted) - 1))
    print(f"\n{n_pass}/{len(ETA_GRID)} eta values PASS the falsification gate on inner-validation.")
    print(f"Plateau (>=2 adjacent eta values both PASS): {adjacent_pass}")

    verdict = "NOT FALSIFIED (gate cleared as a plateau)" if (n_pass >= 2 and adjacent_pass) else \
             ("NOT FALSIFIED but only as an isolated peak (not a plateau) -- treat as weak" if n_pass >= 1 else
              "FALSIFIED (gate fails for every eta tested)")
    print(f"\nVERDICT: {verdict}")

    print(f"\nN_EVALUATED (dynamic Hedge configs only): {N_EVALUATED}")
    print(f"TOTAL_CONFIGS_EVALUATED (incl. every reference run): {TOTAL_CONFIGS_EVALUATED}")
    return {"results": results, "gate_pass": gate_pass, "verdict": verdict,
           "n_pass": n_pass, "adjacent_pass": adjacent_pass}


# ===================================================================== headline

def run_headline(data_dir: str = "data") -> dict:
    """Full train+valid table for (a)-(d), plus a monthly-cadence fixed
    50/50 bonus reference for context against R-52's cited number."""
    global TOTAL_CONFIGS_EVALUATED
    btc_df, eth_df = load_assets(data_dir)
    btc_res = continuous_leg_result(btc_df, SPOT)
    eth_res = continuous_leg_result(eth_df, SPOT)
    TOTAL_CONFIGS_EVALUATED += 2

    print("=== R-96 headline: BTC-solo / ETH-solo / fixed-50-50 / Hedge, train + valid ===")
    header = (f"{'config':<28} {'period':<6} {'final':>10} {'profit%':>9} {'sharpe':>8} "
             f"{'maxDD%':>8} {'ddAdjGrowth':>12} {'trades':>8}")
    print(header)
    out = {"train": {}, "valid": {}}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
        a = config_solo(btc_res, "BTC-solo v4", s, e)
        b = config_solo(eth_res, "ETH-solo v4", s, e)
        c_daily = config_fixed5050(btc_df, eth_df, "D", s, e)
        c_monthly = config_fixed5050(btc_df, eth_df, "MS", s, e)
        out[label] = {"a_btc_solo": a, "b_eth_solo": b, "c_fixed5050_daily": c_daily,
                      "c_fixed5050_monthly_bonus": c_monthly, "d_hedge_by_eta": {}}
        for name, m in (("(a) BTC-solo v4", a), ("(b) ETH-solo v4", b),
                       ("(c) fixed 50/50 (D)", c_daily), ("(c') fixed 50/50 (MS, bonus)", c_monthly)):
            trades = m.get("num_trades", "-")
            print(f"{name:<28} {label:<6} {m['final_balance']:>10.1f} {m['profit_pct']:>+8.1f}% "
                  f"{m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f} {m['dd_adj_growth']:>12.3f} {str(trades):>8}")
        for eta in ETA_GRID:
            d = config_hedge(btc_df, eth_df, eta, s, e)
            out[label]["d_hedge_by_eta"][eta] = d
            print(f"{'(d) hedge eta=' + str(eta):<28} {label:<6} {d['final_balance']:>10.1f} "
                  f"{d['profit_pct']:>+8.1f}% {d['sharpe']:>8.2f} {d['max_dd_pct']:>8.1f} "
                  f"{d['dd_adj_growth']:>12.3f} {'n/a (alloc only)':>8}")

    print(f"\nN_EVALUATED (dynamic Hedge configs only): {N_EVALUATED}")
    print(f"TOTAL_CONFIGS_EVALUATED (incl. every reference run): {TOTAL_CONFIGS_EVALUATED}")
    return out


# =============================================================== diagnostics

def causality_check_hedge(data_dir: str = "data", eta: float = 0.06) -> bool:
    """Truncation-tamper probe on THIS file's new code (Hedge weight
    construction + daily pooling loop), modeled on the pattern in
    ``kelly_regime_covkelly_v3_continuous.py``/``kelly_regime_covkelly.py``.
    """
    btc_df, eth_df = load_assets(data_dir)
    cut = pd.Timestamp("2021-06-30", tz="UTC")
    K = 137.0

    def tamper(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > cut
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        return out

    base = run_hedge_full(btc_df, eth_df, eta)
    up = run_hedge_full(tamper(btc_df, K), tamper(eth_df, K), eta)
    down = run_hedge_full(tamper(btc_df, 1.0 / K), tamper(eth_df, 1.0 / K), eta)

    pre = base["equity"].index <= cut
    b = base["equity"][pre].to_numpy()
    u = up["equity"].reindex(base["equity"].index)[pre].to_numpy()
    d = down["equity"].reindex(base["equity"].index)[pre].to_numpy()
    max_diff_up = float(np.nanmax(np.abs(b - u)))
    max_diff_down = float(np.nanmax(np.abs(b - d)))
    ok = max_diff_up < 1e-6 and max_diff_down < 1e-6
    print(f"causality check (Hedge allocator, eta={eta}): cut={cut.date()}, K={K}")
    print(f"  max |base - up-tampered| pooled equity before cut: {max_diff_up:.3e}")
    print(f"  max |base - down-tampered| pooled equity before cut: {max_diff_down:.3e}")
    print(f"  PASS (pooled equity before cut unchanged): {ok}")
    return ok


def artifact_diagnostics_hedge(data_dir: str = "data", eta: float = 0.06) -> None:
    """Mandatory R^2 exposure-artifact diagnostic: Hedge portfolio vs
    BTC-solo, ETH-solo and fixed-50/50 (same daily cadence)."""
    btc_df, eth_df = load_assets(data_dir)
    print(f"\n=== exposure-artifact diagnostics (Hedge allocator, eta={eta}) ===")
    hedge = run_hedge_full(btc_df, eth_df, eta)
    fixed = run_continuous_full(btc_df, eth_df, "D", "fixed5050")
    btc_full = hedge["btc_full"]
    eth_full = hedge["eth_full"]
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
        dyn_sub = hedge["equity"].loc[s:e]
        r2_btc = r_squared(dyn_sub, btc_full.loc[s:e])
        r2_eth = r_squared(dyn_sub, eth_full.loc[s:e])
        r2_fixed = r_squared(dyn_sub, fixed["equity"].loc[s:e])
        flag = lambda r2, msg: "ARTIFACT" if (np.isfinite(r2) and r2 > 0.95) else "ok"  # noqa: E731
        print(f"[{label}] hedge vs BTC-solo (continuous):   R^2 = {r2_btc:.4f} -> {flag(r2_btc, 'btc')}")
        print(f"[{label}] hedge vs ETH-solo (continuous):   R^2 = {r2_eth:.4f} -> {flag(r2_eth, 'eth')}")
        print(f"[{label}] hedge vs fixed 50/50 (daily):     R^2 = {r2_fixed:.4f} -> {flag(r2_fixed, 'fixed')}")


def print_max_timestamp_read(data_dir: str = "data") -> None:
    btc_df, eth_df = load_assets(data_dir)
    print(f"\nmax timestamp ever read: BTC={btc_df.index.max()}  ETH={eth_df.index.max()}  "
          f"(both must be <= 2022-12-31 23:55:00+00:00)")
    assert btc_df.index.max() <= pd.Timestamp("2022-12-31 23:55:00", tz="UTC")
    assert eth_df.index.max() <= pd.Timestamp("2022-12-31 23:55:00", tz="UTC")


# ===================================================================== CLI

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "eta_sweep":
        eta_sweep()
        print_max_timestamp_read()
    elif cmd == "headline":
        run_headline()
        print_max_timestamp_read()
    elif cmd == "causality":
        causality_check_hedge()
    elif cmd == "artifact":
        artifact_diagnostics_hedge()
    elif cmd == "all":
        eta_sweep()
        causality_check_hedge()
        artifact_diagnostics_hedge()
        print_max_timestamp_read()
        print(f"\nFINAL N_EVALUATED (dynamic Hedge configs only): {N_EVALUATED}")
        print(f"FINAL TOTAL_CONFIGS_EVALUATED (incl. every reference run): {TOTAL_CONFIGS_EVALUATED}")
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
