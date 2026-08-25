"""R-131 — the shared evaluation battery for both branches.

One place that knows how to run a candidate against `kelly_regime_v4` on a
named cell, so the conservative and novel branch runners cannot drift apart
in how they measure. Every cell is inner-validation (2021-01-01 → 2022-12-31)
unless explicitly asked for inner-train; nothing here can reach the holdout
(`r131_shared._assert_no_holdout` runs at load time).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return
from tradebot.metrics import compute_metrics
from tradebot.registry import get_strategy
from tradebot.window import run_period

from r131_shared import (
    FUTURES,
    FUTURES_HIGH_FEE,
    INNER_TRAIN_START,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    SPOT,
    SPOT_HIGH_FEE,
    load_btc_train,
    load_eth_train,
)

SEED = 131

_CACHE: dict = {}


@dataclass
class Cell:
    key: str
    df: pd.DataFrame
    market: MarketSpec
    label: str


def build_cells() -> dict[str, Cell]:
    btc, blabel = load_btc_train()
    eth = load_eth_train()
    return {
        "btc_spot": Cell("btc_spot", btc, SPOT, blabel),
        "btc_futures": Cell("btc_futures", btc, FUTURES, blabel),
        "eth_spot": Cell("eth_spot", eth, SPOT, "coinbase eth spot"),
        "btc_spot_040": Cell("btc_spot_040", btc, SPOT_HIGH_FEE, blabel),
        "btc_futures_040": Cell("btc_futures_040", btc, FUTURES_HIGH_FEE, blabel),
        "eth_spot_040": Cell("eth_spot_040", eth, SPOT_HIGH_FEE, "coinbase eth spot"),
    }


def _run(strategy, cell: Cell, slice_name: str):
    start, end = ((INNER_VAL_START, INNER_VAL_END) if slice_name == "inner-val"
                  else (INNER_TRAIN_START, INNER_TRAIN_END))
    return run_period(strategy, cell.df, start=start, end=end, market=cell.market,
                      start_balance=1000.0, data_label=cell.label)


def baseline(cell: Cell, slice_name: str = "inner-val"):
    """`kelly_regime_v4` on the same cell — cached, it never changes."""
    k = (cell.key, slice_name)
    if k not in _CACHE:
        res = _run(get_strategy("kelly_regime_v4"), cell, slice_name)
        _CACHE[k] = (compute_metrics(res), daily_returns(res.equity), len(res.fills))
    return _CACHE[k]


def compare(factory, cell: Cell, slice_name: str = "inner-val", tag: str = "") -> dict:
    """Run one candidate config on one cell and pair it against v4."""
    t0 = time.time()
    strat = factory()
    res = _run(strat, cell, slice_name)
    m = compute_metrics(res)
    r_cand = daily_returns(res.equity)

    m_v4, r_v4, fills_v4 = baseline(cell, slice_name)
    n = min(len(r_cand), len(r_v4))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_v4.to_numpy()[:n],
                              stat=total_log_return, seed=SEED)
    diag = getattr(strat, "diag", {}) or {}
    return {
        "tag": tag or getattr(strat, "name", "?"),
        "cell": cell.key,
        "slice": slice_name,
        "sharpe": round(m.sharpe, 3),
        "sharpe_v4": round(m_v4.sharpe, 3),
        "d_sharpe": round(m.sharpe - m_v4.sharpe, 3),
        "dd": round(m.max_drawdown_pct, 2),
        "dd_v4": round(m_v4.max_drawdown_pct, 2),
        "d_dd": round(m.max_drawdown_pct - m_v4.max_drawdown_pct, 2),
        # `num_trades` counts round-trip EPISODES (grouped fills), not orders.
        # Turnover is `fills`; both are reported so they cannot be confused.
        "trades": m.num_trades,
        "trades_v4": m_v4.num_trades,
        "fills": len(res.fills),
        "fills_v4": fills_v4,
        # exposure, for the standing "match risk before comparing anything"
        # rule: a Sharpe difference between arms carrying different realized
        # volatility is a statement about the exposures.
        "tim": round(m.time_in_market_pct, 1),
        "tim_v4": round(m_v4.time_in_market_pct, 1),
        "vol": round(float(np.std(r_cand.to_numpy(), ddof=1) * np.sqrt(365.25)), 4),
        "vol_v4": round(float(np.std(r_v4.to_numpy(), ddof=1) * np.sqrt(365.25)), 4),
        "final": round(m.final_balance, 1),
        "final_v4": round(m_v4.final_balance, 1),
        "paired": round(paired.diff.point, 4),
        "lo": round(paired.diff.lo, 4),
        "hi": round(paired.diff.hi, 4),
        "sig": bool(paired.significant),
        "n_pending": diag.get("n_pending"),
        "n_intervened": diag.get("n_intervened"),
        "lam_mean": (round(diag["lam_mean"], 4) if "lam_mean" in diag else None),
        "lam_frac_pos": (round(diag["lam_frac_positive"], 4)
                         if "lam_frac_positive" in diag else None),
        "secs": round(time.time() - t0, 1),
    }


COLS = ["tag", "cell", "slice", "sharpe", "sharpe_v4", "d_sharpe", "dd", "dd_v4",
        "d_dd", "fills", "fills_v4", "trades", "trades_v4", "paired", "lo", "hi",
        "sig", "tim", "tim_v4", "vol", "vol_v4",
        "n_pending", "n_intervened", "lam_mean", "lam_frac_pos"]


def show(rows: list[dict], title: str = "") -> pd.DataFrame:
    t = pd.DataFrame(rows)
    keep = [c for c in COLS if c in t.columns and not t[c].isna().all()]
    if title:
        print(f"\n=== {title} ===")
    print(t[keep].to_string(index=False))
    return t
