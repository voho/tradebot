"""R-133 Step A — the census the pre-registration's A2 gate needs, run FIRST.

Before either branch is written, measure the object the whole round throttles:
`kelly_regime_v4`'s own realized rebalance-event rate, and the causal trailing
EWM of it that both branches condition on.

Two questions, both answerable without a single performance number:

A1. How often does v4's `prepare()` loop actually change `pos` (a "rebalance
    event"), on inner-train and inner-validation, BTC and ETH?
A2. Does the frozen corridor edge `TURNOVER_UPPER = 3 x 1/3.3 = 0.909
    trades/day` ever get reached by that trailing EWM? If it never does, the
    pre-registration's own second named failure mode (INERTNESS) has fired and
    NEITHER branch is testable at the frozen calibration -- which per
    ROUTINE.md ("not tested is not a negative result") would make the round
    untried rather than negative.

Reads inner-train + inner-validation only. No holdout, no market spec, no
fees -- this touches `prepare()` alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from r131_shared import (  # noqa: E402
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    TURNOVER_EWM_SPAN_DAYS,
    TURNOVER_UPPER,
    load_btc_train,
    load_eth_train,
    trailing_turnover_ewm,
)
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402


def rebalance_events(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Re-derive v4's own `pos`-change events from its prepared `target` column.

    v4's `prepare()` writes a latched `target` that only moves when
    `|desired - pos| > deadband`; a bar where `target` differs from the
    previous bar's is exactly one rebalance event.
    """
    strat = get_strategy("kelly_regime_v4")
    prepared = strat.prepare(df.copy())
    t = prepared["target"].to_numpy(dtype=float)
    ev = np.zeros(len(t))
    ev[1:] = (np.abs(np.diff(t)) > 1e-12).astype(float)
    return ev, prepared.index


def census(name: str, df: pd.DataFrame) -> dict:
    ev, idx = rebalance_events(df)
    ewm = trailing_turnover_ewm(ev)

    rows = []
    for label, lo, hi in (
        ("inner-train", INNER_TRAIN_START, INNER_TRAIN_END),
        ("inner-val", INNER_VAL_START, INNER_VAL_END),
    ):
        mask = (idx >= pd.Timestamp(lo, tz=idx.tz)) & (idx <= pd.Timestamp(hi, tz=idx.tz))
        # warmup bars carry no signal; drop the leading all-zero stretch
        sub_ev, sub_ewm = ev[mask], ewm[mask]
        n_bars = int(mask.sum())
        n_days = n_bars / BARS_PER_DAY
        rows.append({
            "market": name,
            "slice": label,
            "bars": n_bars,
            "days": round(n_days, 1),
            "events": int(sub_ev.sum()),
            "events_per_day": round(float(sub_ev.sum()) / n_days, 4) if n_days else float("nan"),
            "ewm_max": round(float(np.nanmax(sub_ewm)), 4),
            "ewm_p99": round(float(np.nanpercentile(sub_ewm, 99)), 4),
            "ewm_p95": round(float(np.nanpercentile(sub_ewm, 95)), 4),
            "ewm_p90": round(float(np.nanpercentile(sub_ewm, 90)), 4),
            "ewm_median": round(float(np.nanmedian(sub_ewm)), 4),
            "frac_bars_over_frozen_edge": round(
                float(np.mean(sub_ewm >= TURNOVER_UPPER)), 6),
        })
    return rows


if __name__ == "__main__":
    btc, label = load_btc_train()
    eth = load_eth_train()

    out = []
    out += census(f"BTC ({label})", btc)
    out += census("ETH (coinbase spot)", eth)
    table = pd.DataFrame(out)

    print(f"\nR-131 Step A — v4 rebalance-event census "
          f"(trailing EWM span = {TURNOVER_EWM_SPAN_DAYS}d, "
          f"frozen corridor edge = {TURNOVER_UPPER:.4f} trades/day)\n")
    print(table.to_string(index=False))

    binds = table["frac_bars_over_frozen_edge"].max() > 0
    print(f"\nA2 non-inertness at the FROZEN edge: "
          f"{'PASS — corridor is reached' if binds else 'FAIL — corridor never reached'}")

    # If the frozen edge is inert, what edge would NOT be? Report the
    # inner-TRAIN quantiles only (inner-validation is the selection slice and
    # must not calibrate the mechanism).
    tr = table[table["slice"] == "inner-train"]
    print("\nInner-train-only quantiles of v4's own trailing turnover "
          "(candidate re-derived corridor edges, no performance information used):")
    print(tr[["market", "ewm_median", "ewm_p90", "ewm_p95", "ewm_p99", "ewm_max"]]
          .to_string(index=False))
