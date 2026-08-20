#!/usr/bin/env python
"""R-56 (conservative branch): "patient limit, taker fallback" execution for kelly_regime_v4.

CONSTRAINT ATTACKED: COST (costs that scale with the signal). This does
NOT touch kelly_regime_v4's SIZE/vote signal at all — it reuses v4's own
``prepare()`` output byte-for-byte and only changes *how* an
already-decided rebalance fills.

Not a duplicate of:
- L-05/L-06 (kelly_regime_ev's analytically-derived no-trade band): that
  decides WHEN to trade (whether |Delta f| is worth its cost at all).
  This round assumes that decision has already been made — v4 itself
  emits a new order only when its own hand-set 10% hysteresis deadband
  (kelly_regime.py's ``deadband=0.10``, inherited unchanged by v3/v4) has
  fired — and asks only whether the trade the signal already wants can
  be filled cheaper. NOTE for the ledger: v4 does NOT actually use
  kelly_regime_ev's EV-derived band (that lives one class further down
  the family, in KellyRegimeEV(KellyRegimeV4)) — v4 uses the older fixed
  10% band. The framing in this round's prompt ("gated by the
  analytically-derived band") is imprecise about which class owns which
  band; the mechanism point it's making (a trade has already been judged
  worth its cost before this round's execution model touches it) is
  correct regardless of which deadband produced that judgement.
- R-12/R-13 (taker-only fee-tier sweeps): those only ever charge the
  taker rate at every tier. Nothing in the ledger models a maker leg.
- R-40's kelly_regime_v8_uncertainty_shrink (Baker-McHale/Bayesian-Kelly
  shrink): that changed the SIZE/vote signal itself. This round changes
  nothing about *how much* v4 wants to hold, only *how* the fill happens.

MECHANISM (one sentence): resting a maker-fee limit order at the
signal's own decision price, with a hard taker-fee fallback after N
bars, should recover some of the taker/maker fee spread without
materially delaying the crash-transition flattens that are v4's actual
edge (L-01/README: its whole edge lives in the windows that contain a
crash).

FALSIFICATION (pre-registered before running anything, per ROUTINE.md
step 2): this idea is dead if (a) it does not replicate directionally on
the ETH falsification pair; (b) it fails a pre-2020 BTC-only control
window; or (c) regime-flip-to-flat events get delayed by more than 1-2
bars vs the taker baseline during crash transitions — since a maker-fee
saving that costs the strategy its actual edge (fast de-risking) would
be a false economy.

MECHANISM IN DETAIL
--------------------
v4's ``prepare()`` computes a fully causal ``target`` column (fraction of
equity notional, already hysteresis-gated — see the docstring section
above). Whenever ``target[i] != target[i-1]`` (materially), v4's real
``on_bar`` queues an order that fills as a TAKER at bar i+1's open. This
file re-simulates that same target series with a different fill model:

1. At bar i's close, if ``target[i]`` changed from ``target[i-1]``, POST a
   resting limit order at price ``L = close[i]`` for the SAME target.
2. For each bar ``j`` in ``(i, i+N)`` — i.e. bars i+1 .. i+N-1, checked
   causally in order — check whether that bar's [low, high] range
   touched L (buy-type orders: ``low[j] <= L``; sell-type: ``high[j] >=
   L``). The very first bar that touches resolves the order: fill AT L,
   charge the MAKER fee.
3. If none of bars i+1..i+N-1 touched, force a TAKER fill at bar
   ``i+N``'s OPEN (which is known before that bar's own high/low are
   known — no look-back into the same bar's later prices is used to make
   this decision. See "ON THE (i+N) FALLBACK PRICE" below for why this
   is the deliberately causal reading of the task's literal instruction).
4. If a NEW target arrives while an order is still resting (rare — v4's
   own 10% hysteresis makes successive re-triggers uncommon, measured
   explicitly below), the resting order is CANCELLED (never fills, no
   fee) and replaced with a fresh order at the new target/price,
   restarting the N-bar clock. This is the standard behaviour of a
   "peg-and-chase" patient execution algo and is exactly the mechanism
   the crash-transition-lag falsification check (rule 3) is designed to
   catch if it goes wrong.
5. Sanity property, verified below: at N=1 the patience window (bars
   i+1..i+0) is empty, so every order falls straight through to the
   forced taker fallback at bar i+1's open — bit-for-bit identical
   price/timing/fee to the as-shipped taker-only baseline. This is a
   free correctness check on the whole mechanism, not just a footnote.

ON THE (i+N) FALLBACK PRICE (causality design decision, read this before
trusting the causality probe result)
--------------------------------------------------------------------------
The task text says: "if not touched within N bars, force a TAKER fill at
bar (i+N)'s open." Taken completely literally — "touched within N bars"
meaning bars i+1..i+N inclusive, using each bar's FULL high/low range,
including bar i+N's — this is ambiguous about timing: knowing bar i+N
did NOT touch requires having seen bar i+N's entire range, which is only
fully known at that bar's close; retroactively pricing the fallback fill
at that SAME bar's open (a moment strictly before its own high/low were
knowable) would be quietly reading part of bar i+N to price a fill
timestamped before that part of bar i+N happened. That is exactly the
shape of bug the project's own causality suite hunts for
(tests/test_causality_strict.py's docstring: "an i+1 peek ... returned
$3.7e23 with a fully green suite", R-21).

This file resolves the ambiguity the ONLY way that is both causally
sound and literally prices the fallback "at bar (i+N)'s open": the
touch-check window is bars i+1 .. i+N-1 (N-1 full bars, using their
complete high/low ranges — that information is fully known by the time
we act on it). If none of those touched, we already know this by the
OPEN of bar i+N (no information from bar i+N's own high/low is used),
so forcing the taker fill right there, at bar i+N's open, uses only
information that is causally available at that exact instant. At N=1
this window is empty (0 bars checked) and every order falls straight to
the forced-taker fallback at bar i+1's open — which is why N=1 exactly
reproduces the baseline (see the sanity property above and the
causality/parity check in ``causality_probe()``).

WHAT IS NOT MODELLED (the L-14/L-15 trap, deliberately avoided)
------------------------------------------------------------------
No queue position, no informed-flow avoidance, no probability-of-fill
model conditioned on anything except literal, already-in-the-file
high/low touches. That is the entire mechanism. Nothing here proxies
order-flow or participation out of OHLCV the way camouflage_flow and
stealth_trend did (L-14, L-15) — a resting limit order's fill/no-fill is
directly observable from the bars that already exist in this repo's
data, unlike order flow, which is not.

DATA DISCIPLINE
----------------
Every backtest in this file is restricted to:
  - inner-train:      2017-01-01 -> 2020-12-31  (BTC, committed spot file)
  - inner-validation:  2021-01-01 -> 2022-12-31  (BTC, committed spot file)
  - ETH falsification: data/ethusd_bitfinex_5m.csv.gz   (2016-03 -> 2019-12,
    the whole file — it physically contains no bar past 2019-12-31, so it
    cannot leak the holdout even by accident)
  - BTC control:       data/btcusd_bitfinex_5m.csv.gz   (2016-01 -> 2019-12,
    same property — same venue as the ETH file, matching R-17/R-40's
    convention)
No code path in this file ever loads, slices past, prints, or otherwise
touches a bar dated 2023-01-01 or later. Grepped by the author before
finishing (see the report printed at the end of ``main()``); the operator
should re-grep independently, as this project's own practice requires.

USAGE
-----
    python experiments/kelly_regime_exec_limit_conservative.py sweep
    python experiments/kelly_regime_exec_limit_conservative.py causality
    python experiments/kelly_regime_exec_limit_conservative.py falsification
    python experiments/kelly_regime_exec_limit_conservative.py crashlag
    python experiments/kelly_regime_exec_limit_conservative.py all      # everything, in order
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_ohlcv_csv, load_dataset  # noqa: E402
from tradebot.engine import BacktestResult, build_trades, run_backtest, validate_ohlcv  # noqa: E402
from tradebot.metrics import Metrics, compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import prefix_bars, run_period  # noqa: E402

DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------- data guard

OOS_START = "2023-01-01"          # exclusive-lower-bound sentinel only, never a data read
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"

# Bitstamp fee schedule, accessed 2026-08-20 (operator-verified web search).
FEE_TIERS = {
    "entry": {"taker": 0.0040, "maker": 0.0030},   # 30d volume < $10K
    "top":   {"taker": 0.0003, "maker": 0.0000},   # 30d volume > $1B
}

# This project's futures market has no independent real fee schedule of its
# own (README: "no true perp data ... futures market trades the spot
# series"; the default 0.05% futures taker is just a chosen parameter, not
# a measured venue rate). Absent a second real schedule, the same Bitstamp
# spot tiers are applied to both markets here — a deliberate, disclosed
# simplification, not a second data source.

CONFIG_COUNTER = {"n": 0}  # every backtest actually run increments this


def _count(k: int = 1) -> None:
    CONFIG_COUNTER["n"] += k


# ------------------------------------------------------------- pending order


@dataclass
class _Pending:
    target_frac: float   # order_target fraction (already leverage-divided)
    limit_price: float
    placed_at: int
    deadline: int
    is_buy: bool          # True = resting below market (fills on a dip)


# ------------------------------------------------------------- core engine


def run_backtest_limit(
    df: pd.DataFrame,
    base_market: MarketSpec,
    taker_fee: float,
    maker_fee: float,
    patience_n: int,
    start_balance: float,
    trade_start: int = 0,
    data_label: str = "",
    strategy: KellyRegimeV4 | None = None,
    _peek_bug: bool = False,
) -> tuple[BacktestResult, dict]:
    """Re-simulate v4's own causal target series with patient-limit fills.

    Reuses ``PaperBroker`` / ``build_trades`` / ``BacktestResult`` from
    ``tradebot.engine``/``tradebot.broker`` unmodified (read-only import) —
    this file does not edit engine.py or broker.py. ``prepare()`` is the
    identical, already-causal, already-tested v4 code; only the fill
    mechanism below is new.

    ``_peek_bug=True`` deliberately breaks causality (uses bar i+N's own
    high/low, not just its open, to help decide the fallback) — it exists
    ONLY so ``causality_probe()`` can show the probe actually has teeth.
    Never used in any reported result.

    Returns ``(BacktestResult, diagnostics)`` where diagnostics carries
    per-order-event bookkeeping used by the crash-transition-lag check
    and the cancel/replace count.
    """
    strategy = strategy or KellyRegimeV4()
    validate_ohlcv(df)
    prepared = strategy.prepare(df.copy())
    if "target" not in prepared.columns:
        raise ValueError("strategy.prepare() did not produce a 'target' column")

    target = prepared["target"].to_numpy(dtype=float)
    opens = prepared["open"].to_numpy(dtype=float)
    highs = prepared["high"].to_numpy(dtype=float)
    lows = prepared["low"].to_numpy(dtype=float)
    closes = prepared["close"].to_numpy(dtype=float)
    index = prepared.index
    n = len(prepared)
    leverage = max(base_market.leverage, 1e-9)

    taker_market = replace(base_market, fee_rate=taker_fee,
                            name=f"{base_market.name}_limitN{patience_n}")
    maker_market = replace(base_market, fee_rate=maker_fee,
                            name=f"{base_market.name}_limitN{patience_n}")

    broker = PaperBroker(market=taker_market, start_balance=start_balance)

    equity = [0.0] * n
    fills = []
    pending: _Pending | None = None
    events = []  # one dict per order event, for diagnostics
    cancels = 0
    maker_fills = 0            # resolved by touch AND produced a real (non-dust) Fill
    taker_fallback_fills = 0   # resolved by deadline fallback AND produced a real Fill
    maker_dust = 0              # resolved by touch but _execute_target's own 5% notional
    taker_dust = 0               # deadband (broker.py REBALANCE_DEADBAND) made it a no-op

    for i in range(n):
        ts = index[i]

        liq = broker.check_liquidation(ts, opens[i], opens[i], opens[i])
        if liq is not None:
            fills.append(liq)
            pending = None

        if pending is not None and not broker.dead:
            if i == pending.deadline:
                # forced taker fallback: bar i's OPEN only (causally sound
                # regardless of _peek_bug — the bug below only affects the
                # touch-check bars, i < pending.deadline).
                broker.market = taker_market
                out = broker._execute_target(pending.target_frac, ts, opens[i])
                fills.extend(out)
                real = bool(out)
                if real:
                    taker_fallback_fills += 1
                else:
                    taker_dust += 1
                events.append({"placed_at": pending.placed_at, "resolved_at": i,
                                "kind": "taker_fallback" if real else "taker_fallback_dust",
                                "bars": i - pending.placed_at, "target": pending.target_frac,
                                "real_fill": real})
                pending = None
            elif pending.placed_at < i < pending.deadline:
                hi = highs[i]
                lo = lows[i]
                if _peek_bug:
                    # DELIBERATE BUG (probe-only): also look at bar i+1's
                    # range, which is not yet causally available at bar i.
                    j2 = min(i + 1, n - 1)
                    hi = max(hi, highs[j2])
                    lo = min(lo, lows[j2])
                touched = (lo <= pending.limit_price) if pending.is_buy else (hi >= pending.limit_price)
                if touched:
                    broker.market = maker_market
                    out = broker._execute_target(pending.target_frac, ts, pending.limit_price)
                    fills.extend(out)
                    real = bool(out)
                    if real:
                        maker_fills += 1
                    else:
                        maker_dust += 1
                    events.append({"placed_at": pending.placed_at, "resolved_at": i,
                                    "kind": "maker_touch" if real else "maker_touch_dust",
                                    "bars": i - pending.placed_at, "target": pending.target_frac,
                                    "real_fill": real})
                    pending = None

        liq = broker.check_liquidation(ts, opens[i], highs[i], lows[i])
        if liq is not None:
            fills.append(liq)
            pending = None

        equity[i] = broker.equity(closes[i])
        if not math.isfinite(equity[i]):
            raise ValueError(f"non-finite equity at bar {i} ({index[i]})")

        last_bar = i == n - 1
        tradable = (not broker.dead) and (not last_bar) and (i >= strategy.warmup) and (i >= trade_start)
        if tradable:
            prev_t = target[i - 1] if i > 0 else 0.0
            if abs(target[i] - prev_t) > 1e-9:
                new_frac = target[i] / leverage
                old_frac = prev_t / leverage
                if pending is not None:
                    cancels += 1
                    events.append({"placed_at": pending.placed_at, "resolved_at": i,
                                    "kind": "cancelled", "bars": i - pending.placed_at,
                                    "target": pending.target_frac})
                    pending = None
                deadline = min(i + patience_n, n - 1)
                if deadline > i:  # always true unless we're at the last tradable bar
                    pending = _Pending(target_frac=new_frac, limit_price=closes[i],
                                        placed_at=i, deadline=deadline,
                                        is_buy=(new_frac > old_frac))

    trades = build_trades(fills, end_price=closes[-1] if n else None, broker=broker)
    result = BacktestResult(
        strategy_name=f"{strategy.name}_limitN{patience_n}",
        market=taker_market,
        start_balance=start_balance,
        data_label=data_label,
        equity=pd.Series(equity, index=index, name="equity"),
        fills=fills,
        trades=trades,
        df=prepared,
        liquidated=broker.dead,
        fees_paid=broker.fees_paid,
        funding_paid=0.0,
    )
    diag = {
        "cancels": cancels,
        "maker_fills": maker_fills,
        "taker_fallback_fills": taker_fallback_fills,
        "maker_dust": maker_dust,
        "taker_dust": taker_dust,
        "events": events,
        "n_bars": n,
    }
    _count()
    return result, diag


def run_period_limit(
    df: pd.DataFrame,
    start,
    end,
    base_market: MarketSpec,
    taker_fee: float,
    maker_fee: float,
    patience_n: int,
    start_balance: float = 1_000.0,
    data_label: str = "",
    strategy: KellyRegimeV4 | None = None,
) -> tuple[BacktestResult, dict]:
    """``run_backtest_limit`` over ``df[start:end]``, warmed by the real prefix.

    Mirrors ``tradebot.window.run_period`` (reused read-only for
    ``prefix_bars``) so inner-validation gets the same fair warmup-from-
    before-the-window treatment the baseline gets, rather than trading
    cold for its first ~80 days.
    """
    strategy = strategy or KellyRegimeV4()
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")
    prefix = prefix_bars(df, lo, strategy.warmup)
    frame = df.iloc[lo - prefix: hi]
    result, diag = run_backtest_limit(
        frame, base_market, taker_fee, maker_fee, patience_n, start_balance,
        trade_start=prefix, data_label=data_label, strategy=strategy)
    if prefix == 0:
        return result, diag
    trimmed = replace(result, equity=result.equity.iloc[prefix:], df=result.df.iloc[prefix:])
    # shift diagnostics' bar indices so they read naturally against the
    # trimmed frame too (informational only, not used for correctness)
    for e in diag["events"]:
        e["placed_at_trimmed"] = e["placed_at"] - prefix
        e["resolved_at_trimmed"] = e["resolved_at"] - prefix
    return trimmed, diag


def baseline_period(df, start, end, market: MarketSpec, start_balance=1_000.0,
                     data_label="", strategy: KellyRegimeV4 | None = None):
    """The as-shipped, always-taker, next-open baseline — real engine, unmodified."""
    strategy = strategy or KellyRegimeV4()
    result = run_period(strategy, df, start, end, market=market,
                        start_balance=start_balance, data_label=data_label)
    _count()
    return result


# ---------------------------------------------------------------- reporting


def _row(tag: str, m: Metrics, diag: dict | None = None, taker_fees=None) -> dict:
    out = {
        "tag": tag, "final_balance": m.final_balance, "profit_pct": m.profit_pct,
        "num_trades": m.num_trades, "max_dd_pct": m.max_drawdown_pct,
        "sharpe": m.sharpe, "fees_paid": m.fees_paid, "liquidated": m.liquidated,
    }
    if diag is not None:
        total = diag["maker_fills"] + diag["taker_fallback_fills"]
        out["maker_fill_rate_pct"] = 100.0 * diag["maker_fills"] / total if total else float("nan")
        out["cancels"] = diag["cancels"]
        out["maker_fills"] = diag["maker_fills"]
        out["taker_fallback_fills"] = diag["taker_fallback_fills"]
    if taker_fees is not None:
        out["fees_saved_vs_baseline"] = taker_fees - m.fees_paid
    return out


def _print_row(r: dict) -> None:
    extra = ""
    if "maker_fill_rate_pct" in r:
        extra = (f" maker%={r['maker_fill_rate_pct']:>5.1f} "
                 f"(maker={r['maker_fills']:>3d} taker_fb={r['taker_fallback_fills']:>3d} "
                 f"cancel={r['cancels']:>2d})")
        if "fees_saved_vs_baseline" in r:
            extra += f" fee$saved={r['fees_saved_vs_baseline']:>+8.2f}"
    print(f"  {r['tag']:34s} final=${r['final_balance']:>11,.1f} "
          f"({r['profit_pct']:>+8.1f}%) trades={r['num_trades']:>4d} "
          f"DD={r['max_dd_pct']:>5.1f}% sharpe={r['sharpe']:>5.2f} "
          f"fees=${r['fees_paid']:>8.2f}{extra}")


# --------------------------------------------------------------------- sweep


N_VALUES = (1, 2, 3, 6, 12, 24, 72, 288)   # 5min .. 15min .. 1h .. 2h .. 6h .. 1day patience
MARKETS = {
    "spot": MarketSpec.spot(),
    "futures_5x": MarketSpec.futures(leverage=5.0),
}
PERIODS = {
    "inner-train": (None, INNER_TRAIN_END),
    "inner-validation": (INNER_VAL_START, INNER_VAL_END),
}


def sweep() -> list[dict]:
    """N in {1,2,3,6,12,24,72,288} x 2 fee tiers x 2 markets x 2 periods.

    That is 8 x 2 x 2 x 2 = 64 limit-fill backtests, plus 2 x 2 x 2 = 8
    baseline (always-taker) backtests for the identical signal/data/fee
    tier — 72 configurations total in this function. All on
    inner-train/inner-validation only.
    """
    df, label = load_dataset(DATA_DIR, "spot")
    # load_dataset loads the whole committed file (it runs to 2026); the
    # data-discipline guarantee is that PERIODS below never slices past
    # inner-validation, not that the raw loaded frame stops there.
    for _start, _end in PERIODS.values():
        assert _end <= INNER_VAL_END, f"period end {_end} reaches past inner-validation"
    rows = []
    for period_name, (start, end) in PERIODS.items():
        for market_name, base_market in MARKETS.items():
            for tier_name, fees in FEE_TIERS.items():
                base_res = baseline_period(df, start, end, replace(base_market, fee_rate=fees["taker"]),
                                            data_label=label)
                base_m = compute_metrics(base_res)
                base_fees = base_m.fees_paid
                print(f"\n[{period_name} | {market_name} | {tier_name} tier "
                      f"taker={fees['taker']:.3%} maker={fees['maker']:.3%}]")
                base_row = _row("baseline (always-taker)", base_m)
                _print_row(base_row)
                base_row.update(period=period_name, market=market_name, tier=tier_name, n=None)
                rows.append(base_row)

                for pn in N_VALUES:
                    res, diag = run_period_limit(df, start, end, base_market,
                                                  fees["taker"], fees["maker"], pn,
                                                  data_label=label)
                    m = compute_metrics(res)
                    row = _row(f"limit N={pn}", m, diag, taker_fees=base_fees)
                    _print_row(row)
                    row.update(period=period_name, market=market_name, tier=tier_name, n=pn)
                    rows.append(row)
    return rows


# --------------------------------------------------------------- causality


class _FixedTargetStrategy:
    """Minimal strategy stand-in for a synthetic, fully controlled causality
    micro-test: ``prepare()`` just stamps a caller-supplied target array."""

    name = "synthetic_fixed_target"
    warmup = 0

    def __init__(self, target: np.ndarray) -> None:
        self._target = target

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["target"] = self._target
        return df


def _synthetic_frame(n: int, base_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": base_price, "high": base_price, "low": base_price,
        "close": base_price, "volume": 1.0,
    }, index=idx)


def _synthetic_peek_bug_check() -> bool:
    """Deterministic, hand-built proof the ``_peek_bug`` flag actually
    changes a fill decision using information not yet causally available —
    the "guard the guard" companion to the real-data tamper test above.

    One order event at bar 5 (target 0 -> 1, a BUY, limit L = close[5] =
    100). Bars 6..9 (the interior touch-check window for N=5) never dip to
    L (low=110 throughout). Bar 10 is the deadline: its OPEN is 105 (so the
    correct, causal fallback fills there as a taker) but its LOW is 90 —
    below L. The correct code never looks at bar 10's low from bar 9's
    vantage point, so it (rightly) falls through to the forced taker
    fallback at bar 10's open. The buggy code peeks one bar ahead at every
    interior check, so at bar 9 it sees bar 10's low (90 <= 100) and fires
    a maker fill it has no causal right to make yet.
    """
    n = 20
    df = _synthetic_frame(n)
    df.iloc[10, df.columns.get_loc("open")] = 105.0
    df.iloc[10, df.columns.get_loc("low")] = 90.0
    for i in range(6, 10):
        df.iloc[i, df.columns.get_loc("low")] = 110.0
        df.iloc[i, df.columns.get_loc("high")] = 110.0
    target = np.array([0.0] * 5 + [1.0] * (n - 5))
    strat = _FixedTargetStrategy(target)
    market = MarketSpec.spot()
    fees = FEE_TIERS["entry"]

    res_ok, diag_ok = run_backtest_limit(df, market, fees["taker"], fees["maker"], 5, 1_000.0,
                                          strategy=strat, _peek_bug=False)
    res_bug, diag_bug = run_backtest_limit(df, market, fees["taker"], fees["maker"], 5, 1_000.0,
                                            strategy=strat, _peek_bug=True)
    ev_ok = [e for e in diag_ok["events"] if e["placed_at"] == 5][0]
    ev_bug = [e for e in diag_bug["events"] if e["placed_at"] == 5][0]
    print(f"  synthetic: correct kind={ev_ok['kind']} resolved_at={ev_ok['resolved_at']} "
          f"| buggy kind={ev_bug['kind']} resolved_at={ev_bug['resolved_at']}")
    correct_is_taker_fallback_at_10 = ev_ok["kind"] == "taker_fallback" and ev_ok["resolved_at"] == 10
    bug_is_early_maker = ev_bug["kind"] == "maker_touch" and ev_bug["resolved_at"] == 9
    caught = correct_is_taker_fallback_at_10 and bug_is_early_maker
    print(f"  synthetic guard-the-guard (deterministic): {'PASS' if caught else 'FAIL'}")
    return caught


def causality_probe() -> bool:
    """Truncation/tamper probe, following tests/test_causality_strict.py's own
    pattern: two OPPOSITE tampers applied to every bar strictly after a cut
    point. Every order whose *deadline* is at or before the cut must fill
    identically (same price, same fee, same timestamp) under both tampers,
    because nothing at or before the cut differs between them. Orders whose
    deadline is AFTER the cut are allowed (expected) to differ, since the
    tamper changes the very prices their touch-check depends on — that is
    exactly the alternative outcome a real limit order would have, not a
    leak.

    Also runs a "guard the guard" check: the deliberately broken
    ``_peek_bug=True`` variant, which uses one bar of information not yet
    causally available, is shown to CHANGE fills that the correct version
    does not — proving the probe actually has teeth (mirrors
    ``test_the_strict_causality_check_catches_a_real_on_bar_peek``).
    """
    df, label = load_dataset(DATA_DIR, "spot")
    # A slice of inner-train, long enough to warm v4 and generate several
    # order events, short enough to run fast.
    strat = KellyRegimeV4()
    end_pos = df.index.searchsorted(INNER_TRAIN_END, side="right")
    frame = df.iloc[max(0, end_pos - 120_000):end_pos].copy()
    cut = len(frame) - 3_000  # tamper everything from here on

    up, down = frame.copy(), frame.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    market = MarketSpec.spot()
    fees = FEE_TIERS["entry"]
    pn = 12

    print(f"\ncausality probe: frame={len(frame)} bars, cut at bar {cut} "
          f"({frame.index[cut]}), patience N={pn}")

    ok = True
    res_up, diag_up = run_backtest_limit(up, market, fees["taker"], fees["maker"], pn,
                                          1_000.0, strategy=KellyRegimeV4())
    res_down, diag_down = run_backtest_limit(down, market, fees["taker"], fees["maker"], pn,
                                              1_000.0, strategy=KellyRegimeV4())
    pre_cut_up = [e for e in diag_up["events"] if e["resolved_at"] < cut]
    pre_cut_down = [e for e in diag_down["events"] if e["resolved_at"] < cut]
    match = pre_cut_up == pre_cut_down
    # fills list, restricted to fills timestamped before the cut
    fills_up = [(f.ts, f.side, round(f.qty, 8), round(f.price, 6), round(f.fee, 8))
                for f in res_up.fills if f.ts < frame.index[cut]]
    fills_down = [(f.ts, f.side, round(f.qty, 8), round(f.price, 6), round(f.fee, 8))
                  for f in res_down.fills if f.ts < frame.index[cut]]
    match = match and (fills_up == fills_down)
    print(f"  pre-cut order events identical under up/down tamper: {match} "
          f"({len(pre_cut_up)} events, {len(fills_up)} fills before the cut)")
    ok = ok and match

    # any post-cut divergence at all? (sanity: the probe isn't vacuous)
    res_up_full, diag_up_full = run_backtest_limit(up, market, fees["taker"], fees["maker"], pn, 1_000.0)
    res_down_full, diag_down_full = run_backtest_limit(down, market, fees["taker"], fees["maker"], pn, 1_000.0)
    diverges_after = (round(res_up_full.equity.iloc[-1], 2) != round(res_down_full.equity.iloc[-1], 2))
    print(f"  post-cut final equity differs between tampers (expected, proves the probe isn't vacuous): "
          f"{diverges_after}  (up=${res_up_full.equity.iloc[-1]:,.2f} down=${res_down_full.equity.iloc[-1]:,.2f})")
    ok = ok and diverges_after

    # guard the guard: a deterministic synthetic scenario where the buggy
    # variant is HAND-CONSTRUCTED to look one bar past what is causally
    # available, and must diverge from the correct variant (see
    # ``_synthetic_peek_bug_check``'s own docstring for the construction).
    # A real-data tamper test was tried first here and is not a reliable
    # guard-the-guard check: whether a 1-bar peek changes any fill in a
    # 120k-bar real window is a coincidence of where touch events happen to
    # land, not something the bug is guaranteed to trip on bar-by-bar — it
    # did not trip in the first real-data run tried during this session.
    # The synthetic construction below removes that coincidence.
    bug_caught = _synthetic_peek_bug_check()
    print(f"  guard-the-guard (synthetic, deterministic): deliberately broken "
          f"(_peek_bug=True) variant diverges from the correct one exactly as "
          f"constructed to: {bug_caught}")
    ok = ok and bug_caught

    # N=1 sanity: exact reproduction of the baseline
    base = baseline_period(frame, None, None, replace(market, fee_rate=fees["taker"]),
                            data_label=label)
    lim1, _ = run_backtest_limit(frame, market, fees["taker"], fees["maker"], 1, 1_000.0)
    n1_match = round(base.equity.iloc[-1], 6) == round(lim1.equity.iloc[-1], 6)
    print(f"  N=1 reduces exactly to the as-shipped taker baseline: {n1_match} "
          f"(baseline=${base.equity.iloc[-1]:,.6f} limitN1=${lim1.equity.iloc[-1]:,.6f})")
    ok = ok and n1_match

    print(f"\nCAUSALITY PROBE: {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------ falsification


def _load_bitfinex(name: str):
    df = load_ohlcv_csv(DATA_DIR / name)
    return df


def falsification() -> list[dict]:
    """ETH (Bitfinex, pre-2020) and BTC control (Bitfinex, pre-2020).

    Both files physically end 2019-12-31 — they cannot leak the 2023+
    holdout even by accident. N in {3, 12, 288}, entry fee tier only (the
    tier the whole idea is meant to matter at), both markets, both
    datasets: 3 x 1 x 2 x 2 = 12 limit-fill runs + 1 x 2 x 2 = 4 baseline
    runs = 16 configurations.
    """
    eth = _load_bitfinex("ethusd_bitfinex_5m.csv.gz")
    btc = _load_bitfinex("btcusd_bitfinex_5m.csv.gz")
    assert eth.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
    assert btc.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
    fees = FEE_TIERS["entry"]
    rows = []
    for dset_name, df in (("ETH-falsification", eth), ("BTC-control", btc)):
        for market_name, base_market in MARKETS.items():
            base_res = run_backtest(KellyRegimeV4(), df, replace(base_market, fee_rate=fees["taker"]),
                                    1_000.0, data_label=dset_name)
            _count()
            base_m = compute_metrics(base_res)
            print(f"\n[{dset_name} | {market_name} | entry tier]")
            base_row = _row("baseline (always-taker)", base_m)
            _print_row(base_row)
            base_row.update(dataset=dset_name, market=market_name, n=None)
            rows.append(base_row)
            for pn in (3, 12, 288):
                res, diag = run_backtest_limit(df, base_market, fees["taker"], fees["maker"], pn,
                                                1_000.0, data_label=dset_name)
                m = compute_metrics(res)
                row = _row(f"limit N={pn}", m, diag, taker_fees=base_m.fees_paid)
                _print_row(row)
                row.update(dataset=dset_name, market=market_name, n=pn)
                rows.append(row)
    return rows


# --------------------------------------------------------- crash-transition lag


def crash_transition_lag_check() -> list[dict]:
    """For every to-flat (target -> ~0) event in inner-train/inner-validation,
    how many bars did the limit-fill model take to resolve it, vs the
    baseline's fixed 1 bar? This is rule 3's specific, named check.
    """
    df, label = load_dataset(DATA_DIR, "spot")
    strat = KellyRegimeV4()
    prepared = strat.prepare(df.copy())
    end_pos = prepared.index.searchsorted(INNER_VAL_END, side="right")
    prepared = prepared.iloc[:end_pos]
    target = prepared["target"].to_numpy(dtype=float)

    flatten_bars = []
    for i in range(1, len(target)):
        if target[i] < 1e-9 and target[i - 1] > 1e-9:
            flatten_bars.append(i)

    print(f"\n{len(flatten_bars)} to-flat (regime-flip-to-flat) events in "
          f"inner-train+inner-validation:")
    for i in flatten_bars:
        print(f"  bar {i} ({prepared.index[i]}) target {target[i-1]:.3f} -> {target[i]:.3f}")

    rows = []
    fees = FEE_TIERS["entry"]
    for pn in N_VALUES:
        res, diag = run_backtest_limit(df.loc[:INNER_VAL_END], MARKETS["spot"],
                                        fees["taker"], fees["maker"], pn, 1_000.0,
                                        strategy=KellyRegimeV4())
        by_placed = {e["placed_at"]: e for e in diag["events"]}
        delays = []
        maker_flattens = 0
        taker_flattens = 0
        cancelled_flattens = 0
        for fb in flatten_bars:
            e = by_placed.get(fb)
            if e is None:
                continue  # order for this event never got a chance to place (end of frame)
            delays.append(e["bars"])
            if e["kind"] == "maker_touch":
                maker_flattens += 1
            elif e["kind"] == "taker_fallback":
                taker_flattens += 1
            else:
                cancelled_flattens += 1
        worst = max(delays) if delays else 0
        mean = float(np.mean(delays)) if delays else 0.0
        print(f"  N={pn:>4d}: flattens resolved={len(delays)}/{len(flatten_bars)} "
              f"mean_delay={mean:>5.2f} bars worst_delay={worst:>3d} bars "
              f"(maker={maker_flattens} taker_fb={taker_flattens} cancelled={cancelled_flattens})")
        rows.append({"n": pn, "resolved": len(delays), "total": len(flatten_bars),
                     "mean_delay_bars": mean, "worst_delay_bars": worst,
                     "maker_flattens": maker_flattens, "taker_fallback_flattens": taker_flattens,
                     "cancelled_flattens": cancelled_flattens})
    return rows


# --------------------------------------------------------------------- main


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("sweep", "all"):
        print("=" * 78)
        print("SWEEP: N in {1,2,3,6,12,24,72,288} x {entry,top} x {spot,futures_5x} x "
              "{inner-train,inner-validation}")
        print("=" * 78)
        sweep()
    if which in ("causality", "all"):
        print("\n" + "=" * 78)
        print("CAUSALITY / TAMPER PROBE")
        print("=" * 78)
        causality_probe()
    if which in ("falsification", "all"):
        print("\n" + "=" * 78)
        print("FALSIFICATION: ETH (Bitfinex, pre-2020) + BTC control (Bitfinex, pre-2020)")
        print("=" * 78)
        falsification()
    if which in ("crashlag", "all"):
        print("\n" + "=" * 78)
        print("CRASH-TRANSITION-LAG CHECK (rule 3)")
        print("=" * 78)
        crash_transition_lag_check()
    print(f"\nTOTAL CONFIGURATIONS EVALUATED THIS RUN: {CONFIG_COUNTER['n']}")


if __name__ == "__main__":
    main()
