#!/usr/bin/env python
"""R-77 (conservative branch): B-24 — a properly pre-registered re-run of R-56's
"least-bad" N in {2,3,6,12,24} patient-limit-order execution subset for
`kelly_regime_v4`.

CONSTRAINT ATTACKED: COST (costs that scale with the signal), same as R-56.
This does NOT touch kelly_regime_v4's SIZE/vote signal at all.

BACKLOG ITEM: B-24. R-56 (`experiments/kelly_regime_exec_limit_conservative.py`)
swept N in {1,2,3,6,12,24,72,288} for a "patient limit, taker-fallback-after-N"
execution model and found: fee savings real and monotonic, but no Sharpe
improvement anywhere clears the project's own +-0.2 noise floor, AND for
N>=3 the pre-registered crash-transition-lag falsification FAILS (patient
limit orders delay v4's flip-to-flat de-risking events during crashes).
R-56 observed, post hoc, that N in [2,24] looked "least bad" -- it captures
most of the fee saving while (mostly) avoiding the N>=72 failure region --
but that subset was never the pre-registered decision set, so R-56 correctly
declined to promote it without a proper pre-registration. B-24 is that gap.
This file closes it: N in {2,3,6,12,24} is now tested as a first-class,
pre-registered decision set (not chosen after seeing results -- it is
exactly R-56's own post-hoc observation, frozen and re-run honestly), with
the SAME falsification battery R-56 used, and the SAME pre-registered
1-2 bar crash-transition-lag threshold -- not loosened.

Not a duplicate of R-56 itself: R-56 swept 8 N-values as one undifferentiated
sweep and never pre-registered a decision rule over any subset of them; this
file pre-registers a decision rule over exactly 5 of those 8 values BEFORE
running anything, and applies it mechanically. Not a duplicate of the R-77
novel branch (`experiments/r77_novel_execution_regime_adaptive.py`, a
separate, independent agent's file, not read or touched here).

MECHANISM (one sentence, unchanged from R-56): resting a maker-fee limit
order at the signal's own decision price, with a hard taker-fee fallback
after N bars, should recover some of the taker/maker fee spread without
materially delaying the crash-transition flattens that are v4's actual
edge (L-01/README: its whole edge lives in the windows that contain a
crash) -- tested here only over the N-range R-56's own results suggested
was "least bad", as a genuine pre-registered test of that specific claim.

PRE-REGISTERED DECISION RULE (write this down BEFORE running anything, per
ROUTINE.md step 4 -- do NOT move this after seeing results):

PROMOTE only if ALL of:
  (a) beats kelly_regime_v4 on inner-validation Sharpe by more than the
      +-0.2 noise floor, OR is a clear drawdown/tail improvement, on BOTH
      spot and futures_5x;
  (b) ETH falsification passes directionally (same sign as BTC, not
      opposite);
  (c) BTC pre-2020 control does not decisively fail (candidate not
      dramatically worse than v4 on that window);
  (d) crash-transition-lag test passes at the 1-2 bar mean-lag threshold
      for ALL FIVE tested N values (2,3,6,12,24), not just the small ones;
  (e) the N in {2,3,6,12,24} neighbourhood is a plateau (similar
      performance across N), not a single lucky peak.
Anything else -> NEGATIVE. If this rule changes after seeing any result,
that will be stated explicitly in the final report and the result
downgraded to in-sample -- not adjusted quietly.

FALSIFICATION (pre-registered before running anything, per ROUTINE.md
step 2, identical in spirit to R-56's): this idea is dead if (a) it does
not replicate directionally on the ETH falsification pair; (b) it fails a
pre-2020 BTC-only control window; or (c) regime-flip-to-flat events get
delayed by more than 1-2 bars vs the taker baseline during crash
transitions for any of the five tested N.

MECHANISM IN DETAIL, ON THE (i+N) FALLBACK PRICE, AND WHAT IS NOT MODELLED
---------------------------------------------------------------------------
Identical to R-56's conservative branch -- see that file's docstring for
the full causal-design discussion (touch-check window is bars i+1..i+N-1,
using their complete high/low ranges; forced taker fallback prices at bar
i+N's OPEN only, never that bar's own high/low). That logic is reused
here essentially verbatim (adapted into this disjoint file, not by
importing or editing the R-56 file) because this round does not change
the fill mechanism at all -- it changes only which N values are treated
as the pre-registered decision set, and adds the properly pre-registered
promotion rule R-56 never wrote down for this subset.

DATA DISCIPLINE
----------------
Every backtest in this file is restricted to:
  - inner-train:      2017-01-01 -> 2020-12-31  (BTC, committed spot file)
  - inner-validation:  2021-01-01 -> 2022-12-31  (BTC, committed spot file)
  - ETH falsification: data/ethusd_bitfinex_5m.csv.gz   (physically ends
    2019-12, cannot leak the holdout even by accident)
  - BTC control:       data/btcusd_bitfinex_5m.csv.gz, Bitfinex venue,
    pre-2020 only (same file/property as R-56 and R-17/R-40's convention)
No code path in this file ever loads, slices past, prints, or otherwise
touches a bar dated 2023-01-01 or later. This is NOT a holdout round.

USAGE
-----
    python experiments/r77_conservative_narrowcap_execution.py sweep
    python experiments/r77_conservative_narrowcap_execution.py causality
    python experiments/r77_conservative_narrowcap_execution.py falsification
    python experiments/r77_conservative_narrowcap_execution.py crashlag
    python experiments/r77_conservative_narrowcap_execution.py decision
    python experiments/r77_conservative_narrowcap_execution.py all      # everything, in order
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

# Bitstamp fee schedule (same schedule R-56 verified, accessed 2026-08-20).
FEE_TIERS = {
    "entry": {"taker": 0.0040, "maker": 0.0030},   # 30d volume < $10K
    "top":   {"taker": 0.0003, "maker": 0.0000},   # 30d volume > $1B
}

# Same disclosed simplification as R-56: this project's futures market has
# no independent real fee schedule, so the Bitstamp spot tiers are applied
# to both markets.

CONFIG_COUNTER = {"n": 0, "sweep": 0, "falsification": 0, "crashlag": 0, "causality": 0}
_PHASE = {"name": "unassigned"}


def _count(k: int = 1) -> None:
    CONFIG_COUNTER["n"] += k
    CONFIG_COUNTER[_PHASE["name"]] = CONFIG_COUNTER.get(_PHASE["name"], 0) + k


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

    Adapted from R-56's conservative branch (same mechanism, same causal
    design decisions -- see this file's module docstring). Reuses
    ``PaperBroker`` / ``build_trades`` / ``BacktestResult`` from
    ``tradebot.engine``/``tradebot.broker`` unmodified (read-only import);
    only the fill mechanism is (adapted) new code in this disjoint file.

    ``_peek_bug=True`` deliberately breaks causality -- exists ONLY so
    ``causality_probe()`` can show the probe has teeth. Never used in any
    reported result.
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
    events = []
    cancels = 0
    maker_fills = 0
    taker_fallback_fills = 0
    maker_dust = 0
    taker_dust = 0

    for i in range(n):
        ts = index[i]

        liq = broker.check_liquidation(ts, opens[i], opens[i], opens[i])
        if liq is not None:
            fills.append(liq)
            pending = None

        if pending is not None and not broker.dead:
            if i == pending.deadline:
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
                if deadline > i:
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
    """``run_backtest_limit`` over ``df[start:end]``, warmed by the real prefix."""
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
    for e in diag["events"]:
        e["placed_at_trimmed"] = e["placed_at"] - prefix
        e["resolved_at_trimmed"] = e["resolved_at"] - prefix
    return trimmed, diag


def baseline_period(df, start, end, market: MarketSpec, start_balance=1_000.0,
                     data_label="", strategy: KellyRegimeV4 | None = None):
    """The as-shipped, always-taker, next-open baseline -- real engine, unmodified."""
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

# PRE-REGISTERED N SET (B-24): {2,3,6,12,24} only -- R-56's own "least bad"
# observation, deliberately excluding N>=72's already-confirmed crash-lag
# failure mode. This is NOT chosen after seeing this round's own results.
N_VALUES = (2, 3, 6, 12, 24)
MARKETS = {
    "spot": MarketSpec.spot(),
    "futures_5x": MarketSpec.futures(leverage=5.0),
}
PERIODS = {
    "inner-train": (None, INNER_TRAIN_END),
    "inner-validation": (INNER_VAL_START, INNER_VAL_END),
}


def sweep() -> list[dict]:
    """N in {2,3,6,12,24} x 2 fee tiers x 2 markets x 2 periods.

    That is 5 x 2 x 2 x 2 = 40 limit-fill backtests, plus 2 x 2 x 2 = 8
    baseline (always-taker) backtests -- 48 configurations total in this
    function. All on inner-train/inner-validation only.
    """
    _PHASE["name"] = "sweep"
    df, label = load_dataset(DATA_DIR, "spot")
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
                base_row.update(period=period_name, market=market_name, tier=tier_name, n=None,
                                 sharpe_delta=0.0)
                rows.append(base_row)

                for pn in N_VALUES:
                    res, diag = run_period_limit(df, start, end, base_market,
                                                  fees["taker"], fees["maker"], pn,
                                                  data_label=label)
                    m = compute_metrics(res)
                    row = _row(f"limit N={pn}", m, diag, taker_fees=base_fees)
                    _print_row(row)
                    row.update(period=period_name, market=market_name, tier=tier_name, n=pn,
                                sharpe_delta=m.sharpe - base_m.sharpe)
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
    changes a fill decision using information not yet causally available.

    Same construction as R-56's guard-the-guard test, reused here since
    the underlying mechanism is unchanged: one BUY order at bar 5
    (target 0 -> 1, limit L=100). Bars 6..9 never touch L. Bar 10 (the
    deadline for N=5) has open=105 but low=90 -- the correct code never
    peeks at bar 10's low from bar 9's vantage point and falls through to
    the forced taker fallback at bar 10's open; the buggy code peeks one
    bar ahead at every interior check and fires an early, causally
    illegal maker fill at bar 9.
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
    pattern and R-56's own causality_probe(): two OPPOSITE tampers applied
    to every bar strictly after a cut point. Every order whose *deadline*
    is at or before the cut must fill identically under both tampers.
    Orders whose deadline is AFTER the cut are allowed to differ (that is
    the alternative outcome a real limit order would have, not a leak).

    Also runs the deterministic "guard the guard" synthetic check, and the
    N=1 sanity property (the patience window collapses to empty, so every
    order falls straight to the forced taker fallback at bar i+1's open --
    an exact, bit-for-bit reproduction of the as-shipped baseline). N=1 is
    NOT part of this round's pre-registered N_VALUES; it is used here only
    as a free correctness diagnostic of the shared fill mechanism.
    """
    _PHASE["name"] = "causality"
    df, label = load_dataset(DATA_DIR, "spot")
    strat = KellyRegimeV4()
    end_pos = df.index.searchsorted(INNER_TRAIN_END, side="right")
    frame = df.iloc[max(0, end_pos - 120_000):end_pos].copy()
    cut = len(frame) - 3_000

    up, down = frame.copy(), frame.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    market = MarketSpec.spot()
    fees = FEE_TIERS["entry"]
    pn = 12  # inside this round's pre-registered N set

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
    fills_up = [(f.ts, f.side, round(f.qty, 8), round(f.price, 6), round(f.fee, 8))
                for f in res_up.fills if f.ts < frame.index[cut]]
    fills_down = [(f.ts, f.side, round(f.qty, 8), round(f.price, 6), round(f.fee, 8))
                  for f in res_down.fills if f.ts < frame.index[cut]]
    match = match and (fills_up == fills_down)
    print(f"  pre-cut order events identical under up/down tamper: {match} "
          f"({len(pre_cut_up)} events, {len(fills_up)} fills before the cut)")
    ok = ok and match

    res_up_full, diag_up_full = run_backtest_limit(up, market, fees["taker"], fees["maker"], pn, 1_000.0)
    res_down_full, diag_down_full = run_backtest_limit(down, market, fees["taker"], fees["maker"], pn, 1_000.0)
    diverges_after = (round(res_up_full.equity.iloc[-1], 2) != round(res_down_full.equity.iloc[-1], 2))
    print(f"  post-cut final equity differs between tampers (expected, proves the probe isn't vacuous): "
          f"{diverges_after}  (up=${res_up_full.equity.iloc[-1]:,.2f} down=${res_down_full.equity.iloc[-1]:,.2f})")
    ok = ok and diverges_after

    bug_caught = _synthetic_peek_bug_check()
    print(f"  guard-the-guard (synthetic, deterministic): deliberately broken "
          f"(_peek_bug=True) variant diverges from the correct one exactly as "
          f"constructed to: {bug_caught}")
    ok = ok and bug_caught

    base = baseline_period(frame, None, None, replace(market, fee_rate=fees["taker"]),
                            data_label=label)
    lim1, _ = run_backtest_limit(frame, market, fees["taker"], fees["maker"], 1, 1_000.0)
    n1_match = round(base.equity.iloc[-1], 6) == round(lim1.equity.iloc[-1], 6)
    print(f"  N=1 reduces exactly to the as-shipped taker baseline (free diagnostic, "
          f"N=1 not in this round's pre-registered set): {n1_match} "
          f"(baseline=${base.equity.iloc[-1]:,.6f} limitN1=${lim1.equity.iloc[-1]:,.6f})")
    ok = ok and n1_match

    print(f"\nCAUSALITY PROBE: {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------ falsification


def _load_bitfinex(name: str):
    df = load_ohlcv_csv(DATA_DIR / name)
    return df


def falsification() -> list[dict]:
    """ETH (Bitfinex, pre-2020) falsification AND BTC pre-2020 control window.

    Both files physically end 2019-12-31 -- they cannot leak the 2023+
    holdout even by accident. N in {2,3,6,12,24} (this round's full
    pre-registered set, not a further-narrowed sample), entry fee tier
    only (the tier the whole idea is meant to matter at), both markets,
    both datasets: 5 x 1 x 2 x 2 = 20 limit-fill runs + 1 x 2 x 2 = 4
    baseline runs = 24 configurations.
    """
    _PHASE["name"] = "falsification"
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
            base_row.update(dataset=dset_name, market=market_name, n=None, sharpe_delta=0.0)
            rows.append(base_row)
            for pn in N_VALUES:
                res, diag = run_backtest_limit(df, base_market, fees["taker"], fees["maker"], pn,
                                                1_000.0, data_label=dset_name)
                m = compute_metrics(res)
                row = _row(f"limit N={pn}", m, diag, taker_fees=base_m.fees_paid)
                _print_row(row)
                row.update(dataset=dset_name, market=market_name, n=pn,
                            sharpe_delta=m.sharpe - base_m.sharpe)
                rows.append(row)
    return rows


# --------------------------------------------------------- crash-transition lag


def crash_transition_lag_check() -> list[dict]:
    """For every to-flat (target -> ~0) event in inner-train/inner-validation,
    how many bars did each of this round's five pre-registered N-values
    take to resolve it, vs the baseline's fixed 1 bar? Five configurations
    (one per N).
    """
    _PHASE["name"] = "crashlag"
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
                continue
            delays.append(e["bars"])
            if e["kind"] == "maker_touch":
                maker_flattens += 1
            elif e["kind"] == "taker_fallback":
                taker_flattens += 1
            else:
                cancelled_flattens += 1
        worst = max(delays) if delays else 0
        mean = float(np.mean(delays)) if delays else 0.0
        n_over_threshold = sum(1 for d in delays if d > 2)
        passed = mean <= 2.0
        print(f"  N={pn:>4d}: flattens resolved={len(delays)}/{len(flatten_bars)} "
              f"mean_delay={mean:>5.2f} bars worst_delay={worst:>3d} bars "
              f"over_2bar_threshold={n_over_threshold:>3d} "
              f"(maker={maker_flattens} taker_fb={taker_flattens} cancelled={cancelled_flattens}) "
              f"-> {'PASS' if passed else 'FAIL'}")
        rows.append({"n": pn, "resolved": len(delays), "total": len(flatten_bars),
                     "mean_delay_bars": mean, "worst_delay_bars": worst,
                     "n_over_2bar_threshold": n_over_threshold,
                     "maker_flattens": maker_flattens, "taker_fallback_flattens": taker_flattens,
                     "cancelled_flattens": cancelled_flattens, "passed_1_2_bar_rule": passed})
    return rows


# --------------------------------------------------------------------- main


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("sweep", "all"):
        print("=" * 78)
        print("SWEEP: N in {2,3,6,12,24} x {entry,top} x {spot,futures_5x} x "
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
        print("CRASH-TRANSITION-LAG CHECK (pre-registered rule (d))")
        print("=" * 78)
        crash_transition_lag_check()
    print(f"\nTOTAL CONFIGURATIONS EVALUATED THIS RUN: {CONFIG_COUNTER['n']}")
    print(f"  breakdown: sweep={CONFIG_COUNTER.get('sweep', 0)} "
          f"falsification={CONFIG_COUNTER.get('falsification', 0)} "
          f"crashlag={CONFIG_COUNTER.get('crashlag', 0)} "
          f"causality-diagnostics={CONFIG_COUNTER.get('causality', 0)}")


if __name__ == "__main__":
    main()
