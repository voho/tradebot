"""R-72 (conservative branch): B-30, made explicit and general.

=====================================================================
WHAT THIS FILE IS
=====================================================================

B-30 (filed by R-64, section B / docs/LEDGER.md) named a fact nobody had
made explicit before: ``broker.py``'s ``REBALANCE_DEADBAND = 0.05`` is
5% of *max notional* (``equity x leverage``), not 5% of equity. On spot
(leverage 1x) those are the same thing. On 5x futures they are not: 5%
of max notional is **25% of equity**, a much coarser filter, and it
silently drops same-sign target adjustments below that size. R-64
measured this once, on ``kelly_regime_v4`` alone (48.1% / 53.8%
fill-through on futures inner-train/inner-val, vs 86.0% / 96.2% on
spot). This file does two things:

1. **Measurement.** Extends that measurement from one strategy to all 25
   registered strategies, on spot and 5x futures, on both inner splits
   (2017 -> 2020-12-31, 2021-01-01 -> 2022-12-31). No promotion, no
   strategy modification -- a strategy-agnostic instrumented broker
   counts *intended* target changes (new "asks" made to the broker) and
   *filled* ones (asks that produced a fill), for every registered
   strategy, unmodified.

2. **A bounded, isolated, strategy-local test.** ``kelly_regime_v4``
   computes its target ignorant of the broker's deadband, so on futures
   roughly half its intended re-sizes silently never happen and its
   *realized* exposure path can diverge from its own vote. This section
   asks: if the deadband were instead a constant 5% of *equity*
   (dividing by leverage, so futures gets the same economics as spot)
   rather than 5% of max notional, what happens to fill rate, turnover,
   fees, tracking error and growth/Sharpe/drawdown? The change lives
   entirely in an ``EquityScaledDeadbandBroker`` subclass defined below
   and monkeypatched into ``tradebot.engine`` only for the duration of
   this file's own runs. ``src/tradebot/broker.py`` is never edited, and
   no registered strategy file is edited. The strategy under test
   (``KellyRegimeV4``) is imported and run **unmodified**.

=====================================================================
WHICH CONSTRAINT, WHICH BACKLOG ITEM, WHAT WOULD FALSIFY IT
=====================================================================

Constraint attacked: **COST** ("costs scale with the signal") by way of
**methodology** -- B-30 is explicit that this is not a strategy question:
"changing it would move every number in the comparison table. What is
needed is the measurement made explicit... Until then the futures column
carries a second caveat alongside funding."

Not a duplicate of R-64 (which measured one strategy, once, as a
byproduct of a different round) or R-66 (which reconfirmed R-64's number
and additionally found the spot-side accidental-minimum-step-filter
effect). This file generalizes both to the full strategy roster and adds
a bounded counterfactual the prior rounds explicitly declined to run
("not a bug to 'fix' unasked").

**Mechanism (part 2).** ``REBALANCE_DEADBAND * equity * leverage`` grows
with leverage; ``REBALANCE_DEADBAND * equity`` does not. If the coarser
futures filter is *why* futures fill-through lags spot's, dividing the
threshold by leverage should raise futures fill-through toward spot's
level, raise turnover and fees on futures, and pull the realized
exposure path closer to the strategy's own intended target path.

**What would make this a null / uninteresting result (named before
running anything).** If tightening the band (more fills) mostly adds
fee-eating churn without changing final growth/Sharpe materially inside
the +/-0.2 Sharpe noise floor (R-20), or if kelly_regime_v4's target path
is dominated by *large* moves (sign flips, full-to-flat swings) that
already clear even the coarse deadband, then most of the "missing" fills
were economically negligible and the second futures caveat, while
correct, would not be worth a follow-up.

=====================================================================
DATA DISCIPLINE
=====================================================================

BTC (Bitstamp spot 5m, ``data/btcusd_spot_5m.csv.gz``) is truncated to
``2022-12-31`` **immediately on load**, before any split, any strategy,
or any measurement touches it. ``assert_no_holdout(df)`` re-checks the
max timestamp after every load as a second, independent guard. Splits are
the project-standard ``INNER_TRAIN`` (... -> 2020-12-31) and
``INNER_VAL`` (2021-01-01 -> 2022-12-31); no bar dated 2023-01-01 or
later is read anywhere in this file. Holdout consultations added: 0.

Usage::

    python experiments/r72_conservative_deadband.py fillrate   # part 1
    python experiments/r72_conservative_deadband.py deadband   # part 2
    python experiments/r72_conservative_deadband.py all        # both
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tradebot.engine as engine_mod  # noqa: E402
from tradebot.broker import MarketSpec, PaperBroker, REBALANCE_DEADBAND  # noqa: E402
from tradebot.data import load_ohlcv_csv  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.orders import Side  # noqa: E402
from tradebot.registry import available_strategies, get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

OUT_DIR = ROOT / "experiments" / "reports"

OOS_START = "2023-01-01"
INNER_TRAIN = (None, "2020-12-31")
INNER_VAL = ("2021-01-01", "2022-12-31")
SPLITS = (("inner-train", INNER_TRAIN), ("inner-val", INNER_VAL))

FEE_BASE = 0.0010     # spot taker, the table's convention
FUT_FEE = 0.0005      # futures taker, r64_shared's convention
FUT_LEVERAGE = 5.0

BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)
SHARPE_NOISE_FLOOR = 0.2  # R-20

# Counts every backtest run through `measure`, so the trials count is
# observed rather than remembered.
_CONFIGS = [0]


def configs_evaluated() -> int:
    return _CONFIGS[0]


def spot(fee: float = FEE_BASE) -> MarketSpec:
    return MarketSpec.spot(fee_rate=fee)


def futures(fee: float = FUT_FEE, leverage: float = FUT_LEVERAGE) -> MarketSpec:
    return MarketSpec.futures(leverage=leverage, fee_rate=fee)


# ================================================================= data


def load_btc_inner() -> pd.DataFrame:
    """BTC truncated at 2022-12-31 immediately on load. No later bar is
    ever held in memory by this file."""
    df = load_ohlcv_csv(ROOT / "data" / "btcusd_spot_5m.csv.gz")
    df = df.loc[:"2022-12-31"]
    assert_no_holdout(df)
    return df


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Second, independent guard: the max timestamp in any frame this file
    touches must be strictly before OOS_START."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# ============================================================ instrumented broker
#
# Part 1's measurement instrument. Subclasses PaperBroker (imported, not
# copied blind) and overrides only `execute`, purely to observe what
# already happens: it logs every order carrying a `target` that represents
# a genuine new ask (the target value changed from the last one this
# broker instance received) and whether that ask produced >=1 Fill. The
# broker's actual decision logic (`_execute_target`, including
# REBALANCE_DEADBAND) is untouched -- this class adds observation, not a
# new policy. Applied to `tradebot.engine` only via a context manager, for
# the duration of one `run_period` call, then restored.


class InstrumentedBroker(PaperBroker):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.intended_log: list[tuple[object, float, bool]] = []
        self._last_target: float | None = None

    def execute(self, order, ts, price):
        fills = super().execute(order, ts, price)
        if order.target is not None:
            t = float(order.target)
            is_new_ask = self._last_target is None or abs(t - self._last_target) > 1e-9
            if is_new_ask:
                self.intended_log.append((ts, t, len(fills) > 0))
            self._last_target = t
        return fills


_LAST_BROKER: list[PaperBroker] = []


def _capturing_post_init(cls):
    orig = cls.__post_init__

    def wrapped(self):
        orig(self)
        _LAST_BROKER.append(self)

    return wrapped


InstrumentedBroker.__post_init__ = _capturing_post_init(InstrumentedBroker)


@contextlib.contextmanager
def _patched_broker(broker_cls):
    """Swap `tradebot.engine`'s module-level `PaperBroker` name for the
    duration of the block, then restore it. `run_backtest` looks up
    `PaperBroker` as a module global at call time, so this is sufficient
    to redirect broker construction without touching broker.py or any
    registered strategy file."""
    orig = engine_mod.PaperBroker
    engine_mod.PaperBroker = broker_cls
    try:
        yield
    finally:
        engine_mod.PaperBroker = orig


def run_instrumented(strategy, df, window, market, balance: float = 1_000.0):
    """One backtest with the instrumented broker. Returns (result, broker)."""
    start, end = window
    _CONFIGS[0] += 1
    _LAST_BROKER.clear()
    with _patched_broker(InstrumentedBroker):
        res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    assert len(_LAST_BROKER) == 1, "expected exactly one broker instance per run"
    return res, _LAST_BROKER[0]


def fill_rate(strategy_name: str, df: pd.DataFrame, window, market: MarketSpec) -> dict:
    """Intended asks vs fills actually executed, for one strategy/market/window."""
    strat = get_strategy(strategy_name)
    res, broker = run_instrumented(strat, df, window, market)
    log = broker.intended_log
    intended = len(log)
    filled = sum(1 for _, _, ok in log if ok)
    return dict(
        strategy=strategy_name,
        market=market.name,
        intended=intended,
        fills=filled,
        below_band=intended - filled,
        fill_rate=(filled / intended if intended else float("nan")),
        n_result_fills=len(res.fills),
    )


# =================================================================== part 1


def cmd_fillrate(df: pd.DataFrame) -> pd.DataFrame:
    names = sorted(available_strategies())
    print("=" * 110)
    print(f"PART 1: fill-through, every registered strategy ({len(names)}) x "
          f"{{spot, futures_5x}} x {{inner-train, inner-val}}")
    print(f"broker.REBALANCE_DEADBAND = {REBALANCE_DEADBAND}  "
          f"(spot threshold = {REBALANCE_DEADBAND:.0%} of equity; "
          f"futures {FUT_LEVERAGE:g}x threshold = "
          f"{REBALANCE_DEADBAND * FUT_LEVERAGE:.0%} of equity)")
    print("=" * 110)

    markets = [("spot", spot(FEE_BASE)), ("futures_5x", futures())]
    rows: list[dict] = []
    t0 = time.time()
    for name in names:
        for split, window in SPLITS:
            for mkt_label, market in markets:
                try:
                    row = fill_rate(name, df, window, market)
                except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
                    row = dict(strategy=name, market=mkt_label, intended=None,
                               fills=None, below_band=None, fill_rate=float("nan"),
                               n_result_fills=None, error=str(exc))
                row["split"] = split
                rows.append(row)
        elapsed = time.time() - t0
        print(f"  ... {name:<24} done  ({elapsed:6.1f}s elapsed)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "r72_fillrate_all_strategies.csv", index=False)

    print()
    print("-" * 110)
    print(f"{'strategy':<24} {'split':<12} {'market':<11} {'intended':>9} "
          f"{'fills':>7} {'fill%':>7}")
    print("-" * 110)
    for _, r in out.iterrows():
        if pd.isna(r.get("intended")) if "error" in out.columns else False:
            print(f"{r['strategy']:<24} {r['split']:<12} {r['market']:<11}"
                  f"  ERROR: {r.get('error')}")
            continue
        fr = r["fill_rate"]
        fr_s = f"{fr:6.1%}" if pd.notna(fr) else "   n/a"
        print(f"{r['strategy']:<24} {r['split']:<12} {r['market']:<11} "
              f"{r['intended']:>9} {r['fills']:>7} {fr_s:>7}")
    return out


def cmd_fillrate_summary(out: pd.DataFrame) -> None:
    """Per-market, per-split aggregate + the spot-vs-futures gap per strategy."""
    print()
    print("=" * 110)
    print("SUMMARY: mean fill rate by market x split (unweighted across strategies "
          "with >=1 intended ask)")
    print("=" * 110)
    ok = out[out["intended"].notna() & (out["intended"] > 0)]
    agg = ok.groupby(["market", "split"])["fill_rate"].agg(["mean", "median", "count"])
    print(agg)

    print()
    print("Strategies with the largest spot-minus-futures fill-rate gap "
          "(inner-train; only strategies with >=20 intended asks on both "
          "markets, i.e. where the deadband has room to bind):")
    tr = ok[ok["split"] == "inner-train"]
    piv = tr.pivot_table(index="strategy", columns="market", values="fill_rate")
    piv_n = tr.pivot_table(index="strategy", columns="market", values="intended")
    active = piv_n.min(axis=1) >= 20
    piv = piv.loc[active]
    if {"spot", "futures_5x"}.issubset(piv.columns) and len(piv):
        piv["gap"] = piv["spot"] - piv["futures_5x"]
        print(piv.sort_values("gap", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))
    else:
        print("  (fewer than 2 markets present, or no strategy clears the >=20-ask bar)")


# =================================================================== part 2
#
# The equity-scaled deadband. One line changed from PaperBroker's own
# `_execute_target`, copied faithfully otherwise (this is the same
# discipline R-64's TradeToBoundary used for KellyRegimeV3.prepare): the
# threshold is `REBALANCE_DEADBAND * equity` instead of
# `REBALANCE_DEADBAND * equity * leverage`. At leverage=1 (spot) the two
# are identical by construction -- asserted below as a regression check.


class EquityScaledDeadbandBroker(PaperBroker):
    """PaperBroker with REBALANCE_DEADBAND expressed as a fraction of
    EQUITY instead of a fraction of MAX NOTIONAL (equity x leverage).

    The only edit vs `PaperBroker._execute_target`: the comparison
    `abs(delta) * price < REBALANCE_DEADBAND * max_notional` becomes
    `abs(delta) * price < REBALANCE_DEADBAND * self.equity(price)`.
    Every other line -- the target clamp, the sign-flip close-then-open,
    the `_max_qty` sizing, the final `_transact` -- is copied unchanged
    from `src/tradebot/broker.py` as of this round. `src/tradebot/broker.py`
    itself is never edited; this class exists only in this experiment file
    and is applied only via the `_patched_broker` context manager below.
    """

    def _execute_target(self, target: float, ts, price: float):
        import math as _math
        from tradebot.orders import Fill  # noqa: F401  (type doc only)

        if not _math.isfinite(target):
            raise ValueError(f"order target must be finite, got {target!r}")
        lo = -1.0 if self.market.allow_short else 0.0
        target = min(1.0, max(lo, target))

        fills = []
        if (self.pos != 0.0 and target != 0.0
                and _math.copysign(1.0, target) != _math.copysign(1.0, self.pos)):
            fill = self._transact(ts, -self.pos, price)
            if fill:
                fills.append(fill)
            if self.dead:
                return fills

        desired = (_math.copysign(self._max_qty(price) * abs(target), target)
                   if target != 0.0 else 0.0)
        delta = desired - self.pos
        # ---- THE ONE CHANGE vs PaperBroker._execute_target ------------------
        # equity, not equity x leverage.
        equity_threshold = self.equity(price)
        if target != 0.0 and self.pos != 0.0 and equity_threshold > 0:
            if abs(delta) * price < REBALANCE_DEADBAND * equity_threshold:
                return fills
        # ----------------------------------------------------------------------
        fill = self._transact(ts, delta, price)
        if fill:
            fills.append(fill)
        return fills


class EquityScaledInstrumentedBroker(EquityScaledDeadbandBroker):
    """The equity-scaled broker, plus the same intended-ask instrumentation
    InstrumentedBroker provides, so part 2 gets fill-rate/turnover numbers
    on the same footing as part 1's table."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.intended_log: list[tuple[object, float, bool]] = []
        self._last_target: float | None = None

    def execute(self, order, ts, price):
        fills = super().execute(order, ts, price)
        if order.target is not None:
            t = float(order.target)
            is_new_ask = self._last_target is None or abs(t - self._last_target) > 1e-9
            if is_new_ask:
                self.intended_log.append((ts, t, len(fills) > 0))
            self._last_target = t
        return fills


EquityScaledInstrumentedBroker.__post_init__ = _capturing_post_init(
    EquityScaledInstrumentedBroker)


def regression_check_spot_identical() -> bool:
    """At leverage=1, REBALANCE_DEADBAND * equity == REBALANCE_DEADBAND *
    (equity * leverage), so the default and equity-scaled brokers must
    make bit-identical decisions on spot. Run a short slice of
    kelly_regime_v4 through both and compare fills exactly."""
    df = load_btc_inner()
    frame = df.loc[:"2017-06-30"]
    m = spot(FEE_BASE)

    with _patched_broker(PaperBroker):
        res_default = run_period(KellyRegimeV4(), frame, None, None, market=m)
    with _patched_broker(EquityScaledDeadbandBroker):
        res_scaled = run_period(KellyRegimeV4(), frame, None, None, market=m)

    same_n = len(res_default.fills) == len(res_scaled.fills)
    same_eq = np.allclose(res_default.equity.to_numpy(), res_scaled.equity.to_numpy())
    ok = same_n and same_eq
    print(f"  regression check (spot, leverage=1): default vs equity-scaled broker "
          f"identical -- fills {len(res_default.fills)}=={len(res_scaled.fills)} "
          f"({same_n}), equity curve identical ({same_eq}) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def realized_exposure_path(res, market: MarketSpec) -> pd.Series:
    """Reconstruct the realized position, as a signed fraction of equity
    (the same units KellyRegimeV4's own `target` column uses, since it
    routes through `order_notional`), at every bar of the run.

    Fills carry the position *after* execution at that bar's timestamp;
    forward-filling onto the equity index gives the position held through
    each bar. Multiple same-timestamp fills (a sign flip: close then
    open) are collapsed to the last (the post-flip position).
    """
    idx = res.equity.index
    if not res.fills:
        return pd.Series(0.0, index=idx)
    pos = 0.0
    ts_list, pos_list = [], []
    for f in res.fills:
        pos += f.qty if f.side is Side.BUY else -f.qty
        ts_list.append(f.ts)
        pos_list.append(pos)
    s = pd.Series(pos_list, index=pd.DatetimeIndex(ts_list))
    s = s[~s.index.duplicated(keep="last")]
    combined = s.reindex(idx.union(s.index)).sort_index().ffill().fillna(0.0)
    combined = combined.reindex(idx)
    closes = res.df.loc[idx, "close"]
    equity = res.equity.replace(0.0, np.nan)
    return (combined * closes / equity).fillna(0.0)


def cell(label: str, res, m: compute_metrics, broker, market: MarketSpec, split: str) -> dict:
    log = broker.intended_log
    intended = len(log)
    filled = sum(1 for _, _, ok in log if ok)
    exposure = realized_exposure_path(res, market)
    if "target" in res.df:
        tgt_series = res.df.loc[res.equity.index, "target"]
        tracking_err = float(np.mean(np.abs(exposure.to_numpy() - tgt_series.to_numpy())))
    else:
        tracking_err = float("nan")
    return dict(
        label=label, split=split, market=market.name,
        intended=intended, fills=filled,
        fill_rate=(filled / intended if intended else float("nan")),
        n_trades=m.num_trades, fees_paid=m.fees_paid,
        final_balance=m.final_balance, sharpe=m.sharpe,
        max_drawdown_pct=m.max_drawdown_pct,
        mean_abs_exposure=float(np.mean(np.abs(exposure))),
        mean_tracking_error=tracking_err,
    )


def run_deadband_arm(broker_cls, split_window, market: MarketSpec, df: pd.DataFrame,
                     label: str, split: str) -> dict:
    _CONFIGS[0] += 1
    _LAST_BROKER.clear()
    start, end = split_window
    with _patched_broker(broker_cls):
        res = run_period(KellyRegimeV4(), df, start, end, market=market)
    assert len(_LAST_BROKER) == 1
    m = compute_metrics(res)
    return cell(label, res, m, _LAST_BROKER[0], market, split), res


def cmd_deadband(df: pd.DataFrame) -> list[dict]:
    print()
    print("=" * 110)
    print("PART 2: kelly_regime_v4, equity-scaled deadband vs default, FUTURES 5x "
          "(isolated broker variant, strategy UNMODIFIED)")
    print("=" * 110)

    ok = regression_check_spot_identical()

    fut = futures()
    rows: list[dict] = []
    results: dict[tuple[str, str], object] = {}
    for split, window in SPLITS:
        default_row, default_res = run_deadband_arm(
            InstrumentedBroker, window, fut, df, "default (5% of max notional)", split)
        scaled_row, scaled_res = run_deadband_arm(
            EquityScaledInstrumentedBroker, window, fut, df,
            "equity-scaled (5% of equity)", split)
        rows.append(default_row)
        rows.append(scaled_row)
        results[(split, "default")] = default_res
        results[(split, "scaled")] = scaled_res

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "r72_equity_scaled_deadband_v4.csv", index=False)

    print()
    hdr = (f"{'label':<32} {'split':<12} {'intended':>9} {'fills':>7} {'fill%':>7} "
           f"{'trades':>7} {'fees $':>10} {'final $':>11} {'Sharpe':>7} {'DD%':>6} "
           f"{'trkErr':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<32} {r['split']:<12} {r['intended']:>9} {r['fills']:>7} "
              f"{r['fill_rate']:>6.1%} {r['n_trades']:>7} {r['fees_paid']:>10,.2f} "
              f"{r['final_balance']:>11,.2f} {r['sharpe']:>7.3f} "
              f"{r['max_drawdown_pct']:>6.1f} {r['mean_tracking_error']:>7.4f}")

    # Difference test (paired bootstrap on daily log growth), matching the
    # project-standard block bootstrap. This is a diagnostic, not a
    # promotion decision -- this round does not promote anything.
    print()
    print("Difference test (equity-scaled minus default), paired stationary "
          "block bootstrap on daily log returns, mean_block=30, n_boot=2000, seed=7:")
    for split, _ in SPLITS:
        d_res = results[(split, "default")]
        s_res = results[(split, "scaled")]
        a = daily_returns(s_res.equity).to_numpy(dtype=float)
        b = daily_returns(d_res.equity).to_numpy(dtype=float)
        n = min(len(a), len(b))
        growth = paired_bootstrap(a[:n], b[:n], total_log_return, **BOOT_KW)
        dd = paired_bootstrap(a[:n], b[:n], max_drawdown_from_returns, **BOOT_KW)
        print(f"  {split:<12} d_logret={growth.diff.point:+.4f} "
              f"[{growth.diff.lo:+.4f}, {growth.diff.hi:+.4f}]   "
              f"d_maxdd_pp={dd.diff.point:+.2f} "
              f"[{dd.diff.lo:+.2f}, {dd.diff.hi:+.2f}]")

    print()
    print(f"Regression check (spot, leverage=1, identical decisions): "
          f"{'PASS' if ok else 'FAIL'}")
    return rows


# ====================================================================== main


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "all"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("R-72 CONSERVATIVE: B-30 made explicit and general -- fill-through "
          "measurement across every registered strategy, and an isolated "
          "equity-scaled-deadband test on kelly_regime_v4")
    print("=" * 110)

    df = load_btc_inner()
    print(f"BTC (inner, truncated at 2022-12-31 on load): {len(df):,} bars, "
          f"{df.index[0]} -> {df.index[-1]}")
    print(f"assert_no_holdout: max timestamp read = {df.index.max()} "
          f"(< {OOS_START}, verified)")

    if cmd in ("all", "fillrate"):
        out = cmd_fillrate(df)
        cmd_fillrate_summary(out)

    if cmd in ("all", "deadband"):
        cmd_deadband(df)

    print()
    print("=" * 110)
    print(f"Configurations evaluated (this file, both parts): {configs_evaluated()}")
    print("Holdout consultations added: 0 (BTC truncated at 2022-12-31 on load; "
          f"assert_no_holdout checked, max ts {df.index.max()})")
    print("=" * 110)


if __name__ == "__main__":
    main()
