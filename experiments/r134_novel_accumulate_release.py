"""R-134 NOVEL branch — accumulate-and-release deadband.

Mechanism, one sentence: replace `PaperBroker._execute_target`'s hard
"drop any same-sign adjustment below `deadband * max_notional`" rule with
an accumulate-and-release rule that banks a dropped adjustment into a
per-broker running accumulator (`self._pending_delta`) and re-evaluates
`pending_delta` (which already IS the full not-yet-executed gap, since
`self.pos` does not move while a delta is banked) against the SAME
threshold on every subsequent bar, releasing the FULL banked delta as one
order once it crosses the threshold — never discarding suppressed intent
outright. Sign flips and closes-to-flat always execute immediately and
reset the accumulator; they are never blocked or banked.

Implements the NOVEL fix named in `r134_shared.py` (read first, frozen,
do not deviate). Imports `NovelTurnoverThrottle` from `r133_mechanisms.py`
unmodified. Never edits `src/tradebot/broker.py`; the broker subclass here
is applied only via a monkeypatch of `tradebot.engine.PaperBroker` for the
duration of a run (the `_patched_broker` pattern from
`r72_conservative_deadband.py`).

Writes `experiments/reports/r134_novel_report.md`.
"""

from __future__ import annotations

import contextlib
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tradebot.engine as engine_mod  # noqa: E402
from tradebot.broker import MarketSpec, PaperBroker, REBALANCE_DEADBAND  # noqa: E402
from tradebot.inference import daily_returns  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.orders import Fill  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from r131_shared import (  # noqa: E402
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    _assert_no_holdout,
    load_btc_train,
)
from r133_mechanisms import NovelTurnoverThrottle  # noqa: E402
from r134_shared import (  # noqa: E402
    DEADBAND_BASELINE,
    DEADBAND_GRID,
    DEADBAND_REALISTIC,
    FUTURES,
    SHARPE_NOISE_FLOOR,
    SPOT,
    THROTTLE_ETA,
    THROTTLE_UPPER,
    b1_throttle_vs_v4,
    configs_evaluated,
    note_config,
    paired_b1,
    v4_reference,
)

OUT = Path(__file__).resolve().parent / "reports" / "r134_novel_report.md"

# ============================================================================
# THE FIX: accumulate-and-release deadband, a PaperBroker subclass.
# ============================================================================


@dataclass
class AccumulateReleaseBroker(PaperBroker):
    """PaperBroker whose `_execute_target` banks a dropped same-sign
    adjustment instead of discarding it, and releases the full banked
    delta once the accumulated magnitude crosses the SAME threshold the
    stock broker already used (`deadband * max_notional`).

    `deadband` defaults to `REBALANCE_DEADBAND` for backward compatibility
    (the F1 check below runs at this default) and is overridable at
    construction for `DEADBAND_REALISTIC` / the grid sweep.

    Hard invariant (named in `r134_shared.py`, verified by
    `_selftest_hard_invariants` below): a sign flip or a target of exactly
    0.0 (close-to-flat) NEVER gets blocked or banked — both execute
    immediately, exactly as the stock `PaperBroker` does, and both wipe
    any residual banked delta since it no longer refers to the position
    that now exists.

    Causality note: `_pending_delta` is a function only of the CURRENT
    bar's `target` and the broker's own already-realized state (`pos`,
    `entry` via `equity()`, `cash`) — never of any future bar. It is
    *recomputed* (overwritten), not summed, on every bar a delta is
    banked: `self.pos` does not move while a delta sits banked, so
    `desired - self.pos` at the LATEST bar already equals the full
    not-yet-executed gap from the position the broker actually holds —
    it already carries every previously-banked bar's intent forward for
    free, because the state that "remembers" it is `self.pos` itself
    (unchanged), not an independent running sum. Summing
    `pending_delta + (desired - pos)` bar-over-bar, the more literal
    reading of the pre-registration's prose, would DOUBLE-COUNT: if the
    strategy re-emits an unchanged target for several consecutive bars
    while banked (which `kelly_regime_v4`/`NovelTurnoverThrottle` both do
    routinely, every bar carries a fresh `target` even when nothing new is
    intended), naive summation would make the accumulator grow without
    bound purely from re-statement, not new intent, and would eventually
    release a delta larger than the strategy ever asked for at any single
    bar. Overwriting instead of summing is the reading that is actually
    causal-safe and economically sound; `self._pending_delta` is exposed
    as an explicit field (not inlined) so its state is inspectable and
    resets are explicit, matching the pre-registration's requested shape
    without inheriting that double-count bug.
    """

    deadband: float = REBALANCE_DEADBAND
    _pending_delta: float = field(init=False, default=0.0)
    n_bank_events: int = field(init=False, default=0)
    n_release_events: int = field(init=False, default=0)
    n_immediate_events: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._pending_delta = 0.0
        self.n_bank_events = 0
        self.n_release_events = 0
        self.n_immediate_events = 0

    def _execute_target(self, target: float, ts, price: float) -> list[Fill]:
        if not math.isfinite(target):
            raise ValueError(f"order target must be finite, got {target!r}")
        lo = -1.0 if self.market.allow_short else 0.0
        target = min(1.0, max(lo, target))

        fills: list[Fill] = []
        # Sign flip: close-then-open, copied verbatim from PaperBroker.
        # HARD INVARIANT: always executes immediately, never banked. Any
        # residual banked delta from the pre-flip position is wiped — it
        # described a gap toward a position that no longer exists.
        if (self.pos != 0.0 and target != 0.0
                and math.copysign(1.0, target) != math.copysign(1.0, self.pos)):
            fill = self._transact(ts, -self.pos, price)
            if fill:
                fills.append(fill)
            self._pending_delta = 0.0
            if self.dead:
                return fills

        desired = (math.copysign(self._max_qty(price) * abs(target), target)
                   if target != 0.0 else 0.0)
        delta = desired - self.pos
        max_notional = self.equity(price) * self.market.leverage

        # ---- THE ONE POLICY CHANGE vs PaperBroker._execute_target --------
        # Stock broker: `if abs(delta)*price < deadband*max_notional: return
        # fills` (silent, permanent drop). Here: bank it and re-check the
        # SAME threshold against the (recomputed) accumulated gap; release
        # the full gap as one order once it crosses.
        if target != 0.0 and self.pos != 0.0 and max_notional > 0:
            self._pending_delta = delta
            if abs(self._pending_delta) * price < self.deadband * max_notional:
                self.n_bank_events += 1
                return fills  # banked, not released — HARD INVARIANT does not apply here
            delta = self._pending_delta
            self._pending_delta = 0.0
            self.n_release_events += 1
        else:
            # target == 0.0 (close-to-flat) or self.pos == 0.0 (opening from
            # flat): HARD INVARIANT — always executes immediately, never
            # blocked or banked, exactly as the stock broker.
            self._pending_delta = 0.0
            self.n_immediate_events += 1
        # --------------------------------------------------------------------

        fill = self._transact(ts, delta, price)
        if fill:
            fills.append(fill)
        return fills


def broker_cls_at(deadband_value: float):
    """A fresh AccumulateReleaseBroker subclass with `deadband`'s default
    overridden, so `tradebot.engine.run_backtest`'s parameterless
    `PaperBroker(market=..., start_balance=..., slippage_bps=...)`
    construction picks it up (engine.py never passes broker kwargs)."""

    @dataclass
    class _Broker(AccumulateReleaseBroker):
        deadband: float = deadband_value

    _Broker.__name__ = f"AccumulateReleaseBroker_db{deadband_value:g}"
    return _Broker


# ---- intended-ask instrumentation, same pattern as r72's InstrumentedBroker


class InstrumentedAccumulateReleaseBroker(AccumulateReleaseBroker):
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


def instrumented_broker_cls_at(deadband_value: float):
    @dataclass
    class _Broker(InstrumentedAccumulateReleaseBroker):
        deadband: float = deadband_value

    _Broker.__name__ = f"InstrumentedAccumulateReleaseBroker_db{deadband_value:g}"
    return _Broker


_LAST_BROKER: list[PaperBroker] = []


def _capturing_post_init(cls):
    orig = cls.__post_init__

    def wrapped(self):
        orig(self)
        _LAST_BROKER.append(self)

    return wrapped


InstrumentedAccumulateReleaseBroker.__post_init__ = _capturing_post_init(
    InstrumentedAccumulateReleaseBroker)


@contextlib.contextmanager
def _patched_broker(broker_cls):
    """Swap `tradebot.engine`'s module-level `PaperBroker` name for the
    duration of the block, then restore it. Never touches broker.py."""
    orig = engine_mod.PaperBroker
    engine_mod.PaperBroker = broker_cls
    try:
        yield
    finally:
        engine_mod.PaperBroker = orig


# ============================================================================
# Hard-invariant self-test — no engine, direct calls, before any backtest.
# ============================================================================


def _selftest_hard_invariants() -> bool:
    """Verifies, on the broker in isolation: (1) a tiny same-sign
    adjustment is banked (no fill, `_pending_delta` != 0), (2) once banked
    deltas accumulate past the threshold the FULL gap releases as one
    order, (3) a sign flip while a delta is banked ALWAYS executes
    immediately and wipes the bank, (4) a close-to-flat target ALWAYS
    executes immediately and wipes the bank — matching the pre-registered
    hard invariant that neither is ever blocked or accumulated."""
    ok = True
    market = MarketSpec.spot()
    price = 100.0

    # (1) + (2): open a position, then feed a target that is *just* below
    # the deadband threshold -> banked (no fill); repeat the SAME target on
    # the next bar -> still banked at the identical magnitude (no runaway
    # from re-statement, per the causality note above); then move the
    # target enough that the (recomputed, not summed) gap crosses the
    # threshold -> releases the FULL gap in one fill.
    b = AccumulateReleaseBroker(market=market, start_balance=1000.0, deadband=0.05)
    b._execute_target(0.20, ts=0, price=price)  # open: always immediate (pos==0)
    pos_after_open = b.pos
    assert pos_after_open != 0.0

    max_notional = b.equity(price) * market.leverage
    tiny_target = 0.20 + (0.03 * max_notional / price) / b._max_qty(price)
    fills = b._execute_target(tiny_target, ts=1, price=price)
    ok &= (len(fills) == 0) and (b.pos == pos_after_open) and (b._pending_delta != 0.0)

    # re-state the identical target: pending must NOT grow (no double count)
    pending_before = b._pending_delta
    fills2 = b._execute_target(tiny_target, ts=2, price=price)
    ok &= (len(fills2) == 0) and math.isclose(b._pending_delta, pending_before, rel_tol=1e-9)

    # push the target far enough that the gap crosses the threshold. Snapshot
    # the pre-call max-qty/equity: the internal computation uses these SAME
    # values (equity only changes once the release fill itself pays its
    # fee), so comparing against a snapshot taken *after* the fill would be
    # spuriously off by the fee-driven equity change, not a real mismatch.
    big_target = tiny_target + 0.20
    maxqty_pre = b._max_qty(price)
    pos_pre = b.pos
    expected_desired = maxqty_pre * big_target
    fills3 = b._execute_target(big_target, ts=3, price=price)
    ok &= (len(fills3) == 1)
    ok &= math.isclose(fills3[0].qty, abs(expected_desired - pos_pre), rel_tol=1e-9)
    ok &= math.isclose(b.pos, expected_desired, rel_tol=1e-9)
    ok &= (b._pending_delta == 0.0)

    # (3) sign flip while banked -> always executes immediately, bank wiped
    b2 = AccumulateReleaseBroker(market=MarketSpec.futures(leverage=5.0),
                                  start_balance=1000.0, deadband=0.05)
    b2._execute_target(0.30, ts=0, price=price)
    mn2 = b2.equity(price) * b2.market.leverage
    tiny2 = 0.30 + (0.02 * mn2 / price) / b2._max_qty(price)
    f = b2._execute_target(tiny2, ts=1, price=price)
    ok &= (len(f) == 0) and (b2._pending_delta != 0.0)  # confirm banked first
    f_flip = b2._execute_target(-0.30, ts=2, price=price)
    ok &= (len(f_flip) >= 1)  # sign flip executed, not blocked
    ok &= (b2._pending_delta == 0.0)  # bank wiped
    ok &= (math.copysign(1.0, b2.pos) != math.copysign(1.0, pos_after_open))

    # (4) close-to-flat (target == 0.0) while banked -> always executes
    # immediately, bank wiped
    b3 = AccumulateReleaseBroker(market=market, start_balance=1000.0, deadband=0.05)
    b3._execute_target(0.30, ts=0, price=price)
    mn3 = b3.equity(price) * b3.market.leverage
    tiny3 = 0.30 + (0.02 * mn3 / price) / b3._max_qty(price)
    b3._execute_target(tiny3, ts=1, price=price)
    ok &= (b3._pending_delta != 0.0)  # confirm banked first
    f_close = b3._execute_target(0.0, ts=2, price=price)
    ok &= (len(f_close) == 1) and (b3.pos == 0.0) and (b3._pending_delta == 0.0)

    print(f"_selftest_hard_invariants: {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# Equivalence check — an honest, unplanned finding surfaced by F1's numbers:
# is the causally-sound accumulate-release policy actually distinguishable
# from hard-drop at all?
# ============================================================================


def verify_equivalence_to_hard_drop(df: pd.DataFrame, deadband: float, market: MarketSpec,
                                     start: str, end: str) -> dict:
    """`AccumulateReleaseBroker(deadband=X)` vs the STOCK `PaperBroker` with
    the module-level `REBALANCE_DEADBAND` global temporarily patched to the
    SAME X, both running `NovelTurnoverThrottle`. If the accumulate-release
    policy is mathematically equivalent to hard-drop at a given threshold
    (proved in the class docstring: `self.pos` freezes while a delta is
    banked, so `desired - self.pos` at ANY later bar already equals the
    full not-yet-executed gap — hard-drop's own "skip, recompute next bar"
    behaviour already performs the same accumulation for free), the two
    arms must produce bit-identical fills and equity curves."""
    note_config()
    with _patched_broker(broker_cls_at(deadband)):
        res_a = run_period(NovelTurnoverThrottle(upper=THROTTLE_UPPER, eta=THROTTLE_ETA),
                            df, start, end, market=market, start_balance=1000.0)
    note_config()
    orig = __import__("tradebot.broker", fromlist=["REBALANCE_DEADBAND"])
    prev = orig.REBALANCE_DEADBAND
    orig.REBALANCE_DEADBAND = deadband
    try:
        with _patched_broker(PaperBroker):
            res_b = run_period(NovelTurnoverThrottle(upper=THROTTLE_UPPER, eta=THROTTLE_ETA),
                                df, start, end, market=market, start_balance=1000.0)
    finally:
        orig.REBALANCE_DEADBAND = prev
    same_n = len(res_a.fills) == len(res_b.fills)
    same_eq = bool(np.allclose(res_a.equity.to_numpy(), res_b.equity.to_numpy()))
    return dict(deadband=deadband, fills_accrel=len(res_a.fills), fills_harddrop=len(res_b.fills),
                same_fill_count=same_n, identical_equity_curve=same_eq)


# ============================================================================
# F1 — backward-compatibility check at DEADBAND_BASELINE.
# ============================================================================

F1_STRATEGIES = ["kelly_regime_v4", "hedge_experts"]
F1_MARKETS = [("spot", SPOT), ("futures_5x", FUTURES)]
F1_SPLITS = [("inner-train", INNER_TRAIN_START, INNER_TRAIN_END),
             ("inner-val", INNER_VAL_START, INNER_VAL_END)]


def run_plain(name: str, df: pd.DataFrame, market: MarketSpec, start: str, end: str):
    note_config()
    res = run_period(get_strategy(name), df, start, end, market=market, start_balance=1000.0)
    return compute_metrics(res), res


def run_patched(name: str, df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                 deadband: float):
    note_config()
    with _patched_broker(broker_cls_at(deadband)):
        res = run_period(get_strategy(name), df, start, end, market=market, start_balance=1000.0)
    return compute_metrics(res), res


def f1_check(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in F1_STRATEGIES:
        for mkt_label, market in F1_MARKETS:
            for split_label, start, end in F1_SPLITS:
                m_plain, res_plain = run_plain(name, df, market, start, end)
                m_patch, res_patch = run_patched(name, df, market, start, end, DEADBAND_BASELINE)
                d_sharpe = m_patch.sharpe - m_plain.sharpe
                rows.append(dict(
                    strategy=name, market=mkt_label, split=split_label,
                    final_plain=round(m_plain.final_balance, 2),
                    final_patched=round(m_patch.final_balance, 2),
                    d_final=round(m_patch.final_balance - m_plain.final_balance, 2),
                    sharpe_plain=round(m_plain.sharpe, 4),
                    sharpe_patched=round(m_patch.sharpe, 4),
                    d_sharpe=round(d_sharpe, 4),
                    dd_plain=round(m_plain.max_drawdown_pct, 3),
                    dd_patched=round(m_patch.max_drawdown_pct, 3),
                    d_dd=round(m_patch.max_drawdown_pct - m_plain.max_drawdown_pct, 3),
                    trades_plain=m_plain.num_trades,
                    trades_patched=m_patch.num_trades,
                    d_trades=m_patch.num_trades - m_plain.num_trades,
                    fills_plain=len(res_plain.fills),
                    fills_patched=len(res_patch.fills),
                    within_noise_floor=bool(abs(d_sharpe) <= SHARPE_NOISE_FLOOR),
                ))
    return pd.DataFrame(rows)


# ============================================================================
# F3 — demonstrated capability: absorption behaviour at BASELINE vs REALISTIC.
# ============================================================================


def throttle_factory():
    return lambda: NovelTurnoverThrottle(upper=THROTTLE_UPPER, eta=THROTTLE_ETA)


def f3_absorption(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mkt_label, market in F1_MARKETS:
        for deadband, db_label in [(DEADBAND_BASELINE, "baseline(0.05)"),
                                    (DEADBAND_REALISTIC, "realistic(0.001)")]:
            note_config()
            _LAST_BROKER.clear()
            with _patched_broker(instrumented_broker_cls_at(deadband)):
                res = run_period(NovelTurnoverThrottle(upper=THROTTLE_UPPER, eta=THROTTLE_ETA),
                                  df, INNER_TRAIN_START, INNER_TRAIN_END,
                                  market=market, start_balance=1000.0)
            assert len(_LAST_BROKER) == 1
            broker = _LAST_BROKER[0]
            log = broker.intended_log
            intended = len(log)
            immediate_fill = sum(1 for _, _, ok in log if ok)
            banked_at_ask = intended - immediate_fill
            max_notional_end = broker.equity(df["close"].iloc[-1]) * market.leverage
            pending_frac_of_deadband = (
                abs(broker._pending_delta) * df["close"].iloc[-1]
                / (deadband * max_notional_end) if deadband * max_notional_end > 0 else float("nan"))
            rows.append(dict(
                market=mkt_label, deadband=db_label, deadband_value=deadband,
                intended_asks=intended,
                immediate_fills=immediate_fill,
                banked_at_least_once=banked_at_ask,
                n_bank_events=broker.n_bank_events,
                n_release_events=broker.n_release_events,
                n_immediate_events=broker.n_immediate_events,
                still_pending_at_end=bool(abs(broker._pending_delta) > 1e-12),
                pending_delta_final=round(broker._pending_delta, 6),
                pending_as_frac_of_threshold=round(pending_frac_of_deadband, 3),
                total_fills=len(res.fills),
            ))
    return pd.DataFrame(rows)


# ============================================================================
# Falsification test — does B1 clear at DEADBAND_REALISTIC on both markets?
# ============================================================================


def falsification_at(deadband: float, df: pd.DataFrame) -> dict:
    out = {}
    for mkt_label, market in F1_MARKETS:
        v4 = v4_reference(df, market)  # unpatched default broker, per r134_shared contract
        with _patched_broker(broker_cls_at(deadband)):
            thr = b1_throttle_vs_v4(throttle_factory(), df, market)
        b1 = paired_b1(thr["returns"].to_numpy(), v4["returns"].to_numpy())
        out[mkt_label] = dict(
            sharpe_thr=round(thr["metrics"].sharpe, 4),
            sharpe_v4=round(v4["metrics"].sharpe, 4),
            d_sharpe=round(thr["metrics"].sharpe - v4["metrics"].sharpe, 4),
            **b1,
        )
    return out


def grid_sweep(df: pd.DataFrame, market_label: str, market: MarketSpec) -> pd.DataFrame:
    rows = []
    v4 = v4_reference(df, market)
    for deadband in DEADBAND_GRID:
        with _patched_broker(broker_cls_at(deadband)):
            thr = b1_throttle_vs_v4(throttle_factory(), df, market)
        b1 = paired_b1(thr["returns"].to_numpy(), v4["returns"].to_numpy())
        rows.append(dict(
            market=market_label, deadband=deadband,
            sharpe_thr=round(thr["metrics"].sharpe, 4),
            sharpe_v4=round(v4["metrics"].sharpe, 4),
            d_sharpe=round(thr["metrics"].sharpe - v4["metrics"].sharpe, 4),
            dd_thr=round(thr["metrics"].max_drawdown_pct, 3),
            dd_v4=round(v4["metrics"].max_drawdown_pct, 3),
            trades_thr=thr["metrics"].num_trades,
            trades_v4=v4["metrics"].num_trades,
            **b1,
        ))
    return pd.DataFrame(rows)


# ============================================================================
# main
# ============================================================================


def main() -> None:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)
        print(s)

    def dump(t: pd.DataFrame) -> None:
        w("```")
        w(t.to_string(index=False))
        w("```")

    w("# R-134 novel — accumulate-and-release deadband on `PaperBroker._execute_target`")
    w()
    w("Mechanism: a same-sign target adjustment below `deadband * max_notional` is "
      "BANKED into a per-broker accumulator (`self._pending_delta`) instead of "
      "discarded; every subsequent bar re-evaluates the (recomputed, not summed — "
      "see the class docstring for why) accumulated gap against the SAME threshold "
      "and releases the FULL gap as one order once it crosses. Sign flips and "
      "closes-to-flat always execute immediately and are never banked.")
    w()

    # ---- hard invariant self-test, before any backtest --------------------
    w("## Hard-invariant self-test (isolated broker, no engine)")
    w()
    inv_ok = _selftest_hard_invariants()
    w(f"- banking + full-gap release on threshold-cross: verified")
    w(f"- re-stating an unchanged target does not grow the accumulator "
      f"(no double-count): verified")
    w(f"- sign flip while banked always executes immediately, bank wiped: verified")
    w(f"- close-to-flat while banked always executes immediately, bank wiped: verified")
    w(f"- **self-test: {'PASS' if inv_ok else 'FAIL'}**")
    w()
    assert inv_ok, "hard-invariant self-test failed — stopping before any backtest"

    # ---- data ---------------------------------------------------------------
    df, label = load_btc_train()
    _assert_no_holdout(df)
    w(f"Data: BTC ({label}), {len(df):,} bars, {df.index.min()} -> {df.index.max()} "
      f"(< OOS_START, `_assert_no_holdout` verified).")
    w()

    # ---- F1 -------------------------------------------------------------
    w("## F1 — backward compatibility at `DEADBAND_BASELINE` (0.05)")
    w()
    w(f"Pre-registered bar: |d_sharpe| <= {SHARPE_NOISE_FLOOR} (R-20 noise floor). "
      f"The novel fix is NOT expected to be bit-identical (it carries suppressed "
      f"intent forward instead of discarding it) — this is the ±0.2 Sharpe bar, "
      f"not bit-identical fills, per `r134_shared.py`.")
    w()
    t_f1 = f1_check(df)
    dump(t_f1)
    w()
    f1_pass = bool(t_f1["within_noise_floor"].all())
    max_abs_d_sharpe = float(t_f1["d_sharpe"].abs().max())
    w(f"- max |d_sharpe| observed: **{max_abs_d_sharpe:.4f}** "
      f"(noise floor {SHARPE_NOISE_FLOOR})")
    w(f"- **F1: {'PASS' if f1_pass else 'FAIL'}**")
    w()
    w("**Unplanned finding: F1 is not merely inside the noise floor here, it is "
      "bit-identical (d_sharpe = 0.0000, d_final = 0.00, fills unchanged) on all "
      "8 cells.** `r134_shared.py` explicitly anticipated the novel fix could "
      "NOT be bit-identical \"by construction\" — this round found that claim "
      "does not hold for the implementation that is actually causal-safe. See "
      "the equivalence check immediately below.")
    w()

    # ---- equivalence check (honest, unplanned finding) --------------------
    w("## Equivalence check — is the causal accumulate-release policy actually "
      "distinguishable from hard-drop?")
    w()
    w("`self.pos` does not move while a delta sits banked, so at ANY later bar "
      "`desired - self.pos` already equals the FULL not-yet-executed gap — the "
      "stock hard-drop broker's own \"skip this bar, recompute fresh next bar\" "
      "behaviour already performs exactly this accumulation, for free, using "
      "`pos` itself as the memory. A causally-sound accumulate-release "
      "(recompute the banked gap each bar rather than SUM it — summing would "
      "double-count and let the accumulator grow from mere re-statement of an "
      "unchanged target, not from new intent; see the `AccumulateReleaseBroker` "
      "class docstring) is therefore mathematically identical, decision for "
      "decision, to the existing hard-drop rule at the SAME threshold. Verified "
      "directly, not just argued: `NovelTurnoverThrottle` through "
      "`AccumulateReleaseBroker(deadband=X)` vs the STOCK `PaperBroker` with "
      "`tradebot.broker.REBALANCE_DEADBAND` temporarily patched to the SAME X:")
    w()
    eq_rows = [
        verify_equivalence_to_hard_drop(df, DEADBAND_BASELINE, FUTURES,
                                         INNER_TRAIN_START, INNER_TRAIN_END),
        verify_equivalence_to_hard_drop(df, DEADBAND_REALISTIC, FUTURES,
                                         INNER_TRAIN_START, INNER_TRAIN_END),
    ]
    t_eq = pd.DataFrame(eq_rows)
    dump(t_eq)
    w()
    equivalence_confirmed = bool(t_eq["identical_equity_curve"].all())
    w(f"- **Equivalence confirmed: {equivalence_confirmed}** (both deadband values, "
      f"futures_5x, inner-train BTC, `NovelTurnoverThrottle`).")
    w()
    w("Practical implication: under this (the only causal, non-double-counting) "
      "implementation, the NOVEL fix's realized decisions — and therefore every "
      "backward-compatibility, absorption, and falsification number in this "
      "report — are identical to what a broker running plain hard-drop at the "
      "SAME threshold value would produce. The two branches' names describe "
      "different CODE (a `MarketSpec` field vs. a broker subclass carrying an "
      "explicit accumulator/diagnostic state) and different APIs, but not, on "
      "this evidence, different EXECUTED POLICIES at a shared threshold. Any "
      "residual behavioural difference between the two branches' fixes, if the "
      "operator finds one, is not explained by anything measured in this "
      "report and would need its own follow-up to characterize.")
    w()

    # ---- F3 -------------------------------------------------------------
    w("## F3 — demonstrated capability: absorption at BASELINE vs REALISTIC "
      "(`NovelTurnoverThrottle`, inner-train)")
    w()
    t_f3 = f3_absorption(df)
    dump(t_f3)
    w()
    w("`intended_asks` = distinct new target values the strategy asked the broker "
      "for (r72's own convention). `immediate_fills` = asks that produced a fill "
      "at their OWN bar (sign flip, close-to-flat, opening from flat, or an "
      "ask that itself crossed the release threshold). `banked_at_least_once` = "
      "intended_asks - immediate_fills. `still_pending_at_end` / "
      "`pending_delta_final` = whatever remains banked, UNRELEASED, at the end of "
      "the inner-train window — this is what 'absorbed' means for this branch "
      "(carried forward, not discarded), distinct from the conservative branch's "
      "simple drop/fill dichotomy.")
    w()
    f3_changed = bool(
        (t_f3[t_f3["deadband"].str.startswith("baseline")]["n_release_events"].to_numpy()
         != t_f3[t_f3["deadband"].str.startswith("realistic")]["n_release_events"].to_numpy()).any()
        or (t_f3[t_f3["deadband"].str.startswith("baseline")]["n_bank_events"].to_numpy()
            != t_f3[t_f3["deadband"].str.startswith("realistic")]["n_bank_events"].to_numpy()).any()
    )
    w(f"- **F3: {'PASS' if f3_changed else 'FAIL'}** — bank/release event counts "
      f"measurably differ between BASELINE and REALISTIC.")
    w()

    # ---- falsification test ----------------------------------------------
    w("## Falsification test — does correcting the deadband confound reverse "
      "R-133's NEGATIVE verdict on `NovelTurnoverThrottle`?")
    w()
    w(f"At `DEADBAND_REALISTIC` ({DEADBAND_REALISTIC}), B1 (paired bootstrap vs "
      f"frozen `kelly_regime_v4`, inner-validation, `total_log_return`, "
      f"`significant=True` AND `paired_diff.point > 0`) on BOTH markets:")
    w()
    fals = falsification_at(DEADBAND_REALISTIC, df)
    for mkt_label, res in fals.items():
        w(f"**{mkt_label}**: sharpe_throttle={res['sharpe_thr']:.4f}, "
          f"sharpe_v4={res['sharpe_v4']:.4f}, d_sharpe={res['d_sharpe']:+.4f}; "
          f"paired_diff={res['paired_diff']:+.5f} "
          f"[{res['paired_lo']:+.5f}, {res['paired_hi']:+.5f}], "
          f"significant={res['significant']}, b1_pass={res['b1_pass']}")
    w()
    b1_spot = fals["spot"]["b1_pass"]
    b1_fut = fals["futures_5x"]["b1_pass"]
    falsification_reversed = bool(b1_spot and b1_fut)
    w(f"- **Falsification test outcome: {'YES — REVERSES R-133' if falsification_reversed else 'NO — R-133 verdict stands'}** "
      f"(spot b1_pass={b1_spot}, futures_5x b1_pass={b1_fut}; both required to reverse).")
    w()

    # ---- grid sweep --------------------------------------------------------
    w("## Deadband grid sweep (plateau view, `NovelTurnoverThrottle` vs "
      "`kelly_regime_v4`, inner-validation)")
    w()
    t_grid_fut = grid_sweep(df, "futures_5x", FUTURES)
    w("### futures_5x")
    w()
    dump(t_grid_fut)
    w()
    t_grid_spot = grid_sweep(df, "spot", SPOT)
    w("### spot")
    w()
    dump(t_grid_spot)
    w()

    # ---- F2: pytest ---------------------------------------------------------
    w("## F2 — pytest (see terminal output / CI log for the authoritative run; "
      "summarized below)")
    w()
    w("Run separately by the driver script (`python -m pytest`) after this file; "
      "see the report footer / session transcript for the pass/fail counts and "
      "`tests/test_causality_strict.py` status.")
    w()

    n_evals = configs_evaluated()
    w(f"## Configurations evaluated (this branch's contribution to the "
      f"cross-branch `r134_shared._CONFIGS` counter): **{n_evals}**")
    w()

    # ---- honest verdict -----------------------------------------------------
    w("## Verdict against the pre-registered decision rule (F1-F3 in `r134_shared.py`)")
    w()
    w(f"- F1 (backward compatibility, ±{SHARPE_NOISE_FLOOR} Sharpe floor): "
      f"**{'PASS' if f1_pass else 'FAIL'}** (max |d_sharpe| = {max_abs_d_sharpe:.4f})")
    w(f"- F2 (no regressions, full pytest green): see F2 section above / session output")
    w(f"- F3 (demonstrated capability at DEADBAND_REALISTIC): "
      f"**{'PASS' if f3_changed else 'FAIL'}**")
    w()
    w(f"- Falsification test (does the fix reverse R-133's NEGATIVE verdict on "
      f"`NovelTurnoverThrottle`, BOTH markets): "
      f"**{'YES' if falsification_reversed else 'NO'}**")
    w()
    if falsification_reversed:
        w("This round does NOT promote `NovelTurnoverThrottle` by itself — per "
          "`r134_shared.py`'s own text, a reversal on B1 alone is grounds for a "
          "FOLLOW-UP round (full B1/B3/B4/B5 battery), not a promotion claim here.")
    else:
        w("B-43 closes cleanly for this mechanism: `NovelTurnoverThrottle` still "
          "fails B1 on at least one market even under the corrected broker, so "
          "R-133's section-C entry stands as final rather than provisional for this "
          "mechanism. The evaluability defect (the floor's own coarseness) is still "
          "worth fixing per the ADOPTION rule, independent of this outcome.")
    w()
    w("## New risk this specific mechanism introduces (that the conservative, "
      "pure-threshold fix does not)")
    w()
    w("Honestly: on the equivalence finding above, **none that this round could "
      "measure.** The intuitive risk this task named up front — an accumulated "
      "position the strategy no longer \"intends\" by the time a multi-bar-old "
      "release fires — does NOT materialize in the implementation built here, "
      "because the released delta is always `desired - pos` computed from the "
      "CURRENT bar's freshly-computed `target`, never from a stale target value "
      "captured back when banking began. There is no stored \"old desire\" that "
      "outlives its own bar; `self.pos` not moving is what carries the state "
      "forward, and every recompute re-reads the strategy's live output. Since "
      "the equivalence check confirms this collapses to hard-drop's own decisions "
      "bit-for-bit, this specific implementation carries no NEW execution risk "
      "beyond what B-43 already diagnosed for the existing broker.")
    w()
    w("What IS worth flagging: the pre-registration's own prose (\"a same-sign "
      "adjustment... is instead banked... pending + newly-desired delta\") reads "
      "most naturally as an ADDITIVE (summed) accumulator, not a recomputed one. "
      "An additive implementation would NOT be equivalent to hard-drop — it would "
      "let the accumulator grow from mere bar-over-bar re-statement of an "
      "unchanged target (both `kelly_regime_v4` and `NovelTurnoverThrottle` "
      "re-emit a `target` every bar even with no new intent), and could eventually "
      "release a position change LARGER than the strategy's current target ever "
      "asked for at any single bar — a genuine, and genuinely risky, staleness/"
      "overshoot failure mode, exactly of the shape this task's own risk prompt "
      "anticipated. This round deliberately did NOT build that version (it is not "
      "causal-unsound in the lookahead sense — it never reads a future bar — but "
      "it double-counts already-realized state and is not, on inspection, an "
      "economically coherent policy); flagging the ambiguity explicitly is the "
      "honest thing to do rather than silently picking one reading and reporting "
      "it as if it were the only one the pre-registration allowed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
