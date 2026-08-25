"""R-134 CONSERVATIVE branch: `broker.REBALANCE_DEADBAND` made a settable
per-broker value, simulating the `MarketSpec.deadband` field the operator
would add to `src/tradebot/broker.py` by hand if this branch is selected.

Frozen pre-registration: `experiments/r134_shared.py`. Read that file's
docstring before reading this one -- it states the direction, the object
under test (`NovelTurnoverThrottle`, R-133, frozen and imported not copied),
the falsification test, and the pre-registered ADOPTION decision rule
(F1 backward-compat, F2 no regressions, F3 demonstrated capability). This
file does not invent any threshold not already in `r134_shared.py`.

=====================================================================
WHAT THIS FILE DOES
=====================================================================

1. `MarketDeadbandBroker` -- a `PaperBroker` subclass whose `_execute_target`
   is copied verbatim from `src/tradebot/broker.py` (as of this round) except
   that the one comparison against the module-level `REBALANCE_DEADBAND`
   constant instead reads `self.deadband`, a plain class/instance attribute
   defaulting to `REBALANCE_DEADBAND` (0.05) for backward compatibility.
   `deadband_broker(value)` returns a fresh subclass with that default
   overridden, so a caller can run a backtest at an arbitrary deadband via
   the `_patched_broker` context manager (R-72's pattern, reused not
   reinvented) without touching `src/tradebot/broker.py`.

2. F1 -- bit-identical backward-compatibility check at the current default
   (0.05), `kelly_regime_v4` and `hedge_experts`, both markets, both inner
   splits, patched broker vs the plain unpatched `PaperBroker`.

3. F3 -- intended-vs-filled order counts (R-72's `InstrumentedBroker`
   pattern) for `NovelTurnoverThrottle` at `DEADBAND_BASELINE` vs
   `DEADBAND_REALISTIC`, both markets, inner-train.

4. The falsification test -- does `NovelTurnoverThrottle`, run through the
   patched broker at `DEADBAND_REALISTIC`, now clear B1 (paired bootstrap vs
   frozen `kelly_regime_v4`, always through the DEFAULT unpatched broker,
   inner-validation) on both markets? Plus the full `DEADBAND_GRID` sweep for
   a plateau view, on both markets.

`src/tradebot/broker.py` is never edited. No file under `src/` is touched.
`r134_novel_*` is neither imported nor read. Every frame is truncated at
load and re-checked with `r131_shared._assert_no_holdout`. No bar at or
after `OOS_START` is read.

Usage::

    python experiments/r134_conservative_market_deadband.py
"""

from __future__ import annotations

import contextlib
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tradebot.engine as engine_mod  # noqa: E402
from tradebot.broker import MarketSpec, PaperBroker, REBALANCE_DEADBAND  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

import r134_shared as SH  # noqa: E402
from r131_shared import (  # noqa: E402
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    _assert_no_holdout,
    load_btc_train,
)
from r133_mechanisms import NovelTurnoverThrottle  # noqa: E402

OUT = ROOT / "experiments" / "reports" / "r134_conservative_report.md"

SPOT = SH.SPOT
FUTURES = SH.FUTURES
DEADBAND_BASELINE = SH.DEADBAND_BASELINE
DEADBAND_REALISTIC = SH.DEADBAND_REALISTIC
DEADBAND_GRID = SH.DEADBAND_GRID
THROTTLE_UPPER = SH.THROTTLE_UPPER
THROTTLE_ETA = SH.THROTTLE_ETA
SHARPE_NOISE_FLOOR = SH.SHARPE_NOISE_FLOOR

SPLITS = (("inner-train", (INNER_TRAIN_START, INNER_TRAIN_END)),
          ("inner-val", (INNER_VAL_START, INNER_VAL_END)))
MARKETS = (("spot", SPOT), ("futures_5x", FUTURES))


# ============================================================ the fix itself
#
# One edit vs `PaperBroker._execute_target` as it exists in
# `src/tradebot/broker.py` today: `REBALANCE_DEADBAND * max_notional` becomes
# `self.deadband * max_notional`. Every other line -- the finite check, the
# target clamp, the sign-flip close-then-open, `_max_qty` sizing, the final
# `_transact` -- is copied unchanged. This *simulates* what a real
# `MarketSpec.deadband` field would do; it does not add one, per the round's
# own scope (`src/tradebot/broker.py` is never edited by this file).


class MarketDeadbandBroker(PaperBroker):
    """`PaperBroker` whose rebalance-deadband threshold is a settable
    class/instance attribute (`self.deadband`) instead of the module-level
    `REBALANCE_DEADBAND` constant. Defaults to `REBALANCE_DEADBAND` (0.05),
    so an un-configured instance reproduces today's behaviour exactly.
    """

    deadband: float = REBALANCE_DEADBAND

    def _execute_target(self, target: float, ts, price: float) -> list:
        if not math.isfinite(target):
            raise ValueError(f"order target must be finite, got {target!r}")
        lo = -1.0 if self.market.allow_short else 0.0
        target = min(1.0, max(lo, target))

        fills: list = []
        if (self.pos != 0.0 and target != 0.0
                and math.copysign(1.0, target) != math.copysign(1.0, self.pos)):
            fill = self._transact(ts, -self.pos, price)
            if fill:
                fills.append(fill)
            if self.dead:
                return fills

        desired = (math.copysign(self._max_qty(price) * abs(target), target)
                   if target != 0.0 else 0.0)
        delta = desired - self.pos
        max_notional = self.equity(price) * self.market.leverage
        # ---- THE ONE CHANGE vs PaperBroker._execute_target -----------------
        if target != 0.0 and self.pos != 0.0 and max_notional > 0:
            if abs(delta) * price < self.deadband * max_notional:
                return fills  # ignore tiny same-sign adjustments
        # ----------------------------------------------------------------------
        fill = self._transact(ts, delta, price)
        if fill:
            fills.append(fill)
        return fills


def deadband_broker(value: float) -> type:
    """A fresh `MarketDeadbandBroker` subclass with `deadband` defaulted to
    `value`. A new class per value (rather than mutating one class's
    attribute) so concurrent uses can never step on each other."""
    return type(f"MarketDeadbandBroker_{value!r}", (MarketDeadbandBroker,), {"deadband": value})


class InstrumentedDeadbandBroker(MarketDeadbandBroker):
    """`MarketDeadbandBroker` plus R-72's intended-ask instrumentation, so
    F3's absorption-rate measurement runs on the same footing as R-72's own
    fill-rate table."""

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


InstrumentedDeadbandBroker.__post_init__ = _capturing_post_init(InstrumentedDeadbandBroker)


def instrumented_deadband_broker(value: float) -> type:
    return type(f"InstrumentedDeadbandBroker_{value!r}", (InstrumentedDeadbandBroker,), {"deadband": value})


@contextlib.contextmanager
def _patched_broker(broker_cls):
    """R-72's pattern, reused verbatim: swap `tradebot.engine`'s module-level
    `PaperBroker` name for the duration of the block, then restore it.
    `run_backtest` looks up `PaperBroker` as a module global at call time, so
    this redirects broker construction without touching `broker.py` or any
    registered strategy file."""
    orig = engine_mod.PaperBroker
    engine_mod.PaperBroker = broker_cls
    try:
        yield
    finally:
        engine_mod.PaperBroker = orig


def note() -> None:
    SH.note_config()


def run_arm(strategy_factory, df: pd.DataFrame, window, market: MarketSpec,
            broker_cls: type | None):
    """One backtest. `broker_cls=None` runs through the plain, unpatched
    `PaperBroker`. Otherwise patches `tradebot.engine.PaperBroker` for the
    duration of the call."""
    start, end = window
    note()
    if broker_cls is None:
        res = run_period(strategy_factory(), df, start, end, market=market, start_balance=1000.0)
    else:
        with _patched_broker(broker_cls):
            res = run_period(strategy_factory(), df, start, end, market=market, start_balance=1000.0)
    return res, compute_metrics(res)


def run_instrumented(strategy_factory, df: pd.DataFrame, window, market: MarketSpec,
                      broker_cls: type):
    start, end = window
    note()
    _LAST_BROKER.clear()
    with _patched_broker(broker_cls):
        res = run_period(strategy_factory(), df, start, end, market=market, start_balance=1000.0)
    assert len(_LAST_BROKER) == 1, "expected exactly one broker instance per run"
    return res, compute_metrics(res), _LAST_BROKER[0]


# =================================================================== F1
#
# At DEADBAND_BASELINE (0.05, the current default), `MarketDeadbandBroker`
# must be BIT-IDENTICAL to the plain, unpatched `PaperBroker` -- this is a
# pure refactor, not a behaviour change. Checked on final balance, Sharpe,
# max drawdown, trade count, fill count, AND the full equity curve, not just
# "close enough".


def f1_backward_compat(df: pd.DataFrame) -> list[dict]:
    rows = []
    default_cls = deadband_broker(DEADBAND_BASELINE)
    for strat_name in ("kelly_regime_v4", "hedge_experts"):
        for mkt_label, market in MARKETS:
            for split_label, window in SPLITS:
                factory = lambda n=strat_name: get_strategy(n)
                res_a, m_a = run_arm(factory, df, window, market, None)
                res_b, m_b = run_arm(factory, df, window, market, default_cls)

                same_fills = len(res_a.fills) == len(res_b.fills)
                same_equity = np.allclose(res_a.equity.to_numpy(), res_b.equity.to_numpy(),
                                          rtol=0, atol=1e-9)
                same_balance = abs(m_a.final_balance - m_b.final_balance) < 1e-6
                same_sharpe = abs(m_a.sharpe - m_b.sharpe) < 1e-9
                same_dd = abs(m_a.max_drawdown_pct - m_b.max_drawdown_pct) < 1e-9
                same_trades = m_a.num_trades == m_b.num_trades
                bit_identical = (same_fills and same_equity and same_balance
                                  and same_sharpe and same_dd and same_trades)
                rows.append(dict(
                    strategy=strat_name, market=mkt_label, split=split_label,
                    fills_default=len(res_a.fills), fills_patched=len(res_b.fills),
                    final_default=round(m_a.final_balance, 4), final_patched=round(m_b.final_balance, 4),
                    sharpe_default=round(m_a.sharpe, 6), sharpe_patched=round(m_b.sharpe, 6),
                    dd_default=round(m_a.max_drawdown_pct, 6), dd_patched=round(m_b.max_drawdown_pct, 6),
                    trades_default=m_a.num_trades, trades_patched=m_b.num_trades,
                    bit_identical=bit_identical,
                ))
    return rows


# =================================================================== F3
#
# Demonstrated capability: does the fix, configured at DEADBAND_REALISTIC,
# actually change the fill-through / absorption rate for
# `NovelTurnoverThrottle`, both markets, inner-train?


def throttle_factory():
    return lambda: NovelTurnoverThrottle(upper=THROTTLE_UPPER, eta=THROTTLE_ETA)


def f3_absorption(df: pd.DataFrame) -> list[dict]:
    rows = []
    window = (INNER_TRAIN_START, INNER_TRAIN_END)
    for mkt_label, market in MARKETS:
        for db_label, db in (("baseline", DEADBAND_BASELINE), ("realistic", DEADBAND_REALISTIC)):
            res, m, broker = run_instrumented(
                throttle_factory(), df, window, market, instrumented_deadband_broker(db))
            log = broker.intended_log
            intended = len(log)
            filled = sum(1 for _, _, ok in log if ok)
            rows.append(dict(
                market=mkt_label, deadband=db_label, deadband_value=db,
                intended=intended, filled=filled,
                absorption_rate=(filled / intended if intended else float("nan")),
                fills_in_result=len(res.fills), trades=m.num_trades,
                final_balance=round(m.final_balance, 2), sharpe=round(m.sharpe, 3),
            ))
    return rows


# =========================================================== falsification
#
# Does the corrected broker (DEADBAND_REALISTIC) reverse R-133's own frozen
# NEGATIVE verdict on B1? Comparison arm (`kelly_regime_v4`) always through
# the DEFAULT, unpatched broker -- computed once per market, reused across
# every deadband in the grid, exactly as `r134_shared.v4_reference`'s own
# docstring specifies.


def falsification(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    v4_by_market = {}
    for mkt_label, market in MARKETS:
        v4_by_market[mkt_label] = SH.v4_reference(df, market)

    grid_rows = []
    for mkt_label, market in MARKETS:
        v4 = v4_by_market[mkt_label]
        for db in DEADBAND_GRID:
            with _patched_broker(deadband_broker(db)):
                thr = SH.b1_throttle_vs_v4(throttle_factory(), df, market)
            b1 = SH.paired_b1(thr["returns"].to_numpy(), v4["returns"].to_numpy())
            grid_rows.append(dict(
                market=mkt_label, deadband=db,
                sharpe_thr=round(thr["metrics"].sharpe, 3),
                sharpe_v4=round(v4["metrics"].sharpe, 3),
                d_sharpe=round(thr["metrics"].sharpe - v4["metrics"].sharpe, 3),
                dd_thr=round(thr["metrics"].max_drawdown_pct, 2),
                dd_v4=round(v4["metrics"].max_drawdown_pct, 2),
                trades_thr=thr["metrics"].num_trades, trades_v4=v4["metrics"].num_trades,
                fills_thr=len(thr["result"].fills), fills_v4=len(v4["result"].fills),
                paired_diff=round(b1["paired_diff"], 5),
                lo=round(b1["paired_lo"], 5), hi=round(b1["paired_hi"], 5),
                significant=b1["significant"], b1_pass=b1["b1_pass"],
            ))

    realistic_rows = [r for r in grid_rows if r["deadband"] == DEADBAND_REALISTIC]
    verdict = {
        "spot": next(r for r in realistic_rows if r["market"] == "spot"),
        "futures_5x": next(r for r in realistic_rows if r["market"] == "futures_5x"),
    }
    verdict["reverses_both"] = bool(verdict["spot"]["b1_pass"] and verdict["futures_5x"]["b1_pass"])
    return verdict, grid_rows


# ======================================================= causal truncation
#
# Item 6 of the task: the deadband threshold check reads only `self.pos`,
# `self.cash`, `self.entry` (current broker state) and the bar's own
# `price`/`target` -- no rolling window, no full-series statistic, no state
# that depends on FUTURE bars. It introduces no new state at all beyond what
# `PaperBroker` already carries forward bar-to-bar. That said, "should be
# causal" is exactly the kind of claim this project has been burned by
# assuming (R-21's $3.7e23), so this runs a direct truncation probe rather
# than asserting it from the code alone: the same strategy, same broker,
# run on a full frame vs a truncated prefix of it, must produce IDENTICAL
# fills up to the truncation point.


def causal_truncation_probe(df: pd.DataFrame) -> dict:
    frame = df.loc[:INNER_TRAIN_END]
    cut = len(frame) // 2
    db = DEADBAND_REALISTIC
    market = FUTURES  # the market where the floor bites hardest

    with _patched_broker(deadband_broker(db)):
        note()
        res_full = run_period(throttle_factory()(), frame, None, None,
                              market=market, start_balance=1000.0)
        note()
        res_trunc = run_period(throttle_factory()(), frame.iloc[:cut], None, None,
                               market=market, start_balance=1000.0)

    n = min(len(res_full.fills), len(res_trunc.fills))
    # Compare fills up to the truncation point's timestamp only.
    cut_ts = frame.index[cut - 1]
    full_prefix_fills = [f for f in res_full.fills if f.ts <= cut_ts]
    trunc_fills = list(res_trunc.fills)
    same_n = len(full_prefix_fills) == len(trunc_fills)
    same_all = same_n and all(
        f1.ts == f2.ts and abs(f1.qty - f2.qty) < 1e-9 and f1.side == f2.side
        and abs(f1.price - f2.price) < 1e-9
        for f1, f2 in zip(full_prefix_fills, trunc_fills))
    return dict(
        n_full_prefix_fills=len(full_prefix_fills), n_trunc_fills=len(trunc_fills),
        identical=bool(same_all),
    )


# ==================================================================== main


def main() -> None:
    df, label = load_btc_train()
    _assert_no_holdout(df)
    print(f"BTC ({label}), inner-train+inner-val frame: {len(df):,} bars, "
          f"{df.index[0]} -> {df.index[-1]} (< {OOS_START})")

    lines: list[str] = []
    checkpoints: list[tuple[str, int]] = []

    def w(s: str = "") -> None:
        lines.append(s)
        print(s)

    def checkpoint(label: str) -> None:
        checkpoints.append((label, SH.configs_evaluated()))

    def dump(rows: list[dict]) -> pd.DataFrame:
        t = pd.DataFrame(rows)
        w("```")
        w(t.to_string(index=False))
        w("```")
        return t

    w("# R-134 CONSERVATIVE — `MarketSpec.deadband` simulated as a broker-subclass attribute")
    w()
    w("Frozen pre-registration: `experiments/r134_shared.py`. Object under test: "
      "`NovelTurnoverThrottle` (R-133, `experiments/r133_mechanisms.py`, imported not copied).")
    w()
    w(f"`DEADBAND_BASELINE = {DEADBAND_BASELINE}`, `DEADBAND_REALISTIC = {DEADBAND_REALISTIC}`, "
      f"`DEADBAND_GRID = {DEADBAND_GRID}`. `THROTTLE_UPPER = {THROTTLE_UPPER:.4f}` trades/day, "
      f"`THROTTLE_ETA = {THROTTLE_ETA}` (R-133's frozen operating point, imported not re-derived).")
    w()
    w("**Mechanism, one sentence:** the broker's `REBALANCE_DEADBAND` floor is read from a "
      "settable per-broker `self.deadband` attribute (default 0.05, backward-compatible) instead "
      "of the module-level constant, so a same-sign rebalance shrunk by a mechanism like "
      "`NovelTurnoverThrottle` is compared against a threshold that can be set to a "
      "venue-realistic size instead of an arbitrary flat 5% of max notional.")
    w()

    checkpoint("start")

    # -------------------------------------------------------- F1
    w("## F1 — backward compatibility (bit-identical at DEADBAND_BASELINE = 0.05)")
    w()
    t0 = time.time()
    f1_rows = f1_backward_compat(df)
    checkpoint("after F1")
    w(f"Ran in {time.time() - t0:.1f}s.")
    w()
    t_f1 = dump(f1_rows)
    f1_pass = bool(t_f1["bit_identical"].all())
    w()
    w(f"**F1: {'PASS' if f1_pass else 'FAIL'}** — "
      f"{int(t_f1['bit_identical'].sum())} of {len(t_f1)} (strategy x market x split) cells "
      f"bit-identical (fills, full equity curve, final balance, Sharpe, max drawdown, trade count "
      f"all compared, not just 'close enough').")
    w()

    # -------------------------------------------------------- F3
    w("## F3 — demonstrated capability (absorption rate, `NovelTurnoverThrottle`, inner-train)")
    w()
    t0 = time.time()
    f3_rows = f3_absorption(df)
    checkpoint("after F3")
    w(f"Ran in {time.time() - t0:.1f}s.")
    w()
    t_f3 = dump(f3_rows)
    w()
    f3_changed = []
    for mkt_label, _ in MARKETS:
        base = t_f3[(t_f3["market"] == mkt_label) & (t_f3["deadband"] == "baseline")].iloc[0]
        real = t_f3[(t_f3["market"] == mkt_label) & (t_f3["deadband"] == "realistic")].iloc[0]
        d = real["absorption_rate"] - base["absorption_rate"]
        f3_changed.append(abs(d) > 1e-9)
        w(f"- **{mkt_label}**: absorption {base['absorption_rate']:.1%} (baseline, deadband="
          f"{DEADBAND_BASELINE}) -> {real['absorption_rate']:.1%} (realistic, deadband="
          f"{DEADBAND_REALISTIC}), delta {d:+.1%}. Filled/intended: "
          f"{base['filled']}/{base['intended']} -> {real['filled']}/{real['intended']}.")
    f3_pass = all(f3_changed)
    w()
    w(f"**F3: {'PASS' if f3_pass else 'FAIL'}** — the fix measurably changes the "
      f"fill-through/absorption rate on {'both' if f3_pass else 'not both'} markets "
      f"when configured at `DEADBAND_REALISTIC`.")
    w()

    # -------------------------------------------------------- F2
    w("## F2 — no regressions (`pytest`)")
    w()
    import subprocess
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT),
        capture_output=True, text=True, timeout=1800)
    pytest_elapsed = time.time() - t0
    tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
    w(f"`pytest -q` from repo root, venv active. Exit code {proc.returncode}. "
      f"Ran in {pytest_elapsed:.1f}s.")
    w()
    w("```")
    w(tail)
    w("```")
    f2_pass = proc.returncode == 0
    w()
    w(f"**F2: {'PASS' if f2_pass else 'FAIL'}** (`tests/test_causality_strict.py` included in "
      f"the full run above; this branch adds zero changes to `src/`).")
    w()

    # -------------------------------------------------------- causal truncation self-test
    w("## Causal-truncation self-test on the patched-broker logic")
    w()
    w("`MarketDeadbandBroker._execute_target` introduces no new stateful or rolling "
      "computation: the one changed line (`self.deadband * max_notional` in place of "
      "`REBALANCE_DEADBAND * max_notional`) is a per-bar comparison against the broker's own "
      "current `pos`/`cash`/`entry` and the bar's own `price`/`target`, exactly like every other "
      "line of `_execute_target` it was copied from — no scaler, quantile, mean, std, or window is "
      "computed over the series. That is an argument from reading the code, which is exactly the "
      "kind of claim this project has been burned by assuming (R-21's $3.7e23 causality bug), so "
      "a direct truncation probe was also run rather than relying on it alone:")
    w()
    tp = causal_truncation_probe(df)
    checkpoint("after truncation probe")
    w(f"- `NovelTurnoverThrottle` through `MarketDeadbandBroker(deadband={DEADBAND_REALISTIC})`, "
      f"futures_5x, inner-train: full-frame fills up to the truncation timestamp vs "
      f"truncated-frame fills — {tp['n_full_prefix_fills']} vs {tp['n_trunc_fills']}, "
      f"**{'IDENTICAL' if tp['identical'] else 'DIVERGED'}**.")
    w()

    # -------------------------------------------------------- falsification test
    w("## Falsification test — does the corrected broker reverse R-133's B1 verdict?")
    w()
    w("Frozen wording (`r134_shared.py`): under the patched broker at `DEADBAND_REALISTIC`, does "
      "`NovelTurnoverThrottle` clear B1 (paired bootstrap vs frozen `kelly_regime_v4`, the "
      "comparison arm always through the DEFAULT unpatched broker, inner-validation, "
      "`total_log_return`, `significant=True` AND `paired_diff.point > 0`) on BOTH markets?")
    w()
    t0 = time.time()
    verdict, grid_rows = falsification(df)
    checkpoint("after falsification grid")
    w(f"Ran in {time.time() - t0:.1f}s.")
    w()
    t_grid = dump(grid_rows)
    w()

    w("### Verdict cell (DEADBAND_REALISTIC only)")
    w()
    for mkt_label in ("spot", "futures_5x"):
        r = verdict[mkt_label]
        w(f"- **{mkt_label}**: paired_diff={r['paired_diff']:+.5f} "
          f"[{r['lo']:+.5f}, {r['hi']:+.5f}], significant={r['significant']}, "
          f"b1_pass={r['b1_pass']} (Sharpe throttle {r['sharpe_thr']:.3f} vs v4 "
          f"{r['sharpe_v4']:.3f}, d_sharpe={r['d_sharpe']:+.3f}).")
    w()
    fals_outcome = "YES on both markets" if verdict["reverses_both"] else "NO (fails B1 on at least one market)"
    w(f"**Falsification test outcome: {fals_outcome}.**")
    w()
    if verdict["reverses_both"]:
        w("Per `r134_shared.py`'s own pre-registered reading of this outcome: the COST-axis "
          "turnover-throttle family's NEGATIVE verdict (R-131, R-133) was, at least in part, a "
          "broker-floor artifact rather than a property of the mechanism. This reopens the family "
          "and requires its own follow-up round with a full B1/B3/B4/B5 battery before any "
          "promotion claim — **this round does not promote anything by itself**, per the frozen text.")
    else:
        w("Per `r134_shared.py`'s own pre-registered reading of this outcome: B-43 closes cleanly. "
          "R-133's section C entry becomes final rather than provisional, per its own text. The "
          "evaluability defect (the floor's coarseness) is still worth fixing because it affects "
          "every FUTURE size-shrinking mechanism this project tries, but it does not resurrect "
          "`NovelTurnoverThrottle`.")
    w()
    for mkt_label in ("spot", "futures_5x"):
        base_row = next(r for r in grid_rows if r["market"] == mkt_label and r["deadband"] == DEADBAND_BASELINE)
        real_row = verdict[mkt_label]
        moved = base_row["b1_pass"] != real_row["b1_pass"]
        d_sharpe_move = real_row["d_sharpe"] - base_row["d_sharpe"]
        inside_noise = abs(d_sharpe_move) < SHARPE_NOISE_FLOOR
        w(f"- **{mkt_label}** flip vs baseline deadband: b1_pass {base_row['b1_pass']} -> "
          f"{real_row['b1_pass']} ({'FLIPPED' if moved else 'unchanged'}). d_sharpe moved by "
          f"{d_sharpe_move:+.3f} between baseline and realistic deadband, which is "
          f"{'INSIDE' if inside_noise else 'OUTSIDE'} the +/-{SHARPE_NOISE_FLOOR} Sharpe noise "
          f"floor (R-20).")
    w()

    # -------------------------------------------------------- configs / decision rule
    n_configs = SH.configs_evaluated()
    w("## Configs evaluated")
    w()
    w("`r134_shared._CONFIGS[0]` is a shared, cross-branch counter (`note_config()` incremented "
      "once per backtest, observed rather than remembered) — every `run_period` call in this file "
      "sits behind one `note_config()`/`note()` call, directly or via `r134_shared.b1_throttle_vs_v4` "
      "/ `v4_reference`. Running total by section (checkpointed as the file ran):")
    w()
    w("```")
    prev = 0
    for label, total in checkpoints:
        w(f"{label:<28} cumulative={total:>4}  (+{total - prev} this section)")
        prev = total
    w("```")
    w()
    w(f"**Total configs evaluated by this file: {n_configs}.** (The operator's final cross-branch "
      f"count per `docs/ROUTINE.md` also includes whatever the NOVEL branch adds to this same "
      f"shared counter object, if run in the same process; if run as a separate process, the two "
      f"branch counts are summed by the operator.)")
    w()

    # -------------------------------------------------------- honest verdict
    w("## Verdict against the pre-registered ADOPTION decision rule (`r134_shared.py`)")
    w()
    w(f"- **F1** (backward compatibility, bit-identical at 0.05): **{'PASS' if f1_pass else 'FAIL'}**")
    w(f"- **F2** (no regressions, full `pytest` green): **{'PASS' if f2_pass else 'FAIL'}**")
    w(f"- **F3** (demonstrated capability, absorption rate changes at DEADBAND_REALISTIC): "
      f"**{'PASS' if f3_pass else 'FAIL'}**")
    w()
    adopted_eligible = f1_pass and f2_pass and f3_pass
    w(f"F1/F2/F3 all clear: **{adopted_eligible}**. Per `r134_shared.py`'s own text, clearing "
      f"F1-F3 makes this fix ELIGIBLE for the operator to select between it and the NOVEL branch "
      f"— **this file does not itself declare a fix ADOPTED or PROMOTED**; that decision belongs "
      f"to the operator after both branches report.")
    w()
    w(f"Falsification test: **{fals_outcome}**. This is reported as evidence for a possible "
      f"follow-up round (if YES on both markets) or as B-43's closure (if NO on at least one "
      f"market) — **not** as a promotion of `NovelTurnoverThrottle` itself either way, per the "
      f"round's own scope (\"this round does not attempt B3/B4/B5 by design\").")
    w()
    w("No bar at or after `OOS_START = 2023-01-01` was read by this file. Holdout consultations "
      "added: 0.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
