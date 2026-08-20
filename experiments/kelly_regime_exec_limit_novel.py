#!/usr/bin/env python
"""kelly_regime_v4 with a probabilistic maker/limit fill model (NOVEL branch, R-56).

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5. Does not modify ``kelly_regime_v4.py``,
``kelly_regime_v3.py``, ``kelly_regime.py``, ``engine.py`` or
``broker.py`` -- all reused read-only as libraries. Zero overlap with the
sibling conservative branch's file (not read, not touched).

The idea
--------
Every registered strategy, including the current leader ``kelly_regime_v4``,
fills every trade as a TAKER market order at the next bar's open
(``tradebot.engine.run_backtest`` / ``tradebot.broker.PaperBroker``). R-12/
R-13 established that at Bitstamp's real entry-tier taker fee (0.40%) "no
strategy here beats buy-and-hold", and R-13's own ``fee_study.py`` names
the gap this fills almost verbatim: *"No Bitstamp TAKER tier reaches
[break-even]; only the maker rate does, and that is a change of order
type -- with fill risk this backtester does not model -- rather than a
change of tier."* This module builds that missing simulation capability
(ROUTINE.md step 1, item 3: "if not [simulable], today's job is to build
the missing simulation capability and record that -- do not proxy it out
of OHLCV").

Constraint attacked: **COST** -- costs that scale with the signal -- via a
different mechanism than every prior COST-axis result: L-05/L-06
(``kelly_regime_ev``) decide **WHEN** to trade (a no-trade deadband);
R-12/R-13 (``fee_study.py``) swept the taker fee TIER. This changes **HOW**
an already-decided trade fills, leaving v4's vote/scale (SIZE) signal
completely untouched -- v4's ``target[i] = frac[i] * scale[i]`` is called
byte-for-byte via the real registered class (see ``compute_v4_signal``
below); only the mechanism that turns a `target` change into fills is
new. Not a duplicate of R-40 (``kelly_regime_v8_uncertainty_shrink``,
Bayesian Kelly shrinkage on the SIZE signal itself, NEGATIVE) -- nothing
here touches ``frac`` or ``scale``, only how a already-decided notional
delta is executed.

Two variants exist in this round. The sibling CONSERVATIVE branch
(``experiments/kelly_regime_exec_limit_conservative.py``, a different
agent's disjoint file, not read here) assumes a resting limit order fills
with 100% certainty the instant the bar's range touches its price. That
is optimistic: a real order resting in a queue at a touched price level
is not guaranteed to fill (adverse selection / queue priority), and this
project has been burned before by treating unavailable microstructure
detail as knowable -- L-14 ``camouflage_flow``'s lesson, verbatim: "BVC
from OHLCV is a price transform, not order flow. Proxying unavailable
data out of price adds no information." This module is the NOVEL,
more-realistic alternative:

1. **Fill-probability discount on touch**, a function of how deep into
   the bar's range the touch went (see "Literature grounding" below) --
   a touch that barely grazes the limit price is treated as less likely
   to have actually filled a resting order than one where price traded
   well through it. OHLCV-only: uses only open/high/low/close/volume
   already in the committed CSV files, nothing invented.
2. **Adaptive posting aggressiveness**: the limit offset from the
   decision bar's close scales with v4's OWN existing conditional-
   volatility-targeting signal (``scale[i]``, reproduced from
   ``kelly_regime_v3.prepare`` -- see ``compute_v4_signal``) as a proxy
   for conviction/urgency: high ``scale[i]`` (the vol-targeting state
   wants large exposure -- a breakout regime) posts CLOSER to market
   (higher fill probability, smaller expected price improvement); low
   ``scale[i]`` posts FURTHER from market (more patient, better price).
3. A hard taker-fallback safety net: any order not filled within
   ``patience_bars`` is swept at the market (taker fee) at that bar's
   open, so exposure can never drift indefinitely from target. This part
   is NOT the novel contribution -- it is a shared safety mechanism, kept
   unconditionally.

Literature grounding for (1), stated precisely and honestly
-------------------------------------------------------------
The operator's brief cites Cont, Kukanov & Stoikov (2014, "The Price
Impact of Order Book Events", J. Financial Econometrics 12(1):47-88) as
an example source. Checked directly (WebSearch, this session): that paper
is actually about the linear relation between *order flow imbalance* (OFI)
and short-horizon price changes, not a queue-position fill-probability
model -- citing it for (1) would be a citation that does not say what it
is cited for. The paper that actually models P(fill) as a function of
queue position and the volume that trades through a price level before
cancellation is **Cont & Kukanov (2017), "Optimal order placement in
limit order markets", Quantitative Finance 17(1):21-39** (formulates the
market/limit order split as a convex optimization over the probability a
resting order at a given queue position gets executed before the horizon,
which is *increasing in the volume that trades through the level* and
*decreasing in the size of the queue ahead of the order*). That is the
correct citation and the one used here.

This project has no order-book depth or queue-position data (OHLCV only,
per ``docs/ROUTINE.md`` step 1 item 3 and the "never proxy unavailable
data out of price" standing rule) -- so the queue length itself is NOT
recovered. What IS recoverable causally from a later bar's own OHLC,
without inventing anything, is how far price traveled *through* the
limit level once touched, which is the same directional driver Cont &
Kukanov's model turns on (more volume trading through a level, for a
similarly-sized queue, raises the fill probability). The proxy used here:

    penetration(bar j) = clip((limit_price - low_j) / (high_j - low_j), 0, 1)   [BUY]
    penetration(bar j) = clip((high_j - limit_price) / (high_j - low_j), 0, 1)  [SELL]
    fill_prob(bar j)   = penetration(bar j) ** gamma

applied only when the bar touches the level at all (``low_j <= limit_price``
for a buy). This is a monotone, bounded [0,1], deterministic-given-inputs
function of a later bar's own OHLC -- exactly the brief's requirement --
and it recovers the right qualitative behavior (barely-grazing touch ->
near-zero fill probability; price trading deep through the level -> fill
probability approaching 1) without claiming to reconstruct the actual
queue-depletion process the cited paper derives under book-depth data
this project does not have. ``gamma`` is swept (0.5 concave / 1.0 linear
/ 2.0 convex) rather than asserted, and a completely non-adaptive,
non-literature-grounded FLAT probability is run as the pre-registered
ablation (requirement 4) to show whether the shape earns its keep over
"some discount, any discount".

Falsification tests, pre-registered before any run (ROUTINE.md step 2)
------------------------------------------------------------------------
This result does NOT survive if:
  (a) it does not replicate DIRECTIONALLY (fee $ saved > 0, maker-fill
      rate > 0, and it does not make the crash-transition lag qualitatively
      worse) on the ETH falsification pair
      (``data/ethusd_bitfinex_5m.csv.gz``, 2016-03-09 -> 2019-12-31, i.e.
      entirely pre-2020, used whole);
  (b) it fails a pre-2020 BTC-only control window (2017-01-01 ->
      2019-12-31, a subset of inner-train);
  (c) fills materially lag during the historical bear-market de-risking
      transitions that are v4's entire edge (L-01: "Its entire edge is in
      the windows that contain one" -- a crash). Every regime-flip-to-flat
      event in inner-train/inner-validation is checked for how many bars
      the limit-chase mechanism took to fully flatten vs. the 1-bar taker
      baseline.

Data discipline
----------------
Everything below is restricted to inner-train (2017-01-01 -> 2020-12-31)
and inner-validation (2021-01-01 -> 2022-12-31) on
``data/btcusd_spot_5m.csv.gz``, loaded via ``tradebot.data.load_dataset``,
plus the ETH falsification file (naturally entirely pre-2020). No bar
dated 2023-01-01 or later is ever constructed, printed, or read by this
module -- grepped for date literals before finishing (see the session
report).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import REBALANCE_DEADBAND, MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_ohlcv_csv  # noqa: E402
from tradebot.engine import BacktestResult, build_trades, validate_ohlcv  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.orders import Fill, Side  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import prefix_bars, run_period  # noqa: E402

# ---------------------------------------------------------------- fee data
# Bitstamp fee schedule, accessed 2026-08-20 (operator, verified via web
# search this round). Entry tier is the realistic one for this project's
# account sizes (R-12/R-13 use it as "the real fee"); top tier included
# only as a sensitivity check, never as the headline.
TAKER_ENTRY = 0.0040
MAKER_ENTRY = 0.0030
TAKER_TOP = 0.0003
MAKER_TOP = 0.0000

# Data discipline: never touch or construct a bar >= this date in this file.
OOS_START = "2023-01-01"
INNER_TRAIN = ("2017-01-01", "2020-12-31")
INNER_VAL = ("2021-01-01", "2022-12-31")
BTC_CONTROL = ("2017-01-01", "2019-12-31")  # pre-2020 control window

V4_WARMUP = KellyRegimeV4().warmup  # 80d anchor + 10 = 23,050 bars


# ============================================================ the v4 signal
def compute_v4_signal(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """v4's ``target[i] = frac[i] * scale[i]`` AND the raw ``scale[i]``.

    ``target`` is taken from the REAL registered ``KellyRegimeV4().prepare``
    (byte-for-byte parity with the strategy this project already tests and
    trusts -- the SIZE/vote signal is untouched by this module).

    ``scale[i]`` -- v3/v4's own conditional-volatility-targeting exposure
    level, used here as the adaptive-posting conviction proxy -- is NOT
    exposed as a column by ``kelly_regime_v3.prepare``, so it is
    reproduced here from the published formula (kelly_regime_v3.py
    lines ~68-93): identical anchor-span EWMs, identical high/low
    in/out hysteresis state machine, identical ``full``/``steady``
    branches. Read-only duplication of a stable, CI-tested, registered
    file -- nothing here is a strategy-signal *change*.
    """
    v4 = KellyRegimeV4()
    prepared = v4.prepare(df.copy())
    target = prepared["target"].to_numpy(dtype=float)

    close = df["close"]
    r = np.log(close).diff()
    vol = (r.ewm(span=v4.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=v4.anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(v4.target_vol / vol, v4.max_leverage)
        steady = np.minimum(v4.target_vol / slow, v4.max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(df)
    scale = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > v4.high_in else (-1 if x < v4.low_in else 0)
            elif state == 1 and x < v4.high_out:
                state = 0
            elif state == -1 and x > v4.low_out:
                state = 0
        scale[i] = full[i] if state != 0 else steady[i]

    assert len(target) == len(scale) == len(df)
    return target, scale, float(v4.max_leverage)


# ============================================================ fill config
@dataclass
class LimitFillConfig:
    """One configuration of the novel execution model."""

    passive_offset_bps: float = 20.0     # posting offset when conviction (scale) is LOW
    aggressive_offset_bps: float = 3.0   # posting offset when conviction (scale) is HIGH
    patience_bars: int = 8               # bars a limit order is allowed to chase (5m bars)
    fill_gamma: float = 1.0              # penetration exponent (literature-grounded model)
    ablation_flat_prob: float | None = None  # set -> ignore penetration/gamma; flat P(fill)
    adaptive_posting: bool = True        # False -> always post at passive_offset (ablation)

    @property
    def passive_frac(self) -> float:
        return self.passive_offset_bps / 10_000.0

    @property
    def aggressive_frac(self) -> float:
        return self.aggressive_offset_bps / 10_000.0

    def tag(self) -> str:
        base = (f"pass{self.passive_offset_bps:g}bp_aggr{self.aggressive_offset_bps:g}bp_"
                f"pat{self.patience_bars}_g{self.fill_gamma:g}")
        if self.ablation_flat_prob is not None:
            base += f"_FLAT{self.ablation_flat_prob:g}"
        if not self.adaptive_posting:
            base += "_fixedpost"
        return base


EPS_QTY = 1e-10
EPS_TARGET = 1e-9


# ============================================================ the simulator
def _try_limit_fill(broker: PaperBroker, order: dict, ts, h: float, l: float,
                     config: LimitFillConfig) -> Fill | None:
    """Resolve one bar of a resting limit order using ONLY that bar's own H/L.

    Causal: called for bar j strictly after the decision bar i that set
    ``order['limit_price']`` (enforced by the caller's ``start_bar`` gate).
    Deterministic given (h, l, limit_price, remaining, gamma) -- no RNG.
    """
    side = order["side"]
    p = order["limit_price"]
    rng = h - l
    if side is Side.BUY:
        touched = l <= p
        depth = max(0.0, p - l)
    else:
        touched = h >= p
        depth = max(0.0, h - p)
    if not touched:
        return None
    pen = 1.0 if rng <= 0.0 else min(1.0, depth / rng)
    fill_prob = config.ablation_flat_prob if config.ablation_flat_prob is not None else pen ** config.fill_gamma
    fill_qty = order["remaining"] * fill_prob
    if fill_qty <= 0.0:
        return None
    delta = fill_qty if side is Side.BUY else -fill_qty
    fill = broker._transact(ts, delta, p, kind="maker")  # noqa: SLF001 (deliberate reuse)
    if fill is None:
        return None  # below exchange min-notional this bar; try again next bar
    order["remaining"] -= fill.qty
    return fill


def _taker_fallback(broker: PaperBroker, order: dict, ts, price: float,
                     taker_market: MarketSpec) -> Fill | None:
    """Sweep whatever remains of ``order`` at the market. The safety net."""
    remaining = order["remaining"]
    if remaining <= EPS_QTY:
        return None
    delta = remaining if order["side"] is Side.BUY else -remaining
    saved = broker.market
    broker.market = taker_market
    try:
        fill = broker._transact(ts, delta, price, kind="taker_fallback")  # noqa: SLF001
    finally:
        broker.market = saved
    if fill is not None:
        order["remaining"] -= fill.qty
    return fill


@dataclass
class RunStats:
    forced_flush_events: int = 0  # times a fresh decision superseded a still-resting order
    maker_fills: int = 0
    maker_qty: float = 0.0
    taker_fallback_fills: int = 0
    taker_fallback_qty: float = 0.0
    decisions: int = 0


def run_limit_backtest(df: pd.DataFrame, market: MarketSpec, start_balance: float,
                        config: LimitFillConfig, trade_start: int = 0,
                        data_label: str = "", fee_maker: float = MAKER_ENTRY,
                        fee_taker: float = TAKER_ENTRY,
                        target: np.ndarray | None = None,
                        scale: np.ndarray | None = None,
                        max_lev_signal: float | None = None,
                        ) -> tuple[BacktestResult, RunStats]:
    """Bar-by-bar simulator for the novel limit/probabilistic-fill model.

    Mirrors ``tradebot.engine.run_backtest``'s per-bar sequence (liquidation
    at open -> fills -> liquidation at extremes -> record equity ->
    decide), but replaces "queue at i, fill at i+1's open" with "queue at
    i, chase a limit price for up to ``patience_bars`` using only bars
    strictly after i, taker-sweep whatever remains". v4's own SIZE signal
    (``target``/``scale``) is untouched -- computed once by
    ``compute_v4_signal`` and passed in (or computed here if omitted).
    """
    validate_ohlcv(df)
    if target is None or scale is None or max_lev_signal is None:
        target, scale, max_lev_signal = compute_v4_signal(df)

    n = len(df)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    index = df.index

    maker_market = replace(market, fee_rate=fee_maker)
    taker_market = replace(market, fee_rate=fee_taker)
    broker = PaperBroker(market=maker_market, start_balance=start_balance)

    equity = [0.0] * n
    fills: list[Fill] = []
    open_order: dict | None = None
    stats = RunStats()
    lev = max(market.leverage, 1e-9)
    lo_frac = -1.0 if market.allow_short else 0.0

    for i in range(n):
        ts = index[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        liq = broker.check_liquidation(ts, o, o, o)
        if liq is not None:
            fills.append(liq)
            open_order = None

        if open_order is not None and not broker.dead:
            if i >= open_order["start_bar"]:
                if i < open_order["deadline"]:
                    fill = _try_limit_fill(broker, open_order, ts, h, l, config)
                    if fill is not None:
                        fills.append(fill)
                        stats.maker_fills += 1
                        stats.maker_qty += fill.qty
                else:
                    fill = _taker_fallback(broker, open_order, ts, o, taker_market)
                    if fill is not None:
                        fills.append(fill)
                        stats.taker_fallback_fills += 1
                        stats.taker_fallback_qty += fill.qty
            if open_order is not None and open_order["remaining"] <= EPS_QTY:
                open_order = None

        if not broker.dead:
            liq = broker.check_liquidation(ts, o, h, l)
            if liq is not None:
                fills.append(liq)
                open_order = None

        equity[i] = broker.equity(c)
        if not math.isfinite(equity[i]):
            raise ValueError(f"equity became non-finite at bar {i} ({index[i]})")

        last_bar = i == n - 1
        if not broker.dead and not last_bar and i >= trade_start:
            changed = (i > 0 and abs(target[i] - target[i - 1]) > EPS_TARGET) or \
                      (i == 0 and abs(target[i]) > EPS_TARGET)
            if changed:
                lev_frac = min(1.0, max(lo_frac, target[i] / lev))
                desired_qty = (math.copysign(broker._max_qty(c) * abs(lev_frac), lev_frac)  # noqa: SLF001
                               if lev_frac != 0.0 else 0.0)
                raw_delta = desired_qty - broker.pos
                max_notional = broker.equity(c) * market.leverage
                is_close_or_flip = (
                    (lev_frac == 0.0 and broker.pos != 0.0) or
                    (broker.pos != 0.0 and lev_frac != 0.0 and
                     math.copysign(1.0, lev_frac) != math.copysign(1.0, broker.pos))
                )
                worth_it = (broker.pos == 0.0 or is_close_or_flip or max_notional <= 0.0 or
                            abs(raw_delta) * c >= REBALANCE_DEADBAND * max_notional)

                if worth_it and abs(raw_delta) > EPS_QTY:
                    stats.decisions += 1
                    if open_order is not None:
                        # A fresh decision supersedes a still-resting order: flatten the
                        # stale remainder now (at this decision bar's own close -- known,
                        # causal) rather than let two orders coexist. Extremely rare given
                        # v4's cadence (~1 decision / 2-3 weeks) vs patience (~tens of
                        # minutes to a few hours); counted for transparency.
                        fill = _taker_fallback(broker, open_order, ts, c, taker_market)
                        if fill is not None:
                            fills.append(fill)
                            stats.taker_fallback_fills += 1
                            stats.taker_fallback_qty += fill.qty
                        open_order = None
                        stats.forced_flush_events += 1
                        desired_qty = (math.copysign(broker._max_qty(c) * abs(lev_frac), lev_frac)  # noqa: SLF001
                                       if lev_frac != 0.0 else 0.0)
                        raw_delta = desired_qty - broker.pos

                    if abs(raw_delta) > EPS_QTY:
                        u = 0.0 if not config.adaptive_posting else min(1.0, max(0.0, scale[i] / max_lev_signal))
                        offset = (config.aggressive_frac + (config.passive_frac - config.aggressive_frac)
                                  * (1.0 - u)) if config.adaptive_posting else config.passive_frac
                        side = Side.BUY if raw_delta > 0 else Side.SELL
                        limit_price = c * (1.0 - offset) if side is Side.BUY else c * (1.0 + offset)
                        open_order = {
                            "side": side, "limit_price": limit_price,
                            "remaining": abs(raw_delta),
                            "start_bar": i + 1, "deadline": i + 1 + config.patience_bars,
                            "decision_bar": i,
                        }

    trades = build_trades(fills, end_price=closes[-1] if n else None, broker=broker)
    result = BacktestResult(
        strategy_name=f"kelly_regime_exec_limit_novel[{config.tag()}]",
        market=market, start_balance=start_balance, data_label=data_label,
        equity=pd.Series(equity, index=index, name="equity"),
        fills=fills, trades=trades, df=df, liquidated=broker.dead,
        fees_paid=broker.fees_paid, funding_paid=broker.funding_paid,
    )
    return result, stats


def run_limit_period(df: pd.DataFrame, start, end, market: MarketSpec,
                      config: LimitFillConfig, start_balance: float = 1_000.0,
                      data_label: str = "", warmup: int = V4_WARMUP,
                      fee_maker: float = MAKER_ENTRY, fee_taker: float = TAKER_ENTRY,
                      ) -> tuple[BacktestResult, RunStats]:
    """``tradebot.window.run_period`` analogue for the novel simulator."""
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")
    prefix = prefix_bars(df, lo, warmup)
    frame = df.iloc[lo - prefix: hi]
    result, stats = run_limit_backtest(frame, market, start_balance, config,
                                        trade_start=prefix, data_label=data_label,
                                        fee_maker=fee_maker, fee_taker=fee_taker)
    if prefix:
        result = replace(result, equity=result.equity.iloc[prefix:], df=result.df.iloc[prefix:])
    return result, stats


# ============================================================ baseline (unmodified engine)
def run_taker_baseline(df: pd.DataFrame, start, end, market: MarketSpec,
                        fee: float, start_balance: float = 1_000.0, data_label: str = ""):
    """The real, unmodified ``kelly_regime_v4`` on the real, unmodified engine."""
    m = replace(market, fee_rate=fee)
    result = run_period(KellyRegimeV4(), df, start, end, market=m,
                         start_balance=start_balance, data_label=data_label)
    return compute_metrics(result)


# ============================================================ reporting helpers
def summarize(tag: str, result: BacktestResult, stats: RunStats | None = None,
              baseline_final: float | None = None, baseline_fees: float | None = None) -> dict:
    m = compute_metrics(result)
    row = {
        "tag": tag, "market": result.market.name, "final": m.final_balance,
        "profit_pct": m.profit_pct, "trades": m.num_trades, "dd_pct": m.max_drawdown_pct,
        "sharpe": m.sharpe, "fees": m.fees_paid, "liquidated": m.liquidated,
    }
    if stats is not None:
        total_qty = stats.maker_qty + stats.taker_fallback_qty
        row["maker_fill_rate"] = stats.maker_qty / total_qty if total_qty > 0 else float("nan")
        row["decisions"] = stats.decisions
        row["forced_flush"] = stats.forced_flush_events
    if baseline_final is not None:
        row["vs_baseline_final_pct"] = 100.0 * (m.final_balance / baseline_final - 1.0)
    if baseline_fees is not None:
        row["fee_dollars_saved"] = baseline_fees - m.fees_paid
    return row


def print_row(row: dict) -> None:
    extra = ""
    if "maker_fill_rate" in row:
        extra = (f" maker_fill={row['maker_fill_rate']:.1%} decisions={row['decisions']:>4d} "
                 f"flush={row['forced_flush']}")
    vb = f" vsBase={row['vs_baseline_final_pct']:+6.1f}%" if "vs_baseline_final_pct" in row else ""
    fs = f" feeSaved=${row['fee_dollars_saved']:>7,.0f}" if "fee_dollars_saved" in row else ""
    print(f"{row['tag']:60s} {row['market']:11s} final=${row['final']:>11,.0f} "
          f"({row['profit_pct']:>+8.1f}%) trades={row['trades']:>4d} DD={row['dd_pct']:>5.1f}% "
          f"sharpe={row['sharpe']:>5.2f} fees=${row['fees']:>8,.0f}{vb}{fs}{extra}"
          f"{' LIQUIDATED' if row['liquidated'] else ''}")


# ============================================================ causality probe
def causality_probe(df: pd.DataFrame, config: LimitFillConfig, market: MarketSpec) -> bool:
    """Tamper probe: decisions at/before a cut bar must be invariant to bars from the cut onward.

    Same construction as ``tests/test_causality_strict.py``'s
    ``test_decisions_ignore_every_bar_after_the_decision_bar``, adapted to
    this module's own decision/order-posting logic (limit price, side,
    size) instead of ``Order`` objects, since this simulator does not go
    through ``Context``/``Strategy.on_bar``. Two OPPOSITE tampers (x3 vs
    /3 on OHLC from the cut) so a leak is forced to show as a diverging
    decision, not left to chance.
    """
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions_at(frame):
        target, scale, max_lev = compute_v4_signal(frame)
        n = len(frame)
        closes = frame["close"].to_numpy(dtype=float)
        lev = max(market.leverage, 1e-9)
        lo_frac = -1.0 if market.allow_short else 0.0
        broker = PaperBroker(market=replace(market, fee_rate=MAKER_ENTRY), start_balance=10_000.0)
        out = {}
        # Replay a plain, no-fill-model version: just the decision (side, limit
        # price, size) computed at each requested bar, from a flat broker (this
        # probe checks the DECISION function, not fill outcomes -- fills of a
        # bar strictly after the decision bar are explicitly allowed to differ,
        # that is the whole point of the model).
        for i in bars:
            if i < 0 or i >= n:
                continue
            c = closes[i]
            lev_frac = min(1.0, max(lo_frac, target[i] / lev))
            desired_qty = (math.copysign(broker._max_qty(c) * abs(lev_frac), lev_frac)  # noqa: SLF001
                           if lev_frac != 0.0 else 0.0)
            raw_delta = desired_qty - broker.pos
            u = min(1.0, max(0.0, scale[i] / max_lev))
            offset = config.aggressive_frac + (config.passive_frac - config.aggressive_frac) * (1.0 - u)
            side = "BUY" if raw_delta > 0 else "SELL"
            limit_price = c * (1.0 - offset) if side == "BUY" else c * (1.0 + offset)
            out[i] = (side, round(abs(raw_delta), 8), round(limit_price, 6))
        return out

    a, b = decisions_at(up), decisions_at(down)
    ok = True
    for bar in bars:
        if bar not in a or bar not in b:
            continue
        if a[bar] != b[bar]:
            ok = False
            print(f"  CAUSALITY VIOLATION at bar {bar}: {a[bar]} vs {b[bar]}")
    return ok


# ============================================================ regime-flip-to-flat check
def find_flip_to_flat_events(target: np.ndarray, index: pd.DatetimeIndex) -> list[int]:
    """Bars where v4's own target signal drops to (near) zero from nonzero -- de-risking events."""
    out = []
    for i in range(1, len(target)):
        if abs(target[i - 1]) > 0.05 and abs(target[i]) < 1e-9:
            out.append(i)
    return out


def crash_lag_check(df: pd.DataFrame, market: MarketSpec, config: LimitFillConfig,
                     start=None, end=None) -> list[dict]:
    """For every flip-to-flat event, bars until FULLY flat (pos == 0) vs the 1-bar taker baseline.

    Reconstructs the position path from ``result.fills`` (not just "time to
    first partial fill", which would understate the lag whenever an order
    only partially fills on its first touched bar) and reports, for each
    event, the first bar strictly after the decision at which the running
    position is back to (within float tolerance of) zero.
    """
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    prefix = prefix_bars(df, lo, V4_WARMUP)
    frame = df.iloc[lo - prefix: hi]
    target, scale, max_lev = compute_v4_signal(frame)
    events = find_flip_to_flat_events(target, frame.index)
    events = [e for e in events if e >= prefix]  # only events inside the measured window

    result, stats = run_limit_backtest(frame, market, 1_000.0, config, trade_start=prefix,
                                        target=target, scale=scale, max_lev_signal=max_lev)

    # running position at each bar, from the fill tape (not exposed by BacktestResult directly)
    n = len(frame)
    pos_at_bar = np.zeros(n)
    pos = 0.0
    fi = 0
    fills = result.fills
    for i in range(n):
        ts = frame.index[i]
        while fi < len(fills) and fills[fi].ts == ts:
            f = fills[fi]
            pos += f.qty if f.side.name == "BUY" else -f.qty
            fi += 1
        pos_at_bar[i] = pos

    rows = []
    for e in events:
        decision_ts = frame.index[e]
        lag_bars = None
        for j in range(e + 1, n):
            if abs(pos_at_bar[j]) < 1e-9:
                lag_bars = j - e
                break
        rows.append({"event_ts": str(decision_ts), "novel_lag_bars": lag_bars,
                     "baseline_lag_bars": 1})
    return rows


# ============================================================ data loading (discipline-checked)
def load_working_frame() -> pd.DataFrame:
    """Full committed CSV, immediately cut to inner-train+inner-validation ONLY.

    Every function below is handed this frame (or a sub-slice of it) --
    never the raw load -- so no bar dated OOS_START or later can reach any
    computation, print, or report in this module.
    """
    btc_full = load_ohlcv_csv(ROOT / "data" / "btcusd_spot_5m.csv.gz")
    btc = btc_full.loc[:pd.Timestamp(INNER_VAL[1], tz="UTC")]
    del btc_full
    assert btc.index[-1] < pd.Timestamp(OOS_START, tz="UTC"), (
        "data discipline violated: a holdout bar leaked into the working frame")
    return btc


SPOT = MarketSpec.spot()
FUT = MarketSpec.futures(leverage=5.0)

# Sweep grid. Offsets are calibrated to the fee band they are trying to
# earn back: Bitstamp entry tier maker-vs-taker is only 10bps (0.40% ->
# 0.30%), so an offset materially larger than that cannot pay for itself
# even at a 100% fill rate -- the grid brackets that band rather than an
# arbitrary range.
GRID_PASSIVE_BPS = (5.0, 10.0, 15.0)
GRID_PATIENCE = (4, 8, 16)
GRID_GAMMA = (0.5, 1.0, 2.0)
GRID_AGGRESSIVE_BPS = (1.0, 3.0)  # aggressive_offset_bps sensitivity at the chosen center


def sweep_main_grid(df: pd.DataFrame, aggressive_bps: float = 3.0) -> list[dict]:
    """Phase 1: passive_offset x patience x gamma, inner-train, spot, entry-tier fee."""
    print("=" * 100 + f"\nMAIN GRID -- inner-train, spot, entry-tier fees (aggressive={aggressive_bps}bp)"
          f"\n{len(GRID_PASSIVE_BPS) * len(GRID_PATIENCE) * len(GRID_GAMMA)} configurations\n" + "=" * 100)
    rows = []
    base_final = run_taker_baseline(df, *INNER_TRAIN, SPOT, TAKER_ENTRY, data_label="real").final_balance
    base_fees = run_taker_baseline(df, *INNER_TRAIN, SPOT, TAKER_ENTRY, data_label="real").fees_paid
    for passive in GRID_PASSIVE_BPS:
        for patience in GRID_PATIENCE:
            for gamma in GRID_GAMMA:
                cfg = LimitFillConfig(passive_offset_bps=passive, aggressive_offset_bps=aggressive_bps,
                                       patience_bars=patience, fill_gamma=gamma)
                res, stats = run_limit_period(df, *INNER_TRAIN, SPOT, cfg, data_label="real")
                row = summarize(cfg.tag(), res, stats, baseline_final=base_final, baseline_fees=base_fees)
                rows.append(row)
                print_row(row)
    print(f"\nbaseline (taker-only v4 @ {TAKER_ENTRY:.2%}): ${base_final:,.0f}  fees=${base_fees:,.0f}")
    return rows


def sweep_aggressive_sensitivity(df: pd.DataFrame, passive_bps: float, patience: int, gamma: float) -> list[dict]:
    print("=" * 100 + f"\nAGGRESSIVE-OFFSET SENSITIVITY at passive={passive_bps}bp pat={patience} g={gamma}\n"
          + "=" * 100)
    rows = []
    for aggr in GRID_AGGRESSIVE_BPS:
        cfg = LimitFillConfig(passive_offset_bps=passive_bps, aggressive_offset_bps=aggr,
                               patience_bars=patience, fill_gamma=gamma)
        res, stats = run_limit_period(df, *INNER_TRAIN, SPOT, cfg, data_label="real")
        row = summarize(cfg.tag(), res, stats)
        rows.append(row)
        print_row(row)
    return rows


def run_ablations(df: pd.DataFrame, passive_bps: float, aggressive_bps: float, gamma: float) -> dict:
    """Requirement 4: literature-grounded fill-prob model vs a flat-probability ablation,
    AND adaptive-posting vs fixed-offset ablation, same touch logic throughout."""
    print("=" * 100 + "\nABLATION A -- flat (non-adaptive) fill probability vs literature-grounded pen**gamma\n"
          + "=" * 100)
    rows_a = []
    for patience in GRID_PATIENCE:
        # the literature-grounded arm, same patience
        cfg_lit = LimitFillConfig(passive_offset_bps=passive_bps, aggressive_offset_bps=aggressive_bps,
                                   patience_bars=patience, fill_gamma=gamma)
        res, stats = run_limit_period(df, *INNER_TRAIN, SPOT, cfg_lit, data_label="real")
        row = summarize(f"LIT  {cfg_lit.tag()}", res, stats)
        rows_a.append(row)
        print_row(row)
        for flat_p in (0.3, 0.5, 0.7):
            cfg_flat = LimitFillConfig(passive_offset_bps=passive_bps, aggressive_offset_bps=aggressive_bps,
                                        patience_bars=patience, ablation_flat_prob=flat_p)
            res, stats = run_limit_period(df, *INNER_TRAIN, SPOT, cfg_flat, data_label="real")
            row = summarize(f"FLAT {cfg_flat.tag()}", res, stats)
            rows_a.append(row)
            print_row(row)

    print("\n" + "=" * 100 + "\nABLATION B -- adaptive posting offset vs fixed (always-passive) offset\n"
          + "=" * 100)
    rows_b = []
    for patience in GRID_PATIENCE:
        cfg_adapt = LimitFillConfig(passive_offset_bps=passive_bps, aggressive_offset_bps=aggressive_bps,
                                     patience_bars=patience, fill_gamma=gamma, adaptive_posting=True)
        res, stats = run_limit_period(df, *INNER_TRAIN, SPOT, cfg_adapt, data_label="real")
        row = summarize(f"ADAPT {cfg_adapt.tag()}", res, stats)
        rows_b.append(row)
        print_row(row)
        cfg_fixed = LimitFillConfig(passive_offset_bps=passive_bps, aggressive_offset_bps=aggressive_bps,
                                     patience_bars=patience, fill_gamma=gamma, adaptive_posting=False)
        res, stats = run_limit_period(df, *INNER_TRAIN, SPOT, cfg_fixed, data_label="real")
        row = summarize(f"FIXED {cfg_fixed.tag()}", res, stats)
        rows_b.append(row)
        print_row(row)
    return {"flat_vs_literature": rows_a, "adaptive_vs_fixed": rows_b}


def validate_final(df: pd.DataFrame, cfgs: list[LimitFillConfig]) -> None:
    """Chosen config + neighbors, both markets, both inner windows, vs baseline, both fee tiers."""
    print("=" * 100 + "\nFINAL VALIDATION -- chosen config + neighbors x {spot, futures} x "
          "{inner-train, inner-validation} x {entry, top fee tier}\n" + "=" * 100)
    for market, mname in ((SPOT, "spot"), (FUT, "futures_5x")):
        for start, end, wname in (INNER_TRAIN + ("inner-train",), INNER_VAL + ("inner-val",)):
            for tier_taker, tier_maker, tname in ((TAKER_ENTRY, MAKER_ENTRY, "entry"),
                                                   (TAKER_TOP, MAKER_TOP, "top")):
                base = run_taker_baseline(df, start, end, market, tier_taker, data_label="real")
                print(f"\n-- {mname} / {wname} / {tname} tier --")
                base_result = run_period(KellyRegimeV4(), df, start, end,
                                          market=replace(market, fee_rate=tier_taker),
                                          start_balance=1_000.0, data_label="real")
                print_row(summarize(f"BASELINE taker-only @ {tier_taker:.2%}", base_result))
                for cfg in cfgs:
                    res, stats = run_limit_period(df, start, end, market, cfg, data_label="real",
                                                   fee_maker=tier_maker, fee_taker=tier_taker)
                    row = summarize(cfg.tag(), res, stats, baseline_final=base.final_balance,
                                     baseline_fees=base.fees_paid)
                    print_row(row)


def falsification_eth(config: LimitFillConfig) -> None:
    print("=" * 100 + "\nFALSIFICATION (a): ETH, entirely pre-2020 (2016-03-09 -> 2019-12-31)\n" + "=" * 100)
    eth = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    assert eth.index[-1] < pd.Timestamp("2020-01-01", tz="UTC"), "ETH file must be pre-2020"
    print(f"{len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}")
    base = run_taker_baseline(eth, None, None, SPOT, TAKER_ENTRY, data_label="ETH")
    base_result = run_period(KellyRegimeV4(), eth, None, None,
                              market=replace(SPOT, fee_rate=TAKER_ENTRY),
                              start_balance=1_000.0, data_label="ETH")
    print_row(summarize("BASELINE taker-only", base_result))
    res, stats = run_limit_period(eth, None, None, SPOT, config, data_label="ETH")
    row = summarize(config.tag(), res, stats, baseline_final=base.final_balance, baseline_fees=base.fees_paid)
    print_row(row)


def falsification_btc_control(df: pd.DataFrame, config: LimitFillConfig) -> None:
    print("=" * 100 + f"\nFALSIFICATION (b): BTC-only pre-2020 control {BTC_CONTROL}\n" + "=" * 100)
    base = run_taker_baseline(df, *BTC_CONTROL, SPOT, TAKER_ENTRY, data_label="real")
    base_result = run_period(KellyRegimeV4(), df, *BTC_CONTROL,
                              market=replace(SPOT, fee_rate=TAKER_ENTRY),
                              start_balance=1_000.0, data_label="real")
    print_row(summarize("BASELINE taker-only", base_result))
    res, stats = run_limit_period(df, *BTC_CONTROL, SPOT, config, data_label="real")
    row = summarize(config.tag(), res, stats, baseline_final=base.final_balance, baseline_fees=base.fees_paid)
    print_row(row)


def falsification_crash_lag(df: pd.DataFrame, config: LimitFillConfig) -> None:
    print("=" * 100 + "\nFALSIFICATION (c): regime-flip-to-flat lag, inner-train+inner-validation\n" + "=" * 100)
    rows = crash_lag_check(df, SPOT, config, start=INNER_TRAIN[0], end=INNER_VAL[1])
    if not rows:
        print("no flip-to-flat events in this window")
        return
    for r in rows:
        print(f"  {r['event_ts']}  novel_lag_bars={r['novel_lag_bars']}  "
              f"baseline_lag_bars={r['baseline_lag_bars']}")
    lags = [r["novel_lag_bars"] for r in rows if r["novel_lag_bars"] is not None]
    print(f"\n{len(rows)} flip-to-flat events; novel model lag: "
          f"mean={np.mean(lags):.1f} max={max(lags)} bars "
          f"(baseline is always 1 bar / 5 minutes)")


def main() -> None:
    df = load_working_frame()
    print(f"{len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (data: real)")
    print(f"restricted to inner-train {INNER_TRAIN} and inner-validation {INNER_VAL}\n")

    print("=" * 78 + "\nCAUSALITY PROBE\n" + "=" * 78)
    probe_df = df.iloc[-60_000:]
    ok = causality_probe(probe_df, LimitFillConfig(), SPOT)
    print(f"causality probe: {'PASS' if ok else 'FAIL'}\n")

    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    if choice in ("sweep", "all"):
        sweep_main_grid(df)

    if choice in ("aggr", "all"):
        sweep_aggressive_sensitivity(df, passive_bps=10.0, patience=8, gamma=1.0)

    if choice in ("ablation", "all"):
        run_ablations(df, passive_bps=10.0, aggressive_bps=3.0, gamma=1.0)

    if choice in ("validate", "all"):
        chosen = LimitFillConfig(passive_offset_bps=10.0, aggressive_offset_bps=3.0,
                                  patience_bars=8, fill_gamma=1.0)
        neighbors = [
            LimitFillConfig(passive_offset_bps=5.0, aggressive_offset_bps=3.0, patience_bars=8, fill_gamma=1.0),
            LimitFillConfig(passive_offset_bps=15.0, aggressive_offset_bps=3.0, patience_bars=8, fill_gamma=1.0),
        ]
        validate_final(df, [chosen] + neighbors)

    if choice in ("falsify", "all"):
        chosen = LimitFillConfig(passive_offset_bps=10.0, aggressive_offset_bps=3.0,
                                  patience_bars=8, fill_gamma=1.0)
        falsification_eth(chosen)
        falsification_btc_control(df, chosen)
        falsification_crash_lag(df, chosen)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[{time.time() - t0:.0f}s]")
    print(f"\n[{time.time() - t0:.0f}s]")
