#!/usr/bin/env python
"""Forward paper-trading recorder for the promoted strategy family on
Bitstamp (B-06).

Run once per closed 5-minute Bitstamp candle. Unlike ``live_bot.py`` this
needs **no exchange credentials and places no real order, ever** — for
each strategy it records, it maintains its own persisted virtual account
(starting paper equity, BTC position 0) in JSON, simulates the fill
through the exact same ``PaperBroker`` fee/rebalance code the backtest
engine uses, and appends one row per decision to that strategy's own
append-only CSV log.

**Behavior change (this version): multi-strategy by default.** Earlier
versions of this script only ever recorded two hardcoded strategies
(``kelly_regime_v4`` and a ``buy_and_hold`` benchmark). It now accepts a
``--strategies`` list and, when that flag is omitted, records the full
default family below instead — every registered strategy docs/LEDGER.md
section A marks PROMOTED or REGISTERED off the main incumbent lineage,
plus the ``buy_and_hold`` benchmark — so running with no flags at all
(e.g. from cron or the ``paper_trading.yml`` GitHub Actions workflow)
grows a comparable forward record for the whole family, not just one
strategy. Every strategy still gets its own state/CSV files, keyed by its
own name (see ``STATE_DIR`` / the ``<strategy>_bitstamp`` prefix below);
nothing is shared between strategies except the one candle fetch per run.

Why this exists: R-29's holdout-consultation count put every Sharpe claim
in this repo's 2023+ dataset past the point deflated Sharpe can support
(docs/LEDGER.md). Forward paper trading is the only uncontaminated record
this project can still generate — this script is B-06 in the ledger
backlog, "the highest-value item on merit" since R-29. It reads nothing
from the committed backtest dataset and does not touch the 2023+ holdout
counter; it only ever calls the live Bitstamp public API.

    # first run: creates reports/paper_trading/*_state.json and *.csv per
    # strategy, prints one INCEPTION line per strategy, exits 0
    python scripts/paper_trade.py

    # every subsequent run, once per closed 5m candle (cron/systemd/the
    # GitHub Actions workflow — see docs/LIVE.md): advances each paper
    # account by exactly one decision, or exits cleanly with "nothing to
    # do" for any strategy whose candle has already been recorded
    python scripts/paper_trade.py

    # record only specific strategies (space-separated registered names)
    python scripts/paper_trade.py --strategies kelly_regime_v4 buy_and_hold

**Honest limitations, stated up front (also in docs/LIVE.md):**

- **Fills at the observed candle's close, not the next open.** The
  backtest and ``live_bot.py`` both use "decide on close, fill at next
  open" — the correct contract when a process is running continuously.
  This recorder runs once per invocation with no book to sit on between
  runs, so it fills at the *current* closed candle's close: the price
  observed when the recorder happens to run, not a guaranteed next-open
  print. Read every recorded fill as an approximation of that ideal, not
  as the ideal itself.
- **Single venue, no order book, no slippage model.** Bitstamp spot only;
  the fill price is the last trade printed on the OHLC candle, with no
  book depth, latency or partial-fill model.
- **Real fee tier.** Uses Bitstamp's 0.40% entry taker fee by default
  (``--taker-fee``), per this project's standing rule to always use real
  costs rather than the 0.10% assumption the headline comparison table
  uses.
- **Paper only, by construction, not by configuration.** This module
  never imports ``place_market_order`` or any signed Bitstamp endpoint —
  there is no ``--live`` flag and no code path that could send a real
  order even with credentials in the environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import REBALANCE_DEADBAND, MarketSpec, PaperBroker  # noqa: E402
from tradebot.exchanges.bitstamp import BitstampSpot  # noqa: E402
from tradebot.live import LiveAccount, compute_signal  # noqa: E402
from tradebot.orders import Order  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402

STATE_DIR = ROOT / "reports" / "paper_trading"
PAPER_START_EQUITY = 1000.0

# The default forward-recording set when --strategies is omitted: the
# main incumbent lineage (docs/LEDGER.md section A) — kelly_regime and
# its four registered variants, PROMOTED or not — plus the buy_and_hold
# benchmark to compare against. Confirmed against docs/LEDGER.md section
# A and src/tradebot/strategies/*.py's `name = ...` class attributes
# (kelly_regime_ev_fast is its own registered class in kelly_regime_ev.py,
# not a parametrization passed at runtime). Deliberately NOT "only the
# PROMOTED ones" — kelly_regime_v2 and the two kelly_regime_ev variants
# are NOT PROMOTED/are plain REGISTERED per the ledger, but a forward
# record of the variants the project seriously considered is exactly the
# kind of uncontaminated evidence B-06 exists to generate, so they are
# tracked too. Every other registered strategy (the NEGATIVE-verdict
# baselines and microstructure experiments in section A) is deliberately
# excluded — recording those would just be spending API calls on
# strategies this project has already rejected.
DEFAULT_STRATEGIES = [
    "kelly_regime_v4",
    "kelly_regime_v3",
    "kelly_regime_v2",
    "kelly_regime",
    "kelly_regime_ev",
    "kelly_regime_ev_fast",
    "buy_and_hold",
]
CSV_COLUMNS = [
    "timestamp", "candle_close_price", "prior_target", "new_target",
    "trade_qty", "fee_paid", "position_after", "cash_after",
    "equity_after", "reason",
]


@dataclass
class PaperState:
    """The recorder's own virtual account — never a real exchange balance."""

    cash: float
    pos: float
    entry: float
    dead: bool
    fees_paid: float
    last_candle_ts: str | None
    inception_ts: str | None
    inception_price: float | None
    n_runs: int

    @classmethod
    def fresh(cls, start_equity: float) -> "PaperState":
        return cls(cash=start_equity, pos=0.0, entry=0.0, dead=False,
                    fees_paid=0.0, last_candle_ts=None, inception_ts=None,
                    inception_price=None, n_runs=0)


def inception_catchup_target(strategy: Strategy, candles: pd.DataFrame) -> float | None:
    """The strategy's raw desired stance as of the latest bar, read
    directly from ``prepare()`` - bypassing ``on_bar``'s own
    emit-only-on-change gate.

    18 of this project's 23 registered strategies (the whole
    ``kelly_regime`` family among them) share one convention verbatim::

        t = float(ctx.bar["target"]); prev = float(ctx.prev["target"]) ...
        if abs(t - prev) > 1e-9: ctx.order_target(t)  # (or order_notional)

    That gate compares the CURRENT bar's precomputed target to the
    PREVIOUS bar's - both pure functions of price history, not of
    account state - so ``compute_signal`` on a freshly cold-started
    account never emits an order at all if the target has simply been
    latched flat for the whole fetched window (exactly what happened the
    first time this recorder ran live against ``kelly_regime_v4`` here:
    the vote had been pinned since before the 500-bar warmup slack even
    began). A paper account that silently sits flat forever while
    believing it tracks the strategy would defeat the entire point of
    B-06, and the identical gap sits in ``bot.py``/``live_bot.py`` too -
    it just cannot be reached from here without touching those files.

    Used at inception, and — since R-78 — on any later run where the
    strategy's desired stance has drifted away from the paper account's
    actual one by more than the broker would ignore (see
    :func:`level_resync_order`). It never overrides a genuine signal:
    both call sites only reach it when ``compute_signal`` emitted nothing
    at all. On spot (``market.leverage == 1``) ``order_notional(t)`` and
    ``order_target(t)`` are the same order, so this is correct
    regardless of which one the strategy calls.

    Returns ``None`` for strategies with no ``target`` column (the
    remaining ~5: e.g. ``macd_rsi``, ``rsi_reversion``, decide from
    ``ctx.position`` directly rather than a precomputed column) - those
    have nothing to read ahead of time, so inception starts flat for
    them, same as it does in ``bot.py`` today.
    """
    prepared = strategy.prepare(candles.copy())
    if "target" not in prepared.columns:
        return None
    return float(prepared["target"].to_numpy()[-1])


def level_resync_order(strategy: Strategy, candles: pd.DataFrame,
                       prior_target: float, market: MarketSpec) -> Order | None:
    """An order that re-syncs the paper account to the strategy's CURRENT
    desired stance, or ``None`` when it is already there.

    **Why this exists (R-78).** The change gate
    :func:`inception_catchup_target` documents is *edge-triggered*: it
    compares bar ``i``'s precomputed target to bar ``i-1``'s. That is
    correct only when something asks the strategy on **every** closed bar,
    which is what the backtest engine and a 5-minute ``bot.py`` loop do.
    This recorder does not: one invocation advances by exactly the newest
    closed candle, so on any schedule slower than the bar interval the
    intermediate candles are never presented to ``on_bar`` at all. A
    target change that lands on one of them is then **permanently
    invisible** — by the next invocation the target has stopped changing,
    the edge gate is silent again, and the account holds a stance the
    strategy abandoned hours ago.

    R-78 measured both halves of that on the record this script had
    already produced: a realized median gap of 50-85 minutes against the
    5-minute bar (so ~1 candle in 10-17 gets a decision), and, on
    2017-2022 data, exactly ``1/k`` of ``kelly_regime_v4``'s 811 target
    changes surviving a 1-in-``k`` decision grid — 10.0% at the realized
    cadence, i.e. ~13 of its ~135 rebalances a year. Priced through the
    real engine, the same cadence cost 0.31-0.41 Sharpe and 26-36% of
    final balance against a full-cadence run. The record was not a
    slightly-delayed ``kelly_regime_v4``; it was a different, much lazier
    strategy wearing its name.

    **The fix is level-triggered, not edge-triggered**, so it is immune to
    the schedule: ask what stance the strategy wants *now* and compare it
    to what the account actually holds *now*. Whether a candle was seen or
    missed stops mattering — only the current desired level does.

    Deliberately conservative in two ways:

    - It is consulted **only when ``compute_signal`` emitted nothing**, so
      a genuine bar-over-bar signal is always used as-is and this can
      never contradict one.
    - It emits only when the broker would actually act on the difference —
      ``broker.REBALANCE_DEADBAND`` of max notional for a same-sign
      adjustment, any move to flat, and any sign flip (the broker always
      executes those two). Below that the broker would ignore the order
      anyway, so emitting one would add a misleading row rather than a
      trade.

    Returns ``None`` for strategies with no ``target`` column (the ~5 that
    decide from ``ctx.position`` directly), which keep the pre-R-78
    behaviour exactly.
    """
    desired = inception_catchup_target(strategy, candles)
    if desired is None:
        return None
    # Fraction of MAX notional, matching Order(target=...)'s own units and
    # the clamp PaperBroker applies. On spot (leverage 1, no short) this is
    # [0, 1]; the recorder only runs spot today, but keep it general.
    lo = -1.0 if market.allow_short else 0.0
    effective = min(max(desired / max(market.leverage, 1e-9), lo), 1.0)
    delta = effective - prior_target
    if abs(delta) <= 1e-9:
        return None
    goes_flat = effective == 0.0 and prior_target != 0.0
    flips = effective * prior_target < 0.0
    if not (goes_flat or flips or abs(delta) > REBALANCE_DEADBAND):
        return None
    return Order(target=desired)


def load_state(path: Path, start_equity: float) -> tuple[PaperState, bool]:
    """Return (state, is_fresh). Fresh means no prior file — inception."""
    if not path.exists():
        return PaperState.fresh(start_equity), True
    data = json.loads(path.read_text())
    return PaperState(**data), False


def save_state(path: Path, state: PaperState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_recorder(strategy_name: str, candles: pd.DataFrame, market: MarketSpec,
                 state_path: Path, csv_path: Path,
                 start_equity: float = PAPER_START_EQUITY,
                 verbose: bool = True) -> str:
    """Advance one paper-trading decision for ``strategy_name`` on ``candles``.

    ``candles`` must be the same OHLCV shape ``compute_signal`` expects —
    closed bars only, oldest first. Idempotent on the candle timestamp:
    calling this twice with the same latest candle is a clean no-op the
    second time (detected from the persisted state, not from the CSV).

    Returns one of "inception", "traded", "unchanged",
    "skipped (no new candle)".
    """
    if candles.empty:
        raise ValueError("no candles to act on")
    latest_ts = candles.index[-1]
    latest_close = float(candles["close"].iloc[-1])

    state, fresh = load_state(state_path, start_equity)
    if fresh and verbose:
        print(f"[{strategy_name}] INCEPTION - no prior state at {state_path}, "
              f"starting a fresh ${start_equity:,.2f} paper account as of "
              f"{latest_ts.isoformat()} @ ${latest_close:,.2f}. This is the "
              "first row of B-06's forward paper-trading record.",
              file=sys.stderr)

    if state.last_candle_ts is not None:
        prior_ts = pd.Timestamp(state.last_candle_ts)
        if prior_ts == latest_ts:
            if verbose:
                print(f"[{strategy_name}] no new closed candle since last run "
                      f"(ts={latest_ts.isoformat()}) - nothing to do",
                      file=sys.stderr)
            return "skipped (no new candle)"
        if prior_ts > latest_ts:
            raise RuntimeError(
                f"{strategy_name}: fetched candle {latest_ts} is OLDER than "
                f"the last recorded one {prior_ts} - venue clock or state "
                "file looks wrong; refusing to record a row out of order")

    strategy = get_strategy(strategy_name)
    # Re-hydrate the persisted paper account onto a PaperBroker so the fill
    # is executed by the exact same fee/rebalance code the backtest engine
    # uses (tradebot.broker.PaperBroker._execute_target / _transact) —
    # not a re-derived formula that could drift from it.
    broker = PaperBroker(market=market, start_balance=start_equity)
    broker.cash, broker.pos, broker.entry = state.cash, state.pos, state.entry
    broker.dead, broker.fees_paid = state.dead, state.fees_paid

    prior_equity = broker.equity(latest_close)
    prior_target = (broker.pos * latest_close / prior_equity) if prior_equity > 0 else 0.0

    account = LiveAccount(position=broker.pos, equity_quote=prior_equity, market=market)
    orders = compute_signal(strategy, candles, account)

    catchup_used = False
    resync_used = False
    if not orders:
        if fresh:
            # See inception_catchup_target(): most strategies only emit an
            # order when their target CHANGES bar over bar, so a cold-started
            # account that happens to catch a strategy mid-latch would
            # otherwise sit flat forever and never actually track it.
            catchup = inception_catchup_target(strategy, candles)
            if catchup is not None and abs(catchup) > 1e-9:
                orders = [Order(target=catchup)]
                catchup_used = True
        else:
            # See level_resync_order(): the same edge-triggered gate also
            # drops every target change that lands on a candle this
            # recorder's schedule skipped. R-78 measured that at ~90% of
            # them on the realized cadence, so a level comparison against
            # the account's actual stance runs on every later invocation.
            resync = level_resync_order(strategy, candles, prior_target, market)
            if resync is not None:
                orders = [resync]
                resync_used = True

    trade_qty = 0.0
    fee_paid = 0.0
    new_target = prior_target
    reason = f"target unchanged ({prior_target:.4f})"
    status = "unchanged"
    if orders and not broker.dead:
        order = orders[-1]  # every registered strategy emits at most one order/bar
        fills = broker.execute(order, latest_ts, latest_close)
        for fill in fills:
            trade_qty += fill.qty if fill.side.name == "BUY" else -fill.qty
            fee_paid += fill.fee
        if order.target is not None:
            new_target = float(order.target)
            if catchup_used:
                reason = (f"INCEPTION CATCH-UP target={order.target:.4f} "
                          "(on_bar's own change-gate emitted nothing; see "
                          "inception_catchup_target)")
            elif resync_used:
                reason = (f"LEVEL RESYNC target={order.target:.4f} from "
                          f"{prior_target:.4f} (on_bar's edge-triggered gate "
                          "emitted nothing; see level_resync_order, R-78)")
            else:
                reason = f"target={order.target:.4f}"
        else:
            reason = f"qty order ({order.side.value} {order.qty})"
        if fills:
            status = "traded"

    equity_after = broker.equity(latest_close)
    append_csv_row(csv_path, {
        "timestamp": latest_ts.isoformat(),
        "candle_close_price": f"{latest_close:.2f}",
        "prior_target": f"{prior_target:.6f}",
        "new_target": f"{new_target:.6f}",
        "trade_qty": f"{trade_qty:.8f}",
        "fee_paid": f"{fee_paid:.4f}",
        "position_after": f"{broker.pos:.8f}",
        "cash_after": f"{broker.cash:.4f}",
        "equity_after": f"{equity_after:.4f}",
        "reason": reason,
    })

    state.cash, state.pos, state.entry = broker.cash, broker.pos, broker.entry
    state.dead, state.fees_paid = broker.dead, broker.fees_paid
    state.last_candle_ts = latest_ts.isoformat()
    state.n_runs += 1
    if fresh:
        state.inception_ts = latest_ts.isoformat()
        state.inception_price = latest_close
    save_state(state_path, state)

    if verbose:
        print(f"[{strategy_name}] {latest_ts.isoformat()} close=${latest_close:,.2f} "
              f"target {prior_target:.3f} -> {new_target:.3f} "
              f"trade_qty={trade_qty:+.6f} fee=${fee_paid:.2f} "
              f"equity=${equity_after:,.2f} ({status})")
    return "inception" if fresh else status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="btcusd")
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES),
                    metavar="STRATEGY",
                    help="space-separated registered strategy names to "
                    "record this run (default: the full promoted/"
                    "registered kelly_regime family plus the buy_and_hold "
                    "benchmark - see DEFAULT_STRATEGIES above and "
                    "docs/LEDGER.md section A). Each gets its own "
                    "reports/paper_trading/<name>_bitstamp{_state.json,.csv} "
                    "pair, exactly like running this script once per name.")
    ap.add_argument("--taker-fee", type=float, default=0.004,
                    help="Bitstamp's real entry taker tier as a fraction "
                    "(default 0.004 = 0.40%%) - never the 0.10%% headline "
                    "assumption")
    ap.add_argument("--paper-equity", type=float, default=PAPER_START_EQUITY,
                    help="starting paper balance in USD; only used at "
                    "inception, ignored on every later run")
    ap.add_argument("--warmup-slack", type=int, default=500,
                    help="extra bars fetched beyond the strategy's warmup, "
                    "matching bot.py's BotConfig.warmup_slack")
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--no-benchmark", action="store_true",
                    help="drop 'buy_and_hold' from --strategies (whether "
                    "defaulted or given explicitly) if present; kept for "
                    "backward compatibility with the old two-strategy "
                    "default's --no-benchmark flag")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # De-dup while preserving order (a hand-written --strategies list
    # could repeat a name; running it twice in one invocation would just
    # double-append an identical row the second time within this process,
    # which run_recorder's own idempotency check would then reject as an
    # internal contradiction rather than silently accept).
    strategies = list(dict.fromkeys(args.strategies))
    if args.no_benchmark:
        strategies = [s for s in strategies if s != "buy_and_hold"]
    if not strategies:
        print("no strategies to record (--strategies resolved to an empty "
              "list, e.g. --no-benchmark with only buy_and_hold given)",
              file=sys.stderr)
        return 2

    state_dir = Path(args.state_dir)
    market = MarketSpec.spot(fee_rate=args.taker_fee)

    # Resolve every strategy up front - fails fast (KeyError, uncaught,
    # exit 1) on a typo'd/unregistered name before any network call, and
    # lets every strategy in this run share ONE fetched candle window
    # sized for the hungriest one, so they all decide from the exact same
    # point-in-time snapshot rather than N slightly-different fetches.
    strategy_warmups = {name: get_strategy(name).warmup for name in strategies}
    max_warmup = max(strategy_warmups.values())
    bars = max_warmup + args.warmup_slack

    # Public candles need no credentials. No key/secret is ever read or
    # passed here, and this module never calls place_market_order or any
    # signed endpoint - paper-only by construction, not configuration.
    exchange = BitstampSpot()
    print(f"bitstamp {args.symbol} · paper-trading {len(strategies)} "
          f"strategies ({', '.join(strategies)}) · taker fee "
          f"{args.taker_fee:.2%} · fetching {bars} bars (max warmup "
          f"{max_warmup} + slack {args.warmup_slack})", file=sys.stderr)
    try:
        candles = exchange.fetch_history(args.symbol, bars=bars,
                                         minutes=args.minutes,
                                         progress=not args.quiet)
    except urllib.error.URLError as exc:
        print(f"could not reach bitstamp: {exc.reason}\n"
              "check outbound HTTPS to bitstamp.net from this machine "
              "(sandboxes and CI runners usually block it)", file=sys.stderr)
        return 2

    ran_any = False
    for name in strategies:
        warmup = strategy_warmups[name]
        if len(candles) < warmup:
            print(f"[{name}] insufficient history: {len(candles)}/{warmup} "
                  "bars - bitstamp has not returned enough candles yet, "
                  "skipping this strategy this run", file=sys.stderr)
            continue
        prefix = f"{name}_bitstamp"
        run_recorder(name, candles, market,
                    state_path=state_dir / f"{prefix}_state.json",
                    csv_path=state_dir / f"{prefix}.csv",
                    start_equity=args.paper_equity, verbose=not args.quiet)
        ran_any = True

    if not ran_any:
        # Every requested strategy was skipped for insufficient history -
        # the one condition (besides the network) this function treats as
        # a soft, expected failure (exit 2) rather than a bug (uncaught
        # exception, exit 1). See docs/LIVE.md and paper_trading.yml,
        # which only treat exit 2 as "not a real job failure".
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
