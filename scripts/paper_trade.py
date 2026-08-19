#!/usr/bin/env python
"""Forward paper-trading recorder for kelly_regime_v4 on Bitstamp (B-06).

Run once per closed 5-minute Bitstamp candle. Unlike ``live_bot.py`` this
needs **no exchange credentials and places no real order, ever** — it
maintains its own persisted virtual account (starting paper equity, BTC
position 0) in JSON, simulates the fill through the exact same
``PaperBroker`` fee/rebalance code the backtest engine uses, and appends
one row per decision to an append-only CSV log. A parallel
``buy_and_hold`` paper account is recorded from the same candle fetch, as
a benchmark to compare against once enough history accumulates.

Why this exists: R-29's holdout-consultation count put every Sharpe claim
in this repo's 2023+ dataset past the point deflated Sharpe can support
(docs/LEDGER.md). Forward paper trading is the only uncontaminated record
this project can still generate — this script is B-06 in the ledger
backlog, "the highest-value item on merit" since R-29. It reads nothing
from the committed backtest dataset and does not touch the 2023+ holdout
counter; it only ever calls the live Bitstamp public API.

    # first run: creates reports/paper_trading/*_state.json and *.csv,
    # prints an INCEPTION line, exits 0
    python scripts/paper_trade.py

    # every subsequent run, once per closed 5m candle (cron/systemd —
    # see docs/LIVE.md): advances the paper account by exactly one
    # decision, or exits cleanly with "nothing to do" if the candle this
    # process would act on has already been recorded
    python scripts/paper_trade.py

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

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.exchanges.bitstamp import BitstampSpot  # noqa: E402
from tradebot.live import LiveAccount, compute_signal  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

STATE_DIR = ROOT / "reports" / "paper_trading"
PAPER_START_EQUITY = 1000.0
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
    ap.add_argument("--strategy", default="kelly_regime_v4")
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
                    help="skip the parallel buy_and_hold paper benchmark")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    strategy_cls_warmup = get_strategy(args.strategy).warmup
    market = MarketSpec.spot(fee_rate=args.taker_fee)
    bars = strategy_cls_warmup + args.warmup_slack

    # Public candles need no credentials. No key/secret is ever read or
    # passed here, and this module never calls place_market_order or any
    # signed endpoint - paper-only by construction, not configuration.
    exchange = BitstampSpot()
    print(f"bitstamp {args.symbol} · paper-trading {args.strategy} · "
          f"taker fee {args.taker_fee:.2%} · fetching {bars} bars "
          f"(warmup {strategy_cls_warmup} + slack {args.warmup_slack})",
          file=sys.stderr)
    try:
        candles = exchange.fetch_history(args.symbol, bars=bars,
                                         minutes=args.minutes,
                                         progress=not args.quiet)
    except urllib.error.URLError as exc:
        print(f"could not reach bitstamp: {exc.reason}\n"
              "check outbound HTTPS to bitstamp.net from this machine "
              "(sandboxes and CI runners usually block it)", file=sys.stderr)
        return 2

    if len(candles) < strategy_cls_warmup:
        print(f"insufficient history: {len(candles)}/{strategy_cls_warmup} "
              "bars - bitstamp has not returned enough candles yet",
              file=sys.stderr)
        return 2

    prefix = f"{args.strategy}_bitstamp"
    run_recorder(args.strategy, candles, market,
                state_path=state_dir / f"{prefix}_state.json",
                csv_path=state_dir / f"{prefix}.csv",
                start_equity=args.paper_equity, verbose=not args.quiet)

    if not args.no_benchmark:
        bh_prefix = "buy_and_hold_bitstamp"
        run_recorder("buy_and_hold", candles, market,
                    state_path=state_dir / f"{bh_prefix}_state.json",
                    csv_path=state_dir / f"{bh_prefix}.csv",
                    start_equity=args.paper_equity, verbose=not args.quiet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
