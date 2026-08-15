#!/usr/bin/env python
"""Run one decision cycle against a real venue. Dry run unless told otherwise.

Credentials come from the environment, never from a file in the repo::

    export BITSTAMP_API_KEY=...
    export BITSTAMP_API_SECRET=...

    # decide and print, place nothing (default)
    python scripts/live_bot.py --venue bitstamp --symbol btcusd

    # same, but actually send the order
    python scripts/live_bot.py --venue bitstamp --symbol btcusd --live

Public candles need no credentials, so ``--dry-run`` works with none set:
useful for checking that the strategy agrees with the backtest on live
data before an account is involved at all.

Call this once per closed candle from cron/systemd. It is stateless - the
position lives in the exchange balance - so a missed run or a restart
costs nothing but the missed rebalance.

**Fees.** Every published result assumes a 0.10% taker fee. Bitstamp's
entry tier is 0.40%. Pass ``--taker-fee`` with your real tier: it feeds
the strategy's own fee awareness, so it is not cosmetic.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.bot import BotConfig, step  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

VENUES = {
    "bitstamp": ("BITSTAMP_API_KEY", "BITSTAMP_API_SECRET", "btcusd", 0.004),
    "binance": ("BINANCE_API_KEY", "BINANCE_API_SECRET", "BTCUSDT", 0.001),
}


def build_exchange(venue: str, live: bool, taker_fee: float | None):
    key_var, secret_var, _, default_fee = VENUES[venue]
    api_key = os.environ.get(key_var, "")
    api_secret = os.environ.get(secret_var, "")
    if live and not (api_key and api_secret):
        raise SystemExit(
            f"--live needs {key_var} and {secret_var} in the environment "
            "(the key must have the Trade permission)")

    if venue == "bitstamp":
        from tradebot.exchanges.bitstamp import BitstampSpot

        ex = BitstampSpot(api_key=api_key, api_secret=api_secret, dry_run=not live)
    else:
        from tradebot.exchanges.binance import BinanceSpot

        ex = BinanceSpot(api_key=api_key, api_secret=api_secret, dry_run=not live)
    ex.taker_fee = default_fee if taker_fee is None else taker_fee
    return ex


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--venue", choices=sorted(VENUES), default="bitstamp")
    ap.add_argument("--symbol", default=None, help="default: the venue's BTC/USD pair")
    ap.add_argument("--strategy", default="kelly_regime_v4")
    ap.add_argument("--taker-fee", type=float, default=None,
                    help="your real tier as a fraction, e.g. 0.004 for 0.40%%")
    ap.add_argument("--min-notional", type=float, default=10.0)
    ap.add_argument("--min-rebalance", type=float, default=0.10,
                    help="skip trades that move less than this fraction of equity")
    ap.add_argument("--live", action="store_true",
                    help="actually place the order (default: dry run)")
    args = ap.parse_args()

    symbol = args.symbol or VENUES[args.venue][2]
    exchange = build_exchange(args.venue, args.live, args.taker_fee)
    strategy = get_strategy(args.strategy)
    config = BotConfig(symbol=symbol, strategy=args.strategy,
                       min_notional=args.min_notional,
                       min_rebalance=args.min_rebalance)

    mode = "LIVE - orders will be sent" if args.live else "dry run - nothing is sent"
    print(f"{args.venue} {symbol} · {args.strategy} · taker fee "
          f"{exchange.taker_fee:.2%} · {mode}", file=sys.stderr)

    try:
        result = step(exchange, config, strategy)
    except urllib.error.URLError as exc:
        # A blocked proxy, a firewall or an offline box all land here, and a
        # raw traceback makes it look like a bug in the bot.
        print(f"could not reach {args.venue}: {exc.reason}\n"
              "check outbound HTTPS to the venue from this machine "
              "(sandboxes and CI runners usually block it)", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # credential / venue-reported errors
        print(f"{exc}", file=sys.stderr)
        return 2

    print(f"target={result.target:.3f} current={result.current:.3f} "
          f"equity=${result.equity:,.2f} price=${result.price:,.2f} "
          f"-> {result.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
