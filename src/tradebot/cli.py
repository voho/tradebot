"""Command line interface: tradebot {run,list,fetch}."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tradebot",
        description="Paper-test and compare BTCUSD 5m trading strategies.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the strategy comparison matrix")
    p_run.add_argument("--data-dir", type=Path, default=Path("data"))
    p_run.add_argument("--out", type=Path, default=Path("reports"))
    p_run.add_argument("--balances", type=float, nargs="+", default=[1_000.0],
                       help="starting balances in USD (default: 1000; results "
                            "are proportional to capital, so one is usually enough)")
    p_run.add_argument("--markets", nargs="+", choices=["spot", "futures"],
                       default=["spot", "futures"])
    p_run.add_argument("--leverage", type=float, default=5.0,
                       help="futures leverage (default: 5)")
    p_run.add_argument("--strategies", nargs="+", default=None,
                       help="strategy names (default: all registered)")
    p_run.add_argument("--slippage-bps", type=float, default=0.0)
    p_run.add_argument("--spot-fee", type=float, default=0.001)
    p_run.add_argument("--futures-fee", type=float, default=0.0005)
    p_run.add_argument("--max-bars", type=int, default=None,
                       help="use only the most recent N bars")

    sub.add_parser("list", help="list registered strategies")

    p_new = sub.add_parser("new", help="scaffold a new strategy file")
    p_new.add_argument("name", help="strategy name, e.g. ema_trend (lowercase_with_underscores)")

    p_fetch = sub.add_parser("fetch", help="download real BTCUSDT 5m data from Binance")
    p_fetch.add_argument("--data-dir", type=Path, default=Path("data"))
    p_fetch.add_argument("--symbol", default="BTCUSDT")
    p_fetch.add_argument("--start", default="2020-01-01",
                         help="long default span so bull and bear regimes are covered")
    p_fetch.add_argument("--end", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "list":
        from tradebot.registry import available_strategies

        for name, cls in sorted(available_strategies().items()):
            doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
            print(f"{name:20s} {doc}")
        return 0

    if args.cmd == "new":
        from tradebot.scaffold import new_strategy

        new_strategy(args.name)
        return 0

    if args.cmd == "fetch":
        from tradebot.fetch import fetch_data

        fetch_data(args.data_dir, symbol=args.symbol, start=args.start, end=args.end)
        return 0

    from tradebot.run import RunConfig, run_matrix

    cfg = RunConfig(
        data_dir=args.data_dir,
        out_dir=args.out,
        balances=list(args.balances),
        markets=list(args.markets),
        leverage=args.leverage,
        spot_fee=args.spot_fee,
        futures_fee=args.futures_fee,
        slippage_bps=args.slippage_bps,
        strategies=args.strategies,
        max_bars=args.max_bars,
    )
    run_matrix(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
