"""Command-line interface.

    gtbot fetch    --exchange binance --days 365 --out data/btcusdt_5m.csv
    gtbot simulate --bars 150000 --seed 0 --out data/sim.csv
    gtbot backtest --data data/btcusdt_5m.csv --tier vip6
    gtbot walkforward --data data/btcusdt_5m.csv --folds 6
    gtbot paper    --data data/btcusdt_5m.csv          # replay paper session
    gtbot report   --data data/btcusdt_5m.csv --out report.md

``fetch`` is the bridge to real data; everything else runs identically on real
or simulated bars because both go through the same canonical schema.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data.schema import BTCUSD_5M, validate
from .data.synthetic import simulate
from .engine.backtest import run_backtest
from .engine.broker import FEE_TIERS, CostModel, ExecutionConfig
from .engine.paper import replay_paper
from .eval import metrics, stats
from .eval.account import DEFAULT_DEPOSIT, DEFAULT_LEVERAGE, DIRECTIONS, format_table, simulate_account
from .eval.report import render_report
from .eval.walkforward import run_walkforward
from .risk import RiskConfig
from .strategy import GameTheoreticStrategy, StrategyConfig

BPY = BTCUSD_5M.bars_per_year


#: Bars the online learner needs before its weights and edge estimate are
#: confident enough for the sizer to allocate.  Below this a run is mostly
#: warm-up and will report few or no trades — which is correct behaviour, not a
#: bug, but is confusing without a warning.
MIN_BARS_FOR_CONVERGENCE = 100_000


def _warn_if_short(bars: pd.DataFrame) -> None:
    if len(bars) < MIN_BARS_FOR_CONVERGENCE:
        print(
            f"note: {len(bars):,} bars is short for the online learner "
            f"(~{MIN_BARS_FOR_CONVERGENCE:,} recommended). Expect few or no trades: "
            "the sizer holds off until the edge estimate is statistically supported.\n",
            file=sys.stderr,
        )


def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"no such file: {path}")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    return validate(df, strict=False)


def _strategy(args) -> GameTheoreticStrategy:
    cost = CostModel.for_tier(args.tier)
    execution = ExecutionConfig(entry_mode=args.entry_mode, exit_mode=args.exit_mode, ttl_bars=1)
    cfg = StrategyConfig(
        horizon=args.horizon,
        entry_signal=args.entry_signal,
        max_hold=args.horizon,
        assumed_cost_bp=cost.round_trip_bp(execution),
    )
    # The risk layer clamps before the backtester's own clip, so a --max-leverage
    # above RiskConfig's default would otherwise be silently ignored and every
    # reported number would come from a 2x run.
    cfg.risk = replace(cfg.risk, max_leverage=args.max_leverage)
    return GameTheoreticStrategy(cfg)


def _execution(args) -> ExecutionConfig:
    return ExecutionConfig(entry_mode=args.entry_mode, exit_mode=args.exit_mode, ttl_bars=1)


# ----------------------------------------------------------------- commands
def cmd_fetch(args) -> None:
    from .data.exchanges import fetch_ohlcv

    df = fetch_ohlcv(args.exchange, symbol=args.symbol, interval=args.interval, days=args.days)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} bars to {out}")


def cmd_simulate(args) -> None:
    res = simulate(args.bars, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.bars.to_csv(out, index=False)
    print(f"wrote {len(res.bars):,} simulated bars to {out}")


def cmd_backtest(args) -> None:
    bars = _load(args.data) if args.data else validate(simulate(args.bars, seed=args.seed).bars)
    _warn_if_short(bars)
    cost = CostModel.for_tier(args.tier)
    execution = _execution(args)
    res = run_backtest(
        bars, _strategy(args), costs=cost, execution=execution, max_leverage=args.max_leverage
    )
    m = metrics.compute(
        res.returns, res.equity, res.position, res.costs,
        bars_per_year=BPY, n_trades=res.n_trades,
    )
    print(f"bars           {len(bars):,}  ({m.years:.2f} years)")
    print(f"fee tier       {args.tier}  (round trip {cost.round_trip_bp(execution):.2f} bp)")
    print(f"execution      {args.entry_mode} in / {args.exit_mode} out")
    print(f"sharpe         {m.sharpe:+.2f}")
    print(f"CAGR           {m.cagr:+.2%}")
    print(f"volatility     {m.ann_vol:.2%}")
    print(f"max drawdown   {m.max_drawdown:.2%}")
    print(f"trades         {m.n_trades:,}  ({m.n_trades / max(m.years, 1e-9):.0f}/yr)")
    print(f"cost drag      {m.cost_drag_annual:.2%}/yr")
    print(f"deflated SR    {m.dsr:.3f}")

    # The account outcome is always reported: a Sharpe ratio does not tell you
    # how many dollars come back, nor whether the account survived the path.
    print()
    accounts = [
        simulate_account(
            bars, tier=args.tier, direction=d, sizing_mode=sm,
            leverage=args.leverage, deposit=args.deposit,
            config=_strategy(args).cfg, execution=execution,
        )
        for sm in ("robust", "fixed")
        for d in DIRECTIONS
    ]
    print(format_table(accounts))
    if any(a.liquidated for a in accounts):
        print("\n*** LIQUIDATED in at least one configuration ***")
    else:
        peak = max(a.margin_use for a in accounts)
        print(f"\nno liquidation; worst bar used {peak:.1%} of the distance to it")

    if args.out:
        res.to_frame().to_csv(args.out, index=False)
        print(f"equity curve -> {args.out}")


def cmd_walkforward(args) -> None:
    bars = _load(args.data) if args.data else validate(simulate(args.bars, seed=args.seed).bars)
    _warn_if_short(bars)
    cost = CostModel.for_tier(args.tier)
    wf = run_walkforward(
        bars,
        lambda: _strategy(args),
        n_folds=args.folds,
        costs=cost,
        execution=_execution(args),
        max_leverage=args.max_leverage,
    )
    print(wf.summary_frame().to_string(index=False))
    print(f"\npooled out-of-sample sharpe {wf.pooled.sharpe:+.2f}  CAGR {wf.pooled.cagr:+.2%}")
    st = stats.summarise(wf.pooled_returns, bars_per_year=BPY)
    print(f"bootstrap 95% CI [{st['sharpe_ci95'][0]:+.2f}, {st['sharpe_ci95'][1]:+.2f}]  "
          f"p(SR<=0) {st['bootstrap_p_value']:.4f}  NW t {st['newey_west_t']:+.2f}")


def cmd_paper(args) -> None:
    bars = _load(args.data) if args.data else validate(simulate(args.bars, seed=args.seed).bars)
    _warn_if_short(bars)
    session, broker = replay_paper(bars, _strategy(args), max_leverage=args.max_leverage)
    print(f"bars seen      {session.bars_seen:,}")
    print(f"decisions      {session.decisions:,}")
    print(f"orders         {session.orders:,}")
    print(f"final equity   {broker.equity():,.2f}")
    print(f"position       {broker.position():+.3f}")
    print(f"last signal    {session.last_signal:+.3f}")
    print(f"est. edge      {session.last_edge_bp:.2f} bp")


def cmd_report(args) -> None:
    bars = _load(args.data) if args.data else validate(simulate(args.bars, seed=args.seed).bars)
    _warn_if_short(bars)
    text = render_report(bars, tiers=list(FEE_TIERS), max_leverage=args.max_leverage,
                         leverage=args.leverage, deposit=args.deposit)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gtbot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--data", help="CSV/parquet of canonical bars; omit to use the simulator")
        sp.add_argument("--bars", type=int, default=150_000, help="simulated bars if --data absent")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--tier", default="vip6", choices=sorted(FEE_TIERS))
        sp.add_argument("--entry-mode", dest="entry_mode", default="taker", choices=["taker", "maker"])
        sp.add_argument("--exit-mode", dest="exit_mode", default="maker", choices=["taker", "maker"])
        sp.add_argument("--horizon", type=int, default=3)
        sp.add_argument("--entry-signal", dest="entry_signal", type=float, default=0.55)
        sp.add_argument("--max-leverage", dest="max_leverage", type=float, default=2.0)
        sp.add_argument("--leverage", type=float, default=DEFAULT_LEVERAGE,
                        help="leverage for the reported account outcome")
        sp.add_argument("--deposit", type=float, default=DEFAULT_DEPOSIT,
                        help="starting capital for the reported account outcome")

    f = sub.add_parser("fetch", help="download real bars from an exchange")
    f.add_argument("--exchange", default="binance", choices=["binance", "bybit", "okx", "coinbase"])
    f.add_argument("--symbol", default="BTCUSDT")
    f.add_argument("--interval", default="5m")
    f.add_argument("--days", type=int, default=365)
    f.add_argument("--out", default="data/btcusdt_5m.csv")
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("simulate", help="generate synthetic bars")
    s.add_argument("--bars", type=int, default=150_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/sim_5m.csv")
    s.set_defaults(func=cmd_simulate)

    b = sub.add_parser("backtest", help="run a single backtest")
    common(b)
    b.add_argument("--out", help="write the equity curve here")
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("walkforward", help="purged, embargoed walk-forward")
    common(w)
    w.add_argument("--folds", type=int, default=6)
    w.set_defaults(func=cmd_walkforward)

    pa = sub.add_parser("paper", help="replay bars through the paper-trading loop")
    common(pa)
    pa.set_defaults(func=cmd_paper)

    r = sub.add_parser("report", help="full markdown report across fee tiers")
    common(r)
    r.add_argument("--out", help="write markdown here")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
