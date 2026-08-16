#!/usr/bin/env python
"""What do perpetual funding payments do to the futures results?

Every futures figure in this repo was computed on a **funding-free perp**,
which is not a real instrument. Perps settle funding every 8 hours; on BTC
2020-2023 it was positive at 86% of settlements and cost a constant long
about 15% a year, charged on notional. The leading strategies hold
leveraged longs, so this is the largest unmodelled cost in the project.

Commands::

    python scripts/funding_study.py rates      # what the funding data says
    python scripts/funding_study.py measured   # 2020-2023, REAL funding
    python scripts/funding_study.py fullperiod # extrapolated, with a band
    python scripts/funding_study.py timing     # is the cost adversely timed?
    python scripts/funding_study.py all

`measured` is the only one that is purely observed. The committed funding
history covers 2020-2023; anything outside that is an assumption, and
`fullperiod` reports a band rather than a number for exactly that reason.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()
LEADERS = ("kelly_regime_v4", "kelly_regime_v3", "kelly_regime", "champions_council")


def _period(name, market, start=None, end=None, funding=None):
    """Backtest over a date range, warmed on the bars before it."""
    strategy = get_strategy(name)
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def rates() -> None:
    """The cost, before any strategy is involved."""
    print(f"{len(REAL):,} settlements  {REAL.index[0]:%Y-%m-%d} -> "
          f"{REAL.index[-1]:%Y-%m-%d}  (Binance BTCUSDT perp)\n")
    print(f"mean 8h rate {REAL.mean():+.6f}  -> {REAL.mean() * 3 * 365.25:+.2%} "
          f"a year to a constant long")
    print(f"positive (longs pay) at {(REAL > 0).mean():.1%} of settlements\n")
    print("by year, annualized cost to a constant long:")
    for year, group in REAL.groupby(REAL.index.year):
        print(f"  {year}  {group.mean() * 3 * 365.25:+7.2%}   "
              f"positive {(group > 0).mean():.0%}")
    print("\nThis is a persistent cost, not noise: it is the price of being on\n"
          "the crowded side of a perp, and the crowd is long.")


def measured() -> None:
    """2020-2023 only, where the funding is observed rather than assumed."""
    start, end = "2020-01-01", "2023-12-31"
    print(f"{start} .. {end}, real funding, $1,000 start\n")
    spot_hold, _ = _period("buy_and_hold", SPOT, start, end)
    print(f"{'strategy':20s} {'funding-free':>13s} {'with funding':>13s} "
          f"{'cost':>6s} {'paid':>10s}")
    for name in LEADERS + ("buy_and_hold",):
        free, _ = _period(name, FUTURES, start, end)
        paid, cost = _period(name, FUTURES, start, end, funding=REAL)
        print(f"{name:20s} ${free.final_balance:>12,.0f} "
              f"${paid.final_balance:>12,.0f} "
              f"{paid.final_balance / free.final_balance - 1:>5.0%} ${cost:>9,.0f}")
    print(f"\n{'buy_and_hold (SPOT 1x)':20s} ${spot_hold.final_balance:>12,.0f}"
          "   <- pays no funding; the benchmark to clear")


def _synthetic_series(rate: float) -> pd.Series:
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    return pd.Series(rate, index=grid)


def _blended() -> pd.Series:
    """Real rates where they exist, the empirical mean everywhere else."""
    filler = _synthetic_series(float(REAL.mean()))
    filler = filler[~filler.index.isin(REAL.index)]
    return pd.concat([filler, REAL]).sort_index()


def fullperiod() -> None:
    """The headline number, with funding. Reported as a band, not a point."""
    mean = float(REAL.mean())
    free, _ = _period("kelly_regime_v4", FUTURES)
    hold, _ = _period("buy_and_hold", SPOT)
    print("kelly_regime_v4, 5x futures, full 2017-2026 period.\n"
          "Funding is OBSERVED for 2020-2023 and ASSUMED elsewhere, so this is\n"
          "a band, not a measurement.\n")
    print(f"{'assumption':34s} {'final':>10s} {'DD':>7s} {'sharpe':>7s}")
    print(f"{'no funding (published headline)':34s} ${free.final_balance:>9,.0f} "
          f"{free.max_drawdown_pct:>6.1f}% {free.sharpe:>7.2f}")
    for label, series in (("real 2020-23 + mean elsewhere", _blended()),
                          ("constant at the mean", _synthetic_series(mean)),
                          ("constant at 2x mean (stress)", _synthetic_series(2 * mean))):
        m, _ = _period("kelly_regime_v4", FUTURES, funding=series)
        print(f"{label:34s} ${m.final_balance:>9,.0f} "
              f"{m.max_drawdown_pct:>6.1f}% {m.sharpe:>7.2f}")
    print(f"\n{'spot buy_and_hold benchmark':34s} ${hold.final_balance:>9,.0f}")
    print("\nThe band straddles the benchmark. The published $156K is an artifact\n"
          "of a funding-free perp and should not be quoted without this caveat.")


def timing() -> None:
    """Is funding merely a cost, or is it adversely timed against this strategy?"""
    result = run_backtest(get_strategy("kelly_regime_v4"), DF, FUTURES, 1_000.0)
    price = result.df["close"].to_numpy()
    equity = result.equity.to_numpy()
    pos = np.zeros(len(price))
    running, last = 0.0, 0
    offset = {ts: i for i, ts in enumerate(result.df.index)}
    for f in result.fills:
        i = offset[f.ts]
        pos[last:i] = running
        running += f.qty if f.side.name == "BUY" else -f.qty
        last = i
    pos[last:] = running
    exposure = np.abs(pos) * price / np.maximum(equity, 1e-9)

    print("why the cost is smaller than a constant long would pay:\n")
    print(f"  flat {100 * (np.abs(pos) < 1e-12).mean():.0f}% of bars - the regime "
          "gate dodges funding along with drawdowns")
    print(f"  mean notional/equity while in market: "
          f"{exposure[exposure > 1e-9].mean():.2f}x\n")

    in_market = pd.Series(np.abs(pos) > 1e-12, index=result.df.index)
    aligned = in_market.reindex(REAL.index, method="ffill").fillna(False)
    held, idle = REAL[aligned.to_numpy()], REAL[~aligned.to_numpy()]
    print("but the cost is adversely timed:\n")
    print(f"  mean rate while HOLDING: {held.mean():+.6f} "
          f"({held.mean() * 3 * 365.25:+.2%}/yr)")
    print(f"  mean rate while FLAT:    {idle.mean():+.6f} "
          f"({idle.mean() * 3 * 365.25:+.2%}/yr)")
    print("\nFunding is richest in exactly the bullish regimes a trend follower\n"
          "wants to be long in, so an average-rate assumption understates the\n"
          "real bill - which is why the blended estimate is worse than the\n"
          "constant-at-mean one.")


COMMANDS = {"rates": rates, "measured": measured,
            "fullperiod": fullperiod, "timing": timing}


def main() -> None:
    if REAL is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python scripts/funding_study.py [{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    main()
