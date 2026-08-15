#!/usr/bin/env python
"""Can a strategy be tuned to survive a high taker fee? Reproduces the answer.

Motivated by a real deployment: Bitstamp's entry tier is 0.40% taker,
4x the 0.10% every table in this repo assumes. The obvious response -
trade less - was tried properly, and this script is the record of why it
does not work.

Four checks, run in order::

    python scripts/fee_study.py ceiling      # gross edge vs the fee drag
    python scripts/fee_study.py breakeven    # the fee each strategy tolerates
    python scripts/fee_study.py tiers        # would climbing the volume ladder fix it?
    python scripts/fee_study.py fills        # where the fee money actually goes
    python scripts/fee_study.py leverage     # does leverage beat the fee?
    python scripts/fee_study.py plateau      # is the parameter grid a plateau?
    python scripts/fee_study.py walkforward  # select in-sample, test out-of-sample
    python scripts/fee_study.py all

``walkforward`` is the one that settles it: 28 of 32 configurations beat
buy-and-hold in-sample and none of them beat it out-of-sample. Findings
written up in docs/LIVE.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, KellyRegime  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
FEES = (0.0, 0.001, 0.002, 0.003, 0.004)
BITSTAMP_TAKER = 0.004
LOOKBACKS = (20, 30, 40, 50, 60, 70, 80, 100)
BANDS = (0.01, 0.03, 0.05, 0.08)


class RawRegimeFilter(KellyRegime):
    """The regime vote with volatility targeting switched off.

    ``target_vol=5.0`` makes ``min(target_vol / vol, 1.0)`` saturate at
    1.0, so exposure is the vote alone. This is the highest-gross-edge
    member of the family on spot and therefore the fairest candidate for
    "can anything survive 0.40%".
    """

    name = "_raw_regime_filter"

    def __init__(self, horizons, band=0.01, **kw):
        super().__init__(horizons=horizons, band=band, target_vol=5.0,
                         max_leverage=1.0, **kw)
        self.warmup = max(horizons) * BARS_PER_DAY + 10


def _full(make, fee):
    return compute_metrics(run_backtest(make(), DF, MarketSpec.spot(fee_rate=fee),
                                        1_000.0, data_label=LABEL))


def _period(make, fee, start=None, end=None):
    return compute_metrics(run_period(make(), DF, start, end,
                                      market=MarketSpec.spot(fee_rate=fee),
                                      start_balance=1_000.0, data_label=LABEL))


def ceiling() -> None:
    """Gross (fee-free) edge vs what turnover costs — the go/no-go number."""
    print("gross edge vs holding, and what the fee takes back (spot, $1,000):\n")
    hold_gross = _full(lambda: get_strategy("buy_and_hold"), 0.0).final_balance
    print(f"{'strategy':20s} {'gross':>11s} {'@0.40%':>11s} {'edge':>7s} "
          f"{'drag':>7s} {'needed':>8s}")
    for name in ("buy_and_hold", "kelly_regime_v4", "kelly_regime_v3", "kelly_regime"):
        gross = _full(lambda n=name: get_strategy(n), 0.0).final_balance
        net = _full(lambda n=name: get_strategy(n), BITSTAMP_TAKER).final_balance
        drag = net / gross
        print(f"{name:20s} ${gross:>10,.0f} ${net:>10,.0f} "
              f"{gross / hold_gross:>6.2f}x {drag:>6.2f}x {1 / drag:>7.2f}x")
    print("\n'edge' is gross vs holding; 'needed' is the gross edge that turnover\n"
          "would require to break even. Needed exceeds edge by more than 2x.")


def breakeven() -> None:
    """The highest fee at which each strategy still beats holding."""
    hold = {f: _full(lambda: get_strategy("buy_and_hold"), f).final_balance for f in FEES}
    print("final balance by taker fee (spot, $1,000):\n")
    print(f"{'strategy':30s} " + "".join(f"{f:>11.2%}" for f in FEES))
    print(f"{'buy_and_hold':30s} " + "".join(f"${hold[f]:>10,.0f}" for f in FEES))
    print("-" * 88)
    cands = {
        "kelly_regime_v4": lambda: get_strategy("kelly_regime_v4"),
        "kelly_regime_v3": lambda: get_strategy("kelly_regime_v3"),
        "raw filter h=50": lambda: RawRegimeFilter((50,)),
        "raw filter h=100 band=.05": lambda: RawRegimeFilter((100,), band=0.05),
    }
    for tag, make in cands.items():
        vals = {f: _full(make, f).final_balance for f in FEES}
        be = "> 0.40%" if vals[0.0] > hold[0.0] else "never"
        for f in FEES[1:]:
            if vals[f] < hold[f]:
                be = f"< {f:.2%}"
                break
        print(f"{tag:30s} " + "".join(f"${vals[f]:>10,.0f}" for f in FEES) + f"   {be}")


BITSTAMP_TIERS = (
    (0.0040, "< $10k/30d, entry taker"),
    (0.0025, "~ $100k-$1M/30d"),
    (0.0020, "> $1M/30d"),
    (0.0012, "> $5M/30d, taker"),
    (0.0003, "> $5M/30d, MAKER (limit orders)"),
)


def tiers() -> None:
    """Would climbing the venue's volume ladder restore the edge?

    Answer: it closes most of the gap and still misses. The break-even is
    bisected rather than asserted, so it stays honest if the strategy or
    the data changes.
    """
    def fin(name, fee):
        return _full(lambda n=name: get_strategy(n), fee).final_balance

    print("kelly_regime_v4 vs buy_and_hold by fee tier (spot, full period):\n")
    print(f"{'fee':>7s}  {'30-day volume tier':32s} {'v4':>10s} {'holding':>10s}  verdict")
    for fee, label in BITSTAMP_TIERS:
        v4, hold = fin("kelly_regime_v4", fee), fin("buy_and_hold", fee)
        verdict = (f"BEATS holding +{v4 / hold - 1:.0%}" if v4 > hold
                   else f"loses {v4 / hold - 1:.0%}")
        print(f"{fee:>6.2%}  {label:32s} ${v4:>9,.0f} ${hold:>9,.0f}  {verdict}")

    lo, hi = 0.0001, 0.0040
    for _ in range(14):
        mid = (lo + hi) / 2
        if fin("kelly_regime_v4", mid) > fin("buy_and_hold", mid):
            lo = mid
        else:
            hi = mid
    print(f"\nbreak-even taker fee: {lo:.3%}")
    print("No Bitstamp TAKER tier reaches it; only the maker rate does, and that\n"
          "is a change of order type - with fill risk this backtester does not\n"
          "model - rather than a change of tier.")

    result = run_backtest(get_strategy("kelly_regime_v4"), DF,
                          MarketSpec.spot(fee_rate=BITSTAMP_TAKER), 1_000.0)
    fills = len(result.fills)
    years = (DF.index[-1] - DF.index[0]).days / 365.25
    print(f"\nturnover reality check: {fills:,} fills over {years:.1f} years "
          f"(~1 every {years * 365 / fills:.0f} days).")
    print("The comparison table's trade count is round-trip EPISODES, not fills.")


def plateau() -> None:
    """A real edge is a region; noise is scattered winners."""
    hold = _full(lambda: get_strategy("buy_and_hold"), BITSTAMP_TAKER).final_balance
    print(f"final balance ($K) @ {BITSTAMP_TAKER:.2%} · holding = ${hold:,.0f} "
          "· '+' beats holding\n")
    print("band\\days " + "".join(f"{h:>9d}" for h in LOOKBACKS))
    wins = 0
    for band in BANDS:
        cells = []
        for h in LOOKBACKS:
            final = _full(lambda h=h, b=band: RawRegimeFilter((h,), band=b),
                          BITSTAMP_TAKER).final_balance
            wins += final > hold
            cells.append(f"{final / 1000:>8.1f}" + ("+" if final > hold else " "))
        print(f"{band:<9.2f}" + "".join(cells))
    print(f"\n{wins}/{len(BANDS) * len(LOOKBACKS)} beat holding - but scattered, not "
          "clustered.\nAdjacent cells swing 2-3x, which is what noise looks like.")


def walkforward() -> None:
    """Select on 2017-2022, evaluate on 2023-2026. The check that settles it."""
    hold_is = _period(lambda: get_strategy("buy_and_hold"), BITSTAMP_TAKER,
                      end="2022-12-31").final_balance
    hold_oos = _period(lambda: get_strategy("buy_and_hold"), BITSTAMP_TAKER,
                       start="2023-01-01").final_balance
    print(f"buy_and_hold   IS ${hold_is:,.0f}   OOS ${hold_oos:,.0f}\n")

    rows = []
    for h in LOOKBACKS:
        for band in BANDS:
            def make(h=h, band=band):
                return RawRegimeFilter((h,), band=band)
            rows.append((
                _period(make, BITSTAMP_TAKER, end="2022-12-31").final_balance,
                _period(make, BITSTAMP_TAKER, start="2023-01-01").final_balance,
                h, band))
    rows.sort(reverse=True)

    print("ranked by IN-SAMPLE, then checked out-of-sample:")
    print(f"{'rank':>4s} {'days':>5s} {'band':>5s} {'IS $':>11s} {'OOS $':>10s}  verdict")
    for k, (is_, oos, h, band) in enumerate(rows[:8], 1):
        verdict = ("beats holding" if oos > hold_oos
                   else f"LOSES ({oos / hold_oos:.0%} of holding)")
        print(f"{k:>4d} {h:>5d} {band:>5.2f} ${is_:>10,.0f} ${oos:>9,.0f}  {verdict}")

    is_win = [r for r in rows if r[0] > hold_is]
    both = [r for r in is_win if r[1] > hold_oos]
    best = rows[0]
    print(f"\nbeat holding in-sample:       {len(is_win)}/{len(rows)}")
    print(f"...and also out-of-sample:    {len(both)}/{len(is_win)}")
    print(f"\nthe config you would have picked (best in-sample: {best[2]}d, "
          f"band {best[3]:.2f}):\n  out-of-sample ${best[1]:,.0f} vs holding "
          f"${hold_oos:,.0f}  ->  {best[1] / hold_oos - 1:+.1%}")


def fills_by_kind() -> None:
    """Where does the fee money actually go? Mostly not where you'd guess."""
    result = run_backtest(get_strategy("kelly_regime_v4"), DF,
                          MarketSpec.spot(fee_rate=BITSTAMP_TAKER), 1_000.0)
    pos = 0.0
    kinds = {"entry": [0, 0.0], "exit": [0, 0.0], "resize": [0, 0.0]}
    for f in result.fills:
        signed = f.qty if f.side.name == "BUY" else -f.qty
        new = pos + signed
        kind = ("entry" if abs(pos) < 1e-12
                else "exit" if abs(new) < 1e-12 else "resize")
        kinds[kind][0] += 1
        kinds[kind][1] += f.fee
        pos = new

    total = sum(v[1] for v in kinds.values())
    print(f"kelly_regime_v4 @ {BITSTAMP_TAKER:.2%} spot: {len(result.fills):,} fills, "
          f"${total:,.0f} of fees on a ${result.final_balance:,.0f} account\n")
    print(f"{'fill type':10s} {'count':>7s} {'fees':>12s}  share")
    for kind, (n, fee) in kinds.items():
        print(f"{kind:10s} {n:>7d} ${fee:>11,.0f}  {fee / total:>6.1%}")
    print("\nMost fills are RESIZES inside an already-open position - the "
          "volatility-targeting\nmachinery, whose upside spot cannot even use "
          "because exposure caps at 1.0.")


def leverage() -> None:
    """Does leverage beat the fee? Not by making it cheaper - fees scale too."""
    fee = BITSTAMP_TAKER
    spot_hold = _full(lambda: get_strategy("buy_and_hold"), fee).final_balance
    print(f"the real alternative is UNLEVERED spot holding: ${spot_hold:,.0f} "
          f"@ {fee:.2%}\n")
    print(f"{'cap':>5s} {'final $':>11s} {'DD%':>6s} {'sharpe':>7s} {'vs spot hold':>13s}")
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
    venue = MarketSpec.futures(leverage=10.0, fee_rate=fee)
    for cap in (2.0, 3.0, 4.0, 6.0):
        m = compute_metrics(run_backtest(KellyRegimeV4(max_leverage=cap), DF,
                                         venue, 1_000.0))
        print(f"{cap:>4.0f}x ${m.final_balance:>10,.0f} {m.max_drawdown_pct:>5.1f}% "
              f"{m.sharpe:>7.2f} {m.final_balance / spot_hold - 1:>12.0%}")
    print("\nOn this single full-history path leverage never catches unlevered\n"
          "holding, because fees scale with notional exactly as returns do.\n"
          "The advantage is in the DISTRIBUTION, not this path - see the stress\n"
          "test in docs/LIVE.md: median window return roughly doubles and the\n"
          "median drawdown halves.")


COMMANDS = {"ceiling": ceiling, "breakeven": breakeven, "tiers": tiers,
            "fills": fills_by_kind, "leverage": leverage,
            "plateau": plateau, "walkforward": walkforward}


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python scripts/fee_study.py [{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    main()
