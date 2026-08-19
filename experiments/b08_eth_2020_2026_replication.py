"""B-08: does kelly_regime_v4 replicate, unchanged, on a second bear market on
a second asset over a genuinely independent period -- ETH 2020-2026?

Not registered: lives under ``experiments/`` per ROUTINE.md step 5. This is
a pure replication exercise, not a tuning round: `kelly_regime_v4` is run
byte-identical (no parameters touched, no import of anything from
`experiments/`). No file under ``src/tradebot/strategies/`` is modified.

Why this is independent of every existing "ETH falsification test" in this
ledger (R-17 and everything built on it): the committed
``ethusd_bitfinex_5m.csv.gz`` stops in 2019-12, so every prior ETH check in
this project shares the *same* 2018 BTC bear window with the main dataset --
not an independent test. ``ethusd_coinbase_spot_5m.csv.gz`` (fetched for
B-15/R-41's basis work, already committed) covers 2019-03-14 -> 2026-08-19,
which is the first time the 2022 ETH bear (Terra/Luna, 3AC, FTX) and the
2023-2026 period have been available for ETH here at all. `kelly_regime_v4`
has never been tuned against any ETH data (zero free parameters were ever
fit on it) so there is no leakage risk in reading any ETH period -- the
"holdout" discipline below is about not moving the goalposts after looking,
not about protecting a fitting process. The BTC 2023+ holdout is NOT read by
this script and its counter is not touched.

=====================================================================
PRE-REGISTRATION -- written before any ETH 2020+ number was computed
=====================================================================

Decision rule (ROUTINE.md step 4's promotion bar, adapted -- the parameter-
neighbourhood clause is moot here because nothing is tuned):

    REPLICATES on a cell (window x market) if, after real costs:
      (a) kelly_regime_v4's final balance beats buy_and_hold's, AND
      (b) either the Sharpe improvement exceeds the project's measured
          +/-0.2 noise floor (R-20), OR it is a drawdown/tail
          improvement -- which this project has repeatedly found to be
          the property that actually replicates (L-01's own lesson,
          R-17's own finding: "the risk property transfers, the return
          property does not").
    FAILS TO REPLICATE otherwise: v4 loses to buy_and_hold after costs,
      or "wins" by a margin inside the noise floor with no real
      drawdown/tail improvement either.
    A DD "improvement" that is arithmetic (v4 running lower average
    exposure than a 1x/5x-static benchmark) rather than a genuine tail
    property is flagged, per the standing rule "match risk before
    comparing anything" -- not scored as a free win.

Pre-registered cells, decided before looking at any ETH 2020+ result:
  (a) 2022 bear window, 2022-05-01 -> 2022-11-30 inclusive (Terra/Luna
      collapse in May through the FTX collapse in November), spot and
      futures 5x.
  (b) Full window, 2020-01-01 -> end of file (2026-08-19), spot and
      futures 5x.

Cost assumption discipline (fee_study.py's pattern, not its code):
  - baseline: this project's standard default fee for each market --
    MarketSpec.spot() (0.10% taker) and MarketSpec.futures(leverage=5)
    (0.05% taker, the project's standing perp assumption).
  - sensitivity: the 0.40% Bitstamp entry taker tier
    (MarketSpec.spot(fee_rate=0.004)), applied to both pre-registered
    windows on spot -- this is the tier fee_study.py itself sensitizes
    (Bitstamp is a spot venue; the project has never defined a "0.40%
    futures tier" as a distinct real-world venue quote, so the futures
    leg is reported at its standard default only, with funding named as
    the caveat below).

Known, named-in-advance limitation: no ETH perpetual funding data exists
in this repository (no ``ethusdt_*_funding_8h.csv.gz`` file). Every futures
number below is therefore a **funding-free upper bound**, exactly the
caveat this project already attaches to every futures column (README's
standing warning; R-14's BTC funding study cut kelly_regime_v4's BTC futures
number from $156K to $36K-$80K). This script does not attempt to proxy or
estimate ETH funding -- that would violate "never proxy unavailable data out
of price" -- it is left as a named gap for a future session.

Total backtest configurations run by this script: 12.
  8 baseline  (2 windows x 2 markets x {kelly_regime_v4, buy_and_hold})
  4 sensitivity (2 windows x spot-only x {kelly_regime_v4, buy_and_hold}
                 @ 0.40% fee)
Nothing is swept or selected between; every cell is reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_coinbase_eth_spot  # noqa: E402
from tradebot.engine import run_backtest, validate_ohlcv  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"

BEAR_2022 = ("2022-05-01", "2022-11-30")
FULL = ("2020-01-01", None)  # None end = last bar in the file (2026-08-19)

SPOT_BASE = MarketSpec.spot()                      # 0.10% taker
SPOT_STRESS = MarketSpec.spot(fee_rate=0.004)       # Bitstamp 0.40% entry tier
FUT_BASE = MarketSpec.futures(leverage=5.0)         # 0.05% taker, no funding data for ETH

CONFIG_COUNT = 0


def ev(strategy, df, market, start, end, tag):
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    m = compute_metrics(result)
    print(f"{tag:42s} {market.name:13s} "
          f"final=${m.final_balance:>13,.0f} ({m.profit_pct:>+9.1f}%) "
          f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>6.2f} {'LIQUIDATED' if m.liquidated else ''}")
    return m


def integrity_check(df: pd.DataFrame) -> None:
    """Sanity check on the loading path itself -- not a re-audit of v4's
    own logic (covered by tests/test_causality_strict.py on the BTC file).
    """
    print("=" * 100)
    print("DATA INTEGRITY CHECK -- ethusd_coinbase_spot_5m.csv.gz")
    print("=" * 100)
    validate_ohlcv(df)  # reuses the project's own OHLCV validator -- monotonic,
                         # no dupes, positive prices, DatetimeIndex. Raises on failure.
    print(f"validate_ohlcv: PASS ({len(df):,} rows, "
          f"{df.index[0]} -> {df.index[-1]})")
    gaps = df.index.to_series().diff().dropna()
    big = gaps[gaps > pd.Timedelta(hours=1)]
    print(f"monotonic index: {df.index.is_monotonic_increasing}  "
          f"duplicate timestamps: {int(df.index.duplicated().sum())}")
    print(f"gaps > 1h: {len(big)} (largest: {gaps.max()}) -- "
          f"v4's warmup is 80 days, so any gap this small cannot corrupt it")
    print("Loader is `load_ohlcv_csv` via `load_coinbase_eth_spot` -- the same "
          "row-wise CSV reader every other canonical file in this project uses; "
          "it computes nothing over the whole series (no scaler/mean/std), so "
          "it carries no full-series-fit lookahead risk of the kind ROUTINE.md warns about.")


def causality_tamper_probe(df: pd.DataFrame, strategy_name: str) -> None:
    """Same tamper methodology as tests/test_causality_strict.py's
    `_decisions` helper, run directly against the new ETH loading path
    (that test module hard-codes the BTC spot loader, so it does not
    exercise this file automatically -- this closes that specific gap,
    honestly, rather than claiming coverage the pytest suite doesn't have).
    """
    print("=" * 100)
    print(f"CAUSALITY TAMPER PROBE -- {strategy_name} on ETH Coinbase spot")
    print("=" * 100)
    tail = df.iloc[-60_000:].copy()
    cut = len(tail) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]

    up, down = tail.copy(), tail.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    market = MarketSpec.futures(leverage=5.0)

    def decisions(frame):
        s = get_strategy(strategy_name)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=market, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    a, b = decisions(up), decisions(down)
    ok = all(oa == ob for oa, ob in zip(a, b))
    print(f"decisions at/before the tamper cut identical under opposite "
          f"post-cut tampers: {'PASS' if ok else 'FAIL'}")
    if not ok:
        for bar, oa, ob in zip(bars, a, b):
            if oa != ob:
                print(f"  MISMATCH at bar {bar}: {oa} vs {ob}")


def main() -> None:
    eth, _label = load_coinbase_eth_spot(DATA_DIR), "real"
    if eth is None:
        raise SystemExit("ethusd_coinbase_spot_5m.csv.gz not found in data/")
    print(f"ETH Coinbase spot: {len(eth):,} bars, "
          f"{eth.index[0]:%Y-%m-%d} -> {eth.index[-1]:%Y-%m-%d}\n")

    integrity_check(eth)
    print()
    causality_tamper_probe(eth, "kelly_regime_v4")
    print()

    windows = {"2022 bear (05-01..11-30)": BEAR_2022,
               "full 2020-01-01..end": FULL}
    markets_base = {"spot": SPOT_BASE, "futures_5x": FUT_BASE}

    print("=" * 100)
    print("BASELINE COSTS (spot 0.10%, futures 0.05%, no ETH funding data available)")
    print("=" * 100)
    results = {}
    for wname, (start, end) in windows.items():
        for mname, market in markets_base.items():
            for sname in ("kelly_regime_v4", "buy_and_hold"):
                tag = f"{wname} | {sname}"
                m = ev(get_strategy(sname), eth, market, start, end, tag)
                results[(wname, mname, "base", sname)] = m
        print()

    print("=" * 100)
    print("SENSITIVITY: 0.40% Bitstamp entry taker tier, spot only")
    print("=" * 100)
    for wname, (start, end) in windows.items():
        for sname in ("kelly_regime_v4", "buy_and_hold"):
            tag = f"{wname} | {sname} @0.40%"
            m = ev(get_strategy(sname), eth, SPOT_STRESS, start, end, tag)
            results[(wname, "spot", "stress", sname)] = m
        print()

    print("=" * 100)
    print(f"Total backtest configurations run: {CONFIG_COUNT}")
    print("=" * 100)

    print("\nVERDICT TABLE (per pre-registered cell, baseline cost only):")
    for wname in windows:
        for mname in markets_base:
            v4 = results[(wname, mname, "base", "kelly_regime_v4")]
            hold = results[(wname, mname, "base", "buy_and_hold")]
            beats = v4.final_balance > hold.final_balance
            dsharpe = v4.sharpe - hold.sharpe
            ddelta = hold.max_drawdown_pct - v4.max_drawdown_pct  # positive = v4 lower DD
            noise_floor_clear = abs(dsharpe) > 0.2
            dd_improved = ddelta > 0
            replicates = beats and (noise_floor_clear and dsharpe > 0 or dd_improved)
            print(f"  {wname:28s} {mname:11s} beats_hold={beats!s:5s} "
                  f"dSharpe={dsharpe:+.2f} dDD(pp)={ddelta:+.1f} "
                  f"-> {'REPLICATES' if replicates else 'FAILS TO REPLICATE'}")


if __name__ == "__main__":
    main()
