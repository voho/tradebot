# tradebot

Paper-testing framework for BTCUSD 5-minute trading strategies. Every
registered strategy is backtested on **spot** and on **5x-leverage
futures**, each with **$1,000** and **$1,000,000** starting balances, and
the results are ranked by **final balance** in one comparison table, with
a chart per run (price + trades, balance curve, drawdown, results box).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

tradebot run            # full matrix -> reports/comparison.md + reports/charts/
tradebot list           # show registered strategies
tradebot new my_idea    # scaffold a new strategy file
pytest                  # test suite (incl. a no-lookahead check for every strategy)
```

Real data ships with the repo (see below), so `tradebot run` works out of
the box. Use `--max-bars 100000` for a quick iteration loop over the most
recent ~1 year instead of the full decade.

## Strategy comparison

Sorted **best to worst** by final balance (each strategy's best config);
every registered strategy MUST appear here — a full `tradebot run`
regenerates the table and CI fails if a strategy is missing from it.
Full metrics (win rate, drawdown, sharpe, fees, ...) live in
[reports/comparison.md](reports/comparison.md).

<!-- comparison:begin -->
_Period: 2017-01-01 to 2026-08-12 (1,010,889 x 5m bars) · data: real, spot (perp proxy)_

| strategy | spot · $1K | spot · $1M | futures_5x · $1K | futures_5x · $1M |
|---|---|---|---|---|
| **buy_and_hold**<br>_Buy everything on the first bar and never trade again._<br>[source](src/tradebot/strategies/buy_and_hold.py) | trades 1<br>profit $65.0K<br>worst $65.0K<br>best $65.0K<br>**after $66.0K** | trades 1<br>profit $65.04M<br>worst $65.04M<br>best $65.04M<br>**after $66.04M** | trades 1<br>profit -$982<br>worst -$982<br>best -$982<br>**after $18.05**<br>LIQUIDATED | trades 1<br>profit -$982.0K<br>worst -$982.0K<br>best -$982.0K<br>**after $18.0K**<br>LIQUIDATED |
| **rsi_reversion**<br>_Mean-reversion: buy oversold dips (RSI < 30), exit on recovery; mirror short overbought on futures._<br>[source](src/tradebot/strategies/rsi_reversion.py) | trades 4,464<br>profit -$995<br>worst -$159<br>best $133<br>**after $4.85** | trades 5,713<br>profit -$1.00M<br>worst -$159.0K<br>best $133.5K<br>**after $365** | trades 2,264<br>profit -$999<br>worst -$346<br>best $246<br>**after $0.77** | trades 2,638<br>profit -$1.00M<br>worst -$345.7K<br>best $246.1K<br>**after $3.41**<br>LIQUIDATED |
| **macd_rsi**<br>_Trend + timing combo: trade RSI pullback recoveries only in the direction of the MACD trend._<br>[source](src/tradebot/strategies/macd_rsi.py) | trades 2,454<br>profit -$995<br>worst -$44.90<br>best $42.97<br>**after $4.96** | trades 5,503<br>profit -$1.00M<br>worst -$44.9K<br>best $43.0K<br>**after $5.00** | trades 1,239<br>profit -$999<br>worst -$150<br>best $110<br>**after $0.94** | trades 2,120<br>profit -$1.00M<br>worst -$149.6K<br>best $110.2K<br>**after $0.98** |
| **macd_cross**<br>_Trend-following: long when MACD crosses above its signal line, flat/short on the cross below._<br>[source](src/tradebot/strategies/macd_cross.py) | trades 4,301<br>profit -$995<br>worst -$46.33<br>best $42.19<br>**after $4.99** | trades 7,945<br>profit -$1.00M<br>worst -$46.3K<br>best $42.2K<br>**after $5.00** | trades 1,464<br>profit -$999<br>worst -$259<br>best $566<br>**after $1.00** | trades 3,306<br>profit -$1.00M<br>worst -$258.6K<br>best $566.0K<br>**after $0.73** |
<!-- comparison:end -->

## Built-in strategies

| strategy | the idea |
|---|---|
| [buy_and_hold](src/tradebot/strategies/buy_and_hold.py) | Buy everything on the first bar, never trade again. BTC has historically rewarded holding through entire cycles — this is the benchmark every active strategy must beat after fees. On 5x futures it doubles as a stress test: a deep drawdown liquidates a passive leveraged long. |
| [macd_cross](src/tradebot/strategies/macd_cross.py) | Trend following. MACD (fast EMA minus slow EMA, 12/26) crossing above its 9-period signal line marks upward momentum early — go long and ride it; cross below → flat (spot) or short (futures). Weakness: on 5m bars it whipsaws in chop and fees eat the edge. |
| [rsi_reversion](src/tradebot/strategies/rsi_reversion.py) | Mean reversion. Sharp moves overshoot: RSI(14) < 30 means the sell-off is stretched, so buy the dip and exit once RSI recovers past 55; mirrored short side (RSI > 70) on futures. Works in ranges, bleeds in strong trends where "oversold" keeps falling. |
| [macd_rsi](src/tradebot/strategies/macd_rsi.py) | Trend + timing combo. Only take RSI pullback recoveries in the direction of the MACD trend (histogram > 0 → longs on RSI crossing up through 45; mirrored short side). Fewer but better-timed trades than either indicator alone. |

## Data

**Committed dataset**: `data/btcusd_spot_5m.csv.gz` — real Bitstamp
BTC/USD 5-minute candles, **2017-01-01 to 2026-08**, ~1.01M bars, no gaps.
Resampled from the 1-minute data in
[ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data)
(MIT-licensed, daily-updated). The span covers the 2017 bull run, 2018
bear, 2020 crash + bull, 2021 top, 2022 bear, and the 2023+ cycle, so
strategies are judged across both bull and bear regimes. Refresh it with:

```bash
git clone --depth 1 https://github.com/ff137/bitstamp-btcusd-minute-data /tmp/bitstamp
python scripts/build_bitstamp_dataset.py --source /tmp/bitstamp
```

Data files are resolved in priority order (all
`timestamp,open,high,low,close,volume`, ms UTC epoch, `.gz` read
transparently):

| priority | file | used by | label |
|---|---|---|---|
| 1 | `btcusdt_perp_5m.csv` (via `tradebot fetch`) | futures | `real` |
| 1 | `btcusdt_spot_aligned_5m.csv` (via `tradebot fetch`) | spot | `real` |
| 2 | `btcusd_spot_5m.csv.gz` (committed) | spot; futures fall back to it | `real` / `spot (perp proxy)` |
| 3 | `synthetic_*.csv` (generated, seeded) | last resort | `SYNTHETIC` |

Without true perp data the futures market trades the spot series — the
perp basis is small, and every table row and chart carries the
`spot (perp proxy)` label so it's never mistaken for real perp fills.
`tradebot fetch` (needs Binance network access) produces the true
perp + aligned-spot pair, which then takes precedence automatically.

## Adding a strategy

Scaffold it (creates `src/tradebot/strategies/<name>.py` with a working
EMA-cross template, auto-discovered on the next run):

```bash
tradebot new my_strategy
pytest                                        # no-lookahead check runs for it automatically
tradebot run --strategies my_strategy buy_and_hold --max-bars 100000   # quick compare
```

Or write the file by hand:

```python
# src/tradebot/strategies/my_strategy.py
import pandas as pd
from tradebot.indicators import ema
from tradebot.registry import register
from tradebot.strategy import Context, Strategy

@register
class MyStrategy(Strategy):
    """One-line description shown in reports."""

    name = "my_strategy"   # unique
    warmup = 100           # bars skipped before the first on_bar call

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Called once with the full OHLCV frame. Add indicator columns.
        # MUST be causal (row i may only use rows <= i): rolling / ewm /
        # shift are fine. A framework test verifies this for every
        # registered strategy.
        df["fast"] = ema(df["close"], 20)
        df["slow"] = ema(df["close"], 100)
        return df

    def on_bar(self, ctx: Context) -> None:
        # Called at every bar close; orders fill at the NEXT bar open.
        if ctx.bar["fast"] > ctx.bar["slow"] and not ctx.in_market:
            ctx.order_target(1.0)      # fully long (fraction of equity x leverage)
        elif ctx.bar["fast"] < ctx.bar["slow"] and ctx.position > 0:
            ctx.close_position()       # or ctx.order_target(-1.0) to short (futures)
```

That's it — `tradebot run` picks it up, tests it on the whole matrix,
ranks it in the comparison table and refreshes the README table. `ctx`
also offers `history(n)`, `equity`, `position`, `can_short`, and raw
`buy(qty)` / `sell(qty)`. `ctx.bar` / `ctx.prev` are fast mapping-style
views (`bar["rsi"]`).

Two rules are CI-enforced for every registered strategy (GitHub Actions
runs the suite on each push/PR):

- it **must have a docstring** describing the idea (first line lands in
  the comparison table and `tradebot list`), and
- it **must appear in the README comparison table** — run the full
  `tradebot run` after adding a strategy, commit the regenerated
  README + reports, and CI stays green.

## Reusing a strategy in a live bot (Bitstamp / Binance / 3Commas)

Strategies are pure decision functions over (candle history, account
state) — no backtest types leak into their API — so the class you
paper-tested is the class you deploy. `tradebot.live` is the extraction
point:

```python
from tradebot.broker import MarketSpec
from tradebot.live import LiveAccount, compute_signal
from tradebot.registry import get_strategy

strategy = get_strategy("macd_rsi")

# on every CLOSED 5m candle (never feed the forming one):
candles = fetch_ohlcv_window()          # same columns as the backtest data
account = LiveAccount(position=btc_position,      # signed, 0 = flat
                      equity_quote=equity_usd,
                      market=MarketSpec.spot())    # or .futures(leverage=5)
orders = compute_signal(strategy, candles, account)
```

The returned orders are venue-agnostic; the adapter is a few lines:

- **Bitstamp / Binance** (REST or websocket loop): for a `target` order
  `f`, desired notional = `equity_quote x leverage x f`; place a market
  order for the delta between that and the current position. `qty`
  orders map 1:1. (`tradebot.fetch` already shows the Binance klines
  pagination needed for the candle window.)
- **3Commas** (signal bots): `target > 0` → send the bot's start/long
  webhook signal, `target == 0` → close signal, `target < 0` → short
  signal; position sizing stays configured in the bot.

Timing contract is identical to the backtest — decide on bar close, act
at the next open — and `tests/test_live.py` proves parity: walking the
data bar-by-bar through `compute_signal` reproduces exactly the
decisions the backtester filled, for every registered strategy on both
markets.

## How the simulation works

- Signals are computed at a bar's **close**; orders fill at the **next
  bar's open** (no lookahead — enforced by tests that truncate future
  data and compare fills).
- **Spot**: long-only, 1x, 0.10% taker fee.
- **Futures**: long/short at configurable leverage (default 5x), 0.05%
  taker fee, cross-margin liquidation at a 0.5% maintenance-margin rate.
  Liquidation uses the analytic liquidation price, checked first at each
  bar's open (before queued orders fill, so a gap through the bankruptcy
  price can never lose more than the account holds) and then against the
  bar's high/low. Accounts floor at zero; a liquidated run stops trading
  and is flagged in the table.
- `order_target(f)` moves the position to `f` × max notional
  (equity × leverage), `f` ∈ [-1, 1] (clamped to [0, 1] on spot).
  Same-sign re-targets inside a 5% deadband are ignored so strategies
  may re-emit their target every bar without churning fees.
- Exchange-style **minimum order notional** ($5) — dust orders are
  skipped (reduces are always allowed).
- Optional slippage (`--slippage-bps`).
- Simplifications: no funding rates, taker-only fills, fills always
  succeed at open ± slippage (no order book depth).

## CLI

```
tradebot run [--balances 1000 1000000] [--markets spot futures]
             [--leverage 5] [--strategies macd_cross ...]
             [--slippage-bps 1] [--spot-fee 0.001] [--futures-fee 0.0005]
             [--max-bars 100000] [--data-dir data] [--out reports]
tradebot list
tradebot new <name>
tradebot fetch [--start 2020-01-01] [--end 2026-08-01] [--symbol BTCUSDT]
```

Outputs: `reports/comparison.md` (+ `.csv`) ranked by final balance per
(market, start balance) group, `reports/charts/<strategy>__<market>__<balance>.png`
per run, and `reports/charts/_all__<market>__<balance>.png` overlaying
all strategies' balance curves.
