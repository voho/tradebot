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

# get real data (needs network access to binance.com / binance.vision)
tradebot fetch --start 2025-01-01

tradebot run            # full matrix -> reports/comparison.md + reports/charts/
tradebot list           # show registered strategies
pytest                  # test suite (incl. a no-lookahead check for every strategy)
```

If the real data CSVs are missing, `tradebot run` falls back to a
deterministic **synthetic** BTC-like series so the pipeline stays
runnable; every chart and table is then clearly labeled `SYNTHETIC`.

## Data

Two canonical files in `data/` (whitelisted in `.gitignore`, format
`timestamp,open,high,low,close,volume`, timestamp in ms UTC):

| file | contents |
|---|---|
| `btcusdt_perp_5m.csv` | BTCUSDT USDT-margined perpetual, 5m klines (traded by the futures market) |
| `btcusdt_spot_aligned_5m.csv` | BTCUSDT spot 5m, aligned to the perp timestamps (traded by the spot market) |

`tradebot fetch` downloads both from Binance public endpoints (bulk
monthly archives first, REST API for the tail) and aligns them.

## Adding a strategy

Create one file in `src/tradebot/strategies/` — it is auto-discovered:

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

That's it — `tradebot run` picks it up, tests it on the whole matrix and
ranks it in the comparison table. `ctx` also offers `history(n)`,
`equity`, `position`, `can_short`, and raw `buy(qty)` / `sell(qty)`.

Built-in baselines: `buy_and_hold`, `macd_cross`, `rsi_reversion`,
`macd_rsi`.

## How the simulation works

- Signals are computed at a bar's **close**; orders fill at the **next
  bar's open** (no lookahead — enforced by tests that truncate future
  data and compare fills).
- **Spot**: long-only, 1x, 0.10% taker fee.
- **Futures**: long/short at configurable leverage (default 5x), 0.05%
  taker fee, cross-margin liquidation at a 0.5% maintenance-margin rate.
  Liquidation uses the analytic liquidation price against each bar's
  high/low; a liquidated run stops trading and is flagged in the table.
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
             [--max-bars 20000] [--data-dir data] [--out reports]
tradebot list
tradebot fetch [--start 2025-01-01] [--end 2025-08-01] [--symbol BTCUSDT]
```

Outputs: `reports/comparison.md` (+ `.csv`) ranked by final balance per
(market, start balance) group, `reports/charts/<strategy>__<market>__<balance>.png`
per run, and `reports/charts/_all__<market>__<balance>.png` overlaying
all strategies' balance curves.
