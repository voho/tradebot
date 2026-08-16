# tradebot

Paper-testing framework for BTCUSD 5-minute trading strategies. Every
registered strategy is backtested on **spot** and on **5x-leverage
futures** from a **$1,000** start, and the results are ranked by **final
balance** in one comparison table, with a chart per run (price + trades,
balance curve, drawdown, results box).

Results are proportional to capital — verified across every strategy,
where the only deviations came from the exchange minimum order size — so
one start balance is the default. Test others with
`tradebot run --balances 1000 1000000`.

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

> 🚨 **The `futures_5x` column ignores funding, and that is worth 2–3x.**
> Perpetuals settle funding every 8 hours; on real Binance data it was
> positive at 86.5% of settlements and cost a constant long ~15% a year.
> Charging it, `kelly_regime_v4`'s $156K becomes **$36K–$80K** depending
> on the period assumed — a band that *straddles* spot buy-and-hold's
> $66K. Worse, funding runs **+20%/yr while the strategy holds** against
> +2.8% while it is flat, because the crowding that drives the signal is
> what sets the rate. Read every futures figure below as an upper bound;
> detail in [docs/VALIDATION.md](docs/VALIDATION.md#funding-the-cost-that-was-missing-and-what-it-does),
> reproduce with `python scripts/funding_study.py all`.

One full-history number can hide a lucky path, so the top three are also
resampled over 40 random windows
(`python scripts/stress_test.py`, charts in
[reports/stress/](reports/stress/), analysis in
[docs/VALIDATION.md](docs/VALIDATION.md)). Headline: on 5x futures,
leveraged buy-and-hold was **liquidated in 26 of 40 windows** — its median
window return is **−98%** — while every `kelly_regime` variant survived
**all 40**, stayed profitable in 85–88% of them, and beat holding in 65%.

<!-- comparison:begin -->
_Period: 2017-01-01 to 2026-08-12 (1,010,889 x 5m bars) · data: real, spot (perp proxy)_

| # | strategy | spot | futures_5x | trades | profit | max DD |
|---|---|---|---|---|---|---|
| 🥇1 | [kelly_regime_v4](src/tradebot/strategies/kelly_regime_v4.py) | 🟢 $66.8K | 🟢 **$156.2K** | 174 | 📈 $155.2K | 35% |
| 🥈2 | [kelly_regime_v3](src/tradebot/strategies/kelly_regime_v3.py) | 🟢 $65.8K | 🟢 **$139.5K** | 147 | 📈 $138.5K | 42% |
| 🥉3 | [kelly_regime_v2](src/tradebot/strategies/kelly_regime_v2.py) | 🟢 $46.4K | 🟢 **$122.0K** | 113 | 📈 $121.0K | 40% |
| 4 | [kelly_regime](src/tradebot/strategies/kelly_regime.py) | 🟢 $42.1K | 🟢 **$108.2K** | 143 | 📈 $107.2K | 43% |
| 5 | [kelly_regime_ev](src/tradebot/strategies/kelly_regime_ev.py) | 🟢 $40.9K | 🟢 **$108.0K** | 135 | 📈 $107.0K | 37% |
| 6 | [kelly_regime_ev_fast](src/tradebot/strategies/kelly_regime_ev.py) | 🟢 **$71.1K** | 🟢 $70.8K | 34 | 📈 $70.1K | 32% |
| 7 | [buy_and_hold](src/tradebot/strategies/buy_and_hold.py) | 🟢 **$66.0K** | 💀 $18.05 | 1 | 📈 $65.0K | 84% ⚠️ |
| 8 | [champions_council](src/tradebot/strategies/champions_council.py) | 🟢 $19.3K | 🟢 **$36.8K** | 261 | 📈 $35.8K | 37% |
| 9 | [hedge_experts](src/tradebot/strategies/hedge_experts.py) | 🟢 **$13.3K** | 🔴 $258 | 2,044 | 📈 $12.3K | 59% ⚠️ |
| 10 | [replicator_book](src/tradebot/strategies/replicator_book.py) | 🟢 **$2,330** | 🔴 $10.58 | 713 | 📈 $1,330 | 38% |
| 11 | [universal_kelly](src/tradebot/strategies/universal_kelly.py) | 🟢 **$1,276** | 🟢 $1,227 | 9 | 📈 $276 | 7% |
| 12 | [harsanyi_crowd](src/tradebot/strategies/harsanyi_crowd.py) | 🔴 **$888** | 🔴 $429 | 91 | 📉 -$112 | 11% |
| 13 | [overshoot_fade](src/tradebot/strategies/overshoot_fade.py) | 🔴 **$662** | 🔴 $33.52 | 189 | 📉 -$338 | 37% |
| 14 | [camouflage_flow](src/tradebot/strategies/camouflage_flow.py) | 🔴 **$548** | 🔴 $0.99 | 802 | 📉 -$452 | 53% ⚠️ |
| 15 | [stealth_trend](src/tradebot/strategies/stealth_trend.py) | 🔴 **$465** | 🔴 $0.38 | 1,605 | 📉 -$535 | 55% ⚠️ |
| 16 | [flow_regime](src/tradebot/strategies/flow_regime.py) | 🔴 **$447** | 🔴 $0.80 | 1,184 | 📉 -$553 | 56% ⚠️ |
| 17 | [game_council](src/tradebot/strategies/game_council.py) | 🔴 **$284** | 🔴 $2.00 | 2,541 | 📉 -$716 | 72% ⚠️ |
| 18 | [minority_oracle](src/tradebot/strategies/minority_oracle.py) | 🔴 **$53.36** | 🔴 $3.83 | 9,039 | 📉 -$947 | 95% ⚠️ |
| 19 | [game_switch](src/tradebot/strategies/game_switch.py) | 🔴 **$5.00** | 🔴 $1.00 | 6,672 | 📉 -$995 | 99% ⚠️ |
| 20 | [regret_grid](src/tradebot/strategies/regret_grid.py) | 🔴 **$5.00** | 🔴 $1.00 | 3,461 | 📉 -$995 | 100% ⚠️ |
| 21 | [tft_trend](src/tradebot/strategies/tft_trend.py) | 🔴 **$4.99** | 🔴 $1.00 | 2,538 | 📉 -$995 | 100% ⚠️ |
| 22 | [macd_cross](src/tradebot/strategies/macd_cross.py) | 🔴 **$4.99** | 🔴 $1.00 | 4,301 | 📉 -$995 | 100% ⚠️ |
| 23 | [macd_rsi](src/tradebot/strategies/macd_rsi.py) | 🔴 **$4.96** | 🔴 $0.94 | 2,454 | 📉 -$995 | 100% ⚠️ |
| 24 | [attrition_reversion](src/tradebot/strategies/attrition_reversion.py) | 🔴 **$4.94** | 🔴 $0.99 | 2,930 | 📉 -$995 | 100% ⚠️ |
| 25 | [rsi_reversion](src/tradebot/strategies/rsi_reversion.py) | 🔴 **$4.85** | 🔴 $0.77 | 4,464 | 📉 -$995 | 100% ⚠️ |

_Balances from a $1,000 start · bold = the strategy's better market · 🟢 profit · 🔴 loss · 💀 liquidated · ⚠️ drawdown over 50%. Trades, profit and max drawdown describe that market._
<!-- comparison:end -->

## The strategies

Where this leads next — funding harvest and funding as a positioning
signal — is in **[docs/ALTERNATIVES.md](docs/ALTERNATIVES.md)**, ranked by
evidence rather than by appeal. The cross-asset falsification test (does
any of this work on ETH?) is in
**[docs/CROSS_ASSET.md](docs/CROSS_ASSET.md)**: the drawdown reduction
replicates on a second asset, the return shortfall replicates too. An
assessment of Elliott waves against this repo's own evidence bar is in
**[docs/ELLIOTT_WAVES.md](docs/ELLIOTT_WAVES.md)**. Untried research
directions, argued from what actually constrains this project rather than
from what sounds promising, are in
**[docs/FRONTIER.md](docs/FRONTIER.md)**.

Each strategy has its own section in **[docs/STRATEGIES.md](docs/STRATEGIES.md)** —
what it is, how it works, and the principles it rests on — ordered best to
worst, with citations. The literature survey behind them is in
[docs/RESEARCH.md](docs/RESEARCH.md); robustness testing (walk-forward,
parameter frontiers, Monte Carlo windows) is in
[docs/VALIDATION.md](docs/VALIDATION.md).

The one-line summary of all twenty-five results: **every strategy that
makes money decides *how much* to hold; every strategy that tries to
predict *what happens next* loses.** On 5-minute bars, after fees, sizing
wins and forecasting loses.

**Nothing is deleted.** Unprofitable strategies stay registered as
documented negative results: knowing that the Kyle/VPIN flow followers,
the minority-game oracle and the fictitious-play machine all lose to fees
on 5m bars is a finding, and keeping them in the table stops the same
ideas being re-tried blind. Explore variants with
`python scripts/experiment.py` (see `frontier`, `horizons`,
`walkforward`) rather than by editing a registered strategy's defaults,
so the comparison table stays a stable record.

## Charts

Every run produces a chart (price with trade markers, balance curve vs a
hold benchmark, drawdown, results box). A curated set is committed; the
rest regenerate with `tradebot run` into `reports/charts/`.

**The best strategy, on 5x futures** — $1,000 → $156K where buy-and-hold
is liquidated:

![kelly_regime_v4 on futures](reports/charts/kelly_regime_v4__futures_5x__1000.png)

**The benchmark it has to beat, on spot** — note the 84% drawdown:

![buy_and_hold on spot](reports/charts/buy_and_hold__spot__1000.png)

**All strategies' balance curves** (spot, $1,000 start; the palette is
capped at eight series per chart, so the group is split into parts):

![all strategies, spot](reports/charts/_all__spot__1000_part1.png)

**Monte Carlo stress test** — 40 random windows, each strategy evaluated
on identical windows against buy-and-hold
([analysis](docs/VALIDATION.md)):

![stress test, futures](reports/stress/stress_futures.png)

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
paper-tested is the class you deploy.

**Ready-made spot adapters ship in the repo**: `BinanceSpot` and
`BitstampSpot` (`src/tradebot/exchanges/`, stdlib `urllib` only, no
dependencies) plus a stateless one-cycle bot loop in
`src/tradebot/bot.py`. Both default to `dry_run=True`.

```python
from tradebot.bot import BotConfig, step
from tradebot.exchanges import BinanceSpot          # or BitstampSpot
from tradebot.registry import get_strategy

exchange = BinanceSpot(api_key, api_secret, dry_run=True)
config   = BotConfig(symbol="BTCUSDT", strategy="kelly_regime_v4")
result   = step(exchange, config, get_strategy(config.strategy))
```

`scripts/live_bot.py` runs one cycle from environment credentials and
dry-runs unless you pass `--live`. CI proves the top-three strategies
compute the **identical target** from paged exchange data and from the
contiguous backtest frame — checked bar for bar across 30 consecutive
candles — that paging is lossless, and that neither adapter ever hands a
strategy the forming candle.

> ⚠️ **Fees decide this.** Every table here assumes a 0.10% taker fee,
> and the break-even is **0.104%** — the published spot edge lives
> entirely inside that margin. At Bitstamp's 0.40% entry tier no strategy
> here beats buy-and-hold ($29.5K vs $65.8K for `kelly_regime_v4`);
> climbing to its $5M/30d taker tier still misses by 4%. Tuning around it
> fails walk-forward: 28 of 32 configurations beat holding in-sample,
> **0 of 28** out-of-sample. Reproduce with `scripts/fee_study.py`;
> analysis in
> [docs/LIVE.md](docs/LIVE.md#read-this-before-trading-bitstamp-spot-at-the-entry-fee-tier).

Setup, cold-start cost and the honest list of what live will *not* match:
**[docs/LIVE.md](docs/LIVE.md)**.

For a custom venue, `tradebot.live` is the lower-level extraction point:

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
