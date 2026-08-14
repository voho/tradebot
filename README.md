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
| **kelly_regime**<br>_Size growth-optimally (fractional Kelly, vol-targeted) while the crowd regime stays bullish._<br>[source](src/tradebot/strategies/kelly_regime.py) | trades 143<br>profit $41.1K<br>worst -$2,428<br>best $14.5K<br>**after $42.1K** | trades 143<br>profit $41.10M<br>worst -$2.43M<br>best $14.50M<br>**after $42.10M** | trades 143<br>profit $107.2K<br>worst -$6,782<br>best $44.9K<br>**after $108.2K** | trades 143<br>profit $107.22M<br>worst -$6.78M<br>best $44.94M<br>**after $108.22M** |
| **buy_and_hold**<br>_Buy everything on the first bar and never trade again._<br>[source](src/tradebot/strategies/buy_and_hold.py) | trades 1<br>profit $65.0K<br>worst $65.0K<br>best $65.0K<br>**after $66.0K** | trades 1<br>profit $65.04M<br>worst $65.04M<br>best $65.04M<br>**after $66.04M** | trades 1<br>profit -$982<br>worst -$982<br>best -$982<br>**after $18.05**<br>LIQUIDATED | trades 1<br>profit -$982.0K<br>worst -$982.0K<br>best -$982.0K<br>**after $18.0K**<br>LIQUIDATED |
| **champions_council**<br>_Combine the games that actually pay: Hedge over their signals, sized by fractional Kelly._<br>[source](src/tradebot/strategies/champions_council.py) | trades 131<br>profit $18.3K<br>worst -$1,150<br>best $8,520<br>**after $19.3K** | trades 132<br>profit $18.32M<br>worst -$1.15M<br>best $8.52M<br>**after $19.32M** | trades 261<br>profit $35.8K<br>worst -$2,070<br>best $18.6K<br>**after $36.8K** | trades 263<br>profit $35.77M<br>worst -$2.07M<br>best $18.57M<br>**after $36.77M** |
| **hedge_experts**<br>_No-regret Hedge blend of technical experts, each charged its own turnover._<br>[source](src/tradebot/strategies/hedge_experts.py) | trades 2,044<br>profit $12.3K<br>worst -$1,469<br>best $11.0K<br>**after $13.3K** | trades 2,159<br>profit $12.27M<br>worst -$1.47M<br>best $11.03M<br>**after $13.27M** | trades 4,103<br>profit -$742<br>worst -$15.9K<br>best $53.0K<br>**after $258** | trades 4,316<br>profit -$742.6K<br>worst -$15.87M<br>best $52.98M<br>**after $257.4K** |
| **replicator_book**<br>_Reallocate capital across trend, value and cash species with replicator dynamics on their realized fee-adjusted fitness._<br>[source](src/tradebot/strategies/replicator_book.py) | trades 713<br>profit $1,330<br>worst -$90.85<br>best $451<br>**after $2,330** | trades 717<br>profit $1.33M<br>worst -$90.8K<br>best $451.4K<br>**after $2.33M** | trades 1,427<br>profit -$989<br>worst -$681<br>best $1,847<br>**after $10.58** | trades 1,435<br>profit -$989.4K<br>worst -$680.7K<br>best $1.85M<br>**after $10.6K** |
| **universal_kelly**<br>_Universal-portfolio exposure: wealth-weighted mixture over fixed exposures, half-Kelly capped._<br>[source](src/tradebot/strategies/universal_kelly.py) | trades 9<br>profit $276<br>worst -$6.99<br>best $131<br>**after $1,276** | trades 1,529<br>profit $202.9K<br>worst -$6,509<br>best $123.9K<br>**after $1.20M** | trades 20<br>profit $227<br>worst -$15.25<br>best $129<br>**after $1,227** | trades 3,047<br>profit $3,730<br>worst -$238<br>best $4,923<br>**after $1.00M** |
| **harsanyi_crowd**<br>_Trade the belief margin over hidden market types, sized down when the trend is crowded._<br>[source](src/tradebot/strategies/harsanyi_crowd.py) | trades 91<br>profit -$112<br>worst -$13.89<br>best $7.93<br>**after $888** | trades 91<br>profit -$111.9K<br>worst -$13.9K<br>best $7,926<br>**after $888.1K** | trades 178<br>profit -$571<br>worst -$73.42<br>best $34.04<br>**after $429** | trades 178<br>profit -$571.3K<br>worst -$73.4K<br>best $34.0K<br>**after $428.7K** |
| **overshoot_fade**<br>_Fade forced-liquidation overshoots once the aggressive flow driving them is exhausted._<br>[source](src/tradebot/strategies/overshoot_fade.py) | trades 189<br>profit -$338<br>worst -$52.43<br>best $21.29<br>**after $662** | trades 189<br>profit -$338.4K<br>worst -$52.4K<br>best $21.3K<br>**after $661.6K** | trades 341<br>profit -$966<br>worst -$115<br>best $71.14<br>**after $33.52** | trades 341<br>profit -$966.5K<br>worst -$115.2K<br>best $71.1K<br>**after $33.5K** |
| **camouflage_flow**<br>_Follow persistent informed order flow recovered from bars via Bulk Volume Classification._<br>[source](src/tradebot/strategies/camouflage_flow.py) | trades 802<br>profit -$452<br>worst -$40.11<br>best $73.99<br>**after $548** | trades 802<br>profit -$452.4K<br>worst -$40.1K<br>best $74.0K<br>**after $547.6K** | trades 1,044<br>profit -$999<br>worst -$205<br>best $101<br>**after $0.99** | trades 1,676<br>profit -$1.00M<br>worst -$204.8K<br>best $100.9K<br>**after $127** |
| **stealth_trend**<br>_Follow momentum only when it prints on deep, high-participation bars where informed flow hides._<br>[source](src/tradebot/strategies/stealth_trend.py) | trades 1,605<br>profit -$535<br>worst -$40.68<br>best $109<br>**after $465** | trades 1,605<br>profit -$535.5K<br>worst -$40.7K<br>best $108.7K<br>**after $464.5K** | trades 248<br>profit -$1,000<br>worst -$144<br>best $37.26<br>**after $0.38** | trades 1,285<br>profit -$1.00M<br>worst -$144.4K<br>best $37.3K<br>**after $0.84** |
| **flow_regime**<br>_Combine the two sides of the microstructure game: follow flow, but fade liquidation overshoots._<br>[source](src/tradebot/strategies/flow_regime.py) | trades 1,184<br>profit -$553<br>worst -$52.56<br>best $102<br>**after $447** | trades 1,184<br>profit -$553.3K<br>worst -$52.6K<br>best $101.6K<br>**after $446.7K** | trades 467<br>profit -$999<br>worst -$107<br>best $50.52<br>**after $0.80** | trades 2,135<br>profit -$1.00M<br>worst -$106.6K<br>best $50.5K<br>**after $1.46** |
| **game_council**<br>_Combination of games: no-regret Hedge allocation over the game strategies' own signals._<br>[source](src/tradebot/strategies/game_council.py) | trades 2,541<br>profit -$716<br>worst -$14.04<br>best $22.92<br>**after $284** | trades 2,541<br>profit -$716.0K<br>worst -$14.0K<br>best $22.9K<br>**after $284.0K** | trades 2,494<br>profit -$998<br>worst -$55.75<br>best $17.63<br>**after $2.00** | trades 4,978<br>profit -$1.00M<br>worst -$55.8K<br>best $17.6K<br>**after $34.66** |
| **minority_oracle**<br>_Trade the abstention-filtered vote of a grand-canonical minority game trained online on binarized returns._<br>[source](src/tradebot/strategies/minority_oracle.py) | trades 9,039<br>profit -$947<br>worst -$2.93<br>best $3.00<br>**after $53.36** | trades 9,039<br>profit -$946.6K<br>worst -$2,930<br>best $2,995<br>**after $53.4K** | trades 7,065<br>profit -$996<br>worst -$10.12<br>best $11.19<br>**after $3.83** | trades 15,493<br>profit -$1.00M<br>worst -$10.1K<br>best $11.2K<br>**after $4.29** |
| **rsi_reversion**<br>_Mean-reversion: buy oversold dips (RSI < 30), exit on recovery; mirror short overbought on futures._<br>[source](src/tradebot/strategies/rsi_reversion.py) | trades 4,464<br>profit -$995<br>worst -$159<br>best $133<br>**after $4.85** | trades 5,713<br>profit -$1.00M<br>worst -$159.0K<br>best $133.5K<br>**after $365** | trades 2,264<br>profit -$999<br>worst -$346<br>best $246<br>**after $0.77** | trades 2,638<br>profit -$1.00M<br>worst -$345.7K<br>best $246.1K<br>**after $3.41**<br>LIQUIDATED |
| **game_switch**<br>_Best-respond to whichever game the market is currently playing by trading only history states with significant conditional drift._<br>[source](src/tradebot/strategies/game_switch.py) | trades 6,672<br>profit -$995<br>worst -$11.38<br>best $9.00<br>**after $5.00** | trades 13,899<br>profit -$1.00M<br>worst -$11.4K<br>best $9,001<br>**after $4.95** | trades 6,449<br>profit -$999<br>worst -$17.26<br>best $21.63<br>**after $1.00** | trades 11,485<br>profit -$1.00M<br>worst -$17.3K<br>best $21.6K<br>**after $1.00** |
| **regret_grid**<br>_Regret-matching+ over a position grid: correlated-equilibrium play against the market._<br>[source](src/tradebot/strategies/regret_grid.py) | trades 3,461<br>profit -$995<br>worst -$46.98<br>best $13.29<br>**after $5.00** | trades 10,834<br>profit -$1.00M<br>worst -$47.0K<br>best $13.3K<br>**after $5.00** | trades 1,389<br>profit -$999<br>worst -$136<br>best $33.17<br>**after $1.00** | trades 4,027<br>profit -$1.00M<br>worst -$136.2K<br>best $33.2K<br>**after $1.00** |
| **macd_rsi**<br>_Trend + timing combo: trade RSI pullback recoveries only in the direction of the MACD trend._<br>[source](src/tradebot/strategies/macd_rsi.py) | trades 2,454<br>profit -$995<br>worst -$44.90<br>best $42.97<br>**after $4.96** | trades 5,503<br>profit -$1.00M<br>worst -$44.9K<br>best $43.0K<br>**after $5.00** | trades 1,239<br>profit -$999<br>worst -$150<br>best $110<br>**after $0.94** | trades 2,120<br>profit -$1.00M<br>worst -$149.6K<br>best $110.2K<br>**after $0.98** |
| **macd_cross**<br>_Trend-following: long when MACD crosses above its signal line, flat/short on the cross below._<br>[source](src/tradebot/strategies/macd_cross.py) | trades 4,301<br>profit -$995<br>worst -$46.33<br>best $42.19<br>**after $4.99** | trades 7,945<br>profit -$1.00M<br>worst -$46.3K<br>best $42.2K<br>**after $5.00** | trades 1,464<br>profit -$999<br>worst -$259<br>best $566<br>**after $1.00** | trades 3,306<br>profit -$1.00M<br>worst -$258.6K<br>best $566.0K<br>**after $0.73** |
| **tft_trend**<br>_Repeated-game trend truce: hold while the market cooperates, forgive one defection, punish two._<br>[source](src/tradebot/strategies/tft_trend.py) | trades 2,538<br>profit -$995<br>worst -$92.03<br>best $36.26<br>**after $4.99** | trades 5,939<br>profit -$1.00M<br>worst -$92.0K<br>best $36.3K<br>**after $4.97** | trades 659<br>profit -$999<br>worst -$125<br>best $120<br>**after $1.00** | trades 1,583<br>profit -$1.00M<br>worst -$125.2K<br>best $119.8K<br>**after $0.94** |
| **attrition_reversion**<br>_Fade deviations from an inventory-shifted fair value; quit when waiting costs exceed the prize._<br>[source](src/tradebot/strategies/attrition_reversion.py) | trades 2,930<br>profit -$995<br>worst -$90.17<br>best $28.72<br>**after $4.94** | trades 7,221<br>profit -$1.00M<br>worst -$90.2K<br>best $28.7K<br>**after $4.97** | trades 1,176<br>profit -$999<br>worst -$457<br>best $103<br>**after $0.99** | trades 2,034<br>profit -$1.00M<br>worst -$456.9K<br>best $103.1K<br>**after $0.95** |
<!-- comparison:end -->

## Built-in strategies

Twenty strategies, grouped by what they are. Each file's docstring carries
the full idea plus its citations; the literature survey behind them is in
[docs/RESEARCH.md](docs/RESEARCH.md), and the walk-forward validation of
the leaders — including where they *lose* — is in
[docs/VALIDATION.md](docs/VALIDATION.md).

**Baselines** — [buy_and_hold](src/tradebot/strategies/buy_and_hold.py)
(the benchmark, and a leverage stress test: it liquidates on 5x),
[macd_cross](src/tradebot/strategies/macd_cross.py),
[rsi_reversion](src/tradebot/strategies/rsi_reversion.py),
[macd_rsi](src/tradebot/strategies/macd_rsi.py).

**Allocators — how much to hold** (the ones that make money; see
VALIDATION.md for why sizing beats prediction here):
[kelly_regime](src/tradebot/strategies/kelly_regime.py) — fractional-Kelly
volatility-targeted exposure, gated by a multi-horizon crowd-regime vote
(Bell & Cover; Cardaliaguet & Lehalle);
[hedge_experts](src/tradebot/strategies/hedge_experts.py) — no-regret
Hedge over ten technical experts (Freund & Schapire);
[replicator_book](src/tradebot/strategies/replicator_book.py) — replicator
dynamics across chartist/fundamentalist/cash species (Taylor & Jonker;
Lux & Marchesi); [universal_kelly](src/tradebot/strategies/universal_kelly.py)
— Cover's universal portfolio over an exposure grid.

**Microstructure games — reading informed flow from bars:**
[camouflage_flow](src/tradebot/strategies/camouflage_flow.py) (Kyle
insider flow via Bulk Volume Classification),
[stealth_trend](src/tradebot/strategies/stealth_trend.py) (momentum gated
by Amihud price impact), [overshoot_fade](src/tradebot/strategies/overshoot_fade.py)
(fade forced-liquidation overshoots — Brunnermeier & Pedersen).

**Learning & equilibrium play:**
[regret_grid](src/tradebot/strategies/regret_grid.py) (regret matching →
correlated equilibrium, Hart & Mas-Colell),
[game_switch](src/tradebot/strategies/game_switch.py) (fictitious play over
history states), [minority_oracle](src/tradebot/strategies/minority_oracle.py)
(a grand-canonical minority game trained online).

**Repeated games & beliefs:**
[tft_trend](src/tradebot/strategies/tft_trend.py) (generous tit-for-tat
truce with the trend — Axelrod),
[attrition_reversion](src/tradebot/strategies/attrition_reversion.py)
(reservation-price reversion with war-of-attrition exits — Avellaneda &
Stoikov; Maynard Smith),
[harsanyi_crowd](src/tradebot/strategies/harsanyi_crowd.py) (Bayesian
belief over hidden market types with a crowding haircut).

**Combinations — games of games:**
[champions_council](src/tradebot/strategies/champions_council.py) (Hedge
over the profitable allocators, risk-shaped by fractional Kelly),
[game_council](src/tradebot/strategies/game_council.py) (Hedge over the
seven game-theoretic members),
[flow_regime](src/tradebot/strategies/flow_regime.py) (flow followers with
a liquidation-event override and a belief veto).

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
