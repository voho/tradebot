# Running a strategy live (Binance / Bitstamp, spot)

The strategies are pure decision functions over (candles, account
state), so the class that was paper-tested is the class that trades.
This document covers the adapters, the cold-start cost, and the
verification that a bot fed **exchange data** makes the **same decisions**
the backtest validated.

> **Spot only.** Long or flat, no leverage, no shorting — so a live bot
> cannot be liquidated. The headline futures numbers in the comparison
> table are *not* what this path reproduces; see "What live will not
> match" below.

## Quick start

```python
from tradebot.bot import BotConfig, step
from tradebot.exchanges import BinanceSpot          # or BitstampSpot
from tradebot.registry import get_strategy

exchange = BinanceSpot(api_key, api_secret, dry_run=True)   # dry run first
config   = BotConfig(symbol="BTCUSDT", strategy="kelly_regime_v4")
strategy = get_strategy(config.strategy)

result = step(exchange, config, strategy)   # call once per closed candle
print(result.target, result.reason)
```

`step()` is one full cycle: page in the candle history, read balances,
ask the strategy for a target, and place only the *delta* order. It is
**stateless** — the position lives in the exchange balance, not in
memory — so a restarted bot resumes exactly where it left off.

Both adapters default to **`dry_run=True`**, which computes and logs the
order without sending it. That default is covered by a test.

## The adapters

| | Binance | Bitstamp |
|---|---|---|
| class | `BinanceSpot` | `BitstampSpot` |
| symbol | `BTCUSDT` | `btcusd` |
| candles | `GET /api/v3/klines` | `GET /api/v2/ohlc/{pair}/` |
| max per request | 1000 | 1000 |
| balances | `GET /api/v3/account` (signed) | `POST /api/v2/balance/` (signed) |
| order | `POST /api/v3/order` | `POST /api/v2/{buy,sell}/market/{pair}/` |
| auth | HMAC-SHA256 over the query string, `X-MBX-APIKEY` | HMAC-SHA256 over a canonical string, `X-Auth*` headers |
| **taker fee** | **0.10%** | **0.40%** entry tier |

Dependencies: **none** — both use `urllib` from the standard library, so
a bot deploys with exactly what the backtester needs.

**The fee difference matters more than the API difference.** Every
result in this repo assumes a 0.10% taker fee. Bitstamp's entry tier is
4x that, which is enough to change conclusions for anything that trades
often. The leading strategies trade ~150 times in nine years, so they
tolerate it; the losing high-turnover strategies would get worse. Set
`taker_fee` to your actual tier — it feeds the strategy's own fee
awareness through `MarketSpec`.

Bitstamp is also the venue the committed dataset comes from, so
backtest and live see the same price series with no venue basis.

## Cold start: how much history a bot must fetch

The leading strategies use slow regime anchors, so they need a long
warmup — and a venue only returns 1000 candles per request:

| strategy | warmup (5m bars) | ≈ days | API calls to warm up |
|---|---|---|---|
| `kelly_regime_v4` | 23,050 | 80 | **24** |
| `kelly_regime_v3` | 28,810 | 100 | **30** |
| `kelly_regime_v2` | 28,810 | 100 | **30** |

`Exchange.fetch_history()` does this paging automatically, walking
backwards with `end_ms` and stitching the pages. At Binance's rate
limits, ~30 calls is a few seconds. This is a **one-time** cost per
process start; a long-running bot then needs one call per candle.

A test asserts the cold start stays under 40 calls, so a strategy whose
warmup silently balloons will fail CI rather than surprise a deployment.

## Verification: exchange data == backtest data

This is the check that matters, and it runs in CI
(`tests/test_exchanges.py`):

1. **Pagination is lossless.** Fetching 29,000 bars in 1000-bar pages
   reproduces the underlying series exactly — same index, same closes,
   no duplicates, no gaps.
2. **Decisions match, bar for bar.** For each of the top three
   strategies, the venue clock is stepped forward 30 consecutive candles;
   at every one, the bot re-pages its whole history and its target must
   equal the one the backtest engine computed at that same bar. One
   matching bar would prove little — page boundaries land at a different
   offset on each new candle, so a stitching bug appears at some
   alignments and not others. A separate test asserts the replay venue
   never serves a bar past its clock, so the parity checks cannot pass
   for the wrong reason.
3. **The forming candle never leaks.** Both adapters drop the still-open
   bar. A strategy that sees a partial candle is reading the future —
   this is the single most common way live bots differ from their
   backtests, and it is tested directly.
4. **The bot loop behaves.** It refuses to trade without enough history,
   respects its rebalance deadband (no order when already at target),
   never targets outside [0, 1] on spot, and never sells more base than
   it holds.

## What live will *not* match

Stated plainly, because the gap between backtest and production is where
money is lost:

- **No funding rates, and futures aren't covered here.** The $156K
  futures headline comes from a 5x leveraged simulation without funding
  costs. This live path is spot; expect spot-like returns
  (`kelly_regime_v4`: $66.8K → see the comparison table), not the
  leveraged number.
- **Market orders, not limit orders.** Slippage is real and is not in
  the fee figure. The backtester supports `--slippage-bps`; run your
  parameters through it before trusting the numbers.
- **No partial fills or rejected orders.** `place_market_order` assumes
  the fill happens. Production code should reconcile against the venue's
  fill report rather than assuming.
- **Exchange minimum notional.** `BotConfig.min_notional` (default $10)
  blocks dust orders; set it to your venue's actual minimum.
- **One asset, one decade, one venue.** Every caveat in
  [VALIDATION.md](VALIDATION.md) still applies.

## Suggested rollout

1. `dry_run=True` for at least a full market cycle of candles; compare
   the logged decisions against `tradebot run` on the same period.
2. Live with an amount you are willing to lose entirely, with
   `min_notional` set high enough that fees stay a small share.
3. Only then size up — and re-run `scripts/stress_test.py` first, since
   the drawdowns in that report are what a live account will actually
   feel.
