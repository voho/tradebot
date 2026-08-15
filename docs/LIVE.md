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

There is a ready-made runner that keeps credentials in the environment
and **dry-runs unless you pass `--live`**:

```bash
export BITSTAMP_API_KEY=...        # key needs the Trade permission
export BITSTAMP_API_SECRET=...

python scripts/live_bot.py --venue bitstamp --symbol btcusd --taker-fee 0.004
python scripts/live_bot.py --venue bitstamp --symbol btcusd --taker-fee 0.004 --live
```

Public candles need no credentials, so the dry run works with nothing set
— useful for confirming the strategy agrees with the backtest on live
data before an account is involved. Run it once per closed 5m candle from
cron or a systemd timer.

Or drive it yourself:

```python
from tradebot.bot import BotConfig, step
from tradebot.exchanges import BitstampSpot         # or BinanceSpot
from tradebot.registry import get_strategy

exchange = BitstampSpot(api_key, api_secret, dry_run=True)   # dry run first
exchange.taker_fee = 0.004                                   # your real tier
config   = BotConfig(symbol="btcusd", strategy="kelly_regime_v4")
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

Bitstamp is also the venue the committed dataset comes from, so
backtest and live see the same price series with no venue basis.

## Read this before trading Bitstamp spot at the entry fee tier

**Every result in this repo assumes a 0.10% taker fee. At Bitstamp's
0.40% entry tier, on spot, none of these strategies beat buy-and-hold.**

Full period, spot, $1,000 start
(`tradebot run --markets spot --spot-fee 0.004`):

| strategy | @0.10% | @0.40% | max DD @0.40% | Sharpe @0.40% |
|---|---|---|---|---|
| buy_and_hold | $66.0K | **$65.8K** | 84.1% | 0.95 |
| `kelly_regime_v4` | $66.8K | $29.5K | 48.7% | **1.18** |
| `kelly_regime_v3` | $65.8K | $32.9K | — | — |
| `kelly_regime_v2` | $46.4K | $22.1K | — | — |
| `kelly_regime` | $42.1K | $19.6K | — | — |

Buy-and-hold barely notices the fee — it trades **once**. The allocators
lose more than half their return, because the 0.30pp of extra cost is
paid on every rebalance and compounds across nine years.

Widening the rebalance deadband recovers some of it and not enough
(`KellyRegimeV4(deadband=...)`, spot @0.40%): 0.20 → $33.2K, 0.30 →
$35.7K, 0.45 → $19.3K, 0.60 → $28.4K. Note that curve is not monotone —
tuning the deadband to your fee tier is curve-fitting on one path, and
the best value here still loses to holding.

**What is still true at 0.40%:** the risk profile. `kelly_regime_v4`
roughly halves the drawdown (48.7% vs 84.1%) and carries a better Sharpe
(1.18 vs 0.95). You are paying return for a much smoother ride, which is
a legitimate trade — just not the trade the headline numbers describe.

Set `taker_fee` to your actual tier either way — it feeds the strategy's
own fee awareness through `MarketSpec`, so it is not cosmetic.

### Can it be tuned to beat the fee? No, and the attempt is instructive

The obvious response is to trade less. It was tried properly and it does
not work. Four independent checks, all pointing the same way:

**1. The arithmetic says no before you start.** `kelly_regime_v4`'s
*gross*, fee-free edge on spot is only **1.33x** holding ($87.8K vs
$66.1K). Its turnover at 0.40% costs a factor of 0.34, so beating holding
would need a gross edge of **2.98x**. The gap is not close, and slowing
the strategy down to save fees shrinks the gross edge too — the two move
together.

**2. Break-even fee, full period, spot:**

| strategy | beats holding below |
|---|---|
| `kelly_regime_v4` | **0.20%** taker |
| `kelly_regime_v3` | **0.10%** taker |
| raw 50-day regime filter, no vol targeting | **0.40%** taker (just barely: $64.9K vs $65.8K) |

Bitstamp's schedule puts 0.10% taker at **$20M** of 30-day volume. At any
tier a retail account will actually see, the return edge is gone.

**3. The parameter grid has no plateau.** Sweeping the raw filter over
8 lookbacks × 4 hysteresis bands at 0.40%, **10 of 32** configurations
beat holding — but scattered, not clustered. Adjacent cells swing 2–3x
($90.4K at a 40-day anchor sits next to $43.5K at 50-day). Compare the
genuine plateau behind `kelly_regime_v4`, where *every* anchor set in an
entire range moved the same way. This is the signature of noise.

**4. Walk-forward kills it outright.** Select on 2017–2022, evaluate on
2023–2026:

- **28 of 32** configurations beat holding in-sample.
- **0 of those 28** beat it out-of-sample.
- The config you would actually have picked (50-day, 3% band) returned
  **−34.5% against holding** out-of-sample.

A core-satellite version (hold a permanent core, manage only the rest)
edges past holding on the full period at a 70–85% core — by 1%, within
noise, monotone in the core fraction with no plateau, and it still loses
out-of-sample. That is not an edge; it is a dial that points at
buy-and-hold.

### What is actually on offer at 0.40%

The strategy is **profitable** — $1,000 → $29,498 is +2,850% over nine
years. It just does not beat holding, and its product is risk, not
return:

| | `kelly_regime_v4` | buy_and_hold |
|---|---|---|
| final balance | $29.5K | **$65.8K** |
| max drawdown | **48.7%** | 84.1% |
| Sharpe | **1.18** | 0.95 |
| out-of-sample max DD | **34.1%** | 54.0% |

That drawdown gap is the one thing that **does** survive out-of-sample,
which matches what the variant research found independently: for this
family the risk reduction is robust and the return improvement is not.
If you want BTC exposure without an 84% drawdown, this delivers it and
costs you return to do so. If you want to beat holding, the edge lives on
leverage — where the claim is not "more return than holding" but "holding
is liquidated in 26 of 40 random windows and this is not" — and that is
out of scope for this spot-only path.

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
