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

## Custom venues and 3Commas

For a venue without a ready-made adapter, `tradebot.live` is the
lower-level extraction point:

```python
from tradebot.broker import MarketSpec
from tradebot.live import LiveAccount, compute_signal
from tradebot.registry import get_strategy

strategy = get_strategy("kelly_regime_v4")

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

The timing contract is identical to the backtest — decide on bar close,
act at the next open — and `tests/test_live.py` proves parity: walking
the data bar-by-bar through `compute_signal` reproduces exactly the
decisions the backtester filled, for every registered strategy on both
markets.

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

**2. The break-even fee is 0.104%, and no Bitstamp taker tier reaches it.**

Bisected against holding on the full period, spot: `kelly_regime_v4`
beats buy-and-hold below a **0.104%** taker fee and loses above it.
Against Bitstamp's volume ladder:

| 30-day volume | taker | `kelly_regime_v4` | vs holding |
|---|---|---|---|
| < $10k (entry) | 0.40% | $29.5K | −55% |
| ~$100k–$1M | 0.25% | $44.4K | −33% |
| > $1M | 0.20% | $50.8K | −23% |
| **> $5M** | **0.12%** | **$63.2K** | **−4%** |
| > $5M, **maker** | **0.03%** | **$80.9K** | **+22%** |

Climbing tiers helps enormously — a −55% gap becomes −4% — and **still
does not cross the line**. The top attainable taker rung sits just above
break-even. The only rung that clears it is the *maker* fee, which is a
change of order type, not of tier.

Volume is not the obstacle it first appears. The comparison table's "174
trades" counts round-trip *episodes*; the strategy actually places
**1,056 fills** over the period, one every ~3 days, sustaining a median
trailing-30-day volume of **2.1x account equity** with only 3% of days
idle. So $5M of 30-day volume needs roughly a **$2.4M account** — large,
but the turnover pattern would genuinely hold the tier rather than
lapsing between bursts.

Two things still stand in the way of the maker row. It needs that $2.4M
*before* the volume qualifies you, and you pay entry rates while building
it. And a strategy acting on a regime flip has to post a limit order and
wait: fills stop being guaranteed, chasing a missed one can cost more
than the fee saved, and **the backtester models none of that**. Treat
+22% as an upper bound on a path this repo has not validated.

**The venue is the cheaper lever than the tier.** Binance spot's standard
0.10% taker is already below break-even with no volume requirement at all
($66.8K vs $66.0K, +1.1%), and 0.075% with the BNB discount gives +8.3%.
Both are thin margins — but they are available at any account size, which
$5M/30d is not.

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

### Fewer trades, or better trades? Both fail the same way

Two obvious remedies, both tested, both failing the same out-of-sample
check.

First, **where the fees actually go** (`fee_study.py fills`). Of 1,056
fills, only 174 are entries and 174 are exits — **708 are resizes inside
an already-open position**, carrying **60% of all fees**. That is the
volatility-targeting machinery, and on spot its upside is unreachable
anyway because exposure caps at 1.0. So the obvious target is the resize,
not the round trip.

**Fewer trades.** Going fully binary — all-in or all-out, no resizes at
all — does produce full-period winners: majority-vote binary returns
$88.4K against holding's $65.8K, and with a 14-day minimum hold $79.2K.
Both **lose out-of-sample** ($2.9K and $3.4K against holding's $3.8K).

**Better trades.** Requiring *unanimous* anchor agreement instead of a
majority — fewer, higher-conviction entries — is worse on every measure
($20.2K full period, $1.8K out-of-sample). Conviction filtering removes
the trades that carried the edge along with the ones that cost fees.

Combined with the earlier deadband, hysteresis-band and decision-clock
sweeps, that is four independent turnover-reduction mechanisms, all with
the same signature: in-sample winners, out-of-sample losers.

### Leverage: the one that works, and not for the reason you'd expect

**Leverage does not make the fee cheaper.** Fees are charged on notional,
so a levered position pays proportionally more of them — the drag scales
with size exactly as the returns do. On the single full-history path,
levered `kelly_regime_v4` never catches unlevered spot holding at 0.40%:

| strategy's `max_leverage` | final | max DD | vs spot holding |
|---|---|---|---|
| 2x | $57.4K | 43.2% | −13% |
| 3x | $59.2K | 43.9% | −10% |
| **4x** | **$60.3K** | 43.9% | −8% |
| 6x | $60.3K | 43.9% | −8% |

It saturates around 4x — above that the volatility target rarely asks for
more — and the whole 2–6x range lands within 5% of itself. A flat
plateau, unlike every spot configuration above.

**The advantage is in the distribution, not that one path.** Over the
same 40 random windows the stress test uses, at 0.40%, fresh account each
window:

| config | median | worst | profitable | median DD | wipeouts |
|---|---|---|---|---|---|
| `kelly_regime_v4` cap 4x, 5x venue | **+94.3%** | **−21.3%** | **82.5%** | **27.5%** | **0/40** |
| `kelly_regime_v4` cap 2x, 5x venue | +95.2% | −21.3% | 82.5% | 27.0% | **0/40** |
| spot buy_and_hold | +48.8% | −51.0% | 72.5% | 52.7% | 0/40 |
| **3x levered buy_and_hold** | +73.6% | **−99.8%** | 52.5% | 87.2% | **14/40** |

Median window return roughly **doubles** against spot holding, the median
drawdown **halves**, the worst window is less than half as bad, and it
was never wiped out. It still only beats holding in ~50% of individual
windows — holding's right tail is fat — so the gain is concentrated in
the left tail, which is the same shape this family shows everywhere else.

Three caveats that matter more than the table:

- **Bitstamp is spot-only.** None of this is executable on the adapter in
  this repo or on Bitstamp credentials; it needs a venue offering BTC
  perpetuals.
- **Levered buy-and-hold is not the alternative it looks like.** At 3x it
  returned $194K on the full path — more than anything else here — and
  wiped out in **14 of 40** windows with an 87% median drawdown. One path
  flatters it enormously.
- Part of the apparent gain from a higher-leverage *venue* is an artifact:
  the broker's rebalance deadband is a fraction of `leverage × equity`, so
  a bigger venue leverage silently widens it and cuts turnover (849 fills
  at a 4x venue, 447 at 10x). The window medians are stable at ~95%
  across venue settings, so the conclusion holds — but the single-path
  numbers move for a reason that has nothing to do with leverage.

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
   and never targets outside [0, 1] on spot. (A fourth property — never
   selling more base than the account holds — is enforced by a clamp in
   `bot.py` itself rather than asserted by a test.)

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

## Forward paper-trading recorder (B-06)

`scripts/paper_trade.py` is a **different tool from `live_bot.py` above**,
for a different job: not "decide once and print", but "keep an honest,
persisted, forward-only track record over time" — the ledger's B-06,
the highest-priority backlog item since R-29. The premise: this repo's
2023+ holdout has been consulted so many times across the whole research
program that deflated Sharpe can no longer support a Sharpe-based claim
from it at all (docs/LEDGER.md). A forward paper record is the one piece
of evidence immune to that, because it cannot have been looked at before
it existed.

```bash
python scripts/paper_trade.py                 # run once per closed 5m candle
```

|                          | `live_bot.py`                       | `paper_trade.py`                              |
|--------------------------|--------------------------------------|------------------------------------------------|
| account                  | real Bitstamp balance (signed API)  | its own persisted virtual account (JSON)        |
| credentials              | needed for `--live` / balances      | **never needed, never read**                    |
| orders sent              | real market order with `--live`     | **never — no such code path exists**            |
| state across runs        | none (reads exchange balance)       | `reports/paper_trading/*_state.json`            |
| output                   | one printed decision                | one persisted decision + one CSV row, every run |
| benchmark                | none                                | parallel `buy_and_hold` paper account           |

It fetches the same public, unauthenticated OHLC candles `live_bot.py`
does, calls the same `tradebot.live.compute_signal` extraction point, and
executes the resulting order through `tradebot.broker.PaperBroker` — the
exact fee/rebalance code the backtest engine itself uses — against its
own `cash`/`pos`/`entry` state, not an exchange balance. It is
**structurally incapable of live trading**: the module never imports
`place_market_order` or any signed endpoint, and there is no `--live`
flag. Two files are written per strategy under `reports/paper_trading/`
(committed — see the `.gitignore` comment there; unlike everything else
under `reports/` this is not a regenerable analysis output, it *is* the
record):

- `<strategy>_bitstamp_state.json` — the persisted virtual account
  (`cash`, `pos`, `entry`, cumulative fees, the last candle acted on).
- `<strategy>_bitstamp.csv` — one append-only row per decision:
  timestamp, candle close, prior/new target, trade qty, fee, resulting
  position/cash/equity, and the reason (the strategy's raw target, or
  `INCEPTION CATCH-UP ...` — see below).

### Scheduling

Once per closed 5m candle, same cadence as `live_bot.py`:

```cron
*/5 * * * * cd /path/to/tradebot && python scripts/paper_trade.py >> reports/paper_trading/cron.log 2>&1
```

or a systemd timer with `OnCalendar=*:0/5` firing a oneshot unit that
runs the same command. It is **idempotent on the candle timestamp**: a
second invocation before the next candle closes detects
`last_candle_ts` already matches and exits 0 having changed nothing — so
an overlapping cron tick or a manual re-run cannot double-count. A
missed run costs nothing but the missed rebalance, the same property
`bot.py` documents for the live path.

### The inception catch-up, and why it exists

A cold-started account needs to be sized to what the strategy currently
wants, not just react to future changes — but 18 of this project's 23
registered strategies (the whole `kelly_regime` family included) share
one convention: `on_bar` only emits an order when its precomputed
`target` column **changes** from the previous bar, since both sides of
that comparison are pure functions of price history, independent of
account state. Running this recorder for real the first time against
live `kelly_regime_v4` on Bitstamp hit exactly this: the vote had been
latched at the same value since before the fetched window even began, so
a plain `compute_signal` call returned no order at all, and a naively
cold-started paper account would have sat at 0% exposure forever while
genuinely believing it was tracking the strategy. `inception_catchup_target()`
in `scripts/paper_trade.py` fixes this **at inception only**: if
`compute_signal` emits nothing on the very first run, it reads the
strategy's raw `target` value directly from `prepare()`'s output,
bypassing the change gate, and enters to that stance instead (clamped to
`[0, 1]` on spot, same as any order). Every later run relies on
`compute_signal` exactly as written — the catch-up never overrides a
genuine decision, only a cold start's blind spot. The identical gap
exists in `bot.py`/`live_bot.py` today (a freshly funded live account
would have the same problem) but is not fixed there, since neither file
is this recorder's to change.

### Honest limitations

- **Fills at the observed candle's close, not the next open.** The
  backtest and `live_bot.py` decide-on-close/fill-at-next-open contract
  assumes a continuously running process. This recorder runs once per
  invocation with nothing sitting on the book between runs, so it fills
  at the *current* closed candle's close — the price observed when the
  recorder happens to run, not a guaranteed next-open print. Read every
  fill as an approximation of the ideal contract, not the ideal itself.
- **Single venue, no order book, no slippage model** — Bitstamp spot
  only, filled at the OHLC candle's printed close.
- **Real fee tier** — Bitstamp's 0.40% entry taker by default
  (`--taker-fee`), never the 0.10% headline assumption.
- **The idempotency check re-fetches the full warmup window every run**
  (24 API calls for `kelly_regime_v4`) purely to read the latest candle's
  timestamp, even when nothing has changed. Correct but not the cheapest
  possible design; a future pass could fetch a small tail window first
  and only page the full history back in when a new candle is confirmed.
- **Does not touch the 2023+ holdout.** It reads only the live public
  feed, never `data/btcusd_spot_5m.csv.gz` — the whole point of B-06 is
  that this record cannot have been looked at before it existed.
