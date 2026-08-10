# gtbot — a game-theoretic BTCUSD 5-minute trading bot

A trading system for BTCUSD 5-minute bars built on repeated-game and
market-microstructure theory, with a backtester, a paper-trading loop, and an
evaluation harness designed to try to *disprove* its own results.

**$1,000 at 5x leverage, over 1.43 years, averaged across 6 held-out seeds:**

| | retail fees | VIP 6 | VIP 9 |
|---|---:|---:|---:|
| long/short | **+$4** | **+$167** | **+$419** |
| long-only | **+$4** | **+$82** | **+$197** |
| long/short, max size every trade | +$14 | +$412 | **+$1,764** |

Worst of all 72 runs: −$143. No liquidations. Full table and the caveats that
matter: [What $1,000 at 5x actually does](#what-1000-at-5x-actually-does).

> **Read this first.** The numbers below were produced on a calibrated
> agent-based market simulator, because the environment this was developed in
> had no network access to any exchange. They are evidence that the machinery
> works and that its edge is not a data-mining artefact. They are **not**
> evidence about the real BTCUSD market. `gtbot fetch` downloads real bars and
> every command runs on them unchanged — see [Running it on real data](#running-it-on-real-data).

---

## The idea

Most of a 5-minute price move is information. A small part is not: it is the
temporary displacement left behind when someone had to trade in a hurry and the
market maker on the other side has not yet worked the inventory off. That
displacement is mechanical, it decays in minutes, and it is the only part of the
move that is predictable.

The bot is organised around identifying it, and every component is a piece of
game theory rather than a pattern-matching rule:

| Component | Game | What it does |
|---|---|---|
| `game/experts.py` | Signalling, Stackelberg | Each **expert** is a hypothesis about *who* is on the other side and what they want |
| `game/regret.py` | Repeated game, no-regret learning | **Hedge** over `{+eᵢ, −eᵢ, flat}`, learning each expert's weight *and sign* from realised payoffs |
| `game/equilibrium.py` | Zero-sum vs. "nature" | Position size is the **maximin** solution of a game against an adversary who picks the true edge from its confidence interval |
| `risk/` | — | Volatility targeting, drawdown governor, exposure caps |

### The players

The two that matter are two independent readings of the *same* latent quantity:

- **`impact_overshoot`** reads it from the price side. Fit Kyle's
  `Δp = λ·(signed volume)` on a trailing window; the residual is the part of the
  move that order flow cannot explain, and under Kyle's model the permanent
  component is exactly the flow-explained part — so the residual is transient by
  construction.
- **`inventory_skew`** reads it from the flow side. An Avellaneda–Stoikov maker
  quotes around a reservation price displaced in proportion to its inventory, and
  that displacement unwinds as the maker hedges. Inventory is not observable but
  is the decayed negative of cumulative taker flow, which is.

`transient_dislocation` averages the two. This is the single most important
design decision in the repository: **either reading alone is worth 2–4 bp per
trade, and the pair is worth ~11 bp.** Averaging two unbiased readings of one
latent variable is estimation theory, not curve fitting — and the payoff is
insensitive to the split (50/50 and 60/40 differ by ~10%), which is how you tell
the difference.

The rest of the roster (`sweep_fader`, `informed_continuation`, `trend_rider`,
`mean_reverter`, `inventory_skew_slow`) are genuine competing hypotheses. The
learner is free to down-weight or invert them, and mostly does.

### Why no-regret learning

The meta-learner's guarantee is that its cumulative payoff stays within
`O(√(T log N))` of the best fixed action in hindsight — with **no distributional
assumption about the market**. That is the right property for a component that
has to survive regime change. The action set includes each expert's negation, so
the bot discovers signs from data rather than inheriting them from the author's
priors; and it includes `flat`, so "don't trade" is a first-class action.

Four things about this learner were wrong in early versions and are worth
knowing, because each silently destroyed the edge:

1. **Learning rate.** Hedge's optimal `η` is `√(8 ln K / T)` ≈ 0.03 here. At
   0.35 the learner put weight −0.76 on a *pure-noise* expert.
2. **Memory.** Uniform mixing caps memory at `1/mix` updates. Resolving an
   information coefficient of 0.02 needs O(20k) observations; `mix=0.004` gave
   it 250 and it chased noise.
3. **Objective mismatch.** The strategy only trades the tail, but the learner
   scores *average* payoff. The two best experts differ by 3% on average payoff
   and by 2.5× on tail payoff. Gating each signal to zero inside 2σ makes an
   expert's average payoff *be* its tail payoff.
4. **Scale.** Payoff scales with signal amplitude, so an expert that is merely
   louder beats an equally skilful quieter one. Every expert is standardised
   before it reaches the action set.

### Why the sizer sometimes refuses to trade

The sizing game's adversary picks the true edge from `[μ̂ − k·SE, μ̂]`, so the
ambiguity set contracts as evidence accumulates. When the estimated edge does
not clear the round-trip cost with statistical confidence, the maximin size is
zero and **the bot does not trade**. On a structureless random walk it takes
zero positions — not a small loss, zero. That is the behaviour you want, and it
is why the negative controls below read exactly `0.00`.

---

## Results

Held-out evaluation: hyperparameters were chosen on seeds 0–3 in
`scripts/search.py`; every number here comes from seeds 100–105, which were
never looked at during development. 150,000 bars ≈ 1.4 years per seed.
Execution is taker-in / maker-out (see [Execution](#execution-matters-more-than-the-signal)).

Reproduce with `python scripts/evaluate.py`.

### Performance depends on the fee tier more than anything else

| Fee tier | Round trip | Net Sharpe | worst seed | CAGR | Vol | Max DD | Trades/yr | Gross Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| retail | 6.65 bp | +0.21 | +0.00 | +0.28% | 0.45% | 0.33% | 23 | +0.51 |
| VIP 3 | 4.95 bp | +0.59 | −1.44 | +3.85% | 2.14% | 1.86% | 119 | +1.26 |
| **VIP 6** | **3.85 bp** | **+1.74** | +0.00 | **+10.35%** | 4.06% | 2.74% | 287 | +2.68 |
| **VIP 9** | **2.05 bp** | **+2.90** | **+1.74** | **+25.10%** | 7.14% | 4.49% | 487 | +3.74 |
| market maker | 2.15 bp | +2.85 | +1.63 | +24.21% | 6.99% | 4.56% | 471 | +3.75 |

The gross edge is 5–9 bp per trade over a 3-bar hold. A retail round trip costs
6.65 bp, so at retail fees **this strategy is not viable** and the sizer
correctly says so by trading 25 times a year instead of 500. This is not a
disappointing detail to be buried; it is the main practical finding. A 5-minute
mean-reversion strategy is a fee-tier business.

### What $1,000 at 5x actually does

Mean over the 6 held-out seeds, 1.43 years each. `robust` is the default (each
trade sized by the equilibrium sizer's conviction); `fixed` takes the full 5x on
every signal, which is what most people mean by "trading at 5x".

| mode | sizing | tier | final $ | P&L | return | CAGR | max DD $ | worst bar | trades/yr | in mkt | fees $ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long/short | robust | retail | 1,004 | **+$4** | +0.4% | +0.3% | 3 | −1 | 16 | 0.0% | 6 |
| long/short | robust | VIP 6 | 1,167 | **+$167** | +16.7% | +11.4% | 30 | −12 | 201 | 0.1% | 75 |
| long/short | robust | VIP 9 | 1,419 | **+$419** | +41.9% | +27.8% | 55 | −20 | 341 | 0.3% | 99 |
| long-only | robust | retail | 1,004 | **+$4** | +0.4% | +0.3% | 2 | −1 | 7 | 0.0% | 3 |
| long-only | robust | VIP 6 | 1,082 | **+$82** | +8.2% | +5.7% | 37 | −14 | 105 | 0.1% | 45 |
| long-only | robust | VIP 9 | 1,197 | **+$197** | +19.7% | +13.4% | 57 | −22 | 174 | 0.1% | 56 |
| long/short | fixed | retail | 1,014 | +$14 | +1.4% | +1.0% | 25 | −8 | 16 | 0.0% | 40 |
| long/short | fixed | VIP 6 | 1,412 | +$412 | +41.2% | +27.4% | 130 | −52 | 202 | 0.1% | 235 |
| long/short | fixed | VIP 9 | 2,764 | **+$1,764** | +176.4% | +103.9% | 231 | −99 | 343 | 0.3% | 465 |
| long-only | fixed | retail | 1,003 | +$3 | +0.3% | +0.2% | 15 | −5 | 8 | 0.0% | 18 |
| long-only | fixed | VIP 6 | 1,179 | +$179 | +17.9% | +12.2% | 122 | −42 | 105 | 0.1% | 128 |
| long-only | fixed | VIP 9 | 1,526 | +$526 | +52.6% | +34.5% | 152 | −66 | 175 | 0.1% | 152 |

**Worst single seed of all 72 runs: $857, a $143 loss.** No liquidations; the
worst bar anywhere consumed 8.9% of the distance to liquidation.

Five things this table says that a Sharpe ratio does not:

1. **At retail fees you make roughly nothing** — $3 to $14 on $1,000 over 17
   months. The fee tier, not the signal, is the binding constraint.
2. **Long-only gives up more than half the profit.** The signal is symmetric:
   dislocations resolve upward and downward about equally often, so refusing
   shorts discards half the opportunities and, because the fixed costs of being
   set up do not halve, rather more than half the profit.
3. **`fixed` sizing earns more and is worse.** At VIP 9 it turns $419 into
   $1,764 — and the drawdown goes from $55 to $231, with Sharpe falling. Sizing a thin edge at maximum exposure buys return with a
   worse-than-proportional increase in risk.
4. **The account is idle ~99.7% of the time.** A 5x *cap* is not 5x *exposure*:
   the robust sizer averages 1.2–1.6x while in a trade and is flat otherwise,
   which is why realised volatility is 7–10%, not 50%.
5. **Fees are a quarter of gross profit.** At VIP 9 long/short fixed, $465 of
   fees against $1,764 of net profit.

Reproduce any row with `gtbot backtest --tier vip9 --leverage 5 --deposit 1000`,
which prints this table for every mode.

### Is it a data-mining artefact?

Pooled across held-out seeds at VIP 6 (~8.5 years of bars):

| Test | Value |
|---|---:|
| Annualised Sharpe | +2.04 |
| Block-bootstrap 95% CI | [+1.42, +2.66] |
| Bootstrap `p(Sharpe ≤ 0)` | 0.0000 |
| Newey–West t-statistic | +6.07 |
| Anytime-valid 95% confidence sequence | [+0.93, +3.16] |
| Probabilistic Sharpe ratio | 1.0000 |
| **Deflated Sharpe ratio (18 trials)** | **0.2756** |

The confidence sequence is the strictest of these: it is valid at *every* sample
size simultaneously, so it stays honest under the repeated looks that
backtesting inevitably involves. It excluded zero after 130,530 bars — about
1.24 years of 5-minute data, which is a fair statement of how much history this
edge needs before anyone should believe it.

The deflated Sharpe ratio rose from **0.0011 to 0.2756** with the improvements
below. It is still not a decisive number, and it is reported rather than
omitted: benchmarked against the best of 18 configurations, a Sharpe of 2.04 now
has real headroom over what a lucky search would produce, but not overwhelming
headroom. The *mechanism* is well supported — the negative controls and the
cross-seed consistency are hard to fake. Treat VIP 9 / market-maker fees as
where this has margin to spare.

### Negative controls

The strategy claims to exploit a specific structure. Destroy the structure and
it must earn nothing:

| Data | Sharpe | Trades |
|---|---:|---:|
| Structureless random walk (same vol) | **0.00** | **0** |
| Block-bootstrapped surrogate | **0.00** | **0** |
| Simulated market | **+1.74** | 287 |

It does not merely lose less on the controls — it declines to trade at all,
because the edge estimator finds no edge and the maximin size is zero.

### Walk-forward is the weakest result

Five purged, embargoed folds of a 420,000-bar (~4 year) series, a **fresh
learner per fold**, VIP 6:

| Seed | Per-fold Sharpe | Pooled |
|---|---|---:|
| 100 | 0.00, 0.00, 0.00, +1.58, +0.44 | +0.69 |
| 101 | 0.00, +0.15, 0.00, 0.00, 0.00 | +0.07 |
| 102 | +1.24, 0.00, +2.86, 0.00, 0.00 | +1.32 |

Pooled results are positive on all three seeds, but most individual folds show
`0.00` — the strategy did not trade at all in them. That is not a failure of the
edge; it is the online learner needing roughly a year of 5-minute data before
its weights and its edge estimate are confident enough for the sizer to allocate.
An 80,000-bar fold with a cold learner is mostly warm-up.

So this test partly measures *learning speed* rather than edge, and it is the
number that most argues for caution. A deployment should warm the learner on all
available history rather than restarting it, which is what `PaperTrader.warm_up`
does.

---

## Improvements, and what they were worth

Four ideas were taken from the search-and-learning literature on
imperfect-information games — a setting that shares this problem's structure:
act under partial observation, against adversaries, where evaluating your own
performance is itself statistically hard. Each was ablated **on training seeds**
before anything was promoted; the held-out numbers above are the winners only.

| variant | Sharpe | worst seed | $1,000 → | verdict |
|---|---:|---:|---:|---|
| baseline | +2.02 | +0.70 | $1,234 | |
| + variance reduction | +2.06 | **+1.36** | $1,256 | kept |
| + re-solved exit | +2.34 | +0.75 | $1,281 | kept |
| **+ both** | **+2.65** | **+2.23** | **$1,324** | **shipped** |
| discounted regret matching | +0.11 | +0.00 | $1,013 | rejected |

### 1. AIVAT-style variance reduction — kept

The sizer allocates on `edge − k·SE`, so halving the standard error is worth as
much as doubling the edge. Most of a trade's payoff variance comes from order
flow arriving *after* entry: unpredictable at entry, therefore zero conditional
mean, therefore removable as a control variate without bias.

This is the construction [AIVAT](https://ojs.aaai.org/index.php/AAAI/article/view/11481)
uses to evaluate agents in imperfect-information games — `v̂(z) = v̂_b(z) + v̂_c(z)`,
a heuristic baseline plus zero-expectation corrections on chance events — where
it cuts the trials needed for a given claim by more than 10×. It fits this
strategy exactly, because the edge *is* the part of the move order flow does not
explain, so subtracting the flow-explained part removes noise and not signal.

It barely moved the mean and roughly doubled the worst seed (+0.70 → +1.36),
which is the signature of an estimator getting more reliable rather than more
optimistic.

### 2. Continual re-solving of the exit — kept

The exit was a fixed holding period. Now each bar re-decides against a learned
continuation value — a one-ply depth-limited search, in the spirit of DeepStack's
continual re-solving. A fixed period both pays to sit in a dislocation that has
already closed and cuts trades that are still reverting. Worth +0.32 Sharpe on
its own, and it *reduced* out-of-sample drawdown from 4.27% to 2.74%.

### 3. Anytime-valid confidence sequences — kept, for evaluation

A fixed-sample confidence interval is only valid if the sample size was fixed
before looking, which backtesting never does. A confidence sequence is valid at
every sample size at once. It reports that this edge becomes conclusive after
~130,500 bars — a concrete answer to "how much history do I need before believing
this", and a considerably more honest one than a p-value computed after the fact.

### 4. Discounted regret matching (CFR+/DCFR) — rejected

Replacing Hedge with regret-matching⁺ and
[DCFR's](https://dl.acm.org/doi/10.1609/aaai.v33i01.33011829) α=1.5/β=0
discounting is attractive on paper: no learning rate to mis-set (this bot lost
its edge twice to a mis-set η), and halving negative regret each step lets an
expert whose sign has flipped recover in tens of observations instead of tens of
thousands. On a planted-signal benchmark it *beat* Hedge (blend IC +0.048 vs
+0.037).

On the real problem it collapsed to near-zero trades at every entry threshold
tried. The reason is structural: regret matching spreads weight across every
action with positive cumulative regret, where Hedge concentrates exponentially.
With 17 actions and signals that are zero 95% of the time, the blend never
becomes decisive enough to clear an entry threshold. The implementation is kept
and unit-tested (`LearnerConfig(rule="dcfr")`) because it may well be right for a
denser action set — but it is not the default, and the benchmark that liked it
was not representative of the deployment.

**Sources:** [AIVAT (Burch, Schmid, Bowling)](https://ojs.aaai.org/index.php/AAAI/article/view/11481) ·
[Search in Imperfect Information Games (Schmid)](https://arxiv.org/pdf/2111.05884) ·
[Discounted Regret Minimization (Brown & Sandholm)](https://arxiv.org/pdf/1809.04040) ·
[AIVAT variance-reduction follow-ups](https://arxiv.org/html/2605.14261)

## Execution matters more than the signal

Three findings from the execution model, all of which cost real money to get
wrong:

**Passive entries are adversely selected.** A resting bid fills only when price
keeps falling — exactly the subset where a mean-reversion entry was wrong. A
maker-in / maker-out configuration has the lowest costs on paper (3.6 bp) and
*loses* (Sharpe −2.4), because the fill selection inverts the edge. The default
crosses the spread to enter and works the exit passively.

**The bar-level maker model is deliberately conservative.** A fill is recognised
only when the bar trades *through* the limit by a tick, as a proxy for queue
position. Touching a price is no guarantee of being filled.

**Costs are modelled explicitly**: maker/taker fees per tier, half-spread, and
square-root impact `k·σ·√(Q/V)`.

---

## Honest limitations

- **The results are on simulated data.** The simulator is calibrated to BTCUSD
  5-minute stylised facts (≈55% annualised vol, excess kurtosis > 5, slowly
  decaying volatility autocorrelation, `acf(1)` ≈ −0.03) and asserted against
  them in `tests/test_synthetic.py`. It is built from market-structure first
  principles and knows nothing about the strategy — but it is still a model, and
  a model can be exploitable in ways a real market is not.
- **The edge is thin.** 5–9 bp per trade against a 2–7 bp round trip. There is
  no configuration here that survives a large increase in costs.
- **The online learner needs data.** It converges over ~100k bars (≈1 year of
  5-minute data). Shorter samples are mostly warm-up — visible directly in the
  walk-forward table, where several cold-started folds never trade at all.
- **Contextual regimes are switched off by default.** Each context cell learns
  independently, so cells cost data; at this information coefficient a cell needs
  O(10k) observations before its weights mean anything. The machinery is there
  for deployments with more history.
- **No slippage beyond the model, no exchange outages, no funding by default**
  (funding is supported, set `CostModel.funding_bp_per_8h`).
- **Liquidation is modelled but never triggered here.** At 5x the account dies
  on a ~19.6% adverse move; the worst bar in 72 runs reached 8.9% of that. On a
  real BTC series with a genuine flash crash the margin is thinner than these
  numbers suggest, and `fixed` sizing at 5x is the configuration that would
  find out first.
- **A $1,000 account has no market impact**, so these returns are scale-free up
  to roughly six figures. Beyond that the square-root impact term starts to
  matter and the per-trade edge erodes.

---

## Running it on real data

```bash
pip install -e .

# 1. real bars (Binance is preferred: its klines carry taker-buy volume,
#    which is what makes the order-flow features directly observable)
gtbot fetch --exchange binance --symbol BTCUSDT --interval 5m --days 730 \
            --out data/btcusdt_5m.csv

# 2. backtest at your actual fee tier
gtbot backtest --data data/btcusdt_5m.csv --tier vip6

# 3. out-of-sample, purged and embargoed
gtbot walkforward --data data/btcusdt_5m.csv --folds 6

# 4. full markdown report across every fee tier
gtbot report --data data/btcusdt_5m.csv --out report.md

# 5. paper trading (replays bars through the live decision path)
gtbot paper --data data/btcusdt_5m.csv
```

Venues without taker-buy volume (Coinbase, OKX) still work — the schema layer
falls back to a bar-level Lee-Ready estimate and flags the frame — but the flow
features will be noisier.

**Backtest and paper trading share one implementation.** `run_backtest` and
`PaperTrader` drive the same strategy object through the same
`prepare / observe / decide / record` contract; only the source of bars and the
destination of orders differ. Going live means implementing one `BrokerAdapter`,
not rewriting the strategy.

---

## Layout

```
src/gtbot/
  data/        schema + validation, exchange fetchers, agent-based simulator
  features/    causal rolling primitives, microstructure, liquidity, regime
  game/        experts, no-regret meta-learner, equilibrium sizer
  risk/        vol targeting, drawdown governor, exposure caps
  engine/      cost & execution model, backtester, paper trader
  eval/        metrics (PSR/DSR), walk-forward, bootstrap & permutation tests
  strategy.py  the assembled pipeline
scripts/
  search.py    hyperparameter search — TRAINING seeds only
  evaluate.py  held-out evaluation — the numbers above
```

### The causality contract

`features[name][t]` may depend on bars `0..t` and nothing else. Decisions at bar
`t` execute at bar `t+1`. This is not documented and hoped for — it is tested by
perturbing the future and asserting the past does not move
(`tests/test_causality.py`), which catches the whole class of bugs that
centred moving averages and full-sample standardisation belong to.

The engine's P&L attribution is anchored by a test asserting that a
always-long, zero-cost strategy reproduces buy-and-hold to floating-point
precision.

```bash
pytest            # ~60 tests
```

## Disclaimer

Research code. Not investment advice. Backtested performance — especially on
simulated data — is not indicative of future results, and an edge of a few basis
points per trade is well within the range that ordinary implementation friction
can erase.
