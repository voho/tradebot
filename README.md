# gtbot — a game-theoretic BTCUSD 5-minute trading bot

A trading system for BTCUSD 5-minute bars built on repeated-game and
market-microstructure theory, with a backtester, a paper-trading loop, and an
evaluation harness designed to try to *disprove* its own results.

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
| retail | 6.65 bp | +0.13 | +0.00 | +0.30% | 0.39% | 0.34% | 25 | +0.39 |
| VIP 3 | 4.95 bp | +0.43 | −0.37 | +2.04% | 2.20% | 2.04% | 148 | +1.11 |
| **VIP 6** | **3.85 bp** | **+1.19** | **+0.60** | **+6.45%** | 4.47% | 4.27% | 294 | +2.08 |
| **VIP 9** | **2.05 bp** | **+2.32** | **+1.28** | **+20.88%** | 7.71% | 5.48% | 516 | +3.09 |
| market maker | 2.15 bp | +2.24 | +1.17 | +19.57% | 7.54% | 5.47% | 502 | +3.07 |

The gross edge is 5–9 bp per trade over a 3-bar hold. A retail round trip costs
6.65 bp, so at retail fees **this strategy is not viable** and the sizer
correctly says so by trading 25 times a year instead of 500. This is not a
disappointing detail to be buried; it is the main practical finding. A 5-minute
mean-reversion strategy is a fee-tier business.

### It is not a data-mining artefact

Pooled across held-out seeds at VIP 6:

- block-bootstrap 95% CI on the Sharpe: **[+0.59, +1.85]**
- bootstrap `p(Sharpe ≤ 0)`: **0.0000**
- Newey–West t-statistic: **+3.66**
- probabilistic Sharpe ratio: **0.9999**

### Negative controls

The strategy claims to exploit a specific structure. Destroy the structure and
it must earn nothing:

| Data | Sharpe | Trades |
|---|---:|---:|
| Structureless random walk (same vol) | **0.00** | **0** |
| Block-bootstrapped surrogate | **0.00** | **0** |
| Simulated market | **+1.19** | 294 |

It does not merely lose less on the controls — it declines to trade at all,
because the edge estimator finds no edge and the maximin size is zero.

---

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
  5-minute data). Shorter samples are mostly warm-up, which is why the
  walk-forward harness uses long folds.
- **Contextual regimes are switched off by default.** Each context cell learns
  independently, so cells cost data; at this information coefficient a cell needs
  O(10k) observations before its weights mean anything. The machinery is there
  for deployments with more history.
- **No slippage beyond the model, no exchange outages, no funding by default**
  (funding is supported, set `CostModel.funding_bp_per_8h`).

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
