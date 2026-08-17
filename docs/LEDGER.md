# The research ledger — everything already tried

This is the memory of the project. **Read it before proposing anything**
(step 0 of [ROUTINE.md](ROUTINE.md)), and append to it at the end of
every session. Its purpose is to stop the same idea being re-tried blind,
and to make the cost of each attempt visible.

Three sections and a backlog:

- **[A. Strategies](#a-strategies-registered)** — every registered
  strategy, what it attacked, what happened.
- **[B. Research directions](#b-research-directions-not-registered-as-strategies)** — experiments, studies and
  methodology work that never became a table row.
- **[C. Ruled out](#c-ruled-out--do-not-re-try-without-new-evidence)** — do not re-try without new evidence.
- **[D. Backlog](#d-backlog-ranked)** — ranked, with blockers.

Backfilled 2026-08-16 from `STRATEGIES.md`, `RESEARCH.md`,
`VALIDATION.md`, `ALTERNATIVES.md`, `CROSS_ASSET.md`, `ELLIOTT_WAVES.md`,
`LIVE.md` and `FRONTIER.md`, which remain the long-form record. Balances
are $1,000 start, full period, from the README comparison table.

## The standing diagnosis

Four constraints bind this project. An entry is only worth making if it
attacks one; the `attacks` column below records which.

| code | constraint |
|---|---|
| **INFO** | One price series. Every strategy consumes the same OHLCV bars. |
| **N≈3** | Effective sample size is ~3 regime events, not 1.01M bars. |
| **ERR** | No error control anywhere in the signal path. |
| **COST** | Costs scale *with* the signal — funding runs +20%/yr while the strategy holds vs +2.8% flat. |
| **SIZE** | (what actually worked) Decide *how much* to hold, not *what happens next*. |

The one-line summary of 25 results: **every strategy that decides how
much to hold makes money; every strategy that predicts what happens next
loses to fees.**

---

## A. Strategies (registered)

| ID | strategy | added | idea | attacks | spot | fut 5x | verdict | lesson |
|---|---|---|---|---|---|---|---|---|
| L-01 | `kelly_regime_v4` | 08-15 | v3 on a doubling anchor ladder (20/40/80d), Müller 1997 / Corsi 2009 HAR | SIZE | $66.8K | $156.2K | **PROMOTED** | Drawdown 35.3% is the robust finding; the return improvement sits inside the ±0.2 Sharpe noise floor and is *not* established. |
| L-02 | `kelly_regime_v3` | 08-15 | Conditional vol targeting — constant notional through normal vol, re-size only on breakout (Bongaerts 2020) | SIZE | $65.8K | $139.5K | **PROMOTED** | Improves every metric in both sub-periods and both markets; flat parameter neighbourhood (8 combos, Sharpe 1.47–1.55). |
| L-03 | `kelly_regime_v2` | 08-15 | Convex vote response: partial anchor agreement = low confidence, not half a signal | SIZE | $46.4K | $122.0K | NOT PROMOTED | Nine of ten metrics improve and it still fails: −6.5% out-of-sample. Kept registered with the failure stated. |
| L-04 | `kelly_regime` | 08-14 | Fractional-Kelly vol-targeted sizing gated on a crowd-regime filter (Cardaliaguet & Lehalle 2018) | SIZE | $42.1K | $108.2K | **PROMOTED** (incumbent) | The one robust finding of the whole project: regime-gated sizing cuts drawdown. First strategy to beat the benchmark. |
| L-05 | `kelly_regime_ev` | 08-16 | Rebalance only when expected gain exceeds the fee: \|Δf\| > 2·fee/(H·σ²) (Constantinides 1986; Davis & Norman 1990) | COST | $40.9K | $108.0K | REGISTERED | Derives analytically what `fee_study.py` found by brute force. The hand-set 10% deadband is ~3x too narrow at 0.10%; at 0.40% the band exceeds 1.0 — **no rebalance is ever worth its cost**. |
| L-06 | `kelly_regime_ev_fast` | 08-16 | Same, shorter holding horizon → wider band, 34 trades | COST | $71.1K | $70.8K | REGISTERED | Best *spot* number in the table, from trading 34 times. Turnover, not signal, was the binding cost on spot. |
| L-07 | `buy_and_hold` | 08-12 | Buy on the first bar, never trade | — | $66.0K | 💀 $18 | BENCHMARK | The bar everything must clear. 84% drawdown on spot; liquidated on 5x, and in 26 of 40 Monte Carlo windows. |
| L-08 | `champions_council` | 08-14 | Hedge over the games that actually pay, sized by fractional Kelly | SIZE | $19.3K | $36.8K | REGISTERED | Ensembling profitable members does not recover the best member. |
| L-09 | `hedge_experts` | 08-12 | No-regret Hedge blend of technical experts, each charged its own turnover (Freund & Schapire 1997) | — | $13.3K | $258 | NEGATIVE | No-regret guarantees are relative to the best expert; when every expert loses to fees, so does the blend. |
| L-10 | `replicator_book` | 08-12 | Replicator dynamics reallocating across trend/value/cash on realized fee-adjusted fitness (Taylor & Jonker 1978; Lux & Marchesi 1999) | — | $2,330 | $10.58 | NEGATIVE | Fitness measured on realized returns is a lagging estimator; the reallocation arrives after the regime. |
| L-11 | `universal_kelly` | 08-12 | Cover's universal portfolio over fixed exposures, half-Kelly capped (Cover 1991) | SIZE | $1,276 | $1,227 | NEGATIVE | Universality is asymptotic. Nine trades in a decade is what the guarantee costs at this horizon. |
| L-12 | `harsanyi_crowd` | 08-12 | Belief margin over hidden market types, sized down when the trend is crowded (Harsanyi 1967; Cardaliaguet & Lehalle 2018) | INFO | $888 | $429 | NEGATIVE | The crowding intuition was right — it is what `kelly_regime` later exploited — but as a *direction* signal rather than a *sizing* input it loses. |
| L-13 | `overshoot_fade` | 08-12 | Fade forced-liquidation overshoots once aggressive flow is exhausted (Brunnermeier & Pedersen 2005) | — | $662 | $33.52 | NEGATIVE | Good win rate, bad tails. Proof that directional accuracy says nothing about an equity curve. |
| L-14 | `camouflage_flow` | 08-12 | Follow informed order flow recovered from bars via Bulk Volume Classification (Kyle 1985; Easley–López de Prado–O'Hara VPIN) | INFO | $548 | $0.99 | NEGATIVE | **BVC from OHLCV is a price transform, not order flow.** Proxying unavailable data out of price adds no information and full turnover. |
| L-15 | `stealth_trend` | 08-12 | Momentum only on deep, high-participation bars where informed flow hides (Admati & Pfleiderer 1988) | INFO | $465 | $0.38 | NEGATIVE | Same lesson as L-14, different dressing. 1,605 trades. |
| L-16 | `flow_regime` | 08-12 | Combine both microstructure sides: follow flow, fade liquidation overshoots | INFO | $447 | $0.80 | NEGATIVE | Combining two losing signals produces a losing signal with more turnover. |
| L-17 | `game_council` | 08-12 | No-regret Hedge allocation over the game strategies' own signals | — | $284 | $2.00 | NEGATIVE | 2,541 trades. Meta-allocation cannot manufacture an edge its members lack. |
| L-18 | `minority_oracle` | 08-12 | Abstention-filtered vote of a grand-canonical minority game trained online on binarized returns (Challet & Zhang 1997) | — | $53 | $3.83 | NEGATIVE | 9,039 trades, 95% drawdown. The GCMG abstention threshold is a no-trade band and it was set far too narrow for a 0.1% round trip. |
| L-19 | `game_switch` | 08-12 | Best-respond to whichever game the market is playing, trading only history states with significant conditional drift (Brown 1951; Marsili 2001) | — | $5.00 | $1.00 | NEGATIVE | Fictitious play on 5m bars: 6,672 trades, 99% drawdown. |
| L-20 | `regret_grid` | 08-12 | Regret-matching+ over a position grid, correlated-equilibrium play (Hart & Mas-Colell 2000) | — | $5.00 | $1.00 | NEGATIVE | Regret minimisation converges to the equilibrium of a game whose payoffs include fees; the equilibrium is "do not play". |
| L-21 | `tft_trend` | 08-12 | Repeated-game trend truce: hold while the market cooperates, forgive one defection, punish two (Axelrod 1984) | — | $4.99 | $1.00 | NEGATIVE | Axelrod's forgiveness *is* a no-trade band — the useful kernel, discovered here by accident and later derived properly in L-05. |
| L-22 | `macd_cross` | 08-12 | Long on MACD cross above signal, flat/short below | — | $4.99 | $1.00 | NEGATIVE | Baseline. 4,301 trades → 100% drawdown. |
| L-23 | `macd_rsi` | 08-12 | RSI pullback recoveries in the direction of the MACD trend | — | $4.96 | $0.94 | NEGATIVE | Baseline. Combining two classic indicators does not beat either. |
| L-24 | `attrition_reversion` | 08-12 | Fade deviations from an inventory-shifted fair value; quit when waiting costs exceed the prize (Avellaneda & Stoikov 2008; Maynard Smith 1974) | COST | $4.94 | $0.99 | NEGATIVE | Avellaneda–Stoikov needs an order book. Run on bar-close fills it is a mean-reversion rule paying taker fees on both sides. |
| L-25 | `rsi_reversion` | 08-12 | Buy RSI < 30, exit on recovery; mirror short on futures | — | $4.85 | $0.77 | NEGATIVE | Baseline. 4,464 trades. |

**Pattern across A:** every entry with `SIZE` in the attacks column is
profitable; every pure predictor is not. The four `INFO` entries all
tried to recover missing information from price and all failed — that is
the most expensive repeated mistake in this table.

---

## B. Research directions (not registered as strategies)

| ID | direction | date | what was done | result | verdict |
|---|---|---|---|---|---|
| R-01 | Markov-switching / HMM regime detection (Hamilton 1989) | 08-15 | Assessed | The *filtered* state is causal, the *smoothed* one that tutorials plot is not. Reported rapid switching is fatal at a 0.1% round trip. | REJECTED on reading |
| R-02 | Statistical jump models (Nystrup 2020; Shu 2024) | 08-15 | Implemented, walk-forward, deterministic restarts | −6–11pp drawdown, ~40% less turnover, **no Sharpe gain** (0.96–1.06 vs 1.09; bootstrap P(gap>0)=0.26). A random-init version looked like a win purely from optimizer noise — the seed alone moved Sharpe 0.13 and growth 40%. | NEGATIVE |
| R-03 | Bayesian online changepoint detection (Adams & MacKay 2007) | 08-15 | Implemented as a severity haircut | Lost: OOS 0.84 vs 1.03. Short run lengths fire on volatility bursts, and in BTC large **up** moves are volatility bursts. | NEGATIVE |
| R-04 | Meta-labeling + triple barrier (López de Prado 2018) | 08-15 | Walk-forward, purged and embargoed logistic secondary model | Hurt in-sample, neutral out-of-sample. Their trend-scanning label looks *forward* and is inadmissible here. | NEGATIVE |
| R-05 | Deep learning (Lim 2019; Momentum Transformer 2021) | 08-15 | Literature assessment; no dependency added | Published edge assumes **2–3bps against our 10bps**, and much of it comes from diversifying across **88–100 instruments** where we have one. Counter-evidence: Makridakis 2018, Buczyński 2023, Zeng AAAI 2023 (one-layer linear beats transformers on 9 benchmarks). | NOT ATTEMPTED, ruled out on grounds |
| R-06 | Anchor ladders of 7–48 moving averages | 08-15 | Swept | Scored at or below the three-anchor vote. Individual anchors wildly dispersed (20d 1.17, 250d 0.59). | NEGATIVE |
| R-07 | Anchor timescale region, 18–28 days | 08-15 | 9 anchor sets | *Every* variant cut drawdown to 35–39% from 41.8%; Sharpe spread 1.52–1.60 sits inside the noise floor. Breaks sharply below ~18d (16/32/64 → 1.46) — a region, not a tuned peak. | INFORMS L-01 |
| R-08 | Better volatility *forecasting* | 08-15 | Timescale blend, 8% better on QLIKE (Patton 2011) | **$52K instead of $115K.** A genuinely better forecast de-levers more promptly into BTC's high-vol, high-forward-Sharpe states. | NEGATIVE — and sign-inverting |
| R-09 | Range volatility estimators (Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang) | 08-15 | Measured on 5m bars | Read **7–18% low** (discretisation bias); a drop-in swap silently raises effective leverage. Their efficiency advantage is against a *daily close-to-close* estimator; the incumbent already averages 288 squared returns/day. | NEGATIVE |
| R-10 | Inverse leverage effect in BTC (Baur & Dimpfl 2018) | 08-15 | Measured forward 5d Sharpe by lagged-vol quintile | High vol forecasts the **highest** forward Sharpe (+1.08 all bars, +2.06 when the gate is bullish) — the opposite of equities. Moreira & Muir (2017) vol-managed alpha is absent-to-inverted here. | **KEY FINDING** — explains L-02 |
| R-11 | Grossman–Zhou drawdown cushion (1993) | 08-15 | Two variants | Whole book: drawdown 28% but return destroyed ($21.6K). Above 1x leverage only: −1.2pp drawdown at ~zero cost. Klass & Nowicki (2005) predicted the former — the cushion rule sells low in a mean-reverting-drawdown asset. | PARTIAL |
| R-12 | Turnover reduction to fit a fee tier | 08-15 | Swept 8 lookbacks × 4 hysteresis bands = **32 configs** | **28 of 32 beat holding in-sample; 0 of 28 out-of-sample.** The one you would have selected lost 34.5% to holding. Gross edge on spot (1.33x) is far below the 2.98x needed at 0.40%, and slowing down shrinks the gross edge in step. | **CLOSED** — the defining negative result |
| R-13 | Fee tier study (`scripts/fee_study.py`) | 08-15 | Measured every Bitstamp tier | Break-even is **0.104%** against an assumed 0.10% — the published spot edge lives entirely inside that margin. At the 0.40% entry tier nothing beats holding ($29.5K vs $65.8K); the $5M/30d tier still misses by 4%. | **CLOSED** |
| R-14 | Funding as a first-class cost (`scripts/funding_study.py`) | 08-16 | Real Binance BTCUSDT funding, compounded | Positive at **86.5%** of settlements, ~15%/yr for a constant long. `kelly_regime_v4`'s $156K becomes **$36K–$80K** — a band straddling spot holding's $66K. Worse: funding runs **+20%/yr while the strategy holds** vs +2.8% flat, because the crowding it detects is what sets the rate. | **KEY FINDING** — the COST constraint |
| R-15 | Funding harvest / cash-and-carry | 08-16 | Compounded the real series, 2020–2023 | +82.0% over 4.0y = **+16.2%/yr**; +14.6% after 0.10% both legs, +9.8% at 0.40%; payer flips 13.5% of settlements; **worst 30-day run −1.31%**. Literature reports carry Sharpe ~6.45, falling to 4.06 from 2024 and **negative in 2025** as it crowded — and our data stops exactly at 2023. | **BLOCKED** on data → B-02 |
| R-16 | Funding as a positioning signal | 08-16 | Quintile and momentum-controlled sort, 2020–2023 | 14-day forward spread Q1−Q5 = **+3.57pp**; high funding predicts negative forward returns unless price is also rising; correlation with trailing return only 0.39, so not a momentum proxy. But middle quintiles are non-monotone (tied clamped values) — a warning about how much is noise. | OPEN hypothesis → B-05 |
| R-17 | Cross-asset falsification on ETH | 08-16 | Bitfinex BTC + ETH, same venue, same window (2016-03→2019-12) | **The risk property transfers, the return property does not exist.** Drawdown cut in all four cells (BTC 83.8→40.1, ETH 94.2→36.5, 5x 85.2→32.1 and 99.3→35.1). Loses to holding on spot on both assets (0.58x, 0.47x). The 236x ETH futures cell is survival, not edge. | **PARTLY ANSWERS N≈3** |
| R-18 | Elliott Wave Theory (± NN, ± game theory) | 08-16 | Assessed against this repo's bar | Not falsifiable as practised — counts are re-labelled after the fact, the exact leak class `test_causality_strict.py` exists to catch. Its one quantitative component (Fibonacci ratios) was refuted by Batchelor & Ramyar. *ElliottAgents* (2024) reports 73.68% vs 57.89% — that is **14/19 vs 11/19**, three extra calls, over a monotonic 2-year rise, with no walk-forward. Its useful kernel (multi-timescale crowd structure) is already `kelly_regime_v4`. | NOT PURSUED |
| R-19 | Monte Carlo window stress test | 08-14 | 40 random windows, identical across strategies | Leveraged buy-and-hold **liquidated in 26 of 40**, median window −98%. Every `kelly_regime` variant survived all 40, profitable in 85–88%, beat holding in 65%. | **KEY FINDING** |
| R-20 | Noise floor measurement | 08-15 | Paired stationary block bootstrap, 30-day blocks, 2,000 resamples | **±0.2 Sharpe.** Smaller differences on one path are not evidence. The analytic SE of a Sharpe *level* (±0.02) is misleadingly tight for *comparing* strategies. | **METHOD** — binds every claim here |
| R-21 | Lookahead probes | 08-15 | Two adversarial probes | A one-day signal broadcast onto 5m bars is worth **+2.1 Sharpe** and *passes* truncation. A strategy that keeps the `prepare()` frame and indexes `i+1` in `on_bar` returned **$3.7e23 at Sharpe 73 with a green suite**. Both now caught by `test_causality_real.py` / `test_causality_strict.py`. | **METHOD** |
| R-22 | Warmup-prefix bias | 08-15 | Audit | Letting a strategy trade the warmup prefix let it be liquidated *before* the window opened — **19 of buy-and-hold's 23 stress liquidations were this artifact**. Slicing to an OOS range left a 100-day-warmup strategy flat for 7.6% of it. Fixed by `run_backtest(trade_start=...)` / `tradebot.window.run_period`; verdicts survived, numbers moved ~75%. | **METHOD** |
| R-23 | Capital scaling | 08-15 | $1K vs $1M across every strategy | Results are proportional to capital; the only deviations came from the exchange minimum order size. One start balance is therefore sufficient. | SETTLED |
| R-24 | Exchange adapter parity | 08-15 | Bar-for-bar over 30 consecutive candles, both adapters | Top-three strategies compute the identical target from paged exchange data and from the contiguous backtest frame; paging is lossless; neither adapter hands a strategy the forming candle. | SETTLED |
| R-25 | Deflated Sharpe / purged CV / bootstrap CIs | — | Cited in `RESEARCH.md`, **never computed** | Every sweep here is a trial that inflates whatever it selected (R-12 ran 32). The comparison table reports points where it should report ranges. | **NOT DONE** → B-04 |
| R-26 | E-process regime detection with unified Kelly sizing (Shafer 2021; Ramdas et al. 2023; Waudby-Smith & Ramdas 2024; Shin, Ramdas & Rinaldo 2024) | 08-17 | Three variants implemented in `experiments/eprocess_regime.py`, swept on the inner split | *(see the pre-registration and results below)* | — |

### R-26 pre-registration — written and committed before the holdout was read

**Idea.** `kelly_regime` answers "is the regime bullish?" with a latched
moving-average vote and "how much do I hold?" with `target_vol/realized_vol`.
Testing by betting says these are one question: the wealth of a betting
martingale against the null *drift is zero* **is** the evidence, and the
Kelly bet that grows it **is** the position.

**Constraint attacked.** ERR (no error control anywhere in the signal
path) and N≈3 (e-processes give non-asymptotic Type-I control valid at
arbitrary stopping times — the only tool on the list that survives a
sample size of three, and the only one that legitimises the optional
stopping this project does constantly).

**Not a duplicate of.** L-04/L-01/L-02/L-03 (heuristic latched vote,
hand-set 1% band); R-01 (HMM — the smoothed state is not causal); R-02
(jump models); and specifically **not R-03** (Bayesian online changepoint
detection, which lost): BOCPD's run-length posterior collapses on
*volatility* bursts, and in BTC large **up** moves are volatility bursts
(R-10), so it fired with the wrong sign. Here volatility is the
*denominator* of the bet — a volatility burst shrinks the position but
does not destroy evidence. Only realized drift moves the e-process.

**Simulable here?** Yes. One price series, causal, no new data.

**Pre-registered failure modes** (named before any code ran): (a) the
evidence process is a smoothed momentum indicator in disguise, so results
sit inside the ±0.2 Sharpe noise floor of the incumbent; (b) evidence
accumulated in the 2017 bull never decays and the gate degenerates to
buy-and-hold; (c) a continuously-varying gate costs turnover and the fees
eat it.

**Frozen configuration.** E1 — evidence gate, incumbent sizer:
`bet_halflife_days=20, alpha=0.05, clip=5, evidence_cap_mult=1.0,
deadband=0.10, target_vol=0.55, max_leverage=2.0`, no evidence decay.
Half-life 20d was selected on inner-validation and coincides with the
18–28 day anchor region R-07 independently found robust; every other knob
is at its a-priori default.

**Decision rule — promote only if all four hold:**

- **P1** on the 2023+ holdout, spot final balance beats `buy_and_hold`;
- **P2** the improvement over `buy_and_hold` is either > +0.2 Sharpe, or a
  drawdown improvement of ≥ 10 percentage points;
- **P3** *(falsification)* on ETH — Bitfinex, the R-17 window, same venue
  and period as its BTC control — the drawdown reduction replicates, i.e.
  max drawdown is no worse than `kelly_regime_v4`'s + 5pp on **both** spot
  and futures. If the risk property does not transfer to a second asset,
  the error-control claim is dead;
- **P4** the parameter neighbourhood is a plateau, not a peak.

**Stated prediction before looking:** P1 fails. The e-process holds 0.32x
the incumbent's mean exposure into a bull holdout, so it cannot out-return
holding. Expected verdict NEGATIVE, with the drawdown reduction as the
finding.

**Secondary question, also fixed in advance** (not part of the promotion
bar): does the evidence gate cut out-of-sample drawdown relative to
`kelly_regime_v4`? That is the scientific result either way.

---

## C. Ruled out — do not re-try without new evidence

| what | why | ref |
|---|---|---|
| More indicators / more ML on 5m bars | 25 strategies and two research rounds; every pure predictor lost to fees. Attacks none of the four constraints. | A, R-05 |
| Recovering order flow from OHLCV | BVC/VPIN proxies are price transforms. Four strategies, four losses. | L-14, L-15, L-16, L-12 |
| Tuning turnover to fit a fee tier | 28 of 32 in-sample, 0 of 28 out-of-sample. | R-12 |
| Higher leverage as a fix for fees | Fees are charged on notional; leverage multiplies cost and return together. Changes the risk profile, not the sign. | R-13 |
| Sentiment / social media | A lagged function of price — not orthogonal information — and revision-prone. | FRONTIER |
| Higher-frequency execution | Turnover is the enemy at every fee tier available. | R-12, R-13 |
| Elliott waves | Unfalsifiable as practised; its testable kernel already implemented. | R-18 |
| Market making, AMM/LVR | Plausibly real; **not simulable** on bar-close fills with no order book. Ruled out on what can be checked, not on merit. | L-24, FRONTIER |
| Options / volatility risk premium | Same — no options data, no way to validate here. | ALTERNATIVES |

---

## D. Backlog (ranked)

| ID | item | attacks | status | note |
|---|---|---|---|---|
| **B-01** | **E-process regime detection with unified Kelly sizing** (Shafer 2021 "Testing by betting"; Ramdas et al. 2023) | ERR, N≈3 | **NEXT** | The regime detector and the position sizer are the same object: the e-process wealth accumulated against "drift is zero" *is* the evidence, and the Kelly bet that grows it *is* the position. Anytime-valid error control is the only tool on this list designed for n≈3, and it legitimises the optional stopping this repo does constantly. Needs no new data. Expect a calibration/drawdown gain, not return alpha. |
| **B-02** | Extend the funding series through 2026 | COST | **OPEN** | One data fetch. Decides B-03 outright: the literature says the carry premium broke in 2024–25 and our data stops in 2023. The single cheapest item that can change a decision. |
| **B-03** | Funding harvest (delta-neutral spot vs short perp) | COST | BLOCKED on B-02 | +16.2%/yr with a −1.31% worst month is a risk profile nothing else here approaches — measured entirely in the good years. Unmodelled: basis risk, short-leg liquidation, exchange/custody risk, borrow cost. |
| **B-04** | Purged CV, deflated Sharpe, block-bootstrap CIs on every headline | ERR | OPEN | Infrastructure, ~a day, raises the quality of every number already in the repo. Given the ±0.2 noise floor most of the table's ordering is probably not significant, and it should say so. Prerequisite for the routine's step 4 to be mechanical rather than improvised. |
| **B-05** | Funding as a gate on the existing strategy (stand flat in the top decile) | COST | OPEN | The low-turnover way to use R-16, and it directly targets the adverse timing in R-14. Higher-turnover standalone reversal use is where strategies go to die (R-12). |
| **B-06** | Forward paper-trading recorder | N≈3 | OPEN | The only *uncontaminated* data this project can still generate. Append the frozen leader's signal and mark price every 5m, never re-run, never tune. `bot.py` + `dry_run=True` already exist. Slow to pay off, which is the argument for starting it now. |
| **B-07** | On-chain features, sign-corrected | INFO | OPEN | The only genuinely orthogonal channel. Enter with the base rate in mind: a 141-predictor study found 67 worked in-sample, 29 survived out-of-sample, **4 beat a random walk at all horizons**. Note the trap: on-chain flows predict *volatility*, and R-08 showed better volatility input makes this strategy worse. **Fix the sign first.** |
| **B-08** | Second bear, second asset, different period (ETH 2020–2026) | N≈3 | OPEN | R-17 shares the 2018 bear with the main dataset, so the two tests are not independent. This closes that gap. |
| **B-09** | Conformal prediction / adaptive conformal by betting | ERR | LOW | Mostly subsumed by B-01. Coverage guarantees degrade exactly when regimes shift, which is when they matter. |
| **B-10** | Deterministic Elliott wave counter | — | LOW | Only as a documented negative result, per R-18. ZigZag pivots, mechanical impulse/corrective rules, no discretion. About a day, converts an unfalsifiable debate into a table row. |

---

## Appending a row

Copy this into the right section at the end of each session. One session,
one entry.

```
| L-nn / R-nn | <name> | <MM-DD> | <idea in one line, with citation> |
  <INFO|N≈3|ERR|COST|SIZE> | <train result> | <holdout result> |
  <PROMOTED|NEGATIVE|BLOCKED|PARKED|SETTLED> | <one-line lesson> |
```

Also record, in the row or a footnote beneath it:

- **configs evaluated** in step 3 (for deflated Sharpe) — and if the
  round ran directions in parallel, the total across ALL branches, not
  just this one;
- the **pre-registered decision rule** (the thresholds fixed before the
  holdout was read) and the **pre-registered falsification test**, with
  their outcomes;
- the **holdout counter**: how many times the 2023+ holdout has been
  consulted across the project to date, this row included;
- whether the **decision rule moved** after seeing the holdout — if so,
  the result is in-sample and must say so;
- the **next step**, which becomes a backlog row if the work continues.

### Holdout consultations to date

| as of | count | note |
|---|---|---|
| 08-16 | ~30 | Backfilled estimate. Every OOS figure in sections A and B came from reading the 2023+ holdout; it has never been pristine. Deflate program-level claims accordingly, and treat forward paper trading (B-06) as the only uncontaminated evidence still obtainable. |
