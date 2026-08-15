# Game theory & algorithmic trading — research notes

Literature survey behind the game-theoretic strategies in
`src/tradebot/strategies/`. Four areas were researched; each strategy's
docstring carries its own citations, and the mapping is at the bottom.
All input is 5m OHLCV bars only (no order book), so each mechanism below
is stated together with its bar-visible footprint.

## 1. Market microstructure games

- **Kyle (1985), "Continuous Auctions and Insider Trading," Econometrica 53(6)** —
  equilibrium of informed trader vs market makers vs noise traders. The
  insider trades gradually, hiding inside noise volume; price impact
  λ = information/noise ratio. *Footprint: persistent signed flow and
  durable drift; λ estimable per bar as |return|/dollar volume
  (Amihud 2002, J. Fin. Markets).*
- **Glosten & Milgrom (1985), JFE 14(1)** — bid/ask spread as a defense
  against adverse selection. *Footprint: when informed arrivals are few,
  wiggles mean-revert; when many, moves are informational. Bar-based
  spread estimate: Corwin & Schultz (2012), J. Finance 67(2).*
- **Easley, Kiefer, O'Hara & Paperman (1996), J. Finance 51(4)** — PIN, the
  probability a trade is informed. **Easley, López de Prado & O'Hara
  (2012), RFS 25(5)** — VPIN flow toxicity via **Bulk Volume
  Classification**: buy volume = V·Φ(Δp/σ) — computable from bars alone
  (confirmed superior in Easley, López de Prado & O'Hara 2016, JFE 120(2)).
- **Admati & Pfleiderer (1988), RFS 1(1)** — liquidity and informed trading
  pool into the same windows; judge volume against its normal level.
- **Brunnermeier & Pedersen (2005), J. Finance 60(4), "Predatory Trading"** —
  leader–follower game around a forced liquidator: price overshoots
  fundamentals, then recovers. *The one mean-reversion trade the
  microstructure games license: fade identified liquidity events after
  flow exhaustion.*
- **Yang & Zhu (2020), RFS 33(4), "Back-Running"** — following detected
  informed flow is equilibrium-consistent up to the leader's camouflage.

## 2. Minority games & evolutionary game theory

- **Arthur (1994), AER 84(2)** — El Farol: no deductive equilibrium; agents
  hold an ecology of predictors selected by recent fitness.
- **Challet & Zhang (1997), Physica A 246** — the Minority Game: ±1 choices,
  minority wins; strategy tables scored by virtual points over m-bit
  histories.
- **Savit, Manuca & Riolo (1999), PRL 82** and **Challet, Marsili & Zecchina
  (2000), PRL 84** — predictability phase transition at α = 2^m/N ≈ 0.34:
  above it, some history states carry exploitable conditional drift
  ("pockets of predictability", order parameter H).
- **Jefferies, Hart, Hui & Johnson (2001), EPJ B 20** — grand-canonical MG:
  agents abstain unless their edge clears a confidence threshold — the
  game-theoretic version of a fee filter. **Johnson, Lamper, Jefferies,
  Hart & Howison (2001), Physica A 299** — train the game on a real
  binarized price series and read its vote as a forecast. **Lamper,
  Howison & Johnson (2002), PRL 88** — ensemble agreement rises before
  large moves.
- **Marsili (2001), Physica A 299** — contrarian expectations ⇒ minority
  game, trend-following ⇒ majority game; real markets mix both.
  **Andersen & Sornette (2003), EPJ B 31, "The $-game"** — with real P&L
  payoffs agents switch opportunistically between reversion and momentum.
- **Taylor & Jonker (1978), Math. Biosci. 40** — replicator dynamics;
  **Lux & Marchesi (1999), Nature 397** — profit-contagion switching
  between chartists and fundamentalists generates fat tails and regimes;
  **Brock & Hommes (1998), JEDC 22** — logit switching with intensity of
  choice β (too high ⇒ instability; here: fee bleed).
- **Satinover & Sornette (2007), EPJ B 60** — "illusion of control":
  always playing the best-scoring rule can lose to score mean-reversion;
  score updates need decay.

## 3. No-regret learning, equilibria and growth optimality

- **Freund & Schapire (1997), JCSS 55(1)** — Hedge/multiplicative weights:
  within 2√(T ln N) of the best expert in hindsight on adversarial
  sequences (survey: Arora, Hazan & Kale 2012, Theory of Computing 8).
  **Herbster & Warmuth (1998), Machine Learning 32** — fixed-share
  tracking of a drifting best expert.
- **Hart & Mas-Colell (2000), Econometrica 68(5)** — regret matching: play
  ∝ positive regret; empirical play converges to correlated equilibrium
  (via Blackwell 1956 approachability). RM⁺ clipping: Tammelin (2014).
- **Brown (1951)** fictitious play; **Robinson (1951), Annals of Math 54** —
  FP converges in zero-sum games. **von Neumann (1928)** minimax;
  **Freund & Schapire (1999), GEB 29** — no-regret play attains at least
  the zero-sum game value (with a flat action, ≥ doing nothing).
- **Cover (1991), Mathematical Finance 1(1), "Universal Portfolios"** —
  wealth-weighted mixture over constant exposures matches the best fixed
  exposure in hindsight to O(log T), assumption-free; Dirichlet(½)
  refinement Cover & Ordentlich (1996, IEEE IT 42); exact minimax
  Ordentlich & Cover (1998, Math. OR 23); costs Blum & Kalai (1999,
  Machine Learning 35).
- **Kelly (1956), BSTJ 35**; **Breiman (1961)** — log-optimal growth.
  **Bell & Cover (1980), Math. OR 5(2)** — Kelly play is the equilibrium
  of the two-investor zero-sum game. **MacLean, Thorp & Ziemba (2010)** —
  fractional Kelly against estimation error.

## 4. Repeated games, attrition, beliefs and crowding

- **Axelrod (1984), The Evolution of Cooperation** — tit-for-tat wins
  repeated-PD tournaments: nice, retaliatory, forgiving, clear.
  **Friedman (1971), Rev. Econ. Studies 38** — grim-trigger equilibria.
  **Nowak & Sigmund (1992, Nature 355; 1993, Nature 364)** — generosity
  and win-stay/lose-shift beat strict TFT under noise. *Trading reading:
  forgiveness is turnover control; the no-trade band is the equilibrium
  object.*
- **Maynard Smith (1974), J. Theor. Biol. 47** — war of attrition: the ESS
  mixes quitting times so waiting cost matches the prize; **Fudenberg &
  Tirole (1986), Econometrica 54** — as time passes without the rival
  quitting, update beliefs that the rival is strong. *Trading reading:
  time-and-cost stops; non-reversion is information.*
- **Harsanyi (1967–68), Management Science 14** — games of incomplete
  information: estimate the opponent's hidden type by Bayesian updating;
  act on belief margins (with hysteresis, so fees are paid only on
  decisive belief moves).
- **Avellaneda & Stoikov (2008), Quantitative Finance 8(3)** — market
  making around a reservation price r = s − q·γ·σ²: fair value shifted
  against inventory; the optimal spread is the fee-aware no-trade band.
- **Cardaliaguet & Lehalle (2018), Math. Fin. Econ. 12(3)** — mean-field
  game of trade crowding: drift is crowd flow; aged, saturated trends
  (rising volume per unit of price progress) carry a strategic crowding
  cost.

## Strategy → grounding map

| strategy | grounded in |
|---|---|
| `camouflage_flow` | Kyle 1985; Easley–López de Prado–O'Hara 2012/2016 (BVC/VPIN); Yang & Zhu 2020 |
| `stealth_trend` | Kyle 1985; Amihud 2002; Admati & Pfleiderer 1988; Barclay & Warner 1993 |
| `overshoot_fade` | Brunnermeier & Pedersen 2005; Glosten & Milgrom 1985; Corwin & Schultz 2012 |
| `minority_oracle` | Challet & Zhang 1997; Savit et al. 1999; Jefferies et al. 2001 (GCMG); Johnson et al. 2001 |
| `replicator_book` | Taylor & Jonker 1978; Lux & Marchesi 1999; Brock & Hommes 1998; Maynard Smith & Price 1973 |
| `game_switch` | Brown 1951; Marsili 2001; Andersen & Sornette 2003 ($-game); Challet–Marsili–Zecchina 2000 |
| `hedge_experts` | Freund & Schapire 1997; Arora–Hazan–Kale 2012; Herbster & Warmuth 1998 |
| `regret_grid` | Hart & Mas-Colell 2000; Blackwell 1956; von Neumann 1928; Freund & Schapire 1999 |
| `universal_kelly` | Cover 1991; Cover & Ordentlich 1996; Bell & Cover 1980; MacLean–Thorp–Ziemba 2010 |
| `tft_trend` | Axelrod 1984; Nowak & Sigmund 1992; Friedman 1971 |
| `attrition_reversion` | Avellaneda & Stoikov 2008; Maynard Smith 1974; Fudenberg & Tirole 1986 |
| `harsanyi_crowd` | Harsanyi 1967–68; Cardaliaguet & Lehalle 2018 |

**Shared design lesson** from all four areas: Axelrod's forgiveness,
Maynard Smith's calibrated persistence, Harsanyi's belief hysteresis,
the GCMG abstention threshold and Avellaneda–Stoikov's spread are
mechanically the same object — a no-trade band sized so the expected
value of switching state exceeds the certain cost of switching. At a
0.1% round trip on 5m bars, that band decides viability.

---

# Improving the best strategy

A second research round targeted `kelly_regime` specifically, covering
machine learning and deep learning alongside the game theory. The
headline result is negative for complexity: **every learned regime
detector failed to beat a moving-average vote**, and the only change that
helped was a one-line reshaping of the existing signal.

## Regime detection: what was tried and what happened

- **Markov-switching / HMM** — Hamilton (1989, Econometrica 57(2)). The
  *filtered* state probability is causal; the *smoothed* one that most
  tutorials plot is not. Practitioner reports of rapid switching are fatal
  at a 0.1% round trip.
- **Statistical jump models** — Nystrup, Lindström & Madsen (2020, ESWA
  150); Shu, Yu & Mulvey (2024, J. Asset Management 25(5)) is the
  strongest out-of-sample evidence in the area (three equity indices,
  1990–2023, with costs and delays, walk-forward). Note their benchmark is
  buy-and-hold and HMM — **not** a moving-average filter. Implemented
  walk-forward with deterministic restarts: it delivered a **6–11pp
  smaller drawdown and ~40% less turnover, but no Sharpe gain**
  (0.96–1.06 vs 1.09 baseline; bootstrap P(gap>0) = 0.26). An earlier
  random-initialisation version looked like a win purely through optimizer
  noise — changing only the seed moved Sharpe 0.13 and growth 40%.
- **Bayesian online changepoint detection** — Adams & MacKay (2007,
  arXiv:0710.3742). Used as a severity haircut it *lost* (OOS 0.84 vs
  1.03): in BTC, short run lengths fire on volatility bursts, and large
  **up** moves are volatility bursts too.
- **Meta-labeling + triple barrier** — López de Prado (2018, *Advances in
  Financial Machine Learning*); Joubert (2022, JFDS 4(3)). A walk-forward,
  purged and embargoed logistic secondary model **hurt in-sample and was
  neutral out-of-sample**. Note their trend-scanning label looks *forward*
  and is not admissible as a causal signal.

## Deep learning: why none was adopted

The positive results are real but do not transfer to this problem. Lim,
Zohren & Roberts (2019, JFDS) and Wood, Giegerich, Roberts & Zohren
(Momentum Transformer, arXiv:2112.08534) train directly on Sharpe and beat
classical momentum — but their cost tolerance is **2–3 bps against our 10
bps round trip**, and much of the edge comes from **diversifying across
88–100 instruments**, where we have one. Against that: Makridakis et al.
(2018, PLoS ONE 13(3)) found classical methods dominate ML on 1,045
series; Buczyński et al. (2023, Eng. Proc. 39(1)) reproduced 15 prominent
deep-learning finance papers and found most do not beat a naive forecast;
Zeng et al. (AAAI 2023) show a one-layer linear model beats transformer
forecasters on nine benchmarks. No torch/sklearn dependency was added.

Sobering for our own baseline: Zakamulin (2014, J. Asset Management; 2018,
Int. Review of Finance) finds moving-average timing performance "highly
overstated" with substantial data-mining bias — which is why the
validation here leans on Monte Carlo windows rather than one path.

## Methodology findings that changed how this repo tests

1. **A one-day lookahead is worth +2.1 Sharpe.** A daily signal broadcast
   onto the same day's 5m bars leaks, *passes* the truncation test, and
   produces Sharpe 3.09 instead of 0.99. Any result in that range should
   be read as a bug report. `tests/test_causality_real.py` now perturbs
   future bars to catch it.
2. **The noise floor is ±0.2 Sharpe** (paired stationary block bootstrap,
   30-day blocks, 2,000 resamples). Smaller differences on one path are
   not evidence — the analytic standard error of a Sharpe *level* (±0.02)
   is misleadingly tight for comparing strategies.
3. **More anchors do not help.** Ladders of 7–48 moving averages scored
   at or below the three-anchor vote, and the individual anchors are
   wildly dispersed (20d: 1.17, 250d: 0.59).
4. **Deflate for trials.** Bailey & López de Prado (2014, JPM 40(5)) —
   every sweep in VALIDATION.md is a trial and inflates the observed
   Sharpe of whatever was selected.
5. **Lookahead can hide inside `on_bar`, not just `prepare`.** A strategy
   that keeps the frame handed to `prepare()` and indexes `i + 1` in
   `on_bar` has perfect foresight and passes truncation, future-bar
   perturbation *and* live parity. Built as a probe, it returned $3.7e23
   from $1,000 at Sharpe 73 with a green suite.
   `tests/test_causality_strict.py` now compares the *orders* a strategy
   queues under two opposite tampers of the future.
6. **A warmup prefix is not free.** Letting a strategy trade through the
   prefix of a resampled window lets it be liquidated before the window
   opens (19 of buy-and-hold's 23 stress-test liquidations were of this
   kind), and slicing a frame to an out-of-sample date range leaves a
   100-day-warmup strategy flat for 7.6% of it while a zero-warmup
   benchmark trades from day one. Warm the state, withhold the trading:
   `run_backtest(trade_start=...)` and `tradebot.window.run_period`.

## What shipped

Three variants, each as a separate registered strategy so the incumbent's
published record stays intact: the convex vote response
(`kelly_regime_v2`, with its failed out-of-sample check reported),
conditional volatility targeting (`kelly_regime_v3`), and the doubling
anchor ladder (`kelly_regime_v4`, the current leader). See
[VALIDATION.md](VALIDATION.md#beta-testing-variants-kelly_regime_v2).

### Anchor timescales: a region, not a peak

The last change was to the regime anchors themselves — 20/40/80 days
instead of 30/50/100 — chosen as a **doubling ladder** rather than fitted.
The multi-timescale prior comes from the heterogeneous-market hypothesis
(Müller et al. 1997, J. Empirical Finance 4(2–3)) and its concrete form in
Corsi's (2009, J. Financial Econometrics 7(2)) HAR model: agents act on
distinct, roughly geometrically spaced horizons, so a cascade of fixed
timescales beats estimating one correct lookback.

What the evidence supports is narrower than the headline balance. Across
nine anchor sets in the 18–28 day range, **every** variant cut max
drawdown to 35–39% from v3's 41.8% — that reduction is the robust
finding. The Sharpe spread over the same plateau (1.52–1.60) sits inside
the ±0.2 noise floor from finding 2 below, so the return improvement is
**not** established by this path. The plateau breaks sharply below ~18
days (16/32/64 scores 1.46), which is the signature of a genuine region
rather than a tuned peak — and, per finding 4, the sweep itself is a
trial that inflates whatever it selected.

## Volatility & sizing: the second half of the round

The sizing research produced the one change that earned promotion, plus
two negative results that invert the textbook reading.

- **Conditional volatility targeting** — Bongaerts, Kang & van Dijk (2020,
  FAJ 76(4)) show conventional continuous targeting fails to consistently
  improve performance and can deepen drawdowns, while adjusting exposure
  only in the volatility *extremes* improves Sharpe at low turnover.
  Implemented as `kelly_regime_v3`: **$139,509 vs $108,221, Sharpe 1.55 vs
  1.42, better out-of-sample, beats the baseline in 75% of random
  windows.**
- **Why it works here** — Baur & Dimpfl (2018, Economics Letters 173):
  crypto has an **inverse leverage effect**, positive shocks raising
  volatility more than negative ones. Measured on this data, the
  highest-volatility quintile carries the *highest* forward Sharpe
  (+1.08 overall, +2.06 when the gate is bullish). So the Moreira & Muir
  (2017, J. Finance) volatility-managed alpha — which requires high
  volatility to forecast low returns — is absent-to-inverted for BTC.
  What remains is Harvey et al.'s (2018, JPM) mechanical tail protection,
  worth roughly +7% Sharpe on buy-and-hold, which conditional targeting
  keeps. Cederburg et al. (2020, JFE 138(1)) independently find
  volatility-managed portfolios do not systematically outperform out of
  sample across 103 equity strategies.
- **Negative result: better volatility forecasting makes this strategy
  worse.** A timescale blend beat the incumbent estimator by 8% on QLIKE
  (Patton 2011, J. Econometrics 160(1), for why QLIKE) — a genuinely
  better forecast — and returned **$52K instead of $115K**, because it
  de-levers more promptly into the high-Sharpe states. Corsi's (2009)
  HAR insight that timescale mixing beats estimator choice held for
  *forecasting* and reversed for *trading*.
- **Negative result: range estimators are biased low at 5m.** Parkinson
  (1980), Garman & Klass (1980), Rogers & Satchell (1991), Yang & Zhang
  (2000) read **7–18% low** on 5-minute bars — the documented
  discretisation bias, since observed extremes sit inside true continuous
  ones. A drop-in swap silently raises effective leverage. Their 5–8x
  efficiency advantage is measured per observation against a *daily
  close-to-close* estimator; the incumbent already averages 288 squared
  returns per day, so there is nothing left to gain. Hansen & Lunde
  (2005, J. Applied Econometrics 20(7)) found nothing beats GARCH(1,1)
  absent a leverage effect — and BTC's is inverted.
- **Drawdown control** — Grossman & Zhou (1993, Mathematical Finance
  3(3)) cushion sizing applied to the *whole* book cut drawdown to 28%
  but destroyed return ($21.6K); applied only to leverage above 1x it
  gave −1.2pp drawdown for roughly zero cost. Klass & Nowicki (2005)
  predicted the former: the cushion rule is not optimal in discrete time
  and systematically sells low in a mean-reverting-drawdown asset.
