# Game theory & algorithmic trading — research notes

Literature survey behind the game-theoretic strategies in
`src/tradebot/strategies/`. Four areas were researched; each strategy's
docstring carries its own citations, and the mapping is in the
[Strategy → grounding map](#strategy--grounding-map) below.
All input is 5m OHLCV bars only (no order book), so each mechanism below
is stated together with its bar-visible footprint.

The **R-189 intraday game round** adds ten registered research candidates;
see the [source review](R189_RESEARCH.md) for primary 2024–2026 research,
foundational algorithms, implementable equations and precise adaptation limits.
It covers cautious optimism, second-order expert games, conditional regret,
vector approachability, robust and cooperative model councils, quantal response,
specialists and defensive forecasting. The
[strategy registry section](STRATEGIES.md#r-189-intraday-game-candidates),
[frozen evaluation harness](../experiments/r189_games.py) and
[evaluation reports](../reports/r189_games/) distinguish research registration
from promotion. The literature does not establish profitable intraday Bitcoin
strategies or guarantee a few actual trades per day.

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
6. **Turnover reduction cannot buy back a fee tier.** At a 0.40% taker
   fee the leading strategy's *gross* edge on spot (1.33x holding) is far
   below what its turnover costs (a 2.98x gross edge would be needed),
   and slowing it down to save fees shrinks the gross edge in step.
   Sweeping the raw regime filter over 8 lookbacks × 4 hysteresis bands,
   **28 of 32 configurations beat holding in-sample and 0 of 28 beat it
   out-of-sample**; the one you would have selected lost 34.5% to
   holding. A negative result worth the space, because "just trade less"
   is the first thing anyone tries. Detail in
   [LIVE.md](LIVE.md#can-it-be-tuned-to-beat-the-fee-no-and-the-attempt-is-instructive).
7. **The deadband should be derived, not chosen.** For a Kelly sizer the
   growth given up by holding exposure ``f`` instead of the desired
   ``f*`` is ``(sigma^2/2)(f - f*)^2`` per unit time, while correcting it
   costs ``fee*|Δf|``. Trading is worth it only when the first exceeds
   the second, i.e. when ``|Δf| > 2*fee/(H*sigma^2)`` for a holding
   horizon ``H`` — the classic transaction-cost no-trade band
   (Constantinides 1986; Davis & Norman 1990) in the form this framework
   needs. Implemented as `kelly_regime_ev`. Two consequences worth
   recording: the hand-set 10% deadband is roughly **3x too narrow** at a
   0.10% fee, and at 0.40% the band exceeds 1.0, meaning **no rebalance
   is ever worth its cost** — an analytic derivation of the result
   `scripts/fee_study.py` reached by brute force.
8. **A band on a *signal threshold* is a different object from a band on
   a *position*, and this repo has repeatedly cited the wrong literature
   for it.** Finding 7's chain (Constantinides 1986; Davis & Norman 1990;
   Janeček & Shreve 2004; Gerhold et al. 2014) is **constant-target,
   single-asset, no signal** — the right family for `kelly_regime_ev` and
   the wrong one for any rule whose target moves with a signal. Corrected
   and extended by a literature commission run alongside R-67:
   - **For a signal-driven target the band does not vanish at zero
     position** (Muhle-Karbe, Reppen & Soner 2017, *Annual Review of
     Financial Economics* 9, 301–331, §5.2 and Eq. 4.15). R-66's reading
     is confirmed at that point but was stated too broadly — the band
     still vanishes, at two *other* levels set by the model parameters,
     so "keeps a strictly positive floor" should read "does not vanish at
     zero target weight."
   - **For a position-capped rule under linear costs, the optimal policy
     *is* a signal-space switching threshold** — de Lataillade, Deremble,
     Potters & Bouchaud (2012), *Journal of Investment Strategies* 1(3),
     91–115. This repo previously cited it only for the cube-root law.
     Its §6.3 also says the leading-order band is **symmetric** about the
     target and any asymmetry is higher order in Γ^(1/3).
   - **The asymmetric long/flat case is a theorem, under conditions.**
     Dai, Zhang & Zhu (2010), *SIAM Journal on Financial Mathematics*
     1(1), 780–810, and Guan, Peng & Xu (2020), arXiv:2008.07082 Thm 3.1:
     with proportional costs and a persistent hidden state, entry sits
     strictly above and exit strictly below the frictionless indifference
     point, the gap opened by the fee. It requires the signal to be a
     sufficient statistic and is single-asset — neither holds cleanly for
     a cross-sectional top-k rule.
   - **The band width is capped.** de Lataillade & Chaouki (2020),
     "Equations and Shape of the Optimal Band Strategy," arXiv:2003.04646
     — note the title, which R-66 recorded wrongly — Eq. (11): the optimal
     tolerance around zero **saturates at ≈1.6 σ_signal**. A larger fee
     does *not* justify a wider band, because the risk cost of waiting
     grows exponentially while the fee saving grows linearly.
   - **Gârleanu & Pedersen (2013) does not cover proportional costs**, in
     their own text: their partial-adjustment rule is "qualitatively
     different from the optimal strategy with proportional or fixed
     transaction costs, which exhibits periods of no trading." Any
     smoothing rule here is an EWMA of a discrete target, warranted by
     Dao et al. (2016) — not by GP.
9. **Turnover reduction is not the finding; net performance is** — and
   the closest published analogue to this repo's cost work is negative.
   Baltas & Kosowski put a significance deadband on time-series momentum
   across **75 futures**, cut turnover **~two-thirds** for ~5% of gross
   Sharpe (1.04 → 0.99), and still found it "does not lead to
   significantly higher risk-adjusted performance." Two calibrations to
   carry with it: Novy-Marx & Velikov (2016, *RFS* 29(1)) find banding
   **fails on high-turnover strategies** (four of six of their fastest
   anomalies stay net-negative after a 10%/50% band, and their stated
   cutoff is 50% one-sided monthly turnover); and in crypto specifically,
   Fieberg, Liedtke, Poddig, Walker & Zaremba (*JFQA* 60(7), 2025,
   3116–3153, **3,244 coins**) find that merely *halving* rebalancing
   frequency destroys **~39% of gross weekly return**, against ~10%
   erosion in NMV's equity anomalies. Budget crypto signal decay at
   30–40%, not 10%.
10. **A warmup prefix is not free.** Letting a strategy trade through the
   prefix of a resampled window lets it be liquidated before the window
   opens (19 of buy-and-hold's 23 stress-test liquidations were of this
   kind), and slicing a frame to an out-of-sample date range leaves a
   100-day-warmup strategy flat for 7.6% of it while a zero-warmup
   benchmark trades from day one. Warm the state, withhold the trading:
   `run_backtest(trade_start=...)` and `tradebot.window.run_period`.
11. **On one instrument, changing the *filter* is not an axis — only
    changing the *response* is** (added by R-89's literature pass, and it
    reorganises this whole file). Levine & Pedersen (2016), "Which Trend
    Is Your Friend?", *Financial Analysts Journal* 72(3), 51–66, show
    that time-series momentum, moving-average crossovers, HP filters and
    Kalman filters are **equivalent representations of one linear
    filter**, differing only in how they weight past returns. If that is
    right, a large share of this project's "new mechanism" rounds were
    re-parameterisations: R-83's Kalman local-linear-trend, R-06/R-07's
    anchor ladders and R-40's bagged ladders all live inside that family.
    The axes that are *not* re-parameterisations, on a single instrument,
    are three: the **nonlinearity of the response** (the map from trend
    strength to exposure), the **path-dependence of the exposure** (state
    carried between bars — a latch, a ratchet, a stop), and the
    **state-dependence of the horizon**. Before R-89 this project had
    varied none of them; R-89 took the first two.
    Two calibrations to carry alongside it. **Valeyre (2025),
    arXiv:2504.10914** (70 futures, 1990–2023) measures an optimal
    single-EMA trend system at portfolio Sharpe **1.24** but **≈0.20 per
    single asset**, with different EMA spans carrying **0.96**
    cross-correlation — the published, quantified version of R-05's
    lesson, and the discount to apply to any panel-derived trend edge
    before testing it here. **Kurth, Eisler, Rej & Bouchaud (2026),
    arXiv:2607.01550** give a microstructural account of fast trend's
    decay (EWM(5,20) Sharpe 0.84 → 0.12 post-2008) that is specific to
    **small-tick** instruments and show that zero-lag execution does not
    recover it — *signal* death, not cost erosion, which if it applied to
    BTC would mean none of R-56/R-64→R-68's execution work could ever
    have fixed what it aimed at. **R-89 tested that prediction directly
    on this data and it does not hold in-sample**: decomposing v4's own
    vote one anchor at a time, the 20-day anchor alone beats the
    three-anchor ensemble on Sharpe, final balance *and* drawdown on
    inner-train (+2.14/+2.32 vs +2.03/+2.28), while on inner-validation
    the ranking inverts completely and 20d becomes the worst of the
    three. The ensemble is never better than its best member in either
    window — it is better than the average member, and immune to which
    member happens to be right, which is the honest reason v4 votes
    rather than picks, and is N≈3 visible in a single table.

## What shipped

Three variants, each as a separate registered strategy so the incumbent's
published record stays intact: the convex vote response
(`kelly_regime_v2`, with its failed out-of-sample check reported),
conditional volatility targeting (`kelly_regime_v3`), and the doubling
anchor ladder (`kelly_regime_v4`, the current leader). See
[VALIDATION.md](VALIDATION.md#beta-testing-the-variants).

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
the ±0.2 noise floor from finding 2 above, so the return improvement is
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

---

# Ten more, in one round: game theory beyond no-regret, and intraday (R-188)

An operator-directed round (September 2026) asked for five more
game-theoretic or state-of-the-art rules and five intraday rules trading
one to ten times a day, all run through the same pre-registered protocol
(ROUTINE.md: tune on 2017–2020, select on 2021–2022, decide once on 2023+
against a rule frozen in advance). **All ten were dropped by the frozen
rule**; the code lives in `experiments/r188_*.py`, the harness and the
frozen rules in `experiments/r188_shared.py`, the numbers in
`docs/LEDGER.md` R-188 and `docs/VALIDATION.md`. What the survey found,
and what the data then said about it:

## Game theory and state-of-the-art sizing

- **Distributionally robust Kelly** — Rujeerapaiboon, Kuhn & Wiesemann
  (2016, Management Science 62(7)), "Robust Growth-Optimal Portfolios";
  Sun & Boyd (2018, arXiv:1812.10371), "Distributional Robust Kelly
  Gambling"; Baker & McHale (2013, Decision Analysis 10(3)), "Optimal
  Betting Under Parameter Uncertainty". Against nature choosing the
  return distribution inside an ambiguity set, the growth-optimal bet is
  Kelly on the *worst-case* drift; with estimation error it is Kelly on
  the drift's lower confidence bound. `robust_kelly` bet fraction-Kelly on
  `min_W(mu_W − kappa·sigma_W/sqrt(W))` over 10/30/90-day windows. **Bar-
  visible consequence:** BTC's daily drift is about a twentieth of its
  daily volatility, so the bound is positive only in the strongest bull
  runs — at `kappa=1` the rule sat flat on 90% of bars, and at the frozen
  `kappa=0.5` it was in the market 18% of the time and still lost to
  holding on both training slices and the holdout. Robustness to
  estimation error is bought entirely with time out of the market, which
  is R-33's exposure lesson arriving from the theory side.
- **Coin betting / parameter-free online learning** — Orabona & Pál
  (2016, NeurIPS), "Coin Betting and Parameter-Free Online Learning";
  Krichevsky & Trofimov (1981); Cutkosky & Orabona (2018). A bettor who
  wagers the KT fraction `S_t/t` of wealth on an adversarial coin is
  within `sqrt(t)` of the best constant bet with no learning rate.
  `coin_betting` did this on daily rounds with a 0.99 forgetting factor.
  It was the only candidate profitable on the holdout at spot ($1,148
  from $1,000; 6% max drawdown) — and its Sharpe (0.63) sat 0.40 below
  buy-and-hold's (1.03), twice the noise floor, because a nearly fair
  coin earns a nearly zero KT bet: mean exposure ~15%. Sizing that is
  provably safe against every sequence is, on this sequence, mostly cash.
- **Level-k / cognitive hierarchy** — Nagel (1995, AER 85(5)); Camerer,
  Ho & Chong (2004, QJE 119(3)); Hommes (2011, JEDC) for the
  learning-to-forecast evidence. `level_k` tracked the fee-charged PnL of
  a slow-trend chaser (level 0), a fast front-runner (level 1) and a fader
  of level 1 (level 2), and played the best response to the level
  currently paying. **Finding:** the anticipating rule lost to its own
  follow-the-leader control on both markets in inner-validation, and both
  died of fees (2–5 trades a day at 0.10%: $5 from $1,000 on every slice).
  Gross of fees the follow-the-leader version made 5x on 2017–2020 and
  +2% on 2021–2022, so there is a fast-horizon signal — worth less than one
  taker fee per trade.
- **Round-number focal points** — Schelling (1960), *The Strategy of
  Conflict*; Osler (2003, J. Finance 58(5)), "Currency Orders and
  Exchange Rate Dynamics", whose order-book evidence is that take-profit
  orders cluster at round numbers and stop-losses just beyond them.
  `focal_levels` traded both predictions (bounce at a level; breakout
  through one) on a 1-2-5 grid of round BTC prices. **Neither survives
  even gross of fees** ($936 and $1,000 from $1,000 on the two training
  slices at zero fee, breakout mode); with fees, 1–3 trades a day is fee
  death. Osler's mechanism needs order-book position; the price footprint
  alone carries nothing tradeable here.
- **Mean-field-game crowding** — Casgrain & Jaimungal (2020, Mathematical
  Finance 30(3)), "Mean-field games with differing beliefs for algorithmic
  trading"; Cardaliaguet & Lehalle (2018). The equilibrium inventory leans
  into one's own drift belief net of the crowd's transient impact.
  `mfg_crowding` held `tanh(20-day trend) − gamma·tanh(2-day trend)`. The
  `gamma=0` control made 22x on 2017–2020; every `gamma>0` made less, and
  on inner-validation `gamma=0.5` won the selection only by losing least
  (−0.32 mean Sharpe). On the holdout it was profitable ($1,100) at a
  Sharpe 0.80 below holding. Subtracting the crowd's chase is a
  de-levering of the trend, and R-62's lesson holds: the trend vote
  carries the signature, the scale factor does not.

## Intraday, one to ten trades a day

- **Noise-area breakout** — Zarattini, Barbon & Aziz (2024, SSRN
  4824172), "Beat the Market: An Effective Intraday Momentum Strategy for
  the S&P500 ETF (SPY)" (Sharpe 1.33 vs 0.60 for holding SPY, ~one trade a
  day, ~$0.0035/share commissions). `noise_area_breakout` reproduced the
  band construction on UTC sessions. It lost on every slice with fees and
  made +38%/+7% *gross* on the two training slices at ~1 trade a day: the
  edge exists and is roughly a third of one round-trip taker fee per trade.
- **Market intraday momentum** — Gao, Han, Li & Zhou (2018, JFE 129(2)):
  the first half-hour predicts the last half-hour on SPY. `intraday_
  momentum` traded the last four UTC hours in the direction of the first
  four. Gross it doubled on 2017–2020 and was flat on 2021–2022; net it
  lost on all three slices at 0.28 trades a day.
- **Hour-of-day drift** — Eross, McGroarty, Urquhart & Wolfe (2019, RIBAF
  49); Baur, Cahill, Godfrey & Liu (2019, FRL 31). R-75 had found BTC's
  hour-of-day *volatility* real and its day-of-week *return* pattern
  noise. This round measured the hour-of-day *return* dispersion the same
  way: observed 2.34e-5 against a block-shift null with mean 2.17e-5 and
  p95 2.53e-5, empirical p = 0.32 — **not distinguishable from noise**,
  and `session_drift` lost on every slice accordingly, gross included
  (−15% gross on 2017–2020). Calendar/session return effects on this
  series are now closed on both axes the timestamp offers.
- **VWAP price pressure** — Hendershott & Menkveld (2014, JFE 114(3)),
  "Price pressures"; Kakushadze & Serur (2018), *151 Trading Strategies*.
  `vwap_reversion` faded 3-sigma deviations from the session VWAP only
  when the deviation exceeded 1.5 round-trip fees. Gross it made 6x on
  2017–2020 (mostly the 2017–2018 chop) and lost on 2021–2022; net it
  lost everywhere. Reversion to an intraday anchor is real in the
  high-volatility years and absent since.
- **Post-jump continuation** — Lee & Mykland (2008, RFS 21(6)) for the
  test; Scaillet, Treccani & Trevisan (2020, J. Financial Econometrics
  18(2)) for Bitcoin's jump dynamics. `jump_momentum` followed a flagged
  5.4-sigma jump for an hour. **The one intraday rule with a clean gross
  edge on both training slices** (Sharpe 1.25 and 1.33 at zero fee, 0.4
  trades a day) — and the sign of the holdout was against it net of fees
  ($551), so continuation after a jump is either fee-sized or absent in
  2023+. `overshoot_fade` (L-13) was the mirror image; both directions are
  now recorded.

## The arithmetic the round confirms

At a 0.10% taker fee a round trip costs 0.20% of notional. One trade a day
is 73% of notional a year in fees; five a day is 365%. Of the seven
candidates trading above 0.25 times a day, five had a positive gross edge
on at least one training slice and **none** had one larger than the fee it
paid. That is the ledger's COST constraint measured on a fresh set of
mechanisms: the signals exist at the hour-to-day horizon, and each one is
smaller than one taker fee. Intraday trading on this venue tier would need
maker fills or a fee tier near zero before any of these rules could be
retested with a different expectation.
