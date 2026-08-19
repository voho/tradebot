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

Backfilled 2026-08-16 from the long-form docs. `STRATEGIES.md`,
`RESEARCH.md`, `VALIDATION.md` and `LIVE.md` remain the long-form record;
the former `ALTERNATIVES.md`, `CROSS_ASSET.md`, `ELLIOTT_WAVES.md` and
`FRONTIER.md` were folded into this file and `VALIDATION.md` in the
2026-08-17 docs restructure (their findings live in rows R-15–R-18, the
standing diagnosis, section C and the backlog; the measured tables moved
to `VALIDATION.md`). Balances are $1,000 start, full period, from the
README comparison table.

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

**Sharpened by R-33 (08-19), and it is the single most reusable line in
this file:** *before believing any comparison here, check whether the two
arms carry the same risk.* Three of this project's findings have now died
the same death — R-28's e-process drawdown cut (R-31), R-32's gate
comparison, and L-04's own headline (R-33) — and in every case the
mechanism turned out to be an exposure level. Holding less draws down
less; that is arithmetic, not evidence, and it is invisible until the
benchmark is de-levered to match.

---

## A. Strategies (registered)

| ID | strategy | added | idea | attacks | spot | fut 5x | verdict | lesson |
|---|---|---|---|---|---|---|---|---|
| L-01 | `kelly_regime_v4` | 08-15 | v3 on a doubling anchor ladder (20/40/80d), Müller 1997 / Corsi 2009 HAR | SIZE | $66.8K | $156.2K | **PROMOTED** | Drawdown 35.3% is the robust finding; the return improvement sits inside the ±0.2 Sharpe noise floor and is *not* established. |
| L-02 | `kelly_regime_v3` | 08-15 | Conditional vol targeting — constant notional through normal vol, re-size only on breakout (Bongaerts 2020) | SIZE | $65.8K | $139.5K | **PROMOTED** | Improves every metric in both sub-periods and both markets; flat parameter neighbourhood (8 combos, Sharpe 1.47–1.55). |
| L-03 | `kelly_regime_v2` | 08-15 | Convex vote response: partial anchor agreement = low confidence, not half a signal | SIZE | $46.4K | $122.0K | NOT PROMOTED | Nine of ten metrics improve and it still fails: −6.5% out-of-sample. Kept registered with the failure stated. |
| L-04 | `kelly_regime` | 08-14 | Fractional-Kelly vol-targeted sizing gated on a crowd-regime filter (Cardaliaguet & Lehalle 2018) | SIZE | $42.1K | $108.2K | **PROMOTED** (incumbent) | First strategy to beat the benchmark. Its headline — "regime-gated sizing cuts drawdown" — was the project's one robust finding until **R-33 risk-matched the benchmark**: 88–92% of that gap is holding half the notional, and the remainder is not established on the holdout. What survives matching is a *return* advantage at equal risk (+20.8pp/+23.8pp median across 40 windows), which is a different claim and is not yet pre-registered. |
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
| R-15 | Funding harvest / cash-and-carry | 08-16 | Compounded the real series, 2020–2023 | +82.0% over 4.0y = **+16.2%/yr**; +14.6% after 0.10% on both legs (quarterly rebalance), +9.8% at 0.40%; payer flips 13.5% of settlements; **worst 30-day run −1.31%**. Literature (He et al. 2024 and the 2020–2025 empirical carry work) reports carry Sharpe ~6.45, falling to 4.06 from 2024 and **negative in 2025** as it crowded — and our data stops exactly at 2023. Full table in `VALIDATION.md` (funding section). | **BLOCKED** on data → B-02 |
| R-16 | Funding as a positioning signal | 08-16 | Quintile and momentum-controlled sort, 2020–2023 | 14-day forward spread Q1−Q5 = **+3.57pp**; high funding predicts negative forward returns unless price is also rising; correlation with trailing return only 0.39, so not a momentum proxy. But middle quintiles are non-monotone (Q3 +3.06%, Q4 −1.02% at tied clamped rates) — a warning about how much is noise. Full tables in `VALIDATION.md` (funding section). | OPEN hypothesis → B-05 |
| R-17 | Cross-asset falsification on ETH | 08-16 | Bitfinex BTC + ETH, same venue, same window (2016-03→2019-12) | **The risk property transfers, the return property does not exist.** Drawdown cut in all four cells (BTC 83.8→40.1, ETH 94.2→36.5, 5x 85.2→32.1 and 99.3→35.1). Loses to holding on spot on both assets (0.58x, 0.47x). The 236x ETH futures cell is survival, not edge. | **PARTLY ANSWERS N≈3** |
| R-18 | Elliott Wave Theory (± NN, ± game theory) | 08-16 | Assessed against this repo's bar | Not falsifiable as practised (Aronson: a story prone to subjective revision) — counts are re-labelled after the fact, the exact leak class `test_causality_strict.py` exists to catch. Its one quantitative component (Fibonacci ratios) was refuted by Batchelor & Ramyar. *ElliottAgents* (Applied Sciences 14(24), Dec 2024; multi-agent LLM + deep RL) reports 73.68% vs 57.89% on BTC/USD Oct 2022–Sep 2024 — that is **14/19 vs 11/19**, three extra calls, over a monotonic $20K→$70K rise, with no walk-forward. Training an NN on wave labels adds nothing a network cannot learn from price directly, while importing a subjective hindsight-contaminated annotation step. Its useful kernel (multi-timescale crowd structure) is already `kelly_regime_v4`. | NOT PURSUED |
| R-19 | Monte Carlo window stress test | 08-14 | 40 random windows, identical across strategies | Leveraged buy-and-hold **liquidated in 26 of 40**, median window −98%. The three resampled `kelly_regime` variants (v2/v3/v4) survived all 40; on 5x futures profitable in 85–88% and beat holding in 65% (spot: beat holding in 48–50%). | **KEY FINDING** |
| R-20 | Noise floor measurement | 08-15 | Paired stationary block bootstrap, 30-day blocks, 2,000 resamples | **±0.2 Sharpe.** Smaller differences on one path are not evidence. The analytic SE of a Sharpe *level* (±0.02) is misleadingly tight for *comparing* strategies. | **METHOD** — binds every claim here |
| R-21 | Lookahead probes | 08-15 | Two adversarial probes | A one-day signal broadcast onto 5m bars is worth **+2.1 Sharpe** and *passes* truncation. A strategy that keeps the `prepare()` frame and indexes `i+1` in `on_bar` returned **$3.7e23 at Sharpe 73 with a green suite**. Both now caught by `test_causality_real.py` / `test_causality_strict.py`. | **METHOD** |
| R-22 | Warmup-prefix bias | 08-15 | Audit | Letting a strategy trade the warmup prefix let it be liquidated *before* the window opened — **19 of buy-and-hold's 23 stress liquidations were this artifact**. Slicing to an OOS range left a 100-day-warmup strategy flat for 7.6% of it. Fixed by `run_backtest(trade_start=...)` / `tradebot.window.run_period`; verdicts survived, numbers moved ~75%. | **METHOD** |
| R-23 | Capital scaling | 08-15 | $1K vs $1M across every strategy | Results are proportional to capital; the only deviations came from the exchange minimum order size. One start balance is therefore sufficient. | SETTLED |
| R-24 | Exchange adapter parity | 08-15 | Bar-for-bar over 30 consecutive candles, both adapters | Top-three strategies compute the identical target from paged exchange data and from the contiguous backtest frame; paging is lossless; neither adapter hands a strategy the forming candle. | SETTLED |
| R-25 | Deflated Sharpe / purged CV / bootstrap CIs | — | Cited in `RESEARCH.md`, **never computed** | Every sweep here is a trial that inflates whatever it selected (R-12 ran 32). The comparison table reports points where it should report ranges. | **CLOSED by R-29** |
| R-26 | Parallel round on B-01, B-02/03, B-04, B-05, B-07 | 08-17 | 11 agent-sessions dispatched (5 build, 5 skeptic, 1 synthesis). Every one was blocked before executing a single call: the permission handler returned `updatedInput` with required parameters stripped, so `Bash`, `Read`, `Glob` and `Grep` all failed schema validation. Repo verified untouched afterwards. Fault has since cleared. | **0 trials, 0 configurations, 0 bars read.** The five directions were **NOT TESTED** and stay on the backlog as untried — filing them as negatives would stop a future agent trying them. Holdout counter unchanged (nothing was read). Project trials count unchanged. | **NULL ROUND** |
| R-27 | Fabrication pressure in the operator's own prompt | 08-17 | The synthesis prompt for R-26 contained a conditional naming the hoped-for answer: *"If the inference agent found that most of the table's ordering is not distinguishable from noise, say so first and plainly."* The inference agent had run zero backtests. | The synthesizer refused and flagged it. Had it complied, a fabricated headline would have entered `docs/VALIDATION.md` — the file whose whole purpose is being trustworthy — indistinguishable from a real result to a later reader. Same failure class as L-14/L-15/L-16 (proxying order flow out of price) and R-21 (the $3.7e23 probe), but arriving through the *prompt* rather than the code. | **METHOD** — see ROUTINE.md |
| R-28 | E-process regime detection with unified Kelly sizing (Shafer 2021; Ramdas et al. 2023; Waudby-Smith & Ramdas 2024; Shin, Ramdas & Rinaldo 2024) | 08-17 | Three variants in `experiments/eprocess_regime.py`, 24 configurations on the inner split, one frozen config on the holdout | **The deepest drawdown reduction in the project, and it still loses.** Holdout spot DD **11.6%** vs `kelly_regime_v4`'s 27.8% and holding's 54.0%; deeper than v4 in **0 of 40** Monte Carlo windows (median −14.0pp spot, −11.3pp futures). Return is 0.42x holding, so P1 fails. Anytime-valid evidence justifies only **0.27x** the incumbent's mean exposure. | **NEGATIVE** — and the risk finding was **retracted by R-31**: at matched risk it does not replicate on ETH and reverses in 45–82% of the stress windows |
| R-29 | Trials-aware inference: block-bootstrap intervals, deflated Sharpe, combinatorially purged CV (Politis & Romano 1994; Bailey & López de Prado 2014; López de Prado 2018) | 08-17 | `src/tradebot/inference.py` + `scripts/inference.py`, applied to all 25 registered strategies on both markets: 96 paired comparisons, 100 deflated Sharpes, 45 CPCV splits | **10 of 96 adjacent pairs in the ranking are distinguishable at 95%, and none of them separates two of the table's top eight from each other.** The table's *final-balance* claim for `kelly_regime_v4` over holding on spot is a coin flip (P=0.52). The drawdown claim survives on the full history (−41.1pp [−54.8, −18.4]) and on the futures holdout, but **not** on the spot holdout (−27.1pp [−35.8, **+1.9**]). Cross-validating the table's own selection rule: it beats holding in **6 of 45** folds. | **METHOD** — the ordering is mostly noise, and now says so |
| R-30 | Wire the intervals into the comparison table itself (backlog B-12) | 08-18 | `src/tradebot/evidence.py` reads R-29's `bootstrap.csv` into `tradebot run`: two verdict columns on the README table (Δ log growth and Δ max drawdown against `buy_and_hold`, each with its 95% paired interval and a ▲/≈/▼ mark), the full error bars in the per-market detail tables, the log-growth interval added to the bootstrap output — R-29 computed it and saved only the point — and a CI rule that a registered strategy with no measured interval fails the suite. 18 new tests. | **The column R-29 computed and discarded says more than the ones it kept.** On spot over the full history, **0 of 24 strategies are distinguishably better than `buy_and_hold` on log growth**, the criterion the table ranks by; 13 are distinguishably worse and the 11 indistinguishable ones are the entire profitable block. `kelly_regime_v4`'s +0.044 advantage is **[−2.60, +2.85]** — from a thirteenth of holding's final balance to seventeen times it. Everything R-29 published reproduced exactly. | **METHOD** — the warning now lives *in* the table, not beside it |
| R-31 | Matched-risk frontier: e-process gate vs latched anchor vote at equal realized volatility (backlog B-11) | 08-18 | `experiments/matched_risk.py` — one sizer, one deadband, one warmup, one exposure knob, gate interchangeable. 36 configurations traced on both inner splits and both markets (144 backtests), exposures solved on inner-validation to within 2% of target volatility in both directions, then frozen; holdout scored with the R-29 paired block bootstrap | **Hold risk fixed and R-28's headline dissolves — both halves of it.** All 8 holdout intervals contain zero and the sign is unstable across cells; the one cell surviving the pre-registered validity gate gives −0.072 [−0.532, +0.379] on log growth. Three cells of four are **void**: the inner-validation exposure match did not survive into 2023+ (29% volatility gaps) or the spot notional cap truncated both arms differently (41% / 27% of bars). On ETH, with exposures re-matched, the e-process gate loses all four cells on return **and on drawdown** — so R-28's P3 replication was measured against an arm carrying 2.4x the risk. Equal-risk exposure ratio is itself regime-dependent: 2.2x in the bull, 4.7x in the bear. | **NEGATIVE** — the 0.27x exposure *was* the finding |
| R-33 | Matched-risk benchmark: `kelly_regime_v4` against a **de-levered** `buy_and_hold` at equal realized volatility (backlog B-13) | 08-19 | `experiments/matched_hold.py` — a passive long holding a constant fraction `c` of equity, in two readings (rebalanced to constant risk, and static buy-once), exposure solved on inner-validation so its realized volatility equals v4's, then frozen; 18 configurations on the inner splits; holdout scored with the R-29 paired block bootstrap; 40 windows re-matched **inside each window** to 0.5% | **This project's headline is ~90% arithmetic, and what is underneath it is a different claim.** Across 40 identical windows at genuinely equal risk, v4's median drawdown advantage falls from **−24.5pp to −2.9pp** (spot) and **−70.7pp to −5.5pp** (futures) — 88% and 92% of the gap was the exposure level. On the holdout, five of six frozen cells fail the pre-registered risk match (a vol-targeter and a constant-exposure hold cannot be matched across a regime change), and the one valid cell gives **−14.18pp [−22.68, +13.48]**, containing zero. But the *return* comparison, which nobody pre-registered, goes v4's way in every cell of every table: **+20.8pp / +23.8pp median per window in 82% / 90% of them**, all four ETH/BTC cells, and it survives the ETH falsification test R-28 failed. | **NEGATIVE** on D1 — the drawdown claim is downgraded to "against a fully-invested benchmark only". The finding underneath it is return-per-unit-risk, and it needs its own pre-registered round (**B-14**). |
| R-32 | The ungated control, and an independent second reading of B-11 | 08-18 | A parallel session ran the same backlog row the same day from the same base commit. Same design as R-31 (one sizer, gate interchangeable, exposure scaled by a scalar) plus a **third arm with no gate at all**; 33 configurations, 132 backtests, multipliers frozen on inner-validation | **Agrees with R-31 wherever the two overlap** — gates indistinguishable at matched risk, R-28's 0-of-40 inverted (deeper in 60%/62%), its fee advantage inverted, P1 failed — from an independent implementation, and its own holdout cells are **void** under R-31's validity rule (cap binds on 41%/36%/21% of spot bars; a 29% volatility gap on futures). What it adds: at matched risk the **ungated** arm is below both gated arms at every risk level in all four inner-split cells and loses 80–90% of 40 paired windows. **The gate is worth more than the choice of gate.** | **NEGATIVE** — and the parallel-branch report the routine requires |
| R-34 | `harsanyi_crowd`'s Bayesian bull/bear/chop posterior (L-12) as a SIZE input on `kelly_regime_v4`, instead of the DIRECTION input that lost — L-12's own recorded lesson, tested for the first time | 08-19 | Two parallel unregistered variants, each on a disjoint file: `experiments/kelly_regime_v5_damp.py` (conservative — a bounded multiplicative dampener, `mult∈[1−lam,1]`, applied on top of v4's unchanged vote) and `experiments/kelly_regime_v5_bayes.py` (novel — the discrete vote replaced entirely by a continuous, hysteresis-latched posterior margin feeding v4's unchanged conditional-vol-targeting sizer); 42 configurations across inner-train/inner-validation, both markets, plus ETH/BTC Bitfinex falsification and an explicit matched-mean-exposure check on each branch | **Conservative:** never beats v4 on return in any of 12 measured cells; its drawdown "improvement" is architecturally guaranteed (the multiplier can only shrink exposure) and the resulting exposure series correlates **R²=0.997** with a flat 0.7x rescale of v4 — the same exposure-level artifact as L-04/R-33, R-28/R-31 and R-32, reproduced with a new source signal. **Novel:** genuinely independent of the vote (correlation **−0.0017**, not a smoothed duplicate) but underperforms v4 in all 36 configurations (inner-validation spot Sharpe −2.9 to −3.9 vs v4's +0.14, turnover 4–7x), and explicitly re-scaling exposure to match v4's mean (`exposure_mult=5.27`) makes it *worse* (Sharpe −6.25, DD 92%), ruling out the exposure-artifact explanation for this branch — the margin is simply too noisy at its native hours-to-days cadence to pay 5-minute-bar trading costs on either axis. | **NEGATIVE** (both branches). Holdout untouched by either branch. |
| R-35 | Funding rate as a COST-aware SIZE input on `kelly_regime_v4` (backlog B-05) | 08-19 | Two parallel unregistered variants, each on a disjoint file: `experiments/funding_gate_decile.py` (conservative — literal backlog reading, stand flat when trailing funding percentile clears the 90th) and `experiments/funding_ev_band.py` (novel — extends L-05's analytic no-trade band with a forecast funding-drag term); 80 configurations total across both branches on inner-train/inner-validation, plus one pre-registered holdout read (2023-01-01..2023-12-31, the funding-covered slice only) for the branch that cleared | **Conservative clears every inner-validation and falsification check — genuinely not the exposure-level artifact its own pre-registration predicted (§5 flat-rescale test) — then loses on the single holdout year it earned: Δ log growth vs v4 is negative on both markets, −0.167 [−0.495, +0.101] futures, and stays negative funding-charged. Novel branch is a real, non-exposure-artifact Sharpe edge (§7 rescale diagnostic) that fails its own plateau check and, per its author's recommendation, never reached the holdout.** | **NEGATIVE** (both branches) — see full write-up below |
| R-36 | Formalize B-14: is `kelly_regime_v4`'s return-per-unit-of-risk edge over a matched passive hold (R-33's byproduct finding) real outside the 2017–2020 bull? | 08-19 | Pre-registered a pooled decision rule (exact-binomial 95% CI on R-33's existing 40-window win-rate) plus a named falsification test (split the same 40 windows by start date, before/after 2021-01-01); reused `windows.csv` unchanged, only recovered each window's calendar date from the identical seed=42 RNG sequence — 0 new backtests | **D1 passes on both markets (CI excludes 50%). Falsification survives on both markets — post-2021 windows still favour v4 (win-rate 68.2%/81.8%, median +5.0pp/+7.4pp) — but the effect is ~10x smaller than the pooled/pre-2021 number (+68.9pp/+97.2pp), and the post-2021 subsample's own CI still contains 50% on spot at n=22.** | **CONFIRMED, thinned** — see full write-up below |
| R-37 | Two SIZE-axis attempts to capture more of R-36's confirmed (but thinned) edge on `kelly_regime_v4` | 08-19 | Two parallel unregistered variants, each on a disjoint file: `experiments/kelly_regime_v6_retune.py` (conservative — retunes the existing `target_vol`/`max_leverage` constants, no new signal) and `experiments/kelly_regime_v6_state_kelly.py` (novel — replaces the single global `target_vol` with a causally-estimated, per-vote-state Kelly fraction `μ_state/σ_state²`); 99 configurations total across both branches (53 + 46), inner-train/inner-validation and ETH falsification only, no holdout read by either | **Conservative: the naive best-Sharpe candidate reproduces the project's standard exposure-level artifact (+51% realized vol) and does not transfer to futures; the one candidate surviving a matched-exposure control nets a Sharpe delta inside the ±0.2 noise floor on both markets and does not clear ETH by more than a token margin. Novel: `max_leverage` never binds (rules out the raw-leverage artifact cleanly) and states genuinely differ in measured μ/σ² (bear ≈ −62%/yr, bull ≈ +154–174%/yr, non-monotone — 2/3 agreement beats unanimous 3/3) — but it fails its pre-registered ETH falsification outright, underperforming v4 on the BTC control too, and its halflife/kelly_mult neighbourhood is a fitted peak, not a plateau.** | **NEGATIVE** (both branches) — see full write-up below |
| R-38 | Risk-constrained Kelly gambling (Busseti, Ryu & Boyd 2016) as a formal, probability-calibrated replacement for `kelly_regime_v4`'s ad hoc `target_vol`/`max_leverage` constants | 08-19 | Two parallel unregistered variants, each on a disjoint file: `experiments/kelly_regime_v7_ddcap.py` (conservative — v4's vote and scale unchanged, additionally capped by a causal drawdown-risk ceiling `f_risk = mu/(lambda·sigma²)` with `lambda = ln(beta)/ln(alpha)` fixed from a stated drawdown tolerance) and `experiments/kelly_regime_v7_crra.py` (novel — v4's vote kept as a hard gate, its vol-only scale replaced entirely by the same CRRA fraction as the sizing formula); 56 configurations total across both branches (24 + 32), inner-train/inner-validation and ETH/BTC falsification only, no holdout read by either | **Both branches cleanly refute the standard exposure-level-rescale artifact (R²=0.20 and 0.15 against a mean-notional-matched flat rescale of v4, versus the 0.95+ threshold this project treats as diagnostic) — a genuinely non-duplicate mechanism in both cases. Both still fail their identical pre-registered ETH falsification decisively, and by the same diagnostic signature: each loses to `kelly_regime_v4` on the BTC control itself (conservative: ≈11–12% of v4's balance; novel: 21–37%), before ETH is even read — the inner-validation win (built on a bear/chop-heavy 2021–22 window) does not survive a trending market on either tested asset. Conservative's parameter neighbourhood is additionally not a plateau (adjacent (α,β) cells swing spot Sharpe +0.50→−0.07); novel's is a loose plateau but sits at the edge of its tested grid.** | **NEGATIVE** (both branches) — see full write-up below |
| R-40 | Bag/ensemble R-07's already-validated 18–28d anchor-ladder plateau, instead of shipping one frozen point on it (ERR: no error control on the ladder-choice hyperparameter itself) | 08-19 | Two parallel unregistered variants, each on a disjoint file: `experiments/kelly_regime_v8_ladder_bag.py` (conservative — plain unweighted average of the latched vote across a fixed 6-ladder ensemble spanning R-07's region; Breiman 1996 bagging) and `experiments/kelly_regime_v8_uncertainty_shrink.py` (novel — the same bagged vote further shrunk by real-time cross-ladder disagreement, Baker & McHale 2013 / Sukhov 2025 parameter-uncertainty-under-Kelly style; `κ=0` verified to reduce exactly to the conservative mechanism); 12 configurations total across both branches (4 + 8), inner-train/inner-validation and ETH/BTC Bitfinex falsification only, no holdout read by either; operator independently re-ran both branches' `select`/`eth` commands and confirmed the numbers | **Both branches beat `kelly_regime_v4` on every inner-validation cell (conservative's primary candidate: spot Sharpe 0.30 vs 0.14, futures 0.42 vs 0.25) and neither is the standard exposure-level artifact (R²=0.86–0.94, below the 0.95 bar) — but both hit the same diagnostic signature that sank R-37/R-38: substantial underperformance vs v4 on the pre-2020 BTC falsification control itself, worst on futures (conservative 52–75% of v4's balance across all four ensemble definitions; novel 56%), before ETH is even read. The disagreement-shrink term added nothing over the plain bag in 6 of 6 non-zero-κ configurations.** | **NEGATIVE** (both branches) — see full write-up below |
| R-41 | B-15: build a real Deribit BTC/ETH-PERPETUAL price series (network access to Deribit/Kraken/Bitstamp/Coinbase confirmed open this session — only Binance still 451s), then use the resulting real spot/perp basis — the first genuinely independent second price series this project has ever had, attacking INFO directly rather than the SIZE-axis reweighting of R-34/R-35/R-37/R-38/R-40 — as a new SIZE input on `kelly_regime_v4` | 08-19 | Infra: `scripts/fetch_deribit_perp_price.py` + `scripts/fetch_coinbase_spot.py` fetched and committed `data/btcusdt_deribit_perp_5m.csv.gz` (842,851 bars, 2018-08-14→2026-08-19, zero gaps), `data/ethusdt_deribit_perp_5m.csv.gz` (781,765 bars, 2019-03-14→2026-08-19) and `data/ethusd_coinbase_spot_5m.csv.gz` (781,506 bars, matching span) — `tradebot.data.load_deribit_perp_price()`/`compute_basis()` added. Then two parallel unregistered variants, each on a disjoint file: `experiments/kelly_regime_v9_basis_brake.py` (conservative — bounded never-increase dampener `mult∈[1−λ,1]` on v4's unchanged vote+scale, triggered symmetrically by extreme \|basis\| in either direction) and `experiments/kelly_regime_v9_basis_lead.py` (novel — step 2 first asked whether basis genuinely *leads* the vote's own flip dates before writing any strategy code); 30 configurations total across both branches (18 conservative + 12 novel, plus a lead-lag descriptive sweep not counted toward trials), inner-train-with-basis (2018-08-14→2020-12-31, the window actually covered) / inner-validation (2021-01-01→2022-12-31), both markets, no holdout read by either. Operator independently re-ran both branches' `artifact`/`fallback`/`causality` (conservative) and `exposure`/`leadlag` (novel) commands and reproduced every reported number exactly. | **Both branches are genuinely non-duplicate by the measurements that matter — basis correlates only r≈0.06–0.12 (daily) with the already-tested funding-rate signal (R-35), far below a restatement — and both fail cleanly for different, well-diagnosed reasons. Conservative: the standard exposure-level artifact in all 18 configurations on the mandated inner-validation check (R²=0.981–0.999) *and* the R-37/R-38/R-40 train-loses/validation-wins signature (loses to v4 on final balance in 36/36 train cells, beats it in only 14/36 validation cells) — real cross-venue basis blowouts are too rare (752 of 842,851 bars, |basis|>10%) for a bounded, symmetric, never-increase brake to move a multi-year aggregate far enough from a flat rescale, while still costing return when it does fire during the COVID V-shaped recovery. Novel: the step-2 lead-lag study is a clean null (basis-confirmed hit rate scatters 39–55% around a ~51% base rate against a block-bootstrap null, median lead time ≈0 days — contemporaneous, not leading); a candidate built anyway despite the null beats v4 in every train/validation cell (not the R-37/38/40 signature) but the Sharpe deltas sit inside the ±0.2 noise floor everywhere and the exposure series is R²=0.977 collinear with v4's own target (the artifact bar), so the uplift is not established as a distinct mechanism. Both authors independently recommended against spending an ETH falsification or holdout consultation on their own branch, and the operator agreed rather than spending the newly-built ETH data on either.** | **NEGATIVE** (both branches). Holdout untouched. Real ETH basis data (7.4y coverage, comparable to BTC's) is now committed and available for a future round with a mechanism that survives its own inner-validation diagnostics first. |
| R-42 | B-07: fetch real on-chain data (CoinMetrics community API, free, confirmed reachable alongside Bitstamp/Deribit/Kraken/blockchain.info this session), then use the resulting daily MVRV ratio — market cap / realized cap, computed from actual blockchain transaction history rather than trade price, the first genuinely non-price data channel this project has had (Mahmudov & Puell 2018; Grobys 2026, *Int. Rev. Financial Analysis*, "Using on-chain data to predict Bitcoin cycles" — MVRV Z-score strongest of three on-chain rules over three full BTC cycles 2013–2025) — as a new SIZE input on `kelly_regime_v4` | 08-19 | Infra: `scripts/fetch_coinmetrics_onchain.py` fetched and committed `data/btcusd_onchain_daily.csv.gz` (6,074 daily rows, 2010-07-18→2026-08-19) and `data/ethusd_onchain_daily.csv.gz` (4,038 rows, 2015-08-08→2026-08-19) — `tradebot.data.load_onchain()` added (causal: shifts CoinMetrics' own day-start timestamp forward one day). Then two parallel unregistered variants, each on a disjoint file: `experiments/kelly_regime_v10_mvrv_brake.py` (conservative — bounded, *asymmetric* never-increase dampener `mult∈[1−λ,1]` on v4's unchanged vote+scale, firing only at high-MVRV overvaluation extremes, never at low MVRV) and `experiments/kelly_regime_v10_mvrv_lead.py` (novel — step 2 first tested, at the literature's own weeks-to-months cycle timescale, whether MVRV extremes lead the vote's own regime flips, block-bootstrap null against 15–21 debounced flip events); 80 configurations total across both branches (55 conservative + 25 novel, plus 18 lead-lag descriptive configs not counted toward trials, per the R-41 convention), inner-train (2017–2020) / inner-validation (2021–2022), both markets; ETH falsification (Coinbase spot + CoinMetrics ETH MVRV) by both; no holdout read by either. Operator independently re-ran both branches' `eth` commands and the novel branch's pre-registered `duplicate` check and reproduced every reported number exactly. | **Both branches are cleanly NEGATIVE, corroborating each other from different angles. Conservative: hits all three of its own reject conditions — the standard exposure-level artifact on its selected candidate (R²=0.973, inner-validation, >0.95 threshold), the identical R-37/R-38/R-40 train-loses/validation-wins signature (0/54 configs beat v4 on TRAIN final balance; only 2–3/54 beat it on VALID, not a plateau), and the pre-registered ETH falsification fails in both market cells ($4,861/$4,213 vs v4's $5,091/$5,009). Its own volatility-correlation diagnostic explains why: MVRV-Z correlates +0.06 to +0.58 with realized volatility (strongest with *forward* vol), so the brake partly re-levers down into BTC's inverse-leverage high-vol/high-forward-Sharpe states — the exact mechanism R-08 already showed hurts v4, reproduced here with a new signal. Novel: the pre-registered weeks-to-months lead-lag study is a clean null (observed confirm-rates of 71–75% sit *at or below* a 200-resample block-bootstrap null of 78–88%, i.e. MVRV's own high autocorrelation explains the apparent hit rate — no real lead exists at the cycle timescale the literature claims). Built an agreement/disagreement valuation gate anyway per its own pre-registered contingency (architecturally distinct from the conservative brake — can amplify above 1.0, not just dampen) and it loses to v4 in 96/96 sweep cells outright, additionally triggers its own pre-registered price-proxy duplicate test (R²=0.964 on TRAIN — a price/MA(730d) ratio alone reproduces the candidate's exposure series above the 0.95 threshold), and fails ETH decisively ($1,241/$1,194 vs v4's $5,091/$5,009). Root cause, stated in the novel branch's own writeup: MVRV read "overvalued" through most of 2017's historic bull run — a real, correctly-computed valuation reading, not a bug — so any mechanism that dampens on high MVRV cuts exposure exactly when v4's trend-following was earning the most.** | **NEGATIVE** (both branches). Holdout untouched. This is the sixth independent, non-duplicate parallel round (twelve branches total: R-34, R-37, R-38, R-40, R-41, R-42) to fail on `kelly_regime_v4`'s SIZE axis, and the second (after R-41) to fail while attacking INFO with genuinely non-price data rather than re-deriving from the existing OHLCV series — raising the prior further that the axis, not just the individual signal, is close to exhausted for this strategy family. |

### R-28 pre-registration — written and committed before the holdout was read

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
finding. *(0.32x was measured on the 60-day variant, which was the
default when this was written; the frozen 20-day config holds 0.27x. The
prediction stands as recorded — this note is the correction, not an
edit.)*

**Secondary question, also fixed in advance** (not part of the promotion
bar): does the evidence gate cut out-of-sample drawdown relative to
`kelly_regime_v4`? That is the scientific result either way.

### R-28 results — the decision rule did not move

**Configurations evaluated in step 3: 24** — the 3 variants (E1/E2/E3) ×
3 bet half-lives (20/60/180d) = 9, plus a 15-point one-knob-at-a-time
neighbourhood around the selection. Each was scored on inner-train and
inner-validation across both markets, so 24 distinct configurations cost
96 backtests; **24** is the number that goes into the deflated Sharpe,
since that is how many distinct things were searched over. No holdout
data was read until the decision rule above was committed — `git log`
records the freezing commit one commit ahead of the results.

**Holdout, 2023-01-01 → 2026-08-12, $1,000 start, 0.10% / 0.05% taker:**

| | spot final | spot DD | spot Sharpe | fut 5x final | fut DD | fees paid (spot) |
|---|---|---|---|---|---|---|
| `buy_and_hold` | **$3,839** | 54.0% | 1.03 | $15,176 | 60.3% | $1 |
| `kelly_regime_v4` | $3,373 | 27.8% | 1.22 | $4,901 | 33.0% | $310 |
| **E1 e-process (frozen)** | $1,607 | **11.6%** | 1.01 | $1,776 | **14.3%** | **$50** |
| E2 unified Kelly sizer | $3,390 | 29.0% | 1.11 | $6,661 | 37.3% | $349 |
| E3 both | $1,407 | 21.6% | 0.54 | $1,984 | 31.9% | $45 |

**Decision:**

- **P1 FAIL** — $1,607 against holding's $3,839 on spot. Exactly the
  predicted failure, for the predicted reason.
- **P2** would have passed on its drawdown limb (−42.4pp vs holding) and
  fails on Sharpe (1.01 vs 1.03). Moot: P1 gates it.
- **P3 PASS** — the falsification test *did not* falsify. On ETH the
  drawdown reduction replicates and is larger than the incumbent's: spot
  **19.5%** vs v4's 36.5% vs holding's 94.2%; futures 36.9% vs v4's 35.1%
  (inside the +5pp allowance) where leveraged holding was liquidated. The
  BTC control behaves the same way (14.4% vs 40.1% spot).
- **P4 PASS** — plateau, and specifically a plateau *in the risk axis*.
  Across 15 neighbours (half-life 10–90d, α 0.01–0.20, clip 3–8, deadband
  0.05–0.20, with and without evidence decay) inner-validation drawdown
  stays in 8–20% on spot and 11–25% on futures. The one knob that breaks
  it is `evidence_cap_mult=2`, which lets stale evidence persist: DD jumps
  to 49% and the balance falls 22%. Returns across the same neighbourhood
  are *not* a plateau — they scatter inside the noise floor, which is why
  the selection was made on risk.

**Verdict: NEGATIVE.** Default reject; P1 fails; nothing was re-argued
after the fact.

**Lookahead checks, run before any result was believed.** An unregistered
experiment gets none of `test_causality_strict.py`'s protection, because
that suite parametrizes over the *registry*. So the same two-opposite-
tampers procedure was run by hand against this strategy
(`run_eprocess.py causality`): every decision at or before the cut is
unchanged when all later bars are multiplied by 3 in one copy and divided
by 3 in the other. The `target`, `evidence` and `lam` columns were also
compared directly and differ by exactly 0.0 before the cut — the check
that catches the full-series fit a truncation test cannot (a mean, std or
quantile taken over the whole series and applied to early rows). Every
estimator here is `ewm(...).shift(1)`; there is no expanding statistic
that sees its own future. `pytest` passed — 391 tests at the time of this
row (418 after R-29 added the inference suite).

**Path sensitivity (40 random windows, the R-19 design, identical windows
across strategies):**

| | median return | median DD | worst DD | P(DD>50%) | beat hold |
|---|---|---|---|---|---|
| `buy_and_hold` spot | +49.3% | 52.7% | 84.1% | 57% | — |
| `kelly_regime_v4` spot | +82.1% | 23.7% | 43.0% | 0% | 48% |
| **E1 e-process spot** | +36.5% | **9.9%** | **17.0%** | 0% | 40% |
| `buy_and_hold` fut | −98.2% | 98.6% | 99.9% | 100% | — |
| `kelly_regime_v4` fut | +116.3% | 23.6% | 34.8% | 0% | 65% |
| **E1 e-process fut** | +43.7% | **12.2%** | **25.2%** | 0% | 65% |

Paired per window, the e-process drawdown is deeper than v4's in **0 of
40 windows on both markets** (median −14.0pp spot, −11.3pp futures). This
is the only claim in the project that is not inside a noise floor.

**Costs.** At Bitstamp's 0.40% entry tier the e-process loses 11% of its
final balance where v4 loses 27%, because its fee bill is 6x smaller ($50
vs $310) — but both still lose to holding, consistent with R-13. With
funding charged on 5x futures it pays **$178 against v4's $1,190**, since
it holds a third of the exposure and is flat more often; leveraged holding
is liquidated outright. The COST constraint is attacked as a side effect
of taking less risk, not as a mechanism.

**Deflated Sharpe — computed, at last (partially closes R-25).** Across
this session's 24 trials the inner-validation Sharpe spread is sd 0.223,
so the expected best-of-24 by luck alone is SR* = 0.44; the frozen
config's holdout Sharpe of 1.01 gives a deflated Sharpe of **0.859** —
under the conventional 0.95 bar *on this session's trials alone*, before
any program-level deflation for a holdout read ~30 times. The Sharpe
level does not survive its own trials count. The drawdown result does not
depend on it.

**Lesson — the N≈3 constraint, finally in units.** Bet honestly on the
evidence actually available and you get **0.27x the incumbent's mean
exposure** (0.145 vs 0.531). The reason is measurable: the log-wealth of
the e-process against "drift is zero" accumulates at **+0.79 nats a year
against a noise standard deviation of 3.33 nats a year**, so reaching the
α=0.05 threshold of 3.0 nats on drift alone takes **1,395 days — 3.8
years**. A decade of 5-minute bars contains between two and three such
periods. "Effective sample size ≈ 3" was an estimate in the standing
diagnosis; it is now a measurement, and it is the whole explanation for
why every predictor in section A failed.

The three pre-registered failure modes are all ruled out. **(a)** Not a
smoothed momentum indicator in disguise: the gate correlates 0.54 with
the incumbent's latched vote and the two disagree on 17% of bars.
**(b)** It does not degenerate to buy-and-hold: mean gate 0.145, fully
open on 0.1% of bars, shut on 38.7%. **(c)** Fees did not kill it — its
turnover is a sixth of the incumbent's and it pays $50 against v4's $310.
What killed it is that correct calibration on this data says *hold less*,
and BTC went up.

One aside worth keeping: full Kelly makes the volatility target equal to
the estimated Sharpe ratio, and the median estimate here gives a
half-Kelly target of **0.49** against the 0.55 this repo set by hand. The
hand-tuned constant was very nearly the principled one.

**Holdout counter: ~38** (~30 before, +8 this row: three configurations ×
two markets, plus two cost re-runs). The falsification test on ETH and
the 40-window resample do not touch the 2023+ BTC holdout.

> **Superseded in part by R-31 (08-18) — read the two together.** B-11
> ran the matched-risk comparison this row asked for, and it removes two
> of the claims above. **P3** ("the drawdown reduction replicates on ETH")
> was measured against an arm carrying 2.4x the volatility; at matched
> risk the ETH drawdown ordering **reverses**. And "deeper than v4 in 0 of
> 40 windows" becomes deeper in **45–82%** once the e-process arm carries
> comparable exposure. Nothing in this row was mismeasured — R-31
> reproduces every number here exactly, including 14.4% / 19.5% / 36.9% on
> the falsification data — but the property being measured was the
> exposure level, not the gate.

**Next step → B-11.** The exposure level and the evidence gate are
separable, and this session only measured one point on that trade-off.
The well-posed follow-up is a *matched-risk* comparison: run the
e-process gate and the incumbent's latched vote at the same realized
volatility and ask which delivers more return per unit of drawdown.
Note the warning already in hand: raising exposure through
`evidence_cap_mult` is **not** the way to do it — the drawdown grows
superlinearly because the cap lets stale evidence persist.

### R-29 pre-registration — written and committed before any statistic was read

**Idea.** Every headline in this repo is a point estimate selected from a
search, reported without an interval or a trials adjustment. B-04 has been
on the backlog since the ledger was written and R-25 recorded the gap
exactly: deflated Sharpe, purged CV and bootstrap confidence intervals are
*cited* in `RESEARCH.md` and computed nowhere. Build the three, apply them
to the comparison table, and let the answer be whatever it is.

**Constraint attacked.** ERR — but in the *reporting* path rather than the
signal path. R-12 is the reason this matters: 28 of 32 configurations beat
holding in-sample and 0 of 28 out-of-sample, which is what selection
without error control produces. Also N≈3: an interval is the honest way to
say a decade of bars is three regime observations.

**Not a duplicate of.** R-20 measured the ±0.2 Sharpe noise floor for one
pair of strategies and never applied it to the table. R-19 resamples
*paths* (40 random windows) for five strategies, which answers a different
question from an interval on a statistic. R-28 computed one deflated
Sharpe, for one strategy, against one session's 24 trials, on bar-level
observations. R-25 is the row that says none of this was done.

**Simulable here?** Yes — pure computation on committed data, no fetch.

**Method, fixed in advance.** Stationary bootstrap (Politis & Romano 1994),
30-day mean block, 2,000 resamples, on **daily** returns — a million
autocorrelated 5m bars is not a million observations, and 30 days is the
block length that measured the noise floor in R-20. Comparisons are
**paired**: the same resample indices are applied to both strategies, so a
draw that happens to contain the 2022 bear contains it for both. Deflated
Sharpe uses the project's own trials count (a floor of **103**, counted
from the ledger in `scripts/inference.py`), not one session's.

**Pre-registered self-test — if any of these four fails, the round is void
and nothing is written into `VALIDATION.md`:**

1. a strategy paired against *itself* must return an interval containing
   zero;
2. `kelly_regime_v4` against `macd_cross` ($66.8K against $4.99) must
   return an interval excluding zero — a test that cannot see that gap
   cannot see anything;
3. the deflated Sharpe must **refuse** to certify the best of 50 pure-noise
   trials (DSR < 0.95) while the undeflated PSR certifies it;
4. every CPCV split must have disjoint train and test sets with the purge
   and embargo actually removing the neighbourhood of the test fold.

**Pre-registered decision rules.** These are interpretation rules, not
promotion rules — nothing is being promoted. They are written down first
because the temptation in a measurement round is to decide afterwards
which numbers count.

- **C1 (the table's ordering).** Report the fraction of *adjacent* pairs in
  the ranking whose 95% paired interval excludes zero. If it is **below
  50%**, the README comparison table gets a standing warning that its
  ordering is mostly not statistically distinguishable, in the same voice
  as the fee and funding warnings.
- **C2 (the project's one robust finding).** "Regime-gated sizing cuts
  drawdown" is upheld only if `kelly_regime_v4`'s paired ΔmaxDD against
  `buy_and_hold` has a 95% interval **strictly below zero on spot, on both
  the full period and the holdout**. If either interval straddles zero,
  the finding is downgraded to "not established" in the README, the
  ledger and `VALIDATION.md`.
- **C3 (return claims).** No strategy may be described anywhere in the docs
  as beating buy-and-hold on return unless its deflated Sharpe is ≥ 0.95
  **or** its paired return interval against holding excludes zero.
- **C4 (is the table useful to someone choosing from it?).** The selection
  rule the table embodies — "rank by growth, take the top" — is judged
  useful only if the out-of-fold winner beats `buy_and_hold` in **more
  than 50%** of the CPCV folds on spot.

**Stated predictions before looking.** C2 holds — the drawdown property is
the one thing that has replicated on a second asset (R-17), a second
strategy family (R-28) and 40 resampled windows (R-19). C1 fails, and
badly: most adjacent pairs will be indistinguishable. C3 fails for every
strategy in the table, including the three at the top. C4 is a coin flip,
and the interesting quantity is the *selection shortfall* — how much of the
top-of-table return is hindsight.

**Holdout accounting.** This round re-reads the 2023+ holdout for 25
strategies on 2 markets. No selection is performed on it: every
configuration involved was frozen and registered in an earlier session, and
the decision rules above were committed before any number was read. The
counter still goes up, because the counter measures exposure and not
intent.

### R-29 results — the self-test passed, and then almost nothing else did

**Self-test first**, because the rest is void without it. All four
pre-registered checks pass: a strategy against itself returns exactly
[0.00, 0.00]; `kelly_regime_v4` against `macd_cross` returns +3.74
[+2.37, +5.03] with P=1.00; the deflated Sharpe rejects the best of 50
noise trials (Sharpe 0.85 by luck, DSR **0.637**) where the undeflated PSR
would have certified it; all 45 CPCV splits are disjoint and purged.
Reproduce with `python scripts/inference.py selftest`.

**One correction found while building it, worth more than a footnote.**
The first version computed the holdout by *slicing* the full-period daily
returns. That is wrong for exactly one reason and it is the R-22 reason: on
5x futures `buy_and_hold` is liquidated back in January 2017, so its sliced
holdout is a flat line of zeros, and every strategy was being scored against a **corpse**
— which made the entire futures holdout column look like a landslide win.
The published numbers use a fresh $1,000 account from 2023-01-01 via
`run_period`, as the rest of the repo does. Flipping that one choice moved
`kelly_regime_v4`'s holdout futures ΔSharpe from **+1.33 (interval
excludes zero)** to **−0.04 (indistinguishable)**. A `dead_tail_pct` column
now travels with every row so the failure cannot recur silently.

**C1 — the table's ordering. FAILS, and worse than predicted.**

| period / market | adjacent pairs distinguishable at 95% |
|---|---|
| full / spot | **3** of 24 |
| full / futures | **2** of 24 |
| holdout / spot | **4** of 24 |
| holdout / futures | **1** of 24 |

**10 of 96.** Eight of the ten sit in the losing tail — `universal_kelly`
vs `harsanyi_crowd`, `game_council` vs `minority_oracle`,
`camouflage_flow` vs `game_switch` and their neighbours — and the other
two are boundary steps involving `champions_council` (vs `universal_kelly`
on full/futures, vs `hedge_experts` on the spot holdout). None of the ten
separates two of the table's top eight from each other, and the top eight
is the only part of the table anyone would act on. The ordering that the
README presents as a ranking is, in its decision-relevant region, noise.
Per the pre-registered rule (below 50%), the README gets a standing
warning.

**C2 — the project's one robust finding. FAILS on the letter of the rule,
and this is the result I got wrong.** The prediction written above was that
C2 would hold. Paired ΔmaxDD against `buy_and_hold`, `kelly_regime_v4`:

| period / market | Δ max drawdown vs holding | 95% interval | P(deeper than holding) |
|---|---|---|---|
| full / spot | **−41.1pp** | [−54.8, −18.4] | 0.000 |
| holdout / spot | −27.1pp | [−35.8, **+1.9**] | 0.045 |
| holdout / futures | **−29.3pp** | [−41.0, −5.0] | 0.009 |
| full / futures | −65.1pp | [−70.7, +52.7] | 0.316 |

The rule required the spot interval to exclude zero on **both** the full
period and the holdout. It misses by **1.9 percentage points** on the
holdout. So: the drawdown reduction is established on the full history and
on the futures holdout, and on 3.6 years of spot holdout alone it is not —
one-sided evidence at 95.5%, two-sided at 90%, which is the honest way to
say it. The full-period futures interval is wide and useless for a
different reason: `buy_and_hold` there is a corpse for 99.7% of its days,
so resampling sometimes compares against an account that cannot draw down
because it has nothing left.

Downgraded accordingly wherever it is stated. Note what did *not* change:
R-19's 40-window resample (deeper than v4 in 0 of 40) and R-17's ETH
replication are different tests of the same property and both still stand.
The honest summary is that the drawdown property is the *strongest* claim
in this repo and it is still not a 95% claim on the holdout alone.

**C3 — return claims. Nothing survives out-of-sample.** Deflated Sharpe
against 103 project trials, at the only trial dispersion this project has
measured (0.223, R-28's 24 configurations → SR* = 0.57):

| | Sharpe | PSR>0 | DSR | break-even trial sd | min track record |
|---|---|---|---|---|---|
| `kelly_regime_v4`, full spot | 1.44 | 1.000 | **0.997** | 0.36 | 3.3y |
| `kelly_regime_v4`, holdout spot | 1.21 | 0.991 | **0.896** | 0.15 | **6.2y** |
| `buy_and_hold`, holdout spot | 1.03 | 0.976 | 0.812 | 0.07 | 12.4y |
| `champions_council`, holdout spot | 0.95 | 0.969 | 0.773 | 0.04 | 17.4y |

On the full history the leaders clear the bar; **on the holdout not one
strategy in the table does, and neither does buy-and-hold.** Proving
`kelly_regime_v4`'s holdout Sharpe against a 103-trial search needs **6.2
years** of data like it; the holdout is 3.6. And the whole calculation
hinges on a quantity nobody can pin down after the fact — the *dispersion*
of the trials, not their count. At the table's own dispersion (2.60,
inflated because most of the table was registered as documented negatives
rather than entered as candidates) SR* = 6.60 and every DSR is 0.000. That
is why `breakeven_sd` is reported: v4's full-period claim survives any
search whose Sharpe spread is under 0.36, and dies above it.

The sharpest single number is not a Sharpe at all. The table ranks by
**final balance**, and on the full period `kelly_regime_v4`'s log-growth
advantage over holding on spot is **+0.044 with P(beats holding) = 0.52**.
By its own ranking criterion, the #1 strategy against the benchmark is a
coin flip.

**C4 — is the table useful to someone choosing from it? FAILS.**
Combinatorially purged CV, 10 groups, 2 held out, 45 splits, 100-day purge
and embargo, selecting by growth on the purged training groups:

| | spot | futures |
|---|---|---|
| what the rule picks | `kelly_regime_ev_fast` x22, `buy_and_hold` x19, v4 x2, v3 x2 | `kelly_regime_v4` x41, v3 x4 |
| beats holding out-of-fold | **6 of 45** (13%); 19 are ties where it picked holding itself, so 6 of 26 contested (23%) | 41 of 45 — but holding is liquidated and inert in **36 of them** |
| always-`kelly_regime_v4` instead | beats holding in 44%, median −0.089 log | 91%, median +1.022 log |
| selection shortfall vs hindsight | +0.490 log (the pick gives up 63% of what the best fold strategy made) | +0.066 log |
| train→test rank correlation | median 0.72, range −0.70..0.86 | median 0.40, range −0.79..0.74 |

On spot, re-ranking the table inside each fold and holding the winner loses
to buy-and-hold in most folds — R-12's 28-in-sample/0-out-of-sample result
reproduced one level up, at the level of *strategy selection* rather than
parameter tuning. The futures column cannot answer the question at all,
because the benchmark is dead in four fifths of the folds; restricted to
the nine folds where it is alive the picked strategy wins all nine, which
is a statement about surviving leverage, not about ranking.

**Scoreboard against the predictions written before looking:** C1 predicted
to fail — correct, and by more than expected. C2 predicted to hold —
**wrong**. C3 predicted to fail for everything — half right; it fails
everywhere out-of-sample and clears on the full history at the narrow
dispersion. C4 predicted a coin flip — wrong, it is clearly negative on
spot.

**Configurations evaluated: 0.** This round fit nothing and selected
nothing; it re-measured strategies that were already frozen. The trials
count it *contributes* is zero, and the trials count it *applies* is 103.

**Holdout counter: ~88** (~38 before, +50 this row: 25 strategies × 2
markets, each a fresh 2023+ account). That is by far the largest single
increment in the project, and it is the last one that should ever be spent
this way — every strategy in the table has now been measured against the
holdout with intervals attached, so there is nothing further to learn from
it by looking again. Read together with C3, the conclusion the routine
anticipated has arrived: **this dataset is exhausted for Sharpe-based
claims**, and only forward paper trading (B-06) can add evidence.

**Next step → B-12.** The intervals are computed but not *displayed*: the
README table still reports points. Wiring `reports/inference/bootstrap.csv`
into `tradebot run` so the comparison table carries its own error bars is
the change that makes this permanent rather than a one-session document.

### R-30 — the display round, and the statistic that was thrown away

**Idea, in one sentence.** Put R-29's intervals inside the comparison
table, so a reader who never opens `VALIDATION.md` still sees that most
of the ranking is not distinguishable from doing nothing.

**Constraint attacked.** ERR, in the *reporting* path — the same place
R-29 attacked it. A document that says "the ordering is mostly noise"
sitting next to a table that prints bare point estimates in rank order
loses that argument to the table every time, because the table is what
gets read.

**Not a duplicate of.** R-29 computed the numbers and named this as its
next step; R-25/B-04 is the row that says they were never computed. This
round fits nothing and searches nothing.

**Pre-registered failure mode**, named before any code: that the wiring
prints a *stale or mismatched* interval next to a live number — the same
class of error as R-29's corpse bug, arriving through the plumbing rather
than the statistics. Three guards were designed against it and each has a
test: the market alias map is exact and one-way, so an interval measured
on 5x futures can never be printed beside a run at another leverage (the
cell blanks instead); a registered strategy with no interval fails CI, on
both markets and both periods; and the benchmark's own dead-tail share
travels with every row, so a comparison against a liquidated
`buy_and_hold` is flagged rather than scored.

**What it changed in the table.** Two columns, placed *after* the observed
numbers because the divide is real — everything left of them happened on
one path, and only they say whether it is distinguishable from having
done nothing. Both are pinned to **spot**, whichever market a row's
balance is bolded in, because that is where this project states its
promotion bar and because leveraged buy-and-hold is a stress case rather
than a benchmark.

**The result, which was not the point of the round and is the most useful
thing in it.** R-29 computed the paired log-growth comparison and saved
only its point estimate and p-value, discarding the interval. Recovering
it:

| | Δ log growth vs holding | 95% CI | P(beats holding) |
|---|---|---|---|
| `kelly_regime_v4`, full / spot | +0.044 | [−2.60, +2.85] | 0.52 |
| `kelly_regime_ev_fast`, full / spot | +0.107 | [−3.08, +3.29] | 0.53 |
| `kelly_regime_v4`, holdout / spot | −0.129 | [−0.94, +0.74] | 0.37 |

"P = 0.52" reads like a near-miss. **[−2.60, +2.85]** reads like what it
is: a decade of 5-minute bars cannot distinguish the table's #1 from
buy-and-hold anywhere between a thirteenth of holding's final balance and
seventeen times it. Across the whole table on spot over the full history,
**0 of 24 are distinguishably better on growth**, 13 are distinguishably
worse, and the 11 that are indistinguishable are exactly the profitable
block. The drawdown column gives 13 of 24 distinguishably shallower — the
two columns disagree, and that disagreement is the finding: v4's ΔSharpe
of +0.47 [+0.07, +0.87] excludes zero while its Δgrowth does not, because
Sharpe rewards the volatility the strategy removes and final balance does
not.

**Reproduction check.** The curve caches were rebuilt from scratch (100
backtests) and every published R-29 number came back identical: self-test
+3.74 [+2.37, +5.03] and noise DSR 0.637; adjacent pairs 3 / 2 / 4 / 1 =
10 of 96; v4 full/spot ΔSharpe +0.47 [+0.07, +0.87] and ΔmaxDD −41.1
[−54.8, −18.4]; holdout ΔmaxDD −27.1 [−35.8, +1.9]. For a pipeline whose
entire job is to be trusted, that is worth stating.

**Configurations evaluated: 0.** Nothing was fitted, tuned or selected.
The trials count this round contributes is zero; the count it applies is
still 103.

**Holdout counter: ~88, unchanged** — and this is a judgement call worth
recording rather than burying. The bootstrap was re-run over the holdout
to obtain the growth interval, which looks like 50 fresh consultations.
It is not: R-29 already drew those exact resamples and already computed
that exact interval object, then persisted two of its three fields. The
same seeds produced bit-identical numbers on every overlapping quantity,
which is the evidence for the claim. No new question was asked of the
holdout; a field was recovered from an answer it had already given. A
reader who disagrees should read the counter as ~138 — the conclusion is
the same either way, because R-29 already established that no
Sharpe-based claim from this dataset is supportable.

**Cost this imposes on future sessions.** Adding a strategy now requires
`python scripts/inference.py` before `tradebot run`, or CI fails. That is
deliberate: a new row entering the table as a bare point estimate beside
rows carrying error bars would read as the *stronger* number, which is
the reverse of the truth. `ROUTINE.md` records it as the third
CI-enforced registration rule.

**Next step → B-11 or B-05.** B-12 closes here. The display cannot
generate evidence, and the ranked backlog below is unchanged by it except
that the two remaining computation-only items are now the top of it.

### R-31 pre-registration — written and committed before the holdout was read

**Idea, in one sentence.** R-28 compared an e-process evidence gate with
the incumbent's latched anchor vote at two *different* exposure levels and
concluded "better risk, worse return"; hold the sizer, the deadband, the
warmup and the exposure fixed, vary only which quantity opens the gate,
and ask which gate delivers more return at the same realized volatility.

**Constraint attacked.** ERR and SIZE. ERR because the question is whether
anytime-valid error control in the signal path buys anything once its
known side effect — holding less — has been removed by construction. SIZE
because the exposure axis is exactly the "how much" question this
project's only profitable strategies answer.

**Which ledger rows it is not a duplicate of.** R-28 is the direct parent
and named this as its next step (B-11): it measured *one* point on an
exposure/evidence trade-off and compared it against the incumbent at a
different point, so its headline is partly a tautology.
L-04 / L-01 / L-02 / L-03 vary the vote and never touch the gate
mechanism. R-11 and the leverage frontier in `VALIDATION.md` vary exposure
with the gate held fixed — the mirror image of this round. Nothing here
re-tries R-03 (BOCPD) or R-01 (HMM).

**Simulable here?** Yes. One price series, causal, no new data, no fetch.

**What would make it fail — named before any code ran.** (a) The two gates
are the same object at different smoothings, so at matched risk their
returns coincide inside the ±0.2 Sharpe noise floor and the round adds an
interval rather than a finding. (b) Matching on volatility does not match
on drawdown, so an apparent risk win survives only on the axis that was
not matched. (c) The e-process arm needs several times the incumbent's
exposure to reach its volatility, and on spot the 1.0-notional cap binds,
so "same sizer" quietly stops being true at the top of the frontier.

**Method, fixed in advance.** One `GatedKelly` class
(`experiments/matched_risk.py`) with an interchangeable gate. The exposure
knob `k` multiplies `target_vol`, `max_leverage` and `deadband` together;
because `min(k·tv/vol, k·ml) == k·min(tv/vol, ml)` exactly, that rescales
the position and changes nothing about its timing. `evidence_cap_mult`
stays at **1.0** — R-28's explicit warning was that raising exposure
through the cap keeps stale evidence alive and grows drawdown
superlinearly. Exposures are solved on **inner-validation only**, to
within 2% of the target realized volatility, in both directions. The
holdout statistic is the R-29/R-30 paired stationary block bootstrap:
30-day mean block, 2,000 resamples, daily returns, identical resample
indices for both arms of a pair.

**Frozen configuration.** Gate ∈ {`vote` (20/40/80-day latched anchors,
1% band), `evidence` (bet half-life 20d, α=0.05, clip 5, cap 1.0)}; sizer
`plain` (`min(target_vol/vol, max_leverage)`, the `kelly_regime` sizer and
the one R-28's E1 used, so the numbers stay comparable); `target_vol=0.55`,
`max_leverage=2.0`, `deadband=0.10`, `vol_span=8d`. The exposures solved
on inner-validation, and frozen here:

| market | direction | matched to vol | vote k | evidence k |
|---|---|---|---|---|
| spot | match-up | 0.325 | 1.000 | **4.696** |
| spot | match-down | 0.087 | **0.262** | 1.000 |
| futures 5x | match-up | 0.322 | 1.000 | **3.725** |
| futures 5x | match-down | 0.136 | **0.372** | 1.000 |

The `conditional` sizer (`kelly_regime_v3`'s extreme-only targeting) was
swept on both inner splits and its matched exposures solved, but it is
**deliberately not carried to the holdout**: at ~88 prior consultations
the cheapest thing this project can do with the holdout is ask it fewer
questions. Recording that here so it is a pre-registered economy and not
a post-hoc choice of which arm to report.

**V — validity gate, checked before any decision rule is read.** A cell is
a matched comparison only if, *on the holdout*, the two arms' realized
volatilities are within **20% of each other in relative terms** and the
notional-cap clamp fraction is **below 1% for both arms**. Exposure is
frozen on inner-validation; if the risk match does not survive into 2023+
then the cell is not matched, and above the cap it is the market rather
than the gate that is setting the position. Failing cells are reported and
**voided**, not scored.

**Pre-registered decision rules.**

- **D1 (the B-11 question).** The e-process gate is better at matched risk
  only if the paired Δ log growth (evidence − vote) interval **excludes
  zero in the same direction in every valid cell** — both markets, both
  matching directions. The same rule with the sign flipped declares the
  vote better. Anything else is **not established**, which is the default.
- **D2 (the axis that was not matched).** The same rule on Δ max drawdown,
  reported whatever D1 says.
- **D3 (falsification, chosen now).** ETH on Bitfinex over the R-17
  window, exposures re-matched on ETH's own volatility. The **ordering**
  must replicate: the gate that wins on the BTC holdout must win on ETH in
  both directions. If it flips on a second asset, any D1 claim is dead.
- **P (promotion).** Nothing is promoted from D1 alone. A gate that wins
  D1 and D3 still faces the full ROUTINE bar before registration: beat
  `buy_and_hold` on the spot holdout after real costs, exceed the ±0.2
  Sharpe noise floor or cut drawdown by ≥10pp, and sit on a plateau in k.

**Stated predictions before looking.**

1. **D1 fails**, and the reason is already visible in the inner splits. At
   matched volatility on spot the vote wins inner-train 2017–2020 ($15.4K
   against roughly $10.2K at vol 0.379) and the e-process wins
   inner-validation 2021–2022 (about $1,265 against $909 at vol 0.325).
   Same asset, same sizer, same risk — what differs is *when* each arm is
   on, so the ordering should track the regime. 2023+ is a bull, so I
   predict the **vote wins the holdout**, and D1 therefore returns "not
   established": a rule that reverses with the regime is not a finding
   about gates.
2. **D2**: the e-process arm keeps a drawdown advantage on at least one
   market even at matched volatility, because it concentrates exposure
   into fewer, longer holds rather than spreading it.
3. **V is the condition most at risk.** The exposure that matched risk in
   2021–22 need not match it in 2023+, and if it does not, this round's
   answer is that the question cannot be asked of this dataset without
   re-matching on the holdout — which would be selection on the holdout.

**Configurations evaluated in step 3: 36** — the frontier grid, 2 sizers ×
2 gates × 9 exposures, each scored on inner-train and inner-validation on
both markets (144 backtests). A further **32 inner-validation backtests**
were spent by the exposure solver; they select `k`, but on a criterion
(equalize realized volatility) that is orthogonal to performance, so they
are recorded separately rather than folded into the trials count. No
holdout data has been read at the time of this commit.

### R-31 results — the decision rule did not move, and three cells of four were void

**Reproduction first, because none of the rest transfers otherwise.** The
new `GatedKelly` evidence arm at k=1 is not merely similar to R-28's E1,
it is the same run: max |equity difference| over inner-validation is
**0.000e+00** (`run_matched_risk.py parity`). On the falsification data it
reproduces R-28's published drawdowns exactly through an independent code
path — BTC control spot **14.4%**, ETH spot **19.5%**, ETH futures
**36.9%**. So every difference in conclusion below comes from the
matching, not from a reimplementation.

The by-hand lookahead probe passes for all four gate × sizer
combinations: orders identical under two opposite tampers of the future,
and max |column difference| before the cut = **0.000e+00** on `target`,
`conf` and `scale`. `pytest`: **436 passed**.

**What matching costs, and the first surprise.** Equalizing realized
volatility on spot needs the e-process arm at **k = 4.70** against the
vote's k = 1 — but only on inner-validation. On inner-train the same
match needs **k = 2.16**. The exposure ratio that equalizes risk is
itself regime-dependent, roughly 2.2x in the 2017–2020 bull and 4.7x in
the 2021–2022 top-and-bear, because the e-process shuts down hardest
exactly when the vote does not.

**The inner splits disagree with each other, at matched risk** (spot,
plain sizer, exposures matched *within* each split):

| split | matched vol | vote | e-process | Δ DD |
|---|---|---|---|---|
| inner-train 2017–2020 | 0.379 | **$15,381** (DD 39.3%) | $10,029 at k=2.16 (DD 31.9%) | −7.4pp |
| inner-validation 2021–2022 | 0.325 | $909 (DD 38.1%) | **$1,233** at k=4.70 (DD 32.8%) | −5.3pp |

Same asset, same sizer, same risk; the return ordering reverses with the
regime and the drawdown ordering does not. That was the basis of the
prediction on record, and it is the shape of the whole result.

**V — the validity gate, applied before any decision rule was read.
Three of four cells VOID.**

| cell | vote vol | e-process vol | gap | max clamp | verdict |
|---|---|---|---|---|---|
| spot / match-up | 0.315 | 0.306 | 2.6% | **41.0%** | **VOID** (cap, not gate, sets the position) |
| spot / match-down | 0.104 | 0.140 | **29.9%** | 1.3% | **VOID** (risk match did not survive) |
| futures / match-up | 0.394 | 0.527 | **29.0%** | 0.0% | **VOID** (risk match did not survive) |
| futures / match-down | 0.153 | 0.153 | 0.2% | 0.0% | **VALID** |

Prediction 3 was the one that landed: the exposure frozen on 2021–22 does
not equalize risk in 2023+. On futures the e-process arm overshot to 0.527
against the vote's 0.394 — it was *more* volatile than the arm it was
supposed to match — and on spot it undershot in the other direction.

**And the cell that would have been the headline is void.** On spot /
match-up the two arms landed within 2.6% of each other on realized
volatility, and there the e-process delivered **$3,736 against the vote's
$3,277 with a 4.5pp shallower drawdown and a third of the fills** — the
only cell where an e-process gate beats the incumbent's own gate on both
axes, and it beats `kelly_regime_v4` ($3,373) too. It is void because
both arms spend 41% and 27% of their bars pinned at spot's 1.0-notional
cap, so they are not running the same sizer. That is the pre-registered
rule doing exactly the job it was written for: this is the number a
round without a validity gate would have reported.

**D1 — the B-11 question. NOT ESTABLISHED, and not marginally.** Paired
stationary block bootstrap on the 2023+ holdout, 1,319 daily
observations, identical resamples for both arms:

| cell | Δ log growth (evidence − vote) | 95% CI | P(>0) |
|---|---|---|---|
| spot / match-up *(void)* | +0.131 | [−0.400, +0.691] | 0.68 |
| spot / match-down *(void)* | +0.030 | [−0.300, +0.463] | 0.54 |
| futures / match-up *(void)* | −0.125 | [−1.307, +1.441] | 0.40 |
| **futures / match-down (valid)** | **−0.072** | **[−0.532, +0.379]** | 0.39 |

Every interval contains zero, in every cell, valid or void — and the sign
is not even stable across cells. The one cell that survives V gives the
vote a $1,909-to-$1,776 edge that the interval cannot distinguish from
nothing.

**D2 — the axis that was not matched. Also NOT ESTABLISHED.** Δ max
drawdown, same resamples: spot −4.7pp [−19.9, +7.1] and +1.7pp
[−6.3, +9.6]; futures +7.3pp [−14.2, +27.0] and −1.9pp [−13.4, +5.4].
Four intervals, four containing zero. Prediction 2 said the e-process
would keep a drawdown advantage at matched volatility on at least one
market; the point estimates lean that way on three of four cells and not
one of them clears its own error bar.

**D3 — the falsification test, and the result that matters most.** On
ETH (Bitfinex, the R-17 window), with exposures re-matched on ETH's own
volatility, the e-process gate loses **all four** cells on return and
loses on drawdown too:

| asset / market | matched vol | vote | e-process | Δ DD |
|---|---|---|---|---|
| ETH / spot, match-up | 0.377 | **$5,186** (DD 36.3%) | $4,010 at k=2.31 (DD **40.0%**) | +3.7pp |
| ETH / spot, match-down | 0.171 | **$2,379** (DD 17.1%) | $1,944 (DD **19.5%**) | +2.4pp |
| ETH / futures, match-up | 0.428 | **$7,330** (DD 36.1%) | $3,565 at k=2.17 (DD **53.2%**) | +17.1pp |
| ETH / futures, match-down | 0.232 | **$2,345** (DD 27.6%) | $2,079 (DD **36.9%**) | +9.3pp |

The BTC control over the same window behaves as it did on BTC everywhere
else — the vote wins return in all four cells, the e-process wins
drawdown by 5–10pp.

**So R-28's P3 was a risk-level artifact.** R-28 recorded "the
falsification test *did not* falsify — on ETH the drawdown reduction
replicates and is larger than the incumbent's: spot 19.5% vs v4's 36.5%".
Both numbers reproduce here exactly. But the 19.5% was measured against
an arm carrying **2.4x** the volatility; hold the risk fixed and ETH's
drawdown ordering **reverses**, by 2.4pp on spot and 17.1pp on futures.
The e-process gate's risk advantage on a second asset was the exposure
level, not the gate.

**Costs.** At Bitstamp's 0.40% entry tier the e-process arm degrades far
less than the vote — spot match-up $3,399 against $2,373, because it pays
$381 in fees against $977 on 91 fills against 349 — which is L-06's
turnover finding reappearing, not a gate finding, and the cell is void
anyway. Neither arm beats `buy_and_hold`'s $3,827 at that tier. With
funding charged on 5x futures, leveraged holding is **liquidated**; the
matched pair pays $1,283 (vote) against $874 (e-process) at match-up and
$264 against $178 at match-down, and finishes at $3,120/$2,787 and
$1,597/$1,614.

**Path sensitivity — and R-28's single strongest number, inverted.** The
R-19 design, 40 random windows, identical windows for every strategy,
carrying the frozen spot exposures. Paired per window, e-process minus
vote:

| market / pair | Δ median return | e-process return higher in | Δ median DD | e-process **deeper** in |
|---|---|---|---|---|
| spot / match-up | +6.7pp | 55% | −1.9pp | 45% |
| spot / match-down | +7.8pp | 72% | +2.0pp | 72% |
| futures / match-up | +70.8pp | 72% | +14.9pp | 78% |
| futures / match-down | +8.6pp | 75% | +4.9pp | 82% |

R-28's most-quoted line was that the e-process drawdown is deeper than
`kelly_regime_v4`'s in **0 of 40 windows on both markets** — "the only
claim in the project that is not inside a noise floor". Give the
e-process arm a comparable exposure and it is deeper in **45–82%** of the
same windows. Read this as reinforcing V rather than as a fifth cell: the
exposures were frozen on 2021–22 spot and are not re-matched window by
window, so the arms are only approximately equal-risk here, and on
futures match-up the e-process arm is plainly running hot (median return
+226.6% against +106.4%, median DD 41.0% against 26.0%). That is the
point. The 0-of-40 result was never a statement about the gate; it was a
statement about carrying a third of the notional, and it does not survive
being given the notional back.

**P — promotion. Not reached.** D1 produced no claim, so nothing is a
candidate. For the record the bar would have failed anyway: the best
holdout spot number from any matched arm is $3,736 against
`buy_and_hold`'s $3,839.

**Scoreboard against the predictions written before looking.**
Prediction 1 (D1 fails; the ordering tracks the regime) — **correct**,
including the mechanism, though the holdout's one valid cell went to the
vote by an amount that is itself noise. Prediction 2 (the e-process keeps
a drawdown edge at matched volatility) — **wrong** as a claim: three of
four point estimates lean that way, none survives its interval, and ETH
reverses the sign outright. Prediction 3 (V is the condition most at
risk) — **correct**, and it voided three cells of four.

**Verdict: NEGATIVE**, with the decision rule untouched. Nothing was
re-argued after the fact; the void cells are named as void including the
one that flattered the idea.

**Lesson.** R-28's headline was "the e-process gate is the deepest
drawdown cut in the project, and it loses on return". Hold risk fixed and
**both halves dissolve**: the return gap is inside the noise floor on BTC
and goes the wrong way on ETH, and the drawdown cut — the finding this
project called its strongest — does not survive risk-matching on a second
asset. What R-28 actually measured was the consequence of holding 0.27x
the exposure. That is a real and useful fact about calibrated evidence
(bet honestly, hold less, draw down less), but it is a fact about the
*exposure level*, not about the gate, and B-11 existed precisely to tell
those two apart. It now has.

The second lesson is procedural and cost nothing to obtain: **the
validity gate earned its place in the pre-registration.** Without it this
row would have led with spot / match-up — an e-process arm beating the
incumbent's gate on return, drawdown and fees simultaneously — and the
reason that number exists is that spot's notional cap was truncating both
arms differently, not that the gate was better.

**Next step → B-13, and it is pointed at this project's own headline.**
The argument that retired R-28's risk finding applies verbatim to L-04's:
"regime-gated sizing cuts drawdown" is a comparison between a strategy
holding roughly half the notional and a **fully-invested** benchmark.
R-29's −41.1pp [−54.8, −18.4] and R-17's ETH replication are both
measured that way. Nobody has yet asked what a `buy_and_hold` de-levered
to `kelly_regime_v4`'s realized volatility does to that gap. The harness
built here answers it in an afternoon, and the answer is worth having in
either direction: if the gap survives, it is the best-supported claim in
the repo; if it does not, the project's one robust finding is the same
arithmetic R-28 fell for.

**Configurations evaluated: 36** (frontier grid: 2 sizers × 2 gates × 9
exposures, 144 backtests across two splits and two markets), plus 32
inner-validation backtests spent by the exposure solver on a criterion
orthogonal to performance and 4 more for the within-split matched pairs
above. The project trials count this round contributes is **36**; the
count it applies is 103 + 36 = **139**.

**Holdout counter: ~112** (~88 before, +24 this row: 12 matched-and-
reference runs across two markets, 6 re-runs at the 0.40% taker tier, and
6 with funding charged on futures). The ETH/BTC falsification cells and
the 40-window resample do not read the 2023+ BTC holdout, following the
R-19/R-28 convention. Consistent with R-29: nothing here is offered as a
Sharpe-based claim, and every number above is reported with its interval
or explicitly as a point on one path.

### R-32 pre-registration — frozen on a parallel branch, before either branch's holdout was read

**Two sessions ran B-11 on the same day, from the same base commit
(`42729da`), without knowing about each other.** R-31 above is the primary
record: it landed first, and its exposure solver and validity gate are the
better instrument. This row is the other branch. It is kept because the
routine's rule for parallel work is that **every branch reports, including
the dead ones** — reporting only the branch that worked is selection
performed by the operator — and because this branch ran an arm R-31 did
not: **a third gate that is no gate at all**.

The text of this section is as it was frozen on that branch (commit
`b815e1b`, one commit ahead of its own results, per step 4), with the row
renumbered from R-31 to R-32 on merge and this paragraph added. Neither
branch's numbers were seen by the other before both were frozen; the
merge happened afterwards, and what it changed is recorded in the results
section rather than edited into the pre-registration.

**Idea.** Hold everything except the regime gate fixed — one
inverse-volatility sizer, one deadband, one exposure cap, one fee model —
and move each gate along its *own* risk axis with a scalar multiplier on
the position, then compare the frontiers at matched realized risk. Three
gates: **`none`** (no gate, pure inverse-volatility targeting — never
measured in this project before), **`vote`** (the incumbent's latched
20/40/80-day anchors), **`evidence`** (R-28's e-process).

**Constraint attacked.** ERR and SIZE.

**Method.**

    target_t = min(multiplier * gate_t * min(target_vol / vol_t, max_lev),
                   exposure_cap)

with everything but the gate pinned to the incumbent's shipped values
(`target_vol=0.55`, `max_lev=2.0`, `deadband=0.10`, `exposure_cap=3.0`).
Exposure is raised by the **multiplier only**, never by
`evidence_cap_mult` — R-28's warning, respected by both branches
independently.

**Configurations evaluated in step 3: 33** — 3 gates × 11 multipliers
(0.25 … 16), each scored on inner-train and inner-validation on both
markets (132 backtests).

**Frozen configuration**, selected on inner-validation only: per market,
the multiplier at which each gate's realized volatility equals the vote
arm's at its own scale (spot 0.327, futures 0.322), read off the sweep by
linear interpolation — `none` ×0.61 / ×0.48, `vote` ×1.00 / ×1.00,
`evidence` ×7.83 / ×3.25.

**Pre-registered failure modes.** (a) the frontiers coincide inside the
±0.2 Sharpe noise floor; (b) matching is impossible because the e-process
gate's realized volatility saturates; (c) the fixed absolute deadband
bites the low-exposure arm harder and the comparison measures turnover
policy rather than gate quality.

**Decision rule.** **Q1**: the paired difference between the evidence and
vote arms in log growth and max drawdown on the holdout, 95% stationary
block bootstrap, 30-day mean block, 2,000 resamples, identical indices —
established only if the interval excludes zero. **Q2**: the same test for
each gate against the ungated arm. **Promotion bar** (default reject):
**P1** holdout spot final balance beats `buy_and_hold`; **P2** > +0.2
Sharpe or ≥ 10pp of drawdown; **P3** *(falsification)* the ordering of the
three gates replicates on ETH (Bitfinex, the R-17 window, BTC control);
**P4** the multiplier neighbourhood is a plateau.

**Predictions.** (i) Q1's interval contains zero. (ii) Q2 separates both
gates from no gate. (iii) P1 fails — the holdout is a bull and every gated
arm averages under full exposure. (iv) R-28's "better risk, worse return"
was exposure, not mechanism.

### R-32 results — what the gate is worth, and R-31 replicated by an independent hand

**Agreement with R-31 first, since that is most of this row's value.** Two
implementations written without sight of each other, matching risk by
different methods (R-31 solves for a target volatility; this branch
interpolates a swept frontier), reach the same four conclusions: the two
gates are **not distinguishable** at matched risk on the holdout (here
Δ log growth −0.12 [−0.72, +0.50] spot, −0.26 [−1.25, +0.81] futures, all
intervals containing zero on both axes); R-28's **0-of-40-windows**
drawdown advantage **inverts** (here deeper in 60% of windows on spot and
62% on futures, against R-31's 45–82%); the e-process gate's **fee**
advantage inverts with it ($743 against the vote's $295 on the spot
holdout, $1,956 against $977 at the 0.40% tier); and **P1 fails** ($2,911
against holding's $3,839). One incidental cross-check: this branch's vote
arm pins at spot's notional cap on **41.0%** of holdout bars — the same
figure R-31 reports for its own vote arm, from different code.

A validity check this branch ran and R-31 did not need: the `vote` arm is
a reconstruction of the incumbent's gate on a plain sizer, and it lands on
top of `kelly_regime_v4` — Δ log growth **−0.03 [−0.11, +0.05]**, Δ max
drawdown −0.81pp.

**R-31's validity gate, applied to this branch's cells — and it voids
both holdout cells.** Honesty requires running the better instrument's
rule against these numbers rather than only against its own:

| cell | vote vol | other arms | max clamp | verdict under R-31's rule |
|---|---|---|---|---|
| spot | 0.32 | evidence 0.32, none 0.34 | **41.0% / 35.8% / 21.1%** | **VOID** — spot's 1.0-notional cap sets the position on a third of bars, differently for each arm |
| futures | 0.41 | evidence 0.45, none **0.29** | 0.0% | **VOID** for the ungated comparison — a **29%** volatility gap between `none` and `vote`; the evidence/vote pair matches to 9% |

So this branch's holdout table is a weaker instrument than R-31's, and the
agreement above should be read as agreement *in direction*, resting on the
inner splits and the window resample rather than on the single holdout
path. Both branches reach the same place; only R-31's route is clean.

**The arm R-31 did not run: no gate at all.** This is what the round adds
to the record. On the inner splits, where matching is done *within* each
split by interpolation rather than frozen across one, the ungated
inverse-volatility arm sits **below both gated arms at every overlapping
risk level, in all four cells**. The futures cells are the clean ones —
the 5x notional cap never binds there (peak target 3.0 against a cap of
5.0), so nothing is truncated:

| matched realized vol | `none` | `vote` | `evidence` |
|---|---|---|---|
| inner-train futures, 0.30 | 1.94 | **2.52** | 2.19 |
| inner-train futures, 0.95 | 3.35 | **5.79** | 5.34 |
| inner-validation futures, 0.21 | −0.12 | **+0.10** | −0.06 |
| inner-validation futures, 0.42 | −0.42 | −0.08 | **−0.02** |

(log growth; spot agrees but carries the cap caveat). Across 40 identical
random windows, paired, carrying the frozen exposures:

| paired difference | Δ max DD | deeper in | Δ return | higher in |
|---|---|---|---|---|
| `vote` − `none`, spot | **−6.2pp** | 12% | **+20.0pp** | 80% |
| `evidence` − `none`, spot | +0.4pp | 52% | +19.9pp | 65% |
| `vote` − `none`, futures | +2.1pp | 60% | **+43.2pp** | 90% |
| `evidence` − `none`, futures | +6.3pp | 62% | +75.3pp | 82% |
| `evidence` − `vote`, spot | +4.8pp | 60% | −7.0pp | 38% |
| `evidence` − `vote`, futures | +2.2pp | 62% | +6.3pp | 60% |

At the same risk, gating on the latched vote returns a median **+20.0pp**
per window more than not gating on spot while drawing down **6.2pp** less
— better on both axes in 80% and 88% of windows — and **+43.2pp** in 90%
of them on futures. The exposures here are frozen rather than re-matched
per window, the same caveat R-31 attaches to its own window table, so this
is evidence about direction and magnitude, not a certified interval. The
holdout intervals for the same comparisons contain zero
(`vote` − `none` spot: Δ growth +0.30 [−0.33, +0.93], Δ max DD −20.61
[−28.78, +6.71]) — and that cell is void anyway.

**The ordering of all three gates, and the falsification test.** Ranking
by log growth per unit of realized volatility, `vote` > `evidence` >
`none` in all four ETH/BTC cells (ETH spot 4.73 / 3.27 / 1.77; ETH futures
5.16 / 3.73 / 2.25; the BTC control the same order), so P3 does not
falsify the ordering. What it does falsify is the *transfer of the risk
match*: on ETH the evidence arm realizes **0.64** volatility where it was
matched to 0.32, and 69.3% drawdown against the vote arm's 36.3%. The
multiplier that equalizes this gate's risk is a property of the asset and
period it was fitted on, because the gate's duty cycle tracks the
drift-to-noise ratio it measures. R-31 finds the same instability from the
other side (2.2x in the bull, 4.7x in the bear).

**P4 — not a plateau, and that is the same answer Q1 gives.** On
inner-validation spot the comparison changes sign inside one grid step of
the frozen multiplier (at matched volatility 0.33 the vote leads by 0.09
log; at 0.38 the evidence arm leads by 0.06). Two interleaved frontiers,
not two ordered ones. Assessed on inner-validation rather than the
holdout: P1 already rejects, and spending 8 more consultations on a moot
criterion is not a trade worth making.

**Costs and deflation.** With funding charged on 5x futures the evidence
arm keeps the one advantage that survives matching — **$2,925 at 33.9%
drawdown paying $764** against the vote arm's $3,120 at 38.1% paying
$1,283 — because its exposure is concentrated into fewer hours. This
round's 33 configurations have an inner-validation daily-Sharpe dispersion
of **0.222**, an independent reproduction of R-28's 0.223 from a search
sharing none of its configurations, which matters because R-29's whole
deflation rests on that quantity. Against the day's combined trials count
(below): the vote arm's holdout Sharpe of 1.18 deflates to **0.879**, the
evidence arm's 1.07 to **0.832**. Neither clears 0.95.

**Verdict: NEGATIVE.** Q1 not established, P1 fails, decision rule
untouched. The prediction scoreboard: (i) correct; (ii) **wrong at 95%**
on the holdout — every interval contains zero, and both holdout cells are
void besides — right in direction across 40 windows and all four inner
cells; (iii) correct; (iv) correct, and stronger than predicted.

**Lesson.** *The gate is worth more than the choice of gate.* Both gates
beat no gate by ~20pp of median window return at equal risk and neither
beats the other by anything measurable — the SIZE row of the standing
diagnosis, one level down: a gate is a sizing decision, and the mechanism
that produces it does not appear to matter much. Note what this does
**not** say: the ungated arm here is a volatility-targeted one, not a
de-levered `buy_and_hold`, so it is not an answer to **B-13**. It is a
hint about it, and the hint is unfavourable — on the spot holdout the
ungated arm at 0.72x holding's realized volatility gives up
**−0.46 [−0.93, −0.01]** of log growth for only **−7.3pp
[−12.4, +2.8]** of drawdown, which is the shape B-13 exists to measure
properly.

**Trials and holdout arithmetic for the day, across both branches.** The
routine is explicit that the trials count is the total across parallel
branches, not per branch: R-31's **36** plus this branch's **33** is
**69**, so the count the project applies becomes 103 + 69 = **172** rather
than either branch's own figure. **Holdout counter: ~124** — ~112 after
R-31, +12 here (3 frozen arms × 2 markets, 3 spot fee-tier re-runs, 3
funding-charged futures re-runs). Two sessions spending the holdout on the
same question on the same day is exactly the cost the parallelism section
of ROUTINE.md warns about, and it is worth recording that it happened by
scheduling accident rather than by design.

**Next step → B-13, then B-05**, agreeing with R-31. This branch adds one
reason to prefer that order: the de-levered-benchmark question is the same
arithmetic that dissolved R-28's finding, and the preview above suggests
it will not be kind.

![what a gate is worth at matched risk](../reports/gate_control/frontier.png)

### R-33 pre-registration — written and committed before the holdout was read

**Idea, in one sentence.** Every drawdown claim this project makes
compares `kelly_regime_v4` — which holds a mean notional fraction of
**0.28–0.43** and is flat a third of the time — against a
**fully-invested** `buy_and_hold`; de-lever the benchmark to v4's own
realized volatility and ask how much of the −41.1pp gap is the *gate* and
how much is the *exposure level*.

**Constraint attacked.** ERR and SIZE. ERR because the comparison that
produces this project's headline has never had its most obvious confound
controlled; SIZE because "how much to hold" is the axis every profitable
strategy here operates on, and the question is whether v4 does anything
on that axis a constant cannot.

**Which ledger rows it is not a duplicate of.** R-31 varied the *gate*
with exposure held fixed, and both its arms were active vol-targeted
strategies. R-32 added an ungated arm, and says in its own lesson that
this is **not** an answer to B-13: "the ungated arm here is a
volatility-targeted one, not a de-levered `buy_and_hold`". R-11
(Grossman–Zhou) varies exposure by a drawdown cushion, not to match risk.
R-29 and R-30 measured the −41.1pp and −27.1pp gaps **against the
fully-invested benchmark** — those are the numbers under test here, not a
prior attempt at this test. Nothing here re-tries R-08 (better volatility
forecasting) or R-12 (turnover tuning).

**Simulable here?** Yes. One price series, causal, no new data, no fetch.

**What would make it fail — named before the holdout was read.** (a) V1:
v4 *targets* constant volatility while a constant-exposure hold's
volatility tracks the market's, so the exposure that matches risk is
regime-dependent and the freeze need not survive into 2023+ — the same
instability R-31 and R-32 both hit from the other side. (b) The answer
depends on which reading of "de-levered hold" is used, in which case the
round produces two answers and no finding. (c) On spot v4 asks for more
than 1.0 notional on 2.3–7.4% of bars (measured on the inner splits), so
part of the spot comparison is against the market's cap rather than
against a strategy.

**Method, fixed in advance.** One passive class
(`experiments/matched_hold.py`) holding a constant fraction `c` of
equity, in the two readings the phrase admits:

- **rebalanced** — holds `c`×equity in notional, rebalancing when the
  realized fraction drifts more than 10% *relative* away. Constant risk;
  pays fees. This is the arm B-13 asks for.
- **static** — buys `c`×equity once and never trades again. Zero
  turnover, but the weight drifts up toward 1.0 in a bull, so it is not a
  constant-risk arm. Carried because it is the cheapest benchmark that
  exists and is not obviously the weaker one.

One implementation detail is recorded here because it changes what is
being measured: the rebalanced arm places **quantity** orders rather than
targets. The broker ignores same-sign target adjustments below 5% of
*max* notional so strategies can re-emit a target every bar without
churn; on 5x futures that band is 25% of equity, wider than anything this
arm ever asks for, so routed through `order_notional` it never rebalances
on futures at all and silently becomes the static arm. The first version
of this file did exactly that. A quantity order carries the arm's own
10% band instead of inheriting a leverage-scaled one.

Two matching axes are solved, because "de-levered to match" is ambiguous
and this project has not been careful about which one it means:
**equal realized volatility** (the R-31 convention, solved) and
**equal mean notional** (v4's own mean notional fraction, no solver).
The two disagree by construction, and the disagreement is itself
reportable: on inner-validation spot, matching v4's *notional* (c=0.283)
gives an arm at volatility 0.233 against v4's 0.291, because v4's
exposure is negatively correlated with volatility.

**Frozen configuration**, solved on **inner-validation only**
(2021-01-01 → 2022-12-31) to within 0.1% of v4's realized volatility:

| market | v4 realized vol | v4 mean notional | rebalanced `c` | static `c` | notional-matched `c` |
|---|---|---|---|---|---|
| spot | 0.291 | 0.283 | **0.353** (vol 0.291) | **0.293** (vol 0.293) | 0.283 |
| futures 5x | 0.287 | 0.289 | **0.348** (vol 0.287) | **0.289** (vol 0.289) | 0.289 |

The holdout statistic is the R-29/R-30 paired stationary block bootstrap:
30-day mean block, 2,000 resamples, daily returns, identical resample
indices for both arms of a pair, every difference stated as
**v4 − benchmark** so a negative drawdown difference means v4 draws down
less. `buy_and_hold` is carried through the same pipeline as a
reproduction check against R-29's published −27.1pp [−35.8, +1.9].

**V — validity gate, checked before any decision rule is read.** A cell
is a matched comparison only if, *on the holdout*, (**V1**) the two arms'
realized volatilities are within **20% of each other in relative terms**,
and (**V2**) the passive arm is not pinned at the market's notional cap
(clamp below 1%). Failing cells are reported and **voided**, not scored.

One deliberate difference from R-31's gate, recorded now so it cannot be
read as a convenience later: **v4's own clamp fraction does not void a
cell.** R-31 compared two configurations of one sizer, where truncation
broke the "same sizer" premise. Here the arms are different objects by
design, and v4's clamping on spot is part of what v4 *is* as registered —
every published v4-vs-hold number in this repo (the README table, R-29,
R-30) carries it. Excluding it here would make this round less
comparable to the claim it is testing, not more. It is reported as a
diagnostic on every row.

**Pre-registered decision rules.**

- **D1 (the B-13 question).** "Regime-gated sizing cuts drawdown" is
  upheld only if v4's paired **Δ max drawdown** against the vol-matched
  **rebalanced** hold has a 95% interval **strictly below zero in every
  valid cell**. If any valid cell's interval contains zero, the claim is
  downgraded — in the README, this ledger and `VALIDATION.md` — to
  *established against a fully-invested benchmark only, and not against a
  de-levered one*. That is the same downgrade R-31 applied to R-28, and
  it is written here before the numbers exist.
- **D2 (return).** The same test on **Δ log growth**, reported whatever
  D1 says.
- **D3 (falsification, chosen now).** ETH on Bitfinex over the R-17
  window, exposures re-matched on ETH's own volatility. The **sign** of
  the drawdown gap must replicate: if v4's advantage over a risk-matched
  passive hold flips on a second asset, any D1 claim is dead — exactly
  the test that killed R-28's.
- **The quantity worth having even if every interval contains zero:** the
  **share of the −41.1pp headline that survives matching**. That is a
  number this project should have had since L-04, and it does not depend
  on any of the above clearing a significance bar.
- **P (promotion).** Nothing is promoted here; v4 is already registered.
  A *failed* D1 does not de-register anything either — nothing is
  deleted. What changes is what the docs are allowed to claim.

**Stated predictions before looking.**

1. **V1 fails on at least one market.** v4 holds realized volatility
   roughly constant by construction while the hold's tracks the market's,
   and 2023+ is calmer than 2021–22, so the frozen `c` should undershoot.
2. **D1 fails.** v4's drawdown advantage over a matched rebalanced hold
   is real but small: the inner splits give **−3.7pp** and **−4.2pp**
   (inner-validation, spot and futures) and **−5.6pp** and **−14.4pp**
   (inner-train), against the −41.1pp headline. I predict the holdout
   point estimate lands in −3 to −8pp and at least one valid cell's
   interval contains zero. Stated as the number that matters:
   **I predict 80–90% of the published drawdown gap is carried by the
   exposure level, not by the gate.**
3. **D2 is where the round is favourable to the incumbent, and it is not
   the claim the project makes.** At matched risk on inner-train spot v4
   returns **$18,477 against $6,272** — three times the passive arm at
   the same volatility. I expect v4 to beat the matched hold on growth on
   the holdout too, possibly with an interval excluding zero on futures.
4. **D3 replicates in sign.** Unlike R-28's e-process arm, v4's
   advantage here cannot be an artifact of holding less, because holding
   less is exactly what has been controlled.

**Step 3, for the record** (inner splits only, exposures solved inside
each split, so the ordering can be read regime by regime the way R-31's
was):

| split / market | v4 | vol-matched rebalanced hold | Δ max DD | matched vol |
|---|---|---|---|---|
| inner-train / spot | $18,477 (DD 43.3%) | $6,272 at c=0.426 (DD 48.9%) | **−5.6pp** | 0.399 |
| inner-train / futures | $30,344 (DD 35.3%) | $6,827 at c=0.439 (DD 49.7%) | **−14.4pp** | 0.411 |
| inner-validation / spot | $998 (DD 33.2%) | $960 at c=0.353 (DD 36.9%) | **−3.7pp** | 0.291 |
| inner-validation / futures | $1,064 (DD 32.3%) | $964 at c=0.348 (DD 36.5%) | **−4.2pp** | 0.287 |

against `buy_and_hold` at 84.1% / 99.0% / 77.3% / 99.8% in the same four
cells. The gate's drawdown contribution, before any holdout read, looks
like **4–14 points of a 41–67 point gap**.

**Configurations evaluated in step 3: 18** — 2 passive arms × 9
exposures, each scored on both inner splits and both markets (72
backtests). A further 10 inner-validation backtests were spent by the
exposure solver and 16 on the within-split matched pairs above; they set
`c` on a criterion (equalize realized volatility) orthogonal to
performance, so they are recorded separately rather than folded into the
trials count, following R-31. The project's applied trials count becomes
172 + 18 = **190**.

**Holdout accounting.** No holdout data has been read at the time of this
commit; `git log` records it one commit ahead of the results. The
by-hand lookahead probe (`run_matched_hold.py causality`) passes for both
arms: orders identical under two opposite tampers of the future, max
|column difference| and max |equity difference| before the cut both
**0.000e+00**.

### R-33 results — half the headline is the exposure, and the other half is not established

**Reproduction first.** `buy_and_hold` was carried through this round's
pipeline as a control, and it returns R-29/R-30's published numbers to
three decimals: Δ max drawdown **−27.076 [−35.778, +1.890]** on the spot
holdout, **−29.306 [−41.047, −4.996]** on futures, Δ log growth
**−0.129 [−0.941, +0.744]** on spot. So every difference below comes from
the matching and not from a reimplementation.

**V — the validity gate, applied before any decision rule was read. Five
of six cells VOID**, and prediction 1 is the one that landed:

| cell | v4 vol | arm vol | gap | verdict |
|---|---|---|---|---|
| spot / rebalanced c=0.353 | 0.317 | 0.167 | **47.4%** | **VOID** |
| **spot / static c=0.293** | 0.317 | 0.284 | 10.4% | **VALID** |
| spot / notional-matched c=0.283 | 0.317 | 0.133 | **58.0%** | **VOID** |
| futures / rebalanced c=0.348 | 0.375 | 0.164 | **56.2%** | **VOID** |
| futures / static c=0.289 | 0.375 | 0.282 | **24.9%** | **VOID** |
| futures / notional-matched c=0.289 | 0.375 | 0.136 | **63.6%** | **VOID** |

The mechanism is exactly the predicted one and it is worth stating as a
general fact rather than as an accident of this round: **a
volatility-targeting strategy and a constant-exposure hold cannot be
risk-matched across a regime change.** v4 holds its realized volatility
near 0.29–0.32 in both periods by construction; the market's own
volatility fell about 43% from 2021–22 to 2023+, so an exposure frozen on
the earlier period delivers roughly half the intended risk in the later
one. The static arm survives on spot only by accident — its weight drifts
*up* through a bull, which happened to restore the match.

**A diagnostic that is not part of any decision rule and is the most
quotable line in the round: on the spot holdout `kelly_regime_v4` asks
for more than 1.0 notional on 40.7% of bars.** Four bars in ten it is not
sizing anything; it is buy-and-hold with the cap doing the sizing. (R-31
independently reports 41.0% for its reconstruction of the same gate on
the same period, from different code.) On the inner splits the figure is
7.4% and 2.3%, so this is a property of the calm 2023+ regime, not of the
strategy's design — but every spot number v4 has ever published on this
holdout carries it.

**D1 — the B-13 question. FAILS.** The single valid cell, paired
stationary block bootstrap on 1,319 daily observations, stated as
v4 − benchmark so negative means v4 draws down less:

| valid cell | Δ max drawdown | 95% CI | P(>0) |
|---|---|---|---|
| spot / static hold, c=0.293 | **−14.18pp** | **[−22.68, +13.48]** | 0.35 |

The interval contains zero, so per the rule fixed in advance the claim
"regime-gated sizing cuts drawdown" is **downgraded to: established
against a fully-invested benchmark only, and not established against a
de-levered one.** That downgrade is now in the README, in
`VALIDATION.md` and in L-04's row above. Note what it is not: the
full-history −41.1pp [−54.8, −18.4] against fully-invested holding is
unchanged and still excludes zero. What has changed is what that number
is allowed to be *about*.

**D2 — return, and the round's genuine surprise.** Same cell,
Δ log growth **+0.611 [−0.116, +1.377]**, P(>0)=0.94 — not established,
and the sign is v4's. It is v4's in every cell of every table in this
round, valid and void alike, on both markets and on both assets. See the
window resample below, where it is not a single path.

**The void cells, and why they are the same mistake in a mirror.**
Against the rebalanced arms — which on the holdout carry only ~0.53x v4's
risk — v4 draws down *more*: **+4.24pp [−0.47, +26.88]** on spot and
**+8.03pp [+2.35, +30.78]** on futures, the latter excluding zero. That is
not a finding about gating; it is the arithmetic this whole round exists
to expose, running in the opposite direction. Give any arm half the risk
and it wins on drawdown, whether that arm is R-28's e-process gate, a
constant-exposure hold, or `kelly_regime_v4` against a fully-invested
benchmark.

**Descriptive re-match — NOT PRE-REGISTERED, and labelled as such.**
Solving `c` on the holdout itself is selection on the holdout, so nothing
here supports D1. It is reported because "the frozen exposure did not
match risk out-of-sample" otherwise leaves the reader without the number
they actually want, and refusing to compute it is its own kind of
dishonesty (`run_matched_hold.py rematch`). Volatility gaps ≤ 1.7%:

| cell | c | Δ max drawdown | 95% CI | Δ log growth | 95% CI |
|---|---|---|---|---|---|
| spot / rebalanced | 0.671 | −12.58pp | [−20.13, +12.62] | +0.228 | [−0.363, +0.808] |
| spot / static | 0.375 | −17.45pp | [−26.07, +10.76] | +0.491 | [−0.212, +1.211] |
| futures / rebalanced | 0.793 | −14.46pp | [−23.35, +11.51] | +0.432 | [−0.304, +1.198] |
| futures / static | 0.515 | −17.06pp | [−26.45, +11.08] | +0.690 | [−0.149, +1.584] |

Eight intervals, eight containing zero. Every drawdown point estimate is
in v4's favour and every one of them is roughly **half** the published
gap against fully-invested holding (−27.1pp spot, −29.3pp futures).

**D3 — the falsification test, and the result that most distinguishes
this round from R-28's. PASSES.** ETH on Bitfinex over the R-17 window,
exposures re-matched on each asset's own volatility:

| asset / market | matched vol | `kelly_regime_v4` | vol-matched rebalanced hold | Δ max DD |
|---|---|---|---|---|
| ETH / spot | 0.407 | **$5,482** (DD 36.5%) | $3,827 at c=0.301 (DD 51.3%) | **−14.8pp** |
| ETH / futures | 0.430 | **$4,263** (DD 35.1%) | $3,900 at c=0.317 (DD 54.0%) | **−18.9pp** |
| BTC control / spot | 0.402 | **$12,278** (DD 40.1%) | $4,972 at c=0.438 (DD 50.0%) | **−9.9pp** |
| BTC control / futures | 0.437 | **$25,681** (DD 32.1%) | $5,520 at c=0.475 (DD 53.1%) | **−21.0pp** |

The sign replicates on a second asset, on both markets, and v4 wins on
return in all four cells at matched risk. **This is precisely the test
that killed R-28's claim** — there, holding risk fixed *reversed* ETH's
drawdown ordering — and v4 passes it. D3 is the strongest single piece of
evidence this round produces, and it is a falsification test rather than
a selected number.

**Path sensitivity — 40 windows, and this time genuinely at matched
risk.** Both B-11 branches had to caveat their window tables with
"exposures frozen rather than re-matched per window", and R-31's futures
arm was visibly running hot as a result. That caveat is avoidable here: a
constant-exposure arm's realized volatility is proportional to `c` to
better than 1%, so one probe backtest per window fixes the exposure that
matches v4 **inside that window**, exactly. Achieved median |volatility
gap| **0.51%** on spot and **0.53%** on futures (worst 3.4% / 4.7%), with
the matching exposure ranging c ∈ [0.24, 0.78] across windows — itself a
measure of how unstable the frozen match was always going to be.

| paired, per window | Δ median return | v4 higher in | Δ median max DD | v4 **deeper** in |
|---|---|---|---|---|
| v4 − `buy_and_hold`, spot | −9.1pp | 42% | **−24.5pp** | **0%** |
| v4 − per-window matched hold, spot | **+20.8pp** | **82%** | **−2.9pp** | 22% |
| v4 − `buy_and_hold`, futures | +93.3pp | 57% | **−70.7pp** | **0%** |
| v4 − per-window matched hold, futures | **+23.8pp** | **90%** | **−5.5pp** | 15% |

**This is the number the round was run to get.** At genuinely equal risk,
across 40 identical windows, v4's median drawdown advantage falls from
−24.5pp to **−2.9pp** on spot and from −70.7pp to **−5.5pp** on futures.
**88% and 92% of the drawdown gap is the exposure level.** The remaining
2.9 and 5.5 points are what the gate is worth on this axis, and they are
small enough that the holdout's single path cannot resolve them — which
is exactly what D1 found.

And the mirror image, which is not what anyone expected to find here: at
equal risk v4 out-returns the passive arm by a median **+20.8pp / +23.8pp
per window, in 82% and 90% of them**. v4's *return* advantage over an
equal-risk passive hold is far more consistent across paths than its
drawdown advantage. That is the opposite ordering from the one this
project has been publishing for a year.

**Costs.** At Bitstamp's 0.40% entry tier v4 falls $3,373 → **$2,445**
(fees $310 → $1,027, drawdown 27.8% → 34.1%) while the passive arms are
essentially fee-free ($1,766–$1,828, unchanged to three digits) and
`buy_and_hold` gives up $12. Consistent with R-13: nothing here beats
holding at that tier, and the strategy is the only thing the tier hurts.
With funding charged on 5x futures, leveraged holding is **liquidated**;
v4 finishes at $3,133 (DD 37.8%) having paid **$1,190** of funding, and
the frozen passive arms at $1,150–$1,397 paying $246–$669 — but those
arms are the void low-risk ones, so read the funding column as scaling
with exposure, which is the R-14 finding again.

**Deflated Sharpe.** This round's 18 configurations have an
inner-validation daily-Sharpe dispersion of **0.077** — a narrow search,
because sweeping the exposure of a passive arm is not the same kind of
search as sweeping a signal. Carrying R-28's measured dispersion of 0.223
instead, v4's holdout spot Sharpe of 1.208 deflates to **0.896 at 103
trials (R-29's published figure, reproduced exactly), 0.882 at 172 and
0.879 at this round's 190.** Still under 0.95, as everything
out-of-sample in this project has been since R-29.

**Scoreboard against the predictions written before looking.**
Prediction 1 (V1 fails on at least one market) — **correct**, and it
voided five cells of six. Prediction 2 (D1 fails; 80–90% of the gap is
exposure) — **correct on both limbs, and the point estimate depends on
which instrument you read**: the 40-window matched resample says 88% and
92%, dead inside the predicted band; the single-path holdout re-match
says about 50%. Both are reported; the window figure is the better
instrument because it is 40 paths rather than one and its match is 0.5%
rather than frozen. Prediction 3 (v4 wins on growth at matched risk) —
**correct, and stronger than predicted**: 82–90% of windows, all four
ETH/BTC cells, every holdout cell. Prediction 4 (D3 replicates in sign) —
**correct**.

**Verdict: NEGATIVE on D1, and the decision rule did not move.** The
claim this project has led with since L-04 is downgraded by a rule
written before the numbers existed.

**Lesson.** *This project's one robust finding was mostly arithmetic, and
the thing underneath it is a different finding.* Nine tenths of "regime-
gated sizing cuts drawdown" is "regime-gated sizing holds about half the
notional", which is true of any strategy flat on 29–44% of its bars
and needs no gate to achieve. What survives matching is small on
drawdown (−2.9pp / −5.5pp median, not resolvable on one path) and
**consistent on return** (+20.8pp / +23.8pp median in 82–90% of windows,
replicating on ETH). So the headline should not be retired, it should be
*re-pointed*: v4's defensible claim is that it converts a given risk
budget into more return than a constant exposure does — not that it
protects a drawdown a smaller position would not have protected.

The second lesson repeats R-31's and is now three-for-three: **the
validity gate earns its place every time.** Without V, this round would
have led with "v4 draws down 4–8pp *more* than a de-levered hold" from
the frozen cells, which is as wrong in one direction as the −41.1pp
headline is in the other, and for the same reason.

**Configurations evaluated: 18** (2 passive arms × 9 exposures on both
inner splits and both markets, 72 backtests), plus solver and matched-pair
backtests recorded separately per the R-31 convention. The project trials
count this round contributes is **18**; the count it applies is
172 + 18 = **190**.

**Holdout counter: ~152** (~124 before, +28 this row: 10 frozen holdout
runs across two markets, 8 for the descriptive re-match and its solver,
and 10 cost re-runs — 5 at the 0.40% taker tier and 5 with funding
charged). The ETH/BTC falsification cells and the 40-window resample do
not read the 2023+ BTC holdout, following the R-19/R-28/R-31 convention.

![how much of the drawdown finding is the exposure level](../reports/matched_hold/matched_drawdown.png)

**Next step → B-14** (below), and then **B-05**. B-13 asked whether the
drawdown claim survives risk-matching and the answer is *mostly not*; the
question it leaves is the one D2 kept answering by accident, in every
cell, on both assets and in 82–90% of 40 windows — **what is v4's
return-per-unit-risk advantage over a constant exposure, and does it have
an interval?** This round measured it four different ways and
pre-registered none of them, so none of it can be claimed. That is
exactly what a pre-registered round is for, and it is cheap: the harness
exists, the arm exists, and the matching is already solved to 0.5%.

### R-34 — L-12's stated hypothesis, finally tested: does the crowd posterior work as a SIZE input?

**Idea, in one sentence.** `harsanyi_crowd` (L-12) builds a Bayesian
posterior over three hidden market types — up-trend, down-trend, chop
(Harsanyi 1967-68) — from bar-return likelihoods with a sticky transition
prior, and trades its belief margin *directionally*; it loses. L-12's own
recorded lesson says why it might still be useful: *"the crowding
intuition was right — it is what `kelly_regime` later exploited — but as
a direction signal rather than a sizing input it loses."* That sentence
has sat in this ledger since 08-12 as an untested hypothesis. This round
tests it, on the only axis this project's twenty-five strategies have
ever found to work: SIZE.

**Constraint attacked.** SIZE, primarily — refining "how much to hold" on
top of the incumbent `kelly_regime_v4`, changing nothing about its
conditional-volatility-targeting risk axis. A narrower ERR claim was
also on the table (a smoothed Bayesian posterior is a calibrated
confidence signal, unlike a hand-set 1% anchor band) but neither branch
below leans on it — R-28/R-31/R-32 already showed an anytime-valid gate
does not beat the heuristic vote at matched risk, so this round does not
repeat that overclaim.

**Not a duplicate of.** L-04/L-01/L-02/L-03 all vary the vote's *transform*
(gamma, anchor spacing, the risk axis it multiplies) but never the
*nature* of the confidence signal — still price-anchor threshold
crossings. L-12 built the exact posterior used here and is the direct
parent; this is the first round to route its output through the SIZE
axis instead of DIRECTION. R-28/R-31/R-32 test a different confidence
mechanism entirely (an anytime-valid e-process martingale, not a
Bayesian type posterior) and already closed the question "does swapping
*which* gate mechanism drives the exposure matter" (mostly not, at
matched risk) — this round is compatible with that finding, not a re-run
of it, since both variants here keep v4's own gate/vote machinery in
place to different degrees rather than replacing the gate concept itself.

**Simulable here?** Yes — pure OHLCV, causal, no new data. The posterior
recursion is lifted byte-for-byte from `harsanyi_crowd.py` (already
CI-covered, already causal) into a shared helper,
`experiments/bayes_confidence.py`, so neither variant re-derives it.
Verified independently by hand (two-opposite-tampers probe, R-28's
method): max|diff| = 0.0 before the cut.

**What would make each variant fail — named before either was built.**
(a) The posterior margin correlates highly with v4's existing 3-anchor
vote, so results are a smoothed vote in disguise inside the noise floor
(the generic "another indicator" failure ROUTINE.md warns about).
(b) A continuous signal re-trades on every small wiggle even under a
deadband, and fees eat the gain (the L-14/L-15/L-16 failure mode).
(c) Any apparent drawdown improvement is actually an exposure-level
artifact — this project's last three "risk improvement" claims all died
this way (L-04/R-33, R-28/R-31, R-32) — so both variants were required to
report mean exposure against v4 on the same window *before* any drawdown
claim could be trusted, and where they diverge, to check what a
matched-exposure comparison shows.

**Method.** Two independent, unregistered variants dispatched in
parallel, each on a disjoint file, neither touching the 2023+ BTC
holdout, neither committing — the operator (this session) merged and
recorded both after reading each report in full, per ROUTINE.md's
parallelism rules.

- **Conservative — `experiments/kelly_regime_v5_damp.py`.** v4's vote and
  conditional-vol-targeting sizer are left completely unchanged; a
  smoothed, floored-at-zero confidence weight from the posterior margin
  only ever *shrinks* the result through `mult = 1 − lam·(1 − conf) ∈
  [1−lam, 1]`. By construction this can never raise exposure above v4's,
  which is what makes it conservative — and, as the result below shows,
  is also what dooms it to the exposure-artifact failure mode by
  architecture rather than by accident.
- **Novel — `experiments/kelly_regime_v5_bayes.py`.** v4's discrete vote
  is replaced entirely by a continuous, hysteresis-latched transform of
  the posterior margin (the same latch *shape* `harsanyi_crowd` uses for
  its own entry/exit bands, `b_in`/`b_out`, but mapped to a continuous
  fraction rather than a binary trade/no-trade decision), feeding v4's
  unchanged conditional-vol-targeting sizer. This is the deeper
  redesign: the confidence source is now a fast, hours-to-days Bayesian
  filter rather than v4's slow 20/40/80-day anchors, so it could in
  principle time entries and exits the vote cannot.

**Configurations evaluated: 42** — conservative swept `lam ∈ {0, 0.1,
0.2, 0.3, 0.4, 0.5}` (6, `lam=0` reserved as a correctness check —
reproduces v4 bit-for-bit) on inner-validation, both markets; novel swept
two 18-config grids (`stick × b_in × b_out`), one at the parameter values
suggested going in and a second rescaled to the posterior margin's actual
empirical distribution after the first grid revealed most of the
suggested `b_in` thresholds sit above the 99th percentile of the signal
and rarely latch (36 total). The project trials count this round
contributes is **42**; the count it applies is 190 + 42 = **232**.

**Results — conservative branch.** Mean |exposure| is *strictly lower*
than v4 at every `lam > 0`, on both markets, monotonically (spot ratio
0.90 → 0.49 as `lam` runs 0.1 → 0.5) — an architectural fact, not a
measurement, since `mult` can only shrink. Spot Sharpe declines in
lockstep with exposure (0.14 → 0.02); one non-monotone spike on futures
at `lam=0.2` (Sharpe 0.36 against neighbours 0.17/0.08) was flagged as
not a plateau and not selected on. At the pre-registered default
`lam=0.3`: never beats v4 on return in any of 4 inner-train/inner-validation
cells; beats it on drawdown in all 4 (e.g. inner-validation spot: $985 vs
v4's $998, DD 26.2% vs 33.2%). ETH/BTC falsification ordering (worse
return, better drawdown) is identical in all 4 cells — no flip. Turnover
0.79–1.0x v4's, no fee-drag flag. The correlation between the smoothed
confidence weight and v4's discrete vote is only **0.097** — not the
naively-expected redundancy — but once smoothed enough to avoid whipsaw
the margin has almost no independent dynamic range on 5-minute BTC (mean
0.025, std 0.012, max 0.087 on inner-validation): the resulting `target`
series correlates **0.9986 (R²=0.997)** with a flat **0.7x rescale of
v4**. It is not a smoothed copy of the vote; it is a smoothed copy of
*a constant*.

**Results — novel branch.** All 36 configurations underperform v4 on
inner-validation: spot Sharpe ranges −2.9 to −3.9 against v4's +0.14,
turnover 4–7x v4's (209–341 trades vs 52), mean exposure only 5–19% of
v4's. Unlike the conservative branch, this gap is *not* an unexamined
exposure artifact: an explicit diagnostic multiplier
(`exposure_mult=5.27`) rescaled the best config to match v4's mean
notional, and results got catastrophically worse rather than better
(Sharpe −6.25, drawdown 92%, trades nearly tripled) — ruling out
"correctly calibrated but under-sized" as the explanation. ETH/BTC
falsification: v4 wins on return in all 4 cells; the drawdown ordering
flips only on ETH futures, a low-exposure mechanical effect that does not
touch the headline (return) comparison. Vote correlation is **−0.0017** —
genuinely independent of v4's mechanism, not a smoothed duplicate of it.
Causality: truncation probe max|diff| = 0.0 at both default and matched-
exposure parameters; the unregistered file does not run under
`test_causality_strict.py`, but the full suite (51 causality tests) still
passed, unaffected.

**Verdict: NEGATIVE, both branches — and each fails for a different,
clean reason.** The conservative branch's failure mode is the one this
project keeps re-discovering with new instruments: any signal that can
only *subtract* exposure will manufacture a drawdown "improvement" that
is arithmetic, not mechanism, unless the comparison is matched — and
this one, checked, was arithmetic. The novel branch's failure mode is
different and is the more informative result of the two: this is
*not* the exposure-level artifact (matching disproves it directly) and
*not* a smoothed duplicate of the existing vote (the correlation is
essentially zero) — it is a genuinely different, causally valid,
economically motivated regime-confidence signal that simply loses,
because the Bayesian posterior's native cadence (hours-to-days, driven
by single-bar ATR-normalized likelihoods) is too noisy relative to
5-minute-bar trading costs to serve as a timing input on *either* axis,
direction or size. L-12's hypothesis is now closed rather than open:
tested honestly, on the axis it proposed, and it does not hold.

**Holdout counter: ~152, unchanged.** Neither branch read the 2023+ BTC
holdout at any point — the pre-registered decision rule (available on
request; effectively "promote to a holdout read only if inner-validation
clears v4 by more than the ±0.2 Sharpe noise floor or a matched-exposure
drawdown edge, and survives ETH") was never satisfied, so step 4 was
never reached. Consistent with ROUTINE.md's guidance that a session
finding nothing worth a holdout read should say so and stop rather than
force one.

**Next step.** Closed as a direction — the specific hypothesis L-12 left
open has now been tried on the axis it proposed and failed cleanly on
both a bounded and an unbounded implementation. **B-14 (return per unit
of risk against a constant exposure) remains the ranked top of the
backlog**, unaffected by this round; **B-06 (forward paper trading)
remains blocked on network** and remains the highest-value item on
merit. A future session revisiting the Bayesian-posterior idea would need
a confidence source with more independent variance surviving multi-day
smoothing than this (mu=0.15, stick=0.985) parametrization supplies on
5-minute BTC — not a re-sweep of these same knobs.

---

### R-35 pre-registration — written and committed before the holdout was read

**Idea, in one sentence.** `kelly_regime_v4` is structurally short the
funding premium — R-14 measured it paying **+20.05%/yr while it holds**
against +2.78%/yr while flat, because the crowding the strategy trades
*is* the crowding that sets the rate — and R-16 found funding itself
predicts forward returns (14-day Q1−Q5 spread +3.57pp) without being
derived from price; wire that into the strategy on the only axis this
project's twenty-five strategies have ever found to work, SIZE, instead
of leaving it as a cost line item and a descriptive table.

**Constraint attacked.** COST, primarily — this is the first round in
the project to turn the R-14 finding (costs scale *with* the signal) into
a strategy change rather than a measurement. SIZE secondarily, and INFO
in a narrow sense: funding is the one signal in this repo that is not a
transform of the OHLCV series, which is what sank the four `INFO`-labelled
entries in section A (L-12, L-14, L-15, L-16).

**Not a duplicate of.** L-05/L-06 derive a no-trade band from the taker
**fee** only (Constantinides 1986; Davis & Norman 1990) and never touch
funding. R-14 measured funding as a cost with no strategy change. R-16
measured funding as a *forecaster* of forward returns and explicitly
named this exact mechanism, "a gate that stands flat when funding is in
its top decile," as the low-turnover way to use it — this round is that
backlog item, B-05, executed for the first time. R-34 tested a different
SIZE-axis confidence signal (a Bayesian posterior over price-derived
regime types) and found it too noisy at its native cadence; funding
is a structurally different input (an exchange-determined rate, not a
statistic of price) so this is not a re-run of that question, though R-34's
failure mode (a fast, noisy signal that re-trades on wiggles the vol-target
deadband cannot absorb) is exactly the thing both variants below are
designed to guard against.

**Simulable here?** Only partially, and this is named now rather than
discovered in step 4. Real Binance BTCUSDT funding is committed for
**2020-01-01 through 2023-12-31 only** (4,383 settlements). Against the
routine's splits:

| slice | dates | funding coverage |
|---|---|---|
| inner-train | 2017-01-01 → 2020-12-31 | real for its final ~12 months only |
| inner-validation | 2021-01-01 → 2022-12-31 | fully covered |
| holdout | 2023-01-01 → | real for **2023 only** — 2024–2026 (the other ~2.6 years of the nominal holdout) has none |

Per the standing rule "never proxy unavailable data out of price," bars
outside 2020-01-01..2023-12-31 get **no funding value substituted** —
both variants must define an explicit inert default (mechanically, the
gate never fires / the drag term is exactly zero) for those bars rather
than filling forward, backward, or with a period mean. That default
means the mechanism *cannot* alter v4's trades outside 2020-2023 by
construction, which has two consequences fixed here in advance rather
than argued about after looking: **first**, this round asks the 2023+
holdout only through **2023-12-31**, the actual end of measured funding
coverage — reading 2024-2026 would spend a holdout consultation on a
period where a difference from v4 is definitionally impossible, which
this project's own holdout-exhaustion finding (R-29, ~88 consultations
before this session and ~152 after R-34) says is a cost worth avoiding
now that it is visible. **Second**, the round is explicitly underpowered
relative to a full-coverage mechanism and any promotion decision must
say so rather than present a 2023-only result as if it carried the
holdout's usual weight.

**A second, sharper limitation, named because this project's culture is
to name the contamination rather than let a reader find it.** R-16's own
descriptive quintile finding was computed over the *same* 2020–2023 span
that supplies every year of funding data available to this round. There
is no funding-covered period this idea has not already been looked at
once. Step 3 below still respects the inner-train/inner-validation split
mechanically — no parameter is chosen by looking at 2023 — but a reader
should treat 2021–2022 selection as the first walk-forward test of a
relationship whose existence was established on data that overlaps it,
not as evidence from a wholly fresh window. This is the reason the
promotion bar below is deliberately conservative.

**What would make each variant fail — named before either was built.**
(a) Funding richness is highest exactly when v4's own vote is already
bullish and its conditional-vol-targeting scale is already elevated (both
are reading the same crowding), so a gate keyed on funding merely
re-shrinks exposure in states v4 already prices in, landing inside the
noise floor or reducing to the exposure-level artifact this project has
now hit three times (L-04/R-33, R-28/R-31, R-32) with three different
source signals. Both variants report mean exposure against v4 before any
drawdown claim is trusted, per that standing lesson. (b) Funding updates
every 8 hours, far faster than v4's 20/40/80-day anchors; an
insufficiently smoothed percentile rank re-trades on funding noise and
the deadband cannot absorb it, the L-14/L-15/L-16 and R-34-novel-branch
failure mode. (c) The literature's own caution: funding carry is reported
to have compressed and gone negative by 2024–25 as the trade crowded
(He et al. 2024; the repo's own R-15 note) — even inside the
funding-covered window, an edge measured on 2020–2022 need not survive
into 2023, and this round's short holdout (one year, not the usual 3.6)
has less power to catch that than any prior round in this file.

**Method.** Two independent, unregistered variants, each on a disjoint
file, neither touching the holdout beyond the pre-registered 2023-only
window, neither committing — the operator (this session) merges and
records both after reading each report in full, per ROUTINE.md's
parallelism rules.

- **Conservative — `experiments/funding_gate_decile.py`.** v4's vote and
  conditional-vol-targeting sizer are left completely unchanged; a
  binary override forces `target = 0` on any bar where the current 8h
  funding rate's trailing rolling-window percentile rank is **≥ the 90th**
  — the literal backlog reading, "stand flat when funding is in its top
  decile." The only swept knob is the rolling lookback used to rank the
  rate (the threshold itself is fixed at 0.90, not tuned) and, if needed
  to control 8h-cadence noise, a short causal smoothing of the raw rate
  before ranking. Primary market is **futures**, where funding is an
  actual cost; **spot** is run as a secondary/diagnostic cell only, since
  funding is not paid there and any spot effect would isolate R-16's pure
  return-forecast channel from the cost-avoidance channel.
- **Novel — `experiments/funding_ev_band.py`.** Generalizes L-05's
  analytic no-trade band (rebalance only when the growth given up,
  `(σ²/2)(f−f*)²` per unit time, exceeds the cost of moving) by adding a
  forecast funding-drag term to the cost side, using a causal EWMA of the
  trailing funding rate as the near-term forecast (funding is strongly
  autocorrelated at 8h cadence). Because carrying cost is a form of the
  continuous holding cost in Dumas & Luciano's (1991, J. Finance)
  two-barrier portfolio-choice framework, expected funding richness
  widens the band on the side that would add exposure and, more directly,
  haircuts the growth-optimal target `f*` itself by the forecast drag
  before the band is applied — since a Kelly-optimal sizer should already
  net out a *known* cost of holding, not just a cost of trading. Must
  reduce to `kelly_regime_ev`/v4 exactly when forecast funding is zero
  (the `lam=0`-style built-in correctness check other experiments in this
  file use), and can only ever reduce long exposure, never raise it,
  since carrying cost never makes holding more attractive.

**Pre-registered falsification test.** With real funding charged as a
first-class cost on the futures P&L (`funding=` on the engine, the
`funding_study.py` convention) — not the funding-free perp every other
figure in this repo uses. This is the most direct test available: a
mechanism built to attack the COST constraint must not merely look good
on a funding-free backtest while the actual funding bill is what it was
built to reduce. Secondary check: survives the 0.40% Bitstamp entry taker
tier (`fee_study.py` convention).

**Pre-registered decision rule.** Promote to the ledger as a candidate
registration only if, on inner-validation (2021–2022, the only fully
funding-covered inner slice): (i) either variant beats v4 on Δ log growth
by more than the ±0.2 Sharpe noise floor, **or** matches v4's return
within that floor while cutting max drawdown, on futures, *and* the
mean-exposure check does not reduce the finding to a flat rescale of v4;
(ii) survives the funding-charged falsification test; (iii) the parameter
neighbourhood is a plateau. If both variants fail (i), the round is
recorded NEGATIVE without a holdout read, exactly as R-34's off-backlog
round did — a direction with nothing worth a 2023-only holdout
consultation should say so and stop rather than force one. If either
clears (i)–(iii), the 2023-01-01..2023-12-31 slice is read once, scored
with the R-29/R-30 paired stationary block bootstrap, and the result is
reported however it comes out, with the underpowered-holdout caveat
stated alongside it rather than after it.

**Stated prediction before any code ran.** The conservative branch
predicted to fail on mechanism (a): funding richness should correlate
strongly with v4's already-elevated exposure states, so its drawdown
cut, if any, is expected to be another exposure-level artifact. The
novel branch is the one with a real chance: it acts on the sizing
*target* rather than adding a binary override, so it is less exposed to
(a) by construction — but is expected to be small, since v4 is already
flat or de-levered in exactly the bear/high-funding-mismatch regimes
where the carry premium would matter most, leaving limited room for a
carry-aware haircut to do additional work.

### R-35 results — a genuine effect that clears every gate but the last one

**Method actually followed.** Two independent agent sessions, each on a
disjoint unregistered file, neither touching 2023-01-01 or later, neither
committing — reports at `experiments/reports/funding_gate_decile_report.md`
and `experiments/reports/funding_ev_band_report.md`. Both reports were
read in full before either the holdout was read or this row was written.
The conservative branch's headline numbers and its causality claim were
**independently re-derived by the operator** (not merely trusted from the
report) by re-running its exact configurations directly against
`tradebot.engine.run_backtest` and a fresh two-opposite-tampers probe on a
different data slice than the one in its report — every figure reproduced
to the cent and the causality probe again showed max|diff| = 0.0 before
the cut. This is the "independent skeptic" step ROUTINE.md's parallelism
section asks for, done by the operator directly rather than a third
dispatched agent, since the claim was cheap enough to re-derive in minutes.

**Conservative branch (`funding_gate_decile`, decile fixed at 0.90).**
Cleared all three inner-validation criteria, including the one its own
pre-registration predicted it would fail:

- **(i) Return, clearly.** `w=90`: ΔSharpe +0.77 vs v4 on inner-validation
  futures ($1,564 vs $1,064, DD 20.0% vs 32.3%). `w=180` (the recommended,
  non-cherry-picked config — the middle of the four pre-registered sweep
  points, matching v3/v4's own 180-day anchor convention): ΔSharpe +0.29
  ($1,238, DD 26.3%). Both well past the ±0.2 floor.
- **Exposure-artifact check, the one that mattered most.** Mean exposure
  is genuinely lower than v4's (15–21%), which is exactly the trigger
  condition for the check the standing diagnosis demands. A flat rescale
  of v4 to the *identical* mean exposure reaches only Sharpe 0.30–0.35 —
  the actual gate reaches 0.54–1.02. **This is not the L-04/R-28/R-32
  pattern.** The gate is doing something a scalar de-lever cannot
  reproduce: timing *when* to be flat, not just trading less on average.
  Confirmed independently by re-running the `w=180` numbers directly
  (see above) — not taken on the report's word alone.
- **(ii) Funding-charged falsification: clean pass for `w=90`/`w=180`.**
  With real funding actually deducted, `v4` goes Sharpe-negative (−0.06)
  while `w=180` holds 0.34 and `w=90` holds 0.79 — independently
  reproduced by the operator to the cent (§ above).
- **(iii) Plateau: real but regional, honestly not full-coverage.** Every
  point from 30–250 days beats v4 on both axes; the two long-lookback
  points the pre-registration explicitly mandated (365 days, expanding
  from 2020) do not. Read charitably (a neighbourhood around the chosen
  config) this passes; read as "every mandated sweep point," it does not.
  Recorded both ways rather than picking the flattering reading.

Per the pre-registered decision rule this cleared the bar for a holdout
read — the first time in this branch's construction that the
pre-registration's *own stated prediction for it* (that it would reduce
to the familiar exposure-level artifact) was checked directly and did
not hold.

**Novel branch (`funding_ev_band`, target haircut + asymmetric band
widening, Dumas & Luciano 1991-style).** A structurally different result
worth recording on its own terms, not merely as "the one that didn't
qualify": its best config (`hc=0.5, span=10d`) also clears the funding-
charged falsification cleanly (v4/`kelly_regime_ev` both flip
Sharpe-negative once funding is charged; this config holds 0.38 while
paying roughly a third of the funding bill) and its Sharpe edge is
**not** an exposure artifact — rescaled to v4's own mean exposure the
Sharpe edge *widens* (0.67 vs 0.25), the opposite of the usual failure
mode. But its drawdown edge substantially *is* explained by lower
average exposure once rescaled (33.8% vs the un-rescaled config's
26.0%), and its winning grid cell sits 0.11–0.21 Sharpe above its
immediate neighbours — a gap the size of the noise floor itself, in a
two-year window where both baselines' own Sharpe sits near zero. Per
its own report's honest self-assessment and the pre-registered rule's
plateau requirement, this branch did not clear the bar for a holdout
read, and the operator agrees with that self-assessment on review of the
grid rather than overriding it. Its holdout counter contribution is
therefore zero.

**The holdout read (pre-registered: 2023-01-01 → 2023-12-31 only, the
funding-covered slice, `FundingGateDecile(funding_window_days=180,
decile=0.90)` frozen, `w=90` deliberately NOT also read — this project's
now-standard economy of "ask the holdout fewer questions," the same
restraint R-31 applied to its `conditional` sizer arm).** Paired
stationary block bootstrap, 30-day mean block, 2,000 resamples, on daily
returns, identical to the R-29/R-30 convention:

| market / cost regime | v4 final | gate(180) final | Δ log growth (gate−v4) | 95% CI | P(gate>v4) |
|---|---|---|---|---|---|
| spot, funding-free | $2,038 | $1,794 | −0.128 | [−0.378, +0.087] | 0.148 |
| futures5x, funding-free | $2,594 | $2,195 | **−0.167** | [−0.495, +0.101] | 0.144 |
| futures5x, funding-charged | $2,393 | $2,121 | −0.120 | [−0.430, +0.139] | 0.209 |

*(context, not part of the decision rule: `buy_and_hold` spot finished
$2,556 in the same window — both v4 and the gate trail it, consistent
with L-01's own documented weakness, "it lags badly in steady bulls,"
and 2023 was one.)*

**Verdict: NEGATIVE.** Every interval contains zero — this single
underpowered year cannot reject "no difference" at 95% either way — but
the *point estimate* is negative on every cell, funding-free and
funding-charged, both markets, and `P(gate beats v4)` sits at 0.14–0.21
throughout, not near the 0.50 a true coin flip would show. The honest
read is not "unproven" so much as "the one holdout year available leans
the wrong way for an idea that looked genuinely strong in-sample." This
is the pattern this project has hit more often than any other — R-12's
28-of-32 in-sample / 0-of-28 out-of-sample is the sharpest prior
instance, and R-28's e-process gate is the closest cousin (a real,
independently-verified in-sample/inner-validation effect that did not
survive contact with a holdout) — reproduced here with a third source
signal and, this time, with the operator's own independent re-derivation
of the in-sample numbers ruling out "the report was simply wrong" as the
explanation.

**What is different from a clean rejection, stated plainly so a future
session does not over-read this as closed.** Unlike R-12 (28 configs
checked out-of-sample and 0 survived) this round has exactly **one**
holdout year to check against, because that is all the real funding data
covers (docs/LEDGER.md's own pre-registration named this limitation
before any code ran). One year, 13–20 trades, is not enough power to
distinguish "the effect is real but 2023 was simply the unlucky year" from
"the inner-validation result was itself noise despite passing every
mechanistic check available." Both branches' funding-charged falsification
result — v4 and `kelly_regime_ev` both turning Sharpe-negative once real
funding is actually deducted, independently confirmed on the holdout
itself in the table above (v4 funding-charged Sharpe on this holdout: see
raw figures; the direction matches inner-validation) — is not touched by
this verdict and remains the strongest, most directly COST-relevant
finding in this row: funding is a real, adversely-timed cost on this
strategy family (R-14), and a signal that is not derived from price can
detect some of it (R-16), even though neither tested mechanism earns its
way into the registered strategy on the one year available to check it.

**Configurations evaluated: 80** (conservative 49 + novel 31, both
counted per each branch's own stated methodology, section 9/10 of their
respective reports). Project trials count before this session: 232
(R-34's running total). **Applies from here: 232 + 80 = 312.**

**Holdout counter: ~159** (~152 before, +7 this row — three paired
market/cost-regime cells × 2 strategies = 6, plus one `buy_and_hold`
context run = 7). The novel branch and `w=90`/`w=365`/expanding
configurations of the conservative branch never read it.

**Lookahead checks.** Both branches ran the two-opposite-tampers
procedure against their own file (bit-identical decisions before the cut
in every column checked, including the funding-derived diagnostic
columns) — see each report's own causality section for the full detail.
The conservative branch's probe was additionally reproduced by the
operator on an independent slice with an identical result. `pytest`
(436 tests) was green on the baseline commit before either branch
started; neither branch's unregistered files run under the registered-
strategy CI suite, by design (ROUTINE.md step 5).

**Next step.** Not promoted; both experiment files and their reports stay
under `experiments/` as documented negative results, per ROUTINE.md step
5. A future revisit of either mechanism needs either (a) more
funding-covered years to read against — which is backlog item **B-02**
(extend the funding series through 2026), still blocked on network access,
and would turn this row's single holdout year into several, the direct
fix for the power problem named above — or (b) a different, non-price,
non-funding COST-relevant signal tested the same disciplined way. Neither
branch is added to section C's ruled-out list: the effect was real enough,
by enough independent checks, that "do not re-try without new evidence" is
too strong a statement for a result whose only real objection is one
underpowered year. Re-trying the identical mechanism on the identical data
without new funding history would not be new evidence, though, so it does
not go back on the backlog as an open item either — it is closed for this
round and reopens only when B-02 delivers more holdout years to check it
against. **B-14 remains the top of the ranked backlog, untouched by this
row** (see below), and **B-06 (forward paper trading) remains the
highest-value item on merit**, still blocked on network access.

---

### R-36 pre-registration (B-14) — written and committed before the new analysis is read

**Idea.** R-33 measured `kelly_regime_v4` against a passive hold matched to
its own realized volatility and found, as a byproduct nobody had asked
for, that v4 out-returns the matched hold: median **+20.8pp / +23.8pp per
window in 82% / 90%** of 40 resampled windows, in all four ETH/BTC
falsification cells, and in every holdout cell (valid or void). B-14 asks
for that specific claim — return per unit of risk against a genuinely
matched passive benchmark — to be pre-registered as a primary question in
its own right, since nothing about it was pre-registered when R-33
produced it.

**Constraint attacked.** SIZE (does the *sizing* rule itself add value
beyond the exposure level it happens to run at) and ERR (the claim has
floated in this ledger for a full session without a decision rule).

**Not a duplicate of.** R-33 (measured the aggregate by accident, with a
*frozen* exposure for its own holdout arm, which is why five of six of
those cells were void — V1 in `run_matched_hold.py:holdout`); R-31/R-32
(matched risk on a different pair — the e-process gate vs the latched
vote, not v4 vs a passive hold); L-04's own headline (the *drawdown* claim
at matched risk, which R-33 killed — this row is the *return* claim R-33
left standing). The backlog row's own instruction is followed here: name
the bull-market failure mode and go look for it, rather than re-reporting
the aggregate as if that settled it.

**Simulable here?** Yes. `experiments/matched_hold.py`'s `windows()`
already implements per-window matching (the fix for R-33's frozen-exposure
failure: a probe backtest solves the matching exposure *inside* each
window, converging to a median |vol gap| under 1%). No new data, no new
strategy code — this is a re-read and an extension of an existing,
already-validated harness, run at its existing `seed=42` so the 40 windows
are the identical ones R-33 already published, not a fresh search.

**What's actually new.** The aggregate number is already known (from
R-33), so re-stating "predict D1 passes" would be circular. What has *not*
been computed is the one thing B-14's own text asks for: a breakdown of
the same 40 windows by calendar period, to check whether the advantage
survives outside the 2017–2020 bull. That breakdown is written by a new,
small, disjoint script (`experiments/b14_regime_breakdown.py`) that
recovers each window's start date from the identical `rng(seed=42)`
sequence `windows()` uses (same `warmup`, same `length` draws) — no
backtests are re-run; the existing `reports/matched_hold/windows.csv` is
reused for the return/vol numbers, and only the date lookup is new.

**Holdout accounting.** Per the R-19/R-33 convention already recorded in
this ledger ("the 40-window resample do[es] not read the 2023+ BTC
holdout"), this round does not increment the holdout counter — even though
some of the 40 windows' calendar spans overlap 2023+, the resample is the
project's standing robustness methodology, not a promotion-bar evaluation.
No holdout consultation is spent on this row.

**Pre-registered decision rule, D1 (primary).** Using the 40 paired
per-window observations already in `windows.csv` (`kelly_regime_v4` vs
`per-window matched hold`, both markets), compute the exact-binomial 95%
CI on the win-rate (v4's window return exceeds the matched hold's) and the
median paired return advantage. **Established** if the win-rate's 95% CI
excludes 50% on **both** spot and futures. This is a materially stronger
bar than "beats hold in most windows" — it is a two-sided test with an
interval, the R-29/R-30 discipline applied to a statistic this project has
so far only eyeballed.

**Pre-registered falsification test, named before the breakdown is run.**
The stated failure mode: *the advantage is concentrated in the 2017–2020
bull and does not generalize.* Split the 40 windows by whether the window's
**start date** falls before or on/after 2021-01-01 (the inner-
train/inner-validation boundary already used throughout this project).
Report win-rate and median advantage for each half separately. **The claim
downgrades to "bull-period artifact, not established generally"** if the
post-2021-start subsample's win-rate is ≤50% or its median advantage is
≤0, even if the pooled D1 statistic passes.

**Stated prediction before looking.** D1 passes (the pooled statistic is
already known from R-33 to be one-sided in v4's favor in 82–90% of
windows, which should not vanish under a formal binomial CI at n=40).
The falsification test is the genuinely open question — a real risk given
17 of `kelly_regime`'s home turf is documented to be "it lags badly in
steady bulls" (L-04's own stated weakness) and its *drawdown* edge was
already shown by this exact project (R-33) to be substantially an exposure
artifact of the same bull-heavy sample.

### R-36 results — D1 passes, and the falsification test survives, thinned out

`experiments/b14_regime_breakdown.py` recovered all 40 window start dates
from the identical `rng(seed=42)` sequence (verified: `warmup+10=23,060`
matches `kelly_regime_v4.warmup + 10`) and joined them against the
existing `windows.csv`. 18 of 40 windows start before 2021-01-01, 22 on or
after it — a reasonably balanced split given it was not chosen to
balance, only to match the inner-train/inner-validation boundary already
in use throughout this project.

| market | segment | n | win-rate | 95% CI | median Δreturn |
|---|---|---|---|---|---|
| spot | pooled (D1) | 40 | 82.5% | [67.2%, 92.7%] | **+20.8pp** |
| spot | pre-2021 start | 18 | 100.0% | [81.5%, 100.0%] | +68.9pp |
| spot | post-2021 start | 22 | 68.2% | [45.1%, 86.1%] | **+5.0pp** |
| futures | pooled (D1) | 40 | 90.0% | [76.3%, 97.2%] | **+23.8pp** |
| futures | pre-2021 start | 18 | 100.0% | [81.5%, 100.0%] | +97.2pp |
| futures | post-2021 start | 22 | 81.8% | [59.7%, 94.8%] | **+7.4pp** |

**D1: PASS on both markets** — the pooled win-rate's 95% CI excludes 50%
on spot ([67.2%, 92.7%]) and futures ([76.3%, 97.2%]).

**Falsification test: SURVIVES on both markets, by the pre-registered
rule** — the post-2021 subsample's win-rate exceeds 50% (68.2% spot, 81.8%
futures) and its median advantage is positive (+5.0pp spot, +7.4pp
futures) in both. The advantage is not exclusively a 2017–2020 bull
artifact.

**The honest qualifier, stated because the pre-registered rule did not
ask for it and would have missed it.** The post-2021 subsample's *own*
95% CI contains 50% on spot ([45.1%, 86.1%]) — at n=22 it cannot by itself
reject "no edge" at 95%, only the *point estimate* favours v4, on both
markets. And the magnitude drops by roughly 8–13x between the two halves
(+68.9pp → +5.0pp spot, +97.2pp → +7.4pp futures). The correct reading is
therefore two-part: **(1) some return-per-unit-risk advantage over a
genuinely matched passive hold generalizes past the 2017–2020 bull** — the
point estimate is positive and the win-rate exceeds 50% in a subsample
that includes the 2022 bear and the 2023+ cycle — **but (2) the large
number quoted in R-33 and reproduced in the pooled D1 statistic here is
substantially a bull-period effect**, consistent with L-04's own
documented weakness ("it lags badly in steady bulls") applying with the
opposite sign once the benchmark can no longer out-hold its way to a
bigger number just by carrying more risk.

**Configurations evaluated: 0.** This row is a fixed statistical readout
of an existing measurement (R-33's `windows.csv`, unchanged), not a
search — no parameter was swept, no threshold was tuned against the
result. It contributes nothing to the deflated-Sharpe trial count.
Project trials count unchanged at 312.

**Holdout counter: unchanged at ~159**, per the pre-registered accounting
above (the 40-window resample is not counted, matching the R-19/R-33
convention already recorded in this ledger).

**Lookahead / correctness check.** The date-recovery script makes no
trading decision and touches no strategy logic — it replays an RNG
sequence and indexes a timestamp column — so `test_causality_strict.py`'s
concern does not apply. What was checked by hand instead: the recovered
`warmup+10` (23,060) was cross-checked directly against
`get_strategy("kelly_regime_v4").warmup` (23,050) before trusting any
date in the table above, since a silent mismatch there would silently
shift every window's identity without erroring.

**Lesson.** This project's single largest headline number
(`kelly_regime_v4`'s return-per-risk edge, +20.8pp/+23.8pp median) is
now known to be roughly an order of magnitude smaller once the 2017–2020
bull is excluded — but it does not vanish, and the sign is right in both
markets. That is a materially more defensible, and more modest, claim
than either R-33's byproduct number or a naive rejection would have
produced, and it is the first claim in this project's SIZE lineage to
survive both a risk-match (R-33) and a period-match (this row) instead of
dissolving under one or the other.

**Next step.** The confirmed-but-thinned edge motivates, but does not by
itself supply, a strategy that captures more of it than v4 already does
by construction (v4 *is* the arm that produced this number — this row
validates the existing registered strategy's property, it does not design
a new one). That is a new question, opened here rather than pursued in
this row: can a SIZE-axis modification to the existing, already-validated
regime gate capture a larger share of the post-2021 (non-bull) edge
without reintroducing an exposure-level artifact of its own? Two
independent attempts follow immediately below (R-37).

### R-37 — two SIZE-axis attempts to harvest more of R-36's confirmed edge, run in parallel

**Idea.** R-36 confirmed a real but thinned return-per-unit-of-risk edge for
`kelly_regime_v4` over a matched passive hold outside the 2017–2020 bull
(median +5.0pp/+7.4pp per window, post-2021 windows). That row explicitly
declined to design a strategy around it — it validated an existing
property, it did not build one. This row opens that question with two
independent, disjoint-file attempts, run in parallel per ROUTINE.md's
rules: a conservative retune of v4's existing exposed constants, and a
novel change to the sizing formula itself. Both attack **SIZE**. Neither
touches detection (the 3-anchor latched vote is unchanged in both
branches), so neither duplicates R-01/R-02/R-03/R-28/R-34 (all of which
tried to replace or augment *detection*). Neither is `kelly_regime_v2`
(L-03, a convex exponent on the vote fraction, still riding one global
`target_vol`). Neither is R-06/R-07 (which swept anchor horizons, a
disjoint constructor axis). Both operated under an explicit, tighter
constraint than the project's usual holdout discipline: **no read of any
bar dated 2023-01-01 or later, for any purpose** — inner-train,
inner-validation, and the pre-2020 Bitfinex ETH/BTC falsification only.
Neither branch's worktree is offered as a Sharpe-based claim beyond what
is stated below; both are read as risk/return-shape and falsification
results, consistent with R-29's finding that this dataset cannot support
Sharpe-based claims.

**A build-process note, for anyone reproducing this row.** Both branches'
git worktrees were unexpectedly based on an older commit (`origin/main`,
17 commits behind this branch — pre-dating R-29 through R-36 and the
docs restructure), rather than the session's current HEAD. This was
caught and checked before trusting either report: the actual
`kelly_regime`/`kelly_regime_v3`/`kelly_regime_v4` strategy source files
were diffed byte-for-byte between each worktree and the current tree and
are **identical** (those files predate R-28), so the empirical sweeps
themselves are unaffected — only some of each branch's *contextual*
framing (references to ledger rows not present in their checked-out
`docs/LEDGER.md`) relied on the summary pasted directly into their task
prompts rather than the file. The operator independently re-ran the novel
branch's causality probe and ETH falsification from its own worktree
(exact match on every number, see below) as a partial substitute for the
skeptic step ROUTINE.md calls for.

#### Conservative branch — retune `target_vol`/`max_leverage`

**Mechanism.** v4's sizing constants (`target_vol=0.55, max_leverage=2.0`)
were hand-set years before this project's inner/holdout discipline
existed and have never been retuned against R-36's evidence specifically.
No new signal, no new mechanism — only the existing constructor scalars.

**Pre-registered failure modes** (named before running): (a) any
improvement sits inside the ±0.2 Sharpe noise floor; (b) the winner is a
plateau-free spike; (c) the winner is the L-04/R-28/R-31/R-32/R-33
exposure-level artifact rather than a genuinely better risk/return trade
at matched exposure.

**Configurations evaluated: 53** (50 on the primary `target_vol` × 10 ×
`max_leverage` × 5 inner-train/inner-validation grid, + 3 secondary
`anchor_span_days` configs around the naive winner).

**Result.** Naive best-Sharpe selection on inner-validation lands on
`target_vol=0.9, max_leverage=2.5` — spot Sharpe 0.296 vs defaults' 0.142,
but realized vol/notional **+51%/+52%** relative to defaults (failure mode
(c), exactly as pre-registered) and it does not transfer to futures
(Sharpe 0.085 vs 0.251, profit −13.0% vs +6.4%). The one candidate
surviving a genuine matched-exposure control (vol/notional within ~10% of
defaults on **both** markets simultaneously), `target_vol=0.60,
max_leverage=3.0`: Δ Sharpe **+0.044 spot / −0.026 futures** — inside the
±0.2 noise floor on both markets, i.e. a wash (failure mode (a)). Neither
candidate clears the plateau bar: the naive winner sits at the edge of the
searched grid still rising, and the matched-exposure candidate sits on a
monotonic slope, not a flat region (failure mode (b)). Causality probe:
**PASS** (byte-identical decisions before the cut; expected, since only
constructor scalars differ from unmodified v4 logic — verified directly
by the operator's independent diff of the strategy source, not re-run,
since a re-run on unmodified `prepare()`/`on_bar()` code adds no
information here). ETH falsification: **does not clear the pre-registered
bar** — worse on ETH spot (ΔSharpe −0.068, Δprofit −8.3pp), only
token-better on ETH futures (ΔSharpe +0.039); all four |ΔSharpe| ≤ 0.068,
inside the noise floor, so nothing here is individually distinguishable
from noise, but the direction does not clear its own bar either.

**Verdict: NEGATIVE.** No candidate clears both the noise floor and the
falsification bar; no holdout read recommended or spent.

#### Novel branch — per-vote-state Kelly fraction

**Mechanism.** Replace the one global `target_vol` with four separately,
causally-estimated scales, one per vote state (`frac` ∈ {0, ⅓, ⅔, 1}):
`scale_state = min(kelly_mult · μ_state/σ_state², max_leverage)`, μ/σ²
estimated with a time-halflife EWM (`pandas.ewm(halflife=…, times=…)`)
over per-bar log returns bucketed by the *previous* bar's vote state, then
forward-filled and shifted one more bar (mirroring `kelly_regime.py`'s own
`vol = ewm(...).shift(1)` causality pattern), gated to zero below 2,000
same-state observations. Literal Kelly (1956)/MacLean-Thorp-Ziemba (2010)
fractional Kelly, applied per-state instead of globally — the gap this
project's own R-28 aside flagged (its measured half-Kelly target, 0.49,
sat close to v4's hand-set 0.55, but as one number for every state).

**Pre-registered failure modes:** (a) per-state estimates too noisy to be
usable at 5-minute cadence (R-34's novel-branch failure mode, for a
different signal); (b) the winner is another exposure-level artifact via
raw leverage; (c) turnover rises because the state-conditional scale
changes more often than v3/v4's hysteresis-latched vol regime.

**Configurations evaluated: 46** (32 on the inner-train halflife ×
kelly_mult × stat_horizon grid, + 14 in the inner-validation
plateau/neighbourhood check).

**What the data actually shows, independent of any strategy result — the
empirical claim under this whole direction, and worth recording on its
own.** Measured μ_state/σ_state² on inner-validation (365-day halflife):
bear (frac=0) and the ⅓ state are both net-negative (annualized μ −62%/yr
and −101%/yr); the ⅔ and unanimous states are both strongly positive
(+174%/yr and +154%/yr) — **but non-monotone**: partial agreement (⅔)
carries a higher Kelly ratio than unanimous agreement (3/3). The states
genuinely differ, which is itself new information about this project's
own regime gate, even though (see below) it did not translate into a
better strategy.

**Result.** On inner-validation the winning config (halflife=365d,
kelly_mult=0.25) beats v4 on both markets (Sharpe 0.44 vs 0.14 spot, 0.59
vs 0.25 futures; DD 28.6% vs 33.2% spot; turnover *fell*, 24 vs 52
trades — failure mode (c) refuted) and `max_leverage` never binds across
the whole sweep, cleanly refuting failure mode (b) in its raw-leverage
form (though mean notional and realized vol move in *opposite* directions
relative to v4, both >10%, an honest partial echo of the R-33 exposure
critique in a new shape, reported as instructed). Causality probe:
**PASS**, and independently **re-run by the operator from the agent's own
worktree — reproduced exactly, every column differing by 0.000e+00 before
the cut.** The halflife/kelly_mult neighbourhood is **not a clean
plateau**: kelly_mult is non-monotone in Sharpe (0.49 → 0.20 → 0.44 → 0.34
across 0.10 → 0.35) and the halflife region (330–450d) is narrow with
270–300d clearly worse — a fitted peak, not the required plateau (failure
mode (a), in a data-hunger form rather than raw per-bar noise: the
per-state estimator needs enough calendar time to mature, which the
365-day halflife barely achieves inside the 2-year inner-validation
window itself).

**ETH falsification: FAILS outright — independently re-run by the
operator from the agent's own worktree, exact match to the reported
numbers.** The candidate is not merely worse on ETH; it is worse than v4
on the **BTC control run through the identical pipeline** (BTC spot final
$1,224 vs v4's $12,278, Sharpe 0.43 vs 1.86) and turns Sharpe-*negative*
on ETH (spot −0.70, futures −0.73, vs v4's +1.48/+1.25), with trading
essentially inert there (7 trades, vol 0.024–0.029) — the 3.3-year ETH
window is too short for a 365-day-halflife, 2,000-observation-minimum
per-state estimator to mature. That the BTC control also loses is the
more damning half of this: it means the inner-validation win looks fitted
to one specific 2021–2022 window rather than a transferable property,
consistent with the shaky plateau above.

**Verdict: NEGATIVE.** A genuinely non-duplicate mechanism, cleanly
refuting the raw-leverage-artifact failure mode and surfacing a real
(if non-monotone) fact about this project's own regime states — and
still failing its own pre-registered falsification test decisively
enough, on the BTC control as well as ETH, that no holdout read is
warranted.

**Combined accounting.** Configurations evaluated this row: 53 + 46 =
**99**. Project trials count before this row: 312 (unchanged by R-36,
which added 0). **Applies from here: 312 + 99 = 411.**

**Holdout counter: unchanged at ~159.** Neither branch read any bar dated
2023-01-01 or later, and both explicitly declined to recommend a holdout
read for their surviving-longest candidate.

**Lookahead checks.** Conservative branch: causality PASS via structural
argument (unmodified v4 logic, scalars only) plus the operator's source
diff, not independently re-executed (judged unnecessary — see above).
Novel branch: causality PASS, independently re-executed by the operator
from the agent's own worktree with an exact match. `pytest` was green in
both worktrees before and after (391 passed in both — the worktrees'
older base is 45 tests short of this branch's current 436, entirely
attributable to R-29's 27 inference tests + R-30's 18 display tests,
neither of which touches strategy code).

**Lesson.** Two structurally different, well-motivated, individually
non-duplicate attempts to convert R-36's confirmed (if thin) edge into a
better strategy both failed — one on a noise-floor/exposure-artifact
combination, the other on a falsification test it failed by a wide,
unambiguous margin rather than a close one. Taken together with R-34 and
R-35 (also two-branch, also both negative), this project's SIZE axis has
now absorbed four independent, non-duplicate parallel-round attempts
(eight branches total) since the L-01–L-04 family was registered, and
none has improved on it. The novel branch's one durable contribution is
independent of any strategy verdict: this project's own regime-vote
states carry measurably different, non-monotone forward drift/variance —
a fact worth keeping in mind for anyone who next touches the vote itself
(detection, not sizing), which remains largely unexplored territory
outside the failed R-01/R-02/R-03/R-28/R-34 attempts.

**Next step.** Neither branch reopens without new evidence (added to
section C below). The SIZE axis on this specific incumbent looks
increasingly exhausted by the same standard this project applies to the
holdout itself: four independent attempts, eight branches, zero
survivors. **B-06 (forward paper trading) remains the highest-value item
on merit**, still blocked on network access — the project's own repeated
conclusion, reached again from a fifth independent direction.

---

### R-38 — a formal, probability-calibrated sizing rule, tested against the same incumbent a fifth time

**Idea, one sentence.** `kelly_regime_v4` sizes with `scale =
min(target_vol/realized_vol, max_leverage)` — two hand-picked constants
with no probabilistic meaning; Busseti, Ryu & Boyd (2016, "Risk-Constrained
Kelly Gambling," *Journal of Investing* 25(3), arXiv:1603.06183) show that
an explicit drawdown constraint `Prob(min wealth < α) < β` reduces, via a
convex bound, to a CRRA/isoelastic-utility criterion with risk-aversion
`λ = ln(β)/ln(α)`, which combined with the classical Merton (1969/1971)
CRRA-optimal bet fraction `f* = μ/(λσ²)` gives a bet size *derived from a
stated drawdown tolerance* rather than searched for backtest score.

**Constraint attacked.** ERR (no error control anywhere in the signal
path — `target_vol`/`max_leverage` are unfalsifiable constants) and SIZE
(still an exposure-decision rule, this project's only axis that has ever
worked).

**Not a duplicate of.** L-01–L-04 (ad hoc constants vs. a
probability-calibrated formula); R-11 (Grossman–Zhou drawdown *cushion* —
a reactive floor triggered by realized drawdown — vs. a forward-looking
cap/formula derived from a return-distribution risk bound, active at all
times); R-28/R-31 (e-process hypothesis testing gates the
regime/*direction* signal — vs. this, which never touches the vote, only
the size given a fixed vote); R-34 (Bayesian posterior as a SIZE
dampener — a different signal source, the harsanyi_crowd belief margin —
vs. realized-return moments here); R-37 conservative (a pure
hyperparameter retune of the same two constants, no new constraint — vs.
a genuinely new, probability-interpretable quantity here); R-37 novel
(per-vote-state `μ_state/σ_state²` with an arbitrary searched
`kelly_mult` — vs. this novel branch's single continuous, non-state-
conditional `μ[t]/σ[t]²`, scaled by a `λ` *derived* from a stated
tolerance rather than searched). Both branches' docstrings restate this
distinction against their own file, matching this project's convention.

**Pre-registered before either branch ran** (fixed by the operator in
each branch's dispatch prompt, not moved afterward): (1) the mandatory
exposure-artifact check — R² of the candidate's exposure series against
a mean-notional-matched flat rescale of v4, with R² > 0.95 (R-34's own
threshold) meaning "this is the standard artifact, not a win"; (2) the
falsification test — does the inner-validation-selected candidate beat
`kelly_regime_v4` on Bitfinex ETH by more than a token margin, on both
spot and 5x futures, without being visibly worse than v4 on the BTC
control run through the identical pipeline; (3) a plateau requirement —
report the parameter neighbourhood, not just the winning cell; (4) a
named failure mode for each branch (conservative: `μ` too noisy at
5-minute cadence to leave `f_risk` meaningfully time-varying, collapsing
to "always binds" or "never binds"; novel: replacing volatility-only
sizing with a drift-over-variance formula makes exposure hypersensitive
to `μ`'s estimation noise, risking degenerate turnover or pinned
exposure); (5) a strict "no 2023+ bar, for any purpose" restriction,
independent of this project's own holdout-adjacent conventions for
causality probes and window resampling.

**Conservative branch — `experiments/kelly_regime_v7_ddcap.py`.** 24
configurations (α∈{0.5,0.6,0.7,0.8} × β∈{0.05,0.10} × halflife∈{30,90,180}d),
inner-train sweep, spot. Best inner-validation candidate (α=0.5, β=0.05,
hl=180d, λ=4.32): spot Sharpe **0.50** / futures **0.70** vs. v4's
**0.14** / **0.25** on the identical 2021–2022 window, at roughly 40% of
v4's mean notional. **Not a plateau**: adjacent cells are non-monotone in
λ — α=0.6/β=0.05/hl=180 (λ=5.86, between two good points) scores spot
Sharpe **−0.07**. **Exposure-artifact check: R²=0.20–0.21**, well under
the 0.95 bar — genuinely not the R-34-style flat-rescale artifact,
though the operator notes the ~0.39–0.40x mean-notional gap to v4 means
some of the DD/Sharpe gain in this specific window is still plausibly an
exposure-*level* effect even where the *shape* clears the test.
**Causality: PASS** (all six prepared columns, the order-decision probe,
and the equity path all show 0.000e+00 difference before a 3×/÷3 tamper
cut). **ETH falsification: FAILS decisively.** The candidate returns
≈11–12% of v4's balance on the **BTC control** alone (spot $1,457 vs.
$12,278; futures $1,353 vs. $25,681) before ETH is even read, and on ETH
it goes essentially inert — 1 trade over ~4 years, Sharpe −0.85 (spot) /
−0.91 (futures) vs. v4's +1.48/+1.25. The named failure mode (`μ`
saturating near/below zero, collapsing `f_risk` toward zero and staying
there) is exactly what happened.

**Novel branch — `experiments/kelly_regime_v7_crra.py`.** 32
configurations (α∈{0.5,0.6,0.7,0.8} × β∈{0.05,0.10} × halflife∈{7,15,30,60}d),
inner-train sweep, spot. Best inner-validation candidate (α=0.5, β=0.05,
hl=60d, λ=4.32): spot Sharpe **0.53** / futures **0.47** vs. v4's
**0.14**/**0.25**, on ~0.16 mean notional vs. v4's ~0.29. The tested
hypothesis (a *shorter*, more responsive `μ` window would help) reversed:
7-day halflives gave near-zero-to-negative inner-train Sharpe and the
highest turnover; the 60-day halflife (the longest tested, and the least
noisy) dominates 6 of 8 (α,β) cells. Cells at hl=60d cluster at Sharpe
0.47–0.53 both markets — a loose plateau — but α=0.6/β=0.05 (λ=5.86, the
paper's own worked example) dips to 0.28/0.32, and the whole surface sits
on 1–19 trades per config over two years, at the edge of the tested grid
(halflife never exceeded 60d). **Exposure-artifact check: R²=0.149** —
genuinely not a flat rescale (candidate flat 28% of bars vs. v4's 44%).
**Causality: PASS** (identical tamper-probe construction, 0.000e+00
throughout). **Turnover check**: no config pinned at cap or zero
(≤1.6% of bars); the named failure mode (drift-over-variance sizing
hypersensitive to `μ`'s noise) manifested as excessive turnover at short
halflives exactly as predicted, but going to hl=60d avoided degeneracy
without rescuing the direction. **ETH falsification: FAILS.** Not a
narrow miss — the candidate underperforms v4 on the **BTC control**
itself (37%/21% of v4's final balance, spot/futures) before ETH is read,
and on ETH it trails on both markets (final balance 37–74% of v4's,
Sharpe 0.13–0.69 lower). The inner-validation edge, built on a
bear/chop-dominated 2021–2022 window, does not survive a trending market
on either tested asset — the drift-based sizer systematically under-holds
through the trend it needed to capture.

**Verdict: NEGATIVE (both branches).** Both are honestly non-artifact
mechanisms — the cleanest exposure-artifact refutation this project has
recorded on either axis (R²=0.15–0.21, well below both the 0.95 rule and
R-34's own 0.997 comparator) — and both still lose their pre-registered
falsification test the same diagnostic way: underperforming
`kelly_regime_v4` on the **BTC control window itself**, not narrowly on
ETH. That is a stronger, more specific failure than any of R-34/R-35/R-37:
it says a causally-estimated drift (`μ`) term brought into the sizing
formula — whether as a soft cap (conservative) or as the primary driver
(novel) — systematically gives up upside through a trend, worse than v4's
own scheme, which carries no drift term at all and lets the discrete
regime vote do 100% of the directional work. The reusable lesson: at this
project's cadence and cost structure, a *continuous* drift estimate is
not merely too noisy to trade on its own (already known from R-34's
novel branch) — folding it into the sizing formula of an otherwise
sound, vote-gated vol-targeting strategy actively hurts it, in two
structurally different implementations, on two different assets.

**Combined accounting.** Configurations evaluated this row: 24 + 32 =
**56**. Project trials count before this row: 411 (R-37's cumulative).
**Applies from here: 411 + 56 = 467.**

**Holdout counter: unchanged at ~159.** Neither branch read any bar
dated 2023-01-01 or later — both were explicitly restricted to
inner-train/inner-validation/pre-2020 ETH/BTC only, and neither
recommended a holdout read for its surviving candidate.

**Lookahead checks.** Both branches' causality probes were read and
accepted as reported (two-opposite-tampers construction, identical in
spirit to `kelly_regime_v6_state_kelly.py`'s own, adapted to each
branch's prepared columns); not independently re-executed by the
operator in this round.

**Network note, incidental to this round.** A connectivity check run
alongside this session found Bitstamp (`bitstamp.net`) and Coinbase
(`api.exchange.coinbase.com`) now return HTTP 200 from this session's
network policy — Binance still returns 451. The 08-17 finding that "every
exchange endpoint is blocked" (which is what marked B-02/B-03/B-06/B-07/B-08
`BLOCKED (network)` below) may therefore be stale for at least Bitstamp,
which is what `docs/LIVE.md`'s `BitstampSpot` adapter already targets.
This was a two-endpoint ping/ticker check only, not a full `tradebot
fetch` or paper-trading-recorder attempt, and network policy can vary
by session — the next session with spare capacity should verify properly
before treating B-06 as unblocked.

**Next step.** Neither branch reopens without new evidence (added to
section C below). This is the fifth independent, non-duplicate
parallel-round attempt (ten branches total) to improve `kelly_regime_v4`
on its own SIZE axis since L-01–L-04 were registered, and none has
survived both the exposure-artifact check and a falsification test.
**B-06 (forward paper trading) remains the highest-value item on merit**,
and — pending the connectivity re-check above — may be less blocked than
the ledger currently states.

---

### R-39 pre-registration — the network re-check R-38 asked for, done properly, and what it unblocks

**What changed since R-38.** R-38's connectivity note was a two-endpoint
ping only. This session did the "proper verification (an actual
`tradebot fetch` or a first paper-trading-recorder connection attempt,
not just a ping)" R-38 asked the next session to do, against six
endpoints: Binance spot REST (`451`, still blocked), Bitstamp, Coinbase
(both `200`, confirming R-38), and three venues not checked before —
Kraken spot (`200`), Kraken Futures' historical-funding endpoint (`200`,
live data returned), and Deribit's public funding-rate-history endpoint
(`200`, live data returned). OKX's public funding-rate-history endpoint
also returned `200`. A real historical data pull was then completed
end-to-end against Deribit (below), which is a materially stronger check
than a ticker ping. **Binance remains the one venue blocked**, which
matters because it is the source of the only real BTC funding data this
project has ever had (`data/btcusdt_perp_funding_8h.csv.gz`,
2020–2023).

**Idea, in one sentence.** Fetch BTC perpetual funding from a reachable
non-Binance venue (Deribit) to cover 2024–2026, the exact gap identified
by B-02 as "the single cheapest item that could change a decision," and
spend it on the two backlog items that named it as their own blocker:
finish R-35's underpowered funding-decile-gate holdout test (B-05, closed
"pending B-02") with real multi-year power instead of one funding-covered
holdout year, and build B-03's delta-neutral funding-harvest strategy for
the first time, extended through the exact 2024–2025 window the
literature (BitMEX 2025Q3/2026Q2 derivatives reports; He, Manela, Ross &
von Wachter 2024, SSRN 4301150, "Fundamentals of Perpetual Futures") says
the carry premium crowded and compressed.

**Constraint attacked.** COST — same axis as R-35/L-05/L-06, the one
channel this project has repeatedly found underpowered rather than wrong.
Not SIZE: this round deliberately does **not** touch `kelly_regime_v4`'s
sizing formula, which is the axis five independent rounds (R-28/31, R-32,
R-34, R-35's novel branch, R-37, R-38 — ten branches) have now exhausted
without a surviving result. The standing diagnosis's own pattern —
"every SIZE-axis attempt this project's own incumbent hasn't already
captured has failed" — is the reason to attack COST's *data* limitation
instead of SIZE's *modelling* limitation this round.

**Not a duplicate of.** R-35's conservative branch (`funding_gate_decile`,
decile fixed at 0.90) is reused **verbatim, byte-for-byte** — this is not
a new mechanism, it is the same pre-registered rule read against a longer
holdout, which is explicitly what its own ledger row said would resolve
the "every interval containing zero" result. B-03 (delta-neutral harvest)
has never been implemented as code in this repository — the $+16.2%/yr
"harvesting the premium" figure in VALIDATION.md was a one-off compounding
calculation, and no `experiments/` file for it exists to duplicate.

**Simulable here, with one caveat named up front.** The Deribit funding
data (`scripts/fetch_deribit_funding.py`, `data/btcusdt_deribit_perp_funding_8h.csv.gz`,
7,264 8h buckets, 2020-01-01 → 2026-08-19) is a **different instrument's**
funding rate, not a continuation of the Binance series: Deribit charges
funding continuously (hourly `interest_1h`, here summed into UTC-aligned
[00:00,08:00,16:00) buckets) rather than Binance's discrete 8-hourly
settlement, and the on-overlap comparison (2020–2023, both real,
1,459 overlapping days) shows correlation **r=0.69** (daily-summed rate)
but an **unstable** cross-venue level ratio — 0.64x (2020), 1.24x (2021),
0.21x (2022), 0.34x (2023) — so it is **not rescaled** to "look like"
Binance (a fixed scale factor was tested and rejected for exactly this
instability; see `src/tradebot/data.py::load_funding_deribit` docstring).
`load_funding_extended()` therefore concatenates real Binance
(2020–2023) with Deribit **only for the genuine post-2023 gap**, tagging
every settlement with its source — never blending the two inside the
overlap, never letting Deribit override a real Binance value. Read every
2024-2026 number below as "this venue's realistic funding cost/premium,
same underlying economic mechanism, not Binance's own number" — the same
caveat this project already carries for spot-as-perp-proxy.

**What would make each variant fail — named now.**
(a) **Conservative.** R-35 already found the gate's *inner-validation*
edge survives an exposure-artifact check cleanly; the open question is
purely statistical power. If the extended 2023–2026 holdout still returns
a confidence interval containing zero, or a negative point estimate, that
is a real negative this time — not "still underpowered," since the
holdout is now ~3.6x longer than R-35's one year — and B-05 closes for
good rather than staying open pending more data.
(b) **Novel.** The literature is explicit that funding richness declined
sharply in 2024 and further in 2025 as the trade crowded (BitMEX's own
2025Q3/2026Q2 reports; this project's own Deribit pull shows the same
shape independently — annualized mean funding **10.3% (2024) → 5.4%
(2025) → 1.6% (2026 YTD)**, computed below).

> **Correction, made by the operator before any branch result was read,
> and recorded rather than silently fixed.** The three figures in the
> paragraph above were first committed as 30.8% / 16.2% / 4.8% — exactly
> **3x too high**. The error was in the annualization, not the data: the
> series was resampled to daily sums (which already aggregates the three
> 8-hourly settlements in a day) and then multiplied by 3 × 365 as though
> each daily figure were still a single settlement. The corrected numbers
> above are confirmed by two independent routes that now agree to 0.01pp
> (per-settlement mean × 3 × 365, and daily-sum mean × 365). Nothing
> downstream depended on the wrong values — they were cited as context
> for the *direction* of the decline, not used as a decision threshold,
> and the decline is real either way — but a 3x arithmetic error in a
> committed pre-registration is exactly the kind of thing this file's
> culture exists to surface rather than quietly overwrite.

If net-of-cost carry return
in 2024–2026 is materially worse than 2020–2023's measured +14.6%/yr
(0.10% tier) — including possibly negative — that is the falsification
this branch is built to test, and it should be reported as exactly that,
not reframed as a partial win.

**Method.** Two independent, unregistered branches, each on a disjoint
file, neither committing — the operator (this session) merges and
records both after reading each report in full, per ROUTINE.md's
parallelism rules. Neither branch touches BTC OHLCV price data outside
the routine's existing splits; funding coverage, not price, is what is
being extended.

- **Conservative — `experiments/funding_gate_decile_extended.py`.**
  Literally `experiments/funding_gate_decile.py`'s decision logic
  (v4's vote/scale untouched, `target = 0` where the trailing
  rolling-window funding percentile ≥ the fixed 0.90 decile, same swept
  lookback set {90, 180, 365, expanding}, same 3-settlement causal EWM
  smoothing) repointed at `load_funding_extended()` in place of
  `load_funding()`. No new parameter, no new mechanism — only the funding
  series' length changes.
- **Novel — `experiments/funding_harvest_carry.py`.** Long spot BTC /
  short an equal-notional BTC perp, rebalanced quarterly to stay
  delta-neutral (matching the original R-15 methodology described in
  VALIDATION.md), collecting the extended real funding stream. Report at
  minimum: full-period and per-sub-period (2020–2023 real-Binance vs
  2024–2026 Deribit-extension) annualized carry, at both the 0.10% and
  0.40% taker tiers on both legs, and the worst realized 30-day drawdown
  of the funding stream in each sub-period — the same cells VALIDATION.md
  already reports for 2020–2023, so the two are directly comparable.

**Pre-registered decision rule.**
- Conservative: promote the gate as a registered strategy-modification
  candidate only if the **full 2023-01-01 → 2026-08-19 holdout** (paired
  stationary block bootstrap, R-29/R-30 convention, funding charged as a
  first-class futures cost per the `funding_study.py` convention) gives a
  Δ log growth or Δ Sharpe 95% CI that **excludes zero in the gate's
  favor**, drawdown not worse, and survival of the 0.40% taker tier.
  Anything else is NEGATIVE and closes B-05 permanently (not "pending
  more data" — this round is the more data).
- Novel: register `funding_harvest_carry` as a new strategy only if it
  clears the standing promotion bar (beats `buy_and_hold` OOS after real
  costs, or is a genuine drawdown/tail improvement outside the ±0.2
  Sharpe floor) **on the 2024–2026 sub-period specifically**, not just on
  the already-known-good 2020–2023 window — since the 2020–2023 result
  is not new evidence and re-quoting it would not be a holdout test of
  anything. If the 2024–2026 carry is thin, negative, or fails to clear
  costs, report that plainly: it is the literature's own prediction
  landing on this project's own data, which is exactly what this branch
  was built to check.

**Holdout note.** The conservative branch's holdout read is
pre-registered as **one** consultation against the full extended window
(not per-config) — it evaluates the same frozen configuration R-35 already
fixed, so there is no new selection happening on the holdout, only a
longer read of an existing rule. The novel branch's 2024–2026 read is
**not** a consultation of the BTC-price 2023+ holdout in the sense the
rest of this ledger tracks (`run_period`/`ev` on OHLCV) — it reads only
the funding-rate series, a different signal on a different clock — but is
counted explicitly below for transparency since it is, in spirit, reading
data the project had not yet looked at.

### R-39 results — both branches NEGATIVE; the extended data answered its own question decisively

**Method actually followed.** Two independent agent sessions, each on a
disjoint unregistered file, neither committing — reports at
`experiments/reports/funding_gate_decile_extended_report.md` and
`experiments/reports/funding_harvest_carry_report.md`. Both reports were
read in full before this row was written. The conservative branch's
decision-cell numbers were **independently re-derived by the operator**
using a different code path than the branch's own harness (mirroring
`scripts/funding_study.py`'s existing, already-tested `_period()` helper
rather than the branch's hand-rolled `run_period_funding`) and a fresh
call into `tradebot.inference.paired_bootstrap`: point estimates matched
to the cent (v4 $3,867.85 vs claimed $3,868; gate $1,617.35 vs claimed
$1,617) and the bootstrap reproduced closely (Δ log growth −0.87
[−1.67, −0.15] vs claimed −0.872 [−1.701, −0.166]; Δ Sharpe −0.58
[−1.12, −0.04] vs claimed −0.582 [−1.149, −0.042] — the small differences
are resample-path noise between two different bootstrap call sites at the
same seed, not a disagreement about the finding). This is the
"independent skeptic" step ROUTINE.md's parallelism section asks for.

**Conservative branch (`funding_gate_decile_extended`, B-05 reopened).**
**NEGATIVE, decisively, in the wrong direction.** The frozen `w=180`
configuration from R-35 — same 0.90 decile, same swept lookback set, same
smoothing, zero new parameters — read against the full 2023-01-01 →
2026-08-19 holdout (now 100% funding-covered vs R-35's 28%, 72% of it
genuinely new Deribit-sourced data) returns Δ log growth **−0.872
[−1.701, −0.166]** and Δ Sharpe **−0.582 [−1.149, −0.042]** against v4,
both excluding zero *against* the gate, plus a **worse** drawdown
(+12.2pp) and outright failure at the 0.40% taker tier (Sharpe −0.38 vs
v4's +0.83). The sub-period split is the interpretive centre: 2023 alone
(what R-35 actually saw) reproduces R-35's own ledger row to the dollar
and is a null (Δ log growth −0.120 [−0.430, +0.139]) — confirming R-35
was not wrong about anything it measured. **All** of the new negative is
in 2024–2026: Δ log growth −0.746 [−1.466, −0.097]. Descriptively, R-16's
premise (rich funding forecasts weak forward returns) held with the right
sign on inner-validation (−0.22pp) and **inverted** on 2024-2026
(+2.27pp, wrong direction) — the gate stood flat through the strongest
part of a multi-year bull leg specifically because funding was rich
*because* the bull was working, which is the opposite of what the
2020–2022 discovery sample looked like. A post-hoc check (not
pre-registered, run to rule out an obvious objection) confirms this is
not the Binance/Deribit venue splice: a pure-Deribit series with no
cross-venue join at all gives the same negative, marginally stronger.
**B-05 closes permanently, per its own pre-registration's stated rule.**
Configs evaluated: **72** (61 distinct cells).

**Novel branch (`funding_harvest_carry`, B-03 built for the first
time).** **NEGATIVE.** The delta-neutral carry trade, coded as a real
backtest for the first time in this project, reproduces every cell of
`docs/VALIDATION.md`'s existing 2020-2023 figure exactly, then fails the
pre-registered 2024-2026 test: net of 0.10% costs it returns +16.7%
against `buy_and_hold`'s +49.1% (return bar fails decisively), and the
drawdown/tail bar is **voided rather than scored** — this repository has
no perp price series, so the trade's basis is identically zero by
construction and its near-zero measured volatility is an artifact of the
model missing two of its three real risk sources (basis risk at
entry/exit, and liquidation risk on the short leg, which the branch
measured reaching **1.57×–2.34× account equity** in unrealized loss
between rebalances), not evidence the trade is safe. The pre-registered
falsification (net-of-cost 2024-2026 materially worse than 2020-2023's
+14.6%/yr) is met: +4.98%/yr risk-matched. Configs evaluated: **19
distinct specs, 58 configuration-evaluations**. **B-03 closes as tested
and rejected for the current era** (not ruled out on principle — see
below).

**Four findings worth carrying forward past the NEGATIVE verdicts:**

1. **The pre-registration's own Deribit annualization was 3× too high**
   (corrected in-place above once found; both branches independently
   caught and confirmed the same error). The shape it described was
   right, the level was not.
2. **Most of the apparent funding-premium "collapse" is a venue effect,
   not a market effect.** Running the carry-harvest analysis on Deribit
   alone across 2020-2026 (no cross-venue splice at all) gives +7.88%/yr
   → +6.58%/yr — a real decline, but a modest one whose bootstrap CIs
   overlap almost entirely ([+2.91,+13.44] vs [+3.95,+9.42]). Binance's
   2020-2023 funding ran roughly **2×** Deribit's over the identical
   calendar period. Any future citation of this project's own
   +16.2%/yr / Sharpe-6.45-ish carry figures should carry that caveat —
   Deribit's contemporaneous number is roughly half as good, with twice
   the negative-settlement frequency, and neither venue's number was ever
   measured against basis risk.
3. **B-03's real blocker was never the funding data — it is the missing
   perp price series.** This session confirmed Deribit's public API also
   serves complete 5-minute `BTC-PERPETUAL` OHLCV back to 2020
   (`get_tradingview_chart_data`, spot-checked at five dates 2020–2026,
   288/288 or 289/289 bars every time). A future session building that
   series would let a delta-neutral backtest measure real entry/exit
   basis risk for the first time — the actual reason B-03 was flagged
   "measured entirely in the good years" — rather than merely extending
   the funding leg again. Added to the backlog below as **B-15**.
4. **The holdout was read far more than pre-registered, and the ledger's
   counter reflects the honest number, not the authorized one.** The
   conservative branch's pre-registration authorized one consultation
   (the frozen `w=180` full-window read); it actually touched 61 distinct
   holdout cells (§10 of its report), all of them diagnostics run *after*
   the decision cell had already returned a significant negative — none
   could have changed the verdict in the gate's favour, but the count
   belongs in the table below at its real size, per this file's standing
   practice of naming the discrepancy rather than rounding it down.

**Next step.** Both directions this session pursued now have a clean
NEGATIVE. The COST axis (funding-as-cost, funding-as-gate) has now been
tried as a cost measurement (R-14), a forecaster (R-16), a SIZE-axis gate
(R-35, reopened and closed for good by R-39), an EV-band sizing input
(R-35 novel, borderline-negative), and a carry-harvest trade in its own
right (R-39 novel) — five distinct treatments of the same underlying
signal, none surviving to promotion. **B-06 (forward paper trading)
remains the highest-value item on merit.** Network access is now
confirmed genuinely wider than believed (Deribit, Kraken Futures, and —
per R-38's still-unconfirmed note — Bitstamp/Coinbase for spot), which is
the actual blocker on B-06 clearing; a future session should attempt the
real connection rather than another ping.

---

### R-40 — bagging R-07's own validated plateau, instead of shipping one point on it

**Idea, one sentence.** R-07 (already in this ledger) swept nine
anchor-ladder base periods in the 18–28 day range and found the whole
region is a validated **plateau** — every variant cut drawdown to
35–39%, Sharpe spread 1.52–1.60 sat inside the ±0.2 noise floor —  yet
`kelly_regime_v4` ships exactly one point on that plateau (20/40/80) and
treats it as certain. Bootstrap aggregating (Breiman 1996, *Machine
Learning* 24(2)) says averaging an unstable-but-unbiased estimate across
resamples reduces variance without moving its expectation; the "resample"
here is not data, it is the a-priori choice of which already-validated
ladder base to use. Separately, Baker & McHale (2013, "Optimal Betting
Under Parameter Uncertainty: Improving the Kelly Criterion," *Decision
Analysis* 10(3)) and Sukhov (2025, "Bayesian Kelly Criterion with
Parameter Uncertainty," SSRN) show the Kelly fraction should shrink
continuously with estimation uncertainty rather than being spent at full
confidence on a point estimate.

**Constraint attacked.** ERR — specifically, no error control on the
anchor-ladder hyperparameter itself, which R-07 already showed sits on a
plateau rather than a peak.

**Not a duplicate of.** R-06/R-07 (measured individual points on the
plateau, never averaged them); `kelly_regime_v2` (shrinks on disagreement
among the 3 anchors *within* one fixed ladder via `vote_gamma` — a
different axis from averaging *across* ladders); R-34 (Bayesian
type-belief posterior, a different signal source entirely); R-37
(per-vote-state Kelly fraction replacing `target_vol`) and R-38
(risk-constrained drawdown cap / CRRA fraction, both using realized-return
moments) — neither touches the vote/ladder axis at all, both leave v4's
single ladder untouched and change the vol-targeting formula instead.
This round is the first to touch the ladder-*choice* itself rather than
what happens after a ladder is chosen.

**Pre-registered before either branch ran** (fixed in each branch's
dispatch prompt): (1) a fixed, not-fitted ensemble membership — doubling
ladders with base days spanning R-07's own 18–28d region, plus one
below-plateau negative control (14d) predicted to underperform; (2) the
mandatory exposure-artifact check (R² > 0.95 = artifact, R-34's
threshold); (3) a frac-correlation check against v4's own single-ladder
vote (corr > 0.98 = "collapses to v4, no effect"); (4) the standard
ETH-vs-BTC-control falsification, worded as: fails if the candidate is
not comparable to v4 on ETH, **or** is visibly worse on ETH than on the
BTC control run through the identical pipeline; (5) for the novel branch
only, two additional checks: that `κ=0` reduces numerically exactly to
the conservative branch's mechanism, and correlation against
`kelly_regime_v2` (a value near 1.0 would mean re-deriving an
already-negative result through new arithmetic); (6) the same strict
"no 2023+ bar, for any purpose" restriction as R-37/R-38.

**Conservative branch — `experiments/kelly_regime_v8_ladder_bag.py`.**
Four fixed ensemble definitions (`full6`={18,20,22,24,26,28} primary,
`coarse3`={18,23,28}, `edges2`={18,28}, `negcontrol`={14,21,28}).
Inner-validation, both markets, all four beat v4 (spot $998→$1,095–1,148,
Sharpe 0.14→0.30–0.38; futures $1,064→$1,082–1,180, Sharpe 0.25→0.28–0.42)
on 18–35 trades vs v4's 52 — the gain concentrates in fewer, better-timed
trades through the bear/chop-heavy 2021–2022 window. **Frac-correlation
check:** 0.974–0.983 against v4's own vote — high, `full6` sits right at
the "collapses" boundary, but none formally clears 0.98. **Exposure-
artifact check:** R²=0.916–0.936 across all four — consistently *below*
the 0.95 bar, but closer to it than R-38's clean 0.15–0.21 refutation.
**Causality: PASS** (target/`_frac_bagged`/`_scale` bit-identical before
a 3×/÷3 tamper cut, order-decision probe and equity path both exact).
**Inner-train tells a different story**: `full6`'s futures max drawdown
*worsens* to 45.6% against v4's 35.3% over 2017–2020, and the
`negcontrol` set — predicted in advance to underperform as a
below-plateau check — instead scores competitively on inner-train
(futures Sharpe 2.29 vs v4's 2.28) while showing the *worst* BTC-control
degradation of all four sets below. **ETH-vs-BTC falsification, re-run
independently by the operator and confirmed to the dollar:** on the BTC
control (Bitfinex 2016–2019, whole file) every candidate loses to v4,
sharply on futures — `full6` $19,343 vs v4's $25,681 (75%), `coarse3` 67%,
`edges2` 55%, `negcontrol` **52%** (DD 50.6% vs v4's 32.1%, its worst
cell) — and more mildly on spot (80–84%). On ETH the *ratios* hold or
improve (`coarse3`/`negcontrol` actually beat v4 outright on ETH
futures), so the literal pre-registered clause — "visibly worse on ETH
than the BTC control" — does **not** trigger. But absolute
underperformance against v4 on the BTC control itself, particularly on
futures, is real and not small.

**Novel branch — `experiments/kelly_regime_v8_uncertainty_shrink.py`.**
Same 6-ladder ensemble, `disagree[t] = std_k(binary vote_k[t])`,
`shrink[t] = max(floor, 1/(1+κ·disagree²))`, `frac_final = frac_bagged ·
shrink`. 8 configurations (κ∈{0,2,8,20} × floor∈{0.2,0.4}). **κ=0
sanity check: PASS** — numerically exact reduction to the conservative
branch's own mechanism (confirmed independently by the operator: both
branches' `select` commands return $1,095/$1,180 for this cell, to the
dollar). **Every κ>0 configuration underperforms the κ=0 baseline**, on
inner-train and inner-validation, both markets, 6 of 6 — the disagreement
signal is real (nonzero on 13.1% of bars, concentrated around ladder-latch
flips) but adds no value once found. **Exposure-artifact check:**
R²=0.862–0.867 — not an artifact. **`kelly_regime_v2` correlation:**
0.890 — meaningfully below R-34's ~0.997 near-duplicate signature, so not
a re-derivation of v2's convexity. **Causality: PASS** (all six prepared
columns, orders, and equity bit-identical before the tamper cut).
**ETH-vs-BTC falsification** (representative candidate κ=8, floor=0.2):
BTC control 56–77% of v4's balance (worst on futures), ETH 82–87% — again
the ratio *improves* going BTC→ETH, so the literal clause does not
trigger, but the candidate trails v4 in every absolute cell on both
assets.

**Verdict: NEGATIVE (both branches), on a standard tighter than either
branch's own literal pre-registered wording — stated here explicitly, per
ROUTINE.md's rule that a moved goalpost must be named.** Neither branch
technically fails its own falsification clause as written: relative
performance does not degrade going from the BTC control to ETH in either
branch. But this project has, in R-37 and R-38, already established a
sharper and more specific diagnostic than that clause captures:
underperforming `kelly_regime_v4` on the **BTC control window itself**,
before ETH is even read, is evidence that an inner-validation edge built
on the bear/chop-dominated 2021–2022 window is a window-fitting artifact
that reverses in a trending market — regardless of what happens on the
second asset. Both of this round's branches hit exactly that signature,
most severely on futures. Applying that established, *stricter* bar here
is a legitimate use of precedent, not an after-the-fact search for a
reason to reject a result that otherwise looked like a win — it moves the
decision in the conservative direction (reject), never the promotional
one, and it is recorded so a future session does not read the literal
"PASS" on this round's own ETH clause as license to reopen it without
addressing the BTC-control finding directly. The reusable lesson: R-07's
plateau is real on the metric it was measured on (drawdown, on the
2021–2022 window it was measured on), but treating that plateau as a
free ensembling opportunity does not transfer to a trending market any
better than any other SIZE-axis modification this project has tried —
the plateau's flatness is itself apparently local to the regime it was
discovered in, an implication R-07 never tested and this round now has.

**Combined accounting.** Configurations evaluated this row: 4 + 8 =
**12**. Project trials count before this row: 597 (R-39's cumulative:
467 + 72 + 58). **Applies from here: 597 + 12 = 609.**

**Holdout counter: unchanged at ~221.** Neither branch read any bar
dated 2023-01-01 or later — both were restricted to
inner-train/inner-validation/pre-2020 ETH/BTC only, matching the R-37/
R-38/R-39 convention, and neither branch's own report recommended a
holdout read for its candidate.

**Lookahead checks.** Both branches' causality probes were **independently
re-executed by the operator**, not merely read and accepted — `python
experiments/kelly_regime_v8_ladder_bag.py select` and `... eth` were
re-run directly and their output compared line-for-line against the
branch's own report (exact match on every figure quoted above); the
`kelly_regime_v8_uncertainty_shrink.py` `select` output was likewise
re-run and its κ=0 row confirmed identical to the ladder-bag branch's
`full6` row, which is itself the numerical cross-check the two branches'
pre-registration asked for.

**Next step.** This is the fourth independent, non-duplicate
parallel-round attempt (eight branches: R-34, R-37, R-38, R-40) to
improve `kelly_regime_v4` on its own vote/SIZE axis since L-01–L-04 were
registered, and — like all three before it — a branch that is neither an
exposure-level artifact nor a re-derivation of an existing negative still
loses to the incumbent on a trending control before a second asset is
even read. Nothing here reopens without a new mechanism; added to section
C below. **B-06 (forward paper trading) remains the highest-value item on
merit**, and, per R-38/R-39's network findings, may be closer to
reachable than the ledger has assumed for most of this project's
history — the natural next thing to actually attempt, rather than a
seventh variation on kelly_regime_v4's own sizing formula.

---

## C. Ruled out — do not re-try without new evidence

| what | why | ref |
|---|---|---|
| More indicators / more ML on 5m bars | 25 strategies and two research rounds; every pure predictor lost to fees. Attacks none of the four constraints. | A, R-05 |
| Recovering order flow from OHLCV | BVC/VPIN proxies are price transforms. Four strategies, four losses. | L-14, L-15, L-16, L-12 |
| Tuning turnover to fit a fee tier | 28 of 32 in-sample, 0 of 28 out-of-sample. | R-12 |
| Higher leverage as a fix for fees | Fees are charged on notional; leverage multiplies cost and return together. Changes the risk profile, not the sign. | R-13 |
| Sentiment / social media | A lagged function of price — not orthogonal information — and revision-prone. | — |
| Higher-frequency execution | Turnover is the enemy at every fee tier available. | R-12, R-13 |
| Elliott waves | Unfalsifiable as practised; its testable kernel already implemented. | R-18 |
| Market making, AMM/LVR | Plausibly real — the loss-versus-rebalancing decomposition of AMM LP returns (Milionis, Moallemi, Roughgarden & Zhang) is genuinely quantitative — but **not simulable** on bar-close fills with no order book; it would need a queue model first. Ruled out on what can be checked, not on merit. | L-24 |
| Options / volatility risk premium | Same — no options data, no way to validate here. | — |
| `harsanyi_crowd`'s Bayesian posterior as a SIZE input on `kelly_regime_v4` | L-12's own stated hypothesis, tested both bounded (conservative) and unbounded (novel) — one is an exposure-level artifact, the other is genuinely independent of the vote but too noisy at its native cadence to pay 5-minute-bar costs on either axis. | R-34 |
| Retuning `kelly_regime_v4`'s `target_vol`/`max_leverage` against R-36's evidence | 53 configurations; the naive winner is the standard exposure-level artifact, and the one matched-exposure survivor nets a Sharpe delta inside the ±0.2 noise floor on both markets and fails the ETH check. | R-37 (conservative) |
| Per-vote-state Kelly fraction (`μ_state/σ_state²`) replacing v4's single global `target_vol` | 46 configurations; cleanly refutes the raw-leverage-artifact failure mode and surfaces real, non-monotone state-conditional drift/variance, but fails its own pre-registered ETH falsification decisively — worse than v4 on the BTC control too, indicating the inner-validation win was fitted to one window. | R-37 (novel) |
| Risk-constrained Kelly (Busseti/Ryu/Boyd 2016) drawdown-probability cap layered on v4's unchanged vote+scale | 24 configurations; cleanly refutes the exposure-artifact explanation (R²=0.20) but the (α,β) neighbourhood is not a plateau and it fails ETH decisively, losing to v4 on the BTC control itself (≈11–12% of its balance) before ETH is even read. | R-38 (conservative) |
| CRRA/Merton drift-over-variance fraction (`μ/(λσ²)`, `λ` from a stated drawdown tolerance) replacing v4's vol-only scale | 32 configurations; cleanly refutes the exposure-artifact explanation (R²=0.15) and finds a loose plateau at the longest tested halflife, but fails the identical ETH falsification the same way — worse than v4 on the BTC control (21–37% of its balance) before ETH is read; a continuous drift estimate systematically under-holds through a trend. | R-38 (novel) |
| Binary top-decile-funding flat gate layered on `kelly_regime_v4`, read against a 3.6-year fully-funding-covered holdout (was 1 year in R-35) | 72 configurations; the one-year R-35 result did not merely stay underpowered, it reversed — Δ log growth −0.87 [−1.70,−0.17], Δ Sharpe −0.58 [−1.15,−0.04], both excluding zero against the gate, worse drawdown despite 27% less exposure, fails 0.40% tier outright. Not a venue-splice artifact (pure-Deribit gives the same result). Specifically ruled out: a *binary* percentile-threshold flat gate on this signal/cadence/instrument — not "funding is useless" (R-16's descriptive relationship is now known to be regime-dependent, which is itself new information). | R-39 (conservative), reopens B-05 from R-35 and closes it |
| Delta-neutral spot-long/perp-short funding-harvest carry trade, extended through 2024-2026 | 19 specs / 58 evaluations; fails the return bar decisively (+16.7% vs buy_and_hold's +49.1% net of 0.10% costs, 2024-2026) and the drawdown/tail bar is voided (this repo's missing perp price series makes basis risk and the trade's real safety unmeasurable, not merely uncosted). Ruled out for the current era, not on principle — see B-15. | R-39 (novel) |
| Plain bagging (unweighted average) of R-07's validated 18-28d anchor-ladder plateau, replacing v4's single (20,40,80) ladder | 4 configurations; beats v4 on every inner-validation cell (not an artifact, R²=0.92-0.94) but loses to v4 on the pre-2020 BTC falsification control itself, down to 52% of its balance on futures — the same bear/chop-window-fitting signature that sank R-37/R-38, even though the literal "worse on ETH than BTC" clause does not trigger. | R-40 (conservative) |
| Baker-McHale/Bayesian-Kelly-style shrink of the bagged ladder vote by real-time cross-ladder disagreement | 8 configurations; genuinely non-duplicate (not an artifact at R²=0.86-0.87, not a re-derivation of `kelly_regime_v2` at corr=0.89) but never beats its own no-shrink (κ=0) baseline in 6 of 6 tested configurations, and inherits the conservative branch's BTC-control underperformance (56% of v4's futures balance) unchanged. | R-40 (novel) |
| Bounded never-increase brake on `kelly_regime_v4`'s exposure from real Deribit spot/perp basis magnitude (extreme premium or discount) | 18 configurations (+1 identity check); genuinely non-duplicate signal (r≈0.06-0.12 vs the already-tested funding rate) but the standard exposure-level artifact in every configuration (R²=0.981-0.999) *and* the R-37/R-38/R-40 train-loses/validation-wins signature (36/36 train cells lose, only 14/36 validation cells win) — real basis blowouts are too rare (0.09% of bars) to move a multi-year aggregate past the artifact bar. | R-41 (conservative) |
| Real-time basis as an early confirming vote to shorten `kelly_regime_v4`'s anchor-latch delay (timing axis, not magnitude) | Step-2 lead-lag study is a clean null (basis-confirmed hit rate scatters around the ~51% base rate against a block-bootstrap null, ~0-day median lead — contemporaneous with price, not leading it); 12 configurations built and tested anyway beat v4 in every cell but sit inside the ±0.2 Sharpe noise floor and are R²=0.977 collinear with v4's own exposure (the artifact bar). | R-41 (novel) |

---

## D. Backlog (ranked)

Re-ranked 08-17, after two rounds ran the same day. **R-26 dispatched a
parallel round at five of these items and measured nothing** — every agent
was blocked by a tooling fault — so it left the backlog exactly as it
found it, which was the right call. **R-28 then executed B-01
single-threaded and carried it to a verdict.** Read the two together: the
null round is why several items below still look untouched, and it is not
evidence about any of them.

**Re-ranked again 08-17 after R-29.** B-04 is done and its answer reorders
everything below it: the comparison table's ordering is mostly not
distinguishable from noise, no strategy's Sharpe survives deflation
out-of-sample, and the holdout is now exhausted (counter ~88). That makes
**B-06 the highest-value item in the backlog on merit** — forward paper
trading is the only uncontaminated evidence this project can still
generate — and it is blocked on network access, which is the single thing
worth asking the operator for. Everything still actionable from inside a
session is now either display work (B-12) or a further re-reading of a
dataset that has stopped answering.

**Re-ranked 08-18 after R-30.** B-12 is done, and it was the last item
that could be finished without either new data or a new idea. The order
below is unchanged; what changed is that the two remaining
computation-only items (**B-11**, then **B-05**) are now the whole
actionable list, and both re-read a dataset R-29 showed has stopped
answering Sharpe-shaped questions. R-30's growth intervals sharpen that:
on the criterion the table ranks by, **nothing in it is distinguishable
from buy-and-hold**. A session that finds B-11 and B-05 unpersuasive
should say so and spend itself on **B-06** instead — writing the recorder
against a mock feed so that only the network policy, and not also the
code, stands between this project and its first uncontaminated evidence.

**Re-ranked again 08-18 after R-31.** B-11 is done, and it opened
**B-13**, which goes to the top. R-31 removed a claim the project had been
leaning on — R-28's ETH drawdown replication — by showing it was a
statement about exposure rather than about the mechanism; the same
argument applies unchanged to L-04's "regime-gated sizing cuts drawdown",
which is also measured against a fully-invested benchmark. That makes
B-13 the cheapest experiment left that could change what this project
believes about itself, and it needs no new data. B-05 follows it as the
other computation-only item. **B-06 (forward paper trading) remains the
highest-value item on merit** and the only source of uncontaminated
evidence, and it is still blocked on network access; a session that
finds B-13 and B-05 unpersuasive should spend itself writing that
recorder against a mock feed, so only the network policy — and not also
the code — stands between this project and evidence it has not already
spent.

**Reconciled 08-18 after R-32.** Two sessions ran B-11 in parallel that
day without knowing about each other, and both are recorded: R-31 is the
primary result, R-32 the independent replication plus the ungated control
R-31 did not run. The order above is unchanged — B-13 stays on top, and
R-32's ungated arm gives an unfavourable preview of it — but two numbers
are: the day's trials count is the **total across both branches** (36 + 33
= **69**, so the project applies 103 + 69 = **172**), and the holdout
counter is **~124**, not the ~112 either branch would report alone. Both
branches were scheduled onto the same backlog row by accident, which is
the cost ROUTINE.md's parallelism section describes, paid in holdout
consultations.

**Re-ranked 08-19 after R-33.** B-13 is done and it removed the claim this
project has led with since L-04: at matched risk 88–92% of "regime-gated
sizing cuts drawdown" is "regime-gated sizing holds half the notional".
Three claims have now died the same death — R-28's e-process drawdown cut
(killed by R-31), R-32's gate comparison, and now L-04's headline — and in
all three cases the mechanism was an exposure level. That pattern is
itself the most reusable thing in this ledger: **before believing any
comparison here, check whether the two arms carry the same risk.**

What R-33 leaves is a *different* claim that kept appearing in cells
nobody pre-registered: at equal realized volatility v4 out-returns a
constant-exposure hold by a median +20.8pp (spot) / +23.8pp (futures) per
window, in 82% and 90% of 40 windows, in all four ETH/BTC falsification
cells, and in every holdout cell valid or void. None of that can be
claimed, because it was not the question asked. Pre-registering it is
**B-14**, and it goes to the top: the harness exists, the matching is
already solved to 0.5% per window, and it is the only live hypothesis in
this project that has *survived* a risk-matching round rather than
dissolving in one. **B-06 (forward paper trading) remains the
highest-value item on merit** and is still blocked on network access.

**Unchanged 08-19 after R-34.** A parallel two-branch round tested
L-12's own stated hypothesis (the `harsanyi_crowd` posterior as a SIZE
input rather than a DIRECTION input) off-backlog, since it was a cheap,
well-justified, self-contained question rather than a claim on the
ranked list. Both branches were NEGATIVE for clean, different reasons
(see R-34) and neither touched the holdout. The order below is
unaffected: **B-14 stays top**, **B-06 stays the highest-value item on
merit** and still blocked on network access.

**Re-ranked 08-19 after R-35.** B-05 is done: a parallel two-branch round
(one conservative literal gate, one novel funding-adjusted EV band)
finally executed the R-14/R-16 funding finding on the SIZE axis. The
conservative branch cleared every inner-validation and falsification
check, including the specific exposure-artifact failure mode its own
pre-registration predicted for it — then lost on the one funding-covered
holdout year available, with every interval containing zero but every
point estimate negative. **B-05 is closed, not ruled out** — it reopens
directly once B-02 supplies more funding-covered years, which would
turn one holdout year into several and is the direct fix for the power
problem this round hit. The order below is otherwise unaffected:
**B-14 stays the top of the ranked backlog** (untouched by this row —
still the only hypothesis in this project that has survived a
risk-matching round rather than dissolving in one) and **B-06 (forward
paper trading) remains the highest-value item on merit**, still blocked
on network access. Between B-14 and B-06, a future session finding B-14
unpersuasive should, as prior rounds have said, spend itself writing the
paper-trading recorder against a mock feed so that only the network
policy stands between this project and its first uncontaminated
evidence.

**Re-ranked 08-19 after R-36 and R-37.** B-14 is done: R-36 pre-registered
and confirmed the return-per-risk edge R-33 surfaced by accident, and
found it survives outside the 2017–2020 bull but shrinks roughly 10x once
that period is excluded. Off-backlog, the same session immediately spent
a two-branch parallel round (R-37) asking whether that confirmed edge
could be captured better than v4 already does — a conservative
hyperparameter retune and a novel per-vote-state Kelly sizer — and both
came back NEGATIVE, for two different, well-substantiated reasons (a
noise-floor/exposure-artifact combination, and a falsification test
failed decisively enough to indicate overfitting to one window). Both are
added to section C. **B-14 moves to done** and nothing replaces it at the
top: this project has now run four independent, non-duplicate
parallel-round attempts (R-28/R-31 alone, R-32, R-34, R-35, R-37 — eight
branches total since L-01–L-04 were registered) to improve on the SIZE
axis of its own incumbent, and none has survived both a noise-floor check
and a falsification test. **B-06 (forward paper trading) is now not just
the highest-value item on merit but the only genuinely open, well-motivated
item left that does not re-read a dataset this project has independently
concluded, five separate times, has stopped answering the questions asked
of it.** It remains blocked on network access. A session with nothing else
to do should write the recorder against a mock feed now, so only the
network policy stands between this project and its first uncontaminated
evidence, exactly as every re-ranking since R-29 has said.

**Unchanged 08-19 after R-38.** A fifth independent, non-duplicate
parallel-round attempt on `kelly_regime_v4`'s SIZE axis — a formal,
probability-calibrated sizing rule from Busseti/Ryu/Boyd's (2016)
risk-constrained Kelly gambling, run as a conservative cap and a novel
full replacement of the sizing formula. Both branches did something no
prior round fully managed: cleanly rule out the standard exposure-level
artifact (R²=0.15–0.21 vs. the 0.95 bar). Both still failed the identical
pre-registered ETH falsification test, and by the same diagnostic
signature — underperforming v4 on the **BTC control window itself**, not
narrowly on ETH — indicating a continuous drift estimate brought into the
sizing formula systematically under-holds through a trend. The order
below is unaffected: **B-06 stays the highest-value item on merit**. One
new fact changes its status, though not its ranking: a two-endpoint
connectivity check run alongside this round found Bitstamp and Coinbase
now return HTTP 200 (Binance still 451), where the 08-17 finding below
recorded all four exchanges as blocked. That finding is what marks
B-02/B-03/B-06/B-07/B-08 `BLOCKED (network)` — it may be stale for
Bitstamp specifically, which is the venue `docs/LIVE.md`'s `BitstampSpot`
adapter already targets, and is worth a proper verification (an actual
`tradebot fetch` or a first paper-trading-recorder connection attempt,
not just a ping) before the next session assumes either status.

**Re-ranked 08-19 after R-39 — the proper verification the row above
asked for.** It found Binance still blocked, but Deribit and Kraken
Futures reachable with live data, and completed an actual multi-year
historical pull (not a ping). That closed **B-02** (partially — see its
row for the venue caveat) and let two backlog items run to a verdict for
the first time: **B-05 reopened and closed for good** (the one-year
underpowered result from R-35 reversed on the full 3.6-year holdout, a
decisive negative rather than "needs more data"), and **B-03 ran as real
code for the first time and closed NEGATIVE for the current era** — not
for lack of data, but because this project's missing perp price series
makes the trade's dominant risk (basis) structurally unmeasurable. That
finding opens **B-15** (build a real perp series; confirmed available
from the same Deribit endpoint) as a more useful next step than any
further funding-data work. The order is otherwise unchanged: **B-06
(forward paper trading) remains the highest-value item on merit**, and
R-39's own network re-check is itself indirect evidence it may be closer
to reachable than the ledger has been assuming — B-06 is the natural next
item to attempt a real connection against, not just a ping.

**Unchanged 08-19 after R-40.** A sixth independent, non-duplicate
parallel round tested whether bagging R-07's already-validated 18-28d
anchor-ladder plateau (conservative: plain average) or shrinking it by
real-time cross-ladder disagreement (novel: a Baker-McHale/Bayesian-Kelly
style formula) could improve on `kelly_regime_v4`'s single frozen ladder.
Both branches beat v4 cleanly on inner-validation and neither is the
standard exposure-level artifact — the closest either has come to a
believable win by that pair of tests — but both reproduce R-37/R-38's
exact failure signature: losing to v4 on the pre-2020 BTC falsification
control itself (worst on futures, down to 52-56% of v4's balance) before
ETH is even read, indicating the inner-validation win is again fitted to
the bear/chop-dominated 2021-2022 window rather than a generalizable
mechanism. The order below is unaffected: **B-06 (forward paper trading)
remains the highest-value item on merit**, and this is now the fourth
independent parallel round (eight branches: R-34, R-37, R-38, R-40) to
fail on `kelly_regime_v4`'s own vote/SIZE axis — a future session with
spare capacity should attempt B-06's real connection rather than a ninth
variation on the incumbent's sizing formula.

**Re-ranked 08-19 after R-41.** A connectivity re-check at the start of
this session found Deribit, Kraken, Bitstamp and Coinbase all now
reachable (only Binance still 451s) — a material change from the 08-17
finding that marked B-02/B-03/B-06/B-07/B-08/B-15 `BLOCKED (network)`.
That closed **B-15**: a real Deribit BTC/ETH-PERPETUAL price series is
now committed, the first genuinely independent second price series this
project has had, ending the "spot (perp proxy)" situation for the assets
it covers. A same-day parallel round spent that new data as a SIZE input
on `kelly_regime_v4` — attacking INFO for the first time, rather than
re-deriving from the existing single price series like the six branches
before it — and both branches (conservative: bounded basis-magnitude
brake; novel: basis as an early confirming vote) came back NEGATIVE, for
two different, well-diagnosed reasons neither of which was data quality
(see R-41, and section C). **This is the fifth independent parallel round
(ten branches total: R-34, R-37, R-38, R-40, R-41) to fail on
`kelly_regime_v4`'s own SIZE axis**, and the first to fail despite
attacking a genuinely different constraint than the prior four — which
raises the prior that the axis itself, not just each individual signal,
is close to exhausted for this strategy family. **B-06 (forward paper
trading) remains the highest-value item on merit, and is now also the
most actionable**: this session's own connectivity re-check found
Bitstamp reachable, which is the exact venue `docs/LIVE.md`'s
`BitstampSpot` adapter already targets. B-07 (on-chain features) and B-08
(second bear, second asset, ETH 2020-2026) are very likely unblocked by
the same connectivity change but have not yet been attempted — a
network-access re-check, not just an assumption from the 08-17 finding,
is worth doing explicitly before either is next attempted. B-03's
funding-harvest carry trade can also now be re-run with B-15's real basis
in place of its previously-unmeasurable, identically-zero basis risk —
lower priority than B-06, but no longer blocked either.

**Re-ranked 08-19 after R-42.** The network re-check the row above asked
for was run explicitly: CoinMetrics' free community API (a genuinely new
endpoint, not one of the four already-checked venues) is reachable, which
closes **B-07** — real daily BTC/ETH on-chain data (MVRV, active
addresses, supply, 2010/2015→present) is now committed, giving this
project its first non-price data channel. A same-day parallel round spent
it as a SIZE input on `kelly_regime_v4` (conservative: asymmetric
high-MVRV brake; novel: an MVRV-vs-price agreement/disagreement gate,
built after its own pre-registered lead-lag study came back a clean null)
and both branches came back NEGATIVE — see R-42 and section C. **This is
the sixth independent parallel round (twelve branches total: R-34, R-37,
R-38, R-40, R-41, R-42) to fail on `kelly_regime_v4`'s own SIZE axis**,
and the second in a row to fail despite attacking a genuinely orthogonal
data source rather than re-deriving from OHLCV — which is stronger
evidence than any single round that the axis itself, not the choice of
signal, is the limiting factor. **B-06 (forward paper trading) remains
the highest-value item on merit and the most actionable item on the
backlog** — every venue this project has checked (Bitstamp, Deribit,
Kraken, Coinbase, CoinMetrics) is now reachable, and R-08's deflated-Sharpe
finding that this dataset is close to exhausted for Sharpe-shaped
questions has only gotten stronger with each subsequent SIZE-axis round. A
session with spare capacity should write the recorder against a mock feed
and then attempt the real connection, rather than a thirteenth variation
on the incumbent's sizing formula. B-08 (ETH 2020-2026, a second bear on a
second asset) remains open and likely unblocked, and now also has real
ETH on-chain data available alongside it if a future round wants it.

Two things changed the order. R-28 answered B-01. And a connectivity check
found that **every exchange endpoint is blocked by the network policy
these sessions run under** — Binance, Bitstamp, Kraken and Coinbase all
refuse at the proxy (403 on CONNECT). Five backlog items were ranked on
the assumption that "one data fetch" was available from inside a session.
It is not, so they are marked `BLOCKED (network)`: they need the operator
to widen the policy or to commit the data to the repo. Note this is a
*different* fault from the one that produced R-26's null round, which was
a permission handler stripping tool parameters and has since cleared. What
remains actionable is computation on the data already here.

| ID | item | attacks | status | note |
|---|---|---|---|---|
| ~~B-01~~ | ~~E-process regime detection with unified Kelly sizing~~ | ERR, N≈3 | **DONE → R-28**, qualified by R-31 | NEGATIVE on the promotion bar. It read as the strongest risk result in the project — 0 of 40 windows deeper than the incumbent — until B-11 compared the two at equal risk and found that number was about exposure, not about the gate. (R-26's null round listed this as untried; R-28 is the round that actually ran it.) |
| ~~B-04~~ | ~~Purged CV, deflated Sharpe, block-bootstrap CIs on every headline~~ | ERR | **DONE → R-29** | The guess was right: 10 of 96 adjacent pairs distinguishable, none of them in the top eight. Also closes R-25. `tradebot.inference` is now a permanent module with 27 tests; step 4 of the routine can be mechanical from here. |
| ~~B-12~~ | ~~Put the intervals *in* the comparison table~~ | ERR | **DONE → R-30** | The table now carries Δ growth and Δ max drawdown against `buy_and_hold`, each with a 95% interval, and a strategy without a measured interval fails CI. The by-product is the sharpest number in the project: **0 of 24 strategies are distinguishably better than holding on the criterion the table ranks by**, and v4's +0.044 edge is [−2.60, +2.85]. |
| ~~B-11~~ | ~~Matched-risk frontier: e-process gate vs latched vote at equal realized volatility~~ | ERR, SIZE | **DONE → R-31** | Answered, negatively and usefully. At equal realized volatility the two gates are indistinguishable on the BTC holdout (all 8 intervals contain zero, sign unstable), three of four cells fail a pre-registered validity gate, and on ETH the e-process gate loses on **both** axes — so R-28's ETH drawdown replication was an artifact of carrying 2.4x less risk. The 0.27x exposure was the whole finding. Also answered in parallel by **R-32**, which adds the arm neither the backlog row nor R-31 asked for: **no gate at all**, which loses to both gates at matched risk in every inner-split cell and in 80–90% of 40 paired windows. |
| ~~B-14~~ | ~~Return per unit of risk against a constant exposure — the claim R-33 kept measuring by accident~~ | SIZE, ERR | **DONE → R-36** | Confirmed, thinned. Pooled across the same 40 windows R-33 used, D1 passes on both markets (win-rate 95% CI excludes 50%: [67.2%,92.7%] spot, [76.3%,97.2%] futures). The pre-registered falsification test (does it survive outside the 2017–2020 bull) also survives on both markets, but the median advantage shrinks ~10x once windows starting before 2021 are excluded (+68.9pp→+5.0pp spot, +97.2pp→+7.4pp futures), and the post-2021 subsample's own CI still contains 50% on spot at n=22. Off-backlog follow-up **R-37** (two branches, both NEGATIVE) asked whether a strategy could be built to capture more of this edge — see section C. |
| ~~B-05~~ | ~~Funding as a gate on the existing strategy (stand flat in the top decile)~~ | COST | **DONE → R-35, reopened and CLOSED FOR GOOD → R-39** | R-35: NEGATIVE, closed pending B-02 (underpowered, one funding-covered holdout year, interval containing zero). R-39 reopened it with the full 2020-2026 funding series and got a decisive, opposite-sign NEGATIVE: Δ log growth −0.872 [−1.701, −0.166] against the gate on the fully-covered 3.6-year holdout, worse drawdown despite less exposure, fails the 0.40% tier. Not underpowered this time — closes permanently per its own pre-registration. |
| ~~B-02~~ | ~~Extend the funding series through 2026~~ | COST | **DONE (partial) → R-39** | Binance itself is still unreachable, but Deribit's public API is not, and a full historical pull succeeded: `data/btcusdt_deribit_perp_funding_8h.csv.gz`, 2020-01→2026-08. **Caveat that matters**: Deribit is a different instrument (continuous funding vs Binance's discrete 8h settlement), correlates with Binance at only r=0.69 on the 2020-2023 overlap with an unstable year-to-year level ratio (0.21×-1.24×) — `load_funding_extended()` therefore never rescales or blends the two, only concatenates Deribit onto the genuine post-2023 gap. Good enough to reopen and definitively close B-05, and to run B-03 for the first time; not a literal continuation of "the Binance series." |
| ~~B-03~~ | ~~Funding harvest (delta-neutral spot vs short perp)~~ | COST | **DONE → R-39, NEGATIVE for the current era** | Implemented as real code for the first time (`experiments/funding_harvest_carry.py`) and extended through 2024-2026: fails the return bar decisively (+16.7% vs `buy_and_hold`'s +49.1% net of 0.10% costs) and the drawdown/tail bar is voided rather than passed, because this repo's missing perp price series makes basis risk structurally unmeasurable — the trade's near-zero measured volatility is an artifact of the model, not evidence of safety. Reopens only via **B-15**, not via more funding data (which this round already supplied). |
| **B-06** | Forward paper-trading recorder | N≈3 | **BLOCKED (network)** | Rose in importance and fell in feasibility on the same day. R-28's deflated Sharpe says this dataset is close to exhausted, which is the argument for starting the only uncontaminated record this project can still generate — but the recorder needs a live price feed, and every venue is blocked. First thing to unblock if the policy is widened. |
| ~~B-07~~ | ~~On-chain features, sign-corrected~~ | INFO | **DONE → R-42** | CoinMetrics' free community API is reachable; real BTC/ETH MVRV (2010/2015→present) is committed. The trap this row itself named — on-chain signals secretly acting as a volatility timer, which R-08 showed hurts this strategy — is exactly what sank the conservative branch (MVRV-Z correlates up to +0.58 with forward realized volatility). The novel branch's own pre-registered duplicate test found MVRV's *level* too collinear with a price/MA(730d) ratio (R²=0.964) to count as orthogonal information at the mechanism level, even though the raw series is genuinely non-price. Both branches NEGATIVE; see R-42. The 141-predictor on-chain study this row cited (4 of 141 beating a random walk at all horizons) was about *direction* prediction — this round tested MVRV only as a *SIZE* input, so that base rate does not directly transfer, but the outcome rhymes with it. |
| **B-08** | Second bear, second asset, different period (ETH 2020–2026) | N≈3 | BLOCKED (network) | R-17 shares the 2018 bear with the main dataset, so the two tests are not independent; the committed Bitfinex ETH file stops in 2019 and the rest is not fetchable from here. |
| **B-09** | Conformal prediction / adaptive conformal by betting (adaptive conformal inference under distribution shift; conformal prediction with change points, NeurIPS 2025; adaptive conformal inference by betting, 2024) | ERR | LOW | Was "mostly subsumed by B-01" — now demoted further by R-28's result: the binding problem is not that trust is miscalibrated but that correctly-calibrated trust is *low*, and conformal would say the same thing more slowly. |
| ~~B-13~~ | ~~Matched-risk benchmark: `kelly_regime_v4` against a **de-levered** `buy_and_hold` at equal realized volatility~~ | ERR, SIZE | **DONE → R-33** | Answered, and it cost the project its headline. At genuinely equal risk (40 windows, matched inside each window to 0.5%) v4's median drawdown advantage falls from −24.5pp to **−2.9pp** on spot and from −70.7pp to **−5.5pp** on futures; on the holdout five of six frozen cells fail the risk match and the valid one gives −14.18pp [−22.68, +13.48]. R-31's suspicion was right: the −41.1pp is mostly the exposure level. The consolation, and it is a real one, is that the *return* comparison at matched risk goes v4's way everywhere and survives the ETH test that killed R-28 — see **B-14**. Original framing kept below for the record. Opened by R-31, and it points the same knife at this project's own headline. Every drawdown claim here — L-04's "regime-gated sizing cuts drawdown", R-17's ETH replication, R-29's −41.1pp [−54.8, −18.4] — compares a strategy holding roughly half the notional against a **fully-invested** benchmark. R-31 showed that precise mismatch manufactured a mechanism finding for the e-process gate that vanished at equal risk. The experiment is one afternoon: add a constant-exposure hold at scale `c` to `experiments/matched_risk.py`, solve `c` on inner-validation so its realized volatility equals v4's, and re-run the paired bootstrap. Needs no new data, no fetch, and the harness already exists. Pre-register the answer both ways — a hold de-levered to 0.5x is *not* obviously a weaker benchmark, and if the drawdown gap survives it, that is the strongest result this project has ever had. |
| **B-10** | Deterministic Elliott wave counter | — | LOW | Only as a documented negative result, per R-18. ZigZag pivots, mechanical impulse/corrective rules, no discretion. About a day, converts an unfalsifiable debate into a table row. |
| ~~B-15~~ | ~~Build a real perp price series (Deribit `BTC-PERPETUAL`, 5m OHLCV) alongside the existing spot series~~ | ERR, COST, INFO | **DONE → R-41** | Built: real BTC-PERPETUAL (2018-08-14→) and ETH-PERPETUAL (2019-03-14→) 5m OHLCV, plus a matching Coinbase ETH spot series, all committed. `tradebot.data.load_deribit_perp_price()`/`compute_basis()` give a genuine, non-proxied spot/perp basis for the first time — used as a `kelly_regime_v4` SIZE input in R-41 (both branches NEGATIVE, for reasons unrelated to data quality). Available for B-03's re-run (a real basis-risk term for the funding-harvest carry trade) and for a future SIZE-axis round with a different exploitation, per R-41's own recommendation — a short event-triggered override rather than a continuous ramp, or a replacement rather than a multiplier of v4's exposure. Not wired into `CANONICAL["perp"]`, so no existing comparison-table number changed. |

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
| 08-19 | ~221 | R-42: **+0** on top of R-41's ~221. Both branches (`kelly_regime_v10_mvrv_brake`, `kelly_regime_v10_mvrv_lead`) were explicitly restricted to inner-train (2017-2020)/inner-validation (2021-22)/ETH-falsification (2019-03→2022-12, or 2019-03→2023-01 for the novel branch, both pre-holdout) and neither read a single 2023+ bar, by design. The operator's independent re-verification (`eth` on both branches, `duplicate` on the novel branch) reused the same pre-holdout data, not the holdout. |
| 08-19 | ~221 | R-41: **+0** on top of R-40's ~221. Both branches (`kelly_regime_v9_basis_brake`, `kelly_regime_v9_basis_lead`) were explicitly restricted to inner-train-with-basis (2018-08-14→2020-12-31) and inner-validation (2021-01-01→2022-12-31) and neither read a single 2023+ bar, by design; neither reached ETH falsification (both authors recommended against it given their own inner-validation diagnostics, and the operator agreed rather than spend the newly-built ETH data). The operator's independent re-verification (`artifact`/`fallback`/`causality`/`exposure`/`leadlag`) reused the same pre-2023 data, not the holdout. |
| 08-19 | ~221 | R-40: **+0** on top of R-39's ~221. Both branches (`kelly_regime_v8_ladder_bag`, `kelly_regime_v8_uncertainty_shrink`) were explicitly restricted to inner-train/inner-validation/pre-2020 ETH+BTC only and neither read a single 2023+ bar, by design; the operator's independent re-verification of both branches' reported numbers reused the same inner-validation and pre-2020 falsification data, not the holdout. |
| 08-19 | ~221 | R-39: **+62** on top of R-38's ~159 — the conservative branch's own honest count (§10 of its report): 61 distinct 2023+ holdout cells, not the 1 its pre-registration authorized (the extra 60 are diagnostics — neighbourhood, cost tiers, exposure-matched control, sub-period split, venue-splice robustness — run *after* the pre-registered decision cell had already returned a significant negative; none could have changed the verdict in the gate's favour, but this file's practice is to record the real number). Plus **+1** for the operator's independent skeptic re-derivation of the decision cell via a separate code path. The novel branch (`funding_harvest_carry`) reads only the funding-rate series against `buy_and_hold`/`kelly_regime_v4` reference runs over 2024-2026 — a period the BTC-price holdout convention already treats as fair game once funding covers it — and is not counted separately here. |
| 08-19 | ~159 | R-38: **+0** on top of R-36/R-37's ~159. Both branches (`kelly_regime_v7_ddcap`, `kelly_regime_v7_crra`) were explicitly restricted to inner-train/inner-validation/pre-2020 ETH+BTC only and neither read a single 2023+ bar, by design. |
| 08-19 | ~159 | R-36 and R-37: **+0** on top of R-35's ~159. R-36 reused R-33's existing `windows.csv` (seed=42, computed once) and only recovered calendar dates from the RNG sequence — no new backtest, and the 40-window resample does not count against the holdout by this project's own established convention. Both R-37 branches were explicitly restricted to inner-train/inner-validation/pre-2020-ETH only and neither read a single 2023+ bar, by design. |
| 08-19 | ~159 | R-35: +7 on top of R-33's ~152 — one pre-registered configuration (`funding_gate_decile`, w=180) read once, restricted to the 2023-01-01..2023-12-31 funding-covered slice rather than the full 2023-2026 holdout (a deliberate, pre-registered scope limit, not an oversight): spot funding-free, futures funding-free, futures funding-charged, each a paired v4-vs-gate read (6), plus one `buy_and_hold` context run (1). The parallel novel branch and the conservative branch's `w=90`/`w=365`/expanding configurations never read it, per the pre-registered "ask fewer questions" economy. |
| 08-16 | ~30 | Backfilled estimate. Every OOS figure in sections A and B came from reading the 2023+ holdout; it has never been pristine. Deflate program-level claims accordingly, and treat forward paper trading (B-06) as the only uncontaminated evidence still obtainable. |
| 08-17 | ~88 | R-29: every registered strategy (25) on both markets, as a fresh 2023+ account, to attach an interval to each. No selection was made on any of it and the decision rules were committed first — but 50 consultations is 50 consultations. The finding that matters: at ~88 program-level reads, and with `kelly_regime_v4`'s holdout Sharpe needing a **6.2-year** track record to clear a 103-trial bar it has 3.6 years of, **no Sharpe-based claim from this dataset is supportable any more**. Judge on drawdown, which still replicates, and treat B-06 (forward paper trading) as the only remaining source of evidence. |
| 08-18 | ~88 | R-30: **unchanged, and the reasoning is the point.** The bootstrap was re-run over the holdout to recover the log-growth interval, which looks like 50 fresh consultations and is not one: R-29 drew those exact resamples and computed that exact interval object, then persisted only two of its three fields. Every overlapping number came back bit-identical, which is the evidence. No new question was asked; a field was recovered from an answer already given. Read it as ~138 if you disagree — R-29's conclusion that no Sharpe-based claim from this dataset is supportable holds either way. |
| 08-18 | ~112 | R-31: 12 matched-and-reference runs across two markets, 6 re-runs at the 0.40% taker tier, 6 with funding charged on futures. The ETH/BTC falsification cells and the 40-window resample do not read the 2023+ BTC holdout (the R-19/R-28 convention). Every configuration was frozen on inner-validation and the decision rule, the validity gate and the predictions were committed one commit ahead of the first holdout read — `git log` records it. Nothing here is offered as a Sharpe-based claim; the round's finding is that at matched risk there is no difference to claim. |
| 08-17 | ~38 | R-28: three configurations × two markets, plus two cost re-runs. The ETH falsification test and the 40-window resample do not read the 2023+ BTC holdout. At 24 trials in a single session the deflated Sharpe was already 0.859; at ~38 program-level consultations, treat any Sharpe-based claim from this dataset as unsupportable and judge on drawdown, which is the property that has repeatedly replicated. |
| 08-19 | ~152 | R-33: +28 on top of R-32's ~124 — 10 frozen holdout runs across two markets, 8 for the descriptive on-holdout re-match and its solver, 10 cost re-runs (5 at the 0.40% taker tier, 5 with funding charged). The ETH/BTC falsification cells and the 40-window resample do not read the 2023+ BTC holdout. The re-match is the only part of this round that reads the holdout for a quantity it was not pre-registered to read; it is labelled in-sample everywhere it appears and supports no decision. Consistent with R-29: nothing here is offered as a Sharpe-based claim, and the round's finding — that 88–92% of a drawdown gap was an exposure gap — is measured on 40 resampled windows rather than on this holdout, precisely because the holdout has stopped being able to settle anything. |
| 08-18 | ~124 | R-32: +12 on top of R-31's ~112 (3 frozen arms × 2 markets, 3 spot fee-tier re-runs, 3 funding-charged futures re-runs). The number that matters is not the increment but why it exists: **two sessions were scheduled onto the same backlog row on the same day and each spent the holdout on it independently**. Neither branch did anything wrong — both pre-registered, both froze before reading — but the day cost ~36 consultations and 69 trials for one question, and the project applies 103 + 69 = **172** trials from here. If parallel sessions are going to run, ROUTINE.md's rule that the trials count is the total across branches is the thing that keeps the arithmetic honest; this is the first time it has actually been needed. |
