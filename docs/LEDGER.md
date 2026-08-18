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
| R-32 | The ungated control, and an independent second reading of B-11 | 08-18 | A parallel session ran the same backlog row the same day from the same base commit. Same design as R-31 (one sizer, gate interchangeable, exposure scaled by a scalar) plus a **third arm with no gate at all**; 33 configurations, 132 backtests, multipliers frozen on inner-validation | **Agrees with R-31 wherever the two overlap** — gates indistinguishable at matched risk, R-28's 0-of-40 inverted (deeper in 60%/62%), its fee advantage inverted, P1 failed — from an independent implementation, and its own holdout cells are **void** under R-31's validity rule (cap binds on 41%/36%/21% of spot bars; a 29% volatility gap on futures). What it adds: at matched risk the **ungated** arm is below both gated arms at every risk level in all four inner-split cells and loses 80–90% of 40 paired windows. **The gate is worth more than the choice of gate.** | **NEGATIVE** — and the parallel-branch report the routine requires |

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
| **B-05** | Funding as a gate on the existing strategy (stand flat in the top decile) | COST | **NEXT** | Actionable: uses the committed 2020–2023 funding file, no fetch. The low-turnover way to use R-16, and it directly targets the adverse timing in R-14. Higher-turnover standalone reversal use is where strategies go to die (R-12). |
| **B-02** | Extend the funding series through 2026 | COST | **BLOCKED (network)** | Still the single cheapest item that could change a decision — the literature says the carry premium broke in 2024–25 and our data stops in 2023 — but Binance is unreachable from these sessions. Needs the operator. |
| **B-03** | Funding harvest (delta-neutral spot vs short perp) | COST | BLOCKED on B-02 | +16.2%/yr with a −1.31% worst month is a risk profile nothing else here approaches — measured entirely in the good years. Unmodelled: basis risk, short-leg liquidation, exchange/custody risk, borrow cost. |
| **B-06** | Forward paper-trading recorder | N≈3 | **BLOCKED (network)** | Rose in importance and fell in feasibility on the same day. R-28's deflated Sharpe says this dataset is close to exhausted, which is the argument for starting the only uncontaminated record this project can still generate — but the recorder needs a live price feed, and every venue is blocked. First thing to unblock if the policy is widened. |
| **B-07** | On-chain features, sign-corrected | INFO | BLOCKED (network) | The only genuinely orthogonal channel. Enter with the base rate in mind: a 141-predictor study found 67 worked in-sample, 29 survived out-of-sample, **4 beat a random walk at all horizons**. Note the trap: on-chain flows predict *volatility*, and R-08 showed better volatility input makes this strategy worse. **Fix the sign first.** |
| **B-08** | Second bear, second asset, different period (ETH 2020–2026) | N≈3 | BLOCKED (network) | R-17 shares the 2018 bear with the main dataset, so the two tests are not independent; the committed Bitfinex ETH file stops in 2019 and the rest is not fetchable from here. |
| **B-09** | Conformal prediction / adaptive conformal by betting (adaptive conformal inference under distribution shift; conformal prediction with change points, NeurIPS 2025; adaptive conformal inference by betting, 2024) | ERR | LOW | Was "mostly subsumed by B-01" — now demoted further by R-28's result: the binding problem is not that trust is miscalibrated but that correctly-calibrated trust is *low*, and conformal would say the same thing more slowly. |
| **B-13** | Matched-risk benchmark: `kelly_regime_v4` against a **de-levered** `buy_and_hold` at equal realized volatility | ERR, SIZE | **NEXT** | Opened by R-31, and it points the same knife at this project's own headline. Every drawdown claim here — L-04's "regime-gated sizing cuts drawdown", R-17's ETH replication, R-29's −41.1pp [−54.8, −18.4] — compares a strategy holding roughly half the notional against a **fully-invested** benchmark. R-31 showed that precise mismatch manufactured a mechanism finding for the e-process gate that vanished at equal risk. The experiment is one afternoon: add a constant-exposure hold at scale `c` to `experiments/matched_risk.py`, solve `c` on inner-validation so its realized volatility equals v4's, and re-run the paired bootstrap. Needs no new data, no fetch, and the harness already exists. Pre-register the answer both ways — a hold de-levered to 0.5x is *not* obviously a weaker benchmark, and if the drawdown gap survives it, that is the strongest result this project has ever had. |
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
| 08-17 | ~88 | R-29: every registered strategy (25) on both markets, as a fresh 2023+ account, to attach an interval to each. No selection was made on any of it and the decision rules were committed first — but 50 consultations is 50 consultations. The finding that matters: at ~88 program-level reads, and with `kelly_regime_v4`'s holdout Sharpe needing a **6.2-year** track record to clear a 103-trial bar it has 3.6 years of, **no Sharpe-based claim from this dataset is supportable any more**. Judge on drawdown, which still replicates, and treat B-06 (forward paper trading) as the only remaining source of evidence. |
| 08-18 | ~88 | R-30: **unchanged, and the reasoning is the point.** The bootstrap was re-run over the holdout to recover the log-growth interval, which looks like 50 fresh consultations and is not one: R-29 drew those exact resamples and computed that exact interval object, then persisted only two of its three fields. Every overlapping number came back bit-identical, which is the evidence. No new question was asked; a field was recovered from an answer already given. Read it as ~138 if you disagree — R-29's conclusion that no Sharpe-based claim from this dataset is supportable holds either way. |
| 08-18 | ~112 | R-31: 12 matched-and-reference runs across two markets, 6 re-runs at the 0.40% taker tier, 6 with funding charged on futures. The ETH/BTC falsification cells and the 40-window resample do not read the 2023+ BTC holdout (the R-19/R-28 convention). Every configuration was frozen on inner-validation and the decision rule, the validity gate and the predictions were committed one commit ahead of the first holdout read — `git log` records it. Nothing here is offered as a Sharpe-based claim; the round's finding is that at matched risk there is no difference to claim. |
| 08-17 | ~38 | R-28: three configurations × two markets, plus two cost re-runs. The ETH falsification test and the 40-window resample do not read the 2023+ BTC holdout. At 24 trials in a single session the deflated Sharpe was already 0.859; at ~38 program-level consultations, treat any Sharpe-based claim from this dataset as unsupportable and judge on drawdown, which is the property that has repeatedly replicated. |
| 08-18 | ~124 | R-32: +12 on top of R-31's ~112 (3 frozen arms × 2 markets, 3 spot fee-tier re-runs, 3 funding-charged futures re-runs). The number that matters is not the increment but why it exists: **two sessions were scheduled onto the same backlog row on the same day and each spent the holdout on it independently**. Neither branch did anything wrong — both pre-registered, both froze before reading — but the day cost ~36 consultations and 69 trials for one question, and the project applies 103 + 69 = **172** trials from here. If parallel sessions are going to run, ROUTINE.md's rule that the trials count is the total across branches is the thing that keeps the arithmetic honest; this is the first time it has actually been needed. |
