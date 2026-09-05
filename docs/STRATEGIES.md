# The strategies, best to worst

The original twenty-seven strategies are ordered as in the historical README
comparison table (final balance on the best market over 2017–2026). Ten additional
[R-189 research candidates](#r-189-intraday-game-candidates) are documented below.
Each
section says **what it is**, **how it works**, and **what principles it
rests on**. Balances below are from a **$1,000** start — results scale
proportionally with capital, so one start balance is enough; "spot" is 1x
long-only, "futures" is 5x with shorting allowed. Full metrics:
[../reports/comparison.md](../reports/comparison.md).
Literature: [RESEARCH.md](RESEARCH.md). Robustness: [VALIDATION.md](VALIDATION.md).

Read the order as buckets, not a ranking: most adjacent steps are not
statistically distinguishable
([VALIDATION.md](VALIDATION.md#how-much-of-the-comparison-table-is-signal)).

The five `kelly_regime_*` variant entries (v2/v3/v4 and the two `ev`
strategies) are variants of the leader and are described together in the
[appendix](#appendix-the-variants); the numbered sections below cover the
twenty-two distinct ideas.

| # | strategy | spot | futures 5x | verdict |
|---|---|---|---|---|
| — | [kelly_regime_v4](#kelly_regime_v4--promoted-and-what-it-is-honestly-good-for) | $66.8K | **$156.2K** | current leader; faster anchors, 35% max DD |
| — | [kelly_regime_v3](#kelly_regime_v3--promoted) | $65.8K | **$139.5K** | conditional volatility targeting |
| — | [kelly_regime_v2](#kelly_regime_v2--not-promoted) | $46.4K | **$122.0K** | convex vote response; not promoted |
| 1 | [kelly_regime](#1-kelly_regime) | $42.1K | **$108.2K** | the base idea; 0 liquidations in every resample |
| — | [kelly_regime_ev](#kelly_regime_ev--kelly_regime_ev_fast--the-derived-no-trade-band) | $40.9K | **$108.0K** | the no-trade band, derived instead of chosen |
| — | [kelly_regime_ev_fast](#kelly_regime_ev--kelly_regime_ev_fast--the-derived-no-trade-band) | **$71.1K** | $70.8K | same rule, 1-day horizon; best spot balance from 34 trades |
| 2 | [buy_and_hold](#2-buy_and_hold) | **$66.0K** | $18 (liquidated) | benchmark; unbeatable on spot, fatal on leverage |
| 3 | [champions_council](#3-champions_council) | $19.3K | **$36.8K** | lowest drawdown of the high-return strategies |
| 4 | [hedge_experts](#4-hedge_experts) | **$13.3K** | $258 | profitable on spot, over-trades on leverage |
| 5 | [elliott_wave_zigzag](#5-elliott_wave_zigzag) | **$5,027** | $12.75 (liquidated) | a second, independent Elliott Wave implementation; not distinguishable from holding |
| 6 | [replicator_book](#6-replicator_book) | **$2,330** | $10.58 | modest but real edge on spot |
| 7 | [universal_kelly](#7-universal_kelly) | **$1,276** | $1,227 | tiny gains, remarkable 7% max drawdown |
| 8 | [harsanyi_crowd](#8-harsanyi_crowd) | $888 | $429 | near break-even, very low exposure |
| 9 | [overshoot_fade](#9-overshoot_fade) | $662 | $33 | good win rate, bad tails |
| 10 | [camouflage_flow](#10-camouflage_flow) | $548 | $0.99 | signal exists, fees eat it |
| 11 | [stealth_trend](#11-stealth_trend) | $465 | $0.38 | as above |
| 12 | [flow_regime](#12-flow_regime) | $447 | $0.80 | combination did not rescue its members |
| 13 | [game_council](#13-game_council) | $284 | $2.00 | can only allocate among losers |
| 14 | [elliott_wave](#14-elliott_wave) | $272 | $81.67 (liquidated) | falsifiable Elliott Wave count, decisively negative |
| 15 | [minority_oracle](#15-minority_oracle) | $53 | $3.83 | honest negative result |
| 16 | [game_switch](#16-game_switch) | $5.00 | $1.00 | fee death |
| 17 | [regret_grid](#17-regret_grid) | $5.00 | $1.00 | fee death |
| 18 | [tft_trend](#18-tft_trend) | $4.99 | $1.00 | fee death |
| 19 | [macd_cross](#19-macd_cross) | $4.99 | $1.00 | baseline; fee death |
| 20 | [macd_rsi](#20-macd_rsi) | $4.96 | $0.94 | baseline; fee death |
| 21 | [attrition_reversion](#21-attrition_reversion) | $4.94 | $0.99 | fee death |
| 22 | [rsi_reversion](#22-rsi_reversion) | $4.85 | $0.77 | baseline; fee death |

> **The pattern in one line:** every strategy that makes money decides
> *how much to hold*; every strategy that tries to predict *what happens
> next* loses. On 5-minute bars, after fees, sizing wins and forecasting
> loses.

---

## R-189 intraday game candidates

All ten are **registered research candidates**. Registration and chart inclusion
do not imply promotion; evaluation is recorded in
[reports/r189_games](../reports/r189_games/), using the frozen
[R-189 harness](../experiments/r189_games.py). The
[source review](R189_RESEARCH.md) gives primary citations, equations and the
limits of each adaptation. The shared implementation is
[intraday_games.py](../src/tradebot/strategies/intraday_games.py).

| strategy | distinct mechanism |
|---|---|
| `cautious_optimism` | Optimistic expert regret with regret-dependent entropy pacing, following the 2025 COMWU construction. |
| `squint_council` | The 2026 shared-variance Squint expert game, using numerical integration and an implicit variance update. |
| `normalhedge_council` | The 2026 NormalHedge.BH constant-potential clock, with a disclosed practical initialization. |
| `swap_regret_council` | Conditional expert-switching regrets and a stationary transition distribution. |
| `blackwell_council` | A vector of return, risk and turnover deficits selects a separating exposure direction. |
| `minimax_council` | Pure-expert maximin across three historical payoff summaries, with cash available. |
| `nash_council` | Finite cooperative bargaining over model-committee surpluses and a concentration penalty. |
| `qre_council` | A fixed-count logit-response approximation to an entropy-regularized expert/scenario game. |
| `sleeping_council` | Confidence-rated AdaNormalHedge updates only active specialists. |
| `defensive_forecast` | Finite-feature K29 calibration against a skeptic, followed by a fee-aware exposure rule. |

The common experts combine clipped `kelly_regime_v4`, intraday trend,
reversion, breakout, buy and cash. Targets are **long-only, at most 1x equity**
on both markets. Closed-bar decisions occur at **00/04/08/12/16/20 UTC**, with
next-open execution: at most six requested rebalances per day. Actual fills
can be fewer; fills and completed round trips are different counts.

A `0.05` exposure band checks both the proposed target and actual held notional.
The broker additionally suppresses small nonzero-position rebalances below
**5% of the market's maximum notional**. That is 5% of equity on spot and
25% on a 5x futures account, so futures execution is coarser even though the
candidate target remains capped at 1x. These bands do not force a minimum
daily trade count. The model scenarios are constructed from prior expert
payoffs, not observations of strategic market participants.

---

## 1. `kelly_regime`

**What it is.** A growth-optimal position sizer that only takes risk while
the market's slow regime is bullish. The idea the whole leading family is
built on: **$1,000 → $108,221** on 5x futures where buy-and-hold is
liquidated, at Sharpe 1.42, roughly half buy-and-hold's drawdown, and just
143 trades in nine years. The three promoted variants in the
[appendix](#appendix-the-variants) change how it *sizes*, never what it
predicts.

**How it works.** Three slow anchors — the 30-, 50- and 100-day mean price
— each vote "bull" when price sits 1% above them and "bear" 1% below,
latching their verdict in between so ordinary chop cannot flip them. The
fraction of anchors voting bull (0, ⅓, ⅔ or 1) scales the exposure. Size
itself is a volatility target: hold `target_vol / realized_vol` of equity,
capped at 2x, so risk stays constant as BTC's volatility swings. A 10%
deadband suppresses cosmetic rebalances.

**Principles.** Kelly (1956) and Breiman (1961) established that the
log-optimal bet maximises long-run growth; Bell & Cover (1980) showed it is
the *equilibrium* of the two-investor investment game — no rule beats it
with probability above one half. Full Kelly is notoriously fragile to
estimation error, so exposure is a fraction of it (MacLean, Thorp &
Ziemba 2010). The regime gate comes from Cardaliaguet & Lehalle's (2018)
mean-field game of trade crowding: price drift *is* the crowd's net flow,
so positive expected drift — the precondition for any positive Kelly
fraction — holds while the mean field is still accumulating. When the crowd
turns distributor, the Kelly fraction of a negative-drift bet is zero, so
it stands flat rather than shorting an asset that has drifted upward for a
decade.

**Where it fails.** It lags badly in steady bulls. Out-of-sample 2023–26,
on the same 5x futures, it returned **+325% against leveraged
buy-and-hold's +1,418%** — nothing forces holding to de-lever when no
crash arrives. Its entire edge is in the windows that contain one, which
is why in-sample (2017–22) the same comparison is $25,486 against a
liquidated $18.

---

## 2. `buy_and_hold`

**What it is.** Buy everything on the first bar, never trade again. The
benchmark — and on spot, still the best absolute return: **$66,044**.

**How it works.** One order, no signal, no exits. On 5x futures the same
instruction is fatal: the position is liquidated in the January 2017
crash, ending at **$18**, and in the stress test it was wiped out in
**26 of 40** random windows, with a median window return of **−98%**.

**Principles.** Not a game-theoretic strategy but the null hypothesis every
other one must beat after fees. It encodes the empirical fact that BTC has
drifted upward across its history, and it doubles as a leverage stress
test: it shows exactly what unmanaged exposure does when the asset draws
down 84%.

---

## 3. `champions_council`

**What it is.** A council that allocates across the strategies that
actually make money, then applies its own risk control: **$36,773** on
futures with the lowest drawdown of the high-return strategies (37%, and
29% out-of-sample; `universal_kelly`'s 7% is lower still, on a fraction
of the return).

**How it works.** It runs `kelly_regime`, `hedge_experts`,
`replicator_book` and `universal_kelly`, plus buy-and-hold and a flat
action, and blends their signals with multiplicative weights that grow
with each member's realized volatility-normalized PnL. The blend is then
put through the same fractional-Kelly volatility target as `kelly_regime`,
so **risk is governed centrally** rather than inherited from whichever
member happens to be leading.

**Principles.** A game of games. Which member is right in the next regime
is unknowable in advance, which is exactly the setting Hedge was built for:
Freund & Schapire (1997) guarantee performance within `O(√(T ln N))` of the
best member in hindsight, with no assumption that returns are well-behaved.
Fixed-share mixing (Herbster & Warmuth 1998) lets leadership drift between
regimes, and the flat action anchors the zero-sum maximin floor.

**Caveat.** Its members were chosen *after* seeing their results on this
data, so its in-sample rank is not evidence — which is why it carries an
explicit out-of-sample split in VALIDATION.md.

---

## 4. `hedge_experts`

**What it is.** A no-regret blend of ten ordinary technical signals:
**$13,277** on spot. Notably, it beat every hand-built game-theoretic
predictor without predicting anything itself.

**How it works.** Ten experts each emit a position: vol-scaled momentum at
1h/6h/1d/1w, MACD, an RSI ramp, one-bar reversion, a Donchian breakout,
always-flat, and buy-and-hold. Every bar, each expert is scored on its
volatility-normalized PnL **minus the fees its own position changes would
have cost** — so an expert that churns is charged for churning. Weights
update multiplicatively and the played position is the weighted blend.

**Principles.** Hedge / multiplicative weights (Freund & Schapire 1997;
Arora, Hazan & Kale 2012) converges to the best expert in hindsight on
adversarial sequences. Including always-flat means the guarantee inherits
a floor: asymptotically it cannot do much worse than staying out. On 5x
futures it over-trades (4,103 trades) and the fee drag overwhelms the
edge.

---

## 5. `elliott_wave_zigzag`

**What it is.** A mechanical, causal, no-discretion implementation of
Elliott Wave theory: **$5,027** on spot (holdout: −17.4%, loses to
`buy_and_hold`), liquidated on futures_5x. Registered per backlog item
B-10 specifically to convert a decades-old, "not falsifiable as
practised" debate (R-18) into one falsifiable table row — it was
registered whether it won or lost, and it lost, exactly like every other
pure directional predictor in this table. Independently and concurrently,
a second, unrelated session built a structurally different implementation
of the same backlog item the same day — `elliott_wave` (#14 below) — and
reached the same qualitative verdict from different geometry; see
`docs/LEDGER.md` R-157 for that round's own collision note.

**How it works.** A single causal forward pass computes a standard 5%
percentage ZigZag over 5-minute closes; confirmed swing pivots alternate
low/high by construction, and each pivot only affects the strategy's
output from the bar it is *confirmed* on — never backdated to the
(earlier) bar of its own price extreme. Starting from each confirmed low
as a candidate wave-0, the strategy tracks P1..P5 and applies Frost &
Prechter's (1978) three hard rules for a 5-wave bull impulse: wave 2 may
not fully retrace wave 1 (and, in this literal reading, must retrace into
the canonical [0.382, 0.786] Fibonacci band); wave 4 may not re-enter
wave 1's price territory; wave 3 may not be the shortest of waves 1/3/5.
Any violation invalidates the count and restarts the search from the
violating pivot. The strategy goes long the bar wave 2 confirms valid
(anticipating wave 3, canonically the strongest leg) and flattens at wave
5's completion or at invalidation, whichever comes first. Long-only;
bear-leg counting and diagonal-triangle exceptions are an explicit,
disclosed simplification. One frozen configuration (`pct=0.05`,
`require_fib_band=True`) — no parameter search, per B-10's own "no
discretion" brief.

**Principles.** Elliott (1938), popularized by Frost & Prechter (1978);
the ZigZag+rule construction here is the standard mechanical reading
practitioners use to remove the after-the-fact relabeling that makes
Elliott counts unfalsifiable "as practised." The one quantitative claim
the theory makes beyond its structural rules — that retracements cluster
at Fibonacci ratios — was tested directly against this project's own
data in R-156's novel branch (a Fibonacci-band ablation on the same
engine's rule-invalidation signal) and, consistent with Batchelor &
Ramyar (2005)'s finding that Fibonacci ratios in the Dow are
indistinguishable from chance, added no detection power over the bare
structural rule. On the holdout, the strategy's paired Δ-Sharpe against
`buy_and_hold` is significant in the *losing* direction on both markets
(spot −1.18 [−2.32, −0.14], futures −1.74 [−2.83, −0.43]); its lower
spot drawdown is not a risk-matched improvement (21% vs 100%
time-in-market, and the paired interval on it contains zero). See
docs/LEDGER.md, R-156, for the full holdout battery, the ETH-Bitfinex
falsification (survives without a catastrophic, BTC-specific failure —
still loses, just not by an outlier margin), and the parallel novel
branch (the same engine's invalidation events, tested as a
regime-timing input to `kelly_regime_v4` against the project's
six-episode Step-A gate: 3/6 best cell, the twelfth structurally
distinct mechanism to fail it).

---

## 6. `replicator_book`

**What it is.** A miniature market ecology that reallocates capital between
trading styles by their realized fitness: **$2,330** on spot.

**How it works.** Five species compete — two trend followers (4h and 1d),
two contrarian "fundamentalists" anchored to 1d and 1w means, and cash.
Each bar every species is scored on its own fee-adjusted PnL, and capital
shares grow exponentially in fitness relative to the population average.
Share caps and a small floor keep any species from going extinct or taking
over.

**Principles.** Replicator dynamics (Taylor & Jonker 1978) — the formal
engine of evolutionary selection — applied to one's own strategy roster.
Lux & Marchesi (1999) showed that contagious switching between chartists
and fundamentalists is itself what generates fat tails and volatility
clustering; this rule *replays that mechanism* rather than trying to
forecast the regimes it produces. The logit switching intensity is Brock &
Hommes' (1998) β, whose warning about instability at high β becomes a
concrete fee constraint here. Cash as a species is the Maynard Smith &
Price (1973) ESS anchor: when nothing is fit, the book goes flat.

**Why futures ($10.58) is so much worse than spot ($2,330), diagnosed by
R-148.** Its `on_bar` places orders via `ctx.order_target`,
whose convention is a fraction of the *market's own* maximum leverage (1.0
= fully using whatever leverage the market allows) — not a fraction of
equity. So on 5x futures, a blended position sitting near its own ±1 cap is
already running close to the full 5x leverage, not 1x notional as the spot
number might suggest; there is no separate leverage-management step. This
has no bearing on the registered strategy's own numbers (unchanged) but
explains, for the first time, why the two markets' results diverge so much
more here than for `kelly_regime*` (which sizes via `ctx.order_notional`,
an absolute, leverage-independent fraction of equity).

---

## 7. `universal_kelly`

**What it is.** Cover's universal portfolio on one asset: **$1,276** on
spot — small gains, but a **7.4% maximum drawdown**, by far the smoothest
equity curve in the suite.

**How it works.** It tracks the hypothetical wealth of 41 fixed exposures
from −1 to +1, and plays the wealth-weighted average of them, halved
(fractional Kelly). Because that average moves only as evidence
accumulates, the position drifts slowly and rarely trades.

**Principles.** Cover (1991) proved this mixture achieves the growth rate
of the best fixed exposure in hindsight to within `O(log T)` on **any**
price sequence — no statistical assumptions at all — and Ordentlich & Cover
(1998) showed the construction is essentially minimax-optimal. Bell &
Cover (1980) supply the game-theoretic reading: log-optimal play is the
equilibrium of the investment game. The cost of that universal guarantee is
lag: it concedes the early part of every new regime by construction.

---

## 8. `harsanyi_crowd`

**What it is.** A Bayesian read on which regime the market is in, sized
down when the trend looks crowded: **$888** on spot — a small loss, with
only 0.1% time in market and an 11% drawdown.

**How it works.** The market's hidden type is one of {up-trend,
down-trend, chop}. Each bar's ATR-normalized move updates a posterior over
those types via Bayes' rule, with a sticky prior because regimes persist.
Position follows the *belief margin* (P(up) − P(down)) behind wide
hysteresis bands. On top, a crowding haircut cuts exposure when a trend is
old **and** its volume efficiency is decaying — more volume buying less
price progress.

**Principles.** Harsanyi (1967–68) resolved games where payoffs are
unknown by having players hold a distribution over opponent *types* and
update it from observed play; that is exactly regime detection done
properly, rather than by indicator threshold. The crowding term is
Cardaliaguet & Lehalle's (2018) mean-field cost: being late to a crowded
trade means buying the crowd's permanent impact right before it stops.

---

## 9. `overshoot_fade`

**What it is.** The one mean-reversion trade microstructure theory
actually licenses — fading a forced-liquidation cascade: **$662** on spot
from just 189 trades, with a 60% win rate but losses that outrun the wins.

**How it works.** It waits for a rare conjunction: a 3-sigma move over an
hour, a volume climax, and range expansion. It then requires *exhaustion* —
signed order flow no longer pushing with the move — before fading it,
sized by the move's extremity, with a profit target at half the expected
retrace, a volatility stop, and a 6-hour time stop.

**Principles.** Brunnermeier & Pedersen (2005) show the equilibrium around
a distressed liquidator: predators trade alongside the forced seller,
price **overshoots** fundamentals, then recovers once the forced flow ends.
Glosten & Milgrom (1985) supply the discipline — fade only when adverse
selection is low, because fading *informed* flow is precisely how market
makers lose — and Corwin & Schultz (2012) give the bar-only spread estimate
that sets the minimum retrace worth trading. The failure mode is
structural: when the "overshoot" is real repricing (a hack, an ETF
headline), it never reverts, giving the short-gamma payoff visible in the
worst-trade column.

---

## 10. `camouflage_flow`

**What it is.** An attempt to detect and follow informed traders hiding
inside ordinary volume: **$548** on spot — the signal is real but does not
clear fees.

**How it works.** Bulk Volume Classification splits each bar's volume into
buys and sells using the volatility-standardized return, giving signed
flow without any order-book data. A 3-hour flow imbalance is z-scored
against its 3-day history; trades fire only when that z-score is extreme,
flow "toxicity" exceeds its weekly median (i.e. the flow looks informed
rather than random), and projected volatility clears several times the
round trip.

**Principles.** In Kyle's (1985) equilibrium the insider deliberately
splits orders to hide inside noise-trader volume, so private information
enters price *slowly* — leaving persistent signed flow and durable drift.
Easley, López de Prado & O'Hara (2012) showed that flow can be recovered
from bars alone (VPIN/BVC), and Yang & Zhu (2020) proved that back-running
detected flow is equilibrium-consistent. The theory holds; the 5-minute
implementation gives back more in fees than the drift is worth.

---

## 11. `stealth_trend`

**What it is.** Momentum that only counts bars where informed traders
could plausibly have been hiding: **$465** on spot.

**How it works.** Each bar's return is weighted by how "Kyle-informative"
it looked — high participation relative to normal volume, low price impact
per unit of volume — and those weighted returns are smoothed into a
momentum estimate. Trading is gated to periods when the market is *deeper*
than its weekly norm, and forced flat when impact spikes (market makers
withdrawing).

**Principles.** Kyle (1985) again: informative moves print on deep,
high-participation bars, while thin-volume moves are transient impact.
Amihud (2002) supplies the per-bar price-impact proxy (|return| / dollar
volume), Admati & Pfleiderer (1988) the reason to judge volume against its
normal level, and Barclay & Warner (1993) the empirical footprint —
cumulative drift on unremarkable bars rather than one dramatic print.

---

## 12. `flow_regime`

**What it is.** A combination that lets the two sides of the microstructure
game arbitrate each other: **$447** on spot. It did not rescue its members.

**How it works.** It takes the consensus of the two flow followers
(`camouflage_flow`, `stealth_trend`), lets `overshoot_fade`'s rare
liquidation-event trade **override** them, and vetoes any consensus entry
that fights the `harsanyi_crowd` regime belief.

**Principles.** The design is theoretically sound: following informed flow
is correct *except* at the moment of a forced-liquidation overshoot, when
the equilibrium continuation is reversion — the two trades are
complementary sides of one game. But a combination inherits its members'
economics, and here every member loses to fees, so arbitration between
them cannot manufacture an edge.

---

## 13. `game_council`

**What it is.** Hedge allocation across the seven pure game-theoretic
strategies: **$284** on spot.

**How it works.** Identical machinery to `champions_council` — each member
scored on fee-charged volatility-normalized PnL, weights updated
multiplicatively, target quantized to a coarse grid to stay
piecewise-constant.

**Principles.** Hedge's guarantee is *relative*: it converges to the best
member in hindsight. When the entire member pool loses money after fees,
converging to the least-bad member is all the theorem promises. That is
the honest lesson of this row, and the reason `champions_council` (same
algorithm, profitable members) sits at #3.

---

## 14. `elliott_wave`

**What it is.** A deterministic, causal ZigZag/Fibonacci Elliott Wave
counter, traded directionally: **$272** on spot, liquidated on futures.
Backlog item B-10 — R-18's 08-16 literature rejection of Elliott Wave
Theory, converted into an actual run instead of left as a citation.

**How it works.** A Wilder-style ATR feeds a causal ZigZag: a pivot is
permanently frozen the instant price reverses `k * ATR` from the running
extreme since the last confirmed pivot (never repainted by a later bar).
Every trailing 6-pivot window is hard-gated as a 5-wave impulse (wave 2
retraces 38.2-100% of wave 1 without exceeding wave 0's start; wave 3 is
never the shortest of waves 1/3/5; wave 4 does not enter wave 1's price
territory — the diagonal-triangle exception is deliberately ignored, per
B-10's "no discretion"), and every trailing 4-pivot window as an A-B-C
correction (B retraces 38.2-78.6% of A). A pattern clearing its hard gates
earns a soft Fibonacci-ratio confidence score (Gaussian kernel around the
canonical 1.618/0.618/1.0 ratios); only a completed pattern with confidence
above a fixed threshold moves the target — impulse up/down to long/short
(short clamps to flat on spot automatically), A-B-C to flat. No Kelly
sizing, no discretion, no relabeling: exactly the falsifiable version of
classical Elliott counting B-10 asked for.

**Principles.** R.N. Elliott's wave principle, as practiced, is
unfalsifiable — counts are relabeled after the fact once they fail, the
same causality-leak class `tests/test_causality_strict.py` exists to catch
applied to a human analyst's own process (Aronson 2006). Its one
quantitative, testable claim — that price respects Fibonacci retracement
ratios — was empirically refuted (Batchelor & Ramyar, "Magic numbers in the
Dow"). The one paper claiming a live edge, *ElliottAgents* (Applied
Sciences 14(24), Dec 2024, multi-agent LLM + deep RL), reports 73.68% vs
57.89% directional accuracy on BTC/USD Oct 2022-Sep 2024 — three extra
correct calls over a single monotonic bull run, no walk-forward split.

**Result (R-157, 08-26): NEGATIVE, decisively.** On the 2023+ holdout the
frozen configuration lost to `buy_and_hold` on both spot (-65.0% vs
+283.9%) and futures_5x (-99.6% vs +1417.6%, funding-free upper bound);
the project's own stationary-block-bootstrap harness confirms both
holdout cells at 95% intervals excluding zero (ΔSharpe -3.33 and -2.60,
Δlog growth -2.39 and -8.24). The identical failure magnitude replicated
on ETH (-66.8% spot, -99.9% futures), ruling out "unlucky on BTC" as an
explanation. Registered despite the negative verdict — per B-10's own
framing ("converts an unfalsifiable debate into a table row") and this
project's convention for instructive negatives (`minority_oracle`,
`game_switch`, `game_council`). Turnover stayed material even after the
ATR/confidence gates (543-1,065 trades in the holdout alone, only 7-16%
time-in-market) — the same INFO+ERR+COST combination every other
directional predictor in this ledger has lost to, now measured for
Elliott Wave specifically. A same-round attempt to repurpose the wave
counter's structural-clarity *confidence* (not its direction) as a SIZE-axis
dampener on `kelly_regime_v4` also failed — see `docs/LEDGER.md` R-157.

---

## 15. `minority_oracle`

**What it is.** A grand-canonical minority game trained online on the
binarized return series: **$53** on spot. A clean negative result.

**How it works.** 65 virtual agents each hold two lookup tables mapping the
last 6 return signs to a ±1 prediction. Tables score points when they would
have predicted correctly; each agent plays its best table, and abstains
entirely unless that table's edge clears a threshold. The population's net
vote drives the position.

**Principles.** Challet & Zhang's (1997) minority game, in the
market-prediction form of Johnson et al. (2001), with the abstention
threshold of Jefferies et al. (2001) — which *is* a transaction-cost
filter in the original model. Savit et al. (1999) and Challet, Marsili &
Zecchina (2000) predict that some history states retain exploitable bias
above the α ≈ 0.34 phase transition, and the agent population is sized to
sit in that regime. **The finding:** whatever conditional bias exists in
5-minute BTC sign histories is smaller than the round-trip fee. Its first
version never traded at all — the original gate demanded a ~60% hit rate
over 12 bars — and loosening it produced trading that loses.

---

## 16. `game_switch`

**What it is.** Fictitious play against the market: **$5.00** on spot.

**How it works.** It keeps per-history-state estimates of the conditional
mean and variance of the next return (32 states from the last 5 return
signs), and trades a state only when it has enough samples, a t-statistic
above 2.5, and an expected move that clears fees. Entries are strict,
exits loose.

**Principles.** Brown's (1951) fictitious play — best-respond to the
opponent's empirical distribution — combined with Marsili's (2001) result
that markets are a time-varying mixture of minority (contrarian) and
majority (trend) games, and the $-game of Andersen & Sornette (2003) where
profit-seeking agents switch between the two. In principle it discovers
which game is being played and best-responds. In practice the conditional
means are non-stationary and tiny relative to fees.

---

## 17. `regret_grid`

**What it is.** Regret matching over a grid of positions: **$5.00** on
spot.

**How it works.** Actions are positions {−1, −0.6, −0.3, 0, 0.3, 0.6, 1}.
Each bar, every action's counterfactual payoff (market PnL minus the fee
adopting it would cost) updates a discounted positive-regret vector, and
the played position is the regret-weighted mean action.

**Principles.** Hart & Mas-Colell (2000) proved that playing in proportion
to positive regret drives average regret to zero and steers empirical play
to the set of correlated equilibria, resting on Blackwell's (1956)
approachability theorem; von Neumann (1928) and Freund & Schapire (1999)
add that no-regret play earns at least the zero-sum game value. It is
parameter-free and elegant — and on 5m bars the per-bar payoffs are so
noisy that it rebalances constantly, converting a sound guarantee into
3,461 trades.

---

## 18. `tft_trend`

**What it is.** Axelrod's tit-for-tat played against the trend:
**$4.99** on spot.

**How it works.** Enter a "truce" (long or short) when fast and slow EMAs
separate by an ATR-scaled hurdle with near-unanimous closes. Ratchet an ATR
trailing line; a close through it is a *defection*. The first defection is
forgiven; a second inside a 12-bar window is punished by exiting. A single
catastrophic 3-ATR bar triggers a grim trigger — flat, and no re-entry
until 96 calm bars pass.

**Principles.** Axelrod (1984) found tit-for-tat won the repeated
prisoner's dilemma tournaments by being nice, retaliatory, forgiving and
clear. Nowak & Sigmund (1992) showed *generous* TFT beats strict TFT under
noise — which is why one defection is forgiven, since a 5-minute bar is a
very noisy signal of betrayal. Friedman (1971) supplies the grim trigger.
The framing is elegant and the turnover control is real (2,538 trades, far
fewer than MACD), but the underlying EMA-separation signal has no edge to
protect.

---

## 19. `macd_cross`

**What it is.** The textbook MACD crossover: **$4.99** on spot from 4,301
trades.

**How it works.** Long when the MACD line (12/26 EMA difference) crosses
above its 9-period signal line, flat (spot) or short (futures) on the
cross below.

**Principles.** Momentum: the crossover marks a shift in short-horizon
trend early. On 5-minute bars it fires constantly in chop; at $1,269 of
fees per $1,000 of capital, the fee bill alone is larger than the account.

---

## 20. `macd_rsi`

**What it is.** Trend filter plus pullback timing: **$4.96** on spot.

**How it works.** Only take RSI pullback recoveries in the direction of the
MACD trend — long when the histogram is positive and RSI crosses back up
through 45; mirrored short. Exit when the trend flips or RSI reaches 75.

**Principles.** Combining a trend filter with a reversion trigger is meant
to cover each indicator's weakness. It does produce fewer, better-timed
trades than either alone (2,454 vs 4,301) — and still loses, because the
problem is not entry quality but that 5-minute signals cannot pay 0.1%
round trips thousands of times.

---

## 21. `attrition_reversion`

**What it is.** Market-maker-style reversion with a war-of-attrition exit:
**$4.94** on spot, despite a 58.6% win rate — one of the highest in the
suite.

**How it works.** Fair value is a 1-day EMA of typical price, shifted
*against* current inventory to form a reservation price. It fades
deviations beyond 2.5 ATR from that reservation price, and exits either on
convergence or when accumulated "waiting cost" — a per-bar charge plus
extra pessimism each time the move extends — exceeds the expected
remaining reversion.

**Principles.** Avellaneda & Stoikov (2008) derive the market maker's
reservation price `r = s − q·γ·σ²`: your fair value shifts against your
inventory, which makes adding to a losing position progressively harder —
an automatic anti-martingale governor. The exit is Maynard Smith's (1974)
war of attrition: the evolutionarily stable rule quits when cumulative
waiting cost matches the prize, and Fudenberg & Tirole (1986) add that
non-reversion is *itself* information about the opponent's strength. It
wins most of its trades and still loses, which is the signature of a
short-gamma payoff paying fees on 2,930 trades.

---

## 22. `rsi_reversion`

**What it is.** The classic oversold-bounce baseline: **$4.85** on spot
from 4,464 trades.

**How it works.** Buy when RSI(14) drops below 30, exit when it recovers
past 55; mirror on the short side (RSI > 70) on futures.

**Principles.** Mean reversion: sharp moves overshoot and snap back. It
wins 57% of its trades and still loses almost everything, because it pays
$4,882 in fees on a $1,000 account and its losses in trending markets are
larger than its many small wins — "oversold" keeps getting more oversold.

---

## Appendix: the variants

Research rounds on the leader produced five registered variants: v2, v3
and v4 from the beta-test round (two earned promotion, one did not), and
the two `ev` strategies from deriving the no-trade band analytically.
Full detail: [VALIDATION.md](VALIDATION.md#beta-testing-the-variants).

### `kelly_regime_v4` — PROMOTED, and what it is honestly good for

**What it is.** `kelly_regime_v3` with one change: the regime anchors move
from the ad-hoc 30/50/100 days to a **doubling ladder, 20/40/80**. Nothing
else — same conditional volatility targeting, same hysteresis, same cap.

**How it works.** Each anchor covers twice the horizon of the one below
it, the same a-priori multi-scale structure MACD (12/26) and the HAR
volatility model (daily/weekly/monthly) use, chosen for its structure
rather than fitted. Faster anchors mean the gate flips out of a
deteriorating regime sooner, which is where the change earns its keep.

**Principles.** Same as v3 (Bongaerts–Kang–van Dijk conditional targeting
on top of BTC's inverse leverage effect); the anchor ladder is the
Müller et al. (1997) heterogeneous-market / Corsi (2009) HAR idea that
several fixed timescales beat one estimated horizon.

**Result, and the part that is *not* established.** Across nine anchor
sets in the 18–28 day range, **every** variant cut max drawdown to 35–39%
from v3's 41.8%, and seven of nine scored Sharpe ≥ 1.52. The **drawdown
reduction is the robust finding**. The Sharpe spread across that plateau
(1.52–1.60) sits inside the ±0.2 path-noise band measured by block
bootstrap, so the return improvement should **not** be read as
established — a claim worth stating explicitly, because the headline
balance is the number people will quote. Below ~18 days the plateau
breaks sharply (16/32/64 scores 1.46), which is what makes this a region
rather than a tuned peak. The beta-test harness promotes it: it beats the
base on the full period, drawdown, out-of-sample, and the Monte Carlo
left tail.

**And the scope of that drawdown finding, measured by R-57.** It is a
comparison against a *fully-invested* benchmark, and R-33 showed 88–92% of
the gap is the exposure level. Run frozen on six instruments this strategy
was never fitted on (BCH, LTC, ETC, DASH, LINK, XTZ, Coinbase 5m
2020-04 → 2026-08), the drawdown advantage against a hold carrying v4's own
mean exposure **inverts on 6 of 6**, while against the fully-invested
benchmark it holds on 6 of 6. On BTC and ETH over a window all eight share
it is present (−5.6pp, −11.5pp). Read the finding with that scope attached:
BTC and ETH, 2 of 8 — see
[VALIDATION.md](VALIDATION.md#six-instruments-it-was-never-fitted-on-the-cross-asset-panel).

### `kelly_regime_v3` — PROMOTED

**What it is.** The leader, but it stops re-sizing continuously. It holds
a **constant notional through normal volatility** and switches to full
inverse-volatility sizing only when volatility breaks out high or low,
latching that state until it retraces. $139,509 vs the incumbent's
$108,221, Sharpe 1.55 vs 1.42, better out-of-sample, and it beats the
baseline in 75% of random windows.

**How it works.** A 180-day anchor defines "normal" volatility. While the
ratio of current to anchor volatility sits inside [0.55, 1.70], notional
is pinned at `target_vol / anchor_vol`. Outside that band it reverts to
`target_vol / current_vol`, and only relaxes back once the ratio returns
inside [0.85, 1.20] — the same hysteresis the regime gate uses, applied
to the risk axis.

**Principles.** Bongaerts, Kang & van Dijk (2020, FAJ 76(4)): conditional
(extremes-only) volatility targeting improves Sharpe and cuts tails where
continuous targeting fails. It bites here because of an asset-class fact —
Baur & Dimpfl (2018, Economics Letters 173) document crypto's **inverse
leverage effect**, and on this data the highest-volatility quintile
carries the *highest* forward Sharpe. Continuous targeting therefore
de-levers into the best states; Moreira & Muir's (2017) volatility-managed
alpha requires the opposite sign and is absent here. What survives is
Harvey et al.'s (2018) mechanical tail protection, which this keeps.

### `kelly_regime_v2` — not promoted

**What it is.** The incumbent with one line changed: exposure
scales with `vote_fraction ** 1.75` instead of `vote_fraction`. Same
anchors, same fractional-Kelly volatility target, same cap and deadband.

**How it works.** The three-anchor vote produces 0, ⅓, ⅔ or 1. The
partial values are the *transitional* states — a fast anchor has flipped
while a slow one has not — where drift is near zero and volatility is
elevated. Raising the vote to a power above 1 shrinks those states toward
flat (⅓ becomes 0.15, ⅔ becomes 0.49) while leaving full agreement at 1.

**Principles.** Growth-optimal (Kelly) exposure scales with expected drift
over variance, and that relationship is convex in agreement rather than
linear, so a linear response over-invests in exactly the state momentum
strategies handle worst (Wood, Roberts & Zohren 2022, JFDS). Shrinking
super-linearly when the edge estimate is least reliable is the standard
fractional-Kelly prescription under parameter uncertainty (MacLean, Thorp
& Ziemba 2010).

**Result and the honest caveat.** It improves the full period ($121,993 vs
$108,221), Sharpe (1.49 vs 1.42), drawdown (39.6% vs 42.6%), turnover
(113 vs 143 trades), the Monte Carlo median window and the Monte Carlo
worst window — but it lands **6.5% below the incumbent out-of-sample**, so
the beta-test harness declines to promote it. That shortfall is inside the
±0.2 Sharpe noise floor, and the effect is a plateau across gamma ∈
[1.25, 4.0] rather than a spike, but the failed check is reported rather
than buried. Full detail:
[VALIDATION.md](VALIDATION.md#beta-testing-the-variants).

### `kelly_regime_ev` / `kelly_regime_ev_fast` — the derived no-trade band

**What they are.** `kelly_regime_v4` with the fixed 10% rebalance
deadband replaced by one *derived* from expected profit: rebalance only
when the expected growth gained exceeds the fee it costs.
`kelly_regime_ev` ($40.9K spot / **$108.0K** futures) uses the measured
3.3-day fill spacing as its horizon; `kelly_regime_ev_fast` (**$71.1K**
spot — the best spot balance in the table, from just 34 trades) is the
same rule at a 1-day horizon, registered so the sensitivity of the one
free parameter is visible in the comparison table rather than argued
about in a docstring.

**How it works.** For a Kelly sizer the growth given up by holding
exposure `f` instead of the desired `f*` is `(σ²/2)(f − f*)²` per unit
time; correcting it costs `fee·|Δf|`. Trading is worth it only when the
first exceeds the second, i.e. when `|Δf| > 2·fee/(H·σ²)` for holding
horizon `H` — the classic transaction-cost no-trade band (Constantinides
1986; Davis & Norman 1990). The band is not tuned; it falls out of the
fee, the volatility and the horizon. A full exit is always allowed, since
standing flat removes the whole position's risk when the regime gate asks
for it.

**What it says about fees.** At a 0.10% fee the derived band is ~3x the
hand-set 10% — the fixed deadband was too narrow, and turnover, not
signal, was the binding cost on spot. At a 0.40% fee the band exceeds
1.0: **no rebalance is ever worth its cost**, and the growth-optimal
policy collapses to buy-and-hold — an analytic derivation of what
`scripts/fee_study.py` found by brute force (ledger rows L-05, L-06;
methodology finding 7 in [RESEARCH.md](RESEARCH.md)).

**The honest caveat.** The drawdown is a genuine region (30–38% for
every horizon from 1 to 5 days), but the *return* over the same sweep
swings 3x between adjacent horizon values, and the measured 3.3-day
default happens to sit near the in-sample best. Read these as a turnover
and drawdown result, not a return result.

---

## Multi-asset strategies

A separate axis from the twenty-five single-instrument strategies above:
a bar-by-bar cross-asset allocator that decides across a panel jointly,
from one shared cash/equity pool, rather than one instrument at a time.
`tradebot.multiasset`'s composition adapter cannot express this (fixed
capital splits decided before the run); `src/tradebot/multi_engine.py` and
`src/tradebot/multi_strategy.py` (backlog **B-32**, closed by **R-107**)
are the native engine and registration path this family needs, tracked
separately in the README's own "Multi-asset strategies" section and in
`reports/inference/bootstrap.csv` under the `portfolio` market rather than
`spot`/`futures_5x`.

### `xsmom_entry_band` — registered, NEGATIVE

**What it is.** The first (and, as of R-107, only) registered multi-asset
strategy: R-63's cross-sectional trend score, held under R-65's rank-buffer
selection loop and R-68's own selected winner (an asymmetric entry band,
`delta_in=0.080, delta_out=0.0` — a new entrant must clear a raised bar,
an incumbent is never forced out early), traded across a six-instrument
panel (BCH, LTC, ETC, DASH, LINK, XTZ, Coinbase 5m spot).

**Why it is registered despite being NEGATIVE.** Five research rounds
(R-63, R-65, R-67, R-68, R-72) built and refined this construction entirely
inside `experiments/`, unable to enter the comparison table because no
multi-asset registration path existed. R-107 built that path and needed a
correctness check: does the new engine reproduce a result this project
already trusts? It does, to full float precision (final balance, drawdown,
every bootstrap interval, matched to R-68's own published numbers bit for
bit). Registering it is that correctness check made permanent and visible,
in the spirit of `minority_oracle` and `game_switch` — an instructive
negative, not a promotion. **`further_work=False`** on R-68's own decision
rule: it neither beats a volatility-matched hold nor a notional-matched one
by an interval excluding zero.

**Principles.** Moskowitz, Ooi & Pedersen (2012) time-series momentum,
cross-sectionally selected and Kelly-scaled by `kelly_regime_v4`'s own
conditional-volatility target (Bongaerts, Kang & van Dijk 2020); Gârleanu &
Pedersen's (2013, 2016) aim-portfolio partial adjustment inherited from
R-65; Constantinides (1986)/Davis & Norman (1990)'s transaction-cost
no-trade band, generalized by R-68 into asymmetric entry/exit thresholds.
Full lineage, citations and numbers: `docs/LEDGER.md` rounds R-63 through
R-68 and R-107.

**A same-round attempt to find a promotable candidate — also NEGATIVE.**
R-107's novel branch tried replacing the equal-weight allocation across the
eligible set with a correlation-aware risk-parity weighting (Maillard,
Roncalli & Teiletche 2010, over a Ledoit & Wolf 2004-shrunk causal
covariance), motivated by Baltas (2015) and Bruder & Roncalli (2012): this
project's own R-63 measured the six-to-eight-instrument panel's mean
pairwise correlation at 0.634 and its Grinold breadth at 1.47 of 8, and
risk-parity is the literature's answer to exactly that. The mechanism's own
predicted effect confirmed directly (realized diversification rose on 100%
of inner-train bars) — but making it bind required widening the eligible
set enough to dilute R-63's own cross-sectional selectivity, and the
decisive battery failed hard as a result (D5 gross signal retention +0.061
against a +0.3415 bar — the construction barely clears even at zero fees).
Not registered — a NEGATIVE result without a table row, per the routine's
usual default; full detail in `docs/LEDGER.md` R-107.

---

## Evaluated and not registered (R-190)

Ten variations of the three PROMOTED parents (`kelly_regime_v4`,
`kelly_regime_v3`, `kelly_regime`) revisit unchanged targets at UTC four-hour
slots, comparing them with actual account exposure. Each parent has
0.05/0.10/0.20 equity-notional execution bands; the tenth averages the three
targets with a 0.10 band. Two auxiliary blend bands test its neighbourhood.
They inherit the Kelly/game-theory basis and add a cost/execution experiment,
not a new directional information source.

All ten failed the frozen promotion rule and achieved only 0.17–0.29
fills/day. Code stays in [r190_variations.py](../experiments/r190_variations.py)
per ROUTINE Step 5. See [research and sources](R190_RESEARCH.md),
[the protocol](../experiments/r190_protocol.md), and
[complete results and chart](../reports/r190_variations/README.md).

## Evaluated and not registered (R-188)

Ten further strategies were built, tuned on 2017–2020, selected on
2021–2022 and judged once on the 2023+ holdout against a rule frozen in
advance (`experiments/r188_shared.py`), in September 2026. All ten were
**dropped** — none beat `buy_and_hold` on the holdout within the ±0.2
Sharpe noise floor — so none appears in the table above, and their code
lives under `experiments/r188_*.py` instead of the strategy package.

| candidate | one line | trades/day (holdout) | holdout spot from $1,000 |
|---|---|---|---|
| `robust_kelly` | distributionally robust Kelly: bet the worst-case drift over 10/30/90-day windows | 0.27 | $600 |
| `coin_betting` | Krichevsky–Trofimov coin-betting fraction on daily returns (Orabona & Pál 2016) | 0.01 | $1,148 (Sharpe 0.63 vs 1.03) |
| `level_k` | cognitive-hierarchy best response to the level currently paying (Camerer–Ho–Chong 2004) | 2.2 | $5 |
| `focal_levels` | round-number focal points, breakout mode (Schelling 1960; Osler 2003) | 1.1 | $79 |
| `mfg_crowding` | mean-field-game inventory: slow trend minus the crowd's chase (Casgrain & Jaimungal 2020) | 0.23 | $1,100 (Sharpe 0.23 vs 1.03) |
| `noise_area_breakout` | time-of-day noise band breakout, flat by day end (Zarattini, Barbon & Aziz 2024) | 0.95 | $312 |
| `intraday_momentum` | first four UTC hours predict the last four (Gao, Han, Li & Zhou 2018) | 0.28 | $656 |
| `session_drift` | adaptive hour-of-day seasonality, t-stat gated | 0.14 | $633 |
| `vwap_reversion` | fade 3-sigma deviations from the session VWAP when they clear the fee | 0.30 | $436 |
| `jump_momentum` | follow a Lee–Mykland (2008) jump for an hour | 0.41 | $551 |

Buy-and-hold on spot over the same holdout: $3,839, Sharpe 1.03, 54% max
drawdown. Literature and the per-strategy reading: [RESEARCH.md](RESEARCH.md#ten-more-in-one-round-game-theory-beyond-no-regret-and-intraday-r-188);
full numbers, training slices, fee-free ceilings and the frozen rule:
`docs/LEDGER.md` R-188.
