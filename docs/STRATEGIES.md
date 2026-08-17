# The strategies, best to worst

Every registered strategy — all twenty-five — ordered as in the README
comparison table (final balance on the best market over 2017–2026). Each
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
twenty distinct ideas.

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
| 5 | [replicator_book](#5-replicator_book) | **$2,330** | $10.58 | modest but real edge on spot |
| 6 | [universal_kelly](#6-universal_kelly) | **$1,276** | $1,227 | tiny gains, remarkable 7% max drawdown |
| 7 | [harsanyi_crowd](#7-harsanyi_crowd) | $888 | $429 | near break-even, very low exposure |
| 8 | [overshoot_fade](#8-overshoot_fade) | $662 | $33 | good win rate, bad tails |
| 9 | [camouflage_flow](#9-camouflage_flow) | $548 | $0.99 | signal exists, fees eat it |
| 10 | [stealth_trend](#10-stealth_trend) | $465 | $0.38 | as above |
| 11 | [flow_regime](#11-flow_regime) | $447 | $0.80 | combination did not rescue its members |
| 12 | [game_council](#12-game_council) | $284 | $2.00 | can only allocate among losers |
| 13 | [minority_oracle](#13-minority_oracle) | $53 | $3.83 | honest negative result |
| 14 | [game_switch](#14-game_switch) | $5.00 | $1.00 | fee death |
| 15 | [regret_grid](#15-regret_grid) | $5.00 | $1.00 | fee death |
| 16 | [tft_trend](#16-tft_trend) | $4.99 | $1.00 | fee death |
| 17 | [macd_cross](#17-macd_cross) | $4.99 | $1.00 | baseline; fee death |
| 18 | [macd_rsi](#18-macd_rsi) | $4.96 | $0.94 | baseline; fee death |
| 19 | [attrition_reversion](#19-attrition_reversion) | $4.94 | $0.99 | fee death |
| 20 | [rsi_reversion](#20-rsi_reversion) | $4.85 | $0.77 | baseline; fee death |

> **The pattern in one line:** every strategy that makes money decides
> *how much to hold*; every strategy that tries to predict *what happens
> next* loses. On 5-minute bars, after fees, sizing wins and forecasting
> loses.

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

## 5. `replicator_book`

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

---

## 6. `universal_kelly`

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

## 7. `harsanyi_crowd`

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

## 8. `overshoot_fade`

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

## 9. `camouflage_flow`

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

## 10. `stealth_trend`

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

## 11. `flow_regime`

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

## 12. `game_council`

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

## 13. `minority_oracle`

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

## 14. `game_switch`

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

## 15. `regret_grid`

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

## 16. `tft_trend`

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

## 17. `macd_cross`

**What it is.** The textbook MACD crossover: **$4.99** on spot from 4,301
trades.

**How it works.** Long when the MACD line (12/26 EMA difference) crosses
above its 9-period signal line, flat (spot) or short (futures) on the
cross below.

**Principles.** Momentum: the crossover marks a shift in short-horizon
trend early. On 5-minute bars it fires constantly in chop; at $1,269 of
fees per $1,000 of capital, the fee bill alone is larger than the account.

---

## 18. `macd_rsi`

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

## 19. `attrition_reversion`

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

## 20. `rsi_reversion`

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
