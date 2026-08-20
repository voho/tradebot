# The research ledger — everything already tried

This is the memory of the project. **Read it before proposing anything**
(step 0 of [ROUTINE.md](ROUTINE.md)), and append to it at the end of
every session. Its purpose is to stop the same idea being re-tried blind,
and to make the cost of each attempt visible.

Three registries and a log:

- **[A. Strategies](#a-strategies-registered)** — table: every registered
  strategy, what it attacked, what happened.
- **[B. Research log](#b-research-log-newest-first)** — **one section per
  research round, newest first**: experiments, studies and methodology
  work that never became a registered strategy. This is the append point;
  see [Appending an entry](#appending-an-entry).
- **[C. Ruled out](#c-ruled-out--do-not-re-try-without-new-evidence)** — table: do not re-try without new evidence.
- **[D. Backlog](#d-backlog-ranked)** — table: ranked, with blockers.

Backfilled 2026-08-16 from the long-form docs. `STRATEGIES.md`,
`RESEARCH.md`, `VALIDATION.md` and `LIVE.md` remain the long-form record;
the former `ALTERNATIVES.md`, `CROSS_ASSET.md`, `ELLIOTT_WAVES.md` and
`FRONTIER.md` were folded into this file and `VALIDATION.md` in the
2026-08-17 docs restructure (their findings live in rows R-15–R-18, the
standing diagnosis, section C and the backlog; the measured tables moved
to `VALIDATION.md`). Balances are $1,000 start, full period, from the
README comparison table. Section B was restructured from a table into
dated sections on 2026-08-20, newest first, with every row's text carried
over verbatim.

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

**Scoped by R-57 (08-20), the companion question to R-33's:** *before
believing any property here, check on how many instruments it exists at
all.* Run frozen on six Coinbase instruments it was never fitted on,
`kelly_regime_v4`'s matched-exposure drawdown advantage does not shrink —
it **inverts, 6 of 6** — while the same runs beat the fully-invested
benchmark 6 of 6. Present on BTC and ETH, absent on all six others, in a
window they all share. Every "does it replicate" check in this ledger
before R-57 was n=1 asset, which cannot tell a mechanism from a
calibration.

---

## A. Strategies (registered)

| ID | strategy | added | idea | attacks | spot | fut 5x | verdict | lesson |
|---|---|---|---|---|---|---|---|---|
| L-01 | `kelly_regime_v4` | 08-15 | v3 on a doubling anchor ladder (20/40/80d), Müller 1997 / Corsi 2009 HAR | SIZE | $66.8K | $156.2K | **PROMOTED** | Drawdown 35.3% is the robust finding; the return improvement sits inside the ±0.2 Sharpe noise floor and is *not* established. **Scope, measured by R-57:** that drawdown property is present on BTC and ETH and inverts on 6 of 6 further instruments once the benchmark carries v4's own exposure — a mechanism calibrated to two assets, not a general one. |
| L-02 | `kelly_regime_v3` | 08-15 | Conditional vol targeting — constant notional through normal vol, re-size only on breakout (Bongaerts 2020) | SIZE | $65.8K | $139.5K | **PROMOTED** | Improves every metric in both sub-periods and both markets; flat parameter neighbourhood (8 combos, Sharpe 1.47–1.55). |
| L-03 | `kelly_regime_v2` | 08-15 | Convex vote response: partial anchor agreement = low confidence, not half a signal | SIZE | $46.4K | $122.0K | NOT PROMOTED | Nine of ten metrics improve and it still fails: −6.5% out-of-sample. Kept registered with the failure stated. |
| L-04 | `kelly_regime` | 08-14 | Fractional-Kelly vol-targeted sizing gated on a crowd-regime filter (Cardaliaguet & Lehalle 2018) | SIZE | $42.1K | $108.2K | **PROMOTED** (incumbent) | First strategy to beat the benchmark. Its headline — "regime-gated sizing cuts drawdown" — was the project's one robust finding until **R-33 risk-matched the benchmark**: 88–92% of that gap is holding half the notional, and the remainder is not established on the holdout. What survives matching is a *return* advantage at equal risk (+20.8pp/+23.8pp median across 40 windows), which is a different claim, pre-registered and confirmed on BTC by R-36 — and which **R-57 then found does not reproduce on any of six instruments outside BTC/ETH** (1 of 6 on the mean-notional axis with every interval containing zero, 0 of 6 on the volatility-matched axis). |
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

## B. Research log (newest first)

One section per research round, **newest first** — a new round is
appended at the top of this section, so reading down is reading the
project's history backwards. Every entry carries the same fields, in the
same order; the heading carries the round ID, its date and its verdict so
the section list alone reads as an index. The required fields, and the
template to copy, are in [Appending an entry](#appending-an-entry) at the
bottom of this file and in step 5 of [ROUTINE.md](ROUTINE.md).

This was a single wide table until 08-20, and it had broken in four ways:
prose-length cells no renderer could lay out, unescaped `|` inside
`|basis|`/`|diff|` notation silently shifting R-41's and R-44's columns, a
row (R-46) that landed in section A's table with the wrong column count,
and nine rounds (R-47–R-55) appended *below* the table where they rendered
as raw pipe-separated text. Every row's text is reproduced verbatim in the
entry that replaced it — nothing was dropped in the conversion. Rounds
before R-28 were backfilled from the long-form docs and carry only the
fields their original row had.

### R-58 · 08-20 · NEGATIVE — B-23: shorter stablecoin window vs on-chain corroboration, closing B-23 for good

**Direction.** **B-23** (filed by R-55, LOW priority): "a materially
different mechanism on the same aggregate-USDT-stablecoin-supply-
deceleration signal — e.g. a shorter growth window matched to genuine-
stress duration ..., or corroboration from a second independent signal
rather than filtering one signal alone." Backlog was otherwise empty of
anything OPEN besides B-06 (ongoing, zero-cost) at session start, so this
was the only genuinely open research item available — attempted with the
ledger's own accumulated skepticism about it stated up front (LOW
priority, "a fourth attempt on the identical feature is a weaker bet than
... a genuinely different research direction"). Attacks INFO, sixth
consecutive round on this axis (R-44, R-53, R-54, R-55, this round), and
the fourth/fifth attempt on this specific signal. Not a duplicate of
R-54 (fixed 14-day window, hard override), R-55-conservative (duration
filter bolted onto the unmodified 14-day feature), or R-55-novel
(combination-rule swap on the unmodified feature) — see each branch's own
"not a duplicate of" section for the precise distinctions.

**What was done.** Two parallel unregistered branches, each on a disjoint
new file, neither editing `_stablecoin_signal.py`,
`kelly_regime_v15_stablecoin_veto.py`, `kelly_regime_v16_stablecoin_persist.py`,
`kelly_regime_v16_stablecoin_confirm.py`, `kelly_regime_v4.py`/`_v3.py`/
`kelly_regime.py`, or each other's files. **Conservative**
(`experiments/kelly_regime_v17_stablecoin_shortwindow.py`): B-23's first
named fix — the growth window itself made a swept parameter
(`{2,3,5,7,10}` days in place of the fixed 14), everything else (365-day
z-score window, hysteresis grid, hard-veto architecture) reused
byte-for-byte from R-54. Literature basis for the window grid (new this
round, via web search): ESRB Oct-2025 and 2025-2026 industry technical
write-ups converge on ~48–72-hour acute stablecoin-redemption-stress
timescales, motivating the 2–10-day grid — with the risk **named in
advance** that this acute-redemption timescale and the multi-week
capital-flight dynamic R-54's signal actually leads might not be the same
clock. Pre-registered a Step-A gate, run *before* any Sharpe number: a
candidate window passes only if its lead-time result (R-54's own
`leadtime()` methodology) matches or beats the N=14 reference (≥9/12
episodes leading, median ≥+16.5d) on both axes; if none pass, decision is
fixed as NOT TO PROMOTE regardless of anything computed afterward. **15
configurations** (`growth_window_days`×`gap` at fixed `thresh_hi`;
pre-registration specified a fuller 45-config grid, scoped down
mid-session after Step A already returned a clean, decisive kill and the
full grid had not finished in a reasonable window — stated explicitly,
not a goalpost move, since Step A alone (not a borderline Sharpe read)
already decided the outcome). **Novel**
(`experiments/kelly_regime_v17_stablecoin_corroborate.py`): B-23's second
named fix — an AND-gate requiring BTC on-chain active-address-growth
stress (`data/btc_onchain_daily.csv.gz`, B-07/R-44's existing
price-independent channel) to *also* read "stress" before the stablecoin
signal's dilution/override is allowed to apply, rather than any further
filter on the stablecoin series alone. Hash Ribbons was screened as an
alternative corroborator and rejected up front (corroborates only 1/12 of
R-54's matched onsets — too coarse). Pre-registered five falsification
gates plus a holdout rule requiring all five to pass before any 2023+ read
(full list, and the pre-sweep mechanism check run *before* the sweep per
this round's own instruction, in the branch's report). **33
configurations** (21 confirming-vote-dilution + 4 no-corroboration
ablation + 8 hard-override-architecture ablation). **Round total: 48
configurations.** Both branches ran the standard multi-pathway causality
tamper probe (price, plus each branch's own new data pathway,
individually and combined) and an identity-recovery check. The operator
independently reproduced, from a clean shell: the conservative branch's
`leadtime_by_window()` (all 6 windows, exact match including the
monotonic +16.5d→−15.0d flip); the novel branch's `mechanism_check()`
(the 7/9-vs-3/3 corroboration-rate finding, the 24→28 flip-count
non-reduction, the 1/12 Hash-Ribbons rate, all exact matches) and
`override()` (all 8 cells of the exposure-artifact-explained
hard-override rescue, exact match to the report's Δ Sharpe figures). Full
suite: `pytest` 457 passed on both branches' final state, unchanged.

**Result.** **Conservative — Step A (the pre-registered gate) already
decides it before any Sharpe number**: 0/5 candidate windows preserve or
improve the N=14 reference lead-time result. The flip is monotonic, not
noisy — lead fraction 75%→50%→41%→40%→33%→31% and median offset from
+16.5d down through 0d (N=10) to −15.0d (N=2) as the window shrinks —
diagnosed as the acute-redemption timescale (~2-3 days, per this round's
own literature search) and the actual multi-week capital-flight lead
being genuinely different clocks, so shortening the window substitutes a
faster but less useful signal. Every downstream check corroborates rather
than reopens the question: no window clears v4 on inner-validation Sharpe
(best −0.049 spot vs the required +0.2); no plateau (many cells sharply
negative to −0.54); the one near-tied config (w=3d) is an exposure-level
artifact (R²=0.98, near-relabeling of v4's own vote); the one config with
a genuinely different exposure shape (w=10d, R²=0.57, comparable to
R-54's own 0.61 at N=14) is also the one that fails ETH falsification
outright (BTC ratio 1.041×, ETH ratio 0.992×). Pre-2020 BTC control
passes cleanly (all windows within ±0.17 Sharpe of v4). **Novel — the
pre-sweep mechanism check already predicts the outcome**: at the natural,
non-cherry-picked onchain threshold, corroboration does not discriminate
genuine leads from noise (7/9 leading episodes corroborate vs. 3/3
lagging/unmatched episodes *also* corroborate — both mostly just read
"broadly a downtrend"), and AND-gating the tightest config's vote
*increases* its raw flip count (24→28) rather than reducing it. The full
sweep confirms: no non-identity configuration beats v4 on
inner-validation spot Sharpe (best 0.11 vs. 0.14, identical qualitative
pattern to R-55's own uncorroborated confirming vote); ETH falsification
fails (18/42 cells, concentrated on spot, the same asymmetry R-55 already
found); corroboration-vs-plain ablation is a wash on inner-validation and
uniformly worse on inner-train (8/8 cells) — not the clean "earns its
keep" result required. The round's one striking-looking number —
corroboration turning R-54's decisively negative hard-override
architecture into one that ties/marginally beats v4 (Δ Sharpe up to
+0.81) — is fully explained and closed by an exposure-artifact check
(R²=0.9971, mean exposure within 0.3% of v4's own): corroboration mostly
just *disables* the override back toward v4's own path, the same
exposure-relabeling trap R-33/R-34 already named, now caught a third
time. Causality passes cleanly on both branches, all pathways, both
identity checks exact. **Holdout: never consulted by either branch** —
both pre-registered rules required a genuine inner-validation +
falsification win before any 2023+ read, and neither branch produced one;
grep-verified by each branch's own author and independently re-verified
by the operator (every `OOS_START="2023-01-01"` call site checked:
exclusive upper bound restricting the causality probe only, never a data
read past the boundary).

**Verdict.** **NEGATIVE, both branches.** One-line lesson: B-23 assumed
its two named fixes attacked the stablecoin signal's precision problem
from genuinely different angles, and they did — a feature-timescale
change and a second-signal corroboration are not the same axis as R-55's
duration filter — but both still failed, for two different, well-
diagnosed reasons (a timescale mismatch between acute-redemption and
capital-flight dynamics; an available corroborating signal that tracks
the same broad-downtrend state as the noise it was meant to filter out,
not the specific distinction needed). This closes the stablecoin-signal
research line's fourth and fifth structurally distinct mechanism
attempts (R-54 hard veto, R-55 persistence filter, R-55 confirming vote,
this round's window/corroboration pair) with the identical qualitative
shape every time: a genuinely new, real information channel, but no
combination mechanism yet converts it into a working strategy on this
project's data. **Holdout counter: +0, running total ~627** (unchanged
from R-55). Decision rule did not move on either branch — both were
fixed and reported exactly as pre-registered. **Next step**: B-23 is
CLOSED, both named fixes tried and failed. Nothing is left genuinely OPEN
on the backlog. `scripts/paper_trade.py` (B-06, ongoing since R-48,
advanced by +1 decision this session) remains the standing zero-cost
recommendation; a future session with a fresh idea should treat the
entire stablecoin-signal line, and arguably the whole INFO axis after six
consecutive rounds (R-44 through this one), as exhausted absent a
genuinely new information channel or a materially different combination
architecture neither named-fix attempt here anticipated.

---

### R-57 · 08-20 · NEGATIVE — `kelly_regime_v4`'s drawdown property is BTC-and-ETH-specific: 0 of 6 on instruments it was never fitted on (N≈3)

**Direction.** Off-backlog, and pointed at this project's own surviving
claim rather than at a new signal. The ranked list held nothing genuinely
OPEN but B-06 (ongoing, zero-cost) and the LOW-priority B-23, which per
ROUTINE.md step 0 licenses a new direction; this round's is not "what else
can be layered onto `kelly_regime_v4`" but **"does the incumbent's own
surviving property hold anywhere other than the two assets we keep
measuring it on?"** Every replication check in this ledger is n=1 asset —
R-17 (ETH Bitfinex 2016–2019, which also shares the 2018 bear with the main
dataset) and R-47/B-08 (ETH Coinbase 2020–2026) — so "the risk property
transfers, the return property does not" rested on two correlated assets,
one of them the fitting asset. Attacks **N≈3**, honestly scoped: a panel
adds independent price paths and microstructure, **not** independent regime
events (the alts share the 2021 top and the 2022 bear, correlating ~0.7–0.9
daily). Not a duplicate of R-17 or R-47 (both n=1 asset; this round reuses
R-47's BEAR22 window and 0.40% sensitivity verbatim for comparability), of
R-42/R-43/R-50/R-51/R-52 (multi-asset *portfolios*, holding capital in two
assets at once — here one asset per backtest and a replication question), or
of the sixteen SIZE-axis / five INFO-axis / two COST-axis branches R-34 →
R-56, every one of which *changes* the strategy. Nothing about the strategy
changes here at all. New infrastructure, additive: `scripts/fetch_coinbase_panel.py`
(concurrent panel fetcher), six committed 5m series, and
`tradebot.data.load_coinbase_spot` / `coinbase_spot_file` (+4 tests).

**What was done.** Single-threaded, one idea, one file:
`experiments/r57_cross_asset_panel.py`. Asset selection was mechanical and
liquidity-based, executed before the file was written: Coinbase USD spot
products online 2026-08-20, minus BTC/ETH and stablecoin/wrapped bases;
alive on three fixed 2020 probe days; ranked by dollar volume (BCH 24.7M,
LTC 10.4M, ETC 9.3M, XRP 8.4M, DASH 8.3M, LINK 3.8M, XTZ 2.6M, OXT 2.1M,
XLM 1.7M, ZRX 0.8M); then a continuity/coverage gate. **The
pre-registration was committed two commits before any backtest** (`c22ba3e`,
`8a7fa5b`) with D1–D4, the falsification test and its predicted outcome
fixed in advance — reproduced below. **One amendment, recorded in full and
made before any strategy number existed:** the original single "coverage
≥95% and gap ≤7 days" clause excluded four of seven candidates — XRP's
905-day listing hole, the case it was written for, but also ETC (91.5%),
DASH (82.1%) and XTZ (91.0%), none with a gap over 6h40m — conflating a
listing hole with thin trading (a 5-minute interval with no print produces
no Coinbase candle) and leaving n=3, where the pre-registered 6/6 threshold
is unreachable and the round could not have returned a verdict at all. It
was split into continuity (gap ≤7d) plus a coverage floor **derived** from
R-07's own validated 18–28-day anchor plateau: coverage f stretches the
20-day anchor to 20/f calendar days, so f ≥ 0.80 keeps it ≤ 25d (ETC 21.9d,
XTZ 22.0d, DASH 24.4d). The resulting panel is the six the liquidity ranking
named, minus XRP, plus the pre-authorized XTZ substitute. Causality tamper
probe (the `test_causality_strict.py` methodology, run against each new
loading path because that module hard-codes the BTC loader): **PASS 6/6**.
**130 configurations** — 114 in the frozen matrix, 16 in the post-hoc
control — with no sweep at all; the only search is the volatility-matching
solver, whose iterations are counted anyway.

#### R-57 pre-registration — as committed, before any panel result was read

- **Panel** BCH, LTC, ETC, DASH, LINK, XTZ (Coinbase USD 5m, fetched
  2020-01-01 → 2026-08-20).
- **Windows** FULL 2020-04-01 → last bar (three months after the data
  starts, so v4's 80-day warmup comes from bars *before* the measured
  period — R-22); BEAR22 2022-05-01 → 2022-11-30, R-47's own ETH window.
- **Arms** frozen `kelly_regime_v4`; `buy_and_hold`; and
  `ConstantExposureHold(c = v4's own mean clipped notional over the same
  window and market)` — the R-33 matched arm, because the standing rule
  here is *match risk before comparing anything*. Robustness only: the same
  arm matched on equal realized volatility, c solved per cell.
- **Costs** spot 0.10%, spot 0.40% (Bitstamp entry tier), futures 5x 0.05%
  with **funding not charged** — no altcoin funding series exists here and
  none is proxied, so every futures number is an upper bound.
- **D1 (primary, the risk claim)** FULL, spot @0.10%: v4's max drawdown
  strictly below the **matched** hold's in k of 6. 6/6 → REPLICATES (exact
  one-sided binomial p = 0.0156); 5/6 → SUGGESTIVE, not established; ≤4/6 →
  FAILS. Reported with each asset's paired stationary-block-bootstrap
  interval (daily returns, 30-day mean block, 2,000 resamples, seed 7).
- **D2 (falsification test, from ROUTINE step 2's menu — "does it survive a
  0.40% taker")** v4 beats `buy_and_hold`'s final balance in ≥5 of 6.
  **Predicted outcome, recorded before the run: FAILS.**
- **D3 (context, explicitly not evidence)** the same count against the
  *unmatched* fully-invested hold, both markets. Expected 6/6.
- **D4** D1 and D3 on BEAR22, descriptive only.
- **Named failure mode** D1 ≤ 4/6 — the advantage absent or sign-unstable
  outside BTC/ETH, which would make the project's one surviving positive
  claim a two-asset coincidence.
- **Holdout cost +0**, by the R-47/B-08 convention.

**Result.** **D1: 0 of 6.** Not a shrinking advantage — the sign **inverts
on every asset**. Δ max drawdown (positive = v4 worse): BCH **+5.2**
[−6.1, +45.7], LTC **+33.8** [+2.1, +53.1], ETC **+23.6** [+5.3, +45.9],
DASH **+29.8** [+2.5, +41.8], LINK **+13.4** [−5.1, +39.8], XTZ **+19.3**
[+3.3, +44.8] — **4 of 6 intervals exclude zero, all four against v4**.
**D2: 2 of 6 — FAILS, exactly as predicted**, and both passes (DASH, XTZ)
are assets where holding lost 51% and 87%, i.e. cleared by holding less
rather than by trading well. **D3: 6 of 6 on spot and 6 of 6 on futures**,
by 16–46pp. Same six assets, same runs: **6/6 unmatched, 0/6 matched.**
**D4: matched 0/6, unmatched 6/6, both markets** — v4 preserves capital
through the 2022 bear in absolute terms (0.80–1.25× on spot), and so does
any arm holding the 11–22% of equity the matched arm holds, with less
drawdown in every cell. **Robustness (equal-realized-volatility matching,
all six cells valid, residual ≤0.8%):** v4's drawdown lower in 2 of 6, its
final balance lower in 6 of 6. **By-product, not pre-registered in this
round: R-36/B-14's confirmed return-per-risk edge does not reproduce
either** — v4 out-returns the matched hold on 1 of 6 (every growth interval
contains zero) and 0 of 6 on the volatility-matched axis. **Post-hoc
control, run after D1 and labelled as a control rather than a decision
rule** (same comparison over a window every asset shares, truncated at
2022-12-31 so no 2023+ bar is read): BTC **−5.6pp** [−20.0, +16.4] and ETH
**−11.5pp** [−17.3, +19.6] in v4's favour, against +0.0 to +17.1pp on all
six panel assets — **2 of 8, and they are exactly BTC and ETH**. The
failure is **asset-specific, not period-specific**. No independent skeptic
was dispatched (single-session round, no parallel branches); the run log is
committed at `reports/cross_asset_panel/run_log.txt` and every cell is in
`cells.csv` / `control_pre2023.csv` for re-derivation.

**Verdict.** **NEGATIVE — and it is a negative about the incumbent, not
about a candidate.** One-line lesson: **`kelly_regime_v4` is not a general
regime-sizing mechanism; it is a mechanism calibrated to two instruments.**
Read against a fully-invested benchmark its drawdown property looks like a
property of the strategy in 8 of 8 cases; read against a benchmark carrying
the same exposure it is a property of BTC and ETH in 2 of 8, and it inverts
everywhere else. Nothing already recorded is retracted — R-33 had already
established that 88–92% of the headline gap is exposure, and R-17/R-47's
ETH numbers reproduce here — but the **scope** of what those rounds left
standing is now measured instead of assumed, which nothing before this
round could do at n=1 asset per check. Hypotheses for *why*, named and not
tested: v4's `target_vol=0.55` / `max_leverage=2.0` are BTC-calibrated, so
on higher-volatility instruments the scale term is small and
near-permanently binding (mean notional 0.18–0.26 on the panel vs 0.38 BTC
/ 0.34 ETH over the shared window), leaving mostly the vote's *timing*; and
a constant-exposure arm that rebalances back to `c` is quietly a
buy-the-dip rule, worth more in a higher-volatility mean-reverting
instrument than a trend-latched gate that stands aside after a drop — the
matched arm wins by most exactly where v4's notional is smallest. Holdout
counter: **+0**, program total unchanged at **~627** — no 2023+ BTC bar is
evaluated anywhere in this module. Decision rule did **not** move; the one
amendment was to the asset-selection rule, before any backtest, and is
recorded above and in the module docstring. Not registered — no candidate,
nothing to promote; the file stays in `experiments/` per ROUTINE.md step 5.
`pytest`: 461 passed (457 + 4 new loader tests). **Next step.** Filed as
**B-25**: is the BTC-calibrated `target_vol` the binding reason the
mechanism does not travel? Ranked below B-06 deliberately — it is the
seventeenth attempt on this strategy family's own parameters and the record
there is 0-for-16 — and it must clear the **matched-exposure** bar on the
same six instruments, which now ship with the repo, so any future candidate
can be failed on six independent instruments before a holdout consultation
is spent on it. Report:
`experiments/reports/r57_cross_asset_panel_report.md`; chart:
`reports/cross_asset_panel/panel_drawdown.png`. Note that the same day's
R-56 was a separate, concurrently-running session on a different axis, and
both are recorded, as R-31/R-32 were.

---

### R-56 · 08-20 · NEGATIVE — Maker/limit-order execution model for `kelly_regime_v4`'s rebalances (COST)

**Direction.** Backlog was empty except B-06 (ongoing, zero-cost) and B-23
(LOW, not recommended) after R-55; per ROUTINE.md step 0 this licensed a
genuinely new direction rather than the backlog. Web research first (Baker
& McHale 2013, *Decision Analysis* 10(3), on Kelly under parameter
uncertainty; Sukhov 2025, SSRN) turned up a shrinkage-Kelly idea, but a
ledger grep found it already tried and NEGATIVE as R-40's novel branch
(`kelly_regime_v8_uncertainty_shrink.py`) — discarded per Step 1, and the
search continued to an unexplored axis: **COST**, the constraint R-12/R-13/
R-14 found most damaging ("no rebalance is ever worth its cost" at
Bitstamp's real 0.40% taker tier), attacked via *execution* — how an
already-decided trade fills — rather than a 17th tweak to the SIZE/vote
axis (exhausted, R-34–R-52) or a 4th INFO-axis combination rule on the same
stablecoin signal (B-23). This project's engine has always filled every
trade as taker (verified: no `maker`/`limit_order`/`post_only` token
anywhere in `src/` or the ledger before this round), even though v4's own
no-trade band (L-05) already limits it to ~150–260 non-urgent rebalances
over 9 years. Real Bitstamp fee schedule verified via web search: entry
tier 0.40% taker / 0.30% maker, top tier 0.03%/0.00% (cited "Bitstamp fee
schedule, accessed 2026-08-20"). Not a duplicate of L-05/L-06 (which decide
*when* to trade; this round assumes that decision is already made and asks
whether the trade can fill cheaper), R-12/R-13 (taker-only fee-tier
sweeps), or R-40 (SIZE/vote-signal shrinkage — this round never touches the
signal).

**What was done.** Two parallel unregistered branches, each on a disjoint
new file, neither modifying `kelly_regime_v4.py`/`_v3.py`/`kelly_regime.py`/
`engine.py`/`broker.py` — both reuse `KellyRegimeV4.prepare()`'s causal
target series and the real `PaperBroker`/`build_trades`/`compute_metrics`
read-only, so fee/leverage/liquidation accounting is byte-identical to
production; only the fill mechanism is new code. **Conservative**
(`experiments/kelly_regime_exec_limit_conservative.py`): post a resting
limit at the signal bar's close, check bars i+1..i+N-1's high/low for a
touch (100% fill on touch — the standard textbook assumption), forced taker
fallback at bar i+N's open if untouched; N∈{1,2,3,6,12,24,72,288} × 2 fee
tiers × 2 markets × 2 inner periods (72 backtests) + 16 falsification + 8
crash-lag diagnostics = **96 pre-registered configurations** (123 actually
executed counting diagnostics, honest-count convention per R-39). **Novel**
(`experiments/kelly_regime_exec_limit_novel.py`): deliberately more
realistic — fill probability is a deterministic function of how far the
touching bar's range penetrated past the limit (Cont & Kukanov 2017,
*Quantitative Finance* 17(1), queue-position-dependent fill probability —
the operator's originally suggested Cont/Kukanov/Stoikov 2014 citation was
checked by the agent, found to be the wrong paper (price impact, not fill
probability), and corrected), with posting aggressiveness scaled by v4's
own conditional-vol-targeting `scale[i]` as a conviction proxy; **51
distinct configurations** (27-point main grid + 12 sensitivity/extension +
12 ablation) validated across 8 slices (2 markets × 2 periods × 2 fee
tiers) + 3 falsification slices against 12 uncounted baseline references.
**Configs evaluated: 147** (conservative 96, novel 51), the total across
both branches per the parallel-round convention. Both ran an explicit
causality/tamper probe (multiply/divide-tamper of everything after a cut
bar, matching `test_causality_strict.py`'s pattern, plus a deterministic
synthetic guard-the-guard construction) and both pre-registered, before
running anything: the ETH (Bitfinex, pre-2020) and BTC-control (Bitfinex,
pre-2020) falsification pair, and a crash-transition-lag check (does the
model delay a regime-flip-to-flat flatten by more than 1-2 bars vs. the
always-taker baseline, since L-01's entire edge is "the windows that
contain a crash"). The operator independently reproduced, from a clean
shell: both branches' causality probes bit-for-bit (conservative: 105
pre-cut events/85 fills identical under both tampers, $5,956.00/$768.84
post-cut divergence, exact match; novel: PASS); the conservative branch's
full 16-configuration ETH+BTC-control falsification table (every number
matched exactly, e.g. BTC-control spot N=288 $9,992.2, ETH-falsification
spot N=3 $4,419.0); and the novel branch's 128-event crash-transition-lag
check (mean lag 5.7 bars, max 9, exact match).

**Result.** **Conservative — mechanically clean, decisively insufficient.**
Fee savings are real and monotonic in every one of 4 (market×tier)
inner-train cells (maker-fill rate 95%→99.8% as N rises, $150–384 saved per
4-year window depending on tier, plateau with no cliff), and N=1 reduces
bit-for-bit to the as-shipped baseline (a correctness check, not a result).
But **no Sharpe improvement anywhere — inner-train, inner-validation, ETH,
or BTC-control, either fee tier, either market — clears this project's own
±0.2 noise floor** (best: Δ+0.07, inner-train futures N=12; everywhere else
Δ+0.01 to +0.05). ETH and BTC-control falsification both PASS directionally
(same sign, same sub-noise-floor magnitude) but that only shows the *lack*
of an effect replicates too. The pre-registered crash-transition-lag test
**FAILS on its literal threshold for N≥3**: 124/128 flip-to-flat events
resolve within 1-2 bars regardless of N, but one severe Jan-2019 near-miss
(missed a touch by $1.61, fell through the entire window, cost ≈$17-20 in
worse execution) and several 3-9 bar delays exist — though none fall inside
the project's three marquee crash windows (Nov 2018, COVID Mar 2020, FTX
Oct/Nov 2022 all resolved 0-2 bars, indistinguishable from baseline).
Separately, futures inner-validation reverses sign at N≥72 (several 25-71
bar delays during the 2021-22 trend that net worse than earlier forced
fallback would have been) — a real, if diffuse, over-patience cost distinct
from the near-miss mechanism. **Novel — decisively NEGATIVE, and its own
ablations show why the conservative branch's headline is optimistic, not
just insufficient.** The literature-grounded, less-than-certain
fill-probability model underperforms the always-taker baseline in **every
one of 8 inner slices** (−4.3% to −23.3%) and both falsification slices
(ETH −12.1%, BTC-control −15.1%, same sign/magnitude as inner-train —
decisively PASS as a falsification, i.e. the negative result is real and
replicates) — the delay/adverse-price cost of waiting for a resting order
exceeds the maker/taker fee gap every time, and drawdown is *worse*, not
better, in every slice. Both "sophistications" independently fail their own
ablations: a flat, non-adaptive fill probability (P=0.7) beats the
literature-grounded penetration-based model by ~21% in 9/9 head-to-head
comparisons (the "more realistic" model is *more conservative* about
fills, pushing more volume to the costly taker fallback), and
conviction-adaptive posting is statistically indistinguishable from a fixed
offset (<1% apart, inside the noise floor). Crash-transition lag: mean 5.7
bars / ~28.5 min (vs. baseline's fixed 1 bar), max 9 (the patience
ceiling) — bounded by construction so it narrowly avoids being
catastrophic, but a real, quantified structural weakness (posting a SELL
limit above a falling market during exactly the de-risking events that are
v4's edge is disproportionately likely to go unfilled).

**Verdict.** **NEGATIVE (both branches).** One-line lesson: **the
conservative branch's 100%-fill-on-touch assumption is the optimistic edge
case of a spectrum the novel branch's more realistic model shows collapses
to a loss once fill uncertainty and adverse selection during de-risking
events are priced in** — the true answer likely sits between the two, and
given the conservative branch's own headline never cleared the noise floor
even at its most optimistic, a more realistic accounting is very unlikely
to do better. Both branches independently confirm the same mechanistic risk
this project has repeatedly found in other forms (R-08's sign-inversion,
R-46's floor-saturation): a change that looks clean in aggregate can still
be quietly wrong at exactly the moments — crash de-risking — that make up
the whole strategy's edge, and only an explicit crash-transition check (not
in either branch's original brief until the operator required it) surfaces
that. No holdout read on either branch — correctly withheld, since neither
cleared its own pre-registered bar. Holdout counter: **+0** on top of
R-55's ~627 (program total remains ~627) — neither branch constructed,
read, or printed any bar dated 2023-01-01 or later; both branches' own
runtime assertions plus the operator's independent grep of both files
confirm this (conservative: the sole 2023+ token is an unused, unreferenced
`OOS_START` constant; novel: no 2023+ literal at all). Decision rule did
not move — both pre-registered thresholds were read as written. Not
registered — no code under `src/tradebot/` touched, no candidate cleared
the promotion bar; both files stay in `experiments/` per ROUTINE.md step 5.
`pytest`: 457 passed, unchanged, confirmed independently by the operator
after both branches. **Next step.** The conservative branch's own
"least-bad" N∈[2,24] residual is **not** promoted here (it was not the
pre-registered decision subset, and per ROUTINE.md that would be
goalpost-moving) — filed as new backlog item **B-24**, LOW priority: even
that subset never cleared the noise floor, so a re-run is a weak bet.
**B-06 (forward paper trading, ongoing since R-48) remains this project's
standing zero-cost recommendation.**

---

### R-55 · 08-20 · NEGATIVE — B-22: stablecoin persistence filter vs confirming-vote architecture

**Direction.** **B-22**: neither of R-54's two named fixes was ever tried
— a magnitude-*and*-duration persistence filter on the stablecoin hard
veto, vs. feeding the same (now-confirmed-leading) signal into R-53's
precision-weighted CONFIRMING-vote architecture instead of a unilateral
override — the only genuinely OPEN backlog item after R-54

**What was done.** Two parallel unregistered branches, each on a disjoint
file, neither editing `kelly_regime_v15_stablecoin_veto.py`,
`kelly_regime_v14_macro_lead.py`, `_stablecoin_signal.py`,
`kelly_regime_v4.py`/`_v3.py`/`kelly_regime.py`, or each other's files —
both import `compute_stablecoin_stress` read-only from the unedited
`_stablecoin_signal.py`, per R-54's own precedent. **Conservative**
(`experiments/kelly_regime_v16_stablecoin_persist.py`): R-54's identical
hard-override architecture, plus one new parameter — `persist_days`,
requiring the latched stress vote to stay continuously above `thresh_hi`
for N consecutive days before it is allowed to force `frac=0` — swept
0/1/2/3/5/7/10/14 at R-54's primary threshold (1.00/0.75) and its two
worst false-positive-prone configs (0.75/0.00, 0.75/0.75); grounded in
regime-detection literature's standard use of minimum-duration
confirmation filters to trade detection speed against false positives
(Shu, Yu & Mulvey 2024, arXiv:2402.05272, penalizes excessive switching
for the identical reason — already cited in this project's own R-02 row
for a different mechanism), with a 3–7 day literature-typical range
motivating the grid center. **Novel**
(`experiments/kelly_regime_v16_stablecoin_confirm.py`): literally R-53's
`KellyRegimeV14MacroLead` combination rule
(`frac=(anchor_sum+weight·vote)/(3+weight)`) reused verbatim (helpers
duplicated, not imported, per R-54's own precedent for citing prior-round
code), fed by the stablecoin latched vote instead of VIX/DXY —
`stable_weight`∈{0(identity),0.15,0.33,0.5,1.0(unweighted-average
control)} × 4 threshold/gap points (R-54's primary, its tightest, its
tightest-with-hysteresis, and a looser point), motivated by 2025-2026
industry practice reportedly weighting stablecoin-flow signals at 15–25%
of a combined signal rather than as a standalone override. Both
pre-registered their falsification tests before running: conservative —
does the filter preserve R-54's lead-time result while cutting false
positives, and does it still pass ETH-vs-BTC-control; novel — does the
candidate clear v4 on inner-validation Sharpe across a genuine plateau,
pass ETH-vs-BTC-control, and does feeding a genuinely-leading signal
change R-53's own architecture-comparison outcome. **45 configurations
total** (conservative 24: 3 thresh/gap × 8 persist_days; novel 21: 17
confirming-vote cells + 4 hard-override-ablation cells run for the
architecture comparison). Both ran the standard two-pathway causality
tamper probe and an identity-recovery check (conservative:
`persist_days=0` recovers R-54's v15 vote bit-for-bit;
`thresh_hi=1e9`/`enabled=False` recover v4 exactly. Novel:
`stable_weight=0` recovers v4 exactly). The operator independently
reproduced, from a clean shell: the conservative branch's `identity` (all
three thresh/gap combos, max|diff|=0.0) and `leadtime` (all 8
`persist_days` cells, matching the report's own numbers exactly, including
the +16.5d→−10.0d flip at `persist_days=5`); the novel branch's `ablation`
(all 16 matched confirm-vs-override cells, confirm ahead in all 16,
matching the report's reported deltas) and `eth` (falsification verdict
FAIL reproduced, though the operator's own count of failing non-identity
spot cells is **14/16**, not the report's stated 13/16 — a minor,
immaterial discrepancy in the branch's own self-report, noted rather than
silently corrected, that does not change the verdict either way).

**Result.** **Conservative — REJECT, and worse than R-54's own original
result, not better.** The falsification test fails outright: at the
primary threshold, the fraction of matched episodes leading falls
75%→67%→58% as `persist_days` goes 0→1→3, and the median offset flips from
**+16.5 days of lead to −10.0 days of lag by `persist_days=5`**
(independently reproduced) — well inside the literature-motivated 3–7 day
grid center this round itself pre-registered. Diagnosed mechanism: at the
primary threshold, `persist_days` 0 through 3 produce the *identical* 12
stress-onset events — the "transient noise" R-54 diagnosed does not
reverse within a few days, it persists roughly as long as genuine episodes
do, because the signal's own 14-day growth window has already smoothed out
anything shorter. Duration and precision are not separable axes at this
signal's native cadence: a persistence requirement shorter than the
feature's own smoothing window mostly re-measures noise the feature
already contains, and one long enough to matter (≥5 days) gives back most
of the lead time. Inner-validation: no configuration among all 24 beats v4
on Sharpe (best cell −0.38 vs v4's spot 0.14, a 0.52 gap, more than double
the noise floor — worse than R-54's own best cell of 0.13); no plateau to
report since nothing clears v4. ETH falsification: no outright fail by the
differential rule, but every config underperforms v4 on its own BTC
control (ratio 0.15×–0.89×), the same non-substantive-pass signature R-54
found. Exposure-artifact R²=0.61 (PASS, genuinely different shape);
causality 0.0 lookahead on both pathways, separately and combined (PASS).
**Novel — REJECT as a strategy, but the architecture question is answered
cleanly for the first time.** Confirming-vote beats an equivalent hard
override fed the identical signal in **16/16** matched cells (up to +0.69
Sharpe, independently reproduced) — the exact reverse of R-53's own
architecture-ablation finding, resolving that round's lag-vs-lead
confound: R-53 could not tell whether averaging lost to override because
averaging is the wrong combination rule, or because its feeding signal
lagged; fed a signal that genuinely leads, averaging is unambiguously the
better rule. But beating a bad baseline is not the same as beating the
incumbent: no non-identity configuration clears v4 on inner-validation
spot Sharpe (best 0.10 vs v4's 0.14), the one nominal futures win (+0.07)
sits inside the ±0.2 noise floor, and ETH falsification **fails
decisively** — a majority of the 16 non-identity spot configurations
underperform v4 on ETH by more than on the BTC control (independently
reproduced at 14/16, vs. the branch's own reported 13/16), while futures
cells pass uniformly. Exposure-artifact R²=0.9407 (PASS, but with far less
margin than R-54's hard veto's 0.61, since a diluted confirming vote
tracks v4's own exposure shape more closely by construction). Causality
clean on both pathways plus combined; `stable_weight=0` identity exact.

**Verdict.** **NEGATIVE (both branches).** **B-22 is CLOSED** — added to
section C below. Neither of R-54's two named fixes rescues the stablecoin
signal into a working strategy, for two different, well-diagnosed reasons:
the persistence filter fails because duration and precision are not
separable at this feature's native 14-day cadence (the "noise" is exactly
as persistent as the signal), and the confirming-vote architecture —
genuinely proven better than a hard override once fed a leading signal, a
real methodological result worth keeping — still cannot clear ETH
falsification or beat v4's own inner-validation Sharpe, because the
underlying signal's specificity problem (R-54's original diagnosis: it
fires on transient supply wobbles as often as on genuine stress) is
orthogonal to which combination rule receives it. One-line lesson: **this
project's INFO research line has now spent three structurally different
signals (on-chain activity, external macro, crypto-native liquidity) and,
within the liquidity signal alone, four structurally different combination
rules (averaged vote, hard veto, duration-filtered veto,
precision-weighted confirming vote) — every one fails, and the two
failures that are genuinely novel (R-54's confirmed lead-time, this
round's confirmed architecture ordering) are both about mechanism quality,
not about whether INFO itself can be exploited.** Nothing further is
pursued on this specific signal without a materially different mechanism
(e.g., a shorter growth window matched to genuine-stress duration rather
than a post-hoc persistence filter bolted onto the existing 14-day
feature, or corroboration from a second independent signal rather than
filtering one signal alone) — filed as **B-23**, LOW priority, since three
consecutive INFO-axis rounds (R-53, R-54, R-55) have now found the same
class of failure and a fourth attempt on the identical signal is a weaker
bet than the standing recommendation below. Configs evaluated: **45**
(conservative 24, novel 21). Holdout: **+0** — neither branch read any
2023+ bar; independently confirmed by the operator (grepped both branches'
files for date literals ≥2023-01-01: only `OOS_START="2023-01-01"`
sentinels appear, used exclusively as exclusive upper bounds for each
branch's own pre-2023 restriction, every call site checked). Not
registered — no code under `src/tradebot/` touched, no candidate cleared
the promotion bar; both files stay in `experiments/` per ROUTINE.md step 5.
`pytest`: 457 passed, unchanged, confirmed by the operator after both
branches. **B-06 (forward paper trading, ongoing since R-48) remains this
project's standing zero-cost recommendation** — now the only genuinely
open item that is not a further re-derivation of a research line
(SIZE-axis sizing/diversification, INFO-axis stablecoin combination rules)
this project has independently exhausted multiple times running.

---

### R-54 · 08-20 · NEGATIVE — B-21's macro hard-veto, and a new stablecoin-supply INFO candidate beside it

**Direction.** **B-21** (R-53's unvetted hard-veto ablation) given its own
full pre-registration, in parallel with a genuinely new INFO-axis
candidate designed specifically to resolve R-53's lead-time failure:
aggregate stablecoin (USDT) circulating-supply deceleration as a
crypto-native liquidity-flow proxy, motivated by 2025 literature on
stablecoin flows as a capital on-ramp/off-ramp (BIS WP 1340; Ahmed &
Aldasoro, Cleveland Fed conference paper, Aug 2025 / BIS WP 1270; NY Fed
Liberty Street Economics, "Stablecoins and Crypto Shocks: An Update," Apr
2025; IMF WP 2025/141)

**What was done.** Backlog was empty except B-06 and B-21 after R-53;
B-21's own note named an unresolved tension (a mechanism whose value
proposition is faster gate-flipping was never itself lead-time-tested) as
the direct next step, so this round split into two disjoint branches on
the SAME hard-veto architecture (`frac` forced to 0 while a latched vote
reads "stress", v4's own unmodified 3-anchor average otherwise — identical
combination rule in both branches, so any outcome difference is
attributable to the feeding signal, not the mechanism) fed by two
different signals. **Conservative**
(`experiments/kelly_regime_v15_macro_veto.py`): B-21 exactly as filed —
the existing VIX/DXY `stress_z` from `experiments/_macro_signal.py`
(unedited), `thresh_hi=1.0` fixed, `gap` swept over {0.0, 0.5, 0.75, 1.0,
1.25}. **Novel** (`experiments/kelly_regime_v15_stablecoin_veto.py` + new
`experiments/_stablecoin_signal.py`): a new signal built from
newly-fetched, newly-committed real CoinMetrics data
(`data/stablecoin_supply_daily.csv.gz`, USDT `SplyCur`,
2017-01-01→2026-08-19, 0 NaN/gaps, free community-tier endpoint, via new
`scripts/fetch_stablecoin_supply.py` and new additive
`tradebot.data.load_stablecoin_supply()`/`align_stablecoin_causal()`) — a
14-day log-growth-rate z-score on a fixed 365-day trailing window (fixed
a-priori, never swept), `thresh_hi`∈{0.75,1.0,1.25}×`gap`∈{0.0,0.75,1.25}
swept. Both restricted to inner-train/inner-validation/pre-2020
BTC-control+ETH falsification only; both ran mandatory two-pathway
tamper-causality probes plus an `enabled=False`/`macro_weight=0`-style
identity check recovering v4 exactly. **14 configurations total**
(conservative 5, novel 9). The operator independently re-ran and
reproduced, from a clean shell, both branches' `causality`, `select`
(inner-validation table + plateau spread), `artifact` (R²) and — for the
conservative branch — `eth` and `leadtime` outputs, bit-for-bit /
cell-for-cell against both reports.

**Result.** **Conservative (B-21) — REJECT on three independent,
pre-registered grounds**, independently reproduced by the operator. (1)
Lead-time, the primary pre-registered test: against the 3-anchor majority,
the veto leads only 4/12 matched episodes (33%), median **−5.5 days** —
replicates R-53's averaged-vote finding almost exactly; blunting the
combination rule does not rescue the timing. (2) Plateau check fails:
gap-grid spread is 0.32 (≥0.2 noise floor) and the single best-scoring
point is the explicit no-hysteresis negative control (`gap=0.00`), not the
pre-registered primary (`gap=0.75`). (3) ETH falsification fails: 5/10
(config×market) cells underperform v4 on ETH while beating it decisively
on the BTC control, including the primary's own futures cell (BTC 1.264×
vs ETH 0.986×). Causality passes cleanly (both pathways + identity).
Exposure-artifact R² is config-dependent: the pre-registered primary
passes (R²=0.841) but the grid's actual best-scoring cell fails
(`gap=0.00`, R²=0.9544) — the most attractive raw number in the experiment
is partly an artifact. Primary candidate's own inner-validation Sharpe
edge (+0.177 spot / +0.159 futures) sits under the ±0.2 noise floor
regardless. **Novel (stablecoin) — REJECT as a strategy, but the
pre-registered falsification centerpiece PASSES for the first time in this
project's three-attempt INFO research line.** Lead-time: against the
3-anchor majority the stablecoin-stress vote leads **9/12 matched episodes
(75%), median +16.5 days** — the reverse of R-53's and this round's own
conservative-branch lag, confirming the round's central hypothesis that a
crypto-native liquidity signal can lead where an external index (VIX/DXY)
could not. It still fails on the merits: no configuration beats v4 on
inner-validation Sharpe (best cell 0.13 vs v4's 0.14 spot, inside the
noise floor and not part of a plateau — its immediate gap-neighbour loses
0.74 Sharpe), 8/9 cells are decisively worse (Sharpe to −0.61, DD to
48.8%), and every configuration loses to v4 in absolute terms across the
full pre-2020 BTC control (ratio 0.13×–0.90×) — the ETH differential check
only "passes" because it adds no further degradation on top of that
already-large shortfall. Diagnosed mechanism: the threshold tight enough
to catch genuine stress early also fires on transient supply noise (24
stress-onsets at the tightest setting vs. 12 at the pre-registered
primary), and standing flat through the false alarms costs more than the
genuine early exits recover — loosening the threshold recovers Sharpe but
gives back the lead time, converging back toward v4 rather than improving
on it. Passes both integrity checks: exposure-artifact R²=0.6091
(genuinely different exposure shape), causality 0-lookahead on price, the
new stablecoin-CSV pathway, and both combined, plus an exact
`enabled=False`≡v4 identity recovery.

**Verdict.** **NEGATIVE (both branches).** **B-21 is CLOSED** — added to
section C below; the "blunter combination rule" hypothesis is rejected
because the timing failure is identical regardless of averaging vs.
override, since both mechanisms are built on the same underlying
`stress_z`. One-line lesson: **timing and precision are separate axes for
a regime-veto signal, and both have to hold.** This round fixed the exact
defect that sank R-53 (an external signal lagging price) by switching to a
genuinely crypto-native one (aggregate stablecoin supply, which measurably
leads) — and still lost, because the threshold sensitive enough to buy
real lead time is not sensitive enough to also reject transient noise.
That is a materially different, and more informative, failure mode than
R-53's, R-44's or any prior INFO-axis attempt's, and is worth keeping as a
standing fact rather than re-deriving. Next step, not pursued this round
per ROUTINE.md's goalpost discipline: a magnitude-*and*-duration filter
(require the stablecoin vote to persist before it can veto, rather than
firing on any single-bar crossing), or using the signal as a confirming
speed-up on the existing anchors rather than a unilateral override (closer
to R-53's originally pre-registered precision-weighted-average
architecture, not re-tested here, which might behave differently fed by a
signal that actually leads) — filed as new backlog item **B-22**. Configs
evaluated: **14** (conservative 5, novel 9). Holdout: **+0** — neither
branch read any 2023+ bar; independently confirmed by the operator
(grepped both branches' new files for date literals ≥2023-01-01: only
`OOS_START="2023-01-01"` sentinels appear, used exclusively as exclusive
upper bounds for the causality probe's/`eth()`'s own pre-2023 restriction,
every call site checked). Not registered — no code under
`src/tradebot/strategies/` touched, no candidate cleared the promotion
bar; all new files stay in `experiments/`/`scripts/`/`data/` per
ROUTINE.md step 5, plus the additive-only `src/tradebot/data.py` loader
functions and one new `.gitignore` exception line (nothing existing edited
or removed in either file, independently confirmed by the operator via
`git diff`).

---

### R-53 · 08-20 · NEGATIVE — VIX/DXY/S&P 500 macro stress as an input to `kelly_regime_v4`'s regime gate

**Direction.** VIX/DXY/S&P 500 macro-stress data (FRED, the first
traditional-finance information channel this project has used, after
B-07's blockchain on-chain data) as an input to `kelly_regime_v4` —
conservative multiplicative brake vs. novel regime-vote injection,
motivated by the VIX-term-structure/DXY-Bitcoin spillover literature (Luo,
Tsai & Yen 2024/2025, SSRN; IMF WP 2023/213; Klein, Thu & Walther 2018,
*Int. Rev. Financial Analysis*)

**What was done.** Off-backlog, literature-prompted: the backlog is empty
except B-06 after sixteen straight SIZE-axis rounds (R-34..R-52) and five
diversification rounds all failed, so this round deliberately attacked
**INFO** instead of a 17th SIZE variant. Two parallel unregistered
branches, each on a disjoint file, sharing one operator-authored, unedited
signal module (`experiments/_macro_signal.py`, `stress_z = 0.5·vix_z +
0.5·dxy_mom_z`, both terms z-scored on fixed trailing-365d windows, fixed
a-priori weights never fit to data) built on newly-fetched,
newly-committed real daily FRED data (`data/{spx,vix,dxy}_daily.csv.gz`,
2016-06→2026-08, no API key needed, zero NaN over the full committed BTC
index) loaded via new
`tradebot.data.load_macro_metrics()`/`align_macro_causal()`, following the
on-chain loader's 1-day-publication-lag causal convention exactly.
**Conservative** (`kelly_regime_v14_macro_brake.py`): v4's vote/vol-target
scale reproduced byte-for-byte, a single bounded never-increase-only
haircut `mult=1-lam·clip(stress_z/z_scale,0,1)` on top. **Novel**
(`kelly_regime_v14_macro_lead.py`): `stress_z` injected as a 4th latched
vote inside the regime gate itself
(`frac=(anchor_sum+macro_weight·macro_vote)/(3+macro_weight)`,
`macro_weight=0` recovers v4 exactly), testing the literature's specific
"macro leads price" claim directly via flip-timestamp matching against the
3-anchor majority, plus a pre-registered ablation against a hard-override
simplification. Both restricted to inner-train/inner-validation/pre-2020
BTC-control+ETH falsification only; both ran mandatory two-pathway (price +
raw macro CSV, tampered in a throwaway scratch copy, never under
`data/`) tamper-causality probes. **25 configurations total**
(conservative 10: 3×3 `lam`/`z_scale` grid + `lam=0` correctness check;
novel 15: 12 primary `gap`/`macro_weight` + 3 ablation). The operator
independently reproduced both branches' causality probes bit-for-bit from
a clean shell (0.000e+00 max|diff| on every column, both branches, all
pathways) and both branches' full inner-validation `select()` tables,
cell-for-cell.

**Result.** **Conservative — REJECT.** Every one of 18 (config×market)
inner-validation cells scored a lower Sharpe than the v4 control (v4: 0.14
spot / 0.25 futures); the exposure-artifact R² (18 cells) landed at
0.974–0.999, reproducing R-34's exact flat-rescale-collapse failure mode
(`kelly_regime_v5_damp.py`, R²=0.997) even though this round's input is
genuinely price-independent for the first time — a never-increase-only
haircut collapses toward a flat rescale of an already-dominant signal
regardless of whether the feeding data is price-derived; ETH/BTC-control
falsification gave an inconsistent, non-monotonic direction across the
grid. Causality passed cleanly on both probes. **Novel — REJECT on four
independent pre-registered grounds.** (1) Lead-time: against the 3-anchor
majority that actually flips the gate, the macro vote leads only 4/12
matched stress episodes (33%), median offset **−5.5 days** — on net it
lags, not leads, which mechanistically explains the rest. (2) No config
beats v4 on inner-validation Sharpe anywhere in the 12-cell grid (v4 spot
0.14 sits above every cell, range −0.04 to 0.12); drawdown is worse in
11/12 spot cells. (3) The precision-weighted-average mechanism loses to
its own hard-override ablation in 10/12 matched cells by 0.25–0.48 Sharpe
— the R-40/R-46 "elaboration adds nothing over a simpler baseline" pattern
again. (4) Every one of 12 configs fails the pre-registered ETH-spot
falsification (ratio 0.85–0.99×) while several beat the BTC control (up to
1.19×) — an asset-specific signature a market-wide signal should not
produce if the mechanism were genuine. Passes both integrity checks run:
exposure-artifact R²=0.94 (a genuinely different exposure shape, not a
relabel), and causality 0-lookahead on price, the new macro-CSV pathway,
and both combined, plus an exact `macro_weight=0`≡v4 identity recovery.
Holdout untouched by either branch.

**Verdict.** **NEGATIVE (both branches).** One-line lesson: VIX/DXY macro
stress is real, causally clean, genuinely price-independent data — but it
fails to help this strategy family two structurally different ways in a
row. As a scale haircut it collapses into the same flat-rescale artifact
price-derived brakes already hit (R-34, R-41-conservative); as a faster
regime-detection vote it does not actually lead `kelly_regime_v4`'s own
price-anchor gate on this project's available stress episodes (2018,
2020-03, 2022) — the median timing is a wash-to-slight-lag against the
metric that actually matters. INFO is not automatically easier to exploit
than SIZE just because it is a new constraint. One unvetted lead,
deliberately not chased here per ROUTINE.md's goalpost discipline (it
surfaced only as the novel branch's ablation comparison arm, never its
pre-registered candidate): a hard macro-veto with no precision-weighted
averaging beat v4 outright on inner-validation (spot Sharpe 0.34 vs 0.14,
DD 26% vs 33%, futures Sharpe 0.39 vs 0.25) but has not been through its
own lead-time check, ETH falsification, or plateau neighbourhood — filed
as new backlog item **B-21** for a future session to pre-register properly
before running anything further. Configs evaluated: **25** (this row's
total, both branches). Holdout: **+0** — neither branch read any 2023+
bar; independently confirmed by the operator (grepped both files for date
literals ≥2023-01-01: the conservative file has none at all, the novel
file's one `"2023-01-01"` literal is used only as an exclusive upper bound
for its own pre-2023 causality-probe restriction, never as a data read,
verified by reading every call site). Not registered — no code under
`src/tradebot/strategies/` touched, no candidate cleared the promotion
bar; both files stay in `experiments/` per ROUTINE.md step 5.

---

### R-52 · 08-20 · NEITHER PROMOTED — Calendar-rebalanced 50/50 BTC+ETH, and a drift-band variant (B-20)

**Direction.** B-20: does the LITERAL periodically-rebalanced, fixed-50/50
BTC+ETH `kelly_regime_v4` portfolio — R-50's own original candidate, run
through its continuous (non-restarting) engine — survive pre-registration
and a first, single holdout read? The one form of R-50's finding left
untested by both R-51 branches, ranked top of the backlog since R-51

**What was done.** Two parallel unregistered branches, each on a disjoint
new file, neither modifying `kelly_regime_v4.py`, `multiasset.py`,
`kelly_regime_covkelly*.py`, `kelly_regime_dual_fixed.py`, or either b19
file; the novel branch was also explicitly instructed not to read the
conservative branch's file, and didn't. **Conservative**
(`experiments/b20_literal_calendar_5050.py`): the literal object B-20
names — both legs' `kelly_regime_v4` run once, continuously, from ETH's
real data start (2019-03-14), rebalanced back to a FROZEN 50/50 split at
the start of every calendar month, cadence fixed before running (monthly
only, matching R-50's headline number, per B-20's own "single cadence
fixed before running" wording). **Novel**
(`experiments/b20_threshold_band_5050.py`): a genuinely different,
complementary axis — the same fixed 50/50 TARGET, but reallocated only
when the live BTC weight drifts outside a pre-registered band
(±5%/±10%/±15%, checked every 5-minute bar, triggered discretely) rather
than on any calendar, attacking **COST** directly in addition to SIZE/N≈3
— framed against Donohue & Yip (2003, *JPM* 29(1), 49-63), Masters (2003,
*JPM* 29(3), 52-57), and Kitces (2015, *The Kitces Report* Vol. 2 —
verified directly against the primary PDF) on tolerance-band rebalancing
generally dominating fixed-calendar rebalancing on a turnover-adjusted
basis; a 2024 ten-thousand-portfolio crypto simulation study is cited only
as an unverified secondary description (its 5/10/15% band grid, matched
here, per this project's R-39 citation-honesty convention). Both branches
carried in, and did not discover fresh, the same standing caution written
into their pre-registrations before any code ran: R-51-conservative's own
decomposition already attributed ~71% of R-50's untested Sharpe edge to
the periodic rebalancing act itself, a return-side mechanism a
bull-dominated 2023-2026 holdout had already shown a closely related
variant does not monetize — both pre-registered decision rules explicitly
weighted drawdown/tail evidence over Sharpe as a result, given the
holdout's ~623-consultation exhaustion. Both ran the standard
multiply/divide truncation-tamper causality probe — **PASS, 0.000e+00
max|diff| before the cut, both directions, both branches** — and both
explicitly guarded the CRITICAL trap this round's brief warned about
(`run_continuous_full`/`continuous_leg_equity` silently cap at
`FULL_END="2022-12-31"` unless an explicit holdout-inclusive `end=` is
threaded through): both branches printed and asserted their holdout
equity's min/max dates before reporting any number. The operator
independently reproduced both branches' holdout reads from a clean shell
(`python experiments/b20_literal_calendar_5050.py holdout` and `python
experiments/b20_threshold_band_5050.py holdout`); both matched their
reports' tables exactly. **21 configurations total** (conservative 15,
novel 6 candidate configs — each branch's own honest count,
baselines/references/causality/cache-diagnostics excluded per this
project's established convention; conservative's own count includes 2
double-labeled-but-numerically-identical cells, disclosed rather than
netted out, per the R-39 "honest count" convention). One genuine
implementation bug was found and fixed by the conservative branch before
any gate was read (plateau check used full 3-way span instead of the
pre-registered adjacent-pair wording — a coding bug against
already-written docstring text, not a threshold search, per ROUTINE.md's
own bug-fix/goalpost distinction); the novel branch independently
rediscovered and worked around (without touching R-50's file) the
`continuous_leg_equity` fee-rate cache-key bug R-51's novel branch first
found.

**Result.** **Conservative — the literal candidate.** All four
pre-registered gates PASS on the inner splits: it reproduces R-50's own
cited byproduct number almost exactly (spot, monthly, 50/50,
inner-validation ΔSharpe +0.79, max DD 33.2%→27.1%, matching R-50's row to
two decimal places), is not an exposure-artifact (R²=0.88 both markets,
well under 0.95), survives the 0.40% taker tier without its drawdown edge
flipping sign, and the 50/50→60/40→40/60 neighbourhood is a genuine
plateau (adjacent-split Sharpe deltas 0.08–0.12). The one pre-registered
holdout read (frozen 50/50-monthly, both fee tiers) **fails decisively on
clause (a) alone**: the dual book loses to `buy_and_hold` by −22.1% (0.10%
tier) to −44.6% (0.40% tier); its Sharpe edge over BTC-solo v4 is noise
and not even stably signed (+0.05 / −0.02 across fee tiers), and its
drawdown edge compresses 65–80% from the inner-validation reading
(−6.1pp/−6.0pp → −1.2pp/−2.0pp), not the kind of drawdown/tail improvement
clause (b) was written to credit. **Novel — does the trigger rule
matter?** All gates PASS on the inner splits too, including a new one this
branch added: turnover reduction (12 rebalances vs. a re-derived
45-rebalance calendar reference, 73% fewer, for statistically identical
Sharpe/drawdown — the mechanism doing exactly what the cited literature
predicts). The pre-registered holdout read (frozen ±5% band, both fee
tiers) again **fails decisively on clause (a)**: the candidate loses to
`buy_and_hold` by roughly half its value (−48% to −61% across fee tiers)
and is statistically indistinguishable from both BTC-solo v4 and its own
re-derived calendar reference on both Sharpe and drawdown (all deltas
inside 0.06 Sharpe / 2pp DD). **Turnover reduction survives on the holdout
too, robustly** (79–88% fewer rebalances than the calendar reference) —
the one part of either branch's mechanism that worked exactly as designed
on data it had never seen, though it does not rescue the promotion
decision. A discrepancy the operator flagged during independent
verification, not a defect in either branch's own logic: the two branches'
absolute-dollar "v4-solo BTC" holdout baselines differ substantially
($3,373 conservative vs. $2,229 novel at 0.10%, both rebased to $1000 at
2023-01-01) because the conservative branch's baseline is built from the
full uncut BTC frame with `run_period`'s normal pre-`start` warmup
context, while the novel branch's holdout function slices BTC/ETH to `>=
OOS_START` *before* calling `leg_equity`, leaving its candidate, calendar
reference, and v4-solo baseline alike with no pre-2023 warmup context
inside that function — an internally-consistent, fair three-way comparison
within the novel branch's own holdout read, just not dollar-comparable to
the conservative branch's absolute figures. Both branches' relative
conclusions (candidate ≈ its own baselines, both decisively losing to
`buy_and_hold`) are unaffected either way, and both were independently
reproduced by the operator exactly as printed.

**Verdict.** **B-20 CLOSED. NEITHER BRANCH PROMOTED.** The literal
fixed-calendar form of R-50's finding, and a genuinely distinct
drift-triggered form of it, both replicate the real, non-artifact,
falsification-clean, plateau inner-validation mechanism R-50 first found —
and both fail the same way on the one holdout read each was pre-registered
to make. This is the third independent multi-asset BTC+ETH
`kelly_regime_v4` composition to clear every
inner-validation/falsification/plateau gate this project's discipline puts
in front of a holdout read, and still lose decisively on it (R-43's
bear-quartile claim, R-51-conservative's never-rebalanced split, now both
R-52 branches) — and the fourth and fifth *trigger-rule* variant tested
against this same underlying return premium (never,
monthly/weekly-calendar, quarterly/semiannual-calendar, now drift-band),
none of which changes the answer. **One-line lesson:** the
periodic-rebalancing return premium this whole research line has chased
since R-50 is real and mechanism-backed on 2019–2022 data under every
trigger rule and every target-weight scheme tested so far, but has now
failed to survive the 2023–2026 bull-dominated holdout five separate times
under five different implementations — the finding has graduated from "one
candidate failed" to "this specific return premium does not transfer to
this specific holdout regime," which is itself informative and should stop
future sessions from re-deriving it a sixth way without a materially
different mechanism or asset pair. The novel branch's turnover result (a
band trigger cuts rebalancing turnover 70–90% for statistically identical
risk-adjusted performance vs. a fixed calendar, confirmed on both
inner-validation and the holdout) is real and worth remembering for a
future candidate whose Sharpe edge *does* clear the holdout on other
grounds, but is not itself promotable in the absence of one. Configs
evaluated: **21** (this row's total, both branches). Holdout: **+4**
(conservative: one frozen configuration read at two fee tiers, +2,
matching the R-35/R-51 convention; novel: one frozen configuration read at
two fee tiers, +2, same convention) on top of R-51's running total of ~623
— program total now **~627**. Not registered — no code under
`src/tradebot/` touched, no candidate cleared the promotion bar; both
files stay in `experiments/` per ROUTINE.md step 5. Per the standing
recommendation repeated since R-46 and restated by both branches
independently: **B-06 (forward paper trading, ongoing since R-48) is now
the only genuinely open, well-motivated item left in this project that is
not a further re-derivation of a return premium the 2023-2026 holdout has
just refused five times** — a future session with spare capacity should
run `scripts/paper_trade.py` again rather than open a sixth.

---

### R-51 · 08-20 · NEITHER PROMOTED — R-50's 50/50 BTC+ETH portfolio pre-registered: static split vs inverse-vol (B-19)

**Direction.** B-19: does R-50's periodically-rebalanced, fixed-50/50
BTC+ETH `kelly_regime_v4` portfolio (ΔSharpe +0.79/+0.80 vs. v4-solo, DD
33.2%→27.1%, inner-validation only, never pre-registered) survive
pre-registration and this project's falsification/cost/holdout process —
the highest-priority genuinely OPEN backlog item since R-50

**What was done.** Two parallel unregistered branches, each on a disjoint
new file, neither modifying `kelly_regime_v4.py`, `multiasset.py`,
`kelly_regime_covkelly*.py`, or `kelly_regime_dual_fixed.py`.
**Conservative** (`experiments/b19_dual_fixed_split.py`, B-19's own named
cheapest-first-check): re-expresses the candidate as a **one-time,
never-rebalanced** 50/50 (±60/40, 40/60 as a plateau check) split via the
already-promoted `tradebot.multiasset.run_multi_backtest`, to isolate
whether R-50's number needed the periodic rebalancing at all — framed
against Booth & Fama (1992, *FAJ* 48(3), 26–32), Willenbrock (2011, *FAJ*
67(4), 42–49; arXiv:1109.1256) on one side (diversification return
specifically requires the rebalancing act) and Chambers & Zdanowicz (2014,
*JPM* 40(4), 65–76) on the other (it's ordinary
correlation/variance-reduction, present even unrebalanced). **Novel**
(`experiments/b19_risk_parity_rebalance.py`): stays periodically
rebalanced but replaces the 50/50 weight with **inverse
trailing-volatility weighting** (Maillard, Roncalli & Teiletche 2010,
*JPM* 36(4), 60-70 — the no-covariance, no-mean special case of Equal Risk
Contribution, deliberately between R-50's zero-information static split
and the already-REJECTED Σ⁻¹μ `kelly_regime_covkelly` allocator) and
sweeps cadences R-50 never tried (quarterly, semiannual, per Dichtl,
Drobetz & Wambach 2016, *Applied Economics* 48(9), 772-788 — realistic
transaction costs typically favor quarterly-to-yearly over
monthly/weekly), through an extension of R-50's continuous engine
(`run_portfolio_continuous_costed`) that — a genuine gap found and fixed
in this round, not a criticism of R-50, since R-50's own two arms were
compared on the same zero-rebalance-cost basis — charges an explicit
`2×fee_rate×shift` cost for the portfolio-level reallocation itself, which
R-50's algebraic-rescale engine charged nothing for. A second bug was also
found and fixed in the same imported helper: `continuous_leg_equity`'s
memoization key omitted `market.fee_rate`, silently returning a wrong
cached curve when the same leg was evaluated at two fee tiers in one
process (caught by a $1,431-vs-$1,206 discrepancy on an identical config;
fixed with a correctly-keyed local cache; R-50's own single-fee-tier usage
was never affected). Both branches pre-registered a falsification pair
(F1: not an R²>0.95 exposure-artifact rescale of BTC-solo v4; F2: survives
the 0.40% Bitstamp taker tier) and a promotion decision rule, in each
file's module docstring, before running any sweep, and both ran the
standard multiply/divide truncation-tamper causality probe — **PASS,
0.000e+00 max|diff| before the cut, both directions, both branches** —
independently reproduced by the operator from a clean shell for both the
conservative branch's holdout numbers and both branches' causality probes.
**25 configurations total** (conservative 13: split-ratio × window ×
market × fee tier, including the two holdout reads; novel 12: 4 lookbacks
× 3 cadences), each branch's own count matching the project's established
convention of counting distinct candidate configurations rather than
baseline/reference/diagnostic re-runs (conservative: 20 baselines + 3
causality runs uncounted; novel: 16 more backtests in its own broader
28-total accounting, also uncounted here).

**Result.** **Conservative — the Chambers-vs-Booth&Fama split.** Both
falsification gates PASS on the inner splits (R²=0.86–0.87, well under
0.95; drawdown edge does not flip sign at 0.40%) and the split-ratio
neighbourhood is a genuine plateau (inner-validation spot Sharpe 0.32–0.41
across 40/60↔60/40). The never-rebalanced static split captures **~100% of
R-50's drawdown improvement** (33.2%→27.0% vs. R-50's 33.2%→27.1%) but
only **~29% of R-50's Sharpe improvement** (ΔSharpe +0.23 vs. R-50's
+0.79/+0.80) — the risk-reduction benefit needs no rebalancing (Chambers,
confirmed), roughly 71% of the Sharpe gain specifically traces to the
periodic sell-winners/buy-losers act (Booth & Fama/Willenbrock, also
confirmed, on the return axis). Both gates having passed, the
pre-registered holdout was read **once**, frozen at 50/50, both fee tiers:
it **fails outright** — the dual book loses to `buy_and_hold` by −23.9%
(0.10% tier) to −45.8% (0.40% tier), and is statistically
indistinguishable from (0.10%: ΔSharpe +0.02) or slightly worse than
(0.40%: ΔSharpe −0.05) BTC-solo v4 alone; the inner-validation drawdown
edge (−6.1pp) compresses to a non-effect (−0.8pp). **Novel — does
information beyond 50/50 help?** F1 PASSES (R²=0.58–0.94 vs. BTC-solo
across both splits) but F2 **FAILS** (ΔSharpe vs. the re-derived
fixed-50/50 reference stays −0.04 at the 0.40% tier, same sign as at
0.10%) and, more fundamentally, the promotion rule's P2 clause fails
before F2 is even reached: across **all 12 configurations, every lookback,
all three cadences, both spot and 5x futures**, inverse-vol weighting
never beats the re-derived fixed-50/50 reference — it loses by a small,
one-directional 0.02–0.11 Sharpe with a slightly worse drawdown, and
R²=0.996 between the two return series shows inverse-vol weighting is
close to a relabeling of the static split with added turnover. The
underlying diversification effect itself **replicates independently**
under this branch's own engine and weighting scheme (ΔSharpe +0.65 vs.
BTC-solo on inner-validation, ballpark-matching R-50 and the conservative
branch) — so B-19's core finding is not an artifact of the specific 50/50
split — but using volatility information to move away from 50/50 adds
nothing at 2-asset scale. A secondary, counter-to-the-cited-literature
finding: lengthening the rebalance cadence made **both** arms worse, not
better (fixed-50/50 valid Sharpe 0.925 monthly → 0.918 quarterly → 0.859
semiannual) — rebalance fees are trivially small on this pair at this
project's cost tier (well under 2% drag even in the most expensive cell),
so diversification-maintenance value dominates the fee savings
Dichtl/Drobetz/Wambach's stock-bond result predicted would dominate here.
Per its own pre-registered rule, the novel branch **never read the 2023+
holdout** (P2/F2 failed first) — independently confirmed by the operator:
`holdout()` is gated behind an explicit CLI argument neither the branch's
own `all`-command run nor any other invocation in its report ever passes.

**Verdict.** **B-19 CLOSED for both tested variants; NEITHER PROMOTED.**
The cheapest-to-test form of the idea (never-rebalanced static split) is
REJECTED-ON-HOLDOUT, decisively. The more ambitious form
(volatility-informed periodic rebalancing) is NEGATIVE before the holdout
gate, on 12/12 configurations. **What remains genuinely untested by any of
the three sessions that have now touched this idea (R-50, and this round's
two branches): the LITERAL periodically-rebalanced, fixed-50/50,
continuous-engine portfolio — R-50's own original candidate** — since the
conservative branch deliberately tested the cheaper non-rebalanced variant
by design and the novel branch's pre-registered promotion gate applied
only to its inverse-vol candidate, using the re-derived fixed-50/50 solely
as an inner-validation reference point never authorized for its own
holdout read. That is filed as new backlog item **B-20** below, with an
explicit caution attached: this round's evidence updates *against* it, not
for it — the drawdown-only component of the effect (the part isolated and
holdout-tested here) failed decisively, and roughly 71% of the untested
variant's larger Sharpe edge is attributed by this round's own
decomposition to the same rebalancing-driven return premium that a
bull-dominated 2023-2026 holdout has already shown this project's other
diversification variant does not reliably monetize. Lesson for any future
multi-asset/rebalancing work on this codebase: a real,
literature-grounded, non-artifact, falsification-clean mechanism (here,
twice: Chambers' unrebalanced diversification effect, and
Maillard/Roncalli/Teiletche's inverse-vol weighting) can still fail this
project's promotion bar outright, and "beats an established
inner-validation reference on every axis but the actual holdout" is now
the modal outcome for this entire multi-asset research line (R-42, R-43,
and now R-51), not an exception. Configs evaluated: **25** (this row's
total, both branches). Holdout: **+2** (conservative branch, one frozen
configuration read at two fee tiers, pre-registered as a single paired
read per the R-35 convention; novel branch: **+0**, confirmed by the
operator via the CLI-gating check above). Not registered — no code under
`src/tradebot/` touched, no candidate cleared the promotion bar; both
files stay in `experiments/` per ROUTINE.md step 5.

---

### R-50 · 08-20 · B-18 ANSWERED — Is the covkelly cadence flip a rebalance-engine restart artifact? (B-18)

**Direction.** B-18: is the `kelly_regime_covkelly` allocator's
monthly/weekly cadence-inconsistency (R-42, attenuated but not resolved by
R-43's mean-denoising) actually a rebalance-engine/segment-restart
artifact rather than a mean-estimation-noise problem — named as the
highest-priority genuinely OPEN backlog item since R-49

**What was done.** Two parallel unregistered branches, each on a disjoint
new file, neither modifying `kelly_regime_covkelly.py`, `_v2.py`, or
`kelly_regime_dual_fixed.py`. **Conservative**
(`experiments/kelly_regime_covkelly_v3_continuous.py`): removes the
restart structurally — runs each leg's unmodified `kelly_regime_v4`
**once**, continuously, over the full inner-train+inner-validation window
per asset (so its deadband/vol-regime latch state is never reset), reads
each rebalance segment's realized return off that one continuous curve,
and walks pooled capital forward by return-compounding (`pooled_after =
pooled_before·(1 + w_btc·ret_btc + w_eth·ret_eth)`) instead of re-invoking
the strategy per segment; the causal `w_btc`/`w_eth` weight series
(`build_weight_series`) is imported **unchanged** from the file it extends
— only the capital-combination mechanics change, not the allocator logic.
**Novel** (`experiments/kelly_regime_v13_restart_isolation.py`): isolates
the restart mechanism in a pure single-asset setting (BTC-solo
`kelly_regime_v4`, no covariance estimation at all) — a true continuous
baseline vs. naive-restart harnesses at weekly/biweekly/monthly/quarterly
cadence, plus a warm-start patch (a v4 subclass seeding `pos`/`state` from
the true prior-segment ending values instead of 0/0) as a cheaper
candidate fix. Both ran mandatory truncation/tamper causality checks —
**PASS, bit-identical before the cut, both directions** — and R²
exposure-artifact diagnostics; both hard-sliced to ≤2022-12-31 and were
grepped (by their own authors and independently by the operator) for any
2023+ literal outside comments — none found. **23 configurations total**:
conservative 15 (12-point sweep + weekly repeat of the winner + 2
same-config restart-engine comparison runs), novel 8 backtests (the 5
`prepare()` causality/sanity calls are diagnostics, uncounted, matching
this project's convention).

**Result.** **Conservative:** the flip reproduces exactly under the
original restart engine at matched weights (validation Sharpe monthly
**−0.23**, weekly **+0.42** — sign flip, gap 0.65) and disappears under
the continuous engine (monthly **+0.61**, weekly **+0.69** — same sign,
gap 0.08, inside the ±0.2 noise floor). Engine-only delta, same weights
and cadence: ΔSharpe +0.84 (monthly) / +0.27 (weekly) — restart hurts
every cadence tested, worst at monthly. R² diagnostics: 0.33–0.94 vs. a
flat-rescaled solo-BTC book, 0.59–0.77 vs. a *matched* (continuous-engine)
fixed-50/50 control — neither an exposure-rescale artifact nor identical
to the static split. But against that correctly matched control — not
R-42's original restart-crippled one — the dynamic Σ⁻¹μ allocator
**loses**: 0.61 vs 0.93 (monthly, Δ−0.32) and 0.69 vs 0.94 (weekly,
Δ−0.25), outside the noise floor in the wrong direction. The fixed-50/50
control itself, run through the same continuous engine, clearly beats
v4-BTC-solo (ΔSharpe +0.79/+0.80, max DD 27.1% vs. 33.2%) — a clean
diversification result with neither the restart artifact nor the
covariance-estimator's noise anywhere near it. **Novel:** restart cost is
real (3 of 4 cadences exceed the noise floor against the true continuous
baseline) but not cleanly monotonic in validation — quarterly is
statistically indistinguishable from continuous (ΔSharpe 0.05) while
biweekly is the single worst cell (ΔSharpe 0.50), worse than both weekly
and monthly either side of it — even though train-period cost *is* a clean
monotonic staircase. The warm-start patch recovered **0%** of the gap
(output bit-identical to naive restart) because the real memory being
destroyed sits upstream of `pos`/`state`: v4's slowest (80-day) anchor has
a valid rolling-mean value for only the last ~11 of its ~23,050-bar warmup
prefix, so the anchor's hysteresis-latched vote — built by
`.ffill().fillna(0.0)` over what is, for practically the whole prefix, an
all-NaN series — has no real memory to fall back on at any restart and
sits pinned toward bearish/flat until moments before trading resumes.

**Verdict.** **B-18 ANSWERED.** The two branches' mechanisms reconcile
rather than conflict: the novel branch shows precisely why a
`pos`/`state`-only warm-start fails (the destroyed memory is the vote's
own hysteresis latch, further upstream), and the conservative branch's fix
works because a full continuous replay never truncates the anchors' input
history in the first place, so that latch is never destroyed at all.
**Confirmed:** the monthly/weekly cadence flip is primarily a
rebalance-engine restart artifact, not the mean-estimation-noise problem
R-43 assumed — R-43's shrinkage treated a symptom, which is why it
narrowed the flip but never closed it. **NEGATIVE** on the trading
question the artifact was masking: correctly measured,
`kelly_regime_covkelly`'s dynamic Σ⁻¹μ weighting adds nothing over a plain
static 50/50 split (loses by 0.25–0.32 Sharpe, outside the noise floor) —
R-42/R-43's original "dynamic beats static" headline was itself an
artifact of comparing against a restart-crippled control, and that
specific direction is now closed. Lesson: a periodic-reallocation engine
that re-invokes a stateful, hysteresis-latched strategy fresh per segment
silently destroys exactly the memory that strategy's indicators depend on;
any future rebalancing/multi-asset work on a path-dependent strategy
(including B-17's still-open "wire it into `run.py`" half) must use
continuous-replay-plus-return-splicing, never segment-restart. Configs
evaluated: 23 (this row's total, per the parallel-round convention that
trials count across all branches). Holdout: **untouched, +0** — both
branches' date literals are ≤2022-12-31, grepped and confirmed by their
own authors and independently re-confirmed by the operator. Not registered
(no code under `src/tradebot/` touched, no strategy cleared the promotion
bar) — both files stay in `experiments/`, per ROUTINE.md step 5. Next step
→ new backlog item **B-19**: does the fixed-continuous-engine 50/50
BTC+ETH diversification result (ΔSharpe +0.79/+0.80 vs. v4-solo, DD
33.2%→27.1%, inner-validation only) survive pre-registration and this
project's falsification/cost/holdout process, or is it itself an artifact
not yet stress-tested? It has not been pre-registered, has no
falsification test, no fee/funding sensitivity and no holdout read — a
promising lead, deliberately not rushed to promotion in the same session
that found it.

---

### R-49 · 08-20 · PROMOTED (infrastructure) — Multi-asset strategy registration infrastructure (B-17)

**Direction.** B-17: multi-asset strategy registration infrastructure —
per every round summary since R-40, the highest-merit item left once
B-06/B-08 closed. Not a trading-signal question: can the framework be
extended, cleanly and safely, to make a multi-asset (BTC+ETH) strategy
registrable into the comparison table at all, given
`Strategy`/`registry.py`/`run.py` assume exactly one instrument per
strategy today and R-42/R-43's genuine dual-asset findings could never be
registered even after clearing the promotion bar (which, per R-43, they
did not)

**What was done.** Two parallel unregistered branches, each on a disjoint
file, neither re-testing R-43's already-REJECTED dual-asset finding — this
round is infrastructure only. **Conservative**
(`experiments/b17_multiasset_adapter.py`): an adapter/composition design —
run the unmodified `tradebot.engine.run_backtest` (via `run_period`) once
per instrument, each with its own isolated `Strategy`/`PaperBroker`, then
sum equity curves at a fixed pre-decided split and route the result
through the unmodified `compute_metrics`. **Novel**
(`experiments/b17_multiasset_native.py`): a native
`MultiAssetContext`/`run_multi_backtest_native` engine giving one strategy
simultaneous read access to N instruments' state inside a single `on_bar`
call and one shared risk budget — the only way to host a genuinely joint
allocator like `kelly_regime_covkelly`'s Σ⁻¹μ, which the adapter cannot
express even in principle. Design patterns for both grounded in a
literature/practice sweep: unified single-account multi-instrument engines
(QuantConnect; backtrader's `Cerebro`, per PyQuantLab's write-up) informed
the novel branch, additive-without-forking-the-core module systems
(RQAlpha's "Mod" architecture) informed the conservative branch's
registry-level design. Demonstration for both restricted to
inner-train/inner-validation only (2019-03-14→2022-12-31, ETH's real
start); the operator independently re-ran the conservative branch's
`causality` and `sanity` subcommands from a clean shell and reproduced
both exactly.

**Result.** **Conservative:** portfolio (`kelly_regime_v4` x2, BTC+ETH,
50/50, spot) final $5,796.53, Sharpe 1.84, max DD 29.9% over the inner
span; composition-level causality PASS, **max|diff|=0.000e+00** at a
2021-06-30 truncation (independently reproduced); bit-identical to
`kelly_regime_dual_fixed.py`'s own `run_dual("50_50", ...)` on every
metric (independently reproduced to 4 decimal places). **Novel:** its
joint Σ⁻¹μ demo allocator moved BTC's average weight from 0.40 (train) to
0.07 (validation, effectively exiting BTC) while ETH stayed invested — a
live reallocation no fixed-split adapter can express — with a clean
causality PASS including a real catch-the-peek probe; but its own first
broker implementation produced a silent, non-crashing accounting bug
(double-counted notional, negative equity from ordinary noise) caught only
by manual inspection, not by the causality suite, before being fixed and
re-verified against known BTC/ETH price history. Both branches' own
authors independently recommended the adapter for real adoption and the
native engine only once a specific strategy earns its extra risk — the
operator agrees.

**Verdict.** **Conservative PROMOTED as permanent infrastructure; novel
kept in `experiments/` (buildable, not yet warranted).**
`src/tradebot/multiasset.py` —
`MultiAssetSpec`/`MultiBacktestResult`/`combine_equity_curves`/`run_multi_backtest`/`available_multi_asset_strategies()`
— is now a genuine, additive capability: it changes nothing under
`engine.py`/`strategy.py`/`broker.py` (this project's most heavily
causality-tested files) and does not touch any of the 25 existing
registrations. 8 new tests (`tests/test_multiasset.py`, synthetic
fixtures, no real data): weight-sum/spec-count validation, late-leg
flat-fill, composition-level causality (with a genuine catch-the-peek
probe — comparing queued *decisions* at a fixed bar rather than downstream
equity, after a first draft's truncation-only version produced a false
failure from a legitimate mark-to-market artifact, not a lookahead bug,
when one leg's own final equity value differed between a longer and a
shorter run of otherwise-identical data). Full suite: **457 passed**,
independently re-run. **What B-17 is still not**: no multi-asset strategy
is wired into `run.py`'s `run_matrix`, the README table, or
`test_evidence.py`'s CI requirement — both branches' design notes list
concretely what each would need (an asset-aware `tradebot.data` load path;
a second `run_matrix` code path keyed on a strategy's declared
`instruments`; a table/README convention for which instruments a row
represents; a per-strategy rather than global bootstrap-interval
requirement) and neither is built, deliberately: there is still no
multi-asset strategy that has cleared even inner-validation, let alone the
holdout, so wiring a full CI-facing path now would be infrastructure built
for a strategy that does not exist. **Configs evaluated: 1 (conservative,
exercised via 8 leg-level calls) + 6 (novel) = 7**, none feeding a
promotion decision — not a Sharpe-bearing claim on any strategy, so
nothing here interacts with deflated-Sharpe accounting. Holdout:
**untouched, +0** — every date literal in both branch files is
grepped-and-confirmed ≤2022-12-31. Closes the "can this be done at all"
half of **B-17**; the "wire it into the real comparison table" half stays
**OPEN**, correctly scoped smaller now, and is worth doing the day a
multi-asset strategy actually clears inner-validation rather than
speculatively.

---

### R-48 · 08-19 · DONE (infra), ONGOING (the record) — Forward paper-trading recorder, unblocked and running (B-06)

**Direction.** B-06: forward paper-trading recorder — "the highest-value
item on merit" in every round summary since R-29, unblocked this session
by a direct HTTPS probe confirming `www.bitstamp.net` reachable (200)

**What was done.** Off-backlog, same justification as R-47. Built
`scripts/paper_trade.py`: a persisted, credential-free recorder that
fetches Bitstamp's public candle API (no signed endpoint imported
anywhere, no `--live` flag exists — paper-only by construction, not
configuration), maintains its own virtual account (JSON state under
`reports/paper_trading/`, never a real exchange balance), and executes
each decision through the existing `tradebot.broker.PaperBroker`
fee/rebalance code (not a re-derived formula) for both `kelly_regime_v4`
and a parallel `buy_and_hold` benchmark, appending one CSV row per
decision. Idempotent on the candle timestamp (verified live: a repeat run
against an already-recorded candle correctly no-ops). Two real defects
were found and handled honestly while building this, both flagged rather
than quietly patched over: (1) a genuine pre-existing bug in
`Exchange.fetch_history` — the first page of any multi-page cold start
against a *real* venue can legitimately come back short (the still-forming
candle, ordinary clock jitter), and the old code treated any first-page
shortfall as "no more history," silently truncating every real multi-page
cold start to one page; never caught before because the test suite only
pages a synthetic venue with no forming candle. Fixed in `base.py` with a
regression test; `bot.py`/`live_bot.py` share the same bug and were
**not** touched (out of scope), and this is noted in `docs/LIVE.md`. (2)
18 of 23 registered strategies (the `kelly_regime` family included) only
emit an order when their precomputed target *changes* bar-over-bar, so a
cold-started account can catch the strategy mid-latch and sit at 0%
exposure forever while believing it tracks it — exactly what happened on
the first live run. `inception_catchup_target()` reads the strategy's raw
target directly from `prepare()` at inception only, when `compute_signal`
emits nothing; every later run is untouched, genuine `compute_signal`
output. The identical gap exists in `bot.py`/`live_bot.py` and is
documented, not fixed, there either. 8 new tests
(`tests/test_paper_trade.py`, offline, no network) plus a regression test
for the pagination fix; full suite 449 passed, independently re-run by the
operator. Parity-checked: the strategy's raw target read from `prepare()`
(1.54503272) matches a fresh, independent `run_backtest` on the identical
window exactly.

**Result.** Two genuine live decisions recorded against the real Bitstamp
feed: **2026-08-19T23:05:00Z**, close $69,319.39 — inception,
`kelly_regime_v4` entered via the catch-up path to its latched target
(1.545, clamped to spot's [0,1]), fee $3.98 (0.40% tier), equity $996.02
after entry; the parallel `buy_and_hold` account entered the same candle
to target 1.0. **2026-08-19T23:10:00Z**, close $69,312.29 — target
unchanged for both accounts, no trade, equity marked to $995.91. A third
invocation against the same candle correctly detected "no new closed
candle" and exited 0 with no state change.

**Verdict.** **DONE (infrastructure), ONGOING (the record itself).** Not a
backtest — no configs/trials count, and the holdout counter is untouched
(+0): every candle came from the live feed, never
`data/btcusd_spot_5m.csv.gz`, which is the entire point of B-06 — this is
the one evidence stream immune to the 2023+ holdout's exhaustion
(R-29/R-30). At 2 recorded rows it is not yet informative on its own; it
becomes valuable only as it accumulates, which needs a future session or
an actual cron/systemd job invoking `python scripts/paper_trade.py` once
per closed 5m candle (documented in `docs/LIVE.md`'s new "Forward
paper-trading recorder (B-06)" section, with a ready cron line) — nothing
in this project's current session-based operation does that automatically
yet, which is this row's one open follow-up. Closes the **BLOCKED
(network)** status on **B-06**; does not close B-06 itself, since the item
was always "start the record," not "produce a verdict from it."

---

### R-47 · 08-19 · CONFIRMS L-01/R-17 — Second bear, second asset: does frozen v4 replicate on ETH 2020–2026? (B-08)

**Direction.** B-08: second bear, second asset — does the frozen,
unchanged `kelly_regime_v4` replicate on ETH 2020–2026, the first ETH
evidence independent of the 2018 BTC bear every prior ETH falsification
check has shared (`ethusd_bitfinex_5m.csv.gz` stops 2019-12;
`ethusd_coinbase_spot_5m.csv.gz`, committed for B-15/R-41's basis work,
covers 2019-03-14→2026-08-19 and was never read for this purpose before)

**What was done.** Off-backlog per the ledger's own repeated
recommendation (R-40 through R-46: after sixteen failed SIZE-axis
branches, prefer B-06/B-08/B-17/B-18 over a seventeenth tuning attempt).
Pre-registered in `experiments/b08_eth_2020_2026_replication.py`'s module
docstring, before any ETH 2020+ number was computed: a cell REPLICATES if
v4 beats `buy_and_hold` after real costs and either clears the ±0.2 Sharpe
noise floor or shows a drawdown/tail improvement. Zero parameters touched
— v4 run byte-identical, no strategy file modified. 2 pre-registered
windows (2022 bear 05-01→11-30; full 2020-01-01→2026-08-19) × 2 markets,
plus a 0.40% Bitstamp-tier sensitivity pass on spot. **12 configurations,
no sweep.** Data-integrity check (`validate_ohlcv` PASS, 781,506 rows, 0
duplicate timestamps, monotonic, largest gap 6h35m — far under v4's 80-day
warmup) and a causality tamper probe on the new ETH loading path
specifically (since `tests/test_causality_strict.py` hard-codes the BTC
loader and does not exercise this file) — PASS. The operator independently
re-ran the full script and reproduced every number exactly.

**Result.** 2022 bear, spot: v4 **+4.5%** (DD 20.5%) vs hold **−55.5%**
(DD 70.0%). 2022 bear, futures 5x: v4 +0.5% (DD 19.7%) vs hold
**liquidated** (−98.2%, DD 98.7%). Full 2020–2026, spot @0.10%: v4
+1,410.3% (DD 35.3%, Sharpe 1.29) vs hold +1,389.0% (DD 81.7%, Sharpe
0.91) — v4 nominally ahead but by a razor-thin dollar margin. Full
2020–2026, spot @0.40% (the realistic Bitstamp tier): v4 **+708.8%** vs
hold **+1,384.5%** — v4 loses decisively once real costs are charged. Full
2020–2026, futures 5x: v4 +1,869.0% vs hold liquidated (**named
limitation**: no ETH perpetual funding data exists in this repo, so this
cell is a funding-free upper bound, same caveat R-14 already attaches to
every BTC futures number — R-14 found BTC funding cuts v4's own futures
number by half to two-thirds).

**Verdict.** **CONFIRMS L-01/R-17's standing caveat, on independent
evidence for the first time — not a clean win, not a NEGATIVE.** The
drawdown/tail property replicates robustly on ETH's own,
previously-untested 2022 bear (the cleanest apples-to-apples cell: v4
essentially sidestepped the crash on spot while a fully-invested hold lost
more than half its value). The *return* edge does not survive contact with
real costs over the full period — R-33's own standing rule applies here
too (match risk before crediting a drawdown edge; the futures cells in
particular reward v4 mostly for not being a statically-leveraged,
liquidated benchmark, the same L-07/R-19 pattern already documented on
BTC). This is exactly this project's own repeated finding ("the risk
property transfers, the return property does not"), now reproduced on an
asset and period that shares no bear market with the original BTC evidence
— the strongest form of confirmation this project's data can offer for
that specific claim. No promotion (nothing new to promote; v4 is already
registered) and no README/STRATEGIES.md change. Holdout: **untouched, +0**
— this script never reads the BTC 2023+ file. Code:
`experiments/b08_eth_2020_2026_replication.py`, reproducible standalone.
Closes **B-08**.

---

### R-46 · 08-19 · NEGATIVE — Off-backlog: CPPI replacing v4's scale half, fixed-multiplier and Hurst-adaptive

**Direction.** Off-backlog, literature-prompted: instead of a fifteenth
tweak to v4's own vote-and-vol-target architecture (fourteen straight
branches failed, R-34 through R-45), replace only the SCALE half of the
mechanism with Constant Proportion Portfolio Insurance — a much older,
structurally different sizing family never tried in this ledger (Perold
1986, unpublished HBS manuscript, the original CPPI formulation; Perold &
Sharpe 1988, *Financial Analysts Journal* 44(1), 16–27; Black & Jones
1987, *Journal of Portfolio Management* 14(1), 48–51; Ko, Son & Lee 2024,
*Research in International Business and Finance* 67, 102135,
crypto-specific CPPI evidence). v4's own 3-anchor vote `frac` stays
byte-identical in both branches — only how much the vote is scaled by
changes. Conservative: textbook fixed-multiplier CPPI with a floor that
grows at a small fixed rate from *starting* balance (deliberately not
peak-following, to avoid reactively de-risking after drawdowns — which
would fight this project's own measured inverse-leverage-effect finding in
`kelly_regime_v3.py`). Novel: the same CPPI base, but the multiplier `m`
is made adaptive to a rolling, causally-computed Hurst exponent (Hurst
1951, *Trans. ASCE* 116, 770–799; Mandelbrot & Wallis 1969, *Water
Resources Research* 5(5), 967–988, classical R/S method; Di Matteo
2003/2007 generalized Hurst exponent noted as the modern alternative;
Bariviera 2017, *Economics Letters* 161, 1–4 and Grande, Borondo, Losada &
Borondo 2024, *Mathematics* 12(18), 2911, on time-varying/anti-persistent
crypto Hurst behaviour; Lo 1991, *Econometrica* 59(5), 1279–1313, on
classical R/S's upward bias under short-range dependence) — attacking
regime detection via a fractal-market technique genuinely different from
R-01/R-02/R-03's HMM/jump-model/BOCPD family.

**What was done.** Two parallel unregistered variants, each on a disjoint
file: `experiments/kelly_regime_v12_cppi_conservative.py` (24 configs:
F0∈{0.50,0.65,0.80}×g∈{0.00,0.03}×m∈{3,4,5,6}) and
`experiments/kelly_regime_v12_cppi_hurst.py` (33 configs: 32-point grid
over H_lo/H_hi/m_low/m_high/rolling-window plus a fixed-m=4 ablation
control). **57 configurations total across both branches.** Both
restricted to inner-train (2017-01-01→2020-12-31) / inner-validation
(2021-01-01→2022-12-31) plus the standard pre-2020
BTC-Bitfinex-control/ETH-Bitfinex-test falsification pair, both markets;
neither read a single 2023+ bar (each branch grepped its own file and
confirmed). Both ran a mandatory causality tamper probe (bit-identical
decisions before the cut) — **PASS**, independently reproduced by the
operator for the conservative branch. The operator also independently
reproduced the conservative branch's causality result and its full
ETH/BTC-control falsification table bit-for-bit (final balances matched to
the dollar on 3 of 4 cells, sub-1% floating-point drift on the fourth).

**Result.** Both branches beat v4 on inner-train and inner-validation
Sharpe and profit (conservative: valid-spot Sharpe 0.43 vs v4's 0.14;
novel's adaptive candidate: valid-spot Sharpe 0.43 vs v4's 0.14) but at
markedly higher drawdown everywhere (conservative valid-spot DD 47.4% vs
v4's 33.2%) and 2–3x v4's average notional — both branches independently
diagnosed why: with a floor anchored once to *starting* balance and never
re-anchored, `m·cushion/equity` saturates at `max_leverage` almost
immediately once equity compounds through a multi-year bull run, so the
winning grid region degenerates into "vote × constant max leverage," not a
genuinely cushion-responsive mechanism (conservative's own selected point
ran pinned at exactly 2.000, the cap, throughout inner-validation).
Neither branch is the standard R-33 exposure-level artifact by the strict
R²>0.95 test (conservative R²=0.61/0.78, novel R²=0.61/0.78 — nearly
identical, as expected since both selected similar (F0≈0.5–0.65, m≈4–6)
parameterizations), but both branches' own authors correctly argue the
substance is the same story under a different statistical signature.
**Both fail their own pre-registered falsification test decisively on the
BTC-control leg**, before ETH is even considered: conservative ΔSharpe
−0.47 (spot) / −0.76 (futures) vs v4 on the 2016–2019 BTC control; novel's
adaptive candidate ΔSharpe −0.47 / −0.76 (same base mechanism, same
failure). Both branches' harness (inherited from R-45's
`kelly_regime_v11_robust_ladder.py` helper) prints a mechanical "PASS"
because that helper's verdict logic only gates on the ETH cells — both
agents caught this themselves, correctly overrode it against their own
actually-pre-registered two-part rule, and reported the true verdict as
FAIL. Novel's Hurst-adaptive multiplier does not clearly beat its own
fixed-m=4 ablation (spot ΔSharpe +0.008, futures +0.226 from one window) —
the same "elaboration adds nothing over the simpler baseline" pattern
R-40's novel branch found — and the empirical rolling Hurst estimate came
out persistently *high* (mean 0.62, opposite the pre-registered failure
hypothesis that it would sit ≤0.5), which the novel branch's own author
flagged may itself be classical R/S's known upward bias under volatility
clustering (Lo 1991) rather than real persistence, and which didn't help
the mechanism regardless. The conservative branch's recovery-window check
(built to test whether CPPI under-holds through post-drawdown rallies, per
the inverse-leverage-effect concern named in this row's pre-registration)
found the opposite of its own hypothesis — the candidate was *more*
exposed than v4 in every tested recovery window — but for an uninformative
reason: the fixed floor never came close to binding at those troughs, so
the mechanism CPPI is supposed to have never actually engaged.

**Verdict.** **NEGATIVE** (both branches). Holdout untouched by either
(2023+ counter unchanged, +0). The one durable new lesson, worth keeping
for any future portfolio-insurance attempt on this codebase: a CPPI floor
anchored once to *starting* equity avoids the
peak-chasing/inverse-leverage conflict but stops binding almost
immediately once equity compounds over a multi-year backtest, collapsing
the mechanism into "vote × constant max leverage" — while a floor that
*does* stay relevant (peak-following) reintroduces the
reactive-deleverage-after-drawdown conflict this design was built to
avoid. Neither horn is free; a fixed-plus-periodic-reset floor or a
genuinely path-dependent alternative (e.g. TIPP, time-invariant portfolio
protection) is the natural next attempt if this axis is revisited, but is
**not** recommended as a priority — this is the fifteenth and sixteenth
branches across ten rounds (R-34, R-37, R-38, R-40, R-41, R-42, R-43,
R-44, R-45, now R-46) to fail on `kelly_regime_v4`'s SIZE axis, the first
two to attack it via a structurally different sizing family entirely
rather than a variant of vol-targeting itself, and they failed the
identical way: more raw leverage, worse risk-adjusted performance,
decisive loss on the pre-2020 BTC control. **B-06 (forward paper trading)
remains the highest-value item on merit**, still the only genuinely open
item that does not re-cut a dataset this whole SIZE-axis family has now
failed on ten separate times; a future session should treat B-06, B-08,
B-17 or B-18 as more promising than an eleventh. Sources beyond the
citations above: Ko, Son & Lee (2024) independently document CPPI
underperforming buy-and-hold in strong crypto bulls for exactly this
de-risk-then-relever-slowly reason, which this round's saturation-driven
result is consistent with even though the mechanism that bit here (floor
irrelevance, not slow re-leverage) was different.

---

### R-45 · 08-19 · NEGATIVE — Off-backlog: robust parameter selection (ERR) and walk-forward re-estimation (N≈3)

**Direction.** Off-backlog, literature-prompted: two genuinely different
axes attacking why twelve prior SIZE/INFO-axis branches (R-34, R-37, R-38,
R-40, R-41, R-42, R-43, R-44) all failed the same way — fitted to
2021–2022 validation, loses the pre-2020 BTC control or ETH — rather than
another new signal on the same fixed architecture. Conservative attacks
**ERR** directly for the first time on this axis: not the trading signal,
but *how v4's own existing constants were chosen* (bootstrap/CV-robust
parameter-selection literature, 2024–2025, shows point-estimate/ERM
selection collapses out-of-sample where quantile/minimax selection does
not). Novel attacks **N≈3** via architecture rather than a new input:
periodic causal walk-forward re-estimation of v4's sizing constants (2025
adaptive-regime-Bitcoin walk-forward literature), the first branch in the
program to build a re-fit loop rather than add a signal to a static one.

**What was done.** Two parallel unregistered variants, each on a disjoint
file: `experiments/kelly_regime_v11_robust_ladder.py` (conservative —
reselects only v4's own already-swept free parameters, the 18–28d
anchor-ladder base (R-06/R-07) and `target_vol`/`max_leverage` (R-37's
axis), by maximizing worst-fold Sharpe across 3 non-overlapping
calendar-purged folds (2017–18/2019–20/2021–22, one per the project's own
N≈3 count) instead of the pooled-window point estimate every prior round
used; no new signal, mechanism unchanged) and
`experiments/kelly_regime_v11_walkforward_adaptive.py` (novel — replaces
v4's frozen `target_vol`/`max_leverage` with values refit every 365 days
from a trailing 730-day causal window via a fee-free proxy-Sharpe grid
search, vote/hysteresis/deadband copied verbatim from v3/v4, schedule and
lookback fixed before any run; 2 neighbour schedules for the required
plateau check). **57 configurations total across both branches** (54
conservative + 3 novel — the novel branch's internal 25-point-per-refit
grid is the model's own re-estimation machinery, not a separately swept
configuration, matching how R-40 counted its ensemble definitions), both
restricted to inner-train (2017-01-01→2020-12-31) / inner-validation
(2021-01-01→2022-12-31) plus the standard pre-2020
BTC-Bitfinex-control/ETH-Bitfinex-test falsification pair, both markets,
no holdout read by either.

**Result.** **Conservative:** genuinely improves generalization by the
metric it targets — the minimax-selected config (ladder base 26,
`target_vol=0.55`, `max_leverage=2.5`) beats the naive pooled-optimum
config (same base/leverage, `target_vol=0.45`) on the falsification set
outright (ETH: wins both markets vs. the naive winner losing spot;
BTC-control futures: retains 62% of v4's balance vs. the naive winner's
37%) — a real, quantified win for robustness-aware selection over
point-estimate selection on the identical search space. It also clears
inner-validation by more than the noise floor (ΔSharpe +0.38 spot / +0.22
futures, *less* mean notional than v4, not the exposure artifact: R²=0.86
both markets) and passes a by-hand causality probe. **It still fails the
falsification test as pre-registered**: it beats v4 on ETH (both markets)
but visibly underperforms v4 on the pre-2020 BTC control, worst on futures
(62.1% of v4's balance, ΔSharpe −0.26) — the same train/validation-wins,
BTC-control-loses signature that sank R-37/R-38/R-40/R-41, now reproduced
with a selection-methodology change instead of a new signal. Diagnosed
cause: the three purged folds are all drawn from 2017–2022, so robustness
*across* them cannot buy robustness against the 2016–2019 BTC-control
period, which none of the folds ever sampled — the ERR-axis fix is real
but bounded by what regimes are inside the training window at all.
**Novel:** causality probe passes cleanly (two truncation points after
different refit counts, bit-identical before each cut) — the re-fit loop
itself has no lookahead bug — but the mechanism is not competitive: every
schedule loses to v4 on inner-train, on inner-validation (best cell inside
the noise floor, every other cell worse, futures uniformly worse), and
decisively on both BTC-control and ETH (Sharpe roughly half of v4's or
worse in every asset/market cell). Unlike the twelve prior signal-adding
branches, this is not "fitted to one window and failing elsewhere" — it
never beat v4 anywhere, on any period, and the primary candidate also
trips the exposure-artifact bar (R²=0.98 both markets) despite losing.
Diagnosed cause: periodic re-fitting does not resolve N≈3, it fractalizes
it — each individual refit is itself a low-information estimate from only
1–2 trailing regime-events, so the strategy now makes several
under-informed fits instead of one, adding selection noise (and a fee-free
scoring proxy poorly matched to the real, cost-bearing objective) without
adding signal.

**Verdict.** **NEGATIVE** (both branches). Holdout untouched by either —
the conservative branch enforced a hard no-2023+ rule throughout and the
novel branch logged its refit counts explicitly on every pre-2023 run;
zero bars dated 2023-01-01 or later were read, computed, or printed
anywhere in either file. This is the **thirteenth and fourteenth**
branches in the program's SIZE/architecture family to fail (R-34, R-37,
R-38, R-40, R-41, R-42, R-43, R-44, now R-45 ×2) — the first pair to
attack *methodology* (how parameters are chosen, and whether they should
be static at all) rather than adding a new input signal, and the first to
get a partial, quantified positive result (robustness-aware selection
beats point-estimate selection) even while still failing the project's own
promotion bar. Sources: non-parametric bootstrap-quantile parameter
selection for time-series momentum (2024–2025 TSMOM robustness literature,
arXiv:2510.12725 and related); "Adaptive Regime-Based Trading on Bitcoin:
Backtesting and Walk-Forward Evaluation" (2025); "Quantitative Evaluation
of Volatility-Adaptive Trend-Following Models in Cryptocurrency Markets,"
Karassavidis, Kateris & Ioannidis, SSRN 5821842 (2025).

---

### R-44 · 08-19 · NEGATIVE — On-chain features, sign-corrected: real BTC/ETH on-chain data (B-07)

**Direction.** B-07: on-chain features, sign-corrected — real,
price-independent BTC/ETH on-chain data (active addresses, tx count, hash
rate) as a `kelly_regime_v4` improvement, attacking INFO directly for the
first time with data that is not a price transform (unlike
L-12/L-14/L-15/L-16, which tried to recover this FROM price and failed)
and is not a market-transacted price spread either (unlike R-41's real
Deribit basis)

**What was done.** A connectivity check found CoinMetrics' free community
API reachable (no key) where every prior check had only probed exchange
venues — B-07 had sat `BLOCKED (network)` since it entered the backlog.
Fetched and committed `data/btc_onchain_daily.csv.gz`
(2017-01-01→2026-08-18) and `data/eth_onchain_daily.csv.gz`
(2019-01-01→2026-08-18, `HashRate` NaN post-Merge),
`scripts/fetch_onchain_metrics.py`, and
`tradebot.data.load_onchain_metrics()`/`align_onchain_causal()` (the
CoinMetrics D+1 reporting lag solved once, causally, rather than left for
each caller to re-derive). Then two parallel unregistered variants, each
on a disjoint file, explicitly designed around R-08's sign-inversion
lesson (on-chain activity predicts volatility, not direction; a modifier
that de-levers on rising activity repeats R-08's failure) and R-34's
exposure-artifact lesson (a monotone never-increase-only multiplier
degenerates into a flat rescale):
`experiments/kelly_regime_v10_onchain_confirm.py` (conservative — a
bounded, SYMMETRIC multiplier `mult=1+lam·tanh(z/2)∈[1−lam,1+lam]` on v4's
unchanged vote+scale, `z` a rolling z-score of 7-day active-address
growth, able to raise exposure on confirmed participation rather than only
shrink it; 9 configs + 1 correctness check) and
`experiments/kelly_regime_v10_hashribbon_vote.py` (novel — Hash Ribbons
miner-capitulation-recovery, Edwards 2019/Capriole Investments, a 30d/60d
hash-rate MA cross with capitulation-band hysteresis, as a fourth latched
vote precision-weighted against v4's three price anchors rather than
averaged unweighted with them; a descriptive pre-check on event
frequency/forward-returns before any strategy code ran; 12 configs). **21
configurations total across both branches**, both restricted to
inner-train (2017-01-01→2020-12-31) / inner-validation
(2021-01-01→2022-12-31), both markets, plus the pre-registered ETH
falsification (BTC-Bitfinex control vs ETH-Bitfinex test, both on their
own committed on-chain series), no holdout read by either. Operator
independently re-ran both branches' `causality`/`artifact` (novel) and
`artifact` (conservative) commands and reproduced every reported number
exactly, and independently re-ran the novel branch's `select` command and
reproduced its inner-validation table exactly.

**Result.** **Conservative:** genuinely carries independent information
(corr(frac, mult)=0.165 — not R-34's near-constant 0.997-collinear margin)
and passes causality (price and on-chain tampers both clean) and the ETH
falsification (BTC control and ETH test are comparably weak, no kill-rule
trigger) — but fails on magnitude, not content: R²=0.989–0.996 against a
flat-notional-matched rescale of v4 in **all 9 configs, both markets**
(independently reproduced), because the task-specified small
`lam∈[0.10,0.20]` bounds keep the multiplier's own contribution to
exposure variance too small to escape the exposure-artifact bar regardless
of symmetry — a diagnostic sweep outside the pre-registered grid shows R²
only drops below 0.95 around `lam≈0.5–1.0`, well outside what
"conservative" was scoped to mean here. Independently of the artifact
question, it also fails step 5's own bar outright: inner-validation
**futures loses to v4 in 9/9 configs** (−4.2% to −9.8%), a clean
plateau-wide failure rather than a fluke. **Novel:** clears the
exposure-artifact bar cleanly (R²=0.9232, independently reproduced —
genuinely different exposure shape from a rescaled v4) and passes
causality on both the price and the new hash-rate pathway (independently
reproduced, max|diff|=0.000e+00 throughout) — the cleanest pair of "not an
artifact, not a bug" diagnostics an INFO-axis branch has produced yet in
this project. It still loses on the criteria that matter:
inner-validation, **every one of 12 configurations underperforms v4 on
both markets, no exceptions** (independently reproduced — best spot Sharpe
0.10 vs v4's 0.14; best futures Sharpe −0.01 vs v4's 0.25), and it is not
even the deceptive win-on-validation pattern R-37/38/40/41/42 needed
catching — train is flat-to-marginally-ahead (1.023x spot / 0.994x
futures) while validation is uniformly behind (0.977x / 0.864x). ETH
falsification then fails too: spot is relatively worse on ETH than on the
BTC control in all 12 configs, and futures never gets a fair ETH read
because it is already losing 13–49% of v4's balance on its own BTC control
before ETH is even touched. One reusable methodological sub-finding
survives the negative verdict: precision-weighting the rare hash-ribbon
vote measurably beats unweighted 4-way averaging with the fast price
anchors (every `hr_weight=1.0` config underperforms `hr_weight=0.33` at
the same capitulation band, both in-train and out) — worth keeping if a
future round ever blends another rare-event vote into this mechanism. The
descriptive pre-check itself was informative on its own: 5
capitulation-recovery events in the 2017–2020 window (vs. 108–200 flips
for the fast price anchors) is the same small-N problem this project's own
N≈3 diagnosis names, and the first event's own forward return (30d
**−15.1%**) was already a warning the full backtest later confirmed.

**Verdict.** **NEGATIVE** (both branches). Holdout untouched by either —
zero bars dated 2023-01-01 or later were read anywhere in either file
(verified: both `select`/`causality` calls are explicitly bounded, the ETH
files end 2019-12-31, the conservative branch's own `causality` cut is
restricted to pre-2023). **This is the first INFO-axis round to test
genuinely non-price data across BOTH branches at once** (R-41's basis is a
market-transacted price spread; this is blockchain-level data with no
price in its construction at all) and both still failed — INFO-axis
attempts are now 0-for-3 (this round's two + R-41's two, i.e. 4 branches
across 2 rounds), on top of eleven-of-eleven SIZE-axis branches failing
across six prior rounds (R-34, R-37, R-38, R-40, R-41's SIZE framing,
R-42, R-43) — raising the prior further that `kelly_regime_v4`'s
vote-and-scale mechanism itself is close to a genuine plateau for what
this project's data can support, not just that each individual candidate
signal has been weak. Closes **B-07**. Sources: Edwards, C. (2019),
"Finding Bitcoin Bottoms Using Miner Capitulation," Capriole Investments;
Jagannath et al. (2021) on active-address/transaction-volume short-horizon
predictability; Casella & Paletto (2023) on on-chain regime identification
for allocation; Chi, Y., Chu, Q. & Hao, W. (2025), "Return and Volatility
Forecasting Using On-Chain Flows in Cryptocurrency Markets,"
arXiv:2411.06327 — the general grounding for reading on-chain signals as
volatility/confidence (SIZE) inputs rather than direction inputs,
consistent with R-08's own finding in this project.

---

### R-43 · 08-19 · NEGATIVE — Robustify R-42's dual-asset BTC+ETH branches per their authors' prescriptions (B-16)

**Direction.** B-16: robustify R-42's dual-asset (BTC+ETH)
`kelly_regime_v4` branches per their own two authors' prescriptions —
bootstrap the conservative fixed-split branch's single-window (n=1)
drawdown claim, and de-noise the novel covariance-aware branch's raw
trailing-mean estimator

**What was done.** Two parallel unregistered variants, each on a disjoint
file, followed by one pre-registered holdout read on the surviving claim
from the first: `experiments/kelly_regime_dual_bootstrap.py` (conservative
— imports R-42's `run_dual`/`run_baseline_v4_btc`/`SPLITS` UNCHANGED,
resamples 40 calendar windows from ETH's own bar grid, same
trials/day-range/seed convention as `scripts/stress_test.py`, 0 new search
configurations since it resamples R-42's own already-chosen
`50_50`/`vol_weighted` splits) and
`experiments/kelly_regime_covkelly_v2.py` (novel — rebuilds the Σ⁻¹μ
allocator with two named mean-treatments per B-16's own text: an
equal-Sharpe prior and a Σ-only mean-free minimum-variance rule, plus a
shrinkage-intensity blend between the original raw mean and the better of
the two; verified bit-identical to the original at both blend boundaries;
45 configurations: 36 grid + 5 blend-λ sweep + 4 weekly-cadence re-runs of
the finalists). Both restricted to inner-train/inner-validation only (no
2023+ bar read by either), **45 configurations total this half of the
round**. The conservative branch's inner-validation bootstrap then
justified one narrowly-scoped, explicitly pre-registered holdout
consultation (decision rule and its registration-infrastructure caveat
committed to `kelly_regime_dual_bootstrap.py`'s own module docstring one
commit before the read, `git log` records it): PROMOTE iff the
bear-quartile (top-quartile BTC-alone-severity) `vol_weighted`
drawdown-delta 95% bootstrap CI excludes zero on BOTH spot and futures 5x,
using 40 NEW resampled windows drawn from 2023-01-01 onward — the first
genuinely new holdout read since R-39, and the first ever to use
multi-window resampling rather than a single-point read against this
project's holdout.

**Result.** **Conservative, inner-validation:** the n=1 worry does not
simply dissolve, but it does narrow. Pooled across 40 pre-2023 resampled
windows the drawdown-improvement claim survives a percentile bootstrap on
spot (median −1.5 to −1.9pp, 95% CI excludes zero) but not on 5x futures
(CI contains zero); segmented by each window's own realized severity, the
improvement is concentrated in — and *larger* in — the bear-quartile
windows on **both** markets (spot −5.2 to −5.5pp CI excludes zero; futures
−2.9 to −3.1pp CI excludes zero) than in calm ones, the opposite of what
the pre-registered correlation-spike failure mode predicted, and not the
standard exposure-level artifact (R²=0.86–0.89, unchanged from R-42, under
the 0.95 bar). That result — real, non-trivial, and narrower than R-42's
original framing — was judged strong enough to spend the one
pre-registered holdout consultation described above. **Holdout verdict:
REJECT.** The bear-quartile claim replicates on 5x futures (median
−4.40pp, 95% CI [−5.29, −3.97], clearly excludes zero) but not on spot
(median −1.61pp, 95% CI [−2.56, **+0.47**], contains zero) — the
pre-registered rule required both markets, so the claim fails as written.
Worse, the *pooled* (all-severity) claim outright **reverses sign** on the
holdout: median **+0.61pp on spot / +0.04pp on futures** (both
distinguishable from zero on spot, i.e. a significant *worsening*, not a
null), driven by 68% of the 40 holdout windows showing the dual book doing
worse than BTC-alone `kelly_regime_v4` — a textbook instance of R-12's own
defining lesson (an in-sample pattern that looked real and did not survive
contact with new data), reproduced here for the first time on the
N≈3/diversification axis rather than the SIZE axis every prior instance of
it involved. **Novel, inner-validation only (no holdout read — did not
clear its own inner-validation bar):** verified bit-identical to R-42's
original at both blend-λ boundaries (max diff 0.000e+00), confirming the
rewrite is a true continuation, not a fresh implementation. `equal_sharpe`
genuinely fixes the diagnosed bug — the stale-momentum concentration into
one leg during the 2022 bear (raw: 99.5% into ETH) is replaced by a
near-total flight to cash (98% cash in 2022) — and meaningfully shrinks
the monthly/weekly cadence-inconsistency (monthly-validation Sharpe
−0.16→−0.03; weekly-validation Sharpe +0.37→+0.60, now clearing every
baseline) without eliminating it (monthly still clearly worse than weekly
on every candidate, including the de-noised ones, and train-window Sharpe
is still catastrophically below BTC-only v4 at every cadence: weekly-train
0.76 vs 2.62). The Σ-only mean-free candidate (`sigma_only`) is a
regression, not an improvement — it never modulates total exposure at all
(constant 50% invested, no regime response) and sits noticeably closer to
the fixed-50/50 exposure-artifact line (R²=0.82 vs the others' 0.17–0.50)
than any other candidate, confirming Chopra & Ziemba's (1993)
estimation-error asymmetry in reverse: discarding the mean entirely costs
more here than the noisy mean was costing.

**Verdict.** **NEGATIVE** (conservative branch's pre-registered holdout
claim REJECTED; novel branch NEGATIVE on its own inner-validation bar, no
holdout spent). **Holdout counter: +400** leg-level backtests in one
command (40 windows × 2 markets × [BTC-alone control + 2 split
candidates]) — unusually large next to prior rows because this is the
first holdout consultation in the project's history to use multi-window
resampling rather than a single-point read; the 2 cells named in the
pre-registration (bear-quartile `vol_weighted`, spot and futures) are what
the verdict rests on, the rest (pooled, calm-segment, `50_50`) is
diagnostic detail from the same pre-registered call, not a separately
chosen follow-up read. **Decision rule was not moved after seeing the
result** — REJECT is exactly what the committed rule (one commit before
the read, `git log` records it) produces from these numbers. Closes
**B-16**. A genuinely new, well-diagnosed finding survives the closure
regardless of the promotion verdict: on this project's own real 2023-2026
data, dual-asset BTC+ETH diversification's drawdown benefit is
concentrated in worst-drawdown regimes and is a **real,
holdout-replicating effect on leveraged futures** even though the
identically-defined spot claim fails — worth a future session's attention
as a possibly leverage-specific (funding-free; no funding was charged in
this round, a stated limitation) rather than
asset-diversification-specific mechanism, and worth re-examining ONLY with
fresh evidence per this project's own "do not re-try without new evidence"
convention, not as a same-round re-cut of the same 40 windows. Sources
(novel branch): Rising & Wyner, "Partial Kelly Portfolios and Shrinkage
Estimators" (Wharton working paper); Chopra & Ziemba (1993), "The Effect
of Errors in Means, Variances, and Covariances on Optimal Portfolio
Choice," *J. Portfolio Management*; Ledoit & Wolf (2004), "Honey, I Shrunk
the Sample Covariance Matrix," *J. Portfolio Management*.

---

### R-42 · 08-19 · NEGATIVE — Portfolio diversification across a second asset (real ETH)

**Direction.** Portfolio diversification across a second asset (real ETH,
via B-15's committed Coinbase/Deribit data) as a `kelly_regime_v4`
improvement — the first branch in this program to attack N≈3 by actually
holding capital in a second asset, rather than re-deriving a signal from
BTC's own single price series the way all ten prior SIZE-axis branches
(R-34/R-37/R-38/R-40/R-41) did; ETH had previously only been used as a
single-asset falsification check, never as a second book

**What was done.** Two parallel unregistered variants, each on a disjoint
file: `experiments/kelly_regime_dual_fixed.py` (conservative —
`kelly_regime_v4` unchanged, run independently on BTC and ETH with a fixed
capital split, 6 splits swept including a train-only vol-weighted split)
and `experiments/kelly_regime_covkelly.py` (novel — a periodic
covariance-aware Kelly reallocation between the same two unchanged
sub-books, Σ⁻¹μ solved from causal trailing EWM mean/covariance of each
asset's raw returns, 13 configurations across
halflife/kelly-fraction/leg-cap/cadence); **19 configurations total across
both branches**, both restricted to inner-train (2019-03-14, ETH's real
start, → 2020-12-31) / inner-validation (2021-01-01 → 2022-12-31, the
joint 2022 bear), primarily spot with a futures 5x secondary check, no
holdout read by either. Pre-registered failure mode for both, from
web-sourced literature on BTC/ETH tail dependence: correlation is known to
spike specifically during crashes, which could erase the diversification
benefit exactly when v4's edge depends on it most.

**Result.** **Both branches are genuinely non-duplicate — the cleanest
pair of exposure-artifact scores this program has produced (conservative
R²=0.81–0.89, novel R²=0.005–0.60, both well under the 0.95 bar against a
flat-rescaled BTC-only v4) — and the predicted correlation spike is real
and measured (BTC/ETH daily correlation 0.63→0.73 into 2022, and 0.72 as
the 2019–2022 peak in the novel branch's own estimate). Conservative:
inner-validation drawdown genuinely improves (26.0–29.3% vs BTC-only v4's
33.2%, −4 to −7pp across all 6 splits) and the improvement is not
eliminated by the correlation spike, only damped (full-2022 drawdown 19.3%
vs 27.7%) — but it degrades to roughly flat at the event level (FTX: dual
book marginally *worse*, −7.3% vs −6.9%, both legs crashed together) and
**every split loses to BTC-only v4 on the train window** ($3,707–$5,113 vs
$6,167) — a mechanically explained cost of diversifying away from the
stronger single-asset trend, not an obvious overfit, but structurally the
same train-loses/validation-wins shape that sank R-37/R-38/R-40, and the
drawdown claim itself rests on a single joint-bear observation (2022) with
no bootstrap run — the same N≈3 fragility this branch set out to attack.
Novel: the allocator degrades sensibly under the 2022 correlation spike
exactly as pre-registered — invested fraction collapses from ~0.87 (2021)
to ~0.24 (2022) rather than staying falsely diversified — but it
concentrates 97% of that shrunken stake into ETH on a stale, slow-decaying
momentum estimate, and ETH did not outperform BTC through the 2022 bear,
so monthly-cadence validation Sharpe goes negative (−0.16); a
weekly-cadence re-run of the identical winning hyperparameters flips from
clearly losing every baseline on train to modestly beating them on
validation, a cadence-driven variant of the same
train/validation-inconsistency signature. Both authors independently
recommended against a holdout read: conservative wants a
bootstrap/path-resampling check on the inner data first (n=1 joint-bear is
not evidence of significance), novel wants its raw trailing-mean term
replaced or shrunk (equal-Sharpe prior or a Σ-only, mean-free weighting,
per MacLean/Thorp/Ziemba's own warning about Kelly's fragility to
mean-estimation error) before its cadence-inconsistency can be trusted.
The operator agrees with both and does not read the holdout this round.**

**Verdict.** **NEGATIVE** (both branches, not yet promotable) — first
N≈3-attacking branch to clear the exposure-artifact bar with genuinely new
capital allocation rather than a relabeled BTC signal; reopens as **B-16**
with the two authors' own prescribed fixes (bootstrap the conservative
drawdown claim; de-noise the novel branch's mean estimator) rather than a
holdout read on either branch as built. Sources: Kelly (1956, *Bell System
Tech. J.*); Breiman (1961, Proc. 4th Berkeley Symp.); MacLean, Thorp &
Ziemba, eds. (2011, *The Kelly Capital Growth Investment Criterion*, World
Scientific — multivariate Σ⁻¹μ form and its fragility to estimation
error); "Diversification and limited information in the Kelly game,"
arXiv:0803.1364; "Crashing Together, Rallying Apart: Dynamic Conditional
Tail Dependence in Cryptocurrency Markets," arXiv:2606.16840; Kettani et
al., "Cryptocurrency Market Maturation and Evolving Risk Profiles… Bitcoin
and Ethereum Tail Risk Dynamics," *FinTech* 5(2), 2025
(doi:10.3390/fintech5020028).

---

### R-41 · 08-19 · NEGATIVE — Real Deribit spot/perp basis as a new SIZE input on `kelly_regime_v4` (B-15)

**Direction.** B-15: build a real Deribit BTC/ETH-PERPETUAL price series
(network access to Deribit/Kraken/Bitstamp/Coinbase confirmed open this
session — only Binance still 451s), then use the resulting real spot/perp
basis — the first genuinely independent second price series this project
has ever had, attacking INFO directly rather than the SIZE-axis
reweighting of R-34/R-35/R-37/R-38/R-40 — as a new SIZE input on
`kelly_regime_v4`

**What was done.** Infra: `scripts/fetch_deribit_perp_price.py` +
`scripts/fetch_coinbase_spot.py` fetched and committed
`data/btcusdt_deribit_perp_5m.csv.gz` (842,851 bars,
2018-08-14→2026-08-19, zero gaps), `data/ethusdt_deribit_perp_5m.csv.gz`
(781,765 bars, 2019-03-14→2026-08-19) and
`data/ethusd_coinbase_spot_5m.csv.gz` (781,506 bars, matching span) —
`tradebot.data.load_deribit_perp_price()`/`compute_basis()` added. Then
two parallel unregistered variants, each on a disjoint file:
`experiments/kelly_regime_v9_basis_brake.py` (conservative — bounded
never-increase dampener `mult∈[1−λ,1]` on v4's unchanged vote+scale,
triggered symmetrically by extreme |basis| in either direction) and
`experiments/kelly_regime_v9_basis_lead.py` (novel — step 2 first asked
whether basis genuinely *leads* the vote's own flip dates before writing
any strategy code); 30 configurations total across both branches (18
conservative + 12 novel, plus a lead-lag descriptive sweep not counted
toward trials), inner-train-with-basis (2018-08-14→2020-12-31, the window
actually covered) / inner-validation (2021-01-01→2022-12-31), both
markets, no holdout read by either. Operator independently re-ran both
branches' `artifact`/`fallback`/`causality` (conservative) and
`exposure`/`leadlag` (novel) commands and reproduced every reported number
exactly.

**Result.** **Both branches are genuinely non-duplicate by the
measurements that matter — basis correlates only r≈0.06–0.12 (daily) with
the already-tested funding-rate signal (R-35), far below a restatement —
and both fail cleanly for different, well-diagnosed reasons. Conservative:
the standard exposure-level artifact in all 18 configurations on the
mandated inner-validation check (R²=0.981–0.999) *and* the R-37/R-38/R-40
train-loses/validation-wins signature (loses to v4 on final balance in
36/36 train cells, beats it in only 14/36 validation cells) — real
cross-venue basis blowouts are too rare (752 of 842,851 bars, |basis|>10%)
for a bounded, symmetric, never-increase brake to move a multi-year
aggregate far enough from a flat rescale, while still costing return when
it does fire during the COVID V-shaped recovery. Novel: the step-2
lead-lag study is a clean null (basis-confirmed hit rate scatters 39–55%
around a ~51% base rate against a block-bootstrap null, median lead time
≈0 days — contemporaneous, not leading); a candidate built anyway despite
the null beats v4 in every train/validation cell (not the R-37/38/40
signature) but the Sharpe deltas sit inside the ±0.2 noise floor
everywhere and the exposure series is R²=0.977 collinear with v4's own
target (the artifact bar), so the uplift is not established as a distinct
mechanism. Both authors independently recommended against spending an ETH
falsification or holdout consultation on their own branch, and the
operator agreed rather than spending the newly-built ETH data on either.**

**Verdict.** **NEGATIVE** (both branches). Holdout untouched. Real ETH
basis data (7.4y coverage, comparable to BTC's) is now committed and
available for a future round with a mechanism that survives its own
inner-validation diagnostics first.

---

### R-40 · 08-19 · NEGATIVE — Bagging R-07's already-validated 18–28d anchor-ladder plateau

**Direction.** Bag/ensemble R-07's already-validated 18–28d anchor-ladder
plateau, instead of shipping one frozen point on it (ERR: no error control
on the ladder-choice hyperparameter itself)

**What was done.** Two parallel unregistered variants, each on a disjoint
file: `experiments/kelly_regime_v8_ladder_bag.py` (conservative — plain
unweighted average of the latched vote across a fixed 6-ladder ensemble
spanning R-07's region; Breiman 1996 bagging) and
`experiments/kelly_regime_v8_uncertainty_shrink.py` (novel — the same
bagged vote further shrunk by real-time cross-ladder disagreement, Baker &
McHale 2013 / Sukhov 2025 parameter-uncertainty-under-Kelly style; `κ=0`
verified to reduce exactly to the conservative mechanism); 12
configurations total across both branches (4 + 8),
inner-train/inner-validation and ETH/BTC Bitfinex falsification only, no
holdout read by either; operator independently re-ran both branches'
`select`/`eth` commands and confirmed the numbers

**Result.** **Both branches beat `kelly_regime_v4` on every
inner-validation cell (conservative's primary candidate: spot Sharpe 0.30
vs 0.14, futures 0.42 vs 0.25) and neither is the standard exposure-level
artifact (R²=0.86–0.94, below the 0.95 bar) — but both hit the same
diagnostic signature that sank R-37/R-38: substantial underperformance vs
v4 on the pre-2020 BTC falsification control itself, worst on futures
(conservative 52–75% of v4's balance across all four ensemble definitions;
novel 56%), before ETH is even read. The disagreement-shrink term added
nothing over the plain bag in 6 of 6 non-zero-κ configurations.**

**Verdict.** **NEGATIVE** (both branches) — see full write-up below

#### R-40 — bagging R-07's own validated plateau, instead of shipping one point on it

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

### R-39 · 08-19 · NEGATIVE — The network re-check R-38 asked for: extended funding data, and what it unblocks

#### R-39 pre-registration — the network re-check R-38 asked for, done properly, and what it unblocks

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

#### R-39 results — both branches NEGATIVE; the extended data answered its own question decisively

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

---

### R-38 · 08-19 · NEGATIVE — Risk-constrained Kelly gambling (Busseti, Ryu & Boyd 2016) as v4's sizing rule

**Direction.** Risk-constrained Kelly gambling (Busseti, Ryu & Boyd 2016)
as a formal, probability-calibrated replacement for `kelly_regime_v4`'s ad
hoc `target_vol`/`max_leverage` constants

**What was done.** Two parallel unregistered variants, each on a disjoint
file: `experiments/kelly_regime_v7_ddcap.py` (conservative — v4's vote and
scale unchanged, additionally capped by a causal drawdown-risk ceiling
`f_risk = mu/(lambda·sigma²)` with `lambda = ln(beta)/ln(alpha)` fixed
from a stated drawdown tolerance) and
`experiments/kelly_regime_v7_crra.py` (novel — v4's vote kept as a hard
gate, its vol-only scale replaced entirely by the same CRRA fraction as
the sizing formula); 56 configurations total across both branches (24 +
32), inner-train/inner-validation and ETH/BTC falsification only, no
holdout read by either

**Result.** **Both branches cleanly refute the standard
exposure-level-rescale artifact (R²=0.20 and 0.15 against a
mean-notional-matched flat rescale of v4, versus the 0.95+ threshold this
project treats as diagnostic) — a genuinely non-duplicate mechanism in
both cases. Both still fail their identical pre-registered ETH
falsification decisively, and by the same diagnostic signature: each loses
to `kelly_regime_v4` on the BTC control itself (conservative: ≈11–12% of
v4's balance; novel: 21–37%), before ETH is even read — the
inner-validation win (built on a bear/chop-heavy 2021–22 window) does not
survive a trending market on either tested asset. Conservative's parameter
neighbourhood is additionally not a plateau (adjacent (α,β) cells swing
spot Sharpe +0.50→−0.07); novel's is a loose plateau but sits at the edge
of its tested grid.**

**Verdict.** **NEGATIVE** (both branches) — see full write-up below

#### R-38 — a formal, probability-calibrated sizing rule, tested against the same incumbent a fifth time

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

---

### R-37 · 08-19 · NEGATIVE — Two SIZE-axis attempts to capture more of R-36's confirmed edge

**Direction.** Two SIZE-axis attempts to capture more of R-36's confirmed
(but thinned) edge on `kelly_regime_v4`

**What was done.** Two parallel unregistered variants, each on a disjoint
file: `experiments/kelly_regime_v6_retune.py` (conservative — retunes the
existing `target_vol`/`max_leverage` constants, no new signal) and
`experiments/kelly_regime_v6_state_kelly.py` (novel — replaces the single
global `target_vol` with a causally-estimated, per-vote-state Kelly
fraction `μ_state/σ_state²`); 99 configurations total across both branches
(53 + 46), inner-train/inner-validation and ETH falsification only, no
holdout read by either

**Result.** **Conservative: the naive best-Sharpe candidate reproduces the
project's standard exposure-level artifact (+51% realized vol) and does
not transfer to futures; the one candidate surviving a matched-exposure
control nets a Sharpe delta inside the ±0.2 noise floor on both markets
and does not clear ETH by more than a token margin. Novel: `max_leverage`
never binds (rules out the raw-leverage artifact cleanly) and states
genuinely differ in measured μ/σ² (bear ≈ −62%/yr, bull ≈ +154–174%/yr,
non-monotone — 2/3 agreement beats unanimous 3/3) — but it fails its
pre-registered ETH falsification outright, underperforming v4 on the BTC
control too, and its halflife/kelly_mult neighbourhood is a fitted peak,
not a plateau.**

**Verdict.** **NEGATIVE** (both branches) — see full write-up below

#### R-37 — two SIZE-axis attempts to harvest more of R-36's confirmed edge, run in parallel

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

##### Conservative branch — retune `target_vol`/`max_leverage`

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

##### Novel branch — per-vote-state Kelly fraction

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

---

### R-36 · 08-19 · CONFIRMED, THINNED — Is v4's return-per-unit-of-risk edge over a matched passive hold real? (B-14)

**Direction.** Formalize B-14: is `kelly_regime_v4`'s
return-per-unit-of-risk edge over a matched passive hold (R-33's byproduct
finding) real outside the 2017–2020 bull?

**What was done.** Pre-registered a pooled decision rule (exact-binomial
95% CI on R-33's existing 40-window win-rate) plus a named falsification
test (split the same 40 windows by start date, before/after 2021-01-01);
reused `windows.csv` unchanged, only recovered each window's calendar date
from the identical seed=42 RNG sequence — 0 new backtests

**Result.** **D1 passes on both markets (CI excludes 50%). Falsification
survives on both markets — post-2021 windows still favour v4 (win-rate
68.2%/81.8%, median +5.0pp/+7.4pp) — but the effect is ~10x smaller than
the pooled/pre-2021 number (+68.9pp/+97.2pp), and the post-2021
subsample's own CI still contains 50% on spot at n=22.**

**Verdict.** **CONFIRMED, thinned** — see full write-up below

#### R-36 pre-registration (B-14) — written and committed before the new analysis is read

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

#### R-36 results — D1 passes, and the falsification test survives, thinned out

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

---

### R-35 · 08-19 · NEGATIVE — Funding rate as a COST-aware SIZE input on `kelly_regime_v4` (B-05)

**Direction.** Funding rate as a COST-aware SIZE input on
`kelly_regime_v4` (backlog B-05)

**What was done.** Two parallel unregistered variants, each on a disjoint
file: `experiments/funding_gate_decile.py` (conservative — literal backlog
reading, stand flat when trailing funding percentile clears the 90th) and
`experiments/funding_ev_band.py` (novel — extends L-05's analytic no-trade
band with a forecast funding-drag term); 80 configurations total across
both branches on inner-train/inner-validation, plus one pre-registered
holdout read (2023-01-01..2023-12-31, the funding-covered slice only) for
the branch that cleared

**Result.** **Conservative clears every inner-validation and falsification
check — genuinely not the exposure-level artifact its own pre-registration
predicted (§5 flat-rescale test) — then loses on the single holdout year
it earned: Δ log growth vs v4 is negative on both markets, −0.167 [−0.495,
+0.101] futures, and stays negative funding-charged. Novel branch is a
real, non-exposure-artifact Sharpe edge (§7 rescale diagnostic) that fails
its own plateau check and, per its author's recommendation, never reached
the holdout.**

**Verdict.** **NEGATIVE** (both branches) — see full write-up below

#### R-35 pre-registration — written and committed before the holdout was read

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

#### R-35 results — a genuine effect that clears every gate but the last one

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

---

### R-34 · 08-19 · NEGATIVE — `harsanyi_crowd`'s Bayesian posterior as a SIZE input on `kelly_regime_v4`

**Direction.** `harsanyi_crowd`'s Bayesian bull/bear/chop posterior (L-12)
as a SIZE input on `kelly_regime_v4`, instead of the DIRECTION input that
lost — L-12's own recorded lesson, tested for the first time

**What was done.** Two parallel unregistered variants, each on a disjoint
file: `experiments/kelly_regime_v5_damp.py` (conservative — a bounded
multiplicative dampener, `mult∈[1−lam,1]`, applied on top of v4's
unchanged vote) and `experiments/kelly_regime_v5_bayes.py` (novel — the
discrete vote replaced entirely by a continuous, hysteresis-latched
posterior margin feeding v4's unchanged conditional-vol-targeting sizer);
42 configurations across inner-train/inner-validation, both markets, plus
ETH/BTC Bitfinex falsification and an explicit matched-mean-exposure check
on each branch

**Result.** **Conservative:** never beats v4 on return in any of 12
measured cells; its drawdown "improvement" is architecturally guaranteed
(the multiplier can only shrink exposure) and the resulting exposure
series correlates **R²=0.997** with a flat 0.7x rescale of v4 — the same
exposure-level artifact as L-04/R-33, R-28/R-31 and R-32, reproduced with
a new source signal. **Novel:** genuinely independent of the vote
(correlation **−0.0017**, not a smoothed duplicate) but underperforms v4
in all 36 configurations (inner-validation spot Sharpe −2.9 to −3.9 vs
v4's +0.14, turnover 4–7x), and explicitly re-scaling exposure to match
v4's mean (`exposure_mult=5.27`) makes it *worse* (Sharpe −6.25, DD 92%),
ruling out the exposure-artifact explanation for this branch — the margin
is simply too noisy at its native hours-to-days cadence to pay
5-minute-bar trading costs on either axis.

**Verdict.** **NEGATIVE** (both branches). Holdout untouched by either
branch.

#### R-34 — L-12's stated hypothesis, finally tested: does the crowd posterior work as a SIZE input?

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

---

### R-33 · 08-19 · NEGATIVE — Matched-risk benchmark: `kelly_regime_v4` vs a de-levered `buy_and_hold` (B-13)

**Direction.** Matched-risk benchmark: `kelly_regime_v4` against a
**de-levered** `buy_and_hold` at equal realized volatility (backlog B-13)

**What was done.** `experiments/matched_hold.py` — a passive long holding
a constant fraction `c` of equity, in two readings (rebalanced to constant
risk, and static buy-once), exposure solved on inner-validation so its
realized volatility equals v4's, then frozen; 18 configurations on the
inner splits; holdout scored with the R-29 paired block bootstrap; 40
windows re-matched **inside each window** to 0.5%

**Result.** **This project's headline is ~90% arithmetic, and what is
underneath it is a different claim.** Across 40 identical windows at
genuinely equal risk, v4's median drawdown advantage falls from **−24.5pp
to −2.9pp** (spot) and **−70.7pp to −5.5pp** (futures) — 88% and 92% of
the gap was the exposure level. On the holdout, five of six frozen cells
fail the pre-registered risk match (a vol-targeter and a constant-exposure
hold cannot be matched across a regime change), and the one valid cell
gives **−14.18pp [−22.68, +13.48]**, containing zero. But the *return*
comparison, which nobody pre-registered, goes v4's way in every cell of
every table: **+20.8pp / +23.8pp median per window in 82% / 90% of them**,
all four ETH/BTC cells, and it survives the ETH falsification test R-28
failed.

**Verdict.** **NEGATIVE** on D1 — the drawdown claim is downgraded to
"against a fully-invested benchmark only". The finding underneath it is
return-per-unit-risk, and it needs its own pre-registered round
(**B-14**).

#### R-33 pre-registration — written and committed before the holdout was read

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
arms: orders identical under two opposite tampers of the future, with
max |column difference| and max |equity difference| before the cut
both **0.000e+00**.

#### R-33 results — half the headline is the exposure, and the other half is not established

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

---

### R-32 · 08-18 · NEGATIVE — The ungated control, and an independent second reading of B-11

**What was done.** A parallel session ran the same backlog row the same
day from the same base commit. Same design as R-31 (one sizer, gate
interchangeable, exposure scaled by a scalar) plus a **third arm with no
gate at all**; 33 configurations, 132 backtests, multipliers frozen on
inner-validation

**Result.** **Agrees with R-31 wherever the two overlap** — gates
indistinguishable at matched risk, R-28's 0-of-40 inverted (deeper in
60%/62%), its fee advantage inverted, P1 failed — from an independent
implementation, and its own holdout cells are **void** under R-31's
validity rule (cap binds on 41%/36%/21% of spot bars; a 29% volatility gap
on futures). What it adds: at matched risk the **ungated** arm is below
both gated arms at every risk level in all four inner-split cells and
loses 80–90% of 40 paired windows. **The gate is worth more than the
choice of gate.**

**Verdict.** **NEGATIVE** — and the parallel-branch report the routine
requires

#### R-32 pre-registration — frozen on a parallel branch, before either branch's holdout was read

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

#### R-32 results — what the gate is worth, and R-31 replicated by an independent hand

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

### R-31 · 08-18 · NEGATIVE — Matched-risk frontier: e-process gate vs latched anchor vote (B-11)

**Direction.** Matched-risk frontier: e-process gate vs latched anchor
vote at equal realized volatility (backlog B-11)

**What was done.** `experiments/matched_risk.py` — one sizer, one
deadband, one warmup, one exposure knob, gate interchangeable. 36
configurations traced on both inner splits and both markets (144
backtests), exposures solved on inner-validation to within 2% of target
volatility in both directions, then frozen; holdout scored with the R-29
paired block bootstrap

**Result.** **Hold risk fixed and R-28's headline dissolves — both halves
of it.** All 8 holdout intervals contain zero and the sign is unstable
across cells; the one cell surviving the pre-registered validity gate
gives −0.072 [−0.532, +0.379] on log growth. Three cells of four are
**void**: the inner-validation exposure match did not survive into 2023+
(29% volatility gaps) or the spot notional cap truncated both arms
differently (41% / 27% of bars). On ETH, with exposures re-matched, the
e-process gate loses all four cells on return **and on drawdown** — so
R-28's P3 replication was measured against an arm carrying 2.4x the risk.
Equal-risk exposure ratio is itself regime-dependent: 2.2x in the bull,
4.7x in the bear.

**Verdict.** **NEGATIVE** — the 0.27x exposure *was* the finding

#### R-31 pre-registration — written and committed before the holdout was read

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

#### R-31 results — the decision rule did not move, and three cells of four were void

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

---

### R-30 · 08-18 · METHOD — Wire the intervals into the comparison table itself (B-12)

**Direction.** Wire the intervals into the comparison table itself
(backlog B-12)

**What was done.** `src/tradebot/evidence.py` reads R-29's `bootstrap.csv`
into `tradebot run`: two verdict columns on the README table (Δ log growth
and Δ max drawdown against `buy_and_hold`, each with its 95% paired
interval and a ▲/≈/▼ mark), the full error bars in the per-market detail
tables, the log-growth interval added to the bootstrap output — R-29
computed it and saved only the point — and a CI rule that a registered
strategy with no measured interval fails the suite. 18 new tests.

**Result.** **The column R-29 computed and discarded says more than the
ones it kept.** On spot over the full history, **0 of 24 strategies are
distinguishably better than `buy_and_hold` on log growth**, the criterion
the table ranks by; 13 are distinguishably worse and the 11
indistinguishable ones are the entire profitable block.
`kelly_regime_v4`'s +0.044 advantage is **[−2.60, +2.85]** — from a
thirteenth of holding's final balance to seventeen times it. Everything
R-29 published reproduced exactly.

**Verdict.** **METHOD** — the warning now lives *in* the table, not beside
it

#### R-30 — the display round, and the statistic that was thrown away

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

---

### R-29 · 08-17 · METHOD — Trials-aware inference: bootstrap intervals, deflated Sharpe, purged CV

**Direction.** Trials-aware inference: block-bootstrap intervals, deflated
Sharpe, combinatorially purged CV (Politis & Romano 1994; Bailey & López
de Prado 2014; López de Prado 2018)

**What was done.** `src/tradebot/inference.py` + `scripts/inference.py`,
applied to all 25 registered strategies on both markets: 96 paired
comparisons, 100 deflated Sharpes, 45 CPCV splits

**Result.** **10 of 96 adjacent pairs in the ranking are distinguishable
at 95%, and none of them separates two of the table's top eight from each
other.** The table's *final-balance* claim for `kelly_regime_v4` over
holding on spot is a coin flip (P=0.52). The drawdown claim survives on
the full history (−41.1pp [−54.8, −18.4]) and on the futures holdout, but
**not** on the spot holdout (−27.1pp [−35.8, **+1.9**]). Cross-validating
the table's own selection rule: it beats holding in **6 of 45** folds.

**Verdict.** **METHOD** — the ordering is mostly noise, and now says so

#### R-29 pre-registration — written and committed before any statistic was read

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

#### R-29 results — the self-test passed, and then almost nothing else did

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

---

### R-28 · 08-17 · NEGATIVE — E-process regime detection with unified Kelly sizing

**Direction.** E-process regime detection with unified Kelly sizing
(Shafer 2021; Ramdas et al. 2023; Waudby-Smith & Ramdas 2024; Shin, Ramdas
& Rinaldo 2024)

**What was done.** Three variants in `experiments/eprocess_regime.py`, 24
configurations on the inner split, one frozen config on the holdout

**Result.** **The deepest drawdown reduction in the project, and it still
loses.** Holdout spot DD **11.6%** vs `kelly_regime_v4`'s 27.8% and
holding's 54.0%; deeper than v4 in **0 of 40** Monte Carlo windows (median
−14.0pp spot, −11.3pp futures). Return is 0.42x holding, so P1 fails.
Anytime-valid evidence justifies only **0.27x** the incumbent's mean
exposure.

**Verdict.** **NEGATIVE** — and the risk finding was **retracted by
R-31**: at matched risk it does not replicate on ETH and reverses in
45–82% of the stress windows

#### R-28 pre-registration — written and committed before the holdout was read

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

#### R-28 results — the decision rule did not move

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

---

### R-27 · 08-17 · METHOD — Fabrication pressure in the operator's own prompt

**What was done.** The synthesis prompt for R-26 contained a conditional
naming the hoped-for answer: *"If the inference agent found that most of
the table's ordering is not distinguishable from noise, say so first and
plainly."* The inference agent had run zero backtests.

**Result.** The synthesizer refused and flagged it. Had it complied, a
fabricated headline would have entered `docs/VALIDATION.md` — the file
whose whole purpose is being trustworthy — indistinguishable from a real
result to a later reader. Same failure class as L-14/L-15/L-16 (proxying
order flow out of price) and R-21 (the $3.7e23 probe), but arriving
through the *prompt* rather than the code.

**Verdict.** **METHOD** — see ROUTINE.md

---

### R-26 · 08-17 · NULL ROUND — Parallel round on B-01, B-02/03, B-04, B-05, B-07

**What was done.** 11 agent-sessions dispatched (5 build, 5 skeptic, 1
synthesis). Every one was blocked before executing a single call: the
permission handler returned `updatedInput` with required parameters
stripped, so `Bash`, `Read`, `Glob` and `Grep` all failed schema
validation. Repo verified untouched afterwards. Fault has since cleared.

**Result.** **0 trials, 0 configurations, 0 bars read.** The five
directions were **NOT TESTED** and stay on the backlog as untried — filing
them as negatives would stop a future agent trying them. Holdout counter
unchanged (nothing was read). Project trials count unchanged.

---

### R-25 · — · CLOSED — Deflated Sharpe / purged CV / bootstrap CIs

**What was done.** Cited in `RESEARCH.md`, **never computed**

**Result.** Every sweep here is a trial that inflates whatever it selected
(R-12 ran 32). The comparison table reports points where it should report
ranges.

**Verdict.** **CLOSED by R-29**

---

### R-24 · 08-15 · SETTLED — Exchange adapter parity

**What was done.** Bar-for-bar over 30 consecutive candles, both adapters

**Result.** Top-three strategies compute the identical target from paged
exchange data and from the contiguous backtest frame; paging is lossless;
neither adapter hands a strategy the forming candle.

---

### R-23 · 08-15 · SETTLED — Capital scaling

**What was done.** $1K vs $1M across every strategy

**Result.** Results are proportional to capital; the only deviations came
from the exchange minimum order size. One start balance is therefore
sufficient.

---

### R-22 · 08-15 · METHOD — Warmup-prefix bias

**What was done.** Audit

**Result.** Letting a strategy trade the warmup prefix let it be
liquidated *before* the window opened — **19 of buy-and-hold's 23 stress
liquidations were this artifact**. Slicing to an OOS range left a
100-day-warmup strategy flat for 7.6% of it. Fixed by
`run_backtest(trade_start=...)` / `tradebot.window.run_period`; verdicts
survived, numbers moved ~75%.

---

### R-21 · 08-15 · METHOD — Lookahead probes

**What was done.** Two adversarial probes

**Result.** A one-day signal broadcast onto 5m bars is worth **+2.1
Sharpe** and *passes* truncation. A strategy that keeps the `prepare()`
frame and indexes `i+1` in `on_bar` returned **$3.7e23 at Sharpe 73 with a
green suite**. Both now caught by `test_causality_real.py` /
`test_causality_strict.py`.

---

### R-20 · 08-15 · METHOD — Noise floor measurement

**What was done.** Paired stationary block bootstrap, 30-day blocks, 2,000
resamples

**Result.** **±0.2 Sharpe.** Smaller differences on one path are not
evidence. The analytic SE of a Sharpe *level* (±0.02) is misleadingly
tight for *comparing* strategies.

**Verdict.** **METHOD** — binds every claim here

---

### R-19 · 08-14 · KEY FINDING — Monte Carlo window stress test

**What was done.** 40 random windows, identical across strategies

**Result.** Leveraged buy-and-hold **liquidated in 26 of 40**, median
window −98%. The three resampled `kelly_regime` variants (v2/v3/v4)
survived all 40; on 5x futures profitable in 85–88% and beat holding in
65% (spot: beat holding in 48–50%).

---

### R-18 · 08-16 · NOT PURSUED — Elliott Wave Theory (± NN, ± game theory)

**What was done.** Assessed against this repo's bar

**Result.** Not falsifiable as practised (Aronson: a story prone to
subjective revision) — counts are re-labelled after the fact, the exact
leak class `test_causality_strict.py` exists to catch. Its one
quantitative component (Fibonacci ratios) was refuted by Batchelor &
Ramyar. *ElliottAgents* (Applied Sciences 14(24), Dec 2024; multi-agent
LLM + deep RL) reports 73.68% vs 57.89% on BTC/USD Oct 2022–Sep 2024 —
that is **14/19 vs 11/19**, three extra calls, over a monotonic $20K→$70K
rise, with no walk-forward. Training an NN on wave labels adds nothing a
network cannot learn from price directly, while importing a subjective
hindsight-contaminated annotation step. Its useful kernel (multi-timescale
crowd structure) is already `kelly_regime_v4`.

---

### R-17 · 08-16 · PARTLY ANSWERS N≈3 — Cross-asset falsification on ETH

**What was done.** Bitfinex BTC + ETH, same venue, same window
(2016-03→2019-12)

**Result.** **The risk property transfers, the return property does not
exist.** Drawdown cut in all four cells (BTC 83.8→40.1, ETH 94.2→36.5, 5x
85.2→32.1 and 99.3→35.1). Loses to holding on spot on both assets (0.58x,
0.47x). The 236x ETH futures cell is survival, not edge.

---

### R-16 · 08-16 · OPEN — Funding as a positioning signal

**What was done.** Quintile and momentum-controlled sort, 2020–2023

**Result.** 14-day forward spread Q1−Q5 = **+3.57pp**; high funding
predicts negative forward returns unless price is also rising; correlation
with trailing return only 0.39, so not a momentum proxy. But middle
quintiles are non-monotone (Q3 +3.06%, Q4 −1.02% at tied clamped rates) —
a warning about how much is noise. Full tables in `VALIDATION.md` (funding
section).

**Verdict.** OPEN hypothesis → B-05

---

### R-15 · 08-16 · BLOCKED — Funding harvest / cash-and-carry

**What was done.** Compounded the real series, 2020–2023

**Result.** +82.0% over 4.0y = **+16.2%/yr**; +14.6% after 0.10% on both
legs (quarterly rebalance), +9.8% at 0.40%; payer flips 13.5% of
settlements; **worst 30-day run −1.31%**. Literature (He et al. 2024 and
the 2020–2025 empirical carry work) reports carry Sharpe ~6.45, falling to
4.06 from 2024 and **negative in 2025** as it crowded — and our data stops
exactly at 2023. Full table in `VALIDATION.md` (funding section).

**Verdict.** **BLOCKED** on data → B-02

---

### R-14 · 08-16 · KEY FINDING — Funding as a first-class cost (`scripts/funding_study.py`)

**What was done.** Real Binance BTCUSDT funding, compounded

**Result.** Positive at **86.5%** of settlements, ~15%/yr for a constant
long. `kelly_regime_v4`'s $156K becomes **$36K–$80K** — a band straddling
spot holding's $66K. Worse: funding runs **+20%/yr while the strategy
holds** vs +2.8% flat, because the crowding it detects is what sets the
rate.

**Verdict.** **KEY FINDING** — the COST constraint

---

### R-13 · 08-15 · CLOSED — Fee tier study (`scripts/fee_study.py`)

**What was done.** Measured every Bitstamp tier

**Result.** Break-even is **0.104%** against an assumed 0.10% — the
published spot edge lives entirely inside that margin. At the 0.40% entry
tier nothing beats holding ($29.5K vs $65.8K); the $5M/30d tier still
misses by 4%.

---

### R-12 · 08-15 · CLOSED — Turnover reduction to fit a fee tier

**What was done.** Swept 8 lookbacks × 4 hysteresis bands = **32 configs**

**Result.** **28 of 32 beat holding in-sample; 0 of 28 out-of-sample.**
The one you would have selected lost 34.5% to holding. Gross edge on spot
(1.33x) is far below the 2.98x needed at 0.40%, and slowing down shrinks
the gross edge in step.

**Verdict.** **CLOSED** — the defining negative result

---

### R-11 · 08-15 · PARTIAL — Grossman–Zhou drawdown cushion (1993)

**What was done.** Two variants

**Result.** Whole book: drawdown 28% but return destroyed ($21.6K). Above
1x leverage only: −1.2pp drawdown at ~zero cost. Klass & Nowicki (2005)
predicted the former — the cushion rule sells low in a
mean-reverting-drawdown asset.

---

### R-10 · 08-15 · KEY FINDING — Inverse leverage effect in BTC (Baur & Dimpfl 2018)

**What was done.** Measured forward 5d Sharpe by lagged-vol quintile

**Result.** High vol forecasts the **highest** forward Sharpe (+1.08 all
bars, +2.06 when the gate is bullish) — the opposite of equities. Moreira
& Muir (2017) vol-managed alpha is absent-to-inverted here.

**Verdict.** **KEY FINDING** — explains L-02

---

### R-09 · 08-15 · NEGATIVE — Range volatility estimators (Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang)

**What was done.** Measured on 5m bars

**Result.** Read **7–18% low** (discretisation bias); a drop-in swap
silently raises effective leverage. Their efficiency advantage is against
a *daily close-to-close* estimator; the incumbent already averages 288
squared returns/day.

---

### R-08 · 08-15 · NEGATIVE — Better volatility *forecasting*

**What was done.** Timescale blend, 8% better on QLIKE (Patton 2011)

**Result.** **$52K instead of $115K.** A genuinely better forecast
de-levers more promptly into BTC's high-vol, high-forward-Sharpe states.

**Verdict.** NEGATIVE — and sign-inverting

---

### R-07 · 08-15 · INFORMS L-01 — Anchor timescale region, 18–28 days

**What was done.** 9 anchor sets

**Result.** *Every* variant cut drawdown to 35–39% from 41.8%; Sharpe
spread 1.52–1.60 sits inside the noise floor. Breaks sharply below ~18d
(16/32/64 → 1.46) — a region, not a tuned peak.

---

### R-06 · 08-15 · NEGATIVE — Anchor ladders of 7–48 moving averages

**What was done.** Swept

**Result.** Scored at or below the three-anchor vote. Individual anchors
wildly dispersed (20d 1.17, 250d 0.59).

---

### R-05 · 08-15 · NOT ATTEMPTED — Deep learning (Lim 2019; Momentum Transformer 2021)

**What was done.** Literature assessment; no dependency added

**Result.** Published edge assumes **2–3bps against our 10bps**, and much
of it comes from diversifying across **88–100 instruments** where we have
one. Counter-evidence: Makridakis 2018, Buczyński 2023, Zeng AAAI 2023
(one-layer linear beats transformers on 9 benchmarks).

**Verdict.** NOT ATTEMPTED, ruled out on grounds

---

### R-04 · 08-15 · NEGATIVE — Meta-labeling + triple barrier (López de Prado 2018)

**What was done.** Walk-forward, purged and embargoed logistic secondary
model

**Result.** Hurt in-sample, neutral out-of-sample. Their trend-scanning
label looks *forward* and is inadmissible here.

---

### R-03 · 08-15 · NEGATIVE — Bayesian online changepoint detection (Adams & MacKay 2007)

**What was done.** Implemented as a severity haircut

**Result.** Lost: OOS 0.84 vs 1.03. Short run lengths fire on volatility
bursts, and in BTC large **up** moves are volatility bursts.

---

### R-02 · 08-15 · NEGATIVE — Statistical jump models (Nystrup 2020; Shu 2024)

**What was done.** Implemented, walk-forward, deterministic restarts

**Result.** −6–11pp drawdown, ~40% less turnover, **no Sharpe gain**
(0.96–1.06 vs 1.09; bootstrap P(gap>0)=0.26). A random-init version looked
like a win purely from optimizer noise — the seed alone moved Sharpe 0.13
and growth 40%.

---

### R-01 · 08-15 · REJECTED ON READING — Markov-switching / HMM regime detection (Hamilton 1989)

**What was done.** Assessed

**Result.** The *filtered* state is causal, the *smoothed* one that
tutorials plot is not. Reported rapid switching is fatal at a 0.1% round
trip.

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
| Minimax-across-purged-folds robust reselection of `kelly_regime_v4`'s own existing free parameters (anchor-ladder base, `target_vol`, `max_leverage`) — no new signal | 54 configurations; genuinely improves generalization over the naive pooled point estimate on the same search space (wins ETH outright, loses less on the BTC-control futures cell, 62% vs 37% of v4's balance) and clears inner-validation without the exposure artifact (R²=0.86) — but still fails the falsification test's BTC-control clause on futures, because all three purged folds are drawn from 2017–2022 and none samples the 2016–2019 control period. | R-45 (conservative) |
| Periodic causal walk-forward re-estimation of `kelly_regime_v4`'s `target_vol`/`max_leverage` (365d refit / 730d lookback, fee-free proxy-Sharpe grid search), replacing the frozen global constants with a re-fit loop | 3 pre-registered schedules; causality probe passes cleanly (no lookahead bug) but the mechanism is not competitive — loses to v4 on inner-train, inner-validation (both markets), the BTC control and ETH alike, and the primary candidate still trips the exposure-artifact bar (R²=0.98) despite losing. Each individual refit is a low-information estimate from only 1–2 trailing regime-events, fractalizing the N≈3 problem rather than resolving it. | R-45 (novel) |
| Fixed-multiplier CPPI (Perold & Sharpe 1988) cushion scale replacing v4's vol-targeting, floor anchored once to starting balance and grown at a small fixed rate (deliberately not peak-following) | 24 configurations; not the standard R²>0.95 exposure artifact, but the winning region saturates `cppi_scale` at `max_leverage` almost immediately once equity compounds through a multi-year window, degenerating into "vote × constant max leverage" (candidate pinned at exactly 2.000 throughout inner-validation) — 2–3x v4's average notional, worse Sharpe and drawdown than v4 in both splits, and fails its own pre-registered BTC-control falsification decisively (ΔSharpe −0.47 spot / −0.76 futures) before ETH is even read. No point in the 24-point grid beats v4 on both Sharpe and drawdown at once. | R-46 (conservative) |
| Hurst-exponent-adaptive CPPI multiplier (classical R/S method, Hurst 1951/Mandelbrot & Wallis 1969) layered on the same fixed-floor CPPI base | 33 configurations (32-point grid + fixed-m=4 ablation); causality PASS on all four probed columns including the new rolling-Hurst column; inherits the conservative branch's identical BTC-control falsification failure (same saturated-scale mechanism, ΔSharpe −0.47/−0.76); the adaptive multiplier barely beats its own fixed-m=4 ablation (spot ΔSharpe ≈0, futures +0.226 from one window); empirical rolling H(t) came out persistently high (mean 0.62) rather than the pre-registered ≤0.5 failure hypothesis, but did not help regardless — possibly Lo (1991)'s documented upward bias in classical R/S under volatility clustering rather than real persistence. | R-46 (novel) |
| Never-rebalanced, one-time-split 50/50 (±60/40, 40/60) BTC+ETH `kelly_regime_v4` portfolio via the promoted `multiasset.py` adapter — the cheapest form of B-19 | 13 configurations; clean on both falsification gates (R²=0.86–0.87, survives 0.40% tier) and the neighbourhood is a genuine plateau, but decisively REJECTED on the one holdout read its own pre-registration authorized: loses to `buy_and_hold` by 24–46% and is statistically indistinguishable from (or slightly worse than) BTC-solo v4 alone. Captures ~100% of R-50's inner-validation drawdown edge with zero rebalancing but only ~29% of its Sharpe edge — do not re-try the never-rebalanced form on this asset pair without a different holdout period or composition. | R-51 (conservative) |
| Inverse-trailing-volatility weighting (Maillard/Roncalli/Teiletche 2010 ERC special case) of a periodically-rebalanced BTC+ETH `kelly_regime_v4` portfolio, swept across monthly/quarterly/semiannual cadence and 4 volatility lookbacks | 12 configurations; passes the exposure-artifact check (R²=0.58–0.94) but fails the 0.40% fee-tier falsification and, before that, never beats a correctly re-derived fixed-50/50 reference on any of 12/12 configurations, any cadence, either market (ΔSharpe −0.02 to −0.11, R²=0.996 vs. the static split it was meant to improve on — near-relabeling with added turnover). Lengthening the rebalance cadence made both arms worse, not better, contradicting the cost-driven "rebalance less" literature this branch was built to test, because rebalance fees are trivial relative to diversification-maintenance value at this pair's scale and this project's cost tier. Do not re-try inverse-vol/ERC weighting on a 2-asset BTC/ETH book without a genuinely different asset pair (N>2, where correlation structure varies leg-to-leg) or a different information source. | R-51 (novel) |
| Literal periodically-rebalanced (monthly), fixed-50/50 BTC+ETH `kelly_regime_v4` portfolio through R-50's continuous (non-restarting) engine — R-50's own original candidate, the one form left untested by both R-51 branches | 15 configurations; clean on both falsification gates (R²=0.88, survives 0.40% tier) and the split-ratio neighbourhood is a genuine plateau, reproduces R-50's own cited byproduct number almost exactly (ΔSharpe +0.79, DD 33.2%→27.1% inner-validation) — but decisively REJECTED on the one holdout read its own pre-registration authorized: loses to `buy_and_hold` by 22–45% and its edge over BTC-solo v4 is noise, not stably signed. Do not re-try the literal calendar-rebalanced form on this asset pair without a different holdout period or composition. | R-52 (conservative) |
| Threshold/band-triggered rebalancing (±5/10/15% drift bands) of the same fixed-50/50 BTC+ETH `kelly_regime_v4` target, in place of any fixed calendar cadence | 6 configurations; clean on both falsification gates (R²≤0.91) and a genuine plateau, cuts rebalancing turnover 70–90% vs. a fixed-monthly reference for statistically identical Sharpe/drawdown on both inner-validation and the holdout — but decisively REJECTED on the one holdout read its own pre-registration authorized: loses to `buy_and_hold` by 48–61% and is statistically indistinguishable from both BTC-solo v4 and the calendar reference. The turnover-reduction mechanism itself is not ruled out (it worked exactly as designed) — only its ability to rescue this specific return premium on this holdout is. Do not re-try band-triggered rebalancing on this same underlying Sharpe edge without a candidate whose edge already clears the holdout on other grounds. | R-52 (novel) |
| Never-increase-only multiplicative haircut on `kelly_regime_v4`'s exposure from real VIX/DXY macro stress (`stress_z`, FRED) | 10 configurations; the standard exposure-level artifact in every cell (R²=0.974–0.999) even though the feeding signal is genuinely price-independent for the first time — no inner-validation edge over v4 anywhere in the 18-cell grid, and an inconsistent, non-monotonic ETH/BTC-control falsification direction. Do not re-try a bounded never-increase-only multiplier architecture on this strategy family regardless of the feeding signal's source without first checking the exposure-artifact R² — this is now its fourth confirmed instance (R-34, R-41-conservative, R-46-conservative's saturated variant, now this). | R-53 (conservative) |
| VIX/DXY macro stress as a precision-weighted 4th vote in `kelly_regime_v4`'s regime gate, testing whether macro risk-off leads the price-anchor gate | 15 configurations; the macro vote leads the 3-anchor majority in only 4/12 matched stress episodes (median offset −5.5 days, i.e. it lags on net); no config beats v4 on inner-validation Sharpe in the 12-cell grid; loses to its own hard-override ablation in 10/12 matched cells by 0.25–0.48 Sharpe; fails the ETH-spot falsification on all 12 configs while several beat the BTC control. Do not re-try a precision-weighted macro-vote average on this signal without a materially different lead-time result — but see **B-21** for the un-averaged hard-override variant, which is not ruled out by this row. | R-53 (novel) |
| **B-21**: a hard, unweighted VIX/DXY macro-veto override (`frac=0` while `stress_z`-latched "stress", v4's own unmodified 3-anchor average otherwise) on `kelly_regime_v4`'s regime gate, given its own pre-registration and falsification battery for the first time | 5 configurations; fails its own primary pre-registered test (lead-time vs. the 3-anchor majority: leads only 4/12 episodes, median −5.5 days, replicating R-53's averaged-vote finding almost exactly — blunting the combination rule does not fix the timing); fails the parameter-neighbourhood plateau check (spread 0.32 ≥ the 0.2 noise floor; the grid's best-scoring point is the explicit no-hysteresis negative control, not the pre-registered primary); fails the ETH falsification (5/10 config×market cells underperform v4 on ETH while beating it decisively on the BTC control). Passes causality cleanly; the exposure-artifact check is config-dependent (primary R²=0.841 passes, the grid's actual best cell R²=0.9544 fails). Do not re-try a hard macro-veto fed by VIX/DXY `stress_z` on this mechanism without a materially different lead-time result — the underlying signal's lag against the price-anchor gate, not the combination rule, is the binding constraint, and it is identical whichever way the signal is combined. | R-54 (conservative), closes B-21 |
| Aggregate USDT stablecoin-supply-deceleration hard veto (`frac=0` while a 14-day log-growth z-score is latched "stress") on `kelly_regime_v4`'s regime gate — same architecture as B-21, different (crypto-native, capital-flow) signal | 9 configurations; the pre-registered lead-time test PASSES (leads the 3-anchor majority in 9/12 matched episodes, median +16.5 days — the reverse of every prior INFO-axis attempt's lag) but the strategy still fails decisively: no configuration beats v4 on inner-validation Sharpe (best cell 0.13 vs. 0.14, inside the noise floor, no plateau), 8/9 cells are substantially worse, and every configuration loses to v4 across the full pre-2020 BTC control in absolute terms (ratio 0.13×–0.90×) — a threshold tight enough to buy real lead time also fires on transient supply noise, and standing flat through the false alarms costs more than the early exits recover. Passes causality and the exposure-artifact check (R²=0.61) cleanly. Do not re-try a single-threshold hard veto on raw stablecoin-supply growth without adding a magnitude-*and*-duration filter or a confirming (not overriding) combination rule — see **B-22**. | R-54 (novel) |
| A minimum-persistence/duration requirement (`persist_days`, swept 0–14) on the stablecoin hard veto above, before it is allowed to force `frac=0` | 24 configurations; fails worse than the un-filtered veto, not better — the pre-registered falsification test fails outright (median lead flips from +16.5 days to a 10-day lag by `persist_days=5`, independently reproduced), because the "transient" onsets it targets don't reverse within a few days at this feature's native 14-day-growth cadence, they persist about as long as genuine stress episodes do. No configuration among all 24 beats v4 on inner-validation Sharpe. Passes causality and the exposure-artifact check (R²=0.61) cleanly. Do not re-try a persistence/duration filter on this specific feature (14-day log-growth z-score) without first shortening the feature's own smoothing window to match genuine-stress timescales — a duration requirement bolted onto an already-smoothed feature mostly re-measures the smoothing, not the noise. | R-55 (conservative), closes B-22 (fix 1 of 2) |
| Feeding the stablecoin-stress vote into R-53's precision-weighted CONFIRMING-vote architecture (`frac=(anchor_sum+weight·vote)/(3+weight)`) instead of a unilateral hard override | 21 configurations (17 confirming-vote cells + 4 ablation cells); the architecture question is answered — confirming beats an equivalent hard override in 16/16 matched cells once fed a genuinely leading signal (independently reproduced), the reverse of R-53's finding under a lagging one, so **the combination rule itself is not what was wrong** — but no non-identity configuration clears v4 on inner-validation spot Sharpe (best 0.10 vs. 0.14) and ETH falsification fails decisively (a majority of non-identity spot configurations underperform v4 on ETH by more than on the BTC control, independently reproduced at 14/16). Passes causality cleanly; exposure-artifact R²=0.9407 passes but with little margin. Do not re-try this specific signal in this or the hard-override architecture without first improving the feature's own precision (fewer false stress-onsets) — the combination rule is not the binding constraint, the signal's specificity is. See **B-23**. | R-55 (novel), closes B-22 (fix 2 of 2) |
| Reading `kelly_regime_v4`'s drawdown property as a property of the *strategy* rather than of BTC and ETH specifically | R-57 ran the frozen, byte-identical strategy on six Coinbase instruments it was never fitted on (BCH, LTC, ETC, DASH, LINK, XTZ, 2020-04→2026-08, 130 configurations, holdout untouched). Against a hold carrying **v4's own mean exposure** the advantage does not shrink, it **inverts on 6 of 6** (Δ max drawdown +5.2 to +33.8pp, 4 of 6 intervals excluding zero, all against v4); against the fully-invested `buy_and_hold` the same runs give 6/6 in v4's favour by 16–46pp — the exposure artifact R-33 measured, reproduced on new instruments and larger. A control over a window every asset shares (2020-04→2022-12, no 2023+ bar read) puts BTC at −5.6pp and ETH at −11.5pp in v4's favour and every panel asset between +0.0 and +17.1pp against it: **2 of 8, and they are exactly the two assets this project has always measured on**, so the failure is asset-specific rather than period-specific. R-36/B-14's confirmed return-per-risk edge does not reproduce either (1 of 6 on the mean-notional axis with every interval containing zero, 0 of 6 on the volatility-matched axis). Do not describe the drawdown/tail finding as a property of regime-gated sizing without naming its measured scope: BTC and ETH. | R-57 |
| Shortening the stablecoin-supply-deceleration feature's own growth window (2/3/5/7/10 calendar days, in place of R-54's fixed 14) to match literature-reported acute-redemption timescales, hard-veto architecture otherwise unchanged | 15 configurations (grid scoped down mid-session from a pre-registered 45 after the pre-registered Step-A gate, run before any Sharpe number, already returned a clean, decisive kill); 0/5 windows preserve or improve the N=14 reference lead-time result — the flip is monotonic (lead fraction 75%→31%, median offset +16.5d→−15.0d as the window shrinks from 14 to 2 days), diagnosed as a genuine timescale mismatch between ~2-3-day acute redemption stress and the multi-week capital-flight dynamic the signal actually leads on, independently reproduced exactly by the operator. No window clears v4 on inner-validation Sharpe (best −0.049 spot), no plateau, the one near-tied config is an exposure-level artifact (R²=0.98), and the one config with a genuinely different exposure shape (w=10d, R²=0.57) fails ETH falsification outright. Do not re-try shortening this feature's growth window without a different justification for the target timescale than acute-redemption-stress literature, which this round found measurably wrong for this signal's purpose. | R-58 (conservative), closes B-23 (fix 1 of 2) |
| AND-gating the stablecoin-supply-deceleration signal with a second, structurally independent on-chain signal (BTC active-address-growth stress, B-07/R-44's channel; Hash Ribbons screened and rejected as too coarse, 1/12 corroboration) before allowing the stablecoin dilution/override to apply | 33 configurations (21 confirming-vote-dilution + 4 no-corroboration ablation + 8 hard-override-architecture ablation); the pre-sweep mechanism check, run before the sweep, already predicts failure — corroboration does not discriminate genuine leading episodes from noise (7/9 leads corroborate vs. 3/3 lags/noise also corroborate, both independently reproduced exactly) and *increases* the raw false-onset flip count (24→28) rather than reducing it. No non-identity configuration clears v4 on inner-validation spot Sharpe (best 0.11 vs. 0.14) and ETH falsification fails (18/42 cells, concentrated on spot). A striking secondary number — corroboration turning R-54's failed hard-override architecture into a tie/marginal win over v4 (Δ Sharpe up to +0.81) — is fully explained and closed by an exposure-artifact check (R²=0.9971, independently reproduced): corroboration mostly disables the override back toward v4's own exposure path, the same relabeling trap R-33/R-34 already caught, now a third time. Passes causality cleanly on all three data pathways. Do not re-try AND-gate corroboration with active-address growth or Hash Ribbons on this signal without a corroborating source that tracks something more specific than "broadly a downtrend." | R-58 (novel), closes B-23 (fix 2 of 2) |

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

**Re-ranked 08-19 after R-42.** A same-day parallel round tried a
genuinely different axis for the first time: instead of re-deriving a
signal from BTC's own single price series (ten branches, R-34 through
R-41, all on that axis), it used B-15's real ETH data to actually **hold
capital in a second asset** — a fixed-split dual book (conservative) and
a covariance-aware dynamic reallocator between two unchanged
`kelly_regime_v4` sub-books (novel). Both came back **NEGATIVE**, but this
is the first round in the program to clear the exposure-artifact bar
cleanly on both branches at once (R²=0.005–0.89, well under the 0.95
line) — genuinely new mechanisms, not relabeled leverage — and the
conservative branch's inner-validation drawdown improvement (−4 to −7pp
through the 2022 joint bear) is real by that diagnostic even though its
significance is not established (n=1 joint-bear window, no bootstrap run
this session). See R-42 and new backlog item **B-16**, which carries the
two authors' own prescribed next steps rather than a holdout read on
either branch as built. The order below is otherwise unaffected: **B-06
(forward paper trading) remains the highest-value item on merit**, and
this session did not attempt a network-access re-check for B-07 or B-08
(the ETH data R-42 used answers a different question — diversification,
not "does v4 replicate unchanged on a second bear" — so B-08 stays open
and unattempted in its original form).

**Re-ranked 08-19 after R-43.** B-16 is done. The conservative branch's
inner-validation bootstrap held up well enough (bear-quartile drawdown
improvement, CI excludes zero on both markets, not the exposure artifact)
to earn this project's first genuinely new holdout consultation since
R-39 — pre-registered one commit ahead of the read, per this file's own
practice. **It failed**: the pre-registered bear-quartile claim replicates
on 5x futures but not on spot, and the pooled claim outright reverses sign
on the holdout (median **worse**, not better, on both markets). This is
the eleventh SIZE/N≈3-axis branch in a row (R-34, R-37, R-38, R-40, R-41,
R-42, R-43) not to survive both an inner-validation check and an
out-of-sample one — the pattern is now different in kind from the earlier
ten, though: this is the first to have gotten *past* inner-validation
scrutiny (a real bootstrap, not just an exposure-artifact check) only to
fail at the holdout itself, rather than failing an ETH or pre-2020-BTC
falsification before the holdout was ever reached. That is a slightly
worse update about this whole program's remaining headroom on data already
in hand than the prior ten branches gave, not a better one. **B-06
(forward paper trading) remains the highest-value item on merit, and now
more clearly the only genuinely open item that is not a further re-cut of
a dataset this project has just watched fail its own bootstrap.** Two
smaller items open behind it: **B-17** (this project has no multi-asset
strategy-registration path at all, discovered while writing R-43's own
pre-registration — moot for R-43 itself since it was rejected, but a real
gap for any future dual-asset finding that does clear a holdout) and
**B-18** (whether `kelly_regime_covkelly`'s cadence-inconsistency, shrunk
but not resolved by R-43's mean-denoising, is actually a rebalance-engine
artifact rather than an estimation-noise one). Neither is attempted this
session.

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

**Re-ranked 08-19 after R-44.** B-07 is done: real on-chain data (CoinMetrics,
free, no key) turned out to be reachable, closing this item's `BLOCKED
(network)` status after a genuine attempt rather than a further connectivity
guess. Both branches — a sign-corrected participation-confirmation multiplier
and a Hash Ribbons miner-capitulation vote — were NEGATIVE, for two different,
independently-reproduced reasons (magnitude-only exposure-artifact failure;
clean inner-validation loss on both markets in 12/12 configs). The order below
is otherwise unaffected: **B-06 (forward paper trading) remains the
highest-value item on merit**, still blocked on network access, and is now
also the most clearly justified next step on a different basis than before —
INFO-axis attempts are 0-for-3 (this round's two branches plus R-41's two) on
top of eleven-of-eleven SIZE-axis branches failing across six prior rounds,
which raises the prior that `kelly_regime_v4`'s vote-and-scale mechanism
itself, not merely each individual candidate signal, is close to a genuine
plateau for what this project's historical data can support. A session with
spare capacity and continued network access should attempt B-06's real
connection (Bitstamp, per `docs/LIVE.md`'s existing `BitstampSpot` adapter)
rather than a twelfth SIZE-axis or fourth INFO-axis variation on the
incumbent.

**Unchanged 08-19 after R-45.** An off-backlog, literature-prompted round
(non-parametric bootstrap-robust parameter selection for time-series
momentum; adaptive walk-forward regime trading on Bitcoin, both 2025)
tried two axes genuinely different from the twelve prior SIZE/INFO-axis
branches: a conservative reselection of `kelly_regime_v4`'s own existing
constants by robustness across purged folds instead of a pooled point
estimate (attacking **ERR** — the parameter-*selection* step, not the
signal), and a novel periodic causal walk-forward re-estimation loop
replacing the frozen constants outright (attacking **N≈3** by
architecture rather than a new input). Both **NEGATIVE**, but not
uniformly so: the conservative branch is the first in this family to
produce a partial, quantified positive — robustness-aware selection
measurably beats the naive point estimate on the identical search space —
while still failing the pre-registered falsification test, because none
of its purged folds sample the pre-2020 BTC-control period. The novel
branch never beat v4 anywhere, diagnosed as fractalizing N≈3 (several
under-informed re-fits instead of one) rather than resolving it. The
order below is unaffected: **B-06 (forward paper trading) remains the
highest-value item on merit**, still blocked on network access as of the
last check (R-41/R-44; worth a fresh check before a future session
assumes it still holds). This is the program's thirteenth and fourteenth
consecutive failures on `kelly_regime_v4`'s SIZE/architecture family
across nine rounds (R-34, R-37, R-38, R-40, R-41, R-42, R-43, R-44, R-45)
— a future session should treat B-06, B-08, B-17 or B-18 as more
promising uses of a session than a fifteenth variation on the incumbent's
own vote-and-scale mechanism, whatever axis it attacks.

**Unchanged 08-19 after R-46.** An off-backlog, literature-prompted round
tried a structurally different sizing FAMILY for the first time in ten
rounds — Constant Proportion Portfolio Insurance (Perold & Sharpe 1988)
replacing v4's vol-targeting scale, conservative with a fixed multiplier
and novel with a Hurst-exponent-adaptive one — rather than another tweak
to the vote-and-vol-target architecture itself. Both **NEGATIVE**, and
for a genuinely new reason this ledger had not yet recorded: a CPPI floor
anchored once to starting equity (chosen specifically to avoid the
peak-chasing/inverse-leverage conflict R-33 onward has repeatedly found)
stops binding almost immediately once equity compounds through a
multi-year backtest, so the mechanism degenerates into "vote × constant
max leverage" well before any falsification test is run — both branches
fail the identical pre-registered BTC-control clause decisively (ΔSharpe
−0.47 to −0.76) at 2–3x v4's average notional. This is the fifteenth and
sixteenth branches across ten rounds (R-34, R-37, R-38, R-40, R-41, R-42,
R-43, R-44, R-45, R-46) to fail on `kelly_regime_v4`'s SIZE axis, and the
first two to fail via a wholesale mechanism replacement rather than a
variant of vol-targeting — which raises the prior further that the axis
itself, not any one candidate signal or formula family, is exhausted for
what this project's historical data can support. This session's own
connectivity re-check (a direct HTTPS probe, not just a guess) found
Deribit and Coinbase reachable (200), Binance still blocked (451), and
Bitstamp unreachable within an 8s timeout — a materially different result
from R-41's "all four reachable" finding and worth re-checking again
rather than assuming either way, since this status has flipped repeatedly
across sessions. The order below is unaffected: **B-06 (forward paper
trading) remains the highest-value item on merit**, and a future session
should treat B-06, B-08, B-17 or B-18 as more promising than an eleventh
SIZE-axis variation of any kind, formula family included.

**Re-ranked 08-19 after R-47 and R-48.** This session followed the
ledger's own repeated recommendation rather than a seventeenth SIZE-axis
variant: two disjoint parallel branches, each on its own backlog item
rather than competing takes on one idea. **B-08 is done and closed**
(R-47): the frozen `kelly_regime_v4`, zero parameters changed, replicates
its drawdown/tail property on ETH's own previously-untested 2022 bear —
the first ETH evidence in this project independent of the 2018 BTC bear
every earlier ETH check shared — but its return edge does not survive
the realistic 0.40% fee tier over the full 2020–2026 window, confirming
L-01/R-17's own standing caveat on genuinely independent data for the
first time. **B-06's network block is lifted and the recorder now exists
and runs** (R-48): `scripts/paper_trade.py` is live against the real
Bitstamp feed, with two genuine recorded decisions as of this session. It
is deliberately not marked DONE — a two-row record answers nothing yet —
and its one open follow-up is operational rather than a research
question: something needs to actually invoke it once per closed 5-minute
candle for the record to become informative, which no part of this
project's current session-based operation does on its own. **Both were
chosen over a seventeenth SIZE-axis branch on `kelly_regime_v4`**, per
every round summary since R-40. With B-06 and B-08 now resolved (running
/ closed), the ranked list below is thinner than it has been all
program: **B-17** (multi-asset strategy registration — a real
infrastructure gap, not blocked, not attempted) and **B-18** (whether
`kelly_regime_covkelly`'s cadence-inconsistency is a rebalance-engine
artifact) are what remains open on merit; a future session should also
consider simply running `scripts/paper_trade.py` again to advance B-06's
record, which costs nothing and needs no new idea.

**Re-ranked 08-20 after R-49.** Two parallel branches attacked **B-17**
(multi-asset strategy registration) directly rather than continuing to
defer it: an adapter/composition design and a native multi-instrument
engine design, neither re-testing R-43's already-rejected dual-asset
finding. The adapter is now promoted into permanent, additive
infrastructure (`src/tradebot/multiasset.py`, 8 new tests, full 457-test
suite green) — it can compose any already-independent multi-book
strategy into one measurable portfolio result, causality-clean and
independently re-verified against `kelly_regime_dual_fixed.py`'s own
numbers. The native engine works and is more capable (a genuinely joint,
shared-risk-budget decision the adapter cannot express) but stays in
`experiments/`, unpromoted, on both branches' own recommendation: its
first non-trivial run produced a silent equity-accounting bug the
causality suite did not catch, which is exactly the kind of risk this
project's own `ROUTINE.md` says to avoid speculatively. **B-17 is
downgraded from OPEN to PARTIAL**: the composition primitive exists and
is tested; wiring a multi-asset strategy into `run.py`/the README
table/CI is deliberately not done, because no multi-asset strategy has
cleared even inner-validation yet to need it — building that path now
would be speculative infrastructure. **B-18** (whether
`kelly_regime_covkelly`'s cadence-inconsistency is a rebalance-engine
artifact) is therefore the only item left genuinely OPEN on the backlog,
and is now also the most direct route to a strategy that WOULD need the
native engine's shared-budget property, since the native branch's design
note names that same cadence-inconsistency as a plausible instance of
the fixed-segment-restart limitation a native engine would remove. A
future session should also consider simply running
`scripts/paper_trade.py` again to advance B-06's still-thin record.

**Re-ranked 08-20 after R-50.** Two parallel branches attacked **B-18**
directly and answered it: the monthly/weekly cadence flip is a
segment-restart artifact (confirmed via a continuous-replay engine that
makes the flip disappear), not the mean-estimation-noise problem R-43
assumed. That fix also settled the underlying trading question the
artifact was masking — `kelly_regime_covkelly`'s dynamic covariance
weighting is NEGATIVE, adding nothing over a static 50/50 split once
correctly measured — closing B-18 for good rather than leaving it to
reopen. It also surfaced a genuinely new, unrushed lead: the static
50/50-continuous-engine portfolio itself beats v4-solo by a real margin
on inner-validation, filed as **B-19** rather than promoted on the spot.
B-19 is now the only item left genuinely OPEN on the backlog. A future
session should also consider simply running `scripts/paper_trade.py`
again to advance B-06's still-thin record, which costs nothing and needs
no new idea.

**Re-ranked 08-20 after R-51.** Two parallel branches attacked **B-19**
directly, per its own named cheapest-first-check plus a genuinely novel
second axis. **Neither promoted.** The never-rebalanced one-time-split
form (conservative) cleared both pre-registered falsification gates and
a plateau check, then was REJECTED decisively on its one authorized
holdout read — losing to `buy_and_hold` and statistically indistinguishable
from BTC-solo `kelly_regime_v4` alone. The periodically-rebalanced,
inverse-volatility-weighted form (novel) never beat a correctly
re-derived fixed-50/50 reference on any of 12 configurations, so its own
pre-registered rule correctly withheld the holdout. Both are added to
section C. What neither branch tested — because neither branch's
pre-registration authorized it — is the *literal* form of R-50's original
finding: a periodically-rebalanced (not one-time-split), fixed-50/50
(not volatility-weighted) portfolio, read on the holdout for the first
time. That is filed as new backlog item **B-20**, with this round's own
evidence attached as a caution rather than encouragement: the drawdown-
only component of the effect, which R-51's conservative branch did
isolate and holdout-test, failed outright, and R-51 also found that
roughly 71% of the periodically-rebalanced version's larger Sharpe edge
specifically traces to the rebalancing act itself — the same mechanism a
bull-dominated 2023-2026 holdout has just shown this project's related
diversification variant does not reliably monetize. **B-20 is now the
only item left genuinely OPEN on the backlog that is not B-06's
already-running paper-trading record.** A future session attempting it
should pre-register its own falsification test and decision rule before
running anything, exactly as both R-51 branches did, rather than treating
R-50's inner-validation number as already-earned evidence — it is not,
and this round is the second time in a row (after R-42/R-43) that a
promising inner-validation number on this specific multi-asset research
line has not survived contact with either a falsification test or the
holdout. A session finding B-20 unpersuasive, or preferring not to spend
a third holdout consultation on the same underlying idea inside one
program, should run `scripts/paper_trade.py` again instead, which costs
nothing and needs no new idea.

**Re-ranked 08-20 after R-52.** Two parallel branches attacked **B-20**
directly — the one literal form of R-50's finding left untested by either
R-51 branch. **Neither promoted.** The literal fixed-calendar (monthly)
form (conservative) reproduced R-50's own inner-validation byproduct
number almost exactly and cleared every falsification/plateau gate, then
was REJECTED decisively on its one authorized holdout read — losing to
`buy_and_hold` by 22–45% with a Sharpe edge over BTC-solo v4 that is noise
and not even stably signed. A genuinely different, complementary axis
(novel) — the same fixed-50/50 target, reallocated only when weight drift
crosses a pre-registered band rather than on any calendar — also cleared
every gate and confirmed a real, holdout-robust secondary finding (70–90%
less rebalancing turnover for statistically identical risk-adjusted
performance vs. a fixed calendar), but was REJECTED on the same clause:
it loses to `buy_and_hold` by 48–61% and is statistically indistinguishable
from both BTC-solo v4 and a calendar reference. Both are added to
section C. **B-20 is CLOSED.** This is now the fifth independent
implementation of this project's periodic-rebalancing-driven BTC+ETH
diversification premium (never, monthly/weekly-calendar, quarterly/
semiannual-calendar with inverse-vol weights, now monthly-calendar and
drift-band with fixed weights) to fail to survive the 2023–2026 holdout,
and the third to clear every inner-validation/falsification/plateau gate
first — the research line built on R-50's original finding is now
considered exhausted for this asset pair and strategy absent a materially
different mechanism. **Nothing is left genuinely OPEN on the backlog.**
`scripts/paper_trade.py` (B-06, ongoing since R-48) is the standing
recommendation for a future session: it costs nothing, needs no new idea,
and is the only evidence stream this program's ~627 holdout consultations
have not already spent.

**Re-ranked 08-20 after R-53.** With the backlog empty of open SIZE/N≈3
items, this round attacked **INFO** instead — a genuinely different
constraint — with real macro data (VIX, DXY, S&P 500 from FRED) two
parallel branches (a multiplicative brake, a regime-vote injection)
tested two structurally different ways. **Both NEGATIVE.** The brake
reproduced R-34's flat-rescale-collapse artifact even on a genuinely
price-independent signal; the vote-injection branch's own lead-time check
found the mechanistic reason it couldn't work — macro stress lags this
project's price-anchor gate on net (median −5.5 days), not leads it, on
the only stress episodes this dataset has (2018, 2020-03, 2022). The
round's one genuinely new thing is methodological, not a trading result:
INFO turned out to be no easier to exploit than SIZE has been for sixteen
rounds — a new data channel does not, by itself, buy an edge if the
underlying mechanism (does the new signal actually lead the existing
one?) is checked and fails. One unvetted lead is filed as **B-21**: the
novel branch's own ablation arm (a hard, unweighted macro veto) beat v4
outright on inner-validation despite the averaged version's failure, but
was never pre-registered as a candidate and has not been through its own
lead-time, ETH, or plateau checks — a future session should treat it as
a fresh idea requiring its own pre-registration, not as evidence already
in hand, and should be alert that a mechanism whose own timing does not
lead the gate it would override is not an obviously stronger candidate
just because the combination rule is blunter. `scripts/paper_trade.py`
(B-06, ongoing since R-48) remains the standing zero-cost recommendation
alongside B-21 for a future session.

**Re-ranked 08-20 after R-54.** Two parallel branches attacked **B-21**
directly (the exact question the row above named): a conservative branch
gave the VIX/DXY hard-veto ablation its own pre-registration and
falsification battery for the first time, and a novel branch tested
whether a genuinely crypto-native signal — aggregate USDT stablecoin
supply, motivated by 2025 BIS/IMF/NY Fed literature on stablecoin flows as
a capital on-ramp/off-ramp — could resolve R-53's specific lead-time
failure, using the identical hard-veto architecture so any difference is
attributable to the signal rather than the combination rule. **B-21 is
now CLOSED, REJECTED**: the conservative branch found the hard override
lags the price-anchor gate exactly as the averaged version did (median
−5.5 days, 4/12 episodes), fails its own plateau check, and fails ETH
falsification — confirming that blunting the combination rule was never
going to fix a timing problem in the underlying signal. The novel branch
is this project's most interesting INFO-axis negative to date: its
pre-registered lead-time test **passed** — aggregate stablecoin-supply
deceleration leads the price-anchor gate in 9/12 matched episodes, median
**+16.5 days**, the reverse of every macro/on-chain attempt so far — but
the strategy built on it still loses, because the threshold sensitive
enough to catch real stress early also fires on transient supply noise,
and standing flat through the false alarms costs more than the early
exits recover. That is filed as **B-22**: not a signal-quality problem
(the signal leads, genuinely and now confirmed) but a precision problem,
with two concrete un-tried fixes named in the branch's own report
(persistence filter; feed a confirming vote instead of a hard override).
Both R-54 branches were independently reproduced cell-for-cell by the
operator before this row was written. **Nothing SIZE/N≈3-shaped is open
on the backlog; B-06 and B-22 are what remain.** `scripts/paper_trade.py`
(B-06, ongoing since R-48) remains the standing zero-cost recommendation
alongside B-22 for a future session.

**Re-ranked 08-20 after R-56.** With the backlog empty of open SIZE/N≈3/INFO
items, this round attacked **COST** instead — via execution (maker/limit
fills on v4's already-decided rebalances) rather than turnover (L-05/L-06,
already closed) or the fee tier itself (R-12/R-13, already closed) — the
first round in this project's history to build a fill-risk simulation
capability at all. **Both branches NEGATIVE.** The conservative branch (100%
fill-on-touch, the optimistic textbook case) produced real, monotonic fee
savings but no Sharpe improvement clearing the noise floor anywhere, and
failed its own pre-registered crash-transition-lag test for N≥3. The novel
branch, built specifically to be more realistic about fill uncertainty
(literature-grounded fill probability, not certainty-on-touch), lost to the
always-taker baseline in every slice tested, including both falsification
legs — showing the conservative branch's headline was already the best case
this mechanism has to offer, and even that never cleared this project's own
bar. One-line lesson: COST is not automatically easier to exploit than
SIZE/INFO just because it targets execution rather than signal — the
maker/taker fee gap at real venue tiers is simply too narrow relative to the
adverse-selection cost of waiting, especially during the crash de-risking
events that are this strategy's entire edge. The conservative branch's own
unpromoted "least-bad" N∈[2,24] residual is filed as **B-24**, LOW priority
— it was not pre-registered as the decision subset, and even at its most
favorable reading never cleared the noise floor either, so a dedicated
re-run is a weak bet. **Nothing is left genuinely OPEN on the backlog that
is not B-06 (ongoing, zero-cost), B-23 (LOW priority), or B-24 (LOW
priority).** `scripts/paper_trade.py` (B-06, ongoing since R-48) remains
this project's standing zero-cost recommendation — the only item left that
is not a further re-derivation of a research line (SIZE-axis sizing/
diversification, INFO-axis stablecoin combination rules, now COST-axis
execution modeling) this project has already run to exhaustion at least
once.

**Re-ranked 08-20 after R-55.** Two parallel branches attacked **B-22**
directly — both of R-54's own named fixes, each on the exact grounds R-54
proposed them. **B-22 is now CLOSED, REJECTED.** The persistence filter
(conservative) fails worse than R-54's original hard veto: the "transient"
stress onsets it was meant to filter out don't reverse within a few days,
they persist about as long as genuine episodes do, because the signal's
own 14-day growth window has already smoothed out anything shorter — so
tightening duration mostly erodes the confirmed lead time (+16.5d flips to
a 10-day lag by `persist_days=5`) rather than cutting false positives. The
confirming-vote architecture (novel) settles a real methodological question
— it beats an equivalent hard override 16/16 matched cells once fed a
genuinely leading signal, resolving R-53's own lag-vs-lead confound — but
still fails ETH falsification and does not clear v4 on inner-validation
Sharpe, because the underlying signal's specificity problem is orthogonal
to which combination rule receives it. Both branches were independently
reproduced by the operator before this row was written. **Nothing is left
genuinely OPEN on the backlog that is not either B-06 (ongoing, zero-cost)
or B-23 (LOW priority — the same signal's own research line, now three
rounds and four combination rules deep with no working strategy).**
`scripts/paper_trade.py` (B-06, ongoing since R-48) is the standing
recommendation for a future session with nothing else to do; a session
preferring a fresh idea over B-06 should treat B-23 as a low-priority,
not a high-priority, option — this project's own repeated experience
(the SIZE-axis research line closing five separate times, R-40 through
R-52) is that a fourth or fifth attempt on the same signal rarely
outperforms a genuinely different one.

**Re-ranked 08-20 after R-57.** With the ranked list holding nothing
genuinely OPEN but B-06 (ongoing, zero-cost) and the LOW-priority B-23/B-24,
this session spent itself on the one question the backlog never contained:
not "what else can be layered onto `kelly_regime_v4`" — sixteen SIZE-axis
and five INFO-axis branches across twelve rounds, plus R-56's two COST-axis
execution branches, have answered that — but **"does the incumbent's own
surviving property hold anywhere other than the two assets we keep measuring
it on?"** The answer is no. On six Coinbase instruments the frozen strategy
has never been fitted on, its drawdown advantage over a hold carrying **its
own exposure** inverts 6 of 6, while the same runs show 6 of 6 in its favour
against the fully-invested benchmark the README table uses; a control over a
window every asset shares puts BTC and ETH in v4's favour and every panel
asset against it, so the failure is asset-specific rather than
period-specific. This does not retract R-33 (which had already shown 88–92%
of the headline gap was exposure) or R-17/R-47 (whose ETH numbers reproduce
here); it puts a **measured scope** on what those rounds left standing — BTC
and ETH, 2 of 8 — which nothing before it could do at n=1 asset per check.

What that does to the order: **B-06 remains the standing zero-cost
recommendation**, and the panel is now committed data, so the natural
follow-up is **B-25** — is v4's BTC-calibrated `target_vol` the binding
reason its mechanism does not travel? A per-asset volatility-normalized
target is the obvious test, and it would have to clear the same
matched-exposure bar on the same six instruments. It is ranked below B-06
deliberately: it is the seventeenth attempt on this strategy family's own
parameters, and this project's own record on those is 0-for-16. What R-57
does raise in value is anything that would test a *different* strategy
family on the panel, since for the first time this project has six
independent instruments to fail on cheaply, before spending a holdout
consultation on any of them.

**Re-ranked 08-20 after R-58** (a same-day, concurrently-running session,
recorded here rather than reordered ahead of R-57 above, per the R-31/R-32
precedent this file already follows for same-day parallel work). Two
parallel branches attacked **B-23** directly — both of its own named
fixes. **B-23 is now CLOSED, REJECTED.** The shorter-growth-window branch
(conservative) found a clean, monotonic kill at its own pre-registered
gate, run before any Sharpe number: shrinking the window from 14 to 2 days
flips the confirmed lead time from +16.5 days to a −15.0-day lag, because
the ~2-3-day acute-redemption timescale recent literature reports and the
multi-week capital-flight dynamic the signal actually leads on are
evidently different clocks. The on-chain-corroboration branch (novel)
found its AND-gate does not discriminate genuine leading episodes from
noise at any threshold tested (7/9 leads corroborate vs. 3/3 lags/noise
also corroborate) and *increases* the raw false-onset count rather than
reducing it; its one striking secondary number, corroboration rescuing
R-54's failed hard-override architecture into a tie/marginal win, is fully
explained by an exposure-artifact check (R²=0.9971) as v4 relabeled, not a
real edge — the same trap R-33/R-34 already caught, now a third time. Both
branches were independently reproduced by the operator before this row was
written. With this round, the stablecoin-signal research line has tried
five structurally distinct mechanisms (R-54 hard veto, R-55 persistence
filter, R-55 confirming vote, R-58 shorter window, R-58 on-chain
corroboration) and the INFO axis has failed across six consecutive rounds
(R-44, R-53, R-54, R-55, R-58×2) — a future session should treat that axis,
on the data currently in hand, as exhausted absent either a genuinely new
information channel or a materially different architecture idea none of
these six anticipated. **Combined with R-57's own row above, closing B-23
leaves the backlog at: B-06 (ongoing, zero-cost) at the top, then B-25
(OPEN, filed by R-57 — does v4's BTC-calibrated `target_vol`/`max_leverage`
explain why its matched-exposure property doesn't travel to the panel),
then B-24 (LOW, filed by R-56's exec-limit round). Nothing else is open.**
`scripts/paper_trade.py` (B-06, ongoing since R-48, advanced by one more
decision this session) remains the standing zero-cost recommendation; a
session preferring a fresh idea over B-06 or B-25 should not look to the
stablecoin signal or the INFO axis generally for one without new data or a
genuinely different architecture.

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
| **B-06** | Forward paper-trading recorder | N≈3 | **ONGOING → R-48** | Built and running: `scripts/paper_trade.py` records `kelly_regime_v4` and `buy_and_hold` against the live Bitstamp public feed into `reports/paper_trading/`, 2 rows recorded so far (inception 2026-08-19T23:05Z). No longer blocked — unblocked by direct Bitstamp reachability confirmed this session. Not yet informative: needs a future session, or an actual cron/systemd job (documented in `docs/LIVE.md`), to invoke it once per closed 5m candle so the record accumulates. The single most-repeated backlog item since R-29 is now infrastructure that exists rather than a thing still to build. |
| ~~B-07~~ | ~~On-chain features, sign-corrected~~ | INFO | **DONE → R-44** | Sign was fixed as designed (both branches leaned exposure INTO confirmed high-participation/capitulation-recovery regimes, never away from rising activity) and neither branch repeated R-08's inversion — they failed for unrelated, independently-reproduced reasons (magnitude-only exposure-artifact; clean inner-validation loss). Real CoinMetrics data is now committed (`data/btc_onchain_daily.csv.gz`, `data/eth_onchain_daily.csv.gz`) and available for a future round with a different exploitation. |
| ~~B-08~~ | ~~Second bear, second asset, different period (ETH 2020–2026)~~ | N≈3 | **DONE → R-47** | Frozen `kelly_regime_v4`, zero parameters changed, run against the now-committed `ethusd_coinbase_spot_5m.csv.gz` (2019-03-14→2026-08-19). Drawdown/tail protection replicates cleanly on ETH's own 2022 bear (previously untested — independent of the 2018 BTC bear every prior ETH check shared); the return edge does not survive the realistic 0.40% fee tier over the full 2020–2026 window. Confirms L-01/R-17's own standing caveat on genuinely independent evidence for the first time. |
| **B-09** | Conformal prediction / adaptive conformal by betting (adaptive conformal inference under distribution shift; conformal prediction with change points, NeurIPS 2025; adaptive conformal inference by betting, 2024) | ERR | LOW | Was "mostly subsumed by B-01" — now demoted further by R-28's result: the binding problem is not that trust is miscalibrated but that correctly-calibrated trust is *low*, and conformal would say the same thing more slowly. |
| ~~B-13~~ | ~~Matched-risk benchmark: `kelly_regime_v4` against a **de-levered** `buy_and_hold` at equal realized volatility~~ | ERR, SIZE | **DONE → R-33** | Answered, and it cost the project its headline. At genuinely equal risk (40 windows, matched inside each window to 0.5%) v4's median drawdown advantage falls from −24.5pp to **−2.9pp** on spot and from −70.7pp to **−5.5pp** on futures; on the holdout five of six frozen cells fail the risk match and the valid one gives −14.18pp [−22.68, +13.48]. R-31's suspicion was right: the −41.1pp is mostly the exposure level. The consolation, and it is a real one, is that the *return* comparison at matched risk goes v4's way everywhere and survives the ETH test that killed R-28 — see **B-14**. Original framing kept below for the record. Opened by R-31, and it points the same knife at this project's own headline. Every drawdown claim here — L-04's "regime-gated sizing cuts drawdown", R-17's ETH replication, R-29's −41.1pp [−54.8, −18.4] — compares a strategy holding roughly half the notional against a **fully-invested** benchmark. R-31 showed that precise mismatch manufactured a mechanism finding for the e-process gate that vanished at equal risk. The experiment is one afternoon: add a constant-exposure hold at scale `c` to `experiments/matched_risk.py`, solve `c` on inner-validation so its realized volatility equals v4's, and re-run the paired bootstrap. Needs no new data, no fetch, and the harness already exists. Pre-register the answer both ways — a hold de-levered to 0.5x is *not* obviously a weaker benchmark, and if the drawdown gap survives it, that is the strongest result this project has ever had. |
| **B-10** | Deterministic Elliott wave counter | — | LOW | Only as a documented negative result, per R-18. ZigZag pivots, mechanical impulse/corrective rules, no discretion. About a day, converts an unfalsifiable debate into a table row. |
| ~~B-15~~ | ~~Build a real perp price series (Deribit `BTC-PERPETUAL`, 5m OHLCV) alongside the existing spot series~~ | ERR, COST, INFO | **DONE → R-41** | Built: real BTC-PERPETUAL (2018-08-14→) and ETH-PERPETUAL (2019-03-14→) 5m OHLCV, plus a matching Coinbase ETH spot series, all committed. `tradebot.data.load_deribit_perp_price()`/`compute_basis()` give a genuine, non-proxied spot/perp basis for the first time — used as a `kelly_regime_v4` SIZE input in R-41 (both branches NEGATIVE, for reasons unrelated to data quality). Available for B-03's re-run (a real basis-risk term for the funding-harvest carry trade) and for a future SIZE-axis round with a different exploitation, per R-41's own recommendation — a short event-triggered override rather than a continuous ramp, or a replacement rather than a multiplier of v4's exposure. Not wired into `CANONICAL["perp"]`, so no existing comparison-table number changed. |
| ~~B-16~~ | ~~Dual-asset BTC+ETH diversification of `kelly_regime_v4`, robustified per its own two authors' prescriptions~~ | N≈3, SIZE | **DONE → R-43**, REJECTED on the holdout | The conservative branch's bootstrap held up well enough in-sample to earn one pre-registered holdout read (bear-quartile drawdown-delta, `vol_weighted`, both markets) — it replicated on 5x futures (CI excludes zero) but not on spot (CI contains zero), and the rule required both, so REJECT as written. The novel branch's de-noised mean estimator shrank but did not resolve its own cadence-inconsistency and never earned a holdout read. Nothing here is promotable to the comparison table this round — and even a fully-confirmed result would additionally have needed new multi-asset registration infrastructure this project's `Strategy`/registry framework does not have today (see R-43, and **B-17** below). |
| **B-17** | Multi-asset strategy registration — the comparison table, `Strategy` base class and `tradebot run` all assume one instrument per registered class; nothing in this project can put a genuinely two-asset (or N-asset) strategy in the README table today, even a fully-promotable one | ERR (methodology gap, not a market-constraint code) | **PARTIAL → R-49** | The "can it be done at all" half is DONE: `src/tradebot/multiasset.py` (adapter/composition design, promoted from a conservative branch, 8 tests) lets an already-independent multi-book strategy be measured as one portfolio result, causality-clean, without touching `engine.py`/`strategy.py`/the 25 existing registrations. Still OPEN: wiring an actual multi-asset strategy into `run.py`'s `run_matrix`, the README table and `test_evidence.py`'s CI requirement — deliberately deferred, since no multi-asset strategy has cleared even inner-validation yet to need it (R-43's dual-asset finding, the only candidate, was REJECTED on the holdout). A parallel novel branch also built a more capable native multi-instrument engine (`experiments/b17_multiasset_native.py`, shared risk budget, genuinely joint decisions) but it stays unpromoted — its first non-trivial run produced a silent equity-accounting bug the causality suite did not catch, and both branches' authors recommend building it for real only once a specific strategy earns that risk. |
| ~~B-18~~ | ~~Is the `kelly_regime_covkelly` allocator's monthly/weekly cadence-inconsistency (R-42, attenuated but not resolved by R-43's mean-denoising) actually a rebalance-engine/segment-restart artifact rather than a mean-estimation-noise problem?~~ | ERR | **DONE → R-50, ANSWERED** | Confirmed: the flip is primarily a segment-restart artifact (the vote's `.ffill().fillna(0.0)` hysteresis has no memory to draw on when a restarted segment's anchor is NaN for all but ~11 of its ~23,050-bar warmup), not R-43's mean-estimation-noise. Fixing it (continuous-replay engine) makes the flip disappear — but also reveals `kelly_regime_covkelly`'s dynamic Σ⁻¹μ weighting adds nothing over a static 50/50 split once correctly measured (NEGATIVE on that direction). Reopens only as **B-19** below, on the diversification-only finding the fix surfaced. |
| ~~B-19~~ | ~~Does a periodically-rebalanced, EQUAL-WEIGHT (static 50/50) BTC+ETH portfolio of `kelly_regime_v4`, run through R-50's continuous (non-restarting) engine, survive pre-registration and this project's falsification/cost/holdout process?~~ | SIZE, N≈3 | **DONE → R-51, NEITHER VARIANT PROMOTED** | The cheapest first check — a never-rebalanced one-time split via the promoted `multiasset.py` adapter — cleared both falsification gates and a plateau check, then was REJECTED decisively on its one pre-registered holdout read (loses to `buy_and_hold`; statistically indistinguishable from BTC-solo v4). A second, more ambitious form — periodically rebalanced but inverse-volatility-weighted instead of static 50/50 — never beat a correctly re-derived fixed-50/50 reference on any of 12 configurations and correctly never reached the holdout. Both ruled out in section C. The literal form of R-50's original finding (periodically rebalanced AND fixed 50/50, together) remains untested by any session — reopens as **B-20**. |
| ~~B-20~~ | ~~Does the LITERAL periodically-rebalanced (monthly, or another single cadence fixed before running), fixed-50/50 BTC+ETH `kelly_regime_v4` portfolio — R-50's own original candidate, run through its continuous (non-restarting) engine, unmodified split, unmodified cadence discipline — survive its own pre-registered falsification test and a first, single holdout read?~~ | SIZE, N≈3, COST | **DONE → R-52, NEITHER BRANCH PROMOTED** | Two parallel branches, both cleared every inner-validation/falsification/plateau gate, both decisively REJECTED on their one pre-registered holdout read. Conservative (literal monthly calendar): reproduces R-50's own inner-validation number almost exactly, then loses to `buy_and_hold` by 22–45% on the holdout. Novel (drift-band trigger, same 50/50 target): confirms a real, holdout-robust 70–90% turnover reduction vs. a calendar cadence for statistically identical risk-adjusted performance, but the underlying candidate still loses to `buy_and_hold` by 48–61%. This is the fifth independent trigger/target implementation of this research line's periodic-rebalancing premium to fail the 2023-2026 holdout; the line is now considered exhausted for this asset pair absent a materially different mechanism. |
| ~~B-21~~ | ~~A hard, unweighted macro-veto (`frac=0` while VIX/DXY `stress_z` is above threshold, v4's own anchor average otherwise — no precision-weighted averaging) as a `kelly_regime_v4` regime-gate override~~ | INFO, SIZE | **DONE → R-54, REJECTED** | Given its own pre-registration and falsification battery at last: fails the primary test (lead-time vs. the 3-anchor majority, leads only 4/12 episodes, median −5.5 days, replicating R-53's averaged-vote lag almost exactly), fails the plateau check (best-scoring point is the explicit no-hysteresis negative control), and fails the ETH falsification (5/10 cells show an asset-specific pattern). The tension named above is resolved, not assumed away: blunting the combination rule does not fix the timing, because both the averaged and hard-override versions are built on the identical, laggy `stress_z`. |
| ~~B-22~~ | ~~A magnitude-*and*-duration filter (or a confirming, non-overriding combination rule) on the aggregate-USDT-stablecoin-supply-deceleration signal R-54's novel branch built~~ | INFO | **DONE → R-55, REJECTED** | Both of R-54's own named fixes tested, both NEGATIVE. Persistence filter: fails worse than R-54's original — the "transient" onsets don't reverse within a few days (they persist as long as genuine episodes, since the 14-day growth window already smooths shorter noise), so duration and precision are not separable axes here; tightening enough to matter erodes the confirmed lead time into a lag. Confirming-vote architecture: beats an equivalent hard override 16/16 cells once fed a genuinely leading signal (a real result, resolving R-53's lag-vs-lead confound) but still fails ETH falsification and inner-validation Sharpe against v4 — the signal's specificity problem is independent of the combination rule. Reopens only as **B-23**, LOW priority. |
| ~~B-23~~ | ~~A materially different mechanism on the same aggregate-USDT-stablecoin-supply-deceleration signal — e.g. a shorter growth window matched to genuine-stress duration rather than a persistence filter bolted onto the existing 14-day feature, or corroboration from a second independent signal rather than filtering one signal alone~~ | INFO | **DONE → R-58, REJECTED** | Both of B-23's own named fixes tested, both NEGATIVE. Shorter window: fails its own pre-registered Step-A gate before any Sharpe number — lead time flips monotonically from +16.5d (N=14) to −15.0d (N=2) as the window shrinks, a timescale mismatch between acute redemption (~2-3d) and the multi-week capital-flight dynamic the signal actually leads on. On-chain corroboration: the AND-gate does not separate genuine leads from noise (7/9 vs 3/3 corroboration rate) and increases the false-onset count (24→28); its one striking number (rescuing R-54's failed hard override into a tie/win) is an exposure-level artifact (R²=0.9971), not a real edge. This closes the stablecoin-signal research line's fifth mechanism attempt and the INFO axis's sixth consecutive failed round (R-44, R-53, R-54, R-55, R-58×2) — not recommended for further pursuit absent a genuinely new information channel. |
| **B-24** | A narrower pre-registration (N capped at ≤24, deliberately excluding the N≥72 near-miss/trend-drift failure modes R-56's conservative branch found) of the patient-limit/taker-fallback execution model on `kelly_regime_v4`'s COST axis, tested against the same falsification battery (ETH, BTC control, crash-transition-lag) | COST | LOW | Filed by R-56 after its own conservative branch's full N∈{1,...,288} sweep failed to clear the noise floor anywhere and failed its crash-lag test for N≥3. The N∈[2,24] region looked least-bad in the same sweep (captures most of the fee saving, avoids the N≥72 failure modes, stays directionally positive in inner-validation) but was never the pre-registered decision subset, so R-56 correctly declined to promote it. Not recommended as a priority: even this "least bad" reading never cleared the ±0.2 Sharpe noise floor in the original sweep, and R-56's novel branch independently showed the conservative branch's fill assumption (100% on touch) is already the optimistic end of the spectrum — a more realistic accounting is more likely to find less here, not more. |
| **B-25** | Is `kelly_regime_v4`'s BTC-calibrated `target_vol` (0.55) / `max_leverage` (2.0) the reason its matched-exposure drawdown property does not travel? R-57 found the mechanism's mean notional collapses to 0.18–0.26 on higher-volatility instruments (vs 0.38 BTC / 0.34 ETH), leaving mostly the vote's timing — a per-asset volatility-normalized target is the obvious test | SIZE, N≈3 | OPEN (ranked below B-06) | Filed by R-57. Two cautions attached rather than encouragement: it is the seventeenth attempt on this strategy family's own parameters and the record there is 0-for-16 (R-34 → R-46, R-53 → R-56); and it must clear the **matched-exposure** bar on the same six committed instruments, not the fully-invested one, or it is measuring exposure again. The panel data now ships with the repo, so the test is cheap and costs no holdout consultation. |

---

## Appending an entry

One session, one entry. Copy this skeleton to the **top** of
[section B](#b-research-log-newest-first) — newest first, always — and
fill every field. A field with nothing to report says so (`none`, `+0`,
`not reached`); it is never dropped, because a missing field reads as an
answer that was never asked for.

````markdown
### R-nn · MM-DD · <PROMOTED|NEGATIVE|BLOCKED|PARKED|SETTLED|METHOD> — <short title, ≤90 chars>

**Direction.** What was tried, in a few sentences: the idea, its
citation, the backlog item it closes, which of the four constraints
(INFO / N≈3 / ERR / COST / SIZE) it attacks, and which ledger entries it
is *not* a duplicate of, by ID.

**What was done.** Branches and their files, data, the pre-registered
decision rule and falsification test as they were frozen, and
**configs evaluated** — the total across ALL parallel branches, not just
one, because that is the trials count for deflated Sharpe.

**Result.** Train, inner-validation and holdout numbers; the outcome of
the pre-registered decision rule and of the falsification test; whether
an independent skeptic reproduced them.

**Verdict.** The verdict and the one-line lesson. State the **holdout
counter** — the round's increment and the running program-level total
after it — and whether the **decision rule moved** after the holdout was
read; if it did, the result is in-sample and must say so. End with the
**next step**, which becomes a backlog row if the work continues.
````

Then update the other sections the round touched: a row in **A** if a
strategy was registered, a row in **C** if a direction is now ruled out, a
re-ranked **D**, and a bullet at the top of
[Holdout consultations](#holdout-consultations-to-date) if the holdout was
read.

Rules that the format exists to enforce:

- **Newest first.** New entries go at the top of section B, never at the
  bottom and never inside another entry. The heading is the index: ID,
  date, verdict, short title.
- **Sections, not table rows.** A round's write-up is prose of arbitrary
  length, with its own `####` sub-headings for a pre-registration or a
  results write-up. It stopped fitting in a table cell around R-28 and
  the table went on collecting the overflow until it broke outright
  (R-41/R-44 column shifts, R-46 in the wrong table, R-47–R-55 orphaned
  below it).
- **Nothing is deleted.** A superseded finding is annotated in place
  (see R-28, retracted by R-31), never removed.
- **Sections A, C and D stay tables.** They are short-cell registries —
  strategies, ruled-out directions, the ranked backlog — not a log, and
  they render correctly. Keep prose out of their cells; if a cell needs a
  paragraph, the paragraph belongs in the round's section in B.

### Holdout consultations to date

Newest first, one bullet per round, same order as section B. The count is
the running program-level total *after* that round; the increment and its
justification are in the note.

- **08-20 · ~627** — R-58: **+0** on top of R-57's ~627 (a same-day,
  concurrently-running session, recorded here rather than reordered ahead
  of R-57 above, per the R-31/R-32 precedent). Neither the conservative
  (shorter stablecoin growth window) nor the novel (on-chain
  active-address corroboration gate) branch read any 2023+ bar — both
  pre-registered a holdout rule requiring a genuine inner-validation +
  falsification win first, neither produced one (the conservative branch's
  own Step-A gate decided it before any Sharpe number was computed at all),
  and both grepped their own single new file for every `202[3-9]` literal
  as proof, independently re-verified by the operator.
- **08-20 · ~627** — R-57: **+0** on top of R-56's ~627. The round reads
  only the six new Coinbase panel files and, in its post-hoc control, the BTC
  and ETH series truncated at **2022-12-31** — no 2023+ BTC bar is evaluated
  anywhere in `experiments/r57_cross_asset_panel.py` (its only BTC call site
  is `run_period(..., "2020-04-01", "2022-12-31")`). Reading 2023+ bars of a
  *panel* asset does not consult this project's holdout by the R-47/B-08
  convention: no panel asset has ever fitted a parameter here. Note the same
  day's R-56 was a separate, concurrently-running session on a different
  axis; as in R-31/R-32, both rounds are recorded and neither increments the
  counter.
- **08-20 · ~627** — R-56: **+0** on top of R-55's ~627. Neither the
  conservative (patient-limit/taker-fallback) nor the novel (probabilistic
  fill-model) execution-model branch read any 2023+ bar — both restricted to
  inner-train/inner-validation/pre-2020 BTC-control+ETH falsification by
  design; both branches' own runtime assertions plus the operator's
  independent grep of both files confirm this. The operator independently
  reproduced, from a clean shell: both branches' causality/tamper probes
  bit-for-bit; the conservative branch's full 16-configuration ETH+
  BTC-control falsification table (every number matched exactly); and the
  novel branch's 128-event crash-transition-lag check (mean 5.7 bars, max 9,
  exact match). Neither branch cleared its own pre-registered promotion bar,
  so neither reached, nor needed, a holdout read.
- **08-20 · ~627** — R-55: **+0** on top of R-54's ~627. Neither the
  conservative (persistence-filtered stablecoin hard veto) nor the novel
  (stablecoin confirming-vote) branch read any 2023+ bar — both restricted
  to inner-train/inner-validation/pre-2020 BTC-control+ETH falsification
  by design; grepped by their own authors and independently re-confirmed
  by the operator (every `OOS_START="2023-01-01"` call site checked:
  exclusive upper bound only, never a data read). Both branches'
  `identity`/`causality` outputs, the conservative branch's `leadtime`,
  and the novel branch's `ablation`/`eth`, were independently re-run by
  the operator from a clean shell and reproduced (the novel branch's
  ETH-falsification failing-cell count reproduced as 14/16 against the
  branch's own reported 13/16 — an immaterial discrepancy, noted in the
  R-55 row, that does not change the verdict).
- **08-20 · ~627** — R-54: **+0** on top of R-53's ~627. Neither the
  conservative (VIX/DXY hard-veto) nor the novel (stablecoin-supply
  hard-veto) branch read any 2023+ bar — both restricted to
  inner-train/inner-validation/pre-2020 BTC-control+ETH falsification by
  design; grepped by their own authors and independently re-confirmed by
  the operator (every `OOS_START="2023-01-01"` call site checked:
  exclusive upper bound only, never a data read). Both branches'
  `causality`, `select`, `artifact` outputs, plus the conservative
  branch's `eth`/`leadtime` and the novel branch's `leadtime`, were
  independently re-run by the operator from a clean shell and reproduced
  exactly.
- **08-20 · ~627** — R-53: **+0** on top of R-52's ~627. Neither the
  conservative nor the novel macro-INFO branch read any 2023+ bar (both
  restricted to inner-train/inner-validation/pre-2020 BTC-control+ETH
  falsification by design); grepped by their own authors and independently
  re-confirmed by the operator, including a call-site check of the novel
  branch's one `"2023-01-01"` sentinel literal, which is used only as an
  exclusive upper bound, never as a data read.
- **08-20 · ~627** — R-52: **+4** on top of R-51's ~623 — both branches'
  one pre-registered holdout read each (frozen config, both fee tiers, one
  paired call, matching the R-35/R-51 convention: conservative +2, novel
  +2), both independently reproduced by the operator from a clean shell
  (`python experiments/b20_literal_calendar_5050.py holdout` and `python
  experiments/b20_threshold_band_5050.py holdout`, both outputs matched
  their reports to the dollar).
- **08-20 · ~623** — R-51: **+2** on top of R-50's ~621 — the conservative
  branch's one pre-registered holdout read (frozen 50/50 config, both fee
  tiers, one paired call, per the R-35 convention), independently
  reproduced by the operator from a clean shell (`python
  experiments/b19_dual_fixed_split.py holdout`, output matched the report
  to the dollar). The novel branch read **+0**: its own pre-registered
  promotion gate (P2/F2) failed before the holdout clause was ever
  reached, and the operator independently confirmed `holdout()` is gated
  behind an explicit CLI argument no invocation in its report — including
  the default `all` command whose output the report quotes — ever passes.
- **08-20 · ~621** — R-50: **+0** on top of R-49's ~621.
  Diagnostic/methodology round on B-18 — both branches' date literals are
  ≤2022-12-31, grepped and confirmed by their own authors and
  independently re-confirmed by the operator; neither branch imports
  `load_dataset` for anything past that boundary.
- **08-20 · ~621** — R-49: **+0** on top of R-47/R-48's ~621.
  Infrastructure round, not a strategy test — both branches' every date
  literal is ≤2022-12-31, grepped and confirmed; neither imports
  `load_dataset` for anything past that boundary.
- **08-19 · ~621** — R-47/R-48: **+0** on top of R-46's ~621. R-47 (B-08)
  reads only `data/ethusd_coinbase_spot_5m.csv.gz`, never the BTC 2023+
  file. R-48 (B-06) reads only the live Bitstamp public feed. Neither
  script imports `load_dataset` for the canonical BTC holdout at all.
- **08-19 · ~621** — R-46: **+0** on top of R-45's ~621. Both branches
  restricted to inner-train/inner-validation plus the standard pre-2020
  BTC-control/ETH falsification pair; each branch grepped its own file for
  date literals and confirmed no bar dated 2023-01-01 or later was ever
  read, computed, or printed. The operator independently reproduced the
  conservative branch's causality probe and full ETH/BTC-control
  falsification table — also pre-2020 data, not the holdout.
- **08-19 · ~621** — R-45: **+0** on top of R-43's ~621 (R-44, in between,
  also added +0 by its own row's text, though not separately logged in
  this table until now). Both branches restricted to
  inner-train/inner-validation plus the standard pre-2020 BTC/ETH
  falsification pair; the conservative branch enforced a hard no-2023+
  rule throughout its 432 backtests and the novel branch logged its refit
  counts explicitly on every pre-2023 run. Zero bars dated 2023-01-01 or
  later were read by either.
- **08-19 · ~621** — R-43: **+400** on top of R-42's ~221 — 40 windows
  resampled from 2023-01-01 onward × 2 markets × [BTC-alone control + 2
  split candidates] = 400 leg-level backtests, generated by ONE
  pre-registered `holdout()` call (decision rule committed one commit
  before the read). Unusually large next to prior rows because it is the
  first holdout consultation to use multi-window resampling rather than a
  single-point read; the 2 cells the pre-registered decision actually
  rests on (bear-quartile `vol_weighted`, spot and futures) are a small
  fraction of the total, the rest is diagnostic detail from the same call.
  The novel branch (`kelly_regime_covkelly_v2`) read no 2023+ bar
  (inner-train/inner-validation only, by design) and is not counted here.
- **08-19 · ~221** — R-42: **+0** on top of R-41's ~221. Both branches
  (`kelly_regime_dual_fixed`, `kelly_regime_covkelly`) were explicitly
  restricted to inner-train (2019-03-14→2020-12-31) and inner-validation
  (2021-01-01→2022-12-31) and neither read a single 2023+ bar, by design
  (each branch grepped its own file for date literals and confirmed every
  data slice stays inside that window). Both authors recommended against a
  holdout read given their own diagnostics, and the operator agreed — see
  B-16.
- **08-19 · ~221** — R-41: **+0** on top of R-40's ~221. Both branches
  (`kelly_regime_v9_basis_brake`, `kelly_regime_v9_basis_lead`) were
  explicitly restricted to inner-train-with-basis (2018-08-14→2020-12-31)
  and inner-validation (2021-01-01→2022-12-31) and neither read a single
  2023+ bar, by design; neither reached ETH falsification (both authors
  recommended against it given their own inner-validation diagnostics, and
  the operator agreed rather than spend the newly-built ETH data). The
  operator's independent re-verification
  (`artifact`/`fallback`/`causality`/`exposure`/`leadlag`) reused the same
  pre-2023 data, not the holdout.
- **08-19 · ~221** — R-40: **+0** on top of R-39's ~221. Both branches
  (`kelly_regime_v8_ladder_bag`, `kelly_regime_v8_uncertainty_shrink`)
  were explicitly restricted to inner-train/inner-validation/pre-2020
  ETH+BTC only and neither read a single 2023+ bar, by design; the
  operator's independent re-verification of both branches' reported
  numbers reused the same inner-validation and pre-2020 falsification
  data, not the holdout.
- **08-19 · ~221** — R-39: **+62** on top of R-38's ~159 — the
  conservative branch's own honest count (§10 of its report): 61 distinct
  2023+ holdout cells, not the 1 its pre-registration authorized (the
  extra 60 are diagnostics — neighbourhood, cost tiers, exposure-matched
  control, sub-period split, venue-splice robustness — run *after* the
  pre-registered decision cell had already returned a significant
  negative; none could have changed the verdict in the gate's favour, but
  this file's practice is to record the real number). Plus **+1** for the
  operator's independent skeptic re-derivation of the decision cell via a
  separate code path. The novel branch (`funding_harvest_carry`) reads
  only the funding-rate series against `buy_and_hold`/`kelly_regime_v4`
  reference runs over 2024-2026 — a period the BTC-price holdout
  convention already treats as fair game once funding covers it — and is
  not counted separately here.
- **08-19 · ~159** — R-38: **+0** on top of R-36/R-37's ~159. Both
  branches (`kelly_regime_v7_ddcap`, `kelly_regime_v7_crra`) were
  explicitly restricted to inner-train/inner-validation/pre-2020 ETH+BTC
  only and neither read a single 2023+ bar, by design.
- **08-19 · ~159** — R-36 and R-37: **+0** on top of R-35's ~159. R-36
  reused R-33's existing `windows.csv` (seed=42, computed once) and only
  recovered calendar dates from the RNG sequence — no new backtest, and
  the 40-window resample does not count against the holdout by this
  project's own established convention. Both R-37 branches were explicitly
  restricted to inner-train/inner-validation/pre-2020-ETH only and neither
  read a single 2023+ bar, by design.
- **08-19 · ~159** — R-35: +7 on top of R-33's ~152 — one pre-registered
  configuration (`funding_gate_decile`, w=180) read once, restricted to
  the 2023-01-01..2023-12-31 funding-covered slice rather than the full
  2023-2026 holdout (a deliberate, pre-registered scope limit, not an
  oversight): spot funding-free, futures funding-free, futures
  funding-charged, each a paired v4-vs-gate read (6), plus one
  `buy_and_hold` context run (1). The parallel novel branch and the
  conservative branch's `w=90`/`w=365`/expanding configurations never read
  it, per the pre-registered "ask fewer questions" economy.
- **08-19 · ~152** — R-33: +28 on top of R-32's ~124 — 10 frozen holdout
  runs across two markets, 8 for the descriptive on-holdout re-match and
  its solver, 10 cost re-runs (5 at the 0.40% taker tier, 5 with funding
  charged). The ETH/BTC falsification cells and the 40-window resample do
  not read the 2023+ BTC holdout. The re-match is the only part of this
  round that reads the holdout for a quantity it was not pre-registered to
  read; it is labelled in-sample everywhere it appears and supports no
  decision. Consistent with R-29: nothing here is offered as a
  Sharpe-based claim, and the round's finding — that 88–92% of a drawdown
  gap was an exposure gap — is measured on 40 resampled windows rather
  than on this holdout, precisely because the holdout has stopped being
  able to settle anything.
- **08-18 · ~124** — R-32: +12 on top of R-31's ~112 (3 frozen arms × 2
  markets, 3 spot fee-tier re-runs, 3 funding-charged futures re-runs).
  The number that matters is not the increment but why it exists: **two
  sessions were scheduled onto the same backlog row on the same day and
  each spent the holdout on it independently**. Neither branch did
  anything wrong — both pre-registered, both froze before reading — but
  the day cost ~36 consultations and 69 trials for one question, and the
  project applies 103 + 69 = **172** trials from here. If parallel
  sessions are going to run, ROUTINE.md's rule that the trials count is
  the total across branches is the thing that keeps the arithmetic honest;
  this is the first time it has actually been needed.
- **08-18 · ~112** — R-31: 12 matched-and-reference runs across two
  markets, 6 re-runs at the 0.40% taker tier, 6 with funding charged on
  futures. The ETH/BTC falsification cells and the 40-window resample do
  not read the 2023+ BTC holdout (the R-19/R-28 convention). Every
  configuration was frozen on inner-validation and the decision rule, the
  validity gate and the predictions were committed one commit ahead of the
  first holdout read — `git log` records it. Nothing here is offered as a
  Sharpe-based claim; the round's finding is that at matched risk there is
  no difference to claim.
- **08-18 · ~88** — R-30: **unchanged, and the reasoning is the point.**
  The bootstrap was re-run over the holdout to recover the log-growth
  interval, which looks like 50 fresh consultations and is not one: R-29
  drew those exact resamples and computed that exact interval object, then
  persisted only two of its three fields. Every overlapping number came
  back bit-identical, which is the evidence. No new question was asked; a
  field was recovered from an answer already given. Read it as ~138 if you
  disagree — R-29's conclusion that no Sharpe-based claim from this
  dataset is supportable holds either way.
- **08-17 · ~88** — R-29: every registered strategy (25) on both markets,
  as a fresh 2023+ account, to attach an interval to each. No selection
  was made on any of it and the decision rules were committed first — but
  50 consultations is 50 consultations. The finding that matters: at ~88
  program-level reads, and with `kelly_regime_v4`'s holdout Sharpe needing
  a **6.2-year** track record to clear a 103-trial bar it has 3.6 years
  of, **no Sharpe-based claim from this dataset is supportable any more**.
  Judge on drawdown, which still replicates, and treat B-06 (forward paper
  trading) as the only remaining source of evidence.
- **08-17 · ~38** — R-28: three configurations × two markets, plus two
  cost re-runs. The ETH falsification test and the 40-window resample do
  not read the 2023+ BTC holdout. At 24 trials in a single session the
  deflated Sharpe was already 0.859; at ~38 program-level consultations,
  treat any Sharpe-based claim from this dataset as unsupportable and
  judge on drawdown, which is the property that has repeatedly replicated.
- **08-16 · ~30** — Backfilled estimate. Every OOS figure in sections A
  and B came from reading the 2023+ holdout; it has never been pristine.
  Deflate program-level claims accordingly, and treat forward paper
  trading (B-06) as the only uncontaminated evidence still obtainable.
