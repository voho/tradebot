# R-178 pre-registration — a synthetic DVOL-priced options overlay on `kelly_regime_v4`

Status: FROZEN before either branch was dispatched or any real-data
performance number was read. This document and `experiments/r178_shared.py`
(and its self-tests above, run and printed to the terminal before either
branch's code was written) are committed to `main` before either branch
starts, per the in-flight-discipline convention (R-131/R-133's lesson).

## Step 0 recap

`git fetch origin main` + `git rev-parse HEAD origin/main` confirmed HEAD
== `origin/main` (`57543a3`) before this round started; no `r<nn>_shared.py`
existed without a section B entry (newest was `r177_shared.py`, `grep -c
"R-177" docs/LEDGER.md` = 7). Step 0b's saturation count, run per
ROUTINE.md's own command, returned **2** consecutive null passes since
R-176's own dispatch — "0–2: normal". The backlog grep returns only
**B-48** (a documentation/formatting instrument fix) plus four
already-inactionable rows (B-06 de-ranked, B-09 LOW, B-17 PARTIAL, B-28
blocked) — so this is a fresh off-backlog literature-prompted round, the
same convention R-160 through R-177 used, and this session's scheduled
brief was the generic "take the best strategy, propose an improvement
direction, research it, dispatch conservative/novel sub-agents, measure,
promote the winner" (same brief R-170/R-175's entries name).

A dedicated research sub-agent (background, web search, no code, no file
writes) surveyed the standing diagnosis, section C's ruled-out table and
recent rounds, and proposed this direction. Confirmed independently by the
operator with a direct grep of the full ~25,900-line ledger for
`black-scholes|black scholes|protective put|covered call|\bcollar\b
|synthetic option|strike price|option overlay|delta.hedg`: **zero hits**,
anywhere. A broader grep for `\boption\b` (10 hits) turns up only English
usage ("two options", "the option to...") and R-73/R-136's own DVOL
work — never an actual derivatives position. DVOL itself has been used
twice before (R-73: DVOL level/momentum as a leading INFO-axis vote input,
NEGATIVE, failed the lead-time gate; R-136: DVOL blended into the SCALE
axis's own volatility *estimate*, NEGATIVE, reproduced R-08's
forecast-quality inversion) and once more as one of four meta-labeling
features (R-170: VRP = DVOL-minus-realized-vol, NEGATIVE, failed the
Step-A AUC gate). In all three, DVOL is consumed as a *predictor* — a
number fed into a vote, a scale, or a classifier. This round never asks
DVOL to predict anything: it prices a genuinely new instrument (a
synthetic options structure) that pays off on realized moves regardless of
whether the vote called them, which no round has tested.

## Step 1 — the four-question filter

**1. Which constraint does it attack?** **N≈3**, primarily. `kelly_regime_v4`'s
entire edge lives in ~3 independent regime events across 9 years (the
standing diagnosis's own framing) — the SIZE layer only ever rescales a
*linear* bet on having read one of those ~3 events correctly, and every
regime-timing mechanism tried against it (12 structurally distinct
constructions through R-155/R-156/R-157, all NEGATIVE) has failed to
out-detect it with statistical confidence, because there is not enough
independent regime evidence to detect *better*. A convex options structure
does not need to call the regime right — it pays off on realized magnitude
directly, on a per-*expiry* clock (N≈150–280 non-overlapping weekly cycles
over DVOL's 2021-03-24 → 2026-08-21 coverage) rather than a per-*regime*
clock (N≈3). Secondarily **COST**: the overlay's own carry (the VRP) is a
persistent, priced cost/benefit that scales with time held, structurally
akin to funding (R-35/R-56) but on a different instrument.

**2. Which ledger entries is it not a duplicate of?** Not R-73 (DVOL as a
directional vote INPUT; here DVOL only prices an instrument, contributes
no directional signal — the vote is v4's own, untouched). Not R-136 (DVOL
blended into the SCALE axis's volatility *estimate*; no volatility
estimate anywhere in v4 is touched here — v4's `scale`/`frac`/`target` are
byte-identical to the registered strategy in both branches). Not R-170
(VRP as one of four meta-labeling *features* gating v4's own `frac*scale`
multiplicatively; here nothing about v4's existing position is gated,
shrunk or multiplied — an entirely separate, additively-summed position is
opened alongside it). Not the promoted `kelly_regime_ev`/`kelly_regime_ev_fast`
(a fee-vs-growth no-trade band on v4's *own* rebalances — no new
instrument). Not `overshoot_fade` (trades the underlying directly around
liquidation cascades, no options). Not any COST-axis round (R-56 maker/limit
execution, R-173 spread estimation) — those model execution cost on the
*existing* linear position; this round adds a nonlinear position with its
own, separately priced cost.

**3. Is it simulable here?** Yes, with one disclosed, hard data
constraint: `data/btc_dvol_daily.csv.gz` / `data/eth_dvol_daily.csv.gz`
(Deribit's official 30-day IV index, fetched by
`scripts/fetch_deribit_dvol_novel.py`, R-73) cover **2021-03-24 →
2026-08-21 only** — a hard limitation, not a bug (options markets did not
exist at scale before Deribit's own DVOL launch), identical to the
constraint R-73/R-136/R-170 already disclosed and worked within. No real
options order book, bid/ask or margin model is simulated (this project's
simulability contract is 5m OHLCV bars, no order book, no queue model);
option prices are computed causally from bar closes and causally-aligned
DVOL via Black-Scholes, `r=0` (no risk-free-rate series exists in this
project's data). A synthetic `cost_bps` haircut on premium at each roll's
opening approximates the bid/ask this project has no data to simulate
directly — the same kind of approximation R-56's maker-model and R-173's
spread-estimator rounds already used for a different cost. This project's
environment has **no scipy** (R-118/R-125, re-confirmed here); the normal
CDF inside Black-Scholes uses the Abramowitz & Stegun (1964, 7.1.26)
rational approximation (max abs. error 1.5e-7), verified against known
points before use (`_norm_cdf([-3,-1,0,1,3])` = `[0.00135, 0.15866, 0.5,
0.84134, 0.99865]`, matching the standard table to 5 decimals) and against
put-call parity (`C - P = S - K` at `r=0`, verified exactly, diff = 0.0).

**Methodology adjustment, made here before any branch or number exists,
following R-170's own precedent exactly:** DVOL's coverage starts
2021-03-24, **inside** ROUTINE.md's own inner-validation window
(2021-01-01 → 2022-12-31) and entirely after inner-train ends
(2020-12-31). An inner-train-only sweep would therefore see **0% DVOL
coverage** and never test the mechanism at all — R-170 hit the identical
problem and resolved it the identical way: **both branches iterate and
select on bars from DVOL's first covered day (2021-03-24) through
2022-12-31** (inner-validation's own end), not inner-train alone.
ROUTINE.md Step 3 itself names inner-validation a training resource, so
this spends nothing the holdout has not already been protected from. The
pre-registered holdout stays exactly `OOS_START = 2023-01-01` onward, which
DVOL covers in full.

**4. What would make it fail?** Named now, before any code ran: (a) the
weekly-rolled premium (paid, conservative branch; received, novel
branch's harvest leg) is swamped by BTC's own realized-move fat tails —
the same short-gamma "wins most weeks, blown up by the rest" signature
`overshoot_fade` and `attrition_reversion` already show in this ledger for
a differently-constructed short-vol trade; (b) the synthetic `cost_bps`
haircut, calibrated blind (no real order-book data to fit it to), turns
out too optimistic and a realistic haircut alone erases any measured edge;
(c) DVOL's own richness relative to realized vol (Alexander & Imeraj 2021's
BVRP ≈0.14) is itself an artifact of Deribit's specific order flow in a
thin, young market and does not persist out of sample; (d) the two-books
sleeve simplification (overlay funded from `overlay_frac * combined
equity`, not from v4's actual margin) understates how much this would
actually cost a fund running one account, and a real single-margin
implementation would be worse than what is measured here.

## Citations

- Israelov, R. & Klein, M. (2016). "Risk and Return of Equity Index
  Collar Strategies." *Journal of Alternative Investments* 19(1):41–54.
  Rules-based equity-index collars systematically underperform because
  they give up upside *and* pay the volatility risk premium — a
  structural drag, not noise. Motivates the conservative branch and its
  named failure mode.
- Bakshi, G. & Kapadia, N. (2003). "Delta-Hedged Gains and the Negative
  Market Volatility Risk Premium." *Review of Financial Studies* 16(2).
  Option sellers are compensated richest exactly in high-realized-vol
  regimes — the economic basis for the novel branch's harvest leg.
- Alexander, C. & Imeraj, A. (2021). "The Bitcoin VIX and its Variance
  Risk Premium." *Journal of Alternative Investments* 23(4):84–109.
  Constructs a CBOE-style BTC implied-vol index from Deribit and measures
  BVRP ≈ 0.14 (≈7x the S&P 500's ≈0.02) — an unusually large, persistent
  premium, and the number this round's honesty check (failure mode (c)
  above) is checking is not an artifact of a single paper.
- Gârleanu, N., Pedersen, L.H. & Poteshman, A.M. (2009). "Demand-Based
  Option Pricing." *Review of Financial Studies* 22(10):4259–4299. Option
  prices carry a demand-pressure premium that depends on who needs
  insurance when — the mechanism motivating the novel branch's
  vote-conditioned stance switch (buy protection when v4's own vote says
  uncertain/bearish, i.e. when *this account* needs insurance; harvest
  premium when the vote says confidently bullish).
- Brock, W.A. & Hommes, C.H. (1998). "Heterogeneous Beliefs and Routes to
  Chaos in a Simple Asset Pricing Model." *Journal of Economic Dynamics
  and Control* 22(8-9). The discrete regime-switching-by-fitness framing
  behind the novel branch's stance rule (this project's own
  `replicator_book`/`harsanyi_crowd` draw on adjacent literature).

## Both branches, frozen before any number is read

Shared engine: `experiments/r178_shared.py`'s `vote_frac` (v4's own
unmodified vote, read-only) and `simulate_overlay` (the causal, weekly
Black-Scholes overlay simulator, self-tested above). Both branches call
`simulate_overlay` on `kelly_regime_v4`'s own unmodified, already-registered
equity curve (from `run_backtest` with no changes) as `base_equity` — v4's
own position, sizing and trades are byte-identical to the registered
strategy in both branches; only an additive overlay differs.

### Conservative branch — literal rolling collar (Israelov & Klein)

**Mechanism.** Every 7 days, buy one OTM put (10% out of the money) and
sell one OTM call (10% out of the money) against `overlay_frac` of the
account's combined equity, unconditionally — a standard protective collar,
`stance ≡ +1` on the put leg and `stance ≡ -1` on the call leg (implemented
as two `simulate_overlay` calls, put-only and call-only legs, summed).
Converts an undetected regime break into a bounded, pre-paid cost instead
of a raw drawdown, at the price of Israelov & Klein's documented VRP drag.

**Sweep (inner-validation window, 2021-03-24 → 2022-12-31, futures_5x and
spot, BTC):** `overlay_frac ∈ {0.25, 0.50, 1.00}` × `moneyness ∈ {(0.90,
1.10), (0.95, 1.05)}` × `cost_bps ∈ {10, 30}` = 12 configs, plus 1 frozen
primary (`overlay_frac=0.5, moneyness=(0.90,1.10), cost_bps=20`) = **13
configs** on the primary market; report both markets for the frozen
primary only (2 more) = **15 total**.

**Falsification test, frozen:** kill it if it **fails the Monte Carlo
stress windows** — pre-registered outcome: run `scripts/stress_test.py`'s
window battery (or an equivalent resample over the DVOL-covered span, if
the full battery's windows predate DVOL coverage) on the frozen primary;
if median simulated max-drawdown is not measurably reduced vs.
`kelly_regime_v4` alone (statistically indistinguishable, i.e. inside the
±0.2 Sharpe-equivalent noise floor R-20 established, or worse), the paid
VRP has bought nothing and the branch is dead.

### Novel branch — fitness-switched VRP harvest/hedge

**Mechanism.** Every 7 days, read v4's own vote `frac_frac` at the roll's
opening bar (from `r178_shared.vote_frac`, unmodified). If `frac ≤ 1/3`
(bearish or genuinely uncertain — the same two states v4 itself treats as
"stand mostly or fully aside"): open a long strangle (`stance=+1` on both
legs) — pay for convexity exactly when the account's own risk read says it
is least confident. If `frac ≥ 2/3` (confidently bullish): open a short
strangle (`stance=-1` on both legs) — harvest Bakshi-Kapadia/Alexander-Imeraj's
BVRP exactly when v4's own vote is not hedging anything already. Same
moneyness/roll/cost mechanics as the conservative branch, via the identical
shared `simulate_overlay` primitive, called once with `stance = where(frac
<= 1/3, +1, -1)` array (no explicit branch needed for the two full-agreement
states beyond the ≤1/3 / ≥2/3 split, since v4's own vote only ever takes the
four values {0, 1/3, 2/3, 1}).

**Sweep (identical window/markets):** `overlay_frac ∈ {0.25, 0.50, 1.00}` ×
`moneyness ∈ {(0.90,1.10), (0.95,1.05)}` × `cost_bps ∈ {10, 30}` = 12
configs, plus 1 frozen primary (same defaults as conservative) = **13**,
plus both-market frozen-primary report = **15 total**.

**Falsification test, frozen:** kill it if it **fails a named
statistical-significance threshold** — pre-registered outcome: paired
stationary-block-bootstrap 95% CI on Δlog-growth (`r178_shared.log_growth`,
matching R-177's own convention) vs. `kelly_regime_v4` alone must exclude
zero on **both** BTC and ETH simultaneously (this project's standing
cross-instrument replication convention, R-57). If either market's CI
contains zero, or the two markets disagree in sign, the branch is dead
regardless of the pooled headline number.

## Promotion bar (both branches, ROUTINE.md's default)

Promote only if **all** hold: beats `kelly_regime_v4` (not merely
`buy_and_hold`, since v4 is the base position both branches keep
unmodified and the object of comparison is what the overlay *adds*) on
the DVOL-covered holdout after the `cost_bps` haircut; the improvement
exceeds the ±0.2 Sharpe noise floor (R-20) or is a genuine, risk-matched
(R-33) drawdown/tail improvement; survives its own falsification test
above; the parameter neighbourhood is a plateau, not a peak (report
neighbours, not just the swept optimum); and the account is never
liquidated (`simulate_overlay`'s own `liquidated` flag) across any tested
configuration or Monte Carlo stress window. Anything else is NEGATIVE.

## Disclosed simplifications and risks, named up front

- **Two-book sleeve, not single-margin.** Overlay notional is sized off
  `overlay_frac * combined equity`, not carved out of v4's own margin — a
  real fund running one account would need to either reduce v4's own
  leverage to free margin or post separate collateral; neither is modeled.
  This makes both branches' numbers optimistic relative to a real
  single-account implementation.
- **No real order book.** `cost_bps` is a flat, calibrated-blind haircut
  on premium at the opening roll; a real Deribit book's actual bid/ask on
  weekly BTC/ETH strangles at these deltas is not in this project's data.
- **r=0.** No risk-free-rate series exists in this project's data.
- **European, cash-settled, no early exercise, no pin risk.** Standard
  simplifications for a synthetic, causally-priced instrument.
- **Liquidation handling.** `simulate_overlay` floors combined equity at
  zero and stops the overlay's own P&L from moving further once
  breached (mirrors this project's broker's own liquidation semantics) —
  confirmed reachable in this round's own Step-0 sanity check (an
  unconstrained SELLER-stance, `overlay_frac=0.5` test, DVOL-only window,
  reached **-$7,060** before flooring). Both branches must report
  `liquidated=True/False` for every configuration, not just the point
  estimate.
