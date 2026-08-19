# funding_harvest_carry — R-39 novel branch (B-03), 2026-08-19

Unregistered experiment. Code: `experiments/funding_harvest_carry.py`
(re-run with `python experiments/funding_harvest_carry.py`). Not
`@register`ed, not auto-discovered, **nothing committed by this session**.
Pre-registration: `docs/LEDGER.md`, "### R-39 pre-registration", the
bullet headed *Novel — `experiments/funding_harvest_carry.py`*.

**Verdict: NEGATIVE.** The trade does not clear the pre-registered bar on
2024-2026. The return bar fails outright; the drawdown/tail bar is
**voided, not passed**, because the risks that decide this trade are
structurally absent from the data (below). Do not register
`funding_harvest_carry` as a strategy.

**Configurations evaluated: 19 distinct trade specifications, 58
configuration-evaluations** (the same spec read over several windows).
Nothing was selected on: the primary spec (quarterly rebalance, roundtrip
fees, gearing 1.0, tiers 0 / 0.10% / 0.40%) was fixed by the
pre-registration before any 2024-2026 number was read; the
monthly/annual/drift rows are the reported neighbourhood.

---

## 0. Read this first: what this backtest structurally cannot see

This is a delta-neutral trade, and **this repository has no perpetual
price series** — the futures market runs on the spot series, labelled
`spot (perp proxy)` everywhere. Long spot and short perp are therefore
modelled by the *same* price series, so:

- **the basis is identically zero by construction.** The spot leg's P&L
  `qty*(P−P₀)` and the short perp's `−qty*(P−P₀)` cancel to the cent at
  every bar. Equity moves only through funding and fees. **Basis risk at
  entry and exit — one of the two or three risks that actually decide
  whether this trade makes money — cannot be measured here at all.** No
  basis was manufactured out of the price series: the standing rule
  ("never proxy unavailable data out of price", `docs/ROUTINE.md`)
  forbids it, and inventing one would have produced a number that looks
  like a measurement and is not.
- **every Sharpe and every drawdown below is an upper bound**, not an
  estimate. A carry Sharpe of 11 in the tables is what you get when the
  dominant risk term is missing from the model, not a discovery.

The full unmodelled-risk list is section 8. It is not boilerplate: for
this trade it *is* the finding, and it is why the tail/drawdown half of
the promotion bar is voided rather than scored.

---

## 1. Parity with `docs/VALIDATION.md` — the existing figure reproduces exactly

`docs/VALIDATION.md` ("The other side of the trade") reports five cells
from a one-off compounding calculation. All five reproduce from this
code, so the tables below are continuous with the committed document
rather than a fresh, unrelated computation.

| cell | VALIDATION.md | this code |
|---|---|---|
| gross funding stream, 2020-2023 | +82.0% over 4.0 yr = **+16.2%/yr** | **+82.0% / +16.15%/yr** |
| after 0.10% taker, quarterly | +14.6%/yr | +14.55%/yr |
| after 0.40% taker, quarterly | +9.8%/yr | +9.75%/yr |
| settlements where the payer flips | 13.5% | 13.5% |
| worst 30-day run | −1.31% | −1.31% |

Two conventions were recovered in the process, and both matter for
reading everything below:

1. **The VALIDATION fee figures are a linear subtraction**, not a
   compounded cost: 4 rebalances/yr × 2 legs × 2 sides × 0.10% = 1.6%/yr,
   and 16.2 − 1.6 = 14.6. At 0.40% it is 6.4%/yr, and 16.2 − 6.4 = 9.8.
   That convention charges a **full round-trip of the entire book every
   quarter**, which is deliberately pessimistic — no desk re-gears that
   way. A "trade only the drift" fee model is reported alongside.
2. **The gross figure assumes notional is reset to equity continuously.**
   That matters more than it sounds: see next.

### The leverage-drift trap in "quarterly rebalancing"

Holding `qty` BTC long spot and `qty` short perp is delta-neutral in BTC
at all times, but the *notional* floats with price while equity does not
(equity only earns funding). Over 2020-2023, a quarterly-rebalanced 1.0×
position drifts to a peak of **2.49× notional/equity**; annual rebalancing
reaches **3.11×**. So:

| 2020-2023, no fees | annualized | peak notional/equity |
|---|---|---|
| continuous rebalance (notional ≡ equity) | +16.15% | 1.00× |
| quarterly rebalance | +21.28% | 2.49× |
| annual rebalance | +24.10% | 3.11× |

The extra return of the slower rebalance is **bought with leverage, not
earned**. `docs/ROUTINE.md`'s standing rule ("match risk before comparing
anything") applies directly, so the **continuous-rebalance row is the
risk-matched read** and is what the headline numbers below use; the
quarterly rows are reported next to it with their `maxLev` so the
difference stays visible. This is the first thing the real backtest
surfaced that the one-off calculation could not.

---

## 2. The pre-registered cells, by sub-period (extended series)

`load_funding_extended`: real Binance 2020-01-01 → 2023-12-31 (4,383
settlements) + Deribit 2024-01-01 → 2026-08-12 (2,863 settlements),
source-tagged, never blended in the overlap, never rescaled. Truncated at
2026-08-12 because the price series ends there.

**Risk-matched read** (notional ≡ equity; fees applied with
VALIDATION.md's own linear quarterly convention so the two documents are
directly comparable):

| window | gross | net 0.10% | net 0.40% | neg settlements | worst 30-day | daily Sharpe (gross) |
|---|---|---|---|---|---|---|
| **2020-2023** (real Binance) | **+16.15%/yr** | **+14.55%/yr** | **+9.75%/yr** | 13.5% | **−1.32%** | 11.05 |
| **2024-2026** (Deribit ext.) | **+6.58%/yr** | **+4.98%/yr** | **+0.18%/yr** | 20.5% | **−0.32%** | 11.38 |
| full 2020-2026 | +12.27%/yr | +10.67%/yr | +5.87%/yr | 16.2% | −1.32% | 10.21 |

**Quarterly-rebalance read** (the literal pre-registered spec; carries
leverage drift, shown as maxLev):

| window | arm | ann | total | Sharpe | worst 30d | maxDD | maxLev |
|---|---|---|---|---|---|---|---|
| 2020-2023 | gross (qtrly) | +21.28% | +116.2% | 10.65 | −0.94% | 1.12% | 2.49× |
| 2020-2023 | 0.10% qtrly | +19.20% | +101.8% | 8.76 | −1.30% | 1.48% | 2.49× |
| 2020-2023 | 0.40% qtrly | +13.13% | +63.7% | 3.24 | −2.40% | 3.37% | 2.49× |
| 2024-2026 | gross (qtrly) | +7.95% | +22.1% | 10.34 | −0.29% | 0.32% | 1.64× |
| 2024-2026 | 0.10% qtrly | +6.10% | +16.7% | 5.56 | −0.66% | 0.71% | 1.64× |
| 2024-2026 | 0.40% qtrly | **+0.71%** | +1.9% | 0.31 | −2.20% | 5.21% | 1.65× |

Read either column and the direction is the same: **the 2024-2026 carry
is roughly a third of the 2020-2023 carry, and at a 0.40% retail taker
tier it is indistinguishable from zero.**

The two cells the pre-registration named as the falsification test:
2020-2023 net-of-0.10% was **+14.6%/yr**; 2024-2026 net-of-0.10% is
**+4.98%/yr** risk-matched (+6.10%/yr on the literal quarterly spec).
That is "materially worse", which is the pre-registered falsification
outcome, met.

---

## 3. The venue-consistent control — and the biggest caveat in this report

The 2020-2023 half is Binance, the 2024-2026 half is Deribit. They are
**different instruments**: Deribit charges funding continuously (hourly
`interest_1h`, summed here into 8h buckets) against Binance's discrete
8-hourly settlement; on the 2020-2023 overlap they correlate at r=0.69
but their level ratio is unstable year to year (0.21×–1.24×), which is
why `load_funding_extended` deliberately does not rescale them. So the
headline 16.15% → 6.58% mixes a market change with a venue change.

Running the identical analysis on **Deribit alone across 2020-2026**
separates the two:

| window | gross (risk-matched) | net 0.10% | net 0.40% | neg | worst 30d | Sharpe |
|---|---|---|---|---|---|---|
| Deribit 2020-2023 | **+7.88%/yr** | +6.28%/yr | +1.48%/yr | 28.0% | −3.32% | 5.71 |
| Deribit 2024-2026 | **+6.58%/yr** | +4.98%/yr | +0.18%/yr | 20.5% | −0.32% | 11.38 |

**Most of the apparent collapse is the venue, not the market.** Binance's
2020-2023 funding was roughly twice Deribit's over the same calendar
period (16.15% vs 7.88%/yr) and flipped negative half as often (13.5% vs
28.0%). Measured within one venue, gross carry went 7.88% → 6.58%/yr —
a decline, but a modest one.

95% stationary block bootstrap (30-day mean block, 2,000 draws, seed 39,
`tradebot.inference` convention), on daily returns of the risk-matched
gross stream:

| window | annualized % | daily Sharpe |
|---|---|---|
| extended 2020-2023 (Binance) | +16.16 [+9.80, +24.27] | 11.05 [9.16, 13.41] |
| extended 2024-2026 (Deribit) | +6.54 [+3.95, +9.42] | 11.38 [8.81, 14.19] |
| Deribit-only 2020-2023 | +7.88 [+2.91, +13.44] | 5.71 [2.16, 9.77] |
| Deribit-only 2024-2026 | +6.54 [+3.95, +9.42] | 11.38 [8.81, 14.19] |
| Deribit-only 2020-2024 | +8.47 [+4.12, +13.25] | 6.59 [3.27, 10.46] |
| Deribit-only 2025-2026 | +3.96 [+2.14, +5.99] | 10.64 [7.57, 14.26] |

**On a venue-consistent basis the decline is directional but not
statistically established**: [+2.91, +13.44] and [+3.95, +9.42] overlap
almost entirely, and even the sharper 2020-2024 vs 2025-2026 split
([+4.12, +13.25] vs [+2.14, +5.99]) overlaps. The cross-venue comparison
looks decisive; the within-venue comparison does not. Anyone quoting
"the carry premium collapsed after 2023" off this dataset should quote
the second pair, not the first.

---

## 4. The shape of the decline, per calendar year

Risk-matched gross (continuous rebalance) plus the two fee tiers on the
literal quarterly spec:

**Extended series** (Binance 2020-2023 → Deribit 2024-2026):

| year | source | gross | 0.10% qtrly | 0.40% qtrly | neg | worst 30d | n |
|---|---|---|---|---|---|---|---|
| 2020 | binance | +18.82% | +25.54% | +18.38% | 14.3% | −1.32% | 1,098 |
| 2021 | binance | +35.94% | +44.91% | +37.84% | 7.3% | −0.28% | 1,095 |
| 2022 | binance | +4.27% | +2.59% | −1.82% | 22.1% | −0.23% | 1,095 |
| 2023 | binance | +8.21% | +8.37% | +2.54% | 10.1% | −0.00% | 1,095 |
| 2024 | deribit | +10.88% | +12.09% | +6.25% | 15.8% | −0.32% | 1,098 |
| 2025 | deribit | +5.57% | +4.31% | −0.61% | 20.1% | −0.14% | 1,095 |
| 2026 YTD | deribit | **+1.53%** | **−0.22%** | **−5.72%** | 28.7% | −0.13% | 670 |

**Deribit-only** (venue-consistent; 2024-2026 rows are identical by
construction):

| year | gross | 0.10% qtrly | 0.40% qtrly | neg | worst 30d |
|---|---|---|---|---|---|
| 2020 | +9.69% | +15.24% | +8.55% | 26.8% | −3.32% |
| 2021 | +17.99% | +23.15% | +16.93% | 22.3% | −0.50% |
| 2022 | **−2.24%** | −3.26% | −7.46% | 47.9% | −2.02% |
| 2023 | +7.10% | +6.92% | +1.13% | 15.3% | −0.70% |
| 2024 | +10.88% | +12.09% | +6.25% | 15.8% | −0.32% |
| 2025 | +5.57% | +4.31% | −0.61% | 20.1% | −0.14% |
| 2026 YTD | +1.53% | −0.22% | −5.72% | 28.7% | −0.13% |

(Per-year rows where the 0.10% column exceeds the gross column are the
leverage-drift effect of section 1, not an arithmetic error: gross is the
risk-matched 1.0× stream, the fee columns are the drifting quarterly one.)

Three things worth noting:

- **The decline is not monotone and 2024 is not the break.** On the
  venue-consistent series, 2024 (+10.88%) was *better* than 2023
  (+7.10%) and far better than 2022 (**−2.24%**, with funding negative at
  48% of settlements). The literature's "crowded and died in 2024" story
  is not what this data shows; what it shows is a decline concentrated in
  **2025 and 2026** (+5.57%, +1.53%).
- **2022 already had a negative-carry year**, on Deribit, in the middle
  of the supposedly good era. The premium was never the reliable annuity
  the 2020-2023 Binance average makes it look.
- **2026 YTD is the first year where the trade loses money at a 0.10%
  taker** (−0.22%/yr) and loses badly at 0.40% (−5.72%/yr).

### A correction the operator should carry into the ledger

The R-39 pre-registration quotes the Deribit extension as *"annualized
mean funding 30.8% (2024) → 16.2% (2025) → 4.8% (2026 YTD)"*. Those
figures are **3× too high**: they annualize the mean of an 8-hour bucket
at 9 buckets/day instead of 3. The correct mean-based annualizations are
**10.3% / 5.4% / 1.6%** (compounded: 10.9% / 5.6% / 1.5%), which is what
this report uses throughout. The *shape* the pre-registration described
is right; the *level* in that sentence is not, and 30.8% for 2024 would
have made 2024 look like the second-richest year on record rather than an
ordinary one.

---

## 5. Sharpe against the literature's 6.45 → 4.06 → negative

| window | gross Sharpe | at 0.10% | at 0.40% |
|---|---|---|---|
| 2020-2023 (Binance) | 11.05 | 8.76 | 3.24 |
| 2024-2026 (Deribit) | 11.38 | 5.56 | 0.31 |
| 2024 only | 14.11 | 9.18 | 2.06 |
| 2025 only | 12.64 | 5.06 | **0.05** |
| 2026 YTD | 8.01 | **0.06** | **−1.61** |
| *literature (cited in VALIDATION.md)* | *~6.45 (2020-23) → 4.06 (2024) → negative (2025)* | | |

Read carefully, because the naive reading is wrong:

- **The gross Sharpes here (11.05, 11.38) are far above the literature's
  6.45 — because the basis risk is missing.** They are upper bounds on a
  quantity whose main variance term is not in the data. They should not
  be quoted as agreeing or disagreeing with a published figure that does
  include it.
- **The gross carry Sharpe did not deteriorate at all** from 2020-2023 to
  2024-2026 (11.05 → 11.38). What deteriorated is the *level* of the
  premium; the funding stream's volatility fell roughly in proportion.
  So "the Sharpe fell" is not what this data says on a gross basis.
- **The one number here that does line up with the literature comes from
  the venue-consistent series.** Deribit-only 2020-2023 gives a gross
  Sharpe of **5.71 [2.16, 9.77]**, against the literature's ~6.45 for the
  same era — a genuine near-match, and it suggests the Binance-based
  11.05 is the outlier rather than the norm. That is one more reason to
  treat `docs/VALIDATION.md`'s 2020-2023 carry cells as *Binance's*
  numbers rather than the market's.
- **The literature's qualitative path is reproduced only once real costs
  are charged.** At a 0.40% retail taker the Sharpe goes 3.24 → 0.31 →
  0.05 (2025) → −1.61 (2026 YTD); at 0.10% it goes 8.76 → 5.56 → 0.06
  (2026). The premium did not stop existing — it stopped being large
  enough to pay for the round-trips needed to harvest it.

---

## 6. Rebalance-frequency sensitivity (robustness, not a knob)

| period | rebalance | fee model | gross | 0.10% | 0.40% | n_reb | maxLev | maxMargin |
|---|---|---|---|---|---|---|---|---|
| 2020-2023 | monthly | roundtrip | +18.51% | +12.83% | −2.76% | 47 | 1.61× | 0.69× |
| 2020-2023 | monthly | drift | +18.51% | +17.93% | +16.21% | 47 | 1.61× | 0.69× |
| 2020-2023 | quarterly | roundtrip | +21.28% | +19.20% | +13.13% | 15 | 2.49× | 1.57× |
| 2020-2023 | quarterly | drift | +21.28% | +20.83% | +19.47% | 15 | 2.49× | 1.57× |
| 2020-2023 | annual | roundtrip | +24.10% | +23.45% | +21.49% | 3 | 3.11× | 2.34× |
| 2020-2023 | annual | drift | +24.10% | +23.74% | +22.65% | 3 | 3.11× | 2.34× |
| 2024-2026 | monthly | roundtrip | +7.20% | +2.03% | **−12.15%** | 31 | 1.42× | 0.43× |
| 2024-2026 | monthly | drift | +7.20% | +6.77% | +5.47% | 31 | 1.42× | 0.43× |
| 2024-2026 | quarterly | roundtrip | +7.95% | +6.10% | +0.71% | 10 | 1.65× | 0.69× |
| 2024-2026 | quarterly | drift | +7.95% | +7.60% | +6.56% | 10 | 1.65× | 0.69× |
| 2024-2026 | annual | roundtrip | +9.01% | +8.48% | +6.90% | 2 | 2.17× | 1.31× |
| 2024-2026 | annual | drift | +9.01% | +8.80% | +8.15% | 2 | 2.17× | 1.31× |

`maxLev` = peak one-leg notional / equity between rebalances. `maxMargin`
= peak unrealized loss on the short leg since its last (re)opening, as a
multiple of account equity.

- **The apparent "slower is better" gradient is not a plateau and not an
  edge.** It is monotone in exactly the direction that buys leverage:
  annual rebalancing wins because it lets notional drift to 3.11× and
  collects funding on a bigger book. Under the risk-matched continuous
  convention the frequency question dissolves entirely (all frequencies
  give the same +16.15% / +6.58%, differing only in fees). There is no
  frequency to optimize here, which is the right answer for a robustness
  check.
- **The fee model dominates the frequency.** Monthly + roundtrip at 0.40%
  is **−12.15%/yr** in 2024-2026 — the round-trips cost more than the
  premium. The same frequency under a drift-only fee model is +5.47%.
  Anyone implementing this must not re-open the whole book on a schedule.
- **`maxMargin` is the number that should worry a practitioner.** At
  annual rebalancing in 2020-2023 the short leg carries an unrealized
  loss of **2.34× account equity** before the next reset. Unless the
  venue cross-margins the spot leg against the short (portfolio margin),
  that account is liquidated long before the rebalance date, and the
  whole carry curve above never happens. This backtest does **not** model
  liquidation; it assumes perfect cross-margin.

---

## 7. Against the promotion bar: `buy_and_hold` and `kelly_regime_v4`, 2024-01-01 → 2026-08-12

| arm | final ($1,000) | total | maxDD | Sharpe |
|---|---|---|---|---|
| `buy_and_hold` (spot 1×) | $1,491 | **+49.1%** | 54.0% | 0.56 |
| `kelly_regime_v4` (5× futures, no funding) | $1,814 | +81.4% | 33.0% | 0.85 |
| `kelly_regime_v4` (5× futures, funding charged) | $1,558 | +55.8% | 34.9% | 0.67 |
| carry (quarterly, gross) | $1,221 | +22.1% | 0.32% | 10.34 |
| carry (quarterly, 0.10%) | $1,167 | **+16.7%** | 0.71% | 5.56 |
| carry (quarterly, 0.40%) | $1,019 | **+1.9%** | 5.21% | 0.31 |

**Return bar: fails, decisively.** Over the pre-registered 2024-2026
sub-period `buy_and_hold` returns +49.1% against the carry's +16.7% at
0.10% and +1.9% at 0.40%. The carry does not beat `buy_and_hold` out of
sample after real costs. It does not come close.

**Drawdown/tail bar: VOID, not passed.** The carry arm's 0.71% drawdown
and 5.56 Sharpe are not comparable to `buy_and_hold`'s 54% and 0.56,
because the arms cannot be risk-matched on this data: the carry arm's
realized volatility is near zero *because two of its three real risk
sources are absent from the model*, not because the trade is safe. There
is no way to lever the carry arm up to `buy_and_hold`'s volatility here
without inventing a basis and a margin model, which the standing rules
forbid. `docs/ROUTINE.md` is explicit about what to do in this case: "if
the arms cannot be matched, say so and void the cell rather than scoring
it." Scoring it would hand the carry a fake tail win on the strength of
the data this repository does not have.

### The comparison that actually decides it: carry vs cash

The carry's capital sits as collateral earning nothing in this model.
That was harmless at a 16%/yr premium and is decisive now. US T-bills
paid roughly 5.3% (2024), 4.3% (2025), ~3.7% (2026 YTD) — stated as
context only; this repo has no rates data and none was proxied.

| year | gross (risk-matched) | net 0.10% | net 0.40% | ~T-bill | excess @0.10% |
|---|---|---|---|---|---|
| 2024 | +10.88% | +9.28% | +4.48% | 5.3% | **+3.98%** |
| 2025 | +5.57% | +3.97% | −0.83% | 4.3% | **−0.33%** |
| 2026 YTD | +1.53% | −0.07% | −4.87% | 3.7% | **−3.92%** |

**By 2025 the delta-neutral carry no longer beats a T-bill**, before any
of the unmodelled risks in section 8 are charged against it — and it
carries exchange, custody, basis and liquidation risk that a T-bill does
not. In 2026 YTD it loses to cash by ~4 percentage points. This is the
single most decision-relevant table in the report.

---

## 8. Unmodelled risks — for this trade, these *are* the result

A carry backtest that reports a Sharpe of 11 without this section is
misleading, so it is stated at full strength:

1. **Basis risk at entry and exit — unmeasurable here, not small.** With
   spot and perp on the same price series the basis is identically zero.
   In reality the perp trades at a premium or discount to spot that moves
   with the same crowding that sets the funding rate; you enter paying it
   and exit receiving it (or the reverse), and in a stressed unwind the
   basis gaps exactly when you want out. A single 1% adverse basis move
   on entry+exit wipes out roughly two months of the 2024-2026 premium.
   **This is the dominant missing term, and every Sharpe above is an
   upper bound because of it.**
2. **Margin and liquidation risk on the short perp leg.** Measured as far
   as this data allows (`maxMargin` in section 6): the short leg's
   unrealized loss reaches **1.57× account equity** at quarterly
   rebalancing and **2.34×** at annual, in 2020-2023. The backtest
   assumes perfect cross-margin between the spot and perp legs and no
   liquidation. Without portfolio margin — or with a venue that
   auto-deleverages — the position is closed at the worst moment and the
   carry curve above never materializes. A real implementation must hold
   idle margin buffer, which lowers the return on total capital below
   every figure printed here.
3. **Exchange and custody risk — the failure mode that actually destroyed
   carry desks in 2022.** The trade requires simultaneous spot custody
   and a derivatives position, usually on venues you do not control.
   FTX-style counterparty failure is not a haircut on the return, it is a
   total loss of the position, and it is uncorrelated with everything
   this backtest measures. A 6.6%/yr premium does not pay for it.
4. **Borrow and collateral costs.** Not modelled. The spot leg has to be
   funded; the short leg has to be margined; neither is free at any
   realistic desk. Conversely, the collateral would earn interest, which
   is also not modelled — section 7's cash comparison is the honest way
   to look at both at once, and it is the comparison the trade fails from
   2025.
5. **Cross-venue splice.** The 2024-2026 half is Deribit, a different
   instrument with a different settlement mechanism, spliced onto a
   Binance history it correlates with only at r=0.69 and whose level
   ratio swings 0.21×–1.24× year to year. Section 3 is the control for
   this and it substantially softens the headline story. Any
   cross-period claim from the extended series inherits this caveat.
6. **One asset, one instrument, ~6.6 years.** BTC only. No ETH funding
   series exists here, so the pre-registered ETH-style falsification is
   not available for this branch.

---

## 9. Verdict against the pre-registered decision rule

The rule, quoted from `docs/LEDGER.md`:

> register `funding_harvest_carry` as a new strategy only if it clears
> the standing promotion bar (beats `buy_and_hold` OOS after real costs,
> or is a genuine drawdown/tail improvement outside the ±0.2 Sharpe
> floor) **on the 2024-2026 sub-period specifically** […] If the
> 2024-2026 carry is thin, negative, or fails to clear costs, report that
> plainly.

| clause | outcome |
|---|---|
| beats `buy_and_hold` OOS after real costs, 2024-2026 | **NO** — +16.7% (0.10%) / +1.9% (0.40%) vs `buy_and_hold`'s +49.1% |
| genuine drawdown/tail improvement outside ±0.2 Sharpe | **VOID** — arms cannot be risk-matched; the carry's low volatility is an artifact of two absent risk sources (§8.1, §8.2), so per `docs/ROUTINE.md` the cell is voided rather than scored |
| pre-registered falsification ("net-of-cost 2024-2026 materially worse than +14.6%/yr") | **FALSIFIED as pre-registered** — +4.98%/yr risk-matched, +6.10%/yr on the literal quarterly spec |
| parameter neighbourhood is a plateau, not a peak | **N/A / degenerate** — the only knob (rebalance frequency) is monotone in leverage and dissolves under risk-matching; nothing to optimize |
| clears costs at the 0.40% tier, 2024-2026 | **NO** — +0.71%/yr quarterly, +0.18%/yr risk-matched; negative in 2025 and 2026 |

**NEGATIVE. Do not register.** B-03 should be closed as *tested and
rejected on the current era*, not left open — with the qualification in
section 3 that the within-venue evidence for a *collapse* is weaker than
the cross-venue headline suggests, and the finding in section 7 that the
trade stopped beating cash in 2025.

### What is genuinely worth carrying forward

- **The literature's prediction landed on this project's own data**, in
  the specific form "the premium stopped covering its costs" rather than
  "the premium vanished". At a 0.40% taker the trade is dead from 2025;
  at 0.10% it is dead from 2026 YTD; gross it is merely thin.
- **The 2020-2023 +16.2%/yr figure in `docs/VALIDATION.md` is a Binance
  number, not a market number.** Deribit over the identical period paid
  +7.88%/yr with twice the negative-settlement frequency and a −3.32%
  worst month. If `docs/VALIDATION.md` keeps quoting the carry figures,
  it should carry that comparison — the risk profile that "nothing else
  in this repo comes close to" is roughly half as good on the other
  venue, and it was never measured against basis risk at all.
- **The pre-registration's Deribit annualization is 3× too high** (§4);
  worth correcting in the ledger before it propagates.
- **B-03's real blocker was never the funding data.** It is the missing
  perp price series. Extending funding through 2026 answered the "is the
  premium still there" question (thinly, and not net of costs), but the
  "is this trade safe" question — the one that made it attractive — is
  still unanswerable here and will stay that way until a perp price
  series exists alongside spot. That is a more useful backlog item than
  any further funding work.

---

## 10. Honest notes on this branch's own work

- The daily Sharpe of a funding stream is inflated by its smoothness and
  is **not comparable** to a price-strategy Sharpe from the comparison
  table. Do not put 5.56 next to `kelly_regime_v4`'s 0.67 and read them
  as the same kind of number.
- The 2026 row is YTD through **2026-08-12** (670 settlements, ~223
  days), truncated by the price series, not the funding series (which
  runs to 08-19). Annualizing 8 months of a declining series overstates
  confidence in the 2026 point.
- The T-bill rates in section 7 are **stated from general knowledge, not
  fetched** — this repo has no rates data. They are indicative context,
  and the 2025 "excess ≈ 0" conclusion would move by a point or two if
  the actual realized bill yields differ.
- The `roundtrip` fee convention is inherited from `docs/VALIDATION.md`
  for comparability and is pessimistic; the `drift` convention is more
  realistic and is reported everywhere alongside. Neither is a free
  choice — both are shown, and the verdict does not change under either.
- Bootstrap intervals cover sampling noise **in the funding rate only**.
  They cannot and do not widen to cover any risk in section 8.
