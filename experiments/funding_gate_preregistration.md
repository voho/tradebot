# Pre-registration — funding as a sizing input on `kelly_regime_v4` (B-05)

Written and committed before any holdout read. Backlog item B-05: "Funding
as a gate on the existing strategy (stand flat in the top decile)."

## Idea, in one sentence

`kelly_regime_v4` infers crowding indirectly, from price sitting above or
below slow moving-average anchors. Funding is a *direct, priced*
measurement of the same crowding — Schmeling, Schrimpf & Todorov (2023,
BIS WP 1087) trace crypto's large, volatile funding basis to leveraged
trend-chasing demand meeting limited arbitrage capital, i.e. funding
**is** the rent the crowd pays for being crowded, exactly the quantity
`kelly_regime`'s anchor vote is a noisy proxy for. Using it should let the
strategy de-lever precisely when the crowd (not the price) says it is
stretched, rather than only when price has already moved.

## Constraint attacked

**COST** primarily (R-14: funding runs +20%/yr while the strategy holds
vs +2.8% flat — an adverse-timed cost this doesn't currently react to)
and, more speculatively, **INFO**: funding is a genuinely second data
series, not a transform of the OHLCV the anchor vote already uses.

## Not a duplicate of

- **L-05/L-06** (`kelly_regime_ev`/`_fast`) derive a no-trade band from
  the one-time *transaction* fee. This derives an adjustment from a
  *continuously accruing* cost (funding) — a different term in the same
  growth equation, not a re-tuning of the same one.
- **R-14** measured what funding costs the existing strategies (a
  diagnostic). **R-15** harvested funding as a *standalone* delta-neutral
  carry trade. **R-16** found funding predicts *forward returns*
  (quintile spread) as a directional signal. This idea is none of
  those: it uses funding only to size down the crowd-regime strategy's
  own existing long, never to trade funding for its own sake or to flip
  direction.
- **R-28/R-31/R-32** (e-process gate) replace the *regime* signal.
  Nothing here touches the regime vote; both variants below only touch
  the *sizing* step, after the vote has already run — a narrower, cheaper
  change than a new gate mechanism, deliberately, given R-31/R-32's
  finding that gate mechanism barely matters at matched risk.

## Simulable here?

Yes, with a real constraint: `data/btcusdt_perp_funding_8h.csv.gz` is
real Binance BTCUSDT funding but covers **2020-01-01 to 2023-12-31 only**
(4,383 settlements). Both variants degrade to exactly `kelly_regime_v4`
outside that window (funding term = 0 by construction, not imputed) —
per the standing rule, missing data is not proxied. All fitting,
selection and evaluation below therefore happens inside the funding
window, split like the project's usual protocol but compressed to fit
it:

| slice | dates | use |
|---|---|---|
| funding-inner-train | 2020-01-01 → 2021-12-31 | fit, sweep freely |
| funding-inner-validation | 2022-01-01 → 2022-12-31 | select the frozen config |
| funding-holdout | 2023-01-01 → 2023-12-31 | *read once, after freezing* |

The futures market with **funding charged** (`funding=load_funding(...)`
passed to `run_backtest`) is the only market this idea touches — spot
pays no funding. The incumbent baseline for every comparison is
**`kelly_regime_v4` on futures with funding charged**, not the
funding-free number in the README table.

## Two variants, designed before any code ran

### Variant A — conservative: funding-decile flat gate

**Mechanism.** Rank each bar's trailing-90-day annualized funding rate
against its own trailing history; when the percentile enters the top
decile (crowded longs), force the position flat regardless of what the
anchor vote says; release it only once the percentile drops back below a
lower exit threshold (latching hysteresis, the same pattern the anchor
vote and v3's volatility breakout already use, so it does not add a new
kind of moving part to the strategy). Binary, low-turnover, minimal
change to v4.

### Variant B — novel: continuous funding-adjusted Kelly exposure

**Mechanism.** `kelly_regime`'s sizing step is `target_vol / vol`, a
stand-in for the growth-optimal `mu / vol^2` calibrated through
`target_vol` instead of an estimated `mu`. A continuously accruing cost
`phi` (annualized funding rate, EWM-smoothed) subtracts linearly from the
numerator of that same ratio, the same way Constantinides (1986) shows a
transaction cost does to the *discrete* no-trade band `kelly_regime_ev`
already derives — here applied to a *continuous* cost instead of a
one-time one, in the spirit of the convenience-yield framing of perp
no-arbitrage (Angeris, Chitra & Evans 2022; He, Manela, Ross & von
Wachter 2024): `exposure_adjusted = max(0, exposure_v4 - k * phi / vol^2)`.
Continuous, not gated; uses funding's *magnitude*, not just its rank;
floored at 0 since the vote never goes short so there is nothing to
protect on the other side. One free parameter (`k`, derivation gives
`k=1` as the no-free-parameter case; swept for a plateau) plus the EWM
smoothing span on `phi`.

## Decision rule — fixed now, before the holdout is read

Promote a variant only if, on **funding-holdout (2023, futures, funding
charged, vs `kelly_regime_v4` funding-charged)**, **all** hold:
1. beats the funding-charged incumbent on log growth, or on max drawdown
   by more than the ±0.2 Sharpe-equivalent noise floor (R-20's bar,
   applied here as in every other row);
2. the funding-inner-validation selection was a **plateau** across the
   swept neighbourhood, not an isolated peak;
3. the paired bootstrap interval (`tradebot.inference.paired_bootstrap`,
   same 30-day block / 2,000-resample protocol as R-29/R-30/R-31) on the
   holdout excludes zero, OR the point estimate is directionally
   consistent with funding-inner-validation (same sign, comparable
   magnitude) — a single favourable point on 250 holdout observations is
   not enough alone, exactly R-29's lesson.

Anything else is NEGATIVE, written up with the same care as a win, per
ROUTINE.md.

## Pre-registered failure modes (named now)

(a) the funding signal is mostly a lagging restatement of the price-based
anchor vote (the two are correlated because both react to the same
trend), so the adjustment fires exactly when the vote has already gated
exposure down, and duplicates rather than adds;
(b) 2020-2023 contains exactly one funding regime (the 2021 mania and its
unwind) — R-15 already flagged that carry Sharpe compressed sharply after
2024 in the wider literature — so a fitted threshold or `k` may be
overfit to that one episode and not generalize even within-sample to
2022's bear, where funding was often negative;
(c) the flat-gate (Variant A) adds turnover at exactly the moments
volatility is highest, and the extra fee/slippage from flipping in and
out eats the funding saved;
(d) with only one holdout year (2023) available, any result is a single
observation at the regime level (N≈3's problem, sharper here) and should
be read as such regardless of which way it comes out.

## Parallel-branch bookkeeping

Two disjoint experiment files, two sub-agents, neither commits. Trials
count is the sum of both branches' swept configurations, reported in the
ledger row. An independent skeptic (this orchestrating session, reading
code and re-deriving numbers) reviews both before either result is
believed, per ROUTINE.md's parallelism section.
