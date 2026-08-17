# Untried directions, from the diagnosis outward

## The diagnosis first, because it rules out most ideas

Twenty-five strategies, two research rounds and a deep-learning pass have
produced one robust finding — regime-gated sizing reduces drawdown — and
no return alpha on either asset tested. The binding constraint is **not
model capacity**. It is:

1. **Information.** One price series. Every strategy here consumes the
   same OHLCV bars, so they are twenty-five ways of re-reading one
   channel.
2. **Effective sample size ≈ 3.** The regime filter acts on multi-year
   cycles; a million bars is autocorrelated detail inside three trials.
   Classical inference has nothing to work with.
3. **No error control anywhere in the signal path.** The regime gate is
   "price 1% above a moving average, latched" — a heuristic with no false
   alarm rate, no confidence, no notion of evidence.
4. **Costs scale with the signal.** Funding runs +20%/yr while the
   strategy holds against +2.8% while flat, because the crowding it
   detects is what sets the rate.

An idea is only worth trying if it attacks one of these. "Another
indicator" attacks none of them, which is why the bottom of the
comparison table looks the way it does.

---

## 1. The regime detector and the position sizer are the same object

> **BUILT, 2026-08-17 — see R-28 in [LEDGER.md](LEDGER.md).** The verdict
> is NEGATIVE on the promotion bar and the section below was right about
> why: it produced a risk improvement, not return alpha. It produced the
> largest one in the project — out-of-sample drawdown 11.6% against the
> incumbent's 27.8% and holding's 54.0%, deeper than the incumbent in
> **0 of 40** Monte Carlo windows, and the property replicates on ETH.
> The number worth remembering: honest anytime-valid evidence justifies
> **0.27x** the exposure the heuristic gate takes, because the e-process
> accumulates evidence at +0.79 nats a year against a noise standard
> deviation of 3.33 — reaching the α=0.05 threshold on drift alone takes
> **3.8 years**, and the dataset holds two to three such periods.
> "Effective sample size ≈ 3" now has a units-carrying measurement.
> Read the rest of this section as the case that was made in advance.

**This is the one I would build.** It is original — a search of the
trading literature turns up no established use — and it is not a
metaphor: the mathematics is literally shared.

**The idea.** Modern game-theoretic statistics (Shafer 2021, "Testing by
betting"; Ramdas, Grünwald, Vovk & Shafer) replaces the p-value with the
**e-value**: evidence against a null is measured by *how much money a
bettor would have made* betting against it at fair odds. The test
statistic is a nonnegative martingale — a **wealth process** — and the
optimal bet size is the **Kelly fraction**.

Now look at what this repo already does. `kelly_regime` asks "is the
market in a bullish regime?" (a hypothesis test, done heuristically) and
then "how much should I hold?" (a Kelly sizing problem, done separately).
Testing by betting says these are one question. The e-process wealth
accumulated against the null "drift is zero" **is** the evidence, and the
Kelly bet that grows it **is** the position.

**What it buys, concretely:**

- **Anytime-valid error control with n≈3.** E-processes give
  non-asymptotic Type-I guarantees valid at *arbitrary stopping times*,
  with no fixed sample size and no asymptotics. That is precisely the
  regime this project is stuck in, and it is what e-values were invented
  for. No other tool on this list survives a sample size of three.
- **A principled confidence, not a latch.** Exposure becomes a function
  of accumulated evidence rather than a 0/⅓/⅔/1 vote with a hand-set 1%
  band. The `vote_gamma` question that `kelly_regime_v2` was invented to
  answer stops being a free parameter.
- **Optional stopping is legal.** Every backtest in this repo peeks at
  results and decides whether to continue — the exact practice that
  invalidates classical p-values and that anytime-valid inference
  legitimises.
- **It costs no new data** and drops straight into the existing
  framework, so the strict-causality suite, walk-forward split and Monte
  Carlo windows judge it on day one.

**Honest risk.** The e-process still needs a bet on *what the alternative
looks like*, and a badly chosen betting function grows slowly. It gives
error control, not clairvoyance — it will make the strategy better
*calibrated*, and calibration is a risk property. Given this project's
record, expect another drawdown improvement rather than return alpha.

**Reading:** Shafer (2021, JRSS-A) "Testing by betting"; Ramdas et al.
(2023) "Game-theoretic statistics and safe anytime-valid inference";
Waudby-Smith & Ramdas (2024) on betting-based confidence sequences.

---

## 2. Calibrated uncertainty instead of point forecasts

**The idea.** Kelly sizing needs a *probability*, and a miscalibrated one
is catastrophic — that is the entire reason fractional Kelly exists and
why this repo caps leverage at 2x out of caution. **Conformal
prediction** produces distribution-free prediction intervals with
finite-sample coverage guarantees. Feed calibrated intervals into the
sizer instead of a heuristic vote and the fractional-Kelly haircut stops
being a guess.

The obstacle is that conformal prediction assumes exchangeability, which
financial time series violate. That is exactly the active research front:
adaptive conformal inference under distribution shift, conformal
prediction with change points (NeurIPS 2025), and — pleasingly —
**adaptive conformal inference *by betting* (2024)**, which is the same
machinery as direction 1.

**Why it fits here.** This project's failures are not "the forecast was
wrong" — they are "the forecast was trusted too much on three
observations." Conformal attacks trust, not accuracy.

**Honest risk.** Coverage guarantees degrade exactly when regimes shift,
which is when they matter. Weaker than direction 1, and mostly
subsumed by it.

---

## 3. New information: on-chain, with a specific warning

Bitcoin is the one asset whose ledger is public. Exchange netflows,
dormancy, realized cap and whale transfers are **genuinely orthogonal to
price** — the only channel on this list that adds information rather than
re-reading the same series.

**But the literature is brutal, and matches this repo's own base rate.**
A comprehensive study started from **141 candidate predictors**: 67 worked
in-sample, 29 survived out-of-sample at some horizon, and **only 4 beat a
random walk at all horizons.** That is a 141→4 attrition, the same shape
as this repo's 28-in-sample→0-out-of-sample fee study. Anyone adding
on-chain features should expect to be one of the 137.

**The one specific finding worth acting on, and its trap.** Research on
intraday on-chain flows finds BTC net inflows *do not* predict returns
but **are reliably associated with volatility**. This repo's strategy is
a volatility-targeting sizer, so volatility is the input it actually
consumes — that looks like a perfect match.

It is a trap. This project already established that **better volatility
forecasting makes this strategy worse**: a forecast 8% better on QLIKE
returned $52K instead of $115K, because it de-levers more promptly into
BTC's high-volatility, high-forward-Sharpe states. Improving an input
that feeds a wrong-signed mapping makes things worse, not better. So the
useful experiment is not "add on-chain vol forecasting" — it is **fix the
sign first**, then ask whether a better input helps.

---

## 4. Earn something other than direction

Every strategy here tries to be paid for *being right about price*. The
alternatives get paid for providing something:

- **Funding harvest** — already measured at +16.2%/yr with a −1.31% worst
  month, and already flagged as possibly dead post-2024. See
  [ALTERNATIVES.md](ALTERNATIVES.md).
- **Liquidity provision / market making.** Avellaneda & Stoikov (2008) is
  already *cited* in this repo's research notes but never implemented —
  the reservation-price-and-inventory-skew framework is the natural
  home of the no-trade band that `kelly_regime_ev` derived from scratch.
  The blocker is honest: this framework models bar-close fills with no
  order book, so a market maker cannot be simulated credibly here. It
  would need a queue model before it meant anything.
- **AMM liquidity and loss-versus-rebalancing.** The modern formalisation
  (Milionis, Moallemi, Roughgarden, Zhang) makes LP returns decomposable
  and hedgeable — a genuinely quantitative frontier. Same blocker: not
  simulable with OHLCV bars.

---

## 5. Fix the inference, not the model

> **BUILT, 2026-08-17 — see R-29 in [LEDGER.md](LEDGER.md).** All three
> items below are now computed, in `tradebot.inference` and
> `scripts/inference.py`. This was the cheapest direction on the page and
> it produced the most uncomfortable result: **10 of 96 adjacent pairs in
> the comparison table are distinguishable at 95%, and none of them is in
> the top eight.** By the table's own criterion — final balance —
> `kelly_regime_v4` against buy-and-hold on spot is a coin flip (P=0.52).
> No strategy's Sharpe clears a deflated bar out-of-sample, and neither
> does buy-and-hold's. Cross-validating the table's own selection rule
> over 45 purged folds, it beats holding in 6. The section below was
> written as "unglamorous"; it turned out to be the round that reclassified
> most of this repo's numbers as buckets rather than a ranking.

Cheap, unglamorous, and it would raise the quality of every result
already in the repo:

- **Combinatorially purged cross-validation** (López de Prado) instead of
  one walk-forward split, with purging and embargo around each fold to
  kill leakage across overlapping windows. It yields a *distribution* of
  out-of-sample paths rather than the single number that this repo has
  repeatedly shown is unreliable.
- **Deflated Sharpe** applied systematically. It is cited in
  [RESEARCH.md](RESEARCH.md) but never actually computed, even though
  every sweep in this project is a trial that inflates whatever it
  selected — the fee study ran 32.
- **Block-bootstrap confidence intervals on every headline**, so the
  comparison table reports ranges rather than points. Given the measured
  ±0.2 Sharpe noise floor, most of the table's ordering is probably not
  significant, and it should say so.

---

## What I would not do

- **More indicators, more ML on 5m bars.** Attacks none of the four
  constraints.
- **Sentiment / social media.** Not orthogonal — it is a lagged function
  of price, and the data is expensive and revision-prone.
- **Higher-frequency execution.** The fee study closed this; turnover is
  the enemy at every tier available.
- **Elliott waves.** Assessed separately in
  [ELLIOTT_WAVES.md](ELLIOTT_WAVES.md).

## Ranking

*Superseded by the backlog at the bottom of [LEDGER.md](LEDGER.md), which
is re-ranked every session. Kept here as the reasoning that produced it.*

1. ~~**E-process regime detection with unified Kelly sizing**~~ — built,
   R-28. Negative on return, and the strongest risk finding in the repo.
2. **Extend the funding series through 2026** — one data fetch that
   opens or closes the carry direction outright. Now blocked: every
   exchange endpoint is refused by the network policy these sessions run
   under, so this needs the operator rather than a session.
3. ~~**Purged CV, deflated Sharpe, bootstrap intervals**~~ — built, R-29.
   It did not make the existing numbers more trustworthy; it showed that
   most of them were never distinguishable in the first place.
4. **On-chain, sign-corrected** — the only genuinely new information
   channel, entered with the 141→4 base rate in mind. Also blocked on
   network access.

The first was the interesting one, and it delivered exactly what this
section predicted it would: better calibration is a risk property. The
second is still the one most likely to change a decision, and is now the
thing to unblock rather than the thing to do.

**What R-29 changes about this whole page.** Directions 1 and 5 are built,
and between them they say the same thing from two sides: honest inference
on this dataset licenses very little. The holdout has been read ~88 times
and the leading strategy would need a 6.2-year track record to clear its
own trials bar on the 3.6 years available. Adding a sixth direction that
re-reads the same decade cannot fix that — only new, uncontaminated data
can, which makes the forward paper-trading recorder (B-06) the highest
item on merit and the network policy the thing standing in its way.
