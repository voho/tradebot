# B-19 (conservative branch) — the never-rebalanced static split (08-20)

Unregistered experiment. Code: `experiments/b19_dual_fixed_split.py`. Not
`@register`ed, not auto-discovered, nothing committed by this session.
`kelly_regime_v4` and `src/tradebot/multiasset.py` are imported and called
unmodified throughout — neither is edited anywhere in this branch.

**Backlog item attacked:** B-19 — does R-50's periodically-rebalanced
static-50/50 BTC+ETH `kelly_regime_v4` portfolio (found as a byproduct of
the B-18 round, ΔSharpe +0.79/+0.80 monthly/weekly, max DD 33.2%→27.1% on
inner-validation, never pre-registered, no falsification test, no holdout
read) survive real scrutiny? B-19's own note names the cheapest first
check: re-express the candidate as a **one-time-split adapter** using the
existing `multiasset.py` — no periodic rebalancing at all — rather than
building periodic-rebalance-via-return-splicing into `multiasset.py`. This
file is that check.

## 1. Idea and mechanism, one sentence

Split capital 50/50 (and, as a plateau check, 60/40 and 40/60) between two
fresh, unmodified `KellyRegimeV4` instances — one on BTC, one on ETH —
**once**, via `tradebot.multiasset.run_multi_backtest`, and never touch the
split again; if imperfectly-correlated bad regimes (N≈3) let a fixed-but-
never-rebalanced blend still draw down less than either single-asset book,
that is a real diversification effect obtained at zero rebalancing
turnover, and it isolates whether R-50's number needed the rebalancing at
all.

## 2. Constraint attacked, and why this is not a duplicate

**SIZE and N≈3** — identical framing to B-19's own row and to
`experiments/kelly_regime_dual_fixed.py` (R-42/R-43, B-16): a second,
imperfectly-synchronized regime-cycle exposure, not a change to v4's own
vote or sizing formula (imported unchanged).

Not a duplicate of:

- **R-50 / `kelly_regime_covkelly_v3_continuous.py`'s `fixed5050_continuous`
  arm** — read directly before writing this file. That arm IS
  periodically rebalanced: every segment boundary (monthly or weekly) it
  recomputes `dollars_b = pooled * w_b` and rescales each leg's already-
  continuous equity curve back to exactly 50/50 of pooled capital. That is
  the periodic-rebalance-to-fixed-weights design the Booth & Fama /
  Willenbrock definition below requires. This file's candidate splits
  capital **once** and never touches it again — weights drift with each
  leg's own performance for the rest of the window. Periodic rebalancing
  is out of scope for this branch by the task's own instruction; a
  separate, disjoint session is attacking a periodically-rebalanced novel
  variant.
- **`kelly_regime_dual_fixed.py` (R-42/R-43, B-16)** — also a one-time,
  never-rebalanced split, and this file's `SPLITS` grid intentionally
  mirrors its weight choices. The difference this file exists to make is
  the one the task specifies: composing the two legs through the tested,
  promoted, general primitive `tradebot.multiasset.run_multi_backtest` /
  `MultiAssetSpec` (R-49) rather than that file's own ad hoc
  `combine_equity` helper (which R-49's module docstring says
  `multiasset.py` was "generalized from"). It also runs the mandatory
  R² exposure-artifact check and a 0.40% fee-tier re-run explicitly
  pre-registered as a joint go/no-go gate before any holdout read — B-16's
  own holdout read answered a different, resampled-window question
  (bear-quartile drawdown-delta) and is cited here, not repeated.

## 3. Sources used (found and read directly, not copied from the task brief)

- Booth, D. G. & Fama, E. F. (1992), "Diversification Returns and Asset
  Contributions," *Financial Analysts Journal* 48(3), 26–32. Confirms the
  compound-return decomposition: a constant-weight (rebalanced) portfolio's
  compound return exceeds the weighted average of its constituents'
  compound returns, and each asset's contribution is approximated by
  subtracting one-half its *covariance* with the portfolio, not its own
  variance.
- Willenbrock, S. (2011), "Diversification Return, Portfolio Rebalancing,
  and the Commodity Return Puzzle," *Financial Analysts Journal* 67(4),
  42–49 (also arXiv:1109.1256). States plainly that the diversification
  return's source is **the act of rebalancing itself** — selling assets
  that have appreciated in relative weight and buying assets that have
  declined — in explicit contrast to a buy-and-hold portfolio, whose
  return path is instead driven by winners growing to dominate it.
- Chambers, D. R. & Zdanowicz, J. S. (2014), "The Limitations of
  Diversification Return," *Journal of Portfolio Management* 40(4), 65–76.
  Directly disputes the above: diversification return is not itself a
  source of *added expected value*; whatever excess a rebalanced portfolio
  shows over its own buy-and-hold traces to **mean-reversion** captured by
  the rebalancing trades, not to variance reduction or diversification.

These three, read together, frame exactly the question this branch is
built to answer empirically on this project's own asset pair and
strategy: is R-50's number a **blending/correlation** effect present with
zero rebalancing (Chambers' reading), or does it specifically need the
periodic sell-winners/buy-losers act (Booth & Fama / Willenbrock's
reading)? §7 below answers it, and the answer is not the same on both
axes.

## 4. Pre-registered falsification test (written before any result was read)

Two checks, **both** must pass on the inner splits before the 2023+
holdout is ever touched:

1. **Exposure-artifact check** (this project's standing "match risk before
   comparing anything" rule — R-33/R-46/L-04 all died of exactly this): the
   candidate's aggregate exposure series must NOT be an R² > 0.95 flat
   rescale of BTC-solo `kelly_regime_v4`'s own exposure, on inner-validation,
   both markets.
2. **0.40% Bitstamp taker fee tier**: the 50/50 candidate's drawdown
   advantage over BTC-solo v4 must not flip sign relative to the 0.10%
   tier, on both inner-train and inner-validation.

If either fails, stop — do not read the holdout. Report NEGATIVE.

## 5. Pre-registered promotion decision rule (written before the holdout was read)

Promote (`PROMOTED-CANDIDATE`) only if, on the 2023+ holdout, using the
**frozen 50/50 configuration and no other**:

- (a) beats `buy_and_hold` OOS after real costs (0.10% and 0.40%);
- (b) the improvement over BTC-solo `kelly_regime_v4` exceeds the ±0.2
  Sharpe noise floor (R-20) **or** is a drawdown/tail improvement;
- (c) survives both falsification checks in §4;
- (d) the 50/50 → 60/40 → 40/60 neighbourhood is a plateau, not a
  knife-edge.

Anything else is `NEGATIVE`. This text was written into
`experiments/b19_dual_fixed_split.py`'s module docstring before `sweep()`,
`select()`, `artifact()` or `feetier()` were ever run, and the rule was
**not edited** after the holdout was read in §8.

## 6. Causality check on this file's own composition code

`multiasset.py` is already causality-tested in isolation
(`tests/test_multiasset.py`) and `kelly_regime_v4` is already
causality-tested by CI. What is new here is the specific call this file
makes: two `KellyRegimeV4` legs on real BTC/ETH data through
`run_multi_backtest` at a fixed split. Standard two-opposite-tampers probe
(cut inside inner-train, 2020-06-30, nowhere near the holdout): bars after
the cut multiplied by 137× in one copy, divided by 137× in the other.

| check | max\|diff\| strictly before cut | result |
|---|---|---|
| portfolio equity, up-tampered vs base | 0.000e+00 | PASS |
| portfolio equity, down-tampered vs base | 0.000e+00 | PASS |

No lookahead in this file's composition.

## 7. Inner-train and inner-validation results

Inner-train = 2019-03-14 (ETH's real start on the committed Coinbase file)
→ 2020-12-31. Inner-validation = 2021-01-01 → 2022-12-31 (the 2022 BTC/ETH
joint bear), matching the task brief's real constraint rather than
ROUTINE.md's generic 2017 example. `buy_and_hold` and `kelly_regime_v4`
baselines were re-derived in this session (not copied from the ledger) and
match the ledger's own cited numbers exactly (e.g. inner-validation spot
v4-solo: $998, Sharpe 0.14, DD 33.2% — identical to the number cited
independently in the `kelly_regime_v5_damp` report), which is itself a
useful cross-check that this file's harness is wired correctly.

### 7a. Inner-train (spot)

| strategy | final | Sharpe | max DD |
|---|---|---|---|
| `kelly_regime_v4` BTC-only | $6,167 | 2.62 | 30.4% |
| dual 50/50 (this candidate) | $4,410 | **2.85** | **29.9%** |
| dual 60/40 BTC | $4,761 | 2.82 | 30.1% |
| dual 40/60 ETH | $4,059 | 2.82 | 29.9% |
| naive 50/50 buy&hold BTC+ETH | $6,604 | 2.04 | 68.9% |
| `buy_and_hold` BTC-only | $7,458 | 1.81 | 71.8% |

### 7b. Inner-validation (spot and 5x futures)

| strategy | market | final | Sharpe | max DD |
|---|---|---|---|---|
| `kelly_regime_v4` BTC-only | spot | $998 | 0.14 | 33.2% |
| dual 50/50 | spot | $1,125 | **0.37** | **27.0%** |
| dual 60/40 BTC | spot | $1,099 | 0.32 | 27.9% |
| dual 40/60 ETH | spot | $1,150 | 0.41 | 26.2% |
| naive 50/50 buy&hold | spot | $1,098 | 0.45 | 79.2% |
| `buy_and_hold` BTC-only | spot | $574 | 0.08 | 77.3% |
| `kelly_regime_v4` BTC-only | futures5x | $1,064 | 0.25 | 32.3% |
| dual 50/50 | futures5x | $1,222 | **0.55** | **27.5%** |
| dual 60/40 BTC | futures5x | $1,191 | 0.50 | 28.4% |
| dual 40/60 ETH | futures5x | $1,254 | 0.58 | 26.7% |

**Neighbourhood is a plateau, not a knife-edge**: on inner-validation spot,
Sharpe spans 0.32–0.41 and drawdown spans 26.2–27.9% across the three
splits — moving in one direction smoothly, no sign flip, no cliff.
Promotion criterion (d) is satisfied on the inner splits.

### 7c. The Chambers-vs-Booth&Fama comparison — the headline result of this branch

R-50's own cited number (periodically-rebalanced, continuous engine,
inner-validation spot, monthly/weekly cadence): **ΔSharpe +0.79/+0.80,
max DD 33.2% → 27.1%** versus BTC-solo v4.

This branch's never-rebalanced static 50/50 (same window, same market,
same baseline): **ΔSharpe = 0.37 − 0.14 = +0.23** (≈29% of R-50's monthly
figure), **max DD = 27.0%** (33.2% → 27.0%, essentially identical to
R-50's 33.2% → 27.1%, within 0.1pp — noise-level agreement).

| axis | R-50 (periodic rebalance) | this branch (never rebalanced) | fraction captured |
|---|---|---|---|
| ΔSharpe | +0.79 / +0.80 | +0.23 | ~29% |
| max DD improvement | 33.2% → 27.1% (−6.1pp) | 33.2% → 27.0% (−6.2pp) | ~100% |

**The risk-side benefit is fully present with zero rebalancing; the
return-side benefit mostly is not.** That is not an ambiguous outcome — it
splits the literature's disagreement cleanly along its own fault line:
the drawdown result supports **Chambers (2014)**'s claim that the
risk-reducing property is ordinary correlation/volatility blending, not
something that requires periodically resetting weights. The Sharpe result
supports **Booth & Fama (1992) / Willenbrock (2011)**'s claim that the
*return*-side diversification premium specifically needs the sell-
winners/buy-losers act of rebalancing — roughly 71% of R-50's Sharpe
improvement evaporates once that act is removed, even though the
correlation structure between the two legs (and therefore the drawdown
benefit) is unchanged. Neither camp is simply wrong on this data; they are
each right about a different axis of the same headline number.

## 8. Falsification test results (both pre-registered, run before the holdout)

### 8a. Exposure-artifact check (§4.1)

Aggregate exposure = Σ(leg target fraction × leg equity) / total portfolio
equity, compared via R² against a mean-matched flat rescale of BTC-solo
v4's own `target` series, inner-validation:

| market | rescale c | corr | R² | verdict |
|---|---|---|---|---|
| spot | 0.953 | 0.9523 | **0.8685** | not a flat rescale — PASSES |
| futures5x | 0.952 | 0.9498 | **0.8617** | not a flat rescale — PASSES |

Both well under the 0.95 threshold. **This branch's §7c result is not the
exposure-level artifact this project has been burned by three times
before (L-04/R-33, R-31, R-32)** — the correlation with a rescaled v4
solo book is real (~0.95, unsurprising since both legs share v4's BTC-vote
architecture) but the fit is not tight enough to be "just leverage in
disguise."

### 8b. 0.40% taker fee-tier check (§4.2)

| window | fee | dual 50/50 final | Sharpe | DD | v4-solo final | Sharpe | DD | ΔSharpe | ΔDD |
|---|---|---|---|---|---|---|---|---|---|
| train | 0.10% | $4,410 | 2.85 | 29.9% | $6,167 | 2.62 | 30.4% | +0.23 | −0.4pp |
| train | 0.40% | $3,808 | 2.59 | 33.7% | $5,294 | 2.42 | 34.5% | +0.17 | −0.8pp |
| validation | 0.10% | $1,125 | 0.37 | 27.0% | $998 | 0.14 | 33.2% | +0.23 | −6.1pp |
| validation | 0.40% | $942 | −0.01 | 33.7% | $834 | −0.17 | 39.7% | +0.15 | −6.0pp |

The drawdown advantage does not flip sign at 0.40% on either window (train:
−0.4pp → −0.8pp; validation: −6.1pp → −6.0pp — if anything slightly more
stable). Sharpe advantage shrinks somewhat (+0.23→+0.17 train, +0.23→+0.15
validation) but stays positive in every cell. **PASSES.**

Both pre-registered gates cleared → per §4/§5, the holdout is read once,
frozen at the 50/50 configuration.

## 9. Holdout read (ONE read, frozen 50/50 config, per §5)

2023-01-01 onward, spot, both fee tiers:

| fee tier | dual 50/50 final | Sharpe | DD | v4-solo final | Sharpe | DD | `buy_and_hold` final | Sharpe |
|---|---|---|---|---|---|---|---|---|
| 0.10% | $2,922 | 1.24 | 27.0% | $3,373 | 1.22 | 27.8% | $3,839 | 1.03 |
| 0.40% | $2,075 | 0.89 | 32.6% | $2,445 | 0.94 | 34.1% | $3,827 | 1.03 |

Applying the pre-registered rule from §5, exactly as written, no goalposts
moved:

- **(a) beats `buy_and_hold` OOS after real costs?** No. At 0.10%, dual
  50/50 ($2,922) loses to holding ($3,839) by −23.9%. At 0.40% the gap
  widens (−45.8%). **FAILS.**
- **(b) improvement over BTC-solo v4 exceeds the ±0.2 Sharpe noise floor,
  or is a drawdown/tail improvement?** ΔSharpe at 0.10% = 1.24 − 1.22 =
  **+0.02** — deep inside the noise floor. ΔDD = 27.0% − 27.8% = **−0.8pp**
  — not a meaningful drawdown improvement (compare to the −6.1pp seen on
  inner-validation: it has almost entirely evaporated). At 0.40%,
  ΔSharpe = 0.89 − 0.94 = **−0.05** (the dual book is now slightly
  *worse* than v4-solo), ΔDD = **−1.5pp**. **FAILS** on both fee tiers.
- (c) survives both falsification tests: yes (§8), but (a) and (b) already
  fail — the falsification test being clean does not rescue a promotion
  bar failure on the actual decision criteria.
- (d) neighbourhood plateau: not re-tested on the holdout by design — only
  the frozen 50/50 config was read there, exactly as pre-registered, so
  this criterion carries over from its inner-validation pass (§7b) rather
  than being re-evaluated on a resource the rule reserves for one look.

**Neither (a) nor (b) holds. Verdict: NEGATIVE, per the rule fixed in §5
before this section was written.** The rule was not moved after seeing
these numbers — no threshold in §5 was edited, and no alternative
candidate (e.g. 40/60 ETH, which looked marginally best on inner-
validation, §7b) was substituted or read on the holdout to try to rescue
the result.

**This also directly answers §7c's headline finding at the point that
matters most.** The drawdown edge that fully replicated a diversification-
only benefit on inner-validation (§7c: 100% of R-50's DD improvement
captured with zero rebalancing) shrinks from −6.1pp to −0.8pp on the
holdout — consistent with this project's own repeated finding
(L-01/R-33's own caveat on v4's headline, R-29's "no Sharpe-based claim
from this dataset is supportable any more") that an inner-validation gap,
however clean it looks and however well it is explained by a real
mechanism, is not guaranteed to survive a fresh three-and-a-half-year
window. 2023-2026 was a strong, comparatively low-volatility BTC/ETH bull
run; a book carrying less exposure than fully-invested holding
underperforms holding in exactly that regime, and this holdout is mostly
that regime.

## 10. Configurations evaluated (deflated-Sharpe bookkeeping)

**13 distinct dual-book (candidate) backtest configurations**, the
free-parameter axis actually being searched (split ratio × window ×
market × fee tier):

1. 50/50, train, spot @0.10%
2. 60/40 BTC, train, spot @0.10%
3. 40/60 ETH, train, spot @0.10%
4. 50/50, validation, spot @0.10%
5. 50/50, validation, futures5x
6. 60/40 BTC, validation, spot @0.10%
7. 60/40 BTC, validation, futures5x
8. 40/60 ETH, validation, spot @0.10%
9. 40/60 ETH, validation, futures5x
10. 50/50, train, spot @0.40% (fee-tier falsification test)
11. 50/50, validation, spot @0.40% (fee-tier falsification test)
12. 50/50, **holdout**, spot @0.10%
13. 50/50, **holdout**, spot @0.40%

Plus, **not counted toward the trials figure above** (baseline/reference
runs, matching `kelly_regime_dual_fixed.py`'s own established convention
for this project): 12 `{v4-solo, naive-dual-hold, buy_and_hold}` baseline
backtests across both windows and both markets in `baselines()`, 4 more
`v4-solo`/`buy_and_hold` reference runs at the 0.40% tier, and 4 more on
the holdout — 20 baseline backtests in total — plus 3 tamper-probe runs for
the causality check (§6), which reads no window past inner-train and is a
correctness check, not a search point.

**Holdout counter: +2** on top of the project's running total of ~621 as
of the last ledger entry (08-20, R-50: +0) — one frozen configuration
(50/50) read at two fee tiers, both pre-registered as a single paired
read, matching the convention used for R-35's fee-tier pair.

## 11. Verdict

**NEGATIVE.**

The mechanism is real and the falsification tests are clean: this is not
an exposure-level artifact (R²=0.86–0.87, well under the 0.95 danger
line), it survives the 0.40% fee tier without its drawdown advantage
flipping sign, and the split-ratio neighbourhood on inner-validation is a
genuine plateau. It also produced a genuinely informative answer to the
literature question this branch was built to test (§7c): **the
diversification/blending risk-reduction benefit does not need periodic
rebalancing (Chambers, 2014 — confirmed on this data), but roughly 71% of
the periodically-rebalanced version's Sharpe improvement does trace to the
rebalancing act itself (Booth & Fama 1992 / Willenbrock 2011 — also
confirmed on this data, on the return axis specifically)**.

None of that clears the project's own promotion bar. On the one holdout
read this branch's pre-registration authorized, the static 50/50 dual
book (a) loses to simple buy-and-hold decisively, at both fee tiers, and
(b) is statistically indistinguishable from — and at the 0.40% tier
slightly *worse* than — BTC-solo `kelly_regime_v4` alone. The 27.0%→33.2%
drawdown edge that looked clean and mechanism-backed on inner-validation
compresses to a 0.8pp non-effect on the holdout. The decision rule from §5
was not moved after this was read.

**One-line lesson:** a real, literature-grounded, non-artifact mechanism
(never-rebalanced diversification cuts drawdown, exactly as Chambers'
theory predicts, with zero rebalancing turnover) can still fail the
promotion bar outright — R-50's ΔSharpe was mostly the rebalancing, and
even the part that IS real (the drawdown cut) does not survive 2023-2026's
bull-dominated holdout at a scale that matters.

**Practical implication for B-19 and B-17.** This settles B-19's own
cheapest-first-check question in the negative for the never-rebalanced
adapter path specifically, without needing to build periodic-rebalance-
via-return-splicing into `multiasset.py` (the more expensive alternative
B-19's own note named) — that infrastructure investment is not justified
by this result. Whatever a periodically-rebalanced dual-book candidate
does on the holdout (the separate, disjoint session's question) is now
the only live thread left in this backlog item; this branch's own
one-time-split candidate is closed as REJECTED-ON-HOLDOUT and should not
be re-tried without new evidence (e.g., a different asset pair, a
different rebalancing-free composition, or a materially different holdout
period).
