# R-184 pre-registration — a PBO/CSCV overfitting audit of `kelly_regime_v4`'s own shipped hyperparameters

Frozen before Stage 0 was computed. See `experiments/r184_shared.py` for the
executable form of everything below; this file is the design record.

## 1. The idea

`kelly_regime_v4`'s shipped hyperparameters (20/40/80-day anchor ladder,
10% deadband) were selected across R-34→R-60 by simple inner-train /
inner-validation point comparison and never audited for overfitting
against their own already-tried neighbourhood using Bailey, Borwein,
López de Prado & Zhu's (2014) combinatorially-symmetric cross-validation
(CSCV) / Probability-of-Backtest-Overfitting (PBO) statistic. This round
computes that number using this project's own already-built
`tradebot.inference.cpcv_splits` / `purged_train_mask` / `fold_mask`
machinery (R-29), and only if it flags the shipped config as more likely
than not overfit does either implementation branch (reselection vs.
ensembling) get exercised.

## 2. Step 1 — the four filter questions

**Q1 — which constraint?** Primarily **ERR** (no error control anywhere in
the signal path) — applied for the first time to the *hyperparameter
selection step* itself rather than the live trading decision (13 prior
ERR-axis rounds — R-28/31, R-87×2, R-104, R-105×2, R-114×2, R-160×2,
R-174×2, R-181×2 — all gate the live vote/exposure decision; none audits
the design-time selection of v4's own anchor span or deadband).
Secondarily **N≈3**: PBO is the standard tool for asking whether a point
estimate is distinguishable from its neighbours given how little
independent history exists, which is exactly this project's own
diagnosis.

**Q2 — not a duplicate of:** R-29 (CPCV on the whole-roster comparison
table's selection rule, never on one strategy's internal knobs, and never
computes the Bailey et al. logit-rank PBO statistic — only a simpler "did
the fold winner beat buy-and-hold" fraction); R-46/R-59/R-60/R-89/R-92/
R-146 (21+ point-estimate hyperparameter retunes via a single inner-train
→ inner-validation comparison, never a combinatorial overfitting
probability over the candidate set); R-101/R-105 (jackknife/bootstrap
confidence discounts applied continuously to the *live* vote/scale, not a
one-time design-time audit of which grid point to ship); R-104 (resamples
the *daily return series* for a live per-bar trust signal, not the
*candidate-hyperparameter grid* for a selection/ensembling signal); R-127
(a pure BTC/ETH diagnostic, different object entirely); R-146 (changes
the anchor's central-tendency estimator, holding span/deadband fixed —
this round holds the estimator fixed and asks only whether the
span/deadband combination itself was well-selected).
`grep -niE "PBO|probability of backtest overfitting|CSCV|Bailey.*Borwein"
docs/LEDGER.md` returns zero hits before this entry.

**Q3 — simulable?** Yes. Everything is the project's standard harness (5m
OHLCV bar-close signals, next-open fills, no order book); the one
missing piece — a function turning `cpcv_splits`/`purged_train_mask`/
`fold_mask` into the actual Bailey et al. PBO statistic — is built in
`r184_shared.py`, not proxied from anything not in the file.

**Q4 — what would make it fail, named now:** (a) PBO(grid) turns out low
(< 0.30) — the shipped config is not flagged as overfit, nothing for
either branch to fix, both stop by construction; (b) PBO(grid) is high but
neither branch's response survives risk-matching or ETH replication —
the shape 20+ SIZE-axis and 13 ERR-axis attempts have already hit.

## 3. Step 2 — literature basis

Bailey, Borwein, López de Prado & Zhu (2014/2016), "The Probability of
Backtest Overfitting", J. Computational Finance 20(4), 39–69 (arXiv:1302.3145).
CSCV: split the return matrix into S contiguous groups, form every
combination of S/2 as "training", the complement as "test"; pick the
in-sample winner in each combination, find its out-of-sample rank,
logit-transform it. PBO = P(logit ≤ 0). The paper's own demonstrations
carry no transaction-cost assumption and no multi-instrument panel — it
is a measurement-method paper; the economic content here comes entirely
from this project's own fully-costed BTC/ETH backtests, not from
anything Bailey et al. claim works.

Arian, Norouzi M. & Seco (2024), "Backtest overfitting in the machine
learning era", Knowledge-Based Systems 305, 112477 — CPCV beats K-Fold/
Purged K-Fold/Walk-Forward on measured PBO and deflated-Sharpe, and its
"Bagged CPCV"/"Adaptive CPCV" extensions improve robustness by
*ensembling* the low-overfit-risk region rather than committing to one
winner — the direct citation for this round's novel branch.

Reused, not re-derived: Politis & Romano (1994) stationary bootstrap;
Bailey & López de Prado (2014) deflated Sharpe; López de Prado (2018)
Ch. 7/11–12 purging/embargo; Lo (2002) analytic Sharpe-difference SE.

## 4. Two variant designs (both gated behind Stage 0)

Shared, frozen engine (`r184_shared.py`): 35-config grid, H1 ∈
{16,18,20,22,24,26,28} days (doubling ladder H1/2·H1/4·H1 held fixed) ×
deadband ∈ {0.05, 0.075, 0.10, 0.15, 0.20}, including the shipped point
(H1=20, deadband=0.10). `target_vol`/`max_leverage` held fixed throughout.
CSCV: n_groups=10, k_test=2 (45 splits, matching `scripts/inference.py`'s
own `cpcv()` default), BTC spot, 2017-01-01→2022-12-31, no holdout read.

**Conservative — PBO-gated reselection.** If Stage 0 flags the shipped
point, replace it with the grid point minimizing its own PBO-contribution
— no new parameter values, a stricter statistic replacing the ad hoc
single-split comparison R-34→R-60 used. Falsification test: ETH ΔSharpe
vs. shipped more negative than −0.05 kills it. Trivial-selection
short-circuit: if the minimizer equals the shipped point, REJECT by
construction.

**Novel — PBO-weighted continuous ensemble.** Blend the `target` exposure
of the k=5 (pre-registered) lowest-PBO-contribution configs into one
inverse-PBO-weighted average at every bar (Arian/Norouzi M./Seco 2024's
"Bagged CPCV" idea), robustness-checked at k=3 and k=7. Same falsification
test as conservative.

## 5. Pre-registered decision rule (partitions the outcome space)

**Stage 0 (shared):** compute PBO(grid) and a group-resampling bootstrap
95% CI around it (500 resamples, resample the 10 CSCV groups with
replacement). *Proceed only if the CI's lower bound clears 0.30* — the
point estimate alone is not trusted, per the power check below. Below
that, **STOP — REJECT both branches, NEGATIVE-BY-CONSTRUCTION.**

**Stage A (if Stage 0 proceeds; inner-validation 2021-01-01→2022-12-31,
both markets):** A1 promotion (ΔSharpe ≥ +0.20 both markets, 95% paired
stationary-block-bootstrap CI excluding zero, OR a risk-matched
[exposure/vol ratio in 0.9–1.1] drawdown improvement ≥5pp both markets);
A2 falsification (ETH ΔSharpe ≥ −0.05, else REJECT regardless of A1); A3
plateau (≥2 of 3 neighbouring grid points / swept k values also clear A1);
A4 fee-tier robustness (sign does not reverse at 0.40%).

**Stage B (holdout, only if A1–A4 all pass):** identical construction,
`OOS_START=2023-01-01` onward. PROMOTE iff it also clears; otherwise
REJECT.

**Power check.** (1) PBO's own resolving power at 45 (highly overlapping,
non-independent) splits: a conservative binomial bound gives SE≈0.0745 at
p=0.5, a ±0.146 half-width — already "not razor-sharp" before any
empirical measurement, which is why the CI (not the point) gates Stage 0.
(2) The +0.20 Sharpe threshold: Lo (2002)'s paired-Sharpe SE at n=730 days,
ρ≈0.95 gives SE≈0.186, matching R-20's own ±0.2 floor almost exactly — the
2-year inner-validation window is the right order of magnitude to resolve
it, confirmed independently of this round's own outcome.

Holdout counter status entering this round: unchanged at ~766 (R-178's
figure; no round since has reached Stage B). This design does not read
the holdout; the counter increments only if Stage B is actually reached.
