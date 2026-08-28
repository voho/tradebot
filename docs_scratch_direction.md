# R-171 design — Online Newton Step sizing for `kelly_regime_v4`'s SCALE factor

Written pre-implementation, pre-holdout. No backtest was run to produce
this document; no file under `experiments/` or `src/` was modified; no
data past 2022-12-31 was read. This is theory/mechanism design only, per
the R-171 brief.

## 1. Direction (one sentence)

Replace `kelly_regime_v4`'s conditional volatility-target `scale`
(`min(target_vol/vol, max_leverage)`) with a scalar learned online by the
**Online Newton Step (ONS)** algorithm for portfolio management (Agarwal,
A., Hazan, E., Kale, S. & Schapire, R. E. (2006), "Algorithms for
Portfolio Management Based on the Newton Method," *Proceedings of the
23rd International Conference on Machine Learning (ICML)*, 9–16) — a
projected second-order online-convex-optimization update that minimizes
**regret against the best fixed leverage in hindsight, over the specific
realized path, with no distributional or i.i.d. assumption** — attacking
**N≈3** (its guarantee needs no repeated independent regime draws to be
valid, unlike every plug-in moment estimator this ledger has tried) and
secondarily **SIZE**. It is explicitly **not a duplicate** of:

- **L-11 `universal_kelly`** (Cover 1991) and **R-149** (which retuned
  `universal_kelly`'s own grid with a vol-target scalar and Herbster–Warmuth
  1998 fixed-share reinjection): Cover's method is a **zeroth-order,
  wealth-weighted grid-integration mixture** over fixed fractions — a static
  Bayesian-style average that L-11 itself found updates too rarely ("nine
  trades in a decade"). ONS is a **first/second-order gradient method**: one
  closed-form Newton step per bar, using the *local* gradient and an
  accumulated Hessian proxy, not an integral over a static grid. Different
  update rule, different (and, per the paper, provably tighter/faster-adapting)
  computational form of the *same* regret target — R-149 retuned Cover's
  grid twice; nobody has run the Newton-step algorithm at all.
- **R-28 / R-31** (e-process "testing by betting" Kelly sizer, retracted):
  R-28's mechanism is a **sequential hypothesis test** — a betting
  martingale against the null "drift = 0," whose *wealth* becomes the
  position. ONS contains no null hypothesis, no p-value, no martingale
  test anywhere: it is a direct regret-minimizing convex-optimization
  update on realized log-growth. R-28 was retracted because its **drawdown
  claim** turned out to be an unmatched-exposure artifact (R-31/R-33); that
  finding is about R-28's *result*, not about online convex optimization
  as a family, and this design's decision rule (§4) explicitly gates on
  the same matched-exposure check R-33 established.
- **R-152/R-153 (CDaR), the Grossman–Zhou family closed by R-93, R-164
  (Barroso–Santa-Clara / Daniel–Moskowitz risk-managed momentum), CPPI and
  Hurst-adaptive CPPI, Busseti–Ryu–Boyd risk-constrained Kelly, CRRA/Merton
  drift-over-variance, "per-vote-state Kelly fraction (`μ_state/σ_state²`)"**:
  every one of these is a **plug-in rule** — it estimates a moment (mean,
  variance, CDaR, drawdown probability) from a trailing window and applies
  a closed-form formula. None uses an online-convex-optimization update
  with a regret guarantee; all inherit the estimation-error fragility
  MacLean–Thorp–Ziemba (2010) warn about and this project has now measured
  failing ~28 times. ONS estimates nothing — it takes a gradient step.
- **R-164's "inverse leverage effect" closure** ("ANY volatility-scaling
  substitution on this strategy's leveraged leg fails," Baur & Dimpfl 2018):
  that finding is specifically about using **realized price volatility as a
  sizing denominator** (`target_vol / vol`), which is backwards on BTC
  because high vol *precedes* high forward Sharpe (R-10). ONS's gradient and
  Hessian are of **realized wealth growth** (`log(1+b·r)`), not of price
  volatility — `vol` never appears anywhere in the ONS update. This is the
  single most important non-duplication argument in this document and is
  re-checked empirically at Step 0 of implementation (§4, kill switch KS-0).
- **"Periodic causal walk-forward re-estimation of `target_vol`/`max_leverage`"**
  (365-day refit, fee-free proxy-Sharpe grid search): a **batch** re-fit on
  a fixed annual schedule with no regret guarantee, versus ONS's continuous
  per-bar recursive update with an O(log T) worst-case guarantee.
- **"Minimax-across-purged-folds robust reselection"**: offline minimax
  parameter reselection across folds — not an online algorithm at all.
- **R-147 (James-Stein shrinkage across the 3 vote anchors)**: operates on
  the **VOTE**, not the SCALE; per R-62's factorization (VOTE carries v4's
  signature, SCALE does not), this design deliberately stays on the SCALE
  slot, the established locus for SIZE-axis primitives since R-62.

## 2. ROUTINE.md Step 1 — the four-question filter

**1. Which constraint does it attack?** Primarily **N≈3**. Every SIZE-axis
mechanism this project has tried before (CPPI, CDaR, Grossman–Zhou,
Barroso–Santa-Clara, Busseti–Ryu–Boyd, CRRA/Merton, per-vote-state Kelly)
is a plug-in estimator of some population moment (mean, variance, drawdown
probability), and all such estimators need enough *independent* history to
be trustworthy — exactly what an effective sample size of ~3 regime events
denies them. ONS's regret bound (Agarwal et al. 2006, Thm. 2/3: cumulative
log-wealth within `O(n log T)` of the best fixed fraction in hindsight, for
`n` assets and `T` rounds) holds **pathwise, against the single realized
sequence, with zero distributional assumption** — it is the one guarantee
on this project's whole SIZE-axis list that does not need repeated
independent regime draws to be *valid* (whether it is *large enough to
matter* is a separate, empirical question — see §4's honest treatment of
this). Secondarily **SIZE** (it decides how much to hold, the axis with
this project's entire track record of wins).

**2. Which ledger entries is it not a duplicate of?** L-11, R-149, R-28,
R-31, R-152, R-153, R-93's Grossman-Zhou closure, R-164, R-166 (vol-response
sign inversion — different object: R-166 flips the *sign* of the existing
vol-target exponent, still using `vol` as the denominator; ONS uses no
volatility denominator at all), the CPPI/Hurst-CPPI pair, Busseti-Ryu-Boyd,
CRRA/Merton, "per-vote-state Kelly fraction," the walk-forward-refit
closure, the minimax-reselection closure, R-147 — reasons given in §1.
`grep -c "Newton\|Agarwal\|Online Newton Step\|ONS\b" docs/LEDGER.md`
returns zero hits outside this document as of R-170; the phrase "online
portfolio selection" appears once, inside R-164's own entry, as an
unelaborated item in a sub-agent's literature-survey list ("duplicates or
infeasible-by-construction") with no round, config count, or measurement
attached to it anywhere in section B or C — per ROUTINE.md's own standing
rule ("not tested is not a negative result… a branch that produced no
evaluated configuration goes back on the backlog as untried, never into
the ruled-out list"), that mention does not constitute a prior attempt.

**3. Is it simulable here?** Yes, with zero new data. The ONS update needs
only the strategy's own bar-by-bar `frac_t` (already computed causally in
`kelly_regime_v3.prepare()`) and the realized next-bar log return of the
underlying asset — both already inside the existing `prepare()` frame. The
Newton step for `b_t` uses only information through bar `t-1` (the standard
`.shift(1)` convention this codebase already uses for `vol`), so causality
is structural, not a property that has to be argued for.

**4. What would make it fail? (named now, before any code)**
- **(a) Corner lock-in.** Projected ONS's Hessian-proxy `A_t` accumulates
  `∇_s∇_s^T` from every bar; a few large-|return| bars early in training can
  make `A_t` dominated by them, driving `b_t` to a corner (0 or `max_leverage`)
  it cannot escape for a long stretch — a known small-sample weakness of
  Newton-type online portfolio algorithms relative to Cover's smoother
  integral, and the reason Agarwal et al.'s own paper recommends a
  regularization floor on `A_t` (`ε·I`). If the learned `b_t` path is
  observed to spend >80% of inner-train bars pinned at a boundary, that is
  a design failure, reported as such, not smoothed over by re-tuning `ε`
  after looking at the result.
- **(b) Inside the noise floor.** The achieved ΔSharpe, even if positive
  and directionally sensible, sits inside R-20's ±0.2 Sharpe bar — the
  outcome ~15 of ~15 prior SCALE-axis substitutions with a reported
  inner-validation ΔSharpe have produced (§4 below).
- **(c) Exposure-collapse artifact.** The ONS-learned `b_t` path correlates
  `R² > 0.95` with v4's existing `target_vol/vol` path — i.e., it has
  relearned the incumbent rather than found anything different — the same
  kill-switch KS-B this project has used since R-33/R-34/R-41/R-53/R-73.

## 3. Variants

Both variants touch **only** the `scale` term inside `kelly_regime_v3.prepare()`
(`desired = frac * scale`); the vote (`frac`) and the 10% deadband are
byte-identical to v4, per R-62's isolation discipline.

### Conservative — literal single-scalar ONS

**Mechanism, one sentence.** Treat "how much to lever the already-decided
vote" as Kelly's original one-asset (bet fraction `b` vs. cash) problem and
solve it online with Agarwal-Hazan-Kale-Schapire's Newton step instead of a
volatility ratio: at each bar, `b_t = Π_[0, max_leverage]( b_{t-1} -
(1/β)·A_{t-1}^{-1}·∇_{t-1} )`, where `∇_{t-1} = -r_{t-1}/(1+b_{t-1}·r_{t-1})`
is the gradient of `-log(1+b·r)` at the previous step's realized bet-payoff
`r_{t-1} = frac_{t-1}·(asset return over that bar)`, and `A_{t-1} = ε·I +
Σ_{s<t} ∇_s∇_s^T` is the paper's own rank-one Hessian-proxy accumulation
(`ε`, `β` set from the paper's own derived formulas given the domain bound
`[0, max_leverage=2.0]` and the empirical gradient bound on this series —
**not swept**, since the whole point of the conservative arm is the paper's
literal, parameter-light construction). `scale_ONS = b_t` replaces
`min(target_vol/vol, max_leverage)` outright; `desired = frac_t · scale_ONS`.
Should make money because it directly maximizes realized log-wealth of the
vote's own payoff stream with a worst-case adaptivity guarantee, rather than
estimating a volatility target that R-10/R-164 already showed points the
wrong way on this asset class.

**Falsification test.** ETH same-sign replication (`scripts/build_bitfinex_dataset.py`
series). **Exact kill outcome:** if the sign of ΔSharpe (ONS vs. v4) on
BTC inner-validation disagrees with the sign of ΔSharpe on ETH
inner-validation, the branch is killed regardless of the BTC number — the
standard convention this project has used since R-47/R-55 for a
single-instrument-calibrated mechanism, and the appropriate test for a
method whose entire motivation is a path-independent worst-case guarantee:
if the guarantee is real, it should not be BTC-specific.

### Novel — vol-state-conditioned ("sleeping-expert") ONS

**Mechanism, one sentence.** Run **three independent ONS learners**, one
per state of `kelly_regime_v3`'s own existing hysteresis vol-state machine
(`state ∈ {-1 low-vol breakout, 0 normal, +1 high-vol breakout}`, already
computed in `prepare()`), each accumulating its own `A_t`/`b_t` and updating
**only on bars where that state is active** — the "sleeping experts"
framework (Freund, Y., Schapire, R. E., Singer, Y. & Warmuth, M. K. (1997),
"Using and Combining Predictors that Specialize," *Proc. 29th ACM STOC*,
334–343) applied to ONS's per-context learning rather than to a discrete
prediction, so that the optimal worst-case leverage the algorithm converges
to is allowed to differ by realized-volatility regime instead of being
forced to average over all of them in one accumulator. Should reduce risk
(not just return) because it lets the "high-vol breakout" state's own ONS
instance learn a different — per R-10, likely *more* aggressive, not
less — leverage than the normal-band instance, without either instance's
Hessian being diluted by bars from the other regime, which the
single-accumulator conservative variant cannot do. This is deliberately
**not** a duplicate of the closed "per-vote-state Kelly fraction
(`μ_state/σ_state²`)" item: that item conditions on the **VOTE** state and
plugs in estimated moments; this conditions on the **existing vol-state**
machine (a different variable already in the code) and uses a regret-minimizing
gradient update (a different mechanism), with no `μ`/`σ²` estimation anywhere.

**Falsification test.** Monte Carlo stress-window survival
(`scripts/stress_test.py` / `scripts/beta_test.py --windows 24`). **Exact
kill outcome:** the novel branch's entire motivating claim is that
state-conditioning should be *more* regime-robust than one shared
accumulator — a categorical claim, so it gets a categorical test, per
R-89's precedent for this exact test type: if the state-conditioned
variant has a **deeper max drawdown than unmodified `kelly_regime_v4` in
more than 12 of the 24 stress windows** on either market, the branch is
killed outright, independent of its point-estimate Sharpe.

## 4. Pre-registered decision rule, with the noise-floor arithmetic done

This is a **backtest** comparison, not the forward B-06 recorder R-78's
warning was about — so the "compute the `n` this threshold implies" step
looks different here than it did for R-78, and that difference is itself
worth stating explicitly rather than skipped: **R-20 already ran that
computation, once, for this exact comparison type** (paired stationary
block bootstrap, 30-day blocks, 2,000 resamples, over this project's
standard training/inner-validation window), and got **±0.2 Sharpe**. Unlike
B-06's forward horizon, the backtest training window is fixed (2017–2022,
~2,192 days ≈ 73 non-overlapping 30-day blocks) and does not grow by
waiting — there is no "run it longer" escape here, so the honest move is
to **apply R-20's number precisely**, not invent a fresh one that sounds
big. Converting it to a return-space check (the "divide by the SD of the
thing being compared" step ROUTINE.md asks for): at v4's own `target_vol =
0.55` (55%/yr), a **0.2 annualized-Sharpe** difference implies an
**annualized mean-return difference of `0.2 × 0.55 ≈ 11 pts/yr`** — larger
than almost every net-of-fee edge this ledger has ever measured surviving
past Step 4, which is the correct scale for a bar that is supposed to be
hard to clear, not a formality.

**Base rate, stated honestly before any number is read:** of this
project's ~15+ prior SCALE-axis substitution attempts with a reported
inner-validation ΔSharpe (CPPI, Hurst-CPPI, CDaR ×2, Grossman-Zhou ×2,
Barroso-Santa-Clara, Daniel-Moskowitz, Busseti-Ryu-Boyd, CRRA/Merton,
per-vote-state Kelly, RCPS/conformal caps, anytime-valid concentration
bounds ×2), **zero** have cleared +0.2 Sharpe on both markets. The prior
probability this one does is therefore low, and the falsification tests in
§3 are chosen to fail fast and cheaply rather than to give the mechanism
the benefit of the doubt.

**PROMOTE a variant only if ALL of:**

1. ΔSharpe (variant vs. `kelly_regime_v4`) on inner-validation ≥ **+0.2**
   on **both** BTC and ETH, **or** a matched-exposure (`exposure_ratio` and
   `vol_ratio` both in `[0.9, 1.1]`, R-33's convention) max-drawdown
   reduction ≥ **5 percentage points** on both markets — chosen to match
   the size of v4's own registered improvement over v3's predecessor
   (41.8% → 35–39%, ≈6–7pp), itself measured by the identical block-bootstrap
   method, so this is not a threshold invented to sound impressive but the
   project's own most recent successful SIZE-axis result, reused as the bar.
2. Survives its variant's pre-registered falsification test (§3).
3. **Plateau, not peak.** Conservative: report `b_t`'s path sensitivity
   over a 3-point bracket of the regularization floor `ε` around the
   paper's own suggested default (not a free-fitted parameter — a
   robustness check on a structural constant). Novel: results must hold
   whether vol-state conditioning uses the existing 3-state machine or a
   collapsed 2-state (any-breakout vs. normal) version — not narrowly
   tuned to exactly 3 buckets.
4. Sign does not reverse at the 0.40% taker fee tier (`scripts/fee_study.py`).

**Any other outcome is NEGATIVE**, including partial passes — per
ROUTINE.md's own rule against reaching for the nearest-looking label when
a result falls between clauses, a fall-through is reported as a
fall-through, not rounded up.

## 5. Files the implementer needs to read

- `src/tradebot/strategies/kelly_regime.py` — base class: `frac`/`vol`
  computation, the `deadband` trade filter, `BARS_PER_DAY`/`BARS_PER_YEAR`.
- `src/tradebot/strategies/kelly_regime_v3.py` — the exact `scale`
  computation (`full`/`steady`/hysteresis `state` machine) this design
  replaces; **only the `scale` line changes**, `frac` and the deadband stay
  byte-identical per R-62.
- `src/tradebot/strategies/kelly_regime_v4.py` — confirms v4 is v3 with
  the `(20,40,80)` anchor ladder; nothing here needs to change.
- `src/tradebot/strategies/universal_kelly.py` (L-11) — read to confirm
  the non-duplication argument in §1 by inspection (Cover's grid-integration
  mixture, not a gradient step) before writing any ONS code.
- `experiments/r170_shared.py` — the pattern for a frozen, read-only,
  operator-authored shared pre-registration module (causal-truncation probe
  convention, disclosed-deviation-before-freeze convention, feature/kill-switch
  structure) that both branches should import rather than duplicate.
- `scripts/experiment.py` — `ev()` / `OOS_START` / `compare()` harness;
  `ev(MyVariant(), end="2020-12-31")` for inner-train, `start="2021-01-01",
  end="2022-12-31"` for inner-validation, per ROUTINE.md Step 3's split.
- `src/tradebot/inference.py` — `stationary_bootstrap_indices` (the exact
  function R-20's ±0.2 floor and this design's promotion bar are computed
  with) and `probabilistic_sharpe_ratio`, for the paired-bootstrap ΔSharpe
  CI both branches must report.
- `scripts/build_bitfinex_dataset.py`, `scripts/fee_study.py`,
  `scripts/stress_test.py` / `scripts/beta_test.py` — the falsification
  instruments named in §3.
- `tests/test_causality_strict.py` — must stay green; the ONS update's
  `.shift(1)`-equivalent lag (using only bar `t-1` and earlier for `b_t`)
  is exactly the pattern this suite checks for.
