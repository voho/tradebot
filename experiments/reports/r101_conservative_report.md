# R-101 (CONSERVATIVE branch) — static jackknife confidence multiplier for `kelly_regime_v4`

**Date:** 2026-08-23
**Idea in one sentence:** Freeze a third multiplicative factor `conf = clip(1 - k*CV, conf_floor, 1.0)` onto `kelly_regime_v4`'s `desired = frac * scale`, where `CV` is the coefficient of variation of a delete-one-group jackknife (Quenouille 1949; Tukey 1958; Efron 1979) of the regime vote's realized log-growth edge across this project's six standard stress episodes, computed once on inner-train and applied as one constant for the strategy's whole life.

**Constraint attacked:** N≈3 — turns the qualitative "effective sample size ≈3" diagnosis into a literal dispersion number, and gates exposure on how much that number should be trusted.

**Verdict: NEGATIVE — Step-0 kill switch failure (KS-B).** Per this branch's own pre-registration, the round stopped immediately after the two kill switches; no Sharpe/backtest sweep was run, inner-validation was touched only in the way KS-B itself requires (see below), and the 2023+ holdout was never read.

---

## Repo-state audit (read this before anything else)

This round's pre-registration was written referencing a ~100-round project
history — it cites R-33, R-53, R-62, R-73, R-93, R-97, R-99, R-100 and a
file `experiments/r99_shared.py` defining `STRESS_EPISODES`. **None of
that exists in this repository.** `docs/LEDGER.md` stops at R-28 (`git
log` shows 40 commits total, the last ledger-writing commit titled
"Merge main; renumber the e-process round to R-28"). There is no
`experiments/r99_shared.py`, no `STRESS_EPISODES` constant anywhere in
the tree, and no ledger row numbered above 28. There is also no
`data/ethusd_coinbase_spot_5m.csv.gz` — the only committed ETH file is
`data/ethusd_bitfinex_5m.csv.gz` (2016-03-09 → 2019-12-31), the same file
R-17 used.

This did not block the assignment: the pre-registration gives the six
`STRESS_EPISODES` dates literally in its own text, so they were copied
verbatim into `experiments/r101_conservative_jackknife_static.py` rather
than imported from a nonexistent file, and the ETH falsification test
(never reached — see below) would have used the real committed Bitfinex
BTC/ETH pair on their shared window, matching R-17. Stating this plainly
rather than silently improvising or pretending the citations checked out.

---

## Mechanism, precisely as implemented

`kelly_regime_v4`'s vote/scale product becomes `desired = frac * scale *
conf`, with `conf` applied **inside** the existing deadband loop (before
the `abs(desired - pos) > deadband` latch), not pasted onto v4's own
output afterward — this matters because the deadband is an *additive*
threshold, so a multiplicative rescale is not exactly linear once it
passes through the latch.

`conf`'s only free inputs are `k` and `conf_floor`. `CV` itself is not a
free parameter — it is measured once from data:

1. Run `kelly_regime_v4` and a control strategy (`frac` forced to 1.0,
   i.e. full exposure always, no vote at all — isolates the pure
   volatility-targeting `scale` term) over inner-train (2017-01-01 →
   2020-12-31), BTC spot, 0.10% fee.
2. `edge_t = log-return(v4)_t − log-return(control)_t`, per 5-minute bar
   — the incremental log-growth the vote itself contributes, as a time
   series.
3. For each of the six pre-registered stress episodes whose ±60-day
   window overlaps inner-train, compute the mean of `edge_t` with that
   window excluded — a leave-one-out (delete-one-group jackknife)
   estimate of the vote's edge.
4. `CV = std(LOO estimates, ddof=1) / mean(LOO estimates)`.

This operationalization of "the vote's realized log-growth edge" was a
judgment call the pre-registration left implicit; a skeptic re-deriving
this should know it is `log-return(v4) − log-return(scale-only control)`,
not, e.g., a raw growth rate or a Sharpe-based edge.

**Correction to the pre-registration's own arithmetic.** The
pre-registration states inner-train "covers only the first TWO of the six
episodes, 2018-01 and 2018-12." That is not correct: inner-train runs
through **2020-12-31**, and the third episode, **2020-03-12**, together
with its ±60-day window (2020-01-12 → 2020-05-11), falls entirely inside
inner-train. The jackknife below therefore has **n=3** leave-one-out
estimates on inner-train, not n=2. Stated plainly, as the pre-registration
itself asked for regarding the n=2 case — this is the honest number, and
it is a fact worth recording about the pre-registration itself, not just
about the mechanism.

---

## KS-A — real dispersion?

Episodes usable on inner-train (±60-day window fully inside 2017-01-01 →
2020-12-31): **3 of 6** — `2018-01-17`, `2018-12-15`, `2020-03-12`
(corrected count, see above).

| episode left out | LOO mean edge/bar |
|---|---|
| 2018-01-17 | 7.764e-07 |
| 2018-12-15 | 2.284e-08 |
| 2020-03-12 | 4.513e-07 |

Grand mean edge/bar over inner-train: **8.435e-07**. Edge std/bar:
**9.200e-04** (≈1,000× the mean — the per-bar edge is noise-dominated,
consistent with everything else this project has found about the vote's
signal-to-noise ratio).

**CV = 0.9067.**

**KS-A: PASS** (0.9067 ≥ 0.10), but with a caveat that belongs in the
record, not buried: the grand mean is three orders of magnitude smaller
than the bar-level noise, so the three LOO means being far apart in
*relative* terms (CV) is close to guaranteed once the mean is that near
zero — a small denominator inflates CV almost automatically. This is
real dispersion in the literal sense the pre-registration asked to
measure, and it is also consistent with "the vote's edge on inner-train
is statistically indistinguishable from zero," which is a different (and
arguably more important) fact than "the edge estimate is unstable." Both
readings point the same direction: n=3 is not enough to trust a specific
number here, which is the whole diagnosis this mechanism exists to act
on.

---

## KS-B — not a flat rescale?

A-priori config (grid midpoint, named before any performance number was
looked at): **k=1.0, conf_floor=0.5** → `conf = clip(1 - 1.0×0.9067, 0.5,
1.0) = clip(0.0933, 0.5, 1.0) = 0.5000`.

R² of the candidate's exposure path (`frac*scale*conf`, conf=0.5) against
v4's own unmodified exposure path (`frac*scale`), over inner-train ∪
inner-validation (2017-01-01 → 2022-12-31, BTC spot, 630,721 bars):

**R² = 0.9736.**

mean|target_v4 − target_cand| = 0.2001 (mean v4 exposure 0.3837, mean
candidate exposure 0.1854 — almost exactly half, as expected from
conf=0.5).

**KS-B: FAIL** (0.9736 ≥ 0.95 is required to pass — the rule is R² < 0.95
passes; R² ≥ 0.95 fails). This is the same exposure-collapse artifact
R-33 diagnosed and that has now killed this SIZE-axis attempt too: a
frozen scalar multiplier inside a deadband loop is, to first order,
exactly a rescale of the original signal — the deadband's additive
threshold perturbs the *timing* of position changes slightly, but not
enough to keep R² below the bar. This was architecturally close to
inevitable for the conservative/static reading specifically — a
**time-varying** `conf` (the "novel" reading this branch was
deliberately not assigned) would not face the same structural problem,
since a moving multiplier is not collinear with the thing it multiplies.

---

## Decision

**Per pre-registration: "If EITHER kill switch fails: STOP."** KS-B
failed (KS-A passed). No Sharpe/backtest sweep of the 10-config grid was
run. Inner-validation was touched only in the way KS-B itself is defined
to use it (R² of the exposure path over inner-train ∪ inner-validation)
— no performance numbers (Sharpe, drawdown, growth) were computed on
inner-validation or anywhere else, and the ETH falsification test, the
plateau check, and the fee-tier comparisons in the standard battery were
never run because the pre-registration gates all of them behind both
kill switches passing. The 2023-01-01+ holdout was not read.

**further_work = False.** The pre-registration's own bar requires KS-A
**and** KS-B **and** the ETH falsification **and** a noise-floor-clearing
improvement, all before recommending the operator run the holdout. KS-B
alone is enough to fail that bar; the round stops here on this branch's
own terms.

## Configurations evaluated (exact count, for the trials ledger)

- **2** configurations of the candidate mechanism `ConservativeStaticJackknifeV4` were actually backtested:
  1. `conf=1.0` (k=0 identity harness sanity check) — confirmed **bit-for-bit
     identical** to `kelly_regime_v4`'s target array and final balance on
     inner-train (max abs diff = 0.0). Correctness check, not a
     performance trial.
  2. `conf=0.5` (k=1.0, conf_floor=0.5, the a-priori KS-B config) — used
     only for the R² computation above, never scored on Sharpe/drawdown.
- **2** supporting diagnostic backtests (unmodified `kelly_regime_v4`,
  and the `frac≡1` scale-only control) were run on inner-train to build
  the six-episode jackknife.
- The pre-registered 9-cell grid (`k∈{0.5,1.0,2.0} × conf_floor∈{0.3,0.5,0.7}`)
  plus its own k=0 sanity row — 10 configs total — was **not** run. Only
  1 of those 10 cells (the a-priori one) was ever instantiated, and only
  for the KS-B check, never for Sharpe.
- **Total backtests run this session: 6** (2 candidate configs + 2
  diagnostic controls + 2 reference `kelly_regime_v4` runs used inside
  KS-A/KS-B), all on data ≤ 2022-12-31. No Sharpe-based selection occurred,
  so no deflated-Sharpe calculation applies to this branch — the negative
  result is a structural one (R² collinearity), not a selected-best-of-N
  result.

## Falsification test (pre-registered, not reached)

ETH falsification (Bitfinex BTC/ETH, shared window) was pre-registered
as this branch's mechanism-level falsification test but was never run:
the pre-registration gates it behind both kill switches, and KS-B failed
first. Not tested is not evidence either way about ETH.

## Plateau check (not reached)

Not applicable — no performance sweep was run to have a neighborhood to
assess.

## Lookahead / causality audit

- `grep -n "202[3-9]"` on `experiments/r101_conservative_jackknife_static.py`
  finds exactly one hit: a prose comment ("this file never calls it on
  2023+ data") — not a date literal used as a bound or a data read.
- `conf` is baked into the strategy as a plain Python float at
  construction time (`ConservativeStaticJackknifeV4(conf=0.5)`), computed
  once, entirely off inner-train (≤2020-12-31) data, outside of and prior
  to any call to `prepare()`. `prepare()` never derives `conf` from the
  frame it receives, so applying the strategy to any period cannot leak
  future information through `conf` even in principle.
- `pytest tests/test_causality_strict.py -q`: **51 passed.**
- `pytest -q` (full suite): **391 passed, 0 failed** (215s).
- The 2023-01-01+ holdout was never read by this branch, for BTC or ETH,
  at any point. Holdout counter: unchanged.

## Lesson

The cheapest, most conservative reading of "confidence-weight the vote"
— a single frozen scalar, computed once and applied forever — is
architecturally almost indistinguishable from just re-picking a smaller
`target_vol`: a constant multiplier inside a threshold-latched exposure
signal is a rescale to first order, and the deadband's additive (not
multiplicative) threshold is not enough perturbation to break that
collinearity below R²=0.95. If the jackknife-CV mechanism is worth
revisiting, it needs the property this branch was specifically assigned
*not* to have: `conf` varying over time (episode-conditional or
rolling), which is precisely the difference between "a differently-tuned
`target_vol`" and "a genuinely new multiplicative factor." That reading
is out of scope for this branch by design — it belongs to whatever
sibling branch in this round was assigned the time-varying / "novel"
version of the same idea, if one was run in parallel.

## Files

- `experiments/r101_conservative_jackknife_static.py` — the mechanism,
  the jackknife, both kill switches, and the (unreached) battery harness.
- `experiments/reports/r101_conservative_report.md` — this report.

No other file in the repository was modified. Nothing was committed.
