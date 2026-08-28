# R-174 pre-registration (08-28) — asymmetric sequential-evidence gate on `kelly_regime_v4`'s exposure INCREASES

## Direction, one sentence

`kelly_regime_v4` re-sizes the instant `frac * scale` (its desired exposure)
moves more than a 10% deadband from the current position, in *either*
direction, with no notion of how much statistical evidence that move
deserves; gate only the bars where the move would **increase** exposure
behind an anytime-valid sequential test of "has the recent return process
actually shown enough evidence of positive drift to justify taking more
risk", while leaving every **decrease** ungated and immediate, exactly as
today.

## Step 0 recap

`git fetch --unshallow` + `git rev-parse HEAD origin/main` confirmed
`HEAD == origin/main` (no in-flight work). `ls experiments/*_shared.py |
sort -n | tail -3` showed R-173 as the newest round, with a section B entry
(`grep -c "R-173" docs/LEDGER.md` = 10) and R-172 likewise (13). Step 0b's
saturation count (`awk` over section E) = **1** consecutive null pass since
R-173's own dispatch — "0–2: normal", a fresh direction is warranted. The
backlog grep (`awk`/`grep` over section D) returns the same single live row,
**B-48** (a documentation/formatting instrument fix, not a strategy-research
item) plus the four already-inactionable rows (B-06 de-ranked, B-09 LOW,
B-17 PARTIAL, B-28 blocked on data this repo does not have) — so an
off-backlog literature-prompted round is the right shape, the same
convention R-160 through R-173 used.

A background research sub-agent surveyed `docs/LEDGER.md` sections B
(rounds R-88 through R-173) and C for closed families, then proposed three
candidates. Its strongest recommendation (a structural-break diagnostic on
ETH's Sept-2022 PoS-transition date) is a measurement round, not a
strategy-improvement round, and does not naturally split into a
"conservative vs novel implementation" pair — a poor fit for this session's
brief (propose an improvement, dispatch two implementation branches,
measure, promote the winner). This entry instead pursues the agent's
second candidate, sharpened using **R-160's own explicit "next step" line**
(quoted below), which does fit that brief.

## Which constraint this attacks: ERR (primary), SIZE (secondarily, since it changes the sizing decision's timing)

No error control exists anywhere in `kelly_regime_v4`'s re-sizing decision:
`apply_deadband` (`kelly_regime_v3.py`'s inlined loop; factored out as
`experiments/r102_shared.py:apply_deadband`) moves the position to
`frac*scale` the instant that value differs from the current position by
more than 10%, symmetrically, whether the move increases or decreases risk,
with no notion of how often a move of that size would occur under pure
noise. R-160 built the first ERR-axis gate on this architecture's *timing*
(online-FDR on the vote's own flip decisions) and closed NEGATIVE, but its
own entry says explicitly:

> "Next step: this specific family (online-FDR gate on vote timing) is
> closed; a future ERR-axis attempt on this architecture should look at
> **gating the SCALE/sizing decision rather than the vote's direction**,
> which no prior ERR-axis round (including this one) has touched."

R-161 and R-167 did subsequently touch SCALE (RCPS/CRC calibrating a
**multiplicative cap on SCALE's magnitude**), but neither gates the
**timing** of a change to `frac*scale` the way R-160 gated the vote's own
flip timing — they bound how large the output can get, not whether enough
evidence exists yet to let it grow. This round is the first ERR-axis attempt
that gates *timing*, generalized from R-160's vote-only mechanism to the
whole `frac*scale` decision, and the first to make the gate **asymmetric**:
gate only increases, never decreases. The asymmetry is deliberate and
motivated directly by this project's own standing diagnosis — R-33/R-57's
finding that this strategy family's entire edge is its ability to de-risk
quickly ahead of a crash — so a symmetric gate (R-160's own shape) risks
destroying the source of the edge along with whatever noise it filters,
which is one candidate explanation for why R-160 failed. Gating only the
Kelly-overbetting-fragility side (MacLean, Thorp & Ziemba 2010: full Kelly
is fragile to estimation error, and estimation error is what an
insufficiently-evidenced *increase* risks acting on) while leaving the
tail-protection side untouched is a genuinely different bet on where the
mechanism's value would come from.

## Literature

- **Wald, A. (1945)**, "Sequential Tests of Statistical Hypotheses",
  *Annals of Mathematical Statistics* 16(2), 117-186. The Sequential
  Probability Ratio Test (SPRT): accumulate the log-likelihood ratio of two
  simple hypotheses observation-by-observation; stop and accept H1 once the
  accumulated LLR crosses `ln((1-beta)/alpha)`, accept H0 once it crosses
  `ln(beta/(1-alpha))`. Classical, textbook, exactly-controlled type-I/II
  error under the assumed model. **Conservative branch's citation.**
- **Grünwald, P., de Heide, R., & Koolen, W. (2024)**, "Safe Testing",
  *Journal of the Royal Statistical Society Series B* 86(5), 1091-1128
  (earlier arXiv:1906.07801, 2019). Unifies e-values / test martingales
  with **GROW (GROwth-rate optimal in Worst case)** e-variables, and shows
  a Bayes factor against a mixture alternative is a canonical GROW
  e-variable when the true parameter is unknown — chosen here specifically
  *because* it is the same growth-rate-optimality logic `kelly_regime`'s own
  docstring already invokes for its sizing rule (Kelly 1956, Breiman 1961),
  applied instead to the *test* that gates the rule. Ville's inequality
  (1939) gives the anytime-valid guarantee: `P_0(sup_t e_t >= 1/alpha) <=
  alpha`. **Novel branch's citation.**
- **MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2010)**, "Good and Bad
  Properties of the Kelly Criterion", *Risk* — full-Kelly is fragile to
  estimation error, already this project's own standing citation for why
  every registered `kelly_regime*` variant sizes at a FRACTION of Kelly.
  Cited here as the reason to expect an increase specifically (not a
  decrease) to be the vulnerable direction: an increase acts on a *belief*
  that conditions have improved, and that belief is exactly what estimation
  error corrupts.
- **Baur, D. G., & Dimpfl, T. (2018)**, *Economics Letters* 173 — already
  this project's citation for v3/v4's conditional-vol-target architecture
  (BTC's inverse leverage effect: the strategy's edge concentrates in
  high-volatility states it must react into QUICKLY). Cited as the reason
  the gate is deliberately asymmetric rather than a repeat of R-160's
  symmetric shape.

## Not a duplicate of

- **R-28 / R-31** (single continuous e-process martingale, retired):
  triggers a **drawdown CUT** — a decrease action — sized continuously as an
  exposure multiplier, and was retired because the cut confounded with
  exposure level (R-31). This round's e-process (novel branch) gates
  **increases only** and never touches a decrease; the mechanism, the
  direction of action, and the reason for retirement are all different.
- **R-87** (Adaptive Conformal Inference on the vote/scale dispersion,
  NEGATIVE): an online coverage-calibration wrapper around a continuous
  confidence *set*; no discrete accept/reject gate, no notion of testing an
  increase specifically.
- **R-160** (online-FDR, LORD/SAFFRON, gating vote FLIPS symmetrically,
  NEGATIVE): gates every flip of the three anchors' own discrete vote,
  whichever direction; this round gates the continuous `frac*scale`
  decision (after the vote is already combined with scale), and only the
  bars where that combined decision would increase exposure. R-160's own
  entry names this exact generalization as untried.
- **R-161 / R-167** (RCPS / CRC / Howard-style confidence sequences
  calibrating a multiplicative CAP on SCALE's magnitude, NEGATIVE): bound
  how large the output can get, evaluated new each bar with no path
  dependence; this round is not a cap at all — an increase of any size is
  eventually allowed once evidence clears, the gate controls *when*, not
  *how much*.
- **R-171** (Online Newton Step / regret-minimization learning the entire
  leverage path, NEGATIVE, closes the online-convex-optimization family):
  learns a leverage sequence to minimize aggregate regret against the best
  fixed-in-hindsight leverage, with no hypothesis-test framing and no
  asymmetry between increases and decreases. This round tests a specific
  null (no drift) before allowing one specific kind of move; it does not
  learn or optimize a path.
- **R-172** (FCR / post-selection confidence bound on the currently-active
  vote PATTERN's own historical edge): a bound on a discrete pattern's
  retrospective average return, recomputed once per bar from the pattern's
  own full history; this round accumulates evidence sequentially, forward,
  episode by episode, and its object is the raw return process, not a
  per-pattern historical average.
- Ledger-wide grep (this round, before any code) confirms zero prior hits
  for "SPRT", "sequential probability ratio", "Wald", "safe testing",
  "GROW", or "Ville" anywhere in `docs/LEDGER.md` outside this entry.

## Is it simulable here?

Yes. Both branches are pure functions of the already-committed 5-minute
OHLCV `close` column (`log(close).diff()` for returns, `v4_symmetric_vol`
for a causal per-bar standard-deviation input already used by the shipped
strategy), bar-close signals, no order book, no queue model — identical
data requirement to every prior SIZE/ERR-axis round on this architecture.

## What would make it fail, named now

1. **The gate never binds** (fewer than 3 pending episodes are ever
   resolved with a delay of >=1 bar, either branch, any alpha) — a
   relabeling of v4, not a tested mechanism (kill switch A1).
2. **Degenerate**: the gated target path is a near-exact rescale of v4's
   own (`R² >= 0.999` against `v4_target`, kill switch A2).
3. **Reproduces R-160's own failure mode**: this project's vote/scale
   architecture earns its edge specifically by reacting fast; delaying
   increases trades away more return capture (the strategy is late into
   genuine new bull regimes) than it saves in avoided false starts — the
   modal outcome for every ERR-axis attempt on this architecture to date
   (R-28/31, R-87 x2, R-104, R-105 x2, R-114 x2, R-160 x2: 0 of 9 promoted).
   Named explicitly because the *prior* on an 10th attempt succeeding,
   given this base rate, is low, and this entry says so before running
   anything rather than after.
4. **Sign-inverts on ETH** — this project's single most common failure mode
   for vote/scale-timing constructions (R-87 novel, R-109, R-114 novel,
   R-160 both branches' own noted asymmetry).
5. **The improvement, if any, sits inside the ±0.2 Sharpe noise floor**
   (R-20) with no offsetting drawdown/tail improvement — 6 of 9 prior
   ERR-axis attempts closed this way.

Noise-floor arithmetic (ROUTINE.md Step 2's own requirement, ported
directly from R-160's identical comparison — same architecture, same
inner-validation window, same paired-bootstrap machinery, so the achievable
`n` is unchanged): R-160 measured `d_sharpe` on inner-validation ranging
-0.49 to +0.24 across 16 configs and 96 cells with every `d_log_growth` CI
including zero. The ±0.2 bar is therefore a real, previously-reached
threshold on this exact comparison, not an untested aspiration.

## Pre-registered constants (frozen before any inner-validation number is read)

Computed from **inner-train only** (2017-01-01 .. 2020-12-31), the Step-3
"fit, sweep, iterate freely" resource, by code in `r174_shared.py` at import
time (not hand-copied, so a transcription error cannot silently produce a
wrong value):

- `MU1 = mean(log(close).diff())` over inner-train = **8.0855e-06** per bar
  (annualized ≈ +85.1%, BTC's own historical drift over 2017-2020) — the
  conservative branch's SPRT alternative-hypothesis drift, and the novel
  branch's mixture-prior scale `TAU = MU1`. Neither is tuned to
  inner-validation or the holdout; both are a single, principled,
  non-arbitrary anchor ("does recent behavior look like the market's own
  historical bull-regime drift"), fixed before either branch ran a single
  real-data comparison.
- `ALPHA_GRID = (0.10, 0.05, 0.20)`, `ALPHA_PRIMARY = 0.10` — matches this
  project's own repeated precedent (`Q_FCR`/`HB_DELTA` = 0.10 in
  r161_shared.py / r172_shared.py) rather than a value chosen for this
  round. `BETA = ALPHA` (symmetric type-I/II budget) for the SPRT branch.
- `SHARPE_NOISE_FLOOR = 0.2` (R-20, ROUTINE.md's own promotion bar).
- `GATE_MIN_DELAYS = 3` (A1), `R2_DEGENERACY_THRESH = 0.999` (A2) — both
  ported verbatim from R-160's own kill-switch thresholds.

## Pre-registered decision rule

Identical shape to R-160's own (already-used, already-defensible bar,
adapted to a candidate-vs-v4 comparison via `r102_shared.compare()`):

**PROMOTE-CANDIDATE** (worth carrying to the holdout) if, on
inner-validation (2021-01-01..2022-12-31), for **at least one** alpha in
`ALPHA_GRID`, on **both** markets (spot and futures_5x):

  (a) the paired bootstrap 95% CI on `d_log_growth` (candidate − v4)
      excludes zero on the positive side, **and**
  (b) `d_sharpe >= +0.2` **or** a risk-matched (`exposure_ratio` and
      `vol_ratio` both in `[0.9, 1.1]`) drawdown improvement, **and**
  (c) the same sign of improvement (on whichever of (a)/(b) fired)
      reproduces on the `eth_replication` slice — not inverted.

Any other outcome on inner-validation is **NEGATIVE** for that branch. A
branch that clears all three moves to the holdout only after the specific
alpha is frozen (no further tuning) and logged in the ledger entry before
`ev(..., start=OOS_START)` is run.

**Falsification test** (ROUTINE.md Step 2's menu of four, pre-registered):
**ETH sign-replication** — clause (c) above IS the falsification test, the
same choice R-160 made for the same reason (the standing diagnosis's
repeated warning that "does it replicate" is otherwise n=1 asset).

## Division of labor (R-89 through R-173's own convention)

`r174_shared.py` (this operator, written and frozen BEFORE either branch is
dispatched) provides: the pre-registered constants above, the causal
per-bar return/sigma inputs, and `run_asymmetric_gate` — a
statistically-neutral state machine (when a pending episode starts/ends,
how a decrease is handled, when the ordinary 10% deadband applies) that
takes an arbitrary `new_episode_state()`/`step_fn(state, r_i, sigma_i)`
pair and produces a gated target path plus the A1/A2 kill-switch counters.
Neither branch may edit `r174_shared.py` or each other's file.

- **`r174_conservative_wald_sprt.py`**: `new_episode_state`/`step_fn`
  implementing the classical Wald (1945) SPRT log-likelihood-ratio
  accumulator (H0: `mu=0`, H1: `mu=MU1`, known variance `sigma_i` per bar).
- **`r174_novel_safe_testing_gro.py`**: `new_episode_state`/`step_fn`
  implementing the Grünwald–de Heide–Koolen (2024) GROW e-variable via a
  sequential Gaussian-mixture Bayes factor (H0: `mu=0`; H1: `mu ~
  N(0, TAU^2)`), thresholded at `1/alpha` per Ville's inequality — no
  formal type-II boundary (a disclosed, structural difference from the
  conservative branch's SPRT, itself worth reporting as a methodological
  finding regardless of which branch wins).

Each branch runs `compare()` from `r102_shared` (imported via
`r174_shared`), reports configs evaluated, and must NOT read a single bar
at or after `OOS_START` (2023-01-01) — `assert_no_holdout` is called by
`compare()`/`run_slice()` on every invocation. Neither branch commits;
the operator merges and records the round once both verdicts are in.

Configs evaluated by this file: 0 (shared infrastructure only).
