# R-172 design — FCR-adjusted selective inference on `kelly_regime_v4`'s
# 3-anchor vote pattern

Written pre-implementation, pre-holdout. No backtest was run to produce
this document beyond the two one-off numeric derivations disclosed in
`experiments/r172_shared.py`'s own docstring (both computed on inner-train
only, matching R-171's convention for its own `G` derivation). No file
under `src/` was modified. No data at or after `OOS_START = 2023-01-01`
was read.

## 0. Provenance

A research sub-agent (per this session's scheduled brief: "take the best
strategy, propose an improvement direction, research it, dispatch
conservative/novel branches, measure, promote the winner") searched
`docs/LEDGER.md` section C and the last ~30 rounds of section B for
ERR-axis attempts, searched the literature for a genuinely unused
statistical framework, and proposed this direction. The operator then
independently verified the two central claims before freezing this
document: (1) `vote_gamma` is real, shipped code (`src/tradebot/strategies/
kelly_regime.py`, inherited unmodified by `kelly_regime_v3`/`v4`, `frac =
frac ** vote_gamma` when `vote_gamma != 1.0`) — confirmed by reading the
file directly; (2) the Benjamini & Yekutieli (2005) formula the design
below depends on — confidence level `1 - q*R/m` for `R` parameters
selected out of `m` considered — confirmed by web search against the
paper's own abstract/summary (JASA 100(469), 71–81), independently of the
sub-agent's report. Berk, Brown, Buja, Zhang & Zhao (2013)'s PoSI citation
(*Annals of Statistics* 41(2), 802–837, DOI 10.1214/12-AOS1077) was
likewise independently confirmed.

## 1. Direction (one sentence)

`kelly_regime_v4`'s vote is the mean of three independently-latched
binary anchor votes (20/40/80-day), so at any bar the strategy is running
on exactly one of **8 discrete anchor-agreement patterns**
(`(v20,v40,v80) ∈ {0,1}³`) — and no round in this ledger has asked
whether trusting the currently-realized pattern's own historical
forward-return edge is honest once the fact that it is *one of 8
implicitly-compared candidates* is accounted for, a textbook
post-selection / winner's-curse problem this project's error-control
(ERR) axis has never attacked from this angle.

## 2. ROUTINE.md Step 1 — the four-question filter

**1. Which constraint does it attack?** **ERR** — no error control anywhere
in the signal path. Specifically: the vote's own combinatorial pattern
selection has never been given a formal, multiplicity-corrected confidence
guarantee. Secondarily touches **SIZE** (the correction acts as a
sizing/vote-response modifier, the axis with this project's entire win
record), but the mechanism under test — False Coverage-statement Rate
(FCR) control — is an ERR-axis object: a *guarantee on interval validity
under selection*, not a sizing rule in its own right.

**2. Which ledger entries is it not a duplicate of?** The ERR axis has
been attempted **far more densely than the "2 attempts" a first grep
suggests** (`docs/LEDGER.md` section C and B, confirmed by direct read
of each round cited): R-31 (e-process, retracted by R-33 as an
exposure-level artifact), R-87 (Adaptive Conformal Inference on VOTE
confidence and SCALE dispersion, both NEGATIVE), R-104 (periodic
bootstrap / continuous PSR significance discount on VOTE), R-105
(delete-one-anchor jackknife / 5-member ensemble disagreement discount on
VOTE), R-106 (cross-model-class disagreement discount on VOTE, using a
BOCPD/Kalman/CSD/Hawkes panel), R-109/R-112/R-113/R-115/R-121/R-122/R-123
(distributional-novelty-of-market-state discount, 7 rounds across 4
architectural axes, explicitly closed by R-123 as "the fourth and final
axis of variation" for *that* construction), R-114 (duration/hazard
discount on VOTE), R-116 (cross-asset breadth-divergence discount), R-147
(James-Stein shrinkage of VOTE combination weights toward a common
target), R-160 (online false-discovery-rate control, LORD/SAFFRON, on
VOTE flip *timing* — a sequential-in-*time* multiplicity correction, not
a combinatorial-*state-space* one), R-161 (Conformal Risk Control / RCPS
tail-loss cap on SCALE output), R-167 (anytime-valid Hoeffding /
betting-confidence-sequence tail-loss cap on SCALE output, the same
architecture as R-161 with a different guarantee family).

This design is not a duplicate of any of them because none treats the
vote's **8-way pattern identity as a finite candidate space requiring
simultaneous-validity correction**:
- R-104/106 test the *aggregate* vote's significance/agreement, never
  distinguishing *which* of the 8 patterns is active — a single running
  statistic, not 8 competing ones, so there is no selection effect to
  correct.
- R-105's jackknife measures *sensitivity to dropping one anchor*, a
  leave-one-out perturbation, not a confidence interval for the realized
  pattern's own historical edge.
- R-109–R-123's novelty family scores *market-state distance from
  historical precedent* (Mahalanobis/kNN/path-signature), a continuous
  feature of price action unrelated to which of the 3 anchors currently
  agree.
- R-114 scores *how long* the current regime has persisted, not *which*
  of 8 combinatorial patterns is realized.
- R-147's shrinkage is *parametric* (toward a common Bayesian prior
  target) and acts on the vote's real-valued *weights*, not a
  nonparametric selective confidence interval for one of 8 discrete
  states.
- R-160's online-FDR controls error across a *sequence of flip events
  through time*; this design controls error across *8 simultaneously
  candidate states*, a different multiplicity structure entirely (BY
  2005's own framing is explicitly about a *fixed candidate set*, not a
  sequential stream — that is Foster & Stine 2008's alpha-investing/LORD
  family, which is what R-160 already used).
- R-161/R-167 never touch VOTE at all — they cap SCALE's *output*
  multiplicatively from the outside; this design corrects an *input* to
  the vote/gamma construction itself and uses a completely different
  guarantee family (finite-candidate simultaneous coverage, not a
  tail-loss risk-control set).
- R-87's conformal wrapper tracks *one* running coverage target (marginal
  validity over time for one estimator), not simultaneous validity across
  8 named candidates — the mathematical object FCR controls (`E[false
  coverage / |selected|]` over a *fixed candidate set*) does not exist in
  R-87's construction at all.

Also not a duplicate of **R-171** (Online Newton Step on SCALE, N≈3 axis,
closed this session): different constraint (ERR vs. N≈3), different slot
(VOTE vs. SCALE), and this design deliberately avoids **both** of R-171's
own named traps — it adds no tail-risk/drawdown penalty to a regret loss
(there is no regret-minimization anywhere here) and imposes no tighter
hard-coded leverage cap (`max_leverage` is untouched; the mechanism only
ever modulates the *vote*, never the leverage domain).

`grep -in "benjamini\|yekutieli\|post-selection\|selective inference\|berk.*brown.*buja\|false coverage" docs/LEDGER.md`
returns zero hits before this entry.

**3. Is it simulable here?** Yes, with zero new data. The three anchor
votes are already computed causally in `kelly_regime.py`/`kelly_regime_v3.py`
and reproduced as standalone functions in `experiments/r102_shared.py`
(`_latched_anchor_vote`, `v4_vote_frac`). The forward-return label used to
build each pattern's historical bucket is derived from the same OHLCV
close series already loaded by every prior round; the only new
requirement is a causal embargo (§3 below), not a new data source.

**4. What would make it fail, named now:**
- **(a) BTC/ETH sign inversion.** Per this project's overwhelming
  track record on this exact shape (R-87, R-104, R-105's ensemble
  branch, all 7 of the R-109 family, R-116 — 8+ prior instances), the
  most likely failure is that a per-pattern historical edge calibrated
  on BTC's single 2017–2022 supercycle sign-inverts on ETH.
- **(b) Sample size too thin to ever bind.** Splitting an already-short
  history 8 ways can leave every pattern's bucket too small for any
  FCR-corrected bound to exclude zero, reproducing R-104's PSR-branch and
  R-161/R-167's Hoeffding-floor inertness (a bound that is *always* too
  wide to bind is observationally identical to "no correction at all").
  §4 computes the `n` this implies before any code is run, per
  ROUTINE.md's own instruction.
- **(c) Relabeling.** The gate/modulation collapses to a near-exact
  rescale of v4's own unmodified target (KS-B below), i.e. it changes
  nothing that v4 didn't already imply.

## 3. Mechanism, citations, and the shared statistical construction

**Mechanism, one sentence.** Because the vote's `frac` value is realized
from one of 8 candidate `(v20,v40,v80)` patterns rather than a single
pre-specified hypothesis, a naive historical mean of "this pattern's own
forward return" is optimistically biased by the implicit comparison
against the other 7; Benjamini & Yekutieli's (2005) False
Coverage-statement Rate control constructs a simultaneously-valid
confidence bound for exactly the parameter that was selected (using
`R=1` of `m=8` in their own `1 - q·R/m` formula, since exactly one
pattern is realized and acted on per bar), and using that corrected bound
— rather than the raw historical mean — to modulate the vote should
suppress transient, look-good-in-hindsight patterns without ever
retuning the vote's own direction.

**Citations.**
- Benjamini, Y. & Yekutieli, D. (2005), "False Discovery Rate–Adjusted
  Multiple Confidence Intervals for Selected Parameters," *Journal of
  the American Statistical Association* 100(469), 71–81.
- Berk, R. A., Brown, L. D., Buja, A., Zhang, K. & Zhao, L. (2013),
  "Valid Post-Selection Inference," *Annals of Statistics* 41(2),
  802–837 — cited for the "simultaneous validity under any selection
  rule" framing that motivates using the *full* candidate count `m=8`
  (not merely the patterns observed so far) as the correction's
  denominator, the conservative (Bonferroni-style) choice both papers
  agree is valid regardless of which selection rule is actually used.

### 3.1 The shared statistical primitive (both branches use this identically)

Both variants are built from **one shared, frozen primitive**
(`experiments/r172_shared.py`, described fully there): for each calendar
day `d`, using the day's own realized 3-anchor pattern `p(d) ∈ {0,...,7}`
(bit `i` = whether anchor `horizons[i]` is bullish, `i=0→20d, i=1→40d,
i=2→80d`) and a forward horizon `H = 5` trading days (a priori, not swept
— the primary configuration; a 3-point robustness bracket is required by
§5 clause 3), maintain a **causally-embargoed, expanding historical
bucket** of `H`-day-forward simple returns realized while pattern `p` was
previously active: only labels for days `d' ≤ d - 1 - H` (i.e. whose own
`H`-day-forward window has fully resolved *before* today) ever enter the
bucket for `p(d)`. From that bucket, compute a one-sided lower confidence
bound (and, for the novel branch, a two-sided width) at the
**FCR-corrected level** `1 - q/m` with `q = 0.10` (matching this
project's own `HB_DELTA=0.10` precedent in `r161_shared.py`, not fitted
to this round's data) and `m = 8` (the full candidate space, fixed
regardless of how many patterns have actually been observed by day `d`,
per Berk et al.'s "valid under any selection rule" argument — using the
*observed* pattern count instead would let the correction get weaker
simply because rare patterns hadn't shown up yet, the wrong direction for
a conservative bound).

The interval uses a **normal approximation** (Acklam's rational
inverse-normal-CDF, self-tested against known quantiles in the module —
see its docstring), not a Student-`t`, and requires a bucket of at least
`MIN_N = 30` resolved observations before it reports anything — below
that, the day is treated as **no evidence yet**, defaulting to v4's own
unmodified behavior (both branches; the alternative, defaulting to
"untrusted," would spend the entire early-history warmup artificially
flat or artificially convex, an exposure-matching artifact of exactly the
kind R-33 warns about, not a property of the mechanism). This is a
disclosed simplification (no scipy in this project's dependency set,
matching `r161_shared.py`'s own disclosed Hoeffding-not-Bentkus choice)
and is checked against known reference quantiles (`Φ⁻¹(0.975) =
1.959964`, etc.) before either branch runs.

## 4. Noise-floor arithmetic, done before any code

Per ROUTINE.md Step 2, and following R-171's own precedent exactly (the
comparison type — paired stationary block bootstrap of inner-train /
inner-validation ΔSharpe — is unchanged from the comparison R-20 already
calibrated): **R-20's ±0.2 Sharpe bar is reused, not reinvented.** At
v4's own `target_vol = 0.55`, a 0.2 annualized-Sharpe difference implies
an annualized mean-return difference of `0.2 × 0.55 ≈ 11 pts/yr` —
larger than nearly every net-of-fee edge that has ever survived Step 4 in
this ledger, which is the correct scale for a bar meant to be hard to
clear.

**The sample-size question specific to this design:** splitting
~2,192 pre-holdout days across 8 patterns gives, under a uniform
allocation, ~274 days/pattern — but the allocation is **not** uniform (a
persistent bull/bear regime keeps 1–2 patterns dominant for years at a
stretch), so several of the 8 patterns will realistically see far fewer
than 30 resolved 5-day-forward observations for long stretches of
inner-train, and the `MIN_N=30` floor means those patterns' days default
to "no evidence yet" (v4-identical behavior) for as long as that holds.
**This is disclosed now, before any run, as the single most likely
proximate cause of a null result** (failure mode (b) in §2.4) — the
design is not expected to bind uniformly across all 8 patterns, and a
report that shows it binding narrowly (on the 2–3 most common patterns
only) is a partial-information result worth stating plainly, not
grounds to loosen `MIN_N` after seeing it.

**Base rate, stated honestly before any number is read:** of the ~12
prior VOTE-slot ERR/discount-family attempts with a reported
inner-validation ΔSharpe (R-87, R-104, R-105, R-106, the 7-round R-109
family, R-114, R-116, R-147, R-160), **zero** have cleared the promotion
bar. The prior probability this one does is low, and both falsification
tests below are chosen to fail fast and cheaply.

## 5. Variants

Both variants touch **only** the vote side of `desired = frac * scale`
(never `scale`, per R-62's isolation discipline — VOTE carries v4's
signature, SCALE does not, confirmed independently 4 times, most recently
R-171). Both are one multiplicative factor inserted **pre-deadband**
(the R-109-family's own established insertion point), computed from the
shared FCR primitive in §3.1.

### Conservative — literal binary FCR gate on the vote

**Mechanism, one sentence.** Test, at the FCR-corrected level, the null
"the currently-active pattern's true mean forward edge is ≤ 0"; trust the
vote fully when the null is rejected on the positive side (`LCB_p(t) >
0`), force the position flat otherwise — the most literal possible use of
a one-sided confidence bound as a hypothesis test, with no free
discount-shape parameter.

`gated_frac(t) = frac(t)` if `LCB_{p(t)}(t) > 0` (or `n_used < MIN_N`,
the disclosed no-evidence default), **else** `0.0`. `desired = gated_frac
* scale`, then v4's own unmodified 10% deadband.

**Falsification test.** ETH same-sign replication (`scripts/
build_bitfinex_dataset.py` / `load_eth()`, this ERR family's own standing
convention since R-47/R-55). **Exact kill outcome:** if the sign of
ΔSharpe (candidate vs. v4) on BTC inner-validation disagrees with the
sign of ΔSharpe on ETH inner-validation, the branch is killed regardless
of the BTC number — appropriate here specifically because per-pattern
historical means are exactly the kind of single-asset overfit this test
exists to catch.

### Novel — FCR interval width modulates the existing `vote_gamma` exponent

**Mechanism, one sentence.** Rather than a hard gate (the architecture
R-109's 7-round family already explored to exhaustion for a *different*
statistical basis), feed the FCR-corrected interval's own **width** — how
imprecisely pinned-down the active pattern's edge is, once multiplicity is
accounted for — into `kelly_regime.py`'s own already-shipped, currently
dormant (v4 uses the default `vote_gamma=1.0`) convex-response exponent,
so a per-bar `gamma_t` replaces the fixed scalar: patterns whose corrected
interval is wide get shrunk more aggressively toward zero (for the
partial-agreement `frac ∈ {1/3, 2/3}` states only — `frac ∈ {0,1}` is a
fixed point of any exponent), while patterns with a tight, positively-
signed corrected interval are treated near-linearly.

`gamma_t = 1 + k · clip(width_{p(t)}(t) / width_ref, 0, 2)`, `k = 1.0`
(PRIMARY, not swept — the 3-point robustness bracket is `k ∈
{0.5, 1.0, 2.0}`), `width_ref` = the median finite two-sided FCR width
observed **on inner-train only** (2017–2020, frozen once, never
recomputed on inner-validation/ETH/holdout — the same "derive once on
inner-train, reuse unchanged" convention R-171 used for its own `G`).
`frac_gamma(t) = frac(t) ** gamma_t`; `desired = frac_gamma * scale`, then
v4's own deadband. When `n_used < MIN_N`, `gamma_t = 1.0` (v4-identical),
the same no-evidence default as the conservative branch.

This enters through a genuinely different slot (an exponent inside the
vote's own construction, not a post-hoc multiplicative discount) than
every closed R-109-family round, and uses the interval's **width**, not
its sign — a materially different statistic from the conservative
branch's binary test, not merely a looser threshold on the same number.

**Falsification test.** 0.40% taker-fee survival (`scripts/fee_study.py`
/ `fee_at()`). **Exact kill outcome:** the branch is killed if the sign of
ΔSharpe reverses between the standard fee tier and the 0.40% tier, on
either market — a per-bar-varying exponent risks raising effective
turnover through the 10% deadband before it can absorb the extra
churn, the identical mechanism R-64/R-65 measured for other
`kelly_regime_v4` factors.

## 6. Kill switches (checked before any promotion-bar comparison)

- **KS-A, non-triviality.** The fraction of inner-train bars where the
  candidate's factor differs from v4's own baseline (conservative:
  `gated_frac != frac`; novel: `gamma_t != 1.0`) must be **≥ 2%**
  (`GATE_MIN_BINDING_FRACTION`, `r161_shared.py`'s own convention) — below
  that, the mechanism never binds and any downstream comparison number is
  vacuous, reported as such rather than as a negative result on merit.
- **KS-B, relabeling.** `R²` of the candidate's final target path against
  v4's own raw `frac*scale` path must be **< 0.95** (`R2_KILL_THRESH`,
  `r161_shared.py`'s own convention) — otherwise the candidate is a
  near-exact rescale of v4, not a genuinely different signal.
- **KS-C, sample-size disclosure (not a kill switch, a mandatory report).**
  For each of the 8 patterns, report the resolved bucket size `n_p`
  reached by `INNER_TRAIN_END` and by `INNER_VAL_END`. Per §4's own
  prediction, at least some patterns are expected to stay below `MIN_N`
  for extended periods; this must be stated in the results write-up
  before any comparison number is read as decisive.

## 7. Pre-registered decision rule (frozen before either branch runs)

**PROMOTE a variant only if ALL of:**

1. ΔSharpe (variant vs. `kelly_regime_v4`) on inner-validation ≥ **+0.2**
   on **both** BTC and ETH (R-20's own noise floor, reused per §4), **or**
   a matched-exposure (`exposure_ratio` and `vol_ratio` both in `[0.9,
   1.1]`, R-33's convention) max-drawdown reduction ≥ **5 percentage
   points** on both markets (v4's own registered improvement over v3,
   reused as the bar per R-171's identical precedent).
2. Survives its variant's pre-registered falsification test (§5).
3. **Plateau, not peak.** Conservative: results hold across the 3-point
   `q ∈ {0.05, 0.10, 0.20}` bracket around the chosen `q=0.10` (a
   robustness check on a structural constant, not a free-fitted
   parameter). Novel: results hold across the 3-point `k ∈ {0.5, 1.0,
   2.0}` bracket around the chosen `k=1.0`.
4. Sign does not reverse at the 0.40% taker fee tier
   (`scripts/fee_study.py`).

**Any other outcome is NEGATIVE**, including partial passes — a
fall-through is reported as a fall-through, per ROUTINE.md's own rule
against reaching for the nearest-looking label.

## 8. Files the implementer needs to read

- `src/tradebot/strategies/kelly_regime.py` — `frac`/`vol` computation,
  `vote_gamma`'s exact convex-response formula (`frac ** vote_gamma`),
  the `deadband` trade filter.
- `src/tradebot/strategies/kelly_regime_v3.py`, `kelly_regime_v4.py` —
  confirm v4 is v3 on a `(20,40,80)` anchor ladder; neither `scale` nor
  the deadband is touched by this round.
- `experiments/r102_shared.py` — `_latched_anchor_vote`, `v4_vote_frac`,
  `v4_scale`, `apply_deadband`, `v4_raw_desired`, `v4_target`, `compare`,
  `TargetStrategy`, `run_slice`, `paired_diff`, `print_rows`, `fee_at`,
  `r_squared`, `causal_truncation_probe_series`, `load_btc`, `load_eth`,
  `assert_no_holdout`.
- `experiments/r161_shared.py` — `daily_close`, `daily_log_return`,
  `daily_last_of`, `broadcast_daily_lambda`, `GATE_MIN_BINDING_FRACTION`,
  `CONST_CAP_R2_THRESH`/`R2_KILL_THRESH`-style conventions, `FEE_TIER`,
  `SHARPE_NOISE_FLOOR` — the daily-resolution / causal-broadcast pattern
  this round's shared module reuses directly rather than reimplementing.
- `experiments/r172_shared.py` — this round's own frozen primitive:
  `anchor_pattern`, `fcr_lower_bounds` (the causal, embargoed,
  FCR-corrected per-pattern bucket construction), `norm_ppf`, kill-switch
  helpers. **Neither branch may edit this file or each other's file.**
- `scripts/build_bitfinex_dataset.py`, `scripts/fee_study.py` — the two
  falsification instruments named in §5.
- `tests/test_causality_strict.py` — must stay green; both branches'
  factor must be verifiable causal by `causal_truncation_probe_series`
  end-to-end through their own strategy wiring, not just on the shared
  primitive in isolation.
