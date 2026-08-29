# R-181 pre-registration (08-29) — a shorter-resolution-time evidence
statistic for `kelly_regime_v4`'s asymmetric ERR-axis gate

Written and committed by the operator BEFORE either branch is dispatched,
per ROUTINE.md's Step 0 collision-avoidance convention.

## Step 1 — selection

**Idea, one sentence:** R-174 built and validated a statistically-neutral
gate (`run_asymmetric_gate`, reused unchanged here) that delays
`kelly_regime_v4`'s exposure-*increasing* re-targets behind a sequential
evidence test, and found both a classical SPRT and a modern GROW e-value
unreachable within an episode's lifetime because BTC's raw 5-minute
drift/volatility ratio is too small; its own closing "next step" line
named the fix ("a statistic with a shorter natural resolution time") and
this round implements it, two ways.

1. **Which constraint does it attack?** **ERR** (primary) — no error
   control exists anywhere in v4's `apply_deadband` re-sizing decision;
   this is the same axis as R-28/R-31/R-87/R-104/R-105/R-114/R-160/R-174,
   attacked here via a different *statistic*, not a different test
   architecture (the architecture, `run_asymmetric_gate`, is reused
   verbatim from R-174 and is not itself under test again).
2. **Not a duplicate of:** R-174 (identical gate architecture, per-bar
   drift/vol statistic and a fixed-TAU point alternative — this round's
   entire reason to exist is changing that one input); R-160 (online-FDR
   on vote FLIPS, symmetric, no coarser statistic, no sequential
   evidence accumulation); R-161/R-167 (RCPS/CRC caps SCALE's magnitude
   each bar independently, no path-dependent evidence, no gate on
   *timing*); R-165 (smooths SCALE's *value*, not a hypothesis test on
   *when* to grant an increase); R-172 (FCR/PoSI on the 8-way vote
   pattern's retrospective average, not a forward sequential
   accumulator — though this round explicitly reuses R-172's own
   causal-daily-broadcast discipline, having found it the hard way);
   R-175 (MSM volatility cascade feeding v4's own hysteresis SCALE
   input directly — this round's TSRV estimator feeds a *gate*'s
   internal Gaussian model, never touches v4's own shipped `scale`);
   R-179/R-180 (meta-labeling classifiers predicting bet *quality* from
   external features — this round predicts nothing and uses no feature
   beyond price; it only decides *when* enough evidence exists for an
   already-decided increase).
3. **Simulable here?** Yes. Both branches are causal transforms of
   `data/`'s already-committed BTC and ETH 5-minute OHLCV closes — no
   order book, no new fetch, no live-equity dependency (`TargetStrategy`,
   R-102's own pattern).
4. **What would make it fail?** Named per branch below, before any
   backtest runs.

## Step 2 — design and citations

- **Wald, A. (1945), "Sequential Tests of Statistical Hypotheses,"
  *Annals of Mathematical Statistics* 16(2), 117–186** — SPRT, reused
  (as R-174 did) for the conservative branch.
- **Zhang, L., Mykland, P.A. & Aït-Sahalia, Y. (2005), "A Tale of Two
  Time Scales: Determining Integrated Volatility With Noisy
  High-Frequency Data," *JASA* 100(472), 1394–1411** — the two-time-scale
  realized-variance (TSRV) bias correction; implemented once, in
  `experiments/r181_shared.py::two_scale_realized_variance`, self-tested
  against synthetic noisy data (TSRV's mean absolute error against the
  known true variance is verified strictly lower than the naive
  full-resolution estimator's).
- **Aït-Sahalia, Y., Mykland, P.A. & Zhang, L. (2005), "How Often to
  Sample a Continuous-Time Process in the Presence of Market
  Microstructure Noise," *Review of Financial Studies* 18(2), 351–416**
  — establishes that naive high-frequency realized variance is biased
  upward by noise unless corrected; the load-bearing claim behind the
  novel branch's own precheck 1.
- **Grünwald, P., de Heide, R. & Koolen, W. (2024), "Safe Testing,"
  *JRSS-B* 86(5), 1091–1128** — mixture/GROW e-value engine for the
  novel branch's state-dependent alternative (already used, and
  independently verified real, in R-174).

### Shared infrastructure (operator-authored, read-only to both branches)

`experiments/r181_shared.py` re-exports R-174's neutral gate
(`run_asymmetric_gate`) and R-102's inner-train/inner-validation/ETH
harness (`compare`, `run_slice`, `paired_diff`, `SLICES`,
`TargetStrategy`) unchanged, and adds two new, causal, self-tested
primitives so neither branch has to re-derive a nontrivial estimator
from scratch:

- `causal_daily_log_return_broadcast` / `causal_daily_log_sigma_broadcast`
  — a one-calendar-day-LAGGED daily aggregate, broadcast onto intraday
  bars (day *D*'s bars see day *D-1*'s own, fully-closed return —
  verified by an explicit two-day synthetic check, in addition to the
  standard truncation probe, that the broadcast value equals the *prior*
  day's return and never the current day's).
- `two_scale_realized_variance` / `two_scale_realized_variance_naive` /
  `causal_bar_sigma_tsrv` — the TSRV estimator, K=5 subsampling grids,
  causal trailing windows, self-tested to actually reduce mean absolute
  error against a known true variance on synthetic noisy data (not
  merely to run without crashing).
- `median_increase_episode_gap_days` — measures, rather than assumes,
  the median gap between v4's own deadband-triggered exposure-INCREASE
  events on a given slice.

**A pre-registered correction, made and disclosed before either branch
was dispatched, per ROUTINE.md's bug-fix allowance:** the research
proposal that motivated this round assumed v4's increase-episodes are
spaced "≈12–22 days apart," reasoning from the ~150–280 total *round-trip
trades* reported over the full nine-year history in `docs/STRATEGIES.md`.
Running `median_increase_episode_gap_days` on inner-train
(2017-01-01→2020-12-31) finds **269 individual increase-episodes, median
gap 2.42 days** — round-trip trade counts and individual re-target
events are different units at up to 100x scale (the same
`num_trades`-vs-turnover distinction ROUTINE.md's own standing rules
warn about, this time on episode counts rather than fill counts). The
conservative branch's reachability precheck (below) uses the MEASURED
2.42-day figure, not the originally-proposed range.

## Frozen decision rules

Both branches attack the SAME comparison
(`compare(candidate_build, control_build=v4_target)` from the shared
module — risk-matched paired diff on inner-train, inner-validation and
the ETH-replication slice, both markets, exactly R-102/R-174's own
convention) and the SAME standard promotion bar. They differ only in
each branch's own Step-0-style reachability/non-degeneracy prechecks,
run BEFORE any Sharpe comparison, exactly mirroring R-170's "stop before
Step B if the primitive is dead on arrival" convention.

### Conservative branch — daily-lagged SPRT

1. **Reachability precheck A** (must run before the SPRT engine itself
   is built): κ = (mean of Σ 288 squared 5-min log returns per day) /
   (variance of the daily close-to-close log return), measured on
   inner-train BTC. **If κ ≤ 1.10, STOP — NEGATIVE by construction**, no
   SPRT code is written (the noise-inflation premise is false).
2. **Reachability precheck B**: at the corrected daily σ, compute each
   α ∈ {0.10, 0.05, 0.20}'s Wald average-sample-number (ASN) in DAYS
   under `MU1_DAILY`/`TAU_DAILY` (from `r181_shared.py`, derived from
   inner-train). **If the ASN in days exceeds 2.42 (the measured median
   increase-episode gap) for ALL THREE α, STOP — NEGATIVE by
   construction** (the fix does not shorten resolution time enough to
   matter on this architecture's own episode cadence). If at least one α
   clears this bar, proceed to Step B.
3. **Kill switch A1** (R-160's own convention, reused): on BTC
   inner-validation, at least one tested α must produce ≥3 episodes
   resolved on a LATER bar than they opened (`delayed_episodes` from
   `run_asymmetric_gate`'s own return value) — else NEGATIVE, the gate
   never actually gated anything.
4. **Kill switch A2** (R-160's own convention, reused): the gated
   candidate's daily-return series must not be a near-exact copy of the
   ungated `v4_target` control (`r_squared(...) < 0.999`) — else
   NEGATIVE, the gate is a no-op.
5. **Promotion bar** (standard, both branches): at least one α clears
   ΔSharpe ≥ **+0.2** (`SHARPE_NOISE_FLOOR`) risk-matched (exposure and
   realized-vol ratio both in [0.9, 1.1], `compare()`'s own
   `risk_matched` flag) on BOTH spot and futures5x inner-validation, OR
   a ≥5pp risk-matched drawdown cut with a paired 95% CI excluding zero;
   the ETH-replication slice must not sign-flip; the effect must be a
   plateau across at least 2 of the 3 tested α, not a spike at one.
   Anything else is NEGATIVE.

### Novel branch — TSRV-corrected GROW e-value

1. **Non-degeneracy precheck**: on BTC inner-train, TSRV
   (`two_scale_realized_variance`) must read **≥10% below**
   (`TSRV_MIN_REDUCTION`) the naive same-window RV
   (`two_scale_realized_variance_naive`) on the majority (>50%) of
   finite windows. **If not, STOP — NEGATIVE by construction** (no
   measurable noise to correct on this data at this window length; the
   branch may retry once at a materially different `window_bars`, e.g.
   BARS_PER_DAY // 2, before concluding NEGATIVE, and must disclose the
   retry).
2. **Kill switch A1 / A2**: identical numeric bars to the conservative
   branch's (3), (4), applied to the TSRV+GROW construction.
3. **Promotion bar**: identical to the conservative branch's (5), swept
   over the novel branch's own free parameter (mixture width /
   `window_bars`), plateau required across at least 2 tested values.

### What would make each fail (Step 1's fourth question, restated)

Conservative: κ too small (no real noise to correct), or the daily
resolution time still exceeds the measured 2.42-day episode cadence at
every α, or the gate never delays anything on real data (A1), or the
Sharpe/drawdown improvement misses the noise floor or sign-flips on ETH.
Novel: TSRV shows no measurable correction over naive RV (no headroom),
or the same downstream failures as the conservative branch.

## Trials count

This module: 0 configurations (shared infrastructure, self-tested only).
Each branch logs and sums its own count in the ledger entry, per
R-163/R-168's convention.

## Scope

Train + inner-validation only. **Neither branch may read a bar dated
2023-01-01 or later.** `assert_no_holdout` (imported from `r102_shared`
via `r181_shared`) must guard every load point, matching every prior
round's convention. If, and only if, a branch clears its own promotion
bar above, the operator (not the branch) decides whether to proceed to
Step 4 (pre-registered holdout evaluation) — per ROUTINE.md, a candidate
that has not cleared Step 3 does not get a holdout read.
