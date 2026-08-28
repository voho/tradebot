# R-175 pre-registration — Markov-Switching Multifractal (MSM) volatility decomposition on `kelly_regime_v4`'s SCALE input

Status: FROZEN before either branch was dispatched or any real-data
performance number was read. Adapted, with the shared engine now built and
self-tested (`experiments/r175_shared.py`), from a research sub-agent's
uncommitted proposal (`experiments/r175_direction_candidate.md`, kept
verbatim for disclosure of the search process). Only framing/status and
the now-concrete engine parameters differ from that candidate; the
mechanism, citations, non-duplication argument and both branches'
falsification tests are unchanged.

## Step 0 recap

`git fetch origin main` + `git rev-parse HEAD origin/main` confirmed HEAD ==
origin/main (55a17d8) before this round started; no `r<nn>_shared.py` exists
without a section B entry (newest is `r174_shared.py`, `grep -c "R-174"
docs/LEDGER.md` = 3). Step 0b's saturation count was 1 consecutive null pass
since R-173's own dispatch ("0-2: normal"). The backlog grep returns only
**B-48** (a documentation/formatting instrument fix, not a strategy-research
item) plus four already-inactionable rows (B-06 de-ranked, B-09 LOW, B-17
PARTIAL, B-28 blocked on data) — so this is a fresh off-backlog
literature-prompted round, the same convention R-160 through R-174 used.

## The idea, in one sentence

Replace the single-span EWM realized-volatility estimator driving
`kelly_regime_v4`'s conditional vol-target (`vol` vs. its own 180-day slow
EWM, in `kelly_regime_v3.KellyRegimeV3.prepare` / `conditional_target_scale`)
with a Calvet & Fisher Markov-Switching Multifractal (MSM) decomposition of
BTC's own realized volatility into a hierarchy of `kbar=6` latent binomial
components at geometrically-spaced persistence, and drive the SAME
hysteresis state machine off either the full decomposition (conservative)
or only its most persistent third (novel) — directly testing this ledger's
own standing explanation (R-08, R-136) for why every prior "better"
volatility forecast has hurt this strategy.

## Step 1 — the four required questions

**1. Which constraint does it attack?** **SIZE**, primarily — it changes
only the volatility-target input (`vol` in `conditional_target_scale`), the
one slot R-62 already showed carries none of v4's headline signature but
that every hysteresis/deadband question ultimately depends on. Secondary:
**COST**, since a materially different vol estimate changes how often the
hysteresis state flips and the 10% deadband is crossed. Does not attack ERR
or INFO — no new information source (MSM is fit on BTC's own return series
already in the file) and no hypothesis test or multiplicity correction is
added.

**2. Which ledger entries is it not a duplicate of, and why the difference
should matter?**

- **Not R-08** (08-15, timescale-blended vol forecast, sign-inverting) and
  **not R-136** (08-25, HAR-RV/DVOL blend, reproduces R-08's inversion on
  the modern conditional-targeting architecture). Both improved forecast
  *accuracy* via an additive, single-horizon-weighted blend and found
  accuracy itself is the problem — a more accurate forecast reacts faster to
  the short vol spikes that precede BTC's sharpest rallies (Baur & Dimpfl
  2018's inverse leverage effect). MSM is a **multiplicative volatility
  cascade** with an explicit persistence/duration structure per component,
  not another point-forecast blend. The novel branch does not chase forecast
  accuracy — it uses MSM's own decomposition to *deliberately discard* the
  highest-frequency, least-persistent component(s), the opposite move from
  R-08/R-136's "blend everything into one better number." If R-08's causal
  story is right, this is the first construction built to test it rather
  than merely re-confirm it by a different forecasting method.
- **Not R-09** (08-15, Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang
  range estimators). MSM is fit on close-to-close log returns at native
  cadence; no high/low range data enters.
- **Not R-85** (08-21, critical slowing down on the VOTE's regime-timing,
  scored against the six-episode detection-lag gate). CSD asks whether
  rising variance predicts a regime break early enough to move the vote
  sooner; this proposal never touches the vote and reclassifies already-
  realized volatility into persistent/transient components for the SCALE
  factor's own denominator only.
- **Not R-152** (CDaR-budgeted leverage cap), **R-161/R-167** (RCPS/CRC
  tail-loss caps), **R-171** (Online Newton Step leverage path), **R-164**
  (Barroso-Santa Clara realized-payoff-variance targeting / Daniel-Moskowitz
  panic-calm multiplier), **R-162** (Kaufman Efficiency Ratio), **R-163**
  (pyramiding/excursion multipliers), or **R-166** (sign-inverting the
  vol-response exponent). Every one of those either caps/learns a multiplier
  on the existing scale's *output*, or substitutes a statistic of the
  *strategy's own realized payoff/drawdown* for the vol ratio. This proposal
  touches neither: it stays inside the existing `target_vol / vol`
  architecture and only replaces what `vol` itself means.
- **Not R-141** (LPPLS bubble-confidence dampener — a bounded multiplicative
  SCALE dampener). No dampener is added; the estimator itself changes.
- Never previously named in this ledger: `docs/LEDGER.md` has zero hits for
  "multifractal", "Calvet", or "MSM" before this entry.

**3. Is it simulable here?** Yes, with only data already in the repo. MSM is
fit purely on the causal log-return series already used for the incumbent
estimator — no order book, no queue model, no funding data required for the
mechanism itself (funding remains available for the standard fee/funding
robustness check, unchanged convention). The filter runs on DAILY
aggregates for tractability (no scipy in this environment, confirmed by
`r118`/`r125`; a `kbar=6` Hamilton filter at 5-minute resolution over ~1M
bars is not feasible in a pure-Python/numpy loop) with a periodic causal
refit — the same daily-refit discipline this project already applies to
HAR-RV (R-136), CDaR (R-152) and RCPS (R-161/R-167). Per R-172's own lesson
about same-day daily-broadcast lookahead, day D's forecast is strictly a
one-step-ahead prediction from data through day D-1's close, verified by
`causal_truncation_probe_series` against real BTC data (not only synthetic)
in `r175_shared._self_test`.

**4. What would falsify it, named now?** (a) the conservative branch is
falsified if its ΔSharpe/Δlog-growth vs. the incumbent estimator is negative
with a paired-bootstrap CI excluding zero on the losing side on BTC
inner-validation — a third confirmation of R-08/R-136's forecast-quality
inversion, this time against a structurally different model family; (b) the
novel branch is falsified independently of any performance number if its
persistence-filtered exposure is **not** higher (less de-risked) than the
incumbent's at a majority of BTC's six known historical volatility-spike
episodes — if the mechanism does not even do the one thing it was built to
do, no performance comparison is needed to kill it.

## Step 2 — citations and design

**Mechanism citations.**
- Calvet, L. & Fisher, A. (2001). "Forecasting Multifractal Volatility."
  *Journal of Econometrics* 105(1), 27–58.
- Calvet, L. & Fisher, A. (2004). "How to Forecast Long-Run Volatility:
  Regime Switching and the Estimation of Multifractal Processes." *Journal
  of Financial Econometrics* 2(1), 49–83. The grid-search ML estimation
  procedure this round's engine follows (low-dimensional `(m0,b,gamma_kbar)`
  parameter space, sigma_bar profiled out in closed form).
- Baur, D. & Dimpfl, T. (2018). "Asymmetric Volatility in Cryptocurrencies."
  *Economics Letters* 173. Already cited in `kelly_regime_v3.py`'s own
  docstring; the novel branch is built to exploit this fact rather than
  fight it.
- arXiv:2507.00575 (2025), "Multifractality in Bitcoin Realized Volatility:
  Implications for Rough Volatility Modelling" — contemporary evidence BTC's
  own realized-vol series carries genuine multi-persistence structure,
  checked by WebSearch this round, not previously cited in this ledger.

**What was checked and rejected before settling on this candidate.** A
drawdown-bucket / auto-deleveraging exposure schedule (two 2025-2026 arXiv
hits) was traced to the same family as the already-twice-closed CPPI (R-46)
and CDaR-budget (R-152) constructions and set aside as a likely duplicate.
Markov-Switching GARCH / Hidden semi-Markov volatility models were set aside
as the same discrete-state-switching basis already closed 11+ times against
the six-episode regime-timing gate (R-01 HMM, R-82 BOCPD, R-96 Hawkes) — MSM
is a continuous-state, component-cascade model, the substantive reason it
survives that closed list. Funding-rate *volatility* (level closed four
ways: R-16, R-35, R-39, R-100) was set aside as too thin a difference to
justify its own round.

**Engine parameters, fixed before any real-data comparison (see
`experiments/r175_shared.py` for the implementation and self-test):**
`kbar=6` (64 joint states), grid of 4×4×4=64 `(m0, b, gamma_kbar)` points,
`sigma_bar` profiled out to the calibration window's own sample std,
2-year trailing causal calibration window, 90-day refit cadence, 180-day
minimum history before the estimate leaves its flat (`multiplier=1.0`)
fallback, `N_PERSIST=2` (the lowest/most persistent third of 6 components)
for the novel branch's structural-only functional.

## Conservative variant

**Mechanism, one sentence.** Fit the MSM(6) model and substitute its
full one-step-ahead multiplier forecast (`msm_full_vol_bars`) for the
incumbent's fast EWM `vol` estimate, leaving `slow`, the hysteresis
thresholds, `target_vol`, `max_leverage` and the 10% deadband
byte-identical to `kelly_regime_v4` today.

**Pre-registered falsification test.** On BTC inner-validation
(2021-01-01..2022-12-31), compute the paired-bootstrap CI on
`d_log_growth`/`d_sharpe` against the incumbent's own `vol` estimator, both
markets. **FALSIFIED** if the CI excludes zero on the losing side on either
market, or if `R²>0.98` between MSM's forecast and the incumbent's EWM vol
(mere relabeling). Passes to the standard promotion table (ΔSharpe≥+0.2 or
risk-matched drawdown improvement, both markets; ETH sign-replication;
survives the 0.40% taker tier) only if neither falsifier fires.

## Novel variant

**Mechanism, one sentence.** Use only the lowest `N_PERSIST=2` of 6 MSM
components' contribution to total variance (`msm_structural_vol_bars`),
deliberately zeroing out the highest-frequency, most transient components,
as the numerator of the existing `ratio = vol/slow` hysteresis state
machine.

**Pre-registered falsification test, two parts, either kills the branch.**
(a) **Mechanism check, decisive regardless of any performance number.** At
the six dated `STRESS_EPISODES` in `r175_shared.py` (reused verbatim from
R-82/R-83/R-85), compare the candidate's realized exposure in the 10
trading days following each episode's own vol spike against the
incumbent's, via `exposure_at_episodes`. **FALSIFIED** if the candidate is
not less de-risked (strictly higher mean exposure) than the incumbent in at
least 4 of 6 episodes. (b) **Standard promotion bar**, only if (a) passes:
same four-clause house convention as (a) above.

## Honest prior, stated before any code ran

Since R-62 isolated that `kelly_regime_v4`'s signature lives in the vote,
not the scale, this ledger has tried roughly fifteen distinct SCALE-axis
mechanism substitutions or caps and zero have promoted; two were
specifically about volatility forecast *quality* (R-08, R-136) and both
inverted sign. The conservative branch is, in substance, a third attempt at
that sub-question with a different model family — prior on clearing the bar
is **low, on the order of 10%**. The novel branch is more interesting
because its falsification test (a) is decisive on mechanism alone,
independent of the noisy Sharpe/drawdown comparison every other SCALE round
has hinged on. A clean negative that narrows "is the transient component
specifically the problem, or is any sufficiently good vol estimate the
problem" is disclosed as the expected, still-instructive outcome.
