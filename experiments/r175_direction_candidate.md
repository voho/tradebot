# R-175 direction candidate — Markov-Switching Multifractal (MSM) volatility decomposition on `kelly_regime_v4`'s SCALE input

Status: PROPOSED, not frozen, not dispatched. No code written. No data touched
beyond reading source files and `docs/LEDGER.md`. For the operator to accept,
reject, or amend before it becomes R-175's pre-registration.

## The idea, in one sentence

Replace the single-span EWM realized-volatility estimator that drives
`kelly_regime_v4`'s conditional vol-target (`vol` vs. its own 180-day slow
EWM, in `kelly_regime_v3.KellyRegimeV3.prepare`) with a Calvet & Fisher
Markov-Switching Multifractal (MSM) decomposition of BTC's own realized
volatility into a hierarchy of latent components at geometrically-spaced
persistence horizons, and drive the vol-target ratio off the model's
slowest-decaying ("structural") components rather than an undifferentiated
blend — directly testing this ledger's own standing explanation (R-08, R-136)
for why every prior "better" volatility forecast has hurt this strategy.

## Step 1 — the four required questions

**1. Which constraint does it attack?** **SIZE**, primarily — it changes the
*volatility-target input*, not the trend vote, so it lands in the one slot
(`vol`/`slow`/`ratio` in `kelly_regime_v3.py`) that R-62 already showed
carries none of v4's headline signature, but is also the slot every
COST-adjacent hysteresis/deadband question ultimately depends on (the state
machine that latches high/low-vol breakouts uses this same `ratio`). Secondary:
**COST**, because a materially different vol estimate changes how often the
hysteresis state flips and therefore how often the 10% deadband is crossed.
Does not attack ERR or INFO — no new information source is introduced (MSM is
fit on BTC's own return series, already in the file) and no hypothesis test or
multiplicity correction is added.

**2. Which ledger entries is it not a duplicate of, and why the difference
should matter?**

- **Not R-08** (08-15, timescale-blended vol forecast, 8% better QLIKE,
  "sign-inverting" — a *better point forecast* de-levers faster into BTC's
  high-vol/high-forward-Sharpe states) and **not R-136** (08-25, HAR-RV blend
  + DVOL implied vol, "reproduces R-08's forecast-quality inversion on the
  modern conditional-targeting architecture"). Both prior rounds improved
  forecast *accuracy* via an **additive, single-horizon-weighted** blend of
  realized-vol estimates and both found that accuracy itself is the problem —
  a more accurate forecast reacts faster to the short vol spikes that precede
  BTC's sharpest rallies (Baur & Dimpfl 2018's inverse leverage effect, already
  cited in `kelly_regime_v3`'s own docstring). MSM is not another point-forecast
  blend: it is a **multiplicative volatility cascade** with an explicit
  persistence/duration structure per component (Calvet & Fisher 2001, 2004).
  The novel branch below does not chase forecast accuracy at all — it uses
  MSM's own decomposition to *deliberately discard* the highest-frequency,
  least-persistent component before computing the SCALE ratio, which is the
  opposite move from R-08/R-136's "blend everything into one better number."
  If R-08's causal story is right (harm comes specifically from reacting to
  short-lived spikes), this is the one construction in the ledger built to
  test that story rather than merely re-confirm it by a different forecasting
  method.
- **Not R-09** (08-15, Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang range
  estimators — discretization bias reading 7-18% low at 5m resolution). MSM is
  fit on close-to-close log returns at the same native cadence the incumbent
  already uses; no high/low range data enters.
- **Not R-85** (08-21, critical slowing down — rising variance/autocorrelation
  as an early-warning signal for the **VOTE's** regime-timing, scored against
  the six-episode detection-lag gate, 0-1/6). CSD asks "can rising variance
  predict a regime break early enough to move the vote sooner." This proposal
  does not touch the vote or try to lead any dated event; it only reclassifies
  already-realized volatility into persistent vs. transient components for the
  **SCALE** factor's own denominator.
- **Not R-152** (08-27, CDaR-budgeted exposure / CDaR-derived leverage cap —
  a drawdown-based risk budget) or **R-161/R-167** (RCPS/CRC tail-loss caps) or
  **R-171** (Online Newton Step learning the leverage path) or **R-164**
  (Barroso-Santa Clara realized-payoff-variance targeting / Daniel-Moskowitz
  panic-calm multiplier) or **R-162** (Kaufman Efficiency Ratio) or **R-163**
  (pyramiding/excursion multipliers) or **R-166** (sign-inverting the
  vol-response exponent). Every one of those closed rounds either (a) caps or
  learns a multiplier on the *output* of the existing scale calculation, or
  (b) substitutes a different *statistic of the strategy's own realized
  payoff/drawdown* for the vol ratio. This proposal touches neither: it stays
  inside the existing `target_vol / vol` architecture and only replaces what
  `vol` (the price-volatility estimate itself) means, decomposed by
  persistence rather than re-weighted by horizon.
- **Not R-141** (LPPLS bubble-confidence dampener — a bounded-above-1
  multiplicative SCALE dampener, degenerate by construction under equality
  exposure-matching). No dampener is added here; the estimator itself changes.
- Never previously named in this ledger: a search of `docs/LEDGER.md` for
  "multifractal", "Calvet", and "MSM" returns zero hits before this entry.

**3. Is it simulable here?** Yes, with only the data already in the repo.
MSM is fit purely on the causal log-return series already used to compute the
incumbent's `r = np.log(close).diff()` — no order book, no queue model, no
funding data required for the mechanism itself (funding is available via
`scripts/funding_study.py` for the standard fee/funding-charged robustness
check, unchanged from every other round's convention). Estimation is a
standard binomial-multiplier MSM(k̄) fit (Calvet & Fisher's own GMM or ML
procedure) refit periodically on an expanding/rolling causal window, exactly
the same causal-refit discipline this project already applies to HAR-RV
(R-136), CDaR (R-152) and RCPS (R-161/R-167).

**4. What would falsify it, named now?** Two independent falsification points,
one per branch (full detail under each variant below): (a) the conservative
branch is falsified if its ΔSharpe/Δlog-growth vs. the incumbent estimator is
negative with a paired-bootstrap CI excluding zero on the losing side on BTC
inner-validation — a third confirmation of R-08/R-136's forecast-quality
inversion, this time against a structurally different model family, not
merely a fourth blend; (b) the novel branch is falsified independently of any
performance number if its own persistence-filtered exposure is **not** higher
(less de-risked) than the incumbent's at a majority of BTC's known historical
volatility-spike-into-rally episodes — i.e., if the mechanism does not even
do the one thing it was built to do, no performance comparison is needed to
kill it.

## Step 2 — citations and design

**Mechanism citations.**
- Calvet, L. & Fisher, A. (2001). "Forecasting Multifractal Volatility."
  *Journal of Econometrics* 105(1), 27–58. Original MSM forecasting result:
  a multiplicative cascade of k̄ latent binomial volatility components at
  geometrically related persistence, out-forecasting single-regime models at
  10-50 day horizons.
- Calvet, L. & Fisher, A. (2004). "How to Forecast Long-Run Volatility:
  Regime Switching and the Estimation of Multifractal Processes." *Journal of
  Financial Econometrics* 2(1), 49–83. The estimation/inference procedure
  (ML and GMM) this proposal's conservative branch would use verbatim.
- Baur, D. & Dimpfl, T. (2018). "Asymmetric Volatility in Cryptocurrencies."
  *Economics Letters* 173. Already cited in `kelly_regime_v3.py`'s own
  docstring for the inverse-leverage-effect fact this proposal's novel branch
  is built to exploit rather than fight.
- Recent supporting evidence, checked by WebSearch this session (not
  previously cited anywhere in this ledger): a 2025 arXiv working paper,
  "Multifractality in Bitcoin Realized Volatility: Implications for Rough
  Volatility Modelling" (arXiv:2507.00575), documents that BTC's own realized
  volatility series carries genuine multifractal/multi-persistence structure
  (Hurst exponents materially above 0.5, non-trivial multifractal spectrum
  width) rather than being a single-regime process — a direct, contemporary
  data point for why a persistence-decomposed estimator (rather than a
  single- or blended-timescale one) is the mechanistically distinct thing to
  try here, not merely a relabeling of R-08/R-136's blends.

**What was checked and rejected before settling on this candidate** (for
honesty, per this project's own convention of disclosing the search, not just
the winner): a drawdown-bucket / auto-deleveraging-style exposure schedule
(arXiv:2603.15963, arXiv:2512.22476, both 2025-2026) was traced back to the
same family as the already-twice-closed CPPI (R-46) and CDaR-budget (R-152)
constructions — reducing exposure as a function of current or trailing
drawdown depth — and was set aside as a likely duplicate rather than tested.
Markov-Switching GARCH / Hidden semi-Markov volatility models (several
2025-2026 hits) were set aside as the same discrete-state-switching basis
already closed eleven-plus times against the six-episode regime-timing gate
(R-01 HMM, R-82 BOCPD, R-96 Hawkes, etc.) — MSM is a **continuous-state,
component-cascade** model, not a discrete regime-switch, which is the
substantive reason it survives that closed list and those two do not. Funding
rate *volatility* (as opposed to level, already closed four ways: R-16, R-35,
R-39, R-100) was considered and set aside as too thin a difference from the
closed funding-level family to justify a round on its own.

## Conservative variant

**Mechanism, one sentence.** Fit the literal Calvet-Fisher MSM(k̄) model
(binomial multiplier distribution, k̄ selected by the standard BIC grid,
e.g. 7-10 components) on BTC's causal log-return series and substitute its
one-step-ahead conditional volatility forecast for the incumbent's fast EWM
`vol` estimate, leaving `slow`, the hysteresis thresholds
(`high_in/high_out/low_in/low_out`), `target_vol`, `max_leverage` and the 10%
deadband byte-identical to `kelly_regime_v4` today.

**Pre-registered falsification test.** On BTC inner-validation
(2021-01-01..2022-12-31), compute the paired-bootstrap CI on
`d_log_growth`/`d_sharpe` against the incumbent's own `vol` estimator, both
markets (spot, futures 5x). **FALSIFIED** if the CI excludes zero on the
losing side on either market (the R-08/R-136 pattern, a third time, on a
structurally distinct model) — or if `R²>0.98` between MSM's forecast and the
incumbent's EWM vol (mere relabeling, no new information content, the R-141
degenerate-construction failure mode). Passes to the standard promotion table
(ΔSharpe≥+0.2 or risk-matched drawdown improvement, both markets; ETH
sign-replication; survives the 0.40% taker tier) only if neither falsifier
fires.

## Novel variant

**Mechanism, one sentence.** Extract MSM's own fitted multiplier-component
hierarchy and compute a "structural volatility" estimate using only the
lowest-frequency (most persistent, longest half-life) third of the k̄
components' contribution to total variance — deliberately zeroing out the
highest-frequency, most transient component(s), which by the model's own
construction are the ones that spike and decay fastest — then use this
persistence-filtered estimate as the numerator of the existing `ratio =
vol/slow` hysteresis state machine, on the hypothesis (Baur & Dimpfl 2018;
R-08's own diagnosis) that the transient component specifically is what was
driving every prior "better forecast" to de-lever into BTC's best forward-
Sharpe states.

**Pre-registered falsification test, two parts, either kills the branch.**
(a) **Mechanism check, decisive regardless of any performance number.** At
the same six dated episodes this ledger's regime-timing gate already uses
(2018 bear onset, COVID crash/recovery, 2021-11 top, Terra/Luna, FTX — the
table reused verbatim from R-82/R-83/R-85/etc.), compare the candidate's
realized exposure in the 5-10 trading days *following* each episode's own
vol spike against the incumbent's and the conservative branch's exposure at
the same points. **FALSIFIED** if the candidate is not less de-risked
(strictly higher mean exposure) than the incumbent in at least 4 of 6
episodes — if the persistence filter does not even change *which* volatility
the strategy reacts to in the one place it was built to matter, no further
number is worth computing. (b) **Standard promotion bar**, only if (a)
passes: ΔSharpe≥+0.2 or risk-matched drawdown improvement on both BTC and ETH
inner-validation, ETH sign-replication, survives the 0.40% taker tier — the
same four-clause house convention R-160 through R-174 used.

## Honest prior, stated before any code

This axis's base rate is bad, and should be named plainly rather than buried.
Since R-62 isolated that `kelly_regime_v4`'s entire signature lives in the
**vote**, not the **scale**, this ledger has tried roughly fifteen distinct
SCALE-axis mechanism substitutions or caps (R-141, R-152 ×2, R-161/R-167 ×4,
R-162, R-163 ×2, R-164 ×2, R-166, R-171 ×2) and **zero have promoted**. Two of
those attempts were specifically about volatility *forecast quality* (R-08,
R-136) and both inverted sign. This proposal's conservative branch is, in
substance, a third attempt at that same specific sub-question with a
different model family, and the honest prior on it clearing the bar is **low,
on the order of 10%**, essentially the ledger's own measured rate for this
exact sub-axis. The novel branch is the more interesting of the two because
its falsification test (a) is decisive on mechanism alone, independent of the
noisy Sharpe/drawdown comparison every other SCALE round has hinged on — it
either changes what the strategy reacts to in the way the theory predicts, or
it does not, and that can be checked directly against six already-dated,
already-used episodes before any P&L number is trusted. Even if (a) passes,
the ledger's own base rate says (b) is the more likely place to fail. This is
disclosed as the expected outcome, not a reason to skip the round: a clean
negative that narrows "is the transient component specifically the problem,
or is any sufficiently good vol estimate the problem" would be a genuinely
new, independently instructive finding for this axis even on failure,
which is this project's own bar for what counts as worth a table row.
