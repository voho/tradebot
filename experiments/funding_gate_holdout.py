"""B-05 holdout evaluation, pre-registered before the 2023 holdout was read.

This file is committed BEFORE its own output is read (the git log entry
for this commit is the pre-registration timestamp, per the convention
R-28/R-31/R-32 established: "the freezing commit is one commit ahead of
the results"). Do not edit the frozen config or the decision rule below
after seeing output from this script.

Context
-------
Two candidate variants for backlog item B-05 (funding as a gate on
kelly_regime_v4, COST constraint) were built and tuned in parallel, each
on its own file, per ROUTINE.md's parallelism rules:

- ``experiments/funding_gate_decile.py`` -- FundingDecileGate. Conservative:
  a single binary mechanism, force flat when trailing funding is in its
  own top decile. On inner-validation (2022) it beat kelly_regime_v4 on
  BOTH markets across 9 of 12 swept configs (a plateau, not a peak), and
  its improvement survived a 0.40% taker fee tier on that same window.
  Gate active ~10-14% of bars -- a targeted, occasional intervention.

- ``experiments/funding_gate_continuous.py`` -- FundingAwareKelly. Novel:
  a continuous funding-aware Kelly haircut extending kelly_regime_ev's
  analytic fee-deadband derivation to a continuously-accruing cost. Also
  beat kelly_regime_v4 on every one of its 9 swept inner-validation
  configs, and won 10/12 paired Monte Carlo windows drawn from
  2020-2022. But its own report flags a serious doubt, in its own words:
  mean exposure drops to 40-75% of kelly_regime_v4's, it de-levers on
  SPOT too (where there is no funding bill to avoid at all), and "the
  clean win on every validation cell is suspicious for the same reason
  R-28's win was -- it may just be de-levering helped in a year v4 lost
  money" (2022 was a losing year for v4: Sharpe -1.44). That is exactly
  the mechanical-delever artifact R-31/R-32 already caught once in this
  project (an e-process gate's "0-of-40-windows" drawdown win dissolved,
  and partly REVERSED, once measured at matched risk against the vote
  gate it was compared to).

Selection between the two candidates (made on training/inner-validation
evidence only, per ROUTINE.md step 3 -- "select on it as often as you
like"; this is NOT the pre-registered holdout step):

**FundingDecileGate is carried forward to the holdout. FundingAwareKelly
is not.** Reasons, weighed before looking at 2023: (1) it is a much
smaller, more targeted intervention (~10-14% of bars) rather than a
persistent ~25-60% de-lever of the whole book, so it is mechanically
less likely to be "just a de-lever that wins because 2022 was a loss" --
the project's own prior finding says a persistent exposure cut is
exactly the shape that manufactures a false risk finding; an occasional,
threshold-triggered one is a smaller target for that critique, though
not immune to it (see the mechanism check below, which is run on the
holdout regardless of outcome). (2) It only fires on the funding
percentile signal itself, never on spot where there is no cost basis to
avoid, whereas the continuous variant's own report flags firing on spot
too as a sign it may be trading R-16's noisy return-predictive signal
rather than "cost avoidance" specifically. (3) It is simpler and closer
to B-05's literal framing ("stand flat in the top decile"). (4) Its own
falsification test (0.40% fee) was run on the SAME window used for
config selection and passed; the continuous variant's falsification
(paired MC windows) is stronger evidence in isolation, but its own
report concludes "not something I'd promote on this evidence alone,"
while the decile gate's does not carry an equivalent self-flagged doubt
of that severity.

This is a documented judgement call, not a coin flip -- and it is
recorded here, before the holdout, specifically so it cannot be quietly
revised if the holdout had gone the other way. FundingAwareKelly's
inner-validation numbers stand as reported by its own session and are
recorded in the ledger as a parallel branch, per "every branch reports,
including the dead ones" -- it simply does not spend a holdout
consultation, because only one candidate is being promoted through this
round.

Frozen configuration
---------------------
``FundingDecileGate(funding_window_days=30, flatten_threshold=0.85)`` --
the top-ranked config on inner-validation that beat kelly_regime_v4 on
BOTH markets, per funding_gate_decile.py's own ``select_config`` rule,
sitting in a neighbourhood (w60/w90/w180 at t=0.85) that also beats v4,
i.e. a plateau along the window-length axis at the selected threshold.

Holdout window
---------------
Funding data is committed 2020-01-01..2023-12-31 only. The project's
real OOS_START is 2023-01-01 and its holdout otherwise runs to the
present (2026-08); nothing here can read funding past 2023-12-31, so the
holdout for THIS strategy specifically is **2023-01-01..2023-12-31 only
-- a single year**, far short of the project's usual 3.6-year holdout.
That is a real limitation of this finding's statistical power, stated
here before any number is read, not discovered afterward.

Pre-registered decision rule (promote only if ALL of P1-P4 hold; default
is REJECT)
-----------------------------------------------------------------------
- **P1 (return).** On 2023-01-01..2023-12-31, spot final balance beats
  ``buy_and_hold`` (the project's binding benchmark).
- **P2 (materiality).** The improvement over ``kelly_regime_v4``
  (funding-charged, same window, futures 5x -- the market where the
  mechanism actually operates) exceeds the +/-0.2 Sharpe noise floor
  (R-20), OR is a drawdown improvement of >= 10 percentage points.
- **P3 (falsification, chosen now).** The improvement over
  ``kelly_regime_v4`` on futures survives Bitstamp's 0.40% taker fee
  tier, re-run on THIS holdout window (not just on the inner-validation
  window where it was already checked once) -- confirming the fee
  result was not an artifact of the specific training window.
- **P4 (plateau, not peak).** The immediate neighbours already
  characterized as a plateau on inner-validation (w60/w90/w180 at
  t=0.85) are re-run on the holdout and checked for the same ordering
  (not required to all individually beat v4, since one year is a small
  sample -- but no wild reversal that would suggest a peak rather than a
  region).

**Recorded regardless of P1-P4, as a mechanism check, not a promotion
criterion (out of scope for a full matched-risk redo -- that is backlog
item B-13, not this round):** kelly_regime_v4's and the candidate's mean
exposure on the holdout window, side by side. If the entire improvement
traces to a lower mean exposure with no funding-specific timing content,
that is flagged explicitly and the result is downgraded in the written
lesson even if P1-P4 formally pass, per this project's standing rule
that a comparison against a fully-invested benchmark is not the same
claim as a comparison against a risk-matched one (R-31, R-32).

**Stated prediction, before looking:** P1 and P3 likely hold (the
0.40% fee result was robust across both inner splits and the gate is a
COST-avoidance mechanism as much as a return one). P2 is the most
uncertain -- one year is a small sample and 2023 is not obviously a year
funding runs unusually rich, unlike the 2020-2021 configs it was tuned
on. The mechanism check will likely show SOME of the gap is attributable
to lower mean exposure, since regime-gated sizing being flat some of the
time always has that side effect (L-04's own robust finding) -- the
question is whether it is ALL of the gap or only part of it.
"""
