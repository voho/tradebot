"""R-142 CONSERVATIVE branch: Deribit front-vs-next-quarter futures
TERM-STRUCTURE SLOPE, z-scored against its own trailing baseline, as a
confirming vote on `kelly_regime_v4`'s 3-anchor gate -- Step A measurement
gate first, this project's established discipline for every INFO-axis
round since R-53 (R-53/R-73/R-74/R-79/R-81/R-84/R-120).

Shared infrastructure (loader, causal dual-quarter slope construction,
z-score, anchor-vote duplication, confirming-vote rule, stress table,
null generator, causality probe, frozen coverage/episode constants) lives
in `experiments/r142_shared.py`, written and frozen by the operator BEFORE
this file computed any gate/backtest number. This file does not edit that
module. `data/btcusd_deribit_quarterly_5m.csv.gz` and the ETH equivalent
were re-fetched by the operator (unmodified `scripts/
fetch_deribit_quarterly_futures.py`, extended `--last-expiry`) before this
round started, to close the coverage gap R-120 left at 2023-03-31 -- see
r142_shared.py's own module docstring for the disclosed coverage numbers.

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). The SLOPE between the two nearest,
   simultaneously-traded Deribit quarterly futures (next-quarter
   annualized basis minus front-quarter annualized basis) is a
   cross-sectional curve-shape signal -- distinct from R-120's own
   single-point LEVEL and MOMENTUM statistics -- that reflects real-time
   term-structure repricing by cash-and-carry and calendar-spread traders,
   and may therefore reach an extreme before `kelly_regime_v4`'s slow
   20/40/80-day price anchors catch up to a genuine regime shift.

   Citations: Bianchi, Fan, Miffre & Zhang (2023), "Exploiting the
   dynamics of commodity futures curves", Journal of Banking & Finance
   (arXiv 2308.00383) -- Nelson-Siegel slope is a separately profitable,
   uncorrelated factor from curve level in commodities, the direct
   citation for why slope is not a re-parameterization of R-120's own
   level/momentum result; Erb & Harvey (2006), Financial Analysts Journal
   62(2); Schmeling, Schrimpf & Todorov (2023/2025), BIS WP 1087; Chi et
   al. (2023), Journal of Futures Markets -- background term-structure/
   crypto-carry literature, R-120's own citations, reused for context.
   Full citation trail and the discipline note on NOT using the 2025-11
   BTC backwardation/bottom episode to pick this round's gate (found
   during this round's own literature search, deliberately excluded from
   both the episode table and this branch's threshold) are in
   `r142_shared.py`'s module docstring -- one citation trail in one place
   (R-81/R-84/R-120's own convention).

   CONSTRAINT ATTACKED: INFO (one price series) -- the SLOPE needs a
   second, simultaneously-listed instrument this project's OHLCV alone
   cannot express; R-120's own front-quarter-only module never computed
   it.

   NOT A DUPLICATE OF: R-120 (front-quarter basis LEVEL and MOMENTUM,
   single maturity point; ruled out, see docs/LEDGER.md's section C) --
   the sibling NOVEL branch also does not touch this branch's Step-A
   architecture, testing a continuous SIZE-axis dampener instead of an
   INFO-axis confirming vote. R-41/`kelly_regime_v9_basis_lead` (spot-vs-
   perpetual basis, funding-reset every 8h); R-73 (DVOL, implied vol, not
   a forward price); R-81 (OI/positioning, not a priced curve quantity);
   R-63/R-76 (cross-COIN pairs, not cross-MATURITY). Grepped
   docs/LEDGER.md for "term structure", "slope", "curve steepness",
   "calendar spread", "quarterly future": only R-120's own LEVEL/MOMENTUM
   entries and this round's own r142_shared.py hit; zero prior SLOPE
   attempts.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy code,
   BTC first, on `r142_shared.USABLE_EPISODES_BTC` (4 of the full
   6-episode table, per this round's own disclosed coverage measurement
   in r142_shared.py -- COVID, 2021-11 top, Terra/Luna, FTX; the two 2018
   episodes are unreachable, no quarterly contract existed yet).

   PRIMARY FEATURE (chosen now, before any number): `slope_z` --
   `r142_shared.dual_quarter_slope(...).slope` z-scored against its own
   trailing `window_days=20` mean/std via `r142_shared.slope_zscore`
   (matches `kelly_regime_v4`'s own fastest anchor, R-120's own
   window-choice convention).

   THRESHOLD: BIDIRECTIONAL, `|slope_z| >= 1.5` -- matching R-81/R-84/
   R-120's own "extreme" convention; a curve-shape signal can indicate
   stress via inversion (extreme backwardation) or via a crowded, blown-
   off contango, so no directional assumption is fixed before the data is
   seen.

   EPISODE-LOCAL SEARCH WINDOW: [onset - 60 days, onset + 60 days], fixed
   before any number was computed, identical to R-81/R-84/R-120's window
   and for the identical reason (v4's own anchors lag price, so the
   nearest-transition search needs room on both sides).

   ANCHOR-GATE "FLIP" DEFINITION and BASIS "CROSSING" DEFINITION: reused
   verbatim from R-81/R-120's disclosed, bug-fixed convention
   (`r142_shared.nearest_transition`, `direction="down"` for the anchor
   flip; the first bidirectional `|slope_z| >= 1.5` crossing for the
   signal) -- all 4 usable BTC episodes are bearish transitions, so
   "down" is the relevant anchor-flip direction; "either direction" is not
   used for the anchor side at all, per R-81's own disclosed lesson.

   NULL: `r142_shared.block_bootstrap_lead_null`, identical construction
   to R-81/R-84/R-120 (circular block-shift, block_days chosen to exceed
   the signal's own autocorrelation length, n_draws=1000, seed fixed
   before any real-data number is read).

   STEP-A PASS BAR: a usable episode counts as a PASS only if (a) a
   `|slope_z| >= 1.5` crossing exists inside the episode's own search
   window, (b) that crossing occurs STRICTLY BEFORE the anchor-gate's own
   down-flip inside the same window, and (c) the measured lead time beats
   the block-bootstrap null's own 90th percentile (one-sided, matching
   R-81/R-84/R-120's own bar). BTC's gate passes only if
   `count(PASS) >= r142_shared.MIN_EPISODES_PASS_BTC` (>= 3 of 4).

3. DECISION RULE (frozen now):
   - If BTC's Step-A gate FAILS (< 3 of 4 usable episodes pass, matching
     the bar every one of the 19 prior INFO-axis signals in this ledger
     has been held to): STOP. Report NEGATIVE at Step A. Do not build a
     confirming-vote strategy, do not touch ETH, do not touch the
     holdout. This is the modal outcome by this ledger's own base rate
     (0-2/6 or 0-2/4 on every one of R-53 through R-141's INFO-axis Step-A
     gates to date) and is named as such, in advance, not as a hedge
     written after seeing a number.
   - If BTC's Step-A gate PASSES (>= 3 of 4): proceed to Step B. Build the
     confirming vote via `r142_shared.confirming_vote_frac(anchor_sum,
     meta_vote, weight)`, where `meta_vote = 1` when `|slope_z| >= 1.5`
     AND its sign at the crossing instant matches the majority sign
     observed across the PASSING BTC episodes in Step A (a mechanical,
     data-driven-but-pre-committed rule, not a free directional choice --
     analogous to R-81/R-84 fixing `direction="down"` for the anchor side
     from the episode table's own known character, not from a fit).
     Sweep `weight` in `{0.5, 1.0, 2.0}` (3 configurations), confirm the
     `weight=0` identity-recovery check, then run docs/ROUTINE.md's
     standard Step 3 (inner-train/inner-validation only, both BTC and ETH,
     spot and futures) and Step 4 (holdout, only after Step 3 looks
     favourable) exactly as R-53/R-55/R-120 did, against this project's
     standing promotion bar (docs/ROUTINE.md's "the promotion bar --
     default is REJECT" section) -- beats buy_and_hold OOS after real
     costs, improvement exceeds the +/-0.2 Sharpe noise floor or is a
     genuine drawdown/tail improvement, survives the 0.40% fee tier, and
     the parameter neighbourhood is a plateau (report weight=0.5/1.0/2.0
     together, not just the best).

4. WHAT WOULD MAKE IT FAIL (named now): fewer than 3 of 4 BTC episodes
   show a `|slope_z|>=1.5` crossing leading the anchor's down-flip inside
   +/-60 days, distinguishable from the block-bootstrap null -- i.e. the
   curve-shape extreme is not, in fact, earlier than v4's own slow anchors
   at the episodes this project already uses to judge every other
   regime-adjacent signal.

CONFIGURATIONS EVALUATED: to be filled in by the implementing session,
counting every Step-A cell (4 episodes x 1 threshold = 4) plus, only if
Step B is reached, every weight grid point x market x period cell.
"""

from __future__ import annotations
