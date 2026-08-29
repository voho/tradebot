# R-179 pre-registration (08-29) — meta-labeling as an ERR-axis confidence gate on `kelly_regime_v4`

## Step 0

`git rev-parse HEAD origin/main` are equal (`902552a`); no undispatched
`r<nn>_shared.py` (newest is `r178_shared.py`, recorded as R-178). Step 0b
consecutive-null-pass count is **2** (the two 08-28 verification-pass rows
directly above this entry) — squarely "0–2: normal," so a fresh
literature-sweep round is in scope. Backlog is B-48 (OPEN, a formatting
instrument fix, not a strategy round), B-06/B-09/B-17/B-28 (blocked/low/
partial) — no live, unblocked strategy-research item, so a new direction is
warranted rather than mandatory backlog work.

## Step 1 — the four questions

**Idea, one sentence.** Wrap `kelly_regime_v4`'s existing vote+scale signal
in a supervised *meta-label* — a secondary model trained on triple-barrier-
labeled historical outcomes of "would holding the currently-desired exposure
have been profitable" — and use its output to gate or scale the primary
signal, rather than acting on the vote unconditionally.

1. **Which constraint does it attack?** **ERR** — "no error control anywhere
   in the signal path." Every ERR-axis attempt in this ledger so far (R-104
   aggregate significance, R-105/R-106 leave-one-out disagreement, R-109–
   R-123 market-state novelty brakes, R-114 regime duration, R-116
   cross-asset divergence, R-147 parametric shrinkage, R-160 online-FDR,
   R-172 FCR/PoSI — 14 rounds) is an *unsupervised* statistic computed from
   the signal itself (a p-value, a disagreement count, a novelty score).
   None of them is fit against a labeled record of what actually happened
   the way the vote acted. Meta-labeling (López de Prado 2018, *Advances in
   Financial Machine Learning*, Wiley, ch. 3) is the standard supervised
   alternative: label historical bet outcomes with the triple-barrier method,
   fit a secondary classifier to predict P(profitable), and use that
   probability to decide whether/how much to bet — Joubert, Barziy & Meyer
   (2022, *J. Financial Data Science* 4(3), "Meta-Labeling: Theory and
   Framework" and 5(2), "Meta-Labeling: Calibration and Position Sizing")
   extend this specifically to *continuous* position-sizing via a calibrated
   probability and a sigmoid bet-sizing function, rather than a binary
   trade/no-trade filter.

2. **Which ledger entries is it not a duplicate of?**
   - Not a duplicate of the 14 ERR-axis rounds above: those are unsupervised
     statistics on the signal; this is a supervised classifier fit against
     realized triple-barrier-labeled outcomes, with an explicit
     purge/embargo boundary rather than an aggregate test statistic.
   - Not a duplicate of R-6x's conformal dispersion estimator (that replaced
     the **scale** (volatility) factor alone, per R-62's factor-isolation
     finding that scale alone carries none of v4's edge); this gates the
     **composite** `frac*scale` decision, downstream of both factors.
   - Not a duplicate of the ten regime-timing detectors (HMM/BOCPD/Kalman
     LLT/CSD/transfer-entropy/Hawkes/POT-GPD/vote-latch/CUSUM/Donchian,
     R-79–R-substantially since): those try to detect a regime change
     *earlier*; meta-labeling does not touch detection timing at all — it
     asks whether an *already-detected* vote state is currently reliable
     enough to act on, a different question, on the same detected vote.
   - Not a duplicate of R-163's trade-level pyramiding or R-171's online
     leverage selection (SIZE-axis, unsupervised, online-convex-optimization
     mechanisms): this is a supervised, offline-refit-then-forward-applied
     classifier, walk-forward, not an online regret-minimizing accumulator.

3. **Is it simulable here?** Yes, entirely from the existing 5m OHLCV frame
   and `kelly_regime_v4`'s own already-computed `frac`/`scale` arrays — no
   new data channel, no order book, no queue model. The triple-barrier
   labels are computed causally (a label at checkpoint `t` only resolves
   using bars `> t`, and is only *used* by the walk-forward classifier once
   its own horizon + embargo have fully elapsed relative to the fit time —
   see `r179_shared.py`'s `walk_forward_meta_prob`).

4. **What would make it fail — named now, before any code beyond the shared
   scaffold below?**
   - **The N≈3 problem, inherited rather than escaped.** `kelly_regime_v4`
     rebalances only ~150–280 times in nine years (per its own docstring,
     143 trades for the base `kelly_regime`). A classifier trained only on
     *actual rebalance* outcomes would have single-digit samples per
     walk-forward refit — untestable. This round's designed fix is to label
     at a **daily** checkpoint (whether or not `v4` actually rebalances that
     day: "would holding the currently-desired exposure over the next
     `horizon_days` have cleared its triple barrier"), giving ~3,400 labels
     over the full history — but this is itself a design choice that could
     be wrong, and is disclosed as such. **Falsification clause A:** if the
     walk-forward classifier's resolved-sample count per refit stays below
     50 for a majority of refit checkpoints in the *training* period
     (2017-2020), the mechanism is inconclusive by construction and the
     branch is recorded NEGATIVE/inconclusive without a trading-level
     verdict, regardless of any backtest number.
   - **The classifier carries no signal.** If the fitted logistic
     coefficients' Wald z-scores stay under 1.0 in magnitude for a majority
     of refits in the training period, the model is statistically
     indistinguishable from its own ridge prior (i.e., from doing nothing) —
     same clause A treatment.
   - **Unmatched risk (R-33).** If either branch's realized volatility or
     average notional differs materially from `v4`-alone's, the comparison
     is voided per this ledger's standing rule, not scored as a win.
   - **Trading-level falsification (if clause A is cleared):** the branch is
     NEGATIVE unless its paired-bootstrap 95% CI for Δlog-growth vs.
     `v4`-alone on **inner-validation** (2021-01-01 → 2022-12-31, the
     selection slice) excludes zero on the winning side, on **both** BTC
     markets (spot and futures_5x), risk-matched. This is the promotion-bar
     gate that decides whether either branch is even worth a holdout read;
     per Step 4, the holdout is not touched unless a branch clears this.

## Step 2 — design

**Mechanism, conservative branch — literal meta-labeling (bet/no-bet).**
At every bar where `v4`'s own deadband logic would change the held
position, only take the change if the walk-forward classifier's current
probability estimate clears a fixed threshold (default 0.50); otherwise
hold the previous position and re-test the same candidate change on the
next bar (a vetoed trade is deferred, not discarded). This is the textbook
López de Prado bet-sizing rule in its simplest (binary) form.

**Mechanism, novel branch — calibrated continuous sizing.** Rather than a
binary veto, multiply `v4`'s desired exposure by a sigmoid function of the
same probability (Joubert/Barziy/Meyer's "optimal sigmoid" position-sizing
family): `multiplier = clip(1 + steepness*(p - 0.5), 0, cap)`. Confidence
above the neutral prior *increases* exposure beyond what the vote alone
would size, and confidence below it shrinks exposure continuously —
matching this project's own standing lesson ("decide *how much* to hold")
rather than adding a fourth binary gate to the fourteen ERR-axis rounds
already in this table.

**Falsification tests:** as named in Step 1, question 4, frozen before
either branch is implemented.

**Shared, read-only engine** (`r179_shared.py`, neither branch may edit):
`vote_frac` and `conditional_scale` (verbatim reproductions of `v4`'s own
two factors, so neither branch re-derives them differently), daily
triple-barrier labeling, a pure-numpy (no scipy/sklearn in this
environment — confirmed, R-118/R-125's finding still holds) ridge-penalized
Newton-Raphson logistic regression, and the walk-forward
fit/purge/embargo/predict loop producing one probability-per-bar array,
forward-filled from each daily checkpoint and never using a checkpoint's
own not-yet-resolved label.

Configs evaluated by this file: 0 (shared infrastructure only, per
R-163/R-168/R-178's convention — each branch counts and reports its own).
