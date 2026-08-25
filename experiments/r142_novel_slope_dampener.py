"""R-142 NOVEL branch: a continuous, TWO-SIDED SIZE-axis exposure
multiplier on `kelly_regime_v4`'s own `scale`, driven by the Deribit
front-vs-next-quarter futures term-structure SLOPE (the same signal the
sibling CONSERVATIVE branch tests as an INFO-axis confirming vote --
these two branches share `experiments/r142_shared.py` and nothing else;
neither edits it, and this file does not read the conservative branch's
Step-A gate result before running its own Step-0/B1-B5 battery).

=====================================================================
PRE-REGISTRATION (frozen before any backtest number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). Modulate `kelly_regime_v4`'s existing
   `scale` by a bounded, two-sided function of the term-structure slope's
   own z-score: damp exposure when the curve is steeply, richly
   contangoed (a crowded, blown-off-looking curve shape -- Bianchi, Fan,
   Miffre & Zhang 2023's own finding that slope-momentum profitability
   "increases with investor sentiment" reads this direction as
   euphoria-adjacent), and modestly AMPLIFY exposure when the curve is
   deeply inverted/backwardated (a capitulation-adjacent curve shape --
   CF Benchmarks 2025, "Revisiting the Bitcoin Basis", and contemporaneous
   reporting on 2025-11 BTC backwardation preceding a local bottom; this
   specific 2025-11 episode is NOT used anywhere in this branch's gates,
   per the discipline note in r142_shared.py's module docstring, only as
   motivating color found during this round's own literature search).

   FORMULA: `scale_novel = scale_v4 * (1 - kappa * tanh(slope_z / 2))`,
   `slope_z` from `r142_shared.slope_zscore` (20-day causal rolling
   window, identical construction to the conservative branch's Step-A
   feature). `tanh(slope_z/2)` bounds the multiplier's own input to
   (-1, 1) regardless of how extreme a single `slope_z` print is (this
   round's own Step-0 data check found raw annualized-slope outliers as
   large as +23/-15 on rare near-expiry ticks; `tanh` prevents one such
   tick from producing an extreme, economically meaningless exposure
   swing -- a numerical-stability choice made before any backtest number,
   not a fitted clip).

   THIS IS DELIBERATELY TWO-SIDED (damp ranges over `[1-kappa, 1+kappa]`,
   both directions reachable, not bounded only above or only below 1) and
   DELIBERATELY NOT CALIBRATED BY EQUALITY MEAN-EXPOSURE MATCHING.
   R-141's own LPPLS dampener (`scale_novel = scale_v4 * max(0.1, 1 -
   kappa*confidence)`) is ONE-SIDED (damp <= 1 always), so matching
   `mean(scale_novel)` to `mean(scale_v4)` under exact equality is
   mathematically forced to kappa=0 (R-141, ruled out -- see
   docs/LEDGER.md's section C, R^2=1.000000 exactly). This construction
   avoids that trap by construction (kappa's effect is symmetric around
   1, so equality-matching does not collapse to a corner solution) --
   verified directly in Step 0 below before it is trusted, exactly as
   R-141's own post-hoc analytical proof should have been run pre-hoc.
   kappa is NOT solved for by any matching procedure at all: it is swept
   on the FIXED, PRE-REGISTERED grid `r142_shared.NOVEL_KAPPA_GRID =
   (0.0, 0.10, 0.20, 0.30)`, and every grid point is reported as a B3
   plateau cell -- none is "the" selected candidate before the battery
   runs.

   Citations: same term-structure-slope citation trail as the
   conservative branch (Bianchi, Fan, Miffre & Zhang 2023; Erb & Harvey
   2006; Schmeling, Schrimpf & Todorov 2023/2025); MacLean, Thorp & Ziemba
   (2010) on fractional-Kelly exposure shrinkage under parameter
   uncertainty, for the general shape of a bounded multiplicative
   dampener (this project's own standing justification for every
   SIZE-axis dampener since R-59).

   CONSTRAINT ATTACKED: SIZE (this is the axis that has actually worked in
   this project, per the standing diagnosis) via a genuinely new
   informational input (the curve slope) no prior SIZE-axis attempt has
   used -- 28+ prior dampeners (R-59 through R-141) all transform either
   the vote, the raw volatility estimate, or a model-derived
   confidence/hazard signal computed from spot alone.

   NOT A DUPLICATE OF: any prior SIZE-axis dampener (grepped
   docs/LEDGER.md's ruled-out table: CVaR, robust/shrinkage Kelly, CPPI,
   uncertainty-shrink, ladders, HAR-vol substitution, LPPLS crash-hazard --
   none uses a futures-curve input); R-141 specifically, distinguished
   above by sidedness and calibration method; the sibling CONSERVATIVE
   branch, which tests the same underlying slope signal as a discrete
   INFO-axis confirming VOTE inside the anchor gate, not a continuous
   SIZE-axis multiplier on `scale`.

2. STEP 0 -- MANDATORY PRE-FLIGHT CHECKS, run before any backtest number:
   (a) IDENTITY-RECOVERY: `kappa=0` must reproduce `kelly_regime_v4`'s own
       `target` array exactly (`np.array_equal`, not merely `allclose`).
   (b) NOT-DEGENERATE-BY-CONSTRUCTION: verify analytically (as R-141's own
       post-hoc proof should have been run pre-hoc) that this dampener's
       mean exposure is NOT monotonic in kappa across the pre-registered
       grid -- i.e., confirm this construction cannot fall into R-141's
       trap before trusting any downstream number. Report
       `mean(scale_novel)` at every grid point next to `mean(scale_v4)`.
   (c) CAUSAL TRUNCATION PROBE on the full `slope_z -> scale_novel`
       pipeline (`r142_shared.truncation_causality_probe`-style check),
       matching every prior round's convention.
   Any failure here stops the round before Step 3/4, same as R-141's own
   Step-0 gate stopping its dampener before B1-B5 ran.

3. STEP 3 (inner-train/inner-validation only, holdout untouched) --
   BATTERY, run for every kappa in the grid, BTC and ETH, spot and
   futures:
   - B1: full-period and inner-validation Sharpe, Sortino, max drawdown,
     turnover, vs `kelly_regime_v4` unmodified, for every kappa grid point
     (a plateau report, not a single winner).
   - B3: the plateau check itself -- report all 4 kappa cells together;
     a candidate is only taken seriously if improvement is monotonic-ish
     or flat across kappa in {0.10, 0.20, 0.30}, not a spike at one value.
   - B4: ETH sign-replication -- the BTC-selected kappa (if any looks
     promising) must not invert sign on ETH, this project's single most
     repeated failure mode (R-33/R-57/R-62/R-64/R-113/R-127/R-137 and
     others).
   - B5 / R-33 RISK-MATCHING: report time-in-market and realized
     annualized volatility for `kelly_regime_v4` and every kappa
     candidate, side by side. A candidate whose realized volatility
     diverges from v4's own by more than 15% is reported but treated as
     UNMATCHED and void for promotion purposes, per this project's
     standing "match risk before comparing anything" rule -- exposure
     divergence is not this round's own multiplier design intent (kappa
     is meant to REDISTRIBUTE risk across regimes, not change its
     average level), so a large divergence signals something is off
     rather than a real finding.

4. STEP 4 -- FALSIFICATION TEST (the one chosen now, before any holdout
   number): the 0.40% taker fee tier (`scripts/fee_study.py`'s real-cost
   convention, R-73/R-141's own choice for an analogous continuous
   dampener). A candidate is only promotable if at least one kappa > 0
   grid point still beats `kelly_regime_v4` at the 0.40% tier, on BOTH
   BTC and ETH, out-of-sample (>= OOS_START) -- the coverage-extension
   fetch this round required (see r142_shared.py) makes this a full,
   unbroken holdout for the first time this signal type has had one.

5. DECISION RULE (frozen now): promote only if ALL of docs/ROUTINE.md's
   standing promotion-bar clauses hold for at least one kappa grid point
   -- beats buy_and_hold OOS after real costs; improvement exceeds the
   +/-0.2 Sharpe noise floor OR is a genuine drawdown/tail improvement;
   survives the 0.40% fee tier; the kappa neighbourhood is a plateau
   (B3); AND passes the R-33 risk-match check (B5) without the 15%
   divergence flag. Failing any one clause is NEGATIVE, reported with the
   same care as a promotion, per docs/ROUTINE.md's own standard.

6. WHAT WOULD MAKE IT FAIL (named now): mean mechanism damps the wrong
   direction relative to buy_and_hold (a Sharpe/drawdown result worse than
   v4 alone at every non-zero kappa); or a real effect exists on BTC alone
   but inverts sign on ETH (this ledger's single most common failure
   mode); or the effect only appears at one isolated kappa rather than
   across the grid (a peak, not a plateau); or it requires an unmatched
   exposure change to show up at all (the R-33/B5 check).

CONFIGURATIONS EVALUATED: to be filled in by the implementing session,
counting every kappa-grid cell (4) x market (BTC/ETH) x period
(full/inner-val) combination in B1/B3, every B4/B5 cell, and the Step-4
fee-tier check.
"""

from __future__ import annotations
