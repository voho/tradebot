# R-194 frozen design — three Kelly v4 x transformer hybrids

Written on 2026-11-18 before any R-194 financial evaluation, after R-193's
holdout had been read.

- **Origin:** operator-requested ("combine the best kelly-regime with
  transformer architecture", "be creative"), not a backlog item. The live
  backlog is unchanged: B-06 ongoing, B-09 low, B-17 partial, B-28
  data-blocked.
- **Prior rounds:** R-193 is complete (NEGATIVE); there is no undispatched
  pre-registration.
- **References:** sources, constraints (R-62, R-73, R-33), prior-round
  overlap and named failure modes are in `experiments/r194_research_notes.md`.
- **Numbers seen so far:** synthetic identity, causality and optimiser checks
  only.

## Mechanisms (fixed before training; no search)

Implementation: `experiments/r194_strategies.py`.

**Common to all three.** Each subclasses `KellyRegimeV4` and keeps:

- its anchors, band and conditional volatility scale, reconstructed by R-192's
  `_parent_features`;
- its native strict 10% target latch (R-192 `_latch`);
- its target-change `order_notional` callback;
- its 80-day warmup.

Transformer components run on R-191's complete-UTC-day clock: a day's value
applies from that day's 23:55 close bar onward. Each component is exactly the
identity until it has enough matured history. Every hybrid's target is capped
at v4's `max_leverage` = 2.

1. **`r194_panel_cross_attention`.**
   - **Memory:** R-193 tokens and matured 20-day outcomes from
     `PANEL_FILES` (nine committed instruments). Complete-day closes are cut
     at the evaluated frame's last timestamp, so training sees nothing after
     2022.
   - **Readability:** a memory is readable once the date of its asset's
     (j+20)-th observation is ≤ today. At least 100 readable memories are
     required.
   - **Attention:** keys are standardised by the moments of the readable
     memories; scaled dot-product attention with key-norm bias, τ = 1, as in
     R-193.
   - **Vote:** 1 if R-193's arithmetic drift (`ū·σ60/√20 + ½σ60²`) is > 0,
     else 0. `frac = (3·frac + w·vote)/(3 + w)`, where `frac` is v4's
     3-anchor mean; plain v4 while no vote exists. w = 1; neighbours 0.5 and 2.
2. **`r194_kelly_episodic_memory`.**
   - **Keys:** daily R-193 7 features + v4's vote fraction + v4's scale (at
     the day's close bar), standardised by moments of all observed keys
     through t.
   - **Values:** matured outcomes, at least 100 of them.
   - **Retrieval:** attention weights a (τ) give μ = a·u and σ² = a·u² − μ².
     The unconditional values μ0 and σ0² are the uniform mean and population
     variance of the same matured u.
   - **Multiplier:** clip((μ/σ²)/(μ0/σ0²), 0, 2) when σ² > 0, σ0² > 0 and
     μ0 > 0; otherwise 1. Target = latch(min(v4 desired × multiplier, 2)).
     τ = 1; neighbours 0.5 and 2.
3. **`r194_residual_transformer`.**
   - **Architecture:** R-193's folded one-layer encoder on 9 features (7
     tokens + v4 vote + v4 scale), d = 8, 2 heads, FFN 16, 617 parameters.
     The output head (`v`, `c`) is initialised to exactly 0.
   - **Correction:** `exp(tanh(z))`.
   - **Training:** antithetic ES, 24 pairs, σ 0.05, 200 Adam steps (lr 0.02),
     init N(0, 0.1²) except the head, seed `194 + refit year`. The objective
     is the mean over prior days of `log1p(max(w·R_{t+1}, −.99)) −
     0.0041·|Δw| − 1e-3·mean(θ²)`, with `w = min(v4 daily target ×
     correction, 2)`; v4's daily target is `min(frac·scale, 2)` at the close
     bar.
   - **Schedule:** tokens are standardised with moments through the refit
     day and clipped ±5. Refits happen on the first observed day of each
     calendar year, or as soon as 252 rows exist with no model. The
     correction is 1 until the first fit.
   - **Target:** latch(min(v4 desired × correction, 2)). L = 30; neighbours
     15 and 60.

## Evaluation, matching and inference: R-192, unchanged

Reuse R-192's harness: `r194_eval.py`, `r194_matched.py` and
`r194_report.py` are R-192's files with names and paths substituted.

- **Evaluator:** `r190_eval.evaluate` under R-192's common initial-target
  convention. Every candidate and v4 get the same explicit positive-target
  entry on the first globally eligible bar.
- **Cells:** R-192's cells and seed-192 windows. Train input is cut before
  2023; inner train ends 2020-12-31, validation is 2021–2022, holdout starts
  2023-01-01.
- **Data:** Bitstamp BTC spot, Coinbase ETH, and Deribit perpetual with
  funding to 2026-08-12 00:40 UTC.
- **Costs:** spot 40bp + 1bp with a $10 minimum; a 10bp discount BTC cell;
  ETH 40bp stress; Deribit 5bp + 1bp plus funding.
- **Counts:** three primaries and both controls (unchanged v4;
  `buy_and_hold`, with an unlevered 1x funded passive) get 3 training cells,
  4 named holdout cells and 48 windows. Six neighbours get the 7 named
  cells. **317 core evaluations: 33 training, 284 holdout.** Controls are
  rerun; agreement with R-192's control receipts is an audit check only.

**Risk matching and inference**, exactly as R-192:

- A passive constant-exposure hold is fitted per primary and cell (start c =
  0.5, ≤ 3 attempts, 2% tolerance), giving 52 comparisons per primary.
- The parent comparison requires daily volatility within 5% of v4's.
- Paired 30-day stationary block bootstrap, 2,000 draws, seed 192, for
  Sharpe, log growth and daily drawdown, against both the parent and the
  matched hold.
- Training-only power for +0.20 Sharpe and 2pp drawdown is reported. It is
  not used to lower anything.

**Deflated Sharpe** uses 9 local trials and program trials 4,850
(cumulative after R-193) plus every R-194 holdout evaluation. Trial SD =
max(0.418538, SD of the 9 spot validation Sharpes).

**Freeze:** hash all of `src/tradebot`, the R-194 files, the imported R-190
through R-193 modules, both R-194 test files, the evaluation data and all
nine panel files, before any holdout cell.

## Exhaustive decision rule (R-192's, verbatim)

PROMOTED only when all common gates **and** at least one complete route
pass; otherwise NEGATIVE.

- **Common gates:**
  - valid parent and passive risk matches in all four primary
    validation/holdout comparisons;
  - at least one executed fill differs from the parent in each validation
    market (an inert change fails);
  - all four named holdout balances > $1,000, with no liquidation in the
    family;
  - a profitable neighbourhood with Sharpe spread ≤ 0.20 in every primary
    cell;
  - program DSR ≥ 0.95 on both primary holdouts.
- **Growth route:**
  - Sharpe gains strictly > +0.20 against both parent and matched passive,
    on validation and holdout;
  - positive lower 95% bounds for Sharpe and log growth against both, on
    both primary holdouts;
  - both primary holdout balances beat full passive;
  - ETH Sharpe ≥ both parent and passive;
  - ≥ 13/24 windows per market with valid risk and higher growth than both.
- **Tail route:**
  - daily drawdown cut by > 2pp against both, on validation and holdout;
  - drawdown-difference upper CI < 0 and log-growth lower CI ≥ 0 against
    both;
  - ETH drawdown lower with no growth loss against both parent and full
    passive;
  - ≥ 13/24 windows per market with a > 2pp drawdown cut and no growth loss
    against both.

R-192's rule partitions the outcome space: each primary is decided
independently, and "all common gates and a complete route" either holds or
does not. There is no post-result reinterpretation. A negative stays in
`experiments/`, unregistered. At operator request, results are shown in a
README "evaluated, not registered" table, as for R-193.

## Independence and finishing

R-192's four-cell financial replay is replaced by R-193's procedure. An
independent reviewer that has seen no result reads the code for lookahead and
spec mismatches before the holdout. Bug fixes only are applied (and recorded),
with any affected training cells rerun before freezing. All tests must pass,
including strict causality and the identity tests. Then record the ledger,
counters and backlog. At operator instruction, merge directly to `main` and
push.
