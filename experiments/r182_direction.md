# R-182 pre-registration (08-29) — a core-satellite venue decomposition
of `kelly_regime_v4`'s target exposure (COST axis)

Written and committed by the operator BEFORE either branch is dispatched,
per ROUTINE.md's Step 0 collision-avoidance convention. The shared,
self-tested engine is `experiments/r182_shared.py` — every number quoted
below was produced by running that file against the real, committed
BTC 5-minute dataset (`data/btcusd_spot_5m.csv.gz` + `load_funding_extended`),
inner-train only (`2017-01-01` → `2020-12-31`), and is reproducible by
re-running `python experiments/r182_shared.py`.

## Step 0 — already satisfied

Confirmed by the orchestrating session before this round started:
`HEAD == origin/main`, no undispatched `r<nn>_shared.py`, 1 consecutive
null pass since the last dispatched round ("0–2: normal" per ROUTINE.md
Step 0b), and the backlog grep returns only **B-48** (a
documentation/formatting instrument fix, not a strategy-research item).
A fresh, off-backlog literature-prompted direction is therefore in scope,
the same posture R-160 through R-181 used.

## Step 1 — selection

**Idea, one sentence:** split `kelly_regime_v4`'s own, unmodified target
exposure into a slow-moving BASE held on unlevered (funding-free) spot
and a fast, SIGNED deviation held on leveraged futures, so nearly all of
v4's turnover pays futures' lower 0.05% taker fee instead of spot's
0.10%/0.40%, while funding is charged only on the smaller deviation
notional rather than gross exposure — a different SPLIT FUNCTION from
R-145/R-151/R-154's already-closed threshold split, targeting the exact
number (a 0.931 extra-fee/funding-saved ratio at the real 0.40% tier)
that closed that family.

1. **Which constraint does it attack?** **COST** — the standing
   diagnosis's own line, "costs scale with the signal — funding runs
   +20%/yr while the strategy holds." This changes neither the signal
   nor the decision (aggregate exposure is v4's own `target`, identical
   at every bar, by construction — verified in `r182_shared.py`'s own
   degenerate-equivalence self-tests, not merely asserted), only which
   venue executes it — the same "instrument choice, not retiming or
   throttling" framing R-145 used, satisfying R-131/R-133's explicit
   rule that a COST-axis attack "has to change the DECISION, not the
   order that follows it" (this changes neither; it changes execution).

2. **Which ledger entries is it not a duplicate of?**
   - **R-145 / R-151 / R-154** (threshold split
     `spot=min(target,thr)`, `fut=max(target-thr,0)`, both legs always
     `>=0`, futures near-idle at the primary threshold — 11 fills over 2
     years). R-154's own fully-bug-fixed harness (B-45 overflow
     redistribution + B-46 joint-move gating, both adopted) still failed
     R-145's own gate at the 0.40% tier with an extra-fee/funding-saved
     ratio of **0.931**, and its closing line calls this strategy's
     venue-routing research space "exhausted." This round's split
     function is different in kind, not degree: BASE tracks a
     slow-moving LEVEL (constant or long-halflife EWMA of `target`), not
     a fixed THRESHOLD on `target`'s own instantaneous value — the
     futures leg here is the ACTIVE, signed leg carrying nearly all of
     v4's turnover, and the spot leg is the near-idle one; R-145's
     construction has it backwards (spot active below the threshold,
     futures idle above it). This is checked explicitly, not assumed:
     `r182_shared.py`'s own measured extra-fee/funding-saved ratios
     (Step 2, below) are 0.048–1.190 across four BASE candidates —
     materially different numbers than R-154's single 0.931, confirming
     the split function's behaviour genuinely differs, not merely its
     label.
   - **R-64** (Gârleanu-Pedersen partial adjustment on the WHOLE
     product, weighted by the 3 VOTE anchors' own decay rates — single
     venue throughout, no venue split; failed because the anchors' decay
     rates are too similar for GP's weight formula to produce
     heterogeneity, a failure mode about anchor decay that does not
     apply here since this round's split point is not a GP weight on
     anchor decay at all).
   - **R-165** (destination/rate axis isolated onto SCALE alone — a
     no-trade region or EWMA-derived rate applied to v4's `scale`
     VALUE, changing what v4 targets). This round leaves `target`
     completely unmodified; only execution venue changes.
   - **R-131 / R-133** (turnover corridor/shrink throttling an
     already-decided single-venue re-target). This round does not
     throttle anything — every bar's aggregate exposure is identical to
     v4's own decision, at full size, immediately.
   - **R-173** (Roll/Corwin-Schultz illiquidity re-pricing and a
     spread-conditioned deadband width — single-venue audit/throttle, no
     venue split).
   - **R-176** (dollar-volume activity-clock resampling of the VOTE, or
     a dollar-bar crowding gate — single instrument, no venue split).
   - **R-178** (a synthetic DVOL-priced options structure ADDED on top
     of v4's existing futures position — a third, additive position,
     not a re-routing of the SAME position across two venues).

3. **Is it simulable here?** Yes, with one new, disjoint,
   experiment-only capability. 5m OHLCV bars, bar-close signals,
   next-open fills, no order book, no queue model — everything needed is
   price, volume (already read by `compute_target`) and the already-
   committed BTC funding series (`load_funding_extended`). The one gap:
   R-145/R-151/R-154's own `HybridBroker` family keeps both legs
   long-only by construction (their `_execute_leg` raises `ValueError`
   on any negative `frac`) because their split never needs a leg to go
   negative. This round's BASE can sit above the instantaneous `target`
   (v4 dips below its own recent/typical level often), so the deviation
   leg must be able to go short — a pure REPLICATION/accounting
   technique (the aggregate is still exactly v4's own non-negative
   `target`), never a new directional bet, and not an order-book or
   queue-model capability. `SignedHybridBroker` in `r182_shared.py` is a
   small, disjoint generalization of the SAME, already-validated two-leg
   mechanics (`_transact_leg`'s fee/PnL arithmetic reused verbatim);
   every new code path is self-tested against trusted oracles before
   being relied on (below).

4. **What would make it fail?** Named now, before either branch writes
   promotion-relevant code:
   - the futures leg's `check_liquidation` fires on inner-train or
     inner-validation (an added, this-round-specific risk control — see
     Step 2's decision rule, clause 5);
   - the extra-fee/funding-saved ratio does not fall and stay below
     R-145's own 0.50 kill bar at the real 0.40% spot tier;
   - R-33 risk-matching fails (time-in-market or realized volatility
     mismatched beyond 1% against plain futures v4);
   - the BTC `d_sharpe` 95% paired-bootstrap CI (hybrid vs. plain
     futures v4) contains zero, or the point estimate is negative, at
     either fee tier — **already measured as the modal outcome on
     inner-train for the naive construction, disclosed in Step 2 below,
     not hidden until a branch reports it.**

## Step 2 — design, citations, and what was measured before any branch ran

**Citations.**
- **Ackerer, D., Hugonnier, J. & Jermann, U. (2024/2025 working paper),
  "Perpetual Futures Pricing"** — an arbitrage-free perpetual is priced
  as a continuously-refinanced spot replication with the funding rate as
  the financing leg; the formal basis (R-145's own citation, reused) for
  treating venue choice as a pure financing decision that leaves the
  target itself untouched. No instrument-count or cost-assumption claim
  to carry over — this is a pricing-identity paper, not an empirical
  backtest, cited only for the financing-decision framing.
- **Dao, C., Nguyen, N.C.P., Sadka, R. & Zhang, S. (2016)** and
  **Gârleanu, N. & Pedersen, L.H. (2013, *J. Finance* 68(6))** —
  decay/EWMA-derived trading rates; R-65/R-67's own citations, reused
  here (per R-165's own precedent) not to smooth a SIGNAL's value but to
  derive a slow-tracking BASE LEVEL for the venue split — a genuinely
  different application of the same "derive a rate from the object's
  own measured persistence" principle. R-65 measured this exact
  methodology moving a DIFFERENT object (the multi-asset panel's
  cross-sectional score) from a 3.44/day to a 0.19/day turnover rate and
  from −7.54 to +0.59 log units against a volatility-matched hold, on
  BTC/ETH/6 further Coinbase instruments at this project's own real
  fee tiers — cited as the precedent for "a derived rate materially
  changes turnover economics," not as evidence this round inherits that
  result (it is a different object, tested fresh below).
- **Schmeling, M., Schrimpf, A. & Todorov, V. (2023, BIS Working Papers
  No. 1087)** — the crypto funding/carry premium is large, volatile, and
  sometimes negative; cited (as R-145 cited it) as a guardrail that this
  design may only ever AVOID a cost already being paid on an unchanged
  position, never harvest a new carry bet (that would duplicate B-03,
  already NEGATIVE per R-39).

**What was measured on real inner-train BTC data before writing this
decision rule (Step 2's mandatory "record what the data actually
shows," not after a branch runs it).**

*First attempt: an unconditional constant BASE (spot always holds
`BASE`, futures always holds `target - BASE`).* Self-measurement (not a
branch report) found this creates a synthetic position during v4's own
genuine FLAT periods (28.7% of inner-train bars: `target == 0`): spot
holds `BASE` long while futures holds `-BASE` short, netting to the
correct zero exposure as TWO OFFSETTING LEGS rather than true flat —
exactly the delta-neutral spot-long/futures-short carry construction
**B-03**, which **R-39 found NEGATIVE**, reproduced here as an
unintended side effect rather than a deliberate bet. Measured directly:
bar-level time-in-market rose from the plain futures baseline's 66.9% to
98.5%, while `target` itself is nonzero on only 71.3% of inner-train
bars. **Fixed** before this file was frozen: both route builders now
gate the base to fire only while `target[i] > 0` (`route_constant_base`,
`route_ewma_base` in `r182_shared.py`), removing the synthetic carry leg
during genuine flat periods. This is disclosed here, not silently
corrected, because it is exactly the kind of measurement-before-freezing
Step 2 exists to catch.

*Gated construction, four BASE candidates, real inner-train BTC
(2017-01-01 → 2020-12-31, 1,460 trading days), `python
experiments/r182_shared.py`:*

| BASE | extra fees / funding saved @0.10% | @0.40% | `d_sharpe` (hybrid − plain futures v4) | 95% CI |
|---|---|---|---|---|
| 0.10 | 0.171 | 0.869 | −0.030 | [−0.151, +0.076] |
| 0.15 | 0.167 | 0.805 | −0.066 | [−0.206, +0.074] |
| **0.25** | **0.048** | **0.430** | −0.193 | [−0.453, +0.035] |
| 0.43 | 0.429 | 1.190 | −0.074 | [−0.257, +0.112] |

Two findings, both real, both load-bearing for the decision rule below:

1. **The cost-ratio mechanism works as designed.** At `BASE=0.25`, the
   ratio at the real 0.40% tier is **0.430** — comfortably under R-145's
   own 0.50 kill bar and less than half of R-154's measured 0.931 for
   the threshold-split family. This is the round's core economic
   hypothesis (a near-static spot leg produces far less incremental fee
   cost than a threshold split that keeps both legs independently
   active) and it is confirmed, quantitatively, on real data — the
   specific number this round set out to move, moved.
2. **Every tested BASE nonetheless shows a NEGATIVE `d_sharpe` point
   estimate against plain futures v4 on inner-train.** The most
   plausible mechanism, named here for the implementing branches to test
   directly rather than merely assumed: v4's own entry/exit transitions
   (`target` crossing zero) are exactly the moments this gated
   construction requires BOTH legs to fire simultaneously (spot
   `0 -> BASE`, futures `0 -> target-BASE`) — i.e., the one-fill event
   plain futures v4 executes at each transition becomes a two-fill event
   here, concentrated exactly on the bars the standing diagnosis's own
   **N≈3** finding says carry most of v4's edge. A mechanism that adds
   friction preferentially at transitions is fighting the worst possible
   bars to add friction to, even while its AVERAGE fee/funding economics
   improve.

Both findings are frozen into the decision rule below rather than
adjudicated by this file — that is Step 3/4's job, on inner-validation
and the holdout respectively, not Step 2's.

**Reachable-n (ROUTINE.md's mandatory power check, against this round's
own measured noise, not an assumed order of magnitude).** Computed by
`reachable_n_report()`: real `paired_bootstrap`/`annualized_sharpe`
(this project's own standard tool, `mean_block=30`, matching R-145's own
convention) on the SAME inner-train sample above, then a block-bootstrap
CI-half-width scaling projection (`half_width ~ C/sqrt(n)`, the same
scaling law R-67/R-68 used) for how many days the CURRENT point
estimate would need to clear significance at its OWN sign. At
`BASE=0.25` (the most favourable cost-ratio candidate): point estimate
−0.193, 95% CI half-width 0.244, implying **n≈2,338 days** to clear
significance **in the negative direction** — i.e., this is not an
under-powered comparison in R-78's sense (both arms track the same
price path, so the paired difference is measured with real, not
astronomical, precision: 1,460 days already resolves a non-trivial CI
half-width of ~0.24 Sharpe), it is a comparison with a **currently
unfavourable point estimate**. Per R-78's own lesson, more data is not
the answer to a wrong-signed effect; a different construction is. This
is why Step 3 is scoped as "fix the diagnosed mechanism," not "sweep
more BASE values on the same construction" — the implementing branches
should not treat "just needs more data" as license to skip the fix
below.

## Step 3 — the two branches (frozen decision rule and falsification
tests, before either branch runs anything)

Both branches import `r182_shared.py` unedited (`SignedHybridBroker`,
`route_constant_base`/`route_ewma_base`, `risk_match_report`,
`reachable_n_report`, `run_signed_hybrid_backtest`) and must re-run its
`__main__` self-test suite (liquidation-formula oracle check,
degenerate-equivalence, signed-futures-leg oracle check, causality
truncation, liquidation safety scan) before trusting any number of their
own — exactly as R-145/R-151/R-154's own branches re-ran that file's
self-checks.

- **Conservative** (`experiments/r182_conservative.py`): the literal
  construction measured above — `route_constant_base`, BASE swept over
  `{0.10, 0.15, 0.25, 0.43}` selected against **inner-validation**
  (2021-01-01 → 2022-12-31, never inner-train, per ROUTINE.md's own
  split discipline — inner-train is calibration-only), no fix for the
  transition-doubling hypothesis. This branch exists to answer, on the
  held-out selection window: does the diagnosed cost-ratio improvement
  survive contact with a genuinely different regime (2021 top, 2022
  bear), or does the negative Sharpe-diff on inner-train simply persist?
  **Falsification test (frozen now):** at the BEST-selected BASE on
  inner-validation, if `d_sharpe` (hybrid vs. plain futures v4, 95%
  paired-bootstrap CI, both fee tiers) does not exclude zero on the
  favourable side, this branch is NEGATIVE — report it as such, exactly
  as the Step-2 measurement above already suggests is the modal outcome,
  not a surprise to explain away.

- **Novel** (`experiments/r182_novel.py`): targets the diagnosed
  transition-doubling mechanism directly. `route_ewma_base`, but with
  the `target[i] > 0` gate replaced by a **persistence/hysteresis**
  gate — the spot base stays "on" as long as v4 has been in-market at
  any point within a trailing window `H` (swept over `{1, 3, 7, 14}`
  days), turning off only after a sustained absence, rather than
  flickering on every single zero-crossing bar. This targets the
  hypothesis precisely: brief in/out excursions (the ones concentrated
  at transitions) no longer force a simultaneous two-leg move, while a
  genuinely extended flat period still zeroes the spot leg out (so the
  B-03 carry-bet problem the Step-2 fix already solved stays solved,
  not reintroduced). **Falsification test (frozen now):** compute the
  SAME `d_sharpe` comparison on inner-validation, both fee tiers; if the
  hysteresis fix does not move the point estimate from negative to
  positive on at least the primary (median-selected) `H`, AND if it does
  not reduce the number of bars on which spot and futures both fire in
  the same direction at a `target`-zero-crossing (a directly measurable
  diagnostic for the named mechanism, not merely inferred from the
  headline number), this branch is NEGATIVE — the specific,
  named-in-advance outcome that would mean the transition-doubling
  hypothesis was wrong, not merely that this particular fix for it
  failed.

**Promotion bar (both branches, reused from R-145's own frozen rule,
plus this round's new clause 5):**

1. `d_sharpe` (hybrid vs. plain `kelly_regime_v4` on `futures_5x`) `>=
   +0.20` on BTC, 95% CI excluding zero, at BOTH fee tiers.
2. extra-fee/funding-saved ratio `< 0.50` at BOTH fee tiers.
3. time-in-market and realized volatility matched within 1% of plain
   futures v4.
4. ETH mechanism check: `spot_frac + fut_frac == target` to floating
   tolerance, no lookahead/liquidation bug (ETH is funding-free — no ETH
   perpetual funding series is committed in this repo, the same R-145
   ceiling — so ETH is never the dollar-savings gate).
5. **(new)** the futures leg's `check_liquidation` never fires on
   inner-train or inner-validation.

Kill bar: (2) fails at either tier, OR any BTC CI in (1) contains zero,
OR (5) trips, OR a branch's own self-tests fail. Default REJECT. Holdout
(`>= 2023-01-01`) is not read by either branch unless it clears (1)-(5)
on inner-validation, at which point Step 4 pre-registers a fresh holdout
consultation exactly as every prior COST-axis round in this ledger has.

**Configs evaluated so far (this file): 0 promotion-relevant backtest
cells** — every number quoted above is a Step-2 measurement used to set
the decision rule, not a promotion-relevant read. `n=8` backtests
(4 BASE × 2 fee tiers) were run for the cost-ratio table, and `n=4`
`paired_bootstrap` calls for the reachable-n table — declared here for
the program-level trials count per ROUTINE.md's parallelism rule, since
they used real inner-train data even though none of them is itself a
promotion-relevant read.

## What this round's engine self-tested (see `r182_shared.py`'s own
`__main__` block; reproducible via `python experiments/r182_shared.py`)

- **Liquidation-formula check**: `SignedHybridBroker.liquidation_price()`
  reproduces `PaperBroker.liquidation_price()` (this project's own,
  already-extensively-tested oracle) to `<1e-9` relative, across 6
  fixtures spanning both signs of `pos_fut` and three position
  magnitudes.
- **Degenerate equivalence**: the all-futures route (`base=0`)
  reproduces plain futures `kelly_regime_v4` to `0.00e+00`; the all-spot
  route reproduces plain spot `kelly_regime_v4` to `0.00e+00` — both
  under a **per-leg** deadband base, a deliberate departure from
  R-151/154's "shared reference" B-44 fix, justified in
  `SignedHybridBroker`'s own docstring and verified here directly (the
  shared-reference convention was tried first and measured to break the
  all-spot degenerate case by 5.5% relative — disclosed, not hidden).
- **Signed-futures-leg oracle check**: a synthetic SIGNED target array
  (v4's real target minus 0.35, clipped to `[-0.35, +0.60]`, spot leg
  held at zero throughout) run through `SignedHybridBroker` reproduces
  the REAL, already-trusted `tradebot.engine.run_backtest` /
  `PaperBroker` path (via a minimal oracle `Strategy` wrapper,
  `ctx.order_notional`) to `0.00e+00` on final balance, fees paid AND
  funding paid, with identical liquidation outcomes — the decisive
  validation of both new code paths this round needed (sign-flip
  close/reopen, signed liquidation formula) against the project's most
  trusted broker, not a hand-derived expectation.
- **Causality truncation**: truncating the tail by one bar changes no
  bar strictly before the cut (max diff `0.00e+00` over 209,952 shared
  bars).
- **Liquidation safety scan**: all 4 constant-BASE and 3 EWMA-halflife
  candidates, both inner-train and inner-validation — zero liquidations
  in every cell (the short leg is mathematically bounded at magnitude
  `BASE` under the constant construction, since `target ∈ [0,
  max_leverage]`, comfortably inside 5x headroom).

Nothing above is a promotion-relevant claim; it is the capability this
round needed built and proven correct before either branch is allowed to
report a number from it.
