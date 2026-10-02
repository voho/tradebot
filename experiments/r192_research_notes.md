# R-192 research notes — five targeted changes to Kelly v4

Prepared 2 October 2026, before financial evaluation. The user explicitly
requested five updates and improvements to the top strategy. The parent is
`kelly_regime_v4`, the first row of the accepted README comparison. These are
five candidate improvements, not five established improvements. None adds an
information source or establishes a new equilibrium, regret, error-control,
or optimal-execution guarantee. The frozen protocol owns evaluation details.

## Parent and scope

`src/tradebot/strategies/kelly_regime_v4.py` changes v3's anchors to 20/40/80
days. `kelly_regime_v3.py` computes a mean-anchor vote with a symmetric 1%
latched band. Lagged eight-day EWM return volatility and its 180-day slow
anchor determine the normal/high/low volatility state. Exposure is vote
fraction times either `0.55 / slow_vol` or `0.55 / fast_vol`, capped at 2.
The strategy's 0.10 equity-notional target deadband is separate from the
broker's market-scaled execution deadband. The inherited callback submits
only when the prepared target changes; a requested order need not fill.

R-62 isolated the vote as the surviving signature. R-33 cautions that lower
drawdown can simply mean lower exposure. R-57 limits cross-asset claims.
R-190's uniform four-hour actual-account rebalance bands failed; R-191's
standalone daily cost/error-control strategies failed. The present request
authorizes focused retries with their differences disclosed. It does not
turn repeated access to the old holdout into fresh evidence.

Each candidate changes one specified control. The other parent calculations,
leverage caps, next-open execution, broker minimums, fees and funding remain
part of its evaluation. These are single-instrument long/cash or bounded
long-exposure strategies, not market makers. No OHLCV variable is relabeled
as order flow. The current data cannot reproduce the order-book games in
`docs/GAME_THEORY_RESEARCH_2026_10.md`.

## Candidate specifications and falsifiers

### 1. Accumulated target-distance budget

Let `d_t = frac_t * scale_t` be the parent's raw desired exposure and `p`
the candidate's last accepted prepared target. With `g_t = d_t - p`, reset
the accumulator when the gap is zero or its sign reverses, then add
`g_t**2 / 288` for the current five-minute observation. Adopt `d_t` and reset the
accumulator when it reaches **0.01**, with neighbors **0.005 and 0.02**.
There is no special flat-target override. This replaces the instantaneous
signal-target trigger; it does not integrate error against the filled broker
account. The units are squared equity exposure times observed-bar days,
not dollars or expected profit. Gaps in observations add no fictitious bars.

The mechanism allows a persistent small target discrepancy to accumulate,
while brief reversals erase the pending discrepancy. A fixed 0.10 gap needs
one observed-bar day at the primary threshold; a fixed 0.50 gap needs 0.04
days. A zero threshold is an explicit disabled/native-identity branch, not
the unguarded limiting formula.

Nearest prior work: L-05/L-06 use instantaneous fee/expected-gain bands;
R-64/R-165 change trade destination or smooth its rate; R-131/R-133 constrain
trailing turnover; R-190 samples actual-account deviation on a fixed clock.
This candidate changes a stateful target-tracking trigger. It remains a
cost-control retry, not proof that those negative findings are overturned.

Reject if reduced activity fails to improve net paired results, its gains
are exposure changes alone, it is absorbed by the broker, or delayed exits
increase losses more than avoided transactions save. Delayed zero targets
are intended here and must not be silently patched after results appear.

### 2. Vote-immediate, clock-scheduled scale refresh

Retain the parent's raw `frac` and `scale`. Apply the native 0.10 target
latch on a **24-hour UTC grid**, with neighbors **12 and 48 hours**, and
immediately whenever the aggregate vote fraction changes. At an eligible
event, use the current `frac * scale`. Between events retain the prepared
target. A volatility-state change alone does not bypass the clock. The
grid is aligned to Unix epoch hours. The disabled setting evaluates every
bar and must reproduce native v4.

R-190 delays every target to the same four-hour schedule; this candidate
retains immediate directional-vote events. R-165 changes the scale's
destination/rate rather than selecting target-update events. R-186 averages
phase offsets, whereas this candidate uses one fixed UTC phase. No optimal
clock or learned signal-decay claim is made.

Reject if scale-only transactions were not a meaningful cost source,
volatility-state changes require faster action, clock phase dominates the
result, or improvements disappear at matched risk and after funding.

### 3. Accumulated anchor-excursion confirmation

Keep the parent's arithmetic 20/40/80-day anchors and 1% bands. For each
anchor independently, while its current state is bearish and price exceeds
`anchor * 1.01`, add `(close / anchor - 1.01) / 288` to its pending-entry
area. While bullish and price is below `anchor * 0.99`, add
`(0.99 - close / anchor) / 288` to its pending-exit area. Reset whenever the
bar no longer requests the opposite state. Flip only once the area reaches
**0.01**, with neighbors **0.005 and 0.02**, then reset. Other v4 operations
are unchanged. Threshold zero restores the native instantaneous latch.

The primary threshold needs one observed-bar day at an excess distance of
1% beyond the existing band, or ten days at 0.1%. There is no maximum-age
override. The accumulator is deterministic path confirmation; it is not a
probability, significance test, e-value, or estimate of independent samples.

R-89 varies band geometry and response shape; R-114 uses regime-age hazard;
R-160/R-174/R-181 use statistical evidence to gate transitions. This tests
cumulative distance beyond the unchanged boundary. The known delay-versus-
whipsaw tradeoff is still present and is a reason for skepticism.

Reject if it primarily delays crash exits or misses recoveries, becomes
inert near the boundary, or appears better only because it holds less risk.
Compare its changed vote episodes and delay distribution with the parent.

### 4. Robust volatility-state classifier only

Use causally clipped returns to compute the fast/slow volatility ratio that
selects the parent's normal/high/low state. Clip parameter is **6** times
the strictly previous raw eight-day EWM return standard deviation, with
neighbors **4 and 8**. Where that reference is unavailable or nonpositive,
retain the raw log return. Apply the parent's eight-day EWM standard
deviation and one-bar lag to the clipped returns, and its 180-day slow EWM
to that robust fast volatility, to form the classifier ratio. Keep
raw-return volatility in both sizing denominators; clipping must
not quietly increase leverage by lowering the volatility used for sizing.
The explicit disabled clipping setting must reproduce native v4.

R-08/R-09/R-136/R-175 replace volatility estimators feeding sizing; R-87
changes the scale dispersion estimator; R-102 separates signed variation.
This focused retry changes only which risk state chooses between the two
unchanged raw sizing values. It can still change exposure through state
selection, so raw-denominator identity does not imply risk matching.

Reject if clipped shocks contain the information needed for risk reduction,
classification is effectively unchanged, or the apparent result is only a
different leverage distribution. Measure state differences and keep actual
returns, fees and funding unmodified in the financial ledger.

### 5. Persistent risk-increase coalescer

Start from the parent's complete prepared target. Accept decreases
immediately. Require an above-accepted-target run of **12 bars** before
adopting the latest increased parent target; neighbors are **6 and 24 bars**.
Changes within that above-target run may coalesce into its latest value.
Reset the run when its condition ceases or a target is accepted. The disabled
one-bar setting must reproduce native v4. An unchanged or withdrawn parent
target at or below the accepted target cancels confirmation; a changed
parent target still above the accepted target does not reset the count.

R-174/R-181 also distinguish increases from decreases, but require sequential
statistical evidence; this requires a fixed persistence duration and claims
no confidence level. R-131/R-133 impose a turnover resource constraint;
R-165 changes adjustment rate. This is a deliberately bounded control retry
of an already difficult asymmetric-delay idea, not a new information channel.

Reject if the delay merely shifts identical fills, suppresses profitable
recoveries, increases effective fill cost, is canceled by broker granularity,
or loses its improvement against a matched-risk parent/control.

## Sources and their limits

1. Bongaerts, Kang and van Dijk, *Conditional Volatility Targeting*,
   Financial Analysts Journal 76(4), 54–71 (2020), first online 4 September.
   [Primary publisher summary](https://rpc.cfainstitute.org/research/financial-analysts-journal/2020/0015198x-2020-1790853)
   and [author-university record](https://repub.eur.nl/pub/130215).
   Motivates the inherited extreme-volatility-state architecture. Its
   equity-market and momentum-factor evidence does not establish these
   candidate thresholds, clipping rule or crypto performance.
2. de Lataillade and Chaouki, *Equations and Shape of the Optimal Band
   Strategy*, arXiv:2003.04646, first 10 March 2020, revision 17 March.
   [Primary record](https://arxiv.org/abs/2003.04646).
   Studies linear trading costs, a price predictor and quadratic risk, with
   explicit results for an Ornstein–Uhlenbeck predictor. It explains why
   transaction costs can create inaction regions. Our accumulators and
   fixed clocks are heuristics, not solutions of that model.
3. Gârleanu and Pedersen, *Dynamic Trading with Predictable Returns and
   Transaction Costs*, Journal of Finance 68(6), 2309–2340 (2013), first
   online 26 July. [Publisher](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080).
   Provides context for separating target formation from trading toward it.
   Its signal-decay/partial-adjustment result and commodity-futures evidence
   do not prove that any R-192 rule is optimal or profitable under the
   repository's proportional-fee model.

Sources were checked for this note. One inherited citation was deliberately
not propagated: R-165 associates arXiv:1607.06373 with a Dao trend-following
paper, but that identifier resolves to *Systemic Risk and Stochastic Games
with Delay*. This note does not rely on that link or alter historic records.

## Data, accounting and interpretation

All five are feasible with existing OHLCV and timestamps; funded evaluation
also requires the already available venue-matched funding series. The
proposed evaluation retains the previous explicit Bitstamp BTC, Coinbase
ETH and Deribit BTC data rather than silently selecting the user's separate
one-year CSVs. The protocol determines dates and cost scenarios. Historical
application of today's entry fee tier is a scenario, not a reconstructed
historical account tier. The generic linear-margin simulator remains an
approximation to Deribit's inverse contract/funding mechanics.

Use the native parent's same-date results, identical costs and funding,
actual fills and exposure, paired uncertainty and valid risk matches.
Neighbor parameters are fixed before financial results; they are not a
grid from which to pick the best holdout. Each disabled-mechanism identity,
prefix/future-perturbation check and independent accounting reproduction
must distinguish prepared targets, submitted orders and actual fills.
Do not turn a causal target-identity check into a financial evaluation.

The research phase that produced this note ran no financial evaluation.
Any later evaluation and any repeated cell must be recorded separately by
the protocol's trial accounting. A candidate is an improvement only after
the frozen decision rule says so; a negative result remains a completed test
of the user's requested modification.
