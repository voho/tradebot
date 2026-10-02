# R192 preregistered improvements to Kelly regime v4

Written 2026-10-02 before any R192 financial evaluation. The user explicitly
requested five updates to the top strategy. The README's first-ranked strategy
is `kelly_regime_v4`; its raw futures ranking omits funding and is not proof
of superiority. This round tests five isolated modifications, with no combined
winner or post-result tuning. Registered parent defaults remain unchanged.

Step 0: origin/main remains 1fd5a57a after fetch; R188 through R191 are complete,
no undispatched registration was found. B06 is ongoing, B09 low, B17 partial,
B28 data-blocked. The requested batch targets COST/ERR, not new information.
The October game-theory literature review requires data absent here; no
order-flow, queue or opponent behavior is fabricated from candles.

## Frozen mechanisms

All variants retain v4's 20/40/80 observed-day anchors, 1% latching bands,
8-day lagged EWM volatility, 180-day slow volatility reference, conditional
volatility thresholds, 0.55 risk budget, 2x signal cap and warmup, except for
the specific change below. One observed day is 288 bars, as in the parent.
Recursive feature history before a measured window is retained causally;
each simulated account starts fresh. None of these heuristics inherits a
financial optimality or statistical coverage guarantee from a paper.

1. **r192_tracking_budget:** replace the 0.10 target-change deadband with an
   accumulated squared tracking-distance budget. At each bar let gap be raw
   desired exposure minus the currently accepted signal target. Reset the
   accumulator when gap is zero or its sign reverses; add gap squared / 288.
   When it reaches 0.01 exposure-squared days, adopt the raw desired target
   and reset. Neighbors 0.005/0.02. This uses signal targets, not actual account
   holdings or a claim about monetary tracking loss. No special flat-exit
   bypass. Persistent small differences can accumulate; large exits can lag.
2. **r192_factor_clock:** apply the original 0.10 target latch only when the
   vote fraction changes or at a UTC 24-hour boundary. Neighbors 12/48 hours,
   aligned to Unix epoch hours. At allowed decisions use the current raw
   desired target. Volatility-state changes alone receive no bypass. This
   schedules scale updates while allowing immediate vote changes; it does
   not rebalance against actual account drift as R190 did.
3. **r192_anchor_confirm:** each anchor's vote needs accumulated opposite
   excursion beyond its original 1% boundary. Add excess relative distance
   / 288 on consecutive bars beyond that boundary; reset if the breach stops.
   Flip at 0.01 return-days, neighbors 0.005/0.02. No maximum delay. Original
   risk sizing and 0.10 target latch remain. This is geometric confirmation,
   not a p-value, e-value or evidence that a new information channel exists.
4. **r192_robust_state:** clip log returns at +/-6 times strictly previous
   raw eight-day EWM standard deviation, neighbors 4/8. Use clipped-return
   fast/slow volatility solely to classify the existing volatility states.
   Use original raw full/steady volatility for the chosen position scale.
   Price anchors and the 0.10 target latch are unchanged. No-clipping identity
   must reproduce v4. Real jumps may be meaningful; clipping can delay useful
   protection and is not an error detector.
5. **r192_confirm_increase:** postprocess native v4 targets. Accept decreases
   immediately; accept an increase after 12 consecutive observed bars above
   the accepted target, neighbors 6/24. Cancel pending confirmation when the
   increase is withdrawn; after confirmation adopt the latest native target.
   One-bar confirmation must be an identity. This can reduce short-lived
   increases, but delayed entry can cost more than it saves.

The implementation must retain native next-open orders, broker deadbands,
minimum sizes, leverage/fee reserves and funding. Signal targets and actually
executed holdings are distinct; mechanism binding must be checked in fills.
Sources and nearest failed predecessors are recorded in r192_research_notes.md.

## Common controls and evaluation cells

The control uses unchanged v4 signal recurrence and subsequent callback.
Every candidate and v4 receives the same explicit initial positive-target
request on its first globally eligible bar, filled at the next open. This is
the R190 fresh-account convention, not a claim of byte-identical native
full-history startup. Identity tests cover both warmup and later windows.
No strategy-specific initial-entry advantage is allowed.

Use committed Bitstamp BTC spot, Coinbase ETH spot, Deribit BTC perpetual
prices and matching observed 8-hour funding through 2026-08-12 00:40 UTC.
Train input is cut before 2023. Inner train ends 2020-12-31; validation is
2021-2022; retrospective holdout begins 2023-01-01. Start each account at
$1,000; include first-day return against that balance.

Primary BTC spot: 40bp taker +1bp slippage and $10 minimum, following the
official Bitstamp schedule checked earlier today. BTC 10bp is a discount
sensitivity. ETH 40bp +1bp/$5 minimum is a generic stress scenario, not an
account-specific quote. Deribit uses conservative legacy 5bp +1bp/$5 minimum,
with venue-matched funding. This remains a generic linear margin/8-hour
funding model, not exact exchange contract or queue mechanics. See R191's
protocol for today's official schedule verification and its limits.

Five primaries, ten fixed neighbors and two controls receive three train
cells and four named holdout cells. Primaries and controls additionally
receive 24 seed-192 paired windows of 120-365 days on BTC spot and funded BTC.
Total core simulations: 455 = 51 train +404 holdout. No candidate is replaced
after training. Passive controls are registered spot buy-and-hold and a
rebalanced constant 1x notional funded long, not liquidating 5x buy-and-hold.
The overlapping windows describe path sensitivity, not independent samples.

## Risk comparison and uncertainty

For each primary, use unchanged v4 as a paired control and independently fit
the existing broker-backed constant-exposure passive hold. Risk matching
applies to both validation markets, both primary holdouts and each beta
window: 260 candidate/passive comparisons. Start passive c at 0.5, multiply
by target/achieved daily volatility, clip to [0.001,1] spot or [0.001,2] perp,
allow three simulations. Relative volatility error <=2% is valid; failures
remain counted and cannot earn a win. Parent comparisons require relative
daily-volatility difference <=5%. No post-result relaxation or uncounted
matching retry. Passive c is fitted ex post for diagnosis, not deployable.

Use paired stationary daily block bootstrap, 30-day mean block, 2,000 draws,
seed192, for Sharpe, log growth and daily maximum drawdown versus both
references on the two validation and two primary holdout cells. Intervals
condition on fitted passive c; fitting uncertainty is not included. Report
bar-level drawdown separately. Before freeze, measure training noise and
approximate history needed for a +0.20 Sharpe effect and a 2pp drawdown effect;
label these projections as stationary-noise approximations, particularly
for nonlinear drawdown. If underpowered, say so; do not lower thresholds.

Deflated Sharpe discloses 15 local configurations and approximately 3,251
prior consultations plus all actual holdout core, matching and audit replays.
Trial dispersion is max(0.418538, SD of 15 spot validation Sharpes).
All failed and repeated financial attempts count. Synthetic unit tests do
not count as financial trials. Freeze source, protocol, tests, audit and
data hashes before holdout. Training ranking is descriptive only.

## Exhaustive decision rule

PROMOTED only when all common gates and at least one route pass; otherwise
NEGATIVE. No default change follows a merely higher point estimate.

Common gates:

- All four primary validation/holdout comparisons have valid parent and
  passive risk matches. Each validation market shows at least one different
  executed fill versus the parent; an inert change fails.
- All four named holdout balances exceed $1,000 and neither primary nor
  its neighbors liquidates in the primary cells.
- Each three-setting neighborhood is profitable with Sharpe spread <=0.20
  in every primary validation/holdout cell.
- Program-level DSR is >=0.95 in both primary holdout markets.
- The selected route's ETH check and window check pass below.

Growth route: validation and holdout Sharpe improvements strictly exceed
+0.20 against both parent and matched passive; both primary holdout markets
have positive lower 95% bounds for Sharpe and log-growth improvements against
both. Both primary holdout balances beat full passive. ETH Sharpe is at least
both native parent and full passive. Each market has >=13/24 windows with
valid risk comparisons and higher growth than both parent and its own
matched passive.

Tail route: validation and holdout daily drawdown reductions exceed 2pp
against both parent and matched passive. Both primary holdout markets have
upper drawdown-difference CI <0 and lower log-growth-difference CI >=0 against
both. ETH daily drawdown is lower with no growth loss against both parent and
full passive. Each market has >=13/24 valid windows with >2pp daily drawdown
reduction and no log-growth loss against both references. This explicitly
tests a tail improvement without calling lower exposure an improvement.

The independent audit fixes r192_tracking_budget before results, reconstructs
its signal and quote-cash account independently on spot/funded validation
and spot/funded holdout, four counted financial replays. It fails loudly on
unsupported liquidation paths. Unit tests cover all mechanisms, identities,
prefix/future causality and startup parity. Run the full suite including
strict causality, record results and consultations in the ledger, and update
the existing review PR with all feedback addressed. Do not merge.
