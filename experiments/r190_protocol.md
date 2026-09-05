# R-190 frozen protocol — ten variations of the three promoted Kelly parents

Written before any R-190 financial evaluation. This operator-directed round
follows `docs/ROUTINE.md`; ten requested candidates replace its usual 1–5.
The parents are L-01 Kelly v4, L-02 Kelly v3 and L-04 original Kelly, all
PROMOTED. L-03 Kelly v2 is NOT PROMOTED and is excluded. No undispatched
round exists: numerically newest shared files R-184 through R-188 each
have a section-B entry; R-189 is complete. The four live backlog entries
remain B-06 ongoing, B-09 low, B-17 partial, B-28 data-blocked. The latest
section-E row dispatched R-189, so the null-pass count is zero.

## Direction, mechanism and falsification

Test whether four-hour execution against actual account exposure improves
the promoted parents' net performance without changing their signals.
This attacks COST and execution error, not information or sample size.
It deliberately revisits known band/ensemble ideas at the user's request:
L-05/L-06, R-37, R-40, R-64/R-66, R-72, R-131/R-133, R-147 and R-186.
The difference is a common actual-account UTC execution policy across the
three accepted parents. This is a composition/replication, not a claim of
a novel algorithm. Primary sources and exact scope: `docs/R190_RESEARCH.md`.

Nine candidates use unchanged parent targets and execution bands
0.05/0.10/0.20 equity notional; candidate ten averages the three targets
and uses 0.10. The common four-hour UTC clock allows at most six orders
per day. Flat targets close residual holdings; no exits manufacture
activity. Parent target generation, its own signal deadband, leverage
caps, and the broker's separate deadband stay intact. Two auxiliary blend
bands, 0.05 and 0.20, provide its parameter neighbourhood; they are controls,
not additional requested candidates. They receive the seven named cells,
not the 48 beta-window cells. No searched or retuned parameters are planned.

Every candidate has the same falsification: it must remain profitable on
ETH at 40bp and on matching Deribit perpetual prices with actual funding;
it must also improve its native-parent comparison under the main 40bp BTC
spot costs. Failure of any condition kills promotion. Named additional
failure modes: more fee-paying churn, delayed regime exits, insufficient
cadence, an exposure mismatch, inert futures bands, and a narrow lucky peak.

## Data and evaluation

Explicit committed datasets, never the resolver-selected one-year CSVs:
Bitstamp BTC spot, Deribit BTC perpetual prices and matching 8h funding,
Coinbase ETH spot. End is 2026-08-12 00:40 UTC. Inner train is 2017–2020;
inner validation is 2021–2022; holdout is 2023 onward. Training preparation
receives only pre-2023 bars. For each period a fresh $1,000 broker uses
causal features prepared with earlier history. Native parents initialize
their already-known target at the first eligible bar; their subsequent
callbacks are unchanged. Candidates first act at an eligible UTC slot.
All orders fill at the next open. Original global warmups remain flat in
inner train; paired daily dates remain aligned. No frozen source changes
after the holdout manifest is written.

Main BTC spot costs are a constant 40bp taker + 1bp slippage, with a $10
minimum order, reflecting the verified current Bitstamp entry tier as a
historical scenario. The 10bp holdout is a separate high-volume discount
scenario. ETH uses 40bp + 1bp as a fee stress, not a claim about a Coinbase
account. Perpetuals use the legacy conservative 5bp + 1bp and matching
Deribit 8h aggregated funding. This retains the generic linear-margin
broker, not an exact inverse-contract or continuous-funding venue model.
Funding coverage must be complete from 2021. No order-book/queue proxy.

Per main candidate/control: inner train spot, inner validation spot,
funded validation; holdout spot, discounted spot, ETH, funded holdout;
24 identical seed-190 windows of 120–365 days each on spot and funded perp.
The 10 candidates + 3 native parents + buy-and-hold receive 55 cells each
(770). Two auxiliary blend neighbours receive seven each (14).
Total planned core cells: **784**. All branches and controls report.
Five-times buy-and-hold is only a liquidation stress, never a scored
risk-matched comparator. Fills, completed round trips, active-day share,
actual mean exposure, realized daily volatility, fees and funding are
reported for every arm. Daily return number one includes PnL from $1,000.

## Risk matching and inference

For all ten candidates, on spot inner validation/holdout and funded inner
validation/holdout, independently simulate an actual constant-exposure
hold using existing `experiments.matched_hold.ConstantExposureHold`.
Start c=0.5; permit at most three backtests, multiplying c by candidate
daily volatility / achieved daily volatility between attempts, clipped
to [0.001, 1] on spot and [0.001, 2] on perpetuals. Charge identical costs
and funding. Stop at relative volatility error <=2%; otherwise mark the
comparison INVALID. This ex-post scalar matches risk for inference only;
it is not a tradable forecast. Every solver attempt is a counted cell,
including failed matches. The control's existing explicit-quantity orders
and relative 10% rebalance band are retained and disclosed.

Native-parent comparisons are valid only within 5% relative realized
daily volatility; otherwise void them, never score lower exposure as
improvement. Blend's native reference is v4; its central-band constituent
results also expose whether it adds anything beyond the common execution
policy. Beta windows use their own parent-risk match, not a coefficient
fitted on a different period. No claim of matched passive-hold beta
outperformance is made: the resampling gate concerns native parents.

Paired stationary block bootstrap: 30-day mean blocks, 2,000 resamples,
seed 190 shared within each cell; daily Sharpe, log growth and maximum
drawdown differences against both native parent and matched passive hold.
Report 95% ranges, validity and control exposures together. The inherited
Sharpe noise floor is +0.20 (R-20), not a newly chosen annual-return hurdle.
Before holdout, report its implied sample requirement using inner-validation
paired daily log-return noise and the +0.20 Sharpe equivalent at candidate
volatility; use block-bootstrap mean SE, 2.8 SE for 80% power at 5% two-sided,
and state that the projected horizon assumes stationary noise.

DSR uses daily return moments and **all** cumulative holdout consultations:
inherited approximate 1,503 plus actual core, matching and audit holdout
cells. Also report local 12-configuration DSR. Trial Sharpe dispersion is
the larger of this round's inner-validation candidate/auxiliary sample SD
and R-189's pre-existing 0.418538, to avoid a tiny neighbourhood manufacturing
certainty. These are approximate, dependent trials, not independent tests.

## Exhaustive decision rule

A candidate is PROMOTED only if **all** of the following hold; otherwise
it is NEGATIVE. Report each Boolean and the failed conditions, with no
post-result reinterpretation:

1. On BOTH spot and funded perpetual inner validation AND holdout, its
   daily Sharpe exceeds its native parent AND matched passive hold by
   strictly more than +0.20; all these comparisons pass their risk matches.
2. On BOTH primary holdout markets, paired 95% lower bounds for Sharpe AND
   log-growth improvements versus BOTH controls exceed zero. Growth must
   exceed the parent in raw final balance as well. Drawdown is reported
   at matched risk; no separate tail-only escape from this growth goal.
3. Profitable (> $1,000) in all four named holdout cells, ETH daily Sharpe
   >= its native reference, and no liquidation in any candidate cell.
4. Primary spot holdout has 2–6 actual fills per calendar day on average.
   Completed round trips/day are separately reported; filling this gate
   does not establish a few completed round trips/day.
5. Program-level DSR >=0.95 on both primary holdout markets.
6. In EACH market, at least 13 of 24 beta windows beat the native parent's
   log growth AND pass the per-window 5% volatility match. Invalid cells
   do not count as wins. Report raw win rates separately.
7. Plateau: in each of the four primary validation/holdout cells, all three
   family bands are profitable and their daily-Sharpe range is <=0.20;
   at least two bands beat their native parent. The blend uses its two
   auxiliary bands and v4 as reference.

All ten are evaluated even if inner validation already predicts failure.
Inner-validation ranking alone selects a descriptive lead (mean spot and
funded daily Sharpe, ties fewer spot fills); holdout never selects defaults.
This is an explicit operator-requested full evaluation, with its repeated
holdout cost recorded. Negative candidates remain in `experiments/`; a
separate round chart/results table links from README, while the accepted
comparison and registered defaults remain a stable record. Only a promoted
candidate earns registration and its required full/holdout intervals.

Independent skeptical reproduction starts only after a primary financial
cell exists. Record hashes, actual totals, protocol deviations if any,
the ledger verdict, and the four backlog statuses; then commit and push
per the requested routine. Preserve the user's pre-existing staged CSVs.
