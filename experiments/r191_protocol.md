# R-191 frozen design — five cost-aware allocation rules

Written on 2026-10-02 before any R-191 financial evaluation. The operator
requested five new strategies following `docs/ROUTINE.md`. Step 0 found
`HEAD == origin/main` at `1fd5a57a` after a clean pull/rebase, no undispatched
round (numeric latest shared files R-184 through R-188 have ledger entries;
R-189/R-190 complete), and zero consecutive null passes. Live backlog rows:
B-06 ongoing, B-09 low, B-17 partial, B-28 data-blocked. This explicitly
requested batch extends known mechanisms; it does not claim five new
information channels or five independently discovered scientific results.

## Direction and mechanism

All five use one risky asset and cash with weights in [0,1], without a
short position or added leverage. They attack COST and/or ERR: estimate a
position whose expected benefit can pay for switching, or reduce exposure
when realized downside uncertainty increases. Sources, original data/cost
scope, prior-round overlap, and limitations are in
`experiments/r191_research_notes.md`. No future paper claim is treated as
evidence of profitability here. The parameters below are fixed before
training; no search, replacement, or winner-selected defaults are planned.

1. **Cost-aware proximal allocation** maximizes
   `7*mean63*w - 0.5*4*7*variance63*w² - lambda*0.0041*abs(w-previous_target)`.
   Variance uses ddof=0. Solve this concave one-dimensional objective analytically, clipped to
   [0,1]. Lambda=2, with neighbors 1 and 4. This puts linear costs inside a
   full standalone allocation objective instead of adding a corridor to
   Kelly targets (R-131/R-133) or changing an ONS update (R-171).
2. **Adaptive downside-quantile budget** holds a positive 90-day momentum (`close_t > close_(t-90)`) with
   weight `min(1,budget/q)`; q is the trailing-252-day empirical quantile of
   `max(0,-daily_return)` at probability `1-alpha`. At each close update
   `alpha += .01*(.10 - (today_loss > previous_q))`, clip to [.01,.50],
   then estimate q using losses through that close. Budget=.03, neighbors
   .02/.04. q=0 means full weight if the trend is positive. This addresses
   realized one-day downside errors, not vote hit rates or mean-return
   confidence bounds (R-87/R-161/R-167). It is a heuristic ACI adaptation,
   with no coverage guarantee under arbitrary crypto returns or clipping.
3. **Cost-gated moving-average reversion** uses the two-asset OLMAR
   passive-aggressive update: predicted relatives `[mean(last W closes)/close,1]`,
   epsilon=1.01, followed by Euclidean projection onto the risky/cash simplex.
   Accept the update only when `abs(predicted_risky_relative-1) > 2*.0041`;
   otherwise keep the previous target. W=10, neighbors 5/20. Unlike R-61's
   z-score vote and R-188's VWAP fade, this is a projected allocation update
   with an explicit fee hurdle. A predicted move is not an observed edge.
4. **Sticky net-reward expert selection** chooses one of cash, full long,
   SMA20>SMA80 trend, or close<SMA20 reversion. At today's close update each
   score as `.99*S + prior_weight*today_return - .0041*abs(prior_weight-prior_prior_weight)`.
   Choose the current expert maximizing `score - penalty*abs(expert_weight-current_target)`.
   Penalty=.01, neighbors .005/.02; deterministic order breaks exact ties.
   This is a hard switching-cost choice rather than R-147/R-189's blended
   councils. Expert rewards are hypothetical one-day scores, not executable
   funded shadow-account returns or a transferred regret guarantee.
5. **Robust block growth allocation** partitions 84 daily returns into six
   consecutive 14-day blocks. On weights {0,.25,.50,.75,1}, maximize
   `14*median(block_mean(w*r-.5*(w*r)^2)) - lambda*.0041*abs(w-current_target)`.
   Prefer least turnover on exact score ties. Lambda=2, neighbors 1/4.
   Unlike R-188's worst-window drift bound or R-45's portfolio minimax
   selector, this uses a block-median growth objective over cash weights.
   It is robust aggregation, not a confidence bound or exact log utility.

The common signal cost hurdle is 40bp fee +1bp slippage on every market,
including the discounted and funded cells. It is a fixed conservative
design assumption, not the fee actually charged in those cells. Recursive
signal targets use their own previous target, not simulated account equity;
they therefore carry only causal price history across evaluation boundaries.

Only complete UTC days produce a daily observation; the decision occurs at
23:55 after the last five-minute close, and fills at the next bar's open.
Earlier bars of the same day must not see its eventual close. Every strategy
uses a common 252-day warmup, compares the target to actual marked holdings
at the daily decision, trades when the equity-notional gap exceeds .05, and
closes residual holdings when the target is zero. Native broker deadbands
and minima still apply; the 5x-capable broker's .25-equity target band may
dominate the experiment band. No queue, order-book, or maker-fill proxy.

## Data and cells

Reuse R-190's fresh-account native engine evaluator, with explicit committed
Bitstamp BTC spot, Coinbase ETH spot, Deribit BTC perpetual price and matching
8-hour funding files, ending 2026-08-12 00:40 UTC. Training feature preparation
receives only rows before 2023. Inner train ends 2020-12-31; inner validation
is 2021-2022; retrospective holdout begins 2023-01-01. Each cell has a fresh
$1,000 account and earlier causal feature history. Report the first day's
PnL relative to that initial balance, not as a dropped observation.

Spot primary: 40bp taker +1bp slippage, $10 minimum. The current standard
Bitstamp schedule (valid 2026-09-01, rendered and checked 2026-10-02) lists
40bp below $10,000 trailing monthly volume and a $10 USD minimum.
https://www.bitstamp.net/fee-schedule/
The 10bp BTC scenario is a volume-discount sensitivity. ETH uses 40bp +1bp
as a stress, not a claim about a Coinbase account's specific tier.
Funded perpetuals charge a conservative legacy 5bp +1bp and matching funding;
Deribit's official page, updated 2026-09-25, still labels its 3.5bp standard
table as upcoming. https://support.deribit.com/hc/en-us/articles/25944746248989-Fees
This is a generic linear-margin/8-hour-funding simulation, not the exchange's
exact inverse-contract, continuous-funding, or rounding model. Fees are fixed
historical scenarios; there is no historical tier reconstruction.

Five primary configurations and ten neighbors each receive seven cells:
inner-train spot, inner-validation spot, funded validation, holdout spot,
discount holdout spot, ETH holdout, funded holdout. Primaries plus native
Kelly v4 and passive buy-and-hold additionally receive 24 common seed-191
windows of 120-365 days on each primary market. Controls also receive seven
named cells. Total **455 core evaluations**, 51 training and 404 holdout.
Spot buy-and-hold is the registered strategy; its funded reference is an
unlevered long (constant-exposure c=1), explicitly labeled as such. A 5x
buy-and-hold liquidation is not a performance benchmark. All named candidates
are evaluated even if training already predicts failure; this is the user's
requested full batch and all repeated consultations are counted.

## Risk matching, uncertainty, and freezing

For each of five primaries, simulate the existing ConstantExposureHold with
identical fees/funding independently on both validation cells, both primary
holdout cells, and all 48 beta windows. Start c=.5, allow three attempts,
multiply c by candidate/achieved daily volatility and clip [.001,1] for spot,
[.001,2] for perpetuals. Stop at <=2% relative error; otherwise INVALID.
Its explicit-quantity orders and relative 10% rebalance band are retained.
All 260 comparisons and every solver attempt (max780) are counted, including
failed matches. c is an ex-post diagnostic fitted inside each cell, not a
deployable forecast. Unmatched cells never earn a win. Report exposure,
time in market, realized volatility, fills, completed episodes, fees and funding.

Paired 30-day stationary block bootstrap, 2,000 resamples, seed191: daily
Sharpe, log growth and daily maximum drawdown versus the matched passive
hold in the four primary validation/holdout cells. Report 95% intervals.
Bar-level drawdown and daily bootstrap drawdown have different resolutions.
Bootstrap intervals condition on the cell's fitted passive exposure; that
exposure is not refitted inside each bootstrap replicate, so its estimation
uncertainty is not included. Beta windows do fit their own independent c.
Kelly v4 and full holding are contextual balances, not risk-matched proof.
Overlapping beta windows measure path sensitivity, not new independent data.

After training, measure the paired daily log-return noise against matched
holding and project the history required for the inherited +.20 Sharpe floor
(R-20), using 2.8 bootstrap standard errors for 80% power at 5% two-sided.
Report if the implied horizon exceeds available history; do not lower the
threshold. Freeze source/data hashes, this protocol, 15 local configurations,
and validation-only descriptive ranking before any R191 holdout evaluation.
Ranking uses mean spot/funded validation Sharpe, ties fewer spot fills.

Deflated Sharpe reports both 15 local trials and prior approximate 2,290 plus
every actual holdout core/matching/audit evaluation. Trial Sharpe dispersion
is max(.418538 inherited from R-189, sample SD of the 15 spot validation
Sharpes). Counts include dependent consultations conservatively; they do not
pretend to be independent trials. No holdout choice changes any default.

## Exhaustive promotion decision

PROMOTED iff **all seven** conditions pass; otherwise NEGATIVE:

1. Both markets on validation and holdout pass the risk match and exceed
   their matched passive hold's daily Sharpe by strictly more than +.20.
2. Both primary holdout markets have strictly positive paired 95% lower
   bounds for Sharpe and log-growth improvement over matched holding.
3. Both primary holdout balances exceed the full passive references.
4. All four named holdout balances exceed $1,000; ETH daily Sharpe is at
   least its passive reference's; no primary cell liquidates. This is each
   strategy's pre-registered ETH/real-fee/funding falsification test.
5. Each three-configuration family is profitable and has a daily Sharpe
   spread <=.20 in each of the four primary validation/holdout cells.
6. Program-level deflated Sharpe is >=.95 on both primary holdout markets.
7. Each market has at least 13/24 higher-growth beta windows versus its own
   valid window-specific matched passive hold; invalid cells are not wins.

Anticipated failures: cost hurdle suppresses investment; tail control simply
holds less; mean reversion fights persistent trends; sticky selection chases
past winners; block aggregation dismisses useful tail gains; neighboring
settings disagree; inference remains underpowered. No post-result tail-only
escape or reinterpreted verdict. A negative is retained under experiments,
not registered. An independent skeptic runs only after at least one primary
financial cell exists and checks the code and arithmetic before any surviving
claim is accepted. Finish all tests including strict causality, record the
ledger/counters/backlog, commit and push the review branch; do not merge.
