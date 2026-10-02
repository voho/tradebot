# R-191 source review and five strategy designs

Research date: 2026-10-02. This note was written before this agent ran any
financial evaluation; this agent did not inspect current-round training or
holdout outcomes. The implementation and frozen protocol are authoritative
for the exact executable definitions and decision rule.

## Selection and scope

The operator explicitly requested five new implementations and tests. The
routine's standing diagnosis and ruled-out registry are already saturated:
R-188 tested ten new standalone rules, R-189 ten council games, and R-190 ten
scheduled Kelly execution variations, all negative. The five below are
distinct implementations/compositions, **not five newly discovered sources of
information or five mechanisms claimed never to have been studied here**.
They attack COST and/or ERR/SIZE. None creates new independent market events,
adds instruments, or escapes the approximately three effective regimes.

The live backlog remains B-06 ongoing, B-09 low, B-17 partial, B-28
data-blocked. This operator-directed round does not silently promote a
low-priority item or reopen a closed result as though its evidence vanished.
In particular, B-09 is not closed merely by building another adaptive
quantile rule.

All proposals use a single risky asset plus zero-interest cash, bounded
long exposure in [0,1], and uniform 252-day warmup. Decisions occur at
23:55 UTC, after all 288 bars of a complete UTC day have closed; incomplete
days cannot produce a daily observation. The allowed order fills at the
next 5-minute open. A calendar day's final close is unavailable earlier
in that day. State and feature histories may warm on prior data; the
measured account starts fresh. At a decision, compare with actual held
exposure using a 0.05 equity-notional band, with explicit zero exits. The
native broker's own deadband remains active. No order-book, queue,
maker-fill or synthetic-flow claim is introduced.

## Five proposals

Notation: p is the most recently completed daily close; r is its simple
return; c is a **fixed 0.0041 design hurdle** (40bp plus 1bp), identical
across all assets and market scenarios. Actual simulator fees, slippage
and funding remain separate and must all be charged. Thus the low-fee
futures design deliberately retains a conservative spot-sized internal
hurdle rather than adapting its decision to the evaluation scenario.
Each strategy has one primary configuration and two named neighbours
fixed in advance. Neighbours test sensitivity; they are not a post-result
search.

### 1. Cost-aware proximal allocation (`r191_cost_proximal`)

**Mechanism:** trade only when a seven-day mean/variance objective pays for
moving away from the previous decision. With trailing 63-day mean mu and
variance v, solve over w in [0,1]:

`7*mu*w - 0.5*4*7*v*w*w - lambda*c*abs(w-w_prev)`.

The one-dimensional solution is a soft-threshold of the unconstrained
quadratic optimum around w_prev, followed by clipping. Zero variance and
unavailable history require explicit deterministic fallbacks. Lambda is 2
for the primary and 1/4 for its neighbours. This is a design penalty on the
previous target, not an exact forecast of broker turnover after holdings
drift; the simulator must still charge every actual fill.

**Overlap:** R-131/R-133 penalized or postponed an already-decided Kelly
rebalance; this rule makes the entire standalone allocation through a
cost-bearing objective. R-171 applied ONS inside Kelly's scale slot; no
Kelly vote or inherited scale is used here. It retains the familiar noisy
mean/variance estimation problem of R-37/R-38/R-188.

**Named failure:** the trailing drift estimate changes too late, or the
cost term merely suppresses useful participation. ETH at 40bp is the
preselected falsification; its frozen profitability and comparator rule
must pass without ETH retuning.

### 2. Adaptive downside-quantile budget (`r191_conformal_tail`)

**Mechanism:** allocate only when the completed daily close exceeds the
close 90 complete daily observations earlier (90-day momentum), with
size inversely proportional to a calibrated one-day downside estimate.
Define loss=max(0,-r), obtain q from the previous 252 resolved losses at
quantile 1-alpha, and use

`w = 1[p_t > p_(t-90)] * min(1, budget/max(q, numerical_floor))`.

After a previously forecast day's loss resolves, update
`alpha += 0.01*(0.10 - 1[loss > prior_q])`, with alpha clipped to [0.01,0.50].
Budget is 0.03 for the primary and 0.02/0.04 for the neighbours. Alpha starts
at 0.10. Retain prior_q for the prediction actually issued before comparing
it with its outcome; do not score an outcome against a quantile fitted on
that same outcome.

This is **ACI-inspired adaptive quantile sizing**, not a claim of a valid
conditional loss guarantee. Clipping alpha, serial dependence, trading
costs, price gaps, and a changing exposure all matter. The one-day raw-price
loss statistic is neither the funded account's realized loss nor its
maximum intraday loss. Report calibration and boundary occupancy if they
are used to support an error-control claim.

**Overlap:** R-87 covered vote hit rates/dispersion; R-161 and R-167 used
risk-controlling mean-loss upper bounds on Kelly scale. Here the explicit
outcome is the resolved next-day downside quantile and the rule is
standalone. This is still a tail-budget SIZE/ERR extension, related to
R-125/R-152; it cannot be sold as new independent information.

**Named failure:** the cap rarely binds, saturates, or makes drawdown look
better solely by reducing exposure. ETH at 40bp is the preselected
falsification, with a simulated matched-risk holding control required for
any claimed risk improvement.

### 3. Cost-gated moving-average reversion (`r191_cost_olmar`)

**Mechanism:** allocate between cash and the risky asset using OLMAR's
passive-aggressive update, but decline a new decision unless the forecast
move exceeds a declared round-trip fee hurdle. Set
`xhat=(SMA_W(p)/p, 1)` and `b=(w,1-w)`. With epsilon=1.01,

`tau=max(0, (epsilon-dot(b,xhat))/sum((xhat-mean(xhat))**2))`;
`b_new=project_simplex(b+tau*(xhat-mean(xhat)))`.

Hold the prior decision when `abs(xhat[0]-1) <= 2*c` or the denominator is
numerically zero. W is 10 days for the primary and 5/20 for its neighbours.
The hurdle is a heuristic economic filter: it is not proof that the
forecast is correct or that a round trip completes within W days.

**Overlap:** R-61 used a rolling z-score reversion vote; R-188 faded
session VWAP. This uses a constrained online portfolio update on daily
cash/risky allocations. Both the reversion hypothesis and transaction-cost
problem remain familiar. It has no cross-sectional diversification here.

**Named failure:** a single trending cryptocurrency does not exhibit the
stock-universe reversion the original paper exploits; the update may
collapse into binary switches and fees may consume the gross gains. ETH
at 40bp is the preselected falsification.

### 4. Sticky net-reward expert selection (`r191_sticky_expert`)

**Mechanism:** choose one of four causal decisions—cash, full long,
20/80-day binary trend, and the binary reversion decision `close < SMA20`
—using resolved net rewards, while charging a penalty for switching
the selected exposure. For expert j, update

`S_j = .99*S_j + previous_weight_j*r - c*abs(previous_weight_j-older_weight_j)`.

Then choose the current expert maximizing
`S_j - penalty*abs(current_expert_weight_j - current_selected_weight)`.
Break exact ties by the fixed expert order (cash, full long, trend, reversion).
Penalty is .01 for the primary and .005/.02 for its neighbours. The
score's transaction deduction belongs to the decision whose return has
just resolved, never an expert signal first calculated after that return.
These are hypothetical arithmetic expert scores using target changes;
they are not independently financed shadow accounts and omit holdings
drift and expert-level funding. Only the selected strategy's actual
broker account is a financial evaluation.

**Overlap:** R-147 already studied fixed-share tracking, and R-189's
councils cover many expert aggregation games. This is a hard discounted
selector with an explicit current-switch penalty, rather than a convex
council allocation. It does **not** implement fixed-share or Shrinking
Dartboard and does not inherit their regret bounds.

**Named failure:** the long expert monopolizes the selector, making the
method a costly holding rule, or discounted switching chases yesterday's
winner. ETH at 40bp is the preselected falsification.

### 5. Robust block growth allocation (`r191_robust_growth`)

**Mechanism:** avoid letting a few extreme daily returns determine the
chosen exposure by using a median of six 14-day block estimates, with a
cost penalty inside the decision. For w in {0,.25,.5,.75,1}, partition the
last 84 daily returns into six contiguous blocks and maximize

`14*median(block_mean(w*r - .5*(w*r)**2)) - lambda*c*abs(w-current_w)`.

Lambda is 2 for the primary and 1/4 for its neighbours; ties prefer least
turnover. The quadratic expression is a log-growth approximation, not
exact compound PnL; execution and reporting use the broker's actual account.
The six blocks are dependent market observations, not six independent
regimes. This is a robust objective, **not a calibrated confidence bound**.

**Overlap:** R-188 used a worst-window drift lower bound in standalone
robust Kelly; R-45 robustly reselected Kelly parameters across folds.
This directly selects a finite exposure from a median block growth
objective. It does not repair the project's effective-sample-size ceiling.

**Named failure:** the median discards rare but essential bull returns,
or the coarse grid creates fee-heavy switches without robust forward
growth. ETH at 40bp is the preselected falsification.

## Primary-source evidence and limits

**Boyd, Busseti, Diamond, Kahn, Koh, Nystrup and Speth (2017),
“Multi-Period Trading via Convex Optimization,” Foundations and Trends in
Optimization 3(1), 1–76.** The framework jointly prices expected returns,
risk, turnover and holding costs. Its illustrative historical sample uses
continuously traded December-2016 S&P 500 constituents over January
2012–December 2016; the authors explicitly acknowledge survivorship bias.
It is a multi-stock universe, not one cryptocurrency. The example sets
5bp spread cost, 1bp short holding cost and a nonlinear impact term. The
framework does not supply a forecasting edge; R-191's daily mean estimate,
seven-day horizon, risk coefficient and bounded scalar version are our
design. [Author page](https://stanford.edu/~boyd/papers/cvx_portfolio.html),
[paper, §7.1](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).

**Gibbs and Candès (2021), “Adaptive Conformal Inference Under Distribution
Shift,” NeurIPS 34.** The paper adapts prediction-set coverage through a
scalar miscoverage update. Its financial illustration predicts daily
squared-return volatility on 12 individual stocks; four are featured in
the main figure. It also studies county vote totals during the 2020 US
election. These are coverage experiments, not fee-charged portfolio
backtests, and no BTC profit claim or exchange execution assumption is
made. R-191 changes the target to downside loss and uses the estimated
quantile for sizing; the paper's theorems must not be asserted for the
clipped adaptation as implemented here.
[Conference paper](https://proceedings.nips.cc/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf).

**Li and Hoi (2012), “On-Line Portfolio Selection with Moving Average
Reversion,” ICML 29.** The primary paper evaluates daily stock portfolios
on four datasets: NYSE(O), 36 assets, 1962–1984; NYSE(N), 23 assets,
1985–2010; DJA, 30 assets, 2001–2003; TSE, 88 assets, 1994–1998. Its base
model assumes zero costs, perfect liquidity and no impact; it separately
sweeps proportional transaction costs from 0% to 1%. It reports improved
portfolio wealth on these particular stock datasets. That empirical
result is not evidence for cash plus one crypto asset at 40bp. R-191 adds
a fee hurdle and uses epsilon=1.01, unlike the original paper's empirical
epsilon=10. [Conference paper, Table 3 and §5.3](https://icml.cc/2012/papers/168.pdf).
The expanded treatment is Li, Hoi, Sahoo and Liu (2015), Artificial
Intelligence 222, 104–123:
[publisher](https://www.sciencedirect.com/science/article/pii/S0004370215000168).

**Herbster and Warmuth (1998), “Tracking the Best Expert,” Machine Learning
32, 151–178**, studies regret relative to sequences partitioned among
experts. **Gofer (2014), “Higher-Order Regret Bounds with Switching Costs,”
COLT/PMLR 35, 210–243**, studies full-information online optimization with
switching costs. These are mathematical online-learning results, not
historical cryptocurrency investment studies; there is no empirical
instrument count or live taker tier to transplant. R-191's sticky
discounted selector is a simple heuristic inspired by the switching-cost
problem, not an implementation of either bound-achieving algorithm.
[Herbster–Warmuth publisher](https://link.springer.com/article/10.1023/A:1007424614876),
[Gofer conference page](https://proceedings.mlr.press/v35/gofer14.html).

**Lugosi and Mendelson (2019), “Mean Estimation and Regression Under
Heavy-Tailed Distributions: A Survey,” Foundations of Computational
Mathematics 19, 1145–1190**, describes median-of-means and related estimators
under explicit sampling and moment assumptions. It is a statistics paper,
not an investment backtest: no traded instruments, fees, or reported
strategy profit. R-191 borrows the robust aggregation idea only. Adjacent
blocks in a dependent return series do not automatically satisfy iid
guarantees; the six-block median is not a confidence interval.
[Author manuscript](https://arxiv.org/abs/1906.04280),
[author publication list](https://sites.google.com/view/shaharmendelson/home/publications).

## Cost, data, and falsification discipline

The parent agent independently checked the official venue pages during
this session. Bitstamp's standard entry tier below $10,000 trailing
30-day volume is 40bp taker, with a $10 USD minimum; the displayed schedule
is valid from 1 September 2026. Deribit's English page still labels the
3.5bp perpetual/futures standard rate as upcoming. Retain 5bp as a
conservative declared scenario, not a verified current account rate, and
charge the matched historical funding separately.
[Bitstamp schedule](https://www.bitstamp.net/fee-schedule/),
[Deribit schedule](https://support.deribit.com/hc/en-us/articles/25944746248989-Fees).

Fixed fee tiers over historical data are scenarios; 1bp slippage is an
assumption, not an observed spread. At a $1,000 account, minimum order sizes
and actual broker suppression can overwhelm a nominal small target
change. Report actual fills and exposure, never infer them from daily
decision counts. The generic funded broker remains a model with disclosed
contract limitations.

The committed R-190 prices end 2026-08-12, so any reuse is an evaluation on
that frozen history, not a claim to incorporate market prices through
2026-10-02. All five can be computed with existing OHLCV; no unfetched
source is proxied from candles. The shared protocol should freeze the
train/validation/holdout dates, all primary and neighbour names, the exact
promotion conjunction, trials counter and fallback verdict before any
holdout evaluation. It should report at least BTC spot at40bp, the10bp
discount scenario, ETH40bp, funded BTC, 24 paired windows, simulated
matched-risk holds, and program-adjusted significance. Noise-based power
diagnostics must use training/validation evidence and cannot redefine the
bar after observing holdout performance.
