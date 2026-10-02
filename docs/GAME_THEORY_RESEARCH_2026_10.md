# Game theory research and new trading ideas

Research cutoff: **2 October 2026**. This review concentrates on primary
2024–2026 research, including revisions available by the cutoff. It separates
published research, working papers, simulations and historical measurements.
It proposes research directions; none is a validated strategy for this bot.

**Assessment:** the most useful new mechanisms require observations of other
participants, their liquidity, or their constraints. Our existing five-minute
OHLCV framework cannot identify most of those mechanisms. More elaborate
games among forecasts would largely revisit the negative R-188 through R-191
experiments. The priority is to establish a new information channel and an
economically meaningful test before selecting an algorithm.

This was a literature and design review: **zero financial evaluations, zero
new holdout consultations, no new round ID, and no strategy registrations**.
The [R-191 results](../reports/r191_strategies/README.md) are unchanged.

## Five proposed research directions

The ranking below is our judgment about research value for this repository,
not a ranking of demonstrated returns. Every direction has a data or
simulation prerequisite that the current candle backtester does not meet.

| Priority | Proposed direction | Strategic mechanism | First prerequisite |
|---|---|---|---|
| 1 | Flow-sensitive execution | Liquidity providers learn from informed flow; choose when to cross or wait using actual transactions and quotes | Signed trade events, synchronized quotes, conservative fill accounting |
| 2 | Funding carry with venue stress | Leveraged traders and venue loss-allocation rules can break an otherwise hedged position | Point-in-time positions or leverage distribution, depth, insurance and liquidation state |
| 3 | Liquidity allocation with competing providers | Other providers dilute fee income and react to attractive price ranges | Historical pool events and reconstructable liquidity positions |
| 4 | Execution against adapting counterparties | Other traders change quotes after observing our execution policy | Own child-order logs and a calibrated interactive order-book simulator |
| 5 | Randomized execution to reduce information leakage | Predictable schedules let others infer the remaining trade | Evidence that our order size or schedule has measurable impact |

### Flow-sensitive execution

**Hypothesis, motivated by Aqsha et al. and Schuhmann et al.:** observed trade
direction and subsequent quote response distinguish persistent informed flow
from temporary liquidity demand sufficiently well to improve execution.
Start with a simple causal flow/markout estimator, where markout means the
price movement after a fill. Keep the parent strategy's trade direction,
quantity and deadline fixed; let the estimator select immediate execution
or a bounded wait. Public signed flow is a weaker information set than a
broker's identifiable client flow, so this is an adaptation, not a replication.

**Test:** compare price-only and flow-added estimators under identical order
constraints. Train on earlier events; freeze rules before evaluating later
dates and another venue. Measure net implementation shortfall, adverse
selection, completion rate and tail cost. Include nonfills, latency and fees.
Reject if the improvement disappears after plausible timestamp delays or
comes from leaving difficult trades unfilled. This differs from the repo's
bar-derived Kyle/VPIN proxies because it adds observed transactions and quotes.

### Funding carry with venue stress

**Hypothesis, motivated by Chitra and Gornall et al.:** a spot/perpetual hedge
can earn better returns per unit of committed collateral if position size
responds to observable liquidation capacity and loss-allocation risk.
Autodeleveraging (ADL) can forcibly close a profitable hedge leg when the
venue cannot absorb losses. A candidate would compare expected funding and
basis income with borrowing, trading, collateral and stress-exit costs.

**Test:** compare against the same carry rule without the added venue-state
inputs, at matched collateral usage and gross exposure. Model both legs,
margin, funding timestamps, forced closes and rehedging. Use multiple
independent stress episodes and quiet periods. Reject if the filter is only
a disguised funding-rate threshold, only works on the already-public October
2025 event, or depends on insurance/position data published after the decision.
R-100 funding divergence and R-145 venue routing are already negative:
rates alone do not make this a new idea. A [public reconstruction](https://github.com/ConejoCapital/HyperMultiAssetedADL)
exists for October 2025; continuous point-in-time participant and insurance
history across independent episodes remains unverified.

### Liquidity allocation with competing providers

**Hypothesis, motivated by Tang et al.:** allocating capital across price
ranges using competitors' current liquidity improves net portfolio returns
after inventory changes, gas and hedging costs, at matched risk, compared
with choosing ranges from volatility alone. Concentrated
liquidity market makers let providers choose where their capital is active;
fee sharing makes the other providers' choices economically relevant.

**Test:** compare a lagged competition-aware allocation with static and
volatility-only ranges, using equal capital and rebalancing opportunities.
Reconstruct swaps, liquidity changes, fees and token inventory exactly.
Account for gas and hedging in the cash ledger; use impermanent loss and
loss-versus-rebalancing as alternative diagnostics, not additive costs that
double-count the same inventory loss. Evaluate untouched pools and dates,
then stress competitors reallocating. Reject if the advantage needs future
fees, frozen competitor positions, or a capital scale at which gas dominates.
This requires a separate pool simulator, not an OHLCV indicator.

### Execution against adapting counterparties

**Hypothesis, motivated by ABIDES-MARL and PyMarketSim:** a policy selected
against diverse, adapting market makers is more resilient than one optimized
against a fixed impact curve. Keep parent size, direction, completion deadline
and inventory constraints fixed; choose child-order timing and aggressiveness.

**Test:** benchmark against TWAP and a participation schedule with training-only
volume profiles. Freeze the policy, then introduce independently trained
opponents and fresh best-response learners. Test different impact, queue and
latency calibrations. Reject gains that vanish when opponents retrain or
simulator price boundaries are removed. A historical replay cannot by itself
establish how opponents would react to a changed policy. This is a larger
infrastructure project and needs a measured cost opportunity first.

### Randomized execution to reduce information leakage

**Hypothesis, inspired by Huang and Zhu:** bounded randomization of legitimate
child-order timing and size reduces what other traders infer about the
remaining parent order. The source studies portfolio choices in a theoretical
information game; applying it to execution is our proposed extension.

**Test:** preserve total quantity, expected timing, deadline and participation
limits. Compare deterministic and randomized schedules against an opponent
trained to infer the residual order. Measure both predictability and net
implementation shortfall. Reject if reduced predictability fails to improve
cost, or delay risk and extra fees offset the benefit. Do not import unbounded
Gaussian portfolio actions. At the repo's $1,000 benchmark size, meaningful
own-order impact is unestablished; this ranks last until measured.

## Primary research behind the ideas

### Information and execution

**Aqsha, Drissi and Sánchez-Betancourt — Strategic Learning and Trading in
Broker-Mediated Markets.** Preprint first December 2024; inspected revision
19 January 2026. A broker and informed client infer each other's private
information. Simulations with impact and trading costs find client-flow
information valuable, while price-only filtering resembles a simple inventory
benchmark. The evidence does not establish that public flow reproduces a
broker's private information advantage. This is the clearest motivation for
an information ablation before another forecasting model.
[Paper](https://arxiv.org/html/2412.20847v2).

**Schuhmann, Köhler, Heckens and Guhr — A New Traders' Game? Empirical Analysis
of Response Functions in a Historical Perspective.** Published in *Physica A*
679, 130981 (2025); inspected arXiv revision 12 October 2025. Uses US trade and
quote data for 2007, 2008, 2014 and 2021, with 99 stocks per year and some
replacements. Finds changing self- and cross-responses to signed trades.
This is real-market evidence of nonstationary flow/price relationships,
not a transaction-cost-adjusted crypto strategy or proof of causation.
[Paper](https://arxiv.org/html/2503.01629v2).

**Cheridito, Dupret and Wu — ABIDES-MARL: A Multi-Agent Reinforcement Learning
Environment for Optimal Execution with Endogenous Liquidity.** Preprint first
3 November 2025; inspected revision 24 August 2026. In simulated order books,
execution and liquidity providers learn together. Fixed-impact policies can
suffer when makers adapt, and learned policies can exploit simulator
boundaries. The reported evaluation is synthetic, not historical financial
validation. Its VWAP profile uses PPO evaluation trajectories; our benchmark
would need a training-only volume profile.
[Paper](https://arxiv.org/html/2511.02016v2).

**Huang and Zhu — Mean-Variance Stackelberg Games with Asymmetric Information.**
Preprint, 3 September 2025. A better-informed leader trades while a follower
observes prices and actions. Entropy-regularized randomization can preserve
the leader's information advantage in the model. This is mathematical
equilibrium analysis, without a historical backtest or real exchange costs.
It motivates testing information leakage, but does not establish that random
trade timing is profitable.
[Paper](https://arxiv.org/html/2509.03669v1).

### Derivatives and decentralized liquidity

**Tang, El-Azouzi, Lee, Chan and Fanti — Game Theoretic Liquidity Provisioning
in Concentrated Liquidity Market Makers.** *POMACS* 9(1), 2025,
DOI 10.1145/3711700; inspected arXiv revision 17 June 2026. Models competing
providers with limited budgets and derives an equilibrium allocation across
price ranges. Historical pool analysis contrasts a full-information
counterfactual using realized same-day outcomes with separate heuristics
using lagged data. Its unilateral comparison holds other providers' strategies
fixed; no untouched prospective validation is established.
Fee dilution provides a concrete strategic mechanism. Reported improvements
are not evidence of live returns after our gas, hedging and execution costs.
[Paper](https://arxiv.org/html/2411.10399v2).

**Chitra — Autodeleveraging: Impossibilities and Optimization.** Preprint first
30 November 2025; inspected corrected revision 16 February 2026. Models venue
loss allocation, including a Stackelberg control problem, and examines
Hyperliquid's 10 October 2025 ADL event. Counterfactual policies replay the
realized price path; public records lack complete internal clearing state.
This supports modeling forced hedge closure and the venue's incentives.
It is a retrospective event analysis,
not a validated advance-warning signal. The corrected version distinguishes
profit haircuts from the value of positions closed; older headline figures
must not be reused interchangeably.
[Paper](https://arxiv.org/html/2512.01112v3).

**Gornall, Rinaldi and Xiao — Perpetual Futures and Basis Risk: Evidence from
Cryptocurrency.** Preliminary working paper dated 19 May 2025, hosted in the
AEA 2026 program; not an AEA journal publication. Combines cryptocurrency
market evidence with a model of constrained arbitrage capital. Perpetuals and
quarterly futures exhibit different liquidity and basis behavior in stress.
Useful for specifying collateral and exit risk; it does not validate a
profitable funding selector for this repo.
[Paper](https://www.aeaweb.org/conference/2026/program/paper/ByyFEfr4).

**Wu, Sui, Thiery and Pai — Measuring CEX-DEX Extracted Value and Searcher
Profitability: The Darkest of the MEV Dark Forest.** *AFT 2025*, published
6 October 2025. Empirical Ethereum analysis covers August 2023–March 2025.
Competition, concentration and integration with block builders affect who
captures arbitrage value. Although the analysis includes estimated fees and
builder payments, actual centralized-exchange hedges are unobserved. Midpoint
hedges and a horizon selected from each searcher's historical sample make
estimated PnL an upper bound. Our inference: an observed price gap is
insufficient grounds for an executable arbitrage claim; access, depth and
inclusion costs are part of the game.
[Proceedings](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.AFT.2025.26).

### Learning guarantees and evaluation

**Cesa-Bianchi, Cesari, Colomboni, Foscari and Pathak — Market Making without
Regret.** *COLT 2025*, PMLR 291:799–837. Establishes what can be learned about
quotes from limited fill feedback under different distributional assumptions.
Its comparator is the best fixed bid/ask pair, and inventory is offloaded at
each round's market price. Sublinear regret under the specified assumptions
does not guarantee positive profit; unrestricted cases permit linear regret.
Real queues, accumulated inventory and exchange costs need separate treatment.
[Proceedings](https://proceedings.mlr.press/v291/cesa-bianchi25a.html).

**Mascioli, Gu, Wang, Chakraborty and Wellman — A Financial Market Simulation
Environment for Trading Agents Using Deep Reinforcement Learning.**
*ICAIF 2024*, pp. 117–125. PyMarketSim supports order matching, private
valuations and empirical game-theoretic analysis. Iteratively adding responses
is useful for exposing fragile policies. In its example only two of 25
agents adapt, limiting conclusions about full-market response. A weak
best-response learner failing to exploit a policy does not prove the policy
is unexploitable.
[Proceedings paper](https://strategicreasoning.org/wp-content/uploads/2024/11/ICAIF24proceedings_PyMarketSim.pdf).

**Dou, Goldstein and Ji — AI-Powered Trading, Algorithmic Collusion, and Price
Efficiency.** NBER Working Paper 34054, July 2025. Theoretical markets and
reinforcement-learning simulations produce supra-competitive outcomes through
learning dynamics. It is not empirical evidence of collusion in observed
crypto trading. Our use is diagnostic: evaluate policies against unfamiliar
opponents rather than assuming profits from familiar self-play will persist.
[Working paper](https://www.nber.org/papers/w34054).

## Recent papers to treat cautiously

**Yang and Xu — Robust Market Making with Hawkes Order Flow and Price Impact
via Adversarial Reinforcement Learning.** Preprint, 19 September 2026. Combines
clustered simulated arrivals, impact, an environmental adversary and recurrent
state. Useful stimulus for stress testing; no historical instrument test or
explicit venue-fee validation. Reviewer concern: Appendix A appears to omit
the inventory/impact expectation term and retain a drift/count cross-term at
the wrong order in the time step. This warrants clarification before relying
on its reduced-game derivation; it does not establish that all numerical
results are invalid. Single-stage equilibrium existence is not PPO convergence.
[Paper](https://arxiv.org/html/2609.22785v1).

**Zhang, Wu, Jia and Sun — Game-Theoretic Modeling of Heterogeneous Investor
Interactions for Stock Price Forecasting.** Preprint, 11 May 2026, described
as intended for conference submission. GameStock combines graphs, price
features and Chinese investor-event disclosures. It reports forecast
correlations rather than net trading returns. Data screening uses presence
over the full sample, requiring a survivorship audit, and event publication
timing needs verification. Reported performance does not win every listed
metric. This is neither an OHLCV-only method nor sufficient evidence to
prioritize a large graph model here.
[Paper](https://arxiv.org/html/2605.23953v1).

**Jafree, Jain and Firoozye — When AI Trading Agents Compete: Adverse Selection
of Meta-Orders by Reinforcement Learning-Based Market Making.** The inspected
31 October 2025 preprint gives its fully aware maker the true presence and
direction of a TWAP metaorder in simulation. That is unavailable information
unless a separate causal detector earns it. Our inference: any metaorder
strategy should first test detection from public events against false
positives, delays and an oracle upper bound; it must not inherit the oracle's
profits as an achievable forecast.
[Paper](https://arxiv.org/html/2510.27334v1).

## Recommended next step

Perform a bounded **data and economics feasibility audit** for the first two
directions before a new strategy round. Establish which historical signed
trades, quotes and venue-state variables are accessible, their timestamps,
coverage and cost. Do not assume a current endpoint provides historical
point-in-time data. If venue state cannot be obtained, keep that idea blocked;
funding rates are not a replacement.

For execution, measure a conservative upper bound on avoidable cost at our
actual trade sizes. If this is negligible, stop before building a multi-agent
simulator. For carry, require honest two-leg, collateral and forced-close
accounting before optimizing entries. For liquidity provision, perform the
same capital-versus-gas feasibility check before implementing pool allocation.

A subsequent experiment should preregister one mechanism, simple baselines,
data splits, capacity/fee assumptions, failure criteria and trial accounting
under [the research routine](ROUTINE.md). Freeze a genuinely new evaluation
period where possible; the heavily consulted historical holdout is not made
fresh by a new model name. Population robustness, matched exposure and
independent accounting complement financial out-of-sample testing; none
replaces it.
