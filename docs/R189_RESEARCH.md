# R-189: ten intraday games — research and implementation scope

Research checked **2026-09-05**. The request is ten new game-theory trading
candidates, including combinations of existing strategies, evaluated at a cadence
of a few trades per day. These are new **research candidates in this repository**.
They adapt both recent 2024–2026 results and older algorithms that those results
build on. They are not ten independently established state-of-the-art profitable
Bitcoin strategies. No source below establishes that claim.

## What the game observes

The common game uses executable, causal strategy recommendations, including the
existing `kelly_regime_v4`, intraday trend/reversion/breakout rules, a buy expert
and cash. A decision uses completed 5-minute OHLCV bars. Its order can execute
only at a later bar's open. An online update must wait until its scored trading
interval is complete; the close-to-next-open return before entry is unavailable
to that decision's payoff. Experts pay their own hypothetical turnover cost.
The default internal fee is `0.0011` per unit of one-way target turnover;
completed net scores are divided by `0.02` and clipped to `[-1,1]`.
Actual broker fees, slippage, funding, position drift and fill counts are evaluated
separately by the existing engine.

Let `a[t,i]` be expert `i`'s proposed exposure, `g[t,i]` its bounded completed-round
net score, `p[t,i]` a council weight, and `u[t,i] = g[t,i] - p[t]·g[t]` the
expert's advantage over the played mixture. The online-game score is a learning
signal, not a replacement for compounded broker equity. In particular,
`sum(p * expert_net_payoff)` need not equal the net payoff of a rebalanced
mixture. Quantization and minimum holding times further change that equality.

The model committees in candidates 6–8 are alternative summaries of the same
history. They are **constructed model scenarios**, not observed market makers,
informed traders, order-flow participants or a measured strategic opponent.
OHLCV does not identify those participants or their utilities. Exposures are
long-only and capped at one times equity, so a futures comparison does not silently
turn a unit signal into five times leverage.

The decision spacing is four hours, at 00/04/08/12/16/20 UTC. Six daily
opportunities do not imply six actual orders, and an order is not a completed
round trip. The
evaluation must report actual fills/day and closed trades/day and reject claims
that the cadence target was reached when the realized counts disagree. It should
not manufacture trades to meet the target.

## Ten mechanisms and their source limits

### 1. `cautious_optimism` — regret-dependent optimistic pacing

[Soleymani, Piliouras and Farina (2025), Cautious Optimism](https://arxiv.org/abs/2506.05005)
derive faster regret in self-play while retaining adversarial regret guarantees
under their stated conditions. The entropy instance's Eq. 9.2 selects
`lambda = argmax_(0,eta] [alpha log(lambda) + logsumexp(lambda*r)]`, where
`r = cumulative(u) + last(u)`, and plays `softmax(lambda*r)`. A scalar derivative
root makes this feasible at intraday decision times. Pacing can slow when all
optimistic regrets become sufficiently negative; this is different from ordinary
Hedge or a learning rate based only on elapsed time.

**Adaptation:** expert rewards are cost-adjusted trading intervals and the
resulting exposure is constrained by the broker. Chosen learning constants are
experimental defaults, not a claim to instantiate every theorem constant. The
paper studies learning dynamics, not Bitcoin returns, execution costs or daily
trade targets. Its self-play rate does not apply to an exogenous price path.

### 2. `squint_council` — common-variance second-order expert game

[Luo (March 2026), A Short Note on a Variant of the Squint Algorithm](https://arxiv.org/abs/2603.03409)
uses the potential `Phi(R,V) = integral_0^.5 (exp(eta*R-eta²*V)-1)/eta d_eta`.
The variant plays `p_i ∝ integral exp(eta*R_i-eta²*V) d_eta`, with a **shared**
variance clock. After observing `u`, update `R += u`, then solve
`sum_i integral eta*exp(eta*R_i-eta²*(V+v))*(v-u_i²) d_eta = 0` for `v`
and set `V += v`. Finite quadrature and bisection approximate these integrals.

**Adaptation:** twelve-point Gauss–Legendre integration and 22 bisection steps
approximate the kernel. Squint halves `u`, so its absolute regret is at most
one and `v` lies in `[0,1]`. The numerical integration and trading layer do not
inherit an exact potential invariant. The theorem concerns quantile regret in a
full-information expert game, with no market dataset or transaction-cost profit
result. This differs from constant-rate Hedge and from R-188's coin-betting
wealth fraction. The original expert-specific Squint variance is a separate
[2015 construction](https://proceedings.mlr.press/v40/Koolen15a.html).

### 3. `normalhedge_council` — constant-potential NormalHedge.BH

[Freund, Harvey, Portella, Qi and Wang (February 2026), A Second Order Regret Bound for NormalHedge](https://arxiv.org/abs/2602.08151)
analyze the potential `phi(x,tau)=exp(x²/(2*tau))/sqrt(tau)` with
`x=max(cumulative(u),0)`. Maintain constant total potential by a scalar search
for `tau`, and play weights proportional to `x*exp(x²/(2*tau))`. Retain raw
cumulative regrets before taking their positive part; truncating every update
would produce a different algorithm.

**Adaptation:** the code fixes its initial clock at one, differing from the
theorem's large `max(512*e²*B²*log(N),1)` initialization. The
new bound uses a second-derivative-weighted variance, not ordinary market
volatility or the weights used for trading. There are no Bitcoin trading or
fee-adjusted performance claims. Unlike the 2009 NormalHedge potential, this
variant includes `tau^(-1/2)` in the preserved total.

### 4. `swap_regret_council` — conditional expert-switching game

[Hart and Mas-Colell (2000), A Simple Adaptive Procedure Leading to Correlated Equilibrium](https://www.ma.huji.ac.il/~hart/abs/adapt.html)
establish conditional regret matching;
[Blum and Mansour (2007), From External to Internal Regret](https://jmlr.org/papers/v8/blum07a.html)
develop general reductions. The implemented finite chain accumulates
`R[i,j] += p_i*(g_j-g_i)`. Positive off-diagonal regrets become transition
rates, scaled to valid probabilities; remaining probability stays on the
diagonal. Add a `1e-6` uniform tremble and solve `p = p*Q`. This differs from
maintaining one scalar regret for each fixed position in `regret_grid`.

[Arunachaleswaran et al. (2025), Swap Regret and Correlated Equilibria Beyond Normal-Form Games](https://arxiv.org/abs/2502.20229)
extend the theory to profile swap regret and polytope games. **Scope:** the
implemented finite stationary-distribution chain is a foundational adaptation,
not their more general 2025 algorithm or the newer
[Dagan et al. reduction (2024; revised 2025)](https://arxiv.org/abs/2310.19786).
The learning papers' correlated-equilibrium and non-manipulability results do
not show that an OHLCV market will supply profitable deviations. The tremble, fee scores
and execution bands require their own empirical evaluation.

### 5. `blackwell_council` — approachability of a vector of constraints

[Dann et al. (2024), Rate-Preserving Reductions for Blackwell Approachability](https://arxiv.org/abs/2406.07585)
show why multi-dimensional approachability need not reduce to ordinary external
regret without losing convergence-rate information. The candidate maintains
positive cumulative deficits for three objectives: return against clipped
Kelly, squared normalized net reward above `0.0625`, and target turnover above
`0.25` per round. A 42-round EW model predicts return and second moment.
It selects a 21-point exposure-grid action minimizing the inner product
between this deficit vector and an estimated next-round constraint vector.

**Adaptation:** this is a finite action, estimated-payoff separating-direction
rule. Its budgets, reference return and predictions are our design choices;
we do not prove that the target set is approachable. It is not the paper's
general reduction or a hard loss cap. No market data or trading-cost evidence
comes from the paper. This is deliberately not the external-regret-only
position-grid RM+ already tried in L-20.

### 6. `minimax_council` — worst-model expert selection

[Tsang, Sit and Wong (2025), Adaptive Robust Online Portfolio Selection](https://www.sciencedirect.com/science/article/pii/S0377221724006933)
develop robust ellipsoidal portfolio selection with transaction costs and
adaptive model parameters. Its empirical portfolio comparisons motivate
including costs inside a robust decision, not merely after selection.

**Adaptation:** construct `M[i,s]` from expert `i`'s mean completed net rewards
over 6, 42 and 180 rounds. Select `argmax_i min_s M[i,s]`, including
cash; tied maximizers share capital. This is a finite pure-action maximin game,
not an implementation of the
paper's ellipsoidal uncertainty set or a claim to solve the full mixed minimax
portfolio. It learns which executable expert survives alternative history
summaries, unlike R-188's lower-confidence-bound Kelly drift fraction. The
paper's reported portfolio superiority supplies no transferable Bitcoin,
four-hour, one-asset or few-trades/day claim.

### 7. `nash_council` — cooperative surplus bargaining

[Benita, Nasini and Nessah, A Cooperative Bargaining Framework for Decentralized Portfolio Optimization (2021; revised 2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3884009)
study allocation among intermediaries with heterogeneous risk/disutility
objectives using Nash and Kalai–Smorodinsky solutions. A current related study,
[Gupta et al. (2025), Cooperative Bargaining Games Without Utilities](https://arxiv.org/abs/2505.14817),
demonstrates direction-oracle bargaining on portfolio preferences; its stock
example compares negotiated objectives across historical investor windows,
not an out-of-sample intraday trading strategy.

**Adaptation:** treat each history summary as a committee with utility
`U_s(w)=w·M[:,s]-0.02*||w||²`. Evaluate pure experts and equal pair mixtures; choose the
allocation maximizing `sum_s log(U_s(w)-d_s)`. Each disagreement point is the
worst candidate utility in that committee minus `0.01`, ensuring
positive surplus. These committees, concentration penalty, disagreement points and restricted
allocation set are explicit modeling choices. We implement neither DiBS nor
a decentralized market. Cooperation over estimated utilities need not improve
future returns; changing disagreement points can materially change the answer.

### 8. `qre_council` — entropy-regularized trader/scenario game

[Cen, Wei and Chi (2021; revised 2023), Fast Policy Extragradient Methods for Competitive Games with Entropy Regularization](https://arxiv.org/abs/2105.15186)
derive efficient methods for quantal-response equilibria in zero-sum games.
[Shukla et al. (2025), Generalized Quantal Response Equilibrium](https://arxiv.org/abs/2507.09928)
extend bounded-rational equilibrium and learning to broader general-sum settings.

**Adaptation:** use the same explicit expert/scenario matrix `M` as minimax,
but solve an entropy-regularized mixed game. Its responses satisfy
`p=softmax(M*q/temperature)` and
`q=softmax(-M.T*p/temperature)`. Thirty-two damped logit-response
predictor/corrector iterations, with temperature `0.05`, produce a practical
mixture; they do not certify an equilibrium residual or implement the exact
multiplicative-policy algorithm of Cen et al. The generalized 2025 solution
is research context. Temperature, scenario construction and output
quantization are trading design choices. No claim is made to infer actual
traders' rationality from candles. This differs from both pure maximin and
the cooperative product of surpluses.

### 9. `sleeping_council` — confidence-rated specialist regret

[Luo and Schapire (2015), Achieving All with No Parameters: Adaptive NormalHedge](https://arxiv.org/abs/1502.05934)
give a confidence-rated extension in which an inactive expert receives no
weight and incurs no regret. With causal activity `I_i`, update
`R_i += I_i*u_i`, `C_i += abs(I_i*u_i)` and weight the active expert by
`I_i*[Phi(R_i+1,C_i+1)-Phi(R_i-1,C_i+1)]/2`, where
`Phi(R,C)=exp(max(R,0)²/(3*C))`. Cash remains available.

**Adaptation:** the current expert recommendations define when specialists
participate; this is our market-regime interpretation. The construction is
foundational, not newly published in 2026. It is included alongside the recent
NormalHedge research above as a distinct comparison that separates conditional
participation from the constant-potential clock. It does not repeat R-171's
three ONS leverage accumulators. The learning theorem is not evidence that
the specialists' advice contains net tradable information.

### 10. `defensive_forecast` — kernel calibration against a skeptic

[Vovk, Takemura and Shafer (2005), Defensive Forecasting](https://proceedings.mlr.press/r5/vovk05a.html)
formulate prediction as a game against betting tests. The K29 construction
maintains `z=sum_s phi(p_s,x_s)*(y_s-p_s)` and chooses a zero of
`S(p)=phi(p,x_current)·z` on `[0,1]`, or the corresponding endpoint when
`S` has constant sign. Finite features, including the proposed probability
and current causal expert advice, make the root inexpensive. Only a completed
round reveals the binary up/down outcome.

[Farina and Perdomo (April 2026), An Efficient Black-Box Reduction from Online Learning to Multicalibration](https://arxiv.org/abs/2604.19592)
revisit defensive forecasting within a new general reduction using expected
variational inequalities. **Scope:** we implement the finite-feature K29
special case, not the complete 2026 reduction. Turning a calibrated up
probability into exposure also requires return magnitudes and a cost hurdle;
that mapping is an empirical adaptation. Calibration alone supplies neither
positive expected trading profit nor fast adaptation after a regime change.

## Non-duplication and evaluation boundaries

The ledger already records plain Hedge allocation (`hedge_experts`,
`game_council`), replicator dynamics (`replicator_book`), external position-grid
regret (`regret_grid`), fictitious play (`game_switch`), coin-betting, cognitive
hierarchy, mean-field crowding and robust Kelly. R-189's differentiators are
the learning state, objective and solution concept above, not renamed price
indicators. Shared expert signals are intentional reuse. A candidate whose
new learning channel is inert must be reported as such, not promoted because
it inherited the incumbent's exposure.

The research sources primarily prove statements about bounded games. Real
returns are not bounded losses without a specified normalization; clipping
changes the objective. Discounting, finite quadrature, a restricted action
set, a fixed number of equilibrium iterations and execution deadbands each
limit which exact theorem can be cited. The practical claim is therefore
that these are **theory-informed, causal, cost-tested adaptations**.

BTC's 2023+ history is a repeatedly reused evaluation slice, as recorded in
the ledger, and must not be described as untouched. Freeze candidate and
selection rules before this round's evaluation; report all ten, including
flat, low-cadence and losing results. Use the original broker for net equity
and actual trades. A fee-free view is diagnostic only. The ledger entry and
generated R-189 reports are the authority for measured results, the final
chosen defaults and keep/drop decisions.
