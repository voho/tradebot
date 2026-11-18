# R-193 research notes — attention-based allocation

Operator request (2026-11-18): *develop three strategies based on the
transformer architecture of "Attention Is All You Need".* This file records
the sources, what they claim on what data and cost, how R-193 answers R-05's
standing reasons for not attempting deep learning, and where the three
families overlap earlier rounds. It was written before any R-193 financial
evaluation; synthetic-only optimiser checks are the only numbers seen.

## Sources

| source | claim | data / cost / breadth |
|---|---|---|
| Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS 30 | Scaled dot-product attention `softmax(QKᵀ/√d_k)V`, multi-head attention, sinusoidal positional encoding and residual + LayerNorm sublayers replace recurrence | Machine translation (WMT 2014). No financial claim. |
| Tsai et al. (2019), *Transformer Dissection*, EMNLP | Attention is a kernel smoother: a softmax over a similarity kernel weighting values | Theory + NLP benchmarks. |
| Ramsauer et al. (2021), *Hopfield Networks is All You Need*, ICLR | Attention is one retrieval step of a continuous associative memory | Theory + immune-repertoire / UCI tasks. |
| Lorenz (1969), *J. Atmos. Sci.* 26 | Analog forecasting: predict from the outcomes of past states that resemble today's | Weather; the oldest form of family 1's mechanism. |
| Lim, Zohren & Roberts (2019), *JFDS* 1(4) | Networks trained directly on Sharpe beat classical time-series momentum | Per R-05: portfolio of futures, costs ~2–3bps. |
| Wood, Giegerich, Roberts & Zohren (2021), *Trading with the Momentum Transformer*, arXiv:2112.08534 | Attention-based momentum model, Sharpe-trained, adapts across regimes | Per R-05/RESEARCH.md: 88–100 instruments, 2–3bps costs. |
| Zeng et al. (2023), *Are Transformers Effective for Time Series Forecasting?*, AAAI | A one-layer linear model beats transformer forecasters on nine benchmarks | Long-horizon forecasting benchmarks, no costs. |
| Salimans et al. (2017), *Evolution Strategies as a Scalable Alternative to RL*, arXiv:1703.03864 | Antithetic Gaussian ES estimates gradients of non-differentiable returns | RL benchmarks; R-193 uses it to train on a cost-charged utility without hand-written backprop. |
| Ba, Kiros & Hinton (2016), arXiv:1607.06450; Kingma & Ba (2015), ICLR | LayerNorm; Adam | Optimisation components only. |

No paper above shows profit for a single crypto instrument at 41bp signal
cost. Nothing here is treated as evidence that R-193 will work.

## Answering R-05

R-05 closed deep learning *on grounds* (no test) for three reasons. R-193
addresses two of them and leaves the third explicitly open:

1. **Cost (2–3bps published vs 10bps+ here).** Addressed by construction:
   every family decides once per complete UTC day and charges R-191's 41bp
   signal cost inside its own objective — family 1 and 2 through R-191's
   proximal L1 hurdle, family 3 in its training utility. The executed fees
   are the native broker's, separately.
2. **Breadth (88–100 instruments vs one).** *Not* addressed. R-63 measured
   this repo's panel breadth at 1.47 of 8, so pooling would not buy the
   published diversification. R-193 tests whether the architecture extracts
   anything from one series — not whether the published portfolio edge
   transfers.
3. **Linear baselines win (Zeng 2023).** Partly addressed: families 1 and 2
   reuse R-191's `_proximal_weight` sizing unchanged, so R-191's
   `r191_cost_proximal` (trailing 63-day mean/variance into the same solver)
   is a same-sizing linear comparator. That comparison is descriptive, read
   from R-191's frozen receipts, and is not a promotion gate.

## The three families

All trade one risky asset against cash in [0, 1], R-191's daily clock,
252-day warmup, 0.05 drift band and causal replay in `prepare`.

**Shared token** (7 features, scale-free): `log(C_t/C_{t-k}) / (σ60·√k)` for
k ∈ {1, 5, 20, 60, 120}, `log(σ20/σ60)`, `log(σ60/σ120)`, σ = trailing
daily log-return std. **Memory value**: the vol-normalised 20-day forward log
return, usable only once matured (`j + 20 ≤ t`).

1. **`r193_attention` — single-head attention analog.** Query = today's token,
   keys = every matured past token, values = their outcomes; standardised
   with feature moments through today. Scores use the key-norm bias
   `q·k − ½|k|²` (exactly a softmax over `−|q−k|²/2` after cancellation),
   scaled by `τ·√d`. Drift `ū·σ60/√20 + σ²/2` and variance `σ60²` go into
   R-191's proximal solver (penalty 2). Mechanism: if today's state resembles
   past states, their realised outcomes are a better drift estimate than the
   trailing mean. Neighbours: τ ∈ {1, 0.5, 2}. Constraint: SIZE (drift
   estimator in a cost-aware Kelly-style sizer). Not a duplicate of R-03/
   R-02/R-82 (regime classifiers) or R-191 (trailing-mean drift): this is a
   nonparametric conditional-drift estimate. Closest prior art is L-12's
   Bayesian crowd and R-34, which estimate state-conditional drift from a
   handful of vote states rather than a continuous similarity kernel.
2. **`r193_multihead` — multi-head attention + learned W_O.** Three heads over
   disjoint subspaces — short `{k=1,5}`, trend `{20,60,120}`, volatility
   `{σ20/σ60, σ60/σ120}` — each attending at τ=1. Their concatenated readouts
   pass through a standardised ridge output projection fitted on matured
   outcomes only, refitted at the first observed day of each calendar year
   (first fit once 252 rows exist). Same proximal sizing. Mechanism: let past
   out-of-sample accuracy decide which subspace's analogs to trust; ridge
   shrinks uninformative heads toward the unconditional mean. Neighbours:
   ridge ∈ {1, 0.25, 4} (slope shrink ≈ 1/(1+ridge)). Constraint: ERR (head
   weights earned on matured forecasts, shrunk by default). Not a duplicate of
   R-147/R-189/R-191 expert councils, which weight *positions* by realised
   PnL; this weights *forecasts* by regression on outcomes.
3. **`r193_transformer` — trained encoder.** Last L daily tokens → linear
   embedding (d=8) + sinusoidal positional encoding → 2-head self-attention →
   residual + LayerNorm → FFN (16) → residual + LayerNorm → sigmoid exposure.
   Trained from a fixed seed by antithetic ES (24 pairs, σ 0.05, 200 Adam
   steps, lr 0.02, decay 1e-3) to maximise mean `log(1 + w_t R_{t+1}) −
   0.0041·|Δw_t|`, refitted on all prior days each calendar year (same first-
   fit rule). 601 parameters. Mechanism: Momentum-Transformer-style direct
   utility training, with the cost this repo actually faces inside the loss.
   Neighbours: L ∈ {30, 15, 60}. Constraint: COST. This is the first test of
   R-05's direction in this repo.

## Anticipated failure modes, named before evaluation

- BTC's strongly positive in-sample drift dominates every memory bank, so
  attention forecasts collapse toward the unconditional mean and families 1–2
  become a near-constant fractional long that holds less than buy-and-hold —
  R-33's exposure artifact, which the matched-risk gates exist to catch.
- Diffuse attention over ~1,000–3,000 keys gives tiny conditional variation;
  the proximal hurdle then suppresses trading altogether.
- The transformer, trained on 2017–2021 bull years, learns "always long"
  (w→1) and adds nothing over passive at matched risk, or overfits ~300–1,500
  samples (N≈3 regimes) and whipsaws after each refit.
- Neighbours disagree (no plateau), or the three families agree only because
  they share the same token and drift.
- Every effect sits inside the ±0.2 Sharpe noise floor (R-20), the inherited
  power problem R-191/R-192 measured at decades of required history.
