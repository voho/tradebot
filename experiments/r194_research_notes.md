# R-194 research notes — Kelly v4 x transformer hybrids

Operator request (2026-11-18), straight after R-193: *"also try 3
strategies that combine the best kelly-regime with transformer
architecture"*, then *"be creative"*. Written before any R-194 financial
evaluation. R-193's NEGATIVE result was known when this was designed. Its
lesson shaped the design: a transformer *replacing* a simple drift estimate
lost to that estimate. So R-194 never replaces v4's vote. Each hybrid keeps
v4 intact and is exactly v4 whenever its transformer component is silent.

## Constraints inherited from the ledger

- **R-62:** v4's signal lives in the trend *vote*; the volatility scale
  carries none of its replicable property. No hybrid only re-tunes the scale.
- **R-73 (section C):** "do not re-try a never-increase-only bounded-brake
  architecture on `kelly_regime_v4` regardless of the feeding signal". Every
  hybrid can raise exposure as well as cut it.
- **R-33:** holding less is not evidence. R-192's rule, reused here, compares
  against both the parent and a volatility-matched passive hold.
- **R-55 / R-84 / R-88 and others:** confirming votes on v4 have failed many
  times with single external signals, mostly on lead-time gates. Hybrid 1 is
  a confirming vote, so it is judged on the full R-192 battery, not on a
  lead-time gate.

## Sources

| source | used for |
|---|---|
| Vaswani et al. (2017), NeurIPS 30 | Encoder-decoder (cross-)attention: queries from one sequence, keys/values from another (hybrid 1); encoder layer (hybrid 3) |
| Ramsauer et al. (2021), ICLR, *Hopfield Networks is All You Need* | Attention as associative-memory retrieval (hybrids 1, 2) |
| Pritzel et al. (2017), ICML, *Neural Episodic Control*; Blundell et al. (2016), arXiv:1606.04460 | Kernel-weighted retrieval from an episodic memory of the agent's own past states and outcomes (hybrid 2) |
| Kelly (1956); Thorp (2006) | Kelly fraction mean/variance; ratio form keeps v4's calibrated level (hybrid 2) |
| He et al. (2016), CVPR, residual learning; Goyal et al. (2017), arXiv:1706.02677, zero-initialised residual branch | Identity-at-initialisation residual correction (hybrid 3) |
| Salimans et al. (2017), arXiv:1703.03864 | Antithetic ES training on a non-differentiable, cost-charged utility (hybrid 3) |
| Wood et al. (2021), arXiv:2112.08534 | Transformer trained directly on a trading utility; per R-05, its edge assumed 2–3bps across 88–100 instruments |

## The three hybrids

**1. `r194_panel_cross_attention` — cross-attention over a nine-instrument
memory, as a fourth vote.**

- **Query:** the traded asset's daily 7-feature token (R-193's).
- **Keys/values:** tokens and matured, vol-normalised 20-day outcomes of nine
  instruments: Bitstamp BTC and Coinbase ETH, LTC, XRP, LINK, BCH, DASH,
  ETC, XTZ.
- **Readability:** a memory becomes readable on the date of its own outcome's
  close, so calendars never leak.
- **The vote:** the attention drift's sign votes with weight w beside v4's
  three anchors, using R-55's `frac=(3·frac + w·vote)/(3 + w)`.
- **Neighbours:** w ∈ {1, 0.5, 2}.
- **Mechanism:** about 9× more analog memories than one series. This answers
  R-05's breadth objection with a different idea: pool instruments to *learn*
  (sample size), not to *trade* (R-63 priced trading breadth at 1.47 of 8).
- **Attacks:** N≈3 and INFO.
- **Not a duplicate of:** R-63/R-65/R-67 (cross-sectional trading), R-187
  (panel correlation as a brake), or R-193 (single-series memory).

**2. `r194_kelly_episodic_memory` — v4 consults its own episodic memory.**

- **Keys:** R-193's 7 features plus v4's own daily vote and scale.
- **Values:** matured outcomes.
- **The multiplier:** attention retrieves the conditional outcome mean and
  variance, giving a Kelly fraction μ/σ². Its ratio to the unconditional
  (uniform-attention) Kelly fraction multiplies v4's target, clipped to
  [0, 2]. It is 1 whenever the unconditional fraction is not positive or
  history is short.
- **Neighbours:** τ ∈ {1, 0.5, 2}.
- **Mechanism:** v4's vote is right on average. The memory asks whether days
  that looked like today, *in v4's own state*, paid more or less than
  average, and scales by that.
- **Attacks:** ERR and SIZE.
- **Not a duplicate of:** R-37's novel arm, a per-discrete-vote-state Kelly
  fraction *replacing* the vol target, which failed on ETH. This is a
  continuous-similarity *ratio* multiplying the unchanged v4 target, and it
  can only depart from 1 by evidence of a difference.

**3. `r194_residual_transformer` — an identity-initialised residual encoder.**

- **Architecture:** R-193's one-layer encoder, reading 9-feature tokens (the
  7 features plus v4's vote and scale) over the last L days. Its output head
  is initialised to zero, so the untrained model outputs exactly 0 and the
  correction exp(tanh 0) = 1 reproduces v4 bit for bit.
- **Training:** ES (R-193's hyperparameters, seed 194 + year) maximises mean
  `log1p(min(v4·e^{tanh z}, 2)·R_{t+1}) − 0.0041·|Δw|`, refitted each
  calendar year. Weight decay pulls the head back toward the identity.
- **Mechanism:** residual learning on top of a strong baseline. The network
  only has to learn *when v4 is wrong*, and it departs from v4 only where
  that improves cost-charged training growth.
- **Neighbours:** L ∈ {30, 15, 60}.
- **Attacks:** COST and ERR.
- **Not a duplicate of:** R-04 meta-labeling (a logistic gate trained on
  labels, which hurt in-sample) or R-193 (an encoder replacing the signal).

## Anticipated failure modes, named before evaluation

- v4 already captures the trend these components see. The components add
  only noise, which turns into extra fills at 40bp spot cost, and the
  inert-change gate or the cost eats them.
- The panel memory is dominated by high-volatility alts with different
  regimes, so the vote tracks alt behaviour rather than BTC's.
- The episodic Kelly ratio is noisy and whipsaws v4, and the 10% latch turns
  that into turnover.
- The residual encoder trains on 2017–2021 bull years and learns "lean in"
  (correction above 1), then over-levers into 2022. Or it stays near the
  identity and is inert.
- Parent-relative power: R-192 measured 0.38–29.5 years to see +0.20 Sharpe
  against v4. This is likely unreachable on ~3.6 holdout years whatever the
  mechanism, and the bar is not lowered.
