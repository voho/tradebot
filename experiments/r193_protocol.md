# R-193 frozen design — three attention allocators

Written on 2026-11-18 before any R-193 financial evaluation. The operator
requested three strategies based on the transformer architecture of Vaswani
et al. (2017). Step 0: `git fetch` shows `HEAD == origin/main` at `6aae0dd`;
the newest frozen `r<nn>_shared.py` files (R-186–R-188) all have section B
entries and R-189–R-192 are complete; zero consecutive null passes; live
backlog rows B-06 ongoing, B-09 low, B-17 partial, B-28 data-blocked. This is
an operator-requested batch, like R-191/R-192, not a backlog item. It is the
first *test* of R-05's direction, which was closed on grounds only. Sources,
the response to R-05 and prior-round overlap are in
`experiments/r193_research_notes.md`. The only numbers seen before this
freeze are synthetic optimiser and causality checks.

## Mechanisms (fixed before training; no search)

Implementation: `experiments/r193_strategies.py`. All three trade one risky
asset against cash in [0, 1], decide on R-191's complete-UTC-day clock (23:55
close of a day with all 288 bars, fill next open), share its 252-day warmup,
its 0.05 actual-exposure drift band and zero-target exit, and replay all
learning causally in `prepare`. The signal cost is R-191's 40bp + 1bp in
every cell; fees actually charged are the native broker's.

Token (7 features): `log(C_t/C_{t−k})/(σ60·√k)` for k ∈ {1,5,20,60,120},
`log(σ20/σ60)`, `log(σ60/σ120)`; σ_n is the trailing n-observation std of
daily log returns. Memory value: `(log C_{j+20} − log C_j)/(σ60_j·√20)`,
usable at t only if `j + 20 ≤ t` (observed-day positions).

1. **`r193_attention`** — single head over all 7 features; ≥100 matured
   memories; features standardised by moments through t; scores
   `(q·k − ½|k|²)/(τ√7)`; readout ū. Drift `ū·σ60/√20 + ½σ60²`, variance
   `σ60²`, into R-191 `_proximal_weight` with penalty 2. τ = 1; neighbours
   0.5 and 2.
2. **`r193_multihead`** — heads short {k=1,5}, trend {20,60,120}, volatility
   {2 vol ratios}, each τ = 1 at √d_h. Standardised ridge output projection
   (penalty ridge·n) on matured head readouts, fitted at the first observed
   day of each calendar year, or as soon as 252 rows exist if no model yet.
   Same drift/proximal sizing. ridge = 1; neighbours 0.25 and 4. Flat until
   the first fit.
3. **`r193_transformer`** — tokens standardised with moments through the refit
   day, clipped ±5; embedding 7→8 + sinusoidal position; 2-head attention
   read out at the last position (exactly causal self-attention's last
   output); residual + LayerNorm; ReLU FFN 16; residual + LayerNorm; sigmoid
   exposure. 601 parameters. Antithetic ES, 24 pairs, σ = 0.05, 200 Adam
   steps (lr 0.02), init N(0, 0.1²), seed `193 + refit year`, objective mean
   `log1p(w_t R_{t+1}) − 0.0041|w_t − w_{t−1}| − 1e-3·mean(θ²)` over every
   prior day whose next return is known. Same refit schedule and 252-row
   first-fit rule. Context L = 30; neighbours 15 and 60. Flat until fitted.

## Data, cells and counts

Reuse R-191's evaluation unchanged: `r190_eval.evaluate` fresh-account native
engine; `r191_eval.specifications` (same cells, same seed-191 windows); the
same committed Bitstamp BTC spot, Coinbase ETH, Deribit BTC perpetual and
matching 8-hour funding files, ending 2026-08-12 00:40 UTC; training
preparation sees only bars before 2023. Inner-train ends 2020-12-31,
inner-validation is 2021–2022, holdout starts 2023-01-01. Spot 40bp + 1bp
with a $10 minimum; 10bp discount BTC cell; ETH 40bp + 1bp stress; funded
perpetual 5bp + 1bp plus funding.

Three primaries and both controls (`buy_and_hold`, its funded reference the
unlevered c = 1 constant-exposure hold, and `kelly_regime_v4`) receive three
training cells, four named holdout cells and 48 windows (24 spot + 24 funded
perpetual). Six neighbours receive the seven named cells. **317 core
evaluations: 33 training, 284 holdout.** Controls are rerun rather than
copied; agreement with R-191's control receipts is an audit check, not an
input.

## Risk matching and inference

Identical to R-191: per primary, ConstantExposureHold matched to each
validation, primary-holdout and window cell's realised daily volatility,
start c = 0.5, ≤3 attempts, 2% tolerance, otherwise INVALID (never a win);
every attempt counted (≤18 training, ≤450 holdout). Paired 30-day stationary
block bootstrap, 2,000 resamples, seed 191, for daily Sharpe, log growth and
daily max drawdown against the matched hold. Training-only power projection
for the +0.20 Sharpe floor as in R-191; the threshold is not lowered if the
implied history is unreachable.

Freeze source/data hashes, this protocol and the validation-only descriptive
ranking (mean spot/funded validation Sharpe, ties fewer spot fills) before
any holdout cell. Deflated Sharpe uses 9 local trials and program trials
4,232 (cumulative after R-192) + every R-193 holdout core/matching
evaluation; trial SD = max(0.418538, SD of the 9 spot validation Sharpes).

## Exhaustive promotion decision (R-191's seven gates, unchanged)

PROMOTED iff **all seven** pass for that primary; otherwise NEGATIVE. Each
primary is decided independently, so every outcome maps to exactly one verdict.

1. Spot and funded, validation and holdout: valid risk match and daily
   Sharpe strictly > matched passive + 0.20.
2. Both primary holdouts: paired 95% lower bounds of ΔSharpe and Δlog-growth
   strictly > 0.
3. Both primary holdout balances exceed the full passive references.
4. Falsification (pre-registered): all four named holdout balances > $1,000;
   ETH daily Sharpe ≥ ETH buy-and-hold's; no primary cell liquidates.
5. Each family of three is profitable with daily-Sharpe spread ≤ 0.20 in each
   of the four primary validation/holdout cells (plateau, not peak).
6. Program deflated Sharpe ≥ 0.95 on both primary holdout markets.
7. ≥ 13/24 window wins per market against valid window-specific matches.

Descriptive only (no gate): the same-sizing linear comparator
`r191_cost_proximal` from R-191's frozen receipts; transformer exposure
statistics (mean, time near 1) to test the "learned always long" failure.

No tail-only route, no post-result reinterpretation. A negative stays in
`experiments/`, unregistered. Before the holdout, an independent reviewer
that has not seen any result reads the strategy code for lookahead and
arithmetic errors; findings are fixed (bugs only) and recorded. All tests,
including strict causality, must pass. Record the ledger entry, counters and
backlog; commit and push the branch; do not merge.
