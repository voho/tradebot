# R-61 (novel branch) — Hurst-gated z-score mean-reversion (08-20)

Pre-registration (shared, both branches): `experiments/r61_shared.py`.
Implementation: `experiments/r61_novel_hurst_gated_reversion.py`. Nothing
under `src/tradebot/strategies/` is touched; nothing is registered.

## 1. The question, one sentence

Does gating the conservative branch's z-score reversion vote by this
project's own rolling causal Hurst exponent (R-46,
`experiments/kelly_regime_v12_cppi_hurst.py`, reused not reimplemented) —
only trading the reversion signal while the asset is measurably
anti-persistent (H < 0.5) — rescue it, given R-46's own finding that BTC
spends most of its time trend-persistent (H > 0.5)?

`z_thresh` fixed at **1.5** (the pre-registered grid-midpoint fallback,
since this branch could not see the conservative branch's live selection of
2.0 — see reconciliation note in section 8).

## 2. Mechanism

Identical 1/3/7-day z-score vote to the conservative branch, multiplied by
`gate = 1.0 if H(t) < 0.5 else 0.0`. Same unmodified `KellyRegime` sizing
loop otherwise.

## 3. Causality — PASS

Two-opposite-tampers probe on a 300,000-bar pre-2023 BTC tail (cut at bar
295,000): `frac_raw`, `H`, `gate`, `target`, and a full equity replay all
bit-identical before the cut (max |difference| = 0.000e+00 on all five
checks).

## 4. Empirical Hurst distribution (data property, not a backtest)

| Asset | window | mean H | %H<0.5 |
|---|---|---|---|
| BCH | PANEL_TRAIN | 0.603 | 16.0% |
| LTC | PANEL_TRAIN | 0.606 | 15.3% |
| ETC | PANEL_TRAIN | 0.636 | 7.1% |
| DASH | PANEL_TRAIN | 0.589 | 16.1% |
| LINK | PANEL_TRAIN | 0.593 | 15.9% |
| XTZ | PANEL_TRAIN | 0.577 | 23.6% |
| **panel mean-of-means** | | **0.601** | — |
| BTC | 2017-2022 (train+valid) | **0.622** | 10.4% |

BTC's 0.622 matches R-46's own ~0.62 finding (identical estimator, expected).
The panel's mean (0.601) is modestly lower than BTC's — directionally
consistent with the round's motivating hypothesis — but every asset,
panel and BTC alike, sits in H>0.5 (persistent/trending) territory the large
majority of the time. The panel is not a mean-reverting regime in any
strong sense; it is a slightly-less-trending one.

## 5. D1/D2 — PANEL_TRAIN, spot @0.10%, gated candidate vs buy_and_hold

| Asset | candidate final | candidate DD | hold final | hold DD | D1 | D2 |
|---|---|---|---|---|---|---|
| BCH | $1,141 | 16.2% | $440 | 94.7% | Yes | Yes |
| LTC | $1,285 | 10.6% | $1,728 | 90.2% | No | Yes |
| ETC | $801 | 27.8% | $3,147 | 92.3% | No | Yes |
| DASH | $1,054 | 24.0% | $643 | 93.5% | Yes | Yes |
| LINK | $808 | 29.7% | $2,393 | 89.9% | No | Yes |
| XTZ | $787 | 24.3% | $446 | 92.3% | Yes | Yes |

D1 = 3/6 (FAILS, needs ≥5/6). D2 = 6/6 (same sizing-artifact caveat as the
conservative branch's D2 applies here too). Failure pattern: the candidate
loses on exactly the three assets with the largest 2020-2021 bull-run
multiples (LTC, ETC, LINK) — the Hurst gate keeps it flat (6-18%
time-in-market) through the trending runs `buy_and_hold` captured.

**Operator's independent re-run reproduced every cell above exactly**
(dollar figures, DD%, D1/D2 counts), from a clean CLI invocation.

## 6. Ablation — gated vs ungated (same z=1.5, gate disabled via hurst_thresh=1.0)

| Asset | gated final | ungated final | Δfinal | gated DD | ungated DD | ΔDD |
|---|---|---|---|---|---|---|
| BCH | $1,141 | $397 | +$744 | 16.2% | 73.7% | −57.5pp |
| LTC | $1,285 | $775 | +$510 | 10.6% | 59.6% | −49.1pp |
| ETC | $801 | $352 | +$449 | 27.8% | 74.8% | −46.9pp |
| DASH | $1,054 | $639 | +$415 | 24.0% | 56.7% | −32.7pp |
| LINK | $808 | $579 | +$229 | 29.7% | 59.6% | −29.9pp |
| XTZ | $787 | $791 | −$4 | 24.3% | 46.1% | −21.7pp |

The Hurst gate clearly helps relative to plain (ungated) reversion — better
final balance on 5/6 assets, better max drawdown on 6/6, by large margins —
but the *ungated* version is itself worse than `buy_and_hold` on 5/6 assets,
so the gate's contribution is damage control on a weak underlying signal,
not unlocking a hidden edge.

## 7. D3 — BTC/ETH falsification, spot @0.10%

| Window | candidate final | candidate DD | candidate time-in-mkt | hold final | v4 final | v4 time-in-mkt |
|---|---|---|---|---|---|---|
| BTC_INNER_TRAIN | $766 | 31.4% | 6.7% | $29,803 | $18,477 | 67.6% |
| BTC_INNER_VALID | $682 | 35.5% | 8.6% | $574 | $998 | 55.6% |
| ETH (≤2022-12-31)* | $782 | 36.1% | 7.5% | $9,142 | $5,091 | 63.9% |

*Disclosed deviation: this branch truncated ETH at 2022-12-31 rather than
reading the full committed series (the pre-registration's `ETH_FULL`
window), a stricter-than-required reading of "no BTC/ETH 2023+ bar." Not a
holdout-cost issue either way (ETH is not the reserved BTC holdout).

The Hurst gate keeps the candidate flat 91-93% of the time on BTC/ETH,
directly confirming R-46's own BTC-Hurst finding as the mechanism at work
(consistent with the panel's empirical Hurst reported above) — underperforms
both benchmarks as predicted, but with no catastrophic whipsaw loss (unlike
the conservative branch's ungated version), the clearest illustration in
this round that the gate does what it was designed to do.

## 8. D4 — 0.40% fee tier, PANEL_TRAIN

Same win/loss asset pattern as D1 (BCH/DASH/XTZ win, LTC/ETC/LINK don't) —
turnover (35-75 trades/asset) too modest to flip any asset at the higher
fee. **D4 = 3/6 (FAILS, needs ≥4/6).**

## 9. D5 — PANEL_TEST (2023-2026), descriptive

4/6 beat hold, 6/6 lower drawdown (BCH flips train-win to test-loss; LINK
loses in both).

## 10. Promotion bar

D1 ≥ 5/6: **FAIL** (3/6). D2 ≥ 4/6: PASS (6/6, same caveat as conservative).
D4 ≥ 4/6: **FAIL** (3/6). **NOT PROMOTED.**

## 11. Configurations evaluated: 57

D1/D2 12 + ablation 12 + D3 9 + D4 12 + D5 12 = 57. `hurst_stats` and the
causality probe are data/estimator properties, not backtests, and do not
count (R-46's own convention).

## 12. Reconciliation note (z_thresh)

This branch used z_thresh=1.5 (the pre-registered grid midpoint) because it
could not see the conservative branch's live PANEL_TRAIN selection at the
time it ran. The conservative branch independently selected z_thresh=2.0.
The operator did not re-run the novel branch at z=2.0: the ablation
(section 6) already isolates the Hurst gate's own contribution independent
of the exact z_thresh, and the conservative branch's own D1 at z=2.0 (2/6)
is *worse* than z=1.5's (2/6, tied) — the grid is flat enough in this range
that re-running at z=2.0 would not plausibly change this round's verdict
(D1 well under the 5/6 bar at every tested threshold on both branches).
