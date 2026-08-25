# R-142 novel -- two-sided Deribit front/next-quarter term-structure-slope SIZE dampener on `kelly_regime_v4`

Code: `experiments/r142_novel_slope_dampener.py` (appended after the frozen pre-registration banner; banner unedited). Built on the frozen, read-only `experiments/r142_shared.py` (`dual_quarter_slope`, `slope_zscore`, `NOVEL_KAPPA_GRID`, coverage constants). Not `@register`ed -- experiments/-only, nothing committed by this session. Mechanism (frozen before any number): `scale_novel = scale_v4 * (1 - kappa*tanh(slope_z/2))`, kappa swept on the fixed pre-registered grid `(0.0, 0.10, 0.20, 0.30)`, never solved for by any matching procedure -- deliberately two-sided and NOT equality-mean-exposure-matched, the specific construction chosen to avoid R-141's provable kappa=0 degeneracy.

## 1. Step 0 -- mandatory pre-flight checks

**Wiring self-test.** `v4_scale_series` (extracted verbatim from `KellyRegimeV3.prepare()`, matching R-125/R-141's own convention), recombined with the vote/deadband logic, reproduces `kelly_regime_v4.prepare()`'s own `target` column exactly on BTC inner-train+inner-validation: **PASS**.

**(a) Identity-recovery.** `NovelSlopeDampener(kappa=0.0).prepare(df)["target"]` vs `kelly_regime_v4.prepare(df)["target"]` on BTC (2017-01-01..2022-12-31), `np.array_equal`: **True**. A second identity check at the backtest level (Section 3 below) additionally confirms all 8 kappa=0 (asset x market x period) cells are bit-identical to the separately-run `kelly_regime_v4` reference cells (same Sharpe, drawdown, trade count, fill count on every one): **True**.

**(b) Not-degenerate-by-construction.** `mean(scale_novel)` at every grid point, BTC inner-train (2017-01-01..2020-12-31), vs `mean(scale_v4) = 0.776763`:

| kappa | mean(scale_novel) | gap vs v4 |
|---|---|---|
| 0.00 | 0.776763 | +0.000000 |
| 0.10 | 0.778687 | +0.001924 |
| 0.20 | 0.780611 | +0.003849 |
| 0.30 | 0.782536 | +0.005773 |

The gap is **positive and increasing** in kappa -- the opposite of R-141's signature (there, an upper-bounded-at-1 multiplier forced the gap to be non-positive and non-increasing, collapsing calibration to kappa=0 exactly). Here the two-sided `tanh` term is *not* symmetric in its effect on the mean once combined with the hysteresis-latched `scale_v4` (upside amplification on backwardation days lands on bars where `scale_v4` also tends to be elevated), so the mean drifts slightly *up*, not down, with kappa -- non-degenerate by this signature. **PASS.**

**(c) Causal truncation probe**, on the full raw-OHLCV+quarterly -> `dual_quarter_slope` -> `slope_zscore` -> `scale_novel` -> `target` pipeline (not merely `r142_shared`'s own causality claim taken on faith), diagnostic kappa=0.20 (so the probe exercises the actual multiplication, not the kappa=0 no-op), truncating each asset's spot series ~400k bars in and comparing target values before the cut (minus a 90-day warmup-tail buffer) against the full-series run:

- BTC: **PASS**
- ETH: **PASS**

All three Step-0 checks pass; proceeding to Step 3.

## 2. Data

BTC: 1,010,889 bars, 2017-01-01 -> 2026-08-12 (spot + Deribit quarterly futures, this round's own coverage-extension re-fetch). `slope_z` NaN fraction 20.79% (fewer than two simultaneously-listed quarterlies). ETH: 781,506 bars, 2019-03-14 -> 2026-08-19, `slope_z` NaN fraction 15.37%. Where `slope_z` is NaN the multiplier defaults to 1.0 (v4's own scale, unmodified) -- a disclosed causal design choice, not a fitted patch. Inner-train ends 2020-12-31, inner-validation is 2021-01-01..2022-12-31, holdout (>= 2023-01-01) untouched until Step 4 below.

## 3. B1/B3 -- kappa-grid plateau (4 kappa x 2 assets x 2 markets x 2 periods = 32 candidate cells + 8 `kelly_regime_v4` reference cells)

```
kappa=0.00  BTC       spot full_inner  sharpe=+1.5256 (v4=+1.5256, d=+0.0000)  dd=43.25% (v4=43.25%)  trades=123 fills=725
kappa=0.00  BTC       spot  inner_val  sharpe=+0.1420 (v4=+0.1420, d=+0.0000)  dd=33.18% (v4=33.18%)  trades=52 fills=256
kappa=0.00  BTC futures_5x full_inner  sharpe=+1.7305 (v4=+1.7305, d=+0.0000)  dd=35.29% (v4=35.29%)  trades=123 fills=402
kappa=0.00  BTC futures_5x  inner_val  sharpe=+0.2514 (v4=+0.2514, d=+0.0000)  dd=32.29% (v4=32.29%)  trades=52 fills=143
kappa=0.00  ETH       spot full_inner  sharpe=+1.3405 (v4=+1.3405, d=+0.0000)  dd=33.09% (v4=33.09%)  trades=72 fills=491
kappa=0.00  ETH       spot  inner_val  sharpe=+0.4999 (v4=+0.4999, d=+0.0000)  dd=33.19% (v4=33.19%)  trades=47 fills=315
kappa=0.00  ETH futures_5x full_inner  sharpe=+1.3213 (v4=+1.3213, d=+0.0000)  dd=32.09% (v4=32.09%)  trades=72 fills=217
kappa=0.00  ETH futures_5x  inner_val  sharpe=+0.6321 (v4=+0.6321, d=+0.0000)  dd=31.59% (v4=31.59%)  trades=47 fills=134

kappa=0.10  BTC       spot full_inner  sharpe=+1.7293 d=+0.2037  dd=43.10%  fills=943
kappa=0.10  BTC       spot  inner_val  sharpe=+0.3603 d=+0.2184  dd=29.65%  fills=280
kappa=0.10  BTC futures_5x full_inner  sharpe=+1.7668 d=+0.0364  dd=34.38%  fills=427
kappa=0.10  BTC futures_5x  inner_val  sharpe=+0.2675 d=+0.0161  dd=32.15%  fills=149
kappa=0.10  ETH       spot full_inner  sharpe=+1.3429 d=+0.0024  dd=33.09%  fills=575
kappa=0.10  ETH       spot  inner_val  sharpe=+0.4956 d=-0.0043  dd=32.80%  fills=369
kappa=0.10  ETH futures_5x full_inner  sharpe=+1.4231 d=+0.1018  dd=32.09%  fills=212
kappa=0.10  ETH futures_5x  inner_val  sharpe=+0.7178 d=+0.0857  dd=29.69%  fills=130

kappa=0.20  BTC       spot full_inner  sharpe=+2.3621 d=+0.8365  dd=42.35%  fills=2330
kappa=0.20  BTC       spot  inner_val  sharpe=+1.3962 d=+1.2542  dd=17.64%  fills=765
kappa=0.20  BTC futures_5x full_inner  sharpe=+2.0172 d=+0.2867  dd=34.62%  fills=585
kappa=0.20  BTC futures_5x  inner_val  sharpe=+0.3163 d=+0.0649  dd=28.37%  fills=184
kappa=0.20  ETH       spot full_inner  sharpe=+1.1358 d=-0.2048  dd=34.31%  fills=2893
kappa=0.20  ETH       spot  inner_val  sharpe=+0.2390 d=-0.2610  dd=36.26%  fills=1640
kappa=0.20  ETH futures_5x full_inner  sharpe=+1.3846 d=+0.0634  dd=32.09%  fills=270
kappa=0.20  ETH futures_5x  inner_val  sharpe=+0.4374 d=-0.1947  dd=28.27%  fills=151

kappa=0.30  BTC       spot full_inner  sharpe=+2.9420 d=+1.4164  dd=41.42%  fills=4379
kappa=0.30  BTC       spot  inner_val  sharpe=+2.4684 d=+2.3264  dd=13.81%  fills=1571
kappa=0.30  BTC futures_5x full_inner  sharpe=+2.5977 d=+0.8672  dd=33.98%  fills=1059
kappa=0.30  BTC futures_5x  inner_val  sharpe=+1.0710 d=+0.8195  dd=16.30%  fills=273
kappa=0.30  ETH       spot full_inner  sharpe=+0.8112 d=-0.5294  dd=49.77%  fills=7207
kappa=0.30  ETH       spot  inner_val  sharpe=-0.1303 d=-0.6302  dd=50.79%  fills=3982
kappa=0.30  ETH futures_5x full_inner  sharpe=+1.2476 d=-0.0736  dd=32.09%  fills=630
kappa=0.30  ETH futures_5x  inner_val  sharpe=+0.3243 d=-0.3078  dd=31.65%  fills=341
```

**B3 plateau view (inner-validation only, all 4 kappa cells side by side):**

| asset | market | v4 sharpe | k=0.00 | k=0.10 | k=0.20 | k=0.30 |
|---|---|---|---|---|---|---|
| BTC | spot | +0.1420 | d=+0.0000 | d=+0.2184 | d=+1.2542 | d=+2.3264 |
| BTC | futures_5x | +0.2514 | d=+0.0000 | d=+0.0161 | d=+0.0649 | d=+0.8195 |
| ETH | spot | +0.4999 | d=+0.0000 | d=-0.0043 | d=-0.2610 | d=-0.6302 |
| ETH | futures_5x | +0.6321 | d=+0.0000 | d=+0.0857 | d=-0.1947 | d=-0.3078 |

**Not a plateau.** On BTC spot the effect *escalates* sharply (+0.22 -> +1.25 -> +2.33), not flat -- the pre-registered B3 test explicitly asked for "monotonic-ish or flat...not a spike at one value," and while the BTC-spot sequence is technically monotonic in sign, its magnitude is not remotely flat: this reads as the mechanism reacting explosively to a small number of large episodes inside the 2021-2022 window, not a stable, generalizable exposure adjustment. BTC futures is much more muted (+0.02 to +0.82) and ETH inverts sign entirely from kappa=0.20 onward on both markets.

## 4. B4 -- ETH sign-replication (this project's single most common failure mode)

| market | kappa | d_sharpe BTC | d_sharpe ETH | same sign |
|---|---|---|---|---|
| spot | 0.10 | +0.2184 | -0.0043 | **No** |
| spot | 0.20 | +1.2542 | -0.2610 | **No** |
| spot | 0.30 | +2.3264 | -0.6302 | **No** |
| futures_5x | 0.10 | +0.0161 | +0.0857 | Yes |
| futures_5x | 0.20 | +0.0649 | -0.1947 | **No** |
| futures_5x | 0.30 | +0.8195 | -0.3078 | **No** |

**1 of 6 (kappa, market) cells replicate sign on ETH. B4 FULL PASS: False.** Precisely the BTC-pass/ETH-invert pattern named in this round's own pre-registered failure mode (R-33/R-57/R-62/R-64/R-113/R-127/R-137 and others).

## 5. B5 / R-33 risk-match

Time-in-market and realized annualized volatility, `kelly_regime_v4` vs every kappa candidate, inner-validation:

| asset | market | v4 tim / vol | k=0.10 | k=0.20 | k=0.30 |
|---|---|---|---|---|---|
| BTC | spot | 55.6% / 0.2879 | 55.6% / 0.2910 (+1.1%) | 55.6% / 0.2787 (+3.2%) | 55.6% / 0.2819 (+2.1%) |
| BTC | futures_5x | 55.6% / 0.2820 | 55.6% / 0.2921 (+3.6%) | 55.6% / 0.2896 (+2.7%) | 55.6% / 0.2884 (+2.3%) |
| ETH | spot | 63.2% / 0.3416 | 63.3% / 0.3429 (+0.4%) | 63.4% / 0.3334 (+2.4%) | 63.4% / 0.3308 (+3.1%) |
| ETH | futures_5x | 63.2% / 0.3565 | 63.3% / 0.3463 (+2.8%) | 63.4% / 0.3436 (+3.6%) | 63.4% / 0.3227 (+9.5%) |

Time-in-market is essentially unchanged (the dampener redistributes exposure magnitude within already-held positions, exactly as designed -- it does not change *whether* the strategy holds a position, only how much). Realized volatility divergence stays under the 15% flag threshold in every one of the 12 cells shown (largest: +9.5%, ETH futures_5x at kappa=0.30). **No cell is flagged.** This confirms the dramatic BTC-spot Sharpe swings in Section 3 are not "holding less" or "holding more" in disguise -- the exposure *level* is essentially matched; what moved is the *timing* of exposure within an already-similar holding pattern, concentrated enough in a few episodes to move BTC-spot Sharpe by up to +2.33 while leaving realized vol nearly unchanged.

## 6. Step 3 decision and CONDITIONAL Step 4

Per the frozen conditional-Step-4 rule (at least one kappa>0 beats v4 on Sharpe **or** drawdown on both BTC and ETH inner-validation, without the 15% risk-mismatch flag, on at least one market), three (market, kappa) pairs cleared this per-market OR-based screen: `(spot, 0.10)`, `(futures_5x, 0.10)`, `(futures_5x, 0.20)` -- note this screen is looser than B4's full sign-replication test (ETH's Sharpe can worsen slightly while its drawdown still improves, as at spot/kappa=0.10), so passing it is not evidence the mechanism itself is sound; it only means the conditional trigger for spending a holdout consultation was met, per the pre-registered wording. Step 4 accordingly ran (holdout, 0.40% taker fee tier, BTC and ETH, both flagged markets):

| market | asset | kappa | sharpe_cand | sharpe_v4 | beats v4 |
|---|---|---|---|---|---|
| futures_5x | BTC | 0.10 | +1.0649 | +1.0042 | **Yes** |
| futures_5x | BTC | 0.20 | +0.8521 | +1.0042 | No |
| futures_5x | ETH | 0.10 | +0.6655 | +0.7410 | No |
| futures_5x | ETH | 0.20 | -0.0571 | +0.7410 | No |
| spot | BTC | 0.10 | +0.9207 | +0.9399 | No |
| spot | BTC | 0.20 | -0.1833 | +0.9399 | No |
| spot | ETH | 0.10 | +0.0589 | +0.5961 | No |
| spot | ETH | 0.20 | -2.9343 | +0.5961 | No |

**No (market, kappa) combination beats `kelly_regime_v4` on both BTC and ETH at the 0.40% fee tier out-of-sample.** ETH is decisively worse everywhere in the holdout at every tested kappa (as low as -2.93 Sharpe on spot at kappa=0.20 -- effectively the exact same amplify-a-few-episodes fragility already visible in Section 3, now landing on the holdout's own idiosyncratic ETH episodes with the wrong sign). Step 4: **FAIL.**

## 7. Configurations evaluated

**52 total** (Step-0 diagnostics -- the wiring self-test, the identity check, the causal truncation probes on both assets -- run but not counted toward this figure, per this project's established convention, R-141's own accounting):
- B1/B3: 8 `kelly_regime_v4` reference cells (2 assets x 2 markets x 2 periods) + 32 candidate cells (4 kappa x 2 assets x 2 markets x 2 periods) = 40.
- B4/B5: read off the same 32 B1 cells, zero additional backtests.
- Step 4 (conditional, triggered): 4 `kelly_regime_v4` holdout-fee-tier reference cells (2 flagged markets x 2 assets) + 8 candidate holdout-fee-tier cells (2 flagged markets x 2 assets x 2 flagged kappas) = 12.
- 40 + 12 = **52**.

## 8. Causality / no-lookahead status

- Custom causal-truncation probe on the FULL raw-OHLCV+quarterly -> slope -> slope_z -> scale_novel -> target pipeline: **PASS** (BTC and ETH, both).
- `pytest tests/test_causality_strict.py`: **51 passed**, 0 failed (33.6s). (`NovelSlopeDampener` is not `@register`ed, so it is not itself swept by this suite's `available_strategies()` loop; its own causality argument rests on the dedicated truncation probe above plus the byte-identical reuse of `kelly_regime_v4`'s own already-verified state machine.)
- Full `pytest -q`: **516 passed**, 0 failed (197.9s).
- The BTC-spot magnitude in Section 3 (Sharpe swinging from +0.14 to +2.47 at kappa=0.30, inner-validation) is large enough to demand the "too good is a bug report first" scrutiny this project's routine requires (R-21). Checked directly: the run is not liquidated (`liquidated=False`), `target` stays finite and bounded (max 1.89, within the strategy's own uncapped-by-market-leverage vol-target formula; the broker's own leverage clamp, verified in `tradebot/strategy.py`'s `order_notional`, caps the *executed* spot position at 1.0x notional regardless of the internal `target` value), and B5's realized-volatility check above shows the arm is risk-matched to v4 within 3.2% on this exact cell -- so the effect is a real, if narrow, timing effect on this specific asset/period/kappa combination, not a leverage or lookahead artifact. It fails to generalize to ETH (Section 4) and fails the holdout fee-tier check (Section 6) regardless.

## 9. Verdict

**NEGATIVE.** Step 0 passes cleanly (identity-recovery exact at both the target-array and backtest level; the two-sided, non-equality-matched construction is confirmed non-degenerate -- mean(scale_novel) moves slightly *away* from, not toward, v4's own mean as kappa rises, the opposite of R-141's forced-collapse signature; causal truncation probe passes on the whole pipeline on both assets). The mechanism does produce a real, non-trivial in-sample effect on BTC, dramatically so on spot at kappa=0.20-0.30 (inner-validation Sharpe +1.25 to +2.33 above `kelly_regime_v4`, risk-matched within the B5 threshold) -- but B3's own plateau requirement is not met (the BTC-spot effect escalates rather than holding flat across the grid, the "spike, not a plateau" failure mode named in this round's own pre-registration), B4's ETH sign-replication test fails on 5 of 6 (kappa, market) cells (the project's single most repeated failure mode), and the CONDITIONAL Step 4 fee-tier holdout check -- triggered because a looser, pre-registered per-market Sharpe-OR-drawdown screen was cleared -- fails decisively: no kappa beats `kelly_regime_v4` on both BTC and ETH at the 0.40% taker tier out-of-sample, with ETH holdout performance actively collapsing (down to -2.93 Sharpe on spot at kappa=0.20). The honest reading is that the term-structure slope carries a real, occasionally large, BTC-specific timing signal in this particular sample window that does not transfer to ETH and does not survive contact with the holdout at a realistic fee tier -- a construction that avoided R-141's mathematical degeneracy trap only to land in this project's much more common empirical trap instead.

**Holdout counter: +1** consultation (Step 4's 12 holdout cells: 4 `kelly_regime_v4` reference + 8 candidate, both markets that cleared the conditional screen, both assets, both flagged kappas, 0.40% fee tier). No bar at or after 2023-01-01 was read anywhere in Step 0/3 (Sections 1-5); the holdout was read only once, in Section 6, after the pre-registered conditional trigger fired.
