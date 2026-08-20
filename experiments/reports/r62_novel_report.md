# R-62 (novel branch) — the latched vote alone, at a constant multiplier (08-20)

Pre-registration (shared, both branches): `experiments/r62_shared.py` (not
edited by this branch). Implementation: `experiments/r62_novel_vote_constant_exposure.py`.
Nothing under `src/tradebot/strategies/` is touched; nothing is registered.

## 1. The question, one sentence

Of `kelly_regime_v4`'s two multiplied factors — `frac` (the latched
20/40/80-day multi-anchor trend vote) and `scale` (the conditional
volatility target) — does `frac` **alone**, driving a constant full-notional
multiplier with the vol-scaling machinery deleted entirely
(`desired[i] = frac[i] * 1.0`), reproduce v4's matched-exposure drawdown
advantage on the six-asset panel that R-57 found inverts (0/6) off BTC/ETH?

## 2. Mechanism

`VoteConstantExposure` (`experiments/r62_novel_vote_constant_exposure.py`)
copies `KellyRegime.prepare`'s latched multi-anchor vote loop byte-for-byte
— three anchors (20/40/80 days), 1% band, hysteresis latching (bullish above
the anchor, bearish below, hold the previous verdict inside the band) —
producing `frac[i] ∈ {0, 1/3, 2/3, 1}`. `desired[i] = frac[i] * c_const`
with `c_const = 1.0` fixed, not swept. No volatility measurement, no EWM vol
span, no breakout-state hysteresis of any kind appears anywhere in the
class. The same 10% deadband/latch mechanics as v4 gate updates to the held
position, and orders are placed via `ctx.order_notional(t)`, identically to
v4.

## 3. Causality — PASS

Two-opposite-tampers probe (`experiments/r57_cross_asset_panel.py`'s
`cmd_causality` methodology, reused not reimplemented) on BCH and LTC:
3x-up / (1/3)x-down price tampers and 7x/(1/7)x volume tampers applied to
the last 5,000 of a 60,000-bar tail, decisions compared at six bars just
before the tamper cut. Orders (side, qty, target) identical under both
tampers for both assets — **PASS**.

```
BCH   decisions identical under opposite post-cut tampers: PASS
LTC   decisions identical under opposite post-cut tampers: PASS
```

`pytest tests/test_causality_strict.py tests/test_causality_real.py -q`:
**101 passed**, no errors (this experiment is unregistered, so it is not
itself exercised by that suite's parametrization — the run confirms nothing
elsewhere in the project was broken).

## 4. D1 — PRIMARY. FULL window (2020-04-01 → last bar), spot @0.10%

Candidate's own mean-notional-matched hold, per the frozen `cell()` harness.

| Asset | cand final | cand DD | matched hold final | matched hold DD | c\_mean | D1 (cand DD < matched DD) |
|---|---|---|---|---|---|---|
| BCH | $812 | 77.5% | $1,952 | 66.7% | 0.45 | No |
| LTC | $375 | 90.5% | $2,103 | 58.6% | 0.47 | No |
| ETC | $1,199 | 88.7% | $2,871 | 53.4% | 0.41 | No |
| DASH | $740 | 85.2% | $1,896 | 61.7% | 0.42 | No |
| LINK | $1,837 | 81.7% | $4,904 | 58.9% | 0.50 | No |
| XTZ | $302 | 82.2% | $1,006 | 65.5% | 0.40 | No |

**D1 = 0/6 → FAILS TO REPLICATE.** Every asset's paired-bootstrap drawdown
gap point estimate is positive (candidate worse), ranging +10.5pp (BCH) to
+38.2pp (ETC); none of the six bootstrap intervals excludes zero in the
candidate's favour.

## 5. D2 — context (mean notional, not a gate)

The candidate's own mean notional per asset, next to v4's own (from R-57's
`cells.csv`, FULL/spot/0.10%):

| Asset | novel c\_mean | v4 c\_mean (R-57) | ratio |
|---|---|---|---|
| BCH | 0.45 | 0.288 | 1.56x |
| LTC | 0.47 | 0.309 | 1.52x |
| ETC | 0.41 | 0.218 | 1.88x |
| DASH | 0.42 | 0.190 | 2.21x |
| LINK | 0.50 | 0.279 | 1.79x |
| XTZ | 0.40 | 0.198 | 2.02x |

**Named surprise:** the pre-registration predicted the novel arm's mean
notional would be *lower* than v4's ("`c_const=1.0` with no vol-boost is
smaller than v4's occasional up-to-2x breakout sizing"). It is the opposite
— 1.5x–2.2x *higher* on every panel asset. The reason is visible in the
mechanism: v4's conditional vol-target divides by these high-volatility
altcoins' *realized* volatility even in its "steady" (non-breakout) state,
which keeps its notional small (~0.19–0.31) most of the time; the vote
alone, with no volatility division at all, sits at fractions of {0, 1/3,
2/3, 1} whenever the trend is "in," which averages out higher. Deleting the
scale factor does not produce a smaller position — it removes the one thing
that was shrinking it.

## 6. D3 — BTC/ETH control (+0 holdout), CONTROL_WINDOW 2020-04-01→2022-12-31, spot @0.10%

| Asset | cand final | cand DD | matched hold final | matched hold DD | c\_mean | D1-style (cand DD < matched DD) |
|---|---|---|---|---|---|---|
| BTC | $3,594 | 40.0% | $1,947 | 51.0% | 0.51 | Yes |
| ETH | $6,236 | 57.5% | $4,778 | 60.2% | 0.57 | Yes |

**D3 = 2/2 → REPLICATES.** Both bootstrap intervals include zero
([-22.3,+19.6] BTC, [-21.0,+30.3] ETH — neither excludes it), so this is a
point-estimate replication, not a statistically sharp one, but the sign is
the fitted-asset direction on both.

## 7. D4 — fee context. FULL window, spot @0.40%

| Asset | cand final | hold final | beats hold? |
|---|---|---|---|
| BCH | $263 | $971 | No |
| LTC | $121 | $1,190 | No |
| ETC | $380 | $1,316 | No |
| DASH | $231 | $487 | No |
| LINK | $474 | $4,616 | No |
| XTZ | $87 | $132 | No |

**D4 = 0/6.**

## 8. BEAR22 (descriptive, R-57's D4 style), 2022-05-01→2022-11-30, spot @0.10%

| Asset | cand DD | matched hold DD | D1-style |
|---|---|---|---|
| BCH | 40.4% | 22.7% | No |
| LTC | 33.9% | 28.6% | No |
| ETC | 37.2% | 22.1% | No |
| DASH | 30.2% | 18.6% | No |
| LINK | 45.9% | 23.7% | No |
| XTZ | 33.1% | 19.3% | No |

0/6 here too — the same failure pattern holds inside the bear window
specifically, not just averaged over the full period.

## 9. Further-work bar

Pre-registered: D1 ≥ 5/6 **AND** D3 ≥ 1/2 **AND** D4 ≥ 4/6.
Measured: D1 = 0/6, D3 = 2/2, D4 = 0/6 → **NOT MET**. D1 alone (0/6, the
worst possible score) closes this line regardless of D3/D4.

## 10. Configurations evaluated

`experiments.r57_cross_asset_panel.CONFIG_COUNT + experiments.r62_shared.extra_config_count()`
= **60**, this branch only (20 cells x 2 `measure()` calls each [hold,
matched-hold] = 40, plus 20 candidate runs counted by
`extra_config_count()` = 60). The round's total is this number plus
whatever the conservative branch reports, summed by the operator per
ROUTINE.md's parallelism rule.

## 11. A bug found while building the runner (not silently patched)

`experiments/r62_shared.py`'s `d1_from_rows` and `d4_from_rows` filter rows
only on `(arm, market, fee)` — there is no `window` parameter. BEAR22 and
the D3 control window both also run at `market="spot", fee=0.001`, the
identical `(market, fee)` pair as the D1 FULL-window slice. A first draft of
this runner accumulated all cells (FULL@0.10%, FULL@0.40%, BEAR22, D3
control) into one `rows` list and then called
`d1_from_rows(rows, "novel", "spot", 0.001)` at the end, exactly as the task
description's own phrasing suggests — this **silently pools BEAR22 and D3
control rows into the D1 count**. Concretely, that first draft printed
"D1 = 2/6" where the "2" were actually the two D3 control rows (BTC, ETH,
both counted as cand_dd < mh_dd), not any panel asset — the true FULL-window
D1 count is 0/6, confirmed directly from the six FULL/spot/0.10% rows in
the saved CSV. Both runs are logged in this branch's working notes; the
final registered numbers above are from the corrected runner, which
pre-filters each decision rule's input to its own window's rows before
calling the frozen `d1_from_rows`/`d4_from_rows` (the functions themselves
are used exactly as written, not reimplemented — only the input slice
changes). This does not change the round's headline verdict (D1 was already
`FAILS TO REPLICATE` under the wrong 2/6, and remains so, more starkly, at
the correct 0/6), but the exact number matters and the shared file should
not be trusted to disambiguate windows sharing a market/fee pair without a
caller-side filter — worth a note for the operator if `r62_shared.py` is
reused by a future round, or if the conservative branch's own `rows`
accumulation hit the same trap. Not fixed in `r62_shared.py` per this
round's instructions (report, don't silently patch); fixed at the call site
in this branch's own runner only.

## 12. Interpretation

**Negative.** The vote alone, driving a constant full-notional multiplier
with no volatility scaling at all, does not reproduce v4's matched-exposure
drawdown advantage on the six-asset panel: 0/6, the worst possible D1 score,
with every one of the six point estimates on the wrong side and by a wide
margin (+10.5pp to +43.3pp, growing worse at the 0.40% fee tier and
persisting inside the BEAR22 sub-window specifically). D3 confirms the
property is still present on the two fitted assets at the FULL D1
methodology's own resolution (2/2, both point estimates in the candidate's
favour, though neither excludes zero), so this arm's failure — like R-59's
and R-60's before it — is specific to the panel, not universal. Combined
with the conservative branch (frac forced to 1.0, v4's scale kept) also
reporting on the same D1 criterion, this is the third and fourth independent
SIZE-axis line of evidence (after R-59's magnitude sweep and R-60's timing
sweep) that the panel's matched-exposure property is not something either
half of v4's `frac x scale` product carries on its own — deleting either
factor, in either direction, removes the property rather than isolating it,
which argues against a clean "one ingredient does it" explanation and for
B-27's alternative reading: whatever produces the property on BTC/ETH is
either the specific *combination*, or — the reading R-59/R-60/R-61 have each
converged back to independently — a property of BTC/ETH's own price paths
that six much-more-volatile, thinner instruments simply do not share,
regardless of which piece of the strategy is doing the deciding. The D2
surprise (this arm's mean notional running *higher* than v4's, not lower as
predicted) is a useful correction to the pre-registration's own intuition
about the mechanism, even though it did not change the direction of the
result: v4's conditional vol-target was doing real, continuous shrinking
work on these high-volatility altcoins even in its "steady" state, not only
in its rare breakout state, so removing it does not produce a smaller,
more-conservative position — it produces a bigger one that still draws down
harder than a passive hold at its own realized exposure.
