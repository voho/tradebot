# R-59 (conservative branch) — per-asset `target_vol` calibration (08-20)

Backlog **B-25**. Pre-registration (shared, both branches): `experiments/r59_shared.py`.
Implementation: `experiments/r59_conservative_calibrated_target.py`. Raw cells:
`reports/r59_conservative/*.csv`. Nothing under `src/tradebot/strategies/` is
touched — `kelly_regime_v4` is used exactly as shipped, with `target_vol`
passed as the constructor argument it already accepts; `max_leverage=2.0` is
never touched, for any asset.

## 1. The question, one sentence

R-57 found `kelly_regime_v4`'s one surviving property — a matched-exposure
drawdown advantage — inverts on 6 of 6 further instruments, and named a
hypothesis: `target_vol=0.55`/`max_leverage=2.0` are BTC-calibrated
constants, so on higher-volatility instruments the sizing term is
structurally smaller. This branch tests the most direct reading of that
fix: solve a per-asset `target_vol_i` so every instrument's own mean
notional matches what BTC's unmodified default already produces, and see
whether the drawdown property comes back.

## 2. Mechanism

For each of 8 assets (BTC, ETH, and R-57's six-asset panel), `target_vol_i`
is solved by proportional iteration (R-33/R-57's `solve_c` pattern, applied
to `target_vol` instead of a passive hold's constant exposure) so that
`kelly_regime_v4(target_vol=target_vol_i, max_leverage=2.0)`'s own mean
clipped notional (`experiments.matched_hold.mean_notional`) matches a single
reference: v4's **unmodified** (`target_vol=0.55`) mean notional on BTC over
CONTROL (2020-04-01 → 2022-12-31). Solved on PANEL_TRAIN/CONTROL only — the
same calendar window under two names — never refit on PANEL_TEST.

Reference (BTC, `target_vol=0.55`, unmodified): mean notional = **0.3803**.

| asset | target_vol_i | achieved notional | residual | note |
|---|---|---|---|---|
| BTC | 0.550 | 0.3803 | 0.00% | exact — the solver's starting guess already hits the reference, since it *is* the reference (self-consistency check passes) |
| ETH | 0.614 | 0.3770 | 0.87% | |
| BCH | 0.936 | 0.3730 | 1.93% | |
| LTC | 0.794 | 0.3755 | 1.26% | |
| ETC | 1.223 | 0.3753 | 1.31% | |
| DASH | 1.125 | 0.3786 | 0.44% | |
| LINK | 0.933 | 0.3746 | 1.50% | |
| XTZ | 1.215 | 0.3784 | 0.49% | |

All 8 solves land within the 2% tolerance (no cap binds). BTC's own solve
returns exactly 0.550, as expected by construction. The panel needs
`target_vol` **1.4×–2.2×** BTC's default to reach the same mean notional —
consistent with R-57's diagnosis that the panel's higher realized volatility
was structurally suppressing the sizing term.

## 3. Causality tamper probe

`experiments.r57_cross_asset_panel`'s tamper methodology, adapted to
construct `KellyRegimeV4(target_vol=target_vol_i)` directly (this branch's
candidate is never registry-default `kelly_regime_v4`, so `get_strategy`
could not be reused unmodified). Run on BTC and 3 panel assets (BCH, LTC,
ETC), each with its own calibrated `target_vol_i`:

**PASS on all 4** — decisions at and before the tamper cut are identical
under opposite post-cut price/volume tampers, for every `target_vol_i`
tested. No lookahead bug; results below are reported as normal.

## 4. D1 (primary) — PANEL_TRAIN, spot @0.10%, calibrated `target_vol_i`

| asset | target_vol_i | c (cand's mean notional) | cand max DD | matched hold max DD | Δ DD (pp, + = cand worse) | 95% paired interval |
|---|---|---|---|---|---|---|
| BCH | 0.936 | 0.37 | 63.7% | 58.8% | **+5.3** | [−14.2, +38.6] |
| LTC | 0.794 | 0.38 | 50.2% | 49.9% | **−0.3** | [−13.9, +27.0] |
| ETC | 1.223 | 0.38 | 62.0% | 48.8% | **+15.3** | [+0.0, +43.9] |
| DASH | 1.125 | 0.38 | 70.8% | 55.1% | **+16.1** | [−6.2, +41.8] |
| LINK | 0.933 | 0.37 | 62.5% | 47.7% | **+17.0** | [−8.9, +39.4] |
| XTZ | 1.215 | 0.38 | 65.9% | 56.0% | **+9.9** | [−0.6, +45.1] |

**0 of 6.** LTC is a near-tie (−0.3pp, interval spans zero) and no other
asset even ties — the other five all get *worse*, not better, once resized
to BTC's own exposure level. One interval (ETC) excludes zero, against the
candidate. **D1 verdict: FAILS** (exact binomial p = 1.0000). Compared with
R-57's frozen `target_vol=0.55` cells on the same window/panel (0/6, sign
inverted on every asset, magnitudes +5.2 to +33.8pp), calibrating
`target_vol` per asset **did not restore the property and barely moved most
of the numbers** — DASH, LINK, XTZ actually have marginally worse Δ DD than
R-57's uncalibrated run; only ETC (23.6→15.3), BCH (5.2→5.3 flat), and LTC
(33.8→−0.3) improved, and only LTC crossed zero.

## 5. D2 (falsification control) — CONTROL window, BTC and ETH, calibrated `target_vol`

| asset | target_vol_i | c | cand max DD | matched hold max DD | Δ DD (pp) | R-57 control | within 5pp tolerance? |
|---|---|---|---|---|---|---|---|
| BTC | 0.550 | 0.38 | 33.2% | 39.3% | **−5.6** | −5.6 | yes (identical — the solve returns the unmodified strategy) |
| ETH | 0.614 | 0.38 | 31.2% | 44.0% | **−12.7** | −11.5 | yes (−1.2pp, well inside tolerance) |

**D2 PASSES.** BTC is unchanged by construction (its calibration returns
`target_vol=0.550`, i.e. no-op). ETH's advantage is slightly *larger* in
magnitude than R-57's number (−12.7 vs −11.5pp), not a regression.

## 6. D3 (generalization, reported not gating) — PANEL_TEST, frozen `target_vol_i`

Same six assets, 2023-01-01 → 2026-08-20, spot @0.10%, no refit:

| asset | cand max DD | matched hold max DD | Δ DD (pp) | 95% interval |
|---|---|---|---|---|
| BCH | 57.0% | 45.0% | +12.9 | [−4.7, +48.6] |
| LTC | 83.2% | 38.5% | +46.3 | [+12.8, +59.8] |
| ETC | 77.3% | 53.8% | +23.9 | [+8.5, +48.1] |
| DASH | 68.9% | 37.2% | +35.1 | [−0.9, +43.6] |
| LINK | 59.1% | 46.6% | +12.4 | [−4.1, +38.7] |
| XTZ | 57.2% | 53.1% | +3.5 | [−2.8, +37.0] |

**0 of 6.** LTC, which was D1's near-tie, is the worst cell here (+46.3pp,
interval excludes zero against the candidate) — the D1 near-miss did not
generalize. Descriptive only, consistent with D1's failure, not a separate
gate.

## 7. D4 (0.40% fee falsification) — PANEL_TRAIN, spot @0.40%, frozen `target_vol_i`

Candidate beats `buy_and_hold`'s final balance in **3 of 6** (BCH, ETC,
DASH — final-balance wins, not drawdown wins). Threshold was ≥5/6.
**FAILS, as predicted before the run.**

## 8. Verdict

`experiments.r59_shared.promoted(k1=0, dd_advantage={"BTC": -5.6, "ETH": -12.7})`
applied mechanically: `k1 >= 5` is False, so **`promoted()` returns False**
regardless of D2 (which passes on its own). Per the pre-registration's
promotion bar (D1 ≥ 5/6 AND D2 passes on both BTC and ETH):

**NEGATIVE.**

Stated precisely, because the failure mode is worth naming: this is *not*
the "fix breaks BTC/ETH" failure the pre-registration flagged as the mirror
risk (D2 passes cleanly, ETH's own control number improved slightly). It is
the plainer failure — matching every asset's *mean sizing scale* to BTC's
does not restore the *sign* of the drawdown property on 5 of 6 panel assets,
and the one asset where the sign came back (LTC, D1 −0.3pp) is also the one
asset that got materially worse on the held-out 2023-2026 window (D3
+46.3pp). Per-asset target_vol recalibration moves each panel asset's mean
notional to BTC's level, but R-57's own diagnosis already named the more
likely mechanism: on the panel, a rebalanced constant-exposure hold behaves
like a buy-the-dip rule, and that mean-reversion benefit — not the sizing
constant — is most of what the matched hold is winning by. Raising
`target_vol` raises the candidate's average exposure but does not change
that its vote-gated trend rule stands aside after drops the matched hold is
quietly buying, so equalizing *average* notional does not equalize the
comparison that matters. This closes B-25's conservative branch: extending
the strategy family's SIZE-axis record to 0-for-19 (pending the novel
branch's own result for the round total).

## 9. Accounting

- **Backtest configurations evaluated: 80** — 20 in the calibration solve
  (1 reference call + 8 assets × up to 3 solver iterations, all converged
  well inside `max_iter=8`), 60 in D1–D4 (6 panel assets × 3 arms × 3
  windows [PANEL_TRAIN@0.10%, PANEL_TEST@0.10%, PANEL_TRAIN@0.40%] + 2
  control assets × 3 arms × 1 window [CONTROL@0.10%] = 54+6=60). Causality
  probe calls `prepare()`/`on_bar()` directly, not `run_period`, so (per
  R-57's own convention) it is not counted as a backtest configuration.
- **Holdout consultations: +0.** BTC and ETH frames are truncated at
  2022-12-31 immediately after loading, before any other line of code
  touches them (`load_control_assets()`); no 2023+ bar of either is ever
  read. Panel-asset reads (train or test) cost +0 per the pre-registration
  (new-instrument evidence, not the reserved BTC/ETH holdout).
- **Decision rules moved: no.** D1–D4 and the promotion bar are exactly as
  committed in `experiments/r59_shared.py`; nothing was chosen after seeing
  a number.
- `pytest -q`: 461 passed (unchanged — this branch added no code under
  `src/`, only two new files under `experiments/`).

## 10. What this changes

The conservative branch's most direct reading of R-57's own hypothesis —
"scale `target_vol` to match BTC's own average exposure, per asset" — does
not restore the matched-exposure drawdown property. The BTC/ETH control
stays intact (D2 passes, confirming the calibration procedure itself is not
the problem — it correctly returns a near-no-op on the two assets where the
property already held), which localizes the failure precisely: it is not
that per-asset calibration breaks something that worked, it is that mean
notional was never the dimension the property lived on. R-57's own named
hypothesis in section 5 of its report — that the matched hold's advantage on
the panel is largely a mean-reversion / buy-the-dip effect the panel's
price dynamics reward and BTC/ETH's don't — is more consistent with these
numbers than the "wrong sizing constant" hypothesis this branch tested.
