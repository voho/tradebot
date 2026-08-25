# R-128 (NOVEL branch) -- an adaptive, state-dependent EV band for `hedge_experts` (08-25)

Unregistered candidate. Code: `experiments/r128_novel_adaptive_band.py`. Not `@register`ed, not auto-discovered, nothing committed by this session. `src/tradebot/strategies/hedge_experts.py` is never edited -- `HedgeExpertsAdaptiveBand` subclasses `HedgeExperts` and reuses `HedgeExperts._experts()` verbatim. Full derivation, non-duplication argument, named failure modes, and the pre-registered decision rule live in `experiments/r128_shared.py`'s module docstring; only summarized here.

**Read this report's Section 5 before the verdict in Section 4.** Two implementation issues, found on review, materially undercut what the numbers below can be taken to mean. They are disclosed in full rather than silently fixed or re-run, per instruction.

## 1. Mechanism recap

`hedge_experts` blends ten causal technical experts with discounted multiplicative weights (Hedge) into a raw signal `x` in `[-1, 1]`, then only re-targets the traded position toward `x` when `abs(x - pos) > hysteresis` for a FIXED `hysteresis = 0.05`. This branch replaces that fixed threshold with the same fee/vol/horizon no-trade-band algebra `kelly_regime_ev.py` already shipped (Constantinides 1986; Davis & Norman 1990):

```
band = 2 * fee / (H_t * sigma**2)
```

but, unlike the CONSERVATIVE branch (one frozen constant `H`), `H_t` here is estimated ONLINE, every bar:

1. A causal EWM lag-1 autocorrelation of the blend signal `x` (`x.ewm(span=S).corr(x.shift(1))`), `S` = `ac_lookback_days * BARS_PER_DAY` trading days (primary: 20 days; swept over `{10, 20, 40, 80}` in B3). Causal by construction: row `i` only uses `x[0..i]` and `x.shift(1)[0..i] = x[-1..i-1]`, both already known at bar `i`'s close.
2. `rho` clipped to `[0.02, 0.98]`, converted to a horizon via `H_t = -1 / (BARS_PER_DAY * ln(rho))`, then `H_t` clipped to `[0.25, 10]` days.
3. `H_t` replaces the fixed `H` in the same band formula, using `ctx.market.fee_rate` and `sigma` = `hedge_experts`'s own already-computed `sig1` (EWM realized vol), annualized like `kelly_regime_ev`'s `_ev_vol`.

`x` is computed in `prepare()` with no hysteresis at all. The band-gated position update lives in `on_bar`, using a live `ctx.position`-derived current fraction (`ctx.position * ctx.close / ctx.equity`) and `ctx.order_notional(desired)`, mirroring `kelly_regime_ev.on_bar`'s body -- **inherited architecture, shared with the CONSERVATIVE branch, not independently invented here** (including the "always allow a full exit to flat" clause). `warmup = 2500`, identical to `HedgeExperts`, chosen so every B1 comparison against the registered baseline gets the same warmup budget (confirmed necessary by a development-time probe: replaying `hedge_experts`'s own target array through the b1_signal harness using its full 4-year history, instead of the 2500-bar budget the baseline itself gets, reproduced neither the baseline's Sharpe nor its final balance -- off by 0.72 Sharpe purely from the Hedge weight vector having converged longer).

## 2. Results table

### 2.1 Causal-truncation self-test

**PASS** -- full-period final balance `14301.933097397301` vs truncated-frame final balance `14301.933097397301` (BTC spot, 2017-01-01..2020-12-31), bit-identical. A second probe compared the raw `prepare()` output columns (`x`, `_ab_H`) between a frame cut at 2020-06-30 and the same frame sliced from the full series -- also **PASS**, `np.allclose` on both columns.

### 2.2 B1 -- BTC signal, spot + futures, full period + inner-validation

| market | window | sharpe_cand | sharpe_base | d_sharpe | boot CI | dd_cand% | dd_base% | trades_cand | trades_base | final_cand | final_base | clears? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spot | full | +1.2058 | +1.0548 | +0.1510 | [-0.532, +1.229] | 47.7 | 59.3 | 8 | 1444 | 14547.2 | 9950.4 | True |
| spot | val | -0.1572 | -0.7114 | +0.5542 | [-0.157, +0.859] | 32.9 | 59.2 | 3 | 451 | 871.7 | 594.0 | True |
| futures | full | (n/a*) | (n/a*) | +0.0962 | [-5.151, +12.066] | 51.1 | 99.9 | 49 | 3002 | 16611.4 | 414.4 | True |
| futures | val | (n/a*) | (n/a*) | +1.2293 | [+2.321, +8.230] | 24.1 | 99.8 | 17 | 662 | 1212.4 | 5.8 | True |

`*` `sharpe_cand`/`sharpe_base` for futures were not separately logged by the run's print statements (only `d_sharpe` and the rest were); the full `b1_signal` return dict for every cell, including the omitted fields, is reproducible by re-running `experiments/r128_novel_adaptive_band.py` (deterministic, no randomness besides the fixed-seed paired bootstrap).

**B1 "clears" definition used (task's own elaboration of the pre-registered rule):** `d_sharpe > 0.2` OR `dd_cand < dd_base`. **All 4 cells clear**, in every case via the drawdown leg (trade counts collapse by 1-2 orders of magnitude in every cell -- see Section 5 for why this is not safely read as timing skill).

### 2.3 B3 -- autocorrelation-lookback plateau (FUTURES, inner-validation)

| lookback (days) | d_sharpe | sign |
|---|---|---|
| 10 | +1.2293 | +1 |
| 20 (primary) | +1.2293 | +1 |
| 40 | +1.2293 | +1 |
| 80 | +1.2293 | +1 |

**B3 PASS (>=3/4 same-signed):** True, 4/4 -- **but see Section 5: these four numbers are not independent evidence of a plateau.** They are byte-identical because `H_t` is pinned at its floor (`H_MIN_DAYS = 0.25`) in all four configurations (diagnostic below), so the lookback parameter has no effect on the executed strategy at all.

**H_t diagnostic (primary config, inner-train, not gating):**

| stat | value |
|---|---|
| median | 0.250 days |
| p5 | 0.250 days |
| p95 | 0.250 days |
| fraction of inner-train bars at the floor (0.25d) | 1.000 |
| fraction of inner-train bars at the ceiling (10d) | 0.000 |

`H_t` sits at its floor on 100% of inner-train bars. The "adaptive" mechanism, as implemented, does not adapt -- see Section 5.1 for the root cause.

### 2.4 B4 -- ETH falsification (spot only, inner-validation, primary config)

ETH spot `d_sharpe = +0.0327`. BTC spot inner-validation `d_sharpe` sign = `+1` (from `spot-val` above), ETH spot `d_sharpe` sign = `+1`. **B4 PASS (sign replicates):** True -- though the ETH effect size (+0.03) is roughly 17x smaller than BTC spot-val's (+0.55), a much weaker replication than the sign-match alone conveys.

### 2.5 B5 -- fee-tier survival (0.40% taker), primary config

| market | window | d_sharpe @0.10%/0.05% | d_sharpe @0.40% | sign flip? |
|---|---|---|---|---|
| spot | full | +0.1510 | +0.3627 | False |
| spot | val | +0.5542 | +2.3632 | False |
| futures | full | +0.0962 | +0.4762 | False |
| futures | val | +1.2293 | +1.6221 | False |

**B5 PASS (no sign flip, any cell):** True

## 3. Configurations evaluated

1 causal-truncation probe (plus 1 supplementary column-level truncation probe, not separately counted) + 4 B1 cells + 4 B3 sweep points + 1 B4 cell + 4 B5 cells = **14 total**. No selection occurred among them -- every cell run is reported, none filtered by outcome.

## 4. Decision-rule verdict (mechanical, as pre-registered)

causal probe=True  B1=True (4/4 clear)  B2=diagnostic-only  B3=True (4/4 same-signed)  B4=True (sign replicates)  B5=True (no flip)

**VERDICT AS COMPUTED: PROMOTE-candidate**

**This mechanical verdict should not be acted on.** Per instruction, the pre-registered rule is applied and reported exactly as written, without adjusting a threshold after seeing the numbers above. But two implementation issues (Section 5), found on review after the numbers were computed, mean the candidate that was actually measured is neither (a) a genuine test of the pre-registered adaptive-horizon mechanism, nor (b) free of a large uncontrolled confound on every leveraged-market cell. The honest reading is **inconclusive pending a corrected re-run**, not a clean promotion.

## 5. Discussion

### 5.1 The core mechanism this round was supposed to test never actually ran

`H_t` is pinned at its floor (`H_MIN_DAYS = 0.25`) on 100% of inner-train bars in the primary config, and identically across all four B3 lookback sweep points (Section 2.3). This is not sampling noise -- it is a direct, checkable consequence of an inconsistency between two of this file's own disclosed clip bounds. Given `BARS_PER_DAY = 288` and `RHO_MAX = 0.98`, the *largest* `H_t` the formula `H_t = -1/(BARS_PER_DAY * ln(rho))` can produce, at the maximum allowed `rho`, is `-1/(288 * ln(0.98)) ≈ 0.172` days -- which is *already below* the `H_MIN_DAYS = 0.25` floor. In other words, for any `rho` in the permitted `[0.02, 0.98]` range, the unclipped `H_t` is mathematically guaranteed to fall below the floor and get clipped back up to exactly `0.25`, regardless of what the blend signal's actual persistence is. `RHO_MAX` would need to exceed `exp(-1/(288*0.25)) ≈ 0.9862` for the floor to ever *not* bind. This was not a fitted choice (if anything it happened by accident, in the conservative direction of leaving less headroom, not more) -- but it is a real calibration bug: the "adaptive, state-dependent horizon" this round pre-registered as its whole reason for existing collapsed, in every configuration tested, to a fixed 0.25-day horizon (equivalently: the widest allowed band `kelly_regime_ev`'s clip permits, since a small `H` maximizes `band = 2*fee/(H*sigma^2)`). Every number in Sections 2.2-2.5 measures **"hedge_experts + a fixed near-maximal, non-adaptive band," not the state-dependent mechanism `r128_shared.py` pre-registered.** B3's "plateau" (Section 2.3) is not evidence of robustness across lookback windows -- it is four identical runs of the same degenerate config, because the swept parameter (`ac_lookback_days`) has no effect once `rho`'s clip already forecloses the floor.

### 5.2 A second, independent confound on every leveraged-market cell (flagged on review, shared with the CONSERVATIVE branch's inherited architecture)

`on_bar` compares `desired_x` (the raw Hedge blend `x` in `[-1, 1]` -- the exact quantity `hedge_experts`'s own `on_bar` always fed straight to `ctx.order_target`, i.e. already expressed as a fraction of *this market's max leveraged notional*) against `current = ctx.position * ctx.close / ctx.equity`, which is an **equity-notional fraction** that ranges up to `±leverage` (±5 on 5x futures), not `±1`. These are different units on any market with leverage != 1. The two only coincide on spot (leverage = 1.0). The subsequent `ctx.order_notional(desired_x)` call then targets `|notional| = desired_x * equity` -- since `desired_x` never exceeds 1 in absolute value, this **caps the strategy's futures exposure at 1x equity notional**, versus `hedge_experts`'s original design intent of up to 5x. This is not a tuning choice; it is an artifact of transplanting `kelly_regime_ev.on_bar`'s body (built for a strategy whose `target` was already an equity-fraction) onto `hedge_experts` (whose `x` was designed as a max-notional fraction) -- an architecture decision shared with, and inherited from, the CONSERVATIVE branch's own transplant, not something this branch introduced independently. It almost certainly explains most of the futures cells' large apparent improvement: baseline `hedge_experts` on futures is levered up to 5x and gets wiped out (max drawdown 99.8-99.9%, final balance $5.80-$414 from $1,000), while the candidate, effectively capped near 1x, survives by trading far smaller size -- a de-leveraging artifact, not evidence the adaptive band finds better trade timing. The SPOT cells (unaffected by this confound, since leverage = 1 there) still clear via the drawdown leg, but with much smaller trade counts (3-8 trades vs. baseline's 451-1444) than a "better-timed, still-active" strategy would suggest -- consistent with Section 5.1's finding that the band is pinned near its widest setting throughout, so most of the SPOT improvement is also a "trade almost never" story rather than a timing edge.

### 5.3 Reading the four named risks from `r128_shared.py` against what actually happened

1. *"the derived band could land close to 0.05 and show a null result"* -- did not happen, but not for the intended reason: the band did not land near a sensible number at all, it landed pinned at its widest permitted setting for a math reason unrelated to the data (Section 5.1).
2. *"a seventh construction inverts sign on ETH"* -- did not happen on the sign (B4 passes), but the ETH effect is ~17x smaller than BTC's own inner-validation effect, which is weak, not sharp, replication, and cannot be trusted further given Section 5.1's finding that BTC's own effect is itself a measurement of the wrong (degenerate) mechanism.
3. *"a band-width algebra built for one stationary Kelly bet may be structurally mismatched to hedge_experts's multi-timescale blend"* -- genuinely untested by this run: since `H_t` never varied, this round supplies no evidence either way about whether state-dependence per se helps or hurts a multi-timescale blend.
4. *"a band wide enough to cut turnover materially could reproduce the LAG failure measured in every regime-timing mechanism on kelly_regime_v4"* -- this is the risk that most plausibly DID materialize: trade counts collapsed 25-150x versus baseline in every cell, i.e. the band did exactly what risk 4 warned it might, though whether that specific behavior constitutes "the LAG failure" versus "a leverage-cap+degenerate-band artifact" cannot be cleanly separated given Sections 5.1-5.2.

### 5.4 Net read

As measured, this branch does not supply evidence for or against the state-dependent-horizon hypothesis it was built to test, because the mechanism that was supposed to vary (`H_t`) never varied under the disclosed clip bounds, and a second, architecture-level confound (Section 5.2, shared with the conservative branch) inflates every leveraged-market cell independent of any timing mechanism. The mechanical decision rule reads PROMOTE-candidate; the honest recommendation is that this result is **not informative as run** and a corrected re-run (wider `RHO_MAX`, e.g. ~0.995-0.999, so `H_t` can actually range across its stated `[0.25, 10]` band; and a resolved unit convention between `desired_x` and `current` before calling `order_notional`) is needed before either the mechanism or the decision rule's outcome can be trusted. No fix or re-run was made in this session per instruction -- this is a disclosure, not a correction.

## 6. Causality / holdout accounting

Max timestamp read anywhere in this branch: `2022-12-31 23:55:00+00:00` (< `OOS_START` 2023-01-01: True). No bar at or after 2023-01-01 was read by this file. `pytest tests/test_causality_strict.py -q`: **51 passed**, 0 failed, in 28.86s.
