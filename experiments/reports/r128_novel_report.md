# R-128 (NOVEL branch) -- an adaptive, state-dependent EV band for `hedge_experts` (08-25)

Unregistered candidate. Code: `experiments/r128_novel_adaptive_band.py`. Not `@register`ed, not auto-discovered, nothing committed to the registry. `src/tradebot/strategies/hedge_experts.py` is never edited -- `HedgeExpertsAdaptiveBand` subclasses `HedgeExperts` and reuses `HedgeExperts._experts()` verbatim. Full derivation, non-duplication argument, named failure modes, and the pre-registered decision rule live in `experiments/r128_shared.py`'s module docstring; only summarized here.

**This report supersedes the branch's own first-run report.** The original agent's run (preserved in git history at commit `3b6d924` and earlier) found and disclosed two implementation issues before any verdict was trusted: (1) `RHO_MAX=0.98` and `H_MIN_DAYS=0.25` were mutually inconsistent, mathematically pinning the "adaptive" horizon to its floor on every bar, so the mechanism this round exists to test never actually varied; (2) `on_bar` compared the blend `x` (already in fraction-of-max-leverage units) against a notional-multiple `current`, then traded via `ctx.order_notional` -- a unit mismatch, invisible on spot, that capped futures exposure near 1x equity regardless of the real 5x cap (an R-33-style exposure artifact, not a timing effect; shared with the conservative branch's own transplant). The operator fixed both (see `r128_novel_adaptive_band.py`'s own POST-HOC CORRECTION docstring addendum for the exact diffs and algebra) and re-ran the full battery. The numbers and verdict below are from the CORRECTED run. Nothing is deleted: the first run's numbers stayed in git history rather than being overwritten silently.

## 1. Mechanism recap

`hedge_experts` blends ten causal technical experts with discounted multiplicative weights (Hedge) into a raw signal `x` in `[-1, 1]`, then only re-targets the traded position toward `x` when `abs(x - pos) > hysteresis` for a FIXED `hysteresis = 0.05`. This branch replaces that fixed threshold with the same fee/vol/horizon no-trade-band algebra `kelly_regime_ev.py` already shipped (Constantinides 1986; Davis & Norman 1990), `band = 2*fee/(H_t*leverage*sigma**2)` (the `*leverage` in the denominator is the post-hoc correction; see below), but, unlike the CONSERVATIVE branch (one frozen constant `H`), `H_t` here is estimated ONLINE, every bar, from a causal EWM lag-1 autocorrelation of the blend `x` itself, converted to an implied AR(1) horizon (`H_t = -1/(BARS_PER_DAY * ln(rho))`, `rho` clipped to `[0.02, 0.999]` post-fix, `H_t` clipped to `[0.25, 10]` days).

**Post-hoc correction applied (both issues, see the code file's own docstring addendum for full derivations):**
1. `RHO_MAX` raised `0.98 -> 0.999` so `H_t` can actually range across its stated band instead of being mathematically forced to its floor on every bar.
2. `current` is now `ctx.position * ctx.close / (equity * leverage)` (fraction of max leverage, matching `x`'s own units) rather than `ctx.position * ctx.close / equity` (a notional-multiple); `_band` gained one extra `/leverage` factor; orders are placed via `ctx.order_target(desired_x)`, matching `hedge_experts`'s own native convention, not `ctx.order_notional`.

Both fixes leave SPOT numbers unaffected only for fix 2 (leverage=1 there); fix 1 changes every cell's `H_t` path, spot included, so the entire battery was re-run rather than only the futures cells.

## 2. Results table (CORRECTED)

### 2.1 Causal-truncation self-test

**PASS** -- full-period final balance `9546.890287484352` vs truncated-frame final balance `9546.890287484352` (BTC spot, 2017-01-01..2020-12-31), bit-identical. The supplementary `prepare()`-column probe (`x`, `_ab_H` between a frame cut at 2020-06-30 and the same frame sliced from the full series) also **PASS**.

### 2.2 B1 -- BTC signal, spot + futures, full period + inner-validation

| market | window | d_sharpe | boot CI | significant | dd_cand% | dd_base% | trades_cand | trades_base | final_cand | final_base |
|---|---|---|---|---|---|---|---|---|---|---|
| spot | full | -0.0127 | [-0.703, +0.576] | False | 52.6 | 59.3 | 82 | 1444 | 9341.1 | 9950.4 |
| spot | val | +0.5706 | [-0.008, +0.704] | False | 38.0 | 59.2 | 28 | 451 | 847.6 | 594.0 |
| futures | full | -0.1460 | [-4.149, -1.174] | **True** | 100.0 | 99.9 | 870 | 3002 | 29.5 | 414.4 |
| futures | val | -0.0310 | [-0.800, +0.480] | False | 99.8 | 99.8 | 351 | 662 | 4.9 | 5.8 |

**B1 "clears" definition (pre-registered):** `d_sharpe > 0.2` OR `dd_cand < dd_base`. Spot-full and spot-val clear (drawdown leg / d_sharpe respectively); futures-full and futures-val do **not** (`d_sharpe` negative, `dd_cand >= dd_base` on both). **B1 FAILS (2/4 cells clear, not all 4).** Note futures-full's `d_sharpe` is now the single most statistically decisive number in this round's whole battery -- its bootstrap CI `[-4.149, -1.174]` excludes zero entirely, on the losing side.

### 2.3 B3 -- autocorrelation-lookback plateau (FUTURES, inner-validation)

| lookback (days) | d_sharpe | sign |
|---|---|---|
| 10 | -0.1822 | -1 |
| 20 (primary) | -0.0310 | -1 |
| 40 | -0.0727 | -1 |
| 80 | -0.1941 | -1 |

**B3 "passes" the mechanical same-signed-majority test (4/4 negative)** -- but read plainly, this is not the plateau the round hoped for: it is a consistent, no-longer-degenerate NEGATIVE effect across the whole lookback range, now that `H_t` genuinely varies (diagnostic below) rather than sitting pinned at one value.

**H_t diagnostic (primary config, inner-train, not gating) -- now genuinely adaptive:**

| stat | value |
|---|---|
| median | 1.766 days |
| p5 | 0.802 days |
| p95 | 3.470 days |
| fraction of inner-train bars at the floor (0.25d) | 0.003 |
| fraction of inner-train bars at the ceiling (10d) | 0.000 |

Confirms the `RHO_MAX` fix worked: `H_t` now spans roughly 0.8-3.5 days across the middle 90% of inner-train bars (vs. 100% pinned at the 0.25d floor before the fix). The mechanism this round was built to test now actually ran -- and the result is negative on futures, not merely untested.

### 2.4 B4 -- ETH falsification (spot only, inner-validation, primary config)

ETH spot `d_sharpe = -0.1241`. BTC spot inner-validation `d_sharpe` sign = `+1` (from `spot-val` above, +0.5706), ETH spot `d_sharpe` sign = `-1`. **Signs do not match. B4 FAILS.**

This is a genuine sign inversion, not a marginal same-sign-different-magnitude result: this construction is the **seventh** distinct mechanism/object on this project to pass some part of a BTC gate and invert sign on ETH, joining R-109, R-113, R-115-conservative, R-125-conservative, R-126 (both branches) and this round's own conservative branch's directional pattern (see R-128's ledger entry discussion for the full count and what R-127 already found about this recurring signature).

### 2.5 B5 -- fee-tier survival (0.40% taker), primary config

| market | window | d_sharpe @0.05%/0.10% | d_sharpe @0.40% | sign flip? |
|---|---|---|---|---|
| spot | full | -0.0127 | +0.7944 | **True** |
| spot | val | +0.5706 | +2.4209 | False |
| futures | full | -0.1460 | +0.0281 | **True** |
| futures | val | -0.0310 | +0.7728 | **True** |

**B5 FAILS** -- three of four cells flip sign at the higher fee tier (the direction of the flip is NEGATIVE-to-positive in every case, i.e. a wider effective band at higher fees happens to avoid more of the churn that was hurting the candidate at the normal tier -- itself a data point that this candidate's problem is turnover-related, not a clean per-trade edge, consistent with Section 3 below).

## 3. Configurations evaluated

1 causal-truncation probe (+1 supplementary column-level probe, not separately counted) + 4 B1 cells + 4 B3 sweep points + 1 B4 cell + 4 B5 cells = **14 total**, identical count to the first (uncorrected) run -- the fixes changed the code, not the battery design. No selection occurred among them; every cell is reported, none filtered by outcome.

## 4. Decision-rule verdict (CORRECTED)

causal probe=True  B1=**False**  B2=diagnostic-only  B3=True (4/4 same-signed, negative)  B4=**False**  B5=**False**

**VERDICT: NEGATIVE.**

(Pre-registered rule from `r128_shared.py`, unaltered after seeing any number: PROMOTE-candidate only if the causal-truncation probe AND B1 AND B3 AND B4 AND B5 all pass. Three of five gates fail. This reverses the first run's mechanical "PROMOTE-candidate" reading, which that report's own Section 4 had already flagged should not be acted on given the two known implementation issues -- the corrected run confirms that caution was warranted.)

## 5. Discussion

**The adaptive mechanism this round was built to test now actually ran, and genuinely does not help.** Section 2.3's `H_t` diagnostic confirms real variation (median 1.8 days, IQR-like spread 0.8-3.5 days) rather than the degenerate floor-pinning of the first run -- so this is no longer an untested hypothesis. Reading the four risks `r128_shared.py` named before any code ran:

1. *"the derived band could land close to 0.05 and show a null result"* -- did not happen; the effect is real and, on futures, decisively negative (futures-full's bootstrap CI excludes zero entirely on the losing side).
2. *"a seventh construction inverts sign on ETH"* -- **this is what happened.** B4 fails cleanly, joining the pattern R-127 spent a full round diagnosing (the calendar window is a good BTC/ETH regime match, not a mismatch; brief ETH-idiosyncratic events explain part but not all of several prior inversions). This round adds a seventh data point to that list rather than resolving it.
3. *"a band-width algebra built for one stationary Kelly bet may be structurally mismatched to hedge_experts's multi-timescale blend"* -- the most likely single explanation for the futures result. Even with a genuinely adaptive `H_t`, the underlying assumption (one dominant persistence timescale drives the whole blend) may not fit a ten-expert construction spanning 1h-1w horizons simultaneously; an autocorrelation statistic computed on the blended output conflates all of those timescales into one number, which may not usefully describe any of them.
4. *"a band wide enough to cut turnover materially could reproduce the LAG failure"* -- turnover did collapse materially (trades_cand vs trades_base: 82 vs 1444 spot-full, 870 vs 3002 futures-full) and the result is negative on the markets where that collapse is largest relative to what was needed, consistent with this risk though not cleanly separable from risk 3.

**Net read:** once measured correctly, this branch is a clean, informative NEGATIVE -- not merely "inconclusive as run" (the first report's honest caveat), but a real result: adaptive, signal-persistence-derived rebalance timing does not rescue `hedge_experts` on leveraged markets, and inverts sign on ETH the same way six prior, structurally unrelated constructions have. The corrected SPOT-val cell (+0.57, matching the conservative branch's own spot-val result closely) is the only genuinely positive, unconfounded number in this branch's whole battery, and it alone cannot clear the round's own "both markets" bar.

## 6. Causality / holdout accounting

Max timestamp read anywhere in this branch: `2022-12-31 23:55:00+00:00` (< `OOS_START` 2023-01-01: True). No bar at or after 2023-01-01 was read by this file. `pytest tests/test_causality_strict.py -q`: **51 passed**, 0 failed (confirmed again after the post-hoc fix, since `on_bar`/`_band` were the files touched).
