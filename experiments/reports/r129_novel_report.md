# R-129 (NOVEL branch) -- per-timescale-bucket EV rebalance bands for `hedge_experts` (08-25)
Unregistered candidate. Code: `experiments/r129_novel_bucket_band.py`. Not `@register`ed, not auto-discovered, nothing committed by this session. `src/tradebot/strategies/hedge_experts.py` is never edited -- `NovelBucketBand` subclasses `HedgeExperts` and reuses `HedgeExperts._experts()` verbatim. Full derivation, literature citation, non-duplication argument, named failure modes, and the pre-registered decision rule live in `experiments/r129_shared.py`'s module docstring (frozen, shared with the parallel CONSERVATIVE branch); only summarized here.
## 1. Mechanism recap

`hedge_experts` blends ten causal technical experts with discounted multiplicative weights (Hedge) into a raw signal `x` in [-1, 1], then only re-targets toward `x` when `abs(x - pos) > hysteresis` for a FIXED `hysteresis = 0.05`. R-128 replaced that one threshold with one EV-derived band on the whole blended `x` and found it NEGATIVE, naming a per-timescale construction as the untested alternative (Ekren, Liu & Muhle-Karbe 2018 on multivariate no-trade regions). This branch groups the ten experts into three timescale buckets (`r129_shared.EXPERT_BUCKET`: FAST = {0,1,4,5,6}, SLOW = {2,3,7}, STATIC = {8,9}), computes each bucket's own Hedge-weighted sub-blend every bar, and applies ONE EV band per bucket (three total, never one on the sum), each derived from that bucket's own STRUCTURAL horizon (median of its members' native lookback/decay horizons, frozen upstream, never fit to a return):

```
band_bucket = clip(2*fee / (H_bucket_years * sigma_market**2 * leverage),
                    MIN_BAND, MAX_BAND)
held[bucket] <- x_bucket   only if |x_bucket - held[bucket]| > band_bucket
x = sum(held.values())
if |x - last_target| > 1e-9: ctx.order_target(x)
```

`fee`/`leverage` read live via `ctx.market`. `MIN_BAND=0.02`, `MAX_BAND=1.0` are `kelly_regime_ev`'s/R-128's own literal defaults, reused unchanged. The expert construction and Hedge weight update (`HedgeExperts._experts`, the weight-update loop) are byte-identical to the registered strategy -- only the re-target decision changes, and it changes at bucket granularity, not per-expert (that is the parallel CONSERVATIVE branch) and not on the single already-blended output (that was R-128).

**Per-bucket instance state and cold start.** `self._held` (dict), `self._last_target` (float), and `self._retarget_count` (dict, diagnostic) are tracked on the strategy instance -- there is no broker-side equivalent of a per-bucket position to read back. `self._held` is initialized once, on the first `on_bar` call that sees a finite positive `sigma_market` (normally `i = warmup = 2500`), to that bar's own live (unbanded) sub-blend values -- a disclosed, bounded cold-start artifact (0.6% of inner-train bars), not a lookahead.

**No unit-consistency bug found.** Unlike R-128's own first draft, this branch never reads a broker-side position back into the band comparison at all -- `self._held` is tracked entirely in the strategy's own fraction-of-max-leverage units from construction, so `x_bucket` vs `held[bucket]` and `x` vs `last_target` are unit-homogeneous by design, and `ctx.order_target` (not `order_notional`) is used throughout. See BUG LOG in the file's own module docstring.
## 2. Results table

### 2.1 Causal-truncation self-test

**PASS** -- full-period final balance 13014.7519 vs truncated-frame final balance 13014.7519 (BTC spot, 2017-01-01..2020-12-31).

### 2.2 B1 -- BTC signal, spot + futures, full period + inner-validation

| market | window | sharpe_cand | sharpe_base | d_sharpe | boot CI | dd_cand% | dd_base% | trades_cand | trades_base | final_cand | final_base | clears? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spot | full | +1.1647 | +1.0548 | +0.1099 | [-0.3343, +0.9005] | 58.85 | 59.27 | 56 | 1444 | 13043.1 | 9950.4 | False |
| spot | val | -0.3208 | -0.7114 | +0.3906 | [-0.2204, +0.6441] | 41.13 | 59.18 | 18 | 451 | 767.3 | 594.0 | True |
| futures | full | +1.0466 | +1.1398 | -0.0932 | [-3.3735, +0.4013] | 99.94 | 99.87 | 547 | 3002 | 104.5 | 414.4 | False |
| futures | val | -0.6092 | -0.7616 | +0.1523 | [-0.5329, +1.1308] | 99.77 | 99.80 | 244 | 662 | 7.8 | 5.8 | False |

**B1 PASS (all 4 cells clear):** False

### 2.3 B3 -- bucket-horizon-multiplier plateau (uniform across FAST/SLOW/STATIC, FUTURES, inner-validation)

| multiplier | fast_days | slow_days | static_days | d_sharpe | boot CI | sign |
|---|---|---|---|---|---|---|
| 0.5x | 0.0243 | 0.5000 | 3.5000 | +0.2990 | [-0.4314, +1.8965] | +1 |
| 1x | 0.0486 | 1.0000 | 7.0000 | +0.1523 | [-0.5329, +1.1308] | +1 |
| 2x | 0.0972 | 2.0000 | 14.0000 | +0.0718 | [-0.5723, +0.6115] | +1 |
| 4x | 0.1944 | 4.0000 | 28.0000 | +0.1135 | [-0.2128, +0.7660] | +1 |

**B3 PASS (>=3/4 same-signed):** True (4/4 share the majority sign)

### 2.4 B4 -- ETH falsification (spot only, inner-validation, primary config)

ETH spot d_sharpe = +0.3206, boot CI = [-0.1000, +0.6354]. BTC spot inner-validation d_sharpe sign = +1, ETH spot d_sharpe sign = +1. **B4 PASS (sign replicates):** True

### 2.5 B5 -- fee-tier survival (0.40% taker), primary config

| market | window | d_sharpe @0.10% | d_sharpe @0.40% | sign flip? |
|---|---|---|---|---|
| spot | full | +0.1099 | +0.9985 | False |
| spot | val | +0.3906 | +2.1785 | False |
| futures | full | -0.0932 | +0.3331 | True |
| futures | val | +0.1523 | +0.9152 | False |

**B5 PASS (no sign flip, any cell):** False

### 2.6 Diagnostic -- per-bucket re-target counts (primary config, BTC spot, full period; failure modes 1-2 in `r129_shared.py`)

| bucket | horizon_days | re-target count | share of bars re-targeted |
|---|---|---|---|
| fast | 0.0486 | 0 | 0.0000% |
| slow | 1.0000 | 333 | 0.0528% |
| static | 7.0000 | 592 | 0.0938% |

Total candidate trades placed (this cell): 56. Baseline `hedge_experts` trades placed (this cell): 1444. A bar can update more than one bucket's held value while producing at most one order (the buckets are summed before `order_target` is called), so the sum of the three counters is an upper, not a 1-to-1, bound on the candidate's own trade count.

## 3. Configurations evaluated

1 causal-truncation probe + 4 B1 cells + 4 B3 sweep points + 1 B4 cell + 4 B5 cells = **14 total**. The Section 2.6 diagnostic table reuses the already-run BTC-spot/full B1 cell's own strategy instance and required no additional configuration. No selection occurred among any of the 14 -- every cell is reported, none filtered by outcome.

## 4. Decision-rule verdict

causal probe=True  B1=False  B2=diagnostic-only  B3=True  B4=True  B5=False

**VERDICT: NEGATIVE**

(Pre-registered rule from `r129_shared.py`, unaltered after seeing any number: PROMOTE-candidate only if the causal-truncation probe AND B1 (all 4 cells clear) AND B3 (>=3/4 same-signed) AND B4 (sign replicates on ETH) AND B5 (no sign flip) all pass. Anything else is NEGATIVE.)

## 5. Discussion

This branch tests the hypothesis `r129_shared.py`'s docstring names as untested by R-128: does damping at the level of three timescale-bucket sub-blends, rather than the single already-blended output, cut hedge_experts's turnover cost without destroying the responsiveness that makes it profitable? Reading the actual numbers against the five named risks, in order:

(1) **Failure mode 1 (weight drift bypasses any pre-blend damping).** Candidate trade counts: spot full 56 vs baseline 1444; spot val 18 vs 451; futures full 547 vs 3002; futures val 244 vs 662. Turnover is materially lower than baseline in every cell -- the bucket bands do cut trade count despite the Hedge weights moving every bar, so this branch is not structurally defeated by failure mode 1.

(2) **Failure mode 2, bucket-level analogue (near-permanent freezing).** Per-bucket re-target counts (BTC spot, full period, 631,008 bars): fast=0 (0.000% of bars), slow=333 (0.053%), static=592 (0.094%). The FAST bucket re-targets ZERO times across the entire full-period run -- its band, formed from the shortest structural horizon (0.0486d, dominated by the 1-bar-reversion expert's 1/288d horizon among its members), saturates at MAX_BAND=1.0 after clipping (this is exactly failure mode 2 named in `r129_shared.py`, realized here at bucket rather than per-expert granularity, and more severe than the risk as originally framed: not merely 'wide bands' but a band wide enough that the FAST bucket's cold-start value, set once at warmup, is NEVER updated again -- five of the ten experts (1h/6h momentum, MACD, RSI, 1-bar reversion) are silently frozen out of the traded blend for the entire run). The SLOW and STATIC buckets, by contrast, both re-target a comparable, non-trivial number of times (333 and 592 respectively) -- the reverse of the naive expectation that the two structurally-inert STATIC experts (always-flat, buy-and-hold) would freeze most: STATIC's sub-blend is entirely the buy-and-hold expert's own Hedge weight (the always-flat expert contributes zero regardless of its weight), and that weight moves enough, relative to STATIC's own comparatively tight ~7-day-horizon band, to re-target more often than the intuitively 'fastest' bucket does. This is the opposite pattern the round's own docstring anticipated and is reported plainly rather than smoothed over.

(3) **B1.** All four cells reported above do NOT all clear the pre-registered OR -- at least one of the four BTC cells fails to show a d_sharpe/CI/drawdown improvement over unmodified hedge_experts.

(4) **B4, the pre-registered falsification test.** ETH spot d_sharpe = +0.3206 (BTC spot inner-validation sign = +1, ETH sign = +1). The sign replicates, consistent with the pre-registered pass condition.

(5) **B3, the LAG-failure check.** Majority sign 4/4 across the (0.5, 1.0, 2.0, 4.0) bucket-horizon multiplier sweep (uniform across all three buckets). No reversal appears across this range.

(6) **Bucket-partition sensitivity (failure mode 5, disclosed limit).** Only ONE FAST/SLOW/STATIC partition was tested, chosen structurally (native signal period) rather than fit to any return. A different partition (e.g. moving the Donchian breakout, whose 68.97-bar decay half-life sits closer to the fast cluster than its nominal SLOW assignment) could plausibly change this result -- this is a real, disclosed limit of testing one partition, not evidence the specific one chosen is uniquely correct.

**Net read:** the pre-registered decision rule reads this as NEGATIVE. The honest caveat, separate from what the rule concludes: turnover reduction (point 1) is the mechanism's clearest, least ambiguous effect regardless of verdict -- whether that turnover reduction translates into a genuine per-trade timing edge, versus mostly a floor-avoidance/drawdown-avoidance effect on leveraged cells (the same caveat R-128's own conservative branch raised), is not separable from the numbers reported here alone; a bucket-level band, being coarser than a per-expert one, gives less room for any single expert's own timing edge to show through untouched, which argues for reading any BTC edge found here cautiously rather than as confirmation the bucket partition itself is well-tuned.

## 6. Causality / holdout accounting

Max timestamp read anywhere in this branch: 2022-12-31 23:55:00+00:00 (< OOS_START 2023-01-01: True). No bar at or after 2023-01-01 was read by this file. `pytest tests/test_causality_strict.py -q`: 51 passed in 47.11s.
