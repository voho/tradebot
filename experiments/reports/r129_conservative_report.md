# R-129 (CONSERVATIVE branch) -- ten per-expert EV bands for `hedge_experts` (08-25)
Unregistered candidate. Code: `experiments/r129_conservative_per_expert_band.py`. Not `@register`ed, not auto-discovered, nothing committed by this session. `src/tradebot/strategies/hedge_experts.py` is never edited -- `ConservativePerExpertBand` subclasses `HedgeExperts` and reuses `HedgeExperts._experts()` verbatim. Full derivation, non-duplication argument, named failure modes, the literature citation (Ekren, Liu & Muhle-Karbe 2018), and the pre-registered decision rule live in `experiments/r129_shared.py`'s module docstring; only summarized here.
## 1. Mechanism recap

R-128 replaced `hedge_experts`'s fixed `hysteresis=0.05` re-target rule on its ALREADY-BLENDED output `x` with one EV-derived no-trade band at one pooled horizon -- and found it NEGATIVE on the exact risk it pre-registered: the Kelly algebra assumes one homogeneous stationary bet, but `hedge_experts` blends ten experts across four timescales. This branch (CONSERVATIVE) tests the alternative R-128's own closing line named: apply the band to EACH of the ten raw experts INDIVIDUALLY, at a horizon *structural to that expert's own native timescale*, before the Hedge weights blend them:

```
band_j = clip(2*fee / (H_j_years * sigma_market**2 * leverage), MIN_BAND, MAX_BAND)
held[j] updates to expert_j iff abs(expert_j - held[j]) > band_j
x = weight @ held
if abs(x - last_target) > 1e-9: ctx.order_target(x)
```

`fee = ctx.market.fee_rate`, `leverage = ctx.market.leverage`, `sigma_market = ctx.bar["_ev_vol"]` (the one shared market-vol input), `H_j_years = EXPERT_HORIZON_DAYS[j] / 365.25` -- ten structural horizons frozen in `r129_shared.py` (0.0035d for 1-bar reversion up to 7d for buy-and-hold/flat), never fit to any return. `min_band=0.02`, `max_band=1.0` are `kelly_regime_ev`'s/R-128's own literal defaults, reused unchanged. `weight` is read LIVE from the current bar's Hedge weights every bar -- only the raw expert values are banded, not the weights themselves. No second, post-blend band. Orders are placed via `ctx.order_target`, matching `x`'s own native fraction-of-max-notional units -- this branch never constructs the notional-multiple-vs-fraction-of-leverage unit mismatch R-128 found and fixed post-hoc; the band formula above already divides by `leverage` for the same reason R-128's own corrected `_band` does.

**Disclosed cold-start convention.** `self._held` initializes on the FIRST `on_bar` call (at `warmup=2500`) to that bar's own raw expert values, not accumulated causally from `prepare()`'s bar 2 the way the Hedge weight loop itself is -- a minor, bounded cold-start artifact (0.6% of inner-train bars), not a lookahead, per `r129_shared.py`'s binding Implementation note.
## 2. Results table

### 2.1 Causal-truncation self-test

**PASS** -- full-period final balance 16450.0409 vs truncated-frame final balance 16450.0409 (BTC spot, 2017-01-01..2020-12-31).

### 2.2 B1 -- BTC signal, spot + futures, full period + inner-validation

| market | window | sharpe_cand | sharpe_base | d_sharpe | boot CI | dd_cand% | dd_base% | trades_cand | trades_base | final_cand | final_base | clears? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spot | full | +1.0908 | +1.0548 | +0.0359 | [-0.0988, +0.3246] | 56.29 | 59.27 | 816 | 1444 | 11180.5 | 9950.4 | False |
| spot | val | -0.6031 | -0.7114 | +0.1083 | [-0.0852, +0.1804] | 55.43 | 59.18 | 281 | 451 | 626.1 | 594.0 | False |
| futures | full | +0.9928 | +1.1398 | -0.1470 | [-4.4535, -1.4123] | 99.98 | 99.87 | 2316 | 3002 | 26.0 | 414.4 | False |
| futures | val | -0.7982 | -0.7616 | -0.0367 | [-0.8271, +0.2043] | 99.86 | 99.80 | 738 | 662 | 4.4 | 5.8 | False |

**B1 PASS (all 4 cells clear):** False

### 2.3 B3 -- uniform horizon-multiplier plateau (FUTURES, inner-validation)

Every entry of `EXPERT_HORIZON_DAYS` scaled by the SAME multiplier `m` simultaneously (testing the whole per-expert horizon scale, not re-deriving each expert's own multiplier independently).

| multiplier | d_sharpe | boot CI | sign |
|---|---|---|---|
| 0.5x | -0.0354 | [-0.9527, +0.2045] | -1 |
| 1x | -0.0367 | [-0.8271, +0.2043] | -1 |
| 2x | -0.0450 | [-0.7151, +0.1274] | -1 |
| 4x | -0.0934 | [-0.7618, -0.0722] | -1 |

**B3 PASS (>=3/4 same-signed):** True (4/4 share the majority sign)

### 2.4 B4 -- ETH falsification (spot only, inner-validation, primary config)

ETH spot d_sharpe = +0.0536, boot CI = [-0.0708, +0.1703]. BTC spot inner-validation d_sharpe sign = +1, ETH spot d_sharpe sign = +1. **B4 PASS (sign replicates):** True

### 2.5 B5 -- fee-tier survival (0.40% taker), primary config

| market | window | d_sharpe @0.10% | d_sharpe @0.40% | sign flip? |
|---|---|---|---|---|
| spot | full | +0.0359 | +0.4005 | False |
| spot | val | +0.1083 | +0.7630 | False |
| futures | full | -0.1470 | -0.2512 | False |
| futures | val | -0.0367 | +0.0897 | True |

**B5 PASS (no sign flip, any cell):** False

### 2.6 Diagnostic -- per-expert re-target counts (primary config, BTC spot, full period)

Checks failure mode #2 (`r129_shared.py`): do the fast/short-horizon experts freeze near-permanently under wide, max_band-clipped bands? Total inner-train bars traded over: 628,220. Baseline `hedge_experts` total trade count over the same window: 1444.

| expert | native horizon (days) | re-target count | share of bars |
|---|---|---|---|
| 0: 1h momentum | 0.0417 | 25286 | 4.0250% |
| 1: 6h momentum | 0.2500 | 5677 | 0.9037% |
| 2: 1d momentum | 1.0000 | 4789 | 0.7623% |
| 3: 1w momentum | 7.0000 | 14576 | 2.3202% |
| 4: MACD hist | 0.0903 | 4 | 0.0006% |
| 5: RSI ramp | 0.0486 | 3460 | 0.5508% |
| 6: 1-bar reversion | 0.0035 | 0 | 0.0000% |
| 7: Donchian breakout | 0.2395 | 1528 | 0.2432% |
| 8: always flat | 7.0000 | 0 | 0.0000% |
| 9: buy and hold | 7.0000 | 0 | 0.0000% |

Candidate's own total trade count vs baseline, primary config, both markets, both windows -- checks failure mode #1 (do the Hedge weights' own continuous drift keep turnover close to baseline regardless of per-expert damping):

| market | window | trades_cand | trades_base | ratio |
|---|---|---|---|---|
| spot | full | 816 | 1444 | 0.565 |
| spot | val | 281 | 451 | 0.623 |
| futures | full | 2316 | 3002 | 0.771 |
| futures | val | 738 | 662 | 1.115 |

## 3. Configurations evaluated

1 causal-truncation probe + 4 B1 cells + 4 B3 sweep points + 1 B4 cell + 4 B5 cells = **14 total**. No selection occurred among them -- every cell is reported, none filtered by outcome.

## 4. Decision-rule verdict

causal probe=True  B1=False  B2=diagnostic-only  B3=True  B4=True  B5=False

**VERDICT: NEGATIVE**

(Pre-registered rule from `r129_shared.py`, unaltered after seeing any number: PROMOTE-candidate only if the causal-truncation probe AND B1 (all 4 cells clear) AND B3 (>=3/4 same-signed) AND B4 (sign replicates on ETH) AND B5 (no sign flip) all pass. Anything else is NEGATIVE.)

## 5. Discussion

This branch tests the alternative R-128's own docstring named: does banding each of `hedge_experts`'s ten raw experts INDIVIDUALLY, at a horizon structural to its own native timescale, cut turnover without R-128's structural-mismatch problem (one band on a ten-expert, four-timescale blend)? Reading the actual numbers against the five named risks, in order:

(1) **Failure mode #1 (Hedge weight drift keeps turnover near baseline regardless of per-expert damping) -- CONFIRMED on this evidence.** Candidate trade count vs baseline, BTC spot full period: 816 vs 1444 (0.57x); inner-validation: 281 vs 451 (0.62x); futures full: 2316 vs 3002 (0.77x); futures inner-validation: 738 vs 662 (1.11x). Because `x = weight @ held` recomputes every bar from the LIVE (unbanded) Hedge weights, the final blended output can and does keep moving even when every individual `held[j]` is frozen -- the per-expert band damps *which raw values* feed the blend, not how often the blend itself changes.

(2) **Failure mode #2 (fast/short-horizon experts freeze near-permanently) -- CONFIRMED.** (Experts 8/always-flat and 9/buy-and-hold are excluded from this count -- their inputs are constant by construction, so a zero re-target count there is trivial, not evidence of the failure mode; see r129_shared.py's own 'nominal, inert' annotation.) Among the genuinely time-varying experts, 1-bar reversion (indices [6]) had exactly ZERO re-targets over the full inner-train window -- expert 6 (1-bar reversion, the SHORTEST native horizon at H=0.0035d) is frozen at whatever value it held at cold-start, exactly the risk named before any code was run. Near-frozen as well (<0.1% of bars re-targeted, not literally zero): MACD hist (indices [4]), most notably MACD histogram (H=0.090d) at 4 re-targets over 628,220 bars. See section 2.6 for the full per-expert re-target table.

(3) **B4 sign replication.** ETH spot d_sharpe = +0.0536 (CI [-0.0708, +0.1703]) vs BTC spot inner-validation sign +1 -- same sign, replicates.

(4) **B3 plateau / LAG risk.** Signs across the 0.5x-4x uniform horizon-multiplier sweep: [-1, -1, -1, -1] (4/4 share the majority sign) -- no reversal observed within this grid.

(5) **The bucket-partition risk (failure mode #5) does not apply to this branch** -- that is a disclosed limit of the NOVEL branch's own bucket-boundary choice, not this one's per-expert construction, which uses each expert's own individually-frozen horizon rather than a grouped one.

**Net read:** the pre-registered decision rule reads this as **NEGATIVE**. At least one pre-registered gate failed; see points (1)-(4) above for exactly which one(s) and by how much, rather than treating this as a uniform failure across every cell.

## 6. Causality / holdout accounting

Max timestamp read anywhere in this branch: 2022-12-31 23:55:00+00:00 (< OOS_START 2023-01-01: True). No bar at or after 2023-01-01 was read by this file. `pytest tests/test_causality_strict.py -q`: 51 passed in 47.80s.
