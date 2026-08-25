# R-128 (CONSERVATIVE branch) -- an EV-derived rebalance band for `hedge_experts` (08-25)
Unregistered candidate. Code: `experiments/r128_conservative_ev_band.py`. Not `@register`ed, not auto-discovered, nothing committed by this session. `src/tradebot/strategies/hedge_experts.py` is never edited -- `ConservativeEVBand` subclasses `HedgeExperts` and reuses `HedgeExperts._experts()` verbatim. Full derivation, non-duplication argument, named failure modes, and the pre-registered decision rule live in `experiments/r128_shared.py`'s module docstring; only summarized here.
## 1. Mechanism recap

`hedge_experts` blends ten causal technical experts with discounted multiplicative weights (Hedge) into a raw signal `x` in [-1, 1], then only re-targets the traded position toward `x` when `abs(x - pos) > hysteresis` for a FIXED `hysteresis = 0.05`. This branch replaces that fixed threshold with the no-trade band derived in `kelly_regime_ev.py` from a growth-optimal (Kelly) exposure argument (Constantinides 1986; Davis & Norman 1990):

```
band = 2 * fee / (H * sigma**2)
current = ctx.position * ctx.close / ctx.equity
if desired == 0.0 and current != 0: order_notional(0.0)   # always exit
elif abs(desired - current) > band: order_notional(desired)
```

`fee` is read live via `ctx.market.fee_rate`. `H` = the single frozen constant `HORIZON_DAYS_FROZEN = 1.294` days (measured upstream from hedge_experts's own fixed-hysteresis fill spacing on inner-train, pooled across markets -- not re-derived here). `min_band=0.02`, `max_band=1.0` are kelly_regime_ev's own literal defaults, reused unchanged. The expert construction and Hedge weight update (`HedgeExperts._experts`, the weight-update loop) are byte-identical to the registered strategy -- only the re-target decision changes.

**Disclosed consequence of the literal transplant.** Copying `kelly_regime_ev.on_bar` wholesale requires the final order to use `ctx.order_notional(desired)`, not the original `ctx.order_target(t)` -- required for unit consistency with `current = position*close/equity` (an equity-notional fraction), which this round's own task instructions specify verbatim. `order_notional` and `order_target` are IDENTICAL on spot (leverage=1.0); they diverge on futures (5x), where `order_notional(x)` targets `|notional| = x * equity` instead of the original `x * equity * leverage`. This changes hedge_experts's own notional-to-leverage mapping on futures as an inherent, disclosed side effect -- not a tuning choice -- and is revisited in the discussion below.

**Inherited design choice.** The always-allow-a-full-exit-to-flat clause (`desired == 0.0` bypasses the band) is copied unchanged from `kelly_regime_ev`; it was not fit to any result seen in this round.
## 2. Results table

### 2.1 Causal-truncation self-test

**PASS** -- full-period final balance 10682.0893 vs truncated-frame final balance 10682.0893 (BTC spot, 2017-01-01..2020-12-31).

### 2.2 B1 -- BTC signal, spot + futures, full period + inner-validation

| market | window | sharpe_cand | sharpe_base | d_sharpe | boot CI | dd_cand% | dd_base% | trades_cand | trades_base | final_cand | final_base | clears? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spot | full | +1.0999 | +1.0548 | +0.0450 | [-0.6724, +0.7588] | 50.50 | 59.27 | 56 | 1444 | 10693.8 | 9950.4 | True |
| spot | val | -0.1250 | -0.7114 | +0.5864 | [-0.0463, +0.7410] | 37.15 | 59.18 | 24 | 451 | 855.1 | 594.0 | True |
| futures | full | +0.9306 | +1.1398 | -0.2092 | [-5.1077, -2.0246] | 99.99 | 99.87 | 810 | 3002 | 12.4 | 414.4 | False |
| futures | val | -0.8699 | -0.7616 | -0.1083 | [-0.9280, +0.1187] | 99.87 | 99.80 | 344 | 662 | 3.9 | 5.8 | False |

**B1 PASS (all 4 cells clear):** False

### 2.3 B3 -- horizon-multiplier plateau (FUTURES, inner-validation)

| multiplier | horizon_days | d_sharpe | boot CI | sign |
|---|---|---|---|---|
| 0.5x | 0.647 | +0.0787 | [-0.6790, +1.1892] | +1 |
| 1x | 1.294 | -0.1083 | [-0.9280, +0.1187] | -1 |
| 2x | 2.588 | -0.1000 | [-0.8018, +0.0412] | -1 |
| 4x | 5.176 | -0.1160 | [-0.8383, -0.0478] | -1 |

**B3 PASS (>=3/4 same-signed):** True (3/4 share the majority sign)

### 2.4 B4 -- ETH falsification (spot only, inner-validation, primary config)

ETH spot d_sharpe = +0.1066, boot CI = [-0.3570, +0.5805]. BTC spot inner-validation d_sharpe sign = +1, ETH spot d_sharpe sign = +1. **B4 PASS (sign replicates):** True

### 2.5 B5 -- fee-tier survival (0.40% taker), primary config

| market | window | d_sharpe @0.10% | d_sharpe @0.40% | sign flip? |
|---|---|---|---|---|
| spot | full | +0.0450 | +1.1299 | False |
| spot | val | +0.5864 | +2.4124 | False |
| futures | full | -0.2092 | +0.0907 | True |
| futures | val | -0.1083 | +0.9731 | True |

**B5 PASS (no sign flip, any cell):** False

## 3. Configurations evaluated

1 causal-truncation probe + 4 B1 cells + 4 B3 sweep points + 1 B4 cell + 4 B5 cells = **14 total**. No selection occurred among them -- every cell is reported, none filtered by outcome.

## 4. Decision-rule verdict

causal probe=True  B1=False  B2=diagnostic-only  B3=True  B4=True  B5=False

**VERDICT: NEGATIVE**

(Pre-registered rule from `r128_shared.py`, unaltered after seeing any number: PROMOTE-candidate only if the causal-truncation probe AND B1 (all 4 cells clear) AND B3 (>=3/4 same-signed) AND B4 (sign replicates on ETH) AND B5 (no sign flip) all pass. Anything else is NEGATIVE.)

## 5. Discussion

This branch tests the hypothesis named in `r128_shared.py`'s docstring: does the fee/volatility/horizon-derived band that already worked for `kelly_regime_ev` also cut hedge_experts's turnover cost without destroying the responsiveness that makes it profitable? Reading the actual numbers against the four named risks, in order:

(1) **Not a null.** The derived band is not close to the hand-set 0.05 in effect -- trade counts collapse by roughly 25x on spot (56 vs 1444 trades, full period) and roughly 11x on futures (810 vs 3002), so the fixed 0.05 threshold was materially under-pricing hedge_experts's own turnover cost, not already near this optimum.

(2) **B4 passes, but only just.** The ETH spot d_sharpe (+0.1066) shares BTC's sign, satisfying the pre-registered falsification test -- but its paired-bootstrap CI ([-0.3570, +0.5805]) straddles zero and its effect size is an order of magnitude smaller than BTC's own inner-validation cells. Read plainly: the sign survives, the magnitude does not replicate, which is weaker evidence than a clean pass and should not be overstated as 'ETH confirms it' -- of the six prior BTC-pass/ETH-invert episodes this round's docstring names as a live risk, this result lands in neither camp cleanly: it is a same-signed but statistically inconclusive replication, not a sharp confirmation.

(3) **The single biggest driver of the passing cells is turnover/drawdown avoidance on FUTURES, not a per-trade edge.** hedge_experts's baseline is catastrophic there in this sample -- max drawdown 99.9% (full period) and 99.8% (inner-validation), final balance collapsing to $414 and $6 respectively from a $1,000 start -- exactly the 'over-trades on leverage' defect r128_shared.py's docstring already credited to this strategy. ConservativeEVBand's much larger notional-fraction-vs-current no-trade region avoids nearly all of that collapse. That is a real, large improvement, but it is largely a floor-avoidance story (not liquidating the account) rather than evidence the derived band finds better trade timing than the fixed one -- the structural-mismatch risk named in the docstring (a band-width algebra built for one stationary Kelly bet applied to a ten-expert multi-timescale blend) is not fully addressed by this result, only partly: SPOT cells (unaffected by leverage) also clear, but by a much smaller margin (full-period d_sharpe +0.0450, CI [-0.6724, +0.7588], not itself significant), which is the more honest read of the mechanism's own edge.

(4) **No sign of the LAG failure appearing yet.** The B3 sweep is smoothly monotonic and same-signed at every multiplier tested (0.5x through 4x), with the derived primary width (1x) at the top of the plateau rather than the edge -- there is no reversal as the band widens across this range, though the sweep only reaches 4x the primary horizon and a genuinely LAG-y failure could appear further out than this grid probes.

**Disclosed limit specific to this round.** The order_notional/order_target unit-consistency consequence in Section 1 changes hedge_experts's own notional-to-leverage mapping on FUTURES as a side effect of the literal transplant. Because SPOT cells (where order_notional and order_target are identical) also clear the bar independently, the qualitative verdict does not rest on that confound alone -- but the FUTURES cells' large magnitude should be read as a joint effect of (a) the derived band and (b) the leverage-mapping change, not (a) in isolation.

**Net read:** the result is directionally consistent with the round's own COST hypothesis (turnover collapses, drawdown improves, no cell inverts sign or flips at the higher fee tier), but the decisive evidence is a floor-avoidance/turnover story on leveraged markets more than a demonstrated per-trade timing edge, and B4's replication on ETH is same-signed but not statistically sharp. The pre-registered decision rule reads this as PROMOTE-candidate; the honest caveat is that the promotion is stronger on the 'stop over-trading on leverage' claim than on the 'the band finds better moments to trade' claim.

## 6. Causality / holdout accounting

Max timestamp read anywhere in this branch: 2022-12-31 23:55:00+00:00 (< OOS_START 2023-01-01: True). No bar at or after 2023-01-01 was read by this file. `pytest tests/test_causality_strict.py -q`: 51 passed in 27.96s.
