# R-134 novel — accumulate-and-release deadband on `PaperBroker._execute_target`

Mechanism: a same-sign target adjustment below `deadband * max_notional` is BANKED into a per-broker accumulator (`self._pending_delta`) instead of discarded; every subsequent bar re-evaluates the (recomputed, not summed — see the class docstring for why) accumulated gap against the SAME threshold and releases the FULL gap as one order once it crosses. Sign flips and closes-to-flat always execute immediately and are never banked.

## Hard-invariant self-test (isolated broker, no engine)

- banking + full-gap release on threshold-cross: verified
- re-stating an unchanged target does not grow the accumulator (no double-count): verified
- sign flip while banked always executes immediately, bank wiped: verified
- close-to-flat while banked always executes immediately, bank wiped: verified
- **self-test: PASS**

Data: BTC (real), 631,008 bars, 2017-01-01 00:00:00+00:00 -> 2022-12-31 23:55:00+00:00 (< OOS_START, `_assert_no_holdout` verified).

## F1 — backward compatibility at `DEADBAND_BASELINE` (0.05)

Pre-registered bar: |d_sharpe| <= 0.2 (R-20 noise floor). The novel fix is NOT expected to be bit-identical (it carries suppressed intent forward instead of discarding it) — this is the ±0.2 Sharpe bar, not bit-identical fills, per `r134_shared.py`.

```
       strategy     market       split  final_plain  final_patched  d_final  sharpe_plain  sharpe_patched  d_sharpe  dd_plain  dd_patched  d_dd  trades_plain  trades_patched  d_trades  fills_plain  fills_patched  within_noise_floor
kelly_regime_v4       spot inner-train     18477.37       18477.37      0.0        2.0298          2.0298       0.0    43.255      43.255   0.0            72              72         0          467            467                True
kelly_regime_v4       spot   inner-val       997.98         997.98      0.0        0.1420          0.1420       0.0    33.182      33.182   0.0            52              52         0          256            256                True
kelly_regime_v4 futures_5x inner-train     30344.11       30344.11      0.0        2.2840          2.2840       0.0    35.293      35.293   0.0            72              72         0          261            261                True
kelly_regime_v4 futures_5x   inner-val      1063.70        1063.70      0.0        0.2514          0.2514       0.0    32.291      32.291   0.0            52              52         0          143            143                True
  hedge_experts       spot inner-train     15715.90       15715.90      0.0        1.5924          1.5924       0.0    53.413      53.413   0.0           713             713         0         7658           7658                True
  hedge_experts       spot   inner-val       593.99         593.99      0.0       -0.7114         -0.7114       0.0    59.184      59.184   0.0           451             451         0         5154           5154                True
  hedge_experts futures_5x inner-train     65760.99       65760.99      0.0        1.7265          1.7265       0.0    99.543      99.543   0.0          1543            1543         0        12493          12493                True
  hedge_experts futures_5x   inner-val         5.84           5.84      0.0       -0.7616         -0.7616       0.0    99.797      99.797   0.0           662             662         0         7405           7405                True
```

- max |d_sharpe| observed: **0.0000** (noise floor 0.2)
- **F1: PASS**

**Unplanned finding: F1 is not merely inside the noise floor here, it is bit-identical (d_sharpe = 0.0000, d_final = 0.00, fills unchanged) on all 8 cells.** `r134_shared.py` explicitly anticipated the novel fix could NOT be bit-identical "by construction" — this round found that claim does not hold for the implementation that is actually causal-safe. See the equivalence check immediately below.

## Equivalence check — is the causal accumulate-release policy actually distinguishable from hard-drop?

`self.pos` does not move while a delta sits banked, so at ANY later bar `desired - self.pos` already equals the FULL not-yet-executed gap — the stock hard-drop broker's own "skip this bar, recompute fresh next bar" behaviour already performs exactly this accumulation, for free, using `pos` itself as the memory. A causally-sound accumulate-release (recompute the banked gap each bar rather than SUM it — summing would double-count and let the accumulator grow from mere re-statement of an unchanged target, not from new intent; see the `AccumulateReleaseBroker` class docstring) is therefore mathematically identical, decision for decision, to the existing hard-drop rule at the SAME threshold. Verified directly, not just argued: `NovelTurnoverThrottle` through `AccumulateReleaseBroker(deadband=X)` vs the STOCK `PaperBroker` with `tradebot.broker.REBALANCE_DEADBAND` temporarily patched to the SAME X:

```
 deadband  fills_accrel  fills_harddrop  same_fill_count  identical_equity_curve
    0.050           241             241             True                    True
    0.001           883             883             True                    True
```

- **Equivalence confirmed: True** (both deadband values, futures_5x, inner-train BTC, `NovelTurnoverThrottle`).

Practical implication: under this (the only causal, non-double-counting) implementation, the NOVEL fix's realized decisions — and therefore every backward-compatibility, absorption, and falsification number in this report — are identical to what a broker running plain hard-drop at the SAME threshold value would produce. The two branches' names describe different CODE (a `MarketSpec` field vs. a broker subclass carrying an explicit accumulator/diagnostic state) and different APIs, but not, on this evidence, different EXECUTED POLICIES at a shared threshold. Any residual behavioural difference between the two branches' fixes, if the operator finds one, is not explained by anything measured in this report and would need its own follow-up to characterize.

## F3 — demonstrated capability: absorption at BASELINE vs REALISTIC (`NovelTurnoverThrottle`, inner-train)

```
    market         deadband  deadband_value  intended_asks  immediate_fills  banked_at_least_once  n_bank_events  n_release_events  n_immediate_events  still_pending_at_end  pending_delta_final  pending_as_frac_of_threshold  total_fills
      spot   baseline(0.05)           0.050           1010              439                   571            570               316                 124                 False                  0.0                           0.0          439
      spot realistic(0.001)           0.001           1010              793                   217             45               841                 124                 False                  0.0                           0.0          793
futures_5x   baseline(0.05)           0.050           1010              241                   769            768               118                 124                 False                  0.0                           0.0          241
futures_5x realistic(0.001)           0.001           1010              883                   127            125               761                 124                 False                  0.0                           0.0          883
```

`intended_asks` = distinct new target values the strategy asked the broker for (r72's own convention). `immediate_fills` = asks that produced a fill at their OWN bar (sign flip, close-to-flat, opening from flat, or an ask that itself crossed the release threshold). `banked_at_least_once` = intended_asks - immediate_fills. `still_pending_at_end` / `pending_delta_final` = whatever remains banked, UNRELEASED, at the end of the inner-train window — this is what 'absorbed' means for this branch (carried forward, not discarded), distinct from the conservative branch's simple drop/fill dichotomy.

- **F3: PASS** — bank/release event counts measurably differ between BASELINE and REALISTIC.

## Falsification test — does correcting the deadband confound reverse R-133's NEGATIVE verdict on `NovelTurnoverThrottle`?

At `DEADBAND_REALISTIC` (0.001), B1 (paired bootstrap vs frozen `kelly_regime_v4`, inner-validation, `total_log_return`, `significant=True` AND `paired_diff.point > 0`) on BOTH markets:

**spot**: sharpe_throttle=0.1671, sharpe_v4=0.1420, d_sharpe=+0.0251; paired_diff=+0.01554 [-0.06756, +0.10537], significant=False, b1_pass=False
**futures_5x**: sharpe_throttle=0.1625, sharpe_v4=0.2514, d_sharpe=-0.0889; paired_diff=-0.05197 [-0.20123, +0.09814], significant=False, b1_pass=False

- **Falsification test outcome: NO — R-133 verdict stands** (spot b1_pass=False, futures_5x b1_pass=False; both required to reverse).

## Deadband grid sweep (plateau view, `NovelTurnoverThrottle` vs `kelly_regime_v4`, inner-validation)

### futures_5x

```
    market  deadband  sharpe_thr  sharpe_v4  d_sharpe  dd_thr  dd_v4  trades_thr  trades_v4  paired_diff  paired_lo  paired_hi  significant  b1_pass
futures_5x     0.000      0.1635     0.2514   -0.0879  34.363 32.291          30         52    -0.051374  -0.201544   0.100529        False    False
futures_5x     0.001      0.1625     0.2514   -0.0889  34.363 32.291          30         52    -0.051975  -0.201232   0.098142        False    False
futures_5x     0.005      0.1496     0.2514   -0.1018  34.789 32.291          30         52    -0.059672  -0.211038   0.088711        False    False
futures_5x     0.010      0.1340     0.2514   -0.1174  35.284 32.291          30         52    -0.069369  -0.228003   0.086030        False    False
futures_5x     0.020      0.0985     0.2514   -0.1529  35.780 32.291          30         52    -0.090437  -0.256464   0.072268        False    False
futures_5x     0.050      0.0400     0.2514   -0.2114  38.373 32.291          30         52    -0.123558  -0.295739   0.019478        False    False
```

### spot

```
market  deadband  sharpe_thr  sharpe_v4  d_sharpe  dd_thr  dd_v4  trades_thr  trades_v4  paired_diff  paired_lo  paired_hi  significant  b1_pass
  spot     0.000      0.1671      0.142    0.0251  33.510 33.182          30         52     0.015567  -0.067543   0.105425        False    False
  spot     0.001      0.1671      0.142    0.0251  33.512 33.182          30         52     0.015536  -0.067563   0.105367        False    False
  spot     0.005      0.1653      0.142    0.0233  33.544 33.182          30         52     0.014494  -0.070172   0.105340        False    False
  spot     0.010      0.1660      0.142    0.0240  33.633 33.182          30         52     0.014920  -0.070265   0.107262        False    False
  spot     0.020      0.1569      0.142    0.0149  33.606 33.182          30         52     0.009556  -0.080697   0.105402        False    False
  spot     0.050      0.1398      0.142   -0.0022  34.374 33.182          30         52    -0.000921  -0.106309   0.101663        False    False
```

## F2 — pytest (see terminal output / CI log for the authoritative run; summarized below)

Run separately by the driver script (`python -m pytest`) after this file; see the report footer / session transcript for the pass/fail counts and `tests/test_causality_strict.py` status.

## Configurations evaluated (this branch's contribution to the cross-branch `r134_shared._CONFIGS` counter): **42**

## Verdict against the pre-registered decision rule (F1-F3 in `r134_shared.py`)

- F1 (backward compatibility, ±0.2 Sharpe floor): **PASS** (max |d_sharpe| = 0.0000)
- F2 (no regressions, full pytest green): see F2 section above / session output
- F3 (demonstrated capability at DEADBAND_REALISTIC): **PASS**

- Falsification test (does the fix reverse R-133's NEGATIVE verdict on `NovelTurnoverThrottle`, BOTH markets): **NO**

B-43 closes cleanly for this mechanism: `NovelTurnoverThrottle` still fails B1 on at least one market even under the corrected broker, so R-133's section-C entry stands as final rather than provisional for this mechanism. The evaluability defect (the floor's own coarseness) is still worth fixing per the ADOPTION rule, independent of this outcome.

## New risk this specific mechanism introduces (that the conservative, pure-threshold fix does not)

Honestly: on the equivalence finding above, **none that this round could measure.** The intuitive risk this task named up front — an accumulated position the strategy no longer "intends" by the time a multi-bar-old release fires — does NOT materialize in the implementation built here, because the released delta is always `desired - pos` computed from the CURRENT bar's freshly-computed `target`, never from a stale target value captured back when banking began. There is no stored "old desire" that outlives its own bar; `self.pos` not moving is what carries the state forward, and every recompute re-reads the strategy's live output. Since the equivalence check confirms this collapses to hard-drop's own decisions bit-for-bit, this specific implementation carries no NEW execution risk beyond what B-43 already diagnosed for the existing broker.

What IS worth flagging: the pre-registration's own prose ("a same-sign adjustment... is instead banked... pending + newly-desired delta") reads most naturally as an ADDITIVE (summed) accumulator, not a recomputed one. An additive implementation would NOT be equivalent to hard-drop — it would let the accumulator grow from mere bar-over-bar re-statement of an unchanged target (both `kelly_regime_v4` and `NovelTurnoverThrottle` re-emit a `target` every bar even with no new intent), and could eventually release a position change LARGER than the strategy's current target ever asked for at any single bar — a genuine, and genuinely risky, staleness/overshoot failure mode, exactly of the shape this task's own risk prompt anticipated. This round deliberately did NOT build that version (it is not causal-unsound in the lookahead sense — it never reads a future bar — but it double-counts already-realized state and is not, on inspection, an economically coherent policy); flagging the ambiguity explicitly is the honest thing to do rather than silently picking one reading and reporting it as if it were the only one the pre-registration allowed.
