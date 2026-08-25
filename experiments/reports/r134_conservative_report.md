# R-134 CONSERVATIVE — `MarketSpec.deadband` simulated as a broker-subclass attribute

Frozen pre-registration: `experiments/r134_shared.py`. Object under test: `NovelTurnoverThrottle` (R-133, `experiments/r133_mechanisms.py`, imported not copied).

`DEADBAND_BASELINE = 0.05`, `DEADBAND_REALISTIC = 0.001`, `DEADBAND_GRID = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]`. `THROTTLE_UPPER = 0.9091` trades/day, `THROTTLE_ETA = 0.5` (R-133's frozen operating point, imported not re-derived).

**Mechanism, one sentence:** the broker's `REBALANCE_DEADBAND` floor is read from a settable per-broker `self.deadband` attribute (default 0.05, backward-compatible) instead of the module-level constant, so a same-sign rebalance shrunk by a mechanism like `NovelTurnoverThrottle` is compared against a threshold that can be set to a venue-realistic size instead of an arbitrary flat 5% of max notional.

## F1 — backward compatibility (bit-identical at DEADBAND_BASELINE = 0.05)

Ran in 88.2s.

```
       strategy     market       split  fills_default  fills_patched  final_default  final_patched  sharpe_default  sharpe_patched  dd_default  dd_patched  trades_default  trades_patched  bit_identical
kelly_regime_v4       spot inner-train            467            467     18477.3700     18477.3700        2.029845        2.029845   43.254828   43.254828              72              72           True
kelly_regime_v4       spot   inner-val            256            256       997.9766       997.9766        0.141970        0.141970   33.182451   33.182451              52              52           True
kelly_regime_v4 futures_5x inner-train            261            261     30344.1083     30344.1083        2.284003        2.284003   35.292685   35.292685              72              72           True
kelly_regime_v4 futures_5x   inner-val            143            143      1063.6961      1063.6961        0.251410        0.251410   32.291158   32.291158              52              52           True
  hedge_experts       spot inner-train           7658           7658     15715.8969     15715.8969        1.592387        1.592387   53.413320   53.413320             713             713           True
  hedge_experts       spot   inner-val           5154           5154       593.9867       593.9867       -0.711422       -0.711422   59.183556   59.183556             451             451           True
  hedge_experts futures_5x inner-train          12493          12493     65760.9905     65760.9905        1.726466        1.726466   99.543369   99.543369            1543            1543           True
  hedge_experts futures_5x   inner-val           7405           7405         5.8401         5.8401       -0.761566       -0.761566   99.797387   99.797387             662             662           True
```

**F1: PASS** — 8 of 8 (strategy x market x split) cells bit-identical (fills, full equity curve, final balance, Sharpe, max drawdown, trade count all compared, not just 'close enough').

## F3 — demonstrated capability (absorption rate, `NovelTurnoverThrottle`, inner-train)

Ran in 20.8s.

```
    market  deadband  deadband_value  intended  filled  absorption_rate  fills_in_result  trades  final_balance  sharpe
      spot  baseline           0.050      1010     439         0.434653              439      62       17992.26   2.010
      spot realistic           0.001      1010     793         0.785149              793      62       18402.49   2.022
futures_5x  baseline           0.050      1010     241         0.238614              241      62       28268.05   2.211
futures_5x realistic           0.001      1010     883         0.874257              883      62       25987.30   2.182
```

- **spot**: absorption 43.5% (baseline, deadband=0.05) -> 78.5% (realistic, deadband=0.001), delta +35.0%. Filled/intended: 439/1010 -> 793/1010.
- **futures_5x**: absorption 23.9% (baseline, deadband=0.05) -> 87.4% (realistic, deadband=0.001), delta +63.6%. Filled/intended: 241/1010 -> 883/1010.

**F3: PASS** — the fix measurably changes the fill-through/absorption rate on both markets when configured at `DEADBAND_REALISTIC`.

## F2 — no regressions (`pytest`)

`pytest -q` from repo root, venv active. Exit code 0. Ran in 224.0s.

```
........................................................................ [ 13%]
........................................................................ [ 27%]
........................................................................ [ 41%]
........................................................................ [ 55%]
........................................................................ [ 69%]
........................................................................ [ 83%]
........................................................................ [ 97%]
............                                                             [100%]
516 passed in 223.15s (0:03:43)
```

**F2: PASS** (`tests/test_causality_strict.py` included in the full run above; this branch adds zero changes to `src/`).

## Causal-truncation self-test on the patched-broker logic

`MarketDeadbandBroker._execute_target` introduces no new stateful or rolling computation: the one changed line (`self.deadband * max_notional` in place of `REBALANCE_DEADBAND * max_notional`) is a per-bar comparison against the broker's own current `pos`/`cash`/`entry` and the bar's own `price`/`target`, exactly like every other line of `_execute_target` it was copied from — no scaler, quantile, mean, std, or window is computed over the series. That is an argument from reading the code, which is exactly the kind of claim this project has been burned by assuming (R-21's $3.7e23 causality bug), so a direct truncation probe was also run rather than relying on it alone:

- `NovelTurnoverThrottle` through `MarketDeadbandBroker(deadband=0.001)`, futures_5x, inner-train: full-frame fills up to the truncation timestamp vs truncated-frame fills — 623 vs 623, **IDENTICAL**.

## Falsification test — does the corrected broker reverse R-133's B1 verdict?

Frozen wording (`r134_shared.py`): under the patched broker at `DEADBAND_REALISTIC`, does `NovelTurnoverThrottle` clear B1 (paired bootstrap vs frozen `kelly_regime_v4`, the comparison arm always through the DEFAULT unpatched broker, inner-validation, `total_log_return`, `significant=True` AND `paired_diff.point > 0`) on BOTH markets?

Ran in 36.0s.

```
    market  deadband  sharpe_thr  sharpe_v4  d_sharpe  dd_thr  dd_v4  trades_thr  trades_v4  fills_thr  fills_v4  paired_diff       lo      hi  significant  b1_pass
      spot     0.000       0.167      0.142     0.025   33.51  33.18          30         52        518       256      0.01557 -0.06754 0.10542        False    False
      spot     0.001       0.167      0.142     0.025   33.51  33.18          30         52        509       256      0.01554 -0.06756 0.10537        False    False
      spot     0.005       0.165      0.142     0.023   33.54  33.18          30         52        458       256      0.01449 -0.07017 0.10534        False    False
      spot     0.010       0.166      0.142     0.024   33.63  33.18          30         52        362       256      0.01492 -0.07026 0.10726        False    False
      spot     0.020       0.157      0.142     0.015   33.61  33.18          30         52        286       256      0.00956 -0.08070 0.10540        False    False
      spot     0.050       0.140      0.142    -0.002   34.37  33.18          30         52        221       256     -0.00092 -0.10631 0.10166        False    False
futures_5x     0.000       0.164      0.251    -0.088   34.36  32.29          30         52        520       143     -0.05137 -0.20154 0.10053        False    False
futures_5x     0.001       0.163      0.251    -0.089   34.36  32.29          30         52        466       143     -0.05197 -0.20123 0.09814        False    False
futures_5x     0.005       0.150      0.251    -0.102   34.79  32.29          30         52        265       143     -0.05967 -0.21104 0.08871        False    False
futures_5x     0.010       0.134      0.251    -0.117   35.28  32.29          30         52        231       143     -0.06937 -0.22800 0.08603        False    False
futures_5x     0.020       0.098      0.251    -0.153   35.78  32.29          30         52        211       143     -0.09044 -0.25646 0.07227        False    False
futures_5x     0.050       0.040      0.251    -0.211   38.37  32.29          30         52        101       143     -0.12356 -0.29574 0.01948        False    False
```

### Verdict cell (DEADBAND_REALISTIC only)

- **spot**: paired_diff=+0.01554 [-0.06756, +0.10537], significant=False, b1_pass=False (Sharpe throttle 0.167 vs v4 0.142, d_sharpe=+0.025).
- **futures_5x**: paired_diff=-0.05197 [-0.20123, +0.09814], significant=False, b1_pass=False (Sharpe throttle 0.163 vs v4 0.251, d_sharpe=-0.089).

**Falsification test outcome: NO (fails B1 on at least one market).**

Per `r134_shared.py`'s own pre-registered reading of this outcome: B-43 closes cleanly. R-133's section C entry becomes final rather than provisional, per its own text. The evaluability defect (the floor's coarseness) is still worth fixing because it affects every FUTURE size-shrinking mechanism this project tries, but it does not resurrect `NovelTurnoverThrottle`.

- **spot** flip vs baseline deadband: b1_pass False -> False (unchanged). d_sharpe moved by +0.027 between baseline and realistic deadband, which is INSIDE the +/-0.2 Sharpe noise floor (R-20).
- **futures_5x** flip vs baseline deadband: b1_pass False -> False (unchanged). d_sharpe moved by +0.122 between baseline and realistic deadband, which is INSIDE the +/-0.2 Sharpe noise floor (R-20).

## Configs evaluated

`r134_shared._CONFIGS[0]` is a shared, cross-branch counter (`note_config()` incremented once per backtest, observed rather than remembered) — every `run_period` call in this file sits behind one `note_config()`/`note()` call, directly or via `r134_shared.b1_throttle_vs_v4` / `v4_reference`. Running total by section (checkpointed as the file ran):

```
start                        cumulative=   0  (+0 this section)
after F1                     cumulative=  16  (+16 this section)
after F3                     cumulative=  20  (+4 this section)
after truncation probe       cumulative=  22  (+2 this section)
after falsification grid     cumulative=  36  (+14 this section)
```

**Total configs evaluated by this file: 36.** (The operator's final cross-branch count per `docs/ROUTINE.md` also includes whatever the NOVEL branch adds to this same shared counter object, if run in the same process; if run as a separate process, the two branch counts are summed by the operator.)

## Verdict against the pre-registered ADOPTION decision rule (`r134_shared.py`)

- **F1** (backward compatibility, bit-identical at 0.05): **PASS**
- **F2** (no regressions, full `pytest` green): **PASS**
- **F3** (demonstrated capability, absorption rate changes at DEADBAND_REALISTIC): **PASS**

F1/F2/F3 all clear: **True**. Per `r134_shared.py`'s own text, clearing F1-F3 makes this fix ELIGIBLE for the operator to select between it and the NOVEL branch — **this file does not itself declare a fix ADOPTED or PROMOTED**; that decision belongs to the operator after both branches report.

Falsification test: **NO (fails B1 on at least one market)**. This is reported as evidence for a possible follow-up round (if YES on both markets) or as B-43's closure (if NO on at least one market) — **not** as a promotion of `NovelTurnoverThrottle` itself either way, per the round's own scope ("this round does not attempt B3/B4/B5 by design").

No bar at or after `OOS_START = 2023-01-01` was read by this file. Holdout consultations added: 0.
