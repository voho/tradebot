# R-60 conservative — KAMA-adaptive vote anchors

Pre-registration: `experiments/r60_shared.py`. Mechanism: `experiments/r60_conservative_kama_anchors.py`. Backlog **B-26**.

## Mechanism

`kelly_regime_v4`'s three fixed-window SMA anchors (20/40/80-day) are replaced by Kaufman (1995) KAMA-style adaptive averages bracketed around the same nominal horizons, so each anchor's effective memory shortens in efficient/trending action and lengthens in noise/chop. Formula, bracket multiplier `k` and warmup derivation are in the module docstring. Everything else (1% latch band, vote-fraction average, fractional-Kelly sizing, the conditional volatility-targeting/breakout hysteresis, 2x cap, 10% deadband) is unchanged from `KellyRegimeV3`/`KellyRegimeV4`.

## k-grid sweep (PANEL_TRAIN only)

| k | D1 count (of 6) | verdict |
|---|---|---|
| 1.5 | 0/6 | FAILS |
| 2.0 | 0/6 | FAILS |
| 3.0 | 0/6 | FAILS |

**Frozen k = 2.0** (best D1 count on PANEL_TRAIN; ties broken toward k=2.0, the grid midpoint, as the least extreme choice).

## D1 — primary, matched-exposure drawdown, PANEL_TRAIN, spot @0.10%

**0/6 -> FAILS** (exact one-sided binomial p=1.0000)

| asset | candidate DD | matched-hold DD | dDD (matched) | 95% bootstrap CI | candidate better |
|---|---|---|---|---|---|
| BCH | 55.8% | 42.5% | +13.0pp | [-3.8, +39.4] | no |
| LTC | 62.5% | 49.5% | +13.5pp | [+2.6, +37.8] | no |
| ETC | 59.5% | 46.6% | +15.2pp | [+3.7, +39.6] | no |
| DASH | 52.1% | 30.0% | +21.9pp | [+5.0, +46.8] | no |
| LINK | 60.6% | 49.7% | +13.9pp | [-4.1, +36.7] | no |
| XTZ | 54.8% | 45.1% | +8.8pp | [-7.5, +34.0] | no |

## D2 — falsification control, CONTROL window (BTC/ETH, 2020-04..2022-12)

| asset | candidate dDD (matched) | R-57 v4 dDD | tolerance | within tolerance |
|---|---|---|---|---|
| BTC | +3.1pp | -5.6pp | +5.0pp | NO |
| ETH | +5.4pp | -11.5pp | +5.0pp | NO |

**D2 overall: FAILS** (must pass on both BTC and ETH).

## D3 — generalization check, PANEL_TEST (descriptive, not a gate)

**0/6 -> FAILS**

| asset | candidate DD | matched-hold DD | dDD (matched) | 95% bootstrap CI | candidate better |
|---|---|---|---|---|---|
| BCH | 60.3% | 29.7% | +31.5pp | [+4.7, +51.6] | no |
| LTC | 16.6% | 0.0% | +16.4pp | [+3.7, +31.8] | no |
| ETC | 17.2% | 0.0% | +16.0pp | [+0.4, +33.9] | no |
| DASH | 0.0% | 0.0% | +0.0pp | [+0.0, +0.0] | no |
| LINK | 43.2% | 8.6% | +35.1pp | [+15.7, +58.0] | no |
| XTZ | 0.0% | 0.0% | +0.0pp | [+0.0, +0.0] | no |

## D4 — 0.40% fee falsification, PANEL_TRAIN (predicted to fail)

Candidate beats `buy_and_hold` final balance in **2/6** assets -> FAILS (as predicted)

| asset | candidate final | hold final | candidate beats hold |
|---|---|---|---|
| BCH | $835 | $438 | yes |
| LTC | $572 | $1,723 | no |
| ETC | $1,130 | $3,137 | no |
| DASH | $564 | $641 | no |
| LINK | $533 | $2,386 | no |
| XTZ | $672 | $444 | yes |

## Total configurations evaluated by this branch: 94

(k-grid sweep: 3 k x 6 panel assets x 3 arms (candidate, buy_and_hold, matched hold) = 54 backtests; D2: 2 assets x 3 arms = 6; D3: 6 assets x 3 arms = 18; D4: 6 assets x 3 arms = 18. D1 reuses the frozen-k sweep cells rather than re-running them.)

## Verdict: NEGATIVE

Promotion bar (pre-registered, `r60_shared.promoted`): D1 >= 5/6 AND D2 passes on both BTC and ETH. Anything else is NEGATIVE.

Holdout consultations added by this round: 0 (no BTC/ETH 2023+ bar read).
