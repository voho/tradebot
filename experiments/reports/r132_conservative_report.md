# R-132 conservative — turnover-corridor deferral band on `kelly_regime_v4`

Frozen corridor edge: `TURNOVER_UPPER = 3.0 x 0.3030 = 0.9091` trades/day, 30-day causal EWM. Inner-validation = 2021-01-01 → 2022-12-31.

## A2 — non-inertness gate (run before any performance number is read)

- deferrals fired (BTC, full train+inner-val frame): **2425**
- overrides — full de-risking exit: 9; move > 2.0 x corridor: 0
- **A2: PASS** — the corridor is reached and the mechanism does change v4's behaviour.

Note on the size override as frozen: `|desired - current| > OVERRIDE_MULT * TURNOVER_UPPER` compares a position magnitude (units of equity notional, range 0-2 here) against a *rate* (trades/day). The frozen text is implemented literally; the resulting threshold is 1.818 in position units, so it fires 0 times and the effective safety valve is the `desired == 0` full-exit clause. Recorded, not silently repaired.

## B3 — plateau over the corridor multiple (BTC spot, inner-validation)

```
          tag     cell     slice  sharpe  sharpe_v4  d_sharpe    dd  dd_v4  d_dd  fills  fills_v4  trades  trades_v4  paired      lo     hi   sig  tim  tim_v4    vol  vol_v4  n_pending  n_intervened
corridor=2.0x btc_spot inner-val   0.279      0.142     0.137 31.30  33.18 -1.88    213       256      46         52  0.0794 -0.0111 0.1831 False 54.4    55.6 0.2808  0.2879       5186          4948
corridor=2.5x btc_spot inner-val   0.169      0.142     0.027 32.34  33.18 -0.85    233       256      49         52  0.0162 -0.0180 0.0549 False 54.8    55.6 0.2859  0.2879       3414          3158
corridor=3.0x btc_spot inner-val   0.114      0.142    -0.028 33.18  33.18  0.00    244       256      50         52 -0.0162 -0.0581 0.0094 False 55.1    55.6 0.2884  0.2879       2058          1791
corridor=3.5x btc_spot inner-val   0.163      0.142     0.021 33.18  33.18 -0.00    249       256      51         52  0.0120 -0.0000 0.0312 False 55.4    55.6 0.2879  0.2879        923           651
corridor=4.0x btc_spot inner-val   0.146      0.142     0.004 33.18  33.18  0.00    253       256      52         52  0.0022 -0.0000 0.0067 False 55.6    55.6 0.2878  0.2879        555           279
```

**B3 (plateau majority positive): PASS** — 4 of 5 grid points beat v4 on Sharpe.

## B1 (signal, both markets) / B4 (ETH replication) / B5 (0.40% taker)

```
          tag            cell       slice  sharpe  sharpe_v4  d_sharpe    dd  dd_v4  d_dd  fills  fills_v4  trades  trades_v4  paired      lo     hi   sig  tim  tim_v4    vol  vol_v4  n_pending  n_intervened
corridor=3.0x        btc_spot   inner-val   0.114      0.142    -0.028 33.18  33.18  0.00    244       256      50         52 -0.0162 -0.0581 0.0094 False 55.1    55.6 0.2884  0.2879       2058          1791
corridor=3.0x     btc_futures   inner-val   0.263      0.251     0.011 32.29  32.29  0.00    137       143      50         52  0.0064 -0.0016 0.0175 False 55.1    55.6 0.2818  0.2820       2058          1791
corridor=3.0x        eth_spot   inner-val   0.537      0.500     0.037 30.89  33.19 -2.30    293       315      43         47  0.0243 -0.0051 0.0560 False 62.9    63.2 0.3385  0.3416       1818          1506
corridor=3.0x    btc_spot_040   inner-val  -0.190     -0.168    -0.022 40.41  39.66  0.75    244       256      50         52 -0.0125 -0.0543 0.0168 False 55.1    55.6 0.2919  0.2914       2058          1791
corridor=3.0x btc_futures_040   inner-val  -0.059     -0.075     0.016 39.80  39.80  0.00    131       137      50         52  0.0091 -0.0015 0.0238 False 55.1    55.6 0.2743  0.2746       2058          1791
corridor=3.0x    eth_spot_040   inner-val   0.295      0.243     0.051 37.46  40.03 -2.56    293       315      43         47  0.0350  0.0029 0.0734  True 62.9    63.2 0.3410  0.3444       1818          1506
corridor=3.0x        btc_spot inner-train   2.028      2.030    -0.002 43.60  43.25  0.34    462       467      72         72 -0.0026 -0.0182 0.0104 False 67.6    67.6 0.3825  0.3824       1197           659
```

- **B1** (beats v4 on both BTC markets, inner-val): FAIL
- **B4** (pre-registered falsification — sign of `d_sharpe` replicates on ETH, and is positive): FAIL (BTC -0.028, ETH +0.037)
- **B5** (survives a 0.40% taker tier on both BTC markets): FAIL
- **B2** (drawdown, diagnostic only, never gates): BTC spot +0.00pp, BTC futures +0.00pp, ETH spot -2.30pp

## Branch diagnostic — deferral behaviour at the six stress episodes

The failure mode named before any code ran: a throttle that damps trading when turnover spikes is damping it exactly when a regime transition — where L-01/R-62 say v4's edge lives — drives a burst of rebalances.

```
                            episode       date  deferral_bars_within_3d
2018 bear onset (post-Dec-2017 top) 2018-01-17                        0
    2018 bear bottom / capitulation 2018-12-15                        0
                2020-03 COVID crash 2020-03-12                        0
 2021-11 top / 2022 bear transition 2021-11-10                        0
        2022-05 Terra/Luna collapse 2022-05-09                        0
               2022-11 FTX collapse 2022-11-08                        0
```

## Branch verdict: **NEGATIVE**

Decision rule as frozen: A2 AND B1 (both markets) AND B3 (plateau majority) AND B4 (full, both markets) AND B5 — all must pass. A2=True, B1=False, B3=True, B4=False, B5=False.

Configurations evaluated on this branch: **12** candidate backtests (5 distinct configurations).
