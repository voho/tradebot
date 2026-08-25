# R-133 novel — online dual-ascent turnover throttle on `kelly_regime_v4`

Frozen: `TURNOVER_UPPER = 0.9091` trades/day (30-day causal EWM), `ETA = 0.5`, `LAMBDA_MAX = 20.0`. Inner-validation = 2021-01-01 → 2022-12-31.

## A2 — non-inertness gate (run before any performance number is read)

- pending rebalances shrunk by the throttle: **951** of 1599 pending-bars
- `lambda` mean 2.880, positive on 15.0% of bars, max seen 20.00 (cap 20.0)
- **A2: PASS**

## B3 — plateau over the corridor multiple (BTC spot, inner-validation)

```
                  tag     cell     slice  sharpe  sharpe_v4  d_sharpe    dd  dd_v4  d_dd  fills  fills_v4  trades  trades_v4  paired      lo     hi   sig  tim  tim_v4    vol  vol_v4  n_pending  n_intervened  lam_mean  lam_frac_pos
corridor=2.0x eta=0.5 btc_spot inner-val   0.054      0.142    -0.088 37.00  33.18  3.82    288       256      22         52 -0.0466 -0.1829 0.0881 False 84.2    55.6 0.2713  0.2879       1353          1205    7.8162        0.4056
corridor=2.5x eta=0.5 btc_spot inner-val   0.090      0.142    -0.052 37.30  33.18  4.11    228       256      25         52 -0.0305 -0.1604 0.0959 False 82.5    55.6 0.2883  0.2879        781           605    6.0651        0.3135
corridor=3.0x eta=0.5 btc_spot inner-val   0.140      0.142    -0.002 34.37  33.18  1.19    221       256      30         52 -0.0009 -0.1075 0.1028 False 75.8    55.6 0.2863  0.2879        551           340    3.6129        0.1870
corridor=3.5x eta=0.5 btc_spot inner-val   0.150      0.142     0.008 32.62  33.18 -0.56    232       256      35         52  0.0055 -0.0888 0.0979 False 67.6    55.6 0.2856  0.2879        511           286    2.9008        0.1479
corridor=4.0x eta=0.5 btc_spot inner-val   0.121      0.142    -0.021 33.18  33.18 -0.00    241       256      41         52 -0.0126 -0.1134 0.0717 False 64.1    55.6 0.2892  0.2879        314            61    0.5898        0.0324
```

## B3 — plateau over the dual-ascent step size `eta` (BTC spot, inner-validation)

```
                   tag     cell     slice  sharpe  sharpe_v4  d_sharpe    dd  dd_v4  d_dd  fills  fills_v4  trades  trades_v4  paired      lo     hi   sig  tim  tim_v4    vol  vol_v4  n_pending  n_intervened  lam_mean  lam_frac_pos
corridor=3.0x eta=0.25 btc_spot inner-val   0.146      0.142     0.004 34.17  33.18  0.99    225       256      30         52  0.0027 -0.0943 0.1028 False 75.8    55.6 0.2855  0.2879        553           342    3.6379        0.1905
 corridor=3.0x eta=1.0 btc_spot inner-val   0.121      0.142    -0.021 34.16  33.18  0.98    223       256      30         52 -0.0111 -0.1176 0.0934 False 75.8    55.6 0.2842  0.2879        552           341    3.5331        0.1812
```

**B3 (plateau majority positive over the corridor grid): FAIL** — 1 of 5 grid points beat v4.

## B1 (signal, both markets) / B4 (ETH replication) / B5 (0.40% taker)

```
                  tag            cell       slice  sharpe  sharpe_v4  d_sharpe    dd  dd_v4  d_dd  fills  fills_v4  trades  trades_v4  paired      lo     hi   sig  tim  tim_v4    vol  vol_v4  n_pending  n_intervened  lam_mean  lam_frac_pos
corridor=3.0x eta=0.5        btc_spot   inner-val   0.140      0.142    -0.002 34.37  33.18  1.19    221       256      30         52 -0.0009 -0.1075 0.1028 False 75.8    55.6 0.2863  0.2879        551           340    3.6129        0.1870
corridor=3.0x eta=0.5     btc_futures   inner-val   0.040      0.251    -0.211 38.37  32.29  6.08    101       143      30         52 -0.1236 -0.3054 0.0245 False 75.8    55.6 0.2871  0.2820        551           340    3.6129        0.1870
corridor=3.0x eta=0.5        eth_spot   inner-val   0.389      0.500    -0.111 35.54  33.19  2.35    275       315      31         47 -0.0766 -0.1760 0.0079 False 73.2    63.2 0.3378  0.3416        443           175    2.0573        0.1182
corridor=3.0x eta=0.5    btc_spot_040   inner-val  -0.109     -0.168     0.059 39.63  39.66 -0.03    221       256      30         52  0.0351 -0.0775 0.1585 False 75.8    55.6 0.2894  0.2914        551           340    3.6129        0.1870
corridor=3.0x eta=0.5 btc_futures_040   inner-val  -0.233     -0.075    -0.158 45.20  39.80  5.39     95       137      30         52 -0.0929 -0.2770 0.0634 False 75.8    55.6 0.2799  0.2746        551           340    3.6129        0.1870
corridor=3.0x eta=0.5    eth_spot_040   inner-val   0.168      0.243    -0.075 40.70  40.03  0.67    275       315      31         47 -0.0487 -0.1443 0.0367 False 73.2    63.2 0.3405  0.3444        443           175    2.0573        0.1182
corridor=3.0x eta=0.5        btc_spot inner-train   2.010      2.030    -0.019 46.45  43.25  3.19    439       467      62         72 -0.0266 -0.1598 0.0938 False 73.1    67.6 0.3802  0.3824       1035           586    2.2709        0.1179
```

- **B1**: FAIL
- **B4** (pre-registered falsification test): FAIL (BTC -0.002, ETH -0.111)
- **B5**: FAIL
- **B2** (diagnostic only): BTC spot +1.19pp, BTC futures +6.08pp, ETH spot +2.35pp

## Branch diagnostic — `lambda`'s trajectory through the six stress episodes

Baseline over the whole BTC frame: `lambda` mean 2.880, positive on 15.0% of bars.

```
                            episode       date  lambda_mean_pm3d  lambda_max_pm3d  frac_throttled_pm3d
2018 bear onset (post-Dec-2017 top) 2018-01-17             20.00           20.000                1.000
    2018 bear bottom / capitulation 2018-12-15              0.00            0.000                0.000
                2020-03 COVID crash 2020-03-12              0.00            0.000                0.000
 2021-11 top / 2022 bear transition 2021-11-10             20.00           20.000                1.000
        2022-05 Terra/Luna collapse 2022-05-09              0.00            0.000                0.000
               2022-11 FTX collapse 2022-11-08              2.52            9.825                0.345
```

## Branch verdict: **NEGATIVE**

A2=True, B1=False, B3=False, B4=False, B5=False.

Configurations evaluated on this branch: **14** candidate backtests (7 distinct configurations).
