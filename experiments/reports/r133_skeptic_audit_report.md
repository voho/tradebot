# R-133 skeptic audit

## 1. Ablation — delete the turnover-feedback channel

The novel branch's `lambda_t` is replaced by a constant. The trailing-turnover EWM is then read by nothing: what remains is `pos += (desired - pos)/(1+lambda)` at a fixed rate — a Gârleanu-Pedersen-style smooth trading rate, the object R-64 (novel) closed on `kelly_regime_v4` with "do not re-try ... at all". The live branch's own realized mean `lambda` on this cell is 3.61, so `lam=3.61` is the matched control.

```
                           tag     cell     slice  sharpe  sharpe_v4  d_sharpe    dd  dd_v4  d_dd  fills  fills_v4  trades  trades_v4  paired      lo     hi   sig   tim  tim_v4    vol  vol_v4  n_pending  n_intervened  lam_mean  lam_frac_pos
 CONTROL lam=1.0 (no feedback) btc_spot inner-val   0.176      0.142     0.034 35.72  33.18  2.54    303       256       1         52  0.0229 -0.1063 0.1587 False  99.5    55.6 0.2680  0.2879        373           373    1.0000         1.000
 CONTROL lam=2.0 (no feedback) btc_spot inner-val   0.175      0.142     0.033 35.90  33.18  2.71    240       256       1         52  0.0224 -0.1348 0.1892 False  99.5    55.6 0.2644  0.2879        538           538    2.0000         1.000
CONTROL lam=3.61 (no feedback) btc_spot inner-val   0.227      0.142     0.085 36.07  33.18  2.89    301       256       1         52  0.0502 -0.1165 0.2265 False  99.5    55.6 0.2705  0.2879        755           755    3.6100         1.000
 CONTROL lam=6.0 (no feedback) btc_spot inner-val   0.219      0.142     0.077 36.70  33.18  3.52    326       256       1         52  0.0461 -0.1283 0.2299 False  99.5    55.6 0.2679  0.2879       1103          1103    6.0000         1.000
CONTROL lam=20.0 (no feedback) btc_spot inner-val   0.292      0.142     0.150 37.78  33.18  4.59    352       256       1         52  0.0806 -0.1350 0.3365 False 100.0    55.6 0.2736  0.2879       2982          2982   20.0000         1.000
    LIVE corridor=3.0x eta=0.5 btc_spot inner-val   0.140      0.142    -0.002 34.37  33.18  1.19    221       256      30         52 -0.0009 -0.1075 0.1028 False  75.8    55.6 0.2863  0.2879        551           340    3.6129         0.187
```

- live branch `d_sharpe` = -0.002 (221 fills, 30 round-trip episodes); matched control (`lam=3.61`, no feedback) = +0.085 (301 fills, 1 episodes). v4 itself: 256 fills, 52 episodes.
- 5 of 5 zero-feedback controls score a higher `d_sharpe` than the live branch. **That is not evidence the feedback channel is decoration, because the arms are not risk-matched** (the standing rule): every control carries +2.54 to +4.59pp more drawdown than v4 and 99.5-100.0% time in market against v4's 55.6%, at realized volatility 0.264-0.274 against v4's 0.288. Higher Sharpe bought with more exposure is an exposure statement.
- What the ablation DOES establish is structural, and it is the finding that explains this whole round. Constant-`lambda` partial adjustment never lands exactly on the target and never reaches zero, so the position is never fully closed: 1 round-trip episode(s) across two years, against v4's 52, while still paying 240-352 fills. The controls are not throttled versions of v4 — they are permanently-invested variants that trade about as often. A shrink-the-order throttle does not convert into less trading; it converts one decisive order into a stream of partial ones and removes the exits.
- And the live branch's `lambda` is not a smooth dial either. The `eta` grid in the novel branch's own report moves `eta` by 4x and changes `lam_mean` by 3% and the fill count not at all, because `lambda` saturates at its `LAMBDA_MAX` cap whenever it leaves zero (positive on only 18.7% of bars, mean 3.61, max seen = the cap). The "smooth, self-regulating control loop" as frozen is behaviourally **bang-bang**.

## 2. Does the conservative branch's best grid point survive off its own cell?

`corridor=2.0x` scored `d_sharpe = +0.137` on BTC spot inner-validation — this round's largest positive number, and the one a careless write-up would report as the result. Every other cell, same frozen configuration:

```
               tag            cell       slice  sharpe  sharpe_v4  d_sharpe    dd  dd_v4  d_dd  fills  fills_v4  trades  trades_v4  paired      lo     hi   sig  tim  tim_v4    vol  vol_v4  n_pending  n_intervened
cons corridor=2.0x        btc_spot   inner-val   0.279      0.142     0.137 31.30  33.18 -1.88    213       256      46         52  0.0794 -0.0111 0.1831 False 54.4    55.6 0.2808  0.2879       5186          4948
cons corridor=2.0x     btc_futures   inner-val   0.384      0.251     0.132 31.05  32.29 -1.24    126       143      46         52  0.0754 -0.0093 0.1867 False 54.4    55.6 0.2784  0.2820       5186          4948
cons corridor=2.0x        eth_spot   inner-val   0.440      0.500    -0.060 33.83  33.19  0.64    247       315      41         47 -0.0402 -0.1624 0.0833 False 61.4    63.2 0.3521  0.3416      10315         10049
cons corridor=2.0x    btc_spot_040   inner-val   0.010     -0.168     0.178 37.04  39.66 -2.62    213       256      46         52  0.1046 -0.0013 0.2279 False 54.4    55.6 0.2828  0.2914       5186          4948
cons corridor=2.0x btc_futures_040   inner-val   0.070     -0.075     0.146 38.99  39.80 -0.82    120       137      46         52  0.0815 -0.0060 0.1964 False 54.4    55.6 0.2687  0.2746       5186          4948
cons corridor=2.0x    eth_spot_040   inner-val   0.229      0.243    -0.014 41.64  40.03  1.61    247       315      41         47 -0.0099 -0.1353 0.1242 False 61.4    63.2 0.3545  0.3444      10315         10049
cons corridor=2.0x        btc_spot inner-train   2.008      2.030    -0.022 42.61  43.25 -0.65    414       467      66         72 -0.0505 -0.1747 0.0619 False 66.7    67.6 0.3810  0.3824      10551         10064
```

- positive `d_sharpe` on 4 of 6 inner-validation cells; bootstrap interval excludes zero on 0 of 6.

## 3. What the interventions bought — turnover accounting

Both branches intervene thousands of times. The COST axis exists to reduce realized trading, so count the interventions against realized FILLS (BTC spot, inner-validation). `turnover_ratio` is fills / v4's fills; `episodes` is `Metrics.num_trades`, a different unit, carried alongside so the two cannot be confused.

```
      branch corridor  interventions  fills  fills_v4  episodes  turnover_ratio  d_sharpe  d_dd
conservative     2.0x           4948    213       256        46           0.832     0.137 -1.88
conservative     2.5x           3158    233       256        49           0.910     0.027 -0.85
conservative     3.0x           1791    244       256        50           0.953    -0.028  0.00
conservative     3.5x            651    249       256        51           0.973     0.021 -0.00
conservative     4.0x            279    253       256        52           0.988     0.004  0.00
       novel     2.0x           1205    288       256        22           1.125    -0.088  3.82
       novel     2.5x            605    228       256        25           0.891    -0.052  4.11
       novel     3.0x            340    221       256        30           0.863    -0.002  1.19
       novel     3.5x            286    232       256        35           0.906     0.008 -0.56
       novel     4.0x             61    241       256        41           0.941    -0.021 -0.00
```

- **conservative** (defer, i.e. change order TIMING): 4948 interventions at its tightest corridor buy a turnover ratio of 0.832 — a 17% cut for ~5000 interventions. Deferral POSTPONES a rebalance; it does not cancel it, so the move executes a few bars later and the fill count barely moves. A defer-only band on a latched target is close to a no-op on cost, whatever its intervention count says.
- **novel** (shrink, i.e. change order SIZE): the turnover ratio is **non-monotone and never much below 1**, 0.863-1.125, and at the TIGHTEST corridor (2.0x) it is **1.125 — more turnover than v4, not less**. This is the round's central mechanical finding: shrinking an order does not remove it, it splits it, so a size throttle on a latched-target strategy converts one decisive rebalance into a sequence of partial ones. Turning the throttle up trades MORE. Meanwhile `d_dd` worsens monotonically as the corridor tightens (-0.00pp at 4.0x → +3.82pp at 2.0x): the mechanism buys extra drawdown and does not even deliver the cost saving it exists for.

## 4. Where the shrunk orders went — `broker.REBALANCE_DEADBAND` absorption

Section 1's controls intend hundreds of re-targets and fill almost none. `tradebot.broker` drops any same-sign adjustment worth less than `REBALANCE_DEADBAND = 5%` of max notional. A mechanism whose whole action is to SHRINK a re-target therefore shrinks its orders straight through that floor: `kelly_regime_v4`'s own strategy-level deadband is 0.10, so a move that has just cleared it, divided by `1 + lambda`, lands at 0.10/(1+lambda) — below the broker's 0.05 at any `lambda > 1`. The intent is recorded in the target column; the order never reaches the tape.

This is the same evaluability defect R-130's skeptic measured on `hedge_experts` (96.8% of intended re-targets absorbed), filed as B-43 by this round. It is confirmed here independently, on `kelly_regime_v4`:

```
                     config  intended_retargets  filled_orders  absorbed_pct
           constant lam=1.0                 333            303           9.0
           constant lam=2.0                 499            240          51.9
          constant lam=3.61                 729            301          58.7
           constant lam=6.0                1083            326          69.9
          constant lam=20.0                2895            352          87.8
   LIVE novel corridor=2.0x                1707            288          83.1
   LIVE novel corridor=3.0x                 564            221          60.8
kelly_regime_v4 (reference)                 268            256           4.5
```

Absorption scales with the shrink factor exactly as the algebra predicts, and it is NOT total: v4 loses 4.5% of its intended re-targets to the floor, a constant `lambda=1` loses 9%, and the live novel branch loses 61% at its frozen corridor and 83% at its tightest. So the deadband is not what makes this round negative — section 3 is — but it does mean the *measured* behaviour of any size-shrinking mechanism here is a blend of the mechanism and the broker's floor, in a proportion that changes with the mechanism's own parameter. **A COST-axis mechanism on this framework that acts on order SIZE cannot be cleanly attributed until `REBALANCE_DEADBAND` is addressed** (B-29). Mechanisms that act on order TIMING, like the conservative branch, are unaffected.

Configurations evaluated in this audit: **30** candidate backtests.
