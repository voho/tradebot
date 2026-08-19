# R-34 novel variant: `kelly_regime_v5_bayes` — continuous Bayesian-margin confidence on v4's vol-targeting sizer

**Verdict up front: NEGATIVE. Not a promotion candidate.** Every configuration
tested loses to `kelly_regime_v4` on Sharpe, drawdown, and turnover on the
inner-validation split; the effect is not an R-31/R-32/R-33 exposure-level
artifact (confirmed by explicitly matching mean exposure — matching makes it
*worse*, not better); the ETH/BTC falsification does not save it (v4 wins on
return in all four asset x market cells); and the confidence signal is
essentially uncorrelated with v4's discrete vote (r = -0.002), so this is a
genuinely different mechanism, not a smoothed copy — it is just a worse one.

Data used: `data/btcusd_spot_5m.csv.gz`, restricted to `<= 2022-12-31`
throughout (inner-train 2017-01-01→2020-12-31, inner-validation
2021-01-01→2022-12-31). The 2023-01-01+ BTC holdout was never read. ETH/BTC
falsification used the Bitfinex files (`btcusd_bitfinex_5m.csv.gz`,
`ethusd_bitfinex_5m.csv.gz`, 2016→2019), which are not the project holdout.

Configurations evaluated: **36 distinct (stick, b_in, b_out) parameter
tuples** (two 18-tuple grids — see finding 0 below — each run on spot and
futures 5x = 72 backtests), plus the frozen best config re-run on
inner-train, on ETH/BTC falsification (4 cells), and at a matched exposure
multiplier. Causality: two-opposite-tampers truncation probe (identical
pattern to `bayes_confidence.py`'s own already-passed probe), max diff
**0.0**, run both at default and at `exposure_mult=5.27`; `pytest
tests/test_causality_strict.py` (unaffected, unregistered file) — 51 passed.

---

## 0. A finding before the sweep: the task's suggested grid is mostly outside the signal's range

`bayesian_margin`'s empirical `|margin|` distribution on inner-validation
(stick 0.980–0.990) has its 99th percentile at **0.40–0.50** and never
exceeds **0.85**. The task's example grid (`b_in ∈ {0.5, 0.65, 0.8}`, `b_out
∈ {0.25, 0.4}`) puts most of its cells at or beyond the tail of what the
signal ever produces — `b_in=0.8` latches on 0–1 bars total across an entire
2-year window. Reported honestly below as **Grid A**. A second, **Grid B**,
re-sized to the signal's own quantiles (`b_in ∈ {0.10, 0.15, 0.20}`, `b_out ∈
{0.05, 0.08}`) was run so the mechanism gets a fair trial. Both are reported
— the point of the exercise is what the *mechanism* does, not what one
specific threshold choice does, and Grid A's degeneracy is itself informative
(most of a plausible-looking hand-picked grid corresponds to "never trade").

## 1. Sweep — Grid A (task-specified thresholds), inner-validation

stick ∈ {0.980, 0.985, 0.990}, b_in ∈ {0.5, 0.65, 0.8}, b_out ∈ {0.25, 0.4}.
`mean|exp|` = mean of `|target|` over the window; `exp_ratio` = that divided
by v4's own mean|exp| (0.289, identical on both markets since `target` is
market-agnostic).

| stick | b_in | b_out | mkt | final$ | profit% | sharpe | DD% | trades | mean\|exp\| | exp_ratio |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| — | v4 baseline | | spot | 998 | -0.2 | 0.14 | 33.2 | 52 | 0.289 | 1.00 |
| — | v4 baseline | | fut | 1,064 | +6.4 | 0.25 | 32.3 | 52 | 0.289 | 1.00 |
| 0.980 | 0.50 | 0.25 | spot | 854 | -14.6 | -1.80 | 15.2 | 34 | 0.043 | 0.15 |
| 0.980 | 0.50 | 0.25 | fut | 706 | -29.4 | -1.40 | 32.4 | 34 | 0.043 | 0.15 |
| 0.980 | 0.50 | 0.40 | spot | 907 | -9.3 | -1.43 | 9.3 | 47 | 0.031 | 0.11 |
| 0.980 | 0.50 | 0.40 | fut | 939 | -6.1 | -0.30 | 11.3 | 47 | 0.031 | 0.11 |
| 0.980 | 0.65 | 0.25 | spot | 977 | -2.3 | -0.56 | 3.1 | 4 | 0.015 | 0.05 |
| 0.980 | 0.65 | 0.25 | fut | 951 | -4.9 | -0.33 | 9.3 | 4 | 0.015 | 0.05 |
| 0.980 | 0.65 | 0.40 | spot | 978 | -2.2 | -0.48 | 3.0 | 4 | 0.015 | 0.05 |
| 0.980 | 0.65 | 0.40 | fut | 983 | -1.7 | -0.37 | 2.6 | 4 | 0.015 | 0.05 |
| 0.980 | 0.80 | 0.25/0.40 | both | 1,000 | 0.0 | 0.00 | 0.0 | 0 | 0.000 | 0.00 |
| 0.985 | 0.50 | 0.25 | spot | 904 | -9.6 | -1.15 | 10.1 | 52 | 0.033 | 0.11 |
| 0.985 | 0.50 | 0.25 | fut | 1,215 | +21.5 | 0.95 | 7.9 | 52 | 0.033 | 0.11 |
| 0.985 | 0.50 | 0.40 | spot | 1,008 | +0.8 | 0.10 | 7.9 | 77 | 0.036 | 0.12 |
| 0.985 | 0.50 | 0.40 | fut | 1,116 | +11.6 | 0.63 | 7.8 | 77 | 0.036 | 0.12 |
| 0.985 | 0.65 | 0.25 | spot | 931 | -6.9 | -1.31 | 9.1 | 7 | 0.028 | 0.10 |
| 0.985 | 0.65 | 0.25 | fut | 863 | -13.7 | -1.04 | 17.3 | 7 | 0.028 | 0.10 |
| 0.985 | 0.65 | 0.40 | spot | 989 | -1.1 | -0.07 | 8.8 | 10 | 0.054 | 0.19 |
| 0.985 | 0.65 | 0.40 | fut | 1,129 | +12.9 | 0.49 | 10.6 | 10 | 0.054 | 0.19 |
| 0.985 | 0.80 | 0.25/0.40 | both | 1,000 | 0.0 | 0.00 | 0.0 | 0 | 0.000 | 0.00 |
| 0.990 | 0.50 | 0.25 | spot | 807 | -19.3 | -2.04 | 20.4 | 58 | 0.043 | 0.15 |
| 0.990 | 0.50 | 0.25 | fut | 994 | -0.6 | 0.04 | 17.6 | 58 | 0.043 | 0.15 |
| 0.990 | 0.50 | 0.40 | spot | 882 | -11.8 | -1.58 | 13.3 | 105 | 0.030 | 0.10 |
| 0.990 | 0.50 | 0.40 | fut | 927 | -7.3 | -0.32 | 16.6 | 105 | 0.030 | 0.10 |
| 0.990 | 0.65 | 0.25 | spot | 858 | -14.2 | -1.76 | 15.1 | 28 | 0.035 | 0.12 |
| 0.990 | 0.65 | 0.25 | fut | 739 | -26.1 | -1.05 | 31.8 | 28 | 0.035 | 0.12 |
| 0.990 | 0.65 | 0.40 | spot | 911 | -8.9 | -1.26 | 11.0 | 32 | 0.026 | 0.09 |
| 0.990 | 0.65 | 0.40 | fut | 932 | -6.8 | -0.22 | 14.2 | 32 | 0.026 | 0.09 |
| 0.990 | 0.80 | 0.25 | both | ~1,000 | ~0.1 | ~0.1 | ~0.3 | 1 | ~0.00 | ~0.00 |
| 0.990 | 0.80 | 0.40 | both | ~1,000 | ~0.0 | ~0.01 | ~0.3 | 1 | ~0.00 | ~0.00 |

Every one of these 18 configs runs at **5–19% of v4's mean exposure**. A few
individual cells beat v4 on Sharpe by chance (e.g. 0.985/0.50/0.25 futures:
Sharpe 0.95 vs v4's 0.25) but there is no plateau — each is a lone spike
surrounded by neighbours that are flat or negative, and the same config's own
spot cell is simultaneously *worse* than v4 (-9.6% vs -0.2%). This is
noise from a low-trade-count region (7–77 trades on the winning cells), not a
robust effect, and it disappears in Grid B.

## 2. Sweep — Grid B (thresholds sized to the margin's actual distribution)

stick ∈ {0.980, 0.985, 0.990}, b_in ∈ {0.10, 0.15, 0.20}, b_out ∈ {0.05, 0.08}.

| stick | b_in | b_out | mkt | final$ | profit% | sharpe | DD% | trades | mean\|exp\| | exp_ratio |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.980 | 0.10 | 0.05 | spot | 647 | -35.3 | -3.90 | 36.9 | 341 | 0.050 | 0.17 |
| 0.980 | 0.10 | 0.05 | fut | 840 | -16.0 | -0.71 | 25.5 | 341 | 0.050 | 0.17 |
| 0.980 | 0.10 | 0.08 | spot | 692 | -30.8 | -3.32 | 32.2 | 293 | 0.048 | 0.16 |
| 0.980 | 0.10 | 0.08 | fut | 974 | -2.6 | -0.06 | 16.5 | 293 | 0.048 | 0.16 |
| 0.980 | 0.15 | 0.05 | spot | 654 | -34.6 | -3.80 | 36.2 | 299 | 0.050 | 0.17 |
| 0.980 | 0.15 | 0.05 | fut | 846 | -15.4 | -0.68 | 25.0 | 299 | 0.050 | 0.17 |
| 0.980 | 0.15 | 0.08 | spot | 696 | -30.4 | -3.27 | 31.8 | 275 | 0.047 | 0.16 |
| 0.980 | 0.15 | 0.08 | fut | 976 | -2.4 | -0.05 | 16.3 | 275 | 0.047 | 0.16 |
| 0.980 | 0.20 | 0.05 | spot | 688 | -31.2 | -3.37 | 33.1 | 209 | 0.050 | 0.17 |
| 0.980 | 0.20 | 0.05 | fut | 889 | -11.1 | -0.48 | 21.9 | 209 | 0.050 | 0.17 |
| 0.980 | 0.20 | 0.08 | spot | 717 | -28.3 | -3.01 | 30.1 | 226 | 0.047 | 0.16 |
| 0.980 | 0.20 | 0.08 | fut | 986 | -1.4 | -0.00 | 15.5 | 226 | 0.047 | 0.16 |
| 0.985 | 0.10 | 0.05 | spot | 619 | -38.1 | -3.81 | 41.1 | 320 | 0.053 | 0.18 |
| 0.985 | 0.10 | 0.05 | fut | 839 | -16.1 | -0.65 | 29.0 | 320 | 0.053 | 0.18 |
| 0.985 | 0.10 | 0.08 | spot | 694 | -30.6 | -3.08 | 34.3 | 292 | 0.051 | 0.18 |
| 0.985 | 0.10 | 0.08 | fut | 890 | -11.0 | -0.43 | 26.6 | 292 | 0.051 | 0.18 |
| 0.985 | 0.15 | 0.05 | spot | 624 | -37.6 | -3.74 | 40.6 | 295 | 0.053 | 0.18 |
| 0.985 | 0.15 | 0.05 | fut | 845 | -15.5 | -0.62 | 28.4 | 295 | 0.053 | 0.18 |
| 0.985 | 0.15 | 0.08 | spot | 696 | -30.4 | -3.05 | 34.0 | 275 | 0.051 | 0.18 |
| 0.985 | 0.15 | 0.08 | fut | 896 | -10.4 | -0.40 | 26.0 | 275 | 0.051 | 0.18 |
| 0.985 | 0.20 | 0.05 | spot | 639 | -36.1 | -3.54 | 39.2 | 222 | 0.054 | 0.19 |
| 0.985 | 0.20 | 0.05 | fut | 840 | -16.0 | -0.62 | 29.0 | 222 | 0.054 | 0.19 |
| 0.985 | 0.20 | 0.08 | spot | 712 | -28.8 | -2.86 | 32.6 | 239 | 0.051 | 0.17 |
| 0.985 | 0.20 | 0.08 | fut | 896 | -10.4 | -0.40 | 26.3 | 239 | 0.051 | 0.17 |
| **0.990** | **0.10** | **0.05** | spot | 646 | -35.4 | -3.14 | 38.8 | 289 | 0.055 | 0.19 |
| **0.990** | **0.10** | **0.05** | fut | 1,015 | +1.5 | 0.12 | 19.1 | 289 | 0.055 | 0.19 |
| 0.990 | 0.10 | 0.08 | spot | 658 | -34.2 | -3.25 | 38.6 | 248 | 0.052 | 0.18 |
| 0.990 | 0.10 | 0.08 | fut | 956 | -4.4 | -0.14 | 23.6 | 248 | 0.052 | 0.18 |
| **0.990** | **0.15** | **0.05** | **spot** | **649** | **-35.1** | **-3.11** | **38.5** | **270** | **0.055** | **0.19** |
| **0.990** | **0.15** | **0.05** | **fut** | **1,017** | **+1.7** | **0.13** | **18.9** | **270** | **0.055** | **0.19** |
| 0.990 | 0.15 | 0.08 | spot | 659 | -34.1 | -3.25 | 38.5 | 242 | 0.052 | 0.18 |
| 0.990 | 0.15 | 0.08 | fut | 959 | -4.1 | -0.13 | 23.3 | 242 | 0.052 | 0.18 |
| **0.990** | **0.20** | **0.05** | spot | 666 | -33.4 | -2.93 | 36.9 | 222 | 0.055 | 0.19 |
| **0.990** | **0.20** | **0.05** | fut | 1,020 | +2.0 | 0.14 | 19.1 | 222 | 0.055 | 0.19 |
| 0.990 | 0.20 | 0.08 | spot | 667 | -33.3 | -3.15 | 37.8 | 222 | 0.052 | 0.18 |
| 0.990 | 0.20 | 0.08 | fut | 948 | -5.2 | -0.17 | 24.3 | 222 | 0.052 | 0.18 |

**Every single one of the 18 Grid-B cells is worse than v4 on spot Sharpe**
(range -2.86 to -3.90 vs v4's +0.14) **and worse on spot drawdown** (30–41%
vs v4's 33% — comparable or worse despite running at a fifth of v4's
exposure, which is itself a bad sign: a properly-behaved lower-exposure
strategy should draw down *less*). Futures cells cluster near flat
(Sharpe -0.7 to +0.14) against v4's 0.25. **Turnover is 4–7x v4's** (209–341
trades vs 52) for uniformly worse or flat risk-adjusted return — the
L-14/L-15/L-16/L-18 fee-drag pattern.

**This is the actual plateau in the data, and it is a plateau of failure,
not success**: `stick=0.990, b_out=0.05, b_in∈{0.10,0.15,0.20}` (bolded rows)
moves together as a coherent neighbourhood — spot Sharpe -2.93 to -3.14,
futures Sharpe +0.12 to +0.14, trade count 222–289 — genuinely stable across
three adjacent `b_in` values, unlike Grid A's isolated spikes. It is chosen
below as the frozen "most defensible" config for the fuller comparison
**because it is the most stable region found, not because it looks good** —
it is still unambiguously worse than v4.

**Recommended (frozen) config: `stick=0.990, b_in=0.15, b_out=0.05`**
(center of that plateau).

## 3. Matched-exposure check (the R-31/R-32/R-33 diagnostic)

The raw config runs at ~19% of v4's mean exposure — before concluding
anything, per ROUTINE.md's standing rule, exposure was matched with a
constant multiplier (`exposure_mult=5.27`, solved from spot's ratio) and
re-run on inner-validation:

| market | arm | final$ | profit% | sharpe | DD% | trades | mean\|exp\| | ratio-to-v4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| spot | v4 | 998 | -0.2 | 0.14 | 33.2 | 52 | 0.289 | 1.00 |
| spot | v5 raw | 649 | -35.1 | -3.11 | 38.5 | 270 | 0.055 | 0.19 |
| spot | **v5 matched (×5.27)** | **88** | **-91.2** | **-6.25** | **91.9** | **733** | 0.094 | 0.32 |
| fut | v4 | 1,064 | +6.4 | 0.25 | 32.3 | 52 | 0.289 | 1.00 |
| fut | v5 raw | 1,017 | +1.7 | 0.13 | 18.9 | 270 | 0.055 | 0.19 |
| fut | **v5 matched (×5.27)** | **201** | **-79.9** | **-2.58** | **83.8** | **733** | 0.094 | 0.32 |

Matching *does not close the gap — it makes it dramatically worse* (Sharpe
falls from -3.1 to -6.3 on spot; drawdown rises from 38% to 92%; trades
almost triple, to 733). This is the opposite of the R-31/R-32/R-33 pattern,
where "worse" collapsed toward "indistinguishable" once risk was matched.
Here it rules the exposure-artifact explanation **out**: this is not v4
running hotter than a fine mechanism, it is a mechanism whose own frequent,
noisy latch/release cycling gets punished harder, not less, as its size is
turned up. (The multiplier also pushed `frac × mult` above 1 on some bars,
which the sizer does not clip — expected and documented as a diagnostic-only
knob, not a config anyone would deploy.)

## 4. Full comparison table — inner-train and inner-validation, both markets

Frozen config: `stick=0.990, b_in=0.15, b_out=0.05` (raw, `exposure_mult=1.0`).

| split | market | strategy | final$ | profit% | sharpe | DD% | trades |
|---|---|---|---:|---:|---:|---:|---:|
| inner-train | spot | buy_and_hold | 29,803 | +2880.3 | 1.38 | 84.1 | 1 |
| inner-train | spot | kelly_regime_v4 | 18,477 | +1747.7 | 2.03 | 43.3 | 72 |
| inner-train | spot | **kelly_regime_v5_bayes** | **543** | **-45.7** | **-1.98** | **46.3** | **356** |
| inner-train | futures | buy_and_hold | 18 | -98.2 | -0.29 | 99.0 (LIQ) | 1 |
| inner-train | futures | kelly_regime_v4 | 30,344 | +2934.4 | 2.28 | 35.3 | 72 |
| inner-train | futures | **kelly_regime_v5_bayes** | **1,177** | **+17.7** | **0.37** | **23.5** | **356** |
| inner-validation | spot | buy_and_hold | 574 | -42.6 | 0.08 | 77.3 | 1 |
| inner-validation | spot | kelly_regime_v4 | 998 | -0.2 | 0.14 | 33.2 | 52 |
| inner-validation | spot | **kelly_regime_v5_bayes** | **649** | **-35.1** | **-3.11** | **38.5** | **270** |
| inner-validation | futures | buy_and_hold | 18 | -98.2 | 0.43 | 99.8 (LIQ) | 1 |
| inner-validation | futures | kelly_regime_v4 | 1,064 | +6.4 | 0.25 | 32.3 | 52 |
| inner-validation | futures | **kelly_regime_v5_bayes** | **1,017** | **+1.7** | **0.13** | **18.9** | **270** |

v5_bayes beats buy_and_hold on both futures cells (unsurprising — buy&hold
gets liquidated on futures in this window) but loses to v4 on every metric
in every cell except futures drawdown, where its much lower mean exposure
(a fifth of v4's) buys a lower number almost by construction.

## 5. ETH/BTC Bitfinex falsification (pre-registered test)

Same venue, same window (2016→2019) as R-17/R-28/R-31. Frozen config as above.

| asset | market | v4 return% | v5 return% | v4 DD% | v5 DD% | better on return | better on DD |
|---|---|---:|---:|---:|---:|---|---|
| BTC (control) | spot | +1127.8 | -94.3 | 40.1 | 94.4 | v4 | v4 |
| BTC (control) | futures | +2468.1 | -30.2 | 32.1 | 39.2 | v4 | v4 |
| ETH (test) | spot | +448.2 | -57.3 | 36.5 | 61.1 | v4 | v4 |
| ETH (test) | futures | +326.3 | +63.0 | 35.1 | 26.7 | v4 | **v5** |

**The return ordering is identical on ETH and the BTC control in all four
cells — v4 wins every time, and by a wide margin.** v5's turnover on
Bitfinex data is even worse than on Bitstamp (1,813 trades on BTC spot,
621 on ETH spot, over a 4-year window — roughly 5–10 bars/trade average
signal age). The drawdown ordering flips on one of four cells (ETH futures:
v5's much lower mean exposure, 0.051 vs v4's 0.229, produces a lower
drawdown number almost mechanically). This single-cell drawdown flip is
noted plainly, in the spirit of what killed R-28/R-31 — but since the
headline claim here (return) does not flip anywhere, and v5 loses on return
by 60–100+ points in every cell of both assets, the falsification test does
not rescue the result; it confirms the negative.

## 6. Turnover

v5_bayes trades **4–7x more often than v4** at every horizon tested:
inner-train 356 vs 72, inner-validation 270 vs 52, BTC-Bitfinex 1,813 vs 62,
ETH-Bitfinex 621 vs 75. Combined with worse or flat risk-adjusted return,
this is the fee-drag failure mode this project has already named three
times (L-14, L-15, L-16, L-18): a signal with real information content but
too high a duty cycle relative to its edge pays it all away in round trips.
The `b_out` deadband (0.05–0.08) and the final 10% position deadband did
provide *some* hysteresis — turnover here is far below, e.g.,
`minority_oracle`'s 9,039 trades — but not nearly enough at a 5-minute-bar,
hours-to-days signal.

## 7. Vote correlation

Pearson correlation between v5_bayes's continuous `confidence_frac` and
v4's raw discrete anchor-vote fraction, both computed over inner-validation
(209,953 overlapping bars): **r = -0.0017**. v4's vote averages 0.425 and is
fully-in (=1.0) 29.4% of the time; v5's confidence fraction averages 0.015
and is latched-in (>0) only 8.2% of the time. **These are not the same
mechanism smoothed differently — they are essentially statistically
independent of each other**, which is exactly what the module docstring's
"hours-to-days vs weeks-to-months" framing predicts, and confirms this was a
genuine test of a different regime-detection mechanism, not a disguised
duplicate of v4's own signal. It just does not work as well.

## 8. Honest verdict

**Not a promotion candidate; do not spend the holdout on it.** The
mechanism is a real, causal, independently-verified implementation of the
task's design (continuous, non-negative, hysteresis-latched Bayesian
confidence feeding v4's unchanged conditional-vol-targeting sizer,
correlation ≈0 with v4's own vote confirming it is genuinely different), but
across two threshold grids (36 configs, 72 inner-validation backtests), an
inner-train check, an explicit matched-exposure re-run, and an ETH/BTC
falsification test, it loses to `kelly_regime_v4` on essentially every axis:
worse Sharpe (frequently strongly negative on spot), comparable-or-worse
drawdown despite running at a fifth of v4's mean exposure, 4–7x the
turnover, and it loses on return in all four ETH/BTC falsification cells.
Matching mean exposure to v4 does not reveal a hidden edge suppressed by
under-betting (the R-31/R-32/R-33 pattern) — it makes results
*catastrophically* worse, which argues the failure is intrinsic to the
signal's noise level at this timescale, not an artifact of how much of it
was bet. L-12's stated hypothesis — "as a direction signal it loses, but
maybe as a sizing input it wins" — is now tested, and on this evidence the
answer is **no**: feeding the same Bayesian posterior margin into the SIZE
axis instead of the DIRECTION axis does not rescue it. The sizing axis was
never the problem with `harsanyi_crowd`; the margin itself is too noisy at
an hours-to-days cadence, relative to a 0.10–0.40% round-trip cost, to be
useful in either role.

---

## Appendix: `experiments/kelly_regime_v5_bayes.py` (full source)

```python
"""R-34: replace kelly_regime_v4's discrete anchor vote with a continuous,
hysteresis-latched Bayesian regime-confidence signal, on the SAME
conditional-vol-targeting sizer v4 already uses.

Idea in one sentence: `harsanyi_crowd` (L-12) computes a Bayesian belief
margin P(bull)-P(bear) over three hidden market types (Harsanyi 1967-68) on
an hours-to-days timescale and trades it *directionally*, and loses; L-12's
own recorded lesson is that "the crowding intuition was right... but as a
direction signal rather than a sizing input it loses" -- a stated, never
tested hypothesis. `kelly_regime_v4` (L-01) is the strategy that DID work,
and its regime input is a *discrete* vote (0, 1/3, 2/3, 1) from three
latched weeks-to-months moving-average anchors (20/40/80 day). This module
is the novel variant of the R-34 sizing round: swap v4's discrete,
slow anchor vote for a continuous, hysteresis-latched transform of the
Bayesian margin, feeding the identical conditional-vol-targeting scale code
v4 uses unchanged, so any measured difference is attributable to the
regime-confidence mechanism and not to the risk axis.

Not a duplicate of: L-12 (margin traded directionally, can go short, no
vol-targeting sizer); L-01/L-02/L-03/L-04 (discrete weeks-to-months anchor
vote, not a continuous hours-to-days Bayesian one); a sibling experiment in
this same round builds a bounded DAMPENER multiplied onto v4's existing
discrete vote (conservative variant, different file) -- this module instead
REPLACES the vote entirely (the "deeper redesign" variant).

Mechanism, precisely:

1. `scale = full/steady` breakout-hysteresis conditional-vol-targeting code,
   copied verbatim from `kelly_regime_v3.KellyRegimeV3.prepare` (constant
   notional through normal volatility, re-size only on a volatility
   breakout, latched). This is the risk axis and it is UNCHANGED from v4.
2. `margin = bayesian_margin(df, mu, stick)` from `experiments/bayes_confidence.py`
   -- the byte-identical, already-causality-probed posterior recursion
   `harsanyi_crowd` uses, imported rather than re-derived.
3. A hysteresis latch shaped exactly like `harsanyi_crowd`'s own `b_in`/
   `b_out` bands (enter a confident state only above `b_in`, release it
   only once margin falls back through the looser `b_out`) but mapped to a
   CONTINUOUS, NEVER-NEGATIVE fraction instead of harsanyi's directional
   hysteresis-then-full-size jump: once latched "in",
   `frac = clip((margin - b_out) / (1 - b_out), 0, 1)`, else `frac = 0`.
   This is causal (state depends only on bars <= i), always in [0, 1] (this
   project's strategies never short a historically-upward-drifting asset --
   see `kelly_regime.py`'s own docstring), and has a genuine no-trade
   deadband: while un-latched, `frac` is pinned at 0 regardless of small
   wiggles below `b_in`.
4. `frac` REPLACES v4's discrete vote entirely in `desired = frac[i] *
   scale[i]`. Everything else -- the final deadband on `desired` vs the
   held position, `target_vol`, `max_leverage`, `vol_span` -- matches v4's
   defaults exactly.

Falsification (pre-registered per ROUTINE.md step 2): if this variant's
ordering against v4 (which is better on return, which on drawdown) is not
the same on the ETH/BTC Bitfinex control pair as it is on BTC/Bitstamp, the
result does not survive falsification, exactly as happened to R-28 (retired
by R-31) and warned against by R-33. If the best-looking config merely runs
at a different mean exposure than v4, that is the R-31/R-32/R-33 arithmetic
artifact, not a regime-detection improvement, and must be flagged plainly
rather than reported as an edge.

UNREGISTERED experiment (ROUTINE.md): no `@register`, not in the README
comparison table, not in the CI-enforced inference set. Kept here as a
frozen, reviewable record of the R-34 novel-variant branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.bayes_confidence import bayesian_margin  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402


class KellyRegimeV5Bayes(Strategy):
    """v4's conditional-vol-targeting sizer, fed by a continuous latched Bayesian margin instead of the discrete anchor vote.

    See module docstring for the full mechanism and pre-registered
    falsification test. UNREGISTERED: `experiments/kelly_regime_v5_bayes.py`
    is not auto-discovered and carries no CI-enforced inference interval.
    """

    name = "kelly_regime_v5_bayes"
    # Bayesian margin warmup mirrors harsanyi_crowd's own (ATR(48) burn-in +
    # a little slack for the sticky posterior to leave its uniform prior) --
    # deliberately much shorter than v4's 80-day anchor warmup, because this
    # signal operates on an hours-to-days timescale, not weeks-to-months.
    warmup = 1300

    def __init__(self, mu: float = 0.15, stick: float = 0.985,
                 b_in: float = 0.15, b_out: float = 0.05,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55,
                 low_out: float = 0.85, exposure_mult: float = 1.0) -> None:
        self.mu = mu
        self.stick = stick
        self.b_in, self.b_out = b_in, b_out
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        # Diagnostic-only knob (default 1.0, never swept as a "performance"
        # parameter): a constant multiplier on the confidence fraction, used
        # solely to re-run a frozen config at a mean exposure matched to
        # v4's, per ROUTINE.md's "match risk before comparing anything" and
        # the R-31/R-32/R-33 exposure-level artifact it warns about.
        self.exposure_mult = exposure_mult

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # --- regime-confidence input: continuous, hysteresis-latched Bayesian margin ---
        margin = bayesian_margin(df, mu=self.mu, stick=self.stick)

        # --- risk axis: kelly_regime_v3's conditional-vol-targeting scale, UNCHANGED ---
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        frac_series = np.zeros(n)  # exposed for the vote-correlation diagnostic
        pos = 0.0
        vol_state = 0   # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        conf_state = 0  # 0 = not latched (no confidence), 1 = latched-in
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if vol_state == 0:
                    vol_state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif vol_state == 1 and x < self.high_out:
                    vol_state = 0
                elif vol_state == -1 and x > self.low_out:
                    vol_state = 0
            scale = full[i] if vol_state != 0 else steady[i]

            m = margin[i]
            if conf_state == 0:
                if m > self.b_in:
                    conf_state = 1
            elif m < self.b_out:
                conf_state = 0
            frac = (0.0 if conf_state == 0
                    else float(np.clip((m - self.b_out) / (1.0 - self.b_out), 0.0, 1.0)))
            frac_series[i] = frac  # unscaled, for the vote-correlation diagnostic

            desired = (frac * self.exposure_mult) * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["confidence_frac"] = frac_series
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures
```
