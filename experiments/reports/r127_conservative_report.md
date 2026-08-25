# R-127 (CONSERVATIVE branch) -- does calendar-window mismatch explain the six-fold BTC-pass/ETH-invert pattern? (08-25)
Unregistered diagnostic. Code: `experiments/r127_conservative_regime_fingerprint.py`. Not `@register`ed, not auto-discovered, nothing committed by this session. No strategy code is written or touched anywhere in this branch. Builds strictly on the frozen `SCAN` computed once by `experiments/r127_shared.py` (95 candidate 730-day ETH windows, weekly-stepped, spanning ETH's full pre-holdout history, each compared to BTC's fixed `INNER_VAL` regime fingerprint by standardized Euclidean distance). This file does not recompute or re-select the window scan -- see the shared module's own "single-frozen-window discipline."
## 1. Direction and question

Six independent prior constructions (R-109, R-113, R-115-conservative, R-125-conservative, R-126 both branches) passed this project's BTC-side gate and inverted sign on ETH's B4 falsification test, under the convention of comparing BTC and ETH on the *identical calendar dates* (`INNER_VAL_START=2021-01-01` to `INNER_VAL_END=2022-12-31`) with no check that the two assets pass through comparable regime composition over those 24 months. This branch asks that question directly, with proper significance tests, building on the already-frozen window scan.
## 2. Methodology

1. **Coarse pre-registered test** (from `r127_shared.py`'s own docstring): does the calendar-matched ETH window's fingerprint distance to BTC's `INNER_VAL` fingerprint fall above the 90th percentile of the null distribution formed by all 95 candidate windows' distances? This is a direct read of the already-frozen `SCAN` dict -- no new computation, only a plain PASS/FAIL statement against the pre-registered bar.
2. **Two dependency-free two-sample significance tests** on BTC `INNER_VAL` vs ETH calendar-window CALENDAR-DAILY log returns (scipy is not installed in this venv; both hand-rolled in pure numpy/pandas, matching this project's own precedent for dependency-free statistical machinery):
   - a **Brown-Forsythe / Levene-type** statistic for equality of variance (median-centered absolute deviations, one-way-ANOVA-style W statistic), with a permutation p-value from 20,000 pool-and-resplit permutations (seed=20260825, fixed before either test ran, not tuned);
   - a **two-sample Kolmogorov-Smirnov** D statistic (max absolute difference between empirical CDFs), with a permutation p-value from the same procedure and permutation count.
   Both use the standard +1/(n_perm+1) correction so a p-value is never reported as literally zero.
3. **Full 10-statistic fingerprint side-by-side**, reading `SCAN['fp_btc_val']`, `SCAN['fingerprints'][cal_idx]` and `SCAN['scale']` directly, so individual moments can be inspected even where the aggregate distance is small.
4. **Robustness check**: fingerprint distance and percentile of the rank-2 through rank-5 closest candidate windows overall (from the same frozen `SCAN['distances']`/`SCAN['windows']`), to show the calendar window's closeness is not a one-off comparison artifact.
5. **Causal check**: every date this file reads is asserted strictly before `OOS_START = 2023-01-01`.
## 3. Causality / no-lookahead check

Explicit assertion that BTC `INNER_VAL`'s last bar, ETH calendar-window's last bar, and the max end date across all 95 frozen `SCAN` windows are all strictly before `OOS_START = 2023-01-01`: **PASS**.
## 4. Result 1 -- coarse pre-registered test

Calendar-matched ETH window: 2021-01-01 .. 2022-12-31. Fingerprint distance to BTC `INNER_VAL` = **2.4331**, percentile among all 95 candidates = **1.05**.

Pre-registered bar: percentile > 90 required to confirm the coarse window-mismatch hypothesis (i.e. the calendar window would need to be an unusually POOR regime match). 1.05 is not above 90 -- in fact it is the single closest match among all 95 candidates (rank 1, `REGIME_MATCHED_ETH_WINDOW == CALENDAR_ETH_WINDOW` exactly, per the shared module).

**Plainly stated: the coarse window-mismatch hypothesis is REFUTED, not ambiguous.** The calendar-matched ETH window is not a poor regime match to BTC's `INNER_VAL` -- it is the best available match by this metric, closer than every other 730-day window in ETH's pre-holdout history.
## 5. Result 2 -- two-sample significance tests on daily log returns

BTC `INNER_VAL` daily log returns: n=729, ETH calendar-window daily log returns: n=729. Raw variance ratio (ETH/BTC) = 1.7765.

| test | statistic | p-value (permutation) | reject H0 at alpha=0.05? |
|---|---|---|---|
| Brown-Forsythe (Levene-type), equal variance | W = 33.2543 | 0.0000 | yes |
| two-sample KS, equal distribution | D = 0.0905 | 0.0050 | yes |

(20,000 pool-and-resplit permutations per test, seed=20260825, fixed before either test ran.)
## 6. Result 3 -- full fingerprint side-by-side

| statistic | BTC INNER_VAL | ETH cal-window | z-diff (SCAN scale) |
|---|---|---|---|
| ann_vol | +0.74290 | +0.98953 | -5.724 |
| vol_of_vol | +0.01285 | +0.01790 | -1.510 |
| mean_daily_ret | -0.00077 | +0.00066 | -1.330 |
| skew | -0.22022 | -0.38063 | +0.239 |
| excess_kurtosis | +2.34250 | +3.95256 | -0.215 |
| acf_lag1 | -0.05456 | -0.05447 | -0.002 |
| acf_lag5 | +0.02617 | -0.06613 | +4.227 |
| acf_lag20 | +0.04974 | +0.05486 | -0.139 |
| max_drawdown | -0.76460 | -0.79350 | +0.345 |
| frac_positive_days | +0.49657 | +0.51644 | -2.068 |

Two coordinates carry most of the aggregate distance: **ann_vol** (z=+5.724 in magnitude) and **acf_lag5** (z=+4.227 in magnitude) alone account for roughly 86% of the squared distance that produces the aggregate RMS figure in Section 4 -- consistent with the significance-test finding above that ETH's calendar-window returns really do carry higher raw volatility than BTC's `INNER_VAL` returns. The finding that this window is nonetheless the single closest match among 95 candidates (Section 4) means the OTHER 94 ETH windows show gaps on these same two coordinates that are as large or larger relative to BTC -- consistent with ETH carrying structurally higher volatility than BTC across essentially its whole pre-holdout history, not just in this one window. The aggregate distance being small is a RELATIVE statement ("closest among ETH's own available windows"), not an ABSOLUTE one ("indistinguishable from BTC in raw scale") -- see Section 5's significance tests, which is exactly the right instrument for the absolute-scale question this table alone cannot answer.
## 7. Result 4 -- robustness across neighbouring windows

| rank | window | distance | percentile |
|---|---|---|---|
| 1 (calendar match) | 2021-01-01..2022-12-31 | 2.4331 | 1.05 |
| 2 | 2020-12-25..2022-12-24 | 2.4914 | 2.11 |
| 3 | 2020-12-18..2022-12-17 | 2.5143 | 3.16 |
| 4 | 2020-10-23..2022-10-22 | 2.5513 | 4.21 |
| 5 | 2020-12-11..2022-12-10 | 2.5859 | 5.26 |

The rank-2 through rank-5 windows are all similarly close (low double-digit or single-digit percentiles), not a cliff after rank 1 -- consistent with adjacent weekly-stepped windows overlapping heavily in their underlying bars and therefore in their fingerprints. This is expected and does not weaken the coarse-test conclusion: it shows the calendar window's closeness sits inside a broad, contiguous region of good matches (late-2020-through-2022-ish ETH windows), not an isolated fluke driven by one comparison.
## 8. Configurations / tests evaluated

This branch performs no strategy backtest and no parameter sweep -- the window scan itself is frozen, already-computed infrastructure from `r127_shared.py`, not a new configuration this branch selects among. The new statistical work this file adds is:

1. coarse percentile check (SCAN['cal_percentile'] vs 90th-pct bar)
2. Levene-type (Brown-Forsythe) permutation test for equal variance, BTC INNER_VAL vs ETH calendar-window daily log returns
3. two-sample Kolmogorov-Smirnov permutation test for equal distribution, same two samples
4. full 10-statistic fingerprint side-by-side table (descriptive, not a hypothesis test, but a new report artifact -- counted)
5. robustness distance/percentile check, rank-2 closest window
6. robustness distance/percentile check, rank-3 closest window
7. robustness distance/percentile check, rank-4 closest window
8. robustness distance/percentile check, rank-5 closest window

8 items total (1 coarse-test read + 2 significance tests + 1 fingerprint table + 4 robustness-window checks). No selection occurred among them -- every item is reported, none is filtered by outcome. No Sharpe/backtest number is computed anywhere in this branch, so no deflated-Sharpe calculation applies, and the holdout counter is unaffected by this branch (no bar at or after `2023-01-01` was read -- see the causal check above).
## 9. Verdict

**REFUTED**, on the pre-registered decision criterion. The coarse pre-registered percentile test is decisive by itself: the calendar-matched ETH window used by all six prior constructions' B4 falsification test is the **single closest regime match to BTC's `INNER_VAL` window among all 95 candidates** spanning ETH's full pre-holdout history (percentile 1.05, far below the 90 required to confirm mismatch). That is the pre-registered bar this branch was built to check, and it fails to clear it in the direction that would support the confound hypothesis.

The two new significance tests add a genuine nuance rather than contradicting the percentile result: BOTH reject equality at alpha=0.05 (Brown-Forsythe p=0.0000, KS p=0.0050) -- ETH's calendar-window daily returns really do carry higher raw variance (ratio 1.78x) and a measurably different shape than BTC's INNER_VAL returns, at n=729 each. This is not a contradiction of the aggregate-distance finding: the fingerprint distance is a STANDARDIZED metric (each coordinate divided by its cross-candidate-window scale specifically so no one high-magnitude statistic like volatility dominates it), so a raw-scale volatility gap large enough to be statistically significant at n=729 can still standardize to a distance that ranks as the single best match among 95 candidates, if ETH's OTHER 94 candidate windows carry a similarly-elevated volatility relative to BTC's fixed target -- which, given ETH's structurally higher realized volatility across its whole pre-holdout history, is exactly what should be expected. The pre-registered decision criterion for this branch is the percentile test, not the two significance tests (added per this round's task as a deeper, supplementary look at individual moments) -- so the branch's verdict is still REFUTED, with the caveat that ETH's window is not identical to BTC's in raw scale, only unusually well-matched in relative regime shape among the ETH windows actually available. 

**What this does and does not mean.** It rules out ONE specific candidate explanation for the six-fold BTC-pass/ETH-invert pattern: the calendar-window convention is not silently comparing a well-matched BTC regime sample against a poorly-matched ETH one. Regime composition (volatility, autocorrelation, skew, drawdown depth, etc., as characterized by this fingerprint) is, if anything, unusually well aligned across the two assets over these 24 months, precisely because BTC and ETH move through the same macro crypto cycle together. **It does NOT explain why the six prior constructions actually inverted sign on ETH** -- that remains an open question, and per this round's own pre-registration, the finer-grained idiosyncratic-divergence hypothesis (brief ETH-specific episodes like Terra/Luna and the Merge disproportionately driving the flip, within an otherwise well-matched 24-month window) is exactly what the sibling NOVEL branch (`experiments/r127_novel_event_excision_retest.py`) was designed to test instead, and its own report should be read for that question rather than this one.

**One-line lesson.** A 40-round-old convention (identical-calendar-date B4 windows) that looked like an unexamined assumption turned out, on the first occasion anyone measured it, to already be doing the right thing -- the six-fold inversion this round set out to investigate needs a different explanation than window mismatch, and that explanation is not this branch's to give.
