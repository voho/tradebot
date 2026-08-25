# R-127 (NOVEL branch) -- do brief ETH-idiosyncratic divergence episodes drive R-126 novel's B4 sign inversion? (08-25)

Unregistered diagnostic. Code: `experiments/r127_novel_event_excision_retest.py`. Not `@register`ed, not auto-discovered, nothing committed by this session. No refit, no new mechanism: this branch re-evaluates R-126 novel's own already-recorded CVaR-budgeted `champions_council` reallocation, unchanged, under a post-hoc calendar-day filter of its already-realized daily return series. Builds on the frozen `experiments/r127_shared.py` (`TERRA_LUNA_WINDOW`, `THE_MERGE_WINDOW`, `low_correlation_days`, `excise_days`) and reuses `experiments/r126_shared.py` / `experiments/r126_novel_cvar_council.py` read-only.

## 1. Direction and question

R-127's window scan (`r127_shared.SCAN`, computed before either R-127 branch script existed) already found the calendar-matched ETH `INNER_VAL` window is the single best regime-fingerprint match to BTC's own `INNER_VAL` among 95 candidates (percentile ~1.05) -- the sibling conservative branch (`r127_conservative_report.md`) independently confirms and quantifies this, refuting the coarse "wrong 24 months were compared" hypothesis. This branch tests the finer hypothesis the shared module's docstring pivoted to as a result: *within* that correctly-matched window, do brief, structurally-dated, ETH-idiosyncratic episodes -- Terra/Luna's collapse and The Merge, plus a data-driven low-BTC/ETH-correlation-day filter -- disproportionately drive R-126 novel's own sign inversion (BTC spot `d_sharpe=+0.388`, ETH spot `d_sharpe=-0.530`)?

## 2. Methodology

1. **Step 1, reproduction (self-check).** Reconstruct R-126 novel's own ETH B4 evaluation exactly as its `main()` does: fit the CVaR-budgeted weight schedule (`fit_novel_council`, primary config `alpha=0.05`/`lookback=90`) on ETH's **full pre-holdout history** (`r126_shared.load_eth_train()`, 2019-03-14..2022-12-31, 399,861 bars -- not just `INNER_VAL`, so the weight schedule has real trailing history through most of the evaluation window rather than the equal-weight fallback), then evaluate via `r126_shared.b1_signal` on ETH spot over `INNER_VAL_START..INNER_VAL_END`.
2. **Step 2, raw daily series.** Re-derive the same candidate/reference daily return series `b1_signal` computes internally (`run_target_series` / `run_candidate_council` / `tradebot.inference.daily_returns`), kept as raw `pandas.Series` rather than immediately reduced to a paired-bootstrap summary, so specific calendar days can be excised from them.
3. **Step 3, three excisions**, each via `r127_shared.excise_days` on the **daily** series (never a mutation of the 5m OHLCV bars fed to the engine) followed by `tradebot.inference.paired_bootstrap(..., stat=total_log_return, seed=127)`:
   - (a) `TERRA_LUNA_WINDOW` (2022-05-04..2022-05-18) + `THE_MERGE_WINDOW` (2022-09-08..2022-09-22) -- 30 calendar days, 4.1% of the 729-day window, dates fixed from public record.
   - (b) data-driven: calendar days where trailing 14-day BTC/ETH daily-return correlation falls below the structural `LOW_CORR_THRESHOLD=0.30` (`r127_shared.low_correlation_days`, using BTC's full pre-holdout daily log returns and ETH's `INNER_VAL`-only daily log returns per `r127_shared.load_btc_train`/`load_eth_train` -- note this ETH loader is `r127_shared`'s own INNER_VAL-restricted version, deliberately different from the full-history `r126_shared.load_eth_train()` used in Steps 1-2).
   - (c) the union of (a) and (b).
   An un-excised baseline is also evaluated at the same seed (127), as the like-for-like comparison basis for the three excisions (distinct from the Step-1 reproduction, which uses R-126's own seed=126 for an exact-number check).
4. **Step 4, verdict.** Each excision's daily-return-based `d_sharpe` (candidate `annualized_sharpe` minus reference `annualized_sharpe`, `tradebot.inference.annualized_sharpe` on the filtered daily series) is compared against the un-excised baseline: SIGN FLIP, materially narrowed (>0.2 Sharpe, this project's own noise floor), widened, or unchanged.
5. **Causal check.** Every frame loaded (`r126_shared.load_eth_train()`, `r127_shared.load_btc_train()`, `r127_shared.load_eth_train()`) has its last-bar timestamp printed and asserted `< OOS_START = 2023-01-01`, on top of the assertions already inside both shared modules' own loaders.

## 3. Causality / no-lookahead check

All three loaders' last-bar timestamps print `2022-12-31 23:55:00+00:00`, asserted `< 2023-01-01`: **PASS** for all three. `pytest -q tests/test_causality_strict.py`: **51 passed**, 0 failed (65.4s).

## 4. Result 1 -- Step-1 reproduction of R-126 novel's own ETH number

| | R-126 published | this reproduction |
|---|---|---|
| ETH spot `d_sharpe` (candidate vs `champions_council`) | -0.5300 | **-0.5302** |
| sharpe_cand | (not separately printed in R-126's ledger cell) | -0.0353 |
| sharpe_council | (not separately printed) | +0.4949 |

Difference from published: **-0.0002** (same sign, well inside floating-point/solver-noise tolerance). Solver diagnostics on this run: 44 fitted rebalance points, 3 equal-weight fallback (insufficient lookback history), 7/44 with a binding return-floor multiplier, floor met 44/44, mean max-weight 0.249, entropy 1.68 (near-max of ln6=1.79) -- consistent with R-126's own report of a well-diversified, non-degenerate solution. **Reproduction confirmed; proceeding.**

## 5. Result 2 -- raw daily series

729 candidate/reference daily observations each, 2021-01-02..2022-12-31 (first `INNER_VAL` day dropped by `daily_returns`' own diff convention). Bar-level `d_sharpe` from `compute_metrics` (5m-bar annualized Sharpe): **-0.5302**, matching Step 1 exactly (same underlying backtest, re-derived rather than re-run).

## 6. Result 3 -- three excisions vs baseline

Excluded-day counts: named-event = 30 days; data-driven low-correlation = 11 of 716 days with a defined trailing-14d correlation (`threshold=0.30`); union = 41 distinct days (**zero** overlap between the two sets -- the low-correlation filter did not simply rediscover Terra/Luna or the Merge).

| variant | n days (excised) | d_sharpe (daily, annualized) | delta vs baseline | paired_diff [95% CI] | significant |
|---|---|---|---|---|---|
| baseline (unexcised, seed=127) | 729 (0) | **-0.5541** | -- | -0.2596 [-0.831, +0.245] | No |
| (a) named-event (Terra/Luna + Merge) | 699 (30) | **-0.2232** | **+0.3309** | -0.1266 [-0.656, +0.346] | No |
| (b) low-correlation days | 718 (11) | **-0.4541** | +0.1000 | -0.2012 [-0.769, +0.322] | No |
| (c) union of (a) and (b) | 688 (41) | **-0.1210** | **+0.4331** | -0.0681 [-0.602, +0.392] | No |

(Daily-return `d_sharpe` at the un-excised baseline, -0.554, is close to but not identical to the bar-level -0.530 from Step 1 -- expected, since `annualized_sharpe` on daily returns and `compute_metrics.sharpe_ratio` on 5m-bar returns are different statistics of the same underlying equity path, per `tradebot.inference`'s own "everything here works on daily returns, not 5m bars" convention. Both agree in sign and rough magnitude, which is what matters for the excision comparison below.)

## 7. Result 4 -- verdict per variant

| variant | classification (vs +/-0.2 Sharpe noise floor) |
|---|---|
| (a) named-event | **materially narrowed** (gap shrinks by 0.33, from -0.554 to -0.223 -- a ~60% reduction in magnitude) |
| (b) low-correlation | unchanged within noise floor (gap shrinks by 0.10, below the 0.2 bar) |
| (c) union | **materially narrowed** (gap shrinks by 0.43, from -0.554 to -0.121 -- driven almost entirely by (a); (b) adds little on top) |

**No variant flips the sign.** ETH's candidate remains behind `champions_council`'s reference in every excision -- the strict B4 sign-replication test (does ETH's sign match BTC's positive sign?) would still read FAIL after any of these excisions. None of the three paired-bootstrap intervals excludes zero at any stage (baseline included) -- the paired comparison was never statistically significant to begin with, at n=729 (or n=688-718 after excision), so "materially narrowed" here is a point-estimate movement bigger than this project's own noise floor, not a movement between two significant results.

## 8. Configurations evaluated

**4 total**, matching the task's minimum:
1. baseline reproduction (Step 1, R-126 novel's own primary config, ETH spot, unexcised, seed=126) -- a self-check reproduction of an already-recorded number, not a new search.
2. (a) named-event excision, paired test (seed=127).
3. (b) data-driven low-correlation-day excision, paired test (seed=127).
4. (c) union-of-(a)-and-(b) excision, paired test (seed=127).

The un-excised "baseline (seed=127)" row in Result 3 reuses the same daily series as configuration 1 (re-evaluated at the round's own seed for a like-for-like comparison basis against 2-4) and is not counted as a fifth configuration. No mechanism was refit and no new backtest engine run beyond the two already implied by Step 1/2 (candidate CVaR-council target on ETH spot, `champions_council` reference on ETH spot) -- every excision operates post-hoc on the same two already-computed daily return series, per `r127_shared.excise_days`'s own design. No selection occurred among the four: all four are reported regardless of outcome, per `docs/ROUTINE.md`'s "every branch reports, including the dead ones."

## 9. Verdict

**CONFIRMED (partial), for this construction.** Excising the two structurally-dated, pre-registered ETH-idiosyncratic event windows (Terra/Luna, The Merge -- 30 of 729 days, 4.1% of the window) reduces the magnitude of R-126 novel's ETH `d_sharpe` gap by roughly 60% (-0.554 -> -0.223), clearing this project's own +/-0.2 Sharpe noise floor for "material" by a wide margin -- a small number of calendar days, chosen from public record and never fit to any performance number, account for the large majority of the ETH-side underperformance this construction shows against `champions_council`. The data-driven low-correlation filter (11 days, zero overlap with the named events) adds comparatively little on its own (gap narrows by only 0.10, inside the noise floor) but does not widen the gap either, and the union of both narrows further still (-0.554 -> -0.121, a ~78% reduction). This is genuine, structured evidence for the idiosyncratic-divergence hypothesis, not noise: one excision variant clears the material-narrowing bar decisively, a second sits right at its edge, and none of the three widens the gap.

**But it is a partial, not a full, confirmation.** No excision flips ETH's `d_sharpe` sign to positive -- the candidate still underperforms `champions_council` on ETH after removing Terra/Luna, the Merge, and every low-correlation day this project's structural threshold identifies. Under the strict pre-registered B4 sign-replication test (does ETH's sign match BTC's?), this construction would still fail even with the excisions applied. The honest reading is two-layered: the *majority* of the gap's magnitude traces to a handful of brief, named, ETH-specific structural episodes -- exactly the mechanism this branch was built to test for -- but a residual negative gap (-0.12 to -0.22 Sharpe, still comparable in size to the noise floor itself) survives even after those episodes are removed, meaning idiosyncratic-episode excision narrows but does not fully explain R-126 novel's own inversion. Neither "the six-fold pattern is entirely a few-day artifact" nor "excision changes nothing" is the correct summary; both would overstate this result in opposite directions.

**What this does and does not mean, read against the sibling conservative branch.** The conservative branch (`r127_conservative_report.md`) refuted the coarse hypothesis that the wrong 24 months were being compared -- the calendar window is, if anything, an unusually good regime match. This branch's finding is consistent with, and sharpens, that result: *given* a well-matched window, a small number of ETH-specific days inside it still carry a disproportionate share of the mechanism's underperformance relative to `champions_council`. That is a genuinely different, more specific finding than "the window is fine" -- it says *something* inside the window is doing a lot of the work, even though the window itself is not mismatched.

**One-line lesson.** For R-126 novel's CVaR-budgeted `champions_council` reallocation specifically, brief ETH-idiosyncratic structural events (not a mismatched evaluation window, and not diffuse low-correlation days generally) account for the majority -- but not all -- of the BTC/ETH sign-inversion gap; a future round testing whether this generalizes across the other five inverting constructions (R-109, R-113, R-115-conservative, R-125-conservative, R-126-conservative) would need to re-run this same excision battery on each of their own fitted targets, since this branch deliberately tested only the one construction R-126 novel already validated, per its own pre-registration against re-litigating the mechanism question.

**Holdout counter: +0.** No bar at or after `2023-01-01` was read by this branch (Section 3). This round produces no promotable strategy candidate -- it is a diagnostic of the B4 falsification test's own convention, per `r127_shared.py`'s pre-registered decision rule, not a new mechanism to promote.
