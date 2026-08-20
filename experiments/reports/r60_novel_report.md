# R-60 (novel branch) — does a CUSUM change-point vote (TIMING, not SCALE) fix v4's panel drawdown inversion? (08-20)

Unregistered experiment (backlog **B-26**). Code:
`experiments/r60_novel_cusum_vote.py`. Shared pre-registration:
`experiments/r60_shared.py` (windows, costs, decision rules, matched-hold
harness — read there, reused here, not restated differently). This
file's own docstring carries the branch-specific pre-registration (the
exact CUSUM formula, the design choice, the (k,h) grid, the selection
rule) written and committed before a single backtest in this file ran.
Nothing under `src/tradebot/strategies/` is touched: `KellyRegimeCusumVote`
is a plain, unregistered `Strategy` subclass constructed directly, never
through `get_strategy()`.

## 1. The question, one sentence

R-57 found `kelly_regime_v4`'s matched-exposure drawdown advantage
inverts on 6 of 6 further Coinbase instruments; R-59 tested the SCALE
axis twice (per-asset `target_vol` calibration, and a self-normalizing
relative-vol scale) and both failed identically — does replacing the
vote's *timing* mechanism (a 20/40/80-day moving-average crossing with a
latch) with a causal, sequential CUSUM change-point detector, which can
in principle react as soon as evidence accumulates rather than only once
enough of the old regime has rolled out of a calendar window, fix it
instead?

## 2. The mechanism

**CUSUM formula** (Page, 1954, *Continuous Inspection Schemes*, Biometrika
41(1/2)), on log-returns `r_t = log(close_t) - log(close_{t-1})`, a
reference drift `k > 0` and threshold `h > 0`:

```
S+_t = max(0, S+_{t-1} + (r_t - k))
S-_t = max(0, S-_{t-1} - (r_t + k))
```

Signal on the first bar where `S+_t >= h` (flip bullish, reset `S+` to 0)
or `S-_t >= h` (flip bearish, reset `S-` to 0); the vote **latches**
between signals, holding the previous verdict — the identical hysteresis
semantics as the moving-average anchors it replaces
(`v.ffill().fillna(0.0)`), including the same bearish/flat (0.0) default
before the first signal fires. Implemented as a genuine bar-by-bar Python
loop (`_cusum_vote`), not a vectorized shortcut, specifically because a
running-sum-with-reset is exactly the kind of recursive computation a
"clever" vectorized rewrite could leak lookahead into — see section 3.

**Design choice: (b), an ensemble of 3 CUSUM detectors, not (a) a single
detector.** `kelly_regime_v4`'s vote is not one signal — it is three
anchors (20/40/80 days) averaged into a continuous `frac` in [0,1]. A
single CUSUM detector (design a) would change two things about the vote
at once: its timing mechanism AND its granularity (continuous average →
binary latch jumping straight between 0.0 and 1.0), confounding the
question this round asks. Design (b) keeps the vote's architecture
parallel to v4's own: three detectors, averaged exactly the way the
three anchors are (`sum(votes)/len(votes)`, then `vote_gamma`), isolating
TIMING as the only axis that changes. The three detectors share one
reference drift `k` and a **doubling threshold ladder** `h, 2h, 4h`
(`CUSUM_LADDER_MULTIPLIERS = (1.0, 2.0, 4.0)`, fixed, not swept) — the
natural CUSUM analogue of v4's own "each anchor covers twice the horizon
of the last" doubling structure, expressed in accumulated-evidence units
rather than calendar days since a CUSUM has no fixed window to double.
Everything else in `prepare()` — the fractional-Kelly vol-targeting scale
terms, the high/low breakout hysteresis state machine, the 10% deadband,
`target_vol=0.55`, `max_leverage=2.0` — is byte-identical to
`KellyRegimeV3.prepare()` (inherited unchanged by `KellyRegimeV4`).

`k` and `h` are **global constants, identical across all 8 assets** (BTC,
ETH, 6 panel) — no per-asset number is fit anywhere in this branch, to
keep this round's question (TIMING, not SCALE) uncontaminated by the
per-asset-calibration confound R-59 already tested and closed.

## 3. Causality tamper probe

`test_causality_strict.py`'s methodology (opposite 3x/÷3 price and 7x/÷7
volume tampers after a cut, decisions compared at 1/2/3/5/10/20 bars
before the cut), run twice — once before the sweep (mid-grid sanity
config, k=0.001/h=0.02) and once after selection (the frozen final
config, which happened to be the same point) — on BTC (2022-12-31 and
earlier only) plus BCH and LTC, constructing `KellyRegimeCusumVote`
directly: **PASS on all 3 assets, both times** — decisions identical
under opposite post-cut tampers. The CUSUM running-sum state is computed
by a genuine forward per-bar loop reading only `r[i]` and the previous
bar's own accumulator, never a batch statistic over the series and never
a future bar.

## 4. The pre-registered grid, and its results

`CUSUM_K_GRID = (0.0005, 0.0010, 0.0020)` × `CUSUM_H_GRID = (0.0100,
0.0200, 0.0400)` — 9 points, derived from a back-of-envelope BTC
per-bar-sigma calculation (`0.55 / sqrt(365.25*288) ≈ 0.0017`), fixed
before any backtest in this file ran. Every point evaluated on
PANEL_TRAIN (D1-analog, 6 assets) and CONTROL (D2-analog, BTC/ETH),
before any other number was read:

| k | h | D1 k1/6 | BTC dDD (margin input) | ETH dDD | D2 margin (pp, larger=safer) |
|---|---|---|---|---|---|
| 0.0010 | 0.02 | 0 | +7.2 | +5.7 | **−12.2** (selected) |
| 0.0020 | 0.04 | 0 | +9.8 | +6.4 | −12.9 |
| 0.0010 | 0.04 | 0 | +3.0 | +9.1 | −15.6 |
| 0.0005 | 0.04 | 0 | +17.1 | +11.1 | −17.7 |
| 0.0005 | 0.02 | 0 | +16.5 | +13.5 | −20.0 |
| 0.0020 | 0.02 | 0 | +20.1 | +12.4 | −20.7 |
| 0.0010 | 0.01 | 0 | +22.4 | +9.3 | −23.0 |
| 0.0020 | 0.01 | 0 | +26.5 | +10.2 | −27.1 |
| 0.0005 | 0.01 | 0 | +28.0 | +26.6 | −33.1 |

**Every one of the 9 grid points scores D1 k1 = 0/6.** This is a
washout of the entire pre-registered grid, not an unlucky selection —
no (k,h) combination in the narrow, pre-registered range produces even
one panel asset where the candidate's matched-exposure drawdown beats
the matched hold's. Every margin is also negative, meaning no grid point
would have passed D2's regression tolerance either. Applying the
pre-registered selection rule (highest D1 k1, ties broken by D2 margin)
mechanically on a 9-way tie at k1=0 selects **k=0.0010, h=0.0200**
(margin −12.2pp, the least-bad of the nine) — frozen for D1–D5 below.

## 5. Results against r60_shared's frozen rules

### D1 (PRIMARY) — PANEL_TRAIN, spot @0.10%, matched-exposure drawdown

| asset | c (mean notional) | candidate DD | matched-hold DD | Δ DD (pp, + = worse) | 95% interval | candidate trades | v4 baseline trades |
|---|---|---|---|---|---|---|---|
| BCH | 0.31 | 62.6% | 50.2% | **+12.3** | [−0.5, +35.8] | 156 | 68 |
| LTC | 0.25 | 54.7% | 35.6% | **+19.2** | [+2.8, +37.9] | 198 | 82 |
| ETC | 0.23 | 54.7% | 31.6% | **+26.0** | [+6.3, +45.9] | 226 | 77 |
| DASH | 0.23 | 44.2% | 34.6% | **+9.8** | [+0.6, +31.7] | 201 | 96 |
| LINK | 0.24 | 49.5% | 34.0% | **+18.3** | [+1.6, +38.5] | 223 | 103 |
| XTZ | 0.20 | 60.7% | 31.9% | **+29.4** | [+9.2, +55.3] | 308 | 92 |

**0 of 6** — exact binomial p = 1.0000 → **FAILS**. The sign is positive
(candidate worse than the matched hold) on every asset, and unlike R-59's
two branches, **5 of 6 intervals exclude zero** here (all five against
the candidate) — a cleaner, more decisive failure than either SCALE-axis
branch produced. Candidate trade counts (156–308) run **1.8x–3.3x**
v4's own baseline trade counts (68–103) on the identical asset/window —
the named whipsaw risk materializes plainly: CUSUM's ensemble, without
v4's calendar-anchor inertia, flips far more often.

### D2 (FALSIFICATION) — CONTROL, BTC/ETH, spot @0.10%

| asset | candidate dDD (matched) | R-57's v4 control | tolerance | candidate trades | v4 trades | verdict |
|---|---|---|---|---|---|---|
| BTC | **+7.2pp** [−7.7, +17.4] | −5.6pp | ≤ base+5pp = −0.6pp | 86 | 61 | **FAILS** |
| ETH | **+5.7pp** [−0.7, +23.3] | −11.5pp | ≤ base+5pp = −6.5pp | 146 | 54 | **FAILS** |

**FAILS on both assets**, and by a wide margin — not a marginal miss.
Unlike R-59's two branches (both of which passed D2 cleanly), the CUSUM
candidate's own matched-exposure drawdown on BTC and ETH is **worse than
the matched hold**, reversing the sign v4 itself achieves on the two
instruments the mechanism was built for. Trading turnover on BTC/ETH is
also elevated (86 vs 61, 146 vs 54) though less dramatically than on the
panel.

### D3 (CRASH-TRANSITION-LAG) — BTC, the three CRASH_WINDOWS, spot

| window | candidate lag (bars) | v4 baseline lag (bars) | candidate faster by |
|---|---|---|---|
| 2018-11 | 2181.0 | 800.0 | **−1381.0** (candidate SLOWER) |
| 2020-03-covid | 0.0 | 4580.0 | +4580.0 |
| 2022-11-ftx | 876.0 | 877.0 | +1.0 (essentially tied) |
| **mean** | **1019.00** | **2085.67** | +1066.67 |

**PASSES** the frozen rule (candidate mean lag 1019.00 bars ≤ baseline
2085.67 + 2 bars) — comfortably, not marginally: the CUSUM ensemble
de-risks after a crash peak roughly twice as fast as v4's calendar-anchor
vote on average, driven almost entirely by the COVID window, where the
candidate was already flat exactly at the price peak (lag 0) while v4's
20/40/80-day anchors took 4580 bars (~15.9 days) to fully latch bearish.

**Flagged explicitly, as instructed, because the aggregate PASS hides
it:** the **2018-11 window is the opposite result** — the candidate took
2181 bars (~7.6 days) to flatten after the local price peak, **1381 bars
(~4.8 days) slower** than v4's own 800-bar (~2.8 day) baseline lag. This
is precisely the named failure mode from this round's own brief: a
CUSUM tuned to be fast on quiet-chop evidence accumulation can be slow to
re-trigger immediately after a large move has already exhausted (or
mis-directed) its running sum, and a crash is exactly such a moment. The
frozen D3 rule is a *mean*-across-three-windows gate, and it passes on
the strength of one dramatic window (COVID) while masking a real,
same-direction-as-warned regression in another (Nov 2018). Read
per-window rather than as a single aggregate number, D3's real result is
2 windows faster, 1 window slower by more than the whole gate's 2-bar
tolerance — a genuine mixed result that a mean-only decision rule
reports as a clean pass.

### D4 (GENERALIZATION, descriptive) — PANEL_TEST 2023-2026, spot @0.10%

| asset | Δ DD (pp, matched) | 95% interval |
|---|---|---|
| BCH | +14.0 | [−0.3, +32.1] |
| LTC | +13.2 | [−4.9, +29.1] |
| ETC | +30.4 | [+7.6, +51.4] |
| DASH | +1.6 | [−16.2, +13.5] |
| LINK | +7.7 | [−10.4, +25.6] |
| XTZ | +13.4 | [−0.3, +41.6] |

**0 of 6**, same inverted sign as D1, on the held-out panel window — not
a gate, but corroborates D1 rather than contradicting it.

### D5 (0.40% fee falsification) — PANEL_TRAIN, spot @0.40%, beats buy_and_hold's final balance

| asset | candidate final | buy_and_hold final |
|---|---|---|
| BCH | $327 | $438 |
| LTC | $430 | $1,723 |
| ETC | $375 | $3,137 |
| DASH | $423 | $641 |
| LINK | $363 | $2,386 |
| XTZ | $177 | $444 |

**0 of 6** (threshold ≥5/6 to "survive") → **FAILS, as predicted**. Worse
than R-57's own v4 (2/6) and R-59's two branches (3/6 each) — the higher
turnover this branch's whipsaw diagnostic already flagged costs
proportionally more at the higher fee tier, on top of the return edge
this project has repeatedly found does not survive real costs.

## 6. Verdict

Applying `experiments.r60_shared.promoted(k1, dd_advantage, candidate_lag,
baseline_lag)` mechanically (requires D1 ≥ 5/6 **and** D2 passing both
BTC and ETH **and** D3 passing):

- D1: 0/6 → FAILS (needed ≥5/6)
- D2: FAILS (BTC +7.2pp, ETH +5.7pp, both exceed the +5pp regression tolerance)
- D3: PASSES (1019.00 bars vs baseline 2085.67 bars) — but see the
  per-window caveat in section 5, which this round's brief specifically
  asked to be taken seriously rather than accepted on the aggregate
  number alone
- `promoted(0, {...}, 1019.00, 2085.67)` → **False**

**NEGATIVE.** The CUSUM ensemble is not merely a weaker fix than v4's own
mechanism — on this evidence it is a *worse* one on every axis except the
aggregate crash-lag number: D1 fails more decisively than R-59's own
novel (relative-vol-scale) branch, which excluded zero on only 2/6
paired-bootstrap intervals (both against that candidate too) against
this branch's 5/6 (also all against the candidate) — the sign is
uniformly wrong on both branches, but this one is statistically sharper
about it. And D2 — which BOTH of R-59's branches passed cleanly — **fails
outright here**, meaning this
branch is the first of the twenty-plus SIZE/TIMING-axis attempts on this
family to break the two instruments the mechanism already worked on
while also failing to fix the six it doesn't. The one place this branch
clearly outperforms the incumbent is the crash-transition-lag aggregate
(faster de-risking after the COVID crash specifically), but that
same test's own per-window breakdown shows the CUSUM ensemble is *slower*
than v4 after the Nov-2018 crash — the specific failure mode the round's
own pre-registration named as the outcome to watch for, materializing in
exactly one of the three marquee windows. Read together with R-59's own
two branches, this closes B-26: the vote/gate's TIMING axis, tested here
with a materially different, causal, sequential detector rather than a
retuned calendar window, is not the fix either. Every axis this strategy
family's own mechanism has now been varied on to answer R-57's panel
question — SCALE (twice, R-59) and TIMING (this round) — is negative;
R-57's own alternative explanation (the matched hold's advantage on
these instruments is largely a buy-the-dip effect priced into the panel's
own higher-volatility, mean-reverting price dynamics, which no version of
`kelly_regime`'s vote-and-scale architecture participates in while it is
stood aside) is the leading account left standing.

## 7. Configurations evaluated

**218** total backtests (`CONFIG_COUNT`, this branch only; causality
probes use `prepare()`/`on_bar()` directly and read no backtest via
`measure()`, so both runs cost 0, matching R-57/R-59's convention; the D3
crash-lag check likewise calls `prepare()` directly with no broker, cost
0):

- Sweep: 9 grid points × 8 assets (6 panel + BTC + ETH) × 2 backtests
  (candidate + matched hold) = **144**
- D1: 6 panel assets × 4 backtests (candidate, buy_and_hold, matched
  hold, v4 baseline) = **24**
- D2: 2 assets (BTC, ETH) × 4 backtests = **8**
- D3: **0** (diagnostic, `prepare()` only)
- D4: 6 panel assets × 3 backtests (candidate, buy_and_hold, matched
  hold) = **18**
- D5: 6 panel assets × 4 backtests = **24**

144 + 24 + 8 + 0 + 18 + 24 = **218**, matching the module's own printed
`CONFIG_COUNT`. Combined with R-59's own 140 (60 novel + 80 conservative)
this round's total across the parallel round is the sum with whatever
the R-60 conservative branch reports separately, per ROUTINE.md's
parallelism convention (trials count is the total across all branches).

Holdout consultations added by this branch: **0** — no BTC/ETH bar past
2022-12-31 is read anywhere in `experiments/r60_novel_cusum_vote.py`
(`_btc_df()`/`_eth_df()`/`_btc_full_for_crash()` all slice to
`:2022-12-31` before any use, including both causality probes and the D3
crash-lag check).

`pytest -q`: **461 passed**, unchanged (nothing under `src/` was touched).

## 8. Raw data

`reports/r60_novel/sweep_cells.csv`, `sweep_summary.csv`,
`d1_panel_train.csv`, `d2_control.csv`, `d3_crash_lag.csv`,
`d4_panel_test.csv`, `d5_panel_train_040.csv`.
