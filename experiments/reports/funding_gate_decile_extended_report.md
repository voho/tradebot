# funding_gate_decile_extended — R-39 conservative branch (08-19)

Unregistered experiment. Code: `experiments/funding_gate_decile_extended.py`.
Not `@register`ed, not auto-discovered, **nothing committed by this session**.
Pre-registration: `docs/LEDGER.md`, "R-39 pre-registration — the network
re-check R-38 asked for, done properly, and what it unblocks", the
**Conservative** branch. This is R-35's funding-decile gate (`B-05`) read
against the extended funding series that R-39 fetched, which is the
"more data" R-35's own row said was the only thing that could reopen it.

---

## VERDICT (stated first, because it is unambiguous)

**NEGATIVE. Decisively so, and in the wrong direction.**

The pre-registered rule is:

> promote … only if the **full 2023-01-01 → 2026-08-19 holdout** (paired
> stationary block bootstrap …, funding charged as a first-class futures
> cost) gives a Δ log growth or Δ Sharpe 95% CI that **excludes zero in
> the gate's favor**, drawdown not worse, and survival of the 0.40% taker
> tier. Anything else is NEGATIVE and closes B-05 permanently (not
> "pending more data" — this round is the more data).

What the holdout returned for the frozen `w=180` configuration, on
futures with funding charged:

| statistic | point | 95% CI | P(gate > v4) |
|---|---|---|---|
| **Δ log growth** | **−0.872** | **[−1.701, −0.166]** | 0.009 |
| **Δ Sharpe** | **−0.582** | **[−1.149, −0.042]** | 0.018 |
| Δ max drawdown (pp) | +12.17 (worse) | [−6.99, +19.56] | 0.708 |

Both primary intervals **exclude zero against the gate**. Drawdown is
worse, not better. The 0.40% taker tier turns the gate Sharpe-negative
(−0.38) while v4 survives at +0.83. Every one of the rule's three clauses
fails, and two of them fail significantly rather than merely failing to
clear.

This is not "still underpowered." R-35's result was an interval containing
zero on one year; this is an interval **excluding** zero against the gate
on 3.6 years, with the pre-registered configuration, on the pre-registered
statistic. **B-05 closes.**

The sub-period split says where R-35's result went (§8): 2023 alone — the
one year R-35 already read — reproduces R-35's ledger table to the dollar
and is a null (Δ log growth −0.120 [−0.430, +0.139]). **All** of the
damage is in 2024–2026, the genuinely unseen funding data:
Δ log growth −0.746 [−1.466, −0.097], Δ Sharpe −0.955 [−1.620, −0.205].

---

## 0. What actually changed, and how "byte-for-byte" is guaranteed

The pre-registration allows exactly one change: `load_funding()` →
`load_funding_extended()`. This branch does not re-type R-35's 300-line
module; it **imports `FundingGateDecile` unmodified and injects the longer
series through the `funding=` constructor argument the R-35 file already
has** (that hook exists in R-35's own file — it was added there for its
own causality probe). The arm is a five-line subclass that overrides
`__init__` (to default the series) and `name` (so tables can label it),
and nothing else.

`identity_check()` runs at the top of every command and asserts:

```
FundingGateDecileExtended.prepare is FundingGateDecile.prepare      # same object
FundingGateDecileExtended.on_bar is FundingGateDecile.on_bar        # same object
FundingGateDecileExtended.warmup == FundingGateDecile.warmup
vars(ext_arm) minus 'funding' == vars(r35_arm) minus 'funding'      # same params
```

So the decision logic is not "a copy that matches" — it is the same
function object. `decile` stays **0.90**, never swept. The lookback set
stays **{90, 180, 365, expanding}**. The 3-settlement causal EWM stays.
No parameter was added.

`§4` independently confirms the equivalence empirically: on every
pre-2023 slice the extended-series arm and the R-35-series arm produce
**identical** final balances and trade counts, because the extension only
appends settlements after 2023-12-31.

### The one seam a reader must know about

`load_funding_extended` concatenates real Binance (2020-01-01 → 2023-12-31,
4,383 settlements, settled 03/11/19 UTC) with Deribit (2,884 8h buckets,
closed 00/08/16 UTC) for the post-2023 gap only, without rescaling —
the cross-venue level ratio is unstable year to year. Because the gate
ranks a rate against a *trailing window of settlements*, for roughly one
window-length after 2024-01-01 a Deribit rate is ranked against a partly
Binance history. That is a real seam and this branch is not permitted to
"fix" it (the pre-registration fixes the mechanism). **§9 tests whether
it explains the result. It does not** — a pure-Deribit series with no
splice at all gives the same, slightly stronger, negative.

---

## 1. Build check — `decile=1.1` must reproduce `kelly_regime_v4` bit-identically

`pctl ∈ [0,1]`, so `pctl >= 1.1` can never fire and the gate must be a
no-op. Run against the **extended** loader, on three slices including one
that spans the Binance→Deribit splice and one that is the whole 2017–2026
series (a loader bug fabricating a value after 2023-12-31 would show up
as a non-zero diff or an out-of-range percentile).

| slice | bars | w=90 | w=180 | w=365 | expanding |
|---|---|---|---|---|---|
| 2020-06-01..2022-12-31 (R-35's own) | 271,585 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2023-01-01..end (spans the splice) | 379,881 | 0.0 | 0.0 | 0.0 | 0.0 |
| full series 2017-2026 | 1,010,889 | 0.0 | 0.0 | 0.0 | 0.0 |

**max|Δtarget| = 0.000e+00 in all 12 cells — PASS, bit-identical.**
Observed `funding_pctl` ranges stay inside [0.001, 1.000] everywhere.
`pytest` on the baseline commit: **439 passed**.

## 2. Causality probe — two-opposite-tampers, plus two extra leak checks

Unregistered strategies get no `test_causality_strict.py` coverage, so
this is run by hand. OHLC × 3 in one copy / ÷ 3 in the other from a cut
point onward, volume × 7 / ÷ 7; every prepared column **and** every order
decision at or before the cut must be bit-identical. Checked at 9 bars
before the cut (1 to 5,000 back). Both `_funding_percentile` code paths
(`.rolling` and `.expanding`) checked separately, on three slices.

| probe slice | bars | cut at | w=90 | w=180 | w=365 | expanding |
|---|---|---|---|---|---|---|
| Deribit region (2024-03 → 2026-08) | 257,481 | 2026-06-03 | PASS | PASS | PASS | PASS |
| spans the splice (2022-06 → 2024-12) | 271,873 | 2024-10-22 | PASS | PASS | PASS | PASS |
| R-35's own region (2020-06 → 2022-12) | 271,585 | 2022-10-22 | PASS | PASS | PASS | PASS |

max|Δ| before the cut = **0.000e+00** for `target`, `v4_target`,
`funding_pctl` and `gated` in all 12 cells; no order decision changed.

**Full-series-statistic check (the failure mode this project has been
bitten by specifically).** A global mean/std/quantile applied to early
rows would make an early rank depend on data the series does not yet
contain. Truncating the funding series must therefore leave every
already-computed rank untouched:

| measured window | settlements dropped | max\|Δpctl\| (all four windows) |
|---|---|---|
| 2021-01-01..2022-12-31 | all ≥ 2023-01-01 | 0.0 |
| 2023-01-01..2023-12-31 | all ≥ 2024-01-01 | 0.0 |
| 2024-01-01..2024-12-31 | all ≥ 2025-01-01 | 0.0 |
| 2025-01-01..2025-12-31 | all ≥ 2026-01-01 | 0.0 |

**Independence check.** Multiplying *every* price bar by 5 over the whole
holdout leaves `funding_pctl` unchanged (max|Δ| = 0.0) — funding is not
reachable from price, which is the entire point of using it (R-16/R-35).

**No lookahead detected on any axis.** Nothing below is explained by a
leak; the gate is doing exactly what it was built to do and losing money
doing it.

## 3. Funding coverage, gate-fire rate, and the venue split

Series: R-35's is 4,383 Binance settlements (2020-01-01 → 2023-12-31);
the extended one is 7,267 (2020-01-01 → 2026-08-19), split
**binance = 4,383 / deribit = 2,884**.

Annualized cost to a constant long, by year (extended series):

| year | annualized | positive | n | source |
|---|---|---|---|---|
| 2020 | +17.20% | 86% | 1,098 | binance |
| 2021 | +30.63% | 93% | 1,095 | binance |
| 2022 | +4.17% | 78% | 1,095 | binance |
| 2023 | +7.87% | 90% | 1,095 | binance |
| 2024 | +10.30% | 84% | 1,098 | deribit |
| 2025 | +5.41% | 79% | 1,095 | deribit |
| 2026 YTD | +1.59% | 65% | 691 | deribit |

Coverage and gate-fire rate, `w=180`, `decile=0.90`:

| slice | bars | funding-covered | gate fires (all bars) | gate fires (covered bars) | Binance-sourced | Deribit-sourced |
|---|---|---|---|---|---|---|
| inner-train 2017-2020 | 420,481 | 24.3% | 2.9% | 11.9% | 25.0% | 0.0% |
| inner-validation 2021-2022 | 209,953 | 100.0% | 8.3% | 8.3% | 100.0% | 0.0% |
| **holdout 2023-01-01 → end** | **379,881** | **100.0%** | **15.9%** | **15.9%** | **27.7%** | **72.3%** |
| — holdout 2023 only | 104,833 | 100.0% | 26.2% | 26.2% | 100.0% | 0.0% |
| — holdout 2024-2026 | 274,761 | 100.0% | 11.8% | 11.8% | 0.0% | 100.0% |

This is the round's headline structural gain: R-35's holdout had funding
for **one** year and the gate was definitionally inert for the other 2.6;
here the holdout is **100% covered**, 3.6× longer, and **72% of it is
funding data this project had never seen**. The inner-train (24.3% /
2.9%) and inner-validation (100% / 8.3%) rows reproduce R-35's §0 table.

Gate-fire rate by calendar year — the splice-seam check:

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| 11.8% | 12.9% | 4.0% | 26.4% | 15.7% | 10.6% | 7.6% |

There is no discontinuity at the 2024 splice — 2024's 15.7% sits between
2023's 26.4% and 2025's 10.6%, and the trend is monotone with the
compressing funding level. The seam is visible but not violent, and §9
tests it directly anyway.

## 4. Reproducing R-35's inner-validation numbers with the extended loader

Pre-registered as a stop condition: 2021–2022 funding is Binance-sourced
in **both** loaders, so these must be unchanged. They are — to the dollar,
in all 20 cells, and the two arms (extended series vs R-35 series) were
also run side by side and are **identical in every one of 16 paired cells**.

| split | market | strategy | final | trades | DD | Sharpe | R-35 report |
|---|---|---|---|---|---|---|---|
| inner-train | futures5x | v4 | $30,344 | 72 | 35.3% | 2.28 | matches |
| inner-train | futures5x | gate(w=90) | $30,899 | 101 | 35.3% | 2.41 | matches |
| inner-train | futures5x | gate(w=180) | $30,222 | 103 | 35.3% | 2.36 | matches |
| inner-train | futures5x | gate(w=365) | $29,129 | 97 | 35.3% | 2.32 | matches |
| inner-train | futures5x | gate(expanding) | $29,129 | 97 | 35.3% | 2.32 | matches |
| inner-train | spot | v4 | $18,477 | 72 | 43.3% | 2.03 | matches |
| inner-train | spot | gate(w=90) | $19,575 | 101 | 43.3% | 2.18 | matches |
| inner-train | spot | gate(w=180) | $18,737 | 103 | 43.3% | 2.12 | matches |
| inner-train | spot | gate(w=365) | $18,171 | 97 | 43.3% | 2.08 | matches |
| inner-train | spot | gate(expanding) | $18,171 | 97 | 43.3% | 2.08 | matches |
| **inner-validation** | futures5x | v4 | $1,064 | 52 | 32.3% | 0.25 | matches |
| **inner-validation** | futures5x | gate(w=90) | $1,564 | 89 | 20.0% | 1.02 | matches |
| **inner-validation** | futures5x | gate(w=180) | $1,238 | 80 | 26.3% | 0.54 | matches |
| inner-validation | futures5x | gate(w=365) | $1,009 | 63 | 32.3% | 0.15 | matches |
| inner-validation | futures5x | gate(expanding) | $1,088 | 65 | 32.3% | 0.29 | matches |
| inner-validation | spot | v4 | $998 | 52 | 33.2% | 0.14 | matches |
| inner-validation | spot | gate(w=90) | $1,380 | 89 | 21.7% | 0.78 | matches |
| inner-validation | spot | gate(w=180) | $1,099 | 80 | 28.3% | 0.31 | matches |
| inner-validation | spot | gate(w=365) | $927 | 63 | 33.2% | −0.02 | matches |
| inner-validation | spot | gate(expanding) | $969 | 65 | 33.2% | 0.06 | matches |

The loader is doing what it says. Nothing downstream is a loader artifact.

## 5. THE HOLDOUT READ — the pre-registered decision

Frozen configuration: `FundingGateDecile(funding_window_days=180,
decile=0.90)`, the R-35 recommendation (middle of the four pre-registered
sweep points, matching v3/v4's own 180-day anchor convention — **not**
chosen by searching for the best Sharpe). Period 2023-01-01 →
2026-08-12 (the last committed OHLCV bar; funding runs to 2026-08-19).
Primary market futures 5×, **funding charged as a first-class cost**
(the `funding_study.py` convention, `funding=` on the engine).
Paired stationary block bootstrap: 30-day mean block, 2,000 resamples,
**identical resample indices for both arms**, on daily returns,
n = 1,319 days. R-29/R-30 convention.

### Point estimates

| market / cost regime | strategy | final | trades | DD | Sharpe | mean exposure | funding paid | fees |
|---|---|---|---|---|---|---|---|---|
| **futures5x, funding CHARGED** | kelly_regime_v4 | **$3,868** | 51 | 34.9% | **1.18** | 0.686 | $794 | $224 |
| **futures5x, funding CHARGED** | **gate w=180 (FROZEN)** | **$1,617** | 134 | 44.3% | **0.59** | 0.503 | $245 | $311 |
| futures5x, funding-free | kelly_regime_v4 | $4,901 | 51 | 33.0% | 1.36 | 0.686 | — | $265 |
| futures5x, funding-free | gate w=180 | $1,820 | 134 | 40.3% | 0.70 | 0.503 | — | $335 |
| spot (diagnostic) | kelly_regime_v4 | $3,373 | 51 | 27.8% | 1.22 | 0.686 | — | $310 |
| spot (diagnostic) | gate w=180 | $1,438 | 134 | 38.6% | 0.52 | 0.503 | — | $431 |
| context | buy_and_hold spot | $3,839 | 1 | 54.0% | 1.03 | — | — | $1 |

### Paired bootstrap, gate(180) − v4

| cell | Δ log growth [95% CI] | Δ Sharpe [95% CI] | Δ max DD pp [95% CI] |
|---|---|---|---|
| **futures5x, funding charged (DECISION)** | **−0.872 [−1.701, −0.166]** \* | **−0.582 [−1.149, −0.042]** \* | +12.17 [−6.99, +19.56] |
| futures5x, funding-free | −0.991 [−1.851, −0.250] \* | −0.648 [−1.217, −0.110] \* | +9.79 [−6.39, +19.33] |
| spot | −0.853 [−1.620, −0.211] \* | −0.699 [−1.251, −0.163] \* | +11.89 [−4.42, +19.65] |

\* interval excludes zero. `P(gate > v4)` on Δ log growth: **0.009**
(futures charged), 0.004 (futures free), 0.002 (spot).

**Read plainly:** the gate loses to the incumbent by roughly 0.87 in log
growth (a factor of ~2.4 in final balance) over 3.6 years, the interval
excludes zero, and it does so while *also* being deeper in drawdown
despite holding **27% less** on average. That last point matters: the
usual failure mode in this repo is a mechanism that looks good only
because it holds less; here the gate holds less **and** draws down more,
which is the opposite artifact and cannot be explained away by exposure.

Funding charged does not rescue it. The gate does pay a genuinely smaller
funding bill ($245 vs $794 — R-14's "the cost is adversely timed" finding
is real and the gate does dodge some of it), but it gives up far more in
foregone return than the $549 it saves.

## 6. Neighbourhood — w=90 / w=365 / expanding on the same holdout

**`w=180` is the pre-registered decision config. The rows below are
neighbourhood context, not a menu.** Reporting them is required by the
plateau clause of the standing promotion bar; selecting from them after
seeing them would be exactly the goalpost move that produced R-12's
28-of-32 in-sample / 0-of-28 out-of-sample. Same holdout, futures5x,
funding charged, same 2,000 shared resamples.

| config | fires | final | DD | Sharpe | exposure | Δ log growth [95% CI] | Δ Sharpe [95% CI] | Δ max DD pp [95% CI] |
|---|---|---|---|---|---|---|---|---|
| kelly_regime_v4 | — | $3,868 | 34.9% | 1.18 | 0.686 | — | — | — |
| gate w=90 | most | $1,413 | 45.2% | 0.46 | 0.530 | −1.007 [−1.818, −0.277] \* | −0.702 [−1.203, −0.201] \* | +13.21 [−6.23, +20.43] |
| **gate w=180 (DECISION)** | | **$1,617** | 44.3% | **0.59** | 0.503 | **−0.872 [−1.701, −0.166]** \* | **−0.582 [−1.149, −0.042]** \* | +12.17 [−6.99, +19.56] |
| gate w=365 | | $2,514 | 31.1% | 0.98 | 0.532 | −0.431 [−1.167, +0.142] | −0.216 [−0.675, +0.195] | −1.29 [−10.27, +7.09] |
| gate expanding | least | $3,333 | 35.1% | 1.10 | 0.670 | −0.149 [−0.490, +0.134] | −0.093 [−0.303, +0.108] | +0.19 [−5.06, +6.69] |

**All four point estimates are negative on both axes.** The ordering is
not a plateau, it is a **dose-response curve running the wrong way**: the
more the gate intervenes, the worse it does, and the configuration that
intervenes least (expanding, which fires rarely because the Deribit years
rank low against the 2020–2021 Binance peak) is the one closest to just
being v4. That is what a mechanism with no edge and non-zero cost looks
like. It is also the exact inverse of R-35's inner-validation ordering,
where the short windows were the strong ones — the in-sample "plateau"
R-35 found from 30–250 days does not merely fail to replicate, it flips.

## 7. Cost checks

### 0.40% taker tier (the pre-registered secondary falsification)

| market / tier | strategy | final | DD | Sharpe | fees |
|---|---|---|---|---|---|
| futures5x @ 0.05% (default), funding charged | kelly_regime_v4 | $3,868 | 34.9% | 1.18 | $224 |
| futures5x @ 0.05%, funding charged | gate w=180 | $1,617 | 44.3% | 0.59 | $311 |
| **futures5x @ 0.40%, funding charged** | kelly_regime_v4 | **$2,365** | 42.4% | **+0.83** | $1,313 |
| **futures5x @ 0.40%, funding charged** | **gate w=180** | **$567** | **71.7%** | **−0.38** | $1,483 |
| spot @ 0.10% | kelly_regime_v4 | $3,373 | 27.8% | 1.22 | $310 |
| spot @ 0.10% | gate w=180 | $1,438 | 38.6% | 0.52 | $431 |
| **spot @ 0.40%** | kelly_regime_v4 | **$2,445** | 34.1% | **+0.94** | $1,027 |
| **spot @ 0.40%** | **gate w=180** | **$684** | **62.5%** | **−0.27** | $1,211 |

**Fails the 0.40% tier outright.** The gate turns 51 trades into 134
(2.6×) — the binary override fires on 15.9% of bars and each entry/exit
of the gate is a full round trip on top of v4's own deadband-latched
turnover. At the entry tier that turnover costs more than the gate's
entire funding saving, and the arm goes Sharpe-negative while v4 stays
comfortably positive. This is the L-14/L-15/L-16 and R-34 failure mode
(a fast signal re-trading on wiggles a slow deadband cannot absorb),
arriving on the axis R-35's own pre-registration flagged as risk (b).

### Exposure-artifact check (R-35's own most important check)

The standing rule (L-04/R-33, R-28/R-31, R-32): match risk before
comparing. Mean |target| and a **flat rescale of v4 to the gate's exact
mean exposure**, same period, same market, funding charged:

| period | arm | mean\|target\| | ratio to v4 | final | DD | Sharpe |
|---|---|---|---|---|---|---|
| **holdout 2023+** | kelly_regime_v4 | 0.6858 | 1.000 | $3,868 | 34.9% | 1.18 |
| holdout 2023+ | **v4 × 0.734** (matched control) | 0.5031 | 0.734 | **$2,959** | **27.2%** | **1.21** |
| holdout 2023+ | **gate w=180** (actual) | 0.5031 | 0.734 | **$1,617** | **44.3%** | **0.59** |
| inner-val 2021-22 | kelly_regime_v4 | 0.2894 | 1.000 | $887 | 34.7% | −0.06 |
| inner-val 2021-22 | **v4 × 0.848** (matched control) | 0.2454 | 0.848 | $934 | 30.8% | −0.00 |
| inner-val 2021-22 | **gate w=180** (actual) | 0.2454 | 0.848 | $1,116 | 27.1% | +0.34 |

Paired bootstrap of gate vs its **matched-exposure** control:

| period | Δ log growth [95% CI] | Δ Sharpe [95% CI] | Δ max DD pp [95% CI] |
|---|---|---|---|
| **holdout 2023+** | −0.604 [−1.286, +0.033] | **−0.612 [−1.196, −0.062]** \* | **+19.27 [+1.85, +29.75]** \* |
| inner-val 2021-22 | +0.176 [−0.134, +0.490] | +0.353 [−0.228, +0.969] | −3.38 [−14.49, +9.52] |

This is the sharpest single result in the report and it deserves to be
read carefully. R-35's most important finding was that its gate **beat**
a matched-exposure rescale of v4 — the check that ruled out the
exposure-level artifact and is why R-35 earned a holdout read at all.
That check is reproduced here exactly on inner-validation (the gate does
beat its control, +0.35 Sharpe, though the interval contains zero once a
bootstrap is put on it, which R-35 did not do for this cell).

On the holdout the same check runs **backwards, significantly**: the gate
is worse than a matched-exposure v4 by 0.61 Sharpe (CI excludes zero) and
**19.3 percentage points of drawdown** (CI excludes zero). The gate's
timing is not neutral — it is actively harmful. A scalar de-lever of v4 to
the identical average exposure would have returned $2,959 with a *smaller*
drawdown than v4 itself; the gate returned $1,617 with a much larger one.

## 8. Sub-period split — 2023 (already seen) vs 2024-2026 (genuinely new)

Futures5x, funding charged, same bootstrap convention.

| sub-period | funding source | strategy | final | trades | DD | Sharpe | exposure | funding paid |
|---|---|---|---|---|---|---|---|---|
| 2023 only | Binance | kelly_regime_v4 | $2,393 | 13 | 28.4% | 2.14 | 0.893 | $152 |
| 2023 only | Binance | gate w=180 | $2,121 | 20 | 27.0% | 2.37 | 0.558 | $53 |
| 2024-01-01 → end | Deribit | kelly_regime_v4 | $1,558 | 39 | 34.9% | 0.67 | 0.601 | $251 |
| 2024-01-01 → end | Deribit | gate w=180 | $741 | 114 | 43.8% | −0.27 | 0.478 | $87 |

| sub-period | n days | Δ log growth [95% CI] | Δ Sharpe [95% CI] | Δ max DD pp [95% CI] | P(gate>v4) |
|---|---|---|---|---|---|
| **2023 only** (Binance, what R-35 saw) | 364 | −0.120 [−0.430, +0.139] | +0.066 [−0.557, +0.734] | −1.35 [−9.53, +2.86] | 0.209 |
| **2024-2026** (Deribit, genuinely new) | 954 | **−0.746 [−1.466, −0.097]** \* | **−0.955 [−1.620, −0.205]** \* | +11.62 [−6.20, +24.25] | 0.012 |

**This is the interpretive centre of the round.** The 2023 row reproduces
R-35's own ledger table *exactly* — v4 $2,393, gate $2,121, Δ log growth
−0.120, CI [−0.430, +0.139], P = 0.209 — which is a strong independent
confirmation that this harness is measuring the same thing the operator
measured in R-35, and confirms that R-35's holdout read was a genuine
null rather than a hidden win. Every bit of the new negative comes from
2024–2026, the funding data no one had seen.

So the answer to the pre-registration's own framing is the unfavourable
one: the edge does **not** live only in 2023 (where it is a null); it is
*absent* in 2023 and *negative* afterwards.

### Why it failed — descriptive, no backtests

R-16's premise is that top-decile funding forecasts weak forward returns
— its measured 14-day Q1−Q5 spread was **+3.57pp**, i.e. the *lowest*
funding quintile beat the *highest* by that much. Mean forward 14-day log return on
gated vs ungated bars, using the identical causal percentile column the
gate itself uses:

| slice | gated bars | ungated bars | spread | n gated |
|---|---|---|---|---|
| inner-validation 2021-2022 | −0.0157 | −0.0135 | **−0.0022** | 17,508 |
| holdout 2023 (Binance) | +0.0390 | +0.0350 | +0.0040 | 27,457 |
| holdout 2024-2026 (Deribit) | +0.0255 | +0.0028 | **+0.0227** | 32,544 |
| holdout 2023 → end (all) | +0.0317 | +0.0107 | +0.0210 | 60,288 |

The premise holds, weakly and with the right sign, on inner-validation
(−0.22pp), and **inverts on the holdout**. Over 2024-2026 the bars where
funding sat in its top decile were followed by *+2.55%* over the next 14
days against *+0.28%* elsewhere — a **+2.27pp spread in the wrong
direction**. The gate stood flat during the best part of the tape.
Mechanistically that is the whole story: in a bull leg where funding is
rich because the market is trending up, "stand aside when funding is
rich" is "stand aside when the trend is working."

## 9. Was it the venue splice? — post-hoc robustness, cannot rescue the verdict

Not part of the decision rule, run because the seam described in §0 is a
legitimate thing for a reader to suspect. Identical frozen config,
futures5x, funding charged.

| variant | strategy | final | Sharpe | Δ log growth [95% CI] | Δ Sharpe [95% CI] |
|---|---|---|---|---|---|
| 2024+ / spliced series (as pre-registered) | gate w=180 | $741 | −0.27 | −0.746 [−1.466, −0.097] \* | −0.955 [−1.620, −0.205] \* |
| **2024+ / pure Deribit (no splice at all)** | gate w=180 | $721 | −0.29 | **−0.773 [−1.421, −0.211]** \* | **−0.964 [−1.507, −0.337]** \* |
| 2025+ / spliced, >1 full 365d window past the splice | gate w=180 | $702 | −0.83 | −0.190 [−0.558, +0.094] | −0.574 [−1.432, +0.190] |

Ranking Deribit rates only against Deribit history — no cross-venue
comparison anywhere — gives the **same** negative, marginally stronger.
The 2025+ slice (where every swept lookback is entirely past the splice)
keeps the negative point estimates, with wider intervals on 588 days as
expected. **The seam is not the explanation.**

## 10. Configurations evaluated

Counted by the harness itself (`N_EVALUATED`, incremented once per
backtest that produces a performance number):

| command | evaluations |
|---|---|
| §4 validation (2 splits × 2 markets × 9 arms) | 36 |
| §5 holdout (3 cells × 2 arms + buy_and_hold) | 7 |
| §6 neighbourhood (v4 + 4 windows) | 5 |
| §7 costs (4 tier cells × 2 arms + 2 exposure blocks × 3) | 14 |
| §8 sub-periods (2 periods × 2 arms) | 4 |
| §9 seam robustness (3 variants × 2 arms) | 6 |
| **total backtest evaluations** | **72** |

Of the 72, **61 are distinct (config × period × market × cost-regime)
cells**; the other 11 are deliberate re-runs of an already-counted cell
(e.g. the decision cell appears again inside §6 and §7). Four further
`run_period_funding` calls inside §7's paired-bootstrap block re-run
configurations already counted and are not double-counted.

Not in the tally, because they generate no Sharpe: 12 prepare-only build
checks (§1), 12 tamper-probe preparations × 2 copies + 16 truncation
probes + 1 independence probe (§2), the coverage tables (§3), and the
forward-return description (§8).

**Report 72 for the deflated-Sharpe trials count** (the conservative
choice). Nothing here needs a deflated-Sharpe calculation to be rejected —
the raw, undeflated, un-trials-adjusted result already fails.

Holdout consultations: this branch read the 2023+ holdout. Counting the
same way the ledger does (one per strategy × period × cost-regime cell
touched), that is **61 distinct cells**, of which the pre-registration
authorised **one consultation** for the frozen `w=180` full-window read.
The honest number is larger than one and the operator should record it as
such: §6–§9 are all holdout reads, taken *after* the decision cell had
already returned a significant negative, and none of them could have
changed the verdict in the gate's favour. They are diagnostics on a
rejection, not a search for a survivor — but they are still reads, and
this file's counter should go into the ledger rather than the
pre-registration's optimistic "one."

## 11. Verdict against the pre-registered decision rule, clause by clause

> "Δ log growth or Δ Sharpe 95% CI that **excludes zero in the gate's
> favor**"

**FAILS.** Both intervals exclude zero **against** the gate:
Δ log growth −0.872 [−1.701, −0.166]; Δ Sharpe −0.582 [−1.149, −0.042].

> "drawdown not worse"

**FAILS.** +12.2pp worse point estimate (interval contains zero), and
+19.3pp worse than a matched-exposure v4 with an interval that
**excludes** zero.

> "survival of the 0.40% taker tier"

**FAILS.** Sharpe −0.38 (futures) / −0.27 (spot) at 0.40%, against v4's
+0.83 / +0.94.

> "Anything else is NEGATIVE and closes B-05 permanently (not 'pending
> more data' — this round is the more data)."

**B-05 is closed.** The decision rule was not moved, the configuration was
not retuned, the decile stayed at 0.90, and the sweep set stayed as
pre-registered. The one year R-35 had was not unlucky — it was the *least*
bad year available; the 2.6 years R-35 could not see are where the
mechanism actually breaks, and it breaks for a reason (§8) rather than by
noise.

### What survives, and what a future session should not over-read

1. **R-35 was not wrong about its own numbers.** Every inner-validation
   figure reproduces exactly (§4), and the 2023 holdout cell reproduces
   R-35's ledger table to the dollar (§8). This is not a "the report was
   simply wrong" outcome; it is a genuine in-sample effect that did not
   generalize — the R-12 / R-28 pattern, hit for the fourth time, this
   time with the extra year of data that was supposed to settle it.
2. **The COST finding from R-35 stands and is worth keeping.** Funding is
   real, adversely timed (R-14), and the gate genuinely cuts the bill
   ($245 vs $794 on the holdout — a 69% reduction while standing aside on
   only 15.9% of bars). What is now established is that *paying that bill
   is worth it*: the exposure the gate declines to hold earns far more
   than the funding it avoids. That is a useful, specific negative for
   any future COST-axis idea on this strategy family.
3. **The plateau claim was the weak point, and it was flagged as such.**
   R-35's own report said honestly that 365/expanding — two of the four
   mandated sweep points — did not show the effect. On the holdout the
   ordering inverts entirely (§6). A "plateau in sign over 30–250 days,
   with the two mandated long windows failing" should in hindsight have
   been read as the sweep falsifying itself, not as a partial pass.
4. **Do not add this to section C's ruled-out list as "funding is
   useless."** What is ruled out is specifically *a binary top-decile
   flat gate keyed on trailing funding percentile, layered on
   `kelly_regime_v4`*, on this instrument, at this cadence, with this
   turnover. The measurement in §8 (funding's forward-return sign flipped
   between 2021-22 and 2024-26) is itself the interesting residue: it says
   R-16's descriptive quintile relationship is **regime-dependent**, and
   the regime it was measured in (2020-2023, on the same data R-16 used)
   is not the regime that followed.

## 12. Things in my own work the operator should distrust

- **The arm is an import, not a copy.** I consider this strictly safer
  than re-typing R-35's module, and `identity_check()` proves the
  functions are the same objects — but it *is* a deviation from a literal
  reading of "create a file containing the logic." If the operator wants
  the copy, the numbers will not change (the injected series is the only
  difference), but the check is worth re-running.
- **`run_period_funding` is my own function**, not `tradebot.window.run_period`
  (which does not accept `funding=`). It mirrors `run_period` exactly plus
  the `funding=` passthrough, and its correctness is indirectly confirmed
  by §4 (20/20 cells match R-35's report, which used the framework path)
  and §8 (the 2023 cell matches the operator's own R-35 holdout numbers to
  the dollar). Still, it is a hand-rolled harness in the load-bearing
  position, and that is exactly where this project has been bitten.
- **`funding_paid` on a trimmed result.** I claim it is in-period because
  `trade_start` keeps the account flat through the prefix and
  `PaperBroker.apply_funding` returns early at `pos == 0`. I read that
  code but did not write a test for it. If it were wrong, the
  funding-charged columns would be overstated for both arms equally, so
  it cannot flip the verdict — but the dollar figures would move.
- **I read the holdout more than the pre-registration's "one"
  consultation** (§10). Every one of those extra reads happened after the
  decision cell had already come back significantly negative, and none
  could have promoted the gate — but the honest count is 61 cells, not 1,
  and the ledger's counter should reflect that.
- **§9's pure-Deribit variant and §8's forward-return table were not
  pre-registered.** They are diagnostics I chose after seeing the
  negative, to test the two most plausible "this is an artifact"
  objections. They cannot support a promotion and I am not using them to;
  a reader who wants only pre-registered content should stop at §7.
- **Checkpoint commits appeared in the repo during this session** (from
  the operator's harness, not from me — I ran no `git add`/`commit`/`push`).
  I mention it only so the operator is not surprised to find this file
  already tracked.
- **One thing genuinely surprised me** and is worth a second pair of eyes:
  the gate holds 27% *less* exposure than v4 on the holdout yet draws
  down 12pp *more*, and 19pp more than an exposure-matched v4 with the
  interval excluding zero. That is a large, unusual effect in the
  direction opposite to this project's usual artifact. My reading is §8's
  mechanism (it stands aside through the strongest legs, then re-enters
  into the pullbacks, and pays 2.6× the turnover to do it), but a skeptic
  should confirm that rather than take it from me.
