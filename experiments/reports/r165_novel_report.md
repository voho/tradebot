# R-165 NOVEL branch — derived-rate EWMA smoothing of `kelly_regime_v4`'s `scale`

**Verdict: REJECT (well-documented negative).**
Branch files: `experiments/r165_novel_ewma.py` (implementation + derivation +
sweep + holdout driver), this report. Pre-registration:
`experiments/r165_shared.py` (read-only, unedited).

---

## 1. Mechanism (one sentence)

Replace v4's raw `scale[i]` with an exponentially smoothed
`eff_scale[i] = (1-a)*eff_scale[i-1] + a*scale[i]`, at a rate `a` derived
from a growth-cost/fee-cost trade-off measured on inner-train alone, and feed
`desired[i] = frac[i] * eff_scale[i]` into v4's own unchanged deadband
position update.

Everything else is `KellyRegimeV3.prepare` copied byte-for-byte: the three
latched anchor votes (`frac`), the `vol`/`slow` series, the `full`/`steady`
legs, the high/low breakout hysteresis `state` machine, and
`if abs(desired - pos) > deadband: pos = desired`. Implemented as
`KellyRegimeV4EwmaScale(KellyRegimeV4)`, **not** `@register`ed (unregistered
experiment, stays out of CI and the comparison table).

Two disclosed implementation details:

- The class adds an `__init__` that carries the rate (`ewma_a`) and does
  nothing else; the dispatch asked for "only overrides `prepare()`" and this
  is the minimum needed to parameterise it. `ewma_a = 1.0` reproduces
  `kelly_regime_v4`'s target path **bit-for-bit** (checked in the script:
  max abs diff `0.000e+00`, `np.array_equal` True).
- The EWMA is seeded at the first bar whose raw `scale` is strictly positive,
  so the recursion does not spend its first half-lives climbing out of the
  zero the warmup leaves behind. That seeding uses only bar `i`'s own value
  and is causal; a truncation probe on the frozen config confirms
  `targets(full)[:cut] == targets(truncated)` exactly (max abs diff
  `0.000e+00`).

## 2. The derivation, and the resulting `a*`

No backtest number enters the derivation. Inputs are v4's own fee rate from
`MarketSpec.spot()`, `BARS_PER_DAY`/`BARS_PER_YEAR` from
`tradebot.strategies.kelly_regime`, and four statistics measured on
**inner-train BTC only** (2017-01-01 → 2020-12-31, 420,192 warm bars).

This generalizes L-05/L-06 (docs/RESEARCH.md finding 7, "the deadband should
be derived, not chosen") from a *discrete band width* to a *continuous
smoothing rate*, which is Dao et al. (2016)'s single-signal framing: a finite
trading rate trades tracking error against turnover cost.

With `x_t = scale[t]`, `y_t = eff_scale[t]`, `e_t = y_t - x_t`, position
`f = frac*y` and frictionless target `f* = frac*x`, the two per-bar costs are

```
growth cost(a) = (sigma_bar^2 / 2) * E[frac^2] * Var(e; a)        # L-05/L-06's (sigma^2/2)(f-f*)^2
fee cost(a)    = fee * E[|frac|] * sqrt(2/pi) * sd(dy; a)          # L-05/L-06's fee*|delta f|
```

Modelling `x` locally as a random walk with per-bar innovation variance
`s^2 = Var(x_t - x_{t-1})` gives the two EWMA identities exactly:

```
e_t = (1-a)(e_{t-1} - dx_t)          =>  Var(e; a)  = s^2 (1-a)^2 / (a(2-a))
dy_t = a (x_t - y_{t-1})             =>  Var(dy; a) = s^2 a / (2-a)
```

so the objective on `a in (0,1]` is

```
C(a) = A * s^2 * (1-a)^2 / (a(2-a))  +  B * s * sqrt(a / (2-a))
A = (sigma_bar^2 / 2) * E[frac^2]
B = fee * E[|frac|] * sqrt(2/pi)
```

`C -> inf` as `a -> 0` (unbounded tracking error) and `C(1) = B*s` (v4's own
instant jump: zero tracking error, maximum turnover). Its small-`a`
stationary point has the closed form

```
                        a* = ( sqrt(2) * A * s / B )^(2/3)
```

Measured inputs and result:

| quantity | value |
|---|---|
| `fee` (`MarketSpec.spot().fee_rate`) | 0.1000 % |
| `s = sd(d scale)` per bar | 4.888937e-03 |
| `sigma_bar^2 = Var(log ret)` per bar | 8.267550e-06 (annualized sigma 0.9326) |
| `E[|frac|]` | 0.558586 |
| `E[frac^2]` | 0.488170 |
| `A` | 2.017987e-06 |
| `B` | 4.456871e-04 |
| **`a*` (closed form, FROZEN PRIMARY)** | **9.932955e-04 per 5m bar** |
| `a*` (numeric argmin of `C` on a 4e5-point log grid) | 9.927960e-04 |
| `a*` (AR(1) refinement, `phi = 0.99988397` from `scale`'s own 20.7-day causal half-life) | 9.080244e-04 |
| implied EWMA half-life | 697 bars = **2.42 days** |
| `C(1) / C(a*)` | 29.9x |

The AR(1) refinement (same objective, `x` an AR(1) at the measured
persistence instead of a random walk) moves `a*` by −8.6%, far inside one
sensitivity step, so the RW form is used as PRIMARY. At the futures taker
(0.05%) the same formula gives `a*` a factor `2^(2/3) = 1.587` larger, i.e.
inside the sensitivity set; one rate is frozen for both markets as
instructed.

### 2b. Pre-registered falsification test — did it fire?

**No.** `a* = 9.93e-4` is three orders of magnitude away from `a = 1`; the
derived rate is a genuinely different, materially slower tracking rule than
v4's instant jump. No alternative derivation was searched for.

The pre-registration states the test in three non-equivalent ways, and they
do not all agree, so all three are reported:

1. `r165_shared.py`'s "**Falsification test, stated precisely**" (and the
   dispatch's own wording): derived rate indistinguishable from `a=1` at bar
   resolution → **does not fire** (`a* = 9.93e-4`).
2. The decision rule's parenthetical, "*half-life inside the 1–15 day
   band*" → **does not fire**: the measured `vol` half-life is 38.8 days.
3. `order_of_magnitude_gap()`'s own returned `falsification_test_fires`
   field, whose criterion is `ratio < 10` against the 2.5–4.6-day anchors →
   **fires** (ratio 8.43). At the 47.2-day value the pre-registration itself
   asserts, this field would have returned False (ratio 10.26), so this
   reading's outcome flips on a number I could not reproduce (next
   paragraph).

Reading (3) is the only one that fires, it contradicts reading (2) in the
same sentence of the frozen file, and it is the one whose input is
unreproducible. Per ROUTINE Step 4 this is reported as an under-specified
clause rather than resolved after the fact. **The verdict below is REJECT
under every reading**, so nothing turns on the ambiguity.

### 2c. Sanity check of the pre-registered half-life — NOT reproduced

`r165_shared.py` states the `vol` series' causal autocorrelation half-life is
**47.2 days** on inner-train BTC. Calling the file's own
`causal_autocorr_halflife_days(realized_vol_series(close, 8*BARS_PER_DAY))`
at its documented defaults gives **38.77 days** (acf(1d) = 0.9677, n = 1460
days). Variants tried, none of which produce 47.2: unshifted `vol` (38.78),
`vol_span=16d` (37.79), `log(vol)` (34.09); `max_lag_days=30` gives 21.18 and
`max_lag_days=90` gives 39.87, so the fit is lag-window sensitive but never
reaches 47.2 at the shipped default of 60. The qualitative claim survives
(`vol` is 8–13x slower than the 2.5–4.6-day anchors, because it is itself an
8-day EWMA), but the specific number in the pre-registration does not
reproduce and is recorded here as such.

## 3. Configurations evaluated

**26 candidate cells**, each paired against a `kelly_regime_v4` control run
on the identical window (control runs are the fixed incumbent, not trials):

| block | cells |
|---|---|
| sensitivity sweep: 5 rates × 2 markets × 2 inner slices | 20 |
| D1 holdout, PRIMARY × 2 markets | 2 |
| D2 holdout at the 0.40% tier, PRIMARY × 2 markets | 2 |
| D3 ETH-A, PRIMARY × 2 markets | 2 |
| **total (this branch)** | **26** |

The turnover diagnostic in §5 re-computes `prepare()` for the same 5 rates on
inner-train; it runs no backtest and adds no cell. **Zero configurations were
selected on the holdout**: `a*` was frozen by §2's derivation before any
backtest existed, and the sweep is reported for plateau/robustness only.

## 4. Inner-train / inner-validation (never used to pick a rate)

`cand$`/`ctrl$` from $1,000; `dSh` = candidate Sharpe − v4 Sharpe; `fills` =
`len(result.fills)` (turnover), `trades` = `Metrics.num_trades` (round-trip
episodes) — both carried, per the standing rule; `expR`/`volR` =
time-in-market and realized-vol ratio vs v4 (D0); `dlogG` = paired
stationary-block bootstrap (30-day blocks, 2,000 resamples) of the
log-growth difference.

| config | slice | market | a | cand$ | ctrl$ | dSh | dDD | fills (v4) | expR | volR | RM | dlogG [95%] | excl 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x0.25 | inner_train | spot | 2.48e-04 | 20,979 | 18,477 | −0.11 | −3.7 | 453 (467) | 1.00 | 1.07 | Y | +0.127 [−0.165, +0.425] | no |
| x0.5 | inner_train | spot | 4.97e-04 | 20,559 | 18,477 | −0.08 | −1.8 | 454 (467) | 1.00 | 1.05 | Y | +0.107 [−0.117, +0.347] | no |
| **PRIMARY** | inner_train | spot | 9.93e-04 | 20,414 | 18,477 | −0.04 | −4.5 | 453 (467) | 1.00 | 1.03 | Y | +0.100 [−0.112, +0.351] | no |
| x2 | inner_train | spot | 1.99e-03 | 18,986 | 18,477 | −0.06 | −3.0 | 467 (467) | 1.00 | 1.02 | Y | +0.027 [−0.127, +0.203] | no |
| x4 | inner_train | spot | 3.97e-03 | 18,934 | 18,477 | −0.03 | −3.5 | 485 (467) | 1.00 | 1.01 | Y | +0.024 [−0.133, +0.213] | no |
| x0.25 | inner_train | futures_5x | 2.48e-04 | 27,631 | 30,344 | −0.29 | −1.5 | 250 (261) | 1.00 | 1.10 | Y | −0.094 [−0.514, +0.291] | no |
| x0.5 | inner_train | futures_5x | 4.97e-04 | 28,182 | 30,344 | −0.29 | +1.7 | 263 (261) | 1.00 | 1.11 | **n** | −0.074 [−0.457, +0.312] | no |
| **PRIMARY** | inner_train | futures_5x | 9.93e-04 | 31,467 | 30,344 | −0.19 | +0.2 | 260 (261) | 1.00 | 1.09 | Y | +0.036 [−0.365, +0.430] | no |
| x2 | inner_train | futures_5x | 1.99e-03 | 33,454 | 30,344 | −0.13 | +1.1 | 261 (261) | 1.00 | 1.09 | Y | +0.098 [−0.203, +0.393] | no |
| x4 | inner_train | futures_5x | 3.97e-03 | 29,720 | 30,344 | −0.11 | −1.6 | 261 (261) | 1.00 | 1.04 | Y | −0.021 [−0.340, +0.293] | no |
| x0.25 | inner_val | spot | 2.48e-04 | 853 | 998 | −0.24 | +0.9 | 253 (256) | 1.00 | 1.03 | Y | −0.156 [−0.453, +0.021] | no |
| x0.5 | inner_val | spot | 4.97e-04 | 969 | 998 | −0.04 | +0.7 | 256 (256) | 1.00 | 1.02 | Y | −0.029 [−0.109, +0.033] | no |
| **PRIMARY** | inner_val | spot | 9.93e-04 | 1,051 | 998 | +0.09 | +0.2 | 257 (256) | 1.00 | 1.03 | Y | +0.052 [−0.022, +0.158] | no |
| x2 | inner_val | spot | 1.99e-03 | 1,028 | 998 | +0.06 | +0.1 | 258 (256) | 1.00 | 1.02 | Y | +0.030 [−0.011, +0.086] | no |
| x4 | inner_val | spot | 3.97e-03 | 1,037 | 998 | +0.07 | +0.4 | 257 (256) | 1.00 | 1.02 | Y | +0.039 [−0.021, +0.120] | no |
| x0.25 | inner_val | futures_5x | 2.48e-04 | 856 | 1,064 | −0.35 | +2.7 | 141 (143) | 1.00 | 1.01 | Y | −0.218 [−0.612, +0.061] | no |
| x0.5 | inner_val | futures_5x | 4.97e-04 | 900 | 1,064 | −0.28 | +3.1 | 145 (143) | 1.00 | 1.01 | Y | −0.167 [−0.383, −0.006] | **YES (adverse)** |
| **PRIMARY** | inner_val | futures_5x | 9.93e-04 | 935 | 1,064 | −0.22 | +3.5 | 143 (143) | 1.00 | 1.02 | Y | −0.129 [−0.321, +0.009] | no |
| x2 | inner_val | futures_5x | 1.99e-03 | 999 | 1,064 | −0.10 | +3.4 | 146 (143) | 1.00 | 1.03 | Y | −0.063 [−0.198, +0.052] | no |
| x4 | inner_val | futures_5x | 3.97e-03 | 1,057 | 1,064 | −0.01 | +3.0 | 147 (143) | 1.00 | 0.99 | Y | −0.006 [−0.200, +0.200] | no |

Inner reading, stated before the holdout was opened: **the mechanism is
inert-to-mildly-harmful.** 19 of 20 bootstrap intervals contain zero; the one
that excludes it (x0.5, inner-val futures) does so on the *adverse* side.
PRIMARY is +0.09 Sharpe on inner-val spot and −0.22 on inner-val futures.
Critically, **fill counts barely move**: 453 vs 467, 257 vs 256, 143 vs 143 —
see §5.

## 5. The diagnostic that explains the whole result

The derivation's fee term prices turnover as the mean absolute per-bar move
of the *frictionless desired* exposure, which scales like `sqrt(a)`. v4 does
not trade that path; it trades the deadband-latched one, and `desired` is
`frac * eff_scale`, whose motion is dominated by `frac`'s discrete 0/⅓/⅔/1
flips. Measured on inner-train BTC:

| a multiple | a | model `E|df|`/bar | actual `E|d eff_scale|` | actual `E|d desired|` | actual `E|d target|` | target jumps |
|---|---|---|---|---|---|---|
| ×0.25 | 2.48e-04 | 2.428e-05 | 3.890e-05 | 2.708e-04 | 2.616e-04 | 503 |
| ×0.5 | 4.97e-04 | 3.434e-05 | 5.456e-05 | 2.807e-04 | 2.689e-04 | 521 |
| **×1** | 9.93e-04 | 4.857e-05 | 6.980e-05 | 2.924e-04 | 2.747e-04 | 529 |
| ×2 | 1.99e-03 | 6.871e-05 | 8.525e-05 | 3.019e-04 | 2.828e-04 | 550 |
| ×4 | 3.97e-03 | 9.721e-05 | 9.909e-05 | 3.087e-04 | 2.887e-04 | 571 |

Across a **16x** change in `a`, the model's fee term changes **4.0x** (as
`sqrt` requires) and the realized traded-path turnover changes **1.10x**
(target jumps 1.14x). The smoothed factor supplies only ~14–24% of
`desired`'s motion; the rest is the vote, which the mechanism does not touch,
and the 10% deadband then absorbs most of what is left. **So the fee term the
derivation minimizes against is, empirically, almost independent of `a`** —
the derived optimum is set almost entirely by the tracking-error term, and
the fee saving the mechanism was built to harvest is roughly an order of
magnitude smaller than the model assumes. This is the branch's real finding,
and it was measured, not argued.

## 6. Holdout — read exactly once, at the frozen `a* = 9.932955e-04`

`start = OOS_START = 2023-01-01`, BTC, both markets. One read; no
re-selection. **Holdout consultation:** this branch consumed one, in a single
scripted pass — 4 candidate cells that touch bars at or after `OOS_START`
(D1 spot/futures, D2 spot/futures at the 0.40% tier) plus their 4
`kelly_regime_v4` controls. D3's ETH series ends 2019-12-31 and reads no
holdout bar. The ledger's running counter should be incremented by the
operator when both R-165 branches are consolidated.

### D1 — paired bootstrap vs `kelly_regime_v4` itself

| market | cand $ | v4 $ | Sharpe | v4 Sharpe | dSh | DD | v4 DD | fills | v4 fills | trades | TiM | v4 TiM | vol | v4 vol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spot | 3,572 | 3,373 | +1.25 | +1.22 | +0.03 | 27.7% | 27.8% | 319 | 332 | 51/51 | 0.708 | 0.708 | 0.324 | 0.321 |
| futures_5x | 5,348 | 4,901 | +1.39 | +1.36 | +0.03 | 31.8% | 33.0% | 314 | 328 | 51/51 | 0.708 | 0.708 | 0.393 | 0.385 |

Paired stationary block bootstrap (30-day blocks, 2,000 resamples,
`tradebot.inference.paired_bootstrap`):

| market | d log-growth [95%] | excludes 0 | d Sharpe [95%] | excludes 0 |
|---|---|---|---|---|
| spot | +0.0573 [−0.0206, +0.1538] | no | +0.0385 [−0.0250, +0.1101] | no |
| futures_5x | +0.0873 [−0.0750, +0.2821] | no | +0.0429 [−0.0586, +0.1484] | no |

### D2 — 0.40% fee tier (`scripts/fee_study.py`'s Bitstamp entry taker)

| market | cand $ | v4 $ | dSh | d log-growth [95%] |
|---|---|---|---|---|
| spot | 2,605 | 2,445 | +0.04 | +0.0633 [−0.0165, +0.1636] |
| futures_5x | 3,354 | 2,989 | +0.07 | +0.1154 [−0.0498, +0.3177] |

### D3 — ETH-A falsification (Bitfinex ETH, 2016-03 → 2019-12)

| market | cand $ | v4 $ | Sharpe | v4 Sharpe | dSh | DD | v4 DD | fills | v4 fills | d log-growth [95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| spot | 4,837 | 5,482 | +1.34 | +1.48 | −0.14 | 33.3% | 36.5% | 455 | 448 | −0.1252 [−0.4520, +0.1817] |
| futures_5x | 3,667 | 4,263 | +1.14 | +1.25 | −0.11 | 45.3% | 35.1% | 231 | 214 | −0.1506 [−0.6152, +0.2885] |

## 7. D0–D6 outcomes

| gate | requirement | result |
|---|---|---|
| **D0** risk-match | TiM and realized vol within 10% of v4, per cell | **PASS** — TiM identical to 3 dp (0.708/0.708) on both markets; vol ratios 1.01 / 1.02 |
| **D1** holdout comparison | 95% paired interval excludes zero favourably on log-growth or Sharpe on **both** markets | **FAIL** — all four intervals contain zero. Point estimates are favourable (+0.06/+0.09 log-growth, +0.03/+0.03 Sharpe) and both sit far inside the ±0.2 Sharpe noise floor |
| **D2** cost mechanism | advantage must **grow** at 0.40% | **PASS** — +0.0573→+0.0633 (spot), +0.0873→+0.1154 (futures). Directionally right, but on a difference that is not distinguishable from zero at either tier |
| **D3** ETH-A | same sign, not reversed | **FAIL** — BTC holdout positive on both markets, ETH negative on both (−0.125, −0.151); Sharpe −0.14 / −0.11, and futures max DD *worsens* 35.1% → 45.3%. Per the frozen rule, a reversal here voids the BTC finding |
| **D4** turnover | total fill count must fall vs v4 | **PASS on the letter, fails on the substance** — 319 vs 332 and 314 vs 328 on holdout (−4%), but §5 shows realized turnover is essentially invariant to `a` (1.10x over a 16x rate change), and on ETH fills *rose* (455 vs 448, 231 vs 214) |
| **D5** plateau not peak | neighbours within the ±0.2 Sharpe floor | **FAIL on inner-val, PASS on inner-train** — inner_val/spot: x0.25 sits 0.33 Sharpe below PRIMARY; inner_val/futures: x4 sits 0.21 above. Both inner_train cells are plateaus (max gap 0.11). Moot: D5 only downgrades a PROMOTE |
| **D6** funding | report `funding_study.py` if D1–D5 all pass | **not run** — the frozen rule conditions it on D1–D5 passing; D1, D3 and D5 fail. Futures figures above are the standing funding-free upper bound |

**Decision rule outcome: default REJECT.** The frozen rule requires D1 (both
markets, either metric), D2, D3 and D4 to hold simultaneously for one frozen
config. D1 and D3 both fail. No clause was reinterpreted, no threshold moved,
and the rule partitions cleanly here (the D1-and-D3 failure lands
unambiguously in REJECT).

## 8. Verdict

**REJECT / NEGATIVE.** The derivation is sound and produced a non-degenerate
rate (2.42-day half-life, 3 orders of magnitude from v4's instant jump), the
pre-registered falsification test did **not** fire under the reading the
pre-registration states precisely, and the mechanism is bit-for-bit identical
to v4 at `a=1` and causally clean. It still does not work: every holdout
interval contains zero, ETH reverses the sign on both markets, and the
turnover saving the whole cost-axis argument rests on is ~4% instead of the
~4x the model prices.

This is a mechanically clean confirmation, on the *other* factor of
`desired = frac * scale`, of R-64's conclusion about the product as a whole —
reached here by a different route: R-64's partial adjustment collapsed
because the anchors' decay rates were too similar; this one collapses because
the factor being smoothed is not where the turnover is. The scope correction
in `r165_shared.py` correctly ruled out R-64's *weight-collapse* mechanism as
inapplicable to a single scalar signal, and it was right to: this branch
failed for an entirely different reason.

**One-line lesson.** Smoothing `scale` cannot buy back fees, because with
v4's 10% deadband in place the traded path's turnover is set by the vote's
discrete flips, not by `scale`'s drift — a 16x change in the smoothing rate
moves realized turnover by 10%, so a derivation that prices turnover as
`sqrt(a)` is minimizing a fee term the strategy does not actually pay.

**Reproduce:** `python experiments/r165_novel_ewma.py all` (derivation +
causality/identity kill switches + 20-cell sweep + turnover diagnostic + D5,
then the holdout block). `derive`, `inner` and `holdout` run the stages
separately; `inner` never touches a bar at or after `OOS_START`.
