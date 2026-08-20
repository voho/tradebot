# R-53 (conservative branch) — macro-stress brake on `kelly_regime_v4` (08-20)

Unregistered experiment. Code: `experiments/kelly_regime_v14_macro_brake.py`.
Not `@register`ed, not auto-discovered, nothing committed by this session.
`kelly_regime_v4` is imported and called unmodified throughout (only via
its byte-for-byte reproduced `prepare()` and via `get_strategy` for
control runs); `experiments/_macro_signal.py` is imported unchanged and
never edited.

## 1. Idea and mechanism, one sentence

Reproduce `kelly_regime_v4`'s vote and conditional-vol-target scale
byte-for-byte, and apply a single bounded, **never-increase-only**
multiplicative haircut `mult = 1 - lam * clip(stress_z / z_scale, 0, 1)`
(so `mult` ranges `[1-lam, 1]`) on top of v4's own target exposure
whenever `stress_z` — this round's new causal composite of VIX level and
20-day DXY momentum, both z-scored on trailing 365-day windows,
`experiments/_macro_signal.py` — is elevated.

## 2. Constraint attacked, and why this is not a duplicate

**INFO** — the project's #1 standing-diagnosis constraint. This is the
third attack on it that is not a transform of the incumbent price series
(after R-41's Deribit basis and R-44's on-chain confirmation), and the
first from data describing the broader financial system (equity fear,
dollar strength) rather than BTC's own market or network.

Not a duplicate of:

- **L-01** (`kelly_regime_v4`) — the strategy this file wraps unchanged;
  every inherited parameter keeps its exact v4 default.
- **L-12** (`harsanyi_crowd`) — a price-derived belief-margin *direction*
  signal that lost. Its architectural cousin R-34 tested L-12's own
  stated hypothesis (feed the same price-derived posterior as a SIZE
  dampener) and it collapsed into an exposure-level artifact — the exact
  failure mode this branch checks for explicitly below, on a genuinely
  different, price-independent input.
- **R-08 / R-10** — R-08's better volatility forecast made results worse
  by de-levering into BTC's own high-vol/high-forward-Sharpe states
  (R-10's inverse-leverage finding). This file's signal is not a
  volatility forecast of BTC itself and never conditions on BTC's own
  realized or implied vol; the never-increase design is a deliberate,
  one-sided risk brake, not a reworked version of R-08's trap (module
  docstring, "why never-increase-only" section).
- **B-07 / R-44** (`kelly_regime_v10_onchain_confirm.py`) — same house
  style and the same INFO constraint, but a different data source
  (blockchain activity, specific to the traded asset's own network) and a
  different architecture (symmetric, can raise exposure).
- **R-41** (`kelly_regime_v9_basis_brake.py`) — the identical
  never-increase-only architectural template this file copies, but a
  different data source: basis is a second, independently *traded* price
  series; VIX/DXY are not traded BTC/ETH instruments and are not derived
  from any BTC/ETH price series at all.
- A sibling agent runs a structurally different exploitation of the same
  shared `stress_z` signal (feeding it into the regime vote itself) in
  parallel this round, on a disjoint file. Not read or coordinated with
  here.

## 3. Pre-registered failure conditions (written before any result was read)

Per the task brief, any of the following kills this direction:

1. The haircut collapses to a near-flat rescale (R² > 0.9 against a
   constant-multiplier control) — R-34's own documented failure mode for
   an identically-shaped never-increase-only multiplier.
2. It doesn't beat `kelly_regime_v4` on Sharpe or drawdown on
   inner-validation (2021-01-01→2022-12-31) by more than the ±0.2 Sharpe
   noise floor (R-20) — or does, but only inside the handful of large
   macro events that generated the composite.
3. It fails the causality tamper probe.
4. It fails the pre-registered ETH falsification test (same qualitative
   direction — haircut correlates with reduced drawdown — expected on
   ETH too, since VIX/DXY are market-wide, not BTC-specific).

No holdout (2023-01-01 onward) is read anywhere in this branch — the
task explicitly scoped this round to inner-train/inner-validation/
falsification only. Grepped this file for date literals: the only
data-slicing literal is `"2022-12-31"` (an upper training-window bound);
every other `2023`/`2024`/`2025`/`2026` occurrence is prose in a
citation or a docstring sentence, not a data read. **Holdout counter
unchanged by this branch: +0.**

## 4. Configurations evaluated (deflated-Sharpe bookkeeping)

**10 distinct configurations**, matching the `kelly_regime_v9_basis_brake.py`/
`kelly_regime_v10_onchain_confirm.py` convention (count once per
config, not once per config×market×window backtest run — a config
re-measured on inner-validation, both markets, or on the ETH/BTC-control
falsification pair does not add to this count):

- 9 = the swept grid, `lam ∈ {0.15, 0.25, 0.35} × z_scale ∈ {1.0, 2.0, 3.0}`
  (grid chosen a-priori from `stress_z`'s own unconditional distribution
  on the full committed spot index — std=0.91, p90=1.11, p95=1.63,
  p99=3.21 — and from bracketing R-41's/R-44's own selected `lam` ranges,
  not fit to inner-validation).
- 1 = the `lam=0` correctness check (must reduce to v4 bit-for-bit).

Backtest *runs* (not counted toward the trials figure above, matching
the established convention): 9 on inner-train (spot), 18 on
inner-validation (9×2 markets) plus 2 `kelly_regime_v4` control runs, 2
for the `lam=0` correctness check, ~20 for the exposure-artifact check
(9×2 markets, re-running cached candidates), a handful for the causality
probes (price + macro tamper, pre-2023 only), 40 for the ETH
falsification (10×2 markets×2 assets), and 4 for the train-vs-validation
overfitting-signature spot check on the grid midpoint.

## 5. Causality

Two independent two-opposite-tampers probes, restricted to strictly
pre-2023 bars (cut at bar 295,000 of a 300,000-bar pre-2023 slice):

| probe | column | max\|diff\| before cut | result |
|---|---|---|---|
| price (×3 / ÷3 from the cut) | `target` | 0.000e+00 | PASS |
| price | `_frac` | 0.000e+00 | PASS |
| price | `_mult` | 0.000e+00 | PASS |
| price | orders at 8 probe bars | match | PASS |
| price | equity | 0.000e+00 | PASS |
| macro (VIX/DXY ×3 / ÷3 from the cut day, in a throwaway scratch copy of the raw CSVs — never under the repo) | `target` | 0.000e+00 | PASS |
| macro | `_mult` | 0.000e+00 | PASS |
| macro | `_stress_z` | 0.000e+00 | PASS |

The macro probe is new to this file: it tampers the raw VIX/DXY CSVs
themselves (not the price frame) via the strategy's injected `data_dir`
argument, exercising this file's one new ingredient — a price-only probe
alone would say nothing about whether the macro merge itself is causal.
**Both probes pass cleanly.** No lookahead in either the price or the
macro data path.

Also verified directly: `stress_z` carries **0 NaN across the entire
1,010,889-bar committed spot index** (2017-01-01 onward) — macro
coverage (2016-06-01) plus the z-score's 60-day `min_periods` warmup
fully precedes the spot series' own start, so no `lam=0`-style fallback
window caveat applies on the primary series (unlike R-41's basis brake,
which needed a shifted inner-train start).

## 6. Inner-train (2017-01-01 → 2020-12-31, spot)

| lam | z_scale | final | max DD | Sharpe | trades |
|---|---|---|---|---|---|
| — | v4 control | $18,477 | 43.3% | 2.03 | 72 |
| 0.15 | 1.0 | $19,605 | 41.8% | 2.09 | 70 |
| 0.15 | 2.0 | $19,115 | 42.3% | 2.06 | 70 |
| 0.15 | 3.0 | $19,125 | 42.4% | 2.06 | 70 |
| 0.25 | 1.0 | $18,917 | 41.6% | 2.09 | 70 |
| 0.25 | 2.0 | $19,454 | 41.5% | 2.08 | 70 |
| 0.25 | 3.0 | $19,137 | 42.2% | 2.07 | 70 |
| 0.35 | 1.0 | $20,072 | 40.2% | 2.15 | 69 |
| 0.35 | 2.0 | $19,890 | 40.9% | 2.11 | 69 |
| 0.35 | 3.0 | $19,629 | 41.7% | 2.09 | 69 |

`lam=0` correctness check: max|target diff| vs v4 = 0.000e+00 — **PASS**,
exact reduction to v4 as designed.

Every swept config beats the v4 control on final balance, Sharpe, *and*
drawdown on inner-train. This is exactly the pattern the promotion bar
warns against measuring in isolation (step 3 is a training resource, not
evidence) — see §7 and §9 for why it does not survive contact with
inner-validation or the artifact check.

## 7. Inner-validation (2021-01-01 → 2022-12-31, both markets)

| lam | z_scale | spot final | spot DD | spot Sharpe | fut final | fut DD | fut Sharpe |
|---|---|---|---|---|---|---|---|
| — | v4 control | $998 | 33.2% | **0.142** | $1,064 | 32.3% | **0.251** |
| 0.15 | 1.0 | $964 | 32.3% | 0.074 | $998 | 34.7% | 0.138 |
| 0.15 | 2.0 | $991 | 32.1% | 0.126 | $1,025 | 33.9% | 0.187 |
| 0.15 | 3.0 | $993 | 32.5% | 0.132 | $1,035 | 33.7% | 0.205 |
| 0.25 | 1.0 | $953 | 31.3% | 0.050 | $1,050 | 29.3% | 0.226 |
| 0.25 | 2.0 | $978 | 31.9% | 0.102 | $1,009 | 34.1% | 0.156 |
| 0.25 | 3.0 | $985 | 32.3% | 0.115 | $1,025 | 33.8% | 0.186 |
| 0.35 | 1.0 | $942 | 30.8% | 0.024 | $989 | 31.5% | 0.114 |
| 0.35 | 2.0 | $973 | 31.3% | 0.091 | $1,035 | 31.8% | 0.202 |
| 0.35 | 3.0 | $980 | 31.9% | 0.106 | $1,010 | 34.1% | 0.159 |

**Every single one of the 18 (config × market) cells has a LOWER Sharpe
than the v4 control.** No config clears v4's Sharpe on either market —
let alone by the ±0.2 noise floor; the gap runs the wrong way, by
−0.02 to −0.12 (spot) and −0.03 to −0.14 (futures).

Drawdown is a mixed bag, not a clean improvement: on spot, every config
is modestly better than v4 (30.8–32.5% vs 33.2%, a 0.7–2.4pp edge). On
**futures the sign flips depending on `z_scale`**: `z_scale=1.0` configs
beat v4 (29.3–31.5% vs 32.3%), but `z_scale=2.0`/`3.0` configs are
*worse* than v4 (33.7–34.1% vs 32.3%). A mechanism whose drawdown effect
changes sign under a ±50% change in one of its two free parameters is
not the "plateau, not a peak" the promotion bar requires — it is closer
to a knife-edge on this axis.

Overfitting-signature check (grid midpoint, `lam=0.25, z_scale=2.0`),
train vs validation, the R-37/38/40 diagnostic:

| window | market | candidate | v4 | result |
|---|---|---|---|---|
| inner-train | spot | $19,454 (DD 41.5%, Sh 2.08) | $18,477 (DD 43.3%, Sh 2.03) | beats v4 |
| inner-train | futures | $31,891 (DD 32.9%, Sh 2.33) | $30,344 (DD 35.3%, Sh 2.28) | beats v4 |
| inner-validation | spot | $978 (DD 31.9%, Sh 0.10) | $998 (DD 33.2%, Sh 0.14) | **LOSES to v4** |
| inner-validation | futures | $1,009 (DD 34.1%, Sh 0.16) | $1,064 (DD 32.3%, Sh 0.25) | **LOSES to v4** |

The exact win-on-train / lose-on-validation signature this project has
repeatedly flagged as diagnostic of an over-fit or noise-driven effect.

## 8. Exposure-artifact check (R-33/R-34's standing threshold)

Mean-notional-matched flat rescale of v4's own `target`, R² against the
candidate's `target`, inner-validation, both markets:

| lam | z_scale | spot R² | futures R² | verdict |
|---|---|---|---|---|
| 0.15 | 1.0 | 0.9945 | 0.9945 | ARTIFACT |
| 0.15 | 2.0 | 0.9978 | 0.9978 | ARTIFACT |
| 0.15 | 3.0 | 0.9989 | 0.9989 | ARTIFACT |
| 0.25 | 1.0 | 0.9870 | 0.9870 | ARTIFACT |
| 0.25 | 2.0 | 0.9942 | 0.9942 | ARTIFACT |
| 0.25 | 3.0 | 0.9973 | 0.9973 | ARTIFACT |
| 0.35 | 1.0 | 0.9744 | 0.9744 | ARTIFACT |
| 0.35 | 2.0 | 0.9908 | 0.9908 | ARTIFACT |
| 0.35 | 3.0 | 0.9948 | 0.9948 | ARTIFACT |

**Every one of the 9×2 = 18 cells clears the R² > 0.95 danger line** —
the lowest value in the whole grid is 0.9744 (`lam=0.35, z_scale=1.0`,
the most aggressive setting, which is also the one furthest from a flat
rescale, as expected). This is R-34's exact failure mode
(`kelly_regime_v5_damp.py`, R²=0.997), reproduced here with a genuinely
price-independent input rather than assumed absent because the data
source differs. Mean notional shrinks only 2–7% relative to v4
(0.265–0.278 vs v4's 0.283 spot; 0.271–0.285 vs v4's 0.289 futures) —
consistent with the small, roughly-monotone drawdown edge on spot in §7
being explained mostly by *how much less* exposure the brake carries on
average, not by *when* it carries less. **Pre-registered failure
condition (1) is triggered on every swept configuration.**

## 9. ETH falsification (standard pre-2020 BTC-control/ETH pair)

`btcusd_bitfinex_5m.csv.gz` (control, 2016-01-01→2019-12-31) vs
`ethusd_bitfinex_5m.csv.gz` (test, 2016-03-09→2019-12-31). Neither file
touches the 2023+ holdout. `stress_z` NaN bars (pre-macro-coverage
fallback, handled by the same `isfinite`→`mult=1` rule as everywhere
else): 55,385 of 396,449 BTC bars, 24,204 of 342,929 ETH bars — both
early-window-only, verified directly rather than assumed.

At the grid midpoint (`lam=0.25, z_scale=2.0`):

| asset | market | v4 control DD | candidate DD | Δ DD | v4 control Sharpe | candidate Sharpe |
|---|---|---|---|---|---|---|
| BTC | spot | 40.1% | 38.8% | −1.3pp | 1.86 | 1.89 |
| BTC | futures | 32.1% | 36.3% | **+4.2pp (worse)** | 2.19 | 2.08 |
| ETH | spot | 36.5% | 35.4% | −1.1pp | 1.48 | 1.49 |
| ETH | futures | 35.1% | 36.1% | **+1.0pp (worse)** | 1.25 | 1.27 |

The qualitative pattern is actually **consistent between BTC and ETH at
this config** (a small drawdown improvement on spot, a small
*deterioration* on leveraged futures) — so this is not a case of "works
on BTC, silently fails on ETH." It is a case of the mechanism not
reliably improving drawdown on *either* asset once leverage interacts
with v4's own vol-targeting hysteresis latch.

More strikingly, the **full BTC-control grid is not a plateau**: several
low-`lam`/low-`z_scale` futures configs are dramatically worse than v4
(`lam=0.15, z_scale=1.0`: DD 39.8% vs v4's 32.1%, final $18,602 vs
$25,681 — a 28% return shortfall), while `lam=0.35, z_scale=1.0` is
*better* than v4 (DD 31.0%, final $26,627) — a non-monotonic swing from a
±50-100% change in one free parameter, on the control asset the brake is
supposed to help most directly. This is consistent with §7's
non-monotonic sign flip on inner-validation futures: small changes in
how much the brake shaves off v4's target interact with v4's own
deadband/vol-targeting-state latch in a way that occasionally amplifies
rather than dampens outcomes — a fragility, not a robust mechanism.

ETH's own grid is comparatively tame (Sharpe 1.47–1.50 spot, 1.24–1.30
futures, all within noise of v4's 1.48/1.25) — no config on ETH is a
disaster the way some BTC-control futures configs are, but none is a
clear win either. **Pre-registered failure condition (4) is triggered**:
the mechanism does not show a reliable "haircut correlates with reduced
drawdown" direction on ETH (or, for that matter, on BTC control either) —
what direction exists is small, inconsistent by market, and in the
futures case frequently the wrong sign.

## 10. Verdict

**NEGATIVE. Reject.**

Against the task's own four pre-registered failure conditions:

1. **Exposure-artifact collapse — TRIGGERED.** All 18 (config×market)
   cells clear R²>0.95 (range 0.974–0.999) on inner-validation. This is
   R-34's own documented failure mode, now confirmed on a genuinely
   price-independent input rather than merely a price-derived one — the
   prior that a new *kind* of data source would avoid this trap does not
   hold here.
2. **No inner-validation edge beyond noise — TRIGGERED.** Every one of
   18 cells has LOWER Sharpe than the v4 control (never mind clearing the
   ±0.2 floor); drawdown is a small, inconsistent effect that flips sign
   on futures depending on `z_scale`.
3. **Causality — PASSES.** Both the standard price probe and the new
   macro-CSV tamper probe show 0.000e+00 max difference before the cut.
   Not the reason this direction fails.
4. **ETH falsification — TRIGGERED.** The direction is not reliably
   "more stress → less drawdown" on either asset; it is small and
   market-dependent on ETH, and swings both dramatically favorable and
   dramatically unfavorable on the BTC control depending on the exact
   `(lam, z_scale)` pair — a fragility signature, not a plateau.

Against ROUTINE.md's own promotion bar (default REJECT): the candidate
does not beat `kelly_regime_v4` on Sharpe anywhere on inner-validation,
its one candidate drawdown improvement (spot) is fully explained by the
exposure-artifact check as a small, roughly-flat rescale rather than
genuine regime differentiation, it fails its own pre-registered
falsification test, and its parameter neighbourhood on the one window
(BTC-control futures) where the brake should matter most is a knife-edge,
not a plateau. No criterion is met.

**One-line lesson:** a never-increase-only multiplicative brake collapses
into a near-flat exposure rescale (R-34's failure mode) even when fed by
data that is genuinely, verifiably independent of price — the failure
mode is architectural (a *shape* — bounded, monotone, single-direction,
layered on top of an already-dominant vote/vol-target signal), not a
property of stale or price-derived inputs specifically; a materially
different mechanism (e.g. feeding the signal into the vote itself, as the
sibling novel branch attempts, or gating on a rarer, more extreme
threshold than this grid's a-priori percentile choices) would be needed
to give a macro-stress signal a real chance to matter on this data.

**Configs evaluated: 10** (see §4). **Holdout consultations: +0** —
nothing in this branch reads 2023-01-01 or later.
