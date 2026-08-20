# R-58 (novel branch) — does self-normalizing exposure fix v4's panel drawdown inversion? (08-20)

Unregistered experiment (backlog **B-25**). Code:
`experiments/r58_novel_relative_vol_scale.py`. Shared pre-registration:
`experiments/r58_shared.py` (windows, costs, decision rules, matched-hold
harness — read there, reused here, not restated differently). Nothing under
`src/tradebot/strategies/` is touched: `KellyRegimeRelativeVol` is a plain,
unregistered `Strategy` subclass constructed directly, never through
`get_strategy()`.

## 1. The question, one sentence

R-57 found `kelly_regime_v4`'s matched-exposure drawdown advantage inverts
on 6 of 6 further Coinbase instruments and named a candidate cause —
`target_vol=0.55`/`max_leverage=2.0` are BTC's absolute volatility scale, so
`target_vol/realized_vol` is structurally smaller and near-permanently
binding on higher-vol instruments; does normalizing volatility against each
instrument's OWN long-run trailing average — zero new fitted parameters —
fix it?

## 2. The mechanism

`KellyRegimeV3.prepare()` (inherited unchanged by `KellyRegimeV4`, which
only changes the default anchor ladder to 20/40/80 days) is replicated
byte-for-byte, with exactly one change. One new causal series is added:

```
long_run_vol = vol.ewm(span=LONG_RUN_SPAN_DAYS * BARS_PER_DAY, min_periods=BARS_PER_DAY).mean()
```

`LONG_RUN_SPAN_DAYS = 720` (~2 years) — a **structural** choice fixed before
any result on this branch was read, not swept or fit: longer than v3's
existing 180-day `anchor_span_days` (the span already behind `slow`, which
drives the breakout hysteresis) so it captures a genuinely long-run
reference distinct from that existing anchor, and long enough to span a
full bull and bear phase on every one of this round's eight assets.

The two exposure-scale terms then use volatility expressed relative to that
long-run level instead of its raw absolute value:

```
vol_rel = vol / long_run_vol                                  # dimensionless, mean ~1 by construction
full    = min(target_vol / vol_rel,               max_leverage)
steady  = min(target_vol / (slow / long_run_vol),  max_leverage)
```

`target_vol=0.55` and `max_leverage=2.0` stay **unchanged global constants**
for BTC, ETH and all six panel assets — no per-asset number is fit anywhere
in this branch, which is the entire point of it (as distinct from the
conservative branch's per-asset calibration). The hysteresis state machine
(`ratio = vol/slow` driving the +1/-1 breakout latch) is untouched — it was
already relative and was not the diagnosed problem. Literature basis:
Barroso & Santa-Clara (2015, JFE 116(1)); Moskowitz, Ooi & Pedersen (2012);
Baltas & Kosowski (2013/2017) — full citations in `experiments/r58_shared.py`.

## 3. Self-consistency check (mechanism's own diagnostic, not a decision rule)

`vol_rel`'s time-average over PANEL_TRAIN/CONTROL (2020-04-01→2022-12-31),
all eight assets:

| asset | mean(vol_rel) |
|---|---|
| BTC | 0.938 |
| ETH | 0.978 |
| BCH | 0.915 |
| LTC | 0.952 |
| ETC | 0.935 |
| DASH | 0.906 |
| LINK | 0.903 |
| XTZ | 0.908 |

All eight sit in a tight 0.90–0.98 band around 1.0 — the normalization
holds by construction on every asset in this round; none is flagged as
"far from 1.0" (band checked: 0.7–1.4). This rules out "the fix didn't even
self-normalize" as the failure explanation for what follows.

## 4. Causality tamper probe

`test_causality_strict.py`'s methodology (opposite 3x/÷3 price and 7x/÷7
volume tampers after a cut, decisions compared at 1/2/3/5/10/20 bars before
the cut), run on BTC (2022-12-31 and earlier only, per this round's holdout
restriction) plus BCH and LTC, constructing `KellyRegimeRelativeVol`
directly: **PASS on all three** — decisions identical under opposite
post-cut tampers. `long_run_vol` is computed the same causal, shift-then-EWM
way as the existing `slow` series, with no full-series statistic and no
negative shift.

## 5. Results against the pre-registered rules

### D1 (primary) — PANEL_TRAIN, spot @0.10%, matched-exposure drawdown

| asset | c (candidate's mean notional) | candidate max DD | matched hold max DD | Δ DD (pp, + = candidate worse) | 95% paired interval |
|---|---|---|---|---|---|
| BCH | 0.25 | 50.9% | 43.3% | **+7.7** | [−10.5, +36.6] |
| LTC | 0.29 | 42.9% | 40.8% | **+1.7** | [−10.0, +25.9] |
| ETC | 0.25 | 48.5% | 34.0% | **+17.3** | [+0.1, +41.2] |
| DASH | 0.25 | 55.4% | 37.4% | **+17.9** | [−2.5, +40.8] |
| LINK | 0.32 | 53.0% | 41.5% | **+14.6** | [−9.5, +36.6] |
| XTZ | 0.26 | 52.9% | 40.9% | **+12.0** | [+0.8, +42.3] |

**0 of 6** — exact binomial p = 1.0000 → **FAILS**. The sign is positive
(candidate worse than the matched hold) on every asset; 2 of 6 intervals
exclude zero, both against the candidate. Mean notional (0.25–0.32) is
somewhat higher than R-57's raw v4 on the same panel (0.18–0.26) — the
normalization does lift sizing a little — but the per-asset drawdown deltas
are, if anything, similar to or only mildly better than R-57's v4 numbers
(R-57: BCH +5.2, LTC +33.8, ETC +23.6, DASH +29.8, LINK +13.4, XTZ +19.3;
here: +7.7, +1.7, +17.3, +17.9, +14.6, +12.0 — LTC's delta shrank
substantially, the rest moved only modestly), and the sign never flips.

### D2 (falsification, control) — CONTROL, BTC/ETH, spot @0.10%

| asset | candidate dDD (matched) | R-57's v4 control | tolerance | verdict |
|---|---|---|---|---|
| BTC | −4.5pp [−17.7, +14.7] | −5.6pp | ≤ base+5pp | within tolerance |
| ETH | −10.8pp [−17.0, +20.3] | −11.5pp | ≤ base+5pp | within tolerance |

**PASSES** — the candidate's own matched-exposure drawdown advantage on
BTC and ETH is essentially unchanged from v4's frozen numbers (−4.5pp vs
−5.6pp, −10.8pp vs −11.5pp, both inside the 5pp regression tolerance). The
fix does not break the two instruments the mechanism already worked on.

### D3 (generalization, descriptive) — PANEL_TEST 2023-01-01→2026-08-20, spot @0.10%

| asset | Δ DD (pp, matched) | 95% interval |
|---|---|---|
| BCH | +9.1 | [−3.5, +42.8] |
| LTC | +34.7 | [+8.1, +54.0] |
| ETC | +20.8 | [+10.2, +47.1] |
| DASH | +24.9 | [+1.7, +38.0] |
| LINK | +10.5 | [−1.9, +33.7] |
| XTZ | +5.3 | [−1.4, +31.8] |

**0 of 6**, same inverted sign as D1, on the held-out panel window. Not a
gate, but it corroborates D1 rather than contradicting it — the fix's
failure is not confined to the training window.

### D4 (0.40% fee falsification) — PANEL_TRAIN, spot @0.40%, beats buy_and_hold's final balance

| asset | candidate final | buy_and_hold final |
|---|---|---|
| BCH | $917 | $438 |
| LTC | $1,251 | $1,723 |
| ETC | $3,049 | $3,137 |
| DASH | $830 | $641 |
| LINK | $1,069 | $2,386 |
| XTZ | $513 | $444 |

**3 of 6** (BCH, DASH, XTZ). Threshold was ≥5/6 to "survive"; this is not
counted as a promotion-relevant surprise (D2's menu prediction was that D4
fails, and by the ≥5/6 bar it does), though the raw count landed higher
than R-57's own D2 on `kelly_regime_v4` (2/6) — worth naming, not
load-bearing, since D1 has already failed the primary gate.

## 6. Verdict

Applying `experiments.r58_shared.promoted(k1, dd_advantage)` mechanically
(requires D1 ≥ 5/6 **and** D2 passing both BTC and ETH):

- D1: 0/6 → FAILS (needed ≥5/6)
- D2: PASSES (BTC −4.5pp, ETH −10.8pp, both within R-57's control ±5pp tolerance)
- `promoted(0, {...})` → **False**

**NEGATIVE.** The failure mode, named plainly: this is not a case of the
mechanism failing to self-normalize (section 3's check holds cleanly, 0.90–
0.98 across all eight assets) or of the fix breaking BTC/ETH (D2 passes
outright). It is that removing the absolute-scale term and replacing it
with a per-instrument-relative one changes exposure levels only modestly on
this panel (mean notional rose from R-57's 0.18–0.26 to 0.25–0.32) and does
not touch the actual mechanism R-57's own diagnosis undersold: the
regime-vote's *timing* on these six instruments, not merely the scale
constant, is what drives the drawdown pattern the matched hold already
prices in. A dimensionless, self-calibrating scale term is not sufficient
to flip a single sign on this panel, on either window. This closes B-25's
novel branch as a second NEGATIVE alongside (pending) the conservative
branch, extending the SIZE-axis record described in the pre-registration.

## 7. Configurations evaluated

**60** total backtests (`CONFIG_COUNT`, this branch only — causality and the
self-consistency check use `prepare()` directly and read no backtest, so
they cost 0): D1 6 assets × 3 arms = 18, D2 2 assets × 3 arms = 6, D3 6
assets × 3 arms = 18, D4 6 assets × 3 arms = 18. Total across both R-58
branches is summed separately per `experiments/r58_shared.py`'s convention.

Holdout consultations added by this branch: **0** — no BTC/ETH bar past
2022-12-31 is read anywhere in `experiments/r58_novel_relative_vol_scale.py`
(BTC via `load_dataset` and ETH via `load_coinbase_spot` are both sliced to
`:2022-12-31` before any use, including the causality probe).

## 8. Raw data

`reports/r58_novel/d1_panel_train.csv`, `d2_control.csv`,
`d3_panel_test.csv`, `d4_panel_train_040.csv`, `vol_rel_selfcheck.csv`.
