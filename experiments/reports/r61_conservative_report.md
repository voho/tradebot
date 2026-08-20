# R-61 (conservative branch) — z-score mean-reversion vote on kelly_regime's sizing (08-20)

Pre-registration (shared, both branches): `experiments/r61_shared.py`.
Implementation: `experiments/r61_conservative_zscore_reversion.py`. Nothing
under `src/tradebot/strategies/` is touched; nothing is registered.

## 1. The question, one sentence

R-59 and R-60 (twenty-one failed retunes of `kelly_regime_v4`'s own TREND
vote) both converged on the same untested alternative: the panel's
matched-exposure drawdown advantage "looks like a buy-the-dip effect these
higher-volatility, more mean-reverting instruments reward." This branch
builds that strategy for the first time — `kelly_regime`'s unmodified
fractional-Kelly vol-targeted sizing loop, driven by a short-horizon
(1/3/7-day) rolling z-score reversion vote instead of a trend-anchor vote —
and asks whether it beats `buy_and_hold` on the six-asset panel.

## 2. Mechanism

`KellyRegimeV61ZscoreReversion.prepare()`: for each horizon in (1, 3, 7)
calendar days (`window = days * 288` bars), `z = (close - roll_mean) /
roll_std`; vote bullish (1.0) where `z < -z_thresh`, bearish (0.0) where
`z > +z_thresh`, else hold the previous latched verdict — averaged across
the three horizons exactly as `KellyRegime` averages its three trend votes.
Sizing loop (`target_vol=0.55`, `max_leverage=2.0`, 10% deadband,
8-day-span EWM realized vol) copied verbatim from
`tradebot.strategies.kelly_regime.KellyRegime`.

## 3. Causality — PASS

Truncation test (three cut points on a 60,000-bar BTC slice: 30000, 50000,
59999) — early-row `target` values bit-identical whether or not later rows
exist in the frame. Tamper probe (opposite ×3/÷3 tail corruptions from a
fixed cut) — decisions at/before the cut identical. **Independently
reproduced by the operator**, bit-for-bit on the dollar figures below.

## 4. D1/D2 — Z_THRESH_GRID x PANEL_TRAIN (2020-04-01 → 2022-12-31), spot @0.10%

| z_thresh | BCH beats final/dd | LTC | ETC | DASH | LINK | XTZ | D1 | D2 | mean ΔmaxDD |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | F/T | F/T | F/T | F/T | F/T | T/T | 1/6 | 6/6 | −26.73pp |
| 1.5 | F/T | F/T | F/T | T/T | F/T | T/T | 2/6 | 6/6 | −30.38pp |
| 2.0 | F/T | F/T | F/T | T/T | F/T | T/T | 2/6 | 6/6 | −32.06pp |

No threshold reaches D1 above 2/6 — the grid plateaus near the floor, not at
a promotable value. **Selected z_thresh = 2.0** (tie-break on D1: mean
ΔmaxDD more negative than z=1.5's −30.38pp). D2 = 6/6 at every threshold,
but this is very likely the SIZE machinery's own known signature (a strategy
that is flat or under-exposed part of the time trivially draws down less
than a fully-invested `buy_and_hold`, R-33/R-57's own "match risk before
comparing anything" lesson) rather than evidence the signal itself is good —
D2 here compares against plain `buy_and_hold`, not a matched-exposure hold,
by the pre-registration's own design.

**Operator's independent re-run of the full grid reproduced every cell
above exactly** (dollar figures, D1/D2 counts, mean ΔmaxDD, and the selected
z_thresh), from a clean import of the branch module.

## 5. D3 — BTC/ETH falsification (candidate z=2.0 vs buy_and_hold vs kelly_regime_v4), spot @0.10%

| Window | candidate final | candidate DD | hold final | hold DD | v4 final | v4 DD |
|---|---|---|---|---|---|---|
| BTC_INNER_TRAIN (2017-2020) | $480 | 76.3% | $29,803 | 84.1% | $18,477 | 43.3% |
| BTC_INNER_VALID (2021-2022) | $519 | 59.7% | $574 | 77.3% | $998 | 33.2% |
| ETH_FULL | $244 | 82.0% | $14,589 | 81.7% | $12,481 | 35.3% |

Predicted underperformance confirmed decisively, and more severely than
predicted: the ungated reversion vote whipsaws against BTC/ETH's dominant
trend, losing final balance by 1-2 orders of magnitude to both benchmarks
while also carrying the *worst* drawdown of the three in every window. No
BTC or ETH bar dated 2023-01-01 or later is read (grep-confirmed).

## 6. D4 — 0.40% fee tier, PANEL_TRAIN, candidate z=2.0 vs buy_and_hold

| Asset | candidate | hold | beats |
|---|---|---|---|
| BCH | $218 | $438 | No |
| LTC | $467 | $1,723 | No |
| ETC | $229 | $3,137 | No |
| DASH | $429 | $641 | No |
| LINK | $463 | $2,386 | No |
| XTZ | $527 | $444 | **Yes** |

D4 = 1/6.

## 7. D5 — PANEL_TEST (2023-2026), descriptive

D1(test) = 2/6, D2(test) = 6/6 — same two winning assets pattern shifts
(LTC/XTZ win instead of DASH/XTZ) but the weak result generalizes rather
than being a train-set fluke.

## 8. Promotion bar

D1 ≥ 5/6: **FAIL** (2/6). D2 ≥ 4/6: PASS (6/6, caveated above). D4 ≥ 4/6:
**FAIL** (1/6). Plateau: PASS (never 0/6). **NOT PROMOTED.**

## 9. Configurations evaluated: 57

Sweep 24 (18 grid cells + 6 hold references) + D3 9 + D4 12 + D5 12 = 57.

## 10. Honest caveats

D3's magnitude (1-2 orders of magnitude worse, not merely "worse") was
flagged by the branch author as a possible bug signal; the operator's
causality re-run and the mechanism's own logic (a fast reversion vote
whipsawing against a multi-year trend, paying fees on every flip) support
reading it as a real, if extreme, instance of the predicted failure mode
rather than a bug.
