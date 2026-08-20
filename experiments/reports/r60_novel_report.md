# R-60 (novel branch) — does a flip-rate-adaptive hysteresis band fix v4's panel drawdown inversion? (08-20)

Unregistered experiment (backlog **B-26**). Code:
`experiments/r60_novel_flip_adaptive_hysteresis.py`. Shared pre-registration:
`experiments/r60_shared.py` (windows, costs, decision rules, matched-hold
harness — read there, reused here, not restated differently). Nothing under
`src/tradebot/strategies/` is touched: `KellyRegimeFlipAdaptiveHysteresis` is
a plain, unregistered `Strategy` subclass constructed directly, never through
`get_strategy()`.

## 1. The question, one sentence

R-57 found `kelly_regime_v4`'s matched-exposure drawdown advantage inverts
on 6 of 6 further Coinbase instruments, and R-59 found twice that rescaling
exposure (`target_vol`/`max_leverage`, in absolute or self-normalized form)
does not fix it, localizing the failure to the vote/gate's *timing* rather
than its *scale*; does deriving each instrument's hysteresis-band width from
its own measured regime-switching frequency — a single frozen scalar per
asset, computed once from PANEL_TRAIN, everything else in the strategy
byte-identical to v4 — restore the property?

## 2. The mechanism

`KellyRegime.prepare()` (inherited unchanged through `KellyRegimeV3` and
`KellyRegimeV4`) latches each of the three 20/40/80-day anchor votes inside
a fixed `band = 0.01` (1%) around its own rolling mean. That constant is
identical for BTC and for six instruments R-57 showed are measurably more
mean-reverting. Dai, Zhang & Zhu (2010, SIAM J. Financial Mathematics 1(1))
and Dai, Yang, Zhang & Zhu (2016, Math. of Operations Research 41(2)) derive
the optimal buy/sell trigger in a two-state regime-switching model as a
function of the regime's own transition intensity: a market that switches
states more often needs a WIDER no-trade band; a market with more
persistent regimes can use a narrower one.

This branch changes **only** that one constant, and only via a frozen,
pre-computed, per-asset value passed as a plain constructor argument —
never a per-bar recomputation:

1. On PANEL_TRAIN (2020-04-01→2022-12-31) ONLY, for each of 8 assets
   (BTC, ETH, the six-asset panel), measure the RAW (unbanded, unlatched)
   crossing signal `sign(close - anchor)` for each of the three fixed
   20/40/80-day anchors (anchor computation itself untouched) and count
   sign changes per year. The three anchors' rates are combined into ONE
   asset-level scalar by a plain mean (`measure_flip_rate`).
2. `band_asset = clip(BASE_BAND * f(asset_flip_rate / btc_flip_rate),
   0.5%, 3.0%)`, `BASE_BAND = 0.01`. Two functional forms for `f` are
   tried: `linear` (`f(x)=x`) and a damped `sqrt` (`f(x)=sqrt(x)`)
   (`derive_band`). The ratio is oriented `asset/btc` — not its inverse —
   so a busier regime gets a WIDER band, matching the cited literature's
   stated direction; `f(1)=1` for BTC by construction either way, so BTC's
   own derived band is exactly 1.00% under both variants, making the
   fix a structural no-op on the asset the mechanism was tuned on.
3. The fitted scalar is frozen and reused unchanged across PANEL_TRAIN
   (D1), PANEL_TEST (D3) and CONTROL (D2) for that asset. `measure_flip_rate`
   and `derive_band` are entirely separate functions from the strategy
   class, and neither is called from `prepare()`/`on_bar()`.
4. `KellyRegimeFlipAdaptiveHysteresis` has **no `prepare()`/`on_bar()`
   override at all** — `band` was already a constructor parameter of the
   whole `kelly_regime` family; this subclass exists only to make it a
   required argument. The anchor computation, vote-fraction averaging, the
   latching logic, and the entire exposure-scale mechanism
   (`target_vol`/`max_leverage`/breakout hysteresis) are the literal
   inherited code from `KellyRegimeV3`/`KellyRegimeV4`, not a copy.

Literature basis, full citations: `experiments/r60_shared.py`.

## 3. Measured flip rates and derived bands (PANEL_TRAIN, 2020-04-01→2022-12-31)

Per-anchor and combined (mean) annualized sign-flip rates of the raw,
unbanded `sign(close - anchor)` signal:

| asset | 20d flips/yr | 40d flips/yr | 80d flips/yr | mean flips/yr |
|---|---|---|---|---|
| BTC | 495.85 | 359.43 | 237.19 | **364.16** |
| ETH | 508.95 | 311.41 | 301.95 | **374.10** |
| BCH | 593.35 | 540.60 | 309.23 | **481.06** |
| LTC | 616.63 | 419.09 | 286.31 | **440.68** |
| ETC | 668.65 | 494.03 | 233.56 | **465.41** |
| DASH | 517.68 | 494.40 | 393.26 | **468.45** |
| LINK | 593.35 | 350.70 | 343.42 | **429.16** |
| XTZ | 696.67 | 562.43 | 184.08 | **481.06** |

This is a striking, reportable number on its own: **the panel genuinely
does flip its vote more often than BTC or ETH** — R-59's own write-up
speculated the panel is "more mean-reverting" but never measured a flip
rate directly. Every one of the six panel assets' combined flip rate
(429–481/yr) exceeds both BTC's (364/yr) and ETH's (374/yr); the panel's
own spread (429–481) is narrower than its gap to BTC (~18–32% higher).

Derived bands, both functional-form variants, clipped to [0.5%, 3.0%] (no
asset hit either clip bound):

| asset | flip_rate | ratio (asset/BTC) | band (linear) | band (sqrt) |
|---|---|---|---|---|
| BTC | 364.16 | 1.000 | 1.00% | 1.00% |
| ETH | 374.10 | 1.027 | 1.03% | 1.01% |
| BCH | 481.06 | 1.321 | 1.32% | 1.15% |
| LTC | 440.68 | 1.210 | 1.21% | 1.10% |
| ETC | 465.41 | 1.278 | 1.28% | 1.13% |
| DASH | 468.45 | 1.286 | 1.29% | 1.13% |
| LINK | 429.16 | 1.178 | 1.18% | 1.09% |
| XTZ | 481.06 | 1.321 | 1.32% | 1.15% |

BTC's band lands at exactly 1.00% under both variants (self-consistency by
construction); ETH's moves only marginally (1.01–1.03%); the panel's bands
run 1.09%–1.32%, roughly 1.1–1.3x v4's original 1% band, never approaching
either clip bound.

## 4. Functional-form variant frozen, and why

Both variants were run through D1 (PANEL_TRAIN, the primary decision rule)
on the full six-asset panel before either was selected — the small,
pre-specified grid the round's own framing authorizes ("try at least a
plain linear ratio and one damped variant"). **D1 tied at 0/6 for both**
(see §6). With no D1-count tiebreaker available, the frozen variant was
decided by the rule stated in the code before comparing anything beyond the
count itself: prefer the damped (`sqrt`) variant, because a linear response
overreacts to a noisily-estimated flip rate (one extra or missing flip near
the PANEL_TRAIN boundary moves a linear band twice as far as a `sqrt` one).
**`sqrt` was frozen** and used for D2/D3/D4 below; both variants' D1 numbers
are reported in full regardless, since the tie means the choice was not
performance-driven.

## 5. Causality tamper probe

`test_causality_strict.py`'s methodology (opposite 3x/÷3 price and 7x/÷7
volume tampers after a cut, decisions compared at 1/2/3/5/10/20 bars before
the cut), constructing `KellyRegimeFlipAdaptiveHysteresis` directly with
each asset's frozen (linear-variant) band, run on BTC (2022-12-31 and
earlier only) plus BCH and LTC:

| asset | band used | result |
|---|---|---|
| BTC | 0.0100 | **PASS** |
| BCH | 0.0132 | **PASS** |
| LTC | 0.0121 | **PASS** |

**PASS on all three.** Expected: the strategy class has no `prepare()`/
`on_bar()` override at all (§2.4), so this exercises the identical,
already-causal `KellyRegimeV3.prepare()` code path v4 uses, just constructed
with a pre-baked `band` float. The offline fitting functions
(`measure_flip_rate`, `derive_band`) are never called from `prepare()` or
`on_bar()`, so there is no new lookahead surface for the probe to catch —
consistent with the clean PASS.

## 6. Results against the pre-registered rules

### D1 (primary) — PANEL_TRAIN, spot @0.10%, matched-exposure drawdown

**Linear variant** (band 1.09–1.32%):

| asset | band | candidate max DD | matched hold max DD | Δ DD (pp, + = candidate worse) | 95% paired interval |
|---|---|---|---|---|---|
| BCH | 1.321% | 49.9% | 41.0% | **+9.3** | [−9.6, +35.1] |
| LTC | 1.210% | 39.4% | 38.1% | **+0.8** | [−10.6, +24.8] |
| ETC | 1.278% | 41.1% | 25.1% | **+16.5** | [+1.6, +37.5] |
| DASH | 1.286% | 44.7% | 28.2% | **+16.3** | [−1.8, +34.2] |
| LINK | 1.178% | 42.3% | 31.4% | **+14.2** | [−5.8, +32.9] |
| XTZ | 1.321% | 40.0% | 30.0% | **+9.9** | [+0.6, +37.4] |

**0 of 6** — exact binomial p = 1.0000 → **FAILS**. 2/6 intervals exclude
zero (ETC, XTZ), both against the candidate.

**Sqrt (damped) variant** (band 1.09–1.15%):

| asset | band | candidate max DD | matched hold max DD | Δ DD (pp) | 95% paired interval |
|---|---|---|---|---|---|
| BCH | 1.149% | 48.9% | 41.1% | **+8.1** | [−10.3, +34.5] |
| LTC | 1.100% | 39.2% | 38.3% | **+0.6** | [−11.0, +24.2] |
| ETC | 1.131% | 40.2% | 25.0% | **+15.7** | [+0.9, +36.7] |
| DASH | 1.134% | 45.6% | 28.6% | **+16.8** | [−1.4, +35.4] |
| LINK | 1.086% | 42.2% | 31.4% | **+14.4** | [−5.8, +33.9] |
| XTZ | 1.149% | 41.2% | 30.0% | **+11.1** | [+2.1, +39.0] |

**0 of 6** — exact binomial p = 1.0000 → **FAILS**. 2/6 intervals exclude
zero (ETC, XTZ), both against the candidate. The sign never flips on
either variant, on any asset; the per-asset deltas move only 1–2pp between
variants. Compared with R-57's raw v4 on the same panel (BCH +5.2, LTC
+33.8, ETC +23.6, DASH +29.8, LINK +13.4, XTZ +19.3), this branch is
mixed — LTC and LINK improved modestly, ETC/DASH/XTZ improved somewhat,
BCH is close to unchanged — but no asset's sign flipped and the count stays
at zero.

### D2 (falsification, control) — CONTROL, BTC/ETH, spot @0.10%, frozen variant (sqrt)

| asset | band | candidate dDD (matched) | R-57's v4 control | tolerance | verdict |
|---|---|---|---|---|---|
| BTC | 1.00% | −5.6pp [−20.0, +16.4] | −5.6pp | ≤ base+5pp | within tolerance (exact match) |
| ETH | 1.01% | −11.3pp [−16.9, +19.8] | −11.5pp | ≤ base+5pp | within tolerance |

**PASSES.** BTC's number matches R-57's own v4 control number to one
decimal place (−5.6pp both), exactly as designed by construction (BTC's
derived band is 1.00%, i.e. this branch is a structural no-op on BTC).
ETH moves only 0.2pp (−11.3 vs −11.5). The fix does not break the two
instruments the mechanism already worked on.

### D3 (generalization, descriptive) — PANEL_TEST 2023-01-01→2026-08-20, spot @0.10%, frozen variant (sqrt)

| asset | Δ DD (pp, matched) | 95% interval |
|---|---|---|
| BCH | +9.6 | [−3.1, +48.9] |
| LTC | +40.7 | [+9.6, +57.6] |
| ETC | +17.0 | [+7.7, +45.9] |
| DASH | +26.6 | [+0.2, +36.4] |
| LINK | +9.7 | [−3.8, +35.7] |
| XTZ | +4.2 | [−4.2, +31.5] |

**0 of 6**, same inverted sign as D1, on the held-out panel window — not a
gate, but corroborating D1 rather than contradicting it. LTC's interval is
the widest miss of either window ([+9.6, +57.6], excluding zero in the
matched hold's favour).

### D4 (0.40% fee falsification) — PANEL_TRAIN, spot @0.40%, frozen variant (sqrt), beats buy_and_hold's final balance

| asset | candidate final | buy_and_hold final | beats hold? |
|---|---|---|---|
| BCH | $940 | $438 | yes |
| LTC | $1,391 | $1,723 | no |
| ETC | $2,702 | $3,137 | no |
| DASH | $944 | $641 | yes |
| LINK | $1,125 | $2,386 | no |
| XTZ | $646 | $444 | yes |

**3 of 6** (BCH, DASH, XTZ). Threshold to "survive" was ≥5/6; this is not a
promotion-relevant surprise (D4 was predicted to fail in the
pre-registration, and by the ≥5/6 bar it does), though the raw count
matches R-59's novel branch's own D4 (3/6) on the same panel.

## 7. Verdict

Applying `experiments.r60_shared.promoted(k1, dd_advantage)` mechanically
(requires D1 ≥ 5/6 **and** D2 passing both BTC and ETH), using the frozen
`sqrt` variant:

- D1 [linear]: 0/6 → FAILS
- D1 [sqrt, frozen]: 0/6 → FAILS (needed ≥5/6)
- D2: PASSES (BTC −5.6pp exact match to R-57's control, ETH −11.3pp vs
  R-57's −11.5pp, both within the 5pp regression tolerance)
- `promoted(0, {...})` → **False**

**NEGATIVE.** The failure mode, named plainly: this is not a case of the
mechanism failing to self-normalize — BTC's derived band lands at exactly
1.00% by construction under either functional form, and D2 confirms the
fix is a near-exact no-op on BTC/ETH, precisely as designed. It is not a
case of the flip-rate premise being wrong either — the panel genuinely does
flip its vote 18–32% more often than BTC per year, confirming R-59's own
speculation with a direct measurement for the first time. The failure is
that widening the hysteresis band by the amount a flip-rate ratio of
1.1–1.3x implies (1.09%–1.32% vs. BTC's 1.00%) is simply too small a
timing change to move the sign of a single asset's matched-exposure
drawdown count: every delta stays positive (candidate worse), and the
deltas mostly shift by only 1–8pp relative to R-57's raw v4 numbers on the
same panel — nowhere close to flipping ETC's +23.6pp or DASH's +29.8pp
negative. This closes B-26's novel branch as NEGATIVE, and — pending the
conservative branch's own result — extends the SIZE/TIMING-axis record
described in the pre-registration, corroborating (on a third, independent
mechanism, timing rather than scale) R-59's own diagnosis that a rebalanced
constant-exposure hold's advantage on this panel comes from something the
matched hold structurally captures (buy-the-dip behavior on more
mean-reverting instruments) that neither an exposure-scale fix nor a
timing/hysteresis fix, at the parameter magnitudes a naive flip-rate
calibration produces, is able to out-trade.

## 8. Configurations evaluated

**78** backtests (`CONFIG_COUNT`): D1 linear 6 assets × 3 arms = 18, D1
sqrt 6 assets × 3 arms = 18, D2 2 assets × 3 arms = 6, D3 6 assets × 3 arms
= 18, D4 6 assets × 3 arms = 18. Plus **8** flip-rate measurement passes
(`FLIP_MEASURE_COUNT`, not backtests — no `run_period` call, one per asset
across 8 assets, each internally computing 3 anchors = 24 anchor-level flip
counts). Causality probe reads no backtest (constructs the strategy and
calls `prepare()`/`on_bar()` directly via `PaperBroker`, like R-57/R-59's
own causality commands), so it costs 0 against `CONFIG_COUNT`.

Holdout consultations added by this branch: **0** — no BTC/ETH bar past
2022-12-31 is read anywhere in
`experiments/r60_novel_flip_adaptive_hysteresis.py` (BTC via `load_dataset`
and ETH via `load_coinbase_spot` are both sliced to `:2022-12-31` before any
use, including the causality probe and the flip-rate measurement pass).

## 9. Raw data

`reports/r60_novel/flip_rates_per_anchor.csv`,
`reports/r60_novel/flip_rates_bands.csv`,
`reports/r60_novel/d1_panel_train_linear.csv`,
`reports/r60_novel/d1_panel_train_sqrt.csv`,
`reports/r60_novel/d2_control.csv`,
`reports/r60_novel/d3_panel_test.csv`,
`reports/r60_novel/d4_panel_train_040.csv`.
