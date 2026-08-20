# R-57 — six instruments the strategy was never fitted on (08-20)

Unregistered experiment. Code: `experiments/r57_cross_asset_panel.py`,
data fetched by `scripts/fetch_coinbase_panel.py`. Nothing under
`src/tradebot/strategies/` is touched; `kelly_regime_v4` runs byte-identical
in every cell below. Pre-registration (and its one amendment) committed
before any backtest: `docs/LEDGER.md`, "R-57 pre-registration", commits
`c22ba3e` and `8a7fa5b`.

## 1. The question, one sentence

The only positive claim this project still leans on — *the risk property
transfers, the return property does not* (R-17, L-01, confirmed once on ETH
by R-47) — rests on two correlated assets, one of which is the asset the
strategy was fitted on; so run the frozen strategy on six instruments it has
never seen and **count**, against a benchmark carrying its own exposure.

## 2. The panel

Coinbase USD spot, 5-minute, 2020-01-01 → 2026-08-20, selected by the
mechanical liquidity rule in the pre-registration (three fixed 2020 probe
days, ranked by dollar volume, then a continuity/coverage gate):

| asset | bars | coverage | largest gap | 20d anchor spans |
|---|---|---|---|---|
| BCH | 697,063 | 99.9% | 6h35m | 20.0d |
| LTC | 697,357 | 99.9% | 6h35m | 20.0d |
| ETC | 638,633 | 91.5% | 6h40m | 21.9d |
| DASH | 572,840 | 82.1% | 6h35m | 24.4d |
| LINK | 697,301 | 99.9% | 6h35m | 20.0d |
| XTZ | 635,163 | 91.0% | 6h40m | 22.0d |
| ~~XRP~~ | 436,839 | 62.6% | **905 days** | — (excluded: Coinbase suspended XRP-USD 2021-01 → 2023-07, exactly the hole the continuity rule was written for) |

Causality tamper probe (the `test_causality_strict.py` methodology, run
against each new loading path because that module hard-codes the BTC
loader): **PASS on all six** — decisions at and before the cut are identical
under opposite post-cut tampers.

## 3. Results against the pre-registered rules

### D1 (primary) — drawdown vs a hold carrying v4's own mean exposure, spot @0.10%, 2020-04-01 → end

| asset | c (v4's mean notional) | v4 max DD | matched hold max DD | Δ DD (pp, + = v4 worse) | 95% paired interval |
|---|---|---|---|---|---|
| BCH | 0.29 | 52.3% | 47.5% | **+5.2** | [−6.1, +45.7] |
| LTC | 0.31 | 74.7% | 42.5% | **+33.8** | [+2.1, +53.1] |
| ETC | 0.22 | 51.3% | 29.5% | **+23.6** | [+5.3, +45.9] |
| DASH | 0.19 | 58.7% | 29.7% | **+29.8** | [+2.5, +41.8] |
| LINK | 0.28 | 47.8% | 38.1% | **+13.4** | [−5.1, +39.8] |
| XTZ | 0.20 | 55.0% | 35.4% | **+19.3** | [+3.3, +44.8] |

**0 of 6.** Not "the advantage shrinks" — the sign **inverts on every
asset**, and 4 of the 6 intervals exclude zero, all four against v4. The
pre-registered verdict is **FAILS TO REPLICATE**, by the widest possible
margin the rule allows.

### D2 (the pre-registered falsification test) — 0.40% taker

v4 beats `buy_and_hold`'s final balance in **2 of 6** (DASH, XTZ — both
assets where holding lost 51% and 87% respectively, so it is a low bar
cleared by holding less, not by trading well). Threshold was ≥5/6.
**FAILS, exactly as predicted before the run.**

### D3 (context, not evidence) — the comparison the README table makes

Against the *fully-invested* `buy_and_hold`, v4's drawdown is lower in
**6 of 6** on spot and **6 of 6** on futures, by 16–46pp. The identical six
assets, the identical runs: 6/6 unmatched, 0/6 matched. This is R-33's
finding reproduced on new instruments and larger — on BTC the matched
comparison merely erased the advantage, here it reverses it.

### D4 — the 2022 bear window (2022-05-01 → 2022-11-30), descriptive

Matched **0/6** on both markets; unmatched **6/6** on both. v4 preserves
capital through the bear in absolute terms (final balances 0.80–1.25× on
spot) — but so does any arm holding 11–22% of equity, which is what the
matched arm holds, and it does so with less drawdown in every cell.

### Robustness — equal-realized-volatility matching instead of mean notional

All six cells valid (match residual ≤ 0.8%). v4's drawdown is lower in
**2 of 6** (BCH 52.3% vs 63.5%, XTZ 55.0% vs 61.5%) and worse in 4; v4's
final balance is lower in **6 of 6**. The two matching axes disagree about
drawdown on two assets and agree about return on all six.

### Return at matched exposure — R-36/B-14's confirmed claim, not pre-registered here

v4 out-returns the mean-notional-matched hold on **1 of 6** (DASH), and all
six paired growth intervals contain zero. On the volatility-matched axis it
is **0 of 6**. R-36 pre-registered and confirmed this claim on BTC (median
+20.8pp/+23.8pp per window, thinning ~10× outside the 2017–2020 bull); it
does not reproduce on any panel asset. Flagged as a by-product, not a
verdict: this round's pre-registration made drawdown the primary rule.

## 4. Post-hoc control — asset-specific, or period-specific?

Run *after* D1 returned 0/6, labelled as a control rather than a decision
rule, and truncated at 2022-12-31 so no 2023+ BTC bar is read (holdout
counter unchanged). Same comparison, same window for everyone:

| asset | v4 DD | matched hold DD | Δ DD (pp) | 95% interval |
|---|---|---|---|---|
| **BTC** | 33.2% | 39.3% | **−5.6** | [−20.0, +16.4] |
| **ETH** | 27.3% | 40.2% | **−11.5** | [−17.3, +19.6] |
| BCH | 47.5% | 41.8% | +6.0 | [−11.7, +33.3] |
| LTC | 38.7% | 38.1% | +0.0 | [−10.8, +24.6] |
| ETC | 40.0% | 25.0% | +15.4 | [+0.8, +36.8] |
| DASH | 46.0% | 28.8% | +17.1 | [−1.2, +36.5] |
| LINK | 42.2% | 31.4% | +14.5 | [−5.6, +34.2] |
| XTZ | 41.8% | 30.1% | +11.6 | [+2.9, +40.2] |

**2 of 8, and they are exactly BTC and ETH.** In the same window, on the
same code, the property is present on the two assets this project has always
measured it on and absent (inverted) on all six new ones. The failure is
**asset-specific, not period-specific** — which is the sharper and worse of
the two readings.

## 5. Why this might be, named as hypotheses, not findings

- v4's `target_vol=0.55` / `max_leverage=2.0` were fitted to BTC's
  volatility level. On assets whose realized volatility runs far above that,
  the scale term is small and nearly always binding: over the shared
  2020-04→2022-12 control window v4's mean notional is 0.18–0.26 on the
  panel against 0.38 on BTC and 0.34 on ETH. What is left of the mechanism
  is then mostly the *timing* of the 20/40/80-day vote.
- A constant-exposure arm that rebalances back to `c` is quietly a
  buy-the-dip rule. In a higher-volatility, more mean-reverting instrument
  that is worth more, and a trend-latched gate that stands aside after a
  drop is worth less. The matched arm's advantage is largest exactly where
  v4's notional is smallest (DASH, ETC, XTZ).
- Nothing here indicts the *engine*: the same code produces the expected
  favourable BTC and ETH numbers in section 4.

## 6. Accounting

- **Backtest configurations evaluated: 130** (114 in the frozen matrix,
  16 in the post-hoc control). No sweep, no selection — the only search is
  the volatility-matching solver, whose iterations are counted anyway.
- **Holdout consultations: +0.** No 2023+ BTC bar is read anywhere in this
  module; the panel assets have never fitted a parameter in this project.
- **Decision rule moved: no.** D1–D4 are as committed. The one amendment
  (the asset-selection coverage clause) was made before any backtest ran and
  is recorded in full in both the ledger and the module docstring.
- `pytest`: 461 passed.

## 7. What this changes

The project's headline drawdown claim now has a measured scope: it holds on
BTC and on ETH, and on six further instruments it does not hold at all. Read
against a fully-invested benchmark it looks like a property of the strategy
in 8 of 8 cases; read against a benchmark carrying the same exposure it is a
property of BTC and ETH in 2 of 8. Whatever `kelly_regime_v4` is doing, it
is not a general regime-sizing mechanism — it is a mechanism calibrated to
two instruments, and this is the first measurement that could tell the
difference.
