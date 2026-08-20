# R-62 conservative branch — `kelly_regime_v4`'s volatility target, alone

**Verdict: NEGATIVE.** The conditional volatility-target factor, run with the
directional vote deleted, reproduces the matched-exposure drawdown property
neither on the panel (D1 **2/6**) nor on the BTC/ETH control where v4 itself
holds it (D3 **0/2**).

## Direction

Backlog item **B-27**'s literal request: `kelly_regime_v4`'s exposure is a
product, `desired[i] = frac[i] * scale[i]`, and no round in this project has
asked what either factor does *alone*. R-59 varied `scale`'s magnitude and
R-60 varied `frac`'s timing; both kept both factors. This branch deletes
`frac` — forces it to `1.0` at every bar, so the strategy never stands aside
— leaving only `kelly_regime_v3`'s conditional/extreme-only volatility
target. Zero new parameters: every constant is v4's own shipped default
(`target_vol=0.55, max_leverage=2.0, vol_span=8d, anchor_span_days=180,
high_in=1.70, high_out=1.20, low_in=0.55, low_out=0.85, deadband=0.10`).
Nothing is tuned; a component is removed.

Pre-registration: `experiments/r62_shared.py`, committed before any number
below was read. Literature: Wang & Yan (2021, JBF 131, 106198) decompose
volatility-managed performance into a *volatility timing* component and a
*return timing* component — this branch isolates the former.

## What was done

`experiments/r62_conservative_voltarget_only.py`, class `VolTargetOnly`:
`KellyRegimeV3.prepare()` copied byte-for-byte with the vote block replaced
by `frac = np.ones(len(df))`. Run frozen against R-57's six-asset Coinbase
panel (BCH, LTC, ETC, DASH, LINK, XTZ), FULL window 2020-04-01 → last bar,
spot @0.10% (D1) and @0.40% (D4), BEAR22 descriptive, plus the BTC/ETH
control window 2020-04-01 → 2022-12-31 (D3).

Every cell matches the candidate against a `ConstantExposureHold` carrying
**the candidate's own** mean clipped notional, so each arm is compared at its
own exposure level rather than against a fully-invested benchmark (the
standing R-33 rule).

**Holdout consultations added: 0** — the control window ends 2022-12-31; no
panel asset ever fitted anything.

## Result

D1 — FULL, spot @0.10%, Δ max drawdown vs the candidate's own matched hold
(negative = candidate draws down less):

| asset | cand final | cand DD | c_mean | matched hold final | hold DD | ΔDD (pp) | 95% CI |
|---|---|---|---|---|---|---|---|
| BCH | $1,138 | 76.6% | 0.63 | $1,811 | 80.5% | **−4.16** | [−9.90, +13.41] |
| LTC | $1,099 | 74.2% | 0.64 | $1,826 | 74.5% | **−0.14** | [−9.14, +15.13] |
| ETC | $1,874 | 70.5% | 0.54 | $2,936 | 68.5% | +3.43 | [−8.81, +14.37] |
| DASH | $1,771 | 72.9% | 0.45 | $1,937 | 64.5% | +9.23 | [−11.94, +13.88] |
| LINK | $3,372 | 63.4% | 0.54 | $5,521 | 61.6% | +2.72 | [−8.79, +17.01] |
| XTZ | $442 | 79.4% | 0.50 | $898 | 75.8% | +3.25 | [−3.72, +18.69] |

**D1 = 2/6** (exact binomial p = 0.891) → **FAILS TO REPLICATE**. Zero of six
intervals exclude zero in either direction — the effect is not merely absent,
it is unmeasurable at this sample size.

**D3 (BTC/ETH control) = 0/2.** This is the branch's most informative number.
On the identical window where `kelly_regime_v4` itself scores 2/2 (BTC
−5.55pp, ETH −11.46pp, R-57's `control_pre2023.csv`), the vol-target-only arm
inverts on both: BTC **+3.05pp** [−5.65, +9.46], ETH **+3.70pp** [−8.40,
+14.85]. Removing the vote does not merely weaken the property — it destroys
it on the two assets that had it.

**D4 (0.40% taker, beats `buy_and_hold` final balance) = 4/6** (wins BCH,
ETC, DASH, XTZ; loses LTC, LINK). Recorded as context, not support: the same
4/6 holds at 0.10%, and the arm's drawdowns are 63–80% throughout, so this is
a return count on a strategy carrying near-full exposure, not a risk finding.
It is the one place this branch outperformed its own pre-registered
prediction (which named a fee-tier failure), and it should be read against
the fact that the matched-hold comparison — the one that controls for
exposure — goes the other way.

BEAR22 (descriptive, n=1 window/asset): matched ΔDD favours the candidate on
4 of 6 (BCH −6.14, LTC −10.69, DASH −10.90, LINK −2.67, XTZ −2.32; ETC
+11.93), every interval containing zero.

**Causality tamper probe: PASS** on BCH and LTC — decisions identical under
opposite ×3 / ÷3 post-cut tampers of the future. Full suite: 461 passed.

**Configurations evaluated (this branch): 60 backtests** (6 assets × 3
window/fee combinations × 2 arms/cell, plus 2 control assets × 2). Summed
with the novel branch for the round's trials count.

## Interpretation

The volatility-targeting machinery, isolated, produces no matched-exposure
drawdown advantage anywhere — not on the panel it was never fitted on, and
not on BTC and ETH where the full strategy demonstrably has one. Since v4's
`scale` factor is exactly Harvey et al. (2018)-style mechanical tail
protection, and since a constant-exposure hold at the same mean notional is
precisely the arm that neutralizes an exposure-level artifact, the reading is
that vol-targeting's tail benefit here is *entirely* about the exposure level
it selects, not about the path it takes to get there. That is consistent with
this project's own R-33/R-57 lineage, in which repeated drawdown findings
dissolved once the benchmark was de-levered to match — and it is the first
time the point has been made about the sizing machinery in isolation rather
than about a comparison.

Read together with the novel branch (vote-only, D3 = 2/2, D1 = 0/6), this
localizes v4's property: it lives in the directional vote, and the
volatility target contributes none of it. In Wang & Yan's terms, the effect
is return timing, not volatility timing. Neither arm is promotable, and this
one does not meet the further-work bar (requires D1 ≥ 5/6; got 2/6).
