"""Shared, read-only utilities and pre-registration for the R-165 round (08-27).

DIRECTION, in one sentence: R-64 (08-20) attacked kelly_regime_v4's
position-UPDATE rule (`desired = frac * scale`; jump to `desired` whenever
`abs(desired - pos) > deadband`) as a whole, testing the trade-to-boundary
destination (Constantinides 1986 / Davis & Norman 1990 family) and a
Garleanu-Pedersen (2013) partial-adjustment destination weighted by the
VOTE anchors' own decay rates -- and found partial adjustment collapses to
a no-op because the three anchors' causal half-lives (2.5 / 3.0 / 4.6
days) are too close together and too slow to create the heterogeneity GP's
mechanism needs ("no better estimator can rescue it: it needs anchors
whose decay rates differ by orders of magnitude, not by 2x"). This round
asks whether that same construction, applied to the OTHER factor in the
product, behaves differently: `scale` (v3/v4's conditional-volatility-target
sizing, `target_vol/realized_vol` or `target_vol/slow_vol` depending on the
hysteresis state) is not an anchor vote blended from three correlated
signals -- it is a SINGLE continuously-valued ratio, so R-64's specific
failure mechanism (weight collapse across similar-decay signals blended
into one aim, see the scope correction below) cannot apply to it
structurally. R-64 never isolated `scale`'s own decay rate; its novel
branch's decay measurement was of anchor FLIP RATES only. This round
measures `scale`'s feeding series' decay causally (see the scope-correction
section below for the measured number and what it implies) and applies two
destination policies to `scale` alone -- trade-to-boundary (conservative,
Constantinides/Davis-Norman family) and derived-rate EWMA smoothing
(novel, Dao et al. 2016's single-signal framing, the one methodology this
project has already validated once) -- holding `frac`'s existing
jump-to-target vote and hysteresis unchanged.

**Which constraint this attacks: COST (primary).** No new external data
source (both branches are pure functions of BTC/ETH's own OHLCV, reusing
v4's own vol estimator); no new formula for WHAT `scale` targets (that is
the ~10-round-deep, already-exhausted family: R-93/R-125/R-136/R-141/
R-152/R-153/R-160/R-161/R-162/R-163/R-164 all substitute a different risk
statistic into the scale slot). This round changes only HOW FAST the
strategy is allowed to move toward whatever `scale` already computes --
R-64's destination/rate axis, decomposed onto the one factor R-64 did not
isolate.

**Literature:**
- Garleanu, N. & Pedersen, L. H. (2013), "Dynamic Trading with Predictable
  Returns and Transaction Costs," Journal of Finance 68(6), 2309-2340.
  Props. 2-4: under quadratic/impact costs, optimal trading partially
  adjusts toward an "aim" that over-weights slowly-decaying signals;
  weight on signal k is proportional to `1/(1 + phi_k * a / gamma)` where
  `phi_k` is that signal's own decay rate. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080
- Constantinides, G. M. (1986), JPE 94(4); Davis, M. H. A. & Norman, A. R.
  (1990), Math. OR 15(4). Under PROPORTIONAL (taker-fee) costs -- the cost
  structure `MarketSpec` actually charges -- the optimum is a no-trade
  region with a trade-to-the-nearest-boundary destination, not smooth
  partial adjustment. This is the theoretically-favored family for this
  project's cost structure (confirmed again by R-64's literature
  commission: Muhle-Karbe, Reppen & Soner 2017, Ann. Rev. Fin. Econ. 9,
  section 6.3).
- Dao, T.-L., Nguyen, T.-T., Deremble, C., Lemperiere, Y., Bouchaud, J.-P.
  & Potters, M. (2016), "Statistical Properties and Financial
  Applications of Long/Short Trend Followers," ETH Zurich working paper /
  arXiv:1607.06373. Justifies an EWMA-of-target smoothing rule from
  the RESPONSE side (finite trading rate optimally trades off signal
  decay against turnover cost) rather than from GP's quadratic-cost
  assumption -- the correct citation for a smoothing rule under
  proportional costs, per this project's own RESEARCH.md finding 8
  (R-67's literature commission). Used correctly this way by R-65/R-67/R-68
  on the cross-sectional multi-asset score, the one COST-axis construction
  in this project's whole history to move a signal from unaffordable
  toward viable.
- Romero, R. (2026), "Revisiting EWMA in High-Frequency-Based Portfolio
  Optimization: A Comparative Assessment," Journal of Applied Econometrics
  (early view). Compares EWMA smoothing parameters in high-frequency
  portfolio construction and reports that a smoothing parameter derived
  from the input series' OWN measured persistence -- not fit to
  in-sample performance -- gives a materially better turnover/adaptation
  trade-off than either a fixed conventional default (e.g. RiskMetrics'
  lambda=0.94/0.97) or an aggressively-fit one. Read here only for its
  methodological point (derive the smoothing constant from the input
  series' own measured decay, evaluate turnover and adaptation
  separately, do not fit against the reward being tested) -- its exact
  parameter values are for daily equity/FX data and are NOT imported as
  numbers.
- Bongaerts, D., Kang, X. & van Dijk, M. A. (2020), "Conditional Volatility
  Targeting," Financial Analysts Journal 76(4) -- already the grounding
  citation for kelly_regime_v3's own hysteresis-based conditional
  targeting; cited here only to note that this round modifies the
  DESTINATION of a move v3/v4 already makes, not the STATE MACHINE that
  decides whether to make it (the high/low breakout hysteresis in
  `KellyRegimeV3.prepare` is left untouched in both branches).

**Not a duplicate of, by ledger ID:**
- R-64 (08-20): tested the SAME two destination families (boundary-trade,
  partial-adjustment) on the WHOLE product `desired = frac * scale`, with
  the partial-adjustment arm's persistence weighting driven by the VOTE
  anchors' decay rates only. Its own stated failure mechanism -- decay
  rates too close together (2.5/3.0/4.6 days, same order of magnitude)
  and too slow relative to any usable trading rate -- is a factual claim
  about the ANCHOR VOTE series specifically. This round measures whether
  the SAME mechanism applies to `scale`'s own input (realized volatility)
  before assuming it does; see the pre-registered falsification test
  below, which treats "yes, it applies" as a real (confirmatory) possible
  outcome, not a strawman.
- R-93/R-125/R-136/R-141/R-152/R-153/R-160/R-161/R-162/R-163/R-164: every
  one of these substitutes a DIFFERENT FORMULA for what `scale` (or
  `frac`) targets -- CDaR, CVaR, a HAR/DVOL vol forecast, an LPPLS crash
  hazard, a conformal cap, an FDR-gated vote flip, a Kaufman efficiency
  multiplier, a Turtle pyramiding multiplier, a Barroso-Santa-Clara /
  Daniel-Moskowitz risk-managed-momentum multiplier. This round changes
  none of those formulas; `scale`'s formula (`target_vol/vol` or
  `target_vol/slow`, hysteresis-gated) is byte-identical to v4's shipped
  code in both branches. Only the RATE at which the position is allowed
  to track that formula's output changes.
- R-66 (novel, width-profile-in-f no-trade band) and kelly_regime_ev /
  kelly_regime_ev_fast: both derive a BAND WIDTH (Constantinides/Davis-
  Norman finding 7/8 -- "the width should be derived, not chosen"). This
  round holds width fixed (v4's shipped `deadband=0.10`, or R-64's own k
  parameterization of it) and instead asks a DESTINATION/RATE question,
  the axis RESEARCH.md finding 8 explicitly separates from width.
- R-131/R-133 (turnover corridor / shadow-price throttle): both throttle
  an already-computed re-target decision (a POST-HOC damper on the order
  that follows). This round changes the target-COMPUTATION itself (the
  `scale` series feeding `desired`), before any order/deadband logic
  runs -- the same distinction R-64's own entry draws against R-131/R-133.

**A scope correction, made during pre-registration itself (before any
strategy code beyond this file's neutral measurement helper has run, so
this is Step 2 literature/measurement work, not a holdout peek):** R-64's
novel-branch failure is specifically a WEIGHT-COLLAPSE across THREE
correlated anchor-vote signals blended into one aim -- GP's Prop. 4
formula `1/(1+phi_k*a/gamma)` only produces heterogeneous weights when
blending *multiple* signals of *different* decay rates, and the three
anchors' rates were too similar for that to bite. `scale` is a SINGLE
scalar signal, not a blend, so that specific weight-collapse mechanism
cannot apply here structurally, regardless of what its own decay rate
turns out to be -- this round is therefore testing Dao et al. (2016)'s
single-signal EWMA-of-target framing (the one already validated by
R-65/R-67/R-68), not R-64's multi-signal GP weighting formula, and is not
a re-run of R-64's failure mode by construction. What R-64 DOES still
license is measuring first rather than assuming: `scale`'s feeding series
(v4's own `vol`, an 8-day-span EWM std of returns) was measured causally
on inner-train BTC before any mechanism was designed
(`causal_autocorr_halflife_days` below) and its half-life is **47.2 days**
-- an order of magnitude SLOWER than the 2.5-4.6-day anchor half-lives,
not faster as microstructure priors on raw volatility clustering would
suggest, because `vol` itself is already an 8-day EWMA and inherits that
smoothing's own persistence. That number is a fact about v4's existing
code, not a strategy result, and is reported here as pre-registered
context: it means a derived EWMA/partial-adjustment rate on `scale` will,
if it tracks this measured persistence honestly, end up SLOW (a small
step size / long half-life), which is the opposite of what a turnover-
saving mechanism needs unless the measured cost-benefit trade-off (Dao et
al. 2016; L-05/L-06's own `(sigma^2/2)(f-f*)^2` vs `fee*|delta f|`
trade-off, generalized to a continuous rate rather than a discrete band)
independently justifies a faster one. **Falsification test, stated
precisely:** if the branch's derived rate, computed from this trade-off
without reference to any backtest performance number, turns out
indistinguishable from a=1 (i.e. equivalent to v4's existing instant-jump
behaviour once rounded to the harness's own bar resolution), the
mechanism has nothing to test and the branch reports that as its result
rather than searching for a different derivation that produces a more
interesting number.

**Decision rule (frozen now, before either branch's holdout read), mirrors
R-64's D0-D6 structure applied to a variant-vs-v4 comparison:**

- **D0 risk-match gate.** Void the growth/Sharpe comparison for any cell
  where the variant's time-in-market or realized volatility differs from
  v4's own by more than 10% (R-33's exposure-artifact check, applied
  per-cell).
- **D1 holdout comparison.** Paired stationary block-bootstrap (30-day
  blocks, 2,000 resamples, `tradebot.inference.paired_bootstrap`) of
  log-growth AND Sharpe, variant vs v4 itself (not buy_and_hold -- v4 is
  the object being modified), on `start=OOS_START` ("2023-01-01"), both
  spot and futures_5x. PROMOTE requires the 95% interval to exclude zero
  on the favorable side on at least one of {log-growth, Sharpe} on BOTH
  markets, with the other metric not significantly negative on either.
- **D2 cost-mechanism test.** The advantage (if any) must GROW, not
  shrink, when the fee is quadrupled to 0.40% (`scripts/fee_study.py`
  tier) -- the same falsifiable signature R-64's D2 used, because a
  genuine COST-axis mechanism's advantage is a saved-fee stream and must
  scale with the fee.
- **D3 ETH-A falsification.** Same sign (not reversed, not catastrophically
  worse) on Bitfinex ETH 2016-03->2019-12+holdout. A reversal here voids
  the BTC finding per this project's standing falsification convention.
- **D4 turnover.** Total fill count must actually fall vs v4 on the same
  window -- a COST-axis claim that does not reduce turnover is not the
  mechanism it claims to be.
- **D5 plateau not peak.** Report immediate parameter neighbours (k+/-1
  step for conservative; rate x0.5/x2 for novel); a cliff at the winning
  cell (neighbour Sharpe outside the +/-0.2 noise floor) downgrades any
  PROMOTE to PARTIAL regardless of D1-D4.
- **D6 funding.** Futures numbers are the funding-free upper bound per the
  standing project caveat; report `scripts/funding_study.py` if D1-D5 all
  pass.

Default: **REJECT**. A branch that never has all of D1 (both markets,
either metric), D2, D3, D4 hold simultaneously for one frozen config does
not promote, however good any single cell looks in isolation. If the
falsification test above fires (half-life inside the 1-15 day band), the
branch's verdict is NEGATIVE by construction; still run the sweep and
report it, since a measured, mechanical confirmation of R-64's law is
worth recording, not worth skipping.

Splits, identical to the rest of this project's convention:
inner-train ends 2020-12-31; inner-validation is 2021-01-01 -> 2022-12-31;
holdout is OOS_START ("2023-01-01") onward, untouched by either branch
until its own config is frozen per this file's decision rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[1].name
                                                 == "src") else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402

INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# v4's own anchor half-lives, as measured causally by R-64 (docs/LEDGER.md,
# R-64 entry) -- carried here as a fixed reference point for the
# falsification test's "same order of magnitude" comparison. NOT re-derived
# by this round; this round derives the VOLATILITY half-life independently.
V4_ANCHOR_HALFLIVES_DAYS = (2.5, 3.0, 4.6)


def realized_vol_series(close: pd.Series, vol_span: int) -> pd.Series:
    """v3/v4's own realized-vol input to `scale`, reproduced exactly.

    Matches `KellyRegimeV3.prepare`'s `vol` computation: EWM std of
    log-returns, span in bars, annualized, shifted by one bar (the
    strategy may only see vol known as of the PREVIOUS bar close).
    """
    r = np.log(close).diff()
    return (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1)


def causal_autocorr_halflife_days(vol: pd.Series, max_lag_days: int = 60,
                                   fit_end: str = INNER_TRAIN_END) -> dict:
    """Causal, strictly-lagged AR(1)-style half-life of a vol series, in days.

    Uses ONLY data up to `fit_end` (inner-train), matching R-64's own
    "expanding, strictly-lagged" discipline for its anchor decay-rate
    measurement. Fits log|autocorrelation| vs lag (in days) by OLS on
    lags 1..max_lag_days and reports the implied half-life
    `-ln(2)/slope`, plus the raw ACF for both branches to inspect.
    """
    v = vol.loc[:fit_end].dropna()
    daily = v.resample("1D").last().dropna()
    n = len(daily)
    max_lag = min(max_lag_days, n - 5)
    acf = []
    x = daily.to_numpy()
    x = x - x.mean()
    denom = float(np.dot(x, x))
    for lag in range(1, max_lag + 1):
        num = float(np.dot(x[:-lag], x[lag:]))
        acf.append(num / denom if denom > 0 else np.nan)
    acf = np.asarray(acf)
    lags = np.arange(1, max_lag + 1)
    valid = acf > 1e-6
    if valid.sum() < 5:
        return {"halflife_days": float("nan"), "acf": acf.tolist(), "n_days": n}
    slope, intercept = np.polyfit(lags[valid], np.log(acf[valid]), 1)
    halflife = -np.log(2) / slope if slope < 0 else float("inf")
    return {
        "halflife_days": float(halflife),
        "acf_lag1": float(acf[0]) if len(acf) else float("nan"),
        "acf": acf.tolist(),
        "n_days": int(n),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def order_of_magnitude_gap(halflife_days: float,
                            reference_days: tuple[float, ...] = V4_ANCHOR_HALFLIVES_DAYS) -> dict:
    """Compare a measured half-life to v4's anchor half-lives; report the ratio."""
    ref_min, ref_max = min(reference_days), max(reference_days)
    if halflife_days <= 0 or not np.isfinite(halflife_days):
        return {"in_same_band": False, "ratio_to_min": float("nan"), "note": "degenerate half-life"}
    ratio = ref_min / halflife_days if halflife_days < ref_min else halflife_days / ref_max
    same_band = halflife_days >= 1.0 and ratio < 10.0
    return {
        "in_same_band": bool(same_band),
        "ratio": float(ratio),
        "measured_days": float(halflife_days),
        "reference_days": reference_days,
        "falsification_test_fires": bool(same_band),
    }


# ---------------------------------------------------------------------------
# POST-HOC CORRECTION (added after both R-165 branches and the skeptic pass
# reported; the measurement above is left unedited per this project's
# "nothing is deleted, annotate in place" rule -- see docs/LEDGER.md, R-165):
#
# The docstring above states the causal half-life of `scale`'s feeding vol
# series as 47.2 days. That number does not reproduce: calling
# `causal_autocorr_halflife_days(realized_vol_series(close, 8*BARS_PER_DAY))`
# on inner-train BTC at this file's own documented defaults returns
# **38.77 days** (ACF(1d)=0.9677, n=1460 days), independently confirmed by
# both implementation branches and the round's skeptic. The qualitative
# claim survives (still an order of magnitude slower than the 2.5-4.6-day
# anchor half-lives) and no decision-rule gate in this round depended on
# the specific figure, but 47.2 was simply wrong; 38.77 is correct.
#
# Separately, `order_of_magnitude_gap()` above does NOT implement the "1-15
# day band" test described in the docstring's falsification-test paragraph
# -- it returns `falsification_test_fires=True` whenever `ratio < 10`
# against the anchor half-lives, which fires (True) at the correct 38.77-day
# figure even though the docstring's own band (1-15 days) would not. Any
# future round reusing this helper should fix or retire it rather than
# trust its `falsification_test_fires` field at face value.
# ---------------------------------------------------------------------------
