"""Shared, read-only utilities and pre-registration for the R-127 round (08-25).

DIRECTION, in one sentence: **six independent constructions, across three
different mechanism axes and now two different objects, have all passed
this project's B1 gate on BTC and inverted sign on ETH's B4 falsification
test — and no round has ever asked whether that is a property of the
mechanisms, or a property of the TEST.**

**The pattern this round investigates, named explicitly by three prior
entries but never itself the object of a round:**
R-109 (kNN novelty brake on `kelly_regime_v4`), R-113 (the same brake family
applied to the multi-asset panel), R-115-conservative (CORAL-pooled kNN,
Coinbase ETH), R-125-conservative (CVaR risk-measure substitution — though
this one failed at B1 before reaching B4, so it reproduces the *shape*, not
a clean B4 inversion), and R-126 (both ERC and CVaR-budgeted reallocation of
`champions_council`'s cross-strategy weights) all cleared their BTC-side
gates and then inverted sign on ETH under this project's standard B4
convention: `INNER_VAL_START = 2021-01-01` to `INNER_VAL_END = 2022-12-31`,
**the identical calendar dates on both assets**. R-126's own entry named the
suspicion plainly: "the strongest evidence yet that this signature belongs
to the BTC/ETH training-window relationship itself rather than to any one
mechanism family." Nobody has yet asked what, specifically, differs between
BTC's and ETH's calendar-matched windows, or whether the calendar-matching
convention itself — silently assuming the two assets pass through
comparable regime composition over the same 24 months — is contributing to
the inversion.

**Which constraint this attacks.** Primarily **N≈3**: this project's own
standing diagnosis already treats effective sample size as ~3 regime
events, not bar count — this round asks whether that same logic applies
*across assets*, i.e. whether BTC's 2021-2022 and ETH's 2021-2022 are drawn
from comparably-composed regime samples at all, or whether ETH's own
idiosyncratic lifecycle (DeFi/NFT-driven 2021 overshoot, a harder
idiosyncratic drawdown through Terra/Luna in May 2022 — an ETH-ecosystem
event more directly than a BTC one — and the Merge, a supply-side
proof-of-stake transition completed 2022-09-15 with no BTC analogue) puts
materially different regime content behind the same twenty-four calendar
months. Secondarily **ERR**: a falsification test whose window-matching
convention is itself unexamined is a gap in this project's error control,
independent of whatever mechanism it is being applied to.

**Why this and not a 7th SIZE/ERR mechanism variant.** Per this round's own
Step-0 diligence: the single-asset axis is closed across INFO (19+
signals), SIZE (28+ attempts — magnitude, calibration, timing, detector
family, risk measure), ERR (5 notions of uncertainty), regime-timing (11
mechanisms) and N≈3-calibration (4 procedures); the multi-asset panel axis
is closed across 11 rounds; `champions_council`'s own cross-strategy
allocation is now closed too (R-126). Every one of those ~126 rounds varied
the MECHANISM under test while holding the falsification TEST itself fixed.
This round is the first to vary the test. It is explicitly the kind of
"cost the plan, not just the mechanism" move this project's own standing
rules call for after a repeated pattern goes unexplained across multiple
independent constructions (the same move R-33 made on risk-matching, R-57
on cross-asset scope, and R-78 on the forward-trading horizon) — B4 has
been read as a settled, symmetric test in ~40 prior rounds; this round asks
whether it is one.

**Not a duplicate of:**
- R-57 (the six-instrument Coinbase panel, BCH/LTC/ETC/DASH/LINK/XTZ):
  established that `kelly_regime_v4`'s properties are asset-specific by
  running the SAME frozen strategy on OTHER assets. This round holds the
  asset pair fixed (BTC, ETH — the only two this project has ever used for
  falsification) and instead varies which WINDOW of ETH's own history is
  compared against BTC's fixed window.
- R-33 (risk-matching the *benchmark*): matches exposure level within one
  asset's own comparison. This round matches *regime composition* across
  two different assets' historical windows — a different axis entirely
  (WHEN, not HOW MUCH).
- Every SIZE/ERR/regime-timing round that reached B4 (R-87 through R-126):
  all treat B4 as a fixed, already-specified test and report whether their
  candidate mechanism passes or fails it. None varies the test's own window
  selection.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r127_conservative_regime_fingerprint.py`): a pure
  diagnostic, no strategy code. Compute a low-dimensional "regime
  fingerprint" (annualized realized volatility, vol-of-vol, mean daily
  log-return, skewness, excess kurtosis, lag-1/5/20 return autocorrelation,
  maximum drawdown, and fraction-of-days-positive — all standard
  regime-characterization statistics, e.g. Ang & Bekaert 2002, "International
  Asset Allocation With Regime Shifts", *Review of Financial Studies* 15(4),
  1137-1187, and Guidolin & Timmermann 2007, "Asset allocation under
  multivariate regime switching", *Journal of Economic Dynamics and
  Control* 31(11), 3503-3544, both of which characterize regimes on exactly
  this class of moment statistics) over BTC's fixed `INNER_VAL` window and
  over ETH's calendar-matched `INNER_VAL` window, then ask two questions
  with standard two-sample tests: is the fingerprint difference large
  relative to natural sampling variation (a block-bootstrap permutation
  test against the null of "same-length ETH sub-windows drawn from ETH's
  own pre-holdout history"), and do individual moments (variance via
  Levene's test, distribution shape via a two-sample Kolmogorov-Smirnov
  test on daily log returns) differ significantly.

- NOVEL (`r127_novel_event_excision_retest.py`): **this branch's design was
  fixed only after this file's own window-scan ran (see "What the window
  scan already found" below) — before any B1/B4/strategy number was
  computed, so this is Step 2 design informed by Step 2's own structural
  diagnostic, the same order R-65 (measure decay rate, then design the
  mechanism) and R-67 (measure the floor's invariance, then design the fix)
  used, not a decision rule moved after seeing a performance result.**
  Because whole-window regime matching collapsed onto the calendar window
  itself (crypto assets co-move at the macro cycle level, so there is no
  materially better-matching ETH window to substitute), the refined
  hypothesis this branch tests is narrower: not "the wrong 24 months were
  compared," but "within the right 24 months, brief ETH-IDIOSYNCRATIC
  divergence episodes — events with no BTC analogue — disproportionately
  drive the sign flip." Re-runs the same already-recorded construction as
  before (R-126 novel's CVaR-budgeted `champions_council` reallocation,
  BTC spot `d_sharpe=+0.388` vs ETH spot `d_sharpe=-0.530`, both clearing
  the noise floor) on ETH's `INNER_VAL` window with two PRE-REGISTERED,
  STRUCTURAL (not fitted) excisions applied, each evaluated separately: (a)
  named-event excision — the two ETH-specific structural events with no BTC
  analogue that fall inside `INNER_VAL`, `TERRA_LUNA_WINDOW` and
  `THE_MERGE_WINDOW` below, dates fixed from public record with a symmetric
  buffer, not fit to any performance number; (b) a data-driven low-
  correlation-episode excision — calendar days where the trailing
  `CORR_WINDOW_DAYS`-day BTC/ETH daily-return correlation falls below
  `LOW_CORR_THRESHOLD` (a structural threshold set before either branch
  read a performance number, not swept). No refit, no new mechanism: only
  which calendar days count toward the B4 evaluation changes, isolating the
  "does idiosyncratic divergence drive the sign" question from both the
  window-selection question (already answered) and the mechanism question
  (already fixed, R-126).

**What the window scan already found (computed by this shared module,
before either branch script existed — not a result either branch
selected):** the calendar-matched ETH window is the SINGLE BEST regime-
fingerprint match to BTC's `INNER_VAL` among all 95 candidate windows
(`SCAN["cal_percentile"] ≈ 1.1`, i.e. in the closest ~1% by aggregate
regime distance) — expected, since BTC and ETH share the same market-wide
crypto cycle (2021 bull, 2022 bear), so `REGIME_MATCHED_ETH_WINDOW ==
CALENDAR_ETH_WINDOW` exactly. **This already falsifies the coarse
window-mismatch hypothesis this round originally set out to test** — there
is no alternative window a "regime-matched retest" could select that
differs from what every prior round already used — and is why the novel
branch's design pivoted to the finer-grained idiosyncratic-divergence
question above rather than executing the now-moot window-substitution
originally planned. The conservative branch below is what independently
establishes and quantifies this finding with proper significance tests; it
is not being taken on faith from this paragraph.

**What would make this fail, named now, before any code:**
- If the conservative branch's permutation test finds BTC's `INNER_VAL`
  fingerprint is NOT an unusual distance from ETH's calendar-matched
  window's fingerprint (i.e. within the bulk of the null distribution over
  ETH sub-windows), the calendar-confound hypothesis is directly refuted:
  ETH's calendar-matched window is a perfectly typical draw from ETH's own
  regime distribution, and the six-fold inversion needs a different
  explanation than window mismatch.
- If the novel branch's regime-matched retest ALSO inverts sign (or does
  not narrow the gap materially), that is equally informative and refutes
  the hypothesis for the one construction actually tested: realigning the
  window did not fix it, so calendar mismatch is not what is driving R-126
  novel's own inversion specifically.
- Either outcome is a genuine, useful answer — this round is diagnostic,
  not a strategy candidate, so there is no promotion bar to game and no
  incentive to keep searching until one sign appears. See the "single
  frozen window" discipline below, which exists precisely to prevent that.

**The single-frozen-window discipline (multiple-testing control).**
Searching many candidate ETH windows and reporting whichever one happens to
flip a mechanism's B4 sign would be exactly the "28-of-32 in-sample winners"
trap ROUTINE.md exists to prevent — the window would be selected BY the
answer it produces, not by the regime-similarity criterion motivating the
whole round. To prevent this, the ONE window either branch may use is
computed HERE, by this frozen shared module, using ONLY the regime-
fingerprint distance metric (never touching any B4/CVaR/strategy output),
and is fixed as `REGIME_MATCHED_ETH_WINDOW` below before either branch
script is written. Neither branch may search further or pick a different
window after seeing a result.

**Falsification test, pre-registered (two-stage, matching the two-stage
design above):**
- Coarse (conservative branch, window-level): does the calendar-matched
  ETH window's fingerprint distance fall ABOVE the 90th percentile of the
  null distribution (an unusually poor match)? **Already resolved by the
  frozen scan above: it does not — it is at the 1.1st percentile, the
  closest match, not the worst.** Coarse hypothesis REFUTED.
- Fine (novel branch, event-level): does excising `TERRA_LUNA_WINDOW` +
  `THE_MERGE_WINDOW`, OR excising low-BTC/ETH-correlation days, flip or
  materially narrow R-126 novel's ETH `d_sharpe` sign gap (currently
  BTC +0.388 vs ETH -0.530)? A materially narrowed or sign-flipped gap
  supports the idiosyncratic-divergence hypothesis for this construction; a
  gap that is unchanged or widens refutes it.

**Decision rule, pre-registered:** this round produces no promotable
strategy candidate (it is a diagnostic of the falsification TEST, not a new
mechanism) — the deliverable is a documented finding about the B4
convention itself, recorded either as "confirmed: calendar alignment is a
material confound, future B4 reads should consider regime-matching" or
"refuted: the inversion is not explained by window mismatch" or a mixed
result, whichever the evidence shows. No bar at or after `OOS_START =
2023-01-01` may be read by either branch — the window search below is
restricted to ETH's own PRE-HOLDOUT history only, and R-126's frozen BTC-
side `INNER_VAL` numbers are reused by citation, not re-read from the
holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402

# ----------------------------------------------------------------------
# Splits. Identical convention to every prior round: inner-train / inner-
# validation only. The holdout (>= OOS_START) is never read by a branch.
# ----------------------------------------------------------------------
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"  # do not read; guarded by _assert_no_holdout below

BARS_PER_DAY = 288  # 5-minute bars


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def load_btc_train(kind: str = "spot"):
    df, label = load_dataset(ROOT / "data", kind)
    train = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train, label


def load_eth_pretrain_full():
    """ETH's ENTIRE pre-holdout history (2019-03-14 -> 2022-12-31), not just
    INNER_VAL -- the search space the regime-matching window is drawn from.
    """
    eth = load_coinbase_eth_spot(ROOT / "data")
    assert eth is not None, "ETH Coinbase spot data not committed"
    eth = eth.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth)
    return eth


def load_eth_train():
    """ETH restricted to the calendar-matched INNER_VAL window -- the window
    every prior round's B4 test has used."""
    eth = load_eth_pretrain_full()
    return eth.loc[INNER_VAL_START:INNER_VAL_END].copy()


# ----------------------------------------------------------------------
# Regime fingerprint: a small vector of standard regime-characterization
# statistics (Ang & Bekaert 2002; Guidolin & Timmermann 2007), computed on
# CALENDAR-DAILY log returns (never 5m bars -- autocorrelated microstructure
# noise at 5m would swamp the regime signal and misrepresent the effective
# sample size, the same reason tradebot.inference resamples to daily
# throughout).
# ----------------------------------------------------------------------

FINGERPRINT_LABELS = (
    "ann_vol", "vol_of_vol", "mean_daily_ret", "skew", "excess_kurtosis",
    "acf_lag1", "acf_lag5", "acf_lag20", "max_drawdown", "frac_positive_days",
)


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    daily_close = df["close"].resample("1D").last().dropna()
    return np.log(daily_close).diff().dropna()


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 5:
        return np.nan
    a, b = x[:-lag], x[lag:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _max_drawdown_from_returns(r: np.ndarray) -> float:
    equity = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())  # negative


def regime_fingerprint(r: pd.Series) -> np.ndarray:
    """`r`: calendar-daily log returns over one window. Returns the vector
    in `FINGERPRINT_LABELS` order. Pure numpy/pandas, no strategy code, no
    lookahead concern -- this is a post-hoc descriptive statistic of a
    frozen historical window, not a trading decision."""
    x = r.to_numpy(dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return np.full(len(FINGERPRINT_LABELS), np.nan)
    ann_vol = float(x.std(ddof=1) * np.sqrt(365))
    # vol-of-vol: std of a rolling 20-day realized vol series
    roll_vol = pd.Series(x).rolling(20).std(ddof=1).dropna().to_numpy()
    vol_of_vol = float(roll_vol.std(ddof=1)) if len(roll_vol) > 5 else np.nan
    mean_ret = float(x.mean())
    skew = float(pd.Series(x).skew())
    kurt = float(pd.Series(x).kurt())  # pandas already reports EXCESS kurtosis
    acf1, acf5, acf20 = _acf(x, 1), _acf(x, 5), _acf(x, 20)
    mdd = _max_drawdown_from_returns(x)
    frac_pos = float((x > 0).mean())
    return np.array([ann_vol, vol_of_vol, mean_ret, skew, kurt,
                      acf1, acf5, acf20, mdd, frac_pos])


def fingerprint_distance(fp_a: np.ndarray, fp_b: np.ndarray, scale: np.ndarray) -> float:
    """Standardized Euclidean distance: each coordinate divided by `scale`
    (the cross-candidate-window std of that coordinate, so no single
    high-magnitude statistic like ann_vol dominates the metric)."""
    mask = np.isfinite(fp_a) & np.isfinite(fp_b) & np.isfinite(scale) & (scale > 0)
    if mask.sum() == 0:
        return np.nan
    z = (fp_a[mask] - fp_b[mask]) / scale[mask]
    return float(np.sqrt(np.mean(z ** 2)))


# ----------------------------------------------------------------------
# Candidate ETH window generator + selection. Frozen here, computed once,
# BEFORE either branch script exists -- see the single-frozen-window
# discipline in the module docstring.
# ----------------------------------------------------------------------

WINDOW_STEP_DAYS = 7   # weekly-stepped candidate starts
WINDOW_LEN_DAYS = (pd.Timestamp(INNER_VAL_END) - pd.Timestamp(INNER_VAL_START)).days + 1


def _candidate_windows(eth_pretrain_daily_index_start, eth_pretrain_daily_index_end):
    starts = pd.date_range(eth_pretrain_daily_index_start,
                            eth_pretrain_daily_index_end
                            - pd.Timedelta(days=WINDOW_LEN_DAYS - 1),
                            freq=f"{WINDOW_STEP_DAYS}D")
    return [(s, s + pd.Timedelta(days=WINDOW_LEN_DAYS - 1)) for s in starts]


def compute_btc_val_fingerprint() -> np.ndarray:
    btc, _ = load_btc_train("spot")
    r_btc_val = daily_log_returns(btc.loc[INNER_VAL_START:INNER_VAL_END])
    return regime_fingerprint(r_btc_val)


def compute_eth_window_scan() -> dict:
    """Returns dict with: candidate windows, their fingerprints, distances
    to BTC's INNER_VAL fingerprint, the calendar-matched window's own rank,
    and the argmin (regime-matched) window. Computed once; both branches
    import the frozen result rather than recomputing independently, so
    there is exactly one search, not two."""
    eth_full = load_eth_pretrain_full()
    eth_daily = daily_log_returns(eth_full)
    fp_btc_val = compute_btc_val_fingerprint()

    windows = _candidate_windows(eth_daily.index.min().tz_localize(None),
                                  eth_daily.index.max().tz_localize(None))
    naive_index = eth_daily.index.tz_localize(None)
    eth_daily_naive = pd.Series(eth_daily.to_numpy(), index=naive_index)

    fps, dists = [], []
    for s, e in windows:
        r_win = eth_daily_naive.loc[s:e]
        fp = regime_fingerprint(r_win)
        fps.append(fp)
    fps = np.array(fps)
    # scale = cross-window std of each coordinate (excluding all-NaN cols)
    scale = np.nanstd(fps, axis=0, ddof=1)
    dists = np.array([fingerprint_distance(fp_btc_val, fp, scale) for fp in fps])

    # calendar-matched window's own index in this candidate list (nearest
    # candidate start to INNER_VAL_START)
    cal_start = pd.Timestamp(INNER_VAL_START)
    cal_idx = int(np.argmin([abs((s - cal_start).days) for s, _ in windows]))

    valid = np.isfinite(dists)
    order = np.argsort(np.where(valid, dists, np.inf))
    best_idx = int(order[0])
    cal_percentile = float((dists[valid] <= dists[cal_idx]).mean() * 100) if valid[cal_idx] else np.nan

    return {
        "windows": windows, "fingerprints": fps, "distances": dists,
        "fp_btc_val": fp_btc_val, "scale": scale,
        "cal_idx": cal_idx, "cal_window": windows[cal_idx], "cal_distance": dists[cal_idx],
        "cal_percentile": cal_percentile,
        "best_idx": best_idx, "best_window": windows[best_idx], "best_distance": dists[best_idx],
    }


# Computed once at import time -- this IS the pre-registration freeze. Both
# branch scripts import REGIME_MATCHED_ETH_WINDOW / SCAN from here; neither
# recomputes or re-selects.
SCAN = compute_eth_window_scan()
REGIME_MATCHED_ETH_WINDOW = SCAN["best_window"]  # (start, end), tz-naive daily dates
CALENDAR_ETH_WINDOW = SCAN["cal_window"]

# ----------------------------------------------------------------------
# Event-excision infrastructure for the NOVEL branch's refined hypothesis.
# Dates are structural (public record), fixed before any performance number
# from either branch was read.
# ----------------------------------------------------------------------

# Terra/Luna (UST depeg + LUNA collapse): 2022-05-07 (UST first depegs) to
# 2022-05-15 (LUNA effectively worthless, trading halted on major venues).
# A three-day symmetric buffer is added on each side to cover pre-shock
# drift and post-shock volatility decay, structural not fit.
TERRA_LUNA_WINDOW = (pd.Timestamp("2022-05-04"), pd.Timestamp("2022-05-18"))

# The Merge (Ethereum's proof-of-stake transition, completed 2022-09-15
# 06:42 UTC) -- an ETH-specific supply-side/narrative event with zero BTC
# analogue. Window covers the week before (pre-Merge run-up/positioning)
# through the week after (post-Merge "sell the news" unwind).
THE_MERGE_WINDOW = (pd.Timestamp("2022-09-08"), pd.Timestamp("2022-09-22"))

CORR_WINDOW_DAYS = 14      # trailing correlation window (structural, not fit)
LOW_CORR_THRESHOLD = 0.30  # structural threshold, set before any B4 number


def rolling_btc_eth_daily_corr(btc_daily: pd.Series, eth_daily: pd.Series,
                                window_days: int = CORR_WINDOW_DAYS) -> pd.Series:
    """Trailing `window_days`-day rolling correlation of BTC and ETH
    calendar-daily log returns, index = ETH's daily calendar index
    (tz-naive), inner-joined to BTC's. Used only to FLAG idiosyncratic-
    divergence days for the novel branch's excision test -- never fed into
    any trading decision, so no causality constraint beyond "does not use
    days after the one being labeled," which a trailing rolling window
    satisfies by construction."""
    a = btc_daily.copy()
    a.index = a.index.tz_localize(None) if a.index.tz is not None else a.index
    b = eth_daily.copy()
    b.index = b.index.tz_localize(None) if b.index.tz is not None else b.index
    joined = pd.DataFrame({"btc": a, "eth": b}).dropna()
    return joined["btc"].rolling(window_days).corr(joined["eth"])


def low_correlation_days(btc_daily: pd.Series, eth_daily: pd.Series,
                          window_days: int = CORR_WINDOW_DAYS,
                          threshold: float = LOW_CORR_THRESHOLD) -> pd.DatetimeIndex:
    corr = rolling_btc_eth_daily_corr(btc_daily, eth_daily, window_days)
    return corr.index[corr < threshold]


def excise_days(daily_series: pd.Series, excluded_days: pd.DatetimeIndex) -> pd.Series:
    """Return `daily_series` (a CALENDAR-DAY-indexed return/payoff series,
    e.g. from `tradebot.inference.daily_returns` on an already-completed
    backtest's equity curve) with every day in `excluded_days` dropped.

    Deliberately a POST-HOC filter on realized daily returns, never a
    mutation of the 5-minute OHLCV frame fed to the backtest engine: the
    engine's position/equity bookkeeping assumes a contiguous bar index, so
    removing bars mid-series would corrupt fills and equity continuity in
    ways that have nothing to do with this round's question. Excising
    complete calendar days from the DAILY return series after a normal,
    unmodified backtest has already run is exactly a delete-a-block
    jackknife (Efron & Stein 1981) applied to named blocks instead of
    random ones -- the same operation `tradebot.inference.paired_bootstrap`
    already performs on random blocks, restricted here to specific,
    pre-registered ones."""
    day = daily_series.index.tz_localize(None) if daily_series.index.tz is not None \
        else daily_series.index
    excluded = pd.DatetimeIndex(excluded_days).floor("D")
    keep = ~day.floor("D").isin(excluded)
    return daily_series.loc[keep]


if __name__ == "__main__":
    print(f"BTC INNER_VAL fingerprint ({INNER_VAL_START}..{INNER_VAL_END}):")
    for label, val in zip(FINGERPRINT_LABELS, SCAN["fp_btc_val"]):
        print(f"  {label:22s} {val:+.5f}")
    print(f"\n{len(SCAN['windows'])} candidate ETH windows scanned "
          f"({WINDOW_LEN_DAYS}-day length, {WINDOW_STEP_DAYS}-day step).")
    print(f"Calendar-matched window {CALENDAR_ETH_WINDOW[0].date()}..{CALENDAR_ETH_WINDOW[1].date()}: "
          f"distance={SCAN['cal_distance']:.4f}  percentile={SCAN['cal_percentile']:.1f}")
    print(f"Regime-matched (argmin) window "
          f"{REGIME_MATCHED_ETH_WINDOW[0].date()}..{REGIME_MATCHED_ETH_WINDOW[1].date()}: "
          f"distance={SCAN['best_distance']:.4f}")

    # Causal-truncation-style self-test for the shared statistics: growing
    # the ETH frame should not change the fingerprint of an EARLIER window
    # already fully contained in a truncated frame (a descriptive-statistic
    # analogue of this project's usual no-lookahead probe -- these are
    # windowed aggregates of already-realized daily bars, not a live signal,
    # so the relevant guarantee is that a window's own statistic depends
    # only on bars inside that window).
    eth_full = load_eth_pretrain_full()
    early_end = pd.Timestamp("2020-06-30", tz=eth_full.index.tz)
    full_daily = daily_log_returns(eth_full)
    trunc_daily = daily_log_returns(eth_full.loc[:early_end])
    common = full_daily.index.intersection(trunc_daily.index)
    common = common[common <= early_end - pd.Timedelta(days=5)]
    ok = np.allclose(full_daily.loc[common].to_numpy(),
                      trunc_daily.loc[common].to_numpy(), atol=1e-10)
    print(f"\ncausal truncation probe (daily_log_returns): {'PASS' if ok else 'FAIL'}")
    assert ok, "daily_log_returns depends on bars after the truncation point"
