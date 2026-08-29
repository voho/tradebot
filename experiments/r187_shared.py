"""Shared, read-only utilities and frozen pre-registration for the R-187 round (08-29).

DIRECTION, in one sentence: `kelly_regime_v4` sizes off ONE instrument's own
realized volatility (`v4_scale`, single-asset), but the crashes that hurt it
most -- Terra/Luna, FTX, the 2021-11 top -- are CONTAGION events, defined by
literature as episodes where cross-asset co-movement spikes ahead of or
alongside the price move itself; this round asks whether a second,
independent multiplier derived from the already-committed 8-instrument
Coinbase panel's own CONNECTEDNESS (not any single asset's volatility) catches
what single-asset vol-targeting reacts to only after the fact.

**Which constraint this attacks:** INFO (a data channel -- the cross-sectional
relationship among the 8 already-committed instruments -- that no prior round
has fed into a SIZE-shaped construction) principally, consumed as a smooth
risk-budgeting multiplier on `kelly_regime_v4`'s own `scale` factor, not as a
directional predictor -- the one construction shape this project's own
25-strategy record (README "Pattern across A") shows survives. Secondarily
COST, since any exposure cut this mechanism produces lands specifically in
high-connectedness periods, which R-63 measured as the panel's own costliest
regime to trade through (mean pairwise correlation 0.634 full-period).

**Literature grounding (fetched via WebSearch before either branch is
dispatched; full citation list from the research pass this round opened
with):**

- Diebold, F.X. & Yilmaz, K. (2012), "Better to Give than to Receive:
  Predictive Directional Measurement of Volatility Spillovers", Int'l J.
  Forecasting 28(1), 57-66; and (2014), "On the Network Topology of Variance
  Decompositions: Measuring the Connectedness of Financial Firms", J.
  Econometrics 182(1), 119-134. Defines the generalized-variance-decomposition
  Total Spillover Index (TSI) this round's novel branch computes: from a
  rolling VAR's forecast-error variance decomposition (Pesaran & Shin 1998
  generalized form, order-invariant), TSI = (sum of cross-variable variance
  shares) / (sum of all variance shares) x 100, in [0, 100].
- Pollet, J.M. & Wilson, M. (2010), "Average Correlation and Stock Market
  Returns", J. Financial Economics 96(3), 364-380. Average pairwise
  correlation as a systemic-risk state variable carrying information distinct
  from any single asset's own variance -- the theoretical warrant for the
  conservative branch's simpler statistic.
- Longin, F. & Solnik, B. (1995, J. International Money and Finance 14(1);
  2001, J. Finance 56(2)). Correlation among asset returns rises specifically
  in high-volatility / bear regimes ("correlation breakdown"), not
  uniformly -- the asymmetry this round's discount is shaped to exploit
  rather than a symmetric always-on brake.
- Applied to crypto specifically: Ji, Q., Bouri, E., Lau, C.K.M. & Roubaud,
  D. (2019), "Dynamic connectedness and integration in cryptocurrency
  markets", International Review of Financial Analysis 63, 257-272 (the
  Diebold-Yilmaz framework applied to a cryptocurrency panel for the first
  time); Corbet, S., Lucey, B., Urquhart, A. & Yarovaya, L. (2019), "Cryptocurrencies
  as a financial asset: A systematic analysis", Int'l Review of Financial
  Analysis 62, 182-199 (survey establishing crypto cross-asset connectedness
  as an active, distinct measurement literature from single-asset volatility).

**Not a duplicate of** (checked against the full ruled-out list and every
"correlation"/"connectedness"/"spillover"/"Diebold" hit in docs/LEDGER.md
before this module was written):
R-63/65/67/68 (the panel's mean pairwise correlation there is a ONE-TIME,
full-period breadth measurement used to price a CROSS-SECTIONAL TREND
signal's affordability; here it is a ROLLING, causal, TIME-VARYING state
variable feeding a SIZE-axis multiplier on the single-asset `kelly_regime_v4`
book, never a cross-sectional allocation). B-21 (VIX/DXY macro-veto -- a
TradFi spillover source, external to the crypto panel, and already-failed on
lead-time; this round's source is the crypto panel itself, not TradFi, and is
checked for the identical lead-time failure mode at Step 0 below before any
further build cost is spent). R-53/R-54/R-73/R-74/R-84/R-85 and the four
regime-timing mechanisms (HMM/BOCPD/Kalman-LLT/CSD, R-82/R-83/R-83/R-85) --
all condition on a SINGLE series' own history (BTC's price, volume, or an
external univariate feed); this round's state variable is intrinsically
MULTI-INSTRUMENT (undefined for a single series), a structurally distinct
input class none of those seven mechanisms could express. R-70 (Ledoit-Wolf
SHRINKAGE of a covariance matrix for PORTFOLIO WEIGHT estimation on the
multi-asset construction, R-63's lineage) -- a different consumer (portfolio
weights vs. a scalar brake on one already-fixed single-asset book) and a
different statistic (shrunk covariance for optimization vs. a spillover/
correlation LEVEL as a state signal). CVaR/CDaR/GZ/EVT/Wasserstein-DRO
drawdown-based caps (R-93, R-97, R-98, R-125, and others) all condition on
`kelly_regime_v4`'s OWN realized P&L; this round never reads the strategy's
own equity curve as an input.

**Simulable here:** yes, with ZERO new data -- the 8-instrument Coinbase panel
(data/{bch,ltc,etc,dash,link,xtz}usd_coinbase_spot_5m.csv.gz, ETH via
data/ethusd_coinbase_spot_5m.csv.gz, BTC via the canonical Bitstamp series)
already committed and used by R-57/R-63/R-65/R-67/R-68. Both branches compute
the connectedness statistic at DAILY frequency from the panel's own 5m closes
(no VAR/statsmodels dependency added to pyproject.toml -- the novel branch's
VAR(1)+GFEVD is implemented with plain OLS via `numpy.linalg.lstsq`, in this
module, so both branches import an identical, already-tested primitive rather
than each re-deriving it) and merge it onto the 5m strategy frame with the
SAME causal shift-by-1-day-then-ffill convention every other daily external
feed in `tradebot/data.py` uses (`align_connectedness_causal` below) -- a bar
may only see the connectedness reading for the most recent day that closed
strictly before that bar's own day.

**A coverage constraint, disclosed before any number is computed:** the panel
starts 2020-01-01 (all six alts), so of `INNER_TRAIN` (2017-01-01 ->
2020-12-31), only the final ~12 months carry a connectedness reading at all;
before that, `align_connectedness_causal` returns NaN and both branches MUST
define NaN -> multiplier 1.0 (unmodified v4, never a silent zero-fill that
would look like a stress brake firing). This is disclosed rather than
worked around because it is itself informative: the two 2018 stress episodes
below are FORCED-NaN for this mechanism by construction, identical to R-100's
FORCED_FAIL_EPISODES treatment of Binance funding's own 2020-01-01 start.

**What would make it fail, named now, in likelihood order:**
(a) NO LEAD -- like B-21 (VIX/DXY) and all four regime-timing mechanisms
    (R-82/83/85), panel connectedness rises WITH or AFTER the price move
    rather than before it. Most likely outcome, per this project's own
    5-for-5 record on "does a state-variable alarm lead a BTC stress
    episode" questions; this is the Step-0 stop below.
(b) TOO RARE TO MATTER -- like R-85's CSD variance branch (1/6, fires once
    in six years), the panel only decorrelates during the exact three
    episodes its own citations describe, contributing an economically
    trivial number of brake-days to two-plus years of inner-validation.
(c) EXPOSURE-ARTIFACT COLLAPSE -- R^2 against a flat rescale of v4's own
    target >= 0.95 (the standard test that killed R-73/R-74/R-79): the
    "connectedness brake" turns out to fire (near-)unconditionally often
    enough that it is indistinguishable from a constant de-lever.
(d) BTC-ONLY -- the discount helps on the BTC book (source panel dominated
    by BTC/ETH's own high mutual correlation) but the identical construction,
    re-run with ETH as the base asset against the SAME panel-wide
    connectedness series, does not replicate -- this project's most common
    single failure signature (R-126, R-149, R-150, R-168, R-185).
(e) COST-NEGATIVE -- the discount adds toggling around its own threshold
    (extra trades) whose fee cost exceeds the drawdown or Sharpe benefit,
    especially at the 0.40% tier.

**Frozen splits** (identical to every prior round in this lineage):
inner-train 2017-01-01 -> 2020-12-31 (fit/debug + the Step-0 gate; connectedness
only defined for the final ~12 months of this window, see above -- NOT a
promotion-relevant number); inner-validation 2021-01-01 -> 2022-12-31 (all
selection, both branches -- fully covered by the panel, and contains the
2021-11 top / Terra / FTX episodes this mechanism's own citations target);
holdout 2023-01-01 -> untouched by both branches.

**Step-0 gate** (computed ONCE, before either branch is implemented in full
-- see `step0_gate()` below): using the conservative branch's own (cheaper)
rolling mean pairwise correlation as the shared Step-0 statistic, test
whether it rises ahead of the STRESS_EPISODES below that the panel can
actually see (2020-03 COVID, thin baseline; 2021-11 top; 2022-05 Terra;
2022-11 FTX -- the two 2018 episodes are FORCED-NaN, disclosed, and excluded
from the gate's own denominator exactly as R-100 excluded them from its
funding gate). Computed over the full pre-holdout history (through
INNER_VAL_END = 2022-12-31, never OOS_START or later) rather than inner-train
alone: three of the four usable episodes (2021-11, Terra, FTX) fall in
2021-2022, and this is a diagnostic "does a leading relationship exist at
all" check on named historical dates -- the same kind of check R-53/R-84/R-85
ran over their own full six-episode table -- not a threshold SELECTION that
would need to stay confined to inner-train to protect inner-validation for
later performance comparison. For each usable episode, measure the lead/lag
(in days) between the episode date and the first day the trailing
correlation crosses into the top quartile of its own trailing-to-date
distribution, and compare the median lead against a block-bootstrap null of
arbitrary time shifts (same construction R-84 used). **Proceed past Step 0
only if the median lead is positive (rises BEFORE, not after) on a majority
of usable episodes AND is not explained by the null at the 50% level** --
deliberately loose (this project's own lead-time gates have never cleared a
strict bar; the point is to catch outright inertness before paying for two
full branch implementations, not to pre-decide the promotion verdict).

**Kill switches (both branches, checked on every reported comparison via
`compare_scaled()` below):**
(i) connectedness-NaN bars (pre-2020-01-01 panel coverage) produce a target
    path IDENTICAL to unmodified `kelly_regime_v4` (multiplier == 1.0 there,
    exactly, not approximately -- checked by assertion in each branch's own
    self-test);
(ii) R^2 of the candidate's target path against v4's own target < 0.98 on
     inner-validation (else this is a relabeling of v4 itself);
(iii) mean |exposure| ratio and realized-volatility ratio against v4 are BOTH
      reported for every cell (R-33's rule: match risk before comparing
      anything) -- since this mechanism is a one-sided discount (multiplier
      in (0, 1], never > 1), `exposure_ratio` < 1 is expected BY CONSTRUCTION
      and is not itself a finding; the promotion test below is read against a
      FLAT RESCALE of v4's own target to the same mean exposure (matching
      R-62's own methodology for judging a scale-axis factor), not against
      v4 unscaled.

**Promotion threshold (inner-validation, both markets, both branches, against
the flat-rescaled-to-equal-exposure v4 control):** all of (a) paired-bootstrap
Delta-Sharpe >= +0.20 (R-20's floor) with 95% CI excluding zero, OR a >=5pp
max-drawdown reduction at matched mean exposure and matched realized
volatility; (b) the branch's own Step-0-consistent lead-time property holds
on inner-validation's three usable episodes, not just inner-train's one
(COVID); (c) the frozen threshold/window is a plateau (report neighbours);
(d) no sign reversal at the 0.40% fee tier; (e) replicates on ETH as the base
asset against the same panel connectedness series.

Both branches import this module and MUST NOT modify it or commit any
changes to it -- it is the frozen reference, exactly as `r102_shared.py` and
`r186_shared.py` have served prior rounds in this lineage.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    V4_ANCHOR_SPAN_DAYS,
    V4_BAND,
    V4_DEADBAND,
    V4_HORIZONS,
    V4_MAX_LEVERAGE,
    V4_TARGET_VOL,
    V4_VOL_SPAN,
    SliceResult,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)
from tradebot.data import load_coinbase_spot, load_dataset  # noqa: E402

DATA_DIR = ROOT / "data"

UNIVERSE_6 = ("BCH", "LTC", "ETC", "DASH", "LINK", "XTZ")
UNIVERSE_8 = ("BTC", "ETH") + UNIVERSE_6
PANEL_COVERAGE_START = pd.Timestamp("2020-01-01", tz="UTC")

# IDENTICAL to R-82/83/84/85/86/88/96/98/99/R-100's own table -- copied
# verbatim, not re-derived.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]
# Panel coverage starts 2020-01-01: the two 2018 episodes cannot be seen by
# this mechanism at all, by construction (same treatment R-100 gave Binance
# funding's own 2020-01-01 start).
FORCED_NAN_EPISODES = {
    "2018 bear onset (post-Dec-2017 top)",
    "2018 bear bottom / capitulation",
}
# COVID's own 60-day pre-onset baseline window opens ten weeks after panel
# coverage starts -- thin, not absent.
THIN_BASELINE_EPISODES = {"2020-03 COVID crash"}

# Conservative branch's rolling window (days) and Step-0 gate window.
ROLL_DAYS = 60
MIN_PERIODS = 30
LEAD_QUANTILE = 0.75  # "connectedness regime" = top quartile of trailing history


# ------------------------------------------------------------------ panel data


def load_universe(tickers: tuple[str, ...] = UNIVERSE_8) -> dict[str, pd.DataFrame]:
    """Raw 5m OHLCV per ticker, identical convention to r63_shared.load_universe:
    BTC is the Bitstamp canonical series, every other ticker is the committed
    Coinbase USD spot file."""
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if t == "BTC":
            df, label = load_dataset(DATA_DIR, "spot")
            if "SYNTH" in label.upper():
                raise RuntimeError(f"refusing to run on synthetic BTC data ({label})")
        else:
            df = load_coinbase_spot(DATA_DIR, t)
        if df is None or df.empty:
            raise RuntimeError(f"no data for {t}")
        frames[t] = df
    return frames


def panel_daily_log_returns(frames: dict[str, pd.DataFrame] | None = None,
                             tickers: tuple[str, ...] = UNIVERSE_8) -> pd.DataFrame:
    """Daily log returns per ticker from each series' own 5m closes,
    resampled to the LAST close of each UTC day. Columns before a ticker's
    own first bar are NaN, never filled -- `pd.DataFrame.corr()` and the
    rolling correlation below both use pairwise-complete observations, so
    this is the correct causal representation of "not yet listed", not a
    zero return."""
    if frames is None:
        frames = load_universe(tickers)
    cols = {}
    for t, df in frames.items():
        daily_close = df["close"].resample("1D").last()
        cols[t] = np.log(daily_close).diff()
    out = pd.concat(cols, axis=1)
    out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index
    return out


def rolling_mean_pairwise_correlation(daily_returns: pd.DataFrame,
                                       window: int = ROLL_DAYS,
                                       min_periods: int = MIN_PERIODS) -> pd.Series:
    """CONSERVATIVE statistic: trailing `window`-day mean of the off-diagonal
    pairwise Pearson correlation across every column PRESENT that day (NaN
    columns for tickers not yet listed are excluded from that day's mean via
    pandas' own pairwise-complete `.corr()`. Causal: day t's value uses only
    returns through day t's own close (the causal 1-day shift onto the 5m
    strategy frame happens in `align_connectedness_causal`, not here)."""
    n_assets = daily_returns.shape[1]
    idx_pairs = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]
    out = pd.Series(index=daily_returns.index, dtype=float)
    values = daily_returns.to_numpy()
    for end in range(len(daily_returns)):
        start = max(0, end - window + 1)
        window_vals = values[start:end + 1]
        if window_vals.shape[0] < min_periods:
            continue
        with np.errstate(invalid="ignore"):
            corr = pd.DataFrame(window_vals).corr().to_numpy()
        pair_vals = [corr[i, j] for i, j in idx_pairs
                     if np.isfinite(corr[i, j])]
        if len(pair_vals) >= 3:  # need at least 3 pairs (>=3 assets) for a meaningful mean
            out.iloc[end] = float(np.mean(pair_vals))
    return out


def align_connectedness_causal(daily_signal: pd.Series, bars: pd.DataFrame) -> pd.Series:
    """Reindex a daily connectedness statistic onto `bars`' 5m index, causally:
    a bar may only see the reading for the most recent day that closed
    strictly before that bar's own day -- identical shift-by-1-day-then-ffill
    convention as `align_mvrv_causal`/`align_onchain_causal`/etc. in
    `tradebot/data.py`. Bars before the first visible row (including all of
    the panel's pre-2020-01-01 history) get NaN, never filled or back-cast."""
    shifted = daily_signal.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    reindexed = shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill()
    return reindexed.reindex(bars.index)


# ------------------------------------------------------- VAR(1) + GFEVD (novel)


def fit_var1_ols(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares VAR(1) fit: y[t] = c + A @ y[t-1] + eps[t], columns =
    variables, rows = time. Returns (A, Sigma) -- the k x k coefficient
    matrix (constant term discarded, only the dynamics matter for GFEVD) and
    the k x k residual covariance. Pure `numpy.linalg.lstsq` OLS, no
    `statsmodels` dependency (not in pyproject.toml, and this project's own
    convention -- see `evidence.py`, `inference.py` -- is hand-rolled
    numpy/pandas statistics throughout)."""
    y = np.asarray(y, dtype=float)
    k = y.shape[1]
    Y = y[1:]
    X = np.hstack([np.ones((y.shape[0] - 1, 1)), y[:-1]])
    coefs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = coefs[1:].T  # k x k, A[i, j] = effect of var j (t-1) on var i (t)
    resid = Y - X @ coefs
    dof = max(Y.shape[0] - X.shape[1], 1)
    Sigma = (resid.T @ resid) / dof
    return A, Sigma


def gfevd_total_spillover(A: np.ndarray, Sigma: np.ndarray, horizon: int = 10) -> float:
    """Diebold & Yilmaz (2012) Total Spillover Index from a VAR(1)'s
    generalized forecast-error variance decomposition (Pesaran & Shin 1998,
    order-invariant). Returns a percentage in [0, 100]: the share of each
    variable's H-step forecast-error variance attributable to shocks in the
    OTHER variables, summed and normalized by the total. A VAR(1)'s MA(inf)
    coefficients are Psi_h = A^h (companion form is the identity for lag 1)."""
    k = A.shape[0]
    sigma_diag = np.diag(Sigma)
    if not np.all(np.isfinite(sigma_diag)) or np.any(sigma_diag <= 0):
        return float("nan")
    theta = np.zeros((k, k))  # theta[i, j] = variance of var i explained by shocks in var j
    psi_pow = np.eye(k)
    denom = np.zeros(k)
    for h in range(horizon):
        num_h = (psi_pow @ Sigma) ** 2 / sigma_diag[None, :]
        theta += num_h
        denom += np.sum((psi_pow @ Sigma) * psi_pow, axis=1)
        psi_pow = psi_pow @ A
    theta = theta / denom[:, None]
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = np.nan
    theta_norm = theta / row_sums  # normalized GFEVD, rows sum to 1
    off_diag_sum = theta_norm.sum() - np.trace(theta_norm)
    tsi = 100.0 * off_diag_sum / k
    return float(tsi)


def rolling_total_spillover_index(daily_returns: pd.DataFrame, window: int = 100,
                                   horizon: int = 10, min_assets: int = 4) -> pd.Series:
    """NOVEL statistic: rolling VAR(1)+GFEVD Total Spillover Index at daily
    frequency. At each day t, fit on the trailing `window` days using only
    columns with a complete (no-NaN) window (so the asset count can grow as
    the panel lists more tickers over inner-train/inner-validation); requires
    >= `min_assets` complete columns or the day is NaN. Causal by construction
    -- day t's value uses only returns through day t's own close."""
    idx = daily_returns.index
    values = daily_returns.to_numpy()
    out = pd.Series(index=idx, dtype=float)
    for end in range(window, len(idx)):
        start = end - window
        win = values[start:end + 1]
        complete = ~np.isnan(win).any(axis=0)
        if complete.sum() < min_assets:
            continue
        sub = win[:, complete]
        try:
            A, Sigma = fit_var1_ols(sub)
            out.iloc[end] = gfevd_total_spillover(A, Sigma, horizon=horizon)
        except np.linalg.LinAlgError:
            continue
    return out


# ------------------------------------------------------------ comparison harness


def flat_rescale_to_exposure(target: np.ndarray, desired_mean_abs: float) -> np.ndarray:
    """Rescale `target` by a single constant so its mean |exposure| matches
    `desired_mean_abs` -- the "did you just hold less" control R-62/R-33
    require before crediting a scale-axis factor with anything beyond
    exposure reduction."""
    target = np.asarray(target, dtype=float)
    current = float(np.nanmean(np.abs(target)))
    if current <= 0:
        return target
    k = desired_mean_abs / current
    return target * k


def compare_scaled(candidate_build, *, label: str, btc: pd.DataFrame | None = None,
                    eth: pd.DataFrame | None = None,
                    slices: dict | None = None) -> list[dict]:
    """Candidate vs (a) unmodified kelly_regime_v4 and (b) v4 flat-rescaled to
    the candidate's own mean exposure, on every frozen slice/market. Never
    reads a bar at or after OOS_START."""
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare_scaled(): btc")
    if eth is None:
        eth = load_eth()
    assert_no_holdout(eth, "compare_scaled(): eth")
    if slices is None:
        slices = {
            "inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
            "inner_val": (INNER_VAL_START, INNER_VAL_END),
        }

    cand = TargetStrategy(candidate_build, name=f"r187_{label}")
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")

    rows = []
    jobs = [(name, start, end, btc, "BTC") for name, (start, end) in slices.items()]
    jobs += [(f"{name}_eth", start, end, eth, "ETH") for name, (start, end) in slices.items()]

    for slice_name, start, end, df, asset in jobs:
        for market in (SPOT, FUTURES):
            a = run_slice(cand, df, start, end, slice_name, market)
            b = run_slice(ctrl, df, start, end, slice_name, market)

            def rescaled_build(d, _mean=a.mean_abs_exposure):
                return flat_rescale_to_exposure(v4_target(d), _mean)

            rescaled_ctrl = TargetStrategy(rescaled_build, name="kelly_regime_v4_rescaled")
            c = run_slice(rescaled_ctrl, df, start, end, slice_name, market)

            pr_vs_v4 = paired_diff(a.daily, b.daily)
            pr_vs_rescaled = paired_diff(a.daily, c.daily)
            exp_ratio = a.mean_abs_exposure / b.mean_abs_exposure if b.mean_abs_exposure else float("nan")
            vol_ratio = a.realized_vol / b.realized_vol if b.realized_vol else float("nan")
            rows.append({
                "label": label, "slice": slice_name, "asset": asset, "market": market.name,
                "cand_final": a.final_balance, "v4_final": b.final_balance,
                "v4rescaled_final": c.final_balance,
                "d_sharpe_v4": a.sharpe - b.sharpe,
                "d_sharpe_rescaled": a.sharpe - c.sharpe,
                "d_dd_v4": a.max_drawdown_pct - b.max_drawdown_pct,
                "d_dd_rescaled": a.max_drawdown_pct - c.max_drawdown_pct,
                "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
                "boot_d_loggrowth_rescaled": pr_vs_rescaled.diff.point,
                "boot_lo_rescaled": pr_vs_rescaled.diff.lo, "boot_hi_rescaled": pr_vs_rescaled.diff.hi,
                "excludes_zero_rescaled": bool(pr_vs_rescaled.diff.lo > 0 or pr_vs_rescaled.diff.hi < 0),
                "boot_d_loggrowth_v4": pr_vs_v4.diff.point,
                "boot_lo_v4": pr_vs_v4.diff.lo, "boot_hi_v4": pr_vs_v4.diff.hi,
            })
    return rows


def print_scaled_rows(rows: list[dict]) -> None:
    hdr = (f"{'label':20s} {'slice':14s} {'asset':4s} {'market':11s} "
           f"{'cand$':>10s} {'v4$':>10s} {'v4resc$':>10s} "
           f"{'dSh(resc)':>9s} {'dDD(resc)':>9s} {'expR':>5s} {'volR':>5s} "
           f"{'dlogG(resc)':>11s} {'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:20]:20s} {r['slice']:14s} {r['asset']:4s} {r['market']:11s} "
              f"{r['cand_final']:10,.0f} {r['v4_final']:10,.0f} {r['v4rescaled_final']:10,.0f} "
              f"{r['d_sharpe_rescaled']:+9.2f} {r['d_dd_rescaled']:+9.1f} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{r['boot_d_loggrowth_rescaled']:+11.3f} "
              f"{r['boot_lo_rescaled']:+8.3f},{r['boot_hi_rescaled']:+8.3f} "
              f"{'YES' if r['excludes_zero_rescaled'] else 'no':>5s}")


# ---------------------------------------------------------------- Step-0 gate


def _block_bootstrap_shift_null(episode_dates: list[pd.Timestamp], signal: pd.Series,
                                 quantile: float = LEAD_QUANTILE, n_shifts: int = 200,
                                 max_shift_days: int = 120, seed: int = 0) -> float:
    """Fraction of `n_shifts` uniformly-random time shifts of `signal` whose
    median 'lead' over `episode_dates` is >= the observed one -- the same
    "is this better than an arbitrary time shift" null R-84 used."""
    rng = np.random.default_rng(seed)
    observed = _median_lead_days(episode_dates, signal, quantile)
    if observed is None:
        return float("nan")
    hits = 0
    valid = 0
    for _ in range(n_shifts):
        shift = int(rng.integers(-max_shift_days, max_shift_days + 1))
        shifted = signal.copy()
        shifted.index = shifted.index + pd.Timedelta(days=shift)
        lead = _median_lead_days(episode_dates, shifted, quantile)
        if lead is None:
            continue
        valid += 1
        if lead >= observed:
            hits += 1
    return hits / valid if valid else float("nan")


def _median_lead_days(episode_dates: list[pd.Timestamp], signal: pd.Series,
                       quantile: float) -> float | None:
    leads = []
    for ep in episode_dates:
        history = signal.loc[:ep].dropna()
        if len(history) < MIN_PERIODS:
            continue
        thresh = history.quantile(quantile)
        recent = history.iloc[-30:]  # look for the crossing in the 30 days before the episode
        crossed = recent[recent >= thresh]
        if crossed.empty:
            continue
        first_cross = crossed.index[0]
        lead = (pd.Timestamp(ep) - first_cross).days
        leads.append(lead)
    return float(np.median(leads)) if leads else None


def step0_gate(btc: pd.DataFrame | None = None) -> dict:
    """Computed ONCE, over the full pre-holdout history (through
    INNER_VAL_END, never OOS_START or later), before either branch is
    implemented in full. Uses the CONSERVATIVE branch's rolling mean pairwise
    correlation (cheaper than the novel branch's rolling VAR) as the shared
    Step-0 statistic. Returns per-episode leads, the median, and the
    block-bootstrap-null p-value."""
    frames = load_universe()
    daily = panel_daily_log_returns(frames)
    if btc is not None:
        assert_no_holdout(btc, "step0_gate")
    daily_pre_holdout = daily.loc[:INNER_VAL_END]
    rho = rolling_mean_pairwise_correlation(daily_pre_holdout)

    usable = [(name, pd.Timestamp(date, tz="UTC")) for name, date in STRESS_EPISODES
              if name not in FORCED_NAN_EPISODES]
    per_episode = {}
    for name, ep in usable:
        history = rho.loc[:ep].dropna()
        if len(history) < MIN_PERIODS:
            per_episode[name] = None
            continue
        thresh = history.quantile(LEAD_QUANTILE)
        recent = history.iloc[-30:]
        crossed = recent[recent >= thresh]
        lead = (ep - crossed.index[0]).days if not crossed.empty else None
        per_episode[name] = lead

    leads = [v for v in per_episode.values() if v is not None]
    median_lead = float(np.median(leads)) if leads else float("nan")
    n_positive = sum(1 for v in leads if v > 0)
    null_p = _block_bootstrap_shift_null([ep for _, ep in usable], rho)

    # null_p = fraction of ARBITRARY time shifts whose lead is >= the observed
    # one; a LOW null_p means the observed lead beats most random shifts (a
    # real leading relationship), a HIGH null_p means the observed lead is
    # unremarkable or worse than noise (R-84's "indistinguishable from an
    # arbitrary time-shift" failure mode). Gate requires null_p <= 0.5 --
    # deliberately loose (beats only a bare plurality of shifts), not the
    # inverted condition an earlier draft of this file had.
    gate_pass = bool(leads and n_positive > len(leads) / 2 and
                      (not np.isnan(null_p)) and null_p <= 0.5)
    return {
        "per_episode_lead_days": per_episode,
        "median_lead_days": median_lead,
        "n_usable_episodes": len(leads),
        "n_positive_lead": n_positive,
        "null_p_value": null_p,
        "gate_pass": gate_pass,
    }


def _self_test() -> None:
    """Sanity + causality checks, run on import."""
    frames = load_universe()
    daily = panel_daily_log_returns(frames)
    assert daily.shape[1] == len(UNIVERSE_8)

    covered = daily.loc["2020-06-01":"2021-06-01"]
    rho = rolling_mean_pairwise_correlation(covered)
    assert rho.notna().sum() > 0, "rolling correlation produced no finite values in a fully-panel-covered sample"

    btc = load_btc()
    small = btc.iloc[: 120 * BARS_PER_DAY]
    aligned = align_connectedness_causal(rho, small)
    assert len(aligned) == len(small)
    # Everything before the panel's first real coverage day must be NaN.
    pre_coverage = aligned.loc[: PANEL_COVERAGE_START - pd.Timedelta(days=1)]
    assert pre_coverage.isna().all(), "connectedness leaked before panel coverage start"

    sub = daily.dropna().iloc[:250]
    if len(sub) >= 150 and sub.shape[1] >= 4:
        A, Sigma = fit_var1_ols(sub.to_numpy())
        tsi = gfevd_total_spillover(A, Sigma)
        assert 0.0 <= tsi <= 100.0 or np.isnan(tsi), f"TSI out of range: {tsi}"

    print("r187_shared self-test: OK "
          f"(panel shape={daily.shape}, rho finite={rho.notna().sum()}, "
          f"no pre-coverage leakage)")


if __name__ == "__main__":
    import json
    _self_test()
    gate = step0_gate()
    print(json.dumps(gate, indent=2, default=str))
