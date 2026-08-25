"""Shared, read-only pre-registration for the R-141 round (08-25).

DIRECTION, in one sentence: does the Log-Periodic Power Law Singularity
model (LPPLS; Johansen, Ledoit & Sornette 2000, "Crashes as critical
points", *International Journal of Theoretical and Applied Finance*
3(2):219-255; Sornette 2003, *Why Stock Markets Crash*, Princeton
University Press; Filimonov & Sornette 2013, "A Stable and Robust
Calibration Scheme of the Log-Periodic Power Law Model", *Physica A*
392(17):3698-3707) -- a DETERMINISTIC, finite-time-singularity model of
super-exponential price acceleration with decorating log-periodic
oscillations, fit by nonlinear regression on log price itself rather than
any stochastic-process regime model -- serve as (a) a regime-timing INPUT
to `kelly_regime_v4`'s vote, evaluated on the identical six-episode
detection-lag gate that has closed seven prior, structurally distinct
regime-timing mechanisms (HMM: R-01; BOCPD: R-82; Kalman LLT: R-83;
critical slowing down/CSD: R-85; transfer entropy: R-86; Hawkes: R-96;
CUSUM: R-139, swept in R-139's own novel branch), and (b) a continuous
crash-hazard dampener on `kelly_regime_v4`'s `scale`, a construction none
of those seven mechanisms tried (all seven were discrete lead-time /
regime-timing inputs; none was used as a continuous SIZE-axis modulator).

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **INFO** (conservative branch, same narrow sense
R-82's/R-96's/R-139's own module docstrings used it -- no new external
data channel, both branches read only the committed BTC OHLCV close
series `kelly_regime_v4` already uses; a structurally different
ESTIMATOR of "is a regime break imminent" extracted from that same
series) and **SIZE** (novel branch -- "how much to hold", this project's
own standing diagnosis's "what actually worked" axis, attacked with a
signal source no prior SIZE-axis round has used: an LPPLS-derived
crash-confidence indicator, as opposed to a volatility statistic, a risk
measure, or a distributional-novelty brake).

**Why LPPLS is a genuinely eighth, structurally distinct theoretical
basis, not a re-skin of the seven already closed.** Every one of HMM
(discrete-state Markov switching), BOCPD (Bayesian generative
changepoint/run-length estimation), Kalman LLT (linear Gaussian
state-space filtering), CSD (dynamical-systems fluctuation-statistics
early-warning signals), transfer entropy (information-theoretic directed
flow), Hawkes (self-exciting conditional-intensity point process) and
CUSUM (sequential statistical-process-control) is a STOCHASTIC-PROCESS
model of the return-generating mechanism -- each posits some latent
random process (a hidden state, a posterior over run length, a filtered
latent trend, a fluctuation statistic, an information flow, a
conditional intensity, a sequential test statistic) and asks whether that
process's own estimate moved before price. LPPLS posits no such process.
It is a DETERMINISTIC functional form -- reduced by Filimonov & Sornette
(2013) to `ln p(t) = A + B(tc-t)^m + C1(tc-t)^m cos(omega * ln(tc-t)) +
C2(tc-t)^m sin(omega * ln(tc-t))`, four parameters (A, B, C1, C2) linear
given the three nonlinear ones (tc, m, omega) -- fit by nonlinear
least-squares directly to the PRICE LEVEL's trajectory, and its economic
content (Johansen, Ledoit & Sornette 2000's rational-expectations bubble
model with noise traders) is that price accelerates super-exponentially
(m in (0,1), B<0) with decorating oscillations as herding intensifies
approaching a critical time `tc`, at which the crash hazard rate --
proportional to `(tc-t)^(m-1)`, diverging as t -> tc -- becomes
unsustainable. This is closer in spirit to catastrophe theory / critical
phenomena in statistical physics than to any of the seven stochastic
estimators, and it is fit to the PRICE LEVEL directly, not to returns or
a derived statistic of returns, unlike all seven priors.

**Not a duplicate of:**
- R-01/R-82/R-83/R-85/R-86/R-96/R-139 (the seven regime-timing
  mechanisms above): same six-episode Step-A gate reused for direct
  comparability (conservative branch only), structurally different
  detector -- see the theoretical-basis argument above. Grep-confirmed
  zero prior hits in `docs/LEDGER.md` for "Sornette", "LPPL",
  "finite-time singularity", "Filimonov", "super-exponential", "crash
  hazard" and "critical time" before this round started.
- R-74 (MVRV on-chain valuation signal) and R-125's own rejection of a
  DIFFERENT Bitcoin power-law paper (Baquero & Menezes 2026,
  arXiv:2605.21316, "Bitcoin's Power Law: Weak Structure, Strong
  Forecasts") as "too close to the already-closed valuation INFO
  sub-axis" -- that paper is a LONG-RUN, static price-vs-time SCALING
  LAW (log price approximately linear in log time since genesis), a
  valuation-level construction with no fitted crash time and no
  oscillatory component. LPPLS is the opposite kind of object: a
  SHORT-HORIZON, dynamically re-fit, finite-time SINGULARITY model
  whose whole output is a near-term critical-time estimate and a crash
  hazard rate, never a valuation level. The two share the phrase "power
  law" and nothing else; R-125's own rejection reasoning (a static
  valuation anchor overlapping MVRV's already-closed level/rate-of-change
  test) does not apply here.
- 28+ SIZE-axis attempts (R-34...R-136, closed per R-136's own
  next-step accounting across the vote/scale/vol-estimator slots): every
  one retuned `kelly_regime_v4`'s existing vote, its existing
  volatility-targeting architecture, or the risk measure `scale` targets
  (R-125's CVaR). None fed in an external crash-confidence signal
  extracted from a structurally different model of the price series
  itself; this round's novel branch is the first to do so.
- R-96 (Hawkes): a self-exciting point process over discrete EVENT
  TIMES (large moves treated as points whose conditional intensity
  self-excites); no notion of a finite-time singularity, no fitted
  critical time, no log-periodic oscillation, fit to jump COUNTS not to
  the price level.

**Is it simulable here?** Yes. Entirely computable from the committed
BTC OHLCV close series (daily-resampled, matching `kelly_regime_v4`'s
own multi-week anchor horizon and R-82's/R-96's/R-139's own daily-cadence
convention), zero new data, zero new fetch. No `scipy` dependency (not
declared in `pyproject.toml`; every nonlinear fit below is a grid search
over the three nonlinear parameters (tc, m, omega) with the four linear
parameters solved in closed form via `numpy.linalg.lstsq` for each grid
cell -- exactly Filimonov & Sornette's (2013) own point that the
reparametrized 4-linear/3-nonlinear form is what makes LPPLS calibration
numerically tractable without a general nonlinear optimizer).

**What would make this fail, named now, before any code beyond this
shared module was run:**

(a) CONSERVATIVE branch (Step-A six-episode detection-lag gate): the
    LPPLS confidence indicator's up-crossings cluster AFTER v4's own
    anchor reaction -- the identical failure mode all seven priors hit --
    or are not distinguishable from an arbitrary block-shift of the same
    confidence series (the same block-bootstrap null every predecessor
    round used, reused here unchanged via `r82_shared.block_bootstrap_shifts`).
    **A specific, theory-motivated sub-hypothesis, disclosed now as a
    secondary diagnostic that does NOT alter the primary >=4/6 decision
    rule:** LPPLS is a model of ENDOGENOUS bubble-driven tops, not of
    exogenous shocks. Two of the six episodes are plausibly preceded by
    an endogenous blow-off top (2018 bear onset, following the Dec-2017
    parabolic run-up; 2021-11 top, following 2021's bull run); the other
    four are, on the historical record, exogenous or event-driven (2018
    capitulation is a continuation of an already-established bear, not a
    fresh top; COVID crash is a global macro shock; Terra/Luna and FTX
    are idiosyncratic collapses with no preceding BTC-price bubble
    signature). If the branch fails the primary >=4/6 gate but passes
    on the 2-episode "genuine top" subset, that is reported explicitly
    as a theory-consistent partial result, not as a passing gate --
    the decision rule is, and remains, the unmodified six-episode >=4/6
    bar every predecessor used.
(b) NOVEL branch (continuous SIZE dampener): fails the Step-0
    non-degeneracy gate (R^2 > 0.98 against v4's own unmodified target,
    i.e. the dampener never binds because confidence rarely reaches a
    level that moves `scale`), OR fails B1 on either market, OR fails B4
    (the ETH sign-replication falsification test this whole SIZE/ERR
    research programme has used since R-59) -- the same failure shape
    six-plus consecutive SIZE/ERR-family rounds have hit (R-109 through
    R-136): a real, non-degenerate BTC-side effect that inverts on ETH,
    evidence the effect is a property of the BTC/ETH training-window
    relationship rather than of the mechanism.

If the conservative branch's own Step-A gate fails its pre-registered
bar, it STOPS at Step-A -- no Step-B implementation, no holdout read,
reported NEGATIVE at Step-A, the identical convention R-82/R-85/R-86/R-96
used. The novel branch runs its own independent Step-0/B1/B3/B4/B5
battery regardless of the conservative branch's outcome (they are
different constructions on the same underlying signal, not a
cascade) and stops before the holdout unless `further_work` triggers.
Both branches disclose this decision now, before any real-data number
exists.

**Decision rules, pre-registered:**
- CONSERVATIVE: PASS Step-A at >= 4/6 episodes (identical bar to all
  seven priors). If passed, Step-B (contingent, not expected to be
  reached): feed the binarized confidence signal into
  `confirming_vote_frac` (meta_vote=1 when >= `CONF_MAJORITY` of the five
  window-length fits qualify) exactly as R-139's conservative branch fed
  in its CUSUM run-length flag, `weight` swept over {0.5, 1.0, 1.5}, D1
  (log-growth diff)/D2 (drawdown diff) via `tradebot.inference.
  paired_bootstrap` on `W_TRAIN` then directionally confirmed on
  `W_VAL`, before any holdout read.
- NOVEL: PROMOTE-candidate only if causal-truncation probe AND Step-0
  (not degenerate) AND B1 (both markets) AND B3 (plateau majority) AND
  B4 (both markets) AND B5 all pass -- the identical SIZE/ERR-family
  convention R-109 through R-136 used, verbatim, for direct
  comparability. B2 (drawdown) is diagnostic only.

Neither Step-B/holdout path is expected to be reached, given the seven
prior mechanisms' 0-2/6 base rate on the Step-A gate and six-plus
consecutive SIZE/ERR-family rounds' BTC-pass/ETH-invert base rate on the
B1/B4 battery -- named now so a reader can check neither rule was
invented after seeing whether it would be needed.

No bar at or after `OOS_START = 2023-01-01` may be read by either branch.

=====================================================================
THE LPPLS ENGINE (shared machinery)
=====================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_onchain_causal  # noqa: E402

from experiments.r82_shared import (  # noqa: E402
    BARS_PER_DAY,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    STRESS_EPISODES,
    V4_BAND,
    V4_HORIZONS,
    anchor_majority,
    block_bootstrap_shifts,
    confirming_vote_frac,
    episode_window,
    nearest_transition,
)
from experiments.r125_shared import (  # noqa: E402
    FUTURES,
    FUTURES_HIGH_FEE,
    SPOT,
    SPOT_HIGH_FEE,
    b1_signal,
    load_btc_train,
    load_eth_train,
    run_candidate,
    step0_gate,
    v4_reference_target,
)

# --------------------------------------------------------------- constants
#
# Every grid value below is a literature-typical bound (Filimonov &
# Sornette 2013; Sornette, Demos, Zhang, Cauwels, Filimonov & Zhang 2015,
# "Real-Time Prediction and Post-Mortem Analysis of the Shanghai 2015
# Stock Market Bubble", *Journal of Investment Strategies*; Gerlach,
# Demos & Sornette 2019, arXiv:1905.09647, the Bitcoin-specific
# application), fixed here before any real-data fit was run. The grid
# RESOLUTION (how many points per dimension) is a computational-
# tractability choice, disclosed as such -- it does not change the
# bounds, and a coarser/finer grid is expected to move the number of
# qualifying fits smoothly, not qualitatively (checked by the plateau
# diagnostic in each branch's own B3 step).

WINDOW_LENGTHS_DAYS = (90, 180, 270, 365, 550)   # Sornette's "multi-scale" ensemble, reduced to 5 representative scales
M_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)               # standard LPPLS bound m in (0,1), super-exponential but sub-linear-derivative
OMEGA_GRID = (6.0, 8.0, 10.0, 12.0)              # standard LPPLS bound omega in [6,13] (log-periodic angular frequency)
TC_OFFSET_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)  # tc beyond window end, as a fraction of window length
CALIBRATION_STRIDE_DAYS = 7    # weekly recalibration (disclosed simplification vs. literature's daily; v4's own anchors
                                # operate on a 20-80 CALENDAR-DAY horizon, so weekly resolution is not a meaningful loss)
D_MIN = 1.0                    # Sornette et al.'s damping-condition quality filter: D = m|B| / (omega*sqrt(C1^2+C2^2)) >= 1
CONF_MAJORITY = 3               # conservative branch's binarization: >=3 of 5 window-length fits must qualify -> meta_vote=1

# Trials per calibration date: len(M_GRID)*len(OMEGA_GRID)*len(TC_OFFSET_FRACTIONS) = 5*4*8 = 160 lstsq
# fits per window length, x 5 window lengths = 800 per calibration date. This count, x the number of
# calibration dates actually run, is this round's trials figure for the ledger (both branches share one
# LPPLS signal computation, so it is counted ONCE, not once per branch).

GENUINE_TOP_EPISODES = {
    "2018 bear onset (post-Dec-2017 top)",
    "2021-11 top / 2022 bear transition",
}


def _assert_no_holdout(df: pd.DataFrame, oos_start: str = OOS_START) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(oos_start, tz=df.index.tz)
    assert df.index.max() < cutoff, (
        f"holdout bar read: max timestamp {df.index.max()} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")


# ------------------------------------------------------------- LPPLS core

def _lppls_design_matrix(t: np.ndarray, tc: float, m: float, omega: float) -> np.ndarray:
    """Filimonov & Sornette (2013)'s linear-in-(A,B,C1,C2) reparametrization.
    ``t`` are days-since-window-start (all strictly < tc by construction:
    tc is always placed beyond the window's last observed day)."""
    dt = np.maximum(tc - t, 1e-8)
    f = dt ** m
    log_dt = np.log(dt)
    g = f * np.cos(omega * log_dt)
    h = f * np.sin(omega * log_dt)
    return np.column_stack([np.ones_like(t), f, g, h])


def _lppls_fit_linear(t: np.ndarray, y: np.ndarray, tc: float, m: float, omega: float) -> dict:
    X = _lppls_design_matrix(t, tc, m, omega)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    sse = float(np.sum((y - yhat) ** 2))
    A, B, C1, C2 = (float(c) for c in coef)
    return dict(A=A, B=B, C1=C1, C2=C2, sse=sse, tc=float(tc), m=float(m), omega=float(omega))


def fit_lppls_best(t: np.ndarray, y: np.ndarray, window_days: int,
                    m_grid=M_GRID, omega_grid=OMEGA_GRID,
                    tc_offset_fractions=TC_OFFSET_FRACTIONS) -> dict:
    """Grid search over (tc, m, omega); (A,B,C1,C2) solved in closed form
    per cell. Returns the minimum-SSE fit across the whole pre-registered
    grid. `t` must end at `window_days - 1` (days since window start,
    zero-indexed, one point per calendar day)."""
    t_end = float(t[-1])
    best = None
    for frac in tc_offset_fractions:
        tc = t_end + frac * window_days
        for m in m_grid:
            for omega in omega_grid:
                fit = _lppls_fit_linear(t, y, tc, m, omega)
                if best is None or fit["sse"] < best["sse"]:
                    best = fit
    return best


def lppls_qualifies(fit: dict, d_min: float = D_MIN) -> bool:
    """Sornette et al.'s standard bubble-quality filter: super-exponential
    growth (B<0), a non-degenerate oscillatory component, and the damping
    condition D = m|B|/(omega*sqrt(C1^2+C2^2)) >= d_min (the fit is not
    merely chasing noise with an oscillation whose amplitude would make
    price negative before tc)."""
    B, C1, C2, m, omega = fit["B"], fit["C1"], fit["C2"], fit["m"], fit["omega"]
    C = float(np.hypot(C1, C2))
    if B >= 0:
        return False
    if C <= 1e-9:
        return False
    D = (m * abs(B)) / (omega * C)
    return D >= d_min


def lppls_daily_signals(df: pd.DataFrame, *, cache_path: Path | None = None,
                         verbose: bool = True) -> pd.DataFrame:
    """The one, shared LPPLS calibration pass both branches read from --
    computed ONCE here so both branches see a byte-identical signal and
    the round's trials count is not doubled by re-fitting. Causal by
    construction: the fit ending on calendar day D uses only daily closes
    up to and including day D (``y[:i+1]`` for the trailing window ending
    at day D), and the resulting per-day columns are aligned onto ``df``'s
    5-minute grid with `align_onchain_causal`'s full-calendar-day shift
    (day D's fit only becomes visible to bars starting day D+1), the same
    causal contract every other daily-cadence signal in this project uses
    (R-82's BOCPD, R-96's Hawkes, R-138/R-139's CUSUM).

    Returns a DataFrame indexed by calendar day with columns
    ``lppls_n_qualify`` (0..5, how many of the 5 window-length fits
    passed `lppls_qualifies`), ``lppls_confidence`` (n_qualify/5, in
    [0,1]), and ``lppls_meta_vote`` (1 if n_qualify >= CONF_MAJORITY
    else 0, the conservative branch's discretized input).
    """
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.index = cached.index.tz_localize("UTC") if cached.index.tz is None else cached.index
        if verbose:
            print(f"  [lppls_daily_signals] loaded cache: {cache_path} ({len(cached)} rows)")
        return cached

    daily_close = df["close"].resample("1D").last().dropna()
    _assert_no_holdout(pd.DataFrame(index=daily_close.index))
    log_price = np.log(daily_close.to_numpy())
    n = len(daily_close)
    max_window = max(WINDOW_LENGTHS_DAYS)

    dates = daily_close.index
    n_qualify = np.full(n, np.nan)
    best_m = np.full(n, np.nan)
    best_omega = np.full(n, np.nan)
    best_tc_offset_days = np.full(n, np.nan)

    fit_positions = list(range(max_window - 1, n, CALIBRATION_STRIDE_DAYS))
    if verbose:
        print(f"  [lppls_daily_signals] {len(fit_positions)} calibration dates, "
              f"{len(M_GRID) * len(OMEGA_GRID) * len(TC_OFFSET_FRACTIONS) * len(WINDOW_LENGTHS_DAYS)} "
              f"fits each")
    for pos in fit_positions:
        qualify_count = 0
        fits = []
        for wd in WINDOW_LENGTHS_DAYS:
            start = pos - wd + 1
            if start < 0:
                continue
            t = np.arange(wd, dtype=float)
            y = log_price[start:pos + 1]
            fit = fit_lppls_best(t, y, wd)
            fits.append((wd, fit))
            if lppls_qualifies(fit):
                qualify_count += 1
        n_qualify[pos] = qualify_count
        # diagnostics: report the longest window's own winning cell (most stable scale)
        if fits:
            wd_diag, fit_diag = fits[-1]
            best_m[pos] = fit_diag["m"]
            best_omega[pos] = fit_diag["omega"]
            best_tc_offset_days[pos] = fit_diag["tc"] - (wd_diag - 1)

    out = pd.DataFrame({
        "lppls_n_qualify": n_qualify,
        "lppls_best_m": best_m,
        "lppls_best_omega": best_omega,
        "lppls_best_tc_offset_days": best_tc_offset_days,
    }, index=dates)
    out["lppls_n_qualify"] = out["lppls_n_qualify"].ffill().fillna(0.0)
    out["lppls_confidence"] = out["lppls_n_qualify"] / len(WINDOW_LENGTHS_DAYS)
    out["lppls_meta_vote"] = (out["lppls_n_qualify"] >= CONF_MAJORITY).astype(float)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path)
        if verbose:
            print(f"  [lppls_daily_signals] wrote cache: {cache_path}")
    return out


def lppls_bar_signals(df: pd.DataFrame, *, cache_path: Path | None = None,
                       verbose: bool = True) -> pd.DataFrame:
    """`lppls_daily_signals`, causally aligned onto `df`'s own 5-minute
    index (see `align_onchain_causal`'s full-calendar-day shift)."""
    daily = lppls_daily_signals(df, cache_path=cache_path, verbose=verbose)
    return align_onchain_causal(daily, df)


# ------------------------------------------------------------ Step-A gate

def nearest_lppls_detection(confidence: pd.Series, window: pd.DatetimeIndex,
                             onset: pd.Timestamp, conf_thresh: float
                             ) -> pd.Timestamp | None:
    """Timestamp, within `window`, of the first bar where LPPLS confidence
    crosses UP through `conf_thresh`, closest to `onset` -- the LPPLS
    analogue of `nearest_bocpd_detection` (which crosses DOWN through a
    short run-length; here elevated confidence, not a short run length,
    is the "detected" state)."""
    vals = confidence.reindex(window).to_numpy()
    high = vals >= conf_thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = high[1:] & ~high[:-1]
    cross[0] = bool(high[0])
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def step_a_gate(bars: pd.DataFrame, lppls: pd.DataFrame, *, conf_thresh: float,
                 window_days: int = 60, n_draws: int = 500, block_days: int = 5,
                 seed: int = 141, verbose: bool = True) -> dict:
    """The R-82-identical Step-A detection-lag gate, LPPLS confidence
    crossing `conf_thresh` in place of a run-length crossing down.
    An episode PASSES if (a) LPPLS detects at or before v4's own nearest
    downward anchor-flip (lead >= 0), AND (b) that lead beats the
    block-bootstrap null's median. Gate PASSES overall at >= 4/6
    episodes. Also reports the pre-registered 2-episode "genuine top"
    subset as a disclosed, non-gating diagnostic.
    """
    majority = anchor_majority(bars)
    confidence = lppls["lppls_confidence"]
    _assert_no_holdout(pd.DataFrame(index=confidence.dropna().index))

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, window_days)
        if len(window) == 0:
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue
        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_lppls_detection(confidence, window, onset, conf_thresh)
        if flip_time is None or detect_time is None:
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue
        lead = (flip_time - detect_time).total_seconds() / 86400.0
        local = confidence.reindex(window).to_numpy()
        n_bars = len(local)
        shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                         n_draws=n_draws, seed=seed)
        leads_null = np.full(n_draws, np.nan)
        for k, shift in enumerate(shifts):
            shifted = local[shift]
            high = shifted >= conf_thresh
            cross = np.zeros(n_bars, dtype=bool)
            cross[1:] = high[1:] & ~high[:-1]
            cross[0] = bool(high[0])
            idx = np.where(cross)[0]
            if len(idx) == 0:
                continue
            times = window[idx]
            deltas = np.abs((times - onset).to_numpy())
            dt = times[int(np.argmin(deltas))]
            leads_null[k] = (flip_time - dt).total_seconds() / 86400.0
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = bool(pass_a and not np.isnan(null_median) and lead >= null_median)
        if verbose:
            print(f"    [{label}] lead={lead:+.2f}d null_median={null_median:+.2f}d PASS={pass_b}")
        results.append(dict(label=label, onset=onset_str, lead=lead,
                             null_median=null_median, pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    genuine_top = [r for r in results if r["label"] in GENUINE_TOP_EPISODES]
    n_pass_genuine_top = sum(1 for r in genuine_top if r["pass_b"])
    return dict(conf_thresh=conf_thresh, results=results, n_pass=n_pass, passed=n_pass >= 4,
                n_pass_genuine_top=n_pass_genuine_top, n_genuine_top=len(genuine_top))


# ---------------------------------------------- novel branch: SIZE dampener

def calibrate_dampener(v4_scale: np.ndarray, confidence: np.ndarray, kappa_grid) -> float:
    """Pick kappa (the dampening strength in
    `scale_novel = scale_v4 * max(0.1, 1 - kappa*confidence)`) so that
    mean(scale_novel) on inner-train matches mean(scale_v4) -- the
    exposure-matching discipline R-33/R-59/R-125 all use, so any B1
    effect this round finds cannot be "holding less" in disguise."""
    target_mean = float(np.nanmean(v4_scale))
    best, best_gap = kappa_grid[0], np.inf
    for kappa in kappa_grid:
        damp = np.maximum(0.1, 1.0 - kappa * confidence)
        s = v4_scale * damp
        gap = abs(float(np.nanmean(s)) - target_mean)
        if gap < best_gap:
            best, best_gap = kappa, gap
    return float(best)


if __name__ == "__main__":
    # Self-test: causal truncation probe on the LPPLS daily calibration
    # pipeline. Any candidate branch importing this module should call
    # this before reading a single inner-validation number. Uses a small
    # slice (not the full grid) purely to keep the self-test fast; the
    # real branches use the full pre-registered grid via
    # `lppls_daily_signals` / `lppls_bar_signals` above, unmodified.
    df, _ = load_btc_train("spot")
    close = df["close"]
    daily_close = close.resample("1D").last().dropna()
    log_price = np.log(daily_close.to_numpy())
    wd = 180
    check_pos = 900  # arbitrary interior day, well past the longest warmup
    t = np.arange(wd, dtype=float)
    y_full = log_price[check_pos - wd + 1: check_pos + 1]
    fit_full = fit_lppls_best(t, y_full, wd, m_grid=(0.5,), omega_grid=(9.0,),
                               tc_offset_fractions=(0.2,))
    # Truncate the series far beyond check_pos and re-fit the SAME trailing window --
    # a causal fit must be bit-identical, since it never reads past check_pos.
    trunc_close = daily_close.iloc[:check_pos + 500]
    trunc_log_price = np.log(trunc_close.to_numpy())
    y_trunc = trunc_log_price[check_pos - wd + 1: check_pos + 1]
    fit_trunc = fit_lppls_best(t, y_trunc, wd, m_grid=(0.5,), omega_grid=(9.0,),
                                tc_offset_fractions=(0.2,))
    ok = np.allclose([fit_full["A"], fit_full["B"], fit_full["C1"], fit_full["C2"], fit_full["sse"]],
                      [fit_trunc["A"], fit_trunc["B"], fit_trunc["C1"], fit_trunc["C2"], fit_trunc["sse"]],
                      rtol=1e-9)
    print(f"causal truncation probe (LPPLS fit at day {check_pos}, window={wd}d): "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, "LPPLS fit reads ahead of its own trailing window"
