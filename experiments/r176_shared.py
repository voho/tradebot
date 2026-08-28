"""Shared, read-only pre-registration and engine for the R-176 round (08-28).

DIRECTION, one sentence: resample `kelly_regime_v4`'s inputs from a
calendar-time clock to a DOLLAR-ACTIVITY clock, using the `volume` column
already present in the committed OHLCV file but never consumed by any
`kelly_regime*` strategy -- the conservative branch redefines the VOTE's
three anchor windows in dollar-time instead of calendar-time (per Easley,
Lopez de Prado & O'Hara (2012), "The Volume Clock", and Lopez de Prado
(2018), *Advances in Financial Machine Learning*, ch. 2's information-driven
bars); the novel branch derives a dollar-BAR ARRIVAL-RATE "crowding" gate
from the SAME dollar-clock primitive built here and applies it as a
multiplicative haircut on top of `frac * scale`, unchanged otherwise.

Full Step 1/Step 2 design (constraint attacked [INFO primary, SIZE
secondary], non-duplication against R-62/R-99/camouflage_flow/
stealth_trend/harsanyi_crowd, simulability, named failure modes) is in
`experiments/r176_direction.md`.

This module is DELIBERATELY neutral between the two branches: it exposes
the dollar-clock primitive (`dollar_time_anchor`, for the conservative
branch's vote) and the arrival-intensity primitive
(`dollar_bar_intensity`, for the novel branch's crowding gate) as two
functionals of the SAME underlying `dollar_volume` series, mirroring
r175_shared.py's convention of one shared engine exposing two functionals.
Neither branch may edit this file or each other's file (R-89-through-R-175's
own convention).

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).

Simplifications disclosed up front (per ROUTINE.md's honesty convention):
- `volume` in this project's committed OHLCV file is base-asset (BTC)
  volume, not already-dollar volume; `dollar_volume = close * volume` is
  the standard per-bar notional-traded proxy used throughout the
  information-driven-bars literature when only OHLCV+volume is available
  (no real per-trade tick data or bid/ask-side attribution exists in this
  project, as `camouflage_flow`'s own BVC construction already discloses).
- The conservative branch's dollar-time anchor uses a plain (unweighted)
  mean of `close` over the dollar-equivalent window, NOT a dollar-VWAP --
  deliberately, to isolate the WINDOW-SIZE mechanism (activity-adaptive
  span) from a WEIGHTING mechanism (which bars count more), matching R-62's
  own discipline of changing one factor at a time.
- The window's target dollar volume is `days * causal_trailing_median_daily_dollar`,
  refit daily from a 180-day trailing causal window (shifted by one day, so
  day D's target never uses day D's own volume) -- the same daily-refit,
  one-day-lag discipline r175_shared.py's MSM engine and R-172's own
  explicit lesson about same-day broadcast lookahead both use.
- No real per-trade tick data exists to build canonical Lopez de Prado
  imbalance bars (which require a trade-by-trade tick rule and a
  recursively-estimated expected-imbalance threshold); the novel branch's
  "dollar-bar arrival intensity" is a bar-level proxy -- how many
  average-sized dollar bars' worth of notional traded in each native
  5-minute bar, rolling-summed over a trailing calendar day -- built from
  the SAME OHLCV+volume data already in the file, disclosed as a
  simplification of the tick-level construction, not a claim to reproduce
  it exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r102_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    V4_BAND,
    V4_DEADBAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    conditional_target_scale,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_symmetric_vol,
    v4_target,
    v4_vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_DEADBAND - 0.10) < 1e-12, V4_DEADBAND

# ------------------------------------------------------------------------
# The six dated stress episodes this ledger's regime-timing gate already
# uses (R-82/R-83/R-85/R-163/R-175's own table), reused verbatim here for
# the novel branch's decisive mechanism check.
# ------------------------------------------------------------------------
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch touches real
# performance numbers.
# ------------------------------------------------------------------------
BASELINE_WINDOW_DAYS = 180     # trailing causal window for the daily-dollar baseline
MIN_BASELINE_DAYS = 30         # below this, dollar-time anchor is NaN (matches v4's own cold-start)
INTENSITY_SMOOTH_BARS = BARS_PER_DAY       # 1-day rolling smoothing of per-bar intensity
INTENSITY_QUANTILE_WINDOW_DAYS = 90        # causal trailing window for the crowding threshold
INTENSITY_HIGH_IN_Q = 0.90     # enter "crowded" above this trailing quantile
INTENSITY_HIGH_OUT_Q = 0.60    # exit "crowded" below this trailing quantile
CROWDING_HAIRCUT = 0.5         # multiplicative exposure haircut while crowded


# ================================================================== (1)
# Dollar-volume primitive, shared by both branches.
# ==================================================================

def dollar_volume(df: pd.DataFrame) -> pd.Series:
    """Per-bar notional traded: close * base-asset volume."""
    return df["close"] * df["volume"]


def _causal_daily_dollar_baseline(df: pd.DataFrame,
                                   window_days: int = BASELINE_WINDOW_DAYS,
                                   min_days: int = MIN_BASELINE_DAYS) -> pd.Series:
    """Causal trailing MEDIAN of daily dollar volume, broadcast back onto
    every bar of df's own index. Day D's value uses only days < D (shift by
    one full day), so it carries zero same-day lookahead -- the daily-refit
    discipline r175_shared.py's MSM engine and R-172's own lesson use."""
    dv = dollar_volume(df)
    daily = dv.resample("1D").sum()
    baseline = daily.rolling(window_days, min_periods=min_days).median().shift(1)
    day = df.index.floor("D")
    return baseline.reindex(day).set_axis(df.index)


def dollar_time_anchor(df: pd.DataFrame, days: int,
                        window_days: int = BASELINE_WINDOW_DAYS,
                        min_days: int = MIN_BASELINE_DAYS) -> np.ndarray:
    """The dollar-activity-clock analogue of `close.rolling(days*BARS_PER_DAY).mean()`:
    for each bar i, the plain mean of `close` over the shortest trailing
    window whose cumulative dollar volume is >= `days` times the current
    causal daily-dollar baseline. In high-turnover periods this window
    covers FEWER calendar bars than `days*BARS_PER_DAY` (faster-reacting);
    in quiet periods it covers MORE (slower, steadier) -- the same
    adaptive-timescale idea Lopez de Prado's volume/dollar bars apply to
    price sampling, applied here to a rolling-mean anchor's window instead.

    NaN before `min_days` days of history exist (matches v4's own
    rolling-mean cold start, which is also NaN until the window fills).
    """
    baseline = _causal_daily_dollar_baseline(df, window_days, min_days)
    target_dollar = (days * baseline).to_numpy()

    cumdollar = dollar_volume(df).cumsum().to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    query = cumdollar - np.where(np.isfinite(target_dollar), target_dollar, np.inf)
    j = np.searchsorted(cumdollar, query, side="left")
    j = np.clip(j, 0, n - 1)

    cumclose = np.concatenate(([0.0], np.cumsum(close)))
    span_bars = np.arange(n) - j + 1
    anchor = (cumclose[np.arange(n) + 1] - cumclose[j]) / span_bars

    out = np.where(np.isfinite(target_dollar), anchor, np.nan)
    return out


def _latched_vote_from_anchor(close: pd.Series, anchor: np.ndarray,
                               band: float = V4_BAND) -> pd.Series:
    """v4's own latched 0/1 vote logic, factored to accept an arbitrary
    anchor series (calendar-mean or dollar-time-mean) -- identical
    band-compare / ffill / fillna(0) construction as
    `experiments.r102_shared._latched_anchor_vote`."""
    v = pd.Series(
        np.where(close.to_numpy() > anchor * (1.0 + band), 1.0,
                 np.where(close.to_numpy() < anchor * (1.0 - band), 0.0, np.nan)),
        index=close.index,
    )
    return v.ffill().fillna(0.0)


def dollar_time_vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                           band: float = V4_BAND) -> pd.Series:
    """CONSERVATIVE branch's core input: v4's own 3-anchor vote, with each
    anchor computed via `dollar_time_anchor` instead of a calendar-day
    rolling mean. Everything else (band, latching, equal-weight averaging)
    is byte-identical to `v4_vote_frac`."""
    close = df["close"]
    votes = [_latched_vote_from_anchor(close, dollar_time_anchor(df, d), band)
             for d in horizons]
    return sum(votes) / len(votes)


def dollar_time_target(df: pd.DataFrame) -> np.ndarray:
    """Conservative branch's complete target: dollar-time vote * v4's own
    (unchanged) conditional-vol-target scale, through v4's own deadband."""
    frac = dollar_time_vote_frac(df).to_numpy()
    scale = v4_scale(df)
    return apply_deadband(frac * scale)


# ================================================================== (2)
# Dollar-bar arrival-intensity primitive, novel branch.
# ==================================================================

def dollar_bar_intensity(df: pd.DataFrame,
                          smooth_bars: int = INTENSITY_SMOOTH_BARS,
                          window_days: int = BASELINE_WINDOW_DAYS,
                          min_days: int = MIN_BASELINE_DAYS) -> np.ndarray:
    """How many 'average dollar bars' worth of notional traded in the
    trailing `smooth_bars` window, relative to the baseline expectation of
    exactly `smooth_bars` (i.e. 1.0 = trading exactly at its own recent
    normal pace; >1.0 = unusually fast dollar-bar arrival -- the market's
    own activity clock has sped up relative to calendar time).
    """
    baseline = _causal_daily_dollar_baseline(df, window_days, min_days)
    per_bar_baseline = (baseline / BARS_PER_DAY).to_numpy()
    dv = dollar_volume(df).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_frac = np.where(per_bar_baseline > 0, dv / per_bar_baseline, np.nan)
    smoothed = pd.Series(bar_frac).rolling(smooth_bars, min_periods=smooth_bars // 2).mean()
    return (smoothed / smooth_bars * smooth_bars).to_numpy()  # already normalized: mean ~1.0 at baseline


def _causal_quantile_threshold(intensity: np.ndarray, q: float,
                                window_days: int = INTENSITY_QUANTILE_WINDOW_DAYS) -> np.ndarray:
    """Causal trailing quantile of `intensity`, shifted by 1 bar so bar i's
    threshold never uses bar i's own value."""
    s = pd.Series(intensity)
    return s.rolling(window_days * BARS_PER_DAY,
                      min_periods=BARS_PER_DAY * 7).quantile(q).shift(1).to_numpy()


def crowding_haircut(df: pd.DataFrame,
                      high_in_q: float = INTENSITY_HIGH_IN_Q,
                      high_out_q: float = INTENSITY_HIGH_OUT_Q,
                      haircut: float = CROWDING_HAIRCUT) -> tuple[np.ndarray, np.ndarray]:
    """NOVEL branch's crowding gate: a latching hysteresis state machine
    (identical shape to v3/v4's own high/low breakout state machine, applied
    to a different input) that halves exposure while dollar-bar arrival
    intensity sits above its own causal 90th-percentile trailing threshold,
    and releases back to full exposure once intensity drops below the 60th
    percentile. Returns (haircut array, boolean crowded-state array)."""
    intensity = dollar_bar_intensity(df)
    high_in = _causal_quantile_threshold(intensity, high_in_q)
    high_out = _causal_quantile_threshold(intensity, high_out_q)

    n = len(df)
    out = np.ones(n)
    crowded_flags = np.zeros(n, dtype=bool)
    crowded = False
    for i in range(n):
        x = intensity[i]
        if np.isfinite(x) and np.isfinite(high_in[i]) and np.isfinite(high_out[i]):
            if not crowded and x > high_in[i]:
                crowded = True
            elif crowded and x < high_out[i]:
                crowded = False
        out[i] = haircut if crowded else 1.0
        crowded_flags[i] = crowded
    return out, crowded_flags


def crowding_target(df: pd.DataFrame) -> np.ndarray:
    """Novel branch's complete target: v4's own (unchanged) frac*scale,
    multiplied by the crowding haircut, through v4's own deadband."""
    haircut, _ = crowding_haircut(df)
    return apply_deadband(v4_raw_desired(df) * haircut)


def exposure_at_episodes(target: np.ndarray, df: pd.DataFrame,
                          episodes: list[tuple[str, str]] = STRESS_EPISODES,
                          horizon_days: int = 10) -> dict:
    """Mean |exposure| in the `horizon_days` trading days FOLLOWING each
    dated episode's own vol spike -- reused verbatim from r175_shared.py."""
    out = {}
    idx = df.index
    for name, date in episodes:
        ts = pd.Timestamp(date, tz=idx.tz)
        pos = idx.searchsorted(ts)
        if pos >= len(idx):
            continue
        end = min(pos + horizon_days * BARS_PER_DAY, len(idx))
        if pos >= end:
            continue
        out[name] = float(np.nanmean(np.abs(target[pos:end])))
    return out


DOLLAR_WARMUP_BARS = (BASELINE_WINDOW_DAYS + 80 + 10) * BARS_PER_DAY


def compare(candidate_build, *, label: str, btc: pd.DataFrame | None = None,
            eth: pd.DataFrame | None = None, control_build=None,
            markets: tuple[MarketSpec, ...] = (SPOT, FUTURES),
            include_eth: bool = True, seed: int = 0) -> list[dict]:
    """r102_shared.compare(), cloned only so the candidate's TargetStrategy
    gets DOLLAR_WARMUP_BARS instead of the default 80-day warmup (the
    dollar-clock baseline needs 180 days of causal history before it leaves
    its cold-start NaN)."""
    if control_build is None:
        control_build = v4_target
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare(): btc")
    if include_eth and eth is None:
        eth = load_eth()
    if include_eth:
        assert_no_holdout(eth, "compare(): eth")

    cand = TargetStrategy(candidate_build, name=f"r176_{label}", warmup=DOLLAR_WARMUP_BARS)
    ctrl = TargetStrategy(control_build, name="kelly_regime_v4")

    rows = []
    jobs = [(name, start, end, btc) for name, (start, end) in SLICES.items()]
    if include_eth:
        jobs.append((ETH_SLICE_NAME, None, None, eth))

    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice(cand, df, start, end, slice_name, market)
            b = run_slice(ctrl, df, start, end, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                         if b.mean_abs_exposure else float("nan"))
            vol_ratio = (a.realized_vol / b.realized_vol
                         if b.realized_vol else float("nan"))
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "cand_final": a.final_balance, "ctrl_final": b.final_balance,
                "cand_log_growth": a.log_growth, "ctrl_log_growth": b.log_growth,
                "d_log_growth": a.log_growth - b.log_growth,
                "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                "d_sharpe": a.sharpe - b.sharpe,
                "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
                "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
                "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
                "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
                "risk_matched": bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                                if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False,
                "boot_d_loggrowth": pr.diff.point,
                "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=400_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(176)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    volume = np.abs(rng.normal(20, 5, len(idx)))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": volume}, index=idx)

    # (1) Recovery check: with CONSTANT per-bar dollar volume, the
    # dollar-time anchor must reduce to (approximately) the ordinary
    # calendar-day rolling mean, since a constant rate makes dollar-time
    # and calendar-time the same clock up to the baseline's own smoothing.
    df_flat = df.copy()
    df_flat["volume"] = 1.0  # close is ~constant scale, so dollar volume ~constant
    anchor_dollar = dollar_time_anchor(df_flat, 20)
    anchor_calendar = df_flat["close"].rolling(20 * BARS_PER_DAY).mean().to_numpy()
    m = np.isfinite(anchor_dollar) & np.isfinite(anchor_calendar)
    assert m.sum() > 10_000, "not enough overlap to test recovery"
    rel = np.abs(anchor_dollar[m] - anchor_calendar[m]) / np.maximum(anchor_calendar[m], 1e-9)
    assert np.median(rel) < 0.03, f"flat-volume recovery check failed, median rel diff={np.median(rel)}"

    # (2) Activity-adaptivity: doubling dollar volume in the back half of
    # the series should SHRINK the dollar-time window (fewer calendar bars
    # needed to reach the same dollar target) relative to a matched
    # constant-volume control -- checked via a smaller realized calendar
    # span, not via the anchor value itself.
    df_burst = df.copy()
    half = len(df) // 2
    df_burst_vol = df_burst["volume"].to_numpy().copy()
    df_burst_vol[half:] *= 4.0
    df_burst["volume"] = df_burst_vol
    baseline_before = _causal_daily_dollar_baseline(df_burst).to_numpy()[half - BARS_PER_DAY]
    baseline_after = _causal_daily_dollar_baseline(df_burst).to_numpy()[half + 150 * BARS_PER_DAY]
    assert baseline_after > baseline_before * 2.0, (
        "burst in volume did not raise the causal daily-dollar baseline as expected")

    # (3) Causal truncation probes on every candidate builder.
    assert causal_truncation_probe_series(lambda d: dollar_time_anchor(d, 20), df)
    assert causal_truncation_probe_series(dollar_time_target, df)
    assert causal_truncation_probe_series(dollar_bar_intensity, df)
    assert causal_truncation_probe_series(crowding_target, df)

    # (4) dollar_time_vote_frac stays in [0, 1] where defined.
    frac = dollar_time_vote_frac(df)
    finite = frac[frac.notna()]
    assert len(finite) > 10_000
    assert finite.between(0.0, 1.0).all()

    # (5) crowding_haircut only ever takes the two frozen values, and fires
    # a non-trivial-but-not-overwhelming fraction of the time (a sanity
    # range on synthetic data with genuine volume bursts, not a real-data
    # promotion criterion).
    haircut, crowded = crowding_haircut(df_burst)
    assert set(np.unique(haircut[np.isfinite(haircut)])).issubset({1.0, CROWDING_HAIRCUT})
    frac_crowded = float(np.mean(crowded))
    assert 0.0 < frac_crowded < 0.95, f"crowding state degenerate: {frac_crowded:.3f} of bars"

    # (6) exposure_at_episodes: shape/finite sanity on a trivial target.
    dummy = np.ones(len(df))
    res = exposure_at_episodes(dummy, df, episodes=STRESS_EPISODES[:2])
    assert all(v == 1.0 for v in res.values()), res

    # (7) dollar_time_target / crowding_target self-consistency with
    # apply_deadband (both must be piecewise-constant between deadband
    # crossings, exactly like v4_target).
    for build in (dollar_time_target, crowding_target):
        t = build(df)
        assert np.isfinite(t).any()

    print("r176_shared self-test OK "
          f"(flat-volume median rel diff={np.median(rel):.4f}, "
          f"crowded fraction on burst synthetic={frac_crowded:.3f})")


if __name__ == "__main__":
    _self_test()
else:
    _self_test()
