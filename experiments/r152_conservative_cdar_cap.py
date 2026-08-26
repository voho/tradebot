"""R-152 CONSERVATIVE branch: dynamic CDaR-derived leverage cap for
``kelly_regime_v4``.

Executes the *conservative* half of the frozen pre-registration in
``experiments/r152_shared.py`` (read that file's module docstring for the
full pre-registration -- direction, literature, the shared ``rolling_cdar``
machinery, the falsification test, and the exact selection/promotion
rules; this file does not restate any of it beyond what is needed to run
the checks). This file does not edit ``r152_shared.py``.

**Mechanism.** ``kelly_regime_v4`` (via ``kelly_regime_v3``) sizes as
``frac[i] * min(target_vol / vol_or_slow[i], max_leverage)``, where
``max_leverage`` is the fixed constant ``2.0``. This branch replaces that
one constant with a per-bar array ``dynamic_cap[i]``, derived from a
rolling causal CDaR_0.95 of a REFERENCE return stream built from v4's own
UNMODIFIED vote-scaled signal, ``frac[i] * r[i]`` -- the raw vote times
the bar's own return, computed *before* any sizing/cap is applied, so
there is no circularity between the derived cap and the path used to
derive it. Everything else -- the vote, the three anchors, the 10%
hysteresis band, ``target_vol``, ``vol_span``, the deadband -- is
byte-identical to ``kelly_regime_v4``; only the cap differs. ``dynamic_cap``
is INVERSELY related to CDaR: worse trailing drawdowns tighten the cap,
calmer trailing drawdowns loosen it.

**Two disclosed implementation choices, stated plainly rather than buried:**

1. ``rolling_cdar``'s own docstring defines its input as "a 1-D array of
   per-bar simple returns" (Step 1 there is ``cumprod(1 + x)``, which only
   compounds correctly for simple returns). ``kelly_regime.py`` itself
   uses LOG returns internally (only ever fed to an EWM std, where the
   log/simple distinction barely matters at 5-minute granularity), but
   this branch's reference wealth PATH is a different object -- it must
   compound correctly bar-to-bar for CDaR's drawdown definition to be
   correct -- so ``reference_returns`` below builds ``r`` from
   ``close.pct_change()`` (simple returns), not ``np.log(close).diff()``.
2. The rescaling ("inner-train mean cap == 2.0") is a CALIBRATION step,
   not a live per-bar computation: ``calibrate_cap_scale`` is called
   ONCE, before any evaluation run, on a DataFrame slice that is itself
   truncated to inner-train (2017-01-01 -> 2020-12-31) by the caller, and
   returns a single float baked into the strategy instance as
   ``cap_scale`` -- the same status as v4's own ``target_vol=0.55`` /
   ``max_leverage=2.0``, which were themselves fit on historical data and
   then hardcoded. ``prepare()`` NEVER recomputes this scale from
   whatever ``df`` it is handed; doing so would silently look ahead for
   any bar earlier than the slice's own end (exactly the "a scaler,
   quantile, mean or std computed over the whole series and applied to
   early rows" lookahead class ``docs/ROUTINE.md`` names explicitly) and
   would additionally make inner-validation and holdout evaluations use a
   scale fit on data that had not occurred yet relative to inner-train
   itself. ``ConservativeCDaRCap.__init__`` raises if used without a
   ``cap_scale`` (see ``_dynamic_cap`` below) so this cannot happen
   silently.

Within ``calibrate_cap_scale``, the exact rescaling used is: let
``inv = 1 / cdar_ref`` over the rows of the inner-train slice where
``cdar_ref`` is finite and positive (i.e. the CDaR window has filled).
``cap_scale = target_mean_cap / mean(inv)``, so that
``dynamic_cap = cap_scale / cdar_ref`` has INNER-TRAIN mean EXACTLY
``target_mean_cap`` (2.0) over those same rows -- not merely
"``2.0 * cdar_ref_mean / cdar_ref``", which (Jensen's inequality, ``x >
0``) would land slightly *above* 2.0 on average. Rows where the CDaR
window has not yet filled (the first ``cdar_window_days`` of any run,
including the very start of inner-train) fall back to v4's own fixed
``max_leverage`` constant -- a placeholder for missing history, not a
peek at anything.

This branch is NOT ``@register``'d and stays in ``experiments/`` per
``docs/ROUTINE.md``.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r152_shared import (  # noqa: E402
    B2_CORRELATION_FLAG,
    CDAR_BETA,
    CDAR_WINDOW_DAYS_DEFAULT,
    CDAR_WINDOW_DAYS_SWEEP,
    DRAWDOWN_IMPROVEMENT_PP,
    EXPOSURE_MATCH_TOL_PP,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    rolling_cdar,
    window_bars,
)

FUTURES = MarketSpec.futures(leverage=5.0)


# ============================================================ (1) mechanism

def compute_vote_frac(close: pd.Series, horizons: tuple[int, ...], band: float) -> np.ndarray:
    """Reproduces KellyRegime/V3/V4.prepare()'s vote+hysteresis logic
    byte-identically (copied, not imported, so this file has no runtime
    dependency on the registered classes' internals beyond construction)."""
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def reference_returns(df: pd.DataFrame, frac: np.ndarray) -> np.ndarray:
    """v4's own unmodified vote-scaled return series: ``frac[i] * r[i]``,
    the raw signal before any sizing -- built from SIMPLE returns (see
    module docstring point 1)."""
    r = df["close"].pct_change().fillna(0.0).to_numpy()
    return frac * r


def calibrate_cap_scale(df_inner_train: pd.DataFrame, cdar_window_days: int, *,
                         target_mean_cap: float = 2.0, beta: float = CDAR_BETA,
                         horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01
                         ) -> float:
    """Fit ``cap_scale`` such that ``dynamic_cap = cap_scale / cdar_ref``
    has an exact inner-train mean of ``target_mean_cap`` (2.0), over rows
    where ``cdar_ref`` is defined. ``df_inner_train`` must already be
    truncated by the caller to the inner-train slice -- this function does
    not slice by date itself, so it cannot silently be handed more than
    that (see module docstring point 2)."""
    close = df_inner_train["close"]
    frac = compute_vote_frac(close, horizons, band)
    ref = reference_returns(df_inner_train, frac)
    wb = window_bars(cdar_window_days)
    cdar_ref = rolling_cdar(ref, wb, beta=beta)
    valid = np.isfinite(cdar_ref) & (cdar_ref > 0)
    if not valid.any():
        raise ValueError(
            f"calibrate_cap_scale: no valid CDaR values in the given slice "
            f"(window_bars={wb} vs {len(df_inner_train)} bars supplied) -- "
            "the CDaR window is longer than the calibration slice.")
    mean_inv = float(np.mean(1.0 / cdar_ref[valid]))
    return float(target_mean_cap / mean_inv)


class ConservativeCDaRCap(KellyRegimeV4):
    """``kelly_regime_v4`` with its fixed ``max_leverage=2.0`` replaced by
    a dynamic, CDaR-derived cap. See module docstring for the full
    mechanism. ``cap_scale`` MUST be supplied (via ``calibrate_cap_scale``
    on an inner-train-only slice) -- there is no live-recalibrating
    default, by design (module docstring point 2)."""

    name = "r152_conservative_cdar_cap"

    def __init__(self, cdar_window_days: int = CDAR_WINDOW_DAYS_DEFAULT,
                 cap_scale: float | None = None, cdar_beta: float = CDAR_BETA,
                 cap_floor: float = 0.05, cap_ceiling: float = 50.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.cdar_window_days = cdar_window_days
        self.cdar_beta = cdar_beta
        self.cap_floor = cap_floor
        self.cap_ceiling = cap_ceiling
        self.cap_scale = cap_scale

    def _dynamic_cap(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (dynamic_cap, cdar_ref) -- cdar_ref (pre-fallback, NaN
        where the window has not filled) is exposed separately so the B2
        diagnostic can restrict its correlation to bars where the
        mechanism is actually engaged, rather than the fallback plateau."""
        if self.cap_scale is None:
            raise ValueError(
                "ConservativeCDaRCap.cap_scale is not set. Calibrate it once via "
                "calibrate_cap_scale() on an inner-train-only slice and pass the "
                "result to the constructor -- prepare() deliberately never "
                "calibrates itself against whatever df it happens to receive "
                "(see module docstring).")
        close = df["close"]
        frac = compute_vote_frac(close, self.horizons, self.band)
        ref = reference_returns(df, frac)
        wb = window_bars(self.cdar_window_days)
        cdar_ref = rolling_cdar(ref, wb, beta=self.cdar_beta)
        with np.errstate(divide="ignore", invalid="ignore"):
            cap = self.cap_scale / cdar_ref
        # Fallback for rows where the CDaR window has not filled (or, in
        # principle, an exact-zero CDaR): v4's own fixed constant. This is
        # a placeholder for MISSING history, not a peek at anything -- it
        # uses no information beyond bar i's own (as-yet-incomplete)
        # rolling window.
        cap = np.where(np.isfinite(cap) & (cap > 0), cap, self.max_leverage)
        cap = np.clip(cap, self.cap_floor, self.cap_ceiling)
        return cap, cdar_ref

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()  # v4's own vol/vote input -- unchanged

        frac = compute_vote_frac(close, self.horizons, self.band)
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())

        dynamic_cap, cdar_ref = self._dynamic_cap(df)  # ONLY thing that differs from v4

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, dynamic_cap)
            steady = np.minimum(self.target_vol / slow, dynamic_cap)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["cdar_dynamic_cap"] = dynamic_cap
        df["cdar_ref"] = cdar_ref  # pre-fallback, NaN where window not yet filled
        df["vol"] = vol  # v4's own realized-vol column, exposed for the B2 diagnostic
        return df


# ============================================================ (2) causal check

def causal_truncation_probe(build_fn, df: pd.DataFrame,
                             cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    """No-lookahead probe (same recipe as r89/r148/r150's shared
    machinery): truncating the frame at k must not change any value up to
    k, and perturbing everything at/after k must not change any value
    before k."""
    full = np.asarray(build_fn(df), dtype=float)
    for cut in cuts:
        k = int(len(df) * cut)
        if k < BARS_PER_DAY * 2:
            continue
        part = np.asarray(build_fn(df.iloc[:k]), dtype=float)
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-8, rtol=1e-7):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-8, rtol=1e-7)))
            raise AssertionError(f"{build_fn.__name__} causality FAIL at cut={cut}: {bad} bars differ")
        perturbed = df.copy()
        tail = perturbed.iloc[k:].copy()
        for col in ("open", "high", "low", "close"):
            if col in tail.columns:
                tail[col] = tail[col] * 3.7 + 1.0
        perturbed.iloc[k:] = tail
        pert = np.asarray(build_fn(perturbed), dtype=float)
        pm = np.isfinite(a) & np.isfinite(pert[:k])
        if not np.allclose(a[pm], pert[:k][pm], atol=1e-8, rtol=1e-7):
            raise AssertionError(f"{build_fn.__name__} peeks at bar>=k, cut={cut}")
    return True


# ============================================================ (3) smoke test

def _smoke_test() -> dict:
    """Synthetic series, per ROUTINE.md step 2 / R-89's own lesson (a 200x
    silent divergence there was caught only by exactly this kind of check
    before real data was touched): verify the cap tightens after a
    synthetic drawdown and loosens once the strategy's own vote-scaled
    path has calmed back down, and that calibration lands the mean cap at
    2.0. Returns a dict of the measured values so the caller can print
    them.

    Uses v4's REAL horizons/band (20/40/80 days, 1%) rather than toy
    values, because the vote's own hysteresis genuinely participates in
    the mechanism (the reference stream is frac[i]*r[i], not r[i] alone)
    -- a smoke test that fakes the vote away would not exercise the same
    code path real data runs. The window/measurement boundaries below were
    picked by inspecting the actual frac/CDaR trace of this synthetic
    series (not tuned against the pass/fail outcome): a ~15% correction
    partially de-risks the vote (frac dips from 1.0 towards 0, exactly the
    hysteresis kelly_regime_v3/v4 already has), which briefly flatlines
    the REFERENCE wealth path once frac hits exactly 0 -- correctly making
    trailing CDaR (and so the cap) degenerate toward the no-drawdown
    fallback once the window no longer contains any nonzero reference
    return at all, a real property of "CDaR of what v4 would have earned
    given its own vote", not a bug. The stress/calm2 windows below are
    chosen to sit in the region where the mechanism is actually engaged
    (frac > 0 somewhere in the trailing window), not that degenerate tail.
    """
    n = 60_000  # ~208 days of 5-minute bars
    idx = pd.date_range("2017-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(152)

    drift = np.log(1.5) / n  # gentle uptrend over the whole series (+50%)
    innov = rng.normal(0, 0.0004, n) + drift
    stress_start, stress_end = 30_000, 30_600  # a ~15% correction, ~2 days
    innov[stress_start:stress_end] -= 0.00028
    close = 10_000 * np.exp(np.cumsum(innov))
    df = pd.DataFrame({"open": close, "high": close * 1.0006, "low": close * 0.9994,
                        "close": close, "volume": 1.0}, index=idx)

    cdar_window_days = 10
    horizons = (20, 40, 80)  # v4's own real anchors
    band = 0.01  # v4's own real band

    cap_scale = calibrate_cap_scale(df, cdar_window_days, beta=CDAR_BETA,
                                     horizons=horizons, band=band)
    strat = ConservativeCDaRCap(horizons=horizons, band=band,
                                 cdar_window_days=cdar_window_days,
                                 cap_scale=cap_scale)
    prepared = strat.prepare(df.copy())
    cap = prepared["cdar_dynamic_cap"].to_numpy()
    cdar_ref = prepared["cdar_ref"].to_numpy()

    calm1 = slice(24_000, 29_500)   # established calm, before the correction
    stress = slice(30_700, 33_100)  # the correction + its immediate CDaR echo
    calm2 = slice(42_000, 50_000)   # after the vote and price have both recovered

    mean_cdar_calm1 = float(np.nanmean(cdar_ref[calm1]))
    mean_cdar_stress = float(np.nanmean(cdar_ref[stress]))
    mean_cdar_calm2 = float(np.nanmean(cdar_ref[calm2]))
    mean_cap_calm1 = float(np.nanmean(cap[calm1]))
    mean_cap_stress = float(np.nanmean(cap[stress]))
    mean_cap_calm2 = float(np.nanmean(cap[calm2]))

    valid = np.isfinite(cdar_ref) & (cdar_ref > 0)
    mean_cap_over_valid_rows = float(np.mean(cap[valid])) if valid.any() else float("nan")

    cdar_rises_in_stress = bool(mean_cdar_stress > mean_cdar_calm1)
    cap_tightens_in_stress = bool(mean_cap_stress < mean_cap_calm1)
    cap_loosens_after_stress = bool(mean_cap_calm2 > mean_cap_stress)
    calibration_ok = bool(abs(mean_cap_over_valid_rows - 2.0) < 0.1)

    # causal truncation probe on the exact mechanism used live
    def _build(frame: pd.DataFrame) -> np.ndarray:
        return strat._dynamic_cap(frame)[0]
    causal_ok = causal_truncation_probe(_build, df, cuts=(0.4, 0.6, 0.85))

    return dict(
        cdar_window_days=cdar_window_days,
        mean_cdar_calm1=mean_cdar_calm1, mean_cdar_stress=mean_cdar_stress,
        mean_cdar_calm2=mean_cdar_calm2,
        mean_cap_calm1=mean_cap_calm1, mean_cap_stress=mean_cap_stress,
        mean_cap_calm2=mean_cap_calm2,
        cap_scale=cap_scale, mean_cap_over_valid_rows=mean_cap_over_valid_rows,
        cdar_rises_in_stress=cdar_rises_in_stress,
        cap_tightens_in_stress=cap_tightens_in_stress,
        cap_loosens_after_stress=cap_loosens_after_stress,
        calibration_mean_within_0_1_of_2=calibration_ok, causal_probe_ok=causal_ok,
        pass_=bool(cdar_rises_in_stress and cap_tightens_in_stress and cap_loosens_after_stress
                    and calibration_ok and causal_ok),
    )


# ============================================================ (4) real-data runner

@dataclass
class SliceMetrics:
    label: str
    final_balance: float
    sharpe: float
    max_drawdown_pct: float
    time_in_market_pct: float
    mean_abs_exposure: float


def run_slice(strategy, df: pd.DataFrame, start, end, market: MarketSpec,
              label: str, balance: float = 1_000.0) -> SliceMetrics:
    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    m = compute_metrics(res)
    exposure = res.df["target"].to_numpy() if "target" in res.df.columns else np.array([np.nan])
    return SliceMetrics(
        label=label, final_balance=m.final_balance, sharpe=m.sharpe,
        max_drawdown_pct=m.max_drawdown_pct, time_in_market_pct=m.time_in_market_pct,
        mean_abs_exposure=float(np.nanmean(np.abs(exposure))),
    )


def b2_correlation(df_inner_train_slice_result_df: pd.DataFrame) -> tuple[float, int]:
    """Pearson r between the dynamic cap and v4's own vol column, on
    inner-train, restricted to rows where the CDaR mechanism is actually
    engaged (cdar_ref finite) -- see module docstring on why the fallback
    plateau is excluded."""
    d = df_inner_train_slice_result_df
    cap = d["cdar_dynamic_cap"].to_numpy()
    ref = d["cdar_ref"].to_numpy()
    vol = d["vol"].to_numpy()
    mask = np.isfinite(ref) & np.isfinite(vol) & np.isfinite(cap)
    n = int(mask.sum())
    if n < 10:
        return float("nan"), n
    r = float(np.corrcoef(cap[mask], vol[mask])[0, 1])
    return r, n


def run_pytest(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest"] + args,
                           cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout[-4000:] + proc.stderr[-4000:]


if __name__ == "__main__":
    print("=" * 100)
    print("R-152 CONSERVATIVE branch: dynamic CDaR leverage cap on kelly_regime_v4")
    print("=" * 100)

    # Per r152_shared.py's declared count: "3 window lengths x 1 CDaR beta x
    # {inner-train fit, inner-validation score} = 3 scored configurations,
    # plus 1 control re-run = 4 per branch". Each window length is ONE
    # configuration (its inner-train fit/B2 measurement and its
    # inner-validation score are the two measurements taken FOR that one
    # configuration, not two separate configurations). The control counts
    # once, not once per window, even though it is re-run per window below
    # for a fresh measurement each time.
    configs_evaluated = 0
    control_run_once = False
    extra_context_runs = 0  # runs beyond the declared 4, disclosed separately

    # ---------------------------------------------------------------- smoke test
    print("\n--- Step 3: smoke test on synthetic data ---")
    smoke = _smoke_test()
    for k, v in smoke.items():
        print(f"  {k:30s} = {v}")
    print(f"\nSMOKE TEST {'PASS' if smoke['pass_'] else 'FAIL'}")
    if not smoke["pass_"]:
        print("*** Smoke test failed -- stopping before touching real data. ***")
        sys.exit(1)

    # ---------------------------------------------------------------- real data
    print("\n--- Loading real BTC dataset ---")
    DF, LABEL = load_dataset(ROOT / "data", "spot")
    print(f"{len(DF):,} bars  {DF.index[0]} -> {DF.index[-1]}  (data: {LABEL})")

    inner_train_slice = DF[DF.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")]
    print(f"inner-train slice for calibration: {len(inner_train_slice):,} bars, "
          f"{inner_train_slice.index[0]} -> {inner_train_slice.index[-1]}")

    control = KellyRegimeV4()

    rows = []
    b2_results = {}
    cap_scales = {}

    for w in CDAR_WINDOW_DAYS_SWEEP:
        print(f"\n{'=' * 60}\nwindow_days = {w}\n{'=' * 60}")
        cap_scale = calibrate_cap_scale(inner_train_slice, w)
        cap_scales[w] = cap_scale
        print(f"calibrated cap_scale (inner-train mean cap -> 2.0) = {cap_scale:.6f}")

        candidate = ConservativeCDaRCap(cdar_window_days=w, cap_scale=cap_scale)

        # ---- Diagnostic B2 (inner-train), run first, reported regardless ----
        res_train_for_b2 = run_period(candidate, DF, None, INNER_TRAIN_END,
                                       market=FUTURES, start_balance=1_000.0)
        r2, n_valid = b2_correlation(res_train_for_b2.df)
        b2_flag = bool(np.isfinite(r2) and abs(r2) >= B2_CORRELATION_FLAG)
        b2_results[w] = (r2, n_valid, b2_flag)
        print(f"B2: Pearson r(dynamic_cap, v4 vol), inner-train, "
              f"n={n_valid} engaged bars: r={r2:+.4f}  "
              f"{'FLAGGED (|r|>=0.85)' if b2_flag else 'not flagged'}")

        # ---- inner-validation comparison ----
        cand_val = run_slice(candidate, DF, INNER_VAL_START, INNER_VAL_END, FUTURES,
                              label=f"cons_w{w}")
        ctrl_val = run_slice(control, DF, INNER_VAL_START, INNER_VAL_END, FUTURES,
                              label="v4_control")
        configs_evaluated += 1  # this window length: 1 scored configuration
                                  # (its inner-train B2 fit + its inner-validation
                                  # score together are the two measurements for it)
        control_run_once = True  # the control itself counts once, not per window

        d_sharpe = cand_val.sharpe - ctrl_val.sharpe
        d_dd = cand_val.max_drawdown_pct - ctrl_val.max_drawdown_pct  # negative = improved
        exposure_gap_pp = abs(cand_val.mean_abs_exposure - ctrl_val.mean_abs_exposure) * 100.0
        tim_gap_pp = abs(cand_val.time_in_market_pct - ctrl_val.time_in_market_pct)

        exposure_matched = (exposure_gap_pp <= EXPOSURE_MATCH_TOL_PP
                             and tim_gap_pp <= EXPOSURE_MATCH_TOL_PP)
        sharpe_or_dd_ok = (d_sharpe >= -SHARPE_NOISE_FLOOR) or (-d_dd >= DRAWDOWN_IMPROVEMENT_PP)

        print(f"\ninner-validation ({INNER_VAL_START} -> {INNER_VAL_END}, futures 5x):")
        print(f"  {'':14s} {'sharpe':>8s} {'maxDD%':>8s} {'TiM%':>7s} {'meanExp':>8s} {'final$':>12s}")
        print(f"  {'candidate':14s} {cand_val.sharpe:8.3f} {cand_val.max_drawdown_pct:8.2f} "
              f"{cand_val.time_in_market_pct:7.2f} {cand_val.mean_abs_exposure:8.3f} "
              f"{cand_val.final_balance:12,.2f}")
        print(f"  {'control(v4)':14s} {ctrl_val.sharpe:8.3f} {ctrl_val.max_drawdown_pct:8.2f} "
              f"{ctrl_val.time_in_market_pct:7.2f} {ctrl_val.mean_abs_exposure:8.3f} "
              f"{ctrl_val.final_balance:12,.2f}")
        print(f"  d_sharpe={d_sharpe:+.4f}  d_maxDD_pp={d_dd:+.2f}  "
              f"exposure_gap_pp={exposure_gap_pp:.2f}  tim_gap_pp={tim_gap_pp:.2f}")
        print(f"  criterion(1) exposure matched (<=15pp both): {exposure_matched}")
        print(f"  criterion(2) sharpe within noise floor OR DD improves>=3pp: {sharpe_or_dd_ok}")

        # ---- training-period comparison, for context ----
        cand_train = run_slice(candidate, DF, None, INNER_TRAIN_END, FUTURES,
                                label=f"cons_w{w}_train")
        ctrl_train = run_slice(control, DF, None, INNER_TRAIN_END, FUTURES,
                                label="v4_control_train")
        extra_context_runs += 2  # cand_train + ctrl_train: reported per step 5,
                                  # not part of the frozen "4 per branch" count
        d_sharpe_train = cand_train.sharpe - ctrl_train.sharpe
        d_dd_train = cand_train.max_drawdown_pct - ctrl_train.max_drawdown_pct
        print(f"\ntraining period (start -> {INNER_TRAIN_END}, futures 5x, context only):")
        print(f"  {'':14s} {'sharpe':>8s} {'maxDD%':>8s} {'TiM%':>7s} {'meanExp':>8s} {'final$':>12s}")
        print(f"  {'candidate':14s} {cand_train.sharpe:8.3f} {cand_train.max_drawdown_pct:8.2f} "
              f"{cand_train.time_in_market_pct:7.2f} {cand_train.mean_abs_exposure:8.3f} "
              f"{cand_train.final_balance:12,.2f}")
        print(f"  {'control(v4)':14s} {ctrl_train.sharpe:8.3f} {ctrl_train.max_drawdown_pct:8.2f} "
              f"{ctrl_train.time_in_market_pct:7.2f} {ctrl_train.mean_abs_exposure:8.3f} "
              f"{ctrl_train.final_balance:12,.2f}")
        print(f"  d_sharpe_train={d_sharpe_train:+.4f}  d_maxDD_train_pp={d_dd_train:+.2f}")

        rows.append(dict(
            window_days=w, cap_scale=cap_scale, r2=r2, n_valid=n_valid, b2_flag=b2_flag,
            cand_val=cand_val, ctrl_val=ctrl_val, d_sharpe=d_sharpe, d_dd=d_dd,
            exposure_gap_pp=exposure_gap_pp, tim_gap_pp=tim_gap_pp,
            exposure_matched=exposure_matched, sharpe_or_dd_ok=sharpe_or_dd_ok,
            cand_train=cand_train, ctrl_train=ctrl_train,
            d_sharpe_train=d_sharpe_train, d_dd_train=d_dd_train,
        ))

    if control_run_once:
        configs_evaluated += 1  # + 1 control re-run, per the frozen declaration

    # ---------------------------------------------------------------- selection rule
    print(f"\n{'=' * 100}\nSELECTION RULE (frozen, inner-validation, vs kelly_regime_v4 control)\n{'=' * 100}")

    crit1 = all(r["exposure_matched"] for r in rows)
    crit2 = all(r["sharpe_or_dd_ok"] for r in rows)
    signs = [1 if r["d_sharpe"] > 0 else (-1 if r["d_sharpe"] < 0 else 0) for r in rows]
    agree_count = max(signs.count(1), signs.count(-1))
    crit3 = agree_count >= 2

    print(f"criterion (1) exposure/TiM matched within {EXPOSURE_MATCH_TOL_PP}pp, all {len(rows)} windows: {crit1}")
    for r in rows:
        print(f"    w={r['window_days']:>3d}: exposure_gap_pp={r['exposure_gap_pp']:.2f}  "
              f"tim_gap_pp={r['tim_gap_pp']:.2f}  matched={r['exposure_matched']}")
    print(f"criterion (2) Sharpe within +/-{SHARPE_NOISE_FLOOR} noise floor OR "
          f"DD improves >={DRAWDOWN_IMPROVEMENT_PP}pp at matched exposure, all windows: {crit2}")
    for r in rows:
        print(f"    w={r['window_days']:>3d}: d_sharpe={r['d_sharpe']:+.4f}  "
              f"d_maxDD_pp={r['d_dd']:+.2f}  ok={r['sharpe_or_dd_ok']}")
    print(f"criterion (3) plateau (>=2 of 3 windows agree in sign of d_sharpe): "
          f"signs={signs}  agree_count={agree_count}  pass={crit3}")

    eligible = crit1 and crit2 and crit3
    print(f"\nSELECTION RULE OUTCOME: {'ELIGIBLE FOR HOLDOUT' if eligible else 'NEGATIVE at inner-validation'}")

    print("\nB2 diagnostic summary (reported regardless of the outcome above):")
    for w, (r2, n_valid, flag) in b2_results.items():
        print(f"  w={w:>3d}: r={r2:+.4f}  n={n_valid}  flagged={flag}")
    any_b2_flag = any(f for _r, _n, f in b2_results.values())
    print(f"any window flagged (|r|>=0.85): {any_b2_flag}")

    # ---------------------------------------------------------------- pytest
    print(f"\n{'=' * 100}\npytest -k causality (not auto-run for this unregistered experiments/ file, "
          "run here anyway per the task's discipline)\n" + "=" * 100)
    rc_c, out_c = run_pytest(["-q", "-k", "causality"])
    print(out_c)
    print(f"returncode={rc_c}")

    print(f"\n{'=' * 100}\npytest -q (full repo suite)\n{'=' * 100}")
    rc_full, out_full = run_pytest(["-q"])
    print(out_full)
    print(f"returncode={rc_full}")

    print(f"\n{'=' * 100}\nCONFIGS EVALUATED\n{'=' * 100}")
    print(f"Selection-stage configs (per r152_shared.py's frozen count, '4 per branch'): "
          f"{configs_evaluated}  (3 window-length configs, each scored via its own "
          f"{{inner-train B2 fit, inner-validation score}} measurement pair, + 1 control re-run)")
    print(f"Additional context/debug runs beyond the declared 4 (NOT part of the selection-stage "
          f"count): {extra_context_runs}  -- the training-period (step-5 'for context') comparison, "
          f"run for the candidate and the control at each of the 3 window lengths, requested by the "
          f"task in addition to the frozen selection-stage rule; disclosed here rather than folded "
          f"into the 4.")
    print("=" * 100)
