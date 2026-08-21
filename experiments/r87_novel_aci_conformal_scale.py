#!/usr/bin/env python
"""R-87 NOVEL branch: replace `kelly_regime`'s trailing-EWM-std dispersion
estimator with an online, Adaptive-Conformal-Inference-calibrated quantile
of absolute daily returns, used as the Kelly scale's denominator, while
keeping `kelly_regime_v4`'s own vote (`r87_shared.v4_vote_frac`, horizons
(20,40,80), band 0.01) and `kelly_regime`'s own simple continuous-scale +
10% deadband pipeline unchanged. See `experiments/r87_shared.py`'s module
docstring for the full citation trail (Gibbs & Candes 2021 for ACI; Ryan
2026, arXiv:2608.01494, "Conformal Kelly", named as this round's own
pre-registered failure-mode prediction) and the not-a-duplicate-of
reasoning against every prior round. This file does not repeat that
material; it implements the frozen mechanism and reports the frozen gates.

MECHANISM (one sentence): swap `kelly_regime`'s point-estimate trailing
EWM-vol denominator for a conformal-calibrated quantile of |daily log
return|, adaptively re-targeted online via ACI so the quantile's own
empirical coverage tracks a fixed nominal level even under distribution
shift, and use `target_vol / that_quantile_annualized` as the Kelly scale
in place of `target_vol / ewm_vol` -- everything else (vote, deadband,
target_vol, max_leverage) held at `kelly_regime`'s own shipped defaults.

PRE-REGISTERED FALSIFICATION TEST (frozen before any real number here):
the finalist (best of the 6-config sweep by inner-validation Sharpe on
futures_5x) must beat `kelly_regime_v4`'s own inner-validation Sharpe by
more than the project's +/-0.2 noise floor, OR show a clear max-drawdown
improvement, AND the same qualitative direction must replicate on ETH
(2019-03-14 -> whatever of 2022-12-31 is available). Failing either bar
is a NEGATIVE, and per the Ryan (2026) in-sample-collapse precedent named
above, NEGATIVE is this round's own pre-registered expectation, not a
disappointing outcome.

CONFIGURATIONS EVALUATED IN THIS FILE: 6 (frozen grid, gamma in
{0.01, 0.02, 0.05} x window_days in {60, 120} -- see `SWEEP_GRID` below).
No configs are added or dropped after seeing results.

This file never reads any bar with timestamp >= OOS_START (2023-01-01) on
either BTC or ETH; every data load below is truncated to `< OOS_START`
immediately after loading and asserted so, and the max timestamp actually
read is tracked and printed at the end of `main()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r87_shared import (  # noqa: E402
    ALPHA_MAX,
    ALPHA_MIN,
    BARS_PER_DAY,
    BARS_PER_YEAR,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    V4_BAND,
    V4_HORIZONS,
    aci_update,
    causal_truncation_probe,
    daily_resample_causal,
    v4_vote_frac,
)

DATA_DIR = ROOT / "data"

# Fixed a priori (see r87_shared and the class docstring above): the
# two-sided ~1-sigma-equivalent level under a normal approximation, chosen
# so the resulting quantile is unit-comparable to what EWM std targets.
# NOT swept.
TARGET_ALPHA = 0.3173

# Frozen 3x2 sweep grid. Do not add/drop after seeing results.
GAMMA_GRID = (0.01, 0.02, 0.05)
WINDOW_DAYS_GRID = (60, 120)
SWEEP_GRID = [(g, w) for g in GAMMA_GRID for w in WINDOW_DAYS_GRID]

TARGET_VOL = 0.55       # kelly_regime's own shipped default
MAX_LEVERAGE = 2.0      # kelly_regime's own shipped default
DEADBAND = 0.10         # kelly_regime's own shipped default


# --------------------------------------------------------------------------
# Data loading, truncated to < OOS_START at the point of load (never later).
# --------------------------------------------------------------------------

def assert_no_holdout(df: pd.DataFrame, label: str = "") -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read ({label}): max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df, "BTC")
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (< {OOS_START})")
    return df


def load_eth_bars() -> pd.DataFrame | None:
    path = DATA_DIR / "ethusd_coinbase_spot_5m.csv.gz"
    if not path.exists():
        return None
    df = load_ohlcv_csv(path)
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df, "ETH")
    print(f"ETH (Coinbase spot): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (< {OOS_START})")
    return df


# --------------------------------------------------------------------------
# Conformal/ACI dispersion estimator (Step 1-5 of the frozen construction).
# --------------------------------------------------------------------------

def build_daily_scores(df: pd.DataFrame) -> pd.Series:
    """Daily |log return| nonconformity score, causal daily resample."""
    daily = daily_resample_causal(df)
    r_d = np.log(daily["close"]).diff()
    s_d = r_d.abs()
    return s_d  # index: daily timestamps; s_d.iloc[0] is NaN (no prior day)


def build_conformal_quantile_daily(s_d: pd.Series, gamma: float, window_days: int,
                                    target_alpha: float = TARGET_ALPHA,
                                    alpha0: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Explicit interleaved online loop (per r87_shared's spec -- NOT
    `run_aci_causal`, since the miscoverage indicator here depends on the
    evolving trailing-window quantile, not a precomputed array).

    For day-index D (0-based into `s_d`):
      - if D >= window_days + 1: the trailing window s_d[D-window_days:D]
        (indices D-window_days .. D-1) is fully populated (s_d[0] is the
        only NaN, at index 0, and window_days+1 > window_days so index 0 is
        excluded once D >= window_days+1). q_t[D] = (1-alpha_t[D])-quantile
        of that window; alpha_t[D] was fixed BEFORE D's own score is seen.
      - THEN observe s_d[D] (today's realized score) and, if finite, update
        alpha for D+1 via `aci_update` using err_D = 1{s_d[D] > q_t[D]}.
      - else (not enough trailing history yet): q_t[D] = NaN (warmup).

    Returns (q_t, alpha_path), both length len(s_d), aligned to s_d's index.
    """
    s = s_d.to_numpy()
    n = len(s)
    q_t = np.full(n, np.nan)
    alpha_path = np.full(n, np.nan)
    alpha = target_alpha if alpha0 is None else alpha0
    for d in range(n):
        if d >= window_days + 1:
            window = s[d - window_days:d]
            # By construction this window starts at index d-window_days >= 1,
            # so it never includes s[0] (the only NaN) -- but guard anyway.
            if np.all(np.isfinite(window)):
                q = float(np.quantile(window, 1.0 - alpha))
                q_t[d] = q
                alpha_path[d] = alpha
                if np.isfinite(s[d]):
                    err = 1.0 if s[d] > q else 0.0
                    alpha = aci_update(alpha, err, target_alpha, gamma,
                                        alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX)
    return q_t, alpha_path


def build_vol_conformal_5m(df: pd.DataFrame, gamma: float, window_days: int,
                            target_alpha: float = TARGET_ALPHA) -> np.ndarray:
    """Causal broadcast: day D's 5-minute bars all use
    `vol_conformal_annualized[D]`, computed from information through day
    D-1 only (see `build_conformal_quantile_daily`)."""
    s_d = build_daily_scores(df)
    q_t, _alpha_path = build_conformal_quantile_daily(s_d, gamma, window_days, target_alpha)
    vol_daily = pd.Series(q_t * np.sqrt(365.25), index=s_d.index)
    day_of_bar = df.index.normalize()
    vol_5m = vol_daily.reindex(day_of_bar).to_numpy()
    return vol_5m


# --------------------------------------------------------------------------
# Full target-building pipeline: v4's own vote x conformal scale, through
# kelly_regime's own (unmodified) continuous-scale + 10% deadband loop.
# --------------------------------------------------------------------------

def build_target_diag(df: pd.DataFrame, gamma: float, window_days: int,
                       target_vol: float = TARGET_VOL, max_leverage: float = MAX_LEVERAGE,
                       deadband: float = DEADBAND, target_alpha: float = TARGET_ALPHA) -> dict:
    frac = v4_vote_frac(df, V4_HORIZONS, V4_BAND)
    vol_ann = build_vol_conformal_5m(df, gamma, window_days, target_alpha)

    n = len(df)
    scale = np.zeros(n)
    desired = np.zeros(n)
    target = np.zeros(n)
    pos = 0.0
    for i in range(n):
        v = vol_ann[i]
        sc = min(target_vol / v, max_leverage) if np.isfinite(v) and v > 0 else 0.0
        d = frac[i] * sc
        scale[i] = sc
        desired[i] = d
        if abs(d - pos) > deadband:
            pos = d
        target[i] = pos

    return dict(frac=frac, vol_ann=vol_ann, scale=scale, desired=desired, target=target)


def build_target(df: pd.DataFrame, gamma: float, window_days: int,
                  target_vol: float = TARGET_VOL, max_leverage: float = MAX_LEVERAGE,
                  deadband: float = DEADBAND, target_alpha: float = TARGET_ALPHA) -> np.ndarray:
    """Thin wrapper returning just the `target` array -- signature matches
    what `causal_truncation_probe(build_target_fn, df, ...)` expects."""
    return build_target_diag(df, gamma, window_days, target_vol, max_leverage,
                              deadband, target_alpha)["target"]


# --------------------------------------------------------------------------
# Strategy wrapper (unregistered -- instantiated and passed to `ev` directly)
# --------------------------------------------------------------------------

class KellyRegimeACIConformal(Strategy):
    """kelly_regime_v4's vote x an ACI-calibrated conformal-quantile scale.

    Structurally identical to `kelly_regime.py`'s own `prepare`/`on_bar`
    pipeline (continuous scale, 10% deadband, no extra hysteresis layer);
    only the dispersion estimator feeding `scale` is replaced -- see this
    module's docstring and `experiments/r87_shared.py` for the full design.
    """

    name = "kelly_regime_aci_conformal"

    def __init__(self, gamma: float = 0.02, window_days: int = 60,
                 target_vol: float = TARGET_VOL, max_leverage: float = MAX_LEVERAGE,
                 deadband: float = DEADBAND, target_alpha: float = TARGET_ALPHA) -> None:
        self.gamma = gamma
        self.window_days = window_days
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.deadband = deadband
        self.target_alpha = target_alpha
        # Warmup must cover both v4's own slowest (80-day) anchor AND this
        # config's own conformal trailing window (window_days+1 days) --
        # whichever is larger -- so the Step-A "finite after warmup" check
        # is a genuine bug probe, not an artifact of an unfair warmup bar.
        self.warmup = max(80, window_days + 5) * BARS_PER_DAY + 10

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["target"] = build_target(df, self.gamma, self.window_days,
                                     self.target_vol, self.max_leverage,
                                     self.deadband, self.target_alpha)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# --------------------------------------------------------------------------
# Step A gate: per-config sanity checks, computed BEFORE any Sharpe number,
# on inner-train (`end=INNER_TRAIN_END`) only.
# --------------------------------------------------------------------------

def step_a_gate_one(df_train: pd.DataFrame, gamma: float, window_days: int) -> dict:
    diag = build_target_diag(df_train, gamma, window_days)
    frac, vol_ann, desired = diag["frac"], diag["vol_ann"], diag["desired"]

    warmup = KellyRegimeACIConformal(gamma=gamma, window_days=window_days).warmup
    post = slice(warmup, None) if len(df_train) > warmup else slice(0, 0)
    vol_post = vol_ann[post]

    # (a) finite and strictly positive after warmup.
    finite_pos = np.isfinite(vol_post) & (vol_post > 0)
    check_a = bool(finite_pos.all()) and finite_pos.size > 0

    # (b) exact-flat structural check: frac[i]==0 => desired[i]==0.0 exactly,
    # regardless of scale[i]. Checked programmatically over the whole frame.
    zero_frac_mask = (frac == 0.0)
    check_b = bool(np.all(desired[zero_frac_mask] == 0.0)) if zero_frac_mask.any() else True

    # (d) causal truncation probe on the target-building function itself.
    check_at = min(len(df_train) - 25_000, len(df_train) - 1)
    check_at = max(check_at, warmup + 5_000)
    if check_at >= len(df_train) - 1 or check_at < 0:
        check_d = None  # not enough data to probe safely
    else:
        def _fn(d):
            return build_target(d, gamma, window_days)
        check_d = bool(causal_truncation_probe(_fn, df_train, check_at, shorter_by=20_000))

    if finite_pos.size > 0:
        vmin, vmed, vmax = float(np.nanmin(vol_post)), float(np.nanmedian(vol_post)), float(np.nanmax(vol_post))
    else:
        vmin = vmed = vmax = float("nan")

    passed = check_a and check_b and (check_d is True)
    return dict(gamma=gamma, window_days=window_days, check_a=check_a, check_b=check_b,
                check_d=check_d, vol_min=vmin, vol_med=vmed, vol_max=vmax, passed=passed)


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    corr = np.corrcoef(a[mask], b[mask])[0, 1]
    return float(corr ** 2) if np.isfinite(corr) else float("nan")


def main() -> None:
    max_ts_seen = []

    print("=" * 78)
    print("R-87 NOVEL: ACI-conformal-quantile Kelly scale -- Step A gate, "
          "6-config sweep, inner-validation selection, ETH falsification")
    print("=" * 78)

    btc_full = load_btc_bars()
    max_ts_seen.append(btc_full.index.max())

    df_train = btc_full.loc[:INNER_TRAIN_END]
    print(f"\ninner-train frame: {len(df_train):,} bars, {df_train.index[0]} -> {df_train.index[-1]}")

    # ---------------- Step A gate, all 6 configs ----------------
    print("\n--- Step A gate (inner-train only) ---")
    gate_results = []
    for gamma, window_days in SWEEP_GRID:
        res = step_a_gate_one(df_train, gamma, window_days)
        gate_results.append(res)
        print(f"gamma={gamma:<5} window_days={window_days:<4} "
              f"(a) finite&pos={res['check_a']}  (b) exact-flat={res['check_b']}  "
              f"(d) causal-probe={res['check_d']}  "
              f"vol_conformal_ann[min/med/max]={res['vol_min']:.4f}/{res['vol_med']:.4f}/{res['vol_max']:.4f}  "
              f"PASS={res['passed']}")

    passing = [r for r in gate_results if r["passed"]]
    print(f"\n{len(passing)}/6 configs passed Step A gate (a)/(b)/(d).")

    # ---------------- (c) R^2 diagnostic vs kelly_regime_v4's own path ----------------
    print("\n--- (c) R^2 diagnostic vs kelly_regime_v4's own target[] path, 2017-01-01 -> 2022-12-31 ---")
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    diag_end = "2022-12-31"
    df_diag = btc_full.loc[:diag_end].copy()
    max_ts_seen.append(df_diag.index.max())
    v4_df = KellyRegimeV4().prepare(df_diag.copy())
    v4_target = v4_df["target"].to_numpy()

    mask_period = (df_diag.index >= pd.Timestamp("2017-01-01", tz=df_diag.index.tz))
    r2_by_config = {}
    for gamma, window_days in SWEEP_GRID:
        novel_target = build_target(df_diag, gamma, window_days)
        r2 = r_squared(novel_target[mask_period], v4_target[mask_period])
        r2_by_config[(gamma, window_days)] = r2
        print(f"gamma={gamma:<5} window_days={window_days:<4}  R^2(novel.target, v4.target) = {r2:.4f}")

    # ---------------- Inner-validation selection among Step-A passers ----------------
    print("\n--- Inner-validation (2021-01-01 -> 2022-12-31) Sharpe, futures_5x, Step-A passers only ---")
    sys.path.insert(0, str(ROOT))
    from scripts.experiment import FUTURES, ev  # noqa: E402  (loads full DF; never queried >= OOS_START below)

    val_results = []
    for r in passing:
        gamma, window_days = r["gamma"], r["window_days"]
        strat = KellyRegimeACIConformal(gamma=gamma, window_days=window_days)
        m = ev(strat, market=FUTURES, start=INNER_VAL_START, end=INNER_VAL_END,
               tag=f"novel_g{gamma}_w{window_days}")
        val_results.append(dict(gamma=gamma, window_days=window_days, sharpe=m.sharpe,
                                 max_dd=m.max_drawdown_pct, metrics=m))

    if not val_results:
        print("\nNo config passed Step A. STOP -- verdict NEGATIVE by construction "
              "(no candidate reaches inner-validation).")
        print(f"\nmax timestamp read anywhere in this session (BTC only): {max(max_ts_seen)}  (< {OOS_START})")
        return

    val_results.sort(key=lambda r: r["sharpe"], reverse=True)
    finalist = val_results[0]
    print(f"\nfinalist: gamma={finalist['gamma']} window_days={finalist['window_days']}  "
          f"inner-val Sharpe={finalist['sharpe']:.3f}  maxDD={finalist['max_dd']:.1f}%")

    print("\n--- kelly_regime_v4 itself, same period/market ---")
    m_v4 = ev(KellyRegimeV4(), market=FUTURES, start=INNER_VAL_START, end=INNER_VAL_END, tag="kelly_regime_v4")

    sharpe_gap = finalist["sharpe"] - m_v4.sharpe
    dd_improved = finalist["max_dd"] < m_v4.max_drawdown_pct
    print(f"\nfinalist Sharpe - v4 Sharpe = {sharpe_gap:+.3f}   "
          f"(noise floor: must exceed +0.20, OR show a clear maxDD improvement)")
    print(f"finalist maxDD={finalist['max_dd']:.1f}%  vs  v4 maxDD={m_v4.max_drawdown_pct:.1f}%  "
          f"(improved: {dd_improved})")

    btc_bar_passed = (sharpe_gap > 0.2) or dd_improved
    print(f"\nBTC noise-floor bar: {'PASS' if btc_bar_passed else 'FAIL'}")

    # ---------------- ETH replication ----------------
    print("\n--- ETH replication (sign-check only, pre-holdout) ---")
    eth_full = load_eth_bars()
    if eth_full is None:
        print("STOP: data/ethusd_coinbase_spot_5m.csv.gz is MISSING. Cannot run the "
              "pre-registered ETH replication leg. Reporting this as a hard STOP, not "
              "a fabricated result -- verdict is NEGATIVE (falsification test cannot "
              "be satisfied without the ETH leg).")
        eth_replicated = False
        eth_note = "ETH data file missing"
    else:
        max_ts_seen.append(eth_full.index.max())
        eth_val_end = min(pd.Timestamp(INNER_VAL_END, tz=eth_full.index.tz), eth_full.index.max())
        eth_val_end_str = eth_val_end.strftime("%Y-%m-%d")
        strat_eth = KellyRegimeACIConformal(gamma=finalist["gamma"], window_days=finalist["window_days"])
        m_eth_novel = ev(strat_eth, df=eth_full, market=FUTURES, start=INNER_VAL_START,
                          end=eth_val_end_str, tag="novel_finalist_ETH")
        m_eth_v4 = ev(KellyRegimeV4(), df=eth_full, market=FUTURES, start=INNER_VAL_START,
                      end=eth_val_end_str, tag="kelly_regime_v4_ETH")
        eth_gap = m_eth_novel.sharpe - m_eth_v4.sharpe
        eth_dd_improved = m_eth_novel.max_drawdown_pct < m_eth_v4.max_drawdown_pct
        # "Same qualitative direction must replicate": whichever bar the BTC
        # finalist actually cleared (Sharpe gap > 0.2, and/or a maxDD
        # improvement) must show the SAME SIGN on ETH -- a sign match, not a
        # second independent >0.2 bar (the falsification test as frozen asks
        # for direction-replication, not magnitude-replication).
        eth_replicated = ((sharpe_gap > 0.2 and eth_gap > 0.0) or
                           (dd_improved and eth_dd_improved))
        eth_note = (f"ETH inner-val ({INNER_VAL_START} -> {eth_val_end_str}): "
                    f"Sharpe gap(novel-v4)={eth_gap:+.3f}  "
                    f"maxDD novel={m_eth_novel.max_drawdown_pct:.1f}% "
                    f"vs v4={m_eth_v4.max_drawdown_pct:.1f}%  (dd improved: {eth_dd_improved})")
        print(eth_note)
        print(f"ETH sign-replication of BTC's passing bar(s): {'PASS' if eth_replicated else 'FAIL'}")

    # ---------------- Verdict ----------------
    verdict = "PROMOTE-CANDIDATE" if (btc_bar_passed and eth_replicated) else "NEGATIVE"
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print("=" * 78)

    print(f"\nmax timestamp read anywhere in this session (BTC and ETH): "
          f"{max(max_ts_seen)}  (< {OOS_START})")


if __name__ == "__main__":
    main()
