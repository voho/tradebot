"""R-101 CONSERVATIVE branch: a frozen, static confidence multiplier for
``kelly_regime_v4``, derived once from a delete-one-group jackknife over
this project's six standard stress episodes (backlog idea, this round's
pre-registration).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

REPO-STATE AUDIT (read this first)
-----------------------------------
This round's pre-registration was written as if against a ~100-round
project history (it cites R-33, R-53, R-62, R-73, R-93, R-97, R-99, R-100
and ``experiments/r99_shared.py``). The actual repository this branch runs
in has a ledger that stops at **R-28** (``docs/LEDGER.md``, confirmed by
``git log`` — 40 commits total) and contains no file named
``r99_shared.py``, no ``STRESS_EPISODES`` constant anywhere in the tree,
and no ledger rows numbered above R-28. There is also no
``data/ethusd_coinbase_spot_5m.csv.gz`` — the only committed ETH file is
``data/ethusd_bitfinex_5m.csv.gz`` (2016-03-09 -> 2019-12-31), the same
file R-17 used for the BTC/ETH falsification test.

None of this blocks the assignment: the pre-registration gives the
``STRESS_EPISODES`` dates literally in its own text, so they are copied
below verbatim rather than imported from a file that does not exist, and
the ETH falsification below uses the real committed Bitfinex BTC/ETH pair
on their shared window, matching R-17's construction exactly. This note
is the honest record of that discrepancy; see the round report
(``experiments/reports/r101_conservative_report.md``) for the full
statement.

The mechanism
-------------
``kelly_regime_v4`` sizes exposure as ``desired = frac * scale``, where
``frac`` is the three-anchor regime vote (0, 1/3, 2/3, 1) and ``scale`` is
the volatility target. This branch introduces a third, purely
multiplicative factor ``conf`` (confidence):

    conf = clip(1 - k * CV, conf_floor, 1.0)
    desired = frac * scale * conf

``CV`` is the coefficient of variation (std/mean) of six delete-one-group
jackknife (Quenouille 1949; Tukey 1958; Efron 1979, "Bootstrap Methods:
Another Look at the Jackknife", Ann. Statist. 7(1)) leave-one-out
estimates of ``kelly_regime_v4``'s realized log-growth edge *over its own
vote* -- i.e. the incremental log-growth the regime vote buys over a
"scale-only" control that holds ``frac=1`` (full exposure, always) and
never gates on the vote at all -- computed by leaving each of the six
standard stress episodes' +/-60-day windows out of the average in turn.

This branch is the CONSERVATIVE, STATIC reading: CV (and therefore
``conf``) is computed exactly ONCE, from the inner-train period only
(2017-01-01 -> 2020-12-31), and frozen as a single constant multiplier
for the strategy's entire life -- no time variation, no re-estimation.

conf is baked in as a plain float at construction time. prepare() never
touches it and never derives it from the frame it is given, so calling
prepare() on any period (this file never calls it on 2023+ data) cannot
leak: conf is a compile-time constant here, not a running statistic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.data import load_ohlcv_csv
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.window import run_period

from experiment import DF, SPOT, FUTURES, OOS_START  # noqa: E402  (BTC spot, full period)

# ---------------------------------------------------------------------------
# Fixed pre-registered constants
# ---------------------------------------------------------------------------

INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"

# Copied verbatim from the pre-registration text (r99_shared.py does not
# exist in this repo -- see module docstring audit note above).
STRESS_EPISODES = [
    "2018-01-17",
    "2018-12-15",
    "2020-03-12",
    "2021-11-10",
    "2022-05-09",
    "2022-11-08",
]

K_GRID = (0.5, 1.0, 2.0)
CONF_FLOOR_GRID = (0.3, 0.5, 0.7)
A_PRIORI_K, A_PRIORI_FLOOR = 1.0, 0.5  # grid midpoint, named before any sweep


def episode_window(date_str: str, half_width_days: int = 60) -> tuple[pd.Timestamp, pd.Timestamp]:
    """+/-60-day window around a stress-episode date (UTC), same convention
    the regime-timing rounds use; no shared helper exists in this repo to
    import, so it is defined here per the pre-registration's fallback."""
    center = pd.Timestamp(date_str, tz="UTC")
    return center - pd.Timedelta(days=half_width_days), center + pd.Timedelta(days=half_width_days)


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------

class _ScaleOnlyV4Control(KellyRegimeV4):
    """v4 with the regime vote forced permanently on (frac=1).

    Used only as the denominator/control for the jackknife edge
    measurement below -- isolates the pure volatility-targeting scale
    factor with no vote gating at all, so ``growth(v4) - growth(this)``
    is the log-growth the vote itself contributes. Never evaluated as a
    candidate strategy and never appears in any Sharpe/backtest sweep.
    """

    name = "_r101_scale_only_control"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
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
            desired = 1.0 * scale  # frac forced to 1: vote fully on, always
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


class ConservativeStaticJackknifeV4(KellyRegimeV4):
    """v4 with a third, frozen multiplicative factor: desired = frac*scale*conf.

    ``conf`` is a plain float baked in at construction time by
    ``compute_static_conf`` (see below) -- computed once, off inner-train
    data only, and never touched again. This class does not compute CV or
    the jackknife itself; it just applies the frozen number inside the
    same deadband loop v3/v4 use, so the deadband nonlinearity sees the
    real, conf-scaled ``desired`` value (matching the pre-registration:
    the multiplier is applied *before* the deadband check, not pasted on
    top of v4's own output afterward).
    """

    name = "kelly_regime_v4_conservative_static_jackknife"

    def __init__(self, conf: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conf = float(conf)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
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
            desired = frac[i] * scale * self.conf  # the one line that differs from v4
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


# ---------------------------------------------------------------------------
# Step 0: the jackknife itself (KS-A input)
# ---------------------------------------------------------------------------

def compute_jackknife_cv(df: pd.DataFrame, market: MarketSpec,
                          period_start: str, period_end: str,
                          episodes: list[str]) -> dict:
    """Delete-one-group jackknife CV of v4's log-growth edge over its vote.

    Runs v4 and the scale-only control once each over [period_start,
    period_end], forms the per-bar edge series (v4's log-return minus the
    control's), then for each stress episode whose +/-60-day window
    overlaps the measured period, computes the mean edge with that window
    excluded (a leave-one-out / delete-one-group estimate). Returns the
    per-episode LOO estimates, which episodes were actually usable, and
    CV = std/mean across them.
    """
    v4 = KellyRegimeV4()
    ctrl = _ScaleOnlyV4Control()

    r_v4 = run_period(v4, df, start=period_start, end=period_end, market=market)
    r_ctrl = run_period(ctrl, df, start=period_start, end=period_end, market=market)

    eq_v4 = r_v4.equity
    eq_ctrl = r_ctrl.equity
    assert eq_v4.index.equals(eq_ctrl.index), "v4 and control equity curves must align"

    logret_v4 = np.log(eq_v4).diff()
    logret_ctrl = np.log(eq_ctrl).diff()
    edge = (logret_v4 - logret_ctrl).dropna()

    lo_bound, hi_bound = edge.index[0], edge.index[-1]

    used_episodes = []
    loo_estimates = []
    for date_str in episodes:
        w_lo, w_hi = episode_window(date_str)
        # only usable if the window has at least some overlap with the
        # measured period -- otherwise "leaving it out" changes nothing
        # and the pseudo-value would just be the grand mean by construction
        if w_hi < lo_bound or w_lo > hi_bound:
            continue
        mask = (edge.index >= w_lo) & (edge.index <= w_hi)
        if not mask.any():
            continue
        used_episodes.append(date_str)
        loo_estimates.append(float(edge[~mask].mean()))

    loo_arr = np.array(loo_estimates, dtype=float)
    if len(loo_arr) >= 2 and abs(loo_arr.mean()) > 1e-12:
        cv = float(loo_arr.std(ddof=1) / loo_arr.mean())
    else:
        cv = float("nan")

    return {
        "used_episodes": used_episodes,
        "n_episodes_in_spec": len(episodes),
        "n_episodes_usable": len(used_episodes),
        "loo_estimates": loo_estimates,
        "cv": cv,
        "grand_mean_edge": float(edge.mean()),
        "edge_std": float(edge.std(ddof=1)),
        "n_bars": int(len(edge)),
    }


def compute_static_conf(k: float, conf_floor: float, cv: float) -> float:
    if not np.isfinite(cv):
        # degenerate dispersion measurement: conf collapses to the identity
        # multiplier rather than dividing by/using a NaN
        return 1.0
    return float(np.clip(1.0 - k * cv, conf_floor, 1.0))


# ---------------------------------------------------------------------------
# KS-B: R^2 of the exposure path against v4's own unmodified exposure path
# ---------------------------------------------------------------------------

def exposure_r_squared(df: pd.DataFrame, market: MarketSpec, conf: float,
                        start: str, end: str) -> dict:
    v4 = KellyRegimeV4()
    cand = ConservativeStaticJackknifeV4(conf=conf)

    r_v4 = run_period(v4, df, start=start, end=end, market=market)
    r_cand = run_period(cand, df, start=start, end=end, market=market)

    t_v4 = r_v4.df["target"].to_numpy()
    t_cand = r_cand.df["target"].to_numpy()
    assert len(t_v4) == len(t_cand)

    # R^2 of cand's exposure path explained by v4's own (simple OLS through
    # an intercept, the standard "is this just a rescale" check)
    if np.std(t_v4) == 0:
        r2 = float("nan")
    else:
        corr = np.corrcoef(t_v4, t_cand)[0, 1]
        r2 = float(corr ** 2)

    return {
        "r2": r2,
        "n_bars": len(t_v4),
        "mean_abs_diff": float(np.mean(np.abs(t_v4 - t_cand))),
        "mean_v4": float(np.mean(t_v4)),
        "mean_cand": float(np.mean(t_cand)),
    }


# ---------------------------------------------------------------------------
# Battery (only reached if both kill switches pass)
# ---------------------------------------------------------------------------

def summarize(m) -> str:
    return (f"final=${m.final_balance:>13,.0f} ({m.profit_pct:>+9.1f}%) "
            f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
            f"sharpe={m.sharpe:>5.2f}{' LIQUIDATED' if m.liquidated else ''}")


def run_battery(configs: list[tuple[float, float, float]]) -> None:
    """configs: list of (k, conf_floor, conf) tuples already computed."""
    from tradebot.metrics import compute_metrics

    print("\n=== inner-train (2017-01-01 -> 2020-12-31), BTC spot ===")
    for k, floor, conf in configs:
        m = compute_metrics(run_period(ConservativeStaticJackknifeV4(conf=conf), DF,
                                        start=INNER_TRAIN_START, end=INNER_TRAIN_END,
                                        market=SPOT))
        print(f"k={k:<4} floor={floor:<4} conf={conf:.4f}  {summarize(m)}")
    m = compute_metrics(run_period(KellyRegimeV4(), DF, start=INNER_TRAIN_START,
                                    end=INNER_TRAIN_END, market=SPOT))
    print(f"{'v4 (unmodified)':24s} {summarize(m)}")

    print("\n=== inner-validation (2021-01-01 -> 2022-12-31), BTC spot ===")
    for k, floor, conf in configs:
        m = compute_metrics(run_period(ConservativeStaticJackknifeV4(conf=conf), DF,
                                        start=INNER_VAL_START, end=INNER_VAL_END,
                                        market=SPOT))
        print(f"k={k:<4} floor={floor:<4} conf={conf:.4f}  {summarize(m)}")
    m = compute_metrics(run_period(KellyRegimeV4(), DF, start=INNER_VAL_START,
                                    end=INNER_VAL_END, market=SPOT))
    print(f"{'v4 (unmodified)':24s} {summarize(m)}")


if __name__ == "__main__":
    print(f"Dataset: {len(DF):,} bars {DF.index[0]} -> {DF.index[-1]}", file=sys.stderr)

    jk = compute_jackknife_cv(DF, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, STRESS_EPISODES)
    print("\n--- KS-A: jackknife dispersion (inner-train only) ---")
    print(f"episodes in spec: {jk['n_episodes_in_spec']}, usable on inner-train: "
          f"{jk['n_episodes_usable']} ({jk['used_episodes']})")
    print(f"LOO edge estimates: {jk['loo_estimates']}")
    print(f"grand mean edge/bar: {jk['grand_mean_edge']:.3e}  edge std/bar: {jk['edge_std']:.3e}")
    print(f"CV = {jk['cv']:.4f}")
    ks_a = np.isfinite(jk["cv"]) and jk["cv"] >= 0.10
    print(f"KS-A (CV >= 0.10): {'PASS' if ks_a else 'FAIL'}")

    conf_apriori = compute_static_conf(A_PRIORI_K, A_PRIORI_FLOOR, jk["cv"])
    print(f"\na-priori config for KS-B: k={A_PRIORI_K}, floor={A_PRIORI_FLOOR} -> conf={conf_apriori:.4f}")

    ksb = exposure_r_squared(DF, SPOT, conf_apriori, INNER_TRAIN_START, INNER_VAL_END)
    print("\n--- KS-B: exposure-path R^2 vs v4 unmodified (inner-train U inner-validation) ---")
    print(f"R^2 = {ksb['r2']:.4f}  n_bars={ksb['n_bars']:,}  mean|diff|={ksb['mean_abs_diff']:.4f}"
          f"  mean(v4)={ksb['mean_v4']:.4f}  mean(cand)={ksb['mean_cand']:.4f}")
    ks_b = np.isfinite(ksb["r2"]) and ksb["r2"] < 0.95
    print(f"KS-B (R^2 < 0.95): {'PASS' if ks_b else 'FAIL'}")

    if not (ks_a and ks_b):
        print("\n*** At least one kill switch FAILED. Stopping per pre-registration. ***")
        sys.exit(0)

    # Only reached if both kill switches pass -- not expected to run, kept
    # for completeness/faithfulness to the pre-registration.
    configs = []
    for k in K_GRID:
        for floor in CONF_FLOOR_GRID:
            configs.append((k, floor, compute_static_conf(k, floor, jk["cv"])))
    configs.append((0.0, A_PRIORI_FLOOR, compute_static_conf(0.0, A_PRIORI_FLOOR, jk["cv"])))
    print(f"\n{len(configs)} configurations to sweep.")
    run_battery(configs)
