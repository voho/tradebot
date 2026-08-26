#!/usr/bin/env python
"""R-152 CONSERVATIVE branch: ``ConservativeCDaRCap`` -- kelly_regime_v4's exact
architecture (3-anchor vote, 20/40/80-day anchors, 1% band, extremes-only
hysteresis latch, 10% deadband) with exactly ONE substitution: the fixed
``max_leverage = 2.0`` constant (used in BOTH the ``full`` and ``steady``
branches of the sizing rule) is replaced by a rolling causal CDaR_0.95
(Chekhlov, Uryasev & Zabarankin 2005), computed via ``r152_shared.rolling_cdar``
over a REFERENCE wealth path built from v4's own unmodified vote-scaled return
series (``frac[i] * r[i]``, the raw signal before any sizing/leverage is
applied -- NOT the executed, capped position, so there is no circularity
between the derived cap and the path used to derive it).

Mechanism (exact): the CDaR series is inverted (``1 / cdar``, the natural
sense in which a risk statistic becomes a leverage CAP -- large drawdown risk
tightens the cap, exactly the same inverse relationship v4's own
``target_vol / realized_vol`` already uses for its risk axis) and then
RESCALED by one scalar fit so its mean over INNER-TRAIN equals 2.0, matching
v4's own constant exactly on average -- this branch changes *when* leverage
tightens, not the average amount of it. Full literature grounding, the
non-duplication argument, and the pre-registered decision rule all live in
``experiments/r152_shared.py``'s own module docstring (read in full before
this file was written); not re-derived here beyond the summary above. This
file NEVER edits ``r152_shared.py`` (frozen, shared with the parallel NOVEL
branch, a disjoint file this session does not read or coordinate with), and
never reads a bar at or after ``r152_shared.OOS_START`` (2023-01-01) unless
the branch actually passes the pre-registered selection rule (self-enforced
by an assertion in ``run_holdout`` below, not just promised in prose).

FALSIFICATION TEST (named in advance, per the round's pre-registration):
survives on ETH -- this branch's dynamic cap (frozen: BTC-inner-train-fitted
rescale factor, selected window length, no ETH-specific tuning) must not
reverse sign against `kelly_regime_v4` and `buy_and_hold` on ETH spot, using
data only through ``INNER_VAL_END`` (2022-12-31) -- ETH holdout is out of
scope per the R-125 precedent (real ETH data does not reliably extend
cleanly past that, and this round does not claim otherwise).

CONFIGURATIONS EVALUATED (ceiling 6 for this branch, per r152_shared.py):
3 (selection-stage window sweep: 180/365/545d, each inner-train fit +
inner-validation score) + 1 (control re-run, unmodified kelly_regime_v4,
identical inner-validation slice) = 4 at selection. If and only if eligible
for holdout: +1 (holdout score, selected window only) +1 (ETH falsification)
= up to 2 more. Diagnostic B2 and the causal-truncation self-test are
diagnostics, not scored configurations, and are not counted (mirrors
r125_conservative_cvar_scale.py's own convention).

USAGE
-----
    python experiments/r152_conservative_cdar_cap.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r152_shared  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# Not one of r152_shared's frozen constants (it doesn't define an inner-train
# START -- only INNER_TRAIN_END). Matches the r125 round's own convention
# (r125_shared.INNER_TRAIN_START) and the dataset's actual first bar.
INNER_TRAIN_START = "2017-01-01"

FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

TARGET_CAP_MEAN = 2.0  # kelly_regime_v4's own fixed max_leverage constant


# ================================================================== (1)
# Shared pieces of KellyRegimeV3.prepare(), factored out so the reference
# return series / calibration / diagnostics can reuse them without
# re-deriving v4's own vote logic. Byte-identical formulas to
# kelly_regime_v3.py / kelly_regime.py -- verbatim extraction, not a
# re-derivation.
# ==================================================================

def vote_frac(close: pd.Series, horizons: tuple[int, ...], band: float,
               vote_gamma: float) -> np.ndarray:
    """v3/v4's own 3-anchor crowd-regime vote, byte-identical formula."""
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    frac = (sum(votes) / len(votes)).to_numpy()
    if vote_gamma != 1.0:
        frac = frac ** vote_gamma
    return frac


def reference_wealth_returns(close: pd.Series, frac: np.ndarray) -> np.ndarray:
    """v4's own UNMODIFIED vote-scaled return series: frac[i] * r[i], the raw
    signal before any sizing/leverage is applied (NOT the executed, capped
    position) -- so there is no circularity between the derived cap and the
    path used to derive it (r152_shared.py's own framing, verbatim)."""
    r = np.log(close).diff().to_numpy()
    return frac * np.nan_to_num(r, nan=0.0)


def raw_leverage_cap(close: pd.Series, frac: np.ndarray, cdar_window_days: int) -> np.ndarray:
    """1 / CDaR_0.95 of the reference wealth path -- the pre-rescale
    candidate leverage cap. Larger drawdown risk (CDaR) -> smaller raw cap,
    the same inverse sense v4's own target_vol/realized_vol already uses."""
    ref = reference_wealth_returns(close, frac)
    wbars = r152_shared.window_bars(cdar_window_days)
    cdar = r152_shared.rolling_cdar(ref, wbars, beta=r152_shared.CDAR_BETA)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_cap = np.where(cdar > 0, 1.0 / cdar, np.nan)
    return raw_cap


def vol_series(close: pd.Series, vol_span: int) -> np.ndarray:
    """v3/v4's own realized-volatility input, byte-identical to
    kelly_regime.py's prepare() (EWM-std of log returns, annualized,
    shift(1)'d)."""
    r = np.log(close).diff()
    return (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def calibrate_rescale_factor(df_inner_train: pd.DataFrame, cdar_window_days: int,
                              horizons: tuple[int, ...] = (20, 40, 80),
                              band: float = 0.01, vote_gamma: float = 1.0,
                              target_cap_mean: float = TARGET_CAP_MEAN) -> float:
    """Fit the ONE scalar this branch calibrates: rescale_factor such that
    mean(rescale_factor * raw_leverage_cap) over INNER-TRAIN equals
    ``target_cap_mean`` (2.0, v4's own constant). Computed ONCE, on
    inner-train only -- the same fixed number is then applied unchanged to
    inner-validation and (if eligible) holdout and ETH."""
    close = df_inner_train["close"]
    frac = vote_frac(close, horizons, band, vote_gamma)
    raw_cap = raw_leverage_cap(close, frac, cdar_window_days)
    mean_raw = float(np.nanmean(raw_cap))
    if not np.isfinite(mean_raw) or mean_raw <= 0:
        raise ValueError(f"cannot calibrate rescale_factor: mean(raw_cap)={mean_raw!r}")
    return target_cap_mean / mean_raw


# ================================================================== (2)
# ConservativeCDaRCap: KellyRegimeV3.prepare(), copied faithfully, with
# exactly one substitution (see module docstring). NOT @register'd --
# experiments/-only, per this round's instructions.
# ==================================================================

class ConservativeCDaRCap(KellyRegimeV4):
    """kelly_regime_v4's exact architecture with the ONE substitution this
    branch tests: the fixed ``max_leverage=2.0`` cap (in BOTH ``full`` and
    ``steady``) becomes a rolling causal CDaR_0.95-derived, inner-train-
    rescaled dynamic cap. Everything else -- ``frac``, the anchors, the 1%
    band, the hysteresis latch on the vol ratio, ``target_vol``, the 10%
    deadband -- is byte-identical to ``KellyRegimeV3.prepare()``.
    """

    name = "r152_conservative_cdar_cap"

    def __init__(self, cdar_window_days: int = r152_shared.CDAR_WINDOW_DAYS_DEFAULT,
                 rescale_factor: float | None = None,
                 horizons: tuple[int, ...] = (20, 40, 80), **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.cdar_window_days = cdar_window_days
        self.rescale_factor = rescale_factor  # must be calibrated before prepare()
        # Warmup must cover both v4's own anchor/vol warmup AND a full CDaR
        # window (up to 545 days), or the dynamic cap would spuriously read
        # as NaN (-> zero exposure) for the first stretch of every period.
        self.warmup = max(80 * BARS_PER_DAY + 10,
                          r152_shared.window_bars(cdar_window_days) + BARS_PER_DAY)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.rescale_factor is None:
            raise ValueError(
                "rescale_factor must be calibrated (calibrate_rescale_factor) "
                "before prepare() is called")
        close = df["close"]
        r = np.log(close).diff()

        # Vote: byte-identical to KellyRegimeV3.prepare() / KellyRegimeV4.
        frac = vote_frac(close, self.horizons, self.band, self.vote_gamma)

        # Hysteresis input: byte-identical to KellyRegimeV3.prepare().
        vol = vol_series(close, self.vol_span)
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        # ---- THE ONE SUBSTITUTION: dynamic CDaR-derived leverage cap in
        # place of the fixed max_leverage=2.0 constant, used in BOTH full
        # and steady. Everything downstream (the ratio, the hysteresis
        # state machine, the deadband) is byte-identical to
        # KellyRegimeV3.prepare().
        raw_cap = raw_leverage_cap(close, frac, self.cdar_window_days)
        dyn_cap = self.rescale_factor * raw_cap

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, dyn_cap)
            steady = np.minimum(self.target_vol / slow, dyn_cap)
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
        return df


def make_candidate_factory(cdar_window_days: int, rescale_factor: float):
    def _factory():
        return ConservativeCDaRCap(cdar_window_days=cdar_window_days,
                                   rescale_factor=rescale_factor)
    return _factory


# ================================================================== (3)
# Causal-truncation self-test (this round's own new code, real BTC data).
# ==================================================================

def causal_truncation_probe(df: pd.DataFrame, rescale_factor: float,
                            cdar_window_days: int = r152_shared.CDAR_WINDOW_DAYS_DEFAULT,
                            cut: int = 400_000) -> bool:
    full = ConservativeCDaRCap(cdar_window_days=cdar_window_days,
                               rescale_factor=rescale_factor).prepare(
        df.copy())["target"].to_numpy()
    trunc = ConservativeCDaRCap(cdar_window_days=cdar_window_days,
                                rescale_factor=rescale_factor).prepare(
        df.iloc[:cut].copy())["target"].to_numpy()
    n_check = min(len(trunc), cut) - BARS_PER_DAY * (cdar_window_days + 1)
    return bool(np.allclose(full[:n_check], trunc[:n_check], equal_nan=True, rtol=1e-9))


# ================================================================== (4)
# Diagnostic B2: Pearson correlation, inner-train, dynamic cap (post-
# rescale) vs v4's own realized-volatility `vol` column.
# ==================================================================

def b2_correlation(df_inner_train: pd.DataFrame, cdar_window_days: int,
                    rescale_factor: float, horizons: tuple[int, ...] = (20, 40, 80),
                    band: float = 0.01, vote_gamma: float = 1.0,
                    vol_span: int = 8 * BARS_PER_DAY) -> float:
    close = df_inner_train["close"]
    frac = vote_frac(close, horizons, band, vote_gamma)
    dyn_cap = rescale_factor * raw_leverage_cap(close, frac, cdar_window_days)
    vol = vol_series(close, vol_span)
    mask = np.isfinite(dyn_cap) & np.isfinite(vol)
    if mask.sum() < 100:
        return float("nan")
    return float(np.corrcoef(dyn_cap[mask], vol[mask])[0, 1])


# ================================================================== (5)
# Scoring helpers.
# ==================================================================

def mean_exposure_pct(result, market: MarketSpec) -> float:
    """mean(clip(|target|, 0, market.leverage)) * 100 -- the percentage-
    point unit EXPOSURE_MATCH_TOL_PP is compared against. Matches this
    project's own r78/matched_hold `mean_notional` convention, expressed as
    a percentage."""
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, market.leverage))) * 100.0


def score_arm(strategy_factory, df: pd.DataFrame, market: MarketSpec,
              start: str, end: str) -> dict:
    strat = strategy_factory()
    result = run_period(strat, df, start=start, end=end, market=market,
                        start_balance=1000.0, data_label="")
    m = compute_metrics(result)
    return {
        "final_balance": m.final_balance,
        "sharpe": m.sharpe,
        "max_drawdown_pct": m.max_drawdown_pct,
        "time_in_market_pct": m.time_in_market_pct,
        "mean_exposure_pct": mean_exposure_pct(result, market),
        "profit_pct": m.profit_pct,
    }


def run_holdout(strategy_factory, df: pd.DataFrame, market: MarketSpec,
                eligible: bool) -> dict:
    """Only ever called with eligible=True from main() after the selection
    rule has actually passed -- hard-enforced here, not just promised in
    prose, per this round's hard constraint on reading OOS bars."""
    assert eligible, ("holdout run blocked: branch has not passed the "
                      "pre-registered selection rule")
    return score_arm(strategy_factory, df, market, r152_shared.OOS_START, None)


def print_row(tag: str, r: dict) -> None:
    print(f"  {tag:<28s} final=${r['final_balance']:>12,.0f} ({r['profit_pct']:>+8.1f}%) "
          f"sharpe={r['sharpe']:>+6.3f} DD={r['max_drawdown_pct']:>5.1f}% "
          f"time_in_mkt={r['time_in_market_pct']:>5.1f}% "
          f"mean_exposure={r['mean_exposure_pct']:>6.1f}%")


# ================================================================== (6)
# Main: B2 -> causal probe -> selection stage -> [holdout + ETH] -> verdict.
# ==================================================================

def main() -> dict:
    t0 = time.time()
    n_configs = 0
    max_ts_seen: list[pd.Timestamp] = []

    print("=" * 78)
    print("R-152 CONSERVATIVE: ConservativeCDaRCap -- kelly_regime_v4's own architecture,")
    print("fixed max_leverage=2.0 replaced by a rolling causal CDaR_0.95-derived,")
    print("inner-train-rescaled dynamic leverage cap.")
    print("=" * 78)

    btc_full, label = load_dataset(ROOT / "data", "spot")
    dev = btc_full.loc[:r152_shared.INNER_VAL_END].copy()
    max_ts_seen.append(dev.index.max())
    assert dev.index[-1] < pd.Timestamp(r152_shared.OOS_START, tz=dev.index.tz), \
        "dev slice breaches OOS_START"
    print(f"\nBTC {label} (dev slice, truncated < {r152_shared.OOS_START}): "
          f"{len(dev):,} bars, {dev.index[0]} -> {dev.index[-1]}")

    inner_train = dev.loc[INNER_TRAIN_START:r152_shared.INNER_TRAIN_END]
    print(f"inner-train: {len(inner_train):,} bars, {inner_train.index[0]} -> "
          f"{inner_train.index[-1]}")

    # -------------------------------------------------------------- fit primary (365d)
    primary_window = r152_shared.CDAR_WINDOW_DAYS_DEFAULT
    print(f"\n-- CALIBRATING rescale_factor (primary window={primary_window}d) on "
          f"inner-train only --")
    rescale_primary = calibrate_rescale_factor(inner_train, primary_window)
    print(f"  rescale_factor (primary, {primary_window}d) = {rescale_primary:.6f}")

    # -------------------------------------------------------------- causal probe
    print("\n" + "=" * 78)
    print("CAUSAL-TRUNCATION SELF-TEST (this round's own new code, real BTC data)")
    print("=" * 78)
    probe_ok = causal_truncation_probe(dev, rescale_primary, primary_window)
    print(f"  causal_truncation_probe (primary config, window={primary_window}d): "
          f"{'PASS' if probe_ok else 'FAIL'}")
    if not probe_ok:
        print("\nCAUSAL PROBE FAILURE -- stopping before any number is trusted.")
        max_ts = max(max_ts_seen)
        return dict(verdict="ABORTED (causal probe failure)", n_configs=n_configs,
                   max_ts=max_ts)

    # -------------------------------------------------------------- diagnostic B2
    print("\n" + "=" * 78)
    print("DIAGNOSTIC B2 -- Pearson correlation, inner-train, dynamic cap (post-rescale) "
          "vs v4's own realized-vol `vol` column (reported regardless of outcome)")
    print("=" * 78)
    b2_r = b2_correlation(inner_train, primary_window, rescale_primary)
    b2_flagged = abs(b2_r) >= r152_shared.B2_CORRELATION_FLAG
    print(f"  r = {b2_r:+.4f}   |r| >= {r152_shared.B2_CORRELATION_FLAG:g} (flagged): {b2_flagged}")
    if b2_flagged:
        print("  FLAGGED: the dynamic cap likely a smoothed relabeling of realized "
              "volatility -- any headline result below must say so explicitly.")

    # -------------------------------------------------------------- selection stage
    print("\n" + "=" * 78)
    print(f"SELECTION STAGE -- inner-validation ({r152_shared.INNER_VAL_START} -> "
          f"{r152_shared.INNER_VAL_END}), futures 5x, BTC")
    print(f"window sweep: {r152_shared.CDAR_WINDOW_DAYS_SWEEP}")
    print("=" * 78)

    ctrl_factory = lambda: get_strategy("kelly_regime_v4")
    ctrl = score_arm(ctrl_factory, dev, FUTURES, r152_shared.INNER_VAL_START,
                     r152_shared.INNER_VAL_END)
    n_configs += 1
    print_row("control (kelly_regime_v4)", ctrl)

    arms = {}
    for wd in r152_shared.CDAR_WINDOW_DAYS_SWEEP:
        rf = rescale_primary if wd == primary_window else calibrate_rescale_factor(inner_train, wd)
        factory = make_candidate_factory(wd, rf)
        r = score_arm(factory, dev, FUTURES, r152_shared.INNER_VAL_START,
                     r152_shared.INNER_VAL_END)
        r["rescale_factor"] = rf
        r["d_sharpe"] = r["sharpe"] - ctrl["sharpe"]
        arms[wd] = r
        n_configs += 1
        print_row(f"window={wd}d (rf={rf:.4f})", r)
        print(f"    d_sharpe={r['d_sharpe']:+.4f}")

    # -------------------------------------------------------------- selection rule
    print("\n" + "-" * 78)
    print("SELECTION RULE (frozen, r152_shared.py)")
    print("-" * 78)

    # Clause 1: matched exposure (time-in-market AND mean exposure within 15pp).
    exposure_checks = {}
    for wd, r in arms.items():
        tim_ok = abs(r["time_in_market_pct"] - ctrl["time_in_market_pct"]) <= r152_shared.EXPOSURE_MATCH_TOL_PP
        exp_ok = abs(r["mean_exposure_pct"] - ctrl["mean_exposure_pct"]) <= r152_shared.EXPOSURE_MATCH_TOL_PP
        exposure_checks[wd] = tim_ok and exp_ok
        print(f"  clause 1 (matched exposure) window={wd}d: time_in_mkt_diff="
              f"{r['time_in_market_pct'] - ctrl['time_in_market_pct']:+.1f}pp "
              f"mean_exposure_diff={r['mean_exposure_pct'] - ctrl['mean_exposure_pct']:+.1f}pp "
              f"-> {'PASS' if exposure_checks[wd] else 'FAIL'}")
    clause1_pass = all(exposure_checks.values())
    print(f"  CLAUSE 1 (all 3 windows matched-exposure): {'PASS' if clause1_pass else 'FAIL'}")

    # Clause 2: Sharpe does not fall > 0.2 OR max-DD improves > 3pp at matched exposure.
    sharpe_checks = {}
    for wd, r in arms.items():
        sharpe_ok = r["d_sharpe"] >= -r152_shared.SHARPE_NOISE_FLOOR
        dd_improve = ctrl["max_drawdown_pct"] - r["max_drawdown_pct"]
        dd_ok = dd_improve > r152_shared.DRAWDOWN_IMPROVEMENT_PP
        sharpe_checks[wd] = sharpe_ok or dd_ok
        print(f"  clause 2 window={wd}d: d_sharpe={r['d_sharpe']:+.4f} "
              f"(>= -{r152_shared.SHARPE_NOISE_FLOOR:g}: {sharpe_ok})  "
              f"dd_improve={dd_improve:+.2f}pp (> {r152_shared.DRAWDOWN_IMPROVEMENT_PP:g}pp: {dd_ok}) "
              f"-> {'PASS' if sharpe_checks[wd] else 'FAIL'}")
    clause2_pass = all(sharpe_checks.values())
    print(f"  CLAUSE 2 (all 3 windows sharpe-or-DD): {'PASS' if clause2_pass else 'FAIL'}")

    # Clause 3: plateau -- >= 2 of 3 windows agree in SIGN of d_sharpe.
    signs = [float(np.sign(r["d_sharpe"])) for r in arms.values()]
    sign_counts = {s: signs.count(s) for s in set(signs)}
    majority_sign, majority_n = max(sign_counts.items(), key=lambda kv: kv[1])
    clause3_pass = majority_n >= 2
    print(f"  clause 3 signs: {dict(zip(arms.keys(), signs))}  majority_sign={majority_sign:+.0f} "
          f"({majority_n}/3)")
    print(f"  CLAUSE 3 (plateau, >=2/3 agree in sign): {'PASS' if clause3_pass else 'FAIL'}")

    eligible = bool(clause1_pass and clause2_pass and clause3_pass)
    print(f"\nSELECTION VERDICT: {'ELIGIBLE FOR HOLDOUT' if eligible else 'NEGATIVE at inner-validation'}")
    if not eligible:
        failed = [name for name, ok in (("clause 1 (matched exposure)", clause1_pass),
                                        ("clause 2 (sharpe/DD)", clause2_pass),
                                        ("clause 3 (plateau)", clause3_pass)) if not ok]
        print(f"Failed: {', '.join(failed)}")

    max_ts = max(max_ts_seen)
    result = dict(
        verdict="NEGATIVE (inner-validation)" if not eligible else "ELIGIBLE FOR HOLDOUT",
        n_configs=n_configs, max_ts=max_ts, b2_r=b2_r, b2_flagged=b2_flagged,
        probe_ok=probe_ok, ctrl=ctrl, arms=arms,
        clause1_pass=clause1_pass, clause2_pass=clause2_pass, clause3_pass=clause3_pass,
        eligible=eligible,
    )

    if not eligible:
        print(f"\nconfigurations evaluated: {n_configs} (selection stage only; "
              f"holdout not read, per hard constraint)")
        print(f"max timestamp read anywhere in this branch: {max_ts} "
              f"(< {r152_shared.OOS_START}: {max_ts < pd.Timestamp(r152_shared.OOS_START, tz=max_ts.tz)})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return result

    # ============================================================ HOLDOUT
    # Only reached if eligible == True.
    print("\n" + "=" * 78)
    print(f"HOLDOUT ({r152_shared.OOS_START} ->), futures 5x, BTC, 0.10% fee tier")
    print("=" * 78)

    # Selected window: best inner-validation Sharpe among windows on the
    # correct-sign (majority) side of the plateau. This picking rule is NOT
    # specified by r152_shared.py beyond the plateau check -- it is this
    # session's own inference, stated explicitly per the round's instructions.
    correct_side = {wd: r for wd, r in arms.items() if np.sign(r["d_sharpe"]) == majority_sign}
    selected_wd = max(correct_side, key=lambda wd: correct_side[wd]["sharpe"])
    selected_rf = arms[selected_wd]["rescale_factor"]
    print(f"Selected window: {selected_wd}d (best inner-val Sharpe among the "
          f"{majority_n}/3 windows on the majority-sign side of the plateau) -- "
          f"picking rule is this session's inference, not specified by r152_shared.py.")

    holdout_factory = make_candidate_factory(selected_wd, selected_rf)
    holdout_ctrl_factory = lambda: get_strategy("kelly_regime_v4")
    holdout_bh_factory = lambda: get_strategy("buy_and_hold")

    holdout_cand = run_holdout(holdout_factory, btc_full, FUTURES, eligible)
    holdout_ctrl = run_holdout(holdout_ctrl_factory, btc_full, FUTURES, eligible)
    holdout_bh = run_holdout(holdout_bh_factory, btc_full, FUTURES, eligible)
    n_configs += 1
    max_ts_seen.append(btc_full.index.max())
    print_row(f"candidate (window={selected_wd}d)", holdout_cand)
    print_row("control (kelly_regime_v4)", holdout_ctrl)
    print_row("buy_and_hold", holdout_bh)

    holdout_cand_high_fee = run_holdout(
        make_candidate_factory(selected_wd, selected_rf), btc_full,
        MarketSpec.futures(leverage=5.0, fee_rate=0.0040), eligible)
    print_row("candidate @0.40% fee (caveat only)", holdout_cand_high_fee)

    beats_bh = holdout_cand["final_balance"] > holdout_bh["final_balance"]
    d_sharpe_holdout = holdout_cand["sharpe"] - holdout_ctrl["sharpe"]
    dd_improve_holdout = holdout_ctrl["max_drawdown_pct"] - holdout_cand["max_drawdown_pct"]
    beats_ctrl = (d_sharpe_holdout > r152_shared.SHARPE_NOISE_FLOOR) or \
        (dd_improve_holdout > r152_shared.DRAWDOWN_IMPROVEMENT_PP)
    print(f"\n  beats buy_and_hold (final balance): {beats_bh}")
    print(f"  d_sharpe vs control = {d_sharpe_holdout:+.4f}  "
          f"dd_improve = {dd_improve_holdout:+.2f}pp  beats control: {beats_ctrl}")

    # -------------------------------------------------------------- ETH falsification
    print("\n" + "=" * 78)
    print(f"ETH FALSIFICATION -- spot, data through {r152_shared.INNER_VAL_END} only "
          f"(ETH holdout out of scope, per the R-125 precedent)")
    print("=" * 78)
    eth_full = load_coinbase_eth_spot(ROOT / "data")
    eth_dev = eth_full.loc[:r152_shared.INNER_VAL_END].copy()
    max_ts_seen.append(eth_dev.index.max())
    assert eth_dev.index[-1] < pd.Timestamp(r152_shared.OOS_START, tz=eth_dev.index.tz)
    print(f"ETH spot (truncated < {r152_shared.OOS_START}): {len(eth_dev):,} bars, "
          f"{eth_dev.index[0]} -> {eth_dev.index[-1]}")
    print(f"scored window: {r152_shared.INNER_VAL_START} -> {r152_shared.INNER_VAL_END} "
          f"(same as BTC inner-validation, for methodological consistency)")

    eth_cand = score_arm(make_candidate_factory(selected_wd, selected_rf), eth_dev, SPOT,
                        r152_shared.INNER_VAL_START, r152_shared.INNER_VAL_END)
    eth_ctrl = score_arm(lambda: get_strategy("kelly_regime_v4"), eth_dev, SPOT,
                        r152_shared.INNER_VAL_START, r152_shared.INNER_VAL_END)
    eth_bh = score_arm(lambda: get_strategy("buy_and_hold"), eth_dev, SPOT,
                       r152_shared.INNER_VAL_START, r152_shared.INNER_VAL_END)
    n_configs += 1
    print_row(f"ETH candidate (window={selected_wd}d)", eth_cand)
    print_row("ETH control (kelly_regime_v4)", eth_ctrl)
    print_row("ETH buy_and_hold", eth_bh)

    eth_beats_bh = eth_cand["final_balance"] > eth_bh["final_balance"]
    eth_d_sharpe = eth_cand["sharpe"] - eth_ctrl["sharpe"]
    btc_holdout_sign = float(np.sign(d_sharpe_holdout))
    eth_sign = float(np.sign(eth_d_sharpe))
    eth_survives = bool(btc_holdout_sign != 0 and eth_sign == btc_holdout_sign)
    print(f"\n  ETH beats buy_and_hold: {eth_beats_bh}   ETH d_sharpe vs control = "
          f"{eth_d_sharpe:+.4f}")
    print(f"  BTC holdout d_sharpe sign = {btc_holdout_sign:+.0f}   ETH d_sharpe sign = "
          f"{eth_sign:+.0f}   SAME SIGN (ETH falsification survives): {eth_survives}")

    # -------------------------------------------------------------- holdout plateau check
    print("\n" + "=" * 78)
    print("HOLDOUT PLATEAU CHECK -- >=2/3 window lengths agree in sign of d_sharpe vs "
          "control, on holdout too (per this round's task instructions; note this adds "
          "runs beyond r152_shared.py's own declared 'selected window only' ceiling)")
    print("=" * 78)
    holdout_signs = {selected_wd: btc_holdout_sign}
    other_windows = [wd for wd in r152_shared.CDAR_WINDOW_DAYS_SWEEP if wd != selected_wd]
    for wd in other_windows:
        rf = arms[wd]["rescale_factor"]
        r = run_holdout(make_candidate_factory(wd, rf), btc_full, FUTURES, eligible)
        n_configs += 1
        d_sharpe_wd = r["sharpe"] - holdout_ctrl["sharpe"]
        holdout_signs[wd] = float(np.sign(d_sharpe_wd))
        print_row(f"holdout window={wd}d", r)
        print(f"    d_sharpe={d_sharpe_wd:+.4f}  sign={holdout_signs[wd]:+.0f}")
    hs = list(holdout_signs.values())
    hs_counts = {s: hs.count(s) for s in set(hs)}
    hs_majority_sign, hs_majority_n = max(hs_counts.items(), key=lambda kv: kv[1])
    holdout_plateau_pass = hs_majority_n >= 2
    print(f"  holdout signs: {holdout_signs}  majority={hs_majority_sign:+.0f} "
          f"({hs_majority_n}/3)  PASS: {holdout_plateau_pass}")

    # -------------------------------------------------------------- promotion verdict
    print("\n" + "=" * 78)
    print("PROMOTION VERDICT (holdout, frozen rule)")
    print("=" * 78)
    promote = bool(beats_bh and beats_ctrl and eth_survives and holdout_plateau_pass)
    verdict = "PROMOTE-eligible" if promote else "NEGATIVE"
    print(f"beats_buy_and_hold={beats_bh}  beats_control(sharpe/DD)={beats_ctrl}  "
          f"eth_falsification_survives={eth_survives}  holdout_plateau={holdout_plateau_pass}")
    print(f"VERDICT: {verdict}")
    if not promote:
        failed = [name for name, ok in (("beats buy_and_hold", beats_bh),
                                        ("beats control (sharpe/DD)", beats_ctrl),
                                        ("ETH falsification", eth_survives),
                                        ("holdout plateau", holdout_plateau_pass)) if not ok]
        print(f"Deciding clause(s): {', '.join(failed)}")

    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs}")
    print(f"max timestamp read anywhere in this branch: {max_ts} "
          f"(OOS bars read: {max_ts >= pd.Timestamp(r152_shared.OOS_START, tz=max_ts.tz)}, "
          f"only because eligible={eligible})")
    print(f"\n[{time.time() - t0:.0f}s]")

    result.update(dict(
        verdict=verdict, n_configs=n_configs, max_ts=max_ts,
        selected_wd=selected_wd, selected_rf=selected_rf,
        holdout_cand=holdout_cand, holdout_ctrl=holdout_ctrl, holdout_bh=holdout_bh,
        holdout_cand_high_fee=holdout_cand_high_fee,
        beats_bh=beats_bh, beats_ctrl=beats_ctrl, d_sharpe_holdout=d_sharpe_holdout,
        dd_improve_holdout=dd_improve_holdout,
        eth_cand=eth_cand, eth_ctrl=eth_ctrl, eth_bh=eth_bh, eth_survives=eth_survives,
        holdout_signs=holdout_signs, holdout_plateau_pass=holdout_plateau_pass,
        promote=promote,
    ))
    return result


if __name__ == "__main__":
    main()
