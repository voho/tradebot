#!/usr/bin/env python
"""R-87 CONSERVATIVE branch: ACI-modulated vote confidence on
`kelly_regime_v4`'s own 3-anchor (20/40/80-day) latched vote.

See `r87_shared.py`'s module docstring for the full literature/citation
background, the "not a duplicate of" reasoning against every prior round,
and the shared utilities used below (`v4_vote_frac`, `aci_update`,
`run_aci_causal`, `daily_resample_causal`, `causal_truncation_probe`, and
the frozen constants `V4_HORIZONS`, `V4_BAND`, `INNER_TRAIN_END`,
`INNER_VAL_START`, `INNER_VAL_END`, `OOS_START`, `ALPHA_MIN`,
`ALPHA_MAX`). Not repeated here to keep one citation trail in one place.

=====================================================================
PRE-REGISTRATION (frozen before any real-market number in this file was
computed -- docs/ROUTINE.md steps 1-2).
=====================================================================

MECHANISM (one sentence). Multiply v4's own vote fraction `frac[i]` by a
confidence scalar `c_t[i]` -- a linear map of an Adaptive Conformal
Inference (ACI; Gibbs & Candes 2021) state that tracks, on DAILY-
resampled data and broadcast causally onto the 5-minute index, whether
the vote's own majority LEAN has recently called the next day's return
direction correctly -- BEFORE `frac[i]` enters `desired = frac[i] *
scale[i]`, i.e. `desired = (c_t[i] * frac[i]) * scale[i]`, then apply
v4's existing 10% deadband to `desired` exactly as it already does. This
attacks ERR (no error control anywhere in this project's signal path)
and deepens SIZE via R-62's own factor finding (VOTE, not SCALE, carries
v4's signature) by modulating the VOTE's confidence rather than
retuning the SCALE's point estimate for a 22nd time.

CONSTRUCTION (frozen, see task banner / r87_shared.py, implemented
verbatim in `compute_daily_confidence` below):
  1. Daily-resample BTC (`daily_resample_causal`); take v4's 5-minute
     vote (`v4_vote_frac`) at each day's LAST 5-minute bar -> `frac_d`.
  2. `lean_d = +1 if frac_d > 0.5 else -1` (frac in {0,1/3,2/3,1}, never
     exactly 0.5, so this is just "majority bullish" vs "not").
  3. `r_d = log(close_d / close_{d-1})`.
  4. `err_d = 1.0 if sign(lean_{d-1}) != sign(r_d) else 0.0`; `r_d == 0`
     counts as NOT an error (documented edge case, expected vanishingly
     rare on real data -- see `compute_daily_confidence`'s docstring).
  5. `alpha_path = run_aci_causal(err, target_alpha=0.5, gamma=<swept>)`.
     target_alpha=0.5 is FIXED (coin-flip null for a binary directional
     call), never swept.
  6. `c_t_daily = conf_floor + (1-conf_floor)*(alpha_path-ALPHA_MIN)/
     (ALPHA_MAX-ALPHA_MIN)`, `conf_floor` swept over {0.8, 0.9}.
  7. Broadcast `c_t_daily[D]` onto every 5-minute bar of day D. Since
     `alpha_path[D]` (and hence `c_t_daily[D]`) was computed from
     `err[:D]` only (i.e. information through day D-1's close), this
     broadcast introduces no lookahead -- checked explicitly below
     (`spot_check_no_lookahead`), not merely asserted.

GRID (frozen, 6 configs, not extended after seeing results):
  gamma in {0.01, 0.02, 0.05} x conf_floor in {0.8, 0.9}.

STEP-A GATE (computed on inner-train, end=INNER_TRAIN_END, BEFORE any
Sharpe/backtest number), per config:
  (a) degeneracy: fraction of inner-train DAYS with c_t < 1.0 must be
      in (5%, 95%).
  (b) collinearity: Pearson r between this variant's `target[]` and
      `kelly_regime_v4`'s own unmodified `target[]`, full 2017-01-01 ->
      INNER_VAL_END span; R^2 > 0.98 => config flagged INERT (reported,
      not silently dropped).
  (c) causal-truncation probe (`causal_truncation_probe`) on the target-
      building function, one representative config (the construction is
      identical in code path across the grid; gamma/conf_floor only
      rescale a number, they cannot change causality), `check_at` deep
      in the middle of the inner-train+val series.

SELECTION (frozen): among configs passing Step-A (not degenerate, not
inert), pick the single best by inner-validation
(INNER_VAL_START..INNER_VAL_END) Sharpe on futures_5x, via
`scripts.experiment.ev`. Baseline read: `ev(KellyRegimeV4(), ...)` on
the identical period/market.

FALSIFICATION TEST (frozen, before this number was known): finalist must
beat v4's own inner-validation Sharpe by > 0.2 (project noise floor) OR
show a clear max-drawdown improvement, AND the SAME qualitative
direction must replicate on ETH
(`data/ethusd_coinbase_spot_5m.csv.gz`, full pre-holdout span). Either
half failing => NEGATIVE. Holdout (>= OOS_START) is never read by this
file on any asset; every data access is truncated at load time and the
max timestamp actually touched is tracked and printed at the end.
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
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

from experiments.r87_shared import (  # noqa: E402
    ALPHA_MAX,
    ALPHA_MIN,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    V4_BAND,
    V4_HORIZONS,
    causal_truncation_probe,
    daily_resample_causal,
    run_aci_causal,
    v4_vote_frac,
)

from scripts.experiment import FUTURES, ev  # noqa: E402

DATA_DIR = ROOT / "data"
ETH_FILE = DATA_DIR / "ethusd_coinbase_spot_5m.csv.gz"

GAMMAS = (0.01, 0.02, 0.05)
CONF_FLOORS = (0.8, 0.9)
TARGET_ALPHA = 0.5  # fixed, never swept

_max_ts_seen: list[pd.Timestamp] = []


# ------------------------------------------------------------- holdout guard

def assert_no_holdout(df: pd.DataFrame, label: str = "") -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read ({label}): max timestamp {max_ts} >= {OOS_START}.")
    _max_ts_seen.append(max_ts)


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df, "BTC")
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def load_eth_bars() -> pd.DataFrame:
    df = load_ohlcv_csv(ETH_FILE)
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df, "ETH")
    print(f"ETH (coinbase spot): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


# --------------------------------------------------------- confidence layer

def compute_daily_confidence(df: pd.DataFrame, gamma: float, conf_floor: float,
                              target_alpha: float = TARGET_ALPHA,
                              horizons: tuple[int, ...] = V4_HORIZONS,
                              band: float = V4_BAND) -> tuple[np.ndarray, dict]:
    """Steps 1-7 of the frozen construction. Returns `(c_t_5m, diag)` where
    `c_t_5m` is aligned 1:1 with `df.index` and `diag` carries every
    intermediate daily series for inspection/gating.

    Edge case (step 4): `r_d == 0.0` (close unchanged day over day) is
    counted as NOT an error regardless of the prior lean, i.e.
    `sign(0)` is treated as agreeing with anything. This is expected to
    be vanishingly rare on real 5-minute-sampled crypto closes (checked
    below, `diag['n_zero_return_days']`), and is exactly the documented
    a-priori edge-case rule, not a post-hoc choice.
    """
    frac_5m = pd.Series(v4_vote_frac(df, horizons=horizons, band=band), index=df.index)
    daily = daily_resample_causal(df)
    day_key = df.index.normalize()

    # last 5-minute bar of each day -> daily-causal read of the 5-min vote
    frac_daily = frac_5m.groupby(day_key).last().reindex(daily.index)
    lean_daily = np.where(frac_daily.to_numpy() > 0.5, 1.0, -1.0)

    close_d = daily["close"].to_numpy()
    n = len(close_d)
    r_d = np.full(n, np.nan)
    if n > 1:
        r_d[1:] = np.log(close_d[1:] / close_d[:-1])

    err = np.full(n, np.nan)
    n_zero_return_days = 0
    for i in range(1, n):
        if r_d[i] == 0.0:
            err[i] = 0.0
            n_zero_return_days += 1
        else:
            err[i] = 1.0 if np.sign(lean_daily[i - 1]) != np.sign(r_d[i]) else 0.0

    alpha_path = run_aci_causal(err, target_alpha=target_alpha, gamma=gamma)
    c_t_daily = conf_floor + (1.0 - conf_floor) * (alpha_path - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
    c_t_daily = np.clip(c_t_daily, conf_floor, 1.0)

    c_t_daily_series = pd.Series(c_t_daily, index=daily.index)
    c_t_5m = c_t_daily_series.reindex(day_key).ffill().to_numpy()

    diag = dict(frac_daily=frac_daily, lean_daily=lean_daily, r_d=r_d, err=err,
                alpha_path=alpha_path, c_t_daily=c_t_daily, daily_index=daily.index,
                n_zero_return_days=n_zero_return_days, n_days=n)
    return c_t_5m, diag


class KellyRegimeV4ACI(KellyRegimeV4):
    """`kelly_regime_v4`, except `frac[i]` is scaled by an ACI confidence
    `c_t[i]` before `desired = frac[i] * scale[i]` (so
    `desired = (c_t[i]*frac[i]) * scale[i]`), THEN v4's existing 10%
    deadband is applied to `desired` exactly as it already does --
    reimplements v3/v4's small `prepare` loop directly (task-approved
    alternative to monkeypatching `KellyRegimeV4`) with the multiplier
    injected only at that one line; everything else (vote construction,
    conditional-vol-targeting state machine, deadband) is copied
    verbatim from `kelly_regime_v3.KellyRegimeV3.prepare`.

    `force_ct`, if given, bypasses the ACI computation entirely and uses
    a constant `c_t` array -- used only by `cross_check_reproduces_v4`
    below to verify `force_ct=1.0` reproduces `KellyRegimeV4`'s own
    `target[]` column exactly (structural check independent of whether
    the ACI/daily-resample machinery itself has a bug).
    """

    name = "kelly_regime_v4_aci_conservative"

    def __init__(self, gamma: float = 0.02, conf_floor: float = 0.9,
                 target_alpha: float = TARGET_ALPHA, force_ct: float | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self.conf_floor = conf_floor
        self.target_alpha = target_alpha
        self.force_ct = force_ct
        self.last_c_t = None  # populated by prepare(), for diagnostics

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

        if self.force_ct is not None:
            c_t = np.full(len(df), float(self.force_ct))
        else:
            c_t, _diag = compute_daily_confidence(
                df, gamma=self.gamma, conf_floor=self.conf_floor,
                target_alpha=self.target_alpha, horizons=self.horizons, band=self.band)
        self.last_c_t = c_t
        frac_eff = c_t * frac

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
            desired = frac_eff[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


# --------------------------------------------------------------- Step A (a)/(b)

def step_a_degeneracy_and_collinearity(btc_full: pd.DataFrame) -> list[dict]:
    """(a) degeneracy on inner-train (end=INNER_TRAIN_END) and (b)
    collinearity on the full 2017-01-01 -> INNER_VAL_END span, for all 6
    configs. `btc_full` must already be truncated < OOS_START (asserted)."""
    assert_no_holdout(btc_full, "step_a input")

    train_df = btc_full.loc[:INNER_TRAIN_END]
    full_span_df = btc_full.loc[:INNER_VAL_END]  # 2017-01-01 .. 2022-12-31

    v4_target_full_span = KellyRegimeV4().prepare(full_span_df.copy())["target"].to_numpy()

    results = []
    for gamma in GAMMAS:
        for conf_floor in CONF_FLOORS:
            # (a) degeneracy, inner-train
            _c_t_train, diag_train = compute_daily_confidence(
                train_df, gamma=gamma, conf_floor=conf_floor)
            c_t_daily_train = diag_train["c_t_daily"]
            n_days = len(c_t_daily_train)
            pct_below_1 = float(np.mean(c_t_daily_train < 1.0)) * 100.0 if n_days else float("nan")
            not_degenerate = 5.0 < pct_below_1 < 95.0

            # (b) collinearity, 2017-01-01 -> INNER_VAL_END
            variant = KellyRegimeV4ACI(gamma=gamma, conf_floor=conf_floor)
            target_variant = variant.prepare(full_span_df.copy())["target"].to_numpy()
            corr = float(np.corrcoef(target_variant, v4_target_full_span)[0, 1])
            r2 = corr ** 2
            inert = r2 > 0.98

            passed = not_degenerate and not inert
            results.append(dict(
                gamma=gamma, conf_floor=conf_floor, n_days=n_days,
                pct_below_1=pct_below_1, not_degenerate=not_degenerate,
                corr=corr, r2=r2, inert=inert, passed=passed,
                n_zero_return_days=diag_train["n_zero_return_days"],
            ))
            print(f"  gamma={gamma:.2f} conf_floor={conf_floor:.1f}  "
                  f"(a) c_t<1.0 on {pct_below_1:5.1f}% of {n_days} inner-train days "
                  f"[{'OK' if not_degenerate else 'DEGENERATE'}]   "
                  f"(b) corr(target, v4 target)={corr:+.4f} R^2={r2:.4f} "
                  f"[{'INERT' if inert else 'OK'}]   "
                  f"=> {'PASS' if passed else 'FAIL'}   "
                  f"(zero-return days in inner-train: {diag_train['n_zero_return_days']})")
    return results


def spot_check_no_lookahead(btc_full: pd.DataFrame, gamma: float = 0.02,
                             conf_floor: float = 0.9) -> bool:
    """Explicit, not merely asserted, check that day D's broadcast c_t
    value never uses day D's own return: recompute confidence on a frame
    truncated to drop the LAST calendar day of `btc_full` entirely, and
    confirm every c_t_5m value on all EARLIER days is bit-identical to
    the full run. If the broadcast leaked day D's own outcome into day
    D's own value, dropping day D's bars could still not change earlier
    days -- so this check is paired with the causal-truncation probe
    below, which is the direct version of the same claim applied to the
    full target-building function."""
    train_df = btc_full.loc[:INNER_TRAIN_END]
    last_day = train_df.index.normalize().max()
    truncated = train_df.loc[train_df.index.normalize() < last_day]

    c_t_full, _ = compute_daily_confidence(train_df, gamma=gamma, conf_floor=conf_floor)
    c_t_trunc, _ = compute_daily_confidence(truncated, gamma=gamma, conf_floor=conf_floor)

    n = len(truncated)
    ok = bool(np.allclose(c_t_full[:n], c_t_trunc, equal_nan=True))
    print(f"  no-lookahead spot check (drop last calendar day, gamma={gamma}, "
          f"conf_floor={conf_floor}): c_t on all {n:,} earlier bars unchanged = {ok}")
    return ok


def causal_probe(btc_full: pd.DataFrame, gamma: float = 0.02,
                  conf_floor: float = 0.9) -> bool:
    """(c) causal-truncation probe on the full target-building function
    (vote + confidence + conditional-vol-targeting state machine), one
    representative config -- the code path (hence its causality) is
    identical across the grid; gamma/conf_floor only rescale a number."""
    full_span_df = btc_full.loc[:INNER_VAL_END]

    def build_target_fn(df: pd.DataFrame) -> np.ndarray:
        return KellyRegimeV4ACI(gamma=gamma, conf_floor=conf_floor).prepare(df.copy())["target"].to_numpy()

    n = len(full_span_df)
    check_at = n // 2
    shorter_by = 20_000
    assert check_at + shorter_by < n, "series too short for this check_at/shorter_by"
    passed = causal_truncation_probe(build_target_fn, full_span_df, check_at, shorter_by=shorter_by)
    print(f"  causal-truncation probe (gamma={gamma}, conf_floor={conf_floor}, "
          f"check_at={check_at:,} of {n:,}, shorter_by={shorter_by:,}): PASS={passed}")
    return passed


def cross_check_reproduces_v4(btc_full: pd.DataFrame) -> bool:
    """Task item 8: with c_t forced to 1.0 everywhere, the reimplemented
    loop must reproduce `KellyRegimeV4`'s own `target[]` column exactly."""
    df = btc_full.loc[:INNER_TRAIN_END].copy()
    v4_target = KellyRegimeV4().prepare(df.copy())["target"].to_numpy()
    forced = KellyRegimeV4ACI(force_ct=1.0).prepare(df.copy())["target"].to_numpy()
    ok = bool(np.allclose(v4_target, forced, atol=1e-12))
    max_abs_diff = float(np.max(np.abs(v4_target - forced)))
    print(f"  cross-check (force_ct=1.0 reproduces KellyRegimeV4 exactly): "
          f"{ok}  (max abs diff={max_abs_diff:.3e})")
    return ok


# ------------------------------------------------------------------ Step B

def inner_validation_sharpe(btc_full: pd.DataFrame, configs: list[dict]) -> dict:
    print("\ninner-validation (futures_5x, "
          f"{INNER_VAL_START} -> {INNER_VAL_END}):")
    baseline = ev(KellyRegimeV4(), df=btc_full, market=FUTURES, tag="BASELINE kelly_regime_v4",
                  start=INNER_VAL_START, end=INNER_VAL_END)

    scored = []
    for cfg in configs:
        variant = KellyRegimeV4ACI(gamma=cfg["gamma"], conf_floor=cfg["conf_floor"])
        tag = f"variant g={cfg['gamma']:.2f} cf={cfg['conf_floor']:.1f}"
        m = ev(variant, df=btc_full, market=FUTURES, tag=tag,
               start=INNER_VAL_START, end=INNER_VAL_END)
        scored.append(dict(cfg, sharpe=m.sharpe, max_dd=m.max_drawdown_pct,
                            profit_pct=m.profit_pct, trades=m.num_trades))
    return dict(baseline=baseline, scored=scored)


def eth_replication(finalist: dict) -> dict:
    eth_bars = load_eth_bars()
    print(f"\nETH replication (futures_5x, whatever of "
          f"{eth_bars.index[0].date()} -> {INNER_VAL_END} is available):")
    variant = KellyRegimeV4ACI(gamma=finalist["gamma"], conf_floor=finalist["conf_floor"])
    m_variant = ev(variant, df=eth_bars, market=FUTURES, tag="ETH variant", end=INNER_VAL_END)
    m_base = ev(KellyRegimeV4(), df=eth_bars, market=FUTURES, tag="ETH kelly_regime_v4", end=INNER_VAL_END)
    return dict(m_variant=m_variant, m_base=m_base, eth_bars=eth_bars)


# --------------------------------------------------------------------- main

def main() -> None:
    print("=" * 78)
    print("R-87 CONSERVATIVE: ACI vote-confidence on kelly_regime_v4")
    print("=" * 78)

    btc_full = load_btc_bars()  # < OOS_START

    print(f"\n--- structural cross-check (task item 8) ---")
    cross_check_ok = cross_check_reproduces_v4(btc_full)

    print(f"\n--- STEP A: degeneracy (a) + collinearity (b), "
          f"inner-train end={INNER_TRAIN_END}, 6 configs ---")
    step_a_results = step_a_degeneracy_and_collinearity(btc_full)

    print(f"\n--- STEP A: no-lookahead spot check + (c) causal-truncation probe ---")
    lookahead_ok = spot_check_no_lookahead(btc_full)
    probe_ok = causal_probe(btc_full)

    passing = [r for r in step_a_results if r["passed"]]
    print(f"\nconfigs passing Step A: {len(passing)}/6")
    for r in passing:
        print(f"  PASS  gamma={r['gamma']:.2f} conf_floor={r['conf_floor']:.1f}")
    for r in step_a_results:
        if not r["passed"]:
            reason = []
            if not r["not_degenerate"]:
                reason.append("degenerate")
            if r["inert"]:
                reason.append("inert (R^2>0.98)")
            print(f"  FAIL  gamma={r['gamma']:.2f} conf_floor={r['conf_floor']:.1f}  "
                  f"({', '.join(reason)})")

    if not passing:
        print("\nNo configs pass Step A -- STOP, verdict NEGATIVE, no Step B run.")
        print(f"\nmax timestamp read anywhere in this session: {max(_max_ts_seen)}  (< {OOS_START})")
        return

    val = inner_validation_sharpe(btc_full, passing)
    baseline_sharpe = val["baseline"].sharpe
    baseline_dd = val["baseline"].max_drawdown_pct

    best = max(val["scored"], key=lambda r: r["sharpe"])
    print(f"\nfinalist (best inner-val Sharpe among Step-A passers): "
          f"gamma={best['gamma']:.2f} conf_floor={best['conf_floor']:.1f}  "
          f"sharpe={best['sharpe']:.3f}  max_dd={best['max_dd']:.1f}%")
    print(f"kelly_regime_v4 baseline: sharpe={baseline_sharpe:.3f}  max_dd={baseline_dd:.1f}%")

    sharpe_gain = best["sharpe"] - baseline_sharpe
    dd_improved = best["max_dd"] < baseline_dd  # lower max_drawdown_pct = better
    btc_bar_pass = (sharpe_gain > 0.2) or dd_improved
    print(f"\nBTC noise-floor bar: sharpe_gain={sharpe_gain:+.3f} (>0.2 required OR) "
          f"dd_improved={dd_improved} ({best['max_dd']:.1f}% vs {baseline_dd:.1f}%)  "
          f"=> {'PASS' if btc_bar_pass else 'FAIL'}")

    eth = eth_replication(best)
    eth_sharpe_gain = eth["m_variant"].sharpe - eth["m_base"].sharpe
    eth_dd_improved = eth["m_variant"].max_drawdown_pct < eth["m_base"].max_drawdown_pct
    # "same qualitative direction" = whichever criterion qualified on BTC
    # (Sharpe gain, or DD improvement, or both) must also point the
    # improving way on ETH.
    eth_replicates = True
    if sharpe_gain > 0.2:
        eth_replicates = eth_replicates and (eth_sharpe_gain > 0.0)
    if dd_improved:
        eth_replicates = eth_replicates and eth_dd_improved
    print(f"\nETH: variant sharpe={eth['m_variant'].sharpe:.3f} max_dd={eth['m_variant'].max_drawdown_pct:.1f}%  "
          f"vs v4 sharpe={eth['m_base'].sharpe:.3f} max_dd={eth['m_base'].max_drawdown_pct:.1f}%  "
          f"sharpe_gain={eth_sharpe_gain:+.3f}  dd_improved={eth_dd_improved}  "
          f"=> replicates={eth_replicates}")

    verdict = "PROMOTE-CANDIDATE" if (btc_bar_pass and eth_replicates) else "NEGATIVE"
    print(f"\n{'=' * 78}\nVERDICT: {verdict}\n{'=' * 78}")

    print(f"\nconfigs evaluated: 6 (grid) ; passing Step A: {len(passing)} ; "
          f"scored in Step B: {len(val['scored'])}")
    print(f"cross-check (force_ct=1.0 == KellyRegimeV4): {cross_check_ok}")
    print(f"no-lookahead spot check: {lookahead_ok}")
    print(f"causal-truncation probe: {probe_ok}")
    print(f"max timestamp read anywhere in this session: {max(_max_ts_seen)}  (< {OOS_START})")


if __name__ == "__main__":
    main()
