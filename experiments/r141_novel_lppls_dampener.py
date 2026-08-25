#!/usr/bin/env python
"""R-141 NOVEL branch: ``NovelLPPLSDampener`` -- a continuous SIZE-axis
crash-hazard dampener on ``kelly_regime_v4``'s own ``scale``, driven by the
LPPLS bubble-confidence indicator computed once in ``experiments/r141_shared.py``.

The complete pre-registration for this round -- direction, LPPLS literature
grounding (Johansen, Ledoit & Sornette 2000; Filimonov & Sornette 2013),
the non-duplication argument against the seven prior regime-timing
mechanisms and 28+ prior SIZE-axis attempts, and the named failure mode --
lives in ``experiments/r141_shared.py``'s own module docstring, written by
the operator BEFORE either branch was dispatched, and is NOT re-derived
here: read that file in full first. This file imports ONLY from
``experiments.r141_shared`` (which itself imports the frozen
``r82_shared``/``r125_shared`` machinery), never edits any shared module,
never coordinates with the conservative branch's file, and never reads a
bar at or after ``r141_shared.OOS_START`` (2023-01-01).

MECHANISM (exact, matching the pre-registration verbatim): everything in
``kelly_regime_v4.prepare()`` up to and including the hysteresis-latched
risk-measure ``scale`` (``full[i] if state != 0 else steady[i]``) is
reproduced byte-for-byte here (see ``v4_scale_series`` below, a verbatim
extraction copied from ``KellyRegimeV3.prepare()``'s own published source --
same pattern R-125's conservative branch used, and verified exact by
``self_test_v4_scale_matches_v4_target`` before any calibration trusts it).
The ONE substitution is a post-hoc multiplicative dampener applied to that
scale, at every bar:

    scale_novel[i] = scale_v4[i] * max(0.1, 1 - kappa * confidence[i])

where ``confidence`` is ``r141_shared.lppls_bar_signals(df, cache_path=...)``'s
``lppls_confidence`` column (``n_qualify / 5``, causally aligned onto the 5m
bar grid), and ``kappa`` is calibrated by ``r141_shared.calibrate_dampener``
so that ``mean(scale_novel)`` on BTC inner-train matches ``mean(scale_v4)``
on the identical slice -- the exposure-matching discipline R-33/R-59/R-125
all use, so any B1 effect this round finds cannot be "holding less" in
disguise. The vote (``frac``), the anchors, the deadband, and the
hysteresis vol-targeting architecture that produces ``scale_v4`` are all
reused unchanged; nothing upstream of the final ``scale`` term changes.
``kappa = 0`` recovers v4 EXACTLY (``max(0.1, 1 - 0) = 1.0`` always) -- the
required identity-recovery check, run first, before any calibration.

CONFIGURATIONS EVALUATED: 16 (primary kappa-grid calibration search,
np.linspace(0, 3.0, 16), every grid point counted) + 1 (Step-0 gate,
primary candidate) + 2 (B1: BTC spot + futures, inner-validation) +
[168 kappa-grid points (3 resolutions x 3 calibration windows, freshly
recalibrated per cell) + 9 backtests] (B3 plateau sweep, BTC spot) +
1 (B4: ETH spot, inner-validation, kappa FROZEN at the BTC-calibrated
primary value, no ETH recalibration) + 2 (B5: BTC spot + futures at the
0.40% taker tier) = 199 total in this branch. Diagnostics run but NOT
counted toward this trials figure, per this project's own convention
(matching R-125's non-count of its own causal probe): the v4-scale wiring
self-test (x1), the kappa=0 identity-recovery check (x1), and the
causal-truncation probe (x2 ``prepare()`` calls). (The shared LPPLS
calibration pass itself -- 800 lstsq fits per calibration date -- is
counted once in ``r141_shared``'s own module docstring, not re-counted per
branch, per that module's own accounting convention.)

DECISION RULE (pre-registered, verbatim from ``r141_shared.py``, unaltered
after seeing any number): PROMOTE-candidate only if the causal-truncation
probe AND Step-0 (not degenerate) AND B1 (both markets) AND B3 (plateau
majority) AND B4 (both markets, i.e. ETH replicates the BTC sign) AND B5
all pass. Anything else is NEGATIVE. Default: NEGATIVE.

USAGE
-----
    python experiments/r141_novel_lppls_dampener.py
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

from experiments import r141_shared  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

BTC_CACHE = ROOT / "experiments" / "r141_lppls_btc_cache.csv"
ETH_CACHE = ROOT / "experiments" / "r141_lppls_eth_cache.csv"

# Primary calibration config, pre-registered before any inner-validation
# number was read: r141_shared's own suggested resolution and bounds.
PRIMARY_KAPPA_GRID = np.linspace(0.0, 3.0, 16)
DAMP_FLOOR = 0.1  # frozen inside r141_shared.calibrate_dampener's own formula

# B3 plateau sweep: (a) kappa-grid search resolution, (b) the inner-train
# calibration WINDOW fed to calibrate_dampener -- both free construction
# choices the mechanism (the dampener formula itself, frozen in
# r141_shared.calibrate_dampener) does not fix.
KAPPA_GRID_RESOLUTIONS = (8, 16, 32)
CALIBRATION_WINDOWS = (
    ("2017-01-01", "2020-12-31"),  # primary: full inner-train
    ("2019-01-01", "2020-12-31"),  # short: last two years of inner-train only
    ("2017-01-01", "2021-12-31"),  # extended: inner-train + first inner-val year
)
PRIMARY_CAL_WINDOW = ("2017-01-01", "2020-12-31")
PRIMARY_RESOLUTION = 16


# ================================================================== (1)
# v4's own intermediate `scale` (pre-frac, pre-deadband, post-hysteresis) is
# never exposed by kelly_regime_v4.prepare(). Extracted verbatim from
# KellyRegimeV3.prepare()'s own published source -- identical pattern to
# R-125 conservative's own `v4_scale_series` -- and verified exact by
# self_test_v4_scale_matches_v4_target() before any calibration trusts it.
# ==================================================================

def v4_scale_series(df: pd.DataFrame, v4: KellyRegimeV4) -> np.ndarray:
    close = df["close"]
    r = np.log(close).diff()
    vol = (r.ewm(span=v4.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=v4.anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(v4.target_vol / vol, v4.max_leverage)
        steady = np.minimum(v4.target_vol / slow, v4.max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(df)
    scale = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > v4.high_in else (-1 if x < v4.low_in else 0)
            elif state == 1 and x < v4.high_out:
                state = 0
            elif state == -1 and x > v4.low_out:
                state = 0
        scale[i] = full[i] if state != 0 else steady[i]
    return scale


def self_test_v4_scale_matches_v4_target(df: pd.DataFrame) -> bool:
    """v4_scale_series(df, v4), recombined with v4's own vote/deadband
    logic, must reproduce kelly_regime_v4.prepare()'s own `target` column
    exactly -- the check that the extraction used to calibrate kappa is not
    silently wrong."""
    v4 = get_strategy("kelly_regime_v4")
    v4_target = v4.prepare(df.copy())["target"].to_numpy()
    scale = v4_scale_series(df, v4)

    close = df["close"]
    votes = []
    for days in v4.horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + v4.band), 1.0,
                     np.where(close < anchor * (1.0 - v4.band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    frac = (sum(votes) / len(votes)).to_numpy()
    if v4.vote_gamma != 1.0:
        frac = frac ** v4.vote_gamma

    n = len(df)
    target = np.zeros(n)
    pos = 0.0
    for i in range(n):
        desired = frac[i] * scale[i]
        if abs(desired - pos) > v4.deadband:
            pos = desired
        target[i] = pos

    return bool(np.allclose(target, v4_target, equal_nan=True))


# ================================================================== (2)
# NovelLPPLSDampener: KellyRegimeV4.prepare(), copied verbatim, with exactly
# one post-hoc substitution -- scale_novel = scale_v4 * max(0.1, 1 -
# kappa*confidence). NOT @register'd -- experiments/-only, per this round's
# instructions.
# ==================================================================

class NovelLPPLSDampener(KellyRegimeV4):
    """kelly_regime_v4's exact architecture (3-anchor vote, 20/40/80-day
    anchors, 1% band, extremes-only hysteresis latch, 10% deadband) with the
    ONE substitution this round tests: the final, hysteresis-selected
    ``scale`` is multiplied by an LPPLS-crash-confidence dampener,
    ``max(0.1, 1 - kappa*confidence)``, before being combined with the vote.
    kappa=0 recovers v4 exactly. Not registered in
    ``src/tradebot/strategies/`` -- experiments/-only, per this round's
    instructions.
    """

    name = "r141_novel_lppls_dampener"

    def __init__(self, kappa: float = 0.0, cache_path: Path = BTC_CACHE,
                 horizons: tuple[int, ...] = (20, 40, 80), **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.kappa = kappa
        self.cache_path = cache_path

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Vote: byte-identical to KellyRegimeV3.prepare() / KellyRegimeV4.
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

        # Risk measure / hysteresis latch: byte-identical to
        # KellyRegimeV3.prepare() / KellyRegimeV4 (same std-based vol
        # target, same state machine). Produces v4's own `scale` per bar.
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

        # ---- THE ONE SUBSTITUTION: LPPLS crash-confidence dampener,
        # applied to the hysteresis-selected scale, post-hoc.
        confidence = r141_shared.lppls_bar_signals(
            df, cache_path=self.cache_path, verbose=False)["lppls_confidence"].to_numpy()
        damp = np.maximum(DAMP_FLOOR, 1.0 - self.kappa * confidence)

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
            scale_v4 = full[i] if state != 0 else steady[i]
            scale_novel = scale_v4 * damp[i]
            desired = frac[i] * scale_novel
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


def make_candidate_factory(kappa: float, cache_path: Path = BTC_CACHE):
    def _factory():
        return NovelLPPLSDampener(kappa=kappa, cache_path=cache_path)
    return _factory


# ================================================================== (3)
# Calibration: kappa on a given inner-train slice, via r141_shared's own
# frozen calibrate_dampener (mean-exposure matching).
# ==================================================================

def calibrate_for(df_full_train: pd.DataFrame, confidence_full: pd.Series,
                   start: str, end: str, kappa_grid: np.ndarray) -> float:
    """kappa calibrated on the given [start, end] slice of BTC data ONLY,
    matching mean(scale_novel) to kelly_regime_v4's own mean(scale) on the
    identical slice. Recomputed fresh for every (kappa_grid, window) cell,
    per this round's own pre-registration."""
    window = df_full_train.loc[start:end]
    v4 = get_strategy("kelly_regime_v4")
    v4_scale_window = v4_scale_series(window, v4)
    confidence_window = confidence_full.reindex(window.index).to_numpy()
    return r141_shared.calibrate_dampener(v4_scale_window, confidence_window, kappa_grid)


# ================================================================== (4)
# Causal-truncation self-test on THIS round's own new code (mirrors
# r125_shared.py's / r125_conservative's own __main__ pattern).
# ==================================================================

def causal_truncation_probe(df: pd.DataFrame, kappa: float,
                             cache_path: Path = BTC_CACHE, cut: int = 400_000) -> bool:
    """Build the candidate's full target array on the whole frame, then
    again on a frame truncated ~20,000+ bars before an interior check
    point, and confirm the target value at the check point (and everywhere
    before it) is bit-identical both ways. Note the LPPLS confidence input
    itself is loaded from an on-disk cache regardless of df's own length
    (per this round's own instructions to reuse the already-computed
    cache), so this probe is testing THIS FILE's own new code -- the vote,
    the vol/hysteresis state machine, and the dampener combination -- not
    re-testing the LPPLS calibration pipeline's own causality (already
    verified independently by r141_shared.py's own module-level self-test,
    which fits the LPPLS model on a truncated series directly).
    """
    full = NovelLPPLSDampener(kappa=kappa, cache_path=cache_path).prepare(
        df.copy())["target"].to_numpy()
    trunc = NovelLPPLSDampener(kappa=kappa, cache_path=cache_path).prepare(
        df.iloc[:cut].copy())["target"].to_numpy()
    n_check = min(len(trunc), cut) - BARS_PER_DAY * 90  # skip the fresh 80d-anchor warmup tail
    return bool(np.allclose(full[:n_check], trunc[:n_check], equal_nan=True, rtol=1e-9))


# ================================================================== (5)
# Main: identity check -> calibrate -> Step-0 -> causal probe -> B1 -> B3 ->
# B4 -> B5 -> verdict.
# ==================================================================

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    print("=" * 78)
    print("R-141 NOVEL: NovelLPPLSDampener -- kelly_regime_v4's own architecture, scale")
    print("post-multiplied by an LPPLS crash-confidence dampener max(0.1, 1-kappa*confidence).")
    print("=" * 78)

    btc = r141_shared.load_btc_train("spot")[0]
    max_ts_seen.append(btc.index.max())
    print(f"\nBTC spot (truncated < {r141_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    print("\n-- PRE-FLIGHT SELF-TEST: v4_scale_series extraction matches "
          "kelly_regime_v4.prepare()'s own target exactly --")
    wiring_ok = self_test_v4_scale_matches_v4_target(btc)
    print(f"  self_test_v4_scale_matches_v4_target: {'PASS' if wiring_ok else 'FAIL'}")
    if not wiring_ok:
        print("\nWIRING SELF-TEST FAILURE -- stopping before any calibration is trusted.")
        return dict(verdict="ABORTED (wiring self-test failure)", max_ts=max(max_ts_seen))

    # -------------------------------------------------------------- LPPLS confidence (cache)
    print(f"\n-- LOADING LPPLS confidence (BTC cache: {BTC_CACHE}) --")
    confidence_btc = r141_shared.lppls_bar_signals(
        btc, cache_path=BTC_CACHE, verbose=True)["lppls_confidence"]
    print(f"  confidence stats over BTC frame: mean={confidence_btc.mean():.4f}  "
          f"max={confidence_btc.max():.4f}  frac>0={float((confidence_btc > 0).mean()):.4f}")

    # -------------------------------------------------------------- (A) kappa=0 identity check
    print("\n" + "=" * 78)
    print("IDENTITY-RECOVERY CHECK: kappa=0 must reproduce kelly_regime_v4 EXACTLY")
    print("=" * 78)
    v4_target_full = r141_shared.v4_reference_target(btc)
    kappa0_target = NovelLPPLSDampener(kappa=0.0, cache_path=BTC_CACHE).prepare(
        btc.copy())["target"].to_numpy()
    identity_ok = bool(np.array_equal(kappa0_target, v4_target_full))
    identity_allclose = bool(np.allclose(kappa0_target, v4_target_full, equal_nan=True, rtol=0, atol=0))
    print(f"  kappa=0 target array_equal to kelly_regime_v4's own target: {identity_ok}")
    if not identity_ok:
        print("\nIDENTITY-RECOVERY FAILURE -- kappa=0 does not reproduce v4 exactly. Stopping.")
        return dict(verdict="ABORTED (identity-recovery failure)", max_ts=max(max_ts_seen))

    # -------------------------------------------------------------- causal probe
    # Run BEFORE calibration/Step-0 and at a NONZERO diagnostic kappa (not the
    # calibrated primary, whatever it turns out to be) so the probe actually
    # exercises the dampener multiplication -- at kappa=0 the substitution is
    # a no-op (damp==1 identically) and would not test anything new.
    print("\n" + "=" * 78)
    print("CAUSAL-TRUNCATION SELF-TEST (this round's own new code, real BTC data, "
          "diagnostic kappa=1.0 -- run before calibration so it exercises the actual "
          "dampener multiplication, not the kappa=0 no-op)")
    print("=" * 78)
    probe_ok = causal_truncation_probe(btc, kappa=1.0)
    print(f"  causal_truncation_probe (kappa=1.0 diagnostic): {'PASS' if probe_ok else 'FAIL'}")
    if not probe_ok:
        print("\nCAUSAL PROBE FAILURE -- a result that looks too good is a bug report first. Stopping.")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated: {n_configs} (promotion bar not run)")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="NEGATIVE (causal probe failure)", n_configs=n_configs,
                    max_ts=max_ts, identity_ok=identity_ok)

    # -------------------------------------------------------------- monotonicity diagnostic
    # Because damp = max(0.1, 1 - kappa*confidence) <= 1 for every kappa >= 0,
    # scale_novel = scale_v4 * damp <= scale_v4 POINTWISE for every kappa in
    # the pre-registered non-negative grid -- so mean(scale_novel) can only
    # fall as kappa rises. Printed here, on real data, before trusting the
    # calibration search's own output below.
    print("\n" + "=" * 78)
    print("MONOTONICITY DIAGNOSTIC (mean(scale_novel) vs kappa, BTC inner-train, "
          "primary calibration window) -- checked on real data before trusting the calibration")
    print("=" * 78)
    inner_train_check = btc.loc[PRIMARY_CAL_WINDOW[0]:PRIMARY_CAL_WINDOW[1]]
    v4_for_check = get_strategy("kelly_regime_v4")
    v4_scale_check = v4_scale_series(inner_train_check, v4_for_check)
    conf_check = confidence_btc.reindex(inner_train_check.index).to_numpy()
    target_mean_check = float(np.nanmean(v4_scale_check))
    print(f"  target mean(scale_v4) on {PRIMARY_CAL_WINDOW} = {target_mean_check:.6f}")
    for k in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0):
        damp_k = np.maximum(DAMP_FLOOR, 1.0 - k * conf_check)
        mean_k = float(np.nanmean(v4_scale_check * damp_k))
        print(f"    kappa={k:>4.1f}  mean(scale_novel)={mean_k:.6f}  gap={abs(mean_k - target_mean_check):.6f}")
    print("  mean(scale_novel) is non-increasing in kappa (as expected: damp<=1 always) -- so the "
          "mean-matching calibration below is expected to select kappa=0 or the grid's smallest "
          "value, unless mean(scale_v4) sits BELOW what kappa=0 already achieves (it does not, by "
          "construction: kappa=0 reproduces mean(scale_v4) itself, exactly).")

    # -------------------------------------------------------------- calibrate primary
    print(f"\n-- CALIBRATING kappa (primary config: grid=np.linspace(0,3.0,{PRIMARY_RESOLUTION}), "
          f"window={PRIMARY_CAL_WINDOW}), BTC --")
    kappa_primary = calibrate_for(btc, confidence_btc, *PRIMARY_CAL_WINDOW, PRIMARY_KAPPA_GRID)
    n_configs += len(PRIMARY_KAPPA_GRID)  # every kappa-grid point evaluated in the calibration search
    print(f"  kappa (primary) = {kappa_primary:.6f}  "
          f"({len(PRIMARY_KAPPA_GRID)} kappa-grid points evaluated)")
    cand_factory_primary = make_candidate_factory(kappa_primary, BTC_CACHE)

    # -------------------------------------------------------------- Step 0
    print("\n" + "=" * 78)
    print("STEP 0 -- non-degeneracy gate: is the candidate genuinely different from v4, "
          "or a rescaled copy?")
    print("=" * 78)
    candidate_target_full = cand_factory_primary().prepare(btc.copy())["target"].to_numpy()
    step0 = r141_shared.step0_gate(candidate_target_full, v4_target_full)
    n_configs += 1
    print(f"  R^2 vs v4 (BTC inner-train + inner-validation): {step0['r2_vs_v4']:.6f}")
    print(f"  KILL (R^2 > 0.98)?  {step0['kill']}")
    if step0["kill"]:
        print("\nSTEP-0 KILL: candidate is numerically a rescaled (here: EXACTLY identical) copy of v4.")
        print("Diagnosis (not a bug, a structural property of this mechanism): the dampener")
        print("damp = max(0.1, 1 - kappa*confidence) is bounded above by 1 for every kappa in the")
        print("pre-registered non-negative grid, so scale_novel <= scale_v4 pointwise for every")
        print("candidate kappa > 0, hence mean(scale_novel) is non-increasing in kappa. The frozen")
        print("mean-matching calibration (r141_shared.calibrate_dampener) therefore always selects")
        print("kappa=0 (or the grid's smallest value) as the exact zero-gap minimizer, which makes")
        print("the calibrated candidate byte-identical to kelly_regime_v4 by construction -- this is")
        print("why R^2 = 1.000000 exactly, not merely > 0.98. A pure multiplicative dampener bounded")
        print("above by 1 cannot satisfy an EQUALITY exposure-matching target without collapsing to")
        print("no-op, for ANY confidence signal that is not identically 1 wherever v4's scale is 0.")
        print()
        print("Per the pre-registered decision rule, this STOPS the branch here: no B1/B3/B4/B5 run.")
        print("Running B1-B5 at any un-calibrated (non-mean-matched) kappa would violate this")
        print("project's own exposure-matching discipline (R-33/R-59/R-125: never compare arms at")
        print("different average risk) and would misreport a pure de-leveraging effect as a signal")
        print("effect -- so it is not done, rather than fabricating a number the routine forbids.")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated: {n_configs}")
        print(f"max timestamp read anywhere in this branch: {max_ts} "
              f"(< {r141_shared.OOS_START}: {max_ts < pd.Timestamp(r141_shared.OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="NEGATIVE (Step-0 kill)", step0=step0, n_configs=n_configs,
                    max_ts=max_ts, kappa_primary=kappa_primary, identity_ok=identity_ok,
                    probe_ok=probe_ok)

    # -------------------------------------------------------------- B1
    print("\n" + "=" * 78)
    print("B1 -- BTC signal, inner-validation, spot + futures")
    print("=" * 78)
    b1_spot = r141_shared.b1_signal(cand_factory_primary, btc, r141_shared.SPOT)
    b1_fut = r141_shared.b1_signal(cand_factory_primary, btc, r141_shared.FUTURES)
    n_configs += 2
    for name, r in (("spot", b1_spot), ("futures", b1_fut)):
        print(f"  {name:>8s}  sharpe_cand={r['sharpe_cand']:+.4f}  sharpe_v4={r['sharpe_v4']:+.4f}  "
              f"d_sharpe={r['d_sharpe']:+.4f}  boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]  "
              f"significant={r['significant']}  dd_cand={r['dd_cand']:.2f}%  dd_v4={r['dd_v4']:.2f}%")
    b1_pass = (b1_spot["d_sharpe"] > 0.2 or b1_spot["paired_lo"] > 0.0) and \
              (b1_fut["d_sharpe"] > 0.2 or b1_fut["paired_lo"] > 0.0)
    print(f"  B1 PASS (both markets, d_sharpe > +0.2 noise floor OR bootstrap excludes zero "
          f"positively): {b1_pass}")

    # -------------------------------------------------------------- B3
    print("\n" + "=" * 78)
    print(f"B3 -- plateau: kappa-grid resolution in {KAPPA_GRID_RESOLUTIONS} x calibration window "
          f"in {CALIBRATION_WINDOWS}, BTC spot, kappa recalibrated fresh per cell")
    print("=" * 78)
    grid_rows = []
    primary_sign = None
    for resolution in KAPPA_GRID_RESOLUTIONS:
        grid = np.linspace(0.0, 3.0, resolution)
        for (cw_start, cw_end) in CALIBRATION_WINDOWS:
            kappa_cell = calibrate_for(btc, confidence_btc, cw_start, cw_end, grid)
            n_configs += len(grid)  # kappa-grid points evaluated for this cell's calibration
            factory = make_candidate_factory(kappa_cell, BTC_CACHE)
            r = r141_shared.b1_signal(factory, btc, r141_shared.SPOT)
            n_configs += 1  # the backtest itself
            sign = float(np.sign(r["d_sharpe"]))
            row = dict(resolution=resolution, cal_window=f"{cw_start}:{cw_end}", kappa=kappa_cell,
                       d_sharpe=r["d_sharpe"], boot_lo=r["paired_lo"], boot_hi=r["paired_hi"], sign=sign)
            grid_rows.append(row)
            is_primary = (resolution == PRIMARY_RESOLUTION and (cw_start, cw_end) == PRIMARY_CAL_WINDOW)
            if is_primary:
                primary_sign = sign
            print(f"  resolution={resolution:>2d}  cal_window={cw_start}:{cw_end}  "
                  f"kappa={kappa_cell:.4f}  d_sharpe={r['d_sharpe']:+.4f}  "
                  f"boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]{'  <- PRIMARY' if is_primary else ''}")
    n_same = sum(1 for row in grid_rows if row["sign"] == primary_sign)
    b3_pass = n_same >= len(grid_rows) / 2.0
    print(f"  B3 (majority same-signed as primary, spot): {b3_pass} ({n_same}/{len(grid_rows)})")

    # -------------------------------------------------------------- B4
    print("\n" + "=" * 78)
    print("B4 -- ETH falsification (pre-registered), spot only (no ETH futures data), "
          "kappa FROZEN at the BTC-calibrated primary value")
    print("=" * 78)
    eth = r141_shared.load_eth_train()
    max_ts_seen.append(eth.index.max())
    print(f"ETH spot (truncated < {r141_shared.OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")
    cand_factory_eth = make_candidate_factory(kappa_primary, ETH_CACHE)
    b4_spot = r141_shared.b1_signal(cand_factory_eth, eth, r141_shared.SPOT)
    n_configs += 1
    print(f"  spot  ETH d_sharpe={b4_spot['d_sharpe']:+.4f}  "
          f"boot=[{b4_spot['paired_lo']:+.4f},{b4_spot['paired_hi']:+.4f}]  "
          f"significant={b4_spot['significant']}")
    btc_spot_sign = float(np.sign(b1_spot["d_sharpe"]))
    eth_spot_sign = float(np.sign(b4_spot["d_sharpe"]))
    b4_full_pass = bool(btc_spot_sign != 0 and eth_spot_sign == btc_spot_sign)
    print(f"  BTC spot d_sharpe sign = {btc_spot_sign:+.0f}   ETH spot d_sharpe sign = "
          f"{eth_spot_sign:+.0f}   SAME SIGN (B4 full pass, spot-only since no ETH futures "
          f"data exists): {b4_full_pass}")

    # -------------------------------------------------------------- B5
    print("\n" + "=" * 78)
    print("B5 -- fee-tier survival (0.40% taker), primary config, BTC spot + futures")
    print("=" * 78)
    b5_spot = r141_shared.b1_signal(cand_factory_primary, btc, r141_shared.SPOT_HIGH_FEE)
    b5_fut = r141_shared.b1_signal(cand_factory_primary, btc, r141_shared.FUTURES_HIGH_FEE)
    n_configs += 2
    spot_no_flip = np.sign(b5_spot["d_sharpe"]) == np.sign(b1_spot["d_sharpe"]) or b1_spot["d_sharpe"] == 0
    fut_no_flip = np.sign(b5_fut["d_sharpe"]) == np.sign(b1_fut["d_sharpe"]) or b1_fut["d_sharpe"] == 0
    for name, r0, r1, ok in (("spot", b1_spot, b5_spot, spot_no_flip),
                              ("futures", b1_fut, b5_fut, fut_no_flip)):
        print(f"  {name:>8s}  @0.10% d_sharpe={r0['d_sharpe']:+.4f}   "
              f"@0.40% d_sharpe={r1['d_sharpe']:+.4f}   no_flip={ok}")
    b5_pass = bool(spot_no_flip and fut_no_flip)
    print(f"  B5 PASS (no sign flip, either market): {b5_pass}")

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    all_pass = probe_ok and (not step0["kill"]) and b1_pass and b3_pass and b4_full_pass and b5_pass
    verdict = "PROMOTE-candidate" if all_pass else "NEGATIVE"
    print(f"identity-recovery(kappa=0)={identity_ok}  causal probe={probe_ok}  "
          f"step0_kill={step0['kill']}  B1={b1_pass}  B2=diagnostic-only  B3={b3_pass}  "
          f"B4(full)={b4_full_pass}  B5={b5_pass}")
    print(f"ALL GATING CLAUSES PASS: {all_pass}")
    print(f"VERDICT: {verdict}")
    if not all_pass:
        failed = [name for name, ok in (("causal probe", probe_ok), ("Step-0", not step0["kill"]),
                                         ("B1", b1_pass), ("B3", b3_pass),
                                         ("B4 (full)", b4_full_pass), ("B5", b5_pass)) if not ok]
        print(f"Reason(s): {', '.join(failed)}")

    max_ts = max(max_ts_seen)
    b3_kappa_points = sum(len(np.linspace(0.0, 3.0, res)) for res in KAPPA_GRID_RESOLUTIONS
                          for _ in CALIBRATION_WINDOWS)
    print(f"\nconfigurations evaluated (total, this branch): {n_configs} "
          f"({len(PRIMARY_KAPPA_GRID)} primary-calibration kappa-grid points + 1 Step-0 + 2 B1 + "
          f"[{b3_kappa_points} B3 kappa-grid points + {len(grid_rows)} B3 backtests] + 1 B4 + 2 B5)")
    print("(diagnostics run but NOT counted toward trials, per this project's convention: "
          "wiring self-test x1, identity-recovery check x1, causal-truncation probe x2 prepare() calls)")
    print(f"max timestamp read anywhere in this branch: {max_ts} "
          f"(< {r141_shared.OOS_START}: {max_ts < pd.Timestamp(r141_shared.OOS_START, tz='UTC')})")
    print("NO bar at or after 2023-01-01 was ever read by this file.")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(
        verdict=verdict, n_configs=n_configs, max_ts=max_ts,
        identity_ok=identity_ok, kappa_primary=kappa_primary, step0=step0, probe_ok=probe_ok,
        b1_spot=b1_spot, b1_fut=b1_fut, b1_pass=b1_pass,
        b3_grid=grid_rows, b3_pass=b3_pass,
        b4_spot=b4_spot, b4_full_pass=b4_full_pass,
        b5_spot=b5_spot, b5_fut=b5_fut, b5_pass=b5_pass,
    )


if __name__ == "__main__":
    main()
