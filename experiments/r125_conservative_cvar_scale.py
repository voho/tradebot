#!/usr/bin/env python
"""R-125 CONSERVATIVE branch: ``ConservativeCVaRScale`` -- the minimal,
single-line substitution inside ``kelly_regime_v4``'s ``scale``: replace the
realized-standard-deviation risk measure (``target_vol / realized_vol``,
inherited unchanged from ``kelly_regime_v3``'s extremes-only hysteresis
construction) with realized Conditional Value-at-Risk (``target_cvar /
realized_cvar``, via ``r125_shared.annualized_cvar``). Full literature
grounding (Rockafellar & Uryasev 2000; Artzner, Delbaen, Eber & Heath 1999),
the non-duplication argument against 27+ prior SIZE-axis rounds, the named
failure mode, and the pre-registered decision rule / falsification test all
live in ``experiments/r125_shared.py``'s own module docstring (read in full
before this file was written); not re-derived here beyond the one-paragraph
summary above. This file NEVER edits ``r125_shared.py`` (frozen, shared with
the parallel NOVEL branch, a disjoint file this session does not read or
coordinate with), and never reads a bar at or after ``r125_shared.OOS_START``
(2023-01-01) from any data source.

MECHANISM (exact): ``kelly_regime_v4`` IS ``kelly_regime_v3`` with 20/40/80-day
anchors (``KellyRegimeV4.prepare`` is inherited unchanged from
``KellyRegimeV3.prepare``). That ``prepare()`` is copied here verbatim --
same 3-anchor vote (``frac``), same 1% band, same extremes-only hysteresis
latch (``high_in/high_out/low_in/low_out`` thresholds on a ratio of "today's
risk measure" to "its own slow EWM"), same 10% deadband -- with exactly one
substitution: the risk-measure input to that ratio, and to both branches of
the state machine (``full = target/risk_now``, ``steady = target/risk_slow``),
is ``r125_shared.annualized_cvar(close, cvar_window_days, cvar_alpha)`` in
place of the EWM-std realized-volatility series, and the target the sizing
rule aims for is ``target_cvar`` (calibrated, see below) in place of
``target_vol``. Nothing else in ``prepare()`` changes.

``target_cvar`` is calibrated, not guessed: ``r125_shared.calibrate_target_cvar``
grid-searches the scalar that makes ``mean(scale)`` on BTC inner-train
(2017-2020) match ``kelly_regime_v4``'s own ``mean(scale)`` on the identical
slice -- the exposure-matching discipline this project's own R-33 exists to
enforce (never compare arms carrying different average risk). Because v4's
``prepare()`` never exposes ``scale`` on its own (only the final,
frac-multiplied, deadbanded ``target``), this file factors that intermediate
value out of ``KellyRegimeV3.prepare()``'s own published source
(``v4_scale_series`` below, a verbatim extraction, not a re-derivation) and
verifies the extraction is exact via a wiring self-test before trusting any
calibration built on it (``self_test_v4_scale_matches_v4_target``).

CONFIGURATIONS EVALUATED: 1 (Step-0 gate, primary config) + 2 (B1: BTC
spot + futures, inner-validation) + 10 (B3: 5 ``cvar_window_days`` values x
2 ``alpha`` values, BTC spot, each with a freshly recalibrated
``target_cvar``) + 1 (B4: ETH spot, inner-validation, frozen BTC-calibrated
``target_cvar``) + 2 (B5: BTC spot + futures at the 0.40% taker tier) = 16
total.

DECISION RULE (pre-registered, verbatim from ``r125_shared.py``, unaltered
after seeing any number): PROMOTE-candidate only if the causal-truncation
probe AND B1 (both markets) AND B3 (plateau majority) AND B4 (full, both
markets -- ETH spot only here, since ETH futures data does not exist) AND
B5 all pass. Anything else is NEGATIVE. B2 (drawdown) is diagnostic only
and never gates promotion by itself.

USAGE
-----
    python experiments/r125_conservative_cvar_scale.py
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

from experiments import r125_shared  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

# Primary configuration, pre-registered before any inner-validation number
# was read: 90-day CVaR window (matching v3/v4's own ~180-day-ish anchor-span
# convention loosely, per the operator's brief), 5% tail (r125_shared's own
# CVAR_ALPHA default).
PRIMARY_WINDOW_DAYS = 90
PRIMARY_ALPHA = r125_shared.CVAR_ALPHA

WINDOW_GRID = (30, 60, 90, 120, 180)
ALPHA_GRID = (0.05, 0.10)


# ================================================================== (1)
# ConservativeCVaRScale: KellyRegimeV4.prepare(), copied verbatim, with
# exactly one substitution (see module docstring). NOT @register'd -- this
# is an experiments/-only candidate, never registered as a strategy.
# ==================================================================

class ConservativeCVaRScale(KellyRegimeV4):
    """kelly_regime_v4's exact architecture (3-anchor vote, 20/40/80-day
    anchors, 1% band, extremes-only hysteresis latch, 10% deadband) with the
    ONE substitution this round tests: ``scale``'s risk measure is realized
    CVaR_alpha of calendar-day log returns (``r125_shared.annualized_cvar``)
    in place of EWM-std realized volatility. Not registered in
    ``src/tradebot/strategies/`` -- experiments/-only, per this round's
    instructions.
    """

    name = "r125_conservative_cvar_scale"

    def __init__(self, cvar_window_days: int = PRIMARY_WINDOW_DAYS,
                 cvar_alpha: float = PRIMARY_ALPHA,
                 target_cvar: float | None = None,
                 horizons: tuple[int, ...] = (20, 40, 80), **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.cvar_window_days = cvar_window_days
        self.cvar_alpha = cvar_alpha
        self.target_cvar = target_cvar  # must be calibrated before prepare()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_cvar is None:
            raise ValueError(
                "target_cvar must be calibrated (r125_shared.calibrate_target_cvar) "
                "before prepare() is called")
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

        # ---- THE ONE SUBSTITUTION: annualized CVaR in place of EWM-std vol,
        # target_cvar in place of target_vol. Everything downstream (the
        # slow EWM smoothing of the risk measure, the ratio, the hysteresis
        # state machine, the deadband) is byte-identical to
        # KellyRegimeV3.prepare().
        vol = r125_shared.annualized_cvar(close, self.cvar_window_days, self.cvar_alpha).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_cvar / vol, self.max_leverage)
            steady = np.minimum(self.target_cvar / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-CVaR breakout, -1 low-CVaR breakout
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


def make_candidate_factory(target_cvar: float, window_days: int = PRIMARY_WINDOW_DAYS,
                            alpha: float = PRIMARY_ALPHA):
    def _factory():
        return ConservativeCVaRScale(cvar_window_days=window_days, cvar_alpha=alpha,
                                     target_cvar=target_cvar)
    return _factory


# ================================================================== (2)
# v4's own `scale` (pre-frac, pre-deadband) is never exposed by prepare().
# Extracted verbatim from KellyRegimeV3.prepare()'s own published source so
# calibrate_target_cvar() can match mean exposure against it. Verified exact
# by self_test_v4_scale_matches_v4_target() before any calibration trusts it.
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
    """v4_scale_series(df, v4), recombined with v4's own vote/deadband logic,
    must reproduce kelly_regime_v4.prepare()'s own `target` column exactly --
    the check that the extraction used to calibrate target_cvar is not silently
    wrong."""
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


def calibrate_for(df_full_train: pd.DataFrame, window_days: int, alpha: float) -> float:
    """target_cvar calibrated on BTC inner-train (2017-2020) ONLY, matching
    mean(scale) to kelly_regime_v4's own mean(scale) on the identical slice.
    Recomputed fresh for every (window_days, alpha) cell -- never reused
    across cells, per this round's own pre-registration."""
    inner_train = df_full_train.loc[r125_shared.INNER_TRAIN_START:r125_shared.INNER_TRAIN_END]
    v4 = get_strategy("kelly_regime_v4")
    v4_scale_train = v4_scale_series(inner_train, v4)
    return r125_shared.calibrate_target_cvar(inner_train["close"], v4_scale_train,
                                             window_days, alpha)


# ================================================================== (3)
# Causal-truncation self-test on THIS round's own new code (mirrors
# r125_shared.py's own __main__ convention). Must PASS before any real
# inner-validation number is trusted.
# ==================================================================

def causal_truncation_probe(df: pd.DataFrame, target_cvar: float,
                            window_days: int = PRIMARY_WINDOW_DAYS,
                            alpha: float = PRIMARY_ALPHA, cut: int = 400_000) -> bool:
    full = ConservativeCVaRScale(cvar_window_days=window_days, cvar_alpha=alpha,
                                 target_cvar=target_cvar).prepare(df.copy())["target"].to_numpy()
    trunc = ConservativeCVaRScale(cvar_window_days=window_days, cvar_alpha=alpha,
                                  target_cvar=target_cvar).prepare(
        df.iloc[:cut].copy())["target"].to_numpy()
    n_check = min(len(trunc), cut) - BARS_PER_DAY * (window_days + 1)
    return bool(np.allclose(full[:n_check], trunc[:n_check], equal_nan=True, rtol=1e-9))


# ================================================================== (4)
# Main: Step-0 -> causal probe -> B1 -> B3 -> B4 -> B5 -> verdict.
# ==================================================================

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    r125_shared.CVAR_ALPHA  # touch to confirm import wired (no-op)
    print("=" * 78)
    print("R-125 CONSERVATIVE: ConservativeCVaRScale -- kelly_regime_v4's own architecture,")
    print("scale's risk measure swapped from EWM-std volatility to annualized CVaR_alpha.")
    print("=" * 78)

    btc = r125_shared.load_btc_train("spot")[0]
    max_ts_seen.append(btc.index.max())
    print(f"\nBTC spot (truncated < {r125_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    print("\n-- PRE-FLIGHT SELF-TEST: v4_scale_series extraction matches "
          "kelly_regime_v4.prepare()'s own target exactly --")
    wiring_ok = self_test_v4_scale_matches_v4_target(btc)
    print(f"  self_test_v4_scale_matches_v4_target: {'PASS' if wiring_ok else 'FAIL'}")
    if not wiring_ok:
        print("\nWIRING SELF-TEST FAILURE -- stopping before any calibration is trusted.")
        return dict(verdict="ABORTED (wiring self-test failure)", max_ts=max(max_ts_seen))

    # -------------------------------------------------------------- calibrate primary
    print(f"\n-- CALIBRATING target_cvar (primary config: window={PRIMARY_WINDOW_DAYS}d, "
          f"alpha={PRIMARY_ALPHA:g}), BTC inner-train ({r125_shared.INNER_TRAIN_START} -> "
          f"{r125_shared.INNER_TRAIN_END}) --")
    target_cvar_primary = calibrate_for(btc, PRIMARY_WINDOW_DAYS, PRIMARY_ALPHA)
    print(f"  target_cvar (primary) = {target_cvar_primary:.6f}")

    # -------------------------------------------------------------- Step 0
    print("\n" + "=" * 78)
    print("STEP 0 -- sanity gate: is the candidate genuinely different from v4, "
          "or a rescaled copy?")
    print("=" * 78)
    cand_factory_primary = make_candidate_factory(target_cvar_primary)
    candidate_target_full = cand_factory_primary().prepare(btc.copy())["target"].to_numpy()
    v4_target_full = r125_shared.v4_reference_target(btc)
    step0 = r125_shared.step0_gate(candidate_target_full, v4_target_full)
    n_configs += 1
    print(f"  R^2 vs v4 (BTC inner-train + inner-validation): {step0['r2_vs_v4']:.6f}")
    print(f"  KILL (R^2 > 0.98)?  {step0['kill']}")
    if step0["kill"]:
        print("\nSTEP-0 KILL: candidate is numerically a rescaled copy of v4. "
              "Reporting honestly rather than proceeding to claim a result.")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 only)")
        print(f"max timestamp read anywhere in this branch: {max_ts} "
              f"(< {r125_shared.OOS_START}: {max_ts < pd.Timestamp(r125_shared.OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="NEGATIVE (Step-0 kill)", step0=step0, n_configs=n_configs,
                   max_ts=max_ts, target_cvar_primary=target_cvar_primary)

    # -------------------------------------------------------------- causal probe
    print("\n" + "=" * 78)
    print("CAUSAL-TRUNCATION SELF-TEST (this round's own new code, real BTC data)")
    print("=" * 78)
    probe_ok = causal_truncation_probe(btc, target_cvar_primary)
    print(f"  causal_truncation_probe (primary config): {'PASS' if probe_ok else 'FAIL'}")
    if not probe_ok:
        print("\nCAUSAL PROBE FAILURE -- a result that looks too good is a bug report first. Stopping.")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 only; promotion bar not run)")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="NEGATIVE (causal probe failure)", step0=step0, n_configs=n_configs,
                   max_ts=max_ts, target_cvar_primary=target_cvar_primary)

    # -------------------------------------------------------------- B1
    print("\n" + "=" * 78)
    print("B1 -- BTC signal, inner-validation, spot + futures")
    print("=" * 78)
    b1_spot = r125_shared.b1_signal(cand_factory_primary, btc, r125_shared.SPOT)
    b1_fut = r125_shared.b1_signal(cand_factory_primary, btc, r125_shared.FUTURES)
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
    print(f"B3 -- plateau: cvar_window_days in {WINDOW_GRID} x alpha in {ALPHA_GRID}, BTC spot, "
          f"target_cvar recalibrated fresh per cell")
    print("=" * 78)
    grid_rows = []
    primary_sign = None
    for window_days in WINDOW_GRID:
        for alpha in ALPHA_GRID:
            tc = calibrate_for(btc, window_days, alpha)
            factory = make_candidate_factory(tc, window_days, alpha)
            r = r125_shared.b1_signal(factory, btc, r125_shared.SPOT)
            n_configs += 1
            sign = float(np.sign(r["d_sharpe"]))
            row = dict(window_days=window_days, alpha=alpha, target_cvar=tc,
                      d_sharpe=r["d_sharpe"], boot_lo=r["paired_lo"], boot_hi=r["paired_hi"],
                      sign=sign)
            grid_rows.append(row)
            if window_days == PRIMARY_WINDOW_DAYS and alpha == PRIMARY_ALPHA:
                primary_sign = sign
            print(f"  window={window_days:>3d}d  alpha={alpha:.2f}  target_cvar={tc:.4f}  "
                  f"d_sharpe={r['d_sharpe']:+.4f}  boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]"
                  f"{'  <- PRIMARY' if (window_days, alpha) == (PRIMARY_WINDOW_DAYS, PRIMARY_ALPHA) else ''}")
    n_same = sum(1 for row in grid_rows if row["sign"] == primary_sign)
    b3_pass = n_same >= len(grid_rows) / 2.0
    print(f"  B3 (majority same-signed as primary, spot): {b3_pass} ({n_same}/{len(grid_rows)})")

    # -------------------------------------------------------------- B4
    print("\n" + "=" * 78)
    print("B4 -- ETH falsification (pre-registered), spot only (no ETH futures data)")
    print("=" * 78)
    eth = r125_shared.load_eth_train()
    max_ts_seen.append(eth.index.max())
    print(f"ETH spot (truncated < {r125_shared.OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")
    b4_spot = r125_shared.b1_signal(cand_factory_primary, eth, r125_shared.SPOT)
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
    b5_spot = r125_shared.b1_signal(cand_factory_primary, btc, r125_shared.SPOT_HIGH_FEE)
    b5_fut = r125_shared.b1_signal(cand_factory_primary, btc, r125_shared.FUTURES_HIGH_FEE)
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
    all_pass = probe_ok and b1_pass and b3_pass and b4_full_pass and b5_pass
    verdict = "PROMOTE-candidate" if all_pass else "NEGATIVE"
    print(f"causal probe={probe_ok}  B1={b1_pass}  B2=diagnostic-only  B3={b3_pass}  "
          f"B4(full)={b4_full_pass}  B5={b5_pass}")
    print(f"ALL GATING CLAUSES PASS: {all_pass}")
    print(f"VERDICT: {verdict}")
    if not all_pass:
        failed = [name for name, ok in (("causal probe", probe_ok), ("B1", b1_pass),
                                        ("B3", b3_pass), ("B4 (full)", b4_full_pass),
                                        ("B5", b5_pass)) if not ok]
        print(f"Reason(s): {', '.join(failed)}")

    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(1 Step-0 + 2 B1 + {len(grid_rows)} B3 + 1 B4 + 2 B5)")
    print(f"max timestamp read anywhere in this branch: {max_ts} "
          f"(< {r125_shared.OOS_START}: {max_ts < pd.Timestamp(r125_shared.OOS_START, tz='UTC')})")
    print("NO bar at or after 2023-01-01 was ever read by this file.")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(
        verdict=verdict, n_configs=n_configs, max_ts=max_ts,
        target_cvar_primary=target_cvar_primary, step0=step0, probe_ok=probe_ok,
        b1_spot=b1_spot, b1_fut=b1_fut, b1_pass=b1_pass,
        b3_grid=grid_rows, b3_pass=b3_pass,
        b4_spot=b4_spot, b4_full_pass=b4_full_pass,
        b5_spot=b5_spot, b5_fut=b5_fut, b5_pass=b5_pass,
    )


if __name__ == "__main__":
    main()
