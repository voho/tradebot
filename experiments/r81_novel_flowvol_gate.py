#!/usr/bin/env python
"""R-81 NOVEL branch, Step A (THE GATE, run FIRST -- see docs/ROUTINE.md step 2/3
and this round's brief): does BTC's own exchange net-flow carry genuine
INCREMENTAL volatility-forecasting power over a standard HAR-RV baseline,
measured BEFORE any strategy code is written?

Idea in one sentence: Ren, Wu & Liu (2024, arXiv:2411.06327) report that
BTC's own net exchange inflow negatively forecasts realized volatility at
every intraday horizon (1-6h) they tested. `kelly_regime_v4`'s SIZE factor
currently divides by trailing (backward-looking) realized volatility with
no forecasting component -- a known lag in how fast its sizing reacts to a
volatility-regime change. This file asks the narrow, falsifiable question
first: does net-flow improve an OUT-OF-SAMPLE volatility forecast at all,
at this project's daily cadence (the paper's horizons are intraday; this
round tests whether that power survives at 1-day-ahead, the horizon that
actually matters for a strategy that re-evaluates sizing once a day's worth
of new bars has closed)? Only if it does is Step B (`r81_novel_flowvol_scale.py`)
attempted.

Constraint attacked: SIZE (kelly_regime_v4's volatility-target scale term).
Not a duplicate of R-59 (retuned `target_vol` magnitude per instrument) or
R-60 (retuned anchor/vote TIMING) -- see `experiments/r81_shared.py`'s
module docstring for the full disambiguation. Also not a duplicate of
R-62 (found the vol-target factor ALONE, with the vote deleted, reproduces
none of v4's headline matched-exposure property) -- this round leaves the
vote and the vol-target factor both in place and asks only whether the
factor's own INPUT (trailing realized vol) can be improved by an external
forecast; R-62's finding is a reason to expect a small effect here even if
the gate passes, not a reason this question is already answered.

RV construction (fixed before any number is computed, per the brief):
`rv[t]` = the standard deviation of day `t`'s own 5-minute log returns
(a standard realized-volatility construction, not the absolute daily
return) -- chosen because it uses the full intraday information in the
bars this project already has, exactly the quantity HAR-RV (Corsi 2009)
is defined on, and because `kelly_regime_v4`'s own trailing-vol input
(`r.ewm(...).std()`) is already a standard-deviation-of-returns
construction, so replacing it with a same-units forecast is a cleaner
substitution than swapping in an absolute-return proxy.

HAR-RV baseline regressors (Corsi 2009 convention, log(RV), causal: day
t's regressors use only data through day t, forecasting rv[t+1]):
    rv_1d[t]  = rv[t]                              (that day's own RV)
    rv_7d[t]  = mean(rv[t-6..t])                   (trailing 7-day mean)
    rv_30d[t] = mean(rv[t-29..t])                  (trailing 30-day mean)

Candidate feature: `net_flow_z[t]`, the rolling-90-day z-score of daily
`net_flow = FlowInExNtv - FlowOutExNtv` (recomputed here from
`r81_shared.load_net_flow('BTC')`; not imported from the conservative
branch's files, per this round's brief). Causal availability: CoinMetrics
reports day D's flow only after D closes, i.e. it becomes visible at day
D+1 00:00 UTC (`align_onchain_causal`'s own docstring). Applying that same
shift to a DAILY grid (`align_onchain_causal(z_raw, daily_grid)` where
`daily_grid` is indexed at each day's own midnight) puts, at grid point t,
the z-score computed through day t-1's flow -- exactly the value that was
visible at any time during day t, including at day t's close when the
regression is imagined to run. This is the identical mechanical
consequence the project's other on-chain features (MVRV, DVOL, macro) get
from the same helper; it is applied here, not re-derived differently.

Two OLS regressions, INNER-TRAIN ONLY (2017-01-01 -> 2020-12-31):
    baseline:  log(rv[t+1]) ~ 1 + log(rv_1d[t]) + log(rv_7d[t]) + log(rv_30d[t])
    candidate: baseline + net_flow_z[t]
evaluated OUT-OF-SAMPLE on inner-validation (2021-01-01 -> 2022-12-31),
coefficients frozen from the inner-train fit (no refitting on validation
for the main comparison).

Pre-registered decision rule (fixed here, before either regression is
fit -- see docs/ROUTINE.md step 2's "compute the bar's own noise" and this
round's brief):
    PROCEED to Step B only if BOTH hold:
      (1) candidate's out-of-sample R² on inner-validation exceeds
          baseline's out-of-sample R² by AT LEAST +0.01 (one percentage
          point of variance explained);
      (2) `net_flow_z`'s fitted coefficient has the SAME SIGN when the
          candidate model is refit on inner-validation ALONE (a stability
          check, not a formal bootstrap -- refit once on 2021-2022, compare
          sign only, not magnitude).
    If either check fails: STOP. Do not write any strategy code. Report
    NEGATIVE at the gate.
Configurations evaluated by this gate: 2 (baseline, candidate) fitted on
inner-train, evaluated OOS on inner-validation, plus 1 stability refit of
the candidate on inner-validation alone = counted as part of this round's
total trials in the final report.

Usage
-----
    python experiments/r81_novel_flowvol_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_onchain_causal, load_dataset  # noqa: E402

from experiments.r81_shared import (  # noqa: E402
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    TRAIN,
    assert_no_holdout,
    load_net_flow,
)

DATA_DIR = ROOT / "data"
Z_WINDOW_DAYS = 90
Z_MIN_PERIODS = 30  # matches the conservative branch's own convention (needs *some* history, not the full window)


# --------------------------------------------------------------- construction


def build_daily_rv(spot: pd.DataFrame) -> pd.Series:
    """rv[t] = std of day t's own 5-minute log returns. Causal by
    construction: day t's value uses only bars stamped within day t."""
    r = np.log(spot["close"]).diff()
    rv = r.resample("1D").std()
    return rv.rename("rv")


def build_net_flow_z_daily(daily_grid_index: pd.DatetimeIndex) -> pd.Series:
    """Rolling-90-day z-score of raw net_flow, causally shifted one day
    late (CoinMetrics publication lag) and reindexed onto `daily_grid_index`
    (each day's own midnight timestamp)."""
    raw = load_net_flow("BTC")
    nf = raw["net_flow"]
    mean = nf.rolling(Z_WINDOW_DAYS, min_periods=Z_MIN_PERIODS).mean()
    std = nf.rolling(Z_WINDOW_DAYS, min_periods=Z_MIN_PERIODS).std()
    z = ((nf - mean) / std).rename("net_flow_z").to_frame()
    grid = pd.DataFrame(index=daily_grid_index)
    aligned = align_onchain_causal(z, grid)["net_flow_z"]
    return aligned


def build_har_frame() -> pd.DataFrame:
    """Assemble the daily regression frame: rv_1d, rv_7d, rv_30d, net_flow_z,
    and the forecast target log(rv[t+1]). Truncated to strictly before
    OOS_START before anything downstream touches it."""
    spot, _ = load_dataset(DATA_DIR, "spot")
    cutoff = pd.Timestamp(OOS_START, tz=spot.index.tz)
    spot = spot.loc[spot.index < cutoff]
    assert_no_holdout(spot)

    rv = build_daily_rv(spot)
    rv_1d = rv
    rv_7d = rv.rolling(7, min_periods=7).mean()
    rv_30d = rv.rolling(30, min_periods=30).mean()

    net_flow_z = build_net_flow_z_daily(rv.index)

    df = pd.DataFrame({
        "rv": rv,
        "rv_1d": rv_1d,
        "rv_7d": rv_7d,
        "rv_30d": rv_30d,
        "net_flow_z": net_flow_z,
    })
    df["log_rv_1d"] = np.log(df["rv_1d"])
    df["log_rv_7d"] = np.log(df["rv_7d"])
    df["log_rv_30d"] = np.log(df["rv_30d"])
    df["target_log_rv_next"] = np.log(df["rv"].shift(-1))
    return df


# --------------------------------------------------------------- regression


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain OLS via lstsq; X must already include an intercept column."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _oos_r2(coef: np.ndarray, X: np.ndarray, y: np.ndarray, train_mean: float) -> float:
    """R² against the TRAIN mean (standard out-of-sample R² convention --
    using the validation-set's own mean would let the benchmark peek at
    validation, understating how hard out-of-sample forecasting is)."""
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - train_mean) ** 2))
    return 1.0 - ss_res / ss_tot


def _design(df: pd.DataFrame, with_flow: bool) -> tuple[np.ndarray, np.ndarray]:
    cols = ["log_rv_1d", "log_rv_7d", "log_rv_30d"] + (["net_flow_z"] if with_flow else [])
    sub = df.dropna(subset=cols + ["target_log_rv_next"])
    X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy() for c in cols])
    y = sub["target_log_rv_next"].to_numpy()
    return X, y, sub.index


def run_gate() -> dict:
    df = build_har_frame()

    train = df.loc[TRAIN[0]:INNER_TRAIN_END]
    valid = df.loc[INNER_VAL_START:INNER_VAL_END]
    assert_no_holdout(train)
    assert_no_holdout(valid)

    print(f"inner-train  {train.index[0].date()} -> {train.index[-1].date()}  ({len(train)} days)")
    print(f"inner-valid  {valid.index[0].date()} -> {valid.index[-1].date()}  ({len(valid)} days)")

    results = {}
    for name, with_flow in (("baseline", False), ("candidate", True)):
        Xtr, ytr, _ = _design(train, with_flow)
        Xva, yva, _ = _design(valid, with_flow)
        coef = _fit_ols(Xtr, ytr)
        train_mean = float(ytr.mean())
        r2_oos = _oos_r2(coef, Xva, yva, train_mean)
        results[name] = {"coef": coef, "r2_oos": r2_oos, "n_train": len(ytr), "n_valid": len(yva)}
        label = ["intercept", "log_rv_1d", "log_rv_7d", "log_rv_30d"] + (["net_flow_z"] if with_flow else [])
        coef_str = ", ".join(f"{lbl}={c:+.4f}" for lbl, c in zip(label, coef))
        print(f"\n{name:10s} n_train={len(ytr):4d} n_valid={len(yva):4d}  OOS R²={r2_oos:+.4f}")
        print(f"  coefs: {coef_str}")

    # Stability check: refit the CANDIDATE on inner-validation alone.
    Xva, yva, _ = _design(valid, with_flow=True)
    coef_refit = _fit_ols(Xva, yva)
    flow_coef_train = results["candidate"]["coef"][-1]
    flow_coef_refit = coef_refit[-1]
    same_sign = bool(np.sign(flow_coef_train) == np.sign(flow_coef_refit)) and flow_coef_train != 0
    print(f"\nnet_flow_z coefficient: inner-train fit = {flow_coef_train:+.4f}, "
          f"inner-validation refit = {flow_coef_refit:+.4f}  "
          f"({'SAME SIGN' if same_sign else 'SIGN FLIPPED'})")

    delta_r2 = results["candidate"]["r2_oos"] - results["baseline"]["r2_oos"]
    check1 = delta_r2 >= 0.01
    check2 = same_sign
    passes = check1 and check2

    print(f"\n{'=' * 78}")
    print("PRE-REGISTERED DECISION RULE")
    print(f"  (1) candidate OOS R² - baseline OOS R² >= +0.01 : "
          f"{delta_r2:+.4f}  {'PASS' if check1 else 'FAIL'}")
    print(f"  (2) net_flow_z coefficient same sign, train vs validation refit : "
          f"{'PASS' if check2 else 'FAIL'}")
    print(f"  OVERALL: {'GATE PASSES -> proceed to Step B' if passes else 'GATE FAILS -> STOP, report NEGATIVE'}")

    return {
        "baseline_r2_oos": results["baseline"]["r2_oos"],
        "candidate_r2_oos": results["candidate"]["r2_oos"],
        "delta_r2": delta_r2,
        "flow_coef_train": flow_coef_train,
        "flow_coef_refit": flow_coef_refit,
        "check1_delta_r2": check1,
        "check2_same_sign": check2,
        "passes": passes,
    }


if __name__ == "__main__":
    run_gate()
