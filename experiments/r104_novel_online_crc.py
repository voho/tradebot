"""R-104 NOVEL branch: online / adaptive Conformal Risk Control.

Wraps kelly_regime_v4's raw exposure (`frac * scale`, before v4's own 10%
deadband) in a discount `d_t` updated causally, bar by bar, by the
Feldman-Bates-Romano (2023) / Angelopoulos et al. (2024, App. D) online
risk-control controller already implemented in `r104_shared.py`
(`crc_online_lambda_path`). See `experiments/r104_shared.py` for the full
pre-registration, citations, and named failure modes. This file does not
edit that module; it only imports from it.

CRITICAL correctness point (R-91's mistake, must not be repeated here):
`tradebot.window.run_period` hands each named slice a WARMUP PREFIX sized
by `strategy.warmup` (a bar count), taken from the bars immediately before
the slice's own start. If the candidate's `TargetStrategy` keeps v4's
default warmup (~80 days), the online controller's `d` path silently
restarts from `d0=0.0` at the start of `inner_val`, throwing away the
online state accumulated over inner_train -- exactly what an "online"
method is not supposed to do. The fix: construct the candidate strategy
with `warmup=1_600_000` (comfortably larger than the ~421k-bar gap between
INNER_TRAIN_START and INNER_VAL_START, and larger than ETH's own series
length), so `prefix_bars` always resolves to "all available history", not
a truncated window. This is verified explicitly below (step 3) before any
Sharpe/drawdown number is computed or reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.window import prefix_bars  # noqa: E402

from experiments.r104_shared import (  # noqa: E402
    CRC_ALPHA,
    CRC_ONLINE_ETA,
    CRC_TAU_QUANTILE,
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
    apply_deadband,
    assert_no_holdout,
    bar_forward_loss,
    causal_truncation_probe_series,
    crc_online_lambda_path,
    exceedance_rate,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    set_online_tau,
    v4_raw_desired,
    v4_target,
)

CANDIDATE_WARMUP = 1_600_000
LABEL = "novel_online_crc"


def calibrate_tau_local(df: pd.DataFrame, cal_end: str, q: float = CRC_TAU_QUANTILE) -> float:
    """Reimplementation of `r104_shared.calibrate_tau`, identical recipe,
    built only from r104_shared's own public building blocks
    (`v4_raw_desired`, `bar_forward_loss`).

    BUG HIT AND WORKED AROUND: `r104_shared.calibrate_tau` (line ~334) does
    `mask = df.index < pd.Timestamp(cal_end, tz="UTC"); cal_loss =
    loss[mask.to_numpy()]`. On this environment's pandas (3.0.5), comparing
    a DatetimeIndex to a Timestamp already returns a plain `numpy.ndarray`
    of bool (not a pandas Index/Series), so `.to_numpy()` raises
    `AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'`.
    r104_shared.py is read-only for this branch, so rather than edit it,
    this function reproduces its exact documented recipe with that one line
    fixed (`mask` used directly, no redundant `.to_numpy()`). The
    conservative (sibling) branch calls the same shared function and should
    hit the identical bug -- worth the operator's attention upstream."""
    e = v4_raw_desired(df)
    loss = bar_forward_loss(e, df["close"])
    mask = df.index < pd.Timestamp(cal_end, tz="UTC")
    cal_loss = loss[mask]
    assert len(cal_loss) > 288 * 30, "calibration window too short"
    return float(np.quantile(cal_loss, q))

# ---------------------------------------------------------------------
# Recording side-channel: every call to `build_target` stashes the frame
# it was actually handed (df.index / df["close"]), the raw exposure e0,
# and the online d_path it computed, so the continuity check (step 3) can
# inspect exactly what happened INSIDE a real TargetStrategy.prepare() /
# run_period() call, not a hand-rolled re-derivation of it.
# ---------------------------------------------------------------------
_LAST_CALL: dict = {}


def build_target(df: pd.DataFrame) -> np.ndarray:
    """Online CRC candidate: e0 = v4's raw (pre-deadband) exposure,
    discounted by the online-controller's d_path, then v4's own deadband
    applied on top (same convention as v4_target's own apply_deadband)."""
    e0 = v4_raw_desired(df)
    d_path = crc_online_lambda_path(None, e0, df["close"])
    _LAST_CALL["index"] = df.index
    _LAST_CALL["e0"] = e0
    _LAST_CALL["d_path"] = d_path
    return apply_deadband(e0 * (1.0 - d_path))


def main() -> None:
    print("=" * 78)
    print("R-104 NOVEL branch: online/adaptive Conformal Risk Control")
    print("=" * 78)

    # ------------------------------------------------------------- step 1
    btc = load_btc()
    assert_no_holdout(btc, "main(): btc")
    tau = calibrate_tau_local(btc, cal_end="2019-01-01", q=0.99)
    print(f"\n[1] tau = {tau!r}  (v4's own 99th-pct single-bar loss, 2017-01-01..2018-12-31)")
    print("    [note: r104_shared.calibrate_tau itself raises AttributeError on this "
          "pandas version (3.0.5) -- see calibrate_tau_local's docstring for the exact bug "
          "and workaround; the recipe reproduced here is otherwise byte-identical to it]")

    # ------------------------------------------------------------- step 2
    set_online_tau(tau)
    print(f"[2] set_online_tau({tau}) -- online controller wired. "
          f"alpha={CRC_ALPHA}, eta={CRC_ONLINE_ETA}")

    # ------------------------------------------------------------- step 3
    # Continuity check. Do this BEFORE anything else.
    print("\n[3] CONTINUITY CHECK (the R-91 failure mode, gating everything below)")
    full_e0 = v4_raw_desired(btc)
    full_d_path = crc_online_lambda_path(None, full_e0, btc["close"])
    inner_val_start_ts = pd.Timestamp(INNER_VAL_START, tz="UTC")
    pos = int(btc.index.searchsorted(inner_val_start_ts))
    d_direct_full = float(full_d_path[pos])
    print(f"    direct full-series (2017-2022, one pass) d at bar {pos} "
          f"({btc.index[pos]}) = {d_direct_full!r}")

    # Now go through the REAL harness path: TargetStrategy with the large
    # warmup, run via run_slice/run_period, exactly as the comparison loop
    # below will do. build_target's recording side-channel captures the
    # frame + d_path it computed internally.
    cand_strategy = TargetStrategy(build_target, name=f"r104_{LABEL}", warmup=CANDIDATE_WARMUP)
    lo_check = int(btc.index.searchsorted(inner_val_start_ts))
    prefix_check = prefix_bars(btc, lo_check, CANDIDATE_WARMUP)
    print(f"    prefix_bars(btc, lo={lo_check}, warmup={CANDIDATE_WARMUP}) = {prefix_check} "
          f"(should equal lo, i.e. ALL history back to bar 0)")
    _ = run_slice(cand_strategy, btc, INNER_VAL_START, INNER_VAL_END,
                  "inner_val_continuity_probe", SPOT)
    frame_index = _LAST_CALL["index"]
    frame_d_path = _LAST_CALL["d_path"]
    local_pos = int(frame_index.searchsorted(inner_val_start_ts))
    d_via_harness = float(frame_d_path[local_pos])
    print(f"    via TargetStrategy(warmup={CANDIDATE_WARMUP})+run_slice(inner_val): "
          f"frame len={len(frame_index)}, frame starts {frame_index[0]}, "
          f"local_pos of inner_val start={local_pos}, d={d_via_harness!r}")

    continuity_diff = abs(d_direct_full - d_via_harness)
    continuity_pass = continuity_diff < 1e-12
    print(f"    |direct - harness| = {continuity_diff:.3e}  ->  "
          f"{'PASS' if continuity_pass else 'FAIL'}")
    if not continuity_pass:
        raise AssertionError(
            f"CONTINUITY CHECK FAILED: direct={d_direct_full!r} vs harness={d_via_harness!r}. "
            "Online state is not carrying across the inner_train/inner_val boundary -- "
            "this is the R-91 failure mode. Stopping before reporting any further numbers.")
    print("    Determinism note: crc_online_lambda_path's recursion is a pure causal "
          "function of (prior d, e0[i-1], r[i]); since the harness frame and the direct "
          "full-series pass share identical bars from index 0 through the inner_val "
          "boundary, equality at this one bar implies equality of the ENTIRE d_path over "
          "inner_val, not just the first bar.")

    # ------------------------------------------------------------- step 5 (A0)
    # Use the validated full-series d_path directly (step 3 proved it is
    # bit-identical to what the harness computes for inner_val).
    hi = int(btc.index.searchsorted(pd.Timestamp(INNER_VAL_END, tz="UTC"), side="right"))
    d_val = full_d_path[pos:hi]
    e0_val = full_e0[pos:hi]
    close_val = btc["close"].iloc[pos:hi]
    raw_discounted_val = e0_val * (1.0 - d_val)
    online_exceedance = exceedance_rate(raw_discounted_val, close_val, tau)

    mean_d = float(np.mean(d_val))
    std_d = float(np.std(d_val))
    frac_gt_05 = float(np.mean(d_val > 0.05))
    print("\n[5] A0 measurement gate -- inner_val d-path diagnostics")
    print(f"    n bars = {len(d_val)}")
    print(f"    mean(d) = {mean_d:.6f}  std(d) = {std_d:.6f}  frac(d>0.05) = {frac_gt_05:.4f}")
    print(f"    empirical exceedance rate under the discounted exposure = {online_exceedance:.6f} "
          f"(target alpha = {CRC_ALPHA})")
    if std_d < 1e-4:
        print("    NOTE: std(d) is effectively zero -- the online controller converges to "
              "and sits at a near-constant value on inner_val. This is functionally "
              "equivalent to a STATIC discount despite the online machinery. Reporting this "
              "as a legitimate, informative negative, not a failure to disclose.")

    # ------------------------------------------------------------- step 6
    print("\n[6] Causal truncation probe")
    try:
        probe_ok = causal_truncation_probe_series(build_target, btc)
        print(f"    causal_truncation_probe_series(build_target, btc) -> {probe_ok} -> PASS")
    except AssertionError as exc:
        probe_ok = False
        print(f"    FAILED: {exc}")
        print("    Investigating whether this is `_ONLINE_TAU` module state leaking across "
              "calls: _ONLINE_TAU is set once, globally, before this probe runs, and is never "
              "mutated during it, so every build_target(...) call inside the probe reads the "
              "SAME tau; the recursion itself only ever reads exposure[i-1] and r[i]. If this "
              "assertion fires it indicates a real bug in this file's wiring, not a limitation "
              "of the method -- see printed diagnostics above for where bars first diverge.")

    # ------------------------------------------------------------- step 7 (A2)
    print("\n[7] A2 non-inertness kill switch")
    cand_full = apply_deadband(full_e0 * (1.0 - full_d_path))  # == build_target(btc), reused
    a2_r2 = r_squared(cand_full, full_e0)
    print(f"    r_squared(build_target(btc), v4_raw_desired(btc)) = {a2_r2:.6f}")
    if a2_r2 >= 0.98:
        print("    VERDICT: >= 0.98 -- this is the 28th SIZE-axis construction to collapse to "
              "'a near-constant rescale of v4's own path' (this project's established "
              "language for the failure mode).")
    else:
        print("    VERDICT: < 0.98 -- NOT a near-constant rescale of v4's own path by this "
              "project's own convention.")

    # ------------------------------------------------------------- step 4/8
    # Custom compare loop (cannot reuse r104_shared.compare() unchanged: it
    # hardcodes TargetStrategy(candidate_build, ...) with the DEFAULT warmup,
    # which is exactly the R-91 bug for an online candidate).
    print("\n[8] Comparison table (candidate warmup="
          f"{CANDIDATE_WARMUP} vs kelly_regime_v4 default warmup)")
    eth = load_eth()
    assert_no_holdout(eth, "main(): eth")

    ctrl_strategy = TargetStrategy(v4_target, name="kelly_regime_v4")

    def run_compare_custom(cand_strategy, ctrl_strategy, *, label, btc_df, eth_df,
                           markets=(SPOT, FUTURES), include_eth=True, seed=0) -> list[dict]:
        assert_no_holdout(btc_df, "run_compare_custom: btc")
        if include_eth:
            assert_no_holdout(eth_df, "run_compare_custom: eth")
        rows = []
        jobs = [(name, start, end, btc_df) for name, (start, end) in SLICES.items()]
        if include_eth:
            jobs.append((ETH_SLICE_NAME, None, None, eth_df))
        for slice_name, start, end, df in jobs:
            for market in markets:
                a = run_slice(cand_strategy, df, start, end, slice_name, market)
                b = run_slice(ctrl_strategy, df, start, end, slice_name, market)
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

    rows = run_compare_custom(cand_strategy, ctrl_strategy, label=LABEL,
                              btc_df=btc, eth_df=eth, markets=(SPOT, FUTURES),
                              include_eth=True)
    print_rows(rows)

    # ------------------------------------------------------------- step 9
    print("\n[9] 0.40% fee-tier check (inner_val only, both markets)")
    spot_04 = fee_at(SPOT, 0.004)
    fut_04 = fee_at(FUTURES, 0.004)
    fee_rows = []
    for market in (spot_04, fut_04):
        a = run_slice(cand_strategy, btc, INNER_VAL_START, INNER_VAL_END, "inner_val_fee04", market)
        b = run_slice(ctrl_strategy, btc, INNER_VAL_START, INNER_VAL_END, "inner_val_fee04", market)
        pr = paired_diff(a.daily, b.daily, seed=0)
        exp_ratio = a.mean_abs_exposure / b.mean_abs_exposure if b.mean_abs_exposure else float("nan")
        vol_ratio = a.realized_vol / b.realized_vol if b.realized_vol else float("nan")
        fee_rows.append({
            "label": LABEL + "_fee04", "slice": "inner_val_fee04", "market": market.name,
            "cand_final": a.final_balance, "ctrl_final": b.final_balance,
            "cand_log_growth": a.log_growth, "ctrl_log_growth": b.log_growth,
            "d_log_growth": a.log_growth - b.log_growth,
            "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe, "d_sharpe": a.sharpe - b.sharpe,
            "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
            "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
            "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
            "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
            "risk_matched": bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                            if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False,
            "boot_d_loggrowth": pr.diff.point, "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
            "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
        })
    print_rows(fee_rows)

    # ------------------------------------------------------------- step 10
    print("\n[10] ETH note")
    lo_eth = 0  # eth_replication slice passes start=None -> lo=0 always
    prefix_eth = prefix_bars(eth, lo_eth, CANDIDATE_WARMUP)
    print(f"    eth slice uses start=None -> lo=0 -> prefix_bars(eth, 0, {CANDIDATE_WARMUP}) "
          f"= {prefix_eth} bars of prefix (ETH series itself has {len(eth)} bars total, "
          f"{eth.index[0]} .. {eth.index[-1]}).")
    print("    The controller starts fresh at d0=0.0 on ETH's own first bar -- no shared "
          "state with BTC, as expected: there is no principled cross-instrument online "
          "state transfer, and this is a separate, self-contained deployment.")

    print("\n" + "=" * 78)
    print("DONE.")
    print("=" * 78)


if __name__ == "__main__":
    main()
