"""R-175 CONSERVATIVE branch (08-28): substitute the MSM(6) FULL one-step-
ahead volatility forecast (`r175_shared.msm_full_vol_bars` /
`msm_full_target`) for `kelly_regime_v4`'s own single-span EWM realized-vol
estimator, everything else (vote, hysteresis thresholds, target_vol,
max_leverage, 10% deadband) byte-identical. See `experiments/r175_direction.md`
("Conservative variant") for the frozen mechanism, non-duplication argument
and pre-registered decision rule; this file only executes that rule and
records the resulting numbers. Does NOT edit `r175_shared.py` or anything
named `r175_novel*` (separate branch, separate agent, per this project's
parallel-branch convention).

Frozen pre-registered decision rule (copied verbatim from
r175_direction.md's "Conservative variant" + the task's own framing, BEFORE
any real-data number in this file was read):

FALSIFICATION GATE (BTC inner-validation, both markets), STOP if EITHER:
  (F1) paired-bootstrap CI on d_log_growth (equivalently d_sharpe) vs.
       kelly_regime_v4 excludes zero on the LOSING side (i.e. cand_boot_hi<0,
       meaning candidate is significantly worse) on spot OR futures_5x.
  (F2) R^2(msm_full_vol_bars, v4_symmetric_vol) > 0.98 on BTC (mere
       relabeling of the incumbent estimator).

If neither falsifier fires -> standard promotion bar, both markets:
  (P1) ΔSharpe >= +0.2 OR a risk-matched drawdown improvement
  (P2) ETH sign-replication in the same direction as BTC
  (P3) survives the 0.40% taker fee tier (`fee_at`)
ALL THREE required for PROMOTE. If the inner-validation promotion clause
looks like it might clear (i.e. neither falsifier fired AND at least one of
P1/P2 looks favorable on inner-validation), exactly ONE holdout read is
permitted: `compare()`-equivalent numbers on BTC start="2023-01-01", both
markets. Pre-registered holdout interpretation (written BEFORE looking,
per the outcomes below) -- see HOLDOUT_PREREGISTRATION below.

Any other outcome (falsifier fires; OR neither falsifier fires but the
promotion clause plainly does not look like it might clear on
inner-validation) => do not read holdout; verdict is NEGATIVE, reported as
a clean fall-through, not rounded up to any favorable label.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.r102_shared import (  # noqa: E402
    FUTURES,
    SPOT,
    fee_at,
    load_btc,
    load_eth,
    r_squared,
    run_slice,
    v4_symmetric_vol,
    v4_target,
)
from experiments.r175_shared import (  # noqa: E402
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    compare,
    msm_full_target,
    msm_full_vol_bars,
    print_rows,
)

CONFIGS_EVALUATED: list[str] = []  # every configuration/backtest cell, logged as we go


def _log(cfg: str) -> None:
    CONFIGS_EVALUATED.append(cfg)


# ------------------------------------------------------------------- gate 1
def run_falsification_gate(btc, eth):
    """F1: paired-bootstrap CI on BTC inner-validation, both markets.
    F2: R^2(msm_full_vol_bars, v4_symmetric_vol) on BTC (whole causal
    series available up to inner-validation end -- no holdout bars read)."""
    print("=" * 100)
    print("FALSIFICATION GATE -- BTC inner-validation, both markets")
    print("=" * 100)
    rows = compare(msm_full_target, label="conservative", btc=btc, eth=eth,
                   markets=(SPOT, FUTURES), include_eth=True)
    for m in ("spot", "futures_5x"):
        _log(f"gate:inner_val:{m}")
        _log(f"gate:inner_train:{m}")
        _log(f"gate:eth_replication:{m}")
    print_rows(rows)

    inner_val_rows = [r for r in rows if r["slice"] == "inner_val"]
    f1_fired = False
    f1_detail = []
    for r in inner_val_rows:
        losing_excluded = r["excludes_zero"] and r["boot_hi"] < 0.0
        f1_detail.append((r["market"], r["boot_lo"], r["boot_hi"], losing_excluded))
        if losing_excluded:
            f1_fired = True

    # F2: R^2 between the candidate's raw vol input and the incumbent's, on
    # BTC restricted to bars through inner-validation end (no holdout read).
    btc_causal = btc.loc[:INNER_VAL_END]
    cand_vol = msm_full_vol_bars(btc_causal)
    ctrl_vol = v4_symmetric_vol(btc_causal)
    r2 = r_squared(cand_vol, ctrl_vol)
    _log("gate:r_squared(msm_full_vol_bars, v4_symmetric_vol) on BTC thru inner_val_end")
    f2_fired = np.isfinite(r2) and r2 > 0.98

    print(f"\nF1 (losing-side CI excludes zero on inner_val, either market): "
          f"{'FIRED' if f1_fired else 'did not fire'}")
    for mkt, lo, hi, excl in f1_detail:
        print(f"    {mkt:12s} boot_lo={lo:+.4f} boot_hi={hi:+.4f} losing_excluded={excl}")
    print(f"F2 (R^2(msm_full_vol_bars, v4_symmetric_vol) > 0.98): "
          f"{'FIRED' if f2_fired else 'did not fire'} (R^2={r2:.4f})")

    return rows, f1_fired, f2_fired, r2


# ------------------------------------------------------------- promotion bar
def check_promotion_clause(rows):
    """Does the inner-validation evidence look like it might clear the
    standard promotion bar, both markets? (Gate to decide whether the ONE
    permitted holdout read happens at all.)"""
    inner_val = [r for r in rows if r["slice"] == "inner_val"]
    eth_rows = [r for r in rows if r["slice"] == "eth_replication"]

    sharpe_ok = all(r["d_sharpe"] >= 0.2 for r in inner_val)
    dd_ok = all(r["risk_matched"] and r["d_dd"] < 0 for r in inner_val)
    p1_ok = sharpe_ok or dd_ok

    btc_dir = np.sign(np.mean([r["d_sharpe"] for r in inner_val]))
    eth_dir = np.sign(np.mean([r["d_sharpe"] for r in eth_rows])) if eth_rows else 0.0
    p2_ok = bool(btc_dir != 0 and btc_dir == eth_dir)

    print(f"\nPromotion-clause pre-check: P1(dSharpe>=0.2 both mkts OR "
          f"risk-matched DD improvement both mkts)={p1_ok}  "
          f"(sharpe_ok={sharpe_ok}, dd_ok={dd_ok})")
    print(f"P2(ETH sign-replication)={p2_ok}  (btc_dir={btc_dir:+.0f}, eth_dir={eth_dir:+.0f})")
    return p1_ok, p2_ok


# --------------------------------------------------------------- fee robustness
def run_fee_check(btc):
    """P3: survives the 0.40% taker fee tier, on BTC inner-validation."""
    print("\n" + "=" * 100)
    print("FEE ROBUSTNESS -- BTC inner-validation @ 0.40% taker, both markets")
    print("=" * 100)
    from experiments.r175_shared import TargetStrategy, MSM_WARMUP_BARS

    cand = TargetStrategy(msm_full_target, name="r175_conservative_fee40", warmup=MSM_WARMUP_BARS)
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4_fee40")
    out = {}
    for market in (SPOT, FUTURES):
        m40 = fee_at(market, 0.0040)
        a = run_slice(cand, btc, INNER_VAL_START, INNER_VAL_END, "inner_val_fee40", m40)
        b = run_slice(ctrl, btc, INNER_VAL_START, INNER_VAL_END, "inner_val_fee40", m40)
        _log(f"fee40:inner_val:{market.name}")
        d_sharpe = a.sharpe - b.sharpe
        d_dd = a.max_drawdown_pct - b.max_drawdown_pct
        out[market.name] = {"cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                             "d_sharpe": d_sharpe, "cand_dd": a.max_drawdown_pct,
                             "ctrl_dd": b.max_drawdown_pct, "d_dd": d_dd}
        print(f"  {market.name:12s} cand_sharpe={a.sharpe:+.3f} ctrl_sharpe={b.sharpe:+.3f} "
              f"d_sharpe={d_sharpe:+.3f}  cand_dd={a.max_drawdown_pct:+.1f} "
              f"ctrl_dd={b.max_drawdown_pct:+.1f} d_dd={d_dd:+.1f}")
    return out


# --------------------------------------------------------------- robustness sweep
def run_robustness_sweep(btc, eth):
    """A small number (2-4) of robustness configs, per task instructions:
    sensitivity of the CONSERVATIVE candidate to REFIT_DAYS, since
    r175_shared.py's own module-level constants are frozen and must not be
    edited. We build LOCAL copies of the two functions that matter
    (`msm_forecast_daily` / `msm_full_vol_bars` / `msm_full_target`) that
    accept a `refit_days` override, reusing every other frozen piece
    (grid, kbar, calib window, min-history) from r175_shared verbatim.
    Evaluated on BTC inner-validation only (still training data)."""
    import experiments.r175_shared as rs

    def make_target_with_refit(refit_days: int):
        def build(df):
            fc = rs.msm_forecast_daily(df, refit_days=refit_days)
            vol = rs._broadcast_vol(df, fc["full_mult"])
            scale = rs.conditional_target_scale(vol)
            return rs.apply_deadband(rs.v4_vote_frac(df).to_numpy() * scale)
        return build

    print("\n" + "=" * 100)
    print("ROBUSTNESS SWEEP -- REFIT_DAYS in {45, 90(frozen), 180}, BTC inner-validation")
    print("=" * 100)
    results = []
    for refit_days in (45, 90, 180):
        label = f"conservative_refit{refit_days}"
        build = make_target_with_refit(refit_days) if refit_days != 90 else msm_full_target
        rows = compare(build, label=label, btc=btc, eth=eth,
                       markets=(SPOT, FUTURES), include_eth=False)
        for r in rows:
            _log(f"robustness:refit{refit_days}:{r['slice']}:{r['market']}")
        print_rows(rows)
        results.append((refit_days, rows))
    return results


# --------------------------------------------------------------------- main
def main():
    btc = load_btc()
    eth = load_eth()

    rows, f1, f2, r2 = run_falsification_gate(btc, eth)

    if f1 or f2:
        print("\n*** FALSIFIED AT GATE ***")
        print(f"    F1 fired: {f1}   F2 (R^2={r2:.4f}) fired: {f2}")
        print("    Per pre-registration: STOP. Do not read holdout.")
        print(f"\nTotal configurations/cells evaluated: {len(CONFIGS_EVALUATED)}")
        return

    print("\nNeither falsifier fired. Proceeding to promotion-clause pre-check.")
    p1_ok, p2_ok = check_promotion_clause(rows)

    fee_out = run_fee_check(btc)
    robustness = run_robustness_sweep(btc, eth)

    might_clear = p1_ok  # P2/ETH checked on the same inner_val rows already
    if might_clear:
        print("\nInner-validation promotion clause looks like it MIGHT clear "
              "(P1 satisfied on inner_val) -- proceeding to the ONE "
              "pre-registered holdout read.")
        print(HOLDOUT_PREREGISTRATION)
        holdout_rows = run_holdout(btc, eth)
        verdict = decide_final_verdict(rows, p1_ok, p2_ok, fee_out, holdout_rows)
    else:
        print("\nInner-validation promotion clause does NOT look like it "
              "might clear (P1 fails on inner_val) -- per pre-registration, "
              "holdout is NOT read. Verdict falls through to NEGATIVE.")
        holdout_rows = None
        verdict = "NEGATIVE (fall-through: inner-validation promotion clause did not clear; holdout not read)"

    print(f"\n{'=' * 100}\nVERDICT: {verdict}\n{'=' * 100}")
    print(f"\nTotal configurations/cells evaluated: {len(CONFIGS_EVALUATED)}")
    for c in CONFIGS_EVALUATED:
        print(f"  - {c}")


HOLDOUT_PREREGISTRATION = """
HOLDOUT PRE-REGISTRATION (written before looking at any bar >= 2023-01-01):
We will run compare()-equivalent numbers via run_slice on BTC start=2023-01-01
(no end, i.e. to the end of the available series), both markets (spot,
futures_5x), for msm_full_target vs kelly_regime_v4.
  - If d_sharpe >= +0.2 on BOTH markets, OR a risk-matched (0.9<=exp_ratio/vol_ratio<=1.1)
    drawdown improvement holds on BOTH markets, AND the inner-validation P2
    (ETH sign-replication) and P3 (0.40% fee survival: d_sharpe stays >= 0 at
    the 0.40% tier on both markets) both hold => PROMOTE.
  - If the holdout sign matches inner-validation's direction but magnitude
    misses the +0.2 Sharpe / drawdown bar => NEGATIVE, reported as
    "directionally consistent but sub-threshold", not rounded up.
  - If the holdout sign is opposite to inner-validation's (i.e. the
    candidate helps in training data but hurts out-of-sample) => NEGATIVE,
    explicitly flagged as a THIRD instance of the R-08/R-136 forecast-
    quality-improvement inversion, this time against MSM.
  - Any result that satisfies neither ADOPT nor either NEGATIVE reading above
    is reported as a fall-through with the raw table, per ROUTINE.md Step 4's
    partition requirement.
"""


def run_holdout(btc, eth):
    print("\n" + "=" * 100)
    print(f"HOLDOUT READ -- BTC start={OOS_START}, both markets (PRE-REGISTERED, ONE READ)")
    print("=" * 100)
    from experiments.r175_shared import TargetStrategy, MSM_WARMUP_BARS
    from experiments.r102_shared import paired_diff

    cand = TargetStrategy(msm_full_target, name="r175_conservative_holdout", warmup=MSM_WARMUP_BARS)
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4_holdout")
    out = []
    for market in (SPOT, FUTURES):
        a = run_slice(cand, btc, OOS_START, None, "holdout", market)
        b = run_slice(ctrl, btc, OOS_START, None, "holdout", market)
        pr = paired_diff(a.daily, b.daily, seed=0)
        exp_ratio = a.mean_abs_exposure / b.mean_abs_exposure if b.mean_abs_exposure else float("nan")
        vol_ratio = a.realized_vol / b.realized_vol if b.realized_vol else float("nan")
        risk_matched = bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1) \
            if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False
        row = {"market": market.name, "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
               "d_sharpe": a.sharpe - b.sharpe, "cand_dd": a.max_drawdown_pct,
               "ctrl_dd": b.max_drawdown_pct, "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
               "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio, "risk_matched": risk_matched,
               "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
               "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0)}
        out.append(row)
        _log(f"holdout:{market.name}")
        print(f"  {market.name:12s} cand_sharpe={a.sharpe:+.3f} ctrl_sharpe={b.sharpe:+.3f} "
              f"d_sharpe={row['d_sharpe']:+.3f}  cand_dd={a.max_drawdown_pct:+.1f} "
              f"ctrl_dd={b.max_drawdown_pct:+.1f} d_dd={row['d_dd']:+.1f}  "
              f"expR={exp_ratio:.2f} volR={vol_ratio:.2f} RM={'Y' if risk_matched else 'n'}  "
              f"boot=[{pr.diff.lo:+.3f},{pr.diff.hi:+.3f}] excl0={row['excludes_zero']}")
    return out


def decide_final_verdict(rows, p1_ok, p2_ok, fee_out, holdout_rows):
    if holdout_rows is None:
        return "NEGATIVE (fall-through, holdout not read)"

    sharpe_ok = all(r["d_sharpe"] >= 0.2 for r in holdout_rows)
    dd_ok = all(r["risk_matched"] and r["d_dd"] < 0 for r in holdout_rows)
    holdout_p1 = sharpe_ok or dd_ok

    fee_ok = all(v["d_sharpe"] >= 0.0 for v in fee_out.values())

    holdout_dir = np.sign(np.mean([r["d_sharpe"] for r in holdout_rows]))
    inner_val = [r for r in rows if r["slice"] == "inner_val"]
    inner_dir = np.sign(np.mean([r["d_sharpe"] for r in inner_val]))

    if holdout_p1 and p2_ok and fee_ok:
        return "PROMOTE"
    if holdout_dir != 0 and holdout_dir != inner_dir:
        return ("NEGATIVE (holdout sign inverts inner-validation's direction -- "
                "third instance of the R-08/R-136 forecast-quality inversion, "
                "this time against MSM)")
    if holdout_dir == inner_dir:
        return "NEGATIVE (directionally consistent with inner-validation but sub-threshold on holdout)"
    return "NEGATIVE (fall-through: no pre-registered clause matched cleanly; see raw table)"


if __name__ == "__main__":
    main()
