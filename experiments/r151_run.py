"""R-151 execution: runs the frozen pre-registration in
``experiments/r151_shared.py`` verbatim. Read that file's docstring for
the decision rule; this file applies it and does not re-derive it.

Inner-validation only. There is no code path here that slices at or after
``OOS_START``: the only window constants imported are ``INNER_VAL_START``
and ``INNER_VAL_END``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from tradebot import inference  # noqa: E402

from r145_shared import (  # noqa: E402
    CONSERVATIVE_THRESHOLDS,
    D_SHARPE_FLOOR,
    EXPOSURE_MATCH_TOL_PCT,
    INNER_VAL_END,
    INNER_VAL_START,
    TURNOVER_SAVINGS_KILL,
    fut_market,
    load_btc,
    plain_v4_period,
    run_hybrid_backtest,
    spot_market,
)
# The comparison code is imported from R-145's own branch file, not
# reimplemented, so the mismatch percentages below are produced by the
# identical measurement path that produced R-145's 1.90-4.81% table.
from r145_conservative import (  # noqa: E402
    FEE_TIERS,
    PRIMARY_THRESHOLD,
    annualized_vol,
    make_route_builder,
    paired_daily,
    plain_target_slice,
)
from r151_shared import (  # noqa: E402
    ADOPT_TOL_PCT,
    ARMS,
    DEADBAND_REF_LEVERAGE,
    EQUIV_TOL,
    FIDELITY_TOL,
    PARTIAL_MIN_REDUCTION,
    REJECT_MAX_MEDIAN_REDUCTION,
    arm_kwargs,
    degenerate_all_futures,
    degenerate_all_spot,
    run_hybrid_backtest_v2,
)

N_CONFIGS = 0  # incremented by every hybrid run below; reported at the end


def _count(fn):
    def wrapped(*a, **kw):
        global N_CONFIGS
        N_CONFIGS += 1
        return fn(*a, **kw)
    return wrapped


run_v2 = _count(run_hybrid_backtest_v2)
run_frozen = _count(run_hybrid_backtest)


# ---------------------------------------------------------------- T4 fidelity

def t4_fidelity(df, funding) -> dict:
    """Arm "leg" must reproduce the frozen r145 harness exactly."""
    out = {}
    for tier, spot_fee in FEE_TIERS:
        frozen = run_frozen(
            df, make_route_builder(PRIMARY_THRESHOLD), spot_market(spot_fee), fut_market(),
            funding=funding, start=INNER_VAL_START, end=INNER_VAL_END,
        )
        mine = run_v2(
            df, make_route_builder(PRIMARY_THRESHOLD), spot_market(spot_fee), fut_market(),
            funding=funding, start=INNER_VAL_START, end=INNER_VAL_END,
            **arm_kwargs("leg"),
        )
        max_eq = float((frozen.equity - mine.equity).abs().max())
        out[tier] = dict(
            d_balance=abs(frozen.final_balance - mine.final_balance) / max(frozen.final_balance, 1.0),
            d_fees=abs(frozen.fees_paid - mine.fees_paid) / max(frozen.fees_paid, 1e-9),
            d_funding=abs(frozen.funding_paid - mine.funding_paid) / max(abs(frozen.funding_paid), 1e-9),
            d_fills=(frozen.fills_spot - mine.fills_spot, frozen.fills_fut - mine.fills_fut),
            max_eq_diff=max_eq,
            frozen_balance=frozen.final_balance,
            frozen_fills=(frozen.fills_spot, frozen.fills_fut),
        )
    return out


# ------------------------------------------------------------ T3 equivalence

def t3_equivalence(df, funding) -> dict:
    """All-futures (must hold under every arm) and all-spot (expected to
    break under a shared base -- quantify, do not wave through)."""
    plain_fut = plain_v4_period(df, fut_market(), funding, INNER_VAL_START, INNER_VAL_END)
    plain_spot = plain_v4_period(df, spot_market(), None, INNER_VAL_START, INNER_VAL_END)
    out = {}
    for arm in ("leg", "shared"):
        hf = run_v2(df, degenerate_all_futures, spot_market(), fut_market(), funding=funding,
                    start=INNER_VAL_START, end=INNER_VAL_END, **arm_kwargs(arm))
        hs = run_v2(df, degenerate_all_spot, spot_market(), fut_market(), funding=funding,
                    start=INNER_VAL_START, end=INNER_VAL_END, **arm_kwargs(arm))
        out[arm] = dict(
            fut_rel=abs(hf.final_balance - plain_fut.final_balance) / max(plain_fut.final_balance, 1.0),
            spot_rel=abs(hs.final_balance - plain_spot.final_balance) / max(plain_spot.final_balance, 1.0),
            fut_balance=hf.final_balance, plain_fut_balance=plain_fut.final_balance,
            spot_balance=hs.final_balance, plain_spot_balance=plain_spot.final_balance,
            fut_fills=hf.fills_fut, spot_fills=hs.fills_spot,
        )
    return out


# ------------------------------------------------------------------- the grid

def run_cell(df, funding, spot_fee, threshold, arm, plain, plain_target) -> dict:
    """R-145's own `run_cell`, with the broker arm as the only addition."""
    hybrid = run_v2(
        df, make_route_builder(threshold), spot_market(spot_fee), fut_market(),
        funding=funding, start=INNER_VAL_START, end=INNER_VAL_END, **arm_kwargs(arm),
    )
    hyb_daily, pln_daily = paired_daily(hybrid.equity, plain.equity)

    spot_frac = np.clip(plain_target, 0.0, threshold)
    fut_frac = plain_target - spot_frac
    hybrid_tim = float(np.mean((spot_frac > 0) | (fut_frac > 0)))
    plain_tim = float(np.mean(plain_target != 0))
    tim_rel_pct = (abs(hybrid_tim - plain_tim) / plain_tim * 100.0) if plain_tim > 0 else float("nan")

    hybrid_vol = annualized_vol(pd.Series(hyb_daily))
    plain_vol = annualized_vol(pd.Series(pln_daily))
    vol_rel_pct = (abs(hybrid_vol - plain_vol) / plain_vol * 100.0) if plain_vol > 0 else float("nan")

    return dict(
        arm=arm, threshold=threshold, tier_fee=spot_fee,
        final_balance=hybrid.final_balance,
        fills_spot=hybrid.fills_spot, fills_fut=hybrid.fills_fut,
        retargets_spot=hybrid.retargets_spot, retargets_fut=hybrid.retargets_fut,
        absorbed_spot=hybrid.absorbed_spot, absorbed_fut=hybrid.absorbed_fut,
        absorb_rate_spot=(hybrid.absorbed_spot / hybrid.retargets_spot
                          if hybrid.retargets_spot else float("nan")),
        absorb_rate_fut=(hybrid.absorbed_fut / hybrid.retargets_fut
                         if hybrid.retargets_fut else float("nan")),
        mean_threshold_spot=hybrid.mean_threshold_spot,
        mean_threshold_fut=hybrid.mean_threshold_fut,
        hybrid_vol=hybrid_vol, plain_vol=plain_vol, vol_rel_pct=vol_rel_pct,
        tim_rel_pct=tim_rel_pct,
        fees_paid=hybrid.fees_paid, funding_paid=hybrid.funding_paid,
        liquidated=hybrid.liquidated,
        hyb_daily=hyb_daily, pln_daily=pln_daily,
        plain_fees=plain.fees_paid, plain_funding=plain.funding_paid,
    )


def main() -> None:
    df, funding, label = load_btc()
    print(f"BTC: {len(df):,} bars ({label}); funding present: {funding is not None}")
    print(f"window: {INNER_VAL_START} -> {INNER_VAL_END} (inner-validation only)")
    print(f"shared-base reference leverage: {DEADBAND_REF_LEVERAGE}\n")

    # ---- fidelity gates first: nothing below is trusted without them ----
    print("=== T4 fidelity: arm 'leg' vs the FROZEN r145_shared harness ===")
    t4 = t4_fidelity(df, funding)
    t4_ok = True
    for tier, d in t4.items():
        ok = (d["d_balance"] < FIDELITY_TOL and d["d_fees"] < FIDELITY_TOL
              and d["d_funding"] < FIDELITY_TOL and d["d_fills"] == (0, 0)
              and d["max_eq_diff"] < FIDELITY_TOL * max(d["frozen_balance"], 1.0))
        t4_ok &= ok
        print(f"  [{tier}] frozen=${d['frozen_balance']:,.6f} fills={d['frozen_fills']}  "
              f"rel_d_balance={d['d_balance']:.3e} rel_d_fees={d['d_fees']:.3e} "
              f"rel_d_funding={d['d_funding']:.3e} d_fills={d['d_fills']} "
              f"max|dequity|={d['max_eq_diff']:.3e}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  T4: {'PASS' if t4_ok else 'FAIL'}\n")

    print("=== T3 degenerate equivalence (all-futures must hold; all-spot is the disclosed cost) ===")
    t3 = t3_equivalence(df, funding)
    for arm, d in t3.items():
        print(f"  [arm={arm}] all-futures rel={d['fut_rel']:.3e} "
              f"(hybrid=${d['fut_balance']:,.2f} vs plain=${d['plain_fut_balance']:,.2f}, "
              f"{d['fut_fills']} fills)")
        print(f"  [arm={arm}] all-spot    rel={d['spot_rel']:.3e} "
              f"(hybrid=${d['spot_balance']:,.2f} vs plain=${d['plain_spot_balance']:,.2f}, "
              f"{d['spot_fills']} fills)")
    t3a_ok = t3["shared"]["fut_rel"] < EQUIV_TOL
    print(f"  T3a (all-futures under 'shared' < {EQUIV_TOL:.0e}): "
          f"{'PASS' if t3a_ok else 'FAIL'}\n")

    # ---- the grid ----
    plain = plain_v4_period(df, fut_market(), funding, INNER_VAL_START, INNER_VAL_END)
    plain_target = plain_target_slice(df, INNER_VAL_START, INNER_VAL_END)
    assert len(plain_target) == len(plain.equity)
    print(f"plain all-futures v4 baseline: ${plain.final_balance:,.2f} "
          f"fees=${plain.fees_paid:,.2f} funding=${plain.funding_paid:,.2f}\n")

    cells: dict[tuple[str, str, float], dict] = {}
    for arm in ARMS:
        for tier, spot_fee in FEE_TIERS:
            for threshold in CONSERVATIVE_THRESHOLDS:
                cells[(arm, tier, threshold)] = run_cell(
                    df, funding, spot_fee, threshold, arm, plain, plain_target)

    print("=== realized-volatility mismatch vs the plain all-futures baseline (R-145 criterion 3) ===")
    print(f"{'tier':<14}{'thr':>5}  " + "".join(f"{a:>14}" for a in ARMS)
          + f"{'B vs A':>12}{'C vs A':>10}")
    reductions_b, reductions_c = [], []
    for tier, _ in FEE_TIERS:
        for threshold in CONSERVATIVE_THRESHOLDS:
            a = cells[("leg", tier, threshold)]["vol_rel_pct"]
            b = cells[("shared", tier, threshold)]["vol_rel_pct"]
            c = cells[("shared+hc", tier, threshold)]["vol_rel_pct"]
            rb = (a - b) / a if a > 0 else float("nan")
            rc = (a - c) / a if a > 0 else float("nan")
            reductions_b.append(rb)
            reductions_c.append(rc)
            print(f"{tier:<14}{threshold:>5.1f}  {a:>13.4f}%{b:>13.4f}%{c:>13.4f}%"
                  f"{rb*100:>11.1f}%{rc*100:>9.1f}%")
    print()

    print("=== throttle instrumentation (threshold=1.0): what the deadband actually did ===")
    for tier, _ in FEE_TIERS:
        for arm in ARMS:
            c = cells[(arm, tier, PRIMARY_THRESHOLD)]
            print(f"  [{tier} arm={arm:<10}] spot: {c['retargets_spot']:>5} re-targets, "
                  f"{c['absorbed_spot']:>5} absorbed ({c['absorb_rate_spot']*100:5.1f}%), "
                  f"mean $threshold={c['mean_threshold_spot']:,.2f} | "
                  f"fut: {c['retargets_fut']:>5} re-targets, {c['absorbed_fut']:>5} absorbed "
                  f"({c['absorb_rate_fut']*100:5.1f}%), mean $threshold={c['mean_threshold_fut']:,.2f}")
            print(f"  {'':>{len(tier)+8}}  fills: spot={c['fills_spot']} fut={c['fills_fut']} "
                  f"(total {c['fills_spot']+c['fills_fut']})  final=${c['final_balance']:,.2f}")
    print()

    # ---- headline verdict ----
    b_cells = [cells[("shared", tier, t)] for tier, _ in FEE_TIERS for t in CONSERVATIVE_THRESHOLDS]
    all_within = all(c["vol_rel_pct"] <= ADOPT_TOL_PCT for c in b_cells)
    all_halved = all(r >= PARTIAL_MIN_REDUCTION for r in reductions_b)
    median_red = float(np.median(reductions_b))
    if all_within and t3a_ok and t4_ok:
        headline = "ADOPT"
    elif t3a_ok and t4_ok and all_halved:
        headline = "PARTIAL"
    elif not (t3a_ok and t4_ok) or median_red < REJECT_MAX_MEDIAN_REDUCTION:
        headline = "REJECT"
    else:
        headline = "PARTIAL-BELOW-BAR"
    print("=== HEADLINE (pre-registered rule) ===")
    print(f"  T4={t4_ok}  T3a={t3a_ok}  "
          f"all six arm-B cells <= {ADOPT_TOL_PCT}%: {all_within}  "
          f"all six reduced >= {PARTIAL_MIN_REDUCTION:.0%}: {all_halved}  "
          f"median reduction: {median_red:.1%}")
    print(f"  --> {headline}\n")

    # ---- secondary: did the defect change R-145's verdict? ----
    print("=== SECONDARY (pre-registered): does the fix move R-145's own rejection? ===")
    for tier, _ in FEE_TIERS:
        for arm in ("leg", "shared"):
            c = cells[(arm, tier, PRIMARY_THRESHOLD)]
            boot = inference.paired_bootstrap(c["hyb_daily"], c["pln_daily"],
                                              inference.annualized_sharpe)
            extra_fees = c["fees_paid"] - c["plain_fees"]
            funding_saved = c["plain_funding"] - c["funding_paid"]
            ratio = (extra_fees / funding_saved) if funding_saved != 0 else float("inf")
            g1 = boot.diff.point >= D_SHARPE_FLOOR and boot.diff.lo > 0.0
            g2 = ratio < TURNOVER_SAVINGS_KILL
            print(f"  [{tier} arm={arm:<7}] d_sharpe={boot.diff.point:+.3f} "
                  f"CI=[{boot.diff.lo:+.3f}, {boot.diff.hi:+.3f}]  "
                  f"extra_fees=${extra_fees:,.2f} funding_saved=${funding_saved:,.2f} "
                  f"ratio={ratio:.3f}  G1={g1} G2={g2}")
    print()
    print(f"vol-match tolerance in force: {EXPOSURE_MATCH_TOL_PCT}%  "
          f"(R-145's own criterion 3 constant, unchanged)")
    print(f"hybrid runs executed (configs evaluated): {N_CONFIGS}")


if __name__ == "__main__":
    main()
