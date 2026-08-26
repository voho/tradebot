"""R-154 execution: runs the frozen pre-registration in
``experiments/r154_shared.py`` verbatim. Read that file's docstring for
the decision rule; this file applies it and does not re-derive it.

Inner-validation only -- no code path here slices at or after
``OOS_START``.
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
    INNER_VAL_END,
    INNER_VAL_START,
    TURNOVER_SAVINGS_KILL,
    fut_market,
    load_btc,
    plain_v4_period,
    spot_market,
)
from r145_conservative import (  # noqa: E402
    FEE_TIERS,
    PRIMARY_THRESHOLD,
    annualized_vol,
    make_route_builder,
    paired_daily,
    plain_target_slice,
)
from r151_shared import (  # noqa: E402
    DEADBAND_REF_LEVERAGE,
    run_hybrid_backtest_v2,
)
from r154_shared import (  # noqa: E402
    ADOPT_TOL_PCT,
    ARM_B_MISMATCH_PCT,
    EQUIV_TOL,
    FIDELITY_TOL,
    PARTIAL_MIN_IMPROVEMENT,
    REJECT_MAX_MEDIAN_IMPROVEMENT,
    TIE_BAND_PCT,
    HybridBrokerB45Only,
    HybridBrokerB46Only,
    HybridBrokerConservative,
    HybridBrokerNovel,
    HybridBrokerV3,
    degenerate_all_futures,
    run_hybrid_backtest_v3,
)

N_CONFIGS = 0


def _count(fn):
    def wrapped(*a, **kw):
        global N_CONFIGS
        N_CONFIGS += 1
        return fn(*a, **kw)
    return wrapped


run_v2 = _count(run_hybrid_backtest_v2)
run_v3 = _count(run_hybrid_backtest_v3)

BRANCHES = {
    "conservative": HybridBrokerConservative,
    "novel": HybridBrokerNovel,
}
DIAGNOSTIC = {
    "+b45-only": HybridBrokerB45Only,
    "+b46-only": HybridBrokerB46Only,
}


def _factory(cls, spot_mkt, fut_mkt, **kw):
    def build():
        return cls(spot=spot_mkt, fut=fut_mkt, start_balance=1_000.0,
                  deadband_base="shared", haircut_base="leg", **kw)
    return build


# ---------------------------------------------------------- fidelity gate

def fidelity_gate(df, funding) -> bool:
    print("=== fidelity gate: HybridBrokerV3(aggregate_throttle=False) vs R-151's own arm B ===")
    ok = True
    for tier, spot_fee in FEE_TIERS:
        ref = run_v2(
            df, make_route_builder(PRIMARY_THRESHOLD), spot_market(spot_fee), fut_market(),
            funding=funding, start=INNER_VAL_START, end=INNER_VAL_END,
            deadband_base="shared", haircut_base="leg",
        )
        mine = run_v3(
            df, make_route_builder(PRIMARY_THRESHOLD), spot_market(spot_fee), fut_market(),
            _factory(HybridBrokerV3, spot_market(spot_fee), fut_market()),
            funding=funding, start=INNER_VAL_START, end=INNER_VAL_END,
        )
        d_balance = abs(ref.final_balance - mine.final_balance) / max(ref.final_balance, 1.0)
        d_fees = abs(ref.fees_paid - mine.fees_paid) / max(ref.fees_paid, 1e-9)
        d_funding = abs(ref.funding_paid - mine.funding_paid) / max(abs(ref.funding_paid), 1e-9)
        d_fills = (ref.fills_spot - mine.fills_spot, ref.fills_fut - mine.fills_fut)
        cell_ok = (d_balance < FIDELITY_TOL and d_fees < FIDELITY_TOL
                  and d_funding < FIDELITY_TOL and d_fills == (0, 0))
        ok &= cell_ok
        print(f"  [{tier}] ref=${ref.final_balance:,.6f} fills={ref.fills_spot, ref.fills_fut}  "
              f"rel_d_balance={d_balance:.3e} rel_d_fees={d_fees:.3e} "
              f"rel_d_funding={d_funding:.3e} d_fills={d_fills}  -> {'PASS' if cell_ok else 'FAIL'}")

    print("\n=== T3a: all-futures degenerate route reproduces plain futures v4 (every branch) ===")
    plain_fut = plain_v4_period(df, fut_market(), funding, INNER_VAL_START, INNER_VAL_END)
    for tier, spot_fee in FEE_TIERS:
        for name, cls in {**BRANCHES, **DIAGNOSTIC, "unfixed": HybridBrokerV3}.items():
            hf = run_v3(
                df, degenerate_all_futures, spot_market(spot_fee), fut_market(),
                _factory(cls, spot_market(spot_fee), fut_market()),
                funding=funding, start=INNER_VAL_START, end=INNER_VAL_END,
            )
            rel = abs(hf.final_balance - plain_fut.final_balance) / max(plain_fut.final_balance, 1.0)
            cell_ok = rel < EQUIV_TOL
            ok &= cell_ok
            print(f"  [{tier} {name:<12}] rel={rel:.3e}  -> {'PASS' if cell_ok else 'FAIL'}")
    print(f"\nfidelity gate: {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------- the grid

def run_cell(df, funding, spot_fee, threshold, cls, plain, plain_target) -> dict:
    hybrid = run_v3(
        df, make_route_builder(threshold), spot_market(spot_fee), fut_market(),
        _factory(cls, spot_market(spot_fee), fut_market()),
        funding=funding, start=INNER_VAL_START, end=INNER_VAL_END,
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
        threshold=threshold, tier_fee=spot_fee,
        final_balance=hybrid.final_balance,
        fills_spot=hybrid.fills_spot, fills_fut=hybrid.fills_fut,
        retargets_spot=hybrid.retargets_spot, retargets_fut=hybrid.retargets_fut,
        absorbed_spot=hybrid.absorbed_spot, absorbed_fut=hybrid.absorbed_fut,
        absorb_rate_spot=(hybrid.absorbed_spot / hybrid.retargets_spot
                          if hybrid.retargets_spot else float("nan")),
        absorb_rate_fut=(hybrid.absorbed_fut / hybrid.retargets_fut
                         if hybrid.retargets_fut else float("nan")),
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
    print(f"window: {INNER_VAL_START} -> {INNER_VAL_END} (inner-validation only)\n")

    gate_ok = fidelity_gate(df, funding)
    if not gate_ok:
        print("FIDELITY GATE FAILED -- stopping. No headline number below is trustworthy.")
        return

    plain = plain_v4_period(df, fut_market(), funding, INNER_VAL_START, INNER_VAL_END)
    plain_target = plain_target_slice(df, INNER_VAL_START, INNER_VAL_END)
    assert len(plain_target) == len(plain.equity)
    print(f"plain all-futures v4 baseline: ${plain.final_balance:,.2f} "
          f"fees=${plain.fees_paid:,.2f} funding=${plain.funding_paid:,.2f}\n")

    cells: dict[tuple[str, str, float], dict] = {}
    for name, cls in BRANCHES.items():
        for tier, spot_fee in FEE_TIERS:
            for threshold in CONSERVATIVE_THRESHOLDS:
                cells[(name, tier, threshold)] = run_cell(
                    df, funding, spot_fee, threshold, cls, plain, plain_target)

    print("=== headline: realized-volatility mismatch vs plain futures baseline (R-145 criterion 3) ===")
    print(f"{'tier':<14}{'thr':>5}  {'arm B (R-151)':>15}{'conservative':>15}{'novel':>15}"
          f"{'cons improve':>15}{'novel improve':>15}")
    improve_cons, improve_novel = [], []
    for tier, _ in FEE_TIERS:
        for threshold in CONSERVATIVE_THRESHOLDS:
            b = ARM_B_MISMATCH_PCT[(tier, threshold)]
            c = cells[("conservative", tier, threshold)]["vol_rel_pct"]
            n = cells[("novel", tier, threshold)]["vol_rel_pct"]
            ic = (b - c) / b if b > 0 else float("nan")
            inn = (b - n) / b if b > 0 else float("nan")
            improve_cons.append(ic)
            improve_novel.append(inn)
            print(f"{tier:<14}{threshold:>5.1f}  {b:>14.4f}%{c:>14.4f}%{n:>14.4f}%"
                  f"{ic*100:>14.1f}%{inn*100:>14.1f}%")
    print()

    print("=== decomposition (primary threshold, both fee tiers): which defect drives what ===")
    for tier, spot_fee in FEE_TIERS:
        for name, cls in DIAGNOSTIC.items():
            c = run_cell(df, funding, spot_fee, PRIMARY_THRESHOLD, cls, plain, plain_target)
            b = ARM_B_MISMATCH_PCT[(tier, PRIMARY_THRESHOLD)]
            print(f"  [{tier} {name:<12}] vol_rel={c['vol_rel_pct']:.4f}% "
                  f"(arm B was {b:.4f}%)  spot absorb={c['absorb_rate_spot']*100:5.1f}% "
                  f"fut absorb={c['absorb_rate_fut']*100:5.1f}%  fills={c['fills_spot']}/{c['fills_fut']}")
    print()

    print("=== throttle instrumentation at threshold=1.2 (where B-45 is active) ===")
    for tier, _ in FEE_TIERS:
        for name in BRANCHES:
            c = cells[(name, tier, 1.2)]
            print(f"  [{tier} {name:<12}] spot: {c['retargets_spot']:>5} re-targets, "
                  f"{c['absorbed_spot']:>5} absorbed ({c['absorb_rate_spot']*100:5.1f}%) | "
                  f"fut: {c['retargets_fut']:>5} re-targets, {c['absorbed_fut']:>5} absorbed "
                  f"({c['absorb_rate_fut']*100:5.1f}%)  fills={c['fills_spot']}/{c['fills_fut']} "
                  f"final=${c['final_balance']:,.2f}")
    print()

    # ---- headline verdict per branch ----
    def verdict(name, cells_for, improves):
        vols = [cells_for[(name, tier, t)]["vol_rel_pct"] for tier, _ in FEE_TIERS
               for t in CONSERVATIVE_THRESHOLDS]
        all_within = all(v <= ADOPT_TOL_PCT for v in vols)
        all_improved = all(i >= PARTIAL_MIN_IMPROVEMENT for i in improves)
        median_improve = float(np.median(improves))
        if all_within:
            return "ADOPT", median_improve
        if all_improved:
            return "PARTIAL", median_improve
        if median_improve < REJECT_MAX_MEDIAN_IMPROVEMENT:
            return "REJECT", median_improve
        return "PARTIAL-BELOW-BAR", median_improve

    v_cons = verdict("conservative", cells, improve_cons)
    v_novel = verdict("novel", cells, improve_novel)
    print("=== HEADLINE (pre-registered rule) ===")
    print(f"  conservative: median improvement over arm B = {v_cons[1]:.1%}  -> {v_cons[0]}")
    print(f"  novel:        median improvement over arm B = {v_novel[1]:.1%}  -> {v_novel[0]}")

    median_cons = float(np.median([cells[("conservative", tier, t)]["vol_rel_pct"]
                                   for tier, _ in FEE_TIERS for t in CONSERVATIVE_THRESHOLDS]))
    median_novel = float(np.median([cells[("novel", tier, t)]["vol_rel_pct"]
                                    for tier, _ in FEE_TIERS for t in CONSERVATIVE_THRESHOLDS]))
    diff_pp = median_cons - median_novel
    if abs(diff_pp) < TIE_BAND_PCT:
        winner = "conservative (tie, default)"
    elif diff_pp > 0:
        winner = "novel (lower median mismatch)"
    else:
        winner = "conservative (lower median mismatch)"
    print(f"  median vol_rel_pct: conservative={median_cons:.4f}%  novel={median_novel:.4f}%  "
          f"diff={diff_pp:+.4f}pp  -> WINNER: {winner}\n")

    # ---- secondary: does the fix flip R-145's own verdict? (== B-47) ----
    print("=== SECONDARY / B-47 (pre-registered): does either fix move R-145's own rejection? ===")
    for tier, _ in FEE_TIERS:
        for name in BRANCHES:
            c = cells[(name, tier, PRIMARY_THRESHOLD)]
            boot = inference.paired_bootstrap(c["hyb_daily"], c["pln_daily"],
                                              inference.annualized_sharpe)
            extra_fees = c["fees_paid"] - c["plain_fees"]
            funding_saved = c["plain_funding"] - c["funding_paid"]
            ratio = (extra_fees / funding_saved) if funding_saved != 0 else float("inf")
            g1 = boot.diff.point >= D_SHARPE_FLOOR and boot.diff.lo > 0.0
            g2 = ratio < TURNOVER_SAVINGS_KILL
            print(f"  [{tier} {name:<12}] d_sharpe={boot.diff.point:+.3f} "
                  f"CI=[{boot.diff.lo:+.3f}, {boot.diff.hi:+.3f}]  "
                  f"extra_fees=${extra_fees:,.2f} funding_saved=${funding_saved:,.2f} "
                  f"ratio={ratio:.3f}  G1={g1} G2={g2}")
    print()
    print(f"hybrid runs executed (configs evaluated): {N_CONFIGS}")


if __name__ == "__main__":
    main()
