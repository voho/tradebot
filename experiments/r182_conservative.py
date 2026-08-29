"""R-182 CONSERVATIVE branch: literal constant-BASE core-satellite venue
routing.

Executes the *conservative* half of the frozen pre-registration in
``experiments/r182_direction.md`` (read that file for the full
pre-registration -- this file does not restate it beyond the decision-rule
constants used below). Verbatim per that freeze:

- mechanism: ``route_constant_base`` (imported from ``r182_shared``, not
  reimplemented) -- a single frozen constant BASE held on spot while
  ``target > 0``, ``target - BASE`` (SIGNED) on futures.
- BASE swept over ``{0.10, 0.15, 0.25, 0.43}`` (the same four candidates
  ``r182_shared.py``'s own Step-2 measurement used on inner-train).
- BASE is selected against **inner-validation only**
  (``INNER_VAL_START`` .. ``INNER_VAL_END``, 2021-01-01 -> 2022-12-31).
  Inner-train is calibration-only per the pre-registration -- this file
  never selects on it. It also never imports, slices, or reads anything at
  or after ``OOS_START`` -- there is no holdout-reading code path here by
  construction, not merely by discipline.
- decision rule: the frozen falsification test (Step 3, conservative
  section) and the 5-clause promotion bar (Step 3, both branches), applied
  exactly, not re-derived or loosened here.

This file does not modify ``r182_shared.py`` and does not commit anything
-- per ROUTINE.md's parallel-branch rules, the operator merges and commits
once after both branches (this one and the "novel" branch) report.
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

from tradebot import inference
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

from r182_shared import (  # noqa: E402
    D_SHARPE_FLOOR,
    EXPOSURE_MATCH_TOL_PCT,
    INNER_VAL_END,
    INNER_VAL_START,
    TURNOVER_SAVINGS_KILL,
    compute_target,
    fut_market,
    load_btc,
    load_eth,
    plain_v4_period,
    route_constant_base,
    run_signed_hybrid_backtest,
    spot_market,
)
from r145_shared import SPOT_FEE_BASE, SPOT_FEE_REAL  # noqa: E402

FEE_TIERS = [("base_0.10pct", SPOT_FEE_BASE), ("real_0.40pct", SPOT_FEE_REAL)]
BASE_CANDIDATES = (0.10, 0.15, 0.25, 0.43)


def make_route_builder(base: float):
    """`run_signed_hybrid_backtest` calls its `route_builder` with the
    sliced DataFrame, not a precomputed target array -- composes
    `route_constant_base` with `compute_target` on that exact sliced
    frame, matching `r182_shared._causality_truncation_check`'s own
    convention (never the unsliced df -- required by
    `run_signed_hybrid_backtest`'s own docstring on why `route_builder`
    gets the sliced frame in the first place: unbounded-memory EWM in
    `KellyRegime.prepare`, though `route_constant_base` itself has no EWM,
    consistency with the shared harness is kept regardless).
    """
    def build(frame: pd.DataFrame):
        return route_constant_base(compute_target(frame), base)
    return build


# --------------------------------------------------------------- alignment

def plain_target_slice(df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    """`target`, computed and trimmed with EXACTLY `plain_v4_period`'s own
    warmup-prefix/trim convention, so the returned array is bar-for-bar
    aligned with both `plain_v4_period`'s and `run_signed_hybrid_backtest`'s
    trimmed equity curves over the same window.
    """
    strategy = KellyRegimeV4()
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    frame = df.iloc[lo - pre: hi]
    target = compute_target(frame)
    return target[pre:] if pre else target


def annualized_vol(daily: pd.Series) -> float:
    if len(daily) < 2:
        return float("nan")
    return float(daily.std(ddof=1) * np.sqrt(inference.DAYS_PER_YEAR))


def paired_daily(a_equity: pd.Series, b_equity: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Daily returns for both curves, aligned on the shared calendar days
    (an inner join) before handing off to `paired_bootstrap`, which
    requires equal-length aligned series.
    """
    a = inference.daily_returns(a_equity)
    b = inference.daily_returns(b_equity)
    idx = a.index.intersection(b.index)
    if len(idx) < max(len(a), len(b)) - 2:
        raise ValueError(
            f"daily-return alignment lost {max(len(a), len(b)) - len(idx)} of "
            f"{max(len(a), len(b))} days -- hybrid and plain periods diverge"
        )
    return a.reindex(idx).to_numpy(dtype=float), b.reindex(idx).to_numpy(dtype=float)


# ------------------------------------------------------------------ cells

def run_cell(btc_df: pd.DataFrame, btc_funding: pd.Series, spot_fee: float,
            base: float, plain, plain_target: np.ndarray) -> dict:
    hybrid = run_signed_hybrid_backtest(
        btc_df, make_route_builder(base), spot_market(spot_fee), fut_market(),
        funding=btc_funding, start=INNER_VAL_START, end=INNER_VAL_END,
    )

    hyb_daily, pln_daily = paired_daily(hybrid.equity, plain.equity)
    boot = inference.paired_bootstrap(hyb_daily, pln_daily, inference.annualized_sharpe)

    extra_fees = hybrid.fees_paid - plain.fees_paid
    funding_saved = plain.funding_paid - hybrid.funding_paid
    turnover_ratio = (extra_fees / funding_saved) if funding_saved != 0 else float("inf")

    # Mechanical routing check for THIS route: spot_frac + fut_frac ==
    # target at every bar (clause 4 of the promotion bar is stated for
    # ETH, checked in the ETH section below; the identity holds by
    # construction for BTC too since route_constant_base's fut leg is
    # literally `target - spot`, checked here directly rather than
    # assumed).
    route = route_constant_base(plain_target, base)
    max_route_err = float(np.max(np.abs((route.spot_frac + route.fut_frac) - plain_target)))

    hybrid_tim = float(np.mean((route.spot_frac != 0.0) | (route.fut_frac != 0.0)))
    plain_tim = float(np.mean(plain_target != 0))
    tim_rel_pct = (abs(hybrid_tim - plain_tim) / plain_tim * 100.0) if plain_tim > 0 else float("nan")

    hybrid_vol = annualized_vol(pd.Series(hyb_daily))
    plain_vol = annualized_vol(pd.Series(pln_daily))
    vol_rel_pct = (abs(hybrid_vol - plain_vol) / plain_vol * 100.0) if plain_vol > 0 else float("nan")

    return dict(
        base=base,
        hybrid=hybrid,
        d_sharpe=boot.diff.point,
        ci_lo=boot.diff.lo,
        ci_hi=boot.diff.hi,
        p_positive=boot.p_positive,
        sharpe_hybrid=boot.stat_a,
        sharpe_plain=boot.stat_b,
        extra_fees=extra_fees,
        funding_saved=funding_saved,
        turnover_ratio=turnover_ratio,
        max_route_err=max_route_err,
        hybrid_tim=hybrid_tim,
        plain_tim=plain_tim,
        tim_rel_pct=tim_rel_pct,
        hybrid_vol=hybrid_vol,
        plain_vol=plain_vol,
        vol_rel_pct=vol_rel_pct,
        liquidated=hybrid.liquidated,
        fills_spot=hybrid.fills_spot,
        fills_fut=hybrid.fills_fut,
    )


def run_conservative() -> dict:
    btc_df, btc_funding, btc_label = load_btc()
    eth_df, eth_funding, eth_label = load_eth()
    assert eth_funding is None, "ETH ceiling violated: a funding series appeared for ETH"

    # Plain (all-futures) v4 does not depend on spot fee or BASE at all
    # (it never touches the spot leg) -- computed once, reused for every
    # cell.
    plain = plain_v4_period(btc_df, fut_market(), btc_funding, INNER_VAL_START, INNER_VAL_END)
    plain_target = plain_target_slice(btc_df, INNER_VAL_START, INNER_VAL_END)
    assert len(plain_target) == len(plain.equity), (
        f"target/equity length mismatch: {len(plain_target)} vs {len(plain.equity)}"
    )

    grid: dict[str, dict[float, dict]] = {}
    n_configs = 0
    for tier_label, spot_fee in FEE_TIERS:
        grid[tier_label] = {}
        for base in BASE_CANDIDATES:
            cell = run_cell(btc_df, btc_funding, spot_fee, base, plain, plain_target)
            grid[tier_label][base] = cell
            n_configs += 1

    # --------------------------------------------------- BASE selection
    # Select the best BASE against inner-validation ONLY (never
    # inner-train, which is calibration-only per the freeze). "Best" is
    # defined as the highest mean d_sharpe across both fee tiers on
    # inner-validation -- the only promotion-relevant BTC comparison this
    # branch is scored against, so it is also the only reasonable
    # selection statistic; ties broken by the lower (safer) turnover
    # ratio at the real 0.40% tier.
    def mean_d_sharpe(base: float) -> float:
        return float(np.mean([grid[tier][base]["d_sharpe"] for tier, _ in FEE_TIERS]))

    def tie_break(base: float) -> float:
        return grid["real_0.40pct"][base]["turnover_ratio"]

    selected_base = max(BASE_CANDIDATES, key=lambda b: (mean_d_sharpe(b), -tie_break(b)))

    # ETH mechanism check, selected BASE only, both fee tiers (routing
    # itself does not depend on fee rate, but the run must still complete
    # cleanly -- i.e. not liquidate -- at both). ETH is funding-free (no
    # ETH perpetual funding series is committed in this repo) so ETH is
    # never the dollar-savings gate -- mechanism/replication only.
    eth_target = plain_target_slice(eth_df, INNER_VAL_START, INNER_VAL_END)
    eth_route = route_constant_base(eth_target, selected_base)
    eth_route_err = float(np.max(np.abs((eth_route.spot_frac + eth_route.fut_frac) - eth_target)))

    eth_checks = {}
    for tier_label, spot_fee in FEE_TIERS:
        eth_hybrid = run_signed_hybrid_backtest(
            eth_df, make_route_builder(selected_base), spot_market(spot_fee), fut_market(),
            funding=None, start=INNER_VAL_START, end=INNER_VAL_END,
        )
        eth_checks[tier_label] = dict(
            route_err=eth_route_err,
            liquidated=eth_hybrid.liquidated,
            final_balance=eth_hybrid.final_balance,
            n_bars=len(eth_target),
            all_finite=bool(np.all(np.isfinite(eth_hybrid.equity.to_numpy()))),
        )

    return dict(
        btc_label=btc_label, eth_label=eth_label,
        plain=plain, grid=grid, eth_checks=eth_checks,
        n_configs=n_configs, selected_base=selected_base,
    )


def _fmt_cell(tier: str, cell: dict) -> str:
    return (
        f"  [{tier}] BASE={cell['base']:.2f}  "
        f"d_sharpe={cell['d_sharpe']:+.3f} CI=[{cell['ci_lo']:+.3f}, {cell['ci_hi']:+.3f}] "
        f"p_pos={cell['p_positive']:.3f}  "
        f"sharpe(hybrid)={cell['sharpe_hybrid']:.3f} sharpe(plain)={cell['sharpe_plain']:.3f}\n"
        f"      extra_fees=${cell['extra_fees']:,.2f} funding_saved=${cell['funding_saved']:,.2f} "
        f"turnover_ratio={cell['turnover_ratio']:.3f}  liquidated={cell['liquidated']}  "
        f"max_route_err={cell['max_route_err']:.2e}\n"
        f"      time-in-market: hybrid={cell['hybrid_tim']*100:.3f}% plain={cell['plain_tim']*100:.3f}% "
        f"(rel diff {cell['tim_rel_pct']:.4f}%)\n"
        f"      realized vol:   hybrid={cell['hybrid_vol']*100:.2f}% plain={cell['plain_vol']*100:.2f}% "
        f"(rel diff {cell['vol_rel_pct']:.4f}%)"
    )


if __name__ == "__main__":
    out = run_conservative()
    print(f"BTC: {out['btc_label']}   ETH: {out['eth_label']}")
    print(f"plain (all-futures) BTC: final_balance=${out['plain'].final_balance:,.2f} "
          f"fees=${out['plain'].fees_paid:,.2f} funding=${out['plain'].funding_paid:,.2f} "
          f"liquidated={out['plain'].liquidated}\n")

    print(f"=== sweep: {out['n_configs']} configs "
          f"({len(BASE_CANDIDATES)} BASE candidates x {len(FEE_TIERS)} fee tiers), "
          f"BTC inner-validation ONLY (2021-01-01 -> 2022-12-31) ===")
    for tier, _ in FEE_TIERS:
        for base in BASE_CANDIDATES:
            print(_fmt_cell(tier, out["grid"][tier][base]))
        print()

    print(f"=== BASE selection (inner-validation only; inner-train is calibration-only, "
          f"never read for selection) ===")
    print(f"  selected BASE = {out['selected_base']:.2f}")

    print("\n=== ETH mechanism check (spot_frac + fut_frac == target), selected BASE ===")
    for tier, checks in out["eth_checks"].items():
        print(f"  [{tier}] max|spot_frac+fut_frac - target|={checks['route_err']:.2e}  "
              f"liquidated={checks['liquidated']} all_finite={checks['all_finite']} "
              f"final_balance=${checks['final_balance']:,.2f} n_bars={checks['n_bars']}")

    sel = out["selected_base"]

    print(f"\n=== FROZEN FALSIFICATION TEST (Step 3, conservative section), BASE={sel:.2f} ===")
    print('  "at the BEST-selected BASE on inner-validation, if d_sharpe (hybrid vs. plain')
    print('  futures v4, 95% paired-bootstrap CI, both fee tiers) does not exclude zero on')
    print('  the favourable side, this branch is NEGATIVE."')
    falsified = False
    for tier, _ in FEE_TIERS:
        cell = out["grid"][tier][sel]
        excludes_favorable = cell["ci_lo"] > 0.0
        print(f"  [{tier}] d_sharpe={cell['d_sharpe']:+.3f} CI=[{cell['ci_lo']:+.3f}, "
              f"{cell['ci_hi']:+.3f}]  excludes zero on favourable side: {excludes_favorable}")
        if not excludes_favorable:
            falsified = True
    print(f"  => FALSIFICATION TEST {'FAILS (branch is NEGATIVE)' if falsified else 'is cleared'}")

    print(f"\n=== promotion bar (5 clauses, frozen), BASE={sel:.2f} ===")
    clause_results = {}
    for tier, _ in FEE_TIERS:
        cell = out["grid"][tier][sel]
        g1 = cell["d_sharpe"] >= D_SHARPE_FLOOR and cell["ci_lo"] > 0.0
        g2 = cell["turnover_ratio"] < TURNOVER_SAVINGS_KILL
        g3 = (cell["tim_rel_pct"] <= EXPOSURE_MATCH_TOL_PCT
              and cell["vol_rel_pct"] <= EXPOSURE_MATCH_TOL_PCT)
        g5 = not cell["liquidated"]
        clause_results[tier] = dict(g1=g1, g2=g2, g3=g3, g5=g5)
        print(f"  [{tier}] (1) d_sharpe>=+0.20 & CI excl 0 = {g1}  "
              f"(2) turnover_ratio<0.50 = {g2} ({cell['turnover_ratio']:.3f})  "
              f"(3) exposure matched<=1% = {g3}  "
              f"(5) no liquidation = {g5}")
    g4_route = all(c["route_err"] < 1e-9 for c in out["eth_checks"].values())
    g4_liq = all(not c["liquidated"] for c in out["eth_checks"].values())
    g4 = g4_route and g4_liq
    print(f"  [ETH] (4) spot_frac+fut_frac==target & no liquidation/lookahead bug = {g4}")

    clause1_all = all(r["g1"] for r in clause_results.values())
    clause2_all = all(r["g2"] for r in clause_results.values())
    clause3_all = all(r["g3"] for r in clause_results.values())
    clause5_all = all(r["g5"] for r in clause_results.values())
    promoted = clause1_all and clause2_all and clause3_all and g4 and clause5_all
    print(f"\n  clause 1 (d_sharpe, both tiers)    : {clause1_all}")
    print(f"  clause 2 (turnover ratio, both tiers): {clause2_all}")
    print(f"  clause 3 (risk-matched, both tiers) : {clause3_all}")
    print(f"  clause 4 (ETH mechanism check)      : {g4}")
    print(f"  clause 5 (no liquidation)           : {clause5_all}")
    print(f"\n  PROMOTION BAR: {'CLEARED' if promoted else 'NOT CLEARED'}")

    print(f"\n=== VERDICT ===")
    if falsified or not promoted:
        print(f"  NEGATIVE -- falsification test {'failed' if falsified else 'passed'}, "
              f"promotion bar {'not cleared' if not promoted else 'cleared'}.")
    else:
        print(f"  PROMOTE-candidate -- both the falsification test and the full 5-clause "
              f"promotion bar cleared at BASE={sel:.2f} on inner-validation. Per the freeze, "
              f"this now proceeds to a fresh Step 4 holdout consultation, not read here.")
