"""R-132 INDEPENDENT SKEPTIC re-run of `r132_novel_diversified_panel.py`'s
`NovelDiversifiedPanel` claim (round R-132, audited 08-25).

Does NOT modify r132_shared.py, r132_conservative_mvrv_expert.py, or
r132_novel_diversified_panel.py. Everything here is read-only re-derivation
plus one new object: a turnover-matched, information-free control, built the
same way R-130's own skeptic built theirs (deleting the reward channel there;
here, an unmodified 10-expert HedgeExperts with `hysteresis` raised until its
own inner-validation trade count lands near NovelDiversifiedPanel's reduced
trade count) to test whether NovelDiversifiedPanel's apparent B1 gain is
simply "any turnover reduction inside hedge_experts's own bad 2021-2022
losing regime mechanically looks like an improvement" (R-130's own finding,
a Gârleanu-Pedersen-style smooth-trading-rate artifact this project already
closed once, R-64) rather than genuine information from the two new experts.

No bar at or after OOS_START is read (reuses r132_shared's own guarded loaders).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from tradebot.strategies.hedge_experts import HedgeExperts  # noqa: E402

from r132_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    SPOT,
    load_btc_train,
    log_growth_diff,
    run_baseline,
    run_strategy,
    sharpe_diff,
)
from r132_novel_diversified_panel import (  # noqa: E402
    HorizonsOnlyPanel,
    NovelDiversifiedPanel,
    load_eth_train,
)


def _report_diff(name, res_cand, res_base):
    sh = sharpe_diff(res_cand, res_base)
    lg = log_growth_diff(res_cand, res_base)
    print(f"  [{name}] d_sharpe={sh.diff.point:+.4f} CI=[{sh.diff.lo:+.4f},{sh.diff.hi:+.4f}] "
          f"p_pos={sh.p_positive:.3f} sig={sh.significant} | "
          f"d_logret={lg.diff.point:+.4f} sig={lg.significant}")
    return sh, lg


if __name__ == "__main__":
    df_btc, label_btc = load_btc_train("spot")

    print("=" * 78)
    print("STEP 1: reproduce NovelDiversifiedPanel + HorizonsOnlyPanel trade counts,")
    print("        BTC, both markets, inner-val -- to find a hysteresis that matches")
    print("        turnover for the control.")
    print("=" * 78)
    targets = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        m_base, res_base = run_baseline(df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        m_novel, res_novel = run_strategy(NovelDiversifiedPanel(), df_btc, mkt,
                                           INNER_VAL_START, INNER_VAL_END, label_btc)
        m_horiz, res_horiz = run_strategy(HorizonsOnlyPanel(), df_btc, mkt,
                                           INNER_VAL_START, INNER_VAL_END, label_btc)
        targets[mkt_name] = m_novel.num_trades
        print(f"  [{mkt_name}] baseline trades={m_base.num_trades} sharpe={m_base.sharpe:.3f} | "
              f"novel trades={m_novel.num_trades} sharpe={m_novel.sharpe:.3f} | "
              f"horizons_only trades={m_horiz.num_trades} sharpe={m_horiz.sharpe:.3f}")

    print()
    print("=" * 78)
    print("STEP 2: search HedgeExperts(hysteresis=h) [unmodified 10-expert panel,")
    print("        ZERO new information] for h whose own inner-val trade count")
    print("        roughly matches NovelDiversifiedPanel's, per market.")
    print("=" * 78)
    hyst_grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_hyst = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        target = targets[mkt_name]
        rows = []
        for h in hyst_grid:
            m, _ = run_strategy(HedgeExperts(hysteresis=h), df_btc, mkt,
                                 INNER_VAL_START, INNER_VAL_END, label_btc)
            rows.append((h, m.num_trades, m.sharpe))
            print(f"    hysteresis={h:.2f} trades={m.num_trades} sharpe={m.sharpe:.3f} "
                  f"(target trades={target})")
        best_h, best_trades, _ = min(rows, key=lambda t: abs(t[1] - target))
        best_hyst[mkt_name] = best_h
        print(f"  [{mkt_name}] closest match: hysteresis={best_h:.2f} trades={best_trades} "
              f"(target {target})")

    print()
    print("=" * 78)
    print("STEP 3 (DECISIVE): turnover-matched, information-free control vs TRUE")
    print("        baseline hedge_experts -- same B1 cells (BTC, full-inner AND")
    print("        inner-val, spot AND futures_5x).")
    print("=" * 78)
    periods = [
        ("full_inner", INNER_TRAIN_START, INNER_VAL_END),
        ("inner_val", INNER_VAL_START, INNER_VAL_END),
    ]
    control_results = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        h = best_hyst[mkt_name]
        for per_name, start, end in periods:
            m_base, res_base = run_baseline(df_btc, mkt, start, end, label_btc)
            m_ctrl, res_ctrl = run_strategy(HedgeExperts(hysteresis=h), df_btc, mkt,
                                             start, end, label_btc)
            print(f"  [control(h={h:.2f}) {mkt_name}/{per_name}] trades={m_ctrl.num_trades} "
                  f"(baseline trades={m_base.num_trades}) sharpe={m_ctrl.sharpe:.3f} "
                  f"(baseline sharpe={m_base.sharpe:.3f})")
            sh, lg = _report_diff(f"control_{mkt_name}_{per_name}", res_ctrl, res_base)
            control_results[(mkt_name, per_name)] = sh.diff.point

    print()
    print("=" * 78)
    print("STEP 4: re-derive NovelDiversifiedPanel's own B1 numbers for direct")
    print("        side-by-side against the control (same cells).")
    print("=" * 78)
    novel_results = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in periods:
            m_base, res_base = run_baseline(df_btc, mkt, start, end, label_btc)
            m_novel, res_novel = run_strategy(NovelDiversifiedPanel(), df_btc, mkt,
                                               start, end, label_btc)
            print(f"  [novel {mkt_name}/{per_name}] trades={m_novel.num_trades} "
                  f"sharpe={m_novel.sharpe:.3f} (baseline trades={m_base.num_trades} "
                  f"sharpe={m_base.sharpe:.3f})")
            sh, lg = _report_diff(f"novel_{mkt_name}_{per_name}", res_novel, res_base)
            novel_results[(mkt_name, per_name)] = sh.diff.point

    print()
    print("=" * 78)
    print("SIDE-BY-SIDE: turnover-matched information-free control vs full novel")
    print("=" * 78)
    for mkt_name in ("spot", "futures_5x"):
        for per_name, _, _ in periods:
            c = control_results[(mkt_name, per_name)]
            n = novel_results[(mkt_name, per_name)]
            print(f"  {mkt_name}/{per_name}: control d_sharpe={c:+.4f}  "
                  f"novel d_sharpe={n:+.4f}  "
                  f"ratio(control/novel)={c / n if abs(n) > 1e-9 else float('nan'):+.2f}")

    print()
    print("=" * 78)
    print("STEP 5: independent re-derivation of HorizonsOnlyPanel's own B1 numbers")
    print("        (cheap ablation -- new-experts-zeroed panel), same cells, to")
    print("        check whether most of the reported gain is from the 2 new")
    print("        experts or from the horizon collapse alone.")
    print("=" * 78)
    horiz_results = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in periods:
            m_base, res_base = run_baseline(df_btc, mkt, start, end, label_btc)
            m_h, res_h = run_strategy(HorizonsOnlyPanel(), df_btc, mkt, start, end, label_btc)
            print(f"  [horizons_only {mkt_name}/{per_name}] trades={m_h.num_trades} "
                  f"sharpe={m_h.sharpe:.3f} (baseline trades={m_base.num_trades} "
                  f"sharpe={m_base.sharpe:.3f})")
            sh, lg = _report_diff(f"horizons_only_{mkt_name}_{per_name}", res_h, res_base)
            horiz_results[(mkt_name, per_name)] = sh.diff.point

    print()
    print("  Contribution split (novel_d_sharpe - horizons_only_d_sharpe ~= what the")
    print("  two new experts added on top of the horizon collapse alone):")
    for mkt_name in ("spot", "futures_5x"):
        for per_name, _, _ in periods:
            n = novel_results[(mkt_name, per_name)]
            ho = horiz_results[(mkt_name, per_name)]
            print(f"    {mkt_name}/{per_name}: horizons_only={ho:+.4f}  full_novel={n:+.4f}  "
                  f"delta_from_new_experts={n - ho:+.4f}")

    print()
    print("=" * 78)
    print("STEP 6: independent reproduction of the two primary decisive cells --")
    print("        B1 spot/full-inner (reported +0.138, CI [+0.026,+0.264]) and")
    print("        B4 ETH spot/inner-val (reported +0.195).")
    print("=" * 78)
    m_base_si, res_base_si = run_baseline(df_btc, SPOT, INNER_TRAIN_START, INNER_VAL_END, label_btc)
    m_novel_si, res_novel_si = run_strategy(NovelDiversifiedPanel(), df_btc, SPOT,
                                             INNER_TRAIN_START, INNER_VAL_END, label_btc)
    sh_si, lg_si = _report_diff("REPRO_B1_spot_full_inner", res_novel_si, res_base_si)

    df_eth = load_eth_train()
    m_base_eth, res_base_eth = run_baseline(df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    strat_eth = NovelDiversifiedPanel(asset="ETH")
    m_cand_eth, res_cand_eth = run_strategy(strat_eth, df_eth, SPOT,
                                             INNER_VAL_START, INNER_VAL_END, "eth_spot")
    sh_eth, lg_eth = _report_diff("REPRO_B4_eth_spot_inner_val", res_cand_eth, res_base_eth)

    print()
    print("DONE.")
