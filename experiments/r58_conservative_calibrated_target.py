"""R-58 conservative branch: per-asset calibration of `kelly_regime_v4`'s
`target_vol` knob (backlog B-25). Pre-registration: `experiments/r58_shared.py`
— read it first; this file implements its "conservative branch" mechanism
and nothing else.

Mechanism (the ONLY thing this branch changes): `target_vol` already exists
as a constructor parameter on `KellyRegime`/`KellyRegimeV3`/`KellyRegimeV4`
(v4 inherits it unchanged). For each of the 8 assets in scope (BTC, ETH, and
the R-57 panel's six), solve a per-asset `target_vol_i` — via the project's
standard proportional-iteration solver (R-33/R-57's `solve_c` pattern,
applied here to `target_vol` instead of a passive hold's constant exposure)
— so that `kelly_regime_v4(target_vol=target_vol_i)`'s OWN mean clipped
notional (`experiments.matched_hold.mean_notional`) matches a single common
reference: v4's UNMODIFIED (`target_vol=0.55`) mean notional on BTC over the
CONTROL window. `max_leverage=2.0` is never touched, for any asset. The
solve happens ONLY on PANEL_TRAIN/CONTROL (2020-04-01..2022-12-31, the same
date range under two names); PANEL_TEST is read only after every
`target_vol_i` is frozen, and never refit.

No BTC or ETH bar past 2022-12-31 is ever loaded or backtested by this
module — both loaders are truncated immediately after reading, before any
other line of code touches the resulting frame.

Usage::

    uv run python experiments/r58_conservative_calibrated_target.py solve
    uv run python experiments/r58_conservative_calibrated_target.py causality
    uv run python experiments/r58_conservative_calibrated_target.py run     # everything, writes report + CSVs
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.r57_cross_asset_panel import (  # noqa: E402
    BOOT_KW,
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    binomial_tail,
)
from experiments.r58_shared import (  # noqa: E402
    CONTROL,
    D2_REGRESSION_TOLERANCE_PP,
    PANEL_TEST,
    PANEL_TRAIN,
    R57_CONTROL_DD_ADVANTAGE,
    d1_verdict,
    d2_passes,
    load_panel,
    promoted,
)
from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_coinbase_spot, load_dataset  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "r58_conservative"
REPORT_PATH = ROOT / "experiments" / "reports" / "r58_conservative_report.md"

MAX_LEVERAGE = 2.0            # unchanged for every asset, always
DEFAULT_TARGET_VOL = 0.55     # v4's shipped default, the calibration seed
BOOT = dict(**BOOT_KW)

CONFIG_COUNT = 0


# ------------------------------------------------------------------ helpers


def measure(strategy, df, start, end, market):
    """One backtest. Every call is counted — there is no free evaluation."""
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


def load_control_assets() -> tuple[Asset, Asset]:
    """BTC and ETH, truncated at CONTROL's end BEFORE any other line touches
    them, so no 2023+ bar of either is ever loaded into a variable this
    module can backtest (the round's holdout guarantee, enforced in code,
    not just by which window string is passed to run_period)."""
    end_ts = pd.Timestamp(CONTROL[1], tz="UTC")
    btc_df, _label = load_dataset(DATA_DIR, "spot")
    btc_df = btc_df[btc_df.index <= end_ts]
    eth_df = load_coinbase_spot(DATA_DIR, "ETH")
    eth_df = eth_df[eth_df.index <= end_ts]
    btc = Asset("BTC", btc_df, coverage=1.0, max_gap=pd.Timedelta(0), qualifies=True)
    eth = Asset("ETH", eth_df, coverage=1.0, max_gap=pd.Timedelta(0), qualifies=True)
    return btc, eth


def solve_target_vol(target_notional: float, df: pd.DataFrame, start, end, market,
                      tol: float = 0.02, max_iter: int = 8, tv_cap: float = 5.0):
    """Proportional-iteration solve (R-33/R-57's `solve_c` pattern) for the
    `target_vol` at which `kelly_regime_v4`'s OWN mean clipped notional
    matches `target_notional` on this asset/window. `max_leverage` is fixed
    at MAX_LEVERAGE throughout — only `target_vol` moves."""
    tv = DEFAULT_TARGET_VOL
    res, _ = measure(KellyRegimeV4(target_vol=tv, max_leverage=MAX_LEVERAGE), df, start, end, market)
    achieved = mean_notional(res)
    for _ in range(max_iter):
        if not np.isfinite(achieved) or achieved <= 0:
            return float("nan"), achieved
        if abs(achieved - target_notional) <= tol * target_notional:
            return tv, achieved
        tv = float(np.clip(tv * (target_notional / achieved), 1e-3, tv_cap))
        res, _ = measure(KellyRegimeV4(target_vol=tv, max_leverage=MAX_LEVERAGE), df, start, end, market)
        achieved = mean_notional(res)
        if tv >= tv_cap and achieved < target_notional:
            return tv, achieved  # cap binds, no match exists
    return tv, achieved


def cell(a: Asset, strategy, window, market, label: str, rows: list) -> dict:
    """One asset x window x market cell: candidate, buy_and_hold, matched
    hold, paired-bootstrap intervals. Identical structure to R-57's `cell()`,
    parameterized on a strategy INSTANCE (the calibrated candidate) instead
    of a registry lookup."""
    start, end = window
    cand_res, cand = measure(strategy, a.df, start, end, market)
    hold_res, hold = measure(get_strategy("buy_and_hold"), a.df, start, end, market)

    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), a.df, start, end, market)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    hold_ret = daily_returns(hold_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret), len(hold_ret))
    dd_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n], max_drawdown_from_returns, **BOOT)
    growth_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n], total_log_return, **BOOT)

    row = {
        "asset": a.ticker, "window": label, "market": market.name,
        "fee": market.fee_rate,
        "cand_final": cand.final_balance, "cand_dd": cand.max_drawdown_pct,
        "cand_sharpe": cand.sharpe, "cand_trades": cand.num_trades,
        "cand_liq": cand.liquidated,
        "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
        "hold_sharpe": hold.sharpe, "hold_liq": hold.liquidated,
        "c_mean_notional": c_mean,
        "mh_final": mh.final_balance, "mh_dd": mh.max_drawdown_pct,
        "mh_sharpe": mh.sharpe,
        "dd_matched_diff": dd_matched.diff.point,
        "dd_matched_lo": dd_matched.diff.lo, "dd_matched_hi": dd_matched.diff.hi,
        "growth_matched_diff": growth_matched.diff.point,
        "growth_matched_lo": growth_matched.diff.lo,
        "growth_matched_hi": growth_matched.diff.hi,
    }
    rows.append(row)
    print(f"  {a.ticker:5s} {label:9s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} "
          f"DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]")
    return row


# --------------------------------------------------------------- calibration


def cmd_solve(all_assets: list[Asset]) -> tuple[dict[str, float], list[dict]]:
    print("=" * 100)
    print("CALIBRATION — solve target_vol_i per asset so v4's own mean notional "
          "matches BTC's default (target_vol=0.55) mean notional, PANEL_TRAIN/CONTROL")
    print("=" * 100)
    btc = next(a for a in all_assets if a.ticker == "BTC")
    ref_res, _ = measure(KellyRegimeV4(target_vol=DEFAULT_TARGET_VOL, max_leverage=MAX_LEVERAGE),
                         btc.df, *CONTROL, SPOT_BASE)
    reference_notional = mean_notional(ref_res)
    print(f"  reference (BTC, target_vol={DEFAULT_TARGET_VOL}, unmodified): "
          f"mean notional = {reference_notional:.4f}")

    target_vols: dict[str, float] = {}
    rows = []
    for a in all_assets:
        window = CONTROL if a.ticker in ("BTC", "ETH") else PANEL_TRAIN
        tv, achieved = solve_target_vol(reference_notional, a.df, *window, SPOT_BASE)
        resid = abs(achieved - reference_notional) / reference_notional if reference_notional else float("nan")
        target_vols[a.ticker] = tv
        rows.append({"asset": a.ticker, "reference_notional": reference_notional,
                     "target_vol": tv, "achieved_notional": achieved, "resid": resid,
                     "valid": resid <= 0.02})
        print(f"  {a.ticker:5s} target_vol={tv:6.3f}  achieved notional={achieved:.4f}  "
              f"resid={resid:6.2%}  {'OK' if resid <= 0.02 else 'CAP BOUND'}")
    return target_vols, rows


# --------------------------------------------------------------- causality


def cmd_causality(target_vols: dict[str, float], probe_assets: list[Asset]) -> bool:
    """R-57's `cmd_causality` tamper probe, adapted: constructs
    `KellyRegimeV4(target_vol=target_vol_i)` directly (this branch's
    candidate never goes through the registry, so `get_strategy` cannot be
    reused unmodified). Must PASS on every asset probed or the calibrated
    candidate has a lookahead bug and no further result is reported."""
    print("=" * 100)
    print("CAUSALITY TAMPER PROBE — calibrated kelly_regime_v4(target_vol=target_vol_i)")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)
    all_ok = True
    for a in probe_assets:
        tv = target_vols[a.ticker]
        tail = a.df.iloc[-60_000:].copy()
        cut = len(tail) - 5_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]
        up, down = tail.copy(), tail.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame):
            s = KellyRegimeV4(target_vol=tv, max_leverage=MAX_LEVERAGE)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=market, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        ok = all(x == y for x, y in zip(decisions(up), decisions(down)))
        all_ok = all_ok and ok
        print(f"  {a.ticker:5s} (target_vol={tv:.3f}) decisions identical under "
              f"opposite post-cut tampers: {'PASS' if ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------- run


def cmd_run() -> None:
    panel = load_panel()
    btc, eth = load_control_assets()
    all_assets = [btc, eth] + panel

    print()
    target_vols, calib_rows = cmd_solve(all_assets)

    print()
    causality_ok = cmd_causality(target_vols, [btc] + panel[:3])
    if not causality_ok:
        raise SystemExit("CAUSALITY PROBE FAILED — refusing to report D1-D4 "
                         "results until the lookahead bug is fixed.")

    print("\n" + "=" * 100)
    print("D1 (PRIMARY) — PANEL_TRAIN, spot @0.10%, calibrated target_vol_i per asset")
    print("=" * 100)
    d1_rows: list[dict] = []
    for a in panel:
        strat = KellyRegimeV4(target_vol=target_vols[a.ticker], max_leverage=MAX_LEVERAGE)
        cell(a, strat, PANEL_TRAIN, SPOT_BASE, "PANEL_TRAIN", d1_rows)

    print("\n" + "=" * 100)
    print("D2 (FALSIFICATION CONTROL) — CONTROL window, BTC and ETH, calibrated target_vol")
    print("=" * 100)
    d2_rows: list[dict] = []
    for a in (btc, eth):
        strat = KellyRegimeV4(target_vol=target_vols[a.ticker], max_leverage=MAX_LEVERAGE)
        cell(a, strat, CONTROL, SPOT_BASE, "CONTROL", d2_rows)

    print("\n" + "=" * 100)
    print("D3 (GENERALIZATION, reported not gating) — PANEL_TEST, spot @0.10%, frozen target_vol_i")
    print("=" * 100)
    d3_rows: list[dict] = []
    for a in panel:
        strat = KellyRegimeV4(target_vol=target_vols[a.ticker], max_leverage=MAX_LEVERAGE)
        cell(a, strat, PANEL_TEST, SPOT_BASE, "PANEL_TEST", d3_rows)

    print("\n" + "=" * 100)
    print("D4 (0.40% FEE FALSIFICATION) — PANEL_TRAIN, spot @0.40%, frozen target_vol_i")
    print("=" * 100)
    d4_rows: list[dict] = []
    for a in panel:
        strat = KellyRegimeV4(target_vol=target_vols[a.ticker], max_leverage=MAX_LEVERAGE)
        cell(a, strat, PANEL_TRAIN, SPOT_REAL, "PANEL_TRAIN", d4_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(OUT_DIR / "calibration.csv", index=False)
    pd.DataFrame(d1_rows).to_csv(OUT_DIR / "d1_panel_train.csv", index=False)
    pd.DataFrame(d2_rows).to_csv(OUT_DIR / "d2_control.csv", index=False)
    pd.DataFrame(d3_rows).to_csv(OUT_DIR / "d3_panel_test.csv", index=False)
    pd.DataFrame(d4_rows).to_csv(OUT_DIR / "d4_fee_falsification.csv", index=False)

    verdicts(target_vols, d1_rows, d2_rows, d3_rows, d4_rows, len(panel))
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")
    print("Holdout consultations added by this round: 0 "
          "(BTC/ETH truncated at 2022-12-31 before any other line touches them; "
          "panel-asset reads cost +0 per the pre-registration)")


def verdicts(target_vols: dict[str, float], d1_rows: list[dict], d2_rows: list[dict],
             d3_rows: list[dict], d4_rows: list[dict], n: int) -> None:
    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULES (experiments/r58_shared.py)")
    print("=" * 100)

    d1 = pd.DataFrame(d1_rows)
    k1 = int((d1.cand_dd < d1.mh_dd).sum())
    excl = int(((d1.dd_matched_lo > 0) | (d1.dd_matched_hi < 0)).sum())
    better_excl = int((d1.dd_matched_hi < 0).sum())
    p1 = binomial_tail(k1, n)
    print(f"D1 (primary, matched-exposure drawdown, PANEL_TRAIN, spot @0.10%): "
          f"{k1}/{n} -> {d1_verdict(k1, n)} (exact binomial p={p1:.4f})")
    print(f"    paired bootstrap: {excl}/{n} intervals exclude zero "
          f"({better_excl}/{n} of them in the candidate's favour)")

    d2 = pd.DataFrame(d2_rows).set_index("asset")
    dd_advantage = {t: float(d2.loc[t, "dd_matched_diff"]) for t in ("BTC", "ETH")}
    print(f"D2 (falsification control, CONTROL window): "
          f"BTC {dd_advantage['BTC']:+.1f}pp (R-57: {R57_CONTROL_DD_ADVANTAGE['BTC']:+.1f}pp), "
          f"ETH {dd_advantage['ETH']:+.1f}pp (R-57: {R57_CONTROL_DD_ADVANTAGE['ETH']:+.1f}pp), "
          f"tolerance {D2_REGRESSION_TOLERANCE_PP:+.1f}pp -> "
          f"{'PASSES' if d2_passes(dd_advantage) else 'FAILS'}")

    d3 = pd.DataFrame(d3_rows)
    k3 = int((d3.cand_dd < d3.mh_dd).sum())
    print(f"D3 (generalization, PANEL_TEST, descriptive): {k3}/{n} -> {d1_verdict(k3, n)}")

    d4 = pd.DataFrame(d4_rows)
    k4 = int((d4.cand_final > d4.hold_final).sum())
    print(f"D4 (0.40% fee falsification, beats buy_and_hold final balance): "
          f"{k4}/{n} -> {'SURVIVES' if k4 >= n - 1 else 'FAILS (as predicted)'}")

    verdict = "PROMOTE-CANDIDATE" if promoted(k1, dd_advantage, n) else "NEGATIVE"
    print(f"\nOVERALL (promoted(k1, dd_advantage) mechanically applied): {verdict}")
    print(f"target_vol_i: " + ", ".join(f"{k}={v:.3f}" for k, v in target_vols.items()))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "solve":
        panel = load_panel()
        btc, eth = load_control_assets()
        cmd_solve([btc, eth] + panel)
        print(f"\nConfigurations evaluated: {CONFIG_COUNT}")
        return
    if cmd == "causality":
        panel = load_panel()
        btc, eth = load_control_assets()
        all_assets = [btc, eth] + panel
        target_vols, _ = cmd_solve(all_assets)
        print()
        ok = cmd_causality(target_vols, [btc] + panel[:3])
        print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
        return
    if cmd == "run":
        cmd_run()
        return
    raise SystemExit(f"unknown command {cmd!r} (solve | causality | run)")


if __name__ == "__main__":
    main()
