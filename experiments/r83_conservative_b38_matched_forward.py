"""R-83 conservative branch — B-38: pre-register and measure a risk-matched
forward comparison for B-06, per ``experiments/r83_conservative_shared.py``'s
frozen pre-registration (read that file first; nothing below may add a new
threshold or change one).

Three arms, on spot, both fee tiers, both inner splits:

  Arm A  buy_and_hold                          — the status quo B-06 records
  Arm B  frozen mean-notional (ConstantExposureHold at v4's inner-train
         mean notional, unchanged thereafter)  — the addendum's shortcut,
         made deployable
  Arm C  RollingMatchedHold (90-day causal EWM of v4's own exposure)
         — this round's actual B-38 proposal

Run::

    python experiments/r83_conservative_b38_matched_forward.py

Holdout: **+0**. Every frame comes from
``r83_conservative_shared.load_truncated()``, which truncates at
2022-12-31 and asserts it (imported from ``r78_shared``, not
re-implemented).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.r83_conservative_shared import (  # noqa: E402
    BARS_PER_YEAR,
    FEE_LIVE,
    FEE_TABLE,
    HORIZON_DAYS,
    INCUMBENT,
    N_PATHS,
    RollingMatchedHold,
    TRADING_DAYS,
    W_TRAIN,
    W_VAL,
    classify,
    cs_horizon,
    diff_stats,
    load_truncated,
    paired_diff,
)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

CONFIGS = 0     # real-data backtest runs, counted as they happen


def _run(strategy, df, label, window, fee):
    global CONFIGS
    CONFIGS += 1
    market = MarketSpec.spot(fee_rate=fee)
    return run_period(strategy, df, window[0], window[1], market=market,
                      data_label=label)


# --------------------------------------------------------------- Arm setup

def build_arms(df: pd.DataFrame, label: str) -> tuple[float, ConstantExposureHold,
                                                       RollingMatchedHold]:
    """c_global (v4's mean notional on inner-train ONLY) and the two arms."""
    res_train = _run(get_strategy(INCUMBENT), df, label, W_TRAIN, FEE_LIVE)
    c_global = mean_notional(res_train)
    arm_b = ConstantExposureHold(c=c_global)
    arm_c = RollingMatchedHold(halflife_days=90.0)
    return c_global, arm_b, arm_c


# ------------------------------------------------------------- step 2 & 4

def measure_all(df: pd.DataFrame, label: str, arm_b, arm_c) -> pd.DataFrame:
    """Noise/horizon for every (window, fee, arm) cell."""
    rows = []
    for wname, window in (("inner-train", W_TRAIN), ("inner-val", W_VAL)):
        for fname, fee in (("0.10% (table)", FEE_TABLE), ("0.40% (live)", FEE_LIVE)):
            for aname, arm in (("A: buy_and_hold", "buy_and_hold"),
                              ("B: frozen mean-notional", arm_b),
                              ("C: rolling matched", arm_c)):
                d = paired_diff(df, label, window, fee, INCUMBENT, arm)
                global CONFIGS
                CONFIGS += 2   # the two backtests paired_diff runs internally
                stats = diff_stats(d)
                rows.append({"window": wname, "fee": fname, "arm": aname,
                             **stats, "_diff": d})
    return pd.DataFrame(rows)


def print_noise_table(df_stats: pd.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("STEP 2 — paired daily noise and the fixed-n floor, all cells")
    print("=" * 92)
    for wname in df_stats["window"].unique():
        for fname in df_stats["fee"].unique():
            sub = df_stats[(df_stats.window == wname) & (df_stats.fee == fname)]
            print(f"\n[{wname} | {fname}]")
            base = sub[sub.arm == "A: buy_and_hold"].iloc[0]
            for _, r in sub.iterrows():
                red = (100.0 * (1 - r.sd_per_day / base.sd_per_day)
                      if r.arm != "A: buy_and_hold" else 0.0)
                print(f"  {r.arm:26s} n={r.n:4d}  mean={r.mean_per_day:+.6f}/day "
                      f"({r.ann_pct:+.1%}/yr)  sd={r.sd_per_day:.6f}  "
                      f"sd_reduction={red:+.1f}%  "
                      f"fixed_n={r.fixed_n_years:>10,.1f}y")


# ---------------------------------------------------------- step 3: transfer

def transfer_test(df: pd.DataFrame, label: str, c_global: float, arm_b, arm_c) -> None:
    """R-33's own falsification, replayed inside the inner splits (no
    holdout available to this round): does a construction frozen/fitted on
    inner-train transfer into inner-validation, the sharpest regime change
    short of the holdout itself?
    """
    print("\n" + "=" * 92)
    print("STEP 3 — R-33's falsification: frozen vs. rolling, inner-train -> inner-val")
    print("=" * 92)

    v4_val = _run(get_strategy(INCUMBENT), df, label, W_VAL, FEE_LIVE)
    v4_val_notional = mean_notional(v4_val)
    v4_val_vol = _realized_vol(v4_val.equity)
    v4_train = _run(get_strategy(INCUMBENT), df, label, W_TRAIN, FEE_LIVE)
    v4_train_notional = mean_notional(v4_train)
    v4_train_vol = _realized_vol(v4_train.equity)
    print(f"\nv4 actual mean notional:  inner-train {v4_train_notional:.3f}  "
          f"inner-val {v4_val_notional:.3f}   (regime shift "
          f"{100.0 * (v4_val_notional / v4_train_notional - 1.0):+.1f}%)")
    print(f"v4 actual realized vol:   inner-train {v4_train_vol:.3f}  "
          f"inner-val {v4_val_vol:.3f}")
    print(f"c_global (Arm B, frozen on inner-train): {c_global:.3f}")

    for tag, window, ref_notional, ref_vol, strat in (
        ("Arm B (frozen on train) IN-SAMPLE on train", W_TRAIN,
         v4_train_notional, v4_train_vol, arm_b),
        ("Arm B (frozen on train) OUT-OF-CALIBRATION on val", W_VAL,
         v4_val_notional, v4_val_vol, arm_b),
        ("Arm C (rolling) on train", W_TRAIN, v4_train_notional, v4_train_vol, arm_c),
        ("Arm C (rolling) on val", W_VAL, v4_val_notional, v4_val_vol, arm_c),
    ):
        res = _run(strat, df, label, window, FEE_LIVE)
        achieved_notional = mean_notional(res)
        achieved_vol = _realized_vol(res.equity)
        notional_gap = abs(achieved_notional - ref_notional) / ref_notional
        vol_gap = abs(achieved_vol - ref_vol) / ref_vol if ref_vol > 0 else float("nan")
        print(f"  {tag:46s} achieved notional {achieved_notional:.3f} "
              f"(gap {notional_gap:>6.1%})   achieved vol {achieved_vol:.3f} "
              f"(gap {vol_gap:>6.1%})")

    print(
        "\nReading: Arm B's train->val gap is the R-33 failure mode replayed on\n"
        "the notional axis. Arm C's train->val gap is the thing this round\n"
        "exists to check — if it is materially smaller, 'genuinely rolling'\n"
        "is a real improvement over the addendum's frozen shortcut, not just\n"
        "a better story; if it is not, this round's falsification test (1c.3)\n"
        "has fired and the round's contribution over the addendum is zero."
    )


def _realized_vol(equity: pd.Series) -> float:
    eq = equity.to_numpy(dtype=float)
    if len(eq) < 3:
        return 0.0
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(np.std(rets, ddof=1) * np.sqrt(BARS_PER_YEAR))


# --------------------------------------------------------------- causality

def causality_check(df: pd.DataFrame, arm_c) -> bool:
    """Hand-rolled tamper probe for `RollingMatchedHold` — experiments get
    no CI protection (`tests/test_causality_strict.py` parametrizes over
    the registry only). Same two-opposite-tampers procedure as
    ``experiments/matched_hold.py``'s own ``causality()``.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    print("\n" + "=" * 92)
    print("CAUSALITY — RollingMatchedHold, by hand")
    print("=" * 92)

    sub = df.iloc[-200_000:].copy()
    cut = len(sub) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = sub.copy(), sub.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = RollingMatchedHold(halflife_days=arm_c.halflife_days)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=MarketSpec.spot(), start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    pa = RollingMatchedHold(halflife_days=arm_c.halflife_days).prepare(up.copy())
    pb = RollingMatchedHold(halflife_days=arm_c.halflife_days).prepare(down.copy())
    worst = float(np.nanmax(np.abs(pa["target"].to_numpy()[:cut]
                                   - pb["target"].to_numpy()[:cut])))
    ok = not bad and worst < 1e-12
    print(f"  orders {'match' if not bad else f'DIFFER at {bad}'}   "
          f"max |target column difference| before the cut = {worst:.3e}   "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  tampered from bar {cut:,} of {len(sub):,}: "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------- falsification F1

def f1_null_calibration(decisive_diff: pd.Series) -> bool:
    print("\n" + "=" * 92)
    print("STEP 1c.3 / ROUTINE step 2 falsification — F1 null calibration, Arm C")
    print("=" * 92)
    null = decisive_diff.to_numpy() - decisive_diff.mean()
    from experiments.r83_conservative_shared import bootstrap_paths, _first_exclusions
    paths = bootstrap_paths(null, HORIZON_DAYS, N_PATHS, seed=831)
    firsts, _ = _first_exclusions(paths)
    rate = 100.0 * np.mean(np.isfinite(firsts))
    ok = rate <= 5.0
    print(f"  recentred inner-val/live-fee Arm C diff (true mean 0): CS excluded "
          f"zero on {rate:.2f}% of {N_PATHS} paths over 25y (bar: <= 5.00%) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------- main

def main() -> None:
    df, label = load_truncated()
    print(f"data: {label}, {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    c_global, arm_b, arm_c = build_arms(df, label)
    print(f"\nc_global (v4 mean notional, inner-train only, frozen for Arm B): "
          f"{c_global:.4f}")
    print(f"Arm C: RollingMatchedHold(halflife_days={arm_c.halflife_days:.0f}, "
          f"warmup={arm_c.warmup:,} bars)")

    stats = measure_all(df, label, arm_b, arm_c)
    print_noise_table(stats)

    transfer_test(df, label, c_global, arm_b, arm_c)

    causality_ok = causality_check(df, arm_c)

    # ------------------------------------------------ the decisive horizon
    print("\n" + "=" * 92)
    print("STEP 2/4 — anytime-valid horizon, all three arms, both windows (live fee)")
    print("=" * 92)
    horizon_rows = []
    for wname in ("inner-train", "inner-val"):
        for aname in ("A: buy_and_hold", "B: frozen mean-notional", "C: rolling matched"):
            row = stats[(stats.window == wname) & (stats.fee == "0.40% (live)")
                        & (stats.arm == aname)].iloc[0]
            h = cs_horizon(row["_diff"])
            horizon_rows.append({"window": wname, "arm": aname, **h})
            n50 = h["n50_days"]
            n50_str = f"{n50:,.0f}d ({n50 / 365:.1f}y)" if np.isfinite(n50) else "never (25y+)"
            print(f"  [{wname:11s}] {aname:26s} n50={n50_str:>18s}  "
                  f"fired={h['fired_pct']:5.1f}%  by_5y={h['by_5y']:5.1f}%  "
                  f"by_25y={h['by_25y']:5.1f}%  for={h['for_pct']:5.1f}% "
                  f"against={h['against_pct']:5.1f}%")

    hdf = pd.DataFrame(horizon_rows)

    # ---------------------------------------------------------- F1 on Arm C
    decisive = stats[(stats.window == "inner-val") & (stats.fee == "0.40% (live)")
                     & (stats.arm == "C: rolling matched")].iloc[0]["_diff"]
    f1_ok = f1_null_calibration(decisive)

    # ------------------------------------------------------- classification
    print("\n" + "=" * 92)
    print("PRE-REGISTERED CLASSIFICATION (decided on Arm C, inner-val, 0.40% live tier)")
    print("=" * 92)
    decisive_row = hdf[(hdf.window == "inner-val") & (hdf.arm == "C: rolling matched")].iloc[0]
    verdict = classify(decisive_row["n50_days"], decisive_row["by_5y"],
                       decisive_row["by_25y"])
    n50 = decisive_row["n50_days"]
    n50_str = f"{n50:,.0f} days ({n50/365:.1f}y)" if np.isfinite(n50) else "never within 25y"
    print(f"n50 = {n50_str}, 5-year firing rate = {decisive_row['by_5y']:.1f}%, "
          f"25-year firing rate = {decisive_row['by_25y']:.1f}%  ->  {verdict}")
    if not (f1_ok and causality_ok):
        print("NOTE: a machinery check FAILED (F1 and/or causality) - the "
              "horizon above is not to be believed until that is resolved.")

    print(f"\nconfigs evaluated (real-data backtest runs): {CONFIGS}")
    print(f"bootstrap paths simulated (not real-data configs): "
          f"{N_PATHS * (len(horizon_rows) + 1):,}")

    print("\n" + "=" * 92)
    print("SUMMARY — noise reduction and horizon, before vs after matching")
    print("=" * 92)
    for wname in ("inner-train", "inner-val"):
        row_a = stats[(stats.window == wname) & (stats.fee == "0.40% (live)")
                     & (stats.arm == "A: buy_and_hold")].iloc[0]
        row_c = stats[(stats.window == wname) & (stats.fee == "0.40% (live)")
                     & (stats.arm == "C: rolling matched")].iloc[0]
        h_a = hdf[(hdf.window == wname) & (hdf.arm == "A: buy_and_hold")].iloc[0]
        h_c = hdf[(hdf.window == wname) & (hdf.arm == "C: rolling matched")].iloc[0]
        red = 100.0 * (1 - row_c.sd_per_day / row_a.sd_per_day)
        n50a = h_a["n50_days"]
        n50c = h_c["n50_days"]
        n50a_s = f"{n50a/365:.1f}y" if np.isfinite(n50a) else "never(25y+)"
        n50c_s = f"{n50c/365:.1f}y" if np.isfinite(n50c) else "never(25y+)"
        print(f"  [{wname}] sd/day {row_a.sd_per_day:.4f} -> {row_c.sd_per_day:.4f} "
              f"({red:+.1f}%)   n50 {n50a_s} -> {n50c_s}")


if __name__ == "__main__":
    main()
