"""R-63 (conservative branch): the diversified time-series-momentum portfolio
of Moskowitz, Ooi & Pedersen (2012), built literally, with this project's own
incumbent as the per-asset rule.

    w_i(t) = clip(v4_target_i(t), 0, 1) / N

where ``v4_target_i`` is ``kelly_regime_v4``'s own desired exposure fraction
for asset i -- obtained through ``r63_shared.v4_targets``, which is
``KellyRegimeV3.prepare()`` byte-for-byte -- and N is the universe size.
Every asset gets 1/N of capital and runs unmodified v4 inside it; the
portfolio's total notional is the mean of the per-asset exposures, so it
never levers and never shorts.

**This arm has ZERO new parameters.** Nothing is swept, nothing selected, no
constant chosen. The only harness constant is ``PRE_DAYS`` (90), the amount
of pre-window calendar the aligned frame carries so that v4's 80-day anchor
is warm on the first EVALUATED bar; it is a warmup, not a strategy
parameter, it is not swept, and it cannot change what the strategy does on
any bar it is evaluated on (v4's anchors are 80 days; 90 days of pre-roll is
the smallest round number that covers them given the data starts
2020-01-01).

Pre-registration (shared, frozen, NOT edited here): ``experiments/r63_shared.py``.
Every window, universe, cost, decision rule, benchmark and the falsification
test come from that file's module docstring. Nothing here relaxes or invents
a rule.

Mechanism (one sentence): if the incumbent's surviving property is a thin
per-asset trend edge, then averaging N of them at 1/N each should keep the
edge and divide the idiosyncratic variance, which is exactly the MOP-2012
construction -- and named failure mode (F1) in the shared docstring predicts
it will not, because eight crypto majors at 0.7-0.9 correlation carry an
effective breadth near 1-2, not N.

Usage::

    python experiments/r63_conservative_panel_tsmom.py checks
    python experiments/r63_conservative_panel_tsmom.py run
    python experiments/r63_conservative_panel_tsmom.py scramble
    python experiments/r63_conservative_panel_tsmom.py all

Outputs (this branch owns these files and only these):
    reports/r63_panel_portfolio/conservative_cells.csv
    reports/r63_panel_portfolio/conservative_scramble.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import csv  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r63_shared as sh  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402

OUT_DIR = sh.OUT_DIR
CELLS_CSV = OUT_DIR / "conservative_cells.csv"
SCRAMBLE_CSV = OUT_DIR / "conservative_scramble.csv"

# Pre-roll carried into the aligned frame so v4's 80-day anchors are warm on
# the first EVALUATED bar. Warmup, not a parameter: not swept, and it changes
# nothing about the rule applied on any evaluated bar.
PRE_DAYS = 90
V4_WARMUP_BARS = 80 * BARS_PER_DAY + 10  # 23,050

ARM = "conservative_panel_tsmom"


# --------------------------------------------------------------- the candidate


def build_targets(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The whole strategy. ``clip(v4_target_i, 0, 1) / N`` per asset.

    Deliberately asset-by-asset: no quantity computed across the panel enters
    any weight, so there is nothing here that could be a cross-sectional or
    full-series fit. N is ``len(aligned)``, the universe size, a literal
    count -- not a data-derived constant.
    """
    assets = list(aligned.keys())
    n = len(assets)
    cols = {}
    for t in assets:
        cols[t] = sh.v4_targets(aligned[t]).clip(0.0, 1.0) / n
    return pd.DataFrame(cols, index=next(iter(aligned.values())).index)


# ------------------------------------------------------------------ plumbing


def _ts(x):
    return pd.Timestamp(x, tz="UTC")


def build_cell(universe, eval_window):
    """Align over ``eval_window`` extended PRE_DAYS backwards, build the target
    matrix over that longer frame, then slice BOTH the targets and the price
    frames down to the evaluation window.

    Returns (targets_eval, aligned_eval, aligned_ext, targets_ext).
    """
    start, end = eval_window
    ext_start = (_ts(start) - pd.Timedelta(days=PRE_DAYS)).strftime("%Y-%m-%d")
    frames = sh.load_universe(universe)
    aligned_ext = sh.align_frames(frames, (ext_start, end))
    targets_ext = build_targets(aligned_ext)

    idx = targets_ext.index
    lo = _ts(start)
    # `r63_shared._hi` extends `end` by a full day, so an `end`-dated window
    # picks up the next day's 00:00 bar. Slice the EVALUATION index strictly
    # inside the pre-registered window instead. (Not a change to the shared
    # file; the extra bar is only ever loaded, never evaluated.)
    hi = None if end is None else _ts(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = idx >= lo if hi is None else (idx >= lo) & (idx <= hi)
    ev = idx[mask]
    if len(ev) == 0:
        raise RuntimeError(f"empty evaluation window {eval_window!r}")

    targets = targets_ext.loc[ev]
    aligned = {t: df.loc[ev] for t, df in aligned_ext.items()}
    return targets, aligned, aligned_ext, targets_ext


def verify_warm(aligned_ext: dict[str, pd.DataFrame], targets_ext: pd.DataFrame,
                first_eval) -> bool:
    """Every asset must have a WARM, non-NaN target on the first evaluated bar.

    Warm means: v4's three anchors (20/40/80 day rolling means) and its
    realized-vol EWM are all finite at that bar, i.e. the target is a real
    decision and not a warmup zero.
    """
    idx = targets_ext.index
    pos = idx.get_indexer([first_eval])[0]
    print(f"  first evaluated bar {first_eval}  (row {pos} of the aligned frame; "
          f"v4 warmup needs {V4_WARMUP_BARS} bars)")
    ok = pos >= V4_WARMUP_BARS
    if not ok:
        print(f"  FAIL: only {pos} bars of pre-roll, need {V4_WARMUP_BARS}")
    for t, df in aligned_ext.items():
        close = df["close"]
        r = np.log(close).diff()
        anchors = {d: close.rolling(int(d * BARS_PER_DAY)).mean().iloc[pos]
                   for d in (20, 40, 80)}
        vol = (r.ewm(span=8 * BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).iloc[pos]
        tgt = targets_ext[t].iloc[pos]
        good = (all(np.isfinite(v) for v in anchors.values())
                and np.isfinite(vol) and not pd.isna(tgt))
        ok = ok and good
        print(f"    {t:5s} anchors20/40/80 finite={[bool(np.isfinite(v)) for v in anchors.values()]} "
              f"vol={vol:.4f} target={tgt:.6f} -> {'WARM' if good else 'NOT WARM'}")
    return bool(ok)


def btc_series_on(idx: pd.Index) -> dict[str, pd.DataFrame]:
    """BTC spot reindexed onto ``idx`` (forward-filled, the same convention
    `align_frames` uses) so BTC_HOLD can be measured on the U6 grid. BTC's
    committed series ends a few days before the Coinbase panel's, so its last
    real price is carried flat over the tail; stated, not hidden."""
    df = sh.load_universe(("BTC",))["BTC"]
    sub = df.reindex(df.index.union(idx)).ffill().reindex(idx)
    if sub[["open", "high", "low", "close"]].isna().any().any():
        raise RuntimeError("BTC: NaNs after reindex onto the panel grid")
    return {"BTC": sub}


# --------------------------------------------------------------------- rows


def _row(decision, window_name, window, universe, market_name, fee, mtn, cmp_) -> dict:
    return {
        "arm": ARM,
        "decision": decision,
        "window": window_name,
        "window_start": window[0] or "first_bar",
        "window_end": window[1] or "last_bar",
        "universe": "+".join(universe),
        "n_assets": len(universe),
        "market": market_name,
        "fee": fee,
        "benchmark": cmp_.pop("_bench"),
        "cand_final": cmp_["cand_final"],
        "bench_final": cmp_["bench_final"],
        "cand_max_dd_pct": cmp_["cand_dd"],
        "bench_max_dd_pct": cmp_["bench_dd"],
        "mean_total_notional": mtn,
        "growth_diff": cmp_["growth_diff"],
        "growth_lo": cmp_["growth_lo"],
        "growth_hi": cmp_["growth_hi"],
        "dd_diff": cmp_["dd_diff"],
        "dd_lo": cmp_["dd_lo"],
        "dd_hi": cmp_["dd_hi"],
        "n_days": cmp_["n_days"],
        "d1_pass": sh.d1_pass(cmp_),
        "d2_pass": sh.d2_pass(cmp_),
        "d3_pass": sh.d3_pass(cmp_),
    }


def _print_row(r: dict) -> None:
    print(f"  [{r['decision']:14s}] vs {r['benchmark']:12s} "
          f"final {r['cand_final']:10.2f} vs {r['bench_final']:10.2f} | "
          f"maxDD {r['cand_max_dd_pct']:6.2f}% vs {r['bench_max_dd_pct']:6.2f}% | "
          f"growth {r['growth_diff']:+.4f} [{r['growth_lo']:+.4f},{r['growth_hi']:+.4f}] | "
          f"dd {r['dd_diff']:+.3f} [{r['dd_lo']:+.3f},{r['dd_hi']:+.3f}] | "
          f"n={r['n_days']}")


# ------------------------------------------------------------------- checks


def cmd_checks() -> dict:
    print("=" * 100)
    print("SELF-CHECKS -- R-63 conservative (panel TSMOM, 1/N x clip(v4,0,1))")
    print("=" * 100)

    t0 = time.time()
    ok, err = sh.check_against_engine()
    print(f"1. check_against_engine(): ok={ok}  relative final-balance error={err:.6f}"
          f"  ({time.time() - t0:.1f}s)")

    targets, aligned, aligned_ext, targets_ext = build_cell(sh.UNIVERSE_6, sh.W_FULL6)
    print("\n   warmth of the first evaluated bar (D1 cell, U6, W_FULL6):")
    warm = verify_warm(aligned_ext, targets_ext, targets.index[0])
    print(f"   all assets warm and non-NaN on the first evaluated bar: {warm}")

    t0 = time.time()
    causal = sh.check_causality(build_targets, aligned_ext)
    print(f"\n2. check_causality(build_targets, aligned): {causal}  "
          f"({time.time() - t0:.1f}s)")

    print("\n4. own sanity check -- full-series-fit hunt (the failure mode ROUTINE")
    print("   tells skeptics to look for: a scaler/quantile/mean/std computed over")
    print("   the whole series and applied to early rows).")
    tamper = sanity_future_tamper(aligned_ext)
    indep = sanity_asset_independence(aligned_ext)
    print(f"   future-tamper probe (3 cut points, x3 / /3 post-cut): {tamper}")
    print(f"   per-asset independence + literal 1/N denominator:     {indep}")

    return {"engine_ok": ok, "engine_err": err, "warm": warm, "causal": causal,
            "tamper": tamper, "independent": indep}


def sanity_future_tamper(aligned_ext: dict[str, pd.DataFrame]) -> bool:
    """Stronger than truncation: keep the SERIES LENGTH fixed and multiply
    every post-cut price by 3 in one copy and by 1/3 in another. Any statistic
    fitted over the whole series (a mean, std, quantile, min-max scaler)
    differs enormously between the two copies, so if a single pre-cut target
    row differs, an early row was reading a whole-series quantity. Truncation
    alone can miss this when the offending statistic is length-normalized.
    """
    n = len(next(iter(aligned_ext.values())).index)
    all_ok = True
    for frac in (0.40, 0.70, 0.90):
        cut = int(n * frac)
        up, down = {}, {}
        for t, df in aligned_ext.items():
            u, d = df.copy(), df.copy()
            for col in ("open", "high", "low", "close"):
                u.iloc[cut:, u.columns.get_loc(col)] *= 3.0
                d.iloc[cut:, d.columns.get_loc(col)] /= 3.0
            up[t], down[t] = u, d
        a = build_targets(up).iloc[:cut].to_numpy(dtype=float)
        b = build_targets(down).iloc[:cut].to_numpy(dtype=float)
        ok = bool(np.array_equal(np.nan_to_num(a), np.nan_to_num(b)))
        print(f"     cut at {frac:.0%} (bar {cut}): pre-cut targets identical "
              f"under opposite post-cut tampers -> {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def sanity_asset_independence(aligned_ext: dict[str, pd.DataFrame]) -> bool:
    """No panel-wide quantity may enter a weight. Rebuilding the matrix from a
    ONE-asset frame must reproduce that asset's column exactly (up to the 1/N
    denominator), which proves the denominator is the literal universe count
    and nothing cross-sectional -- no rank, no panel mean, no panel vol -- is
    used anywhere."""
    full = build_targets(aligned_ext)
    n = len(aligned_ext)
    ok = True
    for t, df in aligned_ext.items():
        solo = build_targets({t: df})[t]  # denominator 1
        same = np.allclose(full[t].to_numpy(dtype=float),
                           solo.to_numpy(dtype=float) / n, atol=0.0, rtol=0.0)
        ok = ok and bool(same)
        print(f"     {t:5s} column == solo-frame v4 target / {n} exactly: "
              f"{'PASS' if same else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------- run


def cmd_run() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    print("=" * 100)
    print("D1 / D2 / D4 -- W_FULL6, U6")
    print("=" * 100)

    targets, aligned, _, _ = build_cell(sh.UNIVERSE_6, sh.W_FULL6)
    idx = targets.index
    print(f"  evaluated bars: {len(idx)}  {idx[0]} -> {idx[-1]}")

    mtn = sh.mean_total_notional(targets)
    print(f"  candidate mean total notional: {mtn:.4f}")

    cand = sh.simulate_portfolio(targets, aligned, sh.SPOT_BASE)
    matched = sh.simulate_portfolio(
        sh.matched_hold_targets(idx, sh.UNIVERSE_6, mtn), aligned, sh.SPOT_BASE)
    ewhold = sh.static_hold_equity(aligned, sh.UNIVERSE_6, sh.SPOT_BASE)
    btc = btc_series_on(idx)
    btchold = sh.static_hold_equity(btc, ("BTC",), sh.SPOT_BASE)

    c = sh.compare(cand, matched); c["_bench"] = "MATCHED_HOLD"
    r = _row("D1_D2", "W_FULL6", sh.W_FULL6, sh.UNIVERSE_6, "spot", 0.001, mtn, c)
    rows.append(r); _print_row(r)
    d1, d2 = r["d1_pass"], r["d2_pass"]

    c = sh.compare(cand, ewhold); c["_bench"] = "EW_HOLD"
    r2 = _row("context", "W_FULL6", sh.W_FULL6, sh.UNIVERSE_6, "spot", 0.001, mtn, c)
    rows.append(r2); _print_row(r2)

    c = sh.compare(cand, btchold); c["_bench"] = "BTC_HOLD"
    r3 = _row("context", "W_FULL6", sh.W_FULL6, sh.UNIVERSE_6, "spot", 0.001, mtn, c)
    rows.append(r3); _print_row(r3)

    # D4 -- same cell, 0.40% taker, candidate final balance vs EW_HOLD's.
    cand4 = sh.simulate_portfolio(targets, aligned, sh.SPOT_REAL)
    ew4 = sh.static_hold_equity(aligned, sh.UNIVERSE_6, sh.SPOT_REAL)
    c = sh.compare(cand4, ew4); c["_bench"] = "EW_HOLD"
    r4 = _row("D4", "W_FULL6", sh.W_FULL6, sh.UNIVERSE_6, "spot", 0.004, mtn, c)
    rows.append(r4); _print_row(r4)
    d4 = r4["cand_final"] > r4["bench_final"]
    print(f"  D4 (final balance beats EW_HOLD at 0.40%): {'PASS' if d4 else 'FAIL'}"
          f"   [pre-registered prediction: FAILS]")

    print()
    print("=" * 100)
    print("D3 -- W_VAL, U8 (2022 bear)")
    print("=" * 100)
    t8, a8, _, _ = build_cell(sh.UNIVERSE_8, sh.W_VAL)
    print(f"  evaluated bars: {len(t8.index)}  {t8.index[0]} -> {t8.index[-1]}")
    mtn8 = sh.mean_total_notional(t8)
    print(f"  candidate mean total notional: {mtn8:.4f}")
    cand8 = sh.simulate_portfolio(t8, a8, sh.SPOT_BASE)
    matched8 = sh.simulate_portfolio(
        sh.matched_hold_targets(t8.index, sh.UNIVERSE_8, mtn8), a8, sh.SPOT_BASE)
    ew8 = sh.static_hold_equity(a8, sh.UNIVERSE_8, sh.SPOT_BASE)

    c = sh.compare(cand8, matched8); c["_bench"] = "MATCHED_HOLD"
    r5 = _row("D3", "W_VAL", sh.W_VAL, sh.UNIVERSE_8, "spot", 0.001, mtn8, c)
    rows.append(r5); _print_row(r5)
    d3 = r5["d3_pass"]

    c = sh.compare(cand8, ew8); c["_bench"] = "EW_HOLD"
    r6 = _row("context", "W_VAL", sh.W_VAL, sh.UNIVERSE_8, "spot", 0.001, mtn8, c)
    rows.append(r6); _print_row(r6)

    print()
    print(f"  D1 (growth vs MATCHED_HOLD, interval excludes 0): {'PASS' if d1 else 'FAIL'}")
    print(f"  D2 (drawdown vs MATCHED_HOLD, interval excludes 0): {'PASS' if d2 else 'FAIL'}")
    print(f"  D3 (2022 U8, growth up AND drawdown down):          {'PASS' if d3 else 'FAIL'}")
    print(f"  D4 (0.40% taker, final balance vs EW_HOLD):         {'PASS' if d4 else 'FAIL'}")

    return rows, {"d1": d1, "d2": d2, "d3": d3, "d4": d4,
                  "d1_growth_diff": rows[0]["growth_diff"]}


# ----------------------------------------------------------------- scramble


def cmd_scramble() -> tuple[list[dict], dict]:
    print("=" * 100)
    print("FALSIFICATION -- cross-section scramble control, seeds 0..9, D1 cell")
    print("=" * 100)
    targets, aligned, _, _ = build_cell(sh.UNIVERSE_6, sh.W_FULL6)
    idx = targets.index
    mtn = sh.mean_total_notional(targets)

    cand = sh.simulate_portfolio(targets, aligned, sh.SPOT_BASE)
    matched = sh.simulate_portfolio(
        sh.matched_hold_targets(idx, sh.UNIVERSE_6, mtn), aligned, sh.SPOT_BASE)
    real = sh.compare(cand, matched)
    print(f"  candidate D1 growth-diff point estimate: {real['growth_diff']:+.5f}")

    out = []
    for seed in sh.SCRAMBLE_SEEDS:
        st = sh.scramble_targets(targets, seed)
        smtn = sh.mean_total_notional(st)
        seq = sh.simulate_portfolio(st, aligned, sh.SPOT_BASE)
        c = sh.compare(seq, matched)
        out.append({
            "arm": ARM, "seed": seed, "window": "W_FULL6",
            "universe": "+".join(sh.UNIVERSE_6), "market": "spot", "fee": 0.001,
            "mean_total_notional": smtn,
            "notional_matches_candidate": bool(abs(smtn - mtn) < 1e-12),
            "cand_final": c["cand_final"], "bench_final": c["bench_final"],
            "cand_max_dd_pct": c["cand_dd"], "bench_max_dd_pct": c["bench_dd"],
            "growth_diff": c["growth_diff"], "growth_lo": c["growth_lo"],
            "growth_hi": c["growth_hi"], "dd_diff": c["dd_diff"],
            "n_days": c["n_days"],
        })
        print(f"    seed {seed}: growth_diff {c['growth_diff']:+.5f}  "
              f"final {c['cand_final']:10.2f}  maxDD {c['cand_dd']:6.2f}%  "
              f"mean_notional {smtn:.4f} (== candidate: {out[-1]['notional_matches_candidate']})")

    diffs = np.array([o["growth_diff"] for o in out], dtype=float)
    p90 = float(np.percentile(diffs, 90))
    survived = bool(real["growth_diff"] > p90)
    print(f"\n  scrambled growth-diff: min {diffs.min():+.5f}  median "
          f"{np.median(diffs):+.5f}  max {diffs.max():+.5f}  p90 {p90:+.5f}")
    print(f"  candidate {real['growth_diff']:+.5f} > p90 {p90:+.5f} -> "
          f"{'SURVIVES' if survived else 'FALSIFIED'}")
    return out, {"real": real["growth_diff"], "p90": p90, "survived": survived}


# -------------------------------------------------------------------- output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    checks = cells = scram = None
    if cmd in ("checks", "all"):
        checks = cmd_checks()
        print()
    if cmd in ("run", "all"):
        cells, verdict = cmd_run()
        write_csv(CELLS_CSV, cells)
        print()
    if cmd in ("scramble", "all"):
        scram, sver = cmd_scramble()
        write_csv(SCRAMBLE_CSV, scram)
        print()

    if cmd == "all":
        fw = sh.further_work(verdict["d1"], verdict["d2"], verdict["d3"],
                             sver["survived"])
        print("=" * 100)
        print(f"further_work(d1={verdict['d1']}, d2={verdict['d2']}, "
              f"d3={verdict['d3']}, scramble_survived={sver['survived']}) = {fw}")
        if not fw:
            print("-> DONE. The W_HOLD holdout is NOT read. +0 holdout consultations.")
        else:
            print("-> Bar cleared. STOPPING and reporting to the operator; the "
                  "holdout read is the operator's decision, not this branch's.")
        print("=" * 100)
    print(f"config_count() = {sh.config_count()}")


if __name__ == "__main__":
    main()
