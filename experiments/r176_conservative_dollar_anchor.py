"""R-176 CONSERVATIVE branch: training-only sweep of `BASELINE_WINDOW_DAYS`
for the dollar-activity-clock vote-anchor substitution defined in
`experiments/r176_shared.py`, plus the two pre-registered falsification
checks from `experiments/r176_direction.md` Step 1 Q4(a).

This file is read-only with respect to `experiments/r176_shared.py` and
`experiments/r102_shared.py` -- it imports from both, edits neither, and
never touches OOS_START / the holdout (every call below goes through
`run_slice`/`compare`-style helpers that assert this, and the two slices
actually swept are `inner_train`/`inner_val`, both entirely inside the
pre-holdout period).

Because `experiments.r176_shared.compare()` hardcodes a fixed
`DOLLAR_WARMUP_BARS` sized for the DEFAULT `BASELINE_WINDOW_DAYS=180`, it
cannot be reused unmodified for a window sweep that goes up to 365 days
(too little warm-up would leave the dollar-time anchor artificially
cold-started on the wider windows). So this file defines a small local
`local_compare()`, structurally identical to `r176_shared.compare()` /
`r102_shared.compare()` (same row schema, same paired-bootstrap machinery,
same risk-matching fields), parameterized by an explicit per-config warmup
-- the "thin wrapper" the round's instructions anticipate. `r176_shared.py`
itself is never edited.

Configs evaluated by this file: see `CONFIGS_EVALUATED` printed at the
bottom, and the breakdown in `main()`'s own printed summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r176_shared import (  # noqa: E402
    STRESS_EPISODES,
    _latched_vote_from_anchor,
    dollar_time_anchor,
    dollar_time_target,
    dollar_time_vote_frac,
)
from experiments.r102_shared import (  # noqa: E402
    BARS_PER_DAY,
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
    V4_BAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

# ------------------------------------------------------------------------
# Pre-registered sweep grid (r176_direction.md, "Conservative variant"):
# BASELINE_WINDOW_DAYS in {90, 120, 180, 270, 365}, inner-train/inner-val
# only, both markets.
# ------------------------------------------------------------------------
WINDOW_GRID: tuple[int, ...] = (90, 120, 180, 270, 365)
MARKETS: tuple = (SPOT, FUTURES)
SWEEP_SLICES: tuple[str, ...] = ("inner_train", "inner_val")
NOISE_FLOOR_SHARPE = 0.2  # ROUTINE.md's own ±0.2 Sharpe noise floor


def warmup_for(window_days: int) -> int:
    """Same formula as r176_shared.DOLLAR_WARMUP_BARS, but a function of the
    swept window instead of the frozen default -- the dollar-time anchor
    needs `window_days` days of causal baseline history before it clears
    its own cold-start NaN, plus the 80-day longest vote horizon, plus a
    10-day pad (identical structure to r176_shared.py's own constant)."""
    return (window_days + 80 + 10) * BARS_PER_DAY


def make_vote_frac(window_days: int):
    """Local variant of `r176_shared.dollar_time_vote_frac`, parameterized
    by `window_days` (r176_shared.py's `dollar_time_anchor` already accepts
    `window_days`; only the vote-level wrapper needs re-plumbing, reusing
    r176_shared's own `_latched_vote_from_anchor` byte-for-byte so the band/
    latch/average logic stays identical to the frozen module)."""

    def build(df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        votes = [
            _latched_vote_from_anchor(
                close, dollar_time_anchor(df, d, window_days=window_days), V4_BAND
            )
            for d in V4_HORIZONS
        ]
        return sum(votes) / len(votes)

    return build


def make_target(window_days: int):
    """Local variant of `r176_shared.dollar_time_target`, parameterized by
    `window_days`. `scale`/deadband are v4's own, byte-identical."""
    vote_fn = make_vote_frac(window_days)

    def build(df: pd.DataFrame) -> np.ndarray:
        frac = vote_fn(df).to_numpy()
        scale = v4_scale(df)
        return apply_deadband(frac * scale)

    return build


def local_compare(candidate_build, *, label: str, warmup: int, btc: pd.DataFrame,
                   eth: pd.DataFrame | None = None, markets: tuple = MARKETS,
                   include_eth: bool = False, slice_names: tuple[str, ...] = SWEEP_SLICES,
                   seed: int = 0) -> list[dict]:
    """Structurally identical to r176_shared.compare() / r102_shared.compare()
    (same row schema, same paired-bootstrap, same risk-matching fields) but
    with an explicit, per-config warmup and an explicit slice-name list, so
    it stays correct across the whole BASELINE_WINDOW_DAYS grid. Never reads
    a bar at/after OOS_START (assert_no_holdout on btc/eth below, and every
    slice window in SLICES/ETH_SLICE_NAME is already pre-holdout by
    construction, exactly as in r176_shared.compare())."""
    assert_no_holdout(btc, "local_compare(): btc")
    if include_eth:
        assert eth is not None
        assert_no_holdout(eth, "local_compare(): eth")

    cand = TargetStrategy(candidate_build, name=f"r176_cons_{label}", warmup=warmup)
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")

    jobs = [(name, *SLICES[name], btc) for name in slice_names]
    if include_eth:
        jobs.append((ETH_SLICE_NAME, None, None, eth))

    rows = []
    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice(cand, df, start, end, slice_name, market)
            b = run_slice(ctrl, df, start, end, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                         if b.mean_abs_exposure else float("nan"))
            vol_ratio = (a.realized_vol / b.realized_vol
                         if b.realized_vol else float("nan"))
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "window_days": None,  # filled by caller
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


def main() -> None:
    btc = load_btc()
    assert_no_holdout(btc, "main(): btc")

    # ==================================================================
    # Main sweep: WINDOW_GRID x MARKETS x SWEEP_SLICES, BTC only, no ETH.
    # ==================================================================
    all_rows: list[dict] = []
    for window_days in WINDOW_GRID:
        rows = local_compare(
            make_target(window_days),
            label=f"w{window_days}",
            warmup=warmup_for(window_days),
            btc=btc,
            markets=MARKETS,
            include_eth=False,
            slice_names=SWEEP_SLICES,
        )
        for r in rows:
            r["window_days"] = window_days
        all_rows.extend(rows)

    n_main = len(WINDOW_GRID) * len(MARKETS) * len(SWEEP_SLICES)
    assert len(all_rows) == n_main

    print("=" * 100)
    print(f"MAIN SWEEP: BASELINE_WINDOW_DAYS in {WINDOW_GRID}, "
          f"markets={[m.name for m in MARKETS]}, slices={SWEEP_SLICES}")
    print(f"({len(WINDOW_GRID)} window values x {len(MARKETS)} markets x "
          f"{len(SWEEP_SLICES)} slices = {n_main} configurations)")
    print("=" * 100)
    hdr = (f"{'window':>7s} {'slice':12s} {'market':8s} {'dSharpe':>8s} {'dDD':>7s} "
           f"{'expR':>6s} {'volR':>6s} {'RM':>3s} {'dlogG':>8s} {'[boot_lo':>9s},{'boot_hi]':>9s} "
           f"{'excl0':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(f"{r['window_days']:7d} {r['slice']:12s} {r['market']:8s} "
              f"{r['d_sharpe']:+8.3f} {r['d_dd']:+7.2f} "
              f"{r['exposure_ratio']:6.2f} {r['vol_ratio']:6.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['boot_d_loggrowth']:+8.3f} {r['boot_lo']:+9.3f},{r['boot_hi']:+9.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>6s}")

    # ==================================================================
    # Falsification test (i): paired-bootstrap CI on d_log_growth excludes
    # zero on the LOSING side (candidate significantly worse) on BTC
    # inner-validation, on either market, at ANY swept window (the strongest
    # form of the check -- covers the whole grid, not just the default).
    # ==================================================================
    val_rows = [r for r in all_rows if r["slice"] == "inner_val"]
    losing_sig = [r for r in val_rows if r["excludes_zero"] and r["boot_hi"] < 0]
    falsified_i = len(losing_sig) > 0

    print()
    print("=" * 100)
    print("FALSIFICATION TEST (i): BTC inner-validation, CI excludes zero on losing side?")
    print("=" * 100)
    for r in val_rows:
        losing = r["excludes_zero"] and r["boot_hi"] < 0
        print(f"  window={r['window_days']:>3d} market={r['market']:8s} "
              f"d_sharpe={r['d_sharpe']:+.3f} boot_CI=[{r['boot_lo']:+.3f},{r['boot_hi']:+.3f}] "
              f"excludes_zero={r['excludes_zero']} losing_side_significant={losing}")
    print(f"-> TEST (i) VERDICT: {'FALSIFIED' if falsified_i else 'not falsified'} "
          f"({len(losing_sig)}/{len(val_rows)} inner-val cells lose significantly)")

    # ==================================================================
    # Falsification test (ii): R^2 between the dollar-time vote (default
    # BASELINE_WINDOW_DAYS=180, i.e. r176_shared.dollar_time_vote_frac
    # unmodified) and v4's own calendar-time vote, on BTC inner-train.
    # >0.98 => mere relabeling => FALSIFIED.
    # ==================================================================
    btc_train_mask = (btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) & \
                      (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))

    dollar_vote_default = dollar_time_vote_frac(btc).to_numpy()  # default window=180
    calendar_vote = v4_vote_frac(btc).to_numpy()
    r2_default = r_squared(dollar_vote_default[btc_train_mask],
                            calendar_vote[btc_train_mask])
    falsified_ii = np.isfinite(r2_default) and r2_default > 0.98

    print()
    print("=" * 100)
    print("FALSIFICATION TEST (ii): R^2(dollar-time vote, v4 calendar vote), BTC inner-train")
    print("=" * 100)
    print(f"  R^2 (default BASELINE_WINDOW_DAYS=180) = {r2_default:.4f} "
          f"(threshold: >0.98 = relabeling)")
    print(f"-> TEST (ii) VERDICT: {'FALSIFIED' if falsified_ii else 'not falsified'}")

    falsified = falsified_i or falsified_ii
    print()
    print("=" * 100)
    print(f"CONSERVATIVE BRANCH OVERALL FALSIFICATION VERDICT: "
          f"{'FALSIFIED' if falsified else 'NOT FALSIFIED'}  "
          f"(test i: {'FAIL' if falsified_i else 'pass'}, "
          f"test ii: {'FAIL' if falsified_ii else 'pass'})")
    print("=" * 100)

    # ==================================================================
    # Winner selection (only meaningful/reported further if not falsified,
    # but computed regardless for transparency): argmax mean d_sharpe over
    # BTC inner-validation, both markets -- the pre-registered decision
    # rule's own Sharpe criterion, applied on inner-validation only, never
    # cherry-picked by eye.
    # ==================================================================
    per_window_val_sharpe = {}
    per_window_val_loggrowth = {}
    for w in WINDOW_GRID:
        cells = [r for r in val_rows if r["window_days"] == w]
        per_window_val_sharpe[w] = float(np.mean([c["d_sharpe"] for c in cells]))
        per_window_val_loggrowth[w] = float(np.mean([c["d_log_growth"] for c in cells]))

    winner = max(WINDOW_GRID, key=lambda w: per_window_val_sharpe[w])
    print()
    print("=" * 100)
    print("WINNER SELECTION: argmax mean(d_sharpe) over BTC inner-val, SPOT+FUTURES")
    print("=" * 100)
    for w in WINDOW_GRID:
        marker = " <== WINNER" if w == winner else ""
        print(f"  window={w:>3d}  mean_d_sharpe(inner_val)={per_window_val_sharpe[w]:+.3f}  "
              f"mean_d_log_growth(inner_val)={per_window_val_loggrowth[w]:+.3f}{marker}")

    # Plateau-vs-peak check: are the neighbours of the winner close to it?
    idx_w = WINDOW_GRID.index(winner)
    neighbours = [WINDOW_GRID[i] for i in (idx_w - 1, idx_w + 1) if 0 <= i < len(WINDOW_GRID)]
    print(f"  plateau check: winner's neighbours in the grid = {neighbours}, "
          f"their mean_d_sharpe = "
          f"{[round(per_window_val_sharpe[n], 3) for n in neighbours]}")

    # ==================================================================
    # Winning configuration: full inner-train + inner-val (already have it)
    # + ETH-replication check.
    # ==================================================================
    eth = load_eth()
    assert_no_holdout(eth, "main(): eth")
    winner_rows_eth = local_compare(
        make_target(winner), label=f"w{winner}_ETHCHECK",
        warmup=warmup_for(winner), btc=btc, eth=eth,
        markets=MARKETS, include_eth=True,
        slice_names=SWEEP_SLICES,
    )
    for r in winner_rows_eth:
        r["window_days"] = winner
    n_eth_check = len(WINDOW_GRID[:1]) * len(MARKETS) * 3  # 1 window x 2 markets x 3 slices

    print()
    print("=" * 100)
    print(f"WINNING CONFIG (window_days={winner}) FULL REPORT: "
          f"inner_train + inner_val + eth_replication, both markets")
    print(f"({len(MARKETS)} markets x 3 slices = {n_eth_check} configurations)")
    print("=" * 100)
    hdr2 = (f"{'slice':16s} {'market':8s} {'dSharpe':>8s} {'dDD':>7s} "
            f"{'expR':>6s} {'volR':>6s} {'RM':>3s} {'dlogG':>8s} {'[boot_lo':>9s},{'boot_hi]':>9s} "
            f"{'excl0':>6s}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in winner_rows_eth:
        print(f"{r['slice']:16s} {r['market']:8s} "
              f"{r['d_sharpe']:+8.3f} {r['d_dd']:+7.2f} "
              f"{r['exposure_ratio']:6.2f} {r['vol_ratio']:6.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['boot_d_loggrowth']:+8.3f} {r['boot_lo']:+9.3f},{r['boot_hi']:+9.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>6s}")

    eth_rows = [r for r in winner_rows_eth if r["slice"] == ETH_SLICE_NAME]
    for r in eth_rows:
        print(f"  ETH replication ({r['market']}): risk_matched={r['risk_matched']} "
              f"exposure_ratio={r['exposure_ratio']:.3f} vol_ratio={r['vol_ratio']:.3f} "
              f"d_sharpe={r['d_sharpe']:+.3f} excludes_zero={r['excludes_zero']}")

    # ==================================================================
    # Standard promotion-adjacent diagnostics (only informative if not
    # falsified -- reported regardless per the task, holdout untouched).
    # ==================================================================
    print()
    print("=" * 100)
    print("PROMOTION-ADJACENT DIAGNOSTICS (inner-validation, informational only -- "
          "holdout NOT touched)")
    print("=" * 100)
    winner_val_cells = [r for r in val_rows if r["window_days"] == winner]
    for r in winner_val_cells:
        beats_floor = r["d_sharpe"] > NOISE_FLOOR_SHARPE
        dd_improved = r["d_dd"] > 0  # candidate's max_drawdown_pct less negative than control's
        print(f"  market={r['market']:8s} d_sharpe={r['d_sharpe']:+.3f} "
              f"(> +{NOISE_FLOOR_SHARPE} noise floor? {beats_floor}) "
              f"d_dd={r['d_dd']:+.2f} (drawdown improvement? {dd_improved}) "
              f"cand_dd={r['cand_dd']:.2f} ctrl_dd={r['ctrl_dd']:.2f}")
    is_plateau = all(abs(per_window_val_sharpe[n] - per_window_val_sharpe[winner]) < 0.3
                      for n in neighbours) if neighbours else False
    print(f"  plateau (neighbours within 0.3 Sharpe of winner)? {is_plateau}")

    # ==================================================================
    # Configuration count ledger.
    # ==================================================================
    total_configs = n_main + n_eth_check
    print()
    print("=" * 100)
    print(f"CONFIGS_EVALUATED = {n_main} (main sweep: {len(WINDOW_GRID)} window values x "
          f"{len(MARKETS)} markets x {len(SWEEP_SLICES)} slices) "
          f"+ {n_eth_check} (winner full report incl. ETH: 1 window x {len(MARKETS)} markets x 3 slices) "
          f"= {total_configs} TOTAL")
    print("=" * 100)

    return {
        "all_rows": all_rows,
        "winner": winner,
        "falsified_i": falsified_i,
        "falsified_ii": falsified_ii,
        "r2_default": r2_default,
        "winner_rows_eth": winner_rows_eth,
        "total_configs": total_configs,
    }


CONFIGS_EVALUATED = (len(WINDOW_GRID) * len(MARKETS) * len(SWEEP_SLICES)) + \
                    (1 * len(MARKETS) * 3)  # main sweep + winner's full incl.-ETH report


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    """Mirrors r176_shared.py's own `_self_test()` pattern on synthetic
    data: causality of THIS file's parameterized builders (not just the
    frozen defaults r176_shared.py already checks), for at least 2 of the
    swept window values."""
    idx = pd.date_range("2017-01-01", periods=300_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(1760001)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    volume = np.abs(rng.normal(20, 5, len(idx)))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": volume}, index=idx)

    for window_days in (90, 365):  # two of the five swept grid points
        vote_fn = make_vote_frac(window_days)
        target_fn = make_target(window_days)
        assert causal_truncation_probe_series(
            lambda d, wd=window_days: dollar_time_anchor(d, 20, window_days=wd), df), \
            f"dollar_time_anchor causality FAIL at window_days={window_days}"
        assert causal_truncation_probe_series(
            lambda d, f=vote_fn: f(d).to_numpy(), df), \
            f"local vote_frac causality FAIL at window_days={window_days}"
        assert causal_truncation_probe_series(target_fn, df), \
            f"local target causality FAIL at window_days={window_days}"

        frac = vote_fn(df)
        finite = frac[frac.notna()]
        assert len(finite) > 5_000, f"too few finite vote values at window_days={window_days}"
        assert finite.between(0.0, 1.0).all(), f"vote out of [0,1] at window_days={window_days}"

    # Sanity: a wider window_days should (on average, not per-bar) shrink
    # fewer bars less often -- i.e. widening the baseline window should not
    # crash or degenerate the vote to all-0/all-1.
    frac_90 = make_vote_frac(90)(df)
    frac_365 = make_vote_frac(365)(df)
    m = frac_90.notna() & frac_365.notna()
    assert frac_90[m].std() > 0 and frac_365[m].std() > 0, \
        "vote degenerated to a constant at one of the tested window_days"

    print(f"r176_conservative_dollar_anchor self-test OK "
          f"(causality + [0,1] range confirmed at window_days in (90, 365); "
          f"CONFIGS_EVALUATED={CONFIGS_EVALUATED})")


_self_test()


if __name__ == "__main__":
    result = main()
    print()
    print(f"FINAL: CONFIGS_EVALUATED = {CONFIGS_EVALUATED} "
          f"(cross-checked against main() total = {result['total_configs']})")
    assert CONFIGS_EVALUATED == result["total_configs"]
