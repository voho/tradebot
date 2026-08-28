#!/usr/bin/env python
"""R-175 NOVEL branch: persistence-filtered MSM(6) structural volatility
(lowest `N_PERSIST` of 6 Calvet-Fisher components only) driving
`kelly_regime_v4`'s hysteresis vol-target scale.

Direction, citations, non-duplication argument, and the shared MSM engine
all live in `experiments/r175_shared.py` (read there first -- this file
does not repeat that reasoning and does not edit that module, which is
frozen/read-only and shared with the sibling CONSERVATIVE branch running in
parallel). This file implements ONLY the novel branch's own pre-registered
falsification test and, if it survives, the standard promotion bar.

THE MECHANISM, exactly (from `r175_direction.md`'s "Novel variant"
section): `r175_shared.msm_structural_target` already computes exposure
using ONLY the lowest `N_PERSIST=2` of the 6 fitted MSM components
(deliberately dropping the fast/transient ones) as the numerator of the
existing `ratio = vol/slow` hysteresis state machine
(`conditional_target_scale`), unchanged from `kelly_regime_v4`. The
hypothesis (R-08, R-136): every prior "better" volatility forecast hurt
this strategy because it reacted faster to short-lived vol spikes that
precede BTC's best forward-Sharpe states (Baur & Dimpfl 2018's inverse
leverage effect) -- so a persistence-filtered estimate that ignores those
spikes should keep the strategy MORE exposed (less de-risked) right when a
real "better forecast" would flee.

PRE-REGISTERED FALSIFICATION TEST, two parts, frozen, unchanged from
`r175_direction.md` and `r175_shared.py`'s docstrings -- reproduced here
verbatim as the operative rule this script implements mechanically:

  (a) MECHANISM CHECK, decisive regardless of any performance number, run
      FIRST. Using `r175_shared.exposure_at_episodes` and
      `r175_shared.STRESS_EPISODES` (six dated episodes), compare the
      candidate's (`msm_structural_target`) realized |exposure| in the 10
      trading days FOLLOWING each episode's own vol spike against the
      control's (`v4_target`), on BTC. FALSIFIED -- STOP HERE, do not
      proceed to (b) or the holdout -- if the candidate is not less
      de-risked (strictly higher mean exposure) than the control in at
      least 4 of 6 episodes. Checked on mechanism alone, not a
      Sharpe/drawdown number; the full 6-episode table is reported either
      way.

  (b) STANDARD PROMOTION BAR, only reachable if (a) passes. On BTC
      inner-validation via `compare(msm_structural_target, label="novel",
      ...)`: ΔSharpe>=+0.2 or a risk-matched drawdown improvement on both
      markets, ETH sign-replication in the same direction, survives the
      0.40% taker fee tier. FALSIFIED if the paired-bootstrap CI on
      `d_log_growth`/`d_sharpe` excludes zero on the losing side on either
      market. If (a) and the training-period read of (b) both look
      promising, ONE holdout read is pre-registered and taken (see the
      HOLDOUT PRE-REGISTRATION section below, written BEFORE this script
      was run against real numbers for that gate).

ROBUSTNESS/PLATEAU CHECK (pre-registered allowance, "2-4 configs, e.g. a
different N_PERSIST value (1 or 3) ... a genuine mechanism should show a
plateau, not a knife-edge"): the mechanism check (a) is ALSO run at
N_PERSIST=1 and N_PERSIST=3 (via `r175_shared.msm_forecast_daily(df,
n_persist=...)` directly, reusing the shared engine's own broadcast/scale/
deadband plumbing -- `r175_shared.py` itself is never modified), alongside
the frozen N_PERSIST=2 primary. Only N_PERSIST=2 is the decisive,
pre-registered gate; the other two are diagnostic context for whether a
failure (or a pass) is a plateau property of "keep few slow components" in
general, or an artifact of the specific frozen N=2 cut.

HOLDOUT PRE-REGISTRATION (written before this script was run against real
numbers, per the pre-registration's own requirement to state this before
looking):
  - If (a) passes (>=4/6 episodes) AND the training-period read of (b)
    clears the promotion bar on BTC inner-validation: take ONE holdout read
    on BTC, both markets (`start=OOS_START`). Conclude PROMOTE if the
    holdout also clears ΔSharpe>=+0.2 or a risk-matched drawdown
    improvement and the falsifiers do not fire; conclude NEGATIVE
    (in-sample-only effect) otherwise.
  - If (a) passes but (b) does not clear on training data: conclude
    NEGATIVE without touching the holdout.
  - If (a) fails (<4/6 episodes): conclude FALSIFIED-AT-MECHANISM-GATE
    without touching (b) or the holdout, regardless of what any performance
    number shows -- this is the whole point of a mechanism gate that is
    decisive "regardless of any performance number".

Run: `uv run python experiments/r175_novel_msm_structural.py` (repo root).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r175_shared import (  # noqa: E402
    ETH_SLICE_NAME,
    FUTURES,
    MSM_KBAR,
    N_PERSIST,
    OOS_START,
    SPOT,
    STRESS_EPISODES,
    _broadcast_vol,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    compare,
    conditional_target_scale,
    exposure_at_episodes,
    fee_at,
    load_btc,
    load_eth,
    msm_forecast_daily,
    msm_structural_target,
    print_rows,
    v4_target,
    v4_vote_frac,
)

assert N_PERSIST == 2, N_PERSIST  # the frozen, decisive value this branch tests

# ================================================================== (1)
# Pre-registered configs: 3 total, N_PERSIST in {1, 2, 3}. N_PERSIST=2 is
# PRIMARY (the frozen candidate, `msm_structural_target` itself); 1 and 3
# are the robustness/plateau sensitivity check named in the task brief.
# ==================================================================
CONFIGS = [1, 2, 3]
PRIMARY_N_PERSIST = 2
assert len(CONFIGS) == 3


def build_structural_target(n_persist: int):
    """Same construction as `r175_shared.msm_structural_target`, but with an
    explicit `n_persist` (the frozen module hardcodes `N_PERSIST=2`) --
    calls the shared engine's own `msm_forecast_daily(df, n_persist=...)`
    directly, per the task brief's explicit allowance, and reuses the
    shared module's own broadcast/scale/deadband functions verbatim (no
    reimplementation, no edit to `r175_shared.py`)."""

    def build(df: pd.DataFrame) -> np.ndarray:
        fc = msm_forecast_daily(df, n_persist=n_persist)
        vol = _broadcast_vol(df, fc["persist_mult"])
        scale = conditional_target_scale(vol)
        return apply_deadband(v4_vote_frac(df).to_numpy() * scale)

    build.__name__ = f"msm_structural_n{n_persist}"
    return build


# --------------------------------------------------------------- reporting

def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def cell(rows: list[dict], label: str, slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["label"] == label and r["slice"] == slice_name and r["market"] == market:
            return r
    return None


def print_episode_table(n_persist: int, exp_cand: dict, exp_ctrl: dict) -> int:
    """Print the 6-episode table for one N_PERSIST value; return the count
    of episodes where the candidate is strictly more exposed (less
    de-risked) than the control."""
    print(f"\n  N_PERSIST={n_persist}:")
    print(f"    {'episode':45s} {'cand |exp|':>11s} {'ctrl |exp|':>11s} {'cand>ctrl?':>10s}")
    count = 0
    for name, _date in STRESS_EPISODES:
        c = exp_cand.get(name, float("nan"))
        k = exp_ctrl.get(name, float("nan"))
        higher = bool(np.isfinite(c) and np.isfinite(k) and c > k)
        count += int(higher)
        print(f"    {name:45s} {c:>11.4f} {k:>11.4f} {'YES' if higher else 'no':>10s}")
    print(f"    -> candidate strictly less de-risked in {count}/6 episodes")
    return count


def main() -> None:
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-175 NOVEL -- persistence-filtered MSM(6) structural volatility "
       f"(N_PERSIST={N_PERSIST} frozen)\ndriving kelly_regime_v4's hysteresis "
       "scale. See r175_shared.py / r175_direction.md for direction, "
       "citations,\nnon-duplication, and the frozen falsification test; "
       "this file implements only the novel branch's own\nmechanism check "
       "and (if it survives) the promotion bar.")

    btc = load_btc()
    eth = load_eth()
    max_ts_seen += [btc.index.max(), eth.index.max()]
    assert_no_holdout(btc, "BTC full (pre-holdout)")
    assert_no_holdout(eth, "ETH full (pre-holdout)")
    print(f"\nBTC (truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")
    print(f"ETH (truncated < {OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")

    # ========================================================== STEP 0
    hr("STEP 0 -- causal truncation probes (shared engine's own vol/target "
       "builders, real BTC data)")
    btc_probe = btc.loc[:"2018-06-30"].copy()
    ok_struct = causal_truncation_probe_series(msm_structural_target, btc_probe)
    print(f"    causal_truncation_probe_series(msm_structural_target, "
          f"btc_probe): {'PASS' if ok_struct else 'FAIL'}")
    ok_robust = {}
    for n in (1, 3):
        b = build_structural_target(n)
        ok_robust[n] = causal_truncation_probe_series(b, btc_probe)
        print(f"    causal_truncation_probe_series(n_persist={n} build, "
              f"btc_probe): {'PASS' if ok_robust[n] else 'FAIL'}")
    causal_ok = ok_struct and all(ok_robust.values())
    print(f"\n    Causality (all 3 builders): {'PASS' if causal_ok else 'FAIL'}")
    if not causal_ok:
        hr("VERDICT")
        print("    VERDICT: NEGATIVE (causal truncation probe FAILED -- "
              "stopping before the mechanism check).")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    # ========================================================== STEP 1
    hr("STEP 1 -- PRE-REGISTERED FALSIFICATION TEST (a): mechanism check, "
       "decisive regardless of any performance number.\nRun FIRST, per "
       "r175_direction.md's own instruction. Candidate = msm_structural_"
       f"target (N_PERSIST={PRIMARY_N_PERSIST}, frozen);\ncontrol = v4_target "
       "(kelly_regime_v4's own unmodified EWM vol estimator). Mean |exposure| "
       "in the 10 trading\ndays FOLLOWING each of BTC's six dated stress "
       "episodes.")

    ctrl_target = v4_target(btc)
    exp_ctrl = exposure_at_episodes(ctrl_target, btc)

    episode_counts: dict[int, int] = {}
    cand_targets: dict[int, np.ndarray] = {}
    for n in CONFIGS:
        build = msm_structural_target if n == PRIMARY_N_PERSIST else build_structural_target(n)
        cand = build(btc)
        cand_targets[n] = cand
        exp_cand = exposure_at_episodes(cand, btc)
        episode_counts[n] = print_episode_table(n, exp_cand, exp_ctrl)

    primary_count = episode_counts[PRIMARY_N_PERSIST]
    mechanism_pass = primary_count >= 4
    print(f"\n    PRIMARY (frozen N_PERSIST={PRIMARY_N_PERSIST}) result: "
          f"{primary_count}/6 episodes candidate less de-risked "
          f"(threshold: >=4/6 to pass)")
    print(f"    Robustness N_PERSIST=1: {episode_counts[1]}/6   "
          f"N_PERSIST=3: {episode_counts[3]}/6")
    plateau = len({episode_counts[1], episode_counts[2], episode_counts[3]}) == 1
    print(f"    Plateau check (all three N_PERSIST give the SAME episode "
          f"count, not a knife-edge at N=2): "
          f"{'PLATEAU (identical count across 1/2/3)' if plateau else 'NOT a plateau -- counts differ by N_PERSIST'}")
    print(f"\n    TEST (a) VERDICT: {'PASS' if mechanism_pass else 'FALSIFIED'} "
          f"({primary_count}/6, need >=4/6)")

    # ========================================================== STEP 7 (counts)
    def report_counts_and_verdict(verdict: str, promotion_rows=None) -> None:
        hr("CONFIGURATION COUNT")
        print(f"    Distinct N_PERSIST configurations evaluated (mechanism "
              f"check, all 3): {len(CONFIGS)}  ({CONFIGS})")
        n_cells = len(promotion_rows) if promotion_rows else 0
        print(f"    Promotion-bar compare() backtest cells run "
              f"(3 slices x 2 markets, PRIMARY config only): {n_cells}")
        print(f"    Plus: 3 causal-truncation probes (one per N_PERSIST "
              "config), not counted as a real-data Sharpe/growth cell "
              "(same accounting convention as R-161/R-167's own entries).")
        print(f"    TOTAL evaluated configurations for the ledger's trials "
              f"count: {len(CONFIGS)} configurations"
              + (f", {n_cells} backtest cells" if n_cells else
                 " (0 backtest cells -- mechanism gate stopped the branch "
                 "before any compare() call)"))

        hr("VERDICT")
        print(f"    VERDICT: {verdict}")
        print(f"\n    Max timestamp read anywhere in this run: "
              f"{max(max_ts_seen)}   (OOS_START = {OOS_START}; strictly "
              f"earlier: {max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")

    if not mechanism_pass:
        hr("STOP -- per r175_direction.md's own pre-registered rule: "
           "test (a) failed, so this branch does NOT proceed to (b) "
           "or the holdout,\nregardless of any performance number.")
        print("    (No compare() call, no fee-tier check, no holdout read "
              "follow from here -- the rule is decisive on mechanism alone.)")
        report_counts_and_verdict("FALSIFIED-AT-MECHANISM-GATE "
                                   f"({primary_count}/6 episodes, need >=4/6)")
        hr("HOLDOUT")
        print("    Holdout consulted: NO. This script never reads a bar at "
              "or after OOS_START (2023-01-01); `load_btc`/`load_eth` "
              "truncate\n    before it and `compare`/`run_slice` (never "
              "called on this path) assert against it on every call.")
        return

    # ========================================================== STEP 2
    # (Only reached if test (a) passes -- not the case in this round's run,
    # but implemented for completeness / future re-runs at a different
    # frozen N_PERSIST.)
    hr("STEP 2 -- PROMOTION BAR (b): compare() on BTC/ETH, inner_train + "
       "inner_val + eth_replication, PRIMARY config only")
    rows = compare(msm_structural_target, label="novel", btc=btc, eth=eth)
    print_rows(rows)

    hr("STEP 3 -- fee-tier robustness (0.40% taker)")
    hi_fee_futures = fee_at(FUTURES, 0.0040)
    hi_fee_spot = fee_at(SPOT, 0.0040)
    fee_rows = compare(msm_structural_target, label="novel_fee40bp", btc=btc, eth=eth,
                       markets=(hi_fee_spot, hi_fee_futures))
    print_rows(fee_rows)

    hr("STEP 4 -- decision rule")
    val_s = cell(rows, "novel", "inner_val", SPOT.name)
    val_f = cell(rows, "novel", "inner_val", FUTURES.name)
    eth_s = cell(rows, "novel", ETH_SLICE_NAME, SPOT.name)
    eth_f = cell(rows, "novel", ETH_SLICE_NAME, FUTURES.name)

    def clears(c: dict) -> bool:
        return bool(c["d_sharpe"] >= 0.2 or (c["risk_matched"] and c["d_dd"] < 0))

    def falsified(c: dict) -> bool:
        return bool(c["excludes_zero"] and c["boot_d_loggrowth"] < 0)

    b_val = clears(val_s) and clears(val_f)
    b_eth_same_sign = (np.sign(eth_s["d_sharpe"]) == np.sign(val_s["d_sharpe"])
                       and np.sign(eth_f["d_sharpe"]) == np.sign(val_f["d_sharpe"]))
    any_falsified = any(falsified(c) for c in (val_s, val_f, eth_s, eth_f))

    print(f"    BTC inner_val clears (dSharpe>=+0.2 or risk-matched DD "
          f"improvement), both markets: {b_val}")
    print(f"    ETH sign-replication (same direction as BTC inner_val, "
          f"both markets): {b_eth_same_sign}")
    print(f"    Falsifier (bootstrap CI excludes zero on the losing side, "
          f"any market): {any_falsified}")

    training_promising = b_val and b_eth_same_sign and not any_falsified
    print(f"\n    Training-period read of (b): "
          f"{'PROMISING -- eligible for holdout' if training_promising else 'DOES NOT CLEAR -- NEGATIVE, no holdout'}")

    if not training_promising:
        report_counts_and_verdict("NEGATIVE (mechanism check passed but "
                                   "promotion bar did not clear on "
                                   "training data)", promotion_rows=rows + fee_rows)
        hr("HOLDOUT")
        print("    Holdout consulted: NO (pre-registered rule: only read "
              "the holdout if both (a) and the training read of (b) look "
              "promising).")
        return

    # ========================================================== STEP 5 (holdout)
    hr("STEP 5 -- ONE pre-registered holdout read (BTC, both markets, "
       f"start={OOS_START})")
    print("    Pre-registered BEFORE looking (see module docstring): "
          "PROMOTE if holdout clears dSharpe>=+0.2 or a risk-matched "
          "drawdown\n    improvement and no falsifier fires; NEGATIVE "
          "(in-sample-only) otherwise. Uses `scripts.experiment.ev` + "
          "`tradebot.data.load_dataset`\n    (the project's own untruncated "
          "holdout-reading path) rather than r175_shared's deliberately "
          "holdout-truncated `load_btc`.")

    from tradebot.data import load_dataset  # noqa: E402  (holdout-only import, isolated here)

    from experiments.r175_shared import TargetStrategy  # noqa: E402

    btc_full, _label = load_dataset(ROOT / "data", "spot")
    cand_strat = TargetStrategy(msm_structural_target, name="r175_novel_msm_structural")
    ctrl_strat = TargetStrategy(v4_target, name="kelly_regime_v4")

    holdout_rows = []
    for market in (SPOT, FUTURES):
        from experiments.r175_shared import paired_diff, run_slice
        a = run_slice(cand_strat, btc_full, OOS_START, None, "holdout", market)
        b = run_slice(ctrl_strat, btc_full, OOS_START, None, "holdout", market)
        pr = paired_diff(a.daily, b.daily)
        d_sharpe = a.sharpe - b.sharpe
        exp_ratio = a.mean_abs_exposure / b.mean_abs_exposure if b.mean_abs_exposure else float("nan")
        vol_ratio = a.realized_vol / b.realized_vol if b.realized_vol else float("nan")
        risk_matched = bool(np.isfinite(exp_ratio) and np.isfinite(vol_ratio)
                            and 0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
        d_dd = a.max_drawdown_pct - b.max_drawdown_pct
        clears_holdout = bool(d_sharpe >= 0.2 or (risk_matched and d_dd < 0))
        falsified_holdout = bool((pr.diff.lo > 0 or pr.diff.hi < 0) and pr.diff.point < 0)
        holdout_rows.append(dict(market=market.name, d_sharpe=d_sharpe, d_dd=d_dd,
                                 risk_matched=risk_matched, clears=clears_holdout,
                                 falsified=falsified_holdout,
                                 boot_lo=pr.diff.lo, boot_hi=pr.diff.hi))
        print(f"    {market.name:11s} dSharpe={d_sharpe:+.3f} dDD={d_dd:+.2f} "
              f"risk_matched={risk_matched} clears={clears_holdout} "
              f"falsified={falsified_holdout} boot=[{pr.diff.lo:+.4f},{pr.diff.hi:+.4f}]")

    holdout_promotes = all(r["clears"] for r in holdout_rows) and not any(r["falsified"] for r in holdout_rows)
    verdict = "PROMOTE" if holdout_promotes else "NEGATIVE (in-sample-only effect, holdout did not replicate)"
    report_counts_and_verdict(verdict, promotion_rows=rows + fee_rows)
    hr("HOLDOUT")
    print(f"    Holdout consulted: YES (BTC, both markets, start={OOS_START}). "
          "Increment the project's holdout counter by 1 in the ledger.")
    return


if __name__ == "__main__":
    main()
