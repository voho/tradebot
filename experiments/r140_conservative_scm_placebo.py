"""R-140 CONSERVATIVE branch: the classical Abadie-Diamond-Hainmueller (2010)
in-space placebo test applied to the SCM machinery frozen in
`experiments/r140_shared.py` (read-only here -- see that module's docstring
for the full pre-registration: mechanism, literature, non-duplication
argument, named failure modes, and the frozen three-way decision rule).

One-paragraph restatement of just this branch's own procedure, for a reader
of this file alone: for each of the 4 `DONOR_COVERED_EPISODES`, on BTC and
separately on ETH, fit SCM donor weights on a `PRE_FIT_DAYS`-long window
immediately preceding the event window (`fit_scm_weights`), build the real
target's synthetic counterfactual (`synthetic_path`), and read off its real
`event_gap`. Then, in turn, treat each of the 6 donor instruments as if it
were the "treated" unit -- refit its own synthetic counterfactual from the
OTHER 5 donors over the SAME fit window -- and compute its own placebo
`event_gap`. Rank the real gap's absolute value within that 6-draw placebo
distribution, Abadie's own convention: `p = (1 + #placebos with |gap| >=
|real gap|) / (1 + n_placebos)`. Report the Step-A validity diagnostic
(`pre_fit_rmspe_ratio`) for every cell BEFORE any significance number. Pool
across the 4 episodes for one BTC read and one ETH read (pooling method
disclosed below, chosen once, before any pooled number was computed). Apply
the frozen three-way decision rule verbatim.

No bar dated `OOS_START = 2023-01-01` or later is read anywhere in this file
-- `load_v4_and_extended_donor_returns` already restricts the target to
`<= INNER_VAL_END` internally (via `candidate_and_matched_daily_logret`);
the donor panel is explicitly truncated to `<= INNER_VAL_END` immediately
after loading below (its own loader has no date restriction, since other
rounds that reuse it read past 2022), and every fit/event window touched is
asserted to end before `OOS_START` (the latest episode, 2022-11-08 FTX, has
its post-window end at 2022-11-28, five weeks before the holdout starts).

--- Implementation choices not fully pinned by the pre-registration,
disclosed explicitly rather than picked silently: ---

1. **Fit-window placement.** The pre-registration fixes `PRE_FIT_DAYS` but
   not exactly which `PRE_FIT_DAYS`-long span it covers. Chosen here: the
   `PRE_FIT_DAYS` calendar days immediately preceding the event window's own
   start (`event_date - WINDOW_PRE_DAYS`), i.e. `fit_end = event_date -
   WINDOW_PRE_DAYS - 1day`, `fit_start = fit_end - (PRE_FIT_DAYS-1)days` --
   so the fit window and the event window never overlap by construction,
   matching the module docstring's own stated intent for `PRE_FIT_DAYS`
   ("chosen ... so no episode's fit window ever overlaps another episode's
   post-window"). Note the COVID episode's fit window reaches back to
   2019-10, before the donor panel's own 2020-01-01 coverage start -- the
   donor intersection in `fit_scm_weights` handles this automatically by
   using whatever is available, so COVID's actual fit-n is shorter than 150
   days. Reported explicitly per cell below.

2. **RMSPE gate values.** Recomputed independently in this file via the
   shared primitives (`fit_scm_weights` / `synthetic_path` / `pre_fit_rmspe`)
   rather than by calling `pre_fit_rmspe_ratio` a second time on top of our
   own loop -- both compute literally the same formula
   (`target_rmspe / median(donor-placebo rmspe)`) from the same deterministic
   fit (same default seed/n_iter), so calling both would just refit every
   donor placebo twice for an identical number. We call `pre_fit_rmspe_ratio`
   directly ONCE per cell as an independent cross-check against our own
   loop's ratio (Step-0 sanity), then use our own loop's numbers (which also
   already produced the placebo event gaps this branch needs) throughout.

3. **Pooling method across the 4 episodes (pre-registration explicitly
   leaves this to the branch; picked once, disclosed, not compared against
   alternatives):** mean **absolute** event gap across the 4 donor-covered
   episodes as the pooled test statistic -- the direct multi-episode
   extension of the per-episode Abadie ranking convention this round already
   uses (`|gap|`, not signed gap, exactly because the smoke test found gaps
   vary in sign across episodes, so a signed pooling would let episodes
   cancel rather than corroborate). The pooled null is the EXACT
   combinatorial permutation distribution of the same statistic: independent
   per-episode donor relabeling (choose 1 of 6 placebo donors for each of the
   4 episodes, 6^4 = 1296 combinations), each combination's statistic being
   the mean of the 4 chosen placebo |gap| values. Independence across
   episodes is the same property the module docstring already leans on to
   justify `PRE_FIT_DAYS` (non-overlapping fit/event windows across
   episodes, so no two episodes' placebo draws share information). This is
   an exact, non-asymptotic permutation p-value (`p = (1 + #combos with
   pooled-placebo-stat >= pooled-real-stat) / (1 + 1296)`), matching this
   project's stated preference for small-N-valid exact tests over CLT-based
   ones (the same reason R-138 chose Nguyen-Wolf over a normal-approximation
   permutation test).

Run: `python experiments/r140_conservative_scm_placebo.py`
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r140_shared import (  # noqa: E402  frozen, read-only
    BTC_P_GATE,
    DONOR_COVERED_EPISODES,
    ETH_P_GATE,
    INNER_VAL_END,
    OOS_START,
    PRE_FIT_DAYS,
    RMSPE_GATE,
    WINDOW_POST_DAYS,
    WINDOW_PRE_DAYS,
    event_gap,
    fit_scm_weights,
    load_v4_and_extended_donor_returns,
    pre_fit_rmspe,
    pre_fit_rmspe_ratio,
    synthetic_path,
)

N_DONORS = 6
N_COMBOS = N_DONORS ** len(DONOR_COVERED_EPISODES)  # 6^4 = 1296

OOS_START_TS = pd.Timestamp(OOS_START, tz="UTC")
INNER_VAL_END_TS = pd.Timestamp(INNER_VAL_END, tz="UTC")

# configuration counter, incremented at every SCM weight-fit call
_config_count = {"n": 0}


def _counted_fit(donors: pd.DataFrame, target: pd.Series,
                  fit_start: pd.Timestamp, fit_end: pd.Timestamp) -> pd.Series:
    _config_count["n"] += 1
    return fit_scm_weights(donors, target, fit_start, fit_end)


def fit_window(event_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    event_ts = pd.Timestamp(event_date, tz="UTC")
    event_window_start = event_ts - pd.Timedelta(days=WINDOW_PRE_DAYS)
    fit_end = event_window_start - pd.Timedelta(days=1)
    fit_start = fit_end - pd.Timedelta(days=PRE_FIT_DAYS - 1)
    return fit_start, fit_end


def assert_no_holdout_touched(event_date: str) -> None:
    event_ts = pd.Timestamp(event_date, tz="UTC")
    post_end = event_ts + pd.Timedelta(days=WINDOW_POST_DAYS)
    assert post_end < OOS_START_TS, (
        f"holdout breach: episode {event_date}'s post-window end {post_end} "
        f"is at/after {OOS_START_TS}"
    )
    fit_start, fit_end = fit_window(event_date)
    assert fit_end < OOS_START_TS, "holdout breach: fit window"


def one_cell(which: str, episode_name: str, episode_date: str,
             donors_full: pd.DataFrame, target_full: pd.Series) -> dict:
    """Real + 6 donor-placebo SCM fits and event gaps for one
    (market/which, episode) cell. Returns a dict of all diagnostics."""
    assert_no_holdout_touched(episode_date)
    fit_start, fit_end = fit_window(episode_date)
    episode_ts = pd.Timestamp(episode_date, tz="UTC")

    # actual fit-n (days available in [fit_start, fit_end] after intersecting
    # target/donor indices) -- reported because COVID's nominal PRE_FIT_DAYS
    # window reaches before donor coverage starts.
    idx_check = donors_full.index.intersection(target_full.index)
    idx_check = idx_check[(idx_check >= fit_start) & (idx_check <= fit_end)]
    fit_n_days = len(idx_check)

    # --- real target ---
    w_real = _counted_fit(donors_full, target_full, fit_start, fit_end)
    synth_real = synthetic_path(donors_full, w_real, donors_full.index)
    real_rmspe = pre_fit_rmspe(target_full, synth_real, fit_start, fit_end)
    real_gap = event_gap(target_full, synth_real, episode_ts)

    # --- 6 donor placebos ---
    placebo_gaps = {}
    placebo_rmspes = {}
    for donor_col in donors_full.columns:
        others = donors_full.drop(columns=[donor_col])
        placebo_target = donors_full[donor_col]
        w_p = _counted_fit(others, placebo_target, fit_start, fit_end)
        synth_p = synthetic_path(others, w_p, others.index)
        placebo_rmspes[donor_col] = pre_fit_rmspe(placebo_target, synth_p, fit_start, fit_end)
        placebo_gaps[donor_col] = event_gap(placebo_target, synth_p, episode_ts)

    median_donor_rmspe = float(np.median(list(placebo_rmspes.values())))
    ratio = real_rmspe / median_donor_rmspe if median_donor_rmspe > 0 else float("inf")

    # Step-0 sanity cross-check: shared pre_fit_rmspe_ratio should agree
    # (same deterministic fit, same formula, computed independently here).
    _config_count["n"] += 7  # pre_fit_rmspe_ratio internally fits 1 target + 6 donors
    ratio_shared, target_rmspe_shared, median_shared = pre_fit_rmspe_ratio(
        donors_full, target_full, fit_start, fit_end
    )
    agree = (
        abs(ratio - ratio_shared) < 1e-9
        and abs(real_rmspe - target_rmspe_shared) < 1e-9
        and abs(median_donor_rmspe - median_shared) < 1e-9
    )

    placebo_abs = np.array([abs(g) for g in placebo_gaps.values()])
    n_exceed = int(np.sum(placebo_abs >= abs(real_gap)))
    p_episode = (1 + n_exceed) / (1 + N_DONORS)

    return {
        "which": which,
        "episode": episode_name,
        "date": episode_date,
        "fit_start": fit_start,
        "fit_end": fit_end,
        "fit_n_days": fit_n_days,
        "rmspe_ratio": ratio,
        "rmspe_ratio_shared_crosscheck": ratio_shared,
        "crosscheck_agrees": agree,
        "target_rmspe": real_rmspe,
        "median_donor_rmspe": median_donor_rmspe,
        "real_gap": real_gap,
        "placebo_gaps": placebo_gaps,
        "n_exceed": n_exceed,
        "p_episode": p_episode,
        "gate_pass": ratio < RMSPE_GATE,
    }


def pool_market(cells: list[dict]) -> dict:
    """Exact combinatorial pooling: mean |gap| across the 4 episodes as the
    statistic, exact null over all 6^4 independent per-episode donor
    relabelings."""
    assert len(cells) == len(DONOR_COVERED_EPISODES)
    real_abs = [abs(c["real_gap"]) for c in cells]
    pooled_real_stat = float(np.mean(real_abs))
    # Signed mean (NOT the significance statistic -- used only to check the
    # decision rule's own "replicates with the SAME SIGN" clause, which the
    # |gap|-based significance statistic above cannot answer by itself).
    pooled_real_signed_mean = float(np.mean([c["real_gap"] for c in cells]))

    per_episode_placebo_abs = [
        np.array([abs(g) for g in c["placebo_gaps"].values()]) for c in cells
    ]  # each length 6, in a fixed donor order per episode (order doesn't
       # matter for the combinatorial product below)

    combo_stats = np.empty(N_COMBOS, dtype=float)
    for i, combo in enumerate(itertools.product(range(N_DONORS), repeat=len(cells))):
        vals = [per_episode_placebo_abs[e][combo[e]] for e in range(len(cells))]
        combo_stats[i] = np.mean(vals)

    n_exceed = int(np.sum(combo_stats >= pooled_real_stat))
    p_pooled = (1 + n_exceed) / (1 + N_COMBOS)

    return {
        "pooled_real_stat": pooled_real_stat,
        "pooled_real_signed_mean": pooled_real_signed_mean,
        "n_combos": N_COMBOS,
        "n_exceed": n_exceed,
        "p_pooled": p_pooled,
    }


def run_market(which: str) -> dict:
    target_full, donors_full = load_v4_and_extended_donor_returns(which)
    # Defensive truncation: load_donor_daily_returns() has no date
    # restriction on its own (other rounds reuse it past OOS_START); this
    # round never reads or uses anything at/after OOS_START.
    donors_full = donors_full.loc[:INNER_VAL_END_TS.tz_localize(None)] \
        if donors_full.index.tz is None else donors_full.loc[:INNER_VAL_END_TS]
    target_full = target_full.loc[:INNER_VAL_END_TS.tz_localize(None)] \
        if target_full.index.tz is None else target_full.loc[:INNER_VAL_END_TS]

    max_touched = max(donors_full.index.max(), target_full.index.max())
    max_touched_ts = pd.Timestamp(max_touched)
    boundary = OOS_START_TS.tz_localize(None) if max_touched_ts.tz is None else OOS_START_TS
    assert max_touched_ts < boundary, (
        f"holdout breach: {which} data touches {max_touched_ts} >= {OOS_START}"
    )

    cells = []
    for episode_name, episode_date in DONOR_COVERED_EPISODES:
        cell = one_cell(which, episode_name, episode_date, donors_full, target_full)
        cells.append(cell)

    gate_fail_count = sum(1 for c in cells if not c["gate_pass"])
    pooled = pool_market(cells)

    return {
        "which": which,
        "max_bar_touched": max_touched_ts,
        "cells": cells,
        "gate_fail_count": gate_fail_count,
        "gate_pass_majority": gate_fail_count < 3,  # majority = >=3 of 4 fail -> INVALID
        "pooled": pooled,
    }


def fmt_cells(cells: list[dict]) -> str:
    lines = []
    for c in cells:
        lines.append(
            f"    {c['episode']:42s} ({c['date']}): rmspe_ratio={c['rmspe_ratio']:.4f} "
            f"(crosscheck={c['rmspe_ratio_shared_crosscheck']:.4f}, agree={c['crosscheck_agrees']}) "
            f"gate_pass={c['gate_pass']} fit_n_days={c['fit_n_days']} "
            f"real_gap={c['real_gap']:+.5f} n_exceed={c['n_exceed']}/6 p_episode={c['p_episode']:.4f}"
        )
        placebo_str = ", ".join(f"{k}={v:+.5f}" for k, v in c["placebo_gaps"].items())
        lines.append(f"        placebo gaps: {placebo_str}")
    return "\n".join(lines)


def main() -> None:
    print("=" * 100)
    print("R-140 CONSERVATIVE: Abadie-Diamond-Hainmueller in-space placebo SCM test")
    print("=" * 100)

    results = {}
    for which in ("btc", "eth"):
        print(f"\n--- {which.upper()} ---")
        res = run_market(which)
        results[which] = res
        print(f"max bar touched: {res['max_bar_touched']}  (OOS_START={OOS_START})")
        print(fmt_cells(res["cells"]))
        print(f"  Step-A gate: {res['gate_fail_count']}/4 episodes FAIL rmspe_ratio<{RMSPE_GATE}")
        pooled = res["pooled"]
        print(f"  pooled |gap| stat: real={pooled['pooled_real_stat']:.5f}  "
              f"n_combos={pooled['n_combos']}  n_exceed={pooled['n_exceed']}  "
              f"p_pooled={pooled['p_pooled']:.6f}")
        print(f"  pooled SIGNED mean gap (sign-check only, not the significance "
              f"statistic): {pooled['pooled_real_signed_mean']:+.5f}")

    print(f"\nTotal SCM weight-fit configurations evaluated: {_config_count['n']}")

    # ---------------- Decision rule (verbatim from r140_shared docstring) --
    btc = results["btc"]
    eth = results["eth"]

    print("\n" + "=" * 100)
    print("DECISION RULE")
    print("=" * 100)

    if not btc["gate_pass_majority"]:
        verdict = "INVALID (Step-A stop)"
        reason = (
            f"BTC Step-A gate failed for {btc['gate_fail_count']}/4 (>=3) donor-covered "
            f"episodes (rmspe_ratio >= {RMSPE_GATE}). Failure mode (a) "
            "(invalidity by construction: donor pool too correlated with target / "
            "common-shock crash to produce a trustworthy pre-fit) is the predicted one."
        )
    else:
        btc_p = btc["pooled"]["p_pooled"]
        eth_p = eth["pooled"]["p_pooled"]
        btc_sign = btc["pooled"]["pooled_real_signed_mean"]
        eth_sign = eth["pooled"]["pooled_real_signed_mean"]
        same_sign = (btc_sign > 0) == (eth_sign > 0)
        btc_sig = btc_p < BTC_P_GATE
        eth_replicates = (eth_p < ETH_P_GATE) and same_sign
        if btc_sig and eth_replicates:
            verdict = "VALID & CONFIRMS"
            reason = (
                f"Step-A gate passes on BTC ({btc['gate_fail_count']}/4 failing, <3). "
                f"Pooled BTC p={btc_p:.6f} < BTC_P_GATE={BTC_P_GATE} "
                f"(signed pooled gap {btc_sign:+.5f}). "
                f"Pooled ETH p={eth_p:.6f} < ETH_P_GATE={ETH_P_GATE} "
                f"(signed pooled gap {eth_sign:+.5f}), SAME SIGN as BTC={same_sign}."
            )
        else:
            verdict = "VALID & DOES NOT CONFIRM"
            reason = (
                f"Step-A gate passes on BTC ({btc['gate_fail_count']}/4 failing, <3). "
                f"Pooled BTC p={btc_p:.6f} ({'<' if btc_sig else '>='} {BTC_P_GATE} -> "
                f"{'sig' if btc_sig else 'NOT sig'}; signed gap {btc_sign:+.5f}). "
                f"Pooled ETH p={eth_p:.6f} ({'<' if eth_p < ETH_P_GATE else '>='} {ETH_P_GATE}"
                f"; signed gap {eth_sign:+.5f}; same sign as BTC={same_sign}) -> "
                f"{'replicates' if eth_replicates else 'does NOT replicate'}."
            )

    print(f"\nVERDICT: {verdict}")
    print(f"REASON: {reason}")
    print("=" * 100)


if __name__ == "__main__":
    main()
