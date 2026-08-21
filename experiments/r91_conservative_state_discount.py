#!/usr/bin/env python
"""R-91 CONSERVATIVE branch: a FIXED, literature-motivated discount factor
delta applied to `kelly_regime_v4`'s raw desired exposure specifically in
the two Goulding-Harvey-Mazzoleni (2023) "turning point" states (Correction,
Rebound), left unchanged (scaler = 1.0) in the two trend-agreement states
(Bull, Bear). The full citation trail, the round's direction, the
not-a-duplicate-of reasoning, the shared state-labelling machinery
(`state_labels`, `is_turning_point`) and the shared A0/A1/A2/A3 gates plus
the B1-B5 promotion bar all live in `experiments/r91_shared.py`'s module
docstring; this file does not repeat that reasoning and does not edit that
module. It implements the frozen mechanism and reports every frozen gate,
pass or fail, exactly as pre-registered.

MECHANISM (one sentence, frozen, no deviation):

    def build_candidate(df, delta):
        state = state_labels(df)
        scaler = np.where(is_turning_point(state), delta, 1.0)
        raw = v4_vote_frac(df) * scaler * v4_scale(df)
        return apply_deadband(raw)

`state_labels`, `is_turning_point`, `v4_vote_frac`, `v4_scale` and
`apply_deadband` are ALL `r91_shared`'s own read-only reproductions of
v4's own factors / GHM's own state partition -- nothing here re-derives
them.

A0 KILL SWITCH -- ALREADY FIRED (per the operator's pre-measurement, which
this file independently re-derives in Step 0 below): causal, inner-train
BTC state-conditional Sharpe-like stats do NOT rank both turning-point
states below both trend-agreement states (Rebound ranks ABOVE Bear). Per
`r91_shared`'s pre-registration this makes the branch NEGATIVE regardless
of any downstream number. Per this project's own R-89 Step-0 convention
(reported in docs/LEDGER.md: a fired kill switch still completes its full
sweep and reports B1-B5 for the record), this file proceeds to implement,
run and fully measure the branch end-to-end anyway -- the verdict printed
at the end already accounts for the fired A0 switch, but every A1-A3, B1-B5
number is a real, executed measurement, not a placeholder.

FROZEN GRID -- exactly 5 configurations, none added or dropped after any
result:
    identity / control point (NOT one of "the 4 swept configs", the A1
    check): delta = 1.0. Must reproduce `r91_shared.v4_target(df)`
    bit-for-bit.
    the swept grid, 4 configs: delta in {0.0, 0.25, 0.5, 0.75}
    (0.0 = fully flatten exposure at turning points; 0.75 = a mild
    one-quarter discount), bracketing "GHM's own qualitative claim taken
    literally" without being fitted to any of this project's own return
    data.

FROZEN DECISION RULE (default REJECT; every clause reported PASS or FAIL,
no threshold moved after seeing any result):

  Step 0 (A0 re-derivation): causal_state_stats on inner-train BTC bar
    returns grouped by state_labels; report mean/vol/Sharpe-like per state
    and whether Correction AND Rebound both rank below Bull AND Bear.

  Step A (mechanism gate, before any performance number is read):
    A1 identity: delta=1.0 reproduces v4_target(df) bit-for-bit on real
        BTC inner-train (max abs diff should be 0.0 or float-epsilon).
    A2 non-inertness: R^2 of each of the 4 swept configs' raw exposure path
        against v4's own raw exposure path, on inner-train, must be < 0.98
        (every config's R^2 reported regardless of pass/fail).
    A3 causality: causal_truncation_probe(build_fn, df) run once per swept
        delta's build function (4) plus the identity build function (1) =
        5 runs total, each reported PASS/FAIL.

  Step B (selection): compare() over slice_names=("inner_train",
    "inner_val"), markets=(SPOT, FUTURES), for each of the 4 swept deltas
    on BTC. Full 4x4 table reported. Selection statistic: the inner-
    validation paired log-growth difference vs v4 on futures_5x, among
    Step-A survivors.

  Promotion bar -- ALL FIVE must hold for "CANDIDATE FOR HOLDOUT", else
  "NEGATIVE" (and, per A0 above, this branch is already disqualified by
  pre-registration independent of what follows):
    B1  paired block-bootstrap difference (candidate vs v4, log growth)
        excludes zero in >= 1 of the 4 (slice x market) cells for the
        finalist, AND the point estimate is positive in ALL FOUR cells.
    B2  EITHER dSharpe > +0.2 on inner-validation on BOTH markets, OR a
        max-drawdown improvement on inner-validation on BOTH markets
        WHERE risk_matched is True for both cells.
    B3  plateau not peak: the finalist's immediate swept-grid neighbours'
        inner-validation futures d_loggrowth reported, stated explicitly
        whether they move the same direction as the finalist.
    B4  falsification, ETH replication (Bitfinex ETH, pre-2023, inner-
        train only -- this series ends 2019-12-31, so inner-validation is
        empty on ETH and this is said plainly): the finalist must show
        the SAME SIGN of d_loggrowth on BTC inner-train and ETH inner-train
        on BOTH markets. Opposite sign on either market fails B4.
    B5  cost robustness: re-run the finalist's BTC inner-validation cells
        at a 0.40% taker fee via `r91_shared.fee_at`. The decisive check
        per project convention is SPOT at 0.40% taker; futures reported
        for completeness. Sign of the B1 point estimate must not reverse.

This file never reads a bar at or after OOS_START (2023-01-01): every load
goes through `r91_shared`'s truncating, asserting loaders, and the max
timestamp actually read anywhere in this run is tracked and printed at the
end of main().
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r91_shared import (  # noqa: E402
    BEAR,
    BULL,
    CORRECTION,
    FUTURES,
    INNER_TRAIN_END,
    OOS_START,
    REBOUND,
    SPOT,
    STATE_NAMES,
    apply_deadband,
    causal_state_stats,
    causal_truncation_probe,
    compare,
    fee_at,
    is_turning_point,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    state_labels,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

# ---------------------------------------------------------------- frozen grid
DELTA_GRID = (0.0, 0.25, 0.5, 0.75)   # the 4 swept configs
IDENTITY_DELTA = 1.0                  # the A1 identity/control point
R2_CEILING = 0.98                     # frozen A2 bar
SHARPE_FLOOR = 0.2                    # frozen B2 bar (R-20 noise floor)
HIGH_FEE = 0.0040                     # frozen B5 taker fee: 0.40%


def label_of(delta: float) -> str:
    return f"delta={delta:g}"


# ------------------------------------------------------------- the mechanism

def raw_exposure(df: pd.DataFrame, delta: float) -> np.ndarray:
    """Candidate's PRE-deadband raw desired exposure (for A2's R^2 check)."""
    state = state_labels(df)
    scaler = np.where(is_turning_point(state), delta, 1.0)
    return v4_vote_frac(df) * scaler * v4_scale(df)


def build_candidate(df: pd.DataFrame, delta: float) -> np.ndarray:
    """Pure function of the bars it is handed -- what compare()/
    causal_truncation_probe call. Frozen mechanism, exactly as pre-
    registered in this file's docstring."""
    return apply_deadband(raw_exposure(df, delta))


def make_builder(delta: float):
    return lambda d: build_candidate(d, delta)


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def cell(rows: list[dict], slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["slice"] == slice_name and r["market"] == market:
            return r
    return None


# --------------------------------------------------------------- Step 0: A0

def step_0(df_train: pd.DataFrame) -> dict:
    hr("STEP 0 -- A0 kill-switch RE-DERIVATION (independent sanity check "
       "of the operator's pre-measurement)")
    state = state_labels(df_train)
    bar_returns = np.log(df_train["close"]).diff().to_numpy()
    stats = causal_state_stats(bar_returns, state)
    print(f"\nCausal, inner-train-only, bar-level log-return state-conditional stats "
          f"(BTC, {len(df_train):,} bars, {df_train.index[0]} -> {df_train.index[-1]}):")
    print(f"\n{'state':>12s} {'n':>10s} {'mean':>12s} {'vol':>10s} {'sharpe-like':>12s}")
    print("-" * 60)
    for k in (BULL, BEAR, CORRECTION, REBOUND):
        s = stats[k]
        print(f"{STATE_NAMES[k]:>12s} {s['n']:>10d} {s['mean']:>12.3e} {s['vol']:>10.3e} "
              f"{s['sharpe']:>12.3f}")

    sharpe = {k: stats[k]["sharpe"] for k in (BULL, BEAR, CORRECTION, REBOUND)}
    ranks_below = (sharpe[CORRECTION] < sharpe[BULL] and sharpe[CORRECTION] < sharpe[BEAR]
                   and sharpe[REBOUND] < sharpe[BULL] and sharpe[REBOUND] < sharpe[BEAR])
    print(f"\nOperator's pre-measurement (quoted in the task brief): Bull +0.190, "
          f"Bear -0.103, Correction -0.066, Rebound +0.185.")
    print(f"This independent re-derivation:                          Bull {sharpe[BULL]:+.3f}, "
          f"Bear {sharpe[BEAR]:+.3f}, Correction {sharpe[CORRECTION]:+.3f}, "
          f"Rebound {sharpe[REBOUND]:+.3f}.")
    print(f"\nA0 rule: Correction AND Rebound must both rank below Bull AND Bear.")
    print(f"    Correction < Bull: {sharpe[CORRECTION] < sharpe[BULL]}   "
          f"Correction < Bear: {sharpe[CORRECTION] < sharpe[BEAR]}")
    print(f"    Rebound   < Bull: {sharpe[REBOUND] < sharpe[BULL]}   "
          f"Rebound   < Bear: {sharpe[REBOUND] < sharpe[BEAR]}")
    print(f"\n    A0: {'PASSES (mechanism replicates)' if ranks_below else 'FAILS -- kill switch FIRES'}"
          f"{'  (Rebound ranks above Bear)' if not ranks_below else ''}")
    return dict(sharpe=sharpe, ranks_below=ranks_below, stats=stats)


# --------------------------------------------------------------- Step A gate

def step_a(df_train: pd.DataFrame) -> dict:
    hr("STEP A -- mechanism gate (identity + 4 swept configs), before any "
       "performance number")

    # ---- A1: identity, bit-for-bit against v4_target on real BTC inner-train.
    mine = build_candidate(df_train, IDENTITY_DELTA)
    theirs = v4_target(df_train)
    max_abs = float(np.max(np.abs(mine - theirs)))
    a1_pass = bool(np.allclose(mine, theirs, atol=1e-12, rtol=0.0))
    exact = bool(np.array_equal(mine, theirs))
    print(f"\nA1 identity (delta={IDENTITY_DELTA}) vs r91_shared.v4_target(df) on real "
          f"BTC inner-train ({len(df_train):,} bars, "
          f"{df_train.index[0]} -> {df_train.index[-1]}):")
    print(f"    max |candidate - v4_target| = {max_abs:.3e}   exact array equality = "
          f"{exact}   allclose(atol=1e-12) = {a1_pass}   -> {'PASS' if a1_pass else 'FAIL'}")
    if not a1_pass:
        raise AssertionError("A1 identity FAILED -- nothing below would be interpretable.")

    # ---- A2: non-inertness -- R^2 of raw exposure vs v4's own raw exposure,
    #          on inner-train, per swept config. Must be < 0.98.
    v4_raw = v4_vote_frac(df_train) * v4_scale(df_train)
    print(f"\nA2 non-inertness -- R^2 of each of the {len(DELTA_GRID)} swept configs' RAW "
          f"exposure path\n    against v4's own raw exposure path (frac * scale, pre-deadband), "
          f"on inner-train.\n    Must be < {R2_CEILING} (reported for every config regardless "
          f"of pass/fail -- information,\n    not a reason to drop a config).")
    print(f"\n{'#':>3s} {'config':>12s} {'R^2 vs v4 raw':>14s} {'A2':>7s}")
    print("-" * 42)
    a2_rows = []
    for i, d in enumerate(DELTA_GRID, 1):
        cand_raw = raw_exposure(df_train, d)
        rsq = r_squared(cand_raw, v4_raw)
        inert = not (rsq < R2_CEILING)
        a2_rows.append(dict(delta=d, label=label_of(d), r2=rsq, inert=inert))
        print(f"{i:>3d} {label_of(d):>12s} {rsq:>14.6f} {'INERT' if inert else 'pass':>7s}")
    n_inert = sum(r["inert"] for r in a2_rows)
    print(f"\n    {n_inert} of {len(DELTA_GRID)} swept configs are INERT (R^2 >= "
          f"{R2_CEILING}) and would be excluded from Step B: "
          f"{', '.join(r['label'] for r in a2_rows if r['inert']) or '(none)'}")

    # ---- A3: causality, identity + every swept config, on real BTC inner-train.
    print("\nA3 causality -- causal_truncation_probe on the real BTC inner-train frame "
          "(rebuild the\n    target on truncated frames at 55% and 80%; the surviving "
          "prefix must match bit-for-bit),\n    run once per swept delta's build "
          "function plus the identity build function (5 runs total).")
    a3_results = {}
    for d in (IDENTITY_DELTA,) + DELTA_GRID:
        ok = causal_truncation_probe(make_builder(d), df_train)
        a3_results[d] = ok
        print(f"    {label_of(d):>12s} : {'PASS' if ok else 'FAIL'}")
    if not all(a3_results.values()):
        raise AssertionError(f"A3 causality FAILED for: "
                             f"{[d for d, ok in a3_results.items() if not ok]}")

    return dict(a1_max_abs=max_abs, a1_pass=a1_pass, a2_rows=a2_rows, a3_results=a3_results)


# --------------------------------------------------------------- Step B eval

def step_b(df_full: pd.DataFrame, live_configs: list[float]) -> dict:
    hr("STEP B -- evaluation: every swept config (all Step-A survivors), all "
       "four (slice x market) cells vs v4")
    print(f"\nEvaluating {len(live_configs)} configs x 4 cells "
          f"(inner_train/inner_val x spot/futures_5x), candidate and control, "
          f"paired block bootstrap (30-day blocks, 2000 draws).")
    print("A negative dlogG means the state-discount LOSES to v4 in that cell.\n")

    by_cfg: dict[float, list[dict]] = {}
    for k, d in enumerate(live_configs, 1):
        lbl = label_of(d)
        rows = compare(make_builder(d), df_full, label=lbl,
                       markets=(SPOT, FUTURES),
                       slice_names=("inner_train", "inner_val"))
        by_cfg[d] = rows
        print(f"--- [{k}/{len(live_configs)}] {lbl}")
        print_rows(rows)
        print()
    return by_cfg


# ------------------------------------------------------------------- runner

def main() -> None:
    max_ts_seen: list[pd.Timestamp] = []
    n_configs_evaluated = 0

    hr("R-91 CONSERVATIVE -- fixed discount on v4's raw exposure at GHM "
       "turning-point states (Correction/Rebound).\n5 frozen configurations. "
       "A0 kill switch already fired per pre-measurement -- proceeding to "
       "full measurement per R-89 Step-0 convention.\nDefault verdict: "
       "NEGATIVE.")
    df_full = load_btc()
    max_ts_seen.append(df_full.index.max())
    df_train = df_full.loc[:INNER_TRAIN_END]
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(df_full):,} bars, "
          f"{df_full.index[0]} -> {df_full.index[-1]}")
    print(f"inner-train frame: {len(df_train):,} bars, {df_train.index[0]} -> "
          f"{df_train.index[-1]}")

    # ------------------------------------------------------------- Step 0
    a0 = step_0(df_train)

    # ------------------------------------------------------------- Step A
    gate = step_a(df_train)
    n_configs_evaluated += 5  # identity + 4 swept, Step-A mechanism gate

    live_configs = list(DELTA_GRID)  # A2 is informational per the brief;
    # report inert configs but do not drop any of the frozen 4 from Step B.
    n_inert = sum(r["inert"] for r in gate["a2_rows"])
    if n_inert:
        print(f"\nNOTE: {n_inert} config(s) flagged INERT at A2 but the frozen grid "
              f"has exactly 4 swept configs and none is dropped after seeing a "
              f"result -- all 4 proceed to Step B, as pre-registered "
              f"(\"report the actual R^2 for every config even if some fail -- "
              f"that is information, not a reason to drop a config\").")

    # ------------------------------------------------------------- Step B
    by_cfg = step_b(df_full, live_configs)
    n_configs_evaluated += len(live_configs) * 2 * 2  # configs x markets x slices

    hr("STEP B surface -- the full swept grid, so the SHAPE is visible, not "
       "just the winner")
    print(f"\n{'delta':>10s} {'train/spot':>11s} {'train/fut5x':>12s} "
          f"{'val/spot':>10s} {'val/fut5x':>10s}   (paired dlogGrowth vs v4)")
    print("-" * 68)
    sel_val_fut: dict[float, float] = {}
    for d in live_configs:
        rows = by_cfg[d]
        c_ts = cell(rows, "inner_train", SPOT.name)["d_loggrowth"]
        c_tf = cell(rows, "inner_train", FUTURES.name)["d_loggrowth"]
        c_vs = cell(rows, "inner_val", SPOT.name)["d_loggrowth"]
        c_vf = cell(rows, "inner_val", FUTURES.name)["d_loggrowth"]
        sel_val_fut[d] = c_vf
        print(f"{d:>10.2f} {c_ts:>11.3f} {c_tf:>12.3f} {c_vs:>10.3f} {c_vf:>10.3f}")

    # ------------------------------------------------------- the finalist
    best_d = max(live_configs, key=lambda d: sel_val_fut[d])
    b_lbl = label_of(best_d)
    b_rows = by_cfg[best_d]

    hr("STEP C -- the pre-registered promotion bar (default REJECT)")
    print(f"\nFINALIST by the frozen selection statistic (inner-validation paired "
          f"dlogGrowth vs v4 on futures_5x):\n    {b_lbl}   selection statistic = "
          f"{sel_val_fut[best_d]:+.3f} log units")
    print("\n    All four cells of the finalist:")
    print_rows(b_rows)

    print("\nRISK MATCH (exposure_ratio / vol_ratio, candidate / v4, from compare()):")
    for r in b_rows:
        print(f"    {r['slice']:11s} {r['market']:11s} expR={r['exposure_ratio']:.3f} "
              f"volR={r['vol_ratio']:.3f} risk_matched={r['risk_matched']}")

    # ---- B1 ---------------------------------------------------------------
    pts = [r["d_loggrowth"] for r in b_rows]
    excl = [r["excludes_zero"] for r in b_rows]
    b1_pos_all = all(p > 0 for p in pts)
    b1_excl_any = any(excl)
    b1 = bool(b1_pos_all and b1_excl_any)
    print("\n--- B1  paired bootstrap excludes zero in >=1 of four cells AND "
          "point estimate positive in all four")
    for r in b_rows:
        print(f"      {r['slice']:11s} {r['market']:11s} dlogG={r['d_loggrowth']:+7.3f} "
              f"[{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]  excludes_zero="
              f"{'YES' if r['excludes_zero'] else 'no'}")
    print(f"      point estimate positive in all four: {b1_pos_all}; "
          f"excludes zero in >=1: {b1_excl_any}")
    print(f"    B1: {'PASS' if b1 else 'FAIL'}")

    # ---- B2 -----------------------------------------------------------------
    v_f = cell(b_rows, "inner_val", FUTURES.name)
    v_s = cell(b_rows, "inner_val", SPOT.name)
    sharpe_leg = (v_f["d_sharpe"] > SHARPE_FLOOR) and (v_s["d_sharpe"] > SHARPE_FLOOR)
    dd_leg = (v_f["d_dd"] < 0 and v_f["risk_matched"] and
              v_s["d_dd"] < 0 and v_s["risk_matched"])
    b2 = bool(sharpe_leg or dd_leg)
    print("\n--- B2  dSharpe > +0.2 on inner-validation on BOTH markets, OR a "
          "max-drawdown improvement on BOTH where risk_matched is True for both")
    print(f"      inner_val futures_5x: dSharpe={v_f['d_sharpe']:+.3f}  "
          f"dMaxDD={v_f['d_dd']:+.2f}pp  risk_matched={v_f['risk_matched']}  "
          f"(cand {v_f['cand_dd']:.1f}% vs v4 {v_f['ctrl_dd']:.1f}%)")
    print(f"      inner_val spot      : dSharpe={v_s['d_sharpe']:+.3f}  "
          f"dMaxDD={v_s['d_dd']:+.2f}pp  risk_matched={v_s['risk_matched']}  "
          f"(cand {v_s['cand_dd']:.1f}% vs v4 {v_s['ctrl_dd']:.1f}%)")
    print(f"      Sharpe leg: {sharpe_leg};  risk-matched drawdown leg: {dd_leg}")
    print(f"    B2: {'PASS' if b2 else 'FAIL'}")

    # ---- B3 -------------------------------------------------------------
    print("\n--- B3  plateau not peak: the finalist's immediate swept-grid "
          "neighbours (inner-validation futures d_loggrowth)")
    idx = DELTA_GRID.index(best_d)
    neigh = []
    for off, tag in ((-1, "lower"), (1, "higher")):
        j = idx + off
        if 0 <= j < len(DELTA_GRID):
            d_n = DELTA_GRID[j]
            neigh.append((tag, d_n, sel_val_fut.get(d_n)))
        else:
            neigh.append((tag, None, None))
    print(f"      finalist               {b_lbl:>12s}  selection = "
          f"{sel_val_fut[best_d]:+.3f}")
    for tag, d_n, v in neigh:
        if d_n is None:
            print(f"      neighbour ({tag:>6s})    (off grid)")
        else:
            print(f"      neighbour ({tag:>6s})    {label_of(d_n):>12s}  "
                  f"selection = {v:+.3f}   (drop from finalist: "
                  f"{v - sel_val_fut[best_d]:+.3f})")
    have = [v for _, _, v in neigh if v is not None]
    if have and sel_val_fut[best_d] != 0:
        together = all(np.sign(v) == np.sign(sel_val_fut[best_d]) and
                       abs(v - sel_val_fut[best_d]) <= 0.5 * abs(sel_val_fut[best_d])
                       for v in have)
    else:
        together = False
    b3 = bool(together)
    print(f"      neighbours move WITH the finalist (same sign, within half its "
          f"own magnitude): {together}")
    print(f"    B3: {'PASS' if b3 else 'FAIL'}  (reported in full per the frozen "
          f"rule; the finalist's neighbourhood shape, not a boolean gate that "
          f"alone flips the verdict)")

    # ---- B4  ETH replication --------------------------------------------
    print("\n--- B4  falsification: ETH replication (Bitfinex ETH, pre-2023)")
    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    print(f"      ETH frame: {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}")
    print("      NOTE: this ETH series ends 2019-12-31, so inner-validation is "
          "EMPTY on ETH.\n      The replication runs on inner-train only -- a "
          "2017-2019 sign check, not a like-for-like\n      repeat of the "
          "selection slice. Stated plainly, not papered over.")
    eth_rows = compare(make_builder(best_d), eth, label=f"ETH_{b_lbl}",
                       markets=(SPOT, FUTURES), slice_names=("inner_train",))
    n_configs_evaluated += 1 * 2 * 1  # finalist x markets x inner_train
    print()
    print_rows(eth_rows)
    btc_train_pts = {m.name: cell(b_rows, "inner_train", m.name)["d_loggrowth"]
                     for m in (SPOT, FUTURES)}
    eth_pts = {r["market"]: r["d_loggrowth"] for r in eth_rows}
    same_sign = {m: bool(np.sign(eth_pts[m]) == np.sign(btc_train_pts[m])
                        and eth_pts[m] != 0)
                 for m in eth_pts}
    b4 = bool(eth_rows) and all(same_sign.values())
    for m in eth_pts:
        print(f"      BTC inner-train {m:11s} dlogG = {btc_train_pts[m]:+7.3f}   "
              f"ETH inner-train {m:11s} dlogG = {eth_pts[m]:+7.3f}   "
              f"same sign: {same_sign[m]}")
    print(f"    B4: {'PASS' if b4 else 'FAIL'}")

    # ---- B5  0.40% taker ------------------------------------------------------
    print("\n--- B5  cost robustness: BTC inner-validation re-run at a 0.40% "
          "taker fee (via r91_shared.fee_at)")
    spot40 = fee_at(SPOT, HIGH_FEE)
    fut40 = fee_at(FUTURES, HIGH_FEE)
    print(f"      market specs (fee_at, same MarketSpec fields, fee_rate swapped "
          f"to {HIGH_FEE:.4f}):\n      {spot40}\n      {fut40}")
    rows40 = compare(make_builder(best_d), df_full, label=f"fee40_{b_lbl}",
                     markets=(spot40, fut40), slice_names=("inner_val",))
    n_configs_evaluated += 1 * 2 * 1  # finalist x fee markets x inner_val
    print()
    print_rows(rows40)
    base_spot = cell(b_rows, "inner_val", SPOT.name)
    base_fut = cell(b_rows, "inner_val", FUTURES.name)
    r40_spot = cell(rows40, "inner_val", spot40.name)
    r40_fut = cell(rows40, "inner_val", fut40.name)
    spot_keeps = bool(np.sign(r40_spot["d_loggrowth"]) == np.sign(base_spot["d_loggrowth"]))
    fut_keeps = bool(np.sign(r40_fut["d_loggrowth"]) == np.sign(base_fut["d_loggrowth"]))
    print(f"      SPOT (decisive per project convention): dlogG at 0.10% taker = "
          f"{base_spot['d_loggrowth']:+7.3f}   at 0.40% taker = "
          f"{r40_spot['d_loggrowth']:+7.3f}   sign preserved: {spot_keeps}")
    print(f"      FUTURES (reported for completeness):   dlogG at 0.05% taker = "
          f"{base_fut['d_loggrowth']:+7.3f}   at 0.40% taker = "
          f"{r40_fut['d_loggrowth']:+7.3f}   sign preserved: {fut_keeps}")
    b5 = spot_keeps  # SPOT at 0.40% taker is the decisive cell per project convention
    print(f"    B5: {'PASS' if b5 else 'FAIL'}   (decisive cell = SPOT@0.40%; "
          f"futures reported for completeness only, per project convention)")

    # ------------------------------------------------------------- verdict
    hr("VERDICT")
    a0_msg = ("PASSES" if a0["ranks_below"]
              else "FIRED -- branch is NEGATIVE by pre-registration regardless of B1-B5")
    print(f"\nA0 kill switch (this round's specific pre-registered override): {a0_msg}")
    clauses = {"B1": b1, "B2": b2, "B4": b4, "B5": b5}
    for k, v in clauses.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    print(f"    (B3 reported above as required, informational per the frozen rule, "
          f"same convention as R-90)")
    b_bar_all_pass = all(clauses.values())
    # Per pre-registration: A0 firing makes the branch NEGATIVE regardless of
    # downstream numbers. This verdict line honours that; B1-B5 are still
    # measured and reported in full above, per the R-89 Step-0 convention.
    promote = bool(a0["ranks_below"]) and b_bar_all_pass
    verdict = "CANDIDATE FOR HOLDOUT" if promote else "NEGATIVE"
    print(f"\n    VERDICT: {verdict}")
    if verdict == "NEGATIVE":
        reasons = []
        if not a0["ranks_below"]:
            reasons.append("A0 kill switch fired (pre-registered, decisive on its own)")
        failed_b = [k for k, v in clauses.items() if not v]
        if failed_b:
            reasons.append(f"B-bar clauses failed: {', '.join(failed_b)}")
        print(f"    Reason(s): {'; '.join(reasons) if reasons else '(none -- see B3 note)'}")
    print("\n    (The decision rule above is exactly the one frozen in "
          "r91_shared.py's docstring and\n    this file's own docstring before "
          "any number was read. No threshold was moved. The\n    holdout itself "
          "is NOT read or touched by this script, win or lose -- that decision "
          "belongs\n    to the operator.)")

    # ---------------------------------------------------------- bookkeeping
    hr("BOOKKEEPING")
    print(f"    Frozen grid                                     : 5 configurations "
          f"(1 identity + 4 swept)")
    print(f"    Step-A mechanism gate run on                    : 5 (identity + 4 "
          f"swept) -- A1 x1, A2 x4, A3 x5")
    print(f"    Step-B evaluated (4 cells each, all 4 swept)     : 4 configs x 2 "
          f"markets x 2 slices = 16 BTC cells")
    print(f"    Finalist ETH replication cells                  : 1 config x 2 "
          f"markets x 1 slice = 2 ETH cells")
    print(f"    Finalist 0.40% fee robustness cells              : 1 config x 2 "
          f"fee-markets x 1 slice = 2 cells")
    print(f"    TOTAL CONFIGURATIONS/CELLS EVALUATED FOR TRIALS COUNT: "
          f"16 (BTC) + 2 (ETH) + 2 (fee) = 20 measured cells, over 5 distinct "
          f"delta settings")
    print(f"\n    A1 max |diff| vs r91_shared.v4_target(): {gate['a1_max_abs']:.3e}")
    print(f"    Max timestamp read anywhere in this run  : {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")


if __name__ == "__main__":
    main()
