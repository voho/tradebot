#!/usr/bin/env python
"""R-89 CONSERVATIVE branch: split `kelly_regime_v4`'s single no-trade band
into an ASYMMETRIC entry/exit pair (`d_in` / `d_out`) on the strategy's own
latched anchor vote, and ask whether the transaction-cost literature's
predicted asymmetry is measurable on the single-asset long/flat case the
theorems were actually proved for. The full citation trail, the round's
direction, the not-a-duplicate-of reasoning and the operator's measured
power calculation live in `experiments/r89_shared.py`'s module docstring;
this file does not repeat them. It implements the frozen mechanism and
reports the frozen gates, every cell, pass or fail.

MECHANISM (one sentence): for each of v4's three anchors (20/40/80-day
rolling means of close), latch that anchor LONG when price sits above
`anchor * (1 + d_in)` and FLAT when it sits below `anchor * (1 - d_out)`,
holding the previous verdict in between -- so the distance price must
travel to ENTER a long is decoupled from the distance it must travel to
EXIT one -- with every other part of v4 held byte-for-byte fixed:
        target = apply_deadband(latched_vote(df, d_in, d_out) * v4_scale(df))
`v4_scale` (conditional vol targeting, target_vol 0.55, max_leverage 2.0),
`apply_deadband` (10% re-target deadband), the anchors and the horizons are
untouched, and at `d_in == d_out == 0.01` this is `kelly_regime_v4`'s own
prepared target bit-for-bit (asserted, gate A1).

WHY IT SHOULD MAKE MONEY: Dai, Zhang & Zhu (2010, SIAM J. Financial
Mathematics 1(1):780-810) and Guan, Peng & Xu (2020, arXiv:2008.07082 Thm
3.1) prove that under proportional costs with a persistent hidden state
the optimal buy boundary sits strictly ABOVE and the sell boundary
strictly BELOW the frictionless indifference point, the gap being opened
by the fee. v4's band is a single symmetric 1% that has never been swept,
decomposed or replaced in 87 rounds; if the theorems bite here, the swept
optimum should sit OFF the diagonal, with `d_in > d_out`.

THE NAMED COUNTER-PREDICTION, frozen before any number was read: de
Lataillade, Deremble, Potters & Bouchaud (2012, J. Investment Strategies
1(3):91-115) Sec. 6.3 states that the leading-order optimal band is
SYMMETRIC, any asymmetry being higher order in Gamma^(1/3). If the swept
optimum lands on or adjacent to the diagonal (d_in ~ d_out), that is this
counter-prediction being confirmed and this branch is a NEGATIVE -- a
clean, diagnosed negative, which is this branch's expected and successful
outcome, not an inconclusive one.

FROZEN GRID -- 25 configurations, no config added or dropped after results:
        d_in  in {0.005, 0.01, 0.02, 0.03, 0.05}
        d_out in {0.005, 0.01, 0.02, 0.03, 0.05}
all 25 combinations. The five diagonal cells are the symmetric controls
(including the identity point d_in = d_out = 0.01 = v4 itself, plus four
symmetric-but-wider/narrower bands); reporting them as their own row set is
what makes an ASYMMETRY claim separable from a pure WIDTH claim.

CONFIGURATIONS EVALUATED IN THIS FILE: 25 -- the frozen grid, all of which
pass through the Step-A mechanism gate; those that clear the A2
non-inertness bar (R^2 < 0.98 against v4's own exposure path on
inner-train) go on to Step B. No exploratory configuration outside the
frozen 25 is evaluated anywhere in this file; the exact counts are printed
at the end of `main()`.

FROZEN DECISION RULE (default REJECT; every clause reported PASS or FAIL,
no threshold moved after seeing any result):
  B1  the paired block-bootstrap difference in log growth vs v4 excludes
      zero in >= 1 of the four (slice x market) cells AND the point
      estimate is positive in all four.
  B2  EITHER dSharpe > +0.2 on inner-validation on BOTH markets, OR a
      clear max-drawdown improvement on both.
  B3  plateau not peak: the finalist's immediate neighbours on both axes
      must move with it, not away from it.
  B4  falsification test -- ETH replication (Bitfinex ETH, pre-2023):
      the frozen finalist must show the SAME SIGN of improvement against
      the same v4 control on both markets. Failing this is a NEGATIVE.
  B5  cost robustness -- re-run finalist and control on inner-validation
      at a 0.40% taker fee; the improvement must not reverse sign.
Selection statistic (frozen): the inner-validation PAIRED log-growth
difference vs v4 on `futures_5x`. Measured power, from `r89_shared`: a 95%
paired interval at the 30-day block convention excludes zero once the
candidate beats v4 by about +0.35 log units over inner-train or +0.13 to
+0.26 over inner-validation; every difference below is reported against
those numbers.

This file never reads a bar at or after OOS_START (2023-01-01): every load
goes through `r89_shared`'s truncating, asserting loaders, and the max
timestamp actually read is tracked and printed at the end of `main()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402

from experiments.r89_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    OOS_START,
    SPOT,
    V4_BAND,
    apply_deadband,
    causal_truncation_probe,
    compare,
    latched_vote,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    v4_scale,
    v4_vote_frac,
)

# ---------------------------------------------------------------- frozen grid
D_IN_GRID = (0.005, 0.01, 0.02, 0.03, 0.05)
D_OUT_GRID = (0.005, 0.01, 0.02, 0.03, 0.05)
GRID = [(di, do) for di in D_IN_GRID for do in D_OUT_GRID]   # 25, row-major
INERT_R2 = 0.98            # frozen A2 bar
SHARPE_FLOOR = 0.2         # frozen B2 bar (the project's noise floor, R-20)
HIGH_FEE = 0.004           # frozen B5 taker fee: 0.40%

# Operator-measured power, quoted so every difference can be read against it.
POWER_TRAIN = 0.35
POWER_VAL_LO, POWER_VAL_HI = 0.13, 0.26


def label_of(d_in: float, d_out: float) -> str:
    return f"in{d_in:g}_out{d_out:g}"


def build_target(df: pd.DataFrame, d_in: float, d_out: float) -> np.ndarray:
    """The frozen mechanism. Pure function of the bars it is handed."""
    return apply_deadband(latched_vote(df, d_in, d_out) * v4_scale(df))


def v4_control_target(df: pd.DataFrame) -> np.ndarray:
    """`kelly_regime_v4`'s own target, rebuilt from v4's own factors."""
    return apply_deadband(v4_vote_frac(df) * v4_scale(df))


def make_builder(d_in: float, d_out: float):
    return lambda d: build_target(d, d_in, d_out)


# ------------------------------------------------------------------ printing

def print_matrix(title: str, cells: dict, fmt: str = "{:+7.3f}",
                 note: str = "") -> None:
    print(f"\n{title}")
    if note:
        print(f"  ({note})")
    head = "  d_in \\ d_out " + "".join(f"{do:>10g}" for do in D_OUT_GRID)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for di in D_IN_GRID:
        row = f"  {di:>12g} "
        for do in D_OUT_GRID:
            v = cells.get((di, do))
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                row += f"{'--':>10s}"
            else:
                row += f"{fmt.format(v):>10s}"
        print(row)
    print("  (diagonal = symmetric controls: (0.005,0.005) (0.01,0.01)=v4 "
          "(0.02,0.02) (0.03,0.03) (0.05,0.05))")


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# --------------------------------------------------------------- Step A gate

def step_a(df_full: pd.DataFrame, df_train: pd.DataFrame) -> tuple[list[dict], float]:
    """Mechanism gate, all 25 configs, BEFORE any performance number is read."""
    hr("STEP A -- mechanism gate (all 25 frozen configs), before any "
       "performance number")

    # ---- A1: identity, bit-for-bit against kelly_regime_v4's own prepare().
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    mine = build_target(df_train, V4_BAND, V4_BAND)
    theirs = KellyRegimeV4().prepare(df_train.copy())["target"].to_numpy()
    max_abs = float(np.max(np.abs(mine - theirs)))
    a1_pass = bool(np.array_equal(mine, theirs))
    print(f"\nA1 identity (d_in = d_out = {V4_BAND}) vs kelly_regime_v4.prepare()"
          f" on the real BTC inner-train frame ({len(df_train):,} bars):")
    print(f"    max |candidate.target - v4.target| = {max_abs:.3e}   "
          f"exact array equality = {a1_pass}   -> {'PASS' if a1_pass else 'FAIL'}")
    if not a1_pass:
        raise AssertionError("A1 identity FAILED -- the mechanism is not a "
                             "superset of v4; nothing below would be interpretable.")

    # v4's own path on inner-train, the A2 reference.
    ctrl_train = v4_control_target(df_train)

    # ---- A2: non-inertness, R^2 of the candidate exposure path vs v4's.
    rows: list[dict] = []
    r2_cells: dict = {}
    for d_in, d_out in GRID:
        path = build_target(df_train, d_in, d_out)
        r2 = r_squared(path, ctrl_train)
        inert = bool(np.isfinite(r2) and r2 >= INERT_R2)
        rows.append(dict(d_in=d_in, d_out=d_out, label=label_of(d_in, d_out),
                         r2_train=r2, inert=inert,
                         mean_exposure=float(np.mean(path[80 * 288:])),
                         diagonal=(d_in == d_out)))
        r2_cells[(d_in, d_out)] = r2

    print(f"\nA2 non-inertness -- R^2(candidate exposure path, v4 exposure path) "
          f"on inner-train; R^2 >= {INERT_R2} is INERT and is excluded from "
          f"Step B selection.")
    print(f"\n{'#':>3s} {'config':>16s} {'d_in':>7s} {'d_out':>7s} {'diag':>5s} "
          f"{'R2_train':>9s} {'mean_exp':>9s} {'A2':>7s}")
    print("-" * 72)
    for i, r in enumerate(rows, 1):
        print(f"{i:>3d} {r['label']:>16s} {r['d_in']:>7g} {r['d_out']:>7g} "
              f"{'YES' if r['diagonal'] else '-':>5s} {r['r2_train']:>9.4f} "
              f"{r['mean_exposure']:>9.3f} "
              f"{'INERT' if r['inert'] else 'pass':>7s}")

    n_inert = sum(r["inert"] for r in rows)
    print(f"\n    {n_inert} of 25 configurations are INERT (R^2 >= {INERT_R2}) "
          f"and are EXCLUDED from Step B selection: "
          f"{', '.join(r['label'] for r in rows if r['inert']) or '(none)'}")
    print(f"    {25 - n_inert} of 25 proceed to Step B.")

    print_matrix("A2 surface -- R^2 of the candidate exposure path against v4's, "
                 "inner-train", r2_cells, fmt="{:7.4f}")

    # ---- A3: causality probe on the identity config (and, later, the finalist).
    print("\nA3 causality -- causal_truncation_probe on the full pre-holdout BTC "
          "frame\n    (rebuild the target on truncated frames at 55% and 80%; the "
          "surviving prefix must match bit-for-bit).")
    ok_id = causal_truncation_probe(make_builder(V4_BAND, V4_BAND), df_full)
    print(f"    identity config {label_of(V4_BAND, V4_BAND):>16s} : "
          f"{'PASS' if ok_id else 'FAIL'}   "
          f"(frame {len(df_full):,} bars, max ts {df_full.index[-1]})")

    return rows, max_abs


# ---------------------------------------------------------------- Step B eval

def step_b(df_full: pd.DataFrame, gate_rows: list[dict]) -> tuple[list[dict], dict]:
    hr("STEP B -- evaluation: every non-inert config, all four (slice x market) "
       "cells vs v4")

    live = [r for r in gate_rows if not r["inert"]]
    print(f"\nEvaluating {len(live)} configurations x 4 cells "
          f"(inner_train/inner_val x spot/futures_5x), candidate and control, "
          f"paired block bootstrap (30-day blocks, 2000 draws).")
    print("A negative dlogG means the asymmetric band LOSES to v4 in that cell.\n")

    all_rows: list[dict] = []
    by_cfg: dict = {}
    for k, r in enumerate(live, 1):
        d_in, d_out = r["d_in"], r["d_out"]
        lbl = r["label"]
        rows = compare(make_builder(d_in, d_out), df_full, label=lbl)
        for row in rows:
            row["d_in"], row["d_out"] = d_in, d_out
        all_rows.extend(rows)
        by_cfg[(d_in, d_out)] = rows
        print(f"--- [{k}/{len(live)}] {lbl}  (d_in={d_in:g}, d_out={d_out:g}"
              f"{', DIAGONAL / symmetric control' if d_in == d_out else ''})")
        print_rows(rows)
        print()

    return all_rows, by_cfg


def cell(rows: list[dict], slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["slice"] == slice_name and r["market"] == market:
            return r
    return None


# ------------------------------------------------------------------- runner

def main() -> None:
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-89 CONSERVATIVE -- asymmetric entry/exit band (d_in / d_out) on "
       "kelly_regime_v4's\nown latched anchor vote. 25 frozen configurations. "
       "Default verdict: REJECT.")
    df_full = load_btc()
    max_ts_seen.append(df_full.index.max())
    df_train = df_full.loc[:INNER_TRAIN_END]
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(df_full):,} bars, "
          f"{df_full.index[0]} -> {df_full.index[-1]}")
    print(f"inner-train frame: {len(df_train):,} bars, {df_train.index[0]} -> "
          f"{df_train.index[-1]}")

    # ------------------------------------------------------------- Step A
    gate_rows, a1_max_abs = step_a(df_full, df_train)

    # ------------------------------------------------------------- Step B
    all_rows, by_cfg = step_b(df_full, gate_rows)

    hr("STEP B surfaces -- the whole 5x5, so the SHAPE is visible, not just "
       "the winner")

    sel_cells = {}
    dsh_val_fut, dsh_val_spot, dsh_train_fut = {}, {}, {}
    dd_val_fut = {}
    for (d_in, d_out), rows in by_cfg.items():
        c = cell(rows, "inner_val", FUTURES.name)
        sel_cells[(d_in, d_out)] = c["d_loggrowth"] if c else None
        dsh_val_fut[(d_in, d_out)] = c["d_sharpe"] if c else None
        dd_val_fut[(d_in, d_out)] = c["d_dd"] if c else None
        c2 = cell(rows, "inner_val", SPOT.name)
        dsh_val_spot[(d_in, d_out)] = c2["d_sharpe"] if c2 else None
        c3 = cell(rows, "inner_train", FUTURES.name)
        dsh_train_fut[(d_in, d_out)] = c3["d_loggrowth"] if c3 else None

    print_matrix(
        "SELECTION STATISTIC -- paired dlogGrowth vs v4, inner-validation, "
        "futures_5x", sel_cells,
        note=f"the interval bar needs about +{POWER_VAL_LO} to +{POWER_VAL_HI} "
             f"here; '--' = INERT, excluded")
    print_matrix(
        "paired dlogGrowth vs v4, inner-TRAIN, futures_5x", dsh_train_fut,
        note=f"the interval bar needs about +{POWER_TRAIN} here")
    print_matrix("dSharpe vs v4, inner-validation, futures_5x", dsh_val_fut,
                 fmt="{:+7.2f}", note=f"B2 needs > +{SHARPE_FLOOR} on BOTH markets")
    print_matrix("dSharpe vs v4, inner-validation, spot", dsh_val_spot,
                 fmt="{:+7.2f}", note=f"B2 needs > +{SHARPE_FLOOR} on BOTH markets")
    print_matrix("dMaxDrawdown vs v4 (percentage points; negative = shallower "
                 "than v4), inner-validation, futures_5x", dd_val_fut,
                 fmt="{:+7.1f}")

    # ---- the diagonal, reported explicitly as its own row set ------------
    print("\nTHE DIAGONAL (symmetric controls) -- reported as its own row set, "
          "which is\nwhat makes an ASYMMETRY claim separable from a pure WIDTH "
          "claim:")
    print(f"  {'config':>16s} {'d=d_in=d_out':>13s} {'sel(dlogG val fut)':>19s} "
          f"{'dSh val fut':>12s} {'dlogG train fut':>16s}")
    print("  " + "-" * 80)
    for d in D_IN_GRID:
        k = (d, d)
        if k in by_cfg:
            print(f"  {label_of(d, d):>16s} {d:>13g} {sel_cells[k]:>19.3f} "
                  f"{dsh_val_fut[k]:>12.2f} {dsh_train_fut[k]:>16.3f}")
        else:
            print(f"  {label_of(d, d):>16s} {d:>13g} {'INERT (= v4 itself)':>19s} "
                  f"{'--':>12s} {'--':>16s}")

    # ------------------------------------------------------- the finalist
    live_keys = [k for k in by_cfg]
    best = max(live_keys, key=lambda k: sel_cells[k])
    b_in, b_out = best
    b_lbl = label_of(b_in, b_out)
    b_rows = by_cfg[best]

    hr("STEP C -- the pre-registered promotion bar (default REJECT)")
    print(f"\nFINALIST by the frozen selection statistic (inner-validation paired "
          f"dlogGrowth vs v4 on futures_5x):\n    {b_lbl}   d_in={b_in:g}, "
          f"d_out={b_out:g}   selection statistic = {sel_cells[best]:+.3f} log "
          f"units")
    print(f"    (bar for the interval to exclude zero on inner-validation: about "
          f"+{POWER_VAL_LO} to +{POWER_VAL_HI}; this is "
          f"{sel_cells[best]:+.3f})")
    print(f"    d_in {'>' if b_in > b_out else ('<' if b_in < b_out else '==')} "
          f"d_out -> the finalist is "
          f"{'OFF-diagonal (asymmetric)' if b_in != b_out else 'ON the diagonal (SYMMETRIC)'}")
    print("\n    All four cells of the finalist:")
    print_rows(b_rows)

    # A3 on the finalist ---------------------------------------------------
    ok_fin = causal_truncation_probe(make_builder(b_in, b_out), df_full)
    print(f"\nA3 causality re-run on the FINALIST {b_lbl}: "
          f"{'PASS' if ok_fin else 'FAIL'}")

    # exposure matching ----------------------------------------------------
    warm = 80 * 288
    cand_path = build_target(df_full, b_in, b_out)[warm:]
    ctrl_path = v4_control_target(df_full)[warm:]
    me_c, me_v = float(np.mean(cand_path)), float(np.mean(ctrl_path))
    print(f"\nRISK MATCH (mean target exposure, fraction of equity, post-warmup, "
          f"2017-2022):\n    finalist {me_c:.4f}   v4 {me_v:.4f}   "
          f"ratio {me_c / me_v:.4f}   difference {me_c - me_v:+.4f}")
    if abs(me_c / me_v - 1.0) > 0.05:
        print("    -> mean exposure differs by more than 5%: any drawdown "
              "difference below is partly arithmetic, NOT evidence.")
    else:
        print("    -> mean exposure is matched to within 5%; the cells are "
              "comparable on exposure.")

    # ---- B1 --------------------------------------------------------------
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

    # ---- B2 --------------------------------------------------------------
    v_f = cell(b_rows, "inner_val", FUTURES.name)
    v_s = cell(b_rows, "inner_val", SPOT.name)
    sharpe_leg = (v_f["d_sharpe"] > SHARPE_FLOOR) and (v_s["d_sharpe"] > SHARPE_FLOOR)
    dd_leg = (v_f["d_dd"] < 0) and (v_s["d_dd"] < 0)
    b2 = bool(sharpe_leg or dd_leg)
    print("\n--- B2  dSharpe > +0.2 on inner-validation on BOTH markets, OR a "
          "clear max-drawdown improvement on both")
    print(f"      inner_val futures_5x: dSharpe={v_f['d_sharpe']:+.3f}  "
          f"dMaxDD={v_f['d_dd']:+.2f}pp  (cand {v_f['cand_dd']:.1f}% vs v4 "
          f"{v_f['ctrl_dd']:.1f}%)")
    print(f"      inner_val spot      : dSharpe={v_s['d_sharpe']:+.3f}  "
          f"dMaxDD={v_s['d_dd']:+.2f}pp  (cand {v_s['cand_dd']:.1f}% vs v4 "
          f"{v_s['ctrl_dd']:.1f}%)")
    print(f"      Sharpe leg: {sharpe_leg};  drawdown leg: {dd_leg}")
    print(f"    B2: {'PASS' if b2 else 'FAIL'}")

    # ---- B3 --------------------------------------------------------------
    print("\n--- B3  plateau not peak: the finalist's immediate neighbours on "
          "both axes")
    i_in, i_out = D_IN_GRID.index(b_in), D_OUT_GRID.index(b_out)
    neigh = []
    for di_off, do_off, axis in ((-1, 0, "d_in-"), (1, 0, "d_in+"),
                                 (0, -1, "d_out-"), (0, 1, "d_out+")):
        j_in, j_out = i_in + di_off, i_out + do_off
        if 0 <= j_in < len(D_IN_GRID) and 0 <= j_out < len(D_OUT_GRID):
            k = (D_IN_GRID[j_in], D_OUT_GRID[j_out])
            v = sel_cells.get(k)
            neigh.append((axis, label_of(*k), v))
        else:
            neigh.append((axis, "(off grid)", None))
    print(f"      finalist            {b_lbl:>16s}  selection = "
          f"{sel_cells[best]:+.3f}")
    for axis, lbl, v in neigh:
        if v is None:
            print(f"      neighbour {axis:>7s}   {lbl:>16s}  selection = "
                  f"{'INERT / off grid':>16s}")
        else:
            print(f"      neighbour {axis:>7s}   {lbl:>16s}  selection = "
                  f"{v:+.3f}   (drop from finalist: {v - sel_cells[best]:+.3f})")
    have = [v for _, _, v in neigh if v is not None]
    # "move together": every available neighbour keeps the finalist's sign and
    # stays within half the finalist's own magnitude of it.
    if have and sel_cells[best] != 0:
        together = all(np.sign(v) == np.sign(sel_cells[best]) and
                       abs(v - sel_cells[best]) <= 0.5 * abs(sel_cells[best])
                       for v in have)
    else:
        together = False
    b3 = bool(together)
    print(f"      neighbours move WITH the finalist (same sign, within half its "
          f"own magnitude): {together}")
    print(f"    B3: {'PASS' if b3 else 'FAIL'}")

    # ---- B4  ETH -----------------------------------------------------------
    print("\n--- B4  falsification test: ETH replication (Bitfinex ETH, "
          "pre-2023)")
    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    print(f"      ETH frame: {len(eth):,} bars, {eth.index[0]} -> "
          f"{eth.index[-1]}")
    eth_slices = []
    for s_name, (s0, s1) in (("inner_train", ("2017-01-01", "2020-12-31")),
                             ("inner_val", ("2021-01-01", "2022-12-31"))):
        n = int(((eth.index >= pd.Timestamp(s0, tz="UTC")) &
                 (eth.index <= pd.Timestamp(s1, tz="UTC"))).sum())
        print(f"      ETH bars inside {s_name}: {n:,}")
        if n > 0:
            eth_slices.append(s_name)
    if "inner_val" not in eth_slices:
        print("      NOTE: this ETH series ends 2019-12-31, so the "
              "inner-VALIDATION slice is empty on ETH.\n"
              "      The replication therefore runs on inner-train only. "
              "Reported as a limitation, not\n      papered over: the ETH leg "
              "is a 2017-2019 sign check, not a like-for-like repeat of the\n"
              "      selection slice.")
    eth_rows = compare(make_builder(b_in, b_out), eth, label=f"ETH_{b_lbl}",
                       slice_names=tuple(eth_slices))
    print()
    print_rows(eth_rows)
    eth_pts = {(r["market"]): r["d_loggrowth"] for r in eth_rows}
    btc_sel_sign = np.sign(sel_cells[best])
    same_sign = {m: bool(np.sign(v) == btc_sel_sign and v != 0)
                 for m, v in eth_pts.items()}
    b4 = bool(eth_rows) and all(same_sign.values())
    print(f"      BTC improvement sign (selection statistic) = "
          f"{'+' if btc_sel_sign > 0 else '-'}")
    for m, v in eth_pts.items():
        print(f"      ETH {m:11s} dlogG = {v:+7.3f}  -> same sign as BTC: "
              f"{same_sign[m]}")
    print(f"    B4: {'PASS' if b4 else 'FAIL'}"
          f"{'  (a FAIL here is a NEGATIVE by pre-registration)' if not b4 else ''}")

    # ---- B5  0.40% taker ----------------------------------------------------
    print("\n--- B5  cost robustness: inner-validation re-run at a 0.40% taker "
          "fee")
    spot40 = MarketSpec(name="spot_40bp", leverage=1.0, fee_rate=HIGH_FEE,
                        allow_short=False)
    fut40 = MarketSpec(name="futures_5x_40bp", leverage=5.0, fee_rate=HIGH_FEE,
                       allow_short=True, pays_funding=True)
    print(f"      market specs built here: {spot40}\n      {fut40}")
    rows40 = compare(make_builder(b_in, b_out), df_full, label=f"fee40_{b_lbl}",
                     markets=(spot40, fut40), slice_names=("inner_val",))
    print()
    print_rows(rows40)
    b5_ok = True
    for r in rows40:
        base_m = SPOT.name if r["market"] == spot40.name else FUTURES.name
        base = cell(b_rows, "inner_val", base_m)
        keeps = bool(np.sign(r["d_loggrowth"]) == np.sign(base["d_loggrowth"]))
        b5_ok &= keeps
        print(f"      {r['market']:16s} dlogG at 0.10%/0.05% = "
              f"{base['d_loggrowth']:+7.3f}   at 0.40% = "
              f"{r['d_loggrowth']:+7.3f}   sign preserved: {keeps}")
    b5 = bool(b5_ok)
    print(f"    B5: {'PASS' if b5 else 'FAIL'}   (this clause asks only that "
          f"the improvement does not REVERSE sign; on a finalist whose\n"
          f"          improvement is already negative, 'sign preserved' means "
          f"the loss persists.)")

    # ------------------------------------------------------------- verdict
    hr("VERDICT")
    clauses = {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5}
    for k, v in clauses.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    promote = all(clauses.values())

    on_or_adjacent = (b_in == b_out) or (abs(D_IN_GRID.index(b_in) -
                                             D_OUT_GRID.index(b_out)) <= 1)
    print(f"\n    Finalist {b_lbl}: "
          f"{'ON the diagonal' if b_in == b_out else 'off the diagonal'}; "
          f"on-or-adjacent-to-diagonal = {on_or_adjacent}")
    print("    Named counter-prediction (de Lataillade, Deremble, Potters & "
          "Bouchaud 2012 Sec. 6.3:\n    the leading-order band is SYMMETRIC, "
          "asymmetry is higher order in Gamma^(1/3)) --")
    if on_or_adjacent:
        print("    the swept optimum sits on or adjacent to the diagonal, "
              "which is the counter-prediction\n    CONFIRMED. By "
              "pre-registration this branch is a NEGATIVE on that ground "
              "alone.")
    else:
        print("    the swept optimum sits away from the diagonal, which is the "
              "counter-prediction\n    CONTRADICTED -- the Dai-Zhang-Zhu / "
              "Guan-Peng-Xu asymmetry would be the live reading.")

    verdict = "PROMOTE" if (promote and not on_or_adjacent) else "NEGATIVE"
    print(f"\n    VERDICT: {verdict}")
    if verdict == "NEGATIVE":
        failed = [k for k, v in clauses.items() if not v]
        print(f"    Clauses failed: {', '.join(failed) if failed else '(none)'}"
              f"{'; plus the counter-prediction confirmation above' if on_or_adjacent else ''}")
    print("\n    (The decision rule above is exactly the one frozen in this "
          "file's docstring before\n    any number was read. No threshold was "
          "moved.)")

    # ---------------------------------------------------------- bookkeeping
    hr("BOOKKEEPING")
    n_inert = sum(r["inert"] for r in gate_rows)
    print(f"    Frozen grid                              : 25 configurations")
    print(f"    Step-A mechanism gate run on             : {len(gate_rows)}")
    print(f"    INERT (R^2 >= {INERT_R2}), excluded from Step B: {n_inert}")
    print(f"    Step-B evaluated (4 cells each)          : {len(by_cfg)}")
    print(f"    Exploratory configs outside the grid     : 0")
    print(f"    TOTAL CONFIGURATIONS EVALUATED           : 25")
    print(f"    (the finalist's ETH and 0.40%-fee re-runs are re-runs of an "
          f"already-counted\n     configuration, not new configurations)")
    print(f"\n    A1 max |diff| vs kelly_regime_v4.prepare(): {a1_max_abs:.3e}")
    print(f"    Max timestamp read anywhere in this run  : {max(max_ts_seen)}"
          f"   (OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")


if __name__ == "__main__":
    main()
