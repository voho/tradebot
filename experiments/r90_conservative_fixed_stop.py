#!/usr/bin/env python
"""R-90 CONSERVATIVE branch: overlay a LITERAL, fixed-percentage trailing
stop (Han, Zhou & Zhu 2016's own construction) on `kelly_regime_v4`'s raw
desired-exposure path, with INSTANT, UNCONDITIONAL restart -- deliberately
the "naive" case Hsieh (2023, arXiv:2303.02613) motivates a restart
mechanism to fix. The full citation trail, the round's direction, the
not-a-duplicate-of reasoning, the shared mechanism (`apply_trailing_stop`)
and the shared risk-match / whipsaw diagnostics all live in
`experiments/r90_shared.py`'s module docstring; this file does not repeat
them and does not edit that file. It implements the frozen mechanism and
reports the frozen gates, every cell, pass or fail, exactly as
pre-registered below.

MECHANISM (one sentence): compute v4's own pre-deadband raw desired
exposure (`v4_raw_desired = frac * scale`, byte-for-byte v4's own
factors), overlay a trailing stop that forces the position flat the
instant `close` falls more than `stop_frac` below the running peak close
of the current long trade, and re-arm the stop UNCONDITIONALLY on the very
next bar (no cooldown, no reclaim gate) so the vote alone re-establishes
exposure once it wants to --

        target = apply_deadband(
            apply_trailing_stop(df, v4_raw_desired(df), stop_frac).target)

with `apply_trailing_stop` and `apply_deadband` both the shared, read-only
implementations in `r90_shared.py`.

WHY IT SHOULD MAKE MONEY: Han, Zhou & Zhu (2016, SSRN 2407199) find that a
literal fixed-percentage stop on US equity momentum, 1926-2013, cuts the
worst month roughly in half and more than doubles Sharpe -- a pure
tail-truncation effect, with no claim that the *average* trade improves.
If that generalises to a single-instrument, vol-targeted, latched-vote
long/flat system on BTC, the effect should show up as a Sharpe or
drawdown improvement concentrated in exactly the episodes where v4 rides
an anchor-vote reversal all the way down before the vote itself flips.

THE NAMED COUNTER-PREDICTION / risk, frozen before any number was read
(B-41's own backlog filing, restated in `r90_shared.py`): on BTC, trailing
stops fire on the ROUTINE 10-20% intra-trend pullbacks that punctuate
every bull run, and unconditional-instant-restart re-entry happens higher
than the exit -- the classic whipsaw, expensive at 10-20bps per round
trip, repeated every time a trend resumes after a routine pullback. This
is exactly the failure mode Hsieh's paper's restart mechanism exists to
prevent, and this branch deliberately does NOT implement that fix, so
that the size of the problem it fixes is measured rather than assumed.
`B4(b)` below is this counter-prediction's direct falsification test.

FROZEN GRID -- 6 configurations, none added or dropped after any result:
    identity / control check (NOT one of "the 5 evaluated configs",
    reported separately as the A1 gate): stop_frac_value = 1.0
    (unreachable -- price cannot fall 100% against a positive peak in one
    bar). Must reproduce `r90_shared.v4_target(df)` bit-for-bit.

    the swept grid, 5 configs: stop_frac_value in
        {0.08, 0.12, 0.16, 0.20, 0.25}
    bracketing the "routine 10-20% intra-trend pullback" magnitude the
    backlog item itself names as the danger zone -- chosen for that
    reason, not fitted.

FROZEN DECISION RULE (default REJECT; every clause reported PASS or FAIL,
no threshold moved after seeing any result). Full text lives in the task
brief and is reproduced here so the rule that ran is the rule printed:

  Step A (mechanism gate, before any performance number is read):
    A1 identity: stop_frac_value=1.0 reproduces v4_target(df) bit-for-bit
        on real BTC inner-train (max abs diff = 0.0).
    A2 non-inertness: each of the 5 swept configs must have
        stop_events.sum() > 0 on inner-train; a config where the stop
        never fires is reported INERT and not scored.
    A3 causality: causal_truncation_probe(build_target_fn, df,
        cuts=(0.55, 0.80)) must pass, on the real BTC inner-train frame.

  Step B (selection): compare() over slice_names=("inner_train",
    "inner_val"), markets=(SPOT, FUTURES). Selection statistic: the
    inner-validation paired log-growth difference vs v4 on futures_5x,
    among the configs that pass Step A. Full 5x4 table reported.

  Promotion bar -- ALL FIVE must hold for "CANDIDATE FOR HOLDOUT",
  else "NEGATIVE":
    B1  paired block-bootstrap difference (candidate vs v4, log growth)
        excludes zero in >= 1 of the 4 cells for the finalist, AND the
        point estimate is positive in ALL FOUR cells.
    B2  EITHER dSharpe > +0.2 on inner-validation on BOTH markets, OR a
        max-drawdown improvement on inner-validation on BOTH markets
        WHERE risk_matched is True for both cells (an improvement where
        risk_matched is False is not evidence -- the standing "held less,
        drew down less" rule).
    B3  plateau not peak: the finalist's immediate grid neighbours' inner-
        validation futures d_loggrowth reported, stated explicitly
        whether they move with the finalist or reverse sharply.
    B4  falsification, two parts, both checked and reported regardless:
        (a) ETH replication (Bitfinex ETH, pre-2023, inner-train only --
            this series ends 2019-12-31, so inner-validation is empty on
            ETH and this is said plainly, not papered over): the finalist
            must show the SAME SIGN of d_loggrowth on both ETH markets as
            it showed on BTC inner-train. Opposite sign on either market
            fails this half of B4.
        (b) whipsaw falsification: stopout_whipsaw_rate() on the
            finalist's own run, inner-train BTC futures_5x. If
            whipsaw_rate > 0.5 AND the finalist's B1 point estimate on
            that same cell (inner-train, futures) is NOT positive, this
            half of B4 fails -- the round's named risk is confirmed.
    B5  cost robustness: re-run the finalist's inner-validation cells
        (both markets) at a 0.40% taker fee
        (MarketSpec(name=..., fee_rate=0.0040, ...) built from the
        actual dataclass fields in tradebot/broker.py, checked before
        use -- no `is_futures` field exists there; futures uses
        `pays_funding=True` like the shared FUTURES spec). The sign of
        the B1 point estimate must not reverse on either market.

This file never reads a bar at or after OOS_START (2023-01-01): every
load goes through `r90_shared`'s truncating, asserting loaders, and the
max timestamp actually read is tracked and printed at the end of main().
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

from experiments.r90_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    OOS_START,
    SPOT,
    apply_deadband,
    apply_trailing_stop,
    causal_truncation_probe,
    compare,
    load_btc,
    load_eth,
    print_rows,
    stopout_whipsaw_rate,
    v4_raw_desired,
    v4_target,
)

# ---------------------------------------------------------------- frozen grid
STOP_GRID = (0.08, 0.12, 0.16, 0.20, 0.25)   # the 5 swept configs
IDENTITY_STOP = 1.0                          # the A1 identity/control point
SHARPE_FLOOR = 0.2                           # frozen B2 bar (R-20 noise floor)
HIGH_FEE = 0.0040                            # frozen B5 taker fee: 0.40%
WHIPSAW_RATE_FAIL = 0.5                      # frozen B4(b) threshold


def label_of(stop_frac_value: float) -> str:
    return f"stop{stop_frac_value:g}"


# ------------------------------------------------------------- the mechanism

def run_mechanism(df: pd.DataFrame, stop_frac_value: float):
    """Run the frozen mechanism once; return (final_target, stop_events).

    Both come from the SAME run so the whipsaw diagnostic and the B1/B2
    performance numbers below are never computed from two independent
    (and potentially inconsistent) executions.
    """
    raw = v4_raw_desired(df)
    stop_frac = np.full(len(df), float(stop_frac_value), dtype=float)
    r = apply_trailing_stop(df, raw, stop_frac,
                            reentry_delay_bars=None, reentry_reclaim=False)
    final = apply_deadband(r.target)
    return final, r.stop_events


def build_target(df: pd.DataFrame, stop_frac_value: float) -> np.ndarray:
    """Pure function of the bars it is handed -- what compare()/causal_truncation_probe call."""
    final, _ = run_mechanism(df, stop_frac_value)
    return final


def make_builder(stop_frac_value: float):
    return lambda d: build_target(d, stop_frac_value)


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


# --------------------------------------------------------------- Step A gate

def step_a(df_full: pd.DataFrame, df_train: pd.DataFrame) -> dict:
    hr("STEP A -- mechanism gate (identity + 5 swept configs), before any "
       "performance number")

    # ---- A1: identity, bit-for-bit against v4_target on real BTC inner-train.
    mine = build_target(df_train, IDENTITY_STOP)
    theirs = v4_target(df_train)
    max_abs = float(np.max(np.abs(mine - theirs)))
    a1_pass = bool(np.array_equal(mine, theirs))
    print(f"\nA1 identity (stop_frac_value={IDENTITY_STOP}) vs r90_shared.v4_target(df) "
          f"on real BTC inner-train ({len(df_train):,} bars, "
          f"{df_train.index[0]} -> {df_train.index[-1]}):")
    print(f"    max |candidate - v4_target| = {max_abs:.3e}   "
          f"exact array equality = {a1_pass}   -> {'PASS' if a1_pass else 'FAIL'}")
    if not a1_pass:
        raise AssertionError("A1 identity FAILED -- nothing below would be interpretable.")

    # ---- A2: non-inertness -- stop_events.sum() > 0 on inner-train, per swept config.
    print(f"\nA2 non-inertness -- on inner-train, each of the {len(STOP_GRID)} swept "
          f"configs must have stop_events.sum() > 0.")
    print(f"\n{'#':>3s} {'config':>10s} {'stop_frac':>10s} {'stop_events':>12s} "
          f"{'A2':>7s}")
    print("-" * 48)
    a2_rows = []
    for i, sf in enumerate(STOP_GRID, 1):
        _final, events = run_mechanism(df_train, sf)
        n_events = int(events.sum())
        inert = n_events == 0
        a2_rows.append(dict(stop_frac=sf, label=label_of(sf), n_events=n_events, inert=inert))
        print(f"{i:>3d} {label_of(sf):>10s} {sf:>10.2f} {n_events:>12d} "
              f"{'INERT' if inert else 'pass':>7s}")
    n_inert = sum(r["inert"] for r in a2_rows)
    print(f"\n    {n_inert} of {len(STOP_GRID)} swept configs are INERT and excluded "
          f"from Step B: {', '.join(r['label'] for r in a2_rows if r['inert']) or '(none)'}")

    # ---- A3: causality, identity + every swept config, on real BTC inner-train.
    print("\nA3 causality -- causal_truncation_probe on the real BTC inner-train frame "
          "(rebuild the\n    target on truncated frames at 55% and 80%; the surviving "
          "prefix must match bit-for-bit),\n    checked for the identity config and "
          "every swept config (the causal structure does not\n    depend on the "
          "scalar stop_frac_value, so this also confirms the check is not "
          "config-specific).")
    a3_results = {}
    for sf in (IDENTITY_STOP,) + STOP_GRID:
        ok = causal_truncation_probe(make_builder(sf), df_train)
        a3_results[sf] = ok
        print(f"    {label_of(sf):>10s} : {'PASS' if ok else 'FAIL'}")
    if not all(a3_results.values()):
        raise AssertionError(f"A3 causality FAILED for: "
                             f"{[sf for sf, ok in a3_results.items() if not ok]}")

    return dict(a1_max_abs=max_abs, a1_pass=a1_pass, a2_rows=a2_rows, a3_results=a3_results)


# --------------------------------------------------------------- Step B eval

def step_b(df_full: pd.DataFrame, live_configs: list[float]) -> dict:
    hr("STEP B -- evaluation: every non-inert swept config, all four "
       "(slice x market) cells vs v4")
    print(f"\nEvaluating {len(live_configs)} configs x 4 cells "
          f"(inner_train/inner_val x spot/futures_5x), candidate and control, "
          f"paired block bootstrap (30-day blocks, 2000 draws).")
    print("A negative dlogG means the trailing stop LOSES to v4 in that cell.\n")

    by_cfg: dict[float, list[dict]] = {}
    for k, sf in enumerate(live_configs, 1):
        lbl = label_of(sf)
        rows = compare(make_builder(sf), df_full, label=lbl,
                       markets=(SPOT, FUTURES),
                       slice_names=("inner_train", "inner_val"))
        by_cfg[sf] = rows
        print(f"--- [{k}/{len(live_configs)}] {lbl}  (stop_frac_value={sf:g})")
        print_rows(rows)
        print()
    return by_cfg


# ------------------------------------------------------------------- runner

def main() -> None:
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-90 CONSERVATIVE -- literal fixed-percentage trailing stop, instant "
       "unconditional\nrestart, overlaid on kelly_regime_v4's raw desired exposure. "
       "6 frozen configurations.\nDefault verdict: REJECT.")
    df_full = load_btc()
    max_ts_seen.append(df_full.index.max())
    df_train = df_full.loc[:INNER_TRAIN_END]
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(df_full):,} bars, "
          f"{df_full.index[0]} -> {df_full.index[-1]}")
    print(f"inner-train frame: {len(df_train):,} bars, {df_train.index[0]} -> "
          f"{df_train.index[-1]}")

    # ------------------------------------------------------------- Step A
    gate = step_a(df_full, df_train)
    live_configs = [r["stop_frac"] for r in gate["a2_rows"] if not r["inert"]]

    # ------------------------------------------------------------- Step B
    by_cfg = step_b(df_full, live_configs)

    hr("STEP B surface -- the full swept grid, so the SHAPE is visible, not "
       "just the winner")
    print(f"\n{'stop_frac':>10s} {'train/spot':>11s} {'train/fut5x':>12s} "
          f"{'val/spot':>10s} {'val/fut5x':>10s}   (paired dlogGrowth vs v4)")
    print("-" * 68)
    sel_val_fut: dict[float, float] = {}
    for sf in live_configs:
        rows = by_cfg[sf]
        c_ts = cell(rows, "inner_train", SPOT.name)["d_loggrowth"]
        c_tf = cell(rows, "inner_train", FUTURES.name)["d_loggrowth"]
        c_vs = cell(rows, "inner_val", SPOT.name)["d_loggrowth"]
        c_vf = cell(rows, "inner_val", FUTURES.name)["d_loggrowth"]
        sel_val_fut[sf] = c_vf
        print(f"{sf:>10.2f} {c_ts:>11.3f} {c_tf:>12.3f} {c_vs:>10.3f} {c_vf:>10.3f}")

    # ------------------------------------------------------- the finalist
    best_sf = max(live_configs, key=lambda sf: sel_val_fut[sf])
    b_lbl = label_of(best_sf)
    b_rows = by_cfg[best_sf]

    hr("STEP C -- the pre-registered promotion bar (default REJECT)")
    print(f"\nFINALIST by the frozen selection statistic (inner-validation paired "
          f"dlogGrowth vs v4 on futures_5x):\n    {b_lbl}   stop_frac_value={best_sf:g}"
          f"   selection statistic = {sel_val_fut[best_sf]:+.3f} log units")
    print("\n    All four cells of the finalist:")
    print_rows(b_rows)

    # exposure / vol ratio, restated from compare()'s own risk-match columns --------
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
    print("\n--- B3  plateau not peak: the finalist's immediate grid neighbours "
          "(inner-validation futures d_loggrowth)")
    idx = STOP_GRID.index(best_sf) if best_sf in STOP_GRID else None
    neigh = []
    if idx is not None:
        for off, tag in ((-1, "lower"), (1, "higher")):
            j = idx + off
            if 0 <= j < len(STOP_GRID):
                sf_n = STOP_GRID[j]
                v = sel_val_fut.get(sf_n)  # None if that neighbour was INERT
                neigh.append((tag, sf_n, v))
            else:
                neigh.append((tag, None, None))
    print(f"      finalist              {b_lbl:>10s}  selection = "
          f"{sel_val_fut[best_sf]:+.3f}")
    for tag, sf_n, v in neigh:
        if sf_n is None:
            print(f"      neighbour ({tag:>6s})   (off grid)")
        elif v is None:
            print(f"      neighbour ({tag:>6s})   {label_of(sf_n):>10s}  "
                  f"selection = INERT / excluded at Step A")
        else:
            print(f"      neighbour ({tag:>6s})   {label_of(sf_n):>10s}  "
                  f"selection = {v:+.3f}   (drop from finalist: {v - sel_val_fut[best_sf]:+.3f})")
    have = [v for _, _, v in neigh if v is not None]
    if have and sel_val_fut[best_sf] != 0:
        together = all(np.sign(v) == np.sign(sel_val_fut[best_sf]) and
                       abs(v - sel_val_fut[best_sf]) <= 0.5 * abs(sel_val_fut[best_sf])
                       for v in have)
    else:
        together = False
    b3 = bool(together)
    print(f"      neighbours move WITH the finalist (same sign, within half its "
          f"own magnitude): {together}")
    print(f"    B3: {'PASS' if b3 else 'FAIL'}  (informational per ROUTINE.md; not "
          f"one of the 5 boolean promotion gates B1/B2/B4a/B4b/B5, but reported in "
          f"full as B3 requires)")

    # ---- B4(a)  ETH ---------------------------------------------------------
    print("\n--- B4(a)  falsification: ETH replication (Bitfinex ETH, pre-2023)")
    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    print(f"      ETH frame: {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}")
    print("      NOTE: this ETH series ends 2019-12-31, so inner-validation is "
          "EMPTY on ETH.\n      The replication runs on inner-train only -- a "
          "2017-2019 sign check, not a like-for-like\n      repeat of the "
          "selection slice. Stated plainly, not papered over.")
    eth_rows = compare(make_builder(best_sf), eth, label=f"ETH_{b_lbl}",
                       markets=(SPOT, FUTURES), slice_names=("inner_train",))
    print()
    print_rows(eth_rows)
    btc_train_fut_sign = np.sign(cell(b_rows, "inner_train", FUTURES.name)["d_loggrowth"])
    eth_pts = {r["market"]: r["d_loggrowth"] for r in eth_rows}
    same_sign = {m: bool(np.sign(v) == btc_train_fut_sign and v != 0)
                 for m, v in eth_pts.items()}
    b4a = bool(eth_rows) and all(same_sign.values())
    print(f"      BTC inner-train futures sign (reference) = "
          f"{'+' if btc_train_fut_sign > 0 else '-'}")
    for m, v in eth_pts.items():
        print(f"      ETH {m:11s} dlogG = {v:+7.3f}  -> same sign as BTC inner-train: "
              f"{same_sign[m]}")
    print(f"    B4(a): {'PASS' if b4a else 'FAIL'}")

    # ---- B4(b)  whipsaw falsification ---------------------------------------
    print("\n--- B4(b)  falsification: stopout whipsaw rate, finalist's own run, "
          "inner-train BTC futures_5x")
    final_train, events_train = run_mechanism(df_train, best_sf)
    diag = stopout_whipsaw_rate(df_train["close"].to_numpy(), final_train, events_train)
    print(f"      stop_events={diag['stop_events']}  "
          f"events_with_reentry_in_horizon={diag['events_with_reentry_in_horizon']}  "
          f"whipsaws={diag['whipsaws']}")
    print(f"      whipsaw_rate={diag['whipsaw_rate']:.3f}  "
          f"mean_whipsaw_log_cost={diag['mean_whipsaw_log_cost']:+.4f}")
    train_fut_point = cell(b_rows, "inner_train", FUTURES.name)["d_loggrowth"]
    print(f"      finalist's B1 point estimate on this same cell (inner-train, "
          f"futures_5x) = {train_fut_point:+.3f}")
    named_risk_confirmed = bool(diag["whipsaw_rate"] > WHIPSAW_RATE_FAIL and
                                train_fut_point <= 0)
    b4b = not named_risk_confirmed
    print(f"      named-risk confirmed (whipsaw_rate > {WHIPSAW_RATE_FAIL} AND "
          f"inner-train futures point estimate NOT positive): {named_risk_confirmed}")
    print(f"    B4(b): {'PASS' if b4b else 'FAIL'}"
          f"{'  (the round named risk is confirmed)' if named_risk_confirmed else ''}")
    b4 = bool(b4a and b4b)
    print(f"    B4 overall (a AND b): {'PASS' if b4 else 'FAIL'}")

    # ---- B5  0.40% taker ------------------------------------------------------
    print("\n--- B5  cost robustness: inner-validation re-run at a 0.40% taker fee")
    spot40 = MarketSpec(name="spot@0.40%", fee_rate=HIGH_FEE, leverage=1.0,
                        allow_short=False)
    fut40 = MarketSpec(name="fut5x@0.40%", fee_rate=HIGH_FEE, leverage=5.0,
                       allow_short=True, pays_funding=True)
    print(f"      market specs built here (fields checked against "
          f"tradebot/broker.py's MarketSpec dataclass;\n      no `is_futures` "
          f"field exists there -- futures uses `pays_funding=True` like the "
          f"shared FUTURES spec):\n      {spot40}\n      {fut40}")
    rows40 = compare(make_builder(best_sf), df_full, label=f"fee40_{b_lbl}",
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
    print(f"    B5: {'PASS' if b5 else 'FAIL'}   (asks only that the "
          f"improvement does not REVERSE sign)")

    # ------------------------------------------------------------- verdict
    hr("VERDICT")
    clauses = {"B1": b1, "B2": b2, "B4(a)": b4a, "B4(b)": b4b, "B5": b5}
    for k, v in clauses.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    print(f"    (B3 reported above as required, informational per the frozen rule)")
    promote = all(clauses.values())
    verdict = "CANDIDATE FOR HOLDOUT" if promote else "NEGATIVE"
    print(f"\n    VERDICT: {verdict}")
    if verdict == "NEGATIVE":
        failed = [k for k, v in clauses.items() if not v]
        print(f"    Clauses failed: {', '.join(failed) if failed else '(none -- see B3 note)'}")
    print("\n    (The decision rule above is exactly the one frozen in this "
          "file's docstring before\n    any number was read. No threshold was "
          "moved. The holdout itself is NOT read or\n    touched by this "
          "script, win or lose -- that is the operator's call.)")

    # ---------------------------------------------------------- bookkeeping
    hr("BOOKKEEPING")
    print(f"    Frozen grid                                    : 6 configurations "
          f"(1 identity + 5 swept)")
    print(f"    Step-A mechanism gate run on                   : 6 (identity + 5 swept)")
    n_inert = sum(r["inert"] for r in gate["a2_rows"])
    print(f"    INERT at A2 (stop never fired), excluded       : {n_inert}")
    print(f"    Step-B evaluated (4 cells each)                : {len(by_cfg)}")
    print(f"    Finalist re-runs (ETH, 0.40%% fee)              : 2 "
          f"(re-runs of an already-counted configuration, not new configurations)")
    print(f"    TOTAL DISTINCT (data, stop_frac_value) SETTINGS: "
          f"{6 + 2} across this file")
    print(f"\n    A1 max |diff| vs r90_shared.v4_target(): {gate['a1_max_abs']:.3e}")
    print(f"    Max timestamp read anywhere in this run  : {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")


if __name__ == "__main__":
    main()
