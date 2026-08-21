#!/usr/bin/env python
"""R-92 NOVEL branch (B-42): a CAUSAL, EXPANDING-WINDOW, ANNUALLY RE-ESTIMATED
Sepp & Lucic (2026) AR(1) anchor span, replacing `kelly_regime_v4`'s frozen
20/40/80 doubling ladder with a ladder that is re-derived every January 1
from all data strictly before that date.

MECHANISM (one sentence). At each annual checkpoint, refit phi (AR(1)
autocorrelation) and mu (drift) of BTC's own volatility-normalized daily
returns on an EXPANDING window of everything seen so far, re-derive the
closed-form SR(nu)-maximizing span via `r92_shared.derive_optimal_span`, and
use a doubling ladder around that span (0.5x, 1x, 2x -- the SAME convention
this round's conservative sibling uses for its one frozen fit) to drive
`v4`'s own vote-times-scale architecture for every bar until the next
checkpoint -- so the anchors can track slow drift in BTC's own
autocorrelation structure across bull/bear/chop regimes, instead of a single
global fit (the conservative branch) or the a-priori 20/40/80 (v4 itself).

This is the direct test of the question the module docstring names: does the
operator's pilot finding on the FULL 2017-2020 window (phi=0.046, mu=0.078,
SR(nu) essentially monotonically increasing to the 200-day grid boundary --
a real, informative "the theory says buy-and-hold" result, not a bug) hold
at every point in BTC's history, or is it an artifact of mixing 2017's bull,
2018's bear/chop and 2019-2020 into one static fit? Splitting the fit
annually and expanding it forward answers that directly: if 2018-heavy
early checkpoints show weaker drift relative to autocorrelation, an interior
optimum can appear there even though the full-period fit never finds one.

--------------------------------------------------------------------------
THREE DESIGN DECISIONS, NAMED EXPLICITLY (none of them a shortcut; each is
disclosed here, before any number below is read, per this round's own
convention of naming simplifications up front)
--------------------------------------------------------------------------
(1) LADDER-FROM-SPAN. A single closed-form optimum is one number; `v4`'s
    architecture wants three anchors. This branch reuses the conservative
    sibling's own convention verbatim: ladder = (round(span/2), span,
    round(span*2)), floored at 1 day -- so the two branches are compared on
    an apples-to-apples ladder-construction rule, and neither branch gets a
    ladder-shape advantage the other lacks.
(2) FALLBACK LADDER, two distinct triggers, both mapped to the SAME v4
    shipped ladder (20, 40, 80) but for different reasons, both logged
    separately in the per-checkpoint table's "source" column:
      (a) A0 fails at a checkpoint (no positive phi, or SR(nu) has no
          interior optimum at that checkpoint) -- the closed form has
          nothing to hand over for that period, so v4's own ladder governs
          it, exactly as if this branch were switched off for that window.
      (b) Before the FIRST checkpoint (BTC: 2017-01-01 -> 2018-12-31 has no
          checkpoint at all, since the first checkpoint requiring >=2 years
          of history is 2019-01-01) there is no derived span yet to use, so
          the same v4 ladder governs this warm-up stretch too.
(3) SEAMS AT CHECKPOINT BOUNDARIES ARE EXPECTED AND CAUSAL. Per the
    operator's construction (`span_ladder_target`, `r92_shared`), each
    checkpoint's piece is built by running that checkpoint's OWN ladder
    through v4's full vote-times-scale-times-deadband pipeline over the
    WHOLE frame, then slicing out only that checkpoint's bar range. Because
    `apply_deadband` is a sequential function of one ladder's entire history
    from bar 0, two adjacent pieces (built under two different ladders) can
    disagree on what the "current position" was at the boundary, producing
    a possible position jump exactly at each January 1. This is NOT
    lookahead (every value used to build any one piece is still a strictly
    causal function of bars at or before its own index -- verified below
    with `causal_truncation_probe` on the assembled, full `build_target`)
    -- it is a real structural property of "assemble independently-deadbanded
    pieces," named here per the operator's own instruction to build it this
    way, not discovered after the fact.

ETH COVERAGE CORRECTION (disclosed, not silently changed). The module
docstring's B4 clause anticipates "ETH coverage starts 2019-03-14"; the
actual committed Bitfinex ETH series (`r92_shared.load_eth()`) starts
2016-03-09 and ends 2019-12-31 -- verified by direct inspection below, not
assumed. This makes ETH's own expanding-window checkpoint schedule *less*
sparse at the start (>=2 years of history is available by 2018-03-09, so
ETH's own first checkpoint is also 2019-01-01) but far sparser at the end:
ETH's series ends before a second checkpoint (2020-01-01) is ever reached,
so ETH gets exactly ONE checkpoint for its whole pre-holdout history. This
is reported plainly as B4's own disclosed limitation, not worked around.

--------------------------------------------------------------------------
THE FROZEN DECISION RULE (from `r92_shared`'s pre-registration; applied
here as the novel branch's own concrete procedure, written before any
checkpoint's phi/mu was read)
--------------------------------------------------------------------------
Novel-branch A0 (checked BEFORE Step A, one gate covering every checkpoint):
  if EVERY checkpoint's own `kill_switch_a0` fails, the branch is
  disqualified by pre-registration -- reported NEGATIVE, no Step B run,
  per the task's own instruction that this is itself a complete, clean,
  more general negative result and forcing a backtest past it would not
  add information.
  If AT LEAST ONE checkpoint passes A0, the branch proceeds to Step A/B in
  full, honestly, per the standing R-89/R-90/R-91 convention that a partial
  or even a fully-fired kill switch does not exempt a branch from being
  measured end-to-end.

Step A (mechanism gate, before any performance number is read):
  A1' reproducibility -- re-running the checkpoint derivation twice from a
      clean call must return identical spans and ladders (determinism).
  A2  non-inertness   -- R^2 of build_target(train) vs v4_target(train) on
      inner-train must be < 0.98, else the construction is inert-by-
      -construction (the derived ladders round back to ~20/40/80 or the
      fallback dominates the window) and is reported, not scored, as a
      genuine test of the theory.
  A3  causality        -- `causal_truncation_probe` (cuts 0.55, 0.80) on the
      full, assembled `build_target`, PLUS every checkpoint's own fit window
      is asserted strictly before its checkpoint date (done inline in
      `compute_checkpoints`, not just at the two probe cuts).

Step B (selection: there is no sweep -- one assembled candidate, measured on
both slices and both markets against v4, exactly as `r92_shared`'s Step B
describes for both branches of this round):
  `compare(build_target, btc, label="r92_novel_rolling_ar1_span")`.

Promotion bar (default REJECT, `r92_shared`'s B1-B5, identical structure to
the conservative sibling):
  B1 paired bootstrap excludes zero in >=1 of 4 cells, positive point
     estimate in all 4.
  B2 dSharpe > +0.2 on inner-validation on both markets, OR a risk-matched
     drawdown improvement on both markets.
  B3 plateau not peak -- for every checkpoint whose A0 passed, the SR(nu)
     curve one grid step either side of the derived span must sit no higher
     than the derived span itself (checked per-checkpoint, since this
     branch has no single grid to re-sweep, only each checkpoint's own
     closed-form curve).
  B4 ETH replication, inner_train only, disclosed sparsity above.
  B5 cost robustness at 0.40% taker, inner-validation, both markets.

This file never reads a bar at or after OOS_START (2023-01-01): every load
goes through `r92_shared`'s truncating loaders, and the maximum timestamp
actually touched anywhere is tracked and printed at the end of main().
`r92_shared.py` is READ-ONLY and is not modified by this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r92_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    V4_BAND,
    V4_HORIZONS,
    causal_truncation_probe,
    compare,
    daily_log_returns,
    derive_optimal_span,
    fee_at,
    fit_ar1,
    kill_switch_a0,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    span_ladder_target,
    v4_target,
    vol_normalized_returns,
)

TAKER_040 = 0.0040  # B5 cost-robustness fee, Bitstamp's real entry taker tier


# ---------------------------------------------------------------------------
# 1. Checkpoint schedule: the first January 1 with >= 2 full years of daily
#    bar-level history before it, then every subsequent January 1 while data
#    remains. Generalised (not hard-coded to BTC's own 2017-01-01 start) so
#    the SAME function, unmodified, produces ETH's own schedule for B4.
# ---------------------------------------------------------------------------
def annual_checkpoints(df: pd.DataFrame) -> list[pd.Timestamp]:
    if len(df) == 0:
        return []
    start = df.index[0]
    end = df.index[-1]
    min_history_date = start + pd.DateOffset(years=2)
    first_year = min_history_date.year
    first_ckpt = pd.Timestamp(year=first_year, month=1, day=1, tz="UTC")
    if first_ckpt < min_history_date:
        first_ckpt = pd.Timestamp(year=first_year + 1, month=1, day=1, tz="UTC")
    ckpts = []
    y = first_ckpt.year
    while True:
        c = pd.Timestamp(year=y, month=1, day=1, tz="UTC")
        if c > end:
            break
        ckpts.append(c)
        y += 1
    return ckpts


def ladder_from_span(span: int) -> tuple[int, int, int]:
    """Design decision (1): doubling ladder around a derived center span,
    the SAME (0.5x, 1x, 2x) convention this round's conservative sibling
    uses, rounded to the nearest integer day, floored at 1."""
    lo = max(1, int(round(span / 2.0)))
    mid = int(span)
    hi = max(mid + 1, int(round(span * 2.0)))
    return (lo, mid, hi)


# ---------------------------------------------------------------------------
# 2. Per-checkpoint causal fit: phi, mu, derived span, A0 verdict, and the
#    ladder that checkpoint hands to every bar until the next one.
# ---------------------------------------------------------------------------
def compute_checkpoints(df: pd.DataFrame) -> list[dict]:
    infos = []
    for c in annual_checkpoints(df):
        fit_df = df[df.index < c]
        if len(fit_df) == 0:
            continue
        # Lookahead guard, checked directly, not just asserted by convention.
        assert fit_df.index[-1] < c, (
            f"LOOKAHEAD: fit window for checkpoint {c} reaches {fit_df.index[-1]}")
        dr = daily_log_returns(fit_df)
        z = vol_normalized_returns(dr)
        phi, mu = fit_ar1(z)
        span, sr_curve, grid = derive_optimal_span(phi, mu)
        a0_pass, a0_reason = kill_switch_a0(phi, sr_curve, grid)
        if a0_pass:
            ladder = ladder_from_span(span)
            source = "derived"
        else:
            ladder = V4_HORIZONS
            source = "fallback(A0 fail)"
        infos.append(dict(
            checkpoint=c, fit_start=fit_df.index[0], fit_end=fit_df.index[-1],
            n_daily_z=len(z), phi=phi, mu=mu, span=span, sr_curve=sr_curve,
            grid=grid, a0_pass=a0_pass, a0_reason=a0_reason,
            ladder=ladder, ladder_source=source,
        ))
    return infos


# ---------------------------------------------------------------------------
# 3. Build the assembled, time-varying-ladder target path: pre-first-
#    -checkpoint fallback segment, then one `span_ladder_target` piece per
#    checkpoint, sliced to that checkpoint's own bar range and concatenated.
#    Small frame-identity cache (same convention as r90/r91's novel
#    branches) since `compare()` and the causal probe call this repeatedly
#    on the same frame.
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _key(df: pd.DataFrame) -> tuple:
    return (len(df), int(df.index[0].value), int(df.index[-1].value),
            float(df["close"].iloc[0]), float(df["close"].iloc[-1]))


def build_target(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros(0)
    key = ("build_target",) + _key(df)
    if key in _CACHE:
        return _CACHE[key]

    infos = compute_checkpoints(df)
    idx = df.index
    out = np.zeros(n)

    first_idx = int(idx.searchsorted(infos[0]["checkpoint"])) if infos else n
    if first_idx > 0:
        # Design decision (2b): no checkpoint governs this warm-up stretch yet.
        pre_path = span_ladder_target(df, V4_HORIZONS)
        out[:first_idx] = pre_path[:first_idx]

    for i, info in enumerate(infos):
        start_idx = int(idx.searchsorted(info["checkpoint"]))
        end_idx = (int(idx.searchsorted(infos[i + 1]["checkpoint"]))
                   if i + 1 < len(infos) else n)
        piece = span_ladder_target(df, info["ladder"])
        out[start_idx:end_idx] = piece[start_idx:end_idx]

    _CACHE[key] = out
    return out


def plateau_check(info: dict) -> tuple[bool | None, str]:
    """B3, per-checkpoint: the SR(nu) curve one grid step either side of the
    derived span must sit no higher than at the derived span itself."""
    if not info["a0_pass"]:
        return None, "n/a (A0 failed at this checkpoint -- no derived optimum to check)"
    grid, sr, span = info["grid"], info["sr_curve"], info["span"]
    idx = int(np.where(grid == span)[0][0])
    left = sr[idx - 1] if idx - 1 >= 0 else -np.inf
    right = sr[idx + 1] if idx + 1 < len(grid) else -np.inf
    ok = bool(left <= sr[idx] and right <= sr[idx])
    return ok, f"SR(span-1)={left:.4f} <= SR(span)={sr[idx]:.4f} >= SR(span+1)={right:.4f}"


def print_checkpoint_table(infos: list[dict], asset: str) -> None:
    print(f"\n{asset} per-checkpoint table ({len(infos)} checkpoint(s)):")
    hdr = (f"{'checkpoint':12s} {'fit_start':12s} {'fit_end':12s} {'n_z':>6s} "
           f"{'phi':>8s} {'mu':>9s} {'span':>6s} {'A0':>6s} {'ladder':>16s} {'source':>18s}")
    print(hdr)
    print("-" * len(hdr))
    for info in infos:
        print(f"{str(info['checkpoint'].date()):12s} "
              f"{str(info['fit_start'].date()):12s} {str(info['fit_end'].date()):12s} "
              f"{info['n_daily_z']:6d} {info['phi']:+8.4f} {info['mu']:+9.5f} "
              f"{info['span']:6d} {'PASS' if info['a0_pass'] else 'FAIL':>6s} "
              f"{str(info['ladder']):>16s} {info['ladder_source']:>18s}")
        print(f"{'':12s} reason: {info['a0_reason']}")


def hdr(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:
    max_ts = []

    hdr("R-92 NOVEL BRANCH -- CAUSAL EXPANDING-WINDOW ANNUAL AR(1) SPAN RE-ESTIMATION (B-42)")
    print("mechanism: at each Jan-1 checkpoint (first: 2 full years of history available), refit")
    print("phi/mu on an EXPANDING window strictly before that date, re-derive the SR(nu)-maximizing")
    print("span, and use a (0.5x,1x,2x) doubling ladder from it to drive v4's vote*scale pipeline")
    print("for every bar until the next checkpoint. Fallback = v4's shipped 20/40/80 ladder, used")
    print("(a) before the first checkpoint and (b) at any checkpoint where A0 fails.")

    btc = load_btc()
    max_ts.append(btc.index.max())
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")

    hdr("PER-CHECKPOINT TABLE -- BTC, full pre-holdout series (the frame Step B actually uses)")
    infos_btc = compute_checkpoints(btc)
    print_checkpoint_table(infos_btc, "BTC")
    print("\nLookahead check: for every checkpoint above, fit_end < checkpoint date is asserted")
    print("inline in compute_checkpoints() -- verified for all checkpoints (no AssertionError raised).")

    # ---- A1' reproducibility (determinism) -------------------------------
    hdr("A1' REPRODUCIBILITY (determinism check)")
    infos_btc_2 = compute_checkpoints(btc)
    a1_ok = all(a["span"] == b["span"] and a["ladder"] == b["ladder"] and a["a0_pass"] == b["a0_pass"]
                for a, b in zip(infos_btc, infos_btc_2))
    print(f"Re-running compute_checkpoints(btc) twice returns identical spans/ladders/A0 verdicts: "
          f"{a1_ok}")

    # ---- novel-branch A0: at least one checkpoint must pass --------------
    hdr("NOVEL-BRANCH A0 -- at least one checkpoint must pass its own kill_switch_a0")
    n_pass = sum(1 for i in infos_btc if i["a0_pass"])
    print(f"{n_pass} of {len(infos_btc)} BTC checkpoints pass A0 (positive phi AND an interior "
          f"SR(nu) optimum).")
    any_pass = n_pass > 0
    if not any_pass:
        hdr("VERDICT")
        print("EVERY checkpoint fails A0: no interior optimum appears at ANY point in BTC's own")
        print("pre-holdout history, under the expanding-window construction. This is a stronger,")
        print("more general version of the operator's own full-period pilot finding -- the branch")
        print("is disqualified by pre-registration and reported NEGATIVE. No Step B backtest is run;")
        print("forcing one past an all-checkpoints A0 failure would not add information.")
        print(f"\nConfigurations evaluated (backtested): 0. Checkpoint fits are closed-form")
        print(f"derivations, not backtests, and do not count toward the trials total.")
        print(f"\nmax timestamp read anywhere: {max(max_ts)}  (< {OOS_START})")
        return

    print("At least one checkpoint passes A0 -- proceeding to the full pre-registered Step A/B checks,")
    print("run and reported end-to-end regardless of how many checkpoints individually pass, per the")
    print("standing R-89/R-90/R-91 convention.")

    # ---- Step A2: non-inertness on inner-train ----------------------------
    hdr("STEP A2 -- NON-INERTNESS (inner-train)")
    train = btc.loc[:INNER_TRAIN_END]
    print(f"inner-train frame: {len(train):,} bars  {train.index[0]} -> {train.index[-1]}")
    infos_train = compute_checkpoints(train)
    print_checkpoint_table(infos_train, "BTC inner-train-only re-derivation")
    print("\n(shown separately: build_target(train) only ever sees checkpoints that fit inside")
    print("train's own 2017-01-01..2020-12-31 range -- fewer than the full-series table above,")
    print("by construction, since it is re-computed fresh on whatever frame it is given.)")

    cand_train = build_target(train)
    ctrl_train = v4_target(train)
    rsq = r_squared(cand_train, ctrl_train)
    a2_pass = rsq < 0.98
    print(f"\nR^2(candidate, v4_target) on inner-train: {rsq:.5f}  "
          f"-> {'non-inert (A2 PASS)' if a2_pass else 'INERT (A2 FAIL)'}")

    # ---- Step A3: causality -------------------------------------------
    hdr("STEP A3 -- CAUSAL TRUNCATION PROBE (full assembled build_target)")
    print("Rebuild the target on 55% and 80% truncations of the full BTC pre-holdout frame; the")
    print("shared prefix must match bit-for-bit -- the check most likely to catch a checkpoint's fit")
    print("or ladder leaking data from after its own checkpoint date, or after the truncation point.")
    try:
        a3_full_ok = causal_truncation_probe(build_target, btc, cuts=(0.55, 0.80))
        a3_full_msg = "PASS (cuts 0.55, 0.80, full BTC frame)"
    except AssertionError as exc:
        a3_full_ok = False
        a3_full_msg = f"FAIL -- {exc}"
    print(f"  {a3_full_msg}")
    try:
        a3_train_ok = causal_truncation_probe(build_target, train, cuts=(0.55, 0.80))
        a3_train_msg = "PASS (cuts 0.55, 0.80, inner-train frame)"
    except AssertionError as exc:
        a3_train_ok = False
        a3_train_msg = f"FAIL -- {exc}"
    print(f"  {a3_train_msg}")
    a3_pass = a3_full_ok and a3_train_ok
    print(f"A3 = {'PASS' if a3_pass else 'FAIL'}")

    # ---- Step B: compare vs v4 ------------------------------------------
    hdr("STEP B -- compare() vs kelly_regime_v4, BTC, 2 slices x 2 markets")
    rows = compare(build_target, btc, label="r92_novel_rolling_ar1_span")
    print()
    print_rows(rows)

    # ---- B1 ---------------------------------------------------------------
    hdr("PROMOTION BAR -- B1-B5")
    pts = [r["d_loggrowth"] for r in rows]
    excl = [r["excludes_zero"] for r in rows]
    b1 = any(excl) and all(p > 0 for p in pts)
    print(f"B1 paired bootstrap: excludes zero in {sum(excl)}/4 cells; positive point estimate in "
          f"{sum(1 for p in pts if p > 0)}/4 cells")
    for r in rows:
        print(f"   {r['slice']:11s} {r['market']:11s} d_loggrowth={r['d_loggrowth']:+.4f} "
              f"[{r['d_lo']:+.4f},{r['d_hi']:+.4f}]  excludes_zero={r['excludes_zero']}")
    print(f"   B1 = {'PASS' if b1 else 'FAIL'}")

    # ---- B2 ---------------------------------------------------------------
    val = [r for r in rows if r["slice"] == "inner_val"]
    dsh = {r["market"]: r["d_sharpe"] for r in val}
    ddd = {r["market"]: r["d_dd"] for r in val}
    rm = {r["market"]: r["risk_matched"] for r in val}
    b2_sharpe = all(v > 0.2 for v in dsh.values())
    b2_dd = all(v < 0.0 for v in ddd.values()) and all(rm.values())
    b2 = b2_sharpe or b2_dd
    print(f"\nB2 noise floor (inner-validation):")
    print(f"   dSharpe: " + ", ".join(f"{k}={v:+.3f}" for k, v in dsh.items())
          + f"   -> both > +0.2: {b2_sharpe}")
    print(f"   dMaxDD:  " + ", ".join(f"{k}={v:+.2f}pp" for k, v in ddd.items())
          + f"   -> both improved AND risk_matched: {b2_dd}")
    print(f"   B2 = {'PASS' if b2 else 'FAIL'} (via "
          f"{'Sharpe' if b2_sharpe else ('matched drawdown' if b2_dd else 'neither')})")

    # ---- B3: per-checkpoint plateau ---------------------------------------
    hdr("B3 -- PLATEAU NOT PEAK (per checkpoint, BTC full-series table)")
    print(f"{'checkpoint':12s} {'A0':>6s} {'plateau':>9s} {'detail':<55s}")
    print("-" * 90)
    b3_results = []
    for info in infos_btc:
        ok, detail = plateau_check(info)
        b3_results.append(ok)
        print(f"{str(info['checkpoint'].date()):12s} {'PASS' if info['a0_pass'] else 'FAIL':>6s} "
              f"{('PASS' if ok else 'FAIL') if ok is not None else 'n/a':>9s} {detail:<55s}")
    applicable = [r for r in b3_results if r is not None]
    b3 = bool(applicable) and all(applicable)
    print(f"\nB3 = {'PASS' if b3 else 'FAIL'} ({sum(1 for r in applicable if r)}/{len(applicable)} "
          f"A0-passing checkpoints show a genuine interior plateau)")

    # ---- B4: ETH replication ------------------------------------------
    hdr("B4 -- FALSIFICATION: ETH REPLICATION (inner_train only)")
    eth = load_eth()
    max_ts.append(eth.index.max())
    print(f"ETH (Bitfinex): {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}")
    print("Correction to the module docstring's assumed '2019-03-14' ETH start: the actual committed")
    print(f"series starts {eth.index[0].date()} and ends {eth.index[-1].date()} -- verified directly")
    print("above, not assumed. ETH's own >=2-year-history threshold is also reached by 2018-03-09, so")
    print("its first checkpoint is 2019-01-01 too, but the series ends 2019-12-31, before a second")
    print("checkpoint (2020-01-01) is ever reached -- ETH gets exactly ONE checkpoint for its whole")
    print("pre-holdout history. Disclosed, not worked around.")
    infos_eth = compute_checkpoints(eth)
    print_checkpoint_table(infos_eth, "ETH")

    eth_rows = compare(build_target, eth, label="r92_novel_rolling_ar1_span", slice_names=("inner_train",))
    print()
    print_rows(eth_rows)
    btc_train_sign = {r["market"]: (1.0 if r["d_loggrowth"] > 0 else -1.0)
                      for r in rows if r["slice"] == "inner_train"}
    eth_ok = []
    for r in eth_rows:
        btc_sign = btc_train_sign.get(r["market"])
        same = btc_sign is not None and np.sign(r["d_loggrowth"]) == btc_sign
        eth_ok.append(same)
        print(f"   {r['market']:11s}: BTC inner-train sign={'+' if btc_sign and btc_sign > 0 else '-'}  "
              f"ETH d_loggrowth={r['d_loggrowth']:+.4f}  same sign: {same}")
    b4 = bool(eth_rows) and all(eth_ok)
    print(f"   B4 = {'PASS' if b4 else 'FAIL'}")

    # ---- B5: cost robustness ---------------------------------------------
    hdr("B5 -- COST ROBUSTNESS: 0.40% TAKER, inner-validation")
    spot_040 = fee_at(SPOT, TAKER_040)
    fut_040 = fee_at(FUTURES, TAKER_040)
    fee_rows = compare(build_target, btc, label="r92_novel_rolling_ar1_span@40bp",
                       markets=(spot_040, fut_040), slice_names=("inner_val",))
    print()
    print_rows(fee_rows)
    val_sign = {r["market"]: (1.0 if r["d_loggrowth"] > 0 else -1.0) for r in val}
    fee_ok = []
    for r in fee_rows:
        base_mkt = "spot" if "spot" in r["market"] else "futures_5x"
        base_sign = val_sign.get(base_mkt)
        same = base_sign is not None and np.sign(r["d_loggrowth"]) == base_sign
        fee_ok.append(same)
        print(f"   {r['market']:14s} d_loggrowth={r['d_loggrowth']:+.4f}  "
              f"(base-fee inner-val {base_mkt} sign={'+' if base_sign and base_sign > 0 else '-'})  "
              f"same sign: {same}")
    b5 = all(fee_ok)
    print(f"   B5 = {'PASS' if b5 else 'FAIL'} (sign reversal vs 0.10% baseline: {not b5})")

    # ---- verdict ------------------------------------------------------
    hdr("VERDICT")
    print(f"Novel-branch A0 (>=1 checkpoint passes): {'PASS' if any_pass else 'FAIL'} "
          f"({n_pass}/{len(infos_btc)} checkpoints)")
    print(f"A1' reproducibility: {'PASS' if a1_ok else 'FAIL'}")
    print(f"A2 non-inertness (R^2={rsq:.5f} < 0.98): {'PASS' if a2_pass else 'FAIL'}")
    print(f"A3 causality: {'PASS' if a3_pass else 'FAIL'}")
    clauses = {"B1": b1, "B2": b2, "B3": b3, "B4 ETH": b4, "B5 0.40% taker": b5}
    for k, v in clauses.items():
        print(f"  {k:16s} {'PASS' if v else 'FAIL'}")
    gate_all = any_pass and a1_ok and a2_pass and a3_pass
    promote = gate_all and all(clauses.values())
    print(f"\nVERDICT: {'CANDIDATE FOR HOLDOUT' if promote else 'NEGATIVE'}")
    if not promote:
        failed = ([] if any_pass else ["novel A0"]) + ([] if a1_ok else ["A1'"]) + \
                 ([] if a2_pass else ["A2"]) + ([] if a3_pass else ["A3"]) + \
                 [k for k, v in clauses.items() if not v]
        print(f"Failing clause(s): {', '.join(failed)}")

    hdr("CONFIGURATIONS EVALUATED")
    print("Per-checkpoint closed-form fits (phi/mu/derive_optimal_span/kill_switch_a0) are NOT")
    print("configurations in the trials-count sense -- they are not backtests.")
    print("Exactly ONE assembled candidate target path was ever run through a backtest:")
    print("  r92_novel_rolling_ar1_span (the single checkpointed, time-varying-ladder path).")
    print(f"  BTC cells (1 config x 2 markets x 2 slices):                     4")
    print(f"  ETH cells (1 config x 2 markets x 1 slice, inner_train only):    2")
    print(f"  fee-robustness cells (1 config x 2 markets x 1 slice, @0.40%):   2")
    print(f"  => total configuration-cells evaluated: 8 (1 distinct configuration)")

    print(f"\nmax timestamp read anywhere in this branch (BTC and ETH): {max(max_ts)}  "
          f"(< {OOS_START}) -- no holdout bar was read.")


if __name__ == "__main__":
    main()
