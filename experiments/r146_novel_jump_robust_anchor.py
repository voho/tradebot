#!/usr/bin/env python
"""R-146 NOVEL branch: ``JumpRobustAnchorKellyV4`` -- replace each of
``kelly_regime_v4``'s three (20/40/80-day) rolling-MEAN vote anchors with a
rolling mean of the SAME window computed over a jump-MASKED close (Lee &
Mykland 2008-style causal per-bar jump exclusion), holding the 1% band,
latching hysteresis, 3-way averaging, ``scale`` and the 10% deadband all
exactly fixed to v4's own shipped values.

Full citation trail, the "not a duplicate of" argument against the ~15
prior SIZE-axis rounds and the parallel conservative (median-anchor)
branch, and the named failure-mode prediction are all already established
in ``r146_shared.py``'s own module docstring (read in full before this file
was written) and are not re-derived here. The one prior draft of this exact
construction that reconstructed a synthetic price path via cumsum of
winsorized returns and diverged (BTC 2017-2020 synthetic endpoint $6.1M vs
real ~$29K) was already caught and discarded before dispatch; this file
re-verifies, on the real BTC series, that the CURRENT masking-based
construction cannot drift that way (section 3 below).

=====================================================================
PRE-REGISTRATION (frozen before any real-data R^2, flag-rate, or backtest
number in this file was computed -- docs/ROUTINE.md steps 1-2, the same
convention ``r102_novel_signed_jump_discount.py`` and
``r103_novel_causal_rls.py`` used). Anything below later contradicted by
what actually happened is stated in the results section, not edited back
into this banner.
=====================================================================

1. MECHANISM (one sentence): ``r146_shared.jump_robust_anchor_vote_frac``
   computes each of v4's own three anchors as a rolling MEAN of `close`
   with jump-flagged bars excluded (masked to NaN, which pandas' rolling
   mean skips within a window), rather than a plain SMA over every bar;
   the vote itself is still computed against the REAL close, with the
   identical 1% band and latching (ffill) hysteresis v4 already ships.

2. STEP-0 NON-DEGENERACY GRID AND SELECTION RULE, PRE-REGISTERED BEFORE ANY
   GRID NUMBER WAS COMPUTED:

   Grid: ``threshold in {3.0, 4.0, 5.0}`` (Lee & Mykland jump-test
   multiplier on the causal MAD-based local vol), ``mad_span=288`` (1 day)
   HELD FIXED throughout, exactly as this round's dispatch specifies (a
   1-D grid, not a 2-D sweep -- ``mad_span`` is not a free parameter here).
   For each cell, on the BTC INNER-TRAIN slice only (2017-01-01 ->
   2020-12-31): compute the candidate's vote over the FULL pre-holdout
   price history (so the rolling 80-day-window anchor has its natural
   warmup exactly as it would in a real backtest), then restrict the two
   reported statistics to bars inside the inner-train window:
     - ``r2``        = ``r_squared(candidate_vote, v4_vote_frac)`` -- this
       is R-146's own A2 kill switch, evaluated on THIS round's own vote,
       not merely re-quoted from the operator's default-parameter check
       (which used threshold=4.0; this file re-verifies it for whichever
       cell the rule below actually selects, per this round's dispatch).
     - ``flag_rate``  = fraction of inner-train bars whose close is masked
       (jump-flagged) at that threshold -- a sanity check that the jump
       test is still selecting RARE events, not degenerating into a
       broad-brush filter.
   A cell QUALIFIES iff ``r2 < R2_THRESH = 0.98`` AND
   ``flag_rate < FLAG_RATE_SANITY = 0.05`` (well under the 5-10% band this
   round's dispatch names as the sanity bound; 0.05 is the tighter, more
   conservative endpoint of that band, fixed here before any cell's actual
   flag rate was computed).

   SELECTION RULE (non-degeneracy ONLY -- no performance number is
   inspected before this rule is applied, the identical logic
   ``r102_novel_signed_jump_discount.py`` used for its own k/floor grid):
   the PRIMARY cell is the GRID-CENTER cell, ``threshold=4.0`` (v4's own
   shipped default, also ``r146_shared``'s function default and the value
   the operator's own pre-dispatch check used), IF it qualifies. If the
   center cell does NOT qualify, the PRIMARY cell is instead the
   qualifying cell closest to 4.0 in absolute threshold distance, ties
   broken by the SMALLER threshold (more conservative: flags MORE bars as
   jumps, i.e. trusts the robust anchor less, not more). If NO cell
   qualifies at all, this file STOPS at Step-0: written up NEGATIVE /
   stopped-at-Step-0, and no promotion-bar code below is run.

3. DRIFT-FREE SELF-CHECK (run on the REAL BTC series, not just
   ``r146_shared``'s own synthetic self-test), before trusting any
   downstream number: for the PRIMARY cell's threshold, verify that (a)
   every UNFLAGGED bar's masked close equals its real close EXACTLY (no
   reconstruction, hence no compounding drift is possible by
   construction), and (b) the flagged fraction is small. This is the
   specific check named in this round's dispatch given that an earlier
   draft of this exact construction (capping-and-reintegrating via cumsum)
   already failed this way once.

4. CAUSAL TRUNCATION PROBE, run before trusting any Step-0 or
   promotion-bar number: ``r146_shared.causal_truncation_probe_vote``
   applied to the PRIMARY cell's own
   ``jump_robust_anchor_vote_frac(df, threshold=primary)`` (already
   covered in the abstract by ``r146_shared``'s own synthetic self-test at
   threshold=4.0; re-run here on the real BTC series, and at whichever
   threshold this file's own selection rule actually picks).

5. PROMOTION BAR (identical shape to R-89/R-93/R-97/R-99/R-101/R-102/
   R-103, reproduced verbatim in ``r146_shared.py``'s own "PROMOTION BAR"
   section -- not re-derived here):
     A2: Step-0 kill switch above (R^2 < 0.98 on the PRIMARY cell).
     B1: paired block-bootstrap (log-growth) via ``compare()``'s
         ``boot_lo``/``boot_hi``/``excludes_zero``, on inner-validation
         (2021-01-01 -> 2022-12-31), BOTH markets -- PASS if
         ``d_sharpe > +0.2`` OR the 95% interval excludes zero.
     B2 (diagnostic only): risk-matched drawdown -- ``exposure_ratio`` /
         ``vol_ratio`` both in [0.9, 1.1].
     B3: plateau -- the OTHER 2 grid cells' inner-validation numbers,
         both markets, reported alongside the primary; PASS requires a
         directionally consistent region, not an isolated spike.
     B4: ETH same-sign falsification on at least one market.
     B5: 0.40% taker-fee-tier re-run (``fee_at(SPOT, 0.004)``), primary
         cell, inner-validation, SPOT (per this round's dispatch: SPOT is
         the required market for B5; FUTURES at the same fee tier is
         additionally reported below for extra rigor, not itself
         required).
   Promote only if A2 does not trip AND B1 passes on >=1 market AND B4
   passes AND B5's edge (if B1 passed) survives in sign -- the exact bar
   ``r146_shared.py`` states; this file does not weaken it after seeing a
   number.

6. WHAT WOULD MAKE THIS FAIL, named now, before any real-data number
   exists (copied near-verbatim from ``r146_shared.py``'s own docstring):
   (a) A2 could trip -- BTC's 5-minute return distribution may simply not
   carry enough single-bar outlier mass at these long (20/40/80-day)
   horizons for excluding a handful of jump bars per window to move the
   anchor materially away from the plain SMA it replaces (Levine &
   Pedersen 2016's linear-filter-equivalence concern, restated for a
   non-linear/exclusion-based construction: it could still simply not
   bind enough to matter at this span). (b) Even if A2 clears, BTC's
   documented INVERSE leverage effect (Baur & Dimpfl 2018, already this
   project's own v3/v4 citation) means large POSITIVE moves, not only
   crashes, drive its volatility -- a jump-robust anchor that excludes
   large moves of EITHER sign could suppress the vote's reaction to a
   genuine bullish breakout as readily as it protects the vote from a
   liquidation-cascade whipsaw. Both are reported honestly below,
   whichever way they come out.

CONFIGURATIONS EVALUATED IN THIS FILE: 3 (Step-0 grid) + 6 (primary cell's
full ``compare()``: 2 BTC slices x 2 markets + ETH x 2 markets) + 4 (other
2 grid cells' inner-validation numbers, both markets, for B3) + 2 (primary
cell at the 0.40% fee tier, SPOT required + FUTURES extra) = 15 total, IF
the Step-0 gate does not stop the branch early.

USAGE
-----
    python experiments/r146_novel_jump_robust_anchor.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r146_shared import (  # noqa: E402
    BARS_PER_DAY,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    OOS_START,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_vote,
    compare,
    fee_at,
    jump_masked_close,
    jump_robust_anchor_vote_frac,
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

# ---------------------------------------------------------- pre-registered
GRID_THRESHOLD = (3.0, 4.0, 5.0)
CENTER_THRESHOLD = 4.0
MAD_SPAN = BARS_PER_DAY  # 288 bars = 1 day, HELD FIXED across the grid
R2_THRESH = 0.98
FLAG_RATE_SANITY = 0.05  # tighter endpoint of the dispatch's "well under 5-10%" band
FEE_TIER = 0.004
SHARPE_NOISE_FLOOR = 0.2


def hdr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (1)
# The mechanism itself: a pure df -> np.ndarray closure over `threshold`,
# built from r146_shared's UNCHANGED jump_robust_anchor_vote_frac / v4_scale
# / apply_deadband.
# ==================================================================

def make_build_target(threshold: float):
    def build(df: pd.DataFrame) -> np.ndarray:
        frac = jump_robust_anchor_vote_frac(df, mad_span=MAD_SPAN, threshold=threshold).to_numpy()
        scale = v4_scale(df)
        desired = frac * scale
        return apply_deadband(desired)

    build.__name__ = f"jump_robust_anchor_thr{threshold:g}"
    return build


# ================================================================== (2)
# Step-0 non-degeneracy grid: A2 (R^2 vs v4's own vote) + flag-rate sanity.
# ==================================================================

def step0_grid(df: pd.DataFrame) -> tuple[list[dict], int]:
    """Computed on the FULL pre-holdout price history (so the 80-day
    anchor has its natural warmup), with both reported statistics
    restricted to bars inside the inner-train window."""
    mask = np.asarray((df.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                       (df.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    ctrl_vote = v4_vote_frac(df).to_numpy()

    rows = []
    for t in GRID_THRESHOLD:
        cand_vote = jump_robust_anchor_vote_frac(df, mad_span=MAD_SPAN, threshold=t).to_numpy()
        masked = jump_masked_close(df, mad_span=MAD_SPAN, threshold=t)
        is_flagged = masked.isna().to_numpy()
        flag_rate_train = float(is_flagged[mask].mean())
        flag_rate_full = float(is_flagged.mean())
        r2 = r_squared(cand_vote[mask], ctrl_vote[mask])
        qualifies = (r2 < R2_THRESH) and (flag_rate_train < FLAG_RATE_SANITY)
        rows.append(dict(threshold=t, r2=r2, flag_rate_train=flag_rate_train,
                          flag_rate_full=flag_rate_full, qualifies=qualifies))
    return rows, n_bars


def select_primary(rows: list[dict]) -> dict | None:
    """Pre-registered selection: grid-center cell (threshold=4.0) if it
    qualifies, else the qualifying cell closest to 4.0 in absolute
    threshold distance, ties broken toward the SMALLER threshold (more
    conservative -- flags more bars, trusts the robust anchor less)."""
    qualifying = [r for r in rows if r["qualifies"]]
    if not qualifying:
        return None
    for r in rows:
        if r["threshold"] == CENTER_THRESHOLD and r["qualifies"]:
            return r

    def dist(r: dict) -> tuple:
        return (abs(r["threshold"] - CENTER_THRESHOLD), r["threshold"])

    return sorted(qualifying, key=dist)[0]


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars; mad_span={MAD_SPAN} bars = 1 day, HELD FIXED)")
    print(f"QUALIFY = r2 < {R2_THRESH} (A2 kill switch) AND "
          f"flag_rate_train < {FLAG_RATE_SANITY:.0%} (sanity: still a RARE-event flag)")
    hdr_line = (f"{'threshold':>9s} {'r2':>8s} {'flag_train':>11s} {'flag_full':>10s} "
                f"{'qualifies':>10s}")
    print(hdr_line)
    print("-" * len(hdr_line))
    for r in rows:
        print(f"{r['threshold']:9.1f} {r['r2']:8.4f} {r['flag_rate_train']:11.4%} "
              f"{r['flag_rate_full']:10.4%} {'YES' if r['qualifies'] else 'no':>10s}")


# ================================================================== (3)
# Drift-free self-check on the REAL BTC series (the specific failure mode
# the operator's earlier draft of this construction already hit once).
# ==================================================================

def drift_free_check(df: pd.DataFrame, threshold: float) -> bool:
    close = df["close"]
    masked = jump_masked_close(df, mad_span=MAD_SPAN, threshold=threshold)
    flagged = masked.isna().to_numpy()
    unflagged = ~flagged
    real_matches = np.allclose(masked.to_numpy()[unflagged], close.to_numpy()[unflagged])
    flag_frac = float(flagged.mean())
    print(f"  threshold={threshold:g}: {int(flagged.sum()):,}/{len(df):,} bars flagged "
          f"({flag_frac:.4%}); every UNFLAGGED bar's masked close == real close: "
          f"{real_matches}")
    print("  (no cumsum / reconstruction anywhere in jump_masked_close -- masking, not "
          "capping-and-reintegrating -- so no one-sided compounding drift is possible by "
          "construction; this reproduces r146_shared's own self-test property on the real "
          "BTC series rather than only synthetic data)")
    return bool(real_matches)


# ================================================================== (4)
# Promotion bar: A2 (above), B1 (bootstrap), B2 (risk-matched dd),
# B3 (plateau), B4 (ETH), B5 (fee tier).
# ==================================================================

def inner_val_rows(build_fn, label: str, btc: pd.DataFrame,
                    markets=(SPOT, FUTURES)) -> list[dict]:
    """Lightweight inner-validation-only comparison (both markets), used
    for the B3 plateau check on the 2 non-primary cells -- avoids the full
    compare() overhead (inner-train + ETH) for cells that are not the
    decision-bearing one."""
    from experiments.r146_shared import INNER_VAL_END, INNER_VAL_START
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r146_{label}")
    rows = []
    for market in markets:
        a = run_slice(cand, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        b = run_slice(ctrl, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        pr = paired_diff(a.daily, b.daily)
        exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                     if b.mean_abs_exposure else float("nan"))
        vol_ratio = (a.realized_vol / b.realized_vol
                     if b.realized_vol else float("nan"))
        risk_matched = (bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                        if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False)
        rows.append(dict(
            label=label, market=market.name,
            d_sharpe=a.sharpe - b.sharpe, d_dd=a.max_drawdown_pct - b.max_drawdown_pct,
            exposure_ratio=exp_ratio, vol_ratio=vol_ratio, risk_matched=risk_matched,
            boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
            excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0),
        ))
    return rows


def print_plateau_table(all_rows: dict[float, list[dict]]) -> None:
    hdr_line = (f"{'threshold':>9s} {'market':>9s} {'dSh':>7s} {'dDD':>7s} "
                f"{'expR':>5s} {'volR':>5s} {'RM':>3s} {'dlogG':>7s} "
                f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr_line)
    print("-" * len(hdr_line))
    for t, rows in all_rows.items():
        for r in rows:
            print(f"{t:9.1f} {r['market']:>9s} {r['d_sharpe']:+7.2f} "
                  f"{r['d_dd']:+7.1f} {r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
                  f"{'Y' if r['risk_matched'] else 'n':>3s} {r['boot_d_loggrowth']:+7.3f} "
                  f"{r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
                  f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def run_promotion_bar(primary: dict, btc: pd.DataFrame, eth: pd.DataFrame,
                       step0_rows: list[dict]) -> dict:
    threshold = primary["threshold"]
    build_primary = make_build_target(threshold)
    label = f"jump_robust_anchor_thr{threshold:g}"

    # --- B1/B2/B4: full compare() over inner_train, inner_val, ETH, both markets.
    hdr(f"PROMOTION BAR -- PRIMARY CELL threshold={threshold:g} (mad_span={MAD_SPAN})")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                   markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_rows_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_rows_primary = [r for r in rows if r["slice"] == "eth_replication"]

    # B1: d_sharpe > +0.2 OR bootstrap interval excludes zero (positive point est.),
    # inner-validation, BOTH markets.
    b1_cells = []
    for r in inner_val_rows_primary:
        passes = (r["excludes_zero"] and r["boot_d_loggrowth"] > 0) or (r["d_sharpe"] > SHARPE_NOISE_FLOOR)
        b1_cells.append(dict(market=r["market"], passes=passes,
                              boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                              d_sharpe=r["d_sharpe"]))
    b1_pass = all(c["passes"] for c in b1_cells)

    # B2: diagnostic only -- drawdown improvement counts only where risk_matched.
    b2_cells = []
    for r in inner_val_rows_primary:
        b2_cells.append(dict(market=r["market"], risk_matched=r["risk_matched"],
                              d_dd=r["d_dd"], voided=not r["risk_matched"]))
    b2_pass = True  # diagnostic only; never itself blocks promotion

    # B3: plateau -- the other 2 grid cells' inner-validation numbers, both markets.
    other_thresholds = [r["threshold"] for r in step0_rows if r["threshold"] != threshold]
    plateau_rows: dict[float, list[dict]] = {}
    for ot in other_thresholds:
        bf = make_build_target(ot)
        olabel = f"jump_robust_anchor_thr{ot:g}"
        plateau_rows[ot] = inner_val_rows(bf, olabel, btc)
    plateau_rows[threshold] = [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                                     d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                                     vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                                     boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                                     boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                                for r in inner_val_rows_primary]
    plateau_rows = dict(sorted(plateau_rows.items()))
    same_sign_flags = [r["d_sharpe"] > 0 for rr in plateau_rows.values() for r in rr]
    b3_pass_directionally_consistent = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0)

    # B4: ETH falsification -- same sign as BTC inner-val, at least one market.
    b4_cells = []
    for r in eth_rows_primary:
        btc_match = next((c for c in inner_val_rows_primary if c["market"] == r["market"]), None)
        same_sign = (btc_match is not None and
                     np.sign(r["d_sharpe"]) == np.sign(btc_match["d_sharpe"]) and
                     r["d_sharpe"] != 0)
        b4_cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                              excludes_zero=r["excludes_zero"],
                              boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                              same_sign_as_btc=same_sign))
    b4_pass = any(c["same_sign_as_btc"] for c in b4_cells)

    # B5: fee tier -- 0.40% taker. SPOT is the required market per this round's
    # dispatch; FUTURES is additionally reported for extra rigor.
    hdr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary cell, inner-validation")
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    fee_rows = inner_val_rows(build_primary, label, btc, markets=fee_markets)
    for r in fee_rows:
        print(f"  {r['market']:>9s}  d_sharpe={r['d_sharpe']:+.3f}  "
              f"boot[{r['boot_lo']:+.3f},{r['boot_hi']:+.3f}]  excl0={r['excludes_zero']}")
    b5_cells = []
    for r in fee_rows:
        base = next((c for c in inner_val_rows_primary if c["market"] == r["market"]), None)
        no_reversal = (base is not None and
                       not (np.sign(r["boot_d_loggrowth"]) != np.sign(base["boot_d_loggrowth"])
                            and r["boot_d_loggrowth"] != 0 and base["boot_d_loggrowth"] != 0))
        b5_cells.append(dict(market=r["market"], boot_d_loggrowth=r["boot_d_loggrowth"],
                              base_boot_d_loggrowth=base["boot_d_loggrowth"] if base else float("nan"),
                              no_reversal=no_reversal))
    b5_spot = next((c for c in b5_cells if c["market"] == SPOT.name), None)
    b5_pass = bool(b5_spot["no_reversal"]) if b5_spot is not None else False

    all_pass = b1_pass and b3_pass_directionally_consistent and b4_pass and (
        b5_pass if b1_pass else True)

    return dict(
        label=label, threshold=threshold,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells, b2_pass=b2_pass,
        b3_plateau_rows=plateau_rows, b3_pass=b3_pass_directionally_consistent,
        b4_cells=b4_cells, b4_pass=b4_pass,
        b5_cells=b5_cells, b5_pass=b5_pass,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 4 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()

    hdr("R-146 NOVEL: JumpRobustAnchorKellyV4 -- Step-0 non-degeneracy grid")
    print("mechanism: each of v4's 3 anchors becomes a rolling MEAN of a jump-MASKED close")
    print("(Lee & Mykland 2008-style causal per-bar jump exclusion, mad_span=288 fixed),")
    print("instead of a plain SMA over every bar. Vote architecture (1% band, latching,")
    print("3-way average) and scale/deadband are byte-identical to v4's own.")

    btc = load_btc()
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")
    assert_no_holdout(btc, "main(): btc")

    step0_rows, n_bars = step0_grid(btc)
    print_step0_table(step0_rows, n_bars)

    primary = select_primary(step0_rows)

    if primary is None:
        hdr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell has both r2 < 0.98 and flag_rate_train < 5%: the jump-robust")
        print("anchor is either a near-exact reparameterization of v4's own vote, or the")
        print("jump test is firing far too often to be a rare-event exclusion, everywhere")
        print("on the pre-registered grid. Per this file's own pre-registration, this")
        print("Step-0 table is the branch's ENTIRE product, written up NEGATIVE /")
        print("stopped-at-Step-0. No promotion-bar code is run, and no data on/after")
        print("2023-01-01 is touched.")
        print(f"\nconfigurations evaluated: 3 (Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {btc.index.max()}  (< {OOS_START})")
        print(f"\n[{time.time()-t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                    n_configs=3)

    print(f"\nPRIMARY CELL SELECTED (non-degeneracy rule only, fixed before any Sharpe/"
          f"performance number was inspected): threshold={primary['threshold']:g}  "
          f"(r2={primary['r2']:.4f}, flag_rate_train={primary['flag_rate_train']:.4%})")
    is_center = (primary["threshold"] == CENTER_THRESHOLD)
    print(f"  selection: {'grid-center cell (4.0) qualified' if is_center else 'grid-center cell did NOT qualify; nearest qualifying cell chosen'}")

    hdr("DRIFT-FREE SELF-CHECK ON REAL BTC DATA (masking, not cap-and-reintegrate)")
    drift_ok = drift_free_check(btc, primary["threshold"])
    print(f"\nDRIFT-FREE CHECK PASSES: {drift_ok}")
    if not drift_ok:
        raise AssertionError("masked close diverges from real close on unflagged bars -- "
                              "this should be impossible by construction; STOP.")

    build_primary = make_build_target(primary["threshold"])

    hdr("CAUSAL-TRUNCATION PROBE (primary cell's own vote, real BTC series)")
    try:
        probe_ok = causal_truncation_probe_vote(
            lambda d: jump_robust_anchor_vote_frac(d, mad_span=MAD_SPAN,
                                                    threshold=primary["threshold"]).to_numpy(),
            btc)
        print(f"  PASS (threshold={primary['threshold']:g})")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    print(f"\nCAUSAL PROBE PASSES: {probe_ok}")

    eth = load_eth()
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")
    assert_no_holdout(eth, "main(): eth")

    bar = run_promotion_bar(primary, btc, eth, step0_rows)

    hdr("A2 -- Step-0 kill switch, re-verified for the PRIMARY cell")
    print(f"  R^2(candidate_vote, v4_vote_frac) on inner-train = {primary['r2']:.4f}  "
          f"(threshold < 0.98 required)  A2_CLEARS={primary['r2'] < R2_THRESH}")

    hdr("B1 -- inner-validation paired block-bootstrap (log-growth), both markets")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  boot[{c['boot_lo']:+.3f},{c['boot_hi']:+.3f}]  "
              f"d_sharpe={c['d_sharpe']:+.3f}  PASS={c['passes']}")
    print(f"B1 PASS (all markets): {bar['b1_pass']}")

    hdr("B2 -- risk-matched drawdown check (diagnostic only; VOID unless risk_matched)")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.1f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hdr("B3 -- plateau, not peak: all 3 grid cells' inner-validation numbers")
    print_plateau_table(bar["b3_plateau_rows"])
    print(f"\nB3 (directionally consistent region, not an isolated spike): {bar['b3_pass']}")

    hdr("B4 -- ETH falsification (same-sign replication, at least one market)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.3f}  "
              f"boot[{c['boot_lo']:+.3f},{c['boot_hi']:+.3f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PASS (>=1 market): {bar['b4_pass']}")

    hdr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign; SPOT required)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS (SPOT): {bar['b5_pass']}")

    hdr("VERDICT")
    a2_clears = primary["r2"] < R2_THRESH
    print(f"A2={a2_clears}  B1={bar['b1_pass']}  B2=diagnostic-only  B3={bar['b3_pass']}  "
          f"B4={bar['b4_pass']}  B5={bar['b5_pass']}  drift_check={drift_ok}  "
          f"causal_probe={probe_ok}")
    all_applicable_pass = (a2_clears and drift_ok and probe_ok and bar["b1_pass"] and
                            bar["b3_pass"] and bar["b4_pass"] and
                            (bar["b5_pass"] if bar["b1_pass"] else True))
    verdict = "PROMOTE-candidate" if all_applicable_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_applicable_pass}")
    print(f"VERDICT: {verdict}")

    n_configs = 3 + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(3 Step-0 grid + 6 primary-cell compare() + 4 plateau (2 cells x 2 markets) "
          f"+ 2 fee-tier)")
    print(f"max timestamp read anywhere in this branch: "
          f"{max(btc.index.max(), eth.index.max())}  (< {OOS_START})")

    print(f"\n[{time.time()-t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary,
                passed_step0=True, drift_ok=drift_ok, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs)


if __name__ == "__main__":
    main()
