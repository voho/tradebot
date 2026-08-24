#!/usr/bin/env python
"""R-102 NOVEL branch: ``SignedJumpDiscountKellyV4`` -- a multiplicative,
Patton & Sheppard (2015)-grounded discount on ``kelly_regime_v4``'s
UNCHANGED ``frac * scale`` product, keyed on the relative signed jump
measure (RSJ) computed from this project's own native 5-minute bars.

Full citation trail, literature grounding, and the "not a duplicate of"
argument against R-93/R-97/R-62/R-87/R-101/R-99/the fifteen INFO-axis
rounds/the parallel conservative branch of this same round are all already
established in ``r102_shared.py``'s own module docstring (read in full
before this file was written) and are not re-derived here.

=====================================================================
PRE-REGISTRATION (frozen before any real-data RSJ, discount, or
backtest number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): Patton & Sheppard (2015) show realized
   variation dominated by its downside component (negative RSJ -- "bad
   volatility") predicts significantly HIGHER future volatility than the
   same amount of upside-dominated variation, an asymmetric-persistence
   effect distinct from (and a refinement of) the ordinary leverage
   effect; if that holds on BTC, discounting exposure specifically when
   RSJ is sharply negative should anticipate the kind of volatility
   escalation v4's own conditional-vol-target gate only reacts to AFTER
   the fact, without touching v4's validated vote/scale architecture at
   all.

2. CONSTRUCTION (exact, applied on top of v4's own UNCHANGED
   ``frac * scale``, never modifying either factor):

       raw[t]       = v4_raw_desired(df)[t]           # frac[t] * scale[t], v4's own, UNCHANGED
       rsj[t]       = relative_signed_jump(df)[t]      # r102_shared, causal, in [-1, 1] or NaN
       discount[t]  = 1.0                                          if rsj[t] >= 0 or not finite
       discount[t]  = clip(1 + k * rsj[t], floor, 1.0)             if rsj[t] < 0
       desired[t]   = raw[t] * discount[t]
       target[t]    = apply_deadband(desired)          # v4's own 10% deadband, AFTER the discount

   NaN RSJ (early-warmup bars, before the EWM downside/upside variances
   have ``min_periods`` history) is treated as "no information" ->
   discount = 1.0, never as 0 and never as an error. Since ``rsj < 0``
   whenever the branch fires, ``1 + k*rsj[t] < 1`` there always, so the
   ``clip(..., floor, 1.0)`` upper bound is inactive by construction and
   exists only to make the boundary explicit; the ``floor`` lower bound is
   the only one that ever binds.

3. STEP-0 NON-DEGENERACY GRID AND SELECTION RULE, PRE-REGISTERED BEFORE
   ANY GRID NUMBER WAS COMPUTED:

   Grid: ``k in {2.0, 4.0, 8.0}`` x ``floor in {0.3, 0.5, 0.7}`` (9 cells,
   fixed a priori). For each cell, on the BTC INNER-TRAIN slice only
   (2017-01-01 -> 2020-12-31), compute ``discount[t]`` and the resulting
   ``target[t]`` over the FULL pre-holdout price history (so every EWM/
   rolling input has its natural warmup available exactly as it would in
   a real backtest), then restrict the two reported statistics to bars
   whose timestamp falls inside the inner-train window:
     - ``bind_frac``  = fraction of those bars with ``discount < 0.95``
       (the mechanism is actually binding, not a no-op);
     - ``r2``         = ``r_squared(target, v4_target(df))`` restricted to
       the same bars (is the candidate's exposure path a near-exact
       rescale of v4's own, i.e. degenerate).
   A cell QUALIFIES iff ``bind_frac > 0.01`` (binds on a non-trivial
   fraction of bars) AND ``r2 < 0.98`` (clearly not a rescale of v4's own
   path). Both thresholds and the grid are named here, before running,
   exactly as specified by this round's dispatch.

   SELECTION RULE (non-degeneracy ONLY, no performance number is
   inspected before this rule is applied): the PRIMARY cell is the
   GRID-CENTER cell, ``k=4.0, floor=0.5``, IF it qualifies -- an a priori,
   non-cherry-picked choice, the same "center of an a priori grid" logic
   R-96's novel branch used for its own one decision-bearing cell. If the
   center cell does NOT qualify, the PRIMARY cell is instead the
   qualifying cell closest to the center in grid-index distance
   (``sqrt((log2(k) - log2(4))^2 + (floor - 0.5)^2)``), ties broken by
   smallest ``k`` then smallest ``floor``. If NO cell qualifies at all,
   this file STOPS at Step-0: the mechanism is a no-op or a rescale of
   v4's own path everywhere on the grid, the branch's entire product is
   this Step-0 table, written up NEGATIVE / stopped-at-Step-0, and no
   promotion-bar code below is run.

4. CAUSAL TRUNCATION PROBE, run before trusting any Step-0 or
   promotion-bar number: ``r102_shared.causal_truncation_probe_series``
   applied to (a) ``relative_signed_jump`` itself (already covered by
   ``r102_shared``'s own self-test, re-run here on the real BTC series
   rather than synthetic data) and (b) this file's own
   ``build_target`` for the PRIMARY cell (a single ``df -> np.ndarray``
   closure over ``k, floor``), which composes ``v4_raw_desired``,
   ``relative_signed_jump`` and ``apply_deadband`` -- all three already
   individually verified causal by ``r102_shared``'s self-test, checked
   here again as the COMPOSED pipeline actually shipped.

5. PROMOTION BAR (docs/ROUTINE.md's own bar, operationalized for this
   round exactly as specified in this round's dispatch -- frozen BEFORE
   any B1-B5 number below was computed):
     B1: paired block-bootstrap (log-growth) via ``compare()``'s
         ``boot_lo``/``boot_hi``/``excludes_zero``, on inner-validation
         (2021-01-01 -> 2022-12-31), BOTH markets -- PASS if a positive
         point estimate excludes zero, OR ``d_sharpe > +0.2`` (this
         project's own +-0.2 Sharpe noise floor, R-20).
     B2: risk-matched drawdown check -- a drawdown improvement counts
         ONLY where ``risk_matched`` is True (``exposure_ratio`` AND
         ``vol_ratio`` both in [0.9, 1.1]); otherwise the drawdown number
         is VOID (standing rule: less exposure trivially draws down
         less).
     B3: plateau, not peak -- the OTHER 8 grid cells' inner-validation
         numbers are reported alongside the primary; PASS requires a
         directionally consistent region, not an isolated spike at the
         one selected cell.
     B4: ETH falsification -- same-sign replication on the Bitfinex ETH
         series (``compare(..., include_eth=True)``).
     B5: fee-tier survival -- ``fee_at(SPOT, 0.004)`` /
         ``fee_at(FUTURES, 0.004)`` (a 0.40% taker tier) on the PRIMARY
         cell, inner-validation slice; PASS if the sign of the paired
         difference does not reverse relative to the standard-fee result.
   PROMOTE-candidate only if ALL clauses that apply hold; otherwise
   NEGATIVE. This file evaluates only up to the inner-validation /
   ETH-replication boundary -- OOS_START (2023-01-01) is never read.

6. WHAT WOULD MAKE THIS FAIL, named now, before any real-data number
   exists (copied near-verbatim from ``r102_shared.py``'s own docstring):
   Patton & Sheppard's asymmetric-persistence result is from equity index
   and single-stock DAILY data; BTC's own realized-variance dynamics may
   not share it (B-42 already found one closed-form equity/futures result
   fail to transfer to BTC). If RSJ is nearly always near zero or rarely
   very negative on BTC (a Step-0 DEGENERACY finding: low ``bind_frac``),
   or if it discounts during ordinary bidirectional volatility clustering
   that carries no forward information (a Step-0 finding that would show
   up as a qualifying cell with no B1-B5 edge), the mechanism is inert or
   adds turnover without benefit. This is reported honestly below,
   whichever way it comes out.

CONFIGURATIONS EVALUATED IN THIS FILE: 9 (Step-0 grid) + 6 (primary
cell's full ``compare()``: 2 BTC slices x 2 markets + ETH x 2 markets)
+ 16 (the other 8 grid cells' inner-validation numbers, both markets,
for the B3 plateau check) + 2 (primary cell at the 0.40% fee tier, both
markets) = 33 total, IF the Step-0 gate does not stop the branch early.

USAGE
-----
    python experiments/r102_novel_signed_jump_discount.py
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

from experiments.r102_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    compare,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    relative_signed_jump,
    run_slice,
    v4_raw_desired,
    v4_target,
)

# ---------------------------------------------------------- pre-registered
GRID_K = (2.0, 4.0, 8.0)
GRID_FLOOR = (0.3, 0.5, 0.7)
CENTER_K, CENTER_FLOOR = 4.0, 0.5
BIND_FRAC_THRESH = 0.01
R2_THRESH = 0.98
FEE_TIER = 0.004
SHARPE_NOISE_FLOOR = 0.2


def hdr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (1)
# The mechanism itself: a pure function of df, built from r102_shared's
# UNCHANGED v4_raw_desired / relative_signed_jump / apply_deadband.
# ==================================================================

def rsj_discount(rsj: np.ndarray, k: float, floor: float) -> np.ndarray:
    """``discount[t] = 1.0`` if ``rsj[t] >= 0`` or not finite; else
    ``clip(1 + k*rsj[t], floor, 1.0)``. Since ``rsj[t] < 0`` on the branch
    that fires, ``1 + k*rsj[t] < 1`` there always -- the upper clip bound
    is never active, only the lower ``floor`` bound is."""
    rsj = np.asarray(rsj, dtype=float)
    disc = np.ones(len(rsj), dtype=float)
    neg = np.isfinite(rsj) & (rsj < 0.0)
    disc[neg] = np.clip(1.0 + k * rsj[neg], floor, 1.0)
    return disc


def make_build_target(k: float, floor: float):
    """A pure ``df -> np.ndarray`` closure: v4's own unchanged
    ``frac * scale``, discounted by the RSJ mechanism, then v4's own
    deadband applied AFTER the discount."""

    def build(df: pd.DataFrame) -> np.ndarray:
        raw = v4_raw_desired(df)
        rsj = relative_signed_jump(df)
        disc = rsj_discount(rsj, k, floor)
        desired = raw * disc
        return apply_deadband(desired)

    build.__name__ = f"signed_jump_discount_k{k:g}_floor{floor:g}"
    return build


# ================================================================== (2)
# Step-0 non-degeneracy grid.
# ==================================================================

def step0_grid(df: pd.DataFrame) -> list[dict]:
    """9-cell grid, computed on the FULL pre-holdout price history (so
    every EWM/rolling input has its natural warmup), with the two
    reported statistics restricted to bars inside the inner-train
    window."""
    mask = np.asarray((df.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                       (df.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    raw = v4_raw_desired(df)
    rsj = relative_signed_jump(df)
    ctrl_target = v4_target(df)

    rows = []
    for k in GRID_K:
        for floor in GRID_FLOOR:
            disc = rsj_discount(rsj, k, floor)
            desired = raw * disc
            target = apply_deadband(desired)
            bind_frac = float(np.mean(disc[mask] < 0.95))
            r2 = r_squared(target[mask], ctrl_target[mask])
            qualifies = (bind_frac > BIND_FRAC_THRESH) and (r2 < R2_THRESH)
            rows.append(dict(k=k, floor=floor, bind_frac=bind_frac, r2=r2,
                              qualifies=qualifies))
    return rows, n_bars


def select_primary(rows: list[dict]) -> dict | None:
    """Pre-registered selection: grid-center cell if it qualifies, else
    the qualifying cell closest to the center in grid-index distance,
    ties broken by smallest k then smallest floor. None if nothing
    qualifies."""
    qualifying = [r for r in rows if r["qualifies"]]
    if not qualifying:
        return None
    for r in rows:
        if r["k"] == CENTER_K and r["floor"] == CENTER_FLOOR and r["qualifies"]:
            return r

    def dist(r: dict) -> tuple:
        d = ((np.log2(r["k"]) - np.log2(CENTER_K)) ** 2 +
             (r["floor"] - CENTER_FLOOR) ** 2) ** 0.5
        return (d, r["k"], r["floor"])

    return sorted(qualifying, key=dist)[0]


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars)")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r2 < {R2_THRESH}")
    hdr_line = f"{'k':>6s} {'floor':>6s} {'bind_frac':>10s} {'r2':>8s} {'qualifies':>10s}"
    print(hdr_line)
    print("-" * len(hdr_line))
    for r in rows:
        print(f"{r['k']:6.1f} {r['floor']:6.2f} {r['bind_frac']:10.4f} {r['r2']:8.4f} "
              f"{'YES' if r['qualifies'] else 'no':>10s}")


# ================================================================== (3)
# Causal truncation probes.
# ==================================================================

def run_causal_probes(df: pd.DataFrame, build_primary) -> tuple[bool, bool]:
    print("\ncausal_truncation_probe_series(relative_signed_jump, df):")
    try:
        ok_rsj = causal_truncation_probe_series(relative_signed_jump, df)
        print("  PASS")
    except AssertionError as e:
        ok_rsj = False
        print(f"  FAIL: {e}")

    print("\ncausal_truncation_probe_series(build_target[primary], df):")
    try:
        ok_build = causal_truncation_probe_series(build_primary, df)
        print("  PASS")
    except AssertionError as e:
        ok_build = False
        print(f"  FAIL: {e}")

    return ok_rsj, ok_build


# ================================================================== (4)
# Promotion bar: B1 (bootstrap), B2 (risk-matched dd), B3 (plateau),
# B4 (ETH), B5 (fee tier).
# ==================================================================

def inner_val_rows(build_fn, label: str, btc: pd.DataFrame,
                    markets=(SPOT, FUTURES)) -> list[dict]:
    """Lightweight inner-validation-only comparison (both markets), used
    for the B3 plateau check on the 8 non-primary cells -- avoids the
    full compare() overhead (inner-train + ETH) for cells that are not
    the decision-bearing one."""
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r102_{label}")
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


def print_plateau_table(all_rows: dict[tuple, list[dict]]) -> None:
    hdr_line = (f"{'k':>6s} {'floor':>6s} {'market':>9s} {'dSh':>7s} {'dDD':>7s} "
                f"{'expR':>5s} {'volR':>5s} {'RM':>3s} {'dlogG':>7s} "
                f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr_line)
    print("-" * len(hdr_line))
    for (k, floor), rows in all_rows.items():
        for r in rows:
            print(f"{k:6.1f} {floor:6.2f} {r['market']:>9s} {r['d_sharpe']:+7.2f} "
                  f"{r['d_dd']:+7.1f} {r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
                  f"{'Y' if r['risk_matched'] else 'n':>3s} {r['boot_d_loggrowth']:+7.3f} "
                  f"{r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
                  f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def run_promotion_bar(primary: dict, btc: pd.DataFrame, eth: pd.DataFrame,
                       step0_rows: list[dict]) -> dict:
    k, floor = primary["k"], primary["floor"]
    build_primary = make_build_target(k, floor)
    label = f"signed_jump_discount_k{k:g}_floor{floor:g}"

    # --- B1/B2/B4: full compare() over inner_train, inner_val, ETH, both markets.
    hdr(f"PROMOTION BAR -- PRIMARY CELL k={k:g}, floor={floor:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                   markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_rows_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_rows_primary = [r for r in rows if r["slice"] == "eth_replication"]
    inner_train_rows_primary = [r for r in rows if r["slice"] == "inner_train"]

    # B1: positive point estimate excluding zero, OR d_sharpe > +0.2, on
    # inner-validation, BOTH markets.
    b1_cells = []
    for r in inner_val_rows_primary:
        passes = (r["excludes_zero"] and r["boot_d_loggrowth"] > 0) or (r["d_sharpe"] > SHARPE_NOISE_FLOOR)
        b1_cells.append(dict(market=r["market"], passes=passes,
                              boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                              d_sharpe=r["d_sharpe"]))
    b1_pass = all(c["passes"] for c in b1_cells)

    # B2: drawdown improvement counts only where risk_matched.
    b2_cells = []
    for r in inner_val_rows_primary:
        dd_improves = r["d_dd"] > 0  # d_dd = cand_dd - ctrl_dd; drawdown pct convention below
        b2_cells.append(dict(market=r["market"], risk_matched=r["risk_matched"],
                              d_dd=r["d_dd"], voided=not r["risk_matched"]))
    # B2 does not gate promotion on its own (a drawdown property, not required to hold);
    # it is reported as a risk-matched diagnostic per docs/ROUTINE.md's standing rule.
    b2_pass = True  # never itself blocks promotion; only voids the dd number if unmatched

    # B3: plateau -- the other 8 grid cells' inner-validation numbers, both markets.
    other_cells = [(r["k"], r["floor"]) for r in step0_rows
                   if not (r["k"] == k and r["floor"] == floor)]
    plateau_rows = {}
    for ok, ofloor in other_cells:
        bf = make_build_target(ok, ofloor)
        olabel = f"signed_jump_discount_k{ok:g}_floor{ofloor:g}"
        plateau_rows[(ok, ofloor)] = inner_val_rows(bf, olabel, btc)
    plateau_rows[(k, floor)] = [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                                      d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                                      vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                                      boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                                      boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                                 for r in inner_val_rows_primary]
    same_sign_as_primary = [r["d_sharpe"] > 0 for cell, rr in plateau_rows.items() for r in rr]
    n_same_sign = sum(1 for r in inner_val_rows_primary if r["d_sharpe"] > 0)
    b3_pass_directionally_consistent = (sum(same_sign_as_primary) >= len(same_sign_as_primary) / 2.0)

    # B4: ETH falsification -- same sign as BTC inner-val, both markets.
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
    b4_pass = all(c["same_sign_as_btc"] for c in b4_cells)

    # B5: fee tier -- 0.40% taker, primary cell, inner-val, both markets.
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
    b5_pass = all(c["no_reversal"] for c in b5_cells)

    all_pass = b1_pass and b2_pass and b3_pass_directionally_consistent and b4_pass and b5_pass

    return dict(
        label=label, k=k, floor=floor,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells, b2_pass=b2_pass,
        b3_plateau_rows=plateau_rows, b3_pass=b3_pass_directionally_consistent,
        b4_cells=b4_cells, b4_pass=b4_pass,
        b5_cells=b5_cells, b5_pass=b5_pass,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 16 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()

    hdr("R-102 NOVEL: SignedJumpDiscountKellyV4 -- Step-0 non-degeneracy grid")
    print("mechanism: Patton & Sheppard (2015) RSJ-keyed multiplicative discount on")
    print("v4's own UNCHANGED frac*scale; discount fires only when RSJ < 0 (downside-")
    print("dominated realized variation), never touching vote or scale.")

    btc = load_btc()
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")
    assert_no_holdout(btc, "main(): btc")

    step0_rows, n_bars = step0_grid(btc)
    print_step0_table(step0_rows, n_bars)

    primary = select_primary(step0_rows)

    if primary is None:
        hdr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell has both bind_frac > 1% and r2 < 0.98: the RSJ discount is")
        print("either a near-total no-op or a near-exact rescale of v4's own path")
        print("everywhere on the pre-registered grid. Per this file's own")
        print("pre-registration, this Step-0 table is the branch's ENTIRE product,")
        print("written up NEGATIVE / stopped-at-Step-0. No promotion-bar code is run,")
        print("and no data on/after 2023-01-01 is touched.")
        print(f"\nconfigurations evaluated: 9 (Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {btc.index.max()}  (< {OOS_START})")
        print(f"\n[{time.time()-t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                    n_configs=9)

    print(f"\nPRIMARY CELL SELECTED (non-degeneracy rule only): "
          f"k={primary['k']:g}, floor={primary['floor']:g}  "
          f"(bind_frac={primary['bind_frac']:.4f}, r2={primary['r2']:.4f})")
    is_center = (primary["k"] == CENTER_K and primary["floor"] == CENTER_FLOOR)
    print(f"  selection: {'grid-center cell qualified' if is_center else 'grid-center cell did NOT qualify; nearest qualifying cell chosen'}")

    build_primary = make_build_target(primary["k"], primary["floor"])

    hdr("CAUSAL-TRUNCATION PROBES")
    probe_rsj_ok, probe_build_ok = run_causal_probes(btc, build_primary)
    probes_pass = probe_rsj_ok and probe_build_ok
    print(f"\nBOTH PROBES PASS: {probes_pass}")

    eth = load_eth()
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")
    assert_no_holdout(eth, "main(): eth")

    bar = run_promotion_bar(primary, btc, eth, step0_rows)

    hdr("B1 -- inner-validation paired block-bootstrap (log-growth), both markets")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  boot[{c['boot_lo']:+.3f},{c['boot_hi']:+.3f}]  "
              f"d_sharpe={c['d_sharpe']:+.3f}  PASS={c['passes']}")
    print(f"B1 PASS (all markets): {bar['b1_pass']}")

    hdr("B2 -- risk-matched drawdown check (VOID unless risk_matched)")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.1f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hdr("B3 -- plateau, not peak: all 9 grid cells' inner-validation numbers")
    print_plateau_table(bar["b3_plateau_rows"])
    print(f"\nB3 (directionally consistent region, not an isolated spike): {bar['b3_pass']}")

    hdr("B4 -- ETH falsification (same-sign replication)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.3f}  "
              f"boot[{c['boot_lo']:+.3f},{c['boot_hi']:+.3f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PASS: {bar['b4_pass']}")

    hdr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    hdr("VERDICT")
    print(f"B1={bar['b1_pass']}  B2=diagnostic-only  B3={bar['b3_pass']}  "
          f"B4={bar['b4_pass']}  B5={bar['b5_pass']}")
    all_applicable_pass = bar["b1_pass"] and bar["b3_pass"] and bar["b4_pass"] and bar["b5_pass"]
    verdict = "PROMOTE-candidate" if all_applicable_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_applicable_pass}")
    print(f"VERDICT: {verdict}")

    n_configs = 9 + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(9 Step-0 grid + 6 primary-cell compare() + 16 plateau (8 cells x 2 markets) "
          f"+ 2 fee-tier)")
    print(f"max timestamp read anywhere in this branch: "
          f"{max(btc.index.max(), eth.index.max())}  (< {OOS_START})")

    print(f"\n[{time.time()-t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary,
                passed_step0=True, probe_rsj_ok=probe_rsj_ok, probe_build_ok=probe_build_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs)


if __name__ == "__main__":
    main()
