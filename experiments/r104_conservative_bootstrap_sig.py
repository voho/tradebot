#!/usr/bin/env python
"""R-104 CONSERVATIVE branch: ``BootstrapSigKellyV4`` -- ``kelly_regime_v4``'s
own unchanged ``frac * scale`` product, multiplied by a PERIODIC (batch,
infrequent) Monte Carlo stationary block-bootstrap (Politis & Romano 1994)
significance discount of the vote's own historical daily edge, before v4's
own 10% deadband is applied. Full citation trail, literature grounding, the
axis this attacks (ERR -- no error control anywhere in the signal path),
and the exhaustive non-duplication argument against every related prior
round (R-28/R-31, R-87, R-101, R-97, every SIZE-axis round) all live in
``experiments/r104_shared.py``'s own module docstring (read in full before
this file was written); none of that is re-derived here. This file also
does not edit, and never reads a bar at or after ``OOS_START`` from, that
module or any other file under ``experiments/`` or ``src/``.

=====================================================================
PRE-REGISTRATION (frozen before any real-data discount, bind_frac, R^2, or
backtest number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): the vote's own historically realized daily
   edge (``vote_only_daily_log_returns`` -- a 1-bar-lagged, fee-free
   reference P&L of holding v4's own latched vote alone) is fed, every
   ``refit_every_days`` days, into a stationary block-bootstrap estimate of
   its own mean and standard error; the resulting t-statistic is converted
   to a discount in ``[floor, 1.0]`` via a fixed, hand-set linear ramp
   (``significance_ramp``: t<=1 -> floor, t>=2 -> 1.0, linear between), held
   constant between refits (a periodic, batch estimator -- Monte Carlo
   bootstrap is comparatively expensive, so it is refit infrequently, unlike
   this round's NOVEL sibling's continuous closed-form HAC/PSR estimator),
   broadcast onto every bar of the day it governs, and multiplied onto v4's
   UNCHANGED ``frac * scale`` before v4's own deadband is applied.

2. CONSTRUCTION (exact; ``r104_shared.py`` already implements every
   primitive below verbatim -- this file only composes them):

       daily[t]        = vote_only_daily_log_returns(df)[t]        # r104_shared
       disc_daily[t]   = expanding_bootstrap_discount(daily, floor=floor,
                             refit_every_days=refit_every_days,
                             min_days=min_days)[t]                  # r104_shared
       disc_bars[t]    = broadcast_daily_to_bars(disc_daily, df.index)[t]
       raw[t]          = v4_raw_desired(df)[t] * disc_bars[t]       # frac*scale, UNCHANGED, then discounted
       target[t]       = apply_deadband(raw)[t]                     # v4's own deadband, AFTER discount

   DEFAULT/PRIMARY CONFIG (pre-registered, not changed after seeing any
   result): ``floor=0.5, refit_every_days=90, min_days=120`` -- all three
   are ``r104_shared.py``'s own pre-registered constants
   (``REFIT_DAYS_DEFAULT=90``, ``MIN_DAYS=120``); ``floor=0.5`` is this
   file's own grid-centre choice, confirmed or overridden only by the
   Step-0 selection rule below, never by a performance number.
   ``mean_block=MEAN_BLOCK_DAYS=30.0`` and ``n_boot=N_BOOT=500`` (the
   bootstrap's own hyperparameters) are ``r104_shared.py``'s fixed
   constants and are not swept anywhere in this file.

3. SCOPE CHOICE, DISCLOSED AND DELIBERATELY *NOT* ACTED ON BY INFLATING
   ``warmup`` -- a documented trap this file avoids by name. The shared
   ``TargetStrategy`` class defaults to an 80-calendar-day warmup
   (``80 * BARS_PER_DAY + 10`` bars), and ``tradebot.window.run_period``'s
   own ``prefix_bars = min(start_pos, warmup)`` hands ``build_target`` only
   that much history before a period's start when it is available -- less
   than this file's own ``MIN_DAYS=120`` alone, so the discount is still
   inside its own burn-in for roughly the first 40 days of every
   non-``inner_train`` slice, and spends the remainder of that slice
   estimating significance from a LOCALLY RESTARTED history (roughly
   80-700 days, depending how far into the slice), not the full
   continuous-since-2017 record the mechanism's own "has the vote's edge
   had time to become significant" framing implies.

   An earlier draft of this file tried to fix that by patching
   ``TargetStrategy.warmup`` to a large sentinel value before calling
   ``compare()``, on the theory that a bigger ``warmup`` simply buys a
   longer prefix. It does not: ``strategy.warmup`` is OVERLOADED in this
   codebase to mean two unrelated things at once -- (1) how many prior bars
   ``run_period`` hands the strategy as prefix (``prefix_bars``, above),
   AND (2) the frame-relative bar index at which ``tradebot.engine.
   run_backtest`` starts calling ``on_bar`` AT ALL
   (``if i >= strategy.warmup: strategy.on_bar(ctx)`` -- gating the call
   itself, not merely order placement). Inflating ``warmup`` to buy (1) a
   long prefix for ``inner_val`` simultaneously breaks (2) for every slice
   whose OWN frame is shorter than that sentinel: with ``warmup`` set to
   2,500 days (720,000 bars) against a 2,190-day (631,008-bar) pre-holdout
   BTC record, ``i >= strategy.warmup`` was never true anywhere in ANY
   slice's frame, so ``on_bar`` never fired at all -- 0 trades, exactly the
   flat "$1,000.00 unchanged" outcome for BOTH the candidate AND
   ``kelly_regime_v4`` on every cell, on the first real run of this file.
   This exact failure mode is already named in this project's own history
   -- R-101's own module docstring records hitting it first ("An earlier
   version of this file set warmup to a huge sentinel... 0 trades on every
   configuration"), and R-103's novel (RLS) branch names it again
   explicitly as the reason it does NOT inflate ``warmup`` either. R-101's
   fix (a bespoke evaluation frame built outside ``run_period`` entirely)
   is not available here without abandoning the shared ``compare()`` /
   ``run_slice()`` machinery this round's own construction explicitly calls
   for; this file instead follows R-103 novel's own precedent -- leave
   ``warmup`` at v4's shipped default (unpatched, untouched) and DISCLOSE
   the resulting local-restart scope limitation plainly, rather than
   papering over it with a change that silently breaks every slice's own
   ``on_bar`` gate. Concretely: ``inner_train`` is unaffected either way (it
   starts at the dataset's true beginning, so there is no meaningful prefix
   to give it regardless of ``warmup``); ``inner_val``/``eth_replication``/
   the B3 and B5 cells all see the discount's expanding-window state
   effectively RESTART close to each slice's own start (net of the default
   80-day prefix), not carry forward continuously from 2017. This is named
   here, before any B1-B5 number was computed, as a real, disclosed
   limitation of measuring this mechanism through the shared harness
   as-is -- not a defect discovered after the fact.

4. STEP-0 NON-DEGENERACY GRID AND SELECTION RULE, PRE-REGISTERED BEFORE ANY
   GRID NUMBER WAS COMPUTED (mirrors R-102/R-103's own grid shape and
   selection-rule convention):

   Grid: ``floor in {0.3, 0.5, 0.7}`` at the default ``refit_every_days=90``
   (3 cells, fixed a priori). For each cell, on BTC's inner-train window
   (2017-01-01 -> 2020-12-31) only:
     - ``bind_frac`` = fraction of inner-train bars where the discount
       (``disc_bars``, broadcast onto the FULL BTC frame exactly as
       ``compare()``'s own ``TargetStrategy.prepare`` would compute it) is
       strictly less than ``1.0 - 1e-9`` (the mechanism is actually
       binding, not a no-op).
     - ``r_sq`` = ``r_squared(build_target(btc, floor=f), v4_target(btc))``,
       both computed over the FULL BTC frame then boolean-masked to the
       inner-train date range before the R^2 is taken (is the candidate's
       exposure path a near-exact rescale of v4's own, i.e. degenerate).
   A cell QUALIFIES iff ``bind_frac > 0.01`` AND ``r_sq < 0.98``.

   SELECTION RULE (non-degeneracy ONLY -- no performance number is
   inspected before this rule is applied): the PRIMARY cell is
   ``floor=0.5`` (grid centre) if it qualifies; otherwise the nearest
   qualifying cell in the pre-registered order ``[0.5, 0.3, 0.7]``. If NONE
   of the three cells qualify, this file STOPS at Step-0: per
   ``r104_shared.py``'s own "WHAT WOULD MAKE THIS FAIL" paragraph, this is
   the pre-registered, R-87-shaped "inert" failure mode -- a legitimate,
   informative NEGATIVE result, not a bug to route around. No B1-B5 code
   runs in that case, and no bar on/after ``OOS_START`` (2023-01-01) is
   ever touched either way.

5. CAUSAL TRUNCATION PROBE, run before trusting any Step-0 or
   promotion-bar number: ``r104_shared.causal_truncation_probe_series``
   (itself re-exported from ``r102_shared``) applied to this file's own
   composed ``build_target`` closure at the PRIMARY config, on real BTC
   data. ``r104_shared.py``'s own self-test already checked
   ``vote_only_daily_log_returns`` / ``expanding_bootstrap_discount`` /
   ``broadcast_daily_to_bars`` on synthetic data with an explicit
   perturbation check; this is this file's own independent verification of
   the FULL composed pipeline (including ``v4_raw_desired`` and
   ``apply_deadband``) on the real, non-synthetic BTC series.

6. PROMOTION BAR (docs/ROUTINE.md's own bar, operationalized exactly as
   every SIZE-axis round since R-89 has used, via ``r102_shared.compare()``
   unchanged -- frozen BEFORE any B1-B5 number below was computed):
     B1 (gating): on ``inner_val``, BOTH markets -- ``d_sharpe > +0.2``
        (R-20's own noise floor) OR the paired block-bootstrap interval
        excludes zero on the positive side (``boot_lo > 0``). Both
        markets' exact numbers reported.
     B2 (diagnostic ONLY, never itself gates promotion): ``d_dd`` and
        ``risk_matched`` (``exposure_ratio`` AND ``vol_ratio`` both in
        ``[0.9, 1.1]``) on ``inner_val``, both markets -- read specifically
        so this round's own headline number cannot repeat R-28's
        unmatched-exposure mistake unnoticed.
     B3 (plateau, not peak, gating): sweep
        ``refit_every_days in {30, 90, 180}`` at the selected primary
        floor, ``inner_val`` only, both markets (3 configs x 2 markets = 6
        cells). This file uses a LIGHTER helper (``inner_val_cells``,
        direct ``run_slice`` + ``paired_diff`` calls, the same idiom
        ``r103_conservative_causal_ols.py``'s own ``inner_val_rows``
        uses) rather than three full ``compare()`` calls, to avoid
        recomputing the ``inner_train``/``eth_replication`` slices three
        extra times for no purpose -- documented here as the chosen
        approach. The ``refit_every_days=90`` cell is NOT recomputed by
        this helper; it is read directly from the primary ``compare()``
        call's own ``inner_val`` rows (identical construction, so
        recomputing it would only reproduce the same numbers under a new
        bootstrap seed for no diagnostic value) -- still counted as 2 of
        the grid's 6 cells per the pre-registered count. PASS requires a
        directionally consistent (same-sign-majority) region across all 6
        cells, not an isolated spike at 90 days.
     B4 (ETH falsification, PRE-REGISTERED, not changed after seeing
        results, gating in its FULL form): ``eth_replication`` rows must
        show the SAME SIGN of ``d_sharpe`` as BTC's own ``inner_val``
        ``d_sharpe``. At least one market matching counts as a PARTIAL
        pass; both markets matching is a FULL pass. Both numbers reported
        exactly, explicitly labelled which.
     B5 (cost robustness, gating): at the selected primary floor,
        ``refit_every_days=90``, BTC ``inner_val`` re-run at a 0.40% taker
        fee tier (``fee_at(SPOT, 0.0040)`` / ``fee_at(FUTURES, 0.0040)``),
        via the same ``inner_val_cells`` helper (the idiom
        ``r102_conservative_downside_vol.py``'s and
        ``r103_conservative_causal_ols.py``'s own B5 sections use: direct
        ``run_slice`` + ``paired_diff`` against a fee-adjusted
        ``MarketSpec``, not a full re-run of ``compare()``). PASS iff the
        SIGN of ``boot_d_loggrowth`` does not reverse relative to the
        standard-fee (0.10%) result on the same market.
   PROMOTE-candidate only if Step-0 selects a primary AND the causal probe
   passes AND B1 AND B3 AND B4 (full form) AND B5 all hold (B2 is
   diagnostic-only). Default: NEGATIVE. This file never reads or reports
   any bar at or after ``OOS_START`` (2023-01-01) regardless of outcome --
   that decision belongs to the operator.

7. WHAT WOULD MAKE THIS FAIL: named already, in full, in ``r104_shared.py``
   itself (three specific, independent failure shapes: the R-87-shaped
   inert-discount pattern this file's own Step-0 kill switch is built to
   catch; a genuinely non-degenerate but too-small-or-too-late discount,
   the R-97/R-101 "real but inert in practice" pattern B1 is built to
   catch; and this branch's OWN specific risk relative to its NOVEL
   sibling -- a periodic batch refit lagging a genuine shift in
   significance by up to its own ``refit_every_days`` cadence, which B3's
   plateau sweep is built to surface as a measurable batch-vs-continuous
   difference). Not re-derived here; reported honestly, whichever way it
   comes out, in the results below.

CONFIGURATIONS EVALUATED IN THIS FILE (IF Step-0 selects a primary): 3
(Step-0 floor grid) + 6 (primary config's full ``compare()``: inner_train x2
markets + inner_val x2 markets + eth_replication x2 markets) + 6 (B3's
refit_every_days grid, 3 configs x 2 markets -- 2 of the 6 reused directly
from the primary ``compare()``'s own inner_val rows, 4 freshly computed) + 2
(B5's 0.40% fee tier, 2 markets) = 17 total. IF Step-0 finds no qualifying
cell, this file stops after the 3 Step-0 cells and reports that outcome
directly (no B1-B5 code runs).

----------------------------------------------------------------------
Run: PYTHONPATH=<repo_root> python3 experiments/r104_conservative_bootstrap_sig.py
(from the repo root, with the project venv active if one is used)
----------------------------------------------------------------------
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

from experiments.r104_shared import (  # noqa: E402
    BARS_PER_DAY,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MIN_DAYS,
    OOS_START,
    REFIT_DAYS_DEFAULT,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    broadcast_daily_to_bars,
    causal_truncation_probe_series,
    compare,
    expanding_bootstrap_discount,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_target,
    vote_only_daily_log_returns,
)

# ---------------------------------------------------------- pre-registered
GRID_FLOOR = (0.3, 0.5, 0.7)
CENTER_FLOOR = 0.5
SELECTION_ORDER = (0.5, 0.3, 0.7)
BIND_FRAC_THRESH = 0.01
R2_THRESH = 0.98
SHARPE_NOISE_FLOOR = 0.2
FEE_TIER = 0.0040
B3_REFIT_GRID = (30, 90, 180)

# Scope choice (see pre-registration item 3 above): TargetStrategy.warmup is
# LEFT AT ITS SHARED DEFAULT (v4's own 80-day value) -- NOT inflated. See
# item 3 for why inflating it is a documented trap (R-101, R-103 novel):
# the same attribute also gates when tradebot.engine.run_backtest starts
# calling on_bar at all, so a large sentinel silences on_bar entirely on
# any slice shorter than the sentinel. The resulting local-restart scope
# limitation on inner_val/eth_replication/B3/B5 is disclosed, not patched
# around.


def hr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (1)
# The mechanism itself: v4's own UNCHANGED frac*scale, multiplied by the
# periodic bootstrap significance discount, deadband applied AFTER.
# ==================================================================

def build_target(df: pd.DataFrame, floor: float = CENTER_FLOOR,
                 refit_every_days: int = REFIT_DAYS_DEFAULT,
                 min_days: int = MIN_DAYS) -> np.ndarray:
    daily = vote_only_daily_log_returns(df)
    disc_daily = expanding_bootstrap_discount(daily, floor=floor,
                                              refit_every_days=refit_every_days,
                                              min_days=min_days)
    disc_bars = broadcast_daily_to_bars(disc_daily, df.index)
    raw = v4_raw_desired(df) * disc_bars
    return apply_deadband(raw)


def build_target_with_disc(df: pd.DataFrame, floor: float,
                           refit_every_days: int = REFIT_DAYS_DEFAULT,
                           min_days: int = MIN_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Same computation as build_target, also returning the bar-broadcast
    discount path itself -- needed for Step-0's bind_frac diagnostic."""
    daily = vote_only_daily_log_returns(df)
    disc_daily = expanding_bootstrap_discount(daily, floor=floor,
                                              refit_every_days=refit_every_days,
                                              min_days=min_days)
    disc_bars = broadcast_daily_to_bars(disc_daily, df.index)
    raw = v4_raw_desired(df) * disc_bars
    return apply_deadband(raw), disc_bars


def make_build_target(floor: float, refit_every_days: int = REFIT_DAYS_DEFAULT,
                      min_days: int = MIN_DAYS):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, floor=floor, refit_every_days=refit_every_days,
                            min_days=min_days)
    _build.__name__ = f"bootstrap_sig_floor{floor:g}_refit{refit_every_days}d"
    return _build


# ================================================================== (2)
# Step-0 non-degeneracy grid: floor in {0.3, 0.5, 0.7} at refit_every_days=90,
# computed on the FULL BTC frame (exactly as compare()'s own
# TargetStrategy.prepare would), masked to inner-train for both diagnostics.
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], int]:
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                       (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())
    ctrl_target = v4_target(btc)

    rows = []
    for floor in GRID_FLOOR:
        target, disc_bars = build_target_with_disc(btc, floor=floor,
                                                    refit_every_days=REFIT_DAYS_DEFAULT,
                                                    min_days=MIN_DAYS)
        bind_frac = float(np.mean(disc_bars[mask] < 1.0 - 1e-9))
        r_sq = r_squared(target[mask], ctrl_target[mask])
        qualifies = (bind_frac > BIND_FRAC_THRESH) and (r_sq < R2_THRESH)
        rows.append(dict(floor=floor, bind_frac=bind_frac, r_sq=r_sq, qualifies=qualifies))
    return rows, n_bars


def select_primary(rows: list[dict]) -> dict | None:
    by_floor = {r["floor"]: r for r in rows}
    for f in SELECTION_ORDER:
        r = by_floor.get(f)
        if r is not None and r["qualifies"]:
            return r
    return None


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars; refit_every_days={REFIT_DAYS_DEFAULT} for every cell)")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r_sq < {R2_THRESH}")
    hdr_line = f"{'floor':>6s} {'bind_frac':>10s} {'r_sq':>8s} {'qualifies':>10s}"
    print(hdr_line)
    print("-" * len(hdr_line))
    for r in rows:
        tag = " <- grid centre" if r["floor"] == CENTER_FLOOR else ""
        print(f"{r['floor']:6.2f} {r['bind_frac']:10.4f} {r['r_sq']:8.4f} "
              f"{'YES' if r['qualifies'] else 'no':>10s}{tag}")


# ================================================================== (3)
# Causal truncation probe on the composed pipeline, real BTC data.
# ==================================================================

def run_causal_probe(df: pd.DataFrame, build_fn) -> bool:
    print(f"\ncausal_truncation_probe_series({build_fn.__name__}, btc):")
    try:
        causal_truncation_probe_series(build_fn, df)
        print("  PASS")
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


# ================================================================== (4)
# Lighter inner_val-only helper (direct run_slice + paired_diff), used by
# both B3 (refit_every_days sweep) and B5 (fee tier) -- documented choice,
# see pre-registration item 6.
# ==================================================================

def inner_val_cells(build_fn, label: str, btc: pd.DataFrame,
                    markets: tuple = (SPOT, FUTURES)) -> list[dict]:
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r104_{label}")
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


# ================================================================== (5)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau),
# B4 (gating falsification), B5 (gating cost robustness).
# ==================================================================

def run_promotion_bar(primary_floor: float, btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    label = f"bootstrap_sig_floor{primary_floor:g}"
    build_primary = make_build_target(primary_floor, refit_every_days=REFIT_DAYS_DEFAULT,
                                      min_days=MIN_DAYS)

    hr(f"PROMOTION BAR -- PRIMARY CONFIG floor={primary_floor:g}, "
       f"refit_every_days={REFIT_DAYS_DEFAULT}, min_days={MIN_DAYS}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                   markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_rows_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_rows_primary = [r for r in rows if r["slice"] == "eth_replication"]

    # ---- B1: Sharpe leg, both markets, inner_val.
    b1_cells = []
    for r in inner_val_rows_primary:
        passes = (r["d_sharpe"] > SHARPE_NOISE_FLOOR) or (r["excludes_zero"] and r["boot_lo"] > 0)
        b1_cells.append(dict(market=r["market"], passes=passes, d_sharpe=r["d_sharpe"],
                             boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                             boot_d_loggrowth=r["boot_d_loggrowth"],
                             excludes_zero=r["excludes_zero"]))
    b1_pass = all(c["passes"] for c in b1_cells)

    # ---- B2: diagnostic only, never gates.
    b2_cells = []
    for r in inner_val_rows_primary:
        b2_cells.append(dict(market=r["market"], d_dd=r["d_dd"], risk_matched=r["risk_matched"],
                             exposure_ratio=r["exposure_ratio"], vol_ratio=r["vol_ratio"]))

    # ---- B3: refit_every_days plateau, inner_val only, both markets.
    plateau_rows: dict[int, list[dict]] = {
        REFIT_DAYS_DEFAULT: [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                                  d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                                  vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                                  boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                                  boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                             for r in inner_val_rows_primary]
    }
    for refit in B3_REFIT_GRID:
        if refit == REFIT_DAYS_DEFAULT:
            continue
        bf = make_build_target(primary_floor, refit_every_days=refit, min_days=MIN_DAYS)
        blabel = f"bootstrap_sig_floor{primary_floor:g}_refit{refit}d"
        plateau_rows[refit] = inner_val_cells(bf, blabel, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for refit in B3_REFIT_GRID for r in plateau_rows[refit]]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False

    # ---- B4: ETH falsification.
    b4_cells = []
    for r in eth_rows_primary:
        btc_match = next((c for c in inner_val_rows_primary if c["market"] == r["market"]), None)
        same_sign = (btc_match is not None and
                    np.sign(r["d_sharpe"]) == np.sign(btc_match["d_sharpe"]) and
                    r["d_sharpe"] != 0 and btc_match["d_sharpe"] != 0)
        b4_cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                             btc_inner_val_d_sharpe=(btc_match["d_sharpe"] if btc_match else float("nan")),
                             boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                             excludes_zero=r["excludes_zero"], same_sign_as_btc=same_sign))
    b4_full_pass = all(c["same_sign_as_btc"] for c in b4_cells) and len(b4_cells) > 0
    b4_partial_pass = any(c["same_sign_as_btc"] for c in b4_cells)

    # ---- B5: fee-tier survival, primary config, inner_val only.
    hr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary config, inner-validation")
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    fee_rows = inner_val_cells(build_primary, label, btc, markets=fee_markets)
    b5_cells = []
    for r in fee_rows:
        base = next((c for c in inner_val_rows_primary if c["market"] == r["market"]), None)
        no_reversal = (base is not None and
                      not (np.sign(r["boot_d_loggrowth"]) != np.sign(base["boot_d_loggrowth"]) and
                           r["boot_d_loggrowth"] != 0 and base["boot_d_loggrowth"] != 0))
        b5_cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                             boot_d_loggrowth=r["boot_d_loggrowth"],
                             base_boot_d_loggrowth=base["boot_d_loggrowth"] if base else float("nan"),
                             no_reversal=no_reversal))
    b5_pass = all(c["no_reversal"] for c in b5_cells)

    all_pass = b1_pass and b3_pass and b4_full_pass and b5_pass

    return dict(
        label=label, floor=primary_floor,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells,
        b3_plateau_rows=plateau_rows, b3_pass=b3_pass,
        b4_cells=b4_cells, b4_full_pass=b4_full_pass, b4_partial_pass=b4_partial_pass,
        b5_cells=b5_cells, b5_pass=b5_pass,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 6 + 2,
    )


def print_plateau_table(plateau_rows: dict[int, list[dict]]) -> None:
    hdr_line = (f"{'refit_days':>10s} {'market':>9s} {'dSh':>7s} {'dDD':>7s} "
               f"{'expR':>5s} {'volR':>5s} {'RM':>3s} {'dlogG':>7s} "
               f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr_line)
    print("-" * len(hdr_line))
    for refit in B3_REFIT_GRID:
        for r in plateau_rows[refit]:
            tag = " (reused from primary compare())" if refit == REFIT_DAYS_DEFAULT else ""
            print(f"{refit:10d} {r['market']:>9s} {r['d_sharpe']:+7.2f} "
                  f"{r['d_dd']:+7.1f} {r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
                  f"{'Y' if r['risk_matched'] else 'n':>3s} {r['boot_d_loggrowth']:+7.3f} "
                  f"{r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
                  f"{'YES' if r['excludes_zero'] else 'no':>5s}{tag}")


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-104 CONSERVATIVE: BootstrapSigKellyV4 -- periodic stationary "
       "block-bootstrap significance discount on v4's own frac*scale")
    print("mechanism: multiply v4's UNCHANGED frac*scale by a discount derived from a periodic")
    print("(batch, refit every refit_every_days) Politis & Romano (1994) stationary block-bootstrap")
    print("t-statistic of the vote's own historical daily edge, held constant between refits;")
    print("v4's own 10% deadband is applied AFTER the discount, exactly v4's existing composition order.")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    print(f"\nTargetStrategy.warmup LEFT AT SHARED DEFAULT ({TargetStrategy.warmup:,} bars, "
          f"{TargetStrategy.warmup / BARS_PER_DAY:g} calendar days) -- NOT inflated; "
          f"see pre-registration item 3 for why, and its disclosed consequence for "
          f"inner_val/eth_replication/B3/B5's local-restart history.")

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY KILL SWITCH (run BEFORE any Sharpe/compare() number)")
    step0_rows, n_bars = step0_grid(btc)
    print_step0_table(step0_rows, n_bars)

    primary = select_primary(step0_rows)

    if primary is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell has both bind_frac > 1% and r_sq < 0.98: the bootstrap significance")
        print("discount is either a near-total no-op or a near-exact rescale of v4's own path")
        print("everywhere on the pre-registered floor grid -- the R-87-shaped 'inert' failure mode")
        print("this round's own pre-registration named as the single most likely outcome. Per this")
        print("file's own pre-registration, this Step-0 table is the branch's ENTIRE product,")
        print("reported NEGATIVE / stopped-at-Step-0. No causal probe, B1-B5 code, or ETH load runs.")
        n_configs = len(GRID_FLOOR)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {max(max_ts_seen)}  "
              f"(< {OOS_START}: {max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                   n_configs=n_configs, max_ts=max(max_ts_seen), verdict="NEGATIVE (Step-0 kill switch)")

    is_center = (primary["floor"] == CENTER_FLOOR)
    print(f"\nPRIMARY CONFIG SELECTED (non-degeneracy rule only): floor={primary['floor']:g}, "
          f"refit_every_days={REFIT_DAYS_DEFAULT}, min_days={MIN_DAYS}  "
          f"(bind_frac={primary['bind_frac']:.4f}, r_sq={primary['r_sq']:.4f})")
    print(f"  selection: {'grid-centre cell qualified' if is_center else 'grid-centre cell did NOT qualify; nearest qualifying cell in order [0.5, 0.3, 0.7] chosen'}")

    build_primary = make_build_target(primary["floor"], refit_every_days=REFIT_DAYS_DEFAULT,
                                      min_days=MIN_DAYS)

    # ==================================================== CAUSAL PROBE
    hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data)")
    probe_ok = run_causal_probe(btc, build_primary)
    print(f"\nCAUSAL PROBE PASS: {probe_ok}")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    bar = run_promotion_bar(primary["floor"], btc, eth)

    hr("B1 -- inner-validation Sharpe leg, both markets "
       "(dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  "
              f"excludes_zero={c['excludes_zero']}  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  "
              f"exposure_ratio={c['exposure_ratio']:.4f}  vol_ratio={c['vol_ratio']:.4f}  "
              f"risk_matched={c['risk_matched']}")

    hr("B3 -- plateau: refit_every_days sweep {30, 90, 180} at primary floor, "
       "inner-validation, both markets")
    print_plateau_table(bar["b3_plateau_rows"])
    print(f"\nB3 (directionally consistent majority across the 6-cell grid): {bar['b3_pass']}")

    hr("B4 -- ETH falsification (pre-registered)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  BTC inner_val d_sharpe={c['btc_inner_val_d_sharpe']:+.4f}  "
              f"ETH d_sharpe={c['d_sharpe']:+.4f}  boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  "
              f"same_sign_as_btc={c['same_sign_as_btc']}")
    print(f"B4 FULL PASS (both markets): {bar['b4_full_pass']}")
    print(f"B4 PARTIAL PASS (at least one market): {bar['b4_partial_pass']}")

    hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"@0.40% boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"@0.10% boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    hr("VERDICT")
    print(f"causal probe = {probe_ok}   B1 = {bar['b1_pass']}   B2 = diagnostic-only   "
          f"B3 = {bar['b3_pass']}   B4(full) = {bar['b4_full_pass']}   "
          f"B4(partial) = {bar['b4_partial_pass']}   B5 = {bar['b5_pass']}")
    all_gates_pass = probe_ok and bar["b1_pass"] and bar["b3_pass"] and bar["b4_full_pass"] and bar["b5_pass"]
    verdict = "PROMOTE-candidate" if all_gates_pass else "NEGATIVE"
    print(f"\nALL GATING CLAUSES PASS (causal AND B1 AND B3 AND B4-full AND B5): {all_gates_pass}")
    print(f"VERDICT: {verdict}")
    if not all_gates_pass:
        failed = [name for name, ok in (
            ("causal probe", probe_ok), ("B1", bar["b1_pass"]), ("B3", bar["b3_pass"]),
            ("B4 (full)", bar["b4_full_pass"]), ("B5", bar["b5_pass"]),
        ) if not ok]
        print(f"Reason(s): {', '.join(failed)}")

    n_configs = 3 + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(3 Step-0 grid + 6 primary compare() + 6 B3 refit grid "
          f"[4 fresh + 2 reused from primary] + 2 B5 fee-tier)")
    max_ts = max(max_ts_seen)
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary, passed_step0=True,
               probe_ok=probe_ok, promotion_bar=bar, verdict=verdict, n_configs=n_configs,
               max_ts=max_ts)


if __name__ == "__main__":
    main()
