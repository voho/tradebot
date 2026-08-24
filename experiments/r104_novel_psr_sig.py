#!/usr/bin/env python
"""R-104 NOVEL branch: ``PSRSigKellyV4`` -- ``kelly_regime_v4``'s own
unchanged ``frac * scale`` product, multiplied by a CONTINUOUS, closed-form
Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012) discount of the
vote's own historical daily edge, recomputed every day from an expanding
window, before v4's own 10% deadband is applied. Full citation trail,
literature grounding, the axis this attacks (ERR -- no error control
anywhere in the signal path), and the exhaustive non-duplication argument
against every related prior round (R-28/R-31, R-87, R-101, R-97, every
SIZE-axis round, and this round's own CONSERVATIVE sibling) all live in
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
   reference P&L of holding v4's own latched vote alone) is fed, EVERY DAY,
   into Bailey & Lopez de Prado (2012)'s Probabilistic Sharpe Ratio --
   ``PSR(0) = P(true Sharpe > 0 | observed data)``, a normal-approximation
   probability corrected for the sample's own skew and kurtosis, computed
   from an EXPANDING window over the vote's own daily returns -- and that
   probability is used DIRECTLY as the discount (clipped only at a floor),
   with no additional shape parameter to hand-choose, multiplied onto v4's
   UNCHANGED ``frac * scale`` before v4's own deadband is applied. Unlike
   this round's CONSERVATIVE sibling (a periodic, batch Monte Carlo
   stationary-block-bootstrap t-statistic fed through a hand-set linear
   ramp, refit infrequently because resampling is comparatively expensive),
   this estimator is CONTINUOUS: PSR is closed-form and cheap, so it is
   recomputed at every single day by construction -- there is no
   ``refit_every_days`` knob here, deliberately, to contrast "periodic
   batch bootstrap" against "continuous closed-form estimator" as cleanly
   as possible.

2. CONSTRUCTION (exact; ``r104_shared.py`` already implements every
   primitive below verbatim -- this file only composes them, per this
   round's own dispatch):

       daily[t]        = vote_only_daily_log_returns(df)[t]        # r104_shared
       disc_daily[t]   = expanding_psr_discount(daily, floor=floor,
                             min_days=min_days)[t]                  # r104_shared
       disc_bars[t]    = broadcast_daily_to_bars(disc_daily, df.index)[t]
       raw[t]          = v4_raw_desired(df)[t] * disc_bars[t]       # frac*scale, UNCHANGED, then discounted
       target[t]       = apply_deadband(raw)[t]                     # v4's own deadband, AFTER discount

   DEFAULT/PRIMARY CONFIG (pre-registered, not changed after seeing any
   result): ``floor=0.5, min_days=120`` -- both ``r104_shared.py``'s own
   pre-registered constant (``MIN_DAYS=120``) and this file's own
   grid-centre choice (``floor=0.5``), confirmed or overridden only by the
   Step-0 selection rule below, never by a performance number.
   ``expanding_psr_discount`` itself uses ``annualized_sharpe``'s own
   ``ddof=1`` sample statistics and Bailey & Lopez de Prado's own
   skew/kurtosis correction (``moments`` + ``probabilistic_sharpe_ratio``)
   with no additional free parameter -- the discount's functional form is
   DERIVED (the PSR value itself), not hand-tuned, in contrast to the
   conservative branch's hand-set ``significance_ramp`` (t<=1 -> floor,
   t>=2 -> 1.0, linear between).

3. SCOPE LIMITATION, DISCLOSED BUT DELIBERATELY NOT "FIXED" BY MUTATING
   ``TargetStrategy.warmup`` (unlike an approach this round's own
   CONSERVATIVE sibling attempted): the shared ``TargetStrategy`` class
   defaults to an 80-calendar-day warmup (``80 * BARS_PER_DAY + 10``
   bars). Under that default, ``run_period``'s own ``prefix_bars =
   min(start_pos, warmup)`` hands this file's ``build_target`` only ~80
   days of history before ``inner_val``'s 2021-01-01 start -- LESS than
   ``MIN_DAYS=120`` alone, meaning the PSR discount is still inside its
   own burn-in (pinned at 1.0, identical to v4) for the first ~40 days of
   inner-validation, and spends the rest of inner-validation estimating
   significance from a comparatively short (~120-650 day) LOCAL tail
   restarted from that 80-day-prior point, rather than continuing the
   genuinely available, continuous-since-2017 estimate ``inner_train``'s
   own reading uses. This is IDENTICAL in shape to the scope choice
   ``r103_shared.py``'s own module docstring names and
   ``r103_novel_causal_rls.py`` disclosed rather than acted on for its RLS
   branch. Before writing this file, one candidate "fix" was investigated
   and REJECTED: globally reassigning ``TargetStrategy.warmup`` to a large
   value (as this round's own CONSERVATIVE sibling does, to
   ``2_500 * BARS_PER_DAY = 720,000`` bars) so that ``inner_val`` sees a
   full multi-year prefix. That value exceeds EVERY frame this file
   evaluates (BTC's entire pre-holdout record is only 631,008 bars; ETH's
   is 342,929) -- since the same ``TargetStrategy.warmup`` attribute also
   gates ``i >= strategy.warmup`` inside ``tradebot.engine.run_backtest``
   (the trade-ELIGIBILITY check, not merely the prefix-length one), a
   value that exceeds a frame's own total bar count means ``on_bar`` NEVER
   fires for that frame -- verified directly by running the CONSERVATIVE
   sibling: every row of its ``compare()`` table reads
   ``cand_final == ctrl_final == $1,000.00``, ``d_sharpe == +0.0000``,
   across EVERY slice (inner_train, inner_val, AND eth_replication) for
   BOTH the candidate and the control, because the identical, globally
   mutated ``warmup`` value silences BOTH ``TargetStrategy`` instances
   ``compare()`` constructs internally -- the exact "0 trades, $1,000.00
   unchanged" failure mode ``r103_shared.py``'s own module docstring names
   as R-101's precedent for why inflating ``warmup`` is not simply safe.
   No value of ``warmup`` can satisfy both goals at once with the SINGLE,
   GLOBAL, class-level attribute ``compare()`` shares across all three
   slices: ``inner_train`` and ``eth_replication`` both start at their own
   frame's TRUE beginning (zero available prefix, per
   ``run_period``'s own ``prefix_bars = min(start_pos, warmup)``), so ANY
   increase to ``warmup`` beyond the shipped default directly delays
   trading INTO those two periods' own measured windows rather than
   "buying" a longer prefix for them the way it does for ``inner_val``.
   This file therefore leaves ``TargetStrategy.warmup`` at its shared
   default, unmodified, and reports the resulting scope limitation
   honestly wherever it is relevant to reading a number below (chiefly:
   ``inner_val``'s, and every ``inner_val``-based B1/B2/B3/B5 cell's,
   first ~40 days sit at v4-parity by construction, understating rather
   than overstating any genuine PSR effect there) rather than risk
   introducing the same silent-zero-trades failure into this branch.

4. STEP-0 NON-DEGENERACY GRID AND SELECTION RULE, PRE-REGISTERED BEFORE ANY
   GRID NUMBER WAS COMPUTED (mirrors R-102/R-103/this round's own
   CONSERVATIVE sibling's grid shape and selection-rule convention):

   Grid: ``floor in {0.3, 0.5, 0.7}`` at the default ``min_days=120`` (3
   cells, fixed a priori). For each cell, on BTC's inner-train window
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
   the pre-registered, R-87-shaped "inert" failure mode (BTC's own
   vote-lean hit rate has historically sat persistently above 50% past the
   burn-in, so the vote's edge may already be statistically significant
   almost everywhere, making PSR rarely bind below 1.0) -- a legitimate,
   informative NEGATIVE result, not a bug to route around. No B1-B5 code
   runs in that case, and no bar on/after ``OOS_START`` (2023-01-01) is
   ever touched either way.

   Also reported at Step-0, PURELY AS A DIAGNOSTIC (does not gate
   selection or qualification): the ``hac_mean_se``-implied t-statistic
   path's summary stats (mean/min/max) over the same inner-train daily
   series, alongside the PSR path's own summary stats -- letting a reader
   see how the closed-form, skew/kurtosis-corrected PSR compares to the
   plainer HAC-studentized t-statistic on identical data.

5. CAUSAL TRUNCATION PROBE + EXPLICIT NO-LOOKAHEAD CHECK, run before
   trusting any Step-0 or promotion-bar number, and run REGARDLESS of
   whether Step-0 selects a primary (this is a verification of the
   CONSTRUCTION's own causal safety, not of its economic result):
   ``r104_shared.causal_truncation_probe_series`` (re-exported from
   ``r102_shared``) applied to this file's own composed ``build_target``
   closure at the pre-registered default config (``floor=0.5,
   min_days=120``), on real BTC data. In addition, since PSR is a
   NONLINEAR function of an expanding window (not a simple linear
   recursion), this file adds one EXPLICIT no-lookahead check in the style
   of ``r104_shared.py``'s own self-test #4 / ``r103_shared.py``'s own
   self-test check 6: a late, day-aligned slice of the real BTC frame is
   perturbed (price columns scaled and offset from that day forward), and
   the bar-level discount path strictly before the perturbed day must be
   BIT-IDENTICAL between the perturbed and unperturbed runs. This is this
   branch's single most safety-critical check, since a bug here would mean
   PSR consumed information the market had not yet revealed.

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
     B3 (plateau, not peak, gating): sweep ``min_days in {60, 120, 250}``
        at the selected primary floor, ``inner_val`` only, both markets (3
        configs x 2 markets = 6 cells). This file uses a LIGHTER helper
        (``inner_val_rows``, direct ``run_slice`` + ``paired_diff`` calls,
        the same idiom the conservative sibling's own ``inner_val_cells``
        and R-102/R-103 novel's own ``inner_val_rows`` use) rather than
        three full ``compare()`` calls, to avoid recomputing the
        ``inner_train``/``eth_replication`` slices three extra times for
        no purpose -- documented here as the chosen approach. The
        ``min_days=120`` cell is NOT recomputed by this helper; it is read
        directly from the primary ``compare()`` call's own ``inner_val``
        rows (identical construction, so recomputing it would only
        reproduce the same numbers under a new bootstrap seed for no
        diagnostic value) -- still counted as 2 of the grid's 6 cells per
        the pre-registered count. PASS requires a directionally consistent
        (same-sign-majority) region across all 6 cells, not an isolated
        spike at 120 days.
     B4 (ETH falsification, PRE-REGISTERED, not changed after seeing
        results, gating in its FULL form): ``eth_replication`` rows must
        show the SAME SIGN of ``d_sharpe`` as BTC's own ``inner_val``
        ``d_sharpe``. At least one market matching counts as a PARTIAL
        pass; both markets matching is a FULL pass. Both numbers reported
        exactly, explicitly labelled which.
     B5 (cost robustness, gating): at the selected primary floor,
        ``min_days=120``, BTC ``inner_val`` re-run at a 0.40% taker fee
        tier (``fee_at(SPOT, 0.0040)`` / ``fee_at(FUTURES, 0.0040)``), via
        the same ``inner_val_rows`` helper (the idiom
        ``r102_novel_signed_jump_discount.py``'s / ``r103_novel_causal_
        rls.py``'s / the conservative sibling's own B5 sections use: direct
        ``run_slice`` + ``paired_diff`` against a fee-adjusted
        ``MarketSpec``, not a full re-run of ``compare()``). PASS iff the
        SIGN of ``boot_d_loggrowth`` (d_log_growth) and of ``d_sharpe`` do
        not reverse relative to the standard-fee (0.10%) result on the
        same market.
   PROMOTE-candidate only if Step-0 selects a primary AND the causal probe
   AND the explicit no-lookahead check both pass AND B1 AND B3 AND B4
   (full form) AND B5 all hold (B2 is diagnostic-only). Default: NEGATIVE.
   This file never reads or reports any bar at or after ``OOS_START``
   (2023-01-01) regardless of outcome.

7. WHAT WOULD MAKE THIS FAIL: named already, in full, in ``r104_shared.py``
   itself (three specific, independent failure shapes: the R-87-shaped
   inert-discount pattern this file's own Step-0 kill switch is built to
   catch; a genuinely non-degenerate but too-small-or-too-late discount,
   the R-97/R-101 "real but inert in practice" pattern B1 is built to
   catch; and this branch's OWN specific risk relative to its CONSERVATIVE
   sibling -- a continuous HAC/PSR estimator on a short expanding window
   being too NOISY day-to-day to produce a stable discount, which B3's
   plateau sweep is built to surface as a measurable batch-vs-continuous
   difference). Not re-derived here; reported honestly, whichever way it
   comes out, in the results below.

CONFIGURATIONS EVALUATED IN THIS FILE (IF Step-0 selects a primary): 3
(Step-0 floor grid) + 6 (primary config's full ``compare()``: inner_train x2
markets + inner_val x2 markets + eth_replication x2 markets) + 6 (B3's
min_days grid, 3 configs x 2 markets -- 2 of the 6 reused directly from the
primary ``compare()``'s own inner_val rows, 4 freshly computed) + 2 (B5's
0.40% fee tier, 2 markets) = 17 total. IF Step-0 finds no qualifying cell,
this file stops after the 3 Step-0 cells and reports that outcome as the
round's own kill-switch finding.

USAGE
-----
    python experiments/r104_novel_psr_sig.py
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
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    broadcast_daily_to_bars,
    causal_truncation_probe_series,
    compare,
    expanding_psr_discount,
    fee_at,
    hac_mean_se,
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
PSR_FLOOR_DEFAULT = 0.5
STEP0_FLOOR_GRID = (0.3, 0.5, 0.7)
SELECTION_ORDER = (0.5, 0.3, 0.7)
BIND_FRAC_THRESH = 0.01
R2_THRESH = 0.98
B3_MIN_DAYS_GRID = (60, 120, 250)
FEE_TIER = 0.004
SHARPE_NOISE_FLOOR = 0.2
# NOTE: TargetStrategy.warmup is deliberately left at its shared default
# (80 * BARS_PER_DAY + 10 bars) -- see pre-registration item 3 for why a
# global reassignment (as this round's own conservative sibling attempted)
# was investigated and rejected as unsafe (it silences on_bar entirely for
# every frame shorter than the reassigned value).


def hr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (1)
# The mechanism itself: a pure function of df, built from r104_shared's
# UNCHANGED v4_raw_desired / vote_only_daily_log_returns /
# expanding_psr_discount / broadcast_daily_to_bars / apply_deadband.
# ==================================================================

def disc_bars_for(df: pd.DataFrame, floor: float, min_days: int = MIN_DAYS) -> np.ndarray:
    """``df -> disc_bars``: the daily PSR discount, broadcast onto every
    bar of the frame. Factored out so Step-0's ``bind_frac`` (needs
    ``disc_bars`` directly) and ``build_target`` (needs it multiplied onto
    ``v4_raw_desired``) share one computation path rather than drifting."""
    daily = vote_only_daily_log_returns(df)
    disc_daily = expanding_psr_discount(daily, floor=floor, min_days=min_days)
    return broadcast_daily_to_bars(disc_daily, df.index)


def build_target(df: pd.DataFrame, floor: float = PSR_FLOOR_DEFAULT,
                 min_days: int = MIN_DAYS) -> np.ndarray:
    """A pure ``df -> np.ndarray``: v4's own unchanged ``frac * scale``,
    discounted by the vote's own causal, continuously-updated PSR, then
    v4's own deadband applied AFTER the discount. Exactly the construction
    specified in this round's dispatch."""
    disc_bars = disc_bars_for(df, floor, min_days)
    raw = v4_raw_desired(df) * disc_bars
    return apply_deadband(raw)


def make_build_target(floor: float, min_days: int = MIN_DAYS):
    def build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, floor=floor, min_days=min_days)
    build.__name__ = f"psr_sig_floor{floor:g}_mind{min_days:d}"
    return build


# ================================================================== (2)
# Step-0 non-degeneracy grid + HAC-t-stat diagnostic (not gating).
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], int]:
    """3-cell grid, computed on the FULL pre-holdout BTC price history (so
    the PSR estimator has its natural, continuous warmup from the
    dataset's true start -- inner_train itself starts at that same point,
    so this is also exactly what inner_train's own promotion-bar reading
    sees), with the two reported statistics restricted to bars inside the
    inner-train window."""
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                      (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    raw_base = v4_raw_desired(btc)
    ctrl_target = v4_target(btc)

    rows = []
    for floor in STEP0_FLOOR_GRID:
        disc_bars = disc_bars_for(btc, floor, MIN_DAYS)
        target = apply_deadband(raw_base * disc_bars)
        bind_frac = float(np.mean(disc_bars[mask] < 1.0 - 1e-9))
        r_sq = r_squared(target[mask], ctrl_target[mask])
        qualifies = (bind_frac > BIND_FRAC_THRESH) and (r_sq < R2_THRESH)
        rows.append(dict(floor=floor, bind_frac=bind_frac, r_sq=r_sq, qualifies=qualifies))
    return rows, n_bars


def select_primary(rows: list[dict]) -> dict | None:
    """Pre-registered selection: floor=0.5 (grid centre) if it qualifies,
    else the nearest qualifying cell in the pre-registered order
    [0.5, 0.3, 0.7]. None if nothing qualifies."""
    by_floor = {r["floor"]: r for r in rows}
    for f in SELECTION_ORDER:
        r = by_floor.get(f)
        if r is not None and r["qualifies"]:
            return r
    return None


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars, min_days={MIN_DAYS})")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r_sq < {R2_THRESH}")
    hdr_line = f"{'floor':>6s} {'bind_frac':>10s} {'r_sq':>8s} {'qualifies':>10s}"
    print(hdr_line)
    print("-" * len(hdr_line))
    for r in rows:
        tag = "  <- grid centre" if r["floor"] == 0.5 else ""
        print(f"{r['floor']:6.2f} {r['bind_frac']:10.4f} {r['r_sq']:8.4f} "
              f"{'YES' if r['qualifies'] else 'no':>10s}{tag}")


def hac_vs_psr_diagnostic(btc: pd.DataFrame) -> dict:
    """Diagnostic ONLY (does not gate Step-0 or the promotion bar): the
    hac_mean_se-implied expanding t-statistic path, alongside the PSR
    path (at the primary default floor/min_days), both restricted to
    BTC's inner-train window -- day i uses only days strictly before i in
    both paths (matching expanding_psr_discount's own convention), so a
    reader can see how the closed-form, skew/kurtosis-corrected PSR
    compares to the plainer HAC t-statistic on identical data."""
    daily_full = vote_only_daily_log_returns(btc)
    daily = daily_full[daily_full.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")]
    n = len(daily)

    t_path = np.full(n, np.nan)
    for i in range(n):
        if i < MIN_DAYS:
            continue
        window = daily.iloc[:i].to_numpy()
        window = window[np.isfinite(window)]
        if len(window) == 0:
            continue
        se = hac_mean_se(window)
        if se and np.isfinite(se) and se > 0:
            t_path[i] = float(np.mean(window)) / se

    psr_path = expanding_psr_discount(daily, floor=0.0, min_days=MIN_DAYS).to_numpy()
    # floor=0.0 recovers the RAW PSR value unclipped (PSR is already in
    # [0, 1] by construction as a normal-CDF probability), so this is not
    # a separate estimator -- it is the primary path's own PSR, reported
    # unclipped for the diagnostic table below.
    psr_warmed = psr_path[MIN_DAYS:]
    t_warmed = t_path[np.isfinite(t_path)]

    return dict(
        n_days=n,
        psr_mean=float(np.mean(psr_warmed)), psr_min=float(np.min(psr_warmed)),
        psr_max=float(np.max(psr_warmed)),
        t_n=int(len(t_warmed)),
        t_mean=float(np.mean(t_warmed)) if len(t_warmed) else float("nan"),
        t_min=float(np.min(t_warmed)) if len(t_warmed) else float("nan"),
        t_max=float(np.max(t_warmed)) if len(t_warmed) else float("nan"),
    )


def print_hac_vs_psr(diag: dict) -> None:
    hr("DIAGNOSTIC (not gating): PSR path vs. HAC-t-statistic path, BTC inner-train")
    print(f"  inner-train days: {diag['n_days']:,}  (min_days={MIN_DAYS} burn-in)")
    print(f"  PSR   (unclipped, floor=0.0): mean={diag['psr_mean']:.4f}  "
          f"min={diag['psr_min']:.4f}  max={diag['psr_max']:.4f}  n={diag['n_days'] - MIN_DAYS:,}")
    print(f"  HAC t-statistic (mean/se):    mean={diag['t_mean']:+.4f}  "
          f"min={diag['t_min']:+.4f}  max={diag['t_max']:+.4f}  n={diag['t_n']:,}")
    print("  (PSR>0.977 corresponds to a two-sided-significance-adjacent t~2.0 under a")
    print("   normal approximation; PSR floors at 0.5 when t=0 by construction. Reported")
    print("   purely so a reader can see how the two estimators track each other on the")
    print("   same data -- neither number gates Step-0 or the promotion bar.)")


# ================================================================== (3)
# Causal truncation probe + explicit no-lookahead perturbation check.
# ==================================================================

def run_causal_probe(btc: pd.DataFrame, build_default) -> bool:
    print(f"\ncausal_truncation_probe_series(build_target[floor={PSR_FLOOR_DEFAULT:g}, "
          f"min_days={MIN_DAYS}], df):")
    try:
        ok = causal_truncation_probe_series(build_default, btc)
        print("  PASS")
    except AssertionError as e:
        ok = False
        print(f"  FAIL: {e}")
    return ok


def explicit_no_lookahead_check(btc: pd.DataFrame, floor: float = PSR_FLOOR_DEFAULT,
                                min_days: int = MIN_DAYS) -> dict:
    """Explicit, additional no-lookahead check (in the style of
    r104_shared.py's own self-test #4 / r103_shared.py's own self-test
    check 6): perturb a late, DAY-ALIGNED slice of the real BTC frame
    (price columns scaled and offset from that day forward) and confirm
    the bar-level discount path strictly before the perturbed day is
    BIT-IDENTICAL between the perturbed and unperturbed runs. Day-aligned
    so the comparison excludes the "incomplete final day" artifact
    r104_shared.py's own self-test names (a cut mid-day creates a
    genuinely different, incomplete final day between the two frames --
    an artifact of the probe's own truncation, not a lookahead in the
    pipeline)."""
    disc_orig = disc_bars_for(btc, floor, min_days)

    perturb_from = (3 * len(btc)) // 4
    perturb_from = (perturb_from // BARS_PER_DAY) * BARS_PER_DAY  # day-align

    btc2 = btc.copy()
    cols = ["open", "high", "low", "close"]
    idx = btc2.columns.get_indexer(cols)
    tail = btc2.iloc[perturb_from:].copy()
    tail.iloc[:, idx] = tail.iloc[:, idx] * 5.0 + 1.0
    btc2.iloc[perturb_from:] = tail

    disc_pert = disc_bars_for(btc2, floor, min_days)

    a, b = disc_orig[:perturb_from], disc_pert[:perturb_from]
    m = np.isfinite(a) & np.isfinite(b)
    n_checked = int(m.sum())
    bit_identical = n_checked > 1000 and np.array_equal(a[m], b[m])
    close_identical = n_checked > 1000 and np.allclose(a[m], b[m], atol=1e-12, rtol=0.0)
    ok = bit_identical or close_identical

    print(f"\nEXPLICIT no-lookahead perturbation check (BTC, perturb_from bar "
          f"{perturb_from:,} of {len(btc):,}, day-aligned):")
    print(f"  bars compared (strictly before perturbed region): {n_checked:,}")
    print(f"  bit-identical (np.array_equal): {bit_identical}")
    print(f"  numerically identical (atol=1e-12): {close_identical}")
    print(f"  RESULT: {'PASS -- discount path unaffected before the perturbed day' if ok else 'FAIL -- PSR discount LEAKS FUTURE INFORMATION into a pre-perturbation bar'}")

    return dict(ok=ok, perturb_from=perturb_from, n_checked=n_checked,
               bit_identical=bit_identical, close_identical=close_identical)


# ================================================================== (4)
# Promotion bar: B1 (bootstrap), B2 (risk-matched dd), B3 (plateau),
# B4 (ETH), B5 (fee tier).
# ==================================================================

def inner_val_rows(build_fn, label: str, btc: pd.DataFrame,
                   markets=(SPOT, FUTURES)) -> list[dict]:
    """Lightweight inner-validation-only comparison (both markets), used
    for the B3 plateau check's non-primary cells and for B5's fee-tier
    cells -- avoids the full compare() overhead (inner_train + ETH) for
    cells that are not the decision-bearing one. Direct run_slice() +
    paired_diff(), the same idiom R-102/R-103 novel and this round's own
    conservative sibling all use for the identical purpose."""
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


def print_plateau_table(all_rows: dict[int, list[dict]]) -> None:
    hdr_line = (f"{'min_days':>8s} {'market':>9s} {'dSh':>7s} {'dDD':>7s} "
               f"{'expR':>5s} {'volR':>5s} {'RM':>3s} {'dlogG':>7s} "
               f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr_line)
    print("-" * len(hdr_line))
    for md, rows in all_rows.items():
        for r in rows:
            print(f"{md:8d} {r['market']:>9s} {r['d_sharpe']:+7.2f} "
                  f"{r['d_dd']:+7.1f} {r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
                  f"{'Y' if r['risk_matched'] else 'n':>3s} {r['boot_d_loggrowth']:+7.3f} "
                  f"{r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
                  f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def run_promotion_bar(primary_floor: float, btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    build_primary = make_build_target(primary_floor, MIN_DAYS)
    label = f"psr_sig_floor{primary_floor:g}_mind{MIN_DAYS:d}"

    hr(f"PROMOTION BAR -- PRIMARY CELL floor={primary_floor:g}, min_days={MIN_DAYS}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                  markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_rows_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_rows_primary = [r for r in rows if r["slice"] == "eth_replication"]

    # B1: d_sharpe > +0.2 OR boot_lo > 0, on inner-validation, BOTH markets.
    b1_cells = []
    for r in inner_val_rows_primary:
        passes = (r["d_sharpe"] > SHARPE_NOISE_FLOOR) or (r["boot_lo"] > 0)
        b1_cells.append(dict(market=r["market"], passes=passes,
                             boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                             d_sharpe=r["d_sharpe"]))
    b1_pass = all(c["passes"] for c in b1_cells)

    # B2: diagnostic only -- drawdown improvement counts only where risk_matched.
    b2_cells = []
    for r in inner_val_rows_primary:
        b2_cells.append(dict(market=r["market"], risk_matched=r["risk_matched"],
                             d_dd=r["d_dd"], voided=not r["risk_matched"]))
    b2_pass = True  # never itself blocks promotion; diagnostic only

    # B3: plateau -- min_days in {60, 120, 250} at the primary floor, inner_val,
    # both markets. min_days=120 reused directly from the primary compare()
    # call's own inner_val rows (identical construction; not recomputed).
    plateau_rows: dict[int, list[dict]] = {}
    for md in B3_MIN_DAYS_GRID:
        if md == MIN_DAYS:
            plateau_rows[md] = [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                                     d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                                     vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                                     boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                                     boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                                for r in inner_val_rows_primary]
        else:
            bf = make_build_target(primary_floor, md)
            olabel = f"psr_sig_floor{primary_floor:g}_mind{md:d}"
            plateau_rows[md] = inner_val_rows(bf, olabel, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rr in plateau_rows.values() for r in rr]
    b3_pass_directionally_consistent = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0)

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
    b4_partial_pass = any(c["same_sign_as_btc"] for c in b4_cells)
    b4_full_pass = all(c["same_sign_as_btc"] for c in b4_cells)

    # B5: fee tier -- 0.40% taker, primary cell, inner-val, both markets, BTC only.
    hr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary cell, BTC inner-validation")
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    fee_rows = inner_val_rows(build_primary, label, btc, markets=fee_markets)
    for r in fee_rows:
        print(f"  {r['market']:>9s}  d_sharpe={r['d_sharpe']:+.3f}  "
              f"boot[{r['boot_lo']:+.3f},{r['boot_hi']:+.3f}]  excl0={r['excludes_zero']}")
    b5_cells = []
    for r in fee_rows:
        base = next((c for c in inner_val_rows_primary if c["market"] == r["market"]), None)
        d_sharpe_no_reversal = (base is not None and
                               not (np.sign(r["d_sharpe"]) != np.sign(base["d_sharpe"])
                                    and r["d_sharpe"] != 0 and base["d_sharpe"] != 0))
        dlog_no_reversal = (base is not None and
                          not (np.sign(r["boot_d_loggrowth"]) != np.sign(base["boot_d_loggrowth"])
                               and r["boot_d_loggrowth"] != 0 and base["boot_d_loggrowth"] != 0))
        b5_cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                             base_d_sharpe=base["d_sharpe"] if base else float("nan"),
                             boot_d_loggrowth=r["boot_d_loggrowth"],
                             base_boot_d_loggrowth=base["boot_d_loggrowth"] if base else float("nan"),
                             d_sharpe_no_reversal=d_sharpe_no_reversal,
                             dlog_no_reversal=dlog_no_reversal,
                             no_reversal=d_sharpe_no_reversal and dlog_no_reversal))
    b5_pass = all(c["no_reversal"] for c in b5_cells)

    all_pass = b1_pass and b2_pass and b3_pass_directionally_consistent and b4_full_pass and b5_pass

    return dict(
        label=label, floor=primary_floor, min_days=MIN_DAYS,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells, b2_pass=b2_pass,
        b3_plateau_rows=plateau_rows, b3_pass=b3_pass_directionally_consistent,
        b4_cells=b4_cells, b4_partial_pass=b4_partial_pass, b4_full_pass=b4_full_pass,
        b5_cells=b5_cells, b5_pass=b5_pass,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 6 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()

    hr("R-104 NOVEL: PSRSigKellyV4 -- Step-0 non-degeneracy grid")
    print("mechanism: continuous, closed-form Bailey & Lopez de Prado (2012) Probabilistic")
    print("Sharpe Ratio of the vote's own historical daily edge, recomputed from an")
    print("expanding window EVERY DAY (no periodic-refit knob), used directly as a")
    print("[floor, 1.0]-clipped discount on v4's own unchanged frac*scale.")
    print(f"\nTargetStrategy.warmup left at its shared default "
          f"({TargetStrategy.warmup:,} bars = {TargetStrategy.warmup / BARS_PER_DAY:g} "
          f"calendar days) -- see pre-registration item 3 for why the conservative "
          f"sibling's global reassignment was investigated and rejected as unsafe.")

    btc = load_btc()
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")
    assert_no_holdout(btc, "main(): btc")

    step0_rows, n_bars = step0_grid(btc)
    print_step0_table(step0_rows, n_bars)

    diag = hac_vs_psr_diagnostic(btc)
    print_hac_vs_psr(diag)

    build_default = make_build_target(PSR_FLOOR_DEFAULT, MIN_DAYS)
    hr("CAUSAL SAFETY -- truncation probe + explicit no-lookahead check")
    probe_ok = run_causal_probe(btc, build_default)
    lookahead = explicit_no_lookahead_check(btc, PSR_FLOOR_DEFAULT, MIN_DAYS)
    causal_safety_ok = probe_ok and lookahead["ok"]
    print(f"\nCAUSAL SAFETY (probe AND explicit no-lookahead check) PASS: {causal_safety_ok}")

    primary = select_primary(step0_rows)
    max_ts_seen = [btc.index.max()]

    if primary is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell has both bind_frac > 1% and r_sq < 0.98: the causal PSR")
        print("discount is either a near-total no-op or a near-exact rescale of v4's own")
        print("path everywhere on the pre-registered grid -- the R-87-shaped 'inert'")
        print("failure mode named in r104_shared.py's own docstring, by a different")
        print("estimator. Per this file's own pre-registration, this Step-0 table (plus")
        print("the diagnostic and causal-safety checks above) is the branch's ENTIRE")
        print("product, written up NEGATIVE / stopped-at-Step-0. No promotion-bar code")
        print("runs, and no ETH data or bar on/after 2023-01-01 is ever touched.")

        hr("VERDICT")
        print(f"Step-0 (3-cell floor grid, bind_frac>1% AND r_sq<0.98): FAIL (no cell qualifies)")
        print(f"causal safety (probe AND explicit no-lookahead check): {causal_safety_ok}")
        print("B1: NOT COMPUTED (Step-0 kill switch)")
        print("B2: NOT COMPUTED (Step-0 kill switch)")
        print("B3: NOT COMPUTED (Step-0 kill switch)")
        print("B4: NOT COMPUTED (Step-0 kill switch)")
        print("B5: NOT COMPUTED (Step-0 kill switch)")
        print("VERDICT: NEGATIVE (Step-0 kill switch)")

        n_configs = 3
        print(f"\nconfigurations evaluated (total): {n_configs} (3 Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {max(max_ts_seen)}  (< {OOS_START})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, diag=diag, primary=None,
                   passed_step0=False, probe_ok=probe_ok, lookahead=lookahead,
                   causal_safety_ok=causal_safety_ok, n_configs=n_configs,
                   verdict="NEGATIVE (Step-0 kill switch)")

    print(f"\nPRIMARY CELL SELECTED (non-degeneracy rule only): floor={primary['floor']:g}  "
          f"(bind_frac={primary['bind_frac']:.4f}, r_sq={primary['r_sq']:.4f})")
    is_center = (primary["floor"] == 0.5)
    print(f"  selection: {'grid-centre cell qualified' if is_center else 'grid-centre cell did NOT qualify; nearest qualifying cell in [0.5, 0.3, 0.7] chosen'}")

    eth = load_eth()
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")
    assert_no_holdout(eth, "main(): eth")
    max_ts_seen.append(eth.index.max())

    bar = run_promotion_bar(primary["floor"], btc, eth)

    hr("B1 -- inner-validation paired block-bootstrap (log-growth), both markets")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  boot_lo={c['boot_lo']:+.4f}  boot_hi={c['boot_hi']:+.4f}  "
              f"d_sharpe={c['d_sharpe']:+.4f}  PASS={c['passes']}")
    print(f"B1 PASS (all markets): {bar['b1_pass']}")

    hr("B2 -- risk-matched drawdown check (VOID unless risk_matched) -- diagnostic only")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hr("B3 -- plateau, not peak: min_days in {60, 120, 250}, primary floor, inner-val")
    print_plateau_table(bar["b3_plateau_rows"])
    print(f"\nB3 (directionally consistent region, not an isolated spike): {bar['b3_pass']}")

    hr("B4 -- ETH falsification (same-sign replication)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PARTIAL PASS (>=1 market): {bar['b4_partial_pass']}   "
          f"B4 FULL PASS (both markets): {bar['b4_full_pass']}")

    hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier d_sharpe={c['d_sharpe']:+.4f}  "
              f"standard-fee d_sharpe={c['base_d_sharpe']:+.4f}  "
              f"fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    hr("VERDICT")
    print(f"causal safety (probe AND explicit no-lookahead check): {causal_safety_ok}")
    print(f"B1={bar['b1_pass']}  B2=diagnostic-only  B3={bar['b3_pass']}  "
          f"B4_full={bar['b4_full_pass']}  B5={bar['b5_pass']}")
    all_applicable_pass = (causal_safety_ok and bar["b1_pass"] and bar["b3_pass"] and
                          bar["b4_full_pass"] and bar["b5_pass"])
    verdict = "PROMOTE-candidate" if all_applicable_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_applicable_pass}")
    print(f"VERDICT: {verdict}")
    if not causal_safety_ok:
        print("NOTE: verdict driven (at least in part) by a causal-safety check failure, "
              "not by B1-B5 alone -- the decision rule as pre-registered treats causal "
              "safety as a prerequisite to trusting ANY B1-B5 number, consistent with "
              "docs/ROUTINE.md's own precedence (a lookahead is a bug report first).")

    n_configs = 3 + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(3 Step-0 grid + 6 primary-cell compare() + 6 B3 plateau (3 min_days x 2 markets) "
          f"+ 2 fee-tier)")
    print(f"max timestamp read anywhere in this branch: {max(max_ts_seen)}  (< {OOS_START})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, diag=diag, primary=primary,
               passed_step0=True, probe_ok=probe_ok, lookahead=lookahead,
               causal_safety_ok=causal_safety_ok, promotion_bar=bar, verdict=verdict,
               n_configs=n_configs)


if __name__ == "__main__":
    main()
