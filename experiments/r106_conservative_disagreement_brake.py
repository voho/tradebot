#!/usr/bin/env python
"""R-106 CONSERVATIVE branch: ``DisagreementBrakeKellyV4`` -- ``kelly_regime_v4``'s
own unchanged ``frac * scale`` product, multiplied by a bounded, monotonic,
NEVER-INCREASE-ONLY discount driven by the literal Bomberger (1996)
cross-sectional-standard-deviation disagreement statistic over the four
already-built, structurally independent causal regime/turbulence detectors
(BOCPD, Kalman LLT, CSD, Hawkes) that ``experiments/r106_shared.py`` (READ-
ONLY infrastructure for both R-106 branches) builds. Full citation trail,
literature grounding (Zarnowitz & Lambros 1987; Bomberger 1996), the axis
this attacks (ERR -- no error control anywhere in the signal path), and the
exhaustive non-duplication argument against every related prior round all
live in ``r106_shared.py``'s own module docstring (read in full before this
file was written); not re-derived here beyond the one-paragraph summary
below. This file never edits, and never reads a bar at or after
``OOS_START`` from, ``r106_shared.py``, ``r106_novel_*.py``, or any other
file under ``experiments/``/``src/`` -- the one exception, disclosed and
scoped in section 6 below, is the pre-registered final HOLDOUT CONSULT this
round's own task brief explicitly requires as part of Step 4's reporting
(never as part of the frozen promotion decision, which is settled entirely
on Step-0/B1/B3/B4/B5 -- all computed on inner-train/inner-validation/ETH,
never on a holdout bar).

=====================================================================
PRE-REGISTRATION (frozen before any real-data R^2, Sharpe, or backtest
number in this file was computed -- docs/ROUTINE.md steps 1-2). Anything
below later contradicted by what actually happened is stated in the
results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): at every bar, compute the cross-sectional
   standard deviation of the four causally normalized [0,1] detector-alarm
   states (``r106_shared.cross_sectional_std`` of
   ``r106_shared.build_normalized_states`` -- literally Bomberger's own
   disagreement statistic), map it through a bounded, monotonic,
   NEVER-INCREASE-ONLY clipped-linear discount into ``[floor, 1.0]``, and
   multiply ``kelly_regime_v4``'s own unchanged ``frac * scale`` product by
   that discount before v4's own 10% deadband is applied.

2. CONSTRUCTION (exact):

       states[t]        = r106_shared.build_normalized_states(df)[t]     # 4 cols, [0,1]
       d[t]             = r106_shared.cross_sectional_std(states)[t]     # Bomberger disagreement
       D_MAX            = 1 / sqrt(3)   # see below -- a fixed, PRE-REGISTERED constant,
                                          # never fit to this round's own data
       frac[t]          = clip(d[t] / D_MAX, 0, 1)
       discount[t]       = 1.0                          if d[t] is NaN (insufficient
                                                          detector warmup -- see 3)
                          = 1.0 - (1.0 - floor) * frac[t]  otherwise
       raw[t]            = v4_raw_desired(df)[t] * discount[t]           # frac*scale, UNCHANGED, discounted
       target[t]         = apply_deadband(raw)[t]                       # v4's own deadband, AFTER discount

   ``D_MAX = 1/sqrt(3) ~= 0.5774`` is the THEORETICAL MAXIMUM of pandas'
   default (``ddof=1``) sample standard deviation of four numbers each
   individually bounded in ``[0,1]``: it is attained only by a perfect
   two-high/two-low split at the extremes (e.g. two states at 0.0, two at
   1.0), giving sample variance ``sum((x-0.5)^2)/(4-1) = 1/3``. Using the
   analytic bound rather than an empirically observed max keeps the
   discount map a fixed, PRE-REGISTERED function of ``d`` alone -- nothing
   about it is fit to this round's own realized disagreement distribution
   (which this file separately reports: observed max on inner-train is
   ~0.50, comfortably inside the analytic 0.577 bound).

   DEFAULT/PRIMARY CONFIG: ``floor`` selected by the Step-0 non-degeneracy
   rule below (grid centre 0.5 preferred per this project's own standing
   ``SELECTION_ORDER = (0.5, 0.3, 0.7)`` convention, confirmed or overridden
   only by that rule, never by a performance number). Grid, per the task
   brief, is EXACTLY ``r105_shared.STEP0_FLOOR_GRID = (0.3, 0.5, 0.7)`` --
   the same 3-cell grid R-104/R-105 used, reused verbatim rather than
   widened.

3. WHY THE NaN-FALLBACK IS `discount=1.0`, NOT AN INVENTED NUMBER: three of
   the four raw alarm scalars (BOCPD, CSD, Hawkes) and the shared
   percentile-rank normalization all require a real trailing history before
   they are defined (``r106_shared.MIN_PERIODS_DAYS=90`` for the percentile
   layer ALONE; empirically, on real BTC data, the SLOWEST member --
   Hawkes -- leaves the disagreement statistic undefined for the first
   ~298 days of any frame that starts at the very beginning of a dataset,
   verified below). During that warmup, "how much do the four detectors
   disagree" is simply unmeasured, not zero and not maximal -- so this
   construction falls back to ``discount=1.0`` (v4's own unmodified path),
   the same "insufficient information -> no adjustment" default every
   detector module in this project's own R-82/83/85/96 chain already uses
   for its own warmup, rather than inventing an artificial "cautious"
   number under missing information. This is DISCLOSED as a real, bounded
   limitation: the deeper implication -- that the framework's default
   80-day ``TargetStrategy.warmup`` (v4's own anchor requirement) is too
   short for THIS construction's genuine ~300-day detector warmup -- is
   handled explicitly in section 6 below (a disclosed, minimal deviation
   from bare ``r105_shared``/``r102_shared`` reuse), not silently absorbed
   into extra NaN-fallback bars that would understate the brake's effect
   for a data-availability reason rather than a genuine information one.

4. STEP-0 SANITY CHECK (this file's own, additional to the shared-module
   Step-0 gate in ``r106_shared.py`` -- already PASSED and NOT re-run here:
   mean pairwise |rho|=0.283 < 0.5, episode-mean CoV=0.189 > 5%): is the
   discount itself a near-constant multiplier (the "flat rescale" failure
   mode R-101's conservative branch and others hit), or does it genuinely
   vary bar-to-bar? For each floor in the pre-registered grid, on BTC's
   inner-train window (2017-01-01 -> 2020-12-31) only:
     - ``r_sq`` = ``r_squared(build_target(btc, floor=f), v4_target(btc))``,
       both over the FULL (pre-holdout) BTC frame, masked to inner-train
       rows before the R^2 is taken (identical convention to R-105's own
       Step-0 grid).
   KILL if ALL THREE floor cells have ``r_sq >= 0.98`` -- if even the most
   aggressive floor (0.3) is a near-exact rescale of v4's own path, the
   disagreement statistic never meaningfully diverges enough, on this
   grid, to matter.
   SELECTION (non-degeneracy only, no performance number inspected before
   this rule is applied): the primary cell is the first floor in
   ``r105_shared.SELECTION_ORDER = (0.5, 0.3, 0.7)`` with ``r_sq < 0.98``.
   If NONE qualifies, this file STOPS at Step-0 -- a legitimate NEGATIVE
   result, not a bug to route around. No B1-B5 code and no ETH/holdout data
   is read in that case.

5. CAUSAL TRUNCATION PROBE, run on real BTC data before trusting any
   Step-0 or promotion-bar number:
   ``r102_shared.causal_truncation_probe_series`` applied to this file's
   own composed ``build_target`` closure (bound to the selected primary
   floor, or ``floor=0.5`` if Step-0 finds no primary). In addition, an
   algebraic-identity self-test verifies ``build_target(df, floor=1.0)``
   (a floor OUTSIDE the pre-registered grid, used only as a wiring check)
   equals ``v4_target(df)`` exactly on both a small synthetic frame and
   real BTC data -- confirming the discount map, ``v4_raw_desired``, and
   ``apply_deadband`` are wired together correctly with no accidental
   double-deadband or scale error.

6. DISCLOSED DEVIATION FROM BARE ``r105_shared``/``r102_shared`` REUSE:
   this construction's real detector warmup (~298-300 days, driven by
   Hawkes) is longer than the framework's ``TargetStrategy`` default
   (v4's own 80-day anchor requirement, ``80*BARS_PER_DAY+10``). Calling
   ``r105_shared.compare()``/``inner_val_rows()``/``b5_fee_tier()`` VERBATIM
   would silently instantiate the candidate with the 80-day default,
   truncating ~7 months of inner-validation's own available prefix to the
   NaN->1.0 fallback for a data-availability reason rather than a genuine
   one (BTC has 4 full years of pre-2021 history available; the 80-day
   default just never asks for more of it). This file therefore defines
   ``compare_106``/``inner_val_rows_106`` -- BYTE-IDENTICAL to
   ``r102_shared.compare()``/``r105_shared.inner_val_rows()`` except that
   the CANDIDATE ``TargetStrategy`` is instantiated with
   ``warmup=CANDIDATE_WARMUP_BARS`` (320 days, a round-number margin above
   the observed 298-day first-valid point) instead of the 80-day class
   default. The CONTROL (``kelly_regime_v4``) strategy's warmup is left
   completely UNCHANGED (v4's own real, shipped 80-day requirement) in
   every call. The dict-only, no-strategy-construction gate helpers
   (``b1_from_inner_val``, ``b2_diagnostic``, ``b4_eth_falsification``) are
   reused VERBATIM from ``r105_shared`` (they only ever read already-
   computed row dicts, so the warmup change flows through them correctly
   without touching their code); ``b5_fee_tier``'s OWN LOGIC (sign-
   no-reversal check) is reproduced here as ``b5_fee_tier_106``, calling
   ``inner_val_rows_106`` instead of ``r105_shared.inner_val_rows``, for
   the same reason. Every other number in this file (candidate final
   balance and Sharpe with the wider warmup) is NOT directly comparable to
   R-101...R-105's own numbers on a bar-for-bar basis (the candidate warms
   longer), but the CONTROL side of every comparison is unchanged, so the
   d_sharpe/paired-bootstrap comparisons this round's promotion bar reads
   remain apples-to-apples between candidate and control.

7. PROMOTION BAR (frozen, ``docs/ROUTINE.md``'s own bar, per this round's
   task brief):
     B1 (gating, PRIMARY bar): ``b1_from_inner_val`` on the primary
        ``compare_106()`` call's own ``inner_val`` rows, BOTH BTC markets --
        ``d_sharpe > +0.2`` OR the paired bootstrap interval's lower bound
        excludes zero on the positive side.
     B2 (diagnostic ONLY, never gates): ``b2_diagnostic`` -- drawdown
        improvement counted only where risk_matched (R-33's own rule).
     B3 (plateau, gating): the SAME pre-registered 3-cell floor grid
        ``{0.3, 0.5, 0.7}`` (per the task brief -- NOT widened), inner-
        validation, both markets (6 rows; the primary floor's own 2 rows
        reused directly from the primary ``compare_106()`` call). PASS
        requires the primary cell's IMMEDIATE GRID NEIGHBOUR(S) (one for an
        edge cell, two for the centre) to show the SAME SIGN of
        ``d_sharpe`` as the primary cell, per market -- "not an isolated
        peak", exactly as specified, without requiring the neighbours to
        also clear +0.2.
     B4 (ETH falsification, gating, PRE-REGISTERED): ``b4_eth_falsification``
        -- require the FULL pass (both markets same-signed as BTC
        inner-validation).
     B5 (fee-tier robustness, gating): ``b5_fee_tier_106`` at the primary
        floor, 0.40% taker, both markets -- no sign reversal on either
        ``d_sharpe`` or the bootstrap log-growth point estimate.
   PROMOTE-candidate only if Step-0 selects a primary AND the causal probe
   passes AND B1 AND B3 AND B4 (full form) AND B5 all hold (B2 is
   diagnostic-only). Default: NEGATIVE. This decision rule is evaluated
   ENTIRELY on inner-train/inner-validation/ETH -- never on a holdout bar,
   regardless of what section 8's holdout consult below shows.

8. HOLDOUT CONSULT (run regardless of the section-7 verdict, per this
   round's task brief -- reported as ADDITIONAL EVIDENCE alongside, never
   as a substitute for, the frozen inner-validation decision rule): OOS
   (``start=OOS_START``) vs ``buy_and_hold``, both markets, standard fee
   tier; 0.40% taker re-run; real 2020-2023 Binance funding charged on the
   FUTURES leg, restricted to the observed overlap with the holdout
   (2023-01-01 -> 2023-12-31, ``scripts/funding_study.py``'s own
   ``measured()`` window and pattern, reproduced here with
   ``TargetStrategy`` in place of a registered strategy); the pre-
   registered falsification test; a reduced-effort Monte Carlo path-
   sensitivity check (disclosed as reduced scope given the session's time
   budget); and a deflated-Sharpe read using this file's own configuration
   count and this project's last-recorded project-wide trial count/
   dispersion (``scripts/inference.py``'s ``PROJECT_TRIALS=190``,
   ``SD_TRIALS_NARROW=0.223`` -- explicitly disclosed as stale by however
   many configurations this round and its NOVEL sibling add, since neither
   total is known to this file).

9. FALSIFICATION TEST (pre-registered, named now): the same risk R-105's
   own docstring named for its anchor-jackknife brake -- that disagreement
   might concentrate in exactly the bars where v4's OWN vote is mid-
   transition and about to resolve favourably, so that discounting there
   removes edge rather than protects against risk, making the brake
   ACTIVELY HARMFUL rather than merely inert. Operationalized exactly as
   specified: if ``d_sharpe`` is NEGATIVE and CONSISTENTLY so across the
   WHOLE pre-registered floor grid (all 3 floors, both markets, on inner-
   validation) -- that specific shape, not merely "B1 fails" -- the named
   risk is read as CONFIRMED. A supplementary (diagnostic-only, not
   gating) mechanistic read is also reported: mean disagreement in bars
   where v4's own vote will change within the next trading day ("mid-
   transition") vs bars where it will not, over inner-train.

CONFIGURATIONS EVALUATED IN THIS FILE (counted programmatically, printed at
the end; the estimate at pre-registration time, IF Step-0 selects a
primary): 3 (Step-0 grid) + 6 (primary compare_106(): inner_train x2
markets + inner_val x2 markets + eth_replication x2 markets) + 6 (B3's
3-floor x 2-market grid, 2 rows reused from primary) + 2 (B5's 0.40% fee
tier, 2 markets) + ~10 (section-8 holdout: candidate/control/buy_and_hold
x2 markets standard fee, candidate/control 0.40% fee x2 markets, candidate/
control funding-charged FUTURES x2) + ~12 (reduced Monte Carlo path check)
= ~39 total. IF Step-0 finds no qualifying cell, this file stops after the
3 Step-0 cells (no B1-B9 code, no ETH/holdout data ever read).

----------------------------------------------------------------------
Run: python experiments/r106_conservative_disagreement_brake.py
(from the repo root, with the project venv active)
----------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import deflated_sharpe_ratio, moments  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r106_shared import (  # noqa: E402
    BARS_PER_DAY,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    build_normalized_states,
    cross_sectional_std,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_target,
    v4_vote_frac,
)

# Reused verbatim from r105_shared -- general chain infrastructure, not the
# round's own read-only r106_shared/r106_novel_* files.
from experiments.r105_shared import (  # noqa: E402
    FEE_TIER,
    R2_THRESH,
    SELECTION_ORDER,
    SHARPE_NOISE_FLOOR,
    STEP0_FLOOR_GRID,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    hr,
    print_plateau_table,
)
from experiments.r102_shared import (  # noqa: E402
    causal_truncation_probe_series,
)

assert STEP0_FLOOR_GRID == (0.3, 0.5, 0.7), STEP0_FLOOR_GRID
assert SELECTION_ORDER == (0.5, 0.3, 0.7), SELECTION_ORDER

FLOOR_GRID = STEP0_FLOOR_GRID

# ---------------------------------------------------------- pre-registered
D_MAX = 1.0 / np.sqrt(3.0)          # analytic max of 4-value ddof=1 sample std, [0,1]-bounded
CANDIDATE_WARMUP_BARS = int(320 * BARS_PER_DAY)   # >= observed 298-day first-valid margin
FUNDING_HOLDOUT_END = "2023-12-31"  # observed Binance funding coverage ends here


# ================================================================== (1)
# The mechanism itself.
# ==================================================================

def disagreement_series(df: pd.DataFrame) -> pd.Series:
    """Bomberger (1996) cross-sectional disagreement statistic over the
    four normalized detector states, r106_shared's own construction,
    unmodified."""
    return cross_sectional_std(build_normalized_states(df))


def discount_from_disagreement(d: np.ndarray, floor: float) -> np.ndarray:
    """Bounded [floor, 1.0], monotonic NON-INCREASING (never scales UP
    exposure) clipped-linear map of the disagreement statistic onto a
    multiplicative brake. NaN disagreement (insufficient detector warmup,
    see docstring item 3) -> 1.0 (no brake), a disclosed default matching
    v4's own unmodified path rather than an invented number."""
    d = np.asarray(d, dtype=float)
    frac = np.clip(d / D_MAX, 0.0, 1.0)
    disc = 1.0 - (1.0 - floor) * frac
    return np.where(np.isfinite(d), disc, 1.0)


def build_target(df: pd.DataFrame, floor: float) -> np.ndarray:
    d = disagreement_series(df).to_numpy()
    disc = discount_from_disagreement(d, floor)
    raw = v4_raw_desired(df) * disc
    return apply_deadband(raw)


def make_build_target(floor: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, floor=floor)
    _build.__name__ = f"disagreement_brake_floor{floor:g}"
    return _build


# ================================================================== (2)
# Pre-flight self-tests.
# ==================================================================

def self_test_discount_bounds() -> bool:
    rng = np.random.default_rng(1060)
    d = rng.uniform(0.0, 1.0, size=2000)
    d[::37] = np.nan
    ok = True
    for floor in FLOOR_GRID:
        disc = discount_from_disagreement(d, floor)
        finite = np.isfinite(d)
        bounded = bool(np.all(disc[finite] >= floor - 1e-12) and np.all(disc[finite] <= 1.0 + 1e-12))
        nan_is_one = bool(np.allclose(disc[~finite], 1.0))
        monotonic_ok = True
        # discount must be non-increasing in d: sort by d and check.
        order = np.argsort(np.where(finite, d, -1))
        sorted_disc = disc[order][np.isfinite(d[order])]
        monotonic_ok = bool(np.all(np.diff(sorted_disc) <= 1e-9))
        print(f"  floor={floor:g}: bounded={bounded}  nan->1.0={nan_is_one}  "
              f"monotonic_non_increasing={monotonic_ok}")
        ok = ok and bounded and nan_is_one and monotonic_ok
    return ok


def self_test_floor1_identity(df: pd.DataFrame) -> bool:
    """floor=1.0 (OUTSIDE the pre-registered grid, wiring check only):
    discount === 1.0 regardless of disagreement -> build_target must equal
    v4_target exactly."""
    a = build_target(df, floor=1.0)
    b = v4_target(df)
    ok = np.allclose(a, b, equal_nan=True)
    print(f"  build_target(floor=1.0) == v4_target exactly? {ok}")
    return bool(ok)


def self_test_first_valid_disagreement(btc: pd.DataFrame) -> int:
    d = disagreement_series(btc)
    fv = d.first_valid_index()
    days = (fv - btc.index[0]).days if fv is not None else -1
    print(f"  first valid disagreement: {fv}  ({days} days after frame start)  "
          f"CANDIDATE_WARMUP_BARS covers {CANDIDATE_WARMUP_BARS / BARS_PER_DAY:.0f} days")
    return days


# ================================================================== (3)
# Step-0: near-constant-multiplier kill switch.
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> list[dict]:
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                       (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    d_full = disagreement_series(btc).to_numpy()
    ctrl_target = v4_target(btc)
    raw_base = v4_raw_desired(btc)

    rows = []
    for floor in FLOOR_GRID:
        disc = discount_from_disagreement(d_full, floor)
        target = apply_deadband(raw_base * disc)
        r_sq = r_squared(target[mask], ctrl_target[mask])
        disc_mask = disc[mask]
        rows.append(dict(
            floor=floor, r_sq=r_sq, qualifies=bool(r_sq < R2_THRESH),
            discount_mean=float(np.nanmean(disc_mask)),
            discount_std=float(np.nanstd(disc_mask)),
            discount_min=float(np.nanmin(disc_mask)),
            frac_below_0999=float(np.mean(disc_mask < 0.999)),
        ))
    return rows


def select_primary(rows: list[dict]) -> dict | None:
    by_floor = {r["floor"]: r for r in rows}
    for f in SELECTION_ORDER:
        r = by_floor.get(f)
        if r is not None and r["qualifies"]:
            return r
    return None


def print_step0_table(rows: list[dict]) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END})")
    print(f"QUALIFY = r_sq < {R2_THRESH} (candidate is NOT a near-exact rescale of v4)")
    hdr = (f"{'floor':>6s} {'r_sq':>8s} {'qualifies':>10s} {'disc_mean':>10s} "
          f"{'disc_std':>9s} {'disc_min':>9s} {'frac<0.999':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = " <- grid centre" if r["floor"] == 0.5 else ""
        print(f"{r['floor']:6.2f} {r['r_sq']:8.4f} {'YES' if r['qualifies'] else 'no':>10s} "
              f"{r['discount_mean']:10.4f} {r['discount_std']:9.4f} {r['discount_min']:9.4f} "
              f"{r['frac_below_0999']:11.4f}{tag}")


# ================================================================== (4)
# compare_106 / inner_val_rows_106 -- byte-identical to r102_shared.compare()
# / r105_shared.inner_val_rows() except the CANDIDATE TargetStrategy uses
# CANDIDATE_WARMUP_BARS instead of the 80-day framework default. See
# docstring item 6.
# ==================================================================

def compare_106(candidate_build, *, label: str, btc: pd.DataFrame, eth: pd.DataFrame,
                markets: tuple = (SPOT, FUTURES), include_eth: bool = True,
                seed: int = 0) -> list[dict]:
    assert_no_holdout(btc, "compare_106(): btc")
    if include_eth:
        assert_no_holdout(eth, "compare_106(): eth")

    cand = TargetStrategy(candidate_build, name=f"r106_{label}", warmup=CANDIDATE_WARMUP_BARS)
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")

    rows = []
    jobs = [(name, start, end, btc) for name, (start, end) in SLICES.items()]
    if include_eth:
        jobs.append((ETH_SLICE_NAME, None, None, eth))

    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice(cand, df, start, end, slice_name, market)
            b = run_slice(ctrl, df, start, end, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                        if b.mean_abs_exposure else float("nan"))
            vol_ratio = (a.realized_vol / b.realized_vol
                        if b.realized_vol else float("nan"))
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "cand_final": a.final_balance, "ctrl_final": b.final_balance,
                "cand_log_growth": a.log_growth, "ctrl_log_growth": b.log_growth,
                "d_log_growth": a.log_growth - b.log_growth,
                "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                "d_sharpe": a.sharpe - b.sharpe,
                "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
                "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
                "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
                "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
                "risk_matched": bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                                if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False,
                "boot_d_loggrowth": pr.diff.point,
                "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


def inner_val_rows_106(build_fn, label: str, btc: pd.DataFrame,
                       markets: tuple = (SPOT, FUTURES)) -> list[dict]:
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r106_{label}", warmup=CANDIDATE_WARMUP_BARS)
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


def b5_fee_tier_106(build_primary, label: str, btc: pd.DataFrame,
                    inner_val_primary: list[dict]) -> tuple[bool, list[dict]]:
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    fee_rows = inner_val_rows_106(build_primary, label, btc, markets=fee_markets)
    cells = []
    for r in fee_rows:
        base = next((c for c in inner_val_primary if c["market"] == r["market"]), None)
        d_sharpe_no_reversal = (base is not None and
                               not (np.sign(r["d_sharpe"]) != np.sign(base["d_sharpe"])
                                    and r["d_sharpe"] != 0 and base["d_sharpe"] != 0))
        dlog_no_reversal = (base is not None and
                          not (np.sign(r["boot_d_loggrowth"]) != np.sign(base["boot_d_loggrowth"])
                               and r["boot_d_loggrowth"] != 0 and base["boot_d_loggrowth"] != 0))
        cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                          base_d_sharpe=base["d_sharpe"] if base else float("nan"),
                          boot_d_loggrowth=r["boot_d_loggrowth"],
                          base_boot_d_loggrowth=base["boot_d_loggrowth"] if base else float("nan"),
                          d_sharpe_no_reversal=d_sharpe_no_reversal,
                          dlog_no_reversal=dlog_no_reversal,
                          no_reversal=d_sharpe_no_reversal and dlog_no_reversal))
    return all(c["no_reversal"] for c in cells), cells


# ================================================================== (5)
# B3: plateau over the pre-registered 3-cell floor grid.
# ==================================================================

def b3_plateau(primary_floor: float, plateau_rows: dict[float, list[dict]]) -> tuple[bool, list[dict]]:
    order = sorted(FLOOR_GRID)
    idx = order.index(primary_floor)
    neighbours = [order[i] for i in (idx - 1, idx + 1) if 0 <= i < len(order) and i != idx]
    primary_signs = {r["market"]: np.sign(r["d_sharpe"]) for r in plateau_rows[primary_floor]}
    detail = []
    ok = True
    for nb in neighbours:
        for r in plateau_rows[nb]:
            prim_sign = primary_signs.get(r["market"], 0.0)
            same = bool(np.sign(r["d_sharpe"]) == prim_sign)
            detail.append(dict(neighbour_floor=nb, market=r["market"], d_sharpe=r["d_sharpe"],
                               primary_d_sharpe=next(pr["d_sharpe"] for pr in plateau_rows[primary_floor]
                                                     if pr["market"] == r["market"]),
                               same_sign_as_primary=same))
            ok = ok and same
    return ok, detail


# ================================================================== (6)
# Falsification test + mechanistic diagnostic.
# ==================================================================

def falsification_test(plateau_rows: dict[float, list[dict]]) -> tuple[bool, list[dict]]:
    """Named risk CONFIRMED iff d_sharpe is negative for EVERY (floor,
    market) cell in the whole pre-registered grid -- the specific
    consistently-harmful shape, not merely 'B1 fails somewhere'."""
    cells = [dict(floor=f, market=r["market"], d_sharpe=r["d_sharpe"])
            for f in FLOOR_GRID for r in plateau_rows[f]]
    all_negative = all(c["d_sharpe"] < 0 for c in cells)
    return all_negative, cells


def midtransition_diagnostic(df: pd.DataFrame) -> dict:
    """Diagnostic only (never gates): does disagreement concentrate in
    bars where v4's own vote will change within the next trading day
    ('mid-transition') vs bars where it will not?"""
    vote = v4_vote_frac(df).to_numpy()
    d = disagreement_series(df).to_numpy()
    horizon = BARS_PER_DAY
    n = len(vote)
    if n <= horizon:
        return dict(mean_transition=float("nan"), mean_stable=float("nan"),
                    n_transition=0, n_stable=0)
    will_change = vote[:-horizon] != vote[horizon:]
    d_head = d[:-horizon]
    finite = np.isfinite(d_head)
    trans = finite & will_change
    stable = finite & ~will_change
    return dict(
        mean_transition=float(np.mean(d_head[trans])) if trans.any() else float("nan"),
        mean_stable=float(np.mean(d_head[stable])) if stable.any() else float("nan"),
        n_transition=int(trans.sum()), n_stable=int(stable.sum()),
    )


# ================================================================== (7)
# Section 8: holdout consult.
# ==================================================================

def load_btc_full() -> pd.DataFrame:
    """The FULL committed BTC spot series, INCLUDING the holdout -- used
    ONLY inside this section, never inside Step-0/B1-B5."""
    df, _label = load_dataset(ROOT / "data", "spot")
    return df


def _daily_from_equity(equity: pd.Series) -> np.ndarray:
    from tradebot.inference import daily_returns
    return daily_returns(equity).to_numpy()


def _total_log_return(d: np.ndarray) -> float:
    from tradebot.inference import total_log_return
    return float(total_log_return(np.asarray(d, dtype=float)))


def holdout_run(strategy, df_full: pd.DataFrame, market: MarketSpec,
                start: str = OOS_START, end: str | None = None,
                funding: pd.Series | None = None) -> dict:
    if funding is None:
        res = run_period(strategy, df_full, start, end, market=market, start_balance=1000.0)
    else:
        lo = int(df_full.index.searchsorted(pd.Timestamp(start, tz="UTC")))
        hi = (len(df_full) if end is None
             else int(df_full.index.searchsorted(pd.Timestamp(end, tz="UTC"), side="right")))
        prefix = min(lo, strategy.warmup)
        frame = df_full.iloc[lo - prefix: hi]
        raw = run_backtest(strategy, frame, market, 1000.0, trade_start=prefix, funding=funding)
        res = raw if prefix == 0 else replace(raw, equity=raw.equity.iloc[prefix:],
                                              df=raw.df.iloc[prefix:])
    m = compute_metrics(res)
    d = _daily_from_equity(res.equity)
    exposure = res.df["target"].to_numpy() if "target" in res.df.columns else None
    return dict(final=m.final_balance, sharpe=m.sharpe, dd=m.max_drawdown_pct,
               trades=m.num_trades, daily=d, log_growth=_total_log_return(d),
               mean_abs_exposure=float(np.nanmean(np.abs(exposure))) if exposure is not None else float("nan"),
               realized_vol=float(np.nanstd(d) * np.sqrt(365.25)) if len(d) > 1 else float("nan"))


def run_holdout_section(primary_floor: float) -> dict:
    hr("SECTION 8 -- HOLDOUT CONSULT (additional evidence; NOT part of the frozen decision rule)")
    build_primary = make_build_target(primary_floor)
    label = f"disagreement_brake_floor{primary_floor:g}"
    btc_full = load_btc_full()
    print(f"full BTC spot dataset: {len(btc_full):,} bars, "
          f"{btc_full.index[0]} -> {btc_full.index[-1]}  "
          f"(holdout portion: {OOS_START} onward)")

    cand = TargetStrategy(build_primary, name=f"r106_{label}", warmup=CANDIDATE_WARMUP_BARS)
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    hold = get_strategy("buy_and_hold")

    n_configs = 0
    hr("8a -- OOS vs buy_and_hold, both markets, standard fee tier")
    oos_rows = []
    for market in (SPOT, FUTURES):
        a = holdout_run(cand, btc_full, market)
        b = holdout_run(ctrl, btc_full, market)
        h = holdout_run(hold, btc_full, market)
        n_configs += 3
        pr_ctrl = paired_diff(a["daily"], b["daily"])
        pr_hold = paired_diff(a["daily"], h["daily"])
        exp_ratio = a["mean_abs_exposure"] / b["mean_abs_exposure"] if b["mean_abs_exposure"] else float("nan")
        vol_ratio = a["realized_vol"] / b["realized_vol"] if b["realized_vol"] else float("nan")
        row = dict(market=market.name,
                  cand_final=a["final"], ctrl_final=b["final"], hold_final=h["final"],
                  cand_sharpe=a["sharpe"], ctrl_sharpe=b["sharpe"], hold_sharpe=h["sharpe"],
                  d_sharpe_vs_ctrl=a["sharpe"] - b["sharpe"], d_sharpe_vs_hold=a["sharpe"] - h["sharpe"],
                  cand_dd=a["dd"], ctrl_dd=b["dd"], hold_dd=h["dd"],
                  exposure_ratio=exp_ratio, vol_ratio=vol_ratio,
                  boot_vs_ctrl_lo=pr_ctrl.diff.lo, boot_vs_ctrl_hi=pr_ctrl.diff.hi,
                  boot_vs_hold_lo=pr_hold.diff.lo, boot_vs_hold_hi=pr_hold.diff.hi,
                  daily=a["daily"])
        oos_rows.append(row)
        print(f"  {market.name:>11s}  cand=${a['final']:>10,.0f} (Sh {a['sharpe']:+.2f} DD {a['dd']:5.1f}%)  "
              f"ctrl=${b['final']:>10,.0f} (Sh {b['sharpe']:+.2f})  "
              f"hold=${h['final']:>10,.0f} (Sh {h['sharpe']:+.2f})  "
              f"dSh(vs ctrl)={row['d_sharpe_vs_ctrl']:+.3f}  dSh(vs hold)={row['d_sharpe_vs_hold']:+.3f}")
        print(f"    boot dlogG vs ctrl [{pr_ctrl.diff.lo:+.3f},{pr_ctrl.diff.hi:+.3f}]   "
              f"vs hold [{pr_hold.diff.lo:+.3f},{pr_hold.diff.hi:+.3f}]   "
              f"exposure_ratio={exp_ratio:.2f} vol_ratio={vol_ratio:.2f}")

    hr("8b -- 0.40% taker fee tier, holdout, both markets")
    fee40_rows = []
    for market in (SPOT, FUTURES):
        fee_mkt = fee_at(market, FEE_TIER)
        a = holdout_run(cand, btc_full, fee_mkt)
        b = holdout_run(ctrl, btc_full, fee_mkt)
        n_configs += 2
        base = next(r for r in oos_rows if r["market"] == market.name)
        d_sharpe = a["sharpe"] - b["sharpe"]
        no_reversal = not (np.sign(d_sharpe) != np.sign(base["d_sharpe_vs_ctrl"])
                          and d_sharpe != 0 and base["d_sharpe_vs_ctrl"] != 0)
        fee40_rows.append(dict(market=market.name, d_sharpe=d_sharpe,
                               base_d_sharpe=base["d_sharpe_vs_ctrl"], no_reversal=no_reversal))
        print(f"  {market.name:>11s}  @0.10%/0.05% dSh={base['d_sharpe_vs_ctrl']:+.3f}   "
              f"@0.40% dSh={d_sharpe:+.3f}   no_reversal={no_reversal}")

    hr("8c -- real 2020-2023 funding charged, FUTURES only, overlap with holdout "
       f"({OOS_START} -> {FUNDING_HOLDOUT_END})")
    real_funding = load_funding(ROOT / "data")
    funding_rows = None
    if real_funding is None:
        print("  no funding data committed -- skipped")
    else:
        a_free = holdout_run(cand, btc_full, FUTURES, start=OOS_START, end=FUNDING_HOLDOUT_END)
        a_paid = holdout_run(cand, btc_full, FUTURES, start=OOS_START, end=FUNDING_HOLDOUT_END, funding=real_funding)
        b_free = holdout_run(ctrl, btc_full, FUTURES, start=OOS_START, end=FUNDING_HOLDOUT_END)
        b_paid = holdout_run(ctrl, btc_full, FUTURES, start=OOS_START, end=FUNDING_HOLDOUT_END, funding=real_funding)
        n_configs += 4
        d_sharpe_free = a_free["sharpe"] - b_free["sharpe"]
        d_sharpe_paid = a_paid["sharpe"] - b_paid["sharpe"]
        funding_rows = dict(cand_free=a_free["final"], cand_paid=a_paid["final"],
                            ctrl_free=b_free["final"], ctrl_paid=b_paid["final"],
                            d_sharpe_free=d_sharpe_free, d_sharpe_paid=d_sharpe_paid,
                            no_reversal=bool(np.sign(d_sharpe_free) == np.sign(d_sharpe_paid)
                                            or d_sharpe_free == 0 or d_sharpe_paid == 0))
        print(f"  candidate: funding-free ${a_free['final']:>10,.0f} (Sh {a_free['sharpe']:+.2f})  "
              f"funding-charged ${a_paid['final']:>10,.0f} (Sh {a_paid['sharpe']:+.2f})")
        print(f"  control:   funding-free ${b_free['final']:>10,.0f} (Sh {b_free['sharpe']:+.2f})  "
              f"funding-charged ${b_paid['final']:>10,.0f} (Sh {b_paid['sharpe']:+.2f})")
        print(f"  dSharpe(cand-ctrl): funding-free={d_sharpe_free:+.3f}  funding-charged={d_sharpe_paid:+.3f}  "
              f"sign preserved={funding_rows['no_reversal']}")

    hr("8d -- reduced-effort Monte Carlo path sensitivity (SPOT only, disclosed reduced scope)")
    mc_rows = None
    N_MC = 12
    rng = np.random.default_rng(106)
    warmup_bars = max(CANDIDATE_WARMUP_BARS, ctrl.warmup) + 10
    min_days, max_days = 90, 365
    mc_cells = []
    if len(btc_full) > warmup_bars + max_days * BARS_PER_DAY:
        for k in range(N_MC):
            length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
            start = int(rng.integers(warmup_bars, len(btc_full) - length))
            window = btc_full.iloc[start - warmup_bars: start + length]
            eval_start = warmup_bars
            ra = run_backtest(cand, window, SPOT, 1000.0, trade_start=eval_start)
            rb = run_backtest(ctrl, window, SPOT, 1000.0, trade_start=eval_start)
            n_configs += 2
            ea, eb = ra.equity.to_numpy(dtype=float), rb.equity.to_numpy(dtype=float)
            base_a, base_b = ea[eval_start], eb[eval_start]
            ret_a = 100.0 * (ea[-1] / base_a - 1.0) if base_a > 0 else -100.0
            ret_b = 100.0 * (eb[-1] / base_b - 1.0) if base_b > 0 else -100.0
            mc_cells.append(dict(trial=k + 1, start=window.index[eval_start], days=length // BARS_PER_DAY,
                                 cand_ret_pct=ret_a, ctrl_ret_pct=ret_b, cand_beats_ctrl=ret_a > ret_b))
        beat_frac = float(np.mean([c["cand_beats_ctrl"] for c in mc_cells]))
        med_cand, med_ctrl = float(np.median([c["cand_ret_pct"] for c in mc_cells])), \
                             float(np.median([c["ctrl_ret_pct"] for c in mc_cells]))
        mc_rows = dict(n_trials=N_MC, beat_frac=beat_frac, median_cand_pct=med_cand, median_ctrl_pct=med_ctrl,
                      cells=mc_cells)
        print(f"  {N_MC} random SPOT windows ({min_days}-{max_days}d): candidate beats control in "
              f"{beat_frac:.0%} of windows; median return cand={med_cand:+.1f}% ctrl={med_ctrl:+.1f}%")
    else:
        print("  insufficient data length for the reduced MC window grid -- skipped")

    hr("8e -- deflated-Sharpe read (this file's own config count + last-recorded project totals)")
    from scripts.inference import PROJECT_TRIALS, SD_TRIALS_NARROW
    dsr_rows = []
    for row in oos_rows:
        d = row["daily"]
        skew, kurt = moments(d)
        dsr_last = deflated_sharpe_ratio(row["cand_sharpe"], len(d), skew, kurt,
                                        PROJECT_TRIALS, SD_TRIALS_NARROW)
        dsr_rows.append(dict(market=row["market"], sharpe=row["cand_sharpe"], n_obs=len(d),
                            skew=skew, kurt=kurt, n_trials=PROJECT_TRIALS, dsr=dsr_last))
        print(f"  {row['market']:>11s}  holdout Sharpe={row['cand_sharpe']:+.2f}  n_obs={len(d)}  "
              f"skew={skew:+.2f} kurt={kurt:.1f}  DSR@n_trials={PROJECT_TRIALS} (project total as of the "
              f"last committed run, NOT counting this round or its NOVEL sibling), "
              f"sd_trials={SD_TRIALS_NARROW}: {dsr_last:.3f}")
    print("  (0.95 is the conventional 'distinguishable from the best of n_trials coin flips' bar)")

    return dict(oos_rows=oos_rows, fee40_rows=fee40_rows, funding_rows=funding_rows,
               mc_rows=mc_rows, dsr_rows=dsr_rows, n_configs=n_configs)


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    hr("R-106 CONSERVATIVE: DisagreementBrakeKellyV4 -- Bomberger (1996) cross-model "
       "disagreement (BOCPD/Kalman/CSD/Hawkes) as a bounded, never-increase-only brake")
    print("mechanism: multiply v4's UNCHANGED frac*scale by a bounded, monotonic discount")
    print("driven by the cross-sectional std of four causally normalized detector-alarm")
    print("states (BOCPD, Kalman LLT, CSD, Hawkes) -- Bomberger's own disagreement statistic.")

    hr("PRE-FLIGHT SELF-TESTS (before any Step-0 number)")
    print("discount bounds/monotonicity/NaN-fallback (synthetic):")
    bounds_ok = self_test_discount_bounds()
    print(f"  -> bounds/monotonicity self-test: {bounds_ok}")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    print("\nfloor=1.0 wiring identity (real BTC data):")
    identity_ok = self_test_floor1_identity(btc)

    print("\nfirst-valid-disagreement / warmup check (real BTC data):")
    self_test_first_valid_disagreement(btc)

    if not (bounds_ok and identity_ok):
        print("\nSELF-TEST FAILURE -- stopping before any Step-0 number is trusted.")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="ABORTED (self-test failure)", max_ts=max(max_ts_seen))

    # ============================================================= STEP 0
    hr("STEP 0 -- NEAR-CONSTANT-MULTIPLIER KILL SWITCH (this file's own, in addition to "
       "r106_shared's already-passed cross-model correlation/CoV gate)")
    step0_rows = step0_grid(btc)
    print_step0_table(step0_rows)
    n_configs += len(FLOOR_GRID)

    primary = select_primary(step0_rows)

    if primary is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("Every floor in the pre-registered grid produces a near-exact rescale of v4's own")
        print("path on inner-train (r_sq >= 0.98 everywhere): the disagreement statistic never")
        print("diverges enough, at this magnitude, to move the exposure path meaningfully.")
        print("Per this file's own pre-registration, this Step-0 table is the branch's ENTIRE")
        print("product, reported NEGATIVE / stopped-at-Step-0. No B1-B5/holdout code runs.")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                   n_configs=n_configs, max_ts=max_ts, verdict="NEGATIVE (Step-0 kill switch)")

    is_center = (primary["floor"] == 0.5)
    print(f"\nPRIMARY CONFIG SELECTED (non-degeneracy rule only): floor={primary['floor']:g}  "
          f"(r_sq={primary['r_sq']:.4f}, discount_mean={primary['discount_mean']:.4f})")
    print(f"  selection: {'grid-centre cell qualified' if is_center else 'grid-centre cell did NOT qualify; nearest qualifying cell in SELECTION_ORDER chosen'}")

    build_primary = make_build_target(primary["floor"])
    label = f"disagreement_brake_floor{primary['floor']:g}"

    # ==================================================== CAUSAL PROBE
    hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data)")
    probe_ok = False
    try:
        causal_truncation_probe_series(build_primary, btc)
        probe_ok = True
        print("  PASS")
    except AssertionError as e:
        print(f"  FAIL: {e}")
    print(f"\nCAUSAL PROBE PASS: {probe_ok}")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    hr(f"PRIMARY compare_106() -- floor={primary['floor']:g}, "
       f"candidate warmup={CANDIDATE_WARMUP_BARS / BARS_PER_DAY:.0f}d (vs v4's own {(80*BARS_PER_DAY+10)/BARS_PER_DAY:.0f}d)")
    rows = compare_106(build_primary, label=label, btc=btc, eth=eth,
                       markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)
    n_configs += len(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    # ---- B1
    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)

    # ---- B2 (diagnostic only)
    b2_ok, b2_cells = b2_diagnostic(inner_val_primary)

    # ---- B3: same pre-registered 3-floor grid, both markets.
    plateau_rows: dict[float, list[dict]] = {
        primary["floor"]: [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                                d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                                vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                                boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                                boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                          for r in inner_val_primary]
    }
    for floor in FLOOR_GRID:
        if abs(floor - primary["floor"]) < 1e-9:
            continue
        bf = make_build_target(floor)
        blabel = f"disagreement_brake_floor{floor:g}"
        plateau_rows[floor] = inner_val_rows_106(bf, blabel, btc)
        n_configs += len(plateau_rows[floor])

    b3_pass, b3_detail = b3_plateau(primary["floor"], plateau_rows)

    # ---- B4: ETH falsification
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)

    # ---- B5: fee-tier robustness
    hr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary config, BTC inner-validation")
    b5_pass, b5_cells = b5_fee_tier_106(build_primary, label, btc, inner_val_primary)
    n_configs += len(b5_cells)

    hr("B1 -- inner-validation Sharpe leg, both markets "
       "(dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in b1_cells:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {b1_pass}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in b2_cells:
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  voided={c['voided']}")

    hr("B3 -- plateau: pre-registered floor grid {0.3,0.5,0.7} at primary selection, "
       "inner-validation, both markets")
    print_plateau_table(plateau_rows)
    for d in b3_detail:
        print(f"  neighbour floor={d['neighbour_floor']:g} {d['market']:>9s}  "
              f"d_sharpe={d['d_sharpe']:+.4f} (primary={d['primary_d_sharpe']:+.4f})  "
              f"same_sign_as_primary={d['same_sign_as_primary']}")
    print(f"\nB3 (primary cell's neighbour(s) share its sign): {b3_pass}")

    hr("B4 -- ETH falsification (pre-registered)")
    for c in b4_cells:
        print(f"  {c['market']:>9s}  ETH d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  same_sign_as_btc={c['same_sign_as_btc']}")
    print(f"B4 FULL PASS (both markets): {b4_full}")
    print(f"B4 PARTIAL PASS (at least one market): {b4_partial}")

    hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in b5_cells:
        print(f"  {c['market']:>9s}  @0.40% d_sharpe={c['d_sharpe']:+.4f}  "
              f"@0.40% boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"@0.10% boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {b5_pass}")

    hr("FALSIFICATION TEST -- consistently-harmful shape across the WHOLE pre-registered grid")
    named_risk_confirmed, fals_cells = falsification_test(plateau_rows)
    for c in fals_cells:
        print(f"  floor={c['floor']:g} {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}")
    print(f"named risk (disagreement discounts favourable mid-transition bars, brake is "
          f"ACTIVELY HARMFUL not merely inert) CONFIRMED: {named_risk_confirmed}")

    hr("supplementary mechanistic diagnostic (diagnostic ONLY, not gating): mean disagreement, "
       "mid-transition (vote changes within 1 day) vs stable bars, inner-train")
    mt = midtransition_diagnostic(btc[(btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                                      (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))])
    print(f"  mean disagreement | mid-transition bars (n={mt['n_transition']:,}): {mt['mean_transition']:.4f}")
    print(f"  mean disagreement | stable bars        (n={mt['n_stable']:,}): {mt['mean_stable']:.4f}")

    hr("VERDICT (Step-0/B1/B2/B3/B4/B5 -- inner-train/inner-validation/ETH ONLY, "
       "never a holdout bar)")
    all_gates_pass = probe_ok and b1_pass and b3_pass and b4_full and b5_pass
    verdict = "PROMOTE-candidate" if all_gates_pass else "NEGATIVE"
    print(f"causal probe = {probe_ok}   B1 = {b1_pass}   B2 = diagnostic-only   "
          f"B3 = {b3_pass}   B4(full) = {b4_full}   B4(partial) = {b4_partial}   B5 = {b5_pass}")
    print(f"\nALL GATING CLAUSES PASS (causal AND B1 AND B3 AND B4-full AND B5): {all_gates_pass}")
    print(f"VERDICT: {verdict}")
    if not all_gates_pass:
        failed = [name for name, ok in (
            ("causal probe", probe_ok), ("B1", b1_pass), ("B3", b3_pass),
            ("B4 (full)", b4_full), ("B5", b5_pass),
        ) if not ok]
        print(f"Reason(s): {', '.join(failed)}")

    # ============================================================= HOLDOUT
    holdout = run_holdout_section(primary["floor"])
    n_configs += holdout["n_configs"]

    # max_ts_seen tracks ONLY the Step-0/B1-B5 reads, which must stay < OOS_START; the
    # holdout section above reads bars >= OOS_START deliberately, per the task brief, and
    # is reported separately rather than folded into this guard.
    hr("SUMMARY")
    print(f"configurations evaluated (total, this file): {n_configs}")
    print(f"max timestamp read in Step-0/B1-B5 (must be < {OOS_START}): {max(max_ts_seen)}  "
          f"(< {OOS_START}: {max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")
    print(f"holdout consult: section 8 above reads bars >= {OOS_START} deliberately, per task brief.")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary, passed_step0=True,
               probe_ok=probe_ok, b1_cells=b1_cells, b1_pass=b1_pass, b2_cells=b2_cells,
               plateau_rows=plateau_rows, b3_pass=b3_pass, b3_detail=b3_detail,
               b4_cells=b4_cells, b4_full=b4_full, b4_partial=b4_partial,
               b5_cells=b5_cells, b5_pass=b5_pass,
               named_risk_confirmed=named_risk_confirmed, fals_cells=fals_cells,
               midtransition=mt, holdout=holdout,
               verdict=verdict, n_configs=n_configs)


if __name__ == "__main__":
    main()
