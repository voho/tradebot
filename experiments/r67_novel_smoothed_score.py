"""R-67 NOVEL branch -- Garleanu-Pedersen partial adjustment applied directly
to R-63's own k=1 target matrix (B-31's third named candidate).

The pre-registration for this round is the module docstring of
`experiments/r67_shared.py`, which is FROZEN and is NOT edited by this file.
This file implements one candidate and measures it; it does not define,
relax or re-derive a decision rule. (`git diff` on `r67_shared.py`,
`r65_shared.py`, `r63_shared.py` and `r63_novel_xsmom_rank.py` is empty for
this branch. Any flaw found in them is REPORTED, never fixed -- the R-63
violation, not repeated.)

=====================================================================
THE RULE, EXACTLY AS PRE-REGISTERED
=====================================================================

B-31's third named candidate: *"letting the partial-adjustment recursion
carry the position through a crossing instead of resetting it."*

    aim_t = r63_baseline_targets(aligned, k=1)[t]     # byte-for-byte
    x_0   = aim_0     at the FIRST BAR THE AIM IS LIVE (first row with any
                      non-zero entry; before that x is the zero vector,
                      which is what R-63's own arm holds there too)
    x_t   = x_{t-1} + a * (aim_t - x_{t-1})           # elementwise, GP(2013)

`a in (0, 1]` is the ONE free parameter. Nothing else in the construction
is touched: R-63's composite cross-sectional score, its conditional
volatility scale driven by the EQUAL-WEIGHT ALL-N basket, its 0.10 deadband
on desired total notional, its equal split among held slots, its 1.0 cap,
`k = 1`. The aim is obtained by CALLING R-63's `build_targets`, not by
re-implementing it, so the substrate cannot drift.

`a = 1.0` is an exact algebraic no-op and must reproduce R-63's k=1 arm.
That identity is checked FIRST, before any other number is read, and the
max absolute difference is reported whatever it is. NOTE, recorded before
the run: the pre-registered form `x + a*(aim - x)` is NOT guaranteed to be
bit-exact at `a = 1` in IEEE-754, because `x + (aim - x)` is not
associative; the algebraically identical convex form `(1-a)*x + a*aim` IS
bit-exact there. This file implements the PRE-REGISTERED form and reports
both differences rather than silently choosing the flattering one.

=====================================================================
HOW THIS DIFFERS FROM R-65's NOVEL ARM  (`r65_novel_aim_portfolio.py`)
=====================================================================

Both apply GP partial adjustment to a weight vector. Everything the
recursion is applied TO is different, and so is the parameter:

1. **The aim.** R-65's aim is a NEW construction: the three horizon
   components (20/40/80d) are kept separate, R-63's top-1 rule is applied
   to EACH of them, and the three one-hot vectors are blended with GP
   persistence weights `W_h ~ a/(a+phi_h)` normalised to sum to 1. It is a
   three-asset-at-a-time, multi-horizon, persistence-weighted portfolio
   that R-63 never built. THIS arm's aim is R-63's own committed
   `build_targets(aligned, 1)` output, unmodified -- a single composite
   score, one asset at a time.
2. **Where the vol scale and the deadband sit.** R-65 rebuilds the sizing
   around its own aim: it computes `s = sum_i x_i` AFTER smoothing and then
   applies `scale` and the 0.10 deadband to the SMOOTHED total. Here the
   scale and the deadband are already baked into the aim (they are R-63's)
   and the recursion runs on the finished, capped, deadbanded weight
   vector. That is the literal reading of B-31 -- see the design decision
   below.
3. **The parameter is fitted, not derived.** R-65 froze `a` at a value
   derived from theory and used its grid only as a plateau check. This
   round's pre-registration mandates a SWEPT `a` selected on W_VAL, and
   requires the derived rate to be reported ALONGSIDE as a diagnostic that
   cannot override the selection. Both are done below and kept separate.
4. **GP part (b) is absent entirely.** R-65 tested persistence weighting
   and found it inert (0.4% of the effect). There is nothing to weight
   here: R-63's composite is a single score. This arm is GP part (a) only,
   which is the half R-65 found was worth ~8.1 log units.
5. **No weight floor.** R-65 zeroed weights below `W_FLOOR = 1e-3` as
   "measurement hygiene". This arm has NO floor -- see the design decision
   on dust below. That is a substantive difference in what the arm holds,
   not a reporting one, and it is why the M1 vacuity caveat below matters
   here and did not there.

Consequence worth stating up front: this arm is NOT a re-run of R-65's, and
its `a = 1.0` endpoint is R-63's raw k=1 arm (turnover ~3.44/day), whereas
R-65's `a = 1` endpoint was its own multi-horizon aim.

=====================================================================
DESIGN DECISIONS -- DECLARED BEFORE ANY NUMBER WAS READ
=====================================================================

**(1) The deadband and the total-notional cap: smooth AFTER them.**
R-63 applies its 0.10 deadband to the DESIRED TOTAL NOTIONAL and then
splits that total equally among the held slots (at k=1: gives all of it to
the single held asset), then caps at 1.0. B-31's sentence is about
"the position" being carried through a crossing, and the pre-registration
says to take R-63's target matrix "byte-for-byte, unmodified, as the aim".
The only construction that satisfies both is: build R-63's finished target
matrix, then smooth it. That is the PRIMARY and the only thing swept.

  *Considered and rejected: smoothing BEFORE the deadband* (i.e. smooth the
  selection/one-hot, then apply `scale`, the deadband and the cap to the
  smoothed total, which is R-65's ordering). Rejected for three reasons,
  all recorded before running: (i) it requires re-implementing R-63's
  sizing block rather than calling `build_targets`, which is exactly the
  "byte-for-byte" clause of the pre-registration; (ii) it composes two
  smoothers -- a latch on top of a geometric filter -- so the measured
  effect could no longer be attributed to `a`; (iii) it is R-65's novel
  arm's ordering, and the one thing this branch must be able to say is
  precisely how it differs from that arm. It is NOT run at all, not even as
  an ablation, so it contributes zero configurations. Stated so the reader
  knows the choice was made in advance rather than after seeing a number.

**(2) Renormalisation: I do it myself, and it provably never fires.**
Every aim row is non-negative and sums to at most 1.0 (R-63 caps `total` at
1.0 and splits it). `x_t` is a convex combination of `x_{t-1}` and `aim_t`
for `a in (0,1]`, so by induction `x_t >= 0` elementwise and
`sum_i x_t,i <= 1` for every t. The clip to [0,1] and the
rescale-if-sum-exceeds-1 are therefore no-ops. They are applied anyway,
inside `build_targets`, rather than relying on `simulate_portfolio` doing
it -- so that the matrix written to CSV and the matrix the turnover and
membership statistics are computed from is the same matrix the simulator
trades. `checks` asserts the no-op numerically and reports max row-sum.

**(3) Dust: NO snap-to-zero, no floor, no ablation with one.**
`x_i` decays geometrically and reaches exactly zero only through float
underflow, so this arm holds a vanishing residual position in every asset
it has ever held, forever. R-66 spent an entire round establishing that
forcing a position to exactly flat is a claim in its own right, and it
lost. A dust floor would be that claim, smuggled in as hygiene, and R-65's
novel arm did exactly that (`W_FLOOR = 1e-3`). This arm does not, and no
floored ablation is run.

  The price of that choice is paid in the M1 statistic and is flagged
  loudly rather than banked: `membership_change_rate` counts changes to the
  indicator `weight > 0`, and an arm that never releases an asset has a
  near-zero membership-change rate for a reason that has NOTHING to do with
  B-31's mechanism. An M1 "pass" earned that way is VACUOUS. The M1 section
  below reports the pre-registered statistic, states plainly whether it is
  vacuous, and reports honest substitutes beside it (changes to the set of
  assets holding >1% of equity, the same at >0.1%, mean continuous tenure,
  and turnover/day, which is the quantity the cost side actually cares
  about and which no floor convention can distort).

**(4) `a` is a frozen scalar, never a function of the evaluation window.**
Grid values are constants. The derived-rate diagnostic's inputs are
measured on W_TRAIN ONLY. `build_targets` is therefore a pure causal
function of the price prefix; the truncation, causality and tail-x10
perturbation probes all check this and the slow-`a` case (a/64) is run
because a slow recursion has the longest memory and is where a leak would
hide.

=====================================================================
THE GRID AND THE SELECTION CRITERION -- FROZEN BEFORE THE SWEEP
=====================================================================

GRID (7 parameter-search cells, as pre-registered by the operator):

    a in {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01}

swept on **W_TRAIN** (2020-04-01 -> 2021-12-31, U8), selected on **W_VAL**
(2022-01-01 -> 2022-12-31, U8).

SELECTION CRITERION, declared here before the sweep ran and honoured as
written: the frozen configuration is the grid cell with the **highest W_VAL
net growth difference versus VOLMATCH_HOLD at 0.10%** -- the D1 decision
statistic evaluated on the selection window, which is R-63's and R-65's own
convention. Tie-break: the more negative W_VAL net drawdown difference. The
GROSS (0 bps) column is NOT filtered on at selection time: it is D5's
diagnostic and selecting on it would be selecting on this round's own
falsification test.

A cell whose `volmatched_hold_equity` returns `matched=False` on W_VAL is
VOIDED and cannot be selected (ROUTINE's standing rule); this is stated in
advance so it is not invented if it happens.

=====================================================================
THE DERIVED RATE -- A DIAGNOSTIC, ALONGSIDE THE SWEEP AND SEPARATE FROM IT
=====================================================================

R-65's strongest single piece of evidence was that a trading rate derived
from theory landed inside an independently-measured affordable window. The
analogue here, declared before running and NOT used to override the frozen
selection rule:

**Measurement (W_TRAIN ONLY, U8).** AR(1) through the origin,
`y_{t+1} = rho*y_t`, fitted by pooled OLS on the **cross-sectionally
demeaned composite score** `y_i(t) = s_i(t) - mean_j s_j(t)`, where `s` is
R-63's own `cross_sectional_score`. Demeaned because a long-only
cross-sectional rank rule only ever consumes the score's cross-sectional
dispersion -- the common level is not tradeable by a ranking. This is
R-65's own estimator applied to R-63's COMPOSITE (the quantity the k=1
rule actually ranks) rather than to its three separate horizon components,
which makes the number directly comparable to R-65's 3.34 / 5.83 / 9.52-day
component half-lives. `phi = -ln(rho)` per bar; half-life `ln2/phi`.

  SECONDARY, reported for the record and not used in either derivation: the
  same AR(1) on the AIM's own dynamics (R-63's k=1 target weight matrix,
  cross-sectionally demeaned), which folds in the deadband and the vol
  scale as well as the score.

**Two implied rates, both reported, neither a selection.**

  (D-a) GP cost-based rate, the same closed form R-65 used, re-derived here
        from W_TRAIN measurements:
            Delta* = R63_TURNOVER_PER_DAY / 288       (per-bar amplitude)
            lambda = 2*fee/Delta*                     (quadratic surrogate
                                                       charging the same
                                                       total cost as the
                                                       linear fee at Delta*)
            a_GP   = sigma * sqrt(gamma/lambda),  gamma = 1 (log utility),
                     sigma = per-bar sd of the EW U8 basket log return on
                     W_TRAIN.
        This depends only on the fee, R-63's measured turnover and sigma --
        NOT on the aim construction -- so it is expected to reproduce
        R-65's 0.00730886 exactly. That is a consistency check, not a new
        result, and it is labelled as one.

  (D-b) Signal-matched rate: set the adjustment half-life equal to the
        composite score's own measured half-life, `ln2/a = ln2/phi`, i.e.
        **a_HL = phi_composite**. This is the literal reading of R-65's own
        one-line lesson ("traded at its own decay rate") and is the rate
        that GP's aim-shrinkage factor `a/(a+phi)` puts at exactly 1/2.

Both are compared against the swept grid and against the W_VAL-selected
winner, and reported whether they agree or disagree.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- inherited from the frozen pre-registration
=====================================================================

(F1) the residual-long precedent (R-64/R-66): carrying a declining asset
through a crossing trades turnover for drawdown at unfavourable odds.
(F2) the forced-exit floor may be a property of the composite score's own
sign changes rather than of the gate, in which case smoothing only relabels
fast exits as slow fades. (F3) the frontier's far end is a concentrated
buy-and-hold; D5 and the fixed-permutation scramble are what catch it.
Named in `r67_shared.py`, restated here so a pass reads as a surprise.

W_HOLD IS NOT READ BY THIS BRANCH UNDER ANY OUTCOME. It is not sliced, not
imported, and the string "2023-01-01" appears in this file only inside this
sentence and in comments. Verified by grep, reported in the write-up.

Run as:
    python3 experiments/r67_novel_smoothed_score.py identity
    python3 experiments/r67_novel_smoothed_score.py derive
    python3 experiments/r67_novel_smoothed_score.py frontier
    python3 experiments/r67_novel_smoothed_score.py checks   [--a X]
    python3 experiments/r67_novel_smoothed_score.py m1       [--a X]
    python3 experiments/r67_novel_smoothed_score.py run      [--a X]
    python3 experiments/r67_novel_smoothed_score.py scramble [--a X]
    python3 experiments/r67_novel_smoothed_score.py all
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.r67_shared import (  # noqa: E402
    BARS_PER_DAY,
    D5_BAR,
    M1_MIN_REDUCTION,
    OUT_DIR,
    R63_TURNOVER_PER_DAY,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_FREE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_TRAIN,
    W_VAL,
    align_frames,
    basket_log_returns,
    check_causality,
    compare,
    config_count,
    cross_sectional_score,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    frontier_row,
    further_work,
    holding_period_days,
    load_universe,
    m1_pass,
    matched_hold_targets,
    mean_total_notional,
    membership_change_rate,
    r63_baseline_targets,
    r65_winner_targets,
    realized_vol,
    scramble_targets,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)

# --------------------------------------------------------------- constants

K = 1  # R-63's own frozen concentration. Not a parameter of this round.

# The pre-registered grid. 7 parameter-search cells, declared before the
# sweep and not modified afterwards.
A_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)

# Frozen by `select` from the W_VAL sweep, on the criterion declared in the
# docstring above, BEFORE any D-cell was touched. `None` until then, and the
# file was written with it as None -- `all` computes the selection itself and
# passes it forward, so no number can be back-filled here to change one.
FROZEN_A = None

# Derived-rate diagnostic constants. GAMMA and the lambda calibration are
# GP's, declared above; SIGMA and PHI are MEASURED on W_TRAIN only by
# `derive` (which re-measures and asserts these values still reproduce, so
# they cannot silently drift). They are frozen scalars, never functions of
# an evaluation window.
FEE = 0.001
GAMMA = 1.0
DELTA_STAR = R63_TURNOVER_PER_DAY / BARS_PER_DAY
LAMBDA_QUAD = 2.0 * FEE / DELTA_STAR
SIGMA_BAR_TRAIN = 0.00299076        # R-65's published W_TRAIN value; `derive`
                                    # re-measures it and reports the drift.
# MEASURED by `derive` on W_TRAIN only (U8, 184,320 bars), by the estimator
# declared in the docstring above BEFORE it was run; recorded here after the
# measurement so a re-run can assert it has not drifted. rho = 0.99959824,
# half-life 5.9894 days. Never used to select anything.
PHI_COMPOSITE_PER_BAR = 4.01839e-04
A_GP = SIGMA_BAR_TRAIN * math.sqrt(GAMMA / LAMBDA_QUAD)


# ------------------------------------------------------------------ signal


def build_targets(aligned: dict[str, pd.DataFrame], a: float) -> pd.DataFrame:
    """R-63's k=1 target matrix, carried through a GP partial-adjustment
    recursion. One free parameter, `a`.

    Pure causal function of the price prefix and the scalar `a`: no
    statistic of any kind is computed over the whole series here, and the
    aim is obtained by CALLING R-63's own builder.
    """
    if not (0.0 < a <= 1.0):
        raise ValueError(f"trading rate a must be in (0, 1]; got {a}")

    aim_df = r63_baseline_targets(aligned, K)
    aim = np.nan_to_num(aim_df.to_numpy(dtype=float), nan=0.0)
    n, m = aim.shape

    X = np.zeros((n, m))
    x = np.zeros(m)
    started = False
    for i in range(n):
        row = aim[i]
        if not started:
            if row.any():
                # x_0 = aim_0 at the first bar the aim is live.
                x = row.copy()
                started = True
            # before that the aim is the zero vector and so is x, which is
            # exactly what R-63's own arm holds on those bars.
        else:
            # THE PRE-REGISTERED FORM, verbatim. Not the convex rearrangement.
            x = x + a * (row - x)
        X[i] = x

    # Defensive only: proved a no-op in the docstring, asserted in `checks`.
    X = np.clip(X, 0.0, 1.0)
    s = X.sum(axis=1)
    over = s > 1.0
    if over.any():
        X[over] = X[over] / s[over][:, None]
    return pd.DataFrame(X, index=aim_df.index, columns=aim_df.columns)


def build_targets_convex(aligned: dict[str, pd.DataFrame], a: float) -> pd.DataFrame:
    """The algebraically identical convex form `(1-a)*x + a*aim`, used ONLY
    to quantify the float non-associativity in the identity check. Never
    used for a decision cell."""
    aim_df = r63_baseline_targets(aligned, K)
    aim = np.nan_to_num(aim_df.to_numpy(dtype=float), nan=0.0)
    n, m = aim.shape
    X = np.zeros((n, m))
    x = np.zeros(m)
    started = False
    for i in range(n):
        row = aim[i]
        if not started:
            if row.any():
                x = row.copy()
                started = True
        else:
            x = (1.0 - a) * x + a * row
        X[i] = x
    return pd.DataFrame(X, index=aim_df.index, columns=aim_df.columns)


# ------------------------------------------------------------------ cells


def _slice_index(warm: dict[str, pd.DataFrame], window):
    """STRICT right-exclusive slice. `r63_shared._hi`'s `end + 1 day`
    convention would admit the following day's 00:00 bar (documented
    amendment in that file); this branch never relies on it."""
    idx = next(iter(warm.values())).index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    return idx


_WARM_CACHE: dict = {}


def _warm_frames(frames, universe, window):
    """Memoize `align_frames`, a pure and expensive function. Changes no
    number; only stops the frontier realigning the same panel 16 times."""
    key = (tuple(universe), window)
    if key not in _WARM_CACHE:
        _WARM_CACHE[key] = align_frames({t: frames[t] for t in universe},
                                        warm_window(window))
    return _WARM_CACHE[key]


def build_cell(frames, universe, window, a, baseline=False, r65=False):
    """Aligned prices + targets, both sliced to the evaluation window.

    ``baseline=True`` builds R-63's own frozen k=1 arm; ``r65=True`` builds
    R-65's frozen conservative winner (the M1 reference). Both go through
    the identical warm-up/slice path as the candidate.
    """
    warm = _warm_frames(frames, universe, window)
    if baseline:
        targets = r63_baseline_targets(warm, K)
    elif r65:
        targets = r65_winner_targets(warm)
    else:
        targets = build_targets(warm, a)
    idx = _slice_index(warm, window)
    score = cross_sectional_score(warm).loc[idx]
    first_warm = bool(np.isfinite(score.iloc[0].to_numpy()).all())
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return aligned_eval, targets.loc[idx], first_warm


def evaluate(targets, aligned, assets, window_name, universe_name, arm, params):
    """One frontier row: 0.10% and 0 bps, both against VOLMATCH_HOLD."""
    out, cmps = {}, {}
    for tag, market in (("net", SPOT_BASE), ("gross", SPOT_FREE)):
        cand = simulate_portfolio(targets, aligned, market)
        bench, c, vol, matched = volmatched_hold_equity(cand, aligned, assets, market)
        if bench is None:
            raise RuntimeError(f"{arm} {window_name}: volmatch produced no benchmark")
        cmps[tag] = compare(cand, bench)
        out[f"{tag}_volmatch_c"] = c
        out[f"{tag}_volmatch_vol"] = vol
        out[f"{tag}_volmatch_matched"] = matched
        out[f"{tag}_cand_vol"] = realized_vol(cand)
    row = frontier_row(arm, params, targets, cmps["net"], cmps["gross"],
                       "VOLMATCH_HOLD", window_name, universe_name, **out)
    row.update(membership_stats(targets))
    return row


# ------------------------------------------------------- honest statistics


def _change_rate(held: np.ndarray, n_bars: int) -> float:
    changes = int((held[1:] != held[:-1]).any(axis=1).sum())
    days = max(n_bars / BARS_PER_DAY, 1e-9)
    return changes / days


def membership_stats(targets: pd.DataFrame) -> dict:
    """The pre-registered M1 statistic and the honest substitutes for it.

    `membership_change_rate` (and `holding_period_days`, which is its
    reciprocal) count changes to the indicator `weight > 0`. A continuously
    smoothed arm with no dust floor never releases an asset, so that
    indicator can freeze at "everything held" and the rate collapses to
    ~zero for a reason unrelated to B-31's mechanism. The `>1%` and `>0.1%`
    variants and the turnover are reported beside it so the reader can see
    which is happening.
    """
    w = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0)
    n = len(w)
    days = max(n / BARS_PER_DAY, 1e-9)
    held0 = w > 0.0
    held1 = w > 0.01     # >1% of equity
    held01 = w > 0.001   # >0.1% of equity
    starts1 = int(held1[0].sum() + (held1[1:] & ~held1[:-1]).sum())
    return {
        "m1_rate_gt0_per_day": _change_rate(held0, n),
        "chg_rate_gt1pct_per_day": _change_rate(held1, n),
        "chg_rate_gt0p1pct_per_day": _change_rate(held01, n),
        "mean_n_held_gt0": float(held0.sum(axis=1).mean()),
        "mean_n_held_gt1pct": float(held1.sum(axis=1).mean()),
        "frac_bars_all_assets_gt0": float((held0.all(axis=1)).mean()),
        "mean_tenure_gt1pct_days": float(held1.sum()) / max(starts1, 1) / BARS_PER_DAY,
        "days": days,
    }


# ------------------------------------------------------------------ io


def write_csv(path, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {path}")
    return path


def fmt_front(r):
    return (f"    hold {r['hold_days']:8.3f}d | turn {r['turnover_per_day']:8.4f}/d"
            f" | mtn {r['mean_notional']:.3f}"
            f" | GROSS {r['gross_growth_diff']:+8.3f}"
            f" [{r['gross_growth_lo']:+7.3f},{r['gross_growth_hi']:+7.3f}]"
            f" | NET {r['net_growth_diff']:+8.3f}"
            f" [{r['net_growth_lo']:+7.3f},{r['net_growth_hi']:+7.3f}]"
            f" | ddiff {r['net_dd_diff']:+7.2f}")


# ------------------------------------------------------------- 1. identity


def cmd_identity(frames):
    """a = 1.0 must reproduce R-63's own k=1 arm. Run before anything else."""
    print("== IDENTITY CHECK: a=1.0 vs r63_baseline_targets(aligned, 1) ==")
    rows = []
    for uni_name, uni, window in (("U8", UNIVERSE_8, W_TRAIN),
                                  ("U6", UNIVERSE_6, W_FULL6)):
        warm = _warm_frames(frames, uni, window)
        base = r63_baseline_targets(warm, K)
        mine = build_targets(warm, 1.0)
        cvx = build_targets_convex(warm, 1.0)
        b = np.nan_to_num(base.to_numpy(dtype=float), nan=0.0)
        m = np.nan_to_num(mine.to_numpy(dtype=float), nan=0.0)
        c = np.nan_to_num(cvx.to_numpy(dtype=float), nan=0.0)
        d_pre = float(np.abs(m - b).max())
        d_cvx = float(np.abs(c - b).max())
        exact = bool(np.array_equal(m, b))
        print(f"  {uni_name} {window[0]}->{window[1] or 'last'}  bars {len(b):,}")
        print(f"    pre-registered form  x + a*(aim-x):  max|diff| = {d_pre:.3e}"
              f"   bit-identical: {exact}")
        print(f"    convex form  (1-a)*x + a*aim:        max|diff| = {d_cvx:.3e}"
              f"   bit-identical: {bool(np.array_equal(c, b))}")
        # sum/clip no-op evidence, on the same matrices
        raw_max_rowsum = float(m.sum(axis=1).max())
        print(f"    max row sum of the smoothed matrix at a=1.0: {raw_max_rowsum:.12f}")
        rows.append({"universe": uni_name, "window": f"{window[0]}..{window[1]}",
                     "n_bars": len(b), "max_abs_diff_preregistered": d_pre,
                     "bit_identical_preregistered": exact,
                     "max_abs_diff_convex": d_cvx,
                     "bit_identical_convex": bool(np.array_equal(c, b)),
                     "max_row_sum": raw_max_rowsum})
    write_csv(OUT_DIR / "novel_identity.csv", rows)
    return rows


# --------------------------------------------------------------- 2. derive


def _ar1_through_origin(y: np.ndarray) -> float:
    a_, b_ = y[:-1].ravel(), y[1:].ravel()
    m = np.isfinite(a_) & np.isfinite(b_)
    return float((a_[m] * b_[m]).sum() / (a_[m] * a_[m]).sum())


def cmd_derive(frames):
    """The derived-rate DIAGNOSTIC. W_TRAIN only. Never a selection."""
    print("== DERIVED RATE (diagnostic, W_TRAIN ONLY, U8) ==")
    warm = _warm_frames(frames, UNIVERSE_8, W_TRAIN)
    idx = _slice_index(warm, W_TRAIN)
    print(f"  fit index: {len(idx):,} bars  {idx[0]} -> {idx[-1]}")
    assert idx[-1] < pd.Timestamp("2022-01-01", tz="UTC"), "fit window leaked past W_TRAIN"

    score = cross_sectional_score(warm).loc[idx].to_numpy(dtype=float)
    y = score - np.nanmean(score, axis=1, keepdims=True)
    rho = _ar1_through_origin(y)
    phi = -math.log(rho)
    hl_days = math.log(2) / phi / BARS_PER_DAY
    rho_raw = _ar1_through_origin(score)
    hl_raw = math.log(2) / -math.log(rho_raw) / BARS_PER_DAY

    # SECONDARY: the aim's own dynamics (R-63's k=1 weight matrix).
    aim = r63_baseline_targets(warm, K).loc[idx].to_numpy(dtype=float)
    ya = aim - aim.mean(axis=1, keepdims=True)
    rho_aim = _ar1_through_origin(ya)
    phi_aim = -math.log(rho_aim) if rho_aim > 0 else float("nan")
    hl_aim = math.log(2) / phi_aim / BARS_PER_DAY if np.isfinite(phi_aim) else float("nan")

    r = basket_log_returns(warm).loc[idx].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    sigma = float(np.std(r, ddof=1))

    a_gp = sigma * math.sqrt(GAMMA / LAMBDA_QUAD)
    a_hl = phi

    print(f"  composite score, xs-demeaned: rho={rho:.8f}  phi={phi:.5e}/bar"
          f"  half-life {hl_days:.4f} d   (raw, unused: {hl_raw:.4f} d)")
    print(f"  SECONDARY aim (k=1 weights, demeaned): rho={rho_aim:.8f}"
          f"  half-life {hl_aim:.4f} d")
    print(f"  sigma (per-bar EW U8 basket sd) = {sigma:.8f}"
          f"  ({sigma * math.sqrt(365.25 * BARS_PER_DAY) * 100:.1f}% ann)")
    print(f"  (D-a) GP cost-based: Delta*={DELTA_STAR:.8f} lambda={LAMBDA_QUAD:.8f}"
          f" -> a_GP = {a_gp:.8f}  (adjust half-life "
          f"{math.log(2)/a_gp/BARS_PER_DAY:.4f} d)")
    print(f"  (D-b) signal-matched: a_HL = phi_composite = {a_hl:.8f}"
          f"  (adjust half-life {math.log(2)/a_hl/BARS_PER_DAY:.4f} d)")

    drift_s = abs(sigma - SIGMA_BAR_TRAIN) / SIGMA_BAR_TRAIN
    print(f"  sigma drift vs R-65's published W_TRAIN value: {drift_s:.2e}")
    if PHI_COMPOSITE_PER_BAR is not None:
        print(f"  phi drift vs the recorded constant: "
              f"{abs(phi - PHI_COMPOSITE_PER_BAR) / PHI_COMPOSITE_PER_BAR:.2e}")

    print("  where the derived rates sit relative to the pre-registered grid:")
    for a in A_GRID:
        print(f"    a={a:<5} : a/a_GP = {a / a_gp:9.2f}x   a/a_HL = {a / a_hl:9.2f}x")

    rows = [{"quantity": "composite_score_xs_demeaned", "rho_per_bar": rho,
             "phi_per_bar": phi, "half_life_days": hl_days,
             "rho_raw": rho_raw, "half_life_days_raw": hl_raw,
             "n_bars": len(idx)},
            {"quantity": "aim_k1_weights_demeaned_SECONDARY",
             "rho_per_bar": rho_aim, "phi_per_bar": phi_aim,
             "half_life_days": hl_aim, "n_bars": len(idx)},
            {"quantity": "derivation", "sigma_bar_train": sigma,
             "delta_star": DELTA_STAR, "lambda_quad": LAMBDA_QUAD,
             "gamma": GAMMA, "fee": FEE, "a_gp_cost_based": a_gp,
             "a_hl_signal_matched": a_hl,
             "a_gp_half_life_days": math.log(2) / a_gp / BARS_PER_DAY,
             "a_hl_half_life_days": math.log(2) / a_hl / BARS_PER_DAY,
             "r65_a_derived_published": 0.00730886}]
    write_csv(OUT_DIR / "novel_derived.csv", rows)
    return {"phi": phi, "half_life_days": hl_days, "sigma": sigma,
            "a_gp": a_gp, "a_hl": a_hl}


# ------------------------------------------------------------- 3. frontier


def cmd_frontier(frames):
    """Every grid cell on W_TRAIN and W_VAL (U8). The decision window is NOT
    touched here."""
    print("== FRONTIER: 7 cells x {W_TRAIN, W_VAL}, U8, vs VOLMATCH_HOLD "
          "@0.10% and 0bps ==")
    rows = []
    for wname, window in (("W_TRAIN", W_TRAIN), ("W_VAL", W_VAL)):
        print(f"  -- {wname} --")
        aligned, tg, warm_ok = build_cell(frames, UNIVERSE_8, window, 1.0,
                                          baseline=True)
        r = evaluate(tg, aligned, UNIVERSE_8, wname, "U8", "R63_BASELINE_k1",
                     {"a": float("nan")})
        r["kind"] = "reference"
        r["first_bar_warm"] = warm_ok
        rows.append(r)
        print("  R63_BASELINE_k1 (reference, not a grid cell)")
        print(fmt_front(r))

        for a in A_GRID:
            aligned, tg, warm_ok = build_cell(frames, UNIVERSE_8, window, a)
            r = evaluate(tg, aligned, UNIVERSE_8, wname, "U8", "smoothed_score",
                         {"a": a})
            r["kind"] = "grid"
            r["first_bar_warm"] = warm_ok
            rows.append(r)
            print(f"  a={a}")
            print(fmt_front(r))
            print(f"      volmatch matched net={r['net_volmatch_matched']} "
                  f"(c={r['net_volmatch_c']:.3f}) gross={r['gross_volmatch_matched']}"
                  f"  | m1_rate {r['m1_rate_gt0_per_day']:.4f}/d "
                  f"chg>1% {r['chg_rate_gt1pct_per_day']:.4f}/d")

    write_csv(OUT_DIR / "novel_frontier.csv", rows)
    return rows


def cmd_select(rows):
    """The frozen selection rule, applied mechanically to the W_VAL rows."""
    print("== SELECTION (criterion declared in the module docstring "
          "before the sweep) ==")
    cands = [r for r in rows if r["window"] == "W_VAL" and r["kind"] == "grid"]
    voided = [r for r in cands if not r["net_volmatch_matched"]]
    for r in voided:
        print(f"  a={r['p_a']}: VOLMATCH did not match on W_VAL -> VOIDED, "
              f"not selectable")
    live = [r for r in cands if r["net_volmatch_matched"]]
    live.sort(key=lambda r: (-r["net_growth_diff"], r["net_dd_diff"]))
    print("  W_VAL ordering by net growth diff vs VOLMATCH_HOLD @0.10%:")
    for i, r in enumerate(live):
        print(f"    {i+1}. a={r['p_a']:<6} net {r['net_growth_diff']:+8.4f}"
              f"  dd {r['net_dd_diff']:+7.3f}  gross {r['gross_growth_diff']:+8.4f}"
              f"  turn {r['turnover_per_day']:.4f}/d")
    win = live[0]
    print(f"  -> FROZEN a = {win['p_a']}")
    return float(win["p_a"])


# --------------------------------------------------------------- 4. checks


def cmd_checks(frames, a):
    print(f"== CORRECTNESS GATES (at the selected a={a}) ==")
    warm = _warm_frames(frames, UNIVERSE_8, W_TRAIN)
    n = len(next(iter(warm.values())))
    out = {}

    # (b) 60% truncation test, bit-identical on overlapping rows.
    cut = int(n * 0.6)
    full = build_targets(warm, a)
    trunc = build_targets({t: df.iloc[:cut] for t, df in warm.items()}, a)
    af = np.nan_to_num(full.iloc[:cut].to_numpy(), nan=0.0)
    bf = np.nan_to_num(trunc.to_numpy(), nan=0.0)
    exact60 = bool(np.array_equal(af, bf))
    close60 = bool(np.allclose(af, bf, atol=1e-12, rtol=0.0))
    dmax = float(np.abs(af - bf).max())
    print(f"  truncation @60% ({cut:,}/{n:,} bars): bit-identical={exact60}"
          f"  atol=1e-12 ok={close60}  max|diff|={dmax:.3e}")
    # diagnose the initial condition if it ever fails
    aim = np.nan_to_num(r63_baseline_targets(warm, K).to_numpy(), nan=0.0)
    first_live = int(np.argmax(aim.any(axis=1)))
    print(f"    first bar the aim is live: row {first_live:,} "
          f"({'inside' if first_live < cut else 'OUTSIDE'} the 60% prefix) "
          f"-> initial condition is {'unchanged' if first_live < cut else 'SHIFTED'} "
          f"under truncation")
    out.update(trunc60_exact=exact60, trunc60_atol1e12=close60,
               trunc60_max_abs_diff=dmax, aim_first_live_row=first_live,
               trunc60_cut_row=cut)

    # (a) r63_shared's own truncation probe, at a and at a/64.
    c1 = check_causality(lambda al: build_targets(al, a), warm)
    a64 = a / 64.0
    c2 = check_causality(lambda al: build_targets(al, a64), warm)
    print(f"  check_causality(a={a}): {c1}")
    print(f"  check_causality(a={a}/64={a64:g}): {c2}")
    out.update(causality_at_a=c1, causality_at_a_over_64=c2, a_over_64=a64)

    # tail-x10 perturbation probe (the whole-series-statistic hunt)
    cutp = int(n * 0.6)
    bad = {}
    for t, df in warm.items():
        d = df.copy()
        for col in ("open", "high", "low", "close"):
            v = d[col].to_numpy(dtype=float).copy()
            v[cutp:] *= 10.0
            d[col] = v
        bad[t] = d
    pa = np.nan_to_num(build_targets(warm, a).to_numpy()[:cutp], nan=0.0)
    pb = np.nan_to_num(build_targets(bad, a).to_numpy()[:cutp], nan=0.0)
    probe = bool(np.allclose(pa, pb, atol=1e-12, rtol=0.0))
    print(f"  perturbation probe (tail x10, early rows unchanged): {probe}")
    out["perturbation_probe"] = probe

    # renormalisation no-op evidence
    W = full.to_numpy()
    print(f"  max row sum {W.sum(axis=1).max():.12f}  min entry {W.min():.3e}"
          f"  -> clip/rescale fired: {bool(W.sum(axis=1).max() > 1.0)}")
    out.update(max_row_sum=float(W.sum(axis=1).max()), min_entry=float(W.min()))

    # monotone response of turnover to `a`
    idx = _slice_index(warm, W_TRAIN)
    resp = []
    print("  turnover / membership response to `a` (W_TRAIN, U8):")
    for av in A_GRID:
        tg = build_targets(warm, av).loc[idx]
        ts = turnover_stats(tg)
        ms = membership_stats(tg)
        resp.append(ts["turnover_per_day"])
        print(f"    a={av:<5} turn {ts['turnover_per_day']:8.4f}/d "
              f"reb {ts['rebalances_per_day']:8.3f}/d "
              f"hold {holding_period_days(tg):8.3f}d "
              f"mtn {mean_total_notional(tg):.3f} "
              f"m1 {ms['m1_rate_gt0_per_day']:.4f}/d "
              f">1% {ms['chg_rate_gt1pct_per_day']:.4f}/d")
    mono = all(resp[i] >= resp[i + 1] for i in range(len(resp) - 1))
    print(f"  turnover non-increasing as `a` falls: {mono}")
    out["turnover_monotone_in_a"] = mono

    write_csv(OUT_DIR / "novel_checks.csv", [out])
    return out


# ------------------------------------------------------------------- 5. M1


def cmd_m1(frames, a):
    """The round's mechanism gate, plus the vacuity diagnosis it needs."""
    print("== M1 (mechanism gate), W_TRAIN, U8 ==")
    warm = _warm_frames(frames, UNIVERSE_8, W_TRAIN)
    idx = _slice_index(warm, W_TRAIN)
    base = r65_winner_targets(warm).loc[idx]
    base_ms = membership_stats(base)
    base_ts = turnover_stats(base)
    print(f"  BASELINE r65_winner (k=1, buffer=0.05, hold_days=1): "
          f"m1 {base_ms['m1_rate_gt0_per_day']:.4f}/d  "
          f">1% {base_ms['chg_rate_gt1pct_per_day']:.4f}/d  "
          f"turn {base_ts['turnover_per_day']:.4f}/d  "
          f"mean held>0 {base_ms['mean_n_held_gt0']:.3f}")

    rows = []
    for av in A_GRID:
        tg = build_targets(warm, av).loc[idx]
        m1 = m1_pass(tg, base)
        ms = membership_stats(tg)
        ts = turnover_stats(tg)
        vac = ms["frac_bars_all_assets_gt0"] > 0.5
        red1 = 1.0 - (ms["chg_rate_gt1pct_per_day"]
                      / base_ms["chg_rate_gt1pct_per_day"])
        row = {"arm": "smoothed_score", "p_a": av, "window": "W_TRAIN",
               "universe": "U8",
               "m1_cand_rate_per_day": m1["cand_rate_per_day"],
               "m1_baseline_rate_per_day": m1["baseline_rate_per_day"],
               "m1_reduction": m1["reduction"], "m1_passed": m1["passed"],
               "m1_vacuous": vac,
               "chg_rate_gt1pct_per_day": ms["chg_rate_gt1pct_per_day"],
               "baseline_chg_rate_gt1pct_per_day":
                   base_ms["chg_rate_gt1pct_per_day"],
               "reduction_gt1pct": red1,
               "chg_rate_gt0p1pct_per_day": ms["chg_rate_gt0p1pct_per_day"],
               "turnover_per_day": ts["turnover_per_day"],
               "baseline_turnover_per_day": base_ts["turnover_per_day"],
               "turnover_reduction": 1.0 - ts["turnover_per_day"]
                                     / base_ts["turnover_per_day"],
               "mean_n_held_gt0": ms["mean_n_held_gt0"],
               "mean_n_held_gt1pct": ms["mean_n_held_gt1pct"],
               "frac_bars_all_assets_gt0": ms["frac_bars_all_assets_gt0"],
               "mean_tenure_gt1pct_days": ms["mean_tenure_gt1pct_days"],
               "selected": bool(abs(av - a) < 1e-15)}
        rows.append(row)
        flag = "  <-- SELECTED" if row["selected"] else ""
        print(f"  a={av:<5} M1 rate {m1['cand_rate_per_day']:9.4f}/d "
              f"red {m1['reduction']:+7.2%} pass={str(m1['passed']):<5} "
              f"| VACUOUS={str(vac):<5} "
              f"| >1% {ms['chg_rate_gt1pct_per_day']:8.4f}/d "
              f"red {red1:+7.2%} "
              f"| turn {ts['turnover_per_day']:8.4f}/d "
              f"| held>0 {ms['mean_n_held_gt0']:.2f}{flag}")

    rows.append({"arm": "R65_WINNER_baseline", "p_a": float("nan"),
                 "window": "W_TRAIN", "universe": "U8",
                 "m1_cand_rate_per_day": base_ms["m1_rate_gt0_per_day"],
                 "chg_rate_gt1pct_per_day": base_ms["chg_rate_gt1pct_per_day"],
                 "chg_rate_gt0p1pct_per_day": base_ms["chg_rate_gt0p1pct_per_day"],
                 "turnover_per_day": base_ts["turnover_per_day"],
                 "mean_n_held_gt0": base_ms["mean_n_held_gt0"],
                 "mean_n_held_gt1pct": base_ms["mean_n_held_gt1pct"],
                 "mean_tenure_gt1pct_days": base_ms["mean_tenure_gt1pct_days"]})
    write_csv(OUT_DIR / "novel_m1.csv", rows)
    sel = [r for r in rows if r.get("selected")][0]
    print(f"  M1 at the selected a={a}: passed={sel['m1_passed']} "
          f"(threshold {M1_MIN_REDUCTION:.0%} reduction), "
          f"VACUOUS={sel['m1_vacuous']}")
    return sel


# --------------------------------------------------- 5b. mechanism probe


def cmd_mech(frames, a):
    """Does the recursion actually carry the position THROUGH a crossing?

    M1 as pre-registered cannot answer this (see the vacuity caveat). This
    measures B-31's channel directly: R-63's aim goes to the all-zero
    vector every time the incumbent's score crosses zero -- that IS the
    forced-exit channel R-65 measured at 0.386/day. The question is what
    this arm holds on exactly those bars. No simulation is run here, so it
    adds nothing to `config_count`.
    """
    print(f"== MECHANISM PROBE (no backtests run), a={a} ==")
    rows = []
    for wname, window, uni in (("W_TRAIN", W_TRAIN, UNIVERSE_8),
                               ("W_FULL6", W_FULL6, UNIVERSE_6)):
        warm = _warm_frames(frames, uni, window)
        idx = _slice_index(warm, window)
        aim = r63_baseline_targets(warm, K).loc[idx].to_numpy(dtype=float)
        x = build_targets(warm, a).loc[idx].to_numpy(dtype=float)
        r65 = r65_winner_targets(warm).loc[idx]
        days = len(idx) / BARS_PER_DAY

        aim_live = aim.sum(axis=1) > 0.0
        # a "forced exit" in R-65's sense: the aim was live and goes flat.
        exits = int((aim_live[:-1] & ~aim_live[1:]).sum())
        flat = ~aim_live
        tot = x.sum(axis=1)
        row = {
            "window": wname, "universe": "U8" if uni is UNIVERSE_8 else "U6",
            "p_a": a, "days": days,
            "aim_flat_frac_bars": float(flat.mean()),
            "aim_forced_exits_per_day": exits / days,
            "aim_forced_exits": exits,
            "smoothed_mean_notional_on_aim_flat_bars":
                float(tot[flat].mean()) if flat.any() else float("nan"),
            "smoothed_frac_of_aim_flat_bars_holding_gt1pct":
                float((tot[flat] > 0.01).mean()) if flat.any() else float("nan"),
            "smoothed_frac_of_aim_flat_bars_holding_gt5pct":
                float((tot[flat] > 0.05).mean()) if flat.any() else float("nan"),
            "aim_mean_notional": float(aim.sum(axis=1).mean()),
            "smoothed_mean_notional": float(tot.mean()),
            "r65_winner_turnover_per_day":
                turnover_stats(r65)["turnover_per_day"],
            "cand_turnover_per_day":
                turnover_stats(build_targets(warm, a).loc[idx])["turnover_per_day"],
        }
        rows.append(row)
        print(f"  {wname}: aim flat on {row['aim_flat_frac_bars']:.1%} of bars; "
              f"R-63's forced exits {row['aim_forced_exits_per_day']:.4f}/day "
              f"({exits} events)")
        print(f"    on those aim-flat bars the smoothed arm holds mean notional "
              f"{row['smoothed_mean_notional_on_aim_flat_bars']:.4f}; "
              f">1%: {row['smoothed_frac_of_aim_flat_bars_holding_gt1pct']:.1%}  "
              f">5%: {row['smoothed_frac_of_aim_flat_bars_holding_gt5pct']:.1%}")
        print(f"    mean notional aim {row['aim_mean_notional']:.4f} vs smoothed "
              f"{row['smoothed_mean_notional']:.4f} | turnover R-65 winner "
              f"{row['r65_winner_turnover_per_day']:.4f}/d vs candidate "
              f"{row['cand_turnover_per_day']:.4f}/d")
    write_csv(OUT_DIR / "novel_mechanism.csv", rows)
    return rows


# ------------------------------------------------------------------ 6. run


def cmd_run(frames, a):
    print(f"== DECISION CELLS at the frozen a={a} ==")
    rows = []

    aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, a)
    print(f"  W_FULL6 bars {len(targets):,}  {targets.index[0]} -> "
          f"{targets.index[-1]}")
    print(f"  first evaluated bar warm for every asset: {warm_ok}")
    if not warm_ok:
        raise RuntimeError("W_FULL6 first evaluated bar not warm")

    # substrate reproduction: R-63's own D1 cell through this branch's path
    _, rt, _ = build_cell(frames, UNIVERSE_6, W_FULL6, 1.0, baseline=True)
    rts = turnover_stats(rt)
    rc = mean_total_notional(rt)
    r_cand = simulate_portfolio(rt, aligned, SPOT_BASE)
    r_mh = simulate_portfolio(matched_hold_targets(rt.index, UNIVERSE_6, rc),
                              aligned, SPOT_BASE)
    r_cmp = compare(r_cand, r_mh)
    print("  [R-63 REFERENCE REPRODUCTION, W_FULL6 U6 vs MATCHED_HOLD]")
    print(f"    turnover {rts['turnover_per_day']:.4f}/d (R-63 published "
          f"{R63_TURNOVER_PER_DAY})   net growth_diff "
          f"{r_cmp['growth_diff']:+.4f} (R-63 published -7.537)   "
          f"cand_final {r_cmp['cand_final']:,.4f} (published 1.4419)  "
          f"mtn {rc:.4f} (published 0.5249)")
    rows.append({"arm": "R63_BASELINE_k1", "window": "W_FULL6", "universe": "U6",
                 "bench": "MATCHED_HOLD", "kind": "substrate reproduction",
                 "turnover_per_day": rts["turnover_per_day"],
                 "hold_days": holding_period_days(rt), "mean_notional": rc,
                 "net_growth_diff": r_cmp["growth_diff"],
                 "net_growth_lo": r_cmp["growth_lo"],
                 "net_growth_hi": r_cmp["growth_hi"],
                 "cand_final": r_cmp["cand_final"],
                 "bench_final": r_cmp["bench_final"],
                 "cand_dd": r_cmp["cand_dd"], "bench_dd": r_cmp["bench_dd"],
                 "n_days": r_cmp["n_days"]})

    d12 = evaluate(targets, aligned, UNIVERSE_6, "W_FULL6", "U6",
                   "smoothed_score", {"a": a})
    d12["kind"] = "D1/D2/D5 primary"
    d12["first_bar_warm"] = warm_ok
    print("  [D1/D2/D5] W_FULL6 U6 vs VOLMATCH_HOLD")
    print(fmt_front(d12))
    print(f"    volmatch matched: net={d12['net_volmatch_matched']} "
          f"(c={d12['net_volmatch_c']:.3f}, bench vol "
          f"{d12['net_volmatch_vol']:.3f} vs cand {d12['net_cand_vol']:.3f}) | "
          f"gross={d12['gross_volmatch_matched']} "
          f"(c={d12['gross_volmatch_c']:.3f})")

    if d12["net_volmatch_matched"]:
        d1, d2 = d1_pass(d12), d2_pass(d12)
    else:
        d1 = d2 = False
        print("    !! VOLMATCH did not match @0.10% -- D1/D2 VOIDED, not scored")
    d5 = d5_pass(d12) if d12["gross_volmatch_matched"] else False
    if not d12["gross_volmatch_matched"]:
        print("    !! VOLMATCH did not match @0bps -- D5 VOIDED, not scored")
    d12["d1_pass"], d12["d2_pass"], d12["d5_pass"] = d1, d2, d5
    rows.append(d12)
    print(f"    D1 PASS={d1}  D2 PASS={d2}  D5 PASS={d5} "
          f"(gross {d12['gross_growth_diff']:+.3f} vs bar {D5_BAR:+.3f})")

    # continuity / context
    cand = simulate_portfolio(targets, aligned, SPOT_BASE)
    c = mean_total_notional(targets)
    mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                            aligned, SPOT_BASE)
    ew = static_hold_equity(aligned, UNIVERSE_6, SPOT_BASE)
    btc = frames["BTC"]
    btc_on = btc.reindex(btc.index.union(targets.index)).ffill().reindex(targets.index)
    btc_eq = static_hold_equity({"BTC": btc_on}, ["BTC"], SPOT_BASE)
    for label, bench in (("MATCHED_HOLD", mh), ("EW_HOLD", ew),
                         ("BTC_HOLD", btc_eq)):
        cm = compare(cand, bench)
        rows.append({"arm": "smoothed_score", "window": "W_FULL6",
                     "universe": "U6", "bench": label, "kind": "context",
                     "p_a": a, "mean_notional": c,
                     "hold_days": holding_period_days(targets),
                     "turnover_per_day": turnover_stats(targets)["turnover_per_day"],
                     "net_growth_diff": cm["growth_diff"],
                     "net_growth_lo": cm["growth_lo"],
                     "net_growth_hi": cm["growth_hi"],
                     "net_dd_diff": cm["dd_diff"], "net_dd_lo": cm["dd_lo"],
                     "net_dd_hi": cm["dd_hi"], "cand_final": cm["cand_final"],
                     "bench_final": cm["bench_final"], "cand_dd": cm["cand_dd"],
                     "bench_dd": cm["bench_dd"], "n_days": cm["n_days"]})
        print(f"  [context vs {label}] cand {cm['cand_final']:,.1f} vs "
              f"{cm['bench_final']:,.1f} | growth {cm['growth_diff']:+.3f} "
              f"[{cm['growth_lo']:+.3f},{cm['growth_hi']:+.3f}] | dd "
              f"{cm['cand_dd']:.1f}% vs {cm['bench_dd']:.1f}% "
              f"({cm['dd_diff']:+.2f})")

    # D4: W_FULL6 @0.40% vs EW_HOLD, final balance
    cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
    ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
    d4 = bool(cand40.iloc[-1] > ew40.iloc[-1])
    rows.append({"arm": "smoothed_score", "window": "W_FULL6", "universe": "U6",
                 "bench": "EW_HOLD", "kind": "D4 @0.40%", "p_a": a,
                 "cand_final": float(cand40.iloc[-1]),
                 "bench_final": float(ew40.iloc[-1]), "d4_pass": d4})
    print(f"  [D4 @0.40%] cand {cand40.iloc[-1]:,.1f} vs EW_HOLD "
          f"{ew40.iloc[-1]:,.1f} -> D4 PASS={d4}")

    # D3: W_VAL, U8
    aligned3, targets3, warm3 = build_cell(frames, UNIVERSE_8, W_VAL, a)
    if not warm3:
        raise RuntimeError("W_VAL first evaluated bar not warm")
    d3row = evaluate(targets3, aligned3, UNIVERSE_8, "W_VAL", "U8",
                     "smoothed_score", {"a": a})
    d3row["kind"] = "D3 inner-validation"
    d3row["first_bar_warm"] = warm3
    print("  [D3] W_VAL U8 vs VOLMATCH_HOLD")
    print(fmt_front(d3row))
    d3 = d3_pass(d3row) if d3row["net_volmatch_matched"] else False
    if not d3row["net_volmatch_matched"]:
        print("    !! VOLMATCH did not match on W_VAL -- D3 VOIDED, not scored")
    d3row["d3_pass"] = d3
    rows.append(d3row)
    print(f"    D3 PASS={d3}  (matched={d3row['net_volmatch_matched']})")

    # MATCHED_HOLD continuity on the D3 cell too
    c3 = mean_total_notional(targets3)
    cand3 = simulate_portfolio(targets3, aligned3, SPOT_BASE)
    mh3 = simulate_portfolio(matched_hold_targets(targets3.index, UNIVERSE_8, c3),
                             aligned3, SPOT_BASE)
    cm3 = compare(cand3, mh3)
    rows.append({"arm": "smoothed_score", "window": "W_VAL", "universe": "U8",
                 "bench": "MATCHED_HOLD", "kind": "context", "p_a": a,
                 "mean_notional": c3, "net_growth_diff": cm3["growth_diff"],
                 "net_growth_lo": cm3["growth_lo"],
                 "net_growth_hi": cm3["growth_hi"],
                 "net_dd_diff": cm3["dd_diff"], "cand_final": cm3["cand_final"],
                 "bench_final": cm3["bench_final"], "cand_dd": cm3["cand_dd"],
                 "bench_dd": cm3["bench_dd"], "n_days": cm3["n_days"]})
    print(f"  [context W_VAL vs MATCHED_HOLD] growth {cm3['growth_diff']:+.3f}"
          f"  dd_diff {cm3['dd_diff']:+.2f}")

    write_csv(OUT_DIR / "novel_cells.csv", rows)
    return {"d1": d1, "d2": d2, "d3": d3, "d5": d5, "d4": d4, "d1_row": d12,
            "targets": targets, "aligned": aligned, "a": a}


# ------------------------------------------------------------- 7. scramble


def cmd_scramble(frames, a, state=None):
    print(f"== FALSIFICATION: scramble controls on the D1 cell, a={a} ==")
    if state is None:
        aligned, targets, _ = build_cell(frames, UNIVERSE_6, W_FULL6, a)
    else:
        aligned, targets = state["aligned"], state["targets"]
    cand = simulate_portfolio(targets, aligned, SPOT_BASE)
    bench, c, vol, matched = volmatched_hold_equity(cand, aligned, UNIVERSE_6,
                                                    SPOT_BASE)
    real = compare(cand, bench)["growth_diff"]
    cand_turn = turnover_stats(targets)["turnover_per_day"]
    print(f"  candidate real growth_diff {real:+.4f}  turnover {cand_turn:.4f}/d"
          f"  (volmatch matched={matched}, c={c:.3f})")

    rows, diffs = [], []
    print("  -- (A) PRE-REGISTERED redraw-per-change scramble (structurally "
          "invalid for a continuous arm; run for the record) --")
    for seed in SCRAMBLE_SEEDS:
        st = scramble_targets(targets, seed)
        eq = simulate_portfolio(st, aligned, SPOT_BASE)
        r = compare(eq, bench)
        diffs.append(r["growth_diff"])
        st_turn = turnover_stats(st)["turnover_per_day"]
        rows.append({"arm": "smoothed_score_scrambled_prereg", "seed": seed,
                     "p_a": a, "window": "W_FULL6", "universe": "U6",
                     "fee": 0.001, "bench": "VOLMATCH_HOLD",
                     "mean_notional": mean_total_notional(st),
                     "turnover_per_day": st_turn,
                     **{k: r[k] for k in ("cand_final", "bench_final", "cand_dd",
                                          "bench_dd", "growth_diff", "growth_lo",
                                          "growth_hi", "dd_diff", "dd_lo",
                                          "dd_hi", "n_days")}})
        print(f"    seed {seed}: growth_diff {r['growth_diff']:+.4f}  "
              f"final {r['cand_final']:>12,.2f}  turnover {st_turn:10.4f}/d")
    p90 = float(np.percentile(diffs, 90))
    surv_pre = bool(real > p90)
    print(f"    real {real:+.4f} vs p90 {p90:+.4f} -> SURVIVED={surv_pre}  "
          f"(candidate turnover {cand_turn:.4f}/d vs control mean "
          f"{np.mean([r['turnover_per_day'] for r in rows]):.4f}/d)")
    rows.append({"arm": "smoothed_score", "seed": -1, "p_a": a,
                 "window": "W_FULL6", "universe": "U6", "fee": 0.001,
                 "bench": "VOLMATCH_HOLD", "growth_diff": real,
                 "scramble_p90": p90, "scramble_survived": surv_pre,
                 "turnover_per_day": cand_turn,
                 "control": "prereg redraw-per-change"})

    print("  -- (B) FIXED-PERMUTATION scramble (L1 isometry, turnover "
          "preserved bar-for-bar; THIS is the one to believe) --")
    fdiffs = []
    cols = list(targets.columns)
    for seed in SCRAMBLE_SEEDS:
        perm = np.random.default_rng(1000 + seed).permutation(len(cols))
        st = pd.DataFrame(targets.to_numpy()[:, perm], index=targets.index,
                          columns=cols)
        eq = simulate_portfolio(st, aligned, SPOT_BASE)
        r = compare(eq, bench)
        fdiffs.append(r["growth_diff"])
        ident = bool(np.array_equal(perm, np.arange(len(cols))))
        st_turn = turnover_stats(st)["turnover_per_day"]
        rows.append({"arm": "smoothed_score_fixedperm", "seed": seed, "p_a": a,
                     "window": "W_FULL6", "universe": "U6", "fee": 0.001,
                     "bench": "VOLMATCH_HOLD", "perm": ";".join(map(str, perm)),
                     "identity_perm": ident,
                     "mean_notional": mean_total_notional(st),
                     "turnover_per_day": st_turn,
                     **{k: r[k] for k in ("cand_final", "bench_final", "cand_dd",
                                          "bench_dd", "growth_diff", "growth_lo",
                                          "growth_hi", "dd_diff", "dd_lo",
                                          "dd_hi", "n_days")}})
        print(f"    seed {seed} perm {perm}: growth_diff {r['growth_diff']:+.4f}"
              f"  turnover {st_turn:.4f}/d{'  (IDENTITY)' if ident else ''}")
    fp90 = float(np.percentile(fdiffs, 90))
    surv_fix = bool(real > fp90)
    n_beat = int(sum(1 for d in fdiffs if real > d))
    print(f"    real {real:+.4f} vs fixed-perm p90 {fp90:+.4f} -> "
          f"SURVIVED={surv_fix}  (beats {n_beat}/10)")
    rows.append({"arm": "smoothed_score", "seed": -2, "p_a": a,
                 "window": "W_FULL6", "universe": "U6", "fee": 0.001,
                 "bench": "VOLMATCH_HOLD", "growth_diff": real,
                 "scramble_p90": fp90, "scramble_survived": surv_fix,
                 "turnover_per_day": cand_turn, "n_beaten": n_beat,
                 "control": "fixed-permutation (believed)"})

    write_csv(OUT_DIR / "novel_scramble.csv", rows)
    return {"prereg": surv_pre, "fixed": surv_fix}


# ------------------------------------------------------- 8. plateau vs peak


def _spearman(x, y):
    def rk(v):
        o = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return o.astype(float)
    a, b = rk(x), rk(y)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def cmd_plateau(rows, a):
    print("== PLATEAU vs PEAK ==")
    tr = {r["p_a"]: r for r in rows if r["window"] == "W_TRAIN"
          and r["kind"] == "grid"}
    va = {r["p_a"]: r for r in rows if r["window"] == "W_VAL"
          and r["kind"] == "grid"}
    order_tr = sorted(tr, key=lambda k: -tr[k]["net_growth_diff"])
    order_va = sorted(va, key=lambda k: -va[k]["net_growth_diff"])
    print("  W_TRAIN ordering (best first): "
          + ", ".join(f"a={k} ({tr[k]['net_growth_diff']:+.3f})" for k in order_tr))
    print("  W_VAL   ordering (best first): "
          + ", ".join(f"a={k} ({va[k]['net_growth_diff']:+.3f})" for k in order_va))
    ks = list(A_GRID)
    rho = _spearman([tr[k]["net_growth_diff"] for k in ks],
                    [va[k]["net_growth_diff"] for k in ks])
    print(f"  Spearman rank correlation of net growth, W_TRAIN vs W_VAL, "
          f"across the {len(ks)} cells: {rho:+.3f}")
    i = ks.index(a)
    nb = [ks[j] for j in (i - 1, i + 1) if 0 <= j < len(ks)]
    print(f"  winner a={a}: W_TRAIN rank {order_tr.index(a)+1}/{len(ks)}, "
          f"W_VAL rank {order_va.index(a)+1}/{len(ks)}")
    for n in nb:
        print(f"    neighbour a={n}: W_VAL {va[n]['net_growth_diff']:+.4f} "
              f"(winner {va[a]['net_growth_diff']:+.4f}) -> "
              f"{'WORSE' if va[n]['net_growth_diff'] < va[a]['net_growth_diff'] else 'BETTER'}"
              f" | W_TRAIN {tr[n]['net_growth_diff']:+.4f} "
              f"(winner {tr[a]['net_growth_diff']:+.4f})")
    return {"spearman_train_val": rho,
            "winner_rank_train": order_tr.index(a) + 1,
            "winner_rank_val": order_va.index(a) + 1}


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["identity", "derive", "frontier", "checks",
                                    "m1", "mech", "run", "scramble", "cells",
                                    "all"])
    ap.add_argument("--a", type=float, default=None)
    args = ap.parse_args()

    frames = load_universe(UNIVERSE_8)
    a = args.a if args.a is not None else FROZEN_A

    if args.cmd == "identity":
        cmd_identity(frames)
    elif args.cmd == "derive":
        cmd_derive(frames)
    elif args.cmd == "frontier":
        rows = cmd_frontier(frames)
        sel = cmd_select(rows)
        cmd_plateau(rows, sel)
    elif args.cmd == "checks":
        cmd_checks(frames, a)
    elif args.cmd == "m1":
        cmd_m1(frames, a)
    elif args.cmd == "mech":
        cmd_mech(frames, a)
    elif args.cmd == "run":
        cmd_run(frames, a)
    elif args.cmd == "scramble":
        cmd_scramble(frames, a)
    elif args.cmd == "cells":
        if a is None:
            raise SystemExit("no frozen a: run `frontier` first or pass --a")
        m1 = cmd_m1(frames, a)
        cmd_mech(frames, a)
        st = cmd_run(frames, a)
        sc = cmd_scramble(frames, a, st)
        fw = further_work(bool(m1["m1_passed"]), st["d1"], st["d2"], st["d3"],
                          st["d5"], sc["fixed"])
        print(f"\n== further_work(m1={m1['m1_passed']}, d1={st['d1']}, "
              f"d2={st['d2']}, d3={st['d3']}, d5={st['d5']}, "
              f"scramble_fixedperm={sc['fixed']}) = {fw} ==")
        print("  -> W_HOLD is NOT read by this branch under any outcome.")
    else:
        cmd_identity(frames)
        cmd_derive(frames)
        rows = cmd_frontier(frames)
        sel = cmd_select(rows)
        cmd_plateau(rows, sel)
        cmd_checks(frames, sel)
        m1 = cmd_m1(frames, sel)
        cmd_mech(frames, sel)
        st = cmd_run(frames, sel)
        sc = cmd_scramble(frames, sel, st)
        fw = further_work(bool(m1["m1_passed"]), st["d1"], st["d2"], st["d3"],
                          st["d5"], sc["fixed"])
        print(f"\n== further_work(m1={m1['m1_passed']}, d1={st['d1']}, "
              f"d2={st['d2']}, d3={st['d3']}, d5={st['d5']}, "
              f"scramble_fixedperm={sc['fixed']}) = {fw} ==")
        print("  -> W_HOLD is NOT read by this branch under any outcome.")

    print(f"\nconfig_count() = {config_count()}")


if __name__ == "__main__":
    main()
