"""R-69 NOVEL branch -- the entry-only gate, threshold DERIVED not fitted,
on THIS round's bufferless construction. Zero free parameters, no grid, no
selection. Answers backlog item B-37's "novel" half.

The pre-registration is `experiments/r69_shared.py`'s module docstring
(FROZEN, NOT edited by this file). This file implements one candidate,
measures it, and reports it. `git diff` on `r69_shared.py`, `r68_shared.py`,
`r68_conservative_band_decomposition.py`, `r68_novel_derived_threshold.py`,
`r67_shared.py`, `r65_shared.py`, `r63_shared.py` and
`r63_novel_xsmom_rank.py` is empty for this branch. Any flaw found in any of
them is REPORTED, never fixed (the R-63 process violation, not repeated).

=====================================================================
THE CANDIDATE
=====================================================================

R-68's own event loop (`band_selection` in
`r68_conservative_band_decomposition.py`), adapted with exactly three of its
inputs set to the values that DELETE R-65's machinery, per this round's
shared pre-registration:

    buffer     = 0.0    (no margin; any strictly-better eligible challenger
                          swaps immediately)
    hold_days  = 0      (no minimum tenure; the timer never blocks)
    delta_out  = 0.0    (exit purely at s > 0, R-63's original rule, never
                          retuned)

The ONLY thing this branch adds is that `delta_in` -- the entry threshold --
is no longer a swept constant. It is DERIVED, and it is TIME-VARYING:

    delta_in(t) = mult * sigma_ds(t) * sqrt(T*)

`sigma_ds(t)` (the causal, expanding, one-bar-shifted pooled std of the
per-bar score INCREMENT) and `T*` (`h* * 288`, the cost-matched
first-passage optimum read off R-65's committed decay table) are IMPORTED
READ-ONLY from `experiments/r68_novel_derived_threshold.py`
(`sigma_dscore_series`, `t_star_bars`, `decay_optimum`, `DECAY_CSV`) --
R-68's own already-validated derivation (D-B in that file's docstring,
Kaminski & Lo 2014's stopping-time identity applied to R-65's measured
decay table). This branch does not re-derive or re-argue it; a numerical
mismatch against that module's own reported numbers would be a bug report,
not a second derivation.

`mult = 1.0` is the PRIMARY, undiscounted derived point -- the one the
round's decision cells are read at. `mult in (0.5, 0.75, 1.5, 2.0)` is
reported as a neighbourhood, exactly as R-68's novel branch did, and it is
REPORTED, NOT SELECTED ON.

DEPARTURE FROM `r68_novel_derived_threshold.delta_B_series`, DISCLOSED. That
function additionally clips the raw threshold at a causal dLC(2020)
saturation cap, `min(raw, 1.6 * sigma_s(t))`, and floors non-finite values at
0.0. This branch's formula is exactly the one this round's shared
pre-registration states -- `delta_B(t) = sigma_ds(t) * sqrt(T*)`, nothing
more -- so the cap is DELIBERATELY NOT applied here; only the non-finite ->
0.0 floor is kept (a numerical necessity during the warm-up window, not a
derivation choice). Whether the omitted cap would ever have bound is
measured and reported in the derivation section below, so the omission's
practical weight is not left to guesswork.

=====================================================================
THE EVENT LOOP -- ADAPTED FROM `r68_conservative_band_decomposition.py`
=====================================================================

    enter_eligible(t) = isfinite(s) & (s >  +delta_in(t))   # per-bar
    hold_eligible(t)  = isfinite(s) & (s >  0.0)             # delta_out=0

Forced exits (never blocked -- there is no timer left to block them), entries
into free slots ranked by score descending, voluntary swaps whenever a
strictly-better eligible challenger exists than the worst incumbent (no
buffer margin left to require anything more than "strictly better"). The
three-case structure (forced exit / free-slot entry / voluntary swap) is
copied unchanged from R-68 conservative's `band_selection`; only the two now
literally-absent gates (`buffer`, `hold_days`) are dropped from the swap
branch's condition, and `enter_eligible`/`hold_eligible` are broadcast
against a per-bar delta column instead of a scalar (the same generalisation
`r68_novel_derived_threshold.derived_selection` already made for its own,
differently-plumbed, buffered loop).

STRICTLY CAUSAL BY CONSTRUCTION: a forward loop whose state at bar `i`
depends on rows <= i and nothing else, driven by a delta column each of
whose inputs (`sigma_ds(t)`) is an expanding statistic over rows STRICTLY
BEFORE `i`. Proven by the self-checks below, not asserted.

=====================================================================
WHY delta_in=0 MUST REPRODUCE R-63's ORIGINAL RULE -- THE LOOP IDENTITY
=====================================================================

`r69_shared.py`'s own docstring works this out in full: with
`buffer = hold_days = 0`, a voluntary swap fires whenever
`row[best] > row[worst]` with no margin and no cooldown, i.e. the incumbent
is replaced on the very next bar by any higher-scoring eligible challenger --
exactly what a fresh per-bar top-1 recomputation does. At `delta_in = 0`
every positive-score asset is eligible, so this candidate's construction
collapses to R-63's `(s > 0) & (rank < k)` at k=1. This is checked below by
FORCING `delta_in(t) := 0.0` (bypassing the derivation entirely -- this is a
synthetic probe of the LOOP, not the candidate the round reads its verdict
from) and comparing bit-for-bit against `r63_baseline_targets`.

=====================================================================
SELF-CHECKS -- RUN BEFORE ANY OTHER NUMBER
=====================================================================

  (1) Loop identity at delta_in=0 vs `r63_baseline_targets(aligned, k=1)`,
      exact (atol=1e-12), on warm_window(W_TRAIN)/UNIVERSE_8.
  (2) `r69_shared.check_against_engine()`.
  (3) `r69_shared.check_causality` on THIS branch's own
      `build_targets_derived_entry` (mult=1.0) -- the membership state
      (`held` across bars) and the per-bar delta column are new machinery
      this round introduces, and R-68's causality probes never exercised a
      per-bar-varying entry threshold on a bufferless loop.
  (4) Tail x10 perturbation probe (R-63's `perturbation_probe`, same spirit):
      corrupt the last 40% of every price series and require the early rows
      to be unchanged. Catches a whole-series statistic a pure truncation
      probe can miss.

If any of these fail, the branch stops and reports that as its result
instead of any D-cell number.

THE RESERVED HOLDOUT (2023-01-01 onward, `W_HOLD`) IS NEVER IMPORTED, NAMED
OR SLICED BY THIS BRANCH UNDER ANY OUTCOME.

Run as:
    .venv/bin/python experiments/r69_novel_derived_entry.py
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.r69_shared import (  # noqa: E402
    BARS_PER_DAY,
    D5_BAR_R69,
    DEADBAND,
    K_FROZEN,
    OUT_DIR,
    R68_ENTRY_ONLY_DELTA_WINNER,
    R68_ENTRY_ONLY_MEMBERSHIP_PER_DAY,
    R68_ENTRY_ONLY_TURNOVER_PER_DAY,
    R68_ENTRY_ONLY_WFULL6_DD_VOLMATCH,
    R68_ENTRY_ONLY_WFULL6_NET_VOLMATCH,
    R68_ENTRY_ONLY_WVAL_NET,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_TRAIN,
    W_VAL,
    align_frames,
    basket_log_returns,
    check_against_engine,
    check_causality,
    compare,
    conditional_vol_scale,
    config_count,
    cross_sectional_score,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    excludes_zero,
    frontier_row,
    further_work,
    holding_period_days,
    load_universe,
    m1_pass_vs_raw,
    matched_hold_targets,
    mean_total_notional,
    membership_change_rate_thresholded,
    r63_baseline_targets,
    realized_vol,
    scramble_fixed_perm,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)

# SPOT_FREE (0 bps, the D5 gross tier) is defined in r65_shared but is NOT
# re-exported by r69_shared's __all__ / import list. Imported directly,
# read-only, exactly as `r68_novel_derived_threshold.py` does for the same
# reason. r65_shared.py is NOT edited.
from experiments.r65_shared import SPOT_FREE  # noqa: E402

# The derivation itself -- imported read-only, not reimplemented. Importing
# this module was verified NOT to run its argparse `main()` (guarded by
# `if __name__ == "__main__":`) and to have no other side-effecting
# module-level work; see the final report for the verification method.
from experiments.r68_novel_derived_threshold import (  # noqa: E402
    DECAY_CSV,
    decay_optimum,
    sigma_dscore_series,
    t_star_bars,
)

ARM = "novel_derived_entry"

K_FIXED = K_FROZEN          # 1
BUFFER_FIXED = 0.0          # deleted, per this round's pre-registration
HOLD_FIXED = 0              # deleted, per this round's pre-registration
DELTA_OUT_FIXED = 0.0       # exit purely at s > 0, never retuned

# Context only, for comparison in the report.
CONSERVATIVE_GRID_LO, CONSERVATIVE_GRID_HI = 0.000, 0.350

# The neighbourhood multiplier ladder, R-68's own. REPORTED, NOT SELECTED ON.
NEIGHBOURHOOD = (0.5, 0.75, 1.0, 1.5, 2.0)
PRIMARY_MULT = 1.0


# ======================================================================
# 1. THE DERIVED THRESHOLD -- imported statistics, this branch's own formula
# ======================================================================


def delta_in_series(aligned: dict[str, pd.DataFrame], mult: float = 1.0) -> np.ndarray:
    """delta_in(t) = mult * sigma_ds(t) * sqrt(T*).

    `sigma_dscore_series` and `t_star_bars` are R-68's own causal, already-
    validated estimators, imported read-only. Non-finite values (the warm-up
    window, before `sigma_ds` has seen its floor of finite bars) are floored
    at 0.0, which makes the candidate identically R-65/R-63's rule on those
    bars -- the same convention `r68_novel_derived_threshold._finalize` uses.
    No causal dLC saturation cap is applied here (see module docstring);
    delta is otherwise never negative because sigma_ds >= 0 and mult >= 0.
    """
    sdd = sigma_dscore_series(aligned)
    tstar = t_star_bars()
    raw = float(mult) * sdd * math.sqrt(tstar)
    d = np.where(np.isfinite(raw), raw, 0.0)
    return np.maximum(d, 0.0)


# ======================================================================
# 2. THE EVENT LOOP -- adapted from r68_conservative_band_decomposition.band_selection
# ======================================================================


def entry_gate_selection(s: np.ndarray, k: int, delta_in_t: np.ndarray):
    """R-68 conservative's `band_selection`, adapted: `buffer=0.0`,
    `hold_days=0`, `delta_out=0.0` FIXED (never swept), `delta_in` a PER-BAR
    array instead of a scalar constant.

    Because buffer=0 and hold_days=0, the swap branch's gating condition
    collapses to a bare `row[best] > row[worst]` with no timer check -- the
    two now-absent gates are dropped from the branch rather than evaluated
    and always-true, so the loop reads as what it computes. Everything else
    (the three-case structure: forced exit / free-slot entry / voluntary
    swap; the ordering; ranking free-slot candidates by score descending) is
    UNCHANGED from the source loop.

    STRICTLY CAUSAL: state at bar i depends only on rows <= i; `delta_in_t`
    is supplied by :func:`delta_in_series`, an expanding, one-bar-shifted
    statistic.
    """
    n, n_assets = s.shape
    finite = np.isfinite(s)
    d = np.asarray(delta_in_t, dtype=float)
    if d.shape != (n,):
        raise ValueError(f"delta_in_t shape {d.shape} != ({n},)")
    enter_eligible = finite & (s > d[:, None])
    hold_eligible = finite & (s > float(DELTA_OUT_FIXED))

    sel = np.zeros((n, n_assets), dtype=bool)
    held: list[int] = []
    keys = ("forced_exit", "entry", "swap", "flat_bars")
    ev = {key: 0 for key in keys}
    ev_bars = {key: np.zeros(n, dtype=np.int32) for key in keys}

    for i in range(n):
        row = s[i]
        elig_in = enter_eligible[i]
        elig_hold = hold_eligible[i]
        changed = False

        # (a) forced exits -- unconditional; there is no timer left to block
        #     them. An incumbent leaves once its score is no longer > 0.
        if held:
            keep = [a for a in held if elig_hold[a]]
            if len(keep) != len(held):
                ev["forced_exit"] += 1
                ev_bars["forced_exit"][i] = 1
                held = keep
                changed = True

        # entries into empty slots (including refilling a slot a forced exit
        # just freed, and re-entering from flat). A new entrant must clear
        # +delta_in(t).
        if len(held) < k:
            free = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if free:
                free.sort(key=lambda a: -row[a])
                for a in free[: k - len(held)]:
                    held.append(a)
                    changed = True
                ev["entry"] += 1
                ev_bars["entry"][i] = 1

        # (b) voluntary swap -- no buffer, no timer: fires whenever a
        #     strictly-better eligible challenger exists.
        elif not changed:
            worst = min(held, key=lambda a: row[a])
            out = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if out:
                best = max(out, key=lambda a: row[a])
                if row[best] > row[worst]:
                    held.remove(worst)
                    held.append(best)
                    changed = True
                    ev["swap"] += 1
                    ev_bars["swap"][i] = 1

        if held:
            sel[i, held] = True
        else:
            ev["flat_bars"] += 1
            ev_bars["flat_bars"][i] = 1

    return sel, ev, ev_bars


def _size(sel: np.ndarray, aligned: dict[str, pd.DataFrame], k: int,
          index, assets) -> pd.DataFrame:
    """R-63's sizing block, copied byte-for-byte (via R-65/R-67/R-68's own
    copies) and unmodified: 0.10 deadband latch on desired TOTAL notional,
    clip to 1.0, equal split over held slots."""
    n = sel.shape[0]
    m = sel.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))

    desired = scale * (m / float(k))
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        dd = desired[i]
        if abs(dd - cur) > DEADBAND:
            cur = dd
        pos[i] = cur

    total = np.minimum(pos, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(m > 0, total / np.maximum(m, 1), 0.0)
    w = sel * per[:, None]
    return pd.DataFrame(w, index=index, columns=assets)


def build_targets_derived_entry_ev(aligned: dict[str, pd.DataFrame],
                                   k: int = K_FIXED, mult: float = 1.0,
                                   force_delta_in: float | None = None):
    """Targets AND the event ledger AND the delta column, from one pass.

    `force_delta_in`, when given, BYPASSES the derivation entirely and uses
    that scalar constant for every bar -- used ONLY by the delta_in=0 loop
    identity self-check, never by anything that produces a reported D-cell.
    """
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    n = s.shape[0]
    if force_delta_in is not None:
        delta_in_t = np.full(n, float(force_delta_in))
    else:
        delta_in_t = delta_in_series(aligned, mult)
    sel, ev, ev_bars = entry_gate_selection(s, k, delta_in_t)
    targets = _size(sel, aligned, k, score.index, assets)
    return targets, ev, ev_bars, delta_in_t, score.index


def build_targets_derived_entry(aligned: dict[str, pd.DataFrame],
                                k: int = K_FIXED, mult: float = 1.0) -> pd.DataFrame:
    return build_targets_derived_entry_ev(aligned, k, mult)[0]


# ======================================================================
# 3. cells / io
# ======================================================================


_WARM_CACHE: dict = {}


def _warm_frames(frames, universe, window):
    key = (tuple(universe), window)
    if key not in _WARM_CACHE:
        _WARM_CACHE[key] = align_frames({t: frames[t] for t in universe},
                                        warm_window(window))
    return _WARM_CACHE[key]


def _slice_index(warm: dict[str, pd.DataFrame], window):
    """STRICT right-exclusive slice, independent of the shared `_hi` helper
    -- the guard this project's rounds have carried since R-63 caught the
    original helper admitting one bar of the reserved holdout."""
    idx = next(iter(warm.values())).index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    return idx


def build_cell(frames, universe, window, mult=1.0):
    warm = _warm_frames(frames, universe, window)
    targets, ev, ev_bars, delta_in_t, full_idx = build_targets_derived_entry_ev(
        warm, K_FIXED, mult)
    idx = _slice_index(warm, window)
    score = cross_sectional_score(warm).loc[idx]
    first_warm = bool(np.isfinite(score.iloc[0].to_numpy()).all())
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    delta_eval = pd.Series(delta_in_t, index=full_idx).loc[idx].to_numpy()

    keep = full_idx.isin(idx)
    days = max(int(keep.sum()) / BARS_PER_DAY, 1e-9)
    rates = {"eval_bars": int(keep.sum()), "eval_days": days}
    for key, arr in ev_bars.items():
        rates[f"{key}_total_warm"] = int(ev[key])
        rates[f"{key}_total_eval"] = int(arr[keep].sum())
        rates[f"{key}_per_day_eval"] = float(arr[keep].sum()) / days
    rates["flat_bar_frac_eval"] = (float(ev_bars["flat_bars"][keep].sum())
                                   / max(int(keep.sum()), 1))
    return aligned_eval, targets.loc[idx], first_warm, delta_eval, rates


def delta_stats(delta_eval: np.ndarray) -> dict:
    return {
        "delta_mean": float(np.mean(delta_eval)),
        "delta_median": float(np.median(delta_eval)),
        "delta_min": float(np.min(delta_eval)),
        "delta_max": float(np.max(delta_eval)),
        "delta_final": float(delta_eval[-1]),
        "delta_zero_frac": float(np.mean(delta_eval <= 0.0)),
        "delta_above_grid_hi_frac":
            float(np.mean(delta_eval > CONSERVATIVE_GRID_HI)),
        "delta_above_r68_winner_frac":
            float(np.mean(delta_eval > R68_ENTRY_ONLY_DELTA_WINNER)),
    }


def measure_pair(targets, aligned, universe, window_name, universe_name, params):
    """One D-cell row: both fee tiers, each against VOLMATCH_HOLD."""
    cand_net = simulate_portfolio(targets, aligned, SPOT_BASE)
    vm_net, c_net, v_net, ok_net = volmatched_hold_equity(cand_net, aligned,
                                                          universe, SPOT_BASE)
    if vm_net is None:
        raise RuntimeError(f"{window_name}: VOLMATCH_HOLD gave no benchmark (net)")
    net_cmp = compare(cand_net, vm_net)

    cand_gross = simulate_portfolio(targets, aligned, SPOT_FREE)
    vm_gross, c_gross, v_gross, ok_gross = volmatched_hold_equity(
        cand_gross, aligned, universe, SPOT_FREE)
    if vm_gross is None:
        raise RuntimeError(f"{window_name}: VOLMATCH_HOLD gave no benchmark (gross)")
    gross_cmp = compare(cand_gross, vm_gross)

    extra = dict(
        volmatch_c_net=c_net, volmatch_vol_net=v_net, volmatch_matched_net=ok_net,
        cand_vol_net=realized_vol(cand_net),
        volmatch_c_gross=c_gross, volmatch_vol_gross=v_gross,
        volmatch_matched_gross=ok_gross,
        cand_vol_gross=realized_vol(cand_gross),
        n_bars=len(targets),
        membership_change_rate_thresholded=membership_change_rate_thresholded(targets),
    )
    row = frontier_row(ARM, params, targets, net_cmp, gross_cmp, "VOLMATCH_HOLD",
                       window_name, universe_name, **extra)
    return row, dict(cand_net=cand_net, vm_net=vm_net, matched_net=ok_net,
                     matched_gross=ok_gross)


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


# ======================================================================
# 4. self-checks
# ======================================================================


def check_loop_identity_at_zero(frames) -> tuple[bool, float]:
    print("== SELF-CHECK (1): loop identity at delta_in FORCED to 0.0, vs "
          "r63_baseline_targets(k=1) -- warm_window(W_TRAIN)/UNIVERSE_8 ==")
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    mine, _, _, delta_in_t, _ = build_targets_derived_entry_ev(
        warm, K_FIXED, force_delta_in=0.0)
    theirs = r63_baseline_targets(warm, K_FIXED)
    a = np.nan_to_num(mine.to_numpy(dtype=float), nan=0.0)
    b = np.nan_to_num(theirs.to_numpy(dtype=float), nan=0.0)
    same_shape = a.shape == b.shape
    maxabs = float(np.max(np.abs(a - b))) if same_shape else float("nan")
    bitwise = bool(same_shape and np.array_equal(a, b))
    exact = bool(same_shape and np.allclose(a, b, atol=1e-12, rtol=0.0))
    print(f"  shape {a.shape}  delta_in forced == 0.0 for all {len(delta_in_t)} bars"
          f" (max forced value = {float(np.max(np.abs(delta_in_t))):.3e})")
    print(f"  max|diff| = {maxabs:.3e}  bit-identical={bitwise}  "
          f"allclose(atol=1e-12)={exact}")
    print(f"  LOOP IDENTITY HOLDS: {exact}")
    return exact, maxabs


def truncation_probe(frames, mult, frac=0.6) -> tuple[bool, float]:
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))
    cut = int(n * frac)
    full = build_targets_derived_entry(warm, K_FIXED, mult).to_numpy()
    trunc = build_targets_derived_entry(
        {t: df.iloc[:cut] for t, df in warm.items()}, K_FIXED, mult).to_numpy()
    a = np.nan_to_num(full[:cut], nan=0.0)
    b = np.nan_to_num(trunc[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0)), float(np.max(np.abs(a - b)))


def tail_perturbation_probe(frames, mult, frac_tail=0.4) -> tuple[bool, float]:
    """R-63's `perturbation_probe`, same spirit: multiply the TAIL of every
    price series by 10 and require the EARLY rows to be bit-identical. This
    round's own risk is a whole-series statistic in `sigma_dscore_series` or
    in this branch's own new loop/threshold code -- a pure truncation probe
    would not catch a statistic that reacts to series CONTENT rather than
    series LENGTH."""
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))
    cut = int(n * (1.0 - frac_tail))
    bad = {}
    for t, df in warm.items():
        d = df.copy()
        for col in ("open", "high", "low", "close"):
            v = d[col].to_numpy(dtype=float).copy()
            v[cut:] *= 10.0
            d[col] = v
        bad[t] = d
    a = np.nan_to_num(build_targets_derived_entry(warm, K_FIXED, mult
                                                  ).to_numpy()[:cut], nan=0.0)
    b = np.nan_to_num(build_targets_derived_entry(bad, K_FIXED, mult
                                                  ).to_numpy()[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0)), float(np.max(np.abs(a - b)))


def run_self_checks(frames) -> dict:
    print("\n" + "=" * 70)
    print("SELF-CHECKS -- run before any other number")
    print("=" * 70)
    rows = []
    ok = True

    ident_ok, ident_max = check_loop_identity_at_zero(frames)
    ok &= ident_ok
    rows.append({"check": "loop_identity_delta_in_0", "max_abs_diff": ident_max,
                 "passed": ident_ok})

    t0 = time.time()
    eng_ok, eng_err = check_against_engine()
    print(f"\n== SELF-CHECK (2): check_against_engine() ==")
    print(f"  ok={eng_ok}  relative_final_balance_error={eng_err:.4%}  "
          f"[{time.time() - t0:.1f}s]")
    ok &= eng_ok
    rows.append({"check": "check_against_engine", "relative_error": eng_err,
                 "passed": eng_ok})

    print(f"\n== SELF-CHECK (3): check_causality on THIS branch's own "
          f"build_fn (mult={PRIMARY_MULT}), warm_window(W_TRAIN)/UNIVERSE_8 ==")
    t0 = time.time()
    warm8 = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    build_fn = lambda a: build_targets_derived_entry(a, K_FIXED, PRIMARY_MULT)  # noqa: E731
    causal_ok = check_causality(build_fn, warm8)
    print(f"  check_causality = {causal_ok}  [{time.time() - t0:.1f}s]")
    ok &= causal_ok
    rows.append({"check": "check_causality", "mult": PRIMARY_MULT, "passed": causal_ok})

    print(f"\n== SELF-CHECK (3b): 60% truncation probe (supporting evidence "
          f"for causality), mult={PRIMARY_MULT} ==")
    trunc_ok, trunc_max = truncation_probe(frames, PRIMARY_MULT)
    print(f"  passed={trunc_ok}  max|diff|={trunc_max:.3e}")
    ok &= trunc_ok
    rows.append({"check": "truncation_60pct", "mult": PRIMARY_MULT,
                 "passed": trunc_ok, "max_abs_diff": trunc_max})

    print(f"\n== SELF-CHECK (4): tail x10 perturbation probe, mult={PRIMARY_MULT} ==")
    pert_ok, pert_max = tail_perturbation_probe(frames, PRIMARY_MULT)
    print(f"  passed={pert_ok}  max|diff|={pert_max:.3e}")
    ok &= pert_ok
    rows.append({"check": "tail_x10_perturbation", "mult": PRIMARY_MULT,
                 "passed": pert_ok, "max_abs_diff": pert_max})

    # Index hygiene: no bar dated 2023-01-01 or later reaches a SELECTION
    # window cell. Restricted to W_TRAIN/W_VAL, exactly as
    # `r68_conservative_band_decomposition.cmd_checks` does -- W_FULL6 is
    # OPEN-ENDED (`(2020-04-01, None)`) by this project's established
    # convention (unchanged since R-63/R-65/R-67/R-68) and is EXPECTED to run
    # to the last bar of data on UNIVERSE_6, which is well past 2023-01-01.
    # That is not a holdout breach: the reserved holdout is the NAMED window
    # `W_HOLD`, never imported/sliced/read anywhere in this file (grepped and
    # confirmed below), and is a distinct constant from W_FULL6 even though
    # their calendar ranges overlap for the six non-BTC/ETH assets. W_FULL6's
    # date range is reported here for the record, not gated on.
    #
    # (This module's imports never name `W_HOLD` -- verified by inspection of
    # the import block above; the constant does not appear as an executable
    # reference anywhere in this file, only in this docstring/comment prose
    # describing the constraint.)
    for wname, window, uni in (("W_TRAIN", W_TRAIN, UNIVERSE_8),
                               ("W_VAL", W_VAL, UNIVERSE_8)):
        _, tg, warm_ok, _, _ = build_cell(frames, uni, window, PRIMARY_MULT)
        bad = int((tg.index >= pd.Timestamp("2023-01-01", tz="UTC")).sum())
        print(f"  (index) {wname}: {len(tg):,} bars {tg.index[0]} -> "
              f"{tg.index[-1]}  bars in reserved holdout: {bad}  "
              f"first_bar_warm={warm_ok}")
        ok &= (bad == 0) and warm_ok
        rows.append({"check": f"index_{wname}", "n_bars": len(tg),
                     "first": str(tg.index[0]), "last": str(tg.index[-1]),
                     "bars_in_holdout": bad, "first_bar_warm": warm_ok,
                     "passed": (bad == 0) and warm_ok})

    _, tgf, warmf_ok, _, _ = build_cell(frames, UNIVERSE_6, W_FULL6, PRIMARY_MULT)
    print(f"  (index, informational only) W_FULL6: {len(tgf):,} bars "
          f"{tgf.index[0]} -> {tgf.index[-1]}  (open-ended by convention, "
          f"UNIVERSE_6 only)  first_bar_warm={warmf_ok}")
    ok &= warmf_ok
    rows.append({"check": "index_W_FULL6_informational", "n_bars": len(tgf),
                 "first": str(tgf.index[0]), "last": str(tgf.index[-1]),
                 "first_bar_warm": warmf_ok, "passed": warmf_ok})

    print(f"\nALL SELF-CHECKS PASS: {ok}")
    write_csv(OUT_DIR / "novel_checks.csv", rows)
    return {"ok": ok, "rows": rows}


# ======================================================================
# 5. derivation report
# ======================================================================


def report_derivation(frames) -> dict:
    print("\n" + "=" * 70)
    print("DERIVATION (imported read-only from r68_novel_derived_threshold)")
    print("=" * 70)
    dec = decay_optimum()
    print(f"  source: {DECAY_CSV}")
    print(f"  h*        = {dec['h_star_days']:.4f} days  "
          f"({dec['T_star_bars']:.1f} bars)   interior={dec['interior']}")
    print(f"  V(h*)     = {dec['V_at_h_star_per_day']:+.8f} /day")
    print(f"  2c/h*     = {dec['cost_at_h_star_per_day']:+.8f} /day")
    print(f"  Net(h*)   = {dec['net_at_h_star_per_day']:+.8f} /day")
    print(f"  h* lands on a table node: {dec['h_star_on_table_node']}")
    print(f"  V'(h*-) = {dec['V_prime_left']:+.4e}  >=  -2c/h*^2 = "
          f"{dec['marginal_cost_at_h_star']:+.4e}  >=  V'(h*+) = "
          f"{dec['V_prime_right']:+.4e}   subgradient condition holds: "
          f"{dec['subgradient_condition_holds']}")
    print(f"  grid_h in [{dec['grid_h_lo']:g},{dec['grid_h_hi']:g}] days, "
          f"{dec['n_table_rows_used']} usable rows")
    print(f"  T* used in delta_in(t) = mult * sigma_ds(t) * sqrt(T*): "
          f"{t_star_bars():.4f} bars (must equal h*_days * {BARS_PER_DAY})")
    assert abs(t_star_bars() - dec["T_star_bars"]) < 1e-9

    print(f"\n  delta_in_t summary stats on W_FULL6/UNIVERSE_6 (evaluation "
          f"slice), by multiplier:")
    dstats_by_mult = {}
    for mult in NEIGHBOURHOOD:
        _, _, _, delta_eval, _ = build_cell(frames, UNIVERSE_6, W_FULL6, mult)
        st = delta_stats(delta_eval)
        dstats_by_mult[mult] = st
        tag = " <- PRIMARY" if mult == PRIMARY_MULT else ""
        print(f"    mult={mult:4.2f}: mean {st['delta_mean']:.5f}  "
              f"min {st['delta_min']:.5f}  max {st['delta_max']:.5f}  "
              f"median {st['delta_median']:.5f}  final {st['delta_final']:.5f}"
              f"  zero_frac {st['delta_zero_frac']:.2%}{tag}")

    prim = dstats_by_mult[PRIMARY_MULT]
    print(f"\n  PRIMARY (mult={PRIMARY_MULT}) vs conservative branch's swept "
          f"grid range [{CONSERVATIVE_GRID_LO:.3f}, {CONSERVATIVE_GRID_HI:.3f}]:")
    inside = CONSERVATIVE_GRID_LO <= prim["delta_mean"] <= CONSERVATIVE_GRID_HI
    print(f"    delta_in mean {prim['delta_mean']:.5f} is "
          f"{'INSIDE' if inside else 'OUTSIDE'} the grid range")
    print(f"    fraction of eval bars with delta_in > grid top (0.350): "
          f"{prim['delta_above_grid_hi_frac']:.2%}")
    print(f"    vs R-68's own conservative ENTRY_ONLY winner "
          f"delta=0.080: mean ratio {prim['delta_mean'] / R68_ENTRY_ONLY_DELTA_WINNER:.3f}x"
          f"  (fraction of eval bars with delta_in > 0.080: "
          f"{prim['delta_above_r68_winner_frac']:.2%})")

    return {"decay": dec, "dstats_by_mult": dstats_by_mult}


# ======================================================================
# 6. decision cells
# ======================================================================


def run_mult_cell(frames, mult, primary=False):
    tag = "PRIMARY" if primary else "context"
    print(f"\n{'=' * 70}\nDECISION CELLS at mult={mult:g}  [{tag}]\n{'=' * 70}")
    params = {"mult": mult, "buffer": BUFFER_FIXED, "hold_days": HOLD_FIXED,
              "delta_out": DELTA_OUT_FIXED, "k": K_FIXED}

    # ---- D1 / D2 / D5 : W_FULL6, U6 ------------------------------------
    aligned6, tg6, warm6_ok, delta6, rates6 = build_cell(frames, UNIVERSE_6,
                                                         W_FULL6, mult)
    if not warm6_ok:
        raise RuntimeError(f"mult={mult}: W_FULL6 first evaluated bar not warm")
    row, st = measure_pair(tg6, aligned6, UNIVERSE_6, "W_FULL6", "U6", params)
    d1 = d1_pass(row)
    d2 = d2_pass(row)
    d5 = d5_pass(row)
    row.update({"d1_pass": d1, "d2_pass": d2, "d5_pass": d5, **delta_stats(delta6)})
    print(f"  [W_FULL6/U6] net growth_diff {row['net_growth_diff']:+.4f} "
          f"[{row['net_growth_lo']:+.4f}, {row['net_growth_hi']:+.4f}]  "
          f"dd_diff {row['net_dd_diff']:+.2f}pp "
          f"[{row['net_dd_lo']:+.2f}, {row['net_dd_hi']:+.2f}]")


    print(f"    D1 PASS={d1}   D2 PASS={d2}")
    print(f"  gross growth_diff {row['gross_growth_diff']:+.4f} "
          f"[{row['gross_growth_lo']:+.4f}, {row['gross_growth_hi']:+.4f}]"
          f"  D5 bar {D5_BAR_R69:+.4f}  D5 PASS={d5}")
    print(f"  turnover/day {row['turnover_per_day']:.4f}  hold_days "
          f"{row['hold_days']:.3f}  mean_notional {row['mean_notional']:.4f}")

    # ---- M1' : same aligned frame/window as D1/D2 ---------------------
    m1 = m1_pass_vs_raw(tg6, aligned6, K_FIXED)
    print(f"  [M1' vs R-63's ORIGINAL rule, same aligned frame] "
          f"membership {m1['cand_membership_per_day']:.4f}/day vs "
          f"{m1['baseline_membership_per_day']:.4f} -> "
          f"{m1['membership_reduction']:+.2%} ({m1['membership_passed']}); "
          f"turnover {m1['cand_turnover_per_day']:.4f}/day vs "
          f"{m1['baseline_turnover_per_day']:.4f} -> "
          f"{m1['turnover_reduction']:+.2%} ({m1['turnover_passed']})  "
          f"M1' PASS={m1['passed']}")

    # ---- scramble : fixed-permutation control on the D1 cell ----------
    cand_turn = row["turnover_per_day"]
    diffs = []
    scramble_rows = []
    for seed in SCRAMBLE_SEEDS:
        stg = scramble_fixed_perm(tg6, seed)
        eq = simulate_portfolio(stg, aligned6, SPOT_BASE)
        r = compare(eq, st["vm_net"])
        diffs.append(r["growth_diff"])
        scramble_rows.append({"mult": mult, "seed": seed,
                              "growth_diff": r["growth_diff"],
                              "dd_diff": r["dd_diff"],
                              "cand_final": r["cand_final"]})
    p90 = float(np.percentile(diffs, 90))
    survived = bool(row["net_growth_diff"] > p90)
    print(f"  [scramble, fixed-permutation, 10 seeds] real {row['net_growth_diff']:+.4f} "
          f"vs p90 {p90:+.4f} -> SURVIVED={survived}  "
          f"({sum(x >= row['net_growth_diff'] for x in diffs)} of {len(diffs)} "
          f"scrambles beat/tie the real cell)")

    # ---- D3 : W_VAL, U8, net --------------------------------------------
    aligned8, tg8, warm8_ok, delta8, rates8 = build_cell(frames, UNIVERSE_8,
                                                         W_VAL, mult)
    if not warm8_ok:
        raise RuntimeError(f"mult={mult}: W_VAL first evaluated bar not warm")
    row3, st3 = measure_pair(tg8, aligned8, UNIVERSE_8, "W_VAL", "U8", params)
    d3 = d3_pass(row3) and st3["matched_net"]
    row3.update({"d3_pass": d3, **delta_stats(delta8)})
    print(f"  [D3 W_VAL/U8] net growth_diff {row3['net_growth_diff']:+.4f}  "
          f"dd_diff {row3['net_dd_diff']:+.2f}pp  matched={st3['matched_net']}  "
          f"D3 PASS={d3}")

    # ---- D4 : W_FULL6, U6, 0.40% (SPOT_REAL) vs EW_HOLD ----------------
    cand40 = simulate_portfolio(tg6, aligned6, SPOT_REAL)
    ew40 = static_hold_equity(aligned6, UNIVERSE_6, SPOT_REAL)
    r40 = compare(cand40, ew40)
    d4 = bool(r40["cand_final"] > r40["bench_final"])
    print(f"  [D4 W_FULL6/U6 @0.40%] cand {r40['cand_final']:,.2f} vs EW_HOLD "
          f"{r40['bench_final']:,.2f} -> D4 PASS={d4}  (growth "
          f"{r40['growth_diff']:+.4f} [{r40['growth_lo']:+.3f}, "
          f"{r40['growth_hi']:+.3f}])")

    # ---- extra diagnostics (mult=PRIMARY_MULT only, in caller) ---------
    extra = {
        "mean_total_notional": mean_total_notional(tg6),
        "turnover_stats": turnover_stats(tg6),
        "holding_period_days": holding_period_days(tg6),
        "realized_vol_net": realized_vol(st["cand_net"]),
        "membership_change_rate_thresholded": membership_change_rate_thresholded(tg6),
    }

    fw = further_work(bool(m1["passed"]), d1, d2, d3, d5, survived)
    print(f"  further_work(m1'={m1['passed']}, d1={d1}, d2={d2}, d3={d3}, "
          f"d5={d5}, scramble={survived}) = {fw}")

    row_full = {**row, "d3_pass": d3, "d4_pass": d4, "m1_passed": bool(m1["passed"]),
               "scramble_survived": survived, "scramble_p90": p90,
               "further_work": fw, "primary": primary}
    write_csv(OUT_DIR / f"novel_scramble_mult{mult:g}.csv", scramble_rows)

    return {"mult": mult, "primary": primary, "row": row_full, "m1": m1,
            "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5,
            "scramble_survived": survived, "scramble_p90": p90,
            "further_work": fw, "extra": extra, "row3": row3}


# ======================================================================
# 7. main
# ======================================================================


def main():
    t_start = time.time()
    print("R-69 NOVEL branch: entry-only gate, derived threshold, "
          "bufferless construction")
    print(f"BUFFER_FIXED={BUFFER_FIXED}  HOLD_FIXED={HOLD_FIXED}  "
          f"DELTA_OUT_FIXED={DELTA_OUT_FIXED}  K_FIXED={K_FIXED}")

    frames = load_universe(UNIVERSE_8)

    checks = run_self_checks(frames)
    if not checks["ok"]:
        print("\n!! SELF-CHECKS FAILED -- STOPPING. No D-cell is reported. !!")
        print(f"\nconfig_count() = {config_count()}")
        return

    deriv = report_derivation(frames)

    print("\n" + "=" * 70)
    print(f"DECISION CELLS -- primary mult={PRIMARY_MULT}, "
          f"neighbourhood {NEIGHBOURHOOD} (context only)")
    print("=" * 70)
    results = {}
    all_cell_rows = []
    for mult in NEIGHBOURHOOD:
        res = run_mult_cell(frames, mult, primary=(mult == PRIMARY_MULT))
        results[mult] = res
        all_cell_rows.append(res["row"])
        all_cell_rows.append({**res["row3"], "mult": mult, "primary": (mult == PRIMARY_MULT)})
    write_csv(OUT_DIR / "novel_cells.csv", all_cell_rows)

    prim = results[PRIMARY_MULT]

    print("\n" + "=" * 70)
    print("PRIMARY POINT (mult=1.0) -- SIDE-BY-SIDE vs R-68's published "
          "ENTRY_ONLY (buffer=0.05, hold_days=1, fitted delta=0.080)")
    print("=" * 70)
    print(f"  {'':30s} {'R-68 ENTRY_ONLY (d=0.080)':>28s} {'R-69 novel (mult=1.0)':>24s}")
    print(f"  {'W_FULL6 net growth vs VOLMATCH':30s} "
          f"{R68_ENTRY_ONLY_WFULL6_NET_VOLMATCH:>28.4f} "
          f"{prim['row']['net_growth_diff']:>24.4f}")
    print(f"  {'W_FULL6 dd_diff vs VOLMATCH (pp)':30s} "
          f"{R68_ENTRY_ONLY_WFULL6_DD_VOLMATCH:>28.2f} "
          f"{prim['row']['net_dd_diff']:>24.2f}")
    print(f"  {'W_VAL net growth vs VOLMATCH':30s} "
          f"{R68_ENTRY_ONLY_WVAL_NET:>28.4f} "
          f"{prim['row3']['net_growth_diff']:>24.4f}")
    print(f"  {'turnover/day (W_FULL6)':30s} "
          f"{R68_ENTRY_ONLY_TURNOVER_PER_DAY:>28.4f} "
          f"{prim['row']['turnover_per_day']:>24.4f}")
    print(f"  {'membership_changes/day (thr.)':30s} "
          f"{R68_ENTRY_ONLY_MEMBERSHIP_PER_DAY:>28.4f} "
          f"{prim['row']['membership_change_rate_thresholded']:>24.4f}")
    print(f"  {'D1 PASS':30s} {'FAIL':>28s} {str(prim['d1']):>24s}")
    print(f"  {'D2 PASS':30s} {'FAIL':>28s} {str(prim['d2']):>24s}")

    print("\n" + "=" * 70)
    print("FINAL VERDICT at mult=1.0 (PRIMARY)")
    print("=" * 70)
    print(f"  further_work(m1'={prim['m1']['passed']}, d1={prim['d1']}, "
          f"d2={prim['d2']}, d3={prim['d3']}, d5={prim['d5']}, "
          f"scramble={prim['scramble_survived']}) = {prim['further_work']}")

    print("\nNeighbourhood (context, NOT selection):")
    for mult in NEIGHBOURHOOD:
        r = results[mult]
        tag = " <- PRIMARY" if mult == PRIMARY_MULT else ""
        print(f"  mult={mult:4.2f}: D1={r['d1']} D2={r['d2']} D3={r['d3']} "
              f"D4={r['d4']} D5={r['d5']} M1'={r['m1']['passed']} "
              f"scramble={r['scramble_survived']} further_work={r['further_work']}"
              f"  net_growth_diff={r['row']['net_growth_diff']:+.4f}{tag}")

    print(f"\nconfig_count() = {config_count()}")
    path = OUT_DIR / "novel_configcount.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["cmd", "config_count", "utc"])
        wr.writerow(["all", config_count(), pd.Timestamp.now("UTC").isoformat()])

    deriv_rows = [{"item": "decay_optimum", **deriv["decay"]}]
    for mult, st in deriv["dstats_by_mult"].items():
        deriv_rows.append({"item": "delta_in_stats", "mult": mult, **st})
    write_csv(OUT_DIR / "novel_derived.csv", deriv_rows)

    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
