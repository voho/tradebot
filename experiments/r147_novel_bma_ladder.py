#!/usr/bin/env python
"""R-147 NOVEL branch: ``BmaLadderKellyV4`` -- replace ``kelly_regime_v4``'s
fixed, unweighted 1/3-each anchor average with a genuinely sequential
Bayesian posterior (Beta-Bernoulli conjugate updating) over a 5-member
alternative-ladder ensemble -- Bayesian Model Averaging in spirit.

EXACT CONSTRUCTION (frozen in ``experiments/r147_shared.py`` before this file
was written; not re-derived here). For each ladder base ``b`` in
``LADDER_BASES = (10, 15, 20, 25, 30)`` (``LADDERS[b] = (b, 2b, 4b)``, so
``LADDERS[20] == (20, 40, 80) == V4_HORIZONS`` exactly -- the shipped ladder
is one ordinary member of this family, not a special case):

  1. ``ladder_frac_b = vote_frac(df, LADDERS[b], V4_BAND)`` -- that ladder's
     OWN unmodified 3-anchor equal-weight vote (v4's own construction,
     verbatim, just at a different base horizon). Takes values in
     ``{0, 1/3, 2/3, 1}``.
  2. ``state_b = latched_state(ladder_frac_b)`` -- binarized majority state.
  3. ``hit_b = spell_hit_series(state_b, df["close"])`` -- causal spell-level
     hit labelling of that ladder's OWN track record.
  4. ``post_b = beta_bernoulli_posterior_mean(hit_b, a0, b0)`` -- fully
     sequential (expanding, not rolling) Beta-Bernoulli posterior mean
     reliability, primary prior ``(a0, b0) = BETA_PRIOR_PRIMARY = (2.0, 2.0)``.

Stack the 5 ladders' ``post_b`` into a ``(n_bars, 5)`` array, row-normalize
(``normalize_weights``, NaN/degenerate-safe, equal-weight fallback by
construction), and blend the 5 ladders' OWN votes (never their internal
anchors individually) by those weights:
``frac_novel = sum_b(weights[:, b] * ladder_frac_b)``. Finally
``target_novel = build_target_from_frac(frac_novel, df)`` -- v4's own
UNCHANGED ``scale``/deadband machinery, untouched.

Full citation trail, the "not a duplicate of" argument against every prior
round (R-40, R-87 x2, R-104, R-105 x2, R-114 x2, R-129/R-130, R-146, every
SIZE-axis round, and this round's own CONSERVATIVE (James-Stein) sibling),
and the three named failure modes are all in ``r147_shared.py``'s own module
docstring (read in full before this file was written); not re-derived here.
This file never edits, and never reads a bar at or after
``r147_shared.OOS_START`` from, ``r147_shared.py`` or any other file.

=====================================================================
PRE-REGISTRATION (frozen before any real-data bind_frac, R^2, Sharpe, or
Monte-Carlo number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

STEP-0 KILL SWITCH (non-degeneracy, gating, applied to the PRIMARY prior
``(2.0, 2.0)`` only -- there is no grid SELECTION step in this branch, unlike
R-105/R-146's floor/threshold grids, because the 5-ladder family and the
primary prior are both fixed a priori by ``r147_shared.py``; the
``BETA_PRIOR_GRID``'s other two entries are a POST-HOC robustness/plateau
check, never a selection search): on BTC inner-train,
  (a) ``bind_frac(weights) > BIND_FRAC_THRESH`` (1%);
  (b) ``r_squared(frac_novel, v4_vote_frac(df)) < R2_DEGENERACY_THRESH`` (0.98).
STOP (report NEGATIVE immediately, no B1-B6) if either fails -- this file
still runs and reports every later check regardless (this project's
convention: every branch reports, including the dead ones), but does not
consult the holdout if the gate fails at any later clause either.

PROMOTION BAR (identical in shape to every SIZE/ERR-axis round since R-89,
restated in full in ``r147_shared.py``'s own module docstring): promote-
recommendation only if ALL hold --
  (1) primary-cell (a0,b0=(2,2)) BTC dSharpe vs kelly_regime_v4, both
      markets, both inner-train and inner-validation, exceeds +/-0.2
      (SHARPE_NOISE_FLOOR) with the bootstrap CI excluding zero on the
      winning side, OR a risk-matched drawdown/tail improvement of
      equivalent rigor (exposure_ratio/vol_ratio/risk_matched from
      compare()'s own output -- R-33's rule: unmatched-risk is not
      evidence);
  (2) the Monte Carlo window check (this round's own pre-registered
      falsification test, chosen because a fast-adapting sequential
      posterior is structurally the construction most prone to chasing one
      historical path) shows a same-sign majority, not a single lucky path;
  (3) survives the 0.40% fee tier, no sign reversal;
  (4) the BETA_PRIOR_GRID sweep is a same-sign plateau;
  (5) bind_frac > 0.01 and r_squared < 0.98 (Step-0, restated);
  (6) causal truncation probe passes.
Given >90% of the last 60 rounds resolved NEGATIVE, a clean negative here is
this project's normal, valued output, not a methodology failure.

FALSIFICATION TEST DESIGN (Monte Carlo windows, pre-registered before any
window's own number was computed): compute ``frac_novel``/``target_novel``
ONCE over the whole inner-train + inner-validation span (2017-01-01 ->
2022-12-31) so the sequential Beta posterior accumulates real, continuous,
causal history from the span's own true start -- never reset cold at a
window boundary. Draw ~40 random (start, length) windows from bar positions
within that span, each requiring >=80 calendar days of real preceding
history within the span before its own start (so the slowest ladder's
80-day-equivalent anchor, and at least the chance of a completed spell, are
warm); slice the ALREADY-COMPUTED, fully causal target/frac arrays at each
window's own [start, end) and re-run each window through
``TargetStrategy``/``run_slice`` (with the strategy's ``warmup`` forced to 0,
since no further indicator history is needed -- the precomputed series
already encodes it) so every window's Sharpe/log-growth number goes through
the SAME fee-charging backtest engine every other cell in this file uses,
not a hand-rolled return calculation. Report the fraction of windows where
the candidate wins on Sharpe (and, secondarily, on log-growth) -- a
same-sign majority is a plateau; concentration near 50% or below is a
single lucky historical path, not a real property of the mechanism.

CONFIGURATIONS EVALUATED IN THIS FILE: 6 (primary cell's full ``compare()``:
inner_train + inner_val + eth_replication x 2 markets) + 40 (Monte Carlo
windows, SPOT only, candidate vs. control each) + 2 (fee-tier re-run,
inner_val, 2 markets) + 4 (BETA_PRIOR_GRID's 2 non-primary priors x 2
markets -- the primary prior's 2 inner_val cells are REUSED from the primary
compare(), not recomputed) = 52 total. The causal truncation probe and the
equal-weight-of-5-ladders degeneracy sanity check are code-correctness unit
checks, not (prior, market, slice) performance cells, and are not counted
in the 52.

USAGE
-----
    python experiments/r147_novel_bma_ladder.py
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

from experiments.r147_shared import (  # noqa: E402
    BARS_PER_DAY,
    BETA_PRIOR_GRID,
    BETA_PRIOR_PRIMARY,
    BIND_FRAC_THRESH,
    FEE_TIER,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    LADDER_BASES,
    LADDERS,
    OOS_START,
    R2_DEGENERACY_THRESH,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TargetStrategy,
    V4_BAND,
    assert_no_holdout,
    beta_bernoulli_posterior_mean,
    bind_frac,
    build_target_from_frac,
    causal_truncation_probe_series,
    compare,
    fee_at,
    latched_state,
    load_btc,
    load_eth,
    normalize_weights,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    spell_hit_series,
    v4_target,
    v4_vote_frac,
    vote_frac,
)


def hr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (1)
# The mechanism itself: 5-member alternative-ladder ensemble, each ladder's
# own equal-weight-of-3 vote, blended by a sequential Beta-Bernoulli
# posterior-reliability weight -- Bayesian Model Averaging in spirit.
# ==================================================================

def compute_ladder_fracs(df: pd.DataFrame) -> dict[int, pd.Series]:
    """One ``vote_frac`` call per ladder (5 total, including the base=20
    member, which is numerically identical to ``v4_vote_frac``)."""
    return {b: vote_frac(df, horizons=LADDERS[b], band=V4_BAND) for b in LADDER_BASES}


def compute_ladder_posts(df: pd.DataFrame, a0: float, b0: float,
                         fracs: dict[int, pd.Series] | None = None) -> dict[int, pd.Series]:
    """Each ladder's own causal spell/hit label and sequential Beta-Bernoulli
    posterior mean reliability -- IDENTICAL machinery for all 5 ladders."""
    if fracs is None:
        fracs = compute_ladder_fracs(df)
    posts: dict[int, pd.Series] = {}
    for b in LADDER_BASES:
        state = latched_state(fracs[b])
        hit = spell_hit_series(state, df["close"])
        posts[b] = beta_bernoulli_posterior_mean(hit, a0, b0)
    return posts


def frac_novel_and_weights(df: pd.DataFrame, a0: float, b0: float):
    """Returns (frac_novel, weights, frac_arr, post_arr). ``weights`` is the
    row-normalized (n_bars, 5) posterior-reliability weight array;
    ``frac_novel`` is the posterior-weighted blend of the 5 ladders' OWN
    votes (each ladder contributes its own already-combined 3-anchor vote
    as one unit, never its internal anchors individually)."""
    fracs = compute_ladder_fracs(df)
    posts = compute_ladder_posts(df, a0, b0, fracs=fracs)
    frac_arr = np.column_stack([fracs[b].to_numpy() for b in LADDER_BASES])
    post_arr = np.column_stack([posts[b].to_numpy() for b in LADDER_BASES])
    weights = normalize_weights(post_arr)
    frac_novel = np.sum(weights * frac_arr, axis=1)
    return frac_novel, weights, frac_arr, post_arr


def build_target(df: pd.DataFrame, a0: float = BETA_PRIOR_PRIMARY[0],
                 b0: float = BETA_PRIOR_PRIMARY[1]) -> np.ndarray:
    """Pure ``df -> np.ndarray`` candidate target, the ``TargetStrategy``/
    ``compare()`` calling convention every branch in this ledger uses."""
    frac_novel, _weights, _frac_arr, _post_arr = frac_novel_and_weights(df, a0, b0)
    return build_target_from_frac(frac_novel, df)


def make_build_target(a0: float, b0: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, a0=a0, b0=b0)
    _build.__name__ = f"bma_ladder_a{a0:g}b{b0:g}"
    return _build


BUILD_PRIMARY = make_build_target(*BETA_PRIOR_PRIMARY)
BUILD_PRIMARY.__name__ = "novel_bma_ladder"


# ================================================================== (2)
# Self-check / causality (docs/ROUTINE.md's own precedence: a lookahead is
# a bug report first). Plus a unit-style sanity check: forcing all 5
# posteriors equal must reduce the blend to the plain equal-weight-of-5-
# ladders average -- explicitly NOT expected to equal v4_target (v4 only
# trades the base=20 ladder alone; an equal blend of 5 ladders is a
# genuinely different, if closely related, object).
# ==================================================================

def equal_weight_sanity_check(df: pd.DataFrame) -> dict:
    fracs = compute_ladder_fracs(df)
    frac_arr = np.column_stack([fracs[b].to_numpy() for b in LADDER_BASES])
    # Any constant, equal-across-columns reliability score must normalize to
    # exactly equal weights, by normalize_weights' own row-normalization.
    forced_equal_posts = np.full_like(frac_arr, 0.5)
    weights = normalize_weights(forced_equal_posts)
    weights_equal = bool(np.allclose(weights, 1.0 / len(LADDER_BASES)))

    frac_forced = np.sum(weights * frac_arr, axis=1)
    frac_equal_mean = np.nanmean(frac_arr, axis=1)
    frac_matches_equal_mean = bool(np.allclose(frac_forced, frac_equal_mean, equal_nan=True))

    target_forced = build_target_from_frac(frac_forced, df)
    target_equal_mean = build_target_from_frac(frac_equal_mean, df)
    target_matches = bool(np.allclose(target_forced, target_equal_mean, equal_nan=True))

    v4t = v4_target(df)
    matches_v4_target = bool(np.allclose(target_forced, v4t, equal_nan=True))

    return dict(weights_equal=weights_equal, frac_matches_equal_mean=frac_matches_equal_mean,
               target_matches=target_matches, matches_v4_target=matches_v4_target)


# ================================================================== (3)
# Step-0 kill switches (gating): bind_frac and R^2 vs v4's own vote, on BTC
# inner-train, primary prior only (no grid-selection search in this branch).
# ==================================================================

def step0_kill_switches(btc: pd.DataFrame) -> dict:
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                      (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    frac_novel, weights, _frac_arr, _post_arr = frac_novel_and_weights(btc, *BETA_PRIOR_PRIMARY)
    bf = bind_frac(weights[mask])
    ctrl_vote = v4_vote_frac(btc).to_numpy()
    r2 = r_squared(frac_novel[mask], ctrl_vote[mask])
    bind_ok = bf > BIND_FRAC_THRESH
    r2_ok = r2 < R2_DEGENERACY_THRESH
    return dict(bind_frac=bf, r_sq=r2, bind_ok=bind_ok, r2_ok=r2_ok,
               n_bars=int(mask.sum()), gate_pass=bool(bind_ok and r2_ok))


# ================================================================== (4)
# Falsification test: Monte Carlo resampled windows. Compute frac_novel /
# target_novel ONCE over the whole inner-train+inner-val span (causal,
# continuous, real history from the span's own true start), then slice the
# ALREADY-COMPUTED result per window and re-run each window through
# TargetStrategy/run_slice (warmup forced to 0) so every window's Sharpe/
# log-growth goes through the identical fee-charging backtest engine every
# other cell in this file uses.
# ==================================================================

def _build_from_series(series: pd.Series, name: str):
    def _build(frame: pd.DataFrame) -> np.ndarray:
        return series.reindex(frame.index).to_numpy()
    _build.__name__ = name
    return _build


def monte_carlo_windows(btc: pd.DataFrame, n_windows: int = 40, seed: int = 147,
                        min_start_offset_days: int = 80, market=SPOT) -> dict:
    span_start = pd.Timestamp(INNER_TRAIN_START, tz="UTC")
    span_end = pd.Timestamp(INNER_VAL_END, tz="UTC")
    span = btc.loc[(btc.index >= span_start) & (btc.index <= span_end)]
    assert_no_holdout(span, "monte_carlo_windows(): span")
    n = len(span)

    # Compute the full causal path ONCE, over the whole span.
    frac_novel, _weights, _frac_arr, _post_arr = frac_novel_and_weights(span, *BETA_PRIOR_PRIMARY)
    novel_target = pd.Series(build_target_from_frac(frac_novel, span), index=span.index)
    v4_full_target = pd.Series(v4_target(span), index=span.index)

    cand_strategy = TargetStrategy(_build_from_series(novel_target, "novel_precomp"),
                                   name="novel_bma_ladder_mc", warmup=0)
    ctrl_strategy = TargetStrategy(_build_from_series(v4_full_target, "v4_precomp"),
                                   name="kelly_regime_v4_mc", warmup=0)

    min_start_bar = min_start_offset_days * BARS_PER_DAY
    length_min_bars = 30 * BARS_PER_DAY
    length_max_bars = min(400 * BARS_PER_DAY, n - min_start_bar - 1)
    assert length_max_bars > length_min_bars, "span too short for the requested MC window grid"

    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_windows):
        length_bars = int(rng.integers(length_min_bars, length_max_bars + 1))
        max_start_bar = n - length_bars
        start_bar = int(rng.integers(min_start_bar, max_start_bar + 1))
        end_bar = start_bar + length_bars - 1  # inclusive end position

        w_start = span.index[start_bar]
        w_end = span.index[end_bar]
        # run_slice's own OOS guard does `pd.Timestamp(end) < pd.Timestamp(OOS_START)`;
        # OOS_START is a naive date string, so it must be compared against a naive
        # timestamp too -- pass ISO strings (date+time, no tz) rather than the
        # tz-aware Timestamp objects themselves, which searchsorted resolves
        # against the (tz-aware) index identically.
        w_start_s = w_start.tz_localize(None).isoformat()
        w_end_s = w_end.tz_localize(None).isoformat()

        a = run_slice(cand_strategy, span, w_start_s, w_end_s, f"mc_window_{i}", market)
        b = run_slice(ctrl_strategy, span, w_start_s, w_end_s, f"mc_window_{i}", market)
        d_sharpe = a.sharpe - b.sharpe
        d_loggrowth = a.log_growth - b.log_growth
        results.append(dict(
            i=i, start=str(w_start.date()), end=str(w_end.date()),
            length_days=length_bars // BARS_PER_DAY,
            cand_sharpe=a.sharpe, ctrl_sharpe=b.sharpe, d_sharpe=d_sharpe,
            cand_loggrowth=a.log_growth, ctrl_loggrowth=b.log_growth, d_loggrowth=d_loggrowth,
            cand_wins_sharpe=bool(d_sharpe > 0), cand_wins_loggrowth=bool(d_loggrowth > 0),
        ))

    win_frac_sharpe = float(np.mean([r["cand_wins_sharpe"] for r in results]))
    win_frac_loggrowth = float(np.mean([r["cand_wins_loggrowth"] for r in results]))
    return dict(results=results, n_windows=len(results), market=market.name,
               win_frac_sharpe=win_frac_sharpe, win_frac_loggrowth=win_frac_loggrowth,
               plateau_pass=bool(win_frac_sharpe > 0.5))


def print_mc_summary(mc: dict) -> None:
    print(f"market={mc['market']}  n_windows={mc['n_windows']}  "
         f"candidate wins on Sharpe: {mc['win_frac_sharpe']:.1%}   "
         f"candidate wins on log-growth: {mc['win_frac_loggrowth']:.1%}")
    print(f"\n{'i':>3s} {'start':>11s} {'end':>11s} {'len(d)':>7s} "
         f"{'cSh':>7s} {'vSh':>7s} {'dSh':>7s} {'dlogG':>8s} {'winSh':>6s} {'winLG':>6s}")
    for r in mc["results"]:
        print(f"{r['i']:3d} {r['start']:>11s} {r['end']:>11s} {r['length_days']:7d} "
             f"{r['cand_sharpe']:7.2f} {r['ctrl_sharpe']:7.2f} {r['d_sharpe']:+7.2f} "
             f"{r['d_loggrowth']:+8.3f} "
             f"{'Y' if r['cand_wins_sharpe'] else 'n':>6s} "
             f"{'Y' if r['cand_wins_loggrowth'] else 'n':>6s}")


# ================================================================== (5)
# Fee robustness: re-run the decisive inner-validation cells at the 0.40%
# taker tier, both markets, and check no sign reversal vs. the standard-fee
# primary cell.
# ==================================================================

def fee_robustness_check(btc: pd.DataFrame, primary_inner_val_rows: list[dict]) -> list[dict]:
    cand = TargetStrategy(BUILD_PRIMARY, name="novel_bma_ladder_fee")
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4_fee")
    rows = []
    for market in (SPOT, FUTURES):
        fee_market = fee_at(market, FEE_TIER)
        a = run_slice(cand, btc, INNER_VAL_START, INNER_VAL_END, "inner_val_fee", fee_market)
        b = run_slice(ctrl, btc, INNER_VAL_START, INNER_VAL_END, "inner_val_fee", fee_market)
        pr = paired_diff(a.daily, b.daily)
        base = next((r for r in primary_inner_val_rows if r["market"] == market.name), None)
        base_d_sharpe = base["d_sharpe"] if base else float("nan")
        base_boot = base["boot_d_loggrowth"] if base else float("nan")
        d_sharpe = a.sharpe - b.sharpe
        boot_d_loggrowth = pr.diff.point
        no_reversal_sharpe = not (np.sign(d_sharpe) != np.sign(base_d_sharpe) and
                                  d_sharpe != 0 and base_d_sharpe != 0)
        no_reversal_boot = not (np.sign(boot_d_loggrowth) != np.sign(base_boot) and
                                boot_d_loggrowth != 0 and base_boot != 0)
        rows.append(dict(market=market.name, cand_sharpe=a.sharpe, ctrl_sharpe=b.sharpe,
                         d_sharpe=d_sharpe, base_d_sharpe=base_d_sharpe,
                         boot_d_loggrowth=boot_d_loggrowth, base_boot_d_loggrowth=base_boot,
                         boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
                         excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0),
                         no_reversal=bool(no_reversal_sharpe and no_reversal_boot)))
    return rows


# ================================================================== (6)
# Plateau check (B4): sweep BETA_PRIOR_GRID on BTC inner-validation. The
# primary prior's 2 cells are REUSED from the main compare() call, not
# recomputed; only the 2 non-primary priors are freshly evaluated.
# ==================================================================

def inner_val_only_rows(build_fn, label: str, btc: pd.DataFrame,
                        markets=(SPOT, FUTURES)) -> list[dict]:
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r147_{label}")
    rows = []
    for market in markets:
        a = run_slice(cand, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        b = run_slice(ctrl, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        pr = paired_diff(a.daily, b.daily)
        rows.append(dict(label=label, market=market.name, d_sharpe=a.sharpe - b.sharpe,
                         boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
                         excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0)))
    return rows


def prior_grid_plateau(btc: pd.DataFrame, primary_inner_val_rows: list[dict]) -> dict:
    plateau: dict[tuple, list[dict]] = {}
    n_fresh = 0
    for (a0, b0) in BETA_PRIOR_GRID:
        if (a0, b0) == BETA_PRIOR_PRIMARY:
            plateau[(a0, b0)] = [dict(label="primary(reused)", market=r["market"],
                                      d_sharpe=r["d_sharpe"], boot_d_loggrowth=r["boot_d_loggrowth"],
                                      boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                                      excludes_zero=r["excludes_zero"])
                                 for r in primary_inner_val_rows]
        else:
            bf = make_build_target(a0, b0)
            plateau[(a0, b0)] = inner_val_only_rows(bf, f"a{a0:g}b{b0:g}", btc)
            n_fresh += len(plateau[(a0, b0)])

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau.values() for r in rows]
    plateau_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return dict(plateau=plateau, plateau_pass=plateau_pass, n_fresh_configs=n_fresh)


def print_plateau_table(plateau: dict) -> None:
    hdr_line = (f"{'prior':>12s} {'market':>9s} {'dSh':>7s} {'dlogG':>8s} "
               f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr_line)
    print("-" * len(hdr_line))
    for (a0, b0), rows in plateau.items():
        for r in rows:
            print(f"{f'({a0:g},{b0:g})':>12s} {r['market']:>9s} {r['d_sharpe']:+7.2f} "
                 f"{r['boot_d_loggrowth']:+8.3f} {r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
                 f"{'YES' if r['excludes_zero'] else 'no':>5s}")


# ================================================================== (7)
# Promotion-bar clause (1): primary-cell dSharpe on BTC inner-train AND
# inner-validation, both markets.
# ==================================================================

def primary_edge_check(rows: list[dict]) -> tuple[bool, list[dict]]:
    cells = []
    for r in rows:
        if r["slice"] not in ("inner_train", "inner_val"):
            continue
        sharpe_edge = (r["d_sharpe"] > SHARPE_NOISE_FLOOR and r["excludes_zero"] and
                      r["boot_d_loggrowth"] > 0)
        risk_matched_improvement = r["risk_matched"] and (r["d_dd"] < 0)
        passes = bool(sharpe_edge or risk_matched_improvement)
        cells.append(dict(slice=r["slice"], market=r["market"], d_sharpe=r["d_sharpe"],
                          boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                          exposure_ratio=r["exposure_ratio"], vol_ratio=r["vol_ratio"],
                          risk_matched=r["risk_matched"], d_dd=r["d_dd"],
                          sharpe_edge=sharpe_edge, risk_matched_improvement=risk_matched_improvement,
                          passes=passes))
    return all(c["passes"] for c in cells) if cells else False, cells


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-147 NOVEL: BmaLadderKellyV4 -- 5-member alternative-ladder ensemble, "
      "combined by a sequential Beta-Bernoulli posterior-reliability weight")
    print("mechanism: each of 5 pre-registered alternative anchor-ladder specifications of v4's")
    print("own directional vote (geometric doubling-ladder family, bases 10/15/20/25/30 days,")
    print("reused verbatim from R-105) gets its OWN continuously-updated sequential Bayesian")
    print("posterior reliability (Beta-Bernoulli conjugate updating on that ladder's own causal")
    print("spell/hit record); the traded signal is the posterior-weighted blend of all 5")
    print("ladders' OWN votes -- Bayesian Model Averaging in spirit, replacing v4's fixed 1/3-")
    print("each anchor average at one level up (across ladders, not across the 3 anchors within")
    print("one ladder -- each ladder's own internal 3-anchor equal-weight vote is held fixed).")
    print(f"\nladders: {LADDERS}")
    print(f"primary Beta prior: {BETA_PRIOR_PRIMARY}   full grid: {BETA_PRIOR_GRID}")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
         f"{btc.index[0]} -> {btc.index[-1]}")

    # ============================================================= STEP 1
    hr("STEP 1 -- SELF-CHECK / CAUSALITY")
    print(f"causal_truncation_probe_series({BUILD_PRIMARY.__name__}, btc):")
    try:
        probe_ok = causal_truncation_probe_series(BUILD_PRIMARY, btc)
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")

    hr("STEP 1b -- EQUAL-WEIGHT-OF-5-LADDERS DEGENERACY SANITY CHECK")
    ewc = equal_weight_sanity_check(btc)
    print(f"forcing all 5 posteriors equal -> weights are exactly 1/5 each: {ewc['weights_equal']}")
    print(f"resulting frac_novel matches the plain equal-weight-of-5-ladders mean: "
         f"{ewc['frac_matches_equal_mean']}")
    print(f"resulting target matches build_target_from_frac(equal-mean-of-5): "
         f"{ewc['target_matches']}")
    print(f"resulting target matches v4_target EXACTLY: {ewc['matches_v4_target']}  "
         f"(NOT expected to match -- v4 trades ONLY the base=20 ladder; an equal blend of 5")
    print("ladders is a genuinely different, if closely related, object. A mismatch here is")
    print("the expected, correct behaviour, not a bug.)")

    # ============================================================= STEP 2
    hr("STEP 2 -- STEP-0 KILL SWITCHES (BTC inner-train, primary prior)")
    step0 = step0_kill_switches(btc)
    print(f"n_bars (inner-train)  = {step0['n_bars']:,}")
    print(f"bind_frac(weights)    = {step0['bind_frac']:.4f}  "
         f"(threshold > {BIND_FRAC_THRESH:.2%}): {'PASS' if step0['bind_ok'] else 'FAIL'}")
    print(f"r_squared(frac_novel, v4_vote_frac) = {step0['r_sq']:.4f}  "
         f"(threshold < {R2_DEGENERACY_THRESH}): {'PASS' if step0['r2_ok'] else 'FAIL'}")
    print(f"\nSTEP-0 GATE PASS: {step0['gate_pass']}")

    if not step0["gate_pass"]:
        print("\nSTEP-0 KILL SWITCH TRIPPED -- per this round's own pre-registration, this is")
        print("reported NEGATIVE immediately. Continuing to run every remaining check below")
        print("regardless (this project's convention: every branch reports, including the")
        print("dead ones), but NO number after this point can rescue the verdict, and the")
        print("holdout will not be touched.")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    # ============================================================= STEP 3
    hr("STEP 3 -- FULL COMPARISON: compare() over inner_train / inner_val / "
      "eth_replication, SPOT + FUTURES")
    rows = compare(BUILD_PRIMARY, label="novel_bma_ladder", btc=btc, eth=eth,
                  markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    inner_train_primary = [r for r in rows if r["slice"] == "inner_train"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    edge_pass, edge_cells = primary_edge_check(rows)
    hr("PROMOTION-BAR CLAUSE (1) -- primary-cell dSharpe, BTC inner-train AND inner-val, "
      "both markets")
    for c in edge_cells:
        print(f"  {c['slice']:>12s} {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
             f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  expR={c['exposure_ratio']:.2f}  "
             f"volR={c['vol_ratio']:.2f}  risk_matched={c['risk_matched']}  d_dd={c['d_dd']:+.2f}pp  "
             f"sharpe_edge={c['sharpe_edge']}  risk_matched_improvement={c['risk_matched_improvement']}  "
             f"PASS={c['passes']}")
    print(f"\nCLAUSE (1) PASS (all 4 cells): {edge_pass}")

    hr("ETH REPLICATION (reported, not itself a promotion-bar clause here -- diagnostic)")
    for r in eth_primary:
        print(f"  {r['market']:>9s}  d_sharpe={r['d_sharpe']:+.4f}  "
             f"boot=[{r['boot_lo']:+.4f},{r['boot_hi']:+.4f}]")

    # ============================================================= STEP 4
    hr("STEP 4 -- FALSIFICATION TEST: MONTE CARLO RESAMPLED WINDOWS "
      "(pre-registered; SPOT market, ~40 windows)")
    mc = monte_carlo_windows(btc, n_windows=40, seed=147, min_start_offset_days=80, market=SPOT)
    print_mc_summary(mc)
    print(f"\nPROMOTION-BAR CLAUSE (2) -- same-sign majority (win_frac_sharpe > 0.5): "
         f"{mc['plateau_pass']}")

    # ============================================================= STEP 5
    hr("STEP 5 -- FEE ROBUSTNESS (0.40% taker), BTC inner-validation, both markets")
    fee_rows = fee_robustness_check(btc, inner_val_primary)
    for r in fee_rows:
        print(f"  {r['market']:>9s}  fee-tier d_sharpe={r['d_sharpe']:+.4f}  "
             f"standard-fee d_sharpe={r['base_d_sharpe']:+.4f}  "
             f"fee-tier boot_d_loggrowth={r['boot_d_loggrowth']:+.4f}  "
             f"standard-fee boot_d_loggrowth={r['base_boot_d_loggrowth']:+.4f}  "
             f"no_reversal={r['no_reversal']}")
    fee_pass = all(r["no_reversal"] for r in fee_rows)
    print(f"\nPROMOTION-BAR CLAUSE (3) -- no sign reversal at 0.40% fee tier (both markets): "
         f"{fee_pass}")

    # ============================================================= STEP 6
    hr("STEP 6 -- PLATEAU CHECK: BETA_PRIOR_GRID sweep, BTC inner-validation, both markets")
    plateau = prior_grid_plateau(btc, inner_val_primary)
    print_plateau_table(plateau["plateau"])
    print(f"\nPROMOTION-BAR CLAUSE (4) -- same-sign plateau across the 3-prior grid: "
         f"{plateau['plateau_pass']}")

    # ============================================================= STEP 7
    hr("STEP 7 -- CONFIGURATION COUNT")
    n_compare = len(rows)                      # 6: 3 slices x 2 markets
    n_mc = mc["n_windows"]                      # 40
    n_fee = len(fee_rows)                       # 2
    n_plateau_fresh = plateau["n_fresh_configs"]  # 4 (2 non-primary priors x 2 markets)
    n_configs = n_compare + n_mc + n_fee + n_plateau_fresh
    print(f"primary compare() cells:            {n_compare}")
    print(f"Monte Carlo windows:                 {n_mc}")
    print(f"fee-tier cells:                      {n_fee}")
    print(f"BETA_PRIOR_GRID fresh cells:          {n_plateau_fresh}  "
         f"(2 primary-prior inner_val cells reused from the primary compare(), not recounted)")
    print(f"TOTAL CONFIGURATIONS EVALUATED:      {n_configs}")
    print("(the causal truncation probe and the equal-weight-of-5-ladders sanity check are")
    print(" code-correctness unit checks, not (prior, market, slice) performance cells, and")
    print(" are excluded from this count.)")

    # ============================================================= VERDICT
    hr("VERDICT")
    clause5 = step0["gate_pass"]
    clause6 = bool(probe_ok)
    all_pass = bool(edge_pass and mc["plateau_pass"] and fee_pass and plateau["plateau_pass"] and
                   clause5 and clause6)
    print(f"(1) primary-cell dSharpe, BTC inner-train+inner-val, both markets: {edge_pass}")
    print(f"(2) Monte Carlo same-sign majority:                                {mc['plateau_pass']}")
    print(f"(3) 0.40% fee-tier survival, no sign reversal:                     {fee_pass}")
    print(f"(4) BETA_PRIOR_GRID same-sign plateau:                             {plateau['plateau_pass']}")
    print(f"(5) Step-0 non-degeneracy (bind_frac>1% AND r_sq<0.98):            {clause5}")
    print(f"(6) causal truncation probe:                                       {clause6}")
    verdict = "PROMOTE-candidate (gate clears; holdout may be consulted)" if all_pass else "NEGATIVE"
    print(f"\nALL PROMOTION-BAR CLAUSES PASS: {all_pass}")
    print(f"VERDICT: {verdict}")
    if all_pass:
        print("\nGate clears on inner-train/inner-val/ETH/fee/Monte-Carlo -- holdout MAY be")
        print("consulted per docs/ROUTINE.md step 4. This file does NOT itself read OOS_START;")
        print("that is a separate, explicit step left to the operator/session that acts on")
        print("this verdict.")
    else:
        print("\nGate does NOT clear -- per docs/ROUTINE.md's own discipline, the holdout is")
        print("precious and is NOT touched. No bar at or after OOS_START is read anywhere in")
        print("this file.")

    max_ts = max(max_ts_seen)
    print(f"\nmax timestamp read anywhere in this branch: {max_ts}  "
         f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, probe_ok=probe_ok, equal_weight_check=ewc, step0=step0,
               compare_rows=rows, edge_pass=edge_pass, edge_cells=edge_cells,
               mc=mc, fee_rows=fee_rows, fee_pass=fee_pass, plateau=plateau,
               n_configs=n_configs, max_ts=max_ts, all_pass=all_pass, verdict=verdict)


if __name__ == "__main__":
    main()
