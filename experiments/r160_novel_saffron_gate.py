#!/usr/bin/env python
"""R-160 NOVEL branch: gate ``kelly_regime_v4``'s three anchor votes' FLIP
decisions through SAFFRON (Ramdas, Zrnic, Wainwright & Jordan 2018, ICML,
arXiv:1802.09098), an online false-discovery-rate control procedure that
improves on classical LORD by estimating, online, which candidate flips are
likely true nulls (p-value above a fixed candidate threshold ``lam``) and
spending NO wealth on them -- so wealth is concentrated on flips that were
ever plausibly going to be accepted, giving more power (a more responsive
gate) than a naive LORD-style gate at the same nominal alpha budget.

EXACT CONSTRUCTION. For each anchor (20/40/80-day, v4's own construction,
unmodified): ``raw, p = anchor_raw_vote_and_pvalues(close, days)`` (frozen in
``r160_shared.py``). At every bar where the raw vote currently disagrees with
the gate's own held state (a flip is "pending"), that bar's p-value is a
fresh SAFFRON test: candidates with ``p_t > lam`` are treated as (probably)
true nulls and consume NO wealth even though "tested"; candidates with
``p_t <= lam`` are real discovery attempts, tested against an adaptively
shrinking threshold ``alpha_t <= lam`` and accepted (gate state jumps to
``raw_vote[i]``) iff ``p_t <= alpha_t``. The three gated per-anchor votes are
combined by v4's own equal-weight ``combine_gated_votes`` and wired through
v4's own unmodified ``build_target_from_frac`` (scale + 10% deadband) --
this file changes nothing about magnitude or sizing, only WHEN a flip lands.

Full citation trail, the "not a duplicate of" argument, the pre-registered
decision rule and the four named failure modes are all in ``r160_shared.py``'s
own module docstring (read in full before this file was written); not
re-derived here. This file never edits, and never reads a bar at or after
``r160_shared.OOS_START`` from, ``r160_shared.py`` or any other file.

SAFFRON GATE -- exact wealth-update formula (stated once here, used
verbatim in ``saffron_gate`` below; see comment on that function for the
line-by-line citation to Ramdas et al. 2018, approximately their Algorithm 1
/ Eqs 3-4, "SAFFRON"):

  Let C(t) = the running COUNT of candidates (bars with a pending flip whose
  own p-value is <= lam) seen strictly before the current candidate, and let
  J = C(t)+1 be the current candidate's own 1-indexed position in that
  candidate-only stream (non-candidates -- p_t > lam -- never increment C
  and are skipped entirely: this is the one-line difference from LORD that
  gives SAFFRON its power gain). Let tau_1 < tau_2 < ... be the candidate
  INDICES (not calendar bar indices) of every past ACCEPTED flip (a
  "rejection" in the online-FDR sense) on this anchor.

    alpha_t = clip( W0 * gamma(J)
                     + (alpha*(1-lam) - W0) * gamma(J - tau_1)   [if >=1 rejection]
                     + alpha*(1-lam) * sum_{k>=2} gamma(J - tau_k),
                     0, lam )

  where gamma(j) = 6/(pi^2 * j^2) for j>=1 (any nonincreasing, nonnegative,
  summable-to-1 sequence satisfies the paper's proof; this closed form is
  used here in place of the paper's own numerically-fit asymptotically
  optimal sequence, for simplicity/reproducibility -- Ramdas et al. 2018
  sec 3 states the guarantee holds "for any such {gamma_j}"), and
  W0 = w0_fraction * alpha is the initial wealth. Accept iff p_t <= alpha_t.
  Wealth is replenished ONLY by past acceptances (the gamma-discounted sum
  above), exactly LORD's mechanism -- the SAFFRON correction is entirely in
  which bars get to consume a slot in the {gamma(1), gamma(2), ...} sequence
  at all: only real candidates (p_t <= lam) do, so a long run of obviously-
  null pending-flip bars (p_t near 1) does NOT erode the threshold available
  to the next real candidate, unlike plain LORD which tests (and thus
  shrinks the discount index for) every pending-flip bar regardless of p_t.

ADDIS BONUS (optional per pre-registration, implemented here as one extra
comparison cell, not the primary deliverable): ``addis_gate`` is the same
recursion with the earned-wealth-per-rejection quantum rescaled from
``alpha*(1-lam)`` to ``alpha*(tau-lam)`` (``tau`` a discard threshold,
``lam < tau < 1``) -- Tian & Ramdas (2019)'s headline intuition ("less
wealth calibrated against a discard region that is provably ~always null
frees more for the tested region"), NOT a verbatim reproduction of their
two-threshold ``C_t^tau``/normalizing-constant bookkeeping (disclosed
honestly in the function's own docstring below). As ``tau -> 1`` this
recursion reduces exactly to ``saffron_gate``.

USAGE
-----
    python experiments/r160_novel_saffron_gate.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r160_shared import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_PRIMARY,
    FUTURES,
    GATE_MIN_DELAYS,
    OOS_START,
    R2_DEGENERACY_THRESH,
    SHARPE_NOISE_FLOOR,
    SPOT,
    V4_HORIZONS,
    W0_FRACTION,
    anchor_raw_vote_and_pvalues,
    assert_no_holdout,
    build_target_from_frac,
    causal_truncation_probe_series,
    combine_gated_votes,
    compare,
    count_delayed_episodes,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    synthetic_zero_drift_frame,
    v4_vote_frac,
)

# Novel-branch-owned hyperparameter grid: SAFFRON's candidate threshold
# `lam`, not part of r160_shared's frozen ALPHA_GRID (which is shared by
# both branches) since it has no analogue in the conservative (LORD) branch.
# PRIMARY = 0.5 (symmetric, "half the null mass is plausibly a candidate"),
# a priori, not fitted; the other two are a plateau/robustness check.
LAM_GRID: tuple[float, ...] = (0.5, 0.3, 0.7)
LAM_PRIMARY = LAM_GRID[0]
TAU_ADDIS = 0.9  # ADDIS discard threshold, fixed a priori (tau > every LAM_GRID value)


def hr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (1)
# The gate itself.
# ==================================================================

def gamma_seq(j: int) -> float:
    """gamma(j) = 6/(pi^2 j^2), j>=1: nonincreasing, nonnegative, sums to
    exactly 1 over j=1,2,... (sum 1/j^2 = pi^2/6). Any such sequence
    satisfies LORD/SAFFRON's FDR proof (Ramdas et al. 2018 sec 3); this
    closed form replaces the paper's own numerically-optimized sequence."""
    if j < 1:
        return 0.0
    return 6.0 / (math.pi ** 2 * j * j)


def saffron_gate(raw_vote: np.ndarray, pvalues: np.ndarray, alpha: float,
                 lam: float = 0.5, w0_fraction: float = 0.5) -> np.ndarray:
    """Causal SAFFRON gate on one anchor's raw 0/1 vote stream.

    At bar 0 the gate starts matched to the raw vote (no test needed -- a
    boundary convention, not a flip decision, and uses only raw_vote[0]).
    From bar 1 on: if `raw_vote[i]` agrees with the gate's currently held
    state, nothing to test (not a "candidate" in the online-FDR sense at
    all -- no flip is pending). If a flip IS pending (raw disagrees), that
    bar's own p-value is tested:
      - `p_t > lam`  -> treated as a likely true null; consumes NO wealth,
        no threshold is even computed, the flip stays pending (may be
        retested next bar with a fresh p-value, since `pvalues` is defined
        at every bar).
      - `p_t <= lam` -> a genuine candidate. `C` (the running candidate
        count on this anchor) increments; `J = C` is this candidate's own
        1-indexed position in the candidate-only stream. The threshold
          alpha_t = clip(W0*gamma(J)
                          + (alpha*(1-lam) - W0) * gamma(J - tau_1)   [k=1]
                          + alpha*(1-lam) * sum_{k>=2} gamma(J - tau_k),
                          0, lam)
        (tau_k = the candidate-index J-value AT each of this anchor's own
        past accepted flips, in order) is computed and, iff `p_t <= alpha_t`,
        the flip is ACCEPTED: gate state jumps to `raw_vote[i]` and this
        candidate's own J is appended to the rejection list, replenishing
        future wealth via the same gamma-discount sequence LORD itself uses
        (Ramdas et al. 2018, approximately their Algorithm 1 / Eqs 3-4 --
        the only departure from their exact notation is the closed-form
        `gamma_seq` above in place of their fitted sequence, and indexing
        purely on the CANDIDATE stream `J`, which is the definitional crux
        of SAFFRON vs LORD).

    Purely causal: the decision at bar i is a function of `raw_vote[:i+1]`
    and `pvalues[:i+1]` only (all state -- `C`, `reject_J`, `gate_state` --
    is carried forward from strictly earlier bars).
    """
    raw = np.asarray(raw_vote, dtype=float)
    p = np.asarray(pvalues, dtype=float)
    n = len(raw)
    gated = np.empty(n, dtype=float)
    if n == 0:
        return gated
    gate_state = raw[0]
    gated[0] = gate_state

    w0 = w0_fraction * alpha
    c = 0
    reject_j: list[int] = []

    for i in range(1, n):
        if raw[i] == gate_state:
            gated[i] = gate_state
            continue
        pt = p[i]
        if pt > lam:
            # Likely true null: no wealth spent, no threshold set, flip
            # stays pending -- the SAFFRON correction over LORD.
            gated[i] = gate_state
            continue
        c += 1
        j = c
        alpha_t = w0 * gamma_seq(j)
        for idx, tau_k in enumerate(reject_j):
            coef = (alpha * (1.0 - lam) - w0) if idx == 0 else alpha * (1.0 - lam)
            alpha_t += coef * gamma_seq(j - tau_k)
        alpha_t = max(0.0, min(alpha_t, lam))
        if pt <= alpha_t:
            gate_state = raw[i]
            reject_j.append(j)
        gated[i] = gate_state
    return gated


def addis_gate(raw_vote: np.ndarray, pvalues: np.ndarray, alpha: float,
               lam: float = 0.5, w0_fraction: float = 0.5,
               tau: float = TAU_ADDIS) -> np.ndarray:
    """ADDIS-flavoured refinement of `saffron_gate` (Tian & Ramdas 2019,
    arXiv:1905.11465): identical recursion, EXCEPT the earned-wealth-per-
    rejection quantum is rescaled from `alpha*(1-lam)` to `alpha*(tau-lam)`
    (`tau` a discard threshold, `lam < tau < 1`) -- the paper's headline
    intuition that less wealth need be "reserved" against a region of
    p-space that is provably ~always null (`p_t > tau`), freeing more for
    the region a real discovery could come from. This is a DISCLOSED
    APPROXIMATION, not a verbatim reproduction of ADDIS's own two-threshold
    `C_t^tau` candidate-count bookkeeping (which additionally tracks a
    SEPARATE, tau-screened running count distinct from `C`'s lam-screened
    one; not implemented here for time). As tau -> 1, `alpha*(tau-lam) ->
    alpha*(1-lam)` and this reduces exactly to `saffron_gate`. Candidacy
    itself is still governed by `lam`, unchanged from SAFFRON."""
    if not (lam < tau < 1.0):
        raise ValueError(f"ADDIS requires lam < tau < 1, got lam={lam}, tau={tau}")
    raw = np.asarray(raw_vote, dtype=float)
    p = np.asarray(pvalues, dtype=float)
    n = len(raw)
    gated = np.empty(n, dtype=float)
    if n == 0:
        return gated
    gate_state = raw[0]
    gated[0] = gate_state

    w0 = w0_fraction * alpha
    c = 0
    reject_j: list[int] = []

    for i in range(1, n):
        if raw[i] == gate_state:
            gated[i] = gate_state
            continue
        pt = p[i]
        if pt > lam:
            gated[i] = gate_state
            continue
        c += 1
        j = c
        alpha_t = w0 * gamma_seq(j)
        for idx, tau_k in enumerate(reject_j):
            coef = (alpha * (tau - lam) - w0) if idx == 0 else alpha * (tau - lam)
            alpha_t += coef * gamma_seq(j - tau_k)
        alpha_t = max(0.0, min(alpha_t, lam))
        if pt <= alpha_t:
            gate_state = raw[i]
            reject_j.append(j)
        gated[i] = gate_state
    return gated


# ================================================================== (2)
# Per-anchor -> combined frac -> target, with content-keyed caching so the
# 2x (market-pair) and partial slice redundancy inside one compare() call
# does not re-run the (pure-Python, per-bar) gate loop needlessly. Keyed on
# CONTENT (index bounds + a close-price fingerprint), never on `id(df)`,
# because `run_period` slices a fresh DataFrame object per call even when
# the content is identical -- and, critically, the causal truncation probe
# feeds two SAME-SHAPED, SAME-INDEX frames (full vs tail-perturbed) that
# must NOT collide in the cache, or the probe would silently pass vacuously.
# ==================================================================

_COMPONENT_CACHE: dict[tuple, tuple] = {}


def _frame_key(df: pd.DataFrame) -> tuple:
    idx = df.index
    c = df["close"].to_numpy()
    return (int(idx[0].value), int(idx[-1].value), len(df), float(np.sum(c)), float(c[-1]))


def saffron_frac_and_anchors(df: pd.DataFrame, alpha: float, lam: float,
                             w0_fraction: float = W0_FRACTION,
                             gate_fn=saffron_gate) -> tuple[np.ndarray, list, list]:
    close = df["close"]
    raws, gateds = [], []
    for days in V4_HORIZONS:
        raw, pv = anchor_raw_vote_and_pvalues(close, days)
        gated = gate_fn(raw, pv, alpha, lam, w0_fraction)
        raws.append(raw)
        gateds.append(gated)
    frac = combine_gated_votes(gateds)
    return frac, raws, gateds


def saffron_components(df: pd.DataFrame, alpha: float, lam: float,
                       w0_fraction: float = W0_FRACTION, gate_fn=saffron_gate,
                       gate_name: str = "saffron") -> tuple:
    key = (_frame_key(df), round(alpha, 6), round(lam, 6), round(w0_fraction, 6), gate_name)
    if key not in _COMPONENT_CACHE:
        frac, raws, gateds = saffron_frac_and_anchors(df, alpha, lam, w0_fraction, gate_fn)
        target = build_target_from_frac(frac, df)
        _COMPONENT_CACHE[key] = (frac, raws, gateds, target)
    return _COMPONENT_CACHE[key]


def make_build_target(alpha: float, lam: float, w0_fraction: float = W0_FRACTION,
                      gate_fn=saffron_gate, gate_name: str = "saffron"):
    def _build(df: pd.DataFrame) -> np.ndarray:
        _frac, _raws, _gateds, target = saffron_components(df, alpha, lam, w0_fraction,
                                                           gate_fn, gate_name)
        return target
    _build.__name__ = f"{gate_name}_a{alpha:g}_l{lam:g}"
    return _build


BUILD_PRIMARY = make_build_target(ALPHA_PRIMARY, LAM_PRIMARY)
BUILD_PRIMARY.__name__ = "novel_saffron_gate"


# ================================================================== (3)
# Calibration self-test: synthetic ZERO-DRIFT noise. Under H0 everywhere,
# a well-calibrated gate should accept flips at roughly its nominal alpha
# rate or less (of CANDIDATE tests), not dramatically more. Sanity check,
# not a proof -- report honestly either way.
# ==================================================================

def calibration_self_test(alpha_grid=ALPHA_GRID, lam_grid=LAM_GRID,
                          w0_fraction: float = W0_FRACTION) -> list[dict]:
    z = synthetic_zero_drift_frame()
    assert_no_holdout(z, "calibration_self_test(): synthetic")
    close = z["close"]
    rows = []
    for alpha in alpha_grid:
        for lam in lam_grid:
            for days in V4_HORIZONS:
                raw, pv = anchor_raw_vote_and_pvalues(close, days)
                gated = saffron_gate(raw, pv, alpha, lam, w0_fraction)
                # Candidate tests: pending-flip bars with p<=lam. Acceptances:
                # bars where the gate's own state just changed. Recomputed by
                # walking the same causal counters the gate itself used,
                # directly from (raw, pv, gated).
                n_candidates = 0
                n_accepted = 0
                gate_state = raw[0]
                for i in range(1, len(raw)):
                    if raw[i] == gate_state:
                        continue
                    if pv[i] > lam:
                        continue
                    n_candidates += 1
                    if gated[i] != gate_state:
                        n_accepted += 1
                        gate_state = gated[i]
                accept_rate = n_accepted / n_candidates if n_candidates else float("nan")
                delayed = count_delayed_episodes(raw, gated)
                rows.append(dict(alpha=alpha, lam=lam, days=days, n_candidates=n_candidates,
                                 n_accepted=n_accepted, accept_rate=accept_rate,
                                 delayed_episodes=delayed))
    return rows


def print_calibration(rows: list[dict]) -> None:
    hdr = f"{'alpha':>6s} {'lam':>5s} {'days':>5s} {'n_cand':>7s} {'n_acc':>6s} {'acc_rate':>9s} {'delayed':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['alpha']:6.2f} {r['lam']:5.2f} {r['days']:5d} {r['n_candidates']:7d} "
             f"{r['n_accepted']:6d} {r['accept_rate']:9.3f} {r['delayed_episodes']:8d}")


# ================================================================== (4)
# Kill switches (A1: gate binds; A2: not degenerate vs v4's own vote),
# computed for EVERY (alpha, lam) config on the full pre-holdout BTC frame
# (== inner-train + inner-validation exactly, since load_btc() truncates at
# OOS_START and INNER_VAL_END+1day == OOS_START).
# ==================================================================

def kill_switch_row(btc: pd.DataFrame, alpha: float, lam: float,
                    w0_fraction: float = W0_FRACTION) -> dict:
    frac, raws, gateds, _target = saffron_components(btc, alpha, lam, w0_fraction)
    delayed = [count_delayed_episodes(r, g) for r, g in zip(raws, gateds)]
    ctrl_vote = v4_vote_frac(btc).to_numpy()
    r2 = r_squared(frac, ctrl_vote)
    a1_bind = max(delayed) >= GATE_MIN_DELAYS
    a2_ok = not (np.isfinite(r2) and r2 > R2_DEGENERACY_THRESH)
    return dict(alpha=alpha, lam=lam, delayed_by_anchor=dict(zip(V4_HORIZONS, delayed)),
               max_delayed=max(delayed), r_sq=r2, a1_bind=a1_bind, a2_ok=a2_ok,
               gate_pass=bool(a1_bind and a2_ok))


# ================================================================== (5)
# Decision rule (r160_shared.py's own PROMOTE-CANDIDATE clause, restated):
# for at least one alpha in ALPHA_GRID, on BOTH markets, on inner-val:
#   (a) paired bootstrap 95% CI on d_log_growth excludes zero on the
#       positive side (boot_lo > 0);
#   (b) d_sharpe >= SHARPE_NOISE_FLOOR OR a risk-matched drawdown
#       improvement (risk_matched and d_dd < 0);
#   (c) the SAME sign of whichever of (a)/(b) fired reproduces on the
#       eth_replication slice (not inverted).
# ==================================================================

def evaluate_decision_rule(rows: list[dict]) -> dict:
    inner_val = {r["market"]: r for r in rows if r["slice"] == "inner_val"}
    eth = {r["market"]: r for r in rows if r["slice"] == "eth_replication"}
    per_market = {}
    for market in ("spot", "futures_5x"):
        iv = inner_val.get(market)
        et = eth.get(market)
        if iv is None or et is None:
            per_market[market] = dict(clause_a=False, clause_b=False, clause_c=False,
                                      passes=False, note="missing row")
            continue
        clause_a = bool(iv["boot_lo"] > 0)
        sharpe_edge = iv["d_sharpe"] >= SHARPE_NOISE_FLOOR
        dd_edge = bool(iv["risk_matched"] and iv["d_dd"] < 0)
        clause_b = bool(sharpe_edge or dd_edge)
        # clause (c): same-sign reproduction on ETH of whichever fired.
        clause_c = False
        fired_via = None
        if clause_a and clause_b:
            if sharpe_edge:
                fired_via = "sharpe"
                clause_c = bool(et["d_sharpe"] > 0)
            elif dd_edge:
                fired_via = "dd"
                clause_c = bool(et["risk_matched"] and et["d_dd"] < 0)
        per_market[market] = dict(clause_a=clause_a, clause_b=clause_b, clause_c=clause_c,
                                  fired_via=fired_via,
                                  passes=bool(clause_a and clause_b and clause_c))
    both_markets_pass = all(per_market[m]["passes"] for m in ("spot", "futures_5x"))
    return dict(per_market=per_market, both_markets_pass=both_markets_pass)


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    hr("R-160 NOVEL: SAFFRON-gated kelly_regime_v4 -- online-FDR-controlled "
      "anchor flip timing")
    print("mechanism: each of the 3 shipped anchors' (20/40/80-day) latched flip decisions is")
    print("gated through SAFFRON (Ramdas, Zrnic, Wainwright & Jordan 2018): a flip only takes")
    print("effect once its bar's causal p-value clears an adaptively-shrinking threshold, where")
    print("wealth (and thus threshold room) is spent ONLY on bars whose p-value is <= lam (a")
    print("plausible candidate), never on bars that look like obvious true nulls -- the specific")
    print("mechanism that gives SAFFRON more power than classical LORD at the same nominal alpha.")
    print(f"\nALPHA_GRID (frozen, r160_shared) = {ALPHA_GRID}   ALPHA_PRIMARY = {ALPHA_PRIMARY}")
    print(f"LAM_GRID (this branch's own)      = {LAM_GRID}       LAM_PRIMARY   = {LAM_PRIMARY}")
    print(f"W0_FRACTION = {W0_FRACTION}")

    btc = load_btc()
    eth = load_eth()
    max_ts_seen.append(btc.index.max())
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(btc, "main(): btc")
    assert_no_holdout(eth, "main(): eth")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
         f"{btc.index[0]} -> {btc.index[-1]}")
    print(f"ETH (Bitfinex replication, truncated < {OOS_START}): {len(eth):,} bars, "
         f"{eth.index[0]} -> {eth.index[-1]}")

    # ============================================================= STEP 1
    hr("STEP 1 -- CAUSALITY: truncation probe on the full candidate build_target")
    try:
        probe_ok = causal_truncation_probe_series(BUILD_PRIMARY, btc)
        print(f"  causal_truncation_probe_series({BUILD_PRIMARY.__name__}, btc): PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  causal_truncation_probe_series({BUILD_PRIMARY.__name__}, btc): FAIL: {e}")

    # ============================================================= STEP 2
    hr("STEP 2 -- CALIBRATION SELF-TEST: synthetic ZERO-DRIFT noise, full ALPHA_GRID x LAM_GRID")
    calib_rows = calibration_self_test()
    print_calibration(calib_rows)
    n_configs += len(calib_rows)
    overshoot = [r for r in calib_rows if np.isfinite(r["accept_rate"]) and r["accept_rate"] > r["alpha"] * 1.5]
    print(f"\n{len(overshoot)}/{len(calib_rows)} (alpha,lam,anchor) cells have an empirical "
         f"acceptance rate > 1.5x their nominal alpha under pure noise.")
    if overshoot:
        print("Cells that overshoot their nominal alpha budget (reported honestly, not hidden):")
        for r in overshoot:
            print(f"    alpha={r['alpha']:.2f} lam={r['lam']:.2f} days={r['days']:3d}  "
                 f"accept_rate={r['accept_rate']:.3f}  n_candidates={r['n_candidates']}")
    else:
        print("No cell materially overshoots its nominal alpha under pure noise.")

    # ============================================================= STEP 3
    hr("STEP 3 -- KILL SWITCHES (A1 gate-binds, A2 non-degeneracy), every "
      "(alpha,lam) config, BTC full pre-holdout (== inner-train+inner-val)")
    kill_rows = []
    for alpha in ALPHA_GRID:
        for lam in LAM_GRID:
            kr = kill_switch_row(btc, alpha, lam)
            kill_rows.append(kr)
            print(f"  alpha={alpha:.2f} lam={lam:.2f}  delayed_by_anchor={kr['delayed_by_anchor']}  "
                 f"max_delayed={kr['max_delayed']:3d} (need >={GATE_MIN_DELAYS})  "
                 f"r_sq(frac,v4_vote)={kr['r_sq']:.5f} (need <{R2_DEGENERACY_THRESH})  "
                 f"A1={'PASS' if kr['a1_bind'] else 'FAIL'} A2={'PASS' if kr['a2_ok'] else 'FAIL'}")
    any_bind = any(kr["a1_bind"] for kr in kill_rows)
    any_degenerate = any(not kr["a2_ok"] for kr in kill_rows)
    print(f"\nA1 (gate binds on >=1 config, >=1 anchor, >={GATE_MIN_DELAYS} delayed episodes): "
         f"{'PASS' if any_bind else 'FAIL'}")
    print(f"A2 (no config is R^2-degenerate vs v4's own vote): "
         f"{'PASS (none degenerate)' if not any_degenerate else 'SOME CONFIGS DEGENERATE'}")

    # ============================================================= STEP 4
    hr(f"STEP 4 -- FULL SWEEP: ALPHA_GRID x LAM_GRID ({len(ALPHA_GRID)}x{len(LAM_GRID)}"
      f"={len(ALPHA_GRID) * len(LAM_GRID)} configs), compare() over inner_train/inner_val/"
      "eth_replication, SPOT+FUTURES")
    all_rows: dict[tuple, list[dict]] = {}
    decision_by_config: dict[tuple, dict] = {}
    for alpha in ALPHA_GRID:
        for lam in LAM_GRID:
            build_fn = make_build_target(alpha, lam)
            label = f"saffron_a{alpha:g}_l{lam:g}"
            print(f"\n--- alpha={alpha:.2f} lam={lam:.2f} ---")
            rows = compare(build_fn, label=label, btc=btc, eth=eth,
                          markets=(SPOT, FUTURES), include_eth=True)
            print_rows(rows)
            n_configs += len(rows)
            all_rows[(alpha, lam)] = rows
            decision = evaluate_decision_rule(rows)
            decision_by_config[(alpha, lam)] = decision
            for market, d in decision["per_market"].items():
                print(f"    decision-rule[{market}]: a={d['clause_a']} b={d['clause_b']} "
                     f"c={d['clause_c']} (fired_via={d.get('fired_via')}) -> passes={d['passes']}")
            print(f"    BOTH MARKETS PASS (this config clears PROMOTE-CANDIDATE): "
                 f"{decision['both_markets_pass']}")

    # ============================================================= STEP 5
    hr("STEP 5 -- ADDIS BONUS: one extra comparison at the primary (alpha,lam), ADDIS vs SAFFRON")
    build_addis = make_build_target(ALPHA_PRIMARY, LAM_PRIMARY, gate_fn=addis_gate, gate_name="addis")
    print(f"tau (ADDIS discard threshold) = {TAU_ADDIS}")
    addis_rows = compare(build_addis, label=f"addis_a{ALPHA_PRIMARY:g}_l{LAM_PRIMARY:g}_t{TAU_ADDIS:g}",
                         btc=btc, eth=eth, markets=(SPOT, FUTURES), include_eth=True)
    print_rows(addis_rows)
    n_configs += len(addis_rows)
    saffron_primary_rows = all_rows[(ALPHA_PRIMARY, LAM_PRIMARY)]
    print("\nADDIS vs SAFFRON at the SAME (alpha, lam) -- both vs v4 control, so compare their "
         "own d_sharpe/d_log_growth as a proxy for relative power:")
    for s_row in saffron_primary_rows:
        a_row = next((r for r in addis_rows if r["slice"] == s_row["slice"]
                     and r["market"] == s_row["market"]), None)
        if a_row is None:
            continue
        print(f"    {s_row['slice']:>16s} {s_row['market']:>11s}  "
             f"SAFFRON d_sharpe={s_row['d_sharpe']:+.4f}  ADDIS d_sharpe={a_row['d_sharpe']:+.4f}  "
             f"SAFFRON d_logG={s_row['boot_d_loggrowth']:+.4f}  "
             f"ADDIS d_logG={a_row['boot_d_loggrowth']:+.4f}")
    saffron_primary_kill = kill_switch_row(btc, ALPHA_PRIMARY, LAM_PRIMARY)
    _addis_frac, addis_raws, addis_gateds, _addis_t = saffron_components(
        btc, ALPHA_PRIMARY, LAM_PRIMARY, gate_fn=addis_gate, gate_name="addis")
    addis_delayed = [count_delayed_episodes(r, g) for r, g in zip(addis_raws, addis_gateds)]
    print(f"\nADDIS delayed-episode counts by anchor (primary alpha/lam): "
         f"{dict(zip(V4_HORIZONS, addis_delayed))}  vs SAFFRON's own: "
         f"{saffron_primary_kill['delayed_by_anchor']}")

    # ============================================================= STEP 6
    hr("STEP 6 -- CONFIGURATION COUNT")
    n_calib = len(calib_rows)
    n_sweep = sum(len(v) for v in all_rows.values())
    n_addis = len(addis_rows)
    print(f"calibration self-test cells (ALPHA_GRID x LAM_GRID x 3 anchors): {n_calib}")
    print(f"main sweep cells ({len(ALPHA_GRID)}x{len(LAM_GRID)} configs x 6 compare() cells each): "
         f"{n_sweep}")
    print(f"ADDIS bonus compare() cells: {n_addis}")
    print(f"TOTAL CONFIGURATIONS EVALUATED (this file): {n_configs}")

    # ============================================================= VERDICT
    hr("VERDICT")
    promote_configs = [(a, l) for (a, l), d in decision_by_config.items() if d["both_markets_pass"]]
    print(f"causal truncation probe PASS:                         {probe_ok}")
    print(f"A1 kill switch (gate binds, >=1 config/anchor):       {any_bind}")
    print(f"A2 kill switch (no R^2-degenerate config):            {not any_degenerate}")
    print(f"configs clearing PROMOTE-CANDIDATE (both markets, a+b+c): {promote_configs if promote_configs else 'NONE'}")
    verdict = ("PROMOTE-candidate (gate clears; holdout may be consulted)"
              if (probe_ok and any_bind and not any_degenerate and promote_configs) else "NEGATIVE")
    print(f"\nVERDICT: {verdict}")
    if verdict.startswith("PROMOTE"):
        print("\nGate clears causality + kill switches + decision rule on >=1 (alpha,lam) config --")
        print("holdout MAY be consulted per docs/ROUTINE.md step 4. This file does NOT itself")
        print("read OOS_START; that is a separate, explicit step left to the operator.")
    else:
        print("\nGate does NOT clear the full decision rule on any swept (alpha,lam) config -- per")
        print("docs/ROUTINE.md's own discipline, the holdout is precious and is NOT touched. No")
        print("bar at or after OOS_START is read anywhere in this file.")

    max_ts = max(max_ts_seen)
    print(f"\nmax timestamp read anywhere in this branch: {max_ts}  "
         f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, probe_ok=probe_ok, calib_rows=calib_rows, kill_rows=kill_rows,
               any_bind=any_bind, any_degenerate=any_degenerate, all_rows=all_rows,
               decision_by_config=decision_by_config, addis_rows=addis_rows,
               n_configs=n_configs, max_ts=max_ts, verdict=verdict,
               promote_configs=promote_configs)


if __name__ == "__main__":
    main()
