"""R-110: does the multi-asset risk-parity allocator R-107 built survive in a
form that does not throw away the cross-sectional signal's own selectivity?

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, neither branch edits it, and it does not itself compute a
verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

R-107 built the multi-asset registration path (closing B-32) and, in the
same round, tested a pure equal-risk-contribution (ERC) reweighting of R-63's
cross-sectional trend allocator. It found the mechanism's own literature-
predicted effect real (realized diversification, DR^2, rises exactly as
Baltas 2015 / Bruder & Roncalli 2012 predict) but NEGATIVE end to end: giving
ERC room to operate required widening the eligible-set cap `k` from R-63's
frozen `k=1` to `k=6`, which diffused the basket enough to gut the underlying
trend score's own selectivity (D1 -2.859 [-4.18,-1.62], D5 gross retention
only +0.061 against R-63/65/67/68's own +0.68 to +2.02). R-107's own
unfiled next step named two untried resolutions, verbatim: "does a milder
rank cap (k=2 or k=3, where the falsification margin was smaller but the
signal's own selectivity survives more intact) or a smoothed/soft
risk-parity blend (rather than a hard top-k eligibility gate feeding a hard
risk-parity solve) resolve the disclosed tension -- untested, and not
assumed promising given the falsification margin was already thin at every
k tried."

This round works BOTH of those, as the two branches:

- **Conservative** runs R-107's OWN unmodified construction (hard hard-cap
  hard-hard ERC substitution, alpha=1 in the notation below) at k=2 and k=3
  through the FULL decisive battery -- something R-107 itself never did
  (it selected one (k, lambda) cell by W_VAL growth across the whole grid
  and ran the battery only on that winner, k=6). This is not a new
  mechanism; it is the specific, named, untested read of an existing one.
- **Novel** builds a genuinely new mechanism R-107 never tried: a
  CONTINUOUS convex blend between R-63's equal-weight split and R-107's ERC
  solve, `w = (1-alpha)*w_eq + alpha*w_rp`, alpha in [0,1], rather than the
  boolean choice between them. This is motivated directly by DeMiguel,
  Garlappi & Uppal (2009), "Optimal Versus Naive Diversification," Review of
  Financial Studies 22(5), 1915-1953 -- the canonical finding that no
  optimized weighting scheme (mean-variance, minimum-variance, and by the
  same logic risk-parity) consistently beats naive 1/N out of sample because
  estimation error in the covariance input offsets the theoretical gain, and
  that BLENDING an optimized weight with 1/N (their own "shrinkage to 1/N"
  robustness check, Section VI) recovers some of the benefit while bounding
  the estimation-error cost -- exactly R-107's own diagnosed failure mode
  (giving the estimator "room to operate," i.e. weight, is what destroyed
  the candidate) restated as the thing DeMiguel et al. already have a fix
  for. Baltas (2015) and Bruder & Roncalli (2012) remain the citations for
  the ERC half of the blend; both are reused from R-107 unmodified, cited
  again here only for the blend framing, not re-derived.

Attacks: **INFO/COST** (the panel-breadth axis, same as R-107's novel
branch) for the novel branch's alpha dimension; **ERR**-adjacent
methodology (does R-107's own infrastructure, read more completely, change
the verdict) for the conservative branch -- matching R-107's own
classification of its two branches.

**Not a duplicate of:**
- R-107 conservative (`src/tradebot/multi_engine.py` etc.): pure
  infrastructure/registration, no mechanism change. Untouched here.
- R-107 novel (`experiments/r107_novel_risk_parity.py`): tested ERC
  substitution (alpha=1, in this round's notation) at k in {2,3,4,6}, lambda
  in {0,0.5,1}, SELECTED ONE CELL (k=6, lambda=1.0) BY W_VAL GROWTH, and ran
  the full decisive battery ONLY on that one cell. This round's conservative
  branch runs the SAME alpha=1 construction but reads k=2 and k=3
  individually through the full battery rather than through a single
  growth-maximizing selection across the whole grid -- a materially
  different question ("does the milder cap survive its OWN battery" vs.
  "which cell has the highest W_VAL growth"). This round's novel branch
  changes the allocation rule itself (a convex blend, never computed by
  R-107 in any form -- R-107's own docstring states explicitly that k=1
  forces alpha's two endpoints to coincide and never varies alpha at all).
- R-63/65/67/68/72: timing/eligibility axis, not touched by either R-107 or
  this round.
- Any single-asset SIZE/ERR-axis round (this operates on the 8-instrument
  panel via R-63's cross-sectional score, not `kelly_regime_v4`'s vote).

**Is it simulable here?** Yes, zero new data -- identical to R-107, through
the identical `simulate_portfolio` engine.

**What would make each branch fail (named now, before any code beyond this
shared module ran):**

  (F1, both) The falsification premise itself: does the construction (pure
       ERC at k=2/3 for conservative; the blend at its selected alpha for
       novel) still raise mean DR^2 above equal-weight's, on the IDENTICAL
       eligible-asset sequence, on W_TRAIN? R-107's own falsification test,
       reused verbatim (`falsification_dr2`), computed fresh at each
       branch's own (k, lambda, alpha) rather than inherited from R-107's
       k=6 cell.
  (F2, conservative) Even if F1 passes, R-107's own standing diagnosis
       ("no mechanism can narrow an interval") may still bind at k=2/3
       exactly as it did at k=6 -- a milder cap buys back selectivity but
       the resulting portfolio is closer to R-63's own k=1 signal, whose
       OWN unweighted five-round history (R-63/65/67/68) never cleared this
       axis's noise floor either. Fixing the "gutted selectivity" complaint
       does not by itself manufacture a promotable edge if R-63's underlying
       signal was already the binding constraint, not the allocator.
  (F3, novel) The blend could land exactly where R-107's own K_GRID sweep
       already looked: for a fixed k, a partial alpha is mathematically a
       point on the line segment between the k's own equal-weight and
       full-ERC portfolios, and if that segment is monotonic in whichever
       metric matters, an intermediate alpha can never beat BOTH endpoints
       simultaneously -- so the mechanism could be real (DeMiguel et al.'s
       own claim: blending trades off bias and variance) yet land on a
       point no better than the better endpoint, which is itself R-107's
       already-published negative or R-63/65/67/68's already-published
       negative. Watched for directly: the branch reports whether ANY
       interior alpha beats both its own endpoints (alpha=0, alpha=1) on
       W_VAL growth, not merely whether it beats one of them.

=====================================================================
WINDOWS, UNIVERSE, GATES, MACHINERY -- INHERITED UNMODIFIED FROM R-63/R-107
=====================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r107_shared import (  # noqa: E402,F401
    K_GRID,
    LAMBDA_GRID,
    MIN_DAYS,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_FREE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    WINDOW_DAYS,
    W_FULL6,
    W_HOLD,
    W_TRAIN,
    W_VAL,
    align_frames,
    build_cov_lookup,
    check_against_engine,
    check_causality,
    check_erc_converges,
    compare,
    config_count,
    diversification_ratio_sq,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    scramble_fixed_perm,
    simulate_portfolio,
    solve_erc,
    static_hold_equity,
    warm_window,
)
from experiments.r107_novel_risk_parity import (  # noqa: E402,F401
    D5_BAR_R68,
    build_cell as r107_build_cell,
    build_targets as r107_rp_targets,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    evaluate,
    frontier_row,
    further_work,
    r63_baseline_targets,
    realized_vol,
    turnover_stats,
    volmatched_hold_equity,
    write_csv,
)
from experiments.r63_novel_xsmom_rank import (  # noqa: E402,F401
    DEADBAND,
    basket_log_returns,
    conditional_vol_scale,
    cross_sectional_score,
)

OUT_DIR = ROOT / "reports" / "r110_blend"

ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
K_GRID_MILD = (2, 3)


# ---------------------------------------------------------------- targets


def build_targets(aligned: dict[str, pd.DataFrame], k: int, lam: float,
                   alpha: float, window_days: int = WINDOW_DAYS,
                   min_days: int = MIN_DAYS) -> pd.DataFrame:
    """Target weight matrix: R-63's eligibility/scale, UNMODIFIED, split
    across the eligible set as a convex blend of equal weight and
    risk-parity: ``w = (1 - alpha) * w_eq + alpha * w_rp``.

    alpha=0 reduces EXACTLY to R-63's own equal-weight construction; alpha=1
    reduces EXACTLY to R-107's own pure-ERC construction (both both-eq
    verified as identities in `checks`, not merely asserted). Both component
    weight vectors are computed over the IDENTICAL eligible support (R-63's
    unmodified `score > 0, rank < k` mask) at every bar, so the blend never
    changes WHICH assets are held, only how the already-decided total
    notional is split across them.
    """
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    n, n_assets = s.shape

    valid = np.isfinite(s)
    s_rank = np.where(valid, s, -np.inf)
    order = np.argsort(-s_rank, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order,
                       np.broadcast_to(np.arange(n_assets), (n, n_assets)), axis=1)
    sel = valid & (s > 0.0) & (rank < k)

    m = sel.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))
    desired = scale * (m / float(k))
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        d = desired[i]
        if abs(d - cur) > DEADBAND:
            cur = d
        pos[i] = cur
    total = np.minimum(pos, 1.0)

    cov_lookup = build_cov_lookup(aligned, assets, lam, window_days, min_days)
    day_key = score.index.normalize()

    w = np.zeros((n, n_assets))
    erc_cache: dict = {}
    missing_cov = 0
    last_day = last_subset = None
    last_w_blend = None
    for i in range(n):
        if total[i] <= 0.0 or m[i] == 0:
            continue
        subset = tuple(np.flatnonzero(sel[i]).tolist())
        mi = len(subset)
        day = day_key[i]
        if day == last_day and subset == last_subset:
            w_blend = last_w_blend
        else:
            w_eq = np.full(mi, 1.0 / mi)
            if alpha <= 0.0:
                w_blend = w_eq
            else:
                cov = cov_lookup.get(day)
                if cov is None:
                    missing_cov += 1
                    w_rp = w_eq
                else:
                    key = (day, subset)
                    w_rp = erc_cache.get(key)
                    if w_rp is None:
                        sub_cov = cov[np.ix_(subset, subset)]
                        w_rp = solve_erc(sub_cov)
                        erc_cache[key] = w_rp
                w_blend = (1.0 - alpha) * w_eq + alpha * w_rp
            last_day, last_subset, last_w_blend = day, subset, w_blend
        w[i, list(subset)] = total[i] * w_blend

    if missing_cov:
        print(f"    [warn] {missing_cov}/{n} bars used equal-weight fallback "
              f"(no covariance history yet)")

    return pd.DataFrame(w, index=score.index, columns=assets)


def build_cell(frames, universe, window, k, lam, alpha):
    sub = {t: frames[t] for t in universe}
    warm = align_frames(sub, warm_window(window))
    targets = build_targets(warm, k, lam, alpha)

    start = pd.Timestamp(window[0], tz="UTC")
    idx = warm[universe[0]].index
    idx = idx[idx >= start]
    if window[1] is not None:
        hi = pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)
        idx = idx[idx < hi]

    score = cross_sectional_score(warm).loc[idx]
    first_warm = bool(np.isfinite(score.iloc[0].to_numpy()).all())

    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return aligned_eval, targets.loc[idx], first_warm


# ---------------------------------------------------------------- checks


def identity_checks(frames) -> bool:
    """alpha=0 == R-63 equal-weight; alpha=1 == R-107 pure-ERC. Both must
    hold bar-for-bar before either branch trusts this module."""
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    a0 = np.nan_to_num(build_targets(warm, 3, 0.5, 0.0).to_numpy(), nan=0.0)
    b0 = np.nan_to_num(r63_baseline_targets(warm, 3).to_numpy(), nan=0.0)
    ok0 = bool(np.allclose(a0, b0, atol=1e-9, rtol=0.0))

    a1 = np.nan_to_num(build_targets(warm, 3, 0.5, 1.0).to_numpy(), nan=0.0)
    b1 = np.nan_to_num(r107_rp_targets(warm, 3, 0.5).to_numpy(), nan=0.0)
    ok1 = bool(np.allclose(a1, b1, atol=1e-9, rtol=0.0))

    print(f"  identity check alpha=0 == R-63 equal-weight (k=3): {ok0}")
    print(f"  identity check alpha=1 == R-107 pure-ERC (k=3, lam=0.5): {ok1}")
    return ok0 and ok1


def perturbation_probe(frames, universe=UNIVERSE_8, k=3, lam=0.5, alpha=0.5,
                        frac_tail=0.4) -> bool:
    warm = align_frames({t: frames[t] for t in universe}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))
    cut = int(n * (1.0 - frac_tail))

    bad = {}
    for t, df in warm.items():
        d = df.copy()
        for c in ("open", "high", "low", "close"):
            v = d[c].to_numpy(dtype=float).copy()
            v[cut:] *= 10.0
            d[c] = v
        bad[t] = d

    a = np.nan_to_num(build_targets(warm, k, lam, alpha).to_numpy()[:cut], nan=0.0)
    b = np.nan_to_num(build_targets(bad, k, lam, alpha).to_numpy()[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0))


# ---------------------------------------------------------------- falsification


def falsification_test(frames, k, lam, alpha, tag):
    """FAIL iff mean(DR^2_blend) <= mean(DR^2_equalweight) on W_TRAIN, over
    bars with >=2 eligible assets, at THIS branch's own (k, lambda, alpha)
    -- computed fresh, never inherited from R-107's k=6 cell."""
    print(f"== FALSIFICATION TEST [{tag}] (inner-train, k={k}, lambda={lam:.2f}, "
          f"alpha={alpha:.2f}) ==")
    aligned = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    bl_targets_full = build_targets(aligned, k, lam, alpha)
    eq_targets_full = r63_baseline_targets(aligned, k)

    start = pd.Timestamp(W_TRAIN[0], tz="UTC")
    hi = pd.Timestamp(W_TRAIN[1], tz="UTC") + pd.Timedelta(days=1)
    idx = aligned[UNIVERSE_8[0]].index
    idx = idx[(idx >= start) & (idx < hi)]

    bl = bl_targets_full.loc[idx]
    eq = eq_targets_full.loc[idx]
    assets = list(bl.columns)

    support_bl = bl.to_numpy(dtype=float) > 0.0
    support_eq = eq.to_numpy(dtype=float) > 0.0
    identical_support = bool(np.array_equal(support_bl, support_eq))
    print(f"  identical eligible-asset support: {identical_support}")

    cov_lookup = build_cov_lookup(aligned, assets, lam)
    day_key = idx.normalize()

    dr2_bl, dr2_eq = [], []
    bl_np = bl.to_numpy(dtype=float)
    eq_np = eq.to_numpy(dtype=float)
    n_eligible2plus = 0
    n_missing_cov = 0
    for i in range(len(idx)):
        row_bl = bl_np[i]
        if int((row_bl > 0.0).sum()) < 2:
            continue
        n_eligible2plus += 1
        cov = cov_lookup.get(day_key[i])
        if cov is None:
            n_missing_cov += 1
            continue
        dr2_bl.append(diversification_ratio_sq(row_bl, cov))
        dr2_eq.append(diversification_ratio_sq(eq_np[i], cov))

    dr2_bl = np.array(dr2_bl, dtype=float)
    dr2_eq = np.array(dr2_eq, dtype=float)
    mask = np.isfinite(dr2_bl) & np.isfinite(dr2_eq)
    n_used = int(mask.sum())
    mean_bl = float(np.mean(dr2_bl[mask])) if n_used else float("nan")
    mean_eq = float(np.mean(dr2_eq[mask])) if n_used else float("nan")
    passed = bool(n_used > 0 and mean_bl > mean_eq)

    print(f"  bars with >=2 eligible: {n_eligible2plus} ({n_missing_cov} skipped, "
          f"{n_used} used)")
    print(f"  mean DR^2  blend={mean_bl:.4f}  equal-weight={mean_eq:.4f}")
    print(f"  FALSIFICATION TEST [{tag}]: {'PASSED' if passed else 'FAILED'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"dr2_blend": dr2_bl[mask], "dr2_eq": dr2_eq[mask]}).to_csv(
        OUT_DIR / f"falsification_dr2_{tag}.csv", index=False)

    return {"k": k, "lambda": lam, "alpha": alpha, "identical_support": identical_support,
            "n_used": n_used, "mean_dr2_blend": mean_bl, "mean_dr2_eq": mean_eq,
            "passed": passed}


# ---------------------------------------------------------------- decisive battery


def decisive_battery(frames, k, lam, alpha, tag):
    """D1/D2/D3/D4/D5 + scramble, at a FROZEN (k, lambda, alpha). Mirrors
    R-107's own `cmd_run`/`cmd_scramble`, generalized over alpha and tagged
    for CSV/print disambiguation between branches sharing this module."""
    print(f"== D1-D5 cells [{tag}]: k={k} lambda={lam:.2f} alpha={alpha:.2f} (FROZEN) ==")
    rows = []

    aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, k, lam, alpha)
    if not warm_ok:
        raise RuntimeError(f"[{tag}] W_FULL6 first evaluated bar not warm")

    d1d2 = evaluate(targets, aligned, UNIVERSE_6, "W_FULL6", "U6", f"blend_{tag}",
                     {"k": k, "lambda": lam, "alpha": alpha})
    d1d2["d1_pass"] = d1_pass(d1d2)
    d1d2["d2_pass"] = d2_pass(d1d2)
    d1d2["d5_pass"] = d5_pass(d1d2)
    rows.append(d1d2)
    print(f"  [D1/D2/D5] net {d1d2['net_growth_diff']:+.4f} "
          f"[{d1d2['net_growth_lo']:+.4f}, {d1d2['net_growth_hi']:+.4f}]  "
          f"net_dd {d1d2['net_dd_diff']:+.2f} "
          f"[{d1d2['net_dd_lo']:+.2f}, {d1d2['net_dd_hi']:+.2f}]  "
          f"gross {d1d2['gross_growth_diff']:+.4f} (bar {D5_BAR_R68:+.4f})")
    print(f"    D1={d1d2['d1_pass']}  D2={d1d2['d2_pass']}  D5={d1d2['d5_pass']}")
    print(f"    mean_notional={d1d2['mean_notional']:.4f}  "
          f"turnover/day={d1d2['turnover_per_day']:.4f}")

    cand = simulate_portfolio(targets, aligned, SPOT_BASE)
    ew = static_hold_equity(aligned, UNIVERSE_6, SPOT_BASE)
    ctx_ew = compare(cand, ew)
    print(f"  [context vs EW_HOLD] growth {ctx_ew['growth_diff']:+.4f}  "
          f"final {ctx_ew['cand_final']:,.0f} vs {ctx_ew['bench_final']:,.0f}")

    cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
    ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
    d4 = compare(cand40, ew40)
    d4_ok = d4["cand_final"] > d4["bench_final"]
    print(f"  [D4 @0.40%] cand {d4['cand_final']:,.2f} vs EW_HOLD "
          f"{d4['bench_final']:,.2f} -> D4 PASS={d4_ok}")
    d1d2["d4_pass"] = d4_ok

    aligned3, targets3, warm3 = build_cell(frames, UNIVERSE_8, W_VAL, k, lam, alpha)
    if not warm3:
        raise RuntimeError(f"[{tag}] W_VAL first evaluated bar not warm")
    d3 = evaluate(targets3, aligned3, UNIVERSE_8, "W_VAL", "U8", f"blend_{tag}",
                  {"k": k, "lambda": lam, "alpha": alpha})
    d3["d3_pass"] = d3_pass(d3)
    rows.append(d3)
    print(f"  [D3] net {d3['net_growth_diff']:+.4f}  net_dd {d3['net_dd_diff']:+.2f}  "
          f"D3={d3['d3_pass']}")

    write_csv(OUT_DIR / f"cells_{tag}.csv", rows)

    # scramble
    bench, c, vol, matched = volmatched_hold_equity(cand, aligned, UNIVERSE_6, SPOT_BASE)
    real = compare(cand, bench)["growth_diff"]
    diffs = []
    srows = []
    for seed in SCRAMBLE_SEEDS:
        st = scramble_fixed_perm(targets, seed)
        eq_s = simulate_portfolio(st, aligned, SPOT_BASE)
        r = compare(eq_s, bench)
        diffs.append(r["growth_diff"])
        srows.append({"seed": seed, "growth_diff": r["growth_diff"]})
    p90 = float(np.percentile(diffs, 90))
    survived = bool(real > p90)
    print(f"  [scramble] real {real:+.4f} vs p90 {p90:+.4f} -> SURVIVED={survived}")
    write_csv(OUT_DIR / f"scramble_{tag}.csv", srows)

    fw = further_work(d1d2["d1_pass"], d1d2["d2_pass"], d3["d3_pass"],
                       d1d2["d5_pass"], survived)
    print(f"  further_work[{tag}] = {fw}")

    return {"tag": tag, "k": k, "lam": lam, "alpha": alpha,
            "d1": d1d2["d1_pass"], "d2": d1d2["d2_pass"], "d3": d3["d3_pass"],
            "d4": d4_ok, "d5": d1d2["d5_pass"], "scramble": survived,
            "further_work": fw, "d1_row": d1d2, "d3_row": d3}
