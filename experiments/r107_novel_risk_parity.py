"""R-107 NOVEL -- correlation-aware risk-parity allocation across R-63's own
eligible set, replacing five rounds' worth of unconditional 1/m equal-weight
splitting. See `r107_shared.py` for the full pre-registration: mechanism,
literature, decision rules, and the falsification test frozen BEFORE any
number below was read.

One sentence: R-63's score and positivity filter decide WHICH assets are
eligible, unmodified; this file changes only HOW MUCH of the already-decided
total notional each eligible asset gets, replacing 1/m with a causal,
rolling equal-risk-contribution (risk-parity) solve over their realized
covariance.

Run as:
    python experiments/r107_novel_risk_parity.py checks
    python experiments/r107_novel_risk_parity.py sweep
    python experiments/r107_novel_risk_parity.py falsify
    python experiments/r107_novel_risk_parity.py run
    python experiments/r107_novel_risk_parity.py scramble
    python experiments/r107_novel_risk_parity.py all
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.r107_shared import (  # noqa: E402
    D5_BAR_R68,
    K_GRID,
    LAMBDA_GRID,
    MIN_DAYS,
    OUT_DIR,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_FREE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    WARM_DAYS,
    WINDOW_DAYS,
    W_FULL6,
    W_HOLD,
    W_TRAIN,
    W_VAL,
    align_frames,
    basket_log_returns,
    build_cov_lookup,
    check_against_engine,
    check_causality,
    check_erc_converges,
    compare,
    conditional_vol_scale,
    config_count,
    cross_sectional_score,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    DEADBAND,
    diversification_ratio_sq,
    frontier_row,
    further_work,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    r63_baseline_targets,
    realized_vol,
    scramble_fixed_perm,
    solve_erc,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)

# ------------------------------------------------------------------ frozen
#
# k: swept below over K_GRID because k=1 (R-63's own frozen value) makes
# risk-parity mathematically identical to equal-weight (a single eligible
# asset gets the whole slice under either rule) -- see r107_shared's
# docstring for why sweeping k is not "re-tuning the signal". lambda: swept
# over LAMBDA_GRID, this round's disclosed departure from Ledoit-Wolf's own
# closed-form optimal intensity. Both frozen on W_VAL, selected on the
# growth_diff-vs-VOLMATCH_HOLD statistic, BEFORE this branch's falsification
# test or any W_FULL6/holdout cell was touched.
K_FROZEN = None
LAMBDA_FROZEN = None


# ------------------------------------------------------------------ signal


def build_targets(aligned: dict[str, pd.DataFrame], k: int, lam: float,
                   window_days: int = WINDOW_DAYS, min_days: int = MIN_DAYS
                   ) -> pd.DataFrame:
    """Target weight matrix: R-63's eligibility and total-notional scale,
    UNMODIFIED, with the per-asset split replaced by a causal risk-parity
    solve over the eligible set's realized covariance.

    Everything through ``total`` (the desired TOTAL notional after the 0.10
    deadband) is copied verbatim from `r63_novel_xsmom_rank.build_targets` --
    same score, same `score > 0, rank < k` eligibility, same
    `conditional_vol_scale`-driven scale, same deadband latch. Only the final
    step differs: R-63 splits `total` as `total/m` per eligible asset; this
    solves for risk-parity weights over the eligible subset's covariance
    (`solve_erc`, cached per (day, subset) since both the day's covariance
    and, usually, the subset persist across many consecutive bars) and
    splits `total` by those weights instead.
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
    total = np.minimum(pos, 1.0)  # long-only spot, unlevered -- identical to R-63

    cov_lookup = build_cov_lookup(aligned, assets, lam, window_days, min_days)
    day_key = score.index.normalize()

    w = np.zeros((n, n_assets))
    erc_cache: dict = {}
    missing_cov = 0
    last_day = last_subset = None
    last_w_rp = None
    for i in range(n):
        if total[i] <= 0.0 or m[i] == 0:
            continue
        subset = tuple(np.flatnonzero(sel[i]).tolist())
        day = day_key[i]
        if day == last_day and subset == last_subset:
            w_rp = last_w_rp
        else:
            cov = cov_lookup.get(day)
            if cov is None:
                missing_cov += 1
                w_rp = np.full(len(subset), 1.0 / len(subset))
            else:
                key = (day, subset)
                w_rp = erc_cache.get(key)
                if w_rp is None:
                    sub_cov = cov[np.ix_(subset, subset)]
                    w_rp = solve_erc(sub_cov)
                    erc_cache[key] = w_rp
            last_day, last_subset, last_w_rp = day, subset, w_rp
        w[i, list(subset)] = total[i] * w_rp

    if missing_cov:
        print(f"    [warn] {missing_cov}/{n} bars used equal-weight fallback "
              f"(no covariance history yet)")

    return pd.DataFrame(w, index=score.index, columns=assets)


# ------------------------------------------------------------------ cells


def build_cell(frames, universe, window, k, lam):
    """Aligned prices + targets, both sliced to the evaluation window --
    identical shape/contract to R-63's own `build_cell`."""
    sub = {t: frames[t] for t in universe}
    warm = align_frames(sub, warm_window(window))
    targets = build_targets(warm, k, lam)

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


def evaluate(targets, aligned, assets, window_name, universe_name, arm, params):
    """One frontier row: 0.10% and 0 bps, both against VOLMATCH_HOLD -- the
    same pattern R-65's novel branch established for a continuous-weight arm."""
    cmps = {}
    out = {}
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
    return row


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


# ------------------------------------------------------------------ checks


def perturbation_probe(frames, universe=UNIVERSE_8, k=3, lam=0.5,
                       frac_tail=0.4) -> bool:
    """Self-check: a whole-series scaler/quantile/mean/std probe, in the
    style of R-63's own `perturbation_probe`. Multiply the TAIL of every
    price series by 10 and rebuild the targets; a strictly causal
    construction cannot move the early rows. This is the check that would
    catch, specifically, a covariance or shrinkage target computed over the
    WHOLE series rather than a rolling causal window."""
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

    a = np.nan_to_num(build_targets(warm, k, lam).to_numpy()[:cut], nan=0.0)
    b = np.nan_to_num(build_targets(bad, k, lam).to_numpy()[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0))


def cmd_checks(frames):
    print("== self-checks ==")

    ok, err = check_against_engine()
    print(f"  check_against_engine (simulator vs live engine, R-63's own gate): "
          f"ok={ok} relative_final_balance_error={err:.6f}")

    erc = check_erc_converges()
    print(f"  check_erc_converges: all_pass={erc['all_pass']}")
    for name, res in erc.items():
        if name == "all_pass":
            continue
        print(f"    {name}: max risk-contribution deviation from 1/m = "
              f"{res['max_rc_dev_from_equal']:.2e}  converged={res['converged']}")

    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    causal = check_causality(lambda a: build_targets(a, 3, 0.5), warm)
    print(f"  check_causality(k=3, lambda=0.5): {causal}")
    causal2 = check_causality(lambda a: build_targets(a, 6, 1.0), warm)
    print(f"  check_causality(k=6, lambda=1.0): {causal2}")

    probe = perturbation_probe(frames)
    print(f"  perturbation_probe (tail x10, early rows unchanged): {probe}")

    # Identity check: at k=1, risk-parity is MATHEMATICALLY FORCED to equal
    # R-63's own equal-weight arm bar-for-bar (a single eligible asset gets
    # the whole slice under either rule) -- verify that literally, since it
    # is the cleanest possible sanity check on `build_targets` reducing
    # correctly to the k=1 special case.
    a1 = np.nan_to_num(build_targets(warm, 1, 0.5).to_numpy(), nan=0.0)
    b1 = np.nan_to_num(r63_baseline_targets(warm, 1).to_numpy(), nan=0.0)
    identity_k1 = bool(np.allclose(a1, b1, atol=1e-9, rtol=0.0))
    print(f"  identity check (k=1 risk-parity == R-63 equal-weight): {identity_k1}")

    return ok and erc["all_pass"] and causal and causal2 and probe and identity_k1


# ------------------------------------------------------------------ sweep


def cmd_sweep(frames):
    print("== (k, lambda) sweep: W_TRAIN then W_VAL, U8, spot 0.10% vs "
          "VOLMATCH_HOLD ==")
    rows = []
    for wname, window in (("W_TRAIN", W_TRAIN), ("W_VAL", W_VAL)):
        for k in K_GRID:
            for lam in LAMBDA_GRID:
                aligned, targets, warm_ok = build_cell(frames, UNIVERSE_8, window, k, lam)
                if not warm_ok:
                    raise RuntimeError(f"{wname} k={k} lam={lam}: first bar not warm")
                row = evaluate(targets, aligned, UNIVERSE_8, wname, "U8",
                               "risk_parity", {"k": k, "lambda": lam})
                row["first_bar_warm"] = warm_ok
                rows.append(row)
                print(f"  {wname} k={k} lam={lam:.2f}  net_growth "
                      f"{row['net_growth_diff']:+.3f} [{row['net_growth_lo']:+.3f},"
                      f" {row['net_growth_hi']:+.3f}]  net_dd {row['net_dd_diff']:+.2f}"
                      f"  mean_notional={row['mean_notional']:.3f}"
                      f"  gross_growth {row['gross_growth_diff']:+.3f}")

    write_csv(OUT_DIR / "rp_sweep.csv", rows)
    return rows


def select_frozen(sweep_rows):
    """Select (k, lambda) on W_VAL's net growth_diff vs VOLMATCH_HOLD,
    tie-broken by net_dd_diff (more negative = better) -- R-63's own
    criterion, applied here to the (k, lambda) plane instead of `k` alone."""
    val_rows = [r for r in sweep_rows if r["window"] == "W_VAL"]
    val_rows.sort(key=lambda r: (-r["net_growth_diff"], r["net_dd_diff"]))
    winner = val_rows[0]
    k = int(winner["p_k"])
    lam = float(winner["p_lambda"])
    print(f"\n  SELECTED on W_VAL: k={k} lambda={lam:.2f}  "
          f"net_growth={winner['net_growth_diff']:+.4f}  "
          f"net_dd={winner['net_dd_diff']:+.2f}")
    print("  full W_VAL ranking (plateau check):")
    for r in val_rows:
        print(f"    k={int(r['p_k'])} lam={float(r['p_lambda']):.2f}  "
              f"net_growth={r['net_growth_diff']:+.4f}  net_dd={r['net_dd_diff']:+.2f}")
    train_rows = {(int(r["p_k"]), float(r["p_lambda"])): r
                  for r in sweep_rows if r["window"] == "W_TRAIN"}
    tr = train_rows.get((k, lam))
    if tr is not None:
        print(f"  same cell on W_TRAIN: net_growth={tr['net_growth_diff']:+.4f}  "
              f"(rank transfer check)")
    return k, lam


# ------------------------------------------------------------------ falsification


def falsification_test(frames, k, lam):
    """PRE-REGISTERED, run BEFORE any W_FULL6/holdout cell. On W_TRAIN, U8,
    at the frozen (k, lambda): does risk-parity raise mean DR^2 (the
    portfolio-level analogue of Grinold breadth) relative to R-63's own
    equal-weight construction, on the IDENTICAL eligible-asset sequence?

    FAIL iff mean(DR^2_riskparity) <= mean(DR^2_equalweight) over bars with
    >= 2 eligible assets. See `r107_shared.py`'s docstring for the full
    rationale and why this is a structural, not a bootstrapped, comparison.
    """
    print(f"== FALSIFICATION TEST (inner-train, k={k}, lambda={lam:.2f}) ==")
    aligned = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    rp_targets_full = build_targets(aligned, k, lam)
    eq_targets_full = r63_baseline_targets(aligned, k)

    start = pd.Timestamp(W_TRAIN[0], tz="UTC")
    hi = pd.Timestamp(W_TRAIN[1], tz="UTC") + pd.Timedelta(days=1)
    idx = aligned[UNIVERSE_8[0]].index
    idx = idx[(idx >= start) & (idx < hi)]

    rp = rp_targets_full.loc[idx]
    eq = eq_targets_full.loc[idx]
    assets = list(rp.columns)

    support_rp = rp.to_numpy(dtype=float) > 0.0
    support_eq = eq.to_numpy(dtype=float) > 0.0
    identical_support = bool(np.array_equal(support_rp, support_eq))
    print(f"  identical eligible-asset support (RP vs equal-weight): "
          f"{identical_support}")
    if not identical_support:
        mism = int(np.any(support_rp != support_eq, axis=1).sum())
        print(f"    [WARNING] {mism}/{len(idx)} bars differ in support -- "
              f"the two constructions are not being compared on the "
              f"identical eligible sequence. Investigate before trusting "
              f"the statistic below.")

    cov_lookup = build_cov_lookup(aligned, assets, lam)
    day_key = idx.normalize()

    dr2_rp, dr2_eq = [], []
    n_eligible2plus = 0
    n_missing_cov = 0
    rp_np = rp.to_numpy(dtype=float)
    eq_np = eq.to_numpy(dtype=float)
    for i in range(len(idx)):
        row_rp = rp_np[i]
        if int((row_rp > 0.0).sum()) < 2:
            continue
        n_eligible2plus += 1
        cov = cov_lookup.get(day_key[i])
        if cov is None:
            n_missing_cov += 1
            continue
        dr2_rp.append(diversification_ratio_sq(row_rp, cov))
        dr2_eq.append(diversification_ratio_sq(eq_np[i], cov))

    dr2_rp = np.array(dr2_rp, dtype=float)
    dr2_eq = np.array(dr2_eq, dtype=float)
    mask = np.isfinite(dr2_rp) & np.isfinite(dr2_eq)
    n_used = int(mask.sum())
    mean_rp = float(np.mean(dr2_rp[mask])) if n_used else float("nan")
    mean_eq = float(np.mean(dr2_eq[mask])) if n_used else float("nan")
    median_rp = float(np.median(dr2_rp[mask])) if n_used else float("nan")
    median_eq = float(np.median(dr2_eq[mask])) if n_used else float("nan")
    frac_rp_higher = float(np.mean(dr2_rp[mask] > dr2_eq[mask])) if n_used else float("nan")

    passed = bool(n_used > 0 and mean_rp > mean_eq)

    print(f"  bars with >=2 eligible assets: {n_eligible2plus} "
          f"({n_missing_cov} lacked a covariance estimate and were skipped, "
          f"{n_used} used)")
    print(f"  mean DR^2  risk-parity={mean_rp:.4f}  equal-weight={mean_eq:.4f}  "
          f"(panel's own raw Grinold breadth, R-63: 1.47 of 8, 1.41 of 6, "
          f"for scale)")
    print(f"  median DR^2  risk-parity={median_rp:.4f}  equal-weight={median_eq:.4f}")
    print(f"  fraction of bars where risk-parity DR^2 > equal-weight DR^2: "
          f"{frac_rp_higher:.3f}")
    print(f"  FALSIFICATION TEST: {'PASSED' if passed else 'FAILED'} "
          f"(mean(DR^2_rp) {'>' if passed else '<='} mean(DR^2_eq))")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "dr2_risk_parity": dr2_rp[mask], "dr2_equal_weight": dr2_eq[mask],
    }).to_csv(OUT_DIR / "falsification_dr2.csv", index=False)

    return {
        "k": k, "lambda": lam, "identical_support": identical_support,
        "n_eligible_2plus": n_eligible2plus, "n_missing_cov": n_missing_cov,
        "n_used": n_used, "mean_dr2_rp": mean_rp, "mean_dr2_eq": mean_eq,
        "median_dr2_rp": median_rp, "median_dr2_eq": median_eq,
        "frac_rp_higher": frac_rp_higher, "passed": passed,
    }


# ------------------------------------------------------------------ run


def cmd_run(frames, k, lam):
    print(f"== D1/D2/D3/D5 cells: k={k} lambda={lam:.2f} (FROZEN) ==")
    rows = []

    aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, k, lam)
    if not warm_ok:
        raise RuntimeError("W_FULL6 first evaluated bar not warm")
    print(f"  bars {len(targets):,}  {targets.index[0]} -> {targets.index[-1]}")

    d1d2 = evaluate(targets, aligned, UNIVERSE_6, "W_FULL6", "U6",
                    "risk_parity", {"k": k, "lambda": lam})
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
          f"turnover/day={d1d2['turnover_per_day']:.4f}  "
          f"hold_days={d1d2['hold_days']:.2f}")

    # Context: EW_HOLD and MATCHED_HOLD, for continuity with R-63's own
    # published numbers even though neither is the primary D1/D2 arm.
    cand = simulate_portfolio(targets, aligned, SPOT_BASE)
    ew = static_hold_equity(aligned, UNIVERSE_6, SPOT_BASE)
    ctx_ew = compare(cand, ew)
    c_mn = mean_total_notional(targets)
    mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c_mn),
                            aligned, SPOT_BASE)
    ctx_mh = compare(cand, mh)
    print(f"  [context vs EW_HOLD] growth {ctx_ew['growth_diff']:+.4f}  "
          f"final {ctx_ew['cand_final']:,.0f} vs {ctx_ew['bench_final']:,.0f}")
    print(f"  [context vs MATCHED_HOLD, R-63's original convention] growth "
          f"{ctx_mh['growth_diff']:+.4f}  dd {ctx_mh['dd_diff']:+.2f}")

    # Exposure-matching check against R-63/R-68's own published mean
    # notional (both used k=1; frozen here may differ -- structurally
    # expected, see r107_shared docstring on why this is not a comparability
    # problem given MATCHED_HOLD/VOLMATCH_HOLD are always computed from the
    # CANDIDATE'S OWN realized notional/vol).
    r63_mean_notional = 0.525  # published, R-63/R-65/R-67/R-68's k=1 D1 cell
    print(f"  [exposure check] this arm's mean total notional={c_mn:.4f} vs "
          f"R-63/R-65/R-67/R-68's own k=1 figure {r63_mean_notional:.3f} "
          f"(difference expected: different k changes m/k; not a leverage bug)")

    # D4: 0.40% taker, candidate final balance vs EW_HOLD.
    cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
    ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
    d4 = compare(cand40, ew40)
    d4_ok = d4["cand_final"] > d4["bench_final"]
    print(f"  [D4 @0.40%] cand {d4['cand_final']:,.2f} vs EW_HOLD "
          f"{d4['bench_final']:,.2f} -> D4 PASS={d4_ok}")
    d1d2["d4_pass"] = d4_ok
    d1d2["d4_cand_final"] = d4["cand_final"]
    d1d2["d4_bench_final"] = d4["bench_final"]

    # D3: W_VAL, U8, spot 0.10%, directional gate vs VOLMATCH_HOLD.
    aligned3, targets3, warm3 = build_cell(frames, UNIVERSE_8, W_VAL, k, lam)
    if not warm3:
        raise RuntimeError("W_VAL first evaluated bar not warm")
    d3 = evaluate(targets3, aligned3, UNIVERSE_8, "W_VAL", "U8",
                 "risk_parity", {"k": k, "lambda": lam})
    d3["d3_pass"] = d3_pass(d3)
    rows.append(d3)
    print(f"  [D3] net {d3['net_growth_diff']:+.4f}  net_dd {d3['net_dd_diff']:+.2f}  "
          f"D3={d3['d3_pass']}")

    write_csv(OUT_DIR / "rp_cells.csv", rows)
    return {"d1": d1d2["d1_pass"], "d2": d1d2["d2_pass"], "d3": d3["d3_pass"],
            "d4": d4_ok, "d5": d1d2["d5_pass"], "d1_row": d1d2,
            "targets": targets, "aligned": aligned, "k": k, "lam": lam}


# ------------------------------------------------------------------ scramble


def cmd_scramble(frames, k, lam, state=None):
    print(f"== FALSIFICATION (D-battery): fixed-permutation scramble, "
          f"seeds 0..9, D1 cell, k={k} lambda={lam:.2f} ==")
    if state is None:
        aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, k, lam)
        if not warm_ok:
            raise RuntimeError("W_FULL6 first evaluated bar not warm")
        cand = simulate_portfolio(targets, aligned, SPOT_BASE)
        bench, c, vol, matched = volmatched_hold_equity(cand, aligned, UNIVERSE_6,
                                                         SPOT_BASE)
        real = compare(cand, bench)["growth_diff"]
    else:
        aligned, targets = state["aligned"], state["targets"]
        real = state["d1_row"]["net_growth_diff"]
        cand = simulate_portfolio(targets, aligned, SPOT_BASE)
        bench, c, vol, matched = volmatched_hold_equity(cand, aligned, UNIVERSE_6,
                                                         SPOT_BASE)

    rows, diffs = [], []
    for seed in SCRAMBLE_SEEDS:
        st = scramble_fixed_perm(targets, seed)
        eq = simulate_portfolio(st, aligned, SPOT_BASE)
        r = compare(eq, bench)
        diffs.append(r["growth_diff"])
        rows.append({"arm": "risk_parity_scrambled", "seed": seed, "p_k": k,
                     "p_lambda": lam, "window": "W_FULL6", "universe": "U6",
                     "fee": 0.001, "bench": "VOLMATCH_HOLD",
                     "mean_notional": mean_total_notional(st),
                     "turnover_per_day": turnover_stats(st)["turnover_per_day"],
                     **{key: r[key] for key in
                        ("cand_final", "bench_final", "cand_dd", "bench_dd",
                         "growth_diff", "growth_lo", "growth_hi",
                         "dd_diff", "dd_lo", "dd_hi", "n_days")}})
        print(f"  seed {seed}: growth_diff {r['growth_diff']:+.4f}  "
              f"final {r['cand_final']:>12,.2f}  dd {r['cand_dd']:5.1f}%")

    p90 = float(np.percentile(diffs, 90))
    survived = bool(real > p90)
    rows.append({"arm": "risk_parity", "seed": -1, "p_k": k, "p_lambda": lam,
                 "window": "W_FULL6", "universe": "U6", "fee": 0.001,
                 "bench": "VOLMATCH_HOLD", "growth_diff": real,
                 "mean_notional": mean_total_notional(targets),
                 "turnover_per_day": turnover_stats(targets)["turnover_per_day"],
                 "scramble_p90": p90, "scramble_survived": survived})
    print(f"  real growth_diff {real:+.4f} vs scramble p90 {p90:+.4f} -> "
          f"SURVIVED={survived}")

    write_csv(OUT_DIR / "rp_scramble.csv", rows)
    return survived


# ------------------------------------------------------------------ main


def main():
    global K_FROZEN, LAMBDA_FROZEN
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "sweep", "falsify", "run",
                                    "scramble", "all"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--lam", type=float, default=None)
    args = ap.parse_args()

    frames = load_universe(UNIVERSE_8)
    start_configs = config_count()

    if args.cmd == "checks":
        ok = cmd_checks(frames)
        print(f"\nall self-checks pass: {ok}")
    elif args.cmd == "sweep":
        cmd_sweep(frames)
    elif args.cmd == "falsify":
        rows = cmd_sweep(frames)
        k, lam = select_frozen(rows)
        falsification_test(frames, k, lam)
    elif args.cmd == "run":
        k = args.k if args.k is not None else K_FROZEN
        lam = args.lam if args.lam is not None else LAMBDA_FROZEN
        if k is None or lam is None:
            raise SystemExit("(k, lambda) not frozen: pass --k/--lam or run `all`")
        cmd_run(frames, k, lam)
    elif args.cmd == "scramble":
        k = args.k if args.k is not None else K_FROZEN
        lam = args.lam if args.lam is not None else LAMBDA_FROZEN
        if k is None or lam is None:
            raise SystemExit("(k, lambda) not frozen: pass --k/--lam or run `all`")
        cmd_scramble(frames, k, lam)
    else:  # all
        ok = cmd_checks(frames)
        print(f"\nall self-checks pass: {ok}\n")
        if not ok:
            raise SystemExit("self-checks failed -- stop before spending any read")

        sweep_rows = cmd_sweep(frames)
        k, lam = select_frozen(sweep_rows)

        print()
        fal = falsification_test(frames, k, lam)
        if not fal["passed"]:
            print("\n== FALSIFICATION TEST FAILED on inner-train. STOP. ==")
            print("   No W_FULL6 read, no W_HOLD read. Reporting the negative.")
            print(f"\nconfig_count() this run = {config_count() - start_configs}  "
                  f"(cumulative process total = {config_count()})")
            return

        print("\n== falsification test PASSED. Proceeding to the decisive "
              "battery on W_FULL6/W_VAL. ==\n")
        st = cmd_run(frames, k, lam)
        surv = cmd_scramble(frames, st["k"], st["lam"], st)
        fw = further_work(st["d1"], st["d2"], st["d3"], st["d5"], surv)
        print(f"\n== further_work(d1={st['d1']}, d2={st['d2']}, d3={st['d3']}, "
              f"d5={st['d5']}, scramble={surv}) = {fw} ==")
        if fw:
            print("  -> STOP. Report to the operator; the W_HOLD read is theirs "
                  "to authorize (+1 holdout consultation).")
        else:
            print("  -> DONE. W_HOLD is NOT read.")

    print(f"\nconfig_count() this run = {config_count() - start_configs}  "
          f"(cumulative process total = {config_count()})")


if __name__ == "__main__":
    main()
