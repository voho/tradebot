#!/usr/bin/env python
"""R-147 CONSERVATIVE branch: `JamesSteinAnchorKellyV4` -- `kelly_regime_v4`'s
fixed, unweighted `frac = (vote_20d + vote_40d + vote_80d) / 3` replaced by a
classical (known-variance) James-Stein-shrunk WEIGHTED average of the SAME
three anchors' own unmodified 0/1 latched votes. Direction, citations, and
the non-duplication argument all live in `experiments/r147_shared.py`'s
module docstring (read there first -- this file does not repeat that
reasoning and does not edit that module, which is frozen/read-only).

The candidate, exactly (mirroring `experiments/r147_shared.py`'s mechanism
section verbatim):

    for each anchor k in (20, 40, 80) days:
        vote_k(t)   = vote_frac(df, (k,), V4_BAND)          # v4's own 0/1 vote, unmodified
        hit_k(t)    = spell_hit_series(latched_state(vote_k), close)
        theta_k(t)  = rolling_reliability(hit_k, m)          # causal rolling hit-rate, m=JS_M_PRIMARY=8

    theta_bar(t) = mean_k(theta_k(t))
    S(t)         = sum_k((theta_k(t) - theta_bar(t))**2)
    sigma2(t)    = max(theta_bar(t)*(1-theta_bar(t)), 1e-9) / m
    c(t)         = 1.0                                       if S(t) <= 1e-12
                 = clip((k-2)*sigma2(t)/S(t), 0, 1)           otherwise   (k=3)
    theta_JS_k(t)= theta_bar(t) + (1 - c(t)) * (theta_k(t) - theta_bar(t))
    weights(t)   = normalize_weights(theta_JS(t))             # (n_bars, 3), NaN/warmup -> equal

    frac_conservative(t)   = sum_k(weights[t,k] * vote_k(t))
    target_conservative(t) = build_target_from_frac(frac_conservative, df)

Everything else -- each anchor's own point-estimate construction (rolling
mean, 1% band, ffill-hysteresis), v4's conditional-volatility `scale` state
machine, and the 10% re-target deadband -- is `kelly_regime_v4`'s own,
unmodified construction (reused directly from `r147_shared.py`, never
re-derived here). `vote_gamma` stays 1.0 throughout (never revisits
`kelly_regime_v2`, already closed).

======================================================================
HEADLINE RESULT, stated before the detail: NEGATIVE. Kill switches PASS --
A1 bind_frac(weights, BTC inner-train) = 0.6746 (>> the 0.01 threshold) and
A2 R^2(candidate frac, v4_vote_frac) = 0.9676 (< the 0.98 ceiling) -- so,
unlike several prior ERR-axis rounds, this branch is not degenerate by
construction; the James-Stein weights genuinely move and genuinely differ
from v4's own vote. But the promotion bar's Sharpe leg (clause 1) never
clears on all four required cells: primary (m=8) BTC dSharpe is
inner-train spot -0.001, inner-train futures +0.041, inner-val spot +0.015,
inner-val futures -0.303 -- every bootstrap CI for total-log-growth
difference includes zero. ETH replication (clause 2) agrees in sign with
BTC inner-val on futures (-0.170 vs -0.303) but NOT on spot (-0.073 vs
+0.015) -- fails the both-markets bar, though it is worth flagging that
the one market where BTC's own primary-cell sign IS negative (futures) is
exactly the market where ETH replicates it: this is "both underperform,"
not a replicated edge, the same caveat R-146's median-anchor branch
recorded. The 0.40% fee tier (clause 3) preserves sign on both markets
(moot, since there is no positive edge at 0.10% for it to protect). The
JS_M_GRID plateau (clause 4) is same-sign on spot (+0.015..+0.059, all
inside the noise floor) but flips sign on futures across the grid
(-0.303 at m=8 vs +0.014 at m=32) -- an isolated-direction result, not a
plateau. Verdict: NEGATIVE. The holdout (>= 2023-01-01) was NOT read --
clause 1 (the Sharpe leg) failed on inner-train and inner-validation on
both markets, so per this project's discipline the holdout was not
touched.
======================================================================

Run: `. .venv/bin/activate && python experiments/r147_conservative_james_stein.py`
(from the repo root).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r147_shared import (  # noqa: E402
    BIND_FRAC_THRESH,
    FEE_TIER,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    JS_M_GRID,
    JS_M_PRIMARY,
    OOS_START,
    R2_DEGENERACY_THRESH,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TargetStrategy,
    V4_BAND,
    V4_HORIZONS,
    assert_no_holdout,
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
    rolling_reliability,
    run_slice,
    spell_hit_series,
    v4_scale,
    v4_target,
    v4_vote_frac,
    vote_frac,
)

# Sanity-check import only (r102_shared, NOT r147_shared -- read-only,
# never edited): used once below to verify vote_frac(df, (days,), band)
# with a single-element horizon tuple really does reproduce one anchor's
# own latched vote directly, before trusting it anywhere else in this file.
from experiments.r102_shared import _latched_anchor_vote  # noqa: E402

K_ANCHORS = len(V4_HORIZONS)
assert K_ANCHORS == 3, V4_HORIZONS


# ================================================================== core
# James-Stein combination machinery. Every function here is a pure
# `np.ndarray`/`pd.DataFrame` transform, built only out of r147_shared's
# frozen primitives (`vote_frac`, `latched_state`, `spell_hit_series`,
# `rolling_reliability`, `normalize_weights`, `build_target_from_frac`).
# ==================================================================

def anchor_vote_matrix(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """(n_bars, k) matrix of each anchor's own unmodified 0/1 latched vote."""
    return np.column_stack([vote_frac(df, (days,), band).to_numpy() for days in horizons])


def anchor_theta_matrix(df: pd.DataFrame, m: int, horizons: tuple[int, ...] = V4_HORIZONS,
                        band: float = V4_BAND) -> np.ndarray:
    """(n_bars, k) matrix of each anchor's own causal rolling reliability
    theta_k(t), window m spells, built via the shared spell/hit primitives."""
    close = df["close"]
    cols = []
    for days in horizons:
        vote = vote_frac(df, (days,), band)
        hit = spell_hit_series(latched_state(vote), close)
        cols.append(rolling_reliability(hit, m).to_numpy())
    return np.column_stack(cols)


def james_stein_weights_from_theta(theta: np.ndarray, m: int) -> np.ndarray:
    """Classical KNOWN-variance James-Stein shrinkage of a (n_bars, k)
    reliability matrix into a (n_bars, k) weight matrix summing to 1 per
    row. k=3 uses the `(k-2)` textbook form (valid since k-2=1>0), a plug-in
    (not jointly estimated) binomial variance sigma2 = theta_bar*(1-theta_bar)/m,
    and c(t)=1 (no-op) wherever S(t) <= 1e-12 (anchors already agree). Any
    NaN input row (warmup) propagates to a NaN output row, which
    `normalize_weights` falls back to equal-weight for."""
    theta = np.asarray(theta, dtype=float)
    n, k = theta.shape
    assert k == K_ANCHORS, k
    theta_bar = np.mean(theta, axis=1)
    diff = theta - theta_bar[:, None]
    S = np.sum(diff ** 2, axis=1)
    sigma2 = np.maximum(theta_bar * (1.0 - theta_bar), 1e-9) / m
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_c = (k - 2) * sigma2 / S
    c = np.where(S <= 1e-12, 1.0, np.clip(raw_c, 0.0, 1.0))
    theta_js = theta_bar[:, None] + (1.0 - c)[:, None] * diff
    return normalize_weights(theta_js)


def js_frac(df: pd.DataFrame, m: int = JS_M_PRIMARY, horizons: tuple[int, ...] = V4_HORIZONS,
           band: float = V4_BAND) -> np.ndarray:
    """The candidate's directional vote fraction: James-Stein-weighted
    combination of the k anchors' own unmodified 0/1 votes."""
    theta = anchor_theta_matrix(df, m, horizons, band)
    weights = james_stein_weights_from_theta(theta, m)
    votes = anchor_vote_matrix(df, horizons, band)
    return np.sum(weights * votes, axis=1)


def make_build_target(m: int = JS_M_PRIMARY, horizons: tuple[int, ...] = V4_HORIZONS,
                      band: float = V4_BAND):
    """Candidate `build_target(df) -> np.ndarray`: v4's own scale and
    deadband, unchanged, fed the James-Stein-weighted vote fraction instead
    of v4's fixed 1/3-each average. Pure function of `df`."""

    def build_target(df: pd.DataFrame) -> np.ndarray:
        frac = js_frac(df, m, horizons, band)
        return build_target_from_frac(frac, df)

    build_target.__name__ = f"conservative_js_m{m}"
    return build_target


build_target = make_build_target(JS_M_PRIMARY)  # frozen primary candidate (m=8)


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def cell(rows: list[dict], label: str, slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["label"] == label and r["slice"] == slice_name and r["market"] == market:
            return r
    return None


def inner_val_rows(build_fn, label: str, btc: pd.DataFrame,
                   markets: tuple = (SPOT, FUTURES)) -> list[dict]:
    """Lightweight BTC-inner-validation-only comparison, for the fee-tier
    re-run (which does not need the full inner-train + ETH overhead)."""
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r147_{label}")
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
            label=label, slice="inner_val", market=market.name,
            d_sharpe=a.sharpe - b.sharpe, d_dd=a.max_drawdown_pct - b.max_drawdown_pct,
            exposure_ratio=exp_ratio, vol_ratio=vol_ratio, risk_matched=risk_matched,
            d_log_growth=a.log_growth - b.log_growth,
            boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
            excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0),
        ))
    return rows


CONFIG_CELLS = 0  # running count of every distinct (m, market, slice) cell evaluated


def main() -> None:
    global CONFIG_CELLS
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-147 CONSERVATIVE -- JamesSteinAnchorKellyV4: kelly_regime_v4's "
       "fixed 1/3-each anchor-vote average\nreplaced by a classical "
       "known-variance James-Stein-shrunk reliability-weighted average. "
       "Default verdict: NEGATIVE.")
    print("\nMECHANISM: James & Stein (1961) -- for k>=3 simultaneously "
          "estimated means, shrinking each\ntoward their common mean "
          "strictly dominates the individual (equal-trust) estimator in "
          "total\nquadratic risk. Applied here to the 3 anchors' own "
          "causal rolling hit-rates theta_20/40/80(t),\nreweighting their "
          "vote combination in proportion to how reliable each has "
          "actually been.")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "BTC full")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    btc_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    assert_no_holdout(btc_train, "BTC inner-train")
    print(f"BTC inner-train slice: {len(btc_train):,} bars, "
          f"{btc_train.index[0]} -> {btc_train.index[-1]}")

    # ========================================================== STEP 1 (a)
    hr("STEP 1a -- single-anchor `vote_frac` sanity check: vote_frac(df, "
       "(days,), band) must equal\n`_latched_anchor_vote(close, days, "
       "band)` bit-for-bit, for every anchor, before trusting it further")
    ok_single = True
    for days in V4_HORIZONS:
        a = vote_frac(btc_train, (days,), V4_BAND).to_numpy()
        b = _latched_anchor_vote(btc_train["close"], days, V4_BAND).to_numpy()
        same = np.array_equal(a, b)
        uniq = sorted(set(np.unique(a).tolist()))
        print(f"    days={days:3d}: vote_frac((days,)) == _latched_anchor_vote: "
              f"{same}   unique values={uniq}")
        ok_single = ok_single and same and set(uniq) <= {0.0, 1.0}
    print(f"\n    Single-anchor reproduction check: {'PASS' if ok_single else 'FAIL'}")
    if not ok_single:
        raise AssertionError("vote_frac((days,), band) does not reproduce the anchor's "
                             "own latched vote -- stopping before any further number.")

    # ========================================================== STEP 1 (b)
    hr("STEP 1b -- causal truncation probe (real BTC inner-train data)")
    causal_ok = True
    try:
        ok1 = causal_truncation_probe_series(build_target, btc_train)
        print(f"    causal_truncation_probe_series(build_target [m={JS_M_PRIMARY}], "
              f"btc_train): {'PASS' if ok1 else 'FAIL'}")
    except AssertionError as e:
        ok1 = False
        print(f"    causal_truncation_probe_series(build_target, btc_train): FAIL ({e})")
    causal_ok = causal_ok and ok1

    def _theta_builder(d: pd.DataFrame) -> np.ndarray:
        return anchor_theta_matrix(d, JS_M_PRIMARY)

    try:
        ok2 = causal_truncation_probe_series(_theta_builder, btc_train)
        print(f"    causal_truncation_probe_series(anchor_theta_matrix, btc_train): "
              f"{'PASS' if ok2 else 'FAIL'}")
    except AssertionError as e:
        ok2 = False
        print(f"    causal_truncation_probe_series(anchor_theta_matrix, btc_train): "
              f"FAIL ({e})")
    causal_ok = causal_ok and ok2

    print(f"\n    Causality: {'PASS' if causal_ok else 'FAIL'}")
    if not causal_ok:
        raise AssertionError("Causal truncation probe FAILED -- stopping before "
                             "any promotion-bar evaluation.")

    # ========================================================== STEP 1 (c)
    hr("STEP 1c -- degenerate-case self-check: theta_k(t) forced EQUAL "
       "across all 3 anchors\n-> weights must be exactly equal (1/3 each) "
       "-> frac must equal v4_vote_frac -> build_target_from_frac must "
       "equal v4_target, BIT-FOR-BIT")
    votes_train = anchor_vote_matrix(btc_train)
    n_train = len(btc_train)
    rng = np.random.default_rng(147)
    # Forced-equal theta: same value across the k columns, varying by row
    # (so S(t)=0 exactly at every bar, the "anchors already agree" case).
    theta_equal_col = rng.uniform(0.40, 0.60, size=n_train)
    theta_equal = np.tile(theta_equal_col[:, None], (1, K_ANCHORS))
    weights_equal = james_stein_weights_from_theta(theta_equal, JS_M_PRIMARY)
    equal_weights_ok = bool(np.allclose(weights_equal, 1.0 / K_ANCHORS))
    print(f"    weights == 1/{K_ANCHORS} everywhere when theta forced equal: "
          f"{equal_weights_ok}")

    frac_equal = np.sum(weights_equal * votes_train, axis=1)
    v4_frac_train = v4_vote_frac(btc_train).to_numpy()
    frac_matches = bool(np.allclose(frac_equal, v4_frac_train, atol=1e-12, rtol=0.0))
    print(f"    frac_conservative(theta forced equal) == v4_vote_frac, "
          f"bit-for-bit: {frac_matches}")

    target_equal = build_target_from_frac(frac_equal, btc_train)
    target_v4 = v4_target(btc_train)
    target_matches = bool(np.allclose(target_equal, target_v4, atol=1e-12, rtol=0.0))
    print(f"    build_target_from_frac(frac_equal) == v4_target(btc_train), "
          f"bit-for-bit: {target_matches}")

    degenerate_ok = equal_weights_ok and frac_matches and target_matches
    print(f"\n    Degenerate-case reduction: {'PASS' if degenerate_ok else 'FAIL'}")
    if not degenerate_ok:
        raise AssertionError("Degenerate (all-equal-theta) case does not reduce to "
                             "v4_target bit-for-bit -- stopping.")

    # ========================================================== STEP 2
    hr("STEP 2 -- Step-0 kill switches on BTC inner-train (primary m="
       f"{JS_M_PRIMARY})")
    theta_train = anchor_theta_matrix(btc_train, JS_M_PRIMARY)
    weights_train = james_stein_weights_from_theta(theta_train, JS_M_PRIMARY)
    bf = bind_frac(weights_train)
    a1_pass = bf > BIND_FRAC_THRESH
    print(f"    A1 bind_frac(weights, inner-train) = {bf:.4f}   "
          f"threshold (must exceed) = {BIND_FRAC_THRESH}   "
          f"-> {'PASS' if a1_pass else 'FAIL (TRIPPED)'}")

    frac_train = np.sum(weights_train * votes_train, axis=1)
    r2 = r_squared(frac_train, v4_frac_train)
    a2_pass = r2 < R2_DEGENERACY_THRESH
    print(f"    A2 R^2(candidate frac, v4_vote_frac, inner-train) = {r2:.4f}   "
          f"ceiling (must stay below) = {R2_DEGENERACY_THRESH}   "
          f"-> {'PASS' if a2_pass else 'FAIL (TRIPPED)'}")

    step0_pass = a1_pass and a2_pass
    print(f"\n    Step-0 kill switches: {'PASS (proceeding)' if step0_pass else 'TRIPPED -- STOP, NEGATIVE'}")
    if not step0_pass:
        hr("VERDICT")
        print("    VERDICT: NEGATIVE (stopped at Step 0 -- kill switch tripped, "
              "per the pre-registered rule)")
        print(f"    A1 bind_frac={bf:.4f} (need > {BIND_FRAC_THRESH}): "
              f"{'ok' if a1_pass else 'TRIPPED'}")
        print(f"    A2 R^2={r2:.4f} (need < {R2_DEGENERACY_THRESH}): "
              f"{'ok' if a2_pass else 'TRIPPED'}")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    # ========================================================== STEP 3
    hr(f"STEP 3 -- full comparison: compare(build_target [m={JS_M_PRIMARY}], "
       "label='conservative_js_m8')\nBTC inner-train / inner-val, ETH "
       "replication, SPOT + FUTURES_5x")
    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "ETH full")

    primary_rows = compare(build_target, label="conservative_js_m8", btc=btc, eth=eth)
    CONFIG_CELLS += len(primary_rows)
    print()
    print_rows(primary_rows)

    btc_train_s = cell(primary_rows, "conservative_js_m8", "inner_train", SPOT.name)
    btc_train_f = cell(primary_rows, "conservative_js_m8", "inner_train", FUTURES.name)
    btc_val_s = cell(primary_rows, "conservative_js_m8", "inner_val", SPOT.name)
    btc_val_f = cell(primary_rows, "conservative_js_m8", "inner_val", FUTURES.name)
    eth_s = cell(primary_rows, "conservative_js_m8", "eth_replication", SPOT.name)
    eth_f = cell(primary_rows, "conservative_js_m8", "eth_replication", FUTURES.name)

    def leg_ok(c: dict) -> bool:
        return (c["d_sharpe"] > SHARPE_NOISE_FLOOR) or (c["excludes_zero"] and c["boot_d_loggrowth"] > 0)

    print("\n    Sharpe-leg (clause 1) readout, primary cell (m=8):")
    for name, c in (("BTC inner-train / spot", btc_train_s),
                    ("BTC inner-train / futures_5x", btc_train_f),
                    ("BTC inner-val / spot", btc_val_s),
                    ("BTC inner-val / futures_5x", btc_val_f)):
        ok = leg_ok(c)
        print(f"      {name:32s} dSharpe={c['d_sharpe']:+.3f}  "
              f"boot=[{c['boot_lo']:+.3f},{c['boot_hi']:+.3f}]  "
              f"excludes_zero={c['excludes_zero']}  "
              f"exposure_ratio={c['exposure_ratio']:.3f}  vol_ratio={c['vol_ratio']:.3f}  "
              f"risk_matched={c['risk_matched']}  -> {'PASS' if ok else 'fail'}")

    clause1_all4 = all(leg_ok(c) for c in (btc_train_s, btc_train_f, btc_val_s, btc_val_f))
    print(f"\n    Clause 1 (ALL FOUR of BTC train+val x spot+futures clear "
          f"+/-{SHARPE_NOISE_FLOOR} Sharpe or bootstrap-excludes-zero on the "
          f"winning side): {'PASS' if clause1_all4 else 'FAIL'}")

    # ========================================================== STEP 4
    hr("STEP 4 -- falsification test: ETH replication SAME SIGN as BTC "
       "inner-validation (pre-registered, clause 2)")
    same_spot = bool(np.sign(eth_s["d_sharpe"]) == np.sign(btc_val_s["d_sharpe"]))
    same_fut = bool(np.sign(eth_f["d_sharpe"]) == np.sign(btc_val_f["d_sharpe"]))
    print(f"    spot:     BTC inner-val dSharpe={btc_val_s['d_sharpe']:+.3f}  "
          f"(boot point={btc_val_s['boot_d_loggrowth']:+.4f})   "
          f"ETH dSharpe={eth_s['d_sharpe']:+.3f}  "
          f"(boot point={eth_s['boot_d_loggrowth']:+.4f})   same sign: {same_spot}")
    print(f"    futures:  BTC inner-val dSharpe={btc_val_f['d_sharpe']:+.3f}  "
          f"(boot point={btc_val_f['boot_d_loggrowth']:+.4f})   "
          f"ETH dSharpe={eth_f['d_sharpe']:+.3f}  "
          f"(boot point={eth_f['boot_d_loggrowth']:+.4f})   same sign: {same_fut}")
    clause2_both = same_spot and same_fut
    clause2_either = same_spot or same_fut
    print(f"\n    Clause 2 (ETH same sign as BTC, BOTH markets): "
          f"{'PASS' if clause2_both else 'FAIL'}")
    print(f"    Clause 2 (ETH same sign as BTC, >=1 market, looser reading): "
          f"{'PASS' if clause2_either else 'FAIL'}")

    # ========================================================== STEP 5
    hr(f"STEP 5 -- fee robustness: BTC inner-validation re-run at "
       f"FEE_TIER={FEE_TIER:.2%} (clause 3)")
    fee_spot = fee_at(SPOT, FEE_TIER)
    fee_fut = fee_at(FUTURES, FEE_TIER)
    fee_rows = inner_val_rows(build_target, "conservative_js_m8_fee40", btc,
                              markets=(fee_spot, fee_fut))
    CONFIG_CELLS += len(fee_rows)
    fee_s = next(r for r in fee_rows if r["market"] == fee_spot.name)
    fee_f = next(r for r in fee_rows if r["market"] == fee_fut.name)
    sign_spot_ok = bool(np.sign(fee_s["d_sharpe"]) == np.sign(btc_val_s["d_sharpe"])
                       or (fee_s["d_sharpe"] == 0 and btc_val_s["d_sharpe"] == 0))
    sign_fut_ok = bool(np.sign(fee_f["d_sharpe"]) == np.sign(btc_val_f["d_sharpe"])
                       or (fee_f["d_sharpe"] == 0 and btc_val_f["d_sharpe"] == 0))
    print(f"    spot:     @0.10% dSharpe={btc_val_s['d_sharpe']:+.3f}   "
          f"@0.40% dSharpe={fee_s['d_sharpe']:+.3f}   sign preserved: {sign_spot_ok}")
    print(f"    futures:  @0.10% dSharpe={btc_val_f['d_sharpe']:+.3f}   "
          f"@0.40% dSharpe={fee_f['d_sharpe']:+.3f}   sign preserved: {sign_fut_ok}")
    clause3_both = sign_spot_ok and sign_fut_ok
    print(f"\n    Clause 3 (0.40% fee, no sign reversal, BOTH markets): "
          f"{'PASS' if clause3_both else 'FAIL'}")

    # ========================================================== STEP 6
    hr(f"STEP 6 -- plateau check (B4): JS_M_GRID={JS_M_GRID} sweep, BTC "
       "inner-validation (clause 4)")
    plateau_rows: dict[int, list[dict]] = {JS_M_PRIMARY: [btc_val_s, btc_val_f]}
    for m in JS_M_GRID:
        if m == JS_M_PRIMARY:
            continue
        rows_m = compare(make_build_target(m), label=f"conservative_js_m{m}",
                         btc=btc, eth=None, include_eth=False)
        CONFIG_CELLS += len(rows_m)
        s = cell(rows_m, f"conservative_js_m{m}", "inner_val", SPOT.name)
        f = cell(rows_m, f"conservative_js_m{m}", "inner_val", FUTURES.name)
        plateau_rows[m] = [s, f]

    print(f"\n    {'m':>4s} {'spot dSharpe':>14s} {'spot boot[lo,hi]':>22s} "
          f"{'fut dSharpe':>14s} {'fut boot[lo,hi]':>22s}")
    for m in JS_M_GRID:
        s, f = plateau_rows[m]
        tag = " <- primary" if m == JS_M_PRIMARY else ""
        print(f"    {m:>4d} {s['d_sharpe']:>+14.3f} "
              f"[{s['boot_lo']:>+7.3f},{s['boot_hi']:>+7.3f}]      "
              f"{f['d_sharpe']:>+14.3f} [{f['boot_lo']:>+7.3f},{f['boot_hi']:>+7.3f}]{tag}")

    spot_signs = {np.sign(plateau_rows[m][0]["d_sharpe"]) for m in JS_M_GRID}
    fut_signs = {np.sign(plateau_rows[m][1]["d_sharpe"]) for m in JS_M_GRID}
    spot_dsharpes = [plateau_rows[m][0]["d_sharpe"] for m in JS_M_GRID]
    fut_dsharpes = [plateau_rows[m][1]["d_sharpe"] for m in JS_M_GRID]
    plateau_same_sign = len(spot_signs) == 1 and len(fut_signs) == 1
    plateau_any_pass_all_m = (all(d > SHARPE_NOISE_FLOOR for d in spot_dsharpes)
                              or all(d > SHARPE_NOISE_FLOOR for d in fut_dsharpes))
    print(f"\n    Sign of dSharpe constant across the full JS_M_GRID: "
          f"spot={len(spot_signs) == 1} (values {sorted(spot_signs)})  "
          f"futures={len(fut_signs) == 1} (values {sorted(fut_signs)})")
    print(f"    spot dSharpe range [{min(spot_dsharpes):+.3f}, {max(spot_dsharpes):+.3f}]   "
          f"futures dSharpe range [{min(fut_dsharpes):+.3f}, {max(fut_dsharpes):+.3f}]")
    clause4_plateau = plateau_same_sign
    print(f"\n    Clause 4 (same-sign plateau across JS_M_GRID, not an "
          f"isolated peak): {'PASS' if clause4_plateau else 'FAIL'}")
    print(f"    (Note: a same-sign plateau of a NEGATIVE dSharpe is a stable "
          f"non-result, not evidence for\n    promotion -- this clause is "
          f"about robustness of whatever sign is found, not about the sign "
          f"itself\n    being favourable; clause 1 already governs "
          f"favourability.)")

    # ========================================================== STEP 7
    hr("STEP 7 -- configuration count")
    print(f"    Primary full compare() (m={JS_M_PRIMARY}, BTC+ETH, 2 markets, "
          f"3 slices): {len(primary_rows)} cells")
    print(f"    Fee-tier re-run (m={JS_M_PRIMARY}, BTC inner-val, 2 fee "
          f"markets): {len(fee_rows)} cells")
    print(f"    Plateau sweep (m in {[m for m in JS_M_GRID if m != JS_M_PRIMARY]}, "
          f"BTC-only, 2 markets, 2 slices each): "
          f"{CONFIG_CELLS - len(primary_rows) - len(fee_rows)} cells")
    print(f"\n    Distinct JS_M values evaluated: {len(JS_M_GRID)} {JS_M_GRID}")
    print(f"    TOTAL (m, market, slice) CELLS RUN THROUGH compare()/run_slice(): "
          f"{CONFIG_CELLS}")

    # ========================================================== VERDICT
    hr("PROMOTION BAR")
    clauses = {
        "(1) primary-cell dSharpe > 0.2 or boot excludes zero, "
        "BOTH markets, BOTH train+val": clause1_all4,
        "(2) ETH replication same sign as BTC inner-val, BOTH markets": clause2_both,
        "(3) survives 0.40% fee tier, no sign reversal, BOTH markets": clause3_both,
        "(4) JS_M_GRID sweep is a same-sign plateau, not an isolated peak": clause4_plateau,
        "(5) non-degeneracy: bind_frac>thresh and R^2<thresh (Step 2)": step0_pass,
        "(6) causal truncation probe passes (Step 1)": causal_ok,
    }
    for k, v in clauses.items():
        print(f"    {k:70s}: {'PASS' if v else 'FAIL'}")
    promote = all(clauses.values())
    verdict = "PROMOTE-recommendation" if promote else "NEGATIVE"
    print(f"\n    VERDICT: {verdict}")
    if not promote:
        failed = [k for k, v in clauses.items() if not v]
        print(f"    Failing clause(s): {'; '.join(failed)}")
        print("    Per the pre-registered gate: since clause (1) is the first-listed "
              "and (per this file's\n    own instructions) an earlier failing check "
              "stops the rest of the promotion-bar reasoning from\n    mattering, the "
              "holdout (>= 2023-01-01) was NOT read. Its own criterion -- 'ALL of the "
              "above\n    clear on inner-train/inner-validation/ETH/fee-tier' -- was "
              "not met.")

    print(f"\n    Max timestamp read anywhere in this run: {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")

    hr("HOLDOUT")
    holdout_msg = ("NO" if not promote
                  else "NOT YET -- gate cleared, awaiting operator go-ahead per the routine")
    print(f"    Holdout consulted: {holdout_msg}")
    print("    This script never reads a bar at or after OOS_START (2023-01-01); "
          "`load_btc`/`load_eth`\n    truncate before it and `compare`/`run_slice` "
          "assert against it on every call.")


if __name__ == "__main__":
    main()
