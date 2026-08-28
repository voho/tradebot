"""R-172 NOVEL branch: FCR interval width modulates `kelly_regime.py`'s own
dormant `vote_gamma` convex-response exponent, per-bar.

MECHANISM (one sentence, design doc S5 "Novel"): feed the FCR-corrected
interval's own two-sided WIDTH -- how imprecisely pinned-down the active
3-anchor pattern's forward-return edge is, once multiplicity is accounted
for -- into `kelly_regime.py`'s already-shipped, currently dormant (v4 uses
the default `vote_gamma=1.0`) convex-response exponent, so a per-bar
`gamma_t` replaces the fixed scalar:

    gamma_t = 1 + k * clip(width_{p(t)}(t) / width_ref, 0, WIDTH_CAP_RATIO)
    frac_gamma(t) = frac(t) ** gamma_t(t)
    desired = frac_gamma * v4_scale(df)
    target = apply_deadband(desired)

`k = 1.0` is PRIMARY (not swept for PRIMARY -- the 3-point robustness
bracket uses `k in {0.5, 1.0, 2.0}`, design doc S7 clause 3). Where
`width` is NaN (`n_used < MIN_N`, "no evidence yet"), `gamma_t = 1.0`
exactly -- v4-identical, the same no-evidence default `fcr_lower_bounds`
itself already uses for `lcb`/`ucb`. This file is built entirely on top
of `experiments/r172_shared.py` (frozen, not editable) and, for `frac`/
`scale`/`deadband`, `experiments/r102_shared.py` (re-exported by
`r172_shared`). Neither file is edited. The sibling conservative branch
(`experiments/r172_conservative_fcr_gate.py`, implemented independently
in a different worktree) is never read or assumed to exist.

--------------------------------------------------------------------------
WIDTH_REF PROVENANCE -- computed ONCE, frozen as a literal constant, reused
UNCHANGED for inner-validation, ETH, and the fee-tier check (design doc
S5's own "Novel" text; mirrors R-171's own disclosed "derive G once on
inner-train only, reuse unchanged" convention for its ONS eps/beta
constants, see `r171_shared.py`'s "PAPER-VS-TEXTBOOK PROVENANCE" section).

Because `width_ref` must be reused UNCHANGED across ETH too (`r172_direction
.md` S5), it cannot be a per-call recomputation filtered from whatever `df`
`build_target` happens to receive -- ETH's own committed history
(2016-03-09..2019-12-31) only partially overlaps [INNER_TRAIN_START,
INNER_TRAIN_END]=[2017-01-01,2020-12-31], and a per-call filter would
silently compute a DIFFERENT number for ETH than the one derived from BTC.
So `WIDTH_REF` below is a literal, hardcoded float, computed exactly once
via the derivation reproduced in `_derive_width_ref` (BTC inner-train only,
never re-run against ETH or the holdout), and independently RE-VERIFIED
against a fresh real-data computation at the top of `main()` before
anything else runs (so a transcription error cannot silently stand,
matching `r171_shared.py`'s own eps/beta self-check convention) -- not
merely asserted by inspection.

Derivation (run once, disclosed exactly, reproduced by `_derive_width_ref`
below):

    idx, lcb, ucb, n_used = fcr_lower_bounds(load_btc())
    width = ucb - lcb                          # NaN where n_used < MIN_N
    mask = (idx >= INNER_TRAIN_START) & (idx <= INNER_TRAIN_END)
    WIDTH_REF = median(width[mask][isfinite(width[mask])])
              = 0.034627313802142595
    # 1,461 calendar days in [2017-01-01, 2020-12-31]; 1,218 of them
    # (83.4%) had n_used >= MIN_N=30 by that day -- consistent with S4's
    # own disclosed prediction that several patterns stay below MIN_N for
    # long stretches of inner-train, reported again as KS-C below.

CAUSALITY CONSEQUENCE OF FREEZING WIDTH_REF AS A LITERAL CONSTANT (rather
than a per-call date-filtered slice of whatever `df` is passed in): since
`WIDTH_REF` does not depend on `df` AT ALL, `build_target(df)` and
`build_target(df.iloc[:k])` use the IDENTICAL `width_ref` for every
truncation cut -- this is what makes `causal_truncation_probe_series`
below pass cleanly regardless of where a cut falls relative to
INNER_TRAIN_END, and is DELIBERATE, not incidental: the real BTC dataset's
own cuts at fractions (0.5, 0.7, 0.9) of its full pre-holdout span
land at 2020-01-01 / 2021-03-14 / 2022-05-26 respectively (verified below,
`_verify_cut_coverage`) -- the 0.5 cut falls BEFORE INNER_TRAIN_END, so a
per-call date-filtered recomputation would have legitimately produced a
DIFFERENT (still causal, just noisier) `width_ref` for that cut alone,
which would have made the probe's prefix-matching assertion fail for a
reason that has nothing to do with an actual lookahead bug. Freezing the
number once, as a literal, sidesteps that entirely -- the same reason
`r171_shared.py` hardcodes `ONS_EPS_BTC`/`ONS_BETA_BTC` as literals instead
of recomputing `G` from whatever frame is passed to its own builder.

Configs evaluated by this file: reported at the end of `main()` (ROUTINE.md
/ R-163-R-171's own running-counter convention). PRIMARY `compare()` = 6
cells; robustness bracket (`k` in {0.5, 2.0}) = 6 cells each = 12; fee-tier
falsification re-run = 4 cells (2 markets x {standard, 0.40%}). Total = 22.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r172_shared import (  # noqa: E402
    DD_REDUCTION_PROMOTE_PP,
    EXPOSURE_MATCH_BAND,
    FEE_TIER,
    FUTURES,
    GATE_MIN_BINDING_FRACTION,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MIN_N,
    R2_KILL_THRESH,
    SHARPE_DELTA_PROMOTE,
    SPOT,
    TargetStrategy,
    WIDTH_CAP_RATIO,
    apply_deadband,
    assert_no_holdout,
    binding_fraction,
    broadcast_daily,
    causal_truncation_probe_series,
    compare,
    fcr_lower_bounds,
    fee_at,
    load_btc,
    load_eth,
    print_rows,
    relabeling_r2,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

# ==========================================================================
# (0) WIDTH_REF: frozen literal, see module docstring for full derivation
#     and provenance. Verified against a fresh real-data computation in
#     `main()`, not merely asserted by inspection.
# ==========================================================================

WIDTH_REF = 0.034627313802142595  # median finite two-sided FCR width,
                                    # BTC inner-train (2017-01-01..2020-12-31)


def _derive_width_ref(btc_full: pd.DataFrame) -> tuple[float, int, int]:
    """Reproduces the WIDTH_REF derivation exactly (see module docstring).
    Returns (width_ref, n_days_in_range, n_finite) for the disclosure/
    verification print in `main()`."""
    idx, lcb, ucb, _n_used = fcr_lower_bounds(btc_full)
    idx = pd.DatetimeIndex(idx)
    width = ucb - lcb
    mask = ((idx >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
             (idx <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    w = width[mask]
    finite = w[np.isfinite(w)]
    ref = float(np.median(finite)) if len(finite) else float("nan")
    return ref, int(mask.sum()), int(len(finite))


PRIMARY_K = 1.0
BRACKET_KS = (0.5, 2.0)  # design doc S7 clause 3: 3-point bracket {0.5,1.0,2.0}
FEE_TIER_STRESS = FEE_TIER  # 0.40% taker, already the round's own falsification tier


# ==========================================================================
# (1) gamma_t construction and the build_target factory.
# ==========================================================================

def gamma_daily_series(df: pd.DataFrame, k: float, width_ref: float = WIDTH_REF):
    """Causal, daily-resolution gamma_t (design doc S5): 1.0 exactly wherever
    `width` is NaN (n_used < MIN_N, "no evidence yet" -- fcr_lower_bounds'
    own contract), else `1 + k*clip(width/width_ref, 0, WIDTH_CAP_RATIO)`.
    Returns (gamma, idx) at daily resolution, ready for `broadcast_daily`."""
    idx, lcb, ucb, _n_used = fcr_lower_bounds(df)
    width = ucb - lcb
    with np.errstate(invalid="ignore"):
        ratio = width / width_ref
        gamma_raw = 1.0 + k * np.clip(ratio, 0.0, WIDTH_CAP_RATIO)
    gamma = np.where(np.isfinite(width), gamma_raw, 1.0)
    return gamma, pd.DatetimeIndex(idx)


def make_build_target(k: float, width_ref: float = WIDTH_REF):
    """Pure `df -> np.ndarray` builder for a given `k` (and, for the
    robustness bracket, the SAME frozen `width_ref` -- only k varies)."""

    def build_target(df: pd.DataFrame) -> np.ndarray:
        gamma_daily, idx = gamma_daily_series(df, k, width_ref)
        # OPERATOR PATCH (post-report, pre-freeze verification): day D's
        # own gamma_daily value is computed from day D's OWN last bar
        # (daily_pattern), so broadcasting it onto day D's own bars
        # unshifted is a one-day lookahead -- the identical bug the
        # sibling conservative branch's implementer found and fixed
        # (experiments/r172_conservative_fcr_gate.py), independently
        # confirmed here by direct inspection because this file's own
        # causal_truncation_probe used only smooth synthetic data (where
        # latched anchor votes almost never flip mid-day, so the bug is
        # numerically invisible to a tolerance-based check) rather than
        # real, choppier BTC data. Fix: lag the daily index by one
        # calendar day before broadcasting, matching this codebase's
        # standard `.shift(1)` convention (e.g. v4_symmetric_vol) and the
        # conservative branch's own fix.
        idx_lagged = idx + pd.Timedelta(days=1)
        gamma_bar = broadcast_daily(gamma_daily, idx_lagged, df.index, fill_value=1.0)
        frac = v4_vote_frac(df).to_numpy()
        with np.errstate(invalid="ignore"):
            frac_gamma = np.power(frac, gamma_bar)
        desired = frac_gamma * v4_scale(df)
        return apply_deadband(desired)

    build_target.__name__ = f"r172_novel_fcr_gamma_k{k:g}"
    return build_target


build_target = make_build_target(PRIMARY_K)  # PRIMARY, k=1.0


# ==========================================================================
# (2) Causal-truncation probe (required check 1). Synthetic data, same
#     generator shape as r172_shared._self_test's own convention (WIDTH_REF
#     is a frozen literal independent of the probe's own df -- see module
#     docstring -- so this exercises the rest of the pipeline's causality:
#     fcr_lower_bounds' bucket accumulation, frac, scale, deadband).
# ==========================================================================

def _synthetic_frame(seed: int, periods: int = 200_000) -> pd.DataFrame:
    idx = pd.date_range("2017-01-01", periods=periods, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    return pd.DataFrame({"open": close, "high": high, "low": low,
                          "close": close, "volume": 1.0}, index=idx)


def _verify_cut_coverage(btc_full: pd.DataFrame,
                          cuts: tuple[float, ...] = (0.5, 0.7, 0.9)) -> dict:
    """Disclosure only (module docstring's own claim, checked against real
    data, not asserted by inspection): for the REAL BTC dataset, where do
    the probe's cuts fall relative to INNER_TRAIN_END? Confirms WIDTH_REF
    must be a frozen literal (not a per-call date filter) for cut=0.5."""
    n = len(btc_full)
    out = {}
    inner_train_end_ts = pd.Timestamp(INNER_TRAIN_END, tz="UTC")
    for cut in cuts:
        k = int(n * cut)
        cut_date = btc_full.index[k - 1]
        out[cut] = {"date": str(cut_date.date()),
                    "before_inner_train_end": bool(cut_date < inner_train_end_ts)}
    return out


def run_causal_probe() -> bool:
    ok = True
    for seed in (172001, 172002):
        df = _synthetic_frame(seed)
        ok = ok and causal_truncation_probe_series(build_target, df, cuts=(0.5, 0.7, 0.9))
    return ok


# ==========================================================================
# (3) KS-A (binding_fraction) / KS-B (relabeling_r2), computed directly as
#     PURE ARRAY functions on inner-train BTC (r172_shared's own kill-switch
#     helpers are pure functions over arrays the caller builds -- no
#     backtest/compare() call needed for these; matches r171_novel's own
#     "kill switches computed BEFORE any compare()/backtest call" ordering).
#
#     Computed from build_target(btc_full) (the FULL, continuous 2017-2022
#     pre-holdout series -- NOT compare()'s own internal 80-day-warmup
#     slice, so fcr_lower_bounds' bucket accumulation genuinely starts at
#     the dataset's own first day, not a truncated re-start) restricted
#     to inner-train bars afterward -- the same "compute once, continuously,
#     over the full available span, then restrict" pattern r171_novel used
#     for its own kill switches and Monte Carlo falsification test.
# ==========================================================================

def _inner_train_mask(index: pd.DatetimeIndex) -> np.ndarray:
    mask = ((index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
            (index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    return np.asarray(mask)


def run_kill_switches(btc_full: pd.DataFrame, k: float = PRIMARY_K) -> dict:
    gamma_daily, idx = gamma_daily_series(btc_full, k)
    gamma_bar = broadcast_daily(gamma_daily, idx, btc_full.index, fill_value=1.0)
    target = make_build_target(k)(btc_full)
    raw_v4 = v4_raw_desired(btc_full)

    train_mask = _inner_train_mask(btc_full.index)
    ks_a = binding_fraction(np.abs(gamma_bar[train_mask] - 1.0) > 1e-12)
    ks_b = relabeling_r2(target[train_mask], raw_v4[train_mask])
    return {
        "k": k,
        "ks_a_binding_fraction": ks_a,
        "ks_a_trips": bool(ks_a < GATE_MIN_BINDING_FRACTION),
        "ks_b_relabeling_r2": ks_b,
        "ks_b_trips": bool(np.isfinite(ks_b) and ks_b >= R2_KILL_THRESH),
    }


# ==========================================================================
# (4) KS-C (mandatory disclosure, not a kill switch): per-pattern resolved
#     bucket size by INNER_TRAIN_END and INNER_VAL_END (design doc S6).
# ==========================================================================

def ks_c_bucket_sizes(btc_full: pd.DataFrame) -> dict:
    idx, _lcb, _ucb, n_used = fcr_lower_bounds(btc_full)
    idx = pd.DatetimeIndex(idx)
    from experiments.r172_shared import daily_pattern  # local import, avoids polluting the top-level namespace
    pattern = daily_pattern(btc_full).reindex(idx).to_numpy()
    out = {}
    for boundary_name, boundary in (("inner_train_end", INNER_TRAIN_END),
                                     ("inner_val_end", INNER_VAL_END)):
        ts = pd.Timestamp(boundary, tz="UTC")
        pos = int(np.searchsorted(idx.values, ts.to_datetime64(), side="right")) - 1
        if pos < 0:
            out[boundary_name] = {p: 0 for p in range(8)}
            continue
        per_pattern = {}
        for p in range(8):
            mask = pattern[: pos + 1] == p
            per_pattern[p] = int(n_used[: pos + 1][mask][-1]) if mask.any() else 0
        out[boundary_name] = per_pattern
    return out


# ==========================================================================
# (5) Fee-tier falsification test (design doc S5's own falsification test
#     for the NOVEL branch, = S7 clause 4 -- the SAME check, not two
#     independent ones -- disclosed as such in the report). BTC inner_val
#     only, both markets, standard tier vs 0.40% taker.
# ==========================================================================

def run_fee_falsification(btc_full: pd.DataFrame, k: float = PRIMARY_K) -> dict:
    build = make_build_target(k)
    cand = TargetStrategy(build, name=f"r172_novel_fcr_gamma_k{k:g}")
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    out = {}
    for market in (SPOT, FUTURES):
        a_std = run_slice(cand, btc_full, INNER_VAL_START, INNER_VAL_END, "inner_val_std", market)
        b_std = run_slice(ctrl, btc_full, INNER_VAL_START, INNER_VAL_END, "inner_val_std", market)
        d_std = a_std.sharpe - b_std.sharpe

        m_stress = fee_at(market, FEE_TIER_STRESS)
        a_str = run_slice(cand, btc_full, INNER_VAL_START, INNER_VAL_END, "inner_val_fee40", m_stress)
        b_str = run_slice(ctrl, btc_full, INNER_VAL_START, INNER_VAL_END, "inner_val_fee40", m_stress)
        d_str = a_str.sharpe - b_str.sharpe

        out[market.name] = {
            "d_sharpe_standard": d_std, "d_sharpe_fee40": d_str,
            "sign_flip": bool(np.sign(d_std) != np.sign(d_str) and d_std != 0 and d_str != 0),
        }
    falsified = any(v["sign_flip"] for v in out.values())
    return {"k": k, "cells": out, "falsified": falsified}


# ==========================================================================
# main
# ==========================================================================

def hr(msg: str) -> None:
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


def _mean_d_sharpe(rows: list[dict], slice_name: str) -> float:
    cells = [r["d_sharpe"] for r in rows if r["slice"] == slice_name]
    return float(np.mean(cells)) if cells else float("nan")


def main() -> None:
    n_configs = 0

    hr("R-172 NOVEL: FCR interval width modulates kelly_regime's dormant vote_gamma exponent")
    print("mechanism: gamma_t = 1 + k*clip(width_p(t)/width_ref, 0, 2); frac_gamma = frac**gamma_t; "
          "desired = frac_gamma*scale; target = deadband(desired). PRIMARY k=1.0.")

    btc_full = load_btc()
    eth_full = load_eth()
    assert_no_holdout(btc_full, "main(): btc")
    assert_no_holdout(eth_full, "main(): eth")
    print(f"\nBTC: {len(btc_full):,} bars ({btc_full.index[0]} .. {btc_full.index[-1]})")
    print(f"ETH: {len(eth_full):,} bars ({eth_full.index[0]} .. {eth_full.index[-1]})")

    # ---- WIDTH_REF re-verification against fresh real-data computation ----
    hr("WIDTH_REF re-verification (frozen literal vs a fresh real-data recomputation)")
    ref_check, n_days, n_finite = _derive_width_ref(btc_full)
    print(f"  hardcoded WIDTH_REF = {WIDTH_REF!r}")
    print(f"  freshly recomputed  = {ref_check!r}  "
          f"(n_days_in_[{INNER_TRAIN_START},{INNER_TRAIN_END}]={n_days}, n_finite_widths={n_finite})")
    assert abs(ref_check - WIDTH_REF) < 1e-9, \
        f"WIDTH_REF drifted from its disclosed derivation: {ref_check} != {WIDTH_REF}"
    print("  MATCH -- WIDTH_REF is not stale.")

    cov = _verify_cut_coverage(btc_full)
    hr("Disclosure: real-BTC causal-probe cut coverage vs INNER_TRAIN_END (module docstring's own claim)")
    for cut, info in cov.items():
        print(f"  cut={cut}: falls at {info['date']}  before_inner_train_end={info['before_inner_train_end']}")
    print("  => at least one cut (0.5) falls BEFORE INNER_TRAIN_END on the real dataset, confirming "
          "WIDTH_REF must be a frozen literal (not a per-call date filter) for the probe below to be "
          "meaningful on real BTC data; the probe itself runs on synthetic data per its own convention, "
          "where this is moot since WIDTH_REF does not depend on the probe's df at all.")

    # ---- (1) REQUIRED CHECK: causality ----
    hr("REQUIRED CHECK 1/3: causal_truncation_probe_series(build_target, ...) end-to-end")
    causal_ok = run_causal_probe()
    print(f"  causal_truncation_probe_series (2 synthetic seeds, cuts=(0.5,0.7,0.9)): {causal_ok}")
    assert causal_ok, "CAUSALITY PROBE FAILED"

    # ---- (2)+(3) REQUIRED CHECKS: KS-A, KS-B, on inner-train BTC ----
    hr("REQUIRED CHECKS 2-3/3: KS-A (non-triviality) and KS-B (relabeling), PRIMARY k=1.0, inner-train BTC")
    ks = run_kill_switches(btc_full, PRIMARY_K)
    print(f"  KS-A binding_fraction (frac of inner-train bars where gamma_t != 1.0) = "
          f"{ks['ks_a_binding_fraction']:.6f}  (>= {GATE_MIN_BINDING_FRACTION} required)  "
          f"trips={ks['ks_a_trips']}")
    print(f"  KS-B relabeling_r2 (candidate final target vs v4_raw_desired) = "
          f"{ks['ks_b_relabeling_r2']:.6f}  (< {R2_KILL_THRESH} required)  trips={ks['ks_b_trips']}")

    # ---- KS-C: mandatory sample-size disclosure (not a kill switch) ----
    hr("KS-C (mandatory disclosure, not a kill switch): per-pattern resolved bucket size n_p")
    ksc = ks_c_bucket_sizes(btc_full)
    for boundary_name in ("inner_train_end", "inner_val_end"):
        print(f"  by {boundary_name}:")
        for p in range(8):
            n_p = ksc[boundary_name][p]
            print(f"    pattern={p:03b} ({p}): n_p={n_p:>5d}  {'>=MIN_N' if n_p >= MIN_N else '<MIN_N (no evidence yet)'}")

    any_ks_trips = ks["ks_a_trips"] or ks["ks_b_trips"]
    if any_ks_trips:
        print("\n*** KILL SWITCH TRIPPED -- STOPPING before compare()/falsification/bracket, per this "
              "project's own precedent (R-170 et seq.). No further backtest is run. ***")
        hr("VERDICT")
        print("NEGATIVE (kill switch tripped). See KS-A/KS-B output above.")
        print(f"\nconfigurations evaluated: {n_configs}")
        return

    # ---- PRIMARY compare() ----
    hr("PRIMARY compare(): BTC (inner_train, inner_val) + ETH (eth_replication), both markets, k=1.0")
    primary_rows = compare(build_target, label="r172_novel_fcr_gamma", btc=btc_full, eth=eth_full)
    n_configs += len(primary_rows)
    print_rows(primary_rows)

    # ---- Fee-tier falsification (design doc S5, = S7 clause 4) ----
    hr("FALSIFICATION TEST (design doc S5, = S7 clause 4): 0.40% taker-fee sign survival, "
      "BTC inner_val, both markets, PRIMARY k=1.0")
    fee_result = run_fee_falsification(btc_full, PRIMARY_K)
    n_configs += 2 * len(fee_result["cells"])  # 2 tiers x N markets
    for market_name, cell in fee_result["cells"].items():
        print(f"  {market_name:12s}: d_sharpe(standard)={cell['d_sharpe_standard']:+.4f}  "
              f"d_sharpe(0.40% taker)={cell['d_sharpe_fee40']:+.4f}  sign_flip={cell['sign_flip']}")
    print(f"  => FALSIFIED={fee_result['falsified']}")

    # ---- Robustness bracket: k in {0.5, 2.0} ----
    hr(f"ROBUSTNESS BRACKET (design doc S7 clause 3): k in {BRACKET_KS} around PRIMARY k={PRIMARY_K}")
    bracket_rows: dict[float, list[dict]] = {PRIMARY_K: primary_rows}
    for k in BRACKET_KS:
        build_k = make_build_target(k)
        rows_k = compare(build_k, label=f"r172_novel_fcr_gamma_k{k:g}", btc=btc_full, eth=eth_full)
        n_configs += len(rows_k)
        bracket_rows[k] = rows_k
        print(f"\n  k={k}:")
        print_rows(rows_k)

    hr("PLATEAU CHECK: does PRIMARY's (k=1.0) sign hold across k in {0.5, 1.0, 2.0}?")
    plateau_ok = True
    for slice_name in ("inner_train", "inner_val", "eth_replication"):
        vals = {k: _mean_d_sharpe(bracket_rows[k], slice_name) for k in (0.5, 1.0, 2.0)}
        signs = {k: (np.sign(v) if np.isfinite(v) else None) for k, v in vals.items()}
        same_sign = len({s for s in signs.values() if s is not None}) <= 1 and None not in signs.values()
        plateau_ok = plateau_ok and same_sign
        print(f"  {slice_name:16s}: " + "  ".join(f"k={k}:{vals[k]:+.3f}" for k in (0.5, 1.0, 2.0)) +
              f"   same_sign_all_k={same_sign}")

    # ---- Promotion-clause evaluation (design doc S7), applied mechanically ----
    hr("PROMOTION-RULE CLAUSES (design doc S7), applied mechanically to the inner-validation/ETH read "
       "(NOVEL's job stops here -- holdout not consulted, per this branch's own scope)")

    def _cell(rows, slice_name, market):
        return next(r for r in rows if r["slice"] == slice_name and r["market"] == market)

    btc_spot = _cell(primary_rows, "inner_val", "spot")
    btc_fut = _cell(primary_rows, "inner_val", "futures_5x")
    eth_spot = _cell(primary_rows, "eth_replication", "spot")
    eth_fut = _cell(primary_rows, "eth_replication", "futures_5x")
    cells4 = (btc_spot, btc_fut, eth_spot, eth_fut)

    clause1_sharpe = all(c["d_sharpe"] >= SHARPE_DELTA_PROMOTE for c in cells4)
    clause1_dd = all(c["risk_matched"] and (-c["d_dd"]) >= DD_REDUCTION_PROMOTE_PP for c in cells4)
    clause1 = bool(clause1_sharpe or clause1_dd)
    clause2 = not fee_result["falsified"]
    clause3 = plateau_ok
    clause4 = not fee_result["falsified"]  # same check as clause2 for this branch, see S5

    print(f"  clause 1 (dSharpe>=+0.2 on BTC+ETH spot+futures inner_val/eth_replication, OR matched-"
          f"exposure DD reduction>=5pp on all four): {clause1}  "
          f"(via {'sharpe' if clause1_sharpe else ('dd' if clause1_dd else 'neither')})")
    for name, c in (("BTC spot", btc_spot), ("BTC futures_5x", btc_fut),
                    ("ETH spot", eth_spot), ("ETH futures_5x", eth_fut)):
        print(f"      {name:16s} d_sharpe={c['d_sharpe']:+.4f}  d_dd={c['d_dd']:+.2f}  "
              f"risk_matched={c['risk_matched']}")
    print(f"  clause 2 (survives pre-registered falsification test = 0.40% fee-tier sign check): {clause2}")
    print(f"  clause 3 (plateau: k in {{0.5,1.0,2.0}} agree in direction)                        : {clause3}")
    print(f"  clause 4 (sign does not reverse at 0.40% fee tier -- SAME check as clause 2 here)  : {clause4}")
    print("  NOTE: for this branch, clauses 2 and 4 are the identical fee-tier check (design doc S5's "
          "own falsification test IS the 0.40% fee-tier sign check named again in S7 clause 4) -- "
          "reported as one finding counted once, not two independent passes.")
    print("  NOTE: ETH has no data at or after INNER_VAL_START (2021-01-01) -- 'ETH' above is the "
          "eth_replication slice (ETH's own full 2016-2019 history), the same disclosed substitution "
          "R-171's own novel branch used for the identical reason.")

    all_pass = bool(clause1 and clause2 and clause3 and clause4)
    hr("FINAL VERDICT (assessment against the frozen decision rule; not a promotion decision -- "
       "operator decides separately whether to consult the holdout)")
    if all_pass:
        print("  ALL FOUR CLAUSES CLEAR -> PROMOTE-CANDIDATE (inner-validation/ETH/fee-tier only; "
              "holdout not read by this branch)")
    else:
        cleared = [n for n, v in (("1", clause1), ("2", clause2), ("3", clause3), ("4", clause4)) if v]
        failed = [n for n, v in (("1", clause1), ("2", clause2), ("3", clause3), ("4", clause4)) if not v]
        print(f"  FALL-THROUGH -> NEGATIVE. Clauses cleared: {cleared or 'none'}. "
              f"Clauses failed: {failed or 'none'}. Per ROUTINE.md, a fall-through is reported as a "
              f"fall-through, not rounded to the nearest label.")

    print(f"\nconfigurations evaluated: {n_configs}")
    print("  breakdown: PRIMARY compare()=6, k=0.5 bracket=6, k=2.0 bracket=6, "
          "fee-tier re-run=4 (2 markets x {standard, 0.40%})  => 22 total")


if __name__ == "__main__":
    main()
