#!/usr/bin/env python
"""R-123 CONSERVATIVE branch: ``ShrinkageKellyMahalanobisBrake`` -- instead of
discounting ``kelly_regime_v4``'s own final, already-deadbanded target
(``v4_target(df) * (1 - discount)``, the architecture five prior rounds --
R-109, R-112, R-115, R-121, R-122 -- all used and which all failed B4 on
ETH), this branch shrinks ``frac`` (the 3-anchor vote, v4's own stand-in for
"the market currently has a positive-drift edge") TOWARD ZERO by the
identical novelty statistic, BEFORE ``frac`` is multiplied by ``scale`` and
BEFORE the deadband is applied -- Baker & McHale (2013)'s "shrink the edge
estimate, not the sized bet" result, applied to v4's own vote. Full
literature grounding, the exhaustive non-duplication argument, and this
round's four NAMED failure risks all live in ``experiments/r123_shared.py``'s
own module docstring (read in full before this file was written); not
re-derived here beyond the summary above. This file NEVER edits
``r123_shared.py`` (frozen, shared with the parallel NOVEL branch,
``experiments/r123_novel_*.py``, a disjoint file this session does not read
or coordinate with), and never reads a bar at or after ``r123_shared.OOS_START``
(2023-01-01) from any data source, regardless of outcome.

MECHANISM (exact, entirely ``r123_shared`` primitives, no new logic added on
top; the ONLY reference construction used is R-109's own conservative
convention -- Mahalanobis distance over the 3-feature ``FEATURE_BUILDERS``
panel, held fixed on purpose so architecture is this round's sole variable):

    daily[t]   = r123_shared.build_daily_features(df)[t]           # log_vol,
                  # anchor_disp, kurtosis (default FEATURE_BUILDERS)
    dist[t]    = r123_shared.rolling_mahalanobis_distance(daily)[t]  # distance
                  # from mean/cov of daily[t-730 : t-1] (min 180 prior days)
    state[t]   = r123_shared.causal_rolling_percentile_rank(dist,
                     window=BASELINE_WINDOW_DAYS, min_periods=MIN_REF_DAYS)[t]
    shrink[t]  = r123_shared.frac_shrink_fraction(df, state, thresh,
                     max_discount)[t]        # in [0,1], ramps thresh -> 1
    target[t]  = r123_shared.conservative_target(df, state, thresh,
                     max_discount)[t]
               = apply_deadband(v4_vote_frac(df) * (1 - shrink) * v4_scale(df))[t]

``scale`` (``v4_symmetric_vol`` -> ``conditional_target_scale``) is reused
byte-for-byte unchanged; only ``frac`` is shrunk, and the shrink happens
BEFORE the deadband sees the product -- the one architectural difference from
every prior round in this ERR sub-axis, and the reason the deadband can now
latch/release differently than any post-deadband-discount construction could
produce (checked below via trade-count comparison against v4's own baseline).

WARMUP NOTE (same correction R-109's own branch file made under its own
name): ``TargetStrategy``'s framework default warmup (v4's own 80-day anchor
requirement) is shorter than this construction's genuine reference-window
requirement (``max(V4_HORIZONS)=80`` days for `anchor_disp` to first be valid,
PLUS ``MIN_REF_DAYS=180`` more days before the Mahalanobis distance is defined
at all -- 260 days minimum). Calling ``r123_shared.compare()``/
``inner_val_rows()``/``b5_fee_tier()`` VERBATIM would silently truncate the
state to its NaN->0 fallback (no shrink, identical to v4) for a large
fraction of every non-inner-train slice, for a data-availability reason
rather than a genuine one. This file therefore defines thin wrappers
(``compare_with_warmup``, ``inner_val_rows_with_warmup``,
``b5_fee_tier_with_warmup``) that are BYTE-IDENTICAL to
``r109_shared.compare()``/``r105_shared.inner_val_rows()``/
``r105_shared.b5_fee_tier()`` except the CANDIDATE ``TargetStrategy`` is
instantiated with ``warmup=CANDIDATE_WARMUP_BARS`` (300 days, derived not fit)
instead of the 80-day class default -- built ONLY from primitives
``r123_shared`` already re-exports (``TargetStrategy``, ``run_slice``,
``paired_diff``, ``fee_at``), never re-deriving the GATING arithmetic itself:
``b1_from_inner_val``, ``b4_eth_falsification`` are called VERBATIM on the
resulting row dicts, and ``b5_fee_tier_with_warmup`` reproduces only the same
no-reversal sign comparison ``r105_shared.b5_fee_tier`` already uses (needed
because that function's own internal call to ``inner_val_rows`` cannot be
redirected to the warmup-aware version). The CONTROL (``kelly_regime_v4``)
strategy's warmup is left completely UNCHANGED in every call, so every
``d_sharpe``/paired-bootstrap comparison stays apples-to-apples.

CONFIGURATIONS EVALUATED (formula): 6 (Step-0 grid, ``CONS_THRESH_GRID`` x
``CONS_MAXD_GRID``) + 6 (primary cell's full ``compare_with_warmup()``:
inner_train x2 markets + inner_val x2 markets + eth_replication x2 markets)
+ 12 (B3's full 6-cell grid x 2 markets, 2 of the 12 reused directly from the
primary's own inner_val rows, 10 freshly computed) + 2 (B5's 0.40% fee tier,
2 markets) = 26 total, IF Step-0 selects a primary. If no cell qualifies,
this file stops after the 6 Step-0 cells.

DECISION RULE (pre-registered, verbatim, unaltered after seeing any number):
PROMOTE-candidate only if the causal-truncation probe AND B1 (both markets)
AND B3 (plateau majority, same sign as primary) AND B4 (full, both markets)
AND B5 all pass. Anything else is NEGATIVE. B2 is diagnostic only.

USAGE
-----
    python experiments/r123_conservative_shrinkage_kelly.py
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

from experiments import r123_shared  # noqa: E402

assert r123_shared.V4_HORIZONS == (20, 40, 80), r123_shared.V4_HORIZONS
assert r123_shared.CONS_THRESH_GRID == (0.80, 0.90, 0.95), r123_shared.CONS_THRESH_GRID
assert r123_shared.CONS_MAXD_GRID == (0.5, 1.0), r123_shared.CONS_MAXD_GRID
assert r123_shared.CONS_SELECTION_ORDER[0] == r123_shared.CONS_PRIMARY, r123_shared.CONS_SELECTION_ORDER

# Derived (not fit): max anchor horizon (80d, for anchor_disp's own first
# valid point) + MIN_REF_DAYS (180d, for the Mahalanobis distance's own min
# reference) + a 40-day round-number margin = 300 days. Identical derivation
# to R-109's own CANDIDATE_WARMUP_DAYS.
CANDIDATE_WARMUP_DAYS = max(r123_shared.V4_HORIZONS) + r123_shared.MIN_REF_DAYS + 40   # = 300
CANDIDATE_WARMUP_BARS = int(CANDIDATE_WARMUP_DAYS * r123_shared.BARS_PER_DAY) + 10


# ================================================================== (1)
# The mechanism itself: build_daily_features -> rolling_mahalanobis_distance
# -> causal_rolling_percentile_rank -> conservative_target (frac-shrink).
# Every step is an r123_shared/r109_shared primitive; nothing new added.
# ==================================================================

def novelty_state(df: pd.DataFrame) -> pd.Series:
    """The [0,1] novelty STATE for every day covered by `df`: R-109's own
    conservative convention -- causal rolling-percentile-rank of the rolling
    Mahalanobis distance of the 3-feature daily panel from its own strictly-
    prior reference window."""
    daily = r123_shared.build_daily_features(df)
    dist = r123_shared.rolling_mahalanobis_distance(daily)
    return r123_shared.causal_rolling_percentile_rank(
        dist, window=r123_shared.BASELINE_WINDOW_DAYS, min_periods=r123_shared.MIN_REF_DAYS)


def build_target(df: pd.DataFrame, thresh: float, max_discount: float) -> np.ndarray:
    """frac shrunk toward 0 by this bar's own novelty state, then multiplied
    by scale, then deadbanded -- computed FRESH from whatever frame `df`
    actually is (state depends on df, so it must be recomputed per-frame;
    this is still causal since novelty_state's own construction is causal)."""
    state = novelty_state(df)
    return r123_shared.conservative_target(df, state, thresh, max_discount)


def make_build_target(thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"shrinkage_kelly_t{thresh:g}_m{max_discount:g}"
    return _build


def cell_label(thresh: float, max_discount: float) -> str:
    return f"shrinkage_kelly_t{thresh:g}_m{max_discount:g}"


# ================================================================== (2)
# Pre-flight wiring check (outside the pre-registered grid): max_discount=0.0
# forces shrink === 0 regardless of state, so conservative_target must equal
# v4_target exactly.
# ==================================================================

def self_test_max_discount_zero_identity(df: pd.DataFrame) -> bool:
    ok = True
    for thresh in (0.5, 0.90, 0.999):
        a = build_target(df, thresh=thresh, max_discount=0.0)
        b = r123_shared.v4_target(df)
        same = np.allclose(a, b, equal_nan=True)
        print(f"  build_target(thresh={thresh:g}, max_discount=0.0) == v4_target exactly? {same}")
        ok = ok and same
    return ok


# ================================================================== (3)
# Step-0: 6-cell (thresh x max_discount) grid, via step0_gate_generic.
# ==================================================================

def step0_grid(btc: pd.DataFrame, state_full: pd.Series) -> list[dict]:
    df_inner_train = btc.loc[r123_shared.INNER_TRAIN_START:r123_shared.INNER_TRAIN_END]
    rows = []
    for thresh in r123_shared.CONS_THRESH_GRID:
        for max_discount in r123_shared.CONS_MAXD_GRID:
            disc = r123_shared.frac_shrink_fraction(df_inner_train, state_full, thresh, max_discount)
            cand = r123_shared.conservative_target(df_inner_train, state_full, thresh, max_discount)
            gate = r123_shared.step0_gate_generic(df_inner_train, cand, disc, state_full)
            rows.append(dict(thresh=thresh, max_discount=max_discount, **gate))
    return rows


def select_primary(rows: list[dict]) -> dict | None:
    by_key = {(r["thresh"], r["max_discount"]): r for r in rows}
    for key in r123_shared.CONS_SELECTION_ORDER:
        r = by_key.get(key)
        if r is not None and r["passed"]:
            return r
    return None


def print_step0_summary_table(rows: list[dict]) -> None:
    print(f"\nSTEP-0 GRID SUMMARY (inner-train slice, {r123_shared.INNER_TRAIN_START} -> "
          f"{r123_shared.INNER_TRAIN_END}, state built on the FULL non-holdout BTC history)")
    hdr = (f"{'thresh':>7s} {'max_d':>6s} {'bind_frac':>10s} {'r2_vs_v4':>9s} "
          f"{'r2_vs_vol':>10s} {'state_cv':>9s} {'passed':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = "  <- CONS_PRIMARY" if (r["thresh"], r["max_discount"]) == r123_shared.CONS_PRIMARY else ""
        print(f"{r['thresh']:7.2f} {r['max_discount']:6.2f} {r['bind_frac']:10.4f} "
              f"{r['r2_vs_v4']:9.4f} {r['r2_vs_vol']:10.4f} {r['state_cv']:9.4f} "
              f"{'YES' if r['passed'] else 'no':>7s}{tag}")


# ================================================================== (4)
# Warmup-aware compare/inner_val_rows/b5_fee_tier -- see module docstring's
# WARMUP NOTE. Built ONLY from r123_shared-re-exported primitives; the
# GATING arithmetic itself (b1_from_inner_val, b4_eth_falsification) is
# always called verbatim on the resulting row dicts, never reimplemented.
# ==================================================================

def compare_with_warmup(candidate_build, *, label: str, btc: pd.DataFrame, eth: pd.DataFrame,
                        markets: tuple = (r123_shared.SPOT, r123_shared.FUTURES),
                        include_eth: bool = True, seed: int = 0) -> list[dict]:
    r123_shared.assert_no_holdout(btc, "compare_with_warmup(): btc")
    if include_eth:
        r123_shared.assert_no_holdout(eth, "compare_with_warmup(): eth")

    cand = r123_shared.TargetStrategy(candidate_build, name=f"r123_{label}", warmup=CANDIDATE_WARMUP_BARS)
    ctrl = r123_shared.TargetStrategy(r123_shared.v4_target, name="kelly_regime_v4")

    rows = []
    jobs = [(name, start, end, btc) for name, (start, end) in r123_shared.SLICES.items()]
    if include_eth:
        jobs.append((r123_shared.ETH_SLICE_NAME, None, None, eth))

    for slice_name, start, end, df in jobs:
        for market in markets:
            a = r123_shared.run_slice(cand, df, start, end, slice_name, market)
            b = r123_shared.run_slice(ctrl, df, start, end, slice_name, market)
            pr = r123_shared.paired_diff(a.daily, b.daily, seed=seed)
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


def inner_val_rows_with_warmup(build_fn, label: str, btc: pd.DataFrame,
                               markets: tuple = (r123_shared.SPOT, r123_shared.FUTURES)) -> list[dict]:
    ctrl = r123_shared.TargetStrategy(r123_shared.v4_target, name="kelly_regime_v4")
    cand = r123_shared.TargetStrategy(build_fn, name=f"r123_{label}", warmup=CANDIDATE_WARMUP_BARS)
    rows = []
    for market in markets:
        a = r123_shared.run_slice(cand, btc, r123_shared.INNER_VAL_START, r123_shared.INNER_VAL_END,
                                  "inner_val", market)
        b = r123_shared.run_slice(ctrl, btc, r123_shared.INNER_VAL_START, r123_shared.INNER_VAL_END,
                                  "inner_val", market)
        pr = r123_shared.paired_diff(a.daily, b.daily)
        exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure if b.mean_abs_exposure else float("nan"))
        vol_ratio = (a.realized_vol / b.realized_vol if b.realized_vol else float("nan"))
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


def b5_fee_tier_with_warmup(build_primary, label: str, btc: pd.DataFrame,
                            inner_val_primary: list[dict]) -> tuple[bool, list[dict]]:
    """Reproduces ONLY the no-reversal sign comparison `r105_shared.b5_fee_tier`
    already uses (needed because that function's own internal `inner_val_rows`
    call cannot be redirected to the warmup-aware version above)."""
    fee_markets = (r123_shared.fee_at(r123_shared.SPOT, r123_shared.FEE_TIER),
                  r123_shared.fee_at(r123_shared.FUTURES, r123_shared.FEE_TIER))
    fee_rows = inner_val_rows_with_warmup(build_primary, label, btc, markets=fee_markets)
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
# B3: plateau -- majority of the FULL 6-cell grid's inner_val d_sharpe signs
# (both markets, 12 rows) agree in sign with the primary cell, per market.
# ==================================================================

def b3_plateau(primary_key: tuple[float, float],
              plateau_rows: dict[tuple[float, float], list[dict]]) -> tuple[bool, list[dict]]:
    primary_signs = {r["market"]: np.sign(r["d_sharpe"]) for r in plateau_rows[primary_key]}
    detail = []
    for key, rows in plateau_rows.items():
        for r in rows:
            prim_sign = primary_signs.get(r["market"], 0.0)
            same = bool(np.sign(r["d_sharpe"]) == prim_sign)
            detail.append(dict(grid_key=key, market=r["market"], d_sharpe=r["d_sharpe"],
                               same_sign_as_primary=same))
    n_same = sum(1 for d in detail if d["same_sign_as_primary"])
    b3_pass = (n_same >= len(detail) / 2.0) if detail else False
    return b3_pass, detail


# ================================================================== (6)
# Promotion bar: B1 (gating), B2 (diagnostic), B3 (gating), B4 (gating),
# B5 (gating). b1_from_inner_val / b4_eth_falsification / b2_diagnostic are
# called VERBATIM on already-built row dicts -- their own threshold/sign
# arithmetic is never reimplemented here.
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                      btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    thresh, max_discount = primary_key
    label = cell_label(thresh, max_discount)
    build_primary = make_build_target(thresh, max_discount)

    r123_shared.hr(f"PROMOTION BAR -- PRIMARY CONFIG thresh={thresh:g} max_discount={max_discount:g}")
    print("compare_with_warmup() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare_with_warmup(build_primary, label=label, btc=btc, eth=eth,
                               markets=(r123_shared.SPOT, r123_shared.FUTURES), include_eth=True)
    r123_shared.print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == r123_shared.ETH_SLICE_NAME]
    inner_train_primary = [r for r in rows if r["slice"] == "inner_train"]

    b1_pass, b1_cells = r123_shared.b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = r123_shared.b2_diagnostic(inner_val_primary)

    plateau_rows: dict[tuple[float, float], list[dict]] = {
        primary_key: [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                           d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                           vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                           boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                           boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                      for r in inner_val_primary]
    }
    for row in step0_rows:
        key = (row["thresh"], row["max_discount"])
        if key == primary_key:
            continue
        bf = make_build_target(*key)
        blabel = cell_label(*key)
        plateau_rows[key] = inner_val_rows_with_warmup(bf, blabel, btc)

    b3_pass, b3_detail = b3_plateau(primary_key, plateau_rows)

    b4_partial, b4_full, b4_cells = r123_shared.b4_eth_falsification(eth_primary, inner_val_primary)

    r123_shared.hr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary config, BTC inner-validation")
    b5_pass, b5_cells = b5_fee_tier_with_warmup(build_primary, label, btc, inner_val_primary)

    n_b3_rows = sum(len(v) for v in plateau_rows.values())

    # Turnover / deadband-interaction diagnostic (r123_shared's own named
    # failure risk #4): approx full-period (2017-2022) trade counts by
    # summing the already-computed inner_train + inner_val legs, BTC SPOT --
    # no extra backtest run needed.
    spot_train = next((r for r in inner_train_primary if r["market"] == "spot"), None)
    spot_val = next((r for r in inner_val_primary if r["market"] == "spot"), None)
    turnover_note = None
    if spot_train and spot_val:
        turnover_note = dict(
            cand_trades_approx_full=spot_train["cand_trades"] + spot_val["cand_trades"],
            ctrl_trades_approx_full=spot_train["ctrl_trades"] + spot_val["ctrl_trades"],
        )

    return dict(
        label=label, thresh=thresh, max_discount=max_discount,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells,
        b3_plateau_rows=plateau_rows, b3_detail=b3_detail, b3_pass=b3_pass,
        b4_cells=b4_cells, b4_partial_pass=b4_partial, b4_full_pass=b4_full,
        b5_cells=b5_cells, b5_pass=b5_pass,
        turnover_note=turnover_note,
        n_configs_promotion_bar=6 + n_b3_rows + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    r123_shared.hr("R-123 CONSERVATIVE: ShrinkageKellyMahalanobisBrake -- shrink v4's own "
                  "vote (frac) toward 0 by the Mahalanobis novelty state, pre-scale, pre-deadband")
    print("mechanism: shrink kelly_regime_v4's 3-anchor vote (frac) toward 0 by (1 - shrink),")
    print("where shrink ramps from 0 to max_discount as TODAY's 3-feature market-state vector's")
    print("Mahalanobis distance from its own trailing 730-day (min 180-day) reference distribution")
    print("rises past `thresh`; THEN multiply by scale; THEN deadband. Baker & McHale (2013):")
    print("shrink the edge estimate, not the sized position. See r123_shared.py for full grounding.")

    btc = r123_shared.load_btc()
    max_ts_seen.append(btc.index.max())
    r123_shared.assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {r123_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    r123_shared.hr("PRE-FLIGHT SELF-TESTS (before any Step-0 number is trusted)")
    print("max_discount=0.0 wiring identity (outside the pre-registered grid):")
    identity_ok = self_test_max_discount_zero_identity(btc)
    print(f"  -> max_discount=0.0 identity: {identity_ok}")

    print("\nbuilding full-history novelty state on the FULL non-holdout BTC frame "
          "(used ONLY for the Step-0 gate, per this file's own pre-registration):")
    state_full = novelty_state(btc)
    fv = state_full.first_valid_index()
    days = (fv - btc.index[0]).days if fv is not None else -1
    print(f"  first valid novelty state: {fv}  ({days} days after frame start)  "
          f"CANDIDATE_WARMUP_BARS covers {CANDIDATE_WARMUP_BARS / r123_shared.BARS_PER_DAY:.0f} days")

    if not identity_ok:
        print("\nSELF-TEST FAILURE -- stopping before any Step-0 number is trusted.")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="ABORTED (self-test failure)", max_ts=max(max_ts_seen))

    # ============================================================= STEP 0
    r123_shared.hr("STEP 0 -- 6-CELL (thresh x max_discount) GATE (run BEFORE any Sharpe/compare() number)")
    step0_rows = step0_grid(btc, state_full)
    for r in step0_rows:
        r123_shared.print_step0_report(cell_label(r["thresh"], r["max_discount"]), r)
    print_step0_summary_table(step0_rows)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        r123_shared.hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell passes all four Step-0 clauses. Per this file's own pre-registration,")
        print("this Step-0 table is the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0.")
        n_configs = len(step0_rows)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 grid only)")
        max_ts = max(max_ts_seen)
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {r123_shared.OOS_START}: {max_ts < pd.Timestamp(r123_shared.OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                   n_configs=n_configs, max_ts=max_ts, verdict="NEGATIVE (Step-0 gate)")

    primary_key = (primary_row["thresh"], primary_row["max_discount"])
    is_default = (primary_key == r123_shared.CONS_SELECTION_ORDER[0])
    print(f"\nPRIMARY CONFIG SELECTED (non-degeneracy rule only): thresh={primary_key[0]:g} "
          f"max_discount={primary_key[1]:g}  (bind_frac={primary_row['bind_frac']:.4f}, "
          f"r2_vs_v4={primary_row['r2_vs_v4']:.4f}, r2_vs_vol={primary_row['r2_vs_vol']:.4f}, "
          f"state_cv={primary_row['state_cv']:.4f})")
    print(f"  selection: {'pre-registered CONS_PRIMARY qualified' if is_default else 'CONS_PRIMARY did NOT qualify; next cell in CONS_SELECTION_ORDER chosen'}")

    build_primary = make_build_target(*primary_key)

    r123_shared.hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data)")
    print(f"causal_truncation_probe_series({build_primary.__name__}, btc):")
    try:
        r123_shared.causal_truncation_probe_series(build_primary, btc)
        probe_ok = True
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    print(f"\nCAUSAL PROBE PASS: {probe_ok}")

    eth = r123_shared.load_eth()
    max_ts_seen.append(eth.index.max())
    r123_shared.assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {r123_shared.OOS_START})")

    if not probe_ok:
        r123_shared.hr("CAUSAL PROBE FAILURE -- STOPPING HERE (lookahead is a bug report first)")
        max_ts = max(max_ts_seen)
        n_configs = len(step0_rows)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 grid only; promotion bar not run)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {r123_shared.OOS_START}: {max_ts < pd.Timestamp(r123_shared.OOS_START, tz='UTC')})")
        return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary_row, passed_step0=True,
                   probe_ok=False, n_configs=n_configs, max_ts=max_ts,
                   verdict="NEGATIVE (causal probe failure)")

    bar = run_promotion_bar(primary_key, step0_rows, btc, eth)

    r123_shared.hr("B1 -- inner-validation Sharpe leg, both markets "
                   "(dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    r123_shared.hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  voided={c['voided']}")

    r123_shared.hr("B3 -- plateau: full 6-cell grid at primary selection, inner-validation, both markets")
    r123_shared.print_plateau_table(bar["b3_plateau_rows"])
    print(f"\nB3 (majority of 6-cell grid, both markets, same-signed as primary): {bar['b3_pass']} "
          f"({sum(1 for d in bar['b3_detail'] if d['same_sign_as_primary'])}/{len(bar['b3_detail'])})")

    r123_shared.hr("B4 -- ETH falsification (pre-registered as this round's one falsification test)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  ETH d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  same_sign_as_btc={c['same_sign_as_btc']}")
    print(f"B4 FULL PASS (both markets): {bar['b4_full_pass']}")
    print(f"B4 PARTIAL PASS (at least one market): {bar['b4_partial_pass']}")

    r123_shared.hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  @0.40% d_sharpe={c['d_sharpe']:+.4f}  "
              f"@0.40% boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"@0.10% boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    r123_shared.hr("TURNOVER / DEADBAND-INTERACTION DIAGNOSTIC (r123_shared docstring risk #4)")
    if bar["turnover_note"]:
        tn = bar["turnover_note"]
        print(f"  BTC SPOT, approx full 2017-2022 period (inner_train + inner_val summed): "
              f"candidate trades={tn['cand_trades_approx_full']}  "
              f"control (v4) trades={tn['ctrl_trades_approx_full']}")
        print("  (reference point named in r123_shared.py's own docstring: v4's unmodified "
              "143-trade full-period baseline)")

    r123_shared.hr("VERDICT")
    print(f"causal probe = {probe_ok}   B1 = {bar['b1_pass']}   B2 = diagnostic-only   "
          f"B3 = {bar['b3_pass']}   B4(full) = {bar['b4_full_pass']}   "
          f"B4(partial) = {bar['b4_partial_pass']}   B5 = {bar['b5_pass']}")
    all_gates_pass = probe_ok and bar["b1_pass"] and bar["b3_pass"] and bar["b4_full_pass"] and bar["b5_pass"]
    verdict = "PROMOTE-candidate" if all_gates_pass else "NEGATIVE"
    print(f"\nALL GATING CLAUSES PASS (causal AND B1 AND B3 AND B4-full AND B5): {all_gates_pass}")
    print(f"VERDICT: {verdict}")
    if not all_gates_pass:
        failed = [name for name, ok in (
            ("causal probe", probe_ok), ("B1", bar["b1_pass"]), ("B3", bar["b3_pass"]),
            ("B4 (full)", bar["b4_full_pass"]), ("B5", bar["b5_pass"]),
        ) if not ok]
        print(f"Reason(s): {', '.join(failed)}")

    n_configs = len(step0_rows) + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"({len(step0_rows)} Step-0 grid + 6 primary compare_with_warmup() + "
          f"{sum(len(v) for v in bar['b3_plateau_rows'].values())} B3 full-grid rows "
          f"[2 reused from primary + rest fresh] + 2 B5 fee-tier)")
    max_ts = max(max_ts_seen)
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {r123_shared.OOS_START}: {max_ts < pd.Timestamp(r123_shared.OOS_START, tz='UTC')})")
    print("NO bar at or after 2023-01-01 was ever read by this file, regardless of outcome.")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary_row, passed_step0=True,
               probe_ok=probe_ok, promotion_bar=bar, verdict=verdict, n_configs=n_configs,
               max_ts=max_ts)


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    """Mirrors r123_shared.py's own convention: causal_truncation_probe_series
    on this file's FULL composed candidate-build function (features -> distance
    -> state -> conservative_target), on synthetic data, run at import time --
    before any real-data Step-0 or promotion-bar number is trusted."""
    idx = pd.date_range("2017-01-01", periods=250_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(123)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                        "close": close, "volume": rng.lognormal(0, 0.5, len(idx))},
                       index=idx)

    probe_fn = make_build_target(*r123_shared.CONS_PRIMARY)
    assert r123_shared.causal_truncation_probe_series(probe_fn, df)

    cand_path = probe_fn(df)
    assert np.isfinite(cand_path).sum() > 1000
    v4_path = r123_shared.v4_target(df)
    m = np.isfinite(cand_path) & np.isfinite(v4_path)
    assert np.all(np.abs(cand_path[m]) <= np.abs(v4_path[m]) + 1e-9), \
        "shrinkage-kelly target exceeds v4's own magnitude somewhere -- shrink math is wrong"

    # max_discount=0.0 wiring identity, on synthetic data too.
    zero_disc = build_target(df, thresh=0.9, max_discount=0.0)
    assert np.allclose(zero_disc, v4_path, equal_nan=True), \
        "max_discount=0.0 does not reproduce v4_target exactly on synthetic data"


_self_test()


if __name__ == "__main__":
    main()
