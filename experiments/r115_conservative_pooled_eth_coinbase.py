#!/usr/bin/env python
"""R-115 CONSERVATIVE branch: ``PooledRefKnnNoveltyBrakeKellyV4_CoinbaseETH``
-- R-112 novel branch's exact pooled-reference kNN distributional-novelty
discount on ``kelly_regime_v4``'s own ``frac * scale`` product, with EXACTLY
ONE change from R-112 novel's own construction: the ETH falsification
instrument's data SOURCE, from `experiments/r109_shared.py`'s ``load_eth()``
(Bitfinex, ends 2019-12-31) to ``experiments/r115_conservative_shared.py``'s
new ``load_eth_coinbase()`` (Coinbase USD spot, the same exchange
``r63_shared.UNIVERSE_6``'s own six panels already use, confirmed by direct
read to span 2019-03-14 through the present -- see that module's own
docstring for the full read-first verification). Nothing else changes: the
5-feature panel (``r109_shared.NOVEL_FEATURE_BUILDERS``), the pooled kNN
mechanism (``r112_shared.rolling_knn_distance_pooled``, k=10,
refit_every=30, CORAL-standardized against ``UNIVERSE_6``), the Step-0 grid,
and the full B1-B5 promotion bar are all held byte-identical to R-112 novel
branch's own file (``experiments/r112_novel_pooled_reference_knn.py``),
which this file is structurally a copy of with that one substitution made.

WHY THIS ROUND, AND WHY NOW: R-112's own closing verdict (docs/LEDGER.md,
R-112 section) diagnosed that its novel branch's B4 (ETH) falsification test
never actually engaged the pooled-reference mechanism it was built to test.
``UNIVERSE_6``'s six panels all start 2020-01-02; R-109/R-112's shared
``load_eth()`` reads a Bitfinex series ending 2019-12-31 -- zero calendar
overlap, so every pooled-kNN refit inside ETH's own evaluation window found
every pool instrument's window empty and silently fell back to the
single-asset construction R-109 already tested and failed B4 with. R-112's
own novel-branch B4 numbers came out bit-for-bit identical to R-109 novel's
own B4 numbers as direct proof the pool never engaged. R-112's own verdict
named the fix explicitly and left it undone, as a "genuine, disclosed change
to the falsification instrument itself... flagged here as the concrete next
step rather than attempted now." This file is exactly that next step, and
nothing else.

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path) -- the same constraint as R-28/retracted, R-87, R-104, R-105 x2,
R-106 x2, R-109 x2, R-112 x2. This is NOT a new ERR-axis mechanism: it is a
data-plumbing fix to R-112 novel branch's own falsification instrument, so
that the B4 test R-112 pre-registered is finally the test R-112 actually
ran. No mechanism, feature, metric, or gate is touched.

**Not a duplicate of:**
- R-112 novel (``experiments/r112_novel_pooled_reference_knn.py``): that
  file's own B4 result never engaged the pool (proven by its bit-for-bit
  identity with R-109 novel's own B4 numbers, both built from the same
  Bitfinex-sourced, pool-non-overlapping ETH frame). This file re-runs the
  IDENTICAL construction with a genuinely overlapping ETH reference pool --
  the first time this round's own hypothesis (a multi-asset pool "might
  close the novel branch's B4 gap") has actually been tested end to end.
- R-112 conservative (``experiments/r112_conservative_returnspace_knn.py``):
  that branch changed the ``anchor_disp`` FEATURE to a return-space
  analogue, single-asset reference, unrelated to this file's change (data
  source of a falsification instrument, pooled reference held fixed at
  R-112 novel's own construction).
- Every other ERR-axis round in this ledger's sub-line (R-28, R-87, R-104,
  R-105, R-106, R-109, R-112): none of them ever swapped ETH's own data
  source; every one either built a new mechanism or changed a feature/
  reference-set choice on a fixed mechanism. This file changes neither --
  it changes which FILE ``load_eth()``-equivalent reads from, nothing about
  what is computed from whatever it reads.

This module's own read-only helper (``experiments/r115_conservative_shared.
py``) does not edit ``r109_shared.py``, ``r112_shared.py``, or
``r63_shared.py`` -- consistent with this project's own established
parallelism convention (copy, do not modify, a frozen prior-round shared
file).

============================================================================
PRE-REGISTERED DECISION RULE (frozen here, BEFORE any real-market number is
computed below -- this is what "pre-registered" means in this ledger).
Identical in shape to R-105-onward's standard bar, and identical in EVERY
particular (bootstrap construction, seeds, thresholds, gate code) to R-109/
R-112's own promotion bar, imported unmodified from ``r105_shared``/
``r112_shared``:
============================================================================

  B1 (gating). ``b1_from_inner_val`` on the primary cell's inner-validation
      rows (2021-01-01 -> 2022-12-31, BTC), BOTH markets (spot, futures_5x):
      passes a market iff d_sharpe > +0.2 (``SHARPE_NOISE_FLOOR``) OR the
      paired stationary-block-bootstrap CI's lower bound for the log-growth
      differential excludes zero favourably (``boot_lo > 0``). Fail if
      either market does not pass.
  B2 (diagnostic, NON-gating). ``b2_diagnostic`` -- risk-matched (exposure-
      and vol-ratio within [0.9, 1.1]) drawdown improvement. Reported, never
      gates the verdict.
  B3 (gating, plateau). The FULL 6-cell Step-0 (thresh, max_discount) grid's
      own inner-validation numbers, both markets (12 cells total, 2 of them
      the primary cell's own rows reused directly) -- PASS requires a
      directionally consistent (same-sign d_sharpe) majority (>= 6 of 12,
      per ``run_b3_full_grid``'s own ``>= len/2`` rule, unmodified from
      R-112 novel).
  B4 (gating) -- THE POINT OF THIS ROUND. ETH falsification, now genuinely
      testable with a pool that actually overlaps ETH's own evaluation
      window: ``b4_eth_falsification`` on ``compare(..., eth=
      load_eth_coinbase())``. FULL pass requires BOTH markets same-sign as
      BTC's own primary-cell inner_val result.
  B5 (gating, cost robustness). ``b5_fee_tier`` at the 0.40% taker tier,
      primary cell, BTC inner_val, both markets -- no sign reversal vs. the
      standard-tier result.

PROMOTION requires the causal-truncation probe AND B1 (both markets) AND B3
(plateau majority) AND B4 (full, both markets) AND B5 (both markets) all to
hold. B2 is diagnostic-only and never gates. Default, absent all of the
above: NEGATIVE. No threshold or decision rule is changed after seeing any
real-market number -- see this file's own "BUGS FOUND, AND WHEN" section
near the bottom of ``main()``'s printed output for an explicit disclosure of
what was and was not fixed before any such number was read.

``pool_dailies`` (the six ``UNIVERSE_6`` instruments' own daily feature
panels, via ``r112_shared.load_pool_daily_panels()``) is built EXACTLY ONCE
at the top of ``main()`` and threaded through every Step-0/B3/B5 cell as a
fixed argument -- never rebuilt per config, never a function of the target
``df`` argument being probed/sliced, identical convention to R-112 novel.

``k`` AND ``refit_every`` -- held at R-109/R-112's own pre-registered
defaults, ``k=10`` and ``refit_every=30`` (verified programmatically below
against the live ``rolling_knn_distance_pooled`` signature). No sweep is
performed; the only Step-0 degrees of freedom are ``(thresh, max_discount)``,
identical 3x2=6-cell grid to R-109/R-112 (``STEP0_THRESH_GRID x
STEP0_MAXD_GRID``, same ``SELECTION_ORDER``).

CAUSAL SAFETY: ``causal_truncation_probe_series`` applied to this file's own
``build_target`` (features -> pooled kNN distance -> percentile-rank state
-> discount -> ``v4_target * (1 - discount)``), with ``pool_dailies`` built
once and closed over BEFORE the probe runs -- identical pattern to R-112
novel. Run on BTC's full non-holdout frame, BEFORE the Step-0 grid is scored
and well before any inner-validation/ETH performance number is computed.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid, 3 thresh x 2 max_discount) + 6 (primary cell's full
``compare()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets, now against the Coinbase ETH frame) + 12 (B3's
full 6-cell grid x 2 markets, 2 of the 12 reused directly from the primary
``compare()``'s own inner_val rows, 10 freshly computed) + 2 (B5's 0.40% fee
tier, 2 markets) = 26 total, identical count to R-112 novel branch's own
file (this round changes a data SOURCE, not the grid). IF Step-0 finds no
qualifying cell, this file stops after the 6 Step-0 cells (6 total). No
``k``/``refit_every`` sweep is performed, so it adds 0 configurations to
either count.

USAGE
-----
    python experiments/r115_conservative_pooled_eth_coinbase.py
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r112_shared import (  # noqa: E402
    BASELINE_WINDOW_DAYS,
    BIND_FRAC_THRESH,
    CV_KILL_THRESH,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    MIN_REF_DAYS,
    NOVEL_FEATURE_BUILDERS,
    OOS_START,
    PRIMARY_MAXD,
    PRIMARY_THRESH,
    R2_VS_V4_THRESH,
    R2_VS_VOL_THRESH,
    SELECTION_ORDER,
    SPOT,
    STEP0_MAXD_GRID,
    STEP0_THRESH_GRID,
    UNIVERSE_6,
    apply_discount,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    build_daily_features,
    causal_rolling_percentile_rank,
    causal_truncation_probe_series,
    compare,
    hr,
    inner_val_rows,
    load_btc,
    load_pool_daily_panels,
    print_plateau_table,
    print_rows,
    rolling_knn_distance_pooled,
)

# THIS ROUND'S ONE CHANGE: ETH via Coinbase spot instead of r109_shared's
# Bitfinex-sourced load_eth(). See r115_conservative_shared.py's own module
# docstring for the direct-read verification of the overlap claim.
from experiments.r115_conservative_shared import load_eth_coinbase  # noqa: E402

# ---------------------------------------------------------- pre-registered
K = 10             # rolling_knn_distance_pooled's own default, held fixed -- verified below
REFIT_EVERY = 30   # rolling_knn_distance_pooled's own default, held fixed -- verified below

_sig = inspect.signature(rolling_knn_distance_pooled).parameters
assert _sig["k"].default == K, ("K does not match rolling_knn_distance_pooled's "
                                 f"own default ({_sig['k'].default}) -- pre-registration text is stale")
assert _sig["refit_every"].default == REFIT_EVERY, (
    "REFIT_EVERY does not match rolling_knn_distance_pooled's own default "
    f"({_sig['refit_every'].default}) -- pre-registration text is stale")


# ================================================================== (1)
# The mechanism itself: 5-feature panel -> POOLED kNN distance (target +
# six UNIVERSE_6 instruments, each CORAL-standardized) -> percentile-rank
# state -> discount on v4's own UNCHANGED frac*scale. `pool_dailies` is
# always a fixed argument/closure, never rebuilt here and never a function
# of `df`. BYTE-IDENTICAL to r112_novel_pooled_reference_knn.py's own
# functions below -- this round changes only which ETH frame gets passed in
# from main(), never how the mechanism itself is composed.
# ==================================================================

def compute_full_state(df: pd.DataFrame, pool_dailies: dict[str, pd.DataFrame],
                        k: int = K, refit_every: int = REFIT_EVERY) -> pd.Series:
    """features -> pooled kNN distance -> causal percentile-rank state, over
    whatever frame `df` is (the caller decides how much history it
    contains). `pool_dailies` is a fixed reference pool, independent of
    `df` -- this is what keeps the function causal in `df` alone."""
    daily = build_daily_features(df, NOVEL_FEATURE_BUILDERS)
    dist = rolling_knn_distance_pooled(daily, pool_dailies, k=k, refit_every=refit_every)
    return causal_rolling_percentile_rank(dist)


def build_target(df: pd.DataFrame, pool_dailies: dict[str, pd.DataFrame],
                  thresh: float = PRIMARY_THRESH, max_discount: float = PRIMARY_MAXD,
                  k: int = K, refit_every: int = REFIT_EVERY) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount), where
    discount is driven by the pooled-kNN novelty state built from `df` and
    the fixed `pool_dailies` closure. Directly usable as a `TargetStrategy`
    candidate on any window (inner_train, inner_val, eth_replication, or a
    truncated probe frame) once `pool_dailies` is bound."""
    state = compute_full_state(df, pool_dailies, k=k, refit_every=refit_every)
    return apply_discount(df, state, thresh, max_discount)


def make_build_target(pool_dailies: dict[str, pd.DataFrame], thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, pool_dailies, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"knn_pooled_novelty_brake_ethcb_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: STEP0_THRESH_GRID x STEP0_MAXD_GRID, scored via step0_gate on
# BTC inner-train, state computed over the FULL non-holdout BTC frame with
# the pooled kNN distance and the fixed `pool_dailies` closure. Identical to
# R-112 novel -- BTC's own data source is unchanged this round.
# ==================================================================

def step0_grid(btc: pd.DataFrame, pool_dailies: dict[str, pd.DataFrame]) -> tuple[list[dict], pd.Series]:
    from experiments.r112_shared import step0_gate  # local import, matches r109/r112_shared's own idiom
    state = compute_full_state(btc, pool_dailies)
    df_inner_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    rows = []
    for thresh in STEP0_THRESH_GRID:
        for maxd in STEP0_MAXD_GRID:
            gate = step0_gate(df_inner_train, state, thresh, maxd)
            rows.append(dict(thresh=thresh, max_discount=maxd, **gate))
    return rows, state


def select_primary(rows: list[dict]) -> dict | None:
    by_key = {(r["thresh"], r["max_discount"]): r for r in rows}
    for key in SELECTION_ORDER:
        r = by_key.get(key)
        if r is not None and r["passed"]:
            return r
    return None


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars, state built from k={K}, refit_every={REFIT_EVERY}d, "
          f"BASELINE_WINDOW_DAYS={BASELINE_WINDOW_DAYS}, MIN_REF_DAYS={MIN_REF_DAYS}, "
          f"POOLED reference: target + {len(UNIVERSE_6)} UNIVERSE_6 instruments, CORAL-standardized)")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r2_vs_v4 < {R2_VS_V4_THRESH} "
          f"AND r2_vs_vol < {R2_VS_VOL_THRESH} AND state_cv >= {CV_KILL_THRESH:.0%}")
    hdr = (f"{'thresh':>7s} {'max_d':>6s} {'bind_frac':>10s} {'r2_vs_v4':>9s} "
           f"{'r2_vs_vol':>9s} {'state_cv':>9s} {'passed':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = "  <- SELECTION_ORDER[0]" if (r["thresh"], r["max_discount"]) == SELECTION_ORDER[0] else ""
        print(f"{r['thresh']:7.2f} {r['max_discount']:6.2f} {r['bind_frac']:10.4f} "
              f"{r['r2_vs_v4']:9.4f} {r['r2_vs_vol']:9.4f} {r['state_cv']:9.4f} "
              f"{'YES' if r['passed'] else 'no':>7s}{tag}")


# ================================================================== (3)
# B3 plateau: the full 6-cell (thresh, max_discount) grid's own
# inner-validation numbers (both markets), primary cell's 2 rows reused
# directly from its own compare(). Identical to R-112 novel -- B3 never
# reads ETH at all (inner_val is BTC-only), so this round's change cannot
# touch B3's own numbers.
# ==================================================================

def run_b3_full_grid(step0_rows: list[dict], primary_key: tuple[float, float],
                      inner_val_primary: list[dict], btc: pd.DataFrame,
                      pool_dailies: dict[str, pd.DataFrame]) -> tuple[dict, bool]:
    plateau_rows: dict[tuple[float, float], list[dict]] = {}
    for r in step0_rows:
        key = (r["thresh"], r["max_discount"])
        if key == primary_key:
            plateau_rows[key] = [dict(market=c["market"], d_sharpe=c["d_sharpe"], d_dd=c["d_dd"],
                                       exposure_ratio=c["exposure_ratio"], vol_ratio=c["vol_ratio"],
                                       risk_matched=c["risk_matched"],
                                       boot_d_loggrowth=c["boot_d_loggrowth"], boot_lo=c["boot_lo"],
                                       boot_hi=c["boot_hi"], excludes_zero=c["excludes_zero"])
                                  for c in inner_val_primary]
        else:
            bf = make_build_target(pool_dailies, *key)
            label = f"knn_pooled_novelty_brake_ethcb_t{key[0]:g}_m{key[1]:g}"
            plateau_rows[key] = inner_val_rows(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


# ================================================================== (4)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# above), B4 (gating falsification -- THIS ROUND'S ONE CHANGE: `eth` is now
# the Coinbase-sourced frame), B5 (gating fee tier).
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                       btc: pd.DataFrame, eth: pd.DataFrame,
                       pool_dailies: dict[str, pd.DataFrame]) -> dict:
    thresh, maxd = primary_key
    build_primary = make_build_target(pool_dailies, thresh, maxd)
    label = f"knn_pooled_novelty_brake_ethcb_t{thresh:g}_m{maxd:g}"

    hr(f"PROMOTION BAR -- PRIMARY CELL thresh={thresh:g}, max_discount={maxd:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    print("  (eth_replication now runs on the Coinbase-sourced ETH frame -- this round's one change)")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                    markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3_full_grid(step0_rows, primary_key, inner_val_primary, btc, pool_dailies)
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)
    b5_pass, b5_cells = b5_fee_tier(build_primary, label, btc, inner_val_primary)

    all_pass = b1_pass and b3_pass and b4_full and b5_pass

    return dict(
        label=label, thresh=thresh, max_discount=maxd, compare_rows=rows,
        inner_val_primary=inner_val_primary, eth_primary=eth_primary,
        b1_pass=b1_pass, b1_cells=b1_cells,
        b2_pass=b2_pass, b2_cells=b2_cells,
        b3_pass=b3_pass, b3_rows=b3_rows,
        b4_partial=b4_partial, b4_full=b4_full, b4_cells=b4_cells,
        b5_pass=b5_pass, b5_cells=b5_cells,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 12 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-115 CONSERVATIVE: PooledRefKnnNoveltyBrakeKellyV4_CoinbaseETH -- R-112 novel's "
       "exact pooled-reference kNN novelty brake, ETH falsification instrument re-sourced "
       "from Bitfinex to Coinbase spot (the one, disclosed change)")
    print("mechanism: identical to r112_novel_pooled_reference_knn.py's 5-feature OHLCV-only daily")
    print("market-state panel (log_vol, anchor_disp, kurtosis, volume_z, skew) -> mean Euclidean")
    print("distance (per-feature standardized) to the k nearest neighbours in a POOLED reference")
    print("SET (target's own trailing 730 days UNION UNIVERSE_6's contemporaneous trailing 730")
    print("days, each CORAL-standardized) -> causal rolling percentile-rank state in [0,1] ->")
    print("linear discount on v4's UNCHANGED frac*scale product. THE ONE CHANGE: ETH is now read")
    print("via r115_conservative_shared.load_eth_coinbase() (Coinbase USD spot,")
    print("data/ethusd_coinbase_spot_5m.csv.gz) instead of r109_shared.load_eth() (Bitfinex,")
    print("data/ethusd_bitfinex_5m.csv.gz, ends 2019-12-31 -- zero overlap with UNIVERSE_6's own")
    print("2020-01-02 start, which is why R-109/R-112's own B4 numbers on the novel-branch")
    print("construction were bit-for-bit identical: the pool never engaged on ETH). Full")
    print("grounding, non-duplication argument, and pre-registered decision rule in this file's")
    print("own module docstring; the pooled-reference mechanism's own grounding is in")
    print("r112_shared.py's module docstring, unmodified.")
    print(f"\nk={K}, refit_every={REFIT_EVERY}d  (both = rolling_knn_distance_pooled's own defaults, "
          f"verified programmatically above; no sweep performed -- see banner reasoning)")
    print(f"STEP0_THRESH_GRID={STEP0_THRESH_GRID}  STEP0_MAXD_GRID={STEP0_MAXD_GRID}  "
          f"({len(STEP0_THRESH_GRID) * len(STEP0_MAXD_GRID)} cells)")
    print(f"SELECTION_ORDER={SELECTION_ORDER}")
    print(f"POOL: UNIVERSE_6={UNIVERSE_6}")

    hr("BUILDING SIX-INSTRUMENT POOL DAILY FEATURE PANELS (built ONCE, reused as a fixed "
       "closure for every Step-0/B3/B5 cell below -- never rebuilt per config)")
    pool_dailies = load_pool_daily_panels()
    for name, panel in pool_dailies.items():
        assert_no_holdout(panel, f"main(): pool instrument {name}")
        max_ts_seen.append(panel.index.max())
        print(f"  {name:>6s}: {len(panel):,} daily rows, {panel.index[0].date()} -> "
              f"{panel.index[-1].date()}  (< {OOS_START})")
    assert "BTC" not in pool_dailies and "ETH" not in pool_dailies

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ================================================== CAUSAL SAFETY FIRST
    hr("CAUSAL TRUNCATION PROBE (composed build_target at the pre-registered primary "
       "(thresh, max_discount), real BTC data, fixed pool_dailies closure, run BEFORE "
       "Step-0 or any inner-val/ETH number)")
    probe_fn = make_build_target(pool_dailies, PRIMARY_THRESH, PRIMARY_MAXD)
    print(f"causal_truncation_probe_series({probe_fn.__name__}, btc):")
    try:
        probe_ok = causal_truncation_probe_series(probe_fn, btc)
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    print(f"\nCAUSAL SAFETY (truncation probe) PASS: {probe_ok}")

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY / NON-DUPLICATION KILL SWITCH "
       "(run BEFORE any inner-validation Sharpe/PnL number)")
    step0_rows, state = step0_grid(btc, pool_dailies)
    n_bars_inner_train = int(np.sum((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                                     (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))))
    print_step0_table(step0_rows, n_bars_inner_train)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train (BTC's own data")
        print("source is unchanged from R-112 novel, so this would mirror R-112 novel's own Step-0")
        print("table exactly): the pooled-reference kNN novelty discount is either a near-total")
        print("no-op, a near-exact rescale of v4's own path, a relabelled volatility rescale, or")
        print("degenerate everywhere on the pre-registered grid. Per this file's own")
        print("pre-registration, this Step-0 table (plus the causal-safety probe above) is the")
        print("branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0. No promotion-bar")
        print("code runs, and no inner-validation Sharpe/PnL number or ETH bar is ever read.")

        hr("VERDICT")
        print("Step-0 (6-cell thresh x max_discount grid): FAIL (no cell qualifies)")
        print(f"causal truncation probe: {probe_ok}")
        print("B1: NOT COMPUTED (Step-0 kill switch)")
        print("B2: NOT COMPUTED (Step-0 kill switch)")
        print("B3: NOT COMPUTED (Step-0 kill switch)")
        print("B4: NOT COMPUTED (Step-0 kill switch)")
        print("B5: NOT COMPUTED (Step-0 kill switch)")
        print("VERDICT: NEGATIVE (Step-0 kill switch)")

        n_configs = len(step0_rows)
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated (total): {n_configs} (6 Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch (BTC, all six pool instruments; "
              f"ETH never read -- Step-0 kill switch stopped before B4): {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                    probe_ok=probe_ok, n_configs=n_configs, max_ts=max_ts,
                    verdict="NEGATIVE (Step-0 kill switch)")

    primary_key = (primary_row["thresh"], primary_row["max_discount"])
    is_selection0 = (primary_key == SELECTION_ORDER[0])
    print(f"\nPRIMARY CELL SELECTED (Step-0 non-degeneracy rule only): "
          f"thresh={primary_key[0]:g}, max_discount={primary_key[1]:g}  "
          f"(bind_frac={primary_row['bind_frac']:.4f}, r2_vs_v4={primary_row['r2_vs_v4']:.4f}, "
          f"r2_vs_vol={primary_row['r2_vs_vol']:.4f}, state_cv={primary_row['state_cv']:.4f})")
    print(f"  selection: {'SELECTION_ORDER[0] qualified' if is_selection0 else 'SELECTION_ORDER[0] did NOT qualify; next qualifying cell in SELECTION_ORDER chosen'}")

    eth = load_eth_coinbase()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth (Coinbase spot)")
    print(f"\nETH (Coinbase spot, truncated < {OOS_START} -- THIS ROUND'S ONE CHANGE): "
          f"{len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")
    print(f"  UNIVERSE_6 pool coverage starts 2020-01-01; ETH (Coinbase) starts "
          f"{eth.index[0].date()} -- genuine pool overlap for 2020-01-01 -> {eth.index[-1].date()} "
          f"(the majority of this ~3.8y eth_replication window; the leading ~9.5mo, "
          f"2019-03-14->2019-12-31, predates UNIVERSE_6 and falls back to the single-asset "
          f"construction for that stretch only), unlike R-109/R-112's own Bitfinex-sourced ETH "
          f"(ended 2019-12-31, ZERO overlap with UNIVERSE_6, every single day).")

    bar = run_promotion_bar(primary_key, step0_rows, btc, eth, pool_dailies)

    hr("B1 -- inner-validation, both markets (dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hr("B3 -- plateau: FULL 6-cell (thresh, max_discount) Step-0 grid, inner-validation, both markets")
    print_plateau_table(bar["b3_rows"])
    print(f"\nB3 (directionally consistent majority across the 12-cell grid): {bar['b3_pass']}")

    hr("B4 -- ETH falsification (pre-registered) -- Coinbase-sourced ETH, THIS ROUND'S OWN POINT")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PARTIAL PASS (>=1 market): {bar['b4_partial']}   B4 FULL PASS (both markets): {bar['b4_full']}")

    hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier d_sharpe={c['d_sharpe']:+.4f}  "
              f"standard-fee d_sharpe={c['base_d_sharpe']:+.4f}  "
              f"fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    hr("VERDICT")
    print(f"causal safety (truncation probe): {probe_ok}")
    print(f"B1={bar['b1_pass']}  B2=diagnostic-only  B3={bar['b3_pass']}  "
          f"B4_full={bar['b4_full']}  B5={bar['b5_pass']}")
    all_applicable_pass = (probe_ok and bar["b1_pass"] and bar["b3_pass"] and
                            bar["b4_full"] and bar["b5_pass"])
    verdict = "PROMOTE-candidate" if all_applicable_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_applicable_pass}")
    print(f"VERDICT: {verdict}")
    if not probe_ok:
        print("NOTE: verdict driven (at least in part) by a causal-safety check failure -- "
              "a lookahead is a bug report first, per docs/ROUTINE.md's own precedence.")
    if all_applicable_pass:
        print("\nNOTE: this would be the FIRST ERR-axis construction in this project's ledger to")
        print("clear B1+B3+B4+B5, after eleven prior attempts (R-28/retracted, R-87, R-104, R-105 x2,")
        print("R-106 x2, R-109 x2, R-112 x2).")

    n_configs = len(step0_rows) + bar["n_configs_promotion_bar"]
    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(6 Step-0 grid + 6 primary-cell compare() + 12 B3 plateau "
          f"[6 (thresh,max_discount) cells x 2 markets, 2 reused from primary] + "
          f"2 B5 fee-tier; k/refit_every not swept, adds 0)")
    print(f"max timestamp read anywhere in this branch (BTC, ETH via Coinbase spot, "
          f"all six pool instruments): {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, pool_dailies=pool_dailies, step0_rows=step0_rows, state=state,
                primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
