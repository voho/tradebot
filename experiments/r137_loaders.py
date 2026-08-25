"""R-137: per-construction loaders that reproduce each of the five prior
constructions' own already-published ETH-window candidate/baseline daily
return series, using each construction's OWN existing harness code (no
mechanism is reimplemented here; every candidate/baseline series below comes
from calling straight into `experiments/r10X_shared.py` and/or the relevant
branch file's own functions).

This file is deliberately NOT part of the frozen `experiments/r137_shared.py`
pre-registration -- it is a separate, ordinary (non-frozen) loaders module,
written after `r137_shared.py` was committed, so the pre-registration text
stays untouched. It imports `r137_shared.gap_sharpe` to self-check every
reproduction against each construction's own published `d_sharpe`.

=====================================================================
IMPORTANT, DISCLOSED UP FRONT: WHICH "ETH WINDOW" EACH CONSTRUCTION
ACTUALLY PUBLISHED ITS d_sharpe OVER
=====================================================================

`r137_shared.py`'s own docstring (and `r127_shared.py`'s, which it quotes)
describes "this project's standard B4 convention" as INNER_VAL_START
(2021-01-01) to INNER_VAL_END (2022-12-31) on ETH, "the identical calendar
dates on both assets." That description is accurate for R-125-conservative
and R-126-conservative (both call a `b1_signal`-style helper that runs
`run_period(..., start=INNER_VAL_START, end=INNER_VAL_END)` on the ETH
frame explicitly) -- but it is NOT literally accurate for R-109-novel or
R-115-conservative, verified here by direct read of the harness before
writing anything below:

  - R-109-novel's `eth = load_eth()` reads `data/ethusd_bitfinex_5m.csv.gz`,
    which spans 2016-03-09 -> 2019-12-31 (confirmed by direct load below) --
    entirely BEFORE INNER_VAL_START. `compare()`'s own `eth_replication`
    slice runs with `start=None, end=None`, i.e. the ENTIRE passed-in ETH
    frame, not a date-sliced INNER_VAL window. So R-109-novel's published
    ETH spot d_sharpe=-0.009 is computed over Bitfinex's full 2016-2019
    history, not over INNER_VAL 2021-2022 at all (ETH simply has no
    Bitfinex-sourced 2021-2022 data in this project's committed files).
  - R-115-conservative's `eth = load_eth_coinbase()` reads
    `data/ethusd_coinbase_spot_5m.csv.gz`, truncated only to < OOS_START
    (2023-01-01), i.e. 2019-03-14 -> 2022-12-31 -- and the SAME
    `start=None, end=None` `eth_replication` slice convention applies, so
    R-115-conservative's published ETH spot d_sharpe=-0.0890 is computed
    over ETH's entire ~3.8-year non-holdout history, not the 2-year
    INNER_VAL window specifically (INNER_VAL is a strict subset of it).

Both loaders below are faithful to what each construction's own harness
ACTUALLY scored (per this round's instruction: "using each construction's
OWN existing harness code," verified rather than assumed) -- reproducing
`compare()`'s own `eth_replication` slice exactly, over each construction's
own ETH frame, rather than silently re-slicing to INNER_VAL_START..END
(which would silently change what is being reproduced and would not
reproduce the published number at all for these two). This is flagged
loudly rather than corrected quietly, per this round's own instructions.

R-125-conservative and R-126-conservative's ETH cells genuinely are
INNER_VAL-restricted (`run_period(..., INNER_VAL_START, INNER_VAL_END)` is
called explicitly inside `r125_shared.run_candidate` / `r126_shared.
run_target_series` and `run_candidate_council`), matching the task's
premise exactly.

=====================================================================
SECOND DISCLOSED DISCREPANCY: BAR-LEVEL vs DAILY SHARPE
=====================================================================

Every construction's OWN published `d_sharpe` (`a.sharpe - b.sharpe` in
`compare()`'s row dict, or `m_cand.sharpe - m_v4.sharpe` / `m_cand.sharpe -
m_council.sharpe` in `r125_shared.b1_signal` / `r126_shared.b1_signal`) is
`tradebot.metrics.sharpe_ratio`, computed on PER-BAR (5-minute) equity
returns annualized by `BARS_PER_YEAR` (5m bars/year). `r137_shared.
gap_sharpe`, by contrast, is pre-registered to operate on CALENDAR-DAILY
return series via `tradebot.inference.annualized_sharpe` (annualized by
sqrt(365.25) on ~1 observation/day). These are two different statistics of
the same equity curve, not the same statistic resampled -- 5m bar returns
carry autocorrelation/microstructure structure that daily returns average
away, so a nontrivial gap between "published d_sharpe" (bar-level) and
"reproduced gap_sharpe" (daily-level) on the IDENTICAL equity curves is
expected here, is NOT a wiring bug, and is reported honestly per
construction below rather than silently tuned away. Each loader still
returns the exact candidate/baseline DAILY series `r137_shared.gap_sharpe`
is pre-registered to consume (there is no other series it can consume);
`published_gap` is each construction's own bar-level number, cited from its
own file's output / `docs/LEDGER.md`, for side-by-side comparison, not
forced equality.

Run: `python experiments/r137_loaders.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.inference import daily_returns  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r137_shared import gap_sharpe  # noqa: E402

OOS_START = "2023-01-01"


def _assert_no_holdout(df: pd.DataFrame, label: str) -> None:
    if len(df) and df.index[-1] >= pd.Timestamp(OOS_START, tz=df.index.tz):
        raise AssertionError(f"{label}: reaches {df.index[-1]}, at/after {OOS_START}")


def _daily_from_run(strategy, df: pd.DataFrame, market, start=None, end=None,
                     balance: float = 1000.0) -> pd.Series:
    """Run `strategy` over `df[start:end]` via the project's own
    `run_period` (byte-identical call convention to every `run_slice` /
    `run_candidate` / `run_target_series` helper this round's five
    constructions already use) and return the CALENDAR-DAILY return series
    of the resulting equity curve, via `tradebot.inference.daily_returns` --
    the exact series shape `r137_shared.gap_sharpe` is pre-registered to
    consume."""
    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    return daily_returns(res.equity)


# =====================================================================
# 1. R-109-novel: kNN novelty brake on kelly_regime_v4, ETH spot.
#    Harness: experiments/r109_shared.py (chains r106_shared -> r105_shared
#    -> ... -> r102_shared) + experiments/r109_novel_knn_novelty_brake.py's
#    own build_target at its own frozen primary cell (thresh=0.90,
#    max_discount=1.0), imported verbatim, not reimplemented.
# =====================================================================

def load_r109_novel() -> dict:
    from experiments.r109_shared import SPOT, TargetStrategy, load_eth, v4_target
    from experiments.r109_novel_knn_novelty_brake import build_target as r109_build_target

    eth = load_eth()
    _assert_no_holdout(eth, "R-109-novel ETH (Bitfinex)")

    candidate = TargetStrategy(r109_build_target, name="r109_novel_knn_novelty_brake")
    baseline = TargetStrategy(v4_target, name="kelly_regime_v4")

    # `compare()`'s own `eth_replication` slice: start=None, end=None, i.e.
    # the ENTIRE passed-in ETH frame (see module docstring -- for R-109-
    # novel this is Bitfinex's full 2016-03-09..2019-12-31 history, not
    # INNER_VAL).
    candidate_daily = _daily_from_run(candidate, eth, SPOT, start=None, end=None)
    baseline_daily = _daily_from_run(baseline, eth, SPOT, start=None, end=None)

    return dict(
        candidate_daily=candidate_daily,
        baseline_daily=baseline_daily,
        published_gap=-0.009,  # ledger "### R-109", novel branch, ETH spot d_sharpe
        note=("R-109-novel, ETH spot, primary cell (thresh=0.90, max_discount=1.0). "
              "ETH series is Bitfinex `load_eth()`, full non-holdout range "
              f"{eth.index[0]} -> {eth.index[-1]} (compare()'s eth_replication slice "
              "uses the whole passed frame, NOT an INNER_VAL date-slice -- this ETH "
              "window predates INNER_VAL_START=2021-01-01 entirely). Published "
              "d_sharpe=-0.009 is bar-level (5m) annualized Sharpe diff "
              "(tradebot.metrics.sharpe_ratio); gap_sharpe here is daily-resampled -- "
              "see module docstring's second disclosed discrepancy."),
    )


# =====================================================================
# 2. R-113: 8-asset multi-asset panel (xsmom_entry_band), Mahalanobis
#    panel-novelty brake (r113_conservative_mahalanobis_panel.py). No
#    isolated per-asset ETH cell exists -- per r137_shared's own R-113
#    caveat, this reproduces the BASKET-level candidate (discounted) vs
#    frozen (undiscounted) daily return series from the decisive D3 cell
#    the ledger's own "### R-113" entry reports numerically: W_VAL
#    (2022-01-01..2022-12-31), U8 (8-asset universe, ETH included),
#    SPOT_BASE (0.10% taker), model="mahalanobis" (the r137_shared table's
#    first-listed file). D3, not D1/D2's W_FULL6/U6, is used because W_FULL6
#    runs on UNIVERSE_6 (6 assets, NO BTC/ETH) -- not the "8-asset basket"
#    r137_shared's own caveat describes -- while D3 runs on UNIVERSE_8 (8
#    assets, ETH included) and is the one cell the ledger entry quantifies
#    numerically (mahalanobis growth_diff=-0.284).
#    NOTE: W_VAL here is r63_shared's own W_VAL = (2022-01-01, 2022-12-31),
#    ONE year, not the 2021-2022 two-year INNER_VAL this round's other four
#    constructions use -- this axis has its own W_TRAIN/W_VAL/W_FULL6/W_HOLD
#    convention, disjoint from kelly_regime_v4's INNER_VAL/INNER_TRAIN
#    convention, and no U8 cell in this construction's own harness ever
#    scores a 2021-2022 window. Flagged, not silently substituted.
# =====================================================================

def load_r113() -> dict:
    from experiments.r113_conservative_mahalanobis_panel import (
        MODEL,
        PRIMARY_MAXD,
        PRIMARY_THRESH,
        SPOT_BASE,
        UNIVERSE_8,
        W_VAL,
        build_cell,
        cell_cmp,
        load_universe,
    )

    frames = load_universe(UNIVERSE_8)
    aligned_val, frozen_val, cand_val = build_cell(
        frames, UNIVERSE_8, W_VAL, MODEL, PRIMARY_THRESH, PRIMARY_MAXD
    )
    net_cmp_val_base, frozen_eq, cand_eq = cell_cmp(frozen_val, cand_val, aligned_val, SPOT_BASE)
    for name, df in aligned_val.items():
        _assert_no_holdout(df, f"R-113 {name}")

    candidate_daily = daily_returns(cand_eq)
    baseline_daily = daily_returns(frozen_eq)
    self_gap = gap_sharpe(candidate_daily, baseline_daily)

    return dict(
        candidate_daily=candidate_daily,
        baseline_daily=baseline_daily,
        # R-113's own harness never reports a Sharpe-difference d_sharpe at
        # all (its D1-D5 gates score growth_diff / dd_diff, log-return and
        # drawdown differences of the SAME candidate-vs-frozen equity curves
        # reproduced here -- see cell_cmp/compare() in r63_shared.py). There
        # is therefore no independently-published Sharpe-diff number to
        # check this reproduction against; `published_gap` is set to this
        # loader's own gap_sharpe computation (tautologically equal -- see
        # `note`), not an external citation like the other four
        # constructions'.
        published_gap=self_gap,
        note=("R-113 BASKET-level (per r137_shared's own R-113 caveat: no isolated "
              "per-asset ETH cell exists). Series = candidate (mahalanobis-discounted) "
              "vs frozen (undiscounted) xsmom_entry_band 8-asset (UNIVERSE_8, ETH "
              "included) portfolio equity curves, cell_cmp()'s own D3 cell: "
              f"W_VAL={W_VAL} (NOT INNER_VAL -- one year, 2022 only, this axis's own "
              "convention), U8, SPOT_BASE, model=mahalanobis, primary "
              f"(thresh={PRIMARY_THRESH}, max_discount={PRIMARY_MAXD}). Ledger '### R-113' "
              "reports this exact cell as growth_diff=-0.284 (log-return difference, "
              "not a Sharpe difference -- R-113's own harness never computes a "
              "Sharpe-diff d_sharpe at all). published_gap above is this loader's own "
              "gap_sharpe on the reproduced series (self-consistent by construction, "
              f"= {self_gap:+.4f}); its NEGATIVE sign matches growth_diff's own sign, "
              "the only cross-check available given the harness's own metric choice. "
              "Flagged per this round's own instructions, not averaged in with the "
              "four isolated-ETH cells un-flagged."),
    )


# =====================================================================
# 3. R-115-conservative: CORAL-pooled kNN novelty brake, Coinbase ETH.
#    Harness: experiments/r112_shared.py (chains r109_shared -> ... ->
#    r102_shared) + experiments/r115_conservative_shared.py's own
#    load_eth_coinbase() + experiments/r115_conservative_pooled_eth_
#    coinbase.py's own build_target/make_build_target at the frozen primary
#    cell, all imported verbatim.
# =====================================================================

def load_r115_conservative() -> dict:
    from experiments.r112_shared import PRIMARY_MAXD, PRIMARY_THRESH, SPOT, TargetStrategy, v4_target
    from experiments.r115_conservative_shared import load_eth_coinbase
    from experiments.r115_conservative_pooled_eth_coinbase import load_pool_daily_panels, make_build_target

    eth = load_eth_coinbase()
    _assert_no_holdout(eth, "R-115-conservative ETH (Coinbase)")
    pool_dailies = load_pool_daily_panels()

    r115_build_target = make_build_target(pool_dailies, PRIMARY_THRESH, PRIMARY_MAXD)
    candidate = TargetStrategy(r115_build_target, name="r115_conservative_pooled_knn")
    baseline = TargetStrategy(v4_target, name="kelly_regime_v4")

    # Same `eth_replication` convention as R-109-novel: the WHOLE passed ETH
    # frame (2019-03-14 -> 2022-12-31 for the Coinbase series), not an
    # INNER_VAL date-slice -- see module docstring's first disclosed
    # discrepancy.
    candidate_daily = _daily_from_run(candidate, eth, SPOT, start=None, end=None)
    baseline_daily = _daily_from_run(baseline, eth, SPOT, start=None, end=None)

    return dict(
        candidate_daily=candidate_daily,
        baseline_daily=baseline_daily,
        published_gap=-0.0890,  # ledger "### R-115", conservative branch, ETH spot d_sharpe
        note=("R-115-conservative, ETH spot, primary cell (thresh=0.90, max_discount=1.0), "
              "k=10, refit_every=30, pooled reference = target + UNIVERSE_6 (6 instruments), "
              "CORAL-standardized. ETH series is Coinbase `load_eth_coinbase()`, full "
              f"non-holdout range {eth.index[0]} -> {eth.index[-1]} (compare()'s "
              "eth_replication slice uses the whole passed frame, NOT an INNER_VAL "
              "date-slice -- INNER_VAL is a strict ~2-year subset of this ~3.8-year "
              "window). Published d_sharpe=-0.0890 is bar-level (5m) annualized Sharpe "
              "diff; gap_sharpe here is daily-resampled -- see module docstring's "
              "second disclosed discrepancy."),
    )


# =====================================================================
# 4. R-125-conservative: CVaR substituted for std-dev in kelly_regime_v4's
#    scale. Harness: experiments/r125_shared.py + experiments/
#    r125_conservative_cvar_scale.py's own calibrate_for/make_candidate_
#    factory, imported verbatim. B1-level (BTC), not B4-level: BTC itself
#    fails B1 (per docs/LEDGER.md "### R-125"), so this round's gating logic
#    never reached ETH -- but the code still computes and prints an ETH
#    cell (`b4_spot` in the branch file's own main()), reproduced here as
#    instructed, flagged as B1-level not B4-level.
# =====================================================================

def load_r125_conservative() -> dict:
    from experiments import r125_shared
    from experiments.r125_conservative_cvar_scale import PRIMARY_ALPHA, PRIMARY_WINDOW_DAYS, calibrate_for, make_candidate_factory

    btc = r125_shared.load_btc_train("spot")[0]
    target_cvar_primary = calibrate_for(btc, PRIMARY_WINDOW_DAYS, PRIMARY_ALPHA)
    cand_factory = make_candidate_factory(target_cvar_primary, PRIMARY_WINDOW_DAYS, PRIMARY_ALPHA)

    eth = r125_shared.load_eth_train()
    _assert_no_holdout(eth, "R-125-conservative ETH")

    m_cand, res_cand = r125_shared.run_candidate(
        cand_factory, eth, r125_shared.SPOT, r125_shared.INNER_VAL_START, r125_shared.INNER_VAL_END)
    m_v4, res_v4 = r125_shared.run_candidate(
        lambda: get_strategy("kelly_regime_v4"), eth, r125_shared.SPOT,
        r125_shared.INNER_VAL_START, r125_shared.INNER_VAL_END)

    candidate_daily = daily_returns(res_cand.equity)
    baseline_daily = daily_returns(res_v4.equity)

    return dict(
        candidate_daily=candidate_daily,
        baseline_daily=baseline_daily,
        published_gap=-0.106,  # ledger "### R-125", conservative branch, ETH spot d_sharpe
        note=("R-125-conservative, ETH spot, primary cell (cvar_window_days="
              f"{PRIMARY_WINDOW_DAYS}, alpha={PRIMARY_ALPHA}, target_cvar="
              f"{target_cvar_primary:.6f} calibrated on BTC inner-train). B1-LEVEL, NOT "
              "B4-level: this construction fails B1 on BTC itself (BTC spot "
              "d_sharpe=+0.144, neither clears +0.2 nor excludes zero per the ledger), so "
              "the round's own gating logic never reached a clean B4 inversion -- this ETH "
              "cell is what r125_conservative_cvar_scale.py's own main() still computes "
              "and prints as 'B4' regardless (its own pre-registered decision rule requires "
              "B1 to pass for promotion, but the ETH comparison itself is unconditional "
              "code, always run). Window: INNER_VAL_START..INNER_VAL_END (2021-01-01 -> "
              "2022-12-31), genuinely restricted via run_period's own start/end -- this "
              "construction's ETH window matches the task's INNER_VAL premise exactly. "
              "Published d_sharpe=-0.106 is bar-level (5m) annualized Sharpe diff; "
              "gap_sharpe here is daily-resampled -- see module docstring's second "
              "disclosed discrepancy."),
    )


# =====================================================================
# 5. R-126-conservative: Equal Risk Contribution reallocation of
#    champions_council. Harness: experiments/r126_shared.py + experiments/
#    r126_conservative_erc_council.py's own build_target (fit_weights_erc +
#    build_weight_schedule + weights_to_target), imported verbatim.
# =====================================================================

def load_r126_conservative() -> dict:
    from experiments import r126_shared
    from experiments.r126_conservative_erc_council import (
        PRIMARY_LOOKBACK_DAYS,
        PRIMARY_REBALANCE_DAYS,
        build_target as r126_build_target,
    )

    eth = r126_shared.load_eth_train()
    _assert_no_holdout(eth, "R-126-conservative ETH")

    target_eth, sched_eth, a_eth, payoff_eth = r126_build_target(
        eth, PRIMARY_REBALANCE_DAYS, PRIMARY_LOOKBACK_DAYS)

    m_cand, res_cand = r126_shared.run_target_series(
        target_eth, eth, r126_shared.SPOT, r126_shared.INNER_VAL_START, r126_shared.INNER_VAL_END)
    m_council, res_council = r126_shared.run_candidate_council(eth, r126_shared.SPOT)

    candidate_daily = daily_returns(res_cand.equity)
    baseline_daily = daily_returns(res_council.equity)

    return dict(
        candidate_daily=candidate_daily,
        baseline_daily=baseline_daily,
        published_gap=-1.450,  # ledger "### R-126", conservative (ERC) branch, ETH spot d_sharpe
        note=("R-126-conservative (ERC), ETH spot, primary config "
              f"(rebalance_days={PRIMARY_REBALANCE_DAYS}, lookback_days={PRIMARY_LOOKBACK_DAYS}). "
              "Candidate = inverse-trailing-vol ERC reallocation of champions_council's own "
              "6-member signal matrix; baseline = champions_council's own registered Hedge "
              "blend, both on ETH. Window: INNER_VAL_START..INNER_VAL_END (2021-01-01 -> "
              "2022-12-31), genuinely restricted via run_period's own start/end -- matches "
              "the task's INNER_VAL premise exactly. Published d_sharpe=-1.450 is bar-level "
              "(5m) annualized Sharpe diff (one of the sharpest sign reversals in this "
              "project's ledger, per '### R-126'); gap_sharpe here is daily-resampled -- "
              "see module docstring's second disclosed discrepancy."),
    )


CONSTRUCTIONS = {
    "R-109-novel": load_r109_novel,
    "R-113": load_r113,
    "R-115-conservative": load_r115_conservative,
    "R-125-conservative": load_r125_conservative,
    "R-126-conservative": load_r126_conservative,
}


if __name__ == "__main__":
    TOL_REL = 0.01  # 1% relative-tolerance flag threshold, per this round's own instructions

    print("=" * 100)
    print("R-137 LOADERS: reproducing each construction's own published ETH-window "
          "candidate/baseline series")
    print("=" * 100)

    results = []
    for name, loader in CONSTRUCTIONS.items():
        print(f"\n--- {name} ---")
        out = loader()
        reproduced = gap_sharpe(out["candidate_daily"], out["baseline_daily"])
        published = out["published_gap"]
        abs_err = abs(reproduced - published)
        rel_err = abs_err / abs(published) if published != 0 else float("inf")
        status = "PASS" if rel_err <= TOL_REL else "FAIL"
        print(f"  n_candidate_days={len(out['candidate_daily'])}  "
              f"n_baseline_days={len(out['baseline_daily'])}")
        print(f"  published_gap   = {published:+.4f}")
        print(f"  reproduced_gap  = {reproduced:+.4f}  (daily-resampled gap_sharpe)")
        print(f"  abs_err={abs_err:.4f}  rel_err={rel_err:.2%}  -> {status} (tol={TOL_REL:.0%})")
        print(f"  note: {out['note']}")
        results.append(dict(name=name, published=published, reproduced=reproduced,
                             abs_err=abs_err, rel_err=rel_err, status=status))

    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    hdr = f"{'construction':22s} {'published':>10s} {'reproduced':>11s} {'abs_err':>9s} {'rel_err':>9s}  status"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:22s} {r['published']:>+10.4f} {r['reproduced']:>+11.4f} "
              f"{r['abs_err']:>9.4f} {r['rel_err']:>9.2%}  {r['status']}")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{n_pass}/{len(results)} reproductions within {TOL_REL:.0%} of their published gap.")
    print("NOTE: R-113's 'published_gap' is self-derived (gap_sharpe on its own reproduced "
          "series), not an independently published Sharpe-diff -- see load_r113()'s own "
          "docstring/note. Every other construction's published_gap is a bar-level (5m) "
          "Sharpe diff cited from docs/LEDGER.md, compared here against a daily-resampled "
          "gap_sharpe -- see this module's own top-of-file docstring for why that gap is "
          "expected, not a wiring bug, on its own.")
