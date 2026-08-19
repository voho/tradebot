#!/usr/bin/env python
"""Put an error bar on every headline in the comparison table (backlog B-04).

The table ranks 25 strategies by final balance on one path. That ordering
has never been tested: R-20 measured a **±0.2 Sharpe noise floor** and
R-25 noted that deflated Sharpe, purged CV and bootstrap intervals were
cited in the docs and computed nowhere. This script computes them, using
:mod:`tradebot.inference`.

Six commands, in the order they should be read:

``selftest``
    Falsify the machinery before trusting its output. A strategy against
    itself must come back indistinguishable; a strategy against a
    100%-drawdown baseline must come back distinguishable; the deflated
    Sharpe must refuse to certify the best of N pure-noise trials.

``bootstrap``
    Stationary block bootstrap (30-day mean block, 2,000 resamples) over
    daily returns, full period and holdout, both markets. Reports each
    strategy's Sharpe and drawdown with an interval, and the **paired**
    difference against ``buy_and_hold`` on identical resamples.

``deflated``
    Deflated Sharpe (Bailey & López de Prado 2014) for every strategy,
    against the number of trials this project has actually run.

``ordering``
    The table's claim is an *order*. This tests every adjacent pair in it
    and counts how many steps down the ranking survive a 95% interval.

``cpcv``
    Combinatorially purged cross-validation of the *selection procedure*
    itself: "rank the table on the training groups, hold the winner on the
    test groups". 45 out-of-fold splits, purged and embargoed, versus the
    single walk-forward number the repo has been quoting.

``charts``
    Forest plots of the same intervals, into ``reports/inference/``.

Usage::

    python scripts/inference.py all          # everything, ~15 min from cold
    python scripts/inference.py curves       # just rebuild the return cache
    python scripts/inference.py bootstrap --strategies kelly_regime_v4 buy_and_hold

Equity curves are cached in ``reports/inference/daily_returns.csv.gz`` so
the statistics can be re-run in seconds. Delete it to force a rebuild.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    annualized_sharpe, bootstrap_interval, cpcv_splits, daily_returns,
    deflated_sharpe_ratio, deflation_breakeven_sd, expected_max_sharpe,
    fold_mask, group_bounds,
    max_drawdown_from_returns, min_track_record_length, moments,
    paired_bootstrap, probabilistic_sharpe_ratio, purged_train_mask,
    stationary_bootstrap_indices, total_log_return,
)
from tradebot.registry import available_strategies, get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

OUT = ROOT / "reports" / "inference"
CACHE = {"full": OUT / "daily_returns.csv.gz",
         "holdout": OUT / "daily_returns_holdout.csv.gz"}
BENCHMARK = "buy_and_hold"
INCUMBENT = "kelly_regime_v4"   # the table's current #1, as a fixed-rule control
OOS_START = "2023-01-01"
MARKETS = {"spot": MarketSpec.spot(), "futures": MarketSpec.futures(leverage=5.0)}

MEAN_BLOCK = 30.0   # days; the R-20 setting that measured the noise floor
N_BOOT = 2_000
LEVEL = 0.95

# Trials this project has run, counted from the ledger: 32 fee-tier
# configurations (R-12/R-13), 24 e-process configurations (R-28), 9 anchor
# sets (R-07), 7 ladder widths (R-06), 4 volatility estimators (R-09), 2
# drawdown-cushion variants (R-11), plus the 25 registered strategies
# themselves — each of which is a configuration someone chose to keep.
# Under-counting is the failure mode that matters, so this is a floor.
#
# Rounds since add to it, and the routine is explicit that parallel branches
# contribute their TOTAL rather than the best one's: +36 (R-31, matched-risk
# frontier) +33 (R-32, the ungated control that ran the same backlog row the
# same day) +18 (R-33, the de-levered-benchmark exposure sweep) = 87.
PROJECT_TRIALS = 190

# The Sharpe dispersion across R-28's 24 configurations on inner-validation -
# the only trial dispersion this project has ever measured, and therefore the
# only defensible "same search" estimate. See deflated() for why it matters
# more than the trials count does.
SD_TRIALS_NARROW = 0.223


# ------------------------------------------------------------------- curves

def build_curves(strategies: list[str], period: str,
                 force: bool = False) -> pd.DataFrame:
    """Backtest every strategy on both markets; cache the daily returns.

    ``period="full"`` runs the whole 2017-2026 history. ``period="holdout"``
    runs a **fresh $1,000 account from 2023-01-01**, warmed on the bars
    before it, exactly as ``run_period`` does everywhere else in this repo.

    The holdout is not a slice of the full run, and the difference is not
    cosmetic: on 5x futures ``buy_and_hold`` is liquidated in early 2017, so a
    slice of the full run scores the benchmark's holdout at a flat zero —
    a corpse — where a fresh account rides the 2023+ bull to $15.2K.
    Measuring against a corpse is the R-22 mistake, and slicing is how it
    would have come back.
    """
    cache = CACHE[period]
    if cache.exists() and not force:
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        want = {f"{s}|{m}" for s in strategies for m in MARKETS}
        if want.issubset(cached.columns):
            return cached
        print(f"{period} cache is missing {len(want - set(cached.columns))} "
              f"series; rebuilding", file=sys.stderr)

    df, label = load_dataset(ROOT / "data", "spot")
    if label == "SYNTHETIC":
        raise SystemExit("real data required; refusing to publish intervals on synthetic")
    print(f"{period}: {len(df):,} bars  {df.index[0]:%Y-%m-%d} -> "
          f"{df.index[-1]:%Y-%m-%d}  (data: {label})", file=sys.stderr)

    series: dict[str, pd.Series] = {}
    for i, name in enumerate(strategies, 1):
        for market_name, market in MARKETS.items():
            t0 = time.time()
            if period == "holdout":
                result = run_period(get_strategy(name), df, OOS_START, None,
                                    market=market, start_balance=1_000.0,
                                    data_label=label)
            else:
                result = run_backtest(get_strategy(name), df, market, 1_000.0,
                                      data_label=label)
            series[f"{name}|{market_name}"] = daily_returns(result.equity)
            print(f"[{period} {i}/{len(strategies)}] {name:22s} {market_name:8s} "
                  f"{time.time() - t0:5.1f}s", file=sys.stderr)
    out = pd.DataFrame(series).sort_index()
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def dead_tail_pct(rets: np.ndarray) -> float:
    """Share of the period after the account's last non-zero day.

    A strategy that is flat 30% of the time because its gate is shut is
    doing its job; one whose *last* 40% of days are all zeros is a corpse,
    and every statistic computed over that tail is measuring the corpse.
    Only the terminal run counts.
    """
    nonzero = np.flatnonzero(rets != 0.0)
    if len(nonzero) == 0:
        return 100.0
    return 100.0 * (len(rets) - 1 - int(nonzero[-1])) / len(rets)


# ----------------------------------------------------------------- selftest

def selftest(curves: dict[str, pd.DataFrame], **_) -> bool:
    """Pre-registered falsification of the machinery, before any claim.

    Named in advance (see the R-29 row in ``docs/LEDGER.md``): if any of
    these four fail, every number this script prints is void.
    """
    rng = np.random.default_rng(0)
    checks: list[tuple[str, bool, str]] = []

    full = curves["full"]
    v4 = full["kelly_regime_v4|spot"].to_numpy()
    dead = full["macd_cross|spot"].to_numpy()
    idx = stationary_bootstrap_indices(len(v4), MEAN_BLOCK, N_BOOT, rng)

    # 1. A series against itself must be indistinguishable from itself.
    same = paired_bootstrap(v4, v4.copy(), annualized_sharpe, indices=idx)
    checks.append(("null pair (v4 vs itself): interval contains 0",
                   not same.significant and abs(same.diff.point) < 1e-9,
                   f"diff {same.diff}, P(>0)={same.p_positive:.2f}"))

    # 2. A real gap must be detected. kelly_regime_v4 ends at $66.8K where
    #    macd_cross ends at $4.99 — if the test cannot see that, it cannot
    #    see anything.
    gap = paired_bootstrap(v4, dead, annualized_sharpe, indices=idx)
    checks.append(("real gap (v4 vs macd_cross): interval excludes 0",
                   gap.significant and gap.p_positive > 0.99,
                   f"diff {gap.diff}, P(>0)={gap.p_positive:.2f}"))

    # 3. Deflated Sharpe must not certify the best of N pure-noise trials.
    n_days, n_trials = len(v4), 50
    noise = rng.normal(0.0, 0.02, size=(n_trials, n_days))
    sharpes = annualized_sharpe(noise)
    best = int(np.argmax(sharpes))
    skew, kurt = moments(noise[best])
    dsr_noise = deflated_sharpe_ratio(float(sharpes[best]), n_days, skew, kurt,
                                      n_trials, float(sharpes.std(ddof=1)))
    checks.append(("deflated Sharpe refuses the best of 50 noise trials",
                   dsr_noise < 0.95,
                   f"best-of-{n_trials} Sharpe {sharpes[best]:.2f}, DSR {dsr_noise:.3f}"))

    # 4. Purged CV splits must be disjoint, and the purge must bite.
    bounds = group_bounds(n_days, 10)
    ok = True
    for test_groups in cpcv_splits(10, 2):
        tr = purged_train_mask(n_days, bounds, test_groups, purge=100, embargo=100)
        te = fold_mask(n_days, bounds, test_groups)
        ok &= not (tr & te).any() and tr.sum() < (~te).sum()
    checks.append(("CPCV splits disjoint and purged", ok,
                   f"{len(cpcv_splits(10, 2))} splits checked"))

    print("\nSELF-TEST — the machinery, before any claim it makes\n")
    for label, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label:52s} {detail}")
    passed = all(c[1] for c in checks)
    print(f"\n  {'all checks pass' if passed else 'FAILED — every number below is void'}")
    return passed


# ---------------------------------------------------------------- bootstrap

def _table(rows: list[dict], name: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    return frame


def bootstrap(curves: dict[str, pd.DataFrame], strategies: list[str],
              **_) -> pd.DataFrame:
    """Intervals on every headline, and the paired comparison against holding."""
    rows = []
    periods = tuple(curves)
    for period in periods:
        window = curves[period]
        n = len(window)
        idx = stationary_bootstrap_indices(n, MEAN_BLOCK, N_BOOT,
                                           np.random.default_rng(7))
        for market in MARKETS:
            bench = window[f"{BENCHMARK}|{market}"].to_numpy()
            for name in strategies:
                r = window[f"{name}|{market}"].to_numpy()
                sharpe = paired_bootstrap(r, bench, annualized_sharpe,
                                          indices=idx, level=LEVEL)
                dd = paired_bootstrap(r, bench, max_drawdown_from_returns,
                                      indices=idx, level=LEVEL)
                growth = paired_bootstrap(r, bench, total_log_return,
                                          indices=idx, level=LEVEL)
                # The marginal interval, from the same resamples. It is a
                # different object from the paired one and much wider: most
                # of a strategy's Sharpe uncertainty is the market's, which
                # is exactly what the pairing cancels.
                own_sharpe = bootstrap_interval(r, annualized_sharpe,
                                                indices=idx, level=LEVEL)
                own_dd = bootstrap_interval(r, max_drawdown_from_returns,
                                            indices=idx, level=LEVEL)
                rows.append({
                    "period": period, "market": market, "strategy": name,
                    "days": n,
                    # A liquidated futures account stops trading, so its
                    # remaining days are exactly zero. Those days are not
                    # "flat and calm" — they are a corpse, and they flatter
                    # every volatility-based statistic computed over them.
                    "dead_tail_pct": dead_tail_pct(r),
                    "sharpe": sharpe.stat_a,
                    "sharpe_lo": own_sharpe.lo, "sharpe_hi": own_sharpe.hi,
                    "d_sharpe": sharpe.diff.point,
                    "d_sharpe_lo": sharpe.diff.lo, "d_sharpe_hi": sharpe.diff.hi,
                    "p_sharpe_beats_hold": sharpe.p_positive,
                    "max_dd_pct": dd.stat_a,
                    "max_dd_lo": own_dd.lo, "max_dd_hi": own_dd.hi,
                    "d_max_dd_pp": dd.diff.point,
                    "d_max_dd_lo": dd.diff.lo, "d_max_dd_hi": dd.diff.hi,
                    "p_dd_deeper_than_hold": dd.p_positive,
                    # Log growth is the table's own ranking criterion, so
                    # its interval is the one the comparison table prints
                    # (B-12). It disagrees with the Sharpe interval more
                    # often than one would like, which is the point.
                    "d_log_growth": growth.diff.point,
                    "d_log_growth_lo": growth.diff.lo,
                    "d_log_growth_hi": growth.diff.hi,
                    "p_growth_beats_hold": growth.p_positive,
                })
    frame = _table(rows, "bootstrap")

    for period in periods:
        for market in MARKETS:
            sub = frame[(frame.period == period) & (frame.market == market)]
            sub = sub.sort_values("sharpe", ascending=False)
            print(f"\n{period.upper()} / {market} — {sub.days.iloc[0]:,} days, "
                  f"{N_BOOT:,} stationary-bootstrap resamples, "
                  f"{MEAN_BLOCK:.0f}-day mean block")
            print(f"  {'strategy':22s} {'Sharpe (95% CI)':>22s} "
                  f"{'ΔSharpe vs hold':>25s} {'P>hold':>7s} "
                  f"{'maxDD':>6s} {'ΔmaxDD vs hold':>25s} "
                  f"{'Δlog growth vs hold':>25s}")
            for _, row in sub.iterrows():
                star = "*" if row.d_sharpe_lo > 0 or row.d_sharpe_hi < 0 else " "
                dstar = "*" if row.d_max_dd_lo > 0 or row.d_max_dd_hi < 0 else " "
                gstar = "*" if row.d_log_growth_lo > 0 or row.d_log_growth_hi < 0 else " "
                print(f"  {row.strategy:22s} {row.sharpe:>6.2f} "
                      f"[{row.sharpe_lo:>+5.2f},{row.sharpe_hi:>+5.2f}] "
                      f"{row.d_sharpe:>+6.2f} [{row.d_sharpe_lo:>+6.2f},"
                      f"{row.d_sharpe_hi:>+6.2f}]{star} "
                      f"{row.p_sharpe_beats_hold:>7.2f} {row.max_dd_pct:>5.1f}% "
                      f"{row.d_max_dd_pp:>+6.1f} [{row.d_max_dd_lo:>+6.1f},"
                      f"{row.d_max_dd_hi:>+6.1f}]{dstar} "
                      f"{row.d_log_growth:>+6.2f} [{row.d_log_growth_lo:>+6.2f},"
                      f"{row.d_log_growth_hi:>+6.2f}]{gstar}")
            print("  * = the 95% interval excludes zero")
            inert = sub[sub.dead_tail_pct > 10.0]
            if len(inert):
                print("  dead before the period ended (>10% of days after the "
                      "last non-zero return — the tail is a corpse, not a flat "
                      "position): "
                      + ", ".join(f"{r.strategy} {r.dead_tail_pct:.0f}%"
                                  for _, r in inert.iterrows()))
    return frame


def ordering(curves: dict[str, pd.DataFrame], strategies: list[str],
             **_) -> pd.DataFrame:
    """How much of the table's ordering is distinguishable from noise?

    Every adjacent pair in the ranking, tested against the same paired
    bootstrap. The table's claim is an *order*; this is the number of
    places in that order that survive.
    """
    rows = []
    periods = tuple(curves)
    for period in periods:
        window = curves[period]
        idx = stationary_bootstrap_indices(len(window), MEAN_BLOCK, N_BOOT,
                                           np.random.default_rng(7))
        for market in MARKETS:
            ranked = sorted(
                strategies,
                key=lambda s: -total_log_return(window[f"{s}|{market}"].to_numpy()))
            for upper, lower in zip(ranked, ranked[1:]):
                res = paired_bootstrap(window[f"{upper}|{market}"].to_numpy(),
                                       window[f"{lower}|{market}"].to_numpy(),
                                       total_log_return, indices=idx, level=LEVEL)
                rows.append({"period": period, "market": market,
                             "above": upper, "below": lower,
                             "d_log_growth": res.diff.point,
                             "lo": res.diff.lo, "hi": res.diff.hi,
                             "p_positive": res.p_positive,
                             "distinguishable": bool(res.significant)})
    frame = _table(rows, "ordering")
    print("\nADJACENT PAIRS IN THE RANKING — is each step down real?\n")
    for period in periods:
        for market in MARKETS:
            sub = frame[(frame.period == period) & (frame.market == market)]
            k = int(sub.distinguishable.sum())
            print(f"  {period:8s} {market:8s}  {k}/{len(sub)} adjacent pairs "
                  f"distinguishable at 95%")
    return frame


# ----------------------------------------------------------------- deflated

def deflated(curves: dict[str, pd.DataFrame], strategies: list[str],
             **_) -> pd.DataFrame:
    """Deflated Sharpe against the project's own trials count.

    Two dispersion assumptions are reported side by side, because the
    deflated Sharpe is far more sensitive to *how spread out* the trials
    were than to how many there were, and this project's answer flips
    between them:

    - ``NARROW`` — 0.223, the Sharpe spread measured across R-28's 24
      configurations on inner-validation. This is the only trial
      dispersion this project has ever actually measured, and it is the
      "same search" quantity Bailey & López de Prado intend.
    - ``TABLE`` — the spread across all 25 registered strategies. It is an
      upper bound rather than an estimate: most of that table was
      registered *as documented negative results*, not entered as
      candidates for promotion, so it overstates the search that produced
      the winner.

    The column that settles it is ``breakeven_sd``: the trial dispersion at
    which the strategy stops clearing DSR 0.95. Compare that against a
    search this project actually ran instead of arguing about which
    assumption is right.
    """
    rows = []
    periods = tuple(curves)
    for period in periods:
        window = curves[period]
        n = len(window)
        for market in MARKETS:
            sharpes = {s: annualized_sharpe(window[f"{s}|{market}"].to_numpy())
                       for s in strategies}
            sd_table = float(np.std(list(sharpes.values()), ddof=1))
            for name in strategies:
                r = window[f"{name}|{market}"].to_numpy()
                skew, kurt = moments(r)
                sr = sharpes[name]
                rows.append({
                    "period": period, "market": market, "strategy": name,
                    "days": n, "sharpe": sr, "skewness": skew, "kurt": kurt,
                    "psr_vs_zero": probabilistic_sharpe_ratio(sr, n, skew, kurt),
                    "n_trials": PROJECT_TRIALS,
                    "sd_narrow": SD_TRIALS_NARROW, "sd_table": sd_table,
                    "sr_star_narrow": expected_max_sharpe(PROJECT_TRIALS,
                                                          SD_TRIALS_NARROW),
                    "sr_star_table": expected_max_sharpe(PROJECT_TRIALS, sd_table),
                    "dsr_narrow": deflated_sharpe_ratio(
                        sr, n, skew, kurt, PROJECT_TRIALS, SD_TRIALS_NARROW),
                    "dsr_table": deflated_sharpe_ratio(
                        sr, n, skew, kurt, PROJECT_TRIALS, sd_table),
                    "breakeven_sd": deflation_breakeven_sd(
                        sr, n, skew, kurt, PROJECT_TRIALS),
                    "min_track_record_days": min_track_record_length(
                        sr, skew, kurt,
                        benchmark=expected_max_sharpe(PROJECT_TRIALS,
                                                      SD_TRIALS_NARROW)),
                })
    frame = _table(rows, "deflated")
    for period in periods:
        for market in MARKETS:
            sub = frame[(frame.period == period) & (frame.market == market)]
            sub = sub.sort_values("sharpe", ascending=False)
            print(f"\n{period.upper()} / {market} — deflated against "
                  f"{PROJECT_TRIALS} project trials.  "
                  f"SR* = {sub.sr_star_narrow.iloc[0]:.2f} at the narrow "
                  f"dispersion ({SD_TRIALS_NARROW}), "
                  f"{sub.sr_star_table.iloc[0]:.2f} at the table's "
                  f"({sub.sd_table.iloc[0]:.2f})")
            print(f"  {'strategy':22s} {'Sharpe':>6s} {'skew':>6s} {'kurt':>6s} "
                  f"{'PSR>0':>6s} {'DSR-n':>6s} {'DSR-t':>6s} "
                  f"{'break-even sd':>14s} {'min record':>11s}")
            # Item access, not attributes: `skew`, `kurt` and friends are
            # also pandas Series methods, and attribute access finds those.
            for _, row in sub.iterrows():
                mtrl = row["min_track_record_days"]
                years = "never" if not np.isfinite(mtrl) else f"{mtrl / 365.25:.1f}y"
                print(f"  {row['strategy']:22s} {row['sharpe']:>6.2f} "
                      f"{row['skewness']:>6.2f} {row['kurt']:>6.1f} "
                      f"{row['psr_vs_zero']:>6.3f} {row['dsr_narrow']:>6.3f} "
                      f"{row['dsr_table']:>6.3f} {row['breakeven_sd']:>14.2f} "
                      f"{years:>11s}")
            print("  DSR >= 0.95 clears the trials bar. break-even sd = the "
                  "trial dispersion at which it stops clearing it.")
    return frame


# --------------------------------------------------------------------- CPCV

def cpcv(curves: dict[str, pd.DataFrame], strategies: list[str],
         n_groups: int = 10, k_test: int = 2, purge: int = 100,
         **_) -> pd.DataFrame:
    """Cross-validate the *selection procedure*, not one strategy.

    The comparison table is a selection rule: "rank by final balance, take
    the top". Running that rule inside each fold and scoring the winner on
    the held-out groups measures what the table is actually worth to
    someone choosing from it — which is not the same question as whether
    any single strategy works, and has never been asked here.
    """
    rows = []
    window = curves["full"]
    n = len(window)
    bounds = group_bounds(n, n_groups)
    for market in MARKETS:
        rets = {s: window[f"{s}|{market}"].to_numpy() for s in strategies}
        for test_groups in cpcv_splits(n_groups, k_test):
            train = purged_train_mask(n, bounds, test_groups, purge=purge,
                                      embargo=purge)
            test = fold_mask(n, bounds, test_groups)
            scores = {s: total_log_return(rets[s][train]) for s in strategies}
            picked = max(scores, key=scores.get)
            oof = {s: total_log_return(rets[s][test]) for s in strategies}
            best_oof = max(oof, key=oof.get)
            rows.append({
                "market": market, "test_groups": "+".join(map(str, test_groups)),
                "train_days": int(train.sum()), "test_days": int(test.sum()),
                "picked": picked,
                "picked_train_log": scores[picked],
                "picked_test_log": oof[picked],
                "hold_test_log": oof[BENCHMARK],
                "excess_vs_hold": oof[picked] - oof[BENCHMARK],
                "best_possible_test_log": oof[best_oof],
                "selection_shortfall": oof[best_oof] - oof[picked],
                # The control: "always hold the table's #1" is a different
                # rule from "re-rank the table in every fold", and the gap
                # between them is what the ranking itself is worth.
                "incumbent_test_log": oof[INCUMBENT],
                "incumbent_excess_vs_hold": oof[INCUMBENT] - oof[BENCHMARK],
                # On 5x futures buy_and_hold is liquidated early and its
                # later folds are all zeros. Beating a corpse is not a
                # result (R-22), so the flag travels with the row.
                "hold_inert": bool((rets[BENCHMARK][test] == 0.0).all()),
                # Spearman = Pearson on the ranks, computed that way to
                # avoid a SciPy dependency the rest of the repo does without.
                "rank_corr": float(pd.Series(scores).rank().corr(
                    pd.Series(oof).rank())),
            })
    frame = _table(rows, "cpcv")

    print(f"\nCPCV — {len(cpcv_splits(n_groups, k_test))} splits "
          f"({n_groups} groups, {k_test} held out, {purge}-day purge + embargo)\n")
    for market in MARKETS:
        sub = frame[frame.market == market]
        picks = sub.picked.value_counts()
        # A fold where the rule picks the benchmark is a tie by
        # construction, not a win. Counting those as "did not beat hold"
        # understates the rule; counting them as wins would flatter it.
        ties = int((sub.picked == BENCHMARK).sum())
        wins = int((sub.excess_vs_hold > 0).sum())
        contested = len(sub) - ties
        print(f"  {market}:")
        print(f"    in-fold winner picked:        "
              + ", ".join(f"{k} x{v}" for k, v in picks.items()))
        print(f"    beats hold out-of-fold:       {wins}/{len(sub)} splits "
              f"({wins / len(sub):.0%}); {ties} of those are ties where the "
              f"rule picked hold itself, so {wins}/{contested} "
              f"({wins / max(contested, 1):.0%}) of contested folds")
        print(f"    always-{INCUMBENT}:  beats hold in "
              f"{(sub.incumbent_excess_vs_hold > 0).mean():.0%} of splits, "
              f"median {sub.incumbent_excess_vs_hold.median():+.3f} log")
        print(f"    median out-of-fold excess:    "
              f"{sub.excess_vs_hold.median():+.3f} log ({np.expm1(sub.excess_vs_hold.median()):+.1%})")
        print(f"    worst / best split:           "
              f"{sub.excess_vs_hold.min():+.3f} / {sub.excess_vs_hold.max():+.3f} log")
        print(f"    selection shortfall (median): "
              f"{sub.selection_shortfall.median():+.3f} log vs the "
              f"best strategy in hindsight")
        print(f"    train->test rank correlation: "
              f"median {sub.rank_corr.median():.2f}, "
              f"range {sub.rank_corr.min():.2f}..{sub.rank_corr.max():.2f}")
        if sub.hold_inert.any():
            live = sub[~sub.hold_inert]
            print(f"    NOTE: buy_and_hold is liquidated and inert in "
                  f"{int(sub.hold_inert.sum())}/{len(sub)} folds. Against the "
                  f"{len(live)} folds where it is alive the selected strategy "
                  f"beats it in {(live.excess_vs_hold > 0).mean():.0%}, "
                  f"median {live.excess_vs_hold.median():+.3f} log.")
    return frame


# ------------------------------------------------------------------- charts

def charts(curves: dict[str, pd.DataFrame], strategies: list[str],
           **_) -> list[Path]:
    """Forest plots: every strategy's difference from holding, with its interval.

    The comparison table is a list of point estimates in rank order, which
    is the most confident possible way to present them. This is the same
    information drawn honestly — an interval per strategy, and a vertical
    line at "no different from buy-and-hold".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tradebot.report import (BASELINE, CRITICAL, GOOD, GRID, INK, MUTED,
                                 PAGE, SURFACE)

    frame = pd.read_csv(OUT / "bootstrap.csv")
    paths = []
    for period in frame.period.unique():
        fig, axes = plt.subplots(1, 4, figsize=(19, 8), sharey=True)
        fig.patch.set_facecolor(PAGE)
        sharpe_cols = ("d_sharpe", "d_sharpe_lo", "d_sharpe_hi", True)
        dd_cols = ("d_max_dd_pp", "d_max_dd_lo", "d_max_dd_hi", False)
        panels = [("spot", *sharpe_cols, "Δ Sharpe vs buy_and_hold"),
                  ("spot", *dd_cols, "Δ max drawdown (pp)"),
                  ("futures", *sharpe_cols, "Δ Sharpe vs buy_and_hold"),
                  ("futures", *dd_cols, "Δ max drawdown (pp)")]
        order = frame[(frame.period == period) & (frame.market == "spot")] \
            .sort_values("d_sharpe")["strategy"].tolist()
        for ax, (market, col, lo_col, hi_col, more_is_better, title) in zip(axes, panels):
            sub = frame[(frame.period == period) & (frame.market == market)]
            sub = sub.set_index("strategy").loc[order]
            lo, hi = sub[lo_col], sub[hi_col]
            y = np.arange(len(sub))
            # "Good" points the other way for drawdown: less is better.
            better = (sub[col] > 0) if more_is_better else (sub[col] < 0)
            excludes_zero = (lo > 0) | (hi < 0)
            colors = [GOOD if b and s else CRITICAL if (not b) and s else MUTED
                      for b, s in zip(better, excludes_zero)]
            ax.set_facecolor(SURFACE)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(BASELINE)
            ax.grid(True, axis="x", color=GRID, linewidth=1.0)
            ax.hlines(y, lo, hi, color=colors, linewidth=2.4, alpha=0.55)
            ax.scatter(sub[col], y, s=34, color=colors, zorder=3,
                       edgecolors=SURFACE, linewidths=1.2)
            ax.axvline(0.0, color=INK, linewidth=1.2)
            ax.set_yticks(y)
            ax.set_yticklabels(sub.index, fontsize=8, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            ax.set_title(f"{market} · {title}", color=INK, fontsize=10, loc="left")
        n_days = int(frame[frame.period == period].days.iloc[0])
        fig.suptitle(
            f"Is the difference real?  ·  {period} ({n_days:,} days)  ·  "
            f"paired stationary bootstrap, {MEAN_BLOCK:.0f}-day blocks, "
            f"{N_BOOT:,} resamples  ·  grey = interval contains zero",
            color=INK, fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = OUT / f"intervals_{period}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=PAGE)
        plt.close(fig)
        paths.append(path)
        print(f"chart: {path}", file=sys.stderr)
    return paths


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", nargs="?", default="all",
                    choices=["all", "curves", "selftest", "bootstrap",
                             "ordering", "deflated", "cpcv", "charts"])
    ap.add_argument("--strategies", nargs="+", default=None,
                    help="default: every registered strategy")
    ap.add_argument("--force", action="store_true", help="rebuild the curve cache")
    args = ap.parse_args()

    strategies = args.strategies or sorted(available_strategies())
    if BENCHMARK not in strategies:
        strategies.append(BENCHMARK)
    curves = {period: build_curves(strategies, period, force=args.force)
              for period in CACHE}
    if args.command == "curves":
        for period, frame in curves.items():
            print(f"{period}: cached {frame.shape[1]} series x "
                  f"{len(frame):,} days -> {CACHE[period]}")
        return

    if args.command in ("all", "selftest"):
        if not selftest(curves) and args.command == "all":
            raise SystemExit("self-test failed; refusing to report statistics")
    for name, fn in (("bootstrap", bootstrap), ("ordering", ordering),
                     ("deflated", deflated), ("cpcv", cpcv), ("charts", charts)):
        if args.command in ("all", name):
            fn(curves, strategies=strategies)
    print(f"\nwritten to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
