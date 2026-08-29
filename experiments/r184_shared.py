"""R-184 shared engine (frozen pre-registration) — see docs/LEDGER.md R-184.

Combinatorial overfitting audit (Bailey, Borwein, Lopez de Prado & Zhu 2014,
"The Probability of Backtest Overfitting", J. Computational Finance 20(4))
of `kelly_regime_v4`'s own shipped hyperparameters (20/40/80-day anchor
ladder, 10% deadband), never previously audited: R-29's own `cpcv()`
(scripts/inference.py) cross-validates the *whole-roster comparison table*
selection rule, never one strategy's own internal knobs, and R-34 through
R-60's 21+ retunes each picked a point via one inner-train/inner-validation
comparison rather than a combinatorial-overfitting statistic.

This module is the frozen, shared, read-only engine both branches import
from — the grid, the CSCV machinery and the Stage-0 gate are fixed here
and neither branch may alter them (only the shared file is ever edited,
and only before either branch is dispatched, per ROUTINE.md's
pre-registration collision convention).

Frozen grid: H1 (first anchor, days) x deadband, 35 points including the
shipped configuration itself (H1=20, deadband=0.10). The doubling ladder
(H1, 2*H1, 4*H1) is held fixed, matching v4's own a-priori structure;
target_vol=0.55 and max_leverage=2.0 are held at their shipped defaults
throughout and excluded from the grid, to avoid re-litigating R-59/R-60's
already-closed target_vol work.

CSCV parameters (n_groups=10, k_test=2, purge=embargo=100 days) match this
project's own existing convention in `scripts/inference.py`'s `cpcv()`
(R-29), which reports "45 CPCV splits" for the identical configuration.
Computed on BTC spot, 2017-01-01 -> 2022-12-31 (train period only, no
holdout read — this is a design-time, non-adaptive computation).
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    cpcv_splits, daily_returns, fold_mask, group_bounds, purged_train_mask,
    total_log_return,
)
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

TRAIN_END = "2022-12-31"
H1_GRID: tuple[int, ...] = (16, 18, 20, 22, 24, 26, 28)
DEADBAND_GRID: tuple[float, ...] = (0.05, 0.075, 0.10, 0.15, 0.20)
SHIPPED = (20, 0.10)
N_GROUPS = 10
K_TEST = 2
PURGE_EMBARGO_DAYS = 100  # matches scripts/inference.py's cpcv() convention
PBO_GATE = 0.30  # Stage-0 threshold, pre-registered

RETURNS_CACHE = ROOT / "experiments" / "r184_returns_matrix.csv.gz"

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()


def config_name(h1: int, deadband: float) -> str:
    return f"h{h1}_d{deadband:g}"


def all_configs() -> list[tuple[int, float]]:
    return list(product(H1_GRID, DEADBAND_GRID))


def build_returns_matrix(force: bool = False) -> pd.DataFrame:
    """Daily log-return column per (H1, deadband) config, BTC spot, train period.

    Cached to ``r184_returns_matrix.csv.gz`` (35 backtests, ~6s each);
    delete the file or pass ``force=True`` to rebuild.
    """
    if RETURNS_CACHE.exists() and not force:
        return pd.read_csv(RETURNS_CACHE, index_col=0, parse_dates=True)

    if LABEL == "SYNTHETIC":
        raise SystemExit("real data required; refusing to compute PBO on synthetic")

    train = DF.loc[:TRAIN_END]
    series: dict[str, pd.Series] = {}
    for i, (h1, deadband) in enumerate(all_configs(), 1):
        t0 = time.time()
        strat = KellyRegimeV4(horizons=(h1, 2 * h1, 4 * h1), deadband=deadband)
        result = run_backtest(strat, train, SPOT, 1_000.0, data_label=LABEL)
        series[config_name(h1, deadband)] = daily_returns(result.equity)
        print(f"[{i}/{len(all_configs())}] {config_name(h1, deadband)}: "
              f"{time.time() - t0:4.1f}s", file=sys.stderr)
    frame = pd.DataFrame(series).sort_index()
    RETURNS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RETURNS_CACHE)
    return frame


def pbo(returns: pd.DataFrame, n_groups: int = N_GROUPS, k_test: int = K_TEST,
        purge: int = PURGE_EMBARGO_DAYS) -> dict:
    """Bailey, Borwein, Lopez de Prado & Zhu (2014) PBO over ``returns``' columns.

    For each of C(n_groups, k_test) CSCV splits: pick the in-sample
    (train-mask) total-log-return winner among all configs, find its
    relative rank among all configs' out-of-sample (test-mask) total log
    returns, logit-transform that rank. PBO is the fraction of splits
    where the logit is <= 0 — the in-sample winner sitting at or below
    the out-of-sample median, i.e. the selection procedure choosing a
    config indistinguishable from (or worse than) picking one at random.
    """
    names = list(returns.columns)
    n = len(returns)
    bounds = group_bounds(n, n_groups)
    logits, picks = [], []
    for test_groups in cpcv_splits(n_groups, k_test):
        train_mask = purged_train_mask(n, bounds, test_groups, purge=purge,
                                       embargo=purge)
        test_mask = fold_mask(n, bounds, test_groups)
        is_scores = {c: total_log_return(returns[c].to_numpy()[train_mask])
                     for c in names}
        picked = max(is_scores, key=is_scores.get)
        oos_scores = {c: total_log_return(returns[c].to_numpy()[test_mask])
                      for c in names}
        order = sorted(oos_scores, key=oos_scores.get)  # worst -> best
        rel_rank = (order.index(picked) + 1 - 0.5) / len(order)
        logit = float(np.log(rel_rank / (1.0 - rel_rank)))
        logits.append(logit)
        picks.append(picked)
    logits_arr = np.array(logits)
    picks_arr = np.array(picks)
    per_config_pbo_contribution = {
        c: float(np.mean((picks_arr == c) & (logits_arr <= 0.0)))
        for c in names
    }
    return {
        "n_splits": len(logits_arr),
        "pbo": float(np.mean(logits_arr <= 0.0)),
        "logits": logits_arr,
        "picks": picks_arr,
        "pbo_contribution": per_config_pbo_contribution,
    }


def bootstrap_pbo_ci(returns: pd.DataFrame, n_boot: int = 500, seed: int = 183,
                     **pbo_kwargs) -> tuple[float, float, float]:
    """Group-resampling CI around PBO(grid): resample the n_groups blocks
    with replacement, recompute PBO each time. Reported alongside the
    point estimate per the pre-registration's power-check requirement —
    Stage 0 fires on the CI, not the point estimate alone.
    """
    rng = np.random.default_rng(seed)
    n_groups = pbo_kwargs.get("n_groups", N_GROUPS)
    n = len(returns)
    bounds = group_bounds(n, n_groups)
    boot_vals = []
    for _ in range(n_boot):
        group_ids = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([np.arange(bounds[g][0], bounds[g][1]) for g in group_ids])
        resampled = returns.iloc[idx].reset_index(drop=True)
        boot_vals.append(pbo(resampled, **pbo_kwargs)["pbo"])
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return float(np.mean(boot_vals)), float(lo), float(hi)


def stage0_gate() -> dict:
    """The shared Stage-0 decision: compute PBO(grid) and its CI, decide
    whether either branch proceeds. Idempotent — safe to call from both
    branches independently; they must get the identical answer since both
    read the same frozen grid and cache.
    """
    returns = build_returns_matrix()
    result = pbo(returns)
    mean_ci, lo, hi = bootstrap_pbo_ci(returns)
    proceed = lo >= PBO_GATE  # require the *interval*, not just the point, to clear
    return {
        **result,
        "returns_shape": returns.shape,
        "ci_mean": mean_ci, "ci_lo": lo, "ci_hi": hi,
        "gate": PBO_GATE,
        "proceed": proceed,
    }


if __name__ == "__main__":
    g = stage0_gate()
    print(f"\nPBO(grid) = {g['pbo']:.3f}  (95% CI [{g['ci_lo']:.3f}, {g['ci_hi']:.3f}], "
          f"mean {g['ci_mean']:.3f}) over {g['n_splits']} splits, "
          f"{g['returns_shape'][1]} configs x {g['returns_shape'][0]} days")
    print(f"Stage-0 gate ({PBO_GATE}): {'PROCEED' if g['proceed'] else 'STOP -- REJECT both branches, NEGATIVE-BY-CONSTRUCTION'}")
    picks = pd.Series(g["picks"]).value_counts()
    print(f"in-sample-winner pick distribution across {g['n_splits']} splits:\n{picks}")
    shipped = config_name(*SHIPPED)
    print(f"shipped config {shipped} picked in-sample: {int(picks.get(shipped, 0))}/{g['n_splits']} splits, "
          f"PBO-contribution {g['pbo_contribution'].get(shipped, 0.0):.3f}")
