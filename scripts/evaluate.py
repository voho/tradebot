"""Full evaluation of the frozen configuration on held-out data.

Nothing in this script is allowed to influence the strategy configuration.
Hyperparameters were chosen in ``scripts/search.py`` on TRAIN_SEEDS; the seeds
used here have never been evaluated during development.  Running this twice and
keeping the better answer would defeat the purpose.

Produces:

1. Fee-tier sweep.  The headline result, because at this frequency the fee tier
   decides viability.
2. Walk-forward with purge and embargo, per fold and pooled.
3. Negative controls.  The strategy must earn ~nothing on a structureless
   random walk and on block-bootstrapped surrogates, or its edge is an artefact
   of its own tails rather than of market structure.
4. Statistical tests: block-bootstrap Sharpe CI, Newey-West t, deflated Sharpe.
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from gtbot.data.schema import validate
from gtbot.data.synthetic import block_bootstrap, make_random_walk, simulate
from gtbot.engine.backtest import run_backtest
from gtbot.engine.broker import FEE_TIERS, CostModel, ExecutionConfig
from gtbot.eval import metrics, stats
from gtbot.eval.walkforward import run_walkforward
from gtbot.strategy import GameTheoreticStrategy, StrategyConfig

# --- frozen configuration -------------------------------------------------
CONFIG = StrategyConfig(horizon=3, entry_signal=0.55, max_hold=3)
EXECUTION = ExecutionConfig(entry_mode="taker", exit_mode="maker", ttl_bars=1)
MAX_LEVERAGE = 2.0

TEST_SEEDS = [100, 101, 102, 103, 104, 105]
N_BARS = 150_000
BPY = 365 * 288
#: Configurations evaluated during development, for the deflated Sharpe ratio.
#: The dispersion must be expressed per observation, in the same units as the
#: Sharpe passed to the test — quoting it annualised inflates the benchmark by
#: sqrt(bars per year) and drives the deflated Sharpe to zero for any strategy.
N_TRIALS = 18
TRIAL_SR_STD_ANNUAL = 1.2
TRIAL_SR_STD = TRIAL_SR_STD_ANNUAL / np.sqrt(BPY)

#: Walk-forward needs folds long enough for the online learner to converge
#: inside each one; 5 folds of a 150k-bar series is ~26k scored bars per fold,
#: which is mostly warm-up.
N_BARS_WF = 420_000
N_FOLDS_WF = 5


def _config_for(tier: str) -> StrategyConfig:
    """Tell the sizer the truth about what a round trip costs at this tier."""
    cost = CostModel.for_tier(tier)
    cfg = StrategyConfig(**{**CONFIG.__dict__})
    cfg.assumed_cost_bp = cost.round_trip_bp(EXECUTION)
    return cfg


def _run(bars, tier: str):
    res = run_backtest(
        bars,
        GameTheoreticStrategy(_config_for(tier)),
        costs=CostModel.for_tier(tier),
        execution=EXECUTION,
        max_leverage=MAX_LEVERAGE,
    )
    return res, metrics.compute(
        res.returns,
        res.equity,
        res.position,
        res.costs,
        bars_per_year=BPY,
        n_trades=res.n_trades,
        n_trials=N_TRIALS,
        trial_sr_std=TRIAL_SR_STD,
    )


def job_tier(args):
    seed, tier = args
    bars = validate(simulate(N_BARS, seed=seed).bars)
    res, m = _run(bars, tier)
    gross = metrics.compute(
        res.gross_returns,
        np.cumprod(1.0 + res.gross_returns),
        res.position,
        res.costs * 0,
        bars_per_year=BPY,
    )
    return tier, seed, m.to_dict(), gross.sharpe, res.returns


def job_control(args):
    seed, kind = args
    if kind == "random_walk":
        bars = validate(make_random_walk(N_BARS, seed=seed))
    elif kind == "block_bootstrap":
        bars = validate(block_bootstrap(simulate(N_BARS, seed=seed).bars, seed=seed))
    else:
        bars = validate(simulate(N_BARS, seed=seed).bars)
    _, m = _run(bars, "vip6")
    return kind, seed, m.to_dict()


def job_walkforward(seed):
    bars = validate(simulate(N_BARS_WF, seed=seed).bars)
    tier = "vip6"
    wf = run_walkforward(
        bars,
        lambda: GameTheoreticStrategy(_config_for(tier)),
        n_folds=N_FOLDS_WF,
        costs=CostModel.for_tier(tier),
        execution=EXECUTION,
        max_leverage=MAX_LEVERAGE,
    )
    return seed, wf.summary_frame().to_dict("records"), wf.pooled.to_dict(), wf.pooled_returns


def main() -> None:
    out: dict = {}
    print("=" * 78)
    print("HELD-OUT EVALUATION  (test seeds never used during development)")
    print(f"config: horizon={CONFIG.horizon} entry_signal={CONFIG.entry_signal} "
          f"max_hold={CONFIG.max_hold}  execution: {EXECUTION.entry_mode}->{EXECUTION.exit_mode}")
    print("=" * 78)

    # --- 1. fee-tier sweep -------------------------------------------------
    print("\n[1] FEE-TIER SWEEP  (mean over test seeds)")
    grid = [(s, t) for t in FEE_TIERS for s in TEST_SEEDS]
    with Pool(4) as p:
        rows = p.map(job_tier, grid)

    by_tier: dict[str, list] = {}
    returns_by_tier: dict[str, list] = {}
    for tier, seed, m, gross_sr, rets in rows:
        by_tier.setdefault(tier, []).append((seed, m, gross_sr))
        returns_by_tier.setdefault(tier, []).append(rets)

    header = f"{'tier':>13s} {'rt cost':>8s} {'sharpe':>8s} {'min':>7s} {'cagr':>8s} {'vol':>7s} {'maxDD':>7s} {'trades':>7s} {'gross SR':>9s}"
    print(header)
    tier_summary = {}
    for tier in FEE_TIERS:
        vals = by_tier[tier]
        sr = np.array([v[1]["sharpe"] for v in vals])
        rt = CostModel.for_tier(tier).round_trip_bp(EXECUTION)
        rec = {
            "round_trip_bp": rt,
            "sharpe_mean": float(sr.mean()),
            "sharpe_min": float(sr.min()),
            "cagr": float(np.mean([v[1]["cagr"] for v in vals])),
            "vol": float(np.mean([v[1]["ann_vol"] for v in vals])),
            "max_dd": float(np.mean([v[1]["max_drawdown"] for v in vals])),
            "trades": float(np.mean([v[1]["n_trades"] for v in vals])),
            "gross_sharpe": float(np.mean([v[2] for v in vals])),
        }
        tier_summary[tier] = rec
        print(f"{tier:>13s} {rt:7.2f}b {rec['sharpe_mean']:+8.2f} {rec['sharpe_min']:+7.2f} "
              f"{rec['cagr']:+8.2%} {rec['vol']:7.2%} {rec['max_dd']:7.2%} {rec['trades']:7.0f} "
              f"{rec['gross_sharpe']:+9.2f}")
    out["fee_tiers"] = tier_summary

    # --- 2. statistics on the pooled vip6 track record ---------------------
    print("\n[2] STATISTICAL TESTS  (tier=vip6, pooled across test seeds)")
    pooled = np.concatenate(returns_by_tier["vip6"])
    st = stats.summarise(pooled, bars_per_year=BPY)
    m_pooled = metrics.compute(
        pooled, np.cumprod(1.0 + pooled), np.zeros_like(pooled), np.zeros_like(pooled),
        bars_per_year=BPY, n_trials=N_TRIALS, trial_sr_std=TRIAL_SR_STD,
    )
    print(f"   annualised Sharpe        {st['sharpe']:+.3f}")
    print(f"   bootstrap 95% CI         [{st['sharpe_ci95'][0]:+.3f}, {st['sharpe_ci95'][1]:+.3f}]")
    print(f"   bootstrap p(SR<=0)       {st['bootstrap_p_value']:.4f}")
    print(f"   Newey-West t-stat        {st['newey_west_t']:+.3f}")
    print(f"   probabilistic SR         {m_pooled.psr:.4f}")
    print(f"   deflated SR ({N_TRIALS} trials) {m_pooled.dsr:.4f}")
    out["statistics"] = {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in st.items()},
        "psr": m_pooled.psr,
        "dsr": m_pooled.dsr,
    }

    # --- 3. negative controls ---------------------------------------------
    print("\n[3] NEGATIVE CONTROLS  (tier=vip6) — these must be ~0")
    ctrl_grid = [(s, k) for k in ("random_walk", "block_bootstrap") for s in TEST_SEEDS[:4]]
    with Pool(4) as p:
        ctrl_rows = p.map(job_control, ctrl_grid)
    controls: dict[str, dict] = {}
    for kind in ("random_walk", "block_bootstrap"):
        vals = [m for k, _, m in ctrl_rows if k == kind]
        sr = np.array([v["sharpe"] for v in vals])
        controls[kind] = {"sharpe_mean": float(sr.mean()), "sharpe_std": float(sr.std()),
                          "trades": float(np.mean([v["n_trades"] for v in vals]))}
        print(f"   {kind:18s} sharpe {sr.mean():+.2f} +/- {sr.std():.2f}   "
              f"trades {controls[kind]['trades']:.0f}")
    real = tier_summary["vip6"]["sharpe_mean"]
    print(f"   {'simulated market':18s} sharpe {real:+.2f}  <-- must clearly exceed the controls")
    out["controls"] = controls

    # --- 4. walk-forward ---------------------------------------------------
    print(f"\n[4] WALK-FORWARD  ({N_FOLDS_WF} folds of a {N_BARS_WF:,}-bar series, "
          f"purged + embargoed, tier=vip6, fresh learner per fold)")
    with Pool(3) as p:
        wf_rows = p.map(job_walkforward, TEST_SEEDS[:3])
    wf_out = []
    for seed, folds, pooled_m, _ in wf_rows:
        srs = [f["sharpe"] for f in folds]
        print(f"   seed {seed}: folds {[f'{s:+.2f}' for s in srs]}  "
              f"pooled sharpe {pooled_m['sharpe']:+.2f}  positive folds {sum(s>0 for s in srs)}/{len(srs)}")
        wf_out.append({"seed": seed, "folds": folds, "pooled": pooled_m})
    out["walkforward"] = wf_out

    path = os.path.join(os.path.dirname(__file__), "..", "evaluation_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwritten: {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
