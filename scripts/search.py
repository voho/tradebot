"""Hyperparameter search on TRAINING seeds only.

Test seeds (100+) are never touched here; they are used once, by
``scripts/evaluate.py``, after the configuration is frozen.
"""
import sys, os, itertools, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from multiprocessing import Pool
from gtbot.data.synthetic import simulate
from gtbot.data.schema import validate
from gtbot.strategy import GameTheoreticStrategy, StrategyConfig
from gtbot.game.equilibrium import AmbiguityConfig
from gtbot.engine.backtest import run_backtest
from gtbot.engine.broker import ExecutionConfig
from gtbot.eval import metrics

TRAIN_SEEDS = [0, 1, 2, 3]
NBARS = 150_000
BPY = 365 * 288
# Mild ambiguity set for the search: the point here is to compare
# configurations, and a sizer that vetoes everything compares nothing.
AMB = AmbiguityConfig(k_sigma=0.5, model_haircut=0.9, risk_aversion=0.06)

def one(args):
    horizon, entry, seed = args
    s = simulate(NBARS, seed=seed)
    bars = validate(s.bars)
    cfg = StrategyConfig(horizon=horizon, entry_signal=entry, max_hold=horizon,
                         assumed_cost_bp=6.65, ambiguity=AMB)
    res = run_backtest(bars, GameTheoreticStrategy(cfg),
                       execution=ExecutionConfig(entry_mode="taker", exit_mode="maker", ttl_bars=1),
                       max_leverage=2.0)
    m = metrics.compute(res.returns, res.equity, res.position, res.costs,
                        bars_per_year=BPY, n_trades=res.n_trades)
    return dict(horizon=horizon, entry=entry, seed=seed, sharpe=m.sharpe, cagr=m.cagr,
                vol=m.ann_vol, mdd=m.max_drawdown, trades=m.n_trades,
                cost=m.cost_drag_annual)

if __name__ == "__main__":
    grid = list(itertools.product([3, 6, 12], [0.55, 0.70], TRAIN_SEEDS))
    with Pool(4) as p:
        rows = p.map(one, grid)
    agg = {}
    for r in rows:
        agg.setdefault((r['horizon'], r['entry']), []).append(r)
    print(f"{'h':>3s} {'entry':>6s} {'sharpe_mean':>12s} {'min':>7s} {'cagr':>8s} {'vol':>7s} {'mdd':>7s} {'trades':>7s} {'cost':>7s}")
    out = []
    for k, v in sorted(agg.items()):
        sh = np.array([x['sharpe'] for x in v])
        rec = dict(horizon=k[0], entry=k[1], sharpe_mean=float(sh.mean()), sharpe_min=float(sh.min()),
                   cagr=float(np.mean([x['cagr'] for x in v])), vol=float(np.mean([x['vol'] for x in v])),
                   mdd=float(np.mean([x['mdd'] for x in v])), trades=float(np.mean([x['trades'] for x in v])),
                   cost=float(np.mean([x['cost'] for x in v])))
        out.append(rec)
        print(f"{k[0]:3d} {k[1]:6.2f} {sh.mean():+12.2f} {sh.min():+7.2f} {rec['cagr']:+8.2%} "
              f"{rec['vol']:7.2%} {rec['mdd']:7.2%} {rec['trades']:7.0f} {rec['cost']:7.2%}")
    best = max(out, key=lambda r: r['sharpe_mean'])
    print(f"\nbest on training: horizon={best['horizon']} entry={best['entry']} sharpe={best['sharpe_mean']:+.2f}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'search_results.json'), 'w'), indent=1)
