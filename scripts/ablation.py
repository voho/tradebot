"""Ablation of the proposed improvements, on TRAINING seeds only.

Each improvement is toggled independently against the shipped baseline so its
individual contribution is visible, and then the survivors are combined.  Test
seeds are not touched here; ``scripts/evaluate.py`` runs the winning
configuration on held-out data afterwards.

The improvements come from the search-and-learning literature on imperfect
information games, principally the CFR+/DCFR line and AIVAT:

``dcfr``
    Discounted regret matching+ instead of Hedge (Tammelin 2014; Brown &
    Sandholm 2019).  No learning rate to mis-set, and asymmetric discounting of
    positive and negative regret lets an expert whose sign has flipped recover
    in tens of observations rather than tens of thousands.

``varred``
    An AIVAT-style control variate (Burch, Schmid et al. 2018) subtracted from
    each realised trade payoff before it reaches the edge estimator.  The flow
    that arrives after entry is this game's chance node: unpredictable at entry,
    so the return it explains has zero conditional mean and can be removed
    without bias, shrinking the estimator's standard error.  That matters
    directly because the sizer allocates on ``edge - k * SE``.

``resolve``
    A one-ply depth-limited re-solve of the exit decision against a learned
    continuation value, in the spirit of DeepStack's continual re-solving,
    replacing the fixed holding period.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from gtbot.data.schema import validate
from gtbot.data.synthetic import simulate
from gtbot.engine.broker import ExecutionConfig
from gtbot.eval.account import simulate_account
from gtbot.game.regret import LearnerConfig
from gtbot.strategies import get as get_preset
from gtbot.strategy import StrategyConfig

TRAIN_SEEDS = [0, 1, 2, 3]
N_BARS = 150_000
TIER = "vip9"
EXECUTION = ExecutionConfig(entry_mode="taker", exit_mode="maker", ttl_bars=1)

# Layered on the shipped preset so the ablation compares against exactly the
# configuration that ships, not a restatement of it that can drift.
_PRESET = get_preset("dislocation_v2")
BASE = dict(horizon=_PRESET.config.horizon, entry_signal=_PRESET.config.entry_signal,
            max_hold=_PRESET.config.max_hold)

#: name -> kwargs layered on top of the baseline.
#:
#: Regret matching spreads weight across every action with positive cumulative
#: regret where Hedge concentrates exponentially, so a DCFR blend is smaller in
#: absolute terms for the same conviction.  Comparing it against Hedge at a
#: fixed entry threshold would reject it for a scale artefact rather than on
#: merit, so it is also tried at thresholds matched to its own signal scale.
VARIANTS: dict[str, dict] = {
    "baseline": dict(variance_reduction=False),
    "+varred": dict(variance_reduction=True),
    "+resolve": dict(variance_reduction=False, adaptive_exit=True),
    "varred+resolve": dict(variance_reduction=True, adaptive_exit=True),
    "dcfr@0.15": dict(variance_reduction=False, entry_signal=0.15,
                      learner=LearnerConfig(rule="dcfr")),
    "dcfr@0.25": dict(variance_reduction=False, entry_signal=0.25,
                      learner=LearnerConfig(rule="dcfr")),
    "dcfr@0.35": dict(variance_reduction=False, entry_signal=0.35,
                      learner=LearnerConfig(rule="dcfr")),
}


def one(args):
    name, seed = args
    bars = validate(simulate(N_BARS, seed=seed).bars)
    cfg = StrategyConfig(**{**BASE, **VARIANTS[name]})
    res = simulate_account(
        bars, tier=TIER, direction="both", sizing_mode="robust",
        leverage=5.0, deposit=1_000.0, config=cfg, execution=EXECUTION,
    )
    return name, seed, res


if __name__ == "__main__":
    grid = list(itertools.product(VARIANTS, TRAIN_SEEDS))
    with Pool(4) as p:
        rows = p.map(one, grid)

    by_name: dict[str, list] = {}
    for name, seed, res in rows:
        by_name.setdefault(name, []).append(res)

    print(f"ABLATION on training seeds {TRAIN_SEEDS}, tier={TIER}, "
          f"$1,000 at 5x over {N_BARS / (365 * 288):.2f}y")
    print(f"{'variant':>12s} {'sharpe':>8s} {'min':>7s} {'final $':>9s} {'P&L $':>8s} "
          f"{'maxDD $':>8s} {'trades':>7s} {'edge SE':>9s}")
    summary = {}
    for name in VARIANTS:
        rs = by_name[name]
        sr = np.array([r.sharpe for r in rs])
        rec = dict(
            sharpe_mean=float(sr.mean()), sharpe_min=float(sr.min()),
            final=float(np.mean([r.final_equity for r in rs])),
            pnl=float(np.mean([r.profit_usd for r in rs])),
            dd=float(np.mean([r.max_drawdown_usd for r in rs])),
            trades=float(np.mean([r.trades for r in rs])),
        )
        summary[name] = rec
        print(f"{name:>12s} {rec['sharpe_mean']:+8.2f} {rec['sharpe_min']:+7.2f} "
              f"{rec['final']:9,.0f} {rec['pnl']:+8,.0f} {rec['dd']:8,.0f} "
              f"{rec['trades']:7.0f}")

    best = max(summary, key=lambda k: summary[k]["sharpe_mean"])
    print(f"\nbest on training: {best} (sharpe {summary[best]['sharpe_mean']:+.2f})")
    json.dump(summary, open(os.path.join(os.path.dirname(__file__), "..",
                                         "ablation_results.json"), "w"), indent=1)
