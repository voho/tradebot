"""R-134 skeptic: randomized stress test of Claim B (AccumulateReleaseBroker
equivalence to hard-drop PaperBroker at the SAME deadband threshold), calling
_execute_target directly bar-by-bar with a random walk of targets that
includes sign flips, close-to-flat, tiny same-sign adjustments, and large
same-sign jumps -- exactly the edge cases the audit was asked to probe."""
import sys
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from tradebot.broker import MarketSpec, PaperBroker
import tradebot.broker as broker_mod
from r134_novel_accumulate_release import AccumulateReleaseBroker

random.seed(0)

def run_stock(market, targets, prices, deadband):
    orig = broker_mod.REBALANCE_DEADBAND
    broker_mod.REBALANCE_DEADBAND = deadband
    try:
        b = PaperBroker(market=market, start_balance=1000.0)
        fills = []
        for i, (t, p) in enumerate(zip(targets, prices)):
            fills.append(b._execute_target(t, i, p))
    finally:
        broker_mod.REBALANCE_DEADBAND = orig
    return b, fills

def run_accrel(market, targets, prices, deadband):
    b = AccumulateReleaseBroker(market=market, start_balance=1000.0, deadband=deadband)
    fills = []
    for i, (t, p) in enumerate(zip(targets, prices)):
        fills.append(b._execute_target(t, i, p))
    return b, fills

N_TRIALS = 200
N_BARS = 60
mismatches = 0
for trial in range(N_TRIALS):
    rng = random.Random(trial)
    allow_short = rng.random() < 0.5
    market = MarketSpec.futures(leverage=5.0) if allow_short else MarketSpec.spot()
    deadband = rng.choice([0.0, 0.001, 0.01, 0.05, 0.1])
    targets = []
    lo = -1.0 if allow_short else 0.0
    cur = 0.0
    for i in range(N_BARS):
        move = rng.choice(["tiny", "small", "big", "flip", "flat", "same"])
        if move == "tiny":
            cur = max(lo, min(1.0, cur + rng.uniform(-0.005, 0.005)))
        elif move == "small":
            cur = max(lo, min(1.0, cur + rng.uniform(-0.03, 0.03)))
        elif move == "big":
            cur = max(lo, min(1.0, cur + rng.uniform(-0.5, 0.5)))
        elif move == "flip":
            cur = -cur if cur != 0 else rng.uniform(-1, 1)
            cur = max(lo, min(1.0, cur))
        elif move == "flat":
            cur = 0.0
        # "same": no change
        targets.append(cur)
    prices = [100.0 * (1.0 + rng.uniform(-0.02, 0.02)) ** i for i in range(N_BARS)]

    b_stock, fills_stock = run_stock(market, targets, prices, deadband)
    b_acc, fills_acc = run_accrel(market, targets, prices, deadband)

    same_fill_shape = [len(f) for f in fills_stock] == [len(f) for f in fills_acc]
    same_qty = all(
        len(fs) == len(fa) and all(abs(x.qty - y.qty) < 1e-9 and x.side == y.side for x, y in zip(fs, fa))
        for fs, fa in zip(fills_stock, fills_acc)
    )
    same_pos = abs(b_stock.pos - b_acc.pos) < 1e-9
    same_cash = abs(b_stock.cash - b_acc.cash) < 1e-6
    if not (same_fill_shape and same_qty and same_pos and same_cash):
        mismatches += 1
        print(f"MISMATCH trial={trial} market={market.name} deadband={deadband}")
        print("  targets:", [round(t,4) for t in targets])
        print("  stock pos/cash:", b_stock.pos, b_stock.cash)
        print("  accrel pos/cash:", b_acc.pos, b_acc.cash)

print(f"\n{N_TRIALS} trials, {mismatches} mismatches.")
