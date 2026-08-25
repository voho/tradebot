"""R-134 skeptic: independent re-run of Claim A (falsification cell) on futures_5x
at DEADBAND_REALISTIC, using r134_shared.paired_b1 directly, and via the
conservative branch's MarketDeadbandBroker (independent of the novel branch's
AccumulateReleaseBroker, to cross-check both implementations agree)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import r134_shared as SH
from r131_shared import load_btc_train, _assert_no_holdout
from r133_mechanisms import NovelTurnoverThrottle
import tradebot.engine as engine_mod
from r134_conservative_market_deadband import deadband_broker, _patched_broker as cons_patched
from r134_novel_accumulate_release import broker_cls_at as novel_broker_cls_at

df, label = load_btc_train()
_assert_no_holdout(df)

FUTURES = SH.FUTURES
DB = SH.DEADBAND_REALISTIC

v4 = SH.v4_reference(df, FUTURES)

def throttle_factory():
    return lambda: NovelTurnoverThrottle(upper=SH.THROTTLE_UPPER, eta=SH.THROTTLE_ETA)

# via conservative broker
with cons_patched(deadband_broker(DB)):
    thr_cons = SH.b1_throttle_vs_v4(throttle_factory(), df, FUTURES)
b1_cons = SH.paired_b1(thr_cons["returns"].to_numpy(), v4["returns"].to_numpy())
print("CONSERVATIVE broker, futures_5x, DEADBAND_REALISTIC:", b1_cons)
print("  sharpe_thr=", thr_cons["metrics"].sharpe, "sharpe_v4=", v4["metrics"].sharpe)
print("  fills_thr=", len(thr_cons["result"].fills), "fills_v4=", len(v4["result"].fills))

# via novel broker
with cons_patched(novel_broker_cls_at(DB)):
    thr_novel = SH.b1_throttle_vs_v4(throttle_factory(), df, FUTURES)
b1_novel = SH.paired_b1(thr_novel["returns"].to_numpy(), v4["returns"].to_numpy())
print("NOVEL broker, futures_5x, DEADBAND_REALISTIC:", b1_novel)
print("  sharpe_thr=", thr_novel["metrics"].sharpe, "sharpe_v4=", v4["metrics"].sharpe)
print("  fills_thr=", len(thr_novel["result"].fills), "fills_v4=", len(v4["result"].fills))

print("\nfills_thr identical between two broker impls:", len(thr_cons["result"].fills) == len(thr_novel["result"].fills))
