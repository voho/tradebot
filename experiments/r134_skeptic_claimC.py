"""R-134 skeptic: independent spot-check of Claim C (F1 bit-identical at
DEADBAND_BASELINE=0.05) for kelly_regime_v4, futures_5x, inner-train,
through both branches' patched brokers vs plain PaperBroker."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import r134_shared as SH
from r131_shared import load_btc_train, _assert_no_holdout, INNER_TRAIN_START, INNER_TRAIN_END
from tradebot.registry import get_strategy
from tradebot.window import run_period
from tradebot.metrics import compute_metrics
from r134_conservative_market_deadband import deadband_broker, _patched_broker as cons_patched
from r134_novel_accumulate_release import broker_cls_at as novel_broker_cls_at

df, label = load_btc_train()
_assert_no_holdout(df)

FUTURES = SH.FUTURES
DB = SH.DEADBAND_BASELINE

res_plain = run_period(get_strategy("kelly_regime_v4"), df, INNER_TRAIN_START, INNER_TRAIN_END,
                        market=FUTURES, start_balance=1000.0)
m_plain = compute_metrics(res_plain)

with cons_patched(deadband_broker(DB)):
    res_cons = run_period(get_strategy("kelly_regime_v4"), df, INNER_TRAIN_START, INNER_TRAIN_END,
                           market=FUTURES, start_balance=1000.0)
m_cons = compute_metrics(res_cons)

with cons_patched(novel_broker_cls_at(DB)):
    res_novel = run_period(get_strategy("kelly_regime_v4"), df, INNER_TRAIN_START, INNER_TRAIN_END,
                            market=FUTURES, start_balance=1000.0)
m_novel = compute_metrics(res_novel)

print("plain: fills=", len(res_plain.fills), "final=", m_plain.final_balance, "sharpe=", m_plain.sharpe)
print("conservative(0.05): fills=", len(res_cons.fills), "final=", m_cons.final_balance, "sharpe=", m_cons.sharpe)
print("novel(0.05): fills=", len(res_novel.fills), "final=", m_novel.final_balance, "sharpe=", m_novel.sharpe)

print("plain vs conservative equity allclose:", np.allclose(res_plain.equity.to_numpy(), res_cons.equity.to_numpy(), atol=1e-9))
print("plain vs novel equity allclose:", np.allclose(res_plain.equity.to_numpy(), res_novel.equity.to_numpy(), atol=1e-9))
