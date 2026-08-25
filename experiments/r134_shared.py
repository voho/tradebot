"""Shared, frozen pre-registration for the R-134 round (08-25).

=====================================================================
DIRECTION
=====================================================================

Closes **B-43** (filed by R-133): `broker.REBALANCE_DEADBAND = 0.05`
(`src/tradebot/broker.py:61`, read at line 235 inside `_execute_target`,
mirrored as `multi_engine.TOTAL_NOTIONAL_DEADBAND`) is a hard-coded module
global, not a per-market or per-mechanism parameter. Any mechanism whose
action is to *shrink* a re-target -- rather than change its sign or size
by a large jump -- gets rounded to "fires at full size" or "silently
dropped" by this floor, in a proportion that moves with how hard the
mechanism itself pushes. R-130's skeptic measured 96.8% absorption for a
`hedge_experts` wrapper's `order_target` calls; R-133 measured 4.5%
absorption for unthrottled `kelly_regime_v4` against 61-83% for its own
`NovelTurnoverThrottle` branch at the frozen corridor. **The floor is not
a constant offset that cancels in a comparison -- it scales with the
mechanism's own parameter**, which means every SIZE-shrinking COST-axis
verdict this project has recorded (R-128-R-133, at minimum) was measured
through an instrument whose own resolution was never characterized.

**Which constraint this attacks: COST**, by way of methodology --
identical framing to B-30 (R-64/R-72), which characterized the SAME floor
for a completely different question (spot-vs-futures fill-through parity)
and is explicitly NOT a duplicate of this item (B-43's own ledger row says
so). Not a duplicate of B-29 (R-66, the snap-to-flat *destination* on the
single-asset rebalance rule -- a different line of `_execute_target`
entirely) or of B-30/R-72 itself (diagnosed the floor's existence and
tested one candidate *fixed-scaling* correction -- REJECTED for changing
`kelly_regime_v4`'s own realized numbers, fees rise and growth/Sharpe/DD
all move the wrong way on point estimate, though every interval contained
zero). **This round is not "does loosening the floor help v4" -- R-72
already answered a close variant of that and the answer was no. This round
is "does the floor's coarseness explain why an already-closed
SIZE-shrinking mechanism read as inert/negative."**

=====================================================================
OBJECT UNDER TEST
=====================================================================

`NovelTurnoverThrottle` (`experiments/r133_mechanisms.py`, frozen and
unmodified, imported not copied), R-133's own dual-variable shrink
mechanism: `shrink pending rebalance by 1/(1+lambda_t)` rather than a hard
skip. R-131/R-133 both found this SPLITS turnover rather than reducing it
(fill-ratio 0.863-1.125, i.e. non-monotone and at the tightest corridor
the throttled branch trades **12.5% MORE** than the strategy it throttles)
-- exactly the signature a coarse floor rounding "shrink by 60%" to
"either the full move or nothing" would produce, since a shrunk order
either still clears a 5%-of-max-notional threshold (fires at its
un-shrunk-adjacent size) or falls under it (vanishes, to reappear once the
next bar's *unshrunk* pending delta re-crosses the floor). This is the
single most direct, already-measured, already-negative candidate this
project has for testing whether B-43's confound is load-bearing.

Secondary, optional object (attempt only if time remains after the
primary battery below): `hedge_experts`'s own `order_target` calls
(R-130's 96.8%-absorption finding). Do not let this block or dilute the
primary report.

=====================================================================
TWO CANDIDATE FIXES (the backlog item's own framing, verbatim)
=====================================================================

- **CONSERVATIVE** (`r134_conservative_market_deadband.py`): make the
  floor a `MarketSpec` field (`deadband: float = REBALANCE_DEADBAND`,
  backward-compatible default), read at `_execute_target` in place of the
  module global. A branch (or a future round) can then set it to a
  venue-realistic value instead of the arbitrary flat 5%. Pure
  configuration change -- the underlying hard-drop-below-threshold
  *policy* is untouched, only its threshold becomes settable.

- **NOVEL** (`r134_novel_accumulate_release.py`): replace the hard drop
  with an accumulate-and-release rule. A same-sign adjustment that would
  be dropped is instead banked into a running per-broker accumulator;
  every subsequent bar re-evaluates `pending + newly-desired delta`
  against the SAME threshold, and releases the full accumulated delta as
  one order once it crosses the threshold (or immediately on a sign flip
  or a close-to-flat target, which must never be blocked). Suppressed
  intent is carried forward rather than discarded outright -- a
  structurally different fix from merely shrinking the same threshold.

Both fixes are implemented as broker subclasses / a monkeypatch of
`tradebot.engine.PaperBroker` for the duration of a run (the pattern R-72's
`EquityScaledDeadbandBroker` already established), never as edits to
`src/tradebot/broker.py` itself -- that stays untouched until the operator
selects a winner after both branches report, exactly as the parallel-round
rule in `docs/ROUTINE.md` requires (disjoint files, no branch commits).

=====================================================================
FALSIFICATION TEST, chosen now, before either branch runs any number
=====================================================================

Does correcting the deadband confound REVERSE R-133's own frozen
`NovelTurnoverThrottle` NEGATIVE verdict -- i.e., under the corrected
broker (deadband set to `DEADBAND_REALISTIC` below, or accumulate-release
with the same threshold), does the throttle now clear **B1** (paired
bootstrap vs `kelly_regime_v4`, inner-validation, `total_log_return`,
`significant=True` AND `paired_diff.point > 0`) on **both markets**
(spot, futures_5x)?

- **YES on both markets**: the COST-axis turnover-throttle family's own
  NEGATIVE verdict (R-131, R-133) was, at least in part, a broker-floor
  artifact rather than a property of the mechanism -- reopens the family,
  requires its own follow-up round with a full B1/B3/B4/B5 battery before
  any promotion claim (this round does not promote anything by itself).
- **NO (still fails B1 on at least one market)**: B-43 closes cleanly.
  R-133's section C entry becomes final rather than provisional, per its
  own text. The evaluability defect is still worth fixing (see the
  ADOPTION rule below) because it affects every FUTURE size-shrinking
  mechanism this project tries, but it does not resurrect this one.

=====================================================================
DECISION RULE for ADOPTING a fix into the permanent framework
(`src/tradebot/broker.py`), pre-registered before either branch runs
=====================================================================

This round is an infrastructure item, not a new-strategy promotion --
B-43's own text: "the deliverable is a measured before/after on an
already-closed mechanism, not a new strategy." The thing being decided is
whether a candidate fix is safe and useful to adopt, not whether a
strategy beats `buy_and_hold`. A fix is **ADOPTED** only if ALL of:

  F1. **Backward compatibility.** At the CURRENT default deadband value
      (0.05, fraction of max notional), the fix does not move
      `kelly_regime_v4`'s or `hedge_experts`'s inner-train + inner-val
      metrics (final balance, Sharpe, max drawdown, trade count) on
      either market by more than the +/-0.2 Sharpe noise floor (R-20).
      The conservative fix is expected to be BIT-IDENTICAL at the default
      (a pure refactor); the novel fix cannot be bit-identical by
      construction (it carries suppressed intent forward instead of
      discarding it) and must instead clear the noise-floor bar.
  F2. **No regressions.** Full `pytest` green, including
      `tests/test_causality_strict.py`.
  F3. **Demonstrated capability.** Configured at `DEADBAND_REALISTIC`
      (below), the fix measurably changes the fill-through / absorption
      rate for `NovelTurnoverThrottle` relative to the current floor (i.e.
      it actually does what it is for).

If both fixes clear F1-F3, the operator selects between them on: which
more directly addresses the floor's own failure mode (silent, binary
rounding of a continuous shrink) without adding a new free parameter or a
persistent-state edge case the causality/no-lookahead suite does not
already cover. If neither clears F1-F3, B-43 is answered but NOT closed as
adopted -- the honest output named in its own row: "a documented ceiling
on what a size-acting mechanism can be shown to do here."

No bar at or after `OOS_START = 2023-01-01` is read anywhere in this
round. This is an inner-train/inner-validation-only round; nothing here
earns a holdout consultation regardless of outcome, because the falsification
test's own decisive cell is B1 on inner-validation, matching the SIZE/
ERR/COST family's own standing holdout-access gate (a round only reaches
the holdout if A2 + B1 + B3 + B4 + B5 all clear first, and this round does
not attempt B3/B4/B5 by design -- a reversal on B1 alone is reported as
grounds for a FOLLOW-UP round, not as grounds to read the holdout here).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec, REBALANCE_DEADBAND  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# R-131/R-133's own constants, imported not re-derived, so the re-run is
# the SAME mechanism at the SAME operating point -- only the broker layer
# underneath it changes.
from r131_shared import (  # noqa: E402
    ETA,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    LAMBDA_MAX,
    OOS_START,
    TURNOVER_UPPER,
    V4_NATURAL_TRADES_PER_DAY,
    _assert_no_holdout,
    load_btc_train,
    load_eth_train,
)

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
SHARPE_NOISE_FLOOR = 0.2  # R-20

# The current, unconditional default -- 5% of max notional (equity x leverage).
DEADBAND_BASELINE = REBALANCE_DEADBAND  # 0.05

# A venue-realistic alternative: MarketSpec.min_notional=$5 at $1000 start
# equity, futures 5x -> max_notional=$5000 -> $5/$5000 = 0.1%. Spot (1x) ->
# $5/$1000 = 0.5%. One shared, deliberately-small value is used across both
# markets for a like-for-like comparison rather than two different floors.
DEADBAND_REALISTIC = 0.001

# Full sweep for the plateau check (R-134's own honesty requirement: report
# neighbours, not just the two endpoints).
DEADBAND_GRID = [0.0, 0.001, 0.005, 0.01, 0.02, DEADBAND_BASELINE]

# R-133's own frozen primary operating point for NovelTurnoverThrottle.
THROTTLE_UPPER = 3.0 * V4_NATURAL_TRADES_PER_DAY
THROTTLE_ETA = ETA
THROTTLE_LAMBDA_MAX = LAMBDA_MAX

# Counts every backtest run through `b1_throttle_vs_v4` or `noise_floor_check`
# across BOTH branches -- the trials count for this round, observed rather
# than remembered. Each branch file imports and increments the SAME list
# object (shared module state), so the operator's final count is the true
# cross-branch total per the parallel-round rule in ROUTINE.md.
_CONFIGS = [0]


def note_config() -> None:
    _CONFIGS[0] += 1


def configs_evaluated() -> int:
    return _CONFIGS[0]


def b1_throttle_vs_v4(throttle_factory, df: pd.DataFrame, market: MarketSpec,
                       seed: int = 134) -> dict:
    """Paired bootstrap, NovelTurnoverThrottle (under whatever broker is
    monkeypatched into `tradebot.engine` at call time) vs frozen
    `kelly_regime_v4` (always the DEFAULT, unpatched broker -- the
    incumbent this project has always compared against), inner-validation,
    `total_log_return`. Identical statistic and window to R-131's own
    `b1_signal`, so the two are directly comparable."""
    note_config()
    strat = throttle_factory()
    res_thr = run_period(strat, df, INNER_VAL_START, INNER_VAL_END,
                          market=market, start_balance=1000.0)
    m_thr = compute_metrics(res_thr)
    r_thr = daily_returns(res_thr.equity)
    return {"metrics": m_thr, "returns": r_thr, "result": res_thr}


def v4_reference(df: pd.DataFrame, market: MarketSpec) -> dict:
    """Frozen kelly_regime_v4 on the SAME window/market, always through the
    unpatched default broker -- the fixed comparison point for every B1
    cell in this round, computed once per (df, market) pair by the caller
    and reused."""
    note_config()
    res_v4 = run_period(lambda_or_strategy(), df, INNER_VAL_START, INNER_VAL_END,
                         market=market, start_balance=1000.0)
    m_v4 = compute_metrics(res_v4)
    r_v4 = daily_returns(res_v4.equity)
    return {"metrics": m_v4, "returns": r_v4, "result": res_v4}


def lambda_or_strategy():
    return get_strategy("kelly_regime_v4")


def paired_b1(thr_returns: np.ndarray, v4_returns: np.ndarray, seed: int = 134) -> dict:
    n = min(len(thr_returns), len(v4_returns))
    paired = paired_bootstrap(thr_returns[:n], v4_returns[:n], stat=total_log_return, seed=seed)
    return {
        "paired_diff": paired.diff.point,
        "paired_lo": paired.diff.lo,
        "paired_hi": paired.diff.hi,
        "significant": paired.significant,
        "b1_pass": bool(paired.significant and paired.diff.point > 0),
    }


if __name__ == "__main__":
    df_check, _ = load_btc_train()
    _assert_no_holdout(df_check)
    print(f"R-134 shared pre-registration loaded OK. "
          f"DEADBAND_BASELINE={DEADBAND_BASELINE}, DEADBAND_REALISTIC={DEADBAND_REALISTIC}, "
          f"grid={DEADBAND_GRID}")
    print(f"max BTC-train timestamp read: {df_check.index.max()} (< {OOS_START})")
