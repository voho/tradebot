"""Shared, read-only utilities and pre-registration for the R-129 round (08-25).

DIRECTION, in one sentence: R-128 replaced `hedge_experts`'s fixed
`hysteresis=0.05` re-target threshold with a Kelly quadratic-cost/linear-fee
no-trade band (Constantinides 1986; Davis & Norman 1990) applied to the
already-Hedge-BLENDED output `x`, at a single frozen horizon -- and found it
NEGATIVE on the exact risk it pre-registered: "the Kelly algebra assumes one
homogeneous stationary bet, and hedge_experts blends ten experts across four
timescales." R-128's own closing line named the untested alternative
explicitly: "a band construction that does not assume one homogeneous
target (e.g. a per-expert or per-timescale no-trade rule rather than one
band on the already-blended output)." This round tests exactly that,
splitting it into the two constructions R-128 itself named.

**Why this and not a `kelly_regime_v4`/`champions_council` variant.** Per
this round's own Step-0 diligence: the single-asset `kelly_regime_v4` axis
is closed across INFO (19+ signals), SIZE (28+ attempts), ERR (5 notions of
uncertainty), regime-timing (11 mechanisms) and N-approx-3 calibration (4
procedures); the multi-asset panel axis is closed (11 rounds); `champions_
council`'s own cross-strategy allocation was tried and closed NEGATIVE
(R-126, both branches); R-127 diagnosed the recurring BTC-pass/ETH-invert
signature itself. `hedge_experts`'s own re-target rule (this round's object)
was tried ONCE, at the blended-output level only (R-128, NEGATIVE, both
branches) -- its own re-ranking names the pre-blend construction as "a
different, untested question," not a closed axis.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r129_conservative_per_expert_band.py`): apply the EV band
  to EACH of `hedge_experts`'s ten raw expert signals `a[:, j]`
  INDIVIDUALLY, before the Hedge weights `p` blend them -- each expert's
  contribution to the blend is only allowed to move when its own change
  clears an EV band computed from a horizon *structural to that expert's own
  native timescale* (its lookback window in bars, converted to days; see
  `EXPERT_HORIZON_DAYS` below), not from the strategy's pooled fill spacing.
  The market's realized volatility `sig1` (shared across all ten -- the
  quadratic-cost derivation is a function of MARKET return variance, not any
  one indicator's own units) and the live market fee/leverage are unchanged
  inputs. The final assembled signal `x = p @ a_banded[i]` is placed via
  `ctx.order_target` every bar it moves, with NO second, post-blend band --
  isolating the effect of pre-blend damping alone.

- NOVEL (`r129_novel_bucket_band.py`): group the ten experts into three
  TIMESCALE BUCKETS by their own native period (see `EXPERT_BUCKET` /
  `BUCKET_HORIZON_DAYS` below: FAST = sub-daily signals, SLOW = multi-day
  signals, STATIC = the two signals that never meaningfully change), compute
  each bucket's own Hedge-weighted SUB-BLEND (`x_bucket = sum(p_j * a_j) for
  j in bucket`), and apply ONE EV band per BUCKET (three bands total, not
  ten) derived from that bucket's own structural horizon, to the sub-blend
  rather than to each raw expert. The final target is the sum of the three
  (independently banded) bucket sub-blends, placed via `ctx.order_target`
  every bar it moves, again with no second band on the sum. This is a
  materially different construction from the conservative branch, not a
  coarser copy of it: it bands three GROUP-LEVEL aggregates computed
  post-weighting-within-group, not ten individual raw signals pre-weighting.

Both derive H STRUCTURALLY (from each expert's own lookback/decay
construction, computed once, frozen below, identical for both branches where
they overlap) rather than fitting either to any realized return -- the same
"not tuned, it falls out of the architecture" discipline `kelly_regime_ev`
and R-128's own conservative branch used.

**Literature grounding for splitting the band at all** (WebSearch, this
session, before either branch was written): Ekren, Liu & Muhle-Karbe
(2018), "Optimal Rebalancing Frequencies for Multidimensional Portfolios",
*Mathematics and Financial Economics* 12(1), 1-24 (arXiv:1510.05097) --
extending the classical single-asset no-trade-region result (Constantinides
1986; Davis & Norman 1990, the derivation `kelly_regime_ev` and R-128 both
use) to MULTIPLE co-traded objects, and finding the jointly optimal
no-trade region is generally WIDER and structurally different from the
Cartesian product of each object's own univariate band -- i.e. the
literature's own answer to "does one pooled band or several per-object
bands span the right no-trade geometry" is neither trivially, which is
exactly the open question both branches below test empirically rather than
assume. This is cited for the general shape of the multivariate problem,
not reproduced: Ekren et al. solve a stochastic-control HJB system for
correlated diffusions, which this project's causal, bar-by-bar simulator
cannot evaluate; both branches below use the same closed-form quadratic-
cost heuristic `kelly_regime_ev` already validated on this data, applied
per-object/per-bucket instead of solving the joint control problem.

**Not a duplicate of:**
- R-128 (`r128_conservative_ev_band.py` / `r128_novel_adaptive_band.py`):
  both R-128 branches band the single ALREADY-BLENDED signal `x`, differing
  only in whether the ONE horizon feeding that ONE band is a fixed constant
  or an online AR(1)-persistence estimate. Neither ever bands anything
  before the blend, and neither ever uses more than one band. This round
  bands ten signals (conservative) or three bucket sub-blends (novel),
  always BEFORE the final sum, always with per-expert/per-bucket
  STRUCTURAL horizons -- a different object (what gets banded) and a
  different horizon source (architecture, not a fitted/measured constant),
  not a re-run of either R-128 branch with a smaller multiplier.
- `kelly_regime_ev`/`kelly_regime_ev_fast`: same algebra family, applied to
  `kelly_regime_v4`'s single homogeneous vote, not to any of `hedge_
  experts`'s ten experts or three buckets.
- B-29/B-31/B-34/B-35/B-37/B-40's entry/exit-band research line (R-64
  through R-69, R-108): `xsmom_entry_band`'s cross-sectional multi-asset
  panel eligibility threshold on `src/tradebot/multi_engine.py` -- a
  different object (panel selection, not a single-instrument expert blend)
  on a different engine.
- R-126 (`champions_council` ERC/CVaR allocation): varies the weight this
  project puts on several already-complete STRATEGIES; this round varies
  the pre-blend construction INSIDE one strategy's own expert ensemble and
  never touches cross-strategy allocation.

**What would make this fail, named now, before any code:**
1. **The sharpest named risk.** Hedge's multiplicative weights `p` update
   EVERY bar with no hysteresis of their own (`HedgeExperts.prepare`'s
   `logw`/`p` loop is unconditional). Damping the raw experts `a[:, j]`
   (conservative) or the group sub-blends (novel) does not stop `p` itself
   from moving continuously -- so `x = p @ a` (or `sum(x_bucket)`) can still
   change every bar purely from weight drift even when every banded input is
   frozen, which could leave turnover close to `hedge_experts`'s own
   baseline regardless of how tightly the pre-blend bands are set. This is
   the one risk that could make BOTH branches structurally unable to cut
   cost by construction, independent of whether the horizons chosen are
   good ones -- report each branch's own trade count against baseline's
   explicitly to check it head-on.
2. Very-short-horizon experts (1-bar reversion H=0.0035d, RSI H=0.049d,
   MACD H=0.090d) receive, by the `band = 2*fee/(H*sigma^2)` formula, very
   WIDE bands after clipping to `max_band=1.0` -- wide enough that these
   experts could freeze near-permanently at whatever value they held at
   warmup end, silently removing them from the conservative branch's
   ensemble rather than smoothing their turnover. Report each expert's own
   realized re-target count to check this directly rather than assume it.
3. Six-plus independent prior mechanisms/objects on this project have
   passed a BTC promotion gate and inverted sign on ETH (R-109, R-113,
   R-115-conservative, R-125-conservative, R-126 both branches, R-128
   conservative weakly) -- a real, named prior that a construction here does
   the same. B4 below is the test built to catch exactly this.
4. A band wide enough to matter could reproduce the LAG failure this
   project has now measured in every one of 11 regime-timing mechanisms on
   `kelly_regime_v4` -- slowing responsiveness enough to give back the edge
   that makes `hedge_experts` profitable on spot in the first place. R-128
   found no sign of this within its own 4x sweep; this round's B3 sweep
   checks the same question for this round's own construction.
5. The novel branch's bucket boundaries (FAST/SLOW/STATIC) are a structural
   judgement call, not fit to any return -- a materially different bucket
   assignment could plausibly change the result, which is a real, disclosed
   limit of testing only one partition rather than evidence the specific
   partition chosen is uniquely correct.

**Implementation note, binding on both branches (so both build a consistent,
correctly-causal state machine and neither repeats R-128's own unit bug):**

1. `prepare()` in both branches stays fully causal and MARKET-INDEPENDENT --
   it must never read `ctx.market` (it runs once, before any market is
   chosen). It reproduces `HedgeExperts.prepare()`'s expert construction
   (`self._experts(df, r, sig1)`, called verbatim, unmodified) and its
   `g`/`logw`/`p` Hedge weight-update loop EXACTLY (same `z_t`, `fee_n`
   using `self.fee_rate` -- the strategy's own internal turnover-belief
   constant used to score expert fitness, NOT the live trading fee charged
   by the broker, which stays untouched from the registered strategy).
   Emit the raw per-bar expert matrix (`expert_0`..`expert_9` columns) and
   the raw per-bar Hedge weight matrix (`weight_0`..`weight_9` columns) --
   MARKET-INDEPENDENT, so both spot and futures runs from the same branch
   read identical `prepare()` output -- plus `_ev_vol` (`sig1 *
   sqrt(BARS_PER_YEAR)`, shifted one bar, identical to R-128's own
   construction). No band, no hysteresis, and no final blended target is
   computed in `prepare()` -- that all happens in `on_bar`, where the live
   market's fee/leverage are available.
2. `on_bar` in both branches maintains the per-expert (conservative, length
   10) or per-bucket (novel, length 3) "held" values as INSTANCE state (a
   numpy array on `self`, e.g. `self._held`), never as a dataframe column --
   there is no broker-side equivalent of a per-expert position to read back
   the way R-128's single-band construction could read `ctx.position`
   directly, so this state must be tracked by the strategy itself.
   **Disclosed cold-start convention (both branches, state identically in
   each branch's own report):** `self._held` is initialized ONCE, on the
   FIRST `on_bar` call (i.e. at `warmup`, guarded by `if self._held is
   None`), to that bar's own raw (unbanded) values -- not accumulated from
   `prepare()`'s bar 2 the way the Hedge weight loop itself is. This is a
   minor, bounded cold-start artifact (warmup = 2,500 of ~404,000 inner-
   train bars, 0.6%), not a lookahead: every subsequent update uses only
   that bar's and earlier bars' data. Report it plainly rather than
   silently matching R-128's own (different, broker-state-based) warmup
   behavior.
3. Band formula, IDENTICAL shape to R-128's own POST-FIX, unit-consistent
   version (`current`/`desired` both in fraction-of-max-leverage units,
   matching `hedge_experts`'s native `ctx.order_target` convention -- avoid
   R-128's original unit bug from the start rather than discovering it
   post-hoc):
       band_j = clip(2*fee / (H_j_years * sigma_market**2 * leverage),
                      MIN_BAND, MAX_BAND)
   `fee = ctx.market.fee_rate`, `leverage = ctx.market.leverage`,
   `sigma_market = ctx.bar["_ev_vol"]` (the ONE shared market-vol input --
   see mechanism section above: the quadratic-cost derivation is a function
   of market return variance, not any one expert's own indicator units),
   `H_j_years = EXPERT_HORIZON_DAYS[j] / 365.25` (conservative) or
   `BUCKET_HORIZON_DAYS[bucket] / 365.25` (novel). `MIN_BAND = 0.02`,
   `MAX_BAND = 1.0` -- `kelly_regime_ev`'s and R-128's own literal
   defaults, reused unchanged, no new reason found to change them.
4. Final order, both branches: assemble the target from the (possibly
   just-updated) held values -- conservative: `x = weight @ held` (a
   10-term dot product, `weight` read live from `prepare()`'s
   `weight_0..9` columns each bar); novel: `x = sum(held_bucket.values())`
   where each bucket's own held value already IS its own
   Hedge-weighted sub-blend (`x_bucket = sum(weight_i * expert_i for i in
   bucket)`, computed fresh each bar from `prepare()`'s live columns, THEN
   compared to `self._held[bucket]` for the band test) -- then
   `ctx.order_target(x)` whenever `abs(x - self._last_target) > 1e-9`
   (`self._last_target` a third piece of instance state, updated after
   every order; the `1e-9` epsilon is float-noise dedup only, exactly the
   idiom `HedgeExperts.on_bar` and every `_Frozen`-style harness in this
   project's prior rounds already use -- NOT a second band. Both branches
   must report each expert's/bucket's own realized re-target count
   (failure modes 1-2 above) as a diagnostic table, not just the final
   blended trade count.

**Falsification test, pre-registered:** B4 -- does the candidate's
`d_sharpe` sign (candidate vs `hedge_experts`, inner-validation) replicate
on ETH? Chosen for continuity with the whole SIZE/ERR/COST research
programme since R-59/R-64/R-125/R-126/R-128.

**Decision rule, pre-registered, matching the SIZE/ERR/COST family's own
convention (R-64...R-128):** PROMOTE-candidate only if the causal-
truncation probe AND B1 (both markets, full period AND inner-validation)
AND B3 (plateau majority across a horizon-multiplier grid) AND B4 (sign
replicates on ETH) AND B5 (no sign flip at 0.40% fee) all pass. B2
(drawdown/turnover) and each expert's/bucket's own re-target count
(failure modes 1-2 above) are diagnostic only and never gate promotion by
themselves.

No bar at or after `OOS_START = 2023-01-01` may be read by either branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ----------------------------------------------------------------------
# Splits. Identical convention to every prior round: inner-train / inner-
# validation only. The holdout (>= OOS_START) is never read by a branch.
# ----------------------------------------------------------------------
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"  # do not read; guarded by _assert_no_holdout below

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT_HIGH_FEE = MarketSpec.spot(fee_rate=0.0040)      # B5: 0.40% taker tier
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_ev's / R-128's own literal band-clip defaults, reused
# unchanged by both branches -- no new reason found to change them.
MIN_BAND = 0.02
MAX_BAND = 1.0
B3_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)

# ----------------------------------------------------------------------
# Frozen per-expert horizons, STRUCTURAL (each expert's own lookback/decay
# construction in `HedgeExperts._experts`, converted bars -> days), computed
# once before either branch was dispatched -- never fit to a return.
# Column order matches `HedgeExperts._experts` EXACTLY (verified against
# `src/tradebot/strategies/hedge_experts.py`):
#   0: 1h  momentum (lookback  12 bars)      ->  12/288 = 0.041667 d
#   1: 6h  momentum (lookback  72 bars)      ->  72/288 = 0.250000 d
#   2: 1d  momentum (lookback 288 bars)      -> 288/288 = 1.000000 d
#   3: 1w  momentum (lookback 2016 bars)     -> 2016/288 = 7.000000 d
#   4: MACD histogram (slow EMA span 26 bars, `tradebot.indicators.macd`'s
#      own default -- the slow leg sets the indicator's characteristic decay
#      time) -> 26/288 = 0.090278 d
#   5: RSI ramp (period 14 bars, `tradebot.indicators.rsi`'s own default)
#      -> 14/288 = 0.048611 d
#   6: 1-bar mean reversion (reacts to the immediately-prior bar's return)
#      -> 1/288 = 0.003472 d
#   7: Donchian breakout (288-bar/1d entry lookback, decayed 0.99/bar once
#      inside the channel -- half-life = ln(0.5)/ln(0.99) = 68.968 bars)
#      -> 68.968/288 = 0.239471 d
#   8: always flat (constant 0.0 -- never moves, horizon is inert; nominal)
#   9: buy and hold (constant 1.0 -- never moves after warmup; nominal)
# ----------------------------------------------------------------------
EXPERT_HORIZON_DAYS = np.array([
    12.0 / 288.0,                          # 0: 1h momentum
    72.0 / 288.0,                          # 1: 6h momentum
    288.0 / 288.0,                         # 2: 1d momentum
    2016.0 / 288.0,                        # 3: 1w momentum
    26.0 / 288.0,                          # 4: MACD histogram
    14.0 / 288.0,                          # 5: RSI ramp
    1.0 / 288.0,                           # 6: 1-bar reversion
    (np.log(0.5) / np.log(0.99)) / 288.0,  # 7: Donchian breakout
    7.0,                                   # 8: always flat (nominal, inert)
    7.0,                                   # 9: buy and hold (nominal, inert)
])
assert EXPERT_HORIZON_DAYS.shape == (10,)

# ----------------------------------------------------------------------
# Frozen bucket assignment (novel branch), by native timescale, structural
# (not fit): FAST = sub-daily signals, SLOW = multi-day signals, STATIC =
# the two signals that are constant almost everywhere. Bucket horizon is
# the MEDIAN of its members' EXPERT_HORIZON_DAYS above (frozen, computed
# once).
# ----------------------------------------------------------------------
EXPERT_BUCKET = ["fast", "fast", "slow", "slow", "fast", "fast", "fast", "slow", "static", "static"]
assert len(EXPERT_BUCKET) == 10

BUCKET_HORIZON_DAYS = {
    bucket: float(np.median(EXPERT_HORIZON_DAYS[[i for i, b in enumerate(EXPERT_BUCKET) if b == bucket]]))
    for bucket in ("fast", "slow", "static")
}
# fast members (0,1,4,5,6): 0.041667,0.25,0.090278,0.048611,0.003472 -> median 0.048611
# slow members (2,3,7): 1.0,7.0,0.239471 -> median 1.0
# static members (8,9): 7.0,7.0 -> median 7.0 (nominal, inert -- never triggers)


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def load_btc_train(kind: str = "spot"):
    df, label = load_dataset(ROOT / "data", kind)
    train = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train, label


def load_eth_train():
    eth = load_coinbase_eth_spot(ROOT / "data")
    assert eth is not None, "ETH spot data not committed"
    eth = eth.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth)
    return eth


# ----------------------------------------------------------------------
# Baseline (unmodified hedge_experts) run/metric helpers, shared so both
# branches score the identical reference.
# ----------------------------------------------------------------------

def run_baseline(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    strat = get_strategy("hedge_experts")
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


if __name__ == "__main__":
    # Self-test: causal truncation probe on THIS module's own baseline
    # plumbing (hedge_experts.prepare() itself is covered by
    # tests/test_causality_strict.py at the framework level).
    df, label = load_btc_train("spot")
    m_full, _ = run_baseline(df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)

    df_trunc = df.loc[:INNER_VAL_END]
    m_trunc, _ = run_baseline(df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)

    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"causal truncation probe (r129_shared baseline plumbing): "
          f"{'PASS' if ok else 'FAIL'} ({m_full.final_balance} vs {m_trunc.final_balance})")
    assert ok, "run_baseline reads ahead of its own truncation point"

    print("EXPERT_HORIZON_DAYS:", EXPERT_HORIZON_DAYS.tolist())
    print("EXPERT_BUCKET:", EXPERT_BUCKET)
    print("BUCKET_HORIZON_DAYS:", BUCKET_HORIZON_DAYS)
    m_spot, _ = run_baseline(df, SPOT, None, INNER_TRAIN_END, label)
    m_fut, _ = run_baseline(df, FUTURES, None, INNER_TRAIN_END, label)
    print(f"baseline inner-train spot: trades={m_spot.num_trades} "
          f"final={m_spot.final_balance:.1f} sharpe={m_spot.sharpe:.3f}")
    print(f"baseline inner-train futures: trades={m_fut.num_trades} "
          f"final={m_fut.final_balance:.1f} sharpe={m_fut.sharpe:.3f}")
