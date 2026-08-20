#!/usr/bin/env python
"""B-20, NOVEL branch: does THRESHOLD/BAND-TRIGGERED rebalancing of a
fixed-50/50 BTC+ETH `kelly_regime_v4` portfolio -- rebalance only when the
live BTC weight drifts outside a band around 50%, rather than on a fixed
calendar -- beat both BTC-solo v4 and a fixed-calendar-monthly reference,
survive falsification, and reduce turnover relative to that reference?

This file is NOT B-20's literal test. B-20's own backlog text asks for the
LITERAL periodically-rebalanced (fixed calendar cadence), fixed-50/50
portfolio R-50 originally found -- that is the CONSERVATIVE branch of this
parallel round (`experiments/b20_literal_calendar_5050.py`, a disjoint
session, not read or depended on here). This file asks a different,
complementary question on the same underlying idea: not WHETHER periodic
rebalancing to 50/50 helps, but WHEN the rebalance should fire. It keeps
the identical 50/50 fixed TARGET weight R-50/B-20 use throughout (unlike
R-51's novel branch, which changed the TARGET weights via inverse-vol
rather than the trigger rule).

Backlog item attacked: **B-20**, novel axis -- "does a genuinely different
rebalance-TRIGGER rule (drift-band, not calendar) on the SAME target and
SAME underlying legs change the answer R-51 already found for the
calendar-cadence family?"

Mechanism, one sentence
------------------------
Hold both legs' CONTINUOUS (non-restarting) `kelly_regime_v4` equity
curves, track the implied live BTC weight of pooled capital every bar, and
rebalance the pool back to exactly 50/50 -- paying an explicit round-trip
fee for the reallocation itself -- the instant that live weight drifts
outside a band [50%-b, 50%+b], rather than on any fixed calendar; between
breaches, no rebalancing trade happens at all, however much calendar time
passes.

Constraint attacked (docs/LEDGER.md standing diagnosis)
-----------------------------------------------------------
SIZE (how much of pooled capital to hold in each leg -- the only axis
this project's own one-line summary says has ever worked), N~3 (a second,
imperfectly-synchronized asset, the same axis R-42/R-43/R-50/R-51 opened),
and, distinctively for this branch, **COST** directly: a band trigger
only pays the rebalancing fee when drift (and hence the expected benefit
of correcting it) is large enough to justify the trade, instead of paying
it on every calendar date regardless of whether anything drifted -- fixed-
calendar rebalancing structurally cannot avoid that, a band trigger can.

Why this is a genuine, pre-registered non-duplicate
-------------------------------------------------------
- Not a duplicate of **R-50** (`kelly_regime_covkelly_v3_continuous.py`):
  R-50's `run_continuous_full` fixed-50/50 arm rebalances on a FIXED
  CALENDAR (monthly/weekly), every period, regardless of how far weights
  actually drifted. This file's engine rebalances only when a band is
  breached -- the calendar plays no role in *when* a trade fires here.
- Not a duplicate of **R-51's conservative branch**
  (`experiments/b19_dual_fixed_split.py`, B-19): that branch tested the
  opposite extreme -- NEVER rebalancing at all (a one-time split, b -> inf
  in this file's own framing) -- and found it captures ~100% of R-50's
  drawdown edge but only ~29% of its Sharpe edge, then REJECTED
  decisively on the one holdout read its pre-registration authorized.
  This file's band grid deliberately brackets a region strictly between
  R-51 conservative's b=inf (never) and R-50/R-51's calendar b=0-equivalent
  (rebalance every period regardless of drift) -- a genuinely untested
  middle ground, not a re-run of either extreme.
- Not a duplicate of **R-51's novel branch**
  (`experiments/b19_risk_parity_rebalance.py`, B-19): that branch kept a
  FIXED CALENDAR cadence (monthly/quarterly/semiannual) but changed the
  TARGET weights away from 50/50 (inverse trailing volatility). This file
  does the reverse: the target stays fixed at 50/50 throughout, and what
  changes is WHEN a rebalance to that same target fires. The two axes
  (target-weight information vs. trigger-rule information) are
  orthogonal, and R-51 novel explicitly did not test this one -- its own
  module docstring states its cadence sweep is "monthly / quarterly /
  semiannual," never threshold-triggered.
- Not a duplicate of the **conservative B-20 branch**
  (`experiments/b20_literal_calendar_5050.py`, a disjoint parallel
  session, not read here): that branch tests the LITERAL calendar-cadence
  form of R-50's finding (a single fixed cadence, e.g. monthly, decided
  before running anything). This file's candidate has NO calendar cadence
  at all in its trigger rule -- rebalances fire at irregular, data-driven
  times whenever the band is breached, which could be more or less often
  than any fixed calendar depending on realized drift. The two branches
  are complementary reads of the same underlying B-20 lead, not
  restatements of each other.

Literature this file is grounded in
----------------------------------------
- Donohue, C. & Yip, K. (2003), "Optimal Portfolio Rebalancing with
  Transaction Costs", Journal of Portfolio Management 29(1), 49-63. The
  foundational quantitative treatment of WHEN to trade under a drift band
  rather than a calendar, given transaction costs.
- Masters, S.J. (2003), "Rebalancing", Journal of Portfolio Management
  29(3), 52-57. Practitioner-facing companion piece comparing calendar and
  tolerance-band rebalancing.
- Kitces, M. (2015), "Optimal Rebalancing: Time Horizons vs Tolerance
  Bands" (The Kitces Report, Vol. 2, 2015; kitces.com). Confirmed directly
  via web search while preparing this file (search performed 2026-08-20;
  primary PDF located at kitces.com/wp-content/uploads/2015/07/
  Kitces-Report-Volume-2-2015-An-In-Depth-Look-At-Rebalancing-
  Strategies.pdf): its central finding is that trades should be triggered
  by how far an allocation has drifted from target ("tolerance bands"),
  not by a time horizon, and that rebalancing more often than about
  annually via a pure calendar rule is generally not worth its cost --
  i.e. the general "band dominates calendar on a turnover-adjusted basis"
  claim this file tests empirically on this project's own asset pair,
  strategy and cost tier.
- A 2024 crypto-specific empirical study, cited here as a SECONDARY
  description only -- I was unable to retrieve or verify the primary
  academic source (author, exact journal/venue) from inside this
  sandboxed session; a web search performed while preparing this file
  turned up consistent secondary summaries (crypto portfolio-management
  blogs/aggregators, not the paper itself) describing a simulation of
  ~10,000 cryptocurrency portfolios (BTC, ETH, USDT, LTC, SOL, DOGE, MATIC)
  comparing time-based rebalancing (daily/weekly/monthly) against
  threshold-based rebalancing at 5%, 10% and 15% drift bands, reporting
  threshold rebalancing generally producing better risk-adjusted returns
  than calendar rebalancing at this asset scale. This is why this file's
  own band grid below is exactly {5%, 10%, 15%} -- deliberately matching
  that secondary-sourced grid so this file's own result is at least
  directly comparable to it, while being explicit that the citation
  itself is unverified against a primary source and should be weighted
  accordingly, per this project's own citation-honesty convention (see
  R-39's funding-harvest citations for the same practice).
- This project's own R-33 ("holding less draws down less, that is
  arithmetic, not evidence") and R-51's own decomposition (roughly 71% of
  the calendar-rebalanced form's larger Sharpe edge over a never-
  rebalanced baseline traces to the periodic sell-winners/buy-losers act
  itself, which a bull-dominated 2023-2026 holdout has already shown a
  closely related variant does not reliably monetize) -- the explicit
  STANDING CAUTION this file's own decision rule below is built to
  respect, not discover fresh after the fact.

Design choices, pre-registered before any result exists
-------------------------------------------------------------
**Resolution: checked every bar, triggered discretely.** Drift is
evaluated at the NATIVE 5-minute bar resolution of the underlying
`continuous_leg_equity` curves -- not daily, not on any coarser check
calendar. This is a genuine design choice with a real alternative (check
only once a day, or once a week, and trade only if the band is breached
AT that check) that this file does NOT implement, and the difference
matters: a "checked daily, triggered daily" design can let drift
overshoot the band between checks, while "checked continuously (every
bar), triggered discretely (only on breach)" -- what this file builds --
fires the instant the band is crossed. This is chosen because monitoring
a live weight costs nothing in this project's OHLCV-only, no-execution-
cost-until-trade simulation (unlike a real venue, where polling has some
operational cost this project has no way to price), and because it is the
more literal reading of the band-trigger literature's own framing
("rebalance when a threshold is CROSSED", not "rebalance if a threshold
happens to be crossed on the day you looked"). A daily-check variant is a
reasonable, cheaper-to-monitor alternative for a future round; it is not
tested here.

**Band grid, fixed before any result exists:** b in {5%, 10%, 15%} of the
50/50 target -- i.e. rebalance whenever the live BTC weight of pooled
capital leaves [45%,55%], [40%,60%] or [35%,65%]. Chosen to bracket the
~15% figure in the (unverified-primary-source) crypto simulation above
while also testing tighter bands the classical Donohue & Yip / Kitces
literature associates with higher turnover but tighter tracking of the
target. This is a SMALL, pre-registered grid (3 configurations), matching
this project's established scale for a first pass at a genuinely new
mechanism (12-24 configs is typical for a parameter grid on an existing
mechanism; 3 is appropriate here because the mechanism itself, not a
multi-dimensional hyperparameter space, is the new thing being tested).

**Target weight: fixed at 50/50, never swept.** This is the one thing
this file deliberately holds identical to R-50/B-20's own candidate and
to the conservative B-20 branch, so any difference found is attributable
to the trigger rule, not to a different target.

**Rebalance cost, charged explicitly.** R-50's original engine charged
nothing for the portfolio-level reallocation trade itself (an R-51-
novel-branch finding, not a criticism of R-50, whose brief was the
segment-restart artifact only). This file charges `2 * fee_rate * shift`
per rebalance (one taker fee to sell the overweight leg, one to buy the
underweight leg -- a full round trip on the shifted notional), applied
IDENTICALLY to the band-triggered candidate and to this file's own
fixed-50/50-monthly calendar reference, so the comparison is apples-to-
apples on cost, not just on cadence.

Pre-registered falsification test (written before any sweep code ran)
--------------------------------------------------------------------------
Both must PASS on the inner splits before the holdout is ever touched:
  (F1) Exposure-artifact check. `r_squared` (imported unchanged from
       `kelly_regime_covkelly.py`) of the candidate's equity series
       against a flat rescale of BTC-solo `kelly_regime_v4`'s own
       (continuous) equity series must be <= 0.95 on BOTH inner splits.
       R^2 > 0.95 on either split means "this is relabeled leverage, not
       diversification" (R-33/R-34/R-42/R-43/R-50/R-51's standing rule).
  (F2) Fee-tier survival. Re-run the candidate, BTC-solo v4, and this
       file's own fixed-50/50-monthly reference at the real 0.40%
       Bitstamp taker tier. FAIL if EITHER of the candidate's advantages
       (over BTC-solo v4, and over the fixed-50/50-monthly reference) has
       a POSITIVE Sharpe delta at 0.10% that turns NEGATIVE at 0.40%
       (a sign flip). A delta that is already non-positive at 0.10% has
       nothing to flip and is judged by the promotion rule below, not F2.
If EITHER fires, STOP -- do not read the 2023+ holdout. Report NEGATIVE.

Pre-registered promotion decision rule (written before any result exists)
--------------------------------------------------------------------------
Promote only if ALL of:
  (a) the frozen winning band beats `buy_and_hold` OOS after real costs
      (0.10% baseline AND 0.40% real taker tier);
  (b) the frozen winning band beats BOTH BTC-solo `kelly_regime_v4` AND
      this file's own fixed-50/50-monthly reference by more than the
      +/-0.2 Sharpe noise floor (R-20), OR shows a drawdown/tail
      improvement over BOTH (per the STANDING CAUTION below, weight
      drawdown/tail evidence over Sharpe magnitude -- the holdout has
      been read ~623 times program-wide and no Sharpe-based claim from it
      is supportable any more, per the ledger's own re-ranking since
      R-29);
  (c) survives F1 and F2 again, re-checked on the holdout itself;
  (d) the band-width neighbourhood (5% / 10% / 15%) is a PLATEAU, not a
      knife-edge -- no metric flips sign or moves by more than the noise
      floor between adjacent bands;
  (e) the candidate GENUINELY REDUCES rebalancing-trade count relative to
      the fixed-50/50-monthly reference over the same window -- this is
      the whole point of a band-trigger design; if it does not reduce
      turnover, the idea has failed on its own terms even if Sharpe
      happens to look fine.
Anything else is NEGATIVE. If this rule is changed after seeing any
result, that will be stated explicitly here and the result downgraded to
in-sample, per ROUTINE.md step 4.

**Standing caution carried into this pre-registration, not discovered
after running anything (per this round's own brief):** R-51's own
decomposition found that most of this whole family's inner-validation
Sharpe edge over a never-rebalanced baseline traces to the same return-
side mechanism (periodic sell-winners/buy-losers) that a bull-dominated
2023-2026 holdout has already shown a closely related variant (R-51
conservative's literal calendar form) does NOT reliably monetize. The
evidence updates AGAINST any member of this family clearing the holdout,
including this one, before this file runs anything. A band trigger
changes WHEN that mechanism fires, not WHAT it is -- so this file's own
prior, stated honestly, is that it is more likely than not to fail the
same way, and gate (b)'s emphasis on drawdown/tail over Sharpe reflects
that prior directly, not just the generic holdout-exhaustion caution.

Data-window rule
-------------------
inner-train = 2019-03-14 (ETH's real start) -> 2020-12-31.
inner-validation = 2021-01-01 -> 2022-12-31 (the joint 2022 BTC/ETH bear).
holdout = 2023-01-01 onward, read AT MOST ONCE -- one paired call across
both fee tiers, mirroring `experiments/b19_dual_fixed_split.py::holdout`'s
convention exactly -- ONLY if every gate above passes.

Hard rules honored
--------------------
- Only this NEW file is touched. `kelly_regime_covkelly_v3_continuous.py`
  (R-50), `kelly_regime_covkelly.py` (R-42/R-43), `kelly_regime_v4.py`,
  `multiasset.py`, `kelly_regime_dual_fixed.py`, and everything under
  `src/tradebot/` are imported from, unmodified. The conservative B-20
  branch's file is never imported or read.
- `continuous_leg_equity` is imported and called UNCHANGED for both legs'
  underlying continuous curves -- the one piece of prior machinery this
  round's brief says must not be reimplemented. Everything about WHEN to
  reallocate pooled capital between the two legs (`run_band_triggered`
  below) is new code written for this file -- no existing function in
  this repo implements threshold/band-triggered rebalancing.
- A known cache-key bug in `continuous_leg_equity` (found and documented
  by R-51's novel branch: its module-level memoization key omits
  `market.fee_rate`, so calling it on the SAME dataframe object at two
  fee tiers in one process silently returns the wrong cached curve) is
  worked around here WITHOUT reimplementing or editing that function --
  see `leg_equity` below for the (different from R-51's) minimal fix.
- No lookahead: `causality_check` runs the standard multiply/divide
  truncation-tamper probe on this file's new band-trigger code path.
- `N_EVALUATED` counts every distinct band-triggered candidate
  configuration backtested (band x market); the fixed-50/50-monthly
  reference, BTC-solo v4 and `buy_and_hold` are NOT counted, matching
  this project's established convention (R-42/R-43/R-50/R-51 all exclude
  baselines/references the same way). `N_BACKTESTS_TOTAL` counts every
  distinct backtest of any kind, reported alongside `N_EVALUATED`.

Usage::

    python experiments/b20_threshold_band_5050.py causality  # mandatory no-lookahead check, run first
    python experiments/b20_threshold_band_5050.py sweep      # step 3: band grid, inner splits, spot + futures
    python experiments/b20_threshold_band_5050.py artifact   # F1: exposure-artifact check
    python experiments/b20_threshold_band_5050.py feetier    # F2: 0.40% taker fee-tier check
    python experiments/b20_threshold_band_5050.py turnover   # gate (e): trade-count comparison vs calendar ref
    python experiments/b20_threshold_band_5050.py gate       # everything above + prints PROCEED/STOP for the holdout
    python experiments/b20_threshold_band_5050.py holdout    # PRE-REGISTERED, run ONLY if gate() says PROCEED
    python experiments/b20_threshold_band_5050.py all        # causality+sweep+artifact+feetier+turnover+gate (no holdout)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402
from tradebot.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.kelly_regime_covkelly import (  # noqa: E402
    SPOT,
    FUTURES5X,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    load_assets,
    r_squared,
    _segment_bounds,
)
from experiments.kelly_regime_covkelly_v3_continuous import (  # noqa: E402
    continuous_leg_equity,
    _segment_returns,
    period_metrics,
    FULL_START, FULL_END,
)

# the ONE place OOS_START is spelled out as a literal in this file, per the
# gating convention set by kelly_regime_dual_bootstrap.py (R-43) and reused
# by every B-19/B-20 branch since
OOS_START = "2023-01-01"

BAND_GRID = (0.05, 0.10, 0.15)          # fraction of the 50/50 target
CALENDAR_FREQ = "MS"                     # fixed-50/50-monthly reference cadence
TARGET_BTC = 0.5                         # never swept -- see docstring

N_EVALUATED = 0        # band-triggered candidate configurations only (band x market)
N_BACKTESTS_TOTAL = 0  # every distinct backtest of any kind


def _count(n: int = 1) -> None:
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += n


# ===================================================== fee-tier-safe leg cache
# See module docstring's "Hard rules honored" section: continuous_leg_equity
# (imported UNCHANGED from R-50's file) memoizes on a key that omits
# market.fee_rate, so calling it on the SAME dataframe object at two fee
# tiers within one process silently returns the wrong cached curve (found
# and documented by R-51's novel branch, which fixed it with its own
# separately-keyed local cache built on a reimplemented call). This file
# takes a smaller fix that touches nothing about continuous_leg_equity's own
# logic: the bug's key ingredient is `id(df)` staying the same across two
# fee tiers. The FIRST time a given (id(df), market.name) pair is sent to
# continuous_leg_equity, pass the original frame (fast, correctly cached
# there). Any SUBSEQENT call for that same (id(df), market.name) pair, if
# it is at a genuinely different fee/leverage/params combination, passes a
# `.copy()` instead -- a fresh id(df) that cannot collide with the earlier
# cache entry inside continuous_leg_equity. A local cache here (keyed
# correctly, including fee_rate) avoids repeating that copy+recompute for
# repeat calls at the SAME combination.
_LEG_CACHE_LOCAL: dict = {}
_SEEN_DF_MARKET: set = set()


def leg_equity(df: pd.DataFrame, market: MarketSpec, v4_kwargs: dict | None = None,
               start: str = FULL_START, end: str = FULL_END,
               start_balance: float = 1000.0) -> pd.Series:
    key = (id(df), market.name, round(market.fee_rate, 6), round(market.leverage, 4),
           tuple(sorted((v4_kwargs or {}).items())), start, end, start_balance)
    if key in _LEG_CACHE_LOCAL:
        return _LEG_CACHE_LOCAL[key]
    dfmkey = (id(df), market.name)
    if dfmkey in _SEEN_DF_MARKET:
        df_call = df.copy()
    else:
        df_call = df
        _SEEN_DF_MARKET.add(dfmkey)
    eq = continuous_leg_equity(df_call, market, v4_kwargs, start=start, end=end,
                                start_balance=start_balance)
    _LEG_CACHE_LOCAL[key] = eq
    return eq


def verify_fee_cache_safe() -> bool:
    """Sanity check for the workaround above: run the SAME BTC frame at
    0.10% then 0.40% and confirm the two curves genuinely differ (proof the
    second call was not served the first call's cached, wrong-fee result)."""
    btc_df, _ = load_assets()
    eq_10 = leg_equity(btc_df, SPOT, start=TRAIN_START, end=TRAIN_END)
    eq_40 = leg_equity(btc_df, MarketSpec.spot(fee_rate=0.004), start=TRAIN_START, end=TRAIN_END)
    diff = float(np.nanmax(np.abs(eq_10.reindex(eq_10.index.union(eq_40.index)).ffill().to_numpy()
                                   - eq_40.reindex(eq_10.index.union(eq_40.index)).ffill().to_numpy())))
    ok = diff > 1.0  # two different fee tiers over ~2 years must diverge by more than a rounding error
    print(f"fee-cache-safety check: max|0.10%-curve - 0.40%-curve| = {diff:.2f}  "
          f"({'OK, curves differ -- no stale-cache collision' if ok else 'SUSPICIOUS, curves nearly identical'})")
    return ok


# ============================================================ band-trigger engine

def run_band_triggered(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame, band: float,
    market: MarketSpec = SPOT, start_balance: float = 1000.0,
    target_btc: float = TARGET_BTC, rebalance_fee_rate: float | None = None,
    v4_kwargs: dict | None = None,
    full_start: str = FULL_START, full_end: str = FULL_END,
) -> dict:
    """The new mechanism this file exists to build: hold both legs'
    CONTINUOUS equity curves (from `continuous_leg_equity`, unchanged),
    walk forward bar by bar tracking the live BTC weight of pooled
    capital, and rebalance back to `target_btc` -- paying an explicit
    round-trip fee -- the instant that weight leaves
    [target_btc - band, target_btc + band]. Between breaches, no trade
    happens no matter how much calendar time passes -- the defining
    difference from every calendar-cadence engine in this project.

    Single forward pass, O(n) in the number of bars: each bar's dollar
    value is computed from the CURRENT segment's scale factors (frozen at
    the last rebalance, or at t=0), and a rebalance both closes out the
    old segment and opens a new one at the SAME bar -- i.e. the bar where
    a breach is detected shows the POST-rebalance value, mirroring the
    convention every calendar-cadence engine in this project's history
    (R-42/R-50/R-51) already uses at its own segment boundaries.
    """
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += 1
    if rebalance_fee_rate is None:
        rebalance_fee_rate = market.fee_rate

    btc_full = leg_equity(btc_df, market, v4_kwargs, start=full_start, end=full_end,
                           start_balance=start_balance)
    eth_full = leg_equity(eth_df, market, v4_kwargs, start=full_start, end=full_end,
                           start_balance=start_balance)

    idx = btc_full.index.union(eth_full.index)
    b_arr = btc_full.reindex(idx).ffill().bfill().to_numpy(dtype=float)
    e_arr = eth_full.reindex(idx).ffill().bfill().to_numpy(dtype=float)
    n = len(idx)

    if n == 0:
        return {"equity": pd.Series(dtype=float), "rebalance_log": pd.DataFrame(),
                "fees_rebalance": 0.0, "n_rebalances": 0,
                "final_balance": start_balance, "btc_full": btc_full, "eth_full": eth_full,
                "idx_min": None, "idx_max": None}

    lo, hi = target_btc - band, target_btc + band

    dollars_b = start_balance * target_btc
    dollars_e = start_balance * (1.0 - target_btc)
    seg_base_b = b_arr[0] if b_arr[0] > 0 else 1.0
    seg_base_e = e_arr[0] if e_arr[0] > 0 else 1.0
    scale_b = dollars_b / seg_base_b
    scale_e = dollars_e / seg_base_e

    out = np.empty(n, dtype=float)
    log_rows: list[dict] = []
    fees_rebalance_total = 0.0
    n_rebalances = 0

    for i in range(n):
        bb = b_arr[i]
        ee = e_arr[i]
        cur_b = bb * scale_b
        cur_e = ee * scale_e
        total = cur_b + cur_e
        out[i] = total

        if i == 0 or total <= 0:
            continue

        w_live = cur_b / total
        if w_live < lo or w_live > hi:
            shift = abs(total * target_btc - cur_b)
            fee_this = 2.0 * rebalance_fee_rate * shift
            fees_rebalance_total += fee_this
            n_rebalances += 1
            pooled_after = max(0.0, total - fee_this)
            dollars_b = pooled_after * target_btc
            dollars_e = pooled_after * (1.0 - target_btc)
            seg_base_b = bb if bb > 0 else 1.0
            seg_base_e = ee if ee > 0 else 1.0
            scale_b = dollars_b / seg_base_b
            scale_e = dollars_e / seg_base_e
            out[i] = dollars_b + dollars_e
            log_rows.append({"date": idx[i], "live_weight_at_trigger": w_live,
                              "pooled_pre": total, "fee": fee_this,
                              "pooled_after": pooled_after})

    equity = pd.Series(out, index=idx)
    return {"equity": equity, "rebalance_log": pd.DataFrame(log_rows),
            "fees_rebalance": fees_rebalance_total, "n_rebalances": n_rebalances,
            "final_balance": float(equity.iloc[-1]),
            "btc_full": btc_full, "eth_full": eth_full,
            "idx_min": idx[0], "idx_max": idx[-1]}


# ================================================ fixed-50/50 calendar reference

def run_calendar_fixed5050_costed(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame, rebalance_freq: str = CALENDAR_FREQ,
    market: MarketSpec = SPOT, start_balance: float = 1000.0,
    rebalance_fee_rate: float | None = None, v4_kwargs: dict | None = None,
    full_start: str = FULL_START, full_end: str = FULL_END,
) -> dict:
    """This file's OWN simple re-derivation of a costed, periodically-
    rebalanced fixed-50/50 reference -- NOT imported from the conservative
    B-20 branch (per the task brief) or from R-51's novel branch (kept
    disjoint, per this round's own hard rule that only this new file is
    touched). Same continuous-leg-plus-segment-slicing mechanism every
    prior calendar-cadence engine in this project's history uses
    (`_segment_bounds`/`_segment_returns`, both imported UNCHANGED), with
    the SAME explicit round-trip rebalance cost `run_band_triggered` above
    charges, so the two engines are cost-comparable by construction.
    """
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += 1
    if rebalance_fee_rate is None:
        rebalance_fee_rate = market.fee_rate

    btc_full = leg_equity(btc_df, market, v4_kwargs, start=full_start, end=full_end,
                           start_balance=start_balance)
    eth_full = leg_equity(eth_df, market, v4_kwargs, start=full_start, end=full_end,
                           start_balance=start_balance)

    bounds = _segment_bounds(full_start, full_end, rebalance_freq)
    btc_segs = _segment_returns(btc_full, bounds)
    eth_segs = _segment_returns(eth_full, bounds)
    n = min(len(btc_segs), len(eth_segs))

    pooled = start_balance
    pieces: list[pd.Series] = []
    fees_rebalance_total = 0.0
    n_rebalances = 0
    prev_b = prev_e = None

    for i in range(n):
        sb, se = btc_segs[i], eth_segs[i]
        seg_start, seg_end = sb["seg_start"], sb["seg_end"]
        if i == 0 or prev_b is None:
            pooled_pre = pooled
            fee_this = 0.0
        else:
            pooled_pre = prev_b + prev_e
            target_b_dollars = pooled_pre * TARGET_BTC
            shift = abs(target_b_dollars - prev_b)
            fee_this = 2.0 * rebalance_fee_rate * shift
            fees_rebalance_total += fee_this
            n_rebalances += 1

        pooled_after = max(0.0, pooled_pre - fee_this)
        dollars_b = pooled_after * TARGET_BTC
        dollars_e = pooled_after * (1.0 - TARGET_BTC)

        btc_sub = btc_full.loc[seg_start:seg_end]
        eth_sub = eth_full.loc[seg_start:seg_end]
        scale_b = (dollars_b / sb["base_val"]) if sb["base_val"] > 0 else 0.0
        scale_e = (dollars_e / se["base_val"]) if se["base_val"] > 0 else 0.0
        btc_leg = btc_sub * scale_b
        eth_leg = eth_sub * scale_e

        idx = btc_leg.index.union(eth_leg.index)
        combined = (btc_leg.reindex(idx).ffill().bfill().fillna(0.0)
                    + eth_leg.reindex(idx).ffill().bfill().fillna(0.0))
        if len(combined) == 0:
            continue
        pieces.append(combined)
        prev_b = float(btc_leg.iloc[-1]) if len(btc_leg) else dollars_b
        prev_e = float(eth_leg.iloc[-1]) if len(eth_leg) else dollars_e
        pooled = float(combined.iloc[-1])

    equity = pd.concat(pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    return {"equity": equity, "fees_rebalance": fees_rebalance_total,
            "n_rebalances": n_rebalances,
            "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance}


# =================================================================== baselines

def _bh_metrics(df: pd.DataFrame, start: str, end: str, market: MarketSpec,
                 start_balance: float = 1000.0) -> dict:
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += 1
    res = run_period(BuyAndHold(), df, start=start, end=end, market=market,
                      start_balance=start_balance)
    eq = res.equity
    return {"final_balance": float(eq.iloc[-1]), "sharpe": sharpe_ratio(eq.to_numpy()),
            "max_dd_pct": max_drawdown_pct(eq.to_numpy())}


def _solo_metrics(btc_df: pd.DataFrame, market: MarketSpec = SPOT,
                   full_start: str = FULL_START, full_end: str = FULL_END) -> dict:
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += 1
    eq = leg_equity(btc_df, market, None, start=full_start, end=full_end)
    return {"train": period_metrics(eq, TRAIN_START, TRAIN_END),
            "valid": period_metrics(eq, VALID_START, VALID_END), "_equity": eq}


# =================================================================== sweep

def eval_band(btc_df, eth_df, band: float, market: MarketSpec = SPOT) -> dict:
    global N_EVALUATED
    N_EVALUATED += 1
    res = run_band_triggered(btc_df, eth_df, band, market=market)
    eq = res["equity"]
    return {"train": period_metrics(eq, TRAIN_START, TRAIN_END),
            "valid": period_metrics(eq, VALID_START, VALID_END),
            "fees_rebalance": res["fees_rebalance"], "n_rebalances": res["n_rebalances"]}


def run_sweep(data_dir: str = "data") -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Band grid x {spot, futures5x} -- inner-train is spot-only per the
    brief (read off the SAME continuous run that gives inner-validation),
    inner-validation is both markets."""
    btc_df, eth_df = load_assets(data_dir)
    rows = []
    for band in BAND_GRID:
        for mname, market in (("spot", SPOT), ("futures5x", FUTURES5X)):
            r = eval_band(btc_df, eth_df, band, market=market)
            rows.append({"band": band, "market": mname, "train": r["train"], "valid": r["valid"],
                         "fees_rebalance": r["fees_rebalance"], "n_rebalances": r["n_rebalances"]})
            print(f"band=+/-{band*100:>4.0f}% market={mname:<10} | "
                  f"train Sharpe={r['train']['sharpe']:>6.2f} DD={r['train']['max_dd_pct']:>5.1f}% | "
                  f"valid Sharpe={r['valid']['sharpe']:>6.2f} DD={r['valid']['max_dd_pct']:>5.1f}% | "
                  f"n_rebalances={r['n_rebalances']:>3} rebal_fees=${r['fees_rebalance']:.2f}")
    print(f"\nconfigs evaluated this call: {len(rows)} (N_EVALUATED so far: {N_EVALUATED})")
    return rows, btc_df, eth_df


def select_best(rows: list[dict]) -> dict:
    """Selection rule, fixed before the sweep ran: among SPOT rows, rank by
    min(train_sharpe, valid_sharpe) -- guards against the train-loses/
    validation-wins overfit signature that sank R-37/R-38/R-40 -- tie-break
    on -valid_max_dd_pct. Spot is the primary selection market, matching
    every prior branch on this backlog item; futures is reported as a
    secondary robustness check on the selected band, not used to select."""
    spot_rows = [r for r in rows if r["market"] == "spot"]

    def score(r):
        return (min(r["train"]["sharpe"], r["valid"]["sharpe"]), -r["valid"]["max_dd_pct"])
    return max(spot_rows, key=score)


def neighbourhood_report(rows: list[dict], best: dict) -> None:
    """P4 / gate (d): is the winning band a plateau or a knife-edge?"""
    print("\n=== band-width neighbourhood (spot, inner-validation Sharpe, sorted) ===")
    spot_rows = sorted([r for r in rows if r["market"] == "spot"], key=lambda r: -r["valid"]["sharpe"])
    for r in spot_rows:
        mark = "  <== WINNER" if r["band"] == best["band"] else ""
        print(f"band=+/-{r['band']*100:>4.0f}%  valid Sharpe={r['valid']['sharpe']:>6.2f}  "
              f"train Sharpe={r['train']['sharpe']:>6.2f}  n_rebalances={r['n_rebalances']:>3}{mark}")


# ================================================================ headline

def run_headline(data_dir: str = "data") -> dict:
    btc_df, eth_df = load_assets(data_dir)
    rows, _, _ = run_sweep(data_dir)
    best = select_best(rows)
    print(f"\nselected best band: +/-{best['band']*100:.0f}%")
    neighbourhood_report(rows, best)
    band = best["band"]

    cand_res = run_band_triggered(btc_df, eth_df, band, market=SPOT)
    cal_res = run_calendar_fixed5050_costed(btc_df, eth_df, market=SPOT)
    solo = _solo_metrics(btc_df, SPOT)
    bh = {"train": _bh_metrics(btc_df, TRAIN_START, TRAIN_END, SPOT),
          "valid": _bh_metrics(btc_df, VALID_START, VALID_END, SPOT)}

    cand = {"train": period_metrics(cand_res["equity"], TRAIN_START, TRAIN_END),
            "valid": period_metrics(cand_res["equity"], VALID_START, VALID_END)}
    cal = {"train": period_metrics(cal_res["equity"], TRAIN_START, TRAIN_END),
           "valid": period_metrics(cal_res["equity"], VALID_START, VALID_END)}

    print(f"\n=== HEADLINE (spot, band=+/-{band*100:.0f}%, monthly calendar reference) ===")
    header = f"{'candidate':<34} {'period':<6} {'final':>10} {'sharpe':>8} {'maxDD%':>8}"
    print(header)
    for name, table in (("band-triggered (candidate)", cand),
                        ("fixed 50/50 monthly (re-derived ref)", cal),
                        ("v4 BTC-solo (reference)", solo),
                        ("buy_and_hold BTC", bh)):
        for label in ("train", "valid"):
            m = table[label]
            print(f"{name:<34} {label:<6} {m['final_balance']:>10.1f} "
                  f"{m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")

    d_sharpe_solo = cand["valid"]["sharpe"] - solo["valid"]["sharpe"]
    d_sharpe_cal = cand["valid"]["sharpe"] - cal["valid"]["sharpe"]
    d_dd_solo = cand["valid"]["max_dd_pct"] - solo["valid"]["max_dd_pct"]
    d_dd_cal = cand["valid"]["max_dd_pct"] - cal["valid"]["max_dd_pct"]
    print(f"\nvalid dSharpe vs v4-solo:          {d_sharpe_solo:+.2f}")
    print(f"valid dSharpe vs fixed-50/50-cal:  {d_sharpe_cal:+.2f}")
    print(f"valid dmaxDD  vs v4-solo:           {d_dd_solo:+.1f}pp")
    print(f"valid dmaxDD  vs fixed-50/50-cal:   {d_dd_cal:+.1f}pp")
    print(f"\ncandidate n_rebalances (train+valid span): {cand_res['n_rebalances']}")
    print(f"calendar reference n_rebalances (same span): {cal_res['n_rebalances']}")
    print(f"total N_EVALUATED (candidate configs): {N_EVALUATED}")
    print(f"total N_BACKTESTS_TOTAL: {N_BACKTESTS_TOTAL}")
    return {"best": best, "rows": rows, "cand": cand, "cal": cal, "solo": solo, "bh": bh,
            "band": band, "btc_df": btc_df, "eth_df": eth_df,
            "cand_res": cand_res, "cal_res": cal_res}


# =============================================================== diagnostics

def causality_check(data_dir: str = "data", band: float = 0.10) -> bool:
    """Truncation tamper probe on THIS file's new band-trigger code path
    (`run_band_triggered`). Same multiply/divide convention R-42/R-50/R-51
    used, extended to this file's mechanism."""
    btc_df, eth_df = load_assets(data_dir)
    cut = pd.Timestamp("2021-06-30", tz="UTC")
    K = 137.0

    def tamper(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > cut
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        return out

    base = run_band_triggered(btc_df, eth_df, band)
    up = run_band_triggered(tamper(btc_df, K), tamper(eth_df, K), band)
    down = run_band_triggered(tamper(btc_df, 1.0 / K), tamper(eth_df, 1.0 / K), band)

    pre = base["equity"].index <= cut
    b = base["equity"][pre].to_numpy()
    u = up["equity"].reindex(base["equity"].index)[pre].to_numpy()
    d = down["equity"].reindex(base["equity"].index)[pre].to_numpy()
    max_diff_up = float(np.nanmax(np.abs(b - u)))
    max_diff_down = float(np.nanmax(np.abs(b - d)))
    ok = max_diff_up < 1e-6 and max_diff_down < 1e-6
    print(f"causality check (band-trigger engine): cut={cut.date()}, K={K}, band=+/-{band*100:.0f}%")
    print(f"  max |base - up-tampered| pooled equity before cut: {max_diff_up:.3e}")
    print(f"  max |base - down-tampered| pooled equity before cut: {max_diff_down:.3e}")
    print(f"  PASS (pooled equity before cut unchanged): {ok}")
    return ok


def artifact_check(btc_df, eth_df, band: float, market: MarketSpec = SPOT) -> dict:
    """F1: R^2 exposure-artifact diagnostic -- candidate vs flat-rescaled
    v4-BTC-solo (continuous), on both inner splits."""
    print("\n=== F1: exposure-artifact diagnostic ===")
    cand_res = run_band_triggered(btc_df, eth_df, band, market=market)
    solo_eq = leg_equity(btc_df, market, None, start=FULL_START, end=FULL_END)

    out = {}
    fail = False
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
        cand_sub = cand_res["equity"].loc[s:e]
        solo_sub = solo_eq.loc[s:e]
        r2 = r_squared(cand_sub, solo_sub)
        flag = "ARTIFACT (R^2>0.95)" if r2 > 0.95 else "ok"
        if r2 > 0.95:
            fail = True
        print(f"[{label}] candidate vs flat-rescaled v4-BTC-solo: R^2={r2:.4f} -> {flag}")
        out[label] = {"r2_solo": r2}
    out["F1_pass"] = not fail
    print(f"\nF1 (exposure-artifact falsification test): {'PASS' if out['F1_pass'] else 'FAIL'}")
    return out


def feetier_check(btc_df, eth_df, band: float, market_kind: str = "spot") -> dict:
    """F2: 0.40% Bitstamp taker tier -- candidate vs BTC-solo v4 AND vs the
    fixed-50/50-monthly reference must not have a positive 0.10%-tier
    Sharpe delta flip negative at 0.40%."""
    BITSTAMP_TAKER = 0.004
    market_01 = SPOT if market_kind == "spot" else FUTURES5X
    market_04 = MarketSpec.spot(fee_rate=BITSTAMP_TAKER) if market_kind == "spot" \
        else MarketSpec.futures(leverage=5.0, fee_rate=BITSTAMP_TAKER)

    print(f"\n=== F2: 0.40% taker fee-tier stress test ({market_kind}) ===")
    rows = {}
    for tag, market in (("0.10%", market_01), ("0.40%", market_04)):
        cand_res = run_band_triggered(btc_df, eth_df, band, market=market)
        cal_res = run_calendar_fixed5050_costed(btc_df, eth_df, market=market)
        solo_eq = leg_equity(btc_df, market, None, start=FULL_START, end=FULL_END)
        bh = _bh_metrics(btc_df, VALID_START, VALID_END, market)

        cand_v = period_metrics(cand_res["equity"], VALID_START, VALID_END)
        cal_v = period_metrics(cal_res["equity"], VALID_START, VALID_END)
        solo_v = period_metrics(solo_eq, VALID_START, VALID_END)

        rows[tag] = {"cand": cand_v, "cal": cal_v, "solo": solo_v, "bh": bh,
                     "d_sharpe_solo": cand_v["sharpe"] - solo_v["sharpe"],
                     "d_sharpe_cal": cand_v["sharpe"] - cal_v["sharpe"]}
        print(f"  @ {tag}: candidate Sharpe={cand_v['sharpe']:.2f}  cal-ref Sharpe={cal_v['sharpe']:.2f}  "
              f"solo Sharpe={solo_v['sharpe']:.2f}  bh Sharpe={bh['sharpe']:.2f}  "
              f"dSharpe(vs solo)={rows[tag]['d_sharpe_solo']:+.2f}  "
              f"dSharpe(vs cal)={rows[tag]['d_sharpe_cal']:+.2f}")

    flip_solo = rows["0.10%"]["d_sharpe_solo"] > 0 and rows["0.40%"]["d_sharpe_solo"] < 0
    flip_cal = rows["0.10%"]["d_sharpe_cal"] > 0 and rows["0.40%"]["d_sharpe_cal"] < 0
    f2_pass = not (flip_solo or flip_cal)
    print(f"\nsign flip vs v4-solo (0.10%->0.40%): {flip_solo}")
    print(f"sign flip vs fixed-50/50-cal (0.10%->0.40%): {flip_cal}")
    print(f"F2 (fee-tier falsification test): {'PASS' if f2_pass else 'FAIL'}")
    return {"rows": rows, "F2_pass": f2_pass}


def turnover_check(btc_df, eth_df, band: float, market: MarketSpec = SPOT) -> dict:
    """Gate (e): does the band-triggered candidate genuinely reduce
    rebalancing-trade count relative to the fixed-50/50-monthly reference,
    over the identical [FULL_START, FULL_END] span?"""
    print("\n=== gate (e): turnover / rebalance-trade-count comparison ===")
    cand_res = run_band_triggered(btc_df, eth_df, band, market=market)
    cal_res = run_calendar_fixed5050_costed(btc_df, eth_df, market=market)
    n_cand = cand_res["n_rebalances"]
    n_cal = cal_res["n_rebalances"]
    reduced = n_cand < n_cal
    print(f"band=+/-{band*100:.0f}% candidate rebalances: {n_cand}  "
          f"(fees ${cand_res['fees_rebalance']:.2f})")
    print(f"fixed-50/50-monthly reference rebalances: {n_cal}  "
          f"(fees ${cal_res['fees_rebalance']:.2f})")
    print(f"gate (e) [genuinely reduces turnover]: {'PASS' if reduced else 'FAIL'}")
    return {"n_candidate": n_cand, "n_calendar": n_cal,
            "fees_candidate": cand_res["fees_rebalance"], "fees_calendar": cal_res["fees_rebalance"],
            "reduces_turnover": reduced}


# ===================================================================== gate

def gate(data_dir: str = "data") -> dict:
    """Runs F1, F2, the plateau check (d) and the turnover check (e) on the
    selected band, and prints the single PROCEED/STOP verdict that decides
    whether `holdout()` may be called. Does NOT call holdout() itself."""
    out = run_headline(data_dir)
    btc_df, eth_df, band = out["btc_df"], out["eth_df"], out["band"]

    f1 = artifact_check(btc_df, eth_df, band)
    f2 = feetier_check(btc_df, eth_df, band)
    to = turnover_check(btc_df, eth_df, band)

    rows = out["rows"]
    spot_rows = sorted([r for r in rows if r["market"] == "spot"], key=lambda r: r["band"])
    sharpes = [r["valid"]["sharpe"] for r in spot_rows]
    plateau = (max(sharpes) - min(sharpes)) <= 0.2  # noise floor, per ROUTINE.md
    print(f"\n=== gate (d): band-width plateau check ===")
    print(f"spot inner-validation Sharpe range across the {len(sharpes)}-band grid: "
          f"{min(sharpes):.2f} to {max(sharpes):.2f} (spread {max(sharpes)-min(sharpes):.2f})")
    print(f"gate (d) [plateau, not knife-edge]: {'PASS' if plateau else 'FAIL'}")

    proceed = f1["F1_pass"] and f2["F2_pass"] and plateau and to["reduces_turnover"]
    print(f"\n=== PRE-HOLDOUT GATE: {'PROCEED to holdout' if proceed else 'STOP -- do not read the holdout'} ===")
    print(f"  F1 (exposure-artifact): {'PASS' if f1['F1_pass'] else 'FAIL'}")
    print(f"  F2 (fee-tier survival): {'PASS' if f2['F2_pass'] else 'FAIL'}")
    print(f"  (d) plateau: {'PASS' if plateau else 'FAIL'}")
    print(f"  (e) turnover reduction: {'PASS' if to['reduces_turnover'] else 'FAIL'}")
    return {"proceed": proceed, "band": band, "f1": f1, "f2": f2, "plateau": plateau,
            "turnover": to, "headline": out}


# ===================================================================== holdout

def holdout(band: float) -> dict:
    """Step 4: the ONE pre-registered holdout read for this claim. Run
    ONLY if `gate()` returns proceed=True. Uses the full, uncut BTC/ETH
    frames -- NOT the hard-sliced `load_assets` -- sliced to >= OOS_START
    HERE, and only here, mirroring `b19_dual_fixed_split.py::holdout`'s
    convention exactly.

    CRITICAL TRAP (per this round's brief): `continuous_leg_equity`
    defaults to `end=FULL_END="2022-12-31"`. Every call in this function
    passes `full_end=holdout_end` EXPLICITLY, computed as the true last
    common date in the uncut data. The min/max dates actually covered are
    printed below and asserted to exceed 2023-01-01 before any number is
    reported, exactly as this round's brief requires.
    """
    from tradebot.data import load_coinbase_eth_spot, load_dataset

    BTC, _ = load_dataset(ROOT / "data", "spot")
    ETH = load_coinbase_eth_spot(ROOT / "data")
    if ETH is None:
        raise RuntimeError("data/ethusd_coinbase_spot_5m.csv.gz not found")

    print(f"=== PRE-REGISTERED HOLDOUT READ ({OOS_START} onward) === band=+/-{band*100:.0f}%")
    holdout_end = str(min(BTC.index[-1], ETH.index[-1]).date())
    print(f"full BTC file: {BTC.index[0]} -> {BTC.index[-1]}")
    print(f"full ETH file: {ETH.index[0]} -> {ETH.index[-1]}")
    print(f"holdout_end computed as min of the two: {holdout_end}")

    btc_h = BTC.loc[OOS_START:]
    eth_h = ETH.loc[OOS_START:]

    out = {}
    for tag, market in (("0.10% baseline", SPOT), ("0.40% real taker", MarketSpec.spot(fee_rate=0.004))):
        cand_res = run_band_triggered(btc_h, eth_h, band, market=market,
                                       full_start=OOS_START, full_end=holdout_end)
        cal_res = run_calendar_fixed5050_costed(btc_h, eth_h, market=market,
                                                  full_start=OOS_START, full_end=holdout_end)
        solo_eq = leg_equity(btc_h, market, None, start=OOS_START, end=holdout_end)
        bh = _bh_metrics(btc_h, OOS_START, holdout_end, market)

        # CRITICAL TRAP sanity check, per this round's brief -- printed and
        # asserted BEFORE any headline number from this call is reported
        cand_idx = cand_res["equity"].index
        idx_min, idx_max = cand_idx.min(), cand_idx.max()
        print(f"\n--- {tag}: candidate equity actually covers "
              f"{idx_min} -> {idx_max} ---")
        assert idx_max > pd.Timestamp("2023-01-01", tz="UTC"), (
            "CRITICAL TRAP HIT: candidate equity does not extend past 2023-01-01 -- "
            "the FULL_END default trap fired. Aborting before reporting any number."
        )

        cand = period_metrics(cand_res["equity"], OOS_START, holdout_end)
        cal = period_metrics(cal_res["equity"], OOS_START, holdout_end)
        solo = period_metrics(solo_eq, OOS_START, holdout_end)

        print(f"candidate:        final=${cand['final_balance']:.0f} Sharpe={cand['sharpe']:.2f} "
              f"DD={cand['max_dd_pct']:.1f}%  n_rebalances={cand_res['n_rebalances']}")
        print(f"fixed-50/50-cal:  final=${cal['final_balance']:.0f} Sharpe={cal['sharpe']:.2f} "
              f"DD={cal['max_dd_pct']:.1f}%  n_rebalances={cal_res['n_rebalances']}")
        print(f"v4 BTC-solo:      final=${solo['final_balance']:.0f} Sharpe={solo['sharpe']:.2f} "
              f"DD={solo['max_dd_pct']:.1f}%")
        print(f"buy_and_hold:     final=${bh['final_balance']:.0f} Sharpe={bh['sharpe']:.2f} "
              f"DD={bh['max_dd_pct']:.1f}%")
        out[tag] = {"candidate": cand, "fixed5050_calendar": cal, "v4_solo": solo,
                    "buy_and_hold": bh, "n_rebalances_candidate": cand_res["n_rebalances"],
                    "n_rebalances_calendar": cal_res["n_rebalances"],
                    "idx_min": idx_min, "idx_max": idx_max}

    print(f"\nholdout reads this call: 1 paired call, 2 fee tiers (matches the pre-registered convention)")
    return out


# ===================================================================== CLI

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "causality":
        causality_check()
    elif cmd == "cache_check":
        verify_fee_cache_safe()
    elif cmd == "sweep":
        run_sweep()
    elif cmd == "select":
        run_headline()
    elif cmd == "artifact":
        btc_df, eth_df = load_assets()
        rows, _, _ = run_sweep()
        best = select_best(rows)
        artifact_check(btc_df, eth_df, best["band"])
    elif cmd == "feetier":
        btc_df, eth_df = load_assets()
        rows, _, _ = run_sweep()
        best = select_best(rows)
        feetier_check(btc_df, eth_df, best["band"])
    elif cmd == "turnover":
        btc_df, eth_df = load_assets()
        rows, _, _ = run_sweep()
        best = select_best(rows)
        turnover_check(btc_df, eth_df, best["band"])
    elif cmd == "gate":
        gate()
    elif cmd == "holdout":
        rows, _, _ = run_sweep()
        best = select_best(rows)
        holdout(best["band"])
    elif cmd == "all":
        ok = causality_check()
        print(f"\ncausality PASS: {ok}\n")
        verify_fee_cache_safe()
        gate()
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
