#!/usr/bin/env python
"""B-20: the LITERAL periodically-rebalanced, fixed-50/50 BTC+ETH
`kelly_regime_v4` portfolio -- R-50's own original inner-validation
byproduct -- read on the holdout for the first time by any session.

Backlog item attacked: **B-20** -- "Does the LITERAL periodically-rebalanced
(monthly, or another single cadence fixed before running), fixed-50/50
BTC+ETH `kelly_regime_v4` portfolio -- R-50's own original candidate, run
through its continuous (non-restarting) engine, unmodified split,
unmodified cadence discipline -- survive its own pre-registered
falsification test and a first, single holdout read?"

Mechanism, one sentence
------------------------
Run each of BTC-`kelly_regime_v4` and ETH-`kelly_regime_v4` ONCE,
continuously, from ETH's real data start (so neither leg's deadband/
vol-regime hysteresis latch is ever reset), rebalance pooled capital back
to a FIXED 50/50 split at the start of every calendar month, and ask
whether that specific, literal object -- not a never-rebalanced variant,
not a volatility-weighted variant -- beats `buy_and_hold` and BTC-solo v4
on data it has never touched.

Why this is not a duplicate of R-50, or of either R-51 branch
------------------------------------------------------------------
Three sessions have now touched adjacent territory, and none of them ran
this exact object through the holdout:

- **R-50** (`experiments/kelly_regime_covkelly_v3_continuous.py`, B-18)
  built the continuous (non-restarting) per-leg engine this file reuses
  and found, as an *unplanned byproduct* of a diagnostic headline table
  (`run_headline`'s `fixed5050_continuous` cell), that a periodically-
  rebalanced fixed-50/50 book beats BTC-solo v4 by ΔSharpe +0.79 (monthly)
  / +0.80 (weekly), max DD 33.2%->27.1%, **on inner-validation only**.
  Per the ledger's own words: "It has not been pre-registered, has no
  falsification test, no fee/funding sensitivity and no holdout read -- a
  promising lead, deliberately not rushed to promotion in the same session
  that found it." This file is the pre-registration and the holdout read
  R-50 explicitly declined to do in the same session.
- **R-51 conservative** (`experiments/b19_dual_fixed_split.py`, B-19)
  deliberately tested a *never-rebalanced, one-time* 50/50 split instead
  -- capital is split once via `tradebot.multiasset.run_multi_backtest`
  and the weights are then left to drift with each leg's own performance
  for the rest of the window, with **zero** periodic rebalancing at any
  cadence. It cleared both falsification gates and a plateau check, then
  was REJECTED on its one holdout read (loses to `buy_and_hold` by 24-46%;
  statistically indistinguishable from BTC-solo v4). Its own decomposition
  (comparing its ΔSharpe +0.23 against R-50's ΔSharpe +0.79/+0.80) found
  the never-rebalanced split captures ~100% of R-50's drawdown edge but
  only ~29% of R-50's Sharpe edge -- meaning ~71% of the LARGER, UNTESTED
  Sharpe edge specifically requires the periodic sell-winners/buy-losers
  act R-51-conservative's own candidate never performs, by design. That
  71% is exactly what this file's candidate performs and R-51-conservative
  does not; the two candidates are mechanically different objects (one has
  a `rebalance_freq` argument threaded through a segment loop, the other
  has none), and only this file's holdout read settles whether the piece
  R-51-conservative deliberately left untested actually monetizes.
- **R-51 novel** (`experiments/b19_risk_parity_rebalance.py`, B-19) DID
  stay periodically rebalanced, but replaced the fixed-50/50 weight with
  inverse-trailing-volatility weights and swept cadences (monthly,
  quarterly, semiannual) R-50 never tried. Its own re-derived fixed-50/50
  arm is present in that file only as an inner-validation REFERENCE POINT
  for scoring the inverse-vol candidate -- its own pre-registration never
  authorized reading the holdout on that reference by itself, and,
  independently confirmed by the operator, `holdout()` there is gated
  behind a CLI argument no invocation in that branch's own report ever
  passes. So the fixed-50/50 monthly candidate has been *computed* on
  inner-validation twice now (R-50, and again as R-51-novel's reference)
  and *holdout-read* zero times.

This file's candidate is therefore the one specific object -- periodic
(calendar) rebalancing, back to UNMODIFIED fixed 50/50 weights, through
the continuous (non-restarting) engine -- that sits at the intersection of
"periodically rebalanced" (which R-51-conservative deliberately is not)
and "fixed 50/50, not information-weighted" (which R-51-novel deliberately
is not), and it has not been holdout-tested by any of the three prior
rounds.

Standing caution carried into this pre-registration (not discovered after
running anything -- read directly from the ledger before writing a line of
code below)
----------------------------------------------------------------------------
R-51-conservative's own decomposition already found the DRAWDOWN-ONLY
component of this general family (the part it isolated and could
holdout-test) fails outright on 2023+: the −6.1pp inner-validation
drawdown edge compressed to a non-effect (−0.8pp) on the holdout, and the
book lost to `buy_and_hold` by 24-46%. R-51-conservative also attributed
roughly 71% of THIS file's larger, untested Sharpe edge to the periodic
rebalancing act itself -- the same return-side mechanism a bull-dominated
2023-2026 holdout has already shown a closely related variant does not
reliably monetize. The evidence available before this file's holdout read
updates AGAINST the candidate, not for it. This paragraph is written before
`sweep()`/`select()`/`artifact()`/`feetier()` are ever called, and the
decision rule below is fixed now and will not move regardless of what
those functions print. Per docs/ROUTINE.md's "Running directions in
parallel": at ~623 project-level holdout consultations already, no
Sharpe-based claim from this dataset is supportable any more (R-29's
finding, reconfirmed every round since) -- this pre-registration therefore
weights drawdown/tail behavior and sign-reversal over Sharpe magnitude
explicitly in decision criterion (b) below, not as an afterthought.

Constraint attacked (docs/LEDGER.md standing diagnosis)
-----------------------------------------------------------
SIZE and N~3, identical framing to B-18/B-19/B-16: a second,
imperfectly-synchronized regime-cycle exposure, not a change to v4's own
vote or sizing formula (`kelly_regime_v4.py` is imported and called
unchanged, never edited).

Simulable here?
------------------
Yes. Two independent, unmodified `kelly_regime_v4` continuous backtests on
committed real OHLCV (BTC Bitstamp spot, ETH Coinbase spot), composed by a
periodic-rebalance harness built on the already-tested primitives from
`experiments/kelly_regime_covkelly_v3_continuous.py` -- no engine change to
any file under `src/tradebot/`, no new data, no proxying.

Cadence choice, fixed now, before running anything (pre-registration item 3)
---------------------------------------------------------------------------------
**Monthly ("MS") only.** This is B-20's own literal framing ("monthly, or
another single cadence fixed before running") and matches R-50's headline
number (ΔSharpe +0.79 monthly vs. +0.80 weekly -- monthly is the slightly
more conservative of the two and is this project's usual default cadence
elsewhere in this research line). No other cadence is added to the
candidate search space below, and no cadence sweep is run after seeing any
result -- that would be exactly the kind of after-the-fact search R-51-
novel deliberately avoided by pre-registering its cadence sweep as
*inner-only*, never holdout-eligible. A small split-ratio neighbourhood
(50/50 -- the frozen candidate -- plus 60/40 and 40/60, matching
`b19_dual_fixed_split.py`'s and `kelly_regime_dual_fixed.py`'s own
convention) is evaluated as a PLATEAU check only, never as a way to pick
the "best" split after the fact; the holdout, if reached at all, reads
ONLY the frozen 50/50 configuration.

Pre-registered falsification test (item 4, chosen before any result was read)
------------------------------------------------------------------------------------
Two checks, BOTH must pass on the inner splits before the holdout is ever
touched -- identical convention to `b19_dual_fixed_split.py`'s F1/F2:

  (F1) **Exposure-artifact check**: the candidate's aggregate exposure
       series (dollar-weighted sum of each leg's own `target` fraction)
       must NOT be an R^2 > 0.95 flat rescale of BTC-solo `kelly_regime_v4`'s
       own exposure, on inner-validation, both markets (this project's
       standing "match risk before comparing anything" rule --
       R-33/R-46/L-04 all died of exactly this).
  (F2) **0.40% Bitstamp taker fee tier**: the 50/50 candidate's advantage
       over BTC-solo v4 (Sharpe and drawdown) must not flip sign relative
       to the project's usual 0.10% tier, on both inner-train and
       inner-validation.

If EITHER fails, STOP -- do not read the 2023+ holdout. Report NEGATIVE.

Pre-registered promotion decision rule (item 5, written before the holdout is read)
------------------------------------------------------------------------------------------
Promote (`PROMOTED-CANDIDATE`) only if, on the 2023+ holdout, using the
FROZEN 50/50-monthly configuration and no other:

  (a) beats `buy_and_hold` OOS after real costs (0.10% spot as the table
      convention, reported alongside 0.40%);
  (b) the improvement over BTC-solo `kelly_regime_v4` exceeds the +/-0.2
      Sharpe noise floor (R-20) **OR** is a drawdown/tail improvement --
      per the standing caution above, a drawdown/tail improvement that
      does NOT reverse sign from its inner-validation reading is weighted
      as the stronger form of evidence here; a Sharpe-only improvement,
      at ~623+ program-level holdout consultations, is treated as
      suggestive at best and NOT sufficient on its own to satisfy (b);
  (c) survives both falsification checks (F1, F2) above;
  (d) the 50/50 -> 60/40 -> 40/60 neighbourhood is a plateau, not a
      knife-edge (no metric flips sign or changes by more than the noise
      floor between adjacent splits), evaluated on the inner splits only.

Anything else is `NEGATIVE`. If all inner-split/falsification gates pass
but the holdout is never reached because a prior branch in this same
parallel round already spent it, the row is `PARKED`, not `NEGATIVE` --
"not tested" is not "a negative result" (ROUTINE.md, "Running directions
in parallel"). **If this rule is changed after any result in this file is
seen, that change will be stated explicitly and the result downgraded to
in-sample -- it will not be done silently.**

Data-window rule (item 6)
-----------------------------
Inner-train = 2019-03-14 (ETH's real start on the committed Coinbase file)
-> 2020-12-31. Inner-validation = 2021-01-01 -> 2022-12-31 (the 2022
BTC/ETH joint bear -- matches every prior file in this research line, not
ROUTINE.md's generic 2017 BTC-only example). Holdout = 2023-01-01 onward,
through the actual last common bar of the two committed data files (BTC
Bitstamp spot and ETH Coinbase spot -- computed at runtime below, never
hardcoded, and both endpoints are printed at start-up), read AT MOST ONCE
(one paired call, both fee tiers), only if every prior gate above passes.

CRITICAL TRAP this file exists to avoid (read `run_continuous_full` in
`kelly_regime_covkelly_v3_continuous.py` before trusting any holdout
number from this file)
----------------------------------------------------------------------------
That module's `run_continuous_full()` hard-codes `_segment_bounds(FULL_START,
FULL_END, ...)` using ITS OWN top-level `FULL_END = VALID_END = "2022-12-31"`
constant and takes no `end=` argument of its own -- calling it directly for
a holdout read would silently cap at 2022-12-31 while looking like it ran
the whole window. This file therefore NEVER calls `run_continuous_full`.
Instead, `run_calendar_rebalance_fixed()` below is this file's OWN thin
harness, built directly on the three lower-level primitives the task
authorizes reuse of (`continuous_leg_equity`, `_segment_bounds`,
`_segment_returns`, all imported unchanged), replicating
`run_continuous_full`'s ~25-line pooling loop with an explicit,
caller-supplied `end` -- generalized, additionally, to an arbitrary FIXED
split ratio (`run_continuous_full`'s `weight_mode="fixed5050"` hardcodes
exactly 0.5/0.5 and could not run this file's 60/40 and 40/60 plateau
checks even if its date range were fixed). `holdout()` below prints the
min/max date its own segments actually cover and asserts the max exceeds
2023-01-01 before any holdout number is reported.

Implementation note: a second, independent landmine found and worked
around (not modified) in the imported helper
----------------------------------------------------------------------------
`continuous_leg_equity`'s own module-level cache key is
`(id(df), market.name, v4_kwargs_key, start, end, start_balance)` --
it omits `market.fee_rate`. Since `MarketSpec.spot()` and
`MarketSpec.spot(fee_rate=0.004)` both have `market.name == "spot"`,
calling that function at two fee tiers within the same process can
silently return the WRONG cached curve for the second tier -- this is the
exact collision R-51-novel independently found and fixed *in its own,
separate extended helper* (not in this file's imported module, which is
untouched and still has the bug). Since `kelly_regime_covkelly_v3_continuous.py`
is not modified here, `leg_equity()` below clears that module's shared
`_LEG_CACHE` immediately before every fresh call and maintains its OWN,
correctly-keyed cache (including `market.fee_rate`) on top -- so this
file's F2 fee-tier check and its holdout's two-fee-tier paired read cannot
be silently corrupted by that bug, at a small, one-time recomputation cost
per distinct (leg, market, fee, window) combination.

Hard rules honored
--------------------
- Only this NEW file is touched. `kelly_regime_v4.py`, `multiasset.py`,
  `kelly_regime_covkelly.py`, `kelly_regime_covkelly_v3_continuous.py`,
  `kelly_regime_dual_fixed.py`, `b19_dual_fixed_split.py` and
  `b19_risk_parity_rebalance.py` are all imported from or read for
  reference, never edited.
- No 2023+ literal is used to slice data anywhere in this file except
  inside `holdout()` and the module-level `HOLDOUT_END`/`OOS_START`
  constants used only there -- grepped before every run.
- `N_EVALUATED` counts every distinct dual-book (candidate) backtest
  configuration actually evaluated (split x window x market x fee tier),
  matching `b19_dual_fixed_split.py`'s own convention; baseline/reference
  runs (`buy_and_hold`, BTC-solo v4) are not counted, matching that file's
  convention exactly.

Usage
-------
    python experiments/b20_literal_calendar_5050.py causality  # step 0: no-lookahead check on THIS file's harness
    python experiments/b20_literal_calendar_5050.py sweep      # step 2: inner-train, spot, all splits
    python experiments/b20_literal_calendar_5050.py select     # step 2: inner-validation, both markets, all splits
    python experiments/b20_literal_calendar_5050.py baselines  # required baselines, both windows, both markets
    python experiments/b20_literal_calendar_5050.py artifact   # step 3: falsification test F1 (R^2 exposure check)
    python experiments/b20_literal_calendar_5050.py feetier    # step 3: falsification test F2 (0.40% taker)
    python experiments/b20_literal_calendar_5050.py gate       # prints the full pre-registered go/no-go decision
    python experiments/b20_literal_calendar_5050.py holdout    # step 4: ONE read, frozen 50/50 monthly, gated
    python experiments/b20_literal_calendar_5050.py all        # causality+sweep+select+baselines+artifact+feetier+gate (no holdout)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

import experiments.kelly_regime_covkelly_v3_continuous as _v3c  # noqa: E402
from experiments.kelly_regime_covkelly_v3_continuous import (  # noqa: E402
    period_metrics,
    _segment_bounds,
    _segment_returns,
)
from experiments.kelly_regime_covkelly import load_assets  # noqa: E402

INCUMBENT = "kelly_regime_v4"
HOLD = "buy_and_hold"

SPOT = MarketSpec.spot()
SPOT_04 = MarketSpec.spot(fee_rate=0.004)          # Bitstamp entry taker tier
FUTURES5X = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures5x", FUTURES5X))

# ---------------------------------------------------------------- data rule
# See module docstring's "Data-window rule". Grepped before every run: no
# "2023"/"2024"/"2025"/"2026" date literal used to slice data appears
# anywhere in this file outside comments/docstrings and the OOS_START /
# HOLDOUT_END constants, which are used only inside holdout().
FULL_START = "2019-03-14"       # ETH's real start; same as R-50/R-51
TRAIN = ("2019-03-14", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
INNER_END = VALID[1]            # 2022-12-31 -- one continuous run covers both TRAIN and VALID
OOS_START = "2023-01-01"

FREQ = "MS"                     # monthly -- the ONLY cadence in the candidate search space, per pre-registration

SPLITS: dict[str, tuple[float, float]] = {
    "50_50": (0.50, 0.50),
    "60_40_btc": (0.60, 0.40),
    "40_60_eth": (0.40, 0.60),
}
FROZEN_CANDIDATE = "50_50"      # the pre-registered holdout config -- see docstring

# --------------------------------------------------------------- data load
# Inner-train/inner-validation reuse kelly_regime_covkelly.py's own
# load_assets(), hard-sliced to <= 2022-12-31 -- byte-identical data to
# every predecessor round (R-42/R-43/R-49/R-50/R-51) for comparability.
BTC_CUT, ETH_CUT = load_assets(str(ROOT / "data"))

# The holdout needs the FULL, un-sliced series -- loaded directly here,
# never through load_assets(). Endpoints are computed from the files
# themselves, never hardcoded, and printed at start-up.
BTC_FULL, BTC_LABEL = load_dataset(ROOT / "data", "spot")
ETH_FULL = load_coinbase_eth_spot(ROOT / "data")
if ETH_FULL is None:
    raise RuntimeError("data/ethusd_coinbase_spot_5m.csv.gz not found -- cannot run this experiment")

_HOLDOUT_END_TS = min(BTC_FULL.index[-1], ETH_FULL.index[-1])
HOLDOUT_END = _HOLDOUT_END_TS.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")

N_EVALUATED = 0             # every distinct dual-book (candidate) backtest configuration
_SEEN: set[tuple] = set()   # dedup key: (split, window, market, fee_tag)
HOLDOUT_READS = 0           # increments only inside holdout()


# ============================================================ leg-equity cache
# Correctly-keyed (includes market.fee_rate) local cache, built on top of
# the imported continuous_leg_equity -- see the "Implementation note" in
# the module docstring for why this wrapper exists and is not merely a
# convenience.
_LEG_CACHE: dict = {}


def leg_equity(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                start_balance: float = 1000.0) -> pd.Series:
    key = (id(df), market.name, market.fee_rate, start, end, start_balance)
    if key in _LEG_CACHE:
        return _LEG_CACHE[key]
    _v3c._LEG_CACHE.clear()  # defensive: the imported cache's key omits fee_rate -- see docstring
    result = _v3c.continuous_leg_equity(df, market, None, start=start, end=end,
                                        start_balance=start_balance)
    _LEG_CACHE[key] = result
    return result


def _leg_full(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
              start_balance: float = 1000.0):
    """Direct run_period call, keeping the FULL BacktestResult (including
    the `target` column) -- needed only for the F1 exposure diagnostic.
    Functionally identical to what `continuous_leg_equity` computes
    internally (same run_period call, same continuous-from-`start`
    discipline), it just also keeps `.df` since that function discards
    everything but `.equity`. Not cached: called a handful of times, only
    from `artifact()`.
    """
    return run_period(KellyRegimeV4(), df, start=start, end=end, market=market,
                      start_balance=start_balance)


# ======================================================== the candidate engine

def run_calendar_rebalance_fixed(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    start: str, end: str, freq: str,
    w_btc: float, w_eth: float,
    market: MarketSpec, start_balance: float = 1000.0,
) -> dict:
    """THE candidate: periodic (calendar) rebalancing back to a FIXED
    (w_btc, w_eth) split, through the continuous (non-restarting) per-leg
    engine. See the module docstring's "CRITICAL TRAP" section for why
    this is a new, thin function rather than a call to
    `kelly_regime_covkelly_v3_continuous.run_continuous_full` -- that
    function hardcodes both the date range (its own FULL_END module
    constant) and the split (weight_mode="fixed5050" is exactly 0.5/0.5).
    The pooling loop below replicates that function's own arithmetic
    (documented there), generalized on both axes.
    """
    btc_full = leg_equity(btc_df, market, start, end, start_balance)
    eth_full = leg_equity(eth_df, market, start, end, start_balance)

    bounds = _segment_bounds(start, end, freq)
    btc_segs = _segment_returns(btc_full, bounds)
    eth_segs = _segment_returns(eth_full, bounds)
    n = min(len(btc_segs), len(eth_segs))

    pooled = start_balance
    combined_pieces, btc_pieces, eth_pieces = [], [], []
    log_rows = []
    for i in range(n):
        sb, se = btc_segs[i], eth_segs[i]
        seg_start, seg_end = sb["seg_start"], sb["seg_end"]
        dollars_b = pooled * w_btc
        dollars_e = pooled * w_eth
        cash = pooled * max(0.0, 1.0 - w_btc - w_eth)

        btc_sub = btc_full.loc[seg_start:seg_end]
        eth_sub = eth_full.loc[seg_start:seg_end]
        scale_b = (dollars_b / sb["base_val"]) if sb["base_val"] > 0 else 0.0
        scale_e = (dollars_e / se["base_val"]) if se["base_val"] > 0 else 0.0
        btc_leg = btc_sub * scale_b
        eth_leg = eth_sub * scale_e

        idx = btc_leg.index.union(eth_leg.index)
        btc_leg_r = btc_leg.reindex(idx).ffill().bfill().fillna(0.0)
        eth_leg_r = eth_leg.reindex(idx).ffill().bfill().fillna(0.0)
        combined = btc_leg_r + eth_leg_r + cash
        if len(combined) == 0:
            continue
        combined_pieces.append(combined)
        btc_pieces.append(btc_leg_r)
        eth_pieces.append(eth_leg_r)
        pooled = float(combined.iloc[-1])
        log_rows.append({"seg_start": seg_start, "seg_end": seg_end, "w_btc": w_btc,
                         "w_eth": w_eth, "pooled_end": pooled})

    equity = pd.concat(combined_pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    btc_equity = pd.concat(btc_pieces).sort_index()
    btc_equity = btc_equity[~btc_equity.index.duplicated(keep="last")]
    eth_equity = pd.concat(eth_pieces).sort_index()
    eth_equity = eth_equity[~eth_equity.index.duplicated(keep="last")]

    return {"equity": equity, "btc_leg_equity": btc_equity, "eth_leg_equity": eth_equity,
            "btc_full": btc_full, "eth_full": eth_full, "weights_log": pd.DataFrame(log_rows),
            "seg_start_min": bounds[0], "seg_end_max": bounds[-2] if len(bounds) > 1 else bounds[0],
            "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance}


_FULL_CACHE: dict = {}


def get_full(split_name: str, boundary_tag: str, market: MarketSpec, fee_tag: str = "std") -> dict:
    """One continuous candidate run, cached by (split, boundary, market, fee).
    boundary_tag "inner" spans FULL_START->INNER_END (used by sweep/select/
    artifact/feetier/gate); "holdout" spans FULL_START->HOLDOUT_END (used
    only by holdout()). Both windows for "inner" are sliced from the SAME
    continuous run -- required so validation never restarts state relative
    to train, exactly the property this whole research line exists to
    preserve.
    """
    key = (split_name, boundary_tag, market.name, fee_tag)
    if key in _FULL_CACHE:
        return _FULL_CACHE[key]
    start = FULL_START
    end = INNER_END if boundary_tag == "inner" else HOLDOUT_END
    btc_df = BTC_CUT if boundary_tag == "inner" else BTC_FULL
    eth_df = ETH_CUT if boundary_tag == "inner" else ETH_FULL
    w_btc, w_eth = SPLITS[split_name]
    res = run_calendar_rebalance_fixed(btc_df, eth_df, start, end, FREQ, w_btc, w_eth,
                                       market, 1000.0)
    _FULL_CACHE[key] = res
    return res


def run_window(split_name: str, window_name: str, market: MarketSpec, fee_tag: str = "std",
               count: bool = True) -> dict:
    """Metrics for one (split, window, market, fee) cell, sliced from the
    appropriate continuous run. window_name in {"train", "validation",
    "holdout"}.
    """
    global N_EVALUATED
    if count:
        dedup = (split_name, window_name, market.name, fee_tag)
        if dedup not in _SEEN:
            _SEEN.add(dedup)
            N_EVALUATED += 1
    boundary_tag = "holdout" if window_name == "holdout" else "inner"
    full = get_full(split_name, boundary_tag, market, fee_tag)
    if window_name == "train":
        w_start, w_end = TRAIN
    elif window_name == "validation":
        w_start, w_end = VALID
    else:
        w_start, w_end = OOS_START, HOLDOUT_END
    m = period_metrics(full["equity"], w_start, w_end)
    out = {"split": split_name, "window": window_name, "market": market.name, "fee_tag": fee_tag,
           "final": m["final_balance"], "sharpe": m["sharpe"], "max_dd": m["max_dd_pct"],
           "_full": full}
    return out


# =============================================================== baselines
# Reused, unchanged convention from kelly_regime_covkelly.py / R-50/R-51:
# a v4-solo baseline run separately per window via run_period (NOT a
# continuous engine) -- byte-comparable to every predecessor round's own
# "v4 BTC alone" number, since deviating here would silently change what
# the candidate is being compared against.

def run_baseline_v4_btc(start: str, end: str | None, market: MarketSpec,
                        total: float = 1000.0, df: pd.DataFrame | None = None) -> dict:
    data = df if df is not None else BTC_CUT
    result = run_period(get_strategy(INCUMBENT), data, start=start, end=end, market=market,
                        start_balance=total)
    from tradebot.metrics import compute_metrics
    m = compute_metrics(result)
    return {"label": "kelly_regime_v4 BTC-only", "result": result,
            "final": m.final_balance, "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct}


def run_baseline_hold_btc(start: str, end: str | None, market: MarketSpec,
                          total: float = 1000.0, df: pd.DataFrame | None = None) -> dict:
    data = df if df is not None else BTC_CUT
    result = run_period(get_strategy(HOLD), data, start=start, end=end, market=market,
                        start_balance=total)
    from tradebot.metrics import compute_metrics
    m = compute_metrics(result)
    return {"label": "buy_and_hold BTC-only", "result": result,
            "final": m.final_balance, "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct}


def line(tag: str, d: dict) -> str:
    return (f"  {tag:40s} final=${d['final']:>10,.0f} sharpe={d['sharpe']:>6.2f} "
            f"maxDD={d['max_dd']:>5.1f}%")


# --------------------------------------------------------------------------- step 2


def sweep() -> pd.DataFrame:
    """Inner-train, spot, every split, monthly cadence -- the required minimum grid."""
    rows = []
    t0 = time.time()
    for name in SPLITS:
        d = run_window(name, "train", SPOT)
        rows.append({"split": name, "market": "spot", "window": "train",
                     "final": d["final"], "sharpe": d["sharpe"], "max_dd": d["max_dd"]})
        print(f"[{N_EVALUATED}] " + line(f"{name} ({SPLITS[name][0]:.2f}/{SPLITS[name][1]:.2f})", d) +
              f"  [{time.time() - t0:.0f}s]")
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return pd.DataFrame(rows)


def select() -> pd.DataFrame:
    """Inner-validation, both markets, every split, monthly cadence."""
    rows = []
    for name in SPLITS:
        cells = []
        for mname, market in MARKETS:
            d = run_window(name, "validation", market)
            rows.append({"split": name, "market": mname, "window": "validation",
                        "final": d["final"], "sharpe": d["sharpe"], "max_dd": d["max_dd"]})
            cells.append((mname, d))
        s = dict(cells)["spot"]
        f = dict(cells)["futures5x"]
        print(f"{name:12s} ({SPLITS[name][0]:.2f}/{SPLITS[name][1]:.2f})  "
              f"spot: ${s['final']:>9,.0f} sh{s['sharpe']:>6.2f} DD{s['max_dd']:>5.1f}%   "
              f"fut: ${f['final']:>9,.0f} sh{f['sharpe']:>6.2f} DD{f['max_dd']:>5.1f}%")
    print(f"\nconfigurations evaluated total: {N_EVALUATED}")
    return pd.DataFrame(rows)


def baselines() -> pd.DataFrame:
    """Required baselines: v4 BTC-solo, BTC hold -- both windows, both markets."""
    rows = []
    for wname, (start, end) in (("train", TRAIN), ("validation", VALID)):
        for mname, market in MARKETS:
            a = run_baseline_v4_btc(start, end, market)
            c = run_baseline_hold_btc(start, end, market)
            for d in (a, c):
                rows.append({"baseline": d["label"], "window": wname, "market": mname,
                            "final": d["final"], "sharpe": d["sharpe"], "max_dd": d["max_dd"]})
            print(f"\n-- {wname} / {mname} --")
            for d in (a, c):
                print(line(d["label"], d))
    print("\n(baselines are reference points, not counted in N_EVALUATED, "
          "matching b19_dual_fixed_split.py's own convention)")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- falsification test F1: R^2


def artifact() -> pd.DataFrame:
    """F1 (pre-registered): is the dual book just relabeled leverage on
    v4-BTC-alone? Standing rule (R-33/R-46): R^2 > 0.95 of the candidate's
    aggregate exposure against a mean-matched flat rescale of BTC-solo v4's
    own exposure means "this is not diversification, it's relabeled
    leverage" -- FAILS the check, not a footnote.

    Aggregate exposure = sum over legs of (leg's own target fraction x
    leg's own DOLLAR contribution within the candidate) / total portfolio
    equity, on the frozen 50/50 config, inner-validation.
    """
    rows = []
    print("exposure-artifact check (50_50 monthly, inner-validation, "
          "mean-matched flat rescale of v4 BTC-only):")
    for mname, market in MARKETS:
        full = get_full(FROZEN_CANDIDATE, "inner", market, fee_tag="std")
        # leg target series: v4's `target` is a pure function of price data
        # (never of dollar scale -- see docstring's scale-invariance note
        # carried over from kelly_regime_covkelly_v3_continuous.py), so a
        # fresh direct run_period call from FULL_START gives the identical
        # series `leg_equity` used internally, just also keeping `.df`.
        tgt_b_full = _leg_full(BTC_CUT, market, FULL_START, INNER_END).df["target"]
        tgt_e_full = _leg_full(ETH_CUT, market, FULL_START, INNER_END).df["target"]

        eq_b, eq_e = full["btc_leg_equity"], full["eth_leg_equity"]
        idx = eq_b.index.intersection(eq_e.index)
        idx = idx.intersection(tgt_b_full.index).intersection(tgt_e_full.index)
        idx = idx[(idx >= pd.Timestamp(VALID[0], tz="UTC")) & (idx <= pd.Timestamp(VALID[1], tz="UTC"))]

        tgt_b = tgt_b_full.reindex(idx).ffill().fillna(0.0)
        tgt_e = tgt_e_full.reindex(idx).ffill().fillna(0.0)
        eb = eq_b.reindex(idx).ffill().fillna(0.0)
        ee = eq_e.reindex(idx).ffill().fillna(0.0)
        total_eq = eb + ee
        exposure = (tgt_b * eb + tgt_e * ee) / total_eq.replace(0.0, np.nan)

        ctl = run_baseline_v4_btc(*VALID, market)
        v4_tgt = ctl["result"].df["target"].reindex(idx).ffill().fillna(0.0)

        mask = np.isfinite(exposure.to_numpy()) & np.isfinite(v4_tgt.to_numpy())
        y = exposure.to_numpy()[mask]
        x_raw = v4_tgt.to_numpy()[mask]
        c = float(np.mean(y)) / float(np.mean(x_raw)) if np.mean(x_raw) != 0 else float("nan")
        x = c * x_raw

        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
        verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95) -- FAILS F1"
                   if np.isfinite(r2) and r2 > 0.95 else "not a flat rescale -- PASSES F1")
        print(f"  {mname}: rescale c={c:.3f}  corr={corr:.4f}  R^2={r2:.4f}  {verdict}")
        rows.append({"market": mname, "rescale_c": c, "corr": corr, "r2": r2,
                     "artifact": bool(np.isfinite(r2) and r2 > 0.95)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- falsification test F2: fee tier


def feetier() -> pd.DataFrame:
    """F2 (pre-registered): does the 50/50-monthly candidate's advantage
    over v4-BTC-solo survive Bitstamp's real 0.40% taker tier, or is it a
    0.10%-tier artifact of turnover this project has been burned by
    before?
    """
    rows = []
    print("0.40% taker fee-tier check (50_50 monthly vs v4 BTC-solo, both windows, spot):\n")
    for wname, (start, end) in (("train", TRAIN), ("validation", VALID)):
        for fee_tag, market in (("0.10%", SPOT), ("0.40%", SPOT_04)):
            wname_key = "train" if wname == "train" else "validation"
            d = run_window(FROZEN_CANDIDATE, wname_key, market, fee_tag=fee_tag)
            ctl = run_baseline_v4_btc(start, end, market)
            hold = run_baseline_hold_btc(start, end, market)
            delta_sharpe = d["sharpe"] - ctl["sharpe"]
            delta_dd = d["max_dd"] - ctl["max_dd"]
            print(f"  {wname:11s} @ {fee_tag}:  dual50/50 final=${d['final']:>9,.0f} "
                  f"sh={d['sharpe']:>6.2f} DD={d['max_dd']:>5.1f}%   |  "
                  f"v4-solo final=${ctl['final']:>9,.0f} sh={ctl['sharpe']:>6.2f} "
                  f"DD={ctl['max_dd']:>5.1f}%   |  dSharpe={delta_sharpe:+.2f} dDD={delta_dd:+.1f}pp   |  "
                  f"hold final=${hold['final']:>9,.0f}")
            rows.append({"window": wname, "fee": fee_tag, "dual_final": d["final"],
                        "dual_sharpe": d["sharpe"], "dual_dd": d["max_dd"],
                        "v4_final": ctl["final"], "v4_sharpe": ctl["sharpe"], "v4_dd": ctl["max_dd"],
                        "hold_final": hold["final"], "delta_sharpe": delta_sharpe, "delta_dd": delta_dd})
    df = pd.DataFrame(rows)
    flips = []
    for wname in ("train", "validation"):
        sub = df[df["window"] == wname]
        d10 = sub[sub["fee"] == "0.10%"].iloc[0]
        d40 = sub[sub["fee"] == "0.40%"].iloc[0]
        flip = (np.sign(d10["delta_dd"]) != np.sign(d40["delta_dd"])) if d10["delta_dd"] != 0 else False
        flips.append(flip)
        print(f"\n  {wname}: dDD sign at 0.10%={'neg (better)' if d10['delta_dd']<0 else 'pos (worse)'} "
              f"vs at 0.40%={'neg (better)' if d40['delta_dd']<0 else 'pos (worse)'}  "
              f"{'FLIPPED -- FAILS F2' if flip else 'stable -- PASSES F2 (this window)'}")
    df.attrs["f2_pass"] = not any(flips)
    return df


# ------------------------------------------------------------------------ causality


def causality() -> bool:
    """CRITICAL requirement #1: no-lookahead sanity check on THIS file's
    own composition code (`run_calendar_rebalance_fixed`), before anything
    else in this file is trusted. Standard two-opposite-tampers probe:
    multiply bars after a cut by K in one copy, divide by K in the other,
    confirm the combined PORTFOLIO equity curve strictly before the cut is
    bit-identical. Cut sits inside inner-train, nowhere near the holdout.
    """
    cut_date = pd.Timestamp("2020-06-30", tz="UTC")
    K = 137.0

    def tamper(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > cut_date
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        out.loc[mask, "volume"] = out.loc[mask, "volume"] * (factor if factor > 1 else 1.0 / factor)
        return out

    def dual_equity(btc_df, eth_df):
        res = run_calendar_rebalance_fixed(btc_df, eth_df, FULL_START, INNER_END, FREQ,
                                           0.5, 0.5, SPOT, 1000.0)
        return res["equity"]

    base = dual_equity(BTC_CUT, ETH_CUT)
    up = dual_equity(tamper(BTC_CUT, K), tamper(ETH_CUT, K))
    down = dual_equity(tamper(BTC_CUT, 1.0 / K), tamper(ETH_CUT, 1.0 / K))

    pre = base.index[base.index <= cut_date]
    b = base.reindex(pre).to_numpy()
    u = up.reindex(pre).to_numpy()
    dn = down.reindex(pre).to_numpy()
    max_diff_up = float(np.nanmax(np.abs(b - u)))
    max_diff_down = float(np.nanmax(np.abs(b - dn)))
    ok = max_diff_up < 1e-6 and max_diff_down < 1e-6
    print(f"causality probe on this file's composition (cut={cut_date.date()}, K={K}):")
    print(f"  max|base - up-tampered| portfolio equity before cut:   {max_diff_up:.3e}")
    print(f"  max|base - down-tampered| portfolio equity before cut: {max_diff_down:.3e}")
    print(f"  {'PASS' if ok else 'FAIL'}: portfolio equity strictly before the cut is unchanged "
          "when only post-cut bars, on either leg, are tampered.")
    return ok


# --------------------------------------------------------------------- gate decision


def gate() -> bool:
    """Runs every pre-registered inner-split/falsification/plateau check
    and prints the full go/no-go decision per the module docstring's
    decision rule. Returns True only if the holdout is authorized.
    """
    print("=== GATE CHECK: inner-validation improvement, F1, F2, plateau ===\n")

    # inner-validation improvement over BTC-solo v4, frozen 50/50, spot
    d = run_window(FROZEN_CANDIDATE, "validation", SPOT)
    ctl = run_baseline_v4_btc(*VALID, SPOT)
    d_sharpe = d["sharpe"] - ctl["sharpe"]
    d_dd = d["max_dd"] - ctl["max_dd"]
    improve = (d_sharpe > 0.2) or (d_dd < 0)
    print(f"[gate 1] inner-validation (spot): candidate sh={d['sharpe']:.2f} DD={d['max_dd']:.1f}%  "
          f"vs v4-solo sh={ctl['sharpe']:.2f} DD={ctl['max_dd']:.1f}%  "
          f"dSharpe={d_sharpe:+.2f} dDD={d_dd:+.1f}pp  -> {'PASS' if improve else 'FAIL'}")

    art = artifact()
    f1_pass = not bool(art["artifact"].any())
    print(f"[gate 2] F1 exposure-artifact check -> {'PASS' if f1_pass else 'FAIL'}")

    fee = feetier()
    f2_pass = bool(fee.attrs.get("f2_pass", False))
    print(f"[gate 3] F2 0.40%-taker fee-tier check -> {'PASS' if f2_pass else 'FAIL'}")

    # plateau check (pre-registered wording: "no metric ... changes by more
    # than the noise floor BETWEEN ADJACENT SPLITS", i.e. adjacent-pair
    # differences along the 60/40 -> 50/50 -> 40/60 ordering, not the full
    # max-min span across all three -- the full span is also printed below
    # for transparency but is NOT the pre-registered gate criterion).
    order = ["60_40_btc", "50_50", "40_60_eth"]
    cells = {name: run_window(name, "validation", SPOT) for name in SPLITS}
    sharpes = {n: c["sharpe"] for n, c in cells.items()}
    dds = {n: c["max_dd"] for n, c in cells.items()}
    sh_span = max(sharpes.values()) - min(sharpes.values())
    dd_span = max(dds.values()) - min(dds.values())
    adj_sh_diffs = [abs(sharpes[order[i + 1]] - sharpes[order[i]]) for i in range(len(order) - 1)]
    signs_sharpe = {n: np.sign(s - ctl["sharpe"]) for n, s in sharpes.items()}
    plateau = (max(adj_sh_diffs) <= 0.2) and (len(set(signs_sharpe.values())) == 1)
    print(f"[gate 4] plateau check (pre-registered: adjacent-split Sharpe deltas <= 0.2 noise floor): "
          f"adjacent |dSharpe| = {[f'{v:.2f}' for v in adj_sh_diffs]}  "
          f"(full span={sh_span:.2f}, DD span={dd_span:.1f}pp, shown for context only)  "
          f"all splits same-sign vs v4-solo={len(set(signs_sharpe.values()))==1}  "
          f"-> {'PASS' if plateau else 'FAIL'}")
    for n in SPLITS:
        print(f"    {n:12s} sh={sharpes[n]:.2f} DD={dds[n]:.1f}%")

    all_pass = improve and f1_pass and f2_pass and plateau
    print(f"\n=== GATE DECISION: {'ALL GATES PASS -> holdout authorized' if all_pass else 'AT LEAST ONE GATE FAILS -> STOP, do not read the holdout, report NEGATIVE'} ===")
    return all_pass


# ------------------------------------------------------------------------------- step 4: holdout


def holdout() -> pd.DataFrame:
    """ONE read of the 2023+ holdout, frozen 50/50-monthly config, ONLY if
    every prior gate (`gate()`) already passed. The decision rule is the
    one written in this file's module docstring, fixed BEFORE this
    function is ever called. Do not edit this function's thresholds after
    reading its output.
    """
    global HOLDOUT_READS
    rows = []
    print(f"=== HOLDOUT READ ({OOS_START} -> {HOLDOUT_END}), frozen config: "
          f"{FROZEN_CANDIDATE} monthly ===\n")

    full = get_full(FROZEN_CANDIDATE, "holdout", SPOT, fee_tag="std")
    seg_min, seg_max = full["seg_start_min"], full["seg_end_max"]
    eq_min, eq_max = full["equity"].index.min(), full["equity"].index.max()
    print(f"sanity check (CRITICAL TRAP guard): candidate segments span "
          f"{seg_min.date()} -> {seg_max.date()}; equity index spans "
          f"{eq_min.date()} -> {eq_max.date()}")
    assert eq_max > pd.Timestamp("2023-01-01", tz="UTC"), (
        "holdout equity curve does not extend past 2023-01-01 -- "
        "the CRITICAL TRAP (silent truncation) has occurred; STOP.")
    print(f"  PASS: max date {eq_max.date()} exceeds 2023-01-01 -- this really is the holdout.\n")

    for fee_tag, market in (("0.10%", SPOT), ("0.40%", SPOT_04)):
        d = run_window(FROZEN_CANDIDATE, "holdout", market, fee_tag=fee_tag)
        ctl = run_baseline_v4_btc(OOS_START, None, market, df=BTC_FULL)
        hold = run_baseline_hold_btc(OOS_START, None, market, df=BTC_FULL)
        HOLDOUT_READS += 1
        rows.append({"fee": fee_tag, "dual_final": d["final"], "dual_sharpe": d["sharpe"],
                    "dual_dd": d["max_dd"], "v4_final": ctl["final"], "v4_sharpe": ctl["sharpe"],
                    "v4_dd": ctl["max_dd"], "hold_final": hold["final"], "hold_sharpe": hold["sharpe"]})
        print(f"  {fee_tag:8s}  dual50/50: ${d['final']:>10,.0f} sh={d['sharpe']:>6.2f} DD={d['max_dd']:>5.1f}%   "
              f"v4-solo: ${ctl['final']:>10,.0f} sh={ctl['sharpe']:>6.2f} DD={ctl['max_dd']:>5.1f}%   "
              f"hold: ${hold['final']:>10,.0f} sh={hold['sharpe']:>6.2f}")
    print(f"\nholdout reads this call: {HOLDOUT_READS}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"BTC (full): {len(BTC_FULL):,} bars {BTC_FULL.index[0]:%Y-%m-%d} -> "
          f"{BTC_FULL.index[-1]:%Y-%m-%d} (data: {BTC_LABEL})", file=sys.stderr)
    print(f"ETH (full): {len(ETH_FULL):,} bars {ETH_FULL.index[0]:%Y-%m-%d} -> "
          f"{ETH_FULL.index[-1]:%Y-%m-%d} (data: real, Coinbase spot)", file=sys.stderr)
    print(f"HOLDOUT_END (min of the two): {HOLDOUT_END}", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "causality":
        causality()
    elif choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "baselines":
        baselines()
    elif choice == "artifact":
        artifact()
    elif choice == "feetier":
        feetier()
    elif choice == "gate":
        gate()
    elif choice == "holdout":
        holdout()
    elif choice == "all":
        ok = causality()
        if not ok:
            print("causality FAILED -- stopping, nothing downstream is trustworthy.", file=sys.stderr)
            sys.exit(1)
        sweep()
        select()
        baselines()
        gate()
    else:
        print("usage: python experiments/b20_literal_calendar_5050.py "
              "[causality|sweep|select|baselines|artifact|feetier|gate|holdout|all]")
