"""R-144: does this project's own N~3 edge-concentration claim (L-01/R-62:
`kelly_regime_v4`'s edge over a risk-matched hold concentrates in a handful
of regime-transition episodes), tested with Nguyen & Wolf's (2026) small-N
permutation machinery (R-138), replicate on MORE episodes than R-138's
original six, and does it replicate on ETH at all when ETH is given its
OWN independently-dated event calendar instead of BTC's borrowed one?

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, NEITHER BRANCH EDITS IT, and it does not itself compute a
verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **N~3** (primary), for the second time directly (R-143
was the first: it ADDED events instead of re-analyzing the ones already on
hand). This round does the same move R-143 named as its own unranked
backlog lead #1 ("re-run R-138's permutation test... with genuinely more N
instead of a new statistic on the same six") for the conservative branch,
and a second, independent instance of "add events" for the novel branch --
not to BTC's calendar, but to ETH's, which until now has never had its own
native event set at all; every ETH check in this ledger (R-17, R-47, R-68,
R-127, R-137, R-138's own C3) has evaluated ETH's price action AROUND
BTC's dates, or excised ETH-idiosyncratic events as a NUISANCE to control
for, never used them as the primary calendar for a test.

Why this is the best available direction: a background research pass this
round (WebSearch across 2023-2026 quantitative-crypto and market-
microstructure literature, cross-checked against the ~98-row "C. Ruled
out" table and the R-134->R-143 backlog re-rankings) found this project's
INFO axis (20+ signals), 8 regime-timing-detector theoretical bases, 28+
SIZE-axis retunes, 5 COST-model families, ERR axis (4-6 independent
notions of uncertainty control) and the multi-asset panel all closed
NEGATIVE, and every literature-suggested refinement to any of them
(conformal-interval-width Kelly scaling, a leave-two-out synthetic-control
placebo refinement) lands in a slot four-to-six independent prior rounds
have already shown carries none of `kelly_regime_v4`'s signature (R-62,
R-87), or targets a failure mode (SCM's placebo *resolution*) that is not
the one R-140 actually diagnosed (a *scale-mismatch* confound). This
round's two backlog-named, not-yet-tried leads are the only candidates
that survived that search.

Citations
---------
- Nguyen, P.A. & Wolf, M. (2026), "The permutation test for event studies
  with a small number of events," Empirical Economics 70 (SSRN 5804142) --
  the machinery this round reuses verbatim from `experiments/r138_shared.py`.
  Not re-derived; imported.
- Gandal, Hamrick, Moore & Oberman (2018), J. Monetary Economics 95, 86-96
  -- already cited by R-143 for the Mt. Gox-era manipulation risk that
  keeps 2013 a disclosed sensitivity window, not primary.
- Ethereum Foundation / contemporaneous reporting for the three ETH-native
  calendar dates below (each independently, publicly documented, verified
  by this round's own WebSearch, dates given in UTC): CoinDesk (2020-11-24,
  2020-12-01) for the Beacon Chain genesis; CoinDesk (2021-08-05) for the
  EIP-1559 / "London" hard fork (Buterin et al., EIP-1559, activated at
  block 12,965,000, 12:34 UTC); Ethereum Foundation roadmap page + wire
  reporting for The Merge (2022-09-15), Ethereum's proof-of-work-to-proof-
  of-stake transition. These are the three ETH-specific, non-contagion,
  non-crash, independently news-dated events inside the training period
  (>= ETH's 2019-03-14 data start, <= INNER_VAL_END) this round's search
  found; a fourth candidate (Shanghai/Shapella, enabling staked-ETH
  withdrawals) falls on 2023-04-12, inside the holdout, and is out of
  scope by construction.

Not a duplicate of
-------------------
- R-138 (the ORIGINAL six-episode permutation test on BTC, 5x futures,
  2017-2022): this round reuses its machinery unedited but changes BOTH
  the event count (6 -> 9, conservative) and the calendar's origin +
  market + era (BTC-borrowed dates + futures 2017-2022 -> ETH's OWN
  protocol-event dates + necessarily SPOT/futures per instrument -- see
  guardrails; novel branch stays on futures like R-138, only the
  conservative branch's forced era-extension moves it to spot).
- R-140 (Synthetic Control Method, a structurally different inference
  procedure -- donor-panel placebo, not event-date permutation -- on the
  SAME six/nine-style narrative calendar; this round changes the calendar,
  not the statistic, for the conservative branch, and does not touch SCM
  at all).
- R-143 (extended BTC data back to 2013 and re-ran the matched-exposure
  DRAWDOWN property and the detection-LAG gate on the extended calendar;
  never ran a SIGNIFICANCE TEST -- permutation or otherwise -- on the
  extended calendar. This round's conservative branch is precisely that:
  R-143's own named next-step (1), executed for the first time).
- R-127 / R-137 (informal, bespoke ETH-idiosyncratic-event EXCISION --
  built to remove ETH-specific noise from a BTC-anchored comparison, not
  to build an ETH-native calendar as the PRIMARY test object; neither
  applies the peer-reviewed Nguyen-Wolf machinery).
- R-57 (six-instrument cross-asset panel breadth on PRICES, a different
  object entirely from an event-date significance test on one strategy's
  own excess-return series).

**Is it simulable here?** Yes, entirely: the conservative branch needs
only `kelly_regime_v4` and `ConstantExposureHold`'s own already-frozen
backtest equity curves plus the already-fetched
`data/btcusd_spot_5m_pre2017.csv.gz` (R-143). The novel branch needs only
the already-committed ETH spot file and dates verified by ordinary web
search of public reporting, not any new fetch script. No order book, no
proxying price out of an unavailable channel.

**What would make each branch fail, named now, before any code (beyond
the shared C1/C2 checks inherited from R-138 -- resolution-floor and
Type-I miscalibration, guarded identically here):**

(a) CONSERVATIVE: adding three episodes may not tighten BTC's p-value at
    all, or may push the calibration check (C1) for `n_events=9` outside
    `CALIBRATION_BAND` even though `n_events=6` was in-band on R-138's own
    (2017-2022, futures) series -- the AR series here is a DIFFERENT
    series (2014-2022, spot, forced by pre-2017 futures unavailability),
    so R-138's own C1 pass does not carry over automatically and must be
    re-run on this series at both n=6 and n=9 as a disclosed diagnostic.
    This branch can, at best, tighten or loosen BTC-ONLY significance --
    it structurally CANNOT touch ETH replication (C3), because none of
    the three new episodes falls inside ETH's 2019-03-14+ data. That
    ceiling is stated here, before any run, so it cannot be read into the
    result afterward as if it were news.
(b) NOVEL: with only three events, the calendar may fail C1 (miscalibrated
    on ETH's own autocorrelation/volatility-clustering structure) or C2
    (p >= 0.05) outright -- a legitimate, informative NEGATIVE closing
    this specific escape hatch (does "borrowed BTC dates" explain R-138's
    ETH failure, or does the effect just not exist on ETH). A second,
    named risk: three protocol-upgrade dates are not independent draws
    from "ETH had a regime transition" in the way an unexpected crash is
    -- two of the three (Beacon Chain genesis, EIP-1559) were scheduled
    weeks in advance and could be pre-positioned around, which is a
    genuine disanalogy with the SUDDEN-shock character of most of BTC's
    six-episode calendar (R-85's own diagnosis: the calendar's sudden-
    shock skew is what closed eight regime-timing mechanisms). This is
    disclosed as an interpretive caveat on any pass, not gated on --
    voiding a passing result for being "the wrong kind of event" without
    a pre-registered rule to do so would be moving the goalposts after
    looking.

**Falsification test, pre-registered, evaluated identically by both
branches on the training period (`<= INNER_VAL_END`); NO bar at or after
`OOS_START = 2023-01-01` may be read by either branch during Step 3:**

CONSERVATIVE: does the 9-episode (or 6-episode-on-the-new-series)
BTC-only permutation p-value clear C1+C2 below. Cannot itself resolve
C3 (ETH replication) -- reported honestly as untouched by this branch,
not folded into a pass.

NOVEL: does the ETH-native 3-episode permutation p-value clear C1+C2
below, using ETH's OWN calendar (not BTC's) for the first time in this
project's history. A pass here is the first-ever population of C3 with a
calendar that is not itself borrowed from BTC, and is the decisive check
for whether R-138's ETH failure was about the ASSET or about the DATES.

**Decision rule, pre-registered verbatim (identical statistical bar to
R-138's C1/C2, reused unedited from `experiments/r138_shared.py`):**

A branch's finding is promotable (a genuine METHOD result, not a
PROMOTED strategy -- this round changes no strategy code, mirroring
R-138's own framing) only if:

1. **C1 (calibration).** `r138_shared.empirical_type1_rate(...)` at
   `ALPHA=0.05` lands in `r138_shared.CALIBRATION_BAND = (0.02, 0.09)` on
   the branch's own series and event count. If not, the branch is VOIDED
   for that series and must say so rather than trust its p-value there.
2. **C2 (significance, resolution-aware).** The two-sided permutation
   p-value for the observed CAAR is `< 0.05`, AND the observed statistic
   is beaten by (more extreme than) more than 2 of the `N_PERM = 20000`
   permutation draws.

Anything else is NEGATIVE or METHOD (an informative finding about this
project's own small-N claim, even without a promotion) -- per ROUTINE.md's
standing culture, exactly like R-131 through R-143. **This round produces
no strategy code change regardless of outcome.** If, and only if, a
branch clears C1+C2 -- and, for a claim of GENUINE ETH replication of the
edge-concentration property (not merely "this branch's own p-value"), the
novel branch's CAAR sign must additionally match the conservative branch's
BTC CAAR sign, mirroring R-138's original C3 spirit even though this round
structurally cannot run R-138's literal C3 -- it is reported as a
genuinely new, partial result and routed to an independent skeptic before
being trusted, per ROUTINE.md's parallelism rules ("dispatch a skeptic
only after the primary reports >= 1 evaluated configuration").
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold  # noqa: E402
from experiments.r138_shared import (  # noqa: E402
    ALPHA,
    CALIBRATION_BAND,
    N_CALIBRATION_TRIALS,
    N_PERM,
    PRIMARY_MARKET,
    WINDOW_POST_DAYS,
    WINDOW_PRE_DAYS,
    caar_statistic,
    car_for_event,
    eligible_pseudo_dates,
    empirical_type1_rate,
    load_eth_train,
    permutation_test,
    realized_vol_daily,
    solve_matched_c,
)
from experiments.r143_shared import load_extended_btc_spot  # noqa: E402
from experiments.r143_novel_extended_gate import (  # noqa: E402
    NEW_PRE2017_EPISODES,
    ORIGINAL_SIX,
)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.inference import daily_returns  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

INNER_TRAIN_START = "2017-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# Conservative branch: forced era extension, BTC combined 2014-2022 file.
BTC_EXTENDED_START = "2014-01-01"   # matches R-143's own PRIMARY_START

SPOT = MarketSpec.spot()

EXTENDED_STRESS_EPISODES_9 = list(ORIGINAL_SIX) + list(NEW_PRE2017_EPISODES)

# Novel branch: ETH's own protocol/monetary-policy calendar. Every date
# below is an independently, publicly documented event (guardrail
# identical to R-143's #7), never selected by inspecting where a detector
# or the price series itself has an extremum, and none is a crash/hack --
# a structurally different KIND of event from the six/nine BTC dates,
# deliberately: it tests the edge-concentration claim's own definition of
# "regime transition" against events whose timing is exogenous to price
# but genuinely ETH-specific, rather than borrowing BTC's crash dates.
ETH_NATIVE_EPISODES = [
    ("2020-12 Beacon Chain genesis (ETH2 launch)", "2020-12-01"),
    ("2021-08 EIP-1559 / London hard fork", "2021-08-05"),
    ("2022-09 The Merge (PoW -> PoS transition)", "2022-09-15"),
]


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def _run(strategy, df, market, start=None, end=None, label=""):
    return run_period(strategy, df, start=start, end=end, market=market,
                       start_balance=1_000.0, data_label=label)


def load_btc_extended_train() -> pd.DataFrame:
    """BTC 2014-01-01 -> INNER_VAL_END, spot (forced: no BTC perpetual
    futures existed before 2017; R-143 guardrail 2). Holdout-safe."""
    full = load_extended_btc_spot()
    train = full.loc[BTC_EXTENDED_START:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train


def candidate_and_matched_daily_logret_on(df: pd.DataFrame, market: MarketSpec,
                                          start: str, end: str, label: str = ""
                                          ) -> tuple[pd.Series, pd.Series, float, float]:
    """Same construction as `r138_shared.candidate_and_matched_daily_logret`,
    generalized to an arbitrary `(start, end)` window and pre-loaded frame,
    so the conservative branch can run it on the extended 2014-2022 BTC
    spot series (R-138's own version hardcodes 2017-2022 futures) while the
    novel branch keeps using `r138_shared`'s own version unedited on ETH.
    """
    cand_res = _run(get_strategy("kelly_regime_v4"), df, market, start, end, label)
    target_vol = realized_vol_daily(cand_res.equity)
    c, achieved_vol = solve_matched_c(target_vol, df, market, start, end, label)
    matched_res = _run(ConstantExposureHold(c, static=False), df, market, start, end, label)
    cand_simple = daily_returns(cand_res.equity)
    matched_simple = daily_returns(matched_res.equity)
    idx = cand_simple.index.intersection(matched_simple.index)
    cand_log = np.log1p(cand_simple.loc[idx].clip(lower=-0.999))
    matched_log = np.log1p(matched_simple.loc[idx].clip(lower=-0.999))
    return cand_log, matched_log, c, achieved_vol
