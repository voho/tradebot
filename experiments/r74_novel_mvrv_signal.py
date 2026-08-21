"""Shared MVRV rate-of-change signal for the R-74 NOVEL branch. Private to
this branch -- the conservative branch (MVRV *level*) owns its own,
disjoint files and is not read or touched here, per this round's brief.

Data: ``data/btc_mvrv_daily.csv.gz`` / ``data/eth_mvrv_daily.csv.gz``,
already fetched and committed (CoinMetrics free community API,
``CapMVRVCur``). Loaded via the already-added, unmodified
``tradebot.data.load_mvrv_ratio`` / ``align_mvrv_causal``.

Idea, one sentence (ROUTINE.md step 2)
---------------------------------------------------------------------------
MVRV = market cap / realized cap (Mahmudov & Puell 2018, building on
Carter & Le Calvez's realized-cap concept, Honeybadger 2018); realized cap
marks every coin at the price it last moved on-chain, so a FALL in MVRV
means market cap is dropping relative to the aggregate cost basis of coins
that have actually settled on-chain -- i.e. the holder base is moving into
unrealized loss -- which is this project's fifth structurally distinct
INFO-axis construction (not a flow like stablecoin supply, R-54; not a
priced volatility expectation like DVOL/VRP, R-73; not a spillover from
the rest of the financial system like VIX/DXY, R-53; not the traded
asset's own network activity like on-chain address counts, B-07/R-44).

Why a RATE OF CHANGE, not the level (that is the conservative branch's job)
---------------------------------------------------------------------------
This project's one confirmed-leading signal (R-54: aggregate stablecoin
supply DECELERATION, +16.5 days median) and every confirmed-lagging one
(R-53: VIX/DXY LEVEL, -5.5d; R-73: DVOL LEVEL, -2.0d) share a pattern: a
LEVEL is a snapshot a trailing price anchor can eventually catch up to on
its own arithmetic, whereas a RATE OF CHANGE can register the early
derivative of a move before a slow trailing average crosses it. R-73 also
found that switching a lagging level to its own rate of change is not
automatically a fix (DVOL's 5-day ROC still lagged, -9.0 days) -- so the
window has to be chosen for what the specific underlying quantity's own
dynamics are, not applied as a rote transformation.

What window, and why (the citable reasoning required BEFORE any number is
computed -- this is the whole point of this file existing before the
lead-time script runs)
---------------------------------------------------------------------------
Realized cap is a STOCK built from the entire supply's cost basis, and it
only reprices when coins actually move on-chain -- mechanically the same
accounting device as Coin Days Destroyed / dormancy-flow analysis, whose
own literature documents that most BTC supply sits in long-term,
infrequently-transacting wallets (median UTXO age measured in months, not
days). Over a horizon SHORTER than the time it typically takes a
meaningful slice of that dormant supply to actually move, realized cap is
close to constant, so ``market_cap = price x circulating_supply`` dominates
the ratio and ``Delta log(MVRV)`` collapses toward ``Delta log(price)`` --
a relabeled price return, not new information beyond what v4's own
trailing-price anchors already see. That is a DIFFERENT, and more
structural, reason than the one that motivated stablecoin supply's short
14-day window (R-54): mint/burn events are on-chain and near-instantaneous
once stress triggers redemptions, so a short window suited that flow.
Realized cap's repricing is not event-triggered in the same way -- it
accrues gradually as holders capitulate, take profit, or otherwise spend
coins that had been sitting still, a process the dormancy literature
describes as unfolding over weeks, not days. A stablecoin-style 14-day
window is therefore very likely the WRONG timescale for MVRV specifically,
for a mechanical reason named before any lead-time number exists, not
discovered after a short window looked bad.

Two windows are used, both fixed a-priori and NOT swept for the
best-looking lead time:

- ``mvrv_roc30_z`` (PRIMARY): 30-day log change in MVRV, long enough to
  give the realized-cap denominator a real chance to move (roughly the
  middle of v4's own 20/40/80-day anchor ladder, and a full month for
  on-chain settlement to occur) while still short enough to test a
  genuinely leading hypothesis against the FASTEST (20-day) anchor rather
  than degenerating into a restatement of v4's own slowest anchor.
- ``mvrv_roc90_z`` (SECONDARY, a robustness companion, not a search): a
  90-day window, matching both Mahmudov & Puell's own original framing of
  MVRV as a multi-month cycle-valuation metric and v4's slowest (80-day)
  anchor. If the mechanism genuinely reflects gradual realized-cap
  repricing rather than price momentum relabeled, 30-day and 90-day should
  tell a CONSISTENT story (comparable lead/lag direction), not a
  contradictory one where only the cherry-picked window looks good --
  that consistency check is itself part of what this file's two features
  are for, decided before either number was computed.

Both features z-scored against their own trailing 365-day distribution
(``min_periods=60``, matching ``_stablecoin_signal.py``'s convention
exactly), computed entirely on the RAW DAILY MVRV series before any causal
shift (rolling/shift are backward-looking only -- ``align_mvrv_causal``
only controls *when* an already-finished daily number becomes visible on
the bar grid, exactly as it does for every other INFO signal in this
project). Sign-flipped so POSITIVE = MVRV falling/decelerating relative to
its own trailing year = risk-off, matching every prior signal's convention
in this project (``_macro_signal.py``, ``_stablecoin_signal.py``,
``_dvol_signal.py`` all use positive = risk-off).

**Named risk, stated before any code ran:** it is a fully legitimate,
real possibility that even a rate-of-change construction still lags,
exactly as R-73 found for DVOL's own 5-day ROC despite DVOL being a
forward-priced signal in level form. If the lead-time check finds a lag,
that is reported plainly as a real negative result, per this round's
pre-registered stop rule, not explained away or rescued by trying a third
window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import align_mvrv_causal, load_mvrv_ratio

ROC_WINDOWS_DAYS = (30, 90)  # (PRIMARY, SECONDARY) -- fixed a-priori, see module docstring
ZSCORE_WINDOW_DAYS = 365
MIN_PERIODS = 60


def compute_mvrv_features(df: pd.DataFrame, data_dir: str | Path, asset: str = "BTC") -> pd.DataFrame:
    """Causal ``mvrv_roc30_z`` / ``mvrv_roc90_z`` aligned to ``df``'s bar
    index, or an all-NaN two-column frame if MVRV data is absent for
    ``asset``.

    Every rolling statistic (the N-day log change, the 365-day z-score
    mean/std) is computed on the raw daily MVRV frame (strictly
    backward-looking: a value at day D uses only days <= D), and only the
    two finished daily series are projected onto the bar grid via
    ``align_mvrv_causal`` -- which additionally shifts one more day
    forward for CoinMetrics' own publication lag, exactly as every other
    INFO-axis loader in this project already does.
    """
    mvrv = load_mvrv_ratio(data_dir, asset=asset)
    cols = [f"mvrv_roc{n}_z" for n in ROC_WINDOWS_DAYS]
    if mvrv is None:
        return pd.DataFrame(index=df.index, columns=cols, dtype=float)

    log_mvrv = np.log(mvrv["mvrv"])
    daily = {}
    for n in ROC_WINDOWS_DAYS:
        roc = log_mvrv - log_mvrv.shift(n)
        z = (roc - roc.rolling(ZSCORE_WINDOW_DAYS, min_periods=MIN_PERIODS).mean()) / roc.rolling(
            ZSCORE_WINDOW_DAYS, min_periods=MIN_PERIODS
        ).std()
        daily[f"mvrv_roc{n}_z"] = -1.0 * z  # sign-flip: positive = MVRV falling = risk-off

    daily_df = pd.DataFrame(daily)
    return align_mvrv_causal(daily_df, df)
