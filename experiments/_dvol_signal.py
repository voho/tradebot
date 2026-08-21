"""Shared DVOL (BTC 30-day implied-volatility index) signal for the R-73
NOVEL branch. Operator-authored, shared infrastructure -- the lead-time
study (``kelly_regime_v19_dvol_leadtime.py``) imports this unchanged, and a
conditional confirming-vote strategy would have too had the lead-time gate
cleared (it did not -- see that file's own result and this round's ledger
entry; no such strategy file was written), so the underlying signal
construction could never be quietly re-tuned between the measurement step
and a strategy step that never happened.

Data: ``data/btc_dvol_daily.csv.gz``, fetched by
``scripts/fetch_deribit_dvol_novel.py`` from Deribit's public
``get_volatility_index_data`` endpoint. Columns ``open, high, low, close``;
``close`` is the value used throughout (matching how VIX close is used in
``experiments/_macro_signal.py``). Coverage: 2021-03-24 -> present -- a hard
data limitation (options markets did not exist at meaningful scale before
Deribit's own DVOL launch), not a bug, and materially shorter than this
project's usual 2017-> history. Do not backfill or proxy this gap.

Mechanism (why this should carry signal, and why it is a genuinely
different construction from every INFO-axis attempt so far): DVOL is
Deribit's official 30-day implied-volatility benchmark for BTC options --
a FORWARD-LOOKING, PRICED market expectation, set today by participants
who are pricing the NEXT 30 days of realized volatility, not a lagging
description of stress already unfolding. On-chain activity (B-07/R-44) and
stablecoin supply (R-54/R-55/R-58) are spot-market/balance-sheet FLOW
proxies -- observable consequences of stress already in progress. VIX/DXY
(R-53) is a priced expectation too, but describes the REST of the
financial system, and R-53's own lead-time study found it lags BTC's own
3-anchor gate (median -5.5 days). DVOL is a priced expectation of BTC's
OWN volatility specifically, which is the one construction in this
project's INFO line that has not yet been measured for lead time.

Two features are computed, EACH PRE-REGISTERED with a fixed threshold
before any episode date is looked at (see
``kelly_regime_v19_dvol_leadtime.py``'s module docstring for the exact
pre-registration text) -- named because the assignment's mechanism
explicitly allows either "a sharp DVOL spike" (a level reading) or "a fast
rate-of-change in DVOL" (a momentum reading), and prior rounds
(R-53: VIX level, R-53: DXY 20d momentum) used one of each construction
too, so testing both here is not an elaboration invented after the fact:

- ``dvol_z``: DVOL close, z-scored against its own trailing 180-day
  mean/std (``min_periods=60`` -- shorter than R-53's 365-day VIX window
  because DVOL's own history is only ~5.4 years long and the inner-train
  split starts within months of the series' first observation; a 365-day
  window would leave almost no inner-train coverage at all). Levels-based,
  like VIX -- DVOL is likewise mean-reverting and stationary in level, not
  a unit-root series.
- ``dvol_roc5_z``: 5-day log change in DVOL, z-scored against the trailing
  180-day distribution of that same 5-day change (``min_periods=60``). A
  short window (5 days, not DXY's 20) because the assignment's mechanism is
  specifically a FAST spike -- a slower rate-of-change window would not
  distinguish "spike" from "grinding elevated level", which ``dvol_z``
  already captures.

Both features are computed on the RAW DAILY series before any causal shift
(rolling stats never see a future day; ``align_dvol_causal`` only controls
*when* an already-finished daily number becomes visible on the bar grid).
Positive values mean elevated/spiking implied volatility, the risk-off
direction every prior INFO-axis vote in this project has used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import align_dvol_causal, load_dvol_index

DVOL_LEVEL_WINDOW_DAYS = 180
DVOL_ROC_DAYS = 5
DVOL_ROC_WINDOW_DAYS = 180
MIN_PERIODS = 60


def compute_dvol_features(df: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    """Causal ``dvol_z`` and ``dvol_roc5_z`` aligned to ``df``'s bar index,
    or an all-NaN two-column frame if DVOL data is absent.

    Every rolling statistic is computed on the raw daily DVOL frame
    (strictly backward-looking), and only the two finished daily series are
    projected onto the bar grid via ``align_dvol_causal`` -- which
    additionally shifts one more day forward for Deribit's own
    daily-bar-close publication lag, mirroring ``align_macro_causal``/
    ``align_stablecoin_causal``. A bar therefore only ever sees values
    computed from DVOL data published on or before its own previous day.
    """
    dvol = load_dvol_index(data_dir)
    if dvol is None:
        return pd.DataFrame(index=df.index, columns=["dvol_z", "dvol_roc5_z"], dtype=float)

    close = dvol["close"]
    dvol_z = (close - close.rolling(DVOL_LEVEL_WINDOW_DAYS, min_periods=MIN_PERIODS).mean()) / close.rolling(
        DVOL_LEVEL_WINDOW_DAYS, min_periods=MIN_PERIODS
    ).std()

    roc = pd.Series(np.log(close) - np.log(close).shift(DVOL_ROC_DAYS), index=close.index)
    roc_z = (roc - roc.rolling(DVOL_ROC_WINDOW_DAYS, min_periods=MIN_PERIODS).mean()) / roc.rolling(
        DVOL_ROC_WINDOW_DAYS, min_periods=MIN_PERIODS
    ).std()

    daily = pd.DataFrame({"dvol_z": dvol_z, "dvol_roc5_z": roc_z})
    return align_dvol_causal(daily, df)
