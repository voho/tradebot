"""Shared variance-risk-premium (VRP) signal for the R-73 conservative branch.

PRE-REGISTRATION -- written before any backtest, sweep or holdout read.
Everything below (mechanism, construction, falsification test, decision
rule, and the DVOL-history limitation) is fixed before
``kelly_regime_v18_vrp_brake.py``'s ``sweep()`` is ever run. See that
file's own module docstring for the grid, the decision rule stated a
second time next to the code that enforces it, and the results.

Data: ``data/btc_dvol_daily.csv.gz`` (and, for the falsification test
only, ``data/eth_dvol_daily.csv.gz``) -- Deribit's public DVOL index,
fetched by ``scripts/fetch_deribit_dvol.py``. DVOL is computed from the
live BTC/ETH options order book: a genuine, non-proxied, forward-looking
implied-volatility index, the first of its kind this project has used.
Confirmed reachable; confirmed by direct backward-paging (not by trusting
a documented start date) that history begins **2021-03-24** for both
currencies and nowhere earlier -- Deribit's own options market did not
carry a computed index before that day.

Mechanism (why this should carry signal)
-----------------------------------------
The variance risk premium, ``VRP = implied variance - expected realized
variance``, is the compensation option sellers earn for bearing the risk
that realized volatility spikes -- a positive, time-varying premium
documented across decades of equity-index options (Bakshi, Kapadia &
Madan 2003, *Review of Financial Studies* 16(1); Carr & Wu 2009, *RFS*
22(3); Bollerslev, Tauchen & Zhou 2009, *RFS* 22(11), the last of which
shows VRP predicts equity returns at a quarterly horizon out-of-sample).
The mechanism this brake exploits is narrower and more specific than "buy
when VRP is high": a VRP that compresses toward zero or goes negative
means the options market is pricing LESS insurance than trailing realized
volatility alone would suggest is warranted -- the literature's
complacency signal, historically a precursor to volatility spikes and
drawdowns (option sellers underpricing risk right before it arrives, the
same dynamic behind the 2018 "Volmageddon" VIX-complex blowup in equity
markets). This file constructs ``VRP_t = DVOL_t / 100 - realized_vol_t``
(both annualized decimals; ``realized_vol_t`` is `` kelly_regime_v4``'s
own trailing EWM realized-volatility estimate, already causal by
construction in that file, reused unchanged here rather than
recomputed differently) and reads a COMPRESSED/NEGATIVE VRP, expressed as
a z-score against its own trailing history, as the risk-off signal --
mechanically the mirror image of R-53's macro brake (which fired on
elevated stress) and architecturally identical in shape to R-41's basis
brake and R-53/R-54's macro/stablecoin brakes: a bounded, NEVER-INCREASE-
ONLY multiplicative haircut on ``kelly_regime_v4``'s own exposure.

Why this signal is not a repeat of every prior INFO-axis failure
-------------------------------------------------------------------
Section C of the ledger rules out "Options / volatility risk premium"
with the reason "no options data, no way to validate here" -- the only
row in that section closed for a DATA reason rather than a mechanism
reason. That gap is what this round fills. More importantly, R-53's and
R-54's post-mortems found every prior INFO candidate (VIX/DXY, aggregate
stablecoin supply) LAGS the 3-anchor price gate -- both are spot-market or
balance-sheet FLOW indicators, describing money that has already moved.
DVOL is priced by options market-makers taking a forward-looking bet on
FUTURE realized volatility; it does not need capital to have already
moved for it to reflect a view. That is a structural reason to expect a
different lead/lag profile than R-53/R-54's signals, not a guarantee of
one -- this file's falsification test below is designed to catch it
directly if the structural argument is wrong in practice.

Known failure mode, checked explicitly (not assumed absent)
-----------------------------------------------------------
R-34's and R-53-conservative's identical multiplier SHAPE (``mult in
[1-lam, 1]``) collapsed into a flat rescale of v4 in disguise in both
prior attempts (R^2 = 0.997 and 0.974-0.999 respectively) regardless of
whether the feeding signal was price-derived or genuinely
price-independent. The identical R-33/R-34 exposure-artifact regression
is run on every swept config below and reported honestly whatever it
says -- this is not assumed fixed by using a "genuinely forward-looking"
input.

The DVOL-history limitation -- stated as a known constraint, not
discovered after the fact
--------------------------------------------------------------------
DVOL has NO history before 2021-03-24. This project's usual primary
falsification gate -- survival on the pre-2020 BTC control period,
``btcusd_bitfinex_5m.csv.gz`` -- CANNOT be run for this signal; there is
no DVOL data to compute a VRP against for that period, and this file does
not synthesize, backfill, or proxy one to get around that (the project's
own standing rule: "never proxy unavailable data out of price"). The
standard inner-train (2017-2020) / inner-validation (2021-2022) split
from ROUTINE.md is therefore inapplicable as written -- inner-train would
be 100% NaN-VRP (mult=1 fallback, i.e. IDENTICAL to v4, testing nothing).
This file instead compresses the entire iteration budget into the
DVOL-covered window before OOS_START, split as:

    DVOL_TRAIN  2021-04-01 -> 2022-04-30   (~13 months, one stress episode:
                                             the May 2021 crash, ~6 weeks
                                             into TRAIN)
    DVOL_VALID  2022-05-01 -> 2022-12-31   (~8 months, TWO stress episodes:
                                             Terra/Luna, May 2022, and FTX,
                                             November 2022)

This is a MATERIALLY WEAKER evidentiary base than any other round in this
ledger: roughly 21 months of total pre-holdout data covering at most
2-3 genuine stress events, against this project's usual multi-year
inner-train/inner-validation split covering the entire 2017-2020 bull/
bear cycle plus 2021-2022. A favorable point estimate here should NOT be
read with the confidence this project extends to a result built on the
full inner split -- n≈2-3 events is not a distribution, it is barely more
than the n≈3 the ledger's own standing diagnosis already flags project-
wide as its second binding constraint. This is named here, before any
number below is produced, exactly as this round's brief requires.

Trailing-window z-score, not a full-series fit
------------------------------------------------
Given the short total history, this file uses a 120-CALENDAR-DAY trailing
rolling window (45-day minimum periods) for the VRP z-score -- shorter
than R-53/R-54's 365-day convention, a deliberate, up-front consequence of
DVOL's short history rather than a value tuned to this round's outcome.
Every rolling statistic (the realized-vol EWM inherited from v4, and the
VRP mean/std computed here) uses only bars at or before the current one --
``pandas.Series.rolling``/``.ewm`` are backward-looking by construction,
and the causal alignment of the daily DVOL close onto the bar grid follows
the identical shift-by-one-day-then-ffill convention already reviewed and
used for FRED (``align_macro_causal``) and CoinMetrics
(``align_onchain_causal``/``align_stablecoin_causal``) data in
``tradebot/data.py`` -- reimplemented locally in this file (not imported
from ``tradebot/data.py``) specifically so this branch touches no file the
sibling novel branch (also built on DVOL) might concurrently want to edit,
per this round's parallelism instructions. A day D's DVOL close is the
volatility index value AT the end of day D (UTC), i.e. AT the instant day
D+1 begins -- unlike FRED/CoinMetrics there is no additional external
publication lag to model, so shifting the index by exactly one day and
forward-filling onto the bar grid is the correct (not merely conservative)
causal timestamp for this data, checked directly by the tamper-causality
probe in ``kelly_regime_v18_vrp_brake.py``.

Pre-registered falsification test (ROUTINE.md step 2, chosen now)
---------------------------------------------------------------------
**Does it survive on ETH?** Deribit publishes an independent ETH DVOL
index (confirmed reachable, identical 2021-03-24 start), so this is a
genuine asset-specific VRP test -- ETH's own implied vol against ETH's
own realized vol, fed through the identical mechanism -- not a proxy
through BTC's signal. Outcome that kills it, named now: if the ETH
construction shows a decisively worse or oppositely-signed pattern
relative to its own control than the BTC construction does relative to
its own control (the R-53/R-54 asset-specific-signature failure mode),
this direction fails, full stop -- a market-wide options-implied-vol
signal that only "works" on the one asset it was designed against is not
evidence of a real mechanism.

Additionally required by this round's brief (not a ROUTINE.md-approved
substitute for the ETH test above, but both are mandatory here):
exposure-artifact R^2 (candidate's target path against a mean-notional-
matched flat rescale of v4's own target; R^2 > 0.95 means "relabeled
leverage, not a real finding") and a plateau check (neighbouring
lam/z_scale cells must not diverge wildly from the selected cell).

Full pre-registered decision rule (promote-to-holdout / reject) is stated
in ``kelly_regime_v18_vrp_brake.py``'s module docstring, next to the code
that enforces it, and is not repeated here to avoid two copies drifting
apart.

Not a duplicate of
-------------------
- Section C's "Options / volatility risk premium" row itself: ruled out
  ONLY for lack of data ("no options data, no way to validate here"),
  never on the merits. That reason is gone; this file is the first
  attempt on the merits.
- **R-41** (``kelly_regime_v9_basis_brake.py``): same never-increase-only
  architecture, but basis is a spot/perp PRICE spread (still fundamentally
  a traded-price signal); DVOL is an options-market volatility index, not
  a price level or spread of BTC/ETH itself.
- **R-53** (``kelly_regime_v14_macro_brake.py``): identical multiplier
  SHAPE and the architectural template this file follows, but macro
  (VIX/DXY) is a TRAINING-adjacent asset class, not options on the traded
  asset itself, and R-53 found it lags. This file's mechanism argument for
  why DVOL should NOT lag the same way is stated above and is exactly what
  the causality/lead-time evidence in the results write-up will test.
- **R-54** (stablecoin supply, ``kelly_regime_v15_stablecoin_veto.py`` and
  siblings): a crypto-native BALANCE-SHEET flow signal (confirmed to lead
  price by R-54, but too imprecise on the merits). DVOL is neither a flow
  signal nor balance-sheet data -- it is a forward-looking price of
  insurance, structurally different from every INFO candidate tried so
  far.
- A sibling agent runs a structurally different exploitation of DVOL (a
  spike/level lead-time signal, not a VRP construction) in parallel this
  round, on disjoint files. Not read or coordinated with here, per
  ROUTINE.md's parallelism rules.

Constraint attacked
--------------------
INFO -- the project's #1 standing-diagnosis constraint, the fourth
information-channel attempt (after R-41's basis, R-44's on-chain, R-53's
macro, R-54's stablecoin), and the first built from a genuinely
forward-looking (rather than trailing-flow or contemporaneous-price)
source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BARS_PER_DAY = 288

DVOL_FILES = {
    "BTC": "btc_dvol_daily.csv.gz",
    "ETH": "eth_dvol_daily.csv.gz",
}

# Fixed a-priori, stated before any sweep is run (see module docstring).
ROLL_WINDOW_DAYS = 120
MIN_WINDOW_DAYS = 45


def load_dvol_raw(data_dir: str | Path, currency: str = "BTC") -> pd.DataFrame | None:
    """Daily DVOL OHLC, or None if the file is absent.

    Local to this module (not added to ``tradebot/data.py``) so this
    branch touches no file the sibling novel branch, also built on DVOL,
    might concurrently want to edit -- see the module docstring.
    """
    if currency not in DVOL_FILES:
        raise ValueError(f"currency must be one of {sorted(DVOL_FILES)}")
    path = Path(data_dir) / DVOL_FILES[currency]
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = df.index.tz_localize("UTC")
    return df[["close"]].rename(columns={"close": "dvol_close"}).astype(float).sort_index()


def align_dvol_causal(dvol: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex daily DVOL close onto ``bars``' index, causally.

    Day D's close is the index value AT the end of day D UTC -- i.e. at
    the instant day D+1 begins. Shifting the daily index forward by one
    day and forward-filling onto the bar grid (identical mechanics to
    ``align_macro_causal``/``align_onchain_causal`` in ``tradebot/data.py``,
    reimplemented here rather than imported -- see module docstring) is
    therefore the correct causal timestamp, not merely a conservative one:
    a bar at time T only ever sees the DVOL close of the most recent day
    that closed strictly before T's own day.
    """
    shifted = dvol.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


def compute_vrp_z(
    df: pd.DataFrame,
    realized_vol,
    data_dir: str | Path,
    currency: str = "BTC",
    roll_window_days: int = ROLL_WINDOW_DAYS,
    min_window_days: int = MIN_WINDOW_DAYS,
) -> pd.Series:
    """Causal VRP z-score aligned to ``df``'s bar index, or all-NaN if DVOL is absent.

    ``realized_vol`` must already be a causal (backward-looking, already
    shifted by the caller) array/Series aligned to ``df.index`` -- this
    function does not shift it again. ``VRP_raw = dvol_close/100 -
    realized_vol`` (both annualized decimals); the z-score is computed
    against a TRAILING rolling window of ``VRP_raw`` itself (never the
    whole series -- see module docstring for why this window is 120 days,
    not the project's usual 365), so early bars naturally get a
    progressively-warming-up window rather than a full-series mean/std
    that could not have been known at the time.
    """
    dvol = load_dvol_raw(data_dir, currency)
    if dvol is None:
        return pd.Series(index=df.index, dtype=float)

    aligned = align_dvol_causal(dvol, df)["dvol_close"] / 100.0
    rv = pd.Series(realized_vol, index=df.index, dtype=float)
    vrp_raw = (aligned - rv).rename("vrp_raw")

    window = int(roll_window_days * BARS_PER_DAY)
    min_periods = int(min_window_days * BARS_PER_DAY)
    mean = vrp_raw.rolling(window, min_periods=min_periods).mean()
    std = vrp_raw.rolling(window, min_periods=min_periods).std()
    vrp_z = (vrp_raw - mean) / std.replace(0.0, pd.NA)
    return vrp_z.rename("vrp_z").astype(float)
