#!/usr/bin/env python
"""kelly_regime_v4 with a bounded, never-increase brake from real on-chain MVRV (CONSERVATIVE branch).

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per docs/ROUTINE.md step 5. Do not decorate with ``@register``.

=========================================================================
PRE-REGISTRATION (written before any inner-validation number was read)
=========================================================================

Mechanism, one sentence
------------------------
Stand down from full exposure -- via a bounded, never-increase multiplier
``mult = 1 - lam * excess(t) in [1-lam, 1]`` applied on top of v4's
unchanged vote and conditional-vol-targeting scale -- when the market is
trading at an extreme premium to its own on-chain cost basis, i.e. when a
causal rolling Z-score of CoinMetrics' MVRV ratio (market cap / realized
cap; Mahmudov & Puell 2018) clears a high threshold; the brake is
deliberately ASYMMETRIC (fires only at high MVRV, never at low/negative
Z) because low-MVRV capitulation is, per this project's own diagnosis, a
state where v4's price-based vote is already bearish-or-flat -- braking
further there would need its own separate justification this round does
not attempt.

Constraint attacked
--------------------
INFO. MVRV is not a transform of the OHLCV close column this project's
every strategy has read to date -- it is priced from the blockchain's own
transaction history (the price each coin last moved on-chain), which two
assets or two moments trading at an identical spot price can differ on
arbitrarily depending on how recently supply changed hands. Also SIZE,
since the brake acts only through the same multiplicative exposure axis
every working strategy in this project already uses.

Which ledger rows this is not a duplicate of
-----------------------------------------------
- **L-12 harsanyi_crowd** (INFO, NEGATIVE): a Bayesian posterior over
  bull/bear/chop built from bar-RETURN likelihoods -- price in, price out.
  MVRV cannot be computed from the Bitstamp OHLCV file at all; it requires
  the CoinMetrics realized-cap series, itself derived from on-chain UTXO
  age/value, a different database entirely.
- **L-14/L-15/L-16** (camouflage_flow / stealth_trend / flow_regime, INFO,
  NEGATIVE): all three reconstruct order-flow proxies (BVC, participation
  filters) OUT OF the same OHLCV bars a naive momentum rule already reads
  -- "a price transform, not order flow" (L-14's own recorded lesson).
  MVRV is the opposite case: it is a foreign, independently-measured
  dataset merged ONTO the price index, not squeezed out of it.
- **R-34 conservative** (``kelly_regime_v5_damp.py``): the SAME bounded
  never-increase architectural template (``mult in [1-lam,1]``, v4's
  vote/scale untouched) -- but its signal (the Harsanyi posterior margin)
  is itself built from bar returns and was shown to correlate R^2=0.997
  with a flat rescale of v4's own exposure once smoothed enough to be
  tradeable -- "a smoothed copy of a constant," the INFO constraint
  unaddressed despite the new architecture. This round's signal is a
  foreign daily series merged onto the bar index; whether it also
  degenerates into a flat rescale is re-checked directly below (Diagnostic
  1), not assumed innocent by analogy.
- **R-41 conservative** (``kelly_regime_v9_basis_brake.py``): the closest
  relative -- same architecture, same session's other new-data-channel
  round -- but basis is ``log(perp/spot)``, a spread between two PRICES;
  no matter how many venues are compared, a price spread is still built
  entirely out of trades, the thing INFO says this project already has
  one series of. MVRV's realized-cap leg is not a price at all: it is a
  running weighted average of the price at which currently-circulating
  coins last changed hands on-chain, which requires ledger data no
  exchange trade feed carries. R-41's brake was SYMMETRIC in the sign of
  basis (both a premium and a discount signal a two-sided squeeze/
  deleveraging risk from the SAME mechanism, forced liquidation); this
  round's brake is explicitly ASYMMETRIC per the task brief's literal
  reading -- justified above, not inherited from R-41.
- **R-35/R-37/R-38/R-40**: none of these introduce a new data source; all
  rework what OHLCV close already implies (retuned constants, per-state
  Kelly, CRRA caps, anchor-ladder bagging). Their shared failure
  signature -- wins on inner-validation (2021-22) but loses to v4 on the
  earlier train window -- is exactly Diagnostic 2 below, run explicitly.

One pre-registered falsification test
----------------------------------------
Confirmed empirically first (see ``data verification`` output): the
Coinbase ETH-USD spot series (``load_coinbase_eth_spot``) covers
2019-03-14 -> present; CoinMetrics ETH MVRV (``load_onchain(dir,"ETH")``)
covers 2015-08-09 -> present with the same daily granularity as BTC. The
overlap window usable WITHOUT touching this project's 2023-01-01 holdout
is **2019-03-14 -> 2022-12-31**.

Test: take the single primary candidate config selected on
inner-validation (chosen below by BTC inner-validation Sharpe among
configs that also pass Diagnostics 1-2), freeze it unchanged, and run it
on ETH spot and ETH 5x (same Coinbase price series, ``MarketSpec.futures``
on it -- this project's standard convention for a second market when no
independent ETH perp series is being used for pricing, R-17). **Kill
condition, stated now:** if the candidate's final balance is BELOW
``kelly_regime_v4``'s on ETH in EITHER market cell over 2019-03-14 ->
2022-12-31, that falsifies the mechanism -- it says the brake fits BTC's
own 2013/2018/2021 MVRV extremes specifically rather than expressing a
transferable "stand down from an extreme cost-basis premium" rule.

What would make me reject this branch outright (decision rule, written
before looking at any inner-validation number)
----------------------------------------------------------------------
REJECT (do not recommend promotion) if ANY of:
1. Diagnostic 1 (exposure-artifact check): R^2 > 0.95 against a
   mean-notional-matched flat rescale of v4's own target on
   inner-validation, either market -- means the brake is just delevering,
   not reading MVRV.
2. Diagnostic 2 (train-vs-validation signature): the selected candidate
   loses to v4 on final balance on the EARLIER inner-train window
   (2017-2020) in either market, while only winning on inner-validation
   (2021-22) -- the exact R-37/R-38/R-40 overfitting signature.
3. The ETH falsification test above fails.
4. Causality probes (price-tamper AND MVRV-tamper) fail.
5. The parameter neighbourhood around the selected candidate is a narrow
   peak (adjacent grid cells swing Sharpe/final-balance sharply) rather
   than a plateau.
Only if ALL five pass would this be a promotion candidate -- and even
then, per my mandate, the 2023+ holdout is NOT to be read this session;
that decision belongs to the orchestrating session.

=========================================================================
Design notes (decided from data exploration BEFORE any backtest was run)
=========================================================================

Data verification (see ``python experiments/kelly_regime_v10_mvrv_brake.py
verify``): ``load_onchain`` returns real CoinMetrics daily MVRV, BTC from
2010-07-19 (2,358 finite days before 2017), ETH from 2015-08-09. It is
NOT a price transform -- ``tradebot/data.py`` reads it from a committed
CSV built by ``scripts/fetch_coinmetrics_onchain.py`` from CoinMetrics'
own realized-cap dataset, a column no OHLCV close can produce. The loader
already shifts CoinMetrics' own day-start timestamp forward by one day
before returning, so ffilling it onto the 5m bar index adds zero extra
lag (reconfirmed directly in ``verify()``, not merely trusted from the
docstring).

**Data-quality finding that shapes the design**: BTC MVRV in the first
~2 weeks after 2010-07-19 is a severe outlier artifact of the network's
own infancy (realized cap trivially small at genesis) -- values as high
as 146x, decaying to normal (<6x) levels by roughly August 2010 and
staying there. A pure *expanding-since-genesis* Z-score (the textbook
formulation of the classic "MVRV Z-score" indicator) would therefore
have its mean/std permanently contaminated by two weeks of data in 2010.
This round does NOT use an expanding-since-genesis window for that
reason -- it uses fixed-length CAUSAL ROLLING windows instead (365/
730/1095 calendar days), all of which age the 2010 contamination out
well before the earliest window relevant to any backtested bar (the
longest swept window, 1095 days, reaches back to 2014-01-01 for a
2017-01-01 evaluation start -- nowhere near 2010). This is a design
decision made from inspecting the raw MVRV distribution, not from any
inner-validation result, and is exactly the kind of choice this
pre-registration exists to freeze before looking further.

Note this differs from Mahmudov & Puell (2018)'s original formulation,
which Z-scores the DOLLAR distance (market cap - realized cap) using the
dollar magnitude's own stdev; ``load_onchain`` exposes only the RATIO
(mvrv = market cap / realized cap), so this round Z-scores the ratio
directly. Grobys (2026, matching this project's own 2017-2026 window)
found the ratio-based MVRV Z-score the strongest of three on-chain rules
tested against buy-and-hold and a Monte Carlo random-entry null across
three full BTC cycles -- the closest published analogue to what is being
attempted here, on materially the same data window.

**Window length is a swept hyperparameter** (365d / 730d / 1095d), not a
single guess, per the task brief.

Causality
---------
The daily Z-score is built with ``.rolling(window, min_periods=window)``
(a FULL window is required before it emits a number -- no partial-window
peeking) on the DAILY on-chain frame, optionally causally EMA-smoothed
(``.ewm(span=smooth_days, min_periods=1)``), then reindexed onto the 5m
spot bar grid with a strict as-of ffill (``_mvrv_z_on_index``, mirroring
``kelly_regime_v9_basis_brake.py``'s ``_basis_on_index`` convention --
never interpolated, never backfilled). No full-series ``.mean()``/
``.std()``/quantile is used anywhere (the R-21 trap this project's
truncation test alone would not catch). Two independent two-opposite-
tampers causality probes are run: the standard PRICE probe (copied
verbatim from ``kelly_regime_v9_basis_brake.py``) and a new probe that
tampers the cached MVRV-Z series itself after the same cut -- exercising
the one new ingredient a price-only probe cannot reach.

Fallback / warmup correctness
-------------------------------
Before a window is fully populated (the daily MVRV series' first
``window`` days), the causal Z-score is NaN by construction
(``min_periods=window``); wherever the reindexed Z is NaN, ``excess`` is
forced to exactly 0.0 (``mult == 1.0``), matching v4 exactly -- verified
below by an exact bit-identical diff against unmodified v4 over that
whole prefix, not merely argued from the code. Because MVRV has 6+ years
of real history before this project's 2017 backtest start, this NaN
prefix ends well before any evaluated bar for every swept window (the
longest, 1095 days, finishes warming by ~2013-07-18) -- so, unlike
``kelly_regime_v9_basis_brake.py`` (whose Deribit coverage begins
mid-backtest, 2018-08-14), the fallback here is a pure correctness
check on a pre-2017 prefix, not a live behaviour during any measured
window.

R-08 trap (on-chain flows secretly timing volatility)
---------------------------------------------------------
B-07's own standing warning: "on-chain flows predict *volatility*, and
R-08 showed better volatility input makes this strategy worse." Checked
directly below (``volcorr``): correlation between the MVRV Z-score (raw
and smoothed) and trailing/forward realized volatility, both inner
windows, both signs reported plainly.

Usage
-----
    python experiments/kelly_regime_v10_mvrv_brake.py verify    # data verification
    python experiments/kelly_regime_v10_mvrv_brake.py sweep     # step 3, inner-train
    python experiments/kelly_regime_v10_mvrv_brake.py select    # step 5, inner-validation, both markets
    python experiments/kelly_regime_v10_mvrv_brake.py signature # train-vs-validation overfitting check
    python experiments/kelly_regime_v10_mvrv_brake.py artifact  # exposure-artifact check
    python experiments/kelly_regime_v10_mvrv_brake.py volcorr   # MVRV-Z vs realized-vol correlation (R-08 trap)
    python experiments/kelly_regime_v10_mvrv_brake.py fallback  # pre-warmup exact-v4 check
    python experiments/kelly_regime_v10_mvrv_brake.py causality # two-opposite-tampers, price + mvrv
    python experiments/kelly_regime_v10_mvrv_brake.py eth       # ETH falsification test
    python experiments/kelly_regime_v10_mvrv_brake.py all       # everything above, in order
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
from tradebot.data import (  # noqa: E402
    load_coinbase_eth_spot,
    load_dataset,
    load_onchain,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

INCUMBENT = "kelly_regime_v4"


# --------------------------------------------------------------------- MVRV data


def _causal_mvrv_z(onchain: pd.DataFrame, window_days: int, smooth_days: float) -> pd.Series:
    """Causal rolling Z-score of the MVRV ratio, on the DAILY on-chain frame.

    ``window_days`` is a FULL calendar-day rolling window (min_periods
    equals the window -- no partial-window number is ever emitted).
    ``smooth_days`` causally EMA-smooths the resulting Z before it is
    returned (span in days; 1.0 effectively disables smoothing). Every
    operation here is ``.rolling`` / ``.ewm`` -- no full-series
    ``.mean()``/``.std()``.
    """
    mvrv = onchain["mvrv"]
    roll_mean = mvrv.rolling(window_days, min_periods=window_days).mean()
    roll_std = mvrv.rolling(window_days, min_periods=window_days).std()
    z = (mvrv - roll_mean) / roll_std.replace(0.0, np.nan)
    if smooth_days and smooth_days > 1.0:
        z = z.ewm(span=smooth_days, min_periods=1).mean()
    return z.rename("mvrv_z")


def _mvrv_z_on_index(spot_index: pd.DatetimeIndex, onchain: pd.DataFrame,
                      window_days: int, smooth_days: float) -> pd.Series:
    """Reindex the daily causal MVRV-Z onto the spot bar grid: strict as-of ffill.

    Mirrors ``kelly_regime_v9_basis_brake.py``'s ``_basis_on_index``
    convention exactly: never interpolated, never backfilled. Bars before
    the Z-score's own first non-NaN day remain NaN.
    """
    z = _causal_mvrv_z(onchain, window_days, smooth_days)
    aligned = (
        z.reindex(spot_index.union(z.index))
        .sort_index()
        .ffill()
        .reindex(spot_index)
    )
    first_valid = z.first_valid_index()
    if first_valid is None:
        return pd.Series(np.nan, index=spot_index)
    return aligned.where(spot_index >= first_valid)


# --------------------------------------------------------------------- strategy


class KellyRegimeV10MvrvBrake(Strategy):
    """v4's vote + conditional vol-targeting exposure, braked (never raised) by on-chain MVRV extremes.

    See module docstring for the full mechanism and pre-registration.
    Every v4-inherited parameter defaults exactly as in
    ``kelly_regime_v4``; ``lam``, ``window_days``, ``smooth_days``,
    ``lo_thresh`` and ``hi_thresh`` are the only new knobs. ``mvrv_z`` is
    injected (a ``pd.Series`` on the full spot index, precomputed once per
    ``(window_days, smooth_days)`` pair) purely for sweep speed -- it
    carries no strategy state and is read off ``df.index`` inside
    ``prepare`` exactly as ``kelly_regime_v9_basis_brake.py`` injects its
    basis series.
    """

    name = "kelly_regime_v10_mvrv_brake"
    # IMPORTANT: this must match v4's warmup, NOT the MVRV Z window length.
    # `engine.run_backtest` uses `strategy.warmup` as an absolute in-frame
    # bar-index gate on when `on_bar` is even called (`i >= strategy.warmup`),
    # not merely "how much prefix history to load" (that is
    # `window.run_period`'s separate, correct job). The vote/vol-targeting
    # components need exactly v4's 80-day in-frame warmup (computed from the
    # backtest frame's own price rolling windows). The MVRV Z-score needs NO
    # additional in-frame warmup: it is precomputed once from the FULL
    # 2010-2026 on-chain history (`_cached_z`, built before `prepare()` ever
    # slices anything), so it is already valid at bar 0 of any window
    # starting in 2017 or later -- see the module docstring's "Fallback /
    # warmup" section. Setting this to the MVRV window length (an earlier,
    # WRONG version of this file used 1095 days) would silently delay
    # `on_bar` -- not just the signal, the actual order issuance -- until
    # ~1095 days into whatever frame is passed, corrupting every backtest
    # uniformly regardless of the swept window/lam/threshold. Caught by
    # comparing full backtest results (not just the `target` column) at
    # lam=0 against v4 -- see `sweep()`'s correctness check.
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 lam: float = 0.5, window_days: int = 730, smooth_days: float = 7.0,
                 lo_thresh: float = 1.0, hi_thresh: float = 2.5,
                 mvrv_z: pd.Series | None = None) -> None:
        # ---- identical to kelly_regime / v3 / v4 -------------------------
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        # ---- new: the MVRV brake -------------------------------------------
        self.lam = lam                    # mult in [1-lam, 1]; 0 = exact v4
        self.window_days = window_days    # causal rolling Z window, calendar days
        self.smooth_days = smooth_days    # causal EMA span on the Z, in days
        self.lo_thresh = lo_thresh        # Z onset of the brake (HIGH side only)
        self.hi_thresh = hi_thresh        # Z where the brake reaches full lam
        self._mvrv_z = mvrv_z if mvrv_z is not None else _cached_z(window_days, smooth_days)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # ---- byte-for-byte v3/v4: latched multi-anchor vote -> frac ------
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()

        # ---- byte-for-byte v3/v4: conditional vol-targeting scale --------
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        # ---- new: the MVRV brake (asymmetric: HIGH mvrv-z only) ---------
        z = self._mvrv_z.reindex(df.index).to_numpy(dtype=float)
        denom = self.hi_thresh - self.lo_thresh
        excess = np.clip((z - self.lo_thresh) / denom, 0.0, 1.0) if denom > 0 else np.zeros_like(z)
        # Hard fallback: wherever the Z-score is unavailable (pre-warmup),
        # force excess=0 (mult=1) -- this is what makes the fallback exact.
        excess = np.where(np.isfinite(z), excess, 0.0)
        mult = 1.0 - self.lam * excess  # in [1-lam, 1], never > 1

        # ---- single causal forward pass: byte-for-byte v3/v4 breakout
        # hysteresis on the vol-targeting state, plus the new brake -------
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal vol band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * mult[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac"] = frac
        df["_mult"] = mult
        df["_excess"] = excess
        df["_mvrv_z"] = z
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
ONCHAIN_BTC = load_onchain(ROOT / "data", "BTC")
if ONCHAIN_BTC is None:
    raise RuntimeError("data/btcusd_onchain_daily.csv.gz not found -- cannot run this experiment")

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

# ---- sweep grid: fixed a-priori choices, not fit to inner-validation ----
# window_days spans 1/2/3-year causal lookbacks (see module docstring for
# why NOT an expanding-since-genesis window). smooth_days trades whipsaw
# against responsiveness on a still-mostly-price-driven daily ratio.
# lo/hi thresholds are Z-score units, picked from the unconditional
# BTC 2017-2022 Z distribution (roughly its 90th/97th percentiles at the
# tighter pair, 95th/99th at the wider one -- see `verify()`), never from
# a backtest result.
WINDOW_DAYS = (365, 730, 1095)
SMOOTH_DAYS = (1.0, 7.0, 30.0)
LAM = (0.3, 0.5, 0.7)
THRESH_PAIRS = ((1.0, 2.5), (1.5, 3.0))

N_EVALUATED = 0
_SEEN_CONFIGS: set[tuple] = set()
_Z_CACHE: dict[tuple, pd.Series] = {}

OUT = ROOT / "reports" / "kelly_regime_v10_mvrv_brake"


def _cached_z(window_days: int, smooth_days: float) -> pd.Series:
    key = (window_days, smooth_days)
    if key not in _Z_CACHE:
        _Z_CACHE[key] = _mvrv_z_on_index(DF.index, ONCHAIN_BTC, window_days, smooth_days)
    return _Z_CACHE[key]


def mean_notional(result) -> float:
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity) -> float:
    eq = equity.to_numpy(dtype=float) if hasattr(equity, "to_numpy") else np.asarray(equity)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(rets.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            count_key: tuple | None = None):
    global N_EVALUATED
    if count_key is not None and count_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(count_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def all_configs():
    for wd in WINDOW_DAYS:
        for sd in SMOOTH_DAYS:
            for lam in LAM:
                for lo, hi in THRESH_PAIRS:
                    yield wd, sd, lam, lo, hi


# --------------------------------------------------------------------------- verify


def verify() -> None:
    """Data verification: load, distribution, causal-availability, no-price-proxy checks."""
    print(f"BTC on-chain: {len(ONCHAIN_BTC):,} rows {ONCHAIN_BTC.index.min():%Y-%m-%d} -> "
          f"{ONCHAIN_BTC.index.max():%Y-%m-%d}")
    mvrv = ONCHAIN_BTC["mvrv"]
    print(f"mvrv: {mvrv.notna().sum():,} finite of {len(mvrv):,}; first finite "
          f"{mvrv.first_valid_index():%Y-%m-%d}")
    print("full-history describe:")
    print(mvrv.describe())
    print("\ntop 10 (note the 2010-07 genesis-era outliers -- see module docstring):")
    print(mvrv.sort_values(ascending=False).head(10))
    sub = mvrv.loc["2017-01-01":"2022-12-31"]
    print(f"\n2017-2022 describe (n={len(sub)}):")
    print(sub.describe())
    print("percentiles:")
    for p in (1, 5, 10, 50, 90, 95, 97, 99):
        print(f"  p{p:2d} = {sub.quantile(p / 100):.3f}")
    print("\ntop 5 (real cycle tops) / bottom 5 (real cycle bottoms), 2017-2022:")
    print(sub.sort_values(ascending=False).head(5))
    print(sub.sort_values().head(5))

    print("\ncausal-availability check: is a day's value available before the NEXT day's "
          "00:00 UTC? (loader already shifts +1 day; confirm by comparing to the raw file)")
    raw = pd.read_csv(ROOT / "data" / "btcusd_onchain_daily.csv.gz", parse_dates=["timestamp"])
    raw_ts0 = pd.Timestamp(raw["timestamp"].iloc[100])
    if raw_ts0.tzinfo is None:
        raw_ts0 = raw_ts0.tz_localize("UTC")
    else:
        raw_ts0 = raw_ts0.tz_convert("UTC")
    loaded_ts0 = ONCHAIN_BTC.index[100]
    print(f"  raw file row 100 timestamp: {raw_ts0}  ->  loaded index: {loaded_ts0}  "
          f"(shift = {loaded_ts0 - raw_ts0})")

    print("\nno silent price-proxy check: mvrv is NOT derivable from the spot OHLCV file --"
          " confirm columns present are exactly mvrv/active_addresses/supply, no open/high/low/close")
    print(f"  columns: {list(ONCHAIN_BTC.columns)}")

    z = _causal_mvrv_z(ONCHAIN_BTC, 730, 7.0)
    print(f"\nexample causal Z (window=730d, smooth=7d): first finite "
          f"{z.first_valid_index()}, describe on 2017-2022:")
    print(z.loc["2017-01-01":"2022-12-31"].describe())


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Step 3: every (window_days, smooth_days, lam, lo, hi) config on inner-train, spot."""
    rows = []
    t0 = time.time()
    for wd, sd, lam, lo, hi in all_configs():
        key = (wd, sd, lam, lo, hi)
        strat = KellyRegimeV10MvrvBrake(window_days=wd, smooth_days=sd, lam=lam,
                                         lo_thresh=lo, hi_thresh=hi)
        m, vol, notional, res = measure(strat, *TRAIN, market=SPOT, count_key=key)
        rows.append({"window_days": wd, "smooth_days": sd, "lam": lam, "lo": lo, "hi": hi,
                     "market": "spot", "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated})
        print(f"[{N_EVALUATED:>3d}] wd={wd:4d}d sd={sd:5.1f}d lam={lam:.1f} lo={lo:.2f} hi={hi:.2f}  "
              f"final=${m.final_balance:>9,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} "
              f"notional={notional:.3f} [{time.time() - t0:.0f}s]")
    # lam=0 correctness check: must reduce to v4 bit-for-bit -- BOTH the
    # prepared target array AND the actual executed backtest result (final
    # balance / trades), not just the array (an earlier, wrong version of
    # this file passed the array check while a warmup bug silently deleted
    # ~3 years of trading from every config -- the array comparison alone
    # did not catch it, because `target` is a pure `prepare()`-time
    # artifact unaffected by whether `on_bar` was actually invoked).
    zero = KellyRegimeV10MvrvBrake(lam=0.0)
    m0, vol0, not0, res0 = measure(zero, *TRAIN, market=SPOT, count_key=("lam0-correctness",))
    v4 = get_strategy(INCUMBENT)
    m4, vol4, not4, res4 = measure(v4, *TRAIN, market=SPOT)
    diff = float(np.max(np.abs(res0.df["target"].to_numpy() - res4.df["target"].reindex(res0.df.index).to_numpy())))
    bal_diff = abs(m0.final_balance - m4.final_balance)
    trades_match = m0.num_trades == m4.num_trades
    print(f"\nlam=0 correctness check (max|target diff| vs v4): {diff:.3e}  "
          f"{'PASS' if diff < 1e-9 else 'FAIL'}")
    print(f"lam=0 EXECUTED-backtest check: final ${m0.final_balance:,.2f} vs v4 ${m4.final_balance:,.2f} "
          f"(diff ${bal_diff:.4f}), trades {m0.num_trades} vs {m4.num_trades}  "
          f"{'PASS' if bal_diff < 1e-2 and trades_match else 'FAIL'}")
    print(f"v4 control (train):  final=${m4.final_balance:>9,.0f} DD={m4.max_drawdown_pct:>5.1f}% "
          f"sharpe={m4.sharpe:>5.2f} trades={m4.num_trades:>4d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return out


# --------------------------------------------------------------------------- step 5


def select() -> pd.DataFrame:
    """Step 5: every config on inner-validation, BOTH markets, vs v4 control."""
    rows = []
    for wd, sd, lam, lo, hi in all_configs():
        strat_kwargs = dict(window_days=wd, smooth_days=sd, lam=lam, lo_thresh=lo, hi_thresh=hi)
        for mname, market in MARKETS:
            strat = KellyRegimeV10MvrvBrake(**strat_kwargs)
            m, vol, notional, res = measure(strat, *VALID, market=market)
            rows.append({"window_days": wd, "smooth_days": sd, "lam": lam, "lo": lo, "hi": hi,
                         "market": mname, "final": m.final_balance, "vol": vol,
                         "notional": notional, "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated})
        s = rows[-2]
        f = rows[-1]
        print(f"wd={wd:4d}d sd={sd:5.1f}d lam={lam:.1f} lo={lo:.2f} hi={hi:.2f}  "
              f"spot: ${s['final']:>9,.0f} DD{s['max_dd']:>5.1f}% sh{s['sharpe']:>5.2f} tr{s['trades']:>4d}   "
              f"fut: ${f['final']:>9,.0f} DD{f['max_dd']:>5.1f}% sh{f['sharpe']:>5.2f} tr{f['trades']:>4d}")
    for mname, market in MARKETS:
        m, vol, notional, res = measure(get_strategy(INCUMBENT), *VALID, market=market)
        rows.append({"window_days": None, "smooth_days": None, "lam": None, "lo": None, "hi": None,
                     "market": mname, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated,
                     "label": "kelly_regime_v4_control"})
    ctl_s = rows[-2]
    ctl_f = rows[-1]
    print(f"{'kelly_regime_v4 (control)':30s} spot: ${ctl_s['final']:>9,.0f} "
          f"DD{ctl_s['max_dd']:>5.1f}% sh{ctl_s['sharpe']:>5.2f} tr{ctl_s['trades']:>4d}   "
          f"fut: ${ctl_f['final']:>9,.0f} DD{ctl_f['max_dd']:>5.1f}% "
          f"sh{ctl_f['sharpe']:>5.2f} tr{ctl_f['trades']:>4d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")
    return out


def train_vs_valid_signature(wd: int, sd: float, lam: float, lo: float, hi: float) -> None:
    """Diagnostic 2: the R-37/R-38/R-40 overfitting signature check, for one named candidate."""
    strat_kwargs = dict(window_days=wd, smooth_days=sd, lam=lam, lo_thresh=lo, hi_thresh=hi)
    print(f"\n=== overfitting-signature check: wd={wd}d sd={sd}d lam={lam} lo={lo} hi={hi} ===")
    for wname, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            cand = KellyRegimeV10MvrvBrake(**strat_kwargs)
            m_c, vol_c, not_c, res_c = measure(cand, start, end, market=market)
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), start, end, market=market)
            beats = "beats v4" if m_c.final_balance > m_v4.final_balance else "LOSES to v4"
            print(f"  {wname:18s} {mname:8s} cand=${m_c.final_balance:>9,.0f} "
                  f"(DD{m_c.max_drawdown_pct:5.1f}% sh{m_c.sharpe:5.2f} tr{m_c.num_trades:4d})  "
                  f"v4=${m_v4.final_balance:>9,.0f} (DD{m_v4.max_drawdown_pct:5.1f}% "
                  f"sh{m_v4.sharpe:5.2f} tr{m_v4.num_trades:4d})   [{beats}]")


# ------------------------------------------------------------------------ diagnostic 1


def exposure_artifact_check() -> None:
    """Diagnostic 1: mandatory exposure-artifact check (R-33/R-34's standing threshold).

    Mean-notional-matched flat rescale of v4's own target, R^2 against
    the candidate's target, inner-validation, both markets.
    """
    print("\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4):")
    for wd, sd, lam, lo, hi in all_configs():
        print(f" wd={wd:4d}d sd={sd:5.1f}d lam={lam:.1f} lo={lo:.2f} hi={hi:.2f}:")
        for mname, market in MARKETS:
            cand = KellyRegimeV10MvrvBrake(window_days=wd, smooth_days=sd, lam=lam,
                                            lo_thresh=lo, hi_thresh=hi)
            m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
            v4 = get_strategy(INCUMBENT)
            m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=market)

            cand_t = res_c.df["target"].to_numpy(dtype=float)
            v4_t = res_v4.df["target"].reindex(res_c.df.index).to_numpy(dtype=float)
            c = not_c / not_v4 if not_v4 > 0 else float("nan")
            flat = c * v4_t

            mask = np.isfinite(cand_t) & np.isfinite(flat)
            x = flat[mask]
            y = cand_t[mask]
            ss_res = float(np.sum((y - x) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

            verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > 0.95
                        else "not a flat rescale by this test")
            print(f"    {mname}: cand notional={not_c:.3f} v4 notional={not_v4:.3f} c={c:.3f}  "
                  f"corr={corr:.4f}  R^2={r2:.4f}  {verdict}")


# ------------------------------------------------------------------------ R-08 trap check


def volcorr() -> None:
    """MVRV-Z vs realized volatility correlation -- is this secretly a vol timer? (R-08 / B-07 trap)."""
    print("\nR-08 trap check: correlation between MVRV-Z (and its brake 'excess') and realized volatility")
    for wname, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        df = DF.loc[start:end]
        r = np.log(df["close"]).diff()
        trailing_vol = (r.rolling(30 * BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
                         * np.sqrt(BARS_PER_YEAR))
        forward_vol = trailing_vol.shift(-30 * BARS_PER_DAY)  # descriptive only, not used by the strategy
        for wd, sd in ((365, 7.0), (730, 7.0), (1095, 7.0)):
            z = _cached_z(wd, sd).reindex(df.index)
            joined = pd.concat([z.rename("z"), trailing_vol.rename("trail_vol"),
                                 forward_vol.rename("fwd_vol")], axis=1).dropna()
            if len(joined) < 100:
                print(f"  {wname:18s} wd={wd}d: insufficient overlap, skipping")
                continue
            r_trail = joined["z"].corr(joined["trail_vol"])
            r_fwd = joined["z"].corr(joined["fwd_vol"])
            print(f"  {wname:18s} wd={wd}d: corr(Z, trailing 30d vol)={r_trail:+.3f}  "
                  f"corr(Z, forward 30d vol)={r_fwd:+.3f}  n={len(joined):,}")


# ------------------------------------------------------------------------ fallback check


def fallback_check() -> None:
    """Hard constraint check: pre-warmup (Z-score NaN), candidate must equal v4 EXACTLY.

    The committed spot dataset starts 2017-01-01. For every swept window
    (up to 1095 days), the causal MVRV-Z is ALREADY valid by then (longest
    window's first-valid date is ~2013-07-18 -- see `verify()`), so there
    is, honestly, NO live pre-warmup bar inside the actual backtest data
    for BTC -- exactly the module docstring's stated design note, checked
    here rather than assumed. That is reported plainly first. A second,
    synthetic-index check then verifies the FALLBACK LOGIC ITSELF is
    correct (using the real on-chain frame's own pre-2013 dates, which DO
    exist, paired with a synthetic constant-price series -- sufficient to
    exercise `_mvrv_z_on_index` + the strategy's `excess`/`mult` fallback
    without needing real pre-2017 spot bars, which this project does not
    have).
    """
    # Use the RAW daily Z's own first-valid date, not the spot-reindexed
    # cache (which is truncated to DF's own 2017-01-01 start and would
    # misreport the true warmup-completion date as 2017-01-01 regardless
    # of the window).
    z_daily = _causal_mvrv_z(ONCHAIN_BTC, 1095, 7.0)
    first_valid = z_daily.first_valid_index()
    pre_coverage_end = first_valid - pd.Timedelta(days=1)
    df = DF.loc[:pre_coverage_end].copy()
    print(f"\nfallback check: spot dataset starts {DF.index[0]:%Y-%m-%d}; the 1095d-window "
          f"MVRV-Z first becomes valid {first_valid:%Y-%m-%d} -- "
          f"{len(df):,} bars in the spot dataset strictly precede that date.")
    if len(df) == 0:
        print("  CONFIRMED: no pre-warmup bars exist in the committed spot dataset for ANY "
              "swept window (longest is 1095d) -- the fallback never fires live during any "
              "measured BTC window. Running a synthetic-index logic check instead:")
        synth_index = pd.date_range("2011-06-01", "2012-06-01", freq="5min", tz="UTC")
        synth = pd.DataFrame({
            "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1.0,
        }, index=synth_index)
        z_synth = _mvrv_z_on_index(synth_index, ONCHAIN_BTC, 1095, 7.0)
        n_nan = int(z_synth.isna().sum())
        print(f"  synthetic index 2011-06-01..2012-06-01 ({len(synth_index):,} bars, real "
              f"on-chain data, well before the 1095d window's {first_valid:%Y-%m-%d} first-valid "
              f"date): {n_nan:,} / {len(z_synth):,} bars have NaN MVRV-Z, as expected  "
              f"{'PASS' if n_nan == len(z_synth) else 'FAIL'}")
        cand = KellyRegimeV10MvrvBrake(lam=0.7, window_days=1095, smooth_days=1.0,
                                        lo_thresh=1.0, hi_thresh=2.5, mvrv_z=z_synth)
        prepared = cand.prepare(synth.copy())
        excess_nonzero = int(np.sum(prepared["_excess"].to_numpy() != 0.0))
        mult_not_one = int(np.sum(prepared["_mult"].to_numpy() != 1.0))
        print(f"  bars with nonzero _excess (should be 0): {excess_nonzero}; "
              f"bars with _mult != 1.0 (should be 0): {mult_not_one}  "
              f"{'PASS -- fallback forces mult=1' if excess_nonzero == 0 and mult_not_one == 0 else 'FAIL'}")
        return
    cand = KellyRegimeV10MvrvBrake(lam=0.7, window_days=1095, smooth_days=1.0,
                                    lo_thresh=1.0, hi_thresh=2.5)
    cand_prepared = cand.prepare(df.copy())
    v4 = get_strategy(INCUMBENT)
    v4_prepared = v4.prepare(df.copy())
    diff = float(np.max(np.abs(
        cand_prepared["target"].to_numpy() - v4_prepared["target"].to_numpy())))
    excess_nonzero = int(np.sum(cand_prepared["_excess"].to_numpy() != 0.0))
    print(f"  max|target diff| vs v4 over the entire pre-warmup region = {diff:.3e}  "
          f"{'PASS -- exact v4 fallback' if diff < 1e-12 else 'FAIL'}")
    print(f"  bars with nonzero _excess (should be 0): {excess_nonzero}")


# ------------------------------------------------------------------------ diagnostic / causality


PRIMARY = dict(window_days=365, smooth_days=7.0, lam=0.5, lo_thresh=1.5, hi_thresh=3.0)


def causality() -> None:
    """Two-opposite-tampers, on PRICE and, separately, on the MVRV-Z series."""
    pre_2023 = DF.loc[:"2022-12-31"]
    df = pre_2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    print("=== price tamper probe ===")
    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def prepared(frame):
        return KellyRegimeV10MvrvBrake(**PRIMARY).prepare(frame.copy())

    pa = prepared(up)
    pb = prepared(down)
    ok = True
    for col in ("target", "_frac", "_mult", "_excess"):
        a = pa[col].to_numpy(dtype=float)[:cut]
        b = pb[col].to_numpy(dtype=float)[:cut]
        worst = float(np.nanmax(np.abs(a - b)))
        good = worst < 1e-9
        ok &= good
        print(f"  column={col:16s} max |difference| before the cut = {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    from tradebot.broker import PaperBroker
    from tradebot.orders import Order

    def decisions(frame):
        s = KellyRegimeV10MvrvBrake(**PRIMARY)
        prep = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        broker.execute(Order(target=0.1), prep.index[0], float(prep["open"].iloc[0]))
        out = []
        for i in bars:
            ctx = Context(prep, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    ok &= not bad
    print(f"  orders {'match' if not bad else f'DIFFER at bars {bad}'} at the probe bars")

    a = run_backtest(KellyRegimeV10MvrvBrake(**PRIMARY), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV10MvrvBrake(**PRIMARY), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")
    print(f"  tampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS' if ok else 'FAIL'} -- no price-dependent decision at or before the cut moves")

    print("\n=== MVRV-Z tamper probe (new to this file) ===")
    z_base = _cached_z(PRIMARY["window_days"], PRIMARY["smooth_days"])
    z_up = z_base.copy()
    z_down = z_base.copy()
    cut_ts = df.index[cut]
    mask_after = z_up.index >= cut_ts
    z_up.loc[mask_after] = z_up.loc[mask_after] * 3.0 + 5.0  # push up, away from zero either sign
    z_down.loc[mask_after] = z_down.loc[mask_after] / 3.0 - 5.0

    strat_up = KellyRegimeV10MvrvBrake(**PRIMARY, mvrv_z=z_up)
    strat_down = KellyRegimeV10MvrvBrake(**PRIMARY, mvrv_z=z_down)
    pu = strat_up.prepare(df.copy())
    pdn = strat_down.prepare(df.copy())
    ok2 = True
    for col in ("target", "_mult", "_excess", "_mvrv_z"):
        a2 = pu[col].to_numpy(dtype=float)[:cut]
        b2 = pdn[col].to_numpy(dtype=float)[:cut]
        worst2 = float(np.nanmax(np.abs(a2 - b2)))
        good2 = worst2 < 1e-9
        ok2 &= good2
        print(f"  column={col:16s} max |difference| before the cut = {worst2:.3e}  "
              f"{'PASS' if good2 else 'FAIL'}")
    print(f"  MVRV-tamper probe: {'PASS' if ok2 else 'FAIL'} -- "
          f"no MVRV-dependent decision at or before the cut moves when ONLY the "
          f"MVRV-Z series (not price) is tampered after the cut")


# ------------------------------------------------------------------------ ETH falsification


def eth() -> None:
    """The pre-registered falsification test: primary candidate vs v4, ETH, 2019-03-14..2022-12-31."""
    eth_price = load_coinbase_eth_spot(ROOT / "data")
    eth_onchain = load_onchain(ROOT / "data", "ETH")
    if eth_price is None or eth_onchain is None:
        print("ETH data not available -- cannot run falsification test")
        return
    print(f"ETH price (Coinbase spot): {len(eth_price):,} bars "
          f"{eth_price.index[0]:%Y-%m-%d} -> {eth_price.index[-1]:%Y-%m-%d}")
    print(f"ETH on-chain: {len(eth_onchain):,} rows "
          f"{eth_onchain.index[0]:%Y-%m-%d} -> {eth_onchain.index[-1]:%Y-%m-%d}, "
          f"mvrv first finite {eth_onchain['mvrv'].first_valid_index():%Y-%m-%d}")
    start, end = "2019-03-14", "2022-12-31"
    print(f"overlap window used (pre-holdout only): {start} -> {end}")

    z = _mvrv_z_on_index(eth_price.index, eth_onchain, PRIMARY["window_days"], PRIMARY["smooth_days"])
    cand = KellyRegimeV10MvrvBrake(**PRIMARY, mvrv_z=z)
    v4 = get_strategy(INCUMBENT)

    eth_spot_mkt = MarketSpec.spot()
    eth_fut_mkt = MarketSpec.futures(leverage=5.0)
    for mname, market in (("spot", eth_spot_mkt), ("futures", eth_fut_mkt)):
        res_c = run_period(cand, eth_price, start, end, market=market,
                            start_balance=1_000.0, data_label="ETH coinbase spot (real)")
        res_v4 = run_period(v4, eth_price, start, end, market=market,
                             start_balance=1_000.0, data_label="ETH coinbase spot (real)")
        m_c, m_v4 = compute_metrics(res_c), compute_metrics(res_v4)
        beats = "beats v4 -- SURVIVES" if m_c.final_balance > m_v4.final_balance else "LOSES to v4 -- FALSIFIED"
        print(f"  ETH {mname:8s} cand=${m_c.final_balance:>9,.0f} "
              f"(DD{m_c.max_drawdown_pct:5.1f}% sh{m_c.sharpe:5.2f} tr{m_c.num_trades:4d})  "
              f"v4=${m_v4.final_balance:>9,.0f} (DD{m_v4.max_drawdown_pct:5.1f}% "
              f"sh{m_v4.sharpe:5.2f} tr{m_v4.num_trades:4d})   [{beats}]")


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"spot: {len(DF):,} bars {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d} (data: {LABEL})",
          file=sys.stderr)
    print(f"on-chain BTC: {len(ONCHAIN_BTC):,} rows {ONCHAIN_BTC.index[0]:%Y-%m-%d} -> "
          f"{ONCHAIN_BTC.index[-1]:%Y-%m-%d} (CoinMetrics, real)", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "verify":
        verify()
    elif choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "signature":
        train_vs_valid_signature(PRIMARY["window_days"], PRIMARY["smooth_days"], PRIMARY["lam"],
                                  PRIMARY["lo_thresh"], PRIMARY["hi_thresh"])
    elif choice == "artifact":
        exposure_artifact_check()
    elif choice == "volcorr":
        volcorr()
    elif choice == "fallback":
        fallback_check()
    elif choice == "causality":
        causality()
    elif choice == "eth":
        eth()
    elif choice == "all":
        verify()
        sweep()
        select()
        train_vs_valid_signature(PRIMARY["window_days"], PRIMARY["smooth_days"], PRIMARY["lam"],
                                  PRIMARY["lo_thresh"], PRIMARY["hi_thresh"])
        exposure_artifact_check()
        volcorr()
        fallback_check()
        causality()
        eth()
    else:
        print("usage: python experiments/kelly_regime_v10_mvrv_brake.py "
              "[verify|sweep|select|signature|artifact|volcorr|fallback|causality|eth|all]")
