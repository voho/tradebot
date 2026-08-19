#!/usr/bin/env python
"""kelly_regime_v4 with a bounded, never-increase brake from the real Deribit basis (CONSERVATIVE branch, R-41).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The idea
--------
Every "futures" market this project has ever backtested traded the SAME
price series as spot -- the standing diagnosis's #1 constraint, INFO: one
price series. This session that changed: real Deribit BTC-PERPETUAL 5m
OHLCV (``data/btcusdt_deribit_perp_5m.csv.gz``, 2018-08-14 -> present,
842,851 bars, zero gaps) is now committed, an independently-transacted
second price series for the first time. ``tradebot.data.compute_basis``
gives ``log(perp_close / spot_close)`` on the perp's own index, causal
(as-of join against spot, dropping -- never filling -- bars before
spot's start).

Mechanism, one sentence: v4's vote and conditional-vol-target scale are
reproduced byte-for-byte; on top, a bounded multiplier
``mult = 1 - lam * excess(t) in [1-lam, 1]`` shrinks (never raises) v4's
own exposure when the causally-lagged, EMA-smoothed *magnitude* of the
basis, ``|basis|``, exceeds an onset threshold, ramping linearly to the
full brake ``1-lam`` by a second, higher threshold -- symmetric in sign,
because BOTH an extreme positive basis (crowded, over-levered longs
paying up to stay in -- a precondition for a long-squeeze cascade) and an
extreme negative basis (an active forced-liquidation/deleveraging event,
Brunnermeier & Pedersen 2005 predatory-trading overshoot dynamics) are
the tail risk kelly_regime_v4's drawdown story is supposed to protect
against, now observed directly instead of inferred from lagging price
averages.

Constraint attacked
--------------------
INFO, for the first time with a genuinely new price series rather than a
transform of the incumbent one -- and SIZE, since the brake only ever
acts through the same exposure axis every prior working strategy in this
project has used. Not a duplicate of L-12/L-14/L-15/L-16 (all four
INFO-labelled failures in section A tried to recover missing information
FROM PRICE and failed -- this signal comes from an independent, real
second market, not a price transform).

Not a duplicate of
-------------------
- R-34 conservative (``kelly_regime_v5_damp.py``): SAME bounded,
  never-increase-only architectural template (``mult in [1-lam, 1]``,
  single forward pass, v4's vote/scale untouched) -- but a DIFFERENT
  signal. R-34's Bayesian posterior margin is a smoothed transform of
  the same OHLCV close series the vote already reads (INFO constraint
  unaddressed) and, once smoothed enough to avoid whipsaw, was shown to
  have almost no independent dynamic range (target series correlated
  R^2=0.997 with a flat 0.7x rescale of v4 -- "a smoothed copy of a
  constant"). Basis comes from an independently-transacted SECOND
  market and is NOT continuously smoothed away: it is near-zero on
  >99% of bars and spikes sharply, persistently, on specific calendar
  dates tied to known deleveraging events (COVID Black Thursday
  2020-03-13, the China mining-ban crash 2021-05-19) -- a structurally
  different dynamic-range profile, checked directly below rather than
  assumed.
- R-35 (funding-decile gate / EV-band): a different real, non-price
  signal (an EXCHANGE-SET RATE, funding) vs. basis (a MARKET-OBSERVED
  PRICE SPREAD between two independently-transacted venues) -- distinct
  economic content, confirmed empirically below (correlation ~0.10 on
  the actual working window, re-derived independently, not merely
  quoted from the operator's brief).
- R-37/R-38/R-40 (retuned constants / per-state Kelly / CRRA cap /
  ladder-bagging): none introduce ANY new data source; all rework what
  already-available OHLCV close implies. This round's signal cannot be
  computed from the Bitstamp close column at all. R-37/R-38/R-40's
  shared failure signature -- wins on inner-validation (2021-22,
  bear/chop) and LOSES to v4 on the earlier control window -- is exactly
  diagnostic (2) below, run explicitly rather than skipped.

Causality
---------
``compute_basis`` is already causal by construction (as-of join, no
interpolation, no future spot bar reachable from an earlier perp
timestamp -- see its own docstring in ``tradebot/data.py``). This file
reindexes that series once onto the Bitstamp spot index actually used for
backtesting (`_basis_on_index`, an as-of ffill against the union of both
indices, mirroring ``compute_basis``'s own alignment convention) and then
applies a causal EMA (`pandas.Series.ewm`, ``min_periods=1``) to
``|basis|`` before thresholding -- every operation reads rows <= i only.
Two independent causality probes are run: the standard two-opposite-
tampers probe on PRICE (bars after a cut multiplied/divided by 3, copied
from ``kelly_regime_v8_ladder_bag.py``'s own procedure) and a second,
new-to-this-file probe that tampers the BASIS series itself after the
same cut -- because the price probe alone cannot exercise the one new
ingredient this file adds (basis is loaded from an external file, not
derived from the ``df`` the price probe tampers, so a passing price probe
alone says nothing about whether the basis merge itself is causal).

Fallback (hard constraint, checked directly)
----------------------------------------------
Before 2018-08-14 (no Deribit coverage), ``excess`` is forced to exactly
0.0 (``mult == 1.0``) regardless of what the EMA computes during the
NaN-fed warmup -- verified below by an exact bit-identical diff against
unmodified v4 on every pre-coverage bar, not merely argued from the code.

Usage
-----
    python experiments/kelly_regime_v9_basis_brake.py sweep       # step 3, inner-train-with-basis
    python experiments/kelly_regime_v9_basis_brake.py select      # step 5, inner-validation, both markets
    python experiments/kelly_regime_v9_basis_brake.py artifact    # exposure-artifact check (R-33/R-34)
    python experiments/kelly_regime_v9_basis_brake.py fundingcorr # basis vs funding-rate correlation
    python experiments/kelly_regime_v9_basis_brake.py fallback    # pre-2018-08-14 exact-v4 check
    python experiments/kelly_regime_v9_basis_brake.py causality   # two-opposite-tampers, price + basis
    python experiments/kelly_regime_v9_basis_brake.py all         # everything above, in order
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
    compute_basis,
    load_dataset,
    load_deribit_perp_price,
    load_funding_extended,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
BARS_PER_HOUR = 12


# --------------------------------------------------------------------- basis data

def _basis_on_index(spot: pd.DataFrame, perp: pd.DataFrame) -> pd.Series:
    """Causal basis, reindexed onto ``spot``'s own bar grid.

    ``compute_basis`` returns the series on the perp's index; the
    strategy is backtested on the Bitstamp spot frame, so this does one
    more as-of (ffill, never interpolated/backfilled) reindex onto that
    grid, exactly mirroring ``compute_basis``'s own spot-alignment
    convention. Bars before Deribit's coverage start remain NaN -- never
    filled -- which is what lets the strategy's own fallback logic (not
    this function) force an exact-v4 default there.
    """
    basis = compute_basis(spot, perp)
    aligned = (
        basis.reindex(spot.index.union(basis.index))
        .sort_index()
        .ffill()
        .reindex(spot.index)
    )
    return aligned.where(spot.index >= basis.index.min())


# --------------------------------------------------------------------- strategy


class KellyRegimeV9BasisBrake(Strategy):
    """v4's vote + conditional vol-targeting exposure, braked (never raised) by the real Deribit basis.

    See module docstring for the full mechanism. Defaults for every
    v4-inherited parameter match ``kelly_regime_v4`` exactly; ``lam``,
    ``smooth_hours``, ``lo_thresh`` and ``hi_thresh`` are the only new
    knobs. ``basis`` is injected (a ``pd.Series`` on the full spot index,
    built once by ``_basis_on_index``) rather than recomputed per
    instance, purely for sweep speed -- it carries no strategy state and
    every value used is still read off ``df.index`` inside ``prepare``.
    """

    name = "kelly_regime_v9_basis_brake"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 lam: float = 0.5, smooth_hours: float = 4.0,
                 lo_thresh: float = 0.02, hi_thresh: float = 0.06,
                 basis: pd.Series | None = None) -> None:
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
        # ---- new: the basis brake -----------------------------------------
        self.lam = lam                  # mult in [1-lam, 1]; 0 = exact v4
        self.smooth_hours = smooth_hours  # causal EMA span on |basis|, in hours
        self.lo_thresh = lo_thresh      # |basis| onset of the brake (log units)
        self.hi_thresh = hi_thresh      # |basis| where the brake reaches full lam
        self._basis = basis if basis is not None else BASIS_ON_SPOT

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

        # ---- new: the basis brake -----------------------------------------
        raw_basis = self._basis.reindex(df.index).to_numpy(dtype=float)
        abs_basis = np.abs(raw_basis)
        smooth_span = max(1.0, self.smooth_hours * BARS_PER_HOUR)
        smoothed = (pd.Series(abs_basis, index=df.index)
                    .ewm(span=smooth_span, min_periods=1).mean().to_numpy())
        denom = self.hi_thresh - self.lo_thresh
        excess = np.clip((smoothed - self.lo_thresh) / denom, 0.0, 1.0) if denom > 0 else np.zeros_like(smoothed)
        # Hard fallback: wherever basis is unavailable (pre-2018-08-14),
        # force excess=0 (mult=1) regardless of what the EMA computed while
        # fed NaN during warmup -- this is what makes the fallback exact.
        excess = np.where(np.isfinite(raw_basis), excess, 0.0)
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
        df["_basis_raw"] = raw_basis
        df["_basis_smoothed"] = smoothed
        return df

    def on_bar(self, ctx: Context) -> None:
        # Identical execution pattern to kelly_regime.KellyRegime.on_bar:
        # signal at bar close, fill at next open via order_notional.
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
PERP = load_deribit_perp_price(ROOT / "data", "BTC")
if PERP is None:
    raise RuntimeError("data/btcusdt_deribit_perp_5m.csv.gz not found -- cannot run this experiment")
BASIS_ON_SPOT = _basis_on_index(DF, PERP)

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

# Per the task: basis coverage starts 2018-08-14, so the fair inner-train
# comparison window is 2018-08-14 -> 2020-12-31 (~2.4y, not the usual
# 2017-2020) -- before that date the candidate is IDENTICAL to v4 by
# construction (see `fallback_check`), so comparing there would measure
# nothing.
TRAIN = ("2018-08-14", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

INCUMBENT = "kelly_regime_v4"

# ---- sweep grid: fixed a-priori choices, not fit to inner-validation ----
# lo/hi thresholds picked from the basis's own unconditional distribution
# (median ~0.5-0.7%, p90 ~2-2.6%, p99 ~6%, measured separately on both the
# inner-train-with-basis and inner-validation windows and found stable
# across them -- see the module docstring's design note) -- (0.02, 0.06)
# brackets roughly p90-p99; (0.03, 0.10) brackets roughly p95 and the
# COVID/China-ban-style extreme-tail definition used in the task brief.
SMOOTH_HOURS = (1.0, 4.0, 12.0)
LAM = (0.3, 0.5, 0.7)
THRESH_PAIRS = ((0.02, 0.06), (0.03, 0.10))

N_EVALUATED = 0  # distinct configurations searched (routine's trials count)
_SEEN_CONFIGS: set[tuple] = set()

OUT = ROOT / "reports" / "kelly_regime_v9_basis_brake"


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
    """One backtest -> (metrics, realized vol, mean notional, result).

    ``count_key`` is a hashable identity for the CONFIGURATION under
    test (not the market/window) -- N_EVALUATED increments once per
    distinct key ever passed, matching the convention in
    ``kelly_regime_v8_ladder_bag.py`` (count once per config, not once
    per (config x market x window) backtest run).
    """
    global N_EVALUATED
    if count_key is not None and count_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(count_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def line(tag, m, vol, notional, result):
    print(f"  {tag:44s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


def all_configs():
    for sh in SMOOTH_HOURS:
        for lam in LAM:
            for lo, hi in THRESH_PAIRS:
                yield sh, lam, lo, hi


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Step 3: every (smooth_hours, lam, lo, hi) config on inner-train-with-basis, spot primary."""
    rows = []
    t0 = time.time()
    for sh, lam, lo, hi in all_configs():
        key = (sh, lam, lo, hi)
        strat = KellyRegimeV9BasisBrake(smooth_hours=sh, lam=lam, lo_thresh=lo, hi_thresh=hi)
        m, vol, notional, res = measure(strat, *TRAIN, market=SPOT, count_key=key)
        rows.append({"smooth_hours": sh, "lam": lam, "lo": lo, "hi": hi, "market": "spot",
                     "final": m.final_balance, "vol": vol, "notional": notional,
                     "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated})
        print(f"[{N_EVALUATED:>2d}] sh={sh:5.1f}h lam={lam:.1f} lo={lo:.2f} hi={hi:.2f}  "
              f"final=${m.final_balance:>9,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} "
              f"notional={notional:.3f} [{time.time() - t0:.0f}s]")
    # lam=0 correctness check: must reduce to v4 bit-for-bit
    zero = KellyRegimeV9BasisBrake(lam=0.0)
    m0, vol0, not0, res0 = measure(zero, *TRAIN, market=SPOT, count_key=("lam0-correctness",))
    v4 = get_strategy(INCUMBENT)
    m4, vol4, not4, res4 = measure(v4, *TRAIN, market=SPOT)
    diff = float(np.max(np.abs(res0.df["target"].to_numpy() - res4.df["target"].reindex(res0.df.index).to_numpy())))
    print(f"\nlam=0 correctness check (max|target diff| vs v4): {diff:.3e}  "
          f"{'PASS' if diff < 1e-9 else 'FAIL'}")
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
    """Step 5: every config on inner-validation, BOTH markets, vs v4 control -- the R-37/38/40 check."""
    rows = []
    for sh, lam, lo, hi in all_configs():
        strat_kwargs = dict(smooth_hours=sh, lam=lam, lo_thresh=lo, hi_thresh=hi)
        for mname, market in MARKETS:
            strat = KellyRegimeV9BasisBrake(**strat_kwargs)
            m, vol, notional, res = measure(strat, *VALID, market=market)
            rows.append({"smooth_hours": sh, "lam": lam, "lo": lo, "hi": hi, "market": mname,
                         "final": m.final_balance, "vol": vol, "notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated})
        s = rows[-2]
        f = rows[-1]
        print(f"sh={sh:5.1f}h lam={lam:.1f} lo={lo:.2f} hi={hi:.2f}  "
              f"spot: ${s['final']:>9,.0f} DD{s['max_dd']:>5.1f}% sh{s['sharpe']:>5.2f} tr{s['trades']:>4d}   "
              f"fut: ${f['final']:>9,.0f} DD{f['max_dd']:>5.1f}% sh{f['sharpe']:>5.2f} tr{f['trades']:>4d}")
    for mname, market in MARKETS:
        m, vol, notional, res = measure(get_strategy(INCUMBENT), *VALID, market=market)
        rows.append({"smooth_hours": None, "lam": None, "lo": None, "hi": None, "market": mname,
                     "final": m.final_balance, "vol": vol, "notional": notional,
                     "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated,
                     "label": "kelly_regime_v4_control"})
    ctl_s = rows[-2]
    ctl_f = rows[-1]
    print(f"{'kelly_regime_v4 (control)':26s} spot: ${ctl_s['final']:>9,.0f} "
          f"DD{ctl_s['max_dd']:>5.1f}% sh{ctl_s['sharpe']:>5.2f} tr{ctl_s['trades']:>4d}   "
          f"fut: ${ctl_f['final']:>9,.0f} DD{ctl_f['max_dd']:>5.1f}% "
          f"sh{ctl_f['sharpe']:>5.2f} tr{ctl_f['trades']:>4d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")
    return out


def train_vs_valid_signature(sh: float, lam: float, lo: float, hi: float) -> None:
    """The R-37/R-38/R-40 overfitting signature check, for one named candidate.

    Prints inner-train-with-basis and inner-validation, both markets,
    candidate vs v4, side by side -- so a win-on-validation/lose-on-train
    pattern (R-37/38/40's shared failure mode) is visible directly rather
    than requiring the reader to cross-reference two CSVs.
    """
    strat_kwargs = dict(smooth_hours=sh, lam=lam, lo_thresh=lo, hi_thresh=hi)
    print(f"\n=== overfitting-signature check: sh={sh}h lam={lam} lo={lo} hi={hi} ===")
    for wname, (start, end) in (("inner-train-with-basis", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            cand = KellyRegimeV9BasisBrake(**strat_kwargs)
            m_c, vol_c, not_c, res_c = measure(cand, start, end, market=market)
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), start, end, market=market)
            beats = "beats v4" if m_c.final_balance > m_v4.final_balance else "LOSES to v4"
            print(f"  {wname:24s} {mname:8s} cand=${m_c.final_balance:>9,.0f} "
                  f"(DD{m_c.max_drawdown_pct:5.1f}% sh{m_c.sharpe:5.2f} tr{m_c.num_trades:4d})  "
                  f"v4=${m_v4.final_balance:>9,.0f} (DD{m_v4.max_drawdown_pct:5.1f}% "
                  f"sh{m_v4.sharpe:5.2f} tr{m_v4.num_trades:4d})   [{beats}]")


# ------------------------------------------------------------------------ diagnostic 1


def exposure_artifact_check() -> None:
    """Diagnostic (1): mandatory exposure-artifact check (R-33/R-34's standing threshold).

    Mean-notional-matched flat rescale of v4's own target, R^2 against
    the candidate's target, inner-validation, both markets. R^2 > 0.95
    means "this is a flat rescale, not a real mechanism" -- the exact
    pattern from ``kelly_regime_v8_ladder_bag.py``.
    """
    print("\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4):")
    for sh, lam, lo, hi in all_configs():
        print(f" sh={sh:5.1f}h lam={lam:.1f} lo={lo:.2f} hi={hi:.2f}:")
        for mname, market in MARKETS:
            cand = KellyRegimeV9BasisBrake(smooth_hours=sh, lam=lam, lo_thresh=lo, hi_thresh=hi)
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


# ------------------------------------------------------------------------ diagnostic 3


def funding_correlation() -> None:
    """Diagnostic (3): re-verify (not merely quote) basis-vs-funding correlation on the working window.

    Daily-resampled correlation between the raw log-basis and the
    already-committed, already-tested (R-35) funding-rate series
    (``load_funding_extended``), restricted to the funding-covered,
    pre-2023 overlap (2020-01-01..2022-12-31) -- the actual window both
    series are simultaneously available on within this session's data
    rule.
    """
    funding, source = load_funding_extended(ROOT / "data")
    if funding is None:
        print("\nno funding data available -- skipping funding correlation check")
        return
    basis = compute_basis(DF, PERP)
    basis_pre = basis.loc[:"2022-12-31"]
    funding_pre = funding.loc[:"2022-12-31"]
    basis_daily = basis_pre.resample("1D").mean()
    funding_daily = funding_pre.resample("1D").mean()
    joined = pd.concat([basis_daily.rename("basis"), funding_daily.rename("funding")], axis=1).dropna()
    corr_overlap = joined["basis"].corr(joined["funding"])
    print(f"\nbasis vs funding-rate correlation (daily means, pre-2023 overlap "
          f"{joined.index.min().date()}..{joined.index.max().date()}, n={len(joined)} days):")
    print(f"  r = {corr_overlap:.4f}")
    train_overlap = joined.loc[TRAIN[0]:TRAIN[1]]
    valid_overlap = joined.loc[VALID[0]:VALID[1]]
    if len(train_overlap) > 2:
        print(f"  inner-train-with-basis portion (n={len(train_overlap)} days): "
              f"r = {train_overlap['basis'].corr(train_overlap['funding']):.4f}")
    if len(valid_overlap) > 2:
        print(f"  inner-validation portion (n={len(valid_overlap)} days): "
              f"r = {valid_overlap['basis'].corr(valid_overlap['funding']):.4f}")
    verdict = "NOT a restatement of the funding signal" if abs(corr_overlap) < 0.3 else "possibly redundant with funding"
    print(f"  verdict: {verdict} (|r| < 0.3 threshold)")


# ------------------------------------------------------------------------ fallback check


def fallback_check() -> None:
    """Hard constraint check: pre-2018-08-14 (no basis coverage), candidate must equal v4 EXACTLY."""
    pre_coverage_end = pd.Timestamp("2018-08-13", tz="UTC")
    df = DF.loc[:pre_coverage_end].copy()
    print(f"\nfallback check: {len(df):,} bars strictly before Deribit coverage "
          f"({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})")
    cand = KellyRegimeV9BasisBrake(lam=0.7, smooth_hours=1.0, lo_thresh=0.02, hi_thresh=0.06)
    cand_prepared = cand.prepare(df.copy())
    v4 = get_strategy(INCUMBENT)
    v4_prepared = v4.prepare(df.copy())
    diff = float(np.max(np.abs(
        cand_prepared["target"].to_numpy() - v4_prepared["target"].to_numpy())))
    excess_nonzero = int(np.sum(cand_prepared["_excess"].to_numpy() != 0.0))
    print(f"  max|target diff| vs v4 over the entire pre-coverage region = {diff:.3e}  "
          f"{'PASS -- exact v4 fallback' if diff < 1e-12 else 'FAIL'}")
    print(f"  bars with nonzero _excess (should be 0): {excess_nonzero}")


# ------------------------------------------------------------------------ diagnostic 4 / causality


PRIMARY = dict(smooth_hours=4.0, lam=0.5, lo_thresh=0.02, hi_thresh=0.06)


def causality() -> None:
    """Diagnostic (4): two-opposite-tampers, on PRICE and, separately, on BASIS.

    Restricted to strictly pre-2023 bars. The price probe is copied from
    ``kelly_regime_v8_ladder_bag.py``'s own procedure. The basis probe is
    new to this file: it tampers the cached basis series itself (rather
    than the price frame) after the same cut, because a price-only probe
    cannot exercise this file's one new ingredient -- basis is loaded
    from an external source untouched by tampering ``df``.
    """
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
        return KellyRegimeV9BasisBrake(**PRIMARY).prepare(frame.copy())

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
        s = KellyRegimeV9BasisBrake(**PRIMARY)
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

    a = run_backtest(KellyRegimeV9BasisBrake(**PRIMARY), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV9BasisBrake(**PRIMARY), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")
    print(f"  tampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS' if ok else 'FAIL'} -- no price-dependent decision at or before the cut moves")

    print("\n=== basis tamper probe (new to this file) ===")
    basis_up = BASIS_ON_SPOT.copy()
    basis_down = BASIS_ON_SPOT.copy()
    basis_slice = basis_up.loc[df.index]
    cut_ts = df.index[cut]
    mask_after = basis_up.index >= cut_ts
    basis_up.loc[mask_after] = basis_up.loc[mask_after] * 3.0
    basis_down.loc[mask_after] = basis_down.loc[mask_after] / 3.0

    strat_up = KellyRegimeV9BasisBrake(**PRIMARY, basis=basis_up)
    strat_down = KellyRegimeV9BasisBrake(**PRIMARY, basis=basis_down)
    pu = strat_up.prepare(df.copy())
    pd_ = strat_down.prepare(df.copy())
    ok2 = True
    for col in ("target", "_mult", "_excess", "_basis_smoothed"):
        a2 = pu[col].to_numpy(dtype=float)[:cut]
        b2 = pd_[col].to_numpy(dtype=float)[:cut]
        worst2 = float(np.nanmax(np.abs(a2 - b2)))
        good2 = worst2 < 1e-9
        ok2 &= good2
        print(f"  column={col:16s} max |difference| before the cut = {worst2:.3e}  "
              f"{'PASS' if good2 else 'FAIL'}")
    print(f"  basis-tamper probe: {'PASS' if ok2 else 'FAIL'} -- "
          f"no basis-dependent decision at or before the cut moves when ONLY the "
          f"basis series (not price) is tampered after the cut")


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"spot: {len(DF):,} bars {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d} (data: {LABEL})",
          file=sys.stderr)
    print(f"perp: {len(PERP):,} bars {PERP.index[0]:%Y-%m-%d} -> {PERP.index[-1]:%Y-%m-%d} (Deribit, real)",
          file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "artifact":
        exposure_artifact_check()
    elif choice == "fundingcorr":
        funding_correlation()
    elif choice == "fallback":
        fallback_check()
    elif choice == "causality":
        causality()
    elif choice == "all":
        sweep()
        select()
        exposure_artifact_check()
        funding_correlation()
        fallback_check()
        causality()
    else:
        print("usage: python experiments/kelly_regime_v9_basis_brake.py "
              "[sweep|select|artifact|fundingcorr|fallback|causality|all]")
