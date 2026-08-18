#!/usr/bin/env python
"""Downside-only / drawdown-sensitive risk denominator for the Kelly sizer.

Novel branch of a parallel round (the other branch works backlog item B-05,
a different idea, disjoint files; see docs/ROUTINE.md "Running directions in
parallel"). Not registered: lives under ``experiments/`` per ROUTINE.md
step 5, and is the ONLY file this session creates or edits.

The question
------------
``kelly_regime_v4`` answers "how much do I hold?" with

    scale_t = min(target_vol / realized_vol_t, max_leverage)

where ``realized_vol_t`` is a SYMMETRIC EWM standard deviation of returns —
it counts an up-burst exactly the same as a down-burst. R-10 measured that
BTC has an *inverse* leverage effect (Baur & Dimpfl 2018): high realized
volatility forecasts the *highest* forward Sharpe here, so a symmetric
denominator de-levers into precisely the best states, and R-08 found that
making the symmetric estimate MORE ACCURATE makes the strategy WORSE for
exactly that reason. Neither result says volatility itself is the wrong
thing to size against — they say a risk measure that cannot distinguish an
up-burst from a down-burst is the wrong thing. This file replaces the
denominator with one that can:

``risk="semidev"``  Sortino & van der Meer (1991, Journal of Portfolio
                    Management 17(4)) downside semi-deviation: the EWM
                    root-mean-square of ``min(r_t, 0)``, i.e. only bars
                    below the target (0) count. An up-burst contributes
                    zero to the denominator; a down-burst still shrinks
                    the position. Floored at ``floor_frac`` times the
                    SAME-period symmetric vol so the denominator cannot
                    collapse to zero in a long uninterrupted uptrend (see
                    "Design notes: the floor" below).

``risk="cdar"``     Chekhlov, Uryasev & Zabarankin (2005, Journal of Risk)
                    Conditional Drawdown-at-Risk: the mean of the worst
                    ``alpha`` fraction of trailing-window drawdowns,
                    computed causally at daily granularity (a bar-level
                    rolling quantile over a 30-90 day window is
                    computationally intractable at 5m resolution; see
                    "Design notes: CDaR causality" below) and broadcast
                    to bars with an explicit one-day shift. Floored at a
                    fixed ``floor_dd`` to guard against the pathology
                    named in the assignment: a naive rolling max-drawdown
                    statistic resets toward zero right after a new
                    all-time high, which would spike exposure exactly at
                    tops. Reported and checked for explicitly below
                    (``python experiments/downside_sizing.py athcheck``).

``risk="symmetric"`` Control arm: the incumbent's own denominator, on the
                    same vote gate and sizer shape, so a difference from
                    ``kelly_regime_v4`` can be attributed to the sizer
                    form (plain vs. conditional/extreme-only) rather than
                    silently blamed on the risk axis.

The vote gate (20/40/80-day latched anchors, 1% band) is copied from
``kelly_regime_v4`` UNCHANGED, per the assignment brief and R-31/R-32's
finding that varying the gate mechanism barely matters ("the gate is worth
more than the choice of gate") — so this file deliberately varies the one
axis (SIZE, i.e. the sizing denominator) this project's evidence says is
where the value is, and leaves the other alone.

Design notes: the floor
------------------------
In a long uninterrupted uptrend (BTC has had several: 2017 H2, 2020-21,
most of 2023-24), ``min(r_t, 0)`` is zero on most bars, so an unfloored
downside semi-deviation drifts toward zero and ``target_vol / risk`` blows
up. The floor ``max(downside_dev_t, floor_frac * sym_vol_t)`` ties the
floor to the SAME bar's symmetric volatility, so it scales with current
market conditions rather than being a stale constant: even in a pure
blow-off top where realized downside is genuinely near zero, the
denominator cannot fall below a fraction of the total volatility the
market is actually exhibiting. ``floor_frac`` is swept in step 3.

Design notes: CDaR causality
-----------------------------
CDaR is computed by resampling ``close`` to daily bars, taking a rolling
peak and drawdown at DAILY granularity, then a rolling tail-mean of that
drawdown series (``_tail_mean``) — cheap at ~3,500 daily observations,
intractable as a bar-level rolling quantile over a 60-day (17,280-bar)
window on a million-row frame. The trap: ``close.resample("1D").last()``
for "today" uses every bar of today INCLUDING ones after the current bar,
because ``prepare()`` receives the whole frame at once. The fix applied
here is to compute the entire daily CDaR series first (which is legitimate
— day d's statistic only summarizes day d's own history), THEN
``.shift(1)`` the finished daily series before broadcasting it to bars
with ``reindex(..., method="ffill")``. After the shift, every bar within
calendar day d — including its very first bar — reads day d-1's fully
realized statistic, never day d's own. This is exactly the kind of
resample-then-reindex construction the causality self-check exists to
catch if done wrong, so it is checked by hand below
(``python experiments/downside_sizing.py causality``), not assumed correct
from the argument above.

Not a duplicate of
-------------------
L-01/L-02/L-03/L-04 vary the VOTE, never the sizing denominator.
``kelly_regime_v3``/R-07 switches between continuous and steady-state
targeting of the SAME symmetric vol at breakout extremes — still cannot
tell an up-burst from a down-burst, just delays reacting to either. R-09
(range estimators: Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang) are
different ESTIMATORS of the same symmetric total-variance quantity, and
read 7-18% low from discretisation bias — a calibration question, not an
asymmetry question. R-08 made the symmetric estimate MORE ACCURATE and
found that this makes the strategy worse, because better calibration
de-levers even more promptly into BTC's good high-vol states — this file
does not make the same quantity more accurate, it measures a DIFFERENT,
asymmetric quantity that is mechanically insensitive to the up-bursts R-08
was over-reacting to. R-11 (Grossman-Zhou drawdown cushion) is a discrete
multiplicative BRAKE — ``exposure *= (wealth - floor) / wealth`` — bolted
on top of the existing sizer; this file does not add a brake, it REPLACES
the sizer's own denominator with a continuous risk statistic, so downside
risk changes the position through the same ``target_vol / risk`` channel
symmetric vol used, not through a separate wealth-floor gate.

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/downside_sizing.py inspect     # what the risk measures look like
    python experiments/downside_sizing.py athcheck     # the ATH-reset pathology, checked explicitly
    python experiments/downside_sizing.py sweep        # step 3 grid
    python experiments/downside_sizing.py neighbours   # plateau check around the frozen pick
    python experiments/downside_sizing.py causality    # by-hand lookahead probe
    python experiments/downside_sizing.py holdout      # step 4, frozen config
    python experiments/downside_sizing.py eth          # pre-registered falsification test
    python experiments/downside_sizing.py windows      # 40-window path check
    python experiments/downside_sizing.py costs        # 0.40% fee tier check
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
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)

N_EVALUATED = 0        # distinct configurations searched in step 3 (deflated Sharpe)
HOLDOUT_READS = 0      # backtests touching any date >= 2023-01-01 (project holdout counter)

OUT = ROOT / "reports" / "downside_sizing"


# --------------------------------------------------------------------------
# Risk measures
# --------------------------------------------------------------------------


def _vote(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> np.ndarray:
    """kelly_regime_v4's gate, byte-for-byte: latched multi-anchor vote."""
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def _symmetric_vol(r: pd.Series, span_bars: int) -> pd.Series:
    """The incumbent's own denominator: EWM std of ALL returns, shift(1)."""
    return (r.ewm(span=span_bars, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1)


def _downside_semidev(r: pd.Series, span_bars: int, sym_vol: pd.Series,
                       floor_frac: float) -> np.ndarray:
    """Sortino & van der Meer (1991): EWM RMS of returns below 0, floored.

    ``r.clip(upper=0.0)`` zeroes every positive return before squaring, so
    an up-burst contributes exactly 0 to the running semi-variance — the
    mechanical difference from ``_symmetric_vol`` that this whole file is
    built around. Floored at ``floor_frac`` times the SAME bar's symmetric
    vol (see the module docstring's "Design notes: the floor").
    """
    downside_sq = r.clip(upper=0.0) ** 2
    downside_var = downside_sq.ewm(span=span_bars, min_periods=BARS_PER_DAY).mean()
    downside_dev = (np.sqrt(downside_var) * np.sqrt(BARS_PER_YEAR)).shift(1)
    floor = floor_frac * sym_vol
    out = np.maximum(downside_dev.to_numpy(), floor.to_numpy())
    return np.where(np.isfinite(out), out, np.nan)


def _tail_mean(arr: np.ndarray, alpha: float) -> float:
    """Mean of the worst ``alpha`` fraction of a 1-D array (largest values)."""
    n = len(arr)
    if n == 0:
        return np.nan
    k = max(1, int(np.ceil(alpha * n)))
    part = np.partition(arr, n - k)[n - k:]
    return float(part.mean())


def _cdar(close: pd.Series, window_days: int, alpha: float,
          floor_dd: float) -> np.ndarray:
    """Chekhlov, Uryasev & Zabarankin (2005): rolling Conditional Drawdown-at-Risk.

    Computed at DAILY granularity (a bar-level rolling quantile over a
    17,000+ bar window is computationally intractable on a million-row
    frame), then shifted a full calendar day and broadcast to bars with
    ``ffill`` — see the module docstring's "Design notes: CDaR causality".
    ``floor_dd`` is the explicit guard against the ATH-reset pathology
    named in the assignment: right after a fresh all-time high, the
    trailing window's worst drawdowns can be genuinely tiny, and an
    unfloored CDaR would let exposure spike exactly there.
    """
    daily_close = close.resample("1D").last().dropna()
    roll_peak = daily_close.rolling(window_days, min_periods=1).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = ((roll_peak - daily_close) / roll_peak).fillna(0.0)
    min_periods = max(5, window_days // 3)
    cdar_daily = dd.rolling(window_days, min_periods=min_periods).apply(
        lambda a: _tail_mean(a, alpha), raw=True)
    cdar_daily = np.maximum(cdar_daily, floor_dd)
    # The one-day shift is what makes this causal: day d's bars read day
    # d-1's fully realized statistic, never day d's own (still-forming, at
    # bar level) drawdown.
    shifted = pd.Series(cdar_daily, index=daily_close.index).shift(1)
    return shifted.reindex(close.index, method="ffill").to_numpy()


# --------------------------------------------------------------------------
# The strategy
# --------------------------------------------------------------------------


class DownsideKelly(Strategy):
    """Fractional-Kelly sizing on the v4 vote gate, downside-only risk denominator.

    ``target_t = vote_t * min(target_risk / risk_measure_t, max_leverage)``,
    latched through a deadband — identical shape to ``kelly_regime``'s
    sizer, with ``risk_measure_t`` swapped for one of three causal,
    price-derived risk statistics (see module docstring). ``risk="cdar"``
    uses ``target_dd``/``floor_dd`` in place of ``target_vol``/an implicit
    vol-based floor, because CDaR is a drawdown FRACTION, not an
    annualized volatility, and the two are not on the same scale.
    """

    name = "downside_kelly"
    # Covers the vote's slowest anchor (80d), the slowest risk-measure
    # option swept below (cdar window up to 90d) and the v3/v4 convention
    # of a 20d margin on top - one warmup long enough for every variant.
    warmup = 100 * BARS_PER_DAY + 10

    def __init__(
        self,
        risk: str = "semidev",
        horizons: tuple[int, ...] = (20, 40, 80),
        band: float = 0.01,
        deadband: float = 0.10,
        max_leverage: float = 2.0,
        # --- semidev / symmetric
        vol_span_days: float = 8.0,
        target_vol: float = 0.55,
        floor_frac: float = 0.35,
        # --- cdar
        cdar_window_days: int = 60,
        cdar_alpha: float = 0.10,
        target_dd: float = 0.15,
        floor_dd: float = 0.03,
    ) -> None:
        if risk not in ("symmetric", "semidev", "cdar"):
            raise ValueError(f"risk must be symmetric/semidev/cdar, got {risk!r}")
        self.risk = risk
        self.horizons = horizons
        self.band = band
        self.deadband = deadband
        self.max_leverage = max_leverage
        self.vol_span_days = vol_span_days
        self.target_vol = target_vol
        self.floor_frac = floor_frac
        self.cdar_window_days = cdar_window_days
        self.cdar_alpha = cdar_alpha
        self.target_dd = target_dd
        self.floor_dd = floor_dd

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()
        span_bars = int(self.vol_span_days * BARS_PER_DAY)

        sym_vol = _symmetric_vol(r, span_bars)
        frac = _vote(close, self.horizons, self.band)

        if self.risk == "symmetric":
            risk_measure = sym_vol.to_numpy()
            target_level = self.target_vol
        elif self.risk == "semidev":
            risk_measure = _downside_semidev(r, span_bars, sym_vol, self.floor_frac)
            target_level = self.target_vol
        else:  # cdar
            risk_measure = _cdar(close, self.cdar_window_days, self.cdar_alpha,
                                 self.floor_dd)
            target_level = self.target_dd

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            v = risk_measure[i]
            scale = min(target_level / v, self.max_leverage) if np.isfinite(v) and v > 0 else 0.0
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["risk_measure"] = risk_measure
        df["vote"] = frac
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def ev(strategy, start=None, end=None, df=None, market=SPOT, tag: str = "",
       balance: float = 1_000.0, count: bool = False) -> object:
    """One backtest, one line. Counts step-3 trials AND holdout reads.

    ``count=True`` marks a distinct configuration for the deflated-Sharpe
    trials count (step 3 only). The holdout counter is separate and
    automatic: any call whose ``start`` is on or after 2023-01-01 (and
    whose frame is the real BTC series, not the ETH falsification data)
    increments it regardless of ``count``, because it measures exposure of
    the actual 2023+ dataset, not intent.
    """
    global N_EVALUATED, HOLDOUT_READS
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    is_holdout = (df is None) and start is not None and pd.Timestamp(start, tz="UTC") >= pd.Timestamp(OOS[0], tz="UTC")
    if is_holdout:
        HOLDOUT_READS += 1
    if start is None and end is None:
        result = run_backtest(strategy, frame, market, balance, data_label=LABEL)
    else:
        result = run_period(strategy, frame, start, end, market=market,
                            start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# ---------------------------------------------------------------------- inspect


def inspect() -> None:
    """What the three risk measures actually look like, before any backtest."""
    r = np.log(DF["close"]).diff()
    sym = _symmetric_vol(r, int(8 * BARS_PER_DAY))
    semi = _downside_semidev(r, int(8 * BARS_PER_DAY), sym, 0.35)
    cdar = _cdar(DF["close"], 60, 0.10, 0.03)

    both = pd.DataFrame({"sym": sym.to_numpy(), "semi": semi}, index=DF.index).dropna()
    print(f"symmetric vol:  mean={both.sym.mean():.3f}  median={both.sym.median():.3f}")
    print(f"downside semidev (floor_frac=0.35, span=8d): "
          f"mean={both.semi.mean():.3f}  median={both.semi.median():.3f}")
    print(f"correlation(sym, semidev) = {both.sym.corr(both.semi):.3f}")
    print(f"fraction of bars where the floor binds (semidev == floor exactly, "
          f"to 1e-9): {float((np.abs(both.semi - 0.35 * both.sym) < 1e-9).mean()):.1%}")

    cd = pd.Series(cdar, index=DF.index).dropna()
    print(f"\nCDaR (window=60d, alpha=0.10, floor=0.03): "
          f"mean={cd.mean():.3f}  median={cd.median():.3f}  "
          f"fraction at floor={(np.abs(cd - 0.03) < 1e-9).mean():.1%}")

    # Pre-registered failure mode (a): is the downside measure basically
    # the same information as symmetric vol in THIS data (crashes come
    # with elevated vol on both sides)?
    ret_year = DF["close"].groupby(DF.index.year).last() / DF["close"].groupby(DF.index.year).first() - 1.0
    print("\nyearly mean risk measure vs BTC year return:")
    sym_y = sym.groupby(DF.index.year).mean()
    semi_s = pd.Series(semi, index=DF.index)
    semi_y = semi_s.groupby(DF.index.year).mean()
    for year in sym_y.index:
        print(f"  {year}  sym={sym_y.get(year, float('nan')):.3f}  "
              f"semi={semi_y.get(year, float('nan')):.3f}  "
              f"BTC return={ret_year.get(year, float('nan')):>+7.1%}")


def athcheck() -> None:
    """The ATH-reset pathology, checked explicitly, per the assignment brief.

    Does CDaR collapse toward its floor right after a fresh all-time high
    (which would spike exposure exactly at tops before a reversal)? And
    does semidev show the analogous behaviour (near-zero right after a
    long uninterrupted run with no down bars)?
    """
    close = DF["close"]
    ath = close.cummax()
    is_new_ath = (close >= ath.shift(1).fillna(-np.inf)) & (close == ath)
    ath_days = DF.index[is_new_ath]
    print(f"{len(ath_days):,} bars register a new all-time high "
          f"({ath_days.min():%Y-%m-%d} .. {ath_days.max():%Y-%m-%d})")

    cdar_uf = _cdar(close, 60, 0.10, floor_dd=0.0)       # unfloored, to see the raw pathology
    cdar_fl = _cdar(close, 60, 0.10, floor_dd=0.03)      # floored
    r = np.log(close).diff()
    sym = _symmetric_vol(r, int(8 * BARS_PER_DAY))
    semi_uf = _downside_semidev(r, int(8 * BARS_PER_DAY), sym, floor_frac=0.0)
    semi_fl = _downside_semidev(r, int(8 * BARS_PER_DAY), sym, floor_frac=0.35)

    df = pd.DataFrame({
        "cdar_unfloored": cdar_uf, "cdar_floored": cdar_fl,
        "semidev_unfloored": semi_uf, "semidev_floored": semi_fl,
    }, index=DF.index)
    at_ath = df[is_new_ath].dropna()
    off_ath = df[~is_new_ath].dropna()
    print("\nrisk measure ON new-ATH bars vs. off them (mean):")
    for col in df.columns:
        print(f"  {col:20s} at-ATH={at_ath[col].mean():.4f}   "
              f"elsewhere={off_ath[col].mean():.4f}   "
              f"ratio={at_ath[col].mean() / max(off_ath[col].mean(), 1e-9):.2f}x")

    # The implied leverage this drives, unfloored vs floored, at the ATH bars.
    lev_uf = np.minimum(0.15 / np.where(cdar_uf > 0, cdar_uf, np.nan), 2.0)
    lev_fl = np.minimum(0.15 / np.where(cdar_fl > 0, cdar_fl, np.nan), 2.0)
    lev = pd.DataFrame({"unfloored": lev_uf, "floored": lev_fl}, index=DF.index)
    print("\nimplied CDaR-sizer leverage (target_dd=0.15) on new-ATH bars:")
    print(f"  unfloored: mean={lev.loc[is_new_ath, 'unfloored'].mean():.2f}  "
          f"max={lev.loc[is_new_ath, 'unfloored'].max():.2f}  "
          f"(vs off-ATH mean {lev.loc[~is_new_ath, 'unfloored'].mean():.2f})")
    print(f"  floored:   mean={lev.loc[is_new_ath, 'floored'].mean():.2f}  "
          f"max={lev.loc[is_new_ath, 'floored'].max():.2f}  "
          f"(vs off-ATH mean {lev.loc[~is_new_ath, 'floored'].mean():.2f})")


# ------------------------------------------------------------------------ sweep


def _semidev_grid():
    out = []
    for span in (4.0, 8.0, 16.0):
        for floor_frac in (0.20, 0.40):
            for tv in (0.40, 0.55):
                out.append((f"semidev span={span:g}d floor={floor_frac:g} tv={tv:g}",
                            dict(risk="semidev", vol_span_days=span,
                                 floor_frac=floor_frac, target_vol=tv)))
    return out


def _cdar_grid():
    out = []
    for window in (30, 60):
        for alpha in (0.10, 0.20):
            for tdd in (0.12, 0.18):
                out.append((f"cdar w={window}d a={alpha:g} tdd={tdd:g}",
                            dict(risk="cdar", cdar_window_days=window,
                                 cdar_alpha=alpha, target_dd=tdd)))
    return out


def _benchmarks(start, end, market, label) -> None:
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}")


def sweep() -> None:
    """Step 3: the risk-measure family and its parameters, inner splits only."""
    variants = [("symmetric (control)", dict(risk="symmetric"))]
    variants += _semidev_grid()
    variants += _cdar_grid()
    for mname, market in MARKETS:
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} variants:")
            for tag, kw in variants:
                ev(DownsideKelly(**kw), start, end, market=market, tag=tag,
                   count=(split == "INNER-TRAIN" and mname == "spot"))
    print(f"\nconfigurations evaluated in sweep(): {N_EVALUATED}")


# Selected on inner-validation: of all 21 step-3 configurations, this is the
# only one that beats kelly_regime_v4 on return, Sharpe AND drawdown
# SIMULTANEOUSLY on both markets in inner-validation (spot $1,123/DD24.1%/
# Sharpe0.35 vs v4's $998/33.2%/0.14; futures $1,238/DD21.3%/Sharpe0.53 vs
# v4's $1,064/32.3%/0.25). See the report for why this selection is treated
# with suspicion rather than celebrated: it badly underperforms v4 on
# inner-train (the opposite split), the exact split-disagreement pattern
# R-28/R-31 traced to lower average exposure suiting a bear/chop validation
# window rather than to a genuine risk-discrimination advantage. The
# neighbourhood check below is what decides which story is right.
FROZEN_RISK = "cdar"
FROZEN = dict(risk="cdar", cdar_window_days=60, cdar_alpha=0.10, target_dd=0.12,
              floor_dd=0.03, deadband=0.10, max_leverage=2.0)


def neighbours() -> None:
    """Plateau check: vary one knob at a time around the frozen selection."""
    grid = [("frozen", {})]
    grid += [(f"cdar_window={w}d", dict(cdar_window_days=w)) for w in (20, 30, 45, 75, 90)]
    grid += [(f"cdar_alpha={a:g}", dict(cdar_alpha=a)) for a in (0.05, 0.15, 0.20, 0.30)]
    grid += [(f"target_dd={t:g}", dict(target_dd=t)) for t in (0.08, 0.10, 0.15, 0.18, 0.22)]
    grid += [(f"floor_dd={f:g}", dict(floor_dd=f)) for f in (0.0, 0.01, 0.05)]
    grid += [(f"deadband={d:g}", dict(deadband=d)) for d in (0.05, 0.15, 0.20)]
    grid += [(f"max_leverage={m:g}", dict(max_leverage=m)) for m in (1.5, 3.0)]
    for mname, market in MARKETS:
        print(f"\nINNER-VALIDATION neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(DownsideKelly(**{**FROZEN, **kw}), *VALID, market=market, tag=tag,
               count=(mname == "spot"))
        print(f"INNER-TRAIN neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(DownsideKelly(**{**FROZEN, **kw}), *TRAIN, market=market, tag=tag)
    print(f"\nconfigurations evaluated in neighbours(): {N_EVALUATED}")


# ---------------------------------------------------------------------- causality


def causality() -> None:
    """The strict by-hand lookahead probe — experiments get no CI protection.

    Two-opposite-tampers procedure (R-28/R-31 convention): every bar
    strictly after ``cut`` is multiplied by 3 in one copy and divided by 3
    in the other; every decision (order) at or before ``cut`` must be
    bit-identical, and the ``target``/``risk_measure`` columns must be
    identical there too (the check a truncation test cannot see — a
    full-series mean/std/quantile applied to early rows). This is the
    check the CDaR resample-then-shift construction most needs, since it
    is the part of this file least like the incumbent's plain
    ``.shift(1)``.
    """
    from tradebot.broker import PaperBroker

    df = DF.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    ok = True
    for risk in ("symmetric", "semidev", "cdar"):
        def decisions(frame):
            s = DownsideKelly(risk=risk)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
        pa = DownsideKelly(risk=risk).prepare(up.copy())
        pb = DownsideKelly(risk=risk).prepare(down.copy())
        worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
                    for c in ("target", "risk_measure", "vote"))
        good = not bad and worst < 1e-9
        ok &= good
        print(f"  risk={risk:10s} orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |column difference| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------ holdout


def holdout() -> None:
    """Step 4. Configuration frozen above; decision rule is in the report."""
    for mname, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}:")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"  {name}")
        ev(DownsideKelly(**FROZEN), *OOS, market=market, tag="  downside_kelly (FROZEN)")
    print(f"\nholdout reads so far (this file): {HOLDOUT_READS}")


def eth() -> None:
    """Pre-registered falsification: does the mechanism survive on ETH?

    Same venue (Bitfinex), same window as R-17/R-28/R-31 — the design this
    project has used every time it asked this question — only the asset
    varies. Does NOT touch the 2023+ BTC holdout (R-19/R-28/R-31 convention).
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        eth_df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(eth_df):,} bars  "
              f"{eth_df.index[0]:%Y-%m-%d} -> {eth_df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            for name in ("buy_and_hold", "kelly_regime_v4"):
                ev(get_strategy(name), None, None, df=eth_df, market=market,
                   tag=f"  {name}")
            ev(DownsideKelly(**FROZEN), None, None, df=eth_df, market=market,
               tag="  downside_kelly (frozen)")


def costs() -> None:
    """Step 4 cost check: the real 0.40% Bitstamp taker tier, spot."""
    print("HOLDOUT 2023+ spot, at both taker tiers:")
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"    {name}")
        ev(DownsideKelly(**FROZEN), *OOS, market=market, tag="    downside_kelly")


def windows(trials: int = 40, seed: int = 42) -> None:
    """Path sensitivity, the R-19 design: identical random windows, both arms."""
    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("downside_kelly", DownsideKelly(**FROZEN))]
    warmup = max(s.warmup for _, s in contenders) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(90, 731) * 288)
        specs.append((int(rng.integers(warmup, len(DF) - length)), length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        for mname, market in MARKETS:
            for name, strat in contenders:
                res = run_backtest(strat, window, market, 1_000.0,
                                   trade_start=warmup, data_label=LABEL)
                eqv = res.equity.to_numpy(dtype=float)
                base, seg = eqv[warmup], eqv[warmup:]
                is_ok = np.isfinite(base) and base > 0
                rows.append({"trial": k, "market": mname, "strategy": name,
                             "return_pct": 100.0 * (seg[-1] / base - 1.0) if is_ok else -100.0,
                             "max_dd_pct": max_drawdown_pct(seg) if is_ok else 100.0,
                             "liquidated": res.liquidated})
        print(f"[{k}/{trials}]", end=" ", flush=True, file=sys.stderr)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "windows.csv", index=False)

    print(f"\n{trials} random windows (90-730 days), identical across strategies:\n")
    for mname, _ in MARKETS:
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name, _ in contenders:
            g = sub[sub.strategy == name].set_index("trial")
            print(f"    {name:16s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"beat hold {(g['return_pct'] > bench).mean():>5.0%}  "
                  f"liq {g.liquidated.mean():>4.0%}")
        a = sub[sub.strategy == "downside_kelly"].set_index("trial")["max_dd_pct"]
        b = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")["max_dd_pct"]
        d = (a - b).dropna()
        ra = sub[sub.strategy == "downside_kelly"].set_index("trial")["return_pct"]
        rb = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")["return_pct"]
        dr = (ra - rb).dropna()
        print(f"    paired DD (downside_kelly - v4): median {d.median():+.1f}pp, "
              f"deeper in {(d > 0).mean():.0%}")
        print(f"    paired return (downside_kelly - v4): median {dr.median():+.1f}pp, "
              f"higher in {(dr > 0).mean():.0%}\n")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"inspect": inspect, "athcheck": athcheck, "sweep": sweep,
            "neighbours": neighbours, "causality": causality, "holdout": holdout,
            "eth": eth, "costs": costs, "windows": windows}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/downside_sizing.py [{'|'.join(cmds)}]")
