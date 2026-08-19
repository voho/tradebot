"""Continuous trend-strength conviction on top of the existing regime vote/sizer.

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

Citation and mechanism
-----------------------
Baltas & Kosowski, "Improving Time-Series Momentum Strategies: The Role of
Trading Signals and Volatility Estimators" (SSRN 2140091; Journal of
Financial Econometrics; also circulated as an EDHEC-Risk / CME Group
study). Their finding: replacing a binary/naive momentum sign with a
**continuous trend-strength signal** — the t-statistic (equivalently,
signed R^2) of a causal OLS fit of price against time over the lookback
window — improves signal quality and *reduces* turnover, because it tells
apart a clean, statistically confident trend from a noisy path that
happens to share the same net sign. A window whose price path is a
straight line scores a large |t|; a window with the same start-to-end
return but a choppy, mean-reverting path in between scores near zero,
because the fit barely explains the variance in price. The binary vote
this repo uses cannot see that difference at all — it only asks whether
the endpoint is above or below the anchor.

Everything downstream of ``kelly_regime_v4`` is unchanged: the same
conditional (extremes-only) volatility target, the same hysteresis, the
same 2x cap. Only the exposure that vote/sizer combination is asked to
hold is modulated by trend cleanliness.

The construction
-----------------
For a causal window of ``W`` *days* ending at (and including) the current
bar, fit ``log(price)`` against the day index by OLS. The regression
slope's t-statistic has the closed-form identity (standard result for
simple linear regression; also the t-test for a Pearson correlation)::

    t = r * sqrt(W - 2) / sqrt(1 - r^2)

where ``r`` is the Pearson correlation between log-price and the day
index over the window. Correlation is invariant to a constant shift of
either variable, so the *global* running index used here (rather than a
window-local ``0..W-1`` index) gives the identical ``r`` — pandas' native
rolling ``.corr()`` is an O(n) causal computation of the window-local OLS
t-statistic, with no lookahead and no per-window Python loop.

**One deliberate departure from the naive version, worth stating plainly.**
The first working version of this fit ran directly on 5-minute bars
(``W`` days x 288 bars/day as the OLS sample size). Its t-statistics came
back in the hundreds (20-day q5/q95 of -140/+203; 80-day -296/+443) —
because a highly autocorrelated 5-minute price path is nowhere near
``W*288`` independent observations, and the classical t-statistic formula
assumes it is. This is the exact warning R-20/R-29 already carry for this
repo's own bootstrap ("a million autocorrelated 5-minute bars is not a
million observations"), rediscovered here in a new spot. The fix used
throughout this file is the same one the project already uses for daily
bootstrap blocks: fit the OLS on the window's **daily-cadence closes**
(``close`` sampled every ``BARS_PER_DAY``-th bar — a fixed causal subset
of the same bar series, not a resample that could peek into a forming
bar), so ``W`` really is close to the regression's degrees of freedom.
That drops the t-statistics into single-to-low-double digits (20-day
q5/q95 -8.0/+11.6; 80-day -17.3/+26.0, measured on the committed data),
where a clip level of a few units is a meaningful, interpretable
threshold rather than a number that saturates on almost every bar. The
daily-cadence value is held constant intraday and reindexed/forward-filled
onto the 5-minute bar grid the strategy trades on — a coarser update
cadence than the vote's own anchors, in the same spirit as the HAR
daily/weekly/monthly components v3/v4 already cite (Corsi 2009).

``t`` is signed (positive for an uptrend, negative for a downtrend, same
convention as the regression slope) and unbounded. It is turned into a
conviction in ``[0, 1]`` — or, for the signed variant, ``[-1, 1]`` — by a
linear clip::

    conviction(t, t_clip) = clip(|t| / t_clip, 0, 1)
    signed_conviction(t, t_clip) = clip(t / t_clip, -1, 1)

This is the simplest member of the weighting-function family Baltas &
Kosowski study (they compare sign / linear / and their preferred
normalized-and-capped forms of the t-statistic against the naive sign
rule); a linear clip is used here rather than a smoother squashing
function so the one free parameter (``t_clip``) has an obvious
interpretation — the t-statistic at which conviction saturates — and a
plateau in it is easy to read off a sweep.

Like the rest of this repo's anchor logic, the window used for the OLS
fit ends at and includes the current bar's close (the same convention
``kelly_regime``'s anchor means use); the position that conviction gates
is filled at the *next* bar's open, one bar later.

Three variants (mechanism, one line each — see the class docstring)::

    mode="overlay"          conviction multiplies the incumbent's LATCHED
                             vote, using v4's own anchors for the fit.
    mode="continuous_vote"  conviction REPLACES the latched vote outright
                             (no latching at all — the OLS window itself
                             supplies the smoothing).
    mode="single_window"    one independent trend-quality window, used as
                             a standalone multiplier on the full v4
                             output (vote fraction x vol-target scale),
                             decoupled from the vote's own anchors.

Not a duplicate of
-------------------
L-01..L-04 (the `kelly_regime` family): all four vary how the vote's
BINARY latched threshold is combined or exponentiated (v2's convex power,
v3/v4's conditional vol targeting) — none of them look at the price path
*inside* an anchor's window, only at whether today's close sits above or
below it.

R-06 (anchor ladders of 7-48 moving averages) and R-07 (anchor timescale
sweep, 18-28 days): both vary WHICH windows vote and HOW MANY, with the
per-anchor vote itself remaining the same 1%-band binary threshold. This
idea holds the anchors fixed and asks a different question of each one:
not "is price above or below you" but "how clean is the path that got
here".

R-28/R-31/R-32 (e-process gate vs latched vote, at raw and matched risk):
these vary WHICH STATISTIC decides the gate is open — accumulated
log-wealth of a sequential drift test, vs. a latched threshold — but both
are functions of the FIRST MOMENT of returns (mean drift) accumulated
over time, and R-31/R-32 showed the two are indistinguishable once
matched for risk. Trend-strength conviction is a different moment
entirely: it is a measure of PATH QUALITY (how well a straight line
explains the window, i.e. how much of the window's variance is trend
versus noise) that is well-defined even when the mean drift is identical
between two windows. A noisy window and a clean window can have the same
e-process wealth and the same latched vote and different t-statistics.

Simulable here?
----------------
Yes. One price series, causal (pandas rolling `.corr()`, no lookahead;
verified by hand below since this file gets no registry-based
`test_causality_strict.py` coverage), no new data, no order book.

Pre-registered failure modes (named before any code ran)
----------------------------------------------------------
(a) the t-statistic conviction is a smoothed/relabelled version of the
    existing latched vote and adds nothing beyond noise — the two should
    then be highly correlated and any inner-validation improvement should
    sit inside the +/-0.2 Sharpe noise floor;
(b) continuous re-scaling raises turnover (the conviction can move every
    bar, unlike a latched vote that only moves at a crossing) enough that
    fees eat any gain, i.e. it survives at 0.10% but not at Bitstamp's
    0.40% entry tier;
(c) it does not survive on ETH (Bitfinex, the R-17 window) — the
    falsification test pre-registered for step 2, below;
(d) the improvement over unmodified `kelly_regime_v4` is inside the
    +/-0.2 Sharpe noise floor and is not a drawdown win either;
(e) the result is a peak at one t_clip/window choice rather than a
    plateau, i.e. it is a lucky clip level rather than a real property of
    trend quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


def rolling_trend_tstat(close: pd.Series, window_days: int) -> np.ndarray:
    """Causal rolling OLS trend t-statistic, fit on daily-cadence closes.

    ``close`` is the full 5-minute-bar series. The fit itself runs on a
    fixed causal subsample — every ``BARS_PER_DAY``-th bar — so the
    regression's degrees of freedom are close to ``window_days`` rather
    than ``window_days * BARS_PER_DAY``; see the module docstring for why
    that matters. The window ends at (and includes) the most recent daily
    sample point, and the fitted value is held constant and
    forward-filled back onto the full 5-minute index — both steps use
    only data at or before the bar in question, so there is no lookahead.
    """
    daily = close.iloc[::BARS_PER_DAY]
    logp = np.log(daily)
    idx = pd.Series(np.arange(len(logp), dtype=float), index=logp.index)
    r = logp.rolling(window_days, min_periods=window_days).corr(idx)
    n = window_days
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = r.to_numpy()
        denom = np.sqrt(np.clip(1.0 - rv * rv, 1e-12, None))
        t_daily = rv * np.sqrt(max(n - 2, 1)) / denom
    t_daily = np.nan_to_num(t_daily, nan=0.0, posinf=0.0, neginf=0.0)
    t_full = pd.Series(t_daily, index=daily.index).reindex(close.index).ffill()
    return t_full.fillna(0.0).to_numpy()


def conviction(t_stat: np.ndarray, t_clip: float) -> np.ndarray:
    """Unsigned trend-quality conviction in [0, 1]: clip(|t| / t_clip, 0, 1)."""
    return np.clip(np.abs(t_stat) / t_clip, 0.0, 1.0)


def signed_conviction(t_stat: np.ndarray, t_clip: float) -> np.ndarray:
    """Signed trend-quality conviction in [-1, 1]: clip(t / t_clip, -1, 1)."""
    return np.clip(t_stat / t_clip, -1.0, 1.0)


class TrendQualityKelly(Strategy):
    """`kelly_regime_v4`'s vote/sizer, gated or replaced by OLS trend-strength.

    Mechanism, one sentence per mode:

    - ``overlay``: multiply the incumbent's latched multi-anchor vote
      fraction by the mean per-anchor trend-quality conviction, so a vote
      that has flipped because of a clean, statistically confident move
      gets full exposure and one that has flipped on noisy chop with the
      same net sign gets less.
    - ``continuous_vote``: replace the latched per-anchor vote outright
      with the signed, clipped t-statistic mapped to [0, 1], so exposure
      responds continuously to trend confidence rather than only at a
      1%-band crossing.
    - ``single_window``: leave the vote and sizer exactly as v4 computes
      them and multiply the whole result by one independent trend-quality
      read, decoupled from the vote's own anchors.

    The sizer beneath all three modes is v4's own conditional (extremes-
    only) volatility target, unchanged, so any result is attributable to
    the conviction term and not to a different sizing rule.
    """

    name = "trend_quality"

    def __init__(
        self,
        mode: str = "overlay",
        t_clip: float = 4.0,
        horizons: tuple[int, ...] = (20, 40, 80),
        band: float = 0.01,
        window_days: float = 60.0,
        sizer: str = "conditional",
        target_vol: float = 0.55,
        max_leverage: float = 2.0,
        vol_span: int = 8 * BARS_PER_DAY,
        deadband: float = 0.10,
        anchor_span_days: int = 180,
        high_in: float = 1.70,
        high_out: float = 1.20,
        low_in: float = 0.55,
        low_out: float = 0.85,
    ) -> None:
        if mode not in ("overlay", "continuous_vote", "single_window"):
            raise ValueError(f"mode must be overlay/continuous_vote/single_window, got {mode!r}")
        if sizer not in ("plain", "conditional"):
            raise ValueError(f"sizer must be 'plain' or 'conditional', got {sizer!r}")
        self.mode = mode
        self.t_clip = t_clip
        self.horizons = horizons
        self.band = band
        self.window_days = window_days
        self.sizer = sizer
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        # warmup: the slowest of the anchors, the single-window fit, and the
        # 180-day conditional-sizer anchor, plus a small guard.
        slowest_days = max((*horizons, window_days if mode == "single_window" else 0,
                            anchor_span_days))
        self.warmup = int(slowest_days * BARS_PER_DAY) + 10

    # -------------------------------------------------------------- the gate

    def _vote(self, close: pd.Series) -> np.ndarray:
        """Latched multi-anchor vote — `kelly_regime`'s gate, unchanged."""
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=close.index,
            )
            votes.append(v.ffill().fillna(0.0))
        return (sum(votes) / len(votes)).to_numpy()

    def _anchor_conviction(self, close: pd.Series) -> np.ndarray:
        """Mean unsigned trend-quality conviction across the vote's own anchors."""
        convs = [conviction(rolling_trend_tstat(close, int(d)), self.t_clip)
                 for d in self.horizons]
        return np.mean(convs, axis=0)

    def _continuous_vote(self, close: pd.Series) -> np.ndarray:
        """Per-anchor signed conviction averaged into a [0, 1] bull fraction."""
        bulls = []
        for d in self.horizons:
            t = rolling_trend_tstat(close, int(d))
            bulls.append((signed_conviction(t, self.t_clip) + 1.0) / 2.0)
        return np.mean(bulls, axis=0)

    # ------------------------------------------------------------- the sizer

    def _scale(self, vol: np.ndarray) -> np.ndarray:
        """v4's own conditional (extremes-only) or plain inverse-vol sizer."""
        with np.errstate(divide="ignore", invalid="ignore"):
            full = np.minimum(self.target_vol / vol, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        if self.sizer == "plain":
            return full

        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        out = np.empty(len(vol))
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(len(vol)):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            out[i] = full[i] if state != 0 else steady[i]
        return out

    # ----------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        logp = np.log(close)
        r = logp.diff()
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        if self.mode == "overlay":
            frac = self._vote(close) * self._anchor_conviction(close)
        elif self.mode == "continuous_vote":
            frac = self._continuous_vote(close)
        else:  # single_window
            t = rolling_trend_tstat(close, int(self.window_days))
            frac = self._vote(close) * conviction(t, self.t_clip)

        scale = self._scale(vol)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac[i] * scale[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["conf"] = frac
        df["scale"] = scale
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
