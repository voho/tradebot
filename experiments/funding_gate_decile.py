"""kelly_regime_v4 + a binary top-decile-funding flat gate (R-35 conservative branch, 08-19).

Unregistered experiment: lives under ``experiments/`` so it is NOT
auto-discovered (docs/ROUTINE.md step 5 / the registry's package-scan in
``tradebot.registry``). Do not decorate with ``@register``.

Pre-registration: ``docs/LEDGER.md``, "R-35 pre-registration -- written and
committed before the holdout was read." This file is exactly the
"Conservative" variant named there. Read that section before changing
anything here.

Idea in one sentence
---------------------
R-14 measured ``kelly_regime_v4`` paying +20.05%/yr in funding while it
holds, against +2.78%/yr while flat, because the crowding the strategy
trades *is* the crowding that sets the rate. R-16 found funding itself
predicts forward returns (14-day Q1-Q5 spread +3.57pp) and named, in so
many words, "a gate that stands flat when funding is in its top decile"
as the low-turnover way to use that finding. This file is that backlog
item (B-05), executed for the first time, on the SIZE axis (the only axis
this project's twenty-five-plus strategies have ever found to work) as a
strict binary override rather than a new predictor.

Mechanism
---------
``kelly_regime_v4``'s vote (the 20/40/80-day latched multi-anchor vote)
and its conditional (extreme-only) volatility-targeting scale (inherited
from v3) are reproduced here BYTE FOR BYTE -- copied directly from
``kelly_regime_v3.KellyRegimeV3.prepare`` / ``kelly_regime_v4.KellyRegimeV4``,
not re-derived and not reached by subclassing v4 and monkeying with its
internals, so there is no room for behavioural drift between what v4 does
and what this file's "v4 part" does. The full, untouched v4 target series
(``v4_target``, including v4's own deadband latch) is computed first, and
ONLY THEN is the gate applied as a bar-wise override:

    df["target"] = np.where(funding_pctl >= decile, 0.0, v4_target)

-- i.e. the override replaces v4's own output on gated bars; it is never
fed back into v4's internal deadband loop, so v4's own hysteresis dynamics
are literally unchanged by this file's existence. This is the simplest
reading of "v4's vote and conditional-vol-targeting sizer are left
completely unchanged" from the pre-registration, and it is the one used
here.

Funding loading and causal alignment
-------------------------------------
Real Binance BTCUSDT funding (``tradebot.data.load_funding``) covers
2020-01-01..2023-12-31 only, 8-hourly, tz-aware UTC (4,383 settlements).
It is aligned onto the 5m OHLCV bar index by forward-fill (each bar sees
the most recent *already-settled* rate, never a future one) with an
explicit cutoff so a plain ffill cannot silently extend the LAST real
settlement's rate forever into 2024-2026 once it runs off the end of the
real series -- that would be exactly the "proxy unavailable data out of
price" mistake this project's standing rules (docs/ROUTINE.md) forbid:

    aligned = funding.reindex(df.index.union(funding.index)).sort_index().ffill().reindex(df.index)
    aligned = aligned.where(df.index <= funding.index.max())

Bars before ``funding.index.min()`` are naturally NaN from the ffill
already (nothing to forward-fill from yet); bars strictly after
``funding.index.max()`` are explicitly forced to NaN by the ``.where``
above rather than left holding the last real rate. Wherever the aligned
rate is NaN the gate is INERT (behaves exactly as plain v4) -- there is
no default rate, no period mean, nothing substituted for a bar the real
data does not cover.

Percentile computation
-----------------------
Ranking is computed in the funding series' OWN native settlement space
(4,383 points, not 1M+ bars), for two reasons: it is the honest read of
"rank the rate against its own trailing history" the pre-registration
asks for (a rolling window measured in *settlements*, since that is the
series' natural sampling rate, not in 5m bars which would just repeat
each settlement ~96 times inside the window), and it is dramatically
cheaper than a rolling ``.apply`` over a million-row frame. Each
settlement's rank uses only settlements at-or-before it
(``rolling(...).apply(lambda x: (x <= x.iloc[-1]).mean())``, or, for the
expanding-window configuration, ``.expanding(...)`` with the same
lambda) -- strictly causal by construction, no different in spirit from
the bar-space rolling/EWM windows v3/v4 already use for volatility. The
resulting settlement-space percentile series is then aligned onto the 5m
bar index with the identical ffill-then-cutoff procedure used for the
raw rate.

Before ranking, the raw 8h rate is smoothed with a short causal EWM in
settlement-space (``smooth_settlements``, default 3 settlements = 1 day).
This is NOT swept -- it is a fixed, a-priori design choice, made for one
reason: the raw rate is already a step function (one value per 8h), so
"smoothing" here is not about killing bar-to-bar noise (there is none
inside an 8h block) but about keeping a single anomalous settlement from
flipping the percentile rank on its own, which is the "single-settlement
flicker at the boundary" the pre-registration flags as a judgment call
for the implementer to make and document. One day (3 settlements) is
short relative to every window swept below (90/180/365 days) and long
enough to average across one full daily cycle of the three settlement
times.

The one swept knob is the ranking lookback (``funding_window_days``):
90, 180, 365, and an expanding-from-2020-01-01 window (``None``). The
decile threshold is FIXED at 0.90, never tuned -- that is what makes this
the conservative branch, per the pre-registration.

Causality
---------
Every step above is either an ``ewm`` or a ``rolling``/``expanding``
window over rows <= i (in settlement-space, then causally aligned by
ffill+cutoff, never bfill, onto bar-space), so row i's ``funding_pctl``
and ``target`` depend only on rows <= i of both the OHLCV frame and the
funding series. Nothing here regresses over the whole frame or applies a
whole-series statistic (a global mean/std/quantile) to early rows. The
v4 half is v3/v4's own, already-shipped, already-tested computation,
copied unmodified. See ``experiments/reports/funding_gate_decile_report.md``
for the two-opposite-tampers verification run against this file
specifically (price tampered, not the funding series, since the funding
series carries its own independent timestamps and is not derived from
price at all -- exactly the point of using it, per the pre-registration's
"Constraint attacked" section).

Falsifiable prediction (recorded before evaluation; see the report for the
result). The pre-registration's own stated prediction for THIS branch,
verbatim: funding richness should correlate strongly with v4's
already-elevated exposure states (both read the same crowding), so any
drawdown cut is expected to be another instance of the exposure-level
artifact this project has hit three times before (L-04/R-33, R-28/R-31,
R-32) rather than a genuine gate-quality effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import load_funding
from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
SETTLEMENTS_PER_DAY = 3  # Binance-style 8h funding

ROOT = Path(__file__).resolve().parents[1]


def _funding_percentile(funding: pd.Series | None, bar_index: pd.DatetimeIndex,
                         window_days: float | None, smooth_settlements: float,
                         min_settlements: int) -> np.ndarray:
    """Causal trailing percentile rank of funding, aligned onto ``bar_index``.

    Returns NaN wherever the real funding series does not cover the bar
    (before its first settlement, after its last, or the file is absent) --
    the caller must treat NaN as "gate inert," never substitute a value.
    """
    if funding is None or len(funding) == 0:
        return np.full(len(bar_index), np.nan)

    # Short causal smoothing in settlement-space (see module docstring):
    # guards against one anomalous settlement flipping the rank at a
    # window boundary. ewm uses only rows <= i, so this stays causal.
    smoothed = (funding.ewm(span=max(1.0, smooth_settlements), min_periods=1).mean()
                if smooth_settlements > 1 else funding)

    if window_days is None:
        # Expanding-from-2020-01-01: whole funding history to date, no fixed window.
        pctl = smoothed.expanding(min_periods=min_settlements).apply(
            lambda x: float((x <= x.iloc[-1]).mean()), raw=False)
    else:
        window = max(min_settlements, int(round(window_days * SETTLEMENTS_PER_DAY)))
        pctl = smoothed.rolling(window, min_periods=min_settlements).apply(
            lambda x: float((x <= x.iloc[-1]).mean()), raw=False)

    # Causal alignment onto the 5m bar grid: each bar sees the most recent
    # *already-computed* settlement rank, forward-filled, and NEVER a rank
    # extrapolated past the real data's last settlement (the "proxy
    # unavailable data out of price" mistake this project forbids).
    aligned = (pctl.reindex(bar_index.union(pctl.index)).sort_index()
               .ffill().reindex(bar_index))
    aligned = aligned.where(bar_index <= funding.index.max())
    return aligned.to_numpy()


class FundingGateDecile(Strategy):
    """kelly_regime_v4, unchanged, plus a binary flat-override on top-decile funding.

    See module docstring for the full mechanism. Every v4-inherited
    parameter defaults to v4's own shipped value; ``funding_window_days``,
    ``decile``, ``funding_smooth_settlements`` and ``funding_min_settlements``
    are the only new knobs. ``decile`` is fixed at 0.90 in every
    evaluation in the report -- it is not swept (pre-registration).
    """

    name = "funding_gate_decile"
    warmup = 80 * BARS_PER_DAY + 10  # matches kelly_regime_v4 exactly

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 vote_gamma: float = 1.0,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 funding_window_days: float | None = 180, decile: float = 0.90,
                 funding_smooth_settlements: float = 3.0,
                 funding_min_settlements: int = 30,
                 funding: pd.Series | None = None) -> None:
        # ---- identical to kelly_regime / v3 / v4 -------------------------
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.vote_gamma = vote_gamma
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        # ---- new: the funding decile gate ---------------------------------
        self.funding_window_days = funding_window_days  # None = expanding from 2020-01-01
        self.decile = decile                             # fixed at 0.90 in every eval
        self.funding_smooth_settlements = funding_smooth_settlements
        self.funding_min_settlements = funding_min_settlements
        # Loaded once at construction, not per-prepare() call; overridable
        # (the causality probe and any out-of-repo-data test pass their own
        # series here rather than reaching into ROOT/"data").
        self.funding = funding if funding is not None else load_funding(ROOT / "data")

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
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

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

        # ---- byte-for-byte v3/v4: single causal forward pass, the FULL,
        # UNMODIFIED v4 target (this file's gate never enters this loop) --
        n = len(df)
        v4_target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
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
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            v4_target[i] = pos

        # ---- new: causal funding-percentile flat gate ---------------------
        pctl = _funding_percentile(self.funding, df.index, self.funding_window_days,
                                   self.funding_smooth_settlements,
                                   self.funding_min_settlements)
        gated = np.isfinite(pctl) & (pctl >= self.decile)
        target = np.where(gated, 0.0, v4_target)

        df["target"] = target
        df["v4_target"] = v4_target      # diagnostics: mean exposure vs v4, artifact check
        df["funding_pctl"] = pctl        # diagnostics: causality probe, gate-firing rate
        df["gated"] = gated
        return df

    def on_bar(self, ctx: Context) -> None:
        # Identical execution pattern to kelly_regime.KellyRegime.on_bar:
        # signal at bar close, fill at next open via order_notional.
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures
