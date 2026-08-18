"""Continuous, momentum-conditioned funding gate on kelly_regime_v4 (B-05, novel variant).

Not `@register`ed, not discovered by `tradebot run` - lives only in
`experiments/`. Owns this one file; the conservative branch of B-05
(`experiments/funding_gate_conservative.py`, a hard flat-in-the-top-decile
gate) is a separate, independent piece of work in the same session and is
not modified or imported here.

Mechanism, one sentence: instead of a hard "flat if funding is in the top
decile" rule, scale exposure down *smoothly* as trailing funding rises
through a soft ramp starting below the top decile, and only let that ramp
bite when short-horizon price momentum has *also* stalled or turned -
because R-16 (`docs/LEDGER.md`) found elevated funding predicts negative
14-day forward returns only when price is *not* also still confirming the
move (Q1-Q5 spread +3.57pp, correlation with trailing return only 0.39,
so this is not a momentum proxy in disguise).

The causal story, in terms of R-14/R-16
----------------------------------------
R-14 established *why* the cost is a problem: funding runs ~+20%/yr while
`kelly_regime_v4` holds vs ~+2.8%/yr while flat, because the same crowd
whose accumulation drives the trend vote is what sets the funding rate.
R-16 refined *when* that crowding is dangerous rather than merely
expensive: high funding with price still rising is a crowded trend that
is still working (Q3, the crowded-but-confirming quintile, actually did
best in R-16's table at +3.06%); high funding with price no longer
confirming is the crowd being *late* - funding paid to hold a position
that is about to give back its gains. A hard decile gate cannot see that
distinction; it dampens identically whether the trend is intact or not,
and its on/off transitions cost turnover exactly where funding is
noisiest (R-16's own warning: the middle quintiles are non-monotone).
This variant targets the *late-crowd* mechanism specifically: dampen only
when BOTH conditions R-16 separated actually hold at once, and dampen by
degree (a ramp) rather than by a step, so a funding reading just below
the old cutoff no longer produces full exposure while one just above it
produces none.

Design
------
Two knobs compose multiplicatively into `funding_scale in [1-max_dampen, 1]`,
applied to `kelly_regime_v4`'s already-computed `target` in `on_bar`, and
ONLY when `ctx.market.pays_funding` is true (checked via `ctx.market`,
never assumed):

1. `ramp` - a continuous, causal rolling percentile-rank of the trailing
   settled funding rate (same construction as the conservative branch's
   gate, so the two are comparable on the same statistic) mapped through
   a linear ramp from 0 at `pct_start` to 1 at `pct_full`. `pct_start`
   sits BELOW the top decile (default 0.75), so the ramp is already
   engaging before a hard-decile gate would open at all, and it saturates
   at `pct_full` rather than at 1.0 sharp, spreading the transition
   across many settlements instead of one - this is idea (a) from the
   brief: continuous, not discrete, to cut the whipsaw a hard cutoff
   forces at the 90th-percentile boundary.
2. `stall` - a continuous, causal short-horizon momentum check: the
   `mom_days`-day trailing simple return (days, not the 20/40/80-day
   anchors - a horizon an order of magnitude shorter, chosen to detect
   "has the move that made funding rich already stopped confirming" on a
   timescale funding itself moves on, three 8h settlements a day). 0 when
   that return is still >= 0 (still confirming, per R-16 the SAFE case
   even at high funding), ramping to 1 as it falls through `-mom_scale`
   (visibly turned). This is idea (b): condition the dampening on R-16's
   actual finding rather than on funding level alone.

`funding_scale = 1 - max_dampen * ramp * stall`. Because it is a product,
either knob at 0 fully disarms the gate: funding merely elevated with the
trend intact (ramp>0, stall=0) does nothing, matching R-16's Q3 result;
a momentum stall with funding still low (ramp=0, stall>0) also does
nothing, since low funding was never R-14/R-16's danger signal.

Outside the committed 2020-01-01..2023-12-31 funding window - before the
first settlement, after the last, or before `lookback_days` of trailing
history has accumulated - `merge_asof` with an 8h tolerance leaves
`pct` NaN rather than carrying a stale value forward, `ramp` is then
forced to 0, and `funding_scale` is exactly 1.0: unmodified
`kelly_regime_v4` behaviour, not an invented signal. Same discipline as
the conservative branch, independently re-derived here.

Constraint attacked: COST (funding scales with the signal). Not a
duplicate of the conservative branch (same backlog row, deliberately
different mechanism - see its docstring) nor of R-15/R-16 (those are
measurement rounds, not a strategy) nor of L-04..L-01 (those vary the
*trend* gate; this only ever touches exposure the trend gate has already
granted, and only on funding-paying markets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context

SETTLEMENTS_PER_DAY = 3  # Binance perp funding settles every 8h


def rolling_funding_percentile(funding: pd.Series, window: int,
                                min_periods: int) -> pd.Series:
    """Causal rolling percentile-rank of each settlement vs. its own trailing window.

    At settlement i, ``rolling(window)`` only ever spans settlements
    ``[i - window + 1, i]`` - strictly indices <= i - so this is causal by
    construction: never a value that has not settled yet, and a fresh
    statistic per row rather than one full-series quantile fit once and
    broadcast onto early rows (the lookahead class
    `test_causality_strict.py` / the by-hand check in this experiment's
    runner guard against). Same construction as
    `funding_gate_conservative.rolling_funding_percentile` (independently
    duplicated here rather than imported, so the two experiment files stay
    disjoint per ROUTINE.md's parallel-branch rule) so the two branches'
    gates are built on the same statistic and differ only in how they use it.
    """
    def _pct(arr: np.ndarray) -> float:
        return float((arr <= arr[-1]).mean())

    return funding.sort_index().rolling(window, min_periods=min_periods).apply(
        _pct, raw=True
    )


class FundingGateNovel(KellyRegimeV4):
    """kelly_regime_v4, exposure smoothly damped when rich funding meets a stalled trend.

    `funding` is the raw settlement-level series (`tradebot.data.load_funding`)
    used only to compute the causal gate signal - independent of whatever
    funding series (if any) the backtest itself charges as a cost.

    Parameters
    ----------
    lookback_days : trailing window (days, converted to settlements at
        3/day) the funding percentile is measured against.
    pct_start, pct_full : the ramp spans funding-percentile
        `[pct_start, pct_full]`; below `pct_start` there is no dampening
        at all (`pct_full` defaults inside the top decile, `pct_start`
        below it, per idea (a) - see module docstring).
    mom_days, mom_scale : the short momentum check. `mom_days`-day
        trailing return; `stall` ramps from 0 (return >= 0) to 1 (return
        <= -mom_scale). `mom_days=0` disables the momentum condition
        entirely (stall is always 1) - kept as a switch so the sweep can
        run the ablation that isolates idea (b)'s contribution: with it
        off, this degenerates to a continuous-ramp-on-funding-alone gate.
    max_dampen : the floor is `1 - max_dampen` of full exposure, reached
        only where both ramp and stall are saturated.
    """

    name = "funding_gate_novel"

    def __init__(self, funding: pd.Series | None = None,
                 lookback_days: int = 60,
                 pct_start: float = 0.75, pct_full: float = 0.97,
                 mom_days: float = 3.0, mom_scale: float = 0.015,
                 max_dampen: float = 0.6, **kwargs) -> None:
        super().__init__(**kwargs)
        self._funding = funding
        self.lookback_days = lookback_days
        self.pct_start = pct_start
        self.pct_full = pct_full
        self.mom_days = mom_days
        self.mom_scale = mom_scale
        self.max_dampen = max_dampen

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # unmodified kelly_regime_v4 "target" column

        funding = self._funding
        if funding is None or len(funding) == 0:
            df["funding_scale"] = 1.0
            return df

        window = max(2, int(round(self.lookback_days * SETTLEMENTS_PER_DAY)))
        pct = rolling_funding_percentile(funding, window, min_periods=window)

        # Causal alignment onto the 5m bar grid: each bar sees only the
        # most recently *settled* funding rate as of its own timestamp
        # (backward as-of join), and only within one settlement interval
        # (8h) of it - a bar further than that from the last known
        # settlement (outside 2020-2023, or before the lookback has
        # accumulated) gets NaN rather than a stale carried-forward value.
        left = pd.DataFrame({"ts": df.index})
        right = (pd.DataFrame({"ts": pct.index, "pct": pct.to_numpy()})
                 .dropna(subset=["ts"]).sort_values("ts"))
        right["ts"] = right["ts"].astype(left["ts"].dtype)
        aligned = pd.merge_asof(left, right, on="ts", direction="backward",
                                 tolerance=pd.Timedelta(hours=8))
        pct_on_bars = aligned["pct"].to_numpy()

        # idea (a): continuous ramp starting below the top decile, not a
        # hard decile cutoff - spreads the on/off transition of the
        # conservative branch's gate across many settlements.
        span = max(self.pct_full - self.pct_start, 1e-9)
        ramp = np.clip((pct_on_bars - self.pct_start) / span, 0.0, 1.0)
        ramp = np.where(np.isnan(pct_on_bars), 0.0, ramp)

        # idea (b): only let the ramp bite when short-horizon momentum has
        # stalled/turned - R-16's actual finding (elevated funding is safe
        # while price still confirms; it is the danger sign only once the
        # move it was paid for has stopped), not "funding is high" alone.
        # A horizon of a few days, an order of magnitude below the
        # 20/40/80-day anchors that set the trend vote itself.
        if self.mom_days > 0:
            mom = df["close"].pct_change(int(round(self.mom_days * BARS_PER_DAY))).to_numpy()
            stall = np.clip(-mom / self.mom_scale, 0.0, 1.0)
            stall = np.where(np.isnan(mom), 0.0, stall)
        else:
            # ablation switch: momentum condition off -> dampens on the
            # funding ramp alone, isolating what idea (b) adds.
            stall = np.ones(len(df))

        df["funding_scale"] = 1.0 - self.max_dampen * ramp * stall
        return df

    def on_bar(self, ctx: Context) -> None:
        def _effective(row) -> float:
            t = float(row["target"])
            if ctx.market.pays_funding:
                t *= float(row["funding_scale"])
            return t

        t = _effective(ctx.bar)
        prev = _effective(ctx.prev) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
