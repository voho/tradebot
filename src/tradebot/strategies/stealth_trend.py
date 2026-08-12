"""Participation-weighted momentum gated by a Kyle-lambda market-depth regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class StealthTrend(Strategy):
    """Follow momentum only when it prints on deep, high-participation bars where informed flow hides.

    Game-theoretic grounding: in Kyle (1985, Econometrica), informed
    traders optimally camouflage in deep markets — informative order flow
    arrives when price impact per unit volume (Kyle's lambda) is LOW, so
    returns printed on heavy volume with little impact carry information,
    while price moves on thin volume are transient liquidity effects that
    revert. Amihud (2002, J. Financial Markets) gives the bar-level lambda
    proxy used here: |return| / dollar volume. Admati & Pfleiderer (1988,
    RFS) show volume clusters endogenously, so both volume and lambda must
    be judged against their own rolling "normal" level rather than in
    absolute terms. Barclay & Warner (1993, JFE) find the informed
    footprint is cumulative drift built from unremarkable trades — hence
    the signal is a slow EWM of participation-weighted returns, not a
    single-bar detector.

    Mechanism: each bar's log return is weighted by capped relative volume
    and discounted by exp(-max(lambda_z, 0)), so high-impact (thin) moves
    are ignored while low-impact, high-participation moves accumulate into
    an EWM momentum score. That score is z-scored against its own rolling
    history and traded with hysteresis: enter scaled positions only when
    |z| clears z_in, the market is currently deep (lambda below its rolling
    median) and recent volatility clears a 4x round-trip-fee hurdle; exit
    to flat when |z| decays below z_out or lambda spikes into a panic
    regime; a deadband suppresses fee-churning small rebalances.
    """

    name = "stealth_trend"
    warmup = 2200

    # z_in/z_out widened after the first paper test (2.3k trades bled to
    # fees): demand rarer momentum extremes and hold them longer.
    def __init__(self, mom_span: int = 144, z_in: float = 1.75, z_out: float = 0.6,
                 z_max: float = 3.5, deadband: float = 0.4, panic_z: float = 1.0,
                 vol_span: int = 288, liq_window: int = 2016, lam_span: int = 36,
                 z_window: int = 864, v_cap: float = 3.0, fee: float = 0.001) -> None:
        self.mom_span = mom_span
        self.z_in, self.z_out, self.z_max = z_in, z_out, z_max
        self.deadband = deadband
        self.panic_z = panic_z
        self.vol_span = vol_span
        self.liq_window = liq_window
        self.lam_span = lam_span
        self.z_window = z_window
        self.v_cap = v_cap
        self.fee = fee

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        volume = df["volume"]
        r = np.log(close).diff()
        sig1 = r.ewm(span=self.vol_span, min_periods=250).std()

        # Amihud (2002) bar-level Kyle-lambda proxy: |return| per unit dollar volume.
        dv = (volume * close).replace(0.0, np.nan)
        lam = r.abs() / dv
        log_lam = np.log(lam.where(lam > 0.0))
        lam_med = lam.rolling(self.liq_window).median()
        lam_z = (log_lam - np.log(lam_med.where(lam_med > 0.0))).ewm(span=self.lam_span).mean()

        # Participation vs its normal level (Admati & Pfleiderer 1988).
        v_med = volume.rolling(self.liq_window).median()
        v_rel = volume / v_med.where(v_med > 0.0)

        # Weight informative bars: high participation, low impact-per-volume.
        w = np.minimum(v_rel, self.v_cap) * np.exp(-np.maximum(lam_z, 0.0))
        mom = (r * w).ewm(span=self.mom_span).mean()
        zm = (mom - mom.rolling(self.z_window).mean()) / mom.rolling(self.z_window).std()

        deep = (lam_z < 0.0).to_numpy()
        panic = (lam_z > self.panic_z).to_numpy()
        # Expected move over the momentum horizon must clear 4x the round-trip fee.
        vol_ok = (sig1.shift(1) * np.sqrt(self.mom_span) > 4.0 * self.fee).to_numpy()

        zm_a = zm.to_numpy()
        n = len(df)
        target = np.empty(n)
        z_in, z_out, z_max, db = self.z_in, self.z_out, self.z_max, self.deadband
        pos = 0.0
        for i in range(n):
            z = zm_a[i]
            if panic[i]:
                pos = 0.0
            elif (z > z_in or z < -z_in) and deep[i] and vol_ok[i]:
                az = z if z > 0.0 else -z
                raw = (az if az < z_max else z_max) / z_max
                if z < 0.0:
                    raw = -raw
                if abs(raw - pos) > db:
                    pos = raw
            elif -z_out < z < z_out:
                pos = 0.0
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
