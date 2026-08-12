"""Fade forced-liquidation overshoots after flow exhaustion, gated by spread costs."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class OvershootFade(Strategy):
    """Fade forced-liquidation overshoots once the aggressive flow driving them is exhausted.

    Game-theoretic grounding: Brunnermeier & Pedersen (2005, J. Finance,
    "Predatory Trading") show that around a distressed liquidator, strategic
    traders first trade in the same direction and withdraw liquidity, so the
    equilibrium price path overshoots fundamentals and then recovers; the
    profitable side of that equilibrium is supplying liquidity at the
    overshoot extreme, after the forced flow is spent. Glosten & Milgrom
    (1985, JFE) add the adverse-selection condition: fading a move is only
    safe when the counterparty flow is uninformed, so the entry demands that
    directional signed flow has already dried up (exhaustion) — persistent
    one-sided flow means information, not distress. Corwin & Schultz (2012,
    J. Finance) provide a high-low based spread estimator that sets the
    minimum worthwhile retrace: the expected reversion must clear round-trip
    fees plus the prevailing spread or the trade is not worth taking.

    Mechanism: an overshoot event is a 1h move beyond ``z_event``
    vol-standardized sigmas on climactic volume (30m volume > ``v_event`` x
    its 24h mean) and range expansion (bar range > ``rng_event`` x its 24h
    median). Signed flow is recovered per bar via Bulk Volume Classification
    (logistic CDF approximation); the 30m flow imbalance in the move's
    direction must sit below ``exhaust``. Expected reversion is ``retrace``
    of the overshoot; it must exceed fees plus the smoothed Corwin-Schultz
    spread. Then fade the move, sized by severity, and exit on a retrace
    target, a one-sigma-1h volatility stop, or a 6h time stop. Events are
    rare (a few per month), so the target is piecewise-constant and flat
    almost always — fees stay negligible by construction.
    """

    name = "overshoot_fade"
    warmup = 600

    def __init__(self, k_window: int = 12, z_event: float = 3.0, v_event: float = 3.0,
                 rng_event: float = 2.0, exhaust: float = 0.10, retrace: float = 0.25,
                 time_stop: int = 72, z_size: float = 5.0, tp_frac: float = 0.5,
                 fee_floor: float = 0.001, cs_span: int = 144) -> None:
        self.k_window = k_window
        self.z_event, self.v_event, self.rng_event = z_event, v_event, rng_event
        self.exhaust = exhaust
        self.retrace = retrace
        self.time_stop = time_stop
        self.z_size = z_size
        self.tp_frac = tp_frac
        self.fee_floor = fee_floor
        self.cs_span = cs_span

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"].clip(lower=1e-12)
        volume = df["volume"]
        k = self.k_window

        r = np.log(close).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        sig_k = sig1.shift(1) * np.sqrt(k)
        z_mv = (np.log(close / close.shift(k)) / sig_k).replace([np.inf, -np.inf], np.nan)

        v_clim = (volume.rolling(6).mean() / volume.rolling(288).mean()
                  ).replace([np.inf, -np.inf], np.nan)
        hl = np.log(high / low)
        rng_ratio = (hl / hl.rolling(288).median()).replace([np.inf, -np.inf], np.nan)
        event = (z_mv.abs() > self.z_event) & (v_clim > self.v_event) & (rng_ratio > self.rng_event)

        # BVC signed flow via logistic approximation of the normal CDF.
        zr = (r / sig1.shift(1)).clip(-4.0, 4.0)
        phi = 1.0 / (1.0 + np.exp(-1.702 * zr))
        sflow = volume * (2.0 * phi - 1.0)
        ofi_fast = (sflow.rolling(6).sum() / volume.rolling(6).sum()
                    ).replace([np.inf, -np.inf], np.nan)

        # Corwin-Schultz two-bar spread estimate, floored at zero and smoothed.
        kcs = 3.0 - 2.0 * np.sqrt(2.0)
        hl2 = hl ** 2
        beta = hl2 + hl2.shift(1)
        gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / kcs - np.sqrt(gamma / kcs)
        spread = (2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))).clip(lower=0.0)
        cs = spread.ewm(span=self.cs_span).mean()

        exp_rev = self.retrace * z_mv.abs() * sig_k
        gate = exp_rev > (self.fee_floor + cs)

        z_a = z_mv.to_numpy()
        ofi_a = ofi_fast.to_numpy()
        event_a = event.to_numpy()
        gate_a = gate.to_numpy()
        exp_rev_a = exp_rev.to_numpy()
        stop_a = (sig1 * np.sqrt(k)).to_numpy()
        logc = np.log(close.to_numpy(dtype=np.float64))
        with np.errstate(invalid="ignore"):
            exh_a = np.sign(z_a) * ofi_a < self.exhaust

        n = len(df)
        target = np.empty(n, dtype=np.float64)
        pos = 0.0
        log_entry = 0.0
        tp = 0.0
        age = 0
        for i in range(n):
            if pos == 0.0:
                if event_a[i] and exh_a[i] and gate_a[i]:
                    z = z_a[i]
                    pos = (-1.0 if z > 0.0 else 1.0) * min(1.0, abs(z) / self.z_size)
                    log_entry = logc[i]
                    tp = self.tp_frac * exp_rev_a[i]
                    age = 0
            else:
                age += 1
                sgn = 1.0 if pos > 0.0 else -1.0
                pnl = sgn * (logc[i] - log_entry)
                stop = stop_a[i]
                if pnl >= tp or (np.isfinite(stop) and pnl <= -stop) or age >= self.time_stop:
                    pos = 0.0
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
