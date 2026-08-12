"""Back-run persistent informed order flow recovered from bars via Bulk Volume Classification."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class CamouflageFlow(Strategy):
    """Follow persistent informed order flow recovered from bars via Bulk Volume Classification.

    Game-theoretic grounding: in Kyle (1985, Econometrica) informed traders
    camouflage by splitting their orders inside noise-trader volume, so private
    information leaks into prices slowly — signed order flow is persistent and
    the induced drift is durable rather than instantaneous. Easley, Lopez de
    Prado & O'Hara (2012, RFS) show that signed flow can be recovered from bar
    data alone via Bulk Volume Classification: buy volume = V * Phi(dp/sigma),
    the basis of VPIN. Yang & Zhu (2020, RFS) prove that back-running such
    detected flow is equilibrium-consistent — a second trader who learns the
    insider's direction from past order flow profits by trading along with it.

    Mechanism: per bar, BVC signs volume with a logistic Phi of the
    vol-standardized return; a 3h rolling flow imbalance is z-scored against
    its 3-day history. Entries require an extreme flow z-score, flow toxicity
    (12h VPIN-like ratio) above its 1-week median — i.e. flow looks informed,
    not noise — and projected 6h volatility clearing a multiple of round-trip
    fees. A deadband plus wide in/out thresholds keeps the target
    piecewise-constant so fees do not eat the edge.
    """

    name = "camouflage_flow"
    warmup = 2200

    def __init__(self, flow_window: int = 36, tox_window: int = 144, z_in: float = 1.5,
                 z_out: float = 0.5, z_max: float = 3.0, deadband: float = 0.25,
                 fee_hurdle: float = 0.004) -> None:
        self.flow_window = flow_window
        self.tox_window = tox_window
        self.z_in, self.z_out, self.z_max = z_in, z_out, z_max
        self.deadband = deadband
        self.fee_hurdle = fee_hurdle

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        volume = df["volume"]

        r = np.log(close).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        zr = (r / sig1.shift(1)).clip(-4.0, 4.0)

        # BVC buy fraction via logistic approximation of the normal CDF.
        phi = 1.0 / (1.0 + np.exp(-1.702 * zr))
        sflow = volume * (2.0 * phi - 1.0)

        ofi = sflow.rolling(self.flow_window).sum() / volume.rolling(self.flow_window).sum()
        tox = sflow.abs().rolling(self.tox_window).sum() / volume.rolling(self.tox_window).sum()
        toxm = tox.rolling(2016).median()
        zf = (ofi - ofi.rolling(864).mean()) / ofi.rolling(864).std()
        vol_ok = sig1.shift(1) * np.sqrt(72.0) > self.fee_hurdle

        zf_a = zf.to_numpy()
        tox_a = tox.to_numpy()
        toxm_a = toxm.to_numpy()
        ok_a = vol_ok.to_numpy()
        valid = np.isfinite(zf_a) & np.isfinite(tox_a) & np.isfinite(toxm_a)

        n = len(df)
        target = np.empty(n, dtype=np.float64)
        span = self.z_max - self.z_in
        pos = 0.0
        for i in range(n):
            if not valid[i]:
                pos = 0.0
            else:
                z = zf_a[i]
                az = abs(z)
                if az > self.z_in and tox_a[i] > toxm_a[i] and ok_a[i]:
                    raw = (1.0 if z > 0.0 else -1.0) * min(1.0, (az - self.z_in) / span)
                    if abs(raw - pos) > self.deadband:
                        pos = raw
                elif az < self.z_out:
                    pos = 0.0
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
