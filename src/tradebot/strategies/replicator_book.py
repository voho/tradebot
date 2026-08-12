"""Replicator-dynamics capital reallocation across a chartist/fundamentalist/cash species book."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class ReplicatorBook(Strategy):
    """Reallocate capital across trend, value and cash species with replicator dynamics on their realized fee-adjusted fitness.

    Game-theoretic grounding: Taylor & Jonker (1978, Mathematical
    Biosciences) formalized replicator dynamics — a strategy's population
    share grows in proportion to its fitness edge over the population
    average. Lux & Marchesi (1999, Nature) showed that profit-driven
    contagion, with traders switching between chartist and fundamentalist
    camps, endogenously generates the market regimes each camp exploits;
    a book that reallocates toward the currently profitable species rides
    that regime structure instead of betting on one camp. Brock & Hommes
    (1998, JEDC) proved the intensity of choice beta in the logit switching
    rule is the bifurcation parameter — too high and the population
    overreacts to noisy fitness, which here means churn and fee bleed, so
    beta stays moderate and a share cap stops all-in stampedes. Maynard
    Smith & Price (1973, Nature) introduced the ESS notion under which a
    riskless outside option resists invasion when every active strategy
    loses net of costs: the zero-return cash species is that anchor, and
    weight drains into it (going flat) when no trading species earns its
    own turnover.

    Mechanism: five species emit positions in [-1, 1] — fast and slow
    EMA-cross trend followers, fast and slow vol-normalized value reverters
    (fundamentalists trading the gap between log price and its EMA fair
    value), and cash. Each species' per-bar pnl is its previous-bar signal
    times the log return minus a 5 bp charge on its own turnover; fitness
    is a long EWMA of that net pnl. Weights follow a discrete
    replicator/logit update w *= exp(beta * (F - w.F)), renormalized, with
    a 0.5 share cap and a 0.02 floor acting as the mutation term that
    keeps extinct species revivable. The traded target is the
    weight-blended signal scaled by 0.75, moved only when it drifts more
    than the deadband from the held position, so the target stays
    piecewise-constant and round-trip fees are not fed by rebalancing
    noise.
    """

    name = "replicator_book"
    warmup = 2200

    def __init__(self, beta: float = 5.0, fitness_halflife: int = 1152,
                 deadband: float = 0.10, share_cap: float = 0.5,
                 share_floor: float = 0.02, scale: float = 0.75,
                 species_fee: float = 0.0005) -> None:
        self.beta = beta
        self.fitness_halflife = fitness_halflife
        self.deadband = deadband
        self.share_cap = share_cap
        self.share_floor = share_floor
        self.scale = scale
        self.species_fee = species_fee

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        log_close = np.log(close)
        r = log_close.diff()
        sig1 = r.ewm(span=288, min_periods=250).std()

        # Chartist species: fast and slow EMA-cross trend followers.
        s_tf_f = np.sign(close.ewm(span=48, adjust=False).mean()
                         - close.ewm(span=192, adjust=False).mean())
        s_tf_s = np.sign(close.ewm(span=288, adjust=False).mean()
                         - close.ewm(span=1152, adjust=False).mean())

        # Fundamentalist species: fade the vol-normalized gap between log
        # price and its EMA fair value (log units keep z dimensionless,
        # since sig1 is a log-return volatility).
        zf = (log_close - log_close.ewm(span=288, adjust=False).mean()) \
            / (sig1 * np.sqrt(288.0))
        zs = (log_close - log_close.ewm(span=2016, adjust=False).mean()) \
            / (sig1 * np.sqrt(2016.0))
        s_fun_f = -np.clip(zf / 2.0, -1.0, 1.0)
        s_fun_s = -np.clip(zs / 2.0, -1.0, 1.0)

        n = len(df)
        sig = np.column_stack([
            np.nan_to_num(s_tf_f.to_numpy(), nan=0.0),
            np.nan_to_num(s_tf_s.to_numpy(), nan=0.0),
            np.nan_to_num(s_fun_f.to_numpy(), nan=0.0),
            np.nan_to_num(s_fun_s.to_numpy(), nan=0.0),
            np.zeros(n),  # cash species
        ])

        # Per-species net pnl: yesterday's signal earns today's return,
        # and each species is charged its own turnover.
        r_a = np.nan_to_num(r.to_numpy(), nan=0.0)
        sig_prev = np.empty_like(sig)
        sig_prev[0] = 0.0
        sig_prev[1:] = sig[:-1]
        pnl = sig_prev * r_a[:, None] - self.species_fee * np.abs(sig - sig_prev)

        # Fitness F_i = lam*F_{i-1} + (1-lam)*pnl_i with F_{-1} = 0
        # (the prepended zero row pins the initial condition).
        alpha = 1.0 / float(self.fitness_halflife)
        fit = (pd.DataFrame(np.vstack([np.zeros((1, 5)), pnl]))
               .ewm(alpha=alpha, adjust=False).mean()
               .to_numpy()[1:])

        beta = self.beta
        cap = self.share_cap
        floor = self.share_floor
        scale = self.scale
        deadband = self.deadband
        w = np.full(5, 0.2)
        target = np.empty(n, dtype=np.float64)
        pos = 0.0
        for i in range(n):
            f_i = fit[i]
            w *= np.exp(beta * (f_i - float(w @ f_i)))
            w /= w.sum()
            np.minimum(w, cap, out=w)
            w /= w.sum()
            np.maximum(w, floor, out=w)
            w /= w.sum()
            raw = float(w @ sig[i]) * scale
            if abs(raw - pos) > deadband:
                pos = min(1.0, max(-1.0, raw))
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
