"""Fictitious-play best response to the market's current minority/majority character via history-conditional return statistics."""

import math

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class GameSwitch(Strategy):
    """Best-respond to whichever game the market is currently playing by trading only history states with significant conditional drift.

    Game-theoretic grounding: in fictitious play (Brown, 1951) a player
    best-responds to the empirical frequency of the opponents' past
    moves; here the "opponent" is the market and the empirical model is
    the exponentially weighted distribution of returns conditional on
    the recent sign history. Marsili (2001, Physica A) showed that one
    and the same market mechanism yields a minority game (contrarian
    payoffs, mean reversion) or a majority game (trend-following
    payoffs, momentum) depending purely on agents' expectations, so the
    profitable best response is not fixed: it switches with the crowd's
    self-fulfilling beliefs. Andersen & Sornette (2003, EPJ B) made the
    same point with the $-game — when agents are paid in realized
    profit rather than minority payoff, they endogenously alternate
    between reversion and momentum. Rather than hard-coding either
    regime, this strategy learns, per history state, which way the
    crowd currently pushes the price and follows the learned
    conditional drift, whatever its sign. Challet, Marsili & Zecchina
    (2000, PRL) supply the trading filter: their predictability order
    parameter H — the average squared conditional mean of the outcome
    given the history state — separates the symmetric (unpredictable,
    H = 0) phase from the asymmetric (exploitable, H > 0) phase, so
    positions are only taken while the running H estimate is positive
    and only in states whose conditional drift is statistically and
    economically significant.

    Mechanism: the last ``memory`` = 5 return signs form one of 32
    history states mu. Per state, an O(1) online update maintains an
    exponentially weighted mean return mu_r[mu], mean square m2_r[mu]
    and effective visit count neff[mu] (decay lam = 1 - 1/halflife per
    visit). Each bar the state that conditioned the just-realized
    return is updated, H is tracked as an EWMA (span = halflife) of
    that state's squared conditional mean, and the register rolls to
    the new state mu'. Entry requires four strict gates: H above
    ``h_min``, neff[mu'] > ``n_min``, a t-statistic
    |mu_r| / sqrt(var/neff) above ``t_gate``, and the drift beating the
    round-trip fee over a ``hold_bars`` horizon. Size scales with the
    excess t-stat, capped at 1. Exits are loose — flatten when the
    current state's drift opposes the position, or when H collapses;
    otherwise hold (inertia). A resize deadband keeps the target
    piecewise-constant so fees do not eat the edge.
    """

    name = "game_switch"
    warmup = 3000

    def __init__(self, memory: int = 5, halflife: int = 2880, n_min: int = 200,
                 t_gate: float = 2.5, hold_bars: int = 24, h_min: float = 1e-9,
                 fee_rate: float = 0.001, prior_var: float = 1e-6,
                 resize_band: float = 0.25) -> None:
        self.memory = memory
        self.halflife = halflife
        self.n_min = n_min
        self.t_gate = t_gate
        self.hold_bars = hold_bars
        self.h_min = h_min
        self.fee_rate = fee_rate
        self.prior_var = prior_var
        self.resize_band = resize_band

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        n_states = 1 << self.memory
        mask = n_states - 1
        lam = 1.0 - 1.0 / float(self.halflife)
        alpha_h = 2.0 / (float(self.halflife) + 1.0)
        gate = self.t_gate
        n_min = float(self.n_min)
        hurdle = 2.0 * self.fee_rate
        horizon = float(self.hold_bars)
        h_min = self.h_min

        r_l = np.log(df["close"]).diff().to_numpy().tolist()

        mu_r = [0.0] * n_states
        m2_r = [self.prior_var] * n_states
        neff = [0.0] * n_states

        target = np.zeros(n, dtype=np.float64)
        mu = 0
        h_est = 0.0
        pos = 0.0
        for i in range(1, n):
            ri = r_l[i]
            if ri != ri:  # NaN return: no update, hold current target
                target[i] = pos
                continue

            # Update the state that conditioned bar i's return (register
            # state BEFORE bar i), then track the predictability H.
            ne = lam * neff[mu] + 1.0
            neff[mu] = ne
            a = 1.0 / ne
            mr = mu_r[mu] + a * (ri - mu_r[mu])
            mu_r[mu] = mr
            m2_r[mu] += a * (ri * ri - m2_r[mu])
            h_est += alpha_h * (mr * mr - h_est)

            # Roll the sign-history register: this is the state that
            # conditions the NEXT bar's return.
            mu = ((mu << 1) | (1 if ri > 0.0 else 0)) & mask

            if h_est < h_min:
                pos = 0.0  # symmetric phase: nothing to exploit
            else:
                drift = mu_r[mu]
                ne2 = neff[mu]
                var = m2_r[mu] - drift * drift
                if var < 1e-12:
                    var = 1e-12
                tstat = drift / math.sqrt(var / (ne2 if ne2 > 1.0 else 1.0))
                at = abs(tstat)
                if ne2 > n_min and at > gate and abs(drift) * horizon > hurdle:
                    size = (at - gate) / gate
                    if size > 1.0:
                        size = 1.0
                    t_new = size if drift > 0.0 else -size
                    if (pos == 0.0 or (t_new > 0.0) != (pos > 0.0)
                            or abs(t_new - pos) > self.resize_band):
                        pos = t_new
                elif pos != 0.0 and drift * pos < 0.0:
                    pos = 0.0  # loose exit: state drift opposes the position
                # else: inertia — keep the current position
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
