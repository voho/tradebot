"""Grand-canonical minority-game agent population trained online on the binarized return series; trade its abstention-filtered vote."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class MinorityOracle(Strategy):
    """Trade the abstention-filtered vote of a grand-canonical minority game trained online on binarized returns.

    Game-theoretic grounding: in the minority game of Challet & Zhang
    (1997, Physica A) a population of inductive agents maps the last m
    binary outcomes to an action via fixed random strategy tables and
    plays its best-scoring table. Savit, Manuca & Riolo (1999, PRL)
    showed the game's behavior is controlled by alpha = 2^m / N, and
    Challet, Marsili & Zecchina (2000, PRL) located the phase transition
    at alpha_c ~ 0.34: above it the crowd is too sparse to arbitrage all
    structure away, so the binary series retains exploitable
    predictability. This strategy sits deliberately in that
    information-rich phase (P = 64 states, K = 65 agents, alpha ~ 1).
    Following Johnson, Lamper, Jefferies, Hart & Howison (2001,
    Physica A), the game is trained on the real (binarized) return
    series rather than on its own endogenous history, turning the agent
    population into an online ensemble predictor of the next return
    sign. The grand-canonical variant of Jefferies, Hart, Hui & Johnson
    (2001, EPJ B) lets agents abstain unless their best table's excess
    hit rate clears a confidence threshold — an abstention rule that
    acts exactly like a transaction-cost filter. Lamper, Howison &
    Johnson (2002, PRL) showed that strong agreement across the active
    ensemble precedes large moves, so positions are taken only when the
    net vote of active agents is decisive.

    Mechanism: each of K agents holds S random +/-1 tables over the
    P = 2^m states of the last m return signs. Every bar, each table's
    hit rate against the realized sign is tracked with an EWMA score
    (half-life ~ score_halflife bars); an agent is active only when its
    best table's excess accuracy exceeds eps. The population's vote is
    the active agents' predictions summed over K. Enter (long or short)
    when |vote| > v_in and the implied per-bar edge — twice the mean
    active excess accuracy times the EWMA return volatility — beats the
    round-trip fee over a 12-bar horizon; exit when |vote| < v_out or
    the vote flips sign. The wide v_in/v_out hysteresis band keeps the
    target piecewise-constant so fees do not eat the signal.
    """

    name = "minority_oracle"
    warmup = 2500

    # Defaults retuned after the first paper test: the original gates
    # (eps=0.02, v_in=0.40, edge_horizon=12) never fired on real data -
    # a 12-bar horizon needs a ~60% hit rate to clear round-trip fees.
    def __init__(self, memory: int = 6, agents: int = 65, tables: int = 2,
                 eps: float = 0.01, v_in: float = 0.15, v_out: float = 0.05,
                 seed: int = 7, score_halflife: int = 4096, vol_span: int = 288,
                 fee_rate: float = 0.001, edge_horizon: int = 96) -> None:
        self.memory = memory
        self.agents = agents
        self.tables = tables
        self.eps = eps
        self.v_in, self.v_out = v_in, v_out
        self.seed = seed
        self.score_halflife = score_halflife
        self.vol_span = vol_span
        self.fee_rate = fee_rate
        self.edge_horizon = edge_horizon

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        m, big_k, big_s = self.memory, self.agents, self.tables
        big_p = 1 << m
        mask = big_p - 1
        lam = 1.0 - 1.0 / float(self.score_halflife)
        c = 1.0 - lam

        rng = np.random.default_rng(self.seed)
        g = rng.choice([-1, 1], size=(big_k, big_s, big_p))
        g_f = g.astype(np.float64)                      # (K, S, P)
        g_flat = g.reshape(big_k * big_s, big_p)        # row = k*S + s

        r = np.log(df["close"]).diff()
        sig = r.ewm(span=self.vol_span, min_periods=self.vol_span).std().to_numpy()
        r_np = r.to_numpy()
        s_up = r_np > 0.0                               # r == 0 (and NaN) counts as -1
        bits = s_up.astype(np.int64)
        if n > 0:
            bits[0] = 0                                 # bar 0 never enters the game

        # State register mu, vectorized: mu_after[i] = ((mu_after[i-1]<<1)|bit_i)&mask
        # with mu_after[0] = 0 == packing the last m bits (zero-padded at the start).
        mu_after = np.zeros(n, dtype=np.int64)
        for k in range(min(m, n)):
            mu_after[k:] += bits[: n - k] << k
        mu_after &= mask
        mu_before = np.empty(n, dtype=np.int64)
        if n > 0:
            mu_before[0] = 0
            mu_before[1:] = mu_after[:-1]

        # Per-bar table hit: (G[:, :, mu_before] == s_i), looked up by a combined
        # (sign, state) code into a precomputed (K*S, 2P) match table.
        match_tab = np.concatenate([(g_flat == -1), (g_flat == 1)], axis=1).astype(np.float64)
        code = mu_before + big_p * bits

        vote = np.zeros(n, dtype=np.float64)
        mean_conf = np.zeros(n, dtype=np.float64)
        n_active = np.zeros(n, dtype=np.int64)

        # EWMA score recursion Q_t = lam*Q_{t-1} + c*match_t, run in chunks:
        # inside a chunk Q_t = lam^{t+1} Q_prev + c*lam^t * cumsum(lam^{-u} match_u).
        # All terms are >= 0 so the scaled cumsum is exact (no cancellation) and
        # each bar's value depends only on bars <= t.
        chunk = 4096
        jj = np.arange(chunk, dtype=np.float64)
        lam_pow = lam ** jj
        lam_inv = lam ** (-jj)
        q_prev = np.zeros(big_k * big_s, dtype=np.float64)

        for start in range(1, n, chunk):
            end = min(start + chunk, n)
            lc = end - start
            mc = match_tab[:, code[start:end]]                        # (K*S, Lc)
            acc = np.cumsum(mc * lam_inv[:lc], axis=1)
            q = lam_pow[:lc] * (lam * q_prev[:, None] + c * acc)      # (K*S, Lc)
            q_prev = q[:, -1].copy()

            q3 = q.reshape(big_k, big_s, lc)
            best = q3.argmax(axis=1)                                  # (K, Lc)
            q_best = np.take_along_axis(q3, best[:, None, :], axis=1)[:, 0, :]
            conf = q_best - 0.5
            active = conf > self.eps                                  # (K, Lc)

            g_cols = g_f[:, :, mu_after[start:end]]                   # (K, S, Lc)
            g_best = np.take_along_axis(g_cols, best[:, None, :], axis=1)[:, 0, :]

            vote[start:end] = (g_best * active).sum(axis=0) / float(big_k)
            na = active.sum(axis=0)
            n_active[start:end] = na
            csum = (conf * active).sum(axis=0)
            mean_conf[start:end] = np.where(na > 0, csum / np.maximum(na, 1), 0.0)

        edge = 2.0 * mean_conf * sig
        edge[n_active == 0] = 0.0

        # Hysteresis state machine (scalar, cheap): enter on a decisive vote whose
        # implied edge over edge_horizon bars beats the round-trip fee; exit when
        # the vote decays inside the band or flips sign.
        target = np.zeros(n, dtype=np.float64)
        vote_l = vote.tolist()
        edge_l = edge.tolist()
        hurdle = 2.0 * self.fee_rate
        horizon = float(self.edge_horizon)
        pos = 0.0
        for i in range(1, n):
            v = vote_l[i]
            if pos == 0.0:
                if abs(v) > self.v_in and edge_l[i] * horizon > hurdle:
                    pos = max(-1.0, min(1.0, v))
            elif abs(v) < self.v_out or (v > 0.0) != (pos > 0.0):
                pos = 0.0
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
