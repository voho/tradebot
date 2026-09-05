"""R-189: ten finite, causal game-learning adaptations for a 4h expert council.

These are bar-data research candidates, not replications of market microstructure
games or claims that the papers' regret/equilibrium guarantees survive trading.
All defaults are fixed before evaluation. Six experts propose equity-notional
fractions: clipped KellyRegimeV4, intraday trend, reversion, breakout, buy, cash.
The three fast experts are price-derived hypotheses, not observed trader types.

An expert's previous close decision enters at the subsequent open. Its reward
is marked at the current decision close, with 0.0011 one-way target-turnover
cost, then divided by 0.02 and clipped to [-1,1]. Only completed rounds train
the learner. This is a proxy council payoff: funding, execution deadbands,
notional drift and close-to-next-open gaps are measured by the outer broker,
not reproduced inside the learner. Signals are scheduled at 00/04/08/12/16/20
UTC, allowing at most six long-only rebalance orders per day. A 0.05 target
and actual-notional band reduces churn without imposing a minimum trade count.
"""

from collections import deque

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context, Strategy


def _softmax(x: np.ndarray) -> np.ndarray:
    w = np.exp(x - np.max(x))
    return w / w.sum()


def _normalize(w: np.ndarray) -> np.ndarray:
    total = w.sum()
    return w / total if total > 1e-14 else np.full(len(w), 1.0 / len(w))


class _IntradayCouncil(Strategy):
    warmup = KellyRegimeV4.warmup
    expert_names = ("kelly_regime_v4", "trend", "reversion", "breakout", "buy", "cash")

    def __init__(self, fee_rate: float = 0.0011, deadband: float = 0.05) -> None:
        if not np.isfinite(fee_rate) or fee_rate < 0:
            raise ValueError("fee_rate must be finite and nonnegative")
        if not np.isfinite(deadband) or not 0 <= deadband < 1:
            raise ValueError("deadband must be finite and in [0,1)")
        self.fee_rate, self.deadband = fee_rate, deadband

    @staticmethod
    def _experts(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        close = df["close"]
        fast = close.ewm(span=48, min_periods=48).mean()
        slow = close.ewm(span=288, min_periods=288).mean()
        sig = np.log(close).diff().ewm(span=288, min_periods=288).std()
        width = (sig * np.sqrt(48)).clip(lower=0.001)
        trend = (0.5 + np.log(fast / slow) / (4 * width)).clip(0, 1)
        reversion = (0.5 - np.log(close / fast) / (4 * width)).clip(0, 1)
        high = df["high"].rolling(288, min_periods=288).max().shift(1)
        low = df["low"].rolling(288, min_periods=288).min().shift(1)
        breakout = ((close - low) / (high - low).clip(lower=1e-12)).clip(0, 1)
        kelly = KellyRegimeV4().prepare(df.copy())["target"].to_numpy().clip(0, 1)
        # An expert that cannot yet be deployed must not earn hypothetical PnL.
        kelly[:KellyRegimeV4.warmup] = 0.0
        a = np.column_stack((kelly, trend, reversion, breakout,
                             np.ones(len(df)), np.zeros(len(df))))
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        awake = np.ones_like(a, dtype=bool)
        awake[:, 0] = np.arange(len(df)) >= KellyRegimeV4.warmup
        strength = np.abs(np.log(fast / slow)).to_numpy()
        awake[:, 1] = strength >= width.to_numpy() * 0.25
        awake[:, 2] = strength < width.to_numpy()
        awake[:, 3] = ((breakout >= 0.75) | (breakout <= 0.25)).to_numpy()
        return a, awake

    def _reset(self) -> None:
        self.weights = np.full(6, 1.0 / 6)
        self._played_target = 0.0
        self._played_turnover = 0.0
        self._round_return = 0.0

    def _update(self, gain: np.ndarray, played: np.ndarray, awake: np.ndarray) -> None:
        raise NotImplementedError

    def _distribution(self, actions: np.ndarray, awake: np.ndarray) -> np.ndarray:
        return self.weights

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        self._reset()  # Repeated live prepare() calls must start from the same prior.
        a, awake = self._experts(df)
        ts = pd.DatetimeIndex(df.index)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        schedule = (ts.hour % 4 == 0) & (ts.minute == 0) & (ts.second == 0)
        slots = np.flatnonzero(schedule & (np.arange(len(df)) >= 288))
        close, opens = df["close"].to_numpy(), df["open"].to_numpy()
        target = np.zeros(len(df))
        previous_slot = None
        previous_actions = np.zeros(6)
        turnover = np.zeros(6)
        played = self.weights.copy()
        pos = 0.0
        for i in slots:
            if previous_slot is not None:
                # i is already closed: the previous signal's next open is known.
                entry = opens[previous_slot + 1]
                self._round_return = float(close[i] / entry - 1.0)
                gain = np.clip((previous_actions * self._round_return
                                - self.fee_rate * turnover) / 0.02, -1.0, 1.0)
                self._update(gain, played, awake[previous_slot])
                target[previous_slot:i] = pos
            played = self._distribution(a[i], awake[i]).copy()
            desired = float(np.clip(played @ a[i], 0.0, 1.0))
            old_pos = pos
            if abs(desired - pos) >= self.deadband or (desired == 0 and pos != 0):
                pos = desired
            self._played_target = pos
            self._played_turnover = abs(pos - old_pos)
            turnover = np.abs(a[i] - previous_actions)
            previous_actions = a[i]
            previous_slot = i
        if previous_slot is not None:
            target[previous_slot:] = pos
        df["target"] = target
        df["game_decision"] = schedule
        return df

    def on_bar(self, ctx: Context) -> None:
        if not bool(ctx.bar["game_decision"]) or ctx.equity <= 0:
            return
        target = float(ctx.bar["target"])
        held = ctx.position * ctx.close / ctx.equity
        if abs(target - held) >= self.deadband or (target == 0 and held != 0):
            ctx.order_notional(target)


@register
class CautiousOptimism(_IntradayCouncil):
    """Cautious optimistic entropy pacing over completed expert rewards.

    Eq. 9.2, https://arxiv.org/html/2506.05005v2 (2025): cumulative
    centered regret R, optimistic r=R+last_regret, p=softmax(lambda*r).
    Lambda maximizes alpha*log(lambda)+logsumexp(lambda*r), lambda<=0.5;
    alpha=4*log(6)^2. Slows when all optimistic regrets are negative.
    """

    name = "cautious_optimism"

    def _reset(self) -> None:
        super()._reset()
        self.regret = np.zeros(6)

    def _update(self, gain, played, awake) -> None:
        r = gain - played @ gain
        self.regret += r
        optimistic = self.regret + r
        alpha = 4 * np.log(6) ** 2
        lo, hi = 1e-12, 0.5
        if alpha / hi + _softmax(hi * optimistic) @ optimistic < 0:
            for _ in range(28):
                mid = (lo + hi) / 2
                if alpha / mid + _softmax(mid * optimistic) @ optimistic > 0:
                    lo = mid
                else:
                    hi = mid
        self.weights = _softmax(hi * optimistic)


@register
class SquintCouncil(_IntradayCouncil):
    """Shared-variance Squint with numerical learning-rate integration.

    https://arxiv.org/html/2603.03409v1 (2026): p_i proportional to
    integral exp(eta*R_i-eta^2*V) d eta, eta in [0,0.5]. The next
    shared variance increment v solves sum_i integral eta*exp(...)
    *(v-r_i^2) d eta=0. Twelve-point Gauss-Legendre quadrature is a
    finite approximation. Gains are halved so centered regrets |r|<=1.
    """

    name = "squint_council"

    def _reset(self) -> None:
        super()._reset()
        self.regret = np.zeros(6)
        self.variance = 0.0
        nodes, weights = np.polynomial.legendre.leggauss(12)
        self.etas = (nodes + 1) / 4
        self.quadrature = weights / 4

    def _integrals(self, variance: float, derivative: bool = False) -> np.ndarray:
        z = self.regret[:, None] * self.etas - variance * self.etas ** 2
        factors = self.quadrature * (self.etas if derivative else 1)
        return np.exp(z - z.max()) @ factors

    def _update(self, gain, played, awake) -> None:
        r = (gain - played @ gain) / 2
        self.regret += r
        lo, hi = 0.0, 1.0
        for _ in range(22):
            v = (lo + hi) / 2
            q = self._integrals(self.variance + v, derivative=True)
            if q @ (v - r * r) > 0:
                hi = v
            else:
                lo = v
        self.variance += (lo + hi) / 2
        self.weights = _normalize(self._integrals(self.variance))


@register
class NormalHedgeCouncil(_IntradayCouncil):
    """Brownian-heat NormalHedge with a solved implicit time parameter.

    https://arxiv.org/html/2602.08151v1 (2026), Section 3.1:
    sum exp([R_i]+^2/(2*t))/sqrt(t)=6/sqrt(t0), weights proportional
    to [R_i]+*exp([R_i]+^2/(2*t)). Practical t0=1 is fixed here;
    it does not meet the paper's sufficient theorem initialization.
    """

    name = "normalhedge_council"

    def _reset(self) -> None:
        super()._reset()
        self.regret = np.zeros(6)
        self.time = 1.0

    def _update(self, gain, played, awake) -> None:
        self.regret += gain - played @ gain
        x = np.maximum(self.regret, 0)

        def potential(t):
            z = x * x / (2 * t)
            return z.max() + np.log(np.exp(z - z.max()).sum()) - 0.5 * np.log(t)

        lo, hi = self.time, self.time
        threshold = np.log(6)
        while potential(hi) > threshold:
            hi *= 2
        for _ in range(28):
            mid = (lo + hi) / 2
            if potential(mid) > threshold:
                lo = mid
            else:
                hi = mid
        self.time = hi
        z = x * x / (2 * hi)
        self.weights = _normalize(x * np.exp(z - z.max()))


@register
class SwapRegretCouncil(_IntradayCouncil):
    """Internal regret matching through a stationary expert-switching chain.

    Finite adaptation of Hart-Mas-Colell regret matching and the
    external-to-internal reduction revisited in modern swap-regret work:
    https://arxiv.org/abs/2310.19786 (2024). R_ij accumulates
    p_i*(g_j-g_i); positive off-diagonal regrets form transition rates.
    The next mix is a stationary distribution, with 1e-6 uniform
    tremble to resolve reducible chains. No finite-game guarantee is
    claimed for this perturbed, transaction-cost trading adaptation.
    """

    name = "swap_regret_council"

    def _reset(self) -> None:
        super()._reset()
        self.swap_regret = np.zeros((6, 6))

    def _update(self, gain, played, awake) -> None:
        self.swap_regret += played[:, None] * (gain[None, :] - gain[:, None])
        rates = np.maximum(self.swap_regret, 0)
        np.fill_diagonal(rates, 0)
        scale = max(float(rates.sum(axis=1).max()), 1.0)
        transition = rates / scale
        np.fill_diagonal(transition, 1 - transition.sum(axis=1))
        transition = (1 - 1e-6) * transition + 1e-6 / 6
        system = transition.T - np.eye(6)
        system[-1] = 1
        rhs = np.zeros(6)
        rhs[-1] = 1
        self.weights = _normalize(np.maximum(np.linalg.solve(system, rhs), 0))


@register
class BlackwellCouncil(_IntradayCouncil):
    """Blackwell-inspired approachability of reward, risk and turnover budgets.

    https://arxiv.org/abs/2406.07585 (2024). Positive cumulative
    constraint deficits select the next exposure by minimizing their
    dot product with predicted constraint violations on a 21-point
    exposure grid. Constraints: match clipped Kelly's net reward,
    squared normalized return <=0.0625, target turnover <=0.25/round.
    A 42-round EW return model predicts violations; these estimated
    constraints are not known to be jointly approachable.
    """

    name = "blackwell_council"

    def _reset(self) -> None:
        super()._reset()
        self.deficit = np.array([1.0, 0.0, 0.0])
        self.mean = 0.0
        self.second = 0.0625

    def _update(self, gain, played, awake) -> None:
        r = np.clip(self._round_return / 0.02, -1, 1)
        realized = self._played_target * r - self.fee_rate / 0.02 * self._played_turnover
        violation = np.array([gain[0] - realized, realized ** 2 - 0.0625,
                              self._played_turnover - 0.25])
        self.deficit = np.maximum(self.deficit + violation, 0)
        self.mean += (r - self.mean) / 42
        self.second += (r * r - self.second) / 42

    def _distribution(self, actions, awake):
        grid = np.linspace(0, 1, 21)
        turnover = np.abs(grid - self._played_target)
        net = grid * self.mean - self.fee_rate / 0.02 * turnover
        benchmark = actions[0] * self.mean
        constraints = np.column_stack((benchmark - net,
                                       grid * grid * self.second - 0.0625,
                                       turnover - 0.25))
        score = constraints @ self.deficit
        x = grid[np.argmin(score)]
        self.weights = np.array([0, 0, 0, 0, x, 1 - x], dtype=float)
        return self.weights


class _ScenarioCouncil(_IntradayCouncil):
    """Historical model scenarios, not observations of strategic opponents."""

    def _reset(self) -> None:
        super()._reset()
        self.history = deque(maxlen=180)

    def _matrix(self, gain: np.ndarray) -> np.ndarray:
        self.history.append(gain.copy())
        history = np.asarray(self.history)
        return np.column_stack([history[-h:].mean(axis=0) for h in (6, 42, 180)])


@register
class MinimaxCouncil(_ScenarioCouncil):
    """Pure expert maximin against three completed-payoff horizon scenarios.

    Contemporary robust-portfolio context (2025):
    https://www.sciencedirect.com/science/article/pii/S0377221724006933.
    Select argmax_i min_s M_is; M contains fee-charged expert mean
    gains over the last 6, 42 and 180 completed rounds. Equal robust
    maximizers share capital. This deliberately finite pure-action
    restriction is not a reproduction of the paper's robust program.
    """

    name = "minimax_council"

    def _update(self, gain, played, awake) -> None:
        worst = self._matrix(gain).min(axis=1)
        self.weights = _normalize((worst == worst.max()).astype(float))


@register
class NashCouncil(_ScenarioCouncil):
    """Finite Nash bargaining over historical model-committee surpluses.

    Portfolio bargaining context (2024 revision):
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3884009.
    Candidates are six pure experts and their 15 equal pair mixtures.
    Each horizon committee has utility M_s*w - 0.02*||w||^2 and
    disagreement min_candidate utility - 0.01. Choose the candidate
    maximizing sum_s log(utility_s-disagreement_s). The committee,
    concentration penalty and disagreement choices are our adaptation.
    """

    name = "nash_council"

    def _reset(self) -> None:
        super()._reset()
        eye = np.eye(6)
        self.candidates = np.vstack([eye] + [(eye[i] + eye[j])[None, :] / 2
                                             for i in range(6) for j in range(i)])

    def _update(self, gain, played, awake) -> None:
        matrix = self._matrix(gain)
        utility = self.candidates @ matrix - 0.02 * np.sum(self.candidates ** 2, axis=1)[:, None]
        disagreement = utility.min(axis=0) - 0.01
        value = np.log(utility - disagreement).sum(axis=1)
        self.weights = self.candidates[np.argmax(value)].copy()


@register
class QRECouncil(_ScenarioCouncil):
    """Entropy-regularized trader versus horizon-model zero-sum game.

    https://arxiv.org/abs/2105.15186 (extragradient entropy games),
    https://arxiv.org/abs/2507.09928 (2025 QRE context). Three
    historical horizon models oppose six experts with payoff M.
    Approximate p=softmax(Mq/0.05), q=softmax(-M.T*p/0.05) with 32
    damped logit-response predictor/corrector steps. This finite QRE
    adaptation has no certified equilibrium residual and does not
    reproduce the cited multiplicative-policy extragradient method.
    """

    name = "qre_council"

    def _update(self, gain, played, awake) -> None:
        matrix = self._matrix(gain)
        p, q = self.weights.copy(), np.full(3, 1 / 3)
        # A payoff-dependent step stabilizes the fixed-point approximation.
        step = min(0.5, 0.025 / max(np.linalg.norm(matrix), 0.025))
        for _ in range(32):
            pp = (1 - step) * p + step * _softmax(matrix @ q / 0.05)
            qq = (1 - step) * q + step * _softmax(-matrix.T @ p / 0.05)
            p = (1 - step) * p + step * _softmax(matrix @ qq / 0.05)
            q = (1 - step) * q + step * _softmax(-matrix.T @ pp / 0.05)
        self.weights = _normalize(p)


@register
class SleepingCouncil(_IntradayCouncil):
    """Confidence-rated AdaNormalHedge across awake intraday specialists.

    https://arxiv.org/html/1502.05934v1 Section 4. Per expert,
    r=awake*(g-p*g), R+=r, C+=abs(r); next weight proportional to
    awake*(Phi(R+1,C+1)-Phi(R-1,C+1)),
    Phi(R,C)=exp(max(R,0)^2/(3*C)). Numerical common exponent
    subtraction preserves ratios. Regime masks are our causal price
    heuristics; buy and cash are always awake. This is foundational
    specialist learning, not a claim of a newly published 2026 rule.
    """

    name = "sleeping_council"

    def _reset(self) -> None:
        super()._reset()
        self.regret = np.zeros(6)
        self.absolute = np.zeros(6)

    def _update(self, gain, played, awake) -> None:
        r = awake * (gain - played @ gain)
        self.regret += r
        self.absolute += np.abs(r)
        upper = np.maximum(self.regret + 1, 0) ** 2 / (3 * (self.absolute + 1))
        lower = np.maximum(self.regret - 1, 0) ** 2 / (3 * (self.absolute + 1))
        self.weights = np.exp(upper - upper.max()) * (-np.expm1(lower - upper))

    def _distribution(self, actions, awake):
        w = self.weights * awake
        return w / w.sum() if w.sum() > 1e-14 else awake / awake.sum()


@register
class DefensiveForecast(_IntradayCouncil):
    """K29 defensive probability forecast conditioned on the council's signals.

    https://proceedings.mlr.press/r5/vovk05a/vovk05a.pdf; revisited
    in 2026 https://arxiv.org/abs/2604.19592. Features phi=(1,p,a)
    give S(p)=phi(p,a)*sum_s phi(p_s,a_s)*(y_s-p_s). Choose a root
    on [0,1], or the sign-consistent endpoint, for completed-round
    up/down y. Translate p into long exposure with EW up/down return
    magnitudes and a one-way fee hurdle; this sizing is our adaptation.
    """

    name = "defensive_forecast"

    def _reset(self) -> None:
        super()._reset()
        self.residual = np.zeros(8)
        self.features = np.zeros(8)
        self.probability = 0.5
        self.up = self.down = 0.005

    def _update(self, gain, played, awake) -> None:
        y = float(self._round_return > 0)
        self.residual += self.features * (y - self.probability)
        if y:
            self.up += (self._round_return - self.up) / 42
        else:
            self.down += (-self._round_return - self.down) / 42

    def _distribution(self, actions, awake):
        intercept = self.residual[0] + actions @ self.residual[2:]
        slope = self.residual[1]
        at_one = intercept + slope
        if intercept == 0 and at_one == 0:
            p = 0.5
        elif intercept * at_one <= 0 and slope != 0:
            p = float(np.clip(-intercept / slope, 0, 1))
        else:
            p = float(intercept > 0)
        self.probability = p
        self.features = np.r_[1.0, p, actions]
        edge = p * self.up - (1 - p) * self.down - self.fee_rate
        x = float(np.clip(edge / max(self.up, self.down, 1e-9), 0, 1))
        self.weights = np.array([0, 0, 0, 0, x, 1 - x], dtype=float)
        return self.weights
