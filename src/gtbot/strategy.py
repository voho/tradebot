"""The assembled game-theoretic strategy.

Decision pipeline, once per bar:

1. **Players.**  Each expert in :mod:`gtbot.game.experts` states a strategic
   hypothesis about who is on the other side and emits a signal in ``[-1, 1]``.
2. **Repeated game.**  A contextual no-regret learner
   (:mod:`gtbot.game.regret`) blends the signed action set
   ``{+e_i, -e_i, flat}`` per regime cell.  It learns each expert's sign and
   weight from realised payoffs, with a shifting-expert tracking bound rather
   than a static one.
3. **Edge estimate.**  An online estimator tracks the mean payoff of the
   *triggered* trade population over the target horizon, together with a
   t-statistic that becomes the sizer's confidence.
4. **Robust sizing.**  A zero-sum game against an adversarial nature
   (:mod:`gtbot.game.equilibrium`) turns edge, cost and confidence into a
   position size that collapses toward zero when the edge is not well supported.
5. **Risk.**  Volatility targeting, a drawdown governor and exposure caps
   (:mod:`gtbot.risk`) decide how much capital that conviction may command.

Every step reads only bars ``0..t`` and every position is executed at ``t+1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features as F
from .data.schema import BTCUSD_5M, BarSpec
from .features.regime import RegimeConfig, n_cells
from .game.equilibrium import AmbiguityConfig, fast_robust_size
from .game.experts import Expert, default_experts, signal_matrix
from .game.regret import ContextualNoRegret, LearnerConfig
from .risk import RiskConfig, RiskManager


@dataclass
class StrategyConfig:
    #: Horizon, in bars, over which the edge is estimated and trades are sized.
    horizon: int = 3
    #: Half-life of the online edge estimator, in *triggered* observations.
    edge_halflife: float = 400.0
    #: Minimum |t-statistic| on the edge before full confidence is granted.
    t_full_confidence: float = 3.0
    #: Round-trip cost assumption used by the sizer, in basis points.
    assumed_cost_bp: float = 6.65  # taker in (4.85) + maker out (1.8)
    #: Blended-signal magnitude required to open a trade.  Expert signals are
    #: already gated at :data:`~gtbot.game.experts.Z_GATE` sigma and rise
    #: linearly to 1 at ``Z_FULL_SCALE``, so 0.45 corresponds to roughly a
    #: 3-sigma dislocation.  Thresholding the blend directly avoids
    #: standardising an already-standardised quantity, which on a signal that
    #: is zero 95% of the time produces wild z-scores.
    entry_signal: float = 0.55
    #: Magnitude at which an open trade is closed (hysteresis band).
    exit_signal: float = 0.10
    #: Hard cap on how long a trade may stay open, in bars.  Defaults to the
    #: horizon so the object traded is exactly the object the edge estimator
    #: measures.
    max_hold: int = 3
    #: Window for z-scoring the blended signal.
    signal_window: int = 2016
    #: Minimum samples in that window before any trade is allowed.
    min_scale_samples: int = 1000
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    ambiguity: AmbiguityConfig = field(default_factory=AmbiguityConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    features: F.FeatureConfig = field(default_factory=F.FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)


class GameTheoreticStrategy:
    """The strategy object driven by both the backtester and the paper trader."""

    def __init__(
        self,
        cfg: StrategyConfig | None = None,
        experts: list[Expert] | None = None,
        spec: BarSpec = BTCUSD_5M,
    ):
        self.cfg = cfg or StrategyConfig()
        self.experts = experts or default_experts()
        self.spec = spec
        self.learner = ContextualNoRegret(
            n_experts=len(self.experts),
            n_contexts=n_cells(self.cfg.regime),
            cfg=self.cfg.learner,
            expert_names=[e.name for e in self.experts],
        )
        self.risk = RiskManager(self.cfg.risk, spec.bars_per_year)

        # Online estimator of the mean payoff of triggered trades.
        self._edge_mean = 0.0
        self._edge_var = 0.0
        self._edge_n = 0.0
        self._last_edge_u = -10**9
        self._alpha = 1.0 - math.exp(-math.log(2.0) / max(self.cfg.edge_halflife, 1.0))

        self._blend = np.zeros(0)
        self._blend_m2 = 0.0  # running variance of the blended signal
        self._blend_mean = 0.0
        self._last_equity = 0.0
        self._pos_target = 0.0
        self._actual_pos = 0.0
        self._entry_bar = -10**9
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def prepare(self, bars: pd.DataFrame) -> None:
        """Vectorised precomputation of features and expert signals."""
        self.fs = F.build(bars, self.cfg.features)
        self.signals = signal_matrix(self.experts, self.fs)
        self.context = self.fs["regime_cell"].astype(int)
        self._log_close = np.log(np.maximum(self.fs.close, 1e-12))
        self._atr_frac = self.fs["atr"]
        self._rv = self.fs["realized_vol"]
        n = len(self.fs)
        self._blend = np.zeros(n)
        self._target = np.zeros(n)
        self._edge = np.zeros(n)
        self._conf = np.zeros(n)
        self._size = np.zeros(n)
        self._z = np.zeros(n)
        self._pos_target = 0.0
        self._actual_pos = 0.0
        self._entry_bar = -10**9
        self._last_edge_u = -10**9
        self.warmup = max(self.cfg.features.warmup, self.cfg.horizon + 2)
        self._worst_mult_cache: dict[int, float] = {}

    def atr_fraction(self) -> np.ndarray:
        return self._atr_frac

    # ------------------------------------------------------------------
    # Online updates
    # ------------------------------------------------------------------
    def observe(self, t: int) -> None:
        """Absorb everything that became known when bar ``t`` closed."""
        # (a) No-regret update.  The payoff is measured over the *same horizon
        #     the strategy actually trades* and is charged the turnover each
        #     action would have incurred.  Scoring actions on a one-bar horizon
        #     while trading a twelve-bar one makes the learner optimise the
        #     wrong game and it will happily learn inverted signs.
        h = self.cfg.horizon
        if t >= h + 1 and t % h == 0:
            # Non-overlapping payoff windows.  Updating every bar feeds the
            # learner h copies of essentially the same observation, which
            # inflates its evidence by ~h without adding information.
            #
            # The payoff is gross.  Charging each action its own per-bar
            # turnover sounds prudent but is badly mis-scaled: a signal that
            # moves every bar would be charged a full round trip every bar
            # (~1.8bp) against mean gross payoffs of ~0.2bp, so the learner
            # ranks experts by smoothness rather than by edge and reliably
            # discards the best one.  Costs belong to the entry threshold, the
            # trade state machine and the sizer, all of which see the actual
            # trading schedule.
            u = t - h
            r = float(self._log_close[t] - self._log_close[u])
            self.learner.update(int(self.context[u]), self.signals[u], r)

        # (b) Edge estimate, conditional on the *trigger* rather than fitted
        #     across all bars.  The payoff of an extreme dislocation is strongly
        #     convex in the signal: a global through-the-origin fit predicts
        #     roughly a third of the return actually realised in the tail the
        #     strategy trades, and sizing off that number vetoes every trade.
        #     Estimating the mean payoff of the triggered population directly
        #     measures the quantity the sizer needs.
        if t >= h + 1:
            u = t - h
            zu = float(self._z[u])
            if abs(zu) >= self.cfg.entry_signal and u >= self._last_edge_u + h:
                # Non-overlapping samples only: overlapping h-bar windows are
                # serially correlated and would inflate the t-statistic that
                # becomes the sizer's confidence.
                self._last_edge_u = u
                pay = math.copysign(1.0, zu) * float(self._log_close[t] - self._log_close[u])
                a = self._alpha
                self._edge_mean += a * (pay - self._edge_mean)
                self._edge_var += a * ((pay - self._edge_mean) ** 2 - self._edge_var)
                self._edge_n = min(self._edge_n + 1.0, 1.0 / a)

    def record(self, t: int, *, target: float, realized_position: float, equity: float) -> None:
        # The trade state machine must key off the position the broker actually
        # holds, not the one the strategy asked for.  A resting limit order that
        # never filled would otherwise leave the strategy believing it is in a
        # trade, ageing out a position it does not own and then "exiting" it.
        if realized_position != 0.0 and self._actual_pos == 0.0:
            self._entry_bar = t
        self._actual_pos = realized_position

        if self._last_equity:
            self.risk.observe(equity / self._last_equity - 1.0, equity)
        else:
            self.risk.observe(0.0, equity)
        self._last_equity = equity
        self._target[t] = target

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def _edge_estimate(self) -> float:
        """Expected signed return over the horizon for a triggered trade."""
        return max(self._edge_mean, 0.0)

    def _confidence(self) -> float:
        """Map the edge estimate's t-statistic onto ``[0, 1]``."""
        if self._edge_n < 30 or self._edge_var <= 1e-24:
            return 0.0
        se = math.sqrt(self._edge_var / max(self._edge_n, 1.0))
        if se <= 1e-18:
            return 0.0
        return float(np.clip((self._edge_mean / se) / self.cfg.t_full_confidence, 0.0, 1.0))

    def _edge_se(self) -> float:
        """Standard error of the edge estimate."""
        if self._edge_n < 5:
            return float("inf")
        return math.sqrt(max(self._edge_var, 0.0) / max(self._edge_n, 1.0))

    def decide(self, t: int) -> float:
        """Target position for bar ``t+1``, as a signed fraction of equity.

        A trade is a discrete object with an entry, a hold and an exit rather
        than a position that is re-derived from scratch every bar.  Re-deriving
        every bar is what turns a real edge into turnover: the signal wobbles
        around the threshold and the strategy pays the spread on every wobble.
        """
        ctx = int(self.context[t])
        blended = self.learner.position(ctx, self.signals[t])
        self._blend[t] = blended

        # Scale is estimated from the trailing window, so the threshold adapts
        # to how active the learner currently is.
        z = blended
        self._z[t] = z

        in_trade = self._actual_pos != 0.0
        held = t - self._entry_bar if in_trade else 0

        if in_trade:
            flipped = np.sign(z) == -np.sign(self._actual_pos) and abs(z) >= self.cfg.entry_signal
            expired = held >= self.cfg.max_hold
            # The fade test may only fire once the trade has had the horizon it
            # was sized for.  Gated signals drop to zero the bar after they
            # fire, so an unconditional fade exit closes every trade after one
            # bar — paying a full round trip to capture a fraction of a
            # reversion that was measured over ``horizon`` bars.
            faded = held >= self.cfg.horizon and abs(z) <= self.cfg.exit_signal
            if not (flipped or expired or faded):
                # Hold the existing position untouched: no churn, no cost.
                return self._pos_target
            self._pos_target = 0.0
            if not flipped:
                return 0.0

        if abs(z) < self.cfg.entry_signal:
            # Also cancels a working order whose signal has since decayed.
            self._pos_target = 0.0
            return 0.0

        # Expected return over the horizon for a trade like this one.
        edge = self._edge_estimate()
        se = self._edge_se()
        confidence = self._confidence()
        vol_h = float(self._rv[t]) * math.sqrt(self.cfg.horizon)
        cost = self.cfg.assumed_cost_bp * 1e-4

        size = fast_robust_size(edge, se, vol_h, cost, self.cfg.ambiguity)
        self._edge[t] = edge
        self._conf[t] = confidence
        self._size[t] = size

        raw = math.copysign(size, blended)
        target = self.risk.apply(raw, self._last_equity or 1.0)
        self._pos_target = target
        return target

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        cells = n_cells(self.cfg.regime)
        weights = np.array([self.learner.expert_weights(c) for c in range(cells)])
        return {
            "expert_names": [e.name for e in self.experts],
            "expert_weights_by_cell": weights,
            "expert_weights_mean": weights.mean(axis=0),
            "flat_weight_mean": float(
                np.mean([self.learner.flat_weight(c) for c in range(cells)])
            ),
            "edge_mean_bp": 1e4 * self._edge_mean,
            "edge_n": self._edge_n,
            "confidence": self._confidence(),
            "mean_size": float(np.mean(self._size[self._size > 0])) if np.any(self._size > 0) else 0.0,
            "n_signals": int(np.sum(np.abs(self._target) > 0)),
            "blend": self._blend,
            "z": self._z,
            "edge": self._edge,
            "size": self._size,
        }
