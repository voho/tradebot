"""No-regret meta-learning over the experts.

The bot treats trading as a repeated game against the market.  It does not try
to estimate which expert is "true"; it plays a no-regret algorithm whose
guarantee is that, over any horizon, its cumulative payoff is within
``O(sqrt(T log N))`` of the best fixed action in hindsight.  That guarantee holds
*without any distributional assumption on the market* — which is exactly the
property you want from a component that has to survive regime changes.

Two rules are implemented:

``Hedge``
    Multiplicative weights (Freund-Schapire).  Weights are proportional to
    ``exp(eta * cumulative payoff)``.

``RegretMatching``
    Hart & Mas-Colell.  Weights are proportional to positive cumulative regret.
    In self-play the empirical joint distribution converges to the set of
    correlated equilibria; here it is used purely as an alternative no-regret
    rule with no learning-rate to tune.

Two design choices matter for trading:

*Signed action set.*  The action set is ``{+e_1..+e_K, -e_1..-e_K, flat}``.
The learner therefore discovers each expert's *sign* from realised payoffs
rather than inheriting it from the author's priors.  This is what lets the same
roster work on a market where a given microstructure effect runs the other way,
and it keeps the no-regret guarantee over the enlarged action set.

*Bounded, non-stationary tracking.*  Payoffs are normalised by a running scale
and clipped, and weights are mixed toward uniform each step.  Mixing turns the
static-regret bound into a tracking (shifting-expert) bound, which is the right
target when the best expert changes over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LearnerConfig:
    # Hedge's optimal learning rate is ``sqrt(8 ln K / T)``.  With ~15 actions
    # and a horizon of tens of thousands of bars that is ~0.02-0.03.  An order
    # of magnitude larger makes the learner chase noise: with per-observation
    # information coefficients around 0.02, a single bar's payoff is almost
    # entirely noise, and an aggressive rate compounds that noise into
    # confident, wrong weights.
    eta: float = 0.03
    # Uniform mixing buys tracking of a changing best expert, but it also caps
    # the learner's memory at ~1/mix updates.  Resolving an IC of 0.02 needs
    # O(20k) observations, so the memory must be at least that long: 2e-5 is
    # ~50k bars (about six months of 5m data).
    mix: float = 2e-5
    payoff_halflife: float = 500.0  # for the running payoff scale
    include_flat: bool = True
    rule: str = "hedge"  # "hedge" | "regret_matching"
    #: Prior weight on the ``+`` action of the named expert; the rest of the
    #: mass is spread uniformly over the remaining actions.
    #:
    #: Hedge is no-regret under any prior — a non-uniform one only changes the
    #: constant, from ``sqrt(T log K)`` to ``sqrt(T log(1/w_0))``.  Anchoring it
    #: matters here because the payoff differences between experts are small
    #: relative to per-observation noise, so a uniform start spends most of the
    #: sample diluting the one hypothesis that theory actually supports.  The
    #: learner can and does move away from the anchor when the data says so.
    prior_expert: str | None = "transient_dislocation"
    prior_weight: float = 0.70


class _SingleLearner:
    """One no-regret learner over a fixed action set."""

    def __init__(self, n_actions: int, cfg: LearnerConfig, prior: np.ndarray | None = None):
        self.cfg = cfg
        self.n = n_actions
        self.prior = (
            np.full(n_actions, 1.0 / n_actions) if prior is None else np.asarray(prior, float)
        )
        self.w = self.prior.copy()
        self.cum_regret = np.zeros(n_actions)
        self._scale = 0.0
        self._scale_alpha = 1.0 - np.exp(-np.log(2.0) / max(cfg.payoff_halflife, 1.0))
        self.updates = 0

    def weights(self) -> np.ndarray:
        if self.cfg.rule == "regret_matching":
            pos = np.maximum(self.cum_regret, 0.0)
            total = pos.sum()
            if total <= 1e-12:
                return np.full(self.n, 1.0 / self.n)
            return pos / total
        return self.w

    def update(self, payoffs: np.ndarray) -> None:
        """Absorb the realised payoff vector for the action set."""
        # Running scale keeps the learning rate meaningful as volatility drifts.
        # Normalise by the *largest* payoff on offer, not the mean across
        # actions: expert signals are gated and therefore zero on ~95% of bars,
        # so a cross-action mean collapses toward zero and every firing action
        # saturates the clip.  The learner would then see only the sign of each
        # payoff and discard its magnitude — which is most of the information,
        # because the whole edge lives in how extreme the dislocation was.
        mag = float(np.max(np.abs(payoffs)))
        self._scale += self._scale_alpha * (mag - self._scale)
        scale = max(self._scale, 1e-12)
        normed = np.clip(payoffs / scale, -3.0, 3.0)

        played = self.weights()
        realized = float(played @ normed)
        self.cum_regret = (1.0 - self.cfg.mix) * self.cum_regret + (normed - realized)

        w = self.w * np.exp(self.cfg.eta * normed)
        w = np.maximum(w, 1e-300)
        w /= w.sum()
        if self.cfg.mix > 0:
            # Mix back toward the prior rather than toward uniform: absent
            # evidence the right resting point is the theory, not ignorance.
            w = (1.0 - self.cfg.mix) * w + self.cfg.mix * self.prior
        self.w = w
        self.updates += 1


@dataclass
class ContextualNoRegret:
    """A no-regret learner per regime cell, over the signed action set.

    ``n_experts`` experts become ``2 * n_experts (+ 1)`` actions.  Call
    :meth:`position` with the current expert signals to get the blended target,
    then :meth:`update` once the payoff for that bar is realised.
    """

    n_experts: int
    n_contexts: int
    cfg: LearnerConfig = field(default_factory=LearnerConfig)
    #: Expert names, in the same order as the signal vector.  Needed to place
    #: the prior on the right action.
    expert_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.n_actions = 2 * self.n_experts + (1 if self.cfg.include_flat else 0)
        prior = self._build_prior()
        self._learners = [
            _SingleLearner(self.n_actions, self.cfg, prior) for _ in range(self.n_contexts)
        ]
        # Action a -> (expert index, sign); flat has expert index -1.
        idx = np.concatenate([np.arange(self.n_experts), np.arange(self.n_experts)])
        sgn = np.concatenate([np.ones(self.n_experts), -np.ones(self.n_experts)])
        if self.cfg.include_flat:
            idx = np.append(idx, -1)
            sgn = np.append(sgn, 0.0)
        self._action_expert = idx.astype(int)
        self._action_sign = sgn

    def _build_prior(self) -> np.ndarray:
        prior = np.full(self.n_actions, 1.0 / self.n_actions)
        name = self.cfg.prior_expert
        if not name or name not in self.expert_names:
            return prior
        idx = self.expert_names.index(name)  # ``+expert`` action index
        w0 = float(np.clip(self.cfg.prior_weight, 0.0, 0.99))
        prior = np.full(self.n_actions, (1.0 - w0) / (self.n_actions - 1))
        prior[idx] = w0
        return prior

    def action_positions(self, signals: np.ndarray) -> np.ndarray:
        """Position each action would take given the current expert signals."""
        padded = np.append(signals, 0.0)  # index -1 picks the appended zero
        return padded[self._action_expert] * self._action_sign

    def position(self, context: int, signals: np.ndarray) -> float:
        w = self._learners[context].weights()
        return float(w @ self.action_positions(signals))

    def update(
        self,
        context: int,
        signals: np.ndarray,
        realized_return: float,
        *,
        prev_signals: np.ndarray | None = None,
        cost: float = 0.0,
    ) -> None:
        """Update with the *net* payoff each action would have earned.

        This is the full-information (rather than bandit) setting: every action's
        counterfactual payoff is computable because we know what position each
        would have taken and what the market subsequently did.

        Charging each action for its own turnover is what stops the learner
        favouring a jittery expert that looks good gross and bleeds net.
        """
        positions = self.action_positions(signals)
        payoff = positions * realized_return
        if cost > 0.0 and prev_signals is not None:
            payoff = payoff - cost * np.abs(positions - self.action_positions(prev_signals))
        self._learners[context].update(payoff)

    def weights(self, context: int) -> np.ndarray:
        return self._learners[context].weights()

    def expert_weights(self, context: int) -> np.ndarray:
        """Net signed weight per expert, for reporting."""
        w = self.weights(context)
        out = np.zeros(self.n_experts)
        for a in range(self.n_actions):
            e = self._action_expert[a]
            if e >= 0:
                out[e] += w[a] * self._action_sign[a]
        return out

    def flat_weight(self, context: int) -> float:
        if not self.cfg.include_flat:
            return 0.0
        return float(self.weights(context)[-1])
