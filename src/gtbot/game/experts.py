"""The players.

Each expert is one *strategic hypothesis* about who is on the other side of the
trade and what they are trying to do.  An expert produces a raw score; the base
class turns that into the signal the meta-learner consumes.

Two invariants make the meta-learner's comparison meaningful, and both were
learned the hard way:

**Common scale.**  Every raw score is standardised over a trailing window
before it reaches the action set.  The learner ranks actions by realised
payoff, and payoff scales with signal amplitude, so an expert that is merely
louder beats an equally skilful quieter one unless amplitudes are equalised.

**Gating at the traded threshold.**  Signals are zero inside ``Z_GATE`` sigma
and rise linearly to one at ``Z_FULL_SCALE``.  The strategy only ever trades
the tail, but the learner scores *average* payoff; without gating those are
different objectives.  On this data the two leading experts differ by 3% on
average payoff and by a factor of 2.5 on tail payoff — a learner optimising the
average reliably picks the wrong one, and no learning-rate tuning fixes it.
Gating makes an expert's average payoff *be* its tail payoff.

The linear-then-clip map (rather than a ``tanh``) matters for the same reason:
a ``tanh`` squashes the 2-4 sigma band into a narrow region near +/-1, which is
exactly the band a dislocation-fading strategy lives in.

All signals are computed vectorised over the whole history, but every value is a
causal function of bars ``0..t`` — see ``tests/test_causality.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import FeatureSet
from ..features.rolling import zscore

#: Trailing window used to standardise raw scores into signals.
Z_WINDOW = 2016

#: Standard deviations below which an expert stays silent.
Z_GATE = 2.0

#: Standard deviations mapped onto a full-strength signal of 1.0.
#:
#: Set well above the entry threshold on purpose.  Per-trade edge keeps rising
#: with the extremity of the dislocation, so if the map saturates near the
#: threshold every event past that point looks identical and the strategy
#: cannot select the deepest — and most profitable — tail.
Z_FULL_SCALE = 6.0


def gate(z: np.ndarray) -> np.ndarray:
    """Map a z-score to a gated, bounded signal in ``[-1, 1]``."""
    span = max(Z_FULL_SCALE - Z_GATE, 1e-9)
    return np.sign(z) * np.clip((np.abs(z) - Z_GATE) / span, 0.0, 1.0)


def peak_hold(x: np.ndarray, decay: float) -> np.ndarray:
    """Hold the most recent signed impulse, decaying geometrically.

    Event experts fire on a single bar but imply a trade lasting several.  This
    turns an impulse into a decaying score without any forward-looking smoothing.
    """
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i in range(x.size):
        carry *= decay
        if abs(x[i]) > abs(carry):
            carry = float(x[i])
        out[i] = carry
    return out


@dataclass(frozen=True)
class Expert:
    """A named strategic hypothesis."""

    name: str
    rationale: str

    def raw(self, fs: FeatureSet) -> np.ndarray:  # pragma: no cover - interface
        """Raw, unstandardised score.  Sign is direction."""
        raise NotImplementedError

    def signal(self, fs: FeatureSet) -> np.ndarray:
        """Standardised, gated signal in ``[-1, 1]``."""
        return gate(zscore(self.raw(fs), Z_WINDOW))


@dataclass(frozen=True)
class ImpactOvershoot(Expert):
    """Kyle-model residual: price moved further than the flow can justify.

    Fit ``dp = lambda * signed volume`` over a trailing window; the residual is
    the part of the move with no order-flow explanation.  Under Kyle's model the
    permanent component is exactly the flow-explained part, so a large residual
    is transient by construction and should revert.
    """

    name: str = "impact_overshoot"
    rationale: str = "fade price moves unexplained by contemporaneous order flow"

    def raw(self, fs: FeatureSet) -> np.ndarray:
        return -fs["impact_residual"]


@dataclass(frozen=True)
class InventorySkew(Expert):
    """Stackelberg game against the market maker.

    The maker moves first (it posts quotes), we move second.  An Avellaneda-
    Stoikov maker prices around a reservation price displaced from the mid in
    proportion to its inventory, and that displacement *unwinds* as the maker
    hedges the position off.  Inventory is not observable, but it is the decayed
    negative of cumulative taker flow, which is: maker long (takers have been
    selling) means price is currently displaced downward and drifts back up as
    the inventory clears.
    """

    name: str = "inventory_skew"
    rationale: str = "anticipate the market maker's inventory-driven quote skew"
    feature: str = "mm_inventory_z"

    def raw(self, fs: FeatureSet) -> np.ndarray:
        return fs[self.feature]


@dataclass(frozen=True)
class SlowInventorySkew(InventorySkew):
    """The same mechanism read over a slower inventory half-life.

    The unwind rate is not observable and differs by venue and regime, so rather
    than fit it we field a fast and a slow reader and let the learner decide.
    """

    name: str = "inventory_skew_slow"
    rationale: str = "maker inventory unwind read over a slower half-life"
    feature: str = "mm_inventory_slow_z"


@dataclass(frozen=True)
class TransientDislocation(Expert):
    """The two readings of the same latent quantity, combined.

    :class:`ImpactOvershoot` and :class:`InventorySkew` are not competing
    hypotheses — they are two largely independent *measurements of one latent
    variable*: how far price sits from its efficient level because of flow the
    maker has yet to unwind.  One reads it from the price side (the move order
    flow cannot explain), the other from the flow side (the inventory itself).

    Averaging two unbiased readings of comparable precision is estimation
    theory, not curve fitting, and the measured payoff is insensitive to the
    exact split — 50/50 and 60/40 differ by about 10%, while either component
    alone is worth a fraction of the pair.
    """

    name: str = "transient_dislocation"
    rationale: str = "combine price-side and flow-side readings of unwound maker inventory"

    def raw(self, fs: FeatureSet) -> np.ndarray:
        price_side = zscore(-fs["impact_residual"], Z_WINDOW)
        flow_side = fs["mm_inventory_z"]
        return 0.5 * (price_side + flow_side)


@dataclass(frozen=True)
class LiquiditySweepFader(Expert):
    """Signalling game: was that breakout information, or a liquidity grab?

    A break of a swing extreme is a *signal* sent to the market.  In a
    separating equilibrium an informed trader breaks the level and keeps going;
    a stop cascade breaks it and closes back inside.  Geometry alone is not
    enough — most bars that poke a level and close back are ordinary noise, and
    fading those loses money — so we require corroborating evidence that the
    excursion was unexplained by order flow.
    """

    name: str = "sweep_fader"
    rationale: str = "fade uninformed stop cascades that close back inside the swept level"
    hold_decay: float = 0.72
    toxicity_cut: float = 0.75

    def raw(self, fs: FeatureSet) -> np.ndarray:
        direction = fs["sweep_dir"]
        unexplained = -np.sign(fs["impact_residual"])
        confirmed = (direction != 0) & (np.sign(direction) == unexplained)
        conviction = fs["cascade_rank"] * np.tanh(2.0 * fs["sweep_penetration"])
        toxicity = np.clip((self.toxicity_cut - fs["vpin_rank"]) / self.toxicity_cut, 0.0, 1.0)
        wick = np.where(direction > 0, fs["lower_wick"], fs["upper_wick"])
        impulse = np.where(confirmed, direction, 0.0) * conviction * toxicity
        impulse = impulse * np.clip(0.4 + wick, 0.0, 1.4)
        return peak_hold(impulse, self.hold_decay)


@dataclass(frozen=True)
class InformedContinuation(Expert):
    """Adverse-selection side of the game: follow flow that looks informed.

    When VPIN is high *and* order-flow imbalance is serially persistent, the
    likeliest explanation is a large informed order being worked over many bars.
    The maker is losing on it and will keep repricing, so the play is to trade
    with the flow rather than against it.
    """

    name: str = "informed_continuation"
    rationale: str = "trade with persistent, toxic order flow (a large order being worked)"
    hold_decay: float = 0.85

    def raw(self, fs: FeatureSet) -> np.ndarray:
        toxicity = np.clip((fs["vpin_rank"] - 0.5) * 2.0, 0.0, 1.0)
        persistence = np.clip(fs["ofi_persistence"] * 4.0, 0.0, 1.0)
        direction = np.tanh(3.0 * fs["ofi_ewma"])
        return peak_hold(direction * toxicity * persistence, self.hold_decay)


@dataclass(frozen=True)
class TrendRider(Expert):
    """Herding equilibrium: when the crowd is coordinated, ride it.

    Gated on the Lo-MacKinlay variance ratio so it only engages when multi-bar
    moves genuinely compound; without that gate a trend follower bleeds in the
    mean-reverting regime that dominates 5-minute crypto data.
    """

    name: str = "trend_rider"
    rationale: str = "follow the trend only while the variance ratio says moves compound"

    def raw(self, fs: FeatureSet) -> np.ndarray:
        engage = np.clip((fs["variance_ratio"] - 1.0) * 2.5, 0.0, 1.0)
        return np.tanh(0.6 * fs["trend_strength"]) * engage


@dataclass(frozen=True)
class MeanReverter(Expert):
    """The mirror of :class:`TrendRider`, active when the variance ratio is low."""

    name: str = "mean_reverter"
    rationale: str = "fade stretched moves while the variance ratio says moves cancel"

    def raw(self, fs: FeatureSet) -> np.ndarray:
        engage = np.clip((1.0 - fs["variance_ratio"]) * 2.5, 0.0, 1.0)
        stretch = np.tanh(0.7 * fs["trend_strength"]) * np.clip(
            np.abs(fs["trend_strength"]) / 3.0, 0.0, 1.0
        )
        return -stretch * engage


def default_experts() -> list[Expert]:
    """The standard roster used by :class:`gtbot.strategy.GameTheoreticStrategy`."""
    return [
        LiquiditySweepFader(),
        InformedContinuation(),
        InventorySkew(),
        SlowInventorySkew(),
        ImpactOvershoot(),
        TransientDislocation(),
        TrendRider(),
        MeanReverter(),
    ]


def signal_matrix(experts: list[Expert], fs: FeatureSet) -> np.ndarray:
    """Stack expert signals into an ``(n_bars, n_experts)`` matrix."""
    return np.column_stack([e.signal(fs) for e in experts])
