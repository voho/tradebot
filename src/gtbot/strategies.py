"""Named, frozen strategy presets.

A strategy is not just a class — it is a class *plus* a specific configuration
that some specific evidence was gathered about.  Keeping the shipped
configuration in dataclass defaults loses that link: any later edit to a default
silently redefines what "the strategy" means and quietly invalidates every
published number.

Each preset here pins a complete configuration together with the metadata that
makes its results interpretable: what it assumes, what was measured, on what
data, and what is known to be wrong with it.  ``tests/test_strategies.py``
asserts the pinned values, so changing a default cannot redefine a preset
without a test failing.

Usage::

    from gtbot.strategies import get

    preset = get("dislocation_v2")
    strategy = preset.build()                     # a fresh GameTheoreticStrategy
    strategy = preset.build(direction="long_only")  # with overrides

From the command line::

    gtbot strategies                       # list presets
    gtbot strategies --name dislocation_v2 --json
    gtbot backtest --strategy dislocation_v2 --tier vip9
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, replace

from .engine.broker import CostModel, ExecutionConfig
from .features import FeatureConfig
from .features.regime import RegimeConfig
from .game.equilibrium import AmbiguityConfig
from .game.regret import LearnerConfig
from .risk import RiskConfig
from .strategy import GameTheoreticStrategy, StrategyConfig

__all__ = [
    "StrategyMetadata",
    "StrategyPreset",
    "DISLOCATION_V1",
    "DISLOCATION_V2",
    "PRESETS",
    "get",
    "available",
]


@dataclass(frozen=True)
class StrategyMetadata:
    """Everything needed to judge a preset without re-deriving it.

    The ``validated_*`` fields are measurements, not aspirations: they were
    produced by ``scripts/evaluate.py`` on seeds the configuration had never
    been evaluated against.  ``limitations`` is not boilerplate — every entry is
    a specific thing that is known to be weak.
    """

    name: str
    version: str
    created: str
    summary: str
    #: The market structure the strategy claims to exploit, in one paragraph.
    thesis: str
    #: What kind of instrument and bar this is for.
    instrument: str
    bar_interval: str
    #: Fee tier at or below which the strategy is worth running, and why.
    minimum_fee_tier: str
    fee_tier_note: str
    #: Bars of history required before the online learner is converged enough
    #: for the sizer to allocate.
    min_bars: int
    #: Held-out performance, keyed by fee tier.
    validated_sharpe: dict[str, float] = field(default_factory=dict)
    validated_cagr: dict[str, float] = field(default_factory=dict)
    validated_max_drawdown: dict[str, float] = field(default_factory=dict)
    #: Dollar outcome for a $1,000 deposit at 5x, keyed by "mode/sizing/tier".
    validated_account_pnl_usd: dict[str, float] = field(default_factory=dict)
    #: Statistical evidence for the pooled held-out track record.
    evidence: dict[str, object] = field(default_factory=dict)
    #: How the configuration was arrived at.
    provenance: list[str] = field(default_factory=list)
    #: Known weaknesses.  Read these before deploying anything.
    limitations: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class StrategyPreset:
    """A pinned configuration plus its metadata."""

    metadata: StrategyMetadata
    config: StrategyConfig
    execution: ExecutionConfig
    #: Leverage and deposit the published account figures assume.
    leverage: float = 5.0
    deposit: float = 1_000.0

    @property
    def name(self) -> str:
        return self.metadata.name

    def build(self, **overrides) -> GameTheoreticStrategy:
        """Instantiate the strategy.

        The returned object owns a *copy* of the configuration, so callers can
        mutate it freely without contaminating the preset — which would defeat
        the point of pinning it.
        """
        cfg = copy.deepcopy(self.config)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"StrategyConfig has no field {key!r}")
            setattr(cfg, key, value)
        return GameTheoreticStrategy(cfg)

    def cost_model(self, tier: str | None = None) -> CostModel:
        return CostModel.for_tier(tier or self.metadata.minimum_fee_tier)

    def config_for_tier(self, tier: str) -> StrategyConfig:
        """A copy of the config with the sizer told the truth about costs."""
        cfg = copy.deepcopy(self.config)
        cfg.assumed_cost_bp = self.cost_model(tier).round_trip_bp(self.execution)
        return cfg

    def describe(self) -> str:
        m = self.metadata
        lines = [
            f"{m.name} v{m.version}  ({m.instrument} {m.bar_interval})",
            "",
            m.summary,
            "",
            "THESIS",
            f"  {m.thesis}",
            "",
            f"REQUIRES   fee tier {m.minimum_fee_tier} or better; "
            f"{m.min_bars:,}+ bars of history",
            f"           execution: {self.execution.entry_mode} in / "
            f"{self.execution.exit_mode} out",
            "",
            "HELD-OUT RESULTS (Sharpe by fee tier)",
        ]
        for tier, sr in m.validated_sharpe.items():
            cagr = m.validated_cagr.get(tier)
            dd = m.validated_max_drawdown.get(tier)
            extra = ""
            if cagr is not None and dd is not None:
                extra = f"   CAGR {cagr:+.2%}   maxDD {dd:.2%}"
            lines.append(f"  {tier:>13s}  {sr:+.2f}{extra}")
        lines += ["", f"${self.deposit:,.0f} AT {self.leverage:g}x (held-out mean)"]
        for k, v in m.validated_account_pnl_usd.items():
            lines.append(f"  {k:>28s}  {v:+,.0f}")
        lines += ["", "EVIDENCE"]
        for k, v in m.evidence.items():
            lines.append(f"  {k}: {v}")
        lines += ["", "PROVENANCE"]
        lines += [f"  - {p}" for p in m.provenance]
        lines += ["", "LIMITATIONS"]
        lines += [f"  - {p}" for p in m.limitations]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# v2 — the shipped strategy
# ---------------------------------------------------------------------------

_V2_METADATA = StrategyMetadata(
    name="dislocation_v2",
    version="2.0.0",
    created="2026-08-10",
    summary=(
        "Fades transient price dislocations left behind by order flow the market "
        "maker has not yet worked off, sized by a distributionally-robust "
        "equilibrium sizer and blended by a no-regret meta-learner."
    ),
    thesis=(
        "Most of a 5-minute move is information and is not predictable. A small "
        "part is the temporary displacement created when somebody had to trade in "
        "a hurry: the maker who absorbed that flow carries inventory it wants to "
        "shed, and the displacement unwinds over the next few bars. Two largely "
        "independent estimators read that displacement -- the price-side Kyle "
        "residual (the move order flow cannot explain) and the flow-side "
        "inventory proxy (the decayed negative of cumulative taker flow). Either "
        "alone is worth 2-4bp per trade; averaged they are worth about 11bp."
    ),
    instrument="BTCUSD perpetual (BTCUSDT)",
    bar_interval="5m",
    minimum_fee_tier="vip6",
    fee_tier_note=(
        "Gross edge is 5-9bp per trade. A retail round trip costs 6.65bp, which "
        "leaves nothing; at VIP6 it is 3.85bp and at VIP9 2.05bp. This is a "
        "fee-tier business before it is a signal business, and the sizer will "
        "correctly refuse to trade at retail fees rather than churn."
    ),
    min_bars=130_530,
    validated_sharpe={
        "retail": 0.21, "vip3": 0.59, "vip6": 1.74, "vip9": 2.90, "market_maker": 2.85,
    },
    validated_cagr={
        "retail": 0.0028, "vip3": 0.0385, "vip6": 0.1035, "vip9": 0.2510,
        "market_maker": 0.2421,
    },
    validated_max_drawdown={
        "retail": 0.0033, "vip3": 0.0186, "vip6": 0.0274, "vip9": 0.0449,
        "market_maker": 0.0456,
    },
    validated_account_pnl_usd={
        "long/short robust vip6": 167.0,
        "long/short robust vip9": 419.0,
        "long-only robust vip6": 82.0,
        "long-only robust vip9": 197.0,
        "long/short fixed vip9": 1764.0,
        "long-only fixed vip9": 526.0,
        "worst of 72 runs": -143.0,
    },
    evidence={
        "held_out_seeds": "100-105, never evaluated during development",
        "bars_per_seed": 150_000,
        "pooled_sharpe_vip6": 2.04,
        "bootstrap_ci95": "[+1.42, +2.66]",
        "bootstrap_p_sharpe_le_0": 0.0,
        "newey_west_t": 6.07,
        "anytime_valid_cs95": "[+0.93, +3.16]",
        "conclusive_after_bars": 130_530,
        "probabilistic_sharpe": 1.0,
        "deflated_sharpe_18_trials": 0.2756,
        "random_walk_control_sharpe": 0.0,
        "block_bootstrap_control_sharpe": 0.0,
        "liquidations": "0 of 72 runs; peak margin use 8.9% of the distance",
    },
    provenance=[
        "Hyperparameters (horizon, entry threshold) chosen on training seeds 0-3 "
        "in scripts/search.py; never tuned on the held-out seeds.",
        "Improvements ablated individually in scripts/ablation.py on training "
        "seeds: baseline +2.02 Sharpe, +variance-reduction +2.06 (worst seed "
        "+0.70 -> +1.36), +re-solved-exit +2.34, both +2.65.",
        "Variance reduction is an AIVAT-style control variate: flow arriving "
        "after entry is this game's chance node, so the return it explains has "
        "zero conditional mean and is removable without bias.",
        "The exit is a one-ply depth-limited re-solve against a learned "
        "continuation value, in the spirit of DeepStack's continual re-solving.",
        "Discounted regret matching (CFR+/DCFR) was tried and REJECTED: it beat "
        "Hedge on a planted-signal benchmark but collapsed to near-zero trades "
        "here, because regret matching spreads weight over a sparse 17-action "
        "set where Hedge concentrates.",
    ],
    limitations=[
        "All published results are on a calibrated agent-based simulator, not "
        "real BTCUSD data. The development environment had no exchange access. "
        "Run `gtbot fetch` and re-evaluate before believing any of it.",
        "Deflated Sharpe is 0.2756 -- real headroom over what the search itself "
        "could produce, but not decisive.",
        "Walk-forward with a cold learner per fold is weak (pooled +0.69 / +0.07 "
        "/ +1.32): most folds never trade because the learner needs ~1 year of "
        "5-minute data to converge. Warm-start from all available history.",
        "Passive entries are adversely selected; a maker-in configuration loses "
        "money even though its modelled costs are lower.",
        "Edge is thin in absolute terms (5-9bp per trade). Nothing here survives "
        "a large increase in costs or a materially worse fill model.",
        "Scale-free only up to roughly six figures of deposit; beyond that the "
        "square-root impact term starts to erode the per-trade edge.",
    ],
    references=[
        "Burch, Schmid, Moravcik, Bowling -- AIVAT: A New Variance Reduction "
        "Technique for Agent Evaluation in Imperfect Information Games (AAAI 2018)",
        "Schmid -- Search in Imperfect Information Games (doctoral thesis, 2021)",
        "Moravcik et al. -- DeepStack (Science 2017): continual re-solving",
        "Brown & Sandholm -- Solving Imperfect-Information Games via Discounted "
        "Regret Minimization (AAAI 2019)",
        "Kyle (1985) -- Continuous Auctions and Insider Trading",
        "Avellaneda & Stoikov (2008) -- High-frequency trading in a limit order book",
        "Bailey & Lopez de Prado (2014) -- The Deflated Sharpe Ratio",
    ],
)

DISLOCATION_V2 = StrategyPreset(
    metadata=_V2_METADATA,
    config=StrategyConfig(
        horizon=3,
        max_hold=3,
        entry_signal=0.55,
        exit_signal=0.10,
        edge_halflife=400.0,
        assumed_cost_bp=2.05,  # vip9 taker-in / maker-out; retune per tier
        variance_reduction=True,
        adaptive_exit=True,
        direction="both",
        sizing_mode="robust",
        signal_window=2016,
        min_scale_samples=1000,
        learner=LearnerConfig(
            rule="hedge",
            eta=0.03,
            mix=2e-5,
            prior_expert="transient_dislocation",
            prior_weight=0.70,
        ),
        ambiguity=AmbiguityConfig(k_sigma=0.5, model_haircut=0.90, risk_aversion=0.06),
        risk=RiskConfig(
            target_vol_annual=0.15,
            max_leverage=5.0,
            drawdown_soft=0.06,
            drawdown_hard=0.15,
            max_vol_scalar=12.0,
        ),
        features=FeatureConfig(),
        regime=RegimeConfig(),
    ),
    execution=ExecutionConfig(entry_mode="taker", exit_mode="maker", ttl_bars=1),
)


# ---------------------------------------------------------------------------
# v1 — the pre-improvement baseline, kept so the ablation is reproducible
# ---------------------------------------------------------------------------

_V1_CONFIG = copy.deepcopy(DISLOCATION_V2.config)
_V1_CONFIG.variance_reduction = False
_V1_CONFIG.adaptive_exit = False

DISLOCATION_V1 = StrategyPreset(
    metadata=replace(
        _V2_METADATA,
        name="dislocation_v1",
        version="1.0.0",
        created="2026-08-09",
        summary=(
            "The pre-improvement baseline: same signal and sizer, but a fixed "
            "holding period and no variance reduction. Retained so the ablation "
            "in the README can be reproduced, not because it should be run."
        ),
        validated_sharpe={"retail": 0.13, "vip3": 0.43, "vip6": 1.19, "vip9": 2.32},
        validated_cagr={"retail": 0.0030, "vip3": 0.0204, "vip6": 0.0645, "vip9": 0.2088},
        validated_max_drawdown={
            "retail": 0.0034, "vip3": 0.0204, "vip6": 0.0427, "vip9": 0.0548,
        },
        validated_account_pnl_usd={
            "long/short robust vip6": 107.0,
            "long/short robust vip9": 342.0,
            "long/short fixed vip9": 1114.0,
        },
        evidence={
            "pooled_sharpe_vip6": 1.21,
            "bootstrap_ci95": "[+0.59, +1.85]",
            "newey_west_t": 3.66,
            "deflated_sharpe_18_trials": 0.0011,
        },
        provenance=["Superseded by dislocation_v2; see scripts/ablation.py."],
    ),
    config=_V1_CONFIG,
    execution=DISLOCATION_V2.execution,
)


PRESETS: dict[str, StrategyPreset] = {
    DISLOCATION_V2.name: DISLOCATION_V2,
    DISLOCATION_V1.name: DISLOCATION_V1,
}

#: What ``--strategy`` defaults to.
DEFAULT_PRESET = DISLOCATION_V2.name


def available() -> list[str]:
    return sorted(PRESETS)


def get(name: str = DEFAULT_PRESET) -> StrategyPreset:
    if name not in PRESETS:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    return PRESETS[name]
