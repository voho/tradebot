"""R-192: five frozen experimental updates to the unchanged Kelly v4 parent.

All recursions consume historical bars during native warmup. All decisions
are prepared causally and callbacks use the parent's target-change rule;
orders retain next-open fills and the broker's native execution controls.
The evaluator may apply an explicitly shared fresh-account initialization.
No experimental configuration is registered as a production strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


PRIMARY_NAMES = (
    "r192_tracking_budget", "r192_factor_clock", "r192_anchor_confirm",
    "r192_robust_state", "r192_confirm_increase",
)
FAMILIES = {name: (name, name + "_low", name + "_high") for name in PRIMARY_NAMES}
ALL_NAMES = tuple(name for family in FAMILIES.values() for name in family)
LABELS = dict(zip(PRIMARY_NAMES, (
    "Accumulated tracking-error budget", "Vote-event and calendar factor clock",
    "Anchor excursion confirmation", "Robust volatility-state classification",
    "Persistent increase confirmation",
)))
PARAMS = {}
for _family, _parameter, _values in (
    (PRIMARY_NAMES[0], "budget", (.01, .005, .02)),
    (PRIMARY_NAMES[1], "period_hours", (24, 12, 48)),
    (PRIMARY_NAMES[2], "confirmation", (.01, .005, .02)),
    (PRIMARY_NAMES[3], "clip_sigma", (6., 4., 8.)),
    (PRIMARY_NAMES[4], "period_bars", (12, 6, 24)),
):
    for _name, _value in zip(FAMILIES[_family], _values):
        PARAMS[_name] = {"family": _family, _parameter: _value}
        if _name != _family:
            LABELS[_name] = LABELS[_family] + f" ({_parameter}={_value:g})"


def _confirmed_anchor(close: np.ndarray, anchor: np.ndarray,
                      confirmation: float, band: float = .01) -> np.ndarray:
    """Require consecutive accumulated excess beyond the opposite-state band."""
    result = np.zeros(len(close))
    state, accumulated = 0., 0.
    for i, (price, reference) in enumerate(zip(close, anchor)):
        if np.isfinite(reference) and reference > 0:
            # Compare prices exactly as the native parent does; computing
            # only ratio > band can disagree at a floating-point boundary.
            crossed = (price > reference * (1 + band) if state == 0
                       else price < reference * (1 - band))
            if crossed:
                excess = (price / reference - (1 + band) if state == 0
                          else (1 - band) - price / reference)
                accumulated += max(excess, 0.) / BARS_PER_DAY
                if accumulated >= confirmation:
                    state = 1. - state
                    accumulated = 0.
            else:
                accumulated = 0.
        else:
            accumulated = 0.
        result[i] = state
    return result


def _volatility(returns: pd.Series, parent: KellyRegimeV4) -> tuple[np.ndarray, np.ndarray]:
    """Native unbiased pandas EWM standard deviation, including its one-bar lag."""
    vol = (returns.ewm(span=parent.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = pd.Series(vol).ewm(span=parent.anchor_span_days * BARS_PER_DAY,
                              min_periods=BARS_PER_DAY).mean().to_numpy()
    return vol, slow


def _vol_states(ratio: np.ndarray, parent: KellyRegimeV4) -> np.ndarray:
    states = np.zeros(len(ratio), dtype=np.int8)
    state = 0
    for i, value in enumerate(ratio):
        if np.isfinite(value):
            if state == 0:
                state = 1 if value > parent.high_in else (-1 if value < parent.low_in else 0)
            elif state == 1 and value < parent.high_out:
                state = 0
            elif state == -1 and value > parent.low_out:
                state = 0
        states[i] = state
    return states


def _parent_features(df: pd.DataFrame, *, confirmation: float | None = None,
                     clip_sigma: float = np.inf) -> dict[str, np.ndarray]:
    """Reconstruct native components; optional changes touch one factor only."""
    parent = KellyRegimeV4()
    close = df["close"]
    returns = np.log(close).diff()
    votes = []
    for days in parent.horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        if confirmation is None:
            vote = pd.Series(
                np.where(close > anchor * (1 + parent.band), 1.,
                         np.where(close < anchor * (1 - parent.band), 0., np.nan)),
                index=df.index,
            ).ffill().fillna(0.)
        else:
            vote = pd.Series(_confirmed_anchor(close.to_numpy(), anchor.to_numpy(),
                                               confirmation, parent.band), index=df.index)
        votes.append(vote)
    frac = (sum(votes) / len(votes)).to_numpy()
    if parent.vote_gamma != 1.:
        frac = frac ** parent.vote_gamma

    vol, slow = _volatility(returns, parent)
    state_vol, state_slow = vol, slow
    if np.isfinite(clip_sigma):
        reference = returns.ewm(span=parent.vol_span, min_periods=BARS_PER_DAY).std().shift(1)
        bound = clip_sigma * reference
        clipped = returns.clip(lower=-bound, upper=bound)
        # No usable past scale means no clipping, including an initially
        # flat market; never suppress its first genuine nonzero return.
        clipped = clipped.where(np.isfinite(reference) & (reference > 0), returns)
        state_vol, state_slow = _volatility(clipped, parent)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(state_slow > 0, state_vol / state_slow, np.nan)
        full = np.minimum(parent.target_vol / vol, parent.max_leverage)
        steady = np.minimum(parent.target_vol / slow, parent.max_leverage)
    full = np.where(np.isfinite(full), full, 0.)
    steady = np.where(np.isfinite(steady), steady, 0.)
    states = _vol_states(ratio, parent)
    desired = frac * np.where(states != 0, full, steady)
    return {"frac": frac, "vol": vol, "slow": slow, "full": full,
            "steady": steady, "state": states, "desired": desired}


def _latch(desired: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Native strict ten-percent target latch, optionally limited to events."""
    target = np.zeros(len(desired))
    accepted = 0.
    for i, value in enumerate(desired):
        if (eligible is None or eligible[i]) and abs(value - accepted) > .10:
            accepted = value
        target[i] = accepted
    return target


def _tracking_budget(desired: np.ndarray, budget: float) -> np.ndarray:
    if budget == 0:
        return _latch(desired)
    target = np.zeros(len(desired))
    accepted, accumulated, direction = 0., 0., 0.
    for i, value in enumerate(desired):
        gap = value - accepted
        sign = np.sign(gap)
        if sign == 0 or sign != direction:
            accumulated = 0.
        direction = sign
        accumulated += gap * gap / BARS_PER_DAY
        if sign != 0 and accumulated >= budget:
            accepted = value
            accumulated, direction = 0., 0.
        target[i] = accepted
    return target


def _factor_events(index: pd.DatetimeIndex, frac: np.ndarray, period_hours: int) -> np.ndarray:
    if period_hours == 0:
        return np.ones(len(index), dtype=bool)
    ts = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    # Timedelta division works with any DatetimeIndex storage resolution.
    elapsed = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(nanoseconds=1)
    calendar = np.asarray(elapsed % pd.Timedelta(hours=period_hours).value == 0)
    previous = np.r_[0., frac[:-1]] if len(frac) else np.array([])
    return calendar | (frac != previous)


def _confirm_increases(parent_target: np.ndarray, period_bars: int) -> np.ndarray:
    target = np.zeros(len(parent_target))
    accepted, consecutive = 0., 0
    for i, value in enumerate(parent_target):
        if value <= accepted:
            accepted, consecutive = value, 0
        else:
            consecutive += 1
            if consecutive >= period_bars:
                accepted, consecutive = value, 0
        target[i] = accepted
    return target


class KellyUpdate(KellyRegimeV4):
    """Frozen causal component update retaining Kelly v4's native order callback."""

    def __init__(self, name: str, **overrides) -> None:
        super().__init__()
        if name not in PARAMS:
            raise ValueError(f"Unknown R-192 strategy: {name}")
        self.name = name
        self.config = dict(PARAMS[name])
        if set(overrides) - (set(self.config) - {"family"}):
            raise ValueError("Only the family's declared parameter can be overridden")
        self.config.update(overrides)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        family = self.config["family"]
        features = _parent_features(
            df,
            confirmation=self.config.get("confirmation"),
            clip_sigma=self.config.get("clip_sigma", np.inf),
        )
        desired = features["desired"]
        if family == "r192_tracking_budget":
            target = _tracking_budget(desired, self.config["budget"])
        elif family == "r192_factor_clock":
            events = _factor_events(df.index, features["frac"], self.config["period_hours"])
            target = _latch(desired, events)
        else:
            target = _latch(desired)
            if family == "r192_confirm_increase":
                target = _confirm_increases(target, self.config["period_bars"])
        df["target"] = target
        return df


def make_strategy(name: str, **overrides) -> KellyUpdate:
    """Construct a named frozen configuration; overrides are for identity tests."""
    return KellyUpdate(name, **overrides)
