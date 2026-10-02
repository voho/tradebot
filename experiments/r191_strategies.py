"""R-191: five frozen, unregistered daily cash/risky allocation mechanisms.

All signal objectives use a fixed 40bp fee plus 1bp slippage (one way),
including when the execution scenario charges different fees. They price
changes in recursive signal targets, not the account's eventual turnover.
Actual broker exposure, fees, slippage, funding and deadbands remain native.

A day's close enters the signal only on its actual 23:55 UTC five-minute
bar after all 288 aligned bars are present. Thus a truncated or incomplete
day cannot supply a premature daily close. Recursive targets and expert
scores start at zero on day253; earlier history supplies features only.
Recursive learning is replayed in prepare; on_bar has no hidden state, so
fresh evaluation accounts can enter an already-established target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy


SIGNAL_COST = .004 + .0001
WARMUP_DAYS = 252
PRIMARY_NAMES = (
    "r191_cost_proximal", "r191_conformal_tail", "r191_cost_olmar",
    "r191_sticky_expert", "r191_robust_growth",
)
FAMILIES = {name: (name, name + "_low", name + "_high") for name in PRIMARY_NAMES}
ALL_NAMES = tuple(name for family in FAMILIES.values() for name in family)
LABELS = dict(zip(PRIMARY_NAMES, (
    "Cost-aware proximal allocation", "Adaptive downside-quantile budget",
    "Cost-gated moving-average reversion", "Sticky net-reward expert selection",
    "Robust block growth allocation",
)))
PARAMS = {}
for _family, _parameter, _values in (
    (PRIMARY_NAMES[0], "penalty", (2., 1., 4.)),
    (PRIMARY_NAMES[1], "budget", (.03, .02, .04)),
    (PRIMARY_NAMES[2], "window", (10, 5, 20)),
    (PRIMARY_NAMES[3], "penalty", (.01, .005, .02)),
    (PRIMARY_NAMES[4], "penalty", (2., 1., 4.)),
):
    for _name, _value in zip(FAMILIES[_family], _values):
        PARAMS[_name] = {"family": _family, _parameter: _value}
        if _name != _family:
            LABELS[_name] = LABELS[_family] + f" ({_parameter}={_value:g})"


def _proximal_weight(mu: float, variance: float, previous: float, penalty: float) -> float:
    """Exact bounded maximum of a concave quadratic with an L1 movement cost."""
    linear, curvature, cost = 7 * mu, 4 * 7 * variance, penalty * SIGNAL_COST
    if curvature <= 0:
        # Linear objective: prefer no turnover when its slope equals the cost.
        return 1. if linear > cost else 0. if linear < -cost else previous
    gradient = linear - curvature * previous
    move = np.sign(gradient) * max(abs(gradient) - cost, 0.) / curvature
    return float(np.clip(previous + move, 0., 1.))


def _adaptive_quantiles(losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score today's loss against yesterday's forecast before refitting today."""
    quantiles = np.full(len(losses), np.nan)
    alphas = np.full(len(losses), .1)
    alpha, previous_q = .1, np.nan
    for i, loss in enumerate(losses):
        if np.isfinite(previous_q) and np.isfinite(loss):
            alpha = float(np.clip(alpha + .01 * (.1 - (loss > previous_q)), .01, .5))
        alphas[i] = alpha
        if i >= WARMUP_DAYS - 1:
            sample = losses[i - WARMUP_DAYS + 1:i + 1]
            if np.isfinite(sample).all():
                quantiles[i] = np.quantile(sample, 1 - alpha)
        previous_q = quantiles[i]
    return quantiles, alphas


def _olmar_weight(forecast: float, previous: float) -> float:
    """Two-asset passive-aggressive update followed by exact simplex projection."""
    if abs(forecast - 1.) <= 2 * SIGNAL_COST:
        return previous
    prediction = np.array([forecast, 1.])
    prior = np.array([previous, 1. - previous])
    centered = prediction - prediction.mean()
    denominator = float(centered @ centered)
    if denominator == 0:
        return previous
    tau = max(0., (1.01 - float(prior @ prediction)) / denominator)
    updated = prior + tau * centered
    return float(np.clip((updated[0] - updated[1] + 1.) / 2., 0., 1.))


def _sticky_targets(close: pd.Series, returns: np.ndarray, penalty: float,
                    start: int = 0) -> np.ndarray:
    """Reward lagged expert weights; choose among today's available decisions."""
    sma20, sma80 = close.rolling(20).mean(), close.rolling(80).mean()
    experts = np.column_stack((
        np.zeros(len(close)), np.ones(len(close)),
        (sma20 > sma80).to_numpy(dtype=float),
        (close < sma20).to_numpy(dtype=float),
    ))
    target = np.zeros(len(close))
    scores = np.zeros(4)
    previous_experts = experts[start - 1] if 0 < start <= len(close) else np.zeros(4)
    older_experts = experts[start - 2] if 1 < start <= len(close) else np.zeros(4)
    weight = 0.
    for i in range(start, len(close)):
        expert = experts[i]
        if np.isfinite(returns[i]):
            scores = (.99 * scores + previous_experts * returns[i]
                      - SIGNAL_COST * np.abs(previous_experts - older_experts))
        objective = scores - penalty * np.abs(expert - weight)
        # Stable exact ties choose cash, long, trend, reversion in that order.
        selected = int(np.argmax(objective))
        weight = float(expert[selected])
        target[i] = weight
        older_experts, previous_experts = previous_experts, expert
    return target


def _block_weight(returns: np.ndarray, previous: float, penalty: float) -> float:
    """Median of six chronological 14-day blocks, with least-turnover ties."""
    grid = np.array([0., .25, .5, .75, 1.])
    weighted = grid[:, None] * returns[None, :]
    growth = weighted - .5 * weighted ** 2
    reward = 14 * np.median(growth.reshape(5, 6, 14).mean(axis=2), axis=1)
    objective = reward - penalty * SIGNAL_COST * np.abs(grid - previous)
    tied = np.flatnonzero(objective == objective.max())
    return float(grid[tied[np.argmin(np.abs(grid[tied] - previous))]])


def _daily_targets(close: pd.Series, config: dict) -> np.ndarray:
    returns = close.pct_change(fill_method=None).to_numpy()
    family = config["family"]
    target = np.zeros(len(close))
    previous = 0.
    if family == "r191_cost_proximal":
        series = pd.Series(returns, index=close.index)
        means, variances = series.rolling(63).mean(), series.rolling(63).var(ddof=0)
        for i in range(WARMUP_DAYS, len(close)):
            mean, variance = means.iloc[i], variances.iloc[i]
            if np.isfinite(mean) and np.isfinite(variance):
                previous = _proximal_weight(mean, max(variance, 0.), previous, config["penalty"])
            target[i] = previous
    elif family == "r191_conformal_tail":
        quantiles, _ = _adaptive_quantiles(np.maximum(0., -returns))
        trend = (close > close.shift(90)).to_numpy()
        valid = np.isfinite(quantiles) & trend
        target[valid] = 1.
        positive_q = valid & (quantiles > 0)
        target[positive_q] = np.minimum(1., config["budget"] / quantiles[positive_q])
    elif family == "r191_cost_olmar":
        forecasts = close.rolling(config["window"]).mean() / close
        for i in range(WARMUP_DAYS, len(close)):
            forecast = forecasts.iloc[i]
            if np.isfinite(forecast):
                previous = _olmar_weight(float(forecast), previous)
            target[i] = previous
    elif family == "r191_sticky_expert":
        target = _sticky_targets(close, returns, config["penalty"], start=WARMUP_DAYS)
    elif family == "r191_robust_growth":
        for i in range(WARMUP_DAYS, len(close)):
            sample = returns[i - 83:i + 1]
            if np.isfinite(sample).all():
                previous = _block_weight(sample, previous, config["penalty"])
            target[i] = previous
    else:
        raise ValueError(f"Unknown R-191 family: {family}")
    # A common observable warmup keeps every family flat for its first
    # 252 complete UTC days, including otherwise shorter-history rules.
    target[:WARMUP_DAYS] = 0.
    return np.clip(target, 0., 1.)


class DailyAllocation(Strategy):
    """Replay a frozen daily allocation and trade actual account exposure drift."""

    warmup = WARMUP_DAYS * 288
    deadband = .05

    def __init__(self, name: str) -> None:
        if name not in PARAMS:
            raise ValueError(f"Unknown R-191 strategy: {name}")
        self.name = name
        self.config = dict(PARAMS[name])

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.DatetimeIndex(df.index)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        aligned = ((ts.minute % 5 == 0) & (ts.second == 0)
                   & (ts.microsecond == 0) & (ts.nanosecond == 0))
        day_counts = pd.Series(aligned, index=ts).groupby(ts.normalize()).transform("sum").to_numpy()
        completed = (ts.hour == 23) & (ts.minute == 55) & aligned & (day_counts == 288)
        close = pd.Series(df["close"].to_numpy()[completed], index=ts[completed])
        daily = pd.Series(_daily_targets(close, self.config), index=close.index)
        df["target"] = daily.reindex(ts, method="ffill").fillna(0.).to_numpy()
        df["r191_decision"] = completed & (np.cumsum(completed) > WARMUP_DAYS)
        return df

    def on_bar(self, ctx: Context) -> None:
        if not bool(ctx.bar["r191_decision"]) or ctx.equity <= 0:
            return
        target = float(ctx.bar["target"])
        held = ctx.position * ctx.close / ctx.equity
        if abs(target - held) > self.deadband or (target == 0 and ctx.position != 0):
            ctx.order_notional(target)


def make_strategy(name: str) -> DailyAllocation:
    return DailyAllocation(name)
