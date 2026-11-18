"""R-193: three frozen, unregistered attention allocators ("Attention Is All You Need").

Each family takes one more step into Vaswani et al. (2017):

1. ``r193_attention``   single-head scaled dot-product attention, no training:
   today's market state queries every past state whose 20-day outcome has
   matured, and the attention-weighted outcome is the drift forecast.
2. ``r193_multihead``   three heads over disjoint feature subspaces (short
   reversal, trend, volatility), concatenated through a ridge-fitted output
   projection W_O refitted once per calendar year on matured outcomes only.
3. ``r193_transformer`` a one-layer transformer encoder (sinusoidal positional
   encoding, 2-head self-attention, residual + LayerNorm, feed-forward) over the
   last L daily tokens, trained end to end by antithetic evolution strategies
   to maximise next-day log growth net of the 41bp signal cost.

All three trade one risky asset against cash in [0, 1] on R-191's complete-UTC-
day clock, and all learning is replayed causally in ``prepare``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.r191_strategies import SIGNAL_COST, WARMUP_DAYS, _proximal_weight
from tradebot.strategy import Context, Strategy

HORIZON = 20          # outcome horizon (days) for the attention value bank
MIN_KEYS = 100        # matured memories required before attention may act
MIN_FIT = 252         # training rows required before a learned layer is fitted
PROXIMAL_PENALTY = 2.  # R-191's frozen proximal cost multiplier
HEADS = {"short": (0, 1), "trend": (2, 3, 4), "vol": (5, 6)}

PRIMARY_NAMES = ("r193_attention", "r193_multihead", "r193_transformer")
FAMILIES = {name: (name, name + "_low", name + "_high") for name in PRIMARY_NAMES}
ALL_NAMES = tuple(name for family in FAMILIES.values() for name in family)
LABELS = dict(zip(PRIMARY_NAMES, (
    "Single-head attention analog allocation",
    "Multi-head attention with ridge output projection",
    "ES-trained one-layer transformer encoder",
)))
PARAMS = {}
for _family, _parameter, _values in (
    (PRIMARY_NAMES[0], "temperature", (1., .5, 2.)),
    (PRIMARY_NAMES[1], "ridge", (1., .25, 4.)),
    (PRIMARY_NAMES[2], "context", (30, 15, 60)),
):
    for _name, _value in zip(FAMILIES[_family], _values):
        PARAMS[_name] = {"family": _family, _parameter: _value}
        if _name != _family:
            LABELS[_name] = LABELS[_family] + f" ({_parameter}={_value:g})"


# ---------------------------------------------------------------- features

def _features(close: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-free daily state, volatility, and matured-outcome values.

    Returns (x, sigma, outcome): x[t] is the 7-feature token, sigma[t] the
    trailing 60-day log-return volatility, and outcome[j] the vol-normalised
    20-day forward log return. outcome[j] is only usable once j + HORIZON <= t.
    """
    log_close = np.log(close)
    lr = log_close.diff()
    sig60 = lr.rolling(60).std()
    columns = [(log_close - log_close.shift(k)) / (sig60 * np.sqrt(k)) for k in (1, 5, 20, 60, 120)]
    columns.append(np.log(lr.rolling(20).std() / sig60))
    columns.append(np.log(sig60 / lr.rolling(120).std()))
    x = np.column_stack([c.to_numpy() for c in columns])
    x[~np.isfinite(x).all(axis=1)] = np.nan
    outcome = ((log_close.shift(-HORIZON) - log_close) / (sig60 * np.sqrt(HORIZON))).to_numpy()
    return x, sig60.to_numpy(), outcome


def _attend(query: np.ndarray, keys: np.ndarray, temperature: float) -> np.ndarray:
    """Scaled dot-product attention weights for one query.

    The -|k|^2/2 key bias turns q.k into -|q-k|^2/2 up to a constant that the
    softmax cancels; without it, unprojected dot products reward extreme keys
    rather than similar ones.
    """
    scores = (keys @ query - .5 * np.einsum("ij,ij->i", keys, keys))
    scores /= temperature * np.sqrt(keys.shape[1])
    weights = np.exp(scores - scores.max())
    return weights / weights.sum()


def _drift(u: float, sigma: float) -> float:
    """Arithmetic daily drift implied by a vol-normalised 20-day log outcome."""
    return u * sigma / np.sqrt(HORIZON) + .5 * sigma ** 2


def _head_outputs(x: np.ndarray, outcome: np.ndarray, temperature: float,
                  heads: dict) -> np.ndarray:
    """Per-day attention readouts; NaN until MIN_KEYS memories have matured.

    Keys and query are standardised with feature moments through day t, so
    query and memories share one scale each day.
    """
    n = len(x)
    out = np.full((n, len(heads)), np.nan)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(outcome)
    for t in range(n):
        if not np.isfinite(x[t]).all():
            continue
        matured = np.flatnonzero(valid[:max(t - HORIZON + 1, 0)])
        if len(matured) < MIN_KEYS:
            continue
        seen = x[:t + 1][np.isfinite(x[:t + 1]).all(axis=1)]
        mean, sd = seen.mean(axis=0), seen.std(axis=0)
        sd[sd == 0] = 1.
        keys = (x[matured] - mean) / sd
        query = (x[t] - mean) / sd
        for h, columns in enumerate(heads.values()):
            cols = list(columns)
            out[t, h] = _attend(query[cols], keys[:, cols], temperature) @ outcome[matured]
    return out


def _refit_days(index: pd.DatetimeIndex) -> np.ndarray:
    """First observed day of each calendar year."""
    years = index.year.to_numpy()
    return np.r_[True, years[1:] != years[:-1]]


def _proximal_path(u: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    target = np.zeros(len(u))
    previous = 0.
    for t in range(WARMUP_DAYS, len(u)):
        if np.isfinite(u[t]) and np.isfinite(sigma[t]):
            previous = _proximal_weight(_drift(u[t], sigma[t]), sigma[t] ** 2,
                                        previous, PROXIMAL_PENALTY)
        target[t] = previous
    return target


# ------------------------------------------------------- family 1: attention

def _attention_targets(close: pd.Series, temperature: float) -> np.ndarray:
    x, sigma, outcome = _features(close)
    u = _head_outputs(x, outcome, temperature, {"all": tuple(range(x.shape[1]))})[:, 0]
    return _proximal_path(u, sigma)


# ------------------------------------------------------- family 2: multi-head

def _ridge(inputs: np.ndarray, y: np.ndarray, ridge: float):
    """Standardised ridge; penalty ridge*n shrinks each slope by ~1/(1+ridge)."""
    mean, sd = inputs.mean(axis=0), inputs.std(axis=0)
    sd[sd == 0] = 1.
    z = (inputs - mean) / sd
    beta = np.linalg.solve(z.T @ z + ridge * len(y) * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return y.mean(), mean, sd, beta


def _multihead_targets(close: pd.Series, ridge: float) -> np.ndarray:
    x, sigma, outcome = _features(close)
    heads = _head_outputs(x, outcome, 1., HEADS)
    refit = _refit_days(close.index)
    usable = np.isfinite(heads).all(axis=1) & np.isfinite(outcome)
    u = np.full(len(x), np.nan)
    model = None
    for t in range(len(x)):
        rows = np.flatnonzero(usable[:max(t - HORIZON + 1, 0)])
        if (refit[t] or model is None) and len(rows) >= MIN_FIT:
            model = _ridge(heads[rows], outcome[rows], ridge)
        if model is not None and np.isfinite(heads[t]).all():
            intercept, mean, sd, beta = model
            u[t] = intercept + ((heads[t] - mean) / sd) @ beta
    return _proximal_path(u, sigma)


# ------------------------------------------------------ family 3: transformer

D_MODEL, N_HEADS, D_FF = 8, 2, 16
ES_PAIRS, ES_SIGMA, ES_STEPS, ES_LR, ES_INIT = 24, .05, 200, .02, .1
WEIGHT_DECAY = 1e-3
SHAPES = {"we": (7, D_MODEL), "wq": (D_MODEL, D_MODEL), "wk": (D_MODEL, D_MODEL),
          "wv": (D_MODEL, D_MODEL), "wo": (D_MODEL, D_MODEL), "w1": (D_MODEL, D_FF),
          "b1": (D_FF,), "w2": (D_FF, D_MODEL), "b2": (D_MODEL,), "v": (D_MODEL,), "c": ()}
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES.values())


def _unpack(theta: np.ndarray) -> dict:
    out, k = {}, 0
    for key, shape in SHAPES.items():
        size = int(np.prod(shape))
        out[key] = theta[k:k + size].reshape(shape)
        k += size
    return out


def _positional(length: int) -> np.ndarray:
    """Vaswani et al. (2017) sinusoidal encoding, position 0 = oldest token."""
    position = np.arange(length)[:, None]
    rate = 1. / 10000 ** (np.arange(0, D_MODEL, 2) / D_MODEL)
    pe = np.zeros((length, D_MODEL))
    pe[:, 0::2] = np.sin(position * rate)
    pe[:, 1::2] = np.cos(position * rate)
    return pe


def _layer_norm(z: np.ndarray) -> np.ndarray:
    return (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-5)


def _encode(theta: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Exposure in (0, 1) for a batch of (B, L, 7) token sequences.

    Only the last position is read out, and with one causal layer its output
    depends only on its own query, so querying from the last token alone is
    exactly the full masked self-attention restricted to the readout.
    """
    p = _unpack(theta)
    batch, length, _ = tokens.shape
    dk = D_MODEL // N_HEADS
    pe = _positional(length)
    e_last = tokens[:, -1] @ p["we"] + pe[-1]
    # Keys/values are linear in e_l = x_l We + pe_l, so fold the projections
    # into the query side instead of projecting all L tokens (~4x faster;
    # tests check equality with the textbook form).
    q = (e_last @ p["wq"]).reshape(batch, N_HEADS, dk)
    r = np.einsum("bhd,ehd->bhe", q, p["wk"].reshape(D_MODEL, N_HEADS, dk))  # (B, H, d)
    scores = (r @ p["we"].T @ np.swapaxes(tokens, 1, 2) + r @ pe.T) / np.sqrt(dk)  # (B, H, L)
    a = np.exp(scores - scores.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    mixed = a @ tokens @ p["we"] + a @ pe                                     # (B, H, d)
    heads = np.einsum("bhe,ehd->bhd", mixed, p["wv"].reshape(D_MODEL, N_HEADS, dk))
    attended = heads.reshape(batch, D_MODEL) @ p["wo"]
    z = _layer_norm(e_last + attended)
    z = _layer_norm(z + np.maximum(z @ p["w1"] + p["b1"], 0.) @ p["w2"] + p["b2"])
    return 1. / (1. + np.exp(-(z @ p["v"] + p["c"])))


def _utility(theta: np.ndarray, tokens: np.ndarray, next_return: np.ndarray) -> float:
    """Mean next-day log growth net of the signal cost on exposure changes."""
    w = _encode(theta, tokens)
    turnover = np.abs(np.diff(w, prepend=w[0]))
    return float(np.mean(np.log1p(w * next_return) - SIGNAL_COST * turnover)
                 - WEIGHT_DECAY * np.mean(theta ** 2))


def _train(tokens: np.ndarray, next_return: np.ndarray, seed: int) -> np.ndarray:
    """Antithetic evolution strategies (Salimans et al. 2017) with Adam steps."""
    rng = np.random.default_rng(seed)
    theta = ES_INIT * rng.standard_normal(N_PARAMS)
    m = np.zeros(N_PARAMS)
    s = np.zeros(N_PARAMS)
    for step in range(1, ES_STEPS + 1):
        eps = rng.standard_normal((ES_PAIRS, N_PARAMS))
        diff = np.array([_utility(theta + ES_SIGMA * e, tokens, next_return)
                         - _utility(theta - ES_SIGMA * e, tokens, next_return) for e in eps])
        grad = eps.T @ diff / (2 * ES_PAIRS * ES_SIGMA)
        m = .9 * m + .1 * grad
        s = .999 * s + .001 * grad ** 2
        theta = theta + ES_LR * (m / (1 - .9 ** step)) / (np.sqrt(s / (1 - .999 ** step)) + 1e-8)
    return theta


def _windows(x: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """(n, L, 7) sequences ending at each day, and which are fully observed."""
    padded = np.vstack([np.full((length - 1, x.shape[1]), np.nan), x])
    seq = np.lib.stride_tricks.sliding_window_view(padded, (length, x.shape[1]))[:, 0]
    return seq, np.isfinite(seq).all(axis=(1, 2))


def _transformer_targets(close: pd.Series, context: int) -> np.ndarray:
    x, _, _ = _features(close)
    next_return = close.pct_change(fill_method=None).shift(-1).to_numpy()
    seq, complete = _windows(x, context)
    refit = _refit_days(close.index)
    target = np.zeros(len(x))
    model = None
    for t in range(len(x)):
        # A training row t needs its next-day return, i.e. day t+1 <= today.
        rows = np.flatnonzero(complete[:t] & np.isfinite(next_return[:t]))
        if (refit[t] or model is None) and len(rows) >= MIN_FIT:
            seen = x[:t + 1][np.isfinite(x[:t + 1]).all(axis=1)]
            mean, sd = seen.mean(axis=0), seen.std(axis=0)
            sd[sd == 0] = 1.
            norm = lambda s: np.clip((s - mean) / sd, -5., 5.)  # noqa: E731
            theta = _train(norm(seq[rows]), next_return[rows], seed=193 + int(close.index[t].year))
            model = (theta, norm)
        if model is not None and complete[t] and t >= WARMUP_DAYS:
            theta, norm = model
            target[t] = _encode(theta, norm(seq[t:t + 1]))[0]
    return target


# ---------------------------------------------------------------- strategy

def _daily_targets(close: pd.Series, config: dict) -> np.ndarray:
    family = config["family"]
    if family == "r193_attention":
        target = _attention_targets(close, config["temperature"])
    elif family == "r193_multihead":
        target = _multihead_targets(close, config["ridge"])
    elif family == "r193_transformer":
        target = _transformer_targets(close, config["context"])
    else:
        raise ValueError(f"Unknown R-193 family: {family}")
    target[:WARMUP_DAYS] = 0.
    return np.clip(target, 0., 1.)


class AttentionAllocation(Strategy):
    """Replay a frozen daily attention allocation; R-191's daily clock and drift band."""

    warmup = WARMUP_DAYS * 288
    deadband = .05

    def __init__(self, name: str) -> None:
        if name not in PARAMS:
            raise ValueError(f"Unknown R-193 strategy: {name}")
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
        df["r193_decision"] = completed & (np.cumsum(completed) > WARMUP_DAYS)
        return df

    def on_bar(self, ctx: Context) -> None:
        if not bool(ctx.bar["r193_decision"]) or ctx.equity <= 0:
            return
        target = float(ctx.bar["target"])
        held = ctx.position * ctx.close / ctx.equity
        if abs(target - held) > self.deadband or (target == 0 and ctx.position != 0):
            ctx.order_notional(target)


def make_strategy(name: str) -> AttentionAllocation:
    return AttentionAllocation(name)
