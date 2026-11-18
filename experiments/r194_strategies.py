"""R-194: three frozen, unregistered Kelly v4 x transformer hybrids.

Each keeps ``kelly_regime_v4``'s anchor vote, conditional volatility scale,
native 10% target latch and target-change callback, and changes one thing
with a transformer component computed on R-191's complete-UTC-day clock:

1. ``r194_panel_cross_attention`` - cross-attention from the traded asset's
   daily state into a memory pooled from nine instruments' matured outcomes;
   its drift sign is a fourth vote beside the 20/40/80-day anchors.
2. ``r194_kelly_episodic_memory`` - attention whose keys include v4's own
   vote and scale; the retrieved conditional Kelly fraction, relative to the
   unconditional one, multiplies v4's target in [0, 2].
3. ``r194_residual_transformer`` - a one-layer encoder (R-193 architecture)
   whose output head starts at zero, so the untrained model is exactly v4;
   ES training learns a multiplicative correction exp(tanh(z)) net of cost.

Until a component has enough matured history it is the identity, so every
hybrid is v4 until its transformer part can speak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.r191_strategies import SIGNAL_COST
from experiments.r192_strategies import _latch, _parent_features
from experiments.r193_strategies import (HORIZON, MIN_FIT, MIN_KEYS, _attend, _drift,
                                         _features, _layer_norm, _positional,
                                         _refit_days, _windows)
from tradebot.data import load_ohlcv_csv
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

DATA = Path(__file__).resolve().parents[1] / "data"
PANEL_FILES = ("btcusd_spot_5m.csv.gz", "ethusd_coinbase_spot_5m.csv.gz",
               "ltcusd_coinbase_spot_5m.csv.gz", "xrpusd_coinbase_spot_5m.csv.gz",
               "linkusd_coinbase_spot_5m.csv.gz", "bchusd_coinbase_spot_5m.csv.gz",
               "dashusd_coinbase_spot_5m.csv.gz", "etcusd_coinbase_spot_5m.csv.gz",
               "xtzusd_coinbase_spot_5m.csv.gz")
MAX_LEVERAGE = KellyRegimeV4().max_leverage

PRIMARY_NAMES = ("r194_panel_cross_attention", "r194_kelly_episodic_memory",
                 "r194_residual_transformer")
FAMILIES = {name: (name, name + "_low", name + "_high") for name in PRIMARY_NAMES}
ALL_NAMES = tuple(name for family in FAMILIES.values() for name in family)
LABELS = dict(zip(PRIMARY_NAMES, (
    "Kelly v4 + nine-instrument cross-attention vote",
    "Kelly v4 x episodic-memory Kelly ratio",
    "Kelly v4 x identity-initialised residual transformer",
)))
PARAMS = {}
for _family, _parameter, _values in (
    (PRIMARY_NAMES[0], "weight", (1., .5, 2.)),
    (PRIMARY_NAMES[1], "temperature", (1., .5, 2.)),
    (PRIMARY_NAMES[2], "context", (30, 15, 60)),
):
    for _name, _value in zip(FAMILIES[_family], _values):
        PARAMS[_name] = {"family": _family, _parameter: _value}
        if _name != _family:
            LABELS[_name] = LABELS[_family] + f" ({_parameter}={_value:g})"


# ---------------------------------------------------------------- daily clock

def _complete_days(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """R-191's clock: the 23:55 bar of each day with all 288 aligned bars."""
    ts = pd.DatetimeIndex(df.index)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    aligned = ((ts.minute % 5 == 0) & (ts.second == 0)
               & (ts.microsecond == 0) & (ts.nanosecond == 0))
    counts = pd.Series(aligned, index=ts).groupby(ts.normalize()).transform("sum").to_numpy()
    completed = (ts.hour == 23) & (ts.minute == 55) & aligned & (counts == 288)
    return completed, pd.Series(df["close"].to_numpy()[completed], index=ts[completed])


def _to_bars(daily: np.ndarray, completed: np.ndarray, before: float) -> np.ndarray:
    """A day's value applies from its own 23:55 close bar onward."""
    position = np.cumsum(completed) - 1
    out = np.full(len(completed), before)
    seen = position >= 0
    out[seen] = daily[position[seen]]
    return np.where(np.isfinite(out), out, before)


def _daily_close(df: pd.DataFrame) -> pd.Series:
    return _complete_days(df)[1]


# -------------------------------------------- family 1: panel cross-attention

def _panel_drift(close: pd.Series, panel: list[pd.Series], temperature: float = 1.) -> np.ndarray:
    """Daily arithmetic drift of ``close`` read from a pooled multi-asset memory.

    A memory (asset, day j) becomes readable on the date of that asset's
    (j + HORIZON)-th observation, so pooling never leaks across calendars.
    Keys are standardised with the moments of the memories readable today.
    """
    xq, sigma, _ = _features(close)
    keys, values, ready = [], [], []
    for series in panel:
        x, _, outcome = _features(series)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(outcome)
        ok[len(series) - HORIZON:] = False
        index = np.flatnonzero(ok)
        keys.append(x[index])
        values.append(outcome[index])
        ready.append(series.index[index + HORIZON].to_numpy())
    if not keys:
        return np.full(len(close), np.nan)
    order = np.argsort(np.concatenate(ready), kind="stable")
    keys = np.concatenate(keys)[order]
    values = np.concatenate(values)[order]
    ready = np.concatenate(ready)[order]
    sums, squares = np.cumsum(keys, axis=0), np.cumsum(keys ** 2, axis=0)
    drift = np.full(len(close), np.nan)
    for t, date in enumerate(close.index.to_numpy()):
        n = int(np.searchsorted(ready, date, side="right"))
        if n < MIN_KEYS or not np.isfinite(xq[t]).all() or not np.isfinite(sigma[t]):
            continue
        mean = sums[n - 1] / n
        sd = np.sqrt(np.maximum(squares[n - 1] / n - mean ** 2, 0.))
        sd[sd == 0] = 1.
        weights = _attend((xq[t] - mean) / sd, (keys[:n] - mean) / sd, temperature)
        drift[t] = _drift(weights @ values[:n], sigma[t])
    return drift


def load_panel(end: pd.Timestamp) -> list[pd.Series]:
    """Complete-day closes of the nine committed instruments, cut at ``end``."""
    return [_daily_close(load_ohlcv_csv(DATA / name).loc[:end]) for name in PANEL_FILES]


# ------------------------------------------ family 2: episodic Kelly memory

def _episodic_multiplier(close: pd.Series, frac: np.ndarray, scale: np.ndarray,
                         temperature: float = 1.) -> np.ndarray:
    """Conditional / unconditional Kelly fraction from v4-state-keyed attention.

    Keys are R-193's seven features plus v4's daily vote and scale; values are
    matured vol-normalised 20-day outcomes. Kelly in outcome units is mean /
    variance; the ratio to the uniform-attention (unconditional) Kelly
    fraction is clipped to [0, 2]. It is 1 when no ratio is defined.
    """
    x, _, outcome = _features(close)
    keys_all = np.column_stack([x, frac, scale])
    valid = np.isfinite(keys_all).all(axis=1) & np.isfinite(outcome)
    seen_ok = np.isfinite(keys_all).all(axis=1)
    sums = np.cumsum(np.where(seen_ok[:, None], keys_all, 0.), axis=0)
    squares = np.cumsum(np.where(seen_ok[:, None], keys_all ** 2, 0.), axis=0)
    counts = np.cumsum(seen_ok)
    out = np.ones(len(close))
    for t in range(len(close)):
        matured = np.flatnonzero(valid[:max(t - HORIZON + 1, 0)])
        if len(matured) < MIN_KEYS or not seen_ok[t]:
            continue
        mean = sums[t] / counts[t]
        sd = np.sqrt(np.maximum(squares[t] / counts[t] - mean ** 2, 0.))
        sd[sd == 0] = 1.
        u = outcome[matured]
        weights = _attend((keys_all[t] - mean) / sd, (keys_all[matured] - mean) / sd, temperature)
        mu, var = weights @ u, weights @ u ** 2 - (weights @ u) ** 2
        mu0, var0 = u.mean(), u.var()
        if var > 0 and var0 > 0 and mu0 > 0:
            out[t] = float(np.clip((mu / var) / (mu0 / var0), 0., 2.))
    return out


# ------------------------------------ family 3: residual transformer on v4

N_IN, D_MODEL, N_HEADS, D_FF = 9, 8, 2, 16
ES_PAIRS, ES_SIGMA, ES_STEPS, ES_LR, ES_INIT = 24, .05, 200, .02, .1
WEIGHT_DECAY = 1e-3
SHAPES = {"we": (N_IN, D_MODEL), "wq": (D_MODEL, D_MODEL), "wk": (D_MODEL, D_MODEL),
          "wv": (D_MODEL, D_MODEL), "wo": (D_MODEL, D_MODEL), "w1": (D_MODEL, D_FF),
          "b1": (D_FF,), "w2": (D_FF, D_MODEL), "b2": (D_MODEL,), "v": (D_MODEL,), "c": ()}
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES.values())
HEAD = slice(N_PARAMS - D_MODEL - 1, N_PARAMS)  # "v" and "c", the output head


def _unpack(theta: np.ndarray) -> dict:
    out, k = {}, 0
    for key, shape in SHAPES.items():
        size = int(np.prod(shape))
        out[key] = theta[k:k + size].reshape(shape)
        k += size
    return out


def _encode(theta: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Log-multiplier tanh(z) for (B, L, 9) sequences; R-193's folded encoder."""
    p = _unpack(theta)
    batch, length, _ = tokens.shape
    dk = D_MODEL // N_HEADS
    pe = _positional(length)
    e_last = tokens[:, -1] @ p["we"] + pe[-1]
    q = (e_last @ p["wq"]).reshape(batch, N_HEADS, dk)
    r = np.einsum("bhd,ehd->bhe", q, p["wk"].reshape(D_MODEL, N_HEADS, dk))
    scores = (r @ p["we"].T @ np.swapaxes(tokens, 1, 2) + r @ pe.T) / np.sqrt(dk)
    a = np.exp(scores - scores.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    mixed = a @ tokens @ p["we"] + a @ pe
    heads = np.einsum("bhe,ehd->bhd", mixed, p["wv"].reshape(D_MODEL, N_HEADS, dk))
    z = _layer_norm(e_last + heads.reshape(batch, D_MODEL) @ p["wo"])
    z = _layer_norm(z + np.maximum(z @ p["w1"] + p["b1"], 0.) @ p["w2"] + p["b2"])
    return np.tanh(z @ p["v"] + p["c"])


def _utility(theta, tokens, base, next_return) -> float:
    """Net log growth of v4's daily target times exp(correction), capped at 2x."""
    w = np.clip(base * np.exp(_encode(theta, tokens)), 0., MAX_LEVERAGE)
    turnover = np.abs(np.diff(w, prepend=w[0]))
    growth = np.log1p(np.maximum(w * next_return, -.99))
    return float(np.mean(growth - SIGNAL_COST * turnover) - WEIGHT_DECAY * np.mean(theta ** 2))


def _initial_theta(rng: np.random.Generator) -> np.ndarray:
    theta = ES_INIT * rng.standard_normal(N_PARAMS)
    theta[HEAD] = 0.  # exact identity: the untrained model reproduces v4
    return theta


def _train(tokens, base, next_return, seed: int) -> np.ndarray:
    """Antithetic ES with Adam, from the identity initialisation."""
    rng = np.random.default_rng(seed)
    theta = _initial_theta(rng)
    m, s = np.zeros(N_PARAMS), np.zeros(N_PARAMS)
    for step in range(1, ES_STEPS + 1):
        eps = rng.standard_normal((ES_PAIRS, N_PARAMS))
        diff = np.array([_utility(theta + ES_SIGMA * e, tokens, base, next_return)
                         - _utility(theta - ES_SIGMA * e, tokens, base, next_return) for e in eps])
        grad = eps.T @ diff / (2 * ES_PAIRS * ES_SIGMA)
        m = .9 * m + .1 * grad
        s = .999 * s + .001 * grad ** 2
        theta = theta + ES_LR * (m / (1 - .9 ** step)) / (np.sqrt(s / (1 - .999 ** step)) + 1e-8)
    return theta


def _residual_multiplier(close: pd.Series, frac: np.ndarray, scale: np.ndarray,
                         base: np.ndarray, context: int) -> np.ndarray:
    """exp(tanh(z)) per day; 1 until the first fit (252 rows), refit yearly."""
    x, _, _ = _features(close)
    tokens = np.column_stack([x, frac, scale])
    tokens[~np.isfinite(tokens).all(axis=1)] = np.nan
    next_return = close.pct_change(fill_method=None).shift(-1).to_numpy()
    seq, complete = _windows(tokens, context)
    refit = _refit_days(close.index)
    out = np.ones(len(close))
    model = None
    for t in range(len(close)):
        rows = np.flatnonzero(complete[:t] & np.isfinite(next_return[:t]) & np.isfinite(base[:t]))
        if (refit[t] or model is None) and len(rows) >= MIN_FIT:
            seen = tokens[:t + 1][np.isfinite(tokens[:t + 1]).all(axis=1)]
            mean, sd = seen.mean(axis=0), seen.std(axis=0)
            sd[sd == 0] = 1.

            def norm(s, mean=mean, sd=sd):
                return np.clip((s - mean) / sd, -5., 5.)

            theta = _train(norm(seq[rows]), base[rows], next_return[rows],
                           seed=194 + int(close.index[t].year))
            model = (theta, norm)
        if model is not None and complete[t]:
            theta, norm = model
            out[t] = float(np.exp(_encode(theta, norm(seq[t:t + 1]))[0]))
    return out


# ---------------------------------------------------------------- strategy

class KellyTransformer(KellyRegimeV4):
    """Kelly v4 with one transformer component; native latch and callback."""

    def __init__(self, name: str, panel=None) -> None:
        super().__init__()
        if name not in PARAMS:
            raise ValueError(f"Unknown R-194 strategy: {name}")
        self.name = name
        self.config = dict(PARAMS[name])
        self.panel = panel  # tests inject synthetic panels; None loads data/

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        family = self.config["family"]
        parts = _parent_features(df)
        frac = parts["frac"]
        scale = np.where(parts["state"] != 0, parts["full"], parts["steady"])
        completed, close = _complete_days(df)
        if family == "r194_panel_cross_attention":
            panel = self.panel if self.panel is not None else load_panel(df.index[-1])
            drift = _panel_drift(close, panel)
            vote = _to_bars(np.where(np.isfinite(drift), (drift > 0).astype(float), np.nan),
                            completed, np.nan)
            w = self.config["weight"]
            # R-55's confirming-vote fraction; plain v4 while the vote is absent.
            frac = np.where(np.isfinite(vote), (len(self.horizons) * frac + w * vote)
                            / (len(self.horizons) + w), frac)
            desired = frac * scale
        else:
            daily_frac, daily_scale = frac[completed], scale[completed]
            if family == "r194_kelly_episodic_memory":
                multiplier = _episodic_multiplier(close, daily_frac, daily_scale,
                                                  self.config["temperature"])
            else:
                base = np.minimum(daily_frac * daily_scale, MAX_LEVERAGE)
                multiplier = _residual_multiplier(close, daily_frac, daily_scale, base,
                                                  self.config["context"])
            desired = np.minimum(frac * scale * _to_bars(multiplier, completed, 1.), MAX_LEVERAGE)
        df["target"] = _latch(desired)
        return df


def make_strategy(name: str, panel=None) -> KellyTransformer:
    return KellyTransformer(name, panel=panel)
