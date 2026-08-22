"""Shared, read-only utilities and pre-registration for the R-93 round (08-21).

DIRECTION, in one sentence: replace `kelly_regime_v4`'s conditional-vol-target
`scale` factor with a **Grossman & Zhou (1993)** drawdown-constrained sizing
rule -- Grossman, S. J., & Zhou, Z. (1993), "Optimal Investment Strategies for
Controlling Drawdowns," Mathematical Finance, 3(3), 241-276 -- while leaving
v4's validated 3-anchor `frac` vote untouched. Their result: an investor who
must never let wealth fall below a fraction (1 - alpha) of its running
maximum should hold a *risky-asset fraction that is itself a function of the
current drawdown from that running peak*, shrinking toward zero as the
drawdown approaches the tolerance alpha. The rule implemented here is the
simplest monotone instantiation of that idea, not Grossman & Zhou's own
closed-form CPPI-style multiplier (disclosed simplification, named up front):

    M_t = cummax(equity_0 .. equity_t)                    (running peak)
    D_t = 1 - equity_t / M_t                               (drawdown fraction)
    scale_GZ(t) = max_leverage * clip(1 - D_t / alpha, 0, 1)

i.e. full `max_leverage` at a fresh peak (D_t=0), linearly de-levered to zero
exposure once the strategy's OWN realized drawdown reaches `alpha`, floored at
zero rather than allowed to go net-short by the rule itself (v4's `frac` vote
already carries all directional information; `scale` has only ever been a
non-negative magnitude in every registered `kelly_regime*` variant).

**Which constraint it attacks: SIZE.** `scale` is exactly the factor R-62
isolated and showed carries NONE of v4's matched-exposure signature by
itself (the vote does, four independent ways: R-62, R-87, and both R-62 arms)
-- and which R-59/R-60/R-38/R-46/R-45 have now retuned or replaced 21
independent times on the SIZE axis with no promotion. This round is the 22nd,
and the first to make `scale` a function of the STRATEGY'S OWN REALIZED
DRAWDOWN rather than of market volatility (v3/v4's conditional vol-target) or
of a fixed constant (R-62's isolation arms) -- a genuinely different state
variable for the same axis: PATH-DEPENDENT ENDOGENOUS RISK, not exogenous
market vol.

**Not a duplicate of:**
- R-38, R-46 -- SIZE-axis retunes of the vol-target magnitude / functional
  form, still driven by market realized vol, not the strategy's own equity
  curve.
- R-59, R-60 -- SIZE-axis retunes of `target_vol`/`max_leverage` magnitude
  (R-59) and vote timing (R-60); neither introduces a drawdown state variable.
- R-62 -- factor ISOLATION (vote alone vs. scale alone, scale forced to a
  frozen constant or v4's own unmodified vol-target); this round does not
  isolate factors, it REPLACES `scale`'s functional form with a different one
  while leaving the vote x scale x deadband architecture intact.
- R-87 -- re-confirms the R-62 vote-carries-the-signature finding on a
  different check; does not touch `scale` at all.
- R-45 -- (named in the brief as another SIZE-axis prior; same category as
  R-38/R-46/R-59/R-60: retunes an existing magnitude/timing parameter, never
  introduces drawdown-conditioning).

None of the above make `scale` a function of the strategy's own realized
equity path. That is what this round tests, and what makes it, structurally,
a genuinely new construction rather than the 22nd retune of the same one.

**A critical structural fact, disclosed here before either branch is coded:**
`scale_GZ` depends on `equity_t`, which is the MARK-TO-MARKET VALUE OF THE
STRATEGY'S OWN ACCOUNT -- itself the product of every past position decision,
fill, fee and (on futures) funding payment. Unlike v4's `frac` (a pure
function of the OHLCV price series, vectorizable in `prepare()`) and v4's own
`scale` (a pure function of realized market volatility, likewise
vectorizable), `scale_GZ` CANNOT be precomputed in `prepare()`: it must be
evaluated bar-by-bar, live, inside `on_bar`, using `ctx.equity` -- which the
engine computes from `broker.equity(closes[i])` BEFORE calling
`strategy.on_bar(ctx)` for bar i (see `engine.run_backtest`), so `ctx.equity`
at bar i already reflects every fill up to and including bar i's open and is
available with zero lookahead. This is still fully causal (bar i's scale
uses only `equity[0..i]`, and the resulting order fills no earlier than bar
i+1's open) but it is a different construction shape than every prior
`kelly_regime*` variant's vectorized `prepare()`, and both downstream
branches must preserve it: DO NOT try to vectorize `scale_GZ` from the raw
price series -- there is no such thing as "the drawdown" independent of which
strategy, sizing, and cost model produced the equity curve being drawn down
from. `running_drawdown`/`scale_gz` below are pure, testable functions of AN
EQUITY SERIES already realized (by a backtest, or, for unit tests, a
synthetic path); `GZScaledKellyV4` below is the reference wiring that
produces that equity series live, one bar at a time, and both branches may
subclass or copy it.

**Falsification risk, named now:** because `scale_GZ` reacts to the
strategy's OWN drawdown, it is inherently reflexive -- a de-levering response
to a drawdown can itself lock in a slower recovery (selling low, structurally,
exactly the failure mode Grossman & Zhou's own paper exists to bound rather
than eliminate: their guarantee is on the FLOOR, not on subsequent upside).
Whether the SIZE axis's 22nd attempt does better than the prior 21 is an open
empirical question this module does not answer -- it only builds and
self-tests the machinery so both branches measure it identically.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both: neither may edit it, so both are measured by
identical machinery. Nothing here reads a bar at or after OOS_START
(2023-01-01); `compare()` asserts this explicitly for every slice it runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns as inference_daily_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# ---------------------------------------------------------------- splits
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

# kelly_regime_v4's own shipped constants (do not change: the control must
# be v4, not a re-parameterisation of it). Verified against
# src/tradebot/strategies/{kelly_regime.py,kelly_regime_v3.py,kelly_regime_v4.py}.
V4_HORIZONS: tuple[int, ...] = (20, 40, 80)
V4_BAND = 0.01
V4_TARGET_VOL = 0.55
V4_MAX_LEVERAGE = 2.0
V4_VOL_SPAN = 8 * BARS_PER_DAY
V4_DEADBAND = 0.10
V4_ANCHOR_SPAN_DAYS = 180
V4_HIGH_IN, V4_HIGH_OUT = 1.70, 1.20
V4_LOW_IN, V4_LOW_OUT = 0.55, 0.85


# ------------------------------------------------------------------ data

def assert_no_holdout(df: pd.DataFrame, label: str = "") -> None:
    """Fail loudly if any bar at or after the holdout boundary is present."""
    if len(df) and df.index[-1] >= pd.Timestamp(OOS_START, tz="UTC"):
        raise AssertionError(
            f"{label}: frame reaches {df.index[-1]}, at/after OOS_START={OOS_START}")


def _truncate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df[df.index < pd.Timestamp(OOS_START, tz="UTC")]
    assert_no_holdout(out, label)
    return out


def load_btc() -> pd.DataFrame:
    """The committed BTC spot series, truncated before the holdout."""
    df, _label = load_dataset(ROOT / "data", "spot")
    return _truncate(df, "BTC")


def load_eth() -> pd.DataFrame:
    """Bitfinex ETH (this project's standing cross-asset replication series)."""
    return _truncate(load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz"), "ETH")


# ================================================================== (1)
# vote_frac: kelly_regime_v4's directional vote, reproduced EXACTLY.
# Byte-for-byte the same construction as KellyRegime.prepare /
# KellyRegimeV3.prepare (KellyRegimeV4 overrides nothing but the horizons
# default) -- three latched SMA-anchor votes, averaged, optional gamma.
# ==================================================================

def _latched_anchor_vote(close: pd.Series, days: int, band: float = V4_BAND) -> pd.Series:
    """One anchor's own latched 0/1 vote, exactly as v4 computes each of its three."""
    anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
    v = pd.Series(
        np.where(close > anchor * (1.0 + band), 1.0,
                 np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
        index=close.index,
    )
    return v.ffill().fillna(0.0)


def vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
             band: float = V4_BAND, vote_gamma: float = 1.0) -> pd.Series:
    """kelly_regime_v4's own directional vote, as a standalone causal function of OHLCV.

    Identical construction to ``KellyRegime.prepare``/``KellyRegimeV3.prepare``'s
    local ``frac`` variable: for each horizon, latch bullish (1) above the
    anchor + band, bearish (0) below anchor - band, hold the previous verdict
    inside the band; average the three latched votes; optionally raise to
    ``vote_gamma`` (v4 itself uses the default 1.0, i.e. no-op). Depends only
    on ``df["close"]`` up to and including the current row -- no scale, no
    equity, no deadband.
    """
    close = df["close"]
    votes = [_latched_anchor_vote(close, days, band) for days in horizons]
    frac = sum(votes) / len(votes)
    if vote_gamma != 1.0:
        frac = frac ** vote_gamma
    return frac


def v4_vote_frac(df: pd.DataFrame) -> pd.Series:
    """kelly_regime_v4's own shipped vote (horizons=20,40,80, band=1%), for the control."""
    return vote_frac(df, V4_HORIZONS, V4_BAND)


# ------------------------------------------------- v4's own scale (control)
# Needed only to reconstruct v4's full target path as the CONTROL strategy in
# compare(); this is the factor R-93 replaces, not something either branch
# should reuse in its own candidate.

def v4_scale(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v3/v4's conditional volatility-target scale factor, reproduced exactly."""
    r = np.log(df["close"]).diff()
    vol = (r.ewm(span=V4_VOL_SPAN, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=V4_ANCHOR_SPAN_DAYS * BARS_PER_DAY,
                               min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(V4_TARGET_VOL / vol, V4_MAX_LEVERAGE)
        steady = np.minimum(V4_TARGET_VOL / slow, V4_MAX_LEVERAGE)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(df)
    out = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > V4_HIGH_IN else (-1 if x < V4_LOW_IN else 0)
            elif state == 1 and x < V4_HIGH_OUT:
                state = 0
            elif state == -1 and x > V4_LOW_OUT:
                state = 0
        out[i] = full[i] if state != 0 else steady[i]
    return out


def apply_deadband(desired: np.ndarray, deadband: float = V4_DEADBAND) -> np.ndarray:
    """v4's own 10% re-target deadband, applied to a desired-exposure path."""
    target = np.zeros(len(desired))
    pos = 0.0
    for i, d in enumerate(desired):
        if abs(d - pos) > deadband:
            pos = float(d)
        target[i] = pos
    return target


def v4_raw_desired(df: pd.DataFrame) -> np.ndarray:
    """v4's desired exposure BEFORE its own 10% deadband: frac * scale."""
    return v4_vote_frac(df).to_numpy() * v4_scale(df)


def v4_target(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v4's complete, final target path (post-deadband) -- the control."""
    return apply_deadband(v4_raw_desired(df))


# ================================================================== (2)+(3)
# running_drawdown / scale_gz: Grossman & Zhou (1993) drawdown-constrained
# sizing, as PURE functions of an already-realized equity series. Both are
# trivially causal by construction (cummax and elementwise ops look only
# backward); causal_truncation_probe below proves it rather than asserting it.
# ==================================================================

def running_drawdown(equity: pd.Series) -> pd.Series:
    """Causal running peak and drawdown fraction of an equity curve.

    ``M_t = cummax(equity_0 .. equity_t)``, ``D_t = 1 - equity_t / M_t``.
    Bar i's value depends only on ``equity[:i+1]`` -- ``cummax`` cannot see
    forward by definition. Returns ``D_t`` (a ``pd.Series``, same index).
    """
    equity = equity.astype(float)
    peak = equity.cummax()
    peak = peak.where(peak > 0, np.nan)  # guard a non-positive/zero peak (dead account)
    dd = 1.0 - equity / peak
    return dd.fillna(0.0)


def scale_gz(equity: pd.Series, alpha: float, max_leverage: float = 2.0) -> pd.Series:
    """Grossman & Zhou (1993) drawdown-constrained scale.

    ``max_leverage * clip(1 - D_t/alpha, 0, 1)`` where ``D_t`` is
    ``running_drawdown(equity)``: full ``max_leverage`` at a fresh peak,
    linearly de-levered to zero once the drawdown reaches ``alpha``, floored
    at zero (the rule is a MAGNITUDE only -- direction is `vote_frac`'s job).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    d = running_drawdown(equity)
    raw = 1.0 - d / alpha
    return max_leverage * raw.clip(lower=0.0, upper=1.0)


# ------------------------------------------------------- reference wiring
# GZScaledKellyV4: v4's vote (vectorized in prepare()) x GZ's drawdown scale
# (evaluated live in on_bar(), because it depends on the account's own
# realized equity -- see the module docstring's "critical structural fact").
# Both branches may subclass or copy this; nothing about it is required to
# survive into their own files unmodified.

class GZScaledKellyV4(Strategy):
    """kelly_regime_v4 with its conditional-vol-target `scale` replaced by
    Grossman & Zhou (1993) drawdown-constrained sizing. `frac` (the vote) is
    v4's own unmodified construction; only `scale` differs.

    `scale` is recomputed every bar from ``ctx.equity`` -- the account's own
    running peak and drawdown -- rather than from a precomputed column, since
    it cannot be vectorized ahead of the backtest that produces it (see the
    module docstring). Internal state (`_peak`, `_pos`) is reset in
    `prepare()`, which the engine calls once at the start of every
    `run_backtest`/`run_period`, so a single instance is safe to reuse across
    multiple runs (as `compare()` below does).
    """

    name = "r93_gz_scaled_kelly"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, alpha: float = 0.3, max_leverage: float = V4_MAX_LEVERAGE,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 deadband: float = V4_DEADBAND) -> None:
        self.alpha = alpha
        self.max_leverage = max_leverage
        self.horizons = horizons
        self.band = band
        self.deadband = deadband
        self._peak: float | None = None
        self._pos = 0.0
        # `target` cannot be precomputed in prepare() (see module docstring),
        # so exposure is logged live here instead of as a df column -- the
        # convention run_slice()/compare() fall back to for any strategy
        # without a "target" column.
        self._exposure_log: list[float] = []

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_frac"] = vote_frac(df, self.horizons, self.band).to_numpy()
        self._peak = None
        self._pos = 0.0
        self._exposure_log = []
        return df

    def on_bar(self, ctx: Context) -> None:
        eq = ctx.equity
        self._peak = eq if self._peak is None else max(self._peak, eq)
        d = 1.0 - eq / self._peak if self._peak > 0 else 0.0
        scale = self.max_leverage * min(max(1.0 - d / self.alpha, 0.0), 1.0)
        frac = float(ctx.bar["_frac"])
        desired = frac * scale

        prev_pos = self._pos
        if abs(desired - prev_pos) > self.deadband:
            self._pos = desired
        if abs(self._pos - prev_pos) > 1e-9:
            ctx.order_notional(self._pos)
        self._exposure_log.append(self._pos)


# ================================================================== (4)
# causal_truncation_probe: this project's standard no-lookahead defense
# (the R-21 lesson -- a subtle full-series peek is worth +2.1 Sharpe and
# passes a naive test). Two probes: one for vote_frac (pure df -> series),
# one for the equity-based GZ machinery, with an explicit peak-monotonicity
# check.
# ==================================================================

def causal_truncation_probe_vote(df: pd.DataFrame,
                                 cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    """Truncate the input frame to [:k] and recompute vote_frac; the shared
    prefix must match the full-series computation exactly."""
    full = vote_frac(df).to_numpy()
    for cut in cuts:
        k = int(len(df) * cut)
        part = vote_frac(df.iloc[:k]).to_numpy()
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-12, rtol=0.0):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-12, rtol=0.0)))
            raise AssertionError(f"vote_frac causality FAIL at cut={cut}: {bad} bars differ")
    return True


def causal_truncation_probe_gz(equity: pd.Series, alpha: float = 0.3,
                               max_leverage: float = 2.0,
                               cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    """Truncate an equity series to [:k] and recompute running_drawdown/scale_gz;
    the shared prefix must match exactly, AND the truncated-frame running peak
    must never exceed the true full-series causal peak at any bar (the R-21
    style check: a lookahead bug that used the FULL series' max instead of a
    running cummax would show up here as the truncated peak being LOWER than
    what a full-series max would report at the same bar, while the causal
    peak itself stays identical between the two computations -- which is
    exactly what we assert).
    """
    full_dd = running_drawdown(equity)
    full_scale = scale_gz(equity, alpha, max_leverage)
    full_peak = equity.cummax()
    n = len(equity)
    for cut in cuts:
        k = int(n * cut)
        if k < 2:
            continue
        eq_k = equity.iloc[:k]
        part_dd = running_drawdown(eq_k)
        part_scale = scale_gz(eq_k, alpha, max_leverage)
        part_peak = eq_k.cummax()

        if not np.allclose(full_dd.iloc[:k].to_numpy(), part_dd.to_numpy(),
                           atol=1e-12, rtol=0.0):
            raise AssertionError(f"running_drawdown causality FAIL at cut={cut}")
        if not np.allclose(full_scale.iloc[:k].to_numpy(), part_scale.to_numpy(),
                           atol=1e-12, rtol=0.0):
            raise AssertionError(f"scale_gz causality FAIL at cut={cut}")
        if not np.array_equal(full_peak.iloc[:k].to_numpy(), part_peak.to_numpy()):
            raise AssertionError(f"running peak on truncated frame diverges at cut={cut}")
        # Never uses equity from bar k or later: perturbing the tail must not
        # move any value inside the shared prefix.
        perturbed = equity.copy()
        perturbed.iloc[k:] = perturbed.iloc[k:] * 1e6 + 1e9
        pert_dd = running_drawdown(perturbed)
        pert_scale = scale_gz(perturbed, alpha, max_leverage)
        if not np.allclose(pert_dd.iloc[:k].to_numpy(), full_dd.iloc[:k].to_numpy(),
                           atol=1e-12, rtol=0.0):
            raise AssertionError(f"running_drawdown peeks at bar>=k, cut={cut}")
        if not np.allclose(pert_scale.iloc[:k].to_numpy(), full_scale.iloc[:k].to_numpy(),
                           atol=1e-12, rtol=0.0):
            raise AssertionError(f"scale_gz peeks at bar>=k, cut={cut}")
    return True


def causal_truncation_probe(df: pd.DataFrame, equity: pd.Series | None = None,
                            alpha: float = 0.3, max_leverage: float = 2.0,
                            cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    """Full probe: vote_frac on ``df``, plus running_drawdown/scale_gz on
    ``equity`` if given (a real backtest equity curve, or a synthetic one)."""
    ok = causal_truncation_probe_vote(df, cuts)
    if equity is not None:
        ok = ok and causal_truncation_probe_gz(equity, alpha, max_leverage, cuts)
    return ok


# ================================================================== (5)
# compare(): run a GZScaledKellyV4 config over inner-train, inner-validation
# and the ETH replication slice, vs kelly_regime_v4, never touching OOS_START.
# ==================================================================

SLICES: dict[str, tuple[str | None, str | None]] = {
    "inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
    "inner_val": (INNER_VAL_START, INNER_VAL_END),
}
# ETH's own frame is already truncated < OOS_START by load_eth(); this slice
# uses its full available history (from first coverage, ~2019-03-14) rather
# than the BTC-denominated inner_train/inner_val boundaries, matching
# r89-r92's "ETH replication" convention.
ETH_SLICE_NAME = "eth_replication"

for _name, (_s, _e) in SLICES.items():
    if _e is not None:
        assert pd.Timestamp(_e) < pd.Timestamp(OOS_START), (
            f"SLICES[{_name!r}] end={_e} is not before OOS_START={OOS_START}")


@dataclass
class SliceResult:
    name: str
    market: str
    final_balance: float
    sharpe: float
    max_drawdown_pct: float
    num_trades: int
    log_growth: float
    daily: np.ndarray
    mean_abs_exposure: float
    realized_vol: float


def daily_simple_returns(equity: pd.Series) -> np.ndarray:
    """Daily SIMPLE returns of a bar-frequency equity curve."""
    return inference_daily_returns(equity).to_numpy()


def run_slice(strategy: Strategy, df: pd.DataFrame, start: str | None, end: str | None,
             slice_name: str, market: MarketSpec = SPOT,
             balance: float = 1_000.0) -> SliceResult:
    """One backtest over an explicit [start, end] window, with a warm prefix.

    Asserts the guard directly on the window boundary before running: no
    call site of this function can silently drift the window past OOS_START.
    """
    if end is not None:
        assert pd.Timestamp(end) < pd.Timestamp(OOS_START), (
            f"run_slice({slice_name!r}): end={end} is not before OOS_START={OOS_START}")
    assert_no_holdout(df, slice_name)  # the source frame itself must already be pre-holdout

    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    assert_no_holdout(res.equity.to_frame(), f"{slice_name} result")
    m = compute_metrics(res)
    d = daily_simple_returns(res.equity)
    if "target" in res.df.columns:
        exposure = res.df["target"].to_numpy()
    elif getattr(strategy, "_exposure_log", None):
        # Strategies whose target cannot be vectorized in prepare() (e.g.
        # GZScaledKellyV4, whose scale depends on the account's own realized
        # equity) log their applied exposure live in on_bar instead.
        exposure = np.asarray(strategy._exposure_log, dtype=float)
    else:
        exposure = np.array([np.nan])
    return SliceResult(
        name=slice_name, market=market.name, final_balance=m.final_balance,
        sharpe=m.sharpe, max_drawdown_pct=m.max_drawdown_pct,
        num_trades=m.num_trades, log_growth=float(total_log_return(d)), daily=d,
        mean_abs_exposure=float(np.nanmean(np.abs(exposure))),
        realized_vol=float(np.nanstd(d) * np.sqrt(365.25)) if len(d) > 1 else float("nan"),
    )


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy
    (used here only for the CONTROL, kelly_regime_v4, whose full target path
    is a pure function of price and so is safely vectorizable)."""

    name = "r93_control"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, build_target, name: str = "r93_control",
                warmup: int | None = None) -> None:
        self._build = build_target
        self.name = name
        if warmup is not None:
            self.warmup = warmup

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = np.asarray(self._build(df), dtype=float)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


def paired_diff(candidate: np.ndarray, control: np.ndarray, *,
                mean_block: float = 30.0, n_boot: int = 2_000, seed: int = 0):
    """Paired stationary-block-bootstrap difference in total log growth."""
    n = min(len(candidate), len(control))
    return paired_bootstrap(np.asarray(candidate[-n:], dtype=float),
                            np.asarray(control[-n:], dtype=float),
                            total_log_return, mean_block=mean_block,
                            n_boot=n_boot, seed=seed)


def compare(candidate: Strategy, *, label: str, btc: pd.DataFrame | None = None,
           eth: pd.DataFrame | None = None, control_build=None,
           markets: tuple[MarketSpec, ...] = (SPOT, FUTURES),
           include_eth: bool = True, seed: int = 0) -> list[dict]:
    """Candidate (a GZScaledKellyV4-shaped Strategy instance) vs kelly_regime_v4
    on inner-train, inner-validation, and the ETH replication slice, on every
    market. Never reads a bar at or after OOS_START -- guarded three ways:
    SLICES' own end dates are asserted at import time above, every source
    frame is asserted pre-holdout, and every result equity curve is asserted
    pre-holdout after the run.
    """
    if control_build is None:
        control_build = v4_target
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare(): btc")
    if include_eth and eth is None:
        eth = load_eth()
    if include_eth:
        assert_no_holdout(eth, "compare(): eth")

    ctrl = TargetStrategy(control_build, name="kelly_regime_v4")

    rows = []
    jobs = [(name, start, end, btc) for name, (start, end) in SLICES.items()]
    if include_eth:
        jobs.append((ETH_SLICE_NAME, None, None, eth))

    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice(candidate, df, start, end, slice_name, market)
            b = run_slice(ctrl, df, start, end, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                        if b.mean_abs_exposure else float("nan"))
            vol_ratio = (a.realized_vol / b.realized_vol
                        if b.realized_vol else float("nan"))
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "cand_final": a.final_balance, "ctrl_final": b.final_balance,
                "cand_log_growth": a.log_growth, "ctrl_log_growth": b.log_growth,
                "d_log_growth": a.log_growth - b.log_growth,
                "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                "d_sharpe": a.sharpe - b.sharpe,
                "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
                "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
                "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
                "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
                "risk_matched": bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                                if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False,
                "boot_d_loggrowth": pr.diff.point,
                "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


def print_rows(rows: list[dict]) -> None:
    """One fixed-width line per cell, so branches' output is diffable."""
    hdr = (f"{'label':22s} {'slice':16s} {'market':11s} {'cand$':>10s} {'ctrl$':>10s} "
          f"{'dSh':>6s} {'dDD':>7s} {'expR':>5s} {'volR':>5s} {'RM':>3s} "
          f"{'dlogG':>7s} {'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:22]:22s} {r['slice']:16s} {r['market']:11s} "
              f"{r['cand_final']:10,.0f} {r['ctrl_final']:10,.0f} "
              f"{r['d_sharpe']:+6.2f} {r['d_dd']:+7.1f} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['boot_d_loggrowth']:+7.3f} {r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def fee_at(market: MarketSpec, fee_rate: float) -> MarketSpec:
    """Same market spec, at a different taker fee (cost-robustness checks)."""
    return MarketSpec(name=market.name, leverage=market.leverage, fee_rate=fee_rate,
                      allow_short=market.allow_short,
                      maintenance_margin_rate=market.maintenance_margin_rate,
                      min_notional=market.min_notional, pays_funding=market.pays_funding)


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    """Fast checks on synthetic data. Run on import (mirrors r89-r92_shared.py's
    convention); the real-data vote_frac/causal-probe/compare() sanity checks
    the round's brief asks for are run separately, against the committed CSV,
    not on every import."""
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(93)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    # (1) vote_frac: self-consistency of the generalised form at v4's own
    # horizons, and the deadband/target reconstruction it feeds.
    raw = v4_raw_desired(df)
    assert np.allclose(v4_target(df), apply_deadband(raw)), \
        "v4_target != apply_deadband(v4_raw_desired)"
    assert np.array_equal(v4_vote_frac(df).to_numpy(), vote_frac(df, V4_HORIZONS).to_numpy())
    assert vote_frac(df).between(0.0, 1.0).all()

    # (2)/(3) running_drawdown / scale_gz on a synthetic equity path with a
    # known peak-then-drawdown-then-recovery shape.
    eq = pd.Series(
        np.concatenate([np.linspace(1000, 2000, 500),
                        np.linspace(2000, 1200, 300),
                        np.linspace(1200, 1800, 400)]),
        index=idx[:1200])
    dd = running_drawdown(eq)
    assert dd.iloc[499] == 0.0  # sitting exactly at a fresh peak
    assert abs(dd.iloc[799] - (1.0 - 1200 / 2000)) < 1e-9  # trough drawdown = 40%
    assert (dd >= 0.0).all() and (dd <= 1.0).all()
    sc = scale_gz(eq, alpha=0.3, max_leverage=2.0)
    assert abs(sc.iloc[499] - 2.0) < 1e-9  # fresh peak -> full max_leverage
    assert sc.iloc[799] == 0.0  # 40% drawdown > 30% alpha -> fully de-levered
    assert (sc >= 0.0).all() and (sc <= 2.0).all()

    # (4) causal truncation probes.
    assert causal_truncation_probe_vote(df)
    assert causal_truncation_probe_gz(eq, alpha=0.3, max_leverage=2.0)

    # GZScaledKellyV4 wiring smoke test (tiny, fast).
    strat = GZScaledKellyV4(alpha=0.3)
    from tradebot.engine import run_backtest
    res = run_backtest(strat, df.iloc[:20_000], SPOT, 1_000.0)
    assert np.isfinite(res.equity).all()
    assert not res.liquidated


_self_test()
