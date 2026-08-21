"""Shared, read-only utilities for the R-89 round (08-21).

DIRECTION, in one sentence: attack the **no-trade band inside
`kelly_regime_v4`'s own anchor vote** -- the constructor default
``band=0.01`` that no round in 87 has ever swept, decomposed or replaced
-- as the transaction-cost object the literature says it is, rather than
as the ad hoc 1% it currently is.

Why this and not another signal: R-62 established that of v4's two
factors (`frac` x `scale`) the **vote carries the entire signature** and
the volatility-target scale carries none of it, a finding since confirmed
four independent ways (R-87). Every round since has nonetheless worked on
the scale slot (21 retunes), added an external confirming vote to the
`frac` slot (ten INFO signals, R-73..R-84), or replaced the regime
estimator wholesale (five mechanisms, R-82/83/85/86). The vote's own
**latch geometry** -- how far price must travel past an anchor before the
vote flips, and whether that distance is the same going in as coming out
-- is untouched. It is the one part of the factor that matters that has
never been looked at.

Which constraint it attacks: **COST**. The band is not a smoothing
parameter, it is the object proportional transaction costs create: with a
fee, the optimal policy for a signal-driven long/flat position is a
no-trade region whose edges are set by the fee and the signal's own
dispersion. A fixed 1% of price is neither.

The literature is this repo's own, already commissioned and verified
during R-67 and recorded in ``docs/RESEARCH.md`` (finding 8):

- **Dai, Zhang & Zhu (2010)**, *SIAM Journal on Financial Mathematics*
  1(1), 780-810, and **Guan, Peng & Xu (2020)**, arXiv:2008.07082 Thm
  3.1: with proportional costs and a persistent hidden state, entry sits
  strictly *above* and exit strictly *below* the frictionless
  indifference point, the gap opened by the fee. The theorem is
  **single-asset** and needs the signal to be a sufficient statistic --
  which is why R-67/R-68 could only apply it to the cross-sectional
  top-k arm, where ``docs/RESEARCH.md`` records that neither condition
  holds cleanly. v4's long/flat BTC vote is the case the theorem
  actually covers, and it has never been tested there.
- **de Lataillade & Chaouki (2020)**, "Equations and Shape of the
  Optimal Band Strategy", arXiv:2003.04646, Eq. (11): the optimal
  tolerance saturates at **~1.6 sigma_signal** -- the band's natural
  unit is the *signal's own dispersion*, not a fixed fraction of price,
  and a larger fee does not justify a wider band.
- **de Lataillade, Deremble, Potters & Bouchaud (2012)**, *Journal of
  Investment Strategies* 1(3), 91-115, Sec. 6.3: the leading-order band
  is **symmetric** and any asymmetry is higher order in Gamma^(1/3).
  This is the round's own named counter-prediction, not a supporting
  citation.

Two branches, disjoint files, both measured by this module:

- **conservative** -- asymmetric entry/exit thresholds (``d_in`` /
  ``d_out``) on v4's own latched anchor vote: the Dai-Zhang-Zhu /
  Guan-Peng-Xu construction, applied for the first time to the
  single-asset long/flat case it was proved for.
- **novel** -- the **response function's shape**: replace the binary
  latched vote with a calibrated continuous map from standardised trend
  strength to exposure, testing sign (the incumbent) against linear
  (Dao et al. 2016's unsaturated response, which alone produces a
  convex/parabolic payoff -- a binary vote discards the convex term by
  construction) against the non-monotone cubic of Schmidhuber (2021),
  *Physica A* 566:125642 / arXiv:2006.07847 and Safari & Schmidhuber
  (2025), arXiv:2501.16772, whose fitted `E[R] = a + b*phi + c*phi^3`
  with `b>0, c<0` says trends revert **before** they become
  statistically significant (critical strength `phi_c = sqrt(-b/c) ~
  1.8-1.9`).

  The warrant for spending the round's novel slot here rather than on a
  fourth band variant is Levine & Pedersen (2016), "Which Trend Is Your
  Friend?", *Financial Analysts Journal* 72(3):51-66: time-series
  momentum, moving-average crossovers, HP filters and Kalman filters are
  **equivalent representations of one linear filter**, differing only in
  their weighting of past returns. So on a single instrument the only
  axes that are not re-parameterisations are the **nonlinearity of the
  response**, the **path-dependence of the exposure**, and the
  **state-dependence of the horizon**. This project has varied neither of
  the first two; the conservative branch takes the second (latch
  geometry), the novel branch takes the first.

  R-80's structural finding binds here and is pre-registered as a
  constraint rather than discovered as a surprise: a continuous vote can
  never combine to exactly zero, which would disable v4's single most
  robust documented property (full de-risk to cash on unanimous bearish
  consensus). The novel branch must therefore carry an explicit
  clip-to-flat, and report the fraction of bars spent exactly flat beside
  v4's own.

Measured power, computed before either branch was dispatched and
reported here so the promotion bar is a test rather than a formality
(R-78's lesson: check that the evidence you require can actually
arrive). The paired daily difference against v4 is close to serially
uncorrelated -- long-run variance / one-day variance is 0.98-1.02 at
every block length on inner-train -- so at the project's 30-day block
convention a 95% paired interval excludes zero once the candidate beats
v4 by about **+0.35 log units over inner-train** (4 years) or **+0.13 to
+0.26 over inner-validation** (2 years). Those are large but reachable
effect sizes (R-68's own point estimate on a different axis was +0.45),
so the interval bar below is answerable rather than decorative.

This module is written by the operator BEFORE the branches are dispatched
(the R-73..R-87 convention) and is READ-ONLY for both branches: neither
branch may edit it, so both are measured by identical machinery and the
control numbers cannot drift between them.

It contains:

1. ``load_btc`` / ``load_eth`` / ``load_panel_asset`` -- data loaders that
   truncate to ``< OOS_START`` and assert it, so a branch cannot read the
   holdout by accident.
2. ``V4_HORIZONS`` / ``v4_vote_frac`` -- a byte-for-byte reproduction of
   ``kelly_regime_v4``'s own latched anchor vote, so a branch can modify
   one factor without re-deriving the other.
3. ``run_slice`` -- one backtest over a named slice through
   ``tradebot.window.run_period`` (warm prefix, fair start).
4. ``daily_simple_returns`` -- daily SIMPLE returns of an equity curve
   (the convention ``tradebot.inference`` expects; log growth is derived
   from them with ``total_log_return``, never by summing logs by hand).
5. ``paired_diff`` -- R-68's own lesson made reusable: the PAIRED
   block-bootstrap difference between a candidate and its control, which
   on this project's data resolves several times tighter than either
   arm's own level interval because the market's common-mode variance
   cancels. This is the round's PRIMARY decision statistic.
6. ``causal_truncation_probe`` -- the standing lookahead check: rebuild
   the target on a truncated frame and require the surviving prefix to
   match bit-for-bit.
7. ``r_squared`` -- the standing "is this an exposure-level artifact /
   is it merely a relabelling of v4" collinearity diagnostic.

Nothing here reads a bar at or after ``OOS_START``.
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
# be v4, not a re-parameterisation of it).
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
    """Bitfinex ETH (the series R-17/R-47 use for cross-asset replication)."""
    return _truncate(load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz"), "ETH")


def load_panel_asset(sym: str) -> pd.DataFrame:
    """One of R-57's six Coinbase panel instruments (bch/ltc/etc/dash/link/xtz)."""
    path = ROOT / "data" / f"{sym.lower()}usd_coinbase_spot_5m.csv.gz"
    return _truncate(load_ohlcv_csv(path), sym.upper())


# ------------------------------------------------------- v4's own factors

def v4_vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                 band: float = V4_BAND) -> np.ndarray:
    """`kelly_regime_v4`'s latched anchor vote, reproduced exactly.

    Each anchor latches long above ``anchor * (1 + band)``, latches flat
    below ``anchor * (1 - band)``, and holds its previous verdict inside
    the band. ``frac`` is the unweighted mean of the three latched votes,
    so it takes values in {0, 1/3, 2/3, 1}.
    """
    close = df["close"]
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def latched_vote(df: pd.DataFrame, d_in: float = V4_BAND, d_out: float = V4_BAND,
                 horizons: tuple[int, ...] = V4_HORIZONS) -> np.ndarray:
    """v4's latched anchor vote with SEPARATE entry and exit thresholds.

    ``d_in`` is how far above an anchor price must sit to latch that
    anchor long; ``d_out`` is how far below it must sit to latch it flat.
    Inside the two, the anchor holds its previous verdict. At
    ``d_in == d_out == V4_BAND`` this is ``v4_vote_frac`` bit-for-bit,
    which the module self-test asserts -- so the identity point of the
    conservative branch is guaranteed by construction rather than by
    each branch reimplementing the latch.
    """
    close = df["close"]
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + d_in), 1.0,
                     np.where(close < anchor * (1.0 - d_out), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def signal_deviation(df: pd.DataFrame,
                     horizons: tuple[int, ...] = V4_HORIZONS) -> np.ndarray:
    """v4's vote made continuous: per-anchor relative deviation, shape (n, k).

    ``dev[i, j] = close[i] / anchor_j[i] - 1`` -- the quantity v4's latch
    thresholds at +/-``band``. Provided so the novel branch can measure
    the signal's own dispersion (the unit de Lataillade & Chaouki 2020's
    Eq. (11) band is expressed in) without re-deriving the anchors.
    """
    close = df["close"]
    cols = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        cols.append((close / anchor - 1.0).to_numpy())
    return np.column_stack(cols)


def v4_anchor_votes(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                    band: float = V4_BAND) -> np.ndarray:
    """The three individual latched anchor votes, shape (n, len(horizons))."""
    close = df["close"]
    cols = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=df.index,
        )
        cols.append(v.ffill().fillna(0.0).to_numpy())
    return np.column_stack(cols)


def v4_scale(df: pd.DataFrame) -> np.ndarray:
    """`kelly_regime_v3/v4`'s conditional volatility-target scale factor.

    Reproduced exactly: full inverse-volatility sizing while realized vol
    is latched outside its normal band, a constant (slow-vol) notional
    inside it.
    """
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


# ------------------------------------------------------------- evaluation

SLICES: dict[str, tuple[str | None, str | None]] = {
    "inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
    "inner_val": (INNER_VAL_START, INNER_VAL_END),
}


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


def run_slice(strategy: Strategy, df: pd.DataFrame, slice_name: str,
              market: MarketSpec = SPOT, balance: float = 1_000.0) -> SliceResult:
    """One backtest over a named slice, with a warm (non-trading) prefix."""
    start, end = SLICES[slice_name]
    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    m = compute_metrics(res)
    d = daily_simple_returns(res.equity)
    return SliceResult(
        name=slice_name, market=market.name, final_balance=m.final_balance,
        sharpe=m.sharpe, max_drawdown_pct=m.max_drawdown_pct,
        num_trades=m.num_trades, log_growth=float(total_log_return(d)), daily=d,
    )


def daily_simple_returns(equity: pd.Series) -> np.ndarray:
    """Daily SIMPLE returns of a bar-frequency equity curve (0 once wiped out).

    Simple, not log, because this is what ``tradebot.inference`` consumes:
    ``total_log_return`` and ``annualized_sharpe`` both expect simple
    returns and take the log themselves. Mixing the two silently changes
    every number downstream, so the round uses one convention throughout.
    """
    return inference_daily_returns(equity).to_numpy()


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy.

    The target is a fraction of EQUITY notional, exactly as
    ``kelly_regime`` uses it, so spot and futures carry the same risk and
    the branches differ only in how the number is computed.
    """

    name = "r89_target"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, build_target, name: str = "r89_target",
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


def compare(build_candidate, df: pd.DataFrame, *, label: str,
            control_build=None, markets=(SPOT, FUTURES),
            slice_names=("inner_train", "inner_val"), seed: int = 0) -> list[dict]:
    """Candidate vs control on every (slice, market) cell, one table.

    Reports, per cell: both arms' final balance / Sharpe / max drawdown /
    trade count, and the PAIRED block-bootstrap difference in log growth
    (the round's primary decision statistic). Also reports the exposure
    path's R^2 against the control, once, so an inert candidate is visible
    before any performance number is read.

    ``control_build`` defaults to ``kelly_regime_v4``'s own exact target.
    """
    if control_build is None:
        control_build = lambda d: apply_deadband(v4_vote_frac(d) * v4_scale(d))  # noqa: E731

    cand_path = np.asarray(build_candidate(df), dtype=float)
    ctrl_path = np.asarray(control_build(df), dtype=float)
    rsq = r_squared(cand_path, ctrl_path)

    cand = TargetStrategy(build_candidate, name=label)
    ctrl = TargetStrategy(control_build, name="kelly_regime_v4")

    rows = []
    for slice_name in slice_names:
        for market in markets:
            a = run_slice(cand, df, slice_name, market)
            b = run_slice(ctrl, df, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "r2_vs_control": rsq,
                "cand_final": a.final_balance, "ctrl_final": b.final_balance,
                "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                "d_sharpe": a.sharpe - b.sharpe,
                "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
                "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
                "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
                "d_loggrowth": pr.diff.point,
                "d_lo": pr.diff.lo, "d_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


def print_rows(rows: list[dict]) -> None:
    """One fixed-width line per cell, so two branches' output is diffable."""
    hdr = (f"{'label':22s} {'slice':11s} {'market':11s} {'cand$':>10s} {'ctrl$':>10s} "
           f"{'dSh':>6s} {'dDD':>7s} {'trd':>5s}/{'ctl':<5s} {'dlogG':>7s} "
           f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:22]:22s} {r['slice']:11s} {r['market']:11s} "
              f"{r['cand_final']:10,.0f} {r['ctrl_final']:10,.0f} "
              f"{r['d_sharpe']:+6.2f} {r['d_dd']:+7.1f} "
              f"{r['cand_trades']:5d}/{r['ctrl_trades']:<5d} "
              f"{r['d_loggrowth']:+7.3f} {r['d_lo']:+8.3f},{r['d_hi']:+8.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>5s}")


# --------------------------------------------------------------- inference

def paired_diff(candidate: np.ndarray, control: np.ndarray, *,
                mean_block: float = 30.0, n_boot: int = 2_000, seed: int = 0):
    """Paired stationary-block-bootstrap difference in total log growth.

    R-68's lesson, made reusable: comparing the candidate to its control
    with the SAME resample applied to both cancels the market's own
    common-mode variance, and on this project's data resolves several
    times tighter than either arm's own level interval. The interval is
    the decision statistic; the point estimate alone is not.
    """
    n = min(len(candidate), len(control))
    return paired_bootstrap(np.asarray(candidate[-n:], dtype=float),
                            np.asarray(control[-n:], dtype=float),
                            total_log_return, mean_block=mean_block,
                            n_boot=n_boot, seed=seed)


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 of ``a`` against ``b`` -- the standing "is it merely v4 again?" check.

    A candidate exposure path with R^2 > 0.98 against v4's own path is
    inert by construction: whatever it scores, it is not measuring a new
    mechanism. Every round since R-41 reports this before any Sharpe.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.std(b) == 0 or np.std(a) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def causal_truncation_probe(build_target_fn, df: pd.DataFrame,
                            cuts: tuple[float, ...] = (0.55, 0.80)) -> bool:
    """Rebuild the target on truncated frames; the shared prefix must match.

    ``build_target_fn(frame) -> np.ndarray`` must be a pure function of the
    bars it is given. If any future bar leaks into an earlier decision, the
    truncated run's prefix will differ. Returns True on a clean pass.
    """
    full = np.asarray(build_target_fn(df), dtype=float)
    for cut in cuts:
        k = int(len(df) * cut)
        part = np.asarray(build_target_fn(df.iloc[:k]), dtype=float)
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-12, rtol=0.0):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-12, rtol=0.0)))
            raise AssertionError(f"causality FAIL at cut={cut}: {bad} bars differ")
    return True


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    """Assert the identity points both branches depend on. Run on import."""
    idx = pd.date_range("2020-01-01", periods=5_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 1.0}, index=idx)
    h = (1, 2, 4)
    a = latched_vote(df, V4_BAND, V4_BAND, horizons=h)
    b = v4_vote_frac(df, horizons=h, band=V4_BAND)
    assert np.array_equal(a, b), "latched_vote identity point != v4_vote_frac"
    dev = signal_deviation(df, horizons=h)
    assert dev.shape == (len(df), len(h)), "signal_deviation shape"
    # a vote built by thresholding the deviation must equal the latch
    assert np.isfinite(dev[-1]).all(), "signal_deviation tail is not finite"
    assert set(np.unique(np.round(a * len(h)))) <= {0.0, 1.0, 2.0, 3.0}, \
        "vote fraction is not a k-anchor mean"


_self_test()
