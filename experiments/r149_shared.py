"""Shared, read-only utilities and pre-registration for the R-149 round (08-26).

DIRECTION, in one sentence: `universal_kelly` (L-11, 08-12) has never had its
own internal exposure-mixture mechanism touched since registration -- its
directional signal is Cover's (1991) universal portfolio over a fixed grid of
41 constant exposures, blended by exponentially-discounted log-wealth and
scaled by a FIXED `kappa=0.5` half-Kelly constant, with no volatility-based
risk management and no tracking/adaptivity in its wealth update -- and this
round adds one improvement to each of those two loci, using two structurally
different mechanisms, while leaving the grid, the discount `gamma`, and the
underlying per-exposure log-wealth recursion completely untouched.

**Which constraint this attacks: SIZE**, the one constraint this project's
own standing diagnosis credits as "what actually worked" -- but for the first
time applied to `universal_kelly` rather than `kelly_regime_v4`,
`champions_council`, `hedge_experts` or `replicator_book`.

**Why this object, not a 20th `kelly_regime_v4` variant or a 4th allocation
mechanism on `champions_council`.** `docs/LEDGER.md`'s standing diagnosis and
R-148's own re-ranking confirm the ranked backlog holds only B-06 (forward
paper trading, already running unattended) and B-44 (a LOW-priority
`HybridBroker` methodology note, not a strategy-improvement direction), and
that every "profitable, registered, multi-signal object this project has now
given a dedicated sizing round" -- `kelly_regime_v4` internally (28+ SIZE-axis
attempts), `champions_council`'s cross-strategy allocation (R-125/R-126),
`hedge_experts`'s expert composition (R-128-R-136), and `replicator_book`'s own
blend (R-148) -- has failed to improve on its own incumbent, closing all four.
R-126's own "next step" is explicit and is taken at face value here: *"no
further axis of variation on either `kelly_regime_v4` or `champions_council`
clears Step-1's non-duplicate filter without real strain; a future session
should look outside both of this project's now-exhausted single-strategy and
portfolio-of-strategies spaces."* `universal_kelly` is neither: it is a FIFTH
registered, profitable object (L-11, $1,276 spot / $1,227 futures, README
row 11) built on a THIRD distinct online-learning primitive (Cover's
universal-portfolio wealth-weighted mixture over a continuous exposure grid,
as opposed to `kelly_regime_v4`'s anchor-vote-times-vol-target or
`champions_council`/`hedge_experts`/`game_council`'s discrete-expert
Hedge/multiplicative-weights blend), and grep-confirmed against
`docs/LEDGER.md`: its only mentions anywhere in the 148-round research log are
its own L-11 registration row, one incidental listing as a `champions_council`
member (R-126, R-148's own text), a footnote about its own min-notional
fill-rate floor (line ~7290), and two mentions as a boundary case inside the
10-of-96 pairwise-bootstrap inference table -- no round has ever varied its
grid, its wealth-update recursion, its `kappa` scalar, or its hysteresis rule.
This is the same "pick a genuinely different OBJECT whose own axis was never
varied" move R-107 (multi-asset), R-125 (`kelly_regime_v4`'s risk measure),
R-126 (`champions_council`'s allocation) and R-148 (`replicator_book`'s
sizing) made when the previous object's axis was exhausted, applied to the
one profitable, registered, never-improved object left standing.

**Not a duplicate of:**
- R-148 (`replicator_book`'s fixed `scale=0.75` replaced by v4's own
  conditional-volatility-target machinery, conservative branch; per-species
  fractional-Kelly sizing, novel branch): this round's CONSERVATIVE branch
  reuses the exact same conditional-volatility-target machinery
  (Bongaerts-Kang-van Dijk 2020), reproduced byte-for-byte from
  `kelly_regime_v3`/`v4` -- but bolted onto a *different* object
  (`universal_kelly`'s continuous-grid wealth mixture, not
  `replicator_book`'s five-species replicator blend). Reusing an
  already-validated primitive on a second, previously-untouched object is the
  established conservative-branch role in this project (R-148 did the same
  move for the first time on `replicator_book`; this round is the second
  application, disclosed as such, not claimed as a fresh mechanism).
- R-125 (the risk MEASURE inside `kelly_regime_v4`'s own `scale` -- standard
  deviation replaced by CVaR): a substitution INSIDE one already-existing
  volatility-target formula. This round's conservative branch does not
  change the risk measure at all; it REPLACES a fixed scalar constant
  (`kappa=0.5`) with the volatility-target construction wholesale, on an
  object that had no risk-based scale of any kind before this round.
- R-146/R-147 (`kelly_regime_v4` vote's own anchor STATISTIC and COMBINATION
  WEIGHTS): both operate on a discrete 3-anchor vote with a band and
  hysteresis, not a continuous 41-point exposure grid with a wealth-weighted
  posterior mean. `universal_kelly` has no "vote" to reweight in that sense --
  its `b_hat` is already the posterior mean of a continuous distribution.
- Fixed-share mixing (Herbster & Warmuth 1998) is NOT new to this project in
  the abstract -- `champions_council`, `game_council` and `hedge_experts` all
  already carry a `fixed_share=1e-4` parameter (grep-confirmed,
  `src/tradebot/strategies/{champions_council,game_council,hedge_experts}.py`)
  -- but in every one of those three it re-injects mass into a discrete
  probability simplex over a handful of NAMED experts, and none of those
  three's own fixed-share rate has ever itself been the subject of a
  dedicated round (it sits at its Freund-Schapire-textbook default
  everywhere it appears, untouched since each strategy's own registration).
  This round's NOVEL branch is the first application of fixed-share
  re-injection to a CONTINUOUS action grid (41 points spanning [-1, 1],
  Cover's own universal-portfolio setting) rather than a small set of named
  experts -- a setting Cesa-Bianchi, Gaillard, Lugosi & Stoltz (2012) treat
  explicitly and is exactly the citation this branch leans on (below), not
  the original discrete-expert Herbster-Warmuth result already embedded
  (unvaried) in three other strategies.

**Literature grounding, fetched and read/re-verified via WebSearch this
session before either branch was dispatched:**
- Cover, T. M. (1991), "Universal Portfolios," *Mathematical Finance* 1(1),
  1-29 -- already this project's own citation for `universal_kelly`'s whole
  mechanism; reused here unchanged as the CONTROL, not retuned by either
  branch.
- Bongaerts, D., Kang, X., & van Dijk, M. (2020), "Conditional Volatility
  Targeting," *Financial Analysts Journal* 76(4) -- conventional continuous
  volatility targeting can fail to improve and can deepen drawdowns;
  re-sizing only in the volatility EXTREMES improves Sharpe and cuts tails at
  low turnover. Already this project's own validated mechanism
  (`kelly_regime_v3`/`v4`, promoted; reused verbatim by R-148's conservative
  branch on `replicator_book`); reused a second time here, unretuned, as this
  round's conservative branch's entire scaling mechanism.
- Baur, D. G., & Dimpfl, T. (2018), "Asymmetric volatility in
  cryptocurrencies," *Economics Letters* 173 -- BTC's inverse leverage effect,
  the asset-class fact that makes extremes-only targeting bite here; already
  this project's own v3/v4/R-148 citation, reused for the same reason.
- Herbster, M., & Warmuth, M. K. (1998), "Tracking the Best Expert," *Machine
  Learning* 32(2), 151-178 -- fixed-share re-injection of probability mass
  lets an exponential-weights scheme track a DRIFTING best action rather than
  converging to a single static one, at a bounded regret cost proportional to
  the number of genuine switches. Already this project's own citation
  (embedded, unvaried, in `champions_council`/`game_council`/`hedge_experts`);
  re-verified via WebSearch this session (Herbster & Warmuth, *Machine
  Learning* 32(2):151-178, August 1998, the concept-drift special issue).
- Cesa-Bianchi, N., Gaillard, P., Lugosi, G., & Stoltz, G. (2012), "Mirror
  Descent Meets Fixed Share (and Feels No Regret)," *Advances in Neural
  Information Processing Systems* 25 (NeurIPS 2012); arXiv:1202.3323 --
  proves fixed-share-style weight sharing generalizes correctly under mirror
  descent / entropic-regularizer updates (the exponential-weights family
  `universal_kelly`'s own wealth-softmax already belongs to), giving shifting
  regret bounds on exactly this project's continuous-grid setting rather than
  only the small discrete-expert case. Verified real via WebSearch this
  session (NeurIPS 2012 proceedings + arXiv:1202.3323); NOT previously cited
  anywhere in this project (grep-confirmed against `docs/LEDGER.md`,
  `docs/RESEARCH.md`, and every `src/tradebot/strategies/*.py` docstring) --
  the citation that licenses this round's novel branch as a genuinely
  different construction from the three strategies' existing token
  `fixed_share=1e-4`, not merely a copy-paste of their own parameter.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
- Conservative: L-11's own registration finding is that `universal_kelly`'s
  posterior mean `b_hat` is an extremely INERT signal -- "nine trades in a
  decade is what the [universal-portfolio] guarantee costs at this horizon."
  If `b_hat` rarely departs far from zero, multiplying it by a scale that can
  reach 2.0x (v4's `max_leverage`) cannot manufacture edge that was never
  there -- it can only either (a) leave the candidate statistically
  indistinguishable from the control (a real risk of tripping the Step-0 A2
  kill switch outright, since scaling a near-zero signal by a near-constant
  multiplier looks close to a no-op), or (b) occasionally amplify a small,
  possibly wrong-signed bet into a materially larger one exactly when
  volatility is already elevated, which is a genuine way to make drawdown
  WORSE rather than better. This is a real, named risk, not a rhetorical one.
- Novel: fixed-share re-injection only helps if the TRUE best fixed exposure
  genuinely drifts within the sample (Herbster & Warmuth's own regret bound
  is stated relative to the number of genuine segment switches). If
  `universal_kelly`'s long, slow `memory_bars=8640`-bar (30-day) discount
  already means its wealth posterior is close to stationary over any window
  this project can measure, periodically re-injecting uniform mass across all
  41 grid points only adds unnecessary churn (and fee drag on every re-target
  the added noise causes) without adding real tracking value -- the same
  "nothing to track" failure mode this project's own three existing
  `fixed_share=1e-4` instances would already be suffering silently if that
  were true generally, which this round's B3 plateau sweep (varying the rate
  itself, something none of those three strategies' own rounds ever did) is
  positioned to detect directly for the first time.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it, so both are measured by
identical machinery, the r89-r148 convention. Nothing here reads a bar at or
after OOS_START (2023-01-01); `compare()` asserts this explicitly for every
slice it runs.
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

# universal_kelly's own shipped constants (do not change: the control must
# be universal_kelly, not a re-parameterisation of it). Verified against
# src/tradebot/strategies/universal_kelly.py.
UK_GRID_POINTS = 41
UK_MEMORY_BARS = 8640
UK_KAPPA = 0.5
UK_HYSTERESIS = 0.05

# kelly_regime_v3/v4's own shipped conditional-vol-target constants,
# reproduced byte-for-byte from src/tradebot/strategies/kelly_regime_v3.py.
# UNTOUCHED by the conservative branch -- reused verbatim, not retuned.
V4_TARGET_VOL = 0.55
V4_MAX_LEVERAGE = 2.0
V4_VOL_SPAN = 8 * BARS_PER_DAY
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
# universal_kelly's own construction, reproduced EXACTLY (grid, discounted
# log-wealth recursion, kappa scale, hysteresis) so it can be called with a
# candidate's own weight-mixing or scale substituted in.
# ==================================================================

def uk_grid(grid_points: int = UK_GRID_POINTS) -> np.ndarray:
    return np.linspace(-1.0, 1.0, grid_points)


def uk_posterior_mean(df: pd.DataFrame, grid_points: int = UK_GRID_POINTS,
                       memory_bars: int = UK_MEMORY_BARS,
                       fixed_share: float = 0.0) -> np.ndarray:
    """The wealth-weighted posterior mean b_hat(t) over the exposure grid.

    Reproduces UniversalKelly.prepare()'s log-wealth recursion EXACTLY when
    fixed_share=0.0 (byte-identical to the registered class). fixed_share>0
    is the novel branch's only change: after each bar's softmax, (1 -
    fixed_share) of the posterior mass is kept and fixed_share is spread
    uniformly back across all grid points -- the identical primitive already
    shipped, unvaried, in champions_council/game_council/hedge_experts
    (`p = (1-fixed_share)*p + fixed_share/num`), applied here to a continuous
    grid for the first time (Cesa-Bianchi et al. 2012) rather than a small
    set of named experts.
    """
    close = df["close"].to_numpy()
    n = len(df)
    grid = uk_grid(grid_points)
    log_wealth = np.zeros(grid_points)
    gamma = 1.0 - 1.0 / memory_bars

    b_hat = np.zeros(n)
    for i in range(1, n):
        ret = close[i] / close[i - 1] - 1.0
        ret = min(max(ret, -0.5), 0.5)
        log_wealth = gamma * log_wealth + np.log1p(np.clip(grid * ret, -0.95, None))
        w = np.exp(log_wealth - log_wealth.max())
        w /= w.sum()
        if fixed_share > 0.0:
            w = (1.0 - fixed_share) * w + fixed_share / grid_points
        b_hat[i] = float(w @ grid)
    return b_hat


def apply_uk_hysteresis(scaled: np.ndarray, hysteresis: float = UK_HYSTERESIS,
                         clip: float = 1.0) -> np.ndarray:
    """universal_kelly's own re-target rule, reproduced verbatim: move only
    if the change exceeds `hysteresis` OR the sign flips; clip to +/-`clip`
    (1.0 for the control and the novel branch; V4_MAX_LEVERAGE for the
    conservative branch, whose scale legitimately ranges up to 2.0x -- see
    that branch's own comment for why re-clamping to 1.0 would be wrong)."""
    n = len(scaled)
    target = np.zeros(n)
    pos = 0.0
    for i in range(n):
        x = min(max(float(scaled[i]), -clip), clip)
        if abs(x - pos) > hysteresis or (x > 0) != (pos > 0) or (x < 0) != (pos < 0):
            pos = x
        target[i] = pos
    return target


def universal_kelly_target(df: pd.DataFrame, grid_points: int = UK_GRID_POINTS,
                            memory_bars: int = UK_MEMORY_BARS, kappa: float = UK_KAPPA,
                            hysteresis: float = UK_HYSTERESIS) -> np.ndarray:
    """universal_kelly's complete, final target path -- the control,
    reproduced from the registered strategy's own defaults."""
    b_hat = uk_posterior_mean(df, grid_points, memory_bars, fixed_share=0.0)
    return apply_uk_hysteresis(kappa * b_hat, hysteresis, clip=1.0)


# ================================================================== (2)
# kelly_regime_v3/v4's conditional volatility-target machinery, reproduced
# byte-for-byte from experiments/r148_shared.py / kelly_regime_v3.py.
# UNTOUCHED by this round -- the conservative branch feeds it the SAME
# BTC/ETH symmetric realized volatility v4 itself uses, not retuned.
# ==================================================================

def v4_symmetric_vol(df: pd.DataFrame, span: int = V4_VOL_SPAN) -> np.ndarray:
    r = np.log(df["close"]).diff()
    return (r.ewm(span=span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def conditional_target_scale(vol: np.ndarray, anchor_span_days: int = V4_ANCHOR_SPAN_DAYS,
                              high_in: float = V4_HIGH_IN, high_out: float = V4_HIGH_OUT,
                              low_in: float = V4_LOW_IN, low_out: float = V4_LOW_OUT,
                              target_vol: float = V4_TARGET_VOL,
                              max_leverage: float = V4_MAX_LEVERAGE) -> np.ndarray:
    vol = np.asarray(vol, dtype=float)
    slow = (pd.Series(vol).ewm(span=anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(target_vol / vol, max_leverage)
        steady = np.minimum(target_vol / slow, max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(vol)
    out = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > high_in else (-1 if x < low_in else 0)
            elif state == 1 and x < high_out:
                state = 0
            elif state == -1 and x > low_out:
                state = 0
        out[i] = full[i] if state != 0 else steady[i]
    return out


def v4_scale(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v3/v4's conditional volatility-target scale factor,
    reproduced exactly. Reused verbatim by the conservative branch below."""
    return conditional_target_scale(v4_symmetric_vol(df))


# ================================================================== (3)
# CONSERVATIVE candidate: replace universal_kelly's fixed kappa=0.5 scalar
# with v4's own (unretuned) conditional-volatility-target scale. The grid,
# the log-wealth recursion and b_hat itself are byte-identical to the
# control; only the scalar that multiplies b_hat, and the clip range that
# must widen to admit it, differ.
# ==================================================================

def conservative_target(df: pd.DataFrame, grid_points: int = UK_GRID_POINTS,
                         memory_bars: int = UK_MEMORY_BARS,
                         hysteresis: float = UK_HYSTERESIS,
                         target_vol: float = V4_TARGET_VOL,
                         max_leverage: float = V4_MAX_LEVERAGE) -> np.ndarray:
    b_hat = uk_posterior_mean(df, grid_points, memory_bars, fixed_share=0.0)
    scale = conditional_target_scale(v4_symmetric_vol(df), target_vol=target_vol,
                                      max_leverage=max_leverage)
    return apply_uk_hysteresis(b_hat * scale, hysteresis, clip=max_leverage)


# ================================================================== (4)
# NOVEL candidate: fixed-share re-injection (Herbster & Warmuth 1998; the
# continuous-grid generalization justified by Cesa-Bianchi et al. 2012)
# into universal_kelly's own wealth posterior, every bar, before computing
# b_hat. kappa, memory_bars, the grid and hysteresis are UNCHANGED -- only
# the weight-mixing step inside the posterior-mean computation differs.
# ==================================================================

def novel_target(df: pd.DataFrame, grid_points: int = UK_GRID_POINTS,
                  memory_bars: int = UK_MEMORY_BARS, kappa: float = UK_KAPPA,
                  hysteresis: float = UK_HYSTERESIS, fixed_share: float = 1e-2) -> np.ndarray:
    b_hat = uk_posterior_mean(df, grid_points, memory_bars, fixed_share=fixed_share)
    return apply_uk_hysteresis(kappa * b_hat, hysteresis, clip=1.0)


# ------------------------------------------------------- causal truncation

def causal_truncation_probe_series(build_fn, df: pd.DataFrame,
                                    cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    full = np.asarray(build_fn(df), dtype=float)
    for cut in cuts:
        k = int(len(df) * cut)
        if k < BARS_PER_DAY * 2:
            continue
        part = np.asarray(build_fn(df.iloc[:k]), dtype=float)
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-8, rtol=1e-7):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-8, rtol=1e-7)))
            raise AssertionError(f"{build_fn.__name__} causality FAIL at cut={cut}: {bad} bars differ")
        perturbed = df.copy()
        tail = perturbed.iloc[k:].copy()
        for col in ("open", "high", "low", "close"):
            if col in tail.columns:
                tail[col] = tail[col] * 3.7 + 1.0
        perturbed.iloc[k:] = tail
        pert = np.asarray(build_fn(perturbed), dtype=float)
        pm = np.isfinite(a) & np.isfinite(pert[:k])
        if not np.allclose(a[pm], pert[:k][pm], atol=1e-8, rtol=1e-7):
            raise AssertionError(f"{build_fn.__name__} peeks at bar>=k, cut={cut}")
    return True


# ================================================================== (5)
# compare(): run any pure `build_target(df) -> np.ndarray` candidate over
# inner-train, inner-validation and the ETH replication slice, vs
# universal_kelly's own control, on BOTH markets. Never touches OOS_START.
# Structurally identical to r148_shared.py's compare().
# ==================================================================

SLICES: dict[str, tuple[str | None, str | None]] = {
    "inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
    "inner_val": (INNER_VAL_START, INNER_VAL_END),
}
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
    return inference_daily_returns(equity).to_numpy()


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy.

    universal_kelly's own registered convention is ``order_notional`` --
    verified against ``src/tradebot/strategies/universal_kelly.py``'s
    ``on_bar`` -- fraction of EQUITY, independent of leverage, with the
    broker's own leverage cap clamping anything the market cannot support
    (spot silently caps at 1x; futures admits up to 5x). This is correct
    for all three of control, conservative and novel here: conservative's
    target legitimately ranges up to ``V4_MAX_LEVERAGE=2.0`` (an absolute
    leverage multiple, exactly like kelly_regime_v4's own scale), which
    ``order_notional`` honours up to the market's own cap without any of
    R-148's order_target/order_notional convention mismatch (that mismatch
    arose only because replicator_book's own convention was order_target;
    universal_kelly's is already order_notional, so no dispatch flag is
    needed here).
    """

    name = "r149_control"
    warmup = UK_MEMORY_BARS + V4_ANCHOR_SPAN_DAYS * BARS_PER_DAY

    def __init__(self, build_target, name: str = "r149_control",
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


def run_slice(strategy: Strategy, df: pd.DataFrame, start: str | None, end: str | None,
              slice_name: str, market: MarketSpec = SPOT,
              balance: float = 1_000.0) -> SliceResult:
    if end is not None:
        assert pd.Timestamp(end) < pd.Timestamp(OOS_START), (
            f"run_slice({slice_name!r}): end={end} is not before OOS_START={OOS_START}")
    assert_no_holdout(df, slice_name)

    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    assert_no_holdout(res.equity.to_frame(), f"{slice_name} result")
    m = compute_metrics(res)
    d = daily_simple_returns(res.equity)
    exposure = res.df["target"].to_numpy() if "target" in res.df.columns else np.array([np.nan])
    return SliceResult(
        name=slice_name, market=market.name, final_balance=m.final_balance,
        sharpe=m.sharpe, max_drawdown_pct=m.max_drawdown_pct,
        num_trades=m.num_trades, log_growth=float(total_log_return(d)), daily=d,
        mean_abs_exposure=float(np.nanmean(np.abs(exposure))),
        realized_vol=float(np.nanstd(d) * np.sqrt(365.25)) if len(d) > 1 else float("nan"),
    )


def paired_diff(candidate: np.ndarray, control: np.ndarray, *,
                 mean_block: float = 30.0, n_boot: int = 2_000, seed: int = 0):
    n = min(len(candidate), len(control))
    return paired_bootstrap(np.asarray(candidate[-n:], dtype=float),
                             np.asarray(control[-n:], dtype=float),
                             total_log_return, mean_block=mean_block,
                             n_boot=n_boot, seed=seed)


def compare(candidate_build, *, label: str, btc: pd.DataFrame | None = None,
            eth: pd.DataFrame | None = None, control_build=None,
            markets: tuple[MarketSpec, ...] = (SPOT, FUTURES),
            include_eth: bool = True, seed: int = 0) -> list[dict]:
    if control_build is None:
        control_build = universal_kelly_target
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare(): btc")
    if include_eth and eth is None:
        eth = load_eth()
    if include_eth:
        assert_no_holdout(eth, "compare(): eth")

    cand = TargetStrategy(candidate_build, name=f"r149_{label}")
    ctrl = TargetStrategy(control_build, name="universal_kelly")

    rows = []
    jobs = [(name, start, end, btc) for name, (start, end) in SLICES.items()]
    if include_eth:
        jobs.append((ETH_SLICE_NAME, None, None, eth))

    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice(cand, df, start, end, slice_name, market)
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
    hdr = (f"{'label':26s} {'slice':16s} {'market':11s} {'cand$':>10s} {'ctrl$':>10s} "
           f"{'dSh':>6s} {'dDD':>7s} {'expR':>5s} {'volR':>5s} {'RM':>3s} "
           f"{'dlogG':>7s} {'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:26]:26s} {r['slice']:16s} {r['market']:11s} "
              f"{r['cand_final']:10,.0f} {r['ctrl_final']:10,.0f} "
              f"{r['d_sharpe']:+6.2f} {r['d_dd']:+7.1f} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['boot_d_loggrowth']:+7.3f} {r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def fee_at(market: MarketSpec, fee_rate: float) -> MarketSpec:
    return MarketSpec(name=market.name, leverage=market.leverage, fee_rate=fee_rate,
                       allow_short=market.allow_short,
                       maintenance_margin_rate=market.maintenance_margin_rate,
                       min_notional=market.min_notional, pays_funding=market.pays_funding)


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2 or np.std(b) == 0:
        return float("nan")
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ============================================================ pre-registration
#
# PROMOTION BAR (identical shape to R-89-R-148's standard bar, adapted to
# this round's control being `universal_kelly` rather than `kelly_regime_v4`
# or `replicator_book`):
#
#  A2 (Step-0 non-inertness kill switch): R^2 of the candidate's own final
#     target path against universal_kelly's unmodified `universal_kelly_target`,
#     computed on inner-train. If R^2 > 0.98, STOP -- the candidate is a
#     disguised no-op, not a genuinely different sizing mechanism, and no
#     Sharpe number is read past this point.
#  B1: bootstrap paired difference in total log-growth, inner-validation,
#     >= 1 of 2 markets: ΔSharpe > +0.2 OR the 95% bootstrap interval
#     excludes zero.
#  B2 (diagnostic, not gating): exposure_ratio / vol_ratio reported for
#     every cell per R-33's standing rule, so a return improvement is never
#     silently an exposure-level artifact read as a mechanism.
#  B3: plateau -- conservative sweeps target_vol in {0.44, 0.55, 0.66}
#     (+/-20% of V4_TARGET_VOL); novel sweeps fixed_share in
#     {1e-2, 3e-2, 1e-1} -- calibrated by the Step-0 A2 non-degeneracy check
#     alone (BEFORE any performance number was read): the token rate already
#     shipped elsewhere (1e-4, champions_council/game_council/hedge_experts)
#     and 1e-3 both produced R^2 > 0.955 against the control's own posterior
#     on inner-train (1e-3: R^2=0.99999874, a near-exact no-op) -- too close
#     to a disguised copy of the control to be a meaningful test of the
#     mechanism at all, let alone clear the A2 kill switch. 1e-2 is the
#     smallest rate on a logarithmic grid found to move R^2 measurably away
#     from 1 (0.9559); 3e-2 and 1e-1 extend the same grid one and two
#     decades further. Sign of
#     d_sharpe on the primary market must hold across the full grid; a
#     single winning cell with no support around it does not clear this bar.
#  B4: ETH same-sign falsification -- the candidate's d_sharpe sign
#     (candidate vs universal_kelly, inner-validation, BTC) must agree with
#     the ETH replication slice's own sign on at least one market.
#  B5: 0.40% taker-fee-tier re-run on spot (fee_at(SPOT, 0.004)) -- the
#     edge, if any, must not require the 0.10% fee tier to exist. Gates
#     only if B1 passed.
#
# Promote-candidate only if A2 does not trip AND B1 passes (>=1 market) AND
# B4 passes AND B5 passes (or is moot because B1 failed). Anything else is
# NEGATIVE. This is the SAME bar both branches must clear; neither may
# weaken it after seeing a number.
# ============================================================


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(149)
    innov = rng.normal(0, 0.0006, len(idx))
    jump_idx = rng.choice(len(idx), size=15, replace=False)
    innov[jump_idx] += rng.choice([-1, 1], size=15) * rng.uniform(0.01, 0.03, size=15)
    innov[30_000:35_000] *= 5.0  # a genuine volatility-regime shift
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": 1.0}, index=idx)

    # (1) control self-consistency: universal_kelly_target reproduces the
    # registered UniversalKelly class exactly.
    from tradebot.strategies.universal_kelly import UniversalKelly
    strat = UniversalKelly()
    prepared = strat.prepare(df.copy())
    assert np.allclose(prepared["target"].to_numpy(), universal_kelly_target(df),
                        atol=1e-9), \
        "universal_kelly_target() diverges from the registered UniversalKelly class"

    # (2) conservative candidate is well-formed and genuinely different.
    cons = conservative_target(df)
    assert np.all(cons >= -V4_MAX_LEVERAGE - 1e-9) and np.all(cons <= V4_MAX_LEVERAGE + 1e-9)
    assert not np.allclose(cons, universal_kelly_target(df))
    scale_probe = conditional_target_scale(v4_symmetric_vol(df))
    assert np.nanmax(scale_probe) > 1.0 + 1e-6, \
        "conditional scale never exceeds 1x on synthetic data -- vol-regime shift is not exercising it"

    # (3) novel candidate is well-formed, bounded, and genuinely different.
    nov = novel_target(df)
    assert np.all(nov >= -1.0 - 1e-9) and np.all(nov <= 1.0 + 1e-9)
    assert not np.allclose(nov, universal_kelly_target(df))
    assert not np.allclose(nov, cons)

    # (4) fixed_share=0 exactly reproduces the control's own posterior mean
    # (sanity: the novel mechanism is a strict generalization, not a
    # different recursion in disguise).
    b0 = uk_posterior_mean(df, fixed_share=0.0)
    b_ctrl_equiv = uk_posterior_mean(df, fixed_share=0.0)
    assert np.allclose(b0, b_ctrl_equiv, atol=1e-12)
    b_fs = uk_posterior_mean(df, fixed_share=1e-2)
    assert not np.allclose(b0, b_fs), "fixed_share=1e-2 produced an identical posterior (degenerate)"

    # (5) causal truncation probes -- no candidate may peek at future bars.
    assert causal_truncation_probe_series(universal_kelly_target, df)
    assert causal_truncation_probe_series(conservative_target, df)
    assert causal_truncation_probe_series(novel_target, df)
    assert causal_truncation_probe_series(v4_scale, df)

    # (6) r_squared sanity.
    assert abs(r_squared(cons, cons) - 1.0) < 1e-9
    assert r_squared(cons, rng.normal(0, 1, len(cons))) < 0.5

    # (7) order_notional dispatch sanity on futures: a target of 2.0 (the
    # conservative branch's own ceiling) should realize ~2x the notional of
    # a target of 1.0 (the control/novel ceiling), confirming order_notional
    # correctly scales past 1x on a leveraged market without needing a
    # separate order_target dispatch flag (unlike R-148's replicator_book,
    # whose native convention was order_target).
    fut = MarketSpec.futures(leverage=5.0)

    class _Probe(Strategy):
        name = "r149_probe"
        warmup = 10

        def __init__(self, level: float) -> None:
            self.level = level

        def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame

        def on_bar(self, ctx: Context) -> None:
            if ctx.i == self.warmup:
                ctx.order_notional(self.level)

    small = df.iloc[:2_000]
    res_1x = run_period(_Probe(1.0), small, None, None, market=fut, start_balance=1_000.0)
    res_2x = run_period(_Probe(2.0), small, None, None, market=fut, start_balance=1_000.0)
    qty_1x = abs(res_1x.fills[0].qty) if res_1x.fills else 0.0
    qty_2x = abs(res_2x.fills[0].qty) if res_2x.fills else 0.0
    assert qty_1x > 0 and qty_2x > 0, "probe fired no fill -- test is not exercising the path"
    ratio = qty_2x / qty_1x
    assert 1.8 < ratio < 2.2, (
        f"order_notional(2.0) vs order_notional(1.0) on 5x futures should differ ~2x, got {ratio:.2f}x")


_self_test()
