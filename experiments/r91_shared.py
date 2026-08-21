"""Shared, read-only utilities and pre-registration for the R-91 round (08-21).

DIRECTION, in one sentence: attack `kelly_regime_v4`'s **vote construction**
with the third of Levine & Pedersen (2016)'s three non-redundant linear-filter
axes -- **state-dependence of the horizon** -- which R-89 named as untested
and R-89 filed as backlog item **B-40** (Goulding, Harvey & Mazzoleni's
momentum-turning-point states), left OPEN by R-90 (which worked sibling item
B-41 instead, now closed).

Why this and not another signal. R-62/R-87 (four independent confirmations)
established that of v4's two factors (`frac x scale`) the **vote carries the
entire signature**; R-89 varied the vote's own latch geometry (asymmetric
entry/exit) and its response-function shape (linear/cubic), both NEGATIVE.
Neither R-89 branch touched the third axis: **is the map from price to
exposure allowed to depend on which of several recurring market STATES the
vote's own anchors currently agree they are in?** That is this round's
question, and it stays on the vote side of `frac x scale` -- consistent with
R-62's factorization finding -- rather than retuning `scale` (21 rounds
already have) or wrapping an external estimator around either factor (five
regime-timing mechanisms, R-82-87, already ruled out on the six-episode
detection-lag gate that this round's construction never touches, because it
classifies every bar into a recurring STATE rather than trying to detect the
DATE of a historical transition faster than v4's own anchors do).

**Which constraint it attacks: SIZE.** Exposure is conditioned on the vote's
own internal agreement structure (an interaction between v4's fastest and
slowest anchors, both already inside the strategy) rather than on any new
external data -- explicitly NOT a twelfth INFO-axis signal, and NOT a sixth
regime-timing mechanism against the exhausted six-episode gate.

**The literature**, verified by web search before either branch was
dispatched:

- **Goulding, Harvey & Mazzoleni (2023)**, "Momentum Turning Points,"
  *Journal of Financial Economics* (also SSRN 3489539; Duke working paper
  P158). Partition an asset's return history into four observable states by
  the agreement/disagreement of a FAST and a SLOW trailing momentum signal:
  **Bull** (both positive), **Bear** (both negative), **Correction** (fast
  negative, slow still positive -- a pullback inside an uptrend) and
  **Rebound** (fast positive, slow still negative -- a bounce inside a
  downtrend). Their empirical finding: Correction and Rebound states carry
  materially higher realization noise and different (often worse
  risk-adjusted) forward returns than the two trend-agreement states, and
  blending fast/slow signal weight conditional on state improves risk-
  adjusted momentum performance versus either fixed speed alone. This
  round's fast/slow proxy is v4's OWN existing anchor extremes -- 20-day
  (fastest of the 20/40/80 ladder) and 80-day (slowest) -- so no new data
  channel is introduced; the middle 40-day anchor is untouched, still used
  by `v4_vote_frac` as shipped, only the ROUND's two new branches use the
  fast/slow pair to classify a state and modulate exposure on top of it.

**Not a duplicate of.** R-59/R-60 (anchor-span/vol-target RESCALING by a
volatility-adaptive rule -- a continuous transform of the scale factor, not
a discrete state built from inter-anchor AGREEMENT). R-40 (unweighted
ensembling across ladders -- no state conditioning at all). R-34 (SIZE input
built from `harsanyi_crowd`'s Bayesian posterior margin -- a different
signal entirely, not v4's own anchor vote). R-82/83/85/86/87 (HMM, BOCPD,
Kalman LLT, CSD, transfer entropy -- external formal regime-DETECTION
mechanisms scored against a six-episode historical detection-lag gate; this
round classifies a recurring STATE at every bar from v4's own two anchors,
never asks "did we detect date X faster", and touches none of those five
mechanisms). R-89 (latch geometry / response-function shape -- both operate
on the aggregated 3-anchor vote's OWN construction; this round leaves that
vote unchanged and multiplies a state-conditional scaler on top of it).
B-42 (deriving the anchor SPAN from a fitted generative model -- a different
follow-on from the same R-89 literature pass; this round changes what the
existing anchors' agreement is used FOR, not their span).

**Falsification test, named now, before any code.** Goulding-Harvey-
Mazzoleni's qualitative finding is that the two disagreement states
(Correction, Rebound) carry higher realization noise / worse risk-adjusted
returns than the two agreement states (Bull, Bear). If a CAUSAL,
inner-train-only measurement of state-conditional Sharpe does NOT rank
Correction and Rebound below Bull and Bear, the paper's qualitative claim
does not replicate on BTC 5-minute bars and any construction built on it
should be read as failing its own motivating mechanism, independent of
whether a swept configuration happens to score well (the same style of kill
switch R-89's novel branch used for Schmidhuber's cubic-reversion
prediction).

**Scope decision, made now and applied identically to both branches, to keep
the identity point simple and the two branches' scope comparable.** Both
branches leave v4's vote UNCHANGED in Bull and Bear states (full agreement),
and apply a scaler only when the bar's state is Correction or Rebound
(disagreement / turning-point states) -- this is a deliberate simplification
of GHM's own all-four-states blend, made so that "scaler ≡ 1 everywhere"
is a trivial, exact identity point (A1 below) and so neither branch can
claim an unearned edge in the two states forming the bulk of v4's own
tested history. This restriction is a design choice pre-registered here,
not a finding, and is disclosed as a limitation on how much of GHM's own
result either branch can be said to test.

Two branches, disjoint files, both measured by this module:

- **conservative** (`r91_conservative_state_discount.py`) -- a FIXED,
  literature-motivated discount factor applied to v4's exposure specifically
  in Correction/Rebound states, swept over a small frozen grid, no
  parameter estimated from this project's own return data.
- **novel** (`r91_novel_causal_state_scaler.py`) -- a continuously-updated,
  CAUSAL (expanding-window, one-bar-lagged) state-conditional performance
  estimate that sets the Correction/Rebound scaler dynamically, the direct
  operationalisation of GHM's own "estimate state-conditional risk/return
  and blend accordingly" mechanism rather than a hand-set constant.

**Pre-registered decision rule, identical structure for both branches
(frozen before any number is read), the R-89/R-90 convention.**

*Step A -- mechanism gate, per configuration, before any performance number
is read:*
- **A1 identity.** The scaler ≡ 1.0 configuration must reproduce
  `kelly_regime_v4`'s target path bit-for-bit.
- **A2 non-inertness.** R² of the candidate's exposure path against v4's own
  must be < 0.98 on inner-train (else the configuration is inert by
  construction and is reported, not scored).
- **A3 causality.** `causal_truncation_probe` passes at two cut depths for
  the branch's own build function.
- **A0 (this round's specific kill switch, checked BEFORE Step A on any
  strategy config).** The falsification test above: causal, inner-train-only
  state-conditional Sharpe must rank Correction and Rebound both below Bull
  and Bear. If it does not, the branch is disqualified by pre-registration
  and reported as NEGATIVE regardless of any downstream number, exactly as
  R-89's Step-0 cubic kill switch was.

*Step B -- selection, on inner-train and inner-validation only; the holdout
is not read by either branch.* The finalist is the best inner-validation
paired log-growth difference vs v4 on `futures_5x`, among Step-A survivors,
with the whole swept neighbourhood reported regardless.

*Promotion bar -- default REJECT. All must hold:*
- **B1.** The paired block-bootstrap difference vs v4 excludes zero in at
  least one of the four (slice x market) cells, and its point estimate is
  positive in all four.
- **B2.** Either ΔSharpe > +0.2 (the R-20 noise floor) on inner-validation on
  both markets, OR a max-drawdown improvement on both markets where
  `risk_matched` (exposure ratio and vol ratio vs v4 both in [0.9, 1.1]) is
  true for both -- an unmatched drawdown improvement is not evidence, per
  the standing R-28/R-32/R-33 rule.
- **B3.** The neighbourhood is a plateau, not a peak: nearest neighbours on
  each swept axis move the same direction as the finalist, reported whether
  or not they do.
- **B4 falsification, ETH replication.** The frozen finalist must show the
  same SIGN of improvement over v4 on Bitfinex ETH pre-2023 (inner-train
  only; ETH coverage ends 2019-12-31), on both markets. Failing it is
  NEGATIVE.
- **B5 cost robustness.** The improvement must not reverse sign at a 0.40%
  taker fee (Bitstamp's real entry tier).

Named counter-prediction (what would make this fail, written before any
code): if the A0 kill switch fires -- Correction/Rebound do not rank below
Bull/Bear in causal inner-train Sharpe -- GHM's own motivating claim does
not exist on this instrument and BOTH branches should be expected to fail,
independent of their internal designs, the same structural outcome R-89's
novel branch hit with Schmidhuber's cubic term.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both branches: neither may edit it, so both are
measured by identical machinery. Nothing here reads a bar at or after
OOS_START.
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

# This round's fast/slow proxy: the extremes of v4's own anchor ladder.
FAST_HORIZON_DAYS = 20
SLOW_HORIZON_DAYS = 80

# State codes
BULL, BEAR, CORRECTION, REBOUND = 0, 1, 2, 3
STATE_NAMES = {BULL: "Bull", BEAR: "Bear", CORRECTION: "Correction", REBOUND: "Rebound"}


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
    """Bitfinex ETH (the series R-17/R-47/R-89/R-90 use for cross-asset replication)."""
    return _truncate(load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz"), "ETH")


# ------------------------------------------------------- v4's own factors

def _latched_anchor_vote(close: pd.Series, days: int, band: float = V4_BAND) -> np.ndarray:
    """One anchor's own latched 0/1 vote, reproduced exactly as v4 computes each of its three."""
    anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
    v = pd.Series(
        np.where(close > anchor * (1.0 + band), 1.0,
                 np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
        index=close.index,
    )
    return v.ffill().fillna(0.0).to_numpy()


def v4_vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                 band: float = V4_BAND) -> np.ndarray:
    """`kelly_regime_v4`'s latched anchor vote, reproduced exactly."""
    close = df["close"]
    votes = [_latched_anchor_vote(close, days, band) for days in horizons]
    return sum(votes) / len(votes)


def v4_scale(df: pd.DataFrame) -> np.ndarray:
    """`kelly_regime_v3/v4`'s conditional volatility-target scale factor, reproduced exactly."""
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


def v4_raw_desired(df: pd.DataFrame) -> np.ndarray:
    """v4's desired exposure BEFORE its own 10% deadband: frac * scale."""
    return v4_vote_frac(df) * v4_scale(df)


def apply_deadband(desired: np.ndarray, deadband: float = V4_DEADBAND) -> np.ndarray:
    """v4's own 10% re-target deadband, applied to a desired-exposure path."""
    target = np.zeros(len(desired))
    pos = 0.0
    for i, d in enumerate(desired):
        if abs(d - pos) > deadband:
            pos = float(d)
        target[i] = pos
    return target


def v4_target(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v4's complete, final target path (post-deadband)."""
    return apply_deadband(v4_raw_desired(df))


# ------------------------------------------------------ GHM state labels

def state_labels(df: pd.DataFrame, fast_days: int = FAST_HORIZON_DAYS,
                  slow_days: int = SLOW_HORIZON_DAYS, band: float = V4_BAND) -> np.ndarray:
    """Goulding-Harvey-Mazzoleni's four states, from v4's own fast/slow anchor votes.

    Bull = fast & slow both bullish; Bear = both bearish; Correction = fast
    bearish while slow still bullish (a pullback inside an uptrend);
    Rebound = fast bullish while slow still bearish (a bounce inside a
    downtrend). Both component votes are v4's own latched per-anchor
    construction (band + hysteresis), so this introduces no new data and no
    new smoothing rule.
    """
    close = df["close"]
    fast = _latched_anchor_vote(close, fast_days, band)
    slow = _latched_anchor_vote(close, slow_days, band)
    state = np.full(len(df), BEAR, dtype=int)
    state[(fast == 1) & (slow == 1)] = BULL
    state[(fast == 0) & (slow == 0)] = BEAR
    state[(fast == 0) & (slow == 1)] = CORRECTION
    state[(fast == 1) & (slow == 0)] = REBOUND
    return state


def is_turning_point(state: np.ndarray) -> np.ndarray:
    """True at bars classified Correction or Rebound (state disagreement)."""
    return (state == CORRECTION) | (state == REBOUND)


def causal_state_stats(daily_like_returns: np.ndarray, state_per_bar: np.ndarray,
                       n_states: int = 4, min_obs: int = 60) -> dict:
    """Descriptive only (NOT causal-per-bar): full-sample mean/vol/Sharpe of
    bar-level log returns conditional on state, for the A0 falsification
    check. Used ONLY as an inner-train-only, one-shot descriptive measurement
    (never fed back into a strategy's own target path), so it does not need
    to be causal bar-by-bar the way a tradable signal would -- it is read
    once, before either branch's strategy code runs, exactly as R-89's
    Step-0 regression fits were.
    """
    out = {}
    r = np.asarray(daily_like_returns, dtype=float)
    s = np.asarray(state_per_bar)[: len(r)]
    for k in range(n_states):
        rk = r[s == k]
        rk = rk[np.isfinite(rk)]
        n = len(rk)
        mean = float(np.mean(rk)) if n else float("nan")
        vol = float(np.std(rk)) if n > 1 else float("nan")
        sharpe = float(mean / vol * np.sqrt(365.25)) if vol and vol > 0 else float("nan")
        out[k] = {"n": n, "mean": mean, "vol": vol, "sharpe": sharpe, "enough": n >= min_obs}
    return out


class CausalStateScaler:
    """Expanding-window, ONE-BAR-LAGGED causal per-state performance tracker.

    Call :meth:`update` once per bar, in order, with bar i's own realized
    bar-return (log return of close[i] over close[i-1]) and bar i's own
    state label; it returns the scaler to use for bar i's OWN decision,
    computed from bars < i only (the running stats are updated with bar i's
    return AFTER the scaler for bar i has been read out) -- so no bar ever
    uses its own realization to size itself. This is the novel branch's
    building block; the conservative branch does not use it.
    """

    def __init__(self, n_states: int = 4, min_obs: int = 250, k: float = 2.0):
        self.n = [0] * n_states
        self.sum = [0.0] * n_states
        self.sumsq = [0.0] * n_states
        self.min_obs = min_obs
        self.k = k

    def scaler_for(self, state: int) -> float:
        """Squash the running causal Sharpe-like stat into (0, 1] via a
        logistic centered at 0 (positive causal edge -> scaler near 1,
        negative or insufficient history -> scaler shrinks toward 0)."""
        n = self.n[state]
        if n < self.min_obs:
            return 1.0  # burn-in: no penalty until there is enough history to judge
        mean = self.sum[state] / n
        var = max(self.sumsq[state] / n - mean * mean, 1e-12)
        sharpe_like = mean / (var ** 0.5) * np.sqrt(365.25 * BARS_PER_DAY)
        return float(1.0 / (1.0 + np.exp(-self.k * sharpe_like)) * 2.0)  # in (0, 2), clipped by caller

    def update(self, state: int, bar_return: float) -> None:
        if not np.isfinite(bar_return):
            return
        self.n[state] += 1
        self.sum[state] += bar_return
        self.sumsq[state] += bar_return * bar_return


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
    mean_abs_exposure: float
    realized_vol: float


def run_slice(strategy: Strategy, df: pd.DataFrame, slice_name: str,
              market: MarketSpec = SPOT, balance: float = 1_000.0) -> SliceResult:
    """One backtest over a named slice, with a warm (non-trading) prefix."""
    start, end = SLICES[slice_name]
    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
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


def daily_simple_returns(equity: pd.Series) -> np.ndarray:
    """Daily SIMPLE returns of a bar-frequency equity curve."""
    return inference_daily_returns(equity).to_numpy()


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy."""

    name = "r91_target"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, build_target, name: str = "r91_target",
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

    Reports the paired block-bootstrap difference in log growth (the
    round's primary decision statistic), plus the mean-exposure ratio and
    realised-volatility ratio candidate/control on every cell -- the
    risk-match diagnostic B2 requires be checked before any drawdown
    improvement is read as evidence.
    """
    if control_build is None:
        control_build = v4_target

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
                "exposure_ratio": (a.mean_abs_exposure / b.mean_abs_exposure
                                   if b.mean_abs_exposure else float("nan")),
                "vol_ratio": (a.realized_vol / b.realized_vol
                              if b.realized_vol else float("nan")),
                "risk_matched": bool(
                    0.9 <= (a.mean_abs_exposure / b.mean_abs_exposure if b.mean_abs_exposure else np.nan) <= 1.1
                    and 0.9 <= (a.realized_vol / b.realized_vol if b.realized_vol else np.nan) <= 1.1),
                "d_loggrowth": pr.diff.point,
                "d_lo": pr.diff.lo, "d_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


def print_rows(rows: list[dict]) -> None:
    """One fixed-width line per cell, so two branches' output is diffable."""
    hdr = (f"{'label':22s} {'slice':11s} {'market':11s} {'cand$':>10s} {'ctrl$':>10s} "
           f"{'dSh':>6s} {'dDD':>7s} {'expR':>5s} {'volR':>5s} {'RM':>3s} "
           f"{'dlogG':>7s} {'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:22]:22s} {r['slice']:11s} {r['market']:11s} "
              f"{r['cand_final']:10,.0f} {r['ctrl_final']:10,.0f} "
              f"{r['d_sharpe']:+6.2f} {r['d_dd']:+7.1f} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['d_loggrowth']:+7.3f} {r['d_lo']:+8.3f},{r['d_hi']:+8.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>5s}")


# --------------------------------------------------------------- inference

def paired_diff(candidate: np.ndarray, control: np.ndarray, *,
                mean_block: float = 30.0, n_boot: int = 2_000, seed: int = 0):
    """Paired stationary-block-bootstrap difference in total log growth."""
    n = min(len(candidate), len(control))
    return paired_bootstrap(np.asarray(candidate[-n:], dtype=float),
                            np.asarray(control[-n:], dtype=float),
                            total_log_return, mean_block=mean_block,
                            n_boot=n_boot, seed=seed)


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 of ``a`` against ``b`` -- the standing "is it merely v4 again?" check."""
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
    """Rebuild the target on truncated frames; the shared prefix must match."""
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


def fee_at(market: MarketSpec, fee_rate: float) -> MarketSpec:
    """Same market spec, at a different taker fee (for the B5 cost-robustness check)."""
    return MarketSpec(name=market.name, leverage=market.leverage, fee_rate=fee_rate,
                      allow_short=market.allow_short,
                      maintenance_margin_rate=market.maintenance_margin_rate,
                      min_notional=market.min_notional, pays_funding=market.pays_funding)


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    """Assert the identity points both branches depend on. Run on import."""
    idx = pd.date_range("2020-01-01", periods=6_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0005, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0005, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    raw = v4_raw_desired(df)
    assert np.allclose(v4_target(df), apply_deadband(raw)), "v4_target != apply_deadband(v4_raw_desired)"

    states = state_labels(df)
    assert set(np.unique(states)) <= {BULL, BEAR, CORRECTION, REBOUND}
    tp = is_turning_point(states)
    assert tp.dtype == bool

    fast = _latched_anchor_vote(df["close"], FAST_HORIZON_DAYS)
    slow = _latched_anchor_vote(df["close"], SLOW_HORIZON_DAYS)
    # Cross-check state_labels against the component votes directly.
    check = np.where((fast == 1) & (slow == 1), BULL,
             np.where((fast == 0) & (slow == 0), BEAR,
             np.where((fast == 0) & (slow == 1), CORRECTION, REBOUND)))
    assert np.array_equal(states, check), "state_labels disagrees with its own component votes"

    scaler = CausalStateScaler(min_obs=10)
    for i in range(500):
        s = scaler.scaler_for(int(states[i]))
        assert 0.0 <= s <= 2.0
        scaler.update(int(states[i]), float(rng.normal(0, 0.001)))

    assert causal_truncation_probe(v4_target, df)


_self_test()
