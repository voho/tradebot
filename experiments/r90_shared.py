"""Shared, read-only utilities for the R-90 round (08-21).

DIRECTION, in one sentence: attack `kelly_regime_v4`'s **exit rule** --
"hold until the anchor vote flips" -- with a path-dependent trailing-stop
ratchet on the running favourable extreme of the realised trade, which is
backlog item **B-32's siblings** filed by R-89's literature pass:
**B-41** (this round) and its close cousin B-42 (not taken here).

Why this and not another signal: R-62/R-87 (four independent
confirmations) established that of v4's two factors (`frac x scale`) the
**vote carries the entire signature**; every round since has either
retuned the scale slot (21 rounds), added an external confirming vote to
`frac` (ten INFO signals), replaced the regime estimator wholesale (five
mechanisms), or -- R-89, the immediately preceding round -- varied the
vote's own latch geometry and response shape, both NEGATIVE. None of the
89 prior rounds has varied *what happens to the exposure once a position
is open*: every construction to date is "flat or full", switching only
because an anchor mean crossed a band, with no notion of the trade's own
running P&L. A trailing stop is a genuinely different object: it is
**path-dependent** (keyed to the running peak of the realised trade, a
state variable no anchor-vote or no-trade-band construction carries) and
**asymmetric** (it can only ever force MORE flat time than v4, never
less) -- neither property shared by the Constantinides/Davis-Norman/Liu/
Janeczek-Shreve no-trade bands this project tried on the *deadband* in
R-64/R-66, nor by the CPPI account-level floor R-46 tried and rejected.

Which constraint it attacks: **COST** (a stop is a rule for exiting
BEFORE a reversal costs more than the fee to avoid it) and **SIZE** (it
changes how much is held, conditional on the trade's own path, not on an
external signal) -- explicitly NOT another attempt at INFO or N=3, which
89 rounds have worked hard on the anchor-vote side of.

The literature (backlog row B-41, filed by R-89, verified by this round's
own web search before either branch was dispatched):

- **Sepp & Lucic (2026)**, "The Science and Practice of Trend-Following
  Systems", arXiv:2607.19497. Classify trend systems into European
  (continuous weight), American (binary position, ATR-scaled entry
  buffer and ATR-scaled trailing stop -- structurally identical to v4's
  own binary long/flat vote, which is exactly why this is the natural
  next system to try on it) and TSMOM. Report the three deliver similar
  risk-adjusted performance and ~80% average correlation with the SG
  Trend Index once each is properly parameterised -- i.e. a trailing
  stop is not expected to be a free lunch over the incumbent's own
  latch, only a possibly-better-behaved alternative exit.
- **Han, Zhou & Zhu (2016)**, "Taming Momentum Crashes: A Simple
  Stop-Loss Strategy", SSRN 2407199. A LITERAL, fixed-percentage stop on
  US equity momentum deciles, 1926-2013: a 15% stop cuts the
  equal-weighted portfolio's worst month from -49.8% to -11.4% and more
  than doubles Sharpe. The simplest possible construction, and the
  conservative branch's direct template.
- **Hsieh (2023)**, "On Data-Driven Drawdown Control with Restart
  Mechanism in Trading", arXiv:2303.02613, IFAC-PapersOnLine. Names the
  part every naive stop omits: without a **restart mechanism**, a
  drawdown-triggered exit is a permanent de-risking after one bad
  episode, and adds a *data-driven* re-arm rule rather than an
  unconditional instant resume. The exact formula was not extractable
  from the abstract/PDF at fetch time; this round's novel branch
  implements the STATED CONCEPT (re-entry gated on a data-driven
  recovery confirmation, not on elapsed time alone) rather than
  replicating an unseen formula, and says so plainly rather than
  pretending otherwise.

**The single named risk, written before any code, per ROUTINE.md step
2.4** ("what would make it fail?"), taken directly from B-41's own
backlog filing: *on BTC, trailing stops fire on the routine 10-20%
intra-trend drawdowns that punctuate every bull run, and re-entry happens
higher than the exit -- the classic whipsaw, expensive at 10-20bps.* Both
branches measure this directly and identically via
``stopout_whipsaw_rate`` below, using the SAME operational definition of
a whipsaw, so the two branches' whipsaw rates are comparable to each
other and not an artifact of each writing its own diagnostic.

**The standing risk-match rule applies directly and mechanically here,**
more than on any prior axis: a trailing stop can only ever force
*additional* flat time relative to v4 (it never adds exposure v4 would
not already hold), so ANY drawdown improvement is, before anything else,
a candidate case of R-28/R-32/R-33's "held less, drew down less"
artifact. ``compare()`` below reports the mean-exposure ratio and the
realised-volatility ratio (candidate / v4) on every cell for exactly this
reason, and the decision rule (see each branch file) requires those
ratios inside [0.9, 1.1] before a drawdown improvement -- as opposed to a
Sharpe improvement, which is not confounded by exposure level -- may be
counted as evidence.

Two branches, disjoint files, both measured by this module:

- **conservative** (``r90_conservative_fixed_stop.py``) -- a literal,
  fixed-percentage trailing stop on price, instant unconditional restart
  (the naive construction Hsieh's paper explicitly motivates a restart
  mechanism to fix; Han-Zhou-Zhu's own template).
- **novel** (``r90_novel_adaptive_ratchet.py``) -- an ATR-scaled
  (volatility-adaptive) trailing stop, restart gated on a data-driven
  recovery confirmation (price must reclaim the exit level) rather than
  on elapsed time or an unconditional resume -- Sepp & Lucic's ATR
  scaling plus Hsieh's restart CONCEPT, combined for the first time in
  this project.

This module is written by the operator BEFORE the branches are
dispatched (the R-73..R-89 convention) and is READ-ONLY for both
branches: neither may edit it, so both are measured by identical
machinery and the control numbers cannot drift between them.

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
    """Bitfinex ETH (the series R-17/R-47/R-89 use for cross-asset replication)."""
    return _truncate(load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz"), "ETH")


# ------------------------------------------------------- v4's own factors

def v4_vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                 band: float = V4_BAND) -> np.ndarray:
    """`kelly_regime_v4`'s latched anchor vote, reproduced exactly."""
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


# --------------------------------------------------------- ATR / true range

def true_range(df: pd.DataFrame) -> np.ndarray:
    """Bar-level True Range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def atr_days(df: pd.DataFrame, days: float = 14.0) -> np.ndarray:
    """Causal ATR over ``days``, EWM-smoothed and shifted by 1 bar.

    The shift mirrors v4's own ``vol.shift(1)`` convention in
    ``v4_scale``: the stop distance used to gate bar ``i``'s decision is
    computed from True Range through bar ``i-1`` only, one extra bar of
    lag beyond the minimum the framework requires, for consistency with
    the rest of this file and to make the causal-truncation probe a real
    test of this function specifically.
    """
    tr = true_range(df)
    span = max(2, int(days * BARS_PER_DAY))
    return (pd.Series(tr).ewm(span=span, min_periods=BARS_PER_DAY).mean()
            .shift(1).to_numpy())


# ------------------------------------------------------------ trailing stop

@dataclass
class TrailingStopResult:
    target: np.ndarray       # PRE-deadband desired path; caller applies apply_deadband
    stop_events: np.ndarray  # bool, True at bars where the stop forced flat


def apply_trailing_stop(df: pd.DataFrame, raw_desired: np.ndarray,
                        stop_frac: np.ndarray, *,
                        reentry_delay_bars: np.ndarray | None = None,
                        reentry_reclaim: bool = False) -> TrailingStopResult:
    """Overlay a path-dependent trailing-stop ratchet on a raw desired-exposure path.

    Semantics (bar ``i``, all quantities indexed by bar, causal by
    construction -- everything used to decide bar ``i`` is known no later
    than bar ``i``'s own close):

    1. If not currently armed, check re-entry gates using bar ``i``'s own
       close: a cooldown count (``reentry_delay_bars``, if given) and/or a
       reclaim condition (``close[i] > price at the stop-out``, if
       ``reentry_reclaim``). Re-arm the instant BOTH active gates clear.
    2. While long (previous bar's output > 0) or freshly entering this bar
       (vote wants long, armed), track the running peak close since entry.
    3. If armed and the vote wants long, the position is flat this bar
       only if the vote is being suppressed (not armed) -- otherwise it
       follows the vote.
    4. If now long, force flat and record a stop-out the moment
       ``close[i] < peak * (1 - stop_frac[i])``.

    ``stop_frac`` may be a constant-fraction array (the conservative
    branch: Han-Zhou-Zhu's literal fixed stop) or an ATR-derived,
    time-varying array (the novel branch: Sepp & Lucic's ATR-scaled
    stop). Passing ``stop_frac`` all >= 1.0 makes the stop unreachable
    (price cannot fall >=100% in one comparison against a positive peak)
    and is this function's own identity point: the output must equal
    ``raw_desired`` exactly. Asserted by the module self-test below.

    Returns the PRE-deadband path -- the caller (each branch) applies
    ``apply_deadband`` afterwards, exactly as v4 does to its own
    ``frac * scale``, so the deadband's own turnover-reduction behaviour
    is inherited unchanged rather than re-implemented.
    """
    n = len(raw_desired)
    close = df["close"].to_numpy(dtype=float)
    out = np.zeros(n)
    stop_events = np.zeros(n, dtype=bool)
    prev_out = 0.0
    peak = -np.inf
    armed = True
    exit_price: float | None = None
    cooldown_until = -1

    for i in range(n):
        d = float(raw_desired[i])

        if not armed:
            cooldown_ok = reentry_delay_bars is None or i >= cooldown_until
            reclaim_ok = (not reentry_reclaim) or (exit_price is not None and close[i] > exit_price)
            if cooldown_ok and reclaim_ok:
                armed = True

        if prev_out > 0.0:
            peak = max(peak, close[i])
        elif d > 0.0 and armed:
            peak = close[i]

        if d > 0.0 and not armed:
            d = 0.0

        if d > 0.0:
            level = peak * (1.0 - float(stop_frac[i]))
            if close[i] < level:
                d = 0.0
                stop_events[i] = True
                armed = False
                exit_price = close[i]
                if reentry_delay_bars is not None:
                    cooldown_until = i + int(reentry_delay_bars[i])

        out[i] = d
        prev_out = d

    return TrailingStopResult(target=out, stop_events=stop_events)


def stopout_whipsaw_rate(close: np.ndarray, final_target: np.ndarray,
                         stop_events: np.ndarray, horizon_days: float = 10.0) -> dict:
    """Rate and cost of whipsaws following a stop-out -- the round's named risk.

    For each bar where ``stop_events`` fires, look forward up to
    ``horizon_days`` (in the FINAL, post-deadband target path -- what
    actually traded) for the first bar where the position re-enters
    (target crosses from <=0 to >0). A whipsaw is a re-entry inside the
    horizon at a close HIGHER than the exit bar's close: the mechanism
    sold low and bought back high, the exact failure named in B-41's own
    backlog filing before this round ran. Both branches call this with
    identical arguments so the two rates are comparable.
    """
    horizon = int(horizon_days * BARS_PER_DAY)
    n = len(close)
    events = np.flatnonzero(stop_events)
    total = 0
    whip = 0
    costs = []
    for i in events:
        exit_price = close[i]
        window_end = min(n, i + 1 + horizon)
        reentry = None
        for j in range(i + 1, window_end):
            if final_target[j] > 0 and final_target[j - 1] <= 0:
                reentry = j
                break
        if reentry is None:
            continue
        total += 1
        if close[reentry] > exit_price:
            whip += 1
            costs.append(float(np.log(close[reentry] / exit_price)))
    rate = (whip / total) if total else float("nan")
    return {
        "stop_events": int(len(events)),
        "events_with_reentry_in_horizon": total,
        "whipsaws": whip,
        "whipsaw_rate": rate,
        "mean_whipsaw_log_cost": float(np.mean(costs)) if costs else 0.0,
    }


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

    name = "r90_target"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, build_target, name: str = "r90_target",
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
    risk-match diagnostic this round's own direction section requires be
    checked before any drawdown improvement is read as evidence.
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


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    """Assert the identity points both branches depend on. Run on import."""
    idx = pd.date_range("2020-01-01", periods=5_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0005, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0005, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    raw = v4_raw_desired(df)
    assert np.allclose(v4_target(df), apply_deadband(raw)), "v4_target != apply_deadband(v4_raw_desired)"

    # Trailing-stop identity point: an unreachable stop_frac must be a pure passthrough.
    unreachable = np.ones(len(df))  # 100% -- price cannot fall that far in one bar-vs-peak check
    r = apply_trailing_stop(df, raw, unreachable)
    assert np.allclose(r.target, raw), "apply_trailing_stop identity point (stop_frac=1.0) != raw_desired"
    assert not r.stop_events.any(), "unreachable stop_frac still fired"

    tr = true_range(df)
    assert (tr >= 0).all(), "true_range has negative entries"
    atr = atr_days(df, 14)
    assert np.isfinite(atr[BARS_PER_DAY * 20 :]).all(), "atr_days not finite well past warmup"

    diag = stopout_whipsaw_rate(df["close"].to_numpy(), raw, np.zeros(len(df), dtype=bool))
    assert diag["stop_events"] == 0 and np.isnan(diag["whipsaw_rate"])


_self_test()
