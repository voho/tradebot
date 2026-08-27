"""R-165 NOVEL branch: a DERIVED-rate EWMA on kelly_regime_v4's `scale` factor.

MECHANISM (one sentence): replace v4's raw `scale[i]` (its
conditional-volatility-target ratio) with an exponentially smoothed
`eff_scale[i] = (1-a)*eff_scale[i-1] + a*scale[i]` whose rate `a` is DERIVED
from a growth-cost/fee-cost trade-off measured on inner-train only -- never
grid-searched against backtest performance -- and feed `desired[i] =
frac[i] * eff_scale[i]` into v4's own unchanged deadband position update.

Everything else in `KellyRegimeV3.prepare` is copied byte-for-byte: the
three latched anchor votes (`frac`), the `vol`/`slow` computation, the
`full`/`steady` legs, the high/low breakout hysteresis `state` machine, and
the `if abs(desired - pos) > deadband: pos = desired` update. The ONLY
change is the one line that inserts `eff_scale` between `scale` and
`desired`.

--------------------------------------------------------------------------
THE DERIVATION (this is the point of the round; no performance number is
used anywhere in it)
--------------------------------------------------------------------------

This project has already derived a *discrete* no-trade band this way
(docs/RESEARCH.md finding 7, L-05/L-06, `kelly_regime_ev`): for a Kelly
sizer, holding exposure `f` instead of the desired `f*` gives up

    growth cost  =  (sigma^2 / 2) * (f - f*)^2      per unit time

while correcting the gap costs

    fee cost     =  fee * |delta f|                  per correction.

R-165's novel branch generalizes that from a *band width* to a *continuous
smoothing rate*, which is the object Dao et al. (2016) frame for a single
signal: a finite trading rate trades signal-tracking error against turnover
cost. Write `x_t = scale[t]`, `y_t = eff_scale[t]`, `e_t = y_t - x_t`. With
the position being `f = frac * y` and the frictionless target `f* = frac *
x`, the two terms per BAR are

    growth cost(a) = (sigma_bar^2 / 2) * E[frac^2] * Var(e; a)
    fee cost(a)    = fee * E[|frac|] * E|dy; a|
                   = fee * E[|frac|] * sqrt(2/pi) * sqrt(Var(dy; a))

where `sigma_bar^2` is the measured per-bar variance of BTC log returns on
inner-train (the "actual position's realized variance contribution"), and
`E|dy| = sqrt(2/pi) * sd(dy)` is the mean absolute move of an approximately
Gaussian increment.

Model `x` locally as a random walk with per-bar innovation variance `s^2 =
Var(x_t - x_{t-1})`, measured on inner-train. (Local RW is the right
approximation because the optimal `a` turns out to have a half-life an order
of magnitude shorter than `scale`'s own mean-reversion half-life; the AR(1)
refinement, which uses the measured persistence explicitly, is computed
below as a robustness check and moves `a*` by well under one sensitivity
step.) Then, exactly:

    e_t = (1-a) * (e_{t-1} - dx_t)
      =>  Var(e; a) = s^2 * (1-a)^2 / (a * (2-a))
    dy_t = a * (x_t - y_{t-1}),  Var(x_t - y_{t-1}) = s^2 / (a * (2-a))
      =>  Var(dy; a) = s^2 * a / (2-a)

so the total per-bar cost to minimize over `a` in (0, 1] is

    C(a) = A * s^2 * (1-a)^2 / (a*(2-a))  +  B * s * sqrt(a / (2-a))
    A = (sigma_bar^2 / 2) * E[frac^2]
    B = fee * E[|frac|] * sqrt(2/pi)

`C` is convex-shaped on (0,1] with `C -> inf` as `a -> 0` (unbounded
tracking error) and `C(1) = B*s` (v4's own instant-jump behaviour: zero
tracking error, maximum turnover). Its small-`a` stationary point has the
closed form

    a* = ( sqrt(2) * A * s / B )^(2/3)

which is the number reported as PRIMARY (the exact numeric argmin of `C` on
a fine grid is also computed and agrees to three significant figures).

PRE-REGISTERED FALSIFICATION TEST (r165_shared.py, "Falsification test,
stated precisely"): if the derived rate is indistinguishable from `a = 1`
once rounded to the harness's bar resolution, the mechanism has nothing to
test and that IS the result -- no searching for a derivation that produces a
more interesting number. This file reports the test's outcome explicitly.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------

    python experiments/r165_novel_ewma.py derive     # derivation only
    python experiments/r165_novel_ewma.py inner      # derivation + sweep
    python experiments/r165_novel_ewma.py holdout    # D0-D6, reads holdout
    python experiments/r165_novel_ewma.py all

`inner` never touches a bar at or after OOS_START. `holdout` does, once, and
only with the already-frozen PRIMARY `a*` (the sensitivity sweep is
diagnostic and is never used to pick a different cell).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    annualized_sharpe,
    daily_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r165_shared import (  # noqa: E402
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    V4_ANCHOR_HALFLIVES_DAYS,
    causal_autocorr_halflife_days,
    order_of_magnitude_gap,
    realized_vol_series,
)

INNER_TRAIN_START = "2017-01-01"
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
FEE_TIER = 0.0040          # Bitstamp entry taker, scripts/fee_study.py's tier
SENSITIVITY_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
RISK_MATCH_TOL = 0.10      # D0: 10% on time-in-market and realized vol
SHARPE_NOISE_FLOOR = 0.20  # R-20


# ============================================================== strategy
class KellyRegimeV4EwmaScale(KellyRegimeV4):
    """kelly_regime_v4 with an EXPONENTIALLY SMOOTHED `scale` factor.

    Not registered (no `@register`): this is an unregistered experiment and
    must stay out of the CI comparison table per docs/ROUTINE.md.

    `prepare()` is `KellyRegimeV3.prepare` copied byte-for-byte -- same
    anchor votes, same `vol`/`slow`, same `full`/`steady`, same hysteresis
    `state` machine, same deadband update -- with exactly one insertion:
    the raw `scale` is passed through `eff = (1-a)*eff + a*scale` before it
    multiplies `frac`. `ewma_a = 1.0` reproduces v4 bit-for-bit.

    The EWMA is seeded at the first bar whose raw `scale` is strictly
    positive (i.e. the first bar v4 itself could size on), so the smoothing
    does not spend its first half-lives climbing out of the zero the warmup
    leaves behind. That seeding is causal (it uses only bar `i`'s own value).
    """

    name = "kelly_regime_v4_ewma_scale"

    def __init__(self, ewma_a: float = 1.0, **kwargs) -> None:
        # NOTE: this __init__ exists only to carry the rate; it adds no
        # behaviour. Everything else is inherited from KellyRegimeV4.
        super().__init__(**kwargs)
        if not (0.0 < float(ewma_a) <= 1.0):
            raise ValueError(f"ewma_a must be in (0, 1], got {ewma_a!r}")
        self.ewma_a = float(ewma_a)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        a = self.ewma_a
        eff = 0.0
        seeded = False
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            # ---- the one inserted line (plus its causal seeding) ----
            if not seeded:
                if scale > 0.0:
                    eff, seeded = scale, True
                else:
                    eff = scale
            else:
                eff = (1.0 - a) * eff + a * scale
            # ---------------------------------------------------------
            desired = frac[i] * eff
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


# ================================================================= data
def _load_btc() -> pd.DataFrame:
    df, _label = load_dataset(ROOT / "data", "spot")
    return df


def _load_eth() -> pd.DataFrame:
    return load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")


def _pre_holdout(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.index < pd.Timestamp(OOS_START, tz="UTC")]


# =========================================================== derivation
@dataclass
class Derivation:
    a_star: float
    a_star_closed_form: float
    a_star_numeric: float
    a_star_ar1: float
    halflife_bars: float
    halflife_days: float
    s: float
    sigma_bar2: float
    sigma_ann: float
    e_frac: float
    e_frac2: float
    fee: float
    A: float
    B: float
    cost_at_astar: float
    cost_at_one: float
    n_bars: int
    phi: float
    scale_halflife_days: float


def _v4_scale_and_frac(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """v4's raw `scale` and `frac` paths, from v4's own code path.

    Built by running `KellyRegimeV4EwmaScale.prepare`'s own components: to
    avoid any risk of drift between the derivation's inputs and the
    strategy's, the raw `scale` is recomputed here with the identical
    formula (a=1 makes `eff == scale`, so this is exactly what v4 sees).
    """
    s = KellyRegimeV4EwmaScale(ewma_a=1.0)
    close = df["close"]
    r = np.log(close).diff()
    votes = []
    for days in s.horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + s.band), 1.0,
                     np.where(close < anchor * (1.0 - s.band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    frac = (sum(votes) / len(votes)).to_numpy()

    vol = (r.ewm(span=s.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=s.anchor_span_days * BARS_PER_DAY,
                               min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(s.target_vol / vol, s.max_leverage)
        steady = np.minimum(s.target_vol / slow, s.max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    out = np.zeros(len(df))
    state = 0
    for i in range(len(df)):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > s.high_in else (-1 if x < s.low_in else 0)
            elif state == 1 and x < s.high_out:
                state = 0
            elif state == -1 and x > s.low_out:
                state = 0
        out[i] = full[i] if state != 0 else steady[i]
    return out, frac


def _cost(a: float, A: float, B: float, s: float) -> float:
    """C(a) = A*Var(e;a) + B*sd(dy;a), the per-bar total cost (see docstring)."""
    var_e = s * s * (1.0 - a) ** 2 / (a * (2.0 - a))
    sd_dy = s * math.sqrt(a / (2.0 - a))
    return A * var_e + B * sd_dy


def _cost_ar1(a: float, A: float, B: float, sigma_x2: float, phi: float) -> float:
    """AR(1) refinement of C(a): same objective, `x` an AR(1) with the
    measured per-bar persistence `phi` and stationary variance `sigma_x2`."""
    b = 1.0 - a
    denom = 1.0 - b * phi
    var_y = a * a * sigma_x2 * (1.0 + b * phi) / ((1.0 - b * b) * denom)
    cov_yx = a * sigma_x2 / denom
    var_e = max(var_y + sigma_x2 - 2.0 * cov_yx, 0.0)
    cov_y1x = a * sigma_x2 * phi / denom          # Cov(x_t, y_{t-1})
    var_gap = max(sigma_x2 + var_y - 2.0 * cov_y1x, 0.0)
    var_dy = a * a * var_gap
    return A * var_e + B * math.sqrt(var_dy)


def derive(df: pd.DataFrame | None = None, fee: float = SPOT.fee_rate,
           verbose: bool = True) -> Derivation:
    """Derive `a*` from inner-train statistics alone. No backtest is run."""
    if df is None:
        df = _load_btc()
    train = df.loc[:INNER_TRAIN_END]
    scale, frac = _v4_scale_and_frac(train)

    warm = np.flatnonzero(scale > 0.0)
    if len(warm) == 0:
        raise RuntimeError("no warm bars in inner-train")
    i0 = int(warm[0])
    sc = scale[i0:]
    fr = frac[i0:]
    r = np.log(train["close"]).diff().to_numpy()[i0:]
    r = r[np.isfinite(r)]

    s = float(np.std(np.diff(sc)))              # per-bar innovation sd of `scale`
    sigma_bar2 = float(np.var(r))               # per-bar log-return variance
    sigma_ann = float(np.sqrt(sigma_bar2 * BARS_PER_YEAR))
    e_frac = float(np.mean(np.abs(fr)))
    e_frac2 = float(np.mean(fr ** 2))

    A = 0.5 * sigma_bar2 * e_frac2
    B = fee * e_frac * math.sqrt(2.0 / math.pi)

    a_closed = float((math.sqrt(2.0) * A * s / B) ** (2.0 / 3.0))
    grid = np.logspace(-7, 0, 400_001)
    costs = np.array([_cost(float(a), A, B, s) for a in grid])
    a_numeric = float(grid[int(np.argmin(costs))])
    a_star = float(min(max(a_closed, 1e-9), 1.0))

    # AR(1) robustness: same objective with the measured persistence.
    scale_series = pd.Series(scale, index=train.index)
    scale_hl_days = float(causal_autocorr_halflife_days(scale_series)["halflife_days"])
    phi = float(math.exp(-math.log(2.0) / (scale_hl_days * BARS_PER_DAY)))
    sigma_x2 = float(np.var(sc))
    costs_ar1 = np.array([_cost_ar1(float(a), A, B, sigma_x2, phi) for a in grid])
    a_ar1 = float(grid[int(np.argmin(costs_ar1))])

    hl_bars = math.log(2.0) / -math.log(1.0 - a_star)
    d = Derivation(
        a_star=a_star, a_star_closed_form=a_closed, a_star_numeric=a_numeric,
        a_star_ar1=a_ar1, halflife_bars=hl_bars, halflife_days=hl_bars / BARS_PER_DAY,
        s=s, sigma_bar2=sigma_bar2, sigma_ann=sigma_ann, e_frac=e_frac,
        e_frac2=e_frac2, fee=fee, A=A, B=B,
        cost_at_astar=_cost(a_star, A, B, s), cost_at_one=_cost(1.0, A, B, s),
        n_bars=len(sc), phi=phi, scale_halflife_days=scale_hl_days,
    )
    if verbose:
        print("--- DERIVATION (inner-train BTC only; no backtest involved) ---")
        print(f"  inner-train slice      : {INNER_TRAIN_START} .. {INNER_TRAIN_END} "
              f"({d.n_bars:,} warm bars)")
        print(f"  fee (MarketSpec)       : {d.fee:.4%}")
        print(f"  sd(d scale) per bar  s : {d.s:.6e}")
        print(f"  Var(log ret) per bar   : {d.sigma_bar2:.6e}  "
              f"(annualized sigma = {d.sigma_ann:.4f})")
        print(f"  E[|frac|]              : {d.e_frac:.6f}")
        print(f"  E[frac^2]              : {d.e_frac2:.6f}")
        print(f"  A = sigma_bar^2/2 * E[frac^2]           = {d.A:.6e}")
        print(f"  B = fee * E[|frac|] * sqrt(2/pi)        = {d.B:.6e}")
        print("  C(a) = A*s^2*(1-a)^2/(a(2-a)) + B*s*sqrt(a/(2-a))")
        print("  a*   = ( sqrt(2) * A * s / B )^(2/3)")
        print(f"  a* (closed form)       : {d.a_star_closed_form:.6e}")
        print(f"  a* (numeric argmin)    : {d.a_star_numeric:.6e}")
        print(f"  a* (AR(1) refinement)  : {d.a_star_ar1:.6e}  "
              f"[phi={d.phi:.8f}, scale half-life {d.scale_halflife_days:.1f}d]")
        print(f"  FROZEN a*              : {d.a_star:.6e}  -> EWMA half-life "
              f"{d.halflife_bars:.0f} bars = {d.halflife_days:.2f} days")
        print(f"  C(a*) = {d.cost_at_astar:.4e}   C(1) = {d.cost_at_one:.4e}   "
              f"ratio = {d.cost_at_one / d.cost_at_astar:.1f}x")
        fires = abs(d.a_star - 1.0) < 0.5 / BARS_PER_DAY
        print(f"  PRE-REGISTERED FALSIFICATION TEST (a* indistinguishable from 1.0 "
              f"at bar resolution): {'FIRES' if fires else 'does not fire'}")
    return d


def sanity_check_halflife(df: pd.DataFrame | None = None) -> dict:
    """Re-measure r165_shared.py's pre-registered `vol` half-life ourselves."""
    if df is None:
        df = _load_btc()
    vol = realized_vol_series(df["close"], 8 * BARS_PER_DAY)
    hl = causal_autocorr_halflife_days(vol)
    gap = order_of_magnitude_gap(hl["halflife_days"])
    print("--- SANITY CHECK of r165_shared.py's pre-registered vol half-life ---")
    print(f"  causal_autocorr_halflife_days(vol) on inner-train BTC: "
          f"{hl['halflife_days']:.2f} days  (acf(1d)={hl['acf_lag1']:.4f}, "
          f"n_days={hl['n_days']})")
    print(f"  r165_shared.py's pre-registration states 47.2 days -- "
          f"{'REPRODUCED' if abs(hl['halflife_days'] - 47.2) < 1.0 else 'NOT REPRODUCED'}")
    print(f"  v4 anchor half-lives {V4_ANCHOR_HALFLIVES_DAYS} -> "
          f"order_of_magnitude_gap: ratio={gap['ratio']:.2f}, "
          f"in_same_band={gap['in_same_band']}")
    return {"halflife": hl, "gap": gap}


def causal_probe(df: pd.DataFrame, a: float, cut: int = 200_000) -> dict:
    """Truncation probe: targets on a truncated frame must match the prefix
    of the targets on the full frame (no bar may use a future bar)."""
    full = KellyRegimeV4EwmaScale(ewma_a=a).prepare(df.copy())["target"].to_numpy()
    trunc = KellyRegimeV4EwmaScale(ewma_a=a).prepare(df.iloc[:cut].copy())["target"].to_numpy()
    same = bool(np.allclose(full[:cut], trunc, atol=0.0, rtol=0.0))
    return {"identical": same,
            "max_abs_diff": float(np.max(np.abs(full[:cut] - trunc)))}


# =============================================================== running
@dataclass
class Cell:
    label: str
    slice_name: str
    market: str
    final_balance: float
    sharpe: float
    max_dd: float
    num_trades: int
    n_fills: int
    log_growth: float
    daily: np.ndarray
    time_in_market: float
    realized_vol: float


def _run(strategy, df: pd.DataFrame, start, end, slice_name: str,
         market: MarketSpec, label: str, funding: pd.Series | None = None) -> Cell:
    if funding is None:
        res = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    else:
        lo = 0 if start is None else int(df.index.searchsorted(start))
        hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
        pre = min(lo, strategy.warmup)
        raw = run_backtest(strategy, df.iloc[lo - pre: hi], market, 1_000.0,
                           trade_start=pre, funding=funding)
        res = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:],
                                           df=raw.df.iloc[pre:])
    m = compute_metrics(res)
    d = daily_returns(res.equity).to_numpy()
    tgt = res.df["target"].to_numpy() if "target" in res.df.columns else np.array([np.nan])
    return Cell(label=label, slice_name=slice_name, market=market.name,
                final_balance=m.final_balance, sharpe=m.sharpe,
                max_dd=m.max_drawdown_pct, num_trades=m.num_trades,
                n_fills=len(res.fills), log_growth=float(total_log_return(d)),
                daily=d, time_in_market=float(np.mean(np.abs(tgt) > 1e-9)),
                realized_vol=float(np.std(d) * math.sqrt(365.25)) if len(d) > 1 else float("nan"))


def _paired(a_cell: Cell, b_cell: Cell, seed: int = 0) -> dict:
    n = min(len(a_cell.daily), len(b_cell.daily))
    a, b = a_cell.daily[-n:], b_cell.daily[-n:]
    pg = paired_bootstrap(a, b, total_log_return, mean_block=30.0, n_boot=2_000, seed=seed)
    ps = paired_bootstrap(a, b, annualized_sharpe, mean_block=30.0, n_boot=2_000, seed=seed)
    return {
        "d_log_growth": pg.diff.point, "g_lo": pg.diff.lo, "g_hi": pg.diff.hi,
        "g_excl0": bool(pg.diff.lo > 0 or pg.diff.hi < 0),
        "d_sharpe_boot": ps.diff.point, "s_lo": ps.diff.lo, "s_hi": ps.diff.hi,
        "s_excl0": bool(ps.diff.lo > 0 or ps.diff.hi < 0),
    }


def compare(a: float, df: pd.DataFrame, start, end, slice_name: str,
            market: MarketSpec, label: str, control_cell: Cell | None = None,
            funding: pd.Series | None = None) -> dict:
    """One candidate cell vs the kelly_regime_v4 control on the same window."""
    cand = _run(KellyRegimeV4EwmaScale(ewma_a=a), df, start, end, slice_name,
                market, label, funding=funding)
    ctrl = control_cell or _run(get_strategy("kelly_regime_v4"), df, start, end,
                                slice_name, market, "kelly_regime_v4", funding=funding)
    boot = _paired(cand, ctrl)
    exp_ratio = cand.time_in_market / ctrl.time_in_market if ctrl.time_in_market else float("nan")
    vol_ratio = cand.realized_vol / ctrl.realized_vol if ctrl.realized_vol else float("nan")
    row = {
        "label": label, "a": a, "slice": slice_name, "market": market.name,
        "cand_final": cand.final_balance, "ctrl_final": ctrl.final_balance,
        "cand_sharpe": cand.sharpe, "ctrl_sharpe": ctrl.sharpe,
        "d_sharpe": cand.sharpe - ctrl.sharpe,
        "cand_dd": cand.max_dd, "ctrl_dd": ctrl.max_dd,
        "d_dd": cand.max_dd - ctrl.max_dd,
        "cand_fills": cand.n_fills, "ctrl_fills": ctrl.n_fills,
        "cand_trades": cand.num_trades, "ctrl_trades": ctrl.num_trades,
        "cand_tim": cand.time_in_market, "ctrl_tim": ctrl.time_in_market,
        "cand_vol": cand.realized_vol, "ctrl_vol": ctrl.realized_vol,
        "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
        "risk_matched": bool(abs(exp_ratio - 1.0) <= RISK_MATCH_TOL
                             and abs(vol_ratio - 1.0) <= RISK_MATCH_TOL),
        "cand_log_growth": cand.log_growth, "ctrl_log_growth": ctrl.log_growth,
    }
    row.update(boot)
    row["_cand"], row["_ctrl"] = cand, ctrl
    return row


HDR = (f"{'label':24s} {'slice':12s} {'market':11s} {'a':>9s} {'cand$':>11s} "
       f"{'ctrl$':>11s} {'dSh':>6s} {'dDD':>7s} {'fills':>6s}/{'ctrl':<6s} "
       f"{'expR':>5s} {'volR':>5s} {'RM':>3s} {'dlogG':>7s} [{'lo':>7s},{'hi':>7s}] {'x0':>4s}")


def print_rows(rows: list[dict]) -> None:
    print(HDR)
    print("-" * len(HDR))
    for r in rows:
        print(f"{r['label'][:24]:24s} {r['slice']:12s} {r['market']:11s} "
              f"{r['a']:9.2e} {r['cand_final']:11,.0f} {r['ctrl_final']:11,.0f} "
              f"{r['d_sharpe']:+6.2f} {r['d_dd']:+7.1f} "
              f"{r['cand_fills']:6d}/{r['ctrl_fills']:<6d} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['d_log_growth']:+7.3f} [{r['g_lo']:+7.3f},{r['g_hi']:+7.3f}] "
              f"{'YES' if r['g_excl0'] else 'no':>4s}")


# ================================================================ stages
def stage_inner(d: Derivation) -> list[dict]:
    """Sensitivity sweep around the frozen a*, inner-train + inner-validation.

    DIAGNOSTIC ONLY: the PRIMARY config is a*, fixed by the derivation above
    before any of these numbers existed. No cell here can promote a
    different rate.
    """
    btc = _pre_holdout(_load_btc())
    rates = []
    for m in SENSITIVITY_MULTIPLIERS:
        a = min(d.a_star * m, 1.0)
        rates.append((m, a))
    rows = []
    slices = (("inner_train", INNER_TRAIN_START, INNER_TRAIN_END),
              ("inner_val", INNER_VAL_START, INNER_VAL_END))
    for slice_name, start, end in slices:
        for market in (SPOT, FUTURES):
            ctrl = _run(get_strategy("kelly_regime_v4"), btc, start, end,
                        slice_name, market, "kelly_regime_v4")
            for mult, a in rates:
                tag = "PRIMARY" if mult == 1.0 else f"x{mult:g}"
                rows.append(compare(a, btc, start, end, slice_name, market,
                                    f"ewma_{tag}", control_cell=ctrl))
    return rows


def turnover_diagnostic(d: Derivation) -> list[dict]:
    """Where the derivation's fee term meets v4's deadband.

    The derivation prices turnover as `E|d f| = E[|frac|] * sqrt(2/pi) *
    s * sqrt(a/(2-a))`, i.e. the mean absolute per-bar move of the
    *frictionless desired* exposure, which scales like `sqrt(a)` for small
    `a`. v4 does not trade the desired path: it trades the deadband-latched
    path. This function measures both, on inner-train BTC, so the gap
    between the modelled elasticity and the realized one is a number rather
    than an argument.
    """
    btc = _pre_holdout(_load_btc())
    train = btc.loc[:INNER_TRAIN_END]
    scale, frac = _v4_scale_and_frac(train)
    rows = []
    for mult in SENSITIVITY_MULTIPLIERS:
        a = min(d.a_star * mult, 1.0)
        eff = np.empty_like(scale)
        cur, seeded = 0.0, False
        for i, sc in enumerate(scale):
            if not seeded:
                cur = sc
                seeded = sc > 0.0
            else:
                cur = (1.0 - a) * cur + a * sc
            eff[i] = cur
        desired = frac * eff
        tgt = KellyRegimeV4EwmaScale(ewma_a=a).prepare(train.copy())["target"].to_numpy()
        rows.append({
            "mult": mult, "a": a,
            "model_sd_dy": d.s * math.sqrt(a / (2.0 - a)),
            "model_E_abs_df": (d.e_frac * math.sqrt(2.0 / math.pi)
                               * d.s * math.sqrt(a / (2.0 - a))),
            "actual_E_abs_d_eff": float(np.mean(np.abs(np.diff(eff)))),
            "actual_E_abs_d_desired": float(np.mean(np.abs(np.diff(desired)))),
            "actual_E_abs_d_target": float(np.mean(np.abs(np.diff(tgt)))),
            "n_target_jumps": int(np.sum(np.abs(np.diff(tgt)) > 1e-12)),
        })
    return rows


def plateau_check(sweep_rows: list[dict]) -> dict:
    """D5: is the neighbourhood of a* a plateau rather than a peak?

    A cliff means a neighbour's Sharpe-vs-v4 sits outside the +/-0.2 noise
    floor from PRIMARY's on the same cell.
    """
    out = {}
    for slice_name in ("inner_train", "inner_val"):
        for market in ("spot", "futures_5x"):
            cells = [r for r in sweep_rows
                     if r["slice"] == slice_name and r["market"] == market]
            if not cells:
                continue
            prim = next(r for r in cells if r["label"] == "ewma_PRIMARY")
            gaps = {r["label"]: r["d_sharpe"] - prim["d_sharpe"] for r in cells}
            out[f"{slice_name}/{market}"] = {
                "d_sharpe_primary": prim["d_sharpe"], "neighbour_gaps": gaps,
                "plateau": all(abs(g) <= SHARPE_NOISE_FLOOR for g in gaps.values()),
            }
    return out


def stage_holdout(d: Derivation, sweep_rows: list[dict]) -> dict:
    """The single holdout read, at the frozen PRIMARY a*, plus D0-D6."""
    a = d.a_star
    btc = _load_btc()
    out: dict = {"a": a}

    # --- D1 core holdout comparison (also supplies D0 and D4) ---
    hold = []
    for market in (SPOT, FUTURES):
        hold.append(compare(a, btc, OOS_START, None, "holdout", market, "ewma_PRIMARY"))
    out["holdout"] = hold

    # --- D2 cost-mechanism: same comparison at the 0.40% tier ---
    fee_rows = []
    for market in (SPOT, FUTURES):
        m40 = replace(market, fee_rate=FEE_TIER)
        fee_rows.append(compare(a, btc, OOS_START, None, "holdout_fee40", m40,
                                "ewma_PRIMARY_fee40"))
    out["fee40"] = fee_rows

    # --- D3 ETH-A falsification (Bitfinex ETH, 2016-03 -> 2019-12) ---
    eth = _load_eth()
    eth_rows = []
    for market in (SPOT, FUTURES):
        eth_rows.append(compare(a, eth, None, None, "eth", market, "ewma_PRIMARY_eth"))
    out["eth"] = eth_rows
    return out


def _fmt_pair(r: dict) -> str:
    return (f"{r['market']:11s} cand ${r['cand_final']:>12,.0f} vs v4 ${r['ctrl_final']:>12,.0f} | "
            f"Sharpe {r['cand_sharpe']:+.2f} vs {r['ctrl_sharpe']:+.2f} (d={r['d_sharpe']:+.2f}) | "
            f"DD {r['cand_dd']:.1f}% vs {r['ctrl_dd']:.1f}% | "
            f"fills {r['cand_fills']} vs {r['ctrl_fills']} | trades {r['cand_trades']} vs {r['ctrl_trades']} | "
            f"TiM {r['cand_tim']:.3f} vs {r['ctrl_tim']:.3f} | "
            f"vol {r['cand_vol']:.3f} vs {r['ctrl_vol']:.3f}")


def main(argv: list[str]) -> None:
    stage = argv[1] if len(argv) > 1 else "all"
    print("=" * 110)
    print("R-165 NOVEL: derived-rate EWMA smoothing of kelly_regime_v4's `scale` factor")
    print("=" * 110)

    btc_full = _load_btc()
    sanity_check_halflife(btc_full)
    print()
    d = derive(btc_full)
    if stage == "derive":
        return

    print("\n--- CAUSALITY: truncation probe on the frozen PRIMARY config ---")
    probe = causal_probe(_pre_holdout(btc_full), d.a_star)
    print(f"  targets(full)[:cut] == targets(truncated): {probe['identical']} "
          f"(max abs diff {probe['max_abs_diff']:.3e})")
    assert probe["identical"], "CAUSALITY PROBE FAILED"

    print("\n--- IDENTITY KILL SWITCH: a=1.0 must reproduce kelly_regime_v4 ---")
    pre = _pre_holdout(btc_full)
    t_a1 = KellyRegimeV4EwmaScale(ewma_a=1.0).prepare(pre.copy())["target"].to_numpy()
    t_v4 = get_strategy("kelly_regime_v4").prepare(pre.copy())["target"].to_numpy()
    print(f"  max abs diff = {np.max(np.abs(t_a1 - t_v4)):.3e}  "
          f"identical={np.array_equal(t_a1, t_v4)}")
    assert np.array_equal(t_a1, t_v4), "a=1.0 does not reproduce v4 bit-for-bit"

    rows: list[dict] = []
    if stage in ("inner", "all"):
        print("\n--- SENSITIVITY SWEEP (diagnostic only; PRIMARY was frozen by the "
              "derivation) : 5 rates x 2 markets x 2 slices = 20 cells ---")
        rows = stage_inner(d)
        print_rows(rows)
        print(f"\n  configurations evaluated in the sweep: {len(rows)}")

        print("\n--- TURNOVER DIAGNOSTIC (inner-train BTC): the derivation's fee "
              "term vs what the deadband actually trades ---")
        print(f"  {'mult':>5s} {'a':>9s} {'model E|df|/bar':>16s} "
              f"{'actual E|d eff|':>16s} {'actual E|d desired|':>20s} "
              f"{'actual E|d target|':>19s} {'target jumps':>13s}")
        for t in turnover_diagnostic(d):
            print(f"  x{t['mult']:<4g} {t['a']:9.2e} {t['model_E_abs_df']:16.3e} "
                  f"{t['actual_E_abs_d_eff']:16.3e} {t['actual_E_abs_d_desired']:20.3e} "
                  f"{t['actual_E_abs_d_target']:19.3e} {t['n_target_jumps']:13d}")

        print("\n--- D5 PLATEAU CHECK (Sharpe-vs-v4 gap of each neighbour from "
              f"PRIMARY, noise floor +/-{SHARPE_NOISE_FLOOR}) ---")
        for cell, info in plateau_check(rows).items():
            gaps = "  ".join(f"{k.replace('ewma_', '')}:{v:+.2f}"
                             for k, v in info["neighbour_gaps"].items())
            print(f"  {cell:24s} dSh(PRIMARY)={info['d_sharpe_primary']:+.2f}  "
                  f"plateau={info['plateau']}  gaps: {gaps}")

    if stage in ("holdout", "all"):
        print("\n" + "=" * 110)
        print(f"HOLDOUT READ (once, frozen a* = {d.a_star:.6e}, half-life "
              f"{d.halflife_days:.2f} days)")
        print("=" * 110)
        res = stage_holdout(d, rows)
        print("\nD1 holdout, BTC, 0.10%/0.05% fees:")
        for r in res["holdout"]:
            print("  " + _fmt_pair(r))
            print(f"    paired log-growth diff {r['d_log_growth']:+.4f} "
                  f"[{r['g_lo']:+.4f}, {r['g_hi']:+.4f}] excl0={r['g_excl0']}   "
                  f"Sharpe diff {r['d_sharpe_boot']:+.4f} "
                  f"[{r['s_lo']:+.4f}, {r['s_hi']:+.4f}] excl0={r['s_excl0']}")
        print("\nD2 holdout at the 0.40% fee tier:")
        for r in res["fee40"]:
            print("  " + _fmt_pair(r))
            print(f"    paired log-growth diff {r['d_log_growth']:+.4f} "
                  f"[{r['g_lo']:+.4f}, {r['g_hi']:+.4f}] excl0={r['g_excl0']}")
        print("\nD3 ETH-A falsification (Bitfinex ETH 2016-03 -> 2019-12):")
        for r in res["eth"]:
            print("  " + _fmt_pair(r))
            print(f"    paired log-growth diff {r['d_log_growth']:+.4f} "
                  f"[{r['g_lo']:+.4f}, {r['g_hi']:+.4f}] excl0={r['g_excl0']}")

        print("\n--- D0-D6 verdict ---")
        d0 = all(r["risk_matched"] for r in res["holdout"])
        d1 = all((r["g_excl0"] and r["d_log_growth"] > 0)
                 or (r["s_excl0"] and r["d_sharpe_boot"] > 0) for r in res["holdout"])
        d2 = all(f["d_log_growth"] > h["d_log_growth"]
                 for f, h in zip(res["fee40"], res["holdout"]))
        d3 = all(np.sign(r["d_log_growth"]) >= 0 for r in res["eth"])
        d4 = all(r["cand_fills"] < r["ctrl_fills"] for r in res["holdout"])
        print(f"  D0 risk-match (both markets within {RISK_MATCH_TOL:.0%}) : {d0}")
        print(f"  D1 paired bootstrap favourable & significant, both markets: {d1}")
        print(f"  D2 advantage grows at 0.40%                              : {d2}")
        print(f"  D3 ETH sign not reversed                                 : {d3}")
        print(f"  D4 fill count falls vs v4                                : {d4}")
        verdict = "PROMOTE" if (d0 and d1 and d2 and d3 and d4) else "REJECT"
        print(f"  => VERDICT (before D5 plateau downgrade): {verdict}")
    print("\ndone.")


if __name__ == "__main__":
    main(sys.argv)
