"""Shared, read-only pre-registration engine for the R-178 round (08-28).

DIRECTION, one sentence: layer a synthetic, causally-priced Deribit-style
options structure (strikes and premia from Black-Scholes off REAL DVOL,
not a new predictor) on top of `kelly_regime_v4`'s own unmodified position,
trading a once-per-regime (N~3) detection problem for a per-expiry
(N~150-280 weekly cycles over DVOL's ~5.4y history) cost/premium problem.
Full Step 1/Step 2 design -- the four-question filter, citations, the
non-duplication argument against R-73/R-136/R-170, and both branches'
frozen falsification rules -- is in `experiments/r178_direction.md`.

This module is DELIBERATELY neutral between the two branches: it exposes
`vote_frac` (v4's own unmodified vote, reproduced read-only so the novel
branch can condition stance on it without touching detection) and
`simulate_overlay` (the shared, causal, weekly-rolling synthetic option
book) as two primitives neither branch may edit -- mirroring
r175_shared.py/r176_shared.py/r177_shared.py's convention.

Mechanics of `simulate_overlay`, stated once so neither branch re-derives
it differently:

- Every `roll_bars` bars, the currently open put+call pair is SETTLED at
  that bar's close (T=0, pure intrinsic value against the settlement
  price) and a fresh pair is opened at the SAME bar, struck off that
  bar's close and that bar's causal DVOL -- bar `i0+roll_bars` is
  simultaneously the old cycle's last mark and the new cycle's epoch-0
  (wash) mark, so no bar's price move is double-counted or skipped.
- Within a cycle both legs are marked to market at every bar via
  Black-Scholes: spot = that bar's close, sigma = that bar's CAUSAL DVOL
  (previous UTC day's published close, via `align_dvol_causal`, never
  same-day), r=0 (disclosed simplification -- no risk-free-rate series
  exists in this project's data, and BTC options venues typically trade
  close to a zero-rate convention). This project's numpy environment has
  no scipy (R-118/R-125's own finding, re-confirmed here); the normal CDF
  used inside Black-Scholes is the Abramowitz & Stegun (1964, 7.1.26)
  rational approximation, max absolute error 1.5e-7 -- negligible next to
  every other simplification already disclosed here.
- Notional is resized at every roll to `overlay_frac * running combined
  equity so far` (base v4 equity plus the overlay's own realized P&L up to
  that bar) -- a "second, independently-funded sleeve" simplification: two
  summed ledgers on one reported account, standard for how overlay P&L is
  reported in practice, NOT something a real single-margin-account fund
  could run completely unmodified. Disclosed explicitly in
  `r178_direction.md`.
- `cost_bps` is charged once per roll, on the opening bar only, as a
  synthetic bid/ask haircut on both legs' premium (no real Deribit order
  book is simulated) -- see the 0.40%-tier re-run in `r178_direction.md`.
- Implementation note: computed one roll-cycle at a time, each cycle fully
  vectorized over its ~`roll_bars` marks (not a per-bar Python loop over
  the whole ~2.4M-bar series) -- ~130-470 Python-level iterations per
  9-year run depending on `roll_bars`, not ~2.4M.

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_dvol_causal  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
DEFAULT_ROLL_DAYS = 7
DEFAULT_ROLL_BARS = DEFAULT_ROLL_DAYS * BARS_PER_DAY

ETH_DVOL_FILE = "eth_dvol_daily.csv.gz"
BTC_DVOL_FILE = "btc_dvol_daily.csv.gz"


# ------------------------------------------------------------- shared vote


def vote_frac(close: pd.Series, horizons: tuple[int, ...] = (20, 40, 80),
              band: float = 0.01) -> np.ndarray:
    """`kelly_regime_v4`'s own 3-anchor latched vote, verbatim, in [0, 1].

    Reproduced read-only (identical construction to `kelly_regime.py`'s
    `prepare()` and r177_shared.py's `unsigned_vote_frac`) purely so the
    novel branch can condition option STANCE on the vote v4 already
    computes -- detection itself is untouched by either branch here.
    """
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


# ------------------------------------------------------------- DVOL sigma


def load_dvol_sigma(data_dir: str | Path, filename: str, bars: pd.DataFrame) -> np.ndarray:
    """Causal implied-vol (as a FRACTION, e.g. 0.55 = 55%/yr) aligned to `bars`.

    DVOL is published in percentage POINTS (empirically confirmed: BTC
    DVOL close ranges 32.4-156.2 over 2021-03-24 -> 2026-08-21), so this
    divides by 100 before use in Black-Scholes. `align_dvol_causal` (the
    project's own shared helper, used unchanged from R-73/R-136) already
    enforces "a bar may only see the most recent day that closed strictly
    before T's own day" -- no additional shift is applied here.
    """
    path = Path(data_dir) / filename
    if not path.exists():
        return np.full(len(bars), np.nan)
    raw = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    raw.index = raw.index.tz_localize("UTC")
    raw = raw.astype(float).sort_index()
    aligned = align_dvol_causal(raw[["close"]], bars)
    return (aligned["close"].to_numpy() / 100.0)


# ------------------------------------------------------------- normal CDF (no scipy)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun (1964) 7.1.26 rational approximation of Phi(x).

    Max absolute error 1.5e-7. This project's environment has no scipy
    (R-118/R-125), so `scipy.stats.norm.cdf` is not available.
    """
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.where(x < 0, -1.0, 1.0)
    ax = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    erf = 1.0 - poly * np.exp(-ax * ax)
    return 0.5 * (1.0 + sign * erf)


def bs_price(S: np.ndarray, K: float, T: np.ndarray, sigma: np.ndarray, is_call: bool,
             r: float = 0.0) -> np.ndarray:
    """European Black-Scholes price, vectorized. `T` in years, `sigma` as a fraction.

    `T<=0` returns intrinsic value; non-finite/non-positive `sigma` returns
    intrinsic value too (a bar before DVOL coverage starts, or the exact
    settlement bar, cannot be/need not be priced off a time value that
    does not exist -- falls back to its floor, disclosed in
    `r178_direction.md`).
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    intrinsic = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    valid = np.isfinite(sigma) & (sigma > 1e-6) & (T > 1e-9) & np.isfinite(S) & (S > 0)
    out = intrinsic.copy()
    if not np.any(valid):
        return out
    Sv, Tv, sv = S[valid], T[valid], sigma[valid]
    sqrtT = np.sqrt(Tv)
    d1 = (np.log(Sv / K) + (r + 0.5 * sv * sv) * Tv) / (sv * sqrtT)
    d2 = d1 - sv * sqrtT
    if is_call:
        val = Sv * _norm_cdf(d1) - K * np.exp(-r * Tv) * _norm_cdf(d2)
    else:
        val = K * np.exp(-r * Tv) * _norm_cdf(-d2) - Sv * _norm_cdf(-d1)
    out[valid] = np.maximum(val, 0.0)
    return out


# ------------------------------------------------------------- overlay engine


def simulate_overlay(close: np.ndarray, sigma: np.ndarray, base_equity: np.ndarray,
                      put_stance: np.ndarray, call_stance: np.ndarray, overlay_frac: float,
                      put_moneyness: float = 0.90, call_moneyness: float = 1.10,
                      roll_bars: int = DEFAULT_ROLL_BARS,
                      cost_bps: float = 0.0, r: float = 0.0) -> dict:
    """Causal, weekly-rolling synthetic options overlay, vectorized per cycle.

    `put_stance[i]`/`call_stance[i]` in {-1, 0, +1}, read only at each
    cycle's OPENING bar `i0`, independently per leg: +1 = long that leg
    (pay premium), -1 = short that leg (receive premium), 0 = leg not
    traded this cycle. A collar is `put_stance≡+1, call_stance≡-1`; a
    long/short strangle is `put_stance==call_stance` (both +1 or both -1)
    on the same cycle. Both legs always use the SAME `qty` (one notional
    budget per cycle, `overlay_frac * combined equity so far`), so a
    collar's put and call sides are sized identically, matching the
    literal Israelov & Klein (2016) construction.

    Returns `combined_equity` (base_equity + cumulative overlay P&L, same
    length as `base_equity`), `overlay_pnl` (per-bar delta), `num_rolls`,
    `total_cost`.
    """
    n = len(close)
    assert len(sigma) == n and len(base_equity) == n
    assert len(put_stance) == n and len(call_stance) == n
    overlay_pnl = np.zeros(n)
    total_cost = 0.0
    num_rolls = 0
    cum_before_cycle = 0.0

    i0 = 0
    while i0 < n - 1:
        i1 = min(i0 + roll_bars, n - 1)
        idx = np.arange(i0, i1 + 1)
        S0 = close[i0]
        equity_so_far = base_equity[i0] + cum_before_cycle
        notional_budget = max(overlay_frac, 0.0) * max(equity_so_far, 0.0)
        qty = notional_budget / S0 if S0 > 0 else 0.0
        K_put = S0 * put_moneyness
        K_call = S0 * call_moneyness
        put_sign = float(put_stance[i0])
        call_sign = float(call_stance[i0])

        T = (i1 - idx) / BARS_PER_YEAR  # T[0] ~= roll_bars/BARS_PER_YEAR, T[-1] = 0.0
        S = close[idx]
        sig = sigma[idx]
        put_vals = bs_price(S, K_put, T, sig, is_call=False, r=r)
        call_vals = bs_price(S, K_call, T, sig, is_call=True, r=r)
        prev_put = np.concatenate(([put_vals[0]], put_vals[:-1]))
        prev_call = np.concatenate(([call_vals[0]], call_vals[:-1]))
        cycle_pnl = qty * (put_sign * (put_vals - prev_put) + call_sign * (call_vals - prev_call))

        open_cost = cost_bps * 1e-4 * qty * (abs(put_sign) * put_vals[0] + abs(call_sign) * call_vals[0])
        cycle_pnl[0] -= open_cost
        total_cost += open_cost
        num_rolls += 1

        overlay_pnl[idx] += cycle_pnl
        cum_before_cycle += cycle_pnl.sum()
        i0 = i1

    overlay_cum = np.cumsum(overlay_pnl)
    combined_equity = base_equity + overlay_cum

    # A short leg (stance=-1) has theoretically unbounded loss between
    # rolls; empirically confirmed to reach negative combined equity in
    # this round's own Step-0 sanity check (SELLER stance, overlay_frac=0.5,
    # DVOL-only window: min -$7,060 from a $1,000 start). Treat the first
    # non-positive bar exactly like this project's own broker treats
    # liquidation: the account is wiped out and stays flat (zero) from
    # that bar on, rather than reporting an unbounded negative number as
    # if it were a recoverable loss.
    liquidated = False
    wipeout = np.where(combined_equity <= 0)[0]
    if len(wipeout):
        liquidated = True
        j = int(wipeout[0])
        combined_equity = combined_equity.copy()
        combined_equity[j:] = 0.0

    return {
        "combined_equity": combined_equity,
        "overlay_pnl": overlay_pnl,
        "num_rolls": num_rolls,
        "total_cost": total_cost,
        "liquidated": liquidated,
    }


# ------------------------------------------------------------- light metrics


def sharpe_ratio(equity: np.ndarray) -> float:
    if len(equity) < 3:
        return 0.0
    prev = equity[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(equity) / prev, 0.0)
    sd = rets.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(rets.mean() / sd * np.sqrt(BARS_PER_YEAR))


def max_drawdown_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    return float(np.nanmax(dd) * 100.0)


def log_growth(equity: np.ndarray) -> float:
    """log(final/start); the additive quantity the project's own paired
    bootstrap intervals are usually computed on (matches R-177's convention)."""
    if len(equity) < 2 or equity[0] <= 0 or equity[-1] <= 0:
        return float("nan")
    return float(np.log(equity[-1] / equity[0]))
