"""R-166 Step-0 gate: does R-10's vol-quintile forward-Sharpe finding survive
a CAUSAL, inner-train-only re-measurement?

R-10 (08-15, docs/LEDGER.md) reported "high vol forecasts the highest
forward Sharpe (+1.08 all bars, +2.06 gate-bullish)" from a single
measurement whose write-up records no methodology detail (no causality
note, no window scope). R-166's whole direction -- inverting the sign of
v4's volatility response -- rests on that finding being real and not an
artifact of full-series quantile cut-points (a lookahead class this
project's own causality rules exist to catch). This script re-measures it
properly, cheaply, before any strategy code is written:

- inner-train only (2017-01-01 -> 2020-12-31), no holdout read
- v4's OWN lagged vol series (vol_span=8d EWM of log returns, shift(1))
  so the bucketing is the exact quantity the strategy would condition on
- EXPANDING-window quintile cut-points (each bar's bucket uses only
  cut-points fit on bars up to and including that bar -- no full-series
  quantile lookahead)
- forward 5-day log return per bucket, annualized Sharpe
- stationary block bootstrap (5-day blocks, 1000 reps) for a 95% CI
- reported both unconditionally and restricted to bars where v4's own
  vote fraction > 0 (R-10's "gate-bullish" reading)

Kill condition (frozen before running): S(q5) - S(q1) <= 0, or its 95%
bootstrap interval contains zero, in EITHER reading -> R-166 stops here,
recorded as a Step-0 kill, no conservative/novel branch built.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from tradebot.data import load_dataset
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR

RNG_SEED = 20260827  # fixed, pre-registered
N_BOOT = 1000
BLOCK_DAYS = 5
FWD_DAYS = 5
N_QUANTILES = 5
VOL_SPAN = 8 * BARS_PER_DAY
TRAIN_END = "2020-12-31"


def stationary_block_bootstrap(x: np.ndarray, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary block bootstrap, mean block length ``block``."""
    n = len(x)
    out = np.empty(n_boot)
    p = 1.0 / block
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
        out[b] = x[idx].mean()
    return out


def annualized_sharpe(daily_fwd_logret: np.ndarray, horizon_days: int) -> float:
    """Sharpe of the (non-overlapping-adjusted) forward-return series, annualized."""
    mu = daily_fwd_logret.mean()
    sd = daily_fwd_logret.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    periods_per_year = 365.25 / horizon_days
    return float(mu / sd * np.sqrt(periods_per_year))


def main() -> None:
    df, label = load_dataset(ROOT / "data", "spot")
    df = df.loc[:TRAIN_END]
    print(f"inner-train: {len(df):,} bars {df.index[0]} -> {df.index[-1]} (data: {label})")

    close = df["close"]
    r = np.log(close).diff()

    # v4's own lagged realized-vol series (identical construction to
    # kelly_regime.KellyRegime.prepare / kelly_regime_v3).
    vol = (r.ewm(span=VOL_SPAN, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1)

    # v4's own vote fraction (20/40/80-day anchors), for the gate-bullish split.
    horizons = (20, 40, 80)
    band = 0.01
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    frac = sum(votes) / len(votes)

    # Forward FWD_DAYS log return, sampled once per day (avoid 288x
    # oversampling autocorrelation in the bootstrap) at daily bar-close.
    daily_close = close.resample("1D").last().dropna()
    daily_vol = vol.reindex(daily_close.index, method="ffill")
    daily_frac = frac.reindex(daily_close.index, method="ffill")
    fwd_logret = (np.log(daily_close).shift(-FWD_DAYS) - np.log(daily_close))

    data = pd.DataFrame({"vol": daily_vol, "frac": daily_frac, "fwd": fwd_logret}).dropna()
    print(f"daily observations after alignment: {len(data):,}")

    # Causal expanding-window quintile assignment: bucket[t] uses cut-points
    # fit on data[:t] only (min 365 days before any bucket is assigned).
    MIN_HISTORY = 365
    vol_arr = data["vol"].to_numpy()
    n = len(vol_arr)
    bucket = np.full(n, -1, dtype=np.int64)
    for t in range(MIN_HISTORY, n):
        cuts = np.quantile(vol_arr[:t], np.linspace(0, 1, N_QUANTILES + 1))
        cuts[0], cuts[-1] = -np.inf, np.inf
        bucket[t] = np.searchsorted(cuts, vol_arr[t], side="right") - 1
        bucket[t] = min(max(bucket[t], 0), N_QUANTILES - 1)
    data["bucket"] = bucket
    data = data.iloc[MIN_HISTORY:]
    print(f"post-warmup observations: {len(data):,}\n")

    rng = np.random.default_rng(RNG_SEED)
    block = BLOCK_DAYS

    def report(subset: pd.DataFrame, title: str) -> tuple[float, float]:
        print(f"=== {title} (n={len(subset):,}) ===")
        sharpes = {}
        for q in range(N_QUANTILES):
            g = subset[subset["bucket"] == q]["fwd"].to_numpy()
            g = g[np.isfinite(g)]
            s = annualized_sharpe(g, FWD_DAYS)
            sharpes[q] = s
            print(f"  q{q + 1} (n={len(g):>5d}): Sharpe = {s:+.3f}")
        q1, q5 = subset[subset["bucket"] == 0]["fwd"].dropna().to_numpy(), \
            subset[subset["bucket"] == N_QUANTILES - 1]["fwd"].dropna().to_numpy()
        spread_point = sharpes[N_QUANTILES - 1] - sharpes[0]
        print(f"  spread S(q5)-S(q1) = {spread_point:+.3f}")

        # Paired-by-position is not meaningful here (different bars); bootstrap
        # each quantile's mean-return path independently via block resampling
        # of ITS OWN bar sequence, then recombine into the Sharpe spread.
        boot_spread = np.empty(N_BOOT)
        for b in range(N_BOOT):
            def resample_block(x: np.ndarray) -> np.ndarray:
                nx = len(x)
                if nx == 0:
                    return x
                idx = np.empty(nx, dtype=np.int64)
                i = rng.integers(0, nx)
                p = 1.0 / block
                for t in range(nx):
                    idx[t] = i
                    i = rng.integers(0, nx) if rng.random() < p else (i + 1) % nx
                return x[idx]

            r1 = resample_block(q1)
            r5 = resample_block(q5)
            s1 = annualized_sharpe(r1, FWD_DAYS) if len(r1) else np.nan
            s5 = annualized_sharpe(r5, FWD_DAYS) if len(r5) else np.nan
            boot_spread[b] = s5 - s1
        boot_spread = boot_spread[np.isfinite(boot_spread)]
        lo, hi = np.percentile(boot_spread, [2.5, 97.5])
        print(f"  95% CI on spread (block bootstrap, {block}-day blocks, {len(boot_spread)} reps): "
              f"[{lo:+.3f}, {hi:+.3f}]")
        excludes_zero = lo > 0 or hi < 0
        print(f"  excludes zero: {excludes_zero}\n")
        return spread_point, (lo, hi)

    spread_all, ci_all = report(data, "unconditional")
    spread_bull, ci_bull = report(data[data["frac"] > 0], "gate-bullish (frac > 0)")

    kill_all = spread_all <= 0 or not (ci_all[0] > 0 or ci_all[1] < 0)
    kill_bull = spread_bull <= 0 or not (ci_bull[0] > 0 or ci_bull[1] < 0)

    print("=" * 60)
    print(f"KILL (unconditional):  {kill_all}")
    print(f"KILL (gate-bullish):   {kill_bull}")
    print(f"OVERALL GATE RESULT:   {'KILL R-166' if (kill_all or kill_bull) else 'PROCEED'}")


if __name__ == "__main__":
    main()
