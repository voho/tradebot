"""R-119, CONSERVATIVE branch: select kelly_regime_v4's free parameters
((ladder_base, target_vol, max_leverage), the same 12-point pre-registered
grid `r119_shared.GRID`) via a PLAIN GEOMETRIC BROWNIAN MOTION diffusion
plus a compound-Poisson jump overlay calibrated EXCLUSIVELY from the three
external literature constants frozen in `experiments/r119_shared.py`
(`EXT_JUMP_PROB_PER_DAY`, `EXT_JUMP_UP_MEAN/STD`, `EXT_JUMP_DOWN_MEAN/STD`).

Mechanism, citations, and the falsification test are all pre-registered in
`experiments/r119_shared.py`'s module docstring -- read that first. This
file supplies exactly one thing `r119_shared.py` leaves to the branch: the
synthetic `path_generator(seed)` used to build the `N_DRAWS` alternative
price histories the grid is scored across.

**GBM parameters -- picked ONCE, before any run, never tuned against
results:**

- Annualized volatility `GBM_SIGMA_ANNUAL = 0.80` (80%/yr). This is the
  standard coarse, round-number figure commonly cited for Bitcoin's
  long-run realized/historical annualized volatility across many
  retail-facing and industry sources (typically quoted somewhere in the
  ~60-100%/yr range depending on the exact window measured, with ~80%
  the figure most often repeated as a single round headline number,
  e.g. in Kraken/CME/Deribit educational materials and periodic
  volatility-index commentary). Chosen from general knowledge, NOT fit
  to `data/btcusd_*` or any file this project ships -- picking the
  single most commonly-repeated round number in the pre-registered
  60-100%/yr instructed range, rather than measuring anything.
- Annualized drift `GBM_MU_ANNUAL = 0.0`. A flat, driftless diffusion --
  the simplest possible choice, requiring no calibration decision at all
  and introducing no implicit bullish/bearish view. Consistent with the
  "conservative" branch's whole point (no additional structure beyond
  the external jump numbers): this is standard-GBM notation where `mu`
  is the (arithmetic) drift of the process before the usual Ito
  variance correction is subtracted in the log-return step, so the
  realized log-price path still very slightly decays in expectation
  purely from `-0.5*sigma^2*dt`, exactly as in any textbook GBM with
  zero drift, not a hand-tuned bearish assumption.

**Jump overlay -- exactly the R-119-shared construction, nothing else:**
each of the `n_days = 1461` calendar days in the 420,768-bar path
independently gets a jump with probability `EXT_JUMP_PROB_PER_DAY`
(~1/7, Scaillet/Treccani/Trevisan 2020); when a day is chosen, ONE bar
within that day (uniform over the 288 intraday bars) receives an extra
log-return jump, sign 50/50, magnitude drawn from
`Normal(EXT_JUMP_UP_MEAN, EXT_JUMP_UP_STD)` or
`Normal(EXT_JUMP_DOWN_MEAN, EXT_JUMP_DOWN_STD)` (MDPI Mathematics
9(20) 2567, 2021). No regime-switching, no volatility clustering, no
other structure -- deliberately, per this round's pre-registered
description of what makes the conservative branch conservative.

**Timing discipline.** A single `path_generator(0)` build plus a single
`score_on_path` backtest on the full 420,768-bar synthetic frame is timed
in `main()` before committing to the full `len(GRID) * N_DRAWS = 12 * 40
= 480`-backtest sweep, exactly as R-118's conservative branch did. Per
explicit instruction for this branch, `N_DRAWS` and `GRID` are run
UNREDUCED regardless of the estimate (R-118's equivalent sweep took
~30-35 minutes, comfortably inside the standing ~60-75 minute budget).

This module reads ZERO real market data -- `path_generator` touches only
`np.random.default_rng(seed)` and the four external constants imported
from `r119_shared`. The only real data touched anywhere in this file is
inside the single, frozen `evaluate_candidate()` call in `main()`, whose
own internals never read a bar at or after `OOS_START` (2023-01-01),
exactly as `r119_shared.py` and `r118_shared.py` both guarantee and
self-test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r119_shared as sh  # noqa: E402

# ------------------------------------------------------------------------
# GBM parameters -- fixed constants, picked before any run, see docstring.
# ------------------------------------------------------------------------
GBM_SIGMA_ANNUAL = 0.80   # 80%/yr, coarse commonly-cited BTC long-run vol
GBM_MU_ANNUAL = 0.0       # flat/driftless -- no calibration decision

N_BARS = 420_768           # 1461 days x 288 bars/day == real inner-train length
N_DAYS = N_BARS // sh.BARS_PER_DAY
assert N_DAYS * sh.BARS_PER_DAY == N_BARS
assert N_DAYS == 1461

START_PRICE = 1_000.0

DT = 1.0 / sh.BARS_PER_YEAR


def build_synthetic_path(seed: int) -> pd.DataFrame:
    """One GBM-diffusion + externally-calibrated compound-Poisson-jump
    synthetic OHLCV path, seeded deterministically. Reads no real data."""
    rng = np.random.default_rng(seed)

    # --- 1. plain GBM diffusion log-returns, one draw per bar ---
    mu_dt = (GBM_MU_ANNUAL - 0.5 * GBM_SIGMA_ANNUAL ** 2) * DT
    sigma_dt = GBM_SIGMA_ANNUAL * np.sqrt(DT)
    diffusion = rng.normal(mu_dt, sigma_dt, N_BARS)

    # --- 2. compound-Poisson jump overlay, calibrated EXCLUSIVELY from
    # r119_shared's external literature constants ---
    jump_returns = np.zeros(N_BARS, dtype=float)
    jump_day_mask = rng.random(N_DAYS) < sh.EXT_JUMP_PROB_PER_DAY
    jump_days = np.flatnonzero(jump_day_mask)
    if len(jump_days):
        bar_within_day = rng.integers(0, sh.BARS_PER_DAY, size=len(jump_days))
        is_up = rng.random(len(jump_days)) < 0.5
        up_draw = rng.normal(sh.EXT_JUMP_UP_MEAN, sh.EXT_JUMP_UP_STD, len(jump_days))
        down_draw = rng.normal(sh.EXT_JUMP_DOWN_MEAN, sh.EXT_JUMP_DOWN_STD, len(jump_days))
        jump_size = np.where(is_up, up_draw, down_draw)
        global_bar = jump_days * sh.BARS_PER_DAY + bar_within_day
        jump_returns[global_bar] = jump_size

    log_returns = diffusion + jump_returns
    close = START_PRICE * np.exp(np.cumsum(log_returns))

    new_index = pd.date_range("2017-01-01", periods=N_BARS, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": 1.0,
        },
        index=new_index,
    )

    # --- sanity checks on our own output before it is ever scored ---
    assert len(df) == N_BARS, f"length mismatch: {len(df)} != {N_BARS}"
    assert df.index.tz is not None, "index must be tz-aware"
    assert np.all(np.isfinite(df.to_numpy())), "non-finite value in synthetic path"
    assert (df[["open", "high", "low", "close"]] > 0).to_numpy().all(), "non-positive price"
    assert (df["high"].to_numpy() >= np.maximum(df["open"].to_numpy(), df["close"].to_numpy()) - 1e-9).all(), \
        "high < max(open, close)"
    assert (df["low"].to_numpy() <= np.minimum(df["open"].to_numpy(), df["close"].to_numpy()) + 1e-9).all(), \
        "low > min(open, close)"

    return df


def path_generator(seed: int) -> pd.DataFrame:
    return build_synthetic_path(seed)


def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    _hr("R-119 CONSERVATIVE: GBM diffusion + external compound-Poisson jump")

    print(f"GBM_SIGMA_ANNUAL={GBM_SIGMA_ANNUAL}  GBM_MU_ANNUAL={GBM_MU_ANNUAL}")
    print(f"EXT_JUMP_PROB_PER_DAY={sh.EXT_JUMP_PROB_PER_DAY:.6f}  "
          f"EXT_JUMP_UP=({sh.EXT_JUMP_UP_MEAN},{sh.EXT_JUMP_UP_STD})  "
          f"EXT_JUMP_DOWN=({sh.EXT_JUMP_DOWN_MEAN},{sh.EXT_JUMP_DOWN_STD})")
    print(f"N_BARS={N_BARS}  N_DAYS={N_DAYS}  grid: {sh.GRID} ({len(sh.GRID)} points)")

    # --- determinism + reasonable-shape check ---
    p0a = build_synthetic_path(0)
    p0b = build_synthetic_path(0)
    assert p0a["close"].equals(p0b["close"]), "path_generator not deterministic for seed=0"
    p1 = build_synthetic_path(1)
    assert not p0a["close"].equals(p1["close"]), "different seeds produced identical paths"
    print(f"determinism check OK. seed=0 path: start={p0a['close'].iloc[0]:.2f} "
          f"end={p0a['close'].iloc[-1]:.2f} min={p0a['close'].min():.2f} "
          f"max={p0a['close'].max():.2f}")

    # --- timing discipline: time path build + one score_on_path call ---
    t0 = time.time()
    probe_path = build_synthetic_path(0)
    t1 = time.time()
    print(f"path_generator(0) build time: {t1 - t0:.2f}s")

    t2 = time.time()
    sh.score_on_path(sh.GRID[0], probe_path, sh.SPOT)
    t3 = time.time()
    per_backtest = t3 - t2
    print(f"single score_on_path timing: {per_backtest:.2f}s/backtest")

    n_draws = sh.N_DRAWS
    est_total_s = per_backtest * len(sh.GRID) * n_draws
    print(f"estimated full sweep ({len(sh.GRID)} x {n_draws} = "
          f"{len(sh.GRID) * n_draws} backtests): {est_total_s / 60.0:.1f} min "
          f"(running unreduced grid/N_DRAWS per instruction, regardless of estimate)")

    _hr(f"Running selection sweep: {len(sh.GRID)} configs x {n_draws} draws "
        f"= {len(sh.GRID) * n_draws} backtests")
    t_sweep0 = time.time()
    best_config, table = sh.select_config(path_generator, n_draws=n_draws)
    t_sweep1 = time.time()
    print(f"sweep wall time: {(t_sweep1 - t_sweep0) / 60.0:.1f} min")

    _hr("Selection table (mean / std / robust CVaR-25% Sharpe across synthetic draws)")
    for cfg in sh.GRID:
        row = table[cfg]
        marker = "  <== WINNER" if cfg == best_config else ""
        marker += "  <== v4 default" if cfg == sh.V4_DEFAULT else ""
        print(f"  base={cfg[0]:3d} target_vol={cfg[1]:.2f} max_lev={cfg[2]:.2f}  "
              f"mean={row['mean']:+.3f} std={row['std']:.3f} "
              f"robust={row['robust']:+.3f}{marker}")

    print(f"\nWinning config (max robust CVaR-25% Sharpe over {n_draws} synthetic draws): "
          f"{best_config}")
    print(f"v4 shipped default for comparison: {sh.V4_DEFAULT}  "
          f"robust={table[sh.V4_DEFAULT]['robust']:+.3f}")

    _hr("Step 4: frozen real-data evaluate_candidate (called exactly once)")
    result = sh.evaluate_candidate(best_config, "R119_conservative")
    sh.print_report(result)

    # --- save everything for reproducibility ---
    out_path = ROOT / "experiments" / "r119_conservative_results.json"
    payload = {
        "branch": "conservative_gbm_jump",
        "gbm_sigma_annual": GBM_SIGMA_ANNUAL,
        "gbm_mu_annual": GBM_MU_ANNUAL,
        "ext_jump_prob_per_day": sh.EXT_JUMP_PROB_PER_DAY,
        "ext_jump_up_mean": sh.EXT_JUMP_UP_MEAN,
        "ext_jump_up_std": sh.EXT_JUMP_UP_STD,
        "ext_jump_down_mean": sh.EXT_JUMP_DOWN_MEAN,
        "ext_jump_down_std": sh.EXT_JUMP_DOWN_STD,
        "n_draws_used": n_draws,
        "n_draws_preregistered": sh.N_DRAWS,
        "grid": [list(c) for c in sh.GRID],
        "n_bars": N_BARS,
        "n_days": N_DAYS,
        "per_backtest_seconds_probe": per_backtest,
        "sweep_wall_seconds": t_sweep1 - t_sweep0,
        "total_backtests": len(sh.GRID) * n_draws,
        "selection_table": {
            f"{c[0]},{c[1]},{c[2]}": v for c, v in table.items()
        },
        "best_config": list(best_config),
        "v4_default": list(sh.V4_DEFAULT),
        "evaluate_candidate_result": result,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved selection table + evaluation to {out_path}")


if __name__ == "__main__":
    main()
