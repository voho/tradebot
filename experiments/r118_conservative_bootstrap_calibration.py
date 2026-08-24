"""R-118, CONSERVATIVE branch: select kelly_regime_v4's free parameters
((ladder_base, target_vol, max_leverage), the same 12-point pre-registered
grid `r118_shared.GRID`) via a STATIONARY BLOCK BOOTSTRAP (Politis &
Romano, 1994) of the real 2017-2020 inner-train BTC bars, instead of
R-45's three fixed calendar folds.

Mechanism, machinery, citations, and the falsification test are all
pre-registered in `experiments/r118_shared.py`'s module docstring -- read
that first. This file supplies exactly one thing `r118_shared.py` leaves
to the branch: the synthetic `path_generator(seed)` used to build the
`N_DRAWS` alternative price histories the grid is scored across.

**Path construction.** For each seed:
  1. Draw one stationary-bootstrap index array of length
     `n = len(inner_train_btc)` via
     `tradebot.inference.stationary_bootstrap_indices(n, mean_block, 1, rng)`,
     `mean_block = MEAN_BLOCK_DAYS * BARS_PER_DAY` (8640 bars = 30 days),
     `rng = np.random.default_rng(seed)` -- deterministic per seed.
  2. Find the block boundaries in that index array: a new block starts
     wherever the index does NOT simply continue the previous position by
     +1 (mod n, so wraparound at the end of the real series is handled
     correctly, matching `stationary_bootstrap_indices`'s own wraparound
     rule).
  3. For each block (in order), slice the REAL OHLCV rows at those
     positions -- so every block carries its true intrabar high/low/volume
     structure verbatim, only relocated in time -- then multiplicatively
     rescale the whole block's OHLC by one constant factor so the block's
     own open matches the running price level left off by the end of the
     previous block (continuous splicing: each block's *relative* OHLC
     dynamics are preserved exactly; only the block's overall price level
     moves).
  4. Concatenate all rescaled blocks and stamp a fresh tz-aware 5-minute
     `DatetimeIndex` of the same length (calendar dates are arbitrary for
     a synthetic path; only bar spacing matters to the engine).

Every synthetic path is sanity-checked (positive prices, high >= max(open,
close), low <= min(open,close), no NaN/Inf, correct length) before it is
scored.

**Timing discipline.** A single `score_on_path` call on the full
420,768-bar inner-train frame was timed at ~4.05s. The pre-registered
sweep is `len(GRID) * N_DRAWS = 12 * 40 = 480` backtests, i.e. an
estimated ~32 minutes -- comfortably inside the ~60-75 minute budget this
round's dispatch note allows, so the full pre-registered `N_DRAWS = 40` is
used unreduced (this is disclosed, not assumed -- see the timing check in
`main()` below, which aborts the decision to keep N_DRAWS=40 only after
confirming the estimate).

Never reads a bar dated 2021-01-01 or later, except via the single
`evaluate_candidate` call in `main()`.
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

from tradebot.inference import stationary_bootstrap_indices  # noqa: E402

import experiments.r118_shared as sh  # noqa: E402

# ------------------------------------------------------------------------
# Real inner-train data (2017-01-01..2020-12-31 BTC bars) -- the ONLY
# source material for synthetic paths. Loaded once at module import.
# ------------------------------------------------------------------------
REAL_DF = sh.load_inner_train_btc()
sh.assert_no_holdout(REAL_DF, "r118 conservative inner_train")

N_REAL = len(REAL_DF)
MEAN_BLOCK_BARS = sh.MEAN_BLOCK_DAYS * sh.BARS_PER_DAY  # 30 days -> 8640 bars

_REAL_OPEN = REAL_DF["open"].to_numpy(dtype=float)
_REAL_HIGH = REAL_DF["high"].to_numpy(dtype=float)
_REAL_LOW = REAL_DF["low"].to_numpy(dtype=float)
_REAL_CLOSE = REAL_DF["close"].to_numpy(dtype=float)
_REAL_VOLUME = REAL_DF["volume"].to_numpy(dtype=float)


def _block_starts(idx: np.ndarray, n: int) -> np.ndarray:
    """Positions in ``idx`` where a new stationary-bootstrap block begins.

    A block *continues* at position t when ``idx[t] == (idx[t-1] + 1) % n``
    -- the exact wraparound rule `stationary_bootstrap_indices` itself
    uses internally. Position 0 is always a block start.
    """
    if n <= 1:
        return np.array([0], dtype=np.int64)
    continuation = idx[1:] == (idx[:-1] + 1) % n
    starts = np.flatnonzero(~continuation) + 1
    return np.concatenate(([0], starts)).astype(np.int64)


def build_synthetic_path(seed: int) -> pd.DataFrame:
    """One stationary-block-bootstrap synthetic OHLCV path, seeded
    deterministically. Real row blocks, spliced with continuous
    multiplicative price rescaling at each block boundary."""
    rng = np.random.default_rng(seed)
    idx_mat = stationary_bootstrap_indices(N_REAL, MEAN_BLOCK_BARS, 1, rng)
    idx = idx_mat[0]

    starts = _block_starts(idx, N_REAL)
    ends = np.concatenate([starts[1:], [N_REAL]])

    out_o = np.empty(N_REAL, dtype=float)
    out_h = np.empty(N_REAL, dtype=float)
    out_l = np.empty(N_REAL, dtype=float)
    out_c = np.empty(N_REAL, dtype=float)
    out_v = np.empty(N_REAL, dtype=float)

    running_level = float(_REAL_CLOSE[0])  # arbitrary starting price level
    for s, e in zip(starts, ends):
        block_idx = idx[s:e]
        bo = _REAL_OPEN[block_idx]
        bh = _REAL_HIGH[block_idx]
        bl = _REAL_LOW[block_idx]
        bc = _REAL_CLOSE[block_idx]
        bv = _REAL_VOLUME[block_idx]

        factor = running_level / bo[0]
        out_o[s:e] = bo * factor
        out_h[s:e] = bh * factor
        out_l[s:e] = bl * factor
        out_c[s:e] = bc * factor
        out_v[s:e] = bv

        running_level = float(bc[-1] * factor)

    new_index = pd.date_range("2017-01-01", periods=N_REAL, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": out_o, "high": out_h, "low": out_l, "close": out_c, "volume": out_v},
        index=new_index,
    )

    # --- sanity checks on our own output before it is ever scored ---
    assert len(df) == N_REAL, f"length mismatch: {len(df)} != {N_REAL}"
    assert np.all(np.isfinite(df.to_numpy())), "non-finite value in synthetic path"
    assert (df[["open", "high", "low", "close"]] > 0).to_numpy().all(), "non-positive price"
    assert (df["high"].to_numpy() >= np.maximum(df["open"].to_numpy(), df["close"].to_numpy()) - 1e-9).all(), \
        "high < max(open, close)"
    assert (df["low"].to_numpy() <= np.minimum(df["open"].to_numpy(), df["close"].to_numpy()) + 1e-9).all(), \
        "low > min(open, close)"

    return df


def path_generator(seed: int) -> pd.DataFrame:
    return build_synthetic_path(seed)


def main() -> None:
    sh.hr("R-118 CONSERVATIVE: stationary block bootstrap calibration/selection")

    print(f"inner_train_btc: n={N_REAL} bars, "
          f"{REAL_DF.index[0]} .. {REAL_DF.index[-1]}")
    print(f"mean_block = {sh.MEAN_BLOCK_DAYS} days = {MEAN_BLOCK_BARS:.0f} bars")
    print(f"grid: {sh.GRID} ({len(sh.GRID)} points)")

    # --- timing discipline: time a handful of real backtests before
    # committing to the full n_draws * len(grid) sweep ---
    probe_path = build_synthetic_path(0)
    n_blocks = len(_block_starts(
        stationary_bootstrap_indices(N_REAL, MEAN_BLOCK_BARS, 1,
                                     np.random.default_rng(0))[0], N_REAL))
    print(f"probe synthetic path built OK, {n_blocks} blocks "
          f"(mean block ~{N_REAL / n_blocks:.0f} bars observed)")

    t0 = time.time()
    for cfg in sh.GRID[:2]:
        sh.score_on_path(cfg, probe_path, sh.SPOT)
    t1 = time.time()
    per_backtest = (t1 - t0) / 2.0
    print(f"probe timing: {per_backtest:.2f}s/backtest")

    n_draws = sh.N_DRAWS
    est_total_s = per_backtest * len(sh.GRID) * n_draws
    print(f"estimated full sweep ({len(sh.GRID)} x {n_draws} = "
          f"{len(sh.GRID) * n_draws} backtests): {est_total_s / 60.0:.1f} min")

    if est_total_s > 75 * 60:
        # Scale down n_draws to keep the sweep under budget, and disclose it.
        n_draws = max(10, int(75 * 60 / (per_backtest * len(sh.GRID))))
        print(f"*** exceeds 75-minute budget at N_DRAWS={sh.N_DRAWS}; "
              f"reducing to N_DRAWS={n_draws} (disclosed) ***")
    else:
        print(f"within budget -- using pre-registered N_DRAWS={n_draws} unreduced")

    sh.hr(f"Running selection sweep: {len(sh.GRID)} configs x {n_draws} draws "
          f"= {len(sh.GRID) * n_draws} backtests")
    t_sweep0 = time.time()
    best_config, table = sh.select_config(path_generator, n_draws=n_draws,
                                          grid=sh.GRID, market=sh.SPOT)
    t_sweep1 = time.time()
    print(f"sweep wall time: {(t_sweep1 - t_sweep0) / 60.0:.1f} min")

    sh.hr("Selection table (mean / std / robust CVaR-25% Sharpe across synthetic draws)")
    for cfg in sh.GRID:
        row = table[cfg]
        marker = "  <== WINNER" if cfg == best_config else ""
        print(f"  base={cfg[0]:3d} target_vol={cfg[1]:.2f} max_lev={cfg[2]:.2f}  "
              f"mean={row['mean']:+.3f} std={row['std']:.3f} "
              f"robust={row['robust']:+.3f}{marker}")

    print(f"\nWinning config (max robust CVaR-25% Sharpe over {n_draws} synthetic draws): "
          f"{best_config}")
    print(f"v4 shipped default for comparison: {sh.V4_DEFAULT}")

    sh.hr("Step 4: frozen real-data evaluate_candidate (called exactly once)")
    result = sh.evaluate_candidate(best_config, "R118_conservative_bootstrap")
    sh.print_report(result)

    # --- save everything for reproducibility ---
    out_path = ROOT / "experiments" / "r118_conservative_results.json"
    payload = {
        "branch": "conservative_bootstrap",
        "n_draws_used": n_draws,
        "n_draws_preregistered": sh.N_DRAWS,
        "grid": [list(c) for c in sh.GRID],
        "mean_block_days": sh.MEAN_BLOCK_DAYS,
        "mean_block_bars": MEAN_BLOCK_BARS,
        "n_real_bars": N_REAL,
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
