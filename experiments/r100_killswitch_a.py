"""Kill Switch A (degeneracy check) for R-100, run once by the operator
before any per-episode number -- same posture as R-96/97/98/99's own Kill
Switch A. Does `cross_venue_divergence_z` actually cross `Z_THRESH=1.5` at
least once across the full pre-holdout history, for each of the 3
`BASELINE_WINDOW_DAYS_GRID` cells, before any cell is chosen PRIMARY?
"""
from __future__ import annotations

import r100_shared as R


def main() -> None:
    daily = R.load_daily_funding_totals("data")
    R.assert_no_holdout(daily)
    print(f"daily funding rows: {len(daily)}  "
          f"binance non-null: {daily['binance'].notna().sum()}  "
          f"deribit non-null: {daily['deribit'].notna().sum()}")
    print(f"date range: {daily.index.min()} -> {daily.index.max()}")

    for w in R.BASELINE_WINDOW_DAYS_GRID:
        z = R.cross_venue_divergence_z(daily, baseline_window_days=w)
        R.assert_no_holdout(z.dropna())
        n_valid = z.notna().sum()
        max_abs_z = z.abs().max()
        n_cross = (z.abs() >= R.Z_THRESH).sum()
        print(f"baseline={w:3d}d  n_valid={n_valid:5d}  max|z|={max_abs_z:.2f}  "
              f"bars_over_thresh={n_cross:4d}  "
              f"{'DEGENERATE (never crosses)' if n_cross == 0 else 'fires'}")

    print()
    print("Per-episode raw-data availability check (does either venue have")
    print("*any* observation inside the 60-day pre-onset baseline window?):")
    for name, onset_str in R.STRESS_EPISODES:
        onset = __import__("pandas").Timestamp(onset_str, tz="UTC")
        lo = onset - __import__("pandas").Timedelta(days=60)
        pre = daily.loc[(daily.index >= lo) & (daily.index < onset)]
        n_b = pre["binance"].notna().sum()
        n_d = pre["deribit"].notna().sum()
        flag = ""
        if name in R.FORCED_FAIL_EPISODES:
            flag = " [FORCED FAIL: predates funding data]"
        elif name in R.THIN_BASELINE_EPISODES:
            flag = " [THIN BASELINE, disclosed]"
        print(f"  {name:45s} binance_obs={n_b:3d} deribit_obs={n_d:3d}{flag}")


if __name__ == "__main__":
    main()
