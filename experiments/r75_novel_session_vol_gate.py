"""R-75 (novel branch): mandatory measurement gate for intraday session
volatility structure, before any strategy is built.

=====================================================================
WHAT THIS FILE IS
=====================================================================

Direction (see docs/LEDGER.md's standing diagnosis and the R-75 prompt):
calendar seasonality in BTC 5-minute returns, specifically the
hour-of-day / session structure of realized volatility -- never tried
in this project before (grep of docs/LEDGER.md and experiments/ turns up
zero prior "hour-of-day" / "intraday seasonality" / "session" signal
work). Citations: Shanaev, Vasenin & Stepanov (2023, Heliyon 9(3):e14236,
"Turn-of-the-candle effect in bitcoin returns"); Andersen & Bollerslev
(1997, J. Finance) for the classical U-shaped intraday volatility
pattern this generalizes from equities to crypto's 24/7 session handoff.

Mechanism (SIZE axis, sizing *responsiveness* not sizing *level* --
disjoint from R-62/B-27's closed level-only axis): kelly_regime_v4's
scale factor sizes off a backward-looking EWM realized-vol estimate
(vol_span = 8*BARS_PER_DAY) that necessarily lags the known, recurring
intraday liquidity cycle. A structural (not fitted) hour-of-day
multiplier on the vol estimate could pre-empt the low-liquidity
Asia-session vol trough / high-liquidity US-open vol spike rather than
reacting to it a full EWM-span late.

This file runs Step A of the R-75 brief: the mandatory measurement gate.
It does NOT build a strategy unless the gate clears. Everything here
reads training-period-only data (inner-train 2017-01-01 -> 2020-12-31
for the primary measurement; the round's OOS_START = 2023-01-01 is
never approached). ``assert_no_holdout`` is checked on every load.

Step A procedure (frozen before any number was computed):
  1. Load training-period BTC data only.
  2. Compute realized vol conditioned on UTC hour-of-day, inner-train only.
  3. Block-bootstrap null (shuffle hour labels in contiguous day-blocks,
     1000 reps) -- proceed only if observed dispersion clearly exceeds
     the null's 95th percentile.
  4. ETH replication on the identical statistic, training period only.
  5. Pre-registered stop rule: proceed to Step B only if (a) the
     bootstrap test clears its 95th-percentile bar AND (b) ETH's shape
     matches BTC's. Otherwise STOP -- report the gate as the round's
     complete product.

Run: ``python experiments/r75_novel_session_vol_gate.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset, load_coinbase_spot  # noqa: E402

OOS_START = "2023-01-01"
INNER_TRAIN_END = "2020-12-31 23:55:00"

RNG_SEED = 20260821  # today's date, fixed before any bootstrap number was seen
N_BOOT = 2000


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Second, independent guard alongside the explicit date slicing below:
    the max timestamp in any frame this file touches must be strictly
    before OOS_START. Mirrors experiments/r72_conservative_deadband.py."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# ============================================================ statistic

def hourly_vol_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute 5-minute log return, conditioned on UTC hour-of-day.

    Absolute log return is used rather than squared (equivalent ranking,
    less tail-dominated -- a design choice made before any number was
    computed, not selected after seeing which looked cleaner). Returned
    as a per-hour DataFrame with the statistic and the bar count backing
    it (uniform because 5m bars tile every hour identically).
    """
    r = np.log(df["close"]).diff().abs()
    hour = df.index.hour
    out = pd.DataFrame({"abs_ret": r.to_numpy(), "hour": hour})
    out = out.dropna()
    grp = out.groupby("hour")["abs_ret"].agg(["mean", "std", "count"])
    return grp.sort_index()


def dispersion_stat(per_hour_mean: np.ndarray) -> float:
    """max - min across the 24 hourly means, in units of the overall mean
    (a scale-free range statistic, chosen a priori)."""
    return float(per_hour_mean.max() - per_hour_mean.min())


def block_bootstrap_null(df: pd.DataFrame, n_boot: int = N_BOOT,
                          seed: int = RNG_SEED) -> np.ndarray:
    """Null distribution for the hour-of-day dispersion statistic.

    Breaks the alignment between the return series and the UTC clock
    while preserving each day's own internal 288-bar sequence as an
    intact block (so any short-range intraday autocorrelation survives
    into the null, unlike an i.i.d. bar shuffle): for each calendar day
    independently, circularly rotate that day's 288-bar return sequence
    by a random offset before reading off which "hour label" (fixed
    positions 0-11 = hour 0, 12-23 = hour 1, ...) each rotated value
    falls under. A day rotated by 0 reproduces the true hour assignment;
    any other rotation systematically misaligns that day's actual
    volatility-by-time-of-day pattern from the nominal hour bucket it
    lands in. With an independent random rotation per day (not a single
    rotation applied to the whole series, which would just relabel all
    24 hours by a constant offset and leave the dispersion statistic
    exactly unchanged), the true hour-of-day structure averages out
    across days while each day's internal block stays intact -- this is
    the "shuffle hour labels in contiguous day-blocks" construction
    named in the pre-registered procedure.

    (A first draft of this null instead permuted the *order* of day
    blocks while keeping canonical hour labels fixed per position; that
    is degenerate for a hour-grouped mean, because grouping-by-hour only
    ever averages the same 24 sets of values regardless of which day
    contributes which block, so its null distribution collapsed to a
    point mass at the observed statistic. Caught before any p-value was
    read, by checking that the null's std was implausibly close to 0.)
    """
    r = np.log(df["close"]).diff().to_numpy()
    valid = ~np.isnan(r)
    r = r[valid]
    ts = df.index[valid]
    abs_r = np.abs(r)

    bars_per_day = 288
    n = len(abs_r)
    n_days = n // bars_per_day
    abs_r = abs_r[: n_days * bars_per_day]
    hours = ts[: n_days * bars_per_day].hour.to_numpy()

    abs_by_day = abs_r.reshape(n_days, bars_per_day)
    hours_by_day = hours.reshape(n_days, bars_per_day)
    canonical = hours_by_day[0]
    assert np.array_equal(hours_by_day, np.tile(canonical, (n_days, 1))), (
        "every day-block must carry the canonical 0..23 hour ramp for the "
        "fixed-position hour labeling below to be valid")

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_boot)
    for b in range(n_boot):
        offsets = rng.integers(0, bars_per_day, size=n_days)
        rotated = np.empty_like(abs_by_day)
        for d in range(n_days):
            rotated[d] = np.roll(abs_by_day[d], offsets[d])
        s = pd.Series(rotated.reshape(-1)).groupby(
            np.tile(canonical, n_days)).mean().to_numpy()
        null_stats[b] = dispersion_stat(s)
    return null_stats


def session_bucket_stats(hourly: pd.Series) -> pd.DataFrame:
    """Aggregate the 24 hourly means into the four a-priori session
    buckets named in the R-75 brief (structural, not fitted):
      Asia-only    00:00-07:00 UTC
      Europe ovlp  07:00-12:00 UTC
      US overlap   12:00-21:00 UTC
      late-US      21:00-24:00 UTC
    """
    buckets = {
        "asia_only_00_07": list(range(0, 7)),
        "europe_overlap_07_12": list(range(7, 12)),
        "us_overlap_12_21": list(range(12, 21)),
        "late_us_21_24": list(range(21, 24)),
    }
    rows = []
    for name, hrs in buckets.items():
        rows.append({"bucket": name, "mean_abs_ret": hourly.loc[hrs].mean(),
                      "hours": f"{hrs[0]:02d}-{hrs[-1]+1:02d}"})
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 78)
    print("R-75 novel branch: Step A measurement gate")
    print("=" * 78)

    # ---- 1. load training-period BTC data only ----
    btc_full, btc_label = load_dataset(ROOT / "data", "spot")
    btc_train = btc_full.loc[:INNER_TRAIN_END]
    assert_no_holdout(btc_train)
    assert btc_train.index.min() < pd.Timestamp("2017-01-02", tz="UTC")
    assert btc_train.index.max() < pd.Timestamp(OOS_START, tz="UTC")
    print(f"\nBTC inner-train: {btc_train.index.min()} -> {btc_train.index.max()} "
          f"({len(btc_train):,} bars, label={btc_label})")

    # ---- 2. hourly vol conditioned on UTC hour, inner-train only ----
    btc_hourly = hourly_vol_stats(btc_train)
    print("\nBTC mean |log-return| by UTC hour (inner-train 2017-2020):")
    print(btc_hourly.to_string(float_format=lambda x: f"{x:.6e}"))

    overall_mean = btc_hourly["mean"].mean()
    btc_disp = dispersion_stat(btc_hourly["mean"].to_numpy())
    btc_disp_pct = btc_disp / overall_mean * 100
    lo_hour = btc_hourly["mean"].idxmin()
    hi_hour = btc_hourly["mean"].idxmax()
    print(f"\nBTC dispersion statistic (max-min hourly mean |ret|): {btc_disp:.6e} "
          f"({btc_disp_pct:.1f}% of overall mean)")
    print(f"BTC trough hour (UTC): {lo_hour:02d}:00  peak hour (UTC): {hi_hour:02d}:00")

    btc_buckets = session_bucket_stats(btc_hourly["mean"])
    print("\nBTC session-bucket means (a priori buckets from the brief):")
    print(btc_buckets.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    # ---- 3. block-bootstrap null ----
    print(f"\nRunning block-bootstrap null ({N_BOOT} reps, seed={RNG_SEED})...")
    null = block_bootstrap_null(btc_train, n_boot=N_BOOT, seed=RNG_SEED)
    p95 = np.percentile(null, 95)
    p99 = np.percentile(null, 99)
    pval = float((null >= btc_disp).mean())
    print(f"Null dispersion: mean={null.mean():.6e} std={null.std():.6e} "
          f"p95={p95:.6e} p99={p99:.6e}")
    print(f"Observed BTC dispersion: {btc_disp:.6e}  "
          f"-> {'EXCEEDS' if btc_disp > p95 else 'does NOT exceed'} p95, "
          f"empirical p-value={pval:.4f}")
    bootstrap_pass = btc_disp > p95

    # ---- 4. ETH replication, training period only ----
    eth_full = load_coinbase_spot(ROOT / "data", "ETH")
    eth_pass = False
    eth_lo_hour = eth_hi_hour = None
    eth_disp = eth_disp_pct = None
    if eth_full is None:
        print("\nETH Coinbase spot file not found -- cannot replicate. Gate FAILS by default.")
    else:
        eth_train = eth_full.loc[:INNER_TRAIN_END]
        assert_no_holdout(eth_train)
        assert eth_train.index.max() < pd.Timestamp(OOS_START, tz="UTC")
        print(f"\nETH inner-train: {eth_train.index.min()} -> {eth_train.index.max()} "
              f"({len(eth_train):,} bars)")
        eth_hourly = hourly_vol_stats(eth_train)
        print("\nETH mean |log-return| by UTC hour (inner-train, ETH's own coverage):")
        print(eth_hourly.to_string(float_format=lambda x: f"{x:.6e}"))

        eth_overall_mean = eth_hourly["mean"].mean()
        eth_disp = dispersion_stat(eth_hourly["mean"].to_numpy())
        eth_disp_pct = eth_disp / eth_overall_mean * 100
        eth_lo_hour = eth_hourly["mean"].idxmin()
        eth_hi_hour = eth_hourly["mean"].idxmax()
        print(f"\nETH dispersion statistic: {eth_disp:.6e} ({eth_disp_pct:.1f}% of overall mean)")
        print(f"ETH trough hour (UTC): {eth_lo_hour:02d}:00  peak hour (UTC): {eth_hi_hour:02d}:00")

        eth_buckets = session_bucket_stats(eth_hourly["mean"])
        print("\nETH session-bucket means:")
        print(eth_buckets.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

        # replication criterion, pre-registered: same low/high hour bucket
        # (within +/-2h tolerance for adjacent-hour noise) and same-signed,
        # comparable-magnitude bucket ordering (Spearman rank correlation
        # of the four bucket means between BTC and ETH).
        lo_match = abs(int(lo_hour) - int(eth_lo_hour)) <= 2 or \
            abs(int(lo_hour) - int(eth_lo_hour)) >= 22  # wraparound (23 vs 0)
        hi_match = abs(int(hi_hour) - int(eth_hi_hour)) <= 2 or \
            abs(int(hi_hour) - int(eth_hi_hour)) >= 22
        rank_corr = np.corrcoef(
            btc_buckets["mean_abs_ret"].rank().to_numpy(),
            eth_buckets["mean_abs_ret"].rank().to_numpy(),
        )[0, 1]
        print(f"\nReplication check: trough within +/-2h = {lo_match}, "
              f"peak within +/-2h = {hi_match}, "
              f"session-bucket rank correlation (BTC vs ETH) = {rank_corr:.3f}")
        eth_pass = lo_match and hi_match and rank_corr > 0.5

    # ---- 5. pre-registered stop rule ----
    gate_pass = bootstrap_pass and eth_pass
    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (frozen before either number above was computed):")
    print("  proceed to Step B only if (a) bootstrap dispersion > null p95 AND")
    print("  (b) ETH's session shape matches BTC's (trough/peak within +/-2h,")
    print("  bucket rank correlation > 0.5).")
    print(f"  (a) bootstrap: {'PASS' if bootstrap_pass else 'FAIL'}")
    print(f"  (b) ETH replication: {'PASS' if eth_pass else 'FAIL'}")
    print(f"  GATE: {'PASS -> proceed to Step B' if gate_pass else 'FAIL -> STOP, report negative'}")
    print("=" * 78)

    # persist a small summary for the report / a skeptic to re-check
    summary = {
        "btc_dispersion": btc_disp,
        "btc_dispersion_pct_of_mean": btc_disp_pct,
        "btc_trough_hour_utc": int(lo_hour),
        "btc_peak_hour_utc": int(hi_hour),
        "null_p95": p95,
        "null_p99": p99,
        "empirical_pvalue": pval,
        "bootstrap_pass": bool(bootstrap_pass),
        "eth_dispersion": eth_disp,
        "eth_dispersion_pct_of_mean": eth_disp_pct,
        "eth_trough_hour_utc": None if eth_lo_hour is None else int(eth_lo_hour),
        "eth_peak_hour_utc": None if eth_hi_hour is None else int(eth_hi_hour),
        "eth_pass": bool(eth_pass),
        "gate_pass": bool(gate_pass),
    }
    out_path = ROOT / "experiments" / "reports" / "r75_novel_gate_summary.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.Series(summary).to_csv(out_path, header=["value"])
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
