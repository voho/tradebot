#!/usr/bin/env python
"""R-75 CONSERVATIVE branch: UTC day-of-week seasonality in BTC 5-minute
returns -- Step A measurement gate.

=====================================================================
WHAT THIS FILE IS
=====================================================================

This project has tried five INFO signals -- on-chain activity (B-07/R-44),
macro spillover VIX/DXY (R-53/R-54), stablecoin flow (R-54/R-55/R-58),
implied vol DVOL/VRP (R-73), MVRV valuation (R-74) -- and all five failed
(section C). Every one of those was an *external* data source. This
direction is different: it extracts information already latent in the
timestamp of every existing 5-minute bar, no new data fetch, no
coverage/staleness risk.

**Citation.** Aharon & Qadan (2019), "Bitcoin and the day-of-the-week
effect", Applied Economics Letters; Kaiser (2019), "Seasonality in
cryptocurrency returns"; Vojtko & Javorska (2023/2024, SSRN 4581124),
"The Seasonality of Bitcoin" -- the general finding that crypto (unlike
equities) trades 24/7 but still shows statistically real day-of-week and
hour-of-day return patterns, plausibly from weekend/holiday liquidity and
retail-flow gaps rather than any institutional open/close.

**Mechanism, one sentence.** Weekend and Monday liquidity gaps (thinner
order books, more retail-dominated flow, no institutional desks) create a
mechanical day-of-week return/volatility pattern that is orthogonal to
kelly_regime_v4's own 20/40/80-day trend vote (a slow, price-level
signal), so a calendar-derived confirming vote could add independent
information without duplicating what the trend anchors already see.

Constraint attacked: INFO (one price series) -- the "one price series"
premise is technically preserved (this is not a new *data source*), but
the constraint is attacked the same way relative timing/deadband work
attacks COST: turning an unused feature of the existing bars (the
timestamp) into new information the strategy does not already consume.
Not a duplicate of any of R-44/R-53/R-54/R-55/R-58/R-73/R-74 -- none of
those used the bar timestamp itself. Grepped `docs/LEDGER.md` and
`docs/RESEARCH.md` for "day-of-week" / "seasonal" / "seasonality" before
starting: zero hits.

Not a duplicate of this round's parallel NOVEL branch, which is an
hour-of-day construction on a different mechanism (intraday liquidity
cycle, not weekly) -- disjoint files, not read, not coordinated with, per
ROUTINE.md's parallelism rules. This branch stays day-of-week only.

=====================================================================
STEP A -- THE MANDATORY MEASUREMENT GATE (this project's own established
discipline for calendar/timing signals -- R-53/R-54/R-73/R-74's novel
branches stopped here when the gate failed, and that WAS the round's
product)
=====================================================================

1. Mean 5-minute log return and realized volatility, conditioned on UTC
   day-of-week (0=Monday..6=Sunday), on the inner-train period
   (2017-01-01 -> 2020-12-31) only.
2. Dispersion statistic: population standard deviation of the 7
   conditional-mean values (a range is also reported for readability).
3. Null: block-bootstrap. BTC 5m bars carry strong short-horizon
   autocorrelation, so shuffling individual bars would inflate the
   dispersion statistic's apparent significance by breaking that
   structure. Instead: partition the inner-train period into contiguous
   7-day (one-week) blocks; for each block, draw one random rotation
   r ~ Uniform{0..6} and add it (mod 7) to every bar's true day-of-week
   label in that block. This preserves each block's internal sequence
   and autocorrelation intact (the bars keep their real order and
   neighbours) while destroying the *alignment* between a bar's assigned
   label and which real calendar day it fell on -- exactly the
   ROUTINE-mandated "shuffle day-of-week labels in contiguous blocks"
   test. Recomputed 2,000 times; proceed only if the observed dispersion
   clearly exceeds the null's 95th percentile.
4. ETH replication: identical statistic, independently, on ETH's own
   price series (`ethusd_coinbase_spot_5m.csv.gz`, its own available
   history, truncated to the training period, i.e. 2019-03-14 ->
   2022-12-31 -- ETH has no earlier Coinbase coverage). The pattern must
   replicate: same worst day(s), comparable sign/magnitude. Standard
   falsification device since R-47/B-08.

**Pre-registered stop rule (fixed before either number below was
computed):** proceed to Step B (build the confirming-vote strategy) only
if BOTH (a) the block-bootstrap dispersion test clears its 95th-percentile
bar on BTC, AND (b) ETH's worst day matches BTC's worst day. If either
fails, STOP -- do not build a strategy. Report the measurement as a clean
negative with the gate itself as the product, exactly as R-73/R-74's
novel branches did.

=====================================================================
DATA DISCIPLINE
=====================================================================

BTC (`data/btcusd_spot_5m.csv.gz` via `tradebot.data.load_dataset`) and
ETH (`data/ethusd_coinbase_spot_5m.csv.gz` via `load_ohlcv_csv`) are both
truncated to strictly before `OOS_START = "2023-01-01"` immediately on
load. `assert_no_holdout(df)` re-checks the max timestamp after every
load as an independent second guard, same pattern as
`experiments/r72_conservative_deadband.py` /
`experiments/r74_conservative_mvrv_level.py`. No bar dated 2023-01-01 or
later is ever read, held, or printed by this file.

Usage::

    python experiments/r75_conservative_dow_signal.py btc     # step A.1-3
    python experiments/r75_conservative_dow_signal.py eth     # step A.4
    python experiments/r75_conservative_dow_signal.py all     # both + verdict
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402

DATA_DIR = ROOT / "data"
OOS_START = "2023-01-01"          # NEVER read in this file
TRAIN = ("2017-01-01", "2020-12-31")  # inner-train, per ROUTINE.md step 3

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

N_BOOT = 2000
BOOT_SEED = 75  # this round's number, fixed once, never re-drawn


# ---------------------------------------------------------------- holdout guard


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before OOS_START. Independent of any truncation already done
    at load time."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# --------------------------------------------------------------------- data


def load_btc_train() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    train = df.loc[TRAIN[0]:TRAIN[1]]
    print(f"BTC ({label}) inner-train: {len(train):,} bars  "
          f"{train.index[0]} -> {train.index[-1]}", file=sys.stderr)
    return train


def load_eth_pre_oos() -> pd.DataFrame:
    """ETH's own available history, training period only. Coinbase ETH-USD
    starts 2019-03-14 -- no earlier coverage exists, so this is ETH's own
    full pre-holdout history, not an artificially shortened window."""
    df = load_ohlcv_csv(DATA_DIR / "ethusd_coinbase_spot_5m.csv.gz")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"ETH (coinbase) pre-OOS history: {len(df):,} bars  "
          f"{df.index[0]} -> {df.index[-1]}", file=sys.stderr)
    return df


# ------------------------------------------------------------- the statistic


def dow_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-UTC-day-of-week mean and std of 5-minute log returns."""
    r = np.log(df["close"]).diff().dropna()
    dow = r.index.dayofweek
    g = pd.Series(r.to_numpy(), index=r.index).groupby(dow)
    out = pd.DataFrame({
        "mean_bps": g.mean() * 1e4,
        "std_bps": g.std() * 1e4,
        "n": g.count(),
    })
    out.index = [DAY_NAMES[i] for i in out.index]
    return out


def dispersion_stat(means: np.ndarray) -> float:
    """Population std of the 7 conditional-mean values."""
    return float(np.std(means, ddof=0))


def block_bootstrap_null(r: pd.Series, n_boot: int = N_BOOT,
                          seed: int = BOOT_SEED) -> np.ndarray:
    """Contiguous-block (one calendar week = 7 days) day-of-week-label
    rotation null. Each block keeps its real bar order and internal
    autocorrelation; only the block's alignment to a specific day-of-week
    label is randomized (a uniform circular rotation of 0..6, drawn once
    per block per bootstrap iteration)."""
    idx = r.index
    dow = idx.dayofweek.to_numpy()
    t0 = idx[0]
    block_id = ((idx - t0).total_seconds() // (7 * 86400)).astype(int).to_numpy()
    n_blocks = int(block_id.max()) + 1
    r_arr = r.to_numpy()

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_boot)
    for b in range(n_boot):
        shifts = rng.integers(0, 7, size=n_blocks)
        shifted_dow = (dow + shifts[block_id]) % 7
        means = pd.Series(r_arr).groupby(shifted_dow).mean().to_numpy()
        null_stats[b] = dispersion_stat(means)
    return null_stats


# --------------------------------------------------------------- step A.1-3: BTC


def measure_btc() -> dict:
    train = load_btc_train()
    stats = dow_stats(train)
    print("\nBTC inner-train (2017-01-01 -> 2020-12-31), per-UTC-day-of-week:")
    print(stats.to_string(float_format=lambda v: f"{v:.4f}"))

    means = stats["mean_bps"].to_numpy() / 1e4  # back to raw log-return units
    obs_stat = dispersion_stat(means)
    obs_range = float(means.max() - means.min())
    worst_day = stats["mean_bps"].idxmin()
    best_day = stats["mean_bps"].idxmax()
    print(f"\nobserved dispersion (pop. std of 7 DOW means): {obs_stat:.6e}")
    print(f"observed range (max-min): {obs_range:.6e}")
    print(f"worst day (lowest mean return): {worst_day}   "
          f"best day (highest mean return): {best_day}")

    r = np.log(train["close"]).diff().dropna()
    null_stats = block_bootstrap_null(r)
    p95 = float(np.percentile(null_stats, 95))
    p99 = float(np.percentile(null_stats, 99))
    pval = float((null_stats >= obs_stat).mean())
    clears = obs_stat > p95
    print(f"\nblock-bootstrap null ({N_BOOT} iterations, whole-week blocks, "
          f"seed={BOOT_SEED}):")
    print(f"  null mean={null_stats.mean():.6e}  p95={p95:.6e}  p99={p99:.6e}")
    print(f"  observed={obs_stat:.6e}  ->  "
          f"{'CLEARS the 95th percentile' if clears else 'DOES NOT CLEAR the 95th percentile'}")
    print(f"  empirical one-sided p-value (null >= observed): {pval:.4f}")

    return dict(stats=stats, worst_day=worst_day, best_day=best_day,
                obs_stat=obs_stat, p95=p95, pval=pval, clears=clears)


# ------------------------------------------------------------- step A.4: ETH


def measure_eth() -> dict:
    eth = load_eth_pre_oos()
    stats = dow_stats(eth)
    print("\nETH pre-OOS history (own history, 2019-03-14 -> 2022-12-31), "
          "per-UTC-day-of-week:")
    print(stats.to_string(float_format=lambda v: f"{v:.4f}"))

    means = stats["mean_bps"].to_numpy() / 1e4
    obs_stat = dispersion_stat(means)
    worst_day = stats["mean_bps"].idxmin()
    best_day = stats["mean_bps"].idxmax()
    print(f"\nETH observed dispersion: {obs_stat:.6e}")
    print(f"ETH worst day: {worst_day}   ETH best day: {best_day}")

    r = np.log(eth["close"]).diff().dropna()
    null_stats = block_bootstrap_null(r)
    p95 = float(np.percentile(null_stats, 95))
    pval = float((null_stats >= obs_stat).mean())
    clears = obs_stat > p95
    print(f"ETH block-bootstrap null: p95={p95:.6e}  observed={obs_stat:.6e}  "
          f"-> {'CLEARS' if clears else 'DOES NOT CLEAR'}  p={pval:.4f}")

    return dict(stats=stats, worst_day=worst_day, best_day=best_day,
                obs_stat=obs_stat, p95=p95, pval=pval, clears=clears)


# --------------------------------------------------------------------- verdict


def all_checks() -> None:
    print("=" * 78)
    print("R-75 CONSERVATIVE: day-of-week seasonality -- STEP A measurement gate")
    print("=" * 78)
    btc = measure_btc()
    print("\n" + "=" * 78)
    eth = measure_eth()

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (fixed before either number above was computed):")
    print("  proceed to Step B only if BOTH:")
    print("    (a) BTC block-bootstrap dispersion clears its 95th percentile, AND")
    print("    (b) ETH's worst day matches BTC's worst day.")
    print("  otherwise: STOP, report as a clean negative, no strategy built.")
    print("=" * 78)

    a_pass = btc["clears"]
    b_pass = btc["worst_day"] == eth["worst_day"]
    print(f"\n(a) BTC dispersion clears 95th percentile: {a_pass}  "
          f"(observed={btc['obs_stat']:.6e}, p95={btc['p95']:.6e}, "
          f"p-value={btc['pval']:.4f})")
    print(f"(b) BTC worst day ({btc['worst_day']}) == ETH worst day "
          f"({eth['worst_day']}): {b_pass}")

    gate = a_pass and b_pass
    print(f"\nGATE VERDICT: {'PASS -> proceed to Step B' if gate else 'FAIL -> STOP, no strategy built'}")
    print(f"max timestamp read anywhere in this session: "
          f"BTC {load_btc_train().index.max()}, "
          f"ETH {load_eth_pre_oos().index.max()}  (both < {OOS_START})")


if __name__ == "__main__":
    cmds = {"btc": measure_btc, "eth": measure_eth, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r75_conservative_dow_signal.py [{'|'.join(cmds)}]")
