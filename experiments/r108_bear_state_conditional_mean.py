"""R-108 (08-24) --- **B-40 sub-claim test.**

The sharpest sub-claim named in the B-40 backlog row, unmodified:

  *Is BTC's conditional mean return in the Bear state (80d<0 AND 7d<0)
  significantly negative on 2017--2022, and does the sign hold after?*

B-40 (filed by R-89, still OPEN) proposes Goulding, Harvey & Mazzoleni
(2023, *Journal of Financial Economics* 149, 378-406, "Momentum turning
points") as a candidate state-conditional sizing overlay. The row's own
guidance -- "Test the sharpest sub-claim first, before building anything"
-- is exactly the shape of this session: no strategy is built and no
existing strategy is touched; a single conditional-mean claim is tested
against BTC's own daily returns.

**Pre-registration, frozen before any number was computed:**

Data. The committed BTC 5m series
(`data/btcusd_spot_5m.csv.gz`, real Bitstamp BTC/USD, 2017-01-01 to
2026-08), aggregated to one daily close per UTC calendar day (the last
5m bar's close). Warmup: the 80 days needed for the slow signal are
consumed silently -- the first labeled day is 2017-03-22 (the 81st
calendar day of the series).

Signals, both computed at the close of day *D* from prices strictly at
or before *D* -- causal by construction.

- Slow signal at day *D*: `sign(close[D] / close[D-80] - 1)`.
- Fast signal at day *D*: `sign(close[D] / close[D-7]  - 1)`.

Signal horizons are the exact numbers named in B-40's own filing (80 and
7 calendar days), which are already load-bearing elsewhere in this
project's own codebase (v4's third anchor is 80 days). The row's citation
is Goulding-Harvey-Mazzoleni 2023, who use one-year (slow) and one-month
(fast) trailing returns; B-40 substitutes 80d/7d without further
justification and this round tests the row's exact prescription rather
than the paper's, per ROUTINE.md ("The backlog is the actual research
plan").

State at day *D*, following Goulding-Harvey-Mazzoleni's own four-state
partition:

  - Bull       = slow >= 0 AND fast >= 0
  - Correction = slow >= 0 AND fast <  0
  - Bear       = slow <  0 AND fast <  0
  - Rebound    = slow <  0 AND fast >= 0

Forward return, the object being conditioned on:
`r[D+1] = log(close[D+1] / close[D])`. Last day's state is dropped (no
forward return available). All returns are *log* returns, matching
`tradebot.inference`'s convention and this project's own bootstrap
machinery. No fees, no funding, no execution -- this is a pure
statistical test on price, not a strategy backtest.

**Primary decision rule (frozen before running):** On days 2017-01-01
through 2022-12-31 UTC (inner-train + inner-validation combined; the
holdout is not read), the Bear-state conditional mean of `r[D+1]`
`PASSES` if BOTH:

1. Point estimate is strictly negative.
2. Upper bound of a 95% stationary block-bootstrap interval on that
   conditional mean, computed by resampling day indices with a 30-day
   mean block (Politis & Romano 1994, `tradebot.inference.
   stationary_bootstrap_indices`), 2000 resamples, seed=7, is strictly
   less than zero.

The block bootstrap is over daily indices in the full 2017-2022 window
rather than only over the Bear-state subset. This preserves the observed
state-transition dynamics of the data: each resample includes whatever
Bear-state days that resample's blocks happen to cover, exactly as they
appear in real time. A subset-only bootstrap would break the state
structure that produced the label.

Guardrails (frozen): the Bear state must have >=30 labeled days in the
2017-2022 window (else the test is not attempted and the row is closed
as UNTESTABLE). Every day covered by the 2017-2022 window must have a
resolvable forward return (else the day is dropped and the drop is
reported).

**Secondary sign check (frozen, holdout, evaluated only if primary
passes):** On the holdout 2023-01-01 through last committed bar, does
the Bear-state conditional mean of `r[D+1]` remain strictly negative
(point estimate only, no significance requirement)? This is a
sign-check, not a promotion test.

**Configs evaluated: 1** (the single specification pre-registered above;
horizons 80 and 7 are named by B-40 and not swept).

**Falsification test, pre-registered:** the primary decision rule IS the
falsification test. B-40's own honest magnitude warning says to expect
the effect to be approximately zero; the CI-excludes-zero clause is what
turns "sign is negative" into a real result rather than one-in-two
noise. There is no separate `--windows` or fee-tier check to attempt
because there is no strategy to attempt them on.

**Not a duplicate of** any prior round. R-01 (HMM), R-82 (BOCPD), R-83
(Kalman LLT), R-85 (CSD), R-96 (Hawkes), R-98 (POT/GPD), R-99
(BV-jump), R-104-106 (ERR-axis discounts) all attempted to *detect
regime changes* against a shared six-episode gate. This round does not
detect anything; it labels each day with a deterministic function of two
observable trend signs and asks a plain conditional-mean question of the
returns. R-06/R-07/R-40/R-45 all *search* for a better anchor horizon
against a strategy's realized P&L; this uses fixed anchors named by the
backlog row and never touches a strategy.

**Holdout counter:** +0 if primary fails (the sign check is not spent);
+1 if primary passes (one sign-check consultation).

Reproduce with:

    python experiments/r108_bear_state_conditional_mean.py

Full output is deterministic (seed=7) and prints a max-timestamp line
for each phase for the ledger's holdout-consultation record.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tradebot.data import load_dataset  # noqa: E402
from tradebot.inference import Interval, stationary_bootstrap_indices  # noqa: E402


# ---------------------------------------------------------------- pre-registered constants
SLOW_DAYS = 80
FAST_DAYS = 7
BOOT_MEAN_BLOCK_DAYS = 30.0
BOOT_N = 2_000
BOOT_SEED = 7
BEAR_MIN_DAYS = 30
LEVEL = 0.95

TRAIN_START = "2017-01-01"
VAL_END = "2022-12-31"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
HOLDOUT_START = "2023-01-01"

BEAR = "Bear"
BULL = "Bull"
CORR = "Correction"
REBO = "Rebound"
STATES = (BULL, CORR, BEAR, REBO)


# ---------------------------------------------------------------- data


def load_daily_close(end: str | None = None) -> pd.Series:
    """UTC daily close (last 5m bar per calendar day), optionally truncated."""
    df, _label = load_dataset(ROOT / "data", "spot")
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta("23:59:59.999")]
    # Take last bar of each UTC calendar day.
    closes = df["close"].groupby(df.index.floor("D")).last()
    closes.index = closes.index.tz_convert("UTC")
    closes.name = "close"
    return closes


# ---------------------------------------------------------------- state classifier


def _classify(slow: pd.Series, fast: pd.Series) -> pd.Series:
    """Four-state label from the two trend-return signs."""
    slow_pos = slow >= 0.0
    fast_pos = fast >= 0.0
    out = pd.Series(index=slow.index, dtype=object)
    out[ slow_pos &  fast_pos] = BULL
    out[ slow_pos & ~fast_pos] = CORR
    out[~slow_pos & ~fast_pos] = BEAR
    out[~slow_pos &  fast_pos] = REBO
    out.name = "state"
    return out


def build_state_frame(close: pd.Series) -> pd.DataFrame:
    """Causal daily frame with columns (close, slow_ret, fast_ret, state, fwd_ret).

    All signals are strictly a function of prices at or before day D.
    fwd_ret is log(close[D+1]/close[D]); the last day is dropped.
    """
    close = close.astype(float).sort_index()
    slow_ret = close / close.shift(SLOW_DAYS) - 1.0
    fast_ret = close / close.shift(FAST_DAYS) - 1.0
    state = _classify(slow_ret, fast_ret)
    fwd_ret = np.log(close.shift(-1) / close)
    df = pd.DataFrame({
        "close": close,
        "slow_ret": slow_ret,
        "fast_ret": fast_ret,
        "state": state,
        "fwd_ret": fwd_ret,
    })
    df = df.dropna(subset=["state", "fwd_ret"])
    return df


# ---------------------------------------------------------------- causality probe


def causality_probe(close: pd.Series) -> None:
    """Sanity: truncating future data must not change past-day states/fwd_returns.

    Given the closed-form causal construction this is not expected to
    catch anything; keeping it in the script anyway because ROUTINE.md
    Step 3 says every experiment should carry its own no-lookahead
    check.
    """
    full = build_state_frame(close)
    truncated_close = close[close.index <= pd.Timestamp("2020-06-30", tz="UTC")]
    truncated = build_state_frame(truncated_close)
    shared = full.index.intersection(truncated.index)
    if len(shared) == 0:
        raise AssertionError("causality probe: no overlap")
    a = full.loc[shared]
    b = truncated.loc[shared]
    for col in ("slow_ret", "fast_ret"):
        diff = (a[col].astype(float) - b[col].astype(float)).abs().max()
        if not (diff < 1e-12 or np.isnan(diff)):
            raise AssertionError(
                f"causality probe: column {col!r} differs by max={diff}")
    disagreements = (a["state"].astype(str) != b["state"].astype(str)).sum()
    if disagreements:
        raise AssertionError(
            f"causality probe: {disagreements} state disagreements on shared index")
    # fwd_ret[D] is *always* forward-looking by 1 day: on the last shared
    # day the truncated frame will have dropped it (no D+1 to use), so we
    # allow that one row of shrinkage. Any *disagreement* on rows both
    # have is a bug.
    common = a.index.intersection(b.index)
    diff = (a.loc[common, "fwd_ret"] - b.loc[common, "fwd_ret"]).abs().max()
    if not (diff < 1e-12 or np.isnan(diff)):
        raise AssertionError(f"causality probe: fwd_ret differs by max={diff}")


# ---------------------------------------------------------------- statistic + bootstrap


def state_mean(df: pd.DataFrame, state: str) -> float:
    """Sample conditional mean of fwd_ret on days labeled `state`."""
    mask = df["state"].astype(str).values == state
    if not mask.any():
        return float("nan")
    return float(df["fwd_ret"].values[mask].mean())


def bootstrap_state_mean(df: pd.DataFrame, state: str, *,
                          mean_block: float = BOOT_MEAN_BLOCK_DAYS,
                          n_boot: int = BOOT_N,
                          seed: int = BOOT_SEED,
                          level: float = LEVEL) -> Interval:
    """Percentile CI for the conditional mean of fwd_ret in `state`.

    Bootstrap resamples day-indices in the full window with block
    structure preserved (30-day mean block, stationary bootstrap), then
    conditions each resample on the state label -- so both the state
    label counts and the returns co-vary the way they do in the data.
    """
    rets = df["fwd_ret"].to_numpy(dtype=float)
    is_state = (df["state"].astype(str).values == state)
    n = len(rets)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot,
                                       np.random.default_rng(seed))
    resampled_rets = rets[idx]           # (n_boot, n)
    resampled_flag = is_state[idx]       # (n_boot, n) bool
    with np.errstate(invalid="ignore"):
        num = np.where(resampled_flag, resampled_rets, 0.0).sum(axis=1)
        den = resampled_flag.sum(axis=1)
        draws = np.where(den > 0, num / den, np.nan)
    point = float(rets[is_state].mean()) if is_state.any() else float("nan")
    tail = (1.0 - level) / 2.0
    lo, hi = np.nanpercentile(draws, [100 * tail, 100 * (1 - tail)])
    return Interval(point, float(lo), float(hi), level)


# ---------------------------------------------------------------- reporting


@dataclass
class StateStats:
    state: str
    n_days: int
    share: float
    mean_fwd_bps: float
    interval_bps: Interval


def _fmt_ci(ci: Interval, scale: float = 1.0, unit: str = "") -> str:
    return f"{ci.point*scale:+.3f}{unit} [{ci.lo*scale:+.3f}, {ci.hi*scale:+.3f}]"


def summarize_states(df: pd.DataFrame, label: str) -> list[StateStats]:
    total = len(df)
    print(f"\n=== State summary --- {label} (n={total} days, "
          f"{df.index.min().date()} .. {df.index.max().date()}) ===")
    print(f"{'state':<12}{'n':>7}{'share':>8}{'mean fwd (bps/day)':>26}"
          f"{'95% CI [bps/day]':>28}")
    print("-" * 81)
    stats = []
    for state in STATES:
        mask = df["state"].astype(str).values == state
        n = int(mask.sum())
        share = n / total if total else 0.0
        if n == 0:
            print(f"{state:<12}{n:>7}{share:>8.1%}{'-':>26}{'-':>28}")
            stats.append(StateStats(state, n, share, float("nan"),
                                    Interval(float("nan"),
                                             float("nan"),
                                             float("nan"), LEVEL)))
            continue
        ci = bootstrap_state_mean(df, state)
        s = StateStats(state, n, share, ci.point * 1e4, ci)
        print(f"{state:<12}{n:>7}{share:>8.1%}"
              f"{ci.point*1e4:>+18.2f}         "
              f"[{ci.lo*1e4:>+7.2f}, {ci.hi*1e4:>+7.2f}]")
        stats.append(s)
    print(f"unconditional mean fwd_ret: "
          f"{df['fwd_ret'].mean()*1e4:+.3f} bps/day "
          f"(sd: {df['fwd_ret'].std()*1e4:.1f} bps/day)")
    return stats


# ---------------------------------------------------------------- decision rule


def primary_decision(bear: StateStats) -> tuple[bool, str]:
    """Frozen: PASS iff point<0 AND CI upper bound<0. Guardrail: >=30 days."""
    if bear.n_days < BEAR_MIN_DAYS:
        return False, (f"UNTESTABLE (Bear state has {bear.n_days} days, "
                       f"below the pre-registered floor of {BEAR_MIN_DAYS})")
    passes = bear.interval_bps.point < 0.0 and bear.interval_bps.hi < 0.0
    if passes:
        return True, (f"PRIMARY PASSES: Bear-state mean {bear.mean_fwd_bps:+.2f} bps/day, "
                      f"95% CI upper bound {bear.interval_bps.hi*1e4:+.2f} bps/day < 0")
    if bear.interval_bps.point >= 0.0:
        return False, (f"PRIMARY FAILS: Bear-state mean is {bear.mean_fwd_bps:+.2f} bps/day, "
                       f"not strictly negative")
    return False, (f"PRIMARY FAILS: Bear-state mean is {bear.mean_fwd_bps:+.2f} bps/day "
                   f"with CI [{bear.interval_bps.lo*1e4:+.2f}, "
                   f"{bear.interval_bps.hi*1e4:+.2f}] bps/day; CI upper bound "
                   f"{bear.interval_bps.hi*1e4:+.2f} bps/day is not strictly < 0")


# ---------------------------------------------------------------- run


def main() -> int:
    print(__doc__.split("\n\n", 1)[0])
    print()
    print(f"Pre-registered constants: SLOW_DAYS={SLOW_DAYS}, FAST_DAYS={FAST_DAYS}, "
          f"BOOT_MEAN_BLOCK_DAYS={BOOT_MEAN_BLOCK_DAYS}, BOOT_N={BOOT_N}, "
          f"BOOT_SEED={BOOT_SEED}, BEAR_MIN_DAYS={BEAR_MIN_DAYS}, LEVEL={LEVEL}")

    # ---- pre-holdout: load only 2017-01-01 through 2022-12-31.
    close_pre = load_daily_close(end=VAL_END)
    print(f"\nLoaded pre-holdout series: {len(close_pre)} daily closes, "
          f"max timestamp read: {close_pre.index.max()}")

    causality_probe(close_pre)
    print("Causality probe: PASS (truncation does not shift state or fwd_ret on shared days)")

    df_pre = build_state_frame(close_pre)
    df_pre = df_pre[df_pre.index >= pd.Timestamp(TRAIN_START, tz="UTC")]
    df_pre = df_pre[df_pre.index <= pd.Timestamp(VAL_END, tz="UTC") + pd.Timedelta("23:59:59.999")]

    # Diagnostic breakdowns.
    df_inner_train = df_pre[df_pre.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC") + pd.Timedelta("23:59:59.999")]
    df_inner_val   = df_pre[df_pre.index >= pd.Timestamp(INNER_VAL_START, tz="UTC")]

    summarize_states(df_inner_train, "inner-train 2017-2020 (diagnostic)")
    summarize_states(df_inner_val,   "inner-validation 2021-2022 (diagnostic)")
    primary_stats = summarize_states(df_pre, "PRIMARY 2017-2022 combined")

    bear = next(s for s in primary_stats if s.state == BEAR)
    passed, msg = primary_decision(bear)
    print()
    print(msg)

    if not passed:
        print("\n--- HOLDOUT NOT READ ---")
        print("Primary decision rule failed; per its own pre-registration the "
              "holdout sign-check is not spent. R-108 records +0 holdout "
              "consultations and B-40 closes NEGATIVE on its primary sub-claim.")
        return 0

    # ---- holdout: load 2023-01-01 through last committed bar.
    print("\n--- HOLDOUT READ (1 consultation, per pre-registration) ---")
    close_all = load_daily_close(end=None)
    print(f"Loaded full series: {len(close_all)} daily closes, "
          f"max timestamp read: {close_all.index.max()}")
    df_all = build_state_frame(close_all)
    df_hold = df_all[df_all.index >= pd.Timestamp(HOLDOUT_START, tz="UTC")]
    hold_stats = summarize_states(df_hold, "HOLDOUT 2023+ (secondary sign check)")
    bear_hold = next(s for s in hold_stats if s.state == BEAR)
    if np.isnan(bear_hold.mean_fwd_bps):
        print("\nHoldout Bear state has zero days; sign check UNRESOLVED.")
    elif bear_hold.mean_fwd_bps < 0:
        print(f"\nSECONDARY PASSES: holdout Bear-state mean "
              f"{bear_hold.mean_fwd_bps:+.2f} bps/day is negative "
              f"(sign held).")
    else:
        print(f"\nSECONDARY FAILS: holdout Bear-state mean "
              f"{bear_hold.mean_fwd_bps:+.2f} bps/day is non-negative "
              f"(sign did NOT hold).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
