#!/usr/bin/env python
"""R-79 CONSERVATIVE branch: Bitcoin halving-cycle phase as a new INFO
signal on kelly_regime_v4 -- Step A measurement gate.

=====================================================================
WHAT THIS FILE IS
=====================================================================

This project has tried six external/derived INFO signals -- on-chain
activity (B-07/R-44), macro VIX/DXY (R-53/R-54), stablecoin flow
(R-54/R-55/R-58), Deribit DVOL/VRP (R-73), MVRV valuation (R-74),
day-of-week/session timing (R-75) -- and all six failed (docs/LEDGER.md
section C). None used Bitcoin's block-reward halving schedule: grepped
docs/LEDGER.md for "halving" / "stock-to-flow" / "cycle phase" before
starting -- the only hit is R-78's unrelated use of "halve"/"halving" in
a sentence about rebalancing frequency (line ~1456, a different subject
entirely), zero hits for the halving-cycle direction itself. Halving
dates are public, deterministic, and known years in advance -- zero
external data fetch, zero coverage/staleness risk, the same class of
"unused feature of the existing timestamp" as R-75's day-of-week/hour-
of-day work, but keyed to the ~4-year protocol-mandated supply-shock
cycle instead of a weekly/intraday liquidity cycle.

Constraint attacked: INFO. Not a duplicate of R-44/R-53/R-54/R-55/R-58/
R-73/R-74/R-75/R-76/R-77 -- none of those used the halving schedule; R-75
used the bar's timestamp too, but at day-of-week/hour-of-day granularity,
a structurally different (and much shorter) periodicity than the
multi-year halving cycle. R-76 (pairs trading) is a cross-instrument
relationship, not a calendar signal, and is unrelated.

**Citations.**
- Gatsios, A. et al. (2025), "Is Bitcoin's Market Maturing? Cumulative
  Abnormal Returns and Volatility in the 2024 Halving and Past Cycles,"
  Journal of Risk and Financial Management 18(5):242 -- finds the
  returns/volatility response to halvings has been SHRINKING across the
  2012->2024 cycles (a market-maturation reading: cycle 1 was the
  biggest reaction, cycle 4 was muted).
- Lim, B.C. (2026), "Issuance Shocks in Mature Crypto Markets: An
  Event-Study Analysis of the Bitcoin Halving and the Financialisation
  of BTC in Cycle Four," SSRN 6589402 -- THE decisive citation for this
  round: a structural break, not a gradual fade. Cycle 4's [-30,+90]-day
  cumulative abnormal return is -41.5%, the OPPOSITE SIGN of cycle 3's
  +172.2% over the identical event window. This is direct evidence, from
  outside this project entirely, that whatever phase-return pattern
  existed across cycles 1-3 already inverted sign on cycle 4 -- an
  out-of-sample replication failure that pre-exists anything computed
  below.
- PlanB (2019) stock-to-flow model -- the popular "halving -> scarcity ->
  higher price" narrative. NOT what this round tests: this is the naive
  price-LEVEL mechanism, separately well documented as broken since the
  2022 bear (price fell far below and stayed below the model's own band
  from June 2022 onward). This round tests a weaker, more general claim
  -- phase-conditioned RETURN/vol DISPERSION, not a price-level target.
- Fama (1970) EMH, general: halving dates are public knowledge years in
  advance. A rational-expectations market should price the known supply
  shock in before it happens. Under EMH there should be NO exploitable
  phase-conditioned drift at all -- this is this round's own null
  hypothesis's economic justification, not just the statistical
  formality of a bootstrap p-value.

**What would make it fail, named before any number below was computed:**
(a) EMH pre-pricing -- no real phase-conditioned dispersion to find, i.e.
    the true-halving dispersion does not clear a placebo-offset null;
(b) even if the one fully-observed pre-holdout cycle (cycle 3,
    2016-07-09 -> 2020-05-11) shows a pattern, Lim (2026) already shows
    such patterns have inverted sign cycle-to-cycle in the real, longer
    historical record -- so an in-sample pattern here is a weak prior
    for out-of-sample persistence, tested directly below via cross-cycle
    rank correlation rather than taken on faith from the citation alone;
(c) any monotonic-ish partition of a multi-year TRENDING price series
    will show SOME apparent phase dispersion by pure confound with the
    generic multi-year trend (BTC went up in 2017, down in 2018, up in
    2020) -- this is why the null below is a placebo-OFFSET null (same
    inter-event spacing statistics as the real halving schedule, but an
    arbitrary phase origin) rather than a naive within-series shuffle,
    which would not control for this confound at all.

**Power limitation, stated per ROUTINE.md step 2's discipline (compute
the n a claim needs before trusting it), quoting `r79_shared.py`'s own
docstring:** at most 1 fully-observed cycle (cycle 3, tail of it: months
~6-45 of a ~46-month cycle, inside inner-train) plus one badly truncated
partial second cycle (cycle 4's own months ~8-31 of ~47, inside inner-
validation only -- neither its first 8 months nor its last ~16 are ever
observed pre-holdout). That is worse than DVOL's n=3 stress episodes
(R-73) and worse than R-74's 5-episode MVRV lead-time table -- this round
has AT MOST a 2-cycle comparison for its cross-cycle replication check,
and even that 2nd cycle is missing one whole phase bucket (see Step A.3
below). A dispersion or rank-correlation finding built on n=2 cycles is
not statistically powerful evidence of a *repeating* phenomenon under any
correction; it can only be a screening gate, which is exactly the role
Step A gives it (a stop rule, not a promotion criterion).

=====================================================================
STEP A -- THE MANDATORY MEASUREMENT GATE (this project's established
discipline for calendar/timing signals -- R-53/R-54/R-73/R-74/R-75's
novel branches stopped here when the gate failed, and that WAS the
round's product)
=====================================================================

A.1-2. Mean 5-minute log return conditioned on `phase_bucket(index,
       n_buckets=4)` (quartiles of the TRUE halving cycle, from
       `r79_shared.py`, read-only), inner-train only (2017-01-01 ->
       2020-12-31). Dispersion statistic = population std of the 4
       conditional means (same shape as R-75's day-of-week statistic).

A.2.   PLACEBO-OFFSET null (the critical, non-generic part -- NOT a
       naive block-bootstrap of bar order, which would not control for
       the trend-confound named above as risk (c)). Draw >=500 (this
       file uses 1,000) placebo reference dates via
       `r79_shared.placebo_reference_dates(n=1000, seed=79)`; for each,
       compute the SAME dispersion statistic using
       `r79_shared.fake_cycle_phase()` bracketed off that placebo date
       instead of the real halvings (same fixed inter-event spacing,
       1406-day mean gap, as the real schedule's 3rd/4th intervals -- the
       module's own default). The true-halving dispersion must clear the
       95th percentile of this placebo distribution -- i.e. slicing BTC's
       inner-train history by the TRUE halving date must show MORE
       phase-dispersion than slicing it by an arbitrary same-spaced fake
       date, or the "pattern" is just generic multi-year trendiness and
       the signal is void by construction.

       Implementation note, disclosed, WITH A MID-RUN DEVIATION: the
       statistic is computed at DAY granularity (one phase value per
       unique calendar day in the window, ~1,461 days for 4 years) then
       broadcast back to bars by calendar day -- a ~288x reduction in
       element count vs. computing at 5-minute-bar granularity
       (~420,481 bars). The approximation error this introduces is
       bounded and negligible: phase advances by only ~1/1406 = 0.00071
       per calendar day, so within any single day every bar's phase
       differs by at most that amount, and a bucket-boundary
       misassignment (a day whose bars actually straddle one of the 4
       quartile edges) can occur on at most ~4 days per ~1,406-day cycle
       -- a rounding error, not a construction change, applying
       identically to every placebo draw, so it cannot bias the
       comparison between the real statistic (computed at true bar
       granularity, unapproximated) and the null distribution in either
       direction.

       First draft's mistake, and the fix (disclosed as a genuine
       deviation from the pre-registered plan's *implementation*, not
       its *design*): the first version still called
       `r79_shared.fake_cycle_phase()` -- which loops per-timestamp in
       pure Python internally -- ONCE PER PLACEBO DRAW even at day
       granularity, and additionally recomputed
       `sorted(set(idx.normalize()))` (a Python-level set/sort over the
       full 420,767-bar index) on every draw, although neither the day
       list nor the bar->day mapping actually changes between draws for
       a fixed measurement window. Measured: ~0.9s/draw, ~15 minutes
       projected for 1,000 draws; the coordinator flagged the process
       stalled at ~7 minutes (the parallel novel branch hit the
       identical bottleneck independently) and it was killed at draw
       400/1000. Fixed by (a) hoisting the day list and the bar->day
       position lookup out of the placebo loop -- computed ONCE, reused
       for all 1,000 draws -- and (b) replacing the repeated call into
       `fake_cycle_phase`'s internal per-element Python loop with an
       independent, numpy-vectorized reimplementation of the identical
       public formula (`_fake_phase_days_vectorized` in this file: `n =
       floor((ts-anchor)/gap); phase = (ts-anchor)/gap - n`, i.e. the
       same `fake_bracket_fn` arithmetic `r79_shared.py` itself uses,
       just without the per-element Python loop). `r79_shared.py` is NOT
       modified anywhere by this file -- the reimplementation is
       independently verified bit-close (max|diff| < 1e-9) against the
       real, unmodified `r79_shared.fake_cycle_phase` on a held-out spot
       sample of anchors (`_verify_vectorized_fake_phase`, asserted
       before the fast path is trusted for the full run) -- the same
       fix, independently applied, the coordinator reported the parallel
       novel branch used for the identical bottleneck. Runtime after the
       fix: the full 1,000-draw placebo null completes in well under a
       minute.

A.3.   CROSS-CYCLE REPLICATION (the only falsification device available
       here -- ETH does NOT have a comparable halving schedule, it moved
       to proof-of-stake in Sept 2022, so this project's usual ETH-
       replication test is inapplicable and is not attempted, stated
       explicitly rather than silently skipped). Using ONLY the inner-
       train-observable portion of the pre-2020-halving cycle (cycle 3,
       2016-07-09 -> 2020-05-11; months ~6-45 as observed inside inner-
       train), rank the phase buckets by mean return. Then, independently,
       using ONLY the inner-validation-observable portion of the post-
       2020-halving cycle (cycle 4, 2020-05-11 -> 2024-04-20; months ~8-
       31 as observed inside inner-validation), rank those buckets the
       same way. Report the Spearman rank correlation between the two
       rankings over the buckets BOTH windows actually observe with a
       material bar count (this file's threshold: >=1,000 bars in each
       window) -- restated below exactly which buckets that is, before
       any correlation number is read. This is doing the job R-75's
       BTC/ETH match did, but across cycles instead of across assets.

**Pre-registered stop rule (fixed now, before any number below was
computed):** proceed to Step B only if BOTH (a) the true-halving
dispersion clears the placebo null's 95th percentile, AND (b) the cross-
cycle rank correlation on shared buckets is >= 0.5 (matching R-75's own
bar for its analogous cross-asset check). If either fails: STOP, do not
build a strategy, report the measurement as a clean negative with the
gate itself as the product, exactly as R-73/R-74/R-75's gated branches
did. The bar is not relaxed after seeing the numbers -- a marginal miss
is reported as "the gate failed marginally" and the round still stops.

**Configs evaluated in this file: 0** (a fixed, non-swept measurement --
this project's standing accounting convention for this exact kind of gate
study, R-53/R-54/R-73/R-74/R-75's own Step-A/step-2 studies contribute 0
the same way; the placebo draws are the null, not a swept parameter).

=====================================================================
DATA DISCIPLINE
=====================================================================

BTC (`data/btcusd_spot_5m.csv.gz` via `tradebot.data.load_dataset`) is
truncated to strictly before `OOS_START = "2023-01-01"` immediately on
load. `assert_no_holdout(df)` re-checks the max timestamp after every
load as an independent second guard, same pattern as
`experiments/r75_conservative_dow_signal.py` /
`experiments/r74_conservative_mvrv_level.py`. No bar dated 2023-01-01 or
later is ever read, held, or printed by this file.

Usage::

    python experiments/r79_conservative_halving_gate.py gate   # everything, Step A + verdict
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

from experiments.r79_shared import (  # noqa: E402
    HALVINGS,
    cycle_phase,
    fake_cycle_phase,
    months_since_halving,
    phase_bucket,
    placebo_reference_dates,
)

DATA_DIR = ROOT / "data"
OOS_START = "2023-01-01"              # NEVER read in this file
TRAIN = ("2017-01-01", "2020-12-31")  # inner-train, per ROUTINE.md step 3
VALID = ("2021-01-01", "2022-12-31")  # inner-validation

N_BUCKETS = 4
N_PLACEBO = 1000            # >=500 per pre-registration; fixed, never re-drawn
PLACEBO_SEED = 79           # this round's number, fixed once
MIN_BUCKET_N = 1_000        # min bars for a bucket to count as "observed" in A.3


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


def load_btc() -> pd.DataFrame:
    """BTC spot, truncated before OOS_START at load time."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def train_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[TRAIN[0]:TRAIN[1]]


def valid_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[VALID[0]:VALID[1]]


# ------------------------------------------------------------- the statistic


def log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"]).diff().dropna()


def dispersion_stat(means: np.ndarray) -> float:
    """Population std of the conditional-mean values."""
    return float(np.std(means, ddof=0))


def bucket_means(r: pd.Series, bucket: pd.Series) -> pd.Series:
    """Mean return per bucket, aligned on r's own index (bucket may carry
    one extra leading row relative to r, from the .diff())."""
    b = bucket.reindex(r.index)
    return pd.Series(r.to_numpy(), index=r.index).groupby(b.to_numpy()).mean()


def bucket_counts(r: pd.Series, bucket: pd.Series) -> pd.Series:
    b = bucket.reindex(r.index)
    return pd.Series(r.to_numpy(), index=r.index).groupby(b.to_numpy()).count()


# --------------------------------------------------------- A.1-2: true dispersion


def measure_true_dispersion(train: pd.DataFrame) -> dict:
    bucket = phase_bucket(train.index, n_buckets=N_BUCKETS)
    r = log_returns(train)
    means = bucket_means(r, bucket)
    counts = bucket_counts(r, bucket)
    means = means.reindex(range(N_BUCKETS))
    counts = counts.reindex(range(N_BUCKETS)).fillna(0).astype(int)

    stat = dispersion_stat(means.to_numpy())
    print("\nBTC inner-train (2017-01-01 -> 2020-12-31), per true-halving-phase-quartile "
          "(0=just-halved .. 3=pre-next-halving):")
    for b in range(N_BUCKETS):
        print(f"  bucket {b}: mean={means[b]*1e4:+.4f} bps  n={counts[b]:,}")
    print(f"observed dispersion (pop. std of {N_BUCKETS} bucket means): {stat:.6e}")
    return dict(stat=stat, means=means, counts=counts)


# ------------------------------------------------------------- A.2: placebo null


def _fake_phase_days_vectorized(days_ns: np.ndarray, anchor_ns: int,
                                 gap_days: float = 1406.0) -> np.ndarray:
    """Independent, numpy-vectorized reimplementation of
    `r79_shared.fake_cycle_phase`'s math (`fake_bracket_fn`'s
    floor/fractional-part construction: n = floor((ts-anchor)/gap);
    phase = (ts-anchor)/gap - n`, i.e. (ts-anchor)/gap mod 1). Verified
    bit-close to the reference `r79_shared.fake_cycle_phase` on a spot
    sample below (`_verify_vectorized_fake_phase`) before being trusted
    for the full 1,000-draw placebo loop -- this file does NOT modify
    `r79_shared.py`, per this round's data-discipline instructions; it
    reimplements the same public, documented formula independently, the
    same fix the parallel novel branch applied to the identical
    bottleneck (disclosed in this file's docstring "DEVIATION" note).
    `days_ns` = int64 nanosecond timestamps (tz-naive, matching
    `fake_cycle_phase`'s own internal `.tz_convert(None)` step).
    """
    gap_ns = np.int64(round(gap_days * 86400 * 1e9))
    delta = days_ns.astype(np.int64) - np.int64(anchor_ns)
    n = np.floor(delta / gap_ns)
    phase = delta / gap_ns - n
    return np.clip(phase, 0.0, 1.0 - 1e-12)


def _verify_vectorized_fake_phase(days: pd.DatetimeIndex, n_check: int = 25,
                                   seed: int = PLACEBO_SEED) -> None:
    """Spot-check: for `n_check` random anchors (drawn from the SAME
    `placebo_reference_dates` pool used by the real run) and the full
    `days` array, the vectorized reimplementation above must match the
    reference `r79_shared.fake_cycle_phase` (called the slow, per-element
    way) to float precision. Raises on any mismatch -- this file refuses
    to trust the fast path unless it is independently shown to compute
    the identical function first."""
    check_anchors = placebo_reference_dates(n=n_check, seed=seed + 1)  # different seed: independent of the actual 1000 draws
    days_naive = days.tz_convert(None) if days.tz is not None else days
    days_ns = days_naive.values.astype("datetime64[ns]").astype(np.int64)
    worst = 0.0
    for anchor in check_anchors:
        ref = fake_cycle_phase(days, anchor).to_numpy()
        anchor_naive_ns = np.int64(anchor.tz_convert(None).value)
        fast = _fake_phase_days_vectorized(days_ns, anchor_naive_ns)
        worst = max(worst, float(np.max(np.abs(ref - fast))))
    assert worst < 1e-9, (
        f"vectorized fake-phase reimplementation diverges from r79_shared's "
        f"reference fake_cycle_phase by {worst:.3e} -- refusing to trust it")
    print(f"  vectorized fake-phase reimplementation verified against "
          f"r79_shared.fake_cycle_phase reference: max|diff|={worst:.3e} over "
          f"{n_check} anchors x {len(days):,} days  PASS", file=sys.stderr)


def placebo_null(train: pd.DataFrame, n: int = N_PLACEBO,
                  seed: int = PLACEBO_SEED) -> np.ndarray:
    """DEVIATION from the first draft of this file, disclosed: the first
    attempt called `r79_shared.fake_cycle_phase` once per placebo draw
    (even at day granularity) and additionally recomputed
    `sorted(set(idx.normalize()))` on every draw -- both pure-Python-level
    operations repeated 1,000x over a window whose day-list and bar->day
    mapping never actually change between draws. Measured: ~0.9s/draw,
    ~15 minutes projected for 1,000 draws; the coordinator flagged this
    stall (the parallel novel branch hit the identical bottleneck in its
    own file) after ~7 minutes and it was killed at draw 400/1000. Fixed
    by (a) hoisting the day list and the bar->day position mapping out of
    the loop (computed ONCE, reused for every anchor), and (b) replacing
    the per-draw call into `r79_shared.fake_cycle_phase`'s internal
    per-timestamp Python loop with an independent numpy-vectorized
    reimplementation of the identical formula
    (`_fake_phase_days_vectorized`), verified bit-close to the reference
    function on a held-out spot sample of anchors
    (`_verify_vectorized_fake_phase`, run once, asserted, before this loop
    trusts it for all 1,000 draws). `r79_shared.py` itself is NOT modified
    anywhere in this file -- this is a private, independently-verified
    reimplementation, per this round's "coordinate only by adding, never
    editing existing functions" instruction.
    """
    r = log_returns(train)
    idx = r.index
    r_vals = r.to_numpy()

    days = pd.DatetimeIndex(sorted(set(idx.normalize())))
    _verify_vectorized_fake_phase(days)

    days_naive = days.tz_convert(None) if days.tz is not None else days
    days_ns = days_naive.values.astype("datetime64[ns]").astype(np.int64)
    idx_naive = idx.tz_convert(None) if idx.tz is not None else idx
    bar_day_ns = idx_naive.normalize().values.astype("datetime64[ns]").astype(np.int64)
    day_pos = np.searchsorted(days_ns, bar_day_ns)  # bar -> position in `days`, computed ONCE

    anchors = placebo_reference_dates(n=n, seed=seed)
    edges = np.linspace(0.0, 1.0, N_BUCKETS + 1)
    stats = np.empty(n)
    t0 = time.time()
    for i, anchor in enumerate(anchors):
        anchor_ns = np.int64(anchor.tz_convert(None).value) if anchor.tz is not None else np.int64(anchor.value)
        phase_days = _fake_phase_days_vectorized(days_ns, anchor_ns)
        bucket_days = np.clip(np.digitize(phase_days, edges[1:-1]), 0, N_BUCKETS - 1)
        bucket_per_bar = bucket_days[day_pos]

        sums = np.bincount(bucket_per_bar, weights=r_vals, minlength=N_BUCKETS)
        counts = np.bincount(bucket_per_bar, minlength=N_BUCKETS)
        with np.errstate(invalid="ignore"):
            means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        stats[i] = dispersion_stat(means[~np.isnan(means)])
        if (i + 1) % 200 == 0:
            print(f"  placebo draw {i + 1}/{n}  ({time.time() - t0:.2f}s elapsed)",
                  file=sys.stderr)
    print(f"  placebo null: {n} draws in {time.time() - t0:.2f}s total", file=sys.stderr)
    return stats


# --------------------------------------------------------- A.3: cross-cycle


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, no scipy dependency (not installed in
    this venv) -- rank both arrays, Pearson-correlate the ranks."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cross_cycle_replication(btc_all: pd.DataFrame) -> dict:
    cycle3_end = HALVINGS[2]   # 2020-05-11, real halving date, public constant
    cycle4_start = HALVINGS[2]

    train = train_slice(btc_all)
    win1 = train.loc[train.index < cycle3_end]     # cycle-3 portion of inner-train
    win2 = valid_slice(btc_all)                     # all of inner-validation, cycle-4 only

    m1_start = int(months_since_halving(pd.DatetimeIndex([win1.index[0]])).iloc[0])
    m1_end = int(months_since_halving(pd.DatetimeIndex([win1.index[-1]])).iloc[0])
    m2_start = int(months_since_halving(pd.DatetimeIndex([win2.index[0]])).iloc[0])
    m2_end = int(months_since_halving(pd.DatetimeIndex([win2.index[-1]])).iloc[0])
    print(f"\nwindow 1 (cycle 3, inner-train-observable): {win1.index[0]} -> "
          f"{win1.index[-1]}  (months since halving {m1_start}-{m1_end})")
    print(f"window 2 (cycle 4, inner-validation-observable): {win2.index[0]} -> "
          f"{win2.index[-1]}  (months since halving {m2_start}-{m2_end})")

    b1 = phase_bucket(win1.index, n_buckets=N_BUCKETS)
    b2 = phase_bucket(win2.index, n_buckets=N_BUCKETS)
    r1 = log_returns(win1)
    r2 = log_returns(win2)
    means1 = bucket_means(r1, b1).reindex(range(N_BUCKETS))
    means2 = bucket_means(r2, b2).reindex(range(N_BUCKETS))
    n1 = bucket_counts(r1, b1).reindex(range(N_BUCKETS)).fillna(0).astype(int)
    n2 = bucket_counts(r2, b2).reindex(range(N_BUCKETS)).fillna(0).astype(int)

    print(f"\nwindow 1 (cycle 3) per-bucket: "
          + "  ".join(f"b{b}: mean={means1[b]*1e4:+.4f}bps n={n1[b]:,}" for b in range(N_BUCKETS)))
    print(f"window 2 (cycle 4) per-bucket: "
          + "  ".join(f"b{b}: mean={means2[b]*1e4:+.4f}bps n={n2[b]:,}" for b in range(N_BUCKETS)))

    shared = [b for b in range(N_BUCKETS)
              if n1[b] >= MIN_BUCKET_N and n2[b] >= MIN_BUCKET_N]
    print(f"\nbuckets observed with >= {MIN_BUCKET_N:,} bars in BOTH windows: {shared}")

    if len(shared) < 3:
        print("fewer than 3 shared buckets -- Spearman correlation is not "
              "meaningfully computable / reportable; treated as a FAIL of "
              "criterion (b) by construction.")
        return dict(shared=shared, rho=float("nan"), means1=means1, means2=means2)

    a = means1.loc[shared].to_numpy()
    b_ = means2.loc[shared].to_numpy()
    rho = spearman(a, b_)
    print(f"Spearman rank correlation over {len(shared)} shared buckets: {rho:.4f}")
    return dict(shared=shared, rho=rho, means1=means1, means2=means2)


# --------------------------------------------------------------------- verdict


def gate() -> None:
    print("=" * 78)
    print("R-79 CONSERVATIVE: halving-cycle phase -- STEP A measurement gate")
    print("=" * 78)

    btc_all = load_btc()
    train = train_slice(btc_all)
    print(f"inner-train: {len(train):,} bars  {train.index[0]} -> {train.index[-1]}")

    true_result = measure_true_dispersion(train)

    print(f"\nplacebo-offset null ({N_PLACEBO} draws, seed={PLACEBO_SEED}, "
          f"gap_days=1406.0 -- r79_shared default, mean of real 3rd/4th "
          f"halving intervals):")
    null_stats = placebo_null(train)
    p95 = float(np.percentile(null_stats, 95))
    p99 = float(np.percentile(null_stats, 99))
    pval = float((null_stats >= true_result["stat"]).mean())
    a_pass = true_result["stat"] > p95
    print(f"  null mean={null_stats.mean():.6e}  p95={p95:.6e}  p99={p99:.6e}")
    print(f"  observed={true_result['stat']:.6e}  -> "
          f"{'CLEARS the 95th percentile' if a_pass else 'DOES NOT CLEAR the 95th percentile'}")
    print(f"  empirical one-sided p-value (null >= observed): {pval:.4f}")

    print("\n" + "=" * 78)
    print("STEP A.3 -- cross-cycle replication")
    print("=" * 78)
    cc = cross_cycle_replication(btc_all)
    b_pass = (not np.isnan(cc["rho"])) and cc["rho"] >= 0.5

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (fixed before any number above was computed):")
    print("  proceed to Step B only if BOTH:")
    print("    (a) true-halving dispersion clears the placebo null's 95th percentile, AND")
    print("    (b) cross-cycle Spearman rank correlation on shared buckets >= 0.5.")
    print("  otherwise: STOP, report as a clean negative, no strategy built.")
    print("=" * 78)
    print(f"\n(a) true dispersion clears placebo p95: {a_pass}  "
          f"(observed={true_result['stat']:.6e}, p95={p95:.6e}, p-value={pval:.4f})")
    print(f"(b) cross-cycle rank correlation >= 0.5: {b_pass}  "
          f"(rho={cc['rho']:.4f}, shared buckets={cc['shared']})")

    passed = a_pass and b_pass
    print(f"\nGATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session: {btc_all.index.max()}  (< {OOS_START})")


if __name__ == "__main__":
    cmds = {"gate": gate}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r79_conservative_halving_gate.py [{'|'.join(cmds)}]")
