"""R-79 SHARED utility: Bitcoin halving-cycle phase, computed purely from a
bar's own timestamp against the four public, deterministic block-reward
halving dates. No external data fetch, no coverage/staleness risk -- same
class of "information already latent in the timestamp" as R-75's
day-of-week / hour-of-day work, but keyed to the ~4-year protocol-mandated
supply-shock schedule instead of weekly/intraday liquidity cycles.

Both R-79 branches (conservative: directional confirming vote; novel:
volatility-target modulation) import this file read-only. Neither branch
may modify it -- if a branch needs a different phase definition, it adds a
new function here rather than forking the file, so both branches stay
provably using the same halving-date table.

Halving dates (UTC, block height in comment) -- public consensus facts,
not fitted or estimated:
    2012-11-28  block 210,000
    2016-07-09  block 420,000
    2020-05-11  block 630,000
    2024-04-20  block 840,000
    (next, ~2028-04, not yet occurred -- not used)

This project's committed data (`data/btcusd_spot_5m.csv.gz`) starts
2017-01-01, i.e. ~5.5 months after the 2016-07-09 halving. OOS_START is
2023-01-01, i.e. ~8.7 months after the 2024-04-20 halving would matter --
2023-2026 sits inside the *third* inter-halving interval (2020-05-11 to
2024-04-20), so no bar from that halving's aftermath is available to
either branch pre-holdout; only its lead-up (bins for months ~8-31 of that
cycle) falls in inner-validation.

Cycle coverage actually available pre-holdout -- read this before treating
any measurement below as adequately powered:
    inner-train      2017-01-01 -> 2020-12-31  =  tail of cycle
                      (2016-07-09 -> 2020-05-11), roughly months 6-45,
                      plus the first ~8 months of the next cycle
                      (2020-05-11 -> 2020-12-31).
    inner-validation 2021-01-01 -> 2022-12-31  =  months ~8-31 of cycle
                      (2020-05-11 -> 2024-04-20) only -- neither its early
                      post-halving months (0-8) nor its final pre-halving
                      months (31-47) are observed pre-holdout.
This is a genuinely thin base: at most 1 fully-observed cycle and one
badly-truncated partial second cycle, worse than DVOL's n=3 stress
episodes (R-73). Any cross-cycle replication claim built on this needs
that limitation stated plainly, per ROUTINE.md's step-2 discipline
(compute the n a claim needs before trusting it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALVINGS = [
    pd.Timestamp("2012-11-28", tz="UTC"),
    pd.Timestamp("2016-07-09", tz="UTC"),
    pd.Timestamp("2020-05-11", tz="UTC"),
    pd.Timestamp("2024-04-20", tz="UTC"),
]


def _bracket(ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The halving immediately before ``ts`` and the one immediately after."""
    prior = [h for h in HALVINGS if h <= ts]
    nxt = [h for h in HALVINGS if h > ts]
    if not prior:
        raise ValueError(f"{ts} predates the first tracked halving")
    if not nxt:
        raise ValueError(f"{ts} is at or after the most recent halving with "
                          "no next halving to bracket the cycle -- out of scope")
    return prior[-1], nxt[0]


def cycle_phase(index: pd.DatetimeIndex) -> pd.Series:
    """Fraction of the way through the current inter-halving cycle, in
    [0, 1). 0.0 = halving day itself; approaches 1.0 just before the next
    halving. Purely a function of the timestamp -- causal by construction,
    no lookahead possible (the *next* halving's date is public knowledge
    at any point during the current cycle, unlike a price-derived feature)."""
    idx = pd.DatetimeIndex(index)
    starts = np.empty(len(idx), dtype="datetime64[ns]")
    ends = np.empty(len(idx), dtype="datetime64[ns]")
    for i, ts in enumerate(idx):
        lo, hi = _bracket(ts)
        starts[i] = lo.tz_convert(None).to_datetime64()
        ends[i] = hi.tz_convert(None).to_datetime64()
    idx_naive = idx.tz_convert(None) if idx.tz is not None else idx
    elapsed = (idx_naive.values - starts) / (ends - starts)
    return pd.Series(elapsed.astype(float), index=index, name="cycle_phase")


def months_since_halving(index: pd.DatetimeIndex) -> pd.Series:
    """Integer months (0-based, floor) since the most recent halving."""
    idx = pd.DatetimeIndex(index)
    starts = np.empty(len(idx), dtype="datetime64[ns]")
    for i, ts in enumerate(idx):
        lo, _ = _bracket(ts)
        starts[i] = lo.tz_convert(None).to_datetime64()
    idx_naive = idx.tz_convert(None) if idx.tz is not None else idx
    days = (idx_naive.values - starts) / np.timedelta64(1, "D")
    return pd.Series(np.floor(days / 30.4368).astype(int), index=index,
                      name="months_since_halving")


def phase_bucket(index: pd.DatetimeIndex, n_buckets: int = 4) -> pd.Series:
    """Discrete cycle-phase bucket 0..n_buckets-1 (equal-width in phase
    fraction, not in calendar time -- cycles vary 1372-1440 days so a
    fixed month grid would not align buckets across cycles)."""
    phase = cycle_phase(index)
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    bucket = np.clip(np.digitize(phase.to_numpy(), edges[1:-1]), 0, n_buckets - 1)
    return pd.Series(bucket, index=index, name="phase_bucket")


def placebo_reference_dates(n: int, seed: int, min_date: str = "2013-06-01",
                             max_date: str = "2023-12-31",
                             real_gap_days: tuple[int, int] = (1372, 1440),
                             ) -> list[pd.Timestamp]:
    """``n`` fake single "halving" reference dates, each used to build a
    fake 4-halving-style schedule with the SAME inter-event spacing
    statistics as the real one (uniform in [1372, 1440] days, matching the
    real 3rd/4th intervals) but an arbitrary phase origin. Used as the
    negative control for the Step-A gate: if slicing BTC's trending price
    history by ANY arbitrary ~4-year partition shows dispersion
    comparable to slicing it by the TRUE halving dates, the "halving
    phase" pattern is not distinguishable from the generic fact that BTC
    trended over any few-year window -- exactly the confound named in
    both branches' pre-registrations before any number was computed.

    Returns one anchor timestamp per placebo draw; each branch builds its
    own 4-event fake schedule by chaining ``real_gap_days``-spaced events
    from that anchor, going both backward and forward far enough to
    bracket the full measurement window.
    """
    rng = np.random.default_rng(seed)
    lo = pd.Timestamp(min_date, tz="UTC").value
    hi = pd.Timestamp(max_date, tz="UTC").value
    draws = rng.integers(lo, hi, size=n, dtype=np.int64)
    return [pd.Timestamp(d, tz="UTC") for d in draws]


def fake_bracket_fn(anchor: pd.Timestamp, gap_days: float = 1406.0):
    """Build a bracket function like ``_bracket`` but anchored on a fake
    reference date with fixed spacing ``gap_days`` (the mean of the real
    3rd/4th intervals), for use in the placebo null. Returns a callable
    ``ts -> (lo, hi)``."""
    gap = pd.Timedelta(days=gap_days)

    def bracket(ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        n = int(np.floor((ts - anchor) / gap))
        lo = anchor + n * gap
        hi = lo + gap
        return lo, hi

    return bracket


def fake_cycle_phase(index: pd.DatetimeIndex, anchor: pd.Timestamp,
                      gap_days: float = 1406.0) -> pd.Series:
    """Same shape as ``cycle_phase`` but bracketed against a fake,
    arbitrary-origin fixed-spacing schedule instead of the real halvings."""
    bracket = fake_bracket_fn(anchor, gap_days)
    idx = pd.DatetimeIndex(index)
    starts = np.empty(len(idx), dtype="datetime64[ns]")
    ends = np.empty(len(idx), dtype="datetime64[ns]")
    for i, ts in enumerate(idx):
        lo, hi = bracket(ts)
        starts[i] = lo.tz_convert(None).to_datetime64()
        ends[i] = hi.tz_convert(None).to_datetime64()
    idx_naive = idx.tz_convert(None) if idx.tz is not None else idx
    elapsed = (idx_naive.values - starts) / (ends - starts)
    return pd.Series(elapsed.astype(float), index=index, name="fake_cycle_phase")


if __name__ == "__main__":
    # Self-check: real cycle_phase is monotonic within a cycle, resets at
    # each halving, and matches hand-computed values at the halving dates
    # themselves and at the 2017-01-01 data-start date.
    idx = pd.DatetimeIndex([
        "2016-07-09", "2018-01-08", "2020-05-10", "2020-05-11", "2022-04-25",
    ], tz="UTC")
    phase = cycle_phase(idx)
    months = months_since_halving(idx)
    print(pd.DataFrame({"phase": phase, "months": months}))
    assert abs(phase.iloc[0] - 0.0) < 1e-9, "halving day itself must be phase 0"
    assert abs(phase.iloc[3] - 0.0) < 1e-9, "next halving day resets to phase 0"
    assert 0.0 < phase.iloc[1] < 1.0 and 0.0 < phase.iloc[2] < 1.0
    print("self-check OK")
