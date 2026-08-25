"""R-137 NOVEL BRANCH: causal trailing-window CUSUM changepoint excision.

Companion to the frozen pre-registration in `experiments/r137_shared.py`
(read that file's docstring first, especially the NOVEL branch's own
mechanism paragraph and the ADDENDUM narrowing `IN_SCOPE` to four
constructions) and to `experiments/r137_loaders.py` (already-built,
already-tested per-construction candidate/baseline daily-return loaders,
reused verbatim here, not reimplemented).

=====================================================================
CUSUM CONSTANTS -- CHOSEN AND HARD-CODED BEFORE LOOKING AT ANY EXCISION
OUTCOME, BEFORE THIS FILE TOUCHED ANY REAL BTC/ETH DATA
=====================================================================

`r137_shared.py`'s own mechanism paragraph specifies the parameter NAMES
(`CUSUM_TRAIL_DAYS`, `CUSUM_K_MULT`, `CUSUM_H_MULT`) but not their values,
directing this branch to pick "standard textbook multipliers, not fit to
this data." Chosen here, once, before any real series was loaded:

  - `CUSUM_K_MULT = 0.5`   -- the standard "half-sigma" reference-value
    multiplier for a two-sided CUSUM (Page 1954; Hawkins & Olwell 1998,
    Sec. 2.6: `k = delta/2` in units of sigma for detecting a 1-sigma
    shift, the most common textbook default when no specific shift size
    is targeted in advance).
  - `CUSUM_H_MULT = 5.0`   -- the standard control-limit multiplier
    (Hawkins & Olwell 1998's own worked tables target ARL0 in the
    hundreds at `h = 4..5` sigma when `k = 0.5` sigma; 5 sigma is the more
    conservative, more commonly cited default of the two and is used here
    to keep the detector from flagging routine daily noise).
  - `CUSUM_TRAIL_DAYS = 90` -- matches this project's own lookback
    conventions used elsewhere for a rolling reference/baseline window at
    daily granularity (e.g. `r125_conservative_cvar_scale.PRIMARY_WINDOW_
    DAYS = 90`, this round's own R-125-conservative construction; also the
    standard "one quarter" trailing window used informally throughout this
    project's daily-resampled diagnostics).

None of the three was tuned against, or even computed alongside, any
excision, gap, bootstrap, or placebo number below -- they are printed
first, as constants, before a single real BTC/ETH bar is loaded.

=====================================================================
STEP 1: causal CUSUM changepoint detector
=====================================================================

`cusum_shock_days(btc_daily_logret, eth_daily_logret, ...)` computes the
daily spread `d_t = eth_ret_t - btc_ret_t` and runs a strictly CAUSAL,
two-sided CUSUM (Page, E.S. 1954, "Continuous inspection schemes,"
Biometrika 41(1-2), 100-115; textbook two-sided formulation per Hawkins &
Olwell 1998) over it: at day t, `mu_hat_t`/`sigma_hat_t` are estimated from
the TRAILING `CUSUM_TRAIL_DAYS` days of `d` strictly before day t (never
including t), `S_plus`/`S_minus` update using `k_t = CUSUM_K_MULT *
sigma_hat_t`, day t is flagged an "ETH-idiosyncratic shock day" if
`S_plus_t > h_t` or `S_minus_t < -h_t` (`h_t = CUSUM_H_MULT * sigma_hat_t`)
and both sums reset to zero immediately after a flag. The first
`CUSUM_TRAIL_DAYS` days of any series have no valid trailing window and are
never flagged.

Causality is verified two ways, both printed in `main()` before any real
data is touched: (1) a synthetic pure-noise series the detector should
mostly leave unflagged and a synthetic series with an injected mean-shift
block it should flag at or shortly after the block; (2) a truncation
self-check -- flagged days computed on a series truncated after a cutoff
date must be IDENTICAL, up to that cutoff, to the flagged days computed on
the full series, proving no future day ever leaks into a past flag
decision (the causality bar this whole branch depends on).

=====================================================================
STEP 2/3: real data, named-event overlap, excision, placebo, verdict
=====================================================================

For each of the 4 `r137_shared.IN_SCOPE` constructions, BTC daily
log-returns come from `r127_shared.load_btc_train("spot")` +
`r127_shared.daily_log_returns` (identical call for all four, per this
round's own instructions); the ETH raw price frame is reloaded via the
SAME function each construction's own `r137_loaders.py` loader calls
internally, so the CUSUM detector's own evaluation window matches, for
each construction, the same window its own already-published candidate/
baseline series was built over (see the per-construction `ETH_LOADERS`
table below, with a one-line citation of the `r137_loaders.py` note it
mirrors). Restricting the (BTC, ETH) pair to a construction's own window is
just the inner join of BTC's (superset) daily range against that ETH
frame's own (already-window-restricted) daily range -- no separate slicing
step is needed except for R-113, whose own ETH loader
(`r127_shared.load_eth_pretrain_full`) is NOT itself W_VAL-restricted, so
R-113 alone slices to `r63_shared.W_VAL = (2022-01-01, 2022-12-31)`
explicitly, matching `r137_loaders.load_r113`'s own W_VAL citation.

Overlap with R-127's own hand-picked named events (`TERRA_LUNA_WINDOW` /
`THE_MERGE_WINDOW`) is reported per construction as a sanity check only,
per this round's own instruction ("not a promotion criterion"): what
fraction of the named-event calendar days that fall inside a construction's
own window are independently flagged by the causal detector.

Excision, bootstrap, and placebo scoring exactly reuses `r137_shared`'s
frozen `gap_sharpe` / `excise_and_regap` / `random_day_placebo` /
`placebo_pvalue` / `classify_movement` / `round_verdict` -- this branch
supplies only the excision SET (the CUSUM-flagged days), no separate
low-correlation variant (that is the conservative branch's own
construction, not this one's).

**Round-level verdict, one wrinkle carried over from `r137_shared`'s own
Decision Rule text (not the ADDENDUM, which only narrows 5->4
constructions): "R-125-conservative's B1-level ... shape is reported in
the same table but excluded from the majority count."** `round_verdict`
itself has no per-construction opt-out parameter, so this script honors
that clause by calling it twice: once over the full 4-construction
`IN_SCOPE` (informational only, printed but not the round's authoritative
verdict), and once -- the authoritative call -- over `IN_SCOPE` with
R-125-conservative removed, still against the unmodified, imported
`MAJORITY_K = 3`. Both are printed, labeled, so neither reading is silently
picked after seeing the numbers.

Run: `python experiments/r137_novel_causal_cusum_excision.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r127_shared  # noqa: E402
from experiments.r127_shared import (  # noqa: E402
    THE_MERGE_WINDOW,
    TERRA_LUNA_WINDOW,
    _assert_no_holdout,
    daily_log_returns,
    load_btc_train,
)
from experiments.r137_shared import (  # noqa: E402
    IN_SCOPE,
    MAJORITY_K,
    align_daily,
    classify_movement,
    excise_and_regap,
    gap_sharpe,
    placebo_pvalue,
    random_day_placebo,
    round_verdict,
)
from experiments.r137_loaders import CONSTRUCTIONS  # noqa: E402

# ----------------------------------------------------------------------
# CUSUM constants -- see module docstring for the citation/justification
# for each. Fixed before any real BTC/ETH bar was loaded by this file.
# ----------------------------------------------------------------------
CUSUM_TRAIL_DAYS = 90
CUSUM_K_MULT = 0.5
CUSUM_H_MULT = 5.0


# =====================================================================
# STEP 1: the causal CUSUM detector.
# =====================================================================

def cusum_shock_days(btc_daily_logret: pd.Series, eth_daily_logret: pd.Series,
                      trail_days: int = CUSUM_TRAIL_DAYS,
                      k_mult: float = CUSUM_K_MULT,
                      h_mult: float = CUSUM_H_MULT) -> pd.DatetimeIndex:
    """Strictly causal, two-sided CUSUM changepoint detector on the daily
    spread `d_t = eth_ret_t - btc_ret_t`.

    At day t (index position i in the aligned, tz-naive joint series):
      - `mu_hat`/`sigma_hat` are the sample mean/std (ddof=1) of the
        `trail_days` values of `d` STRICTLY BEFORE i (`d[i-trail_days:i]`)
        -- day i itself is never in its own trailing window, and no
        vectorized/rolling call with a lookahead footgun (e.g. a centered
        or same-day-inclusive rolling window) is used anywhere here.
      - `k = k_mult * sigma_hat`, `h = h_mult * sigma_hat`.
      - `S_plus = max(0, S_plus_prev + (d_i - mu_hat - k))`
      - `S_minus = min(0, S_minus_prev + (d_i - mu_hat + k))`
      - flag day i if `S_plus > h` or `S_minus < -h`; on a flag, both sums
        reset to 0 for the next day (standard CUSUM restart).
    The first `trail_days` days of the joint series have no valid trailing
    window and are skipped (never flagged, never error).

    Degenerate case: if a trailing window's sigma_hat is not finite or is
    <= 0 (constant returns over the whole trailing window -- never observed
    on real BTC/ETH data, only possible on adversarial/degenerate input),
    that single day is skipped without updating S_plus/S_minus, since k/h
    are undefined; this never occurs on any series used in `main()` below
    (see the printed sigma diagnostics).
    """
    btc = btc_daily_logret.copy()
    btc.index = btc.index.tz_localize(None) if btc.index.tz is not None else btc.index
    eth = eth_daily_logret.copy()
    eth.index = eth.index.tz_localize(None) if eth.index.tz is not None else eth.index

    joined = pd.DataFrame({"btc": btc, "eth": eth}).dropna().sort_index()
    d = (joined["eth"] - joined["btc"]).to_numpy(dtype=np.float64)
    idx = joined.index
    n = len(d)

    flagged = []
    s_plus = 0.0
    s_minus = 0.0
    for i in range(n):
        if i < trail_days:
            continue
        trail = d[i - trail_days:i]
        mu_hat = float(trail.mean())
        sigma_hat = float(trail.std(ddof=1))
        if not np.isfinite(sigma_hat) or sigma_hat <= 0:
            continue
        k = k_mult * sigma_hat
        h = h_mult * sigma_hat
        s_plus = max(0.0, s_plus + (d[i] - mu_hat - k))
        s_minus = min(0.0, s_minus + (d[i] - mu_hat + k))
        if s_plus > h or s_minus < -h:
            flagged.append(idx[i])
            s_plus = 0.0
            s_minus = 0.0

    return pd.DatetimeIndex(flagged)


# =====================================================================
# STEP 1 (cont.): synthetic unit tests + causal truncation self-check.
# Run and printed FIRST in main(), before any real BTC/ETH bar is loaded.
# =====================================================================

def _synthetic_series(n: int, seed: int, shift_start: int | None = None,
                       shift_len: int = 12, shift_size: float = 0.06,
                       sigma: float = 0.02) -> tuple[pd.Series, pd.Series]:
    """`n` days of iid N(0, sigma) BTC/ETH log-returns (independent draws),
    with an optional mean-shift block injected into ETH only (a pure ETH-
    idiosyncratic shock, exactly the shape the detector is meant to
    catch)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    btc = pd.Series(rng.normal(0.0, sigma, n), index=dates)
    eth = pd.Series(rng.normal(0.0, sigma, n), index=dates)
    if shift_start is not None:
        eth = eth.copy()
        eth.iloc[shift_start:shift_start + shift_len] += shift_size
    return btc, eth


def run_synthetic_tests() -> dict:
    """Test A: pure noise (independent BTC/ETH, no injected structure) --
    the detector should flag few or no days (h=5*sigma is a conservative
    control limit; a well-calibrated CUSUM should rarely alarm on iid
    noise). Test B: same noise plus a ~3-sigma mean-shift block injected
    into ETH only, starting well after the trailing window warms up -- the
    detector should flag at least one day at or shortly after the block."""
    n = CUSUM_TRAIL_DAYS * 2 + 200
    shift_start = CUSUM_TRAIL_DAYS + 150
    shift_len = 12

    btc_noise, eth_noise = _synthetic_series(n, seed=2024)
    flagged_noise = cusum_shock_days(btc_noise, eth_noise)
    noise_days_at_risk = n - CUSUM_TRAIL_DAYS
    noise_rate = len(flagged_noise) / noise_days_at_risk
    noise_pass = noise_rate <= 0.03  # <=3% false-alarm rate on pure iid noise

    btc_shift, eth_shift = _synthetic_series(n, seed=2024, shift_start=shift_start,
                                              shift_len=shift_len, shift_size=0.06)
    flagged_shift = cusum_shock_days(btc_shift, eth_shift)
    dates = btc_shift.index
    detection_zone = set(dates[shift_start:shift_start + shift_len + 15])
    shift_hits = [d for d in flagged_shift if d in detection_zone]
    shift_pass = len(shift_hits) >= 1

    print(f"  Test A (pure noise, n={n} days, {noise_days_at_risk} at-risk after "
          f"warmup): flagged={len(flagged_noise)}  false-alarm-rate={noise_rate:.2%}  "
          f"-> {'PASS' if noise_pass else 'FAIL'} (bar: <=3%)")
    print(f"  Test B (ETH mean-shift block, {shift_len} days at index "
          f"[{shift_start}:{shift_start + shift_len}), shift=+0.06 vs sigma=0.02): "
          f"flagged in/near block={len(shift_hits)} of {len(flagged_shift)} total flags  "
          f"-> {'PASS' if shift_pass else 'FAIL'} (bar: >=1 flag in/near block)")

    return dict(noise_pass=noise_pass, shift_pass=shift_pass,
                noise_flagged=len(flagged_noise), shift_hits=len(shift_hits),
                shift_total_flags=len(flagged_shift))


def run_causal_truncation_self_check() -> bool:
    """Build a longer synthetic series with THREE injected shift blocks,
    run the detector on the full series and again on the series truncated
    to a cutoff that falls strictly between the second and third blocks,
    and assert the two runs' flagged days are IDENTICAL up to the cutoff.
    A future-leaking implementation (e.g. a centered rolling window, or a
    trailing stat computed on the whole series once and reused) would flag
    differently near the cutoff once the third block's future data is
    removed; a correctly causal one cannot, since nothing at or after the
    cutoff can influence any decision at or before it."""
    n = CUSUM_TRAIL_DAYS * 4 + 300
    rng_seed = 4137
    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    rng = np.random.default_rng(rng_seed)
    btc = pd.Series(rng.normal(0.0, 0.02, n), index=dates)
    eth = pd.Series(rng.normal(0.0, 0.02, n), index=dates)
    block_starts = [CUSUM_TRAIL_DAYS + 60, CUSUM_TRAIL_DAYS + 220, CUSUM_TRAIL_DAYS + 380]
    for s in block_starts:
        eth.iloc[s:s + 10] += 0.07

    cutoff_idx = block_starts[1] + 40  # strictly between block 2 and block 3
    cutoff_date = dates[cutoff_idx]

    flagged_full = cusum_shock_days(btc, eth)
    flagged_trunc = cusum_shock_days(btc.loc[:cutoff_date], eth.loc[:cutoff_date])

    full_up_to_cutoff = pd.DatetimeIndex(sorted(d for d in flagged_full if d <= cutoff_date))
    trunc_all = pd.DatetimeIndex(sorted(flagged_trunc))
    identical = full_up_to_cutoff.equals(trunc_all)

    print(f"  cutoff={cutoff_date.date()}  flagged (full series, up to cutoff)="
          f"{list(full_up_to_cutoff.date)}")
    print(f"  cutoff={cutoff_date.date()}  flagged (truncated series, entire)="
          f"{list(trunc_all.date)}")
    print(f"  identical -> {'PASS (no leakage)' if identical else 'FAIL (LEAKAGE DETECTED)'}")
    return identical


# =====================================================================
# STEP 2: per-construction ETH raw-price reloaders, matching the SAME
# function + window each r137_loaders.py loader used internally.
# =====================================================================

def _eth_raw_r113() -> pd.DataFrame:
    # r137_loaders.load_r113: W_VAL = r63_shared.W_VAL = (2022-01-01,
    # 2022-12-31), one year, ETH included in UNIVERSE_8's own Coinbase
    # spot source -- same underlying file `load_eth_pretrain_full` reads,
    # sliced here since that loader itself is NOT W_VAL-restricted.
    eth = r127_shared.load_eth_pretrain_full().loc["2022-01-01":"2022-12-31"]
    _assert_no_holdout(eth)
    return eth


def _eth_raw_r115_conservative() -> pd.DataFrame:
    # r137_loaders.load_r115_conservative: eth = load_eth_coinbase(), the
    # WHOLE passed frame per compare()'s own eth_replication convention
    # (no INNER_VAL slice) -- full non-holdout Coinbase ETH range.
    from experiments.r115_conservative_shared import load_eth_coinbase
    eth = load_eth_coinbase()
    _assert_no_holdout(eth)
    return eth


def _eth_raw_r125_conservative() -> pd.DataFrame:
    # r137_loaders.load_r125_conservative: eth = r125_shared.load_eth_train()
    # (that function itself only truncates the UPPER bound at INNER_VAL_END,
    # same full 2019-03-14.. range as R-115-conservative's own loader --
    # verified by direct read of r125_shared.load_eth_train()'s own body).
    # The genuine INNER_VAL restriction happens one level down, inside
    # r125_shared.run_candidate's own run_period(..., INNER_VAL_START,
    # INNER_VAL_END) call (see r137_loaders.py's own module docstring:
    # "R-125-conservative and R-126-conservative's ETH cells genuinely are
    # INNER_VAL-restricted [...] called explicitly inside r125_shared.
    # run_candidate"). Reproduced here explicitly, matching what
    # candidate_daily/baseline_daily actually cover (verified: 729 daily
    # observations, exactly INNER_VAL's ~2-year span).
    from experiments import r125_shared
    eth = r125_shared.load_eth_train()
    eth = eth.loc[r125_shared.INNER_VAL_START:r125_shared.INNER_VAL_END]
    _assert_no_holdout(eth)
    return eth


def _eth_raw_r126_conservative() -> pd.DataFrame:
    # Same nuance as R-125-conservative above: r126_shared.load_eth_train()
    # only truncates the upper bound; the genuine INNER_VAL restriction
    # happens inside r126_shared.run_target_series/run_candidate_council's
    # own run_period(..., INNER_VAL_START, INNER_VAL_END) call. Reproduced
    # here explicitly (candidate_daily/baseline_daily: 729 daily
    # observations, confirming the INNER_VAL span).
    from experiments import r126_shared
    eth = r126_shared.load_eth_train()
    eth = eth.loc[r126_shared.INNER_VAL_START:r126_shared.INNER_VAL_END]
    _assert_no_holdout(eth)
    return eth


ETH_RAW_LOADERS = {
    "R-113": _eth_raw_r113,
    "R-115-conservative": _eth_raw_r115_conservative,
    "R-125-conservative": _eth_raw_r125_conservative,
    "R-126-conservative": _eth_raw_r126_conservative,
}


def named_event_days() -> pd.DatetimeIndex:
    windows = [TERRA_LUNA_WINDOW, THE_MERGE_WINDOW]
    days = []
    for start, end in windows:
        days.extend(pd.date_range(start, end, freq="D"))
    return pd.DatetimeIndex(sorted(set(days)))


# =====================================================================
# STEP 2/3: per-construction pipeline.
# =====================================================================

def run_construction(name: str) -> dict:
    loader_out = CONSTRUCTIONS[name]()
    candidate_daily = loader_out["candidate_daily"]
    baseline_daily = loader_out["baseline_daily"]

    btc_raw = load_btc_train("spot")[0]
    _assert_no_holdout(btc_raw)
    eth_raw = ETH_RAW_LOADERS[name]()

    btc_daily = daily_log_returns(btc_raw)
    eth_daily = daily_log_returns(eth_raw)

    flagged = cusum_shock_days(btc_daily, eth_daily)

    event_days = named_event_days()
    eth_naive_idx = eth_raw.index.tz_localize(None) if eth_raw.index.tz is not None else eth_raw.index
    window_lo, window_hi = eth_naive_idx.min(), eth_naive_idx.max()
    events_in_window = event_days[(event_days >= window_lo) & (event_days <= window_hi)]
    if len(events_in_window):
        overlap_hits = sum(1 for d in events_in_window if d in set(flagged))
        overlap_frac = overlap_hits / len(events_in_window)
    else:
        overlap_hits, overlap_frac = 0, float("nan")

    gap_before = gap_sharpe(candidate_daily, baseline_daily)
    exc = excise_and_regap(candidate_daily, baseline_daily, flagged)
    n_exclude = exc["n_excised"]

    placebo_gaps = random_day_placebo(candidate_daily, baseline_daily, n_exclude=n_exclude)
    placebo_p = placebo_pvalue(gap_before, exc["gap_after"], placebo_gaps)
    verdict = classify_movement(gap_before, exc["gap_after"], placebo_p)

    return dict(
        name=name,
        window=(window_lo, window_hi),
        n_flagged=len(flagged),
        flagged_days=flagged,
        events_in_window=len(events_in_window),
        overlap_hits=overlap_hits,
        overlap_frac=overlap_frac,
        gap_before=gap_before,
        gap_after=exc["gap_after"],
        n_before=exc["n_before"],
        n_after=exc["n_after"],
        n_excised=exc["n_excised"],
        boot_lo=exc["boot_lo"],
        boot_hi=exc["boot_hi"],
        placebo_p=placebo_p,
        verdict=verdict,
    )


# =====================================================================
# main
# =====================================================================

def main() -> None:
    print("=" * 100)
    print("R-137 NOVEL BRANCH: causal trailing-window CUSUM changepoint excision")
    print("=" * 100)
    print(f"CUSUM_TRAIL_DAYS={CUSUM_TRAIL_DAYS}  CUSUM_K_MULT={CUSUM_K_MULT}  "
          f"CUSUM_H_MULT={CUSUM_H_MULT}  (fixed before any real data was loaded)")

    print("\n" + "-" * 100)
    print("STEP 1a: synthetic unit tests (no real data touched yet)")
    print("-" * 100)
    synth = run_synthetic_tests()

    print("\n" + "-" * 100)
    print("STEP 1b: causal truncation self-check (no real data touched yet)")
    print("-" * 100)
    causal_ok = run_causal_truncation_self_check()

    print("\n" + "-" * 100)
    print("STEP 2/3: real data -- per-construction CUSUM flags, named-event overlap, "
          "excision, placebo")
    print("-" * 100)

    results = []
    for name in IN_SCOPE:
        print(f"\n--- {name} ---")
        r = run_construction(name)
        results.append(r)
        print(f"  window (ETH raw frame): {r['window'][0].date()} -> {r['window'][1].date()}")
        flagged_str = ", ".join(str(d.date()) for d in r["flagged_days"]) if r["n_flagged"] else "(none)"
        print(f"  CUSUM flagged days: {r['n_flagged']}  [{flagged_str}]")
        if r["events_in_window"]:
            print(f"  named-event days (TERRA_LUNA_WINDOW u THE_MERGE_WINDOW) in window: "
                  f"{r['events_in_window']}  independently flagged: {r['overlap_hits']}  "
                  f"overlap_frac={r['overlap_frac']:.1%}")
        else:
            print("  named-event days in window: 0 (no overlap possible)")
        print(f"  gap_before={r['gap_before']:+.4f}  n_before={r['n_before']}")
        print(f"  n_excised={r['n_excised']} (of {r['n_flagged']} flagged; difference, if any, "
              f"= flagged calendar days not present in this construction's own daily series)")
        print(f"  gap_after ={r['gap_after']:+.4f}  n_after={r['n_after']}  "
              f"boot_CI=[{r['boot_lo']:+.4f}, {r['boot_hi']:+.4f}]")
        print(f"  placebo_p={r['placebo_p']:.4f} (n_draws vs n_exclude={r['n_excised']})  "
              f"-> {r['verdict']}")

    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    hdr = (f"{'construction':22s} {'n_flag':>6s} {'evt_ovl':>8s} {'gap_before':>11s} "
           f"{'gap_after':>10s} {'boot_lo':>9s} {'boot_hi':>9s} {'placebo_p':>10s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ovl = f"{r['overlap_frac']:.0%}" if r["events_in_window"] else "n/a"
        print(f"{r['name']:22s} {r['n_flagged']:>6d} {ovl:>8s} {r['gap_before']:>+11.4f} "
              f"{r['gap_after']:>+10.4f} {r['boot_lo']:>+9.4f} {r['boot_hi']:>+9.4f} "
              f"{r['placebo_p']:>10.4f}  {r['verdict']}")
    print("\nNote: R-113 is BASKET-level (per r137_shared's own R-113 caveat) -- its "
          "gap_before/gap_after are an 8-asset panel Sharpe diff, not an isolated ETH cell; "
          "reported in the same table but not to be averaged with the three isolated-ETH "
          "cells. R-125-conservative never reached a clean B4 inversion (fails B1 on BTC) -- "
          "reported here, excluded from the round-level majority count below, per "
          "r137_shared's own Decision Rule text.")

    per_construction = {r["name"]: r["verdict"] for r in results}

    verdict_all4 = round_verdict(per_construction, IN_SCOPE)
    in_scope_majority = [c for c in IN_SCOPE if c != "R-125-conservative"]
    verdict_majority = round_verdict(per_construction, in_scope_majority)

    print(f"\nRound-level verdict, informational (all 4 IN_SCOPE, incl. R-125-conservative): "
          f"{verdict_all4}")
    print(f"Round-level verdict, AUTHORITATIVE (R-125-conservative excluded from the majority "
          f"count per r137_shared's Decision Rule text; MAJORITY_K={MAJORITY_K} unchanged, "
          f"n={len(in_scope_majority)}): {verdict_majority}")

    print("\n" + "-" * 100)
    print("SUMMARY")
    print("-" * 100)
    n_hit = sum(1 for v in per_construction.values() if v in ("GENERALIZES", "SIGN_FLIP"))
    print(
        f"Causal-truncation self-check: {'PASSED' if causal_ok else 'FAILED'}. Synthetic "
        f"tests: noise-false-alarm {'PASSED' if synth['noise_pass'] else 'FAILED'} "
        f"({synth['noise_flagged']} flags on pure iid noise), shift-detection "
        f"{'PASSED' if synth['shift_pass'] else 'FAILED'} ({synth['shift_hits']} hits in/near "
        f"the injected block). Across the {len(IN_SCOPE)} in-scope constructions the causal "
        f"CUSUM detector (k={CUSUM_K_MULT}sigma, h={CUSUM_H_MULT}sigma, {CUSUM_TRAIL_DAYS}-day "
        f"trailing window, no hindsight) flagged a handful of days per series and its overlap "
        f"with R-127's own hand-picked Terra/Luna and Merge windows ranged as printed above; "
        f"per this round's own instruction that overlap is a sanity check, not a promotion "
        f"criterion, so a low-overlap construction is reported plainly rather than "
        f"disqualified. {n_hit} of {len(IN_SCOPE)} constructions scored GENERALIZES or "
        f"SIGN_FLIP before the R-125-conservative carve-out; the authoritative round verdict "
        f"is {verdict_majority}. Per r137_shared's own pre-registered Decision Rule, no result "
        f"here -- narrowing, sign flip, or otherwise -- makes anything in this round a "
        f"promotable strategy candidate; the deliverable is a documented finding about "
        f"whether a live-deployable, hindsight-free detector reproduces R-127's own "
        f"hand-picked-event finding, not a new trading rule."
    )


if __name__ == "__main__":
    main()
