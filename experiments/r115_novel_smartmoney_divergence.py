#!/usr/bin/env python
"""R-115 NOVEL branch: retail-vs-smart-money long/short DIVERGENCE --
Step A measurement gate ONLY, in the R-79/R-81/R-82's own style (a fixed,
pre-registered lead-time comparison against dated historical regime
transitions, run BEFORE any strategy code). No strategy is built unless
this gate passes.

=====================================================================
PRE-REGISTRATION (frozen before any real-market lead/lag number in this
file was computed -- docs/ROUTINE.md steps 1-2). Data EXPLORATION (raw
coverage-gap percentages, which columns are/aren't NaN where) was run
first, deliberately -- the brief itself instructs verifying the data
before writing code -- but no episode's LEAD, no null distribution, and
no pass/fail number was computed before every constant below was fixed.
If anything below is later contradicted by what actually happened, that
is stated in the results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). ``count_long_short_ratio`` (Binance's
   ALL-ACCOUNT, retail-dominated long/short ratio) and
   ``sum_toptrader_long_short_ratio`` (Binance's TOP-TRADER ratio, R-81's
   own signal) each z-scored against their own 14-day trailing baseline;
   when the two diverge sharply (``|divergence_z| = |smart_z - retail_z|``
   extreme), retail is one-sidedly positioned in a way smart money is
   NOT following -- the exact crowded, poorly-informed positioning
   Barber & Odean (2000, JF 55(2)) find precedes retail losses in
   traditional markets -- theorized here to precede a forced-deleveraging
   reversal in crypto derivatives specifically. See
   ``r115_novel_shared.py``'s own module docstring for the full formula,
   sign convention, and the literature search run before this file was
   written (crypto-specific academic precedent: Dunbar & Owusu-Amoako
   2023, JBEF 39:100812, on retail/speculative Bitcoin-futures
   positioning predicting returns -- a different venue/split than this
   round's exact construction, disclosed as suggestive support, not a
   direct precedent; practitioner-only sources found beyond that,
   reported as heuristic, not academic, evidence).

   Constraint attacked: INFO -- a genuinely new construction (this
   project's 16th INFO-axis signal attempt, all 15 priors NEGATIVE per
   this round's own brief) from an already-committed data column
   (``count_long_short_ratio``) that R-81 fetched but never built a
   feature from. NOT a re-parameterization of R-81: R-81's ``ls_z`` used
   the top-trader ratio ALONE ("is this one trader class crowded");
   this is a two-class DIVERGENCE ("is retail crowded a way smart money
   is not confirming") -- an economically distinct claim.

   Not a duplicate of any entry in ``r81_shared.py``'s own "not a
   duplicate of" list (R-35/R-39 funding-alone flat gate, R-73 DVOL,
   R-53/R-54 VIX/DXY, R-80 meta-labeling, the ruled-out
   order-flow-from-OHLCV line) -- unchanged by this round, not repeated
   here. Not a duplicate of R-88 (Binance taker buy/sell volume ratio,
   an order-FLOW construction) or R-84 (raw traded volume) -- both are
   activity/flow measures with no notion of WHICH class of trader is
   positioned which way; this round's signal is purely a POSITIONING
   (stock, not flow) comparison across two disjoint trader classes.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy code.

   ASSETS: BTC (all 3 of ``r81_shared.STRESS_EPISODES``, metrics coverage
   2020-09-01 onward) and ETH (metrics coverage 2021-12-01 onward, so
   only the 2 episodes whose ONSET postdates that start -- 2022-05
   Terra/Luna and 2022-11 FTX -- have ANY window overlap at all; the
   2021-top/2022-bear episode's search window predates ETH coverage
   entirely and is excluded by construction, not measured and then
   discarded). This departs from R-81's own conservative branch, which
   excluded ETH from Step-A entirely on the grounds that even the two
   later episodes would have "a materially thinner baseline" than BTC's;
   this round tests them anyway (the 14-day z-score baseline this round
   actually uses needs far less history than that concern implied -- by
   Terra's search-window start, 2022-05-04, ETH metrics already has ~5
   months of history, over 10x the baseline window) and reports what
   happens rather than assuming the prior round's more conservative
   choice still applies to a different construction.

   PRIMARY METRIC: ``divergence_z`` (``r115_novel_shared.divergence_z``,
   14-day trailing z-score on each leg, reused unchanged from
   ``r81_shared.crowding_z``'s own window choice). EXTREME THRESHOLD:
   ``|divergence_z| >= 1.5`` -- reusing R-81's own "1.5-sigma" bar
   verbatim (the brief's own suggestion), not re-picked after looking at
   this signal's own distribution.

   EPISODE-LOCAL SEARCH WINDOW: ``[onset - 60 days, onset + 60 days]``,
   fixed a priori -- R-81's conservative branch's own window (R-82 used
   the identical +/-60d convention independently), reused verbatim rather
   than re-derived.

   REFERENCE "DECISION" TIMESTAMP: the ``anchor_majority`` DOWNWARD
   transition (majority decreases) whose timestamp is nearest the
   episode's onset within the search window -- v4's own gate actually
   de-risking, exactly R-81 conservative's flip definition (chosen
   there, and reused here, specifically because an "any-direction"
   nearest-transition rule was shown by that round to pick a spurious
   bullish blip near the 2021 top; using the corrected, already-
   validated rule here rather than re-deriving it and risking the same
   mistake). If no downward transition exists in the window (majority
   already at its floor, or the episode's local trend never triggers a
   NEW down-transition within +/-60d), that episode is UNMATCHED, not
   forced into a spurious comparison.

   CROSSING DEFINITION: the first bar where ``|divergence_z|`` crosses
   INTO >= 1.5 (previous bar below, this bar at/above) whose timestamp is
   nearest the episode's onset within the same window. If no such
   crossing exists (most likely where the top-trader leg is NaN across
   the whole window -- see the coverage-gap diagnostic below, run BEFORE
   this gate and reported in the results section), that episode is
   UNMATCHED.

   LEAD = (flip_time - crossing_time) in days. Positive = the divergence
   signal reached its extreme BEFORE v4's own reaction.

   NULL: ``r81_shared.block_bootstrap_lead_null`` circularly block-shifts
   the LOCAL (episode-window) ``divergence_z`` array (block_days=5,
   n_draws=500, seed=115081 -- fixed now, this round's own seed, never
   R-81's 81 or R-82's 82) and recomputes "crossing nearest the REAL,
   unshifted onset" against the same fixed, real flip time -- identical
   null construction to R-81 conservative's, applied to this round's own
   signal.

   PER-EPISODE PASS CRITERION (this round's brief, verbatim): an episode
   PASSES iff BOTH (a) lead >= 0 (crossed at or before the flip) AND
   (b) lead >= the null's own median (a materially weaker bar than R-81
   conservative's "beats the null's 90th percentile" -- this round's
   brief specifies the median, matching R-82's own weaker Step-A bar, not
   R-81's stricter one; used exactly as specified, not tightened or
   loosened after seeing a number).

   PRE-REGISTERED STOP RULE: among MATCHED episodes only (both a flip and
   a crossing found in-window -- unmatched episodes are reported but
   excluded from the pass-rate denominator, matching R-81 novel's own
   "among matched episodes" accounting), proceed to Step B only if a
   MAJORITY (> half of matched) PASS. If fewer than half of matched
   episodes pass, or zero episodes are matched at all: STOP, report the
   gate result as this branch's whole product, build no strategy code.

3. CONFIGS EVALUATED: 1 -- the single pre-registered ``divergence_z``
   construction (14-day window, 1.5-sigma threshold, +/-60d search) is
   counted as this file's one configuration, following R-81 NOVEL
   branch's own disclosed accounting convention (a fixed measurement
   gate with exactly one frozen construction, distinguished from R-81
   CONSERVATIVE's "0" convention for a signal it treated as a bare
   statistic with no construction choices of its own -- this round's
   signal, like R-81's cascade trigger, bundles several fixed design
   choices into one named construction, so it is counted as 1, stated
   here rather than forced into false consistency with the other
   branch's own accounting, exactly as R-81's own ledger entry did).

4. WHAT WOULD MAKE STEP A FAIL, named now: the same failure mode as
   every one of the 15 prior INFO-axis attempts -- the divergence
   extreme is reached AFTER v4's own reaction (its slow anchor vote
   already "knows" by the time retail/smart-money positioning shows an
   extreme split), or a positive lead is indistinguishable from an
   arbitrary time-shift of the same series (generic autocorrelation, not
   a genuine early-warning property). A further, DATA-SPECIFIC failure
   mode named now, before running anything: because ``divergence_z``
   needs BOTH legs, it is NaN wherever EITHER leg is NaN -- so even
   though this file's own coverage-gap check (see below) may show
   ``count_long_short_ratio`` is nearly complete, the OTHER leg
   (``sum_toptrader_long_short_ratio``, R-81's already-documented ~38%/
   81% BTC/ETH gap) can still starve ``divergence_z`` of data across
   most of an episode's own window, exactly the FTX-episode confound
   R-81 diagnosed for its own signal -- disclosed here as a known risk
   to this round's OWN construction as well, not just R-81's, since it
   is inherited by construction, not avoided by adding a second column.

Run: ``python experiments/r115_novel_smartmoney_divergence.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

import experiments.r115_novel_shared as shared  # noqa: E402

DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------- consts
DIVERGENCE_THRESH = 1.5   # fixed a priori, reused from R-81's |ls_z| bar
WINDOW_DAYS = 60           # +/- days around each episode's onset, fixed a priori
N_DRAWS = 500               # block-bootstrap null draws, fixed a priori
BLOCK_DAYS = 5              # null block length in days, fixed a priori
NULL_SEED = 115081          # fixed once, this round's own seed

RAW_COLS_FOR_GAP_CHECK = ["count_long_short_ratio", "sum_toptrader_long_short_ratio"]

# Which of r81_shared.STRESS_EPISODES are testable per asset. ETH metrics
# start 2021-12-01 -- episode 0 ("2021-top / 2022-bear transition", onset
# 2021-11-10) predates that entirely, excluded by construction (not
# measured then discarded).
ASSET_EPISODES = {
    "BTC": list(range(len(shared.STRESS_EPISODES))),
    "ETH": [i for i, (_, onset) in enumerate(shared.STRESS_EPISODES)
            if pd.Timestamp(onset, tz="UTC") - pd.Timedelta(days=WINDOW_DAYS)
            >= pd.Timestamp(shared.METRICS_START["ETH"], tz="UTC")],
}


# ---------------------------------------------------------------- holdout guard

def assert_no_holdout(df: pd.DataFrame, label: str = "") -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before OOS_START."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(shared.OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read{' (' + label + ')' if label else ''}: "
        f"max timestamp {max_ts} >= {shared.OOS_START}. "
        "This file must never read data on or after the holdout start.")


# --------------------------------------------------------------------- data

def load_bars() -> pd.DataFrame:
    """BTC spot bars, truncated strictly before OOS_START at load time.
    ``anchor_majority`` is computed on these -- BTC bars, not per-asset --
    matching r81_shared/r81_conservative's own convention: `kelly_regime_v4`
    is a BTC strategy, so the "v4's own reaction" reference timestamp is
    always BTC's vote, for both the BTC and the ETH divergence gates (an
    ETH-native anchor vote is not this project's registered strategy)."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(shared.OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df, "bars")
    print(f"BTC bars ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {shared.OOS_START})")
    return df


# ------------------------------------------------------------- flip / crossing

def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp) -> pd.Timestamp | None:
    """Nearest-to-onset DOWNWARD transition of `series` within `window`.
    Identical rule to r81_conservative_crowding_vote.py's own (validated)
    `direction="down"` case -- see this file's pre-registration section
    for why that rule, not "any direction", is used."""
    vals = series.reindex(window).to_numpy()
    changed = np.zeros(len(vals), dtype=bool)
    changed[1:] = vals[1:] < vals[:-1]
    idx = np.where(changed)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def nearest_crossing(z: pd.Series, window: pd.DatetimeIndex,
                      onset: pd.Timestamp, thresh: float = DIVERGENCE_THRESH) -> pd.Timestamp | None:
    """Nearest-to-onset first-crossing of |z| INTO >= thresh within
    `window`. NaN bars are treated as "not above threshold" (a missing
    reading cannot itself be a crossing) -- the same honest, no-special-
    casing treatment r81_conservative_crowding_vote.py gave its own
    heavily-gapped `ls_z`."""
    vals = z.reindex(window).to_numpy()
    above = np.abs(vals) >= thresh   # NaN >= thresh -> False, by design
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = above[1:] & ~above[:-1]
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars_index: pd.DatetimeIndex, onset_str: str) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=WINDOW_DAYS)
    hi = onset + pd.Timedelta(days=WINDOW_DAYS)
    window = bars_index[(bars_index >= lo) & (bars_index <= hi)]
    return onset, window


# --------------------------------------------------------------------- null

def episode_null_leads(div_z: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, flip_time: pd.Timestamp) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode -- circularly
    shift the LOCAL divergence_z array and recompute "crossing nearest the
    real, unshifted onset" against the fixed, real flip_time."""
    local = div_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = shared.block_bootstrap_lead_null(
        event_offsets_days=np.array([0.0]), n_bars=n_bars,
        block_days=BLOCK_DAYS, n_draws=N_DRAWS, seed=NULL_SEED)

    leads = np.full(N_DRAWS, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        above = np.abs(shifted) >= DIVERGENCE_THRESH
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = above[1:] & ~above[:-1]
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        cross_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - cross_time).total_seconds() / 86400.0
    return leads


# ------------------------------------------------------------ causality probe

def build_divergence_from_df(bars: pd.DataFrame, metrics_full: pd.DataFrame) -> np.ndarray:
    """End-to-end load-to-feature pipeline for the truncation probe:
    ``metrics_full`` (already loaded once, independent of ``bars``' own
    length) aligned/derived onto ``bars.index`` only, so a shorter
    ``bars`` frame can never change an earlier bar's divergence_z if the
    pipeline is truly causal."""
    feats = shared.divergence_z(metrics_full, bars)
    return feats["divergence_z"].to_numpy()


# --------------------------------------------------------------------- gate

def gate_for_asset(asset: str, bars: pd.DataFrame, majority: pd.Series) -> list[dict]:
    print("\n" + "-" * 78)
    print(f"{asset}: coverage-gap diagnostic + Step-A gate")
    print("-" * 78)

    metrics = shared.load_metrics_truncated(DATA_DIR, asset)
    if metrics is None:
        print(f"  {asset} metrics file missing -- cannot run this gate.")
        return []
    assert_no_holdout(metrics, f"{asset} metrics")
    print(f"  metrics coverage: {metrics.index.min()} -> {metrics.index.max()} "
          f"({len(metrics):,} rows)")

    gaps_whole = shared.coverage_gap_pct(metrics, RAW_COLS_FOR_GAP_CHECK)
    print(f"  whole-window NaN%: count_long_short_ratio="
          f"{gaps_whole['count_long_short_ratio']:.2f}%  "
          f"sum_toptrader_long_short_ratio="
          f"{gaps_whole['sum_toptrader_long_short_ratio']:.2f}%")

    feats = shared.divergence_z(metrics, bars)
    assert_no_holdout(feats, f"{asset} divergence_z")
    div_z = feats["divergence_z"]

    results = []
    for ep_idx in ASSET_EPISODES[asset]:
        label, onset_str = shared.STRESS_EPISODES[ep_idx]
        onset, window = episode_window(bars.index, onset_str)

        gap_win = shared.episode_window_gap_pct(metrics, RAW_COLS_FOR_GAP_CHECK,
                                                 onset_str, WINDOW_DAYS)
        print(f"\n  [{asset} / {label}] onset={onset_str}  window=+/-{WINDOW_DAYS}d")
        if gap_win is None:
            print(f"    window predates {asset} metrics coverage entirely -- "
                  f"EXCLUDED (not measured, per pre-registration).")
            continue
        print(f"    in-window NaN%: count_long_short_ratio="
              f"{gap_win['count_long_short_ratio']:.2f}%  "
              f"sum_toptrader_long_short_ratio="
              f"{gap_win['sum_toptrader_long_short_ratio']:.2f}%")

        if len(window) == 0:
            print(f"    window has ZERO bars in the bar frame -- FAIL by construction.")
            results.append(dict(asset=asset, label=label, onset=onset_str, matched=False,
                                 lead=float("nan"), pass_ep=False))
            continue

        flip_time = nearest_transition(majority, window, onset)
        cross_time = nearest_crossing(div_z, window, onset)

        if flip_time is None or cross_time is None:
            reason = ("no anchor-gate downward transition" if flip_time is None
                       else "no |divergence_z|>=1.5 crossing")
            print(f"    {reason} found in window -- UNMATCHED.")
            results.append(dict(asset=asset, label=label, onset=onset_str, matched=False,
                                 lead=float("nan"), pass_ep=False,
                                 flip_time=flip_time, cross_time=cross_time))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(div_z, window, onset, flip_time)
        valid_null = null_leads[~np.isnan(null_leads)]
        null_median = float(np.median(valid_null)) if len(valid_null) else float("nan")
        null_p90 = float(np.percentile(valid_null, 90)) if len(valid_null) else float("nan")
        pass_ep = (lead >= 0) and (not np.isnan(null_median)) and (lead >= null_median)

        local_majority = majority.reindex(window).to_numpy()
        flip_pos = int(window.get_indexer([flip_time])[0])
        prev_val = local_majority[flip_pos - 1] if flip_pos > 0 else float("nan")
        new_val = local_majority[flip_pos]

        print(f"    anchor-gate nearest downward transition: {flip_time}  "
              f"(majority {prev_val:.3f} -> {new_val:.3f})")
        print(f"    divergence_z nearest crossing (|z|>={DIVERGENCE_THRESH}): {cross_time}")
        print(f"    LEAD = {lead:+.2f} days  "
              f"({'divergence LED' if lead >= 0 else 'divergence LAGGED'})")
        print(f"    null ({N_DRAWS} draws, valid={len(valid_null)}): "
              f"median={null_median:+.2f}d  p90={null_p90:+.2f}d")
        print(f"    PASS (lead>=0 AND lead>=null median): {pass_ep}")

        results.append(dict(asset=asset, label=label, onset=onset_str, matched=True,
                             lead=lead, pass_ep=pass_ep, null_median=null_median,
                             null_p90=null_p90, flip_time=flip_time, cross_time=cross_time))

    return results


def main() -> None:
    print("=" * 78)
    print("R-115 NOVEL: retail-vs-smart-money divergence_z -- Step A gate")
    print("=" * 78)

    bars = load_bars()
    majority = shared.anchor_majority(bars)
    assert_no_holdout(majority.to_frame(), "anchor_majority")

    all_results: list[dict] = []
    max_ts_seen = bars.index.max()
    for asset in ("BTC", "ETH"):
        results = gate_for_asset(asset, bars, majority)
        all_results.extend(results)

    # -------------------------------------------------- causality probe
    print("\n" + "-" * 78)
    print("Causal truncation probe: load_metrics_truncated -> divergence_z")
    print("-" * 78)
    metrics_btc = shared.load_metrics_truncated(DATA_DIR, "BTC")
    cutoff_end = pd.Timestamp(shared.METRICS_END, tz=bars.index.tz) + pd.Timedelta(days=1)
    bars_win = bars.loc[(bars.index >= metrics_btc.index.min()) & (bars.index < cutoff_end)].copy()
    assert_no_holdout(bars_win, "probe bars_win")

    for check_at in (50_000, 100_000):
        if check_at >= len(bars_win) - 20_000:
            continue
        ok = shared.truncation_causality_probe(
            lambda df: build_divergence_from_df(df, metrics_btc), bars_win, check_at)
        print(f"  check_at={check_at:>7d} ({bars_win.index[check_at]}): "
              f"{'PASS (causal)' if ok else 'FAIL (LOOKAHEAD)'}")
        assert ok, "divergence_z pipeline is not causal -- stop, do not trust Step A"

    # ------------------------------------------------------------ summary
    matched = [r for r in all_results if r["matched"]]
    n_matched = len(matched)
    n_pass = sum(1 for r in matched if r["pass_ep"])
    gate_pass = n_matched > 0 and n_pass > n_matched / 2.0

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (frozen before any number above was computed):")
    print("  an episode PASSES iff (a) lead>=0 AND (b) lead >= its own")
    print(f"  {N_DRAWS}-draw block-bootstrap null's MEDIAN.")
    print("  proceed to Step B only if a MAJORITY of MATCHED episodes PASS.")
    print("=" * 78)
    print(f"\n{'asset':6s} {'episode':38s} {'matched':8s} {'lead(d)':>10s} "
          f"{'null_med':>10s} {'pass':>6s}")
    for r in all_results:
        if r["matched"]:
            print(f"{r['asset']:6s} {r['label']:38s} {'yes':8s} {r['lead']:>+10.2f} "
                  f"{r['null_median']:>+10.2f} {str(r['pass_ep']):>6s}")
        else:
            print(f"{r['asset']:6s} {r['label']:38s} {'no':8s} {'--':>10s} {'--':>10s} {'--':>6s}")

    print(f"\nMatched episodes: {n_matched}   Passing: {n_pass}   "
          f"Majority bar: > {n_matched / 2.0:.1f}")
    print(f"GATE VERDICT: {'PASS -> proceed to Step B' if gate_pass else 'FAIL -> STOP, no strategy built'}")

    print("\nConfigs evaluated: 1 (the single pre-registered divergence_z "
          "construction; the block-bootstrap null and causality probe are "
          "diagnostics of that one construction, per this round's own "
          "accounting convention -- see module docstring section 3).")

    print(f"\nmax timestamp read anywhere in this session: {max_ts_seen}  "
          f"(< {shared.OOS_START})")

    if not gate_pass:
        print("\nStopping here per the pre-registered rule. No strategy code is built.")
        return

    print("\n" + "=" * 78)
    print("Gate PASSED. Step B would build the confirming-vote/strategy code")
    print("here. Not reached in this run.")
    print("=" * 78)


if __name__ == "__main__":
    main()
