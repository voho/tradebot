#!/usr/bin/env python
"""R-81 (novel branch): does a real-time crowding fingerprint of an
ACTIVE deleveraging cascade lead `kelly_regime_v4`'s own slow anchor
vote at this project's 3 dated stress episodes -- the mandatory
measurement gate, run BEFORE any strategy code (per the brief, and per
this project's own R-53/R-73/R-74/R-75 novel-branch discipline of
"measure the lead time before building anything").

=============================================================================
MECHANISM AND CONSTRAINT (ROUTINE.md step 1-2, written before any number ran)
=============================================================================

One sentence: a sharp DROP in open interest (`oi_chg_z` strongly
negative) coinciding with the top-trader long/short ratio (`ls_z`)
snapping back toward neutral from a recent extreme is the real-time
fingerprint of an ACTIVE, already-in-progress forced-deleveraging
cascade (evidence the cascade IS happening, not a forecast that one
will), so it is a candidate event-triggered override that could
shorten `kelly_regime_v4`'s exit lag specifically during the rare bars
where a cascade is already unwinding -- reacting to real-time evidence,
rather than waiting for the 20/40/80-day anchor vote to eventually
catch up.

Constraint attacked: ERR / N~=3 (`docs/LEDGER.md`'s standing diagnosis).
This project's whole measured edge concentrates in a handful of
regime-transition events (B-38/R-78's own framing), and v4 currently has
no mechanism that reacts to evidence a cascade is UNDERWAY -- only to its
own slow, latched anchor vote. Shrinking the reaction lag at exactly
those events is a direct attack on the effective-sample-size problem,
not "another indicator" claiming to predict everyday returns (which
attacks neither INFO/N~=3/ERR per `docs/ROUTINE.md` step 1's own filter).

Not a duplicate of R-56/R-77 (`docs/LEDGER.md`): those rounds changed HOW
an already-decided order fills (patient-limit posting, N-bar taker
fallback) -- pure execution mechanics on the SAME decision timing v4
already has. This round changes WHEN the decision itself fires, using
independent real-time market-structure evidence (Binance futures
positioning) that R-56/R-77 never touched. Confirmed by re-reading both
ledger entries in full before writing any code here (see this session's
transcript): R-56's own two branches and R-77's own two branches are all
scoped to fill mechanics (limit-order patience N, fill-probability
models, adaptive posting urgency) -- none of the four branches across
those two rounds ever proposed changing the SIGNAL that decides direction
or timing of a rebalance.

Not a duplicate of any other row in `r81_shared.py`'s own "not a
duplicate of" list (R-35/R-39 funding-alone flat gate, R-73 DVOL,
R-53/R-54 VIX/DXY, R-80 meta-labeling, the ruled-out
order-flow-from-OHLCV line) -- see that module's docstring, not repeated
here.

=============================================================================
STEP A -- the mandatory measurement gate (frozen BEFORE any number ran)
=============================================================================

Data: BTC only. `r81_shared.STRESS_EPISODES` gives 3 dated stress
episodes inside the Binance metrics feed's coverage window
(2020-09-01 -> `INNER_VAL_END`=2022-12-31). ETH's metrics history only
starts 2021-12-01 -- too short to give ANY of the 3 episodes a
meaningful pre-episode baseline (episode 1's onset, 2021-11-10, predates
ETH metrics coverage entirely; episodes 2-3 would have <6 months of
history to build a 14-day rolling z-score baseline against, materially
thinner than BTC's). Stated here, explicitly, before computing anything:
**this Step-A gate is BTC-only.** No ETH Step-A number is computed or
reported -- there is no honest way to compute one.

Reference "decision" timestamp per episode: `anchor_majority()`'s own
downward flip (majority crosses from >=0.5, net-bullish-or-split, to
<0.5, net-bearish) -- the exact timestamp v4's own 3-anchor vote actually
acts on to start de-risking. If the majority is ALREADY <0.5 at the start
of an episode's search window (the vote had already flipped bearish
before this specific episode, e.g. from an earlier stage of the same
bear market -- the exact confound R-73 named for Terra/Luna), that
episode is marked **already-bearish-going-in / unmatched**, not forced
into a spurious match.

Cascade-signature construction (pre-registered, BEFORE any number was
computed -- these exact thresholds, chosen from the mechanism's own
plain-language description, are never retuned after seeing a result):

    recent_ls_extreme[i] = rolling max of |ls_z| over the trailing
                            RECENT_WINDOW_BARS bars, STRICTLY BEFORE bar i
                            (`.shift(1).rolling(...).max()`)
    snapback[i]           = recent_ls_extreme[i] >= LS_EXTREME_THRESH
                             AND |ls_z[i]| <= SNAPBACK_FRAC * recent_ls_extreme[i]
    cascade_trigger[i]    = snapback[i] AND oi_chg_z[i] <= OI_THRESH

    LS_EXTREME_THRESH  = 1.5   (a genuinely stretched long/short ratio)
    SNAPBACK_FRAC      = 0.5   (retraced to under half that extreme)
    OI_THRESH          = -1.5  (open interest contracting sharply vs. its
                                own trailing baseline -- positions closing)
    RECENT_WINDOW_BARS = 288   (1 day, `BARS_PER_DAY`)

Search window per episode: [onset - SEARCH_PRE_DAYS, onset +
SEARCH_POST_DAYS], capped at `r81_shared.METRICS_END` (never reads the
holdout). SEARCH_PRE_DAYS=5, SEARCH_POST_DAYS=90 -- both fixed before any
number was computed, generous enough to give the anchor vote's own
20/40/80-day rolling-mean construction room to actually flip, per its own
documented multi-day-to-multi-week lag.

Lead = (vote-flip bar position) - (first-cascade-trigger bar position),
in bars, for episodes where BOTH are found in-window ("matched"
episodes). Null: `r81_shared.block_bootstrap_lead_null` draws
`N_NULL_DRAWS` independent circular rotations of the (ls_z, oi_chg_z)
pair over the metrics-covered bar range (BLOCK_DAYS=5, seed=81081);
`cascade_trigger` is recomputed from each rotated pair and re-matched
against the SAME real, fixed vote-flip dates, giving a null distribution
of "mean lead across matched episodes" under random phase alignment.

**PRE-REGISTERED STOP RULE (frozen before any Step-A number was
computed).** Proceed to Step B only if ALL FOUR hold:

  (1) at least 2 of the 3 episodes are "matched" (both a cascade trigger
      and a genuine vote flip found in-window -- not already-bearish);
  (2) among matched episodes, a majority (>=2) show the cascade trigger
      firing BEFORE the vote flip (lead > 0 bars);
  (3) the mean lead across those leading episodes is >= MIN_LEAD_BARS=12
      (a floor against a same-bar/next-bar coincidence being read as
      "leading" -- the literal reading of the brief's "several bars",
      deliberately a low bar since the interesting test is (4), not this
      one);
  (4) the observed mean lead across ALL matched episodes exceeds the
      block-bootstrap null's one-sided 95th percentile (empirical p<0.05
      -- this isn't just noise).

If this fails: STOP, do not build a strategy, report the gate result as
this branch's whole product -- a complete, valid negative result for this
project, exactly like R-73's and R-75's novel branches. This is the same
failure mode that killed R-53 (VIX/DXY lagged), R-73 (DVOL lagged), R-74
(MVRV lagged): a signal merely confirming a crash already fully priced in
by the existing gate, tested here instead of assumed.

Run: ``python experiments/r81_novel_cascade_exit.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r81_shared as shared  # noqa: E402
from scripts.experiment import DF, OOS_START  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402

DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------- consts
# Cascade-signature thresholds -- pre-registered before any number ran.
LS_EXTREME_THRESH = 1.5
SNAPBACK_FRAC = 0.5
OI_THRESH = -1.5
RECENT_WINDOW_BARS = 1 * BARS_PER_DAY

# Search window per episode -- pre-registered before any number ran.
SEARCH_PRE_DAYS = 5
SEARCH_POST_DAYS = 90

# Pre-registered stop-rule parameters.
MIN_LEAD_BARS = 12
MAJORITY_REQUIRED = 2          # of 3 episodes must be matched
LEAD_MAJORITY_REQUIRED = 2     # of matched episodes must show positive lead

# Block-bootstrap null.
BLOCK_DAYS = 5
N_NULL_DRAWS = 1000
NULL_SEED = 81081


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Never read a bar on/after OOS_START -- mirrors r79/r80's own guard."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    assert df.index.max() < cutoff, (
        f"holdout bar read: max timestamp {df.index.max()} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# ======================================================================
# Cascade-signature construction (causal: shift(1) + backward rolling only)
# ======================================================================

def cascade_trigger(ls_z: pd.Series, oi_chg_z: pd.Series) -> pd.Series:
    """Boolean cascade-in-progress fingerprint, per the pre-registered
    construction in the module docstring. Causal by construction: the
    only lookahead-sensitive piece (`recent_ls_extreme`) is explicitly
    shifted by 1 bar before its rolling max, so bar i's trigger depends
    only on ls_z/oi_chg_z at bars <= i.
    """
    recent_ls_extreme = ls_z.abs().shift(1).rolling(
        RECENT_WINDOW_BARS, min_periods=RECENT_WINDOW_BARS // 4).max()
    snapback = (recent_ls_extreme >= LS_EXTREME_THRESH) & (
        ls_z.abs() <= SNAPBACK_FRAC * recent_ls_extreme)
    trig = snapback & (oi_chg_z <= OI_THRESH)
    return trig.fillna(False)


def build_cascade_trigger_from_df(df: pd.DataFrame) -> np.ndarray:
    """End-to-end, load-metrics-to-trigger pipeline, for the causality
    truncation probe. Loads the full (pre-truncated-to-METRICS_END)
    metrics file fresh each call -- independent of `df`'s own length --
    and aligns/derives everything from `df.index` only, so a shorter
    `df` can never change an earlier bar's trigger value if the pipeline
    is truly causal.
    """
    metrics = shared.load_crowding_inputs(DATA_DIR, "BTC")
    feats = shared.crowding_z(metrics, df)
    trig = cascade_trigger(feats["ls_z"], feats["oi_chg_z"])
    return trig.to_numpy().astype(float)


# ======================================================================
# Reference "decision" timestamp: anchor_majority's own downward flip
# ======================================================================

def downward_flip_mask(majority: pd.Series) -> pd.Series:
    """True at bars where the 3-anchor majority crosses from >=0.5 to
    <0.5 -- the exact bar v4's own vote turns net-bearish."""
    prev = majority.shift(1)
    return (prev >= 0.5) & (majority < 0.5)


# ======================================================================
# Per-episode matching
# ======================================================================

def find_first_true(mask: np.ndarray, lo: int, hi: int) -> int | None:
    """First position in mask[lo:hi] (inclusive lo, exclusive hi) that is
    True, or None."""
    seg = mask[lo:hi]
    hit = np.flatnonzero(seg)
    if len(hit) == 0:
        return None
    return lo + int(hit[0])


def match_episode(index: pd.DatetimeIndex, trig: np.ndarray, majority: np.ndarray,
                   onset: str, metrics_end: str) -> dict:
    onset_ts = pd.Timestamp(onset, tz="UTC")
    win_start = onset_ts - pd.Timedelta(days=SEARCH_PRE_DAYS)
    win_end = min(onset_ts + pd.Timedelta(days=SEARCH_POST_DAYS),
                  pd.Timestamp(metrics_end, tz="UTC") + pd.Timedelta(days=1))

    lo = int(index.searchsorted(win_start))
    hi = int(index.searchsorted(win_end))
    lo = max(lo, 0)
    hi = min(hi, len(index))

    result = {
        "onset": onset, "window": (win_start, win_end),
        "window_bars": hi - lo, "trigger_pos": None, "trigger_ts": None,
        "flip_pos": None, "flip_ts": None, "already_bearish": False,
        "matched": False, "lead_bars": None,
    }
    if hi <= lo:
        return result

    trig_pos = find_first_true(trig, lo, hi)
    result["trigger_pos"] = trig_pos
    result["trigger_ts"] = index[trig_pos] if trig_pos is not None else None

    if majority[lo] < 0.5:
        result["already_bearish"] = True
        return result

    flip_mask = np.zeros(len(majority), dtype=bool)
    flip_mask[1:] = (majority[:-1] >= 0.5) & (majority[1:] < 0.5)
    flip_pos = find_first_true(flip_mask, lo, hi)
    result["flip_pos"] = flip_pos
    result["flip_ts"] = index[flip_pos] if flip_pos is not None else None

    if trig_pos is not None and flip_pos is not None:
        result["matched"] = True
        result["lead_bars"] = flip_pos - trig_pos  # positive = trigger leads

    return result


def mean_lead_for_draw(index: pd.DatetimeIndex, trig: np.ndarray, majority: np.ndarray,
                        episodes: list, metrics_end: str) -> float | None:
    """Mean lead across episodes that are matched under the REAL majority
    (fixed) but a possibly-rotated `trig` array -- used both for the real
    measurement and for each null draw."""
    leads = []
    for label, onset in episodes:
        r = match_episode(index, trig, majority, onset, metrics_end)
        if r["matched"]:
            leads.append(r["lead_bars"])
    if not leads:
        return None
    return float(np.mean(leads))


def main() -> None:
    print("=" * 78)
    print("R-81 novel branch: Step A measurement gate (cascade-in-progress")
    print("fast-exit override) -- BTC only (see module docstring for why)")
    print("=" * 78)

    df_full = DF.copy()
    assert_no_holdout(df_full[df_full.index <= pd.Timestamp(shared.METRICS_END, tz=df_full.index.tz)])

    metrics = shared.load_crowding_inputs(DATA_DIR, "BTC")
    assert metrics is not None, "BTC metrics file missing -- cannot run this gate"
    print(f"\nBTC metrics coverage: {metrics.index.min()} -> {metrics.index.max()} "
          f"({len(metrics):,} rows)")

    # Restrict the bar frame to the metrics-covered window (+ enough lead-in
    # for the 14-day rolling z-score baseline to warm up) so the null
    # rotation below draws only from bars where crowding data actually
    # exists -- rotating in the pre-2020-09 all-NaN region would understate
    # the null's power. Never touches the holdout: capped at METRICS_END.
    cutoff_end = pd.Timestamp(shared.METRICS_END, tz=df_full.index.tz) + pd.Timedelta(days=1)
    df_win = df_full.loc[(df_full.index >= metrics.index.min()) & (df_full.index < cutoff_end)].copy()
    assert_no_holdout(df_win)
    print(f"Working window (bars): {df_win.index.min()} -> {df_win.index.max()} "
          f"({len(df_win):,} bars)")

    feats = shared.crowding_z(metrics, df_win, window_days=14)
    valid_from = feats.dropna().index.min()
    print(f"crowding_z first fully-warmed-up bar: {valid_from}")

    # anchor_majority needs the FULL price history for its rolling anchors
    # (80-day slowest horizon) -- computed on df_full, then sliced onto
    # df_win's index. df_full itself is never read past METRICS_END below.
    majority_full = shared.anchor_majority(df_full)
    majority = majority_full.loc[df_win.index].to_numpy()

    trig_series = cascade_trigger(feats["ls_z"], feats["oi_chg_z"])
    trig = trig_series.to_numpy()
    print(f"cascade_trigger fires on {trig.sum():,} / {len(trig):,} bars "
          f"({trig.mean() * 100:.3f}%) over the working window")

    # ---- causality probe on the full metrics-to-trigger pipeline --------
    print("\n--- causality probe: load_crowding_inputs -> crowding_z -> "
          "cascade_trigger pipeline, 2 check points ---")
    for check_at in (100_000, 200_000):
        if check_at >= len(df_win) - 20_000:
            continue
        ok = shared.truncation_causality_probe(build_cascade_trigger_from_df, df_win, check_at)
        print(f"  check_at={check_at:>7d} ({df_win.index[check_at]}): "
              f"{'PASS (causal)' if ok else 'FAIL (LOOKAHEAD)'}")
        assert ok, "cascade_trigger pipeline is not causal -- stop, do not trust Step A"

    # ---- per-episode matching --------------------------------------------
    print("\n--- per-episode matching (BTC only) ---")
    episode_results = []
    for label, onset in shared.STRESS_EPISODES:
        r = match_episode(df_win.index, trig, majority, onset, shared.METRICS_END)
        episode_results.append((label, r))
        print(f"\n  [{label}] onset={onset}")
        print(f"    search window: {r['window'][0].date()} -> {r['window'][1].date()} "
              f"({r['window_bars']:,} bars)")
        if r["already_bearish"]:
            print("    ALREADY-BEARISH-GOING-IN: majority < 0.5 at window start -- "
                  "no genuine flip to reference for this episode (R-73's Terra confound).")
            continue
        print(f"    cascade trigger: "
              f"{'first at ' + str(r['trigger_ts']) if r['trigger_ts'] is not None else 'NOT FOUND in window'}")
        print(f"    vote downward flip: "
              f"{'first at ' + str(r['flip_ts']) if r['flip_ts'] is not None else 'NOT FOUND in window'}")
        if r["matched"]:
            lead_bars = r["lead_bars"]
            lead_days = lead_bars / BARS_PER_DAY
            print(f"    MATCHED: lead = {lead_bars:+d} bars ({lead_days:+.2f} days) "
                  f"[positive = cascade signal leads the vote]")
        else:
            print("    UNMATCHED: one or both of trigger/flip not found in window.")

    matched = [(lbl, r) for lbl, r in episode_results if r["matched"]]
    n_matched = len(matched)
    leading = [r["lead_bars"] for lbl, r in matched if r["lead_bars"] > 0]
    n_leading = len(leading)
    mean_lead_leading = float(np.mean(leading)) if leading else None
    mean_lead_matched = (float(np.mean([r["lead_bars"] for _, r in matched]))
                         if matched else None)

    print("\n--- summary ---")
    print(f"  episodes matched: {n_matched}/3")
    print(f"  matched episodes with positive lead (signal before flip): {n_leading}/{n_matched if n_matched else 0}")
    if mean_lead_matched is not None:
        print(f"  mean lead across ALL matched episodes: {mean_lead_matched:+.1f} bars "
              f"({mean_lead_matched / BARS_PER_DAY:+.2f} days)")
    if mean_lead_leading is not None:
        print(f"  mean lead across LEADING matched episodes only: {mean_lead_leading:+.1f} bars "
              f"({mean_lead_leading / BARS_PER_DAY:+.2f} days)")

    crit1 = n_matched >= MAJORITY_REQUIRED
    crit2 = n_leading >= LEAD_MAJORITY_REQUIRED
    crit3 = (mean_lead_leading is not None) and (mean_lead_leading >= MIN_LEAD_BARS)

    # ---- block-bootstrap null ---------------------------------------------
    print(f"\n--- block-bootstrap null ({N_NULL_DRAWS} draws, block={BLOCK_DAYS}d, "
          f"seed={NULL_SEED}) ---")
    ls_arr = feats["ls_z"].to_numpy()
    oi_arr = feats["oi_chg_z"].to_numpy()
    n_bars = len(df_win)
    event_offsets_days = np.array([
        (pd.Timestamp(onset, tz="UTC") - df_win.index[0]).total_seconds() / 86400.0
        for _, onset in shared.STRESS_EPISODES
    ])
    draws = shared.block_bootstrap_lead_null(event_offsets_days, n_bars, BLOCK_DAYS,
                                              N_NULL_DRAWS, NULL_SEED)

    null_means = []
    for shift_idx in draws:
        ls_rot = pd.Series(ls_arr[shift_idx], index=df_win.index)
        oi_rot = pd.Series(oi_arr[shift_idx], index=df_win.index)
        trig_rot = cascade_trigger(ls_rot, oi_rot).to_numpy()
        m = mean_lead_for_draw(df_win.index, trig_rot, majority, shared.STRESS_EPISODES,
                                shared.METRICS_END)
        if m is not None:
            null_means.append(m)
    null_means = np.array(null_means)
    n_null_valid = len(null_means)
    print(f"  null draws with >=1 matched episode: {n_null_valid}/{N_NULL_DRAWS}")

    if n_null_valid > 0 and mean_lead_matched is not None:
        null_p95 = float(np.percentile(null_means, 95))
        null_mean = float(null_means.mean())
        null_std = float(null_means.std())
        pval = float((null_means >= mean_lead_matched).mean())
        print(f"  null mean-lead: mean={null_mean:+.1f} std={null_std:.1f} p95={null_p95:+.1f} bars")
        print(f"  observed mean lead (matched episodes) = {mean_lead_matched:+.1f} bars vs "
              f"null p95={null_p95:+.1f} bars -> "
              f"{'EXCEEDS' if mean_lead_matched > null_p95 else 'does NOT exceed'}, "
              f"empirical p={pval:.4f}")
        crit4 = mean_lead_matched > null_p95
    else:
        print("  null could not be evaluated (no matched episodes in observed data or "
              "no null draw ever matched) -- criterion (4) FAILS by construction.")
        null_p95 = None
        pval = None
        crit4 = False

    gate_pass = crit1 and crit2 and crit3 and crit4

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (frozen before any Step-A number was computed):")
    print(f"  (1) >= {MAJORITY_REQUIRED}/3 episodes matched:                  "
          f"{'PASS' if crit1 else 'FAIL'} ({n_matched}/3)")
    print(f"  (2) >= {LEAD_MAJORITY_REQUIRED} matched episodes show positive lead: "
          f"{'PASS' if crit2 else 'FAIL'} ({n_leading}/{n_matched if n_matched else 0})")
    print(f"  (3) mean lead (leading episodes) >= {MIN_LEAD_BARS} bars:  "
          f"{'PASS' if crit3 else 'FAIL'} "
          f"({mean_lead_leading if mean_lead_leading is not None else 'n/a'})")
    print(f"  (4) observed mean lead exceeds block-bootstrap null p95:  "
          f"{'PASS' if crit4 else 'FAIL'}")
    print(f"  GATE: {'PASS -> proceed to Step B' if gate_pass else 'FAIL -> STOP, report negative'}")
    print("=" * 78)

    print("\nConfigs evaluated: 1 (the single pre-registered cascade-signature "
          "construction; the block-bootstrap null and causality probes are "
          "diagnostics of that one construction, not additional swept "
          "configurations, per this project's Step-A accounting convention "
          "-- R-53/R-73/R-74/R-75/R-79's own novel-branch gates count the same way).")

    if not gate_pass:
        print("\nStopping here per the pre-registered rule. No strategy code is built.")
        return

    print("\n" + "=" * 78)
    print("Gate PASSED. Step B would build the exit override here. See the")
    print("report accompanying this file for what Step B, if reached, contains.")
    print("=" * 78)


if __name__ == "__main__":
    main()
