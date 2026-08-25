#!/usr/bin/env python
"""R-143 NOVEL branch: does the six-episode Step-A detection-lag gate --
the instrument that has now closed EIGHT structurally distinct
regime-timing detectors (HMM R-01, BOCPD R-82, Kalman LLT R-83, CSD R-85,
transfer entropy R-86, Hawkes R-96, CUSUM R-139, LPPLS R-141), each at
0-2/6 -- generalize past the six dates it was built from?

This is NOT a ninth detector. Nothing new is estimated. Two ALREADY-CLOSED
detectors (BOCPD, the best prior score at 2/6; CUSUM, R-139) plus v4's own
incumbent anchor vote are re-run, unmodified and re-imported rather than
re-derived, against an EXTENDED episode calendar: `r100_shared`'s standing
six PLUS three independently news-dated pre-2017 episodes that the gate has
never seen, made reachable for the first time by R-143's backward data
extension (`r143_shared.load_extended_btc_spot`, BTC 5m spot back to
2013-01-01).

R-85's own closing diagnosis is the hypothesis under test: "the gate's own
six episodes are dominated by sudden, news- or liquidation-driven shocks;
the one slow-building exception (2018) is the only episode any of the four
mechanisms has ever detected early". If that composition -- not the
detectors -- is why nothing clears the gate, then adding episodes should
MOVE at least one detector's per-episode verdicts. If the extended calendar
leaves every verdict where it was, the gate's negative verdicts are a
property of the detectors, not of the six dates, and eight closures stand
on firmer ground than they did.

=====================================================================
GUARDRAIL 7 (r143_shared.py, frozen before this file was written)
=====================================================================
"New episodes in the extended calendar (novel branch) must be
independently, publicly documented market events dated by external
reporting (a news event, an exchange collapse, a regulatory announcement)
-- never a date selected by inspecting where a detector or the price series
itself has an extremum."

Every onset date below was fixed from external reporting BEFORE any price
series, chart, extremum or detector output for that window was inspected.
The date-selection procedure was: search for publicly documented
exchange-collapse / hack / regulatory events in 2014-2016, take the date of
the FIRST public announcement of the event (the least discretionary choice
available -- not the date of the largest daily move, not a local low, not a
detector firing). No candidate date was moved, nudged or re-picked after
any number in this file existed. Two further consequences of that rule,
disclosed here rather than discovered later:

- The Mt. Gox onset is 2014-02-07 (the first public halt announcement), NOT
  2014-02-24 (trading suspension), NOT 2014-02-28 (bankruptcy filing) and
  NOT the price trough. 2014-02-07 is the earliest of the three, i.e. the
  choice least favourable to any detector, since a detector must react
  earlier still to score a positive lead.
- 2013-12-05 (PBoC bans Chinese financial institutions from handling
  bitcoin) was CONSIDERED and EXCLUDED: it falls in 2013, which
  `r143_shared` guardrail 1 admits only as a disclosed sensitivity window
  (Gandal et al. 2018 Mt. Gox-era manipulation), and its +/-60d window would
  sit almost entirely inside that contaminated year. Excluded a priori, not
  after seeing whether it would have helped.
- Deliberately NOT included: any "the market ground down through 2015"
  style episode. There is no specific, independently-dated news event for
  such a period, so any onset would have to be read off the price series --
  exactly the circularity guardrail 7 forbids. Three well-documented
  episodes are reported instead of padding to four.

=====================================================================
THE THREE NEW EPISODES
=====================================================================

1. **2014-02-07 -- Mt. Gox halts all bitcoin withdrawals.**
   Mt. Gox, then handling ~70% of global bitcoin volume, announced on
   2014-02-07 that it was suspending ALL bitcoin withdrawals, publicly
   blaming transaction malleability; the halt escalated to a full trading
   suspension and the site going dark on 2014-02-24, and to a Tokyo
   bankruptcy filing on 2014-02-28 disclosing ~850,000 BTC missing.
   (CoinDesk, "Mt. Gox Halts ALL Bitcoin Withdrawals, Price Drop Follows",
   2014-02-07; en.bitcoin.it/wiki/Collapse_of_Mt._Gox.)
   CLASSIFICATION: **slow build-up** (R-85's axis). This is the important
   one for the hypothesis under test: unlike a single-day liquidation
   shock, the Mt. Gox failure is a three-week, publicly staged
   deterioration (halt -> suspension -> bankruptcy) that opens a
   year-long bear market -- structurally the same shape as the 2018 bear
   onset, which is the ONLY episode any of the eight closed detectors has
   ever caught early. Disclosed as a hybrid: 2014-02-07 itself was a
   discrete news event with a same-day price drop; it is the multi-week
   escalation and the regime it opens, not that one day, that earns the
   slow-build-up label.

2. **2015-01-05 -- Bitstamp hot-wallet breach; exchange suspends service.**
   Bitstamp announced on 2015-01-05 (09:00 UTC) that it was suspending all
   service after operational-wallet compromise of "less than 19,000 BTC"
   (~US$5M); trading resumed 2015-01-09. (Bitstamp's own 2015-01-05
   announcement; CNBC, "Major bitcoin exchange suspended after price
   plunge", 2015-01-05; TechCrunch 2015-01-05.)
   CLASSIFICATION: **sudden shock** -- single-day breach and halt.
   DISCLOSED DATA CAVEAT, named before the gate ran: this project's BTC
   spot series IS Bitstamp, so the venue's own outage is inside the data.
   Bitstamp's public API serves 2015-01-06/07/08 as 288 empty candles a day
   (volume exactly 0.0, close frozen at 276.80), i.e. three consecutive
   days of EXACTLY ZERO daily log return sitting one day after the onset.
   Every detector here reads daily returns, so those three days carry no
   information by construction. This is genuine venue data, not a fill
   artifact of `scripts/fetch_bitstamp_early.py` (which does no
   reindexing/ffill at all), and it is part of the event rather than a
   defect -- but any FAIL on this episode must be read with it in view, and
   it is flagged in the output table rather than folded silently into a
   score.

3. **2016-08-02 -- Bitfinex hack (119,756 BTC).**
   Bitfinex announced on 2016-08-02 that 119,756 BTC (~US$72M) had been
   stolen from its multi-signature hot wallets in the space of about three
   hours; BTC fell roughly 20% on the announcement.
   (en.wikipedia.org/wiki/2016_Bitfinex_hack; CoinDesk, "The Bitfinex
   Bitcoin Hack: What We Know (And Don't Know)", 2016-08-03.)
   CLASSIFICATION: **sudden shock** -- single-day news-driven repricing,
   largely retraced within weeks.

Resulting calendar composition (R-85's axis), stated before any result:
original six = 1 slow build-up (2018 onset) + 5 sudden shocks; extended
nine = 2 slow build-ups + 7 sudden shocks. The extension roughly doubles
the slow-build-up count, which is the specific direction R-85's diagnosis
says should matter.

=====================================================================
WHAT IS RUN, AND WHAT COUNTS AS THE FINDING
=====================================================================
Three baselines, all imported verbatim, none re-implemented:

(a) v4's OWN anchor vote (`r143_shared.anchor_majority`, itself
    `r100_shared`'s copy) as its own detector -- lead is 0 by construction
    wherever a downward flip exists, so this arm is really testing the one
    thing that is NOT trivial about it: does v4's anchor vote react AT ALL
    inside +/-60 days of each onset? Every other detector's score is
    defined relative to that flip; where no flip exists the gate is
    undefined and every detector fails by construction, not on merit.
(b) BOCPD -- `r82_shared.bocpd_daily_causal_signals`, default hazard,
    `K_SHORT_DAYS=5`, `nearest_bocpd_detection`. Best prior score, 2/6.
(c) CUSUM -- `r139_shared.cusum_daily_causal_signals` at R-137/R-138's
    frozen textbook constants (trail=90, k=0.5, h=5.0), plus R-139's own
    36-cell sweep grid re-run on the extended calendar.

Pass rule, null and search window are R-82's, unchanged and not restated
in a friendlier form: episode PASSES iff (a) lead >= 0 against v4's own
nearest downward anchor flip AND (b) lead >= the block-bootstrap null's
median (500 draws, 5-day blocks). Window +/-60d. The only thing this file
changes is the LIST OF EPISODES.

CONTROL, run first: the original six on the CANONICAL 2017-> file alone,
to confirm this file reproduces R-82's/R-139's published numbers before
anything is attributed to the calendar. The extended series prepends four
years of history to every detector's own state, so a changed 2017+ number
would be a DATA effect, not a calendar effect; the two are separated
explicitly rather than assumed away.

Holdout: every series is truncated to < `OOS_START` (2023-01-01) and
asserted. This file reads no holdout bar.
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

from experiments.r143_shared import load_extended_btc_spot  # noqa: E402
from experiments.r82_shared import (  # noqa: E402
    K_SHORT_DAYS,
    OOS_START,
    STRESS_EPISODES,
    anchor_majority,
    block_bootstrap_shifts,
    bocpd_daily_causal_signals,
    episode_window,
    nearest_bocpd_detection,
    nearest_transition,
    truncation_causality_probe,
)
from experiments.r138_shared import (  # noqa: E402
    CUSUM_H_MULT,
    CUSUM_K_MULT,
    CUSUM_TRAIL_DAYS,
)
from experiments.r139_shared import (  # noqa: E402
    NOVEL_H_GRID,
    NOVEL_K_GRID,
    NOVEL_TRAIL_GRID,
    cusum_daily_causal_signals,
)

DATA_DIR = ROOT / "data"

# R-82's own gate constants, unchanged.
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 82  # R-82's seed, so the original-six control is bit-comparable

# ------------------------------------------------------------------ calendar

ORIGINAL_SIX = list(STRESS_EPISODES)  # verbatim, unchanged

# Guardrail 7: every onset below is the date of the first public
# announcement of an independently reported event. See module docstring for
# the citation and the sudden-shock / slow-build-up classification.
NEW_PRE2017_EPISODES = [
    ("2014-02 Mt. Gox withdrawal halt / collapse", "2014-02-07"),
    ("2015-01 Bitstamp breach / service halt", "2015-01-05"),
    ("2016-08 Bitfinex hack", "2016-08-02"),
]

EXTENDED_STRESS_EPISODES = ORIGINAL_SIX + NEW_PRE2017_EPISODES

EPISODE_KIND = {
    "2018 bear onset (post-Dec-2017 top)": "slow build-up",
    "2018 bear bottom / capitulation": "sudden shock",
    "2020-03 COVID crash": "sudden shock",
    "2021-11 top / 2022 bear transition": "sudden shock",
    "2022-05 Terra/Luna collapse": "sudden shock",
    "2022-11 FTX collapse": "sudden shock",
    "2014-02 Mt. Gox withdrawal halt / collapse": "slow build-up",
    "2015-01 Bitstamp breach / service halt": "sudden shock",
    "2016-08 Bitfinex hack": "sudden shock",
}

# Episodes with a disclosed data caveat, flagged in every table.
DATA_CAVEAT = {
    "2015-01 Bitstamp breach / service halt":
        "3 zero-volume frozen-close days (2015-01-06..08) inside the window",
}

NEW_LABELS = {lbl for lbl, _ in NEW_PRE2017_EPISODES}


# --------------------------------------------------------------- gate infra
#
# Identical construction to `r82_gate.py` / `r139_shared.step_a_gate`, with
# the episode list lifted out into a parameter. The real-detection calls go
# through the imported helpers (`nearest_transition`,
# `nearest_bocpd_detection`); only the null's inner loop is inlined, exactly
# as `r82_gate.null_leads` inlines it.


def _crossing_indices(local: np.ndarray, kind: str, k_short: int) -> np.ndarray:
    """Indices of 'the detector fired here' inside a local window array.

    ``kind='run_length'``: down-crossing to <= k_short (BOCPD/CUSUM).
    ``kind='down_transition'``: a strictly downward step (the anchor vote).
    Both replicate the imported helpers' own logic bit for bit.
    """
    n = len(local)
    fired = np.zeros(n, dtype=bool)
    if kind == "run_length":
        short = local <= k_short
        fired[1:] = short[1:] & ~short[:-1]
        fired[0] = bool(short[0]) if n else False
    elif kind == "down_transition":
        fired[1:] = local[1:] < local[:-1]
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return np.where(fired)[0]


def _null_leads(series: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                 flip_time: pd.Timestamp, kind: str, k_short: int = K_SHORT_DAYS,
                 n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                 seed: int = NULL_SEED) -> np.ndarray:
    local = series.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        idx = _crossing_indices(local[shift], kind, k_short)
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        detect_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - detect_time).total_seconds() / 86400.0
    return leads


def detection_lag_gate(bars: pd.DataFrame, majority: pd.Series, detector: pd.Series,
                        episodes: list[tuple[str, str]], *, kind: str,
                        k_short: int = K_SHORT_DAYS, window_days: int = WINDOW_DAYS,
                        name: str = "") -> list[dict]:
    """R-82's Step-A detection-lag gate over an arbitrary episode list."""
    out = []
    for label, onset_str in episodes:
        onset, window = episode_window(bars, onset_str, window_days)
        row = dict(label=label, onset=onset_str, kind=EPISODE_KIND.get(label, "?"),
                    detector=name, lead=float("nan"), null_median=float("nan"),
                    pass_b=False, reason="")
        if len(window) == 0:
            row["reason"] = "no bars in window"
            out.append(row)
            continue
        flip_time = nearest_transition(majority, window, onset, direction="down")
        if kind == "run_length":
            detect_time = nearest_bocpd_detection(detector, window, onset, k_short)
        else:
            detect_time = nearest_transition(detector, window, onset, direction="down")
        row["anchor_flip"] = flip_time
        row["detect"] = detect_time
        if flip_time is None:
            row["reason"] = "no v4 anchor flip in window (gate undefined)"
            out.append(row)
            continue
        if detect_time is None:
            row["reason"] = "no detector firing in window"
            out.append(row)
            continue
        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = _null_leads(detector, window, onset, flip_time, kind, k_short)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        row["lead"] = lead
        row["null_median"] = null_median
        row["n_null"] = int(len(valid))
        row["pass_b"] = bool(lead >= 0 and not np.isnan(null_median) and lead >= null_median)
        out.append(row)
    return out


# ------------------------------------------------------------------ loading


def _truncate(df: pd.DataFrame) -> pd.DataFrame:
    cut = pd.Timestamp(OOS_START, tz="UTC")
    out = df.loc[df.index < cut].copy()
    assert out.index.max() < cut, "holdout bar read"
    return out


def load_bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(canonical 2017-> control bars, extended 2013-> bars), both < OOS_START."""
    canon, label = load_dataset(DATA_DIR, "spot")
    canon = _truncate(canon)
    ext = _truncate(load_extended_btc_spot())
    print(f"canonical ({label}): {len(canon):,} bars  {canon.index[0]} -> {canon.index[-1]}",
          file=sys.stderr)
    print(f"extended            : {len(ext):,} bars  {ext.index[0]} -> {ext.index[-1]}",
          file=sys.stderr)
    return canon, ext


# ----------------------------------------------------------------- printing


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n  {title}")
    print("  " + "-" * 108)
    print(f"  {'episode':44s} {'onset':11s} {'kind':13s} {'lead(d)':>9s} "
          f"{'null med':>9s}  {'PASS':5s}  note")
    for r in rows:
        lead = r["lead"]
        nm = r["null_median"]
        lead_s = f"{lead:+9.2f}" if not np.isnan(lead) else "        -"
        nm_s = f"{nm:+9.2f}" if not np.isnan(nm) else "        -"
        note = r.get("reason", "")
        cav = DATA_CAVEAT.get(r["label"])
        if cav:
            note = (note + "; " if note else "") + "[data caveat]"
        print(f"  {r['label'][:44]:44s} {r['onset']:11s} {r['kind']:13s} {lead_s} "
              f"{nm_s}  {str(r['pass_b']):5s}  {note}")


def print_timing_detail(rows: list[dict], title: str, window_days: int = WINDOW_DAYS) -> None:
    """Raw flip/detection timestamps, so a large positive lead can be checked
    against the window-edge 'already-bearish-going-in' confound this project's
    own R-73/R-81/R-83 write-ups flag for exactly this gate."""
    print(f"\n  {title} -- raw timing (window edge = onset +/- {window_days}d)")
    print("  " + "-" * 108)
    print(f"  {'episode':44s} {'v4 anchor flip':22s} {'detector fired':22s} {'edge?':6s}")
    for r in rows:
        flip = r.get("anchor_flip")
        det = r.get("detect")
        edge = ""
        if det is not None:
            onset = pd.Timestamp(r["onset"], tz="UTC")
            days_from_onset = (det - onset).total_seconds() / 86400.0
            if abs(abs(days_from_onset) - window_days) < 3.0:
                edge = "EDGE"
        print(f"  {r['label'][:44]:44s} {str(flip):22s} {str(det):22s} {edge:6s}")


def split_score(rows: list[dict]) -> tuple[int, int, int, int]:
    orig = [r for r in rows if r["label"] not in NEW_LABELS]
    new = [r for r in rows if r["label"] in NEW_LABELS]
    return (sum(r["pass_b"] for r in orig), len(orig),
            sum(r["pass_b"] for r in new), len(new))


def report(rows: list[dict], detector: str) -> dict:
    o_pass, o_n, n_pass, n_n = split_score(rows)
    print_table([r for r in rows if r["label"] not in NEW_LABELS],
                f"{detector}: ORIGINAL six episodes")
    new_rows = [r for r in rows if r["label"] in NEW_LABELS]
    print_table(new_rows, f"{detector}: NEW pre-2017 episodes")
    if new_rows:
        print_timing_detail(new_rows, f"{detector}: NEW pre-2017 episodes")
    print(f"\n  {detector} SCORE: original {o_pass}/{o_n}   new {n_pass}/{n_n}   "
          f"EXTENDED TOTAL {o_pass + n_pass}/{o_n + n_n}")
    return dict(detector=detector, rows=rows, orig_pass=o_pass, orig_n=o_n,
                new_pass=n_pass, new_n=n_n)


def verdict_map(rows: list[dict]) -> dict[str, bool]:
    return {r["label"]: bool(r["pass_b"]) for r in rows}


# --------------------------------------------------------------------- main


def main(run_sweep: bool = True) -> dict:
    t0 = time.time()
    print("=" * 112)
    print("R-143 NOVEL: does the six-episode Step-A detection-lag gate generalize")
    print("             past the six dates it was built from?")
    print("=" * 112)
    print(f"\ncalendar: {len(ORIGINAL_SIX)} original + {len(NEW_PRE2017_EPISODES)} new "
          f"= {len(EXTENDED_STRESS_EPISODES)} episodes")
    for lbl, d in EXTENDED_STRESS_EPISODES:
        tag = "NEW" if lbl in NEW_LABELS else "   "
        print(f"  {tag} {d}  {EPISODE_KIND[lbl]:13s}  {lbl}")
    n_slow = sum(1 for lbl, _ in EXTENDED_STRESS_EPISODES if EPISODE_KIND[lbl] == "slow build-up")
    print(f"  composition: {n_slow} slow build-up / "
          f"{len(EXTENDED_STRESS_EPISODES) - n_slow} sudden shock "
          f"(original six were 1 / 5)")

    canon, ext = load_bars()

    # ---------------------------------------------------------------- control
    print("\n" + "=" * 112)
    print("CONTROL: original six on the CANONICAL 2017-> series (reproduces R-82 / R-139)")
    print("=" * 112)
    maj_c = anchor_majority(canon)
    bocpd_c = bocpd_daily_causal_signals(canon)["bocpd_map_run_length"]
    cusum_c = cusum_daily_causal_signals(
        canon, trail_days=CUSUM_TRAIL_DAYS, k_mult=CUSUM_K_MULT,
        h_mult=CUSUM_H_MULT)["cusum_run_length"]
    ctrl = {}
    ctrl["anchor"] = report(detection_lag_gate(canon, maj_c, maj_c, ORIGINAL_SIX,
                                                kind="down_transition", name="anchor"),
                             "ANCHOR VOTE (canonical)")
    ctrl["bocpd"] = report(detection_lag_gate(canon, maj_c, bocpd_c, ORIGINAL_SIX,
                                               kind="run_length", name="bocpd"),
                            "BOCPD (canonical)")
    ctrl["cusum"] = report(detection_lag_gate(canon, maj_c, cusum_c, ORIGINAL_SIX,
                                               kind="run_length", name="cusum"),
                            "CUSUM fixed constants (canonical)")

    # --------------------------------------------------------------- extended
    print("\n" + "=" * 112)
    print("EXTENDED CALENDAR on the EXTENDED 2013-> series")
    print("=" * 112)
    maj_e = anchor_majority(ext)
    bocpd_e = bocpd_daily_causal_signals(ext)["bocpd_map_run_length"]
    cusum_e = cusum_daily_causal_signals(
        ext, trail_days=CUSUM_TRAIL_DAYS, k_mult=CUSUM_K_MULT,
        h_mult=CUSUM_H_MULT)["cusum_run_length"]
    for s in (maj_e, bocpd_e, cusum_e):
        assert s.index.max() < pd.Timestamp(OOS_START, tz="UTC")

    ext_res = {}
    ext_res["anchor"] = report(detection_lag_gate(ext, maj_e, maj_e, EXTENDED_STRESS_EPISODES,
                                                   kind="down_transition", name="anchor"),
                                "ANCHOR VOTE (extended)")
    ext_res["bocpd"] = report(detection_lag_gate(ext, maj_e, bocpd_e, EXTENDED_STRESS_EPISODES,
                                                  kind="run_length", name="bocpd"),
                               "BOCPD (extended)")
    ext_res["cusum"] = report(detection_lag_gate(ext, maj_e, cusum_e, EXTENDED_STRESS_EPISODES,
                                                  kind="run_length", name="cusum"),
                               "CUSUM fixed constants (extended)")

    # ---------------------------------------------------- causality probes
    # The detectors are imported unmodified from rounds that already probed
    # them, but they have never been run on the PRE-2017 bars before, so the
    # standard truncation probe is re-run here on the extended series.
    print("\n" + "=" * 112)
    print("CAUSALITY: truncation probes on the EXTENDED series (new pre-2017 bars)")
    print("=" * 112)
    probe_at = int(len(ext) * 0.25)  # lands in 2015, inside the new data
    probes = {
        "anchor_majority": lambda d: anchor_majority(d).to_numpy(),
        "bocpd_map_run_length":
            lambda d: bocpd_daily_causal_signals(d)["bocpd_map_run_length"].to_numpy(),
        "cusum_run_length":
            lambda d: cusum_daily_causal_signals(
                d, trail_days=CUSUM_TRAIL_DAYS, k_mult=CUSUM_K_MULT,
                h_mult=CUSUM_H_MULT)["cusum_run_length"].to_numpy(),
    }
    for nm, fn in probes.items():
        ok = truncation_causality_probe(fn, ext, check_at=probe_at, shorter_by=20_000)
        print(f"  {nm:24s} causal at bar {probe_at:,} ({ext.index[probe_at]}): {ok}")
        assert ok, f"{nm} failed the truncation causality probe"

    # ------------------------------------------------- data-effect separation
    print("\n" + "=" * 112)
    print("DATA EFFECT vs CALENDAR EFFECT: original six, canonical vs extended series")
    print("=" * 112)
    print("  (a changed 2017+ verdict here is caused by the four extra years of detector")
    print("   state, NOT by the new episodes -- separated so nothing is misattributed)")
    for key in ("anchor", "bocpd", "cusum"):
        vc = verdict_map(ctrl[key]["rows"])
        ve = verdict_map([r for r in ext_res[key]["rows"] if r["label"] not in NEW_LABELS])
        diffs = [k for k in vc if vc[k] != ve[k]]
        print(f"  {key:7s}: canonical {ctrl[key]['orig_pass']}/6 -> "
              f"extended-series {ext_res[key]['orig_pass']}/6   "
              f"per-episode verdict changes: {diffs if diffs else 'NONE'}")

    # ------------------------------------------------------------ CUSUM sweep
    sweep = None
    if run_sweep:
        print("\n" + "=" * 112)
        print("CUSUM 36-cell sweep (R-139's own frozen grid) on the EXTENDED calendar")
        print("=" * 112)
        sweep = []
        for trail in NOVEL_TRAIL_GRID:
            for k_mult in NOVEL_K_GRID:
                for h_mult in NOVEL_H_GRID:
                    rl = cusum_daily_causal_signals(
                        ext, trail_days=trail, k_mult=k_mult,
                        h_mult=h_mult)["cusum_run_length"]
                    rows = detection_lag_gate(ext, maj_e, rl, EXTENDED_STRESS_EPISODES,
                                               kind="run_length", name="cusum")
                    o_p, o_n, n_p, n_n = split_score(rows)
                    sweep.append(dict(trail_days=trail, k_mult=k_mult, h_mult=h_mult,
                                      orig_pass=o_p, new_pass=n_p,
                                      total=o_p + n_p, rows=rows))
                    print(f"  trail={trail:3d} k={k_mult:.2f} h={h_mult:.1f}  "
                          f"original {o_p}/{o_n}  new {n_p}/{n_n}  "
                          f"extended {o_p + n_p}/{o_n + n_n}")
        best = max(sweep, key=lambda c: c["total"])
        print(f"\n  best cell on the EXTENDED calendar: trail={best['trail_days']} "
              f"k={best['k_mult']} h={best['h_mult']}  -> {best['total']}/9 "
              f"(original {best['orig_pass']}/6, new {best['new_pass']}/3)")
        best_orig = max(sweep, key=lambda c: c["orig_pass"])
        print(f"  best cell on the ORIGINAL six alone : trail={best_orig['trail_days']} "
              f"k={best_orig['k_mult']} h={best_orig['h_mult']}  -> "
              f"{best_orig['orig_pass']}/6")

    # ----------------------------------------------------------- Step-3 finding
    print("\n" + "=" * 112)
    print("STEP-3 FINDING: did the EXTENDED calendar change any detector's verdict?")
    print("=" * 112)
    for key in ("anchor", "bocpd", "cusum"):
        r = ext_res[key]
        orig_rate = r["orig_pass"] / r["orig_n"]
        new_rate = r["new_pass"] / r["new_n"] if r["new_n"] else float("nan")
        print(f"  {key:7s}: original {r['orig_pass']}/{r['orig_n']} ({orig_rate:.0%})   "
              f"new pre-2017 {r['new_pass']}/{r['new_n']} ({new_rate:.0%})   "
              f"extended {r['orig_pass'] + r['new_pass']}/{r['orig_n'] + r['new_n']}")
        slow = [x for x in r["rows"] if x["kind"] == "slow build-up"]
        sudden = [x for x in r["rows"] if x["kind"] == "sudden shock"]
        print(f"           by kind: slow build-up "
              f"{sum(x['pass_b'] for x in slow)}/{len(slow)}   "
              f"sudden shock {sum(x['pass_b'] for x in sudden)}/{len(sudden)}")

    print(f"\nmax timestamp read anywhere: "
          f"{max(canon.index.max(), ext.index.max())}  (< {OOS_START})")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("configurations evaluated: 0 outside R-139's own already-registered 36-cell grid "
          "(no new parameter was chosen against these episodes)")

    return dict(control=ctrl, extended=ext_res, sweep=sweep)


if __name__ == "__main__":
    main(run_sweep="--no-sweep" not in sys.argv)
