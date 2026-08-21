#!/usr/bin/env python
"""R-73 NOVEL branch, step 1 (REQUIRED FIRST, per ROUTINE.md discipline and
this round's brief): does BTC DVOL (Deribit's official 30-day
implied-volatility index) LEAD `kelly_regime_v4`'s own 3-anchor price-gate
flip, measured BEFORE any strategy code is written?

THIS FILE IS THE GATE. Step 5 (build a confirming-vote strategy,
``kelly_regime_v19_dvol_confirm.py``) runs ONLY if the decision rule below
resolves favorably. If DVOL lags, like VIX/DXY did (R-53), that is itself
the round's answer.

Idea, one sentence
------------------
Options-implied volatility should react to new information faster than a
slow trailing 20/40/80-day price average, because option writers are
pricing forward risk continuously rather than waiting for price to cross a
trailing anchor -- if that is true, a DVOL spike (or fast rate-of-change)
should cross a stress threshold BEFORE `kelly_regime_v4`'s own 3-anchor
majority flips bearish, in the handful of BTC stress episodes that occurred
within DVOL's ~2021-03-24-> coverage window.

Constraint attacked: INFO (one price series). Fifth attempt on this
constraint in this project (B-07/R-44 on-chain, R-53 VIX/DXY, R-54/R-55/R-58
stablecoin supply, now DVOL) -- but the FIRST attempt at a forward-looking,
PRICED signal rather than a spot-flow or balance-sheet proxy. See
``experiments/_dvol_signal.py``'s module docstring for the full mechanism
argument and why this is not a duplicate of R-53's VIX construction (VIX
describes the rest of the financial system and was found to LAG; DVOL is
BTC's own implied vol, never before measured for lead time in this project).

Hard data limitation, named up front, not discovered after running anything:
DVOL history starts 2021-03-24. Of this project's usual stress episodes
(2018 crash, 2020-03 COVID crash, 2022 broad bear), only the tail end of the
2022 bear falls inside DVOL's coverage, and 2018/2020-03 are entirely
outside it. Usable matched episodes within coverage: **May 2021 crash,
Terra/Luna collapse (May 2022), FTX collapse (Nov 2022) -- n=3**, far fewer
than R-53's/R-54's n=12. This is a materially weaker evidentiary base and is
reported as such regardless of outcome.

Pre-registered signal construction (fixed BEFORE any episode date is looked
at; see ``_dvol_signal.py`` for the code)
-----------------------------------------------------------------------------
Two features, both z-scored on a trailing 180-day window (min_periods=60;
shorter than R-53's 365-day VIX window because DVOL's own history barely
exceeds 5 years and inner-train starts within months of the series' first
observation):

    dvol_z       -- DVOL close, trailing 180d z-score (level/"spike" reading)
    dvol_roc5_z  -- 5-day log change in DVOL, trailing 180d z-score
                    ("fast rate-of-change" reading)

Latched hysteresis vote on EACH feature independently, using this project's
standing convention (identical hysteresis discipline and identical
thresh_hi=1.0/gap=0.75 primary as R-53's `_macro_vote`, R-54/R-55's
`_stable_vote` -- NOT re-tuned for this round, chosen for continuity so the
DVOL result is comparable to the VIX/stablecoin results on the same footing):

    vote -> 0 ("stress")  when the feature crosses ABOVE thresh_hi=1.0
    vote -> 1 ("calm")    when the feature crosses BELOW thresh_lo=0.25 (gap=0.75)
    vote unchanged (latched) in between; defaults to 1 ("calm")

Pre-registered episode set (fixed BEFORE any lead-time number is computed;
these are well-known historical dates, not fit to this data)
-----------------------------------------------------------------------------
    1. May 2021 crash        epicenter 2021-05-19 (BTC ATH 2021-04-14 ~$64.8k
                              -> intraday low ~$30k on 2021-05-19)
    2. Terra/Luna collapse    epicenter 2022-05-09 (UST depeg onset ~2022-05-07,
                              full collapse by 2022-05-12)
    3. FTX collapse           epicenter 2022-11-08 (withdrawal halt 2022-11-08,
                              bankruptcy filing 2022-11-11)

Search window: +/- 45 days around each epicenter (narrower than R-53's/
R-54's +/-180d `nearest()` window, because with n=3 episodes a wide window
risks matching a DVOL vote transition from a DIFFERENT, unrelated stress
wobble rather than the named episode -- a narrower, symmetric window fixed
before looking is the conservative choice here).

Pre-registered decision rule (fixed BEFORE any lead-time number is read)
-----------------------------------------------------------------------------
Proceed to step 5 (build ``kelly_regime_v19_dvol_confirm.py``) ONLY IF:
    (a) at least 2 of the 3 episodes show DVOL (either feature) leading the
        3-anchor majority's own bear-onset, AND
    (b) the median lead is positive.
Otherwise: STOP HERE. Report NEGATIVE. Do not write any strategy code.
This mirrors R-53's own finding threshold (it required a majority-leading,
positive-median result before treating the signal as a live candidate) and
is written before this file's numbers are computed.

If the rule is cleared, step 5 must ALSO measure false-onset rate /
specificity (how many DVOL vote transitions fire with NO matched episode
nearby) before declaring anything monetizable -- named here because R-54/
R-55's stablecoin signal cleared an identical lead-time gate and then failed
on exactly this axis (fired on noise as often as on genuine stress).

Usage
-----
    python experiments/kelly_regime_v19_dvol_leadtime.py descriptive
    python experiments/kelly_regime_v19_dvol_leadtime.py leadtime
    python experiments/kelly_regime_v19_dvol_leadtime.py all
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
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402

from experiments._dvol_signal import compute_dvol_features  # noqa: E402

DATA_DIR = ROOT / "data"

OOS_START = "2023-01-01"  # never read in this file -- see the restriction applied below

THRESH_HI = 1.0   # fixed a-priori, matches R-53/R-54/R-55 primary, not swept for this gate
GAP = 0.75        # fixed a-priori, matches R-53/R-54/R-55 primary, not swept for this gate

EPISODES = [
    ("May 2021 crash", pd.Timestamp("2021-05-19", tz="UTC")),
    ("Terra/Luna collapse", pd.Timestamp("2022-05-09", tz="UTC")),
    ("FTX collapse", pd.Timestamp("2022-11-08", tz="UTC")),
]
SEARCH_WINDOW_DAYS = 45


def build_dataframe() -> tuple[pd.DataFrame, str]:
    spot, label = load_dataset(DATA_DIR, "spot")
    dvol = compute_dvol_features(spot, DATA_DIR)
    out = spot.copy()
    out["dvol_z_visible"] = dvol["dvol_z"]
    out["dvol_roc5_z_visible"] = dvol["dvol_roc5_z"]
    return out, label


DF, LABEL = build_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  (data: {LABEL}); "
      f"DVOL coverage {DF['dvol_z_visible'].notna().sum():,} bars "
      f"from {DF['dvol_z_visible'].dropna().index[0]:%Y-%m-%d} "
      f"to {DF['dvol_z_visible'].dropna().index[-1]:%Y-%m-%d}", file=sys.stderr)


# ---------------------------------------------------------------- the votes


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote. Duplicated
    (not imported) from kelly_regime_v14_macro_lead.py, the established
    precedent for this exact helper (v15/v16 files did the same)."""
    votes = {}
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes[days] = v.ffill().fillna(0.0)
    return votes


def _latched_vote(feature: pd.Series, thresh_hi: float, gap: float) -> pd.Series:
    """Latched 0/1 hysteresis vote, identical discipline to R-53's
    `_macro_vote` / R-54-55's `_stable_vote`. vote=0 ("stress") requires
    crossing ABOVE thresh_hi; vote=1 ("calm") requires falling back BELOW
    thresh_lo = thresh_hi - gap. Defaults to 1.0 wherever feature is NaN."""
    thresh_lo = thresh_hi - gap
    raw = np.where(feature > thresh_hi, 0.0,
                    np.where(feature < thresh_lo, 1.0, np.nan))
    return pd.Series(raw, index=feature.index).ffill().fillna(1.0)


def _daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Daily-resampled transition INTO target_value, deduplicated so
    transitions within min_gap_days of a prior one count as one episode's
    onset. Identical logic (and the same hand-caught boolean-dtype bug fix)
    as kelly_regime_v14_macro_lead.py's `_daily_transitions`."""
    daily = series.resample("1D").last().ffill()
    is_target = (daily == target_value)
    prev = is_target.shift(fill_value=False)
    onsets = daily.index[is_target & (~prev)]
    kept = []
    for d in onsets:
        if not kept or (d - kept[-1]).days >= min_gap_days:
            kept.append(d)
    return kept


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    cov = DF["dvol_z_visible"].dropna()
    lo = cov.index[0]
    hi = min(cov.index[-1], pd.Timestamp(OOS_START, tz="UTC") - pd.Timedelta(days=1))
    frame = DF.loc[lo:hi]
    close = frame["close"]

    print(f"descriptive window (DVOL coverage): {lo:%Y-%m-%d} -> {hi:%Y-%m-%d}  "
          f"({(hi - lo).days} days)")
    print("\nprice-anchor vote transition counts over this window (context):")
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * BARS_PER_DAY)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    for feat_name in ("dvol_z_visible", "dvol_roc5_z_visible"):
        feat = frame[feat_name]
        print(f"\n{feat_name} summary: mean={feat.mean():.2f} std={feat.std():.2f} "
              f"min={feat.min():.2f} max={feat.max():.2f}")
        vote = _latched_vote(feat, THRESH_HI, GAP)
        flips_to_stress = int(((vote == 0.0) & (vote.shift() == 1.0)).sum())
        flips_to_calm = int(((vote == 1.0) & (vote.shift() == 0.0)).sum())
        print(f"  latched vote (thresh_hi={THRESH_HI}, gap={GAP}): "
              f"{flips_to_stress} stress-onset event(s), {flips_to_calm} calm-return event(s)")


# ------------------------------------------------------- step 1: lead-time (THE GATE)


def leadtime() -> None:
    # Restricted to strictly pre-2023 bars (matches R-53/R-54's own
    # `leadtime()` precedent of `lo, hi = TRAIN[0], VALID[1]`) -- this is a
    # descriptive step, but its output must not expose or be influenced by
    # any 2023+ (holdout) date, since all 3 pre-registered episodes are
    # pre-2023 anyway and nothing is lost by the restriction.
    cov = DF["dvol_z_visible"].dropna()
    lo = cov.index[0]
    hi = min(cov.index[-1], pd.Timestamp(OOS_START, tz="UTC") - pd.Timedelta(days=1))
    frame = DF.loc[lo:hi]
    close = frame["close"]

    votes = _anchor_votes(close, (20, 40, 80), 0.01)
    anchor_sum = sum(votes.values())
    majority_bear = (anchor_sum < 1.5).astype(float)  # 1 when >=2 of 3 anchors bearish
    majority_onsets = _daily_transitions(majority_bear, 1.0)

    print(f"3-anchor MAJORITY bear-onset episodes in DVOL-coverage window: {len(majority_onsets)}")
    print(f"  {[d.date().isoformat() for d in majority_onsets]}")

    def nearest_onset(target_date, candidates, window_days):
        best, best_dist = None, None
        for c in candidates:
            dist = (c - target_date).days
            if abs(dist) <= window_days and (best_dist is None or abs(dist) < abs(best_dist)):
                best, best_dist = c, dist
        return best, best_dist

    results = {}
    for feat_name in ("dvol_z_visible", "dvol_roc5_z_visible"):
        feat = frame[feat_name]
        vote = _latched_vote(feat, THRESH_HI, GAP)
        bear = 1.0 - vote
        dvol_onsets = _daily_transitions(bear, 1.0)
        print(f"\n{'=' * 70}\nFEATURE: {feat_name} "
              f"({len(dvol_onsets)} stress-onset event(s) in coverage window)")
        print(f"  {[d.date().isoformat() for d in dvol_onsets]}")

        leads = []
        print(f"\n  episode-by-episode (positive lead_days = DVOL flips FIRST):")
        for ep_name, epicenter in EPISODES:
            dvol_match, dvol_dist = nearest_onset(epicenter, dvol_onsets, SEARCH_WINDOW_DAYS)
            anchor_match, anchor_dist = nearest_onset(epicenter, majority_onsets, SEARCH_WINDOW_DAYS)
            if dvol_match is None or anchor_match is None:
                print(f"    {ep_name:22s} epicenter {epicenter.date()}: "
                      f"DVOL onset={'none within window' if dvol_match is None else dvol_match.date()}  "
                      f"anchor onset={'none within window' if anchor_match is None else anchor_match.date()}  "
                      f"-> UNMATCHED, excluded from median")
                continue
            lead_days = (anchor_match - dvol_match).days
            leads.append(lead_days)
            print(f"    {ep_name:22s} epicenter {epicenter.date()}: "
                  f"DVOL onset={dvol_match.date()}  anchor onset={anchor_match.date()}  "
                  f"lead_days={lead_days:+d}")

        results[feat_name] = leads
        if leads:
            n_lead = sum(1 for x in leads if x > 0)
            print(f"\n  SUMMARY [{feat_name}]: {len(leads)}/{len(EPISODES)} matched episode(s), "
                  f"{n_lead}/{len(leads)} DVOL-leads, median lead_days={float(np.median(leads)):+.1f}, "
                  f"individual leads={leads}")
        else:
            print(f"\n  SUMMARY [{feat_name}]: no matched pairs -- cannot assess lead/lag")

    print(f"\n{'=' * 70}\nPRE-REGISTERED DECISION RULE: proceed to step 5 only if >=2/3 episodes "
          f"show DVOL leading (either feature) AND median lead is positive.")
    for feat_name, leads in results.items():
        if not leads:
            print(f"  {feat_name}: NO -- no matched episodes")
            continue
        n_lead = sum(1 for x in leads if x > 0)
        median = float(np.median(leads))
        clears = (n_lead >= 2) and (median > 0)
        print(f"  {feat_name}: n_lead={n_lead}/{len(leads)}  median={median:+.1f}  "
              f"{'CLEARS -> proceed to step 5' if clears else 'DOES NOT CLEAR'}")
    any_clears = any(
        leads and sum(1 for x in leads if x > 0) >= 2 and float(np.median(leads)) > 0
        for leads in results.values()
    )
    print(f"\nOVERALL: {'AT LEAST ONE FEATURE CLEARS -> proceed to step 5, with a false-onset/specificity check' if any_clears else 'NEITHER FEATURE CLEARS -> STOP, report NEGATIVE, do not build a strategy'}")


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2b -- descriptive")
    print("=" * 78)
    descriptive()
    print("\n" + "=" * 78)
    print("STEP 1 (THE GATE) -- lead-time")
    print("=" * 78)
    leadtime()


if __name__ == "__main__":
    cmds = {"descriptive": descriptive, "leadtime": leadtime, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v19_dvol_leadtime.py [{'|'.join(cmds)}]")
