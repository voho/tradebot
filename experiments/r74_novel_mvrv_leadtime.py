#!/usr/bin/env python
"""R-74 NOVEL branch, step 1 (REQUIRED FIRST, per ROUTINE.md discipline and
this round's brief): does BTC MVRV's own RATE OF CHANGE lead
``kelly_regime_v4``'s own 3-anchor price-gate flip, measured BEFORE any
strategy code is written?

THIS FILE IS THE GATE. A confirming-vote strategy is written only if the
decision rule below resolves favorably. If MVRV's rate of change lags,
like DVOL's level AND its own rate of change did (R-73), or like VIX/DXY's
level did (R-53), that is itself the round's answer, per the pre-registered
stop rule below.

Constraint attacked: INFO (one price series). Sixth attempt on this
constraint in this project (B-07/R-44 on-chain, R-53 VIX/DXY, R-54/R-55/R-58
stablecoin supply, R-73 DVOL/VRP, now MVRV) -- the first to use a
holder-cost-basis / aggregate profit-loss construction rather than a flow,
a priced expectation, or a spillover. See ``experiments/r74_novel_mvrv_signal.py``'s
module docstring for the full mechanism argument and the citable reasoning
for the 30-day PRIMARY / 90-day SECONDARY windows, fixed BEFORE this file
computed anything.

Episode set (fixed BEFORE any lead-time number is computed; the exact five
episodes named in this round's brief, in full -- none dropped, none added)
-----------------------------------------------------------------------------
    1. 2018 bear onset          epicenter 2018-01-17 (BTC ATH 2017-12-17
                                 ~$19.7k; the mid-January 2018 leg down is
                                 the widely-cited onset of the 2018 bear)
    2. 2020-03 COVID crash       epicenter 2020-03-12 ("Black Thursday",
                                 BTC fell ~50% in a day)
    3. 2021 top / 2022 bear      epicenter 2021-11-10 (BTC ATH ~$69k; the
       transition                cycle top the subsequent bear transitioned
                                 from)
    4. 2022-05 Terra/Luna        epicenter 2022-05-09 (UST depeg onset
                                 ~2022-05-07, full collapse by 2022-05-12;
                                 same epicenter R-73 used)
    5. 2022-11 FTX collapse      epicenter 2022-11-08 (withdrawal halt
                                 2022-11-08, bankruptcy filing 2022-11-11;
                                 same epicenter R-73 used)

All five are matchable inside MVRV's 2016-01-01-> coverage and this
project's own inner-train+inner-validation window (2017-01-01 ->
2022-12-31) -- unlike R-73's DVOL study (2021-03-> coverage, only 3 of the
usual episodes reachable), MVRV's longer history means none are dropped for
lack of data. Per this round's brief: do not cherry-pick which episodes
count -- all five are used, whatever the individual match/no-match/lead/lag
outcome turns out to be.

Search window: +/- 90 days around each epicenter. Reasoning, fixed before
computing anything: the five epicenters are well separated (the closest
adjacent pair, 2021-11-10 and 2022-05-09, is ~180 days apart), so a 90-day
window is comfortably inside half that spacing and cannot let two episodes'
matches collide; it is also 3x the PRIMARY feature's own 30-day smoothing
window (matching the rough ratio R-73 used: 9x its 5-day feature window
against a 45-day search window) and equal to the SECONDARY feature's own
90-day window, giving a matching window that is proportionate to a signal
this project has now reasoned should move more slowly than DVOL's or
stablecoin's own signals.

Pre-registered decision rule (fixed BEFORE any lead-time number is read)
-----------------------------------------------------------------------------
For EACH feature (mvrv_roc30_z, mvrv_roc90_z) independently: among the
matched episodes (both an MVRV-feature stress onset AND a v4 3-anchor
majority bear onset found within +/-90 days of the epicenter), does a
MAJORITY lead (lead_days > 0) with a POSITIVE median lead_days?

Proceed to step 2 (build a confirming-vote strategy) ONLY IF at least one
of the two features clears that bar -- the same "either feature" logic
R-73's own pre-registration used, for direct comparability across this
project's INFO-axis rounds. Otherwise: STOP HERE. Report NEGATIVE. Do not
build any strategy, do not sweep any configuration.

Usage
-----
    python experiments/r74_novel_mvrv_leadtime.py descriptive
    python experiments/r74_novel_mvrv_leadtime.py leadtime
    python experiments/r74_novel_mvrv_leadtime.py all
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

from experiments.r74_novel_mvrv_signal import ROC_WINDOWS_DAYS, compute_mvrv_features  # noqa: E402

DATA_DIR = ROOT / "data"

OOS_START = "2023-01-01"  # never read past this in this file -- restriction applied below
TRAIN = ("2017-01-01", "2020-12-31")     # inner-train, matches ROUTINE.md / R-54's own split
VALID = ("2021-01-01", "2022-12-31")     # inner-validation

THRESH_HI = 1.0   # fixed a-priori, matches R-53/R-54/R-55/R-73 primary, not swept for this gate
GAP = 0.75        # fixed a-priori, matches R-53/R-54/R-55/R-73 primary, not swept for this gate

EPISODES = [
    ("2018 bear onset", pd.Timestamp("2018-01-17", tz="UTC")),
    ("2020-03 COVID crash", pd.Timestamp("2020-03-12", tz="UTC")),
    ("2021 top / 2022 bear transition", pd.Timestamp("2021-11-10", tz="UTC")),
    ("2022-05 Terra/Luna", pd.Timestamp("2022-05-09", tz="UTC")),
    ("2022-11 FTX collapse", pd.Timestamp("2022-11-08", tz="UTC")),
]
SEARCH_WINDOW_DAYS = 90


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Second, independent guard (matches r72_conservative_deadband.py's
    convention): the max timestamp in any frame this file touches must be
    strictly before OOS_START."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


def build_dataframe() -> tuple[pd.DataFrame, str]:
    spot, label = load_dataset(DATA_DIR, "spot")
    mvrv = compute_mvrv_features(spot, DATA_DIR, asset="BTC")
    out = spot.copy()
    for col in mvrv.columns:
        out[col] = mvrv[col]
    return out, label


DF, LABEL = build_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  (data: {LABEL}); "
      f"MVRV feature coverage: " + ", ".join(
          f"{c} {DF[c].notna().sum():,} bars from {DF[c].dropna().index[0]:%Y-%m-%d}"
          for c in [f"mvrv_roc{n}_z" for n in ROC_WINDOWS_DAYS]
      ), file=sys.stderr)


# ---------------------------------------------------------------- the votes


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote. Duplicated
    (not imported), the established precedent this project's R-53/R-54/
    R-55/R-73 lead-time studies all follow."""
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
    """Latched 0/1 hysteresis vote, identical discipline to R-53/R-54-55/
    R-73's own vote helpers. vote=0 ("stress") requires crossing ABOVE
    thresh_hi; vote=1 ("calm") requires falling back BELOW thresh_lo =
    thresh_hi - gap. Defaults to 1.0 wherever feature is NaN."""
    thresh_lo = thresh_hi - gap
    raw = np.where(feature > thresh_hi, 0.0,
                    np.where(feature < thresh_lo, 1.0, np.nan))
    return pd.Series(raw, index=feature.index).ffill().fillna(1.0)


def _daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Daily-resampled transition INTO target_value, deduplicated so
    transitions within min_gap_days of a prior one count as one episode's
    onset. Identical logic (including the shift(fill_value=False) fix R-53
    hand-caught) to every prior lead-time study in this project."""
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
    lo, hi = pd.Timestamp(TRAIN[0], tz="UTC"), pd.Timestamp(VALID[1], tz="UTC")
    frame = DF.loc[lo:hi]
    assert_no_holdout(frame)
    close = frame["close"]

    print(f"descriptive window (inner-train + inner-validation): {lo.date()} -> {hi.date()}")
    print("\nprice-anchor vote transition counts over this window (context):")
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * BARS_PER_DAY)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    for n in ROC_WINDOWS_DAYS:
        col = f"mvrv_roc{n}_z"
        feat = frame[col]
        print(f"\n{col} summary: mean={feat.mean():.2f} std={feat.std():.2f} "
              f"min={feat.min():.2f} max={feat.max():.2f} "
              f"coverage={feat.notna().sum():,}/{len(feat):,} bars")
        vote = _latched_vote(feat, THRESH_HI, GAP)
        flips_to_stress = int(((vote == 0.0) & (vote.shift() == 1.0)).sum())
        flips_to_calm = int(((vote == 1.0) & (vote.shift() == 0.0)).sum())
        print(f"  latched vote (thresh_hi={THRESH_HI}, gap={GAP}): "
              f"{flips_to_stress} stress-onset event(s), {flips_to_calm} calm-return event(s)")


# ------------------------------------------------------- step 1: lead-time (THE GATE)


def leadtime() -> None:
    lo, hi = pd.Timestamp(TRAIN[0], tz="UTC"), pd.Timestamp(VALID[1], tz="UTC")
    frame = DF.loc[lo:hi]
    assert_no_holdout(frame)
    close = frame["close"]

    votes = _anchor_votes(close, (20, 40, 80), 0.01)
    anchor_sum = sum(votes.values())
    majority_bear = (anchor_sum < 1.5).astype(float)  # 1 when >=2 of 3 anchors bearish
    majority_onsets = _daily_transitions(majority_bear, 1.0)

    print(f"3-anchor MAJORITY bear-onset episodes in window: {len(majority_onsets)}")
    print(f"  {[d.date().isoformat() for d in majority_onsets]}")

    def nearest_onset(target_date, candidates, window_days):
        best, best_dist = None, None
        for c in candidates:
            dist = (c - target_date).days
            if abs(dist) <= window_days and (best_dist is None or abs(dist) < abs(best_dist)):
                best, best_dist = c, dist
        return best, best_dist

    results = {}
    for n in ROC_WINDOWS_DAYS:
        col = f"mvrv_roc{n}_z"
        feat = frame[col]
        vote = _latched_vote(feat, THRESH_HI, GAP)
        bear = 1.0 - vote
        mvrv_onsets = _daily_transitions(bear, 1.0)
        print(f"\n{'=' * 70}\nFEATURE: {col} "
              f"({len(mvrv_onsets)} stress-onset event(s) in window)")
        print(f"  {[d.date().isoformat() for d in mvrv_onsets]}")

        leads = []
        print(f"\n  episode-by-episode (positive lead_days = MVRV flips FIRST):")
        for ep_name, epicenter in EPISODES:
            mvrv_match, mvrv_dist = nearest_onset(epicenter, mvrv_onsets, SEARCH_WINDOW_DAYS)
            anchor_match, anchor_dist = nearest_onset(epicenter, majority_onsets, SEARCH_WINDOW_DAYS)
            if mvrv_match is None or anchor_match is None:
                print(f"    {ep_name:35s} epicenter {epicenter.date()}: "
                      f"MVRV onset={'none within window' if mvrv_match is None else mvrv_match.date()}  "
                      f"anchor onset={'none within window' if anchor_match is None else anchor_match.date()}  "
                      f"-> UNMATCHED, excluded from median")
                continue
            lead_days = (anchor_match - mvrv_match).days
            leads.append(lead_days)
            print(f"    {ep_name:35s} epicenter {epicenter.date()}: "
                  f"MVRV onset={mvrv_match.date()}  anchor onset={anchor_match.date()}  "
                  f"lead_days={lead_days:+d}")

        results[col] = leads
        if leads:
            n_lead = sum(1 for x in leads if x > 0)
            print(f"\n  SUMMARY [{col}]: {len(leads)}/{len(EPISODES)} matched episode(s), "
                  f"{n_lead}/{len(leads)} MVRV-leads, median lead_days={float(np.median(leads)):+.1f}, "
                  f"individual leads={leads}")
        else:
            print(f"\n  SUMMARY [{col}]: no matched pairs -- cannot assess lead/lag")

    print(f"\n{'=' * 70}\nPRE-REGISTERED DECISION RULE: proceed to step 2 only if a MAJORITY of "
          f"matched episodes lead (either feature) AND median lead is positive.")
    for col, leads in results.items():
        if not leads:
            print(f"  {col}: NO -- no matched episodes")
            continue
        n_lead = sum(1 for x in leads if x > 0)
        median = float(np.median(leads))
        clears = (n_lead > len(leads) / 2) and (median > 0)
        print(f"  {col}: n_lead={n_lead}/{len(leads)}  median={median:+.1f}  "
              f"{'CLEARS -> proceed to step 2' if clears else 'DOES NOT CLEAR'}")
    any_clears = any(
        leads and sum(1 for x in leads if x > 0) > len(leads) / 2 and float(np.median(leads)) > 0
        for leads in results.values()
    )
    print(f"\nOVERALL: {'AT LEAST ONE FEATURE CLEARS -> proceed to step 2 (build a strategy)' if any_clears else 'NEITHER FEATURE CLEARS -> STOP, report NEGATIVE, do not build a strategy'}")


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
        print(f"usage: python experiments/r74_novel_mvrv_leadtime.py [{'|'.join(cmds)}]")
