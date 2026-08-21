#!/usr/bin/env python
"""R-81 CONSERVATIVE branch, step 1 (REQUIRED FIRST, per docs/ROUTINE.md
discipline and this round's brief): does BTC's own CoinMetrics exchange
net-flow (`FlowInExNtv - FlowOutExNtv`, positive = coins moving ONTO
exchanges, read as latent selling pressure) LEAD `kelly_regime_v4`'s own
3-anchor price-gate flip into its bear state, measured BEFORE any strategy
code is written?

THIS FILE IS THE GATE. A confirming-vote strategy (`r81_conservative_confirm.py`)
is written only if the pre-registered decision rule below resolves
favorably. The operator's brief states the honest prior explicitly: Ren,
Wu & Liu (2024, arXiv:2411.06327) found BTC's OWN net exchange inflow
*generally lacks return-forecasting power* at intraday horizons (only a
weak effect at 4h) -- this gate is expected to fail, and the job is to
measure that honestly, not manufacture a win.

Constraint attacked: INFO (one price series). Ninth attempt on this axis
(B-07/R-44 network activity, R-53 VIX/DXY, R-54/R-55/R-58 stablecoin
supply, R-73 DVOL/VRP, R-74 MVRV, R-75 calendar structure, R-79
halving-cycle phase, R-80 meta-labeling on the strategy's own vote, now
R-81 exchange net-flow) -- the first CAPITAL-CUSTODY flow rather than a
network-activity count, valuation ratio, spillover level, priced
expectation, calendar feature, cyclical phase, or self-referential
confidence signal. See `experiments/r81_shared.py`'s module docstring for
the full citation and the reasoning for why this splits from the
r81_novel branch (which takes the paper's ETH-focused / volatility-axis
finding; this branch takes the paper's own skeptical BTC-return finding,
going in expecting it to fail, per the brief).

Episode set, search window, decision-rule shape: identical to R-73/R-74/
R-75/R-79/R-80's own lead-time gates, for direct cross-round
comparability -- imported unchanged from `experiments/r81_shared.py`
(`EPISODES`, `SEARCH_WINDOW_DAYS`, `TRAIN`, `VALID`), not re-derived here.

Pre-registered decision rule (fixed BEFORE any lead-time number is
computed -- copied verbatim from this round's brief)
-----------------------------------------------------------------------------
Proceed to step 2 (build a confirming-vote strategy) ONLY IF a MAJORITY of
the matched (both sides found within +/-90 days) episodes have
`lead_days > 0` (net-flow-bear onset strictly earlier than the 3-anchor
majority-bear onset) AND the median `lead_days` across matched episodes is
positive. If fewer than 3 of the 5 episodes match on both sides, that also
counts as a failure (too little evidence to justify building on).

If the rule fails: STOP. Do not write any strategy code. Report NEGATIVE
at the gate.

Usage
-----
    python experiments/r81_conservative_leadtime.py descriptive
    python experiments/r81_conservative_leadtime.py leadtime
    python experiments/r81_conservative_leadtime.py all
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

from experiments.r81_shared import (  # noqa: E402
    DATA_DIR,
    EPISODES,
    SEARCH_WINDOW_DAYS,
    TRAIN,
    VALID,
    anchor_votes,
    assert_no_holdout,
    daily_transitions,
    nearest_onset,
    net_flow_on_bars,
)

THRESH_HI = 1.0   # fixed a-priori, matches R-53/R-54/R-55/R-73/R-74's own primary, not swept for this gate
GAP = 0.75        # fixed a-priori, matches R-53/R-54/R-55/R-73/R-74's own primary, not swept for this gate
ROLL_WINDOW_DAYS = 90  # trailing z-score window on the DAILY net_flow series, fixed a priori


def build_dataframe() -> tuple[pd.DataFrame, str]:
    """BTC spot bars + causally-aligned net_flow_z (z-scored on the DAILY
    series BEFORE alignment onto 5m bars, per the brief -- z-scoring after
    alignment would compute the rolling window in bar-units, not
    day-units)."""
    spot, label = load_dataset(DATA_DIR, "spot")

    from experiments.r81_shared import load_net_flow
    daily = load_net_flow("BTC")
    daily_z = (daily["net_flow"] - daily["net_flow"].rolling(ROLL_WINDOW_DAYS).mean()) / \
        daily["net_flow"].rolling(ROLL_WINDOW_DAYS).std()
    daily_z = daily_z.to_frame("net_flow_z")

    from tradebot.data import align_onchain_causal
    aligned = align_onchain_causal(daily_z, spot)

    out = spot.copy()
    out["net_flow_z"] = aligned["net_flow_z"]
    return out, label


DF, LABEL = build_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  (data: {LABEL}); "
      f"net_flow_z coverage: {DF['net_flow_z'].notna().sum():,} bars from "
      f"{DF['net_flow_z'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    lo, hi = pd.Timestamp(TRAIN[0], tz="UTC"), pd.Timestamp(VALID[1], tz="UTC")
    frame = DF.loc[lo:hi]
    assert_no_holdout(frame)
    close = frame["close"]

    print(f"descriptive window (inner-train + inner-validation): {lo.date()} -> {hi.date()}")
    print("\nprice-anchor vote transition counts over this window (context):")
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * 288)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    feat = frame["net_flow_z"]
    print(f"\nnet_flow_z summary: mean={feat.mean():.2f} std={feat.std():.2f} "
          f"min={feat.min():.2f} max={feat.max():.2f} "
          f"coverage={feat.notna().sum():,}/{len(feat):,} bars")

    bear = np.where(feat > THRESH_HI, 1.0, np.where(feat < THRESH_HI - GAP, 0.0, np.nan))
    bear = pd.Series(bear, index=feat.index).ffill().fillna(0.0)
    flips_to_stress = int(((bear == 1.0) & (bear.shift() == 0.0)).sum())
    flips_to_calm = int(((bear == 0.0) & (bear.shift() == 1.0)).sum())
    print(f"  latched net-flow-bear vote (thresh_hi={THRESH_HI}, gap={GAP}): "
          f"{flips_to_stress} stress-onset event(s), {flips_to_calm} calm-return event(s)")


# ------------------------------------------------------- step 1: lead-time (THE GATE)


def leadtime() -> dict:
    lo, hi = pd.Timestamp(TRAIN[0], tz="UTC"), pd.Timestamp(VALID[1], tz="UTC")
    frame = DF.loc[lo:hi]
    assert_no_holdout(frame)
    close = frame["close"]

    # kelly_regime_v4's own 3-anchor MAJORITY-bear onset dates.
    votes = anchor_votes(frame)
    anchor_sum = sum(votes)
    majority_bear = (anchor_sum < 1.5).astype(float)  # 1 when >=2 of 3 anchors bearish
    majority_onsets = daily_transitions(majority_bear, 1.0)
    print(f"3-anchor MAJORITY bear-onset episodes in window: {len(majority_onsets)}")
    print(f"  {[d.date().isoformat() for d in majority_onsets]}")

    # Net-flow-bear onset dates (latched hysteresis vote on net_flow_z, bear=1=stress).
    feat = frame["net_flow_z"]
    bear = np.where(feat > THRESH_HI, 1.0, np.where(feat < THRESH_HI - GAP, 0.0, np.nan))
    bear = pd.Series(bear, index=feat.index).ffill().fillna(0.0)
    netflow_onsets = daily_transitions(bear, 1.0)
    print(f"\nnet-flow-bear onset episodes in window ({len(netflow_onsets)}):")
    print(f"  {[d.date().isoformat() for d in netflow_onsets]}")

    leads = []
    matched = 0
    print(f"\nepisode-by-episode (positive lead_days = net-flow signal flips FIRST):")
    print(f"{'episode':35s} {'netflow onset':>14s} {'anchor onset':>14s} {'lead_days':>10s} matched")
    for ep_name, epicenter in EPISODES:
        nf_match, nf_dist = nearest_onset(epicenter, netflow_onsets, SEARCH_WINDOW_DAYS)
        anchor_match, anchor_dist = nearest_onset(epicenter, majority_onsets, SEARCH_WINDOW_DAYS)
        if nf_match is None or anchor_match is None:
            print(f"{ep_name:35s} {'none' if nf_match is None else nf_match.date().isoformat():>14} "
                  f"{'none' if anchor_match is None else anchor_match.date().isoformat():>14} "
                  f"{'--':>10} NO")
            continue
        lead_days = (anchor_match - nf_match).days
        leads.append(lead_days)
        matched += 1
        print(f"{ep_name:35s} {nf_match.date().isoformat():>14} {anchor_match.date().isoformat():>14} "
              f"{lead_days:>+10d} YES")

    n_lead = sum(1 for x in leads if x > 0)
    median = float(np.median(leads)) if leads else float("nan")
    print(f"\nSUMMARY: matched={matched}/{len(EPISODES)}  leading={n_lead}/{matched if matched else 0}  "
          f"median lead_days={median:+.1f}" if leads else f"\nSUMMARY: matched={matched}/{len(EPISODES)} -- no matched pairs")

    majority_leads = matched >= 3 and n_lead > matched / 2
    positive_median = matched >= 3 and leads and median > 0
    clears = bool(matched >= 3 and majority_leads and positive_median)

    print(f"\n{'=' * 78}")
    print("PRE-REGISTERED DECISION RULE: proceed to step 2 only if >=3 episodes match, "
          "a MAJORITY of matched episodes lead (lead_days>0), AND median lead_days>0.")
    print(f"  matched={matched} (>=3: {matched >= 3})  "
          f"majority-leading={n_lead}/{matched if matched else 0} ({majority_leads})  "
          f"median={median:+.1f} (>0: {positive_median})")
    print(f"\nOVERALL: {'CLEARS -> proceed to step 2 (build a strategy)' if clears else 'DOES NOT CLEAR -> STOP, report NEGATIVE, do not build a strategy'}")

    return dict(matched=matched, n_lead=n_lead, median=median, leads=leads, clears=clears)


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
        print(f"usage: python experiments/r81_conservative_leadtime.py [{'|'.join(cmds)}]")
