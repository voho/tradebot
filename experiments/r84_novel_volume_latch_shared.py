"""Shared, read-only utilities for the R-84 NOVEL branch (08-21):
volume-modulated anchor-latch confirmation speed.

Mechanism, one sentence (restated from the operator's dispatch and
`r84_shared.py`'s module docstring, which carries the full citation trail
and not-a-duplicate rationale this file does not repeat): `kelly_regime_v4`'s
three anchor votes each flip the instant price crosses a FIXED 1% band
around their own rolling-mean anchor and then latch until the opposite
crossing, with no notion of how much evidence that crossing carries;
grounded in the Mixture-of-Distributions Hypothesis (Clark 1973; Tauchen &
Pitts 1983) and Easley & O'Hara's (1992, J. Finance 47(2)) sequential-trade
result that informed trading concentrates volume around genuine
information events, this branch makes each anchor's effective band NARROWER
(so a crossing needs LESS price movement, i.e. less confirmation, to flip
the vote) at bars where participation is unusually high, and WIDER (more
confirmation required) at bars where participation is unremarkable or low.

This is the "combination architecture not yet tried" the operator specified:
it modulates the vote's OWN latch/confirmation dynamics directly, rather
than (a) adding volume as a 4th confirming vote (`confirming_vote_frac`,
the conservative branch's job, R-53/R-55's validated formula) or
(b) multiplying the resulting exposure by a bounded brake (the four
independently-confirmed degenerate family: R-34, R-41-conservative,
R-53-conservative, R-73-conservative, all R^2 > 0.95 flat-rescale-of-v4's-
own-exposure-path). Neither prior failure mode applies here: this file
never touches `target`/exposure at all, and there is no cumulative
never-increase-only state -- `band_eff(t)` is a memoryless, bounded
function of `volume_z(t)` alone, recomputed fresh every bar.

------------------------------------------------------------------------
EXACT CONSTRUCTION (frozen before any real BTC/ETH number was computed)
------------------------------------------------------------------------

For each of v4's 3 anchor horizons (20/40/80 days), the fixed band
`V4_BAND = 0.01` is replaced by a per-bar effective band

    band_eff(t) = V4_BAND * f(volume_z(t))

    f(z) = FLOOR_RATIO + (CAP_RATIO - FLOOR_RATIO) / (1 + exp(GAIN * z))

using `r84_shared.volume_z` UNCHANGED (causal log-volume z-score against
its own trailing 20-day mean/std -- imported, not duplicated, since this
is this round's own shared prep file, not a sibling round's module the
"duplicated not imported" convention (R-54/R-55) is protecting against).
`z` is filled to 0.0 wherever `volume_z` is NaN (warmup / degenerate std)
so `f(0) = 1.0` exactly and the mechanism recovers `band_eff == V4_BAND`
identically -- the same "weight==0 recovers v4 exactly" identity-recovery
discipline every confirming-vote round in this project's history has
carried (R-53/R-55/R-80/R-81).

Parameters, fixed now, reasoned rather than fit to any real market number
(the same "fixed before touching real data" discipline R-82's
`hazard_lambda` and R-83's `SIGMA_ZETA` used, scaled down here because the
construction has only one nonlinear knob, GAIN, rather than three):

    FLOOR_RATIO = 0.40   # band can shrink to 40% of nominal (as low as
                          # 0.40%) when volume is very elevated
    CAP_RATIO   = 1.60   # band can widen to 160% of nominal (up to 1.60%)
                          # when volume is very low -- FLOOR/CAP chosen
                          # symmetric about 1.0 (0.40 = 1/2.5, 1.60 =
                          # 1 + 0.6, i.e. +/-60% around the neutral point)
                          # so the mechanism does not carry a built-in bias
                          # toward faster or slower confirmation on average
    GAIN        = 1.0    # z is a z-score (unit variance by construction),
                          # so GAIN=1.0 means a "typical" +/-1 sigma
                          # participation event moves f about a third of
                          # the way to its floor/cap, and a +/-2 sigma
                          # event -- a genuinely unusual bar -- moves it
                          # most of the way there (see the printed sanity
                          # table in the gate file: f(-2)=1.46, f(-1)=1.24,
                          # f(0)=1.00, f(+1)=0.76, f(+2)=0.54). No grid
                          # search against real data selected these three
                          # numbers; they are a deliberately simple,
                          # symmetric, bounded first construction, exactly
                          # as the operator's brief asked for.

`band_eff` is shared across all 3 anchors (one `volume_z`, one `f(z)`
series, reused for the 20/40/80-day anchors) rather than a separate
per-horizon volume window -- the simplest construction that still tests
the mechanism, per the operator's explicit "keep it as simple as you can
while still being a genuine test" instruction. A per-horizon volume
window is a legitimate future elaboration, not attempted here.

Causality: `band_eff(t)` depends only on `volume[<=t]` (via `volume_z`,
itself causal per `r84_shared.volume_z`'s own docstring), and the crossing
test at bar t compares `close[t]` against `anchor[t] * (1 +/- band_eff[t])`
where `anchor[t]` is a rolling mean of `close[<=t]`. This is the identical
"row i depends only on rows <= i" contract `r84_shared.anchor_votes`
itself already uses (bar t's own close is compared against bar t's own
threshold) -- band_eff(t) using volume[t] is no more forward-looking than
v4's existing vote using close[t]. Verified empirically, not merely
argued, by `truncation_causality_probe` in the gate file.

Not a duplicate of the conservative branch (separate file, adds volume as
a 4th DISCRETE confirming vote via `confirming_vote_frac`) or of any prior
brake round (no exposure multiplication, no cumulative state).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r84_shared import (  # noqa: E402
    BARS_PER_DAY,
    OOS_START,
    STRESS_EPISODES,
    V4_BAND,
    V4_HORIZONS,
    anchor_votes as v4_anchor_votes,
    anchor_majority as v4_anchor_majority,
    assert_no_holdout,
    block_bootstrap_shifts,
    truncation_causality_probe,
    volume_z,
)

# ------------------------------------------------------- mechanism params

FLOOR_RATIO = 0.40
CAP_RATIO = 1.60
GAIN = 1.0
VOL_WINDOW_DAYS = 20  # r84_shared.volume_z's own default, reused unchanged


def confirmation_ratio(z: pd.Series, floor_ratio: float = FLOOR_RATIO,
                        cap_ratio: float = CAP_RATIO, gain: float = GAIN) -> pd.Series:
    """`f(z)` -- bounded, memoryless, symmetric-about-1.0 logistic squash.

    `z` is filled to 0.0 first (NaN -> f=1.0, exact v4 recovery). Clipped
    inside the exponential to +/-50 purely for float safety; volume_z's
    own trailing-window z-score essentially never reaches that magnitude.
    """
    zz = z.fillna(0.0).clip(-50.0, 50.0)
    return floor_ratio + (cap_ratio - floor_ratio) / (1.0 + np.exp(gain * zz))


def anchor_votes_volume_modulated(
    df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
    floor_ratio: float = FLOOR_RATIO, cap_ratio: float = CAP_RATIO, gain: float = GAIN,
    vol_window_days: int = VOL_WINDOW_DAYS,
) -> list[pd.Series]:
    """v4's 3 anchor votes, but with a per-bar volume-modulated effective
    band in place of the fixed `V4_BAND`. Causal: see module docstring.
    """
    close = df["close"]
    z = volume_z(df, window_days=vol_window_days)
    ratio = confirmation_ratio(z, floor_ratio, cap_ratio, gain)
    band_eff = (band * ratio).to_numpy()

    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean().to_numpy()
        c = close.to_numpy()
        raw = np.where(c > anchor * (1.0 + band_eff), 1.0,
                        np.where(c < anchor * (1.0 - band_eff), 0.0, np.nan))
        v = pd.Series(raw, index=df.index)
        votes.append(v.ffill().fillna(0.0))
    return votes


def anchor_majority_volume_modulated(
    df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
    floor_ratio: float = FLOOR_RATIO, cap_ratio: float = CAP_RATIO, gain: float = GAIN,
    vol_window_days: int = VOL_WINDOW_DAYS,
) -> pd.Series:
    votes = anchor_votes_volume_modulated(df, horizons, band, floor_ratio, cap_ratio,
                                           gain, vol_window_days)
    return sum(votes) / len(votes)


# --------------------------------------------------------- Step-A gate infra
# Duplicated from r82_shared/r83_novel_kalman_shared (byte-for-byte
# construction, not the functions themselves, since neither of those
# modules exports what's needed and this round's own convention is
# "import r84_shared, duplicate what a sibling round's file would have
# had to duplicate anyway").

def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
    """Timestamp, within `window`, of the `series` transition closest to
    `onset`."""
    vals = series.reindex(window).to_numpy()
    changed = np.zeros(len(vals), dtype=bool)
    if direction == "down":
        changed[1:] = vals[1:] < vals[:-1]
    elif direction == "any":
        changed[1:] = vals[1:] != vals[:-1]
    else:
        raise ValueError(f"unknown direction {direction!r}")
    idx = np.where(changed)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars: pd.DataFrame, onset_str: str, window_days: int = 60
                    ) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window
