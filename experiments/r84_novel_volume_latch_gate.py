#!/usr/bin/env python
"""R-84 NOVEL branch operator-style measurement: the Step-A detection-lag
gate for volume-modulated anchor-latch confirmation speed, run BEFORE any
strategy code is built -- same "operator-measurement-before-branch"
convention R-78/R-80/R-81/R-82/R-83 all used (fixed, non-swept measurement,
so paying for it once is correct).

=============================================================================
PRE-REGISTRATION (frozen before this file was ever run against real data)
=============================================================================

1. MECHANISM, one sentence: `kelly_regime_v4`'s three anchor votes flip the
   instant price crosses a FIXED 1% band around each anchor and then latch
   with no notion of how much evidence a crossing carries; grounded in
   Easley & O'Hara's (1992, J. Finance 47(2)) sequential-trade result that
   informed trading concentrates volume around genuine information events,
   this branch narrows each anchor's effective band (requiring LESS price
   movement, i.e. less confirmation, to flip) when volume is unusually
   elevated at the crossing bar, and widens it (requiring MORE confirmation)
   when volume is unremarkable or low -- see
   `r84_novel_volume_latch_shared.py`'s module docstring for the full
   `band_eff(t) = V4_BAND * f(volume_z(t))` construction, its three fixed
   parameters (FLOOR_RATIO=0.40, CAP_RATIO=1.60, GAIN=1.0, all reasoned
   before any real-data number was read, not fit to one), and the causality
   argument. This directly targets the vote/direction factor R-62 showed
   carries v4's real matched-exposure drawdown signature, via a mechanism
   (modulating the vote's OWN confirmation dynamics) no prior round in this
   ledger has tried -- see `r84_novel_volume_latch_shared.py` and
   `r84_shared.py` for the full not-a-duplicate list (nine INFO-axis
   rounds, four independently-confirmed degenerate brakes, R-62's
   vote-vs-scale isolation, R-80's discrete-vote lesson).

2. DETECTION-LAG DEFINITION, mirroring R-82/R-83's own construction on
   this identical six-episode table, but adapted per the operator's
   explicit instruction to compare the modified vote against v4's OWN
   unmodified vote (not merely against the onset date): for each episode
   in `r84_shared.STRESS_EPISODES`, within a +/-60-day search window
   around its onset (`episode_window`):
   - v4's OWN reaction (reference): nearest DOWNWARD transition of the
     UNMODIFIED `anchor_majority` (`r84_shared.anchor_majority`, fixed
     V4_BAND=0.01 on all three anchors) to the onset (`nearest_transition`,
     direction="down").
   - This branch's reaction: nearest DOWNWARD transition of the
     VOLUME-MODULATED `anchor_majority_volume_modulated` to the onset,
     same window, same `nearest_transition` call.
   - LEAD = (v4_own_flip_time - modified_flip_time) in days. Positive =
     the volume-modulated vote detected the break BEFORE v4's own
     unmodified vote reacted to the identical price series.
   Each side's own lag-to-onset is also reported (descriptive context,
   not part of the gating decision), and a block-bootstrap null on the
   modified vote's LEAD (`block_bootstrap_shifts`, block_days=5, N=500,
   seed=84, byte-for-byte the same construction R-81/R-82/R-83 used) is
   reported as a SECONDARY, non-gating diagnostic -- see point 4.

3. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed -- this is the operator's own literal instruction, adopted
   verbatim as the binding rule rather than substituted for a different
   one after the fact): PROCEED TO STEP B (build a real strategy variant)
   ONLY IF the modified vote detects STRICTLY EARLIER than v4's own
   unmodified vote (LEAD > 0, strictly) on a MAJORITY (>= 4 of 6) of the
   six episodes. If fewer than 4 of 6 show genuine improvement: STOP,
   report this file's result as the round's whole product, write it up as
   NEGATIVE, do not build any further strategy code. The bar is not
   relaxed after seeing the numbers.

4. SECONDARY (NON-GATING) DIAGNOSTIC, included for comparability with
   R-82/R-83's own reporting style but explicitly NOT part of the frozen
   stop rule above: does the observed LEAD also clear the block-bootstrap
   null's median (`leads_null`, same construction as R-82/R-83)? This is
   reported per episode so a skeptic can see whether a "pass" under rule 3
   is also non-degenerate against a shift-null, or is a small positive
   LEAD indistinguishable from noise -- but it does not change which
   episodes count toward the 4-of-6 threshold, because rule 3 is the
   operator's literal pre-registered instruction and this file does not
   substitute a different (even if arguably more rigorous) rule for it
   after the fact.

5. WHAT WOULD MAKE THIS GATE FAIL, named now, before any number was
   computed: fewer than 4 of 6 episodes show a strictly earlier downward
   flip under the volume-modulated vote than under v4's own unmodified
   vote. Two failure sub-modes are named in advance, by analogy with
   R-83's Kalman gate (the closest structural template):
   (i) NO EFFECT: because `band_eff` is bounded to [0.40, 1.60] x V4_BAND
       and volume_z reverts to its own trailing mean quickly, the modified
       vote's flip dates could be IDENTICAL to v4's own on most episodes
       (LEAD == 0 exactly) if the sudden volume spike that accompanies a
       real crash arrives at the same bar the price crossing itself would
       have happened anyway -- band narrowing only pulls a flip earlier if
       price is ALREADY close enough to the anchor edge that a narrower
       band is crossed on an earlier bar than the original 1% band would
       have been. On a violent one-bar crash this margin may not exist:
       price can gap through both the narrow and the wide band in the
       same bar, in which case narrowing the band cannot pull detection
       earlier at all, no matter how large the accompanying volume spike
       is -- the mechanism can only advance detection by however many
       bars price was already "in flight" toward the original band edge
       before the volume spike hit.
   (ii) WRONG-SIDED: elevated volume at a crossing might not reliably mark
       genuine new information at ALL of these six specific episodes (some
       are multi-week grinds, e.g. the 2018 bear onset, where no single
       violent-volume bar exists to modulate anything) -- if participation
       is not actually elevated near the moment v4's own vote flips on
       these particular episodes, `band_eff` sits near its neutral value
       (ratio near 1.0) throughout, and LEAD stays at or near 0 by
       construction, not because the theory is wrong but because these six
       episodes may not be the ones where a volume signature is sharpest.
   PRE-REGISTERED EXPECTATION, stated before this file was run against
   real data: mode (i) (no-effect / LEAD == 0 on most episodes) is
   expected to be the dominant failure mode if the gate fails, because a
   BAND-WIDTH modulation is mechanically a much smaller lever on flip
   timing than either BOCPD's run-length posterior (R-82) or the Kalman
   filter's continuously-updated slope estimate (R-83) -- both of THOSE
   mechanisms replace the entire detection statistic, while this one only
   perturbs a threshold by, at most, 60% around a level the underlying
   price series was going to cross anyway. A materially positive result
   here would therefore be a genuinely stronger finding than either prior
   novel-branch failure, precisely because the mechanism has less room to
   move a flip date than either of the two constructions that already
   failed this identical gate.

CONFIGURATIONS EVALUATED AGAINST REAL MARKET DATA IN THIS FILE: 0 (a
fixed, non-swept measurement gate, using
`r84_novel_volume_latch_shared`'s three named parameters throughout --
no sweep against real BTC data occurs here; a Step-B sweep, if the gate
passes, is pre-registered separately, before any Step-B number is read).

Run: ``python experiments/r84_novel_volume_latch_gate.py``
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

from experiments.r84_shared import OOS_START, STRESS_EPISODES  # noqa: E402
from experiments.r84_novel_volume_latch_shared import (  # noqa: E402
    CAP_RATIO,
    FLOOR_RATIO,
    GAIN,
    anchor_majority_volume_modulated,
    assert_no_holdout,
    block_bootstrap_shifts,
    confirmation_ratio,
    episode_window,
    nearest_transition,
    truncation_causality_probe,
    v4_anchor_majority,
    volume_z,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 84


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def f_sanity_table() -> None:
    print("-" * 78)
    print(f"confirmation_ratio f(z) sanity table "
          f"(FLOOR={FLOOR_RATIO}, CAP={CAP_RATIO}, GAIN={GAIN}; no market data read)")
    print("-" * 78)
    z = pd.Series([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    f = confirmation_ratio(z)
    for zi, fi in zip(z, f):
        print(f"  z={zi:+.1f}  f(z)={fi:.3f}  band_eff = {fi*1.0:.3f} x V4_BAND")
    print()


def causality_probe(bars: pd.DataFrame) -> None:
    print("-" * 78)
    print("Causality probe: does the volume-modulated majority[check_at] change "
          "if bars after it are dropped?")
    print("-" * 78)

    def build_majority(df: pd.DataFrame) -> np.ndarray:
        return anchor_majority_volume_modulated(df).to_numpy()

    for check_at in (250_000, 350_000):
        ok = truncation_causality_probe(build_majority, bars, check_at)
        print(f"  check_at={check_at:>7d}: {'PASS (causal)' if ok else 'FAIL (LOOKAHEAD)'}")
        assert ok, "volume-modulated anchor majority is not causal -- stop, do not trust this gate"

    # also probe the raw volume_z feature in isolation, since that is the
    # new causal ingredient this branch introduces
    def build_volz(df: pd.DataFrame) -> np.ndarray:
        return volume_z(df).to_numpy()

    for check_at in (250_000, 350_000):
        ok = truncation_causality_probe(build_volz, bars, check_at)
        print(f"  volume_z check_at={check_at:>7d}: {'PASS (causal)' if ok else 'FAIL (LOOKAHEAD)'}")
        assert ok, "volume_z is not causal -- stop, do not trust this gate"
    print()


def null_leads(modified: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                v4_flip_time: pd.Timestamp, n_draws: int = N_DRAWS,
                block_days: int = BLOCK_DAYS, seed: int = NULL_SEED) -> np.ndarray:
    local = modified.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        changed = np.zeros(n_bars, dtype=bool)
        changed[1:] = shifted[1:] < shifted[:-1]
        idx = np.where(changed)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        detect_time = times[int(np.argmin(deltas))]
        leads[k] = (v4_flip_time - detect_time).total_seconds() / 86400.0
    return leads


def gate() -> dict:
    print("=" * 78)
    print("R-84 NOVEL MEASUREMENT: volume-modulated anchor latch vs v4's own "
          "unmodified anchor latch -- STEP A detection-lag gate")
    print("=" * 78)

    f_sanity_table()

    bars = load_btc_bars()
    causality_probe(bars)

    v4_majority = v4_anchor_majority(bars)
    mod_majority = anchor_majority_volume_modulated(bars)
    assert_no_holdout(bars)

    print(f"search window=+/-{WINDOW_DAYS}d  null: {N_DRAWS} draws, "
          f"block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, pass_primary=False, lead=float("nan")))
            continue

        v4_flip = nearest_transition(v4_majority, window, onset, direction="down")
        mod_flip = nearest_transition(mod_majority, window, onset, direction="down")

        if v4_flip is None or mod_flip is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no v4 transition' if v4_flip is None else 'no modified-vote transition'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, pass_primary=False, lead=float("nan")))
            continue

        lead = (v4_flip - mod_flip).total_seconds() / 86400.0
        v4_lag_to_onset = (v4_flip - onset).total_seconds() / 86400.0
        mod_lag_to_onset = (mod_flip - onset).total_seconds() / 86400.0

        leads_null = null_leads(mod_majority, window, onset, v4_flip)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")

        pass_primary = lead > 0.0  # THE frozen, binding rule (point 3 above)
        pass_secondary = pass_primary and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 (unmodified) nearest downward flip:  {v4_flip}  "
              f"(lag to onset {v4_lag_to_onset:+.2f}d)")
        print(f"    volume-modulated nearest downward flip: {mod_flip}  "
              f"(lag to onset {mod_lag_to_onset:+.2f}d)")
        print(f"    LEAD (v4_flip - modified_flip) = {lead:+.2f} days   "
              f"null median={null_median:+.2f}d (valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (primary, frozen rule) lead>0: {pass_primary}   "
              f"(secondary diagnostic, non-gating) lead>=null median: {pass_secondary}")

        results.append(dict(label=label, onset=onset_str, v4_flip=v4_flip, mod_flip=mod_flip,
                             lead=lead, v4_lag_to_onset=v4_lag_to_onset,
                             mod_lag_to_onset=mod_lag_to_onset, null_median=null_median,
                             pass_primary=pass_primary, pass_secondary=pass_secondary))

    n_pass = sum(1 for r in results if r["pass_primary"])
    n_pass_secondary = sum(1 for r in results if r.get("pass_secondary"))
    passed = n_pass >= 4

    print("\n" + "=" * 78)
    print(f"{'episode':42s} {'lead(d)':>9s} {'PRIMARY':>8s} {'secondary':>10s}")
    for r in results:
        print(f"  {r['label']:40s} {r.get('lead', float('nan')):>+9.2f} "
              f"{str(r['pass_primary']):>8s} {str(r.get('pass_secondary', False)):>10s}")
    print(f"\nEpisodes passing PRIMARY (frozen) rule, lead>0: {n_pass}/6")
    print(f"Episodes also passing secondary (null-median) diagnostic: {n_pass_secondary}/6")
    print(f"GATE VERDICT (primary rule): "
          f"{'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated against real market data in this file: 0 "
          f"(fixed measurement gate)")
    print(f"max timestamp read anywhere in this session: {bars.index.max()}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, n_pass_secondary=n_pass_secondary, passed=passed)


if __name__ == "__main__":
    gate()
