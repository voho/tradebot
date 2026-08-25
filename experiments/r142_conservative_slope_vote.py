"""R-142 CONSERVATIVE branch: Deribit front-vs-next-quarter futures
TERM-STRUCTURE SLOPE, z-scored against its own trailing baseline, as a
confirming vote on `kelly_regime_v4`'s 3-anchor gate -- Step A measurement
gate first, this project's established discipline for every INFO-axis
round since R-53 (R-53/R-73/R-74/R-79/R-81/R-84/R-120).

Shared infrastructure (loader, causal dual-quarter slope construction,
z-score, anchor-vote duplication, confirming-vote rule, stress table,
null generator, causality probe, frozen coverage/episode constants) lives
in `experiments/r142_shared.py`, written and frozen by the operator BEFORE
this file computed any gate/backtest number. This file does not edit that
module. `data/btcusd_deribit_quarterly_5m.csv.gz` and the ETH equivalent
were re-fetched by the operator (unmodified `scripts/
fetch_deribit_quarterly_futures.py`, extended `--last-expiry`) before this
round started, to close the coverage gap R-120 left at 2023-03-31 -- see
r142_shared.py's own module docstring for the disclosed coverage numbers.

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). The SLOPE between the two nearest,
   simultaneously-traded Deribit quarterly futures (next-quarter
   annualized basis minus front-quarter annualized basis) is a
   cross-sectional curve-shape signal -- distinct from R-120's own
   single-point LEVEL and MOMENTUM statistics -- that reflects real-time
   term-structure repricing by cash-and-carry and calendar-spread traders,
   and may therefore reach an extreme before `kelly_regime_v4`'s slow
   20/40/80-day price anchors catch up to a genuine regime shift.

   Citations: Bianchi, Fan, Miffre & Zhang (2023), "Exploiting the
   dynamics of commodity futures curves", Journal of Banking & Finance
   (arXiv 2308.00383) -- Nelson-Siegel slope is a separately profitable,
   uncorrelated factor from curve level in commodities, the direct
   citation for why slope is not a re-parameterization of R-120's own
   level/momentum result; Erb & Harvey (2006), Financial Analysts Journal
   62(2); Schmeling, Schrimpf & Todorov (2023/2025), BIS WP 1087; Chi et
   al. (2023), Journal of Futures Markets -- background term-structure/
   crypto-carry literature, R-120's own citations, reused for context.
   Full citation trail and the discipline note on NOT using the 2025-11
   BTC backwardation/bottom episode to pick this round's gate (found
   during this round's own literature search, deliberately excluded from
   both the episode table and this branch's threshold) are in
   `r142_shared.py`'s module docstring -- one citation trail in one place
   (R-81/R-84/R-120's own convention).

   CONSTRAINT ATTACKED: INFO (one price series) -- the SLOPE needs a
   second, simultaneously-listed instrument this project's OHLCV alone
   cannot express; R-120's own front-quarter-only module never computed
   it.

   NOT A DUPLICATE OF: R-120 (front-quarter basis LEVEL and MOMENTUM,
   single maturity point; ruled out, see docs/LEDGER.md's section C) --
   the sibling NOVEL branch also does not touch this branch's Step-A
   architecture, testing a continuous SIZE-axis dampener instead of an
   INFO-axis confirming vote. R-41/`kelly_regime_v9_basis_lead` (spot-vs-
   perpetual basis, funding-reset every 8h); R-73 (DVOL, implied vol, not
   a forward price); R-81 (OI/positioning, not a priced curve quantity);
   R-63/R-76 (cross-COIN pairs, not cross-MATURITY). Grepped
   docs/LEDGER.md for "term structure", "slope", "curve steepness",
   "calendar spread", "quarterly future": only R-120's own LEVEL/MOMENTUM
   entries and this round's own r142_shared.py hit; zero prior SLOPE
   attempts.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy code,
   BTC first, on `r142_shared.USABLE_EPISODES_BTC` (4 of the full
   6-episode table, per this round's own disclosed coverage measurement
   in r142_shared.py -- COVID, 2021-11 top, Terra/Luna, FTX; the two 2018
   episodes are unreachable, no quarterly contract existed yet).

   PRIMARY FEATURE (chosen now, before any number): `slope_z` --
   `r142_shared.dual_quarter_slope(...).slope` z-scored against its own
   trailing `window_days=20` mean/std via `r142_shared.slope_zscore`
   (matches `kelly_regime_v4`'s own fastest anchor, R-120's own
   window-choice convention).

   THRESHOLD: BIDIRECTIONAL, `|slope_z| >= 1.5` -- matching R-81/R-84/
   R-120's own "extreme" convention; a curve-shape signal can indicate
   stress via inversion (extreme backwardation) or via a crowded, blown-
   off contango, so no directional assumption is fixed before the data is
   seen.

   EPISODE-LOCAL SEARCH WINDOW: [onset - 60 days, onset + 60 days], fixed
   before any number was computed, identical to R-81/R-84/R-120's window
   and for the identical reason (v4's own anchors lag price, so the
   nearest-transition search needs room on both sides).

   ANCHOR-GATE "FLIP" DEFINITION and BASIS "CROSSING" DEFINITION: reused
   verbatim from R-81/R-120's disclosed, bug-fixed convention
   (`r142_shared.nearest_transition`, `direction="down"` for the anchor
   flip; the first bidirectional `|slope_z| >= 1.5` crossing for the
   signal) -- all 4 usable BTC episodes are bearish transitions, so
   "down" is the relevant anchor-flip direction; "either direction" is not
   used for the anchor side at all, per R-81's own disclosed lesson.

   NULL: `r142_shared.block_bootstrap_lead_null`, identical construction
   to R-81/R-84/R-120 (circular block-shift, block_days chosen to exceed
   the signal's own autocorrelation length, n_draws=1000, seed fixed
   before any real-data number is read).

   STEP-A PASS BAR: a usable episode counts as a PASS only if (a) a
   `|slope_z| >= 1.5` crossing exists inside the episode's own search
   window, (b) that crossing occurs STRICTLY BEFORE the anchor-gate's own
   down-flip inside the same window, and (c) the measured lead time beats
   the block-bootstrap null's own 90th percentile (one-sided, matching
   R-81/R-84/R-120's own bar). BTC's gate passes only if
   `count(PASS) >= r142_shared.MIN_EPISODES_PASS_BTC` (>= 3 of 4).

3. DECISION RULE (frozen now):
   - If BTC's Step-A gate FAILS (< 3 of 4 usable episodes pass, matching
     the bar every one of the 19 prior INFO-axis signals in this ledger
     has been held to): STOP. Report NEGATIVE at Step A. Do not build a
     confirming-vote strategy, do not touch ETH, do not touch the
     holdout. This is the modal outcome by this ledger's own base rate
     (0-2/6 or 0-2/4 on every one of R-53 through R-141's INFO-axis Step-A
     gates to date) and is named as such, in advance, not as a hedge
     written after seeing a number.
   - If BTC's Step-A gate PASSES (>= 3 of 4): proceed to Step B. Build the
     confirming vote via `r142_shared.confirming_vote_frac(anchor_sum,
     meta_vote, weight)`, where `meta_vote = 1` when `|slope_z| >= 1.5`
     AND its sign at the crossing instant matches the majority sign
     observed across the PASSING BTC episodes in Step A (a mechanical,
     data-driven-but-pre-committed rule, not a free directional choice --
     analogous to R-81/R-84 fixing `direction="down"` for the anchor side
     from the episode table's own known character, not from a fit).
     Sweep `weight` in `{0.5, 1.0, 2.0}` (3 configurations), confirm the
     `weight=0` identity-recovery check, then run docs/ROUTINE.md's
     standard Step 3 (inner-train/inner-validation only, both BTC and ETH,
     spot and futures) and Step 4 (holdout, only after Step 3 looks
     favourable) exactly as R-53/R-55/R-120 did, against this project's
     standing promotion bar (docs/ROUTINE.md's "the promotion bar --
     default is REJECT" section) -- beats buy_and_hold OOS after real
     costs, improvement exceeds the +/-0.2 Sharpe noise floor or is a
     genuine drawdown/tail improvement, survives the 0.40% fee tier, and
     the parameter neighbourhood is a plateau (report weight=0.5/1.0/2.0
     together, not just the best).

4. WHAT WOULD MAKE IT FAIL (named now): fewer than 3 of 4 BTC episodes
   show a `|slope_z|>=1.5` crossing leading the anchor's down-flip inside
   +/-60 days, distinguishable from the block-bootstrap null -- i.e. the
   curve-shape extreme is not, in fact, earlier than v4's own slow anchors
   at the episodes this project already uses to judge every other
   regime-adjacent signal.

CONFIGURATIONS EVALUATED: to be filled in by the implementing session,
counting every Step-A cell (4 episodes x 1 threshold = 4) plus, only if
Step B is reached, every weight grid point x market x period cell.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_coinbase_eth_spot, load_ohlcv_csv  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r142_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    MIN_EPISODES_PASS_BTC,
    OOS_START,
    USABLE_EPISODES_BTC,
    USABLE_EPISODES_ETH,
    V4_BAND,
    V4_HORIZONS,
    anchor_majority,
    anchor_votes,
    block_bootstrap_lead_null,
    confirming_vote_frac,
    dual_quarter_slope,
    load_deribit_quarterly,
    nearest_transition,
    slope_zscore,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"

# ---- Step A constants, fixed a priori (see banner item 2) -----------------
Z_THRESH = 1.5
SEARCH_WINDOW_DAYS = 60
SLOPE_BASELINE_DAYS = 20   # matches r142_shared.slope_zscore's own default
N_DRAWS = 1000
# BLOCK_DAYS: the pre-registration (banner item 2, "NULL") asks for a
# block length that "comfortably exceeds the 20-day z-score window", so
# that a circular block-shift does not just resample within the same
# autocorrelated stretch the rolling z-score itself creates (a block
# shorter than ~20 days would let two shifted copies of the SAME
# 20-day-smoothed hump land inside one block, understating the null's own
# spread). 35 days = 1.75x the 20-day window: comfortably past it without
# stretching so far that a 60-day EPISODE window (SEARCH_WINDOW_DAYS)
# starts running out of distinct block positions to draw from. Chosen
# once, before any real-data lead/lag number was computed; not swept.
BLOCK_DAYS = 35
NULL_SEED = 142  # this round's own number, matching R-120's seed=120 convention


# ---------------------------------------------------------------- holdout guard

def assert_no_holdout(df: pd.DataFrame) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before OOS_START. Independent of any truncation already done
    at load time (R-79/R-81/R-84/R-120's own convention)."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# --------------------------------------------------------------------- data

def load_btc_bars() -> pd.DataFrame:
    """BTC spot (`tradebot.data.load_ohlcv_csv`, per the task's explicit
    instruction to mirror R-120's mechanical loading, not `load_dataset`'s
    perp-fallback path), truncated strictly before OOS_START at load time."""
    df = load_ohlcv_csv(DATA_DIR / "btcusd_spot_5m.csv.gz")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC spot: {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def compute_slope_z(bars: pd.DataFrame, quarterly: pd.DataFrame,
                     window_days: int = SLOPE_BASELINE_DAYS) -> pd.Series:
    """`dual_quarter_slope(...).slope` z-scored against its own trailing
    `window_days` mean/std, via `r142_shared.slope_zscore` (both causal by
    construction -- `dual_quarter_slope`'s own row-local column selection
    plus `merge_asof(direction="backward")`, and a trailing `.rolling()`)."""
    slope = dual_quarter_slope(bars, quarterly)["slope"]
    return slope_zscore(slope, window_days=window_days)


# ------------------------------------------------------------- flip / crossing

def nearest_crossing_bidirectional(z: np.ndarray, window: pd.DatetimeIndex,
                                    onset: pd.Timestamp,
                                    thresh: float = Z_THRESH) -> pd.Timestamp | None:
    """First BIDIRECTIONAL crossing (prior bar |z|<thresh, this bar
    |z|>=thresh) whose timestamp is closest to `onset` in `window`. NaN
    entries are treated as "not above threshold" (never trigger or clear a
    crossing on their own). Reused verbatim from R-120's own construction
    (`experiments/r120_conservative_basis_level.py`), applied to `slope_z`
    instead of `basis_z`."""
    above = np.abs(z) >= thresh
    above = np.where(np.isnan(z), False, above)
    cross = np.zeros(len(z), dtype=bool)
    cross[1:] = above[1:] & ~above[:-1]
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars_index: pd.DatetimeIndex, onset_str: str,
                    window_days: int = SEARCH_WINDOW_DAYS) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars_index[(bars_index >= lo) & (bars_index <= hi)]
    return onset, window


# --------------------------------------------------------------------- null

def episode_null_leads(slope_z: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, flip_time: pd.Timestamp,
                        n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                        seed: int = NULL_SEED) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode: circularly
    shift the LOCAL slope_z array (within `window`) and recompute the
    "crossing nearest the real, unshifted onset" against the fixed, real
    `flip_time`. Identical construction to R-120's own `episode_null_leads`."""
    local = slope_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_lead_null(n_bars=n_bars, block_days=block_days,
                                        n_draws=n_draws, seed=seed)

    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        cross_time = nearest_crossing_bidirectional(shifted, window, onset)
        if cross_time is None:
            continue
        leads[k] = (flip_time - cross_time).total_seconds() / 86400.0
    return leads


# --------------------------------------------------------------------- gate

def gate() -> dict:
    print("=" * 78)
    print("R-142 CONSERVATIVE: dual-quarter SLOPE confirming vote -- STEP A lead-time gate")
    print("=" * 78)

    bars = load_btc_bars()
    quarterly = load_deribit_quarterly(DATA_DIR, asset="BTC")
    assert quarterly is not None, "BTC Deribit quarterly file missing"
    # Truncate the raw contract file's own rows to strictly before OOS_START
    # too (not just `bars`), matching R-120's own disclosed convention: the
    # re-fetched file now covers the holdout, so this truncation is no
    # longer a no-op -- explicitly removing rows this file must never read.
    quarterly = quarterly.loc[quarterly.index < pd.Timestamp(OOS_START, tz=quarterly.index.tz)].copy()
    assert_no_holdout(quarterly)

    majority = anchor_majority(bars, horizons=V4_HORIZONS, band=V4_BAND)
    majority_arr = majority.to_numpy()
    slope_z = compute_slope_z(bars, quarterly, window_days=SLOPE_BASELINE_DAYS)

    n_nonnan = int(slope_z.notna().sum())
    print(f"\nBTC quarterly: {len(quarterly):,} raw rows, "
          f"{quarterly['instrument'].nunique()} contracts")
    print(f"slope_z: {n_nonnan:,}/{len(slope_z):,} bars non-NaN "
          f"({slope_z.first_valid_index()} -> {slope_z.last_valid_index()})")
    print(f"\nprimary feature: slope_z (dual_quarter_slope, {SLOPE_BASELINE_DAYS}-day "
          f"trailing baseline)  threshold: BIDIRECTIONAL |slope_z|>={Z_THRESH}  "
          f"search window=+/-{SEARCH_WINDOW_DAYS}d  null: {N_DRAWS} draws, "
          f"block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in USABLE_EPISODES_BTC:
        onset, window = episode_window(bars.index, onset_str)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars in range "
                  f"-- outside data coverage. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, flip=None, cross=None,
                                 lead=float("nan"), pass_lead=False, pass_null=False,
                                 null_p90=float("nan")))
            continue

        flip_time = nearest_transition(majority_arr, bars.index, onset,
                                        SEARCH_WINDOW_DAYS, direction="down")
        local_sz = slope_z.reindex(window).to_numpy()
        cross_time = nearest_crossing_bidirectional(local_sz, window, onset)

        if flip_time is None or cross_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no slope_z crossing'} "
                  f"found in +/-{SEARCH_WINDOW_DAYS}d window. FAIL by construction "
                  f"(lead undefined).")
            results.append(dict(label=label, onset=onset_str, flip=flip_time,
                                 cross=cross_time, lead=float("nan"),
                                 pass_lead=False, pass_null=False, null_p90=float("nan")))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(slope_z, window, onset, flip_time)
        valid_null = null_leads[~np.isnan(null_leads)]
        null_p90 = float(np.percentile(valid_null, 90)) if len(valid_null) else float("nan")
        null_median = float(np.median(valid_null)) if len(valid_null) else float("nan")
        pass_lead = lead > 0
        pass_null = pass_lead and (not np.isnan(null_p90)) and (lead > null_p90)

        local_majority = majority.reindex(window).to_numpy()
        flip_pos = int(window.get_indexer([flip_time])[0])
        prev_val = local_majority[flip_pos - 1] if flip_pos > 0 else float("nan")
        new_val = local_majority[flip_pos]
        # Sign of slope_z AT the crossing instant, for Step B's data-driven
        # (but pre-committed, banner item 3) directional rule, recorded
        # here regardless of PASS/FAIL so it is available if the gate
        # passes and never used to influence the gate verdict itself.
        cross_pos = int(window.get_indexer([cross_time])[0])
        cross_sign = float(np.sign(local_sz[cross_pos]))
        print(f"[{label}] onset={onset_str}")
        print(f"    anchor-gate nearest transition: {flip_time}  "
              f"(majority {prev_val:.3f} -> {new_val:.3f})")
        print(f"    slope_z nearest crossing (|z|>={Z_THRESH}): {cross_time}  "
              f"(sign={'+' if cross_sign > 0 else '-'})")
        print(f"    LEAD = {lead:+.2f} days  "
              f"({'slope LED' if lead > 0 else 'slope LAGGED/coincided'})")
        print(f"    null ({N_DRAWS} draws): median={null_median:+.2f}d  p90={null_p90:+.2f}d  "
              f"(valid draws: {len(valid_null)}/{N_DRAWS})")
        print(f"    PASS (lead>0): {pass_lead}   PASS (lead > null p90): {pass_null}")

        results.append(dict(label=label, onset=onset_str, flip=str(flip_time),
                             cross=str(cross_time), lead=lead, pass_lead=pass_lead,
                             pass_null=pass_null, null_p90=null_p90,
                             null_median=null_median, cross_sign=cross_sign))

    n_pass = sum(1 for r in results if r["pass_null"])
    passed = n_pass >= MIN_EPISODES_PASS_BTC

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (fixed before any number above was computed):")
    print("  an episode PASSES iff (a) lead>0 AND (b) lead exceeds its own")
    print(f"  {N_DRAWS}-draw block-bootstrap null's 90th percentile.")
    print(f"  proceed to Step B only if >= {MIN_EPISODES_PASS_BTC} of "
          f"{len(USABLE_EPISODES_BTC)} episodes PASS.")
    print("=" * 78)
    for r in results:
        lead_str = f"{r['lead']:+.2f}d" if np.isfinite(r["lead"]) else "undefined"
        print(f"  {r['label']:42s} lead={lead_str:>10s}  PASS={r['pass_null']}")
    print(f"\nEpisodes passing: {n_pass}/{len(USABLE_EPISODES_BTC)}")
    print(f"GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")

    print(f"\nconfigurations evaluated in this file's Step A: "
          f"{len(USABLE_EPISODES_BTC)} episodes x 1 threshold = {len(USABLE_EPISODES_BTC)} cells "
          f"(fixed measurement gate, no sweep)")
    print(f"max timestamp read anywhere in this session so far: "
          f"{max(bars.index.max(), quarterly.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed)


# --------------------------------------------------------------------- causality

def run_causality_probe_stepA() -> list[bool]:
    """Causal truncation probe on THIS branch's own signal construction
    (`slope_z`), per the task's explicit instruction: a suspiciously good
    result is a bug report first (R-21). Truncates the BTC spot frame
    partway through and confirms an early row's `slope_z` value is
    unchanged, using the frozen `quarterly` contract file (itself already
    causal by construction inside `dual_quarter_slope`'s own merge_asof)."""
    bars = load_btc_bars()
    quarterly = load_deribit_quarterly(DATA_DIR, asset="BTC")
    quarterly = quarterly.loc[quarterly.index < pd.Timestamp(OOS_START, tz=quarterly.index.tz)].copy()

    def build_target(frame: pd.DataFrame) -> np.ndarray:
        q = quarterly.loc[quarterly.index <= frame.index.max()]
        return compute_slope_z(frame, q, window_days=SLOPE_BASELINE_DAYS).to_numpy()

    results = []
    for check_at in (150_000, 250_000, 350_000):
        if check_at >= len(bars):
            continue
        ok = truncation_causality_probe(build_target, bars, check_at)
        print(f"[causality] slope_z check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    gate_result = gate()
    out = {"branch": "r142_conservative_slope_vote", "gate": gate_result}

    print("\n=== causal truncation probe on slope_z (this branch's own construction) ===")
    out["causality_probe"] = run_causality_probe_stepA()

    if gate_result["passed"]:
        print("\nSTEP A PASSED the pre-registered stop rule -- Step B is NOT "
              "implemented in this pass. Per docs/ROUTINE.md and this "
              "branch's own decision rule, Step B (confirming-vote "
              "strategy, sweep, inner-train/inner-validation, ETH, and any "
              "eventual holdout consultation) is built and run as a "
              "separate, explicit follow-on now that Step A's gate number "
              "is known -- it is not written blind, before this result "
              "existed, the same discipline the pre-registration itself "
              "applies to the holdout.")
    else:
        print("\nSTEP A FAILED the pre-registered stop rule. Per this file's own "
              "pre-registration (banner item 3), no confirming-vote strategy is "
              "built, ETH is not touched, and the holdout is not read. This "
              "gate result is this branch's whole product.")

    results_path = ROOT / "experiments" / "r142_conservative_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {results_path}")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmds = {"gate": gate, "main": main}
    choice = sys.argv[1] if len(sys.argv) > 1 else "main"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r142_conservative_slope_vote.py [{'|'.join(cmds)}]")
