#!/usr/bin/env python
"""R-120 NOVEL branch: the Deribit calendar-basis's own RATE OF CHANGE
(basis MOMENTUM), not its level, as a lead-time candidate against
`kelly_regime_v4`'s 3-anchor gate -- Step A measurement gate first, this
project's established discipline for every INFO-axis round since R-53.

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). Anything below later
contradicted by what actually happened is stated in the results section,
not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). A rapidly RISING or FALLING calendar basis
   -- carry conditions changing quickly, not merely being extreme --
   signals an active repricing by cash-and-carry/term traders (a FLOW,
   not a level) that `kelly_regime_v4`'s slow 20/40/80-day price anchors
   have not yet caught up to, distinct from the CONSERVATIVE sibling
   running this round (`r120_conservative_basis_level.py`, not read or
   touched by this file), whose claim is that an extreme LEVEL of the
   same basis is itself informative.

   CONSTRAINT ATTACKED: INFO -- the same new instrument (Deribit's dated
   quarterly BTC future) as the conservative sibling, a genuinely
   different STATISTIC computed on it (rate of change vs. level). A
   basis that is LOW but RISING FAST scores very differently on this
   branch's feature than on the sibling's; the two are not a
   re-parameterization of each other.

   Citations (full trail in `experiments/r120_shared.py`'s own
   docstring, not re-derived here, per this project's "one citation
   trail in one place" convention): Schmeling, Schrimpf & Todorov (2023,
   rev. 2025), "Crypto carry", BIS Working Papers No. 1087; Chi et al.
   (2023), "An empirical investigation on risk factors in cryptocurrency
   futures", Journal of Futures Markets -- BASIS and BASIS-MOMENTUM are
   two distinct, only weakly correlated persistent factors in crypto
   futures (adapting Boons & Prado's commodity basis-momentum concept,
   and Erb & Harvey 2006, Financial Analysts Journal 62(2), to crypto),
   i.e. the basis's rate of change carries information not subsumed by
   its level. Stated honestly, not overclaimed: Chi et al.'s exact
   construction needs a full simultaneous multi-maturity curve
   (basis_far - basis_near) to measure curvature/basis-momentum in their
   cross-sectional sense; this project's data has only the front-quarter
   contract at each point in time. This branch instead measures the
   front-quarter basis's OWN time-series momentum (a first difference /
   rate of change over a trailing window) -- an honest operationalization
   of "is carry accelerating", not a replication of their cross-sectional
   factor.

   NOT A DUPLICATE OF:
   - The CONSERVATIVE sibling this round (`r120_conservative_basis_level.py`):
     tests the same basis's LEVEL, z-scored; this branch tests its RATE
     OF CHANGE, z-scored. Same instrument, different statistic, disjoint
     files, neither reads the other.
   - R-41/`kelly_regime_v9_basis_lead` (spot-vs-PERPETUAL basis): a
     different instrument pair. That basis resets every 8h at funding
     settlement and is mechanically mean-reverting by construction
     (Zhang 2026, SSRN 6185958); a dated quarterly future resolves only
     at its fixed expiry -- a structurally different statistical object,
     not a re-parameterization.
   - B-05/R-35/R-39 (raw Binance funding, COST-axis flat gate): a
     per-8h payment, not a term-structure/roll-yield quantity.
   - R-73 (DVOL level/ROC): implied volatility, not a forward price.
   - R-81 (Binance OI + top-trader long/short crowding): positioning/
     leverage STOCKS, not a rate-of-change of a priced term-structure
     quantity -- though R-81's own `oi_chg_z` feature IS a rate-of-change
     construction (of open interest, a different underlying series from
     basis), cited here as the closest methodological precedent for
     "rate of change as the feature itself" in this ledger, not as a
     duplicate of what is differenced.
   - R-63/R-76 (cross-coin pairs, e.g. BTC-vs-ETH spot): pair two
     DIFFERENT COINS' spot series; this pairs the SAME coin across two
     MATURITIES (spot vs. its own dated future).
   - R-74 (MVRV rate-of-change, novel branch): the closest prior
     STRUCTURAL precedent in this ledger for "does differencing fix a
     lagging level" -- R-74's own answer was no, for a reason specific to
     MVRV (realized cap is too price-coupled at settlement timescales
     shorter than dormant-supply turnover to decouple from price). This
     branch's own named risk (below) is analogous in spirit -- a rate of
     change is not automatically a fix for a lagging level -- but the
     specific mechanism named for THIS signal is noise amplification from
     differencing, not price-coupling, because basis (unlike MVRV) is not
     numerator-dominated by spot price the same way.
   - Grepped `docs/LEDGER.md` for "basis moment", "rate of change",
     "roll yield", "term structure", "calendar future", "contango",
     "backwardation": the only prior hit is R-74's own MVRV rate-of-change
     branch (cited above) and R-81's `oi_chg_z` mention within its own
     crowding-round writeup (also cited above); no round has tried a
     basis-momentum/rate-of-change-of-basis construction.

2. NAMED FAILURE MODE (before any code ran). The same failure every one
   of this project's 17 prior INFO-axis signal attempts has hit: the
   momentum extremity arrives AFTER (not before) the anchor gate's own
   reaction, or a positive lead (if any) is indistinguishable from
   generic autocorrelation/regime-persistence in a slow z-scored series
   rather than a real early-warning property. A SPECIFIC additional risk
   for THIS construction, named now: differencing a noisy series
   amplifies its noise (a first difference of `ann_basis` inherits and
   roughly doubles the variance of two nearby, only loosely correlated
   basis readings), so `basis_momentum` may simply be too noisy to ever
   cross a stable |z|>=1.5 threshold MEANINGFULLY before random,
   regime-independent basis jitter does -- i.e. the crossing that "leads"
   an episode may be indistinguishable from a crossing that would have
   happened at almost any other time, which is exactly what the
   block-bootstrap null below is built to catch. If Step A fails, this
   file reports which of these two failure modes (lag vs. noise
   amplification) was actually observed, rather than defaulting to "it
   lagged" as a catch-all. The modal, fully expected outcome is FAILURE
   -- given 0 of 17 prior INFO-axis signals in this ledger have led, and
   this branch's own sibling shares the same base rate risk on the same
   instrument -- and a clean, well-documented negative is this branch's
   fully successful product if that is what happens. This file does not
   force a positive result.

3. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy
   code, BTC only, on `experiments.r120_shared.USABLE_EPISODES_BTC` (4
   episodes: COVID 2020-03-12, 2021-top 2021-11-10, Terra/Luna
   2022-05-09, FTX 2022-11-08 -- BTC's calendar-basis coverage starts
   2019-01-01, per `r120_shared`'s own disclosed coverage caveat, ruling
   out the two 2018 episodes for both branches this round, symmetrically).

   FEATURE: `basis_momentum` = the causal rate of change of `ann_basis`
   (`r120_shared.front_quarter_basis`'s annualized calendar basis) over a
   trailing `MOM_DAYS=5` window -- `ann_basis.diff(MOM_DAYS * BARS_PER_DAY)`,
   a plain backward first difference (chosen over an EWM-smoothed
   difference because a plain diff is the more literal "rate of change"
   and imposes no extra smoothing that could itself manufacture a
   spurious lead by construction -- documented here as the choice made,
   not swept for best appearance). Purely causal: row i depends only on
   `ann_basis[i]` and `ann_basis[i - MOM_DAYS*BARS_PER_DAY]`, both already
   causal by `front_quarter_basis`'s own construction. Then z-scored
   against ITS OWN trailing `WINDOW_DAYS=20`-day rolling mean/std (the
   same baseline window as the conservative sibling's own level z-score,
   for a fair architectural comparison -- the difference between
   branches is the feature construction, not the baseline window). Call
   the result `mom_z`.

   THRESHOLD: BIDIRECTIONAL, `|mom_z| >= 1.5` -- a rapid move in either
   direction (accelerating contango OR accelerating backwardation) is
   informative under the mechanism statement above, unlike a one-sided
   construction.

   EPISODE-LOCAL SEARCH WINDOW: onset +/- `WINDOW_DAYS_SEARCH=60` days,
   matching every prior INFO-axis round's window and the conservative
   sibling's, since `kelly_regime_v4`'s own anchors react to price with a
   lag not pinned to an event's calendar onset.

   ANCHOR-GATE "FLIP" DEFINITION: `r120_shared.nearest_transition(...,
   direction="down")` -- reused verbatim, same as the conservative
   sibling; all 4 usable episodes are bearish transitions.

   MOMENTUM "CROSSING" DEFINITION: the first BIDIRECTIONAL crossing
   (prior bar `|mom_z| < 1.5`, this bar `|mom_z| >= 1.5`) nearest the
   episode's onset within the search window.

   LEAD = (flip_time - crossing_time) in days. Positive = the momentum
   extremity was reached BEFORE the anchor gate's own nearest reaction.

   NULL: `r120_shared.block_bootstrap_lead_null(n_bars=<local episode
   window length>, block_days=5, n_draws=500, seed=1200)` -- a seed
   DISTINCT from the conservative sibling's (120), so the two branches'
   null draws are independent -- circularly block-shifts the LOCAL
   (episode-window) `mom_z` array and recomputes the bidirectional
   crossing nearest the real, fixed `flip_time` against each shifted
   copy.

   PASS RULE, per episode: LEAD > 0 AND LEAD exceeds the 90th percentile
   of that episode's own 500-draw null lead distribution.

   STOP RULE (fixed now, before any number below was computed): proceed
   to Step B only if `>= MIN_EPISODES_PASS_BTC` (3 of 4, from
   `r120_shared`) episodes PASS. If fewer pass: STOP, report NEGATIVE
   with the full 4-episode table, do not write any Step B strategy code.

4. STEP B -- CONTINGENT PRE-REGISTRATION (frozen now, before Step A's
   numbers exist; only executed, and only ADDED to this file, if Step
   A's stop rule passes).

   IDENTITY CHECK, run first: `confirming_vote_frac(anchor_sum,
   meta_vote, weight=0)` must recover `kelly_regime_v4`'s `target` array
   bit-for-bit.

   CONSTRUCTION: `meta_vote` tracks v4's fastest (20-day) anchor vote,
   updating only on a bar where `|mom_z| >= 1.5` (a "confirmed" bar);
   otherwise it latches its last confirmed value (R-53/R-54/R-55's
   hysteresis-latch pattern, keyed on the bidirectional momentum-extreme
   gate instead of a level threshold-band). Before the first confirmed
   bar, defaults to the fast anchor's own then-current value (no dilution
   while unconfirmed).

   frac = confirming_vote_frac(anchor_sum, meta_vote, weight)

   SWEEP GRID (fixed a priori): `weight` in {0.5, 1.0, 2.0, 4.0} x
   `mom_window_days` (the momentum lookback, THIS branch's own free
   parameter, swept instead of the baseline window since the baseline
   window is fixed for comparability -- see Step A) in {3, 5, 10} -- 12
   configurations. Plus a threshold-sensitivity check at the primary
   point (`weight=1.0, mom_window_days=5`) over `Z_THRESH in {1.0, 2.0}`
   -- 2 configurations (1.5 already covered by the main grid's
   weight=1.0/mom_window_days=5 cell). Plus the identity check
   (`weight=0`) -- 1 configuration. Total: 15 configurations, evaluated
   on inner-train (2019-01-01 -> 2020-12-31 -- basis coverage starts
   2019-01-01, NOT the project's full 2017 inner-train start) and
   inner-validation (2021-01-01 -> 2022-12-31), both spot and futures_5x
   markets.

   MANDATORY CHECKS: (i) exposure-artifact R^2 -- candidate `target` vs.
   a mean-notional-matched rescale of v4's own `target`, inner-validation,
   both markets; R^2 > 0.95 = fail; (ii) ETH falsification
   (`data/ethusd_deribit_quarterly_5m.csv.gz` + `load_coinbase_eth_spot`,
   `USABLE_EPISODES_ETH`, `ETH_BASIS_COVERAGE_START` from `r120_shared`
   -- same sign of edge must replicate); (iii)
   `truncation_causality_probe` on the momentum-confirmed meta-vote
   construction; (iv) the 0.40% fee tier, if this stage is reached.

   PRE-REGISTERED HOLDOUT DECISION RULE (frozen now, contingent on
   reaching Step B): read the 2023+ holdout ONLY IF ALL of (a) the
   primary configuration's inner-validation Sharpe improvement over
   `kelly_regime_v4` exceeds the +/-0.2 noise floor (R-20) on BOTH
   markets, on a genuine parameter plateau; (b) exposure-artifact R^2 <=
   0.95 both markets; (c) ETH falsification replicates the same sign;
   (d) the causality probe passes. If ANY fail: report NEGATIVE, the
   holdout is never read.

5. CONFIGS EVALUATED IN STEP A: 0 (a fixed, non-swept measurement gate,
   this project's standing convention). Step B's count, if reached, is
   15 as itemized above.

USAGE
-----
    python experiments/r120_novel_basis_momentum.py            # everything
    python experiments/r120_novel_basis_momentum.py stepA       # gate only
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

from experiments.r120_shared import (  # noqa: E402
    BARS_PER_DAY,
    BTC_BASIS_COVERAGE_START,
    MIN_EPISODES_PASS_BTC,
    OOS_START,
    USABLE_EPISODES_BTC,
    anchor_majority,
    block_bootstrap_lead_null,
    front_quarter_basis,
    load_deribit_quarterly,
    nearest_transition,
)

DATA_DIR = ROOT / "data"

# ------------------------------------------------------------ Step A params
MOM_DAYS = 5                 # trailing rate-of-change window for basis_momentum
BASELINE_WINDOW_DAYS = 20    # z-score baseline window, matches sibling's level baseline
Z_THRESH_A = 1.5             # bidirectional
WINDOW_DAYS_SEARCH = 60      # episode-local search window, +/- days
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 1200             # distinct from the conservative sibling's seed (120)

CONFIG_COUNTER = {"stepA": 0, "stepB": 0, "diagnostic": 0}


def _count(kind: str, k: int = 1) -> None:
    CONFIG_COUNTER[kind] += k


# ---------------------------------------------------------------- holdout guard
def assert_no_holdout(obj) -> None:
    """Hard guard, same pattern as every prior INFO-axis round's own file: the
    max timestamp anywhere this file touches must be strictly before OOS_START."""
    idx = obj.index if hasattr(obj, "index") else obj
    if len(idx) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz="UTC")
    max_ts = pd.Timestamp(idx.max())
    if max_ts.tzinfo is None:
        max_ts = max_ts.tz_localize("UTC")
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# =====================================================================
# feature construction
# =====================================================================

def compute_mom_z(spot: pd.DataFrame, quarterly: pd.DataFrame,
                   mom_days: int = MOM_DAYS,
                   window_days: int = BASELINE_WINDOW_DAYS) -> tuple[pd.Series, pd.Series]:
    """`basis_momentum` = causal `ann_basis.diff(mom_days*BARS_PER_DAY)`, then
    z-scored against its own trailing `window_days`-day rolling mean/std.
    Purely causal: row i depends only on ann_basis[<=i]."""
    fb = front_quarter_basis(spot, quarterly)
    ann_basis = fb["ann_basis"]
    mom_bars = int(mom_days * BARS_PER_DAY)
    mom_raw = ann_basis.diff(mom_bars)
    win_bars = int(window_days * BARS_PER_DAY)
    roll_mean = mom_raw.rolling(win_bars).mean()
    roll_std = mom_raw.rolling(win_bars).std()
    mom_z = (mom_raw - roll_mean) / roll_std
    return mom_z, ann_basis


# =====================================================================
# episode window / crossing helpers
# =====================================================================

def episode_window(bars_index: pd.DatetimeIndex, onset_str: str,
                    window_days: int = WINDOW_DAYS_SEARCH) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars_index[(bars_index >= lo) & (bars_index <= hi)]
    return onset, window


def nearest_bidirectional_crossing(mom_z: pd.Series, window: pd.DatetimeIndex,
                                    onset: pd.Timestamp,
                                    thresh: float = Z_THRESH_A) -> pd.Timestamp | None:
    """First bidirectional crossing (prior |mom_z|<thresh, this bar
    |mom_z|>=thresh) nearest onset within `window`."""
    vals = mom_z.reindex(window).to_numpy()
    extreme = np.abs(vals) >= thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = extreme[1:] & ~extreme[:-1]
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_null_leads(mom_z: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, flip_time: pd.Timestamp,
                        thresh: float = Z_THRESH_A) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode: circularly
    shift the LOCAL mom_z array (within `window`) and recompute the
    bidirectional crossing nearest the real, unshifted onset, against the
    fixed, real `flip_time`."""
    local = mom_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_lead_null(n_bars=n_bars, block_days=BLOCK_DAYS,
                                        n_draws=N_DRAWS, seed=NULL_SEED)
    leads = np.full(len(shifts), np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        extreme = np.abs(shifted) >= thresh
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = extreme[1:] & ~extreme[:-1]
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        cross_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - cross_time).total_seconds() / 86400.0
    return leads


# =====================================================================
# STEP A -- lead-time gate
# =====================================================================

def load_btc_bars_for_gate() -> pd.DataFrame:
    from scripts.experiment import DF as BTC_DF
    df = BTC_DF.loc[BTC_DF.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(df)
    print(f"BTC spot: {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (< {OOS_START})",
          file=sys.stderr)
    return df


def step_a_gate() -> dict:
    print("=" * 78)
    print("R-120 NOVEL: calendar-basis MOMENTUM -- STEP A lead-time gate (mom_z)")
    print("=" * 78)

    bars = load_btc_bars_for_gate()
    majority = anchor_majority(bars)

    quarterly = load_deribit_quarterly(DATA_DIR, asset="BTC")
    assert quarterly is not None, "BTC Deribit quarterly futures file missing"
    quarterly_pre_holdout = quarterly.loc[quarterly.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(quarterly_pre_holdout)

    mom_z, ann_basis = compute_mom_z(bars, quarterly_pre_holdout)
    assert_no_holdout(mom_z.to_frame())

    n_valid_total = int(np.isfinite(mom_z.to_numpy()).sum())
    print(f"\nfeature: basis_momentum = ann_basis.diff({MOM_DAYS}d), z-scored on its own "
          f"trailing {BASELINE_WINDOW_DAYS}d mean/std  ->  mom_z  "
          f"({n_valid_total:,}/{len(mom_z):,} bars finite, coverage from "
          f"{BTC_BASIS_COVERAGE_START})")
    print(f"threshold: BIDIRECTIONAL |mom_z|>={Z_THRESH_A}  search window=+/-{WINDOW_DAYS_SEARCH}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in USABLE_EPISODES_BTC:
        onset, window = episode_window(bars.index, onset_str)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars in range -- "
                  f"outside data coverage. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                 pass_a=False, pass_b=False, null_p90=float("nan")))
            continue

        flip_time = nearest_transition(majority.to_numpy(), bars.index, onset,
                                        window_days=WINDOW_DAYS_SEARCH, direction="down")
        cross_time = nearest_bidirectional_crossing(mom_z, window, onset, Z_THRESH_A)
        n_valid_local = int(np.isfinite(mom_z.reindex(window).to_numpy()).sum())

        if flip_time is None or cross_time is None:
            reason = "no anchor-gate transition" if flip_time is None else "no mom_z crossing"
            print(f"[{label}] onset={onset_str}: {reason} found in +/-{WINDOW_DAYS_SEARCH}d "
                  f"window ({n_valid_local}/{len(window)} bars have finite mom_z). "
                  f"FAIL by construction (lead undefined).")
            results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                 pass_a=False, pass_b=False, null_p90=float("nan")))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(mom_z, window, onset, flip_time, Z_THRESH_A)
        valid_null = null_leads[~np.isnan(null_leads)]
        null_p90 = float(np.percentile(valid_null, 90)) if len(valid_null) else float("nan")
        null_median = float(np.median(valid_null)) if len(valid_null) else float("nan")
        pass_a = lead > 0
        pass_b = pass_a and (not np.isnan(null_p90)) and (lead > null_p90)

        print(f"[{label}] onset={onset_str}")
        print(f"    anchor-gate nearest downward transition: {flip_time}")
        print(f"    mom_z nearest bidirectional crossing (|mom_z|>={Z_THRESH_A}): {cross_time}  "
              f"({n_valid_local}/{len(window)} bars finite)")
        print(f"    LEAD = {lead:+.2f} days  "
              f"({'momentum LED' if lead > 0 else 'momentum LAGGED/coincided'})")
        print(f"    null ({N_DRAWS} draws): median={null_median:+.2f}d  p90={null_p90:+.2f}d  "
              f"valid draws: {len(valid_null)}/{N_DRAWS}")
        print(f"    PASS (a) lead>0: {pass_a}   PASS (b) lead>null p90: {pass_b}")

        results.append(dict(label=label, onset=onset_str, lead=lead, pass_a=pass_a,
                             pass_b=pass_b, null_p90=null_p90, null_median=null_median,
                             n_valid_local=n_valid_local, window_len=len(window)))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= MIN_EPISODES_PASS_BTC

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE: episode PASSES iff LEAD>0 AND LEAD exceeds its")
    print(f"own {N_DRAWS}-draw block-bootstrap null's 90th percentile. Proceed to Step B")
    print(f"only if >= {MIN_EPISODES_PASS_BTC} of {len(USABLE_EPISODES_BTC)} episodes PASS.")
    print("=" * 78)
    for r in results:
        lead_str = f"{r['lead']:+.2f}d" if np.isfinite(r["lead"]) else "undefined"
        print(f"  {r['label']:42s} lead={lead_str:>10s}  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/{len(USABLE_EPISODES_BTC)}")
    print(f"STEP A GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy code written'}")
    print(f"\nconfigurations evaluated in Step A: 0 (fixed measurement gate)")
    print(f"max timestamp read in Step A: "
          f"{max(bars.index.max(), mom_z.dropna().index.max() if mom_z.notna().any() else bars.index.max())}"
          f"  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed, bars=bars, mom_z=mom_z,
                ann_basis=ann_basis, majority=majority)


# =====================================================================
# main
# =====================================================================

def main() -> None:
    t0 = time.time()
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    gate_result = step_a_gate()

    out = dict(
        branch="r120_novel_basis_momentum",
        step_a_results=[
            {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
             for k, v in r.items()}
            for r in gate_result["results"]
        ],
        n_pass=gate_result["n_pass"],
        min_required=MIN_EPISODES_PASS_BTC,
        n_episodes=len(USABLE_EPISODES_BTC),
        step_a_passed=gate_result["passed"],
        configs_evaluated={"stepA": 0, "stepB": 0, "diagnostic": 0},
        params=dict(mom_days=MOM_DAYS, baseline_window_days=BASELINE_WINDOW_DAYS,
                    z_thresh_a=Z_THRESH_A, window_days_search=WINDOW_DAYS_SEARCH,
                    n_draws=N_DRAWS, block_days=BLOCK_DAYS, null_seed=NULL_SEED),
    )

    if choice == "stepA":
        with open(ROOT / "experiments" / "r120_novel_results.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nCONFIGS EVALUATED: stepA={CONFIG_COUNTER['stepA']} "
              f"stepB={CONFIG_COUNTER['stepB']} diagnostic={CONFIG_COUNTER['diagnostic']}")
        print(f"[{time.time()-t0:.0f}s]")
        return

    if not gate_result["passed"]:
        print("\n" + "#" * 78)
        print("# STEP A FAILED ITS PRE-REGISTERED STOP RULE.")
        print("# Per this file's own pre-registration, STOP HERE. No Step-B strategy")
        print("# code is written or run. This gate result is this branch's whole product.")
        print("#" * 78)
        with open(ROOT / "experiments" / "r120_novel_results.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nCONFIGS EVALUATED (TOTAL): stepA=0 stepB=0 diagnostic=0")
        print(f"[{time.time()-t0:.0f}s]")
        return

    print("\n# STEP A PASSED -- Step B would run here (not yet implemented in this "
          "# run; see the file's own contingent pre-registration in the module "
          "# docstring, item 4).")
    with open(ROOT / "experiments" / "r120_novel_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
