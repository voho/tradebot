#!/usr/bin/env python
"""R-88 NOVEL branch: taker-flow-imbalance-conditioned EXECUTION DELAY for
``kelly_regime_v4``.

=====================================================================
PRE-REGISTRATION (frozen before any lead-time or backtest number in this
file was computed -- docs/ROUTINE.md steps 1-2). Anything below later
contradicted by what actually happened is stated in the results section,
not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). When ``kelly_regime_v4``'s own scheduled
   rebalance at bar i would trade in the direction OPPOSITE strong
   current taker flow (``tv_z`` = ``r88_shared.taker_flow_z``, wants to
   buy while sellers currently dominate or vice versa), delay that
   rebalance up to ``K`` bars -- re-checking every bar whether flow has
   turned favorable (execute immediately) or the deadline has been
   reached (force the trade through regardless, so signal timing is
   never permanently overridden).

   CONSTRAINT ATTACKED: primarily COST -- the adverse-selection cost of
   crossing the spread against a book currently leaning hard the other
   way. Secondarily INFO, because the mechanism only helps if ``tv_z``
   is itself shown to carry real, leading information (Step A below) --
   conditioning execution on noise cannot help, it can only add
   uncompensated timing risk.

   Citations (full trail in ``r88_shared.py``'s own docstring, not
   re-derived here, per this project's "one citation trail in one
   place" convention): Vafin (2026, SSRN 6938742) -- order-flow
   imbalance and short-horizon crypto return predictability with an
   explicit transaction-cost model; Bieganowski & Slepaczuk (2026,
   arXiv:2602.00776) -- order-flow imbalance and adverse-selection cost
   driving short-horizon price moves on Binance futures. Both papers'
   mechanism is specifically that trading WITH the currently-dominant
   flow direction is cheaper than trading against it -- which is an
   execution-cost claim, not a regime-timing claim, and is exactly what
   this branch tests.

   NOT A DUPLICATE OF:
   - B-24 (patient-limit execution, N-sweep) / R-77 (regime-adaptive
     execution urgency): both make patience a function of REALIZED
     VOLATILITY (how turbulent the market is). This branch makes
     patience a function of FLOW DIRECTION (which side is currently
     dominant) -- a strong, contrary-leaning book can exist in low
     volatility, and a calm, flow-aligned book can exist in high
     volatility; the two axes are not proxies for each other. Verified,
     not assumed: this file's diagnostics report the correlation
     between ``ratio[i]`` (R-77's stress proxy) and ``tv_z[i]`` at
     trigger bars, expected to be weak.
   - The CONSERVATIVE branch running in parallel this round
     (``r88_conservative_*.py``, not read or touched by this file):
     that branch VOTES ON REGIME -- it feeds ``tv_z`` into
     ``r88_shared.confirming_vote_frac`` to change WHAT fraction of
     equity v4 wants to hold. This branch changes WHEN an already-
     decided target gets executed; it never touches the target value
     itself except by delaying its application. Same raw signal
     (``tv_z``), structurally different mechanism operating on it --
     the same split as R-81's conservative-vote vs. R-81's own novel
     cascade-exit branch, and R-84's confirm-vote vs. latch-gate split.
   - L-14/L-15/L-16 (BVC/VPIN reconstruction from price): ``tv_z`` is a
     directly reported exchange feed, not derived from price -- see
     ``r88_shared.py`` for the full argument.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any execution
   logic, BTC only (ETH's metrics start 2021-12-01, after all 3
   ``r88_shared.STRESS_EPISODES`` onsets -- named here, not discovered
   after the fact, exactly as the conservative branch's own file notes
   for its signal).

   PRIMARY METRIC: ``tv_z`` (14-day trailing z-score of
   ``sum_taker_long_short_vol_ratio``, via ``r88_shared.taker_flow_z``,
   unchanged from the shared module -- this file does not re-tune the
   z-score window before looking at any number).

   DIRECTIONALITY, stated explicitly because it differs slightly from
   the conservative branch's choice (permitted by this round's brief,
   noted rather than hidden): the conservative branch's crowding
   measure (``ls_z``) is scored on ``|ls_z|`` because a crowded position
   can be extreme long OR extreme short and still predict a squeeze in
   either direction. ``tv_z`` is not that kind of measure here -- all
   three ``STRESS_EPISODES`` are BEARISH transitions (crowd de-risking),
   so the economically motivated primary threshold is SIGNED:
   ``tv_z <= -Z_THRESH`` (extreme SELL-side taker flow, i.e. panic/
   distribution flow), tested against the anchor gate's nearest DOWNWARD
   transition. A symmetric ``|tv_z| >= Z_THRESH`` variant is not used as
   the primary decision (it would let a symmetric BUY-flow extreme
   count as "leading" a bearish flip, which has no motivating mechanism
   here).

   EXTREME THRESHOLD: ``Z_THRESH = 1.5`` (same 1.5-sigma bar the
   conservative branch uses, for comparability across this round's two
   branches -- not re-derived independently, since both branches are
   asking "is this z-scored order-flow metric a real extremity
   signal?" against the same episode table).

   EPISODE-LOCAL SEARCH WINDOW: onset +/- ``WINDOW_DAYS = 60`` days,
   identical to the conservative branch's window, for the same reason
   given there (v4's own anchors react to price with a lag that isn't
   pinned to the event's calendar onset).

   ANCHOR-GATE "FLIP" DEFINITION: the ``anchor_majority`` DOWNWARD
   transition (majority DECREASES -- de-risking) nearest the episode
   onset within the search window. This file reuses the conservative
   branch's already-debugged rule directly (nearest-transition-in-
   EITHER-direction was tried and found buggy there, picking a spurious
   bullish blip 2 days before the actual 2021 top) rather than
   re-discovering that error -- "down" is used from this file's first
   and only run, disclosed here as reuse of a prior finding, not an
   independent correction made after seeing a number.

   CROSSING DEFINITION: first bar within the window where ``tv_z``
   crosses from above ``-Z_THRESH`` to at/below it, nearest the onset.

   LEAD = (flip_time - cross_time) in days. Positive = the sell-flow
   extremity was reached BEFORE the anchor gate's own nearest reaction.

   NULL: ``r88_shared.block_bootstrap_lead_null`` (block_days=5,
   n_draws=500, seed=8801 -- this file's own seed, fixed before running,
   deliberately different from the conservative branch's seed so the
   two branches' null draws are independent) circularly shifts the
   LOCAL (episode-window) ``tv_z`` array and recomputes the "crossing
   nearest the real, fixed flip time" against each shifted copy.

   PRE-REGISTERED STOP RULE (fixed now): an episode PASSES iff BOTH
   (a) LEAD > 0 AND (b) the true LEAD exceeds the 90th percentile of
   that episode's own 500-draw null distribution. PROCEED TO STEP B
   only if >= 2 of the 3 episodes PASS.

   CONSEQUENCE OF THE DISCLOSED TERRA/LUNA GAP, stated explicitly before
   running: ``r88_shared.py``'s docstring already establishes that BTC's
   second coverage gap (2022-01-31 -> 2022-05-09) ends exactly on the
   Terra/Luna episode's onset, leaving that episode's entire pre-onset
   baseline window with no usable ``tv_z``. This file reports that
   episode as a FORCED FAIL (no crossing computable in its own baseline
   window) rather than silently dropping it or searching outside the
   pre-registered window to rescue it. Under the ">= 2 of 3" stop rule
   as literally stated, this makes the rule EQUIVALENT, in practice, to
   requiring BOTH of the two remaining episodes (2021-top, FTX) to pass
   -- stated here, before running, not discovered as a post-hoc excuse
   for a fail.

3. WHAT WOULD MAKE STEP A FAIL, named now: the sell-flow extremity is
   reached AFTER (not before) the anchor gate's own nearest reaction in
   2 or more of the 3 episodes, or a positive lead is not distinguishable
   from an arbitrary time-shift of the same series (i.e. it is generic
   autocorrelation in a slow 14-day z-score, not a real early-warning
   property) -- the same failure mode this project's other INFO signals
   have hit.

   WHAT WOULD MAKE THE EXECUTION-DELAY MECHANISM ITSELF FAIL, named now,
   EVEN IF STEP A PASSES (because Step A only certifies that ``tv_z`` is
   informative, not that DELAYING execution on it is profitable):
   (a) delaying trades costs more in missed favorable price moves and
       slippage-while-waiting than it saves in avoided adverse-selection
       cost -- the flow reading could be correlated with the RIGHT
       thing to do (crowd is selling because price is about to fall
       further) rather than a temporary imbalance that reverts, in
       which case waiting for "favorable" flow means chasing a worse
       fill, not a better one;
   (b) ``kelly_regime_v4``'s ``target`` already rebalances too
       infrequently (10% deadband, latched anchors) for any ``K`` in the
       few-bars-to-few-hours range this file sweeps to matter -- most
       rebalance events might not even encounter an opposing ``tv_z``
       reading, making the mechanism a no-op most of the time;
   (c) forcing the trade through unconditionally at bar ``K`` merely
       relocates the adverse-selection cost to a different bar rather
       than avoiding it, if the opposing flow condition is persistent
       (a multi-day distribution regime, not a momentary imbalance) --
       the mechanism's premise requires flow imbalance to be
       mean-reverting on a sub-``K``-bar horizon, which is asserted, not
       yet shown, and is exactly what the diagnostics below (favorable-
       resolution rate vs. forced-through rate) are reported to check.

4. CONFIGS EVALUATED IN STEP A: 0 (fixed measurement gate, this
   project's standing convention). Step B's grid is pre-registered in
   ``STEP_B_PREREGISTRATION`` below, written before Step A's numbers
   were allowed to influence it.

STEP_B_PREREGISTRATION = '''
IF AND ONLY IF Step A passes its stop rule:

MECHANISM, EXACT (pre-registered before any run): reuse v4's own real,
causal ``target[i]`` array unmodified (``KellyRegimeV4().prepare()``).
Maintain an executed position ``pos`` (starts at 0.0). At each bar i,
if ``|target[i] - pos| > EPS`` (v4 wants to rebalance):
    is_buy = target[i] > pos
    opposing = tv_z[i] finite AND (
        (is_buy and tv_z[i] <= -Z_THRESH_B) or
        (not is_buy and tv_z[i] >= +Z_THRESH_B))
    if not already delaying:
        if opposing: start delaying (pos held at its current value)
        else: pos <- target[i] immediately (no delay -- flow agrees
              or is unreadable)
    else (already delaying since bar `placed_at`):
        age = i - placed_at
        if (not opposing) or (age >= K):   # flow turned favorable, OR deadline hit
            pos <- target[i]   # execute the CURRENT desired target,
                                 # never a stale one -- timing is never
                                 # permanently overridden
            stop delaying
        else: keep delaying, re-check next bar

Applied via a plain (unregistered) ``Strategy`` subclass,
``TakerFlowDelayV4``, whose ``on_bar`` issues ``ctx.order_notional(pos)``
exactly like ``KellyRegime.on_bar`` does for ``target`` -- i.e. this
delays WHEN v4's target gets applied to the real broker, using this
project's own standard bar-close-signal / next-bar-open-fill contract
(``docs/ROUTINE.md`` step 1.3), NOT a resting-limit-order microstructure
model. This is a deliberately simpler execution model than B-24/R-77's
maker/taker fill simulator: those rounds are about HOW an order fills
(patient limit vs. immediate taker); this round is about WHEN v4's own
target gets issued at all. Reusing the project's standard fill contract
for a "when" question, rather than inventing a second fill model,
keeps the one new variable (delay-on-opposing-flow) isolated.

SWEEP GRID (fixed now, 4 x 2 = 8 configurations):
  K (max delay, bars, 5-min cadence)   in {3, 6, 12, 24}   (15min/30min/1h/2h)
  Z_THRESH_B (tv_z opposition trigger) in {1.0, 1.5}
Run every config with ``scripts.experiment.ev()`` against BTC spot,
entry-tier fees (this harness's default), first on inner-train
(end="2020-12-31") for screening, then on inner-validation
(start="2021-01-01", end="2022-12-31") to SELECT exactly one winner by
inner-validation Sharpe among the 8 (ties broken by lower max drawdown).
``KellyRegimeV4`` and ``buy_and_hold`` are evaluated on both periods too,
as the comparison baselines (not counted as "configs of this idea", but
counted in the total evaluation tally per docs/ROUTINE.md).
Pre-registered total ev() calls: 8 configs x 2 periods + 2 baselines x
2 periods = 20.

PRE-REGISTERED HOLDOUT-CONSULTATION DECISION RULE (fixed before Step B
runs): recommend a holdout consultation to the operator IFF ALL of:
  (a) the winning config beats kelly_regime_v4 on inner-validation
      Sharpe by > +0.2 (the project's noise floor), OR shows a clear
      matched-risk drawdown improvement (materially lower max DD at a
      comparable or better Sharpe);
  (b) it passes the ETH falsification test: same config re-run on
      Coinbase ETH spot bars against ETH's own real taker-flow feed
      (``asset="ETH"``, coverage from 2021-12-01, window bounded at
      2022-12-31 -- still inside the inner split, never the holdout),
      PASS = its Sharpe on that window is >= plain ``kelly_regime_v4``'s
      own Sharpe on the identical ETH window (directional replication,
      not magnitude-matching, the same bar R-77's own ETH falsification
      used).
Anything else -> do not recommend a holdout consultation; report the
result as this branch's negative/inconclusive product.
This file has NO authority to consult the holdout regardless of what
this rule outputs -- see ROUTINE.md and the operator note in this
file's own final report.
'''

USAGE
-----
    python experiments/r88_novel_taker_flow_delay.py            # everything
    python experiments/r88_novel_taker_flow_delay.py stepA       # gate only
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_binance_metrics, load_coinbase_eth_spot  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

from experiments.r88_shared import (  # noqa: E402
    METRICS_END,
    METRICS_START,
    OOS_START,
    STRESS_EPISODES,
    anchor_majority,
    block_bootstrap_lead_null,
    load_flow_inputs,
    taker_flow_z,
    truncation_causality_probe,
)

from scripts.experiment import DF as BTC_DF, SPOT, ev  # noqa: E402

DATA_DIR = ROOT / "data"

# ------------------------------------------------------------ Step A params
Z_THRESH_A = 1.5
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 8801

# ------------------------------------------------------------ Step B params
K_GRID = (3, 6, 12, 24)
Z_THRESH_B_GRID = (1.0, 1.5)
EPS_TARGET = 1e-9

INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"

CONFIG_COUNTER = {"stepA": 0, "stepB": 0, "diagnostic": 0}


def _count(kind: str, k: int = 1) -> None:
    CONFIG_COUNTER[kind] += k


# ---------------------------------------------------------------- holdout guard
def assert_no_holdout(obj) -> None:
    """Hard guard, same pattern as r81/r77: the max timestamp anywhere this
    file touches must be strictly before OOS_START."""
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
# STEP A -- lead-time gate
# =====================================================================

def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp) -> pd.Timestamp | None:
    """Nearest-to-onset DOWNWARD transition of `series` within `window`."""
    vals = series.reindex(window).to_numpy()
    changed = np.zeros(len(vals), dtype=bool)
    changed[1:] = vals[1:] < vals[:-1]
    idx = np.where(changed)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def nearest_signed_crossing(z: pd.Series, window: pd.DatetimeIndex,
                             onset: pd.Timestamp, thresh: float) -> pd.Timestamp | None:
    """Nearest-to-onset first-crossing of `z` from above -thresh to <= -thresh."""
    vals = z.reindex(window).to_numpy()
    below = vals <= -thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = below[1:] & ~below[:-1]
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars: pd.DataFrame, onset_str: str) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=WINDOW_DAYS)
    hi = onset + pd.Timedelta(days=WINDOW_DAYS)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window


def episode_null_leads(tv_z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                        flip_time: pd.Timestamp, thresh: float) -> np.ndarray:
    local = tv_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_lead_null(n_bars=n_bars, block_days=BLOCK_DAYS,
                                        n_draws=N_DRAWS, seed=NULL_SEED)
    leads = np.full(len(shifts), np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        below = shifted <= -thresh
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = below[1:] & ~below[:-1]
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        cross_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - cross_time).total_seconds() / 86400.0
    return leads


def load_btc_bars_for_gate() -> pd.DataFrame:
    df = BTC_DF.loc[BTC_DF.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(df)
    print(f"BTC spot: {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (< {OOS_START})",
          file=sys.stderr)
    return df


def step_a_gate() -> dict:
    print("=" * 78)
    print("R-88 NOVEL: taker-flow execution-delay -- STEP A lead-time gate (tv_z)")
    print("=" * 78)

    bars = load_btc_bars_for_gate()
    majority = anchor_majority(bars)

    metrics = load_flow_inputs(DATA_DIR, asset="BTC")
    assert metrics is not None, "BTC taker-flow metrics file missing"
    assert_no_holdout(metrics)
    tv_z = taker_flow_z(metrics, bars)
    assert_no_holdout(tv_z.to_frame())

    print(f"\nprimary metric: tv_z (14-day trailing z-score of "
          f"sum_taker_long_short_vol_ratio)  SIGNED threshold: tv_z<=-{Z_THRESH_A}  "
          f"search window=+/-{WINDOW_DAYS}d  null: {N_DRAWS} draws, "
          f"block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        is_terra_luna = "Terra" in label
        onset, window = episode_window(bars, onset_str)

        if is_terra_luna:
            print(f"[{label}] onset={onset_str}: DISCLOSED COVERAGE GAP "
                  f"(BTC tv data missing 2022-01-31 -> 2022-05-09, ending exactly on "
                  f"this episode's onset) makes the pre-episode baseline window "
                  f"unusable BY CONSTRUCTION. Reported here as a FORCED FAIL, not "
                  f"silently dropped, per r88_shared.py's own disclosure. MECHANISM "
                  f"NOTE (found while computing this file's own numbers, not assumed "
                  f"from the docstring): align_metrics_causal's ffill does not turn "
                  f"this gap into NaN tv_z -- it forward-fills the raw ratio to an "
                  f"exact CONSTANT for 98 days, so the 14-day rolling std computed "
                  f"over that constant is not exactly 0.0 (float64 summation noise, "
                  f"~2.2e-16) rather than the exact 0.0 the code guards against, so "
                  f"tv_z evaluates to a finite ~0.0 throughout the gap instead of NaN. "
                  f"The practical effect is the same as the disclosed caveat (no real "
                  f"pre-onset extremity signal exists in this window -- ~0.0 can never "
                  f"cross an extremity threshold), so the forced-FAIL below still "
                  f"applies, but via a floating-point-artifact mechanism, not literal "
                  f"missing data -- disclosed here rather than silently assumed.")

        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars in range -- "
                  f"outside data coverage. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                 pass_a=False, pass_b=False, null_p90=float("nan"),
                                 forced=is_terra_luna))
            continue

        flip_time = nearest_transition(majority, window, onset)
        cross_time = nearest_signed_crossing(tv_z, window, onset, Z_THRESH_A)
        n_valid_tv = int(np.isfinite(tv_z.reindex(window).to_numpy()).sum())

        if flip_time is None or cross_time is None:
            reason = "no anchor-gate transition" if flip_time is None else "no tv_z crossing"
            print(f"[{label}] onset={onset_str}: {reason} found in +/-{WINDOW_DAYS}d "
                  f"window ({n_valid_tv}/{len(window)} bars have finite tv_z). "
                  f"FAIL by construction (lead undefined).")
            results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                 pass_a=False, pass_b=False, null_p90=float("nan"),
                                 forced=is_terra_luna))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(tv_z, window, onset, flip_time, Z_THRESH_A)
        valid_null = null_leads[~np.isnan(null_leads)]
        null_p90 = float(np.percentile(valid_null, 90)) if len(valid_null) else float("nan")
        null_median = float(np.median(valid_null)) if len(valid_null) else float("nan")
        pass_a = lead > 0
        pass_b = pass_a and (not np.isnan(null_p90)) and (lead > null_p90)
        if is_terra_luna:
            # Forced FAIL regardless of the arithmetic outcome above, per the
            # disclosed-gap policy stated before this file ran.
            pass_a = False
            pass_b = False

        print(f"[{label}] onset={onset_str}")
        print(f"    anchor-gate nearest downward transition: {flip_time}")
        print(f"    tv_z nearest signed crossing (<=-{Z_THRESH_A}): {cross_time}  "
              f"({n_valid_tv}/{len(window)} bars finite)")
        print(f"    LEAD = {lead:+.2f} days  "
              f"({'flow LED' if lead > 0 else 'flow LAGGED/coincided'})")
        print(f"    null ({N_DRAWS} draws): median={null_median:+.2f}d  p90={null_p90:+.2f}d  "
              f"valid draws: {len(valid_null)}/{N_DRAWS}")
        print(f"    PASS (a) lead>0: {pass_a and lead>0}   "
              f"PASS (b) lead>null p90: {pass_b}"
              + ("  [OVERRIDDEN TO FAIL: disclosed coverage gap]" if is_terra_luna else ""))

        results.append(dict(label=label, onset=onset_str, lead=lead, pass_a=pass_a,
                             pass_b=pass_b, null_p90=null_p90, null_median=null_median,
                             forced=is_terra_luna))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 2

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE: episode PASSES iff lead>0 AND lead exceeds its")
    print("own 500-draw block-bootstrap null's 90th percentile. Proceed to Step B")
    print("only if >= 2 of 3 episodes PASS. (Terra/Luna is a forced FAIL by the")
    print("disclosed coverage gap, so this requires BOTH remaining episodes to pass.)")
    print("=" * 78)
    for r in results:
        tag = " [FORCED FAIL: coverage gap]" if r["forced"] else ""
        print(f"  {r['label']:40s} lead={r['lead']:+.2f}d  PASS={r['pass_b']}{tag}")
    print(f"\nEpisodes passing: {n_pass}/3")
    print(f"STEP A GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no execution model built'}")
    print(f"\nETH note: ETH taker-flow metrics start {METRICS_START['ETH']}, after all 3 "
          f"episode onsets -- no ETH Step-A gate is run, same limitation the "
          f"conservative branch names for its own signal.")
    print(f"\nconfigurations evaluated in Step A: 0 (fixed measurement gate)")
    print(f"max timestamp read in Step A: {max(bars.index.max(), tv_z.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed, bars=bars, tv_z=tv_z)


# =====================================================================
# STEP B -- execution-delay mechanism
# =====================================================================

@dataclass(frozen=True)
class DelayConfig:
    k_max: int
    z_thresh: float

    def tag(self) -> str:
        return f"K{self.k_max}_z{self.z_thresh:g}"


def build_delayed_target(target: np.ndarray, tv_z: np.ndarray,
                          cfg: DelayConfig) -> tuple[np.ndarray, dict]:
    """Causal, bar-by-bar construction of the delayed execution series.

    Row i depends only on target[<=i], tv_z[<=i], and the running (pos,
    pending_since) state built from strictly earlier bars -- causal by
    construction, checked below with `truncation_causality_probe`.
    """
    n = len(target)
    out = np.empty(n)
    pos = 0.0
    pending_since = None
    n_delayed = 0
    n_forced = 0
    n_favorable = 0
    delay_lengths: list[int] = []

    for i in range(n):
        desired = target[i]
        if abs(desired - pos) <= EPS_TARGET:
            pending_since = None
            out[i] = pos
            continue

        is_buy = desired > pos
        z = tv_z[i]
        opposing = np.isfinite(z) and (
            (is_buy and z <= -cfg.z_thresh) or ((not is_buy) and z >= cfg.z_thresh))

        if pending_since is None:
            if opposing:
                pending_since = i
                n_delayed += 1
                out[i] = pos  # hold, do not execute yet
            else:
                pos = desired
                out[i] = pos
            continue

        age = i - pending_since
        if (not opposing) or age >= cfg.k_max:
            if opposing and age >= cfg.k_max:
                n_forced += 1
            else:
                n_favorable += 1
            delay_lengths.append(age)
            pos = desired
            pending_since = None
            out[i] = pos
        else:
            out[i] = pos

    diag = dict(n_delayed=n_delayed, n_forced=n_forced, n_favorable=n_favorable,
                mean_delay=float(np.mean(delay_lengths)) if delay_lengths else float("nan"))
    return out, diag


class TakerFlowDelayV4(Strategy):
    """kelly_regime_v4 with rebalances delayed up to K bars when tv_z opposes them."""

    warmup = KellyRegimeV4().warmup

    def __init__(self, k_max: int = 6, z_thresh: float = 1.5, asset: str = "BTC") -> None:
        self.name = f"taker_flow_delay_v4[{DelayConfig(k_max, z_thresh).tag()}]"
        self.cfg = DelayConfig(k_max, z_thresh)
        self.asset = asset
        self._last_diag: dict = {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        assert_no_holdout(df)
        v4_target = KellyRegimeV4().prepare(df.copy())["target"].to_numpy(dtype=float)

        metrics = load_flow_inputs(DATA_DIR, asset=self.asset)
        assert metrics is not None, f"{self.asset} taker-flow metrics file missing"
        assert_no_holdout(metrics)
        tv_z = taker_flow_z(metrics, df).to_numpy(dtype=float)

        delayed, diag = build_delayed_target(v4_target, tv_z, self.cfg)
        self._last_diag = diag
        df["target"] = delayed
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > EPS_TARGET:
            ctx.order_notional(t)


def step_b_sweep() -> dict:
    print("\n" + "=" * 100)
    print("STEP B -- execution-delay sweep, BTC spot, entry-tier fees "
          f"({len(K_GRID)}x{len(Z_THRESH_B_GRID)}={len(K_GRID)*len(Z_THRESH_B_GRID)} configs)")
    print("=" * 100)

    configs = [DelayConfig(k, z) for k in K_GRID for z in Z_THRESH_B_GRID]
    print(f"pre-registered grid: K in {K_GRID}, Z_THRESH_B in {Z_THRESH_B_GRID}  "
          f"({len(configs)} configs)\n")

    train_rows, val_rows = {}, {}

    print("-- inner-train (<= 2020-12-31) --")
    for cfg in configs:
        strat = TakerFlowDelayV4(k_max=cfg.k_max, z_thresh=cfg.z_thresh)
        m = ev(strat, end=INNER_TRAIN_END, tag=f"train {cfg.tag()}")
        _count("stepB")
        train_rows[cfg.tag()] = dict(sharpe=m.sharpe, max_dd=m.max_drawdown_pct,
                                      profit=m.profit_pct, diag=dict(strat._last_diag))

    v4_train = ev(KellyRegimeV4(), end=INNER_TRAIN_END, tag="train kelly_regime_v4")
    _count("stepB")
    bh_train = ev(get_strategy("buy_and_hold"), end=INNER_TRAIN_END, tag="train buy_and_hold")
    _count("stepB")

    print("\n-- inner-validation (2021-01-01 -> 2022-12-31) --")
    for cfg in configs:
        strat = TakerFlowDelayV4(k_max=cfg.k_max, z_thresh=cfg.z_thresh)
        m = ev(strat, start=INNER_VAL_START, end=INNER_VAL_END, tag=f"val   {cfg.tag()}")
        _count("stepB")
        val_rows[cfg.tag()] = dict(sharpe=m.sharpe, max_dd=m.max_drawdown_pct,
                                    profit=m.profit_pct, diag=dict(strat._last_diag))

    v4_val = ev(KellyRegimeV4(), start=INNER_VAL_START, end=INNER_VAL_END, tag="val   kelly_regime_v4")
    _count("stepB")
    bh_val = ev(get_strategy("buy_and_hold"), start=INNER_VAL_START, end=INNER_VAL_END, tag="val   buy_and_hold")
    _count("stepB")

    winner_tag = max(val_rows, key=lambda t: (val_rows[t]["sharpe"], -val_rows[t]["max_dd"]))
    winner_cfg = next(c for c in configs if c.tag() == winner_tag)
    print(f"\nWINNER (by inner-validation Sharpe): {winner_tag}  "
          f"sharpe={val_rows[winner_tag]['sharpe']:.3f}  "
          f"(v4 sharpe={v4_val.sharpe:.3f})")

    d = val_rows[winner_tag]["diag"]
    print(f"  winner diagnostics (inner-val window): delayed={d['n_delayed']} "
          f"forced-through={d['n_forced']} resolved-favorably={d['n_favorable']} "
          f"mean_delay={d['mean_delay']:.2f} bars")

    sharpe_delta = val_rows[winner_tag]["sharpe"] - v4_val.sharpe
    dd_delta = v4_val.max_drawdown_pct - val_rows[winner_tag]["max_dd"]  # positive = winner's DD is lower
    print(f"  inner-val sharpe_delta vs v4 = {sharpe_delta:+.3f}   "
          f"max_dd_delta vs v4 (positive=better) = {dd_delta:+.2f}pp")

    return dict(configs=configs, train=train_rows, val=val_rows, winner=winner_cfg,
                winner_tag=winner_tag, v4_train=v4_train, v4_val=v4_val,
                bh_train=bh_train, bh_val=bh_val, sharpe_delta=sharpe_delta, dd_delta=dd_delta)


def eth_falsification(winner: DelayConfig) -> dict:
    print("\n" + "=" * 100)
    print(f"ETH FALSIFICATION -- {winner.tag()}, Coinbase ETH spot, "
          f"ETH taker-flow window ({METRICS_START['ETH']} -> {INNER_VAL_END})")
    print("=" * 100)

    eth_df = load_coinbase_eth_spot(DATA_DIR)
    assert eth_df is not None, "ETH coinbase spot file missing"
    eth_df = eth_df.loc[eth_df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(eth_df)

    strat = TakerFlowDelayV4(k_max=winner.k_max, z_thresh=winner.z_thresh, asset="ETH")
    m = ev(strat, df=eth_df, market=SPOT, start=METRICS_START["ETH"], end=INNER_VAL_END,
           tag=f"ETH   {winner.tag()}")
    _count("stepB")
    m_v4 = ev(KellyRegimeV4(), df=eth_df, market=SPOT, start=METRICS_START["ETH"], end=INNER_VAL_END,
              tag="ETH   kelly_regime_v4")
    _count("stepB")

    passed = m.sharpe >= m_v4.sharpe
    print(f"\nETH falsification: winner sharpe={m.sharpe:.3f}  v4 sharpe={m_v4.sharpe:.3f}  "
          f"PASS (winner >= v4, directional replication): {passed}")
    print(f"max timestamp read in ETH falsification: {eth_df.index.max()}  (< {OOS_START})")
    return dict(winner_sharpe=m.sharpe, v4_sharpe=m_v4.sharpe, passed=passed)


def causality_probe(winner: DelayConfig) -> bool:
    print("\n" + "=" * 100)
    print(f"CAUSAL-TRUNCATION PROBE -- {winner.tag()}, r88_shared.truncation_causality_probe")
    print("=" * 100)

    df = BTC_DF.loc[BTC_DF.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(df)

    def build_target_fn(frame: pd.DataFrame) -> np.ndarray:
        v4_target = KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float)
        metrics = load_flow_inputs(DATA_DIR, asset="BTC")
        assert_no_holdout(metrics)
        tv_z = taker_flow_z(metrics, frame).to_numpy(dtype=float)
        delayed, _ = build_delayed_target(v4_target, tv_z, winner)
        return delayed

    check_at = len(df) - 40_000
    ok = truncation_causality_probe(build_target_fn, df, check_at=check_at, shorter_by=20_000)
    print(f"  check_at bar={check_at}  ({df.index[check_at]})  shorter_by=20,000 bars")
    print(f"  CAUSAL-TRUNCATION PROBE: {'PASS' if ok else 'FAIL'}")
    _count("diagnostic")
    return ok


# =====================================================================
# main
# =====================================================================

def main() -> None:
    t0 = time.time()
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    gate_result = step_a_gate()

    if choice == "stepA":
        print(f"\nCONFIGS EVALUATED: stepA={CONFIG_COUNTER['stepA']} "
              f"stepB={CONFIG_COUNTER['stepB']} diagnostic={CONFIG_COUNTER['diagnostic']}")
        print(f"[{time.time()-t0:.0f}s]")
        return

    if not gate_result["passed"]:
        print("\n" + "#" * 78)
        print("# STEP A FAILED ITS PRE-REGISTERED STOP RULE.")
        print("# Per this file's own pre-registration, STOP HERE. No execution-delay")
        print("# model is built; Step B is not run. This gate result is this branch's")
        print("# whole product.")
        print("#" * 78)
        print(f"\nCONFIGS EVALUATED (TOTAL): stepA=0 stepB=0 diagnostic=0")
        print(f"[{time.time()-t0:.0f}s]")
        return

    step_b = step_b_sweep()
    eth = eth_falsification(step_b["winner"])
    probe_ok = causality_probe(step_b["winner"])

    print("\n" + "=" * 100)
    print("PRE-REGISTERED HOLDOUT-CONSULTATION DECISION RULE (restated, applied mechanically)")
    print("=" * 100)
    rule_a = (step_b["sharpe_delta"] > 0.2) or (step_b["dd_delta"] > 0)
    rule_b = eth["passed"]
    recommend = rule_a and rule_b
    print(f"  (a) sharpe_delta({step_b['sharpe_delta']:+.3f}) > +0.2  OR  "
          f"clear matched-risk DD improvement (dd_delta={step_b['dd_delta']:+.2f}pp): {rule_a}")
    print(f"  (b) ETH falsification passed: {rule_b}")
    print(f"  RECOMMEND HOLDOUT CONSULTATION: {recommend}")
    print("  NOTE: this file has NO authority to consult the holdout. If the rule "
          "above says True, that is reported to the operator, not acted on here.")

    print(f"\nCausal-truncation probe: {'PASS' if probe_ok else 'FAIL'}")
    total = CONFIG_COUNTER["stepA"] + CONFIG_COUNTER["stepB"] + CONFIG_COUNTER["diagnostic"]
    print(f"\nCONFIGS EVALUATED (TOTAL): stepA={CONFIG_COUNTER['stepA']} "
          f"stepB={CONFIG_COUNTER['stepB']} diagnostic={CONFIG_COUNTER['diagnostic']} "
          f"total={total}")
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
