#!/usr/bin/env python
"""R-96 NOVEL branch: Hawkes cluster-intensity-conditioned EXECUTION DELAY
(BRAKE) for ``kelly_regime_v4``.

=====================================================================
PRE-REGISTRATION (frozen before any real-data Hawkes number in this file
was computed -- docs/ROUTINE.md steps 1-2). Anything below later
contradicted by what actually happened is stated in the results section
printed at run time, never edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). When ``kelly_regime_v4``'s own scheduled
   rebalance at bar i would trade WHILE a Hawkes self-excitation jump
   CLUSTER is actively in progress (``hawkes_intensity_zscore`` >=
   ``Z_THRESH``, primary grid cell n=0.5/halflife=7d from
   ``r96_shared.py``), delay that rebalance up to ``K`` bars --
   re-checking every bar whether the cluster has subsided (z has fallen
   back below ``Z_THRESH``: execute immediately) or the deadline has been
   reached (force the trade through regardless, so v4's own signal timing
   is never permanently overridden). This is direction-BLIND: unlike
   R-88's taker-flow delay (which only delays a trade that opposes the
   CURRENT flow direction), this brake delays ANY pending rebalance while
   a cluster is in progress, because the mechanism's premise is about
   EXECUTION QUALITY during active clustering (wider effective spreads,
   more whipsaw / adverse short-term path) rather than about which side
   of the book is currently dominant -- see ``r96_shared.py``'s own
   docstring, which frames this branch's driver as "clustering intensity"
   (a property with no notion of buy/sell direction at all), not
   order-flow imbalance.

   CONSTRAINT ATTACKED: primarily COST (executing into an active jump
   cluster likely means crossing a wider effective spread / worse
   short-horizon path than executing once the cluster has cooled), same
   constraint R-77/B-24 and R-88 both attacked via execution timing.
   Secondarily INFO, because the mechanism only helps if Hawkes cluster
   intensity is itself shown to predict elevated near-term realized
   volatility (the STEP-0 GATE below) -- conditioning execution on noise
   cannot help, it can only add uncompensated timing risk.

   Citation trail (full detail in ``r96_shared.py``'s own docstring, not
   re-derived here, per this project's "one citation trail in one place"
   convention): Hawkes (1971, Biometrika); Bacry, Mastromatteo & Muzy
   (2015, Market Microstructure and Liquidity) -- self-excitation as the
   causal generator of volatility clustering, i.e. a cluster in progress
   is a period where the CONDITIONAL RATE of further jumps is elevated,
   independent of realized volatility LEVEL; Barndorff-Nielsen & Shephard
   (2004/2006) -- the bipower-variation jump detector defining event
   times.

   NOT A DUPLICATE OF:
   - R-77/B-24 (volatility-driven execution timing, closed NEGATIVE):
     both make patience a function of a VOLATILITY LEVEL (v4's own
     ratio[i] = vol[i]/slow[i]). This branch makes patience a function of
     CLUSTERING INTENSITY -- the conditional rate of further jumps given
     recent jump timing, a quantity no volatility estimator captures (two
     periods with identical realized volatility can have very different
     Hawkes branching-ratio-implied intensity depending on whether that
     volatility arose from one isolated jump or several temporally
     clustered ones). Architecturally this branch reuses R-77's general
     "delay execution during stress, release early or at a deadline"
     shape (``experiments/r77_novel_execution_regime_adaptive.py``), with
     a structurally different DRIVER, exactly the precedent R-88 itself
     set reusing R-77's shape with taker-flow direction as the driver.
   - R-88 (taker-flow execution delay): reuses the bounded-delay-then-
     force ARCHITECTURE (``DelayConfig`` / ``build_delayed_target`` /
     Strategy subclass / K x threshold sweep / ETH falsification / causal
     truncation probe -- structurally identical file shape, deliberately,
     per this round's brief) but keys the delay trigger on Hawkes cluster
     intensity (a purely price-derived, univariate, DIRECTION-BLIND
     signal) rather than signed taker-flow-vs-trade-direction opposition
     (a bivariate buy/sell imbalance reported by the venue, scored
     against the pending trade's own direction). R-88's mechanism only
     delays a trade that fights current flow; this mechanism delays every
     trade while a cluster is active, regardless of the trade's own
     direction -- a different trigger condition on an already-validated
     architecture.
   - The CONSERVATIVE branch running in parallel this round
     (``r96_conservative_hawkes_alarm.py``, not read or touched by this
     file): that branch votes on REGIME (feeds Hawkes intensity into
     v4's own anchor-vote machinery as a detection-lag alarm, tested
     against the six dated stress episodes). This branch changes WHEN an
     already-decided target gets executed; it never touches the target
     value itself except by delaying its application -- same split as
     R-88's own conservative/novel split, and R-81's / R-84's.

2. STEP 0 -- THE MANDATORY PRE-REGISTERED MEASUREMENT GATE, run BEFORE
   any execution-delay logic is built or backtested, BTC inner-train ONLY
   (bars with timestamp < ``INNER_VAL_START`` = 2021-01-01, i.e. strictly
   the 2017-01-01 -> 2020-12-31 inner-train slice -- inner-validation and
   the holdout are never read by this gate).

   QUOTE, from ``r96_shared.py``'s own docstring (the gate this file is
   required to implement): "if realized volatility/whipsaw frequency in
   the bars immediately following a Hawkes-intensity spike is NOT
   significantly elevated relative to the unconditional baseline on
   inner-train, delaying execution during a cluster buys nothing and the
   branch must stop at that pre-registered gate before any delay
   mechanism is built or backtested."

   EXACT DESIGN (fixed before any real number was computed):

   (a) SIGNAL: ``hawkes_intensity_zscore`` at the PRIMARY grid cell
       (n=0.5, halflife=7 days -- the middle of ``r96_shared.N_GRID`` and
       ``r96_shared.HALFLIFE_DAYS_GRID``), causally aligned onto the
       5-minute bar index via ``r96_shared.align_daily_causal`` (a bar at
       time T sees only the most recently CLOSED day's z-score -- the
       same shift convention every prior round in this file's lineage
       uses).

   (b) "CLUSTER SPIKE" bars: the set S of 5-minute bar indices i where
       the causally-aligned z-score crosses UPWARD through
       ``Z_THRESH = 2.0`` (``r96_shared.Z_THRESH``, unchanged): both
       z[i] and z[i-1] finite, z[i-1] < 2.0, z[i] >= 2.0. (Requiring
       z[i-1] finite -- rather than treating a NaN predecessor as "below
       threshold" -- avoids counting the warmup bar where the baseline
       window first fills as a spurious "crossing".) Spikes whose forward
       measurement window would run past the end of inner-train are
       dropped (stated, not silently included as a truncated window).

   (c) FORWARD WINDOW: ``N = 288`` bars (one full calendar day of 5-
       minute bars), measured on bars STRICTLY AFTER the spike bar (bars
       [i+1, i+N]) -- never including bar i itself, preserving causality
       (a strategy could only react after observing the spike). N is
       fixed at one full day, not an hourly sub-window, BECAUSE the
       Hawkes signal itself is defined at DAILY granularity in
       ``r96_shared.py`` (jump events are a once-per-calendar-day flag);
       measuring at the same cadence the signal is defined at is the
       natural "immediately following" horizon and avoids injecting a
       second, arbitrary sub-day timescale into a daily-cadence
       question. This is stated explicitly, per the task brief's
       requirement, BEFORE the number is computed.

   (d) METRIC: realized variance of the forward window,
       ``RV_window(i) = sum_{k=i+1}^{i+N} r_k^2`` (raw sum of squared
       5-minute log returns -- IDENTICAL units/definition to the RV
       already used inside ``r96_shared.intraday_relative_jump``, for
       consistency across this round's files, rather than introducing a
       second volatility convention such as annualized stdev).

   (e) SIGNIFICANCE TEST (pre-registered BEFORE computing any number):
       a block-bootstrap permutation test, reusing
       ``r96_shared.block_bootstrap_shifts`` (block_days=5, n_draws=1000,
       seed=9601 -- fixed now, this file's own seed, chosen before any
       run). Each draw returns a circular block-shift PERMUTATION of the
       bar index; apply that permutation to the log-return array itself
       (NOT to the spike positions) to build a "pseudo-history" whose
       within-block serial correlation structure is preserved but whose
       temporal link to the REAL spike positions is broken. For each
       draw, recompute the mean RV_window over the SAME (real, fixed)
       spike bar positions but using the shifted-return array --
       producing 1000 null-hypothesis values for "mean forward RV at
       these calendar positions if Hawkes spikes carried no information
       about what follows them". Compare the OBSERVED mean RV_window
       (real returns, real spike positions) to this null distribution.

   (f) PRE-REGISTERED PASS/FAIL THRESHOLD (stated now, before the number
       exists): the gate PASSES ("significantly elevated") iff BOTH:
         (i)  at least 20 valid spike events exist in inner-train (a
              power floor -- below this the test is not trusted to
              resolve real from noise regardless of the p-value it
              happens to produce, named now per ROUTINE.md step 2's
              "compute the n a threshold implies and check it is
              reachable" discipline);
         (ii) the observed mean RV_window exceeds the 95th percentile of
              the 1000-draw null distribution (one-sided test at
              alpha=0.05 -- one-sided because the pre-registered
              hypothesis is specifically ELEVATION, not a two-sided
              "different from baseline").
       If EITHER fails, the gate FAILS. Per ``r96_shared.py``'s own
       docstring and this round's brief, a gate FAILURE means: STOP HERE.
       No execution-delay mechanism (Step B) is built or backtested; the
       gate result is this branch's entire product.

   The UNCONDITIONAL BASELINE (all overlapping N=288-bar rolling-window
   RV values across the whole inner-train bar series) is also reported,
   descriptively, alongside the formal block-bootstrap test -- satisfying
   the brief's "versus the unconditional baseline" instruction as a
   sanity-check number, while the actual pass/fail decision is made by
   the block-bootstrap test in (e)/(f), which is the "simple, honest
   significance check" the brief requires rather than an eyeballed point
   comparison.

3. STEP B PRE-REGISTRATION (this section is written and frozen BEFORE
   Step 0's gate number is computed -- IF AND ONLY IF Step 0 passes):

   MECHANISM, EXACT: reuse v4's own real, causal ``target[i]`` array
   unmodified (``KellyRegimeV4().prepare()``). Maintain an executed
   position ``pos`` (starts at 0.0) and the causally-aligned Hawkes
   z-score array (n=0.5, halflife=7 -- FIXED at the Step-0 gate's own
   primary cell throughout Step B; only K and Z_THRESH are swept, never
   n/halflife, so Step B is testing "does bounded delay on THIS
   already-gated signal help", not re-discovering a new grid cell after
   seeing Step 0's result). At each bar i, if
   ``|target[i] - pos| > EPS`` (v4 wants to rebalance):
       spiking = z[i] finite AND z[i] >= Z_THRESH_B
       if not already delaying:
           if spiking: start delaying (pos held at its current value)
           else: pos <- target[i] immediately (no cluster in progress)
       else (already delaying since bar `placed_at`):
           age = i - placed_at
           if (not spiking) or (age >= K):   # cluster subsided, OR
                                                # deadline hit
               pos <- target[i]   # execute the CURRENT desired target,
                                    # never a stale one
               stop delaying
           else: keep delaying, re-check next bar

   Applied via a plain (unregistered) ``Strategy`` subclass,
   ``HawkesDelayV4``, whose ``on_bar`` issues ``ctx.order_notional(pos)``
   exactly like R-88's ``TakerFlowDelayV4`` -- this delays WHEN v4's
   target gets applied to the real broker, using this project's standard
   bar-close-signal / next-bar-open-fill contract, not a resting-limit-
   order microstructure model (same simplification R-88 made, for the
   same reason: isolate the one new variable).

   SWEEP GRID (fixed now, 4 x 2 = 8 configurations, per the task brief):
     K (max delay, bars, 5-min cadence)      in {6, 12, 24, 48}
                                                (30min / 1h / 2h / 4h)
     Z_THRESH_B (cluster-spike trigger)      in {1.5, 2.0}
   Run every config with ``scripts.experiment.ev()`` against BTC spot,
   entry-tier fees (this harness's default), first on inner-train
   (end="2020-12-31") for screening, then on inner-validation
   (start="2021-01-01", end="2022-12-31") to SELECT exactly one winner by
   inner-validation Sharpe among the 8 (ties broken by lower max
   drawdown). ``KellyRegimeV4`` and ``buy_and_hold`` are evaluated on
   both periods too, as comparison baselines (not counted as "configs of
   this idea", but counted in the total evaluation tally per
   docs/ROUTINE.md).
   Pre-registered total ev() calls: 8 configs x 2 periods + 2 baselines
   x 2 periods = 20.

   PRE-REGISTERED HOLDOUT-CONSULTATION DECISION RULE (fixed before Step B
   runs): recommend a holdout consultation to the operator IFF ALL of:
     (a) the winning config beats kelly_regime_v4 on inner-validation
         Sharpe by > +0.2 (the project's noise floor), OR shows a clear
         matched-risk drawdown improvement (materially lower max DD at a
         comparable or better Sharpe);
     (b) it passes the ETH falsification test: same config re-run on
         Coinbase ETH spot bars (own price-derived Hawkes signal, no
         external metrics feed needed), window bounded at 2022-12-31
         (still inside the inner split, never the holdout), PASS = its
         Sharpe on that window is >= plain ``kelly_regime_v4``'s own
         Sharpe on the identical ETH window (directional replication, not
         magnitude-matching, same bar R-77's and R-88's own ETH
         falsification used).
   Anything else -> do not recommend a holdout consultation; report the
   result as this branch's negative/inconclusive product.
   This file has NO authority to consult the holdout regardless of what
   this rule outputs.

4. WHAT WOULD MAKE STEP B FAIL, named now, EVEN IF STEP 0 PASSES (Step 0
   only certifies that a cluster-in-progress predicts elevated near-term
   RV, not that DELAYING execution on it is profitable):
   (a) the elevated RV Step 0 measures could as easily be UNFAVORABLE
       continued momentum as reversible noise -- delaying doesn't avoid
       a bad fill if the price keeps moving the same direction through
       the whole delay window, it just relocates the bad fill later;
   (b) v4's target already rebalances too infrequently (10% deadband,
       latched anchors) for any K in the tested range to matter -- most
       rebalance events might never encounter an active cluster, making
       the mechanism a no-op most of the time;
   (c) forcing the trade through unconditionally at bar K merely
       relocates whatever cost clustering imposes to a different bar
       rather than avoiding it, if clusters regularly outlast every
       tested K (a multi-day cascade, not a short-lived spike) -- this is
       exactly what the diagnostics below (forced-through rate vs.
       resolved-favorably rate) are reported to check.

CONFIGS EVALUATED IN STEP 0: 0 (fixed measurement gate, this project's
standing convention -- the 1000 block-bootstrap draws are a null-
distribution computation, not separate strategy configurations).

USAGE
-----
    python experiments/r96_novel_hawkes_delay_brake.py            # everything
    python experiments/r96_novel_hawkes_delay_brake.py step0       # gate only
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

from tradebot.data import load_coinbase_eth_spot  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

from experiments.r96_shared import (  # noqa: E402
    INNER_TRAIN_END,
    INNER_VAL_START,
    INNER_VAL_END,
    OOS_START,
    Z_THRESH,
    align_daily_causal,
    assert_no_holdout,
    block_bootstrap_shifts,
    hawkes_intensity_daily,
    hawkes_intensity_zscore,
    intraday_relative_jump,
    truncation_causality_probe,
)

from scripts.experiment import DF as BTC_DF, SPOT, ev  # noqa: E402

DATA_DIR = ROOT / "data"

# ------------------------------------------------------------ Step 0 params
PRIMARY_N = 0.5
PRIMARY_HALFLIFE = 7.0
FORWARD_WINDOW_BARS = 288          # one calendar day of 5-minute bars
MIN_SPIKE_EVENTS = 20              # power floor, pre-registered
NULL_DRAWS = 1000
NULL_BLOCK_DAYS = 5
NULL_SEED = 9601
SIG_PCTL = 95.0                    # one-sided, alpha = 0.05

# ------------------------------------------------------------ Step B params
K_GRID = (6, 12, 24, 48)
Z_THRESH_B_GRID = (1.5, 2.0)
EPS_TARGET = 1e-9

CONFIG_COUNTER = {"step0": 0, "stepB": 0, "diagnostic": 0}


def _count(kind: str, k: int = 1) -> None:
    CONFIG_COUNTER[kind] += k


def load_btc_bars(end: str | None = None) -> pd.DataFrame:
    """BTC spot bars, always strictly before the holdout, optionally
    further restricted to `end` (exclusive upper timestamp bound)."""
    df = BTC_DF.loc[BTC_DF.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    if end is not None:
        df = df.loc[df.index < pd.Timestamp(end, tz="UTC")].copy()
    assert_no_holdout(df)
    return df


def compute_causal_hawkes_z(df: pd.DataFrame, n: float = PRIMARY_N,
                             halflife_days: float = PRIMARY_HALFLIFE) -> pd.Series:
    """Bar-indexed, causally-aligned Hawkes intensity z-score, primary
    grid cell, computed from `df` alone (holdout-guarded at every step)."""
    assert_no_holdout(df)
    jump_flag = intraday_relative_jump(df)
    assert_no_holdout(jump_flag.to_frame())
    lam = hawkes_intensity_daily(jump_flag, n=n, halflife_days=halflife_days)
    assert_no_holdout(lam.to_frame())
    z_daily = hawkes_intensity_zscore(lam)
    assert_no_holdout(z_daily.to_frame())
    z_bars = align_daily_causal(z_daily, df)
    assert_no_holdout(z_bars.to_frame())
    return z_bars


# =====================================================================
# STEP 0 -- pre-registered "does a cluster spike predict elevated
# near-term realized volatility" gate
# =====================================================================

def find_upward_crossings(z: np.ndarray, thresh: float) -> np.ndarray:
    """Bar indices i where z[i-1] and z[i] are both finite, z[i-1] <
    thresh <= z[i]. Index 0 can never be a crossing (no predecessor)."""
    finite = np.isfinite(z)
    high = z >= thresh
    valid = finite[1:] & finite[:-1]
    cross = np.zeros(len(z), dtype=bool)
    cross[1:] = valid & high[1:] & ~high[:-1]
    return np.where(cross)[0]


def step_0_gate() -> dict:
    print("=" * 78)
    print("R-96 NOVEL: Hawkes cluster-intensity execution brake -- STEP 0 gate")
    print("=" * 78)

    bars = load_btc_bars(end=INNER_VAL_START)  # inner-train only, 2017-01-01 -> 2020-12-31
    print(f"inner-train BTC spot: {len(bars):,} bars  {bars.index[0]} -> {bars.index[-1]}")

    z = compute_causal_hawkes_z(bars).to_numpy(dtype=float)
    r = np.log(bars["close"]).diff().to_numpy(dtype=float).copy()
    r[0] = 0.0  # first bar has no return; excluded from any window regardless
    n_bars = len(bars)

    print(f"\nsignal: hawkes_intensity_zscore, primary grid cell n={PRIMARY_N} "
          f"halflife={PRIMARY_HALFLIFE}d, Z_THRESH={Z_THRESH}")
    print(f"finite z bars: {int(np.isfinite(z).sum()):,}/{n_bars:,} "
          f"(first finite at {bars.index[np.where(np.isfinite(z))[0][0]] if np.isfinite(z).any() else 'n/a'})")

    all_spikes = find_upward_crossings(z, Z_THRESH)
    # Drop spikes whose forward window would run past the end of inner-train.
    usable = all_spikes[all_spikes + FORWARD_WINDOW_BARS < n_bars]
    dropped = len(all_spikes) - len(usable)
    print(f"\ncluster-spike upward crossings found: {len(all_spikes)}  "
          f"(dropped {dropped} with insufficient forward window; usable={len(usable)})")

    if len(usable) > 0:
        example_times = bars.index[usable[:5]]
        print(f"first few spike timestamps: {list(example_times)}")

    def rv_window(returns: np.ndarray, i: int) -> float:
        seg = returns[i + 1: i + 1 + FORWARD_WINDOW_BARS]
        return float(np.sum(seg ** 2))

    observed_rv = np.array([rv_window(r, i) for i in usable]) if len(usable) else np.array([])
    observed_mean = float(np.mean(observed_rv)) if len(observed_rv) else float("nan")

    # Descriptive unconditional baseline: all overlapping N-bar rolling-window
    # RV values across the whole inner-train series (reported, not used for
    # the formal pass/fail decision).
    r2 = r ** 2
    csum = np.concatenate([[0.0], np.cumsum(r2)])
    if n_bars > FORWARD_WINDOW_BARS:
        roll_rv = csum[FORWARD_WINDOW_BARS:] - csum[:-FORWARD_WINDOW_BARS]
    else:
        roll_rv = np.array([])
    baseline_mean = float(np.mean(roll_rv)) if len(roll_rv) else float("nan")
    baseline_median = float(np.median(roll_rv)) if len(roll_rv) else float("nan")

    print(f"\nobserved mean RV_window (forward {FORWARD_WINDOW_BARS}-bar, post-spike): "
          f"{observed_mean:.6e}  (n={len(observed_rv)} spike events)")
    print(f"unconditional baseline (all overlapping {FORWARD_WINDOW_BARS}-bar windows, "
          f"inner-train): mean={baseline_mean:.6e}  median={baseline_median:.6e}  "
          f"(n={len(roll_rv):,} windows)")
    if np.isfinite(observed_mean) and np.isfinite(baseline_mean) and baseline_mean > 0:
        print(f"observed/baseline ratio: {observed_mean / baseline_mean:.3f}x")

    # --- Pre-registered block-bootstrap significance test ---
    enough_events = len(usable) >= MIN_SPIKE_EVENTS
    null_means = np.array([])
    null_p95 = float("nan")
    sig = False
    if enough_events:
        shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=NULL_BLOCK_DAYS,
                                         n_draws=NULL_DRAWS, seed=NULL_SEED)
        null_means = np.empty(len(shifts))
        for k, shift in enumerate(shifts):
            r_shifted = r[shift]
            vals = np.array([rv_window(r_shifted, i) for i in usable])
            null_means[k] = float(np.mean(vals))
        null_p95 = float(np.percentile(null_means, SIG_PCTL))
        sig = bool(observed_mean > null_p95)

    print(f"\nblock-bootstrap null: {NULL_DRAWS} draws, block_days={NULL_BLOCK_DAYS}, "
          f"seed={NULL_SEED} (permutes RETURNS cyclically in blocks, keeps real spike "
          f"positions fixed, recomputes mean forward RV at those positions)")
    if enough_events:
        print(f"null distribution of mean RV_window: mean={float(np.mean(null_means)):.6e}  "
              f"p50={float(np.median(null_means)):.6e}  p95={null_p95:.6e}")
        print(f"observed mean RV_window ({observed_mean:.6e}) > null p95 ({null_p95:.6e}): {sig}")
    else:
        print(f"SKIPPED: only {len(usable)} usable spike events, below the pre-registered "
              f"power floor of {MIN_SPIKE_EVENTS} -- test not run, treated as FAIL.")

    passed = enough_events and sig

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE: gate PASSES iff (i) >= 20 usable spike events, AND")
    print("(ii) observed mean forward-288-bar RV exceeds the 95th percentile of the")
    print("1000-draw block-bootstrap null. Otherwise: STOP, no execution-delay model built.")
    print("=" * 78)
    print(f"  usable spike events: {len(usable)}  (floor: {MIN_SPIKE_EVENTS})  -> {enough_events}")
    print(f"  observed > null p95: {sig}")
    print(f"STEP 0 GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no execution model built'}")
    print(f"\nconfigurations evaluated in Step 0: 0 (fixed measurement gate)")
    print(f"max timestamp read in Step 0: {bars.index.max()}  (< {INNER_VAL_START})")

    return dict(bars=bars, z=z, usable_spikes=usable, observed_mean=observed_mean,
                baseline_mean=baseline_mean, null_means=null_means, null_p95=null_p95,
                enough_events=enough_events, sig=sig, passed=passed)


# =====================================================================
# STEP B -- execution-delay mechanism (only if Step 0 passes)
# =====================================================================

@dataclass(frozen=True)
class DelayConfig:
    k_max: int
    z_thresh: float

    def tag(self) -> str:
        return f"K{self.k_max}_z{self.z_thresh:g}"


def build_delayed_target(target: np.ndarray, z_bars: np.ndarray,
                          cfg: DelayConfig) -> tuple[np.ndarray, dict]:
    """Causal, bar-by-bar construction of the delayed execution series.

    Row i depends only on target[<=i], z_bars[<=i], and the running (pos,
    pending_since) state built from strictly earlier bars -- causal by
    construction, checked below with `truncation_causality_probe`.
    Direction-blind: any pending rebalance is delayed while z_bars[i] >=
    cfg.z_thresh (a cluster is in progress), regardless of whether the
    rebalance is a buy or a sell.
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

        z = z_bars[i]
        spiking = np.isfinite(z) and (z >= cfg.z_thresh)

        if pending_since is None:
            if spiking:
                pending_since = i
                n_delayed += 1
                out[i] = pos  # hold, do not execute yet
            else:
                pos = desired
                out[i] = pos
            continue

        age = i - pending_since
        if (not spiking) or age >= cfg.k_max:
            if spiking and age >= cfg.k_max:
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


class HawkesDelayV4(Strategy):
    """kelly_regime_v4 with rebalances delayed up to K bars while a Hawkes
    jump cluster (intensity z-score >= z_thresh) is in progress."""

    warmup = KellyRegimeV4().warmup

    def __init__(self, k_max: int = 12, z_thresh: float = 2.0) -> None:
        self.name = f"hawkes_delay_v4[{DelayConfig(k_max, z_thresh).tag()}]"
        self.cfg = DelayConfig(k_max, z_thresh)
        self._last_diag: dict = {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        assert_no_holdout(df)
        v4_target = KellyRegimeV4().prepare(df.copy())["target"].to_numpy(dtype=float)
        z_bars = compute_causal_hawkes_z(df).to_numpy(dtype=float)

        delayed, diag = build_delayed_target(v4_target, z_bars, self.cfg)
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
    print("STEP B -- Hawkes execution-delay sweep, BTC spot, entry-tier fees "
          f"({len(K_GRID)}x{len(Z_THRESH_B_GRID)}={len(K_GRID)*len(Z_THRESH_B_GRID)} configs)")
    print("=" * 100)

    configs = [DelayConfig(k, z) for k in K_GRID for z in Z_THRESH_B_GRID]
    print(f"pre-registered grid: K in {K_GRID}, Z_THRESH_B in {Z_THRESH_B_GRID}  "
          f"({len(configs)} configs); n/halflife fixed at primary cell "
          f"(n={PRIMARY_N}, halflife={PRIMARY_HALFLIFE}d)\n")

    train_rows, val_rows = {}, {}

    print("-- inner-train (<= 2020-12-31) --")
    for cfg in configs:
        strat = HawkesDelayV4(k_max=cfg.k_max, z_thresh=cfg.z_thresh)
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
        strat = HawkesDelayV4(k_max=cfg.k_max, z_thresh=cfg.z_thresh)
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
          f"window ({INNER_VAL_START} -> {INNER_VAL_END})")
    print("=" * 100)

    eth_df = load_coinbase_eth_spot(DATA_DIR)
    assert eth_df is not None, "ETH coinbase spot file missing"
    eth_df = eth_df.loc[eth_df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(eth_df)
    print(f"ETH coinbase spot bars available (< holdout): {len(eth_df):,}  "
          f"{eth_df.index[0]} -> {eth_df.index[-1]}")

    strat = HawkesDelayV4(k_max=winner.k_max, z_thresh=winner.z_thresh)
    m = ev(strat, df=eth_df, market=SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
           tag=f"ETH   {winner.tag()}")
    _count("stepB")
    m_v4 = ev(KellyRegimeV4(), df=eth_df, market=SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
              tag="ETH   kelly_regime_v4")
    _count("stepB")

    passed = m.sharpe >= m_v4.sharpe
    print(f"\nETH falsification: winner sharpe={m.sharpe:.3f}  v4 sharpe={m_v4.sharpe:.3f}  "
          f"PASS (winner >= v4, directional replication): {passed}")
    print(f"max timestamp read in ETH falsification: {eth_df.index.max()}  (< {OOS_START})")
    return dict(winner_sharpe=m.sharpe, v4_sharpe=m_v4.sharpe, passed=passed)


def causality_probe(winner: DelayConfig) -> bool:
    print("\n" + "=" * 100)
    print(f"CAUSAL-TRUNCATION PROBE -- {winner.tag()}, r96_shared.truncation_causality_probe")
    print("=" * 100)

    df = load_btc_bars()
    assert_no_holdout(df)

    def build_target_fn(frame: pd.DataFrame) -> np.ndarray:
        v4_target = KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float)
        z_bars = compute_causal_hawkes_z(frame).to_numpy(dtype=float)
        delayed, _ = build_delayed_target(v4_target, z_bars, winner)
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

    gate_result = step_0_gate()

    if choice == "step0":
        print(f"\nCONFIGS EVALUATED: step0={CONFIG_COUNTER['step0']} "
              f"stepB={CONFIG_COUNTER['stepB']} diagnostic={CONFIG_COUNTER['diagnostic']}")
        print(f"[{time.time()-t0:.0f}s]")
        return

    if not gate_result["passed"]:
        print("\n" + "#" * 78)
        print("# STEP 0 FAILED ITS PRE-REGISTERED STOP RULE.")
        print("# Per this file's own pre-registration (and r96_shared.py's docstring),")
        print("# STOP HERE. No execution-delay model is built; Step B is not run. This")
        print("# gate result is this branch's whole product.")
        print("#" * 78)
        print(f"\nCONFIGS EVALUATED (TOTAL): step0=0 stepB=0 diagnostic=0")
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
    total = CONFIG_COUNTER["step0"] + CONFIG_COUNTER["stepB"] + CONFIG_COUNTER["diagnostic"]
    print(f"\nCONFIGS EVALUATED (TOTAL): step0={CONFIG_COUNTER['step0']} "
          f"stepB={CONFIG_COUNTER['stepB']} diagnostic={CONFIG_COUNTER['diagnostic']} "
          f"total={total}")
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
