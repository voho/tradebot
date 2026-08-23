#!/usr/bin/env python
"""R-96 NOVEL branch: Hawkes cluster-INTENSITY-conditioned EXECUTION BRAKE
for ``kelly_regime_v4`` -- Step-0 sub-claim measurement gate ONLY.

=====================================================================
PRE-REGISTRATION (frozen before any real-data number in this file was
computed -- docs/ROUTINE.md steps 1-2, ``r96_shared.py``'s own closing
docstring paragraph names this exact gate as the round's a-priori
expected stop point). Anything below later contradicted by what actually
happened is stated in the results section, not edited back into this
banner.
=====================================================================

1. MECHANISM (one sentence). If bars immediately following a Hawkes-
   intensity cluster spike show significantly elevated realized
   volatility / whipsaw (position-flip) frequency relative to the
   unconditional baseline, delaying ``kelly_regime_v4``'s scheduled
   rebalance during such a spike could reduce adverse-execution cost; if
   not, there is nothing for a delay mechanism to buy, and the branch
   stops here.

   This reuses R-88's own validated "bounded-delay-then-force"
   architecture shape (postpone a scheduled rebalance up to K bars,
   re-check every bar, force through at a deadline) but keyed on Hawkes
   CLUSTER INTENSITY -- a purely price-derived, univariate conditional
   event-rate -- rather than R-88's order-flow direction or R-77/B-24's
   volatility LEVEL. See ``r96_shared.py``'s module docstring for the
   full citation trail and not-a-duplicate-of argument; not re-derived
   here.

2. SCOPE: INNER-TRAIN ONLY (``r96_shared.INNER_TRAIN_END`` =
   "2020-12-31"), strictly narrower than this project's general holdout
   rule (``OOS_START`` = "2023-01-01"). No bar dated 2021-01-01 or later
   is read, printed, or used anywhere in this file. Enforced in code by
   ``assert_no_inner_val_or_later`` below (max timestamp < "2021-01-01"),
   not merely asserted in prose -- this is the narrowest-scope gate in
   the whole R-96 round.

3. PRIMARY CONFIG (fixed a priori, matching the conservative branch's own
   choice for consistency across this round's two branches -- disclosed
   here as reuse, not independently re-derived): ``n=0.5``,
   ``halflife_days=7`` (the a-priori median cell of each of
   ``r96_shared.N_GRID`` / ``HALFLIFE_DAYS_GRID``). ``event_flag =
   intraday_relative_jump(bars)``; ``lam = hawkes_intensity_daily(
   event_flag, n=0.5, halflife_days=7)``; ``z_daily =
   hawkes_intensity_zscore(lam)``.

4. SPIKE-ONSET DEFINITION (disclosed a-priori design choice, avoids
   double-counting consecutive days inside one cluster as independent
   samples): day ``t`` is a "spike onset" iff ``z_daily[t] >=
   Z_THRESH(2.0)`` AND (``t`` is the first row, OR ``z_daily[t-1] <
   Z_THRESH`` or NaN) -- the START of each cluster episode only.

5. TWO SUB-CLAIMS, BOTH on inner-train only:

   (a) FORWARD REALIZED VOLATILITY. For each spike-onset day ``t``, the
       realized volatility (RV = sum of squared 5-minute log returns) of
       calendar day ``t+1``, via a standalone ``daily_rv(bars)`` helper
       recomputing RV directly from bars (``r96_shared`` does not expose
       RV as a standalone function -- only the RJ statistic derived from
       it). Baseline population = the identical day-``t+1``-style forward
       RV for every non-onset day in the same inner-train range, using
       ITS OWN next-day RV.

   (b) FORWARD WHIPSAW (POSITION-FLIP) FREQUENCY. For each spike-onset
       day ``t``, the count of ``anchor_majority`` value-changes (v4's
       own vote, via ``r96_shared.anchor_majority``) occurring in the
       following 3 calendar days (``t+1`` through ``t+3`` inclusive) --
       this project's actual position-relevant "whipsaw" unit, not raw
       price sign flips. Baseline = the identical 3-day-forward flip
       count for every non-onset day.

   COMPARISON METHOD (identical shape for both, seeded independently):
   plain two-sample comparison (onset mean, baseline mean, difference)
   PLUS a 95% CI via block bootstrap of the NON-ONSET (baseline)
   population only (block_days=7, n_draws=1000, holding the onset-day
   sample fixed) -- a dependency-free (numpy only, no scipy) circular
   moving-block bootstrap of the baseline mean's sampling distribution,
   matching this project's standing hand-rolled-statistics convention
   (R-65/67/68/79/85/86). Seeds: 9602 for (a), 9603 for (b) (independent
   families, fixed a priori). One-sided empirical p-value = fraction of
   the 1000 bootstrap baseline-mean resamples that meet or exceed the
   TRUE onset-day mean.

6. PRE-REGISTERED STOP RULE (frozen, not relaxed after seeing numbers):
   proceed to build a delay-execution strategy ONLY IF BOTH (a) and (b)
   show the true onset-day mean exceeding the 95th percentile of its own
   block-bootstrapped baseline-mean distribution (equivalently, one-sided
   empirical p < 0.05). If EITHER metric fails this bar: STOP. Report
   NEGATIVE. Do not write any strategy/delay-mechanism code. Do not run
   any ``scripts.experiment.ev()`` backtest. This matches
   ``r96_shared.py``'s own docstring, which already names this branch's
   a-priori expected outcome: raw volatility/whipsaw elevation after a
   jump cluster is close to tautological given the Hawkes construction
   itself (metric (a) may pass mechanically), while whether that
   elevation actually shows up as EXTRA position flips v4's own
   latched/hysteresis vote has not already absorbed (metric (b)) is the
   less obvious, economically decisive question -- if (a) passes but (b)
   does not, that is called out explicitly below as the economically
   meaningful reading, not silently averaged away.

7. CAUSALITY. ``causality_probe()`` re-verifies, independently in this
   file (this project's standing rule: every round re-runs the probe
   itself rather than trusting a prior claim), that the
   ``daily_rv``/Hawkes-z composition used above is causal, via
   ``r96_shared.truncation_causality_probe``, probed at a point well
   inside inner-train ("2019-06-01"), same convention as the conservative
   branch's own ``causality_probe()``.

8. CONFIGURATIONS EVALUATED: 0 (a fixed, non-swept measurement gate --
   this project's standing convention for a Step-0/Step-A gate) + 1
   diagnostic causal-truncation probe.

WHAT WOULD MAKE THIS FAIL, named now: exactly the failure named in
``r96_shared.py``'s own closing paragraph -- realized volatility rising
after a jump cluster is close to definitional (a Hawkes cluster spike
IS, by construction, a run of recent large moves; the following day's
volatility level is highly likely to still be elevated by pure
persistence, no delay mechanism required to notice that). Whether extra
POSITION FLIPS follow is a materially different, and the actually
decisive, question -- v4's own 20/40/80-day rolling-mean anchors with a
1% deadband already smooth over ordinary day-to-day volatility, so an
elevated-RV day converts into an extra vote flip only if the move is
large enough, and persistent enough, to cross an anchor band it had not
already crossed. If it does not, delaying execution during a cluster
buys nothing to protect against, whatever the RV metric alone says.

USAGE
-----
    python experiments/r96_novel_execution_brake.py
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

from experiments.r96_shared import (  # noqa: E402
    INNER_TRAIN_END,
    INNER_VAL_START,
    OOS_START,
    Z_THRESH,
    align_daily_causal,
    anchor_majority,
    assert_no_holdout,
    hawkes_intensity_daily,
    hawkes_intensity_zscore,
    intraday_relative_jump,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"

PRIMARY_N = 0.5
PRIMARY_HALFLIFE_DAYS = 7

BLOCK_DAYS = 7
N_DRAWS = 1_000
SEED_RV = 9602
SEED_WHIPSAW = 9603
WHIPSAW_FWD_DAYS = 3

CONFIG_COUNTER = {"gate": 0, "diagnostic": 0}


# ---------------------------------------------------------------- holdout guard


def assert_no_inner_val_or_later(obj) -> None:
    """This branch's own TIGHTER guard than ``r96_shared.assert_no_holdout``
    (whose cutoff is ``OOS_START`` = "2023-01-01"): this file's Step-0 gate
    is pre-registered as inner-train ONLY, so the max timestamp anywhere
    this file touches must be strictly before ``INNER_VAL_START`` =
    "2021-01-01"."""
    idx = obj.index if hasattr(obj, "index") else obj
    if len(idx) == 0:
        return
    cutoff = pd.Timestamp(INNER_VAL_START, tz="UTC")
    max_ts = pd.Timestamp(idx.max())
    if max_ts.tzinfo is None:
        max_ts = max_ts.tz_localize("UTC")
    assert max_ts < cutoff, (
        f"inner-val-or-later bar read: max timestamp {max_ts} >= {INNER_VAL_START}. "
        "This file's Step-0 gate must never read inner-validation or holdout data.")


def load_btc_bars_inner_train() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(INNER_TRAIN_END, tz=df.index.tz)].copy()
    assert_no_inner_val_or_later(df)
    assert_no_holdout(df)  # redundant second guard against OOS_START too
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {INNER_TRAIN_END}, inner-train ONLY -- this file's own scope, "
          f"stricter than {OOS_START})")
    print(f"max timestamp read: {df.index.max()}  (< {INNER_VAL_START})")
    return df


# ---------------------------------------------------------------- helpers


def daily_rv(bars: pd.DataFrame) -> pd.Series:
    """Daily realized variance (sum of squared 5-minute log returns) -- the
    RV half of `intraday_relative_jump`'s own RJ statistic, exposed
    standalone here since `r96_shared` does not export it directly (its
    docstring: recompute directly from bars). Entirely within-day, no
    cross-day lookahead."""
    close = bars["close"]
    r = np.log(close).diff()
    day = bars.index.floor("D")
    frame = pd.DataFrame({"r": r.to_numpy(), "day": day}).dropna(subset=["r"])
    rv_by_day = frame.groupby("day")["r"].apply(lambda g: float(np.sum(g.to_numpy() ** 2)),
                                                 include_groups=False)
    rv_by_day.index = pd.DatetimeIndex(rv_by_day.index, tz="UTC")
    return rv_by_day.rename("rv")


def daily_flip_counts(bars: pd.DataFrame) -> pd.Series:
    """Number of `anchor_majority` value-changes per calendar day -- v4's
    own position-relevant whipsaw unit (not raw price sign flips)."""
    majority = anchor_majority(bars)
    changed = (majority.diff().fillna(0.0) != 0.0).to_numpy()
    day = bars.index.floor("D")
    counts = pd.Series(changed.astype(float), index=day).groupby(level=0).sum()
    counts.index = pd.DatetimeIndex(counts.index, tz="UTC")
    return counts.rename("flip_count")


def forward_window_series(daily: pd.Series, offset_days: int) -> pd.Series:
    """`daily[t]` -> the value observed on day `t + offset_days`, NaN if
    that day is not present in `daily`'s own index (e.g. the inner-train
    boundary, where t+1..t+3 may fall outside the loaded range)."""
    target_idx = daily.index + pd.Timedelta(days=offset_days)
    out = daily.reindex(target_idx)
    out.index = daily.index
    return out


def spike_onsets(z_daily: pd.Series, z_thresh: float = Z_THRESH) -> pd.DatetimeIndex:
    """Day t is a spike ONSET iff z[t] >= z_thresh AND (t is the first row,
    or z[t-1] < z_thresh or NaN) -- the START of each cluster episode only,
    per this file's pre-registered section 4."""
    idx = z_daily.index
    vals = z_daily.to_numpy(dtype=float)
    high = np.where(np.isnan(vals), False, vals >= z_thresh)
    onset = np.zeros(len(high), dtype=bool)
    if len(high):
        onset[0] = bool(high[0])
        onset[1:] = high[1:] & (~high[:-1])
    return idx[onset]


def block_bootstrap_mean(values: np.ndarray, block_days: int, n_draws: int, seed: int) -> np.ndarray:
    """Dependency-free (numpy only) circular moving-block bootstrap of the
    MEAN of `values` (already in chronological order) -- matching this
    project's standing hand-rolled-statistics convention (R-65/67/68/79/
    85/86, no scipy). Each of `n_draws` resamples draws ceil(n/block_days)
    blocks of `block_days` consecutive (circularly wrapped) observations
    with replacement and averages the length-n concatenation."""
    rng = np.random.default_rng(seed)
    n = len(values)
    assert n > 0, "empty baseline population -- cannot bootstrap"
    n_blocks = int(np.ceil(n / block_days))
    offsets = np.arange(block_days)
    means = np.empty(n_draws)
    for d in range(n_draws):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]) % n
        means[d] = float(np.mean(values[idx.ravel()[:n]]))
    return means


def sub_claim_gate(values: pd.Series, onset_days: pd.DatetimeIndex, block_days: int,
                    n_draws: int, seed: int, label: str) -> dict:
    """Plain two-sample comparison (onset mean vs. baseline mean) plus the
    block-bootstrap-of-the-baseline-mean CI/one-sided-p construction, per
    this file's pre-registered section 5/6."""
    valid = values.dropna().sort_index()
    is_onset = valid.index.isin(onset_days)
    onset_vals = valid.to_numpy()[is_onset]
    baseline_vals = valid.to_numpy()[~is_onset]

    onset_mean = float(np.mean(onset_vals)) if len(onset_vals) else float("nan")
    baseline_mean = float(np.mean(baseline_vals)) if len(baseline_vals) else float("nan")
    diff = onset_mean - baseline_mean

    boot_means = block_bootstrap_mean(baseline_vals, block_days, n_draws, seed)
    p95 = float(np.percentile(boot_means, 95))
    p_value = float(np.mean(boot_means >= onset_mean)) if len(onset_vals) else float("nan")
    passed = bool(len(onset_vals) > 0 and onset_mean > p95)

    print(f"\n{label}")
    print(f"  spike onsets total: {len(onset_days)}   with valid forward value: {len(onset_vals)}")
    print(f"  baseline (non-onset) days with valid forward value: {len(baseline_vals)}")
    print(f"  onset mean     = {onset_mean:.6g}")
    print(f"  baseline mean  = {baseline_mean:.6g}")
    print(f"  difference     = {diff:+.6g}")
    print(f"  block-bootstrap of baseline mean (block_days={block_days}, "
          f"n_draws={n_draws}, seed={seed}): p95={p95:.6g}")
    print(f"  one-sided empirical p-value (frac of bootstrap baseline means >= onset mean): "
          f"{p_value:.4f}")
    print(f"  PASS (onset mean > bootstrap-baseline p95, i.e. p<0.05): {passed}")

    return dict(label=label, n_onset=len(onset_vals), n_onset_total=len(onset_days),
                n_baseline=len(baseline_vals), onset_mean=onset_mean,
                baseline_mean=baseline_mean, diff=diff, p95=p95, p_value=p_value,
                passed=passed)


# ---------------------------------------------------------------- causality


def causality_probe() -> bool:
    print("\n" + "=" * 78)
    print("CAUSAL-TRUNCATION PROBE -- daily_rv/Hawkes-z composition, "
          "r96_shared.truncation_causality_probe")
    print("=" * 78)

    bars = load_btc_bars_inner_train()
    check_at = bars.index.get_indexer([pd.Timestamp("2019-06-01", tz="UTC")], method="nearest")[0]

    def build(df: pd.DataFrame) -> np.ndarray:
        event_flag = intraday_relative_jump(df)
        lam = hawkes_intensity_daily(event_flag, n=PRIMARY_N, halflife_days=PRIMARY_HALFLIFE_DAYS)
        z = hawkes_intensity_zscore(lam)
        return align_daily_causal(z, df).to_numpy()

    ok = truncation_causality_probe(build, bars, check_at, shorter_by=20_000)
    print(f"  check_at bar={check_at}  (~{bars.index[check_at]})  shorter_by=20,000 bars")
    print(f"  CAUSAL-TRUNCATION PROBE: {'PASS' if ok else 'FAIL'}")
    CONFIG_COUNTER["diagnostic"] += 1
    return ok


# ---------------------------------------------------------------- main gate


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("R-96 NOVEL: Hawkes cluster-intensity execution brake -- Step-0 sub-claim gate")
    print("=" * 78)

    bars = load_btc_bars_inner_train()

    event_flag = intraday_relative_jump(bars)
    lam = hawkes_intensity_daily(event_flag, n=PRIMARY_N, halflife_days=PRIMARY_HALFLIFE_DAYS)
    z_daily = hawkes_intensity_zscore(lam)
    assert_no_inner_val_or_later(z_daily.dropna())

    onset_days = spike_onsets(z_daily, Z_THRESH)
    print(f"\nprimary config: n={PRIMARY_N}  halflife_days={PRIMARY_HALFLIFE_DAYS}  "
          f"Z_THRESH={Z_THRESH}")
    print(f"spike-onset days found in inner-train: {len(onset_days)}")
    if len(onset_days):
        print(f"  first onset: {onset_days.min()}   last onset: {onset_days.max()}")

    # ---- (a) forward realized volatility ----
    rv = daily_rv(bars)
    fwd_rv = forward_window_series(rv, offset_days=1)
    result_a = sub_claim_gate(fwd_rv, onset_days, BLOCK_DAYS, N_DRAWS, SEED_RV,
                               "(a) FORWARD REALIZED VOLATILITY (day t+1 RV)")

    # ---- (b) forward whipsaw (position-flip) frequency ----
    flips = daily_flip_counts(bars)
    fwd_flip = forward_window_series(flips, 1)
    for k in range(2, WHIPSAW_FWD_DAYS + 1):
        fwd_flip = fwd_flip + forward_window_series(flips, k)  # NaN if any offset day missing
    result_b = sub_claim_gate(fwd_flip, onset_days, BLOCK_DAYS, N_DRAWS, SEED_WHIPSAW,
                               f"(b) FORWARD WHIPSAW COUNT (t+1..t+{WHIPSAW_FWD_DAYS} anchor_majority flips)")

    CONFIG_COUNTER["gate"] = 0  # fixed, non-swept measurement gate

    gate_passed = result_a["passed"] and result_b["passed"]

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE: proceed to build a delay-execution strategy ONLY IF")
    print("BOTH (a) and (b) show onset mean > 95th percentile of their own block-bootstrap")
    print("baseline distribution (one-sided empirical p < 0.05).")
    print("=" * 78)
    print(f"  (a) forward RV       PASS: {result_a['passed']}  (p={result_a['p_value']:.4f})")
    print(f"  (b) forward whipsaw  PASS: {result_b['passed']}  (p={result_b['p_value']:.4f})")

    if result_a["passed"] and not result_b["passed"]:
        print("\n  NOTE (the economically meaningful reading, named a priori in this file's own")
        print("  banner and in r96_shared.py's docstring): (a) passed -- forward volatility is")
        print("  elevated after a Hawkes cluster spike, close to definitional given the Hawkes")
        print("  construction itself (a cluster IS a run of large recent moves; persistence")
        print("  alone predicts the next day is still turbulent). (b) failed -- that elevated")
        print("  volatility does NOT translate into extra kelly_regime_v4 anchor-vote flips.")
        print("  v4's own 20/40/80-day rolling anchors with a 1% deadband already absorb this")
        print("  volatility without extra position changes, so there is nothing an execution")
        print("  brake would protect against: v4 is not about to whipsaw its own position")
        print("  during these spikes, whatever the raw RV number says.")

    print(f"\nSTEP-0 GATE VERDICT: {'PASS -> could proceed to build the delay mechanism' if gate_passed else 'FAIL -> STOP'}")
    if not gate_passed:
        print("\n" + "#" * 78)
        print("# STEP-0 SUB-CLAIM GATE FAILED ITS PRE-REGISTERED STOP RULE.")
        print("# Per this file's own pre-registration (section 6), STOP HERE. No delay-")
        print("# execution strategy code is written, and no scripts.experiment.ev() backtest")
        print("# is run. This gate result is this branch's entire product.")
        print("#" * 78)

    probe_ok = causality_probe()

    print(f"\nconfigurations evaluated: gate={CONFIG_COUNTER['gate']} "
          f"(fixed, non-swept measurement gate) diagnostic={CONFIG_COUNTER['diagnostic']} "
          f"(causal-truncation probe)")
    print(f"causal-truncation probe: {'PASS' if probe_ok else 'FAIL'}")
    print(f"\nmax timestamp read: {bars.index.max()}  (< {INNER_VAL_START})")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
