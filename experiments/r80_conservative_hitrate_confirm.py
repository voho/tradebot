#!/usr/bin/env python
"""R-80 CONSERVATIVE branch: a causal trailing hit-rate confirming vote on
`kelly_regime_v4`'s 3-anchor regime gate -- attacking ERR.

=====================================================================
PRE-REGISTRATION (frozen before any backtest number in this file was
computed -- see docs/ROUTINE.md step 2/4). Everything under this banner
was written and committed to before `main()` executed a single `ev()`
call. If anything below is later contradicted by "what actually
happened", that will be stated explicitly in the text report, not edited
here.
=====================================================================

1. MECHANISM (one sentence). `kelly_regime_v4` acts on its 3-anchor vote
   with full weight regardless of whether that vote has recently been
   right or wrong (the ERR constraint in docs/LEDGER.md); this branch
   adds a 4th, purely-frequentist confirming vote -- the trailing,
   same-regime-bucket hit rate of fully-resolved forward returns -- so
   exposure leans harder into the vote in regimes where it has recently
   paid off and backs off where it has recently been wrong, via R-53/
   R-55's validated CONFIRMING-VOTE combination rule (not a brake: R-34/
   R-41/R-53-conservative/R-73-conservative are 4-for-4 failed on that
   architecture regardless of signal, per docs/LEDGER.md's standing
   lesson, and this round does not repeat it).

   Not a duplicate of R-04: R-04's meta-labeling label looked *forward*
   past the point of use (an explicit, diagnosed lookahead bug in the
   LABEL, not a refutation of meta-labeling as a mechanism). This round's
   label (see `compute_meta_vote` below) is only ever read at bar `i`
   once every forward-return window it is built from has closed strictly
   before bar `i` -- verified both by construction and by the mandatory
   truncation-causality probe (docs/LEDGER.md's now-standard tool,
   `r80_shared.truncation_causality_probe`).

   Not a duplicate of the brake pattern: `confirming_vote_frac` combines
   the anchor vote and the meta-vote via `frac=(anchor_sum+weight*meta)/
   (3+weight)`, which can push frac UP (meta_vote>anchor average) or DOWN
   (meta_vote<anchor average) -- it is not a `mult<=1` multiplicative
   haircut of an existing exposure.

2. PRE-REGISTERED FALSIFICATION TEST, chosen now, before any number.
   `r80_shared.placebo_offset_indices` supplies `n_draws` circular
   time-shifts of a length-n index. This round applies that shift to the
   *hit-label series* (whether each historical bar's forward return was
   positive) while leaving the true regime-bucket sequence (`anchor_sum`
   at each bar) untouched -- i.e. it asks: does conditioning the
   confirming vote on "the regime bucket that was actually in force"
   produce more apparent structure than conditioning it on a randomly
   time-shifted copy of the very same win/loss sequence? A plain
   within-series shuffle would not control for the fact that BTC's price
   history is a multi-year trending series in which almost any
   coarse partition looks like it has "structure" (this is exactly the
   confound R-79 controlled for with the identical placebo-offset device
   on a different signal, and the standing recommended template per that
   round's own writeup).

   Protocol, frozen: primary candidate = weight=1.0, W_days=90, H_days=5
   (chosen a priori -- see SWEEP GRID below for the reasoning, not fit to
   any result). Compute its inner-validation SPOT Sharpe delta over
   `kelly_regime_v4` on (a) the true, zero-shift alignment and (b) N=20
   placebo-offset draws (`block_days=90`, matching the primary's own
   trailing window, `seed=8053`, fixed here before running).

   DECISION RULE (the outcome that kills it, named now): KILL if the true
   delta does not exceed the placebo null's 90th percentile, OR if the
   true delta is not at least one null standard deviation above the null
   mean. Both conditions must be met to call the mechanism "doing more
   than generic trend/autocorrelation structure"; failing either is a
   kill for this specific falsification gate, independent of whatever
   the raw inner-validation Sharpe numbers say. This does not by itself
   decide promotion (the operator applies the ±0.2 Sharpe floor and the
   holdout separately, per docs/ROUTINE.md step 4) -- it decides whether
   the confirming-vote's apparent edge, if any, is worth reading the
   holdout for at all.

3. SWEEP GRID, fixed a priori (documented reasoning, not tuned to any
   inner-validation number seen before this file was written):
   - `weight` in {0.5, 1.0, 2.0, 4.0} -- 1.0 gives the meta-vote equal
     footing with a single one of the three existing anchor votes in the
     `(anchor_sum+weight*meta)/(3+weight)` formula; 0.5/2.0/4.0 bracket
     it by 2x on each side.
   - `W_days` in {30, 60, 90, 180} -- 90 days (the primary) is long
     enough to accumulate dozens of same-bucket resolved episodes even
     split four ways across buckets, short enough to stay adaptive to a
     regime shift within a year; 30/60/180 bracket it.
   - `H_days` fixed at 5 for the main grid (a short multi-day swing
     horizon matching the timescale the latched anchors themselves
     operate on), with a small separate sensitivity check at the primary
     weight/W_days point over `H_days` in {2, 10, 20}.
   - `min_count = 100`: the minimum number of resolved, same-bucket
     observations required in the trailing window before the meta-vote
     is trusted; below that, `meta_vote = 0.5` (neutral -- the default
     before-warmup value named in the task). Fixed a priori, never swept.
   16 (weight x W_days) + 3 (H_days sensitivity) + 1 (weight=0 identity
   check) = 20 configurations. Every one of them is run and reported
   below, win or lose.

4. WHAT WOULD MAKE IT FAIL, named now: (a) the confirming-vote's
   inner-validation Sharpe advantage over `kelly_regime_v4` does not
   clear the project's own +/-0.2 Sharpe noise floor on either market;
   (b) it does clear that floor but fails the placebo-offset
   falsification gate above (i.e. a random time-alignment of the exact
   same win/loss history produces just as much apparent edge); (c) the
   winning region is an isolated peak, not a plateau, when neighbouring
   grid cells are inspected. Any of the three is enough to call this
   NEGATIVE.
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

from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r80_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    V4_BAND,
    V4_HORIZONS,
    anchor_votes,
    confirming_vote_frac,
    placebo_offset_indices,
    truncation_causality_probe,
)
from scripts.experiment import DF, FUTURES, SPOT, ev  # noqa: E402

MIN_COUNT = 100  # pre-registered, never swept -- see banner item 3.


# --------------------------------------------------------------------- signal

def compute_meta_vote(df: pd.DataFrame, W_days: int, H_days: int,
                       min_count: int = MIN_COUNT,
                       shift_indices: np.ndarray | None = None) -> np.ndarray:
    """The causal trailing hit-rate confidence vote, in [0, 1].

    At bar i: bucket the current regime as `anchor_sum[i]` (already
    exactly one of {0,1,2,3}, the sum of the three latched 0/1 anchor
    votes -- the task's own suggested "round anchor_sum/3 to the nearest
    of {0,1/3,2/3,1}" is exact here since anchor_sum is already integral,
    so bucketing is simply that integer). Look at bars j in the trailing
    window [i-H_bars-W_bars, i-H_bars-1] that were in the SAME bucket,
    and compute the fraction of them whose forward H-day log return
    (computed at construction time as `log(close[j+H_bars]) -
    log(close[j])`) was positive.

    Causality: any j used in the window satisfies j+H_bars <= i-1, i.e.
    every input close price the label depends on is strictly before bar
    i's own close -- the fix for R-04's forward-looking label. Before
    `min_count` same-bucket resolved observations exist in the window,
    `meta_vote = 0.5` (neutral), per the pre-registered before-warmup
    default.

    `shift_indices`, when given, circularly reindexes the *hit-label*
    array (`hits = 1[fwd_ret > 0]`) before it is combined with the (real,
    unshifted) bucket sequence -- the falsification-test hook, never used
    by the actual strategy (default None).
    """
    close = df["close"].to_numpy()
    n = len(close)

    votes = anchor_votes(df)
    anchor_sum = sum(v.to_numpy() for v in votes)  # exactly {0,1,2,3}
    bucket = anchor_sum.astype(int)

    H_bars = int(round(H_days * BARS_PER_DAY))
    W_bars = int(round(W_days * BARS_PER_DAY))

    log_close = np.log(close)
    fwd_ret = np.full(n, np.nan)
    if H_bars < n:
        fwd_ret[: n - H_bars] = log_close[H_bars:] - log_close[: n - H_bars]
    hits = np.where(np.isfinite(fwd_ret), (fwd_ret > 0).astype(float), np.nan)

    if shift_indices is not None:
        # Falsification-test hook only: break the true time alignment
        # between "regime bucket in force" and "realized win/loss",
        # keeping each series' own marginal structure (R-79's device).
        hits = hits[shift_indices]

    hits_s = pd.Series(hits)
    valid = (~hits_s.isna()).astype(float)
    hits_filled = hits_s.fillna(0.0)

    meta_vote = np.full(n, 0.5)
    for b in range(4):
        ind = (bucket == b).astype(float)
        ind_valid = ind * valid.to_numpy()
        ind_hit = ind * hits_filled.to_numpy()

        cnt = pd.Series(ind_valid).rolling(W_bars, min_periods=1).sum()
        num = pd.Series(ind_hit).rolling(W_bars, min_periods=1).sum()

        # Shift so the value read at position i reflects the trailing
        # window ending at i-H_bars-1, never i or later.
        cnt_shift = cnt.shift(H_bars + 1).to_numpy()
        num_shift = num.shift(H_bars + 1).to_numpy()

        with np.errstate(invalid="ignore", divide="ignore"):
            rate = num_shift / cnt_shift
        use = (bucket == b) & np.isfinite(cnt_shift) & (cnt_shift >= min_count)
        meta_vote[use] = rate[use]

    return meta_vote


def build_target_primary(df: pd.DataFrame) -> np.ndarray:
    """Target-construction function for the truncation causality probe,
    frozen at the pre-registered primary candidate."""
    return HitRateConfirmKelly(weight=1.0, W_days=90, H_days=5).prepare(
        df.copy())["target"].to_numpy()


# ------------------------------------------------------------------ strategy

class HitRateConfirmKelly(Strategy):
    """kelly_regime_v4 + a causal trailing hit-rate confirming vote (R-80, unregistered).

    Structurally `kelly_regime_v3`/`_v4`'s own prepare()/on_bar() with one
    change: the plain 3-anchor average `frac = anchor_sum/3` is replaced
    by `confirming_vote_frac(anchor_sum, meta_vote, weight)`. `weight=0`
    must recover v4 bit-for-bit (see `run_identity_check` below). Not
    `@register`ed -- this file stays in experiments/ per docs/ROUTINE.md.
    """

    name = "r80_conservative_hitrate_confirm"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, weight: float = 1.0, W_days: int = 90, H_days: int = 5,
                 min_count: int = MIN_COUNT,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 placebo_shift_indices: np.ndarray | None = None) -> None:
        self.weight = weight
        self.W_days = W_days
        self.H_days = H_days
        self.min_count = min_count
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self._placebo_shift_indices = placebo_shift_indices

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = anchor_votes(df, horizons=self.horizons, band=self.band)
        anchor_sum = sum(v.to_numpy() for v in votes)

        meta_vote = compute_meta_vote(df, self.W_days, self.H_days,
                                       min_count=self.min_count,
                                       shift_indices=self._placebo_shift_indices)
        frac = confirming_vote_frac(anchor_sum, meta_vote, self.weight)

        # Identical conditional-volatility-targeting scale to kelly_regime_v3/_v4.
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# --------------------------------------------------------------------- checks

def run_identity_check() -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = DF.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = HitRateConfirmKelly(weight=0.0, W_days=90, H_days=5).prepare(
        df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_causality_probe() -> list[bool]:
    """Truncation probe at three points well past warmup, inner-train only."""
    df = DF.loc[:INNER_TRAIN_END].copy()
    results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = truncation_causality_probe(build_target_primary, df, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


# ----------------------------------------------------------------------- eval

def eval_config(weight: float, W_days: int, H_days: int, tag: str) -> dict:
    """Run one config on both inner splits, both markets. Returns metrics dict."""
    out = {}
    for split_name, kw in (
        ("train", dict(end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = HitRateConfirmKelly(weight=weight, W_days=W_days, H_days=H_days)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_baselines() -> dict:
    out = {}
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (
            ("train", dict(end=INNER_TRAIN_END)),
            ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
        ):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                m = ev(get_strategy(name), market=mkt,
                       tag=f"{name} {split_name} {mkt_name}", **kw)
                out[(name, split_name, mkt_name)] = m
    return out


def run_sweep() -> dict:
    results = {}
    weights = (0.5, 1.0, 2.0, 4.0)
    W_grid = (30, 60, 90, 180)
    for weight in weights:
        for W_days in W_grid:
            tag = f"w{weight} W{W_days} H5"
            results[("main", weight, W_days, 5)] = eval_config(weight, W_days, 5, tag)
    # H_days sensitivity at the pre-registered primary (weight=1.0, W_days=90)
    for H_days in (2, 10, 20):
        tag = f"w1.0 W90 H{H_days}"
        results[("hsens", 1.0, 90, H_days)] = eval_config(1.0, 90, H_days, tag)
    return results


def run_falsification(n_draws: int = 20, block_days: int = 90, seed: int = 8053) -> dict:
    """Placebo-offset falsification gate on the pre-registered primary candidate."""
    warmup = HitRateConfirmKelly.warmup
    lo = int(DF.index.searchsorted(INNER_VAL_START))
    hi = int(DF.index.searchsorted(INNER_VAL_END, side="right"))
    prefix = min(lo, warmup)
    n_frame = hi - (lo - prefix)  # exact length of the frame run_period builds

    v4_val_spot = ev(get_strategy("kelly_regime_v4"), market=SPOT,
                      start=INNER_VAL_START, end=INNER_VAL_END,
                      tag="falsification: v4 val spot (ref)")

    true_strat = HitRateConfirmKelly(weight=1.0, W_days=90, H_days=5)
    true_m = ev(true_strat, market=SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
                tag="falsification: TRUE alignment")
    true_delta = true_m.sharpe - v4_val_spot.sharpe

    shifts = placebo_offset_indices(n_frame, block_days, n_draws, seed)
    null_deltas = []
    for k, shift in enumerate(shifts):
        strat = HitRateConfirmKelly(weight=1.0, W_days=90, H_days=5,
                                     placebo_shift_indices=shift)
        m = ev(strat, market=SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
               tag=f"falsification: placebo {k}")
        null_deltas.append(m.sharpe - v4_val_spot.sharpe)

    null_deltas = np.array(null_deltas)
    p90 = float(np.percentile(null_deltas, 90))
    null_mean = float(np.mean(null_deltas))
    null_std = float(np.std(null_deltas, ddof=1))
    pass_p90 = true_delta > p90
    pass_std = true_delta > null_mean + null_std
    print(f"[falsification] true_delta={true_delta:+.4f}  "
          f"null: mean={null_mean:+.4f} std={null_std:.4f} p90={p90:+.4f}")
    print(f"[falsification] pass_p90={pass_p90}  pass_1std={pass_std}  "
          f"OVERALL={'PASS' if (pass_p90 and pass_std) else 'KILL'}")
    return dict(true_delta=true_delta, null_deltas=null_deltas.tolist(),
                null_mean=null_mean, null_std=null_std, p90=p90,
                passed=bool(pass_p90 and pass_std))


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}\n")

    n_configs = 0

    print("=== identity check ===")
    run_identity_check()

    print("\n=== causality probe ===")
    run_causality_probe()

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    run_baselines()

    print("\n=== sweep (main grid + H-sensitivity) ===")
    sweep_results = run_sweep()
    n_configs += len(sweep_results)

    print("\n=== falsification (placebo-offset, primary candidate) ===")
    fals = run_falsification()

    n_configs += 1  # identity check (weight=0) counted as a configuration evaluated

    print(f"\nTotal configurations evaluated: {n_configs}")
    print(f"Total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
