#!/usr/bin/env python
"""R-94 NOVEL branch (08-22): Wikipedia "Bitcoin" pageview attention as an
ATTENTION-EXHAUSTION CONFIRMATION-DELAY modulator on kelly_regime_v4's own
anchor-vote latch.

Mechanism, citations, INFO-axis not-a-duplicate list and data-coverage
verification are all frozen in ``experiments/r94_shared.py``'s module
docstring and are not re-derived here -- only the parts specific to THIS
branch (the contrarian/exhaustion reading, Kristoufek 2013) are restated.

=============================================================================
PRE-REGISTRATION (frozen before any real-data number in this file was read)
=============================================================================

1. MECHANISM, one sentence: Kristoufek (2013, *Scientific Reports* 3:3415)
   finds BTC's largest positive price deviations from trend (bubble-like
   run-ups) are accompanied/preceded by EXTREME pageview/search attention
   spikes that historically mark EXHAUSTION rather than confirmation, so a
   fresh v4 anchor-vote flip INTO a long stance that coincides with an
   extreme attention spike (top decile / z >= 2.5 of `r94_shared.attention_z`)
   should require additional confirmation persistence -- a longer required
   latch-hold before the flip is accepted -- before v4's `frac` moves; when
   attention is NOT extreme, this file must behave IDENTICALLY to
   `kelly_regime_v4` (zero new parameters active in the null case).

   NOT the never-increase-only bounded-brake family this ledger has closed
   4-for-4 (R-34, R-41, R-53, R-73): this construction never multiplies
   exposure down and has no cumulative "can only shrink" state. It DELAYS
   an entry commit and later fully un-delays it once the entry's own
   persistence condition is met (or the underlying price condition breaks,
   in which case the candidate entry is abandoned, not remembered as a
   debt) -- structurally the same "temporal delay, not a permanent cap"
   distinction R-88's own taker-flow execution-delay branch relied on, here
   applied to the VOTE's own latch rather than to `target`'s execution.
   ONLY 0->1 (entering long) transitions of each anchor vote are ever
   delayed; 0<-1 (de-risking) transitions are always immediate, exactly as
   in unmodified v4 -- delaying an exit would risk reproducing the brake
   family's failure by keeping exposure high longer than v4 itself wants,
   which is the one thing this construction must not do.

2. STEP 0 -- PRE-REGISTERED SUB-CLAIM TEST, run BEFORE any latch/strategy
   code is built (same "test the sharpest sub-claim first" convention
   B-40/R-91 used). INNER-TRAIN ONLY (2017-01-01 -> 2020-12-31).

   Daily granularity (BTC close resampled to 1D; `attention_z`'s own
   5-minute series is constant within a UTC day by construction -- shift
   by 1 day, ffill -- so resampling to the last 5-minute value of each day
   loses no information and removes the 5-minute-grid overlap that would
   otherwise make forward N-day windows almost entirely non-independent).
   For N in {5, 10, 20} calendar days and threshold definition in
   {"decile" (in-sample top-decile of `attention_z` over inner-train),
   "z25" (fixed z >= 2.5)}: is the forward N-day return on days where
   `attention_z` clears the threshold significantly MORE NEGATIVE than the
   unconditional forward N-day return over the same period?

   TOOL: `tradebot.inference.paired_bootstrap` (this project's standard
   block-bootstrap machinery -- Politis & Romano 1994 stationary
   bootstrap), `mean_block=30` (days), `n_boot=2000`, `seed=93`, applied to
   `a` = forward-return series with non-extreme days masked to NaN, `b` =
   the full forward-return series, `stat` = nanmean -- so
   `stat(a)-stat(b)` is exactly (conditional mean - unconditional mean) on
   every resample, pairing the SAME resampled days into both legs.

   PRIMARY CELL, named now, before any number was computed: N=10,
   threshold="z25" -- the middle horizon (long enough to let an exhaustion
   effect show, short enough to still be "the sharpest" available cell),
   and the fixed z>=2.5 definition because it is the more literal reading
   of Kristoufek's own "extreme spike" language (a relative top-decile day
   is not necessarily an extreme day if the whole window is calm).

   PRE-REGISTERED STOP RULE, fixed now: PROCEED to Step B only if BOTH
   (a) the PRIMARY cell's point estimate is negative AND its 95% paired
   bootstrap CI excludes zero, AND (b) at least 4 of the remaining 5
   secondary cells (N in {5,20} x both thresholds, plus N=10/"decile")
   also show a negative point estimate (sign agreement only, not required
   to be individually significant) -- the same ">=4 of 6" bar R-84's own
   Step-A gate used, with one cell designated mandatory rather than left
   to a majority vote alone. Failing either clause: STOP, report this
   file's Step 0 result as the round's whole product, do not build any
   strategy/latch code. The bar is not relaxed after seeing the numbers.

3. STEP B (only if Step 0 passes): a plain `Strategy` subclass,
   `AttentionExhaustionDelayV4`, that reuses `KellyRegimeV4`'s own
   constructor parameters (horizons, band, vol/regime-scale machinery)
   unchanged and replaces only the 3 anchor votes' entry-transition timing
   with `attention_modulated_votes` below (causal: bar i's committed vote
   depends only on rows <= i and the running (latched, pending_since)
   state built from strictly earlier bars).

   SWEEP GRID (fixed now, 3x3=9 configs): `Z_THRESH_GRID = (2.0, 2.5, 3.0)`
   (entry-trigger extremity) x `PERSIST_DAYS_GRID = (1, 3, 7)` (required
   consecutive days the raw crossing condition must keep holding before
   the flip commits). Every config run on inner-train (screening) then
   inner-validation (selection, by inner-validation Sharpe, ties broken by
   lower max drawdown) against `kelly_regime_v4` and `buy_and_hold` on BTC
   spot, entry-tier fees.

   MANDATORY CHECKS on the inner-validation winner:
   (1) IDENTITY RECOVERY: `persist_days=0` (or, independently,
       `z_thresh=+inf`) must reproduce `kelly_regime_v4`'s `target` array
       bit-for-bit -- checked directly on the array, not just on
       backtest metrics.
   (2) EXPOSURE-ARTIFACT R^2: candidate's daily return series regressed
       (OLS, intercept + scale) against `kelly_regime_v4`'s own daily
       return series on the same period; R^2 > ~0.99 means "only changed
       exposure level," not a genuine mechanism (this project's repeated
       flat-rescale-artifact failure mode, R-33/R-34/R-73/R-91 among
       others).
   (3) ETH FALSIFICATION: the identical winning config, applied to
       Coinbase ETH spot bars (`ethusd_coinbase_spot_5m.csv.gz`), using
       the SAME BTC "Bitcoin"-article attention_z aligned onto ETH's own
       bar grid (there is no separate Wikipedia "Ethereum" signal built
       this round -- this tests whether BTC-attention-driven exhaustion
       timing transfers to a second asset's price series, the same
       directional-replication bar R-88's own ETH falsification used:
       PASS = candidate Sharpe >= plain `kelly_regime_v4`'s own Sharpe on
       the identical ETH window).
   (4) NOISE FLOOR: inner-validation Sharpe delta vs `kelly_regime_v4`
       must exceed +-0.2 (R-20), or show a clear matched-risk drawdown
       improvement, to be worth a holdout consultation.

4. WHAT WOULD MAKE STEP B FAIL, named now: (i) the exhaustion effect is
   real in Step 0 but too small/rare (attention-extreme entry triggers may
   simply not coincide often enough with v4's own anchor-vote flips inside
   a 4-8 year window to move the backtest at all -- a no-op by rarity,
   not by construction, mirroring R-84's own named failure mode (i));
   (ii) the exposure-artifact check fires because delaying entries mostly
   just shifts v4's own exposure curve later in time without changing its
   shape -- a "phase-shifted rescale," a variant of the standing artifact
   this project has hit repeatedly; (iii) the delay helps BTC specifically
   because BTC's own historical bubble episodes (Dec-2017, Nov-2021) are
   overfit-adjacent single events inside a short inner-train window, which
   the ETH falsification is designed to catch.

CONFIGS EVALUATED: counted and printed at the end of each stage
(`CONFIG_COUNTER`) -- 0 in Step 0 (a fixed measurement, no sweep against
real data), up to 9 (grid) + baselines + ETH + identity check in Step B.

Run: ``python experiments/r94_novel_attention_exhaustion_delay.py``
     ``python experiments/r94_novel_attention_exhaustion_delay.py step0``  (gate only)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    align_wikipedia_causal,
    load_coinbase_eth_spot,
    load_dataset,
    load_wikipedia_pageviews,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import daily_returns, paired_bootstrap  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# ENVIRONMENT NOTE (not part of the frozen pre-registration, disclosed here
# because it changes how this file is written): `experiments/r94_shared.py`
# was overwritten partway through this session by an unrelated, concurrently
# running R-94 effort (a Grossman-Zhou drawdown-sizing round, already closed
# NEGATIVE and merged to `main` -- see `git log -- experiments/r94_shared.py`
# and commit `cf2aa5b`), which collided with this branch's own round number
# and clobbered the shared Wikipedia-pageview prep file this branch's
# pre-registration depends on. Per this round's own instruction ("do not
# edit experiments/r94_shared.py"), that file is left untouched here. The
# constants and helper functions below are duplicated VERBATIM from the
# original Wikipedia-pageview `r94_shared.py` as it read at dispatch time
# (recovered read-only via `git show 19d04aa:experiments/r94_shared.py`,
# the prep commit for THIS branch's own round, never edited by this file) --
# not re-derived, not re-reasoned, byte-identical logic to what this
# branch's pre-registration was written against. This is reported to the
# operator as an environment anomaly, not silently absorbed.
# ---------------------------------------------------------------------------

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"


def load_pageviews_5m(data_dir) -> pd.DataFrame | None:
    """Daily pageviews causally aligned onto the 5-minute BTC bar grid, or
    None if the fetch script has not been run. Callers must still truncate
    to their own step's cutoff themselves -- this loader does not enforce
    the holdout boundary. Verbatim duplicate, see note above."""
    from tradebot.data import load_ohlcv_csv
    pv = load_wikipedia_pageviews(data_dir)
    if pv is None:
        return None
    bars = load_ohlcv_csv(f"{data_dir}/btcusd_spot_5m.csv.gz")
    return align_wikipedia_causal(pv, bars)


def attention_z(views_5m: pd.Series, window_days: int = 20) -> pd.Series:
    """Causal log-pageview z-score against its own trailing `window_days`
    mean/std. Verbatim duplicate, see note above."""
    log_v = np.log(views_5m.replace(0.0, np.nan).ffill())
    w = int(window_days * BARS_PER_DAY)
    mean = log_v.rolling(w, min_periods=w // 4).mean()
    std = log_v.rolling(w, min_periods=w // 4).std()
    return (log_v - mean) / std.replace(0.0, np.nan)
SPOT = MarketSpec.spot()

# ------------------------------------------------------------ Step 0 params
STEP0_N_GRID = (5, 10, 20)
STEP0_THRESH_DEFS = ("decile", "z25")
STEP0_PRIMARY = (10, "z25")
STEP0_MEAN_BLOCK = 30.0
STEP0_N_BOOT = 2000
STEP0_SEED = 93

# ------------------------------------------------------------ Step B params
Z_THRESH_GRID = (2.0, 2.5, 3.0)
PERSIST_DAYS_GRID = (1, 3, 7)
EPS_TARGET = 1e-9

CONFIG_COUNTER = {"step0": 0, "stepB": 0, "diagnostic": 0}


def _count(kind: str, k: int = 1) -> None:
    CONFIG_COUNTER[kind] += k


# ---------------------------------------------------------------- holdout guard
def assert_no_holdout(obj) -> None:
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
# STEP 0 -- pre-registered sub-claim test
# =====================================================================

def load_btc_inner_train() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}) inner-train: {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(<= {INNER_TRAIN_END})", file=sys.stderr)
    return df


def _nanmean_stat(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=-1)


def step0_cell(z_daily: pd.Series, fwd: pd.Series, n_days: int, thresh_def: str) -> dict:
    valid = z_daily.notna() & fwd.notna()
    zv = z_daily[valid].to_numpy()
    fv = fwd[valid].to_numpy()
    n = len(fv)
    if thresh_def == "decile":
        thresh = float(np.quantile(zv, 0.90))
        mask = zv >= thresh
    else:
        thresh = 2.5
        mask = zv >= thresh

    a = np.where(mask, fv, np.nan)
    b = fv.copy()

    result = paired_bootstrap(a, b, _nanmean_stat, mean_block=STEP0_MEAN_BLOCK,
                               n_boot=STEP0_N_BOOT, seed=STEP0_SEED)
    n_extreme = int(mask.sum())
    negative_sign = result.diff.point < 0.0
    significant = result.significant
    passed = negative_sign and significant

    print(f"  N={n_days:>2d}d  thresh={thresh_def:6s} (z>={thresh:.3f})  "
          f"n_days_valid={n:4d}  n_extreme={n_extreme:3d} ({100*n_extreme/n:.1f}%)  "
          f"cond_mean={result.stat_a:+.4f}  uncond_mean={result.stat_b:+.4f}  "
          f"diff={result.diff}  p(diff>0)={result.p_positive:.3f}  "
          f"NEGATIVE_SIGN={negative_sign}  SIGNIFICANT(CI excl. 0)={significant}  "
          f"PASS={passed}")

    return dict(n_days=n_days, thresh_def=thresh_def, thresh=thresh, n_valid=n,
                n_extreme=n_extreme, cond_mean=result.stat_a, uncond_mean=result.stat_b,
                diff_point=result.diff.point, diff_lo=result.diff.lo, diff_hi=result.diff.hi,
                p_positive=result.p_positive, negative_sign=negative_sign,
                significant=significant, passed=passed)


def step0_subclaim_test() -> dict:
    print("=" * 78)
    print("R-94 NOVEL STEP 0: attention-exhaustion sub-claim test (inner-train only)")
    print("=" * 78)

    bars = load_btc_inner_train()
    pv_full = load_pageviews_5m(DATA_DIR)
    assert pv_full is not None, "wikipedia pageviews file missing"
    pv = pv_full.reindex(bars.index)
    assert_no_holdout(pv)

    z_5m = attention_z(pv["views"])
    assert_no_holdout(z_5m.to_frame())

    close_daily = bars["close"].resample("1D").last().dropna()
    z_daily = z_5m.resample("1D").last().reindex(close_daily.index)

    print(f"\ndaily observations: {len(close_daily)}  "
          f"({close_daily.index[0].date()} -> {close_daily.index[-1].date()})")
    print(f"block-bootstrap: mean_block={STEP0_MEAN_BLOCK:.0f}d  n_boot={STEP0_N_BOOT}  "
          f"seed={STEP0_SEED}  (tradebot.inference.paired_bootstrap)")
    print(f"PRIMARY cell (pre-registered): N={STEP0_PRIMARY[0]}d, thresh={STEP0_PRIMARY[1]!r}\n")

    cells = {}
    for n_days in STEP0_N_GRID:
        fwd = close_daily.shift(-n_days) / close_daily - 1.0
        for thresh_def in STEP0_THRESH_DEFS:
            cells[(n_days, thresh_def)] = step0_cell(z_daily, fwd, n_days, thresh_def)

    primary = cells[STEP0_PRIMARY]
    secondary = [c for k, c in cells.items() if k != STEP0_PRIMARY]
    n_secondary_negative = sum(1 for c in secondary if c["negative_sign"])

    proceed = primary["passed"] and (n_secondary_negative >= 4)

    print("\n" + "=" * 78)
    print("STEP 0 SUMMARY")
    print("=" * 78)
    print(f"{'N':>4s} {'thresh':>8s} {'cond_mean':>11s} {'uncond_mean':>12s} "
          f"{'diff':>22s} {'sign':>6s} {'sig':>5s}")
    for (n_days, thresh_def), c in cells.items():
        tag = " <-- PRIMARY" if (n_days, thresh_def) == STEP0_PRIMARY else ""
        print(f"{n_days:>4d} {thresh_def:>8s} {c['cond_mean']:>+11.4f} {c['uncond_mean']:>+12.4f} "
              f"[{c['diff_lo']:+.4f}, {c['diff_hi']:+.4f}] {'neg' if c['negative_sign'] else 'pos':>6s} "
              f"{'Y' if c['significant'] else 'n':>5s}{tag}")

    print(f"\nPRIMARY cell (N={STEP0_PRIMARY[0]}d, {STEP0_PRIMARY[1]}) PASS: {primary['passed']}")
    print(f"Secondary cells with negative sign: {n_secondary_negative}/5 (need >=4)")
    print(f"PRE-REGISTERED STOP RULE VERDICT: "
          f"{'PROCEED to Step B' if proceed else 'STOP -- do not build strategy code'}")
    print(f"\nconfigurations evaluated against real market data in Step 0: 0 "
          f"(fixed measurement, no sweep)")
    print(f"max timestamp read in Step 0: {bars.index.max()}  (<= {INNER_TRAIN_END}, "
          f"strictly < {OOS_START})")

    return dict(cells=cells, primary=primary, n_secondary_negative=n_secondary_negative,
                proceed=proceed)


# =====================================================================
# STEP B -- attention-exhaustion confirmation-delay strategy
# =====================================================================

def attention_z_for(df: pd.DataFrame) -> np.ndarray:
    """attention_z aligned onto an arbitrary bars frame (BTC or ETH)."""
    pv = load_wikipedia_pageviews(DATA_DIR)
    assert pv is not None, "wikipedia pageviews file missing"
    aligned = align_wikipedia_causal(pv, df)
    return attention_z(aligned["views"]).to_numpy()


def attention_modulated_votes(df: pd.DataFrame, z: np.ndarray,
                               horizons: tuple[int, ...], band: float,
                               z_thresh: float, persist_bars: int) -> list[pd.Series]:
    """v4's 3 anchor votes, with 0->1 (entering long) transitions delayed by
    `persist_bars` consecutive bars of continued raw crossing whenever the
    trigger bar's `z` clears `z_thresh`. 0<-1 (de-risking) transitions and
    the dead-zone latch/ffill are untouched -- causal, row i depends only
    on rows <= i and running state built from strictly earlier bars.

    Identity recovery: `persist_bars<=0` OR `z_thresh=+inf` reproduces
    `r94_shared.anchor_votes` bit-for-bit (verified in `identity_check`).
    """
    close = df["close"].to_numpy()
    n = len(close)
    votes = []
    for days in horizons:
        anchor = df["close"].rolling(int(days * BARS_PER_DAY)).mean().to_numpy()
        raw_hi = close > anchor * (1.0 + band)
        raw_lo = close < anchor * (1.0 - band)

        vote = np.empty(n)
        latched = 0.0
        pending_since = -1
        for i in range(n):
            if raw_lo[i]:
                latched = 0.0
                pending_since = -1
            elif raw_hi[i]:
                if latched == 1.0:
                    pending_since = -1
                elif pending_since == -1:
                    zi = z[i]
                    extreme = np.isfinite(zi) and zi >= z_thresh
                    if not extreme or persist_bars <= 0:
                        latched = 1.0
                    else:
                        pending_since = i
                else:
                    if i - pending_since >= persist_bars:
                        latched = 1.0
                        pending_since = -1
            else:
                pending_since = -1
            vote[i] = latched
        votes.append(pd.Series(vote, index=df.index))
    return votes


class AttentionExhaustionDelayV4(Strategy):
    """kelly_regime_v4 whose anchor-vote LONG entries are held pending extra
    confirmation when they trigger during an extreme Wikipedia-attention
    spike (R-94 NOVEL)."""

    warmup = KellyRegimeV4().warmup

    def __init__(self, z_thresh: float = 2.5, persist_days: int = 3, **kwargs) -> None:
        self._base = KellyRegimeV4(**kwargs)
        self.z_thresh = z_thresh
        self.persist_days = persist_days
        self.name = f"attn_exhaustion_delay_v4[z{z_thresh:g}_p{persist_days}]"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        assert_no_holdout(df)
        base = self._base
        close = df["close"]
        r = np.log(close).diff()

        z = attention_z_for(df)
        persist_bars = int(self.persist_days * BARS_PER_DAY)
        votes = attention_modulated_votes(df, z, base.horizons, base.band,
                                           self.z_thresh, persist_bars)
        frac = (sum(votes) / len(votes)).to_numpy()
        if base.vote_gamma != 1.0:
            frac = frac ** base.vote_gamma

        vol = (r.ewm(span=base.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=base.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(base.target_vol / vol, base.max_leverage)
            steady = np.minimum(base.target_vol / slow, base.max_leverage)
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
                    state = 1 if x > base.high_in else (-1 if x < base.low_in else 0)
                elif state == 1 and x < base.high_out:
                    state = 0
                elif state == -1 and x > base.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > base.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > EPS_TARGET:
            ctx.order_notional(t)


# ------------------------------------------------------------ diagnostics

def identity_check(bars: pd.DataFrame) -> bool:
    """persist_days=0 must reproduce kelly_regime_v4's target array
    bit-for-bit; independently, so must z_thresh=+inf."""
    print("\n" + "-" * 78)
    print("IDENTITY RECOVERY CHECK (persist_days=0, and z_thresh=+inf)")
    print("-" * 78)
    v4_target = KellyRegimeV4().prepare(bars.copy())["target"].to_numpy()

    a = AttentionExhaustionDelayV4(z_thresh=2.5, persist_days=0)
    a_target = a.prepare(bars.copy())["target"].to_numpy()
    ok_a = np.array_equal(v4_target, a_target)
    print(f"  persist_days=0 : identical to v4 target array: {ok_a}")

    b = AttentionExhaustionDelayV4(z_thresh=float("inf"), persist_days=7)
    b_target = b.prepare(bars.copy())["target"].to_numpy()
    ok_b = np.array_equal(v4_target, b_target)
    print(f"  z_thresh=+inf  : identical to v4 target array: {ok_b}")

    ok = ok_a and ok_b
    print(f"  IDENTITY RECOVERY: {'PASS' if ok else 'FAIL'}")
    _count("diagnostic")
    return ok


def causality_probe(bars: pd.DataFrame, z_thresh: float, persist_days: int) -> bool:
    print("\n" + "-" * 78)
    print(f"CAUSAL-TRUNCATION PROBE -- z{z_thresh:g}_p{persist_days}")
    print("-" * 78)

    def build(frame: pd.DataFrame) -> np.ndarray:
        strat = AttentionExhaustionDelayV4(z_thresh=z_thresh, persist_days=persist_days)
        return strat.prepare(frame.copy())["target"].to_numpy()

    check_at = len(bars) - 40_000
    shorter_by = 20_000
    full = build(bars)
    short = build(bars.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    ok = bool(np.isclose(a, b, equal_nan=True))
    print(f"  check_at bar={check_at}  ({bars.index[check_at]})  shorter_by={shorter_by}")
    print(f"  target[check_at] full-series={a:.6f}  truncated={b:.6f}  CAUSAL: {ok}")
    _count("diagnostic")
    return ok


# ------------------------------------------------------------ eval helpers

def raw_run(strategy: Strategy, df: pd.DataFrame, market: MarketSpec,
            start=None, end=None, tag: str = "") -> tuple[object, object]:
    """Like scripts.experiment.ev, but also returns the raw BacktestResult
    so the equity curve is available for the exposure-artifact regression."""
    t0 = time.time()
    if start is None and end is None:
        result = run_backtest(strategy, df, market, 1_000.0, data_label="")
    else:
        result = run_period(strategy, df, start, end, market=market,
                             start_balance=1_000.0, data_label="")
    m = compute_metrics(result)
    print(f"{tag or strategy.name:32s} {market.name:11s} "
          f"final=${m.final_balance:>13,.0f} ({m.profit_pct:>+9.1f}%) "
          f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} {'LIQUIDATED' if m.liquidated else ''} "
          f"[{time.time() - t0:.0f}s]")
    return m, result


def exposure_artifact_r2(candidate_result, v4_result) -> dict:
    cand_ret = daily_returns(candidate_result.equity)
    v4_ret = daily_returns(v4_result.equity)
    idx = cand_ret.index.intersection(v4_ret.index)
    a = cand_ret.reindex(idx).to_numpy()
    b = v4_ret.reindex(idx).to_numpy()
    if len(a) < 3 or np.std(b) == 0:
        return dict(r2=float("nan"), scale_c=float("nan"), n=len(a))
    corr = float(np.corrcoef(a, b)[0, 1])
    r2 = corr ** 2
    scale_c = float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    return dict(r2=r2, corr=corr, scale_c=scale_c, n=len(a))


# =====================================================================
# main
# =====================================================================

def step_b_sweep(bars: pd.DataFrame) -> dict:
    print("\n" + "=" * 100)
    print(f"STEP B -- attention-exhaustion delay sweep, BTC spot, entry-tier fees "
          f"({len(Z_THRESH_GRID)}x{len(PERSIST_DAYS_GRID)}={len(Z_THRESH_GRID)*len(PERSIST_DAYS_GRID)} configs)")
    print("=" * 100)

    configs = [(z, p) for z in Z_THRESH_GRID for p in PERSIST_DAYS_GRID]
    print(f"pre-registered grid: Z_THRESH in {Z_THRESH_GRID}, PERSIST_DAYS in {PERSIST_DAYS_GRID}\n")

    train_rows, val_rows = {}, {}

    print("-- inner-train (<= 2020-12-31) --")
    for z, p in configs:
        strat = AttentionExhaustionDelayV4(z_thresh=z, persist_days=p)
        m, _ = raw_run(strat, bars, SPOT, end=INNER_TRAIN_END, tag=f"train z{z:g}_p{p}")
        _count("stepB")
        train_rows[(z, p)] = dict(sharpe=m.sharpe, max_dd=m.max_drawdown_pct, profit=m.profit_pct)

    v4_train_m, _ = raw_run(KellyRegimeV4(), bars, SPOT, end=INNER_TRAIN_END, tag="train kelly_regime_v4")
    _count("stepB")
    bh_train_m, _ = raw_run(get_strategy("buy_and_hold"), bars, SPOT, end=INNER_TRAIN_END, tag="train buy_and_hold")
    _count("stepB")

    print("\n-- inner-validation (2021-01-01 -> 2022-12-31) --")
    for z, p in configs:
        strat = AttentionExhaustionDelayV4(z_thresh=z, persist_days=p)
        m, res = raw_run(strat, bars, SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
                          tag=f"val   z{z:g}_p{p}")
        _count("stepB")
        val_rows[(z, p)] = dict(sharpe=m.sharpe, max_dd=m.max_drawdown_pct, profit=m.profit_pct,
                                 result=res)

    v4_val_m, v4_val_res = raw_run(KellyRegimeV4(), bars, SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
                                    tag="val   kelly_regime_v4")
    _count("stepB")
    bh_val_m, _ = raw_run(get_strategy("buy_and_hold"), bars, SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
                           tag="val   buy_and_hold")
    _count("stepB")

    winner_key = max(val_rows, key=lambda k: (val_rows[k]["sharpe"], -val_rows[k]["max_dd"]))
    z_w, p_w = winner_key
    print(f"\nWINNER (by inner-validation Sharpe): z{z_w:g}_p{p_w}  "
          f"sharpe={val_rows[winner_key]['sharpe']:.3f}  (v4 sharpe={v4_val_m.sharpe:.3f})")

    sharpe_delta = val_rows[winner_key]["sharpe"] - v4_val_m.sharpe
    dd_delta = v4_val_m.max_drawdown_pct - val_rows[winner_key]["max_dd"]
    print(f"  inner-val sharpe_delta vs v4 = {sharpe_delta:+.3f}   "
          f"max_dd_delta vs v4 (positive=better) = {dd_delta:+.2f}pp")

    return dict(configs=configs, train=train_rows, val=val_rows,
                winner=(z_w, p_w), winner_result=val_rows[winner_key]["result"],
                v4_val_m=v4_val_m, v4_val_result=v4_val_res,
                v4_train_m=v4_train_m, bh_train_m=bh_train_m, bh_val_m=bh_val_m,
                sharpe_delta=sharpe_delta, dd_delta=dd_delta)


def eth_falsification(z_w: float, p_w: int) -> dict:
    print("\n" + "=" * 100)
    print(f"ETH FALSIFICATION -- z{z_w:g}_p{p_w}, Coinbase ETH spot, "
          f"BTC-wikipedia attention_z aligned onto ETH bars")
    print("=" * 100)

    eth_df = load_coinbase_eth_spot(DATA_DIR)
    assert eth_df is not None, "ETH coinbase spot file missing"
    eth_df = eth_df.loc[eth_df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(eth_df)
    print(f"ETH bars: {len(eth_df):,}  {eth_df.index[0]} -> {eth_df.index[-1]}  (< {OOS_START})")

    strat = AttentionExhaustionDelayV4(z_thresh=z_w, persist_days=p_w)
    m, _ = raw_run(strat, eth_df, SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
                    tag=f"ETH   z{z_w:g}_p{p_w}")
    _count("stepB")
    m_v4, _ = raw_run(KellyRegimeV4(), eth_df, SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
                       tag="ETH   kelly_regime_v4")
    _count("stepB")

    passed = m.sharpe >= m_v4.sharpe
    print(f"\nETH falsification: winner sharpe={m.sharpe:.3f}  v4 sharpe={m_v4.sharpe:.3f}  "
          f"PASS (winner >= v4, directional replication): {passed}")
    print(f"max timestamp read in ETH falsification: {eth_df.index.max()}  (< {OOS_START})")
    return dict(winner_sharpe=m.sharpe, v4_sharpe=m_v4.sharpe, passed=passed)


def main() -> None:
    t0 = time.time()
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    step0 = step0_subclaim_test()

    if choice == "step0":
        print(f"\nCONFIGS EVALUATED: step0={CONFIG_COUNTER['step0']} "
              f"stepB={CONFIG_COUNTER['stepB']} diagnostic={CONFIG_COUNTER['diagnostic']}")
        print(f"[{time.time()-t0:.0f}s]")
        return

    if not step0["proceed"]:
        print("\n" + "#" * 78)
        print("# STEP 0 FAILED ITS PRE-REGISTERED STOP RULE.")
        print("# Per this file's own pre-registration, STOP HERE. No latch/strategy code")
        print("# is exercised as a real backtest; this gate result is this branch's whole")
        print("# product.")
        print("#" * 78)
        total = CONFIG_COUNTER["step0"] + CONFIG_COUNTER["stepB"] + CONFIG_COUNTER["diagnostic"]
        print(f"\nCONFIGS EVALUATED (TOTAL): {total}")
        print(f"[{time.time()-t0:.0f}s]")
        return

    # ---- Step 0 passed: build and evaluate Step B ----
    df, label = load_dataset(DATA_DIR, "spot")
    bars = df.loc[df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(bars)
    print(f"\nBTC ({label}) full pre-holdout: {len(bars):,} bars  "
          f"{bars.index[0]} -> {bars.index[-1]}  (< {OOS_START})")

    id_ok = identity_check(bars)
    assert id_ok, "identity recovery FAILED -- stop, do not trust this branch's Step B numbers"

    step_b = step_b_sweep(bars)
    z_w, p_w = step_b["winner"]

    print("\n" + "-" * 78)
    print(f"EXPOSURE-ARTIFACT R^2 CHECK -- winner z{z_w:g}_p{p_w} vs kelly_regime_v4, inner-validation")
    print("-" * 78)
    artifact = exposure_artifact_r2(step_b["winner_result"], step_b["v4_val_result"])
    print(f"  n_days={artifact['n']}  corr={artifact.get('corr', float('nan')):.4f}  "
          f"R^2={artifact['r2']:.4f}  rescale_c={artifact['scale_c']:.4f}")
    artifact_flag = np.isfinite(artifact["r2"]) and artifact["r2"] > 0.99
    print(f"  {'FLAT-RESCALE ARTIFACT (R^2>0.99)' if artifact_flag else 'not a flat-rescale artifact'}")

    eth = eth_falsification(z_w, p_w)
    causal_ok = causality_probe(bars, z_w, p_w)

    print("\n" + "=" * 100)
    print("PRE-REGISTERED HOLDOUT-CONSULTATION DECISION RULE (applied mechanically)")
    print("=" * 100)
    rule_noise_floor = (step_b["sharpe_delta"] > 0.2) or (step_b["dd_delta"] > 0)
    rule_not_artifact = not artifact_flag
    rule_eth = eth["passed"]
    rule_causal = causal_ok
    recommend = rule_noise_floor and rule_not_artifact and rule_eth and rule_causal
    print(f"  (a) noise floor: sharpe_delta({step_b['sharpe_delta']:+.3f}) > +0.2  OR  "
          f"drawdown improvement (dd_delta={step_b['dd_delta']:+.2f}pp): {rule_noise_floor}")
    print(f"  (b) not a flat-rescale artifact (R^2 <= 0.99): {rule_not_artifact}")
    print(f"  (c) ETH falsification passed: {rule_eth}")
    print(f"  (d) causal-truncation probe passed: {rule_causal}")
    print(f"  RECOMMEND HOLDOUT CONSULTATION: {recommend}")
    print("  NOTE: this file has NO authority to consult the holdout on its own; if the")
    print("  rule above is True, that recommendation is reported to the operator/report,")
    print("  and the holdout is read once, mechanically, per docs/ROUTINE.md step 4.")

    holdout = None
    if recommend:
        print("\n" + "=" * 100)
        print(f"STEP 4 -- HOLDOUT (start={OOS_START}), winner z{z_w:g}_p{p_w}, frozen config")
        print("=" * 100)
        winner_strat = AttentionExhaustionDelayV4(z_thresh=z_w, persist_days=p_w)
        m_hold, _ = raw_run(winner_strat, df, SPOT, start=OOS_START, tag=f"HOLDOUT z{z_w:g}_p{p_w}")
        _count("stepB")
        m_v4_hold, _ = raw_run(KellyRegimeV4(), df, SPOT, start=OOS_START, tag="HOLDOUT kelly_regime_v4")
        _count("stepB")
        m_bh_hold, _ = raw_run(get_strategy("buy_and_hold"), df, SPOT, start=OOS_START, tag="HOLDOUT buy_and_hold")
        _count("stepB")
        beats_bh = m_hold.sharpe > m_bh_hold.sharpe and m_hold.profit_pct > m_bh_hold.profit_pct
        beats_v4_floor = (m_hold.sharpe - m_v4_hold.sharpe) > 0.2
        print(f"\nHOLDOUT beats buy_and_hold (sharpe & profit): {beats_bh}")
        print(f"HOLDOUT beats kelly_regime_v4 by >+0.2 Sharpe: {beats_v4_floor}")
        holdout = dict(candidate=m_hold, v4=m_v4_hold, bh=m_bh_hold,
                        beats_bh=beats_bh, beats_v4_floor=beats_v4_floor)

    total = CONFIG_COUNTER["step0"] + CONFIG_COUNTER["stepB"] + CONFIG_COUNTER["diagnostic"]
    print(f"\nCONFIGS EVALUATED (TOTAL): step0={CONFIG_COUNTER['step0']} "
          f"stepB={CONFIG_COUNTER['stepB']} diagnostic={CONFIG_COUNTER['diagnostic']} total={total}")
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
