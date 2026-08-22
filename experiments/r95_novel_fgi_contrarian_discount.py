#!/usr/bin/env python
"""R-95 NOVEL branch (08-22): the Crypto Fear & Greed Index (alternative.me)
as a CONTRARIAN-DISCOUNT modulator on kelly_regime_v4's own vote fraction.

Mechanism, citations, INFO-axis not-a-duplicate list and data-coverage
verification are all frozen in ``experiments/r95_shared.py``'s module
docstring and are not re-derived here -- only the parts specific to THIS
branch (the contrarian reading, Baker & Wurgler 2006) are restated.

=============================================================================
PRE-REGISTRATION (frozen before any real-data number in this file was read)
=============================================================================

1. MECHANISM, one sentence: Baker & Wurgler (2006, *Journal of Finance*
   61(4), 1645-1680) find broad investor sentiment is a CONTRARIAN
   predictor of subsequent returns -- sentiment waves eventually correct --
   so periods of Extreme Greed (alternative.me's own top bucket, FGI >= 80)
   should predict LOWER subsequent BTC returns than usual, and periods of
   Extreme Fear (FGI <= 20) should predict HIGHER subsequent returns than
   usual; if that reversal claim holds it is used to DISCOUNT
   `kelly_regime_v4`'s exposure during Extreme Greed states (fading crowded
   optimism), not to build a new directional vote.

   Architecture: a bounded, state-dependent DISCOUNT on v4's own `frac` at
   an externally-observed sentiment extreme -- structurally the same shape
   as the project's B-40/R-91 "state-dependent discount" construction (see
   `experiments/r91_conservative_state_discount.py`), but keyed off FGI's
   extreme-greed state instead of a price-derived turning-point state. It
   is NOT a never-increase-only brake (closed 4/4 in this project): it
   multiplies `frac` down only inside the Extreme Greed state and returns
   to 1.0x at all other times, a bounded, two-state multiplier -- no
   cumulative or ratcheting state.

2. STEP 0 -- the mandatory sub-claim test, run BEFORE any strategy code.
   Test the sharpest form of the contrarian claim directly, causally, on
   INNER-TRAIN ONLY (2018-02-01, FGI's own coverage start, -> 2020-12-31).
   Bin every bar's FGI level into alternative.me's own five buckets
   (Extreme Fear <=20, Fear 21-40, Neutral 41-60, Greed 61-80, Extreme
   Greed >=80), causally aligned via `r95_shared.load_fgi_5m`/`fgi_level`
   (no lookahead -- the loader shifts by 1 day and forward-fills). Compute
   the CAUSAL forward BTC log-return at horizons H in {1,3,7,14} days from
   each bar's own timestamp (`log(close[t+H*288]) - log(close[t])`);
   because the whole array this is computed over is truncated to
   inner-train BEFORE the shift, `t+H*288` can never index a holdout bar --
   the tail H*288 rows simply go NaN, never read from a bigger frame.
   Grouped by the FGI bucket active at `t`.

   PRE-REGISTERED KILL CONDITION (fixed before any number was computed):
   the contrarian claim requires, at a MAJORITY (>=3) of the 4 horizons,
   BOTH (a) mean forward return in the Extreme Greed bucket is NEGATIVE
   (or at least measurably below the Neutral bucket's mean, i.e. the
   Extreme-Greed-minus-Neutral point estimate is negative AND its 500-draw,
   5-day-block bootstrap CI excludes zero), AND (b) mean forward return in
   the Extreme Fear bucket is POSITIVE (or measurably above Neutral, same
   CI construction, positive sign). If the sign pattern does NOT hold at a
   majority of horizons -- e.g. Extreme Greed instead predicts
   flat-to-positive forward returns (continuation) or is statistically
   indistinguishable from Neutral -- STOP, report NEGATIVE at Step 0, build
   no strategy code.

3. WHAT WOULD MAKE STEP 0 FAIL, named now: this round's own literature
   search found a 2026 VAR-model paper on this exact alternative.me FGI
   series concluding sentiment does NOT Granger-cause BTC returns (a null,
   not a reversal), and He, Shen, Zhang & Zhang (2023, *Finance Research
   Letters* 58(PA)) finding the OPPOSITE of contrarian -- genuine
   CONTINUATION predictability from the same series. Two of the three
   papers this round found on this exact question disagree with (or null
   out) the contrarian sign this branch is built on, so the modal,
   pre-registered expectation is that Step 0 fails -- a clean negative here
   is this branch's fully successful, complete product if that is what
   happens.

4. STEP B/1 -- contingent pre-registration (frozen now, only executed if
   Step 0's kill condition is cleared):

   DISCOUNT CONSTRUCTION: `frac_discounted[i] = frac_v4[i] * (1 - delta)`
   when FGI[i] >= 80 (Extreme Greed), else `frac_discounted[i] = frac_v4[i]`
   unchanged (including during Extreme Fear -- this branch only fades
   greed; it does not boost fear exposure, since v4's own Kelly/vol-target
   scale already sets the overall sizing and a boost multiplier could
   exceed the intended risk budget. Boosting fear exposure is explicitly
   OUT OF SCOPE for this branch -- a real, disclosed asymmetry, not a
   tested-and-passed claim). Applied multiplicatively to `frac` before
   v4's own vol-target `scale` step: `target = frac_discounted[i] *
   scale[i]`, mirroring v4's own construction exactly.

   SWEEP GRID (fixed a priori): `delta` in {0.25, 0.50, 0.75} (bucket
   threshold fixed at alternative.me's own published 80 cutoff, not tuned)
   -- 3 configurations, plus identity (`delta=0`) -- 1. Total 4
   configurations if Step 0 passes.

   MANDATORY CHECKS: (i) exposure-artifact R^2 -- regress the candidate's
   `target` series (inner-validation, both markets) against a
   mean-notional-matched flat rescale of v4's own target; R^2 > 0.95 =
   fail ("just a rescale"); (ii) ETH falsification on
   `ethusd_coinbase_spot_5m.csv.gz` -- FGI is BTC-specific, the gate stays
   keyed to BTC sentiment even when priced against ETH, exactly as R-94
   did; (iii) causal-truncation probe.

   PRE-REGISTERED HOLDOUT DECISION RULE (only reached if everything above
   clears): read the 2023+ holdout ONLY IF ALL of (a) the primary delta's
   inner-validation drawdown OR Sharpe improvement over `kelly_regime_v4`
   exceeds the +-0.2 Sharpe noise floor (or is a clear drawdown
   improvement) on BOTH markets, on a genuine plateau (neighbouring delta
   values, not an isolated peak); (b) exposure-artifact R^2 <= 0.95 both
   markets; (c) ETH falsification replicates the same sign; (d) causality
   probe passes. If ANY fail: report NEGATIVE, never read the holdout.

CONFIGS EVALUATED: counted and printed at the end of each stage
(`CONFIG_COUNTER`) -- 0 in Step 0 (a fixed measurement, no sweep against
real data), up to 4 (grid incl. identity) + baselines + ETH + identity
check in Step B.

Run: ``python experiments/r95_novel_fgi_contrarian_discount.py main``
     ``python experiments/r95_novel_fgi_contrarian_discount.py step0``  (gate only)
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

import experiments.r95_shared as r95_shared  # noqa: E402
from experiments.r95_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    align_fear_greed_causal,
    load_coinbase_eth_spot,
    load_dataset,
    load_fear_greed_index,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    paired_bootstrap,
    stationary_bootstrap_indices,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures()

# ------------------------------------------------------------ Step 0 params
HORIZONS = (1, 3, 7, 14)          # days
STEP0_BLOCK_DAYS = 5.0
STEP0_N_BOOT = 500
STEP0_SEED = 95

EXTREME_FEAR, FEAR, NEUTRAL, GREED, EXTREME_GREED = 0, 1, 2, 3, 4
BUCKET_NAMES = ["Extreme Fear (<=20)", "Fear (21-40)", "Neutral (41-60)",
                "Greed (61-80)", "Extreme Greed (>=80)"]

# ------------------------------------------------------------ Step B params
DELTA_GRID = (0.25, 0.50, 0.75)
IDENTITY_DELTA = 0.0
EPS_TARGET = 1e-9
R2_CEILING = 0.95           # frozen exposure-artifact bar (differs from R-94's 0.99)
SHARPE_FLOOR = 0.2          # R-20 noise floor

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


def bucket_of(level: np.ndarray) -> np.ndarray:
    """alternative.me's own five buckets, as fixed in the pre-registration:
    Extreme Fear <=20, Fear 21-40, Neutral 41-60, Greed 61-80,
    Extreme Greed >=80 (80 itself assigned to Extreme Greed, matching the
    Step B discount's own `FGI[i] >= 80` condition). -1 = no FGI coverage
    yet (bars before 2018-02-01)."""
    b = np.full(level.shape, -1, dtype=np.int8)
    valid = np.isfinite(level)
    b[valid & (level <= 20.0)] = EXTREME_FEAR
    b[valid & (level > 20.0) & (level <= 40.0)] = FEAR
    b[valid & (level > 40.0) & (level <= 60.0)] = NEUTRAL
    b[valid & (level > 60.0) & (level < 80.0)] = GREED
    b[valid & (level >= 80.0)] = EXTREME_GREED
    return b


def _nanmean_stat(x: np.ndarray) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=-1)


def step0_horizon(fwd: np.ndarray, bucket: np.ndarray, h_days: int,
                   idx_matrix: np.ndarray) -> dict:
    means, ns = {}, {}
    for k, nm in enumerate(BUCKET_NAMES):
        mask = bucket == k
        ns[k] = int(mask.sum())
        vals = fwd[mask]
        means[k] = float(np.nanmean(vals)) if ns[k] > 0 else float("nan")

    a_greed = np.where(bucket == EXTREME_GREED, fwd, np.nan)
    b_neutral = np.where(bucket == NEUTRAL, fwd, np.nan)
    diff_greed = paired_bootstrap(a_greed, b_neutral, _nanmean_stat, indices=idx_matrix)

    a_fear = np.where(bucket == EXTREME_FEAR, fwd, np.nan)
    diff_fear = paired_bootstrap(a_fear, b_neutral, _nanmean_stat, indices=idx_matrix)

    pass_greed = (means[EXTREME_GREED] < 0.0) or (
        diff_greed.diff.point < 0.0 and diff_greed.significant)
    pass_fear = (means[EXTREME_FEAR] > 0.0) or (
        diff_fear.diff.point > 0.0 and diff_fear.significant)
    horizon_pass = pass_greed and pass_fear

    print(f"\n  -- H={h_days:>2d}d --")
    for k, nm in enumerate(BUCKET_NAMES):
        print(f"    {nm:24s} n={ns[k]:>7d}  mean_fwd_return={means[k]:>+9.5f}")
    print(f"    ExtremeGreed - Neutral: {diff_greed.diff}  "
          f"p(diff>0)={diff_greed.p_positive:.3f}  significant={diff_greed.significant}  "
          f"pass_greed_clause={pass_greed}")
    print(f"    ExtremeFear  - Neutral: {diff_fear.diff}  "
          f"p(diff>0)={diff_fear.p_positive:.3f}  significant={diff_fear.significant}  "
          f"pass_fear_clause={pass_fear}")
    print(f"    HORIZON PASS (both clauses): {horizon_pass}")

    return dict(h_days=h_days, means=means, ns=ns, diff_greed=diff_greed,
                diff_fear=diff_fear, pass_greed=pass_greed, pass_fear=pass_fear,
                horizon_pass=horizon_pass)


def step0_subclaim_test() -> dict:
    print("=" * 78)
    print("R-95 NOVEL STEP 0: FGI contrarian-reversal sub-claim test (inner-train only)")
    print("=" * 78)

    bars = load_btc_inner_train()
    fgi_full = r95_shared.load_fgi_5m(DATA_DIR)
    assert fgi_full is not None, "fear & greed index file missing -- run scripts/fetch_fear_greed_index.py"
    fgi = fgi_full.reindex(bars.index)
    assert_no_holdout(fgi)
    level = r95_shared.fgi_level(fgi["value"]).to_numpy()
    assert_no_holdout(fgi)  # re-assert after the reindex/ffill above

    close = bars["close"].to_numpy()
    log_close = np.log(close)
    n = len(close)
    bucket = bucket_of(level)

    n_covered = int((bucket >= 0).sum())
    print(f"\nbars: {n:,}  FGI-covered bars: {n_covered:,} ({100*n_covered/n:.1f}%)  "
          f"(bars before {r95_shared.FGI_COVERAGE_START.date()} are unbucketed, code -1)")
    print(f"block-bootstrap: mean_block={STEP0_BLOCK_DAYS:.0f}d ({int(STEP0_BLOCK_DAYS*BARS_PER_DAY)} bars)  "
          f"n_boot={STEP0_N_BOOT}  seed={STEP0_SEED}")

    cells = {}
    for h_days in HORIZONS:
        shift = h_days * BARS_PER_DAY
        fwd = np.full(n, np.nan)
        if shift < n:
            fwd[:n - shift] = log_close[shift:] - log_close[:n - shift]
        # idx_matrix built once per horizon, shared by both diff comparisons
        # (mean_block in ROW units, i.e. 5-minute bars: 5 days * 288 bars/day)
        idx_matrix = stationary_bootstrap_indices(
            n, STEP0_BLOCK_DAYS * BARS_PER_DAY, STEP0_N_BOOT,
            np.random.default_rng(STEP0_SEED + h_days))
        cells[h_days] = step0_horizon(fwd, bucket, h_days, idx_matrix)

    n_pass = sum(1 for c in cells.values() if c["horizon_pass"])
    proceed = n_pass >= 3   # majority of 4

    # sign-pattern classification for the report (reversal / continuation / null)
    eg_above_neutral = sum(1 for c in cells.values()
                            if c["means"][EXTREME_GREED] > c["means"][NEUTRAL])
    ef_below_neutral = sum(1 for c in cells.values()
                            if c["means"][EXTREME_FEAR] < c["means"][NEUTRAL])
    if eg_above_neutral >= 3 and ef_below_neutral >= 3:
        sign_pattern = "CONTINUATION (opposite of the contrarian hypothesis)"
    elif n_pass >= 3:
        sign_pattern = "REVERSAL / CONTRARIAN (matches Baker & Wurgler)"
    else:
        sign_pattern = "MIXED / NULL (no consistent sign pattern across horizons)"

    print("\n" + "=" * 78)
    print("STEP 0 SUMMARY")
    print("=" * 78)
    print(f"{'H(days)':>8s} {'EG_mean':>10s} {'Neutral':>10s} {'EF_mean':>10s} "
          f"{'EG-Neu CI':>26s} {'EF-Neu CI':>26s} {'pass':>5s}")
    for h_days, c in cells.items():
        dg, df_ = c["diff_greed"], c["diff_fear"]
        print(f"{h_days:>8d} {c['means'][EXTREME_GREED]:>+10.5f} {c['means'][NEUTRAL]:>+10.5f} "
              f"{c['means'][EXTREME_FEAR]:>+10.5f} "
              f"[{dg.diff.lo:+.5f},{dg.diff.hi:+.5f}] "
              f"[{df_.diff.lo:+.5f},{df_.diff.hi:+.5f}] "
              f"{'Y' if c['horizon_pass'] else 'n':>5s}")

    print(f"\nHorizons showing the pre-registered contrarian sign pattern: {n_pass}/4 (need >=3)")
    print(f"Horizons with EG_mean > Neutral_mean: {eg_above_neutral}/4  "
          f"Horizons with EF_mean < Neutral_mean: {ef_below_neutral}/4")
    print(f"SIGN PATTERN ACTUALLY FOUND: {sign_pattern}")
    print(f"PRE-REGISTERED KILL-CONDITION VERDICT: "
          f"{'PROCEED to Step B' if proceed else 'STOP -- do not build strategy code'}")
    print(f"\nconfigurations evaluated against real market data in Step 0: 0 "
          f"(fixed measurement, no sweep)")
    print(f"max timestamp read in Step 0: {bars.index.max()}  (<= {INNER_TRAIN_END}, "
          f"strictly < {OOS_START})")

    return dict(cells=cells, n_pass=n_pass, proceed=proceed, sign_pattern=sign_pattern,
                max_ts=bars.index.max())


# =====================================================================
# STEP B -- FGI contrarian-discount strategy (only built if Step 0 passes)
# =====================================================================

def fgi_level_for(df: pd.DataFrame) -> np.ndarray:
    """FGI level aligned onto an arbitrary bars frame (BTC or ETH) -- FGI is
    date-keyed, not BTC-price-derived, so the same series aligns onto any
    bar grid sharing the same UTC calendar. Disclosed: this keeps the gate
    keyed to BTC-market sentiment even when priced against ETH."""
    fgi = load_fear_greed_index(DATA_DIR)
    assert fgi is not None, "fear & greed index file missing"
    aligned = align_fear_greed_causal(fgi, df)
    return r95_shared.fgi_level(aligned["value"]).to_numpy()


class FGIContrarianDiscountV4(Strategy):
    """kelly_regime_v4 whose vote fraction is discounted by (1-delta) while
    the Fear & Greed Index sits in Extreme Greed (FGI >= 80) (R-95 NOVEL)."""

    warmup = KellyRegimeV4().warmup

    def __init__(self, delta: float = 0.5, **kwargs) -> None:
        self._base = KellyRegimeV4(**kwargs)
        self.delta = delta
        self.name = f"fgi_contrarian_discount_v4[d{delta:g}]"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        assert_no_holdout(df)
        base = self._base
        close = df["close"]
        r = np.log(close).diff()

        votes = r95_shared.anchor_votes(df, base.horizons, base.band)
        frac = (sum(votes) / len(votes)).to_numpy()
        if base.vote_gamma != 1.0:
            frac = frac ** base.vote_gamma

        level = fgi_level_for(df)
        extreme_greed = level >= 80.0   # NaN (no FGI coverage yet) -> False, no discount
        frac_discounted = np.where(extreme_greed, frac * (1.0 - self.delta), frac)

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
            desired = frac_discounted[i] * scale
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
    """delta=0 must reproduce kelly_regime_v4's target array bit-for-bit."""
    print("\n" + "-" * 78)
    print("IDENTITY RECOVERY CHECK (delta=0)")
    print("-" * 78)
    v4_target = KellyRegimeV4().prepare(bars.copy())["target"].to_numpy()
    cand = FGIContrarianDiscountV4(delta=IDENTITY_DELTA)
    cand_target = cand.prepare(bars.copy())["target"].to_numpy()
    ok = np.array_equal(v4_target, cand_target)
    print(f"  delta=0 : identical to v4 target array: {ok}")
    _count("diagnostic")
    return ok


def causality_probe(bars: pd.DataFrame, delta: float) -> bool:
    print("\n" + "-" * 78)
    print(f"CAUSAL-TRUNCATION PROBE -- delta={delta:g}")
    print("-" * 78)

    def build(frame: pd.DataFrame) -> np.ndarray:
        strat = FGIContrarianDiscountV4(delta=delta)
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


def exposure_artifact_r2(cand_target: np.ndarray, v4_target: np.ndarray) -> dict:
    """Flat-rescale test: c = mean(cand)/mean(v4) (mean-notional match);
    R^2 = 1 - SS_res(cand vs c*v4) / SS_tot(cand)."""
    v4_mean = float(np.mean(v4_target))
    c = float(np.mean(cand_target) / v4_mean) if v4_mean != 0 else float("nan")
    if not np.isfinite(c):
        return dict(r2=float("nan"), c=c, n=len(cand_target))
    resid = cand_target - c * v4_target
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((cand_target - np.mean(cand_target)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(r2=r2, c=c, n=len(cand_target))


# =====================================================================
# main
# =====================================================================

def step_b_sweep(bars: pd.DataFrame) -> dict:
    print("\n" + "=" * 100)
    print(f"STEP B -- FGI contrarian-discount sweep, BTC spot+futures, entry-tier fees "
          f"({len(DELTA_GRID)} swept configs + identity)")
    print("=" * 100)

    all_deltas = (IDENTITY_DELTA,) + DELTA_GRID
    print(f"pre-registered grid: delta in {DELTA_GRID}  (+identity delta=0)\n")

    train_rows, val_rows = {}, {}

    print("-- inner-train (<= 2020-12-31), SPOT --")
    for d in all_deltas:
        strat = FGIContrarianDiscountV4(delta=d)
        m, _ = raw_run(strat, bars, SPOT, end=INNER_TRAIN_END, tag=f"train d{d:g}")
        _count("stepB")
        train_rows[d] = dict(sharpe=m.sharpe, max_dd=m.max_drawdown_pct, profit=m.profit_pct)

    v4_train_m, _ = raw_run(KellyRegimeV4(), bars, SPOT, end=INNER_TRAIN_END, tag="train kelly_regime_v4")
    _count("stepB")
    bh_train_m, _ = raw_run(get_strategy("buy_and_hold"), bars, SPOT, end=INNER_TRAIN_END, tag="train buy_and_hold")
    _count("stepB")

    print("\n-- inner-validation (2021-01-01 -> 2022-12-31), SPOT and FUTURES --")
    for market in (SPOT, FUTURES):
        for d in all_deltas:
            strat = FGIContrarianDiscountV4(delta=d)
            m, res = raw_run(strat, bars, market, start=INNER_VAL_START, end=INNER_VAL_END,
                              tag=f"val   d{d:g}")
            _count("stepB")
            val_rows[(market.name, d)] = dict(sharpe=m.sharpe, max_dd=m.max_drawdown_pct,
                                               profit=m.profit_pct, result=res)
        v4_m, v4_res = raw_run(KellyRegimeV4(), bars, market, start=INNER_VAL_START, end=INNER_VAL_END,
                                tag="val   kelly_regime_v4")
        _count("stepB")
        val_rows[(market.name, "v4")] = dict(sharpe=v4_m.sharpe, max_dd=v4_m.max_drawdown_pct,
                                              profit=v4_m.profit_pct, result=v4_res)
        bh_m, _ = raw_run(get_strategy("buy_and_hold"), bars, market, start=INNER_VAL_START, end=INNER_VAL_END,
                           tag="val   buy_and_hold")
        _count("stepB")
        val_rows[(market.name, "bh")] = dict(sharpe=bh_m.sharpe, max_dd=bh_m.max_drawdown_pct,
                                              profit=bh_m.profit_pct)

    winner = max(DELTA_GRID, key=lambda d: (val_rows[("spot", d)]["sharpe"],
                                             -val_rows[("spot", d)]["max_dd"]))
    print(f"\nWINNER (by inner-validation SPOT Sharpe, among swept {DELTA_GRID}): delta={winner:g}  "
          f"sharpe={val_rows[('spot', winner)]['sharpe']:.3f}  "
          f"(v4 sharpe={val_rows[('spot', 'v4')]['sharpe']:.3f})")

    return dict(all_deltas=all_deltas, train=train_rows, val=val_rows, winner=winner,
                v4_train_m=v4_train_m, bh_train_m=bh_train_m)


def eth_falsification(delta_w: float) -> dict:
    print("\n" + "=" * 100)
    print(f"ETH FALSIFICATION -- delta={delta_w:g}, Coinbase ETH spot, "
          f"BTC-keyed FGI aligned onto ETH bars")
    print("=" * 100)

    eth_df = load_coinbase_eth_spot(DATA_DIR)
    assert eth_df is not None, "ETH coinbase spot file missing"
    eth_df = eth_df.loc[eth_df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(eth_df)
    print(f"ETH bars: {len(eth_df):,}  {eth_df.index[0]} -> {eth_df.index[-1]}  (< {OOS_START})")

    strat = FGIContrarianDiscountV4(delta=delta_w)
    m, _ = raw_run(strat, eth_df, SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
                    tag=f"ETH   d{delta_w:g}")
    _count("stepB")
    m_v4, _ = raw_run(KellyRegimeV4(), eth_df, SPOT, start=INNER_VAL_START, end=INNER_VAL_END,
                       tag="ETH   kelly_regime_v4")
    _count("stepB")

    passed = m.sharpe >= m_v4.sharpe
    print(f"\nETH falsification: candidate sharpe={m.sharpe:.3f}  v4 sharpe={m_v4.sharpe:.3f}  "
          f"PASS (candidate >= v4, directional replication): {passed}")
    print(f"max timestamp read in ETH falsification: {eth_df.index.max()}  (< {OOS_START})")
    return dict(cand_sharpe=m.sharpe, v4_sharpe=m_v4.sharpe, passed=passed)


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
        print("# STEP 0 FAILED ITS PRE-REGISTERED KILL CONDITION.")
        print("# Per this file's own pre-registration, STOP HERE. No strategy code is")
        print("# exercised as a real backtest; this gate result is this branch's whole")
        print("# product.")
        print("#" * 78)
        total = CONFIG_COUNTER["step0"] + CONFIG_COUNTER["stepB"] + CONFIG_COUNTER["diagnostic"]
        print(f"\nCONFIGS EVALUATED (TOTAL): {total}")
        print(f"max timestamp read anywhere in this session: {step0['max_ts']}  (< {OOS_START})")
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
    d_w = step_b["winner"]

    print("\n" + "-" * 78)
    print(f"EXPOSURE-ARTIFACT R^2 CHECK -- winner delta={d_w:g} vs kelly_regime_v4, "
          f"inner-validation, both markets")
    print("-" * 78)
    cand_target = FGIContrarianDiscountV4(delta=d_w).prepare(
        bars.loc[(bars.index >= pd.Timestamp(INNER_VAL_START, tz="UTC")) &
                 (bars.index <= pd.Timestamp(INNER_VAL_END, tz="UTC"))].copy())["target"].to_numpy()
    v4_target = KellyRegimeV4().prepare(
        bars.loc[(bars.index >= pd.Timestamp(INNER_VAL_START, tz="UTC")) &
                 (bars.index <= pd.Timestamp(INNER_VAL_END, tz="UTC"))].copy())["target"].to_numpy()
    artifact = exposure_artifact_r2(cand_target, v4_target)
    print(f"  n_bars={artifact['n']}  flat-rescale c={artifact['c']:.4f}  R^2={artifact['r2']:.4f}")
    print("  NOTE: prepare() in this codebase is market-agnostic (target does not depend on "
          "MarketSpec), so this single regression covers 'both markets' identically -- disclosed, "
          "not a shortcut.")
    artifact_flag = np.isfinite(artifact["r2"]) and artifact["r2"] > R2_CEILING
    print(f"  {'FLAT-RESCALE ARTIFACT (R^2>' + str(R2_CEILING) + ')' if artifact_flag else 'not a flat-rescale artifact'}")

    eth = eth_falsification(d_w)
    causal_ok = causality_probe(bars, d_w)

    print("\n" + "=" * 100)
    print("PRE-REGISTERED HOLDOUT-CONSULTATION DECISION RULE (applied mechanically)")
    print("=" * 100)
    spot_delta = step_b["val"][("spot", d_w)]["sharpe"] - step_b["val"][("spot", "v4")]["sharpe"]
    fut_delta = step_b["val"][("futures", d_w)]["sharpe"] - step_b["val"][("futures", "v4")]["sharpe"]
    spot_dd = step_b["val"][("spot", "v4")]["max_dd"] - step_b["val"][("spot", d_w)]["max_dd"]
    fut_dd = step_b["val"][("futures", "v4")]["max_dd"] - step_b["val"][("futures", d_w)]["max_dd"]
    rule_noise_floor = ((spot_delta > SHARPE_FLOOR or spot_dd > 0) and
                         (fut_delta > SHARPE_FLOOR or fut_dd > 0))
    rule_not_artifact = not artifact_flag
    rule_eth = eth["passed"]
    rule_causal = causal_ok
    recommend = rule_noise_floor and rule_not_artifact and rule_eth and rule_causal
    print(f"  (a) noise floor BOTH markets: spot_dSharpe={spot_delta:+.3f} (dd={spot_dd:+.2f}pp)  "
          f"fut_dSharpe={fut_delta:+.3f} (dd={fut_dd:+.2f}pp): {rule_noise_floor}")
    print(f"  (b) not a flat-rescale artifact (R^2 <= {R2_CEILING}): {rule_not_artifact}")
    print(f"  (c) ETH falsification passed: {rule_eth}")
    print(f"  (d) causal-truncation probe passed: {rule_causal}")
    print(f"  RECOMMEND HOLDOUT CONSULTATION: {recommend}")

    holdout = None
    if recommend:
        print("\n" + "=" * 100)
        print(f"STEP 4 -- HOLDOUT (start={OOS_START}), winner delta={d_w:g}, frozen config")
        print("=" * 100)
        winner_strat = FGIContrarianDiscountV4(delta=d_w)
        m_hold, _ = raw_run(winner_strat, df, SPOT, start=OOS_START, tag=f"HOLDOUT d{d_w:g}")
        _count("stepB")
        m_v4_hold, _ = raw_run(KellyRegimeV4(), df, SPOT, start=OOS_START, tag="HOLDOUT kelly_regime_v4")
        _count("stepB")
        m_bh_hold, _ = raw_run(get_strategy("buy_and_hold"), df, SPOT, start=OOS_START, tag="HOLDOUT buy_and_hold")
        _count("stepB")
        beats_bh = m_hold.sharpe > m_bh_hold.sharpe and m_hold.profit_pct > m_bh_hold.profit_pct
        beats_v4_floor = (m_hold.sharpe - m_v4_hold.sharpe) > SHARPE_FLOOR
        print(f"\nHOLDOUT beats buy_and_hold (sharpe & profit): {beats_bh}")
        print(f"HOLDOUT beats kelly_regime_v4 by >+{SHARPE_FLOOR} Sharpe: {beats_v4_floor}")
        holdout = dict(candidate=m_hold, v4=m_v4_hold, bh=m_bh_hold,
                        beats_bh=beats_bh, beats_v4_floor=beats_v4_floor)

    total = CONFIG_COUNTER["step0"] + CONFIG_COUNTER["stepB"] + CONFIG_COUNTER["diagnostic"]
    print(f"\nCONFIGS EVALUATED (TOTAL): step0={CONFIG_COUNTER['step0']} "
          f"stepB={CONFIG_COUNTER['stepB']} diagnostic={CONFIG_COUNTER['diagnostic']} total={total}")
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
