#!/usr/bin/env python
"""R-99 NOVEL branch: a Generalized-Hurst-Exponent-derived REBALANCE BAND WIDTH
for ``kelly_regime_v4`` -- widen the fixed 10% deadband when the market is
locally anti-persistent/noisy (H<0.5) and narrow it when locally persistent/
trending (H>0.5), on the theory that a rebalance triggered in a rough/noisy
local regime is more likely reacting to noise than to real drift.

Full citation trail, "not a duplicate of" case, and shared estimator are in
``experiments/r99_shared.py`` (frozen, read-only, written by the operator).
This file restates the construction only to the extent needed to pre-register
it precisely; see ``r99_shared.py``'s module docstring for the full case.

This is a COST-axis construction (fewer, better-timed rebalances). It never
touches ``kelly_regime_v4``'s directional vote or its ``scale`` (conditional
volatility target) -- only WHEN the strategy is willing to move toward its
already-computed desired exposure. Per R-62 (four independent confirmations
that ``scale`` carries none of v4's signature), this branch does not go near
``scale``.

=====================================================================
PRE-REGISTRATION (frozen before any real-data GHE/z-score/deadband number in
this file was computed -- docs/ROUTINE.md step 2). Anything below later
contradicted by what actually happened is stated in the RESULTS section at
the bottom of this file (appended after running), never edited back into
this banner.
=====================================================================

1. MECHANISM (one sentence): a locally anti-persistent market (Generalized
   Hurst Exponent H<0.5, Barabasi & Vicsek 1991 / Di Matteo 2007, computed
   causally on BTC's own daily log close via ``r99_shared.rolling_ghe_signal``)
   has a worse short-horizon signal-to-noise ratio, so a deadband-triggered
   rebalance inside it is more likely reacting to noise than to real drift and
   should require a bigger move to fire (wider band); the reverse (H>0.5,
   locally persistent/trending) means less delay is warranted (narrower band).

2. CONSTRUCTION, exactly:

   ``z = ghe_signal_zscore(rolling_ghe_signal(daily_log_prices(bars),
   PRIMARY_FIT_WINDOW_DAYS=180))``, aligned onto the 5-minute bar index via
   ``align_daily_causal`` -- the IDENTICAL estimator, IDENTICAL primary
   window (180 days), IDENTICAL smoothing (``DETECTION_WINDOW_DAYS=90``) and
   IDENTICAL baseline (``BASELINE_WINDOW_DAYS=730``) the conservative
   branch's alarm uses. No second, undisclosed estimator variant is
   introduced anywhere in this file.

   ``multiplier(z) = clip(1.0 - K*z, FLOOR, CEIL)``, ``FLOOR=0.5``,
   ``CEIL=2.0`` -- fixed a priori, symmetric (band can shrink to half or
   grow to double the base), chosen before any number was seen, never
   retuned. ``z>0`` (more persistent than trailing baseline) -> multiplier
   <1 -> band narrows. ``z<0`` (more anti-persistent) -> multiplier >1 ->
   band widens. Where ``z`` is not finite (no baseline yet, e.g. the first
   ~2 years while ``BASELINE_WINDOW_DAYS=730`` warms up), multiplier = 1.0
   exactly, i.e. the strategy behaves identically to unmodified
   ``kelly_regime_v4`` until the GHE baseline exists.

   ``K in {0.15, 0.30, 0.50}`` -- an a-priori 3-point grid. NOT searched
   beyond this; no other K value is evaluated anywhere in this file.

   ``BASE_DEADBAND = 0.10`` (v4's own fixed value).
   ``effective_deadband[i] = BASE_DEADBAND * multiplier(z[i])``, so the band
   ranges over ``[0.05, 0.20]`` depending on K and the realized z (bounded by
   FLOOR/CEIL regardless of K, since the clip is on the multiplier, not on
   the raw ``1-K*z`` value).

3. IMPLEMENTATION: a new ``Strategy`` subclass, ``KellyRegimeV4GheBand``,
   defined below. It is NOT decorated with ``@register`` (an experiment,
   not a registered strategy per docs/ROUTINE.md's registration rules) and
   does not edit any file under ``src/tradebot/strategies/``. Its
   ``prepare()`` is a byte-for-byte copy of ``KellyRegimeV3.prepare()`` (the
   method ``kelly_regime_v4`` actually runs, since v4 only overrides
   ``__init__``'s default horizons) with exactly one line changed: the
   hysteresis loop's ``if abs(desired - pos) > self.deadband`` becomes
   ``if abs(desired - pos) > effective_deadband[i]``. The 20/40/80 anchor
   vote, the v3-style conditional volatility targeting (state machine +
   ``full``/``steady`` scale), and the 2x leverage cap are copied verbatim
   and are otherwise untouched. ``scale`` itself is never modified by this
   branch, per R-62/R-33's standing warnings quoted in the dispatch.

4. STEP-0 GATE, pre-registered NOW, before any real-data number, on
   INNER-TRAIN ONLY (``end="2020-12-31"``):

   a. NON-DEGENERACY CHECK: report mean/std/min/max of
      ``effective_deadband`` over inner-train, and the % of bars where it
      differs from the fixed 0.10 base by more than 10% relatively, for
      each of the 3 K values. DECISION RULE: if ALL THREE K values produce
      an effective deadband within +/-5% of 0.10 for MORE THAN 95% of bars,
      the construction has collapsed to a near-constant band (the same
      failure mode 23 prior SIZE-axis rounds hit) and this branch STOPS
      here, NEGATIVE, without running any backtest.

   b. EFFECT CHECK (only if (a) does not stop the branch): compare each K
      variant against unmodified ``kelly_regime_v4`` on inner-train,
      futures 5x (turnover-relevant quantities -- ``num_trades`` -- do not
      depend on market choice, since ``target[i]`` is computed identically
      regardless of market; Sharpe/drawdown/final balance are read off the
      futures-5x run, this project's standard risk-bearing market).
      DECISION RULE, using this project's own standing ±0.2 Sharpe noise
      floor (R-20), not an invented threshold: if turnover does not drop by
      at least 10% relative for AT LEAST ONE K in the grid, OR if Sharpe
      drops by more than 0.2 versus baseline for EVERY K, the branch's own
      motivating claim ("cuts turnover without degrading Sharpe below the
      noise floor") has failed and the branch STOPS here, NEGATIVE.

5. IF STEP-0 PASSES for >=1 K: evaluate the surviving K variant(s) on
   inner-validation (``start="2021-01-01", end="2022-12-31"``) against
   ``kelly_regime_v4`` baseline, same metrics, on BOTH BTC futures 5x and
   spot. Verdict follows docs/ROUTINE.md step 4's bar (beats v4; the
   improvement exceeds the ±0.2 Sharpe noise floor or is a genuine
   drawdown/tail improvement; plateau across the K grid, not a single-K
   spike) but the TRUE holdout (``>=2023-01-01``) is never touched in this
   branch -- "out-of-sample" here means inner-validation only, exactly as
   the conservative branch's own Step-B.

6. FALSIFICATION -- what would make this fail, named now: (i) the
   multiplier collapses to ~1.0 almost everywhere because 180-day GHE
   rarely deviates >1-2 baseline-sigma from its own 730-day trailing mean
   on this series -- the same collapse-to-constant failure 23 SIZE-axis
   attempts have hit, now on the COST axis; (ii) even if the band is
   genuinely time-varying, it fails to concentrate on bars that were
   actually about to be reverted (i.e., turnover does not fall); (iii) the
   band does fall but at the cost of missing real trend moves, i.e. Sharpe
   degrades below the ±0.2 floor at every K in the grid.

7. CAUSALITY: every series is guarded with ``r99_shared.assert_no_holdout``
   before being used, and no bar dated ``>= 2023-01-01`` is read anywhere in
   this file. The causal truncation probe
   (``r99_shared.truncation_causality_probe``) is run on the aligned z-score
   series and on the K=0.30 effective-deadband series at 3 ``check_at``
   values, reported PASS/FAIL.

RESULTS are appended as a module-level comment block at the very bottom of
this file after running -- not edited back into the banner above.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r99_shared as shared  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# --------------------------------------------------------------- constants

BASE_DEADBAND = 0.10
K_GRID = (0.15, 0.30, 0.50)
FLOOR = 0.5
CEIL = 2.0
NOISE_FLOOR_SHARPE = 0.2          # this project's standing bar (R-20)
TURNOVER_DROP_THRESH = 0.10       # 10% relative, pre-registered in the dispatch
DEGEN_BAND_PCT = 0.05             # +/-5% of base counts as "near-constant"
DEGEN_FRACTION = 0.95             # >95% of bars within that band => degenerate

INNER_TRAIN_END = shared.INNER_TRAIN_END
INNER_VAL_START = shared.INNER_VAL_START
INNER_VAL_END = shared.INNER_VAL_END
OOS_START = shared.OOS_START


# ----------------------------------------------------------- GHE -> band


def ghe_multiplier(z: pd.Series, k: float, floor: float = FLOOR, ceil: float = CEIL) -> pd.Series:
    """``clip(1 - K*z, FLOOR, CEIL)``, with NaN z -> multiplier 1.0 (no
    baseline yet -> behave exactly like unmodified v4's fixed band)."""
    raw = 1.0 - k * z
    m = raw.clip(lower=floor, upper=ceil)
    m = m.where(np.isfinite(z), 1.0)
    return m


def z_score_series(bars: pd.DataFrame, fit_window_days: int = shared.PRIMARY_FIT_WINDOW_DAYS) -> pd.Series:
    """The IDENTICAL estimator/window the conservative branch's alarm uses,
    aligned onto ``bars``' 5-minute index. Causal by construction (see
    ``r99_shared.rolling_ghe_signal`` / ``align_daily_causal``)."""
    daily_prices = shared.daily_log_prices(bars)
    ghe = shared.rolling_ghe_signal(daily_prices, fit_window_days)
    z = shared.ghe_signal_zscore(ghe)
    aligned = shared.align_daily_causal(z, bars)
    shared.assert_no_holdout(aligned.dropna())
    return aligned


def effective_deadband_series(bars: pd.DataFrame, k: float,
                               fit_window_days: int = shared.PRIMARY_FIT_WINDOW_DAYS,
                               base: float = BASE_DEADBAND) -> pd.Series:
    z = z_score_series(bars, fit_window_days)
    mult = ghe_multiplier(z, k)
    eff = base * mult
    shared.assert_no_holdout(eff.dropna())
    return eff


# --------------------------------------------------------------- strategy


class KellyRegimeV4GheBand(KellyRegimeV4):
    """``kelly_regime_v4`` with the fixed 10% rebalance deadband replaced by
    a GHE-derived one: ``effective_deadband[i] = 0.10 * clip(1 - K*z[i],
    0.5, 2.0)``. Everything else -- the 20/40/80 anchor vote, v3's
    conditional volatility targeting, the 2x cap -- is a byte-for-byte copy
    of ``KellyRegimeV3.prepare()``, changing only the hysteresis loop's
    comparison threshold. Not registered; experiment-only, per
    docs/ROUTINE.md.
    """

    name = "kelly_regime_v4_ghe_band"

    def __init__(self, ghe_k: float = 0.15,
                 ghe_fit_window_days: int = shared.PRIMARY_FIT_WINDOW_DAYS,
                 ghe_floor: float = FLOOR, ghe_cap: float = CEIL,
                 base_deadband: float = BASE_DEADBAND, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ghe_k = ghe_k
        self.ghe_fit_window_days = ghe_fit_window_days
        self.ghe_floor = ghe_floor
        self.ghe_cap = ghe_cap
        self.base_deadband = base_deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # ---- byte-for-byte copy of KellyRegimeV3.prepare()'s vote/vol/state
        # machinery, down to variable names, up to the one changed line in
        # the hysteresis loop (marked below).
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * shared.BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=shared.BARS_PER_DAY).std()
               * np.sqrt(shared.BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * shared.BARS_PER_DAY,
                                   min_periods=shared.BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        # ---- the one addition: a GHE-derived per-bar effective deadband,
        # built from THIS SAME frame (whatever warmup/period slice prepare()
        # was called with), never from a longer series.
        eff_deadband = effective_deadband_series(
            df, self.ghe_k, self.ghe_fit_window_days, self.base_deadband
        ).to_numpy()
        df["ghe_effective_deadband"] = eff_deadband  # exposed for diagnostics

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
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
            # ---- CHANGED LINE (the only behavioural change vs v4): fixed
            # self.deadband -> per-bar GHE-derived effective_deadband.
            if abs(desired - pos) > eff_deadband[i]:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# --------------------------------------------------------------- harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

_CONFIGS_EVALUATED = 0


def ev(strategy, market=None, balance: float = 1_000.0, tag: str = "",
       start=None, end=None, count: bool = True):
    """Same pattern as ``scripts/experiment.py``'s ``ev()``, copied so this
    file has no import-time side effects on that module."""
    global _CONFIGS_EVALUATED
    market = SPOT if market is None else market
    t0 = time.time()
    result = run_period(strategy, DF, start, end, market=market, start_balance=balance,
                         data_label=LABEL)
    m = compute_metrics(result)
    if count:
        _CONFIGS_EVALUATED += 1
    print(f"{tag or strategy.name:28s} {market.name:11s} "
          f"final=${m.final_balance:>13,.0f} ({m.profit_pct:>+9.1f}%) "
          f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} [{time.time() - t0:.0f}s]")
    return m


def build_frame(end=None, start=None) -> pd.DataFrame:
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    return DF.iloc[lo:hi]


# --------------------------------------------------------------- Step 0

def step0_nondegeneracy() -> dict:
    print("\n=== STEP-0a: non-degeneracy of effective_deadband (inner-train) ===")
    frame = build_frame(end=INNER_TRAIN_END)
    shared.assert_no_holdout(frame)
    results = {}
    for k in K_GRID:
        eff = effective_deadband_series(frame, k).dropna()
        rel_diff = (eff - BASE_DEADBAND).abs() / BASE_DEADBAND
        within_5pct = float((rel_diff <= DEGEN_BAND_PCT).mean())
        differs_gt_10pct = float((rel_diff > 0.10).mean())
        stats = dict(
            k=k, mean=float(eff.mean()), std=float(eff.std()),
            min=float(eff.min()), max=float(eff.max()),
            pct_within_5pct_of_base=within_5pct,
            pct_differs_gt10pct_from_base=differs_gt_10pct,
            n=int(len(eff)),
        )
        results[k] = stats
        print(f"  K={k:.2f}  mean={stats['mean']:.4f} std={stats['std']:.4f} "
              f"min={stats['min']:.4f} max={stats['max']:.4f}  "
              f"within+/-5%={within_5pct*100:5.1f}%  differs>10%={differs_gt_10pct*100:5.1f}%  "
              f"(n={stats['n']})")
    all_degenerate = all(results[k]["pct_within_5pct_of_base"] > DEGEN_FRACTION for k in K_GRID)
    print(f"  DEGENERACY VERDICT: {'DEGENERATE (STOP)' if all_degenerate else 'non-degenerate, proceed'}")
    return {"per_k": results, "all_degenerate": all_degenerate}


def step0_effect_check() -> dict:
    print("\n=== STEP-0b: effect check vs kelly_regime_v4 (inner-train, futures 5x) ===")
    baseline = ev(KellyRegimeV4(), market=FUTURES, tag="baseline v4", end=INNER_TRAIN_END)
    variants = {}
    for k in K_GRID:
        m = ev(KellyRegimeV4GheBand(ghe_k=k), market=FUTURES, tag=f"ghe_band K={k}",
               end=INNER_TRAIN_END)
        turnover_drop = (baseline.num_trades - m.num_trades) / baseline.num_trades if baseline.num_trades else 0.0
        sharpe_diff = m.sharpe - baseline.sharpe
        variants[k] = dict(metrics=m, turnover_drop=turnover_drop, sharpe_diff=sharpe_diff)
        print(f"    K={k:.2f}: turnover_drop={turnover_drop*100:+6.1f}%  "
              f"sharpe_diff={sharpe_diff:+.3f}  DD={m.max_drawdown_pct:.1f}%  "
              f"final=${m.final_balance:,.0f}")
    any_turnover_drop = any(v["turnover_drop"] >= TURNOVER_DROP_THRESH for v in variants.values())
    all_sharpe_bad = all(v["sharpe_diff"] <= -NOISE_FLOOR_SHARPE for v in variants.values())
    failed = (not any_turnover_drop) or all_sharpe_bad
    reason = None
    if not any_turnover_drop:
        reason = "turnover-did-not-drop (no K reached -10% relative trades vs baseline)"
    elif all_sharpe_bad:
        reason = "sharpe-dropped-too-much (every K lost more than 0.2 Sharpe vs baseline)"
    print(f"  EFFECT-CHECK VERDICT: {'FAIL -> STOP (' + str(reason) + ')' if failed else 'pass, proceed'}")
    return {"baseline": baseline, "variants": variants, "failed": failed, "reason": reason,
            "any_turnover_drop": any_turnover_drop, "all_sharpe_bad": all_sharpe_bad}


# --------------------------------------------------------- inner-validation

def inner_validation(surviving_ks) -> dict:
    print("\n=== STEP 3: inner-validation (2021-01-01 -> 2022-12-31), futures 5x + spot ===")
    out = {}
    for market, mkt_obj in (("futures5x", FUTURES), ("spot", SPOT)):
        base = ev(KellyRegimeV4(), market=mkt_obj, tag="baseline v4",
                  start=INNER_VAL_START, end=INNER_VAL_END)
        out[market] = {"baseline": base, "variants": {}}
        for k in surviving_ks:
            m = ev(KellyRegimeV4GheBand(ghe_k=k), market=mkt_obj, tag=f"ghe_band K={k}",
                   start=INNER_VAL_START, end=INNER_VAL_END)
            turnover_drop = (base.num_trades - m.num_trades) / base.num_trades if base.num_trades else 0.0
            sharpe_diff = m.sharpe - base.sharpe
            dd_improve = base.max_drawdown_pct - m.max_drawdown_pct
            out[market]["variants"][k] = dict(
                metrics=m, turnover_drop=turnover_drop, sharpe_diff=sharpe_diff,
                dd_improve_pct=dd_improve,
            )
            print(f"    [{market}] K={k}: turnover_drop={turnover_drop*100:+6.1f}%  "
                  f"sharpe_diff={sharpe_diff:+.3f}  DD_improve={dd_improve:+.1f}pp  "
                  f"final=${m.final_balance:,.0f} (base ${base.final_balance:,.0f})")
    return out


# ------------------------------------------------------- truncation probe

def truncation_probe() -> dict:
    print("\n=== Causal truncation probe (z-score and K=0.30 effective_deadband) ===")
    probe_frame = build_frame(end=INNER_VAL_END)  # < OOS_START throughout
    shared.assert_no_holdout(probe_frame)

    def z_fn(d):
        return z_score_series(d).to_numpy()

    def deadband_fn(d):
        return effective_deadband_series(d, 0.30).to_numpy()

    check_ats = [100_000, 300_000, 550_000]
    results = {}
    for name, fn in (("z_score", z_fn), ("effective_deadband_K0.30", deadband_fn)):
        passes = []
        for c in check_ats:
            ok = shared.truncation_causality_probe(fn, probe_frame, check_at=c)
            passes.append(ok)
            print(f"    {name} check_at={c}: {'PASS' if ok else 'FAIL'}")
        results[name] = passes
    all_pass = all(all(v) for v in results.values())
    print(f"  TRUNCATION PROBE VERDICT: {'PASS' if all_pass else 'FAIL'}")
    return {"per_series": results, "check_ats": check_ats, "all_pass": all_pass}


# --------------------------------------------------------------- main

def main() -> None:
    t_start = time.time()
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  (data: {LABEL})")
    print(f"K grid: {K_GRID}  FLOOR={FLOOR}  CEIL={CEIL}  BASE_DEADBAND={BASE_DEADBAND}")

    # The truncation-causality probe is run regardless of the Step-0 verdict
    # (per the dispatch: "guard every series ... and report PASS/FAIL" is
    # not conditioned on the branch surviving Step-0).
    probe = truncation_probe()

    degeneracy = step0_nondegeneracy()
    if degeneracy["all_degenerate"]:
        print("\n### VERDICT: NEGATIVE -- Step-0a (non-degeneracy) failed. ###")
        print("The GHE-derived multiplier collapses to a near-constant band "
              "(within +/-5% of 0.10 for >95% of bars) at every K in the grid. "
              "Same collapse-to-constant failure mode as 23 prior SIZE-axis rounds, "
              "now observed on the COST axis. Stopping before any backtest.")
        print(f"\nTotal configs evaluated: {_CONFIGS_EVALUATED}")
        print(f"Truncation probe: {'PASS' if probe['all_pass'] else 'FAIL'}")
        _report_wallclock(t_start)
        return

    effect = step0_effect_check()
    if effect["failed"]:
        print(f"\n### VERDICT: NEGATIVE -- Step-0b (effect check) failed: {effect['reason']}. ###")
        print(f"\nTotal configs evaluated: {_CONFIGS_EVALUATED}")
        print(f"Truncation probe: {'PASS' if probe['all_pass'] else 'FAIL'}")
        _report_wallclock(t_start)
        return

    surviving_ks = [k for k, v in effect["variants"].items()
                    if v["turnover_drop"] >= TURNOVER_DROP_THRESH and v["sharpe_diff"] > -NOISE_FLOOR_SHARPE]
    if not surviving_ks:
        # Defensive: the aggregate gate in step0_effect_check can pass (>=1 K
        # dropped turnover enough, not every K crashed Sharpe) even if the
        # specific K meeting the turnover bar is the one with a bad Sharpe.
        # Evaluate every K in that case rather than silently narrowing.
        surviving_ks = list(K_GRID)
    print(f"\nProceeding to inner-validation with K in {surviving_ks}")

    validation = inner_validation(surviving_ks)

    print(f"\nTotal configs evaluated: {_CONFIGS_EVALUATED}")
    print(f"Truncation probe: {'PASS' if probe['all_pass'] else 'FAIL'}")
    _report_wallclock(t_start)


def _report_wallclock(t_start: float) -> None:
    print(f"\nWall-clock: {time.time() - t_start:.0f}s  |  configs evaluated so far: {_CONFIGS_EVALUATED}")


if __name__ == "__main__":
    main()


# =====================================================================
# RESULTS (appended after running -- NOT edited back into the banner above,
# per this file's own pre-registration discipline). Full run transcript in
# ``experiments/r99_novel_ghe_band_output.txt``. Reproducible via
# ``python experiments/r99_novel_ghe_band.py``, ~18s wall-clock.
# =====================================================================
#
# TRUNCATION PROBE: PASS, 6/6 (z-score and K=0.30 effective_deadband, each
# at check_at in {100_000, 300_000, 550_000}).
#
# STEP-0a (non-degeneracy, inner-train, n=420,481 bars):
#   K=0.15  mean=0.1011 std=0.0149 min=0.0749 max=0.1373  within+/-5%=28.6%  differs>10%=52.3%
#   K=0.30  mean=0.1021 std=0.0299 min=0.0500 max=0.1746  within+/-5%=20.8%  differs>10%=71.4%
#   K=0.50  mean=0.1037 std=0.0459 min=0.0500 max=0.2000  within+/-5%=17.5%  differs>10%=77.3%
#   -> NOT degenerate: none of the 3 K values keeps >95% of bars within
#      +/-5% of the 0.10 base (all three are well under 30%). The GHE
#      signal genuinely moves the band across a real distribution -- this
#      construction does NOT reproduce the 23-prior-round collapse-to-
#      constant failure mode. Step-0a PASSES.
#
# STEP-0b (effect check vs kelly_regime_v4, inner-train, futures 5x):
#   baseline v4      trades=72  sharpe=2.28  DD=35.3%  final=$30,344
#   K=0.15           trades=72  sharpe=2.29  DD=35.3%  final=$30,393   turnover_drop=+0.0%   sharpe_diff=+0.001
#   K=0.30           trades=70  sharpe=2.27  DD=36.7%  final=$29,701   turnover_drop=+2.8%   sharpe_diff=-0.011
#   K=0.50           trades=70  sharpe=2.27  DD=35.7%  final=$29,270   turnover_drop=+2.8%   sharpe_diff=-0.010
#   -> FAILS the pre-registered effect check: the best turnover reduction
#      across the whole 3-point grid is 2.8% (K=0.30, K=0.50), against the
#      pre-registered 10% relative bar. Sharpe is flat either way (worst
#      case -0.011, nowhere near the -0.2 noise floor), so the branch does
#      NOT fail on Sharpe degradation -- it fails purely because the
#      construction barely changes how often the strategy trades.
#
# VERDICT: NEGATIVE. Stopped at Step-0b. Failure mode: turnover-did-not-drop
# (NOT degeneracy, NOT Sharpe-degradation -- the non-degeneracy check
# (Step-0a) passed cleanly and Sharpe is statistically indistinguishable
# from baseline at every K). Per the pre-registered decision rule, this
# means the branch's own motivating claim -- "cuts turnover/whipsaw" --
# failed on real data, so the branch stops here without touching
# inner-validation or the holdout.
#
# WHY, mechanistically (diagnostic, not part of the pre-registered decision
# rule, offered because the non-degeneracy/effect-check split is itself the
# round's most reusable finding): the effective_deadband series is a
# genuinely time-varying signal (std 0.015-0.046 depending on K, range as
# wide as [0.050, 0.200] at K=0.50) -- the SIGNAL is not degenerate. But
# kelly_regime_v4's actual rebalance triggers are dominated by the 20/40/80
# anchor vote's own latched regime flips (frac jumping between {0, 1/3, 2/3,
# 1}) and the v3-style high/low volatility STATE transitions -- both of
# which move `desired` by a large fraction of full exposure when they fire,
# so whether the deadband is 0.05 or 0.20 almost never changes whether that
# particular move clears it. The band's own value only matters on the bars
# where `|desired - pos|` sits close to its threshold, and inner-train has
# very few such near-boundary bars (v4 already trades only 72 times in ~4
# years) -- so a per-bar multiplier on an already-rare trigger event has
# little surface area to act on. This is a DIFFERENT failure mode from the
# 23 prior SIZE-axis collapses (which failed because the modulating SIGNAL
# itself was near-constant); here the signal moves genuinely, but the
# strategy's own trigger structure (sparse, large, latch-driven) is nearly
# insensitive to where exactly the threshold sits. Worth naming precisely
# for future COST-axis attempts on this strategy: modulating the BAND WIDTH
# of an already-latched, already-sparse trigger has little to bite on;
# a construction that instead modulated something continuous-and-frequent
# (e.g. the volatility-state hysteresis thresholds themselves, or the
# vote's own band `V4_BAND`) might have more surface area, though that
# would cross into territory this round's dispatch explicitly reserved for
# other constructions (R-62's scale/vote factorization, R-89's asymmetric
# band, etc.) and is out of scope here.
#
# Total configs evaluated: 4 (baseline v4 + K in {0.15, 0.30, 0.50}, all on
# inner-train only -- inner-validation and the holdout were never reached).
# Wall-clock: 18s.
#
# Implementation gotchas: none of substance. `prepare()` receives whatever
# frame `run_period` slices (warmup prefix + measured period), so the GHE/
# z-score/effective_deadband series is recomputed fresh, causally, for each
# `ev()` call rather than shared across calls -- correct but means repeated
# calls at overlapping periods redo the (cheap, ~0.2-1.5s) daily-cadence GHE
# fit each time; not worth caching at this scale. `align_daily_causal`'s
# ffill means the first bar of a run without an already-buffered daily row
# does not throw, but does mean the array is never actually NaN post-align
# for `daily_log_prices` gaps as long as SOME prior day exists in the frame
# -- confirmed by direct inspection (`eff.isna().sum() == 0` on every frame
# tested), consistent with `ghe_multiplier`'s NaN->1.0 fallback simply never
# being exercised past the very first day or two of the dataset.
