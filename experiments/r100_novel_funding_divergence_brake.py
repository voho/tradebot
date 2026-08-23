#!/usr/bin/env python
"""R-100 NOVEL branch: cross-venue funding-divergence-stress-conditioned
EXECUTION DELAY for ``kelly_regime_v4`` ("the brake").

=====================================================================
PRE-REGISTRATION (frozen before any signal, diagnostic, or backtest
number in this file was computed -- docs/ROUTINE.md steps 1-2). Anything
below later contradicted by what actually happened is stated in the
results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). A blown-out cross-venue Binance-vs-Deribit
   BTC perpetual funding divergence signals limits-to-arbitrage stress
   between the two venues (Zhivkov 2026: ~95% of large cross-venue
   spreads force an early exit) -- a bad time to execute a rebalance,
   because liquidity/spread conditions are likely to be poor -- so
   DELAYING (never skipping) ``kelly_regime_v4``'s scheduled rebalance
   during divergence-stress windows should cut realized execution cost
   without giving up much directional edge, since v4's own signal moves
   on a weeks timescale relative to any plausible delay (hours).

   CONSTRAINT ATTACKED: COST -- the adverse-selection/spread cost of
   executing a rebalance while cross-venue arbitrage capacity is
   stressed. This is contingent on the same INFO-axis signal the
   CONSERVATIVE branch this round is separately evaluating
   (``r100_shared.cross_venue_divergence_z``) first being a real
   distress measure, not on this branch re-deriving that -- same
   INFO->COST dependency structure as R-88's own pair (see
   ``r100_shared.py``'s own docstring for the full citation trail:
   Zhivkov 2026; Inan 2025/26; He/Manela/Ross/von Wachter 2024 -- not
   re-derived here, this project's "one citation trail in one place"
   convention).

   NOT A DUPLICATE OF (full trail also in ``r100_shared.py``):
   - B-24 / R-77 (execution-timing on realized VOLATILITY): this brake
     conditions on cross-venue DIVERGENCE STRESS, a different mechanism,
     per ``r100_shared.py``'s own disclosure.
   - R-88's novel branch (taker-flow-conditioned delay): same
     ARCHITECTURE (delay a rebalance up to K bars, force it through at
     the deadline, never delay a de-risking move), different trigger
     signal (order-flow direction vs. cross-venue funding-divergence
     magnitude) and different directionality (R-88 is SIGNED, opposing
     the trade's own direction; this brake is a scalar STRESS gate that
     does not care which way v4 wants to move, only whether the market
     is judged too stressed to execute an exposure INCREASE right now).
   - The CONSERVATIVE branch running in parallel this round (not read or
     touched by this file): that branch votes on regime using the same
     ``div_z`` signal, changing WHAT v4 wants to hold. This branch never
     touches the target value, only WHEN an already-decided increase
     gets applied.

2. STRESS MEASURE (defined now, before any number was computed).
   ``stress[i] = |div_z[i]| >= threshold``, where ``div_z`` is
   ``r100_shared.cross_venue_divergence_z`` at
   ``PRIMARY_BASELINE_WINDOW_DAYS=60`` (the shared module's own grid
   centre, chosen there before either branch ran, not re-picked here).
   The raw |div_z| level is used, NOT a rolling volatility of the
   series, because the mechanism (Zhivkov 2026) is that a large,
   PRESENT divergence itself indicates a currently-stressed arbitrage
   channel -- the claim is about the level of segmentation right now,
   not about how erratically the divergence has been moving. A
   volatility-of-divergence measure would be answering a different
   question ("is the spread itself unstable") that has no motivating
   citation in this round's literature trail. ``|div_z|`` (unsigned) is
   used, not a signed threshold, because BOTH directions of divergence
   (Binance running hot relative to Deribit, or the reverse) reflect the
   same underlying limits-to-arbitrage stress in Zhivkov's framing --
   unlike R-88's taker-flow measure, there is no reason here to expect
   only one sign of divergence to matter. ``div_z`` is NaN before
   funding history opens a full baseline window (~2020-03), which
   causally means "stress status unknown" and is treated as NOT stressed
   (``False``) -- the conservative, causal default; this is disclosed
   and quantified in the diagnostics below, not glossed over.

3. CONSTRUCTION (frozen before any run). ``FundingDivergenceBrakeV4`` is
   a plain (unregistered) ``Strategy`` subclass whose ``prepare()`` calls
   ``KellyRegimeV4().prepare(df.copy())`` UNMODIFIED to get v4's own real,
   causal ``target`` array (v4's sizing math is reused verbatim, never
   reimplemented), then post-processes that array with the delay logic
   below before handing a (possibly delayed) ``target`` column to the
   engine via the project's standard ``ctx.order_notional`` bar-close/
   next-open fill contract (same execution-timing convention R-88 used,
   for the same reason: this round is about WHEN v4's target is applied,
   not about a resting-limit microstructure model).

   At each bar i, with ``pos`` the currently-executed target and
   ``desired = target[i]``:
     - no change (``|desired - pos| <= EPS``): hold, no state change.
     - DE-RISKING (``|desired| < |pos|``, i.e. moving toward flat): ALWAYS
       applied immediately, cancelling any pending delay -- the same
       "full exit is always allowed" principle ``kelly_regime``'s own
       no-trade-band construction uses (docs/STRATEGIES.md appendix).
       ``kelly_regime_v4``'s own vote fraction and vol-scale are both
       always >= 0 (frac in {0,1/3,2/3,1}, scale = min(target_vol/vol,
       max_leverage) >= 0), so v4 never actually issues a negative
       target in this dataset -- verified as a diagnostic below, not
       assumed -- meaning "de-risking" and "moving toward zero" coincide
       exactly here; the general rule above is stated for robustness,
       not because a sign flip is expected to occur.
     - INCREASING OR FLIPPING (``|desired| >= |pos|``): if not already
       delaying, and ``stress[i]`` is True, START delaying (hold ``pos``
       at its current value, remember ``placed_at = i``). If not
       already delaying and ``stress[i]`` is False, apply ``pos <-
       desired`` immediately. If already delaying, re-check every bar:
       apply ``pos <- target[i]`` (the CURRENT desired target, never a
       stale one) the moment EITHER ``stress[i]`` turns False OR
       ``i - placed_at >= K`` (the hard cap -- a delay, not an
       indefinite freeze).

   Causal by construction: row i depends only on ``target[<=i]``,
   ``stress[<=i]``, and running state built from strictly earlier rows.
   Verified with ``r100_shared.truncation_causality_probe`` below.

4. SWEEP GRID (fixed now, 3x3=9 configs, not retuned after seeing any
   result):
     threshold (div_z stress cutoff)  in (1.0, 1.5, 2.0)
     K (max delay, 5-min bars)        in (6, 24, 96)    (30min / 2h / 8h)

   Evaluated on 4 CELLS: {inner-train (end=2020-12-31),
   inner-validation (2021-01-01 -> 2022-12-31)} x {spot, futures 5x}.
   Pre-registered total backtests: 9 configs x 4 cells = 36, plus
   ``kelly_regime_v4`` itself on the same 4 cells (4 more, the paired
   baseline every delta below is measured against) = 40 backtests, plus
   1 causal-truncation-probe diagnostic pair. Fees are this harness's
   entry tier (``scripts.experiment.ev``'s default); funding is NOT
   charged (this project's harness default when no ``funding=`` series is
   passed), matching R-88's own convention for an architecture-comparison
   round where both arms trade near-identical position paths under
   identical cost assumptions -- the DELTA between arms, not either arm's
   absolute level, is what this round measures, and funding/real-fee
   levels would apply near-symmetrically to both.

5. WINNER SELECTION (fixed now): among the 9 grid configs, the winner is
   the one with the highest SUM of inner-validation Sharpe across
   {spot, futures 5x} (ties broken by the lower mean max-drawdown across
   those two cells) -- selecting on inner-validation is this project's
   standard convention (docs/ROUTINE.md step 3), never on the holdout,
   which this file never reads (enforced by
   ``r100_shared.assert_no_holdout`` after every load).

6. PRE-REGISTERED PROMOTION/KILL BAR for the winner (stated now, before
   any run): compute the paired Sharpe difference (delayed winner minus
   plain ``kelly_regime_v4``) on each of the 4 cells via
   ``tradebot.inference.paired_bootstrap`` (this project's standard
   paired stationary block bootstrap -- 30-day mean block, 2,000
   resamples, the SAME resample applied to both arms so the market's own
   variance cancels -- reused directly from ``tradebot.inference``
   rather than reimplemented). The branch is a GENUINE CANDIDATE only if,
   on AT LEAST 2 of the 4 cells, the delta exceeds this project's +-0.2
   Sharpe noise floor (R-20) AND the 95% CI excludes zero on the positive
   side, AND no cell shows a SIGNIFICANT WORSENING (a 95% CI entirely
   below zero). Anything else -- including "mostly flat, no cell moves"
   -- is NEGATIVE.

7. WHAT WOULD MAKE THIS FAIL, named now:
   (a) v4's own target already rebalances too infrequently for any K in
       the 30min-8h range to matter: R-72/B-30 already found v4's own
       10% deadband discards roughly half of scheduled rebalances at 5x,
       so an execution-timing brake sitting on top of an already-sparse
       schedule may simply have too few "delay opportunities" per year
       to move the needle -- checked directly below (diagnostic, before
       any Sharpe number) by counting how many of v4's own
       exposure-increasing bars actually overlap ``stress``;
   (b) the stress condition itself fires too rarely (e.g. concentrated
       in a handful of episodes) for a delay mechanism riding on it to
       have enough independent opportunities to show a distinguishable
       effect over only ~2 years of inner-validation;
   (c) delaying costs more in missed favorable moves than it saves in
       avoided adverse-selection cost -- a real divergence-stress episode
       could be the START of a genuine repricing (the right time to move
       fast), not a transient dislocation that reverts within K bars, in
       which case "wait for calm" means chasing a worse fill;
   (d) funding coverage starts 2020-01-01 (``r100_shared.py``'s own
       disclosed caveat), so roughly the first three years of
       inner-train (2017-2019) have NO stress signal at all (NaN ->
       treated as not-stressed) -- if that dead zone dominates the
       inner-train cell's own bar count, inner-train is close to a
       structural no-op by construction and only inner-validation (full
       coverage) is informative; quantified below, not assumed.

CONFIGS EVALUATED IN THIS FILE: 40 backtests (9x4 grid-vs-baseline pairs
+ 4 standalone ``kelly_regime_v4`` baseline runs) + 1 causal-truncation
diagnostic. Never touches ``docs/LEDGER.md``, git, or
``experiments/r100_shared.py``.

USAGE
-----
    python experiments/r100_novel_funding_divergence_brake.py
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

from tradebot.inference import (  # noqa: E402
    annualized_sharpe,
    daily_returns,
    paired_bootstrap,
    stationary_bootstrap_indices,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments import r100_shared  # noqa: E402
from scripts.experiment import DF as BTC_DF, FUTURES, LABEL, SPOT, OOS_START  # noqa: E402

DATA_DIR = ROOT / "data"

# ------------------------------------------------------------ pre-registered params
THRESH_GRID = (1.0, 1.5, 2.0)
K_GRID = (6, 24, 96)  # 5-min bars: 30min / 2h / 8h
EPS_TARGET = 1e-9
NOISE_FLOOR = 0.2          # R-20's own +-0.2 Sharpe noise floor
MEAN_BLOCK = 30.0          # this project's standard 30-day mean block
N_BOOT = 2_000             # this project's standard resample count
BOOT_SEED = 100024         # fixed now, this file's own seed

INNER_TRAIN_END = r100_shared.INNER_TRAIN_END
INNER_VAL_START = r100_shared.INNER_VAL_START
INNER_VAL_END = r100_shared.INNER_VAL_END

CELLS = [
    ("train", "spot", None, INNER_TRAIN_END, SPOT),
    ("train", "futures5x", None, INNER_TRAIN_END, FUTURES),
    ("val", "spot", INNER_VAL_START, INNER_VAL_END, SPOT),
    ("val", "futures5x", INNER_VAL_START, INNER_VAL_END, FUTURES),
]

CONFIG_COUNTER = {"grid": 0, "baseline": 0, "diagnostic": 0}


def _count(kind: str, k: int = 1) -> None:
    CONFIG_COUNTER[kind] += k


# =====================================================================
# signal construction
# =====================================================================

def load_price_bars() -> pd.DataFrame:
    df = BTC_DF.loc[BTC_DF.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    r100_shared.assert_no_holdout(df)
    print(f"BTC spot bars: {len(df):,}  {df.index[0]} -> {df.index[-1]}  (< {OOS_START})",
          file=sys.stderr)
    return df


def build_div_z_aligned(bars: pd.DataFrame) -> pd.Series:
    """Cross-venue divergence z, aligned causally onto ``bars``' 5m index."""
    daily = r100_shared.load_daily_funding_totals(DATA_DIR)
    r100_shared.assert_no_holdout(daily)
    div_z = r100_shared.cross_venue_divergence_z(
        daily, baseline_window_days=r100_shared.PRIMARY_BASELINE_WINDOW_DAYS)
    r100_shared.assert_no_holdout(div_z)
    aligned = r100_shared.align_daily_causal(div_z, bars)
    r100_shared.assert_no_holdout(aligned)
    return aligned


# =====================================================================
# delay mechanism
# =====================================================================

@dataclass(frozen=True)
class DelayConfig:
    threshold: float
    k_max: int

    def tag(self) -> str:
        return f"z{self.threshold:g}_K{self.k_max}"


def build_delayed_target(target: np.ndarray, div_z: np.ndarray,
                          cfg: DelayConfig) -> tuple[np.ndarray, dict]:
    """Causal, bar-by-bar construction of the brake-delayed execution series.

    Row i depends only on target[<=i], div_z[<=i], and running (pos,
    pending_since) state built from strictly earlier bars.
    """
    n = len(target)
    out = np.empty(n)
    stress = np.zeros(n, dtype=bool)
    finite = np.isfinite(div_z)
    stress[finite] = np.abs(div_z[finite]) >= cfg.threshold

    pos = 0.0
    pending_since = None
    n_derisk = 0
    n_increase_events = 0   # bars where an increase/flip was DESIRED
    n_delayed = 0           # of those, how many started a delay
    n_forced = 0
    n_favorable = 0
    delay_lengths: list[int] = []

    for i in range(n):
        desired = target[i]
        diff = desired - pos
        if abs(diff) <= EPS_TARGET:
            out[i] = pos
            continue

        is_derisk = abs(desired) < abs(pos) - EPS_TARGET
        if is_derisk:
            n_derisk += 1
            pos = desired
            pending_since = None
            out[i] = pos
            continue

        # increasing-or-flipping change desired
        if pending_since is None:
            n_increase_events += 1  # a FRESH increase decision, counted once
            if stress[i]:
                pending_since = i
                n_delayed += 1
                out[i] = pos  # hold, do not execute yet
            else:
                pos = desired
                out[i] = pos
            continue

        age = i - pending_since
        if (not stress[i]) or age >= cfg.k_max:
            if stress[i] and age >= cfg.k_max:
                n_forced += 1
            else:
                n_favorable += 1
            delay_lengths.append(age)
            pos = desired
            pending_since = None
            out[i] = pos
        else:
            out[i] = pos

    diag = dict(n_derisk=n_derisk, n_increase_events=n_increase_events,
                n_delayed=n_delayed, n_forced=n_forced, n_favorable=n_favorable,
                mean_delay=float(np.mean(delay_lengths)) if delay_lengths else float("nan"),
                stress_frac=float(stress.mean()), finite_frac=float(finite.mean()))
    return out, diag


class FundingDivergenceBrakeV4(Strategy):
    """kelly_regime_v4 with exposure-increasing rebalances delayed up to K
    bars while cross-venue funding-divergence stress (|div_z|>=threshold)
    is active; de-risking rebalances are always applied immediately."""

    warmup = KellyRegimeV4().warmup

    def __init__(self, threshold: float = 1.5, k_max: int = 24) -> None:
        self.cfg = DelayConfig(threshold, k_max)
        self.name = f"funding_divergence_brake_v4[{self.cfg.tag()}]"
        self._last_diag: dict = {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        r100_shared.assert_no_holdout(df)
        v4_target = KellyRegimeV4().prepare(df.copy())["target"].to_numpy(dtype=float)
        div_z = build_div_z_aligned(df).to_numpy(dtype=float)
        delayed, diag = build_delayed_target(v4_target, div_z, self.cfg)
        self._last_diag = diag
        df["target"] = delayed
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > EPS_TARGET:
            ctx.order_notional(t)


# =====================================================================
# diagnostics (0 backtests -- pure signal/array analysis)
# =====================================================================

def run_diagnostics(bars: pd.DataFrame) -> dict:
    print("\n" + "=" * 100)
    print("DIAGNOSTIC -- stress-firing rate and overlap with v4's own rebalances "
          "(inner period, no backtests run)")
    print("=" * 100)

    v4_target = KellyRegimeV4().prepare(bars.copy())["target"].to_numpy(dtype=float)
    n_negative = int(np.sum(v4_target < -EPS_TARGET))
    print(f"v4's own target: min={v4_target.min():.4f} max={v4_target.max():.4f} "
          f"negative bars={n_negative}/{len(v4_target)} "
          f"(0 confirms v4 never shorts in this dataset, as assumed in the "
          f"pre-registration's de-risking rule)")

    div_z = build_div_z_aligned(bars).to_numpy(dtype=float)
    n_years = (bars.index[-1] - bars.index[0]).days / 365.25

    # v4's own exposure-increasing bars (the only ones eligible for delay)
    diff = np.diff(v4_target, prepend=0.0)
    increasing = diff > EPS_TARGET
    n_increasing = int(increasing.sum())
    print(f"\nv4's own exposure-increasing target-changing bars: {n_increasing} "
          f"over {n_years:.2f} years  ({n_increasing / n_years:.1f} / year)")

    rows = []
    for th in THRESH_GRID:
        finite = np.isfinite(div_z)
        stress = np.zeros(len(div_z), dtype=bool)
        stress[finite] = np.abs(div_z[finite]) >= th
        stress_frac = float(stress.mean())
        stress_per_year = float(stress.sum()) / n_years
        overlap = int((increasing & stress).sum())
        overlap_pct = 100.0 * overlap / n_increasing if n_increasing else float("nan")
        rows.append(dict(threshold=th, stress_frac=stress_frac,
                          stress_bars_per_year=stress_per_year,
                          finite_frac=float(finite.mean()),
                          overlap_bars=overlap, overlap_pct=overlap_pct))
        print(f"  threshold={th:g}: stress active {100*stress_frac:5.1f}% of bars "
              f"({stress_per_year:8.0f} bars/yr, {stress_per_year/r100_shared.BARS_PER_DAY:5.1f} d/yr)  "
              f"div_z finite {100*float(finite.mean()):5.1f}% of bars  "
              f"overlap with v4's own increasing bars: {overlap}/{n_increasing} ({overlap_pct:5.1f}%)")

    train_mask = bars.index < pd.Timestamp(INNER_TRAIN_END, tz="UTC") + pd.Timedelta(days=1)
    print(f"\ncoverage caveat: div_z is finite for {100*float(np.isfinite(div_z[train_mask]).mean()):.1f}% "
          f"of inner-train bars vs {100*float(np.isfinite(div_z[~train_mask]).mean()):.1f}% of "
          f"inner-validation bars (funding starts 2020-01-01; inner-train runs 2017-2020).")
    return dict(rows=rows, n_increasing=n_increasing, n_years=n_years)


def causality_probe(bars: pd.DataFrame, cfg: DelayConfig) -> bool:
    print("\n" + "=" * 100)
    print(f"CAUSAL-TRUNCATION PROBE -- {cfg.tag()}, r100_shared.truncation_causality_probe")
    print("=" * 100)

    def build_target_fn(frame: pd.DataFrame) -> np.ndarray:
        v4_target = KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float)
        div_z = build_div_z_aligned(frame).to_numpy(dtype=float)
        delayed, _ = build_delayed_target(v4_target, div_z, cfg)
        return delayed

    check_at = len(bars) - 40_000
    ok = r100_shared.truncation_causality_probe(build_target_fn, bars,
                                                 check_at=check_at, shorter_by=20_000)
    print(f"  check_at bar={check_at}  ({bars.index[check_at]})  shorter_by=20,000 bars")
    print(f"  CAUSAL-TRUNCATION PROBE: {'PASS' if ok else 'FAIL'}")
    _count("diagnostic")
    return ok


# =====================================================================
# backtest cell runner
# =====================================================================

def run_cell(strategy: Strategy, market, start, end, tag: str):
    result = run_period(strategy, BTC_DF, start, end, market=market,
                        start_balance=1_000.0, data_label=LABEL)
    r100_shared.assert_no_holdout(result.df)
    m = compute_metrics(result)
    print(f"{tag:40s} {market.name:11s} final=${m.final_balance:>13,.0f} "
          f"({m.profit_pct:>+9.1f}%) trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}")
    rets = daily_returns(result.equity)
    return m, rets


# =====================================================================
# main sweep
# =====================================================================

def main() -> None:
    t0 = time.time()
    bars = load_price_bars()

    diag = run_diagnostics(bars)

    configs = [DelayConfig(th, k) for th in THRESH_GRID for k in K_GRID]
    print(f"\npre-registered grid: threshold in {THRESH_GRID}, K in {K_GRID}  "
          f"({len(configs)} configs) x 4 cells = {len(configs) * 4} backtests, "
          f"+4 kelly_regime_v4 baselines")

    max_ts_seen = bars.index.max()

    # baseline v4 + bootstrap index matrix, once per cell
    cell_baseline = {}
    cell_indices = {}
    for period, mkt_label, start, end, market in CELLS:
        key = (period, mkt_label)
        m, rets = run_cell(KellyRegimeV4(), market, start, end,
                            tag=f"[baseline] kelly_regime_v4 {period}/{mkt_label}")
        _count("baseline")
        cell_baseline[key] = (m, rets)
        n = len(rets)
        idx = stationary_bootstrap_indices(n, MEAN_BLOCK, N_BOOT,
                                            np.random.default_rng(BOOT_SEED + hash(key) % 10_000))
        cell_indices[key] = idx
        max_ts_seen = max(max_ts_seen, rets.index.max())

    grid_results: dict[tuple, dict] = {}  # (cfg.tag(), cell_key) -> row dict
    for cfg in configs:
        for period, mkt_label, start, end, market in CELLS:
            key = (period, mkt_label)
            strat = FundingDivergenceBrakeV4(threshold=cfg.threshold, k_max=cfg.k_max)
            m, rets = run_cell(strat, market, start, end,
                                tag=f"{cfg.tag()} {period}/{mkt_label}")
            _count("grid")
            base_m, base_rets = cell_baseline[key]
            if len(rets) != len(base_rets):
                # align on the intersection of daily index (should not
                # normally happen -- both arms share warmup/window)
                common = rets.index.intersection(base_rets.index)
                rets_a, rets_b = rets.loc[common].to_numpy(), base_rets.loc[common].to_numpy()
            else:
                rets_a, rets_b = rets.to_numpy(), base_rets.to_numpy()
            pr = paired_bootstrap(rets_a, rets_b, annualized_sharpe,
                                  mean_block=MEAN_BLOCK, n_boot=N_BOOT,
                                  seed=BOOT_SEED, indices=cell_indices[key])
            grid_results[(cfg.tag(), key)] = dict(
                cfg=cfg, period=period, market=mkt_label,
                sharpe_delayed=m.sharpe, sharpe_v4=base_m.sharpe,
                delta=pr.diff.point, lo=pr.diff.lo, hi=pr.diff.hi,
                significant=pr.significant, diag=dict(strat._last_diag))
            max_ts_seen = max(max_ts_seen, rets.index.max())

    print("\n" + "=" * 100)
    print("FULL 3x3x4-CELL RESULT GRID")
    print("=" * 100)
    print(f"{'config':14s} {'cell':16s} {'sharpe_delayed':>14s} {'sharpe_v4':>10s} "
          f"{'delta':>8s} {'95% CI':>18s} {'sig':>5s}")
    for cfg in configs:
        for period, mkt_label, *_ in CELLS:
            key = (period, mkt_label)
            r = grid_results[(cfg.tag(), key)]
            print(f"{cfg.tag():14s} {period+'/'+mkt_label:16s} {r['sharpe_delayed']:14.3f} "
                  f"{r['sharpe_v4']:10.3f} {r['delta']:+8.3f} "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  {'Y' if r['significant'] else 'n'}")

    # winner selection: max sum of inner-VAL sharpe (spot+futures)
    def val_score(cfg: DelayConfig) -> tuple[float, float]:
        s = grid_results[(cfg.tag(), ("val", "spot"))]
        f = grid_results[(cfg.tag(), ("val", "futures5x"))]
        return (s["sharpe_delayed"] + f["sharpe_delayed"], -(s["diag"].get("mean_delay", 0) or 0))

    winner_cfg = max(configs, key=val_score)
    print(f"\nWINNER (by inner-validation Sharpe sum, spot+futures): {winner_cfg.tag()}")
    for period, mkt_label, *_ in CELLS:
        d = grid_results[(winner_cfg.tag(), (period, mkt_label))]["diag"]
        print(f"  {period}/{mkt_label:10s} diag: increase_events={d['n_increase_events']} "
              f"delayed={d['n_delayed']} forced_through={d['n_forced']} "
              f"resolved_favorably={d['n_favorable']} mean_delay={d['mean_delay']:.2f} bars "
              f"stress_frac={d['stress_frac']:.3f} div_z_finite_frac={d['finite_frac']:.3f}")

    print("\n" + "=" * 100)
    print(f"PRE-REGISTERED PROMOTION BAR applied to winner {winner_cfg.tag()}")
    print("=" * 100)
    n_pass, n_worsen = 0, 0
    for period, mkt_label, *_ in CELLS:
        key = (period, mkt_label)
        r = grid_results[(winner_cfg.tag(), key)]
        clears = (r["delta"] > NOISE_FLOOR) and r["significant"] and r["delta"] > 0
        worsens = r["significant"] and r["delta"] < 0
        n_pass += int(clears)
        n_worsen += int(worsens)
        print(f"  {period}/{mkt_label:10s} delta={r['delta']:+.3f} CI=[{r['lo']:+.3f},{r['hi']:+.3f}] "
              f"clears_bar={clears} significant_worsening={worsens}")
    candidate = (n_pass >= 2) and (n_worsen == 0)
    print(f"\ncells clearing bar: {n_pass}/4   cells significantly worse: {n_worsen}/4")
    print(f"VERDICT: {'GENUINE CANDIDATE -> recommend holdout consultation to operator' if candidate else 'NEGATIVE'}")
    print("NOTE: this file has NO authority to consult the holdout regardless of this outcome.")

    probe_ok = causality_probe(bars, winner_cfg)

    total = CONFIG_COUNTER["grid"] + CONFIG_COUNTER["baseline"] + CONFIG_COUNTER["diagnostic"]
    print(f"\nCAUSAL-TRUNCATION PROBE: {'PASS' if probe_ok else 'FAIL'}")
    print(f"CONFIGS EVALUATED: grid={CONFIG_COUNTER['grid']} baseline={CONFIG_COUNTER['baseline']} "
          f"diagnostic={CONFIG_COUNTER['diagnostic']} total={total}")
    print(f"max timestamp read anywhere in this file: {max_ts_seen}  (< {OOS_START})")
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
