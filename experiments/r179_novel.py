#!/usr/bin/env python
"""R-179 NOVEL branch (08-29): calibrated continuous meta-sizing on top of
`kelly_regime_v4`'s own vote+scale signal.

See `experiments/r179_direction.md` ("Novel branch") for the frozen
mechanism, non-duplication argument and falsification rules; this file only
executes that design and records the resulting numbers. Does NOT edit
`experiments/r179_shared.py` or `experiments/r179_direction.md`, and does
not touch the sibling conservative branch (separate agent, separate file,
per this project's parallel-branch convention) or anything under
`src/tradebot/`.

Mechanism, one sentence (frozen, unchanged from the pre-registration):
``m(p) = clip(1 + steepness*(p-0.5), 0, cap)``; ``final_desired =
min(frac*scale*m(p), max_leverage)``; ``m=1.0`` wherever the walk-forward
meta-classifier's probability `p` is NaN (before its first valid refit --
the disclosed neutral warmup behaviour); the SAME 10% deadband v4 itself
uses is then applied sequentially to `final_desired`, exactly as v4 applies
it to its own `desired = frac*scale`. `frac`/`scale`/`vol_ratio` come from
`r179_shared.vote_frac`/`conditional_scale` (bit-identical to v4's own
factors); `p` comes from `r179_shared.walk_forward_meta_prob`, unmodified.

Evaluation protocol (docs/ROUTINE.md Step 3): `scripts/experiment.py`'s
`ev()` helper, train/inner-validation only -- 2023-01-01 onward is never
read by this file. See `main()` for the full sweep composition and exact
config count.

Usage
-----
    python experiments/r179_novel.py
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

from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r179_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    conditional_scale,
    vote_frac,
    walk_forward_meta_prob,
)

from scripts.experiment import DF, FUTURES, LABEL, SPOT, ev  # noqa: E402

TRAIN_END = "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"

N_EVALUATED = 0  # distinct parameter configurations run, for the report


# ============================================================================
# (1) the Strategy itself
# ============================================================================


class R179NovelMetaSigmoid(Strategy):
    """v4's own frac*scale, continuously rescaled by a sigmoid of a
    walk-forward meta-labeling probability (Joubert, Barziy & Meyer 2022's
    "optimal sigmoid" position sizing), instead of a binary bet/no-bet veto.

    frac/scale/vol_ratio are v4's own factors, computed verbatim via
    `r179_shared.vote_frac`/`conditional_scale` (never re-derived here).
    `vol_daily` is the same causal EWM realized-vol array `conditional_scale`
    computes internally (reproduced here since `conditional_scale` returns
    only `scale`/`vol_ratio`), passed to `r179_shared.walk_forward_meta_prob`
    unmodified. The resulting per-bar probability `p` feeds

        m(p) = clip(1 + steepness*(p - 0.5), 0, cap)
        final_desired = min(frac*scale*m(p), max_leverage)

    with m=1.0 (neutral -- v4-identical) wherever `p` is NaN (before the
    classifier's first valid refit). v4's own 10% deadband is then applied
    sequentially to `final_desired`, exactly as v4 applies it to its own
    `desired = frac*scale`: the meta-signal changes WHAT is desired, not
    whether the deadband logic runs.

    `diag_` (set by `prepare()`) is `walk_forward_meta_prob`'s own
    diagnostic dict for whatever slice of data this instance was last run
    on -- this round's pre-registered falsification clause A reads this,
    not any trading number.
    """

    name = "r179_novel_meta_sigmoid"
    warmup = 80 * BARS_PER_DAY + 10  # same convention as kelly_regime_v4

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85,
                 k: float = 1.0, horizon_days: int = 3, refit_days: int = 60,
                 embargo_days: int = 3, min_samples: int = 50,
                 steepness: float = 2.0, cap: float = 2.0) -> None:
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.k = k
        self.horizon_days = horizon_days
        self.refit_days = refit_days
        self.embargo_days = embargo_days
        self.min_samples = min_samples
        self.steepness = steepness
        self.cap = cap
        # populated by prepare(), read back by the caller after a run
        self.diag_: dict | None = None
        self.n_refit_checkpoints_: int | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        n = len(df)

        frac = vote_frac(close, horizons=self.horizons, band=self.band)
        scale, vol_ratio = conditional_scale(
            close, target_vol=self.target_vol, max_leverage=self.max_leverage,
            vol_span=self.vol_span, anchor_span_days=self.anchor_span_days,
            high_in=self.high_in, high_out=self.high_out,
            low_in=self.low_in, low_out=self.low_out)

        # v4's own causal EWM realized-vol array (conditional_scale's own
        # internal `vol`, reproduced verbatim -- conditional_scale returns
        # only `scale`/`vol_ratio`, see r179_shared.py's own docstring).
        r = np.log(close).diff()
        vol_daily = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
                     * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        prob, diag = walk_forward_meta_prob(
            df.index, close.to_numpy(dtype=float), vol_daily, frac, vol_ratio,
            k=self.k, horizon_days=self.horizon_days, refit_days=self.refit_days,
            embargo_days=self.embargo_days, min_samples=self.min_samples)

        self.diag_ = diag
        self.n_refit_checkpoints_ = len(range(0, n, self.refit_days * BARS_PER_DAY))

        m = np.clip(1.0 + self.steepness * (prob - 0.5), 0.0, self.cap)
        m = np.where(np.isfinite(prob), m, 1.0)  # neutral (v4-identical) during warmup

        final_desired = np.minimum(frac * scale * m, self.max_leverage)

        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = final_desired[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_prob"] = prob
        df["_m"] = m
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ============================================================================
# (2) measurement helpers (realized vol / notional / diag, beyond what
#     ev()'s printed Metrics line carries)
# ============================================================================


def mean_notional(result) -> float:
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity: pd.Series) -> float:
    eq = equity.to_numpy(dtype=float)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(rets.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


def full_measure(strategy: Strategy, start, end, market, count: bool = False) -> dict:
    """One backtest -> metrics + realized vol + mean notional + the raw result."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    result = run_period(strategy, DF, start, end, market=market,
                         start_balance=1_000.0, data_label=LABEL)
    m = compute_metrics(result)
    return dict(metrics=m, vol=realized_vol(result.equity), notional=mean_notional(result),
                result=result, diag=getattr(strategy, "diag_", None))


def diag_summary(diag: dict, n_checkpoints: int) -> dict:
    """Clause A's own two conditions, computed from `diag`.

    `n_no_fit` = refit checkpoints where the expanding-window eligible-label
    count had not yet reached `min_samples` (no model existed yet) -- by
    construction this count only ever occurs in an unbroken prefix (the
    eligible set is non-decreasing), so it is exactly "checkpoints below the
    50-sample bar". `frac_z_below_1_of_checkpoints` treats those same
    no-fit checkpoints as ALSO failing the z-score clause (there is no
    fitted coefficient at all, i.e. no evidence of signal) -- the
    conservative reading used for the actual clause-A verdict below;
    `frac_z_below_1_of_fits` (informative only) restricts to refits that
    actually happened.
    """
    n_at = diag["n_at_refit"]
    z = diag["max_abs_z"]
    refits = diag["refits"]
    n_no_fit = n_checkpoints - refits
    frac_below_min_samples = n_no_fit / n_checkpoints if n_checkpoints else float("nan")
    n_z_bad = sum(1 for x in z if x < 1.0)
    frac_z_below_1_of_fits = n_z_bad / refits if refits else float("nan")
    frac_z_below_1_of_checkpoints = (n_z_bad + n_no_fit) / n_checkpoints if n_checkpoints else float("nan")
    return dict(
        n_checkpoints=n_checkpoints, refits=refits,
        median_n_at_refit=float(np.median(n_at)) if n_at else float("nan"),
        min_n_at_refit=int(min(n_at)) if n_at else 0,
        median_max_abs_z=float(np.median(z)) if z else float("nan"),
        frac_below_min_samples=frac_below_min_samples,
        frac_z_below_1_of_fits=frac_z_below_1_of_fits,
        frac_z_below_1_of_checkpoints=frac_z_below_1_of_checkpoints,
        clause_a_fires=bool(frac_below_min_samples > 0.5 or frac_z_below_1_of_checkpoints > 0.5),
    )


def print_diag(tag: str, ds: dict) -> None:
    print(f"  {tag:28s} checkpoints={ds['n_checkpoints']:>4d} actual_refits={ds['refits']:>4d} "
          f"median_n_at_refit={ds['median_n_at_refit']:>6.0f} min_n={ds['min_n_at_refit']:>4d} "
          f"median_max|z|={ds['median_max_abs_z']:>5.2f} "
          f"frac_checkpoints_below_min_samples={ds['frac_below_min_samples']:.2f} "
          f"frac_z<1(of_fits)={ds['frac_z_below_1_of_fits']:.2f} "
          f"frac_z<1(of_checkpoints,conservative)={ds['frac_z_below_1_of_checkpoints']:.2f} "
          f"CLAUSE_A_FIRES={ds['clause_a_fires']}")


# ============================================================================
# (3) phase 1: exploratory (refit_days, horizon_days) x clause-A scan, TRAIN
# ============================================================================

RH_GRID = ((30, 1), (30, 3), (90, 1), (90, 3))


def phase1_explore() -> dict:
    """Run the 4 (refit_days, horizon_days) corners at steepness=2.0, cap=2.0
    (defaults) on inner-train, SPOT, to (a) sanity-check the mechanism trades
    at all and (b) get clause A's diagnostic for each corner -- steepness/cap
    do not affect walk_forward_meta_prob's own diag at all (they only enter
    the sigmoid multiplier downstream), so this single scan is reused for
    EVERY steepness x cap config sharing the same (refit_days, horizon_days)
    below, rather than recomputed once per config."""
    print("\n=== PHASE 1: exploratory (refit_days, horizon_days) scan, clause A, "
          "inner-train, SPOT (steepness=2.0, cap=2.0 fixed) ===")
    out = {}
    for rd, hd in RH_GRID:
        strat = R179NovelMetaSigmoid(refit_days=rd, horizon_days=hd)
        m = ev(strat, market=SPOT, tag=f"train rd={rd} hd={hd}", end=TRAIN_END)
        global N_EVALUATED
        N_EVALUATED += 1
        ds = diag_summary(strat.diag_, strat.n_refit_checkpoints_)
        print_diag(f"rd={rd} hd={hd}", ds)
        out[(rd, hd)] = dict(metrics=m, diag=ds)
    return out


# ============================================================================
# (4) phase 2: steepness x cap sweep at the chosen (refit_days, horizon_days),
#     inner-validation, BOTH markets -- this is the actual selection grid.
# ============================================================================

STEEPNESS_GRID = (1.0, 2.0, 4.0)
CAP_GRID = (1.5, 2.0)


def compare_to_v4(cand_result, v4_result) -> dict:
    """Δ Sharpe (per-bar metrics.sharpe) and Δ log-growth (daily, matches
    this project's own daily-return bootstrap convention -- see
    tradebot.inference's module docstring)."""
    cand_daily = daily_returns(cand_result.equity).to_numpy()
    v4_daily = daily_returns(v4_result.equity).to_numpy()
    n = min(len(cand_daily), len(v4_daily))
    cand_daily, v4_daily = cand_daily[-n:], v4_daily[-n:]
    d_log = float(total_log_return(cand_daily) - total_log_return(v4_daily))
    return dict(d_log_growth=d_log, cand_daily=cand_daily, v4_daily=v4_daily)


def sweep_configs(configs: list[dict], v4_cache: dict) -> pd.DataFrame:
    """Run `configs` on inner-validation, BOTH markets; compare each to
    v4-alone on the identical slice/market (v4_cache holds those, run once)."""
    rows = []
    for cfg in configs:
        for market_name, market in (("spot", SPOT), ("futures_5x", FUTURES)):
            strat = R179NovelMetaSigmoid(**cfg)
            out = full_measure(strat, VAL_START, VAL_END, market, count=True)
            m = out["metrics"]
            v4 = v4_cache[market_name]
            cmp_ = compare_to_v4(out["result"], v4["result"])
            row = dict(**cfg, market=market_name,
                       final=m.final_balance, sharpe=m.sharpe, dd=m.max_drawdown_pct,
                       vol=out["vol"], notional=out["notional"],
                       time_in_mkt=m.time_in_market_pct,
                       d_sharpe=m.sharpe - v4["metrics"].sharpe,
                       d_log_growth=cmp_["d_log_growth"],
                       vol_ratio_vs_v4=out["vol"] / v4["vol"] if v4["vol"] else float("nan"))
            rows.append(row)
            print(f"  rd={cfg['refit_days']:>3d} hd={cfg['horizon_days']:>2d} "
                  f"steep={cfg['steepness']:.1f} cap={cfg['cap']:.1f} {market_name:11s} "
                  f"final=${m.final_balance:>9,.0f} sharpe={m.sharpe:>5.2f} "
                  f"(v4={v4['metrics'].sharpe:>5.2f}, d={row['d_sharpe']:+.3f}) "
                  f"DD={m.max_drawdown_pct:>5.1f}% vol={out['vol']:.3f} (v4={v4['vol']:.3f}) "
                  f"notional={out['notional']:.3f} (v4={v4['notional']:.3f}) "
                  f"d_log_growth={cmp_['d_log_growth']:+.4f}")
    return pd.DataFrame(rows)


# ============================================================================
# main
# ============================================================================


def hr(msg: str) -> None:
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


def main() -> None:
    t0 = time.time()
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(holdout 2023-01-01 onward is NEVER read by this file)")

    # ---- v4-alone control, cached once per market, inner-validation ----
    hr("v4-alone control, inner-validation (2021-01-01 -> 2022-12-31), both markets")
    v4_cache = {}
    for market_name, market in (("spot", SPOT), ("futures_5x", FUTURES)):
        out = full_measure(KellyRegimeV4(), VAL_START, VAL_END, market)
        v4_cache[market_name] = out
        m = out["metrics"]
        print(f"  v4 alone  {market_name:11s} final=${m.final_balance:>9,.0f} "
              f"sharpe={m.sharpe:>5.2f} DD={m.max_drawdown_pct:>5.1f}% "
              f"vol={out['vol']:.3f} notional={out['notional']:.3f} "
              f"time_in_mkt={m.time_in_market_pct:.1f}%")

    # ---- Phase 1: exploratory (refit_days, horizon_days) x clause-A, train ----
    phase1 = phase1_explore()

    hr("PHASE 1 CLAUSE-A VERDICT PER (refit_days, horizon_days) CORNER")
    clears = {}
    for (rd, hd), info in phase1.items():
        fires = info["diag"]["clause_a_fires"]
        clears[(rd, hd)] = not fires
        print(f"  rd={rd:>3d} hd={hd}: clause A {'FIRES (inconclusive)' if fires else 'CLEARS'}")

    # Choose the primary (refit_days, horizon_days) for the steepness x cap
    # grid: prefer a corner where clause A clears; among those, prefer the
    # one with the healthiest diagnostics (highest median max|z|, i.e. the
    # least marginal signal) -- a light, pre-trading-number heuristic, not
    # itself the promotion test.
    clearing = [k for k, v in clears.items() if v]
    pool = clearing if clearing else list(phase1.keys())
    rd_star, hd_star = max(pool, key=lambda k: phase1[k]["diag"]["median_max_abs_z"])
    print(f"\n  => primary (refit_days, horizon_days) = ({rd_star}, {hd_star}) "
          f"[{'clause A clears' if clears[(rd_star, hd_star)] else 'clause A ALSO fires here -- see below'}]")

    # ---- Phase 2: steepness x cap sweep at the chosen corner, inner-validation ----
    hr(f"PHASE 2: steepness x cap sweep at (refit_days={rd_star}, horizon_days={hd_star}), "
       f"inner-validation, both markets ({len(STEEPNESS_GRID)}x{len(CAP_GRID)}={len(STEEPNESS_GRID)*len(CAP_GRID)} configs)")
    primary_configs = [dict(refit_days=rd_star, horizon_days=hd_star, steepness=s, cap=c)
                       for s in STEEPNESS_GRID for c in CAP_GRID]
    primary_rows = sweep_configs(primary_configs, v4_cache)

    # ---- Phase 3: robustness -- the other (refit_days, horizon_days) corners
    #      at the best steepness/cap found above ----
    fut_rows = primary_rows[primary_rows.market == "futures_5x"].copy()
    spot_rows = primary_rows[primary_rows.market == "spot"].copy()
    fut_rows["joint_d_sharpe"] = fut_rows["d_sharpe"] + spot_rows["d_sharpe"].to_numpy()
    best_idx = fut_rows["joint_d_sharpe"].idxmax()
    best_row = primary_rows.loc[primary_rows.index == best_idx].iloc[0]
    steep_star, cap_star = best_row["steepness"], best_row["cap"]
    print(f"\n  => best (steepness, cap) at this corner, by joint (spot+futures) d_sharpe on "
          f"inner-validation = ({steep_star}, {cap_star})")

    other_corners = [k for k in RH_GRID if k != (rd_star, hd_star)]
    hr(f"PHASE 3: robustness -- other (refit_days, horizon_days) corners at "
       f"steepness={steep_star}, cap={cap_star} ({len(other_corners)} configs)")
    robust_configs = [dict(refit_days=rd, horizon_days=hd, steepness=steep_star, cap=cap_star)
                      for rd, hd in other_corners]
    robust_rows = sweep_configs(robust_configs, v4_cache)

    all_rows = pd.concat([primary_rows, robust_rows], ignore_index=True)

    # ---- Best overall config selection (joint spot+futures d_sharpe, inner-val) ----
    hr("BEST OVERALL CONFIG (joint spot+futures d_sharpe, inner-validation)")
    fut_all = all_rows[all_rows.market == "futures_5x"].reset_index(drop=True)
    spot_all = all_rows[all_rows.market == "spot"].reset_index(drop=True)
    fut_all["joint_d_sharpe"] = fut_all["d_sharpe"] + spot_all["d_sharpe"]
    best_i = fut_all["joint_d_sharpe"].idxmax()
    best_cfg = dict(refit_days=int(fut_all.loc[best_i, "refit_days"]),
                    horizon_days=int(fut_all.loc[best_i, "horizon_days"]),
                    steepness=float(fut_all.loc[best_i, "steepness"]),
                    cap=float(fut_all.loc[best_i, "cap"]))
    print(f"  {best_cfg}")

    # Clause A verdict for the WINNING config's own (refit_days, horizon_days):
    ds_best = phase1[(best_cfg["refit_days"], best_cfg["horizon_days"])]["diag"]

    hr("FULL COMPARISON TABLE: best config vs v4-alone, inner-validation, both markets")
    final_rows = []
    boot_results = {}
    for market_name, market in (("spot", SPOT), ("futures_5x", FUTURES)):
        strat = R179NovelMetaSigmoid(**best_cfg)
        cand = full_measure(strat, VAL_START, VAL_END, market)
        v4 = v4_cache[market_name]
        cm, vm = cand["metrics"], v4["metrics"]
        cmp_ = compare_to_v4(cand["result"], v4["result"])
        boot = paired_bootstrap(cmp_["cand_daily"], cmp_["v4_daily"], total_log_return,
                                 mean_block=30.0, n_boot=2_000, seed=179)
        boot_results[market_name] = boot
        final_rows.append(dict(
            market=market_name,
            cand_final=cm.final_balance, v4_final=vm.final_balance,
            cand_sharpe=cm.sharpe, v4_sharpe=vm.sharpe, d_sharpe=cm.sharpe - vm.sharpe,
            cand_dd=cm.max_drawdown_pct, v4_dd=vm.max_drawdown_pct,
            cand_vol=cand["vol"], v4_vol=v4["vol"],
            cand_notional=cand["notional"], v4_notional=v4["notional"],
            cand_tim=cm.time_in_market_pct, v4_tim=vm.time_in_market_pct,
            d_log_growth=cmp_["d_log_growth"],
            boot_lo=boot.diff.lo, boot_hi=boot.diff.hi, boot_significant=boot.significant,
        ))
        print(f"\n  -- {market_name} --")
        print(f"     final balance:    cand=${cm.final_balance:>10,.0f}   v4=${vm.final_balance:>10,.0f}")
        print(f"     sharpe:           cand={cm.sharpe:>6.2f}          v4={vm.sharpe:>6.2f}   "
              f"d_sharpe={cm.sharpe - vm.sharpe:+.3f}")
        print(f"     max drawdown:     cand={cm.max_drawdown_pct:>6.1f}%         v4={vm.max_drawdown_pct:>6.1f}%")
        print(f"     realized vol:     cand={cand['vol']:.3f}          v4={v4['vol']:.3f}   "
              f"ratio={cand['vol']/v4['vol'] if v4['vol'] else float('nan'):.2f}")
        print(f"     avg notional:     cand={cand['notional']:.3f}          v4={v4['notional']:.3f}   "
              f"ratio={cand['notional']/v4['notional'] if v4['notional'] else float('nan'):.2f}")
        print(f"     time in market:   cand={cm.time_in_market_pct:.1f}%         v4={vm.time_in_market_pct:.1f}%")
        print(f"     d_log_growth (daily, full window) = {cmp_['d_log_growth']:+.4f}")
        print(f"     paired block-bootstrap 95% CI on d_log_growth (daily, 30d mean block, n_boot=2000): "
              f"[{boot.diff.lo:+.4f}, {boot.diff.hi:+.4f}]  significant={boot.significant}  "
              f"p(diff>0)={boot.p_positive:.3f}")

    final_df = pd.DataFrame(final_rows)

    # ---- Promotion-bar gate, applied mechanically ----
    hr("PRE-REGISTERED PROMOTION-BAR GATE (r179_direction.md Step 1 Q4 / Step 2), applied mechanically")
    print(f"  clause A (best config's own (refit_days={best_cfg['refit_days']}, "
          f"horizon_days={best_cfg['horizon_days']}) corner):")
    print_diag("winning corner", ds_best)
    clause_a_clears = not ds_best["clause_a_fires"]

    # R-33's own standing convention (see e.g. experiments/r102_shared.py,
    # r105/r146/r148/r149/r150_shared.py): risk-matched iff BOTH the
    # notional-exposure ratio and the realized-vol ratio fall in [0.9, 1.1].
    vol_ratios = final_df["cand_vol"] / final_df["v4_vol"]
    notional_ratios = final_df["cand_notional"] / final_df["v4_notional"]
    risk_matched = bool(((vol_ratios - 1.0).abs() <= 0.10).all()
                         and ((notional_ratios - 1.0).abs() <= 0.10).all())
    both_sig_positive = bool((final_df["boot_significant"] & (final_df["d_log_growth"] > 0)).all())
    both_beat_noise_floor = bool((final_df["d_sharpe"].abs() > 0.2).all() and (final_df["d_sharpe"] > 0).all())

    print(f"\n  clause A (classifier carries resolvable signal): {clause_a_clears}")
    print(f"  risk-matched (R-33 convention: notional AND realized-vol ratios both in [0.9,1.1] on BOTH "
          f"markets): {risk_matched}")
    print(f"    vol ratios (cand/v4):      {dict(zip(final_df['market'], vol_ratios.round(3)))}")
    print(f"    notional ratios (cand/v4): {dict(zip(final_df['market'], notional_ratios.round(3)))}")
    print(f"  paired-bootstrap-plausible improvement, same sign, BOTH markets: {both_sig_positive}")
    print(f"  |d_sharpe| exceeds the +/-0.2 noise floor AND is positive, BOTH markets: {both_beat_noise_floor}")

    promote = bool(clause_a_clears and risk_matched and both_sig_positive)

    hr("VERDICT")
    if not clause_a_clears:
        print("NEGATIVE / INCONCLUSIVE -- clause A fires: the walk-forward classifier does not clear its "
              "own pre-registered sample-size/signal-strength bar in the training period. Per "
              "r179_direction.md, this is inconclusive BY CONSTRUCTION regardless of the backtest numbers "
              "above, which are reported for completeness only.")
    elif not risk_matched:
        print("NEGATIVE (R-33) -- the candidate's realized volatility differs materially from v4-alone's on "
              "at least one market (see vol ratios above): this branch deliberately varies exposure, and "
              "the resulting comparison is not risk-matched, so any apparent Sharpe/log-growth improvement "
              "cannot be distinguished from simply running hotter or cooler than v4. Not scored as a win.")
    elif promote:
        print("PROMOTE-CANDIDATE (inner-validation only; holdout not read by this branch) -- clause A "
              "clears, the realized-risk profile is matched to v4-alone within +/-10% (notional and vol) "
              "on both markets, and the paired-bootstrap 95% CI on d_log_growth excludes zero on the "
              "winning side on BOTH BTC markets.")
    else:
        print("NEGATIVE -- clause A clears and risk is matched, but the pre-registered bar (paired-bootstrap "
              "plausible improvement on BOTH markets, risk-matched) is not met. See the comparison table "
              "above for where it falls short.")

    print(f"\nconfigurations evaluated: {N_EVALUATED}")
    print(f"  breakdown: phase 1 (refit_days x horizon_days corners, clause-A scan, train, SPOT) = "
          f"{len(RH_GRID)}; phase 2 (steepness x cap sweep at the chosen corner, inner-val, both markets) = "
          f"{len(primary_configs)} distinct configs (x2 markets each); phase 3 (robustness -- other "
          f"corners at best steepness/cap, inner-val, both markets) = {len(robust_configs)} distinct "
          f"configs (x2 markets each). Distinct parameter tuples = "
          f"{len(RH_GRID) + len(primary_configs) + len(robust_configs)} "
          f"(one tuple, the chosen corner at steepness=2.0/cap=2.0, is evaluated on both train AND "
          f"inner-validation -- counted once here since it is one configuration, run on two periods).")
    print(f"\ntotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
