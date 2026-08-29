#!/usr/bin/env python
"""R-179 CONSERVATIVE branch: literal binary meta-label veto on
`kelly_regime_v4`'s own `frac*scale` deadband decision.

Direction, citations, non-duplication argument and the frozen falsification
clauses all live in `experiments/r179_direction.md` (read there first); the
shared, read-only engine (`vote_frac`, `conditional_scale`,
`walk_forward_meta_prob`) lives in `experiments/r179_shared.py` and is never
edited or re-derived here -- this file only wires those primitives into one
`Strategy` and reports the sweep.

THE MECHANISM, exactly (Lopez de Prado 2018, ch. 3's textbook binary
meta-label, "bet or don't"):

v4's own sequential deadband loop is reproduced bar-by-bar, verbatim:

    desired = frac[i] * scale[i]
    if abs(desired - pos) > deadband:
        pos = desired          # <- v4 always takes this branch here

The ONLY change: the assignment `pos = desired` is taken only when the
walk-forward meta-classifier's CURRENT probability clears `threshold`
(default 0.50). If it does not, `pos` holds at its previous value and the
same candidate `desired` is re-tested on every later bar (a vetoed trade is
DEFERRED, never discarded -- it fires the moment either the gate opens or
`desired` itself changes again and re-clears the deadband). While the
classifier has not yet reached `min_samples` resolved labels at its first
refit, `prob[i]` is NaN by `walk_forward_meta_prob`'s own contract; this
branch's disclosed neutral behaviour for that warmup stretch is to treat a
NaN probability as an OPEN gate -- i.e. behave exactly like v4 until the
classifier actually has an opinion, per r179_direction.md's requirement that
each branch state its own warmup convention.

No lookahead: `frac`/`scale`/`vol_ratio` are `r179_shared`'s own verbatim,
already-causal reproductions of v4's factors; `vol_daily` is the identical
causal EWM estimator (`shift(1)`-ed) `conditional_scale` uses internally;
`walk_forward_meta_prob` forward-fills each bar's probability from the most
recent refit whose fit set is causal as of that refit's own instant. The
deadband loop above reads only `frac[i]`, `scale[i]`, `prob[i]` at the
CURRENT bar `i` -- nothing here reads `close` or any other array at an index
other than the bar being processed.

Run: `source .venv/bin/activate && python experiments/r179_conservative.py`
(from the repo root). Runtime note: `walk_forward_meta_prob` (frac/scale/
vol/prob/diag) is cached per (horizon_days, refit_days, k, embargo_days,
min_samples, data-slice fingerprint) inside `prepare()`, since it does not
depend on `threshold` or on which market (spot/futures) is being backtested
-- so the 16-config x 2-market sweep below only actually runs the expensive
walk-forward classifier 4 times (one per distinct (horizon_days, refit_days)
pair), not 32.
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

from tradebot.registry import register  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r179_shared import (  # noqa: E402
    conditional_scale,
    vote_frac,
    walk_forward_meta_prob,
)

OOS_START = "2023-01-01"  # never read below this
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"


# ====================================================================== (1)
# The strategy itself.
# ======================================================================

_CACHE: dict[tuple, tuple] = {}


def _cache_key(close: pd.Series, cfg: tuple) -> tuple:
    c = close.to_numpy()
    checksum = round(float(np.sum(c)), 2)
    return (len(c), close.index[0].value, close.index[-1].value,
            round(float(c[0]), 8), round(float(c[-1]), 8), checksum, cfg)


def _causal_vol_daily(close: pd.Series, vol_span: int) -> np.ndarray:
    """v4's own causal EWM annualized realized-vol array, verbatim (see
    `conditional_scale`'s identical internal computation in r179_shared.py --
    exposed here as a standalone array since `walk_forward_meta_prob` needs
    it directly as its `vol_daily` argument)."""
    r = np.log(close).diff()
    return (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


@register
class R179ConservativeMetaVeto(Strategy):
    """`kelly_regime_v4`'s vote+scale signal, gated by a binary walk-forward
    meta-label veto (Lopez de Prado 2018 ch. 3) on every deadband-crossing
    rebalance. See module docstring for the exact mechanism; full design in
    `experiments/r179_direction.md`."""

    name = "r179_conservative_meta_veto"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4's own

    def __init__(self, threshold: float = 0.50, k: float = 1.0,
                 horizon_days: int = 3, refit_days: int = 90,
                 embargo_days: int = 3, min_samples: int = 50,
                 deadband: float = 0.10, vol_span: int = 8 * BARS_PER_DAY,
                 use_cache: bool = True) -> None:
        self.threshold = threshold
        self.k = k
        self.horizon_days = horizon_days
        self.refit_days = refit_days
        self.embargo_days = embargo_days
        self.min_samples = min_samples
        self.deadband = deadband
        self.vol_span = vol_span
        self.use_cache = use_cache
        # populated by prepare(), read back for reporting (diag/prob/frac/scale)
        self.last_diag: dict | None = None
        self.last_prob: np.ndarray | None = None
        self.last_frac: np.ndarray | None = None
        self.last_scale: np.ndarray | None = None
        self.last_gate_open_frac: float | None = None

    def _heavy(self, close: pd.Series, index: pd.DatetimeIndex):
        """(frac, scale, vol_ratio, vol_daily, prob, diag) -- everything
        independent of `threshold` and of which market is being backtested."""
        cfg = (self.k, self.horizon_days, self.refit_days, self.embargo_days,
               self.min_samples, self.vol_span)
        key = _cache_key(close, cfg) if self.use_cache else None
        if key is not None and key in _CACHE:
            return _CACHE[key]

        frac = vote_frac(close)
        scale, vol_ratio = conditional_scale(close, vol_span=self.vol_span)
        vol_daily = _causal_vol_daily(close, self.vol_span)
        prob, diag = walk_forward_meta_prob(
            index, close.to_numpy(), vol_daily, frac, vol_ratio,
            k=self.k, horizon_days=self.horizon_days, refit_days=self.refit_days,
            embargo_days=self.embargo_days, min_samples=self.min_samples,
        )
        result = (frac, scale, vol_ratio, vol_daily, prob, diag)
        if key is not None:
            _CACHE[key] = result
        return result

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        frac, scale, vol_ratio, vol_daily, prob, diag = self._heavy(close, df.index)
        self.last_diag = diag
        self.last_prob = prob
        self.last_frac = frac
        self.last_scale = scale

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        gate_open_count = 0
        crossings = 0
        for i in range(n):
            desired = frac[i] * scale[i]
            if abs(desired - pos) > self.deadband:
                crossings += 1
                p = prob[i]
                # Warmup contract: NaN probability (classifier not yet fit)
                # defaults the gate OPEN -- act exactly like v4 until the
                # classifier has an opinion (r179_direction.md's required
                # disclosed neutral behaviour).
                gate_open = (not np.isfinite(p)) or (p >= self.threshold)
                if gate_open:
                    gate_open_count += 1
                    pos = desired
            target[i] = pos
        self.last_gate_open_frac = (gate_open_count / crossings) if crossings else float("nan")

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ====================================================================== (2)
# Self-test: causal-truncation probe (this branch's own no-lookahead check,
# on top of what `walk_forward_meta_prob` already guarantees).
# ======================================================================

def _causal_truncation_probe(cfg: dict, df: pd.DataFrame, cut: int) -> bool:
    """Truncate `df` at `cut` bars, rerun `prepare()`, and check that every
    bar strictly before `cut - horizon_days*BARS_PER_DAY - embargo_days*
    BARS_PER_DAY` (i.e. every bar whose label/refit inputs cannot possibly
    reach past the truncation point) produced an IDENTICAL `target` value in
    both the full and truncated runs. A generous margin (well past the
    horizon+embargo boundary) is used so this is a real causality check, not
    a numerically-fragile one."""
    strat_full = R179ConservativeMetaVeto(**cfg, use_cache=False)
    strat_cut = R179ConservativeMetaVeto(**cfg, use_cache=False)
    full = strat_full.prepare(df.copy())
    trunc = strat_cut.prepare(df.iloc[:cut].copy())
    margin = (cfg.get("horizon_days", 3) + cfg.get("embargo_days", 3) + 5) * BARS_PER_DAY
    safe = cut - margin
    if safe <= 0:
        return True
    a = full["target"].to_numpy()[:safe]
    b = trunc["target"].to_numpy()[:safe]
    return bool(np.allclose(a, b, atol=1e-9, equal_nan=True))


def _self_test() -> None:
    from tradebot.data import load_dataset

    df, _label = load_dataset(ROOT / "data", "spot")
    train = df.loc[:INNER_TRAIN_END]
    assert train.index.max() < pd.Timestamp(OOS_START, tz="UTC")
    cfg = dict(threshold=0.5, k=1.0, horizon_days=3, refit_days=90,
               embargo_days=3, min_samples=50)
    cut = len(train) // 2
    ok = _causal_truncation_probe(cfg, train, cut)
    assert ok, "causal truncation probe FAILED -- prepare() is peeking ahead"


_self_test()


# ====================================================================== (3)
# Reporting helpers.
# ======================================================================

def _extended_metrics(strategy: Strategy, df: pd.DataFrame, market, start, end,
                       balance: float = 1_000.0) -> dict:
    """Run one strategy/market/period and return the standard `Metrics` PLUS
    realized volatility (annualized std of per-bar equity returns), average
    notional (mean |target| over the measured period -- the fraction of
    equity `order_notional` targets, identical risk units on spot/futures),
    so R-33's risk-matching rule can actually be checked, not asserted."""
    from tradebot.metrics import compute_metrics
    from tradebot.window import run_period

    result = run_period(strategy, df, start, end, market=market, start_balance=balance)
    m = compute_metrics(result)
    eq = result.equity.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(eq[:-1] > 0, np.diff(eq) / eq[:-1], 0.0)
    realized_vol_pct = float(np.std(rets, ddof=1) * np.sqrt(BARS_PER_YEAR) * 100.0) if len(rets) > 1 else 0.0
    target = result.df["target"].to_numpy() if "target" in result.df.columns else np.zeros(len(result.df))
    avg_notional = float(np.mean(np.abs(target))) if len(target) else 0.0
    return dict(
        final_balance=m.final_balance, profit_pct=m.profit_pct, sharpe=m.sharpe,
        max_dd=m.max_drawdown_pct, time_in_market=m.time_in_market_pct,
        num_trades=m.num_trades, realized_vol_pct=realized_vol_pct,
        avg_notional=avg_notional, liquidated=m.liquidated,
    )


def _log_growth(final_balance: float, start_balance: float = 1_000.0) -> float:
    return float(np.log(final_balance / start_balance))


def _risk_matched(a: dict, b: dict, vol_tol: float = 0.15, notional_tol: float = 0.15) -> bool:
    """R-33's standing rule: a comparison is only evidence if both arms carry
    matched realized risk. `vol_tol`/`notional_tol` are RELATIVE tolerances
    on realized vol and average notional."""
    if a["realized_vol_pct"] <= 0 or b["realized_vol_pct"] <= 0:
        return False
    vol_ratio = a["realized_vol_pct"] / b["realized_vol_pct"]
    not_ratio = (a["avg_notional"] / b["avg_notional"]) if b["avg_notional"] > 0 else float("nan")
    return bool(abs(vol_ratio - 1.0) <= vol_tol and np.isfinite(not_ratio)
                and abs(not_ratio - 1.0) <= notional_tol)


# ====================================================================== (4)
# Main sweep.
# ======================================================================

def main() -> None:
    from scripts.experiment import DF, FUTURES, SPOT
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    assert DF.index.max() >= pd.Timestamp(OOS_START, tz="UTC"), "sanity: dataset should extend past OOS_START"
    # This script itself never reads DF at/after OOS_START -- every ev()/
    # extended-metrics call below is bounded by INNER_TRAIN_END/INNER_VAL_END.

    print("=" * 78)
    print("R-179 CONSERVATIVE -- binary meta-label veto on kelly_regime_v4")
    print("=" * 78)

    THRESHOLDS = (0.45, 0.50, 0.55, 0.60)
    REFIT_DAYS = (30, 90)
    HORIZON_DAYS = (1, 3)
    K = 1.0
    EMBARGO_DAYS = 3
    MIN_SAMPLES = 50

    configs = [dict(threshold=th, k=K, horizon_days=hd, refit_days=rd,
                     embargo_days=EMBARGO_DAYS, min_samples=MIN_SAMPLES)
               for th in THRESHOLDS for rd in REFIT_DAYS for hd in HORIZON_DAYS]
    print(f"\nGrid: {len(THRESHOLDS)} thresholds x {len(REFIT_DAYS)} refit_days x "
          f"{len(HORIZON_DAYS)} horizon_days = {len(configs)} configs "
          f"(k={K} fixed, embargo_days={EMBARGO_DAYS} fixed). Running all "
          f"{len(configs)} -- the heavy walk-forward computation is cached "
          f"per (horizon_days, refit_days) pair, so wall time stays low.")

    # ---------------------------------------------------------- Clause A
    print("\n" + "-" * 78)
    print("CLAUSE A (falsification, r179_direction.md Step 1 Q4) -- diag from "
          f"the TRAINING period (dataset start .. {INNER_TRAIN_END}) only, "
          "per (horizon_days, refit_days) pair (independent of threshold).")
    print("-" * 78)
    train_df = DF.loc[:INNER_TRAIN_END]
    print(f"Training slice: {len(train_df):,} bars, {train_df.index[0]} -> {train_df.index[-1]}")

    clause_a_by_pair: dict[tuple, dict] = {}
    for hd in HORIZON_DAYS:
        for rd in REFIT_DAYS:
            probe = R179ConservativeMetaVeto(threshold=0.5, k=K, horizon_days=hd,
                                              refit_days=rd, embargo_days=EMBARGO_DAYS,
                                              min_samples=MIN_SAMPLES, use_cache=True)
            probe.prepare(train_df.copy())
            diag = probe.last_diag
            n = len(train_df)
            total_checkpoints = len(range(0, n, rd * BARS_PER_DAY))
            bad = diag["clause_a_bad_refits"]
            frac_bad = bad / total_checkpoints if total_checkpoints else float("nan")
            majority_bad = frac_bad > 0.5
            avg_n = float(np.mean(diag["n_at_refit"])) if diag["n_at_refit"] else float("nan")
            avg_z = float(np.mean(diag["max_abs_z"])) if diag["max_abs_z"] else float("nan")
            min_n = min(diag["n_at_refit"]) if diag["n_at_refit"] else float("nan")
            clause_a_by_pair[(hd, rd)] = dict(
                total_checkpoints=total_checkpoints, actual_refits=diag["refits"],
                bad=bad, frac_bad=frac_bad, majority_bad=majority_bad,
                avg_n_at_refit=avg_n, min_n_at_refit=min_n, avg_max_abs_z=avg_z,
            )
            print(f"\n  horizon_days={hd} refit_days={rd}: "
                  f"total_checkpoints={total_checkpoints} actual_fits={diag['refits']} "
                  f"bad(low-n-or-low-z)={bad} ({frac_bad:.0%})")
            print(f"      avg n_at_refit={avg_n:.0f} (min={min_n}), avg max|z|={avg_z:.2f}")
            print(f"      CLAUSE A {'FIRES (inconclusive by construction)' if majority_bad else 'clear'} "
                  f"for this (horizon_days, refit_days) pair")

    any_pair_clear = any(not v["majority_bad"] for v in clause_a_by_pair.values())
    print(f"\nCLAUSE A overall: {'at least one (horizon_days, refit_days) pair is CLEAR'if any_pair_clear else 'EVERY pair FIRES -- mechanism inconclusive by construction, full stop'}")

    # ---------------------------------------------------------- sweep on inner-validation
    print("\n" + "-" * 78)
    print(f"INNER-VALIDATION sweep ({INNER_VAL_START} -> {INNER_VAL_END}), both markets, "
          "all configs")
    print("-" * 78)

    rows = []
    t0 = time.time()
    for cfg in configs:
        strat_spot = R179ConservativeMetaVeto(**cfg)
        strat_fut = R179ConservativeMetaVeto(**cfg)
        m_spot = _extended_metrics(strat_spot, DF, SPOT, INNER_VAL_START, INNER_VAL_END)
        m_fut = _extended_metrics(strat_fut, DF, FUTURES, INNER_VAL_START, INNER_VAL_END)
        gate_open_frac = strat_spot.last_gate_open_frac
        rows.append(dict(cfg=cfg, spot=m_spot, futures=m_fut, gate_open_frac=gate_open_frac))
        print(f"  th={cfg['threshold']:.2f} hd={cfg['horizon_days']} rd={cfg['refit_days']:>2d}  "
              f"gate_open={gate_open_frac:.0%}  "
              f"SPOT final=${m_spot['final_balance']:>9,.0f} sharpe={m_spot['sharpe']:>5.2f} "
              f"DD={m_spot['max_dd']:>5.1f}% vol={m_spot['realized_vol_pct']:>5.1f}% "
              f"notional={m_spot['avg_notional']:.3f} tim={m_spot['time_in_market']:>5.1f}% | "
              f"FUT final=${m_fut['final_balance']:>9,.0f} sharpe={m_fut['sharpe']:>5.2f} "
              f"DD={m_fut['max_dd']:>5.1f}% vol={m_fut['realized_vol_pct']:>5.1f}% "
              f"notional={m_fut['avg_notional']:.3f} tim={m_fut['time_in_market']:>5.1f}%")
    print(f"\n({len(configs)} configs x 2 markets = {len(configs) * 2} backtests, "
          f"{time.time() - t0:.0f}s total)")

    # ---------------------------------------------------------- v4-alone baseline
    print("\n" + "-" * 78)
    print("v4-ALONE baseline, same slice/markets")
    print("-" * 78)
    v4_spot = _extended_metrics(KellyRegimeV4(), DF, SPOT, INNER_VAL_START, INNER_VAL_END)
    v4_fut = _extended_metrics(KellyRegimeV4(), DF, FUTURES, INNER_VAL_START, INNER_VAL_END)
    print(f"  SPOT    final=${v4_spot['final_balance']:>9,.0f} sharpe={v4_spot['sharpe']:>5.2f} "
          f"DD={v4_spot['max_dd']:>5.1f}% vol={v4_spot['realized_vol_pct']:>5.1f}% "
          f"time_in_mkt={v4_spot['time_in_market']:>5.1f}% avg_notional={v4_spot['avg_notional']:.3f} "
          f"trades={v4_spot['num_trades']}")
    print(f"  FUTURES final=${v4_fut['final_balance']:>9,.0f} sharpe={v4_fut['sharpe']:>5.2f} "
          f"DD={v4_fut['max_dd']:>5.1f}% vol={v4_fut['realized_vol_pct']:>5.1f}% "
          f"time_in_mkt={v4_fut['time_in_market']:>5.1f}% avg_notional={v4_fut['avg_notional']:.3f} "
          f"trades={v4_fut['num_trades']}")

    # ---------------------------------------------------------- selection + promotion-bar check
    print("\n" + "-" * 78)
    print("SELECTION -- rank configs by min(spot,futures) Delta-log-growth vs v4-alone")
    print("-" * 78)
    SHARPE_NOISE_FLOOR = 0.2
    for row in rows:
        row["d_loggrowth_spot"] = _log_growth(row["spot"]["final_balance"]) - _log_growth(v4_spot["final_balance"])
        row["d_loggrowth_fut"] = _log_growth(row["futures"]["final_balance"]) - _log_growth(v4_fut["final_balance"])
        row["d_sharpe_spot"] = row["spot"]["sharpe"] - v4_spot["sharpe"]
        row["d_sharpe_fut"] = row["futures"]["sharpe"] - v4_fut["sharpe"]
        row["risk_matched_spot"] = _risk_matched(row["spot"], v4_spot)
        row["risk_matched_fut"] = _risk_matched(row["futures"], v4_fut)
        row["min_d_loggrowth"] = min(row["d_loggrowth_spot"], row["d_loggrowth_fut"])

    rows.sort(key=lambda r: r["min_d_loggrowth"], reverse=True)
    print("\nAll configs, risk-matching status (R-33: must be true on BOTH markets "
          "for a comparison to count as evidence at all):")
    for r in rows:
        print(f"  th={r['cfg']['threshold']:.2f} hd={r['cfg']['horizon_days']} rd={r['cfg']['refit_days']:>2d}  "
              f"min(dlog)={r['min_d_loggrowth']:+.4f}  dSharpe(spot/fut)={r['d_sharpe_spot']:+.3f}/{r['d_sharpe_fut']:+.3f}  "
              f"risk_matched(spot/fut)={r['risk_matched_spot']}/{r['risk_matched_fut']}  "
              f"notional(spot/fut)={r['spot']['avg_notional']:.3f}/{r['futures']['avg_notional']:.3f} "
              f"(v4: {v4_spot['avg_notional']:.3f}/{v4_fut['avg_notional']:.3f})")

    risk_matched_rows = [r for r in rows if r["risk_matched_spot"] and r["risk_matched_fut"]]
    print(f"\n{len(risk_matched_rows)} of {len(rows)} configs are risk-matched on BOTH markets "
          f"(realized vol and avg notional within 15% of v4-alone's) -- ONLY these are "
          "valid comparisons per R-33; the rest are de-risking effects, not evidence "
          "the meta-label mechanism itself adds value.")
    best = risk_matched_rows[0] if risk_matched_rows else rows[0]
    print(f"\nBest {'RISK-MATCHED ' if risk_matched_rows else '(NOT risk-matched, reported for completeness only) '}"
          f"config by worst-of-two-markets Delta-log-growth: {best['cfg']}")
    print(f"  spot:    Delta-log-growth={best['d_loggrowth_spot']:+.4f}  Delta-Sharpe={best['d_sharpe_spot']:+.3f}  "
          f"risk_matched={best['risk_matched_spot']}  (vol {best['spot']['realized_vol_pct']:.1f}% vs v4 {v4_spot['realized_vol_pct']:.1f}%, "
          f"notional {best['spot']['avg_notional']:.3f} vs v4 {v4_spot['avg_notional']:.3f})")
    print(f"  futures: Delta-log-growth={best['d_loggrowth_fut']:+.4f}  Delta-Sharpe={best['d_sharpe_fut']:+.3f}  "
          f"risk_matched={best['risk_matched_fut']}  (vol {best['futures']['realized_vol_pct']:.1f}% vs v4 {v4_fut['realized_vol_pct']:.1f}%, "
          f"notional {best['futures']['avg_notional']:.3f} vs v4 {v4_fut['avg_notional']:.3f})")

    clears_spot = best["d_sharpe_spot"] > SHARPE_NOISE_FLOOR and best["d_loggrowth_spot"] > 0
    clears_fut = best["d_sharpe_fut"] > SHARPE_NOISE_FLOOR and best["d_loggrowth_fut"] > 0
    both_risk_matched = best["risk_matched_spot"] and best["risk_matched_fut"]
    promote = clears_spot and clears_fut and both_risk_matched

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"Configs evaluated: {len(configs)} (all of the pre-registered 16), "
          f"each on both SPOT and FUTURES_5x on inner-validation.")
    print(f"Clause A: {'at least one (horizon_days,refit_days) pair CLEAR' if any_pair_clear else 'EVERY pair FIRES'} "
          "(see table above for per-pair n_at_refit / max|z| numbers).")
    print(f"Best config's Delta-Sharpe vs v4-alone: spot={best['d_sharpe_spot']:+.3f} "
          f"futures={best['d_sharpe_fut']:+.3f}  (noise floor +/-{SHARPE_NOISE_FLOOR})")
    print(f"Risk-matched (R-33): spot={best['risk_matched_spot']} futures={best['risk_matched_fut']}")

    if not any_pair_clear:
        print("\nVERDICT: INCONCLUSIVE (clause A fires for every grid pair) -- "
              "per r179_direction.md, no trading-level verdict is licensed "
              "regardless of the backtest table above.")
    elif not promote:
        print("\nVERDICT: NEGATIVE -- best config does not clear the pre-registered "
              "promotion bar (Delta-Sharpe > +0.2 noise floor AND positive "
              "Delta-log-growth, risk-matched, on BOTH markets) on inner-validation.")
    else:
        print("\nVERDICT: PROMOTE-CANDIDATE -- best config clears the pre-registered "
              "promotion bar on BOTH markets, risk-matched. Not yet holdout-tested "
              "(per the routine, holdout is only consulted after this gate clears "
              "and the operator freezes the specific config).")

    print(f"\nMax timestamp read by this script's sweep/selection logic: "
          f"{INNER_VAL_END} (inner-validation upper bound). No bar at/after "
          f"{OOS_START} was read for any metric above.")


if __name__ == "__main__":
    main()
